import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from agri_data import adaptive_selector as selector


def forecast(origin, horizon, predicted):
    target=origin+dt.timedelta(days=horizon)
    return {"origin":origin.isoformat(),"target":target.isoformat(),"horizon":horizon,"predicted":predicted}


def maps_from_rows(rows):
    return {(dt.date.fromisoformat(r["origin"]),dt.date.fromisoformat(r["target"]),r["horizon"]):r["predicted"] for r in rows}


def make_source():
    products=[]
    start=dt.date(2023,12,1)
    for pid in selector.PRODUCTS:
        actuals={}; baseline=[];ridge=[];hist=[]
        for i in range(61):
            target=start+dt.timedelta(days=i)
            actual=100.0+(i%5)
            origin=target-dt.timedelta(days=1)
            actuals[target]=actual
            baseline.append(forecast(origin,1,actual+1+(i%3)*.1))
            ridge.append(forecast(origin,1,actual+.2+(i%2)*.1))
            hist.append(forecast(origin,1,actual+.5+(i%2)*.1))
        products.append({"id":pid,"observations":[{"date":d.isoformat(),"actual":v} for d,v in actuals.items()],
            "predictions":baseline,"runs":[{"modelId":"ridge","cadence":"daily","predictions":ridge},
                {"modelId":"hist_gb","cadence":"daily","predictions":hist}]})
    return {"schemaVersion":2,"products":products}


class AdaptiveSelectorTests(unittest.TestCase):
    def test_forecast_target_index_is_built_once_by_horizon(self):
        origin=dt.date(2024,1,1); rows={m:{} for m in selector.MODELS}
        rows["baseline"][(origin,origin+dt.timedelta(days=2),2)]=12.0
        rows["ridge"][(origin,origin+dt.timedelta(days=2),2)]=11.0
        rows["hist_gb"][(origin,origin+dt.timedelta(days=2),2)]=10.0
        indexed=selector._index_forecasts(rows)
        self.assertEqual(indexed["baseline"][2][origin+dt.timedelta(days=2)],12.0)
        self.assertNotIn(1,indexed["baseline"])

    def test_temporal_cutoff_and_730_day_window(self):
        origin=dt.date(2024,1,1); horizon=1
        cutoff=origin-dt.timedelta(days=2)
        dates=[cutoff,cutoff-dt.timedelta(days=729),cutoff-dt.timedelta(days=730),cutoff+dt.timedelta(days=1)]
        actuals={d:100.0 for d in dates}
        forecasts={m:{} for m in selector.MODELS}
        for d in dates:
            o=d-dt.timedelta(days=horizon)
            forecasts["baseline"][(o,d,horizon)]=110
            forecasts["ridge"][(o,d,horizon)]=101
            forecasts["hist_gb"][(o,d,horizon)]=102
        r=selector._select_one("cabbage",horizon,origin,actuals,selector._index_forecasts(forecasts))
        self.assertEqual(r["historyCutoff"],cutoff.isoformat())
        self.assertEqual(r["historyWindowStart"],(cutoff-dt.timedelta(days=729)).isoformat())
        self.assertEqual(r["commonThreeModelDates"],2)
        self.assertEqual(r["selectedModel"],"baseline")

    def test_sparse_pepper_falls_back_to_baseline(self):
        origin=dt.date(2024,1,1); actuals={}; forecasts={m:{} for m in selector.MODELS}
        for i in range(29):
            d=dt.date(2023,12,30)-dt.timedelta(days=i);actuals[d]=10
            o=d-dt.timedelta(days=1)
            for m,p in (("baseline",12),("ridge",10),("hist_gb",11)):
                forecasts[m][(o,d,1)]=p
        r=selector._select_one("pepper",1,origin,actuals,selector._index_forecasts(forecasts))
        self.assertEqual(r["selectedModel"],"baseline")
        self.assertEqual(r["selectionReason"],"insufficient_common_history")

    def test_zero_baseline_ape_fallback_and_exact_two_percent_qualifies(self):
        origin=dt.date(2024,1,1)
        cases=((100,100,100,"baseline","baseline_ape_zero_no_relative_improvement"),
               (110,109.8,111,"ridge","selected_qualified_candidate"))
        for baseline_pred,ridge_pred,hist_pred,expected_model,expected_reason in cases:
            actuals={};forecasts={m:{} for m in selector.MODELS}
            for i in range(30):
                d=dt.date(2023,12,30)-dt.timedelta(days=i);actuals[d]=100
                o=d-dt.timedelta(days=1)
                for m,pred in (("baseline",baseline_pred),("ridge",ridge_pred),("hist_gb",hist_pred)):
                    forecasts[m][(o,d,1)]=pred
            result=selector._select_one("cabbage",1,origin,actuals,selector._index_forecasts(forecasts))
            self.assertEqual(result["selectedModel"],expected_model)
            self.assertEqual(result["selectionReason"],expected_reason)

    def test_candidate_gate_and_deterministic_tie(self):
        origin=dt.date(2024,1,1); actuals={}; forecasts={m:{} for m in selector.MODELS}
        for i in range(30):
            d=dt.date(2023,12,30)-dt.timedelta(days=i);actuals[d]=100
            o=d-dt.timedelta(days=1)
            for m,p in (("baseline",110),("ridge",100),("hist_gb",100)):
                forecasts[m][(o,d,1)]=p
        r=selector._select_one("cabbage",1,origin,actuals,selector._index_forecasts(forecasts))
        self.assertEqual(r["selectedModel"],"ridge")
        self.assertEqual(r["commonThreeModelDates"],30)

    def test_common_dates_require_every_prediction_and_actual(self):
        origin=dt.date(2024,1,1); actuals={}; forecasts={m:{} for m in selector.MODELS}
        for i in range(31):
            d=dt.date(2023,12,30)-dt.timedelta(days=i);actuals[d]=100
            o=d-dt.timedelta(days=1)
            for m,p in (("baseline",110),("ridge",100),("hist_gb",101)):
                forecasts[m][(o,d,1)]=p
        first=next(iter(forecasts["baseline"]))
        missing=next(key for key in forecasts["hist_gb"] if key!=first)
        forecasts["hist_gb"].pop(missing)
        no_actual=first[1]
        actuals.pop(no_actual)
        r=selector._select_one("cabbage",1,origin,actuals,selector._index_forecasts(forecasts))
        self.assertEqual(r["commonThreeModelDates"],29)
        self.assertEqual(r["missingAnyForecastDates"],1)

    def test_full_analysis_reports_dm_holm100_and_metrics(self):
        r=selector.analyze(make_source(),"synthetic-sha")
        self.assertEqual(r["inputSha256"],"synthetic-sha")
        self.assertEqual(len(r["evaluation"]),len(selector.PRODUCTS)*len(selector.HORIZONS))
        self.assertEqual(r["multipleTesting"]["familySize"],100)
        self.assertEqual(len(r["multipleTesting"]["tests"]),100)
        cabbage=r["evaluation"][0]
        self.assertIn("mae",cabbage["metrics"]["selected"])
        self.assertEqual(cabbage["coverage"]["baselineRowsWithActual"],30)
        self.assertEqual(cabbage["pairedTests"]["absolute_error"]["primaryHAC9"]["n"],30)
        self.assertEqual(r["selectionSummary"]["cabbage|1"]["historyNominee"].get("ridge"),29)

    def test_missing_current_selected_forecast_falls_back_to_baseline(self):
        source=make_source()
        cabbage=next(p for p in source["products"] if p["id"]=="cabbage")
        ridge=next(r for r in cabbage["runs"] if r["modelId"]=="ridge")
        ridge["predictions"]=[r for r in ridge["predictions"] if r["target"]!="2024-01-05"]
        report=selector.analyze(source)
        coverage=next(r["coverage"] for r in report["evaluation"] if r["productId"]=="cabbage" and r["horizon"]==1)
        self.assertEqual(coverage["selectedModelMissingFallbacks"],1)
        self.assertEqual(coverage["fallbackByModel"],{"ridge":1})
        self.assertEqual(coverage["baselineRowsWithActual"],30)

    def test_constant_loss_difference_is_unavailable_not_significant(self):
        rows=[]
        for i in range(40):
            target=(dt.date(2024,1,1)+dt.timedelta(days=i)).isoformat()
            rows.append({"target":target,"horizon":1,"actual":123.0,"selectedPred":123.2,"baselinePred":123.7,
                         "selected":{"mae":.2},"baseline":{"mae":.7}})
        result=selector._dm(rows,"mae",9)
        self.assertEqual(result["status"],"unavailable_nonpositive_variance")
        self.assertIsNone(result["pValue"])

    def test_json_and_markdown_are_written_atomically(self):
        report=selector.analyze(make_source(),"synthetic-sha")
        with tempfile.TemporaryDirectory() as directory:
            jp=Path(directory)/"report.json";mp=Path(directory)/"report.md"
            selector.write_report(report,jp,mp)
            loaded=json.loads(jp.read_text(encoding="utf-8"))
            self.assertEqual(loaded["inputSha256"],"synthetic-sha")
            self.assertIn("Holm100",mp.read_text(encoding="utf-8"))
            self.assertEqual({x.name for x in Path(directory).iterdir()},{"report.json","report.md"})

    def test_inference_minimum_pairs_and_holm_missing_p_conservative(self):
        # A 29-row horizon must remain in each fixed family with p=1.
        source=make_source()
        for p in source["products"]:
            p["predictions"]=[x for x in p["predictions"] if x["target"]<="2024-01-29"]
            for run in p["runs"]:run["predictions"]=[x for x in run["predictions"] if x["target"]<="2024-01-29"]
            p["observations"]=[x for x in p["observations"] if x["date"]<="2024-01-29"]
        r=selector.analyze(source)
        test=next(x for x in r["multipleTesting"]["tests"] if x["productId"]=="cabbage" and x["horizon"]==1 and x["loss"]=="absolute_error")
        self.assertEqual(test["primaryHAC9"]["status"],"unavailable_insufficient_pairs")
        self.assertEqual(test["primaryHAC9"]["holmAdjustedPValue"],1)


if __name__=="__main__":unittest.main()
