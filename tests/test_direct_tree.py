import datetime as dt
import math
import unittest
from unittest import mock

from agri_data import dashboard, direct_tree


class Estimator:
    def fit(self, x, y): return self
    def predict(self, x): return [0.0] * len(x)


class FailAfterOnePrediction(Estimator):
    def __init__(self): self.calls=0
    def predict(self, x):
        self.calls+=1
        if self.calls==2: raise RuntimeError("synthetic batch failure")
        return [0.0] * len(x)


def fixture_data():
    start=dt.date(2015,1,1); products=[]; forecast_products=[]; ridge_rows=[]
    for pid in dashboard.PRODUCTS:
        obs=[]; pred=[]; ridge=[]; pooled=[]
        for i in range(130):
            day=start+dt.timedelta(days=i); price=100+i*.2
            obs.append({"date":day.isoformat(),"actual":price,"volumeKg":10})
            if i>=90:
                origin=day-dt.timedelta(days=1)
                for h in (1,):
                    # candidate eval horizon is patched to one; common sources are complete.
                    row={"target":day.isoformat(),"origin":(day-dt.timedelta(days=h)).isoformat(),"horizon":h,"predicted":price}
                    pred.append(row); pooled.append(row)
                    if i>=100: ridge.append({"product":pid,"horizon":h,"target":day.isoformat(),"actual":price,"direct":price})
        products.append({"id":pid,"observations":obs})
        forecast_products.append({"id":pid,"predictions":pred,"runs":[{"modelId":"hist_gb","cadence":"monthly","predictions":pooled}]})
        ridge_rows.extend(ridge)
    prices={"schemaVersion":1,"products":products}
    forecasts={"schemaVersion":2,"products":forecast_products}
    cov={f"{p}/1":{"candidateAvailableDates":[r['target'] for r in ridge_rows if r['product']==p]} for p in dashboard.PRODUCTS}
    ridge_report={"protocol":{"id":"direct-horizon-ridge-2026-10-06"},"pairedRows":ridge_rows,"coverage":cov}
    return prices,forecasts,ridge_report


class DirectTreeTests(unittest.TestCase):
    def test_candidate_is_strictly_before_fit_origin_and_has_fit_provenance(self):
        prices,forecasts,ridge=fixture_data()
        with mock.patch.object(direct_tree,"HORIZONS",(1,)), \
             mock.patch.object(direct_tree,"FAMILY_SIZE",30), \
             mock.patch.object(direct_tree,"EVAL_START",dt.date(2015,4,15)), \
             mock.patch.object(direct_tree,"EVAL_END",dt.date(2015,4,25)), \
             mock.patch.object(direct_tree,"_model",return_value=Estimator()):
            result=direct_tree.build(prices,forecasts,ridge)
        fits=result["fitAudits"][f"{list(dashboard.PRODUCTS)[0]}/1"]["months"]
        self.assertTrue(fits)
        self.assertTrue(all(dt.date.fromisoformat(f["fitTrainTargetEnd"]) < dt.date.fromisoformat(f["fitOrigin"]) for f in fits if f["status"]=="fitted"))
        self.assertTrue(all("fitDistinctTargets" in row and row["fitSamples"]>=90 for row in result["predictions"]))
        self.assertEqual(len(result["comparisons"]),30)
        self.assertEqual(result["protocol"]["trainTargetsStrictlyBeforeFitOrigin"],True)

    def test_candidate_coverage_uses_raw_candidate_dates(self):
        prices,forecasts,ridge=fixture_data()
        # Remove one Ridge date while keeping that date available from the candidate and other controls.
        ridge["pairedRows"]=[r for r in ridge["pairedRows"] if r["target"]!="2015-04-16"]
        ridge["coverage"][f"{list(dashboard.PRODUCTS)[0]}/1"]["candidateAvailableDates"].remove("2015-04-16")
        with mock.patch.object(direct_tree,"HORIZONS",(1,)), mock.patch.object(direct_tree,"FAMILY_SIZE",30), \
             mock.patch.object(direct_tree,"EVAL_START",dt.date(2015,4,15)), mock.patch.object(direct_tree,"EVAL_END",dt.date(2015,4,25)), \
             mock.patch.object(direct_tree,"_model",return_value=Estimator()):
            result=direct_tree.build(prices,forecasts,ridge)
        cov=result["coverage"][f"{list(dashboard.PRODUCTS)[0]}/1"]
        self.assertEqual(cov["candidateCoverageLossDates"]["monthly_pooled_hist_gb"],0)
        self.assertLess(cov["commonPairedCount"],cov["candidateAvailableCount"])

    def test_month_failure_discards_partial_predictions(self):
        prices,forecasts,ridge=fixture_data()
        with mock.patch.object(direct_tree,"HORIZONS",(1,)), mock.patch.object(direct_tree,"FAMILY_SIZE",30), \
             mock.patch.object(direct_tree,"EVAL_START",dt.date(2015,4,15)), mock.patch.object(direct_tree,"EVAL_END",dt.date(2015,4,25)), \
             mock.patch.object(direct_tree,"_model",side_effect=FailAfterOnePrediction):
            result=direct_tree.build(prices,forecasts,ridge)
        self.assertEqual(result["predictions"],[])
        self.assertTrue(all(r["status"]=="fit_failed" for audit in result["fitAudits"].values() for r in audit["months"]))

    def test_metric_direction_alone_is_not_inferential_worsening(self):
        rows=[]
        for _ in range(6):
            rows.append({"candidateMetric":2.0,"comparatorMetric":1.0,
                "dm":{"9":{"status":"tested","n":50,"meanLossDifference":0.1,"holmSignificant":False}}})
        coverage={"candidateCoverageLossDates":{c:0 for c in direct_tree.COMPARATORS}}
        result=direct_tree._conclusion(rows,coverage)
        self.assertEqual(result["conclusion"],"inconclusive")
        self.assertEqual(result["descriptiveMetricDirection"],"higher")
        self.assertFalse(result["significantWorseningEvidence"])

    def test_ridge_actual_must_match_frozen_prices(self):
        prices,forecasts,ridge=fixture_data()
        target_row=next(r for r in ridge["pairedRows"] if r["target"]=="2015-04-16")
        target_row["actual"]+=1
        with mock.patch.object(direct_tree,"HORIZONS",(1,)), mock.patch.object(direct_tree,"FAMILY_SIZE",30), \
             mock.patch.object(direct_tree,"EVAL_START",dt.date(2015,4,15)), mock.patch.object(direct_tree,"EVAL_END",dt.date(2015,4,25)), \
             mock.patch.object(direct_tree,"_model",return_value=Estimator()):
            with self.assertRaisesRegex(ValueError,"actual does not match"):
                direct_tree.build(prices,forecasts,ridge)

    def test_monthly_histgb_parser_requires_unique_valid_predictions(self):
        _, forecasts, _=fixture_data()
        parsed=direct_tree._monthly_hist_gb(forecasts)
        self.assertTrue(parsed[list(dashboard.PRODUCTS)[0]])
        forecasts["products"][0]["runs"].append(forecasts["products"][0]["runs"][0])
        with self.assertRaises(ValueError): direct_tree._monthly_hist_gb(forecasts)


if __name__=="__main__": unittest.main()
