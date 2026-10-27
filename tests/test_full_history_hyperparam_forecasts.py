import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agri_data import full_history_hyperparam_forecasts as hp


class ConstantModel:
    def __init__(self,value=0.):self.value=value
    def predict(self,X):return [self.value for _ in X]


def cv_rows(family, scores=None, missing=None):
    configs=hp._configurations(family);rows=[]
    for c in configs:
        for year in (2023,2024):
            for h in hp.HORIZONS:
                if (c["id"],year,h)==missing:continue
                score=(scores or {}).get(c["id"],1.)
                rows.append({"configId":c["id"],"year":year,"horizon":h,"status":"scored","mape":score,"mae":score*10})
    return rows


class HyperparameterForecastTests(unittest.TestCase):
    def test_frozen_grid_and_default_configs(self):
        self.assertEqual([x["alpha"] for x in hp._configurations("ridge")],[.01,.1,1.,10.,100.,1000.])
        self.assertEqual([x["id"] for x in hp.XGB_CONFIGS],["X0","X1","X2","X3","X4","X5"])
        self.assertEqual([x["id"] for x in hp.RF_CONFIGS],["R0","R1","R2","R3","R4","R5"])
        self.assertEqual(hp.DEFAULTS,{"ridge":"ridge_a10","xgb":"X0","rf":"R0"})
        self.assertEqual(hp.FEATURE_LIBRARY_SHA256,"58bcde98d11239494ae13cc6169fb4c7f84a49a15ad11cf5b3cf5c49b4f39389")
        self.assertEqual([(c["n_estimators"],c["max_depth"],c["min_child_weight"],c["learning_rate"],c["reg_lambda"]) for c in hp.XGB_CONFIGS],[(300,3,15,.05,5),(200,2,30,.05,10),(500,2,10,.03,5),(200,4,10,.1,5),(500,3,5,.03,1),(300,4,30,.05,10)])
        self.assertEqual([(c["max_depth"],c["min_samples_leaf"],c["max_features"]) for c in hp.RF_CONFIGS],[(8,10,.7),(5,20,.5),(12,5,.5),(8,5,1.),(None,20,.7),(5,10,1.)])

    def test_selection_weights_year_then_horizon_and_uses_deterministic_ties(self):
        rows=cv_rows("ridge",{"ridge_a0p01":1.5,"ridge_a0p1":1.5,"ridge_a1":2.,"ridge_a10":3.,"ridge_a100":4.,"ridge_a1000":5.})
        result=hp._choose_family("ridge",rows)
        self.assertEqual(result["chosenConfig"]["id"],"ridge_a0p01")
        self.assertTrue(result["defaultSupported"])
        # A single unsupported 2024 distance disqualifies the config, while a
        # disqualified default makes the entire family unsupported.
        rows=cv_rows("ridge",missing=("ridge_a10",2024,10))
        result=hp._choose_family("ridge",rows)
        self.assertIsNone(result["chosenConfig"])
        self.assertFalse(result["defaultSupported"])
        self.assertEqual(result["reason"],"default_configuration_disqualified")

    def test_cv_uses_origin_year_cutoff_and_records_all_cells(self):
        data=[]
        for i in range(90):
            day=dt.date(2014,1,1)+dt.timedelta(days=i*10)
            data.append({"target":day,"horizon":1,"features":[float(i)],"p0":100.,"actual":100.,"y":0.,"audit":{}})
        data.append({"target":dt.date(2022,12,31),"horizon":1,"features":[90.],"p0":100.,"actual":100.,"y":0.,"audit":{}})
        data.append({"target":dt.date(2023,12,31),"horizon":1,"features":[91.],"p0":100.,"actual":100.,"y":0.,"audit":{}})
        for year in (2023,2024):
            for i in range(30):
                day=dt.date(year,1,3)+dt.timedelta(days=i)
                data.append({"target":day,"horizon":1,"features":[float(i)],"p0":100.,"actual":100.,"y":0.,"audit":{}})
        cache={(p,h):data for p in hp.PRODUCTS for h in hp.HORIZONS}
        config={"id":"only","alpha":10.}
        with mock.patch.object(hp,"FAMILIES",("ridge",)),mock.patch.object(hp,"_configurations",return_value=(config,)),mock.patch.object(hp,"_fit",return_value=ConstantModel()):
            cv=hp._cross_validate(cache)
        cells=cv[hp.PRODUCTS[0]]["ridge"]
        self.assertEqual(len(cells),20)
        self.assertTrue(all(r["status"]=="scored" and r["validationCount"]==(31 if r["year"]==2023 else 30) for r in cells))
        self.assertTrue(all(r["validationFirst"]==f"{r['year']}-01-03" for r in cells))
        self.assertTrue(all(r["validationLast"]==("2023-12-31" if r["year"]==2023 else "2024-02-01") for r in cells))
        self.assertTrue(all(r["trainLast"]==("2022-12-31" if r["year"]==2023 else "2023-12-31") for r in cells))

    def test_tree_cv_parallel_results_match_sequential_including_failures(self):
        data=[]
        for i in range(90):
            day=dt.date(2014,1,1)+dt.timedelta(days=i*10)
            data.append({"target":day,"horizon":1,"features":[float(i)],"p0":100.,"actual":100.,"y":0.})
        data.append({"target":dt.date(2022,12,31),"horizon":1,"features":[91.],"p0":100.,"actual":100.,"y":0.})
        for year in (2023,2024):
            for i in range(30):
                day=dt.date(year,1,3)+dt.timedelta(days=i)
                data.append({"target":day,"horizon":1,"features":[float(i)],"p0":100.,"actual":100.,"y":0.})
        cache={("cabbage",h):[{**r,"horizon":h} for r in data] for h in (1,2)}
        configs=({"id":"good","alpha":1.},{"id":"bad","alpha":2.})
        def fit(family,config,X,y):
            if config["id"]=="bad":raise RuntimeError("synthetic deterministic failure")
            return ConstantModel()
        with mock.patch.object(hp,"PRODUCTS",("cabbage",)),mock.patch.object(hp,"HORIZONS",(1,2)),\
             mock.patch.object(hp,"FAMILIES",("rf",)),mock.patch.object(hp,"_configurations",return_value=configs),\
             mock.patch.object(hp,"_fit",side_effect=fit):
            serial=hp._cross_validate(cache,max_workers=1)
        with mock.patch.object(hp,"PRODUCTS",("cabbage",)),mock.patch.object(hp,"HORIZONS",(1,2)),\
             mock.patch.object(hp,"FAMILIES",("rf",)),mock.patch.object(hp,"_configurations",return_value=configs),\
             mock.patch.object(hp,"_fit",side_effect=fit):
            parallel=hp._cross_validate(cache,max_workers=4)
        self.assertEqual(serial,parallel)
        rows=serial["cabbage"]["rf"]
        self.assertEqual([r["configId"] for r in rows],["good","good","good","good","bad","bad","bad","bad"])
        self.assertEqual(sum(r["status"]=="failed" for r in rows),4)

    def test_answer_free_phase_schema_and_exclusive_output(self):
        day=dt.date(2025,1,2);h=1
        train=[{"target":dt.date(2020,1,1),"features":[0.],"p0":100.,"actual":100.,"y":0.},
               {"target":day,"features":[1.],"p0":100.,"actual":None,"y":None}]
        cache={(p,hh):train for p in hp.PRODUCTS for hh in hp.HORIZONS}
        selection={p:{f:{"chosenConfig":hp._default_config(f),"defaultSupported":True,"reason":None} for f in hp.FAMILIES} for p in hp.PRODUCTS}
        grid=[(hp.PRODUCTS[0],day,h)]
        with mock.patch.object(hp,"_fit_period",return_value=(ConstantModel(),{"count":100,"firstTarget":"2014-01-01","lastTarget":"2024-12-31"},None)):
            rows=hp._emit_phase(cache,selection,"confirmation",grid,max_workers=1)
            parallel=hp._emit_phase(cache,selection,"confirmation",grid,max_workers=4)
        self.assertEqual(rows,parallel)
        row=rows[0]
        self.assertEqual(set(row["models"]),{"ridgeTuned","ridgeDefault","xgbTuned","xgbDefault","rfTuned","rfDefault"})
        self.assertTrue(all(x["status"]=="issued" and x["prediction"]>0 for x in row["models"].values()))
        self.assertNotIn("actual",row)
        def fit_with_tree_failure(family,config,data,end):
            training={"count":100}
            return (None,training,"fit_failed:RuntimeError") if family=="rf" else (ConstantModel(),training,None)
        with mock.patch.object(hp,"_fit_period",side_effect=fit_with_tree_failure):
            serial_failure=hp._emit_phase(cache,selection,"confirmation",grid,max_workers=1)
            parallel_failure=hp._emit_phase(cache,selection,"confirmation",grid,max_workers=4)
        self.assertEqual(serial_failure,parallel_failure)
        self.assertEqual(row["models"].keys(),parallel_failure[0]["models"].keys())
        self.assertTrue(all(parallel_failure[0]["models"][arm]["reason"]=="fit_failed:RuntimeError" for arm in ("rfTuned","rfDefault")))
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/"manifest.json";hp.write_exclusive(path,{"ok":True})
            with self.assertRaises(FileExistsError):hp.write_exclusive(path,{"ok":False})
            self.assertEqual(json.loads(path.read_text()),{"ok":True})

    def test_manifest_validator_checks_both_complete_calendar_grids(self):
        train={"target":dt.date(2020,1,1),"features":[0.],"p0":100.,"actual":100.,"y":0.}
        conf_day=dt.date(2025,1,2);eval_day=dt.date(2026,1,2)
        conf_item={**train,"target":conf_day,"actual":None,"y":None}
        eval_item={**train,"target":eval_day,"actual":None,"y":None}
        cache={(p,h):[train,conf_item,eval_item] for p in hp.PRODUCTS for h in hp.HORIZONS}
        selection={p:{f:{"chosenConfig":hp._default_config(f),"defaultSupported":True,"reason":None} for f in hp.FAMILIES} for p in hp.PRODUCTS}
        with mock.patch.object(hp,"PRODUCTS",("cabbage",)),mock.patch.object(hp,"HORIZONS",(1,)),\
             mock.patch.object(hp,"CONF_START",conf_day),mock.patch.object(hp,"CONF_END",conf_day),\
             mock.patch.object(hp,"EVAL_START",eval_day),mock.patch.object(hp,"EVAL_END",eval_day),\
             mock.patch.object(hp,"_fit_period",return_value=(ConstantModel(),{"count":100},None)):
            confirmation=hp._emit_phase(cache,selection,"confirmation",hp._grid(conf_day,conf_day))
            evaluation=hp._emit_phase(cache,selection,"evaluation",hp._grid(eval_day,eval_day),{("cabbage",eval_day,1):123.})
            doc={"phase":"forecast_only","targetValuesAttached":False,"confirmationForecasts":confirmation,"evaluationForecasts":evaluation}
            doc["manifestSha256"]=__import__("hashlib").sha256(hp._canonical(doc)).hexdigest()
            hp._validate_manifest(doc)
            broken={**doc,"evaluationForecasts":[{**evaluation[0],"fixed50":{"status":"issued","reason":None,"prediction":None}}]}
            broken["manifestSha256"]=__import__("hashlib").sha256(hp._canonical({k:v for k,v in broken.items() if k!="manifestSha256"})).hexdigest()
            with self.assertRaises(ValueError):hp._validate_manifest(broken)

    def test_cache_never_attaches_2026_actuals(self):
        day1,day2=dt.date(2026,1,1),dt.date(2026,1,2)
        rec={day1:{"price":100.,"volume":10.,"rowCount":2.,"markets":frozenset()},day2:{"price":999.,"volume":10.,"rowCount":2.,"markets":frozenset()}}
        with mock.patch.object(hp,"PRODUCTS",("cabbage",)),mock.patch.object(hp,"HORIZONS",(1,)),mock.patch.object(hp,"EVAL_END",day2):
            cache=hp._make_cache({"cabbage":rec})[("cabbage",1)]
        target_row=next(r for r in cache if r["target"]==day2)
        self.assertIsNone(target_row["actual"]);self.assertIsNone(target_row["y"])

    def test_synthetic_build_replays_answer_free_two_period_grids_without_model_fitting(self):
        start=dt.date(2014,1,1)
        raw={"schemaVersion":1,"products":[{"id":p,"observations":[
            {"date":start.isoformat(),"actual":"100","volumeKg":"10","rowCount":"2","marketCodes":["110001"]},
            {"date":(start+dt.timedelta(days=1)).isoformat(),"actual":"101","volumeKg":"11","rowCount":"3","marketCodes":["110001","110008"]},
        ]} for p in hp.narrow.PRODUCTS]}
        fixed={"phase":"forecast_only","targetValuesAttached":False,"forecasts":[]}
        target=start+dt.timedelta(days=2)
        with mock.patch.object(hp,"PRODUCTS",("cabbage",)),mock.patch.object(hp,"HORIZONS",(1,)),\
             mock.patch.object(hp,"CONF_START",target),mock.patch.object(hp,"CONF_END",target),\
             mock.patch.object(hp,"EVAL_START",target),mock.patch.object(hp,"EVAL_END",target),\
             mock.patch.object(hp,"_load_xgb",return_value=None):
            doc=hp.build(raw,fixed,price_sha="synthetic-prices",fixed50_sha="synthetic-fixed50")
            hp._validate_manifest(doc)
        self.assertEqual(doc["plannedConfirmationCount"],1)
        self.assertEqual(doc["plannedEvaluationCount"],1)
        row=doc["confirmationForecasts"][0]
        self.assertTrue(all(x["status"]=="not_issued" for x in row["models"].values()))
        self.assertFalse(doc["targetValuesAttached"])
        self.assertIsNotNone(doc["manifestSha256"])


if __name__=="__main__":unittest.main()
