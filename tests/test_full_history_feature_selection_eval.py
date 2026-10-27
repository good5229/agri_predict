import datetime as dt
import unittest
from unittest import mock

from agri_data import full_history_feature_selection_eval as ev


def synthetic_prices():
    days=[]; d=dt.date(2014,1,1); end=dt.date(2026,1,4)
    while d<=end:
        days.append(d); d+=dt.timedelta(days=7)
    if days[-1]!=end: days.append(end)
    products=[]
    for pi,p in enumerate(ev.PRODUCTS):
        rows=[]
        for i,day in enumerate(days):
            rows.append({"date":day.isoformat(),"actual":100+pi*20+i*.05+(i%11)*.03,"volumeKg":400+i%29,"rowCount":70+i%17,"marketCodes":["110001","110008"]})
        products.append({"id":p,"observations":rows})
    return {"schemaVersion":1,"products":products}


def comparators(products=None,horizons=None,days=3):
    products=products or ev.PRODUCTS; horizons=horizons or ev.HORIZONS
    rows=[]
    for p in products:
        for i in range(days):
            d=dt.date(2026,1,1)+dt.timedelta(days=i)
            for h in horizons:
                rows.append({"product":p,"target":d.isoformat(),"horizon":h,"richRidge":110.,"fixed50":105.})
    rich={"phase":"forecast_only","targetValuesAttached":False,"forecasts":rows}
    fixed={"phase":"forecast_only","targetValuesAttached":False,"forecasts":[{"product":p,"target":d.isoformat(),"horizon":h,"predictions":{"fixed50":105.}} for p in products for i in range(days) for d in [dt.date(2026,1,1)+dt.timedelta(days=i)] for h in horizons]}
    return rich,fixed


class FixedModel:
    def predict(self, X): return [0.0 for _ in X]


class FeatureSelectionTests(unittest.TestCase):
    def test_grid_is_full_calendar_answer_free_and_excludes_prefit_origins(self):
        data=synthetic_prices(); cmp,fix=comparators(("cabbage",),(1,)); data["products"]=[p for p in data["products"] if p["id"]=="cabbage"]
        with mock.patch.object(ev,"PRODUCTS",("cabbage",)), mock.patch.object(ev,"HORIZONS",(1,)), mock.patch.object(ev,"EVAL_START",dt.date(2026,1,1)), mock.patch.object(ev,"EVAL_END",dt.date(2026,1,3)), mock.patch.object(ev,"MIN_TRAIN",3), mock.patch.object(ev,"_check_sources"), mock.patch.object(ev,"_ridge",side_effect=lambda X,y:FixedModel()):
            manifest=ev.build(data,cmp,fix)
            self.assertEqual(manifest["phase"],"forecast_only")
            self.assertIs(manifest["targetValuesAttached"],False)
            self.assertEqual(manifest["plannedForecastCount"],3)
            self.assertTrue(all("actual" not in r for r in manifest["forecasts"]))
            self.assertFalse(manifest["forecasts"][0]["eligible"])
            self.assertEqual(manifest["forecasts"][0]["selectedReason"],"origin_before_2026")
            self.assertTrue(all(r["eligible"] for r in manifest["forecasts"][1:]))
            ev._check_manifest(manifest)

    def test_feature_builder_cutoff_excludes_post_origin_mutations(self):
        raw=synthetic_prices(); records,_=ev._raw_maps(raw); rec=records["cabbage"]; ordered=sorted(rec); pmap={d:r["price"] for d,r in rec.items()}
        target=dt.date(2026,1,4); h=1
        x,p0,a=ev._feature(rec,target,h,ordered,pmap)
        changed={d:dict(r) for d,r in rec.items()}; changed_prices=dict(pmap)
        origin=target-dt.timedelta(days=h)
        for d in changed:
            if d>=origin:
                changed[d]["price"]*=1000;changed[d]["volume"]*=1000;changed[d]["rowCount"]*=1000;changed_prices[d]*=1000
        x2,p02,a2=ev._feature(changed,target,h,ordered,changed_prices)
        self.assertEqual((x,p0,a),(x2,p02,a2))

    def test_selection_tie_breaks_to_smaller_n_and_has_no_2026_targets(self):
        raw=synthetic_prices();cmp,fix=comparators(("cabbage",),(1,)); raw["products"]=[p for p in raw["products"] if p["id"]=="cabbage"]
        with mock.patch.object(ev,"PRODUCTS",("cabbage",)), mock.patch.object(ev,"HORIZONS",(1,)), mock.patch.object(ev,"EVAL_START",dt.date(2026,1,1)), mock.patch.object(ev,"EVAL_END",dt.date(2026,1,3)), mock.patch.object(ev,"MIN_TRAIN",3), mock.patch.object(ev,"_check_sources"), mock.patch.object(ev,"_ridge",side_effect=lambda X,y:FixedModel()):
            out=ev.build(raw,cmp,fix)
        sel=out["selection"]["cabbage"]
        self.assertEqual(sel["selectedN"],8)
        
        validation=[x for v in sel["nValidation"].values() for x in v["horizons"]]
        self.assertTrue(all(x["firstTarget"][:4]<="2025" for x in validation))
        self.assertTrue(all(x["trainingCount"]>0 and x["trainingFirst"] is not None and x["trainingLast"]<="2024-12-31" for x in validation))
        self.assertEqual(len(out["protocol"]["featureNames"]),len(ev.library.FEATURE_NAMES))



    def test_forecast_ignores_target_actual_and_issues_when_target_is_unobserved(self):
        base=synthetic_prices(); base["products"]=[p for p in base["products"] if p["id"]=="cabbage"]
        rich,fix=comparators(("cabbage",),(1,),days=4)
        changed=__import__("copy").deepcopy(base)
        for row in changed["products"][0]["observations"]:
            if row["date"]=="2026-01-04":row["actual"]*=1000
        absent=__import__("copy").deepcopy(base)
        absent["products"][0]["observations"]=[r for r in absent["products"][0]["observations"] if r["date"]!="2026-01-04"]
        with mock.patch.object(ev,"PRODUCTS",("cabbage",)), mock.patch.object(ev,"HORIZONS",(1,)), mock.patch.object(ev,"EVAL_START",dt.date(2026,1,1)), mock.patch.object(ev,"EVAL_END",dt.date(2026,1,4)), mock.patch.object(ev,"MIN_TRAIN",3), mock.patch.object(ev,"_check_sources"), mock.patch.object(ev,"_ridge",side_effect=lambda X,y:FixedModel()):
            a=ev.build(base,rich,fix); b=ev.build(changed,rich,fix); c=ev.build(absent,rich,fix)
        def forecast(doc):return next(r for r in doc["forecasts"] if r["target"]=="2026-01-04")
        ra,rb,rc=map(forecast,(a,b,c))
        self.assertEqual((ra["selected"],ra["allFeature"]),(rb["selected"],rb["allFeature"]))
        self.assertEqual((ra["selected"],ra["allFeature"]),(rc["selected"],rc["allFeature"]))
        self.assertEqual(ra["selectedStatus"],"issued");self.assertEqual(rc["selectedStatus"],"issued")

    def test_score_uses_four_way_common_pairs_and_full_holm_families(self):
        start=dt.date(2026,1,11); end=dt.date(2026,2,9)
        forecasts=[]; products=[]
        for p in ev.PRODUCTS:
            observations=[]
            for i in range(30):
                day=start+dt.timedelta(days=i); actual=100.+i
                observations.append({"date":day.isoformat(),"actual":actual})
                for h in ev.HORIZONS:
                    forecasts.append({"product":p,"target":day.isoformat(),"origin":(day-dt.timedelta(days=h)).isoformat(),"horizon":h,"eligible":True,"selected":actual+1.,"selectedStatus":"issued","selectedReason":None,"allFeature":actual+2.,"allFeatureStatus":"issued","allFeatureReason":None,"richRidge":actual+3.,"fixed50":actual+4.})
            products.append({"id":p,"observations":observations})
        with mock.patch.object(ev,"EVAL_START",start),mock.patch.object(ev,"EVAL_END",end):
            manifest={"phase":"forecast_only","targetValuesAttached":False,"forecasts":forecasts}
            manifest["manifestSha256"]=__import__("hashlib").sha256(ev._canonical(manifest)).hexdigest()
            result=ev.score(manifest,{"schemaVersion":1,"products":products})
        self.assertEqual(len(result["commonPairs"]),5*30*10)
        self.assertEqual(len(result["dm"]),800)
        self.assertEqual(len(result["metrics"]),5*10*4)
        chosen=next(r for r in result["metrics"] if r["product"]=="cabbage" and r["horizon"]==1 and r["comparison"]=="selected_vs_fixed50")
        self.assertEqual(chosen["a"]["n"],30);self.assertEqual(chosen["b"]["n"],30)
        self.assertAlmostEqual(chosen["a"]["mae"],1.);self.assertAlmostEqual(chosen["b"]["mae"],4.)
        self.assertTrue(all(r["n"]==30 and r["status"] in ("tested","zero_spread") for r in result["dm"]))
        with mock.patch.object(ev,"EVAL_START",start),mock.patch.object(ev,"EVAL_END",end):
            worse={"phase":"forecast_only","targetValuesAttached":False,"forecasts":[dict(r,selected=r["fixed50"]+1.) for r in forecasts]}
            worse["manifestSha256"]=__import__("hashlib").sha256(ev._canonical(worse)).hexdigest()
            worse_result=ev.score(worse,{"schemaVersion":1,"products":products})
            cabbage_gate=next(g for g in worse_result["candidateGates"] if g["product"]=="cabbage")
            self.assertLess(cabbage_gate["mapeImprovementPctPoints"],0.)
            self.assertEqual(cabbage_gate["decision"],"retain_existing_method")


    def test_no_rank_eligible_horizon_skips_all_candidate_fits(self):
        raw=synthetic_prices(); raw["products"]=[p for p in raw["products"] if p["id"]=="cabbage"]
        rich,fix=comparators(("cabbage",),(1,),days=3)
        with mock.patch.object(ev,"PRODUCTS",("cabbage",)), mock.patch.object(ev,"HORIZONS",(1,)), mock.patch.object(ev,"EVAL_START",dt.date(2026,1,1)), mock.patch.object(ev,"EVAL_END",dt.date(2026,1,3)), mock.patch.object(ev,"MIN_TRAIN",10000), mock.patch.object(ev,"_check_sources"), mock.patch.object(ev,"_ridge",side_effect=AssertionError("must not fit empty ranking")):
            out=ev.build(raw,rich,fix)
        sel=out["selection"]["cabbage"]
        self.assertIsNone(sel["selectedN"])
        self.assertEqual(sel["nValidation"]["8"]["reason"],"no_rank_eligible_horizons")
        self.assertTrue(all(r["selectedStatus"]=="not_issued" for r in out["forecasts"]))

    def test_n_selection_mae_is_in_price_units(self):
        self.assertAlmostEqual(ev._mae([0.],[__import__("math").log(2.)],[100.]),100.)

    def test_dm_and_holm_preserve_unavailable_tests_in_family(self):
        d={dt.date(2026,1,1)+dt.timedelta(days=i):float(i%5) for i in range(35)}
        tested=ev._dm(d,9);small=ev._dm(dict(list(d.items())[:10]),9)
        self.assertEqual(tested["status"],"tested")
        self.assertEqual(small["status"],"insufficient_pairs")
        rows=[{"pValue":tested["pValue"]},{"pValue":1.0}];ev._holm(rows)
        self.assertEqual(len(rows),2);self.assertGreaterEqual(rows[0]["holmPValue"],rows[0]["pValue"])

if __name__=="__main__":unittest.main()
