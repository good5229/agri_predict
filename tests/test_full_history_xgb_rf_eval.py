import datetime as dt
import unittest
import sys
from types import SimpleNamespace
from unittest import mock

from agri_data import full_history_xgb_rf_eval as ev


def prices(end=dt.date(2026,1,12)):
    start=dt.date(2014,1,1)
    if end.year==2014:
        dates=[start+dt.timedelta(days=i) for i in range((end-start).days+1)]
    else:
        dates=[];day=start
        while day<=dt.date(2025,12,31):
            dates.append(day);day+=dt.timedelta(days=7)
        if dates[-1]!=dt.date(2025,12,31):dates.append(dt.date(2025,12,31))
        eval_day=dt.date(2026,1,1)
        while eval_day<=end:dates.append(eval_day);eval_day+=dt.timedelta(days=1)
    products=[]
    for j,p in enumerate(ev.PRODUCTS):
        rows=[]
        for i,day in enumerate(dates):
            rows.append({"date":day.isoformat(),"actual":100+j*20+i*.02+((i%13)-6)*.1,"volumeKg":500+j*10+i%23,"rowCount":100+i%19,"marketCodes":["110001","110008"] if i%5 else ["110001"]})
        products.append({"id":p,"observations":rows})
    return {"schemaVersion":1,"products":products}


def narrow_manifest(rows=()):
    return {"phase":"forecast_only","targetValuesAttached":False,"forecasts":[
        {"product":p,"target":d,"horizon":h,"prediction":120.,"persistence":100.,"fixed50":110.} for p,d,h in rows]}


def fixed50(rows=()):
    return {"phase":"forecast_only","targetValuesAttached":False,"forecasts":[
        {"product":p,"target":d,"horizon":h,"predictions":{"fixed50":110.}} for p,d,h in rows]}


class FakeModel:
    def __init__(self,value=0.,overflow=False):self.value=value;self.overflow=overflow
    def fit(self,x,y):return self
    def predict(self,x):
        if self.overflow:return [0. if i==0 else 1000. for i in range(len(x))]
        return [self.value for _ in x]


class XgbRfTests(unittest.TestCase):
    def setUp(self):
        self.patches=[mock.patch.object(ev,"_load_xgboost",return_value=object()),
                      mock.patch.object(ev,"_fit_xgb",side_effect=lambda x,y:FakeModel()),
                      mock.patch.object(ev,"_fit_rf",side_effect=lambda x,y:FakeModel()),
                      mock.patch.object(ev,"_fit_rich_ridge",side_effect=lambda x,y:FakeModel())]
        for p in self.patches:p.start()
        self.addCleanup(self._stop)
    def _stop(self):
        for p in reversed(self.patches):p.stop()

    def test_rich_price_vector_is_exact_ridge_prefix_and_post_origin_changes_do_not_leak(self):
        raw=prices(); records,audit=ev._raw_maps(raw); rec=records["cabbage"];price_values={d:r["price"] for d,r in rec.items()};ordered=sorted(rec)
        target=dt.date(2026,1,12);h=1
        feat,p0,_=ev._rich_features(rec,target,h,ordered,price_values)
        base,p0b,_=ev.narrow._features(price_values,target,h,ordered)
        self.assertEqual(feat[:len(base)],base);self.assertEqual(p0,p0b)
        mutated={d:dict(v) for d,v in rec.items()};changed_prices=dict(price_values)
        for day in list(mutated):
            if day>dt.date(2026,1,10):
                mutated[day]["price"]*=1000;mutated[day]["volume"]*=1000;mutated[day]["rowCount"]*=1000
                changed_prices[day]*=1000
        feat2,p02,_=ev._rich_features(mutated,target,h,ordered,changed_prices)
        self.assertEqual((feat,p0),(feat2,p02))

    def test_price_windows_volume_masks_and_source_null_or_signed_values(self):
        raw=prices(end=dt.date(2014,1,5));cabbage=raw["products"][0]
        cabbage["observations"][0]["volumeKg"]=-7
        cabbage["observations"][1]["rowCount"]=None
        cabbage["observations"][2]["actual"]=None
        original=[dict(r) for r in cabbage["observations"]]
        records,_=ev._raw_maps(raw)
        self.assertEqual(cabbage["observations"],original)
        # For p0=Jan 2, earlier negative volume is absent, with ratio masks false.
        features,p0,audit=ev._rich_features(records["cabbage"],dt.date(2014,1,4),1)
        self.assertIsNotNone(features)
        base_count=audit["priceFeatureCount"]
        # The three volume windows each end with [ratio, count, availability].
        self.assertEqual(features[base_count+12],0.)
        self.assertEqual(features[base_count+14],0.)
        self.assertEqual(features[base_count+21],0.)
        self.assertEqual(features[base_count+23],0.)
        self.assertGreater(ev._raw_maps(raw)[1]["cabbage"]["invalidOrNonpositiveVolume"],0)

    def test_participating_market_count_and_prior_dual_market_share(self):
        dates=[dt.date(2026,1,i) for i in range(1,5)]
        rec={d:{"price":100.+i,"volume":10.+i,"rowCount":20.+i,"markets":frozenset(("110001","110008")) if i in (1,3,4) else frozenset(("110001",)),"rawVolume":None,"rawRowCount":None,"rawMarketCodes":None} for i,d in enumerate(dates,1)}
        feat,_,audit=ev._rich_features(rec,dt.date(2026,1,6),1,dates,{d:r["price"] for d,r in rec.items()})
        base_count=audit["priceFeatureCount"]
        self.assertEqual(feat[base_count+24],2.)
        self.assertAlmostEqual(feat[base_count+25],2/3)

    def test_build_and_score_use_only_all_five_model_common_pairs(self):
        data=prices();keys=[("cabbage","2026-01-02",1)]
        with mock.patch.object(ev,"EVAL_END",dt.date(2026,1,4)):
            manifest=ev.build(data,narrow_manifest(keys),fixed50(keys))
            row=next(r for r in manifest["forecasts"] if r["product"]=="cabbage" and r["target"]=="2026-01-02" and r["horizon"]==1)
            self.assertEqual(row["xgbStatus"],"issued");self.assertEqual(row["rfStatus"],"issued");self.assertEqual(row["richRidgeStatus"],"issued")
            result=ev.score(manifest,data)
        self.assertEqual(len(result["commonPairs"]),1)
        self.assertEqual(result["commonPairs"][0]["target"],"2026-01-02")
        self.assertEqual(len([r for r in result["dm"] if r["bandwidth"]==9]),600)
        self.assertEqual(len([r for r in result["dm"] if r["bandwidth"]==28]),600)
        self.assertEqual(result["coverage"]["cabbage/1"]["planned"],4)

    def test_failed_batch_has_no_scored_rows_and_dependency_failure_is_explicit(self):
        data=prices();keys=[("cabbage","2026-01-02",1), ("cabbage","2026-01-03",1)]
        with mock.patch.object(ev,"EVAL_END",dt.date(2026,1,4)),mock.patch.object(ev,"_fit_rf",side_effect=lambda x,y:FakeModel(overflow=True)):
            manifest=ev.build(data,narrow_manifest(keys),fixed50(keys))
            failed=[r for r in manifest["forecasts"] if r["product"]=="cabbage" and r["horizon"]==1 and r["target"] in ("2026-01-02","2026-01-03")]
            self.assertTrue(all(r["rfStatus"]=="not_issued" and r["rf"] is None for r in failed))
            score=ev.score(manifest,data)
        self.assertFalse(any(r["product"]=="cabbage" and r["horizon"]==1 for r in score["commonPairs"]))
        with mock.patch.object(ev,"_load_xgboost",side_effect=RuntimeError("XGBoost 2.1.4 unavailable")),mock.patch.object(ev,"_fit_xgb") as fit:
            with self.assertRaisesRegex(RuntimeError,"2.1.4 unavailable"):
                ev.build(data,narrow_manifest(),fixed50())
            fit.assert_not_called()

    def test_xgboost_version_mismatch_fails_without_path_restrictions(self):
        self.patches[0].stop()
        fake=SimpleNamespace(__version__="3.0.0",__file__="/any/installed/path/xgboost.py",XGBRegressor=object)
        with mock.patch.dict(sys.modules,{"xgboost":fake}):
            with self.assertRaisesRegex(RuntimeError,"2.1.4 required; found 3.0.0"):
                ev._load_xgboost()

    def test_constant_fractional_dm_is_inconclusive(self):
        for value in (.1,-.1):
            diff={dt.date(2026,1,1)+dt.timedelta(days=i):value for i in range(40)}
            row=ev._dm(diff,9)
            self.assertEqual(row["status"],"zero_spread");self.assertEqual(row["pValue"],1.)


if __name__=="__main__":unittest.main()
