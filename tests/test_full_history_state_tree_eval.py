import datetime as dt
import hashlib
import copy
import math
import unittest
from unittest import mock

from agri_data import full_history_state_tree_eval as ev


def prices(end=dt.date(2026,1,20)):
    start=dt.date(2014,1,1); n=(end-start).days+1; products=[]
    for j,p in enumerate(ev.PRODUCTS):
        observations=[]
        for i in range(n):
            day=start+dt.timedelta(days=i)
            observations.append({"date":day.isoformat(),"actual":100+j*10+i*.01+((i%11)-5)*.1,"volumeKg":500+j*20+i%17,"rowCount":i%20+1})
        products.append({"id":p,"observations":observations})
    return {"schemaVersion":1,"products":products}


def ridge_manifest(rows=()):
    return {"phase":"forecast_only","targetValuesAttached":False,"forecasts":[
        {"product":p,"target":d,"horizon":h,"prediction":120.,"fixed50":110.} for p,d,h in rows]}


def fixed50(rows=()):
    return {"phase":"forecast_only","targetValuesAttached":False,"forecasts":[
        {"product":p,"target":d,"horizon":h,"predictions":{"fixed50":110.}} for p,d,h in rows]}


class ZeroTree:
    def fit(self,x,y): return self
    def predict(self,x): return [0. for _ in x]


class StateTreeTests(unittest.TestCase):
    def test_volume_features_use_strictly_pre_p0_windows_and_track_missingness(self):
        p0=dt.date(2024,1,8)
        volumes={dt.date(2024,1,1):2.,dt.date(2024,1,7):6.,p0:100.,dt.date(2024,1,9):99999.}
        f=ev._volume_features(volumes,p0)
        self.assertAlmostEqual(f[0],__import__('math').log(100/4))
        self.assertAlmostEqual(f[2],0.0)
        self.assertEqual(f[6],2.0)
        # No earlier volume exists for the 30-day median in this variant.
        only_v0=ev._volume_features({p0:100.,dt.date(2024,1,9):99999.},p0)
        self.assertEqual(only_v0[:6],[0.,0.,0.,0.,0.,0.])
        self.assertEqual(only_v0[6:],[0.,0.])

    def test_tree_price_matches_ridge_vector_and_post_origin_values_do_not_change_features(self):
        target=dt.date(2026,1,10); horizon=2
        price={dt.date(2026,1,i):100.+i for i in range(1,13)}
        volume={dt.date(2026,1,i):500.+10*i for i in range(1,13)}
        pfeat,vfeat,p0,audit=ev._combined_features(price,volume,target,horizon)
        ridge_features,ridge_p0,ridge_audit=ev.ridge._features(price,target,horizon)
        self.assertEqual(pfeat,ridge_features);self.assertEqual(p0,ridge_p0);self.assertEqual(audit,ridge_audit)
        self.assertEqual(vfeat[:len(pfeat)],ridge_features)
        changed_price=dict(price);changed_volume=dict(volume)
        for i in (8,9,10,11,12):
            changed_price[dt.date(2026,1,i)]*=1000
            changed_volume[dt.date(2026,1,i)]*=1000
        changed=ev._combined_features(changed_price,changed_volume,target,horizon)
        self.assertEqual((pfeat,vfeat,p0),(changed[0],changed[1],changed[2]))

    def test_invalid_volume_is_masked_and_source_is_not_mutated(self):
        raw=prices(end=dt.date(2014,1,3)); before=copy.deepcopy(raw)
        cabbage=next(p for p in raw["products"] if p["id"]=="cabbage")
        cabbage["observations"][0]["volumeKg"]=-5
        cabbage["observations"][1]["volumeKg"]=None
        raw_before=copy.deepcopy(raw)
        volumes,audit=ev._volume_data(raw)
        f=ev._volume_features(volumes["cabbage"],dt.date(2014,1,1))
        self.assertEqual(f[:6],[0.,0.,0.,0.,0.,0.])
        self.assertEqual(audit["cabbage"]["nonpositiveOrInvalidVolumeRows"],2)
        self.assertEqual(raw,raw_before)
        self.assertNotEqual(raw,before)

    def test_build_uses_2014_targets_and_keeps_prefit_and_stale_rows_unissued(self):
        raw=prices()
        # Remove recent radish observations so a late January target has a stale p0.
        radish=next(p for p in raw["products"] if p["id"]=="radish")
        radish["observations"]=[r for r in radish["observations"] if r["date"]<"2026-01-01"]
        with mock.patch.object(ev,"EVAL_END",dt.date(2026,1,20)),mock.patch.object(ev,"_fit_tree",return_value=ZeroTree()):
            doc=ev.build(raw,ridge_manifest(),fixed50())
        self.assertEqual(doc["plannedForecastCount"],5*20*10)
        h1=next(r for r in doc["forecasts"] if r["product"]=="cabbage" and r["horizon"]==1 and r["target"]=="2026-01-02")
        self.assertEqual(h1["training"]["firstTarget"][:4],"2014")
        self.assertEqual(h1["training"]["lastTarget"],"2025-12-31")
        self.assertEqual(h1["treePriceStatus"],"issued")
        early=next(r for r in doc["forecasts"] if r["product"]=="cabbage" and r["horizon"]==10 and r["target"]=="2026-01-10")
        self.assertFalse(early["eligible"])
        self.assertEqual(early["treeVolumeReason"],"model_not_yet_fitted_at_origin")
        stale=next(r for r in doc["forecasts"] if r["product"]=="radish" and r["horizon"]==1 and r["target"]=="2026-01-20")
        self.assertEqual(stale["treePriceStatus"],"not_issued")
        self.assertEqual(stale["treePriceReason"],"p0_older_than_14_days")

    def test_score_uses_four_model_fixed50_common_pairs_and_full_planned_coverage(self):
        raw=prices()
        keyrows=[("cabbage","2026-01-01",1),("cabbage","2026-01-04",1)]
        with mock.patch.object(ev,"EVAL_END",dt.date(2026,1,4)),mock.patch.object(ev,"_fit_tree",return_value=ZeroTree()):
            doc=ev.build(raw,ridge_manifest(keyrows),fixed50(keyrows))
            row=next(r for r in doc["forecasts"] if r["product"]=="cabbage" and r["target"]=="2026-01-04" and r["horizon"]==1)
            row["treeVolumeStatus"]="not_issued";row["treeVolumeReason"]="synthetic_model_failure";row["treeVolume"]=None
            unsigned=dict(doc);unsigned.pop("manifestSha256");doc["manifestSha256"]=hashlib.sha256(ev._canonical(unsigned)).hexdigest()
            answers=prices()
            cabbage=next(p for p in answers["products"] if p["id"]=="cabbage")
            cabbage["observations"]=[r for r in cabbage["observations"] if r["date"]!="2026-01-03"]
            result=ev.score(doc,answers)
        self.assertEqual(len(result["commonPairs"]),0)
        self.assertEqual(result["coverage"]["cabbage/1"]["planned"],4)
        self.assertEqual(result["coverage"]["cabbage/1"]["excludedPrefit"],1)
        self.assertEqual(result["coverage"]["cabbage/1"]["treeVolumeCoverageLossTargets"],["2026-01-04"])
        self.assertEqual(result["temporalCoverage"]["cabbage/month/2026-01"]["planned"],40)
        self.assertEqual(result["temporalCoverage"]["cabbage/month/2026-01"]["positiveActualPairs"],30)
        self.assertFalse(next(g for g in result["cropGates"] if g["product"]=="cabbage")["coverageNondeclining"])
        self.assertEqual(sum(r["bandwidth"]==9 for r in result["dm"]),400)
        self.assertEqual(sum(r["bandwidth"]==28 for r in result["dm"]),400)

    def test_partial_prediction_overflow_clears_volume_arm_and_failed_rows_do_not_score(self):
        class OverflowTree:
            def fit(self,x,y):return self
            def predict(self,x):return [0. if i==0 else 1000. for i in range(len(x))]
        raw=prices()
        keys=[("cabbage","2026-01-02",1),("cabbage","2026-01-03",1)]
        sequence=[]
        for _ in range(50): sequence.extend([ZeroTree(),OverflowTree()])
        with mock.patch.object(ev,"EVAL_END",dt.date(2026,1,4)),mock.patch.object(ev,"_fit_tree",side_effect=sequence):
            doc=ev.build(raw,ridge_manifest(keys),fixed50(keys))
            failing=[r for r in doc["forecasts"] if r["product"]=="cabbage" and r["horizon"]==1 and r["target"] in ("2026-01-02","2026-01-03")]
            self.assertTrue(all(r["treeVolumeStatus"]=="not_issued" and r["treeVolume"] is None for r in failing))
            scored=ev.score(doc,raw)
        self.assertFalse(any(r["product"]=="cabbage" and r["horizon"]==1 for r in scored["commonPairs"]))

    def test_constant_fractional_dm_differences_are_unavailable(self):
        for value in (0.1,-0.1):
            differences={dt.date(2026,1,1)+dt.timedelta(days=i):value for i in range(40)}
            result=ev._dm(differences,9)
            self.assertEqual(result["status"],"zero_spread")
            self.assertEqual(result["pValue"],1.0)


if __name__=="__main__":unittest.main()
