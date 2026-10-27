import datetime as dt
import math
import statistics
import unittest

from agri_data.full_history_feature_library import FEATURE_NAMES, features
from agri_data import full_history_xgb_rf_eval as prior

MARKETS=frozenset(("110001","110008"))

def slow_expanded(records,target,horizon):
    days=sorted(records);prices={d:r["price"] for d,r in records.items()}
    base,p0,audit=prior._rich_features(records,target,horizon,days,prices)
    if base is None:return None
    p0day=dt.date.fromisoformat(audit["p0Date"]);visible=[d for d in days if d<=p0day];extra=[]
    for w in (3,14,60):
        cutoff=p0day-dt.timedelta(days=w);ds=[d for d in visible if cutoff-dt.timedelta(days=w)<=d<=cutoff and prices[d]>0]
        d=max(ds) if ds else None;v=prices.get(d) if d else None
        extra += [math.log(p0/v) if v else 0.,float(v is not None),float((p0day-d).days if d else 0)]
    for w in (14,60):
        ds=[d for d in visible if p0day-dt.timedelta(days=w-1)<=d<=p0day and prices[d]>0]
        logs=[math.log(prices[d]) for d in ds];has=len(ds)>=3
        extra += [(logs[-1]-logs[0])/(ds[-1]-ds[0]).days if has and ds[-1]!=ds[0] else 0.,float(len(ds)),float(has)]
    def prior_values(w,field):
        return [float(records[d][field]) for d in visible if p0day-dt.timedelta(days=w)<=d<p0day and records[d].get(field) is not None and float(records[d][field])>0]
    for w in (14,60):
        vals=prior_values(w,"volume");med=statistics.median(vals) if vals else None;cur=records[p0day].get("volume");present=cur is not None and med is not None
        extra += [math.log(float(cur)/med) if present else 0.,float(len(vals)),float(present)]
    for w in (7,90):
        vals=prior_values(w,"rowCount");med=statistics.median(vals) if vals else None;cur=records[p0day].get("rowCount");present=cur is not None and med is not None
        extra += [math.log(float(cur)/med) if present else 0.,float(len(vals)),float(present)]
    for w in (7,90):
        obs=[records[d] for d in visible if p0day-dt.timedelta(days=w)<=d<p0day]
        dual=sum(MARKETS.issubset(x.get("markets",())) for x in obs)
        extra += [dual/len(obs) if obs else 0.,float(len(obs))]
    return base+extra


def records(start=dt.date(2025,1,1),end=dt.date(2026,4,4)):
    out={};day=start;i=0
    while day<=end:
        out[day]={"price":100.+i*.1,"volume":500.+i%17,"rowCount":30.+i%9,
                  "markets":frozenset(("110001","110008")) if i%3 else frozenset(("110001",))}
        day+=dt.timedelta(days=1);i+=1
    return out


class FeatureLibraryTests(unittest.TestCase):
    def test_schema_names_and_length_are_frozen(self):
        self.assertEqual(len(FEATURE_NAMES),71)
        self.assertEqual(FEATURE_NAMES[:14],("price_logratio_p1","price_present_p1","price_gap_p1","price_logratio_p7","price_present_p7","price_gap_p7","price_logratio_p30","price_present_p30","price_gap_p30","p0_age_days","target_weekday_sin","target_weekday_cos","target_annual_sin","target_annual_cos"))
        self.assertEqual(FEATURE_NAMES[14:26],("price_logratio_mean_7d","price_log_std_7d","price_log_range_7d","price_count_7d","price_logratio_mean_30d","price_log_std_30d","price_log_range_30d","price_count_30d","price_logratio_mean_90d","price_log_std_90d","price_log_range_90d","price_count_90d"))
        self.assertEqual(FEATURE_NAMES[-4:],("dual_market_share_7d","market_observation_count_7d","dual_market_share_90d","market_observation_count_90d"))
        rec=records();v,p0,a=features(rec,dt.date(2026,4,3),1)
        self.assertEqual(len(v),71);self.assertEqual(a["featureNames"],list(FEATURE_NAMES));self.assertEqual(p0,rec[dt.date(2026,4,1)]["price"])
        ordered=sorted(rec);pv={d:r["price"] for d,r in rec.items()}
        base,_,_=prior._rich_features(rec,dt.date(2026,4,3),1,ordered,pv)
        self.assertEqual(v[:40],base)
        self.assertEqual(v[14:26],base[14:26])

    def test_records_after_origin_minus_one_do_not_change_any_feature(self):
        rec=records();target=dt.date(2026,4,3);v,p0,_=features(rec,target,1)
        changed={d:dict(row) for d,row in rec.items()}
        for d,row in changed.items():
            if d>dt.date(2026,4,1):
                row["price"]*=1000;row["volume"]*=1000;row["rowCount"]*=1000;row["markets"]=frozenset()
        v2,p02,_=features(changed,target,1)
        self.assertEqual((v,p0),(v2,p02))

    def test_missing_volume_rowcount_and_market_windows_have_explicit_zero_masks(self):
        rec=records(end=dt.date(2026,4,4));p0day=dt.date(2026,4,1)
        for d,row in rec.items():
            if d<p0day:row["volume"]=None;row["rowCount"]=None;row["markets"]=frozenset()
        rec[p0day]["volume"]=None;rec[p0day]["rowCount"]=None
        v,_,_=features(rec,dt.date(2026,4,3),1)
        self.assertEqual(v[55:61],[0.,0.,0.,0.,0.,0.])
        self.assertEqual(v[61:67],[0.,0.,0.,0.,0.,0.])
        self.assertEqual(v[67:71],[0.,7.,0.,90.])

    def test_slope_window_excludes_p0_minus_w_but_includes_p0_minus_w_plus_one(self):
        rec=records();p0day=dt.date(2026,4,1);target=dt.date(2026,4,3)
        base,_,_=features(rec,target,1)
        excluded={d:dict(row) for d,row in rec.items()}
        excluded[p0day-dt.timedelta(days=14)]["price"]*=1000
        out,_,_=features(excluded,target,1)
        self.assertEqual(out[49],base[49])  # 14-day slope starts at vector index 49.
        included={d:dict(row) for d,row in rec.items()}
        included[p0day-dt.timedelta(days=13)]["price"]*=1000
        changed,_,_=features(included,target,1)
        self.assertNotEqual(changed[49],base[49])

    def test_bisect_implementation_matches_slow_reference_on_sparse_missing_cases(self):
        full=records();sparse={d:dict(r) for d,r in full.items() if d.day%3!=0 or d in (dt.date(2026,4,1),dt.date(2026,3,25))}
        for i,d in enumerate(sorted(sparse)):
            if i%7==0:sparse[d]["volume"]=None
            if i%11==0:sparse[d]["rowCount"]=None
            if i%5==0:sparse[d]["markets"]=frozenset()
        for target in (dt.date(2026,3,28),dt.date(2026,4,3),dt.date(2026,4,4)):
            got,_,_=features(sparse,target,1)
            want=slow_expanded(sparse,target,1)
            self.assertEqual(got,want)


if __name__=="__main__":unittest.main()
