import datetime as dt
import unittest
from unittest import mock

from agri_data import full_history_return_eval as ev


def prices(start=dt.date(2014, 1, 1), end=dt.date(2026, 1, 20)):
    products=[]
    n=(end-start).days+1
    for j,p in enumerate(ev.PRODUCTS):
        obs=[]
        for i in range(n):
            d=start+dt.timedelta(days=i)
            # Smooth deterministic positive prices; no dependency on labels at issue time.
            obs.append({"date":d.isoformat(),"actual":100+j*10+i*.01+((i%11)-5)*.1})
        products.append({"id":p,"observations":obs})
    return {"schemaVersion":1,"products":products}


def fixed50():
    return {"phase":"forecast_only","targetValuesAttached":False,"forecasts":[]}


class FullHistoryReturnTests(unittest.TestCase):
    def test_relative_features_use_only_prior_observations_and_keep_missing_masks(self):
        target=dt.date(2024, 2, 1)
        series={dt.date(2024,1,1):100.,dt.date(2024,1,25):110.,dt.date(2024,1,31):121.,dt.date(2024,2,1):999.}
        f,p0,a=ev._features(series,target,1)
        self.assertAlmostEqual(p0,110.)  # origin Jan 31, information through Jan 30
        # Make any post-cutoff values extreme: the feature vector must remain identical.
        changed={**series,dt.date(2024,1,31):99999.,dt.date(2024,2,1):88888.}
        f2,p02,_=ev._features(changed,target,1)
        self.assertEqual((f,p0),(f2,p02))
        sparse={dt.date(2024,1,30):100.,dt.date(2024,2,1):500.}
        fs,_,_=ev._features(sparse,target,1)
        self.assertEqual(fs[0:3],[0.0,0.0,0.0])

    def test_build_includes_2014_training_and_blocks_early_january_origins(self):
        raw=prices(end=dt.date(2026,1,12))
        with mock.patch.object(ev,"EVAL_END",dt.date(2026,1,12)):
            result=ev.build(raw,fixed50())
        self.assertEqual(result["plannedForecastCount"],5*12*10)
        r=next(x for x in result["forecasts"] if x["product"]=="cabbage" and x["horizon"]==1 and x["target"]=="2026-01-01")
        self.assertTrue(r["training"]["firstTarget"].startswith("2014-"))
        self.assertEqual(r["training"]["lastTarget"],"2025-12-31")
        self.assertEqual(r["status"],"not_issued")
        self.assertEqual(r["reason"],"model_not_yet_fitted_at_origin")
        h10=next(x for x in result["forecasts"] if x["product"]=="cabbage" and x["horizon"]==10 and x["target"]=="2026-01-10")
        self.assertEqual(h10["status"],"not_issued")
        self.assertEqual(h10["reason"],"model_not_yet_fitted_at_origin")
        later=next(x for x in result["forecasts"] if x["product"]=="cabbage" and x["horizon"]==1 and x["target"]=="2026-01-12")
        self.assertEqual(later["status"],"issued")
        self.assertLessEqual(dt.date.fromisoformat(later["origin"]),ev.EVAL_END)

    def test_scoring_uses_only_same_row_positive_actual_pairs(self):
        actual=prices(end=dt.date(2026,1,20))
        for p in actual["products"]:
            p["observations"]=[{"date":"2026-01-01","actual":0.},{"date":"2026-01-02","actual":100.}]
        with mock.patch.object(ev,"EVAL_END",dt.date(2026,1,2)):
            manifest=ev.build(prices(end=dt.date(2026,1,20)),fixed50())
            scored=ev.score(manifest,actual)
        self.assertEqual(len(scored["scoredPairs"]),5*10)
        self.assertEqual(scored["coverage"]["cabbage/1"]["persistenceCommon"],1)
        self.assertEqual(scored["coverage"]["cabbage/1"]["fixed50Common"],0)
        self.assertEqual(scored["scoredPairs"][0]["actual"],100.)


    def test_calendar_coverage_includes_missing_actual_days_and_gates_full_planned_fixed50_keys(self):
        raw=prices(end=dt.date(2026,1,20))
        baseline={"phase":"forecast_only","targetValuesAttached":False,"forecasts":[
            {"product":"cabbage","target":"2026-01-01","horizon":1,"predictions":{"fixed50":100.}},
            {"product":"cabbage","target":"2026-01-02","horizon":1,"predictions":{"fixed50":100.}}]}
        actual=prices(end=dt.date(2026,1,20))
        for p in actual["products"]:
            p["observations"]=[{"date":"2026-01-02","actual":100.}]
        with mock.patch.object(ev,"EVAL_END",dt.date(2026,1,10)):
            manifest=ev.build(raw,baseline)
            result=ev.score(manifest,actual)
        month=result["temporalCoverage"]["cabbage/month/2026-01"]
        self.assertEqual(month["planned"],100)  # all 10 calendar days x 10 horizons, including no actuals
        self.assertEqual(month["positiveActualPairs"],10)
        self.assertEqual(result["coverage"]["cabbage/1"]["fixed50AvailableOnPlannedGrid"],2)
        self.assertEqual(result["coverage"]["cabbage/1"]["candidateCoverageLossTargets"],["2026-01-01"])
        gate=next(x for x in result["cropGates"] if x["product"]=="cabbage")
        self.assertFalse(gate["coverageNondeclining"])

    def test_partial_batch_overflow_clears_all_predictions_and_unissued_rows_never_score(self):
        class ExplodingEstimator:
            def fit(self, x, y): return self
            def predict(self, x): return [0.0 if i==0 else 1000.0 for i in range(len(x))]
        raw=prices(end=dt.date(2026,1,10))
        with mock.patch.object(ev,"EVAL_END",dt.date(2026,1,4)), mock.patch("sklearn.pipeline.make_pipeline",return_value=ExplodingEstimator()):
            manifest=ev.build(raw,fixed50())
            failed=[r for r in manifest["forecasts"] if r["product"]=="cabbage" and r["horizon"]==1 and r["target"] in ("2026-01-02","2026-01-03","2026-01-04")]
            self.assertTrue(all(r["status"]=="not_issued" and r["prediction"] is None for r in failed))
            scored=ev.score(manifest,raw)
        self.assertEqual(scored["coverage"]["cabbage/1"]["persistenceCommon"],0)
        self.assertTrue(all(r["candidate"] is None for r in scored["scoredPairs"] if r["product"]=="cabbage" and r["horizon"]==1))

    def test_dm_unavailable_reason_and_holm_keep_fixed_family(self):
        out=ev._dm({dt.date(2026,1,1):1.0},9)
        self.assertEqual(out["status"],"insufficient_pairs")
        self.assertEqual(out["reason"],"fewer_than_30_pairs")
        self.assertEqual(out["pValue"],1.0)
        family=[dict(out) for _ in range(5*10*2*2)]
        ev._holm(family)
        self.assertTrue(all(x["holmPValue"]==1.0 and not x["holmSignificant"] for x in family))
        for constant in (0.1,-0.1):
            repeated={dt.date(2026,1,1)+dt.timedelta(days=i):constant for i in range(40)}
            dm=ev._dm(repeated,9)
            self.assertEqual(dm["status"],"zero_spread")
            self.assertEqual(dm["pValue"],1.0)


if __name__=="__main__": unittest.main()
