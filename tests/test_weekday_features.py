import datetime as dt
import math
import unittest
from unittest import mock
from agri_data import dashboard, weekday_features as wf

class WeekdayFeaturesTests(unittest.TestCase):
    def test_56_day_window_is_strictly_before_origin_and_same_target_weekday(self):
        origin=dt.date(2024,5,20); target=origin+dt.timedelta(days=7)
        series={d:(100+(origin-d).days,10) for d in (origin-dt.timedelta(days=i) for i in range(56, -2, -1))}
        # The lower bound is included; origin is excluded from weekday history and included as the latest valid source.
        got=wf.weekday_features(series,origin,target)
        expected=sorted(math.log(v[0]) for d,v in series.items()
                        if origin-dt.timedelta(days=56)<=d<origin and d.weekday()==target.weekday())
        self.assertEqual(got[1],8.0)
        mid=len(expected)//2
        median=expected[mid] if len(expected)%2 else (expected[mid-1]+expected[mid])/2
        self.assertAlmostEqual(got[0],median)
        self.assertEqual(got[2],7.0)
        self.assertAlmostEqual(got[3],math.log(series[origin][0])-median)

    def test_empty_weekday_history_has_explicit_count_and_sentinel(self):
        origin=dt.date(2024,1,8); target=origin+dt.timedelta(days=1)
        series={origin-dt.timedelta(days=1):(125.0,1)}
        self.assertEqual(wf.weekday_features(series,origin,target),(0.0,0.0,57.0,0.0))

    def test_hist_gb_comparator_keeps_every_frozen_horizon(self):
        rows=[{"target":"2024-01-20","origin":(dt.date(2024,1,20)-dt.timedelta(days=h)).isoformat(),"horizon":h,"predicted":100+h} for h in wf.HORIZONS]
        raw={"schemaVersion":2,"products":[{"id":p,"runs":[{"modelId":"hist_gb","cadence":"monthly","predictions":rows if p=="cabbage" else []}]} for p in dashboard.PRODUCTS]}
        values=wf._monthly_hist_gb_all_horizons(raw)
        self.assertEqual(sorted(h for _,h in values["cabbage"]),list(wf.HORIZONS))

    def test_candidate_coverage_loss_uses_raw_prediction_dates(self):
        self.assertEqual(wf._coverage_loss(["a","b"],["a","c"]),["b"])

    def test_macro_is_null_unless_all_50_groups_have_paired_rows(self):
        values,counts=wf._macro_metrics([])
        self.assertEqual(counts["candidate"],0)
        self.assertIsNone(values["candidate"])

    def test_joint_gate_requires_all_six_tests_metrics_and_coverage(self):
        summaries=[]; coverage={}
        for p in wf.PRODUCTS:
          for h in wf.HORIZONS:
            key=f"{p}/{h}";coverage[key]={"candidateCoverageLossDates":{c:[] for c in wf.COMPARATORS}}
            for c in wf.COMPARATORS:
              for loss in wf.LOSSES:
                summaries.append({"product":p,"horizon":h,"comparator":c,"loss":loss,"candidateMetric":1.,"comparatorMetric":2.,"dm":{"9":{"status":"tested","n":100,"holmSignificant":True,"meanLossDifference":-1.}}})
        outcome=wf._joint_conclusions(summaries,coverage)[0]
        self.assertEqual(outcome["conclusion"],"improvement")
        summaries[0]["dm"]["9"]["holmSignificant"]=False
        self.assertEqual(wf._joint_conclusions(summaries,coverage)[0]["conclusion"],"inconclusive")
        summaries[0]["dm"]["9"].update(holmSignificant=True,meanLossDifference=1.)
        self.assertEqual(wf._joint_conclusions(summaries,coverage)[0]["conclusion"],"worsening_or_mixed")
        summaries[0]["dm"]["9"].update(meanLossDifference=-1.)
        coverage["cabbage/1"]["candidateCoverageLossDates"][wf.COMPARATORS[0]]=["2024-01-01"]
        self.assertEqual(wf._joint_conclusions(summaries,coverage)[0]["conclusion"],"inconclusive")

    def test_metrics_uses_common_pairs_and_positive_price_denominator(self):
        m=wf._metrics([100,200],[110,180])
        self.assertEqual(m['mae'],15)
        self.assertAlmostEqual(m['mape'],10)

    def test_failed_month_fit_emits_no_partial_predictions(self):
        class Fails:
            def fit(self,x,y): raise RuntimeError('synthetic fit failure')
        start=dt.date(2022,1,1); end=dt.date(2023,1,6)
        observations=[]
        for i in range((end-start).days+1):
            d=start+dt.timedelta(days=i); observations.append({'date':d.isoformat(),'actual':100+i/20,'volumeKg':100})
        prices={'schemaVersion':1,'products':[{'id':p,'observations':[r for r in observations if p!='cabbage' or not ('2022-12-01' <= r['date'] <= '2023-01-01')]} for p in dashboard.PRODUCTS]}
        pred_products=[]; ridge_rows=[]
        for p in dashboard.PRODUCTS:
            runs=[{'modelId':'hist_gb','cadence':'monthly','predictions':[]}]
            pers=[]
            for i in range((end-start).days+1):
                d=start+dt.timedelta(days=i)
                for h in wf.HORIZONS:
                    o=d-dt.timedelta(days=h)
                    if o<dt.date(2023,1,1): continue
                    row={'target':d.isoformat(),'origin':o.isoformat(),'horizon':h,'predicted':100.0}
                    pers.append(row)
                    if p=='cabbage' and d.year==2023 and d.month==1:runs[0]['predictions'].append(row)
            pred_products.append({'id':p,'runs':runs,'predictions':pers})
        # Historical direct Ridge comparator schema.
        for p in dashboard.PRODUCTS:
            for d in (dt.date(2023,1,2),dt.date(2023,1,3),dt.date(2023,1,4),dt.date(2023,1,5),dt.date(2023,1,6)):
                h=1;ridge_rows.append({'product':p,'horizon':h,'target':d.isoformat(),'direct':100.0})
        report={'protocol':{'id':'direct-horizon-ridge-2026-10-06'},'pairedRows':ridge_rows}
        with mock.patch.object(wf,'EVAL_START',dt.date(2023,1,1)),mock.patch.object(wf,'EVAL_END',end),mock.patch.object(wf,'MIN_TRAIN_DAYS',90),mock.patch.object(wf,'_model',return_value=Fails()):
            result=wf.build(prices,{'schemaVersion':2,'products':pred_products},report)
        self.assertFalse(any(r['product']=='cabbage' and r['horizon']==1 and r['target'].startswith('2023-01') for r in result['predictions']))
        self.assertEqual(result['fitAudits']['cabbage/1']['months'][-1]['status'],'fit_failed')
        self.assertEqual(result['fitTotals']['fitSuccesses'],0)
        attempted=result['fitAudits']['cabbage/1']['months'][-1]
        self.assertGreater(attempted['fitDistinctTargets'],90)
        self.assertLess(attempted['fitTrainTargetEnd'],attempted['fitOrigin'])
        cov=result['coverage']['cabbage/1']
        self.assertEqual(cov['eligibleDateCount'],5)
        self.assertEqual(cov['commonPairedCount'],0)
        self.assertEqual(len(cov['exclusionReasons']),5)
        self.assertEqual(len(cov['candidateCoverageLossDates']['persistence']),5)
        self.assertIn('source_older_than_max_age',cov['exclusionReasons']['2023-01-02'])

if __name__=='__main__': unittest.main()
