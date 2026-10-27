import datetime as dt
import hashlib
import unittest
from unittest import mock

from agri_data import full_history_hyperparam_score as ev


def small_manifest(*, tuned_offset=1., default_offset=4., fixed_offset=7.):
    confirmation=[]; evaluation=[]; observations=[]
    for year, start, rows in ((2025,dt.date(2025,1,11),confirmation),(2026,dt.date(2026,1,11),evaluation)):
        for i in range(30):
            day=start+dt.timedelta(days=i); actual=100.+i
            models={}
            for arm in ev.ARMS:
                offset=(tuned_offset if arm=="ridgeTuned" else default_offset if arm=="ridgeDefault" else 2. if arm.endswith("Tuned") else 3.)
                models[arm]={"status":"issued","reason":None,"prediction":actual+offset,"training":{"count":100,"firstTarget":"2014-01-01","lastTarget":"2025-12-31"}}
            row={"product":"cabbage","target":day.isoformat(),"origin":(day-dt.timedelta(days=1)).isoformat(),"horizon":1,"eligible":True,"models":models}
            if year==2026:row["fixed50"]={"status":"issued","reason":None,"prediction":actual+fixed_offset}
            rows.append(row)
            observations.append({"date":day.isoformat(),"actual":actual})
    doc={"schemaVersion":1,"phase":"forecast_only","targetValuesAttached":False,"selection":{"cabbage":{"ridge":{"chosenConfig":{"id":"ridge_a1"}}}},"confirmationForecasts":confirmation,"evaluationForecasts":evaluation}
    doc["manifestSha256"]=hashlib.sha256(ev._canonical(doc)).hexdigest()
    prices={"schemaVersion":1,"products":[{"id":"cabbage","observations":observations}]}
    return doc,prices

class HyperparamScoreTests(unittest.TestCase):
    def _patched(self):
        return [mock.patch.object(ev,"PRODUCTS",("cabbage",)),mock.patch.object(ev.price_helper,"PRODUCTS",("cabbage",)),mock.patch.object(ev,"HORIZONS",(1,)),
                mock.patch.object(ev,"CONFIRM_START",dt.date(2025,1,11)),mock.patch.object(ev,"CONFIRM_END",dt.date(2025,2,9)),
                mock.patch.object(ev,"EVAL_START",dt.date(2026,1,11)),mock.patch.object(ev,"EVAL_END",dt.date(2026,2,9)),
                mock.patch.object(ev,"FAMILIES",(("ridge","ridgeTuned","ridgeDefault"),)),
                mock.patch.object(ev,"CONFIRM_COMPARISONS",(("ridge_tuned_vs_default","ridgeTuned","ridgeDefault"),)),
                mock.patch.object(ev,"EVAL_COMPARISONS",(("ridge_tuned_vs_default","ridgeTuned","ridgeDefault"),("ridge_tuned_vs_fixed50","ridgeTuned","fixed50"),("ridge_default_vs_fixed50","ridgeDefault","fixed50")))]
    def test_replay_gate_and_common_pair_metrics_families_and_coverage(self):
        manifest,prices=small_manifest()
        with self.assertRaisesRegex(ValueError,"replay"):
            ev.score(manifest,prices)
        patches=self._patched()
        for p in patches:p.start()
        self.addCleanup(lambda:[p.stop() for p in reversed(patches)])
        result=ev.score(manifest,prices,replay_verified=True)
        self.assertEqual(len(result["confirmation2025"]["commonPairs"]),30)
        self.assertEqual(len(result["evaluation2026"]["commonPairs"]),30)
        self.assertEqual(len(result["confirmation2025"]["dm"]),4)
        self.assertEqual(len(result["evaluation2026"]["dm"]),12)
        self.assertEqual(result["confirmation2025"]["dmSummaryByBandwidth"][0]["familySize"],2)
        self.assertEqual(result["evaluation2026"]["dmSummaryByBandwidth"][1]["familySize"],6)
        cell=next(r for r in result["evaluation2026"]["metrics"] if r["comparison"]=="ridge_tuned_vs_fixed50")
        self.assertEqual((cell["a"]["n"],cell["b"]["n"]),(30,30))
        self.assertEqual(cell["a"]["mae"],1.);self.assertEqual(cell["b"]["mae"],7.)
        self.assertEqual(result["evaluation2026"]["coverage"]["cabbage/1"]["commonPairs"],30)
        self.assertIn("cabbage/month/2026-01",result["evaluation2026"]["temporalCoverage"])
        self.assertIn("cabbage/quarter/2025-Q1",result["confirmation2025"]["temporalCoverage"])
        gate=result["candidateGates"][0]
        self.assertTrue(gate["confirmationAll10DistancesAtLeast30Pairs"])
        self.assertEqual(gate["decision"],"retain_existing_method")
    def test_status_hash_and_inference_unavailable_reasons(self):
        manifest,prices=small_manifest()
        bad=dict(manifest);bad["manifestSha256"]="0"*64
        with self.assertRaisesRegex(ValueError,"hash"):
            ev._check_manifest(bad)
        row=ev._dm({dt.date(2026,1,1)+dt.timedelta(days=i):1. for i in range(30)},9)
        self.assertEqual(row["status"],"zero_spread");self.assertEqual(row["pValue"],1.)
        row=ev._dm({dt.date(2026,1,1)+dt.timedelta(days=i):float(i) for i in range(12)},28)
        self.assertEqual(row["status"],"insufficient_pairs");self.assertIn("fewer_than_30",row["reason"])
    def test_nonpositive_or_missing_predictions_leave_common_population(self):
        manifest,prices=small_manifest()
        manifest["confirmationForecasts"][0]["models"]["ridgeTuned"]={"status":"not_issued","reason":"synthetic_missing","prediction":None}
        unsigned=dict(manifest);unsigned.pop("manifestSha256");manifest["manifestSha256"]=hashlib.sha256(ev._canonical(unsigned)).hexdigest()
        with mock.patch.object(ev,"PRODUCTS",("cabbage",)),mock.patch.object(ev.price_helper,"PRODUCTS",("cabbage",)),mock.patch.object(ev,"HORIZONS",(1,)),mock.patch.object(ev,"CONFIRM_START",dt.date(2025,1,11)),mock.patch.object(ev,"CONFIRM_END",dt.date(2025,2,9)),mock.patch.object(ev,"EVAL_START",dt.date(2026,1,11)),mock.patch.object(ev,"EVAL_END",dt.date(2026,2,9)),mock.patch.object(ev,"FAMILIES",(("ridge","ridgeTuned","ridgeDefault"),)),mock.patch.object(ev,"CONFIRM_COMPARISONS",(("ridge_tuned_vs_default","ridgeTuned","ridgeDefault"),)),mock.patch.object(ev,"EVAL_COMPARISONS",(("ridge_tuned_vs_default","ridgeTuned","ridgeDefault"),("ridge_tuned_vs_fixed50","ridgeTuned","fixed50"),("ridge_default_vs_fixed50","ridgeDefault","fixed50"))):
            result=ev.score(manifest,prices,replay_verified=True)
        self.assertEqual(len(result["confirmation2025"]["commonPairs"]),29)
        self.assertEqual(result["confirmation2025"]["coverage"]["cabbage/1"]["ridgeTunedIssued"],29)
        self.assertEqual(result["confirmation2025"]["coverage"]["cabbage/1"]["ridgeTunedVsridgeDefaultCoverageLossTargets"],["2025-01-11"])
        self.assertFalse(result["candidateGates"][0]["confirmationCoverageNondeclining"])

if __name__=="__main__":unittest.main()
