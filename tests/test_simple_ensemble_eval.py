import datetime as dt
import unittest

from agri_data import simple_ensemble_eval as ev


def _reports(n=30):
    start = dt.date(2025, 9, 25)
    direct, hist = [], []
    for i in range(n):
        target = (start + dt.timedelta(days=i)).isoformat()
        prod = ev.PRODUCTS[0]; h = 1
        common = {"product": prod, "target": target, "horizon": h, "actual": 100.0,
                  "direct": 110.0, "persistence": 120.0}
        direct.append(dict(common))
        hist.append({"product":prod,"target":target,"horizon":h,"origin":(dt.date.fromisoformat(target)-dt.timedelta(days=1)).isoformat(),
                     "actual":100.0,"directRidge":110.0,"persistence":120.0,"fullHistoryHistGb":90.0})
    # validate_reports is also used by analyze; tests shrink the frozen fixture count.
    return {"pairedRows":direct}, {"pairedRows":hist}


class SimpleEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.old_shared, self.old_eval = ev.EXPECTED_SHARED, ev.EXPECTED_EVALUATION
        ev.EXPECTED_SHARED = ev.EXPECTED_EVALUATION = 30

    def tearDown(self):
        ev.EXPECTED_SHARED, ev.EXPECTED_EVALUATION = self.old_shared, self.old_eval

    def test_validates_common_keys_identity_and_origin(self):
        d, h = _reports()
        a, b = ev.validate_reports(d, h)
        self.assertEqual(len(a), 30); self.assertEqual(len(b), 30)
        h["pairedRows"][0]["actual"] = 101
        with self.assertRaisesRegex(ValueError, "mismatch"):
            ev.validate_reports(d, h)

    def test_duplicate_key_is_rejected(self):
        d, h = _reports()
        d["pairedRows"].append(dict(d["pairedRows"][0]))
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            ev.validate_reports(d, h)

    def test_candidate_and_fallback_metrics_and_family_shape(self):
        d, h = _reports()
        h["pairedRows"][0]["fullHistoryHistGb"] = None
        out = ev.analyze(d, h, {"direct":"a", "histgb":"b"}, now="fixed")
        self.assertEqual(out["coverage"]["n"], 30)
        self.assertEqual(out["coverage"]["fallbackCount"], 1)
        self.assertEqual(out["pairedRows"][1]["candidate"], 100)
        self.assertEqual(len(out["multipleTesting"]["HAC9"]), 300)
        self.assertEqual(len(out["multipleTesting"]["HAC28"]), 300)
        self.assertEqual(out["cells"][0]["n"], 30)
        macro = out["macro"]
        self.assertEqual(macro["fiveProductEqualWeight"]["candidate"]["availableProducts"], 1)
        self.assertAlmostEqual(macro["fiveProductEqualWeight"]["candidate"]["productEqualWeightMape"], 1/3)
        self.assertEqual(macro["fiveProductEqualWeight"]["full_history_histgb"]["availableProducts"], 1)
        self.assertEqual(macro["byProduct"][ev.PRODUCTS[0]]["candidate"]["availableHorizons"], 1)
        self.assertEqual(macro["byProduct"][ev.PRODUCTS[1]]["candidate"]["mape"], None)
        self.assertEqual(out["cells"][0]["conclusionHAC9"], "inconclusive")

    def test_full_protocol_is_embedded_before_execution(self):
        protocol = ev.FROZEN_PROTOCOL
        self.assertEqual(protocol["expectedEvaluationPairs"], 12_508)
        self.assertEqual(protocol["dm"]["primaryCalendarBartlettHacLagDays"], 9)
        self.assertEqual(protocol["dm"]["sensitivityCalendarBartlettHacLagDays"], 28)
        self.assertEqual(protocol["dm"]["holmFamilySizePerBandwidth"], 300)
        self.assertIn("All six", protocol["jointCellImprovement"])

    def test_directional_conclusion_uses_sign_and_holm_significance(self):
        t = {"status":"available", "meanLossDifference":-1, "holmSignificantAt005":True}
        self.assertEqual(ev._direction(t), "improved")
        t["meanLossDifference"] = 1
        self.assertEqual(ev._direction(t), "worsened")
        t["holmSignificantAt005"] = False
        self.assertEqual(ev._direction(t), "inconclusive")

    def test_invalid_origin_is_rejected(self):
        d, h = _reports()
        h["pairedRows"][0]["origin"] = "2025-09-20"
        with self.assertRaisesRegex(ValueError, "origin/horizon"):
            ev.validate_reports(d, h)


if __name__ == "__main__":
    unittest.main()
