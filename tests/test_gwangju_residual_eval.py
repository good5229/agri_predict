import datetime as dt
import unittest

from agri_data import dashboard
from agri_data.gwangju_residual_eval import _combine_directions, _feature, _test_direction, build


class GwangjuResidualFeatureTests(unittest.TestCase):
    def test_significant_positive_loss_difference_is_reported_as_worsening(self):
        adverse = {"status": "tested", "holmSignificant": True, "meanLossDifference": 2.5}
        favorable = {"status": "tested", "holmSignificant": True, "meanLossDifference": -1.0}
        self.assertEqual(_test_direction(adverse), "worsening")
        self.assertEqual(_test_direction(favorable), "improvement")
        self.assertEqual(_combine_directions(["worsening", "improvement"]), "mixed_evidence")
        self.assertEqual(_combine_directions(["worsening", "inconclusive"]), "worsening")
        self.assertEqual(_combine_directions(["inconclusive", "inference_unavailable"]), "inference_unavailable")

    def test_features_use_only_origin_or_earlier_and_fixed_lag_cutoff(self):
        origin = dt.date(2025, 1, 20)
        gw = {
            dt.date(2025, 1, 13): {"price": 20.0, "weight": 40.0},
            dt.date(2025, 1, 19): {"price": 30.0, "weight": 50.0},
            dt.date(2025, 1, 21): {"price": 999.0, "weight": 999.0},
        }
        core = {dt.date(2025, 1, 19): (10.0, 100.0), dt.date(2025, 1, 21): (888.0, 888.0)}
        vec, audit = _feature(origin, gw, core)
        self.assertEqual(audit["status"], "available")
        self.assertEqual(audit["gwangjuDate"], "2025-01-19")
        self.assertEqual(audit["coreDate"], "2025-01-19")
        self.assertEqual(audit["lagDate"], "2025-01-13")
        self.assertAlmostEqual(vec[0], __import__("math").log(3))
        self.assertAlmostEqual(vec[3], __import__("math").log(1.5))
        self.assertEqual(vec[4], 1.0)

    def test_stale_gwangju_source_causes_missing_features(self):
        origin = dt.date(2025, 1, 20)
        vec, audit = _feature(origin,
            {dt.date(2025, 1, 1): {"price": 10, "weight": 20}},
            {dt.date(2025, 1, 19): (11, 30)})
        self.assertIsNone(vec)
        self.assertEqual(audit["status"], "source_missing_or_stale")


class GwangjuResidualBuildTests(unittest.TestCase):
    def test_training_cutoff_and_eval_fallback_are_explicit(self):
        product = "cabbage"
        rows, observation_map, gw = [], {}, {product: {}}
        start = dt.date(2023, 1, 1)
        observation_map[start-dt.timedelta(days=1)] = {"date": (start-dt.timedelta(days=1)).isoformat(), "actual": 90, "volumeKg": 900}
        gw[product][start-dt.timedelta(days=1)] = {"price": 80, "weight": 500}
        # 100 distinct training targets. Gwangju/core source dates precede every origin.
        for i in range(100):
            target = start + dt.timedelta(days=i)
            origin = target - dt.timedelta(days=1)
            observation_map[target] = {"date": target.isoformat(), "actual": 100 + i, "volumeKg": 1000}
            gw[product][origin] = {"price": 80 + i, "weight": 500 + i}
            rows.append({"product": product, "horizon": 1, "target": target.isoformat(), "origin": origin.isoformat(),
                         "actual": 100 + i, "direct": 95 + i, "persistence": 90 if i == 0 else 99 + i})
        # A target in the fixed evaluation range has no valid Gwangju feature and must fall back.
        target = dt.date(2025, 10, 1); origin = target - dt.timedelta(days=1)
        observation_map[origin] = {"date": origin.isoformat(), "actual": 200, "volumeKg": 1000}
        observation_map[target] = {"date": target.isoformat(), "actual": 220, "volumeKg": 1100}
        rows.append({"product": product, "horizon": 1, "target": target.isoformat(), "origin": origin.isoformat(),
                     "actual": 220, "direct": 210, "persistence": 200})
        # Direct validator requires a complete five-product dashboard schema.
        observations = list(observation_map.values())
        price_products = [{"id": p, "observations": observations if p == product else []} for p in dashboard.PRODUCTS]
        report = {"protocol": {"id": "direct-horizon-ridge-2026-10-06"}, "pairedRows": rows}
        result = build({"schemaVersion": 1, "products": price_products}, report, gw)
        audit = result["fitAudits"]["cabbage/1"]
        self.assertEqual(audit["distinctTrainingTargets"], 100)
        self.assertEqual(audit["status"], "fitted")
        self.assertEqual(len(result["predictions"]), 1)
        self.assertEqual(result["predictions"][0]["candidate"], 210)
        self.assertIsNotNone(result["predictions"][0]["fallbackReason"])
        self.assertEqual(len(result["comparisons"]), 200)
        self.assertFalse(result["goalGate"]["goalMet"])


if __name__ == "__main__":
    unittest.main()
