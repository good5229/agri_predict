import datetime as dt
import math
import unittest

from agri_data import dashboard
from agri_data import static_blend as blend


def _row(origin, predicted, horizon=1):
    target = origin + dt.timedelta(days=horizon)
    return {"origin": origin.isoformat(), "target": target.isoformat(), "horizon": horizon, "predicted": predicted}


def _source():
    products = []
    for pid in dashboard.PRODUCTS:
        actuals, baseline, ridge, hist = [], [], [], []
        for i in range(3):
            target = dt.date(2024, 1, 2) + dt.timedelta(days=i)
            origin = target - dt.timedelta(days=1)
            actuals.append({"date": target.isoformat(), "actual": 100.0})
            baseline.append(_row(origin, 100.0))
            ridge.append(_row(origin, 99.0))
            hist.append(_row(origin, 101.0))
        products.append({"id": pid, "observations": actuals, "predictions": baseline,
                         "runs": [{"modelId": "ridge", "cadence": "daily", "predictions": ridge},
                                  {"modelId": "hist_gb", "cadence": "daily", "predictions": hist}]})
    return {"schemaVersion": 2, "products": products}


def _passing_source():
    products = []
    for pid in dashboard.PRODUCTS:
        actuals, baseline, ridge, hist = [], [], [], []
        day = dt.date(2017, 1, 1)
        end = dt.date(2024, 1, 3)
        while day <= end:
            actual = 100.0 + (day.toordinal() % 7)
            origin = day - dt.timedelta(days=1)
            actuals.append({"date": day.isoformat(), "actual": actual})
            baseline.append(_row(origin, actual + 5.0))
            ridge.append(_row(origin, actual + (0.1 if day.toordinal() % 2 else -0.1)))
            hist.append(_row(origin, actual + 10.0))
            day += dt.timedelta(days=1)
        products.append({"id": pid, "observations": actuals, "predictions": baseline,
                         "runs": [{"modelId": "ridge", "cadence": "daily", "predictions": ridge},
                                  {"modelId": "hist_gb", "cadence": "daily", "predictions": hist}]})
    return {"schemaVersion": 2, "products": products}


class StaticBlendTests(unittest.TestCase):
    def test_adjustment_shrinks_model_revision_toward_persistence(self):
        self.assertEqual(blend._adjust(100, 80, .25), 95)
        self.assertEqual(blend._adjust(100, 120, .5), 110)
        self.assertTrue(math.isnan(blend._adjust(10, -30, 1)))

    def test_fixed_families_and_baseline_reselection_are_preserved(self):
        report = blend.analyze(_source())
        self.assertEqual(len(report["results"]), 50)
        self.assertEqual(len(report["multipleTesting"]["HAC9"]), 200)
        self.assertEqual(len(report["multipleTesting"]["HAC28"]), 200)
        self.assertTrue(all(r["selection"]["selectedModel"] == "baseline" for r in report["results"]))
        self.assertTrue(all(t["status"] == "unavailable_baseline_reselected" for t in report["multipleTesting"]["HAC9"]))

    def test_holm_adjustment_is_monotone_and_family_size_aware(self):
        tests = [{"productId": "a", "horizon": 1, "comparator": "baseline", "loss": "ae", "status": "available", "pValue": .01},
                 {"productId": "b", "horizon": 1, "comparator": "parent", "loss": "ape", "status": "unavailable", "pValue": None}]
        blend._holm(tests)
        self.assertAlmostEqual(tests[0]["holmAdjustedPValue"], .02)
        self.assertEqual(tests[1]["holmAdjustedPValue"], 1)

    def test_conclusion_requires_both_losses_and_distinguishes_mixed_and_unavailable(self):
        tests = [{"status": "available", "meanLossDifference": -1, "holmSignificantAt005": True},
                 {"status": "available", "meanLossDifference": -2, "holmSignificantAt005": True}]
        self.assertEqual(blend._conclusion(tests), "improved")
        tests[1]["holmSignificantAt005"] = False
        self.assertEqual(blend._conclusion(tests), "mixed_evidence")
        tests[1]["status"] = "unavailable_nonpositive_variance"
        self.assertEqual(blend._conclusion(tests), "inference_unavailable")
        tests[1]["status"] = "available"; tests[0]["holmSignificantAt005"] = False
        self.assertEqual(blend._conclusion(tests), "no_evidence")

    def test_nominee_validation_origin_boundary_and_period_coverage(self):
        report = blend.analyze(_passing_source())
        row = next(r for r in report["results"] if r["productId"] == next(iter(dashboard.PRODUCTS)) and r["horizon"] == 1)
        self.assertEqual(row["selection"]["development"]["n"], 1826)
        self.assertTrue(row["selection"]["validation"]["kept"])
        self.assertEqual(row["selection"]["selectedModel"], "ridge")
        self.assertEqual(row["selection"]["weight"], 1.0)
        # 2024-01-01 is forecast from a pre-period origin and is excluded.
        self.assertEqual(row["coverage"]["eligibleEvaluationActualTargets"], 2)
        self.assertEqual(row["coverage"]["evaluationCommonForecastOrigins"], 2)
        self.assertEqual(row["coverage"]["evaluationCommonOriginsWithActual"], 2)
        self.assertEqual(row["coverage"]["evaluationExcludedEligibleTargets"], 0)
        self.assertEqual(row["coverage"]["evaluationN"], 2)
        self.assertEqual(row["coverage"]["evaluationOriginMinimum"], "2024-01-01")
        season = row["coverage"]["bySeason"]["winter"]["baseline"]
        self.assertEqual(season["n"], 2)
        self.assertEqual(season["selected"]["n"], 2)
        self.assertEqual(row["coverage"]["bySeason"]["winter"]["parent"]["comparator"]["n"], 2)
        self.assertIn("meanLossDifference", season)
        self.assertIn("direction", season)
        self.assertIn("mae", row["comparisons"]["baseline"]["meanLossDifference"])
        self.assertIn("mape", row["comparisons"]["parent"]["meanLossDifference"])
        self.assertIn(row["evaluationConclusion"]["HAC9"], {"improved", "worsened", "mixed_evidence", "no_evidence", "inference_unavailable"})


if __name__ == "__main__":
    unittest.main()
