import datetime as dt
import contextlib
import unittest
from unittest import mock

from agri_data import dashboard, direct_horizon


class DirectHorizonStatisticsTests(unittest.TestCase):
    def test_calendar_hac_does_not_compress_missing_dates(self):
        contiguous = {dt.date(2024, 1, 1) + dt.timedelta(days=i): 1.0 for i in range(40)}
        sparse = {dt.date(2024, 1, 1) + dt.timedelta(days=2 * i): 1.0 for i in range(40)}
        contiguous_result = direct_horizon._dm(contiguous, 9)
        sparse_result = direct_horizon._dm(sparse, 9)
        self.assertEqual(contiguous_result["status"], "zero_or_invalid_hac_variance")
        self.assertEqual(sparse_result["status"], "zero_or_invalid_hac_variance")
        # A nonconstant sequence distinguishes calendar gaps from row-index lags.
        values = [float(i % 7 - 3) for i in range(40)]
        contiguous = {dt.date(2024, 1, 1) + dt.timedelta(days=i): v for i, v in enumerate(values)}
        sparse = {dt.date(2024, 1, 1) + dt.timedelta(days=2 * i): v for i, v in enumerate(values)}
        self.assertNotEqual(direct_horizon._dm(contiguous, 9)["statistic"],
                            direct_horizon._dm(sparse, 9)["statistic"])

    def test_unavailable_dm_keeps_p_one_for_fixed_family(self):
        row = direct_horizon._dm({dt.date(2024, 1, 1): 1.0}, 9)
        self.assertEqual(row["pValue"], 1.0)
        family = [dict(row) for _ in range(direct_horizon.FAMILY_SIZE)]
        direct_horizon._holm(family)
        self.assertTrue(all(item["holmPValue"] == 1.0 and not item["holmSignificant"] for item in family))

    def test_monthly_and_persistence_maps_select_expected_runs(self):
        day = "2024-02-01"
        raw = {"schemaVersion": 2, "products": [
            {"id": "cabbage", "runs": [
                {"modelId": "ridge", "cadence": "monthly", "predictions": [
                    {"target": day, "origin": "2024-01-31", "horizon": 1, "predicted": 11.0}]},
                {"modelId": "ridge", "cadence": "daily", "predictions": [
                    {"target": day, "origin": "2024-01-31", "horizon": 1, "predicted": 12.0}]},
            ], "predictions": [{"target": day, "origin": "2024-01-31", "horizon": 1, "predicted": 10.0}]} if pid == "cabbage" else {"id": pid, "runs": [], "predictions": []}
            for pid in dashboard.PRODUCTS]}
        monthly, daily, persistence = direct_horizon._forecast_maps(raw)
        key = (dt.date(2024, 2, 1), 1)
        self.assertEqual(monthly["cabbage"][key], 11.0)
        self.assertEqual(daily["cabbage"][key], 12.0)
        self.assertEqual(persistence["cabbage"][key], 10.0)
        with self.assertRaisesRegex(ValueError, "schemaVersion 2"):
            direct_horizon._forecast_maps({**raw, "schemaVersion": 1})
        raw["products"][0]["predictions"][0]["origin"] = "2024-01-30"
        with self.assertRaisesRegex(ValueError, "origin mismatch"):
            direct_horizon._forecast_maps(raw)

    def test_synthetic_direct_fits_never_train_on_targets_after_fit_origin(self):
        class ZeroEstimator:
            def fit(self, x, y):
                return self

            def predict(self, x):
                return [0.0]

        first = dt.date(2014, 1, 1)
        last = dt.date(2015, 1, 10)
        products = []
        forecast_products = []
        for pid in dashboard.PRODUCTS:
            observations = []
            runs = [{"modelId": "ridge", "cadence": "monthly", "predictions": []}]
            baseline_predictions = []
            for i in range((last - first).days + 1):
                day = first + dt.timedelta(days=i)
                observations.append({"date": day.isoformat(), "actual": 100.0 + i / 10.0, "volumeKg": 1000.0})
                for horizon in direct_horizon.HORIZONS:
                    if day - dt.timedelta(days=horizon) < first:
                        continue
                    row = {"target": day.isoformat(), "origin": (day - dt.timedelta(days=horizon)).isoformat(),
                           "horizon": horizon, "predicted": 100.0 + i / 10.0}
                    runs[0]["predictions"].append(row)
                    baseline_predictions.append(row)
            products.append({"id": pid, "observations": observations})
            forecast_products.append({"id": pid, "runs": runs, "predictions": baseline_predictions})
        prices = {"schemaVersion": 1, "products": products}
        forecasts = {"schemaVersion": 2, "products": forecast_products}
        with mock.patch.object(direct_horizon, "EVAL_START", dt.date(2015, 1, 1)), \
             mock.patch.object(direct_horizon, "EVAL_END", dt.date(2015, 1, 10)), \
             mock.patch.object(direct_horizon, "MIN_TRAIN_DAYS", 90), \
             mock.patch.object(direct_horizon.modeling, "_new_estimator", return_value=ZeroEstimator()), \
             mock.patch("threadpoolctl.threadpool_limits", return_value=contextlib.nullcontext()):
            result = direct_horizon.build(prices, forecasts)
        self.assertTrue(result["predictions"])
        for row in result["predictions"]:
            self.assertLessEqual(row["fitTrainTargetEnd"], row["fitOrigin"])
            self.assertGreaterEqual(row["origin"], "2015-01-01")
        self.assertTrue(result["fitTotals"]["fitAttempts"] > 0)
        self.assertEqual(result["fitTotals"]["fitAttempts"], result["fitTotals"]["fitSuccesses"])
        self.assertGreater(result["comparisons"][0]["n"], 0)
        audit = result["coverage"]["cabbage/1"]
        self.assertEqual(audit["eligibleDateCount"], len(audit["eligibleEvaluationDates"]))
        self.assertEqual(audit["commonPairedDates"], sorted(audit["commonPairedDates"]))
        self.assertEqual(result["jointConclusions"][0]["conclusion"], "inference_unavailable")

        stale_prices = {"schemaVersion": 1, "products": [dict(p, observations=list(p["observations"]))
            for p in products]}
        cabbage = next(p for p in stale_prices["products"] if p["id"] == "cabbage")
        cabbage["observations"] = [row for row in cabbage["observations"]
            if not dt.date(2014, 12, 20) <= dt.date.fromisoformat(row["date"]) <= dt.date(2015, 1, 5)]
        with mock.patch.object(direct_horizon, "EVAL_START", dt.date(2015, 1, 1)), \
             mock.patch.object(direct_horizon, "EVAL_END", dt.date(2015, 1, 10)), \
             mock.patch.object(direct_horizon, "MIN_TRAIN_DAYS", 90), \
             mock.patch.object(direct_horizon.modeling, "_new_estimator", return_value=ZeroEstimator()), \
             mock.patch("threadpoolctl.threadpool_limits", return_value=contextlib.nullcontext()):
            stale_result = direct_horizon.build(stale_prices, forecasts)
        stale_audit = stale_result["coverage"]["cabbage/1"]
        self.assertIn("2015-01-06", stale_audit["eligibleEvaluationDates"])
        self.assertIn("2015-01-06", stale_audit["candidateCoverageLossDates"]["monthly_pooled_ridge"])
        self.assertIn("feature_unavailable_or_source_too_old", stale_audit["exclusionReasons"]["2015-01-06"])
        cabbage_joint = next(row for row in stale_result["jointConclusions"]
                             if row["product"] == "cabbage" and row["horizon"] == 1)
        self.assertFalse(cabbage_joint["coverageNondeclining"])

        empty_baseline = {"schemaVersion": 2, "products": [dict(p, predictions=[])
            for p in forecast_products]}
        with mock.patch.object(direct_horizon, "EVAL_START", dt.date(2015, 1, 1)), \
             mock.patch.object(direct_horizon, "EVAL_END", dt.date(2015, 1, 10)), \
             mock.patch.object(direct_horizon, "MIN_TRAIN_DAYS", 90), \
             mock.patch.object(direct_horizon.modeling, "_new_estimator", return_value=ZeroEstimator()), \
             mock.patch("threadpoolctl.threadpool_limits", return_value=contextlib.nullcontext()):
            with self.assertRaisesRegex(ValueError, "zero global coverage or common paired dates"):
                direct_horizon.build(prices, empty_baseline)

    def test_joint_conclusion_requires_all_four_hac9_tests_and_nondeclining_coverage(self):
        summaries = []
        coverage = {}
        for pid in dashboard.PRODUCTS:
            for horizon in direct_horizon.HORIZONS:
                key = f"{pid}/{horizon}"
                coverage[key] = {"candidateCoverageLossDates": {
                    comparator: [] for comparator in direct_horizon.COMPARATORS}}
                for comparator in direct_horizon.COMPARATORS:
                    for loss in direct_horizon.LOSSES:
                        summaries.append({"product": pid, "horizon": horizon, "comparator": comparator,
                            "loss": loss, "directMetric": 1.0, "comparatorMetric": 2.0,
                            "dm": {"9": {"status": "tested", "n": 100, "holmSignificant": True,
                                "meanLossDifference": -1.0}}})
        def outcome():
            return next(row["conclusion"] for row in direct_horizon._joint_conclusions(summaries, coverage)
                        if row["product"] == "cabbage" and row["horizon"] == 1)

        self.assertEqual(outcome(), "improvement")
        summaries[3]["dm"]["9"]["holmSignificant"] = False
        self.assertEqual(outcome(), "inconclusive")
        summaries[3]["dm"]["9"]["holmSignificant"] = True
        coverage["cabbage/1"]["candidateCoverageLossDates"]["monthly_pooled_ridge"] = ["2023-01-02"]
        self.assertEqual(outcome(), "inconclusive")
        coverage["cabbage/1"]["candidateCoverageLossDates"]["monthly_pooled_ridge"] = []
        summaries[3]["directMetric"] = 3.0
        self.assertEqual(outcome(), "worsening_or_mixed")


if __name__ == "__main__":
    unittest.main()
