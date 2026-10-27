import datetime as dt
import json
import unittest
from unittest import mock

from agri_data import dashboard, recent_window as rw


class RecentWindowTests(unittest.TestCase):
    def test_window_includes_lower_boundary_and_excludes_fit_origin(self):
        origin = dt.date(2026, 10, 6)
        rows = [(origin - dt.timedelta(days=n), n) for n in (731, 730, 400, 1, 0)]
        self.assertEqual([r[1] for r in rw._training_rows(rows, origin, recent=True)], [730, 400, 1])
        self.assertEqual([r[1] for r in rw._training_rows(rows, origin, recent=False)], [731, 730, 400, 1])

    def test_late_inverse_failure_publishes_no_partial_month_and_coverage_is_json_safe(self):
        start, end = dt.date(2020, 1, 1), dt.date(2023, 1, 15)
        observations = []
        for i in range((end-start).days + 1):
            day = start + dt.timedelta(days=i)
            observations.append({"date": day.isoformat(), "actual": 100 + i / 20, "volumeKg": 50.0})
        prices = {"schemaVersion": 1, "products": [{"id": p, "observations": observations} for p in dashboard.PRODUCTS]}
        eval_days = [dt.date(2023, 1, i) for i in range(11, 16)]
        pred_products, ridge_rows = [], []
        for product in dashboard.PRODUCTS:
            pooled_rows, persistence_rows = [], []
            for day in eval_days:
                for horizon in rw.HORIZONS:
                    origin = day-dt.timedelta(days=horizon)
                    actual = 100 + (day-start).days / 20
                    row = {"target": day.isoformat(), "origin": origin.isoformat(), "horizon": horizon, "predicted": 100.0}
                    persistence_rows.append(row); pooled_rows.append(row)
                    ridge_rows.append({"product": product, "horizon": horizon, "target": day.isoformat(), "actual": actual, "direct": 100.0})
            pred_products.append({"id": product, "runs": [{"modelId": "hist_gb", "cadence": "monthly", "predictions": pooled_rows}], "predictions": persistence_rows})
        forecasts = {"schemaVersion": 2, "products": pred_products}
        ridge = {"protocol": {"id": "direct-horizon-ridge-2026-10-06"}, "pairedRows": ridge_rows}

        class Estimator:
            def fit(self, x, y):
                return self
            def predict(self, x): return [0.0] * len(x)

        def estimator_factory(_):
            return Estimator()
        original_inverse = rw.modeling._inverse_log_ratio
        inverse_calls = 0
        def inverse_with_late_failure(*args):
            nonlocal inverse_calls
            inverse_calls += 1
            if inverse_calls == 20:
                raise RuntimeError("synthetic late inverse conversion failure")
            return original_inverse(*args)
        with mock.patch.object(rw, "EVAL_START", dt.date(2023, 1, 1)), \
             mock.patch.object(rw, "EVAL_END", dt.date(2023, 1, 15)), \
             mock.patch.object(rw.modeling, "_new_estimator", side_effect=estimator_factory), \
             mock.patch.object(rw.modeling, "_inverse_log_ratio", side_effect=inverse_with_late_failure):
            result = rw.build(prices, forecasts, ridge)
        self.assertEqual(len(result["comparisons"]), len(dashboard.PRODUCTS) * 10 * 4 * 2)
        self.assertEqual(result["protocol"]["horizons"], list(range(1, 11)))
        self.assertEqual(result["coverage"]["cabbage/10"]["commonPairedCount"], 5)
        failed_month = result["fitAudits"]["cabbage/1"]["months"][0]
        self.assertEqual(failed_month["arms"]["recent"]["status"], "fit_failed", repr(failed_month))
        self.assertEqual(failed_month["arms"]["full"]["status"], "fitted")
        self.assertFalse(any(r["product"] == "cabbage" and r["horizon"] == 1 and r["arm"] == "recent" for r in result["predictions"]))
        self.assertEqual(result["fitTotals"]["fitAttempts"]["recent"], result["fitTotals"]["fitSuccesses"]["recent"] + 1)
        lost = result["coverage"]["cabbage/1"]["candidateCoverageLossDates"][rw.COMPARATORS[0]]
        self.assertEqual(lost, [dt.date(2023, 1, i).isoformat() for i in range(2, 16)])
        json.dumps(result, allow_nan=False)
        self.assertTrue(all(r["distinctTrainingTargets"] >= rw.MIN_TRAIN_DAYS for r in result["predictions"] if r["arm"] == "recent"))


if __name__ == "__main__": unittest.main()
