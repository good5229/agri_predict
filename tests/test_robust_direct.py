import datetime as dt
import contextlib
import unittest
from unittest import mock

from agri_data import dashboard, robust_direct


class RobustDirectTests(unittest.TestCase):
    def test_holm_keeps_unavailable_tests_in_fixed_family(self):
        rows = [{"pValue": p} for p in (.001, .02, .5)] + [{"pValue": 1.0} for _ in range(87)]
        robust_direct.holm(rows)
        self.assertEqual(len(rows), robust_direct.FAMILY_SIZE)
        self.assertGreaterEqual(rows[0]["holmPValue"], .09)
        self.assertFalse(rows[-1]["holmSignificant"])

    def test_calendar_dm_uses_calendar_gaps_and_requires_thirty_pairs(self):
        values = [float((i*i + 3*i) % 11) for i in range(35)]
        sparse = {dt.date(2024, 1, 1) + dt.timedelta(days=2*i): values[i] for i in range(35)}
        dense = {dt.date(2024, 1, 1) + dt.timedelta(days=i): values[i] for i in range(35)}
        self.assertEqual(robust_direct.dm(dict(list(sparse.items())[:29]), 9)["status"], "insufficient_pairs")
        self.assertNotEqual(robust_direct.dm(sparse, 9)["statistic"], robust_direct.dm(dense, 9)["statistic"])

    def test_report_pairing_is_by_product_horizon_and_target(self):
        day = dt.date(2024, 1, 2)
        raw = {"protocol": {"id": "direct-horizon-ridge-2026-10-06"}, "coverage": {
            "cabbage/5": {"candidateAvailableDates": [day.isoformat()], "comparatorAvailableDates": {
                "monthly_pooled_ridge": [day.isoformat()], "persistence": [day.isoformat()]}}}, "pairedRows": [
            {"product": "cabbage", "horizon": 5, "target": day.isoformat(), "actual": 10,
             "direct": 11, "monthlyPooledRidge": 12, "persistence": 13}]}
        got, availability = robust_direct._reference(raw)
        self.assertEqual(got[("cabbage", 5, day)]["direct_ridge"], 11)
        self.assertIn(day, availability["cabbage/5"]["direct_ridge"])
        self.assertNotIn(("cabbage", 1, day), got)

    def test_coverage_loss_uses_each_controls_full_availability(self):
        day1, day2, day3 = (dt.date(2024, 1, n) for n in (1, 2, 3))
        losses = robust_direct._coverage_losses({"direct_ridge": {day1, day2},
            "monthly_pooled_ridge": {day1, day2, day3}, "persistence": {day2, day3}}, {day1, day2})
        self.assertEqual(losses, {"direct_ridge": 0, "monthly_pooled_ridge": 1, "persistence": 1})

    def test_joint_decision_requires_six_tests_metrics_and_coverage(self):
        rows, coverage = [], {}
        for pid in dashboard.PRODUCTS:
            for h in robust_direct.HORIZONS:
                coverage[f"{pid}/{h}"] = {"candidateCoverageLossCounts": {c: 0 for c in robust_direct.COMPARATORS}}
                for comparator in robust_direct.COMPARATORS:
                    for loss in robust_direct.LOSSES:
                        rows.append({"product": pid, "horizon": h, "comparator": comparator, "loss": loss,
                            "n": 50, "huberMetric": 1.0, "comparatorMetric": 2.0,
                            "dm": {"9": {"status": "tested", "holmSignificant": True, "meanLossDifference": -1.0}}})
        result = robust_direct.joint_conclusions(rows, coverage)
        self.assertEqual(result[0]["conclusion"], "improvement")
        rows[5]["dm"]["9"]["holmSignificant"] = False
        self.assertEqual(robust_direct.joint_conclusions(rows, coverage)[0]["conclusion"], "inconclusive")
        rows[5]["dm"]["9"]["holmSignificant"] = True
        coverage["cabbage/1"]["candidateCoverageLossCounts"]["persistence"] = 1
        self.assertEqual(robust_direct.joint_conclusions(rows, coverage)[0]["conclusion"], "inconclusive")

    def test_build_records_training_cutoff_and_failed_convergence_without_imputation(self):
        first, last = dt.date(2014, 1, 1), dt.date(2015, 1, 10)
        products, paired = [], []
        for pid in dashboard.PRODUCTS:
            obs = []
            for i in range((last-first).days+1):
                day = first + dt.timedelta(days=i)
                obs.append({"date": day.isoformat(), "actual": 100+i*.1, "volumeKg": 1000})
                if dt.date(2015, 1, 1) <= day <= last and (day-dt.timedelta(days=1)) >= dt.date(2015, 1, 1):
                    paired.append({"product": pid, "horizon": 1, "target": day.isoformat(), "actual": 100+i*.1,
                        "direct": 100+i*.1, "monthlyPooledRidge": 100+i*.1, "persistence": 100+i*.1})
            products.append({"id": pid, "observations": obs})
        coverage_rows = {f"{pid}/1": {"candidateAvailableDates": [r["target"] for r in paired if r["product"] == pid],
            "comparatorAvailableDates": {"monthly_pooled_ridge": [r["target"] for r in paired if r["product"] == pid],
            "persistence": [r["target"] for r in paired if r["product"] == pid]}} for pid in dashboard.PRODUCTS}
        report = {"protocol": {"id": "direct-horizon-ridge-2026-10-06"}, "coverage": coverage_rows, "pairedRows": paired}
        class Estimator:
            n_iter_ = 1
            def fit(self, x, y): self.x = x; return self
            def predict(self, x): return [0.0]
        class Pipeline(list):
            def fit(self, x, y):
                self.x, self.y = x, y
                self[-1].fit(x, y)
                return self
            def predict(self, x): return self[-1].predict(x)
        with mock.patch.object(robust_direct, "EVAL_START", dt.date(2015, 1, 1)), \
             mock.patch.object(robust_direct, "EVAL_END", last), \
             mock.patch.object(robust_direct, "HORIZONS", (1,)), \
             mock.patch.object(robust_direct, "FAMILY_SIZE", len(dashboard.PRODUCTS)*1*3*2), \
             mock.patch.object(robust_direct, "threadpool_limits", return_value=contextlib.nullcontext()), \
             mock.patch.object(robust_direct, "_new_model", side_effect=lambda: Pipeline([object(), Estimator()])):
            result = robust_direct.build({"schemaVersion": 1, "products": products}, report)
        audit = result["fitAudits"]["cabbage/1"]
        self.assertTrue(audit["fitAttempts"] > 0)
        self.assertTrue(all(m["trainTargetEnd"] is None or m["trainTargetEnd"] <= m["fitOrigin"] for m in audit["months"]))
        self.assertEqual(result["fitTotals"]["fitFailures"], 0, audit["months"][:2])
        self.assertTrue(result["predictions"])
        self.assertTrue(all(all(k in row for k in ("sourceDate", "fitOrigin", "fitTrainTargetEnd", "fitDistinctTargets", "fitSamples", "clipped")) for row in result["predictions"]))

        class Failed(Pipeline):
            def fit(self, x, y):
                self[-1].n_iter_ = 1000
                return self
        with mock.patch.object(robust_direct, "EVAL_START", dt.date(2015, 1, 1)), \
             mock.patch.object(robust_direct, "EVAL_END", last), \
             mock.patch.object(robust_direct, "HORIZONS", (1,)), \
             mock.patch.object(robust_direct, "FAMILY_SIZE", len(dashboard.PRODUCTS)*1*3*2), \
             mock.patch.object(robust_direct, "threadpool_limits", return_value=contextlib.nullcontext()), \
             mock.patch.object(robust_direct, "_new_model", side_effect=lambda: Failed([object(), Estimator()])):
            failed = robust_direct.build({"schemaVersion": 1, "products": products}, report)
        self.assertGreater(failed["fitTotals"]["fitFailures"], 0)
        self.assertEqual(failed["fitTotals"]["fitSuccesses"], 0)
        self.assertEqual(failed["coverage"]["cabbage/1"]["commonPairedCount"], 0)

        class Nonfinite(Estimator):
            def predict(self, x): return [float("inf")]
        with mock.patch.object(robust_direct, "EVAL_START", dt.date(2015, 1, 1)), \
             mock.patch.object(robust_direct, "EVAL_END", last), \
             mock.patch.object(robust_direct, "HORIZONS", (1,)), \
             mock.patch.object(robust_direct, "FAMILY_SIZE", len(dashboard.PRODUCTS)*1*3*2), \
             mock.patch.object(robust_direct, "threadpool_limits", return_value=contextlib.nullcontext()), \
             mock.patch.object(robust_direct, "_new_model", side_effect=lambda: Pipeline([object(), Nonfinite()])):
            invalid = robust_direct.build({"schemaVersion": 1, "products": products}, report)
        self.assertGreater(invalid["fitTotals"]["fitFailures"], 0)
        self.assertEqual(invalid["fitTotals"]["fitSuccesses"], 0)
        self.assertEqual(invalid["predictions"], [])


if __name__ == "__main__":
    unittest.main()
