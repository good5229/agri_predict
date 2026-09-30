import datetime as dt
import hashlib
import math
import unittest

from agri_data import correction_experiment as experiment, significance


def synthetic_source():
    obs_start = dt.date(2017, 6, 1)
    obs_end = experiment.EVALUATION_END
    dates = []
    day = obs_start
    while day <= obs_end:
        dates.append(day)
        day += dt.timedelta(days=1)
    target_start = dt.date(2017, 7, 1)
    target_end = dt.date(2018, 12, 31)
    targets = []
    day = target_start
    while day <= target_end:
        targets.append(day)
        day += dt.timedelta(days=1)
    targets.extend(dt.date(2019, 1, 1) + dt.timedelta(days=i) for i in range(10))
    products = []
    for product_index, product_id in enumerate(significance.PRODUCT_IDS):
        actual = {day.isoformat(): 150.0 + product_index * 30 + 0.03 * (day - obs_start).days
                  + 2.0 * math.sin((day - obs_start).days / 7) for day in dates}
        baseline_rows = []
        ridge_rows = []
        for target in targets:
            for horizon in range(1, 11):
                origin = target - dt.timedelta(days=horizon)
                origin_price = actual[origin.isoformat()]
                y = actual[target.isoformat()]
                ridge_prediction = origin_price + 0.65 * (y - origin_price) + 0.1 * math.sin(target.day)
                common = {"target": target.isoformat(), "origin": origin.isoformat(), "horizon": horizon,
                          "sourceDate": origin.isoformat()}
                baseline_rows.append({**common, "predicted": origin_price})
                ridge_rows.append({**common, "predicted": ridge_prediction, "fitCutoff": origin.isoformat()})
        runs = []
        for model_id in significance.MODEL_IDS:
            for cadence in significance.CADENCES:
                predictions = ridge_rows if model_id == "ridge" and cadence == "daily" else []
                runs.append({"modelId": model_id, "cadence": cadence, "predictions": predictions})
        products.append({"id": product_id,
                         "observations": [{"date": d.isoformat(), "actual": actual[d.isoformat()],
                                           "volumeKg": 50.0, "rowCount": 1} for d in dates],
                         "predictions": baseline_rows, "runs": runs})
    return {"schemaVersion": 2,
            "meta": {"evaluationStart": "2017-01-01", "evaluationEnd": experiment.EVALUATION_END.isoformat()},
            "products": products}


class CorrectionExperimentTests(unittest.TestCase):
    def test_type7_quantile_and_weight_selection_gates(self):
        self.assertEqual(experiment._quantile75([0.0, 10.0, 20.0, 30.0]), 22.5)
        self.assertEqual(experiment._select_weight([], 60)["weight"], 0.0)
        rows = [{"actual": 100.0 + (i % 3), "baseline": 105.0, "ridge": 99.0 + (i % 3)}
                for i in range(60)]
        fit = experiment._select_weight(rows, 60)
        self.assertIn(fit["weight"], experiment.WEIGHTS)
        self.assertLess(fit["weight"], 1.0)
        self.assertEqual(experiment._losses(rows, 0.0)[0],
                         sum(abs(row["baseline"] - row["actual"]) for row in rows) / len(rows))
        self.assertEqual(experiment._losses(rows, 1.0)[0],
                         sum(abs(row["ridge"] - row["actual"]) for row in rows) / len(rows))

    def test_training_window_is_180_days_with_strict_target_embargo(self):
        fit = dt.date(2019, 1, 1)
        dates = [fit - dt.timedelta(days=180), fit - dt.timedelta(days=1), fit]
        base = {(day.isoformat(), 1): {"predicted": 10.0} for day in dates}
        ridge = {(day.isoformat(), 1): {"predicted": 11.0} for day in dates}
        actual = {day.isoformat(): 12.0 for day in dates}
        selected = experiment._training_rows(base, ridge, actual, 1, fit)
        self.assertEqual([row["target"] for row in selected], [dates[0].isoformat(), dates[1].isoformat()])
        self.assertTrue(all(dt.date.fromisoformat(row["target"]) < fit for row in selected))

    def test_regime_insufficient_bucket_falls_back_to_common_weight(self):
        rows = [{"baseline": 100.0, "ridge": 101.0, "actual": 100.5} for _ in range(45)]
        rows.extend({"baseline": 100.0, "ridge": 120.0, "actual": 110.0} for _ in range(15))
        common = experiment._select_weight(rows, experiment.GLOBAL_MIN_N)
        regime = experiment._fit_regimes(rows, common)
        self.assertEqual(regime["status"], "fit")
        self.assertEqual(regime["regimeRule"], "low <= threshold; high > threshold")
        self.assertEqual(regime["regimes"]["high"]["status"], "fallback_common_weight")
        self.assertEqual(regime["regimes"]["high"]["weight"], common["weight"])

    def test_monthly_fit_ignores_actuals_at_or_after_fit_origin(self):
        source = synthetic_source()
        product = source["products"][0]
        dates = {row["date"]: row["actual"] for row in product["observations"]}
        _, forecast_maps, _, _ = significance._validate_input(source)
        baseline, ridge = forecast_maps[(product["id"], "baseline")], forecast_maps[(product["id"], "ridge", "daily")]
        fit_origin = dt.date(2019, 1, 1)
        before = experiment._fit_month(product["id"], 1, fit_origin, baseline, ridge, dates)
        changed = {day: (value * 1.7 if day >= fit_origin.isoformat() else value) for day, value in dates.items()}
        after = experiment._fit_month(product["id"], 1, fit_origin, baseline, ridge, changed)
        self.assertEqual(before, after)
        self.assertLess(dt.date.fromisoformat(before["trainingTargetEnd"]), fit_origin)

    def test_fixed_family_predictions_logs_and_conclusions(self):
        source = synthetic_source()
        raw_hash = "synthetic-frozen-hash"
        result = experiment.analyze(source, raw_hash, expected_sha256=raw_hash)
        self.assertEqual(len(result["comparisons"]), 240)
        self.assertEqual(len(result["conclusions"]), 60)
        self.assertEqual(len(result["correctedPredictions"]), 2 * 3 * 10 * 10)
        self.assertEqual(result["protocol"]["primaryInference"]["familySize"], 240)
        keys = [(row["productId"], row["candidateId"], row["horizon"], row["comparatorId"], row["loss"])
                for row in result["comparisons"]]
        self.assertEqual(len(set(keys)), 240)
        fit_keys = [(row["productId"], row["candidateId"], row["horizon"], row["fitOrigin"])
                    for row in result["fits"]]
        self.assertEqual(len(set(fit_keys)), len(fit_keys))
        for fit in result["fits"]:
            self.assertTrue(fit["fitOrigin"].endswith("-01"))
            if fit["trainingTargetEnd"]:
                self.assertLess(fit["trainingTargetEnd"], fit["fitOrigin"])
        for prediction in result["correctedPredictions"]:
            self.assertLessEqual(prediction["fitCutoff"], prediction["origin"])
            self.assertIn("weight", prediction)
            self.assertIn("trainingN", prediction)
        self.assertEqual(result["productDiagnostics"][0]["candidateHorizons"]["fixed_half/h1"]
                         ["matchedMetrics"]["candidate"]["n"], 10)
        regime_summary = result["fitSummary"]["correction_regime"]
        self.assertEqual(regime_summary["regimeFallbackCount"],
                         2 * regime_summary["regimeGlobalFallbackFitCount"]
                         + regime_summary["regimeBucketFallbackCount"])
        self.assertEqual(result["productDiagnostics"][0]["candidateHorizons"]["adaptive/h1"]
                         ["spikeDefinition"]["eligibleN"], 10)
        md = experiment.markdown_report(result)
        self.assertIn("후보 RMSE", md)
        self.assertIn("primary AE/APE Holm p", md)
        self.assertIn("fallback", md)

    def test_wrong_frozen_hash_rejected(self):
        with self.assertRaisesRegex(experiment.ExperimentError, "frozen prediction archive hash"):
            experiment.analyze(synthetic_source(), "wrong-hash")


if __name__ == "__main__":
    unittest.main()
