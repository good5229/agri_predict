import datetime as dt
import json
import math
import tempfile
import unittest
from pathlib import Path

from agri_data import radish_feature_search as rfs


def _series(start, end):
    result = {}
    day = start
    i = 0
    while day <= end:
        result[day] = (100.0 + i * 0.03 + (i % 11), 500.0 + (i % 17))
        day += dt.timedelta(days=1)
        i += 1
    return result


class RadishFeatureSearchTests(unittest.TestCase):
    def test_sixteen_features_and_strict_origin_prefix(self):
        base = _series(dt.date(2018, 1, 1), dt.date(2022, 12, 31))
        origin = dt.date(2021, 6, 1)
        target = origin + dt.timedelta(days=5)
        x1, source1 = rfs._features(base, origin, target)
        extended = dict(base)
        extended[origin] = (999999.0, 999999.0)
        extended[target] = (888888.0, 888888.0)
        x2, source2 = rfs._features(extended, origin, target)
        self.assertEqual(len(x1), 16)
        self.assertEqual(x1, x2)
        self.assertEqual(source1, source2)
        self.assertEqual([name for name, _ in rfs.FEATURES][13], "sma7_minus_sma28_log_volume")

    def test_selected_and_full_features_use_strict_origin_and_full_prior_windows(self):
        base = _series(dt.date(2020, 1, 1), dt.date(2021, 12, 31))
        origin = dt.date(2021, 6, 1)
        target = origin + dt.timedelta(days=3)
        selected, selected_source = rfs._features(base, origin, target)
        full_source, full = rfs._strict_full_row(base, origin, target, 3, {})
        self.assertEqual(selected_source, full_source)
        expected7 = [math.log(base[origin-dt.timedelta(days=i)][0]) for i in range(1, 8)]
        self.assertAlmostEqual(selected[1], sum(expected7)/7)
        self.assertAlmostEqual(selected[2], sum(math.log(base[origin-dt.timedelta(days=i)][0]) for i in range(1, 29))/28)
        self.assertEqual(len(full), len(rfs.modeling.FEATURE_NAMES))
        self.assertAlmostEqual(full[15], sum(expected7)/7)
        self.assertAlmostEqual(full[22], sum(math.log(base[origin-dt.timedelta(days=i)][0]) for i in range(1, 29))/28)
        changed = dict(base)
        changed[origin] = (1_000_000.0, 1_000_000.0)
        changed[target] = (2_000_000.0, 2_000_000.0)
        selected_after, _ = rfs._features(changed, origin, target)
        _, full_after = rfs._strict_full_row(changed, origin, target, 3, {})
        self.assertEqual(selected, selected_after)
        self.assertEqual(full, full_after)

    def test_validation_training_cutoff_uses_earliest_forecast_origin(self):
        month = dt.date(2020, 2, 1)
        samples = [{"target": dt.date(2020, 1, 25), "origin": dt.date(2020, 1, 25), "h": 1,
                    "x": (1.0,) * 16},
                   {"target": dt.date(2020, 2, 1), "origin": dt.date(2020, 1, 22), "h": 10,
                    "x": (1.0,) * 16}]
        train, valid, cutoff = rfs._validation_fold(month, samples)
        self.assertEqual(cutoff, dt.date(2020, 1, 22))
        self.assertEqual([s["target"] for s in valid], [dt.date(2020, 2, 1)])
        self.assertEqual(train, [])

    def test_log_ratio_prediction_uses_its_own_source_price(self):
        self.assertAlmostEqual(rfs._price_from_log_ratio(123.0, math.log(2.0)), 246.0)
        self.assertNotAlmostEqual(rfs._price_from_log_ratio(123.0, math.log(2.0)),
                                  rfs._price_from_log_ratio(456.0, math.log(2.0)))

    def test_event_exposure_uses_strict_origin_and_feature_dependencies(self):
        raw = _series(dt.date(2023, 1, 1), dt.date(2023, 1, 21))
        raw[dt.date(2023, 1, 21)] = (10000.0, 0.5)
        without_event = dict(raw)
        without_event.pop(rfs.EVENT_DAY)
        selected_latest = [rfs.IDX["latest_log_price"]]
        self.assertFalse(rfs._event_input_exposed(raw, without_event, dt.date(2023, 1, 12),
                                                  dt.date(2023, 1, 22), 10, selected_latest, "full_ridge"))
        self.assertTrue(rfs._event_input_exposed(raw, without_event, dt.date(2023, 1, 22),
                                                 dt.date(2023, 1, 23), 1, selected_latest, "full_ridge"))
        self.assertTrue(rfs._event_input_exposed(raw, without_event, dt.date(2026, 1, 1),
                                                 dt.date(2026, 1, 2), 1,
                                                 [rfs.IDX["ewma7_log_price"]], "full_ridge"))
        self.assertEqual(rfs._event_slices(dt.date(2023, 1, 21)), ["event_target"])
        self.assertEqual(rfs._event_slices(dt.date(2023, 1, 22), True), ["event_input_followup"])
        self.assertEqual(rfs._event_slices(dt.date(2024, 1, 1), True), ["event_input_followup", "from_2024"])

    def test_conclusion_marks_primary_dm_unavailable_with_reason(self):
        def comparison(status):
            test={"status":status,"holmPValue":None,"meanLossDifference":0.0}
            return {"tests":{"absolute_error":{"hac9":test},
                             "absolute_percentage_error":{"hac9":{"status":"tested","holmPValue":0.5}}},
                    "metrics":{"selected":{"mae":1.0,"mapePercent":1.0},
                              "full_ridge":{"mae":1.0,"mapePercent":1.0}},"comparator":"full_ridge"}
        self.assertIn("inference_unavailable: inference_not_supported_under_30_pairs",
                      rfs._conclusion(comparison("inference_not_supported_under_30_pairs")))
        self.assertIn("inference_unavailable: inference_not_supported_constant_loss_difference",
                      rfs._conclusion(comparison("inference_not_supported_constant_loss_difference")))

    def test_training_labels_are_selected_by_strict_target_before_origin(self):
        samples = rfs._candidate_samples(_series(dt.date(2017, 1, 1), dt.date(2019, 12, 31)))
        cutoff = dt.date(2019, 1, 1)
        training = [s for s in samples if s["target"] < cutoff]
        self.assertTrue(training)
        self.assertTrue(all(s["target"] < cutoff for s in training))

    def test_report_writes_json_and_markdown(self):
        report = {"inputSha256": "abc", "selection": {"selectedFeatureIds": []},
                  "test": {"comparisons": []}}
        with tempfile.TemporaryDirectory() as tmp:
            js, md = Path(tmp) / "report.json", Path(tmp) / "report.md"
            rfs.write_report(report, js, md)
            self.assertEqual(json.loads(js.read_text()), report)
            self.assertIn("# 무 시계열", md.read_text())

    def test_no_folds_and_zero_common_pairs_are_reported_without_false_improvement(self):
        obs = _series(dt.date(2014, 1, 1), dt.date(2017, 12, 31))
        product = {"id": "radish", "observations": [
            {"date": day.isoformat(), "actual": price, "volumeKg": volume}
            for day, (price, volume) in obs.items()]}
        report = rfs.analyze({"products": [product]}, "synthetic-input-sha")
        self.assertEqual(rfs.TEST_START, dt.date(2023, 1, 11))
        self.assertEqual(report["selection"]["validationMonthlyFolds"], 0)
        self.assertEqual(report["selection"]["selectedFeatureIds"], [])
        self.assertEqual(report["test"]["holmFamilySizePerBandwidth"], 12)
        self.assertEqual(len(report["test"]["comparisons"]), 6)
        self.assertFalse(report["test"]["promotionAssessment"]["eligibleForFutureReview"])
        self.assertTrue(all(c["n"] == 0 and c["conclusion"].startswith("inference_unavailable:")
                            for c in report["test"]["comparisons"]))
        self.assertTrue(all(test["status"] == "inference_not_supported_under_30_pairs"
                            for c in report["test"]["comparisons"]
                            for loss in c["tests"].values() for test in loss.values()))
        self.assertTrue(all(test["holmFamilySize"] == 12
                            for c in report["test"]["comparisons"]
                            for loss in c["tests"].values() for test in loss.values()))
        self.assertIn("eligibleSeriesSha256", report["provenance"])
        self.assertIn("selectedTrainingSha256", report["provenance"])
        json.dumps(report, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
