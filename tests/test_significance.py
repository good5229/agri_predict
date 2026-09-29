import datetime as dt
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from agri_data import experiments, significance


def make_dashboard(days=35, *, omit=0, varied=True):
    first = dt.date(2017, 1, 1)
    dates = [(first + dt.timedelta(days=i)).isoformat() for i in range(days)]
    warmup_dates = [(first - dt.timedelta(days=i)).isoformat() for i in range(10, 0, -1)]
    products = []
    for product_id in significance.PRODUCT_IDS:
        observations = [{"date": day, "actual": 90.0 + i, "volumeKg": 50.0, "rowCount": 1}
                       for i, day in enumerate(warmup_dates)]
        observations.extend({"date": day, "actual": 100.0 + i, "volumeKg": 50.0, "rowCount": 1}
                            for i, day in enumerate(dates))
        actual = {row["date"]: row["actual"] for row in observations}
        baseline = []
        for horizon in significance.HORIZONS:
            for day in dates:
                target = dt.date.fromisoformat(day)
                origin = target - dt.timedelta(days=horizon)
                idx = dates.index(day)
                error = 4.0 + (idx % 3) if varied else 4.0
                baseline.append({"target": day, "origin": origin.isoformat(), "horizon": horizon,
                                 "predicted": actual[day] + error, "sourceDate": origin.isoformat()})
        runs = []
        for model_id in significance.MODEL_IDS:
            for cadence in significance.CADENCES:
                predictions = []
                for horizon in significance.HORIZONS:
                    for i, day in enumerate(dates):
                        if model_id == "ridge" and cadence == "daily" and horizon == 1 and i < omit:
                            continue
                        target = dt.date.fromisoformat(day)
                        origin = target - dt.timedelta(days=horizon)
                        error = (1.0 + (i % 2)) if varied else 1.0
                        predictions.append({"target": day, "origin": origin.isoformat(), "horizon": horizon,
                                            "predicted": actual[day] + error, "sourceDate": origin.isoformat(),
                                            "fitCutoff": origin.isoformat()})
                runs.append({"modelId": model_id, "cadence": cadence, "status": "available",
                             "predictions": predictions, "fits": []})
        products.append({"id": product_id, "observations": observations,
                         "predictions": baseline, "runs": runs})
    return {"schemaVersion": 2, "generatedAt": "2026-09-29T00:00:00+00:00",
            "meta": {"evaluationStart": dates[0], "evaluationEnd": dates[-1]},
            "products": products}


class SignificanceTests(unittest.TestCase):
    def test_full_family_pairing_direction_and_primary_sensitivity_shapes(self):
        result = significance.analyze(make_dashboard())
        self.assertEqual(result["protocol"]["primary"]["holmFamilySize"], 800)
        self.assertEqual(result["protocol"]["sensitivity"]["holmFamilySize"], 800)
        self.assertEqual(len(result["comparisons"]), 800)
        self.assertEqual(len(result["conclusions"]), 400)
        row = next(item for item in result["comparisons"] if item["productId"] == "cabbage"
                   and item["modelId"] == "ridge" and item["cadence"] == "daily"
                   and item["horizon"] == 1 and item["loss"] == "absolute_error")
        self.assertEqual(row["primary"]["n"], 35)
        self.assertEqual(row["primary"]["test"]["status"], "available")
        self.assertLess(row["primary"]["test"]["meanLossDifference"], 0)
        self.assertIn("holmAdjustedPValue", row["primary"]["test"])
        self.assertIn("calendarHacMaxLagDays", row["sensitivity"]["test"])
        outcome = next(item for item in result["conclusions"] if item["productId"] == "cabbage"
                       and item["modelId"] == "ridge" and item["cadence"] == "daily"
                       and item["horizon"] == 1)
        self.assertEqual(outcome["status"], "improved")
        self.assertEqual(outcome["sensitivityStatus"], "improved")
        markdown = significance.markdown_report(result)
        self.assertIn("Horizon 1 daily overview", markdown)
        self.assertIn("| cabbage | ridge | improved |", markdown)
        self.assertLess(markdown.index("## Horizon 1 daily overview"), markdown.index("## Improved comparisons"))
        self.assertLess(markdown.index("## Improved comparisons"), markdown.index("| cabbage | ridge | daily | 1 |"))

    def test_irregular_calendar_dm_matches_shared_calendar_hac_implementation(self):
        actuals, candidate, baseline = {}, [], []
        first = dt.date(2020, 1, 1)
        for i in range(35):
            day = first + dt.timedelta(days=i + i // 5)
            target = day.isoformat()
            actual = 200.0 + i
            actuals[(target, 2)] = actual
            origin = day - dt.timedelta(days=2)
            candidate.append({"target": target, "horizon": 2, "origin": origin.isoformat(),
                              "predicted": actual + 1 + i % 2})
            baseline.append({"target": target, "horizon": 2, "origin": origin.isoformat(),
                             "predicted": actual + 4 + i % 3})
        expected = experiments._dm(candidate, baseline, actuals, 2, "absolute_error", max_lag=9)
        actual = significance._dm_stats(candidate, baseline, actuals, 2, "absolute_error", 9,
                                        len(actuals), expected["meanLossDifference"])
        self.assertEqual(actual["status"], "available")
        self.assertAlmostEqual(actual["statistic"], expected["statistic"])
        self.assertAlmostEqual(actual["pValue"], expected["pValue"])

    def test_29_pairs_are_reported_but_unavailable_and_not_dropped_from_holm(self):
        result = significance.analyze(make_dashboard(35, omit=6))
        row = next(item for item in result["comparisons"] if item["productId"] == "cabbage"
                   and item["modelId"] == "ridge" and item["cadence"] == "daily"
                   and item["horizon"] == 1 and item["loss"] == "absolute_error")
        self.assertEqual(row["primary"]["n"], 29)
        self.assertEqual(row["primary"]["eligibleActualCount"], 35)
        self.assertEqual(row["primary"]["unmatchedPairCount"], 6)
        self.assertEqual(row["primary"]["test"]["status"], "unavailable_insufficient_pairs")
        self.assertIsNone(row["primary"]["test"]["pValue"])
        self.assertEqual(row["primary"]["test"]["holmAdjustedPValue"], 1.0)
        daily_h1 = next(line for line in significance.markdown_report(result).splitlines()
                        if line.startswith("| cabbage | ridge | unavailable | 29/29 |"))
        self.assertEqual(daily_h1.count("| — |"), 2)

    def test_unavailable_tests_keep_full_800_family_multiplier(self):
        records = [{"primary": {"test": {"pValue": None, "status": "unavailable"}}} for _ in range(799)]
        records.append({"primary": {"test": {"pValue": 0.001, "status": "available"}}})
        significance._apply_fixed_holm(records, "primary")
        self.assertEqual(records[-1]["primary"]["test"]["holmAdjustedPValue"], 0.8)
        self.assertTrue(all(record["primary"]["test"]["holmAdjustedPValue"] == 1.0 for record in records[:799]))

    def test_validation_rejects_duplicate_keys_nonpositive_actual_and_future_causality(self):
        source = make_dashboard()
        source["products"][0]["runs"][0]["predictions"].append(
            dict(source["products"][0]["runs"][0]["predictions"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            significance.analyze(source)
        source = make_dashboard()
        source["products"][0]["observations"][0]["actual"] = 0
        with self.assertRaisesRegex(ValueError, "positive finite"):
            significance.analyze(source)
        source = make_dashboard()
        source["products"][0]["runs"][0]["predictions"][0]["fitCutoff"] = "2018-01-01"
        with self.assertRaisesRegex(ValueError, "after origin"):
            significance.analyze(source)
        source = make_dashboard()
        source["products"][0]["runs"][0]["predictions"][0]["sourceDate"] = "2014-12-01"
        with self.assertRaisesRegex(ValueError, "not an observed"):
            significance.analyze(source)
        source = make_dashboard()
        source["products"][0]["observations"].append(
            {"date": "2016-12-10", "actual": 90.0, "volumeKg": 50.0, "rowCount": 1})
        source["products"][0]["runs"][0]["predictions"][0]["sourceDate"] = "2016-12-10"
        with self.assertRaisesRegex(ValueError, "maximum allowed source age"):
            significance.analyze(source)

    def test_cli_uses_exact_input_bytes_hash_and_refuses_existing_output(self):
        source = make_dashboard(30)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path, output_path, markdown_path = root / "input.json", root / "sig.json", root / "sig.md"
            raw = (json.dumps(source, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
            input_path.write_bytes(raw)
            argv = ["--input", str(input_path), "--output", str(output_path), "--markdown", str(markdown_path)]
            self.assertEqual(significance.main(argv), 0)
            report = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(report["protocol"]["inputSha256"], hashlib.sha256(raw).hexdigest())
            self.assertIn("not preregistered", markdown_path.read_text(encoding="utf-8"))
            previous = output_path.read_bytes()
            self.assertEqual(significance.main(argv), 2)
            self.assertEqual(output_path.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
