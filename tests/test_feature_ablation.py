import datetime as dt
import tempfile
import unittest
from pathlib import Path

from agri_data import feature_ablation as fa


class FeatureAblationTests(unittest.TestCase):
    def test_prices_allow_signed_volume_and_reject_bad_price(self):
        source = {"schemaVersion": 1, "products": [
            {"id": p, "observations": [{"date": "2024-01-01", "actual": 10, "volumeKg": -2},
                                          {"date": "2024-01-02", "actual": 0, "volumeKg": 5}]}
            for p in fa.PRODUCTS]}
        series, excluded = fa._parse_prices(source)
        self.assertEqual(series[fa.PRODUCTS[0]][dt.date(2024, 1, 1)]["volumeKg"], -2)
        self.assertEqual(excluded[fa.PRODUCTS[0]]["nonpositive_price"], 1)

    def test_features_use_only_dates_strictly_before_origin_and_keep_signed_volume(self):
        series = {dt.date(2024, 1, 1): {"price": 10., "volumeKg": -200.},
                  dt.date(2024, 1, 2): {"price": 1000., "volumeKg": -99.},
                  dt.date(2024, 1, 3): {"price": 25., "volumeKg": 2.}}
        result = fa._price_features(series, dt.date(2024, 1, 3), dt.date(2024, 1, 4), 1)
        self.assertEqual(result["sourceDate"], dt.date(2024, 1, 2))
        self.assertAlmostEqual(result["featuresP"][0], __import__("math").log(1000.))
        # The source day is included in the prior 7-day volume sum; origin day is not.
        self.assertLess(result["volumeFeatures"][0], 0)

    def test_weather_requires_all_three_stations_and_seven_full_days(self):
        origin = dt.date(2024, 2, 10)
        weather = {s: {origin-dt.timedelta(days=i): {"avgTa": float(i)} for i in range(1, 8)}
                   for s in fa.STATIONS}
        values, _ = fa._weather_window(weather, {}, origin)
        self.assertEqual(values, (4., 4., 4.))
        del weather[fa.STATIONS[0]][origin-dt.timedelta(days=2)]
        self.assertIsNone(fa._weather_window(weather, {}, origin))

    def test_samples_separate_price_possible_and_weather_complete(self):
        day = dt.date(2024, 1, 11)
        series = {"crop": {day-dt.timedelta(days=2): {"price": 10., "volumeKg": 1.},
                           day-dt.timedelta(days=1): {"price": 11., "volumeKg": 1.},
                           day: {"price": 12., "volumeKg": 1.}}}
        samples, counts = fa._make_samples(series, {}, {}, "crop", 1)
        self.assertEqual(samples, [])
        self.assertEqual(counts["pricePossibleTargets"], 1)
        self.assertEqual(counts["weatherMissingTargets"], 1)

    def test_dm_minimum_pairs_and_constant_difference_guard(self):
        rows = [{"target": dt.date(2024, 1, i+1), "horizon": 1, "actual": 10., "P": 9., "persistence": 9.}
                for i in range(fa.MIN_PAIRS)]
        result = fa._dm_test(rows, "P", "persistence", "absolute_error", 9)
        self.assertEqual(result["status"], "unavailable_nonpositive_variance")
        self.assertEqual(result["n"], fa.MIN_PAIRS)
        short = fa._dm_test(rows[:fa.MIN_PAIRS-1], "P", "persistence", "absolute_error", 9)
        self.assertEqual(short["status"], "unavailable_insufficient_pairs")

    def test_holm_family_applies_full_150_denominator(self):
        tests = []
        for i in range(150):
            tests.append({"productId": fa.PRODUCTS[i % 5], "horizon": fa.HORIZONS[i % 3],
                          "comparison": str(i), "loss": "absolute_error",
                          "test": {"pValue": .001, "status": "available"}})
        fa._holm(tests, "test")
        self.assertAlmostEqual(tests[0]["test"]["holmAdjustedPValue"], .15)
        self.assertTrue(all("holmAdjustedPValue" in row["test"] for row in tests))

    def test_json_and_markdown_write_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = {"comparisons": [], "sampleCounts": {}}
            fa.write_report(report, root / "out.json", root / "out.md")
            self.assertIn("# 선행 거래량", (root / "out.md").read_text())
            self.assertTrue((root / "out.json").read_text().endswith("\n"))


if __name__ == "__main__":
    unittest.main()
