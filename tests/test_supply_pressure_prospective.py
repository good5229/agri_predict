import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path

from agri_data import dashboard, daily_training, supply_pressure_prospective as prospective


class SupplyPressureProspectiveTests(unittest.TestCase):
    def test_historical_feature_excludes_origin_and_future_observations(self):
        start = dt.date(2019, 1, 1)
        series = {start + dt.timedelta(days=i): (100.0 + i, 20.0 + i) for i in range(1700)}
        origin, target = dt.date(2023, 6, 20), dt.date(2023, 6, 25)
        row = prospective._feature(series, origin, target, 5)
        self.assertIsNotNone(row)
        changed = dict(series)
        changed[origin] = (900000.0, 900000.0)
        changed[origin + dt.timedelta(days=1)] = (800000.0, 800000.0)
        self.assertEqual(row, prospective._feature(changed, origin, target, 5))

    def test_seasonal_baseline_uses_three_complete_years_before_origin(self):
        target, origin = dt.date(2024, 1, 5), dt.date(2023, 12, 31)
        series = {dt.date(y, 1, 5): (float(y), 5.0) for y in (2020, 2021, 2022)}
        # Dec 22-31 of 2019 lies inside a raw ±14-day window but cannot count for 2020.
        series.update({dt.date(2019, 12, d): (1.0, 5.0) for d in range(22, 32)})
        self.assertEqual(prospective._seasonal(series, target, origin), 2021.0)
        incomplete = {day: value for day, value in series.items() if day.year != 2021}
        self.assertIsNone(prospective._seasonal(incomplete, target, origin))

    def test_nonpositive_aggregate_volume_is_rejected(self):
        start = dt.date(2019, 1, 1)
        series = {start + dt.timedelta(days=i): (100.0 + i, 20.0) for i in range(1700)}
        target = dt.date(2023, 6, 25)
        origin = target - dt.timedelta(days=5)
        # Raw signed adjustments remain upstream; only positive aggregate snapshot volumes enter this schema.
        with self.assertRaises(ValueError):
            prospective._series({"id": "cabbage", "observations": [
                {"date": (origin - dt.timedelta(days=1)).isoformat(), "actual": 110.0, "volumeKg": -7.0}]})
        self.assertIsNotNone(prospective._feature(series, origin, target, 5))

    def test_direct_ridge_requires_ninety_distinct_targets(self):
        rows = [(dt.date(2020, 1, 1) + dt.timedelta(days=i), (float(i), float(i % 7)), float(i) / 10)
                for i in range(90)]
        with self.assertRaisesRegex(ValueError, "fewer than 90"):
            prospective._fit_direct_ridge(rows[:89])
        model = prospective._fit_direct_ridge(rows)
        self.assertEqual(len(model.predict([(90.0, 6.0)])), 1)

    def test_immutable_writer_links_complete_file_and_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "record.json"
            prospective._write_immutable(output, "complete")
            with self.assertRaises(FileExistsError):
                prospective._write_immutable(output, "replacement")
            self.assertEqual(output.read_text(encoding="utf-8"), "complete")
            self.assertEqual(list(Path(temp).glob("*.tmp")), [])

    def _verified_fixture(self, root):
        issue, cutoff = dt.date(2026, 10, 8), dt.date(2026, 10, 6)
        run = root / "run"
        run.mkdir()
        prices = {"schemaVersion": 1, "generatedAt": "2026-10-07T03:00:00+00:00",
                  "meta": {"contextEnd": cutoff.isoformat()},
                  "products": [{"id": pid, "observations": []} for pid in dashboard.PRODUCTS]}
        prov = [
            {"date": (daily_training.CONTEXT_START + dt.timedelta(days=i)).isoformat(),
             "market": market, "product": product,
             "responseAt": "2026-10-07T03:00:00+00:00"}
            for i in range((cutoff - daily_training.CONTEXT_START).days + 1)
            for market, product in daily_training.EXPECTED_SCOPES
        ]
        provenance_path = run / "selected-provenance.json"
        provenance_path.write_text(json.dumps(prov), encoding="utf-8")
        status = {"status": "complete", "cutoff": cutoff.isoformat(),
                  "collectionStatus": {"status": "complete"}, "runDirectory": str(run),
                  "selectedProvenance": str(provenance_path), "finishedAtKst": "2026-10-07T15:00:00+09:00"}
        status["snapshotSha256"] = daily_training._pipeline_fingerprint(prices, prov)
        prices_path, status_path = run / "prices.json", run / "run-status.json"
        prices_path.write_text(json.dumps(prices), encoding="utf-8")
        status_path.write_text(json.dumps(status), encoding="utf-8")
        return issue, cutoff, prices_path, status_path

    def test_snapshot_validation_rejects_cutoff_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            issue, cutoff, prices_path, status_path = self._verified_fixture(Path(temp))
            status = json.loads(status_path.read_text())
            status["cutoff"] = "2026-10-05"
            with self.assertRaisesRegex(ValueError, "cutoff mismatch"):
                prospective.validate_snapshot(prices_path.read_bytes(), json.dumps(status).encode(), issue, cutoff)

    def test_forecast_output_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            issue, cutoff, prices_path, status_path = self._verified_fixture(Path(temp))
            output = Path(temp) / "forecast.json"
            output.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prospective.build(prices_path, status_path, issue, cutoff, output)
            self.assertEqual(output.read_text(encoding="utf-8"), "preserve")


if __name__ == "__main__":
    unittest.main()
