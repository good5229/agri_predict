import datetime as dt
import unittest
from unittest import mock

from agri_data import kma_residual_eval as ev
from agri_data.kma_historical_forecast import ForecastArchive, ForecastRecord

KST = dt.timezone(dt.timedelta(hours=9))


def synthetic_inputs(missing_eval_origin=None):
    targets = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(140)]
    targets += [dt.date(2025, 9, 25) + dt.timedelta(days=i) for i in range(40)]
    observations = {dt.date(2023, 12, 31) + dt.timedelta(days=i): 80. + i for i in range(700)}
    products = [{"id": p, "observations": [{"date": d.isoformat(), "actual": v, "volumeKg": 1}
                                               for d, v in sorted(observations.items())]}
                for p in ev.PRODUCTS]
    paired = []
    records = []
    for i, target in enumerate(targets):
        origin = target - dt.timedelta(days=1)
        actual = observations[target]
        persistence = observations[origin]
        paired.append({"product": ev.PRODUCTS[0], "horizon": 1, "target": target.isoformat(),
                       "actual": actual, "direct": actual * (1.05 if i % 2 else .96),
                       "persistence": persistence})
        if origin == missing_eval_origin:
            continue
        issue = dt.datetime.combine(origin, dt.time(8), KST)
        for offset in (1, 2):
            for hour in (6, 12, 18):
                valid = dt.datetime.combine(origin + dt.timedelta(days=offset), dt.time(hour), KST)
                records.append(ForecastRecord(issue.astimezone(dt.timezone.utc), issue, offset * 24 + hour - 8,
                    valid.astimezone(dt.timezone.utc), valid, float(5 + offset + hour / 100), str(5 + offset + hour / 100),
                    4, "synthetic", True))
    archive = ForecastArchive("synthetic.zip", "a" * 64, "data.csv", "b" * 64, b"synthetic", "60_127",
        dt.date(2024, 1, 1), dt.date(2026, 1, 1), tuple(records), (), 0, {"completeness_claim": "synthetic"})
    return {"schemaVersion": 1, "products": products}, {"protocol": {"id": "direct-horizon-ridge-2026-10-06"},
        "pairedRows": paired}, [archive]


class KmaResidualEvalTests(unittest.TestCase):
    def test_fixed_fit_uses_weather_and_preserves_fallback_provenance(self):
        missing = dt.date(2025, 10, 1)
        prices, report, archives = synthetic_inputs(missing)
        prices["products"] = [p for p in prices["products"] if p["id"] == "cabbage"]
        with mock.patch.object(ev, "PRODUCTS", ("cabbage",)), mock.patch.object(ev, "HORIZONS", (1,)), \
             mock.patch.object(ev, "FAMILY_SIZE", 4), mock.patch.object(ev, "TRAIN_START", dt.date(2024, 1, 1)), \
             mock.patch.object(ev, "TRAIN_END", dt.date(2024, 5, 19)), mock.patch.object(ev, "EVAL_START", dt.date(2025, 9, 25)), \
             mock.patch.object(ev, "EVAL_END", dt.date(2025, 11, 3)):
            result = ev.build(prices, report, archives, hashes={"fixture": "synthetic"})
        self.assertEqual(result["fitAudits"]["cabbage/1"]["status"], "fitted")
        self.assertEqual(result["fitAudits"]["cabbage/1"]["distinctTrainingTargets"], 140)
        rows = result["predictions"]
        fallback = next(r for r in rows if r["origin"] == missing.isoformat())
        self.assertEqual(fallback["candidate"], fallback["direct"])
        self.assertEqual(fallback["fallbackReason"], "missing_forecast")
        self.assertEqual(fallback["archiveSha256"], ["a" * 64])
        self.assertEqual(len(rows), 40)
        self.assertEqual(result["comparisons"][0]["dm"]["9"]["n"], 40)
        self.assertEqual(len(result["comparisons"]), 4)

    def test_conflicting_valid_time_is_unavailable(self):
        origin = dt.date(2025, 9, 25)
        issue = dt.datetime.combine(origin, dt.time(8), KST)
        valid = dt.datetime.combine(origin + dt.timedelta(days=1), dt.time(6), KST)
        weather = {(issue, valid): [(1.0, "a"), (2.0, "b")]}
        feature, audit = ev._features(origin, weather)
        self.assertIsNone(feature)
        self.assertEqual(audit["status"], "conflicting_forecast")

    def test_quarantine_uses_parser_utc_marker_date_and_does_not_claim_issue_timestamp(self):
        origin = dt.date(2025, 10, 1)  # 08:00 KST is 23:00 UTC on September 30.
        feature, audit = ev._features(origin, {}, quarantined_issue_dates={dt.date(2025, 9, 30)})
        self.assertIsNone(feature)
        self.assertEqual(audit["status"], "quarantined_rows_present_but_forecast_missing")
        self.assertNotIn("issueKst", audit)
        self.assertEqual(audit["expectedIssueKst"], "2025-10-01T08:00:00+09:00")

    def test_report_origin_is_derived_when_absent_and_checked_when_present(self):
        prices, report, _ = synthetic_inputs()
        prices["products"] = [p for p in prices["products"] if p["id"] == "cabbage"]
        with mock.patch.object(ev, "PRODUCTS", ("cabbage",)):
            parsed = ev._direct_rows(report, prices)
            self.assertEqual(parsed[("cabbage", 1, dt.date(2024, 1, 1))]["origin"], dt.date(2023, 12, 31))
            report["pairedRows"][0]["origin"] = "2024-01-01"
            with self.assertRaisesRegex(ValueError, "origin/target/horizon mismatch"):
                ev._direct_rows(report, prices)

    def test_goal_gate_uses_equal_weighted_all_horizon_mape(self):
        grouped = {}
        for product, values in (("one", [0.] * 9 + [100.]), ("two", [6.] * 10), ("three", [6.] * 10)):
            for horizon, candidate_mape in enumerate(values, 1):
                grouped[f"{product}/{horizon}"] = [{"actual": 100., "candidate": 100. + candidate_mape,
                    "direct": 100., "persistence": 100.}]
        result = ev._point_estimate_goal_gate(grouped, ("one", "two", "three", "missing"))
        self.assertEqual(result["equalHorizonMapeByProduct"]["one"]["candidate"], 10.0)
        self.assertIsNone(result["equalHorizonMapeByProduct"]["missing"]["candidate"])
        self.assertEqual(result["under10PercentProductCount"], 2)
        self.assertEqual(result["under7PercentProductCount"], 2)
        self.assertTrue(result["goalMet"])

    def test_report_actual_and_persistence_must_match_prices(self):
        prices, report, archives = synthetic_inputs()
        prices["products"] = [p for p in prices["products"] if p["id"] == "cabbage"]
        report["pairedRows"][0]["actual"] += 1
        with mock.patch.object(ev, "PRODUCTS", ("cabbage",)):
            with self.assertRaisesRegex(ValueError, "actual does not align"):
                ev._direct_rows(report, prices)


if __name__ == "__main__":
    unittest.main()
