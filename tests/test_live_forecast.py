import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

from agri_data import collector, coverage, daily_training, dashboard, live_forecast, modeling


class LiveForecastTests(unittest.TestCase):
    def _series(self, start=dt.date(2020, 1, 1), days=120):
        return {start + dt.timedelta(days=i): (100.0 + i, 10.0 + i) for i in range(days)}

    def _radish_event_product(self, event_day=dt.date(2023, 1, 21), prior_end_offset=1):
        prior_end = event_day - dt.timedelta(days=prior_end_offset)
        prior_start = prior_end - dt.timedelta(days=27)
        observations = []
        for offset in range(28):
            day = prior_start + dt.timedelta(days=offset)
            observations.append({"date": day.isoformat(), "actual": 100.0 + offset,
                                 "volumeKg": 50000.0})
        observations.append({"date": event_day.isoformat(), "actual": 2000.0, "volumeKg": 394.0})
        return {"id": "radish", "observations": observations}

    def test_radish_guard_uses_previous_eligible_source_and_removes_target_label(self):
        event = dt.date(2023, 1, 21)
        product = self._radish_event_product(event)
        cutoff = event
        raw, eligible, metadata = live_forecast._forecast_input_series(product, cutoff)
        result = live_forecast._forecast_product(product, event + dt.timedelta(days=1), cutoff)
        self.assertIn(event, raw)
        self.assertNotIn(event, eligible)
        self.assertEqual(metadata["lastRawObservationDate"], event.isoformat())
        self.assertEqual(metadata["lastEligibleSourceDate"], (event - dt.timedelta(days=1)).isoformat())
        self.assertEqual(result["lastObservedPriceDate"], (event - dt.timedelta(days=1)).isoformat())
        self.assertEqual(result["sourceDate"], (event - dt.timedelta(days=1)).isoformat())
        self.assertEqual(result["freshnessState"], "fresh")
        self.assertEqual(result["qualityGuard"]["flaggedDateCount"], 1)
        _, target_days = live_forecast._strict_samples(eligible, cutoff)
        self.assertNotIn(event, target_days)
        self.assertEqual(result["method"], "persistence_fallback")
        self.assertTrue(all(row["sourceDate"] == (event - dt.timedelta(days=1)).isoformat()
                            for row in result["predictions"]))

    def test_other_product_is_unchanged_by_radish_guard(self):
        product = self._radish_event_product()
        product["id"] = "cabbage"
        cutoff = dt.date(2023, 1, 21)
        raw, eligible, metadata = live_forecast._forecast_input_series(product, cutoff)
        self.assertEqual(eligible, raw)
        self.assertFalse(metadata["enabled"])
        self.assertEqual(metadata["flaggedDateCount"], 0)
        self.assertIn(cutoff, eligible)

    def test_guard_never_reads_observations_after_verified_cutoff(self):
        event = dt.date(2023, 1, 21)
        product = self._radish_event_product(event)
        product["observations"].append({"date": (event + dt.timedelta(days=1)).isoformat(),
                                         "actual": 1e9, "volumeKg": 1e9})
        cutoff = event - dt.timedelta(days=1)
        raw, eligible, metadata = live_forecast._forecast_input_series(product, cutoff)
        self.assertNotIn(event, raw)
        self.assertEqual(raw, eligible)
        self.assertEqual(metadata["flaggedDateCount"], 0)
        self.assertEqual(metadata["lastRawObservationDate"], cutoff.isoformat())
        self.assertEqual(metadata["observationsAfterCutoffIgnoredCount"], 2)

    def test_guarded_source_older_than_fourteen_days_is_stale(self):
        event = dt.date(2023, 1, 21)
        product = self._radish_event_product(event, prior_end_offset=15)
        result = live_forecast._forecast_product(product, event + dt.timedelta(days=1), event)
        self.assertEqual(result["qualityGuard"]["flaggedDateCount"], 1)
        self.assertEqual(result["qualityGuard"]["lastRawObservationDate"], event.isoformat())
        self.assertEqual(result["lastObservedPriceDate"], (event - dt.timedelta(days=15)).isoformat())
        self.assertEqual(result["freshnessState"], "stale")
        self.assertEqual(result["status"], "unavailable_no_fresh_price")
        self.assertEqual(result["predictions"], [])

    def test_training_features_exclude_values_on_historical_origin(self):
        series = self._series()
        target = dt.date(2020, 4, 20)
        origin = target - dt.timedelta(days=1)
        before, _ = live_forecast._strict_samples(series, dt.date(2020, 4, 29))
        row_before = next(row for row in before if row[0] == target)
        changed = dict(series)
        changed[origin] = (999999.0, 999999.0)
        after, _ = live_forecast._strict_samples(changed, dt.date(2020, 4, 29))
        row_after = next(row for row in after if row[0] == target)
        self.assertEqual(row_before[1], row_after[1])
        self.assertEqual(row_before[2], row_after[2])

    def test_strict_samples_match_filtered_asof_series_and_keep_horizon(self):
        start = dt.date(2020, 1, 1)
        series = {start + dt.timedelta(days=i): (100.0 + i, 10.0 + i)
                  for i in (0, 1, 7, 8, 14, 15, 22, 29, 30, 50, 89, 90, 91, 120, 121, 150, 179)}
        cutoff = start + dt.timedelta(days=179)
        samples, _ = live_forecast._strict_samples(series, cutoff)
        expected = []
        expected_days = set()
        for target, (price, _) in series.items():
            if target > cutoff:
                continue
            for horizon in modeling.HORIZONS:
                origin = target - dt.timedelta(days=horizon)
                eligible = {day: value for day, value in series.items() if day < origin}
                feature = modeling._feature_row(eligible, origin, target, horizon, {})
                if feature is None:
                    continue
                source_day, vector = feature
                expected.append((target, vector, modeling.math.log(price / eligible[source_day][0])))
                expected_days.add(target)
        self.assertEqual(samples, expected)
        self.assertEqual(set(target for target, _, _ in samples), expected_days)

    def test_live_feature_uses_cutoff_day_as_final_input(self):
        issue = dt.date(2020, 5, 1)
        product = {"id": "cabbage", "observations": [
            {"date": (issue - dt.timedelta(days=1)).isoformat(), "actual": 100, "volumeKg": 10},
            {"date": issue.isoformat(), "actual": 9999, "volumeKg": 9999},
        ]}
        series = modeling._observed_series(product)
        cache = {}
        vector = modeling._feature_row(series, issue - dt.timedelta(days=1),
                                       issue + dt.timedelta(days=7), 7, cache)[1]
        baseline = modeling._feature_row({issue - dt.timedelta(days=1): (100.0, 10.0)},
                                         issue - dt.timedelta(days=1),
                                         issue + dt.timedelta(days=7), 7, {})[1]
        self.assertEqual(vector, baseline)

    def test_insufficient_history_uses_persistence_and_freshness_is_explicit(self):
        issue = dt.date(2020, 2, 1)
        product = {"id": "cabbage", "observations": [
            {"date": (issue - dt.timedelta(days=1)).isoformat(), "actual": 123.0, "volumeKg": 50.0}
        ]}
        result = live_forecast._forecast_product(product, issue, issue - dt.timedelta(days=1))
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["method"], "persistence_fallback")
        self.assertEqual(result["freshnessState"], "fresh")
        self.assertEqual(len(result["predictions"]), 10)
        self.assertTrue(all(row["predictedPrice"] == 123.0 for row in result["predictions"]))

    def test_stale_series_does_not_issue_persistence_values(self):
        issue = dt.date(2020, 2, 1)
        product = {"id": "pepper", "observations": [
            {"date": (issue - dt.timedelta(days=dashboard.MAX_SOURCE_AGE_DAYS + 1)).isoformat(),
             "actual": 123.0, "volumeKg": 50.0}
        ]}
        result = live_forecast._forecast_product(product, issue, issue - dt.timedelta(days=1))
        self.assertEqual(result["freshnessState"], "stale")
        self.assertEqual(result["status"], "unavailable_no_fresh_price")
        self.assertEqual(result["predictions"], [])

    def test_cutoff_scope_states_include_latest_failed_state(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.sqlite"
            db = collector.open_db(str(path))
            day = "2026-10-05"
            endpoint = f"{collector.PORTAL}/B552845/katOrigin/trades"
            params = {"cond[trd_clcln_ymd::EQ]": day, "cond[whsl_mrkt_cd::EQ]": "110001",
                      "cond[gds_lclsf_cd::EQ]": "10", "cond[gds_mclsf_cd::EQ]": "01"}
            job_id = collector.begin_job(db, "prices", {"fixture": True})
            key = collector.task_key("prices_origin", endpoint, params)
            db.execute("""INSERT INTO planned_queries(job_id,task_key,dataset,source_url,query_json,state,
                         row_count,epoch_id) VALUES(?,?,?,?,?,'failed',NULL,'e1')""",
                       (job_id, key, "prices_origin", endpoint, collector.json_text(params)))
            db.commit()
            db.close()
            readonly = coverage.open_readonly(str(path))
            try:
                states = live_forecast._coverage_states(readonly, dt.date.fromisoformat(day))
            finally:
                readonly.close()
            self.assertEqual(len(states), len(daily_training.EXPECTED_SCOPES))
            cabbage = next(row for row in states if row["market"] == "110001" and row["product"] == "cabbage")
            self.assertEqual(cabbage["latestState"], "failed")
            self.assertEqual(cabbage["selectedOutcome"], "no_valid_selected_response")
            missing = next(row for row in states if row["market"] == "110008" and row["product"] == "pepper")
            self.assertEqual(missing["selectedOutcome"], "no_manifest_scope")

    def test_publication_is_gated_until_ten_kst(self):
        issue_time = dt.datetime(2026, 10, 6, 9, 59, tzinfo=ZoneInfo("Asia/Seoul"))
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(live_forecast.LiveForecastError, "10:00 KST"):
                live_forecast.issue_live_forecast(root=temp, db_path="missing.sqlite",
                                                   today=issue_time.date(), issue_time=issue_time)

    def test_publication_timestamp_must_remain_on_issue_day(self):
        zone = ZoneInfo("Asia/Seoul")
        issue_day = dt.date(2026, 10, 6)
        live_forecast._validate_publication_window(dt.datetime(2026, 10, 6, 10, 1, tzinfo=zone), issue_day)
        with self.assertRaisesRegex(live_forecast.LiveForecastError, "rolled over"):
            live_forecast._validate_publication_window(dt.datetime(2026, 10, 7, 0, 1, tzinfo=zone), issue_day)


if __name__ == "__main__":
    unittest.main()
