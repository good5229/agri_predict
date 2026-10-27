import datetime as dt
import unittest

from agri_data import modeling, volume_guard, volume_guard_experiment


class VolumeGuardTests(unittest.TestCase):
    def _synthetic(self, future_volume=1000.0):
        event = dt.date(2023, 1, 21)
        start = event - dt.timedelta(days=28)
        result = {}
        for offset in range(28):
            day = start + dt.timedelta(days=offset)
            result[day] = (500.0 + offset, 50000.0 + offset * 10)
        result[event] = (9000.0, 394.0)
        result[event + dt.timedelta(days=1)] = (550.0, future_volume)
        result[event + dt.timedelta(days=2)] = (560.0, 21000.0)
        return result

    def test_394kg_event_is_quarantined_and_future_volume_does_not_change_decision(self):
        initial = volume_guard.quarantine_dates(self._synthetic())
        changed_future = volume_guard.quarantine_dates(self._synthetic(future_volume=1e12))
        event = next(row for row in initial if row["date"] == "2023-01-21")
        changed_event = next(row for row in changed_future if row["date"] == "2023-01-21")
        self.assertEqual(event["volumeKg"], 394.0)
        self.assertEqual(event["priorObservedDays"], 28)
        self.assertEqual(event["prior28ObservedDayMedianKg"], 50135.0)
        self.assertLess(event["fractionOfMedian"], 0.01)
        self.assertEqual(event, changed_event)

    def test_price_gate_keeps_2019_like_normal_price_but_flags_2023_price_spike(self):
        start = dt.date(2019, 8, 3) - dt.timedelta(days=28)
        series = {start + dt.timedelta(days=i): (320.0 + (i % 3), 500000.0 + i)
                  for i in range(28)}
        normal_day = dt.date(2019, 8, 3)
        series[normal_day] = (300.0, 4180.0)
        self.assertIn(normal_day.isoformat(), {row["date"] for row in volume_guard.quarantine_dates(series)})
        self.assertNotIn(normal_day.isoformat(), {row["date"] for row in volume_guard.quarantine_dates(series, 4.0)})

        event_series = self._synthetic()
        event_day = dt.date(2023, 1, 21)
        self.assertIn(event_day.isoformat(), {row["date"] for row in volume_guard.quarantine_dates(event_series, 4.0)})

    def test_price_gate_is_strictly_prior_only_and_detects_low_price_extreme(self):
        target = dt.date(2023, 1, 21)
        start = target - dt.timedelta(days=28)
        series = {start + dt.timedelta(days=i): (100.0, 50000.0) for i in range(28)}
        series[target] = (400.0, 100.0)  # Inclusive upper threshold: exactly 4x.
        baseline = volume_guard.quarantine_dates(series, 4.0)
        later = dict(series)
        later[target + dt.timedelta(days=1)] = (1e9, 1e9)
        after_future_change = volume_guard.quarantine_dates(later, 4.0)
        self.assertEqual(next(row for row in baseline if row["date"] == target.isoformat()),
                         next(row for row in after_future_change if row["date"] == target.isoformat()))
        series[target] = (25.0, 100.0)  # Inclusive lower threshold: exactly 1/4.
        lower = volume_guard.quarantine_dates(series, 4.0)
        self.assertIn(target.isoformat(), {row["date"] for row in lower})
        self.assertEqual(next(row for row in lower if row["date"] == target.isoformat())[
                         "prior28ObservedDayMedianPrice"], 100.0)

    def test_default_price_gate_is_disabled_for_first_report_reproducibility(self):
        series = self._synthetic()
        default = volume_guard.quarantine_dates(series)
        explicit_none = volume_guard.quarantine_dates(series, None)
        self.assertEqual(default, explicit_none)
        self.assertEqual(default[0]["priceRatioThreshold"], None)

    def test_quality_series_removes_price_and_volume_from_all_later_inputs(self):
        raw = self._synthetic()
        flags = volume_guard.quarantine_dates(raw)
        filtered = volume_guard.quality_series(raw, flags)
        event = dt.date(2023, 1, 21)
        self.assertIn(event, raw)
        self.assertNotIn(event, filtered)
        origin_after_event = event
        source = volume_guard.source_day(filtered, origin_after_event)
        self.assertEqual(source, event - dt.timedelta(days=1))
        rows = volume_guard.persistence_predictions(filtered, [origin_after_event], (1,))
        self.assertEqual(rows[0]["sourceDate"], (event - dt.timedelta(days=1)).isoformat())
        self.assertEqual(rows[0]["predicted"], raw[event - dt.timedelta(days=1)][0])

    def test_same_date_forecast_cannot_use_future_event_price(self):
        raw = self._synthetic()
        filtered = volume_guard.quality_series(raw, volume_guard.quarantine_dates(raw))
        origin = dt.date(2023, 1, 20)
        rows = volume_guard.persistence_predictions(filtered, [origin], (1,))
        self.assertEqual(rows[0]["target"], "2023-01-21")
        self.assertEqual(rows[0]["sourceDate"], "2023-01-20")
        self.assertNotEqual(rows[0]["predicted"], raw[dt.date(2023, 1, 21)][0])

    def test_minimum_history_and_strict_threshold(self):
        start = dt.date(2023, 1, 1)
        series = {start + dt.timedelta(days=i): (100.0, 10000.0) for i in range(13)}
        series[start + dt.timedelta(days=13)] = (100.0, 50.0)
        self.assertEqual(volume_guard.quarantine_dates(series), [])
        for i in range(14, 28):
            series[start + dt.timedelta(days=i)] = (100.0, 10000.0)
        day = start + dt.timedelta(days=28)
        series[day] = (100.0, 100.0)
        self.assertEqual(volume_guard.quarantine_dates(series), [])  # exactly 1% is not below 1%.

    def test_paired_metrics_ignore_other_horizons_but_report_training_can_keep_them(self):
        origin = dt.date(2023, 1, 1)
        actuals = {origin + dt.timedelta(days=h): 100.0 for h in (1, 2, 5, 10)}
        rows = {}
        for horizon in (1, 2, 5, 10):
            key = (origin, origin + dt.timedelta(days=horizon), horizon)
            rows[key] = {"predicted": 100.0, "sourceDateParsed": origin}
        pairs = volume_guard_experiment._paired_rows(rows, rows, actuals, "baseline", {
            origin: (100.0, 1.0)
        })
        self.assertEqual({row["horizon"] for row in pairs}, {1, 5, 10})
        eligible = volume_guard_experiment._eligible_keys(rows, actuals)
        self.assertEqual({key[2] for key in eligible}, {1, 5, 10})
        self.assertIn(2, modeling.HORIZONS)  # Candidate training retains the full horizon set.

    def test_constant_dm_difference_is_unavailable_and_holm_retains_family(self):
        rows = [(dt.date(2020, 1, 1) + dt.timedelta(days=i), 0.1) for i in range(100)]
        test = volume_guard.dm_calendar_hac(rows, 9)
        self.assertEqual(test["status"], "inference_not_supported_constant_loss_difference")
        self.assertIsNone(test["pValue"])
        family = [test] + [{"pValue": 0.01 / (12 - i)} for i in range(1, 12)]
        volume_guard.holm_adjust(family)
        self.assertEqual(family[0]["holmPValue"], 1.0)
        self.assertEqual(family[0]["holmFamilySize"], 12)

    def test_event_exposure_uses_exact_ridge_source_lag_and_windows(self):
        event = volume_guard.EVENT_DAY
        # Event remains lag 7 at origin + 7 although newer prices exist.
        series = {event - dt.timedelta(days=100): (90.0, 10.0),
                  event - dt.timedelta(days=1): (99.0, 10.0)}
        for offset in range(0, 11):
            series[event + dt.timedelta(days=offset)] = (100.0 + offset, 1000.0)
        ridge_lag = volume_guard_experiment._event_input_exposure(
            series, event + dt.timedelta(days=7), "ridge_daily")
        self.assertFalse(ridge_lag["latestSource"])
        self.assertIn(7, ridge_lag["lags"])
        self.assertTrue(ridge_lag["any"])

        # At +10 the event is in 28/90-day windows only; the shorter lags resolve later dates.
        ridge_window = volume_guard_experiment._event_input_exposure(
            series, event + dt.timedelta(days=10), "ridge_daily")
        self.assertFalse(ridge_window["latestSource"])
        self.assertEqual(ridge_window["lags"], [])
        self.assertEqual(ridge_window["windows"], [28, 90])
        self.assertFalse(volume_guard_experiment._event_input_exposure(
            series, event + dt.timedelta(days=10), "baseline")["any"])

    def test_dual_loss_conclusion_requires_both_holm_significant_and_reports_28_day_sensitivity(self):
        tests = []
        for bandwidth in (9, 28):
            for method in volume_guard_experiment.METHODS:
                for horizon in volume_guard.HORIZONS:
                    for loss in volume_guard_experiment.LOSSES:
                        tests.append({"bandwidthCalendarDays": bandwidth, "method": method,
                                      "horizon": horizon, "loss": loss,
                                      "status": "tested", "meanLossDifference": -1.0,
                                      "holmPValue": 0.01})
        result = volume_guard_experiment._dual_loss_conclusions(tests)
        self.assertEqual(len(result), 12)
        self.assertEqual({row["bandwidthCalendarDays"] for row in result}, {9, 28})
        self.assertTrue(all(row["verdict"] == "improved" for row in result))
        tests[0]["holmPValue"] = 0.9
        result = volume_guard_experiment._dual_loss_conclusions(tests)
        row = next(row for row in result if row["bandwidthCalendarDays"] == 9
                   and row["method"] == tests[0]["method"] and row["horizon"] == tests[0]["horizon"])
        self.assertEqual(row["verdict"], "no_evidence")


if __name__ == "__main__":
    unittest.main()
