import datetime as dt

from agri_data import supply_pressure_reentry_screen as screen


def _seasonal_gap_series():
    values = {}
    for year in (2021, 2022, 2023):
        values[dt.date(year, 8, 16)] = (90.0 + year - 2021, 12.0)
    # The 7-day reference and current week exist; the latest observation at
    # or before the 28-day reference date is too old.
    values[dt.date(2024, 7, 1)] = (100.0, 10.0)
    for day in range(7, 16):
        values[dt.date(2024, 8, day)] = (101.0, 11.0)
    return values


def test_reentry_only_for_28_day_failure_and_future_values_do_not_leak():
    origin = dt.date(2024, 8, 15)
    target = origin + dt.timedelta(days=1)
    series = _seasonal_gap_series()
    built = screen.issue_feature(series, origin, target, 1)
    assert built is not None and built[2] is True
    features = built[0]
    assert len(features) == 15
    assert features[6] == 0.0  # priceChange28
    assert features[7] == 0.0  # logMeanVolume7Minus28
    assert features[9] == 0.0  # horizonTimesLogMeanVolume7Minus28
    assert features[11] > 0  # coverage7 is retained
    assert features[12] == 0.0  # coverage28 is neutralized
    assert features[13] > 0.0  # log mean positive volume over observed prior 7-day rows
    assert features[14] == 1.0  # missing28 indicator

    scaled_prior_week = dict(series)
    for day, (price, volume) in tuple(scaled_prior_week.items()):
        if origin - dt.timedelta(days=7) <= day < origin:
            scaled_prior_week[day] = (price, volume * 1000)
    scaled = screen.issue_feature(scaled_prior_week, origin, target, 1)
    assert scaled is not None and scaled[0][13] > features[13]

    changed_future = dict(series)
    changed_future[target] = (999999.0, 999999.0)
    changed_future[dt.date(2024, 9, 1)] = (888888.0, 888888.0)
    assert screen.issue_feature(changed_future, origin, target, 1) == built


def test_no_reentry_when_failure_is_not_28_day_history():
    origin = dt.date(2024, 8, 15)
    target = origin + dt.timedelta(days=1)
    series = _seasonal_gap_series()
    # Remove all three seasonal anchors: 28-day history still fails, but the
    # independent seasonal requirement prevents entry into the fallback.
    series = {day: value for day, value in series.items() if day.year == 2024}
    assert screen.issue_feature(series, origin, target, 1) is None

    # With complete daily history the original 13 features are preserved and
    # the reentry indicator is zero.
    full = {}
    day = dt.date(2021, 1, 1)
    while day <= dt.date(2024, 12, 31):
        full[day] = (100.0, 10.0)
        day += dt.timedelta(days=1)
    result = screen.issue_feature(full, origin, target, 1)
    assert result is not None and result[2] is False
    assert len(result[0]) == 15
    assert result[0][13] > 0.0 and result[0][14] == 0.0


def test_prior_targets_are_distinct_and_gate_only_counts_reentry_pairs(monkeypatch):
    issue_origin = dt.date(2017, 1, 8)
    series = {dt.date(2017, 1, day): (100.0, 1.0) for day in range(1, 8)}
    calls = []

    def fake_issue_feature(_series, origin, target, horizon):
        calls.append((origin, target, horizon))
        return ((0.0,) * 15, origin - dt.timedelta(days=1), True)

    monkeypatch.setattr(screen, "issue_feature", fake_issue_feature)
    found = screen.prior_reentry_target_days(series, issue_origin)
    assert found == {dt.date(2017, 1, day) for day in range(2, 8)}
    assert all(target < issue_origin for _, target, _ in calls)
    assert len(calls) > len(found)  # overlapping pairs do not inflate the day count


def test_history_minimum_applies_only_to_pepper_reentry_target_pairs(monkeypatch):
    checked_origin = screen.ORIGINS[0]
    rows = [
        {"productId": "pepper", "origin": checked_origin.isoformat(),
         "reentryBranch": True, "targetStatus": "present"},
        {"productId": "pepper", "origin": screen.ORIGINS[1].isoformat(),
         "reentryBranch": True, "targetStatus": "missing_observation"},
        {"productId": "pepper", "origin": screen.ORIGINS[2].isoformat(),
         "reentryBranch": False, "targetStatus": "present"},
    ]
    calls = []

    def count_targets(_series, origin, _cache=None):
        calls.append(origin)
        return {dt.date(2017, 1, day) for day in range(1, 31)}

    monkeypatch.setattr(screen, "prior_reentry_target_days", count_targets)
    checked = screen.pepper_reentry_history_by_origin(rows, {})
    assert calls == [checked_origin]
    assert checked == [{"origin": checked_origin.isoformat(),
                        "distinctPriorReentryTargets": 30, "minimum": 30, "passed": True}]
