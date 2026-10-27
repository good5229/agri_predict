import datetime as dt
import math

from agri_data import modeling
from agri_data import full_history_feature_library as library
from agri_data import robust_price_features as mod


def _history(n=140):
    end = dt.date(2024, 6, 30)
    records = {}
    series = {}
    for index in range(n):
        day = end - dt.timedelta(days=n - index - 1)
        price = 90.0 + index * 0.37 + (index % 9) * 0.11
        volume = 500.0 + (index % 11) * 13.0
        row = {"price": price, "volume": volume, "rowCount": 6 + index % 5,
               "markets": frozenset(library.MARKET_CODES)}
        records[day] = row
        series[day] = (price, volume)
    return end, records, series


def test_output_preserves_frozen_106_prefix_and_appends_nine_named_features():
    cutoff, records, series = _history()
    target, horizon = cutoff + dt.timedelta(days=4), 3
    vector, p0, audit = mod.features(records, series, target, horizon)
    assert vector is not None and p0 is not None
    expected_cutoff = target - dt.timedelta(days=horizon + 1)
    causal_records = {day: row for day, row in records.items() if day <= expected_cutoff}
    causal_series = {day: row for day, row in series.items() if day <= expected_cutoff}
    original40 = modeling._feature_row(causal_series, expected_cutoff, target, horizon, {})
    days = sorted(causal_series)
    prices = {day: pair[0] for day, pair in causal_series.items()}
    expanded, expected_p0, _ = library.features(causal_records, target, horizon, days, prices)
    old106 = list(original40[1]) + [value for name, value in zip(library.FEATURE_NAMES, expanded)
                                   if name not in mod.OMITTED_LIBRARY_FEATURES]
    assert vector[:106] == old106
    assert len(vector) == 115 == len(mod.FEATURE_NAMES)
    assert p0 == expected_p0
    assert mod.FEATURE_NAMES[:106] == mod.ORIGINAL_106_NAMES
    assert mod.FEATURE_NAMES[106:] == mod.ROBUST_FEATURE_NAMES
    assert audit["originalFeatureCount"] == 106
    assert audit["addedFeatureCount"] == 9


def test_robust_window_values_follow_raw_median_linear_log_iqr_and_log_mad():
    cutoff = dt.date(2024, 6, 30)
    values = [100.0, 110.0, 130.0, 170.0, 210.0, 260.0, 400.0]
    records = {}
    series = {}
    for offset, price in enumerate(values):
        day = cutoff - dt.timedelta(days=6 - offset)
        records[day] = {"price": price, "volume": 20.0 + offset,
                        "rowCount": 1, "markets": frozenset()}
        series[day] = (price, 20.0 + offset)
    vector, p0, audit = mod.features(records, series, cutoff + dt.timedelta(days=2), 1)
    assert vector is not None and p0 == values[-1]
    logs = [math.log(value) for value in values]
    median_log = sorted(logs)[len(logs) // 2]
    expected = [math.log(p0 / sorted(values)[len(values) // 2]),
                mod._linear_quantile(logs, .75) - mod._linear_quantile(logs, .25),
                sorted(abs(value - median_log) for value in logs)[len(logs) // 2]]
    assert vector[106:109] == expected
    assert audit["windowAudit"]["7"] == {"positivePriceVolumeDays": 7,
                                         "minimumRequired": 3, "supported": True}


def test_linear_quantile_uses_interpolation_between_order_statistics():
    assert mod._linear_quantile([1.0, 2.0, 3.0, 4.0], .25) == 1.75
    assert mod._linear_quantile([1.0, 2.0, 3.0, 4.0], .75) == 3.25


def test_unsupported_window_zeroes_all_three_values_and_keeps_support_audit():
    cutoff = dt.date(2024, 6, 30)
    records = {}
    series = {}
    for day, price in ((cutoff - dt.timedelta(days=1), 100.0),
                       (cutoff, 110.0)):
        row = {"price": price, "volume": 12.0, "rowCount": 1,
               "markets": frozenset()}
        records[day] = row
        series[day] = (price, 12.0)
    vector, p0, audit = mod.features(records, series, cutoff + dt.timedelta(days=2), 1)
    assert vector is not None and p0 == 110.0
    assert vector[106:109] == [0.0, 0.0, 0.0]
    assert audit["windowAudit"]["7"] == {"positivePriceVolumeDays": 2,
                                         "minimumRequired": 3, "supported": False}


def test_future_rows_are_removed_before_any_feature_value_access():
    class MustNotRead:
        def __float__(self):
            raise AssertionError("post-cutoff value accessed")

    cutoff, records, series = _history()
    target = cutoff + dt.timedelta(days=2)
    future = cutoff + dt.timedelta(days=1)
    records[future] = {"price": MustNotRead(), "volume": MustNotRead(),
                       "rowCount": MustNotRead(), "markets": MustNotRead()}
    series[future] = (MustNotRead(), MustNotRead())
    vector, p0, audit = mod.features(records, series, target, 1)
    assert vector is not None and p0 is not None
    assert audit["featureCutoff"] == cutoff.isoformat()


def test_witness_keeps_all_106_old_features_equal_but_changes_robust_block():
    result = mod.analyze_synthetic_witness()
    assert result["realDataRead"] is False
    assert result["noFit"] is True
    assert result["sameP0"] is True
    assert result["oldFeatureCount"] == 106
    assert result["newFeatureCount"] == 115
    assert result["oldFeatureMaxAbsDelta"] <= 1e-12
    assert result["otherWindowAuditSame"] is True
    assert result["robustFeatureDeltas"]["price_log_p0_to_median_90d"] > 1e-3
    assert result["robustFeatureDeltas"]["price_log_mad_90d"] > 1e-3


def test_snapshot_diagnostic_cap_drops_later_values_before_reading_them():
    class MustNotRead:
        def __float__(self):
            raise AssertionError("post-cap value accessed")

    raw = {"schemaVersion": 1, "products": [
        {"id": crop, "observations": [
            {"date": "2024-11-30", "actual": "100", "volumeKg": "20"},
            {"date": "2024-12-01", "actual": MustNotRead(), "volumeKg": MustNotRead()},
        ]} for crop in mod.PRODUCTS]}
    trimmed, audit = mod._trim_snapshot(raw)
    assert all([row["date"] for row in product["observations"]] == ["2024-11-30"]
               for product in trimmed["products"])
    assert all(item["skippedLaterDates"] == 1 for item in audit.values())
    assert mod.MAX_READ_DATE == dt.date(2024, 11, 30)
    assert len(mod._sample_origins()) == 132


def test_snapshot_provenance_hashes_transitive_narrow_feature_helper():
    raw = {"schemaVersion": 1,
           "products": [{"id": crop, "observations": []} for crop in mod.PRODUCTS]}
    report = mod._diagnostic_from_raw(raw, source_sha256="synthetic")
    assert report["inputSha256"]["narrowFeatureHelper"] == mod._hash(
        mod.Path(mod.narrow.__file__))
