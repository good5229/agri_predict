import datetime as dt

from agri_data.price_path_identifiability import analyze_synthetic_collision
from agri_data import price_path_identifiability as mod


def test_interior_price_swap_collides_in_frozen_106_but_not_ordered_calendar_vector():
    result = analyze_synthetic_collision()
    assert result["realDataRead"] is False
    assert result["noFit"] is True
    assert result["frozenFeatureCount"] == 106
    assert result["p0"]["sameAcrossHistories"] is True
    assert result["frozenFeatureMaxAbsDelta"] <= result["roundOffTolerance"]
    assert result["frozenFeaturesChangedAboveTolerance"] == []
    assert set(result["orderedCalendarValueChanges"]) == set(result["swappedDays"])
    assert result["orderedCalendarVectorDistinguishesHistories"] is True
    assert result["orderedCalendarMaskChanges"] == []
    assert result["existingSimilarTrajectoryRepresentation"]["usesExactOrderedDailyPath"] is False


def test_anchor_audit_matches_original_lags_and_library_anchor_slope_schema():
    anchors = analyze_synthetic_collision()["unchangedAnchors"]
    assert anchors == {
        "direct40Lags": {
            "lag1": "2026-05-29", "lag7": "2026-05-23",
            "lag28": "2026-05-02", "lag90": "2026-03-01",
        },
        "libraryBaseAnchors": {
            "p1PreviousObservation": "2026-05-29",
            "p7": "2026-05-23", "p30": "2026-04-30",
        },
        "expandedAnchors": {
            "anchor3d": "2026-05-27", "anchor14d": "2026-05-16",
            "anchor60d": "2026-03-31",
        },
        "expandedSlopeEndpoints": {
            "slope14dStart": "2026-05-17", "slope60dStart": "2026-04-01",
            "slope14dEnd": "2026-05-30", "slope60dEnd": "2026-05-30",
        },
    }


def test_snapshot_trim_skips_future_values_before_parsing_or_retaining_them():
    class MustNotRead:
        def __float__(self):
            raise AssertionError("future price/volume was read")

    raw = {"schemaVersion": 1, "products": [
        {"id": product, "observations": [
            {"date": "2024-01-01", "actual": "100", "volumeKg": "10"},
            {"date": "2024-01-02", "actual": MustNotRead(), "volumeKg": MustNotRead(),
             "marketCodes": MustNotRead()},
        ]} for product in mod.PRODUCTS
    ]}
    trimmed, audit = mod._trim_snapshot(raw, dt.date(2024, 1, 1))
    assert all([row["date"] for row in product["observations"]] == ["2024-01-01"]
               for product in trimmed["products"])
    assert all(row["skippedLaterDates"] == 1 for row in audit.values())


def test_swap_status_marks_missing_pair_without_substituting_another_date():
    source_day = dt.date(2024, 2, 1)
    first = source_day - dt.timedelta(days=5)
    second = source_day - dt.timedelta(days=6)
    history = {source_day: {"price": 100.0}, first: {"price": 90.0}}
    status, reported_first, reported_second = mod._swap_status(history, source_day)
    assert status == "missing_pair"
    assert (reported_first, reported_second) == (first, second)


def _history_with_p0_age(origin, age):
    source_day = origin - dt.timedelta(days=age)
    records = {}
    for i in range(121):
        day = source_day - dt.timedelta(days=120 - i)
        records[day] = {"price": 100.0 + i, "volume": 500.0 + i,
                        "rowCount": 10, "markets": frozenset(mod.library.MARKET_CODES)}
    return records


def test_snapshot_swap_dates_are_relative_to_p0_and_age_is_relative_to_origin():
    origin = dt.date(2024, 6, 1)
    records = _history_with_p0_age(origin, 14)
    row = mod._sample_record("cabbage", origin, {}, records)
    source_day = origin - dt.timedelta(days=14)
    assert row["sourceAgeDays"] == 14
    assert row["swapDay5"] == (source_day - dt.timedelta(days=5)).isoformat()
    assert row["swapDay6"] == (source_day - dt.timedelta(days=6)).isoformat()


def test_snapshot_age_15_is_stale_while_age_14_is_eligible():
    origin = dt.date(2024, 6, 1)
    at_limit = mod._sample_record("cabbage", origin, {}, _history_with_p0_age(origin, 14))
    beyond = mod._sample_record("cabbage", origin, {}, _history_with_p0_age(origin, 15))
    assert at_limit["sourceAgeDays"] == 14
    assert at_limit["swapStatus"] in {"eligible", "missing_pair", "equal_price", "feature_unavailable"}
    assert beyond["sourceAgeDays"] == 15
    assert beyond["swapStatus"] == "stale_source"


def test_feature_unavailable_is_classified_without_aborting_sample(monkeypatch):
    origin = dt.date(2024, 6, 1)
    records = _history_with_p0_age(origin, 1)
    monkeypatch.setattr(mod, "_feature_vector", lambda *a, **k: (_ for _ in ()).throw(
        mod.FeatureUnavailable("synthetic unavailable")))
    row = mod._sample_record("cabbage", origin, {}, records)
    assert row["swapStatus"] == "feature_unavailable"
    assert row["frozen106MaxAbsDelta"] is None


def test_snapshot_builder_keeps_the_full_660_sample_fixed_grid():
    raw = {"schemaVersion": 1,
           "products": [{"id": product, "observations": []} for product in mod.PRODUCTS]}
    result = mod.build_snapshot_diagnostic(raw, source_sha256="synthetic")
    assert result["sampleProtocol"]["plannedSamples"] == 660
    assert len(result["samples"]) == 660
    assert result["sampleProtocol"]["maximumReadDate"] == "2024-11-30"
    assert result["samples"][0]["origin"] == "2014-01-01"
    assert result["samples"][-1]["origin"] == "2024-12-01"
    for summary in result["perCropCounts"].values():
        assert summary["planned"] == 132
        assert summary["noSource"] == 132
        assert summary["eligible"] == summary["staleSource"] == 0
    assert all("actual" not in row and "price" not in row and "volume" not in row
               for row in result["samples"])
