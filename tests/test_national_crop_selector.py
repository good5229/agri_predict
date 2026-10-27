import datetime as dt
import json
from pathlib import Path

import pytest

from agri_data import national_crop_selector as selector


def _source(actuals):
    return {"observations": {p: {day.isoformat(): {"price": value} for day, value in series.items()}
                             for p, series in actuals.items()}}


def _selection_rows(*, short_one_horizon=False, add_late_rows=False):
    actuals = {p: {} for p in selector.PRODUCTS}
    rows = []
    for day_offset in range(30):
        target = dt.date(2026, 4, 1) + dt.timedelta(days=day_offset)
        for product in selector.PRODUCTS:
            actuals[product][target] = 100.0
            for horizon in selector.HORIZONS:
                if short_one_horizon and product == "radish" and horizon == 1 and day_offset == 29:
                    continue
                origin = target - dt.timedelta(days=horizon)
                preds = {m: 100.0 for m in selector.SELECTION_METHODS}
                if product == "cabbage":
                    preds["ridge_alpha_100"] = 105.0
                rows.append({"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                    "horizon": horizon, "cutoff": (origin - dt.timedelta(days=2)).isoformat(),
                    "predictions": preds})
    if add_late_rows:
        for day_offset in range(11):
            target = dt.date(2026, 6, 20) + dt.timedelta(days=day_offset)
            for product in selector.PRODUCTS:
                actuals[product][target] = 100000.0
                for horizon in selector.HORIZONS:
                    origin = target - dt.timedelta(days=horizon)
                    rows.append({"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                        "horizon": horizon, "cutoff": (origin - dt.timedelta(days=2)).isoformat(),
                        "predictions": {m: 100000.0 for m in selector.SELECTION_METHODS}})
    return rows, _source(actuals)


def _candidate_doc(forecasts):
    doc = {"protocolId": selector.PROTOCOL_ID, "phase": "forecast_only", "targetValuesAttached": False,
        "codeSha256": selector._sha(Path(selector.__file__).read_bytes()), "plannedForecastCount": len(forecasts),
        "forecasts": forecasts}
    doc["manifestSha256"] = selector._hash(doc)
    return doc


def test_crop_selection_uses_common_pairs_tie_priority_and_ignores_june_20_30():
    rows, source = _selection_rows(add_late_rows=True)
    result = selector._selection(rows, source)
    assert result["choices"]["cabbage"]["method"] == "ridge_alpha_10"
    for product in selector.PRODUCTS:
        assert result["diagnostics"][product]["selectionWindow"] == ["2026-04-01", "2026-06-19"]
        assert result["diagnostics"][product]["sampleCountsByHorizon"]["1"] == 30
    assert result["plannedPairs"] == 30 * len(selector.PRODUCTS) * len(selector.HORIZONS)
    without_late, source_without_late = _selection_rows(add_late_rows=False)
    result_without_late = selector._selection(without_late, source_without_late)
    assert result["choices"] == result_without_late["choices"]
    first_jul_target_cutoff = selector.FRESH_START - dt.timedelta(days=10 + 2)
    assert first_jul_target_cutoff == selector.SELECTION_END
    assert selector.SELECTION_END == dt.date(2026, 6, 19)


def test_any_horizon_below_30_forces_ridge100_for_that_crop():
    rows, source = _selection_rows(short_one_horizon=True)
    result = selector._selection(rows, source)
    assert result["choices"]["radish"] == {"method": "ridge_alpha_100", "status": "fallback",
                                            "fallbackReason": "one_or_more_horizons_below_30_common_pairs"}


def test_selected_forecast_fallback_and_answer_free_manifest_validation():
    raw = {"product": "cabbage", "origin": "2026-06-30", "target": "2026-07-01", "horizon": 1,
           "cutoff": "2026-06-28", "predictions": {"ridge_alpha_100": 123.0, "ridge_alpha_10": 122.0,
           "calendar_sarima": None, "fixed_blend_50_50": 124.0, "persistence": 125.0},
           "lineage": {"calendar_sarima": {"fallback": True}}}
    row = selector._make_candidate_row(raw, {"method": "calendar_sarima"})
    assert row["predictions"]["crop_selected"] == 123.0
    assert row["lineage"]["fallbackReason"] == "selected_method_invalid:calendar_sarima"
    assert "actual" not in row and "targetValue" not in row
    doc = _candidate_doc([row])
    selector.verify_manifest(doc)
    tampered = dict(doc)
    tampered["forecasts"] = [dict(row, actual=4.0)]
    with pytest.raises(ValueError, match="manifest hash mismatch"):
        selector.verify_manifest(tampered)


def test_input_byte_hash_tampering_rejected_without_external_data(tmp_path, monkeypatch):
    path = tmp_path / "development.json"
    path.write_text('{"bad":true}', encoding="utf-8")
    monkeypatch.setitem(selector.frozen.MANIFEST_PATHS, "development", str(path))
    with pytest.raises(ValueError, match="byte SHA-256 mismatch"):
        selector.frozen._load_frozen_manifest("development")


def test_replay_rejects_changed_selected_predictions(monkeypatch):
    row = selector._make_candidate_row({"product": "cabbage", "origin": "2026-06-30", "target": "2026-07-01",
        "horizon": 1, "cutoff": "2026-06-28", "predictions": {"ridge_alpha_100": 10.0,
        "calendar_sarima": 11.0, "persistence": 12.0}, "lineage": {}}, {"method": "ridge_alpha_100"})
    original = _candidate_doc([row])
    changed_row = dict(row)
    changed_row["predictions"] = {**row["predictions"], "crop_selected": 999.0}
    rebuilt = _candidate_doc([changed_row])
    monkeypatch.setattr(selector, "build_manifest", lambda *_: rebuilt)
    with pytest.raises(ValueError, match="replay mismatch"):
        selector._verify_replay(original, {}, {}, {})


def test_fixed_dm_family_and_joint_statuses():
    assert len(selector.PRODUCTS) * len(selector.HORIZONS) * len(selector.COMPARATORS) * 2 == 240
    diffs = {dt.date(2026, 7, 1) + dt.timedelta(days=i): float(i % 5) for i in range(40)}
    unavailable = selector.frozen._dm({day: 0.1 for day in diffs}, 9)
    assert unavailable["status"] == "inference_unavailable_nonpositive_variance"
    assert unavailable["pValue"] == 1.0
    assert selector.frozen._dm(dict(list(diffs.items())[:29]), 9)["pValue"] == 1.0


def test_create_only_output_preserves_existing_artifact(tmp_path):
    path = tmp_path / "forecast.json"
    selector._write_json_create_only(path, {"phase": "forecast_only"})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        selector._write_json_create_only(path, {"phase": "scored"})
    assert path.read_bytes() == before
