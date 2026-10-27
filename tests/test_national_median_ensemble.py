import datetime as dt
import json
from pathlib import Path

import pytest

from agri_data import national_median_ensemble as median


def _forecast(*, ridge=100.0, sarima=100.0, persistence=900.0):
    return {"product": "cabbage", "origin": "2026-01-01", "target": "2026-01-02", "horizon": 1,
            "cutoff": "2025-12-30",
            "predictions": {"ridge_alpha_100": ridge, "calendar_sarima": sarima, "persistence": persistence},
            "lineage": {"ridge_alpha_100": {"fallback": False}}}


def _manifest(forecasts):
    doc = {"protocolId": median.PROTOCOL_ID, "phase": "forecast_only", "targetValuesAttached": False,
           "codeSha256": median._sha(Path(median.__file__).read_bytes()), "plannedForecastCount": len(forecasts),
           "forecasts": forecasts}
    doc["manifestSha256"] = median._hash(doc)
    return doc


def test_median_is_robust_to_one_extreme_component_and_has_no_answers():
    row = _forecast(ridge=100.0, sarima=101.0, persistence=9000.0)
    candidate = median._compose_forecast(row)
    assert candidate["predictions"]["median_ensemble"] == 101.0
    assert candidate["lineage"]["fallback"] is False
    assert "actual" not in candidate and "targetValue" not in candidate
    doc = _manifest([candidate])
    median.verify_manifest(doc)


def test_invalid_component_falls_back_to_ridge_and_records_exact_reason():
    for bad in (None, 0, float("inf"), float("nan"), "missing"):
        candidate = median._compose_forecast(_forecast(sarima=bad))
        assert candidate["predictions"]["median_ensemble"] == 100.0
        assert candidate["lineage"]["fallback"] is True
        assert candidate["lineage"]["fallbackReason"] == "invalid_component:calendar_sarima"
    candidate = median._compose_forecast(_forecast(ridge=None))
    assert candidate["predictions"]["median_ensemble"] is None
    assert "ridge_alpha_100" in candidate["lineage"]["fallbackReason"]


def test_pinned_manifest_byte_hash_rejects_tampering_hermetically(tmp_path, monkeypatch):
    path = tmp_path / "manifest.json"
    path.write_text('{"manifestSha256":"bad"}', encoding="utf-8")
    monkeypatch.setitem(median.MANIFEST_PATHS, "development", str(path))
    with pytest.raises(ValueError, match="byte SHA-256 mismatch"):
        median._load_frozen_manifest("development")
    assert path.read_text(encoding="utf-8") == '{"manifestSha256":"bad"}'


def test_candidate_manifest_replay_detects_forecast_tamper(monkeypatch):
    doc = _manifest([median._compose_forecast(_forecast())])
    rebuilt = _manifest([median._compose_forecast(_forecast(persistence=999.0))])
    monkeypatch.setattr(median, "build_manifest", lambda *_: rebuilt)
    with pytest.raises(ValueError, match="replay mismatch"):
        median._verify_replay(doc, {}, {}, {})


def test_calendar_dm_family_holm_and_unavailable_status():
    diffs = {dt.date(2026, 1, 1) + dt.timedelta(days=i): (1.0 if i % 2 else -1.0) for i in range(35)}
    available = median._dm(diffs, 9)
    unavailable = median._dm(dict(list(diffs.items())[:12]), 9)
    assert available["status"] == "available"
    assert unavailable["status"] == "inference_unavailable_insufficient_pairs"
    constant = {dt.date(2026, 1, 1) + dt.timedelta(days=i): 0.1 for i in range(35)}
    zero_variance = median._dm(constant, 9)
    assert zero_variance["status"] == "inference_unavailable_nonpositive_variance"
    assert zero_variance["pValue"] == 1.0
    records = [{"dm": {"9": median._dm(diffs if i == 0 else dict(list(diffs.items())[:12]), 9)}} for i in range(240)]
    median._holm_family(records, 9)
    assert len(records) == 240
    assert all("holmAdjustedPValue" in r["dm"]["9"] for r in records)
    assert records[-1]["dm"]["9"]["pValue"] == 1.0
    assert records[-1]["dm"]["9"]["conclusion"] == "inference_unavailable"


def test_create_only_output_does_not_replace_existing_artifact(tmp_path):
    path = tmp_path / "candidate.json"
    median._write_json_create_only(path, {"phase": "forecast_only"})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        median._write_json_create_only(path, {"phase": "scored"})
    assert path.read_bytes() == original
