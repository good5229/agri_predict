import csv
import datetime as dt
import hashlib
import json

import pytest

from agri_data import direct_price_path_v2 as producer


def _canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                        allow_nan=False) + "\n").encode("utf-8")


def _pinned_fixture(tmp_path, monkeypatch):
    manifest_path = tmp_path / "validation-manifest.json"
    csv_path = tmp_path / "validation-predictions.csv"
    state = {"fitId": None, "status": "issued", "fallbackReason": None,
             "cutoff": "2026-03-30", "directLogPrediction": 0.1,
             "directPrediction": 105.0, "sarimaPrediction": 103.0,
             "blendPrediction": 104.0}
    condition_states = {name: dict(state) for name in ("B", "C", "D", "P")}
    condition_states["B"].update(status="fallback", fallbackReason="independent E106 diagnostic",
                                  blendPrediction=97.0)
    nested_row = {"product": "cabbage", "origin": "2026-04-01", "target": "2026-04-02",
                  "originDate": "2026-04-01", "targetDate": "2026-04-02", "horizon": 1,
                  "eligible": True, "p0": 100.0, "p0Date": "2026-03-31",
                  "conditions": condition_states}
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=producer._BASELINE_CSV_FIELDS)
        writer.writeheader()
        for condition, row_state in condition_states.items():
            writer.writerow({"product": "cabbage", "origin": "2026-04-01", "target": "2026-04-02",
                             "originDate": "2026-04-01", "targetDate": "2026-04-02", "horizon": 1,
                             "eligible": True, "p0": 100.0, "p0Date": "2026-03-31", "fitId": "",
                             "condition": condition, "status": row_state["status"],
                             "directLogPrediction": row_state["directLogPrediction"],
                             "directPrediction": row_state["directPrediction"],
                             "sarimaPrediction": row_state["sarimaPrediction"],
                             "blendPrediction": row_state["blendPrediction"],
                             "fallbackReason": row_state["fallbackReason"] or "",
                             "cutoff": row_state["cutoff"]})
    csv_sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest = {"phase": "V", "protocolSha256": producer.BASELINE_PROTOCOL_SHA256,
                "forecastFile": csv_path.name, "forecastFileSha256": csv_sha,
                "forecasts": [nested_row]}
    manifest_path.write_bytes(_canonical(manifest))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    monkeypatch.setitem(producer.baseline_source.BASELINE_PATHS, "V", manifest_path)
    monkeypatch.setitem(producer.baseline_source.BASELINE_SHA256, "V", manifest_sha)
    monkeypatch.setitem(producer.BASELINE_EXPECTED_ELIGIBLE, "V", 1)
    return manifest_path, manifest, csv_path


def test_baseline_loader_crosschecks_nested_manifest_and_hashed_csv(tmp_path, monkeypatch):
    _pinned_fixture(tmp_path, monkeypatch)

    manifest, rows = producer._baseline_phase("V")

    assert manifest["forecastCsvPath"].name == "validation-predictions.csv"
    assert len(rows) == 1
    assert rows[0] == {"product": "cabbage", "origin": dt.date(2026, 4, 1),
                       "target": dt.date(2026, 4, 2), "horizon": 1,
                       "B": 104.0, "BSourceStatus": "issued", "baselineReason": None}


def test_baseline_loader_rejects_nested_csv_value_mismatch(tmp_path, monkeypatch):
    _, manifest, csv_path = _pinned_fixture(tmp_path, monkeypatch)
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["blendPrediction"] = "999"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=producer._BASELINE_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest["forecastFileSha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    manifest_path = producer.baseline_source.BASELINE_PATHS["V"]
    manifest_path.write_bytes(_canonical(manifest))
    monkeypatch.setitem(producer.baseline_source.BASELINE_SHA256, "V",
                        hashlib.sha256(manifest_path.read_bytes()).hexdigest())

    with pytest.raises(ValueError, match="blendPrediction mismatch"):
        producer._baseline_phase("V")


def test_baseline_loader_maps_saved_c_to_b_and_preserves_distinct_saved_b(tmp_path, monkeypatch):
    _pinned_fixture(tmp_path, monkeypatch)

    _, rows = producer._baseline_phase("V")

    assert rows[0]["B"] == 104.0
    assert rows[0]["BSourceStatus"] == "issued"
    assert rows[0]["baselineReason"] is None


def test_transition_config_fixture_allows_export_runtime_only(monkeypatch):
    data_path = "agri_data/direct_price_path_data.py"
    original = {"resolvedConfigSha256": "frozen-config",
                "sourceCodeFiles": {"agri_data/direct_price_path.py": producer._ORIGINAL_MAIN_SHA256,
                                    data_path: "d" * 64},
                "model": {"width": 48},
                "runtime": {"python": "3.10.2", "torch": "2.5.1", "device": "mps"}}
    current = {**original,
               "sourceCodeFiles": {"agri_data/direct_price_path_v2.py": "v" * 64,
                                   data_path: "d" * 64},
               "runtime": {"python": "3.13.1", "torch": None}}
    current.pop("resolvedConfigSha256")
    transition = {"originalConfigInternalSha256": original["resolvedConfigSha256"],
                  "revisedMainSha256": current["sourceCodeFiles"]["agri_data/direct_price_path_v2.py"]}
    monkeypatch.setattr(producer, "_load_implementation_transition", lambda **_: transition)

    producer._validate_implementation_transition(original, current, verify_native_runtime=False)
    with pytest.raises(ValueError, match="frozen configuration"):
        producer._validate_implementation_transition(original, current, verify_native_runtime=True)
