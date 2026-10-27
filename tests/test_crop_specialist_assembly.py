import hashlib
import json
import datetime as dt
import sys
import types
from pathlib import Path

import pytest

from agri_data import crop_specialist_assembly as assembly


def _canon(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                        separators=(",", ":"), allow_nan=False) + "\n").encode()


def _manifest(path: Path, rows, version: str, rhs_hash: str):
    doc = {"phase": "V", "targetValuesAttached": False,
           "priceSnapshotSha256": version, "targetSnapshotSha256": version,
           "baselineManifestSha256": rhs_hash, "forecasts": rows}
    doc["manifestSha256"] = hashlib.sha256(_canon(doc)).hexdigest()
    path.write_bytes(_canon(doc))


def _sources(tmp_path, *, values=None, missing=None):
    prices = tmp_path / "prices.json"
    rows = []
    for product in assembly.PRODUCTS:
        rows.append({"id": product, "observations": [{"date": "2014-01-03", "actual": 5.0},
                                                       {"date": "2026-04-11", "actual": 100.0}]})
    prices.write_bytes(_canon({"products": rows}))
    version = hashlib.sha256(prices.read_bytes()).hexdigest()
    values = values or {"A": 110.0, "L": 105.0, "P": 105.0, "B": 120.0}
    missing = missing or set()
    raw_rhs_rows = []
    for product in assembly.PRODUCTS:
        for h in assembly.HORIZONS:
            target = "2026-04-11"
            origin = (assembly._date(target) - __import__("datetime").timedelta(days=h)).isoformat()
            raw_rhs_rows.append({"product": product, "origin": origin, "target": target,
                                 "horizon": h,
                                 "conditions": {"C": {"status": "issued", "blendPrediction": values["B"]},
                                                "B": {"status": "issued", "blendPrediction": values["B"]}}})
    rhs_path = tmp_path / "rhs.json"
    rhs_path.write_bytes(_canon({"phase": "V", "forecasts": raw_rhs_rows}))
    rhs_hash = hashlib.sha256(rhs_path.read_bytes()).hexdigest()
    docs = {}
    for model in ("A", "B", "L", "P"):
        forecast_rows = []
        for product in assembly.PRODUCTS:
            for h in assembly.HORIZONS:
                # Keep the same target and exact day horizon while supplying ten distinct origins.
                target = "2026-04-11"
                origin = (assembly._date(target) - __import__("datetime").timedelta(days=h)).isoformat()
                value = None if (product, model, h) in missing else values[model]
                payload = {"prediction": value, "status": "issued" if value else "not_issued"}
                row = {"product": product, "origin": origin, "target": target, "horizon": h}
                if model == "A":
                    row["CAND_A"] = payload
                elif model == "B":
                    row["B"] = {"E106_RHS": payload}
                else:
                    row["models"] = {model: payload}
                forecast_rows.append(row)
        path = tmp_path / f"{model}.json"
        _manifest(path, forecast_rows, version, rhs_hash)
        docs[model] = path
    return docs, prices, rhs_path, rhs_hash


def test_selection_uses_equal_horizon_metrics_and_fixed_tie_order(tmp_path):
    docs, prices, rhs, rhs_hash = _sources(tmp_path)
    result = assembly.build_selection(source_paths=docs, prices_path=prices,
                                      rhs_baseline_path=rhs, rhs_baseline_sha256=rhs_hash)
    assert result["actualCoverage"]["positiveActualTargetCount"] == 4
    for product in assembly.PRODUCTS:
        choice = result["products"][product]
        assert choice["selectedModelId"] == "L"
        assert choice["eligibleAlternatives"] == ["A", "L", "P"]
        assert choice["metrics"]["L"]["mape"] == 5.0
        assert choice["metrics"]["L"]["mae"] == 5.0
        assert choice["metrics"]["L"]["coverage"] == 1.0


def test_selection_rejects_candidate_with_coverage_loss(tmp_path):
    docs, prices, rhs, rhs_hash = _sources(tmp_path, values={"A": 115.0, "L": 110.0, "P": 100.0, "B": 120.0},
                                           missing={("cabbage", "P", 10)})
    result = assembly.build_selection(source_paths=docs, prices_path=prices,
                                      rhs_baseline_path=rhs, rhs_baseline_sha256=rhs_hash)
    cabbage = result["products"]["cabbage"]
    assert cabbage["eligibility"]["P"]["mapeStrictlyBelowB"] is None
    assert cabbage["eligibility"]["P"]["coverageNoLowerThanB"] is False
    assert cabbage["selectedModelId"] == "L"


def test_source_parser_rejects_duplicate_exact_keys():
    doc = {"phase": "V", "targetValuesAttached": False, "forecasts": [
        {"product": "cabbage", "origin": "2026-04-10", "target": "2026-04-11", "horizon": 1,
         "B": {"E106_RHS": {"prediction": 100, "status": "issued"}}},
        {"product": "cabbage", "origin": "2026-04-10", "target": "2026-04-11", "horizon": 1,
         "B": {"E106_RHS": {"prediction": 100, "status": "issued"}}},
    ]}
    with pytest.raises(ValueError, match="duplicate exact key"):
        assembly._source_rows(doc, "B", "version")


def test_p_technical_fallback_is_not_scored_as_direct_candidate():
    doc = {"phase": "V", "targetValuesAttached": False, "forecasts": [
        {"product": "cabbage", "origin": "2026-04-10", "target": "2026-04-11", "horizon": 1,
         "models": {"P": {"prediction": 100, "status": "issued", "modelUsed": False,
                           "rawModelStatus": "not_issued", "rawModelPrediction": None}}},
    ]}
    parsed = assembly._source_rows(doc, "P", "version")
    assert next(iter(parsed.values())) == {"prediction": None, "status": "not_issued"}


def test_positive_prediction_with_unavailable_status_is_not_issued():
    doc = {"phase": "V", "targetValuesAttached": False, "forecasts": [
        {"product": "cabbage", "origin": "2026-04-10", "target": "2026-04-11", "horizon": 1,
         "models": {"L": {"prediction": 100, "status": "not_issued"}}},
    ]}
    parsed = assembly._source_rows(doc, "L", "version")
    assert next(iter(parsed.values())) == {"prediction": None, "status": "not_issued"}


def test_temporal_fit_evidence_requires_past_labels_and_cutoff_features():
    rows = [{"target": dt.date(2026, 7, 9), "sourceDate": dt.date(2026, 7, 7),
             "cutoff": dt.date(2026, 7, 8)},
            {"target": dt.date(2026, 7, 10), "sourceDate": dt.date(2026, 7, 9),
             "cutoff": dt.date(2026, 7, 9)}]
    good = assembly._temporal_fit_evidence(rows, dt.date(2026, 7, 11))
    assert good == {"fitTargetMax": "2026-07-10",
                    "trainingTargetsBeforeFitOrigin": True,
                    "allFeatureDatesAtOrBeforeCutoff": True}
    rows[1]["sourceDate"] = dt.date(2026, 7, 10)
    bad = assembly._temporal_fit_evidence(rows, dt.date(2026, 7, 11))
    assert bad["allFeatureDatesAtOrBeforeCutoff"] is False
    assert assembly._temporal_fit_evidence(rows, dt.date(2026, 7, 10))[
        "trainingTargetsBeforeFitOrigin"] is False


def test_month_key_and_run_t_resume_replays_frozen_forecasts(tmp_path, monkeypatch):
    assert assembly._month_key("2026-07-11") == dt.date(2026, 7, 1)
    phase = tmp_path / "T"
    phase.mkdir()
    forecasts = phase / "forecasts.json"
    forecasts.write_bytes(_canon({"manifestSha256": "frozen", "forecastRowsSha256": "rows",
                                  "forecastKeyCount": 30, "plannedFitCount": 150,
                                  "actualEstimatorFitCount": 150,
                                  "newEstimatorFitCount": 150}))
    monkeypatch.setattr(assembly, "verify_saved_t_replay",
                        lambda **kwargs: {"replayVerificationSha256": "replayed"})
    result = assembly.run_t(output_dir=tmp_path)
    assert result["status"] == "saved_forecasts_replayed"
    assert result["manifestSha256"] == "frozen"
    assert result["replayVerificationSha256"] == "replayed"


def test_thread_cap_validation_uses_launch_environment_only(monkeypatch):
    for name in assembly.THREAD_CAP_ENV:
        monkeypatch.setenv(name, "1")
    # The installed threadpoolctl crashes while probing OpenBLAS in this runtime.
    # CAND_A/CAND_B model configs already pin their estimator threads to one.
    broken_threadpoolctl = types.ModuleType("threadpoolctl")
    monkeypatch.setitem(sys.modules, "threadpoolctl", broken_threadpoolctl)
    assembly._enforce_thread_limits()
    monkeypatch.setenv(assembly.THREAD_CAP_ENV[0], "2")
    with pytest.raises(RuntimeError, match="all six frozen thread caps"):
        assembly._enforce_thread_limits()
