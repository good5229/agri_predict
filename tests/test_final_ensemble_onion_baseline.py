from __future__ import annotations

import datetime as dt
import shutil

import pytest

from agri_data import final_ensemble_onion_baseline as baseline


def test_expected_origin_keys_include_all_horizons_and_boundary_targets():
    keys = baseline.expected_origin_keys(dt.date(2025, 12, 30), dt.date(2025, 12, 31))
    assert len(keys) == 20
    assert ("onion", "2025-12-31", "2026-01-10", 10) in keys
    assert ("onion", "2025-12-30", "2025-12-31", 1) in keys


def test_extract_reuses_exact_onion_causal50_rows(monkeypatch):
    monkeypatch.setattr(baseline, "_verify_equal_info_manifest", lambda doc: None)
    doc = {
        "manifestSha256": "a" * 64,
        "confirmationForecasts": [
            {"product": "onion", "origin": "2025-12-30", "target": "2025-12-31", "horizon": 1,
             "eligible": True,
             "models": {"causal50": {"status": "issued", "prediction": "123.5", "reason": None},
                        "causalDirect": {"status": "issued", "prediction": 120}}},
            {"product": "garlic", "origin": "2025-12-30", "target": "2025-12-31", "horizon": 1,
             "eligible": True,
             "models": {"causal50": {"status": "issued", "prediction": 99},
                        "causalDirect": {"status": "issued", "prediction": 98}}},
        ],
        "evaluationForecasts": [
            {"product": "onion", "origin": "2025-12-31", "target": "2026-01-10", "horizon": 10,
             "eligible": True,
             "models": {"causal50": {"status": "not_issued", "prediction": None,
                                      "reason": "sarima_failed"},
                        "causalDirect": {"status": "issued", "prediction": 120}}},
        ],
    }
    rows = baseline._extract_reusable_rows(doc)
    assert set(rows) == {
        ("onion", "2025-12-30", "2025-12-31", 1),
        ("onion", "2025-12-31", "2026-01-10", 10),
    }
    issued = rows[("onion", "2025-12-30", "2025-12-31", 1)]
    assert issued["B"] == 123.5
    assert issued["sourceStatus"] == "issued"
    assert issued["sourcePhase"] == "confirmation"
    assert issued["reuse"] is True
    failed = rows[("onion", "2025-12-31", "2026-01-10", 10)]
    assert failed["B"] is None
    assert failed["sourceStatus"] == "not_issued"
    assert failed["sourceReason"] == "sarima_failed"
    assert issued["sourceSarima"] is None


def test_phase_ineligible_source_placeholder_is_not_reusable(monkeypatch):
    monkeypatch.setattr(baseline, "_verify_equal_info_manifest", lambda doc: None)
    row = {"product": "onion", "origin": "2025-12-31", "target": "2026-01-10", "horizon": 10,
           "eligible": False,
           "models": {"causal50": {"status": "not_issued", "prediction": None,
                                    "reason": "model_not_yet_fitted_at_origin"},
                      "causalDirect": {"status": "not_issued", "prediction": None}}}
    assert baseline._extract_reusable_rows({"manifestSha256": "a" * 64,
                                             "confirmationForecasts": [],
                                             "evaluationForecasts": [row]}) == {}


def test_extract_rejects_nonpositive_issued_values(monkeypatch):
    monkeypatch.setattr(baseline, "_verify_equal_info_manifest", lambda doc: None)
    row = {"product": "onion", "origin": "2025-12-30", "target": "2025-12-31", "horizon": 1,
           "eligible": True,
           "models": {"causal50": {"status": "issued", "prediction": 0},
                      "causalDirect": {"status": "issued", "prediction": 1}}}
    with pytest.raises(ValueError, match="invalid issued"):
        baseline._extract_reusable_rows({"manifestSha256": "a" * 64,
                                         "confirmationForecasts": [row], "evaluationForecasts": []})


def test_extract_rejects_duplicate_keys(monkeypatch):
    monkeypatch.setattr(baseline, "_verify_equal_info_manifest", lambda doc: None)
    row = {"product": "onion", "origin": "2025-12-30", "target": "2025-12-31", "horizon": 1,
           "eligible": True,
           "models": {"causal50": {"status": "not_issued", "prediction": None},
                      "causalDirect": {"status": "not_issued", "prediction": None}}}
    with pytest.raises(ValueError, match="duplicate"):
        baseline._extract_reusable_rows({"manifestSha256": "a" * 64,
                                         "confirmationForecasts": [row], "evaluationForecasts": [row]})


def test_rhs_comparison_binds_exact_onion_key_price_and_status():
    key = ("onion", "2026-03-31", "2026-04-01", 1)
    saved = {key: {"B": 101.25, "BSourceStatus": "fallback"}}
    rhs = [{"product": "onion", "origin": dt.date(2026, 3, 31),
            "target": dt.date(2026, 4, 1), "horizon": 1,
            "B": 101.25, "BSourceStatus": "fallback"}]
    result = baseline.compare_with_rhs(saved, rhs, "V", expected_count=1)
    assert result["commonOnionKeys"] == 1
    assert result["equalBStatusAndPrice"] is True
    assert result["mismatchCount"] == 0


def test_rhs_comparison_rejects_price_or_status_drift():
    key = ("onion", "2026-03-31", "2026-04-01", 1)
    saved = {key: {"B": 101.25, "BSourceStatus": "issued"}}
    rhs = [{"product": "onion", "origin": dt.date(2026, 3, 31),
            "target": dt.date(2026, 4, 1), "horizon": 1,
            "B": 101.2501, "BSourceStatus": "fallback"}]
    with pytest.raises(ValueError, match="differs from pinned RHS"):
        baseline.compare_with_rhs(saved, rhs, "V", expected_count=1)


def test_fit_boundary_is_first_calendar_month_origin_not_first_missing_key():
    assert baseline._month_first_eligible_origin("2025-12", dt.date(2021, 1, 1)) == dt.date(2025, 12, 1)
    assert baseline._month_first_eligible_origin("2021-01", dt.date(2021, 1, 15)) == dt.date(2021, 1, 15)


def test_native_reload_probe_rejects_prediction_drift():
    baseline._assert_native_replay([1.25, 2.5], [1.25, 2.5], "same")
    with pytest.raises(ValueError, match="replay mismatch"):
        baseline._assert_native_replay([1.25], [1.2501], "drift")


def test_sarima_append_failure_is_retried_at_next_origin(monkeypatch):
    from agri_data import ten_strategy_benchmark

    monkeypatch.setattr(ten_strategy_benchmark, "_calendar_values",
                        lambda series, start, end: [series.get(start)])

    class FailsOnce:
        def __init__(self):
            self.calls = 0

        def append(self, _values, refit=False):
            assert refit is False
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("transient append failure")
            return self

    state = FailsOnce()
    day = dt.date(2026, 1, 1)
    state, state_day, error = baseline._advance_sarima_state(
        state, dt.date(2025, 12, 31), dt.date(2026, 1, 2), {day: 1.0})
    assert error == "sarima_state_update_failed:RuntimeError"
    assert state_day == dt.date(2025, 12, 31)
    state, state_day, error = baseline._advance_sarima_state(
        state, state_day, dt.date(2026, 1, 3), {day: 1.0})
    assert error is None
    assert state_day == dt.date(2026, 1, 2)
    assert state.calls == 3


def test_month_checkpoint_recovers_previous_pair_after_interrupted_replace(tmp_path):
    from agri_data import direct_price_path_v2
    from agri_data import equal_info_online_forecasts as equal_info

    code_hashes = {
        "baselineCacheModule": baseline.file_sha256(baseline.Path(baseline.__file__)),
        "equalInfoModule": baseline.file_sha256(baseline.Path(equal_info.__file__)),
        "directPricePathV2": baseline.file_sha256(baseline.Path(direct_price_path_v2.__file__)),
    }
    key_row = {"product": "onion", "origin": "2025-12-30", "target": "2025-12-31",
               "horizon": 1, "B": 100.0, "sourceStatus": "issued"}
    cache_doc = {"schemaVersion": 1, "phase": "answer_free",
                 "targetValuesAttached": False, "rows": [key_row]}
    manifest = {"phase": "answer_free", "targetValuesAttached": False,
                "equalInfoManifestFileSha256": baseline.EQUAL_INFO_MANIFEST_SHA256,
                "sourceCodeSha256": code_hashes, "fitInventory": []}
    baseline._commit_cache(tmp_path, cache_doc, manifest)
    shutil.copyfile(tmp_path / baseline.CACHE_NAME, tmp_path / (baseline.CACHE_NAME + ".previous"))
    shutil.copyfile(tmp_path / baseline.MANIFEST_NAME, tmp_path / (baseline.MANIFEST_NAME + ".previous"))
    (tmp_path / baseline.CACHE_NAME).write_text("interrupted replacement")
    (tmp_path / ".baseline-cache-commit-pending").write_text("interrupted")

    rows, fits = baseline._load_previous_cache(tmp_path)
    assert rows[("onion", "2025-12-30", "2025-12-31", 1)]["B"] == 100.0
    assert fits == []


@pytest.mark.parametrize("failure", ["producer_code", "model_artifact"])
def test_pending_marker_does_not_bypass_cache_integrity_errors(tmp_path, failure):
    from agri_data import direct_price_path_v2
    from agri_data import equal_info_online_forecasts as equal_info

    code_hashes = {
        "baselineCacheModule": baseline.file_sha256(baseline.Path(baseline.__file__)),
        "equalInfoModule": baseline.file_sha256(baseline.Path(equal_info.__file__)),
        "directPricePathV2": baseline.file_sha256(baseline.Path(direct_price_path_v2.__file__)),
    }
    fits = []
    if failure == "producer_code":
        code_hashes["baselineCacheModule"] = "0" * 64
    else:
        (tmp_path / "model.joblib").write_bytes(b"model bytes")
        fits = [{"fitId": "broken", "modelPath": "model.joblib", "modelSha256": "0" * 64}]
    cache_doc = {"schemaVersion": 1, "phase": "answer_free",
                 "targetValuesAttached": False, "rows": []}
    manifest = {"phase": "answer_free", "targetValuesAttached": False,
                "equalInfoManifestFileSha256": baseline.EQUAL_INFO_MANIFEST_SHA256,
                "sourceCodeSha256": code_hashes, "fitInventory": fits}
    baseline._commit_cache(tmp_path, cache_doc, manifest)
    (tmp_path / ".baseline-cache-commit-pending").write_text("stale marker")

    with pytest.raises(ValueError, match="pending cache commit failed integrity validation"):
        baseline._load_previous_cache(tmp_path)
