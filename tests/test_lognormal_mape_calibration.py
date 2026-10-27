import datetime as dt
import math
import copy

import pytest

from agri_data import dashboard
from agri_data import lognormal_mape_calibration as calibration


def _row(product, horizon, target, actual=100.0, direct=100.0, persistence=100.0):
    target = dt.date.fromisoformat(target)
    origin = target - dt.timedelta(days=horizon)
    return {"product": product, "horizon": horizon, "target": target.isoformat(),
            "origin": origin.isoformat(), "actual": actual,
            "direct": direct, "persistence": persistence}


def _report(rows):
    return {"schemaVersion": 1, "protocol": {"id": "direct-horizon-ridge-2026-10-06"},
            "pairedRows": rows}


def _history_rows(n=30):
    rows = []
    first = dt.date(2025, 8, 1)
    for i in range(n):
        target = first + dt.timedelta(days=i)
        residual = 0.02 if i % 2 else -0.02
        rows.append(_row("cabbage", 1, target.isoformat(), actual=100 * math.exp(residual)))
    return rows


def test_temporal_filter_excludes_recent_and_current_future_targets_and_uses_population_variance():
    rows = _history_rows(30)
    # Evaluation origin is 2025-09-24: window is [2024-09-24, 2025-09-10].
    rows.extend([_row("cabbage", 1, "2024-09-23", actual=100000),
                 _row("cabbage", 1, "2024-09-24", actual=101),
                 _row("cabbage", 1, "2025-09-10", actual=102),
                 _row("cabbage", 1, "2025-09-11", actual=10000),
                 _row("cabbage", 1, "2025-09-26", actual=0)])
    rows.append(_row("cabbage", 1, "2025-09-25", actual=0, direct=100))
    manifest = calibration.build_forecast_manifest(_report(rows),
                                                    direct_report_sha256=calibration.DIRECT_SHA256,
                                                    code_sha256="a" * 64, expected_pairs=None)
    forecast = manifest["forecasts"][0]
    audit = forecast["calibration"]
    assert forecast["target"] == "2025-09-25"
    assert audit["residualCount"] == 32
    assert audit["windowStart"] == "2024-09-24"
    assert audit["windowEnd"] == "2025-09-10"
    residuals = [math.log(row["actual"] / row["direct"]) for row in rows
                 if "2024-09-24" <= row["target"] <= "2025-09-10"]
    mean = sum(residuals) / len(residuals)
    variance = sum((x - mean) ** 2 for x in residuals) / len(residuals)
    expected = max(-0.1, min(0.1, mean - variance))
    assert math.isclose(audit["populationVariance"], variance, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(audit["logCorrection"], expected, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(forecast["candidate"], 100 * math.exp(expected), rel_tol=1e-12)
    assert "actual" not in forecast and manifest["targetValuesAttached"] is False


def test_fewer_than_thirty_residuals_falls_back_to_direct_and_clips():
    rows = _history_rows(29) + [_row("cabbage", 1, "2025-09-25", actual=110)]
    manifest = calibration.build_forecast_manifest(_report(rows),
                                                    direct_report_sha256=calibration.DIRECT_SHA256,
                                                    code_sha256="b" * 64, expected_pairs=None)
    forecast = manifest["forecasts"][0]
    assert forecast["fallback"] is True
    assert forecast["candidate"] == forecast["direct"]
    assert forecast["calibration"]["residualCount"] == 29

    high = _history_rows(30)
    for row in high[:]:
        row["actual"] = row["direct"] * math.exp(0.5)
    high.append(_row("cabbage", 1, "2025-09-25", actual=110))
    forecast = calibration.build_forecast_manifest(_report(high), direct_report_sha256=calibration.DIRECT_SHA256,
                                                   code_sha256="c" * 64, expected_pairs=None)["forecasts"][0]
    assert forecast["calibration"]["logCorrection"] == pytest.approx(0.1)


def test_duplicate_keys_and_pinned_sha_mismatch_reject():
    rows = _history_rows(2)
    with pytest.raises(calibration.CalibrationError, match="duplicate direct key"):
        calibration.build_forecast_manifest(_report(rows + [rows[0]]),
                                             direct_report_sha256=calibration.DIRECT_SHA256,
                                             code_sha256="d" * 64, expected_pairs=None)
    with pytest.raises(calibration.CalibrationError, match="SHA-256"):
        calibration.build_forecast_manifest(_report(rows), direct_report_sha256="0" * 64,
                                             code_sha256="d" * 64, expected_pairs=None)


def test_score_crosschecks_keys_and_actuals_and_retains_dm_unavailable_family():
    direct_rows, candidates, blend_forecasts, cell_rows = [], [], [], []
    for p_index, product in enumerate(calibration.PRODUCTS):
        for horizon in calibration.HORIZONS:
            actual, direct, persistence, blend = 100.0 + p_index, 102.0 + p_index, 104.0 + p_index, 101.0 + p_index
            row = _row(product, horizon, "2025-09-25", actual, direct, persistence)
            direct_rows.append(row)
            forecast = {"product": product, "horizon": horizon, "origin": row["origin"],
                        "target": row["target"], "direct": direct, "persistence": persistence,
                        "candidate": direct, "fallback": True,
                        "fallbackReason": "fewer_than_30_eligible_residuals",
                        "calibration": {"residualCount": 0, "logCorrection": 0}}
            candidates.append(forecast)
            blend_forecasts.append({"product": product, "horizon": horizon, "origin": row["origin"],
                                    "target": row["target"],
                                    "predictions": {"direct_ridge": direct, "blend": blend,
                                                    "persistence": persistence}})
            cell_rows.append({"product": product, "horizon": horizon, "plannedPairs": 1,
                              "metrics": {"blend": {"mae": abs(actual-blend), "mape": 100*abs(actual-blend)/actual},
                                          "direct_ridge": {"mae": abs(actual-direct), "mape": 100*abs(actual-direct)/actual},
                                          "persistence": {"mae": abs(actual-persistence), "mape": 100*abs(actual-persistence)/actual}}})
    forecast_manifest = calibration.build_forecast_manifest(
        _report(direct_rows), direct_report_sha256=calibration.DIRECT_SHA256,
        code_sha256="e" * 64, expected_pairs=None)
    blend_manifest = {"phase": "forecast_only", "targetValuesAttached": False,
                      "manifestSha256": "internal-blend-hash", "forecasts": blend_forecasts}
    blend_report = {"manifestSha256": "internal-blend-hash",
                    "inputSha256": {"directReport": calibration.DIRECT_SHA256}, "cellMetrics": cell_rows}
    result = calibration.score_manifest(forecast_manifest, _report(direct_rows), blend_manifest, blend_report,
                                        direct_report_sha256=calibration.DIRECT_SHA256,
                                        code_sha256="e" * 64, expected_pairs=None)
    assert result["pairedKeyCount"] == 50
    assert len(result["dm"]["comparisons"]) == 300
    assert all(row["dm"]["9"]["status"] == "insufficient_pairs" for row in result["dm"]["comparisons"])
    assert len(result["byProductHorizon"]) == 50

    def rehash(value):
        value.pop("manifestSha256", None)
        value["manifestSha256"] = calibration.sha256(calibration.canonical(value))
        return value

    forged = copy.deepcopy(forecast_manifest)
    row = forged["forecasts"][0]
    row["candidate"] = row["direct"] * math.exp(0.01)
    row["fallback"] = False
    row["fallbackReason"] = None
    row["calibration"].update({"residualCount": 30, "windowStart": "2025-09-01",
                               "windowEnd": "2025-09-10", "logCorrection": 0.01})
    with pytest.raises(calibration.CalibrationError, match="recomputed|fallback"):
        calibration.score_manifest(rehash(forged), _report(direct_rows), blend_manifest, blend_report,
                                   direct_report_sha256=calibration.DIRECT_SHA256,
                                   code_sha256="e" * 64, expected_pairs=None)

    forged = copy.deepcopy(forecast_manifest)
    forged["forecasts"][0]["calibration"]["residualCount"] = 1
    with pytest.raises(calibration.CalibrationError, match="audit value mismatch"):
        calibration.score_manifest(rehash(forged), _report(direct_rows), blend_manifest, blend_report,
                                   direct_report_sha256=calibration.DIRECT_SHA256,
                                   code_sha256="e" * 64, expected_pairs=None)

    forged = copy.deepcopy(forecast_manifest)
    forged["forecasts"][0]["calibration"]["windowStart"] = "1900-01-01"
    with pytest.raises(calibration.CalibrationError, match="audit value mismatch"):
        calibration.score_manifest(rehash(forged), _report(direct_rows), blend_manifest, blend_report,
                                   direct_report_sha256=calibration.DIRECT_SHA256,
                                   code_sha256="e" * 64, expected_pairs=None)

    forged = copy.deepcopy(forecast_manifest)
    forged["protocol"] = {**calibration.PROTOCOL, "correction": "tampered"}
    with pytest.raises(calibration.CalibrationError, match="protocol mismatch"):
        calibration.score_manifest(rehash(forged), _report(direct_rows), blend_manifest, blend_report,
                                   direct_report_sha256=calibration.DIRECT_SHA256,
                                   code_sha256="e" * 64, expected_pairs=None)

    bad_report = _report([dict(direct_rows[0], actual=999)] + direct_rows[1:])
    # A changed truth must invalidate the aggregate blend report cross-check.
    with pytest.raises(calibration.CalibrationError, match="metric mismatch"):
        calibration.score_manifest(forecast_manifest, bad_report, blend_manifest, blend_report,
                                   direct_report_sha256=calibration.DIRECT_SHA256,
                                   code_sha256="e" * 64, expected_pairs=None)
