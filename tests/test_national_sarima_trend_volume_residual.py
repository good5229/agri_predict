import copy
import datetime as dt
import json
import math

import numpy as np
import pytest

from agri_data import national_sarima_trend_volume_residual as residual
from agri_data import national_trend_volume_interaction as shared


def _series(start=dt.date(2025, 7, 1), end=dt.date(2026, 4, 10)):
    rows = {}
    for i in range((end - start).days + 1):
        day = start + dt.timedelta(days=i)
        rows[day] = {"price": 200 + 0.12 * i + 3 * math.sin(i / 8),
                     "quantity": 100 + 0.18 * i + 6 * math.cos(i / 11),
                     "negativeQuantityCount": 0, "sourceRowCount": 1}
    return shared._Series(rows)


def test_features_use_only_o_minus_two_source_rows():
    target, horizon = dt.date(2026, 3, 20), 5
    series = _series(end=dt.date(2026, 3, 30))
    first = shared._features(series, target, horizon)
    changed = copy.deepcopy(series.rows)
    cutoff = dt.date.fromisoformat(first["cutoff"])
    changed[target]["price"] = 999999.0
    changed[cutoff + dt.timedelta(days=1)]["quantity"] = -99999.0
    later = shared._features(shared._Series(changed), target, horizon)
    assert first["features"] == later["features"]
    assert first["volumeX"] == later["volumeX"]
    assert first["p0"]["date"] <= first["cutoff"]
    assert all(dt.date.fromisoformat(day) <= cutoff for day in first["recentVolume"]["dates"] + first["priorVolume"]["dates"])


def test_training_uses_only_past_nonfallback_sarima_rows_and_same_pairs():
    series = _series(end=dt.date(2026, 3, 30))
    baseline_rows = {}
    first_target = dt.date(2025, 10, 1)
    for target in series.price_days:
        if target < first_target:
            continue
        for horizon in residual.HORIZONS:
            origin = target - dt.timedelta(days=horizon)
            key = ("radish", origin.isoformat(), target.isoformat(), horizon)
            fallback = target.day % 4 == 0
            baseline_rows[key] = {"product": "radish", "origin": origin.isoformat(), "target": target.isoformat(),
                "horizon": horizon, "predictions": {"calendar_sarima": series.rows[target]["price"] * 0.97},
                "lineage": {"calendar_sarima": {"fallback": fallback,
                    "nativeStatus": "unavailable" if fallback else "available"}}}
    fit = residual._fit_residual_models("radish", series, baseline_rows, dt.date(2026, 4, 1))
    info = fit["fit"]
    assert fit["fitFailure"] is None
    assert info["uniqueTrainingTargetDays"] >= 60
    assert info["trainingRowCount"] == len(info["trainingRows"])
    assert all(dt.date.fromisoformat(row["target"]) <= dt.date.fromisoformat(info["fitCutoff"]) for row in info["trainingRows"])
    assert all(dt.date.fromisoformat(row["featureCutoff"]) == dt.date.fromisoformat(row["target"]) - dt.timedelta(days=row["horizon"] + 2) for row in info["trainingRows"])
    assert all(dt.date.fromisoformat(row["target"]).day % 4 != 0 for row in info["trainingRows"])
    assert fit["models"][residual.METHOD_PRICE]["trainingRowsSha256"] == fit["models"][residual.METHOD_VOLUME]["trainingRowsSha256"]


def _model(names, intercept=0.0):
    return {"featureNames": list(names), "scalerMean": [0.0] * len(names),
            "scalerScale": [1.0] * len(names), "coefficients": [0.0] * len(names),
            "intercept": intercept}


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_correction_uses_same_sarima_fallback_for_both(bad):
    feature = {"state": "available", "p0": {"price": 100.0},
        "features": [0.0] * len(residual.PRICE_FEATURE_NAMES),
        "volumeFeatures": [0.0] * len(residual.VOLUME_FEATURE_NAMES)}
    fit = {"fitFailure": None, "models": {residual.METHOD_PRICE: _model(residual.PRICE_FEATURE_NAMES),
        residual.METHOD_VOLUME: _model(residual.VOLUME_FEATURE_NAMES, bad)}}
    predictions, fallback = residual._correct_pair(fit, feature, 200.0, 190.0)
    assert predictions == {residual.METHOD_PRICE: 200.0, residual.METHOD_VOLUME: 200.0}
    assert fallback["fallbackBase"] == "calendar_sarima"
    assert "nonfinite_raw_log_residual_prediction" in fallback["fallbackReason"]


def test_unavailable_sarima_uses_fixed_blend_as_common_fallback():
    preds, state = residual._correct_pair({"fitFailure": "bad fit"}, {"state": "unavailable"}, None, 190.0)
    assert preds == {residual.METHOD_PRICE: 190.0, residual.METHOD_VOLUME: 190.0}
    assert state["fallbackBase"] == "fixed_blend_50_50"


def test_synthetic_builder_and_score_path_cover_frozen_windows():
    start, end = dt.date(2025, 6, 1), dt.date(2026, 6, 30)
    observations = {p: {} for p in residual.PRODUCTS}
    for offset in range((end-start).days+1):
        day = start+dt.timedelta(days=offset)
        for index, product in enumerate(residual.PRODUCTS):
            price = 150 + index*15 + offset*0.12 + 2*math.sin(offset/8)
            quantity = 100 + index + offset*0.15 + 5*math.cos(offset/11)
            observations[product][day.isoformat()] = {"price": str(price), "priceReason": None,
                "productNames": [shared.PRODUCT_NAMES[product]], "nameConflictCount": 0,
                "quantitySum": str(quantity), "negativeQuantityCount": 0, "rowCount": 1}
    baseline_rows = []
    for offset in range((dt.date(2026, 6, 30)-dt.date(2025, 10, 1)).days+1):
        target = dt.date(2025, 10, 1)+dt.timedelta(days=offset)
        for product in residual.PRODUCTS:
            for horizon in residual.HORIZONS:
                origin = target-dt.timedelta(days=horizon)
                baseline_rows.append({"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                    "horizon": horizon, "cutoff": (origin-dt.timedelta(days=2)).isoformat(),
                    "predictions": {"calendar_sarima": 150.0, "fixed_blend_50_50": 151.0},
                    "lineage": {"calendar_sarima": {"fallback": False, "nativeStatus": "available"},
                                "fixed_blend_50_50": {"fallback": False}}})
    parsed = {"archiveSha256": residual.ARCHIVE_SHA256, "archiveByteCount": 100,
        "sourceRowCount": len(observations)*((end-start).days+1), "periods": [],
        "exactDuplicateRowCount": 0, "observations": observations}
    source_digests = residual._source_digests(parsed)
    baseline = {"sourceDigests": source_digests, "forecasts": baseline_rows}
    doc = residual._build_from_parsed(parsed, baseline, residual.BASELINE_MANIFEST_SHA256)
    residual.verify_manifest(doc)
    assert len(doc["forecasts"]) == 7240
    assert len(doc["monthlyFits"]) == 28
    assert doc["monthlyFits"]["radish/2025-12"]["fit"]["uniqueTrainingTargetDays"] >= 60
    report = residual._score_from_inputs(doc, parsed)
    assert [period["planned"] for period in report["periods"].values()] == [3600, 3640]
    assert all(period["holmFamilySizePerBandwidth"] == 240 for period in report["periods"].values())


def _answer_free_manifest():
    forecasts = []
    for i in range(residual.TARGET_DAYS):
        target = residual.TARGET_START + dt.timedelta(days=i)
        for product in residual.PRODUCTS:
            for horizon in residual.HORIZONS:
                origin = target - dt.timedelta(days=horizon)
                cutoff = origin - dt.timedelta(days=2)
                feature = {"cutoff": cutoff.isoformat(), "p0": None, "p7": None, "p28": None,
                    "recentVolume": {"dates": []}, "priorVolume": {"dates": []}}
                forecasts.append({"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                    "horizon": horizon, "cutoff": cutoff.isoformat(),
                    "predictions": {method: 100.0 for method in residual.METHODS},
                    "lineage": {"feature": feature, "fallback": {"fallback": True,
                        "fallbackBase": "calendar_sarima", "fallbackReason": "synthetic"}}})
    doc = {"schemaVersion": 1, "protocolId": residual.PROTOCOL_ID, "phase": "forecast_only",
        "targetValuesAttached": False, "protocol": residual.PROTOCOL,
        "source": {"archiveSha256": residual.ARCHIVE_SHA256},
        "baselineManifestSha256": residual.BASELINE_MANIFEST_SHA256,
        "baselineManifestContentSha256": residual.BASELINE_CONTENT_SHA256,
        "codeSha256": residual._sha(residual.Path(residual.__file__).read_bytes()),
        "plannedForecastCount": residual.PLANNED_COUNT, "monthlyFits": {}, "forecasts": forecasts}
    doc["manifestSha256"] = residual._hash(doc)
    return doc


def test_manifest_tamper_and_full_replay_mismatch_fail(monkeypatch):
    doc = _answer_free_manifest()
    residual.verify_manifest(doc)
    tampered = copy.deepcopy(doc)
    tampered["forecasts"][0]["predictions"][residual.METHOD_VOLUME] = 101.0
    with pytest.raises(ValueError, match="content hash"):
        residual.verify_manifest(tampered)
    rebuilt = copy.deepcopy(doc)
    rebuilt["forecasts"][0]["predictions"][residual.METHOD_VOLUME] = 102.0
    rebuilt["manifestSha256"] = residual._hash({k: v for k, v in rebuilt.items() if k != "manifestSha256"})
    monkeypatch.setattr(residual, "build_manifest", lambda archive, baseline: rebuilt)
    with pytest.raises(ValueError, match="full SARIMA residual replay mismatch"):
        residual._verify_replay(doc, b"synthetic archive", b"synthetic baseline")


def test_period_report_has_240_tests_per_band_and_joint_results():
    target = dt.date(2026, 2, 1)
    selected = []
    actuals = {p: {target: 100.0} for p in residual.PRODUCTS}
    for product in residual.PRODUCTS:
        for horizon in residual.HORIZONS:
            origin = target-dt.timedelta(days=horizon)
            selected.append({"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                "horizon": horizon, "predictions": {m: 101.0 for m in residual.METHODS},
                "lineage": {"feature": {"trend7": 0.02}, "fallback": {"fallback": False}}})
    report = residual._period_report(selected, actuals)
    assert len(report["dmComparisons"]) == 480
    assert report["holmFamilySizePerBandwidth"] == 240
    assert len(report["jointDmConclusions"]) == 120
    assert all(row["dm"]["status"] == "insufficient_pairs" for row in report["dmComparisons"])

    constant = {dt.date(2026, 1, 1)+dt.timedelta(days=i): 0.1 for i in range(40)}
    for bandwidth in residual.BANDWIDTHS:
        result = shared._dm(constant, bandwidth)
        assert result["status"] == "zero_or_invalid_hac_variance"
        assert result["pValue"] == 1.0


def test_source_sha_gate_and_create_only_output(tmp_path):
    with pytest.raises(ValueError, match="national archive byte SHA-256 mismatch"):
        residual._verified_inputs(b"not pinned", b"not pinned")
    with pytest.raises(ValueError, match="development baseline manifest byte SHA-256 mismatch"):
        residual._verify_baseline_doc({}, b"not pinned")
    path = tmp_path / "nested" / "report.json"
    residual.write_create_only(path, {"ok": True})
    assert json.loads(path.read_text()) == {"ok": True}
    with pytest.raises(FileExistsError):
        residual.write_create_only(path, {"ok": True})
