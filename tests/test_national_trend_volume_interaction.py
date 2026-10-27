import copy
import datetime as dt
import json
import math

import numpy as np
import pytest

from agri_data import national_trend_volume_interaction as experiment


def _series(start=dt.date(2025, 7, 1), end=dt.date(2026, 4, 10)):
    rows = {}
    i = 0
    while start + dt.timedelta(days=i) <= end:
        day = start + dt.timedelta(days=i)
        price = 200 + 0.13 * i + 3 * math.sin(i / 9)
        quantity = 120 + 0.2 * i + 8 * math.sin(i / 13)
        rows[day] = {"price": price, "quantity": quantity, "negativeQuantityCount": 0, "sourceRowCount": 1}
        i += 1
    return experiment._Series(rows)


def test_features_stop_at_o_minus_two_and_ignore_target_or_later_values():
    target = dt.date(2026, 3, 20)
    horizon = 5
    origin = target - dt.timedelta(days=horizon)
    cutoff = origin - dt.timedelta(days=2)
    original = _series(end=target + dt.timedelta(days=5))
    first = experiment._features(original, target, horizon)
    changed_rows = copy.deepcopy(original.rows)
    changed_rows[target]["price"] = 999999.0
    changed_rows[cutoff + dt.timedelta(days=1)]["price"] = 888888.0
    changed_rows[target + dt.timedelta(days=1)]["quantity"] = -99999.0
    changed = experiment._features(experiment._Series(changed_rows), target, horizon)

    assert first["features"] == changed["features"]
    assert first["volumeX"] == changed["volumeX"]
    assert first["cutoff"] == cutoff.isoformat()
    assert all(dt.date.fromisoformat(d) <= cutoff for d in first["recentVolume"]["dates"] + first["priorVolume"]["dates"])
    assert first["p0"]["date"] <= cutoff.isoformat()
    assert first["p7"]["date"] <= first["p7ReferenceDate"]
    assert first["p28"]["date"] <= first["p28ReferenceDate"]


def test_both_models_use_identical_training_rows_and_fallback_together():
    series = _series()
    fit = experiment._fit_models("radish", series, dt.date(2026, 4, 1))
    assert fit["fitFailure"] is None
    assert fit["fit"]["uniqueTrainingTargetDays"] >= 60
    assert fit["models"][experiment.METHOD_PRICE]["trainingRowsSha256"] == fit["models"][experiment.METHOD_VOLUME]["trainingRowsSha256"]
    assert fit["fit"]["trainingRowCount"] == len(fit["fit"]["trainingRows"])

    target = dt.date(2026, 4, 9)
    feature = experiment._features(experiment._Series({}), target, 1)
    predictions, fallback = experiment._predict_pair(fit, feature, 321.0)
    assert predictions == {experiment.METHOD_PRICE: 321.0, experiment.METHOD_VOLUME: 321.0}
    assert fallback["fallback"] is True
    assert fallback["fallbackReason"].startswith("current_feature_unavailable:")

    training = fit["fit"]["trainingRows"]
    assert all(dt.date.fromisoformat(row["target"]) <= dt.date.fromisoformat(fit["fit"]["fitCutoff"]) for row in training)
    assert all(dt.date.fromisoformat(row["featureCutoff"]) == dt.date.fromisoformat(row["target"]) - dt.timedelta(days=row["horizon"] + 2) for row in training)


@pytest.mark.parametrize("bad_intercept", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_raw_log_ratio_in_either_model_falls_back_both(bad_intercept):
    feature = {"state": "available", "p0": {"price": 100.0},
               "features": [0.0] * len(experiment.PRICE_FEATURE_NAMES),
               "volumeFeatures": [0.0] * len(experiment.VOLUME_FEATURE_NAMES)}
    def model(names, intercept):
        return {"featureNames": list(names), "scalerMean": [0.0] * len(names),
                "scalerScale": [1.0] * len(names), "coefficients": [0.0] * len(names),
                "intercept": intercept}
    fit = {"fitFailure": None, "models": {
        experiment.METHOD_PRICE: model(experiment.PRICE_FEATURE_NAMES, 0.0),
        experiment.METHOD_VOLUME: model(experiment.VOLUME_FEATURE_NAMES, bad_intercept)}}

    predictions, fallback = experiment._predict_pair(fit, feature, 321.0)

    assert predictions == {experiment.METHOD_PRICE: 321.0, experiment.METHOD_VOLUME: 321.0}
    assert fallback["fallback"] is True
    assert "nonfinite_log_ratio_prediction" in fallback["fallbackReason"]


def test_synthetic_manifest_builder_covers_exact_grid_and_fixes_monthly_models():
    start, end = dt.date(2025, 6, 1), dt.date(2026, 6, 30)
    observations = {p: {} for p in experiment.PRODUCTS}
    for offset in range((end - start).days + 1):
        day = start + dt.timedelta(days=offset)
        for index, product in enumerate(experiment.PRODUCTS):
            price = 100 + index * 10 + offset * 0.1 + 2 * math.sin(offset / 9)
            quantity = 100 + index + offset * 0.2 + 4 * math.cos(offset / 11)
            observations[product][day.isoformat()] = {"price": str(price), "priceReason": None,
                "productNames": [experiment.PRODUCT_NAMES[product]], "nameConflictCount": 0,
                "quantitySum": str(quantity), "negativeQuantityCount": 0, "rowCount": 1}
    baseline_rows = []
    for offset in range(273):
        target = dt.date(2026, 1, 1) + dt.timedelta(days=offset)
        for product in experiment.PRODUCTS:
            for horizon in experiment.HORIZONS:
                origin = target - dt.timedelta(days=horizon)
                baseline_rows.append({"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                    "horizon": horizon, "predictions": {"fixed_blend_50_50": 110.0, "calendar_sarima": 111.0}, "lineage": {}})
    parsed = {"archiveSha256": experiment.ARCHIVE_SHA256, "archiveByteCount": 123,
        "sourceRowCount": len(observations) * (end - start).days, "periods": [],
        "exactDuplicateRowCount": 0, "observations": observations}
    doc = experiment._build_from_parsed(parsed, {"forecasts": baseline_rows}, experiment.BASELINE_MANIFEST_SHA256)
    experiment.verify_manifest(doc)
    assert doc["plannedForecastCount"] == 7240
    assert len(doc["monthlyFits"]) == 28
    assert all("actual" not in row and "targetValue" not in row for row in doc["forecasts"])
    assert any(fit["models"] for fit in doc["monthlyFits"].values())


def _answer_free_manifest():
    forecasts = []
    for target in (experiment.TARGET_START + dt.timedelta(days=i) for i in range(experiment.TARGET_DAYS)):
        for product in experiment.PRODUCTS:
            for horizon in experiment.HORIZONS:
                origin = target - dt.timedelta(days=horizon)
                cutoff = origin - dt.timedelta(days=2)
                feature = {"cutoff": cutoff.isoformat(), "p0": None, "p7": None, "p28": None,
                           "recentVolume": {"dates": []}, "priorVolume": {"dates": []}}
                forecasts.append({"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                    "horizon": horizon, "cutoff": cutoff.isoformat(),
                    "predictions": {method: 100.0 for method in experiment.METHODS},
                    "lineage": {"feature": feature, "fallback": {"fallback": True, "fallbackReason": "synthetic"}}})
    doc = {"schemaVersion": 1, "protocolId": experiment.PROTOCOL_ID, "phase": "forecast_only",
        "targetValuesAttached": False, "protocol": experiment.PROTOCOL,
        "source": {"archiveSha256": experiment.ARCHIVE_SHA256},
        "baselineManifestSha256": experiment.BASELINE_MANIFEST_SHA256,
        "codeSha256": experiment._sha(experiment.Path(experiment.__file__).read_bytes()),
        "plannedForecastCount": experiment.PLANNED_COUNT, "forecasts": forecasts, "monthlyFits": {}}
    doc["manifestSha256"] = experiment._hash(doc)
    return doc


def test_manifest_tamper_and_full_replay_mismatch_are_rejected(monkeypatch):
    doc = _answer_free_manifest()
    experiment.verify_manifest(doc)
    changed = copy.deepcopy(doc)
    changed["forecasts"][0]["predictions"][experiment.METHOD_VOLUME] = 105.0
    with pytest.raises(ValueError, match="content hash"):
        experiment.verify_manifest(changed)

    rebuilt = copy.deepcopy(doc)
    rebuilt["forecasts"][0]["predictions"][experiment.METHOD_VOLUME] = 101.0
    rebuilt["manifestSha256"] = experiment._hash({k: v for k, v in rebuilt.items() if k != "manifestSha256"})
    monkeypatch.setattr(experiment, "build_manifest", lambda archive, baseline: rebuilt)
    with pytest.raises(ValueError, match="full trend-volume replay mismatch"):
        experiment._verify_replay(doc, b"synthetic archive", b"synthetic baseline")


def test_dm_family_has_240_comparisons_per_bandwidth_and_constant_decimal_guard():
    target = dt.date(2026, 1, 10)
    selected = []
    actuals = {p: {target: 100.0} for p in experiment.PRODUCTS}
    for product in experiment.PRODUCTS:
        for horizon in experiment.HORIZONS:
            origin = target - dt.timedelta(days=horizon)
            selected.append({"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                "horizon": horizon, "predictions": {method: 101.0 for method in experiment.METHODS},
                "lineage": {"feature": {"trend7": 0.1}, "fallback": {"fallback": False, "fallbackReason": None}}})
    report = experiment._period_report(selected, actuals)
    assert len(report["dmComparisons"]) == 480
    assert report["holmFamilySizePerBandwidth"] == 240
    assert all(row["dm"]["status"] == "insufficient_pairs" for row in report["dmComparisons"])

    constant = {dt.date(2026, 1, 1) + dt.timedelta(days=i): 0.1 for i in range(40)}
    for bandwidth in experiment.BANDWIDTHS:
        result = experiment._dm(constant, bandwidth)
        assert result["status"] == "zero_or_invalid_hac_variance"
        assert result["pValue"] == 1.0


def test_create_only_output_refuses_overwrite(tmp_path):
    path = tmp_path / "nested" / "forecast.json"
    payload = {"ok": True}
    experiment.write_create_only(path, payload)
    assert json.loads(path.read_text()) == payload
    with pytest.raises(FileExistsError):
        experiment.write_create_only(path, payload)


def test_public_input_loader_rejects_unpinned_archive_before_parsing():
    with pytest.raises(ValueError, match="national archive byte SHA-256 mismatch"):
        experiment._verified_inputs(b"not the pinned archive", b"not the pinned manifest")
