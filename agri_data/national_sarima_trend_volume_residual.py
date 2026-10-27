"""Frozen limited trend/volume residual correction on national SARIMA forecasts."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import direct_horizon, national_median_ensemble as baseline_inputs
from . import national_price_selection as base
from . import national_transfer, national_trend_volume_interaction as trend

PROTOCOL_ID = "national-sarima-trend-volume-residual-2026-10-09"
PRODUCTS, HORIZONS = trend.PRODUCTS, trend.HORIZONS
TARGET_START, TARGET_END, TARGET_DAYS = trend.TARGET_START, trend.TARGET_END, trend.TARGET_DAYS
PLANNED_COUNT = trend.PLANNED_COUNT
WINDOWS = trend.WINDOWS
METHOD_PRICE = "sarima_residual_price_only"
METHOD_VOLUME = "sarima_residual_volume_interaction"
BASELINES = ("calendar_sarima", "fixed_blend_50_50")
METHODS = (METHOD_PRICE, METHOD_VOLUME, *BASELINES)
COMPARATORS = (METHOD_PRICE, *BASELINES)
BANDWIDTHS = trend.BANDWIDTHS
MIN_PAIRS, MIN_TRAINING_TARGET_DAYS = trend.MIN_PAIRS, trend.MIN_TRAINING_TARGET_DAYS
TRAINING_LOOKBACK_DAYS, RIDGE_ALPHA = trend.TRAINING_LOOKBACK_DAYS, trend.RIDGE_ALPHA
RESIDUAL_CLIP = 0.10
PRICE_FEATURE_NAMES, VOLUME_FEATURE_NAMES = trend.PRICE_FEATURE_NAMES, trend.VOLUME_FEATURE_NAMES
ARCHIVE_SHA256 = national_transfer.ARCHIVE_SHA256
BASELINE_MANIFEST_PATH = baseline_inputs.PROJECT_ROOT / baseline_inputs.MANIFEST_PATHS["development"]
BASELINE_MANIFEST_SHA256 = baseline_inputs.MANIFEST_SHA256["development"]
BASELINE_CONTENT_SHA256 = baseline_inputs.MANIFEST_CONTENT_SHA256["development"]
OUTPUT_DIR = baseline_inputs.PROJECT_ROOT / "outputs/experiments/national-sarima-trend-volume-residual-20261009"
MANIFEST_PATH, REPORT_PATH = OUTPUT_DIR / "forecast-manifest.json", OUTPUT_DIR / "report.json"
FAMILY_SIZE_PER_BANDWIDTH = len(PRODUCTS) * len(HORIZONS) * len(COMPARATORS) * 2
PROTOCOL = {
    "id": PROTOCOL_ID, "archiveSha256": ARCHIVE_SHA256,
    "baselineManifestSha256": BASELINE_MANIFEST_SHA256,
    "baselineContentSha256": BASELINE_CONTENT_SHA256,
    "trainingBaseline": "all pinned development SARIMA issued forecast rows for 2025-10-01 through 2026-06-30; training rows require positive actual and positive nonfallback SARIMA",
    "targetWindow": [TARGET_START.isoformat(), TARGET_END.isoformat()],
    "plannedForecasts": PLANNED_COUNT,
    "features": "same o-2 cutoff, p0/p7/p28, trend7/trend28, abs(trend7), horizon/10, target-day annual sin/cos, p0 age, signed-volume x and x*trend7/x*trend28 as national_trend_volume_interaction",
    "monthlyFit": "per product, pooled horizons 1..10; first planned origin per origin-month; targets <= fit origin-2 and in prior 120 calendar days; each row features at its own origin-2; only positive actual, positive nonfallback historical SARIMA and complete common features; same rows; >=60 distinct target days",
    "models": {"pipeline": "StandardScaler + Ridge(alpha=100)", "priceOnlyFeatures": list(PRICE_FEATURE_NAMES),
        "volumeFeatures": list(VOLUME_FEATURE_NAMES), "target": "log(actual / historical SARIMA forecast)",
        "output": "current SARIMA * exp(clip(predicted log residual, -0.10, +0.10))"},
    "fallback": "both corrections fall back to current positive SARIMA when fit/features/prediction are invalid; if current SARIMA is not positive finite, both use fixed 50:50; preserve reason",
    "evaluation": "Jan-Mar and Apr-Jun; common positive target and positive predictions across both corrections/SARIMA/fixed 50:50; 40-cell equal weight; 3 comparator x 4 product x 10 horizon x AE/APE = 240 DM tests per bandwidth; calendar Bartlett HAC9/HAC28 and separate Holm alpha .05",
    "interpretation": "two-window retrospective point-estimate gate only; no independent or official score evidence; first-publication timing unverified; nonsignificance is not equivalence",
}


def _sha(raw: bytes) -> str:
    return trend._sha(raw)


def _hash(value: Any) -> str:
    return trend._hash(value)


def _date(value: str) -> dt.date:
    return trend._date(value)


def _positive(value: Any) -> float | None:
    return trend._positive(value)


def _expected_keys() -> set[tuple[str, str, str, int]]:
    return trend._expected_keys()


def _dev_expected_keys() -> set[tuple[str, str, str, int]]:
    start, end = dt.date(2025, 10, 1), dt.date(2026, 6, 30)
    return {(p, (target-dt.timedelta(days=h)).isoformat(), target.isoformat(), h)
            for target in (start + dt.timedelta(days=i) for i in range((end-start).days+1))
            for p in PRODUCTS for h in HORIZONS}


def _source_digests(parsed: dict) -> dict:
    return {"archiveSha256": parsed.get("archiveSha256"),
            "combinedObservationsSha256": base._canonical_hash(parsed.get("observations"))}


def _verify_baseline_doc(doc: dict, raw: bytes) -> dict:
    if _sha(raw) != BASELINE_MANIFEST_SHA256:
        raise ValueError("development baseline manifest byte SHA-256 mismatch")
    if doc.get("manifestSha256") != BASELINE_CONTENT_SHA256:
        raise ValueError("development baseline manifest internal SHA mismatch")
    body = dict(doc); body.pop("manifestSha256", None)
    if base._canonical_hash(body) != doc.get("manifestSha256"):
        raise ValueError("development baseline manifest content hash mismatch")
    base._verify_manifest(doc)
    if doc.get("window") != ["2025-10-01", "2026-06-30"]:
        raise ValueError("development baseline window mismatch")
    rows = doc.get("forecasts")
    if not isinstance(rows, list) or len(rows) != len(_dev_expected_keys()):
        raise ValueError("development baseline planned count mismatch")
    observed = set()
    for row in rows:
        key = baseline_inputs._forecast_identity(row)
        observed.add(key)
        if "actual" in row or "targetValue" in row:
            raise ValueError("answer-bearing baseline forecast row")
        predictions = row.get("predictions")
        if not isinstance(predictions, dict) or not {"calendar_sarima", "fixed_blend_50_50"}.issubset(predictions):
            raise ValueError("development baseline is missing a required comparator")
    if observed != _dev_expected_keys():
        raise ValueError("development baseline forecast grid mismatch")
    return doc


def _verified_inputs(archive_bytes: bytes, baseline_bytes: bytes) -> tuple[dict, dict, str]:
    if _sha(archive_bytes) != ARCHIVE_SHA256:
        raise ValueError("national archive byte SHA-256 mismatch")
    parsed = national_transfer.parse_archive(archive_bytes, expected_sha256=ARCHIVE_SHA256)
    try:
        baseline_doc = json.loads(baseline_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("development baseline is not valid JSON") from exc
    _verify_baseline_doc(baseline_doc, baseline_bytes)
    if baseline_doc.get("sourceDigests") != _source_digests(parsed):
        raise ValueError("development baseline source digests differ from pinned national archive")
    return parsed, baseline_doc, _sha(baseline_bytes)


def _baseline_rows(doc: dict) -> dict[tuple, dict]:
    rows = {}
    for row in doc["forecasts"]:
        key = baseline_inputs._forecast_identity(row)
        if key in rows:
            raise ValueError("duplicate development baseline forecast key")
        rows[key] = row
    return rows


def _fit_residual_models(product: str, series: trend._Series, baseline_rows: dict[tuple, dict], fit_origin: dt.date) -> dict:
    fit_cutoff = fit_origin - dt.timedelta(days=2)
    target_start = fit_cutoff - dt.timedelta(days=TRAINING_LOOKBACK_DAYS-1)
    train_rows = []
    by_horizon = {str(h): 0 for h in HORIZONS}
    for target in series.price_days:
        if not target_start <= target <= fit_cutoff:
            continue
        actual = series.rows[target]["price"]
        for horizon in HORIZONS:
            historical_origin = target - dt.timedelta(days=horizon)
            key = (product, historical_origin.isoformat(), target.isoformat(), horizon)
            base_row = baseline_rows.get(key)
            if base_row is None:
                continue
            sarima = _positive(base_row.get("predictions", {}).get("calendar_sarima"))
            sarima_lineage = base_row.get("lineage", {}).get("calendar_sarima", {})
            if sarima is None or sarima_lineage.get("fallback") is not False or sarima_lineage.get("nativeStatus") != "available":
                continue
            feature = trend._features(series, target, horizon)
            if feature.get("state") != "available":
                continue
            residual = math.log(actual / sarima)
            if not math.isfinite(residual):
                continue
            source_keys = trend._observation_keys(product, feature, target)
            row = {"product": product, "target": target.isoformat(), "origin": historical_origin.isoformat(),
                "horizon": horizon, "baselineForecastKey": [product, historical_origin.isoformat(), target.isoformat(), horizon],
                "featureCutoff": feature["cutoff"], "sarimaPrediction": sarima,
                "actualTargetPrice": actual, "targetLogResidual": residual,
                "priceFeatures": feature["features"], "volumeFeatures": feature["volumeFeatures"],
                "sourceObservationKeys": source_keys, "sourceObservationKeysSha256": _hash(source_keys)}
            train_rows.append(row); by_horizon[str(horizon)] += 1
    target_dates = sorted({row["target"] for row in train_rows})
    fit_info = {"fitOrigin": fit_origin.isoformat(), "fitCutoff": fit_cutoff.isoformat(),
        "trainingTargetStart": target_start.isoformat(), "trainingTargetEnd": fit_cutoff.isoformat(),
        "uniqueTrainingTargetDays": len(target_dates), "trainingRowCount": len(train_rows),
        "trainingRowsByHorizon": by_horizon, "trainingTargetDates": target_dates,
        "trainingRowsSha256": _hash(train_rows),
        "trainingSourceObservationKeysSha256": _hash(sorted({k for r in train_rows for k in r["sourceObservationKeys"]})),
        "trainingRows": train_rows}
    result = {"fit": fit_info, "models": {}, "fitFailure": None}
    if len(target_dates) < MIN_TRAINING_TARGET_DAYS:
        result["fitFailure"] = f"insufficient_unique_training_target_days:{len(target_dates)}"
        return result
    y = np.asarray([r["targetLogResidual"] for r in train_rows], dtype=float)
    try:
        for method, feature_name, names in ((METHOD_PRICE, "priceFeatures", PRICE_FEATURE_NAMES),
                                             (METHOD_VOLUME, "volumeFeatures", VOLUME_FEATURE_NAMES)):
            matrix = np.asarray([r[feature_name] for r in train_rows], dtype=float)
            scaler = StandardScaler(); scaled = scaler.fit_transform(matrix)
            estimator = Ridge(alpha=RIDGE_ALPHA).fit(scaled, y)
            means = [float(v) for v in scaler.mean_]
            scales = [float(v) for v in scaler.scale_]
            coefficients = [float(v) for v in estimator.coef_]
            intercept = float(estimator.intercept_)
            if not all(math.isfinite(v) for v in (*means, *scales, *coefficients, intercept)):
                raise ValueError("nonfinite fitted residual model parameter")
            result["models"][method] = {"featureNames": list(names), "scalerMean": means,
                "scalerScale": scales, "coefficients": coefficients, "intercept": intercept,
                "alpha": RIDGE_ALPHA, "trainingRowsSha256": fit_info["trainingRowsSha256"]}
    except Exception as exc:
        result["models"] = {}; result["fitFailure"] = f"{type(exc).__name__}:{exc}"
    return result


def _correct_pair(fit: dict, feature: dict, sarima: float | None, fixed: float | None) -> tuple[dict, dict]:
    if sarima is None:
        fallback = fixed
        return {METHOD_PRICE: fallback, METHOD_VOLUME: fallback}, {"fallback": True,
            "fallbackBase": "fixed_blend_50_50", "fallbackReason": "current_sarima_unavailable"}
    if fit.get("fitFailure"):
        return {METHOD_PRICE: sarima, METHOD_VOLUME: sarima}, {"fallback": True,
            "fallbackBase": "calendar_sarima", "fallbackReason": fit["fitFailure"]}
    if feature.get("state") != "available":
        return {METHOD_PRICE: sarima, METHOD_VOLUME: sarima}, {"fallback": True,
            "fallbackBase": "calendar_sarima", "fallbackReason": f"current_feature_unavailable:{feature.get('state')}"}
    output = {}
    try:
        for method, feature_name in ((METHOD_PRICE, "features"), (METHOD_VOLUME, "volumeFeatures")):
            model = fit["models"][method]
            vector = np.asarray(feature[feature_name], dtype=float)
            scaled = (vector-np.asarray(model["scalerMean"]))/np.asarray(model["scalerScale"])
            raw_residual = float(np.dot(scaled, np.asarray(model["coefficients"]))+model["intercept"])
            if not math.isfinite(raw_residual):
                raise ValueError("nonfinite_raw_log_residual_prediction")
            clipped = max(-RESIDUAL_CLIP, min(RESIDUAL_CLIP, raw_residual))
            corrected = sarima*math.exp(clipped)
            if not math.isfinite(corrected) or corrected <= 0:
                raise ValueError("nonpositive_or_nonfinite_corrected_prediction")
            output[method] = corrected
    except Exception as exc:
        return {METHOD_PRICE: sarima, METHOD_VOLUME: sarima}, {"fallback": True,
            "fallbackBase": "calendar_sarima", "fallbackReason": f"candidate_prediction_failure:{type(exc).__name__}:{exc}"}
    return output, {"fallback": False, "fallbackBase": None, "fallbackReason": None,
        "rawResiduals": {method: (math.log(output[method]/sarima)) for method in output},
        "clipBounds": [-RESIDUAL_CLIP, RESIDUAL_CLIP]}


def _build_from_parsed(parsed: dict, baseline_doc: dict, baseline_file_sha: str) -> dict:
    digests = _source_digests(parsed)
    if baseline_doc.get("sourceDigests") != digests:
        raise ValueError("development baseline source digests differ from archive")
    rows = _baseline_rows(baseline_doc)
    expected_dev = _dev_expected_keys()
    if set(rows) != expected_dev:
        raise ValueError("development baseline forecast grid mismatch")
    expected = _expected_keys()
    series = {product: trend._Series(trend._source_days(parsed, product)) for product in PRODUCTS}
    first_origin = {}
    for product, origin_text, _, _ in expected:
        origin = _date(origin_text); key = (product, origin.strftime("%Y-%m"))
        first_origin[key] = min(first_origin.get(key, origin), origin)
    monthly_fits = {(product, month): _fit_residual_models(product, series[product], rows, origin)
        for (product, month), origin in sorted(first_origin.items())}
    forecasts = []
    for product, origin_text, target_text, horizon in sorted(expected, key=lambda k:(k[2], k[0], k[3])):
        origin, target = _date(origin_text), _date(target_text)
        base_row = rows[(product, origin_text, target_text, horizon)]
        sarima = _positive(base_row.get("predictions", {}).get("calendar_sarima"))
        fixed = _positive(base_row.get("predictions", {}).get("fixed_blend_50_50"))
        feature = trend._features(series[product], target, horizon)
        fit_key = (product, origin.strftime("%Y-%m")); fit = monthly_fits[fit_key]
        corrections, fallback = _correct_pair(fit, feature, sarima, fixed)
        predictions = {METHOD_PRICE: corrections[METHOD_PRICE], METHOD_VOLUME: corrections[METHOD_VOLUME],
            "calendar_sarima": sarima, "fixed_blend_50_50": fixed}
        forecasts.append({"product": product, "origin": origin_text, "target": target_text,
            "horizon": horizon, "cutoff": (origin-dt.timedelta(days=2)).isoformat(),
            "predictions": predictions,
            "lineage": {"fitKey": f"{product}/{fit_key[1]}", "feature": feature,
                "fallback": fallback, "baselineSarimaLineage": base_row.get("lineage", {}).get("calendar_sarima", {}),
                "baselineFixedLineage": base_row.get("lineage", {}).get("fixed_blend_50_50", {})}})
    doc = {"schemaVersion": 1, "protocolId": PROTOCOL_ID, "phase": "forecast_only",
        "targetValuesAttached": False, "protocol": PROTOCOL,
        "source": {"archiveSha256": parsed.get("archiveSha256"), "sourceDigests": digests,
            "archiveByteCount": parsed.get("archiveByteCount"), "sourceRowCount": parsed.get("sourceRowCount"),
            "periods": parsed.get("periods"), "exactDuplicateRowCount": parsed.get("exactDuplicateRowCount")},
        "baselineManifestSha256": baseline_file_sha, "baselineManifestContentSha256": BASELINE_CONTENT_SHA256,
        "codeSha256": _sha(Path(__file__).read_bytes()), "products": list(PRODUCTS), "horizons": list(HORIZONS),
        "window": [TARGET_START.isoformat(), TARGET_END.isoformat()], "plannedForecastCount": len(forecasts),
        "monthlyFits": {f"{product}/{month}": fit for (product, month), fit in sorted(monthly_fits.items())},
        "forecasts": forecasts}
    doc["manifestSha256"] = _hash(doc)
    return doc


def build_manifest(archive_bytes: bytes, baseline_bytes: bytes) -> dict:
    parsed, baseline_doc, baseline_file_sha = _verified_inputs(archive_bytes, baseline_bytes)
    return _build_from_parsed(parsed, baseline_doc, baseline_file_sha)


def verify_manifest(doc: dict) -> None:
    if doc.get("protocolId") != PROTOCOL_ID or doc.get("phase") != "forecast_only" or doc.get("targetValuesAttached") is not False:
        raise ValueError("candidate manifest must be answer-free")
    if doc.get("codeSha256") != _sha(Path(__file__).read_bytes()):
        raise ValueError("candidate code hash mismatch")
    body = dict(doc); saved = body.pop("manifestSha256", None)
    if saved != _hash(body):
        raise ValueError("candidate manifest content hash mismatch")
    if doc.get("baselineManifestSha256") != BASELINE_MANIFEST_SHA256 or doc.get("baselineManifestContentSha256") != BASELINE_CONTENT_SHA256:
        raise ValueError("pinned baseline manifest hashes mismatch")
    if doc.get("source", {}).get("archiveSha256") != ARCHIVE_SHA256:
        raise ValueError("candidate archive hash mismatch")
    rows, seen = doc.get("forecasts"), set()
    if not isinstance(rows, list) or len(rows) != PLANNED_COUNT or doc.get("plannedForecastCount") != PLANNED_COUNT:
        raise ValueError("planned forecast count mismatch")
    for row in rows:
        key = (row.get("product"), row.get("origin"), row.get("target"), row.get("horizon"))
        if key in seen:
            raise ValueError("duplicate candidate forecast key")
        seen.add(key)
        if "actual" in row or "targetValue" in row:
            raise ValueError("target answer leaked into manifest")
        origin, target = _date(row["origin"]), _date(row["target"])
        if target-origin != dt.timedelta(days=int(row["horizon"])) or row.get("cutoff") != (origin-dt.timedelta(days=2)).isoformat():
            raise ValueError("forecast origin/cutoff mismatch")
        feature = row.get("lineage", {}).get("feature", {})
        if feature.get("cutoff") != row["cutoff"]:
            raise ValueError("feature cutoff mismatch")
        for keyname in ("p0", "p7", "p28"):
            val = feature.get(keyname)
            if val and _date(val["date"]) > _date(row["cutoff"]):
                raise ValueError("post-cutoff price feature")
        for keyname in ("recentVolume", "priorVolume"):
            if any(_date(day) > _date(row["cutoff"]) for day in feature.get(keyname, {}).get("dates", [])):
                raise ValueError("post-cutoff volume feature")
    if seen != _expected_keys():
        raise ValueError("candidate forecast grid mismatch")
    for fit in doc.get("monthlyFits", {}).values():
        fit_info = fit["fit"]; cutoff = _date(fit_info["fitCutoff"])
        for row in fit_info.get("trainingRows", []):
            target = _date(row["target"])
            if target > cutoff or _date(row["featureCutoff"]) != target-dt.timedelta(days=int(row["horizon"])+2):
                raise ValueError("training row cutoff leakage")


def _verify_replay(doc: dict, archive_bytes: bytes, baseline_bytes: bytes) -> None:
    verify_manifest(doc)
    rebuilt = build_manifest(archive_bytes, baseline_bytes)
    if doc.get("manifestSha256") != rebuilt.get("manifestSha256") or doc.get("forecasts") != rebuilt.get("forecasts"):
        raise ValueError("full SARIMA residual replay mismatch")


def _period_report(selected: list[dict], actuals: dict[str, dict[dt.date, float]], observations: dict | None = None) -> dict:
    common = []
    for row in selected:
        actual = _positive(actuals.get(row["product"], {}).get(_date(row["target"])))
        pred = {m: _positive(row["predictions"].get(m)) for m in METHODS}
        if actual is not None and all(v is not None for v in pred.values()):
            common.append({**row, "actual": actual, "predictions": pred})
    cells, tests, paired = {}, [], []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            cell = [r for r in common if r["product"] == product and r["horizon"] == horizon]
            metrics = {m: direct_horizon._metrics([r["actual"] for r in cell], [r["predictions"][m] for r in cell]) for m in METHODS}
            cells[f"{product}/{horizon}"] = {"n": len(cell), "metrics": metrics}
            for comparator in COMPARATORS:
                for loss in ("mae", "mape"):
                    if loss == "mae":
                        differences = {_date(r["target"]): abs(r["predictions"][METHOD_VOLUME]-r["actual"])-abs(r["predictions"][comparator]-r["actual"]) for r in cell}
                    else:
                        differences = {_date(r["target"]): 100*(abs(r["predictions"][METHOD_VOLUME]-r["actual"])-abs(r["predictions"][comparator]-r["actual"]))/r["actual"] for r in cell}
                    for bandwidth in BANDWIDTHS:
                        dm = trend._dm(differences, bandwidth)
                        tests.append({"product": product, "horizon": horizon, "comparator": comparator,
                            "loss": loss, "family": "sarima_residual_vs_comparators", "dm": dm})
            for comparator in COMPARATORS:
                paired.append({"product": product, "horizon": horizon, "comparator": comparator, "n": len(cell),
                    "meanPairedMaeDifference": float(np.mean([abs(r["predictions"][METHOD_VOLUME]-r["actual"])-abs(r["predictions"][comparator]-r["actual"]) for r in cell])) if cell else None,
                    "meanPairedMapeDifferencePp": float(np.mean([100*(abs(r["predictions"][METHOD_VOLUME]-r["actual"])-abs(r["predictions"][comparator]-r["actual"]))/r["actual"] for r in cell])) if cell else None})
    for bandwidth in BANDWIDTHS:
        family = [r for r in tests if r["dm"]["calendarHacMaxLagDays"] == bandwidth]
        if len(family) != FAMILY_SIZE_PER_BANDWIDTH:
            raise ValueError("frozen DM family size mismatch")
        adjusted = [{"pValue": r["dm"]["pValue"]} for r in family]
        direct_horizon._holm(adjusted)
        for row, adj in zip(family, adjusted):
            dm = row["dm"]; dm["holmAdjustedPValue"] = adj["holmPValue"]
            dm["holmSignificant"] = bool(adj["holmSignificant"] and dm["status"] == "tested")
            mean = dm.get("meanLossDifference")
            dm["conclusion"] = "inference_unavailable" if dm["status"] != "tested" else "improvement" if dm["holmSignificant"] and mean < 0 else "worsening" if dm["holmSignificant"] and mean > 0 else "inconclusive"
    indexed = {(r["product"], r["horizon"], r["comparator"], r["loss"], r["dm"]["calendarHacMaxLagDays"]): r["dm"] for r in tests}
    joint = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            for comparator in COMPARATORS:
                outcomes = [indexed[(product, horizon, comparator, loss, bandwidth)]["conclusion"]
                            for loss in ("mae", "mape") for bandwidth in BANDWIDTHS]
                if all(value == "improvement" for value in outcomes):
                    conclusion = "improvement"
                elif all(value == "worsening" for value in outcomes):
                    conclusion = "worsening"
                elif "inference_unavailable" in outcomes:
                    conclusion = "inference_unavailable"
                else:
                    conclusion = "inconclusive"
                joint.append({"product": product, "horizon": horizon, "comparator": comparator,
                              "conclusion": conclusion, "byLossAndBandwidth": {
                                  f"{loss}/hac{bandwidth}": indexed[(product, horizon, comparator, loss, bandwidth)]["conclusion"]
                                  for loss in ("mae", "mape") for bandwidth in BANDWIDTHS}})
    complete = all(cells[f"{p}/{h}"]["n"] > 0 for p in PRODUCTS for h in HORIZONS)
    macro = {m: {loss: float(np.mean([cells[f"{p}/{h}"]["metrics"][m][loss] for p in PRODUCTS for h in HORIZONS])) if complete else None
                for loss in ("mae", "mape")} for m in METHODS}
    coverage = {}
    for method in METHODS:
        issued = sum(_positive(r["predictions"].get(method)) is not None for r in selected)
        on_target = sum(_date(r["target"]) in actuals.get(r["product"], {}) and _positive(r["predictions"].get(method)) is not None for r in selected)
        coverage[method] = {"planned": len(selected), "issued": issued, "issuedOnPositiveTarget": on_target}
    states = {}
    fallback_counts = {}
    for row in selected:
        raw = (observations or {}).get(row["product"], {}).get(row["target"])
        state = "no_observation" if raw is None else raw.get("priceReason") or "valid_positive_price"
        states[state] = states.get(state, 0) + 1
        fb = row["lineage"]["fallback"]
        if fb["fallback"]:
            reason = fb.get("fallbackReason") or "unspecified"
            fallback_counts[reason] = fallback_counts.get(reason, 0) + 1
    trend_diag = {}
    for product in PRODUCTS:
        product_rows = [r for r in common if r["product"] == product]
        for label, predicate in (("trend7Positive", lambda r: r["lineage"]["feature"].get("trend7") is not None and r["lineage"]["feature"]["trend7"] > 0),
                                 ("trend7Nonpositive", lambda r: r["lineage"]["feature"].get("trend7") is not None and r["lineage"]["feature"]["trend7"] <= 0)):
            subset = [r for r in product_rows if predicate(r)]
            trend_diag[f"{product}/{label}"] = {"n": len(subset), "metrics": {m: direct_horizon._metrics([r["actual"] for r in subset], [r["predictions"][m] for r in subset]) for m in METHODS}}
    return {"planned": len(selected), "validCommonPairs": len(common), "complete40Cells": complete,
        "coverage": coverage, "targetSourceStateCounts": dict(sorted(states.items())),
        "candidateFallbackByReason": dict(sorted(fallback_counts.items())), "macro40EqualCell": macro,
        "cells": cells, "pairedCellLosses": paired, "dmComparisons": tests, "jointDmConclusions": joint,
        "holmFamilySizePerBandwidth": FAMILY_SIZE_PER_BANDWIDTH, "trend7SignDiagnostic": trend_diag}


def _practical_gate(period: dict) -> dict:
    macro, complete = period["macro40EqualCell"], period["complete40Cells"]
    baseline_candidates = [(macro[m]["mape"], m) for m in ("calendar_sarima", "fixed_blend_50_50") if macro[m]["mape"] is not None]
    best_mape, best_method = min(baseline_candidates, default=(None, None))
    gain_best = best_mape-macro[METHOD_VOLUME]["mape"] if complete and best_mape is not None else None
    gain_price = macro[METHOD_PRICE]["mape"]-macro[METHOD_VOLUME]["mape"] if complete and macro[METHOD_PRICE]["mape"] is not None else None
    coverage_ok = best_method is not None and all(period["coverage"][METHOD_VOLUME]["issued"] >= period["coverage"][b]["issued"] for b in BASELINES)
    product_changes = {}
    for product in PRODUCTS:
        per_method = {m: {loss: [period["cells"][f"{product}/{h}"]["metrics"][m][loss] for h in HORIZONS] for loss in ("mape", "mae")} for m in METHODS}
        available = all(v is not None for m in METHODS for loss in ("mape", "mae") for v in per_method[m][loss])
        if available:
            best_product = min(BASELINES, key=lambda m: float(np.mean(per_method[m]["mape"])))
            bmape, bmae = float(np.mean(per_method[best_product]["mape"])), float(np.mean(per_method[best_product]["mae"]))
            mape_change = float(np.mean(per_method[METHOD_VOLUME]["mape"]))-bmape
            mae_change = 100*(float(np.mean(per_method[METHOD_VOLUME]["mae"]))-bmae)/bmae if bmae > 0 else None
        else:
            best_product, mape_change, mae_change = None, None, None
        product_changes[product] = {"bestExistingBaselineByMape": best_product, "mapeChangePp": mape_change, "maeChangePct": mae_change}
    product_ok = all(v["mapeChangePp"] is not None and v["mapeChangePp"] <= 1 and v["maeChangePct"] is not None and v["maeChangePct"] <= 5 for v in product_changes.values())
    criteria = {"complete40CellCommonCoverage": complete,
        "mapeAtLeast0_25ppBelowPriceOnly": gain_price is not None and gain_price >= 0.25,
        "mapeAtLeast0_25ppBelowBestExistingBaseline": gain_best is not None and gain_best >= 0.25,
        "issuedCoverageNotBelowEitherExistingBaseline": coverage_ok,
        "noProductExceedsBaselineWorseningLimits": product_ok}
    return {"status": "pass" if all(criteria.values()) else "fail", "criteria": criteria,
        "bestExistingBaseline": best_method, "mapeGainVsBestBaselinePp": gain_best,
        "mapeGainVsPriceOnlyPp": gain_price, "productChanges": product_changes}


def _actual_prices(parsed: dict) -> dict[str, dict[dt.date, float]]:
    return {p: {day: row["price"] for day, row in trend._source_days(parsed, p).items()} for p in PRODUCTS}


def _score_from_inputs(doc: dict, parsed: dict) -> dict:
    actuals = _actual_prices(parsed)
    report = {"protocolId": PROTOCOL_ID, "candidateManifestSha256": doc["manifestSha256"], "periods": {}}
    for start, end in WINDOWS:
        selected = [r for r in doc["forecasts"] if start <= _date(r["target"]) <= end]
        key = f"{start.isoformat()}_{end.isoformat()}"
        period = _period_report(selected, actuals, parsed.get("observations", {}))
        period["predeclaredGate"] = _practical_gate(period)
        report["periods"][key] = period
    gates = [report["periods"][f"{a.isoformat()}_{b.isoformat()}"]["predeclaredGate"] for a, b in WINDOWS]
    report["twoWindowPracticalConclusion"] = "pass" if all(g["status"] == "pass" for g in gates) else "fail"
    report["interpretation"] = PROTOCOL["interpretation"]
    report["reportSha256"] = _hash(report)
    return report


def score_manifest(doc: dict, archive_bytes: bytes, baseline_bytes: bytes) -> dict:
    _verify_replay(doc, archive_bytes, baseline_bytes)
    parsed, _, _ = _verified_inputs(archive_bytes, baseline_bytes)
    return _score_from_inputs(doc, parsed)


def write_create_only(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)+"\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="create an answer-free residual-correction manifest")
    build.add_argument("--archive", type=Path, default=national_transfer.ROOT / "data/raw/national-wholesale-20240708-20260630.zip")
    build.add_argument("--baseline", type=Path, default=BASELINE_MANIFEST_PATH)
    build.add_argument("--output", type=Path, default=MANIFEST_PATH)
    score = sub.add_parser("score", help="replay the manifest before attaching target outcomes")
    score.add_argument("--archive", type=Path, default=national_transfer.ROOT / "data/raw/national-wholesale-20240708-20260630.zip")
    score.add_argument("--baseline", type=Path, default=BASELINE_MANIFEST_PATH)
    score.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    score.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)
    archive_bytes, baseline_bytes = args.archive.read_bytes(), args.baseline.read_bytes()
    if args.command == "build":
        doc = build_manifest(archive_bytes, baseline_bytes); write_create_only(args.output, doc)
        print(json.dumps({"output": str(args.output), "manifestSha256": doc["manifestSha256"], "plannedForecastCount": len(doc["forecasts"])}, sort_keys=True))
    else:
        doc = json.loads(args.manifest.read_bytes()); report = score_manifest(doc, archive_bytes, baseline_bytes)
        write_create_only(args.output, report)
        print(json.dumps({"output": str(args.output), "reportSha256": report["reportSha256"], "conclusion": report["twoWindowPracticalConclusion"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
