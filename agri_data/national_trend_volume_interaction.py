"""Frozen retrospective direct-forecast experiment for price-trend/volume interactions."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import direct_horizon, national_lag_guard as baseline_protocol, national_transfer

PROTOCOL_ID = "national-trend-volume-interaction-2026-10-09"
PRODUCTS = tuple(national_transfer.PRODUCT_CODES)
PRODUCT_NAMES = national_transfer.PRODUCT_NAMES
HORIZONS = tuple(range(1, 11))
TARGET_START = dt.date(2026, 1, 1)
TARGET_END = dt.date(2026, 6, 30)
TARGET_DAYS = (TARGET_END - TARGET_START).days + 1
PLANNED_COUNT = TARGET_DAYS * len(PRODUCTS) * len(HORIZONS)
WINDOWS = ((dt.date(2026, 1, 1), dt.date(2026, 3, 31)),
           (dt.date(2026, 4, 1), dt.date(2026, 6, 30)))
METHOD_PRICE = "ridge_trend_price_only"
METHOD_VOLUME = "ridge_trend_volume_interaction"
BASELINES = ("fixed_blend_50_50", "calendar_sarima")
METHODS = (METHOD_PRICE, METHOD_VOLUME, *BASELINES)
COMPARATORS = (METHOD_PRICE, *BASELINES)
BANDWIDTHS = (9, 28)
MIN_PAIRS = 30
MIN_TRAINING_TARGET_DAYS = 60
TRAINING_LOOKBACK_DAYS = 120
MIN_RECENT_VOLUME_DAYS = 4
MIN_PRIOR_VOLUME_DAYS = 12
PRICE_MAX_AGE = 14
LOG_RATIO_CLIP = 0.25
RIDGE_ALPHA = 100.0
ALPHA = 0.05
FAMILY_SIZE_PER_BANDWIDTH = len(PRODUCTS) * len(HORIZONS) * len(COMPARATORS) * 2
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_PATH = PROJECT_ROOT / "data/raw/national-wholesale-20240708-20260630.zip"
ARCHIVE_SHA256 = national_transfer.ARCHIVE_SHA256
BASELINE_MANIFEST_PATH = PROJECT_ROOT / "outputs/experiments/national-lag-guard-20261009/forecast-manifest.json"
BASELINE_MANIFEST_SHA256 = "6904c61eda899261ff544cb18518d8b38070365eb123a26d3b545ff029d03b6c"
OUTPUT_DIR = PROJECT_ROOT / "outputs/experiments/national-trend-volume-interaction-20261009"
MANIFEST_PATH = OUTPUT_DIR / "forecast-manifest.json"
REPORT_PATH = OUTPUT_DIR / "report.json"
PRICE_FEATURE_NAMES = ("trend7", "trend28", "abs_trend7", "horizon_div_10", "target_day_sin", "target_day_cos", "p0_age_div_14")
VOLUME_FEATURE_NAMES = (*PRICE_FEATURE_NAMES, "volume_log_ratio", "volume_x_trend7", "volume_x_trend28")
PROTOCOL = {
    "id": PROTOCOL_ID,
    "archiveSha256": ARCHIVE_SHA256,
    "baselineManifestSha256": BASELINE_MANIFEST_SHA256,
    "baselineMethods": list(BASELINES),
    "plannedForecasts": PLANNED_COUNT,
    "targetWindow": [TARGET_START.isoformat(), TARGET_END.isoformat()],
    "originCutoff": "target minus horizon is origin; all source features stop at origin minus 2 calendar days",
    "priceFeatures": "p0 latest positive price at/before cutoff (<=14 days old); p7 and p28 latest positive prices at/before cutoff-7/cutoff-28 (<=14 days old at each reference); log(p0/p7), log(p0/p28)",
    "volumeFeature": "log(recent mean / prior mean) of signed national daily net quantity on valid positive-price days; recent origin-8..origin-2 >=4 days; prior origin-29..origin-9 >=12 days; no fill",
    "monthlyFit": "per product, pooled horizons 1..10; fit once at first planned origin of each origin-month; training targets <= fit origin-2 and within the preceding 120 calendar days; each row features reconstructed at its own origin-2 cutoff; same complete price+volume rows for both fits; >=60 distinct target days",
    "models": {"pipeline": "StandardScaler then Ridge(alpha=100)", "priceFeatureOrder": list(PRICE_FEATURE_NAMES), "volumeFeatureOrder": list(VOLUME_FEATURE_NAMES), "target": "log(actual target price / p0)", "prediction": "p0 * exp(clip(predicted log ratio, -0.25, +0.25))"},
    "fallback": "both candidates use the same fixed 50:50 value when fit unavailable, current price or volume feature unavailable, or either candidate prediction is nonfinite/nonpositive",
    "evaluation": "Jan-Mar and Apr-Jun separately; same positive actual and positive predictions across both candidates and fixed blend/SARIMA; 40-cell equal-weight and cell/product paired metrics; DM families 3 comparators x 4 products x 10 horizons x AE/APE = 240 per bandwidth, calendar Bartlett HAC9/HAC28, Holm alpha .05",
    "practicalGate": "both windows: interaction candidate MAPE at least 0.25 pp below price-only and the lower-MAPE of SARIMA/fixed 50:50; issued coverage no lower than either existing baseline; no product more than +1 pp MAPE or +5% MAE worse than the best-MAPE existing baseline",
    "limits": "retrospective revised data, not independent or official score evidence; 10am publication version and July-September volume equivalence are unverified; nonsignificance is not equivalence",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hash(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def _positive(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid ISO date") from exc


def _source_days(parsed: dict, product: str) -> dict[dt.date, dict]:
    result = {}
    for text, row in parsed.get("observations", {}).get(product, {}).items():
        price = _positive(row.get("price"))
        try:
            quantity = float(row["quantitySum"])
        except (KeyError, TypeError, ValueError, OverflowError):
            quantity = math.nan
        if (row.get("priceReason") is None and row.get("productNames") == [PRODUCT_NAMES[product]]
                and row.get("nameConflictCount") == 0 and price is not None and math.isfinite(quantity)):
            result[_date(text)] = {"price": price, "quantity": quantity,
                                   "negativeQuantityCount": int(row.get("negativeQuantityCount", 0)),
                                   "sourceRowCount": int(row.get("rowCount", 0))}
    return result


class _Series:
    def __init__(self, rows: dict[dt.date, dict]):
        self.rows = rows
        self.price_days = sorted(rows)

    def latest(self, reference: dt.date) -> tuple[dt.date, dict] | None:
        idx = bisect_right(self.price_days, reference) - 1
        if idx < 0:
            return None
        day = self.price_days[idx]
        if (reference - day).days > PRICE_MAX_AGE:
            return None
        return day, self.rows[day]


def _volume_window(series: _Series, start: dt.date, end: dt.date) -> dict:
    selected = [(day, series.rows[day]) for day in series.price_days if start <= day <= end]
    negative = [(day, row) for day, row in selected if row["quantity"] < 0]
    return {"start": start.isoformat(), "end": end.isoformat(), "observedDays": len(selected),
            "signedMean": float(np.mean([row["quantity"] for _, row in selected])) if selected else None,
            "negativeSignedQuantityDays": len(negative),
            "negativeQuantitySourceRows": sum(row["negativeQuantityCount"] for _, row in selected),
            "dates": [day.isoformat() for day, _ in selected]}


def _features(series: _Series, target: dt.date, horizon: int) -> dict:
    origin = target - dt.timedelta(days=horizon)
    cutoff = origin - dt.timedelta(days=2)
    p0 = series.latest(cutoff)
    p7_reference = cutoff - dt.timedelta(days=7)
    p28_reference = cutoff - dt.timedelta(days=28)
    p7 = series.latest(p7_reference)
    p28 = series.latest(p28_reference)
    recent = _volume_window(series, origin - dt.timedelta(days=8), origin - dt.timedelta(days=2))
    prior = _volume_window(series, origin - dt.timedelta(days=29), origin - dt.timedelta(days=9))
    feature_row = {"target": target.isoformat(), "origin": origin.isoformat(), "cutoff": cutoff.isoformat(),
        "p0": {"date": p0[0].isoformat(), "price": p0[1]["price"], "ageDays": (cutoff-p0[0]).days} if p0 else None,
        "p7ReferenceDate": p7_reference.isoformat(),
        "p7": {"date": p7[0].isoformat(), "price": p7[1]["price"], "ageDays": (p7_reference-p7[0]).days} if p7 else None,
        "p28ReferenceDate": p28_reference.isoformat(),
        "p28": {"date": p28[0].isoformat(), "price": p28[1]["price"], "ageDays": (p28_reference-p28[0]).days} if p28 else None,
        "recentVolume": recent, "priorVolume": prior, "state": None, "features": None, "volumeX": None}
    if p0 is None:
        feature_row["state"] = "p0_unavailable_or_stale"
        return feature_row
    if p7 is None:
        feature_row["state"] = "p7_unavailable_or_stale"
        return feature_row
    if p28 is None:
        feature_row["state"] = "p28_unavailable_or_stale"
        return feature_row
    if recent["observedDays"] < MIN_RECENT_VOLUME_DAYS:
        feature_row["state"] = "recent_volume_below_minimum_days"
        return feature_row
    if prior["observedDays"] < MIN_PRIOR_VOLUME_DAYS:
        feature_row["state"] = "prior_volume_below_minimum_days"
        return feature_row
    recent_mean, prior_mean = recent["signedMean"], prior["signedMean"]
    if recent_mean is None or prior_mean is None or recent_mean <= 0 or prior_mean <= 0:
        feature_row["state"] = "nonpositive_signed_volume_mean"
        return feature_row
    trend7 = math.log(p0[1]["price"] / p7[1]["price"])
    trend28 = math.log(p0[1]["price"] / p28[1]["price"])
    volume_x = math.log(recent_mean / prior_mean)
    year_day = target.timetuple().tm_yday
    price_features = [trend7, trend28, abs(trend7), horizon / 10.0,
        math.sin(2 * math.pi * year_day / 365.25), math.cos(2 * math.pi * year_day / 365.25),
        feature_row["p0"]["ageDays"] / 14.0]
    if not all(math.isfinite(value) for value in (*price_features, volume_x)):
        feature_row["state"] = "nonfinite_feature"
        return feature_row
    feature_row.update({"state": "available", "features": price_features,
        "volumeX": volume_x, "trend7": trend7, "trend28": trend28,
        "volumeFeatures": [*price_features, volume_x, volume_x * trend7, volume_x * trend28]})
    return feature_row


def _observation_keys(product: str, feature: dict, target: dt.date) -> list[str]:
    days = {target.isoformat()}
    for key in ("p0", "p7", "p28"):
        if feature.get(key):
            days.add(feature[key]["date"])
    for key in ("recentVolume", "priorVolume"):
        days.update(feature.get(key, {}).get("dates", []))
    return [f"{product}:{day}" for day in sorted(days)]


def _fit_models(product: str, series: _Series, fit_origin: dt.date) -> dict:
    fit_cutoff = fit_origin - dt.timedelta(days=2)
    start = fit_cutoff - dt.timedelta(days=TRAINING_LOOKBACK_DAYS - 1)
    rows = []
    by_horizon = {str(h): 0 for h in HORIZONS}
    for target in series.price_days:
        if not start <= target <= fit_cutoff:
            continue
        actual = series.rows[target]["price"]
        for horizon in HORIZONS:
            feature = _features(series, target, horizon)
            if feature["state"] != "available":
                continue
            y = math.log(actual / feature["p0"]["price"])
            if not math.isfinite(y):
                continue
            row = {"product": product, "target": target.isoformat(), "horizon": horizon,
                "featureCutoff": feature["cutoff"], "p0Date": feature["p0"]["date"],
                "p0Price": feature["p0"]["price"], "actualTargetPrice": actual,
                "targetLogRatio": y, "priceFeatures": feature["features"],
                "volumeFeatures": feature["volumeFeatures"], "volumeX": feature["volumeX"],
                "sourceObservationKeys": _observation_keys(product, feature, target)}
            row["sourceObservationKeysSha256"] = _hash(row["sourceObservationKeys"])
            rows.append(row)
            by_horizon[str(horizon)] += 1
    unique_targets = sorted({row["target"] for row in rows})
    fit_base = {"fitOrigin": fit_origin.isoformat(), "fitCutoff": fit_cutoff.isoformat(),
        "trainingTargetStart": start.isoformat(), "trainingTargetEnd": fit_cutoff.isoformat(),
        "uniqueTrainingTargetDays": len(unique_targets), "trainingRowCount": len(rows),
        "trainingRowsByHorizon": by_horizon, "trainingTargetDates": unique_targets,
        "trainingRowsSha256": _hash(rows),
        "trainingSourceObservationKeysSha256": _hash(sorted({key for row in rows for key in row["sourceObservationKeys"]})),
        "trainingRows": rows}
    result = {"fit": fit_base, "models": {}, "fitFailure": None}
    if len(unique_targets) < MIN_TRAINING_TARGET_DAYS:
        result["fitFailure"] = f"insufficient_unique_training_target_days:{len(unique_targets)}"
        return result
    x_price = np.asarray([row["priceFeatures"] for row in rows], dtype=float)
    x_volume = np.asarray([row["volumeFeatures"] for row in rows], dtype=float)
    y_values = np.asarray([row["targetLogRatio"] for row in rows], dtype=float)
    try:
        for method, matrix, names in ((METHOD_PRICE, x_price, PRICE_FEATURE_NAMES),
                                      (METHOD_VOLUME, x_volume, VOLUME_FEATURE_NAMES)):
            scaler = StandardScaler()
            scaled = scaler.fit_transform(matrix)
            estimator = Ridge(alpha=RIDGE_ALPHA)
            estimator.fit(scaled, y_values)
            means = [float(value) for value in scaler.mean_]
            scales = [float(value) for value in scaler.scale_]
            coefs = [float(value) for value in estimator.coef_]
            intercept = float(estimator.intercept_)
            if not all(math.isfinite(v) for v in (*means, *scales, *coefs, intercept)):
                raise ValueError("nonfinite fitted model parameter")
            result["models"][method] = {"featureNames": list(names), "scalerMean": means,
                "scalerScale": scales, "coefficients": coefs, "intercept": intercept,
                "alpha": RIDGE_ALPHA, "trainingRowsSha256": fit_base["trainingRowsSha256"]}
    except Exception as exc:
        result["models"] = {}
        result["fitFailure"] = f"{type(exc).__name__}:{exc}"
    return result


def _predict_pair(fit: dict, feature: dict, fixed_blend: float | None) -> tuple[dict, dict]:
    if fixed_blend is None:
        reason = "fixed_blend_unavailable"
        return {METHOD_PRICE: None, METHOD_VOLUME: None}, {"fallback": True, "fallbackReason": reason}
    if fit.get("fitFailure"):
        return {METHOD_PRICE: fixed_blend, METHOD_VOLUME: fixed_blend}, {"fallback": True, "fallbackReason": fit["fitFailure"]}
    if feature.get("state") != "available":
        return {METHOD_PRICE: fixed_blend, METHOD_VOLUME: fixed_blend}, {"fallback": True,
            "fallbackReason": f"current_feature_unavailable:{feature.get('state')}"}
    p0 = feature["p0"]["price"]
    output = {}
    try:
        for method, vector_key in ((METHOD_PRICE, "features"), (METHOD_VOLUME, "volumeFeatures")):
            model = fit["models"][method]
            values = np.asarray(feature[vector_key], dtype=float)
            scaled = (values - np.asarray(model["scalerMean"])) / np.asarray(model["scalerScale"])
            log_ratio = float(np.dot(scaled, np.asarray(model["coefficients"])) + model["intercept"])
            if not math.isfinite(log_ratio):
                raise ValueError("nonfinite_log_ratio_prediction")
            log_ratio = max(-LOG_RATIO_CLIP, min(LOG_RATIO_CLIP, log_ratio))
            prediction = p0 * math.exp(log_ratio)
            if not math.isfinite(prediction) or prediction <= 0:
                raise ValueError("nonpositive_or_nonfinite_prediction")
            output[method] = prediction
    except Exception as exc:
        return {METHOD_PRICE: fixed_blend, METHOD_VOLUME: fixed_blend}, {"fallback": True,
            "fallbackReason": f"candidate_prediction_failure:{type(exc).__name__}:{exc}"}
    return output, {"fallback": False, "fallbackReason": None}


def _expected_keys() -> set[tuple[str, str, str, int]]:
    return {(p, (target - dt.timedelta(days=h)).isoformat(), target.isoformat(), h)
        for target in (TARGET_START + dt.timedelta(days=i) for i in range(TARGET_DAYS))
        for p in PRODUCTS for h in HORIZONS}


def _build_from_parsed(parsed: dict, baseline_doc: dict, baseline_file_sha: str) -> dict:
    if set(parsed.get("observations", {})) != set(PRODUCTS):
        raise ValueError("national archive product coverage mismatch")
    baseline_rows = {}
    for row in baseline_doc.get("forecasts", []):
        key = (row.get("product"), row.get("origin"), row.get("target"), row.get("horizon"))
        if key in baseline_rows:
            raise ValueError("duplicate baseline forecast key")
        baseline_rows[key] = row
    expected = _expected_keys()
    if not expected.issubset(baseline_rows):
        raise ValueError("baseline forecast manifest does not cover Jan-Jun grid")
    series = {p: _Series(_source_days(parsed, p)) for p in PRODUCTS}
    first_origin: dict[tuple[str, str], dt.date] = {}
    for product, origin_text, _, _ in expected:
        origin = _date(origin_text)
        key = (product, origin.strftime("%Y-%m"))
        first_origin[key] = min(first_origin.get(key, origin), origin)
    fits = {key: _fit_models(product, series[product], origin)
            for (product, month), origin in sorted(first_origin.items())
            for key in [(product, month)]}
    forecasts = []
    for product, origin_text, target_text, horizon in sorted(expected, key=lambda k: (k[2], k[0], k[3])):
        target, origin = _date(target_text), _date(origin_text)
        source_row = baseline_rows[(product, origin_text, target_text, horizon)]
        fixed = _positive(source_row.get("predictions", {}).get("fixed_blend_50_50"))
        sarima = _positive(source_row.get("predictions", {}).get("calendar_sarima"))
        fit_key = (product, origin.strftime("%Y-%m"))
        fit = fits[fit_key]
        feature = _features(series[product], target, horizon)
        predictions, fallback = _predict_pair(fit, feature, fixed)
        predictions["fixed_blend_50_50"] = fixed
        predictions["calendar_sarima"] = sarima
        forecast = {"product": product, "origin": origin_text, "target": target_text,
            "horizon": horizon, "cutoff": (origin-dt.timedelta(days=2)).isoformat(),
            "predictions": predictions,
            "lineage": {"fitKey": f"{product}/{fit_key[1]}", "feature": feature,
                "fallback": fallback, "baselineLineage": source_row.get("lineage", {})}}
        forecasts.append(forecast)
    doc = {"schemaVersion": 1, "protocolId": PROTOCOL_ID, "phase": "forecast_only",
        "targetValuesAttached": False, "protocol": PROTOCOL,
        "source": {"archiveSha256": parsed.get("archiveSha256"),
            "archiveByteCount": parsed.get("archiveByteCount"), "sourceRowCount": parsed.get("sourceRowCount"),
            "periods": parsed.get("periods"), "exactDuplicateRowCount": parsed.get("exactDuplicateRowCount")},
        "baselineManifestSha256": baseline_file_sha,
        "codeSha256": _sha(Path(__file__).read_bytes()), "products": list(PRODUCTS),
        "horizons": list(HORIZONS), "window": [TARGET_START.isoformat(), TARGET_END.isoformat()],
        "plannedForecastCount": len(forecasts), "monthlyFits": {f"{p}/{m}": fit for (p, m), fit in sorted(fits.items())},
        "forecasts": forecasts}
    doc["manifestSha256"] = _hash(doc)
    return doc


def _verified_inputs(archive_bytes: bytes, baseline_bytes: bytes) -> tuple[dict, dict, str]:
    if _sha(archive_bytes) != ARCHIVE_SHA256:
        raise ValueError("national archive byte SHA-256 mismatch")
    parsed = national_transfer.parse_archive(archive_bytes, expected_sha256=ARCHIVE_SHA256)
    baseline_sha = _sha(baseline_bytes)
    if baseline_sha != BASELINE_MANIFEST_SHA256:
        raise ValueError("baseline manifest byte SHA-256 mismatch")
    try:
        baseline_doc = json.loads(baseline_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("baseline manifest is not valid JSON") from exc
    baseline_protocol.verify_manifest(baseline_doc)
    return parsed, baseline_doc, baseline_sha


def build_manifest(archive_bytes: bytes, baseline_bytes: bytes) -> dict:
    parsed, baseline_doc, baseline_sha = _verified_inputs(archive_bytes, baseline_bytes)
    return _build_from_parsed(parsed, baseline_doc, baseline_sha)


def verify_manifest(doc: dict) -> None:
    if doc.get("protocolId") != PROTOCOL_ID or doc.get("phase") != "forecast_only" or doc.get("targetValuesAttached") is not False:
        raise ValueError("candidate manifest must be answer-free")
    if doc.get("codeSha256") != _sha(Path(__file__).read_bytes()):
        raise ValueError("candidate code hash mismatch")
    body = dict(doc)
    saved = body.pop("manifestSha256", None)
    if saved != _hash(body):
        raise ValueError("candidate manifest content hash mismatch")
    rows = doc.get("forecasts")
    expected = _expected_keys()
    seen = set()
    if not isinstance(rows, list) or len(rows) != PLANNED_COUNT or doc.get("plannedForecastCount") != PLANNED_COUNT:
        raise ValueError("planned forecast count mismatch")
    if doc.get("source", {}).get("archiveSha256") != ARCHIVE_SHA256:
        raise ValueError("candidate source archive hash mismatch")
    if doc.get("baselineManifestSha256") != BASELINE_MANIFEST_SHA256:
        raise ValueError("candidate baseline manifest hash mismatch")
    for row in rows:
        key = (row.get("product"), row.get("origin"), row.get("target"), row.get("horizon"))
        if key in seen:
            raise ValueError("duplicate candidate forecast key")
        seen.add(key)
        if "actual" in row or "targetValue" in row:
            raise ValueError("target value leaked into forecast manifest")
        origin, target = _date(row["origin"]), _date(row["target"])
        if row.get("cutoff") != (origin-dt.timedelta(days=2)).isoformat():
            raise ValueError("forecast source cutoff mismatch")
        feature = row.get("lineage", {}).get("feature", {})
        if feature.get("cutoff") != row["cutoff"]:
            raise ValueError("feature cutoff mismatch")
        for field in ("p0", "p7", "p28"):
            ref = feature.get(field)
            if ref is not None:
                if _date(ref["date"]) > _date(row["cutoff"]):
                    raise ValueError("feature uses post-cutoff price")
        for field in ("recentVolume", "priorVolume"):
            if any(_date(day) > _date(row["cutoff"]) for day in feature.get(field, {}).get("dates", [])):
                raise ValueError("feature uses post-cutoff volume")
    if seen != expected:
        raise ValueError("candidate forecast grid mismatch")
    for key, fit in doc.get("monthlyFits", {}).items():
        fit_cutoff = _date(fit["fit"]["fitCutoff"])
        for training in fit["fit"].get("trainingRows", []):
            target = _date(training["target"])
            own_cutoff = target - dt.timedelta(days=int(training["horizon"]) + 2)
            if target > fit_cutoff or _date(training["featureCutoff"]) != own_cutoff:
                raise ValueError("training row leaks future target/source information")


def _verify_replay(doc: dict, archive_bytes: bytes, baseline_bytes: bytes) -> None:
    verify_manifest(doc)
    rebuilt = build_manifest(archive_bytes, baseline_bytes)
    if doc.get("manifestSha256") != rebuilt.get("manifestSha256") or doc.get("forecasts") != rebuilt.get("forecasts"):
        raise ValueError("full trend-volume replay mismatch")


def _dm(diff: dict[dt.date, float], bandwidth: int) -> dict:
    # Guard constant decimal differences before floating-point mean centering.
    if len(diff) >= MIN_PAIRS and all(value == next(iter(diff.values())) for value in diff.values()):
        return {"status": "zero_or_invalid_hac_variance", "n": len(diff),
            "meanLossDifference": float(np.mean(list(diff.values()))), "statistic": None,
            "pValue": 1.0, "calendarHacMaxLagDays": bandwidth}
    result = direct_horizon._dm(diff, bandwidth)
    result["calendarHacMaxLagDays"] = bandwidth
    return result


def _period_report(selected: list[dict], actuals: dict[str, dict[dt.date, float]], observations: dict | None = None) -> dict:
    common = []
    for row in selected:
        actual = _positive(actuals.get(row["product"], {}).get(_date(row["target"])))
        preds = {method: _positive(row["predictions"].get(method)) for method in METHODS}
        if actual is not None and all(value is not None for value in preds.values()):
            common.append({**row, "actual": actual, "predictions": preds})
    cells, comparisons = {}, []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            cell = [row for row in common if row["product"] == product and row["horizon"] == horizon]
            metrics = {method: direct_horizon._metrics([r["actual"] for r in cell], [r["predictions"][method] for r in cell]) for method in METHODS}
            cells[f"{product}/{horizon}"] = {"n": len(cell), "metrics": metrics}
            for comparator in COMPARATORS:
                for loss in ("mae", "mape"):
                    if loss == "mae":
                        diffs = {_date(r["target"]): abs(r["predictions"][METHOD_VOLUME]-r["actual"])-abs(r["predictions"][comparator]-r["actual"]) for r in cell}
                    else:
                        diffs = {_date(r["target"]): 100*(abs(r["predictions"][METHOD_VOLUME]-r["actual"])-abs(r["predictions"][comparator]-r["actual"]))/r["actual"] for r in cell}
                    for bandwidth in BANDWIDTHS:
                        dm = _dm(diffs, bandwidth)
                        comparisons.append({"product": product, "horizon": horizon, "comparator": comparator,
                            "loss": loss, "family": "interaction_vs_comparators", "dm": dm})
    for bandwidth in BANDWIDTHS:
        family = [r for r in comparisons if r["dm"]["calendarHacMaxLagDays"] == bandwidth]
        if len(family) != FAMILY_SIZE_PER_BANDWIDTH:
            raise ValueError("frozen DM family size mismatch")
        holm_rows = [{"pValue": row["dm"]["pValue"]} for row in family]
        direct_horizon._holm(holm_rows)
        for row, adjusted in zip(family, holm_rows):
            dm = row["dm"]
            dm["holmAdjustedPValue"] = adjusted["holmPValue"]
            dm["holmSignificant"] = bool(adjusted["holmSignificant"] and dm["status"] == "tested")
            mean = dm.get("meanLossDifference")
            dm["conclusion"] = "inference_unavailable" if dm["status"] != "tested" else "improvement" if dm["holmSignificant"] and mean < 0 else "worsening" if dm["holmSignificant"] and mean > 0 else "inconclusive"
    all_cells = all(cells[f"{p}/{h}"]["n"] > 0 for p in PRODUCTS for h in HORIZONS)
    macro = {method: {loss: float(np.mean([cells[f"{p}/{h}"]["metrics"][method][loss] for p in PRODUCTS for h in HORIZONS])) if all_cells else None
                for loss in ("mae", "mape")} for method in METHODS}
    coverage = {}
    planned = len(selected)
    for method in METHODS:
        issued = sum(_positive(row["predictions"].get(method)) is not None for row in selected)
        valid_issued = sum(_date(row["target"]) in actuals.get(row["product"], {}) and _positive(row["predictions"].get(method)) is not None for row in selected)
        coverage[method] = {"planned": planned, "issued": issued, "issuedOnPositiveTarget": valid_issued}
    target_states = {}
    for row in selected:
        raw = (observations or {}).get(row["product"], {}).get(row["target"])
        state = "no_observation" if raw is None else raw.get("priceReason") or "valid_positive_price"
        target_states[state] = target_states.get(state, 0) + 1
    exclusions = {"targetPriceUnavailable": sum(_date(row["target"]) not in actuals.get(row["product"], {}) or _positive(actuals.get(row["product"], {}).get(_date(row["target"]))) is None for row in selected),
        "targetSourceStateCounts": dict(sorted(target_states.items())), "candidateFallbackByReason": {}}
    for row in selected:
        fallback = row["lineage"]["fallback"]
        if fallback["fallback"]:
            reason = fallback["fallbackReason"] or "unspecified"
            exclusions["candidateFallbackByReason"][reason] = exclusions["candidateFallbackByReason"].get(reason, 0) + 1
    trend_diag = {}
    for product in PRODUCTS:
        product_rows = [r for r in common if r["product"] == product]
        for label, keep in (("trend7Positive", lambda r: r["lineage"]["feature"].get("trend7") is not None and r["lineage"]["feature"]["trend7"] > 0),
                            ("trend7Nonpositive", lambda r: r["lineage"]["feature"].get("trend7") is not None and r["lineage"]["feature"]["trend7"] <= 0)):
            subset = [r for r in product_rows if keep(r)]
            trend_diag[f"{product}/{label}"] = {"n": len(subset), "metrics": {method: direct_horizon._metrics(
                [r["actual"] for r in subset], [r["predictions"][method] for r in subset]) for method in METHODS}}
    paired = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            cell = [r for r in common if r["product"] == product and r["horizon"] == horizon]
            for comparator in COMPARATORS:
                paired.append({"product": product, "horizon": horizon, "comparator": comparator, "n": len(cell),
                    "meanPairedMaeDifference": float(np.mean([abs(r["predictions"][METHOD_VOLUME]-r["actual"])-abs(r["predictions"][comparator]-r["actual"]) for r in cell])) if cell else None,
                    "meanPairedMapeDifferencePp": float(np.mean([100*(abs(r["predictions"][METHOD_VOLUME]-r["actual"])-abs(r["predictions"][comparator]-r["actual"]))/r["actual"] for r in cell])) if cell else None})
    return {"planned": planned, "validCommonPairs": len(common), "complete40Cells": all_cells,
        "coverage": coverage, "exclusions": exclusions, "macro40EqualCell": macro,
        "cells": cells, "pairedCellLosses": paired, "dmComparisons": comparisons,
        "holmFamilySizePerBandwidth": FAMILY_SIZE_PER_BANDWIDTH,
        "trend7SignDiagnostic": trend_diag}


def _practical_gate(period: dict) -> dict:
    macro = period["macro40EqualCell"]
    complete = period["complete40Cells"]
    baseline_mape = [(macro[m]["mape"], m) for m in BASELINES if macro[m]["mape"] is not None]
    best_mape, best_method = min(baseline_mape, default=(None, None))
    gain_best = best_mape - macro[METHOD_VOLUME]["mape"] if complete and best_mape is not None else None
    gain_price = macro[METHOD_PRICE]["mape"] - macro[METHOD_VOLUME]["mape"] if complete and macro[METHOD_PRICE]["mape"] is not None else None
    coverage_ok = best_method is not None and all(period["coverage"][METHOD_VOLUME]["issued"] >= period["coverage"][m]["issued"] for m in BASELINES)
    product_changes = {}
    for product in PRODUCTS:
        candidate_mape, baseline_mape_for_product = [], {m: [] for m in BASELINES}
        candidate_mae, baseline_mae_for_product = [], {m: [] for m in BASELINES}
        for horizon in HORIZONS:
            cell = period["cells"][f"{product}/{horizon}"]["metrics"]
            candidate_mape.append(cell[METHOD_VOLUME]["mape"]); candidate_mae.append(cell[METHOD_VOLUME]["mae"])
            for m in BASELINES:
                baseline_mape_for_product[m].append(cell[m]["mape"]); baseline_mae_for_product[m].append(cell[m]["mae"])
        if all(v is not None for v in candidate_mape + sum(baseline_mape_for_product.values(), [])):
            bm = {m: float(np.mean(baseline_mape_for_product[m])) for m in BASELINES}
            best_product_method = min(bm, key=bm.get)
            product_mape_change = float(np.mean(candidate_mape)) - bm[best_product_method]
            base_mae = float(np.mean(baseline_mae_for_product[best_product_method]))
            mae_change = (100*(float(np.mean(candidate_mae))-base_mae)/base_mae) if base_mae > 0 else None
        else:
            best_product_method, product_mape_change, mae_change = None, None, None
        product_changes[product] = {"bestExistingBaselineByMape": best_product_method,
            "mapeChangePp": product_mape_change, "maeChangePct": mae_change}
    product_ok = all(v["mapeChangePp"] is not None and v["mapeChangePp"] <= 1.0 and v["maeChangePct"] is not None and v["maeChangePct"] <= 5.0 for v in product_changes.values())
    criteria = {"complete40CellCommonCoverage": complete,
        "mapeAtLeast0_25ppBelowPriceOnly": gain_price is not None and gain_price >= 0.25,
        "mapeAtLeast0_25ppBelowBestExistingBaseline": gain_best is not None and gain_best >= 0.25,
        "issuedCoverageNotBelowEitherExistingBaseline": coverage_ok,
        "noProductExceedsBaselineWorseningLimits": product_ok}
    return {"status": "pass" if all(criteria.values()) else "fail", "criteria": criteria,
        "bestExistingBaseline": best_method, "mapeGainVsBestBaselinePp": gain_best,
        "mapeGainVsPriceOnlyPp": gain_price, "productChanges": product_changes}


def _actual_prices(parsed: dict) -> dict[str, dict[dt.date, float]]:
    out = {}
    for product in PRODUCTS:
        out[product] = {}
        for text, row in parsed.get("observations", {}).get(product, {}).items():
            value = _positive(row.get("price"))
            if row.get("priceReason") is None and row.get("productNames") == [PRODUCT_NAMES[product]] and row.get("nameConflictCount") == 0 and value is not None:
                out[product][_date(text)] = value
    return out


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
    report["interpretation"] = PROTOCOL["limits"]
    report["reportSha256"] = _hash(report)
    return report


def score_manifest(doc: dict, archive_bytes: bytes, baseline_bytes: bytes) -> dict:
    _verify_replay(doc, archive_bytes, baseline_bytes)
    parsed, _, _ = _verified_inputs(archive_bytes, baseline_bytes)
    return _score_from_inputs(doc, parsed)


def write_create_only(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="create a target-answer-free forecast manifest")
    build.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    build.add_argument("--baseline", type=Path, default=BASELINE_MANIFEST_PATH)
    build.add_argument("--output", type=Path, default=MANIFEST_PATH)
    score = sub.add_parser("score", help="replay a manifest, then attach the frozen target actuals")
    score.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    score.add_argument("--baseline", type=Path, default=BASELINE_MANIFEST_PATH)
    score.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    score.add_argument("--output", type=Path, default=REPORT_PATH)
    args = parser.parse_args(argv)
    archive_bytes = _read_bytes(args.archive)
    baseline_bytes = _read_bytes(args.baseline)
    if args.command == "build":
        doc = build_manifest(archive_bytes, baseline_bytes)
        write_create_only(args.output, doc)
        print(json.dumps({"output": str(args.output), "manifestSha256": doc["manifestSha256"],
                          "plannedForecastCount": doc["plannedForecastCount"]}, sort_keys=True))
    else:
        doc = json.loads(args.manifest.read_bytes())
        report = score_manifest(doc, archive_bytes, baseline_bytes)
        write_create_only(args.output, report)
        print(json.dumps({"output": str(args.output), "reportSha256": report["reportSha256"],
                          "conclusion": report["twoWindowPracticalConclusion"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
