"""Frozen Ridge comparison for robust daily-price distribution features.

Forecast manifests contain no target answers. Scoring and evaluation fitting
are gated by full deterministic replay of the screen artifacts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

from . import path_mlp_forecasts as source
from . import path_mlp_score as stats
from . import robust_price_features as robust

ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = source.PRICES_PATH
SCREEN_COMPONENTS_PATH = source.SCREEN_COMPONENTS_PATH
EVALUATION_COMPONENTS_PATH = source.EVALUATION_COMPONENTS_PATH
OUT_DIR = ROOT / "outputs/experiments/robust-feature-20261010"
SCREEN_MANIFEST_PATH = OUT_DIR / "screen-manifest.json"
SCREEN_REPORT_PATH = OUT_DIR / "screen-report.json"
SCREEN_DECISION_PATH = OUT_DIR / "screen-decision.json"
EVALUATION_MANIFEST_PATH = OUT_DIR / "evaluation-manifest.json"
EVALUATION_REPORT_PATH = OUT_DIR / "evaluation-report.json"

PRODUCTS = source.PRODUCTS
HORIZONS = source.HORIZONS
TRAIN_START = dt.date(2014, 1, 1)
SCREEN_START, SCREEN_END, SCREEN_ORIGIN_START = dt.date(2024, 9, 25), dt.date(2025, 8, 31), dt.date(2024, 9, 15)
EVAL_START, EVAL_END, EVAL_ORIGIN_START = dt.date(2026, 1, 1), dt.date(2026, 9, 27), dt.date(2026, 1, 1)
MIN_TRAIN = 90
BANDWIDTHS = (9, 28)
ALPHA = 0.05
MODELS = ("ridge106", "ridge115")
COMPARISONS = (("ridge115_vs_fixed50", "blend115", "fixed50"),
               ("ridge106_vs_fixed50", "blend106", "fixed50"),
               ("ridge115_vs_ridge106", "blend115", "blend106"))
LOSSES = ("ae", "ape")
PRICES_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
SCREEN_COMPONENTS_SHA256 = "6ade501d5818cca33ad973963d12349f51b5ee3bed27afcd2cc023ae972fed74"
EVALUATION_COMPONENTS_SHA256 = "1497ee362945fcab45353a7b2403be752af3bf63c5f584cc0aedb2cc111ceaa8"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _pretty(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _code_hashes() -> dict[str, str]:
    modules = {"featureHelper": robust, "forecastSourceHelpers": source,
               "directHorizon": source.direct_horizon, "narrowFeatures": source.narrow,
               "richFeatures": source.rich, "featureLibrary": source.library,
               "modeling": source.modeling, "statsHelper": stats, "scorer": sys.modules[__name__]}
    return {name: _sha(Path(module.__file__).read_bytes()) for name, module in modules.items()}


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _date(value: Any) -> dt.date | None:
    return source._date(value)


def _phase_spec(phase: str) -> tuple[dt.date, dt.date, dt.date]:
    if phase == "screen":
        return SCREEN_START, SCREEN_END, SCREEN_ORIGIN_START
    if phase == "evaluation":
        return EVAL_START, EVAL_END, EVAL_ORIGIN_START
    raise ValueError("phase must be screen or evaluation")


def _grid(phase: str) -> list[tuple[str, dt.date, int]]:
    start, end, _ = _phase_spec(phase)
    return [(crop, start + dt.timedelta(days=offset), horizon)
            for crop in PRODUCTS for offset in range((end - start).days + 1) for horizon in HORIZONS]


def _component_map(raw: dict[str, Any], phase: str):
    return source._component_rows(raw, phase)


def _model_record(status: str, prediction: float | None, reason: str | None = None, **extra: Any) -> dict[str, Any]:
    return {"status": status, "reason": reason, "prediction": prediction, **extra}


def _emit(callback: Callable[[dict[str, Any]], None] | None, item: dict[str, Any]) -> None:
    if callback is not None:
        callback(item)


def _features_cached(cache: dict, records, direct_series, crop: str, target: dt.date, horizon: int):
    key = (crop, target, horizon)
    if key not in cache:
        cutoff = target - dt.timedelta(days=horizon + 1)
        causal_records = {day: row for day, row in records[crop].items() if day <= cutoff}
        causal_direct = {day: value for day, value in direct_series[crop].items() if day <= cutoff}
        vector, p0, audit = robust.features(causal_records, causal_direct, target, horizon)
        if audit.get("featureCutoff") != cutoff.isoformat():
            raise ValueError("robust feature helper violated strict origin-1 cutoff")
        if vector is not None:
            if len(vector) != len(robust.FEATURE_NAMES) or len(vector) != 115:
                raise ValueError("robust feature helper must return exactly 115 values")
            if len(robust.ORIGINAL_106_NAMES) != 106 or tuple(robust.FEATURE_NAMES[:106]) != tuple(robust.ORIGINAL_106_NAMES):
                raise ValueError("robust helper does not preserve the exact original 106-feature prefix")
            if _positive(p0) is None:
                raise ValueError("robust helper returned features without a positive p0")
        cache[key] = (vector, p0, audit)
    return cache[key]


def _source_payload(prices_raw: bytes, phase: str):
    prices = json.loads(prices_raw)
    _, end, _ = _phase_spec(phase)
    trimmed, trim_audit = source._trim_prices(prices, end)
    direct_series, records = source._source_rows(trimmed)
    return prices, direct_series, records, trim_audit


def _fit_pair(training: list[tuple[dt.date, float, list[float], float, dict[str, Any]]], feature_count: int):
    if len({row[0] for row in training}) < MIN_TRAIN:
        return None, {"status": "insufficient_training", "distinctTargets": len({row[0] for row in training}), "samples": len(training)}
    try:
        import numpy as np
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise RuntimeError("numpy, scikit-learn and threadpoolctl are required") from exc
    x = np.asarray([row[2][:feature_count] for row in training], dtype=float)
    y = np.asarray([math.log(row[1] / row[3]) for row in training], dtype=float)
    if x.shape != (len(training), feature_count) or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None, {"status": "invalid_training_values", "distinctTargets": len({row[0] for row in training}), "samples": len(training)}
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    try:
        with threadpool_limits(limits=1):
            estimator.fit(x, y)
    except Exception as exc:
        return None, {"status": "fit_failed", "reason": type(exc).__name__, "distinctTargets": len({row[0] for row in training}), "samples": len(training)}
    return estimator, {"status": "fitted", "distinctTargets": len({row[0] for row in training}), "samples": len(training),
                      "maxTrainingTarget": max(row[0] for row in training).isoformat(), "featureCount": feature_count}


def _predict(estimator, feature: list[float] | None, p0: float | None, feature_count: int):
    if estimator is None:
        return _model_record("not_issued", None, "model_not_fitted")
    if feature is None or _positive(p0) is None:
        return _model_record("not_issued", None, "feature_or_p0_unavailable")
    try:
        import numpy as np
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            value = float(estimator.predict(np.asarray([feature[:feature_count]], dtype=float))[0])
        prediction = float(p0) * math.exp(value)
    except (OverflowError, ValueError, FloatingPointError):
        return _model_record("not_issued", None, "inverse_transform_nonfinite")
    if not math.isfinite(prediction) or prediction <= 0:
        return _model_record("not_issued", None, "inverse_transform_nonpositive_or_nonfinite")
    return _model_record("issued", prediction)


def build(prices_raw: bytes, components_raw: bytes, phase: str, *, decision_raw: bytes | None = None,
          screen_manifest_raw: bytes | None = None, screen_report_raw: bytes | None = None,
          progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Create answer-free forecasts. Evaluation replay validation precedes model fitting."""
    start, end, origin_start = _phase_spec(phase)
    selections: dict[str, str] = {crop: "retain_fixed50" for crop in PRODUCTS}
    replay_binding = None
    if phase == "evaluation":
        if not decision_raw or not screen_manifest_raw or not screen_report_raw:
            raise ValueError("evaluation requires frozen screen decision, manifest and report")
        decision = json.loads(decision_raw)
        # The screen scorer replays the screen manifest and recomputes its report before
        # any evaluation fit can begin. Screen component bytes remain pinned.
        replay_binding = validate_screen_artifacts(screen_manifest_raw, prices_raw,
            SCREEN_COMPONENTS_PATH.read_bytes(), screen_report_raw, decision_raw)
        selections = replay_binding["selection"]
        if decision.get("selections") != selections or not any(value in MODELS for value in selections.values()):
            raise ValueError("no screen-selected model to evaluate; stop before evaluation fitting")
    elif any(value is not None for value in (decision_raw, screen_manifest_raw, screen_report_raw)):
        raise ValueError("screen phase cannot accept evaluation selection artifacts")

    pinned_environment = source._check_environment()

    prices_doc, direct_series, records, trim_audit = _source_payload(prices_raw, phase)
    components_doc = json.loads(components_raw)
    component_rows = _component_map(components_doc, phase)
    prices_sha, components_sha = _sha(prices_raw), _sha(components_raw)
    expected_components_sha = SCREEN_COMPONENTS_SHA256 if phase == "screen" else EVALUATION_COMPONENTS_SHA256
    if prices_sha != PRICES_SHA256 or components_sha != expected_components_sha:
        raise ValueError("pinned price/component input SHA-256 mismatch")
    feature_cache: dict[tuple[str, dt.date, int], tuple[list[float] | None, float | None, dict[str, Any]]] = {}
    price_maps = {crop: {day: pair[0] for day, pair in series.items()} for crop, series in direct_series.items()}
    forecast_map = {}
    fit_audit = []
    phase_keys = _grid(phase)
    by_month: dict[tuple[int, int], list[tuple[str, dt.date, int]]] = {}
    for key in phase_keys:
        crop, target, horizon = key
        origin = target - dt.timedelta(days=horizon)
        if origin < origin_start:
            component = source._component_fields(key, component_rows)
            forecast_map[key] = {"crop": crop, "target_date": target.isoformat(), "origin_date": origin.isoformat(),
                "horizon": horizon, "fitOrigin": None, "eligible": False,
                "featureAudit": {"status": "origin_before_phase_start", "featureCutoff": (origin-dt.timedelta(days=1)).isoformat()},
                "ridge106": _model_record("not_issued", None, "origin_before_phase_start"),
                "ridge115": _model_record("not_issued", None, "origin_before_phase_start"),
                "blend106": _model_record("not_issued", None, "origin_before_phase_start"),
                "blend115": _model_record("not_issued", None, "origin_before_phase_start"), **component}
        else:
            by_month.setdefault((origin.year, origin.month), []).append(key)

    for (year, month), month_keys in sorted(by_month.items()):
        fit_origin = min(target - dt.timedelta(days=horizon) for _, target, horizon in month_keys)
        for crop in PRODUCTS:
            crop_keys = [key for key in month_keys if key[0] == crop]
            for horizon in HORIZONS:
                keys = [key for key in crop_keys if key[2] == horizon]
                active = set(MODELS) if phase == "screen" else ({selections[crop]} | ({"ridge106"} if selections[crop] == "ridge115" else set()))
                training = []
                feature_support = {"positivePriceVolumeTargetsBeforeFitOrigin": 0, "featureAvailable": 0,
                    "staleP0": 0, "otherFeatureUnavailable": 0,
                    "robustWindowSupport": {str(window): {"trainingRows": 0, "supported": 0, "unsupported": 0,
                        "observedDays": 0} for window in (7, 28, 90)}}
                for target in sorted(price_maps[crop]):
                    if target < TRAIN_START or target >= fit_origin:
                        continue
                    feature_support["positivePriceVolumeTargetsBeforeFitOrigin"] += 1
                    actual = price_maps[crop][target]
                    vector, p0, audit = _features_cached(feature_cache, records, direct_series, crop, target, horizon)
                    if vector is not None and _positive(p0) is not None:
                        training.append((target, actual, vector, p0, audit))
                        feature_support["featureAvailable"] += 1
                        for window, info in audit.get("windowAudit", {}).items():
                            record = feature_support["robustWindowSupport"][window]
                            record["trainingRows"] += 1
                            record["supported" if info.get("supported") else "unsupported"] += 1
                            record["observedDays"] += info.get("positivePriceVolumeDays", 0)
                    elif audit.get("status") == "stale_p0":
                        feature_support["staleP0"] += 1
                    else:
                        feature_support["otherFeatureUnavailable"] += 1
                estimator_by_model = {}
                audit_by_model = {}
                for model in MODELS:
                    if model not in active:
                        estimator_by_model[model] = None
                        audit_by_model[model] = {"status": "not_selected_for_evaluation", "distinctTargets": 0, "samples": 0}
                    else:
                        count = 106 if model == "ridge106" else 115
                        estimator_by_model[model], audit_by_model[model] = _fit_pair(training, count)
                        audit_by_model[model]["trainingFeatureSupport"] = feature_support
                for target_key in keys:
                    _, target, h = target_key
                    vector, p0, feature_audit = _features_cached(feature_cache, records, direct_series, crop, target, h)
                    component = source._component_fields(target_key, component_rows)
                    row = {"crop": crop, "target_date": target.isoformat(),
                           "origin_date": (target - dt.timedelta(days=h)).isoformat(), "horizon": h,
                           "fitOrigin": fit_origin.isoformat(), "eligible": target - dt.timedelta(days=h) >= origin_start,
                           "featureAudit": feature_audit,
                           "ridge106": _predict(estimator_by_model["ridge106"], vector, p0, 106) if "ridge106" in active else _model_record("not_issued", None, "model_not_selected_for_evaluation"),
                           "ridge115": _predict(estimator_by_model["ridge115"], vector, p0, 115) if "ridge115" in active else _model_record("not_issued", None, "model_not_selected_for_evaluation"),
                           **component}
                    for model, blend in (("ridge106", "blend106"), ("ridge115", "blend115")):
                        pred = row[model]["prediction"] if row[model]["status"] == "issued" else None
                        sarima = component["sarima"]["prediction"] if component["sarima"]["status"] == "issued" else None
                        if pred is None:
                            row[blend] = _model_record("not_issued", None, row[model]["reason"])
                        else:
                            combined = pred if sarima is None else 0.5 * pred + 0.5 * sarima
                            row[blend] = _model_record("issued", combined, None if sarima is not None else "sarima_missing_ridge_only_fallback",
                                                       sarimaFallback=sarima is None)
                    forecast_map[target_key] = row
                for model in MODELS:
                    fit_audit.append({"month": f"{year:04d}-{month:02d}", "crop": crop, "horizon": horizon,
                                      "fitOrigin": fit_origin.isoformat(), "model": model, **audit_by_model[model]})
            _emit(progress, {"event": "monthly_crop_complete", "phase": phase, "month": f"{year:04d}-{month:02d}", "crop": crop})

    env = {**pinned_environment, "python": sys.version.split()[0], "blasThreads": 1}
    forecasts = [forecast_map[key] for key in phase_keys]
    unsigned = {"schemaVersion": 1, "phase": "forecast_only", "phaseName": phase,
        "targetValuesAttached": False, "selectedModels": selections,
        "inputSha256": {"prices": prices_sha, "components": components_sha,
            "code": _code_hashes(),
            **({"screenManifest": replay_binding["manifestSha256"], "screenReport": replay_binding["screenReportSha256"],
                "screenDecision": replay_binding["decisionSha256"]} if replay_binding else {})},
        "protocol": {"trainingStart": TRAIN_START.isoformat(), "minimumDistinctTargets": MIN_TRAIN,
            "fitOriginRule": "earliest eligible origin in each origin month", "featureCountByModel": {"ridge106": 106, "ridge115": 115},
            "features": "strictly through origin-1; p0 age<=14; same positive-price/positive-volume domain",
            "ridgeAlpha": 10.0, "target": "log(actual/p0)", "inverse": "p0*exp(predicted_log_return)",
            "blend": "50:50 with existing same-key SARIMA; ridge-only if SARIMA unavailable; no clipping",
            "selectedModels": selections},
        "environment": env, "sourceAudit": trim_audit, "fitAudit": fit_audit, "forecasts": forecasts}
    unsigned["manifestSha256"] = _sha(_canonical(unsigned))
    return unsigned


def validate_manifest_bytes(manifest_raw: bytes, prices_raw: bytes, components_raw: bytes, phase: str, *,
                            decision_raw: bytes | None = None, screen_manifest_raw: bytes | None = None,
                            screen_report_raw: bytes | None = None) -> dict[str, Any]:
    manifest = json.loads(manifest_raw)
    if manifest.get("phase") != "forecast_only" or manifest.get("phaseName") != phase or manifest.get("targetValuesAttached") is not False:
        raise ValueError("robust feature manifest must be answer-free for requested phase")
    expected = build(prices_raw, components_raw, phase, decision_raw=decision_raw,
                     screen_manifest_raw=screen_manifest_raw, screen_report_raw=screen_report_raw)
    if manifest != expected:
        raise ValueError("robust feature manifest differs from full deterministic replay")
    expected_bytes = json.dumps(expected, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    if manifest_raw != expected_bytes:
        raise ValueError("robust feature manifest is not canonical JSON")
    return manifest


def _write_exclusive(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")


def _forecast_prediction(row: dict[str, Any], field: str) -> float | None:
    item = row.get(field)
    if not isinstance(item, dict) or item.get("status") != "issued":
        return None
    return _positive(item.get("prediction"))


def _key(row: dict[str, Any]) -> tuple[str, dt.date, int]:
    crop, day, horizon = row.get("crop"), _date(row.get("target_date")), row.get("horizon")
    if crop not in PRODUCTS or day is None or isinstance(horizon, bool) or horizon not in HORIZONS:
        raise ValueError("invalid robust forecast key")
    return crop, day, horizon


def _actuals(prices: dict[str, Any], start: dt.date, end: dt.date):
    raw = source._trim_prices(prices, end)[0]
    price_series = source.narrow._series(raw)
    return {(crop, day): value for crop, rows in price_series.items() for day, value in rows.items() if start <= day <= end}


def _metrics(items: list[tuple[float, float]]) -> dict[str, Any]:
    if not items:
        return {"n": 0, "mae": None, "mape": None}
    return {"n": len(items), "mae": math.fsum(abs(pred - actual) for actual, pred in items) / len(items),
            "mape": 100 * math.fsum(abs(pred - actual) / actual for actual, pred in items) / len(items)}


def _score_common(rows, actuals, start, end, active_comparisons=None, *, require_threeway=False):
    eligible = [row for row in rows if start <= _key(row)[1] <= end and row.get("eligible") is True]
    dm, metrics, coverage = [], [], {}
    if active_comparisons is None:
        active_comparisons = {crop: {item[0] for item in COMPARISONS} for crop in PRODUCTS}
    elif not isinstance(active_comparisons, dict):
        active_comparisons = {crop: set(active_comparisons) for crop in PRODUCTS}
    threeway_comparisons = ({item[0] for item in COMPARISONS} if require_threeway is True
                            else set(require_threeway) if require_threeway else set())
    for crop in PRODUCTS:
        for horizon in HORIZONS:
            cell = [row for row in eligible if _key(row)[0] == crop and _key(row)[2] == horizon]
            for comp, a, b in COMPARISONS:
                pairs = []
                for row in cell:
                    key = _key(row)
                    actual = actuals.get((crop, key[1]))
                    pa, pb = _forecast_prediction(row, a), _forecast_prediction(row, b)
                    if comp in threeway_comparisons:
                        both_arms = all(_forecast_prediction(row, model) is not None
                                        for model in ("fixed50", "blend106", "blend115"))
                    else:
                        both_arms = pa is not None and pb is not None
                    if actual is not None and pa is not None and pb is not None and both_arms:
                        pairs.append((key[1], actual, pa, pb))
                ma = _metrics([(actual, pa) for _, actual, pa, _ in pairs])
                mb = _metrics([(actual, pb) for _, actual, _, pb in pairs])
                metrics.append({"crop": crop, "horizon": horizon, "comparison": comp, "n": len(pairs),
                                "uniqueTargets": len({item[0] for item in pairs}),
                                "firstTarget": min((item[0] for item in pairs), default=None).isoformat() if pairs else None,
                                "lastTarget": max((item[0] for item in pairs), default=None).isoformat() if pairs else None,
                                "calendarMonths": len({f"{day.year:04d}-{day.month:02d}" for day, *_ in pairs}),
                                "calendarQuarters": len({f"{day.year:04d}-Q{(day.month-1)//3+1}" for day, *_ in pairs}),
                                "a": ma, "b": mb})
                for loss in LOSSES:
                    for bandwidth in BANDWIDTHS:
                        if comp not in active_comparisons.get(crop, set()):
                            result = {"status": "not_selected_for_evaluation", "n": 0,
                                      "meanLossDifference": None, "statistic": None, "pValue": 1.0}
                        else:
                            diff = {day: (abs(pa-actual)-abs(pb-actual) if loss == "ae"
                                         else abs(pa-actual)/actual-abs(pb-actual)/actual)
                                    for day, actual, pa, pb in pairs}
                            result = stats._dm(diff, bandwidth)
                        dm.append({"crop": crop, "horizon": horizon, "comparison": comp,
                                   "loss": loss, "bandwidth": bandwidth, **result})
    for bandwidth in BANDWIDTHS:
        family = [row for row in dm if row["bandwidth"] == bandwidth]
        expected_family = len(PRODUCTS) * len(HORIZONS) * len(COMPARISONS) * len(LOSSES)
        if len(family) != expected_family:
            raise ValueError(f"frozen DM family must contain exactly {expected_family} tests per bandwidth")
        stats._holm(family)
    coverage_by_cell = []
    for crop in PRODUCTS:
        crop_rows = [row for row in eligible if _key(row)[0] == crop]
        coverage[crop] = {"planned": sum(_key(row)[0] == crop and start <= _key(row)[1] <= end for row in rows),
            "eligible": len(crop_rows),
            "fixed50IssuedEligible": sum(_forecast_prediction(row, "fixed50") is not None for row in crop_rows),
            "ridge106IssuedEligible": sum(_forecast_prediction(row, "blend106") is not None for row in crop_rows),
            "ridge115IssuedEligible": sum(_forecast_prediction(row, "blend115") is not None for row in crop_rows),
            "eligibleFixed50KeysLostByEitherArm": [row["target_date"]+f"/h{row['horizon']}" for row in crop_rows
                if _forecast_prediction(row, "fixed50") is not None and
                (_forecast_prediction(row, "blend106") is None or _forecast_prediction(row, "blend115") is None)]}
        for horizon in HORIZONS:
            cell = [row for row in rows if _key(row)[0] == crop and _key(row)[2] == horizon
                    and start <= _key(row)[1] <= end]
            eligible_cell = [row for row in cell if row.get("eligible") is True]
            coverage_by_cell.append({"crop": crop, "horizon": horizon, "planned": len(cell), "eligible": len(eligible_cell),
                "fixed50Issued": sum(_forecast_prediction(row, "fixed50") is not None for row in eligible_cell),
                "ridge106Issued": sum(_forecast_prediction(row, "blend106") is not None for row in eligible_cell),
                "ridge115Issued": sum(_forecast_prediction(row, "blend115") is not None for row in eligible_cell),
                "commonPairsByComparison": {comp: next(item["n"] for item in metrics if item["crop"] == crop and item.get("horizon") == horizon and item["comparison"] == comp) for comp, _, _ in COMPARISONS},
                "uniqueTargetsByComparison": {comp: next(item["uniqueTargets"] for item in metrics if item["crop"] == crop and item.get("horizon") == horizon and item["comparison"] == comp) for comp, _, _ in COMPARISONS},
                "calendarMonthsByComparison": {comp: next(item["calendarMonths"] for item in metrics if item["crop"] == crop and item.get("horizon") == horizon and item["comparison"] == comp) for comp, _, _ in COMPARISONS},
                "calendarQuartersByComparison": {comp: next(item["calendarQuarters"] for item in metrics if item["crop"] == crop and item.get("horizon") == horizon and item["comparison"] == comp) for comp, _, _ in COMPARISONS}})
    for crop in PRODUCTS:
        for comp, _, _ in COMPARISONS:
            horizon_rows = [row for row in metrics if row["crop"] == crop and row["comparison"] == comp]
            supported = [row for row in horizon_rows if row["n"] > 0]
            metrics.append({"crop": crop, "comparison": comp, "horizonsIncluded": len(supported),
                "aEqualHorizonMae": math.fsum(row["a"]["mae"] for row in supported) / len(supported) if supported else None,
                "bEqualHorizonMae": math.fsum(row["b"]["mae"] for row in supported) / len(supported) if supported else None,
                "aEqualHorizonMape": math.fsum(row["a"]["mape"] for row in supported) / len(supported) if supported else None,
                "bEqualHorizonMape": math.fsum(row["b"]["mape"] for row in supported) / len(supported) if supported else None})
    summaries = []
    for bandwidth in BANDWIDTHS:
        family = [row for row in dm if row["bandwidth"] == bandwidth]
        summaries.append({"bandwidth": bandwidth, "familySize": len(family),
            "significantImprovement": sum(row["status"] == "tested" and row.get("holmSignificant") and row.get("meanLossDifference", 0) < 0 for row in family),
            "significantWorsening": sum(row["status"] == "tested" and row.get("holmSignificant") and row.get("meanLossDifference", 0) > 0 for row in family),
            "inconclusive": sum(row["status"] == "tested" and not row.get("holmSignificant") for row in family),
            "inferenceUnavailable": sum(row["status"] != "tested" for row in family)})
    coverage_by_period = []
    periods = sorted({(row["target_date"][:7], "month") for row in rows if start <= _key(row)[1] <= end}
                     | {(f"{_key(row)[1].year:04d}-Q{(_key(row)[1].month-1)//3+1}", "quarter")
                        for row in rows if start <= _key(row)[1] <= end})
    for period, period_type in periods:
        for crop in PRODUCTS:
            for horizon in HORIZONS:
                planned_rows = [row for row in rows if start <= _key(row)[1] <= end and _key(row)[0] == crop
                                and _key(row)[2] == horizon and
                                ((period_type == "month" and _key(row)[1].strftime("%Y-%m") == period) or
                                 (period_type == "quarter" and f"{_key(row)[1].year:04d}-Q{(_key(row)[1].month-1)//3+1}" == period))]
                eligible_rows = [row for row in planned_rows if row.get("eligible") is True]
                common = {}
                for comp, a, b in COMPARISONS:
                    common[comp] = sum(actuals.get((crop, _key(row)[1])) is not None
                        and _forecast_prediction(row, a) is not None and _forecast_prediction(row, b) is not None
                        and (comp not in threeway_comparisons or all(_forecast_prediction(row, model) is not None
                            for model in ("fixed50", "blend106", "blend115"))) for row in eligible_rows)
                coverage_by_period.append({"periodType": period_type, "period": period, "crop": crop, "horizon": horizon,
                    "planned": len(planned_rows), "eligible": len(eligible_rows),
                    "positiveActual": sum(actuals.get((crop, _key(row)[1])) is not None for row in eligible_rows),
                    "fixed50Issued": sum(_forecast_prediction(row, "fixed50") is not None for row in eligible_rows),
                    "ridge106Issued": sum(_forecast_prediction(row, "blend106") is not None for row in eligible_rows),
                    "ridge115Issued": sum(_forecast_prediction(row, "blend115") is not None for row in eligible_rows),
                    "commonPairsByComparison": common})
    return {"coverage": coverage, "coverageByCell": coverage_by_cell, "coverageByPeriod": coverage_by_period,
            "metrics": metrics, "dm": dm, "dmSummaryByBandwidth": summaries}


def _joint_improvement(dm, crop, comparison):
    for horizon in HORIZONS:
        tests = {(item["loss"], item["bandwidth"]): item for item in dm
                 if item["crop"] == crop and item["horizon"] == horizon and item["comparison"] == comparison}
        if all((item := tests.get((loss, bandwidth))) is not None and item.get("holmSignificant")
               and item.get("meanLossDifference") is not None and item["meanLossDifference"] < 0
               for loss in LOSSES for bandwidth in BANDWIDTHS):
            return True
    return False


def _gate_screen(rows, scored, *, evaluation_candidates=None):
    gates = []
    for crop in PRODUCTS:
        candidate_results = {}
        any_lost = bool(scored["coverage"][crop]["eligibleFixed50KeysLostByEitherArm"])
        for candidate, metric_name, comparison in (("ridge106", "blend106", "ridge106_vs_fixed50"),
                                                    ("ridge115", "blend115", "ridge115_vs_fixed50")):
            macro = next(row for row in scored["metrics"] if row["crop"] == crop and row["comparison"] == comparison and "aEqualHorizonMape" in row)
            horizon_rows = [row for row in scored["metrics"] if row["crop"] == crop and row["comparison"] == comparison and "horizon" in row]
            no_harm = len(horizon_rows) == 10 and all(row["a"]["mape"] is not None and row["b"]["mape"] is not None and row["a"]["mape"] <= row["b"]["mape"] + 1.0 for row in horizon_rows)
            joint = _joint_improvement(scored["dm"], crop, comparison)
            all30 = len(horizon_rows) == 10 and all(row["a"]["n"] >= 30 for row in horizon_rows)
            gain = macro["aEqualHorizonMape"] is not None and macro["bEqualHorizonMape"] is not None and macro["aEqualHorizonMape"] <= macro["bEqualHorizonMape"] - .5
            mae = macro["aEqualHorizonMae"] is not None and macro["bEqualHorizonMae"] is not None and macro["aEqualHorizonMae"] < macro["bEqualHorizonMae"]
            lost = any_lost
            evaluation_candidate = (evaluation_candidates or {}).get(crop)
            if evaluation_candidates is not None:
                field = "blend106" if candidate == "ridge106" else "blend115"
                lost = any(_key(row)[0] == crop and row.get("eligible") is True
                    and _forecast_prediction(row, "fixed50") is not None
                    and _forecast_prediction(row, field) is None for row in rows)
            if evaluation_candidates is not None and candidate != evaluation_candidate:
                continue
            passed = all((all30, gain, mae, not lost, no_harm, joint))
            candidate_results[candidate] = {"passed": passed, "macro": macro}
            gates.append({"crop": crop, "candidate": candidate, "benchmark": "fixed50",
                "all10HorizonsAtLeast30CommonPairs": all30, "candidateEqualHorizonMape": macro["aEqualHorizonMape"],
                "fixed50EqualHorizonMape": macro["bEqualHorizonMape"],
                "mapeImprovementPctPoints": macro["bEqualHorizonMape"] - macro["aEqualHorizonMape"] if gain or macro["aEqualHorizonMape"] is not None and macro["bEqualHorizonMape"] is not None else None,
                "candidateEqualHorizonMae": macro["aEqualHorizonMae"], "fixed50EqualHorizonMae": macro["bEqualHorizonMae"],
                "coverageNondeclining": not lost, "anyCandidateLostFixed50IssuedKey": lost,
                "noHorizonMapeWorseByMoreThan1PctPoint": no_harm, "jointAeApeHolmImprovementAtSomeHorizonBothHac": joint,
                "decision": "eligible_for_selection" if passed else "retain_fixed50"})
        if evaluation_candidates is not None:
            selected = evaluation_candidates.get(crop, "retain_fixed50")
            passed_models = []
        else:
            passed_models = [name for name, result in candidate_results.items() if result["passed"]]
        passed_models.sort(key=lambda model: (candidate_results[model]["macro"]["aEqualHorizonMape"],
                                              candidate_results[model]["macro"]["aEqualHorizonMae"],
                                              MODELS.index(model)))
        if evaluation_candidates is None:
            selected = passed_models[0] if passed_models else "retain_fixed50"
        for row in gates:
            if row["crop"] == crop:
                row["selected"] = row["candidate"] == selected
    return gates


def _feature_effect(gates, scored):
    out = []
    for crop in PRODUCTS:
        macro = next(row for row in scored["metrics"] if row["crop"] == crop and row["comparison"] == "ridge115_vs_ridge106" and "aEqualHorizonMape" in row)
        joint = _joint_improvement(scored["dm"], crop, "ridge115_vs_ridge106")
        mean_better = macro["aEqualHorizonMape"] is not None and macro["bEqualHorizonMape"] is not None and macro["aEqualHorizonMape"] < macro["bEqualHorizonMape"] and macro["aEqualHorizonMae"] < macro["bEqualHorizonMae"]
        out.append({"crop": crop, "featureEffectClaim": "supported" if mean_better and joint else "unconfirmed",
                    "115MeanAeApeBothBetter": mean_better, "jointAeApeHolmImprovementAtSomeHorizonBothHac": joint,
                    "candidateEqualHorizonMape": macro["aEqualHorizonMape"], "baselineEqualHorizonMape": macro["bEqualHorizonMape"],
                    "candidateEqualHorizonMae": macro["aEqualHorizonMae"], "baselineEqualHorizonMae": macro["bEqualHorizonMae"]})
    return out


def score_screen(manifest: dict[str, Any], prices: dict[str, Any], components: dict[str, Any], *,
                 manifest_bytes: bytes, prices_bytes: bytes, components_bytes: bytes) -> dict[str, Any]:
    validate_manifest_bytes(manifest_bytes, prices_bytes, components_bytes, "screen")
    rows = manifest.get("forecasts")
    if not isinstance(rows, list):
        raise ValueError("screen forecast rows missing")
    actuals = _actuals(prices, SCREEN_START, SCREEN_END)
    scored = _score_common(rows, actuals, SCREEN_START, SCREEN_END, require_threeway=True)
    gates = _gate_screen(rows, scored)
    selected = {}
    for crop in PRODUCTS:
        candidates = [row for row in gates if row["crop"] == crop and row["decision"] == "eligible_for_selection"]
        candidates.sort(key=lambda row: (row["candidateEqualHorizonMape"], row["candidateEqualHorizonMae"], MODELS.index(row["candidate"])))
        selected[crop] = candidates[0]["candidate"] if candidates else "retain_fixed50"
    return {"phase": "screen_scored", "manifestSha256": _sha(manifest_bytes),
            "pricesSha256": _sha(prices_bytes), "componentsSha256": _sha(components_bytes),
            "scorerSha256": _sha(Path(__file__).read_bytes()),
            **scored, "candidateGates": gates, "featureEffect": _feature_effect(gates, scored), "selection": selected}


def _decision(manifest_bytes, report_bytes, selection):
    unsigned = {"schemaVersion": 1, "phase": "selection_decision",
                "screenManifestSha256": _sha(manifest_bytes), "screenReportSha256": _sha(report_bytes),
                "scorerSha256": _sha(Path(__file__).read_bytes()), "selections": selection}
    unsigned["decisionSha256"] = _sha((json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
    return unsigned


def validate_screen_artifacts(screen_manifest_bytes: bytes, prices_bytes: bytes, screen_components_bytes: bytes,
                              screen_report_bytes: bytes, decision_bytes: bytes) -> dict[str, Any]:
    manifest = json.loads(screen_manifest_bytes)
    report = json.loads(screen_report_bytes)
    decision = json.loads(decision_bytes)
    components = json.loads(screen_components_bytes)
    prices = json.loads(prices_bytes)
    recomputed = score_screen(manifest, prices, components, manifest_bytes=screen_manifest_bytes,
                              prices_bytes=prices_bytes, components_bytes=screen_components_bytes)
    if _pretty(recomputed) != screen_report_bytes:
        raise ValueError("screen report differs from complete replay and rescore")
    expected_decision = _decision(screen_manifest_bytes, screen_report_bytes, recomputed["selection"])
    if expected_decision != decision:
        raise ValueError("screen decision differs from replayed selection")
    return {"selection": recomputed["selection"], "manifestSha256": _sha(screen_manifest_bytes),
            "screenReportSha256": _sha(screen_report_bytes), "decisionSha256": decision["decisionSha256"]}


def _policy_rows(rows, actuals, selections):
    output = []
    for row in rows:
        crop, day, horizon = _key(row)
        fixed = _forecast_prediction(row, "fixed50")
        if row.get("fixed50KeyPresent") is not True or fixed is None:
            continue
        actual = actuals.get((crop, day))
        candidate = selections.get(crop, "retain_fixed50")
        candidate_field = "blend106" if candidate == "ridge106" else "blend115" if candidate == "ridge115" else None
        candidate_prediction = _forecast_prediction(row, candidate_field) if candidate_field else None
        eligible = row.get("eligible") is True
        preperiod = not eligible
        fallback = candidate_field is None or candidate_prediction is None
        output.append({"crop": crop, "target_date": day.isoformat(), "horizon": horizon,
            "origin_date": row["origin_date"], "actual": actual, "screenSelection": candidate,
            "evaluationGateDecision": None, "candidate": candidate_prediction, "fixed50": fixed,
            "policy": fixed if fallback else candidate_prediction,
            "fallback": fallback, "preperiodOriginFallback": preperiod,
            "candidateMissingReason": row[candidate_field]["reason"] if candidate_field and candidate_prediction is None else None})
    return output


def score_evaluation(manifest: dict[str, Any], prices: dict[str, Any], components: dict[str, Any], *,
                     manifest_bytes: bytes, prices_bytes: bytes, components_bytes: bytes,
                     screen_manifest_bytes: bytes, screen_report_bytes: bytes, screen_decision_bytes: bytes,
                     screen_components_bytes: bytes) -> dict[str, Any]:
    binding = validate_screen_artifacts(screen_manifest_bytes, prices_bytes, screen_components_bytes,
                                        screen_report_bytes, screen_decision_bytes)
    validate_manifest_bytes(manifest_bytes, prices_bytes, components_bytes, "evaluation",
        decision_raw=screen_decision_bytes, screen_manifest_raw=screen_manifest_bytes, screen_report_raw=screen_report_bytes)
    selection = binding["selection"]
    actuals = _actuals(prices, EVAL_START, EVAL_END)
    rows = manifest["forecasts"]
    active = {crop: set() for crop in PRODUCTS}
    for crop, candidate in selection.items():
        if candidate == "ridge106":
            active[crop].add("ridge106_vs_fixed50")
        elif candidate == "ridge115":
            active[crop].update(("ridge115_vs_fixed50", "ridge115_vs_ridge106"))
    threeway = {"ridge115_vs_fixed50", "ridge115_vs_ridge106"}
    eval_threeway = {comp for crop, candidate in selection.items() if candidate == "ridge115" for comp in threeway}
    scored = _score_common(rows, actuals, EVAL_START, EVAL_END, active_comparisons=active,
                           require_threeway=eval_threeway)
    gates = []
    for crop, candidate in selection.items():
        comp = "ridge106_vs_fixed50" if candidate == "ridge106" else "ridge115_vs_fixed50" if candidate == "ridge115" else None
        if comp is None:
            gates.append({"crop": crop, "candidate": candidate, "decision": "retain_fixed50", "reason": "not_selected_by_frozen_screen"})
            continue
        selected_gates = _gate_screen(rows, scored, evaluation_candidates=selection)
        gate = next(row for row in selected_gates if row["crop"] == crop and row["candidate"] == candidate)
        gate = {**gate, "decision": "selected_candidate_passes_2026_gate" if gate["decision"] == "eligible_for_selection" else "selected_candidate_fails_2026_gate_policy_retained"}
        if candidate == "ridge115":
            effect = next(row for row in _feature_effect([], scored) if row["crop"] == crop)
            gate["featureEffect"] = effect
        gates.append(gate)
    policy = _policy_rows(rows, actuals, selection)
    for row in policy:
        row["evaluationGateDecision"] = next((gate["decision"] for gate in gates if gate["crop"] == row["crop"]), "retain_fixed50")
    portfolio = []
    for crop in PRODUCTS:
        for horizon in HORIZONS:
            subset = [row for row in policy if row["crop"] == crop and row["horizon"] == horizon]
            policy_metric = _metrics([(row["actual"], row["policy"]) for row in subset if row["actual"] is not None])
            base_metric = _metrics([(row["actual"], row["fixed50"]) for row in subset if row["actual"] is not None])
            portfolio.append({"crop": crop, "horizon": horizon, "n": policy_metric["n"],
                              "policy": policy_metric, "fixed50": base_metric,
                              "fallbackCount": sum(row["fallback"] for row in subset),
                              "preperiodOriginFallbackCount": sum(row["preperiodOriginFallback"] for row in subset),
                              "eligibleCandidateFailureCount": sum(row["fallback"] and not row["preperiodOriginFallback"] and row["screenSelection"] != "retain_fixed50" for row in subset)})
    supported_cells = [row for row in portfolio if row["policy"]["n"] > 0 and row["fixed50"]["n"] > 0]
    portfolio_summary = {"supportedCells": len(supported_cells), "plannedCells": len(PRODUCTS) * len(HORIZONS),
        "equalCellPolicyMae": math.fsum(row["policy"]["mae"] for row in supported_cells) / len(supported_cells) if supported_cells else None,
        "equalCellFixed50Mae": math.fsum(row["fixed50"]["mae"] for row in supported_cells) / len(supported_cells) if supported_cells else None,
        "equalCellPolicyMape": math.fsum(row["policy"]["mape"] for row in supported_cells) / len(supported_cells) if supported_cells else None,
        "equalCellFixed50Mape": math.fsum(row["fixed50"]["mape"] for row in supported_cells) / len(supported_cells) if supported_cells else None,
        "allCellsSupported": len(supported_cells) == len(PRODUCTS) * len(HORIZONS),
        "interpretation": "descriptive_only_no_added_inference_family"}
    report = {"phase": "evaluation_scored", "manifestSha256": _sha(manifest_bytes),
        "screenManifestSha256": binding["manifestSha256"], "screenReportSha256": binding["screenReportSha256"],
        "screenDecisionSha256": binding["decisionSha256"], "scorerSha256": _sha(Path(__file__).read_bytes()),
        **scored, "evaluationGates": gates,
        "policyCoverage": {"sourceFixed50Keys": len(policy),
            "preperiodOriginFallbackKeys": sum(row["preperiodOriginFallback"] for row in policy),
            "eligibleCandidateFailureKeys": sum(row["fallback"] and not row["preperiodOriginFallback"] and row["screenSelection"] != "retain_fixed50" for row in policy),
            "policyIssued": sum(_positive(row["policy"]) is not None for row in policy)},
        "policyMetrics": portfolio, "policyPortfolioEqualWeightAcross50Cells": portfolio_summary,
        "policyRows": policy}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("phase", choices=("screen", "evaluation"))
    build_parser.add_argument("output", type=Path)
    build_parser.add_argument("--prices", type=Path, default=PRICES_PATH)
    build_parser.add_argument("--components", type=Path)
    build_parser.add_argument("--decision", type=Path)
    build_parser.add_argument("--screen-manifest", type=Path)
    build_parser.add_argument("--screen-report", type=Path)
    score_screen_parser = sub.add_parser("score-screen")
    score_screen_parser.add_argument("manifest", type=Path); score_screen_parser.add_argument("report", type=Path); score_screen_parser.add_argument("decision", type=Path)
    score_screen_parser.add_argument("--prices", type=Path, default=PRICES_PATH); score_screen_parser.add_argument("--components", type=Path, default=SCREEN_COMPONENTS_PATH)
    score_eval = sub.add_parser("score-evaluation")
    for name in ("manifest", "screen_manifest", "screen_report", "screen_decision", "report"):
        score_eval.add_argument(name, type=Path)
    score_eval.add_argument("--prices", type=Path, default=PRICES_PATH); score_eval.add_argument("--components", type=Path, default=EVALUATION_COMPONENTS_PATH); score_eval.add_argument("--screen-components", type=Path, default=SCREEN_COMPONENTS_PATH)
    args = parser.parse_args(argv)
    if args.command == "build":
        components_path = args.components or (SCREEN_COMPONENTS_PATH if args.phase == "screen" else EVALUATION_COMPONENTS_PATH)
        doc = build(args.prices.read_bytes(), components_path.read_bytes(), args.phase,
                    decision_raw=args.decision.read_bytes() if args.decision else None,
                    screen_manifest_raw=args.screen_manifest.read_bytes() if args.screen_manifest else None,
                    screen_report_raw=args.screen_report.read_bytes() if args.screen_report else None,
                    progress=lambda event: print(json.dumps(event, sort_keys=True), file=sys.stderr))
        _write_exclusive(args.output, doc)
    elif args.command == "score-screen":
        mraw, praw, craw = args.manifest.read_bytes(), args.prices.read_bytes(), args.components.read_bytes()
        report = score_screen(json.loads(mraw), json.loads(praw), json.loads(craw), manifest_bytes=mraw, prices_bytes=praw, components_bytes=craw)
        rraw = _pretty(report)
        decision = _decision(mraw, rraw, report["selection"])
        _write_exclusive(args.report, report); _write_exclusive(args.decision, decision)
    else:
        mraw, praw, craw = args.manifest.read_bytes(), args.prices.read_bytes(), args.components.read_bytes()
        smraw, srraw, sd_raw, sc_raw = args.screen_manifest.read_bytes(), args.screen_report.read_bytes(), args.screen_decision.read_bytes(), args.screen_components.read_bytes()
        report = score_evaluation(json.loads(mraw), json.loads(praw), json.loads(craw), manifest_bytes=mraw, prices_bytes=praw, components_bytes=craw,
            screen_manifest_bytes=smraw, screen_report_bytes=srraw, screen_decision_bytes=sd_raw, screen_components_bytes=sc_raw)
        _write_exclusive(args.report, report)


if __name__ == "__main__":
    main()
