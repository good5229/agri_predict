"""Monthly causal Ridge forecasts preserving original 40 features.

This answer-free builder compares the exact original direct-horizon 40-feature
schema with an additive 66-feature block derived from the archived 71-feature
library. It does not read evaluation outcomes, use an API, tune parameters, or
refit SARIMA; it inherits the frozen causal SARIMA forecasts from the pinned
equal-information manifest.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import direct_horizon, equal_info_online_forecasts as online
from . import full_history_feature_library as library
from . import full_history_return_eval as narrow
from . import full_history_xgb_rf_eval as rich
from . import modeling

ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
BASELINE_PATH = ROOT / "outputs/experiments/equal-info-online-20261010/forecast-manifest.json"
OUT_DIR = ROOT / "outputs/experiments/preserved-feature-20261010"
CONFIRMATION_OUT = OUT_DIR / "confirmation-manifest.json"
CONFIRMATION_REPORT = OUT_DIR / "confirmation-report.json"
EVALUATION_OUT = OUT_DIR / "evaluation-manifest.json"
CONFIRMATION_DECISION = OUT_DIR / "confirmation-decision.json"

PRICES_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
BASELINE_SHA256 = "d7d74823754c2a48d33cc4ddaeab674c686601fb927d399d84c5dbcd235184df"
ONLINE_MODULE_SHA256 = "84e68a9a3eb362e59a336440a0bf9254eacb68dfb881e6cf3abdeeba4a215471"
LIBRARY_SHA256 = "58bcde98d11239494ae13cc6169fb4c7f84a49a15ad11cf5b3cf5c49b4f39389"
RICH_HELPER_SHA256 = "1951b34bc019ccd9fe680776f4277105e1edb470d3d7d5a37802c781088d9245"
MODELING_SHA256 = "0ef149c231cc19f5990548822c10afecb16957863eddbd436d7d61a06c9963fc"
DIRECT_HELPER_SHA256 = "1cf3f8ce5455d9ce1f8d4eeb6e14f84e465acf4f5e6d512670b102115fb96d40"
NARROW_HELPER_SHA256 = online.NARROW_HELPER_SHA256

PRODUCTS = online.PRODUCTS
HORIZONS = online.HORIZONS
MIN_TRAIN = 90
RIDGE_ALPHA = 10.0
PHASES = {
    "confirmation": (online.CONF_START, online.CONF_END),
    "evaluation": (online.EVAL_START, online.EVAL_END),
}
OMITTED_LIBRARY_FEATURES = frozenset({
    "p0_age_days", "target_weekday_sin", "target_weekday_cos",
    "target_annual_sin", "target_annual_cos",
})


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unsigned_sha(value: dict[str, Any], field: str) -> str:
    unsigned = dict(value)
    unsigned.pop(field, None)
    return hashlib.sha256(_canonical(unsigned).rstrip(b"\n")).hexdigest()


def _date(value: Any) -> dt.date | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _expected_keys(phase: str) -> list[tuple[str, dt.date, int]]:
    start, end = PHASES[phase]
    return [(p, start + dt.timedelta(days=i), h)
            for p in PRODUCTS for i in range((end - start).days + 1) for h in HORIZONS]


def _trim_price_snapshot(raw: dict[str, Any], phase_end: dt.date) -> tuple[dict[str, Any], dict[str, int]]:
    """Exclude later and malformed-date answer rows before value parsing."""
    if raw.get("schemaVersion") != 1 or not isinstance(raw.get("products"), list):
        raise ValueError("prices must use dashboard schemaVersion 1")
    trimmed = {k: v for k, v in raw.items() if k != "products"}
    trimmed_products = []
    audit = {}
    for product in raw["products"]:
        if not isinstance(product, dict) or product.get("id") not in PRODUCTS:
            raise ValueError("unknown or malformed product in price snapshot")
        observations = product.get("observations")
        if not isinstance(observations, list):
            raise ValueError("price product observations must be a list")
        kept = []
        skipped_later = skipped_invalid_date = 0
        for row in observations:
            if not isinstance(row, dict):
                skipped_invalid_date += 1
                continue
            day = _date(row.get("date"))
            if day is None:
                skipped_invalid_date += 1
            elif day > phase_end:
                skipped_later += 1
            else:
                kept.append(row)
        trimmed_products.append({**product, "observations": kept})
        audit[product["id"]] = {"includedThroughPhaseEnd": len(kept), "skippedLaterDates": skipped_later,
                                "skippedInvalidDateRows": skipped_invalid_date}
    if {p["id"] for p in trimmed_products} != set(PRODUCTS):
        raise ValueError("prices must contain every product")
    trimmed["products"] = trimmed_products
    return trimmed, audit


def _check_hash(path: Path, expected: str, label: str) -> None:
    if _sha(path) != expected:
        raise ValueError(f"pinned {label} SHA-256 mismatch")


def _check_inputs() -> None:
    _check_hash(PRICES_PATH, PRICES_SHA256, "prices")
    _check_hash(BASELINE_PATH, BASELINE_SHA256, "equal-info baseline manifest")
    helper_hashes = {
        "equal-info forecast builder": (Path(online.__file__), ONLINE_MODULE_SHA256),
        "71-feature library": (Path(library.__file__), LIBRARY_SHA256),
        "rich feature helper": (Path(rich.__file__), RICH_HELPER_SHA256),
        "original feature helper": (Path(modeling.__file__), MODELING_SHA256),
        "direct horizon helper": (Path(direct_horizon.__file__), DIRECT_HELPER_SHA256),
        "narrow feature helper": (Path(narrow.__file__), NARROW_HELPER_SHA256),
    }
    for label, (path, expected) in helper_hashes.items():
        _check_hash(path, expected, label)


def _validate_baseline(raw: dict[str, Any]) -> None:
    online._validate(raw)
    if raw.get("inputSha256", {}).get("prices") != PRICES_SHA256:
        raise ValueError("equal-info baseline price lineage mismatch")
    if raw.get("inputSha256", {}).get("module") != ONLINE_MODULE_SHA256:
        raise ValueError("equal-info baseline source lineage mismatch")


def _baseline_index(raw: dict[str, Any], phase: str) -> dict[tuple[str, dt.date, int], dict[str, Any]]:
    key = "confirmationForecasts" if phase == "confirmation" else "evaluationForecasts"
    index: dict[tuple[str, dt.date, int], dict[str, Any]] = {}
    for row in raw[key]:
        identity = (row["product"], _date(row["target"]), row["horizon"])
        if identity[1] is None or identity in index:
            raise ValueError("invalid or duplicate equal-info forecast key")
        index[identity] = row
    if set(index) != set(_expected_keys(phase)):
        raise ValueError("equal-info baseline phase grid mismatch")
    return index


def _decision_fields(decision: dict[str, Any], confirmation_path: Path) -> tuple[list[str], dict[str, str]]:
    required = {
        "schemaVersion", "kind", "confirmationForecastManifestSha256",
        "confirmationInternalManifestSha256", "confirmationEvaluationSha256",
        "scorerSha256", "selectedCrops", "cropDecisions", "decisionSha256",
    }
    if not isinstance(decision, dict) or set(decision) != required:
        raise ValueError("confirmation decision schema mismatch")
    if decision["schemaVersion"] != 1 or decision["kind"] != "preserved_feature_confirmation_gate":
        raise ValueError("invalid confirmation decision type")
    if _unsigned_sha(decision, "decisionSha256") != decision.get("decisionSha256"):
        raise ValueError("confirmation decision digest mismatch")
    if not confirmation_path.is_file() or _sha(confirmation_path) != decision["confirmationForecastManifestSha256"]:
        raise ValueError("confirmation manifest file hash does not match signed decision")
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    _validate_manifest(confirmation, "confirmation")
    if confirmation.get("manifestSha256") != decision["confirmationInternalManifestSha256"]:
        raise ValueError("confirmation manifest internal digest does not match decision")
    for field in ("confirmationEvaluationSha256", "scorerSha256"):
        value = decision.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"invalid confirmation decision {field}")
    selected = decision.get("selectedCrops")
    crops = decision.get("cropDecisions")
    if (not isinstance(selected, list) or len(selected) != len(set(selected))
            or any(p not in PRODUCTS for p in selected)
            or not isinstance(crops, list) or len(crops) != len(PRODUCTS)):
        raise ValueError("invalid confirmation crop decisions")
    by_product = {x.get("product"): x for x in crops if isinstance(x, dict)}
    if set(by_product) != set(PRODUCTS) or [p for p in PRODUCTS if by_product[p].get("eligible") is True] != selected:
        raise ValueError("selected crop list disagrees with per-crop eligibility decisions")
    hashes = {
        "confirmationForecastManifest": decision["confirmationForecastManifestSha256"],
        "confirmationInternalManifest": decision["confirmationInternalManifestSha256"],
        "confirmationEvaluation": decision["confirmationEvaluationSha256"],
        "confirmationDecision": decision["decisionSha256"],
        "confirmationScorer": decision["scorerSha256"],
    }
    return selected, hashes


def _feature_pair(
    product: str,
    target: dt.date,
    horizon: int,
    direct_series: dict[str, dict[dt.date, tuple[float, float]]],
    records: dict[str, dict[dt.date, dict[str, Any]]],
    caches: dict[str, dict[Any, Any]],
    ordered: dict[str, list[dt.date]],
    price_maps: dict[str, dict[dt.date, float]],
) -> dict[str, Any] | None:
    series = direct_series[product]
    original = modeling._feature_row(series, target - dt.timedelta(days=horizon + 1), target, horizon, caches[product])
    if original is None:
        return None
    source_day, original_vector = original
    origin = target - dt.timedelta(days=horizon)
    if (origin - source_day).days > 14:
        return None
    expanded, p0, audit = library.features(records[product], target, horizon, ordered[product], price_maps[product])
    if expanded is None or p0 is None:
        raise ValueError(f"expanded feature row missing where original row exists: {product}/{target}/h{horizon}")
    expanded_source = _date(audit.get("p0Date"))
    if expanded_source != source_day or not math.isclose(p0, series[source_day][0], rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"p0 mismatch between original and expanded feature rows: {product}/{target}/h{horizon}")
    appended = [value for name, value in zip(library.FEATURE_NAMES, expanded) if name not in OMITTED_LIBRARY_FEATURES]
    if len(original_vector) != len(modeling.FEATURE_NAMES) or len(original_vector) != 40 or len(appended) != 66:
        raise ValueError("expected exact 40-feature prefix plus 66-feature additive block")
    vector = list(original_vector) + appended
    if not all(math.isfinite(float(value)) for value in vector):
        raise ValueError("nonfinite preserved feature row")
    return {"original40": list(original_vector), "preserved106": vector,
            "p0": series[source_day][0], "p0Date": source_day,
            "p0AgeDays": (origin - source_day).days, "featureAudit": audit}


def _phase_grid_index(baseline: dict[tuple[str, dt.date, int], dict[str, Any]], phase: str) -> dict[tuple[str, dt.date, int], dict[str, Any]]:
    start, _ = PHASES[phase]
    selected: dict[tuple[str, dt.date, int], dict[str, Any]] = {}
    for key, row in baseline.items():
        origin = key[1] - dt.timedelta(days=key[2])
        if origin >= start:
            selected[key] = row
    return selected


def _group_monthly_keys(keys: list[tuple[str, dt.date, int]], start: dt.date, phase: str,
                        selected: set[str], baseline_rows: dict[tuple[str, dt.date, int], dict[str, Any]],
                        forecasts: dict[tuple[str, dt.date, int], dict[str, Any]]) -> dict[tuple[int, int], list[tuple[str, dt.date, int]]]:
    monthly: dict[tuple[int, int], list[tuple[str, dt.date, int]]] = {}
    for key in keys:
        origin = key[1] - dt.timedelta(days=key[2])
        if origin >= start:
            monthly.setdefault((origin.year, origin.month), []).append(key)
        else:
            forecasts[key] = _unissued_grid_row(key, phase, selected,
                                                "origin_before_phase_start", baseline_rows[key])
    return monthly


def _fit_audit(targets: list[dt.date], fit_origin: dt.date) -> dict[str, Any]:
    rows = sorted(t for t in targets if t < fit_origin)
    return {"count": len(rows), "firstTarget": rows[0].isoformat() if rows else None,
            "lastTarget": rows[-1].isoformat() if rows else None,
            "fitOrigin": fit_origin.isoformat(), "minimumTrainingTargets": MIN_TRAIN}


def _assert_training_support(actual: dict[str, Any], expected: dict[str, Any], product: str,
                             horizon: int, month: tuple[int, int]) -> None:
    for field in ("count", "firstTarget", "lastTarget", "fitOrigin"):
        if actual.get(field) != expected.get(field):
            raise ValueError(f"direct/candidate training support mismatch in {field} for {product}/h{horizon}/{month}: {actual.get(field)} vs {expected.get(field)}")


def _issued(prediction: float, training: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    if not math.isfinite(prediction) or prediction <= 0:
        raise ValueError("issued prediction must be finite and positive")
    return {"status": "issued", "reason": None, "prediction": prediction,
            "training": training, "provenance": provenance}


def _predict_row(fit: Any, vector: list[float], p0: float, training: dict[str, Any],
                 provenance: dict[str, Any]) -> dict[str, Any]:
    try:
        prediction = p0 * math.exp(float(fit.predict([vector])[0]))
        return _issued(prediction, training, provenance)
    except Exception as exc:
        return _not_issued(f"prediction_failed:{type(exc).__name__}", training, provenance)


def _not_issued(reason: str, training: dict[str, Any] | None = None, provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"status": "not_issued", "reason": reason, "prediction": None,
            "training": training, "provenance": provenance or {}}


def _not_selected() -> dict[str, Any]:
    return {"status": "not_selected", "reason": "crop_not_selected_by_confirmation_gate",
            "prediction": None, "training": None, "provenance": {}}


def _policy_forecast(selected: bool, candidate_blend: dict[str, Any], fixed50: dict[str, Any]) -> dict[str, Any]:
    if not selected:
        return {"method": "fixed50", "status": fixed50["status"], "prediction": fixed50["prediction"],
                "reason": "crop_not_selected_by_confirmation_gate", "provenance": {"fallback": True}}
    if candidate_blend.get("status") == "issued":
        return {"method": "preserved106Blend", "status": candidate_blend["status"],
                "prediction": candidate_blend["prediction"], "reason": None,
                "provenance": copy.deepcopy(candidate_blend.get("provenance", {}))}
    return {"method": "fixed50", "status": fixed50["status"], "prediction": fixed50["prediction"],
            "reason": "selected_candidate_unavailable_fallback", "provenance": {"fallback": True}}


def _assert_prediction_matches(actual: dict[str, Any], expected: dict[str, Any], label: str, key: tuple[str, dt.date, int]) -> None:
    if actual.get("status") != expected.get("status"):
        raise ValueError(f"{label} status differs from frozen equal-info manifest at {key}")
    if actual.get("status") == "issued":
        left, right = _positive(actual.get("prediction")), _positive(expected.get("prediction"))
        if left is None or right is None or not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError(f"{label} prediction differs from frozen equal-info manifest at {key}")


def build(
    prices: dict[str, Any],
    baseline_manifest: dict[str, Any],
    phase: str,
    *,
    selected_crops: list[str] | None = None,
    decision_hashes: dict[str, str] | None = None,
    input_hashes: dict[str, str] | None = None,
    progress: Any = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError("phase must be confirmation or evaluation")
    if phase == "confirmation" and selected_crops is not None:
        raise ValueError("confirmation phase must not preselect crops")
    if phase == "evaluation" and (selected_crops is None or decision_hashes is None):
        raise ValueError("evaluation phase requires a verified confirmation decision")
    selected = set(PRODUCTS if phase == "confirmation" else selected_crops or [])
    if phase == "evaluation" and not selected:
        raise ValueError("no crop passed confirmation; evaluation phase must not fit or score")
    if any(p not in PRODUCTS for p in selected):
        raise ValueError("unknown selected crop")

    start, end = PHASES[phase]
    baseline_rows = _baseline_index(baseline_manifest, phase)
    phase_prices, phase_row_audit = _trim_price_snapshot(prices, end)
    direct_series = direct_horizon._observations(phase_prices)
    raw_maps, source_audit = rich._raw_maps(phase_prices)
    # All model fitting and feature generation share exactly the observed
    # positive-price/positive-volume dates used by direct_horizon.
    records: dict[str, dict[dt.date, dict[str, Any]]] = {}
    price_maps: dict[str, dict[dt.date, float]] = {}
    ordered: dict[str, list[dt.date]] = {}
    for product in PRODUCTS:
        domain = direct_series[product]
        if not set(domain).issubset(raw_maps[product]):
            raise ValueError(f"direct observation is absent from rich source map for {product}")
        records[product] = {d: dict(raw_maps[product][d]) for d in domain}
        price_maps[product] = {d: value[0] for d, value in domain.items()}
        ordered[product] = sorted(domain)
        for d, (price, volume) in domain.items():
            rec = records[product][d]
            if not math.isclose(rec["price"], price, rel_tol=0.0, abs_tol=1e-12) or rec["volume"] is None or not math.isclose(rec["volume"], volume, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"price/volume anchor mismatch for {product}/{d}")

    phase_keys = _expected_keys(phase)
    by_crop_horizon: dict[tuple[str, int], list[tuple[str, dt.date, int]]] = {}
    for key in phase_keys:
        by_crop_horizon.setdefault((key[0], key[2]), []).append(key)
    forecasts: dict[tuple[str, dt.date, int], dict[str, Any]] = {}

    for product in PRODUCTS:
        if phase == "evaluation" and product not in selected:
            for key in phase_keys:
                if key[0] == product:
                    forecasts[key] = _unselected_policy_row(key, baseline_rows[key])
            continue
        model_rows = {key: baseline_rows[key] for key in phase_keys if key[0] == product}
        for horizon in HORIZONS:
            keys = sorted(by_crop_horizon[(product, horizon)], key=lambda k:k[1])
            monthly = _group_monthly_keys(keys, start, phase, selected, baseline_rows, forecasts)
            if not monthly:
                continue

            # Build all chronological training rows and all target rows once
            # per crop/horizon; every monthly fit then selects only target < fitOrigin.
            feature_cache = {}
            rows_by_target: dict[dt.date, dict[str, Any] | None] = {}
            all_targets = sorted(set(direct_series[product]) | {key[1] for key in keys})
            for target in all_targets:
                if target < online.narrow.TRAIN_START and target not in {k[1] for k in keys}:
                    continue
                rows_by_target[target] = _feature_pair(product, target, horizon, direct_series, records,
                                                       {product: feature_cache}, ordered, price_maps)
            training_targets = [target for target in direct_series[product]
                                if target >= online.narrow.TRAIN_START and rows_by_target.get(target) is not None]
            target_cache = {key[1]: rows_by_target.get(key[1]) for key in keys}

            for month in sorted(monthly):
                month_keys = monthly[month]
                fit_origin = min(k[1] - dt.timedelta(days=horizon) for k in month_keys)
                training = [target for target in training_targets if target < fit_origin]
                audit = _fit_audit(training, fit_origin)
                expected_audit = None
                # Frozen reference audit is the decisive same-target training-row check.
                for key in month_keys:
                    source_audit_row = baseline_rows[key].get("models", {}).get("causalDirect", {}).get("training")
                    if isinstance(source_audit_row, dict):
                        expected_audit = source_audit_row
                        break
                if expected_audit is None:
                    raise ValueError(f"missing frozen causalDirect training audit for {product}/h{horizon}/{month}")
                _assert_training_support(audit, expected_audit, product, horizon, month)

                direct_fit = candidate_fit = None
                direct_error = candidate_error = None
                if len(training) < MIN_TRAIN:
                    direct_error = candidate_error = "fewer_than_90_valid_training_targets"
                else:
                    x40 = [rows_by_target[t]["original40"] for t in training]
                    x106 = [rows_by_target[t]["preserved106"] for t in training]
                    y = [math.log(direct_series[product][t][0] / rows_by_target[t]["p0"]) for t in training]
                    try:
                        direct_fit = online._fit_ridge(x40, y, RIDGE_ALPHA)
                    except Exception as exc:
                        direct_error = f"ridge_fit_failed:{type(exc).__name__}"
                    if product in selected:
                        try:
                            candidate_fit = online._fit_ridge(x106, y, RIDGE_ALPHA)
                        except Exception as exc:
                            candidate_error = f"ridge_fit_failed:{type(exc).__name__}"
                if progress:
                    progress(phase, product, horizon, month, len(training))

                for key in month_keys:
                    _, target, _ = key
                    origin = target - dt.timedelta(days=horizon)
                    base_row = baseline_rows[key]
                    base_models = base_row["models"]
                    cached = target_cache[target]
                    direct_model_row: dict[str, Any]
                    candidate_model_row: dict[str, Any]
                    if direct_error or direct_fit is None:
                        direct_model_row = _not_issued(direct_error or "baseline_fit_unavailable", audit)
                    elif cached is None:
                        direct_model_row = _not_issued("feature_unavailable_or_source_too_old", audit)
                    else:
                        direct_model_row = _predict_row(direct_fit, cached["original40"], cached["p0"], audit,
                            {"p0Date": cached["p0Date"].isoformat(), "sourceAgeDays": cached["p0AgeDays"]})
                    _assert_prediction_matches(direct_model_row, base_models["causalDirect"], "causalDirect", key)

                    if product not in selected:
                        candidate_model_row = _not_selected()
                    elif candidate_error or candidate_fit is None:
                        candidate_model_row = _not_issued(candidate_error or "candidate_fit_unavailable", audit)
                    elif cached is None:
                        candidate_model_row = _not_issued("feature_unavailable_or_source_too_old", audit)
                    else:
                        candidate_model_row = _predict_row(candidate_fit, cached["preserved106"], cached["p0"], audit,
                            {"p0Date": cached["p0Date"].isoformat(), "sourceAgeDays": cached["p0AgeDays"],
                             "featureCount": 106, "featureAudit": cached["featureAudit"]})

                    sarima_row = copy.deepcopy(base_models["causalSarima"])
                    causal50 = online._blend_ridge_sarima(direct_model_row, sarima_row)
                    _assert_prediction_matches(causal50, base_models["causal50"], "causal50", key)
                    candidate_blend = _not_selected() if product not in selected else online._blend_ridge_sarima(candidate_model_row, sarima_row)
                    fixed_row = copy.deepcopy(base_models["fixed50"])
                    policy = _policy_forecast(product in selected, candidate_blend, fixed_row)
                    forecasts[key] = {
                        "product": product, "target": target.isoformat(), "origin": origin.isoformat(), "horizon": horizon,
                        "eligible": origin >= start, "selectedCrop": product in selected if phase == "evaluation" else None,
                        "models": {"causalDirect": direct_model_row, "causalSarima": sarima_row,
                                   "causal50": causal50, "fixed50": fixed_row,
                                   "preserved106": candidate_model_row, "preserved106Blend": candidate_blend},
                        "policyForecast": policy,
                    }

    ordered_forecasts = [forecasts[k] for k in phase_keys]
    doc = {
        "schemaVersion": 1,
        "phase": "forecast_only",
        "phaseName": phase,
        "targetValuesAttached": False,
        "protocol": {
            "id": "preserved-feature-20261010-v1", "phase": phase,
            "products": list(PRODUCTS), "horizons": list(HORIZONS),
            "targetDates": [start.isoformat(), end.isoformat()],
            "monthlyRefit": "first eligible origin in each calendar month; only target < fitOrigin in training",
            "sameTrainingSupport": "direct_horizon._observations positive price-and-volume target dates; exact count/first/last/fitOrigin replayed against frozen causalDirect audit",
            "featureCutoff": "origin-1 for both training and prediction features; p0 must match source date",
            "minimumTrainingTargets": MIN_TRAIN, "ridgeAlpha": RIDGE_ALPHA,
            "featureNames": list(modeling.FEATURE_NAMES) + [n for n in library.FEATURE_NAMES if n not in OMITTED_LIBRARY_FEATURES],
            "featureSchema": {"originalPrefixCount": 40, "appendedCount": 66, "totalCount": 106,
                              "omittedFrom71": sorted(OMITTED_LIBRARY_FEATURES)},
            "blend": "0.5*preserved106 Ridge + 0.5*identical frozen causalSarima; same component fallback as causal50",
            "evaluationSelection": selected_crops if phase == "evaluation" else None,
            "portfolioPolicy": "selected crops use preserved106Blend when issued and otherwise fall back to frozen fixed50; unselected crops retain frozen fixed50. Raw candidate status remains separate so fallback cannot mask coverage loss.",
            "confirmationDecisionHashes": decision_hashes or {},
            "causalSarima": "inherited without refit from frozen equal-info-online manifest",
            "fixed50": "inherited as descriptive same-day-information benchmark and whole-five-crop fallback for confirmation-unselected products",
        },
        "inputSha256": {**(input_hashes or {}),
                        "prices": PRICES_SHA256, "equalInfoBaseline": BASELINE_SHA256,
                        "equalInfoBuilder": ONLINE_MODULE_SHA256, "fullHistoryFeatureLibrary": LIBRARY_SHA256,
                        "richFeatureHelper": RICH_HELPER_SHA256, "modelingHelper": MODELING_SHA256,
                        "directHorizonHelper": DIRECT_HELPER_SHA256,
                        "narrowFeatureHelper": NARROW_HELPER_SHA256,
                        "preservedFeatureGenerator": _sha(Path(__file__))},
        "sourceAudit": {"rawPriceRecordsThroughPhaseEnd": source_audit,
                        "priceRowCutoff": {"phaseEnd": end.isoformat(), "byProduct": phase_row_audit},
                        "directPositivePriceVolumeDays": {p: len(direct_series[p]) for p in PRODUCTS}},
        "plannedCount": len(ordered_forecasts),
        "selectedCrops": selected_crops if phase == "evaluation" else None,
        "forecasts": ordered_forecasts,
    }
    doc["manifestSha256"] = hashlib.sha256(_canonical(doc).rstrip(b"\n")).hexdigest()
    return doc


def _unissued_grid_row(key: tuple[str, dt.date, int], phase: str, selected: set[str], reason: str,
                       baseline_row: dict[str, Any]) -> dict[str, Any]:
    product, target, horizon = key
    origin = target - dt.timedelta(days=horizon)
    none = _not_issued(reason)
    source_models = baseline_row["models"]
    is_selected = product in selected
    use_fixed_policy = phase == "evaluation"
    policy = {"method": "fixed50" if use_fixed_policy else "preserved106Blend",
              "status": source_models["fixed50"]["status"] if use_fixed_policy else "not_issued",
              "prediction": source_models["fixed50"]["prediction"] if use_fixed_policy else None,
              "reason": "selected_candidate_unavailable_fallback" if is_selected and use_fixed_policy else reason,
              "provenance": {"fallback": True}}
    return {"product": product, "target": target.isoformat(), "origin": origin.isoformat(), "horizon": horizon,
            "eligible": False, "selectedCrop": is_selected if phase == "evaluation" else None,
            "models": {"causalDirect": copy.deepcopy(source_models["causalDirect"]),
                       "causalSarima": copy.deepcopy(source_models["causalSarima"]),
                       "causal50": copy.deepcopy(source_models["causal50"]),
                       "fixed50": copy.deepcopy(source_models["fixed50"]),
                       "preserved106": copy.deepcopy(none) if is_selected else _not_selected(),
                       "preserved106Blend": copy.deepcopy(none) if is_selected else _not_selected()},
            "policyForecast": policy}


def _unselected_policy_row(key: tuple[str, dt.date, int], baseline_row: dict[str, Any]) -> dict[str, Any]:
    product, target, horizon = key
    origin = target - dt.timedelta(days=horizon)
    source_models = baseline_row["models"]
    fixed = copy.deepcopy(source_models["fixed50"])
    not_selected = _not_selected()
    return {"product": product, "target": target.isoformat(), "origin": origin.isoformat(), "horizon": horizon,
            "eligible": origin >= PHASES["evaluation"][0], "selectedCrop": False,
            "models": {"causalDirect": copy.deepcopy(source_models["causalDirect"]),
                       "causalSarima": copy.deepcopy(source_models["causalSarima"]),
                       "causal50": copy.deepcopy(source_models["causal50"]),
                       "fixed50": fixed, "preserved106": not_selected,
                       "preserved106Blend": copy.deepcopy(not_selected)},
            "policyForecast": _policy_forecast(False, not_selected, fixed)}


def _validate_manifest(doc: dict[str, Any], phase: str) -> None:
    if phase not in PHASES or doc.get("phase") != "forecast_only" or doc.get("phaseName") != phase or doc.get("targetValuesAttached") is not False:
        raise ValueError("invalid preserved-feature answer-free phase manifest")
    if _unsigned_sha(doc, "manifestSha256") != doc.get("manifestSha256"):
        raise ValueError("preserved-feature manifest digest mismatch")
    rows = doc.get("forecasts")
    expected = set(_expected_keys(phase))
    if not isinstance(rows, list):
        raise ValueError("preserved-feature forecasts must be a list")
    actual = set()
    for row in rows:
        if not isinstance(row, dict) or row.get("product") not in PRODUCTS:
            raise ValueError("invalid preserved-feature forecast row")
        day, origin, h = _date(row.get("target")), _date(row.get("origin")), row.get("horizon")
        key = (row["product"], day, h)
        if day is None or origin != day - dt.timedelta(days=h) or h not in HORIZONS or key in actual:
            raise ValueError("invalid or duplicate preserved-feature key")
        actual.add(key)
        models = row.get("models")
        if not isinstance(models, dict) or set(models) != {"causalDirect", "causalSarima", "causal50", "fixed50", "preserved106", "preserved106Blend"}:
            raise ValueError("preserved-feature method grid mismatch")
        for method, prediction in models.items():
            status = prediction.get("status")
            if status not in ("issued", "not_issued", "not_selected"):
                raise ValueError(f"invalid {method} status")
            value = _positive(prediction.get("prediction"))
            if status == "issued" and value is None or status != "issued" and prediction.get("prediction") is not None:
                raise ValueError(f"invalid {method} prediction/status pairing")
    if actual != expected or doc.get("plannedCount") != len(expected):
        raise ValueError("incomplete preserved-feature forecast grid")


def _load_decision(decision_path: Path, confirmation_path: Path) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    return decision, *_decision_fields(decision, confirmation_path)


def build_from_paths(phase: str, decision_path: Path | None = None, progress: Any = None,
                     confirmation_path: Path | None = None,
                     confirmation_report_path: Path | None = None) -> dict[str, Any]:
    _check_inputs()
    prices = json.loads(PRICES_PATH.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    _validate_baseline(baseline)
    if phase == "confirmation":
        selected, decision_hashes = None, None
    elif phase == "evaluation":
        if decision_path is None:
            decision_path = CONFIRMATION_DECISION
        if confirmation_path is None:
            confirmation_path = CONFIRMATION_OUT
        if confirmation_report_path is None:
            confirmation_report_path = CONFIRMATION_REPORT
        from . import preserved_feature_score as score
        selected, decision_hashes = score.validate_confirmation_gate_paths(
            confirmation_path, confirmation_report_path, decision_path,
            baseline_path=BASELINE_PATH, prices_path=PRICES_PATH)
    else:
        raise ValueError("phase must be confirmation or evaluation")
    hashes = {"pricesFile": PRICES_SHA256, "equalInfoManifestFile": BASELINE_SHA256}
    if decision_path is not None and phase == "evaluation":
        hashes["confirmationDecisionFile"] = _sha(decision_path)
        hashes["confirmationReportFile"] = _sha(confirmation_report_path)
    return build(prices, baseline, phase, selected_crops=selected, decision_hashes=decision_hashes,
                 input_hashes=hashes, progress=progress)


def write_exclusive(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical(doc))


def verify_manifest(path: Path, phase: str, decision_path: Path | None = None,
                    confirmation_path: Path | None = None,
                    confirmation_report_path: Path | None = None) -> bool:
    recorded = json.loads(path.read_text(encoding="utf-8"))
    _validate_manifest(recorded, phase)
    rebuilt = build_from_paths(phase, decision_path, confirmation_path=confirmation_path,
                               confirmation_report_path=confirmation_report_path)
    return _canonical(recorded) == _canonical(rebuilt)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--phase", choices=tuple(PHASES), required=True)
    build_parser.add_argument("--decision", type=Path)
    build_parser.add_argument("--confirmation-manifest", type=Path)
    build_parser.add_argument("--confirmation-report", type=Path)
    build_parser.add_argument("--output", type=Path)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--phase", choices=tuple(PHASES), required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--decision", type=Path)
    verify_parser.add_argument("--confirmation-manifest", type=Path)
    verify_parser.add_argument("--confirmation-report", type=Path)
    args = parser.parse_args(argv)
    if args.command == "build":
        if args.phase == "confirmation" and args.decision is not None:
            parser.error("confirmation build does not accept --decision")
        if args.phase == "evaluation" and args.decision is None and not CONFIRMATION_DECISION.is_file():
            parser.error("evaluation build requires a signed --decision")
        if args.phase == "evaluation" and args.confirmation_report is None and not CONFIRMATION_REPORT.is_file():
            parser.error("evaluation build requires a confirmation report")
        output = args.output or (CONFIRMATION_OUT if args.phase == "confirmation" else EVALUATION_OUT)
        doc = build_from_paths(args.phase, args.decision,
                               progress=lambda phase, crop, h, month, n: print(f"{phase} {crop}/h{h} {month[0]}-{month[1]:02d}: {n} rows", flush=True),
                               confirmation_path=args.confirmation_manifest,
                               confirmation_report_path=args.confirmation_report)
        write_exclusive(output, doc)
    else:
        if not verify_manifest(args.manifest, args.phase, args.decision, args.confirmation_manifest,
                               args.confirmation_report):
            raise SystemExit("manifest replay mismatch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
