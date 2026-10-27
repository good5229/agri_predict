"""Issue same-snapshot prospective comparisons for causal50, E and E106.

This command reads only a completed local daily price snapshot and its selected
source provenance. It creates forecasts, never joins target outcomes, and does
not publish into the existing dashboard or overwrite prior issue artifacts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sqlite3
import subprocess
import sys
import uuid
from importlib import metadata
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from . import four_crop_forecast as four
from . import equal_info_online_forecasts as online
from . import preserved_feature_forecasts as preserved
from . import prospective_eval
from . import trend_response_correction as trend
from . import dashboard
from . import daily_training, market_mix
from . import e106_prospective_eval as evaluator

ROOT = Path(__file__).resolve().parents[1]
KST = ZoneInfo("Asia/Seoul")
PRODUCTS = ("cabbage", "radish", "onion", "garlic")
HORIZONS = tuple(range(1, 11))
MARKETS = ("110001", "110008")
SELECTION_PATH = ROOT / "outputs/experiments/trend-response-correction-20261010/selection.json"
SELECTION_SHA256 = "59fa52c3f7c9664d52ad8ce23987eb4d6c70af0fca9907c25596a84eb77ba26b"
OUT_ROOT = ROOT / "outputs/prospective-e106"
STATUS_PATH = ROOT / "runs/daily-training-status.json"
FINGERPRINT_PYTHON = ROOT / ".venv/bin/python"
FINGERPRINT_MODE = "validate-snapshot"
ISSUER_RUNTIME = {"python": (3, 11), "numpy": "1.26.4", "scikit-learn": "1.3.0",
                  "statsmodels": "0.14.0", "scipy": "1.10.1", "joblib": "1.2.0"}


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _package_version(name: str):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _validate_model_runtime():
    versions = {name: _package_version(name) for name in
                ("numpy", "scikit-learn", "statsmodels", "scipy", "joblib")}
    mismatches = []
    if tuple(sys.version_info[:2]) != ISSUER_RUNTIME["python"]:
        mismatches.append(f"python={sys.version_info.major}.{sys.version_info.minor}")
    for name, expected in ISSUER_RUNTIME.items():
        if name == "python":
            continue
        if versions.get(name) != expected:
            mismatches.append(f"{name}={versions.get(name)} (expected {expected})")
    if mismatches:
        raise RuntimeError("model-fitting runtime mismatch: " + ", ".join(mismatches))
    return {"python": sys.version.split()[0], "executable": str(Path(sys.executable).resolve()),
            "packages": versions, "requirements": {k: v for k, v in ISSUER_RUNTIME.items()}}


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _date(value: Any, label: str) -> dt.date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc


def _timestamp(value: Any, label: str) -> dt.datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a timezone-aware timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be a timezone-aware timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must be a timezone-aware timestamp")
    return parsed.astimezone(KST)


def _load_selection(selection_raw: bytes) -> dict[tuple[str, str], dict[str, Any]]:
    if _sha(selection_raw) != SELECTION_SHA256:
        raise ValueError("pinned trend selection file hash mismatch")
    selection = json.loads(selection_raw)
    unsigned = dict(selection)
    digest = unsigned.pop("selectionSha256", None)
    if digest != _sha(_canonical(unsigned)):
        raise ValueError("pinned trend selection internal digest mismatch")
    if selection.get("phase") != "selection_only" or selection.get("targetValuesAttached") is not False:
        raise ValueError("trend selection must be answer-free")
    rows = selection.get("groups")
    if not isinstance(rows, list):
        raise ValueError("trend selection groups are missing")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed trend selection row")
        key = row.get("product"), row.get("regime")
        if key in result or row.get("regime") not in ("aligned", "conflicting"):
            raise ValueError("duplicate or invalid trend selection group")
        scale = row.get("scale")
        if scale not in (0.75, 1.0, 1.25):
            raise ValueError("unsupported frozen trend scale")
        result[key] = row
    expected = {("cabbage", "aligned"): 1.25, ("cabbage", "conflicting"): 0.75,
                ("radish", "aligned"): 1.0, ("radish", "conflicting"): 1.0}
    if {key: row["scale"] for key, row in result.items() if key in expected} != expected:
        raise ValueError("frozen trend selection values changed")
    return result


def _validate_prices(raw: dict[str, Any], issue_date: dt.date, raw_bytes: bytes):
    if not isinstance(raw, dict) or raw.get("schemaVersion") != 1:
        raise ValueError("price snapshot must use dashboard schemaVersion 1")
    meta = raw.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("price snapshot meta is required")
    cutoff = _date(meta.get("evaluationEnd"), "snapshot evaluationEnd")
    data_end = _date(meta.get("dataEnd"), "snapshot dataEnd")
    if cutoff > issue_date - dt.timedelta(days=1) or data_end > cutoff:
        raise ValueError("price snapshot contains data after the issue-date cutoff")
    if not isinstance(meta.get("aggregation"), str) or "signed" not in meta.get("aggregation", "") and "부호 있는 총금액" not in meta.get("aggregation", ""):
        raise ValueError("price snapshot does not document its signed aggregation")
    if (not isinstance(meta.get("marketNames"), list)
            or set(meta["marketNames"]) != set(dashboard.MARKETS.values())):
        raise ValueError("price snapshot marketNames do not match the validated two markets")
    records, _audit = online._records(raw)
    direct_series = online.direct_horizon._observations(raw)
    if set(PRODUCTS).difference(records) or set(PRODUCTS).difference(direct_series):
        raise ValueError("snapshot lacks one or more forecast products")
    # Validate the exact source market identity carried by each price row.
    for crop in PRODUCTS:
        for day, record in records[crop].items():
            if day > cutoff or day > data_end:
                raise ValueError(f"price row is after declared snapshot end: {crop}/{day}")
            if not record["markets"] or not record["markets"].issubset(set(MARKETS)):
                raise ValueError(f"price row has missing or unknown market identity: {crop}/{day}")
            if record.get("volume") is None or _positive(record.get("price")) is None:
                raise ValueError(f"price row lacks positive eligible price/volume: {crop}/{day}")
    return cutoff, data_end, _sha(raw_bytes), records, direct_series


def _validate_provenance(rows: Any, cutoff: dt.date, issued_at: dt.datetime | None = None):
    if not isinstance(rows, list):
        raise ValueError("provenance must be the daily-training selected-provenance array")
    scopes = prospective_eval._validate_provenance(rows, selected_end=cutoff)
    for row in rows:
        response_at = _timestamp(row.get("responseAt"), "selected source responseAt")
        if issued_at is not None and response_at > issued_at:
            raise ValueError("selected source responseAt is after the captured issue timestamp")
    # Every represented day is a complete set of market/product scopes. Missing
    # dates remain unknown and are not converted into empty responses.
    normalized = {}
    for (day, product), by_market in scopes.items():
        if product not in PRODUCTS:
            continue
        normalized.setdefault(day, {})[product] = by_market
    if not normalized:
        raise ValueError("provenance contains no rows for the four forecast products")
    return scopes


def _validate_price_provenance(records, scopes):
    for crop in PRODUCTS:
        for day in records[crop]:
            selected = scopes.get((day, crop), {})
            if set(selected) != set(MARKETS):
                raise ValueError(f"price observation has incomplete selected source scopes: {crop}/{day}")
            if sum(selected[market]["rowCount"] for market in MARKETS) <= 0:
                raise ValueError(f"price observation conflicts with normal-empty source records: {crop}/{day}")


def _source_audit(records, direct_series, scopes, cutoff: dt.date, issue_date: dt.date):
    audit = {}
    audit_end = issue_date - dt.timedelta(days=1)
    start28, start90 = audit_end - dt.timedelta(days=27), audit_end - dt.timedelta(days=89)
    for crop in PRODUCTS:
        series = records[crop]
        last = max(series) if series else None
        statuses = []
        for offset in range(89, -1, -1):
            day = audit_end - dt.timedelta(days=offset)
            row = scopes.get((day, crop), {})
            if set(row) != set(MARKETS):
                state, rows, selected_sources = "incomplete_or_unknown", None, None
            else:
                rows = sum(row[m]["rowCount"] for m in MARKETS)
                state = "complete_empty_response" if rows == 0 else "complete_rows"
                selected_sources = {m: {"rowCount": row[m]["rowCount"], "jobId": row[m]["jobId"],
                                        "epochId": row[m]["epochId"], "responseAt": row[m]["responseAt"],
                                        "selectedPrevious": row[m]["selectedPrevious"]}
                                    for m in MARKETS}
            statuses.append({"date": day.isoformat(), "state": state, "selectedRowCount": rows,
                             "selectedSources": selected_sources})
        audit[crop] = {
            "lastObservedPriceDate": last.isoformat() if last else None,
            "sourceAgeDaysAtIssue": (issue_date - last).days if last else None,
            "positivePriceAndVolumeObservationCount28d": sum(start28 <= d <= audit_end for d in direct_series[crop]),
            "positivePriceAndVolumeObservationCount90d": sum(start90 <= d <= audit_end for d in direct_series[crop]),
            "completeEmptyScopeDays90d": sum(r["state"] == "complete_empty_response" for r in statuses),
            "incompleteOrUnknownScopeDays90d": sum(r["state"] == "incomplete_or_unknown" for r in statuses),
            "sourceStates90d": statuses,
        }
        audit[crop]["sourceWindowEnd"] = audit_end.isoformat()
        audit[crop]["actualSnapshotCutoff"] = cutoff.isoformat()
        audit[crop]["unknownDaysAfterSnapshotCutoff"] = max(0, (audit_end - cutoff).days)
    return audit


def _trend_prediction(crop, horizon, baseline, p0, features, selection):
    base = _positive(baseline)
    if base is None:
        return {"status": "not_issued", "prediction": None, "reason": "causal50_not_issued", "applied": False}
    if crop not in ("cabbage", "radish") or horizon not in range(4, 11):
        return {"status": "issued", "prediction": base, "reason": "unchanged_by_E_policy", "applied": False}
    if features.get("reason"):
        return {"status": "issued", "prediction": base, "reason": features["reason"], "applied": False}
    regime = features.get("regime")
    if regime not in ("aligned", "conflicting"):
        return {"status": "issued", "prediction": base, "reason": "neutral_trend", "applied": False}
    scale_row = selection.get((crop, regime))
    if not scale_row or not scale_row.get("supported"):
        return {"status": "issued", "prediction": base, "reason": "unsupported_selection_scale_one", "applied": False}
    value = trend._scale_movement(base, p0, scale_row["scale"])
    if value is None or not math.isfinite(value) or value <= 0:
        return {"status": "issued", "prediction": base, "reason": "nonfinite_scaled_prediction_fallback_causal50", "applied": False}
    return {"status": "issued", "prediction": value, "reason": None,
            "applied": scale_row["scale"] != 1.0, "scale": scale_row["scale"], "regime": scale_row["regime"]}


def _prepare_preserved_inputs(raw):
    rich_records, _ = preserved.rich._raw_maps(raw)
    direct = online.direct_horizon._observations(raw)
    records, ordered, prices, caches = {}, {}, {}, {}
    for crop in PRODUCTS:
        domain = direct[crop]
        if not set(domain).issubset(rich_records[crop]):
            raise ValueError(f"direct training domain is absent from preserved feature source: {crop}")
        records[crop] = {day: dict(rich_records[crop][day]) for day in domain}
        ordered[crop] = sorted(domain)
        prices[crop] = {day: value[0] for day, value in domain.items()}
        caches[crop] = {}
    return records, direct, ordered, prices, caches


def _finite_array(values):
    """JSON-safe numeric array with shape and exact nonfinite-position markers."""
    import numpy as np
    array = np.asarray(values, dtype=np.float64)
    flat, exceptional = [], []
    for index, value in enumerate(array.ravel(order="C")):
        number = float(value)
        if math.isfinite(number):
            flat.append(number)
        else:
            flat.append(None)
            exceptional.append({"flatIndex": index, "kind": "nan" if math.isnan(number) else ("positive_infinity" if number > 0 else "negative_infinity")})
    return {"shape": list(array.shape), "values": flat, "nonfinite": exceptional}


def _ridge_state(record):
    model = record.get("model")
    model_id = record.get("modelId")
    if model_id not in {"direct40", "preserved106"}:
        raise ValueError("Ridge state record requires an explicit direct40 or preserved106 model identity")
    if model is None:
        return {"modelId": model_id, "product": record.get("product"), "horizon": record.get("horizon"),
                "fitOrigin": record.get("fitOrigin").isoformat() if record.get("fitOrigin") else None,
                "featureOrder": list(record.get("featureNames") or []),
                "status": "unavailable", "reason": record.get("reason"),
                "training": record.get("trainingAudit")}
    scaler, ridge = model.named_steps["standardscaler"], model.named_steps["ridge"]
    targets = record["trainingTargets"]
    input_dates = record["trainingInputDates"]
    return {"modelId": model_id, "product": record["product"], "horizon": record["horizon"],
            "fitOrigin": record["fitOrigin"].isoformat(), "status": "fitted",
            "featureOrder": list(record["featureNames"]), "ridgeAlpha": float(ridge.alpha),
            "fitIntercept": bool(ridge.fit_intercept), "ridgeCoef": _finite_array(ridge.coef_),
            "ridgeIntercept": float(ridge.intercept_),
            "scaler": {"withMean": bool(scaler.with_mean), "withStd": bool(scaler.with_std),
                       "mean": _finite_array(scaler.mean_), "scale": _finite_array(scaler.scale_),
                       "variance": _finite_array(scaler.var_)},
            "training": {"count": len(targets), "trainingKeysSha256": _sha(_canonical([d.isoformat() for d in targets])),
                         "maxInputDate": max(input_dates).isoformat() if input_dates else None,
                         "maxTruthDate": max(targets).isoformat() if targets else None,
                         "fitOrigin": record["fitOrigin"].isoformat(),
                         "firstTruthDate": min(targets).isoformat() if targets else None,
                         "featureOrderSha256": _sha(_canonical(list(record["featureNames"])))} }


def _last_state(result, field, covariance=False):
    values = getattr(result.filter_results, field)
    if covariance:
        values = values[:, :, -1]
    else:
        values = values[:, -1]
    return _finite_array(values)


def _sarima_snapshot(result, *, product, fit_origin, calendar_date, fit_audit, event, update=None, reason=None):
    model = result.model
    rows = getattr(model.data, "row_labels", None)
    start = rows[0].date().isoformat() if rows is not None and len(rows) else None
    end = rows[-1].date().isoformat() if rows is not None and len(rows) else None
    record = {"product": product, "event": event, "fitOrigin": fit_origin.isoformat(),
              "calendarDate": calendar_date.isoformat() if calendar_date else None,
              "fitAudit": fit_audit, "model": {"order": list(model.order),
              "seasonalOrder": list(model.seasonal_order), "trend": model.trend,
              "enforceStationarity": bool(model.enforce_stationarity),
              "enforceInvertibility": bool(model.enforce_invertibility),
              "stateDimension": int(model.k_states), "shockDimension": int(model.k_posdef),
              "params": _finite_array(result.params),
              "calendarIndex": {"start": start, "end": end,
                                "length": int(len(rows)) if rows is not None else None,
                                "frequency": "D"}},
              "filteredState": _last_state(result, "filtered_state"),
              "filteredStateCovariance": _last_state(result, "filtered_state_cov", covariance=True),
              "predictedState": _last_state(result, "predicted_state"),
              "predictedStateCovariance": _last_state(result, "predicted_state_cov", covariance=True)}
    initialization = model.ssm.initialization
    record["model"].update({
        "initializationType": getattr(initialization, "initialization_type", None),
        "initializationConstant": _finite_array(getattr(initialization, "constant", [])),
        "stationaryInitialCovariance": _finite_array(getattr(initialization, "stationary_cov", [])),
        "approximateDiffuseVariance": float(getattr(initialization, "approximate_diffuse_variance", 0.0)),
        "filterOptions": {name: int(getattr(model.ssm, name)) for name in
                           ("filter_method", "inversion_method", "stability_method", "conserve_memory", "filter_timing")},
        "filterTolerance": float(model.ssm.tolerance),
        "initialState": _finite_array(result.filter_results.initial_state),
        "initialStateCovariance": _finite_array(result.filter_results.initial_state_cov)})
    if update is not None:
        scalar = float(update.iloc[0] if hasattr(update, "iloc") else update[0])
        record["updateInput"] = {"value": scalar if math.isfinite(scalar) else None,
                                 "state": "observed" if math.isfinite(scalar) else "missing_calendar_observation",
                                 "units": "log_price"}
    if reason is not None:
        record["reason"] = reason
    return record


def _make_state_sink():
    state = {"ridgeModels": [], "sarimaModels": {}}

    def capture(record):
        if record.get("kind") == "ridge" or "model" in record:
            state["ridgeModels"].append(_ridge_state(record))
            return
        if record.get("kind") != "sarima":
            raise ValueError("unknown model-state capture event")
        product = record["product"]
        model = state["sarimaModels"].setdefault(product, {"updates": []})
        event = record["event"]
        if event == "fit_failed":
            model["status"] = "fit_failed"
            model["reason"] = record["reason"]
        elif event == "pre_update_checkpoint":
            model["status"] = "fitted"
            model["preUpdateCheckpoint"] = _sarima_snapshot(
                record["fit"], product=product, fit_origin=record["fitOrigin"],
                calendar_date=record["calendarDate"], fit_audit=record["fitAudit"], event=event)
        elif event == "state_update":
            model["updates"].append(_sarima_snapshot(
                record["fit"], product=product, fit_origin=record["fitOrigin"],
                calendar_date=record["calendarDate"], fit_audit=None, event=event,
                update=record["update"]))
        elif event == "state_update_failed":
            model["updates"].append({"event": event, "calendarDate": record["calendarDate"].isoformat(),
                                     "reason": record["reason"]})
            model["status"] = "update_failed"
        else:
            raise ValueError(f"unknown SARIMA state event: {event}")

    return state, capture


def _sarima_state_as_of(model_state: dict[str, Any], cutoff: dt.date) -> dict[str, Any] | None:
    """Select the last persisted SARIMA filter state no later than a hidden-data cutoff."""
    if model_state.get("status") != "fitted":
        return None
    checkpoint = model_state.get("preUpdateCheckpoint")
    if not isinstance(checkpoint, dict):
        raise ValueError("fitted SARIMA state lacks its pre-update checkpoint")
    checkpoint_day = dt.date.fromisoformat(checkpoint["calendarDate"])
    if checkpoint_day > cutoff:
        raise ValueError("requested cutoff predates the saved SARIMA checkpoint; hidden-state replay is unavailable")
    selected = checkpoint
    for update in model_state.get("updates", []):
        if update.get("event") == "state_update_failed":
            break
        day = dt.date.fromisoformat(update["calendarDate"])
        if day > cutoff:
            break
        selected = update
    return selected


def _fit_garlic_106(raw, issue_date, causal_rows, progress=None, *, state_sink=None):
    records, direct, ordered, prices, caches = _prepare_preserved_inputs(raw)
    fit_origin = issue_date.replace(day=1)
    source_dates = sorted(direct["garlic"])
    result = {}
    for horizon in HORIZONS:
        rows = {}
        for target in source_dates:
            feat = preserved._feature_pair("garlic", target, horizon, direct, records,
                                            {"garlic": caches["garlic"]}, ordered, prices)
            if feat is not None:
                rows[target] = feat
        training = [day for day in source_dates if day >= preserved.online.narrow.TRAIN_START
                    and day < fit_origin and day in rows]
        audit = preserved._fit_audit(training, fit_origin)
        expected = causal_rows[horizon]["models"].get("causalDirect", {}).get("training")
        if not isinstance(expected, dict):
            raise ValueError(f"missing frozen causalDirect training audit for garlic/h{horizon}")
        preserved._assert_training_support(audit, expected, "garlic", horizon,
                                           (fit_origin.year, fit_origin.month))
        fit, error = None, None
        if len(training) < preserved.MIN_TRAIN:
            error = "fewer_than_90_valid_training_targets"
        else:
            x = [rows[day]["preserved106"] for day in training]
            y = [math.log(direct["garlic"][day][0] / rows[day]["p0"]) for day in training]
            try:
                fit = online._fit_ridge(x, y, preserved.RIDGE_ALPHA)
            except Exception as exc:
                error = f"ridge106_fit_failed:{type(exc).__name__}"
            if fit is not None and state_sink is not None:
                names = list(online.modeling.FEATURE_NAMES) + [
                    name for name in preserved.library.FEATURE_NAMES
                    if name not in preserved.OMITTED_LIBRARY_FEATURES]
            state_sink({"kind": "ridge", "model": fit, "modelId": "preserved106", "product": "garlic", "horizon": horizon,
                            "fitOrigin": fit_origin, "alpha": preserved.RIDGE_ALPHA,
                            "trainingTargets": list(training),
                            "trainingInputDates": [rows[d]["p0Date"] for d in training],
                            "trainingAudit": audit, "featureNames": names})
        target = issue_date + dt.timedelta(days=horizon)
        feat = preserved._feature_pair("garlic", target, horizon, direct, records,
                                       {"garlic": caches["garlic"]}, ordered, prices)
        prediction = None
        reason = error
        if not error and fit is not None and feat is not None:
            try:
                prediction = feat["p0"] * math.exp(float(fit.predict([feat["preserved106"]])[0]))
                if _positive(prediction) is None:
                    prediction, reason = None, "invalid_ridge106_prediction"
            except Exception as exc:
                reason = f"ridge106_predict_failed:{type(exc).__name__}"
        elif feat is None:
            reason = reason or "feature_unavailable_or_source_too_old"
        base = causal_rows[horizon]["models"]["causal50"]
        sarima = causal_rows[horizon]["models"]["causalSarima"]
        direct_row = {"status": "issued" if prediction else "not_issued", "prediction": prediction,
                      "reason": None if prediction else reason, "training": {**audit, "featureCount": 106, "alpha": preserved.RIDGE_ALPHA},
                      "provenance": ({"p0Date": feat["p0Date"].isoformat(), "p0AgeDays": feat["p0AgeDays"]} if feat else {})}
        if direct_row["status"] == "issued":
            blend = online._blend_ridge_sarima(direct_row, sarima)
            e106 = blend if blend.get("status") == "issued" else base
            fallback = _blend_fallback_reason(blend)
        else:
            blend = {"status": "not_issued", "prediction": None, "reason": reason,
                     "provenance": {"fallback": False}}
            e106, fallback = base, "candidate_unavailable_fallback_E"
        if fit is None and state_sink is not None:
            names = list(online.modeling.FEATURE_NAMES) + [
                name for name in preserved.library.FEATURE_NAMES
                if name not in preserved.OMITTED_LIBRARY_FEATURES]
            state_sink({"kind": "ridge", "modelId": "preserved106", "product": "garlic", "horizon": horizon,
                        "fitOrigin": fit_origin, "status": "unavailable", "reason": error,
                        "trainingAudit": audit, "featureNames": names})
        if not e106 or e106.get("status") != "issued":
            policy = {"status": "not_issued", "prediction": None,
                      "reason": e106.get("reason") or "E_and_E106_unavailable", "fallbackReason": fallback}
        else:
            policy = {"status": "issued", "prediction": e106["prediction"],
                      "reason": None, "fallbackReason": fallback}
        result[horizon] = {"direct106": direct_row, "blend106": blend, "forecast": policy}
        if progress:
            progress({"event": "ridge106_fit", "product": "garlic", "horizon": horizon,
                      "fitOrigin": fit_origin.isoformat(), "trainingTargets": len(training),
                      "status": "fitted" if fit is not None else error})
    return result


def _blend_fallback_reason(blend):
    if blend.get("status") != "issued":
        return "candidate_blend_unavailable_fallback_E"
    if blend.get("provenance", {}).get("fallback"):
        return "sarima_unavailable_direct106_only_fallback"
    return None


def _module_hashes():
    modules = {"issuer": Path(__file__), "fourCropForecast": Path(four.__file__),
               "equalInfoForecasts": Path(online.__file__), "preservedFeatureForecasts": Path(preserved.__file__),
               "trendResponseCorrection": Path(trend.__file__), "dashboardSchema": Path(dashboard.__file__),
               "directHorizon": Path(online.direct_horizon.__file__),
               "modeling": Path(online.modeling.__file__), "featureLibrary": Path(preserved.library.__file__),
               "richPriceMaps": Path(preserved.rich.__file__), "narrowPriceHelper": Path(preserved.narrow.__file__),
               "calendarSarima": Path(online.ten.__file__),
               "dailyTraining": Path(daily_training.__file__), "marketMix": Path(market_mix.__file__),
               "marketExtension": Path(market_mix.me.__file__)}
    return {name: _sha(path.read_bytes()) for name, path in modules.items()}


def _fingerprint_probe(prices_path: Path, provenance_path: Path,
                       expected_prices_sha: str, expected_provenance_sha: str, db_path: Path):
    """Validate frozen source bytes and reconcile signed database rows in the pinned runtime."""
    prices_raw, provenance_raw = prices_path.read_bytes(), provenance_path.read_bytes()
    if _sha(prices_raw) != expected_prices_sha or _sha(provenance_raw) != expected_provenance_sha:
        raise ValueError("source file bytes changed before the pinned fingerprint check")
    prices, provenance = json.loads(prices_raw), json.loads(provenance_raw)
    if not isinstance(provenance, list):
        raise ValueError("selected provenance must be an array for snapshot fingerprint validation")
    fingerprint = daily_training._pipeline_fingerprint(prices, provenance)
    core = online.direct_horizon._observations(prices)
    try:
        _markets, extraction = market_mix.extract_market_data(db_path, prices, core)
    except Exception as exc:
        raise ValueError(f"signed raw-source reconciliation failed: {type(exc).__name__}: {exc}") from exc
    if not isinstance(extraction, dict) or not isinstance(extraction.get("selected"), dict):
        raise ValueError("signed raw-source reconciliation lacks its selected provenance section")
    selected = extraction["selected"]
    required = ("selectedRequestHashes", "selectedScopesSha256", "selectedScopeCount",
                "selectedQueryStates", "selectedRequestCount")
    if any(key not in selected for key in required):
        raise ValueError("signed raw-source reconciliation lacks selected provenance hashes")
    if not isinstance(extraction.get("reconciliation"), dict) or not isinstance(extraction.get("broad"), dict):
        raise ValueError("signed raw-source reconciliation lacks reconciliation or broad-source results")
    packages = {name: _package_version(name) for name in
                ("numpy", "scikit-learn", "scipy", "joblib", "statsmodels")}
    return {"fingerprint": fingerprint, "pricesSha256": _sha(prices_raw),
            "provenanceSha256": _sha(provenance_raw),
            "runtime": {"python": sys.version.split()[0], "executable": str(Path(sys.executable).resolve()),
                        "packages": packages, "sqliteVersion": sqlite3.sqlite_version},
            "databaseReconciliation": {
                "dailyTrainingFingerprintRecomputed": fingerprint,
                "selectedScopeSha256": selected["selectedScopesSha256"],
                "selectedScopeCount": selected["selectedScopeCount"],
                "selectedRequestCount": selected["selectedRequestCount"],
                "selectedQueryStates": selected["selectedQueryStates"],
                "selectedRequestHashes": selected["selectedRequestHashes"],
                "reconciliation": extraction["reconciliation"],
                "broadScopesSha256": extraction["broad"].get("scopesSha256"),
                "broadScopeCount": extraction["broad"].get("scopeCount")}}


def _verify_snapshot_content(prices_raw: bytes, provenance_raw: bytes, latest: dict[str, Any], db_path: Path,
                             *, prices_path: Path, provenance_path: Path,
                             fingerprint_python: Path = FINGERPRINT_PYTHON):
    """Validate exact source files and signed DB provenance before fitting."""
    if not fingerprint_python.is_file() or not os.access(fingerprint_python, os.X_OK):
        raise ValueError(f"pinned fingerprint interpreter is unavailable: {fingerprint_python}")
    if prices_path.read_bytes() != prices_raw or provenance_path.read_bytes() != provenance_raw:
        raise ValueError("source file bytes changed before the pinned fingerprint check")
    command = [str(fingerprint_python), "-m", "agri_data.e106_prospective", FINGERPRINT_MODE,
               "--prices", str(prices_path.resolve()), "--provenance", str(provenance_path.resolve()),
               "--prices-sha256", _sha(prices_raw), "--provenance-sha256", _sha(provenance_raw),
               "--db", str(db_path.resolve())]
    helper_env = dict(os.environ)
    helper_env.pop("PYTHONPATH", None)
    helper_env.pop("PYTHONHOME", None)
    helper_env["PYTHONNOUSERSITE"] = "1"
    try:
        completed = subprocess.run(command, cwd=ROOT, env=helper_env, text=True, capture_output=True,
                                   timeout=900, check=False)
    except subprocess.TimeoutExpired as exc:
        raise ValueError("pinned source-validation subprocess exceeded the 900-second limit") from exc
    except Exception as exc:
        raise ValueError(f"pinned source-validation subprocess could not run: {type(exc).__name__}") from exc
    if completed.returncode != 0:
        raise ValueError(f"pinned source-validation subprocess failed before fitting (exit {completed.returncode})")
    try:
        probe = json.loads(completed.stdout)
        recomputed = probe["fingerprint"]
        probe_prices_sha, probe_provenance_sha = probe["pricesSha256"], probe["provenanceSha256"]
        probe_runtime = probe["runtime"]
        binding = probe["databaseReconciliation"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("pinned source-validation subprocess returned an invalid result") from exc
    if probe_prices_sha != _sha(prices_raw) or probe_provenance_sha != _sha(provenance_raw):
        raise ValueError("pinned source-validation subprocess read different source-file bytes")
    if recomputed != latest.get("snapshotSha256"):
        raise ValueError("daily-training snapshot fingerprint mismatch; frozen source code or source files changed")
    if not isinstance(binding, dict) or binding.get("dailyTrainingFingerprintRecomputed") != recomputed:
        raise ValueError("pinned source-validation subprocess returned a mismatched binding")
    if (not isinstance(probe_runtime, dict) or not isinstance(probe_runtime.get("python"), str)
            or not isinstance(probe_runtime.get("executable"), str)
            or not isinstance(probe_runtime.get("packages"), dict)
            or not isinstance(probe_runtime.get("sqliteVersion"), str)):
        raise ValueError("pinned source-validation subprocess returned invalid runtime details")
    required = ("selectedScopeSha256", "selectedScopeCount", "selectedRequestCount", "selectedQueryStates",
                "selectedRequestHashes", "reconciliation", "broadScopesSha256", "broadScopeCount")
    if any(key not in binding for key in required):
        raise ValueError("pinned source-validation subprocess returned an incomplete reconciliation")
    if (not isinstance(binding["selectedScopeSha256"], str)
            or not isinstance(binding["selectedQueryStates"], dict)
            or not isinstance(binding["selectedRequestHashes"], dict)
            or not isinstance(binding["reconciliation"], dict)
            or isinstance(binding["selectedScopeCount"], bool)
            or not isinstance(binding["selectedScopeCount"], int)
            or isinstance(binding["selectedRequestCount"], bool)
            or not isinstance(binding["selectedRequestCount"], int)):
        raise ValueError("pinned source-validation subprocess returned invalid reconciliation fields")
    return {**binding, "fingerprintValidationRuntime": probe_runtime}


def build_forecast(prices_raw: bytes, provenance_raw: bytes, selection_raw: bytes,
                   issue_date: dt.date, issued_at: dt.datetime | None = None, progress=None, *,
                   state_sink=None):
    issuer_runtime = _validate_model_runtime()
    if issued_at is None:
        issued_at = dt.datetime.now(KST)
    if issued_at.tzinfo is None or issued_at.utcoffset() is None:
        raise ValueError("issued_at must be timezone-aware")
    issued_at = issued_at.astimezone(KST)
    if issued_at.date() != issue_date:
        raise ValueError("issue date must match issue timestamp in KST")
    if issue_date > dt.datetime.now(KST).date():
        raise ValueError("issue date cannot be in the future")
    prices = json.loads(prices_raw)
    provenance_rows = json.loads(provenance_raw)
    cutoff, data_end, prices_sha, records, direct = _validate_prices(prices, issue_date, prices_raw)
    scopes = _validate_provenance(provenance_rows, cutoff, issued_at)
    _validate_price_provenance(records, scopes)
    selection = _load_selection(selection_raw)
    rows_by_crop = {}
    for crop in PRODUCTS:
        rows_by_crop[crop] = four._forecast_crop(records, direct, crop, issue_date, cutoff, progress,
                                                  state_sink=state_sink)
    causal = {(row["product"], row["horizon"]): row for rows in rows_by_crop.values() for row in rows}
    price_series = direct
    output_rows = []
    garlic_r106 = _fit_garlic_106(prices, issue_date,
                                  {h: causal[("garlic", h)] for h in HORIZONS}, progress,
                                  state_sink=state_sink)
    for crop in PRODUCTS:
        source = max((d for d in direct[crop] if d <= issue_date - dt.timedelta(days=1)), default=None)
        age = (issue_date - source).days if source else None
        for horizon in HORIZONS:
            row = causal[(crop, horizon)]
            baseline = row["models"]["causal50"]
            vector, p0, audit = online._direct_feature(direct[crop], issue_date + dt.timedelta(days=horizon), horizon, {})
            features = trend._price_features(price_series, crop, issue_date)
            if _positive(p0) is None:
                features = {"reason": audit.get("reason", "p0_missing_or_older_than_14_days"), **features}
            e = _trend_prediction(crop, horizon, baseline.get("prediction"), p0, features, selection)
            if crop == "garlic":
                garlic = garlic_r106[horizon]
                e106 = garlic["forecast"]
                r106, blend106 = garlic["direct106"], garlic["blend106"]
                fallback_reason = e106.get("fallbackReason")
            else:
                e106 = {"status": e["status"], "prediction": e["prediction"], "reason": e["reason"],
                        "fallbackReason": None}
                r106 = blend106 = None
                fallback_reason = None
            if p0 is None:
                p0 = None
            row_out = {"product": crop, "origin": issue_date.isoformat(),
                       "target": (issue_date + dt.timedelta(days=horizon)).isoformat(),
                       "horizon": horizon, "fitOrigin": row["fitOrigin"], "sourceCutoff": cutoff.isoformat(),
                       "p0": p0, "p0Date": audit.get("p0Date"), "sourceAgeDays": audit.get("p0AgeDays"),
                       "ETrendFeatures": {key: features.get(key) for key in
                                          ("cutoff", "regime", "p0Date", "p7Date", "p28Date", "p0", "p7", "p28",
                                           "momentum7Log", "momentum28Log", "reason")},
                       "trendFeatureStatus": features.get("reason"), "ESelectionScale": e.get("scale"),
                       "ERegime": e.get("regime"), "EApplied": e.get("applied", False),
                       "causal50": baseline, "causalDirect": row["models"]["causalDirect"],
                       "causalSarima": row["models"]["causalSarima"], "E": e,
                       "R106": r106, "E106Blend": blend106, "E106": e106,
                       "E106FallbackReason": fallback_reason}
            output_rows.append(row_out)
    output_rows.sort(key=lambda r: (PRODUCTS.index(r["product"]), r["horizon"]))
    if len(output_rows) != 40:
        raise AssertionError("issued candidate grid must contain exactly 40 keys")
    for row in output_rows:
        for method in ("causal50", "E", "E106"):
            model = row[method]
            if model.get("status") == "issued" and _positive(model.get("prediction")) is None:
                raise ValueError(f"invalid issued prediction {row['product']}/{row['horizon']}/{method}")
    state_audit = _source_audit(records, direct, scopes, cutoff, issue_date)
    provenance = json.loads(provenance_raw)
    result = {"schemaVersion": 1, "phase": "prospective_forecast_only", "targetValuesAttached": False,
              "issueDateKst": issue_date.isoformat(), "issueAtKst": issued_at.isoformat(timespec="seconds"),
              "targetDateRange": [(issue_date + dt.timedelta(days=1)).isoformat(),
                                  (issue_date + dt.timedelta(days=10)).isoformat()],
              "products": list(PRODUCTS), "horizons": list(HORIZONS), "plannedCount": 40,
              "methods": ["causal50", "E", "E106"],
              "sources": {"pricesPath": None, "provenancePath": None, "pricesFileSha256": prices_sha,
                          "provenanceFileSha256": _sha(provenance_raw),
                          "provenanceCanonicalSha256": _sha(_canonical(provenance)),
                          "selectionFileSha256": _sha(selection_raw), "sourceCutoff": cutoff.isoformat(),
                          "dataEnd": data_end.isoformat(), "noTargetOutcomesRead": True},
              "sourceAvailability": state_audit,
              "protocol": {"id": "prospective-e106-v1", "causal50": "four_crop_forecast._forecast_crop; direct Ridge40 alpha10 plus monthly SARIMA and existing blend/fallback",
                           "E": "frozen trend-response selection applied to causal50; all other keys retain causal50",
                           "E106": "E on non-garlic crops; garlic uses monthly direct Ridge106 plus same SARIMA blend; unavailable candidate falls back to E",
                           "featureCutoff": "issueDate minus one calendar day; snapshot cutoff cannot exceed this",
                           "firstInferenceReview": "first KST month-end after the same 10-day-mature issue cohort has at least 30 distinct positive paired target dates for each garlic horizon",
                           "accuracyClaim": "future outcomes are not yet available; no accuracy or statistical superiority claim is made",
                           "normalEmptyIsNotMarketClosure": True},
              "inputSha256": {"prices": prices_sha, "provenance": _sha(provenance_raw),
                              "selection": _sha(selection_raw), **_module_hashes()},
              "runtime": issuer_runtime,
              "forecasts": output_rows}
    result["forecastSha256"] = _sha(_canonical(result))
    return result


def _verify_latest_success(status_path: Path, prices_path: Path, provenance_path: Path,
                           *, captured_issue_at: dt.datetime | None = None):
    status = json.loads(status_path.read_text(encoding="utf-8"))
    latest = status.get("latestSuccess") if isinstance(status, dict) else None
    if not isinstance(latest, dict):
        raise ValueError("daily-training latestSuccess is unavailable")
    if captured_issue_at is not None:
        latest_finished = _timestamp(latest.get("finishedAtKst"), "daily-training latestSuccess finishedAtKst")
        if latest_finished > captured_issue_at:
            raise ValueError("daily-training completion is after the captured issue timestamp")
    run_dir = (ROOT / latest["runDirectory"]).resolve()
    if prices_path.resolve() != (run_dir / "prices.json").resolve():
        raise ValueError("prices file is not from daily-training latestSuccess")
    expected_provenance = (ROOT / latest["selectedProvenance"]).resolve()
    if provenance_path.resolve() != expected_provenance:
        raise ValueError("provenance file is not from daily-training latestSuccess")
    status_sha = _sha(status_path.read_bytes())
    run_status_path = run_dir / "run-status.json"
    if not run_status_path.is_file():
        raise ValueError("immutable daily-training run-status record is missing")
    run_status = json.loads(run_status_path.read_bytes())
    if (run_status.get("status") != "complete"
            or run_status.get("runId") != latest.get("runId")
            or run_status.get("cutoff") != latest.get("cutoff")
            or run_status.get("snapshotSha256") != latest.get("snapshotSha256")):
        raise ValueError("latestSuccess does not match a complete immutable run-status record")
    finished = _timestamp(run_status.get("finishedAtKst"), "daily-training run-status finishedAtKst")
    if captured_issue_at is not None and finished > captured_issue_at:
        raise ValueError("daily-training completion is after the captured issue timestamp")
    if latest.get("finishedAtKst") != run_status.get("finishedAtKst"):
        raise ValueError("daily-training completion timestamp differs from latestSuccess")
    return latest, status_sha


def _write_idempotent(out_dir: Path, result: dict[str, Any]):
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "forecast.json"
    payload = _pretty(result)
    if target.exists():
        if target.read_bytes() != payload:
            raise FileExistsError(f"immutable issue already exists with different contents: {target}")
        return target, False
    temp = target.with_name(target.name + f".{os.getpid()}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise FileExistsError(f"immutable issue already exists with different contents: {target}")
            return target, False
    finally:
        temp.unlink(missing_ok=True)
    return target, True


def _claim_source(status_path: Path, prices_path: Path, provenance_path: Path):
    """Best-effort source binding before validation so a failed attempt still has a claim."""
    try:
        status = json.loads(status_path.read_bytes())
        latest = status["latestSuccess"]
        source = {"runId": latest["runId"], "snapshotSha256": latest["snapshotSha256"],
                  "cutoff": latest["cutoff"], "pricesFileSha256": _sha(prices_path.read_bytes()),
                  "provenanceFileSha256": _sha(provenance_path.read_bytes())}
        return source if all(isinstance(v, str) and v for v in source.values()) else None
    except (OSError, KeyError, TypeError, ValueError):
        return None


def _require_claim_source(claim: dict[str, Any], actual_source: dict[str, Any]):
    required = {"runId", "snapshotSha256", "cutoff", "pricesFileSha256", "provenanceFileSha256"}
    if set(actual_source) != required or not all(isinstance(value, str) and value for value in actual_source.values()):
        raise ValueError("validated source binding is incomplete")
    if claim.get("source") is None or claim["source"] != actual_source:
        raise ValueError("validated source binding differs from the immutable first-attempt claim")


def _claim_path(registry_root: Path, issue_date: dt.date) -> Path:
    return registry_root / "days" / issue_date.isoformat() / "claim.json"


def _start_issue_claim(registry_root: Path, issue_date: dt.date, started_at: dt.datetime,
                       artifact_target: str, source: dict[str, Any] | None, *,
                       requested_source_paths: dict[str, str] | None = None,
                       source_binding_reason: str | None = None):
    if issue_date <= dt.date(2026, 10, 11):
        raise ValueError("the Oct 11 bootstrap is already selected; first future state begins Oct 12")
    if started_at.date() != issue_date or started_at.time().replace(tzinfo=None) < dt.time(18, 0):
        raise ValueError("a prospective first daily claim requires a same-day start at or after 18:00 KST")
    if started_at.tzinfo is None or started_at.utcoffset() != dt.timedelta(hours=9):
        raise ValueError("claim start timestamp must be timezone-aware KST")
    required_source = {"runId", "snapshotSha256", "cutoff", "pricesFileSha256", "provenanceFileSha256"}
    if source is not None and (not isinstance(source, dict) or set(source) != required_source or not all(
            isinstance(value, str) and value for value in source.values())):
        raise ValueError("a known claim source binding must be complete")
    protocol_path = registry_root / "protocol.json"
    if not protocol_path.is_file():
        evaluator.bootstrap_registry(registry_root)
    protocol = evaluator._load_protocol(registry_root)
    path = _claim_path(registry_root, issue_date)
    if path.exists():
        raise FileExistsError(f"first daily attempt is already claimed: {path}")
    claim = {"schemaVersion": 1, "issueDateKst": issue_date.isoformat(),
             "attemptId": f"e106-{issue_date.strftime('%Y%m%d')}-{uuid.uuid4().hex}",
             "startedAtKst": started_at.isoformat(timespec="seconds"), "attemptOrdinal": 1,
             "plannedForecastKeyCount": 40,
             "plannedForecastGridSha256": evaluator.grid_sha256(issue_date),
             "source": source, "artifactTarget": artifact_target,
             "protocolSha256": protocol["protocolSha256"],
             "sourceBindingStatus": "available" if source is not None else "unavailable",
             "requestedSourcePaths": requested_source_paths or {},
             "sourceBindingReason": source_binding_reason if source is None else None}
    claim["claimSha256"] = _sha(_canonical(claim))
    created = evaluator._write_create_only(path, claim)
    if not created:
        raise FileExistsError(f"first daily attempt was claimed concurrently: {path}")
    return claim


def _terminal_path(registry_root: Path, issue_date: dt.date) -> Path:
    return registry_root / "days" / issue_date.isoformat() / "terminal.json"


def _finish_issue_claim(registry_root: Path, issue_date: dt.date, claim: dict[str, Any],
                        *, status: str, finished_at: dt.datetime, artifact_path: str | None = None,
                        artifact_raw_sha256: str | None = None, artifact_canonical_sha256: str | None = None,
                        failure_type: str | None = None):
    if status not in {"complete", "failed"}:
        raise ValueError("terminal status must be complete or failed")
    if finished_at.tzinfo is None or finished_at.utcoffset() != dt.timedelta(hours=9):
        raise ValueError("terminal timestamp must be timezone-aware KST")
    started_at = _timestamp(claim["startedAtKst"], "claim startedAtKst")
    if finished_at < started_at:
        raise ValueError("terminal timestamp precedes claim start")
    if status == "complete" and (artifact_path != claim["artifactTarget"]
                                  or not artifact_raw_sha256 or not artifact_canonical_sha256):
        raise ValueError("complete terminal must bind the claimed artifact path and both hashes")
    if status == "complete":
        required_source = {"runId", "snapshotSha256", "cutoff", "pricesFileSha256", "provenanceFileSha256"}
        source = claim.get("source")
        if not isinstance(source, dict) or set(source) != required_source or not all(
                isinstance(value, str) and value for value in source.values()):
            raise ValueError("complete terminal requires the complete claimed source binding")
    terminal = {"schemaVersion": 1, "attemptId": claim["attemptId"], "status": status,
                "protocolSha256": claim["protocolSha256"], "claimSha256": claim["claimSha256"],
                "finishedAtKst": finished_at.isoformat(timespec="seconds")}
    if status == "complete":
        terminal.update({"artifactPath": artifact_path,
                         "artifactRawSha256": artifact_raw_sha256,
                         "artifactCanonicalSha256": artifact_canonical_sha256,
                         "source": claim["source"]})
    elif failure_type:
        terminal["failureType"] = failure_type
    terminal["terminalSha256"] = _sha(_canonical(terminal))
    path = _terminal_path(registry_root, issue_date)
    if not evaluator._write_create_only(path, terminal):
        raise FileExistsError(f"daily attempt already has an immutable terminal: {path}")
    return terminal


def _model_state_document(issue_date: dt.date, attempt_id: str, artifact_target: str,
                          source: dict[str, Any], forecast: dict[str, Any], captured: dict[str, Any]):
    ridge = sorted(captured["ridgeModels"], key=lambda row: (row.get("modelId", ""), row.get("product", ""), row.get("horizon", 0)))
    expected_ridge = ({("direct40", product, horizon) for product in PRODUCTS for horizon in HORIZONS}
                      | {("preserved106", "garlic", horizon) for horizon in HORIZONS})
    actual_ridge = {(row.get("modelId"), row.get("product"), row.get("horizon")) for row in ridge}
    if actual_ridge != expected_ridge or len(ridge) != len(expected_ridge):
        raise ValueError("model-state capture is incomplete or ambiguous: expected 40 direct40 and 10 preserved106 Ridge fit records")
    sarima = captured["sarimaModels"]
    if set(sarima) != set(PRODUCTS):
        raise ValueError("model-state capture is incomplete: expected four SARIMA fit records")
    doc = {"schemaVersion": 1, "kind": "prospective-e106-model-state-v1",
           "issueDateKst": issue_date.isoformat(), "attemptId": attempt_id,
           "forecastArtifactTarget": artifact_target,
           "source": source, "inputSha256": forecast["inputSha256"], "runtime": forecast["runtime"],
           "ridgeModels": ridge, "sarimaModels": sarima,
           "sarimaReplayRule": "start from pre-update checkpoint and use only persisted daily updates through the requested cutoff; missing calendar dates remain missing"}
    doc["modelStateSha256"] = _sha(_canonical(doc))
    return doc


def run_issue(issue_date: dt.date, prices_path: Path, provenance_path: Path,
              output_dir: Path | None = None, *, status_path: Path = STATUS_PATH,
              db_path: Path = ROOT / "data/agri.sqlite", progress=None,
              registry_path: Path = evaluator.DEFAULT_REGISTRY):
    captured_issue_at = dt.datetime.now(KST)
    if captured_issue_at.date() != issue_date:
        raise ValueError("issue date must be today's KST date for a live prospective issue")
    out_dir = output_dir or OUT_ROOT / issue_date.isoformat()
    target_path = Path(out_dir).resolve() / "forecast.json"
    state_path = Path(out_dir).resolve() / "model-state.json"
    try:
        artifact_target = target_path.relative_to(ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("prospective artifacts must remain under the project root") from exc
    initial_source = _claim_source(status_path, prices_path, provenance_path)
    claim = _start_issue_claim(registry_path, issue_date, captured_issue_at, artifact_target,
                               initial_source,
                               requested_source_paths={"status": four._display_source_path(status_path),
                                                       "prices": four._display_source_path(prices_path),
                                                       "provenance": four._display_source_path(provenance_path)},
                               source_binding_reason=(None if initial_source is not None else
                                                      "source metadata or input file unavailable before validation"))
    terminal_path = _terminal_path(registry_path, issue_date)
    try:
        if target_path.exists() or state_path.exists():
            raise FileExistsError("daily attempt has pre-existing forecast or model-state output")
        latest, status_sha = _verify_latest_success(status_path, prices_path, provenance_path,
                                                    captured_issue_at=captured_issue_at)
        if dt.date.fromisoformat(latest["cutoff"]) > issue_date - dt.timedelta(days=1):
            raise ValueError("completed daily snapshot cutoff must be no later than issue date minus one")
        prices_raw, provenance_raw = prices_path.read_bytes(), provenance_path.read_bytes()
        actual_claim_source = {"runId": latest["runId"], "snapshotSha256": latest["snapshotSha256"],
                               "cutoff": latest["cutoff"], "pricesFileSha256": _sha(prices_raw),
                               "provenanceFileSha256": _sha(provenance_raw)}
        _require_claim_source(claim, actual_claim_source)
        content_binding = _verify_snapshot_content(prices_raw, provenance_raw, latest, db_path,
                                                   prices_path=prices_path, provenance_path=provenance_path)
        selection_raw = SELECTION_PATH.read_bytes()
        latest_cutoff = dt.date.fromisoformat(latest["cutoff"])
        actual_cutoff, _data_end, _source_sha, records, _direct = _validate_prices(
            json.loads(prices_raw), issue_date, prices_raw)
        if actual_cutoff != latest_cutoff:
            raise ValueError("prices snapshot cutoff differs from daily-training latestSuccess")
        scopes = _validate_provenance(json.loads(provenance_raw), latest_cutoff, captured_issue_at)
        _validate_price_provenance(records, scopes)
        _load_selection(selection_raw)
        captured_state, state_sink = _make_state_sink()
        result = build_forecast(prices_raw, provenance_raw, selection_raw, issue_date,
                                issued_at=captured_issue_at, progress=progress, state_sink=state_sink)
        result["sources"].update({"dailyTrainingRunId": latest["runId"],
                                  "pricesPath": four._display_source_path(prices_path),
                                  "provenancePath": four._display_source_path(provenance_path),
                                  "dailyTrainingSnapshotRecordSha256": latest["snapshotSha256"],
                                  "dailyTrainingStatusFileSha256": status_sha,
                                  "dailyTrainingFinishedAtKst": latest["finishedAtKst"],
                                  "dailyTrainingCutoff": latest["cutoff"],
                                  "databasePath": four._display_source_path(db_path),
                                  "databaseReconciliation": content_binding,
                                  "issueAttempt": {"attemptId": claim["attemptId"],
                                                    "claimSha256": claim["claimSha256"],
                                                    "protocolSha256": claim["protocolSha256"]}})
        state_doc = _model_state_document(issue_date, claim["attemptId"], artifact_target,
                                          claim["source"], result, captured_state)
        state_created = evaluator._write_create_only(state_path, state_doc)
        if not state_created:
            raise FileExistsError("model-state artifact already exists")
        result["sources"]["modelState"] = {"path": four._display_source_path(state_path),
                                           "rawSha256": _sha(state_path.read_bytes()),
                                           "canonicalSha256": state_doc["modelStateSha256"]}
        result.pop("forecastSha256", None)
        result["forecastSha256"] = _sha(_canonical(result))
        artifact_path, created = _write_idempotent(Path(out_dir), result)
        if not created:
            raise FileExistsError("forecast artifact already exists for the claimed issue")
        raw_sha = _sha(artifact_path.read_bytes())
        unsigned = dict(result)
        unsigned.pop("forecastSha256", None)
        canonical_sha = _sha(_canonical(unsigned))
        _finish_issue_claim(registry_path, issue_date, claim, status="complete",
                            finished_at=dt.datetime.now(KST), artifact_path=artifact_target,
                            artifact_raw_sha256=raw_sha, artifact_canonical_sha256=canonical_sha)
        return artifact_path, created
    except Exception as exc:
        if not terminal_path.exists():
            try:
                _finish_issue_claim(registry_path, issue_date, claim, status="failed",
                                    finished_at=dt.datetime.now(KST), failure_type=type(exc).__name__)
            except Exception:
                pass
        raise


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == FINGERPRINT_MODE:
        probe_parser = argparse.ArgumentParser(
            description="Validate source fingerprint and signed DB reconciliation in the pinned runtime.")
        probe_parser.add_argument("mode")
        probe_parser.add_argument("--prices", type=Path, required=True)
        probe_parser.add_argument("--provenance", type=Path, required=True)
        probe_parser.add_argument("--prices-sha256", required=True)
        probe_parser.add_argument("--provenance-sha256", required=True)
        probe_parser.add_argument("--db", type=Path, required=True)
        args = probe_parser.parse_args(argv)
        try:
            result = _fingerprint_probe(args.prices, args.provenance,
                                        args.prices_sha256, args.provenance_sha256, args.db)
        except Exception as exc:
            print(f"fingerprint validation failed: {type(exc).__name__}", file=sys.stderr)
            return 2
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--issue-date", type=dt.date.fromisoformat, default=dt.datetime.now(KST).date())
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--status", type=Path, default=STATUS_PATH)
    parser.add_argument("--db", type=Path, default=ROOT / "data/agri.sqlite")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    target, created = run_issue(args.issue_date, args.prices, args.provenance, args.output_dir,
                                status_path=args.status, db_path=args.db)
    print(json.dumps({"forecast": str(target), "created": created}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
