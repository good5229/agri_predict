"""Frozen offline national-price model selection and validation.

Run development before fresh collection with ``build_manifest({"archivePath": ...},
start=2025-10-01, end=2026-06-30)`` then score that manifest against the same
archive. After fresh collection, build a Jul-Sep manifest with both
``archivePath`` and ``coverageReportPath`` and pass the hash-verified frozen
development score as ``selection_report`` to ``score_manifest``. Forecast
manifests contain no target values; scoring rebuilds the complete grid before
it attaches outcomes.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import direct_horizon, national_transfer, ten_strategy_benchmark

PRODUCTS = tuple(national_transfer.PRODUCT_CODES)
HORIZONS = tuple(range(1, 11))
RIDGE_ALPHAS = (10.0, 100.0)
BANDWIDTHS = (9, 28)
ALPHA = 0.05
MIN_PAIRS = 30
PROTOCOL_ID = "national-price-selection-2026-10-09"
DEV_WINDOWS = {
    "stability": (dt.date(2025, 10, 1), dt.date(2026, 3, 31)),
    "selection": (dt.date(2026, 4, 1), dt.date(2026, 6, 30)),
}
FRESH_WINDOW = (dt.date(2026, 7, 1), dt.date(2026, 9, 30))
METHODS = ("ridge_alpha_10", "ridge_alpha_100", "calendar_sarima", "fixed_blend_50_50")
TIE_ORDER = ("ridge_alpha_100", "ridge_alpha_10", "fixed_blend_50_50", "calendar_sarima")
SARIMA_SPEC = {"order": (1, 1, 1), "seasonal_order": (1, 0, 0, 7), "trend": "n"}
PROTOCOL = {
    "id": PROTOCOL_ID,
    "source": "parsed national_transfer archive observations; all signed-source provenance remains in source archive",
    "cutoff": "target minus horizon minus 2 calendar days; origin is target minus horizon",
    "ridge": "per product/horizon/origin-calendar-month StandardScaler + Ridge; alpha 10 or 100; expanding training targets at or before first monthly refit cutoff; refit monthly",
    "sarima": "per product/horizon/origin-calendar-month log-price calendar SARIMA(1,1,1)(1,0,0)[7], fixed specification, no additional differencing search; missing dates remain NaN; refit monthly",
    "fallback": "fit or forecast failure uses latest positive observed national price at/before cutoff, only if age <= 14 calendar days; native/fallback lineage retained",
    "selection": "strict common pairs across four methods; Apr-Jun 40-cell macro MAPE; qualify only with 40-cell equal-weight MAE <= 1.01 x Ridge alpha100 40-cell equal-weight MAE and total issued coverage >= Ridge alpha100; choose lowest MAPE, frozen tie order; any cell <30 pairs or missing -> unsupported, default Ridge alpha100",
    "stability": "Oct-Mar reports same metrics and inference but does not select",
    "inference": "development: all four candidates versus Ridge alpha100 and persistence; 4 x 40 cells x 2 comparators x 2 losses = 640 DM tests per bandwidth, separate Holm HAC9/HAC28 families; fresh holdout: frozen selected candidate versus Ridge alpha100 and persistence, 160 tests per bandwidth; fewer than 30 pairs or invalid HAC variance is inference_unavailable; Apr-Jun candidate inference is exploratory after selection",
    "market_state": "missing product-date is unknown_absent unless caller supplies explicit market-status evidence; never infer market closure from absence; observed row with unusable price retains its priceReason",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def _load_source(raw_bundle: dict) -> dict:
    """Rebuild the parsed target source from immutable raw inputs via its adapter."""
    if not isinstance(raw_bundle, dict) or set(raw_bundle) not in ({"archivePath"}, {"archivePath", "coverageReportPath"}):
        raise ValueError("raw_bundle must contain archivePath and optionally coverageReportPath")
    archive_path = Path(raw_bundle["archivePath"])
    archive_bytes = archive_path.read_bytes()
    if "coverageReportPath" not in raw_bundle:
        parsed = national_transfer.parse_archive(archive_bytes, expected_sha256=national_transfer.ARCHIVE_SHA256)
        parsed["sourceDigests"] = {"archiveSha256": _sha(archive_bytes),
            "combinedObservationsSha256": _canonical_hash(parsed["observations"])}
        return parsed
    report_path = Path(raw_bundle["coverageReportPath"])
    report_bytes = report_path.read_bytes()
    from . import nongnet_national_fresh_holdout as source_adapter
    parsed = source_adapter.build_combined_target_source(report_path, archive_path=archive_path)
    digests = parsed.get("sourceDigests")
    if parsed.get("schemaVersion") != 1 or not isinstance(digests, dict):
        raise ValueError("combined target source schema invalid")
    if digests.get("archiveSha256") != _sha(archive_bytes) or digests.get("archiveSha256") != national_transfer.ARCHIVE_SHA256:
        raise ValueError("source archive SHA-256 is not the pinned national archive")
    if digests.get("coverageReportSha256") != _sha(report_bytes):
        raise ValueError("coverage report SHA-256 mismatch")
    if digests.get("combinedObservationsSha256") != _canonical_hash(parsed.get("observations")):
        raise ValueError("combined observations SHA-256 mismatch")
    return parsed


def _positive(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def _prices(archive: dict) -> dict[str, dict[dt.date, float]]:
    obs = archive.get("observations")
    if not isinstance(obs, dict) or set(obs) != set(PRODUCTS):
        raise ValueError("archive must contain exactly the four national products")
    result: dict[str, dict[dt.date, float]] = {}
    for product in PRODUCTS:
        result[product] = {}
        if not isinstance(obs[product], dict):
            raise ValueError("archive product observations invalid")
        for day_text, row in obs[product].items():
            day = dt.date.fromisoformat(day_text)
            price = _positive(row.get("price"))
            if price is not None:
                result[product][day] = price
    return result


def _feature(series: dict[dt.date, float], target: dt.date, horizon: int) -> list[float] | None:
    cutoff = target - dt.timedelta(days=horizon + 2)
    observed = sorted((d, v) for d, v in series.items() if d <= cutoff)
    if not observed or (cutoff - observed[-1][0]).days > 14:
        return None
    last_day, last = observed[-1]
    logs = [(d, math.log(v)) for d, v in observed if (cutoff - d).days <= 56]
    vals = [math.log(last)]
    for window in (7, 14, 28, 56):
        window_values = [v for d, v in logs if (cutoff - d).days < window]
        vals.extend((float(np.mean(window_values)) - math.log(last) if window_values else 0.0,
                     float(np.std(window_values)) if len(window_values) > 1 else 0.0,
                     float(len(window_values))))
    vals.extend(((cutoff - last_day).days, float(horizon),
                 math.sin(2 * math.pi * target.weekday() / 7), math.cos(2 * math.pi * target.weekday() / 7),
                 math.sin(2 * math.pi * target.timetuple().tm_yday / 365.25),
                 math.cos(2 * math.pi * target.timetuple().tm_yday / 365.25)))
    return vals


def _ridge_fit(series: dict[dt.date, float], horizon: int, cutoff: dt.date, alpha: float):
    xs, ys, label_days = [], [], []
    for target, actual in sorted(series.items()):
        if target > cutoff:
            continue
        feat = _feature(series, target, horizon)
        if feat is not None:
            xs.append(feat)
            ys.append(math.log(actual))
            label_days.append(target)
    if len(xs) < 30:
        raise ValueError("fewer_than_30_training_pairs")
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(np.asarray(xs), np.asarray(ys))
    return model, len(xs), max(label_days)


def _sarima_fit(series: dict[dt.date, float], cutoff: dt.date):
    # Use exactly the fixed calendar fit used by the benchmark module, without
    # its validation-driven differencing search.
    return ten_strategy_benchmark._fit_calendar(series, min(series), cutoff, "sarima", 1)


def _sarima_predict(fit, horizon: int) -> float:
    pred = fit.get_forecast(steps=horizon + 2).predicted_mean
    log_value = float(pred.iloc[-1] if hasattr(pred, "iloc") else pred[-1])
    value = math.exp(log_value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("sarima_prediction_not_positive_finite")
    return value


def _forecast_one(series: dict[dt.date, float], product: str, target: dt.date, horizon: int,
                  states: dict, method: str) -> tuple[float | None, dict]:
    cutoff = target - dt.timedelta(days=horizon + 2)
    origin_month = (target - dt.timedelta(days=horizon)).strftime("%Y-%m")
    state_key = (product, origin_month) if method == "calendar_sarima" else (product, horizon, origin_month)
    state = states.setdefault(state_key, {})
    latest = max((d for d in series if d <= cutoff), default=None)
    fallback = series.get(latest) if latest and (cutoff - latest).days <= 14 else None
    try:
        if method.startswith("ridge_"):
            alpha = 10.0 if method == "ridge_alpha_10" else 100.0
            ridge_key = f"ridge_{int(alpha)}"
            if ridge_key not in state:
                state[ridge_key], state[ridge_key + "TrainingPairs"], state[ridge_key + "MaxLabelDate"] = _ridge_fit(series, horizon, cutoff, alpha)
                state[ridge_key + "FitCutoff"] = cutoff
            if state[ridge_key + "FitCutoff"] > cutoff:
                raise ValueError("ridge_fit_cutoff_after_forecast_cutoff")
            feat = _feature(series, target, horizon)
            if feat is None:
                raise ValueError("no_recent_asof_price_feature")
            value = math.exp(float(state[ridge_key].predict(np.asarray([feat]))[0]))
        elif method == "calendar_sarima":
            cached_failure = state.get("sarimaFailure")
            if cached_failure:
                raise RuntimeError("cached SARIMA fit/update failure")
            if "sarima" not in state:
                try:
                    state["sarima"] = _sarima_fit({d: v for d, v in series.items() if d <= cutoff}, cutoff)
                except Exception as exc:
                    state["sarimaFailure"] = type(exc).__name__ + ":" + str(exc)
                    raise
                state["fitCutoff"] = cutoff
                state["updatedThrough"] = cutoff
            elif cutoff > state["updatedThrough"]:
                try:
                    state["sarima"] = state["sarima"].append(
                        ten_strategy_benchmark._calendar_values(series, state["updatedThrough"] + dt.timedelta(days=1), cutoff),
                        refit=False)
                except Exception as exc:
                    state["sarimaFailure"] = type(exc).__name__ + ":" + str(exc)
                    raise
                state["updatedThrough"] = cutoff
            value = _sarima_predict(state["sarima"], horizon)
        else:
            raise ValueError("unknown_native_method")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("prediction_not_positive_finite")
        fit_date = state.get(ridge_key + "FitCutoff", cutoff) if method.startswith("ridge_") else state.get("fitCutoff", cutoff)
        if fit_date > cutoff:
            raise ValueError("model_fit_cutoff_after_forecast_cutoff")
        state_through = state.get("updatedThrough", fit_date)
        if state_through > cutoff:
            raise ValueError("model_state_after_forecast_cutoff")
        return value, {"nativeStatus": "available", "fallback": False, "fallbackReason": None,
                       "fitCutoff": fit_date.isoformat(),
                       "trainingPairs": state.get(ridge_key + "TrainingPairs") if method.startswith("ridge_") else None,
                       "maxTrainingTargetDate": state.get(ridge_key + "MaxLabelDate").isoformat() if method.startswith("ridge_") else None,
                       "stateThroughDate": state_through.isoformat() if method == "calendar_sarima" else None}
    except Exception as exc:
        reason = state.get("sarimaFailure") if method == "calendar_sarima" else None
        reason = reason or (type(exc).__name__ + ":" + str(exc))
        if fallback is None:
            return None, {"nativeStatus": "unavailable", "fallback": False, "fallbackReason": reason,
                          "fallbackPriceDate": latest.isoformat() if latest else None}
        return fallback, {"nativeStatus": "unavailable", "fallback": True, "fallbackReason": reason,
                          "fallbackPriceDate": latest.isoformat(), "fallbackValue": fallback}


def _build_manifest(archive: dict, source_digests: dict, *, start: dt.date, end: dt.date) -> dict:
    if end < start:
        raise ValueError("invalid target window")
    series = _prices(archive)
    states: dict = {}
    forecast_keys = []
    day = start
    while day <= end:
        for product in PRODUCTS:
            for horizon in HORIZONS:
                forecast_keys.append((product, day - dt.timedelta(days=horizon), day, horizon))
        day += dt.timedelta(days=1)
    # Process increasing origins so a monthly SARIMA state can append each
    # newly available calendar day once and serve every horizon at that origin.
    forecasts = []
    for product, origin, target, horizon in sorted(forecast_keys, key=lambda x: (x[0], x[1], x[3], x[2])):
        cutoff = origin - dt.timedelta(days=2)
        predictions: dict[str, float | None] = {}
        lineage = {}
        for method in METHODS[:3]:
            value, audit = _forecast_one(series[product], product, target, horizon, states, method)
            predictions[method] = value
            lineage[method] = audit
        ridge = predictions["ridge_alpha_100"]
        sarima = predictions["calendar_sarima"]
        latest = max((d for d in series[product] if d <= cutoff), default=None)
        persistence = (series[product][latest] if latest and (cutoff - latest).days <= 14 else None)
        blend_native = (ridge is not None and sarima is not None
            and not lineage["ridge_alpha_100"]["fallback"]
            and not lineage["calendar_sarima"]["fallback"])
        predictions["fixed_blend_50_50"] = ((ridge + sarima) / 2 if blend_native else persistence)
        predictions["persistence"] = persistence
        lineage["persistence"] = {"available": persistence is not None,
            "sourceDate": latest.isoformat() if latest and persistence is not None else None}
        lineage["fixed_blend_50_50"] = {"rule": "arithmetic 0.5/0.5 when both component predictions exist; latest valid as-of persistence if either is unavailable",
            "fallback": not blend_native,
            "fallbackReason": ({"ridge": lineage["ridge_alpha_100"]["fallbackReason"],
                "sarima": lineage["calendar_sarima"]["fallbackReason"]}
                if not blend_native else None), "fallbackPriceDate": latest.isoformat() if persistence is not None else None}
        forecasts.append({"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
            "horizon": horizon, "cutoff": cutoff.isoformat(), "predictions": predictions, "lineage": lineage})
    forecasts.sort(key=lambda r: (r["product"], r["target"], r["horizon"]))
    doc = {"schemaVersion": 1, "protocol": PROTOCOL, "phase": "forecast_only", "targetValuesAttached": False,
        "sourceDigests": source_digests, "sourceBundleSha256": _canonical_hash(source_digests),
        "codeSha256": _sha(Path(__file__).read_bytes()), "window": [start.isoformat(), end.isoformat()],
        "products": list(PRODUCTS), "horizons": list(HORIZONS), "candidateMethods": list(METHODS),
        "protocolDetails": PROTOCOL, "plannedForecastCount": len(forecasts), "forecasts": forecasts}
    doc["manifestSha256"] = _canonical_hash(doc)
    return doc


def build_manifest(raw_bundle: dict, *, start: dt.date, end: dt.date) -> dict:
    """Build answer-free forecasts from the validated immutable raw bundle."""
    source = _load_source(raw_bundle)
    return _build_manifest(source, source["sourceDigests"], start=start, end=end)


def _verify_manifest(manifest: dict) -> None:
    if manifest.get("protocol") != PROTOCOL or manifest.get("phase") != "forecast_only" or manifest.get("targetValuesAttached") is not False:
        raise ValueError("manifest is not a forecast-only national selection manifest")
    if any("actual" in r or "targetValue" in r for r in manifest.get("forecasts", [])):
        raise ValueError("answer-bearing field in forecast manifest")
    code_sha = _sha(Path(__file__).read_bytes())
    if manifest.get("codeSha256") != code_sha:
        raise ValueError("manifest code SHA-256 differs from current implementation")
    if manifest.get("sourceBundleSha256") != _canonical_hash(manifest.get("sourceDigests")):
        raise ValueError("manifest source bundle digest mismatch")
    saved = manifest.get("manifestSha256")
    copy = dict(manifest)
    copy.pop("manifestSha256", None)
    if saved != _canonical_hash(copy):
        raise ValueError("manifest hash mismatch")


def _status(archive: dict, product: str, day: dt.date, explicit_status: dict | None) -> tuple[str, dict | None]:
    row = archive["observations"][product].get(day.isoformat())
    if row is not None:
        if _positive(row.get("price")) is not None:
            return "observed_trade", row
        return "observed_unusable_price:" + str(row.get("priceReason") or "unknown_reason"), row
    if explicit_status:
        status = explicit_status.get((product, day.isoformat()))
        if status in ("market_closed", "no_trade", "normal_empty", "empty_response", "no_trade_unconfirmed",
                      "api_error", "conflicting_responses", "questionable_observed", "requested_date_missing",
                      "not_queried", "source_failure"):
            return status, None
    return "unknown_absent", None


def _pair_rows(manifest: dict, archive: dict, status_evidence: dict | None = None) -> tuple[list[dict], dict]:
    _verify_manifest(manifest)
    rows, exclusions = [], defaultdict(int)
    seen = set()
    for f in manifest["forecasts"]:
        key = (f["product"], f["origin"], f["target"], f["horizon"])
        if key in seen:
            raise ValueError("duplicate forecast identity")
        seen.add(key)
        target = dt.date.fromisoformat(f["target"])
        status, observed = _status(archive, f["product"], target, status_evidence)
        actual = _positive(observed.get("price")) if observed else None
        if actual is None:
            exclusions[status] += 1
            continue
        if any(_positive(f["predictions"].get(m)) is None for m in METHODS):
            exclusions["incomplete_common_candidate_forecast"] += 1
            continue
        rows.append({**{k: f[k] for k in ("product", "origin", "target", "horizon", "cutoff")},
                     "actual": actual, "marketStatus": status,
                     "predictions": {m: _positive(f["predictions"].get(m)) for m in (*METHODS, "persistence")}})
    return rows, dict(exclusions)


def _cell_metrics(rows: list[dict], methods=METHODS) -> dict:
    out = {}
    for method in methods:
        metrics = direct_horizon._metrics([r["actual"] for r in rows], [r["predictions"][method] for r in rows])
        out[method] = metrics
    return out


def _dm_family(rows: list[dict], candidate: str | tuple[str, ...], comparators=("ridge_alpha_100", "persistence")) -> list[dict]:
    summaries = []
    candidates = (candidate,) if isinstance(candidate, str) else tuple(candidate)
    for product in PRODUCTS:
        for horizon in HORIZONS:
            cell = [r for r in rows if r["product"] == product and r["horizon"] == horizon]
            for method in candidates:
                for comparator in comparators:
                    for loss in ("mae", "mape"):
                        diffs = {}
                        paired = [r for r in cell if _positive(r["predictions"].get(method)) is not None
                                  and _positive(r["predictions"].get(comparator)) is not None]
                        for r in paired:
                            y, a, b = r["actual"], r["predictions"][method], r["predictions"][comparator]
                            diffs[dt.date.fromisoformat(r["target"])] = (abs(y-a)-abs(y-b) if loss == "mae"
                                else 100.0 * (abs(y-a)-abs(y-b)) / y)
                        item = {"product": product, "horizon": horizon, "candidate": method,
                                "comparator": comparator, "loss": loss, "n": len(diffs),
                                "candidateMetric": _cell_metrics(paired, (method,))[method][loss],
                                "comparatorMetric": _cell_metrics(paired, (comparator,))[comparator][loss],
                                "dm": {str(bw): direct_horizon._dm(diffs, bw) for bw in BANDWIDTHS}}
                        summaries.append(item)
    for bw in BANDWIDTHS:
        family = [r["dm"][str(bw)] for r in summaries]
        if len(family) != len(PRODUCTS) * len(HORIZONS) * len(candidates) * len(comparators) * 2:
            raise AssertionError("frozen national DM family size mismatch")
        direct_horizon._holm(family)
        for row, result in zip(summaries, family):
            mean = result.get("meanLossDifference")
            result["conclusion"] = ("improvement" if result.get("status") == "tested" and mean is not None and mean < 0 and result.get("holmSignificant")
                                     else "worsening" if result.get("status") == "tested" and mean is not None and mean > 0 and result.get("holmSignificant")
                                     else "inconclusive" if result.get("status") == "tested" else "inference_unavailable")
    return summaries


def _joint_cell_verdicts(results: list[dict]) -> list[dict]:
    indexed = {(r["product"], r["horizon"], r["candidate"], r["comparator"], r["loss"]): r for r in results}
    cells = []
    identities = sorted({(r["product"], r["horizon"], r["candidate"], r["comparator"]) for r in results})
    for product, horizon, candidate, comparator in identities:
        ae = indexed[(product, horizon, candidate, comparator, "mae")]
        ape = indexed[(product, horizon, candidate, comparator, "mape")]
        cell = {"product": product, "horizon": horizon, "candidate": candidate, "comparator": comparator,
                "nMae": ae["n"], "nMape": ape["n"], "verdictByBandwidth": {}}
        for bw in BANDWIDTHS:
            a, p = ae["dm"][str(bw)], ape["dm"][str(bw)]
            ad, pd = a["conclusion"], p["conclusion"]
            unavailable = ad == "inference_unavailable" or pd == "inference_unavailable"
            joint = (ad if ad == pd and ad in ("improvement", "worsening") else
                     "inference_unavailable" if unavailable else "inconclusive")
            cell["verdictByBandwidth"][str(bw)] = {"mae": ad, "mape": pd, "joint": joint,
                "directionPattern": f"mae={ad};mape={pd}",
                "mixedDirection": ad in ("improvement", "worsening") and pd in ("improvement", "worsening") and ad != pd}
        cells.append(cell)
    return cells


def _macro_40(rows: list[dict]) -> dict:
    cells = {}
    for product in PRODUCTS:
        for horizon in HORIZONS:
            cell = [r for r in rows if r["product"] == product and r["horizon"] == horizon]
            cells[f"{product}/{horizon}"] = {"n": len(cell), **_cell_metrics(cell)}
    complete = all(v["n"] >= MIN_PAIRS and all(v[m]["mape"] is not None for m in METHODS) for v in cells.values())
    macro = {m: (sum(cells[k][m]["mape"] for k in cells) / 40 if complete else None) for m in METHODS}
    macro_mae = {m: (sum(cells[k][m]["mae"] for k in cells) / 40 if complete else None) for m in METHODS}
    return {"complete40Cells": complete, "minimumPairsPerCell": MIN_PAIRS,
            "macroMape": macro, "macroMae": macro_mae, "cells": cells}


def _forecast_coverage(manifest: dict, start: dt.date, end: dt.date) -> dict:
    methods = (*METHODS, "persistence")
    by_crop = {m: {p: {"planned": 0, "issued": 0, "native": 0, "fallback": 0} for p in PRODUCTS} for m in methods}
    for row in manifest["forecasts"]:
        target = dt.date.fromisoformat(row["target"])
        if not start <= target <= end:
            continue
        product = row["product"]
        for method in methods:
            item = by_crop[method][product]
            item["planned"] += 1
            issued = _positive(row["predictions"].get(method)) is not None
            if issued:
                item["issued"] += 1
            audit = row.get("lineage", {}).get(method, {})
            fallback = bool(audit.get("fallback")) and issued
            native = issued and not fallback and (audit.get("nativeStatus") == "available" or audit.get("available") is True or method == "fixed_blend_50_50")
            item["native"] += int(native)
            item["fallback"] += int(fallback)
    total = {}
    for method in methods:
        rows = list(by_crop[method].values())
        agg = {key: sum(r[key] for r in rows) for key in ("planned", "issued", "native", "fallback")}
        agg["issuedRate"] = agg["issued"] / agg["planned"] if agg["planned"] else None
        total[method] = agg
    return {"total": total, "byCrop": by_crop}


def _per_crop_mape(rows: list[dict], candidate: str) -> dict:
    output = {}
    for product in PRODUCTS:
        cell_values = {}
        for horizon in HORIZONS:
            cell = [r for r in rows if r["product"] == product and r["horizon"] == horizon]
            metric = _cell_metrics(cell, (candidate, "ridge_alpha_100"))
            cell_values[str(horizon)] = {"n": len(cell), "selectedMape": metric[candidate]["mape"],
                "ridgeAlpha100Mape": metric["ridge_alpha_100"]["mape"],
                "changePp": (metric[candidate]["mape"] - metric["ridge_alpha_100"]["mape"]
                             if metric[candidate]["mape"] is not None and metric["ridge_alpha_100"]["mape"] is not None else None)}
        valid = [v for v in cell_values.values() if v["changePp"] is not None]
        output[product] = {"byHorizon": cell_values,
            "selectedMacroMape": (sum(v["selectedMape"] for v in valid) / len(valid) if valid else None),
            "ridgeAlpha100MacroMape": (sum(v["ridgeAlpha100Mape"] for v in valid) / len(valid) if valid else None),
            "changePp": (sum(v["changePp"] for v in valid) / len(valid) if valid else None),
            "horizonsWithMetrics": len(valid)}
    return output


def _fresh_gate(metrics: dict, coverage: dict, crop_mape: dict, joint_cells: list[dict], candidate: str) -> dict:
    complete = metrics["complete40Cells"]
    selected_mape = metrics["macroMape"].get(candidate)
    baseline_mape = metrics["macroMape"].get("ridge_alpha_100")
    selected_mae = metrics["macroMae"].get(candidate)
    baseline_mae = metrics["macroMae"].get("ridge_alpha_100")
    mape_gain = baseline_mape - selected_mape if selected_mape is not None and baseline_mape is not None else None
    mae_change_pct = 100 * (selected_mae - baseline_mae) / baseline_mae if selected_mae is not None and baseline_mae not in (None, 0) else None
    crop_changes = {p: crop_mape[p]["changePp"] for p in PRODUCTS}
    improved = [r for r in joint_cells if r["candidate"] == candidate and r["comparator"] == "ridge_alpha_100"
                and r["verdictByBandwidth"]["9"]["joint"] == "improvement"
                and r["verdictByBandwidth"]["28"]["joint"] == "improvement"]
    issued_selected = coverage["total"][candidate]["issued"]
    issued_baseline = coverage["total"]["ridge_alpha_100"]["issued"]
    criteria = {
        "mapeImprovementAtLeast0_5pp": mape_gain is not None and mape_gain >= 0.5,
        "issuedCoverageNotDecreased": issued_selected >= issued_baseline,
        "eachCropMapeWorseningAtMost1pp": all(v is not None and v <= 1.0 for v in crop_changes.values()),
        "macroMaeWorseningAtMost5pct": mae_change_pct is not None and mae_change_pct <= 5.0,
        "atLeastOneCellJointlyImprovedAtHac9AndHac28": len(improved) >= 1,
    }
    return {"status": ("unsupported" if not complete else "pass" if all(criteria.values()) else "fail"),
        "candidate": candidate, "criteria": criteria, "thresholds": {"macroMapeGainPp": 0.5,
            "perCropMapeWorseningPpMaximum": 1.0, "macroMaeWorseningPctMaximum": 5.0,
            "jointImprovementBandwidthDays": [9, 28], "jointImprovementMinimumCells": 1},
        "macroMapeGainPp": mape_gain, "macroMaeWorseningPct": mae_change_pct,
        "candidateIssued": issued_selected, "ridgeAlpha100Issued": issued_baseline,
        "cropMapeChangePp": crop_changes, "jointImprovedCellsHac9": len(improved),
        "complete40Cells": complete}


def _select(selection_rows: list[dict], expected_coverage: dict[str, dict[str, int]]) -> dict:
    report = _macro_40(selection_rows)
    macro_mae = report["macroMae"]
    issued_total = {m: sum(expected_coverage.get(m, {}).values()) for m in expected_coverage}
    eligible = []
    if report["complete40Cells"]:
        for method in METHODS:
            if (macro_mae[method] <= 1.01 * macro_mae["ridge_alpha_100"]
                    and issued_total.get(method, 0) >= issued_total.get("ridge_alpha_100", 0)):
                eligible.append(method)
    if not report["complete40Cells"] or not eligible:
        selected, status = "ridge_alpha_100", "unsupported_default_ridge_alpha_100"
    else:
        selected = min(eligible, key=lambda m: (report["macroMape"][m], TIE_ORDER.index(m)))
        status = "selected"
    return {"status": status, "selected": selected, "eligible": eligible,
            "qualification": "40-cell equal-weight MAE <= 1.01 x Ridge alpha100 40-cell equal-weight MAE and total issued coverage >= Ridge alpha100",
            "macroMae": macro_mae, "issuedCoverageTotal": issued_total,
            **report}


def _coverage_status_evidence(raw_bundle: dict) -> dict:
    if "coverageReportPath" not in raw_bundle:
        return {}
    report = json.loads(Path(raw_bundle["coverageReportPath"]).read_text(encoding="utf-8"))
    crop_map = {"cabbage": "cabbage", "radish": "radish", "onion": "onion", "pepper": "pepper"}
    return {(crop_map[row["category"]], row["date"]): row["state"]
            for row in report.get("coverage", []) if row.get("category") in crop_map
            and row.get("state") != "observed"}


def score_manifest(manifest: dict, raw_bundle: dict, *, status_evidence: dict | None = None,
                   selection_report: dict | None = None) -> dict:
    """Attach outcomes separately; fresh-window scoring requires frozen selection."""
    _verify_manifest(manifest)
    source = _load_source(raw_bundle)
    source_digests = source["sourceDigests"]
    if manifest.get("sourceDigests") != source_digests:
        raise ValueError("manifest source digests do not match the supplied raw bundle")
    start, end = (dt.date.fromisoformat(x) for x in manifest["window"])
    rebuilt = _build_manifest(source, source_digests, start=start, end=end)
    if manifest.get("forecasts") != rebuilt.get("forecasts"):
        raise ValueError("forecast-only rows do not rebuild from supplied raw bundle")
    if status_evidence is None:
        status_evidence = _coverage_status_evidence(raw_bundle)
    pairs, exclusions = _pair_rows(manifest, source, status_evidence)
    outcomes = {}
    for name, (a, b) in DEV_WINDOWS.items():
        lo, hi = max(start, a), min(end, b)
        if lo <= hi:
            outcomes[name] = [r for r in pairs if lo <= dt.date.fromisoformat(r["target"]) <= hi]
    report = {"schemaVersion": 1, "protocol": PROTOCOL, "manifestSha256": manifest["manifestSha256"],
        "sourceDigests": source_digests, "exclusionCounts": exclusions, "commonScoredPairs": len(pairs),
        "marketStatusCounts": dict(sorted((k, sum(r["marketStatus"] == k for r in pairs)) for k in {r["marketStatus"] for r in pairs})),
        "development": {}, "selection": None}
    window_lo, window_hi = dt.date.fromisoformat(manifest["window"][0]), dt.date.fromisoformat(manifest["window"][1])
    overlaps_fresh = window_lo <= FRESH_WINDOW[1] and window_hi >= FRESH_WINDOW[0]
    frozen_candidate = None
    if selection_report is not None:
        if selection_report.get("protocol") != PROTOCOL or not isinstance(selection_report.get("selection"), dict):
            raise ValueError("selection report protocol or selection missing")
        digest = selection_report.get("reportSha256")
        body = dict(selection_report)
        body.pop("reportSha256", None)
        if digest != _canonical_hash(body):
            raise ValueError("selection report hash mismatch")
        selection_window = selection_report.get("development", {}).get("selection", {})
        if (selection_report.get("sourceDigests", {}).get("archiveSha256") != source_digests.get("archiveSha256")
                or selection_window.get("start") != DEV_WINDOWS["selection"][0].isoformat()
                or selection_window.get("end") != DEV_WINDOWS["selection"][1].isoformat()):
            raise ValueError("selection report is not bound to the pre-July archive and Apr-Jun window")
        frozen_candidate = selection_report["selection"].get("selected")
        if frozen_candidate not in METHODS:
            raise ValueError("selection report candidate invalid")
    if overlaps_fresh and frozen_candidate is None:
        raise ValueError("fresh holdout scoring requires the frozen pre-July selection report")
    coverage_by_window: dict[str, dict[str, dict[str, int]]] = {}
    for name, rows in outcomes.items():
        metrics = _macro_40(rows)
        report["development"][name] = {"start": max(start, DEV_WINDOWS[name][0]).isoformat(),
            "end": min(end, DEV_WINDOWS[name][1]).isoformat(), "commonPairs": len(rows), "macro40": metrics}
        truth_keys = set()
        lo, hi = max(start, DEV_WINDOWS[name][0]), min(end, DEV_WINDOWS[name][1])
        for f in manifest["forecasts"]:
            target = dt.date.fromisoformat(f["target"])
            if lo <= target <= hi:
                status, observed = _status(source, f["product"], target, status_evidence)
                if observed is not None and _positive(observed.get("price")) is not None:
                    truth_keys.add((f["product"], f["target"], f["horizon"]))
        raw_counts = {method: defaultdict(int) for method in (*METHODS, "persistence")}
        for f in manifest["forecasts"]:
            target = dt.date.fromisoformat(f["target"])
            if not (max(start, DEV_WINDOWS[name][0]) <= target <= min(end, DEV_WINDOWS[name][1])):
                continue
            key = (f["product"], f["target"], f["horizon"])
            if key not in truth_keys:
                continue
            cell_key = f"{f['product']}/{f['horizon']}"
            for method in (*METHODS, "persistence"):
                if _positive(f["predictions"].get(method)) is not None:
                    raw_counts[method][cell_key] += 1
        coverage_by_window[name] = {m: dict(v) for m, v in raw_counts.items()}
        report["development"][name]["issuedCoverageByCell"] = coverage_by_window[name]
        if name == "selection":
            report["selection"] = _select(rows, coverage_by_window[name])
    selected = report["selection"]["selected"] if report["selection"] else "ridge_alpha_100"
    for name, rows in outcomes.items():
        development_tests = _dm_family(rows, METHODS)
        report["development"][name]["comparisonsAllCandidatesVsRidgeAndPersistence"] = development_tests
        report["development"][name]["jointCellVerdictsAllCandidates"] = _joint_cell_verdicts(development_tests)
        report["development"][name]["holmFamilySizePerBandwidth"] = len(development_tests)
        report["development"][name]["inferenceInterpretation"] = (
            "all candidate comparisons are exploratory selection diagnostics; the Apr-Jun selected method was chosen on that window"
            if name == "selection" else "all candidate comparisons are stability diagnostics, not selection inputs")
    if overlaps_fresh:
        fresh_rows = [r for r in pairs if FRESH_WINDOW[0] <= dt.date.fromisoformat(r["target"]) <= FRESH_WINDOW[1]]
        fresh_metrics = _macro_40(fresh_rows)
        fresh_tests = _dm_family(fresh_rows, frozen_candidate)
        fresh_joint = _joint_cell_verdicts(fresh_tests)
        coverage = _forecast_coverage(manifest, FRESH_WINDOW[0], FRESH_WINDOW[1])
        crop_mape = _per_crop_mape(fresh_rows, frozen_candidate)
        report["freshHoldout"] = {"start": FRESH_WINDOW[0].isoformat(), "end": FRESH_WINDOW[1].isoformat(),
            "selectedCandidate": frozen_candidate, "commonPairs": len(fresh_rows), "macro40": fresh_metrics,
            "coverage": coverage, "perCropMape": crop_mape,
            "comparisonsSelectedVsRidgeAndPersistence": fresh_tests, "jointCellVerdicts": fresh_joint,
            "holmFamilySizePerBandwidth": len(fresh_tests),
            "overallGate": _fresh_gate(fresh_metrics, coverage, crop_mape, fresh_joint, frozen_candidate),
            "inferenceInterpretation": "unseen holdout; selected candidate was frozen using only the pre-July selection report"}
    report["limitations"] = ["Absence of a national product-date is unknown_absent without explicit status evidence; market closure is never inferred.",
        "The archive is a revised retrospective source; first publication times are unknown."]
    report["reportSha256"] = _canonical_hash(report)
    return report
