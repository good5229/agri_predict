"""Four-crop causal50 forecast, retrospective paired evaluation, and proposal page.

This consumes frozen local price/forecast files only. It does not collect data,
train the held-out evaluation rows, or claim that the retrospective comparison is
an independent confirmation.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from . import equal_info_online_forecasts as online
from . import equal_info_online_score as online_score

ROOT = Path(__file__).resolve().parents[1]
LIVE_PRICES = ROOT / "outputs/daily/2026-10-09/20261009T120010-26aa2047/prices.json"
LIVE_PRICES_SHA256 = "3e9eca5e1da69812fb3e82894dc1c143af1bc752a33740cc03850d1439792691"
LIVE_RUN_ID = "20261009T120010-26aa2047"
LIVE_CUTOFF = dt.date(2026, 10, 8)
LIVE_SNAPSHOT_RECORD_SHA256 = "82f04b382636d0af4fc380733b549eb6a6ee214eccfdb30c5b995fb24a56d22a"
HISTORICAL_MANIFEST = ROOT / "outputs/experiments/equal-info-online-20261010/forecast-manifest.json"
HISTORICAL_MANIFEST_SHA256 = "d7d74823754c2a48d33cc4ddaeab674c686601fb927d399d84c5dbcd235184df"
HISTORICAL_PRICES_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
OUT_DIR = ROOT / "outputs/experiments/four-crop-forecast-20261010"
PRODUCTS = ("cabbage", "radish", "onion", "garlic")
HORIZONS = tuple(range(1, 11))
METHODS = ("causal50", "causalDirect", "causalSarima", "persistence")
COMPARISONS = (("causal50_vs_causalDirect", "causal50", "causalDirect"),
               ("causal50_vs_causalSarima", "causal50", "causalSarima"),
               ("causal50_vs_persistence", "causal50", "persistence"))
BANDWIDTHS = (9, 28)
EVAL_START, EVAL_END = online_score.EVAL_START, online_score.EVAL_END
NAMES = {"cabbage": "배추", "radish": "무", "onion": "양파", "garlic": "깐마늘"}
DISPLAY_METHODS = {"causal50": "혼합 예측", "causalDirect": "Ridge", "causalSarima": "SARIMA", "persistence": "직전가격 유지"}
DISPLAY_STATUSES = {"source_available": "자료 사용 가능", "source_older_than_14_days": "최신 자료가 14일 초과", "no_eligible_source": "사용 가능한 자료 없음"}


def _canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _pretty(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _positive(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _date(value):
    try:
        return dt.date.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _method_metrics(pairs, method):
    if not pairs:
        return {"n": 0, "mae": None, "mape": None}
    errors = [abs(row[method] - row["actual"]) for row in pairs]
    return {"n": len(pairs), "mae": math.fsum(errors) / len(errors),
            "mape": 100 * math.fsum(error / row["actual"] for error, row in zip(errors, pairs)) / len(errors)}


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_snapshot(raw, raw_bytes, path: Path):
    file_sha = _sha_bytes(raw_bytes)
    if raw.get("schemaVersion") != 1 or not isinstance(raw.get("products"), list):
        raise ValueError("price source must use dashboard schemaVersion 1")
    cutoff = _date(raw.get("meta", {}).get("evaluationEnd"))
    if cutoff is None:
        raise ValueError("price snapshot has no declared cutoff")
    if cutoff != LIVE_CUTOFF:
        raise ValueError("frozen latest-success snapshot cutoff changed")
    for product in raw["products"]:
        if not isinstance(product, dict) or not isinstance(product.get("observations"), list):
            raise ValueError("malformed product observations")
        for row in product["observations"]:
            day = _date(row.get("date")) if isinstance(row, dict) else None
            if day is not None and day > cutoff:
                raise ValueError(f"observation after declared source cutoff: {product.get('id')}/{day}")
    if file_sha != LIVE_PRICES_SHA256:
        raise ValueError("frozen latest-success prices.json file hash changed")
    records, _audit = online._records(raw)
    direct_series = online.direct_horizon._observations(raw)
    return cutoff, file_sha, records, direct_series


def _snapshot_status(record, cutoff, issue_date, source_path, source_sha):
    source_date, age = online._latest_pre_origin_source(record, issue_date)
    try:
        display_path = str(source_path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        display_path = str(source_path.resolve())
    return {"lastObservedPriceDate": source_date.isoformat() if source_date else None,
            "sourceAgeDaysAtIssue": age,
            "sourceCutoff": cutoff.isoformat(),
            "noObservationDaysAreImputed": True,
            "status": ("no_eligible_source" if source_date is None else
                       "source_older_than_14_days" if age > 14 else "source_available"),
            "sourcePath": display_path, "pricesFileSha256": source_sha}


def _ridge_training_dates(series, horizon, fit_origin, expected_audit):
    """Reconstruct the direct-Ridge row identities with its unchanged selector."""
    cache, targets, input_dates = {}, [], []
    for target, (actual, _volume) in sorted(series.items()):
        if target >= fit_origin or target < online.narrow.TRAIN_START:
            continue
        found = online.modeling._feature_row(
            series, target - dt.timedelta(days=horizon + 1), target, horizon, cache)
        if found is None:
            continue
        source, _vector = found
        if (target - dt.timedelta(days=horizon) - source).days > 14:
            continue
        targets.append(target)
        input_dates.append(source)
    audit = {"count": len(targets), "firstTarget": targets[0].isoformat() if targets else None,
             "lastTarget": targets[-1].isoformat() if targets else None,
             "fitOrigin": fit_origin.isoformat()}
    if audit != expected_audit:
        raise ValueError("saved Ridge state training-date reconstruction differs from fitted audit")
    return targets, input_dates


def _forecast_crop(records, direct_series, crop, issue_date, source_cutoff, progress=None, *, state_sink=None):
    fit_origin = issue_date.replace(day=1)
    if fit_origin > issue_date:
        raise ValueError("fit origin cannot be after issue date")
    source_records = records[crop]
    series = direct_series[crop]
    ordered = sorted(source_records)
    price_map = {day: row["price"] for day, row in source_records.items()}
    rows_by_h = {}
    for horizon in HORIZONS:
        model, fit_audit, fit_error = online._period_fit(records, direct_series, crop, horizon,
                                                         fit_origin, 10.0, "direct")
        if state_sink is not None:
            if model is None:
                state_sink({"kind": "ridge", "product": crop, "horizon": horizon,
                            "modelId": "direct40",
                            "fitOrigin": fit_origin, "status": "unavailable", "reason": fit_error,
                            "trainingAudit": fit_audit, "featureNames": list(online.modeling.FEATURE_NAMES)})
            else:
                targets, input_dates = _ridge_training_dates(series, horizon, fit_origin, fit_audit)
                state_sink({"kind": "ridge", "model": model, "product": crop, "horizon": horizon,
                            "modelId": "direct40",
                            "fitOrigin": fit_origin, "alpha": 10.0, "trainingTargets": targets,
                            "trainingInputDates": input_dates, "trainingAudit": fit_audit,
                            "featureNames": list(online.modeling.FEATURE_NAMES)})
        if progress:
            progress({"event": "ridge_fit", "crop": crop, "horizon": horizon,
                      "samples": fit_audit.get("count", 0), "status": "fitted" if model is not None and not fit_error else fit_error})
        rows_by_h[horizon] = (model, fit_audit, fit_error)

    sarima_fit, sarima_audit, sarima_error = None, {}, None
    try:
        sarima_fit, sarima_audit = online._fit_sarima(price_map, fit_origin)
    except Exception as exc:
        sarima_error = f"sarima_fit_failed:{type(exc).__name__}:{exc}"
        if state_sink is not None:
            state_sink({"kind": "sarima", "event": "fit_failed", "product": crop,
                        "fitOrigin": fit_origin, "reason": sarima_error})
    else:
        if state_sink is not None:
            state_sink({"kind": "sarima", "event": "pre_update_checkpoint", "product": crop,
                        "fitOrigin": fit_origin, "fit": sarima_fit, "fitAudit": sarima_audit,
                        "calendarDate": fit_origin - dt.timedelta(days=1)})
    if progress:
        progress({"event": "sarima_fit", "crop": crop,
                  "status": "fitted" if sarima_fit is not None else sarima_error})
    state_day = fit_origin - dt.timedelta(days=1) if sarima_fit is not None else None
    state_error = None
    if sarima_fit is not None:
        # Append every calendar day from the monthly fit through issue-1.
        # _calendar_values emits NaN for absent dates, including Oct 9 here.
        while state_day is not None and state_day < issue_date - dt.timedelta(days=1):
            next_day = state_day + dt.timedelta(days=1)
            try:
                update = online.ten._calendar_values(price_map, next_day, next_day)
                updated_fit = sarima_fit.append(update, refit=False)
            except Exception as exc:
                state_error = f"sarima_state_update_failed:{type(exc).__name__}:{exc}"
                if state_sink is not None:
                    state_sink({"kind": "sarima", "event": "state_update_failed", "product": crop,
                                "fitOrigin": fit_origin, "calendarDate": next_day, "reason": state_error})
                break
            sarima_fit = updated_fit
            state_day = next_day
            if state_sink is not None:
                state_sink({"kind": "sarima", "event": "state_update", "product": crop,
                            "fitOrigin": fit_origin, "calendarDate": next_day,
                            "update": update, "fit": sarima_fit})

    feature_cache = {}
    source_day, source_age = online._latest_pre_origin_source(source_records, issue_date)
    rows = []
    for horizon in HORIZONS:
        target = issue_date + dt.timedelta(days=horizon)
        origin = target - dt.timedelta(days=horizon)
        if origin != issue_date:
            raise AssertionError("all target horizons must share the issue-date origin")
        vector, p0, feature_audit = online._direct_feature(series, target, horizon, feature_cache)
        direct_model, fit_audit, fit_error = rows_by_h[horizon]
        direct_prediction = None
        direct_reason = fit_error or feature_audit.get("reason")
        if direct_model is not None and not fit_error and vector is not None and p0 is not None:
            try:
                prediction = p0 * math.exp(float(direct_model.predict([vector])[0]))
                if math.isfinite(prediction) and prediction > 0:
                    direct_prediction, direct_reason = prediction, None
                else:
                    direct_reason = "nonpositive_or_nonfinite_prediction"
            except Exception as exc:
                direct_reason = f"prediction_failed:{type(exc).__name__}:{exc}"

        sarima_prediction, sarima_reason = None, sarima_error or state_error
        if source_day is None:
            sarima_reason = sarima_reason or "no_source_observation"
        elif source_age > 14:
            sarima_reason = sarima_reason or "latest_source_older_than_14_days"
        elif sarima_fit is not None and not state_error:
            try:
                mean = sarima_fit.get_forecast(steps=horizon + 1).predicted_mean
                log_prediction = float(mean.iloc[-1] if hasattr(mean, "iloc") else mean[-1])
                candidate = math.exp(log_prediction)
                if math.isfinite(candidate) and candidate > 0:
                    sarima_prediction, sarima_reason = candidate, None
                else:
                    sarima_reason = "invalid_sarima_prediction"
            except Exception as exc:
                sarima_reason = f"sarima_forecast_failed:{type(exc).__name__}:{exc}"
        causal50 = None
        fallback_reason = None
        if direct_prediction is not None:
            causal50 = direct_prediction if sarima_prediction is None else .5 * (direct_prediction + sarima_prediction)
            if sarima_prediction is None:
                fallback_reason = sarima_reason or "sarima_unavailable_direct_only_fallback"
        persistence = p0 if _positive(p0) is not None else None
        methods = {
            "causal50": _prediction(causal50, None if causal50 is not None else direct_reason or sarima_reason,
                                     fallback=fallback_reason is not None, fallbackReason=fallback_reason),
            "causalDirect": _prediction(direct_prediction, direct_reason,
                                          training={**fit_audit, "alpha": 10.0,
                                                    "featureCount": len(online.modeling.FEATURE_NAMES),
                                                    "fitOrigin": fit_origin.isoformat()},
                                          provenance={"featureAudit": feature_audit}),
            "causalSarima": _prediction(sarima_prediction, sarima_reason,
                                          training={**sarima_audit, "fitOrigin": fit_origin.isoformat(),
                                                    "stateUpdatedThrough": state_day.isoformat() if state_day else None},
                                          provenance={"lastActualDate": source_day.isoformat() if source_day else None,
                                                      "requestedCutoff": (issue_date-dt.timedelta(days=1)).isoformat()}),
            "persistence": _prediction(persistence, None if persistence is not None else feature_audit.get("reason"),
                                        provenance={"sourceDate": feature_audit.get("p0Date"),
                                                    "sourceAgeDays": feature_audit.get("p0AgeDays")}),
        }
        rows.append({"product": crop, "productNameKo": NAMES[crop],
                     "target": target.isoformat(), "origin": issue_date.isoformat(),
                     "horizon": horizon, "eligible": True,
                     "sourceCutoff": source_cutoff.isoformat(), "fitOrigin": fit_origin.isoformat(),
                     "models": methods})
    return rows


def _prediction(value, reason=None, **extra):
    value = _positive(value)
    if value is None:
        return {"status": "not_issued", "prediction": None, "reason": reason or "model_unavailable", **extra}
    return {"status": "issued", "prediction": value, "reason": None, **extra}


def build_future(prices_path: Path, issue_date: dt.date, *, progress=None):
    if issue_date != dt.date(2026, 10, 10):
        raise ValueError("this frozen demonstration is authorized for issue date 2026-10-10")
    raw_bytes = prices_path.read_bytes()
    raw = json.loads(raw_bytes)
    cutoff, prices_sha, records, direct_series = _validate_snapshot(raw, raw_bytes, prices_path)
    if cutoff > issue_date - dt.timedelta(days=1):
        raise ValueError("price snapshot includes observations after issue-date cutoff")
    # Verify existing source bindings without rebuilding its historical forecasts.
    online._check_inputs()
    rows = []
    for crop in PRODUCTS:
        rows.extend(_forecast_crop(records, direct_series, crop, issue_date, cutoff, progress))
    rows.sort(key=lambda row: (PRODUCTS.index(row["product"]), row["horizon"]))
    if len(rows) != 40:
        raise ValueError(f"expected 40 planned future keys, got {len(rows)}")
    issued_at = dt.datetime.now(ZoneInfo("Asia/Seoul")).isoformat(timespec="seconds")
    result = {"schemaVersion": 1, "phase": "forecast_only", "targetValuesAttached": False,
            "issuedAtKst": issued_at, "issueDate": issue_date.isoformat(),
            "targetDateRange": [(issue_date + dt.timedelta(days=1)).isoformat(),
                                (issue_date + dt.timedelta(days=10)).isoformat()],
            "products": list(PRODUCTS), "horizons": list(HORIZONS), "plannedCount": 40,
            "source": {"runId": LIVE_RUN_ID, "sourceCutoff": cutoff.isoformat(),
                       "pricesPath": _display_source_path(prices_path), "pricesFileSha256": prices_sha,
                       "dailyTrainingStatusSnapshotRecordSha256": LIVE_SNAPSHOT_RECORD_SHA256,
                       "snapshotRecordHashIsDistinctFromPricesFileHash": True},
            "protocol": {"name": "existing_causal50", "description": "original 40-feature direct Ridge alpha=10, plus the existing daily-updated calendar SARIMA and 50:50 blend; SARIMA absence falls back to direct Ridge",
                         "fitOrigin": issue_date.replace(day=1).isoformat(),
                         "fitRule": "one direct Ridge per crop and horizon, labels target < October 1; month-start 730-calendar-day SARIMA fit through September 30",
                         "featureCutoff": "issue-date origin minus one day; source observations limited to frozen snapshot cutoff",
                         "sarimaStateRule": "daily append through issue-date minus one; absent dates remain NaN; forecast horizon+1 steps",
                         "sourceAgeLimitDays": 14, "blendWeights": [0.5, 0.5],
                         "notObservedDaysAreNotClosure": True},
            "inputSha256": {"prices": prices_sha, "historicalManifest": HISTORICAL_MANIFEST_SHA256,
                            **_code_hashes()},
            "sourceAvailability": {crop: _snapshot_status(records[crop], cutoff, issue_date, prices_path, prices_sha)
                                   for crop in PRODUCTS},
            "forecasts": rows}
    result["forecastSha256"] = _sha_bytes(_canonical(result))
    return result


def _code_hashes():
    modules = {"fourCropForecast": Path(__file__), "equalInfoForecasts": Path(online.__file__),
               "equalInfoScore": Path(online_score.__file__), "modeling": Path(online.modeling.__file__),
               "directHorizon": Path(online.direct_horizon.__file__), "sarimaCalendar": Path(online.ten.__file__),
               "rawPriceMaps": Path(online.rich.__file__), "priceSeries": Path(online.narrow.__file__)}
    return {name: _sha_bytes(path.read_bytes()) for name, path in modules.items()}


def _display_source_path(path: Path):
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _historical_rows(manifest, prices, manifest_raw):
    if _sha_bytes(manifest_raw) != HISTORICAL_MANIFEST_SHA256:
        raise ValueError("frozen historical forecast manifest file hash mismatch")
    unsigned = dict(manifest)
    digest = unsigned.pop("manifestSha256", None)
    if digest != _sha_bytes(online._canonical(unsigned)):
        raise ValueError("frozen historical forecast manifest content digest mismatch")
    if manifest.get("phase") != "forecast_only" or manifest.get("targetValuesAttached") is not False:
        raise ValueError("historical manifest must be answer-free")
    input_hashes = manifest.get("inputSha256", {})
    if input_hashes.get("prices") != HISTORICAL_PRICES_SHA256:
        raise ValueError("historical manifest price binding mismatch")
    if input_hashes.get("module") != "84e68a9a3eb362e59a336440a0bf9254eacb68dfb881e6cf3abdeeba4a215471":
        raise ValueError("historical manifest forecast-code binding mismatch")
    if manifest.get("protocol", {}).get("products") != list(online.PRODUCTS):
        raise ValueError("historical forecast product binding mismatch")
    return [row for row in manifest.get("evaluationForecasts", [])
            if row.get("product") in PRODUCTS and row.get("eligible") is True
            and EVAL_START <= _date(row.get("target")) <= EVAL_END]


def _score_historical(manifest_path=HISTORICAL_MANIFEST, prices_path=online.PRICES_PATH):
    manifest_raw = manifest_path.read_bytes()
    manifest = json.loads(manifest_raw)
    prices_raw = prices_path.read_bytes()
    if _sha_bytes(prices_raw) != HISTORICAL_PRICES_SHA256:
        raise ValueError("historical price input hash mismatch")
    prices = json.loads(prices_raw)
    rows = _historical_rows(manifest, prices, manifest_raw)
    price_series = online.narrow._series(prices)
    actuals = {(crop, day): value for crop, values in price_series.items() for day, value in values.items()
               if EVAL_START <= day <= EVAL_END}
    by_cell = []
    coverage = []
    temporal = []
    lookup = {(r["product"], _date(r["target"]), r["horizon"]): r for r in rows}
    expected = {(crop, EVAL_START + dt.timedelta(days=i), h)
                for crop in PRODUCTS for i in range((EVAL_END-EVAL_START).days + 1) for h in HORIZONS
                if EVAL_START <= EVAL_START + dt.timedelta(days=i) - dt.timedelta(days=h)}
    if set(lookup) != expected:
        raise ValueError("historical eligible forecast grid does not match frozen four-crop evaluation keys")

    month_days = {}
    day = EVAL_START
    while day <= EVAL_END:
        month_days.setdefault(day.strftime("%Y-%m"), set()).add(day)
        day += dt.timedelta(days=1)
    all_dm = []
    for crop in PRODUCTS:
        for horizon in HORIZONS:
            planned = [row for row in rows if row["product"] == crop and row["horizon"] == horizon]
            eligible = [row for row in planned if row["eligible"] is True]
            issued = {method: sum(_positive(row["models"].get(method, {}).get("prediction")) is not None
                                  for row in eligible) for method in METHODS}
            pairs = []
            for row in eligible:
                target = _date(row["target"])
                actual = actuals.get((crop, target))
                values = {method: _positive(row["models"].get(method, {}).get("prediction")) for method in METHODS}
                if actual is not None and all(value is not None for value in values.values()):
                    pairs.append({"target": target, "actual": actual, **values})
            cell_metrics = {method: _method_metrics(pairs, method) for method in METHODS}
            dates = sorted({row["target"] for row in pairs})
            by_cell.append({"product": crop, "productNameKo": NAMES[crop], "horizon": horizon,
                            "plannedEligibleKeys": len(planned), "positiveActualPairs": sum((crop, _date(r["target"])) in actuals for r in eligible),
                            "issued": issued, "allFourWayCommonPairs": len(pairs),
                            "uniqueCommonTargetDates": len(dates),
                            "commonTargetRange": [dates[0].isoformat(), dates[-1].isoformat()] if dates else None,
                            "metrics": cell_metrics})
            coverage.append({"product": crop, "horizon": horizon, "plannedEligibleKeys": len(planned),
                             "issuedByMethod": issued, "positiveActualPairs": sum((crop, _date(r["target"])) in actuals for r in eligible),
                             "allFourWayCommonPairs": len(pairs), "uniqueCommonTargetDates": len(dates)})
            for name, a, b in COMPARISONS:
                for loss in ("ae", "ape"):
                    differences = {}
                    for pair in pairs:
                        ea, eb = abs(pair[a] - pair["actual"]), abs(pair[b] - pair["actual"])
                        differences[pair["target"]] = ea - eb if loss == "ae" else (ea-eb)/pair["actual"]
                    for bandwidth in BANDWIDTHS:
                        test = online_score._dm(differences, bandwidth)
                        all_dm.append({"product": crop, "horizon": horizon, "comparison": name,
                                       "loss": loss, "bandwidth": bandwidth, **test})

    equal_horizon = []
    for crop in PRODUCTS:
        crop_cells = [row for row in by_cell if row["product"] == crop]
        for method in METHODS:
            supported = [r["metrics"][method] for r in crop_cells if r["metrics"][method]["n"] > 0]
            equal_horizon.append({"product": crop, "productNameKo": NAMES[crop], "method": method,
                                  "horizonsIncluded": len(supported),
                                  "meanHorizonMae": math.fsum(r["mae"] for r in supported)/len(supported) if supported else None,
                                  "meanHorizonMape": math.fsum(r["mape"] for r in supported)/len(supported) if supported else None,
                                  "weighting": "equal weight across horizons with a supported exact four-way sample"})
        for month, days in month_days.items():
            eligible_month = [r for r in rows if r["product"] == crop and _date(r["target"]) in days and r["eligible"]]
            pairs_by_target_h = {}
            for row in eligible_month:
                target = _date(row["target"])
                actual = actuals.get((crop, target))
                values = {method: _positive(row["models"].get(method, {}).get("prediction")) for method in METHODS}
                if actual is not None and all(v is not None for v in values.values()):
                    pairs_by_target_h.setdefault(target, []).append({"actual": actual, **values})
            all_pairs = [pair for target in sorted(pairs_by_target_h) for pair in pairs_by_target_h[target]]
            temporal.append({"product": crop, "month": month,
                             "plannedEligibleKeys": len(eligible_month),
                             "positiveActualTargets": len({d for d in days if (crop, d) in actuals}),
                             "allFourWayCommonPairs": len(all_pairs),
                             "uniqueCommonTargetDates": len(pairs_by_target_h),
                             "metrics": {method: _method_metrics(all_pairs, method) for method in METHODS}})

    for bandwidth in BANDWIDTHS:
        family = [row for row in all_dm if row["bandwidth"] == bandwidth]
        if len(family) != 240:
            raise ValueError(f"four-crop DM family must contain 240 hypotheses per bandwidth, got {len(family)}")
        online_score._holm(family)
    dm_summary = [{"bandwidth": bw, "familySize": 240,
                   "tested": sum(r["bandwidth"] == bw and r["status"] == "tested" for r in all_dm),
                   "significantImprovement": sum(r["bandwidth"] == bw and r["status"] == "tested" and r["holmSignificant"] and r["meanLossDifference"] < 0 for r in all_dm),
                   "significantWorsening": sum(r["bandwidth"] == bw and r["status"] == "tested" and r["holmSignificant"] and r["meanLossDifference"] > 0 for r in all_dm),
                   "inferenceUnavailable": sum(r["bandwidth"] == bw and r["status"] != "tested" for r in all_dm),
                   "holmAlpha": .05} for bw in BANDWIDTHS]
    return {"phase": "retrospective_descriptive_only", "sourceManifestSha256": _sha_bytes(manifest_raw),
            "historicalPricesSha256": _sha_bytes(prices_raw), "targetRange": [EVAL_START.isoformat(), EVAL_END.isoformat()],
            "codeSha256": _code_hashes(),
            "products": list(PRODUCTS), "methods": list(METHODS), "sampleRule": "evaluation rows with eligible origin only; exact four-method common keys and positive actuals within each product/horizon; target dates remain calendar indexed",
            "retrospectiveNotIndependentConfirmation": True,
            "metricsByProductHorizon": by_cell, "equalHorizonMetrics": equal_horizon,
            "monthlyMetrics": temporal, "coverage": coverage, "dm": all_dm,
            "dmSummaryByBandwidth": dm_summary,
            "interpretation": "Descriptive four-crop regrouping of a previously viewed and replay-verified 2026 period. It is not a new holdout, winner selection, or independent confirmation. Nonsignificance is not equivalence."}


def _metric_map(evaluation):
    return {(r["product"], r["method"]): r for r in evaluation["equalHorizonMetrics"]}


def _svg_chart(crop, actual_rows, forecasts):
    width, height, left, right, top, bottom = 720, 240, 52, 12, 15, 35
    values = [float(r["actual"]) for r in actual_rows]
    values += [m["prediction"] for row in forecasts for m in [row["models"]["causal50"]] if m["status"] == "issued"]
    if not values:
        return '<svg viewBox="0 0 720 240" role="img"><text x="20" y="40">자료 부족</text></svg>'
    lo, hi = min(values), max(values)
    pad = max((hi-lo)*.08, max(abs(hi), 1)*.01)
    lo, hi = lo-pad, hi+pad
    start = min([_date(r["date"]) for r in actual_rows] + [_date(r["target"]) for r in forecasts])
    end = max(_date(r["target"]) for r in forecasts)
    def xy(day, value):
        x = left + (_date(day)-start).days / max(1, (end-start).days) * (width-left-right)
        y = top + (hi-value) / max(1e-9, hi-lo) * (height-top-bottom)
        return f"{x:.1f},{y:.1f}"
    actual_points = " ".join(xy(r["date"], float(r["actual"])) for r in actual_rows)
    predicted = [r for r in forecasts if r["models"]["causal50"]["status"] == "issued"]
    forecast_points = " ".join(xy(r["target"], r["models"]["causal50"]["prediction"]) for r in predicted)
    actual_x = xy(actual_rows[-1]["date"], values[0]).split(",")[0] if actual_rows else "0"
    mid_y = height - bottom
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(NAMES[crop])} 실제 가격과 10일 예측 추이">'
            f'<line x1="{left}" y1="{mid_y}" x2="{width-right}" y2="{mid_y}" stroke="#d8dee8"/>'
            f'<line x1="{actual_x}" y1="{top}" x2="{actual_x}" y2="{mid_y}" stroke="#9aa6b2" stroke-dasharray="4 4"/>'
            f'<polyline points="{actual_points}" fill="none" stroke="#2864dc" stroke-width="2.5"/>'
            f'<polyline points="{forecast_points}" fill="none" stroke="#dc6328" stroke-width="3" stroke-dasharray="6 4"/>'
            f'<text x="{left}" y="{height-10}" font-size="12">실제 관측</text>'
            f'<text x="{left+90}" y="{height-10}" font-size="12">주황 점선: 혼합 예측</text>'
            f'<text x="8" y="{top+8}" font-size="11">{hi:.0f}</text><text x="8" y="{mid_y}" font-size="11">{lo:.0f}</text></svg>')


def render_html(forecast, evaluation, prices):
    actual_series = online.direct_horizon._observations(prices)
    metrics = _metric_map(evaluation)
    dm = evaluation["dmSummaryByBandwidth"]
    page = ["<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
            "<title>네 품목 가격 예측 실증</title><style>body{font:15px/1.6 system-ui,sans-serif;margin:0;background:#f5f7fa;color:#172433}main{max-width:1200px;margin:auto;padding:24px}.hero,.card{background:white;border:1px solid #dce2e9;border-radius:14px;padding:20px;margin:16px 0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:14px}.muted{color:#526172}.pill{display:inline-block;background:#edf3ff;border-radius:99px;padding:2px 10px;margin:3px}table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px;border-bottom:1px solid #e4e8ed;text-align:right}th:first-child,td:first-child{text-align:left}svg{width:100%;height:auto;background:#fff}.proposal{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}.proposal article{background:#f5f7fa;padding:14px;border-radius:10px}.small{font-size:12px}a{color:#2458bd}</style></head><body><main>"]
    source = forecast["source"]
    hashes = " · ".join(f"{html.escape(k)}: <code>{v}</code>" for k, v in sorted(forecast["inputSha256"].items()))
    page.append(f"<section class='hero'><h1>네 품목 향후 10일 가격 예측</h1><p>발행 기준일 {html.escape(forecast['issueDate'])} · 목표 {html.escape(forecast['targetDateRange'][0])}~{html.escape(forecast['targetDateRange'][1])} · 실제 생성 시각 {html.escape(forecast['issuedAtKst'])}</p><p>배추·무·양파·깐마늘의 최근 가격 흐름과 열흘 전망을 함께 살펴보는 화면입니다. 서울 가락·강서 도매 경매가격(원/kg)을 대상으로 하며 전국 공식 점수는 아닙니다. 건고추는 후속 확장 범위입니다.</p><p>입력 자료 cutoff는 {html.escape(source['sourceCutoff'])}입니다. 품목별 최신 관측일을 함께 표시했습니다.</p><details><summary>자료·방법·재현 정보</summary><p>원본 가격 파일 SHA-256 <code>{html.escape(source['pricesFileSha256'])}</code>; 수집 실행기록의 snapshot SHA-256 <code>{html.escape(source['dailyTrainingStatusSnapshotRecordSha256'])}</code>은 파일 해시와 다른 값입니다. 관측되지 않은 달력 날짜는 NaN으로 처리하며 휴장으로 해석하지 않았습니다.</p><p>월 첫 원점인 10월 1일에 품목×거리별 직접 Ridge(40개 특징, α=10)와 직전 730달력일 SARIMA를 적합했습니다. SARIMA는 10월 9일까지 상태를 갱신했고, 빠진 날짜는 NaN으로 두었습니다. 두 예측을 50:50으로 결합하고 SARIMA가 미발행이면 Ridge를 사용합니다. 모든 학습 정답은 적합 원점 이전입니다.</p><p>코드·입력 SHA-256: {hashes}</p></details><p class='muted'>과거 평가는 이미 살펴본 2026년 기간의 후향적 재집계입니다. 미래 가격은 아직 확인되지 않아 이번 예측의 정확도는 미평가 상태입니다.</p></section>")
    page.append("<section class='grid'>")
    for crop in PRODUCTS:
        fr = [r for r in forecast["forecasts"] if r["product"] == crop]
        history = sorted((d, pair[0]) for d, pair in actual_series[crop].items() if d <= _date(source["sourceCutoff"]))
        tail = [{"date": d.isoformat(), "actual": value} for d, value in history[-60:]]
        source_age = forecast["sourceAvailability"][crop]
        source_status = DISPLAY_STATUSES.get(source_age['status'], source_age['status'])
        page.append(f"<section class='card'><h2>{NAMES[crop]} <span class='pill'>최신 가격 {html.escape(source_age['lastObservedPriceDate'] or '없음')}</span><span class='pill'>{html.escape(source_status)}</span></h2><p>실제 마지막 관측 기준 나이: {source_age['sourceAgeDaysAtIssue']}일</p>{_svg_chart(crop, tail, fr)}<h3>2026년 후향적 공통표본 오차</h3><p class='small'>2026-01-01~09-27 기존 평가 중 원점 적격 행만 사용. 네 방법이 모두 발행되고 실제 양수 가격도 있는 동일 표본이며, 거리별 표본 수는 다를 수 있습니다.</p><table><thead><tr><th>방법</th><th>지원 거리</th><th>MAPE %</th><th>MAE 원/kg</th></tr></thead><tbody>")
        for method in METHODS:
            r = metrics.get((crop, method), {})
            mape = "—" if r.get("meanHorizonMape") is None else f"{r['meanHorizonMape']:.3f}"
            mae = "—" if r.get("meanHorizonMae") is None else f"{r['meanHorizonMae']:.2f}"
            page.append(f"<tr><td>{html.escape(DISPLAY_METHODS[method])}</td><td>{r.get('horizonsIncluded',0)}/10</td><td>{mape}</td><td>{mae}</td></tr>")
        page.append("</tbody></table></section>")
    page.append("</section><section class='card'><h2>향후 40개 예측</h2><p class='small'>각 행은 품목×목표일(거리)입니다. 미발행은 빈칸과 사유로 남깁니다.</p><div style='overflow:auto'><table><thead><tr><th>품목</th><th>목표일</th><th>거리</th><th>혼합 예측</th><th>Ridge</th><th>SARIMA</th><th>직전가격 유지</th><th>상태</th></tr></thead><tbody>")
    for row in forecast["forecasts"]:
        cells = []
        for method in METHODS:
            item = row["models"][method]
            cells.append("—" if item["status"] != "issued" else f"{item['prediction']:.2f}")
        status = "발행" if row["models"]["causal50"]["status"] == "issued" else html.escape(row["models"]["causal50"]["reason"] or "미발행")
        page.append(f"<tr><td>{NAMES[row['product']]}</td><td>{row['target']}</td><td>{row['horizon']}</td>" + "".join(f"<td>{value}</td>" for value in cells) + f"<td>{status}</td></tr>")
    page.append("</tbody></table></div></section>")
    page.append("<section class='card'><h2>후향 비교의 통계 요약</h2><p>세 대비×네 품목×열 거리×두 손실의 240개 가설을 HAC9·HAC28 각각 Holm 5%로 보정했습니다. 사전 고정된 과거 기간을 네 품목으로 다시 묶은 탐색적 기술 평가이며, 새 holdout·독립 확증·모형 선택 결과가 아닙니다.</p>")
    for row in dm:
        page.append(f"<p>HAC {row['bandwidth']}일: 가족 240개, 계산 {row['tested']}개, 유의 개선 {row['significantImprovement']}개, 유의 악화 {row['significantWorsening']}개, 추론 불가 {row['inferenceUnavailable']}개.</p>")
    page.append("<p>유의하지 않음은 동등성이나 무해성의 증명이 아닙니다. 미래 40개 예측은 목표일이 지나 실제 가격을 확인한 뒤에만 평가할 수 있습니다.</p><p><a href='evaluation.json'>세부 평가 JSON</a> · <a href='forecast.json'>발행 시점 예측 JSON</a> · <a href='forecast.csv'>예측 CSV</a></p></section>")
    page.append("<section class='card'><h2>제안서용 요약</h2><div class='proposal'><article><h3>예측아이디어</h3><p>과거 가격 수준·추세·계절성을 학습한 두 예측을 결합해 향후 1~10일 가격을 전망한다.</p></article><article><h3>데이터</h3><p>2014년 이후 보유한 서울 가락·강서 도매 경매의 kg당 일별 가격을 사용한다. 원점 전날까지의 관측을 입력하고 날짜 결측은 그대로 둔다. 두 시장 대리 목표이며 전국 경락가 공식 점수와 다르다.</p></article><article><h3>활용안</h3><p>도매시장 참여자가 품목별 최근 가격 흐름과 열흘 전망을 함께 보고 구매·출하 시점을 검토하는 참고자료로 제공한다. 화면에는 기준 자료일과 최신 관측일을 표시한다. 전망은 참고치이며 실제 가격과 다를 수 있다. 건고추는 후속 확장 범위다.</p></article></div></section>")
    page.append("</main></body></html>")
    return "\n".join(page)


def _csv_bytes(forecast):
    import io
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["product", "product_name_ko", "issue_date", "target_date", "horizon_days", "method", "status", "prediction_krw_per_kg", "reason", "fit_origin", "source_cutoff"])
    for row in forecast["forecasts"]:
        for method in METHODS:
            item = row["models"][method]
            writer.writerow([row["product"], NAMES[row["product"]], forecast["issueDate"], row["target"], row["horizon"], method,
                             item["status"], item["prediction"], item["reason"], row["fitOrigin"], row["sourceCutoff"]])
    return stream.getvalue().encode("utf-8-sig")


def _write_bundle(output_dir, forecast, evaluation, html_text):
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = {"forecast.json": _pretty(forecast), "evaluation.json": _pretty(evaluation),
               "forecast.csv": _csv_bytes(forecast), "index.html": html_text.encode("utf-8")}
    existing = [name for name in targets if (output_dir / name).exists()]
    if existing:
        raise FileExistsError("refusing to overwrite existing output: " + ", ".join(existing))
    written = []
    try:
        for name, content in targets.items():
            with (output_dir / name).open("xb") as stream:
                stream.write(content)
            written.append(name)
    except Exception:
        # Preserve any successful earlier write; never silently replace it.
        raise


def build_bundle(issue_date: dt.date, prices_path: Path, output_dir: Path, *, progress=None):
    if output_dir.resolve() == ROOT.resolve() or ROOT not in output_dir.resolve().parents:
        raise ValueError("output directory must be a child of the project root")
    if any((output_dir / name).exists() for name in ("forecast.json", "evaluation.json", "forecast.csv", "index.html")):
        raise FileExistsError("output bundle already exists; choose a new directory")
    future = build_future(prices_path, issue_date, progress=progress)
    evaluation = _score_historical(HISTORICAL_MANIFEST, online.PRICES_PATH)
    html_text = render_html(future, evaluation, json.loads(prices_path.read_text(encoding="utf-8")))
    _write_bundle(output_dir, future, evaluation, html_text)
    return {"outputDir": str(output_dir), "futureForecasts": len(future["forecasts"]),
            "historicalCells": len(evaluation["metricsByProductHorizon"]),
            "forecastFileSha256": _sha_bytes(_pretty(future))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--issue-date", required=True, type=dt.date.fromisoformat)
    build.add_argument("--prices", type=Path, default=LIVE_PRICES)
    build.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    if args.command == "build":
        result = build_bundle(args.issue_date, args.prices, args.output_dir,
            progress=lambda event: print(json.dumps(event, ensure_ascii=False, sort_keys=True), flush=True))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
