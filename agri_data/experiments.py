"""Pre-registered staged price experiments, separated into dev and holdout modes."""
from __future__ import annotations

import argparse
import datetime as dt
import os
import hashlib
import json
import math
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from . import coverage, dashboard, modeling

PRODUCT_IDS = tuple(dashboard.PRODUCTS)
MARKETS = dict(dashboard.MARKETS)
HORIZONS = tuple(range(1, 11))
GROUPS = {"1-3": (1, 2, 3), "4-7": (4, 5, 6, 7), "8-10": (8, 9, 10)}
CADENCES = modeling.CADENCES
MIN_TRAIN_DAYS = modeling.MIN_TRAIN_TARGET_DAYS
DIMENSIONS = {"subclass": ("gds_sclsf_cd", "gds_sclsf_nm"),
              "grade": ("grd_cd", "grd_nm"),
              "origin": ("plor_cd", "plor_nm")}

PROTOCOL = {
    "version": 1,
    "target": "서울가락·서울강서 경매/kg, 부호를 보존한 총금액 합계/총중량 합계",
    "vintage": "최신 검증 완료 응답을 이용한 회고적 수정본; 최초 공개시각 unknown",
    "maxSourceAgeDays": dashboard.MAX_SOURCE_AGE_DAYS,
    "minTrainingTargetDays": MIN_TRAIN_DAYS,
    "defaultCadence": "monthly",
    "estimators": {"ridge": modeling._model_parameters("ridge"),
                   "hist_gb": modeling._model_parameters("hist_gb")},
    "selection": {
        "development": "2015 only; frozen before 2016-Q1 holdout",
        "gate": "fixed parent와 같은 product×horizon×target 날짜에서 MAE와 MAPE 모두 strict improvement 및 coverage 비감소",
        "fallback": "gate 실패 시 parent 유지; E0 자체가 sparse면 persistence baseline",
        "sparse": "90 distinct training target days 미만은 insufficient_data; 관측을 보간하지 않음",
        "split": "개발 2015-01-01..2015-12-31; untouched exploratory holdout 2016-01-01..2016-03-31",
    },
    "stages": [
        {"id": "E0", "config": "기존 modeling feature 및 log-ratio, monthly refit, HistGB; Ridge도 비교 보존"},
        {"id": "S1", "candidates": ["relative", "same_weekday", "trading_observation", "combined"],
         "target": "기존 log-ratio", "grouping": "pooled", "cadence": "monthly"},
        {"id": "S2", "candidate": "S1 winner + market/composition panel"},
    {"id": "S3", "candidates": ["price_delta", "price_level", "grouped horizons 1-3/4-7/8-10"],
     "alsoCompare": "selected parent target with grouped horizons independently",
         "parent": "S2 winner or unchanged parent"},
        {"id": "S4", "source": "publication-evidence.json",
         "policy": "문서에 기록된 실제 발표본은 발표 다음날부터 사용; 현재 API 개정본은 사용하지 않음"},
    ],
    "dm": {"losses": ["absolute_error", "absolute_percentage_error"],
          "hacCalendarLagDays": 9, "alpha": 0.05, "twoSided": True,
          "missingDatePolicy": "평균/n은 유효 pair만 사용하고 실제 달력 간격 공분산에만 결측이 contribution 0",
          "holmFamily": "사전 고정한 final monthly winner-vs-E0, product×horizon×loss"},
    "holdout": "2016-Q1은 한 번만 평가; cadence 비교 reporting only, selectionReopened=false",
}

def parse_day(value: Any) -> dt.date:
    try:
        return dt.date.fromisoformat(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid ISO date: {value}") from exc

def validate_range(start: dt.date, end: dt.date) -> None:
    if start > end:
        raise ValueError("start date must not be after end date")
    if end > dt.datetime.now(dt.timezone.utc).date():
        raise ValueError("future dates are not permitted")

def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()

def _panel_hash(panel: dict[str, Any]) -> str:
    return _json_hash(panel)

def verify_target_equivalence(panel: dict[str, Any], baseline_dashboard: dict[str, Any],
                              start: dt.date = dt.date(2014, 1, 1),
                              end: dt.date = dt.date(2015, 12, 31)) -> dict[str, Any]:
    """Fail closed unless target dates, weighted prices, and weights match E0."""
    if baseline_dashboard.get("schemaVersion") != 1:
        raise ValueError("target guard requires price dashboard schemaVersion 1")
    panel_start, panel_end = parse_day(panel["panelStart"]), parse_day(panel["panelEnd"])
    lo, hi = max(start, panel_start), min(end, panel_end)
    if lo > hi:
        raise ValueError("price panel has no overlap with the frozen E0 target range")
    current_products = {item["id"]: item for item in panel.get("products", [])}
    reference_products = {item["id"]: item for item in baseline_dashboard.get("products", [])}
    if set(current_products) != set(PRODUCT_IDS) or set(reference_products) != set(PRODUCT_IDS):
        raise ValueError("target guard requires all five dashboard product scopes")
    compared = {}
    for product in PRODUCT_IDS:
        def scoped(rows: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
            return {str(row["date"]): (float(row["actual"]), float(row["volumeKg"]))
                    for row in rows if lo <= parse_day(row["date"]) <= hi}
        current = scoped(current_products[product].get("observations", []))
        reference = scoped(reference_products[product].get("observations", []))
        if set(current) != set(reference):
            missing = sorted(set(reference)-set(current))[:5]
            extra = sorted(set(current)-set(reference))[:5]
            raise ValueError(f"E0 target date mismatch for {product}: missing={missing}, extra={extra}")
        for day in sorted(current):
            for label, actual, expected in (("actual", current[day][0], reference[day][0]),
                                            ("volumeKg", current[day][1], reference[day][1])):
                if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-7):
                    raise ValueError(f"E0 {label} mismatch for {product} on {day}")
        compared[product] = len(current)
    return {"status": "verified", "start": lo.isoformat(), "end": hi.isoformat(),
            "observationsByProduct": compared,
            "baselineDashboardSha256": _json_hash(baseline_dashboard),
            "tolerance": {"absolute": 1e-7, "relative": 1e-10}}

def _group(horizon: int, grouping: str) -> str:
    if grouping == "pooled":
        return "pooled"
    for label, values in GROUPS.items():
        if horizon in values:
            return label
    raise ValueError("unsupported horizon grouping")

def _dim_stats(weights: dict[str, float], denominator: float) -> dict[str, float]:
    shares = [float(v / denominator) for v in weights.values()] if denominator > 0 else []
    return {"distinctCategories": float(len(shares)),
            "largestSignedVolumeShare": max(shares, default=0.0),
            "concentrationHhi": sum(x * x for x in shares),
            "signedVolumeShares": {name: float(value / denominator) for name, value in sorted(weights.items())}
            if denominator > 0 else {}}

def build_price_panel(db: sqlite3.Connection, start: dt.date, end: dt.date) -> dict[str, Any]:
    """Read only the latest validated selected price requests and stream rows into daily cells."""
    validate_range(start, end)
    buckets = defaultdict(lambda: {"amount": Decimal(0), "volume": Decimal(0), "rows": 0,
                                   "dimensions": {k: defaultdict(Decimal) for k in DIMENSIONS},
                                   "flags": Counter()})
    states, garlic_specific = {}, set()
    sources, jobs, requests = set(), set(), set()
    diagnostics = {p: {"incompleteBroadMarketDays": 0, "missingOrIncompleteQueries": 0,
                       "excludedRows": Counter(), "qualityFlags": Counter(),
                       "queryStates": Counter(), "compositionFieldsSeen": set(), "includedRows": 0}
                  for p in PRODUCT_IDS}
    for rec in db.execute(dashboard.PRICE_SQL, (start.isoformat(), end.isoformat(), *MARKETS.keys())):
        query = dashboard._json_object(rec["query_json"])
        if query is None:
            continue
        product = dashboard._query_product(query)
        day = dashboard._parse_date(query.get("cond[trd_clcln_ymd::EQ]"))
        market = str(query.get("cond[whsl_mrkt_cd::EQ]", ""))
        subquery = str(query.get("cond[gds_sclsf_cd::EQ]", ""))
        if product not in PRODUCT_IDS or day is None or market not in MARKETS:
            continue
        state = rec["selected_state"] if rec["selected_state"] in ("complete", "empty") else (rec["latest_state"] or "unavailable")
        states[(product, day, market, subquery)] = str(state)
        diagnostics[product]["queryStates"][str(state)] += 1
        if product == "garlic" and subquery in dashboard.PEELED_GARLIC_LABELS and state in ("complete", "empty"):
            garlic_specific.add((day, market, subquery))
        if rec["source_url"]:
            sources.add(str(rec["source_url"]))
        if rec["selected_job_id"] is not None:
            jobs.add(int(rec["selected_job_id"]))
        if rec["request_id"] is None or rec["payload_json"] is None:
            continue
        requests.add(int(rec["request_id"]))
        diag = diagnostics[product]
        row = dashboard._json_object(rec["payload_json"])
        if row is None:
            diag["excludedRows"]["invalid_json"] += 1
            continue
        if dashboard._product_id(row) != product:
            diag["excludedRows"]["category_mismatch"] += 1
            continue
        subclass = str(row.get("gds_sclsf_cd", ""))
        if product == "garlic":
            if not dashboard._garlic_class_is_verified(row):
                diag["excludedRows"]["unverified_garlic_class"] += 1
                continue
            if subquery and subclass != subquery:
                diag["excludedRows"]["subclass_query_mismatch"] += 1
                continue
            if not subquery and (day, market, subclass) in garlic_specific:
                diag["excludedRows"]["specific_scope_preferred"] += 1
                continue
        if row.get("trd_se") != "경매":
            diag["excludedRows"]["not_auction"] += 1
            continue
        if row.get("unit_nm") != "kg":
            diag["excludedRows"]["unit_not_kg"] += 1
            continue
        if str(row.get("whsl_mrkt_cd", "")) != market or dashboard._parse_date(row.get("trd_clcln_ymd")) != day:
            diag["excludedRows"]["source_scope_mismatch"] += 1
            continue
        amount, volume = dashboard._as_decimal(row.get("totprc")), dashboard._as_decimal(row.get("unit_tot_qty"))
        if amount is None or volume is None:
            diag["excludedRows"]["invalid_amount_volume"] += 1
            continue
        b = buckets[(product, day, market)]
        b["amount"] += amount
        b["volume"] += volume
        b["rows"] += 1
        diag["includedRows"] += 1
        for name, fields in DIMENSIONS.items():
            val = next((row.get(field) for field in fields if row.get(field) not in (None, "")), None)
            if val is not None:
                b["dimensions"][name][str(val)] += volume
                diag["compositionFieldsSeen"].add(name)
        for field in {field for fields in DIMENSIONS.values() for field in fields}:
            value = row.get(field)
            if value not in (None, ""):
                b.setdefault("compositionFields", defaultdict(lambda: defaultdict(Decimal)))[field][str(value)] += volume
                diag["compositionFieldsSeen"].add(field)
        try:
            flags = json.loads(rec["quality_flags"] or "[]")
        except (json.JSONDecodeError, TypeError):
            flags = []
        if isinstance(flags, list):
            for flag in flags:
                b["flags"][str(flag)] += 1
                diag["qualityFlags"][str(flag)] += 1

    products, days_total = [], (end - start).days + 1
    for product in PRODUCT_IDS:
        obs, diag = [], diagnostics[product]
        for n in range(days_total):
            day = start + dt.timedelta(days=n)
            okay = all(states.get((product, day, market, ""), "unplanned") in ("complete", "empty")
                       for market in MARKETS)
            if not okay:
                diag["incompleteBroadMarketDays"] += 1
                diag["missingOrIncompleteQueries"] += sum(
                    states.get((product, day, market, ""), "unplanned") not in ("complete", "empty")
                    for market in MARKETS)
                continue
            market_data, amount_sum, volume_sum, row_count = {}, Decimal(0), Decimal(0), 0
            for market in MARKETS:
                b = buckets.get((product, day, market))
                if not b:
                    continue
                # The daily target sums signed amounts and volumes across both
                # markets before division. A market with a nonpositive subtotal
                # still contributes its adjustments to that signed sum.
                amount_sum += b["amount"]
                volume_sum += b["volume"]
                row_count += b["rows"]
                if b["volume"] > 0:
                    price = float(b["amount"] / b["volume"])
                    if price > 0 and math.isfinite(price):
                        market_data[market] = {
                            "price": price, "volumeKg": float(b["volume"]), "rowCount": b["rows"],
                            "qualityFlagCount": sum(b["flags"].values()),
                            "qualityFlags": dict(sorted(b["flags"].items())),
                            "composition": {k: _dim_stats(v, b["volume"])
                                            for k, v in b["dimensions"].items() if v},
                            "compositionFieldShares": {
                                field: {name: float(value/b["volume"])
                                        for name, value in sorted(weights.items())}
                                for field, weights in b.get("compositionFields", {}).items()},
                        }
            if volume_sum <= 0 or row_count == 0:
                continue
            price = float(amount_sum / volume_sum)
            if price > 0 and math.isfinite(price):
                obs.append({"date": day.isoformat(), "actual": price, "volumeKg": float(volume_sum),
                            "rowCount": row_count, "marketData": market_data})
        diag["excludedRows"] = dict(sorted(diag["excludedRows"].items()))
        diag["qualityFlags"] = dict(sorted(diag["qualityFlags"].items()))
        diag["queryStates"] = dict(sorted(diag["queryStates"].items()))
        diag["compositionFieldsSeen"] = sorted(diag["compositionFieldsSeen"])
        products.append({"id": product, "observations": obs, "diagnostics": diag})
    return {"products": products, "panelStart": start.isoformat(), "panelEnd": end.isoformat(),
            "marketCodes": list(MARKETS), "marketNames": list(MARKETS.values()),
            "sourceUrls": sorted(sources), "selectedJobIds": sorted(jobs),
            "selectedRequestCount": len(requests), "publicationTimeKnown": False}

def _series(product: dict[str, Any]) -> dict[dt.date, tuple[float, float]]:
    return modeling._observed_series(product)

def _feature(product: dict[str, Any], series: dict[dt.date, tuple[float, float]],
             origin: dt.date, target: dt.date, horizon: int, feature_set: str,
             cache: dict[Any, Any], publication_values: list[dict[str, Any]] | None = None
             ) -> tuple[dt.date, tuple[float, ...]] | None:
    key = ("exp", product["id"], origin, target, horizon, feature_set)
    if key in cache:
        return cache[key]
    base = modeling._feature_row(series, origin, target, horizon, cache)
    if base is None:
        cache[key] = None
        return None
    source, values = base
    last_log = math.log(series[source][0])
    extras = []
    modes = set(feature_set.split("+"))
    if "combined" in modes:
        modes.update(("relative", "same_weekday", "trading_observation"))
    days = [d for d in series if d <= origin]
    if "relative" in modes:
        for window in (7, 28, 90):
            values_in_window = [math.log(series[d][0]) for d in days if (origin - d).days < window]
            extras.append(last_log - sum(values_in_window) / len(values_in_window) if values_in_window else 0.0)
    if "same_weekday" in modes:
        prior = [d for d in reversed(days) if d < origin and d.weekday() == target.weekday()][:3]
        for i in range(3):
            extras.extend((math.log(series[prior[i]][0]) - last_log,
                           float((origin - prior[i]).days), 1.0) if i < len(prior) else (0.0, 0.0, 0.0))
    if "trading_observation" in modes:
        prior = list(reversed(days[-5:]))
        for i in range(1, 5):
            extras.extend((math.log(series[prior[i]][0]) - last_log, float((origin-prior[i]).days), 1.0)
                          if i < len(prior) else (0.0, 0.0, 0.0))
    if "market_composition" in modes:
        by_day = {parse_day(x["date"]): x for x in product.get("observations", [])}
        latest = {m: None for m in MARKETS}
        for d in reversed(days):
            row = by_day.get(d, {})
            for m in MARKETS:
                if latest[m] is None and m in row.get("marketData", {}):
                    latest[m] = (d, row["marketData"][m])
            if all(latest.values()):
                break
        fresh_latest = {m: value for m, value in latest.items()
                        if value and (origin-value[0]).days <= dashboard.MAX_SOURCE_AGE_DAYS}
        total_vol = sum(value[1]["volumeKg"] for value in fresh_latest.values())
        for m in MARKETS:
            item = latest[m]
            if not item:
                extras.extend([0.0] * 14)
                continue
            d, cell = item
            age = (origin-d).days
            if age > dashboard.MAX_SOURCE_AGE_DAYS:
                extras.extend((0.0, 0.0, float(age), *([0.0] * 11)))
                continue
            extras.extend((math.log(cell["price"])-last_log,
                           cell["volumeKg"]/total_vol if total_vol > 0 else 0.0,
                           float(age), float(cell["rowCount"]), float(cell["qualityFlagCount"])))
            for dim in ("subclass", "grade", "origin"):
                stats = cell.get("composition", {}).get(dim, {})
                extras.extend((float(stats.get("distinctCategories", 0)),
                               float(stats.get("largestSignedVolumeShare", 0)),
                               float(stats.get("concentrationHhi", 0))))
    if "publication" in modes:
        extras.extend(_publish_features(origin, publication_values or []))
    result = (source, tuple(values) + tuple(extras))
    if not all(math.isfinite(v) for v in result[1]):
        raise ValueError("non-finite feature")
    cache[key] = result
    return result

def _label(target: float, source: float, transform: str) -> float:
    if transform == "log_ratio": return math.log(target)-math.log(source)
    if transform == "price_delta": return target-source
    if transform == "price_level": return target
    raise ValueError("unknown target transform")

def _inverse(value: float, source: float, transform: str) -> float:
    if transform == "log_ratio": result = source * math.exp(min(700, max(-700, value)))
    elif transform == "price_delta": result = source + value
    elif transform == "price_level": result = value
    else: raise ValueError("unknown target transform")
    if not math.isfinite(result) or result <= 0:
        raise ValueError("prediction is not finite positive")
    return result

def _baseline(series: dict[dt.date, tuple[float, float]], start: dt.date, end: dt.date) -> list[dict[str, Any]]:
    from bisect import bisect_right
    days = tuple(series)
    out = []
    target = start
    while target <= end:
        if target in series:
            for h in HORIZONS:
                origin = target-dt.timedelta(days=h)
                i = bisect_right(days, origin)-1
                if i >= 0 and (origin-days[i]).days <= dashboard.MAX_SOURCE_AGE_DAYS:
                    out.append({"target": target.isoformat(), "origin": origin.isoformat(), "horizon": h,
                                "predicted": series[days[i]][0], "sourceDate": days[i].isoformat()})
        target += dt.timedelta(days=1)
    return out

def _fit(product: dict[str, Any], config: dict[str, str], start: dt.date, end: dt.date,
         min_days: int, publication_values: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    import numpy as np
    from threadpoolctl import threadpool_limits
    series = _series(product)
    if not series:
        return {"status": "insufficient_data", "predictions": [], "fits": [], "diagnostics": {"issued": 0}}
    cache, samples = {}, []
    for target, (target_price, _) in series.items():
        for h in HORIZONS:
            origin = target-dt.timedelta(days=h)
            feat = _feature(product, series, origin, target, h, config["features"], cache, publication_values)
            if feat:
                source, vector = feat
                samples.append((target, origin, h, vector, _label(target_price, series[source][0], config["target"])))
    samples.sort(key=lambda x: (x[0], x[1], x[2]))
    groups = ("pooled",) if config["grouping"] == "pooled" else tuple(GROUPS)
    preds, fits, estimators, fit_cutoffs = [], [], {}, {}
    last_period = None
    fit_count = issued = invalid_predictions = 0
    with threadpool_limits(limits=1):
        origin = start-dt.timedelta(days=max(HORIZONS))
        while origin < end:
            period = modeling._cadence_key(config["cadence"], origin)
            if period != last_period:
                last_period = period
                for group in groups:
                    train = [x for x in samples if x[0] <= origin and _group(x[2], config["grouping"]) == group]
                    targets = sorted({x[0] for x in train})
                    if len(targets) < min_days:
                        estimators.pop(group, None)
                        fit_cutoffs.pop(group, None)
                        fits.append({"cutoff": origin.isoformat(), "group": group, "targetDays": len(targets),
                                     "samples": len(train), "status": "insufficient_data"})
                        continue
                    estimator = modeling._new_estimator(config["model"])
                    estimator.fit(np.asarray([x[3] for x in train], dtype=np.float64),
                                  np.asarray([x[4] for x in train], dtype=np.float64))
                    estimators[group] = estimator
                    fit_cutoffs[group] = origin
                    fit_count += 1
                    fits.append({"cutoff": origin.isoformat(), "group": group, "targetDays": len(targets),
                                 "samples": len(train), "trainStart": targets[0].isoformat(),
                                 "trainEnd": targets[-1].isoformat(), "status": "fitted"})
            for h in HORIZONS:
                group = _group(h, config["grouping"])
                estimator = estimators.get(group)
                fit_cutoff = fit_cutoffs.get(group)
                if estimator is None or fit_cutoff is None:
                    continue
                target = origin+dt.timedelta(days=h)
                feat = _feature(product, series, origin, target, h, config["features"], cache, publication_values)
                if not feat:
                    continue
                source, vector = feat
                raw = float(estimator.predict([vector])[0])
                if not math.isfinite(raw):
                    continue
                try:
                    prediction = _inverse(raw, series[source][0], config["target"])
                except ValueError:
                    invalid_predictions += 1
                    continue
                issued += 1
                if start <= target <= end and target in series:
                    preds.append({"origin": origin.isoformat(), "target": target.isoformat(), "horizon": h,
                                  "predicted": prediction, "sourceDate": source.isoformat(),
                                  "fitCutoff": fit_cutoff.isoformat(), "fitGroup": group})
            origin += dt.timedelta(days=1)
    status = "available" if fit_count and issued else "insufficient_data"
    reason = None
    if not fit_count:
        reason = f"서로 다른 훈련 목표 날짜가 최소 {min_days}일에 미달합니다."
    elif not issued:
        reason = "예측 원점에서 14일 이내의 유효 관측을 확보하지 못했습니다."
    return {"status": status, **({"reason": reason} if reason else {}),
            "predictions": preds, "fits": fits,
            "diagnostics": {"issued": issued, "matched": len(preds), "successfulFits": fit_count,
                            "insufficientFitAttempts": sum(x["status"] == "insufficient_data" for x in fits),
                            "invalidInversePredictions": invalid_predictions}}

def _actuals(product: dict[str, Any], start: dt.date, end: dt.date) -> dict[tuple[str, int], float]:
    series = _series(product)
    return {(day.isoformat(), h): price for day, (price, _) in series.items() if start <= day <= end for h in HORIZONS}

def _metrics(predictions: list[dict[str, Any]], actuals: dict[tuple[str, int], float],
             keys: set[tuple[str, int]] | None = None) -> dict[str, Any]:
    values = {(r["target"], int(r["horizon"])): float(r["predicted"]) for r in predictions}
    common = set(values) & set(actuals)
    if keys is not None: common &= keys
    if not common: return {"n": 0, "mae": None, "mape": None, "rmse": None, "coverage": 0.0}
    errors = [(values[k]-actuals[k], actuals[k]) for k in sorted(common)]
    n = len(errors)
    return {"n": n, "mae": sum(abs(e) for e, _ in errors)/n,
            "mape": sum(abs(e)/a*100 for e, a in errors)/n,
            "rmse": math.sqrt(sum(e*e for e, _ in errors)/n),
            "coverage": n/max(1, len(keys) if keys is not None else len(actuals))}

def _metrics_by_horizon(predictions: list[dict[str, Any]],
                         actuals: dict[tuple[str, int], float]) -> dict[str, Any]:
    output = {}
    for horizon in HORIZONS:
        keys = {key for key in actuals if key[1] == horizon}
        output[str(horizon)] = _metrics(predictions, actuals, keys)
    return output

def _metrics_by_period(predictions: list[dict[str, Any]], actuals: dict[tuple[str, int], float],
                       period: str) -> dict[str, Any]:
    def label(target: str) -> str:
        day = parse_day(target)
        if period == "month": return day.strftime("%Y-%m")
        if period == "week":
            iso = day.isocalendar()
            return f"{iso.year}-W{iso.week:02d}"
        if period == "day": return day.isoformat()
        if day.month in (12, 1, 2):
            year = day.year if day.month == 12 else day.year - 1
            return f"{year}-{year+1}-winter"
        elif day.month in (3, 4, 5): season = "spring"
        elif day.month in (6, 7, 8): season = "summer"
        else: season = "autumn"
        return f"{day.year}-{season}"
    labels = sorted({label(target) for target, _ in actuals})
    return {item: _metrics(predictions, actuals,
                           {key for key in actuals if label(key[0]) == item}) for item in labels}

def _dm(candidate: list[dict[str, Any]], parent: list[dict[str, Any]],
        actuals: dict[tuple[str, int], float], horizon: int, loss: str, max_lag: int = 9) -> dict[str, Any]:
    from bisect import bisect_left
    c = {(r["target"], int(r["horizon"])): float(r["predicted"]) for r in candidate}
    p = {(r["target"], int(r["horizon"])): float(r["predicted"]) for r in parent}
    keys = sorted(k for k in actuals if k[1] == horizon and k in c and k in p)
    n = len(keys)
    if n < 10: return {"n": n, "status": "unavailable_insufficient_pairs"}
    points = []
    for target, h in keys:
        actual = actuals[(target, h)]
        a, b = abs(c[(target, h)]-actual), abs(p[(target, h)]-actual)
        if loss == "absolute_percentage_error": a, b = a/actual*100, b/actual*100
        points.append((parse_day(target), a-b))
    mean = sum(v for _, v in points)/n
    centered = [(d, v-mean) for d, v in points]
    day_map = dict(centered)
    variance_sum = sum(v*v for _, v in centered)
    for lag in range(1, max_lag+1):
        cov = sum(v*day_map.get(d-dt.timedelta(days=lag), 0.0) for d, v in centered)
        variance_sum += 2*(1-lag/(max_lag+1))*cov
    if variance_sum <= 0 or not math.isfinite(variance_sum):
        return {"n": n, "status": "unavailable_nonpositive_variance"}
    stat = mean/(math.sqrt(variance_sum)/n)
    return {"n": n, "status": "available", "meanLossDifference": mean, "statistic": stat,
            "pValue": math.erfc(abs(stat)/math.sqrt(2)), "calendarHacMaxLagDays": max_lag}

def _improves(candidate: list[dict[str, Any]], parent: list[dict[str, Any]],
              actuals: dict[tuple[str, int], float]) -> tuple[bool, dict[str, Any]]:
    cm = {(r["target"], int(r["horizon"])): r for r in candidate}
    pm = {(r["target"], int(r["horizon"])): r for r in parent}
    common = set(cm) & set(pm) & set(actuals)
    a, b = _metrics([cm[k] for k in common], actuals, common), _metrics([pm[k] for k in common], actuals, common)
    cov_a, cov_b = len(set(cm)&set(actuals)), len(set(pm)&set(actuals))
    return bool(a["n"] >= 10 and cov_a >= cov_b and a["mae"] < b["mae"] and a["mape"] < b["mape"]), {
        "candidate": a, "parent": b, "candidateCoverage": cov_a, "parentCoverage": cov_b,
        "coverageNondecrease": cov_a >= cov_b,
        "samePairsByHorizon": {str(h): {"candidate": _metrics([cm[k] for k in common if k[1] == h], actuals,
                                                                 {k for k in common if k[1] == h}),
                                         "parent": _metrics([pm[k] for k in common if k[1] == h], actuals,
                                                             {k for k in common if k[1] == h})}
                               for h in HORIZONS}}

def _dm_family(candidate: list[dict[str, Any]], parent: list[dict[str, Any]],
               actuals: dict[tuple[str, int], float], product_id: str = "") -> list[dict[str, Any]]:
    rows = []
    for h in HORIZONS:
        for loss in ("absolute_error", "absolute_percentage_error"):
            rows.append({"productId": product_id, "horizon": h, "loss": loss,
                         **_dm(candidate, parent, actuals, h, loss)})
    return rows

def _configs() -> list[tuple[str, dict[str, str]]]:
    base = {"model": "hist_gb", "features": "core", "target": "log_ratio",
            "grouping": "pooled", "cadence": "monthly"}
    out = [("E0", base), ("E0-ridge", {**base, "model": "ridge"})]
    out.extend((f"S1-{f}", {**base, "features": f}) for f in
               ("relative", "same_weekday", "trading_observation", "combined"))
    return out

def _publication_evidence(path: str | Path | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if path is None or not Path(path).is_file():
        return [], {"status": "skipped", "reason": "검증된 발표 근거 파일 없음"}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for doc in data.get("documents", []):
        if not (doc.get("sha256") and doc.get("url") and doc.get("publication_evidence_pdf_page")
                and doc.get("published_at_precision") in ("minute", "day")):
            continue
        published = doc.get("published_at")
        try: release = dt.datetime.fromisoformat(published)
        except (TypeError, ValueError): continue
        for item in doc.get("observations", []):
            # Only onion matches the current daily target definition; broad garlic,
            # seasonal cabbage/radish, and generic pepper are intentionally excluded.
            if item.get("crop") != "onion" or item.get("measure") not in ("production", "cultivated_area"):
                continue
            value = item.get("value")
            if isinstance(value, (int, float)) and value > 0 and math.isfinite(float(value)):
                rows.append({"measure": item["measure"], "value": float(value),
                             "referenceYear": int(item["reference_year"]),
                             "availableDate": (release.date()+dt.timedelta(days=1)).isoformat(),
                             "publishedAt": published, "documentId": doc.get("id"), "url": doc.get("url")})
    return rows, {"status": "available" if rows else "skipped", "includedObservations": len(rows),
                  "reason": None if rows else "타깃과 일치하는 발표본이 없습니다.",
                  "documentIds": sorted({row["documentId"] for row in rows}),
                  "sourceUrls": sorted({row["url"] for row in rows}),
                  "excludedDefinitions": ["garlic_all≠깐마늘", "계절 배추/무≠일별 전체 타깃", "고추≠건고추 확인 불충분"]}

def _publish_features(origin: dt.date, rows: list[dict[str, Any]]) -> tuple[float, ...]:
    best = {}
    for row in rows:
        if parse_day(row["availableDate"]) <= origin:
            measure = row["measure"]
            if measure not in best or (row["referenceYear"], row["publishedAt"]) > (best[measure]["referenceYear"], best[measure]["publishedAt"]):
                best[measure] = row
    output = []
    for measure in ("cultivated_area", "production"):
        row = best.get(measure)
        output.extend((math.log(row["value"]), float((origin-parse_day(row["availableDate"])).days), 1.0)
                      if row else (0.0, 0.0, 0.0))
    return tuple(output)

def _fit_logged(product: dict[str, Any], config: dict[str, str], start: dt.date, end: dt.date,
                min_days: int, label: str, callback: Callable[[dict[str, Any]], None] | None,
                publication_values: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    base = {"productId": product["id"], "stage": label, "model": config["model"],
            "features": config["features"], "target": config["target"],
            "grouping": config["grouping"], "cadence": config["cadence"]}
    if callback:
        callback({**base, "state": "started"})
    run = _fit(product, config, start, end, min_days, publication_values)
    if callback:
        callback({**base, "state": "completed", "fits": run["diagnostics"].get("successfulFits", 0),
                  "matchedPairs": run["diagnostics"].get("matched", 0),
                  "elapsedSeconds": time.perf_counter() - started})
    return run

def _summarize_product(product: dict[str, Any], start: dt.date, end: dt.date,
                       min_days: int, evidence: list[dict[str, Any]],
                       progress_callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    series = _series(product)
    actuals = _actuals(product, start, end)
    baseline = _baseline(series, start, end)
    base_config = {"model": "hist_gb", "features": "core", "target": "log_ratio",
                   "grouping": "pooled", "cadence": "monthly"}
    e0 = _fit_logged(product, base_config, start, end, min_days, "E0", progress_callback)
    ridge_config = {**base_config, "model": "ridge"}
    ridge_run = _fit_logged(product, ridge_config, start, end, min_days, "E0-ridge", progress_callback)
    trials = [{"id": "E0", "config": base_config, "run": e0,
               "metrics": _metrics(e0["predictions"], actuals),
               "baselineMetrics": _metrics(baseline, actuals),
               "pairedDm": _dm_family(e0["predictions"], baseline, actuals, product["id"])}]
    trials.append({"id": "E0-ridge", "config": ridge_config, "run": ridge_run,
                   "comparison": _comparison(ridge_run["predictions"], e0["predictions"], actuals),
                   "pairedDm": _dm_family(ridge_run["predictions"], e0["predictions"], actuals, product["id"])})
    parent_config, parent_run = base_config, e0
    # S1 tests one factor at a time; do not conflate targets or horizon groups here.
    viable = []
    if e0["status"] == "available":
        for name, config in _configs()[2:]:
            run = _fit_logged(product, config, start, end, min_days, name, progress_callback)
            change, comparison = _improves(run["predictions"], e0["predictions"], actuals)
            trials.append({"id": name, "config": config, "run": run, "comparison": comparison,
                           "pairedDm": _dm_family(run["predictions"], e0["predictions"], actuals, product["id"])})
            if change:
                m = comparison["candidate"]
                viable.append((m["mape"], m["mae"], name, config, run))
    s1_changed = bool(viable)
    if viable:
        _, _, _, parent_config, parent_run = min(viable)
    # Keep Ridge E0 visible as a benchmark; stage gates remain fixed to parent.
    s2_parent_run = parent_run
    s2_config = {**parent_config, "features": parent_config["features"] + "+market_composition"}
    s2 = _fit_logged(product, s2_config, start, end, min_days, "S2-market-composition", progress_callback) if parent_run["status"] == "available" else {
        "status": "insufficient_data", "predictions": [], "fits": [], "diagnostics": {"issued": 0}}
    s2_changed, s2_comparison = _improves(s2["predictions"], parent_run["predictions"], actuals)
    if s2_changed: parent_config, parent_run = s2_config, s2
    trials.append({"id": "S2-market-composition", "config": s2_config, "run": s2,
                   "comparison": s2_comparison, "selected": s2_changed,
                   "pairedDm": _dm_family(s2["predictions"], s2_parent_run["predictions"], actuals, product["id"])})
    # If S1 had no success, S3 explicitly holds E0 features while still probing
    # target/forecast-structure changes as directed by the protocol.
    s3_candidates = []
    s3_parent_config, s3_parent_run = parent_config, parent_run
    if s3_parent_run["status"] == "available":
        for transform in (s3_parent_config["target"], "price_delta", "price_level"):
            for grouping in ("pooled", "grouped"):
                if transform == s3_parent_config["target"] and grouping == s3_parent_config["grouping"]:
                    continue
                config = {**s3_parent_config, "target": transform, "grouping": grouping}
                run = _fit_logged(product, config, start, end, min_days, f"S3-{transform}-{grouping}", progress_callback)
                changed, comparison = _improves(run["predictions"], s3_parent_run["predictions"], actuals)
                name = f"S3-{transform}-{grouping}"
                trials.append({"id": name, "config": config, "run": run,
                               "comparison": comparison, "selected": False,
                               "pairedDm": _dm_family(run["predictions"], s3_parent_run["predictions"], actuals,
                                                      product["id"])})
                if changed:
                    m = comparison["candidate"]
                    s3_candidates.append((m["mape"], m["mae"], name, config, run))
    s3_changed = bool(s3_candidates)
    if s3_candidates:
        _, _, s3name, parent_config, parent_run = min(s3_candidates)
        next(t for t in trials if t["id"] == s3name)["selected"] = True
    s4 = {"status": "skipped", "reason": "타깃과 정의가 일치하는 공표본 근거가 없습니다."}
    if product["id"] == "onion" and evidence:
        s4_config = {**parent_config, "features": parent_config["features"] + "+publication"}
        s4_run = _fit_logged(product, s4_config, start, end, min_days, "S4-publication-asof",
                             progress_callback, publication_values=evidence)
        s4_changed, s4_comparison = _improves(s4_run["predictions"], parent_run["predictions"], actuals)
        trials.append({"id": "S4-publication-asof", "config": s4_config, "run": s4_run,
                       "comparison": s4_comparison, "selected": s4_changed,
                       "pairedDm": _dm_family(s4_run["predictions"], parent_run["predictions"], actuals)})
        if s4_changed:
            parent_config, parent_run = s4_config, s4_run
        s4 = {"status": "evaluated", "selected": s4_changed,
              "reason": "공식 문서에 기록된 양파 발표본만 사용하고 공개 다음날부터 가용 처리"}
    baseline_fallback = parent_run["status"] != "available"
    frozen = ({"model": "persistence", "features": "none", "target": "price_level",
               "grouping": "none", "cadence": "monthly"} if baseline_fallback else parent_config)
    return {"productId": product["id"], "status": "baseline_fallback" if baseline_fallback else "available",
            "trials": [{"id": t["id"], "config": t["config"], "status": t["run"]["status"],
                        "reason": t["run"].get("reason"), "metrics": _metrics(t["run"]["predictions"], actuals),
                        "metricsByHorizon": _metrics_by_horizon(t["run"]["predictions"], actuals),
                        "metricsByMonth": _metrics_by_period(t["run"]["predictions"], actuals, "month"),
                        "metricsByDay": _metrics_by_period(t["run"]["predictions"], actuals, "day"),
                        "metricsByWeek": _metrics_by_period(t["run"]["predictions"], actuals, "week"),
                        "metricsBySeason": _metrics_by_period(t["run"]["predictions"], actuals, "season"),
                        "predictions": t["run"]["predictions"], "fits": t["run"]["fits"],
                        "diagnostics": t["run"]["diagnostics"],
                        **{k: v for k, v in t.items() if k in ("comparison", "selected", "baselineMetrics", "pairedDm")}}
                       for t in trials],
            "baselinePredictions": baseline, "frozenWinner": frozen,
            "selection": {"s1Changed": s1_changed, "s2Changed": s2_changed, "s3Changed": s3_changed,
                          "baselineFallback": baseline_fallback, "stage4": s4},
            "diagnostics": product.get("diagnostics", {})}

def run_development(panel: dict[str, Any], start: dt.date = dt.date(2015, 1, 1),
                    end: dt.date = dt.date(2015, 12, 31), min_days: int = MIN_TRAIN_DAYS,
                    publication_path: str | Path | None = "docs/publication-evidence.json",
                    progress_callback: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    validate_range(start, end)
    if start != dt.date(2015, 1, 1) or end != dt.date(2015, 12, 31):
        raise ValueError("selection development range is fixed to 2015-01-01..2015-12-31")
    if min_days < 1: raise ValueError("min_days must be positive")
    evidence, evidence_meta = _publication_evidence(publication_path)
    products = [_summarize_product(p, start, end, min_days, evidence, progress_callback) for p in panel["products"]]
    frozen = {p["productId"]: {"winner": p["frozenWinner"], "e0": {"model": "hist_gb", "features": "core",
               "target": "log_ratio", "grouping": "pooled", "cadence": "monthly"},
               "selection": p["selection"]} for p in products}
    output = {"schemaVersion": 1, "mode": "development",
              "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
              "protocol": PROTOCOL, "protocolSha256": _json_hash(PROTOCOL),
              "developmentRange": {"start": start.isoformat(), "end": end.isoformat()},
              "pricePanelSha256": _panel_hash(panel),
              "selectionFrozen": True, "holdoutEvaluated": False,
              "panel": {k: panel[k] for k in ("panelStart", "panelEnd", "marketCodes", "marketNames",
                                                "sourceUrls", "selectedJobIds", "selectedRequestCount",
                                                "publicationTimeKnown")},
              "publicationEvidence": evidence_meta, "frozenSelection": frozen, "products": products,
              "publicationEvidenceSha256": (_hash_file(publication_path)
                                              if publication_path and Path(publication_path).is_file() else None),
              "limitations": ["2015 개발구간은 탐색과 모델 선택에 사용되어 독립 검증이 아닙니다.",
                              "가격 원자료는 당시 공개본을 복원하지 못한 수정본 기반 회고자료입니다.",
                              "2016-Q1 홀드아웃은 계절 제한 탐색 결과만 제공합니다."]}
    return output

def run_holdout(panel: dict[str, Any], development: dict[str, Any],
                start: dt.date = dt.date(2016, 1, 1), end: dt.date = dt.date(2016, 3, 31),
                min_days: int = MIN_TRAIN_DAYS, cadences: tuple[str, ...] = CADENCES,
                progress_callback: Callable[[dict[str, Any]], None] | None = None,
                publication_values: list[dict[str, Any]] | None = None,
                publication_sha256: str | None = None,
                claim_callback: Callable[[], Path] | None = None) -> dict[str, Any]:
    validate_range(start, end)
    if start != dt.date(2016, 1, 1) or end != dt.date(2016, 3, 31):
        raise ValueError("untouched holdout range is fixed to 2016-Q1")
    if development.get("mode") != "development" or not development.get("selectionFrozen"):
        raise ValueError("holdout requires a frozen development manifest")
    if development.get("protocolSha256") != _json_hash(PROTOCOL):
        raise ValueError("development protocol hash mismatch")
    current_scope = {k: panel.get(k) for k in ("panelStart", "panelEnd", "marketCodes", "marketNames", "sourceUrls")}
    frozen_scope = {k: development.get("panel", {}).get(k) for k in current_scope}
    if current_scope != frozen_scope:
        raise ValueError("holdout panel scope differs from frozen development panel")
    if development.get("pricePanelSha256") != _panel_hash(panel):
        raise ValueError("holdout price panel contents differ from frozen development panel")
    expected_publication_hash = development.get("publicationEvidenceSha256")
    if expected_publication_hash != publication_sha256:
        raise ValueError("holdout publication evidence differs from frozen development evidence")
    if any(c not in CADENCES for c in cadences): raise ValueError("invalid cadence")
    if claim_callback is not None:
        claim_callback()
    results = []
    for product in panel["products"]:
        pid = product["id"]
        frozen = development.get("frozenSelection", {}).get(pid)
        if not frozen: raise ValueError(f"no frozen selection for {pid}")
        actuals = _actuals(product, start, end)
        series = _series(product)
        baseline = _baseline(series, start, end)
        configs = [("persistence", None), ("E0", frozen["e0"]), ("winner", frozen["winner"])]
        runs, run_cache = [], {}
        for label, config0 in configs:
            for cadence in cadences:
                if config0 is None or config0.get("model") == "persistence":
                    run = {"status": "baseline", "predictions": baseline, "fits": [],
                           "diagnostics": {"issued": len(baseline), "matched": len(baseline)}}
                    config = {"model": "persistence", "features": "none", "target": "price_level",
                              "grouping": "none", "cadence": cadence}
                else:
                    config = {**config0, "cadence": cadence}
                    key = _json_hash(config)
                    if key in run_cache:
                        run = run_cache[key]
                    else:
                        run = _fit_logged(product, config, start, end, min_days, f"holdout-{label}",
                                          progress_callback,
                                          publication_values=(publication_values
                                                              if "publication" in config["features"].split("+")
                                                              else None))
                        run_cache[key] = run
                runs.append({"id": label, "config": config, "run": run})
        comp_results = []
        for item in runs:
            parent = baseline if item["id"] != "winner" else next(
                (x["run"]["predictions"] for x in runs if x["id"] == "E0"
                 and x["config"]["cadence"] == item["config"]["cadence"]), baseline)
            comp = _comparison(item["run"]["predictions"], parent, actuals)
            comp_results.append({"id": item["id"], "config": item["config"], "status": item["run"]["status"],
                                 "predictions": item["run"]["predictions"], "fits": item["run"]["fits"],
                                 "diagnostics": item["run"]["diagnostics"], **comp})
        results.append({"productId": pid, "comparisons": comp_results})
    # Confirmatory family is exactly monthly winner-vs-E0 by product×horizon×loss.
    family = []
    for product in results:
        winner = next((r for r in product["comparisons"] if r["id"] == "winner" and r["config"]["cadence"] == "monthly"), None)
        parent = next((r for r in product["comparisons"] if r["id"] == "E0" and r["config"]["cadence"] == "monthly"), None)
        if winner and parent:
            actuals = _actuals(next(p for p in panel["products"] if p["id"] == product["productId"]), start, end)
            for h in HORIZONS:
                for loss in ("absolute_error", "absolute_percentage_error"):
                    family.append({"productId": product["productId"], "horizon": h, "loss": loss,
                                   **_dm(winner["predictions"], parent["predictions"], actuals, h, loss)})
    _apply_holm(family)
    return {"schemaVersion": 1, "mode": "holdout",
            "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "protocol": PROTOCOL, "protocolSha256": _json_hash(PROTOCOL),
            "developmentManifestSha256": _json_hash(development),
            "pricePanelSha256": _panel_hash(panel),
            "publicationEvidenceSha256": publication_sha256,
            "holdoutRange": {"start": start.isoformat(), "end": end.isoformat()},
            "selectionFrozen": True, "selectionReopened": False,
            "panel": {k: panel[k] for k in ("panelStart", "panelEnd", "marketCodes", "marketNames",
                                              "sourceUrls", "selectedJobIds", "selectedRequestCount",
                                              "publicationTimeKnown")},
            "products": results,
            "confirmatoryDmFamily": family,
            "limitations": ["2016-Q1은 계절 제한 탐색 평가이며 연중·다년 성능의 증거가 아닙니다.",
                            "cadence 결과는 보고용이며 후보 선택을 바꾸지 않습니다.",
                            "가격 자료의 과거 공개시점은 복원되지 않았습니다."]}

def _comparison(candidate: list[dict[str, Any]], parent: list[dict[str, Any]],
                actuals: dict[tuple[str, int], float]) -> dict[str, Any]:
    c = {(r["target"], int(r["horizon"])): r for r in candidate}
    p = {(r["target"], int(r["horizon"])): r for r in parent}
    common = set(c) & set(p) & set(actuals)
    return {"metricsSamePairs": _metrics([c[k] for k in common], actuals, common),
            "parentMetricsSamePairs": _metrics([p[k] for k in common], actuals, common),
            "metricsSamePairsByHorizon": {str(h): {"candidate": _metrics([c[k] for k in common if k[1] == h], actuals,
                                                                              {k for k in common if k[1] == h}),
                                                      "parent": _metrics([p[k] for k in common if k[1] == h], actuals,
                                                                          {k for k in common if k[1] == h})}
                                           for h in HORIZONS},
            "coverageCandidateCount": len(set(c)&set(actuals)),
            "coverageCandidateRate": len(set(c)&set(actuals))/max(1, len(actuals)),
            "coverageParentCount": len(set(p)&set(actuals)),
            "coverageParentRate": len(set(p)&set(actuals))/max(1, len(actuals)),
            "candidateCoverageAll": _metrics(candidate, actuals),
            "candidateMetricsByDay": _metrics_by_period(candidate, actuals, "day"),
            "candidateMetricsByWeek": _metrics_by_period(candidate, actuals, "week"),
            "candidateMetricsByMonth": _metrics_by_period(candidate, actuals, "month"),
            "candidateMetricsBySeason": _metrics_by_period(candidate, actuals, "season")}

def _apply_holm(rows: list[dict[str, Any]]) -> None:
    eligible = sorted((r for r in rows if r.get("pValue") is not None), key=lambda r:r["pValue"])
    # The family is fixed at 5 products × 10 horizons × 2 losses. A test that
    # cannot be computed remains in the family as a conservative p=1.
    running, m = 0.0, len(rows)
    for i, row in enumerate(eligible):
        running = max(running, min(1.0, (m-i)*row["pValue"]))
        row["holmAdjustedPValue"] = running
        row["holmSignificantAt005"] = running < .05

def _claim_holdout(development_file: str | Path, output_file: str | Path,
                   development: dict[str, Any]) -> Path:
    dev_hash = _json_hash(development)
    claim = Path(development_file).resolve().parent / f".holdout-{dev_hash}.claim.json"
    claim.parent.mkdir(parents=True, exist_ok=True)
    payload = {"status": "running", "developmentManifestSha256": dev_hash,
               "protocolSha256": development.get("protocolSha256"),
               "output": str(Path(output_file).resolve()),
               "startedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(claim, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try: claim.unlink()
        except OSError: pass
        raise
    return claim

def _update_claim(path: Path | None, status: str, error_type: str | None = None) -> None:
    if path is None:
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["status"] = status
    payload["finishedAt"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if error_type:
        payload["errorType"] = error_type
    dashboard.write_json_atomic(path, payload)

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen staged agricultural price experiments.")
    parser.add_argument("--db", default="data/agri.sqlite")
    parser.add_argument("--mode", choices=("development", "holdout"), default="development")
    parser.add_argument("--output", default="outputs/experiments-development.json")
    parser.add_argument("--development-file")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--panel-start", default="2014-01-01")
    parser.add_argument("--panel-end", default="2016-03-31")
    parser.add_argument("--cadences", default="daily,weekly,monthly,seasonal")
    parser.add_argument("--publication-evidence", default="docs/publication-evidence.json")
    parser.add_argument("--baseline-dashboard", default="outputs/prices-dashboard.json",
                        help="frozen schema v1 output used to verify E0 target equivalence")
    args = parser.parse_args(argv)
    db = None
    claim_path: Path | None = None
    try:
        default_start, default_end = (("2015-01-01", "2015-12-31") if args.mode == "development"
                                      else ("2016-01-01", "2016-03-31"))
        start = parse_day(args.start_date or default_start)
        end = parse_day(args.end_date or default_end)
        pstart, pend = parse_day(args.panel_start), parse_day(args.panel_end)
        validate_range(start, end); validate_range(pstart, pend)
        if Path(args.db).resolve() == Path(args.output).resolve():
            raise ValueError("output path must differ from the read-only database path")
        if Path(args.output).exists():
            raise ValueError("refusing to overwrite an existing experiment output")
        if args.development_file and Path(args.development_file).resolve() == Path(args.output).resolve():
            raise ValueError("holdout output must not overwrite the frozen development manifest")
        db = coverage.open_readonly(args.db)
        panel = build_price_panel(db, pstart, pend)
        baseline_dashboard = json.loads(Path(args.baseline_dashboard).read_text(encoding="utf-8"))
        target_equivalence = verify_target_equivalence(panel, baseline_dashboard)
        target_equivalence_hash = _json_hash(target_equivalence)
        input_hash = _hash_file(args.db)
        def progress(info: dict[str, Any]) -> None:
            if info["state"] == "started":
                print(f"experiment start product={info['productId']} stage={info['stage']} model={info['model']} "
                      f"features={info['features']} target={info['target']} group={info['grouping']} "
                      f"cadence={info['cadence']}", file=sys.stderr, flush=True)
            else:
                print(f"experiment done product={info['productId']} stage={info['stage']} fits={info['fits']} "
                      f"matched_pairs={info['matchedPairs']} elapsed_s={info['elapsedSeconds']:.1f}",
                      file=sys.stderr, flush=True)
        if args.mode == "development":
            result = run_development(panel, start, end, MIN_TRAIN_DAYS, args.publication_evidence, progress)
        else:
            if not args.development_file: raise ValueError("--development-file is required for holdout mode")
            dev = json.loads(Path(args.development_file).read_text(encoding="utf-8"))
            if dev.get("targetEquivalenceSha256") != target_equivalence_hash:
                raise ValueError("holdout E0 target-equivalence proof differs from frozen development proof")
            pub_values, _ = _publication_evidence(args.publication_evidence)
            pub_hash = _hash_file(args.publication_evidence) if Path(args.publication_evidence).is_file() else None
            cadence_list = tuple(x.strip() for x in args.cadences.split(",") if x.strip())
            def claim_once() -> Path:
                nonlocal claim_path
                claim_path = _claim_holdout(args.development_file, args.output, dev)
                return claim_path
            result = run_holdout(panel, dev, start, end, MIN_TRAIN_DAYS, cadence_list, progress,
                                 pub_values, pub_hash, claim_once)
        result["inputDbSha256"] = input_hash
        result["targetEquivalence"] = target_equivalence
        result["targetEquivalenceSha256"] = target_equivalence_hash
        dashboard.write_json_atomic(args.output, result)
        if claim_path is not None:
            _update_claim(claim_path, "completed")
    except (OSError, sqlite3.Error, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        if claim_path is not None:
            try: _update_claim(claim_path, "failed", type(exc).__name__)
            except (OSError, ValueError, json.JSONDecodeError): pass
        print(f"experiment run failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if db is not None: db.close()
    return 0

def _hash_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""): digest.update(chunk)
    return digest.hexdigest()

if __name__ == "__main__":
    raise SystemExit(main())
