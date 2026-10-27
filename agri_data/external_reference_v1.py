"""Causal within-product price-path experiment inspired by external references.

The original competition code is not copied. This adapter keeps the current
representative-price target and basic40 feature set, then adds supported
market/subclass/grade price paths as masked causal inputs. It separates
answer-free forecast construction from scoring and never calls an API.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import platform
import sqlite3
from bisect import bisect_right
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from . import (dashboard, modeling, equal_info_online_score as inference,
               research_progress_diagnostic as bootstrap)
from . import chronos2_market_context as rhs_source
from .chronos2_market_context import BASELINE_PATHS, BASELINE_SHA256, PHASES

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data/agri.sqlite"
PRICES_PATH = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
OUT = ROOT / "outputs/experiments/external-reference-v1-20261012"
TRAIN_START = dt.date(2014, 1, 1)
PRODUCTS = tuple(dashboard.PRODUCTS)[:4]
HORIZONS = tuple(range(1, 11))
CHANNEL_SUPPORT_END = dt.date(2026, 3, 31)  # frozen before V
MIN_CHANNEL_DAYS = 120
MAX_CHANNELS = 8
CHANNEL_LAGS = (0, 1, 2, 3, 7, 14, 28)
MAX_CHANNEL_AGE = 14
MIN_TRAIN_ROWS = 90
LOSSES = ("ae", "ape")
BANDWIDTHS = (9, 28)
BOOTSTRAP_BLOCKS = (14, 28)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 0
# Bound extraction and target reconciliation to the fixed target snapshot's
# contextEnd, not the snapshot file's later generation date.
SOURCE_END = dt.date(2026, 9, 27)
COMPARISONS = (("CAND_vs_CTRL", "CAND_A", "CTRL_LGB"),
               ("CAND_vs_RHS", "CAND_A", "E106_RHS"),
               ("CTRL_vs_RHS", "CTRL_LGB", "E106_RHS"))
ALL_CELLS = frozenset((p, h) for p in PRODUCTS for h in HORIZONS)

LIGHTGBM_CONFIG = {
    "objective": "regression", "n_estimators": 200, "num_leaves": 15,
    "learning_rate": 0.03, "min_child_samples": 20,
    "reg_lambda": 1.0, "random_state": 0, "n_jobs": 1,
    "deterministic": True, "force_col_wise": True, "verbosity": -1,
}


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _price_from_log_ratio(p0: float, log_ratio: float) -> float:
    if not math.isfinite(p0) or p0 <= 0 or not math.isfinite(log_ratio):
        raise ValueError("p0 and predicted log ratio must be finite; p0 must be positive")
    log_price = math.log(p0) + log_ratio
    try:
        price = math.exp(log_price)
    except OverflowError as exc:
        raise ValueError("predicted price overflows floating-point range") from exc
    if not math.isfinite(price) or price <= 0:
        raise ValueError("predicted price is not finite and positive")
    return price


def _safe_date(value: Any) -> dt.date | None:
    try:
        return dt.date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def _decimal(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        v = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return v if v.is_finite() else None


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _product(row: Mapping[str, Any], query: Mapping[str, Any]) -> str | None:
    product = dashboard._query_product(query)
    return product if product in PRODUCTS else None


def _channel_key(market: str, subclass: str, grade: str,
                 grain: str) -> tuple[str, ...]:
    if grain == "market_subclass_grade":
        return market, subclass, grade
    if grain == "market_subclass":
        return market, subclass
    if grain == "market":
        return (market,)
    raise ValueError("unsupported channel grain")


def _transactions_to_channels(rows: Iterable[Mapping[str, Any]]) -> tuple[dict[str, dict[tuple[str, ...], dict[dt.date, dict[str, Any]]]], dict[str, Any]]:
    """Aggregate selected raw transactions without dropping signed values."""
    buckets: dict[tuple[str, tuple[str, ...], dt.date], dict[str, Any]] = {}
    market_totals: dict[tuple[str, str, dt.date], dict[str, Any]] = {}
    label_history: dict[tuple[str, tuple[str, ...]], dict[tuple[str, str], list[str]]] = defaultdict(dict)
    quality_flags: Counter[str] = Counter()
    states: dict[tuple[str, dt.date, str, str], str] = {}
    garlic_specific: set[tuple[dt.date, str, str]] = set()
    provenance: set[str] = set()
    source_urls: set[str] = set()
    retrieval_times: set[str] = set()
    query_state_counts: Counter[str] = Counter()
    query_manifest: dict[tuple[str, dt.date, str, str], dict[str, Any]] = {}
    excluded: Counter[str] = Counter()
    negative_adjustment_rows = 0
    negative_adjustments: list[dict[str, Any]] = []
    invalid_cells: Counter[str] = Counter()
    cell_issues: list[dict[str, Any]] = []
    request_ids: set[str] = set()

    for source in rows:
        query = source.get("query")
        row = source.get("payload")
        if not isinstance(query, Mapping):
            excluded["invalid_query"] += 1
            continue
        product = _product(row if isinstance(row, Mapping) else {}, query)
        day = _safe_date(query.get("cond[trd_clcln_ymd::EQ]"))
        market = str(query.get("cond[whsl_mrkt_cd::EQ]", ""))
        if product not in PRODUCTS or day is None or market not in dashboard.MARKETS:
            continue
        subclass_query = str(query.get("cond[gds_sclsf_cd::EQ]", ""))
        state = str(source.get("selectedState", "unavailable"))
        states[(product, day, market, subclass_query)] = state
        query_key = (product, day, market, subclass_query)
        query_entry = query_manifest.setdefault(query_key, {
            "product": product, "day": day.isoformat(), "market": market,
            "subclassQuery": subclass_query, "selectedState": state,
            "latestState": source.get("latestState"), "latestEpochId": source.get("latestEpochId"),
            "selectedEpochId": source.get("selectedEpochId"), "latestJobId": source.get("latestJobId"),
            "selectedJobId": source.get("selectedJobId"), "selectedPrevious": bool(source.get("selectedPrevious")),
            "excludedEpochCandidates": source.get("excludedEpochCandidates"), "sourceUrl": source.get("sourceUrl"),
            "taskKey": source.get("taskKey"),
            "pages": []})
        if source.get("requestId") is not None:
            page = {"requestId": source.get("requestId"), "pageNo": source.get("pageNo"),
                    "retrievedAt": source.get("retrievedAt"), "apiStatus": source.get("apiStatus")}
            if page not in query_entry["pages"]:
                query_entry["pages"].append(page)
        if product == "garlic" and subclass_query in dashboard.PEELED_GARLIC_LABELS and state in ("complete", "empty"):
            garlic_specific.add((day, market, subclass_query))
        if source.get("sourceUrl"):
            if source.get("requestId") is not None:
                provenance.add(str(source["requestId"]))
            source_urls.add(str(source["sourceUrl"]))
        if source.get("retrievedAt"):
            retrieval_times.add(str(source["retrievedAt"]))
        if not isinstance(row, Mapping):
            continue
        if dashboard._product_id(row) != product:
            excluded["category_mismatch"] += 1
            continue
        subclass = str(row.get("gds_sclsf_cd") or "(missing)")
        if product == "garlic":
            if not dashboard._garlic_class_is_verified(row):
                excluded["unverified_garlic_class"] += 1
                continue
            if subclass_query and subclass != subclass_query:
                excluded["subclass_query_mismatch"] += 1
                continue
            if not subclass_query and (day, market, subclass) in garlic_specific:
                excluded["broad_duplicate_of_specific_scope"] += 1
                continue
        if row.get("trd_se") != "경매" or row.get("unit_nm") != "kg":
            excluded["not_auction_kg"] += 1
            continue
        if str(row.get("whsl_mrkt_cd", "")) != market or _safe_date(row.get("trd_clcln_ymd")) != day:
            excluded["source_scope_mismatch"] += 1
            continue
        amount, volume = _decimal(row.get("totprc")), _decimal(row.get("unit_tot_qty"))
        if amount is None or volume is None:
            excluded["invalid_amount_or_volume"] += 1
            invalid_cells["null_or_invalid_amount_or_volume"] += 1
            cell_issues.append({"product": product, "market": market, "day": day.isoformat(),
                                "subclass": str(row.get("gds_sclsf_cd") or "(missing)"),
                                "grade": str(row.get("grd_cd") or "(missing)"),
                                "requestId": source.get("requestId"),
                                "transactionId": row.get("trd_no", row.get("trns_no")),
                                "reason": "null_or_invalid_amount_or_volume",
                                "rawAmount": row.get("totprc"), "rawVolumeKg": row.get("unit_tot_qty")})
            continue
        if amount < 0 or volume < 0:
            negative_adjustment_rows += 1
            negative_adjustments.append({"product": product, "day": day.isoformat(), "market": market,
                                         "subclass": str(row.get("gds_sclsf_cd") or "(missing)"),
                                         "grade": str(row.get("grd_cd") or "(missing)"),
                                         "requestId": source.get("requestId"),
                                         "transactionId": row.get("trd_no", row.get("trns_no")),
                                         "rawAmount": row.get("totprc"), "rawVolumeKg": row.get("unit_tot_qty")})
        grade = str(row.get("grd_cd") or "(missing)")
        subclass_name = str(row.get("gds_sclsf_nm") or "(missing)")
        grade_name = str(row.get("grd_nm") or "(missing)")
        flags = source.get("qualityFlags", [])
        if isinstance(flags, list):
            for flag in flags:
                quality_flags[str(flag)] += 1
        total = market_totals.setdefault((product, market, day),
                                         {"amount": Decimal(0), "volume": Decimal(0), "rows": 0})
        total["amount"] += amount
        total["volume"] += volume
        total["rows"] += 1
        for grain in ("market_subclass_grade", "market_subclass", "market"):
            key = (product, _channel_key(market, subclass, grade, grain), day)
            cell = buckets.setdefault(key, {"amount": Decimal(0), "volume": Decimal(0), "rows": 0})
            cell["amount"] += amount
            cell["volume"] += volume
            cell["rows"] += 1
            labels = label_history[(product, key[1])]
            span = labels.setdefault((subclass_name, grade_name), [day.isoformat(), day.isoformat()])
            span[0] = min(span[0], day.isoformat())
            span[1] = max(span[1], day.isoformat())

    channels: dict[str, dict[tuple[str, ...], dict[dt.date, dict[str, Any]]]] = {
        p: defaultdict(dict) for p in PRODUCTS
    }
    for (product, key, day), cell in buckets.items():
        # Require a complete broad item query in that market-day. This avoids
        # treating a partial category page as a complete channel observation.
        if states.get((product, day, key[0], ""), "unplanned") not in ("complete", "empty"):
            invalid_cells["incomplete_broad_query"] += 1
            cell_issues.append({"product": product, "channelId": list(key), "day": day.isoformat(),
                                "signedAmount": str(cell["amount"]), "signedVolumeKg": str(cell["volume"]),
                                "rows": int(cell["rows"]), "reason": "incomplete_broad_query"})
            continue
        volume = cell["volume"]
        if volume <= 0:
            reason = "nonpositive_net_volume"
            invalid_cells[reason] += 1
            cell_issues.append({"product": product, "channelId": list(key), "day": day.isoformat(),
                                "signedAmount": str(cell["amount"]), "signedVolumeKg": str(volume),
                                "rows": int(cell["rows"]), "reason": reason})
            continue
        price = cell["amount"] / volume
        if price <= 0:
            reason = "nonpositive_net_amount_or_price"
            invalid_cells[reason] += 1
            cell_issues.append({"product": product, "channelId": list(key), "day": day.isoformat(),
                                "signedAmount": str(cell["amount"]), "signedVolumeKg": str(volume),
                                "signedPrice": str(price), "rows": int(cell["rows"]), "reason": reason})
            continue
        price_float = float(price)
        if not math.isfinite(price_float):
            excluded["nonfinite_channel_price"] += 1
            invalid_cells["nonfinite_channel_price"] += 1
            cell_issues.append({"product": product, "channelId": list(key), "day": day.isoformat(),
                                "signedAmount": str(cell["amount"]), "signedVolumeKg": str(volume),
                                "rows": int(cell["rows"]), "reason": "nonfinite_channel_price"})
            continue
        channels[product][key][day] = {
            "price": price_float, "amountSigned": str(cell["amount"]),
            "volumeKgSigned": str(volume), "rows": int(cell["rows"]),
        }
    result = {p: dict(v) for p, v in channels.items()}
    for query_record in query_manifest.values():
        query_state_counts[str(query_record["selectedState"])] += 1
    audit = {"excludedRows": dict(sorted(excluded.items())),
             "selectedRequestProvenanceCount": len(provenance),
             "selectedRequestIds": sorted(provenance),
             "sourceUrls": sorted(source_urls),
             "retrievalTimeRange": [min(retrieval_times), max(retrieval_times)] if retrieval_times else None,
             "selectedQueryStates": dict(sorted(query_state_counts.items())),
             "selectedQueryManifest": sorted(query_manifest.values(), key=lambda q: (
                 q["product"], q["day"], q["market"], q["subclassQuery"])),
             "publicationTimeKnown": False,
             "effectiveCodeValidity": "unknown; only observed first/last dates are recorded",
             "channelLabelHistory": {f"{product}/{'/'.join(key)}": [
                 {"subclassName": names[0], "gradeName": names[1],
                  "firstObservedDate": dates[0], "lastObservedDate": dates[1]}
                 for names, dates in sorted(values.items())]
                 for (product, key), values in sorted(label_history.items())},
             "qualityFlagCounts": dict(sorted(quality_flags.items())),
             "negativeAdjustmentRows": negative_adjustment_rows,
             "negativeAdjustments": negative_adjustments,
             "cellIssueCounts": dict(sorted(invalid_cells.items())),
             "cellIssues": cell_issues,
             "aggregation": "signed amount sum / signed net kg sum; positive net kg and positive finite price required"}
    target_totals = {}
    target_issues = []
    for product in PRODUCTS:
        seen_days = {day for (p, _, day) in market_totals if p == product}
        for day in seen_days:
            # `states` uses subquery as the fourth key; require a completed broad query
            # for both markets before asserting daily target reconciliation.
            broad_states = [states.get((product, day, market, ""), "unplanned")
                            for market in dashboard.MARKETS]
            if not all(state in ("complete", "empty") for state in broad_states):
                continue
            amount_sum = Decimal(0)
            volume_sum = Decimal(0)
            row_count = 0
            for market in dashboard.MARKETS:
                part = market_totals.get((product, market, day))
                if part:
                    amount_sum += part["amount"]
                    volume_sum += part["volume"]
                    row_count += part["rows"]
            if volume_sum > 0 and row_count > 0 and amount_sum > 0:
                target_price = float(amount_sum / volume_sum)
                if not math.isfinite(target_price):
                    raise ValueError(f"raw target price is outside finite float range: {product}/{day}")
                target_totals[f"{product}/{day.isoformat()}"] = {
                    "signedAmount": str(amount_sum), "signedVolumeKg": str(volume_sum),
                    "price": target_price, "rowCount": row_count,
                    "marketStates": broad_states}
            elif row_count > 0:
                reason = ("nonpositive_daily_net_volume" if volume_sum <= 0 else
                          "nonpositive_daily_net_amount_or_price" if amount_sum <= 0 else
                          "no_daily_transaction_rows")
                target_issues.append({"product": product, "day": day.isoformat(),
                                      "signedAmount": str(amount_sum), "signedVolumeKg": str(volume_sum),
                                      "rowCount": row_count, "marketStates": broad_states,
                                      "reason": reason})
    audit["rawDailyTargets"] = target_totals
    audit["rawDailyTargetIssues"] = target_issues
    return result, audit


def extract_channels(db: sqlite3.Connection, start: dt.date, end: dt.date) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read the same latest validated requests used by the existing dashboard."""
    if start > end:
        raise ValueError("channel extraction start must not follow end")
    db.row_factory = sqlite3.Row
    db.create_function("coverage_task_key", 3, dashboard.coverage._request_task_key, deterministic=True)
    digest = hashlib.sha256()
    selected_rows = 0

    def iter_rows():
        nonlocal selected_rows
        for rec in db.execute(dashboard.PRICE_SQL, (start.isoformat(), end.isoformat(), *dashboard.MARKETS.keys())):
            query = dashboard._json_object(rec["query_json"])
            if not isinstance(query, Mapping):
                continue
            payload = dashboard._json_object(rec["payload_json"]) if rec["payload_json"] else None
            row = {"query": query, "payload": payload,
                   "selectedState": rec["selected_state"], "requestId": rec["request_id"],
                   "retrievedAt": rec["retrieved_at"], "sourceUrl": rec["source_url"],
                   "qualityFlags": _json_list(rec["quality_flags"]),
                   "pageNo": rec["page_no"], "apiStatus": rec["api_status"],
                   "latestState": rec["latest_state"], "latestEpochId": rec["latest_epoch_id"],
                   "selectedEpochId": rec["selected_epoch_id"],
                   "latestJobId": rec["latest_job_id"], "selectedJobId": rec["selected_job_id"],
                   "selectedPrevious": rec["selected_previous"],
                   "excludedEpochCandidates": rec["excluded_epoch_candidates"],
                   "taskKey": rec["task_key"]}
            raw = _canonical(row)
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
            selected_rows += 1
            yield row

    channels, audit = _transactions_to_channels(iter_rows())
    audit.update({"queryRange": [start.isoformat(), end.isoformat()],
                  "selectedRawRows": selected_rows, "selectedRowsSha256": digest.hexdigest()})
    return channels, audit


def select_channels(channels: Mapping[str, Mapping[tuple[str, ...], Mapping[dt.date, Any]]],
                    support_end: dt.date = CHANNEL_SUPPORT_END,
                    min_days: int = MIN_CHANNEL_DAYS,
                    max_channels: int = MAX_CHANNELS) -> dict[str, Any]:
    """Choose channels only from pre-V observed support; deterministic fallback grain."""
    if min_days < 1 or max_channels < 1:
        raise ValueError("channel support and cap must be positive")
    out: dict[str, Any] = {}
    tiers = ((3, "market_subclass_grade"), (2, "market_subclass"), (1, "market"))
    for product in PRODUCTS:
        selected: list[dict[str, Any]] = []
        attempted = []
        for width, grain in tiers:
            supported = []
            for key, series in channels.get(product, {}).items():
                if len(key) != width:
                    continue
                days = sorted(d for d, value in series.items()
                              if d <= support_end and _finite_positive(value.get("price") if isinstance(value, Mapping) else value))
                if len(days) >= min_days:
                    supported.append((len(days), key, days))
            attempted.append({"grain": grain, "supportedCount": len(supported),
                              "minimumDistinctObservedDays": min_days})
            if supported:
                supported.sort(key=lambda item: (-item[0], item[1]))
                selected = [{"channelId": list(key), "grain": grain,
                             "supportDays": count, "supportStart": days[0].isoformat(),
                             "supportEnd": days[-1].isoformat(),
                             "firstObserved": days[0].isoformat(),
                             "lastObserved": days[-1].isoformat(),
                             "selectionCutoff": support_end.isoformat(),
                             "selectionRule": "support days descending, channel ID ascending",
                             "selectionStatus": "selected",
                             "noveltyNote": ("market-only channel overlaps prior market-context input; not evidence of new detailed dimensions"
                                             if grain == "market" else "retains within-market subclass/grade identity")}
                            for count, key, days in supported[:max_channels]]
                break
        out[product] = {"selected": selected, "attemptedGrains": attempted,
                        "status": "selected" if selected else "no_supported_channel",
                        "supportCutoff": support_end.isoformat(),
                        "maxChannels": max_channels}
    return out


def _finite_positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _channel_series(channels: Mapping[tuple[str, ...], Mapping[dt.date, Any]],
                    selection: Mapping[str, Any], product: str) -> list[tuple[tuple[str, ...], dict[dt.date, float], tuple[dt.date, ...]]]:
    result = []
    for item in selection[product]["selected"]:
        key = tuple(item["channelId"])
        series = {}
        for day, value in channels.get(product, {}).get(key, {}).items():
            price = _finite_positive(value.get("price") if isinstance(value, Mapping) else value)
            if price is not None:
                series[day] = price
        result.append((key, series, tuple(sorted(series))))
    return result


def channel_path_features(channel_series: list[tuple[tuple[str, ...], Mapping[dt.date, float]]],
                          p0: float, cutoff: dt.date) -> tuple[list[float], list[dict[str, Any]]]:
    """Return latest carried and exact-date prices with distinct masks/ages."""
    if not math.isfinite(p0) or p0 <= 0:
        raise ValueError("representative p0 must be finite and positive")
    features: list[float] = []
    source_audit: list[dict[str, Any]] = []
    for channel in channel_series:
        key, series = channel[0], channel[1]
        days = channel[2] if len(channel) > 2 else tuple(sorted(series))
        for lag in CHANNEL_LAGS:
            ref_day = cutoff - dt.timedelta(days=lag)
            source_index = bisect_right(days, ref_day) - 1
            source_day = days[source_index] if source_index >= 0 else None
            price = series.get(source_day) if source_day is not None else None
            age = (ref_day - source_day).days if source_day is not None else None
            mask = float(price is not None and age is not None and 0 <= age <= MAX_CHANNEL_AGE)
            rel_log = math.log(float(price) / p0) if mask else 0.0
            exact = _finite_positive(series.get(ref_day))
            exact_mask = float(exact is not None)
            exact_rel = math.log(exact / p0) if exact is not None else 0.0
            features.extend((rel_log, float(age or 0), mask, exact_rel, exact_mask))
            source_audit.append({"channelId": list(key), "lag": lag,
                                 "referenceDate": ref_day.isoformat(),
                                 "asOfSourceDate": source_day.isoformat() if source_day else None,
                                 "asOfAgeDays": age,
                                 "availableWithin14Days": bool(mask),
                                 "exactDateObserved": bool(exact_mask)})
    return features, source_audit


def _feature_row(series: Mapping[dt.date, tuple[float, float]],
                 channel_series: list[tuple],
                 origin: dt.date, target: dt.date, horizon: int,
                 feature_cache: dict[Any, Any], *, include_path_audit: bool = False) -> dict[str, Any] | None:
    cutoff = origin - dt.timedelta(days=1)
    base = modeling._feature_row(dict(series), cutoff, target, horizon, feature_cache)
    if base is None:
        return None
    source_day, vector = base
    if (origin - source_day).days > dashboard.MAX_SOURCE_AGE_DAYS:
        return None
    p0 = float(series[source_day][0])
    extras, source_audit = channel_path_features(channel_series, p0, cutoff)
    full = [float(v) for v in vector] + extras
    if len(vector) != 40 or not all(math.isfinite(v) for v in full):
        raise ValueError("feature vector has invalid width or nonfinite values")
    result = {"basic40": list(map(float, vector)), "candidate": full,
            "p0": p0, "p0Date": source_day.isoformat(), "cutoff": cutoff.isoformat(),
            "sourceAgeDays": (cutoff - source_day).days,
            "pathObservedCount": int(sum(extras[i] for i in range(2, len(extras), 5)))}
    if include_path_audit:
        result["pathSourceAudit"] = source_audit
    return result


def _month_start(day: dt.date) -> dt.date:
    return day.replace(day=1)


def _fixed_month_fit_origin(fit_month: dt.date, fit_keys: list[tuple[str, dt.date, int]],
                            baseline: Mapping[tuple[str, dt.date, int], Any]) -> dt.date:
    boundary = _month_start(fit_month)
    if any(baseline[key]["origin"] < boundary for key in fit_keys):
        raise ValueError("forecast origin precedes fixed monthly fit boundary")
    return boundary


def _execution_identity() -> dict[str, Any]:
    runtime = _check_runtime()
    from . import equal_info_online_forecasts as online
    direct = {"dashboard": dashboard.__file__, "modeling": modeling.__file__,
              "dashboard_coverage": dashboard.coverage.__file__,
              "equal_info_online_forecasts": online.__file__,
              "direct_horizon": online.direct_horizon.__file__,
              "equal_info_online_score": inference.__file__,
              "research_progress_diagnostic": bootstrap.__file__,
              "chronos2_market_context": rhs_source.__file__}
    dependencies = {name: _file_sha(Path(path)) for name, path in direct.items()}
    import lightgbm, numpy, scipy, joblib
    dependencies.update({name: _file_sha(Path(module.__file__)) for name, module in
                         (("lightgbm", lightgbm), ("numpy", numpy), ("scipy", scipy), ("joblib", joblib))})
    return {"sourceCodeSha256": _file_sha(Path(__file__)),
            "protocolSha256": _sha(_canonical(protocol())), "runtime": runtime,
            "dependencyInitFileSha256": dependencies,
            "modelConfigSha256": _sha(_canonical(LIGHTGBM_CONFIG))}


def _estimator_factory(**config: Any):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:
        raise RuntimeError("lightgbm is required for CTRL_LGB/CAND_A") from exc
    return LGBMRegressor(**config)


def _fit_pair(train_rows: list[dict[str, Any]], estimator_factory: Callable[..., Any] | None = None):
    factory = estimator_factory or _estimator_factory
    y = [r["label"] for r in train_rows]
    ctrl = factory(**LIGHTGBM_CONFIG)
    cand = factory(**LIGHTGBM_CONFIG)
    ctrl.fit([r["basic40"] for r in train_rows], y)
    cand.fit([r["candidate"] for r in train_rows], y)
    return ctrl, cand


def _load_or_fit_pair(train_rows: list[dict[str, Any]], model_path: Path,
                      identity: Mapping[str, Any], estimator_factory: Callable[..., Any] | None = None):
    """Reuse exact fold checkpoints after interruption; reject stale/mismatched files."""
    import joblib
    targets_digest = _sha(_canonical([{"target": r["target"].isoformat(), "label": r["label"],
                                       "basic40": r["basic40"], "candidate": r["candidate"]}
                                      for r in train_rows]))
    checkpoint_id = {**identity, "executionIdentity": _execution_identity(),
                      "trainingRows": len(train_rows),
                      "trainingTargetsSha256": targets_digest,
                      "modelConfig": LIGHTGBM_CONFIG}
    if model_path.is_file():
        bundle = joblib.load(model_path)
        if bundle.get("checkpointIdentity") != checkpoint_id:
            raise ValueError(f"existing model checkpoint does not match frozen fold: {model_path}")
        return bundle["CTRL_LGB"], bundle["CAND_A"], _file_sha(model_path), "reused", _sha(_canonical(checkpoint_id))
    ctrl, cand = _fit_pair(train_rows, estimator_factory)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    temp = model_path.with_suffix(model_path.suffix + ".tmp")
    joblib.dump({"checkpointIdentity": checkpoint_id, "CTRL_LGB": ctrl, "CAND_A": cand}, temp)
    if model_path.exists():
        temp.unlink(missing_ok=True)
        raise ValueError("model checkpoint appeared during write; inspect before resuming")
    temp.replace(model_path)
    return ctrl, cand, _file_sha(model_path), "fit", _sha(_canonical(checkpoint_id))


def _baseline_index(doc: Mapping[str, Any], phase: str) -> dict[tuple[str, dt.date, int], dict[str, Any]]:
    if phase not in PHASES or doc.get("phase") != phase or not isinstance(doc.get("forecasts"), list):
        raise ValueError("invalid saved RHS phase manifest")
    rows = {}
    for row in doc["forecasts"]:
        try:
            product = row["product"]
            origin = dt.date.fromisoformat(row.get("originDate", row.get("origin")))
            target = dt.date.fromisoformat(row.get("targetDate", row.get("target")))
            horizon = row["horizon"]
            conditions = row["conditions"]
            rhs = conditions["C"]
            e106 = conditions["B"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("saved RHS row is missing key, C or B") from exc
        if product not in PRODUCTS or type(horizon) is not int or horizon not in HORIZONS or target-origin != dt.timedelta(days=horizon):
            continue
        start, end, origin_floor = PHASES[phase]
        if not (start <= target <= end and origin >= origin_floor):
            continue
        key = product, target, horizon
        if key in rows:
            raise ValueError(f"duplicate RHS baseline key {key}")
        rows[key] = {"product": product, "origin": origin, "target": target,
                     "horizon": horizon,
                     "E106_RHS": _baseline_value(rhs), "E106": _baseline_value(e106),
                     "rhsStatus": rhs.get("status"), "e106Status": e106.get("status")}
    if not rows:
        raise ValueError("saved RHS manifest contains no eligible forecast rows")
    return rows


def _baseline_value(row: Mapping[str, Any]) -> float | None:
    if row.get("status") not in ("issued", "fallback"):
        return None
    value = _finite_positive(row.get("blendPrediction"))
    return value


def _actuals_from_snapshot(prices: Mapping[str, Any]) -> dict[tuple[str, dt.date], Any]:
    actuals = {}
    for item in prices.get("products", []):
        product = item.get("id")
        if product not in PRODUCTS:
            continue
        for row in item.get("observations", []):
            day = _safe_date(row.get("date"))
            if day is not None:
                actuals[(product, day)] = row.get("actual")
    return actuals


def build_forecasts(phase: str, baseline: Mapping[str, Any], prices: Mapping[str, Any],
                    channel_data: Mapping[str, Mapping[tuple[str, ...], Mapping[dt.date, Any]]],
                    channel_selection: Mapping[str, Any], *, out_dir: Path = OUT,
                    estimator_factory: Callable[..., Any] | None = None,
                    source_cache_sha256: str | None = None) -> dict[str, Any]:
    """Fit same-row monthly LGB pairs and save answer-free phase predictions/models."""
    if phase not in PHASES:
        raise ValueError("phase must be V or T")
    _check_runtime()
    path = out_dir / phase / "forecast-manifest.json"
    if path.is_file():
        frozen = json.loads(path.read_text(encoding="utf-8"))
        validate_forecast_manifest(frozen, phase)
        if frozen.get("sourceCacheSha256") != source_cache_sha256:
            raise ValueError("existing forecast manifest belongs to a different source cache")
        if frozen.get("sourceCodeSha256") != _file_sha(Path(__file__)):
            raise ValueError("existing forecast manifest was generated by a different implementation")
        return frozen
    if prices.get("schemaVersion") != 1:
        raise ValueError("expected frozen dashboard prices schemaVersion 1")
    from . import equal_info_online_forecasts as online
    if _file_sha(PRICES_PATH) != online.PRICES_SHA256:
        raise ValueError("frozen representative price snapshot hash mismatch")
    if prices != json.loads(PRICES_PATH.read_text(encoding="utf-8")):
        raise ValueError("forecast builder prices differ from the frozen target snapshot")
    base_rows = _baseline_index(baseline, phase)
    fit_records = []
    execution_identity = _execution_identity()
    model_dir = out_dir / phase / "models"
    forecast_rows = []
    for product in PRODUCTS:
        price_rows = prices.get("products", [])
        product_record = next((p for p in price_rows if p.get("id") == product), None)
        if product_record is None:
            raise ValueError(f"price snapshot lacks {product}")
        series = modeling._observed_series(product_record)
        selected_series = _channel_series(channel_data, channel_selection, product)
        keys = sorted((key for key in base_rows if key[0] == product), key=lambda k: (base_rows[k]["origin"], k[2]))
        by_horizon: dict[int, list[tuple[str, dt.date, int]]] = defaultdict(list)
        for key in keys:
            by_horizon[key[2]].append(key)
        for horizon, horizon_keys in sorted(by_horizon.items()):
            feature_cache: dict[Any, Any] = {}
            targets = sorted(set(series) | {base_rows[k]["target"] for k in horizon_keys})
            cached: dict[dt.date, dict[str, Any] | None] = {}
            forecast_targets = {base_rows[k]["target"] for k in horizon_keys}
            for target in targets:
                origin = target - dt.timedelta(days=horizon)
                cached[target] = _feature_row(series, selected_series, origin, target, horizon, feature_cache,
                                              include_path_audit=target in forecast_targets)
            fit_groups: dict[dt.date, list[tuple[str, dt.date, int]]] = defaultdict(list)
            for key in horizon_keys:
                fit_groups[_month_start(base_rows[key]["origin"])].append(key)
            for fit_month, fit_keys in sorted(fit_groups.items()):
                fit_origin = _fixed_month_fit_origin(fit_month, fit_keys, base_rows)
                train_rows = []
                for target in sorted(series):
                    if target < TRAIN_START or target >= fit_origin:
                        continue
                    feature = cached.get(target)
                    if feature is None:
                        continue
                    train_rows.append({**feature, "target": target,
                                       "label": math.log(series[target][0] / feature["p0"])})
                audit = {"product": product, "horizon": horizon,
                         "fitMonth": fit_month.isoformat(), "fitOrigin": fit_origin.isoformat(),
                         "trainingRows": len(train_rows),
                         "trainingFirstTarget": train_rows[0]["target"].isoformat() if train_rows else None,
                         "trainingLastTarget": train_rows[-1]["target"].isoformat() if train_rows else None,
                         "trainingTargetsStrictlyBeforeFitOrigin": all(r["target"] < fit_origin for r in train_rows),
                         "modelStatus": "not_fit", "reason": None}
                models = None
                if len(train_rows) >= MIN_TRAIN_ROWS:
                    try:
                        model_dir.mkdir(parents=True, exist_ok=True)
                        model_path = model_dir / f"{product}-h{horizon:02d}-{fit_month:%Y-%m}.joblib"
                        identity = {"product": product, "horizon": horizon,
                                    "fitOrigin": fit_origin.isoformat(),
                                    "sourceCacheSha256": source_cache_sha256,
                                    "channelSelectionSha256": _sha(_canonical(channel_selection))}
                        ctrl, cand, model_hash, checkpoint_state, checkpoint_identity_sha = _load_or_fit_pair(
                            train_rows, model_path, identity, estimator_factory)
                        models = ctrl, cand
                        audit["modelStatus"] = checkpoint_state
                        audit["modelFile"] = str(model_path.relative_to(out_dir))
                        audit["modelFileSha256"] = model_hash
                        audit["checkpointIdentitySha256"] = checkpoint_identity_sha
                    except Exception as exc:
                        audit["modelStatus"] = "fit_failed"
                        audit["reason"] = type(exc).__name__
                else:
                    audit["modelStatus"] = "not_fit"
                    audit["reason"] = "fewer_than_90_shared_training_rows"
                if models is not None:
                    pass
                fit_records.append(audit)
                for key in fit_keys:
                    base = base_rows[key]
                    feature = cached.get(base["target"])
                    result: dict[str, Any] = {
                        "product": product, "origin": base["origin"].isoformat(),
                        "cutoff": (base["origin"] - dt.timedelta(days=1)).isoformat(),
                        "target": base["target"].isoformat(), "horizon": horizon,
                        "baseline": {k: base[k] for k in ("E106_RHS", "E106", "rhsStatus", "e106Status")},
                        "CTRL_LGB": {"status": "not_issued", "prediction": None},
                        "CAND_A": {"status": "not_issued", "prediction": None},
                    }
                    if models is not None and feature is not None:
                        try:
                            ctrl_log = float(models[0].predict([feature["basic40"]])[0])
                            cand_log = float(models[1].predict([feature["candidate"]])[0])
                            p0 = feature["p0"]
                            result["CTRL_LGB"] = {"status": "issued", "prediction": _price_from_log_ratio(p0, ctrl_log)}
                            result["CAND_A"] = {"status": "issued", "prediction": _price_from_log_ratio(p0, cand_log),
                                                "p0": p0, "p0Date": feature["p0Date"],
                                                "pathAvailableLagSlots": feature["pathObservedCount"],
                                                "pathSourceAudit": feature["pathSourceAudit"]}
                        except Exception as exc:
                            result["CTRL_LGB"] = {"status": "prediction_failed", "prediction": None, "reason": type(exc).__name__}
                            result["CAND_A"] = {"status": "prediction_failed", "prediction": None, "reason": type(exc).__name__}
                    elif feature is None:
                        result["CTRL_LGB"]["reason"] = "target_cutoff_price_missing_or_stale"
                        result["CAND_A"]["reason"] = "target_cutoff_price_missing_or_stale"
                    else:
                        result["CTRL_LGB"]["reason"] = audit["reason"]
                        result["CAND_A"]["reason"] = audit["reason"]
                    forecast_rows.append(result)
    if len(fit_records) > 240:
        raise ValueError(f"phase produced {len(fit_records)} monthly folds, exceeding 240")
    forecast_rows.sort(key=lambda r: (r["product"], r["target"], r["horizon"]))
    baseline_path = BASELINE_PATHS[phase]
    if _file_sha(baseline_path) != BASELINE_SHA256[phase]:
        raise ValueError("saved RHS baseline manifest hash mismatch")
    baseline_sha = _file_sha(baseline_path)
    doc = {"schemaVersion": 1, "experiment": "external-reference-v1",
           "phase": phase, "targetValuesAttached": False,
           "protocol": protocol(), "baselineManifestSha256": baseline_sha,
           "sourceCacheSha256": source_cache_sha256,
           "sourceCodeSha256": _file_sha(Path(__file__)),
           "executionIdentity": execution_identity,
           "runtime": _check_runtime(),
           "priceSnapshotSha256": _file_sha(PRICES_PATH),
           "featureNamesByProduct": {product: list(modeling.FEATURE_NAMES) + [
               f"path[{':'.join(item['channelId'])}].lag{lag}.{suffix}"
               for item in channel_selection[product]["selected"] for lag in CHANNEL_LAGS
               for suffix in ("asOfRelativeLogPrice", "asOfAgeDays", "availableWithin14Days",
                              "exactRelativeLogPrice", "exactDateObserved")]
               for product in PRODUCTS},
           "channelSelection": dict(channel_selection),
           "forecasts": forecast_rows, "fitRecords": fit_records,
           "coverage": _coverage(forecast_rows, base_rows),
           "pairedFitCount": len(fit_records),
           "individualEstimatorFitCount": 2 * sum(r.get("modelFile") is not None and r.get("modelStatus") == "fit" for r in fit_records),
           "modelFileCount": sum(r.get("modelFile") is not None for r in fit_records)}
    doc["manifestSha256"] = _sha(_canonical(doc))
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_once(path, _canonical(doc))
    return doc


def verify_saved_model_replay(phase: str, manifest: Mapping[str, Any], prices: Mapping[str, Any],
                             channels: Mapping[str, Mapping[tuple[str, ...], Mapping[dt.date, Any]]],
                             selection: Mapping[str, Any], *, out_dir: Path = OUT) -> dict[str, Any]:
    """Reload every issued forecast's exact checkpoint and replay before scoring."""
    import joblib
    validate_forecast_manifest(manifest, phase)
    identity = _execution_identity()
    if manifest.get("executionIdentity") != identity:
        raise ValueError("forecast execution identity differs from current code/dependencies/runtime")
    if manifest.get("priceSnapshotSha256") != _file_sha(PRICES_PATH):
        raise ValueError("forecast model replay price snapshot hash mismatch")
    if prices != json.loads(PRICES_PATH.read_text(encoding="utf-8")):
        raise ValueError("forecast model replay received a different frozen price snapshot")
    checkpoint_records = {}
    for record in manifest.get("fitRecords", []):
        key = (record["product"], record["horizon"], record["fitMonth"])
        if key in checkpoint_records:
            raise ValueError(f"duplicate fold record for model replay: {key}")
        checkpoint_records[key] = record
    loaded = {}
    series_by_product = {item["id"]: modeling._observed_series(item)
                         for item in prices.get("products", []) if item.get("id") in PRODUCTS}
    channels_by_product = {product: _channel_series(channels, selection, product)
                           for product in PRODUCTS}
    feature_caches: dict[tuple[str, int], dict[Any, Any]] = defaultdict(dict)
    replayed = 0
    failures = []
    for row in manifest["forecasts"]:
        if row["CAND_A"].get("status") != "issued" or row["CTRL_LGB"].get("status") != "issued":
            continue
        origin = dt.date.fromisoformat(row["origin"])
        target = dt.date.fromisoformat(row["target"])
        fit_month = _month_start(origin).isoformat()
        fold_key = (row["product"], row["horizon"], fit_month)
        record = checkpoint_records.get(fold_key)
        if not record or record.get("modelStatus") not in ("fit", "reused"):
            failures.append({"key": fold_key, "reason": "missing_or_unusable_fold_record"})
            continue
        rel = Path(record["modelFile"])
        model_path = (out_dir / rel).resolve()
        model_root = (out_dir / phase / "models").resolve()
        if model_root not in model_path.parents:
            failures.append({"key": fold_key, "reason": "model_path_outside_model_directory"})
            continue
        if not model_path.is_file() or _file_sha(model_path) != record.get("modelFileSha256"):
            failures.append({"key": fold_key, "reason": "model_file_missing_or_hash_mismatch"})
            continue
        if model_path not in loaded:
            try:
                bundle = joblib.load(model_path)
                checkpoint_identity = bundle.get("checkpointIdentity")
                if not isinstance(checkpoint_identity, Mapping):
                    raise ValueError("checkpoint identity absent")
                if checkpoint_identity.get("executionIdentity") != identity:
                    raise ValueError("checkpoint code/runtime/dependency identity mismatch")
                if _sha(_canonical(checkpoint_identity)) != record.get("checkpointIdentitySha256"):
                    raise ValueError("checkpoint identity hash differs from forecast manifest")
                loaded[model_path] = (bundle["CTRL_LGB"], bundle["CAND_A"])
            except Exception as exc:
                failures.append({"key": fold_key, "reason": f"checkpoint_load_or_identity:{type(exc).__name__}"})
                continue
        product = row["product"]
        feature = _feature_row(series_by_product.get(product, {}), channels_by_product[product],
                               origin, target, row["horizon"],
                               feature_caches[(product, row["horizon"])])
        if feature is None:
            failures.append({"key": (product, row["target"], row["horizon"]), "reason": "feature_reconstruction_failed"})
            continue
        if row["CAND_A"].get("p0") != feature["p0"] or row["CAND_A"].get("p0Date") != feature["p0Date"]:
            failures.append({"key": (product, row["target"], row["horizon"]), "reason": "p0_reconstruction_mismatch"})
            continue
        ctrl, cand = loaded[model_path]
        try:
            replayed_values = {
                "CTRL_LGB": _price_from_log_ratio(feature["p0"], float(ctrl.predict([feature["basic40"]])[0])),
                "CAND_A": _price_from_log_ratio(feature["p0"], float(cand.predict([feature["candidate"]])[0]))}
        except Exception as exc:
            failures.append({"key": (product, row["target"], row["horizon"]),
                             "reason": f"prediction_replay_failed:{type(exc).__name__}"})
            continue
        for name, value in replayed_values.items():
            saved = row[name].get("prediction")
            if not isinstance(saved, (int, float)) or not math.isclose(value, saved, rel_tol=1e-12, abs_tol=1e-12):
                failures.append({"key": (product, row["target"], row["horizon"]),
                                 "model": name, "reason": "replayed_prediction_mismatch"})
                break
        else:
            replayed += 1
    result = {"schemaVersion": 1, "phase": phase, "status": "pass" if not failures else "fail",
              "forecastManifestSha256": manifest.get("manifestSha256"),
              "executionIdentity": identity, "modelsLoaded": len(loaded),
              "pairedForecastsReplayed": replayed, "failures": failures}
    result["verificationSha256"] = _sha(_canonical(result))
    verification_path = out_dir / phase / "model-replay-verification.json"
    _write_once(verification_path, _canonical(result))
    if failures:
        raise ValueError(f"saved model replay verification failed: {len(failures)} issue(s)")
    return result


def _coverage(rows: list[dict[str, Any]], baseline: Mapping[tuple[str, dt.date, int], Any],
              actuals: Mapping[tuple[str, dt.date], float] | None = None) -> dict[str, Any]:
    out = {}
    index = {(r["product"], dt.date.fromisoformat(r["target"]), r["horizon"]): r for r in rows}
    for product in PRODUCTS:
        for horizon in HORIZONS:
            keys = [key for key in baseline if key[0] == product and key[2] == horizon]
            part = [index.get(key) for key in keys]
            base_issued = sum(baseline[k]["E106_RHS"] is not None for k in keys)
            rhs_issued_keys = [k[1].isoformat() for k in keys if baseline[k]["E106_RHS"] is not None]
            candidate_issued_keys = [k[1].isoformat() for k, r in zip(keys, part)
                                     if r is not None and r["CAND_A"]["status"] == "issued"]
            positive_answer_keys = ([k[1].isoformat() for k in keys
                                     if _finite_positive(actuals.get((product, k[1]))) is not None]
                                    if actuals is not None else [])
            missing_answer_keys = ([k[1].isoformat() for k in keys if actuals is not None and (product, k[1]) not in actuals]
                                   if actuals is not None else [])
            nonpositive_answer_keys = ([k[1].isoformat() for k in keys
                                        if actuals is not None and (product, k[1]) in actuals and
                                        _finite_positive(actuals[(product, k[1])]) is None]
                                       if actuals is not None else [])
            out[f"{product}/{horizon}"] = {
                "rhsEligibleKeys": len(keys), "rhsIssued": base_issued,
                "rhsFallback": sum(baseline[k].get("rhsStatus") == "fallback" for k in keys),
                "rhsNotIssued": sum(baseline[k]["E106_RHS"] is None for k in keys),
                "e106Fallback": sum(baseline[k].get("e106Status") == "fallback" for k in keys),
                "e106NotIssued": sum(baseline[k]["E106"] is None for k in keys),
                "ctrlIssued": sum(r is not None and r["CTRL_LGB"]["status"] == "issued" for r in part),
                "candidateIssued": sum(r is not None and r["CAND_A"]["status"] == "issued" for r in part),
                "candidateFailed": sum(r is not None and r["CAND_A"]["status"] == "prediction_failed" for r in part),
                "candidateWithheld": sum(r is not None and r["CAND_A"]["status"] == "not_issued" for r in part),
                "candidateNotIssued": sum(r is None or r["CAND_A"]["status"] != "issued" for r in part),
                "rhsIssuedKeys": rhs_issued_keys,
                "candidateMissingRhsIssuedKeys": sorted(set(rhs_issued_keys) - set(candidate_issued_keys)),
                "candidateMissingRhsIssuedCount": len(set(rhs_issued_keys) - set(candidate_issued_keys)),
                "positiveAnswerKeys": positive_answer_keys,
                "positiveAnswerCount": len(positive_answer_keys),
                "missingAnswerKeys": missing_answer_keys,
                "missingAnswerCount": len(missing_answer_keys),
                "nonpositiveAnswerKeys": nonpositive_answer_keys,
                "nonpositiveAnswerCount": len(nonpositive_answer_keys),
                "candidateMissingPositiveAnswerKeys": sorted(set(positive_answer_keys) - set(candidate_issued_keys)),
                "candidateMissingPositiveAnswerCount": len(set(positive_answer_keys) - set(candidate_issued_keys)),
                "ctrlMissingKeys": [k[1].isoformat() for k, r in zip(keys, part) if r is None or r["CTRL_LGB"]["status"] != "issued"],
                "candidateMissingKeys": [k[1].isoformat() for k, r in zip(keys, part) if r is None or r["CAND_A"]["status"] != "issued"],
            }
    out["portfolio"] = {name: sum(v[name] for k, v in out.items() if "/" in k)
                        for name in ("rhsEligibleKeys", "rhsIssued", "rhsFallback", "rhsNotIssued",
                                     "e106Fallback", "e106NotIssued", "ctrlIssued", "candidateIssued",
                                     "candidateFailed", "candidateWithheld", "candidateNotIssued",
                                     "candidateMissingRhsIssuedCount", "positiveAnswerCount", "missingAnswerCount",
                                     "nonpositiveAnswerCount", "candidateMissingPositiveAnswerCount")}
    return out


def protocol() -> dict[str, Any]:
    return {"id": "external-reference-v1-20261012", "products": list(PRODUCTS),
            "horizons": list(HORIZONS), "phases": {k: [v[0].isoformat(), v[1].isoformat(), v[2].isoformat()] for k, v in PHASES.items()},
            "comparisons": [{"id": x[0], "left": x[1], "right": x[2]} for x in COMPARISONS],
            "target": "existing representative daily auction price: signed amount sum / signed net kg sum; current snapshot and target eligibility unchanged",
            "baseFeatures": "basic40 for all four products, frozen before V",
            "targetExpression": "log(representative actual / latest representative price available by cutoff)",
            "originAndCutoff": "target=origin+horizon; cutoff=origin-1; all paths and basic inputs end by cutoff",
            "training": {"refit": "monthly product×horizon direct LightGBM",
                         "rowCutoff": "training target < calendar month start for fit month",
                         "sharedRows": True, "minimumSharedRows": MIN_TRAIN_ROWS},
            "channelSupport": {"selectionEnd": CHANNEL_SUPPORT_END.isoformat(),
                                "minimumDistinctDays": MIN_CHANNEL_DAYS, "maximumPerProduct": MAX_CHANNELS,
                                "grainOrder": ["market×subclass×grade", "market×subclass", "market"],
                                "selection": "support descending then ID ascending; first supported grain per product"},
            "candidateInputs": {"channels": "same-product market/subclass/grade paths; current representative is still the target",
                                "lags": list(CHANNEL_LAGS),
                                "perLag": ["as-of log(channel price / representative p0)", "as-of age days",
                                           "available within 14 days mask", "exact-calendar-lag relative log price",
                                           "exact-date observed mask"],
                                "maxAgeDays": MAX_CHANNEL_AGE,
                                "missing": "masked zeros for unavailable source; no interpolation; source dates retained in forecast audit"},
            "model": {"class": "LightGBM LGBMRegressor", **LIGHTGBM_CONFIG},
            "runtime": {"python": "3.11.4", "lightgbm": "4.6.0", "numpy": "1.26.4",
                        "scipy": "1.10.1", "joblib": "1.2.0"},
            "scoring": {"pairing": "same common paired rows per comparison", "cellFamilySize": 40,
                        "DM": {"losses": list(LOSSES), "bandwidths": list(BANDWIDTHS),
                               "holmAlpha": 0.05, "familySizePerBandwidth": 240, "minimumPairs": 30},
                        "blockBootstrap": {"blocks": list(BOOTSTRAP_BLOCKS), "replicates": BOOTSTRAP_REPLICATES,
                                           "seed": BOOTSTRAP_SEED, "jointCalendarSampling": True}},
            "VGate": "CAND_A vs stored E106-RHS: 40-cell equal-weight MAPE strict decrease, MAE no increase, and no coverage loss; otherwise T is not run",
            "referenceTransfer": "independently authored feature idea; upstream repository has no LICENSE, no source code copied",
            "publicationTime": "unknown; current validated retrieval vintage only"}


def _check_runtime() -> dict[str, str]:
    versions = {"python": platform.python_version(),
                "lightgbm": importlib.metadata.version("lightgbm"),
                "numpy": importlib.metadata.version("numpy"),
                "scipy": importlib.metadata.version("scipy"),
                "joblib": importlib.metadata.version("joblib")}
    if versions != protocol()["runtime"]:
        raise RuntimeError(f"external-reference frozen runtime mismatch: {versions}")
    return versions


def verify_target_reconciliation(audit: Mapping[str, Any], prices: Mapping[str, Any],
                                 start: dt.date = TRAIN_START, end: dt.date = SOURCE_END) -> dict[str, Any]:
    """Require raw selected transactions to replay the existing daily target."""
    expected: dict[tuple[str, dt.date], tuple[float, float]] = {}
    for item in prices.get("products", []):
        product = item.get("id")
        if product not in PRODUCTS:
            continue
        for row in item.get("observations", []):
            day = _safe_date(row.get("date"))
            if day is None or not start <= day <= end:
                continue
            price = _finite_positive(row.get("actual"))
            volume = _finite_positive(row.get("volumeKg"))
            if price is None or volume is None:
                raise ValueError("frozen target contains invalid price or net volume")
            expected[(product, day)] = price, volume
    raw = audit.get("rawDailyTargets")
    if not isinstance(raw, Mapping):
        raise ValueError("raw extraction lacks daily reconciliation totals")
    actual = {}
    for key, row in raw.items():
        product, day_raw = key.split("/", 1)
        day = _safe_date(day_raw)
        if product in PRODUCTS and day and start <= day <= end:
            actual[(product, day)] = (_finite_positive(row.get("price")),
                                      _finite_positive(row.get("signedVolumeKg")))
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        fmt = lambda keys: [f"{product}/{day.isoformat()}" for product, day in keys[:10]]
        raise ValueError(f"raw target dates differ from frozen target: missing={len(missing)} "
                         f"examples={fmt(missing)}, extra={len(extra)} examples={fmt(extra)}; "
                         f"sourceRange={start.isoformat()}..{end.isoformat()}")
    for key in sorted(expected):
        for label, got, want in (("price", actual[key][0], expected[key][0]),
                                 ("volumeKg", actual[key][1], expected[key][1])):
            if got is None or not math.isclose(got, want, rel_tol=1e-10, abs_tol=1e-7):
                raise ValueError(f"raw target {label} mismatch for {key[0]}/{key[1]}: {got} != {want}")
    return {"status": "verified", "start": start.isoformat(), "end": end.isoformat(),
            "observationsByProduct": {p: sum(1 for product, _ in expected if product == p) for p in PRODUCTS},
            "targetObservations": len(expected), "tolerance": {"absolute": 1e-7, "relative": 1e-10}}


def validate_snapshot_range(prices: Mapping[str, Any], *, start: dt.date = TRAIN_START,
                            end: dt.date = SOURCE_END) -> dict[str, str]:
    """Bind source extraction bounds to the frozen target snapshot metadata."""
    meta = prices.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError("frozen target snapshot lacks range metadata")
    snapshot_start = _safe_date(meta.get("contextStart"))
    snapshot_end = _safe_date(meta.get("contextEnd"))
    if snapshot_start != start or snapshot_end != end:
        raise ValueError("requested raw-source range differs from frozen snapshot contextStart/contextEnd")
    if meta.get("evaluationEnd") != end.isoformat():
        raise ValueError("frozen target evaluationEnd differs from its contextEnd")
    return {"contextStart": snapshot_start.isoformat(), "contextEnd": snapshot_end.isoformat()}


def _serialize_channels(channels: Mapping[str, Mapping[tuple[str, ...], Mapping[dt.date, Any]]],
                        selection: Mapping[str, Any]) -> dict[str, Any]:
    data = {}
    for product in PRODUCTS:
        data[product] = []
        for item in selection[product]["selected"]:
            key = tuple(item["channelId"])
            observations = []
            for day, value in sorted(channels.get(product, {}).get(key, {}).items()):
                record = dict(value) if isinstance(value, Mapping) else {"price": value}
                observations.append({"date": day.isoformat(), **record})
            data[product].append({"channelId": list(key), "grain": item["grain"],
                                  "observations": observations})
    return data


def _deserialize_channels(serialized: Mapping[str, Any]) -> dict[str, dict[tuple[str, ...], dict[dt.date, Any]]]:
    result = {}
    for product in PRODUCTS:
        channels = {}
        for channel in serialized.get(product, []):
            key = tuple(channel["channelId"])
            channels[key] = {dt.date.fromisoformat(row["date"]): {k: v for k, v in row.items() if k != "date"}
                              for row in channel["observations"]}
        result[product] = channels
    return result


def preflight(*, db_path: Path = DB_PATH, prices_path: Path = PRICES_PATH,
              output_dir: Path = OUT, end: dt.date = SOURCE_END) -> dict[str, Any]:
    """Extract once, verify target replay, freeze training-only channel support/cache."""
    prices_raw = prices_path.read_bytes()
    prices = json.loads(prices_raw)
    from . import equal_info_online_forecasts as online
    if _sha(prices_raw) != online.PRICES_SHA256:
        raise ValueError("frozen representative price snapshot hash mismatch")
    if end > SOURCE_END:
        raise ValueError("raw extraction end exceeds the frozen representative snapshot")
    validate_snapshot_range(prices, start=TRAIN_START, end=end)
    path = output_dir / "source-cache.json"
    if path.is_file():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("source", {}).get("snapshotSha256") != _sha(prices_raw):
            raise ValueError("existing source cache belongs to a different price snapshot")
        if cached.get("source", {}).get("extractedRange") != [TRAIN_START.isoformat(), end.isoformat()]:
            raise ValueError("existing source cache covers a different raw-history range")
        channels, selection = load_source_cache(path)
        if not channels or not selection:
            raise ValueError("existing source cache is empty")
        return cached
    uri = f"file:{db_path.resolve()}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        channels, source_audit = extract_channels(db, TRAIN_START, end)
    reconciliation = verify_target_reconciliation(source_audit, prices, TRAIN_START, end)
    selection = select_channels(channels)
    cached = _serialize_channels(channels, selection)
    doc = {"schemaVersion": 1, "experiment": "external-reference-v1",
           "source": {"databasePath": str(db_path.relative_to(ROOT)),
                      "query": "dashboard.PRICE_SQL selected latest validated request rows",
                      "snapshotPath": str(prices_path.relative_to(ROOT)),
                      "snapshotSha256": _sha(prices_raw), "extractedRange": [TRAIN_START.isoformat(), end.isoformat()],
                      "selectedRowsSha256": source_audit["selectedRowsSha256"]},
           "sourceCodeSha256": _file_sha(Path(__file__)),
           "targetReconciliation": reconciliation,
           "channelSelection": selection, "sourceAudit": source_audit,
           "channels": cached, "protocol": protocol()}
    doc["cacheSha256"] = _sha(_canonical(doc))
    _write_once(path, _canonical(doc))
    return doc


def load_source_cache(path: Path = OUT / "source-cache.json") -> tuple[dict[str, Any], dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    digest = doc.get("cacheSha256")
    unsigned = dict(doc)
    unsigned.pop("cacheSha256", None)
    if digest != _sha(_canonical(unsigned)):
        raise ValueError("source extraction cache hash mismatch")
    if doc.get("sourceCodeSha256") != _file_sha(Path(__file__)):
        raise ValueError("source cache was generated by a different experiment implementation")
    if doc.get("targetReconciliation", {}).get("status") != "verified":
        raise ValueError("source cache target reconciliation is not verified")
    return _deserialize_channels(doc["channels"]), doc["channelSelection"]


def _write_once(path: Path, raw: bytes) -> None:
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"refusing to overwrite frozen artifact {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(raw)
    temp.replace(path)


def validate_forecast_manifest(doc: Mapping[str, Any], phase: str) -> None:
    if doc.get("schemaVersion") != 1 or doc.get("phase") != phase or doc.get("targetValuesAttached") is not False:
        raise ValueError("expected answer-free external-reference phase manifest")
    unsigned = dict(doc)
    digest = unsigned.pop("manifestSha256", None)
    if digest != _sha(_canonical(unsigned)):
        raise ValueError("forecast manifest self-hash mismatch")
    pinned_baseline = _baseline_data_for_phase(phase)
    expected = set(pinned_baseline)
    seen = set()
    for row in doc.get("forecasts", []):
        key = (row.get("product"), dt.date.fromisoformat(row["target"]), row.get("horizon"))
        if key not in expected or key in seen:
            raise ValueError("forecast manifest has unexpected or duplicate key")
        origin = dt.date.fromisoformat(row["origin"])
        cutoff = dt.date.fromisoformat(row["cutoff"])
        if row["target"] != (origin + dt.timedelta(days=row["horizon"])).isoformat() or cutoff != origin - dt.timedelta(days=1):
            raise ValueError("forecast target/origin/cutoff relation mismatch")
        baseline = row.get("baseline", {})
        pinned = pinned_baseline[key]
        if origin != pinned["origin"]:
            raise ValueError(f"forecast origin differs from pinned baseline: {key}")
        for field in ("E106_RHS", "E106", "rhsStatus", "e106Status"):
            expected_value = (pinned[field] if field in pinned else None)
            if baseline.get(field) != expected_value:
                raise ValueError(f"forecast copied baseline field differs from pinned baseline: {key}/{field}")
        for name in ("CTRL_LGB", "CAND_A"):
            state = row.get(name, {})
            if state.get("status") == "issued" and _finite_positive(state.get("prediction")) is None:
                raise ValueError("issued candidate forecast must be finite positive")
            if state.get("status") != "issued" and state.get("prediction") is not None:
                raise ValueError("unissued forecast cannot contain a prediction")
        if ((row["CTRL_LGB"].get("status") == "issued") !=
                (row["CAND_A"].get("status") == "issued")):
            raise ValueError("CTRL_LGB and CAND_A issuance must remain paired")
        seen.add(key)
    if seen != expected:
        raise ValueError("forecast manifest does not cover the saved RHS phase key set")
    fit_records = doc.get("fitRecords", [])
    if len(fit_records) > 240 or doc.get("pairedFitCount", len(fit_records)) != len(fit_records):
        raise ValueError("forecast manifest monthly fold count is inconsistent or exceeds 240")
    for record in fit_records:
        month = dt.date.fromisoformat(record["fitMonth"])
        fit_origin = dt.date.fromisoformat(record["fitOrigin"])
        if fit_origin != _month_start(month) or not record.get("trainingTargetsStrictlyBeforeFitOrigin"):
            raise ValueError("monthly fold does not use its fixed calendar-month training boundary")


def _baseline_data_for_phase(phase: str) -> dict[tuple[str, dt.date, int], dict[str, Any]]:
    path = BASELINE_PATHS[phase]
    if _file_sha(path) != BASELINE_SHA256[phase]:
        raise ValueError("saved RHS baseline manifest hash mismatch")
    doc = json.loads(path.read_text(encoding="utf-8"))
    return _baseline_index(doc, phase)


def _baseline_index_rows_for_phase(phase: str) -> set[tuple[str, dt.date, int]]:
    return set(_baseline_data_for_phase(phase))


def _common_rows(doc: Mapping[str, Any], actuals: Mapping[tuple[str, dt.date], float]) -> dict[str, list[dict[str, Any]]]:
    sets = {name: [] for name, _, _ in COMPARISONS}
    for row in doc["forecasts"]:
        product = row["product"]
        target = dt.date.fromisoformat(row["target"])
        horizon = row["horizon"]
        actual = _finite_positive(actuals.get((product, target)))
        if actual is None:
            continue
        values = {"CAND_A": _finite_positive(row["CAND_A"].get("prediction")) if row["CAND_A"].get("status") == "issued" else None,
                  "CTRL_LGB": _finite_positive(row["CTRL_LGB"].get("prediction")) if row["CTRL_LGB"].get("status") == "issued" else None,
                  "E106_RHS": _finite_positive(row["baseline"].get("E106_RHS")),
                  "E106": _finite_positive(row["baseline"].get("E106"))}
        base = {"product": product, "target": target.isoformat(), "horizon": horizon, "actual": actual}
        for name, left, right in COMPARISONS:
            if values[left] is not None and values[right] is not None:
                sets[name].append({**base, left: values[left], right: values[right]})
    return sets


def _fixed_holm(records: list[dict[str, Any]]) -> None:
    pvalues = [float(r["pValue"]) if r.get("status") == "tested" else 1.0 for r in records]
    order = sorted(range(len(records)), key=lambda i: pvalues[i])
    adjusted = [1.0] * len(records)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(records) - rank) * pvalues[idx]))
        adjusted[idx] = running
    for row, p in zip(records, adjusted):
        row["holmAdjustedPValue"] = p
        row["holmSignificant"] = row.get("status") == "tested" and p < 0.05


def score_manifest(doc: Mapping[str, Any], actuals: Mapping[tuple[str, dt.date], float], *, phase: str) -> dict[str, Any]:
    """Paired metrics, complete 240-test families, and common-calendar intervals."""
    validate_forecast_manifest(doc, phase)
    pairs = _common_rows(doc, actuals)
    cell_metrics = []
    dm = []
    for comparison, left, right in COMPARISONS:
        values = pairs[comparison]
        indexed = defaultdict(list)
        for row in values:
            indexed[(row["product"], row["horizon"])].append(row)
        for product in PRODUCTS:
            for horizon in HORIZONS:
                cell = indexed[(product, horizon)]
                n = len(cell)
                cell_metrics.append({"comparison": comparison, "product": product, "horizon": horizon,
                                     "n": n, "uniqueTargets": len({r["target"] for r in cell}),
                                     "firstTarget": min((r["target"] for r in cell), default=None),
                                     "lastTarget": max((r["target"] for r in cell), default=None),
                                     "leftMae": math.fsum(abs(r[left]-r["actual"]) for r in cell)/n if n else None,
                                     "rightMae": math.fsum(abs(r[right]-r["actual"]) for r in cell)/n if n else None,
                                     "leftMape": math.fsum(100*abs(r[left]-r["actual"])/r["actual"] for r in cell)/n if n else None,
                                     "rightMape": math.fsum(100*abs(r[right]-r["actual"])/r["actual"] for r in cell)/n if n else None})
                for loss in LOSSES:
                    diffs = {dt.date.fromisoformat(r["target"]):
                             (abs(r[left]-r["actual"])-abs(r[right]-r["actual"]) if loss == "ae" else
                              100*(abs(r[left]-r["actual"])-abs(r[right]-r["actual"]))/r["actual"])
                             for r in cell}
                    for bw in BANDWIDTHS:
                        dm.append({"comparison": comparison, "product": product, "horizon": horizon,
                                   "loss": loss, "bandwidth": bw, **inference._dm(diffs, bw)})
    for bw in BANDWIDTHS:
        family = [r for r in dm if r["bandwidth"] == bw]
        if len(family) != 240:
            raise ValueError("fixed DM family must retain all 240 comparison-cell-loss tests")
        _fixed_holm(family)

    bootstrap_results = {}
    comparison_summaries = {}
    bootstrap_start, bootstrap_end, _ = PHASES[phase]
    for comparison, left, right in COMPARISONS:
        rows = pairs[comparison]
        expected_present = {(r["product"], r["horizon"]) for r in rows}
        for metric in LOSSES:
            key = f"{comparison}/{metric}"
            if expected_present != set(ALL_CELLS):
                bootstrap_results[key] = {"status": "unsupported_missing_fixed_cells", "cellsPresent": len(expected_present),
                                          "cellsExpected": 40, "estimates": {}}
                comparison_summaries[key] = {"status": "unsupported_missing_fixed_cells",
                                             "pairedRows": len(rows), "cellsPresent": len(expected_present),
                                             "cellsExpected": 40}
                continue
            estimates = {}
            metric_rows = []
            for row in rows:
                metric_rows.append({**row, "a": row[left], "b": row[right]})
            for block in BOOTSTRAP_BLOCKS:
                result = bootstrap.circular_block_bootstrap(
                    metric_rows, a="a", b="b", metric=metric, block=block,
                    replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
                    calendar_start=bootstrap_start, calendar_end=bootstrap_end)
                estimates[str(block)] = result
            point = bootstrap._cell_means(metric_rows, "a", "b", metric, expected_cells=ALL_CELLS)
            bootstrap_results[key] = {"status": "complete", "pointDifference": point["difference"],
                                      "differenceDefinition": "left minus right; negative favors left",
                                      "estimates": estimates}
            comparison_summaries[key] = {"status": "complete", "pairedRows": len(rows),
                                         "cellsIncluded": 40, "leftEqualCellMean": point["a"],
                                         "rightEqualCellMean": point["b"],
                                         "difference": point["difference"],
                                         "differenceDefinition": "left minus right; negative favors left"}
    # RHS−E106 is a reference diagnostic only and does not expand the test family.
    refs = []
    for row in doc["forecasts"]:
        target = dt.date.fromisoformat(row["target"])
        actual = _finite_positive(actuals.get((row["product"], target)))
        rhs = _finite_positive(row["baseline"].get("E106_RHS"))
        e106 = _finite_positive(row["baseline"].get("E106"))
        if actual is not None and rhs is not None and e106 is not None:
            refs.append({"product": row["product"], "target": target.isoformat(), "horizon": row["horizon"],
                         "actual": actual, "a": rhs, "b": e106})
    reference = _equal_cell_metrics(refs, "a", "b")
    live_coverage = _coverage_from_doc(doc, actuals)
    v_gate = _validation_gate(cell_metrics, live_coverage, phase)
    report = {"schemaVersion": 1, "phase": phase, "forecastManifestSha256": doc.get("manifestSha256"),
              "comparisonRows": {k: len(v) for k, v in pairs.items()},
              "cellMetrics": cell_metrics, "dm": dm,
              "equalCellSummaries": comparison_summaries,
              "dmSummaryByBandwidth": [{"bandwidth": bw,
                "familySize": sum(r["bandwidth"] == bw for r in dm),
                "significantImprovement": sum(r["bandwidth"] == bw and r.get("holmSignificant") and r.get("meanLossDifference", 0) < 0 for r in dm),
                "significantWorsening": sum(r["bandwidth"] == bw and r.get("holmSignificant") and r.get("meanLossDifference", 0) > 0 for r in dm),
                "inferenceUnavailable": sum(r["bandwidth"] == bw and r.get("status") != "tested" for r in dm)} for bw in BANDWIDTHS],
              "commonCalendarBlockBootstrap": bootstrap_results,
              "storedRHSvsE106Reference": reference,
              "coverage": live_coverage, "validationGate": v_gate,
              "limitations": ["current retrieved-vintage retrospective inputs; first-publication time unknown",
                              "V and T are previously inspected retrospective periods, not independent future validation",
                              "nonsignificant DM results are inconclusive, not equivalence"]}
    report["reportSha256"] = _sha(_canonical(report))
    return report


def _equal_cell_metrics(rows: list[dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    means = bootstrap._cell_means(rows, a, b, "ae", expected_cells=ALL_CELLS)
    mape = bootstrap._cell_means(rows, a, b, "ape", expected_cells=ALL_CELLS)
    return {"status": "complete" if means and mape else "unsupported_missing_fixed_cells",
            "n": len(rows), "cellsPresent": len({(r["product"], r["horizon"]) for r in rows}),
            "rhsMae": means["a"] if means else None, "e106Mae": means["b"] if means else None,
            "rhsMape": mape["a"] if mape else None, "e106Mape": mape["b"] if mape else None,
            "rhsMinusE106Mae": means["difference"] if means else None,
            "rhsMinusE106MapePp": mape["difference"] if mape else None}


def _coverage_from_doc(doc: Mapping[str, Any], actuals: Mapping[tuple[str, dt.date], float]) -> dict[str, Any]:
    return _coverage(doc["forecasts"], _baseline_data_for_phase(doc["phase"]), actuals)


def _validation_gate(cell_metrics: list[dict[str, Any]], coverage: Mapping[str, Any], phase: str) -> dict[str, Any]:
    if phase != "V":
        return {"status": "not_applicable_outside_V"}
    candidate_rhs = [r for r in cell_metrics if r["comparison"] == "CAND_vs_RHS" and r["n"]]
    if {(r["product"], r["horizon"]) for r in candidate_rhs} != set(ALL_CELLS):
        return {"status": "fail", "reason": "missing_candidate_rhs_cells", "cellsPresent": len(candidate_rhs), "cellsExpected": 40}
    # Four products and ten horizons receive equal weight; cells use their
    # common paired rows, with identical key sets required by _common_rows.
    left_mape = math.fsum(r["leftMape"] for r in candidate_rhs) / len(candidate_rhs)
    right_mape = math.fsum(r["rightMape"] for r in candidate_rhs) / len(candidate_rhs)
    left_mae = math.fsum(r["leftMae"] for r in candidate_rhs) / len(candidate_rhs)
    right_mae = math.fsum(r["rightMae"] for r in candidate_rhs) / len(candidate_rhs)
    no_loss = True
    for item in coverage.values():
        if not isinstance(item, Mapping) or "rhsEligibleKeys" not in item:
            continue
        if item.get("candidateMissingRhsIssuedKeys") or item.get("candidateMissingPositiveAnswerKeys"):
            no_loss = False
    passed = left_mape < right_mape and left_mae <= right_mae and no_loss
    return {"status": "pass" if passed else "fail", "passed": passed,
            "candidate40CellEqualHorizonMape": left_mape, "rhs40CellEqualHorizonMape": right_mape,
            "candidate40CellEqualHorizonMae": left_mae, "rhs40CellEqualHorizonMae": right_mae,
            "candidateCoverageNoLoss": no_loss,
            "rule": "strictly lower MAPE, no higher MAE, and no issuance coverage loss"}


def score_and_save(phase: str, manifest_path: Path, prices_path: Path = PRICES_PATH,
                   out_dir: Path = OUT) -> dict[str, Any]:
    doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require_replay_verification(phase, doc, out_dir)
    prices = json.loads(prices_path.read_text(encoding="utf-8"))
    from . import equal_info_online_forecasts as online
    actuals = _actuals_from_snapshot(prices)
    report = score_manifest(doc, actuals, phase=phase)
    path = out_dir / phase / "score-report.json"
    _write_once(path, _canonical(report))
    return report


def _require_replay_verification(phase: str, manifest: Mapping[str, Any], out_dir: Path) -> dict[str, Any]:
    path = out_dir / phase / "model-replay-verification.json"
    if not path.is_file():
        raise ValueError("forecast cannot be scored before saved-model replay verification")
    result = json.loads(path.read_text(encoding="utf-8"))
    unsigned = dict(result)
    digest = unsigned.pop("verificationSha256", None)
    if digest != _sha(_canonical(unsigned)) or result.get("status") != "pass":
        raise ValueError("saved-model replay verification is invalid or failed")
    if result.get("forecastManifestSha256") != manifest.get("manifestSha256"):
        raise ValueError("saved-model replay verification is bound to a different forecast manifest")
    if result.get("executionIdentity") != _execution_identity():
        raise ValueError("saved-model replay verification belongs to a different execution identity")
    expected_replayed = sum(row["CTRL_LGB"].get("status") == "issued" and
                            row["CAND_A"].get("status") == "issued" for row in manifest.get("forecasts", []))
    if result.get("failures") or result.get("pairedForecastsReplayed") != expected_replayed:
        raise ValueError("saved-model replay verification does not cover every issued pair")
    return result


def run_phase(phase: str, *, output_dir: Path = OUT,
              estimator_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Run one approved phase from the immutable source cache and saved RHS baseline."""
    if phase not in PHASES:
        raise ValueError("phase must be V or T")
    cache_path = output_dir / "source-cache.json"
    channels, selection = load_source_cache(cache_path)
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    baseline_path = BASELINE_PATHS[phase]
    if _file_sha(baseline_path) != BASELINE_SHA256[phase]:
        raise ValueError("saved RHS baseline manifest hash mismatch")
    if phase == "T":
        v_manifest_path = output_dir / "V/forecast-manifest.json"
        v_report_path = output_dir / "V/score-report.json"
        if not v_manifest_path.is_file() or not v_report_path.is_file():
            raise ValueError("T requires completed, scored V artifacts")
        v_manifest = json.loads(v_manifest_path.read_text(encoding="utf-8"))
        v_report = json.loads(v_report_path.read_text(encoding="utf-8"))
        validate_forecast_manifest(v_manifest, "V")
        _require_replay_verification("V", v_manifest, output_dir)
        if v_report.get("forecastManifestSha256") != v_manifest.get("manifestSha256"):
            raise ValueError("V score is not bound to the saved forecast manifest")
        unsigned_report = dict(v_report)
        v_report_digest = unsigned_report.pop("reportSha256", None)
        if v_report_digest != _sha(_canonical(unsigned_report)):
            raise ValueError("V score report self-hash mismatch")
        if v_manifest.get("sourceCacheSha256") != cache.get("cacheSha256"):
            raise ValueError("T cannot use a different source cache than V")
        if v_manifest.get("sourceCodeSha256") != _file_sha(Path(__file__)):
            raise ValueError("T cannot continue under a changed experiment implementation")
        if v_report.get("validationGate", {}).get("status") != "pass":
            raise ValueError("frozen V gate failed; T is not permitted")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    price_raw = PRICES_PATH.read_bytes()
    from . import equal_info_online_forecasts as online
    if _sha(price_raw) != online.PRICES_SHA256:
        raise ValueError("frozen representative price snapshot hash mismatch")
    prices = json.loads(price_raw)
    manifest = build_forecasts(phase, baseline, prices, channels, selection,
                               out_dir=output_dir, estimator_factory=estimator_factory,
                               source_cache_sha256=cache["cacheSha256"])
    replay = verify_saved_model_replay(phase, manifest, prices, channels, selection, out_dir=output_dir)
    if replay.get("status") != "pass":
        raise ValueError("saved-model replay verification did not pass")
    report_path = output_dir / phase / "score-report.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("forecastManifestSha256") != manifest.get("manifestSha256"):
            raise ValueError("existing score report belongs to a different forecast manifest")
        unsigned_report = dict(report)
        report_digest = unsigned_report.pop("reportSha256", None)
        if report_digest != _sha(_canonical(unsigned_report)):
            raise ValueError("existing score report self-hash mismatch")
    else:
        actuals = _actuals_from_snapshot(prices)
        report = score_manifest(manifest, actuals, phase=phase)
        _write_once(report_path, _canonical(report))
    return {"phase": phase, "manifestSha256": manifest["manifestSha256"],
            "scoreReportSha256": report["reportSha256"],
            "validationGate": report.get("validationGate"),
            "pairedFitCount": manifest["pairedFitCount"],
            "individualEstimatorFitCount": manifest["individualEstimatorFitCount"],
            "coverage": manifest["coverage"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "run-v", "run-t"), required=True)
    args = parser.parse_args(argv)
    if args.stage == "preflight":
        result = preflight()
        print(json.dumps({"status": "preflight_complete", "targetReconciliation": result["targetReconciliation"],
                          "channelSelection": {p: result["channelSelection"][p]["status"] for p in PRODUCTS},
                          "sourceCacheSha256": result["cacheSha256"]}, ensure_ascii=False, sort_keys=True))
    elif args.stage == "run-v":
        result = run_phase("V")
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        result = run_phase("T")
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
