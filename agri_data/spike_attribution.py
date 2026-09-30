"""Read-only transaction and weather attribution diagnostics for price shocks.

All findings are retrospective descriptive associations. They do not identify
causes and do not alter or retrain the archived forecasting models.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sqlite3
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import collector, dashboard, diagnostics, experiments, significance

FROZEN_SHA256 = "9d6ac4d6a00c581821e97b4c4de074d129af6f06b8c8b38fe7ecbbf398c277f7"
CONTEXT_START = dt.date(2018, 12, 1)
EVALUATION_START = dt.date(2019, 1, 1)
EVALUATION_END = dt.date(2020, 2, 18)
PRODUCTS = ("cabbage", "radish")
HORIZON = 1
SPIKE_FRACTION = 0.10
MARKET_TOL = 1e-7
ALPHA = 0.05
TEST_COUNT = 12
MISSING = "(missing)"
DIMENSIONS = {"subclass": ("gds_sclsf_cd", "gds_sclsf_nm"),
              "grade": ("grd_cd", "grd_nm")}


def _date(value: Any, label: str) -> dt.date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        return dt.date.fromisoformat(value[:10])
    except ValueError as exc:
        raise ValueError(f"invalid {label}") from exc


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _decimal(value: Any, label: str) -> Decimal:
    result = dashboard._as_decimal(value)
    if result is None:
        raise ValueError(f"invalid {label} in selected price row")
    return result


def _tvd(left: dict[str, float], right: dict[str, float]) -> float | None:
    if not left or not right:
        return None
    if any(value < -1e-10 for value in (*left.values(), *right.values())):
        return None
    if not math.isclose(sum(left.values()), 1.0, abs_tol=1e-8) or not math.isclose(sum(right.values()), 1.0, abs_tol=1e-8):
        return None
    return 0.5 * sum(abs(left.get(key, 0.0) - right.get(key, 0.0))
                      for key in set(left) | set(right))


def _day_cells(rows: list[dict[str, Any]], denominator: float) -> dict[str, Any]:
    """Build signed cell prices/shares; never drop or renormalize invalid cells."""
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    market: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = (str(row["market"]), str(row["subclass"]), str(row["grade"]))
        cell = cells.setdefault(key, {"amount": Decimal(0), "volume": Decimal(0), "rows": 0})
        cell["amount"] += row["amount"]
        cell["volume"] += row["volume"]
        cell["rows"] += 1
        mk = market.setdefault(key[0], {"amount": Decimal(0), "volume": Decimal(0), "rows": 0})
        mk["amount"] += row["amount"]
        mk["volume"] += row["volume"]
        mk["rows"] += 1
    result: dict[tuple[str, str, str], dict[str, Any]] = {}
    invalid = []
    for key, cell in cells.items():
        q, a = float(cell["volume"]), float(cell["amount"])
        price = a / q if q != 0 else None
        if q <= 0 or price is None or not math.isfinite(price) or price <= 0:
            invalid.append({"cell": list(key), "netVolumeKg": q, "netAmount": a,
                            "price": price, "rowCount": cell["rows"]})
        else:
            result[key] = {"amount": a, "volume": q, "price": price,
                           "share": q / denominator, "rowCount": cell["rows"]}
    market_result = {}
    market_invalid = []
    for code, cell in market.items():
        q, a = float(cell["volume"]), float(cell["amount"])
        price = a / q if q != 0 else None
        if q <= 0 or price is None or not math.isfinite(price) or price <= 0:
            market_invalid.append({"market": code, "netVolumeKg": q, "netAmount": a,
                                   "price": price, "rowCount": cell["rows"]})
        else:
            market_result[code] = {"amount": a, "volume": q, "price": price,
                                   "share": q / denominator, "rowCount": cell["rows"]}
    return {"cells": result, "invalidCells": invalid, "market": market_result,
            "invalidMarkets": market_invalid,
            "allCellShares": {"sum": sum(float(c["volume"]) / denominator for c in cells.values()) if denominator else None,
                              "denominatorVolumeKg": denominator}}


def decompose_pair(previous: dict[str, Any], current: dict[str, Any],
                   previous_price: float, current_price: float) -> dict[str, Any]:
    """Midpoint within/mix plus entry/exit decomposition of aggregate price change."""
    left, right = previous["cells"], current["cells"]
    keys0, keys1 = set(left), set(right)
    common = keys0 & keys1
    if previous["invalidCells"] or current["invalidCells"]:
        return {"status": "unavailable_nonpositive_or_invalid_cell", "invalidPreviousCells": previous["invalidCells"],
                "invalidCurrentCells": current["invalidCells"], "renormalized": False}
    within_terms, mix_terms, cell_terms = [], [], []
    for key in sorted(common):
        a, b = left[key], right[key]
        within = (a["share"] + b["share"]) / 2 * (b["price"] - a["price"])
        mix = (a["price"] + b["price"]) / 2 * (b["share"] - a["share"])
        within_terms.append(within)
        mix_terms.append(mix)
        cell_terms.append({"cell": list(key), "within": within, "mix": mix,
                           "total": within + mix})
    entry = sum(right[key]["share"] * right[key]["price"] for key in keys1 - keys0)
    exit_value = -sum(left[key]["share"] * left[key]["price"] for key in keys0 - keys1)
    within_total, mix_total = sum(within_terms), sum(mix_terms)
    turnover = entry + exit_value
    delta = current_price - previous_price
    residual = delta - (within_total + mix_total + turnover)
    tolerance = max(1e-8, 1e-9 * max(abs(delta), abs(current_price), abs(previous_price), 1.0))
    if abs(residual) > tolerance:
        raise ValueError("price composition decomposition does not reconcile")
    common_mass = sum(min(left[key]["share"], right[key]["share"]) for key in common)
    entry_mass = sum(right[key]["share"] for key in keys1 - keys0)
    exit_mass = sum(left[key]["share"] for key in keys0 - keys1)
    cell_terms.extend({"cell": list(key), "within": 0.0, "mix": 0.0,
                       "total": right[key]["share"] * right[key]["price"], "kind": "entry"}
                      for key in sorted(keys1 - keys0))
    cell_terms.extend({"cell": list(key), "within": 0.0, "mix": 0.0,
                       "total": -left[key]["share"] * left[key]["price"], "kind": "exit"}
                      for key in sorted(keys0 - keys1))
    return {"status": "available", "priceDelta": delta, "within": within_total,
            "mix": mix_total, "turnover": turnover, "entry": entry, "exit": exit_value,
            "reconciliationResidual": residual, "tolerance": tolerance,
            "commonMassCoverage": common_mass, "entryMass": entry_mass, "exitMass": exit_mass,
            "commonCellCount": len(common), "entryCellCount": len(keys1 - keys0),
            "exitCellCount": len(keys0 - keys1), "topCellContributions": sorted(
                cell_terms, key=lambda row: (-abs(row["total"]), row["cell"]))[:5],
            "renormalized": False}


def _query_product(query: dict[str, Any]) -> str | None:
    return dashboard._query_product(query)


def extract_prices(db: sqlite3.Connection, start: dt.date = CONTEXT_START,
                   end: dt.date = EVALUATION_END) -> dict[str, Any]:
    """Stream selected normalized auction/kg records using the dashboard query policy."""
    buckets: dict[str, dict[dt.date, dict[str, Any]]] = {p: defaultdict(lambda: {
        "amount": Decimal(0), "volume": Decimal(0), "rows": 0, "markets": defaultdict(list)})
        for p in PRODUCTS}
    states: dict[str, dict[tuple[dt.date, str], str]] = {p: {} for p in PRODUCTS}
    provenance = {p: {"requestIds": set(), "jobIds": set(), "sourceUrls": set(),
                      "retrievedAt": [], "publishedUnknownRequestCount": 0,
                      "qualityFlags": Counter(), "negativeAmountRows": 0,
                      "negativeVolumeRows": 0, "zeroVolumeRows": 0,
                      "selectedEpochs": set(), "requestMetadata": {}} for p in PRODUCTS}
    for record in db.execute(dashboard.PRICE_SQL, (start.isoformat(), end.isoformat(), *dashboard.MARKETS.keys())):
        query = dashboard._json_object(record["query_json"])
        product = _query_product(query or {})
        if product not in PRODUCTS or query is None:
            continue
        query_day = dashboard._parse_date(query.get("cond[trd_clcln_ymd::EQ]"))
        market = str(query.get("cond[whsl_mrkt_cd::EQ]", ""))
        query_subclass = str(query.get("cond[gds_sclsf_cd::EQ]", ""))
        if query_day is None or market not in dashboard.MARKETS or query_subclass:
            # These products use their broad market/day scope; any narrower scope
            # would overlap the broad total and is deliberately excluded.
            continue
        state = record["selected_state"] if record["selected_state"] in ("complete", "empty") else (record["latest_state"] or "unavailable")
        states[product][(query_day, market)] = str(state)
        if record["request_id"] is None:
            if record["selected_state"] not in ("complete", "empty"):
                continue
            if record["selected_row_count"] != 0:
                raise ValueError(f"selected manifest query has no request row for {product}/{query_day}/{market}")
            continue
        if record["selected_state"] not in ("complete", "empty"):
            raise ValueError(f"selected query is not complete for {product}/{query_day}/{market}")
        meta = provenance[product]
        meta["requestIds"].add(int(record["request_id"]))
        meta["jobIds"].add(int(record["selected_job_id"]))
        meta["sourceUrls"].add(str(record["source_url"]))
        meta["selectedEpochs"].add((int(record["selected_job_id"]), str(record["selected_epoch_id"])))
        request_id = int(record["request_id"])
        if request_id not in meta["requestMetadata"]:
            request_meta = db.execute("SELECT response_sha256 FROM source_requests WHERE request_id=?",
                                      (request_id,)).fetchone()
            meta["retrievedAt"].append(str(record["retrieved_at"]))
            meta["requestMetadata"][request_id] = {
                "requestId": request_id, "jobId": int(record["selected_job_id"]),
                "epochId": str(record["selected_epoch_id"]),
                "responseSha256": request_meta["response_sha256"] if request_meta else None,
                "retrievedAt": str(record["retrieved_at"]), "publishedAt": record["published_at"],
                "versionPublishedAt": record["version_published_at"],
                "publicationTimeUnknown": record["published_at"] is None or record["version_published_at"] is None}
            if record["published_at"] is None or record["version_published_at"] is None:
                meta["publishedUnknownRequestCount"] += 1
        if record["payload_json"] is None:
            continue
        row = dashboard._json_object(record["payload_json"])
        if row is None:
            raise ValueError(f"invalid selected row JSON for request {record['request_id']}")
        if dashboard._product_id(row) != product:
            continue
        if row.get("trd_se") != "경매" or row.get("unit_nm") != "kg":
            continue
        if str(row.get("whsl_mrkt_cd", "")) != market:
            raise ValueError(f"selected row market does not match query {record['request_id']}")
        source_day = dashboard._parse_date(row.get("trd_clcln_ymd"))
        if source_day != query_day:
            raise ValueError(f"selected row date does not match query {record['request_id']}")
        amount = _decimal(row.get("totprc"), "totprc")
        volume = _decimal(row.get("unit_tot_qty"), "unit_tot_qty")
        bucket = buckets[product][source_day]
        bucket["amount"] += amount
        bucket["volume"] += volume
        bucket["rows"] += 1
        bucket["markets"][market].append({
            "amount": amount, "volume": volume,
            "subclass": str(row.get("gds_sclsf_cd") or MISSING),
            "subclassName": str(row.get("gds_sclsf_nm") or MISSING),
            "grade": str(row.get("grd_cd") or MISSING),
            "gradeName": str(row.get("grd_nm") or MISSING),
            "qualityFlags": json.loads(record["quality_flags"] or "[]") if record["quality_flags"] else [],
        })
        if amount < 0: meta["negativeAmountRows"] += 1
        if volume < 0: meta["negativeVolumeRows"] += 1
        if volume == 0: meta["zeroVolumeRows"] += 1
        for flag in (json.loads(record["quality_flags"] or "[]") if record["quality_flags"] else []):
            meta["qualityFlags"][str(flag)] += 1

    series: dict[str, dict[dt.date, dict[str, Any]]] = {p: {} for p in PRODUCTS}
    market_rows: dict[str, dict[dt.date, dict[str, list[dict[str, Any]]]]] = {p: {} for p in PRODUCTS}
    for product in PRODUCTS:
        for day, bucket in buckets[product].items():
            if not all(states[product].get((day, market)) in ("complete", "empty") for market in dashboard.MARKETS):
                continue
            if bucket["volume"] <= 0 or bucket["rows"] == 0:
                continue
            price = float(bucket["amount"] / bucket["volume"])
            if not math.isfinite(price) or price <= 0:
                continue
            market_parts = {}
            market_rows[product][day] = {}
            for market in dashboard.MARKETS:
                rows = bucket["markets"].get(market, [])
                market_rows[product][day][market] = rows
                if not rows:
                    continue
                amount, volume = sum((r["amount"] for r in rows), Decimal(0)), sum((r["volume"] for r in rows), Decimal(0))
                market_price = float(amount / volume) if volume > 0 else None
                market_parts[market] = {"amount": float(amount), "volumeKg": float(volume),
                                        "price": market_price, "rowCount": len(rows)}
            series[product][day] = {"actual": price, "volumeKg": float(bucket["volume"]),
                                    "amount": float(bucket["amount"]), "rowCount": bucket["rows"],
                                    "marketData": market_parts}
        meta = provenance[product]
        for key in ("requestIds", "jobIds", "sourceUrls", "selectedEpochs"):
            meta[key] = sorted([list(v) if isinstance(v, tuple) else v for v in meta[key]])
        meta["retrievedAtRange"] = [min(meta["retrievedAt"], default=None), max(meta["retrievedAt"], default=None)]
        meta.pop("retrievedAt")
        meta["qualityFlags"] = dict(sorted(meta["qualityFlags"].items()))
        meta["requestMetadata"] = [meta["requestMetadata"][key] for key in sorted(meta["requestMetadata"])]
    return {"series": series, "marketRows": market_rows, "provenance": provenance,
            "queryStates": states, "contextStart": start.isoformat(), "contextEnd": end.isoformat()}


def _selected_predictions(source: dict[str, Any], product_id: str):
    product_map, validated, start, end = significance._validate_input(source)
    if start > EVALUATION_START or end < EVALUATION_END:
        raise ValueError("frozen prediction period must contain the fixed spike evaluation period")
    product = product_map[product_id]
    model = validated[(product_id, "ridge", "daily")]
    base = validated[(product_id, "baseline")]
    obs = {dt.date.fromisoformat(row["date"]): _number(row["actual"], "actual")
           for row in product["observations"]
           if EVALUATION_START <= dt.date.fromisoformat(row["date"]) <= EVALUATION_END}
    targets = {dt.date.fromisoformat(day) for day, horizon in model if horizon == HORIZON
               and EVALUATION_START <= dt.date.fromisoformat(day) <= EVALUATION_END}
    common = sorted(day for day in targets if day in obs and (day.isoformat(), HORIZON) in base)
    return product, model, base, obs, common


def _distribution(shares: dict[str, float], values: list[dict[str, Any]]) -> dict[str, Any]:
    volume = sum((r["volume"] for r in values), Decimal(0))
    if volume <= 0:
        return {"status": "unavailable_nonpositive_net_volume", "shares": {}, "renormalized": False}
    weights: dict[str, Decimal] = defaultdict(Decimal)
    for row in values:
        weights[shares[row["key"]]] += row["volume"]
    output = {key: float(value / volume) for key, value in sorted(weights.items())}
    return {"status": "available" if all(x >= 0 for x in output.values()) else "unavailable_negative_share",
            "shares": output, "renormalized": False}


def _market_tvd(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    return _tvd(left, right)


def _daily_composition(rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    all_rows = [row for market in rows.values() for row in market]
    amount = sum((row["amount"] for row in all_rows), Decimal(0))
    volume = sum((row["volume"] for row in all_rows), Decimal(0))
    cell_rows = []
    market_weights: dict[str, Decimal] = defaultdict(Decimal)
    market_totals: dict[str, dict[str, Any]] = {}
    grade_weights: dict[str, Decimal] = defaultdict(Decimal)
    for market, items in rows.items():
        market_amount = sum((row["amount"] for row in items), Decimal(0))
        market_volume = sum((row["volume"] for row in items), Decimal(0))
        market_price = float(market_amount / market_volume) if market_volume > 0 else None
        market_totals[market] = {"amount": float(market_amount), "volumeKg": float(market_volume),
            "price": market_price, "rowCount": len(items),
            "status": "available" if market_price is not None and math.isfinite(market_price) and market_price > 0 else "unavailable_nonpositive_net"}
        for row in items:
            key = (market, row["subclass"], row["grade"])
            cell_rows.append({"key": key, "amount": row["amount"], "volume": row["volume"], "rowCount": 1})
            market_weights[market] += row["volume"]
            grade_weights[row["grade"]] += row["volume"]
    return {"amount": float(amount), "volume": float(volume), "rowCount": len(all_rows),
            "cells": cell_rows, "marketData": market_totals,
            "marketShares": {k: float(v / volume) for k, v in market_weights.items()} if volume > 0 else {},
            "gradeShares": {k: float(v / volume) for k, v in grade_weights.items()} if volume > 0 else {},
            "marketInvalid": any(v < 0 for v in market_weights.values()) or volume <= 0,
            "gradeInvalid": any(v < 0 for v in grade_weights.values()) or volume <= 0}


def _decompose_raw_pair(previous: dict[str, Any], current: dict[str, Any],
                        previous_price: float, current_price: float, *, market_only: bool = False) -> dict[str, Any]:
    def cells(day):
        grouped = {}
        for row in day["cells"]:
            key = (row["key"][0],) if market_only else row["key"]
            cell = grouped.setdefault(key, {"amount": Decimal(0), "volume": Decimal(0), "rows": 0})
            cell["amount"] += row["amount"]
            cell["volume"] += row["volume"]
            cell["rows"] += row.get("rowCount", 1)
        out, invalid = {}, []
        for key, cell in grouped.items():
            q, a = float(cell["volume"]), float(cell["amount"])
            price = a/q if q != 0 else None
            if q <= 0 or price is None or not math.isfinite(price) or price <= 0:
                invalid.append({"cell": list(key), "netVolumeKg": q, "netAmount": a,
                                "rowCount": cell["rows"], "price": price})
            else:
                out[key] = {"price": price, "share": q / day["volume"], "volume": q,
                            "amount": a, "rowCount": cell["rows"]}
        return out, invalid
    left, bad_left = cells(previous); right, bad_right = cells(current)
    if bad_left or bad_right:
        return {"status": "unavailable_nonpositive_or_invalid_cell", "invalidPreviousCells": bad_left,
                "invalidCurrentCells": bad_right, "renormalized": False}
    result = decompose_pair({"cells": left, "invalidCells": []}, {"cells": right, "invalidCells": []},
                            previous_price, current_price)
    result["previousCells"] = [{"cell": list(key), **value} for key, value in sorted(left.items())]
    result["currentCells"] = [{"cell": list(key), **value} for key, value in sorted(right.items())]
    return result


def _events(series: dict[dt.date, dict[str, Any]], market_rows: dict[dt.date, dict[str, list[dict[str, Any]]]],
            common_dates: list[dt.date], baseline: dict, ridge: dict, actuals: dict[dt.date, float],
            weather: dict[str, dict[dt.date, dict[str, Any]]]):
    ordered = sorted(series)
    prior_by_day = {day: ordered[i-1] for i, day in enumerate(ordered) if i}
    records = []
    for day in common_dates:
        prior = prior_by_day.get(day)
        if prior is None:
            continue
        current_price, prior_price = actuals[day], series[prior]["actual"]
        if current_price <= 0 or prior_price <= 0:
            raise ValueError("price observations must be positive")
        signed_log = math.log(current_price / prior_price)
        gap = (day - prior).days
        prior_parts, current_parts = _daily_composition(market_rows.get(prior, {})), _daily_composition(market_rows.get(day, {}))
        total_actual = series[day]
        db_price = float(prior_parts["amount"] / prior_parts["volume"]) if prior_parts["volume"] > 0 else None
        db_cur_price = float(current_parts["amount"] / current_parts["volume"]) if current_parts["volume"] > 0 else None
        decomposition = _decompose_raw_pair(prior_parts, current_parts, prior_price, current_price)
        market_decomposition = _decompose_raw_pair(prior_parts, current_parts, prior_price, current_price,
                                                   market_only=True)
        if db_price is None or db_cur_price is None or not math.isclose(db_price, prior_price, rel_tol=1e-10, abs_tol=1e-7) \
                or not math.isclose(db_cur_price, current_price, rel_tol=1e-10, abs_tol=1e-7):
            raise ValueError(f"cell totals do not reconcile to selected price on {day}")
        current_prediction = ridge[(day.isoformat(), HORIZON)]["predicted"]
        base_prediction = baseline[(day.isoformat(), HORIZON)]["predicted"]
        next_day = next((candidate for candidate in ordered if candidate > day), None)
        records.append({"date": day.isoformat(), "previousDate": prior.isoformat(), "gapDays": gap,
            "actual": current_price, "previousActual": prior_price,
            "signedLogChange": signed_log, "absLogChange": abs(signed_log),
            "signedPercentChange": (current_price / prior_price - 1) * 100,
            "absLogChangePercent": abs(signed_log) * 100,
            "direction": "up" if signed_log > 0 else "down" if signed_log < 0 else "unchanged",
            "actualVolumeKg": total_actual["volumeKg"], "previousVolumeKg": series[prior]["volumeKg"],
            "weekday": day.strftime("%A"),
            "volumeChangePct": ((total_actual["volumeKg"] / series[prior]["volumeKg"] - 1) * 100
                                 if total_actual["volumeKg"] > 0 and series[prior]["volumeKg"] > 0 else None),
            "rowCount": total_actual["rowCount"], "previousRowCount": series[prior]["rowCount"],
            "marketPriceChanges": {market: {"previousPrice": prior_parts["marketData"].get(market, {}).get("price"),
                "currentPrice": current_parts["marketData"].get(market, {}).get("price"),
                "changePct": ((current_parts["marketData"][market]["price"] / prior_parts["marketData"][market]["price"] - 1)*100
                    if prior_parts["marketData"].get(market, {}).get("status") == "available"
                    and current_parts["marketData"].get(market, {}).get("status") == "available" else None),
                "previousRowCount": prior_parts["marketData"].get(market, {}).get("rowCount", 0),
                "currentRowCount": current_parts["marketData"].get(market, {}).get("rowCount", 0)}
                for market in dashboard.MARKETS},
            "baselinePrediction": base_prediction, "ridgePrediction": current_prediction,
            "baselineAPE": abs(base_prediction-current_price)/current_price*100,
            "ridgeAPE": abs(current_prediction-current_price)/current_price*100,
            "marketData": total_actual["marketData"], "cellDecomposition": decomposition,
            "marketPriceDecomposition": market_decomposition,
            "marketTvd": _market_tvd(prior_parts["marketShares"], current_parts["marketShares"])
                if not (prior_parts["marketInvalid"] or current_parts["marketInvalid"]) else None,
            "gradeTvd": _market_tvd(prior_parts["gradeShares"], current_parts["gradeShares"])
                if not (prior_parts["gradeInvalid"] or current_parts["gradeInvalid"]) else None,
            "nextObservation": ({"date": next_day.isoformat(), "gapDays": (next_day-day).days,
                "actual": series[next_day]["actual"], "returnFromEventPct":
                    (series[next_day]["actual"]/current_price-1)*100,
                "absLogReturnFromEvent": abs(math.log(series[next_day]["actual"]/current_price))}
                if next_day else None),
            "weatherLag7": {station: {feature: {"value": (window := diagnostics._weather_window_value(
                    weather[station], day, feature))[0], "status": window[1]}
                    for feature in ("avgTa_7day_mean", "daily_range_7day_mean", "sumRn_7day_sum")}
                    for station in diagnostics.STATIONS}})
    ordered_spikes = sorted(records, key=lambda row: (-row["absLogChange"], row["date"]))
    n_spikes = math.ceil(len(records) * SPIKE_FRACTION)
    spike_dates = {row["date"] for row in ordered_spikes[:n_spikes]}
    cutoff = ordered_spikes[n_spikes-1]["absLogChange"] if n_spikes else None
    for row in records:
        row["spikeTopDecile"] = row["date"] in spike_dates
    return records, ordered_spikes[:n_spikes], cutoff


def _subset_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = Counter("1" if row["gapDays"] == 1 else "2-7" if row["gapDays"] <= 7 else ">7" for row in rows)
    volume = [abs(row["volumeChangePct"]) for row in rows if row["volumeChangePct"] is not None]
    return {"n": len(rows), "meanAbsLogPriceChangePct": statistics.fmean(r["absLogChangePercent"] for r in rows) if rows else None,
            "meanAbsVolumeChangePct": statistics.fmean(volume) if volume else None,
            "meanMarketTvd": statistics.fmean([r["marketTvd"] for r in rows if r["marketTvd"] is not None]) if any(r["marketTvd"] is not None for r in rows) else None,
            "meanGradeTvd": statistics.fmean([r["gradeTvd"] for r in rows if r["gradeTvd"] is not None]) if any(r["gradeTvd"] is not None for r in rows) else None,
            "gapCounts": dict(sorted(gaps.items())), "meanRowCount": statistics.fmean(r["rowCount"] for r in rows) if rows else None,
            "meanAuctionVolumeKg": statistics.fmean(r["actualVolumeKg"] for r in rows) if rows else None,
            "upCount": sum(r["direction"] == "up" for r in rows), "downCount": sum(r["direction"] == "down" for r in rows),
            "consecutiveCalendarPairCount": sum(r["gapDays"] == 1 for r in rows)}


def _consecutive_pairs(series: dict[dt.date, dict[str, Any]], comp: dict[dt.date, dict[str, Any]],
                       current_dates: list[dt.date]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = []
    for day in sorted(set(current_dates)):
        prior = day - dt.timedelta(days=1)
        if prior not in series or day not in series or prior not in comp or day not in comp:
            continue
        p0, p1 = series[prior]["actual"], series[day]["actual"]
        q0, q1 = series[prior]["volumeKg"], series[day]["volumeKg"]
        if min(p0, p1) <= 0:
            continue
        y = 100 * abs(math.log(p1/p0))
        x_volume = 100*abs(math.log(q1/q0)) if q0 > 0 and q1 > 0 else None
        market_tvd = _market_tvd(comp[prior]["marketShares"], comp[day]["marketShares"]) \
            if not (comp[prior]["marketInvalid"] or comp[day]["marketInvalid"]) else None
        grade_tvd = _market_tvd(comp[prior]["gradeShares"], comp[day]["gradeShares"]) \
            if not (comp[prior]["gradeInvalid"] or comp[day]["gradeInvalid"]) else None
        rows.append({"date": day, "y": y, "x_volume": x_volume, "x_market_tvd": market_tvd,
                     "x_grade_tvd": grade_tvd})
    return rows, {"consecutiveCalendarPricePairs": len(rows)}


def _associations(price_series: dict[str, dict[dt.date, dict[str, Any]]],
                  raw_composition: dict[str, dict[dt.date, dict[str, Any]]],
                  weather: dict[str, dict[dt.date, dict[str, Any]]],
                  current_dates: dict[str, list[dt.date]]) -> list[dict[str, Any]]:
    results = []
    for product in PRODUCTS:
        pairs, _ = _consecutive_pairs(price_series[product], raw_composition[product], current_dates[product])
        predictors = [("absolute_log_volume_change_pct", lambda r: r["x_volume"], "100×absolute log kg change (%)"),
                      ("market_tvd", lambda r: r["x_market_tvd"], "market share TVD [0,1]"),
                      ("grade_tvd", lambda r: r["x_grade_tvd"], "grade share TVD [0,1]")]
        for name, getter, units in predictors:
            samples = [{"date": row["date"], "x": getter(row), "y": row["y"]}
                       for row in pairs if getter(row) is not None]
            results.append({"testId": f"{product}.{name}", "productId": product,
                            "feature": name, "featureUnit": units,
                            "outcome": "100×absolute log price change (%)",
                            "controls": "intercept + month/year/weekday fixed effects; drop-first over observed levels",
                            "eligibleConsecutivePairs": len(pairs), "n": len(samples),
                            "sampleStart": min((row["date"] for row in samples), default=None).isoformat() if samples else None,
                            "sampleEnd": max((row["date"] for row in samples), default=None).isoformat() if samples else None,
                            "excludedMissingOrInvalidFeature": len(pairs)-len(samples),
                            "primary": diagnostics._regression(samples, 9),
                            "sensitivity": diagnostics._regression(samples, 28)})
        for station in diagnostics.STATIONS:
            feature = "avgTa_7day_mean"
            samples, reasons = [], Counter()
            for pair in pairs:
                value, reason = diagnostics._weather_window_value(weather[station], pair["date"], feature)
                if value is None:
                    reasons[reason] += 1
                else:
                    samples.append({"date": pair["date"], "x": value, "y": pair["y"]})
            results.append({"testId": f"{product}.station_{station}.avgTa_7day_mean", "productId": product,
                            "feature": "avgTa_7day_mean", "station": station,
                            "featureUnit": "°C; exact prior seven calendar days",
                            "outcome": "100×absolute log price change (%)",
                            "controls": "intercept + month/year/weekday fixed effects; drop-first over observed levels",
                            "eligibleConsecutivePairs": len(pairs), "n": len(samples),
                            "sampleStart": min((row["date"] for row in samples), default=None).isoformat() if samples else None,
                            "sampleEnd": max((row["date"] for row in samples), default=None).isoformat() if samples else None,
                            "excludedWindowsByReason": dict(sorted(reasons.items())),
                            "primary": diagnostics._regression(samples, 9),
                            "sensitivity": diagnostics._regression(samples, 28)})
    if len(results) != TEST_COUNT:
        raise ValueError(f"fixed association family must contain {TEST_COUNT} tests")
    for bandwidth in ("primary", "sensitivity"):
        rows = [{"pValue": row[bandwidth]["pValue"] if row[bandwidth]["pValue"] is not None else 1.0,
                 "result": row} for row in results]
        experiments._apply_holm(rows)
        for item in rows:
            statistic = item["result"][bandwidth]
            statistic["holmAdjustedPValue"] = item["holmAdjustedPValue"]
            statistic["holmSignificantAt005"] = statistic["status"] == "available" and item["holmAdjustedPValue"] < ALPHA
    return results


def build_report(db: sqlite3.Connection, source: dict[str, Any], source_bytes: bytes,
                 weather_values: dict[str, dict[dt.date, dict[str, Any]]] | None = None,
                 weather_provenance: dict[str, Any] | None = None) -> dict[str, Any]:
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    if source_hash != FROZEN_SHA256:
        raise ValueError("frozen prediction source SHA-256 mismatch")
    if source.get("schemaVersion") != 2:
        raise ValueError("expected frozen dashboard schemaVersion 2")
    panel_products = {}
    for product in PRODUCTS:
        product_data, model, baseline, exported_actuals, common = _selected_predictions(source, product)
        if len(common) != 344:
            raise ValueError(f"{product} expected 344 common daily-h1 actuals; got {len(common)}")
        panel_products[product] = (product_data, exported_actuals, common, model, baseline)
    db_data = extract_prices(db)
    for product in PRODUCTS:
        product_data, exported, common, _, _ = panel_products[product]
        actual_series = db_data["series"][product]
        export_context = {}
        for row in product_data.get("observations", []):
            day = _date(row.get("date"), "export date")
            if CONTEXT_START <= day <= EVALUATION_END:
                export_context[day] = row
        db_context = {day: row for day, row in actual_series.items() if CONTEXT_START <= day <= EVALUATION_END}
        if set(export_context) != set(db_context):
            raise ValueError(f"frozen export and selected DB context dates differ for {product}")
        if any("rowCount" not in row for row in export_context.values()):
            raise ValueError(f"frozen export omits rowCount needed for source reconciliation: {product}")
        row_count_comparison = "verified"
        for day, source_row in export_context.items():
            obs = db_context[day]
            source_price = _number(source_row.get("actual"), "export actual")
            source_volume = _number(source_row.get("volumeKg"), "export volumeKg")
            if (not math.isclose(obs["actual"], source_price, rel_tol=1e-10, abs_tol=1e-7)
                    or not math.isclose(obs["volumeKg"], source_volume, rel_tol=1e-10, abs_tol=1e-7)):
                raise ValueError(f"database target mismatch with frozen export for {product} on {day}")
            if "rowCount" in source_row:
                if isinstance(source_row["rowCount"], bool) or int(source_row["rowCount"]) != obs["rowCount"]:
                    raise ValueError(f"database row-count mismatch with frozen export for {product} on {day}")
        if any(day not in actual_series for day in common):
            raise ValueError(f"common prediction target missing selected DB observation for {product}")
        panel_products[product] += (row_count_comparison,)

    if weather_values is None:
        selected, wq = diagnostics._weather_plans(db, CONTEXT_START, EVALUATION_END)
        weather_values, weather_provenance = diagnostics._weather_records(db, selected, CONTEXT_START, EVALUATION_END)
        weather_provenance = {"selection": wq, "records": weather_provenance,
                              "publicationTimesUnknown": True}
    products = {}
    price_series = {}
    composition = {}
    evaluation_dates = {}
    for product in PRODUCTS:
        product_data, exported, common, model, baseline, row_count_comparison = panel_products[product]
        series = db_data["series"][product]
        market_rows = db_data["marketRows"][product]
        events, spikes, cutoff = _events(series, market_rows, common, baseline, model, exported, weather_values)
        if len(events) != len(common):
            raise ValueError(f"{product} lacks a previous observed price for one or more common targets")
        spike_dates = {row["date"] for row in spikes}
        nonspikes = [row for row in events if row["date"] not in spike_dates]
        spike_rows = [row for row in events if row["date"] in spike_dates]
        observed_days = sorted(d for d in series if CONTEXT_START <= d <= EVALUATION_END)
        prior_actual = sorted(d for d in series if d < EVALUATION_START)
        low_cut = statistics.quantiles([series[d]["actual"] for d in common], n=4, method="inclusive")[0]
        low_rows = [row for row in events if row["actual"] <= low_cut]
        ape_sorted = sorted(events, key=lambda row: (-row["ridgeAPE"], row["date"]))
        ape_top_n = math.ceil(len(events) * 0.10)
        ape_top = ape_sorted[:ape_top_n]
        all_ape = sum(row["ridgeAPE"] for row in events)
        low_ape = sum(row["ridgeAPE"] for row in low_rows)
        next_by_event = [row["nextObservation"] for row in spikes if row["nextObservation"]]
        composition_top = spikes[:5]
        price_series[product] = series
        # Build denominator-stable composition inputs for regression on all calendar pairs.
        composition[product] = {}
        for day, rows in market_rows.items():
            composition[product][day] = _daily_composition(rows)
        evaluation_dates[product] = common
        products[product] = {"name": dashboard.PRODUCTS[product]["name"],
            "evaluationCommonN": len(events), "spikeN": len(spikes),
            "spikeDefinition": {"measure": "abs(log(actual_t / previous_observed_actual))",
                "topFraction": SPIKE_FRACTION, "selectionCount": len(spikes),
                "cutoffAbsLogChange": cutoff,
                "cutoffDirectionSpecificPercentChanges": {
                    "upPct": (math.expm1(cutoff)*100 if cutoff is not None else None),
                    "downPct": ((1-math.exp(-cutoff))*100 if cutoff is not None else None)},
                "ties": "descending absolute log change, then ascending target date"},
            "summary": {"all": _subset_summary(events), "spikeTopDecile": _subset_summary(spike_rows),
                "nonSpike": _subset_summary(nonspikes),
                "forecastErrors": {"all": {kind: _forecast_error_summary(events, events, kind)
                        for kind in ("baseline", "ridge")},
                    "spikeTopDecile": {kind: _forecast_error_summary(spike_rows, events, kind)
                        for kind in ("baseline", "ridge")},
                    "nonSpike": {kind: _forecast_error_summary(nonspikes, events, kind)
                        for kind in ("baseline", "ridge")}},
                "lowActualBottomQuartile": {"cutoffActual": low_cut, "n": len(low_rows),
                    "ridgeAPEContributionShare": low_ape/all_ape if all_ape else None,
                    "ridgeMeanAPE": statistics.fmean(r["ridgeAPE"] for r in low_rows) if low_rows else None},
                "ridgeApeTopDecile": {"n": len(ape_top), "selectionCount": ape_top_n,
                    "apeContributionShare": sum(r["ridgeAPE"] for r in ape_top)/all_ape if all_ape else None,
                    "dates": [r["date"] for r in ape_top]},
                "forecastErrorsOnCommonTargets": {"n": len(events),
                    "baseline": _forecast_metrics(events, "baseline"), "ridge": _forecast_metrics(events, "ridge")}},
            "spikeEvents": events,
            "topFiveAbsoluteLogChangeEvents": composition_top,
            "diagnostics": {"availableDbDatesContext": len(observed_days),
                "contextFirstDate": min(observed_days).isoformat() if observed_days else None,
                "contextLastDate": max(observed_days).isoformat() if observed_days else None,
                "priorContextObservationCount": len(prior_actual),
                "exportRowCountComparison": row_count_comparison,
                "allSelectedRowsByMarket": _market_row_summary(market_rows, common),
                "allEventCellDecompositionAvailable": sum(r["cellDecomposition"]["status"] == "available" for r in events),
                "allEventCellDecompositionUnavailable": sum(r["cellDecomposition"]["status"] != "available" for r in events),
                "topDecileNextObservationAvailable": len(next_by_event)}}

    associations = _associations(price_series, composition, weather_values, evaluation_dates)
    return {"schemaVersion": 1, "generatedAt": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "protocol": {"contextStart": CONTEXT_START.isoformat(), "evaluationStart": EVALUATION_START.isoformat(),
            "evaluationEnd": EVALUATION_END.isoformat(), "predictionSourceSha256": source_hash,
            "predictionSourceEvaluationPeriod": {"start": source["meta"]["evaluationStart"], "end": source["meta"]["evaluationEnd"]},
            "sourcePriceDefinition": "dashboard selected latest-complete aT normalized rows; trd_se=경매 and unit_nm=kg; signed totprc / signed unit_tot_qty; no domestic/origin filtering",
            "marketCodes": list(dashboard.MARKETS), "cellKey": ["market", "gds_sclsf_cd", "grd_cd"],
            "missingDimension": MISSING, "publishedAtKnown": False,
            "spikeDefinition": "top ceil(10%) by absolute log change from immediately previous observed price date in common daily-h1 sample",
            "associationFamilySize": TEST_COUNT, "associationFamily": "2 products × (volume-change, market TVD, grade TVD, 3 station weather features)",
            "controls": "intercept + month/year/weekday fixed effects", "HACCalendarLagsDays": [9, 28],
            "holmAlpha": ALPHA, "interpretation": "retrospective descriptive associations; no causality claim; weather was not operationally available by publication vintage",
            "selectionPolicy": "coverage.SELECTION_CTE / dashboard.PRICE_SQL latest validated completed query epoch"},
        "provenance": {"databaseReadOnly": True, "selectedPriceRequests": db_data["provenance"],
            "weather": weather_provenance or {}, "sources": {"predictionExport": "frozen dashboard predictions.json", "weatherApi": collector.ASOS},
            "sourcePublicationTimes": "unknown; retrieved response vintage only"},
        "products": products, "associations": associations}


def _export_volume(source: dict[str, Any], product_id: str, day: dt.date) -> float:
    item = next(row for row in source["products"] if row["id"] == product_id)
    match = next(row for row in item["observations"] if row["date"] == day.isoformat())
    return _number(match["volumeKg"], "export volumeKg")


def _forecast_metrics(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    forecasts = [row for row in rows]
    if not forecasts:
        return {"mae": None, "mape": None, "rmse": None}
    prediction_key = "baselinePrediction" if kind == "baseline" else "ridgePrediction"
    errors = [abs(row[prediction_key]-row["actual"]) for row in forecasts]
    return {"mae": statistics.fmean(errors), "mape": statistics.fmean(100*e/row["actual"] for e,row in zip(errors,forecasts)),
            "rmse": math.sqrt(statistics.fmean(e*e for e in errors))}


def _forecast_error_summary(rows: list[dict[str, Any]], all_rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    key = "baselinePrediction" if kind == "baseline" else "ridgePrediction"
    def totals(items):
        ae = [abs(row[key] - row["actual"]) for row in items]
        ape = [100 * value / row["actual"] for value, row in zip(ae, items)]
        sse = [value * value for value in ae]
        return ae, ape, sse
    ae, ape, sse = totals(rows)
    total_ae, total_ape, total_sse = totals(all_rows)
    return {"n": len(rows), "mae": statistics.fmean(ae) if ae else None,
        "mape": statistics.fmean(ape) if ape else None,
        "rmse": math.sqrt(statistics.fmean(sse)) if sse else None,
        "sumAbsoluteError": sum(ae), "sumAbsolutePercentageError": sum(ape), "sumSquaredError": sum(sse),
        "totalAbsoluteErrorContributionShare": sum(ae)/sum(total_ae) if sum(total_ae) else None,
        "totalAbsolutePercentageErrorContributionShare": sum(ape)/sum(total_ape) if sum(total_ape) else None,
        "totalSquaredErrorContributionShare": sum(sse)/sum(total_sse) if sum(total_sse) else None}


def _market_row_summary(market_rows: dict[dt.date, dict[str, list[dict[str, Any]]]], dates: list[dt.date]):
    by_market = {}
    for market in dashboard.MARKETS:
        selected = [row for day in dates for row in market_rows.get(day, {}).get(market, [])]
        by_market[market] = {"rowCount": len(selected), "signedVolumeKg": float(sum((r["volume"] for r in selected), Decimal(0))),
                             "negativeVolumeRows": sum(r["volume"] < 0 for r in selected),
                             "zeroVolumeRows": sum(r["volume"] == 0 for r in selected)}
    return by_market


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# 배추·무 급변 거래구성 진단", "",
        f"평가: {report['protocol']['evaluationStart']}~{report['protocol']['evaluationEnd']}. "
        "수정본 기반 후향 진단이며 가격 원자료의 최초 공개시점은 확인되지 않았습니다.", "",
        "급변은 직전 관측 대비 절대 로그변화 상위 10%입니다. 아래는 기술적 연관성으로 인과 증거가 아닙니다.", "",
        "| 품목 | 공통표본 | 급변일 | 급변 cutoff (abs log) | 급변 거래량변화 | 급변 시장 TVD | 급변 등급 TVD | Ridge MAE | Ridge MAPE | 기준선 MAE | 기준선 MAPE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for product, value in report["products"].items():
        spike = value["summary"]["spikeTopDecile"]
        baseline = value["summary"]["forecastErrors"]["spikeTopDecile"]["baseline"]
        ridge = value["summary"]["forecastErrors"]["spikeTopDecile"]["ridge"]
        cutoff = value["spikeDefinition"]["cutoffAbsLogChange"]
        lines.append(f"| {value['name']} | {value['evaluationCommonN']} | {value['spikeN']} | {cutoff:.4f} | "
                     f"{_fmt(spike['meanAbsVolumeChangePct'])} | {_fmt(spike['meanMarketTvd'])} | {_fmt(spike['meanGradeTvd'])} | "
                     f"{_fmt(ridge['mae'])} | {_fmt(ridge['mape'])}% | {_fmt(baseline['mae'])} | {_fmt(baseline['mape'])}% |")
    for product, value in report["products"].items():
        lines.extend(["", f"## {value['name']}: 상위 급변 사례", "",
                      "| 날짜 | 직전 날짜 | 간격 | 이전가 | 실제가 | 로그변화(부호) | 거래량 변화 | 시장 TVD | 등급 TVD | Ridge APE | 분해상태 | 되돌림 다음 관측 |",
                      "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"])
        for row in value["topFiveAbsoluteLogChangeEvents"]:
            nxt = row["nextObservation"]
            back = f"{nxt['date']} ({nxt['returnFromEventPct']:+.1f}%)" if nxt else "끝에서 검열"
            lines.append(f"| {row['date']} | {row['previousDate']} | {row['gapDays']} | {row['previousActual']:.1f} | {row['actual']:.1f} | "
                         f"{row['signedLogChange']:+.4f} | {_fmt(row['volumeChangePct'])}% | {_fmt(row['marketTvd'])} | {_fmt(row['gradeTvd'])} | "
                         f"{row['ridgeAPE']:.1f}% | {row['cellDecomposition']['status']} | {back} |")
        lines.append("")
    lines.extend(["## 고정 회귀 12개", "", "회귀 표본은 연속 달력일만 사용하고, 기상 입력은 해당 날짜 직전 7일입니다. HAC 달력 lag 9/28일 및 별도 Holm 보정을 보고합니다. 이는 회고적 상관 분석이며 공개시점이 확인된 운영 예측변수가 아닙니다.", "",
        "| 품목 | 변수 | 지점 | n | 계수 | HAC9 p (Holm) | HAC28 p (Holm) | 상태 |",
        "|---|---|---:|---:|---:|---:|---:|---|"])
    for row in report["associations"]:
        primary, sensitivity = row["primary"], row["sensitivity"]
        pp = _fmt(primary.get("pValue")) if primary["status"] == "available" else "—"
        ph = _fmt(primary.get("holmAdjustedPValue")) if primary["status"] == "available" else "—"
        sp = _fmt(sensitivity.get("pValue")) if sensitivity["status"] == "available" else "—"
        sh = _fmt(sensitivity.get("holmAdjustedPValue")) if sensitivity["status"] == "available" else "—"
        lines.append(f"| {row['productId']} | {row['feature']} | {row.get('station') or '—'} | {row['n']} | {_fmt(primary['coefficient'])} | "
                     f"{pp} ({ph}) | {sp} ({sh}) | {primary['status']} |")
    lines.append("")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "—" if value is None else f"{float(value):.3f}"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        try: os.unlink(tmp_name)
        except FileNotFoundError: pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="data/agri.sqlite")
    parser.add_argument("--predictions", default="outputs/daily/2026-09-30/20260930T122342-844989ce/predictions.json")
    parser.add_argument("--output-dir", default="outputs/diagnostics/spikes-20260930")
    args = parser.parse_args(argv)
    try:
        raw = Path(args.predictions).read_bytes()
        source = json.loads(raw)
        db = diagnostics.coverage.open_readonly(args.db)
        try:
            db.execute("BEGIN")
            report = build_report(db, source, raw)
            db.rollback()
        finally:
            db.close()
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=False)
        _atomic_write(out / "report.json", json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        _atomic_write(out / "report.md", to_markdown(report))
        return 0
    except (OSError, sqlite3.Error, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"spike attribution failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
