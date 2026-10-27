"""Offline, read-only three-date R01 example concordance audit."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import dashboard, direct_horizon, market_extension, market_mix

ROOT = Path(__file__).resolve().parents[1]
DATES = ("2026-09-11", "2026-09-12", "2026-09-14")
OFFICIAL = {
    "2026-09-11": {"priceWonPerKg": 1164, "inboundTonnes": 1743},
    "2026-09-12": {"priceWonPerKg": 1164, "inboundTonnes": 1349},
    "2026-09-14": {"priceWonPerKg": 1277, "inboundTonnes": 1758},
}
OUTPUT = ROOT / "outputs/diagnostics/r01-concordance-20261007"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _classify_payload(query: dict[str, Any], payload: dict[str, Any], market: str,
                     specific_scopes: set[tuple[str, str, str]]) -> str | None:
    """Return the precise exclusion reason used for a candidate selected payload."""
    if dashboard._query_product(query) != "cabbage":
        return "query_product_mismatch"
    valid, reason = market_extension._valid_payload_scope(payload, query, "cabbage", market)
    if not valid:
        return reason
    day = dashboard._parse_date(query.get("cond[trd_clcln_ymd::EQ]"))
    query_sub = str(query.get("cond[gds_sclsf_cd::EQ]", ""))
    subclass = str(payload.get("gds_sclsf_cd", ""))
    if day and market_extension._overlap_shadowed("cabbage", day.isoformat(), market, query_sub, subclass, specific_scopes):
        return "overlap_shadowed_by_selected_specific_scope"
    if payload.get("trd_se") != "경매":
        return "trade_mode_not_auction"
    if payload.get("unit_nm") != "kg":
        return "unit_not_kg"
    if dashboard._as_decimal(payload.get("totprc")) is None:
        return "invalid_signed_amount"
    if dashboard._as_decimal(payload.get("unit_tot_qty")) is None:
        return "invalid_signed_net_kg"
    return None


def _is_cabbage_query(query: dict[str, Any]) -> bool:
    return dashboard._query_product(query) == "cabbage"


def _json_safe_broad_provenance(value: dict[str, Any]) -> dict[str, Any]:
    """Keep selector evidence while making nullable state counts JSON-sortable."""
    counts = value.get("selectedStates", {})
    return {
        "scopeCount": value.get("scopeCount"),
        "scopesSha256": value.get("scopesSha256"),
        "selectedStates": {
            ("ambiguous_or_unavailable" if state is None else str(state)): count
            for state, count in counts.items()
        },
        "sourceSelectionPolicy": value.get("sourceSelectionPolicy"),
    }


def summarize_rows(rows: list[dict[str, Any]], scope_states: dict[tuple[str, str], str],
                   totals: dict[tuple[str, str], dict[str, Any]],
                   response_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    """Summarize selected payloads with the same row eligibility as market_extension."""
    details: dict[tuple[str, str], dict[str, Any]] = {}
    for day in DATES:
        for market in dashboard.MARKETS:
            state = scope_states.get((day, market))
            if state not in ("complete", "empty"):
                raise ValueError(f"incomplete broad query for cabbage/{day}/{market}: {state or 'missing'}")
            details[(day, market)] = {"selectedState": state, "payloadRows": 0,
                "includedAuctionKgRows": 0, "excludedRows": 0, "dimensions": {},
                "excludedCategories": {}}
    for rec in rows:
        day, market = rec["date"], rec["market"]
        if day not in DATES or market not in dashboard.MARKETS:
            continue
        key = (day, market)
        entry = details[key]
        payload = rec.get("payload") or {}
        entry["payloadRows"] += 1
        large_code, large_name = payload.get("gds_lclsf_cd"), payload.get("gds_lclsf_nm")
        medium_code, medium_name = payload.get("gds_mclsf_cd"), payload.get("gds_mclsf_nm")
        subclass_code, subclass_name = payload.get("gds_sclsf_cd"), payload.get("gds_sclsf_nm")
        grade_code, grade_name = payload.get("grd_cd"), payload.get("grd_nm")
        origin_code, origin_name = payload.get("plor_cd"), payload.get("plor_nm")
        dims = {"queryProduct": rec.get("queryProduct"), "product": dashboard._product_id(payload),
            "productClass": large_code, "subclass": subclass_code, "grade": grade_name,
            "origin": origin_name, "productLargeClassCode": large_code,
            "productLargeClassName": large_name, "productMediumClassCode": medium_code,
            "productMediumClassName": medium_name, "subclassCode": subclass_code,
            "subclassName": subclass_name, "gradeCode": grade_code, "gradeName": grade_name,
            "originCode": origin_code, "originName": origin_name,
            "tradeMode": payload.get("trd_se"), "unit": payload.get("unit_nm")}
        label = json.dumps(dims, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        entry["dimensions"][label] = entry["dimensions"].get(label, 0) + 1
        reason = rec.get("exclusionReason")
        if reason is None:
            entry["includedAuctionKgRows"] += 1
        else:
            entry["excludedRows"] += 1
            entry["excludedCategories"][reason] = entry["excludedCategories"].get(reason, 0) + 1
    report_dates = {}
    for day in DATES:
        market_entries = {}
        for market in dashboard.MARKETS:
            item = details[(day, market)]
            raw_total = totals.get((day, market), {})
            if item["includedAuctionKgRows"] != raw_total.get("eligibleAuctionKgRows", 0):
                raise ValueError(f"raw eligibility count mismatch for cabbage/{day}/{market}")
            amount = Decimal(str(raw_total.get("signedNetAmount", 0)))
            kg = Decimal(str(raw_total.get("signedNetWeight", 0)))
            market_entries[market] = {**item, "signedNetAmount": str(amount),
                "signedNetKg": str(kg), "priceWonPerKg": float(amount / kg) if kg > 0 else None}
        amt = sum((Decimal(market_entries[m]["signedNetAmount"]) for m in dashboard.MARKETS), Decimal(0))
        kg = sum((Decimal(market_entries[m]["signedNetKg"]) for m in dashboard.MARKETS), Decimal(0))
        report_dates[day] = {"officialR01": OFFICIAL[day], "markets": market_entries,
            "twoMarketSignedNetAmount": str(amt), "twoMarketSignedNetKg": str(kg),
            "twoMarketPriceWonPerKg": float(amt / kg) if kg > 0 else None,
            "absolutePriceDifferenceWonPerKg": abs(float(amt / kg) - OFFICIAL[day]["priceWonPerKg"]) if kg > 0 else None}
    return {"dates": report_dates, "selectedResponseSha256": response_hashes or {},
        "selectedResponseHashCount": len(response_hashes or {}),
        "limitations": ["inbound t vs settled netkg not comparable", "R01 row inclusion/aggregation/vintage unknown"]}


def build_report(db_path: Path, prices_path: Path) -> dict[str, Any]:
    raw_bytes = prices_path.read_bytes()
    if _sha(raw_bytes) != market_mix.EXPECTED_PRICES_SHA256:
        raise ValueError("prices snapshot SHA-256 differs from the pinned market_mix input")
    raw = json.loads(raw_bytes)
    core = direct_horizon._observations(raw)
    uri = "file:" + str(db_path.resolve()) + "?mode=ro"
    db = sqlite3.connect(uri, uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    try:
        db.execute("BEGIN")
        start = dt.date.fromisoformat(raw["meta"]["contextStart"])
        end = dt.date.fromisoformat(raw["meta"]["contextEnd"])
        states, broad_provenance = market_extension._core_broad_query_states(db, start)
        _, selected_scopes, diag = market_extension._extract_core_market_prices(db, start, end, True)
        reconciliation = market_mix._reconcile(raw, core, states, diag, end)
        broad = {(day, market): states.get(("cabbage", dt.date.fromisoformat(day), market), {}).get("selectedState")
                 for day in DATES for market in dashboard.MARKETS}
        request_hashes: dict[str, str] = {}
        request_ids: set[int] = set()
        cte = market_extension._scoped_selection_cte(dt.date.fromisoformat(DATES[0]),
            dt.date.fromisoformat(DATES[-1]), tuple(dashboard.MARKETS))
        for scope in db.execute(cte + """SELECT p.selected_job_id,p.selected_epoch_id,p.task_key,p.source_url,p.query_json
                FROM picked p WHERE p.dataset='prices_origin'"""):
            try:
                query = json.loads(scope[4])
            except (TypeError, json.JSONDecodeError):
                continue
            date = dashboard._parse_date(query.get("cond[trd_clcln_ymd::EQ]"))
            market = str(query.get("cond[whsl_mrkt_cd::EQ]", ""))
            if not date or date.isoformat() not in DATES or market not in dashboard.MARKETS:
                continue
            if dashboard._query_product(query) != "cabbage":
                continue
            for request_id, sha in db.execute("""SELECT request_id,response_sha256 FROM source_requests
                    WHERE job_id=? AND epoch_id=? AND dataset='prices_origin' AND source_url=?
                      AND coverage_task_key(dataset,source_url,query_json)=?""",
                    (scope[0], scope[1], scope[3], scope[2])):
                request_ids.add(int(request_id))
                if sha:
                    request_hashes[str(request_id)] = sha
        records = []
        scope_specific = set()
        for selected in db.execute(cte + """SELECT p.query_json,p.selected_state FROM picked p
                WHERE p.dataset='prices_origin'"""):
            try:
                query = json.loads(selected[0])
            except (TypeError, json.JSONDecodeError):
                continue
            day = dashboard._parse_date(query.get("cond[trd_clcln_ymd::EQ]"))
            market = str(query.get("cond[whsl_mrkt_cd::EQ]", ""))
            if day and day.isoformat() in DATES and market in dashboard.MARKETS:
                scope = market_extension._selected_specific_scope(query, selected[1])
                if scope:
                    scope_specific.add(scope)
        bounded_price_sql = cte + dashboard.PRICE_SQL[len(market_extension.coverage.SELECTION_CTE):]
        for rec in db.execute(bounded_price_sql, (DATES[0], DATES[-1], *dashboard.MARKETS.keys())):
            if rec["selected_state"] not in ("complete", "empty") or not rec["payload_json"]:
                continue
            try:
                query, payload = json.loads(rec["query_json"]), json.loads(rec["payload_json"])
            except (TypeError, json.JSONDecodeError):
                continue
            if not _is_cabbage_query(query):
                continue
            day = dashboard._parse_date(query.get("cond[trd_clcln_ymd::EQ]"))
            market = str(query.get("cond[whsl_mrkt_cd::EQ]", ""))
            if not day or day.isoformat() not in DATES or market not in dashboard.MARKETS:
                continue
            query_product = dashboard._query_product(query)
            reason = _classify_payload(query, payload, market, scope_specific)
            if day and day.isoformat() in DATES and market in dashboard.MARKETS:
                records.append({"date": day.isoformat(), "market": market, "payload": payload,
                    "queryProduct": query_product, "exclusionReason": reason})
        totals = {(day, market): diag.get(("cabbage", dt.date.fromisoformat(day), market), {})
                  for day in DATES for market in dashboard.MARKETS}
        summarized = summarize_rows(records, broad, totals, request_hashes)
        db.rollback()
    finally:
        db.close()
    return {"schemaVersion": 1, "audit": "R01 cabbage three-date source concordance",
        "viewedOn": "2026-09-15", "officialTemplatePage": 3,
        "pricesSnapshotSha256": _sha(raw_bytes), "marketMixReconciliation": reconciliation,
        "broadSelection": _json_safe_broad_provenance(broad_provenance),
        "selectedRequestCount": len(request_ids), "source": summarized}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=market_mix.DB_PATH)
    parser.add_argument("--prices", type=Path, default=market_mix.INPUT_PRICES)
    parser.add_argument("--output", type=Path, default=OUTPUT / "concordance.json")
    args = parser.parse_args(argv)
    report = build_report(args.db, args.prices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(args.output)
    print(json.dumps({"output": str(args.output), "dates": len(report["source"]["dates"]),
        "selectedResponseHashCount": report["source"]["selectedResponseHashCount"],
        "reconciliation": report["marketMixReconciliation"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
