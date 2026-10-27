"""Offline walk-forward experiment using Gwangju Seobu market information."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from . import coverage, dashboard, experiments, modeling

MARKET = "240004"
GARLIC_CLASSES = {"06", "16", "41", "42", "43"}
TRAIN_START, TRAIN_END = dt.date(2017, 1, 1), dt.date(2018, 12, 31)
EVAL_START, EVAL_END = dt.date(2019, 1, 1), dt.date(2020, 12, 31)
MIN_TRAIN = 90
MAX_AGE = 14
PRODUCTS = tuple(dashboard.PRODUCTS)
PAIRS = (("candidate", "control"), ("candidate", "persistence"))
LOSSES = ("MAE", "MAPE")
FAMILY_SIZE = len(PRODUCTS) * len(PAIRS) * len(LOSSES)
REVALIDATION_TRAIN_START = dt.date(2017, 1, 1)
REVALIDATION_EVAL_START = dt.date(2023, 1, 1)
REVALIDATION_MIN_CUTOFF = dt.date(2026, 9, 27)
REVALIDATION_FAMILY_SIZE = len(PRODUCTS) * 2 * len(LOSSES)
PROTOCOL = {
    "id": "gwangju-market-extension-2026-10-05",
    "initialTraining": [TRAIN_START.isoformat(), TRAIN_END.isoformat()],
    "evaluation": [EVAL_START.isoformat(), EVAL_END.isoformat()],
    "walkForward": "expanding; each target label strictly before forecast origin",
    "horizonDays": 1, "maximumSourceAgeDays": MAX_AGE, "minimumTrainingTargetDays": MIN_TRAIN,
    "target": "five Garak/Gangseo auction/kg product targets; signed totprc / signed unit_tot_qty; Ridge target log(target/as-of source) with existing inverse",
    "controlFeatures": "core price, volume, lags, rolling price summaries and calendar",
    "candidateFeatures": "control + Gwangju last price/weight/age + log price differences to last eligible Garak and Gangseo prices",
    "gwangjuGarlicSubclasses": sorted(GARLIC_CLASSES),
    "model": {"scaler": "StandardScaler fitted per origin", "estimator": "Ridge", "alpha": 10},
    "evaluationSample": "common target dates with target actual, valid core source and valid Gwangju source",
    "dm": {"twoSided": True, "primaryCalendarBartlettHacDays": 9, "sensitivityCalendarBartlettHacDays": 28,
           "fixedHolmFamilySize": FAMILY_SIZE, "holmAlpha": 0.05, "minimumPairs": 30,
           "unavailableTestsRemainInFamilyAsP1": True},
    "availabilityClaim": "retrospective retrieved revision; first publication time unknown",
}
REVALIDATION_PROTOCOL = {
    "id": "gwangju-market-extension-revalidation-2026-10-06",
    "initialTrainingStart": REVALIDATION_TRAIN_START.isoformat(),
    "trainingUsesTargetsStrictlyBeforeEachOrigin": True,
    "evaluationStart": REVALIDATION_EVAL_START.isoformat(),
    "evaluationEnd": "derived from longest contiguous verified daily broad-query coverage across five products and three markets; minimum 2026-09-27",
    "horizonDays": 1,
    "maximumSourceAgeDays": MAX_AGE,
    "minimumTrainingTargetDays": MIN_TRAIN,
    "target": PROTOCOL["target"],
    "controlFeatures": PROTOCOL["controlFeatures"],
    "candidateFeatures": PROTOCOL["candidateFeatures"],
    "model": PROTOCOL["model"],
    "comparisons": ["candidate vs control", "candidate vs persistence"],
    "dm": {"twoSided": True, "primaryCalendarBartlettHacDays": 9,
        "sensitivityCalendarBartlettHacDays": 28, "fixedHolmFamilySize": REVALIDATION_FAMILY_SIZE,
        "holmAlpha": 0.05, "minimumPairs": 30,
        "inferenceGate": {"minimumPairedDates": 90, "minimumYears": 2,
            "minimumSeasons": 3, "minimumDatesPerSeason": 10},
        "unavailableTestsRemainInFamilyAsP1": True},
    "scope": "separate post-2022 exploratory revalidation; prior 2017-2020 report is unchanged and not pooled",
    "availabilityClaim": "retrospective retrieved revision; first publication time unknown; no 09:00 or prospective performance claim",
    "limitations": "a validated empty response is distinct from a failed request, but the available records do not identify whether an empty day was a market closure or an open-day no-trade; closure status remains unknown",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _scoped_selection_cte(start: dt.date, end: dt.date, markets: tuple[str, ...]) -> str:
    """Keep the collector's epoch validation while limiting it to required scopes."""
    market_sql = ",".join("'" + market.replace("'", "''") + "'" for market in markets)
    qdate = "json_extract(query_json,'$.\"cond[trd_clcln_ymd::EQ]\"')"
    qmarket = "json_extract(query_json,'$.\"cond[whsl_mrkt_cd::EQ]\"')"
    scope = f"{qdate} BETWEEN '{start.isoformat()}' AND '{end.isoformat()}' AND {qmarket} IN ({market_sql})"
    cte = coverage.SELECTION_CTE
    cte = cte.replace("FROM source_requests WHERE dataset IN ('prices_origin','prices_sale')",
                      f"FROM source_requests WHERE dataset='prices_origin' AND {scope}", 1)
    cte = cte.replace("WHERE p.dataset IN ('prices_origin','prices_sale')",
                      f"WHERE p.dataset='prices_origin' AND {scope.replace('query_json','p.query_json')}", 1)
    cte = cte.replace("WHERE r.dataset IN ('prices_origin','prices_sale')",
                      f"WHERE r.dataset='prices_origin' AND {scope}", 1)
    if cte == coverage.SELECTION_CTE:
        raise RuntimeError("failed to scope collector selection CTE")
    return cte


def _verified_common_cutoff(states: dict[tuple[str, dt.date, str], Any],
                            start: dt.date = REVALIDATION_EVAL_START,
                            minimum: dt.date = REVALIDATION_MIN_CUTOFF) -> tuple[dt.date, dict[str, Any]]:
    """Require a contiguous calendar run of selected, validated broad scopes."""
    relevant = {(p, d, m): value for (p, d, m), value in states.items()
                if p in PRODUCTS and m in (MARKET, *dashboard.MARKETS) and d >= start}
    latest = max((d for _, d, _ in relevant), default=None)
    if latest is None:
        raise ValueError("no verified broad-query scopes exist on or after 2023-01-01")
    bad_dates = []
    cutoff = start - dt.timedelta(days=1)
    day = start
    while day <= latest:
        missing = []
        for product in PRODUCTS:
            for market in (MARKET, *dashboard.MARKETS):
                value = relevant.get((product, day, market))
                state = value.get("selectedState") if isinstance(value, dict) else value
                if state not in ("complete", "empty"):
                    missing.append({"productId": product, "marketCode": market,
                        "reason": "missing_scope" if value is None else "not_validated",
                        "latestState": value.get("latestState") if isinstance(value, dict) else None})
        if missing:
            bad_dates = missing
            break
        cutoff = day
        day += dt.timedelta(days=1)
    if cutoff < minimum:
        detail = json.dumps(bad_dates[:6], ensure_ascii=False, sort_keys=True)
        raise ValueError(f"verified common cutoff {cutoff.isoformat()} is earlier than required minimum "
                         f"{minimum.isoformat()}; first gap details: {detail}")
    return cutoff, {"coverageStart": start.isoformat(), "coverageEnd": cutoff.isoformat(),
                    "firstGapDate": day.isoformat() if day <= latest else None,
                    "firstGapScopes": bad_dates, "latestPlannedScopeDate": latest.isoformat(),
                    "requiredBroadScopesPerDate": len(PRODUCTS) * 3,
                    "selectedCompleteOrEmptyScopeCount": sum(1 for (p,d,m),value in relevant.items()
                        if d <= cutoff and (value.get("selectedState") if isinstance(value,dict) else value) in ("complete","empty"))}


def _core_broad_query_states(db: sqlite3.Connection, start: dt.date = REVALIDATION_EVAL_START
                             ) -> tuple[dict[tuple[str, dt.date, str], dict[str, Any]], dict[str, Any]]:
    """Select date/product/market broad-query states using the shared validated epoch CTE."""
    markets = (MARKET, *dashboard.MARKETS)
    planned = []
    for row in db.execute("SELECT query_json FROM planned_queries WHERE dataset='prices_origin'"):
        try:
            query = json.loads(row[0])
            day_text = coverage.canonical_day(query.get("cond[trd_clcln_ymd::EQ]"))
            market = str(query.get("cond[whsl_mrkt_cd::EQ]", ""))
            if not day_text or day_text < start.isoformat() or market not in markets:
                continue
            if query.get("cond[gds_sclsf_cd::EQ]", ""):
                continue
            dimensions = coverage.price_dimensions(query, "prices_origin")
            if dimensions and dimensions[2] in PRODUCTS:
                planned.append(dt.date.fromisoformat(day_text))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    if not planned:
        raise ValueError("no relevant broad price-query plans on or after 2023-01-01")
    search_end = max(planned)
    db.create_function("coverage_task_key", 3, coverage._request_task_key, deterministic=True)
    cte = _scoped_selection_cte(start, search_end, markets)
    states: dict[tuple[str, dt.date, str], dict[str, Any]] = {}
    manifest = []
    query = cte + """SELECT p.query_json,p.selected_state,p.latest_state,p.selected_job_id,
        p.selected_epoch_id,p.task_key,p.selected_row_count,p.selected_previous
        FROM picked p WHERE p.dataset='prices_origin'"""
    for row in db.execute(query):
        try:
            source_query = json.loads(row[0])
            day_text = coverage.canonical_day(source_query.get("cond[trd_clcln_ymd::EQ]"))
            market = str(source_query.get("cond[whsl_mrkt_cd::EQ]", ""))
            if not day_text or source_query.get("cond[gds_sclsf_cd::EQ]", ""):
                continue
            dimensions = coverage.price_dimensions(source_query, "prices_origin")
            product = dimensions[2] if dimensions else None
            if product not in PRODUCTS or market not in markets:
                continue
            day = dt.date.fromisoformat(day_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        key = product, day, market
        entry = {"selectedState": row[1], "latestState": row[2], "selectedJobId": row[3],
                 "selectedEpochId": row[4], "taskKey": row[5], "selectedRowCount": row[6],
                 "selectedPrevious": bool(row[7])}
        if key in states and states[key] != entry:
            states[key] = {"selectedState": None, "latestState": "ambiguous_duplicate_scope"}
        else:
            states[key] = entry
        manifest.append({"productId": product, "date": day_text, "marketCode": market,
            "querySha256": _sha(json.dumps(source_query, sort_keys=True, separators=(",", ":")).encode()),
            **entry})
    scope_digest = _sha(json.dumps(sorted(manifest, key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":"))),
        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())
    return states, {"scopeCount": len(manifest), "scopesSha256": scope_digest,
        "selectedStates": {state: sum(1 for x in states.values() if x["selectedState"] == state)
                           for state in ("complete", "empty", None)},
        "sourceSelectionPolicy": "shared coverage.SELECTION_CTE latest validated completed epoch; complete/empty are usable; missing/failed scopes are never imputed"}


def _coverage_gate(targets: list[dt.date]) -> dict[str, Any]:
    years=sorted({d.year for d in targets})
    def season(day):
        return "winter" if day.month in (12,1,2) else "spring" if day.month in (3,4,5) else "summer" if day.month in (6,7,8) else "autumn"
    seasons=defaultdict(int)
    for day in targets: seasons[season(day)]+=1
    passed=len(targets)>=90 and len(years)>=2 and sum(n>=10 for n in seasons.values())>=3
    return {"passed":passed,"pairedDates":len(targets),"years":years,
        "seasonCounts":dict(sorted(seasons.items())),"required":{"pairedDates":90,"years":2,
        "seasons":3,"datesPerSeason":10}}


def _season_name(day: dt.date) -> str:
    return "winter" if day.month in (12,1,2) else "spring" if day.month in (3,4,5) else "summer" if day.month in (6,7,8) else "autumn"


def _constant_difference(values: list[float]) -> bool:
    """Detect a constant loss difference despite floating-point rounding noise."""
    if len(values) < 2:
        return True
    scale = max(1.0, max(abs(value) for value in values))
    return max(values) - min(values) <= 1e-12 * scale


def _unobserved_target_reason(product: str, day: dt.date,
                              states: dict[tuple[str,dt.date,str],dict[str,Any]], diagnostics) -> str:
    core_markets=tuple(dashboard.MARKETS)
    entries=[states.get((product,day,market)) for market in core_markets]
    if any(value is None for value in entries): return "unqueried_scope"
    if any(value.get("selectedState") not in ("complete","empty") for value in entries):
        latest=[value.get("latestState") for value in entries if value]
        return "api_failure" if any(state in ("failed","error") for state in latest) else "unverified_scope"
    if all(value["selectedState"]=="empty" for value in entries): return "verified_empty_response_closure_or_no_trade_unknown"
    row_count=sum(diagnostics.get((product,day,market),{}).get("eligibleAuctionKgRows",0) for market in core_markets)
    net_weight=sum(diagnostics.get((product,day,market),{}).get("signedNetWeight",Decimal(0)) for market in core_markets)
    net_amount=sum(diagnostics.get((product,day,market),{}).get("signedNetAmount",Decimal(0)) for market in core_markets)
    if row_count==0: return "no_eligible_auction_kg_rows"
    if net_weight<=0: return "nonpositive_net_weight"
    if net_amount<=0 or net_amount/net_weight<=0: return "nonpositive_net_price"
    return "positive_source_target_missing_from_input_snapshot"


def _validate_revalidation_input(raw: dict[str,Any], core, states, diagnostics,
                                 cutoff: dt.date) -> dict[str,Any]:
    meta=raw.get("meta")
    if not isinstance(meta,dict): raise ValueError("new-period prices input must include dashboard meta coverage")
    context_end=meta.get("contextEnd")
    try: context_end=dt.date.fromisoformat(context_end)
    except (TypeError,ValueError) as exc: raise ValueError("new-period input has no valid meta.contextEnd") from exc
    context_start=meta.get("contextStart")
    try: context_start=dt.date.fromisoformat(context_start)
    except (TypeError,ValueError) as exc: raise ValueError("new-period input has no valid meta.contextStart") from exc
    if context_start>REVALIDATION_TRAIN_START:
        raise ValueError(f"prices input contextStart {context_start} follows required training start {REVALIDATION_TRAIN_START}")
    if context_end<cutoff:
        raise ValueError(f"prices input contextEnd {context_end} precedes verified cutoff {cutoff}")
    matched=0;observed=0;date_count=(cutoff-REVALIDATION_EVAL_START).days+1
    for product in PRODUCTS:
        for offset in range(date_count):
            day=REVALIDATION_EVAL_START+dt.timedelta(days=offset)
            if any((product,day,market) not in states or states[(product,day,market)].get("selectedState") not in ("complete","empty")
                   for market in dashboard.MARKETS):
                continue
            amount=Decimal(0);weight=Decimal(0)
            for market in dashboard.MARKETS:
                value=diagnostics.get((product,day,market),{})
                amount+=value.get("signedNetAmount",Decimal(0));weight+=value.get("signedNetWeight",Decimal(0))
            expected=(float(amount/weight) if weight>0 and amount/weight>0 else None)
            input_row=core[product].get(day)
            if expected is None:
                if input_row is not None: raise ValueError(f"input/database target mismatch at {product}/{day}: snapshot has a price but selected rows do not")
            else:
                if input_row is None: raise ValueError(f"input/database target mismatch at {product}/{day}: validated target absent from prices snapshot")
                if not math.isclose(input_row["price"],expected,rel_tol=1e-10,abs_tol=1e-8) or not math.isclose(input_row["volume"],float(weight),rel_tol=1e-10,abs_tol=1e-8):
                    raise ValueError(f"input/database target mismatch at {product}/{day}: price or signed net weight differs")
                observed+=1
            matched+=1
    return {"contextStart":context_start.isoformat(),"contextEnd":context_end.isoformat(),
        "evaluationTargetsChecked":matched,"positiveSourceTargetsReconciled":observed,
        "rule":"for each product/day through cutoff with validated Garak/Gangseo broad scopes, compare prices.json actual and net kg with selected signed DB rows"}


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def latest_prices_file(root: Path) -> Path:
    root=root.resolve()
    candidates = []
    for path in (root / "outputs/daily").glob("*/*/prices.json"):
        try:
            status = json.loads((path.parent / "run-status.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        run_directory=status.get("runDirectory")
        if not isinstance(run_directory,str):
            continue
        reported_dir=Path(run_directory)
        if not reported_dir.is_absolute(): reported_dir=root/reported_dir
        if status.get("status") == "complete" and reported_dir.resolve() == path.parent.resolve():
            candidates.append((str(status.get("finishedAtKst", "")), path))
    if not candidates: raise ValueError("no completed daily run containing prices.json")
    return max(candidates, key=lambda x: (x[0], str(x[1])))[1]


def _load_core_bytes(raw_bytes: bytes) -> tuple[dict[str, Any], dict[str, dict[dt.date, dict[str, Any]]]]:
    raw = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(raw, dict) or raw.get("schemaVersion") != 1:
        raise ValueError("core prices.json must have dashboard schemaVersion 1")
    products = {p["id"]: p for p in raw.get("products", [])}
    if set(products) != set(PRODUCTS): raise ValueError("core prices.json must contain all five products")
    maps = {}
    for product, item in products.items():
        obs = {}
        for row in item.get("observations", []):
            day = dt.date.fromisoformat(row["date"])
            price, volume = float(row["actual"]), float(row["volumeKg"])
            if not math.isfinite(price) or price <= 0 or not math.isfinite(volume) or volume <= 0:
                raise ValueError(f"invalid price observation {product} {day}")
            if day in obs: raise ValueError(f"duplicate core observation {product} {day}")
            markets = row.get("marketData", {})
            obs[day] = {"price": price, "volume": volume, "marketData": markets if isinstance(markets, dict) else {}}
        maps[product] = dict(sorted(obs.items()))
    return raw, maps


def _job_info(db: sqlite3.Connection, job_id: int, expected_start: str, expected_end: str) -> dict[str, Any]:
    row = db.execute("SELECT command,config_json,state FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    if not row or row[0] != "prices" or row[2] != "complete": raise ValueError(f"job {job_id} is not a completed prices job")
    config = json.loads(row[1])
    if config.get("market") != [MARKET] or config.get("start") != expected_start or config.get("end") != expected_end:
        raise ValueError(f"job {job_id} does not match the frozen Gwangju date/market scope")
    return config


def _valid_payload_scope(payload: dict[str, Any], query: dict[str, Any], product: str,
                         market: str) -> tuple[bool, str]:
    query_day = coverage.canonical_day(query.get("cond[trd_clcln_ymd::EQ]"))
    source_day = dashboard._parse_date(payload.get("trd_clcln_ymd"))
    if source_day is None or source_day.isoformat() != query_day: return False, "source_date_request_mismatch"
    query_market = str(query.get("cond[whsl_mrkt_cd::EQ]", ""))
    if query_market != market or str(payload.get("whsl_mrkt_cd", "")) != query_market: return False, "market_mismatch"
    if dashboard._product_id(payload) != product: return False, "payload_category_mismatch"
    subclass = str(payload.get("gds_sclsf_cd", ""))
    if product == "garlic" and not dashboard._garlic_class_is_verified(payload): return False, "not_verified_peeled_garlic_class"
    query_subclass = str(query.get("cond[gds_sclsf_cd::EQ]", ""))
    if query_subclass and query_subclass != subclass: return False, "payload_subclass_query_mismatch"
    return True, subclass


def _overlap_shadowed(product: str, date_text: str, market: str, query_subclass: str,
                      payload_subclass: str, specific_scopes: set[tuple[str, str, str]]) -> bool:
    return bool(product == "garlic" and not query_subclass and
                (date_text, market, payload_subclass) in specific_scopes)


def _selected_specific_scope(query: dict[str, Any], state: str | None) -> tuple[str, str, str] | None:
    dims=coverage.price_dimensions(query,"prices_origin")
    date_text=coverage.canonical_day(query.get("cond[trd_clcln_ymd::EQ]"))
    market=str(query.get("cond[whsl_mrkt_cd::EQ]",""))
    subclass=str(query.get("cond[gds_sclsf_cd::EQ]",""))
    if dims and dims[2]=="garlic" and subclass and date_text and state in ("complete","empty"):
        return date_text,market,subclass
    return None


def extract_gwangju(db: sqlite3.Connection) -> tuple[dict[str, dict[dt.date, dict[str, Any]]], dict[str, Any]]:
    info = [_job_info(db, 30, "2017-01-01", "2018-12-31"), _job_info(db, 31, "2019-01-01", "2020-12-31")]
    db.create_function("coverage_task_key", 3, coverage._request_task_key, deterministic=True)
    cte = _scoped_selection_cte(TRAIN_START, EVAL_END, (MARKET,))
    selected = list(db.execute(cte + """SELECT p.latest_job_id,p.selected_job_id,p.selected_epoch_id,
        p.source_url,p.task_key,p.query_json,p.selected_state,p.latest_state
        FROM picked p WHERE p.dataset='prices_origin' AND (p.latest_job_id IN (30,31) OR p.selected_job_id IN (30,31))"""))
    query_states = defaultdict(int)
    specific_scopes = set()
    selection_manifest=[]
    for latest_job,selected_job,epoch,url,task,query_text,state,latest_state in selected:
        key=json.loads(query_text); product_info=coverage.price_dimensions(key,"prices_origin")
        product=product_info[2] if product_info else ""
        date_text=coverage.canonical_day(key.get("cond[trd_clcln_ymd::EQ]"))
        market=str(key.get("cond[whsl_mrkt_cd::EQ]","")); sub=str(key.get("cond[gds_sclsf_cd::EQ]",""))
        query_states[state or f"unavailable:{latest_state or 'unqueried'}"] += 1
        selection_manifest.append({"jobId":selected_job,"epochId":epoch,"taskKey":task,"product":product,"date":date_text,
                                  "market":market,"subclass":sub or None,"selectedState":state,
                                  "latestJobId":latest_job,"latestState":latest_state})
        scope=_selected_specific_scope(key,state)
        if scope:specific_scopes.add(scope)
    rows = db.execute(cte + """SELECT r.request_id,s.response_sha256,p.selected_state,p.query_json,
        n.row_no,n.payload_json,n.quality_flags FROM picked p
        JOIN request_pages r ON r.job_id=p.selected_job_id AND r.dataset=p.dataset
          AND r.source_url=p.source_url AND r.task_key=p.task_key AND r.epoch_id=p.selected_epoch_id
          AND r.state='complete'
        JOIN source_requests s ON s.request_id=r.request_id
        LEFT JOIN normalized_records n ON n.request_id=r.request_id
        WHERE p.dataset='prices_origin' AND p.selected_job_id IN (30,31)
        ORDER BY p.query_json,r.request_id,n.row_no""")
    values: dict[tuple[str, dt.date, str], dict[str, Decimal]] = defaultdict(lambda: {"amount": Decimal(0), "weight": Decimal(0), "rows": Decimal(0)})
    req_hashes, reqids = {}, set()
    for row in rows:
        rid, rsha, state, query_text, _, payload_text, _ = row
        reqids.add(rid); req_hashes[str(rid)] = rsha
        try: query = json.loads(query_text)
        except (TypeError, json.JSONDecodeError) as e: raise ValueError(f"invalid query_json in request {rid}") from e
        day_s = coverage.canonical_day(query.get("cond[trd_clcln_ymd::EQ]"))
        if not day_s or query.get("cond[whsl_mrkt_cd::EQ]") != MARKET: continue
        prod = coverage.price_dimensions(query, "prices_origin")
        product = prod[2] if prod else None
        if product not in PRODUCTS: continue
        subclass_query = str(query.get("cond[gds_sclsf_cd::EQ]", ""))
        if state != "complete" or payload_text is None: continue
        try: p = json.loads(payload_text)
        except json.JSONDecodeError: continue
        valid, subclass = _valid_payload_scope(p,query,product,MARKET)
        if not valid or _overlap_shadowed(product,day_s,MARKET,subclass_query,subclass,specific_scopes): continue
        if p.get("trd_se") != "경매" or p.get("unit_nm") != "kg": continue
        try:
            amount, weight = Decimal(str(p.get("totprc"))), Decimal(str(p.get("unit_tot_qty")))
        except (InvalidOperation, TypeError): continue
        if not amount.is_finite() or not weight.is_finite(): continue
        values[(product,dt.date.fromisoformat(day_s),subclass_query)]["amount"] += amount
        values[(product,dt.date.fromisoformat(day_s),subclass_query)]["weight"] += weight
        values[(product,dt.date.fromisoformat(day_s),subclass_query)]["rows"] += 1
    out = {p: {} for p in PRODUCTS}
    daily: dict[tuple[str,dt.date],dict[str,Decimal]]=defaultdict(lambda:{"amount":Decimal(0),"weight":Decimal(0),"rows":Decimal(0)})
    for (product,day,_scope),cell in values.items():
        daily[(product,day)]["amount"]+=cell["amount"];daily[(product,day)]["weight"]+=cell["weight"];daily[(product,day)]["rows"]+=cell["rows"]
    for (product, day), cell in daily.items():
        if cell["weight"] <= 0 or cell["amount"] / cell["weight"] <= 0: continue
        out[product][day] = {"price": float(cell["amount"] / cell["weight"]), "weight": float(cell["weight"]), "rows": int(cell["rows"])}
    provenance = {"jobIds": [30, 31], "requestIds": sorted(reqids), "responseHashes": req_hashes,
                  "requestCount": len(reqids), "jobConfigs": info,
                  "selectedQueryStates": dict(sorted(query_states.items())),
                  "selectedScopeCount":len(selection_manifest),
                  "selectedScopesSha256":_sha(json.dumps(sorted(selection_manifest,key=lambda x:json.dumps(x,sort_keys=True,separators=(",",":"))),ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()),
                  "sourceAvailability": "retrieved historical revision; first publication time unknown"}
    return out, provenance


def extract_gwangju_period(db: sqlite3.Connection, start: dt.date, end: dt.date
                           ) -> tuple[dict[str, dict[dt.date, dict[str, Any]]], dict[str, Any]]:
    """Extract validated Gwangju rows for the separate post-2022 experiment."""
    db.create_function("coverage_task_key", 3, coverage._request_task_key, deterministic=True)
    cte = _scoped_selection_cte(start, end, (MARKET,))
    scopes = list(db.execute(cte + """SELECT p.query_json,p.selected_state,p.latest_state,p.selected_job_id,
        p.selected_epoch_id,p.task_key,p.selected_row_count,p.selected_previous
        FROM picked p WHERE p.dataset='prices_origin'"""))
    scope_manifest=[]; states=defaultdict(int); job_ids=set()
    for query_text,state,latest_state,job_id,epoch,task,row_count,previous in scopes:
        try: query=json.loads(query_text)
        except (TypeError,json.JSONDecodeError): continue
        day=coverage.canonical_day(query.get("cond[trd_clcln_ymd::EQ]"))
        market=str(query.get("cond[whsl_mrkt_cd::EQ]",""))
        dimensions=coverage.price_dimensions(query,"prices_origin")
        product=dimensions[2] if dimensions else None
        if not day or market!=MARKET or product not in PRODUCTS: continue
        state_key=state or f"unavailable:{latest_state or 'unqueried'}"
        states[state_key]+=1
        if job_id is not None: job_ids.add(int(job_id))
        scope_manifest.append({"productId":product,"date":day,"marketCode":market,
            "subclass":query.get("cond[gds_sclsf_cd::EQ]") or None,"taskKey":task,
            "selectedState":state,"latestState":latest_state,"selectedJobId":job_id,
            "selectedEpochId":epoch,"selectedRowCount":row_count,"selectedPrevious":bool(previous)})
    configs={}
    for job_id in sorted(job_ids):
        row=db.execute("SELECT command,config_json,state FROM jobs WHERE job_id=?",(job_id,)).fetchone()
        if not row or row[0]!="prices" or row[2]!="complete":
            raise ValueError(f"selected Gwangju source job {job_id} is not complete")
        config=json.loads(row[1])
        if MARKET not in config.get("market",[]):
            raise ValueError(f"selected Gwangju source job {job_id} is outside market {MARKET}")
        configs[str(job_id)]=config
    for scope in scope_manifest:
        if scope["selectedJobId"] is None: continue
        config=configs.get(str(scope["selectedJobId"]))
        try: config_start=dt.date.fromisoformat(config["start"]);config_end=dt.date.fromisoformat(config["end"])
        except (TypeError,KeyError,ValueError) as exc:
            raise ValueError(f"selected Gwangju job {scope['selectedJobId']} has no valid date scope") from exc
        scope_day=dt.date.fromisoformat(scope["date"])
        if not config_start<=scope_day<=config_end:
            raise ValueError(f"selected Gwangju job {scope['selectedJobId']} does not cover {scope_day}")
    rows=db.execute(cte+"""SELECT s.request_id,s.response_sha256,p.selected_state,p.query_json,
        n.row_no,n.payload_json FROM picked p
        JOIN request_pages r ON r.job_id=p.selected_job_id AND r.dataset=p.dataset
          AND r.source_url=p.source_url AND r.task_key=p.task_key AND r.epoch_id=p.selected_epoch_id
          AND r.state='complete'
        JOIN source_requests s ON s.request_id=r.request_id
        LEFT JOIN normalized_records n ON n.request_id=r.request_id
        WHERE p.dataset='prices_origin' ORDER BY p.query_json,s.request_id,n.row_no""")
    sums=defaultdict(lambda:[Decimal(0),Decimal(0),0]);request_hashes={};request_ids=set()
    specific_scopes=set()
    for query_text,state,latest_state,*rest in scopes:
        try: query=json.loads(query_text)
        except (TypeError,json.JSONDecodeError): continue
        scope=_selected_specific_scope(query,state)
        if scope: specific_scopes.add(scope)
    for request_id,response_hash,state,query_text,row_no,payload_text in rows:
        request_ids.add(int(request_id));request_hashes[str(request_id)]=response_hash
        if state!="complete" or payload_text is None: continue
        try: query=json.loads(query_text); payload=json.loads(payload_text)
        except (TypeError,json.JSONDecodeError): continue
        date_text=coverage.canonical_day(query.get("cond[trd_clcln_ymd::EQ]"))
        product_info=coverage.price_dimensions(query,"prices_origin")
        product=product_info[2] if product_info else None
        subclass_query=str(query.get("cond[gds_sclsf_cd::EQ]",""))
        if not date_text or product not in PRODUCTS: continue
        valid,subclass=_valid_payload_scope(payload,query,product,MARKET)
        if (not valid or _overlap_shadowed(product,date_text,MARKET,subclass_query,subclass,specific_scopes)
                or payload.get("trd_se")!="경매" or payload.get("unit_nm")!="kg"):
            continue
        try: amount=Decimal(str(payload.get("totprc"))); weight=Decimal(str(payload.get("unit_tot_qty")))
        except (InvalidOperation,TypeError): continue
        if not amount.is_finite() or not weight.is_finite(): continue
        bucket=sums[(product,dt.date.fromisoformat(date_text),subclass_query)]
        bucket[0]+=amount;bucket[1]+=weight;bucket[2]+=1
    daily=defaultdict(lambda:[Decimal(0),Decimal(0),0])
    for (product,day,_subclass),(amount,weight,count) in sums.items():
        bucket=daily[(product,day)];bucket[0]+=amount;bucket[1]+=weight;bucket[2]+=count
    prices={p:{} for p in PRODUCTS}
    for (product,day),(amount,weight,count) in daily.items():
        if weight>0 and amount/weight>0:
            prices[product][day]={"price":float(amount/weight),"weight":float(weight),"rows":count}
    ordered_scopes=sorted(scope_manifest,key=lambda x:json.dumps(x,sort_keys=True,separators=(",",":")))
    provenance={"start":start.isoformat(),"end":end.isoformat(),"jobIds":sorted(job_ids),"jobConfigs":configs,
        "requestIds":sorted(request_ids),"responseHashes":request_hashes,"requestCount":len(request_ids),
        "selectedQueryStates":dict(sorted(states.items())),"selectedScopeCount":len(scope_manifest),
        "selectedScopesSha256":_sha(json.dumps(ordered_scopes,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()),
        "sourceAvailability":"retrieved historical revision; first publication time unknown"}
    return prices,provenance


def _ridge_fit(x_train, y_train, x_pred):
    model = modeling._new_estimator("ridge")
    model.fit(x_train, y_train)
    return float(model.predict([x_pred])[0])


def _holm_fixed(records: list[dict[str, Any]], test_name: str) -> None:
    ordered=sorted(records,key=lambda r:(r["tests"][test_name]["pValue"]
                                          if r["tests"][test_name]["pValue"] is not None else 1.0))
    running=0.0
    for i,r in enumerate(ordered):
        test=r["tests"][test_name]; pval=test["pValue"]
        running=max(running,min(1.0,(FAMILY_SIZE-i)*(pval if pval is not None else 1.0)))
        test["holmAdjustedPValue"]=running
        test["holmSignificantAt005"]=test["status"]=="available" and running<.05


def _pair_conclusion(pair: list[dict[str, Any]]) -> dict[str, Any]:
    unavailable=not pair or any(r["n"]<30 or r["tests"]["hac9"]["status"]!="available"
                                 or r["tests"]["hac28"]["status"]!="available" for r in pair)
    worse=[r["loss"] for r in pair if r["tests"]["hac9"].get("holmSignificantAt005") and r["meanLossDifference"]>0]
    better=[r["loss"] for r in pair if r["tests"]["hac9"].get("holmSignificantAt005") and r["meanLossDifference"]<0]
    if unavailable: status="inference_unavailable"
    elif len(better)==2: status="improved"
    elif len(worse)==2: status="worsened"
    elif worse or better: status="mixed_evidence"
    else: status="no_evidence"
    return {"status":status,"significantlyWorsenedLosses":worse,
            "significantlyImprovedLosses":better,"requiresBothLossesForImprovement":True}


def _market_differences(market_series: dict[str, dict[dt.date, float]], origin: dt.date,
                        gwangju_price: float, gwangju_date: dt.date) -> list[float]:
    values = []
    for market in dashboard.MARKETS:
        prior = [day for day in market_series.get(market, {}) if day <= origin]
        day = max(prior) if prior else None
        price = market_series.get(market, {}).get(day) if day else None
        if (price is None or (origin-day).days > MAX_AGE or (origin-gwangju_date).days > MAX_AGE
                or not math.isfinite(float(price)) or float(price) <= 0):
            values.extend([0.0, 0.0])
        else:
            values.extend([math.log(float(price)) - math.log(gwangju_price), 1.0])
    return values


def _extract_core_market_prices(db: sqlite3.Connection, start_day: dt.date = TRAIN_START,
                                end_day: dt.date = EVAL_END, include_diagnostics: bool = False):
    """Selected signed daily auction/kg by market, using dashboard query selection."""
    db.create_function("coverage_task_key", 3, coverage._request_task_key, deterministic=True)
    start, end = start_day.isoformat(), end_day.isoformat()
    prices = {p: {m: {} for m in dashboard.MARKETS} for p in PRODUCTS}
    sums = defaultdict(lambda: [Decimal(0), Decimal(0), 0])
    states = defaultdict(int)
    request_ids = set()
    cte=_scoped_selection_cte(start_day,end_day,tuple(dashboard.MARKETS))
    scope_rows=list(db.execute(cte+"""SELECT p.latest_job_id,p.selected_job_id,p.selected_epoch_id,p.task_key,p.query_json,p.selected_state,p.latest_state
        FROM picked p WHERE p.dataset='prices_origin'"""))
    specific_scopes=set()
    selection_manifest=[]
    states=defaultdict(int)
    for latest_job,selected_job,epoch,task,query_text,state,latest_state in scope_rows:
        try: query=json.loads(query_text)
        except (TypeError,json.JSONDecodeError): continue
        dims=coverage.price_dimensions(query,"prices_origin")
        if not dims: continue
        sub=str(query.get("cond[gds_sclsf_cd::EQ]","")); day=coverage.canonical_day(query.get("cond[trd_clcln_ymd::EQ]"))
        market=str(query.get("cond[whsl_mrkt_cd::EQ]",""))
        selection_manifest.append({"jobId":selected_job,"epochId":epoch,"taskKey":task,"product":dims[2],"date":day,
                                   "market":market,"subclass":sub or None,"selectedState":state,
                                   "latestJobId":latest_job,"latestState":latest_state})
        states[str(state or f"unavailable:{latest_state or 'unqueried'}")]+=1
        scope=_selected_specific_scope(query,state)
        if scope:specific_scopes.add(scope)
    price_sql=cte+dashboard.PRICE_SQL[len(coverage.SELECTION_CTE):]
    for rec in db.execute(price_sql, (start, end, *dashboard.MARKETS.keys())):
        try: query = json.loads(rec["query_json"])
        except (TypeError, json.JSONDecodeError): continue
        product = dashboard._query_product(query)
        day = dashboard._parse_date(query.get("cond[trd_clcln_ymd::EQ]"))
        market = str(query.get("cond[whsl_mrkt_cd::EQ]", ""))
        if product not in PRODUCTS or day is None or market not in dashboard.MARKETS: continue
        if rec["selected_state"] not in ("complete", "empty"): continue
        if rec["request_id"] is not None:
            request_ids.add(int(rec["request_id"]))
        if not rec["payload_json"]: continue
        try: row = json.loads(rec["payload_json"])
        except json.JSONDecodeError: continue
        valid,subclass=_valid_payload_scope(row,query,product,market)
        query_sub=str(query.get("cond[gds_sclsf_cd::EQ]",""))
        if not valid or _overlap_shadowed(product,day.isoformat(),market,query_sub,subclass,specific_scopes): continue
        if row.get("trd_se") != "경매" or row.get("unit_nm") != "kg": continue
        amount, volume = dashboard._as_decimal(row.get("totprc")), dashboard._as_decimal(row.get("unit_tot_qty"))
        if amount is None or volume is None: continue
        sums[(product, market, day,query_sub)][0] += amount; sums[(product, market, day,query_sub)][1] += volume
        sums[(product, market, day,query_sub)][2] += 1
    daily=defaultdict(lambda:[Decimal(0),Decimal(0)])
    for (product,market,day,_scope),(amount,volume,_count) in sums.items():
        daily[(product,market,day)][0]+=amount;daily[(product,market,day)][1]+=volume
    diagnostics={}
    for (product,market,day,_scope),(amount,volume,count) in sums.items():
        key=(product,day,market)
        info=diagnostics.setdefault(key,{"eligibleAuctionKgRows":0,"signedNetAmount":Decimal(0),"signedNetWeight":Decimal(0)})
        info["eligibleAuctionKgRows"]+=count;info["signedNetAmount"]+=amount;info["signedNetWeight"]+=volume
    for (product, market, day), (amount, volume) in daily.items():
        if volume > 0 and amount / volume > 0:
            prices[product][market][day] = float(amount / volume)
    hashes={str(r[0]):r[1] for rid in sorted(request_ids)
            for r in db.execute("SELECT request_id,response_sha256 FROM source_requests WHERE request_id=?",(rid,))
            if r[1]}
    provenance={"selectedRequestHashes": hashes, "selectedRequestCount": len(hashes), "selectedQueryStates": dict(sorted(states.items())),
                    "selectedScopeCount":len(selection_manifest),
                    "selectedScopesSha256":_sha(json.dumps(sorted(selection_manifest,key=lambda x:json.dumps(x,sort_keys=True,separators=(",",":"))),ensure_ascii=False,sort_keys=True,separators=(",",":")).encode())}
    if include_diagnostics:
        return prices, provenance, diagnostics
    return prices, provenance


def _g_features(obs: dict[dt.date, dict[str, Any]], origin: dt.date):
    prior = [d for d in obs if d <= origin]
    if not prior: return None
    d = max(prior); age = (origin-d).days
    if age > MAX_AGE: return None
    market = obs[d]
    return d, [math.log(market["price"]), math.log(market["weight"]), float(age)]


def run_analysis(core_path: Path, db_path: Path) -> dict[str, Any]:
    core_bytes = core_path.read_bytes()
    raw, core = _load_core_bytes(core_bytes)
    db_uri = "file:" + str(db_path.resolve()) + "?mode=ro"
    db = sqlite3.connect(db_uri, uri=True); db.row_factory = sqlite3.Row; db.execute("PRAGMA query_only=ON")
    try:
        db.execute("BEGIN")
        gw, gwprov = extract_gwangju(db)
        core_market_prices, coreprov = _extract_core_market_prices(db)
        db.rollback()
    finally: db.close()
    predictions: dict[str, list[dict[str, Any]]] = {p: [] for p in PRODUCTS}
    counts = {p: defaultdict(int) for p in PRODUCTS}
    for p in PRODUCTS:
        actuals, market = core[p], gw[p]
        source_product = next(row for row in raw["products"] if row["id"] == p)
        series = modeling._observed_series(source_product)
        feature_cache: dict[Any, Any] = {}
        prepared = {}
        for target in sorted(d for d in series if TRAIN_START <= d <= EVAL_END):
            origin = target - dt.timedelta(days=1)
            core_feature = modeling._feature_row(series, origin, target, 1, feature_cache)
            gz_feature = _g_features(market, origin)
            if core_feature is None or gz_feature is None:
                continue
            source_date, control_x = core_feature
            gdate, gcell = gz_feature
            candidate_x = tuple(control_x) + tuple(gcell) + tuple(
                _market_differences(core_market_prices[p], origin, market[gdate]["price"], gdate))
            prepared[target] = {"control": tuple(control_x), "candidate": candidate_x,
                                "sourceDate": source_date, "gwangjuSourceDate": gdate}
        for target in (EVAL_START + dt.timedelta(days=i) for i in range((EVAL_END-EVAL_START).days+1)):
            origin = target-dt.timedelta(days=1)
            actual = actuals.get(target)
            if actual is None: counts[p]["no_target_actual"] += 1; continue
            item = prepared.get(target)
            if item is None:
                if modeling._feature_row(series, origin, target, 1, feature_cache) is None:
                    counts[p]["stale_or_missing_core_source"] += 1
                else: counts[p]["gwangju_missing_or_stale"] += 1
                continue
            # Common eligible targets define every model's evaluation sample.
            # Expanding walk-forward fit: initial 2017–18 window plus every
            # previously observed, eligible evaluation target strictly before origin.
            train_targets = [t for t in prepared if TRAIN_START <= t < origin]
            if len(train_targets) < MIN_TRAIN: counts[p]["insufficient_train"] += 1; continue
            train_xc = [prepared[t]["control"] for t in train_targets]
            train_xg = [prepared[t]["candidate"] for t in train_targets]
            # Existing Ridge target is log(target/source), inverted with the same as-of source.
            train_y = [math.log(series[t][0] / series[prepared[t]["sourceDate"]][0]) for t in train_targets]
            # Add latest price differences to matching target-product series when available.
            source = item["sourceDate"]
            control_pred = modeling._inverse_log_ratio(series[source][0], _ridge_fit(train_xc, train_y, item["control"]))[0]
            candidate_pred = modeling._inverse_log_ratio(series[source][0], _ridge_fit(train_xg, train_y, item["candidate"]))[0]
            predictions[p].append({"target": target.isoformat(), "origin": origin.isoformat(), "actual": actual["price"],
                "candidate": candidate_pred, "control": control_pred, "persistence": series[source][0], "sourceDate": source.isoformat(),
                "gwangjuSourceDate": item["gwangjuSourceDate"].isoformat(), "trainingN": len(train_y)})
            counts[p]["common_targets"] += 1
    records = []
    for p in PRODUCTS:
        rows = predictions[p]
        for cand, base in PAIRS:
            for loss in LOSSES:
                vals = [(abs(r[cand]-r["actual"]), abs(r[base]-r["actual"])) for r in rows]
                if loss == "MAPE": vals = [(a/r["actual"]*100,b/r["actual"]*100) for (a,b),r in zip(vals,rows)]
                n=len(vals); cm=sum(x[0] for x in vals)/n if n else None; bm=sum(x[1] for x in vals)/n if n else None
                candidate_dm=[{"target":r["target"],"horizon":1,"predicted":r[cand]} for r in rows]
                baseline_dm=[{"target":r["target"],"horizon":1,"predicted":r[base]} for r in rows]
                actual_dm={(r["target"],1):r["actual"] for r in rows}
                tests={}
                for lag,label in ((9,"hac9"),(28,"hac28")):
                    d=(experiments._dm(candidate_dm,baseline_dm,actual_dm,1,"absolute_percentage_error" if loss=="MAPE" else "absolute_error",max_lag=lag)
                       if n >= 30 else {"status":"unavailable_insufficient_pairs"})
                    tests[label]={"n":n,"status":d.get("status"),"statistic":d.get("statistic"),"pValue":d.get("pValue"),"calendarHacMaxLagDays":lag}
                records.append({"productId":p,"candidate":cand,"comparator":base,"loss":loss,"n":n,
                    "candidateMean":cm,"comparatorMean":bm,"meanLossDifference":cm-bm if n else None,"tests":tests})
    for test_name in ("hac9","hac28"):
        _holm_fixed(records,test_name)
    by_product=[]
    for p in PRODUCTS:
        rows=predictions[p]
        means={name:{"MAE":None,"MAPE":None} for name in ("candidate","control","persistence")}
        for name in means:
            if rows:
                means[name]={"MAE":sum(abs(r[name]-r["actual"]) for r in rows)/len(rows),
                    "MAPE":sum(abs(r[name]-r["actual"])/r["actual"]*100 for r in rows)/len(rows)}
        comps=[r for r in records if r["productId"]==p]
        status={}
        for base in ("control","persistence"):
            pair=[r for r in comps if r["comparator"]==base]
            status[base]=_pair_conclusion(pair)
        pair_status={r["status"] for r in status.values()}
        if pair_status == {"improved"}: overall="improved"
        elif pair_status == {"worsened"}: overall="worsened"
        elif "improved" in pair_status or "worsened" in pair_status or "mixed_evidence" in pair_status:
            overall="mixed_evidence"
        elif "inference_unavailable" in pair_status: overall="inference_unavailable"
        else: overall="no_evidence"
        status["overall"]=overall
        by_product.append({"id":p,"name":dashboard.PRODUCTS[p]["name"],"coverage":{"commonTargets":len(rows),"eligibleTargetDays":(EVAL_END-EVAL_START).days+1,"excludedByReason":dict(counts[p]),"evaluationStart":EVAL_START.isoformat(),"evaluationEnd":EVAL_END.isoformat()},"metrics":means,"conclusions":status,"predictions":rows})
    response_hashes={**gwprov["responseHashes"], **coreprov["selectedRequestHashes"]}
    digest_inputs={"responseHashes":response_hashes,"gwangjuScopes":gwprov["selectedScopesSha256"],
                   "coreMarketScopes":coreprov["selectedScopesSha256"]}
    inputs_hash=_sha(core_bytes+json.dumps(digest_inputs,sort_keys=True,separators=(",",":")).encode())
    return {"schemaVersion":1,"status":"complete","protocol":PROTOCOL,"protocolSha256":_sha(json.dumps(PROTOCOL,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()),"inputSha256":inputs_hash,"inputs":{"corePrices":{"path":str(core_path),"sha256":_sha(core_bytes)},"databasePath":str(db_path),"gwangju":gwprov,"coreMarkets":coreprov},"products":by_product,"comparisons":records}


def _subperiod_metrics(rows):
    groups=defaultdict(list)
    for row in rows:
        day=dt.date.fromisoformat(row["target"])
        season="winter" if day.month in (12,1,2) else "spring" if day.month in (3,4,5) else "summer" if day.month in (6,7,8) else "autumn"
        groups[("year",str(day.year))].append(row)
        groups[("season",f"{day.year}-{season}")].append(row)
    result=[]
    for (kind,label),items in sorted(groups.items()):
        metrics={}
        for name in ("candidate","control","persistence"):
            metrics[name]={"MAE":sum(abs(x[name]-x["actual"]) for x in items)/len(items),
                "MAPE":sum(abs(x[name]-x["actual"])/x["actual"]*100 for x in items)/len(items)}
        result.append({"periodKind":kind,"period":label,"n":len(items),"metrics":metrics})
    return result


def _revalidation_pair_conclusion(pair):
    unavailable=not pair or any(r["n"]<90 or r["tests"]["hac9"]["status"]!="available" for r in pair)
    worse=[r["loss"] for r in pair if r["tests"]["hac9"].get("holmSignificantAt005") and r["meanLossDifference"]>0]
    better=[r["loss"] for r in pair if r["tests"]["hac9"].get("holmSignificantAt005") and r["meanLossDifference"]<0]
    if unavailable: status="inference_unavailable"
    elif len(better)==2: status="improved"
    elif len(worse)==2: status="worsened"
    elif worse or better: status="mixed_evidence"
    else: status="no_evidence"
    return {"status":status,"significantlyWorsenedLosses":worse,
        "significantlyImprovedLosses":better,"requiresBothLossesForImprovement":True,
        "primaryBandwidth":"HAC9","sensitivityBandwidth":"HAC28"}


def _common_revalidation_cutoff(states, input_context_end=None):
    cutoff, detail=_verified_common_cutoff(states)
    if input_context_end is not None:
        try: context=dt.date.fromisoformat(input_context_end)
        except (TypeError,ValueError) as exc: raise ValueError("input prices meta.contextEnd must be an ISO date") from exc
        if cutoff>context:
            raise ValueError(f"validated database cutoff {cutoff} is later than input prices contextEnd {context}; refresh the input snapshot")
    return cutoff,detail


def run_revalidation_analysis(core_path: Path, db_path: Path,
                             manifest_path: Path | None = None) -> dict[str, Any]:
    """Separate 2023+ revalidation; legacy 2019-20 result remains untouched."""
    core_bytes=core_path.read_bytes();raw,core=_load_core_bytes(core_bytes)
    if not isinstance(raw.get("meta"),dict): raise ValueError("revalidation requires prices.json meta provenance")
    db_uri="file:"+str(db_path.resolve())+"?mode=ro"
    db=sqlite3.connect(db_uri,uri=True);db.row_factory=sqlite3.Row;db.execute("PRAGMA query_only=ON")
    try:
        db.execute("BEGIN")
        broad_states,broad_prov=_core_broad_query_states(db)
        cutoff,cutoff_prov=_common_revalidation_cutoff(broad_states,raw["meta"].get("contextEnd"))
        gw,gwprov=extract_gwangju_period(db,REVALIDATION_TRAIN_START,cutoff)
        market_prices,market_prov,market_diag=_extract_core_market_prices(db,REVALIDATION_TRAIN_START,cutoff,True)
        snapshot_check=_validate_revalidation_input(raw,core,broad_states,market_diag,cutoff)
        db.rollback()
    finally: db.close()
    response_hashes={**gwprov["responseHashes"],**market_prov["selectedRequestHashes"]}
    input_hash=_sha(core_bytes+json.dumps({"responseHashes":response_hashes,
        "gwangjuScopes":gwprov["selectedScopesSha256"],"coreMarketScopes":market_prov["selectedScopesSha256"],
        "broadScopes":broad_prov["scopesSha256"]},sort_keys=True,separators=(",",":")).encode())
    protocol={**REVALIDATION_PROTOCOL,"evaluation":[REVALIDATION_EVAL_START.isoformat(),cutoff.isoformat()],
        "cutoffRuleDetails":cutoff_prov}
    protocol_hash=_sha(json.dumps(protocol,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode())
    frozen_manifest={"schemaVersion":1,"frozenBeforeModelFit":True,"protocol":protocol,
        "protocolSha256":protocol_hash,"inputSha256":input_hash,
        "inputs":{"corePrices":{"path":str(core_path.resolve()),"sha256":_sha(core_bytes),
            "contextStart":snapshot_check["contextStart"],"contextEnd":snapshot_check["contextEnd"],
            "snapshotValidation":snapshot_check},"databasePath":str(db_path.resolve()),
            "cutoff":cutoff_prov,"broadQueries":broad_prov,"gwangju":gwprov,"coreMarkets":market_prov}}
    frozen_manifest_text=json.dumps(frozen_manifest,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    frozen_manifest_hash=_sha(frozen_manifest_text.encode("utf-8"))
    if manifest_path is not None:
        _atomic(Path(manifest_path),frozen_manifest_text)
    predictions={p:[] for p in PRODUCTS};counts={p:defaultdict(int) for p in PRODUCTS}
    period_exclusions={p:defaultdict(lambda:defaultdict(int)) for p in PRODUCTS}
    initial_training_dates_by_product={}
    for product in PRODUCTS:
        source_product=next(x for x in raw["products"] if x["id"]==product)
        series=modeling._observed_series(source_product);feature_cache={};prepared={}
        for target in sorted(d for d in series if REVALIDATION_TRAIN_START<=d<=cutoff):
            origin=target-dt.timedelta(days=1)
            core_row=modeling._feature_row(series,origin,target,1,feature_cache)
            gw_row=_g_features(gw[product],origin)
            if core_row is None or gw_row is None: continue
            source_date,control_x=core_row;gw_date,gw_features=gw_row
            candidate_x=tuple(control_x)+tuple(gw_features)+tuple(_market_differences(
                market_prices[product],origin,gw[product][gw_date]["price"],gw_date))
            prepared[target]={"control":tuple(control_x),"candidate":candidate_x,
                "sourceDate":source_date,"gwangjuSourceDate":gw_date}
        initial_training_dates_by_product[product]=sum(
            REVALIDATION_TRAIN_START<=day<=dt.date(2022,12,31) for day in prepared)
        eval_days=(REVALIDATION_EVAL_START+dt.timedelta(days=i)
            for i in range((cutoff-REVALIDATION_EVAL_START).days+1))
        for target in eval_days:
            period=f"{target.year}-{_season_name(target)}"
            def exclude(reason):
                counts[product][reason]+=1
                period_exclusions[product][period][reason]+=1
            actual=series.get(target)
            if actual is None:
                exclude(_unobserved_target_reason(product,target,broad_states,market_diag))
                continue
            item=prepared.get(target)
            if item is None:
                if modeling._feature_row(series,target-dt.timedelta(days=1),target,1,feature_cache) is None:
                    exclude("core_source_missing_or_stale")
                else: exclude("gwangju_source_missing_or_stale")
                continue
            origin=target-dt.timedelta(days=1)
            train_targets=[t for t in prepared if REVALIDATION_TRAIN_START<=t<origin]
            if len(train_targets)<MIN_TRAIN:
                exclude("insufficient_training_targets");continue
            train_xc=[prepared[t]["control"] for t in train_targets]
            train_xg=[prepared[t]["candidate"] for t in train_targets]
            train_y=[math.log(series[t][0]/series[prepared[t]["sourceDate"]][0]) for t in train_targets]
            source=item["sourceDate"]
            control_pred=modeling._inverse_log_ratio(series[source][0],_ridge_fit(train_xc,train_y,item["control"]))[0]
            candidate_pred=modeling._inverse_log_ratio(series[source][0],_ridge_fit(train_xg,train_y,item["candidate"]))[0]
            predictions[product].append({"target":target.isoformat(),"origin":origin.isoformat(),"actual":actual[0],
                "candidate":candidate_pred,"control":control_pred,"persistence":series[source][0],
                "sourceDate":source.isoformat(),"gwangjuSourceDate":item["gwangjuSourceDate"].isoformat(),
                "trainingN":len(train_targets)})
            counts[product]["common_targets"]+=1
            period_exclusions[product][period]["common_targets"]+=1
    comparisons=[];product_rows=[]
    for product in PRODUCTS:
        rows=predictions[product];gate=_coverage_gate([dt.date.fromisoformat(r["target"]) for r in rows])
        metrics={name:{"MAE":None,"MAPE":None} for name in ("candidate","control","persistence")}
        if rows:
            for name in metrics:
                metrics[name]={"MAE":sum(abs(r[name]-r["actual"]) for r in rows)/len(rows),
                    "MAPE":sum(abs(r[name]-r["actual"])/r["actual"]*100 for r in rows)/len(rows)}
        for candidate,base in PAIRS:
            for loss in LOSSES:
                candidate_values=[];base_values=[]
                for row in rows:
                    scale=100/row["actual"] if loss=="MAPE" else 1
                    candidate_values.append(abs(row[candidate]-row["actual"])*scale)
                    base_values.append(abs(row[base]-row["actual"])*scale)
                n=len(rows);cm=sum(candidate_values)/n if n else None;bm=sum(base_values)/n if n else None
                tests={}
                for lag,label in ((9,"hac9"),(28,"hac28")):
                    if not gate["passed"]:
                        result={"status":"unavailable_sample_gate","statistic":None,"pValue":None}
                    elif n<30:
                        result={"status":"unavailable_insufficient_pairs","statistic":None,"pValue":None}
                    elif _constant_difference([a-b for a,b in zip(candidate_values,base_values)]):
                        result={"status":"unavailable_nonpositive_variance","statistic":None,"pValue":None}
                    else:
                        cand=[{"target":r["target"],"horizon":1,"predicted":r[candidate]} for r in rows]
                        cmp=[{"target":r["target"],"horizon":1,"predicted":r[base]} for r in rows]
                        actuals={(r["target"],1):r["actual"] for r in rows}
                        dm=experiments._dm(cand,cmp,actuals,1,"absolute_percentage_error" if loss=="MAPE" else "absolute_error",max_lag=lag)
                        result={"status":dm.get("status"),"statistic":dm.get("statistic"),"pValue":dm.get("pValue")}
                    tests[label]={**result,"n":n,"calendarHacMaxLagDays":lag}
                comparisons.append({"productId":product,"candidate":candidate,"comparator":base,"loss":loss,"n":n,
                    "candidateMean":cm,"comparatorMean":bm,"meanLossDifference":cm-bm if n else None,"tests":tests})
        product_rows.append({"id":product,"name":dashboard.PRODUCTS[product]["name"],
            "coverage":{"commonTargets":len(rows),"eligibleTargetDays":(cutoff-REVALIDATION_EVAL_START).days+1,
                "excludedByReason":dict(counts[product]),"evaluationStart":REVALIDATION_EVAL_START.isoformat(),
                "evaluationEnd":cutoff.isoformat(),"inferenceGate":gate,
                "initialTrainingDatesWithBothSources":initial_training_dates_by_product[product],
                "exclusionsByYearSeason":{period:dict(sorted(reasons.items()))
                    for period,reasons in sorted(period_exclusions[product].items())}},"metrics":metrics,
            "subperiodMetrics":_subperiod_metrics(rows),"predictions":rows})
    for name in ("hac9","hac28"):_holm_fixed(comparisons,name)
    for product_row in product_rows:
        product=product_row["id"]
        product_comps=[x for x in comparisons if x["productId"]==product]
        conclusions={base:_revalidation_pair_conclusion([x for x in product_comps if x["comparator"]==base])
            for base in ("control","persistence")}
        if not product_row["coverage"]["inferenceGate"]["passed"]:
            for conclusion in conclusions.values(): conclusion["status"]="inference_unavailable_sample_gate"
        statuses={x["status"] for x in conclusions.values()}
        conclusions["overall"]="improved" if statuses=={"improved"} else "worsened" if statuses=={"worsened"} else (
            "mixed_evidence" if statuses & {"improved","worsened","mixed_evidence"} else
            "inference_unavailable" if "inference_unavailable" in statuses or "inference_unavailable_sample_gate" in statuses else "no_evidence")
        product_row["conclusions"]=conclusions
    return {"schemaVersion":1,"status":"complete","protocol":protocol,"protocolSha256":protocol_hash,
        "inputSha256":input_hash,"inputs":frozen_manifest["inputs"],"analysisManifest":{"path":str(Path(manifest_path).resolve()) if manifest_path else None,
            "sha256":frozen_manifest_hash,
            "persistedBeforeModelFit":manifest_path is not None},
        "products":product_rows,"comparisons":comparisons}


def write_report(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic(output_dir/"report.json",json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    def fmt(value): return "unavailable" if value is None else f"{value:.4f}"
    def test_text(test):
        p=fmt(test.get("pValue")); adjusted=fmt(test.get("holmAdjustedPValue"))
        if test.get("status")!="available": return f"{test.get('status','unavailable')} (p=1 in Holm family)"
        return f"stat={fmt(test.get('statistic'))}, p={p}, Holm={adjusted}, sig={str(bool(test.get('holmSignificantAt005'))).lower()}"
    def loss_decision(row):
        d=row.get("meanLossDifference")
        return "unavailable" if d is None else ("candidate lower" if d<0 else "candidate higher" if d>0 else "tie")
    evaluation=report["protocol"]["evaluation"]
    lines=["# Gwangju Seobu market information experiment","",f"Status: {report['status']}",
           f"Evaluation targets: {evaluation[0]} through {evaluation[1]} (one calendar day ahead).",
           "The selected daily target uses the existing signed Garak/Gangseo auction-per-kg definition. Prices are retrospective retrieved revisions; first publication time is unknown. This is not a real-time availability claim.",
           "", "## Coverage and descriptive metrics", "",
           "| Product | Common target dates | Excluded dates by reason | Candidate MAE / MAPE | Control MAE / MAPE | Persistence MAE / MAPE |",
           "|---|---:|---|---|---|---|"]
    for product in report["products"]:
        coverage_row=product["coverage"]; metrics=product["metrics"]
        excluded={key:value for key,value in coverage_row.get("excludedByReason",{}).items() if key!="common_targets"}
        exclusions=json.dumps(excluded,ensure_ascii=False,sort_keys=True,separators=(",",":"))
        cells=[]
        for model in ("candidate","control","persistence"):
            m=metrics[model]; cells.append(f"{fmt(m.get('MAE'))} / {fmt(m.get('MAPE'))}%")
        lines.append(f"| {product['name']} | {coverage_row['commonTargets']} / {coverage_row['eligibleTargetDays']} | `{exclusions}` | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.extend(["","## Paired loss tests","",
        "Each comparison uses the same common target dates. Mean loss difference is candidate minus comparator, so a negative value favors the candidate. Holm correction is calculated separately over the fixed 20 comparisons for HAC9 and HAC28; unavailable tests remain in each family with p=1.","",
        "| Product | Comparator | Loss | n | Candidate mean | Comparator mean | Difference | HAC9 primary (Holm) | HAC28 sensitivity (Holm) | Direction |",
        "|---|---|---|---:|---:|---:|---:|---|---|---|"])
    for row in report.get("comparisons",[]):
        product=next((p["name"] for p in report["products"] if p["id"]==row["productId"]),row["productId"])
        lines.append(f"| {product} | {row['comparator']} | {row['loss']} | {row['n']} | {fmt(row.get('candidateMean'))} | {fmt(row.get('comparatorMean'))} | {fmt(row.get('meanLossDifference'))} | {test_text(row['tests']['hac9'])} | {test_text(row['tests']['hac28'])} | {loss_decision(row)} |")
    lines.extend(["","## Conclusions","","Improvement requires lower MAE and MAPE with significant primary tests after the fixed-family Holm correction. Non-significance is not evidence of equivalence.",""])
    for product in report["products"]:
        conclusions=product.get("conclusions",{})
        lines.append(f"### {product['name']}: overall `{conclusions.get('overall','unavailable')}`")
        for comparator in ("control","persistence"):
            conclusion=conclusions.get(comparator,{})
            if isinstance(conclusion,dict):
                lines.append(f"- Candidate vs {comparator}: **{conclusion.get('status','unavailable')}**; significantly worsened losses: {', '.join(conclusion.get('significantlyWorsenedLosses',[])) or 'none'}; significantly improved losses: {', '.join(conclusion.get('significantlyImprovedLosses',[])) or 'none'}.")
            else: lines.append(f"- Candidate vs {comparator}: **{conclusion}**.")
    if report.get("protocol",{}).get("id")==REVALIDATION_PROTOCOL["id"]:
        initial_counts="; ".join(f"{p['name']}: {p['coverage'].get('initialTrainingDatesWithBothSources',0)}" for p in report["products"])
        lines.extend(["","## Year and season coverage","",
            f"Prepared 2017-2022 training target dates with both core and Gwangju sources: {initial_counts}. These exploratory grouped metrics expose uneven dates and possible local worsening; the 90-date, 2-year, 3-season gate is shown per product in JSON. Verified empty responses are not labeled market closures.",
            "","| Product | Period | n | Candidate MAE / MAPE | Control MAE / MAPE | Persistence MAE / MAPE |",
            "|---|---|---:|---|---|---|"])
        for product in report["products"]:
            for item in product.get("subperiodMetrics",[]):
                m=item["metrics"]
                lines.append(f"| {product['name']} | {item['period']} | {item['n']} | {fmt(m['candidate']['MAE'])} / {fmt(m['candidate']['MAPE'])}% | {fmt(m['control']['MAE'])} / {fmt(m['control']['MAPE'])}% | {fmt(m['persistence']['MAE'])} / {fmt(m['persistence']['MAPE'])}% |")
        cutoff=report.get("inputs",{}).get("cutoff",{})
        snapshot=report.get("inputs",{}).get("corePrices",{}).get("snapshotValidation",{})
        lines.extend(["",f"Cutoff: {cutoff.get('coverageStart','?')} through {cutoff.get('coverageEnd','?')}; common scopes per date: {cutoff.get('requiredBroadScopesPerDate','?')}; input context end: {snapshot.get('contextEnd','?')}.",
            "The cutoff is derived from the contiguous verified complete/empty broad-scope coverage and is never selected based on model scores. The new period is not pooled with the earlier 2019-2020 report.",""])
    lines.extend(["","## Protocol and limits","",
        f"Protocol SHA-256: `{report.get('protocolSha256','unavailable')}`. Input SHA-256: `{report.get('inputSha256','unavailable')}`.",
        "This retrospective comparison uses the current validated source revisions and cannot establish what was available at the original forecast time. It is not a new untouched holdout. The report does not establish equivalence or justify replacing the operating dashboard model.",""])
    _atomic(output_dir/"report.md","\n".join(lines))


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",default="."); parser.add_argument("--prices"); parser.add_argument("--db",default="data/agri.sqlite")
    parser.add_argument("--period",choices=("legacy","revalidation-2023"),default="legacy")
    parser.add_argument("--output-dir")
    args=parser.parse_args(argv); root=Path(args.root).resolve(); core=Path(args.prices) if args.prices else latest_prices_file(root)
    db_path=Path(args.db) if Path(args.db).is_absolute() else root/args.db
    if args.period=="legacy":
        if args.output_dir is None: parser.error("--output-dir is required for the legacy period")
        report=run_analysis(core,db_path); output_dir=Path(args.output_dir)
    else:
        output_dir=Path(args.output_dir) if args.output_dir else root/"outputs/market-extension-revalidation-2023"
        if output_dir.resolve()==(root/"outputs/market-extension-20261005").resolve():
            parser.error("the 2023+ revalidation must use a separate output directory; the legacy report is immutable")
        report=run_revalidation_analysis(core,db_path,output_dir/"analysis-manifest.json")
    write_report(report,output_dir); return 0


if __name__ == "__main__": raise SystemExit(main())
