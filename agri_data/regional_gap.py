"""Offline matched-cell price-gap analysis for Gwangju and Seoul markets."""
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

from . import coverage, dashboard, market_extension

START, END = dt.date(2017, 1, 1), dt.date(2020, 12, 31)
GWANGJU, GARAK, GANGSEO = "240004", "110001", "110008"
MARKETS = (GWANGJU, GARAK, GANGSEO)
DIMENSIONS = (("gds_sclsf_cd", "gds_sclsf_nm"), ("grd_cd", "grd_nm"),
              ("sz_cd", "sz_nm"), ("pkg_cd", "pkg_nm"))
PRODUCTS = tuple(dashboard.PRODUCTS)
GARLIC_CLASSES = {"06", "16", "41", "42", "43"}
PROTOCOL = {"id": "regional-market-gap-2026-10-05", "period": [START.isoformat(), END.isoformat()],
    "markets": {"gwangju": GWANGJU, "garak": GARAK, "gangseo": GANGSEO},
    "pairing": "exact nonempty subclass, grade, size and package codes; all market names present and non-placeholder, with no disagreement",
    "primary": {"practicalMeanBoundary": "log(1.05)", "smallGapBoundary": "log(1.10)", "smallGapShare": .8,
        "HACCalendarDays": [28, 9], "bootstrapMovingBlockDays": 28, "bootstrapReplicates": 9999,
        "minimumDates": 90, "minimumYears": 2, "minimumSeasons": 3, "minimumDatesPerSeason": 10,
        "holmFamily": "5 products x 3 tests", "alpha": .05},
    "price": "signed sum(totprc)/signed sum(unit_tot_qty), auction/kg only",
    "interpretation": "observed differences across these wholesale markets; not city-wide or causal"}


def _json(value):
    try: return json.loads(value) if value else {}
    except (TypeError, json.JSONDecodeError): return {}


def _sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _cell_key(payload: dict[str, Any]):
    vals = tuple(str(payload.get(code, "")).strip() if payload.get(code) is not None else "" for code, _ in DIMENSIONS)
    if any(not x for x in vals): return None
    return vals


def _code_status(payload):
    return _dimension_code_status([payload.get(code) for code,_ in DIMENSIONS])


def _dimension_code_status(vals):
    if any(v is None or not str(v).strip() for v in vals): return "missing_dimension_code"
    if any(str(v).strip() in {".", "-", "--", "?", "미상", "기타"} for v in vals): return "placeholder_dimension_code"
    return None


def _names_for(payload):
    return [{str(payload.get(name)).strip() if payload.get(name) is not None else ""} for _, name in DIMENSIONS]


def _merge_names(target, incoming):
    if not target: return [set(x) for x in incoming]
    for i, names in enumerate(incoming): target[i].update(names)
    return target


def _combine_market_names(left, right):
    if not left or not right: return []
    return [set(left[i]) | set(right[i]) for i in range(len(DIMENSIONS))]


def _name_status(payloads, policy="strict"):
    """Return whether all four dimensions have stable, meaningful labels."""
    if not payloads: return "missing_dimension_name"
    if isinstance(payloads[0], dict):
        names = [set() for _ in DIMENSIONS]
        for p in payloads:
            for i, (_, name_key) in enumerate(DIMENSIONS):
                raw=p.get(name_key); names[i].add(str(raw).strip() if raw is not None else "")
    else:
        names = payloads
    placeholders={".", "-", "--", "?", "미상", "기타"}
    if policy=="strict":
        for values in names:
            if "" in values: return "missing_dimension_name"
            if any(x in placeholders for x in values): return "placeholder_dimension_name"
            if len(values)!=1:return "dimension_name_conflict"
        return "compatible"
    if policy!="recorded_code":raise ValueError(f"unknown name policy: {policy}")
    unknown=False
    for values in names:
        # The exploratory protocol permits only absent labels and the exact
        # dot marker. Other placeholder-like strings remain exclusions.
        if any(x in {"-", "--", "?", "미상", "기타"} for x in values):
            return "placeholder_dimension_name"
        known={x for x in values if x and x != "."}
        if len(known)>1:return "dimension_name_conflict"
        if not values or "" in values or "." in values or not known:unknown=True
    return "unknown_label" if unknown else "compatible"


def _source_cells(db: sqlite3.Connection):
    """Select latest validated requests for the scoped markets, streaming normalized rows once."""
    db.create_function("coverage_task_key", 3, coverage._request_task_key, deterministic=True)
    cte = market_extension._scoped_selection_cte(START, END, MARKETS)
    picked = db.execute(cte + """SELECT p.latest_job_id,p.selected_job_id,p.selected_epoch_id,p.source_url,p.task_key,
        p.query_json,p.selected_state,p.latest_state,p.selected_previous FROM picked p WHERE p.dataset='prices_origin'""").fetchall()
    specific = set()
    manifests = []
    states = defaultdict(int); scope_states = defaultdict(list)
    for row in picked:
        q = _json(row[5]); day = coverage.canonical_day(q.get("cond[trd_clcln_ymd::EQ]")); market = str(q.get("cond[whsl_mrkt_cd::EQ]", ""))
        prod = coverage.price_dimensions(q, "prices_origin"); product = prod[2] if prod else ""
        sub = str(q.get("cond[gds_sclsf_cd::EQ]", "")); state = row[6]
        normalized_state = ("source_incomplete:stale_fallback" if row[8] else state if state in ("complete", "empty")
                            else f"source_incomplete:{row[7] or state or 'unqueried'}")
        states[normalized_state] += 1
        if day and market in MARKETS and product in PRODUCTS:
            scope_states[f"{product}|{day}|{market}"].append(normalized_state)
        manifests.append({"latestJobId": row[0], "selectedJobId": row[1], "epoch": row[2], "task": row[4], "date": day, "market": market, "product": product, "subclass": sub or None, "state": normalized_state,"latestState":row[7],"selectedState":state,"selectedPrevious":bool(row[8])})
        scope = market_extension._selected_specific_scope(q, state)
        if scope: specific.add(scope)
    cursor = db.execute(cte + """SELECT r.request_id,r.retrieved_at,r.published_at,r.version_published_at,s.response_sha256,
        p.selected_state,p.query_json,n.row_no,n.payload_json FROM picked p
        JOIN request_pages r ON r.job_id=p.selected_job_id AND r.dataset=p.dataset AND r.source_url=p.source_url
          AND r.task_key=p.task_key AND r.epoch_id=p.selected_epoch_id AND r.state IN ('complete','empty')
        JOIN source_requests s ON s.request_id=r.request_id
        LEFT JOIN normalized_records n ON n.request_id=r.request_id
        WHERE p.dataset='prices_origin' AND p.selected_state IN ('complete','empty') ORDER BY p.query_json,r.request_id,n.row_no""")
    cells = defaultdict(lambda: {"amount": Decimal(0), "weight": Decimal(0), "rows": 0, "names": []})
    hashes = {}
    exclusions = defaultdict(dict)
    provenance = {}
    raw_totals=defaultdict(lambda:{"amount":Decimal(0),"weight":Decimal(0),"rows":0})
    eligible_cells=defaultdict(lambda:{"weight":Decimal(0)})
    for row in cursor:
        rid, retrieved, published, version_published, response_hash, state, query_text, _, payload_text = row
        q = _json(query_text); p = _json(payload_text)
        date = coverage.canonical_day(q.get("cond[trd_clcln_ymd::EQ]")); market = str(q.get("cond[whsl_mrkt_cd::EQ]", ""))
        dim = coverage.price_dimensions(q, "prices_origin"); product = dim[2] if dim else ""
        if not date or product not in PRODUCTS or market not in MARKETS: continue
        hashes[str(rid)] = response_hash
        provenance[str(rid)] = {"requestId": rid, "responseSha256": response_hash, "retrievedAt": retrieved,
            "publishedAt": published, "versionPublishedAt": version_published,
            "publicationTimeUnknown": published is None or version_published is None}
        if not p or state != "complete": continue
        valid, subclass = market_extension._valid_payload_scope(p, q, product, market)
        reason = None
        if not valid: reason = subclass
        elif market_extension._overlap_shadowed(product, date, market, str(q.get("cond[gds_sclsf_cd::EQ]", "")), subclass, specific): reason = "overlap_shadowed"
        elif p.get("trd_se") != "경매": reason = "not_auction"
        elif p.get("unit_nm") != "kg": reason = "not_kg"
        elif product == "garlic" and str(p.get("gds_sclsf_cd", "")) not in GARLIC_CLASSES: reason = "unverified_garlic_subclass"
        try: amount, weight = Decimal(str(p.get("totprc"))), Decimal(str(p.get("unit_tot_qty")))
        except (InvalidOperation, TypeError): amount, weight, reason = Decimal(0), Decimal(0), reason or "invalid_signed_amount_or_weight"
        if not amount.is_finite() or not weight.is_finite(): reason = reason or "invalid_signed_amount_or_weight"
        if not reason and p.get("trd_se")=="경매" and p.get("unit_nm")=="kg":
            raw_totals[(product,date,market)]["amount"]+=amount
            raw_totals[(product,date,market)]["weight"]+=weight
            raw_totals[(product,date,market)]["rows"]+=1
            raw_codes=tuple(str(p.get(code)).strip() if p.get(code) is not None else "" for code,_ in DIMENSIONS)
            eligible_cells[(product,date,market,raw_codes)]["weight"]+=weight
        if reason:
            item=exclusions[product].setdefault(reason, {"rows":0,"weightKg":Decimal(0)})
            item["rows"] += 1
            item["weightKg"] += weight if weight.is_finite() else Decimal(0)
            continue
        bad_code=_code_status(p)
        if bad_code:
            item=exclusions[product].setdefault(bad_code,{"rows":0,"weightKg":Decimal(0)})
            item["rows"]+=1;item["weightKg"]+=weight;continue
        key = (product, date, market, str(p.get("gds_sclsf_cd", "")), _cell_key(p))
        c = cells[key]; c["amount"] += amount; c["weight"] += weight; c["rows"] += 1; c["names"] = _merge_names(c["names"], _names_for(p))
    clean_exclusions = {product: {reason: {"rows": value["rows"], "weightKg": float(value["weightKg"])}
                                  for reason, value in reasons.items()} for product, reasons in exclusions.items()}
    manifests=sorted(manifests,key=lambda x:json.dumps(x,sort_keys=True,separators=(",",":")))
    raw_totals={k:dict(v) for k,v in raw_totals.items()}
    eligible_cells={k:dict(v) for k,v in eligible_cells.items()}
    return cells, {"counts":dict(states),"scopeStates":dict(scope_states)}, {"period": [START.isoformat(), END.isoformat()], "markets": list(MARKETS),
        "selectedScopeCount": len(manifests), "selectedScopesSha256": _sha(manifests),"selectedScopes":manifests,
        "requestCount": len(provenance), "requests": provenance, "responseHashes": hashes}, clean_exclusions, raw_totals, eligible_cells


def _market_cells(cells):
    out = defaultdict(dict)
    for (product, day, market, subclass, dims), val in cells.items():
        key = (product, day, subclass, dims)
        rec = out[market].setdefault(key, {"amount": Decimal(0), "weight": Decimal(0), "names": [], "rows": 0})
        rec["amount"] += val["amount"]; rec["weight"] += val["weight"]; rec["names"] = _merge_names(rec["names"], val["names"]); rec["rows"] += val["rows"]
    return out


def _scope_ready(states, product, day, market):
    if not isinstance(states, dict) or "scopeStates" not in states: return True
    scope=states["scopeStates"].get(f"{product}|{day}|{market}", [])
    return bool(scope) and all(x in ("complete", "empty") for x in scope)


def _season_number(day: dt.date) -> int:
    return 0 if day.month in (12,1,2) else 1 if day.month in (3,4,5) else 2 if day.month in (6,7,8) else 3


def _pair_sensitivity(market, product, left, right, states=None, name_policy="strict"):
    keys = {k for k in market.get(left, {}) if k[0] == product} | {k for k in market.get(right, {}) if k[0] == product}
    by_day = defaultdict(lambda: {"sum": 0.0, "weight": 0.0, "cells": 0})
    excluded = defaultdict(lambda:{"cells":0,"weightKg":Decimal(0)})
    for k in keys:
        code_reason=_dimension_code_status(k[3] if isinstance(k[3],(tuple,list)) else (None,))
        if code_reason:
            a,b=market.get(left,{}).get(k),market.get(right,{}).get(k);present=[x for x in (a,b) if x]
            excluded[code_reason]["cells"]+=1;excluded[code_reason]["weightKg"]+=sum((x["weight"] for x in present),Decimal(0));continue
        if not _scope_ready(states, product, k[1], left) or not _scope_ready(states, product, k[1], right):
            excluded["source_incomplete"]["cells"] += 1; continue
        a, b = market.get(left, {}).get(k), market.get(right, {}).get(k)
        if not a or not b:
            x=a or b;excluded["missing_market_cell"]["cells"]+=1;excluded["missing_market_cell"]["weightKg"]+=x["weight"] if x else Decimal(0);continue
        if min(a["amount"], b["amount"], a["weight"], b["weight"]) <= 0:
            excluded["nonpositive_net_cell"]["cells"] += 1;excluded["nonpositive_net_cell"]["weightKg"]+=max(a["weight"],Decimal(0))+max(b["weight"],Decimal(0));continue
        status = _name_status(_combine_market_names(a["names"],b["names"]),name_policy)
        if status not in ("compatible","unknown_label"): excluded[status]["cells"] += 1;excluded[status]["weightKg"]+=min(a["weight"],b["weight"]);continue
        weight = min(a["weight"], b["weight"])
        value = math.log(float(a["amount"] / a["weight"]) / float(b["amount"] / b["weight"]))
        day = by_day[k[1]]; day["sum"] += float(weight) * value; day["weight"] += float(weight); day["cells"] += 1
    gaps = {d: v["sum"] / v["weight"] for d,v in by_day.items() if v["weight"] > 0}
    return {"markets": [left, right], "pairedDates": len(gaps), "pairedCells": sum(x["cells"] for x in by_day.values()),
        "meanDailyLogGap": sum(gaps.values()) / len(gaps) if gaps else None, "excludedCellCounts": {k:v["cells"] for k,v in excluded.items()},"excludedCellWeightKg":{k:float(v["weightKg"]) for k,v in excluded.items()},
        "dailyLogGaps": gaps}


def _daily_hac(values: dict[dt.date, float], lag: int):
    dates = sorted(values); n = len(dates)
    if not n: return None
    mean = sum(values[d] for d in dates) / n
    residual = {d: values[d] - mean for d in dates}
    gamma0 = sum(x*x for x in residual.values()) / n
    var = gamma0
    for k in range(1, lag + 1):
        cov = sum(residual.get(d, 0.0) * residual.get(d - dt.timedelta(days=k), 0.0) for d in dates) / n
        var += 2 * (1 - k / (lag + 1)) * cov
    if not math.isfinite(var) or var <= 0: return {"status":"unavailable_nonpositive_or_nonfinite_hac_variance","n":n,"lagDays":lag}
    se = math.sqrt(var / n)
    z = mean / se if se else (0.0 if mean == 0 else math.copysign(math.inf, mean))
    return {"status":"available","mean": mean, "se": se, "z": z if math.isfinite(z) else None, "pTwoSided": math.erfc(abs(z) / math.sqrt(2)) if math.isfinite(z) else 0.0,
            "ci90": [mean - 1.6448536269514722 * se, mean + 1.6448536269514722 * se], "n": n, "lagDays": lag}


def _block_lower(values: dict[dt.date, float], threshold: float, seed: int = 20261005, reps: int = 9999):
    import random
    dates = sorted(values); n = len(dates)
    if not n: return None
    first, last = dates[0], dates[-1]
    calendar = [first + dt.timedelta(days=i) for i in range((last-first).days+1)]
    marked = [float(abs(values[d]) <= threshold) if d in values else None for d in calendar]
    observed_share = sum(float(abs(values[d]) <= threshold) for d in dates) / n
    if len({float(abs(values[d]) <= threshold) for d in dates}) < 2:
        return {"status":"unavailable_degenerate_indicator_bootstrap","observed":observed_share,"n":n,"blockDays":28,"replicates":reps,"seed":seed}
    rng = random.Random(seed); block = 28; estimates = []; T=len(calendar)
    for _ in range(reps):
        sample = []
        while len(sample) < T:
            start = rng.randrange(len(marked))
            sample.extend(marked[(start + j) % len(marked)] for j in range(block))
        observed = [x for x in sample[:T] if x is not None]
        if not observed: return {"status":"unavailable_no_observed_dates_in_bootstrap_replicate","observed":observed_share,"n":n,"blockDays":block,"replicates":reps,"seed":seed}
        estimates.append(sum(observed) / len(observed))
    estimates.sort()
    # Center the block-bootstrap distribution at the null boundary for a one-sided test.
    deviations = sorted(x - observed_share for x in estimates)
    p_value = (1 + sum(x >= observed_share - .8 for x in deviations)) / (reps + 1)
    return {"status":"available","method":"circular calendar moving block bootstrap","observed": observed_share, "lower95": estimates[max(0, math.ceil(.05*reps)-1)], "pValue": p_value, "n": n,
            "calendarDays":T,"blockDays": block, "replicates": reps, "seed": seed}


def _holm(rows, key):
    ordered = sorted(rows, key=lambda r: (r[key]["pValue"] if r[key].get("pValue") is not None else 1.0, r["productId"]))
    running = 0.0
    for i, row in enumerate(ordered):
        p = row[key].get("pValue"); adjusted = min(1.0, max(running, (len(ordered)-i) * (p if p is not None else 1.0)))
        running = adjusted; row[key]["holmAdjustedPValue"] = adjusted; row[key]["holmSignificantAt005"] = adjusted <= .05


def _infer(values: dict[dt.date, float]):
    n = len(values); years = {d.year for d in values}; seasons = defaultdict(int)
    season = _season_number
    for d in values: seasons[season(d)] += 1
    gate = n >= 90 and len(years) >= 2 and sum(x >= 10 for x in seasons.values()) >= 3
    if not gate:
        why = []
        if n < 90: why.append("fewer_than_90_paired_dates")
        if len(years) < 2: why.append("fewer_than_2_years")
        if sum(x >= 10 for x in seasons.values()) < 3: why.append("fewer_than_3_seasons_with_10_dates")
        return {"status": "inference_unavailable", "reasons": why, "pairedDates": n, "years": sorted(years), "seasonDates": dict(seasons)}
    hac28, hac9 = _daily_hac(values, 28), _daily_hac(values, 9)
    if hac28["status"] != "available" or hac9["status"] != "available":
        return {"status":"inference_unavailable","reasons":[hac28.get("status"),hac9.get("status")],"pairedDates":n,"years":sorted(years),"seasonDates":dict(seasons)}
    mean = hac28["mean"]; margin = math.log(1.05); bound=math.log(1.10)
    def one_sided(diff, se, upper):
        if se == 0: return 0.0 if ((diff < upper) if upper is not None else (diff > -upper)) else 1.0
        z=(upper-diff)/se if upper is not None else (diff+(-upper))/se
        return .5*math.erfc(z/math.sqrt(2))
    # TOST: H0 mean <= -margin OR mean >= margin. Equivalence requires both tails below alpha.
    se=hac28["se"]
    p_low = .5*math.erfc(((mean + margin)/se)/math.sqrt(2)) if se else (0.0 if mean > -margin else 1.0)
    p_high = .5*math.erfc(((margin - mean)/se)/math.sqrt(2)) if se else (0.0 if mean < margin else 1.0)
    share=_block_lower(values,bound)
    if share.get("status") != "available":
        return {"status":"inference_unavailable","reasons":[share.get("status")],"pairedDates":n,"years":sorted(years),"seasonDates":dict(seasons),"meanDifferenceHAC28":hac28,"meanDifferenceHAC9":hac9}
    sensitivity={}
    for label, delta in (("3_percent", math.log(1.03)), ("10_percent", math.log(1.10))):
        if se:
            lo=.5*math.erfc(((mean+delta)/se)/math.sqrt(2)); hi=.5*math.erfc(((delta-mean)/se)/math.sqrt(2))
        else:
            lo=0.0 if mean > -delta else 1.0; hi=0.0 if mean < delta else 1.0
        sensitivity[label]={"boundary":delta,"pValue":max(lo,hi),"pLower":lo,"pUpper":hi,"hacDays":28}
    return {"status":"available", "pairedDates":n,"years":sorted(years),"seasonDates":dict(seasons),
        "meanDifferenceHAC28":hac28,"meanDifferenceHAC9":hac9,
        "equivalenceTOST":{"boundary":margin,"pValue":max(p_low,p_high),"pLower":p_low,"pUpper":p_high,"hacDays":28},
        "equivalenceTOSTHAC9":{"boundary":margin,"pValue":max(.5*math.erfc(((mean+margin)/hac9["se"])/math.sqrt(2)),.5*math.erfc(((margin-mean)/hac9["se"])/math.sqrt(2))),"hacDays":9},
        "smallGapDays":{**share,"boundary":bound,"requiredShare":.8,"passesUnadjusted":share["lower95"] >= .8},
        "meanZeroTest":{"pValue":hac28["pTwoSided"],"hacDays":28},
        "sensitivityMeanBoundaries":sensitivity}


def analyze_cells(cells, states=None, provenance=None, exclusions=None, raw_totals=None, eligible_cells=None, name_policy="strict"):
    """Pair strict compatible cells and run frozen descriptive/inference protocol."""
    market=_market_cells(cells);records=[];test_records=[];product_reports=[]
    season=_season_number
    for product in PRODUCTS:
        daily=defaultdict(lambda:{"sum":0.0,"weight":0.0,"cells":0})
        raw=defaultdict(lambda:{m:{"a":Decimal(0),"w":Decimal(0)} for m in MARKETS})
        source_w=defaultdict(lambda:{m:Decimal(0) for m in MARKETS})
        excluded=defaultdict(lambda:{"cells":0,"weightKg":Decimal(0)})
        unknown_cells=0;unknown_weight=0.0
        if raw_totals is not None:
            for (p,day,m),x in raw_totals.items():
                if p==product:
                    day_obj=dt.date.fromisoformat(day)
                    raw[day_obj][m]["a"]+=x["amount"];raw[day_obj][m]["w"]+=x["weight"]
        if eligible_cells is not None:
            for (p,day,m,codes),x in eligible_cells.items():
                if p==product:
                    day_obj=dt.date.fromisoformat(day)
                    source_w[day_obj][m]+=max(x["weight"],Decimal(0))
        elif raw_totals is not None:
            for (p,day,m),x in raw_totals.items():
                if p==product:
                    day_obj=dt.date.fromisoformat(day)
                    source_w[day_obj][m]+=max(x["weight"],Decimal(0))
        else:
            for m in MARKETS:
                for (p,day,sub,dims),x in market.get(m,{}).items():
                    if p!=product: continue
                    day_obj=dt.date.fromisoformat(day)
                    raw[day_obj][m]["a"]+=x["amount"];raw[day_obj][m]["w"]+=x["weight"]
                    source_w[day_obj][m]+=max(x["weight"],Decimal(0))
        seoul=defaultdict(lambda:{"amount":Decimal(0),"weight":Decimal(0),"names":[],"rows":0})
        for m in (GARAK,GANGSEO):
            for k,x in market.get(m,{}).items():
                if k[0]!=product:continue
                z=seoul[k];z["amount"]+=x["amount"];z["weight"]+=x["weight"];z["names"]=_merge_names(z["names"],x["names"]);z["rows"]+=x["rows"]
        keys={k for k in market.get(GWANGJU,{}) if k[0]==product}|{k for k in seoul if k[0]==product}
        incomplete_days=set()
        for k in keys:
            g=market.get(GWANGJU,{}).get(k);s=seoul.get(k)
            code_reason=_dimension_code_status(k[3] if isinstance(k[3],(tuple,list)) else (None,))
            if code_reason:
                excluded[code_reason]["cells"]+=1
                excluded[code_reason]["weightKg"]+=(g["weight"] if g else Decimal(0))+(s["weight"] if s else Decimal(0))
                continue
            if states and "scopeStates" in states and any(not _scope_ready(states,product,k[1],m) for m in MARKETS):
                incomplete_days.add(k[1]);excluded["source_incomplete"]["cells"]+=1
                excluded["source_incomplete"]["weightKg"]+=(g["weight"] if g else Decimal(0))+(s["weight"] if s else Decimal(0));continue
            if not g or not s:
                x=g or s;excluded["missing_market_cell"]["cells"]+=1;excluded["missing_market_cell"]["weightKg"]+=x["weight"] if x else Decimal(0);continue
            if g["weight"]<=0 or s["weight"]<=0 or g["amount"]<=0 or s["amount"]<=0:
                excluded["nonpositive_net_cell"]["cells"]+=1;excluded["nonpositive_net_cell"]["weightKg"]+=max(g["weight"],Decimal(0))+max(s["weight"],Decimal(0));continue
            status=_name_status(_combine_market_names(g["names"],s["names"]),name_policy)
            if status not in ("compatible","unknown_label"):excluded[status]["cells"]+=1;excluded[status]["weightKg"]+=min(g["weight"],s["weight"]);continue
            wg,ws=g["weight"],s["weight"];shared=min(wg,ws);d=dt.date.fromisoformat(k[1]);diff=math.log(float(g["amount"]/wg)/float(s["amount"]/ws))
            day=daily[d];day["sum"]+=float(shared)*diff;day["weight"]+=float(shared);day["cells"]+=1
            if status=="unknown_label":unknown_cells+=1;unknown_weight+=float(shared)
            records.append({"date":k[1],"productId":product,"subclass":k[2],"dimensionCodes":list(k[3]),"gwangjuPrice":float(g["amount"]/wg),"seoulPrice":float(s["amount"]/ws),"sharedWeightKg":float(shared),"logGap":diff})
        values={d:x["sum"]/x["weight"] for d,x in daily.items() if x["weight"]>0}
        matched_weight=sum(x["weight"] for x in daily.values());matched_cells=sum(x["cells"] for x in daily.values())
        daily_coverage={}
        for d in sorted(set(daily)|set(source_w)):
            x=daily.get(d,{"cells":0,"weight":0.0})
            sw=source_w[d][GARAK]+source_w[d][GANGSEO]
            daily_coverage[d.isoformat()]={"matchedCells":x["cells"],"matchedWeightKg":x["weight"],"gwangjuEligibleWeightKg":float(source_w[d][GWANGJU]),"garakEligibleWeightKg":float(source_w[d][GARAK]),"gangseoEligibleWeightKg":float(source_w[d][GANGSEO]),"matchedShareOfGwangju":x["weight"]/float(source_w[d][GWANGJU]) if source_w[d][GWANGJU]>0 else None,"matchedShareOfSeoul":x["weight"]/float(sw) if sw>0 else None}
        def periods(mode):
            ids=sorted({(d.year if mode=="year" else season(d)) for d in values})
            result={}
            for ident in ids:
                dates=[d for d in values if (d.year if mode=="year" else season(d))==ident]
                cell_n=sum(daily[d]["cells"] for d in dates);weight=sum(daily[d]["weight"] for d in dates)
                gw=sum(source_w[d][GWANGJU] for d in dates);sl=sum(source_w[d][GARAK]+source_w[d][GANGSEO] for d in dates)
                label=str(ident if mode=="year" else ("DJF","MAM","JJA","SON")[ident])
                gar=sum(source_w[d][GARAK] for d in dates);gang=sum(source_w[d][GANGSEO] for d in dates)
                result[label]={"pairedDates":len(dates),"meanLogGap":sum(values[d] for d in dates)/len(dates),"matchedCells":cell_n,"matchedWeightKg":weight,"gwangjuEligibleWeightKg":float(gw),"garakEligibleWeightKg":float(gar),"gangseoEligibleWeightKg":float(gang),"seoulEligibleWeightKg":float(sl),"matchedShareOfGwangju":weight/float(gw) if gw>0 else None,"matchedShareOfSeoul":weight/float(sl) if sl>0 else None}
            return result
        raw_gaps={}
        for day,x in raw.items():
            gw=x[GWANGJU];a=x[GARAK]["a"]+x[GANGSEO]["a"];w=x[GARAK]["w"]+x[GANGSEO]["w"]
            raw_gaps[day]=(math.log(float(gw["a"]/gw["w"])/float(a/w)) if gw["w"]>0 and w>0 and gw["a"]>0 and a>0 else None)
        product_report={"productId":product,"matchedDates":len(values),"matchedCells":matched_cells,"matchedWeightKg":matched_weight,"dailyCoverage":daily_coverage,"excludedCellCounts":{k:v["cells"] for k,v in excluded.items()},"excludedCellWeightKg":{k:float(v["weightKg"]) for k,v in excluded.items()},"sourceIncompleteDates":len(incomplete_days),"yearCoverage":periods("year"),"seasonCoverage":periods("season"),"inference":_infer(values),"pairedDailyLogGaps":{d.isoformat():v for d,v in values.items()},"rawAggregateDailyLogGaps":{str(d):v for d,v in raw_gaps.items()}}
        if name_policy=="recorded_code":
            product_report.update({"namePolicy":name_policy,"unknownLabelMatchedCells":unknown_cells,"unknownLabelMatchedWeightKg":unknown_weight,
                "unknownLabelMatchedCellShare":unknown_cells/matched_cells if matched_cells else None,
                "unknownLabelMatchedWeightShare":unknown_weight/matched_weight if matched_weight else None})
        product_report["marketSensitivities"]={"gwangjuVsGarak":_pair_sensitivity(market,product,GWANGJU,GARAK,states,name_policy),"gwangjuVsGangseo":_pair_sensitivity(market,product,GWANGJU,GANGSEO,states,name_policy),"garakVsGangseo":_pair_sensitivity(market,product,GARAK,GANGSEO,states,name_policy)}
        product_reports.append(product_report)
    for p in product_reports:
        inf=p["inference"]
        for name,key in (("meanZeroTest","meanZeroTest"),("equivalenceTOST","equivalenceTOST"),("smallGapDays","smallGapDays")):
            item=inf.get(key,{}) if inf.get("status")=="available" else {}
            test_records.append({"productId":p["productId"],"test":name,"pValue":item.get("pValue"),"available":inf.get("status")=="available"})
    # Fixed family of 15 tests, including unavailable tests with p=1.
    ordered=sorted(test_records,key=lambda r:(r["pValue"] if r["pValue"] is not None else 1.0,r["productId"],r["test"]))
    running=0.0
    for i,r in enumerate(ordered):
        running=max(running,min(1.0,(len(ordered)-i)*(r["pValue"] if r["pValue"] is not None else 1.0)))
        r["holmAdjustedPValue"]=running;r["holmSignificantAt005"]=running<=.05
    lookup={(r["productId"],r["test"]):r for r in test_records}
    for p in product_reports:
        p["holmTests"]={test:lookup[(p["productId"],test)] for test in ("meanZeroTest","equivalenceTOST","smallGapDays")}
        if p["inference"].get("status")=="available":
            p["inference"]["conclusion"]=("practically_small" if p["holmTests"]["equivalenceTOST"]["holmSignificantAt005"] and p["holmTests"]["smallGapDays"]["holmSignificantAt005"] and p["inference"]["smallGapDays"]["lower95"] >= .8 else "inconclusive")
    protocol=PROTOCOL if name_policy=="strict" else {**PROTOCOL,"id":PROTOCOL["id"]+"-recorded-code-exploratory","pairing":"exact four nonempty dimension codes; allow only missing labels or the exact '.' label; exclude other placeholders (-, --, ?, 미상, 기타) and conflicts among present nonplaceholder labels"}
    report={"schemaVersion":1,"status":"complete","protocol":protocol,"protocolSha256":_sha(protocol),
        "inputSha256":_sha(provenance or {}),"selectedQueryStates":states or {},"provenance":provenance or {},
        "exclusions":exclusions or {},"products":product_reports,"cellPairs":records,"multipleTesting":test_records}
    if name_policy=="recorded_code":
        report.update({"namePolicy":name_policy,"exploratoryPostHoc":True,
            "interpretation":"Post-hoc exploratory sensitivity. Recorded code agreement does not prove same physical size or quality. The prior strict zero-cell result and raw aggregate series are unchanged."})
    return report


def run_analysis(db_path: Path,name_policy="strict"):
    db=sqlite3.connect("file:"+str(db_path.resolve())+"?mode=ro",uri=True); db.row_factory=sqlite3.Row;db.execute("PRAGMA query_only=ON")
    try:
        db.execute("BEGIN"); cells,states,prov,excluded,raw,eligible_cells=_source_cells(db); report=analyze_cells(cells,states,prov,excluded,raw,eligible_cells,name_policy);db.rollback();return report
    finally: db.close()


def write_report(report, json_path: Path, markdown_path: Path):
    json_text=json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n"
    recorded=report.get("namePolicy")=="recorded_code"
    lines=["# 도매시장 matched-cell 가격차 분석","",
        f"기간: {report['protocol']['period'][0]}–{report['protocol']['period'][1]}. 광주({GWANGJU}) 대 서울 가락·강서({GARAK}, {GANGSEO}).",
        ("기록 코드 일치만 요구하는 post-hoc 탐색 민감도입니다. 명칭 누락과 정확히 `.`인 경우만 허용합니다. `-`, `--`, `?`, `미상`, `기타` 등 다른 placeholder 및 존재하는 비-placeholder 명칭 충돌은 제외합니다. 이전 strict 0셀 결과와 전체 일가격 보조 결과는 그대로 유지됩니다. 코드가 실제 크기·품질의 동일성을 보장하지 않습니다."
         if recorded else "코드 4종 일치와 비어 있지 않은 비-placeholder 명칭 호환을 요구합니다. 동일 코드만으로 물리적 품질 동등성을 증명하지는 않습니다."),
        "이는 세 도매시장의 관측 가격차이며 도시 전체 효과나 인과 효과로 해석하지 않습니다. 품질·운송·출하자·구매자 구성은 통제되지 않습니다.","",
        "| 품목 | 날짜 | 페어 셀 | 매칭 순중량 (kg) | 평균 가격비 (90% CI) | 작은 차이 날짜 비율 (95% 하한) | Holm p: 0 / TOST / 비율 | HAC9 TOST p | 평균 0차이 검정 | 실무상 작은 차이 판정 |"+(" 명칭 불명 셀/중량 비율 |" if recorded else ""),
        "|---|---:|---:|---:|---|---|---|---:|---|---|"+("---:|" if recorded else "")]
    for p in report["products"]:
        inf=p["inference"]; hac=inf.get("meanDifferenceHAC28",{}); ci=hac.get("ci90")
        ratio=(math.exp(hac["mean"]) if hac.get("mean") is not None else None)
        ratio_ci=[math.exp(x) for x in ci] if ci else None
        mean_text=(f"{ratio:.3f} ({ratio_ci[0]:.3f}–{ratio_ci[1]:.3f})" if ratio is not None else "추론 불가")
        share=inf.get("smallGapDays",{})
        share_text=(f"{share['observed']:.1%} (하한 {share['lower95']:.1%})" if share.get("status")=="available" else "추론 불가")
        tests=p["holmTests"]
        ps=" / ".join("추론불가" if not tests[k]["available"] else f"{tests[k]['holmAdjustedPValue']:.3g}" for k in ("meanZeroTest","equivalenceTOST","smallGapDays"))
        hac9=inf.get("equivalenceTOSTHAC9",{}).get("pValue")
        hac9_text=f"{hac9:.3g}" if hac9 is not None else "추론불가"
        zero=tests["meanZeroTest"]
        mean=inf.get("meanDifferenceHAC28",{}).get("mean")
        if not zero["available"]:
            zero_text="추론 불가"
        elif zero["holmSignificantAt005"]:
            direction="광주 높음" if mean is not None and mean>0 else "광주 낮음" if mean is not None and mean<0 else "방향 0"
            zero_text=f"차이 검출: {direction} (Holm p={zero['holmAdjustedPValue']:.3g})"
        else:
            zero_text=f"유의차 미검출 (Holm p={zero['holmAdjustedPValue']:.3g})"
        practical=("작은 차이 확인" if inf.get("conclusion")=="practically_small" else
                   "추론 불가" if inf.get("status")!="available" else "작은 차이 미확인")
        row=f"| {p['productId']} | {p['matchedDates']} | {p['matchedCells']} | {p['matchedWeightKg']:.1f} | {mean_text} | {share_text} | {ps} | {hac9_text} | {zero_text} | {practical} |"
        if recorded:
            count_share=p.get("unknownLabelMatchedCellShare");weight_share=p.get("unknownLabelMatchedWeightShare")
            row+=f" {count_share:.1%} / {weight_share:.1%} |" if count_share is not None and weight_share is not None else " 추론불가 |"
        lines.append(row)
    for p in report["products"]:
        years=", ".join(f"{k}: {v['pairedDates']}일·{v['meanLogGap']:+.3f}" for k,v in p["yearCoverage"].items()) or "없음"
        seasons=", ".join(f"{k}: {v['pairedDates']}일·{v['meanLogGap']:+.3f}" for k,v in p["seasonCoverage"].items()) or "없음"
        detail=f"- {p['productId']}: 연도({years}), 계절({seasons}); placeholder 제외 {p['excludedCellWeightKg'].get('placeholder_dimension_name',0):.1f} kg, source incomplete dates {p['sourceIncompleteDates']}."
        if recorded:detail+=f" 명칭 불명 매칭 {p['unknownLabelMatchedCells']}셀/{p['unknownLabelMatchedWeightKg']:.1f}kg."
        lines.append(detail)
    lines.extend(["","평균 격차의 90% CI와 TOST는 Bartlett HAC 28일, HAC 9일은 민감도 분석입니다. 작은 차이 날짜 비율은 28일 원형 달력 이동블록 부트스트랩 9,999회로 평가합니다. Holm 보정은 5품목×3 검정의 고정 15개 가족에 적용하며 추론 불가 검정은 p=1로 둡니다. 3%·10% 평균 경계는 JSON 민감도 결과를 확인하세요.","",
        "연도·계절별 평균격차 및 셀·중량 커버리지는 JSON의 yearCoverage, seasonCoverage, dailyCoverage에 있습니다. rawAggregateDailyLogGaps는 전체 품목 구성 차이를 포함하는 보조 결과입니다. 코드·명칭 불확실성과 원천 조회 미완료는 제외/차단되며 빈 날짜는 보간하지 않습니다.","",
        f"원천 요청 수: {report['provenance'].get('requestCount',0)}; 선택 query scopes: {report['provenance'].get('selectedScopeCount',0)}. Scope 상태: {json.dumps(report.get('selectedQueryStates',{}).get('counts',{}),ensure_ascii=False,sort_keys=True)}.",
        "원자료는 회고적으로 검증된 retrievals이며 원래 공개 시각을 보장하지 않습니다. 선택된 자료 범위가 작아도 코드 호환은 규격 의미의 대체 검증이 아닙니다.",""])
    _atomic_text(json_path,json_text);_atomic_text(markdown_path,"\n".join(lines))


def _atomic_text(path: Path,text: str):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as stream:
            stream.write(text);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--db",default="data/agri.sqlite");ap.add_argument("--output",required=True);ap.add_argument("--report");ap.add_argument("--name-policy",choices=("strict","recorded_code"),default="strict");args=ap.parse_args(argv)
    db=Path(args.db).resolve();out=Path(args.output).resolve();md=Path(args.report).resolve() if args.report else out.parent/"report.md"
    if out==db or md==db: raise ValueError("output paths must not overwrite the SQLite database")
    if out==md: raise ValueError("JSON and Markdown outputs must use different paths")
    report=run_analysis(db,args.name_policy);write_report(report,out,md);return 0


if __name__=="__main__": raise SystemExit(main())
