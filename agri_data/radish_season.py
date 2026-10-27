"""Frozen, retrospective radish subclass-composition Ridge experiment.

The module is deliberately split into a pure ``build`` path and a read-only
SQLite ``extract_market_data`` path so protocol behavior can be tested without
opening the project database.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sqlite3
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import dashboard, direct_horizon as dh, market_extension as me, market_recompose as mr, modeling, recent_window as rw

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data/agri.sqlite"
INPUT_PRICES = dh.INPUT_PRICES
INPUT_PREDICTIONS = dh.INPUT_PREDICTIONS
INPUT_RIDGE = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
INPUT_RECOMPOSE = ROOT / "outputs/experiments/market-recompose-20261007/report.json"
OUTPUT = ROOT / "outputs/experiments/radish-season-20261007/report.json"
EXPECTED = {
    "pricesSha256": "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8",
    "predictionsSha256": "c41ed5d8799306f151b1c5135399a606982f12623bf709a4438969d8cc389113",
    "directRidgeSha256": "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67",
    "marketRecomposeSha256": "e5f23f1012dad2acc774c83f995e2ee2851409af913ee82483aa3d953ee87581",
}
PRODUCT = "radish"
MARKETS = tuple(dashboard.MARKETS)
HORIZONS = tuple(range(1, 11))
CATEGORIES = ("01", "02", "03", "04", "05", "other")
KNOWN_NAMES = {"01": "봄무", "02": "여름무", "03": "고냉지무", "04": "가을무", "05": "저장무"}
EVAL_START, EVAL_END = dt.date(2023, 1, 1), dt.date(2026, 9, 27)
MIN_TRAIN, MAX_AGE = 90, 14
COMPARATORS, LOSSES, BANDWIDTHS = ("directRidge", "persistence", "marketRecompose"), ("mae", "mape"), (9, 28)
PROTOCOL = {
    "id": "radish-season-share-ridge-2026-10-07",
    "categories": {**KNOWN_NAMES, "other": "모든 기타/미확인 코드 합계; 원 코드 유지"},
    "features": "기존 modeling._feature_row + pre-origin 7/28 calendar-day signed subclass net-kg shares and denominator flags; latest positive category/same-day aggregate price log ratio, age, presence (<=14d); negative adjustment row ratios",
    "fit": "per horizon expanding StandardScaler+Ridge(alpha=10), first eligible origin per month, training targets strictly before fit origin, >=90 distinct target dates; frozen direct Ridge fallback",
    "evaluation": "targets 2023-01-01..2026-09-27, origins >=2023-01-01, horizons 1..10",
    "comparisonFamily": "10 horizons x 3 comparators x 2 losses = 60 two-sided DM tests; calendar Bartlett HAC9/HAC28 and separate Holm correction; unavailable p=1",
    "limitations": "retrospective revised snapshot; historical first publication time unknown; exploratory result is not an operating model",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _day(value: Any) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def _finite(value: Any) -> float | None:
    try: x = float(value)
    except (TypeError, ValueError, OverflowError): return None
    return x if math.isfinite(x) else None


def _source_series(markets: dict[str, Any]):
    """Map category -> date -> signed aggregate source statistics."""
    out = {c: {} for c in CATEGORIES}
    for raw_code, days in markets.items():
        code = str(raw_code)
        cat = code if code in KNOWN_NAMES else "other"
        for raw_day, item in days.items():
            day = _day(raw_day)
            # If several unrecognized codes are supplied, sum signed rows before deriving price.
            row = out[cat].setdefault(day, {"amount": Decimal(0), "weight": Decimal(0), "rows": 0, "negativeRows": 0, "names": set(), "codes": set()})
            row["amount"] += Decimal(str(item.get("amount", 0)))
            row["weight"] += Decimal(str(item.get("weight", 0)))
            row["rows"] += int(item.get("rows", 0))
            row["negativeRows"] += int(item.get("negativeRows", 0))
            row["names"].update(item.get("names", ()))
            row["codes"].update(item.get("codes", (code,)))
    return out


def _feature_values(markets: dict[str, Any], aggregate, origin: dt.date):
    cats = markets if set(markets) == set(CATEGORIES) and all(isinstance(v, dict) for v in markets.values()) else _source_series(markets)
    vals, audit = [], {}
    for window in (7, 28):
        start = origin - dt.timedelta(days=window)
        amounts, weights = [], []
        for cat in CATEGORIES:
            rows = [r for d, r in cats[cat].items() if start <= d < origin]
            w = sum((r["weight"] for r in rows), Decimal(0))
            total = sum((x["weight"] for c in CATEGORIES for d, x in cats[c].items() if start <= d < origin), Decimal(0))
            valid = total > 0
            share = float(w / total) if valid else 0.0
            neg = sum(r["negativeRows"] for r in rows)
            count = sum(r["rows"] for r in rows)
            vals.extend((share, float(valid)))
            audit[f"{cat}/share{window}"] = {"value": share if valid else None, "denominatorAvailable": valid, "signedNetKg": str(w), "totalSignedNetKg": str(total)}
            amounts.append(neg / count if count else 0.0)
        vals.extend(amounts)
    # Category price relative to same-day aggregate price; only positive observed category prices.
    for cat in CATEGORIES:
        eligible = []
        for day, r in cats[cat].items():
            if day >= origin or (origin-day).days > MAX_AGE or r["weight"] <= 0 or r["amount"] <= 0: continue
            a = aggregate.get(day)
            if a and _finite(a[0]) and float(a[0]) > 0:
                eligible.append((day, float(r["amount"] / r["weight"]), float(a[0])))
        if eligible:
            day, price, total_price = max(eligible)
            vals.extend((math.log(price / total_price), float((origin-day).days), 1.0))
            audit[f"{cat}/relativePrice"] = {"date": day.isoformat(), "ageDays": (origin-day).days, "available": True}
        else:
            vals.extend((0.0, float(MAX_AGE+1), 0.0))
            audit[f"{cat}/relativePrice"] = {"date": None, "ageDays": None, "available": False}
    return tuple(vals), audit


def _combine_markets(markets):
    """Sum both market streams per subclass/date without losing unknown codes."""
    combined = defaultdict(dict)
    for market in MARKETS:
        for code, days in markets.get(market, {}).items():
            for day, row in days.items():
                key = _day(day)
                dst = combined[code].setdefault(key, {"amount": Decimal(0), "weight": Decimal(0), "rows": 0, "negativeRows": 0, "names": set(), "codes": set()})
                dst["amount"] += Decimal(str(row["amount"]))
                dst["weight"] += Decimal(str(row["weight"]))
                dst["rows"] += row["rows"]
                dst["negativeRows"] += row["negativeRows"]
                dst["names"].update(row["names"])
                dst["codes"].update(row["codes"])
    return dict(combined)


def _extract_market_data(db_path: Path, raw: dict[str, Any], core):
    """Read all selected radish subclass scopes; no API calls or writes."""
    start, end = _day(raw["meta"]["contextStart"]), _day(raw["meta"]["contextEnd"])
    db = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    db.create_function("coverage_task_key", 3, me.coverage._request_task_key, deterministic=True)
    try:
        db.execute("BEGIN")
        broad_states, broad_audit = me._core_broad_query_states(db, start)
        for offset in range((end - start).days + 1):
            day = start + dt.timedelta(days=offset)
            for market in MARKETS:
                state = broad_states.get((PRODUCT, day, market), {}).get("selectedState")
                if state not in ("complete", "empty"):
                    raise ValueError(f"incomplete broad radish query {market}/{day}: {state or 'missing'}")
        cte = me._scoped_selection_cte(start, end, MARKETS)
        scope_rows = list(db.execute(cte + """SELECT p.query_json,p.selected_state,p.selected_job_id,p.selected_epoch_id FROM picked p
            WHERE p.dataset='prices_origin' AND json_extract(p.query_json,'$.\"cond[trd_clcln_ymd::EQ]\"') BETWEEN ? AND ?
              AND json_extract(p.query_json,'$.\"cond[whsl_mrkt_cd::EQ]\"') IN (?,?)
              AND json_extract(p.query_json,'$.\"cond[gds_lclsf_cd::EQ]\"')='11'
              AND json_extract(p.query_json,'$.\"cond[gds_mclsf_cd::EQ]\"')='01'""", (start.isoformat(), end.isoformat(), *MARKETS)))
        specific_scopes = set()
        for scope in scope_rows:
            query = json.loads(scope["query_json"])
            date_text = me.coverage.canonical_day(query.get("cond[trd_clcln_ymd::EQ]"))
            market = str(query.get("cond[whsl_mrkt_cd::EQ]", ""))
            subclass = str(query.get("cond[gds_sclsf_cd::EQ]", ""))
            if date_text and market and subclass and scope["selected_state"] in ("complete", "empty"):
                specific_scopes.add((date_text, market, subclass))
        # Deliberately query without the dashboard's subclass allow-list.
        sql = cte + """SELECT p.query_json,p.selected_state,p.selected_job_id,p.selected_epoch_id,
          r.request_id,r.page_no,n.reference_date,n.payload_json
          FROM picked p LEFT JOIN request_pages r ON r.job_id=p.selected_job_id AND r.dataset=p.dataset
            AND r.source_url=p.source_url AND r.task_key=p.task_key AND r.epoch_id=p.selected_epoch_id AND r.state='complete'
          LEFT JOIN normalized_records n ON n.request_id=r.request_id
          WHERE p.dataset='prices_origin' AND json_extract(p.query_json,'$.\"cond[trd_clcln_ymd::EQ]\"') BETWEEN ? AND ?
            AND json_extract(p.query_json,'$.\"cond[whsl_mrkt_cd::EQ]\"') IN (?,?)
            AND json_extract(p.query_json,'$.\"cond[gds_lclsf_cd::EQ]\"')='11'
            AND json_extract(p.query_json,'$.\"cond[gds_mclsf_cd::EQ]\"')='01'
          ORDER BY p.task_key,r.page_no,n.row_no"""
        states, scopes, req_hashes = {}, set(), {}
        sums = defaultdict(lambda: {"amount": Decimal(0), "weight": Decimal(0), "rows": 0, "negativeRows": 0, "names": set(), "codes": set()})
        for rec in db.execute(sql, (start.isoformat(), end.isoformat(), *MARKETS)):
            q = json.loads(rec["query_json"])
            market = str(q.get("cond[whsl_mrkt_cd::EQ]")); day = _day(q.get("cond[trd_clcln_ymd::EQ]"))
            if rec["selected_state"] not in ("complete", "empty"):
                raise ValueError(f"incomplete selected radish scope {market}/{day}: {rec['selected_state']}")
            subclass = str(q.get("cond[gds_sclsf_cd::EQ]", "")) or "<null>"
            code = str(q.get("cond[gds_sclsf_cd::EQ]", ""))
            scopes.add((market, day.isoformat(), subclass, rec["selected_state"], rec["selected_job_id"], rec["selected_epoch_id"]))
            if rec["request_id"] is not None:
                req = db.execute("SELECT response_sha256 FROM source_requests WHERE request_id=?", (rec["request_id"],)).fetchone()
                if req and req[0]: req_hashes[str(rec["request_id"])] = req[0]
            if not rec["payload_json"]: continue
            row = json.loads(rec["payload_json"])
            valid, payload_code = me._valid_payload_scope(row, q, PRODUCT, market)
            if not valid: raise ValueError("selected radish payload does not match its requested scope")
            code = str(payload_code or code or "<null>")
            if not str(q.get("cond[gds_sclsf_cd::EQ]", "")) and (day.isoformat(), market, code) in specific_scopes:
                raise ValueError(f"ambiguous_radish_scope_overlap: broad and subclass-selected rows both cover {market}/{day}/{code}; frozen core selection cannot be reconciled without changing target semantics")
            if row.get("trd_se") != "경매" or row.get("unit_nm") != "kg": continue
            amount, weight = dashboard._as_decimal(row.get("totprc")), dashboard._as_decimal(row.get("unit_tot_qty"))
            if amount is None or weight is None: continue
            name = row.get("gds_sclsf_nm")
            item = sums[(market, day, code)]
            item["amount"] += amount; item["weight"] += weight; item["rows"] += 1
            item["negativeRows"] += int(amount < 0 or weight < 0)
            item["codes"].add(code)
            if name is not None: item["names"].add(str(name))
        # Re-read the independent selected market/day signed ledger and compare exact Decimal sums,
        # including zero/negative net amounts and weights.
        _, _, selected_diagnostics = me._extract_core_market_prices(db, start, end, True)
        for offset in range((end-start).days+1):
            day = start + dt.timedelta(days=offset)
            for market in MARKETS:
                amount = sum((v["amount"] for (m, d, _), v in sums.items() if m == market and d == day), Decimal(0))
                weight = sum((v["weight"] for (m, d, _), v in sums.items() if m == market and d == day), Decimal(0))
                expected = selected_diagnostics.get((PRODUCT, day, market), {})
                if amount != expected.get("signedNetAmount", Decimal(0)) or weight != expected.get("signedNetWeight", Decimal(0)):
                    raise ValueError(f"selected radish signed market/day totals mismatch: {market}/{day}")
        db.rollback()
    finally: db.close()
    # Require every day/market to have a usable complete/empty broad scope.
    if not scopes: raise ValueError("no selected radish subclass scopes found")
    states_count = defaultdict(int)
    for *_, state, __, ___ in scopes: states_count[state] += 1
    # Reconcile signed aggregate against frozen core. Unknown classes are never dropped.
    markets = {m: defaultdict(dict) for m in MARKETS}
    audit = []
    for (market, day, code), value in sorted(sums.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        markets[market][code][day] = {**value, "amount": value["amount"], "weight": value["weight"], "codes": value["codes"], "names": value["names"]}
        audit.append({"market": market, "date": day.isoformat(), "code": code, "observedNames": sorted(value["names"]), "knownName": KNOWN_NAMES.get(code), "nameConflict": len(value["names"]) > 1 or (code in KNOWN_NAMES and bool(value["names"]) and KNOWN_NAMES[code] not in value["names"]), "signedAmount": str(value["amount"]), "signedNetKg": str(value["weight"]), "rowCount": value["rows"], "negativeAdjustmentRows": value["negativeRows"]})
    # Preserve missing and empty days: a missing core observation is acceptable only
    # when both independently reconciled market ledgers sum to no positive target.
    checked = (end - start).days + 1
    by_day = defaultdict(lambda: [Decimal(0), Decimal(0)])
    for (market, day, code), row in sums.items():
        by_day[day][0] += row["amount"]; by_day[day][1] += row["weight"]
    for offset in range((end-start).days+1):
        day = start + dt.timedelta(days=offset)
        amount, weight = by_day.get(day, (Decimal(0), Decimal(0)))
        core_row = core[PRODUCT].get(day)
        if weight > 0 and amount > 0:
            price = float(amount / weight)
            if core_row is None or not math.isclose(price, core_row[0], rel_tol=1e-10, abs_tol=1e-7) or not math.isclose(float(weight), core_row[1], rel_tol=1e-10, abs_tol=1e-7):
                raise ValueError(f"all-class radish signed aggregate mismatch for {day}")
        elif core_row is not None:
            raise ValueError(f"unexpected frozen radish target without positive signed aggregate: {day}")
    broad_audit = dict(broad_audit)
    selected_state_counts = broad_audit.get("selectedStates")
    if isinstance(selected_state_counts, dict):
        # JSON object keys must be strings; retain the missing-state zero/nonzero count.
        broad_audit["selectedStates"] = {("missing" if state is None else str(state)): count
                                           for state, count in selected_state_counts.items()}
    return markets, {"selectedRequestHashes": req_hashes, "selectedQueryStates": dict(states_count), "selectedScopeCount": len(scopes), "selectedScopeSha256": sha256(json.dumps(sorted(scopes), separators=(",", ":")).encode()), "broadScopeAudit": broad_audit, "codeNameAudit": audit, "reconciliation": {"checkedMarketDates": checked * len(MARKETS), "positiveSignedTotals": len(by_day), "scope": "selected auction/kg rows across every radish subclass; exact signed Decimal amount and net kg per market/date"}}


def build(prices, markets, forecasts, ridge_report, recompose_report, *, hashes=None):
    """Fit monthly causal candidate and evaluate on exact frozen prediction keys."""
    series = dh._observations(prices)[PRODUCT]
    _, _, persistence = dh._forecast_maps(forecasts)
    ridge, ridge_actual = rw._direct_ridge_reference(ridge_report)
    for (p, h, target), expected in ridge_actual.items():
        if p == PRODUCT and target in series and not math.isclose(series[target][0], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"pinned direct Ridge actual mismatch: {p}/{h}/{target}")
    recomposed = {}
    for row in recompose_report.get("pairedRows", []):
        key = (row["product"], int(row["horizon"]), _day(row["target"]))
        if key in recomposed: raise ValueError(f"duplicate market-recompose key {key}")
        origin = _day(row["origin"])
        if origin != key[2] - dt.timedelta(days=key[1]): raise ValueError(f"market-recompose origin mismatch {key}")
        if key[0] == PRODUCT and key[2] in series:
            if row.get("actual") is None or not math.isclose(float(row["actual"]), series[key[2]][0], rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"market-recompose actual differs from frozen target: {key}")
        recomposed[key] = _finite(row.get("candidatePrediction"))
    cats = _source_series(_combine_markets(markets))
    sample, audits, feature_cache, base_cache = {}, {}, {}, {}
    import numpy as np
    from threadpoolctl import threadpool_limits
    for h in HORIZONS:
        training = []
        for target, (actual, _) in sorted(series.items()):
            origin = target - dt.timedelta(days=h)
            if origin < dt.date(2014, 1, 1): continue
            base_key = (h, origin, target)
            if base_key not in base_cache: base_cache[base_key] = modeling._feature_row(series, origin, target, h, {})
            base = base_cache[base_key]
            if base is None: continue
            if origin not in feature_cache: feature_cache[origin] = _feature_values(cats, series, origin)
            extra, _ = feature_cache[origin]
            training.append((target, origin, tuple(base[1]) + extra, math.log(actual / series[base[0]][0])))
        month_rows = defaultdict(list)
        for target in series:
            origin = target - dt.timedelta(days=h)
            if EVAL_START <= target <= EVAL_END and origin >= EVAL_START:
                base_key = (h, origin, target)
                if base_key not in base_cache: base_cache[base_key] = modeling._feature_row(series, origin, target, h, {})
                base = base_cache[base_key]
                if origin not in feature_cache: feature_cache[origin] = _feature_values(cats, series, origin)
                extra, audit = feature_cache[origin]
                vector = tuple(base[1]) + extra if base is not None else None
                month_rows[(origin.year, origin.month)].append((target, origin, vector, base[0] if base else None, audit))
        for ym, rows in sorted(month_rows.items()):
            fit_origin = min(r[1] for r in rows)
            train = [r for r in training if r[0] < fit_origin]
            distinct = len({r[0] for r in train})
            key = f"{h}/{ym[0]:04d}-{ym[1]:02d}"
            model = None; status = "insufficient_training" if distinct < MIN_TRAIN else "fit_failed"
            if distinct >= MIN_TRAIN:
                try:
                    with threadpool_limits(limits=1):
                        model = modeling._new_estimator("ridge")
                        model.fit(np.asarray([r[2] for r in train]), np.asarray([r[3] for r in train]))
                    status = "fitted"
                except Exception as exc:
                    audits[key] = {"status": "fit_failed", "fitOrigin": fit_origin.isoformat(), "distinctTargets": distinct, "error": str(exc)[:200]}
            if status != "fitted": audits[key] = {"status": status, "fitOrigin": fit_origin.isoformat(), "distinctTargets": distinct}
            for target, origin, vector, source, feature_audit in rows:
                base_pred = ridge.get((PRODUCT, h, target))
                candidate = None; fallback = False; why = None
                if model is not None and vector is not None and source is not None:
                    try:
                        with threadpool_limits(limits=1): delta = float(model.predict(np.asarray([vector]))[0])
                        if math.isfinite(delta):
                            candidate, _ = modeling._inverse_log_ratio(series[source][0], delta)
                            if not math.isfinite(candidate) or candidate <= 0: candidate = None
                    except Exception: candidate = None
                if candidate is None:
                    candidate = base_pred; fallback = True
                    why = status if model is None else "base_features_missing" if vector is None else "invalid_candidate"
                # Candidate issuance is strictly tied to the pinned fallback availability.
                if base_pred is None: candidate = None
                sample[(h, target)] = {"horizon": h, "origin": origin, "target": target, "actual": series[target][0], "candidate": candidate, "directRidge": base_pred, "persistence": persistence[PRODUCT].get((target, h)), "marketRecompose": recomposed.get((PRODUCT, h, target)), "fallback": fallback, "fallbackReason": why, "featureAvailability": feature_audit}
                audits.setdefault(key, {"status": status, "fitOrigin": fit_origin.isoformat(), "distinctTargets": distinct})
    comparisons, coverage = [], {}
    for h in HORIZONS:
        eligible = {t for t in series if EVAL_START <= t <= EVAL_END and t-dt.timedelta(days=h) >= EVAL_START}
        records = [r for (hh, _), r in sample.items() if hh == h]
        issued = {"candidate": {r["target"] for r in records if r["candidate"] is not None},
                  "directRidge": {t for (p, hh, t) in ridge if p == PRODUCT and hh == h and t in eligible},
                  "persistence": {t for (t, hh) in persistence[PRODUCT] if hh == h and t in eligible},
                  "marketRecompose": {t for (p, hh, t), v in recomposed.items() if p == PRODUCT and hh == h and t in eligible and v is not None}}
        coverage[str(h)] = {"eligible": len(eligible), **{f"{name}Issued": len(issued[name]) for name in issued}, "allArmPaired": len(issued["candidate"] & issued["directRidge"] & issued["persistence"] & issued["marketRecompose"]), "candidateCoverageLossDates": {c: sorted(t.isoformat() for t in issued[c] - issued["candidate"]) for c in COMPARATORS}, "fallbackCount": sum(r["fallback"] for r in records), "fallbackReasons": dict((x, sum(r["fallbackReason"] == x for r in records)) for x in sorted({r["fallbackReason"] for r in records if r["fallbackReason"]}))}
        for comp in COMPARATORS:
            rows = [r for r in records if r["candidate"] is not None and r[comp] is not None]
            a = [r["actual"] for r in rows]
            for loss in LOSSES:
                diff = {}
                for r in rows:
                    val = abs(r["actual"]-r["candidate"]) - abs(r["actual"]-r[comp])
                    if loss == "mape": val *= 100/r["actual"]
                    diff[r["target"]] = val
                comparisons.append({"horizon": h, "comparator": comp, "loss": loss, "n": len(rows), "candidateMetric": dh._metrics(a, [r["candidate"] for r in rows])[loss] if rows else None, "comparatorMetric": dh._metrics(a, [r[comp] for r in rows])[loss] if rows else None, "dm": {str(b): dh._dm(diff, b) for b in BANDWIDTHS}})
    if len(comparisons) != 60: raise AssertionError("frozen 60-test family invariant failed")
    for b in BANDWIDTHS: dh._holm([x["dm"][str(b)] for x in comparisons])
    paired = [{**r, "origin": r["origin"].isoformat(), "target": r["target"].isoformat()} for r in sample.values() if all(r.get(name) is not None for name in ("candidate", *COMPARATORS))]
    joint, strata = [], []
    season = lambda d: "winter" if d.month in (12,1,2) else "spring" if d.month in (3,4,5) else "summer" if d.month in (6,7,8) else "autumn"
    for h in HORIZONS:
        required = [x for x in comparisons if x["horizon"] == h]
        enough = all(x["n"] >= 30 and x["dm"]["9"]["status"] == "tested" for x in required)
        metrics_lower = all(x["candidateMetric"] is not None and x["comparatorMetric"] is not None and x["candidateMetric"] < x["comparatorMetric"] for x in required)
        all_significant = all(x["dm"]["9"].get("holmSignificant") and x["dm"]["9"].get("meanLossDifference", 0) < 0 for x in required)
        coverage_ok = all(not coverage[str(h)]["candidateCoverageLossDates"][c] for c in COMPARATORS)
        adverse = any(x["dm"]["9"].get("holmSignificant") and x["dm"]["9"].get("meanLossDifference", 0) > 0 for x in required)
        conclusion = "inference_unavailable" if not enough else "improvement" if metrics_lower and all_significant and coverage_ok else "worsening_or_mixed" if adverse else "inconclusive"
        joint.append({"horizon": h, "conclusion": conclusion, "inferenceAvailable": enough, "allSixHac9HolmTestsImprove": all_significant, "allSixPairedMetricsImprove": metrics_lower, "coverageNondeclining": coverage_ok})
        for kind, fn in (("year", lambda d: str(d.year)), ("season", season)):
            for comp in COMPARATORS:
                groups = defaultdict(list)
                for r in sample.values():
                    if r["horizon"] == h and r["candidate"] is not None and r[comp] is not None:
                        groups[fn(r["target"])].append(r)
                for label, group in sorted(groups.items()):
                    strata.append({"horizon": h, "comparator": comp, "stratum": kind, "value": label, "n": len(group), "candidate": dh._metrics([r["actual"] for r in group], [r["candidate"] for r in group]), comp: dh._metrics([r["actual"] for r in group], [r[comp] for r in group])})
    macro = {}
    for name in ("candidate", *COMPARATORS):
        vals = [dh._metrics([r["actual"] for r in paired if r["horizon"] == h], [r[name] for r in paired if r["horizon"] == h])["mape"] for h in HORIZONS]
        macro[name] = sum(vals)/len(vals) if vals and all(x is not None for x in vals) else None
    feature_availability = {}
    for cat in CATEGORIES:
        feature_availability[cat] = {
            **{f"share{window}": {"available": sum(bool(r["featureAvailability"].get(f"{cat}/share{window}", {}).get("denominatorAvailable")) for r in sample.values()), "eligible": len(sample)} for window in (7, 28)},
            "relativePrice": {"available": sum(bool(r["featureAvailability"].get(f"{cat}/relativePrice", {}).get("available")) for r in sample.values()), "eligible": len(sample)},
        }
    return {"schemaVersion": 1, "status": "complete", "protocol": PROTOCOL, "hashes": hashes or {}, "fitAudits": audits, "pairedRows": paired, "comparisons": comparisons, "coverage": coverage, "jointConclusions": joint, "strata": strata, "macroMetricsEqualHorizonMape": macro, "featureAvailability": feature_availability, "categoryAudit": {"codeNameAudit": "see inputs", "categoryRows": {c: len(v) for c, v in cats.items()}}, "limitations": PROTOCOL["limitations"]}


def run(*, prices_path=INPUT_PRICES, predictions_path=INPUT_PREDICTIONS, ridge_path=INPUT_RIDGE, recompose_path=INPUT_RECOMPOSE, db_path=DB_PATH):
    paths = {"pricesSha256": prices_path, "predictionsSha256": predictions_path, "directRidgeSha256": ridge_path, "marketRecomposeSha256": recompose_path}
    raw_bytes = {k: p.read_bytes() for k, p in paths.items()}
    hashes = {k: sha256(v) for k, v in raw_bytes.items()}
    if hashes != EXPECTED: raise ValueError("input SHA-256 does not match frozen protocol")
    vals = {k: json.loads(v) for k, v in raw_bytes.items()}
    core = dh._observations(vals["pricesSha256"])
    markets, provenance = _extract_market_data(db_path, vals["pricesSha256"], core)
    report = build(vals["pricesSha256"], markets, vals["predictionsSha256"], vals["directRidgeSha256"], vals["marketRecomposeSha256"], hashes=hashes)
    report["inputs"] = provenance
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=OUTPUT)
    a = p.parse_args(argv)
    result = run()
    raw = (json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")) + "\n").encode()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("xb") as f: f.write(raw); f.flush()
    print(json.dumps({"output": str(a.output), "sha256": sha256(raw), "pairs": len(result["pairedRows"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
