"""Frozen offline Garak/Gangseo market-composition experiment (no API access)."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import sqlite3
from decimal import Decimal
from pathlib import Path
from typing import Any

from . import dashboard, direct_horizon as dh, modeling, market_extension as me, recent_window as rw

ROOT = Path(__file__).resolve().parents[1]
INPUT_PRICES = dh.INPUT_PRICES
INPUT_PREDICTIONS = dh.INPUT_PREDICTIONS
INPUT_RIDGE_REPORT = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
DB_PATH = ROOT / "data/agri.sqlite"
EXPECTED_PRICES_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
EXPECTED_PREDICTIONS_SHA256 = "c41ed5d8799306f151b1c5135399a606982f12623bf709a4438969d8cc389113"
EXPECTED_RIDGE_SHA256 = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
EXPECTED_RECENT_REPORT_SHA256 = "1032caa80628cbab54c38d9efe408339348ca9e6341b0df4e3306df64a606a1e"
MAX_AGE = 14
MIN_TRAIN = 90
HORIZONS = tuple(range(1, 11))
MARKETS = tuple(dashboard.MARKETS)
PROTOCOL = {
    "id": "garak-gangseo-market-mix-direct-hist-gb-2026-10-06",
    "featureGroups": {"perMarket": ["logPrice", "ageDays", "log1pNetKg", "available"],
        "sameDay": ["logPriceSpreadGarakMinusGangseo", "garakWeightShare", "commonAgeDays", "available"]},
    "missingSentinels": {"perMarket": [0, 15, 0, 0], "sameDay": [0, 0, 15, 0]},
    "asOfRule": "last positive selected observation on or before origin, age <= 14 calendar days",
    "sameDayRule": "latest day on or before origin with positive observations from both markets, age <= 14",
    "targets": "only frozen prices.json actual observation days; horizons 1..10 calendar days",
    "fit": "monthly expanding HistGB; first eligible origin per origin month; target < fit origin; minimum 90 distinct target days",
    "limitations": "retrospective revised data; historical first publication time unknown",
}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                        allow_nan=False) + "\n").encode("utf-8")


def _features(market: dict[str, dict[dt.date, dict[str, Any]]], origin: dt.date) -> list[float]:
    """Twelve market-mix values; unavailable sources use age sentinel 15."""
    vals: list[float] = []
    sources = {}
    for code in MARKETS:
        eligible = [d for d, row in market.get(code, {}).items() if d <= origin and
                    (origin-d).days <= MAX_AGE and row.get("price", 0) > 0 and row.get("weight", 0) > 0]
        day = max(eligible) if eligible else None
        if day is None:
            vals.extend((0., float(MAX_AGE + 1), 0., 0.))
        else:
            row = market[code][day]
            vals.extend((math.log(float(row["price"])), float((origin-day).days),
                         math.log1p(float(row["weight"])), 1.))
        sources[code] = day
    common = set(market.get(MARKETS[0], {})) & set(market.get(MARKETS[1], {}))
    common = [d for d in common if d <= origin and (origin-d).days <= MAX_AGE and
              all(market[m][d].get("price", 0) > 0 and market[m][d].get("weight", 0) > 0 for m in MARKETS)]
    day = max(common) if common else None
    if day is None:
        vals.extend((0., 0., float(MAX_AGE + 1), 0.))
    else:
        g, j = (market[m][day] for m in MARKETS)
        vals.extend((math.log(float(g["price"]))-math.log(float(j["price"])),
                     float(g["weight"])/(float(g["weight"])+float(j["weight"])),
                     float((origin-day).days), 1.))
    return vals


def _reconcile(raw: dict[str, Any], core, states, diagnostics, end: dt.date) -> dict[str, Any]:
    """Require complete/empty broad scopes and exact signed amount/weight aggregation."""
    try:
        start = dt.date.fromisoformat(raw["meta"]["contextStart"])
        frozen_end = dt.date.fromisoformat(raw["meta"]["contextEnd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("frozen prices input requires valid contextStart/contextEnd") from exc
    if frozen_end != end or start > me.REVALIDATION_TRAIN_START:
        raise ValueError("frozen price context must cover the full 2014+ training period")
    checked = positive = 0
    raw_products = {p.get("id"): p for p in raw.get("products", [])}
    for product in dashboard.PRODUCTS:
        obs_rows = {dt.date.fromisoformat(r["date"]): r for r in raw_products[product].get("observations", [])}
        for offset in range((frozen_end-start).days+1):
            day = start + dt.timedelta(days=offset)
            for market in MARKETS:
                state = states.get((product, day, market), {}).get("selectedState")
                if state not in ("complete", "empty"):
                    raise ValueError(f"incomplete broad query for {product}/{day}/{market}: {state or 'missing'}")
            amount = sum((diagnostics.get((product, day, m), {}).get("signedNetAmount", Decimal(0)) for m in MARKETS), Decimal(0))
            weight = sum((diagnostics.get((product, day, m), {}).get("signedNetWeight", Decimal(0)) for m in MARKETS), Decimal(0))
            expected = float(amount/weight) if weight > 0 and amount/weight > 0 else None
            row = core[product].get(day)
            if expected is None:
                if row is not None:
                    raise ValueError(f"snapshot/database mismatch for {product}/{day}: unexpected target observation")
            elif row is None or not math.isclose(row[0], expected, rel_tol=1e-10, abs_tol=1e-8) or not math.isclose(row[1], float(weight), rel_tol=1e-10, abs_tol=1e-8):
                raise ValueError(f"snapshot/database mismatch for {product}/{day}: signed aggregate amount/weight differs")
            source = obs_rows.get(day)
            if (source is None) != (expected is None):
                raise ValueError(f"snapshot/database mismatch for {product}/{day}: observation inclusion differs")
            if source is not None:
                contributing = sorted(m for m in MARKETS if diagnostics.get((product, day, m), {}).get("eligibleAuctionKgRows", 0) > 0)
                if source.get("marketCodes") != contributing:
                    raise ValueError(f"snapshot/database mismatch for {product}/{day}: contributing market codes differ")
            checked += 1
            positive += expected is not None
    return {"checkedProductDates": checked, "positiveTargetsReconciled": positive,
            "rule": "signed per-market amount and net kg sum; both broad states complete or empty"}


def extract_market_data(db_path: Path, raw: dict[str, Any], core):
    """Read selected prices in a read-only transaction and validate their frozen snapshot."""
    uri = "file:" + str(db_path.resolve()) + "?mode=ro"
    db = sqlite3.connect(uri, uri=True); db.row_factory = sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    try:
        db.execute("BEGIN")
        start = dt.date.fromisoformat(raw["meta"]["contextStart"])
        end = dt.date.fromisoformat(raw["meta"]["contextEnd"])
        states, broad = me._core_broad_query_states(db, start)
        prices, provenance, diag = me._extract_core_market_prices(db, start, end, True)
        check = _reconcile(raw, core, states, diag, dt.date.fromisoformat(raw["meta"]["contextEnd"]))
        db.rollback()
    finally:
        db.close()
    markets = {p: {m: {} for m in MARKETS} for p in dashboard.PRODUCTS}
    # Composition features are allowed only on frozen positive target-observation days.
    target_days = {p: set(core[p]) for p in dashboard.PRODUCTS}
    for (product, day, market), v in diag.items():
        if day not in target_days[product]:
            continue
        amount, weight = v["signedNetAmount"], v["signedNetWeight"]
        if weight > 0 and amount / weight > 0:
            markets[product][market][day] = {"price": float(amount/weight), "weight": float(weight)}
    return markets, {"broad": broad, "selected": provenance, "reconciliation": check}


def _observations(prices):
    series = dh._observations(prices)
    return series


def _issued(method: str, mapping, product: str, horizon: int, target: dt.date) -> bool:
    if method in ("marketMixHistGb", "fullHistoryHistGb", "directRidge"):
        return (product, horizon, target) in mapping
    return (target, horizon) in mapping.get(product, {})


def _value(method: str, mapping, product: str, horizon: int, target: dt.date):
    return mapping.get((product, horizon, target)) if method in ("marketMixHistGb", "fullHistoryHistGb", "directRidge") else mapping.get(product, {}).get((target, horizon))


def _cell_conclusion(tests, coverage_losses):
    available=all(x["dm"]["9"]["status"]=="tested" and x["n"]>=30 for x in tests)
    both_better=all(x["candidateMetric"] is not None and x["comparatorMetric"] is not None and x["candidateMetric"]<x["comparatorMetric"] for x in tests)
    significant=all(x["dm"]["9"].get("holmSignificant") and x["dm"]["9"].get("meanLossDifference",0)<0 for x in tests)
    coverage_ok=not any(coverage_losses)
    adverse=any(x["dm"]["9"]["status"]=="tested" and x["dm"]["9"].get("holmSignificant") and x["dm"]["9"].get("meanLossDifference",0)>0 for x in tests)
    conclusion="inference_unavailable" if not available else "improvement" if both_better and significant and coverage_ok else "worsening_or_mixed" if adverse else "inconclusive"
    return {"conclusion":conclusion,"inferenceAvailable":available,"allEightHac9HolmTestsImprove":significant,
        "allEightPairedMetricsImprove":both_better,"coverageNondeclining":coverage_ok}


def build(raw, markets, forecasts, ridge_report, full_report, *, hashes=None):
    """Construct monthly direct HistGB predictions on the common frozen sample."""
    series_by = _observations(raw)
    _, _, persistence = dh._forecast_maps(forecasts)
    monthly = rw._monthly_hist_gb(forecasts)
    ridge, ridge_actual = rw._direct_ridge_reference(ridge_report)
    # Reuse the frozen full-history arm and verify it matches the pinned report.
    full_predictions = {}
    for r in full_report.get("predictions", []):
        if r.get("arm") == "full":
            key = (r["product"], int(r["horizon"]), dt.date.fromisoformat(r["target"]))
            if key in full_predictions:
                raise ValueError(f"duplicate full-history HistGB reference row: {key}")
            full_predictions[key] = float(r["predicted"])
    # Training labels are based on the same as-of target-price source used by direct_horizon.
    candidate, audits, feature_available = {}, {}, set()
    import numpy as np
    from threadpoolctl import threadpool_limits
    for p, series in series_by.items():
        cache = {}; market = markets[p]
        for h in HORIZONS:
            samples = []
            for target, (actual, _) in series.items():
                origin = target-dt.timedelta(days=h)
                if origin < dt.date(2014, 1, 1): continue
                base = modeling._feature_row(series, origin, target, h, cache)
                if base is None: continue
                feature_available.add((p,h,target))
                samples.append((target, origin, actual, tuple(base[1])+tuple(_features(market, origin)),
                               math.log(actual/series[base[0]][0])))
            samples.sort(key=lambda r:r[0])
            months = {}
            for target in series:
                origin=target-dt.timedelta(days=h)
                if dh.EVAL_START <= target <= dh.EVAL_END and origin >= dh.EVAL_START: months.setdefault((origin.year,origin.month),[]).append((target,origin))
            smap={(r[0],r[1]):r for r in samples}
            for ym, targets in sorted(months.items()):
                fit_origin=min(o for _,o in targets)
                train=[r for r in samples if r[0] < fit_origin]
                n=len(set(r[0] for r in train))
                audit_key=f"{p}/{h}/{ym[0]:04d}-{ym[1]:02d}"
                if n < MIN_TRAIN:
                    audits[audit_key]={"fitOrigin":fit_origin.isoformat(),"distinctTargets":n,"status":"insufficient_training"}
                    continue
                valid=[smap[(t,o)] for t,o in targets if (t,o) in smap]
                try:
                    with threadpool_limits(limits=1):
                        model=modeling._new_estimator("hist_gb")
                        model.fit(np.asarray([r[3] for r in train]),np.asarray([r[4] for r in train]))
                        predicted=model.predict(np.asarray([r[3] for r in valid])) if valid else []
                except Exception as exc:
                    audits[audit_key]={"fitOrigin":fit_origin.isoformat(),"distinctTargets":n,
                        "status":"fit_failed","errorType":type(exc).__name__,"error":str(exc)[:200]}
                    continue
                for r,delta in zip(valid,predicted):
                    src=modeling._feature_row(series,r[1],r[0],h,cache)[0]
                    price,_=modeling._inverse_log_ratio(series[src][0],float(delta))
                    candidate[(p,h,r[0])] = price
                audits[audit_key]={"fitOrigin":fit_origin.isoformat(),"distinctTargets":n,"trainTargetEnd":max(r[0] for r in train).isoformat(),"validTargets":len(valid),"status":"fitted"}
    # Keep issuance sets independently for coverage; paired rows remain common to all methods.
    method_maps={"marketMixHistGb":candidate,"fullHistoryHistGb":full_predictions,
        "monthlyPooledHistGb":monthly,"directRidge":ridge,"persistence":persistence}
    eligible_by={}
    for p,series in series_by.items():
        for h in HORIZONS:
            eligible_by[(p,h)]={t for t in series if dh.EVAL_START<=t<=dh.EVAL_END and t-dt.timedelta(days=h)>=dh.EVAL_START}
    paired=[]
    for p,series in series_by.items():
        for h in HORIZONS:
            for target,(actual,_) in series.items():
                if target not in eligible_by[(p,h)]: continue
                key=(p,h,target); origin=target-dt.timedelta(days=h)
                values=(candidate.get(key),full_predictions.get(key),monthly[p].get((target,h)),ridge.get((p,h,target)),persistence[p].get((target,h)))
                if any(v is None for v in values): continue
                if (p,h,target) in ridge_actual and not math.isclose(actual,ridge_actual[(p,h,target)],rel_tol=1e-12): raise ValueError("Ridge actual differs from frozen prices")
                paired.append({"product":p,"horizon":h,"target":target.isoformat(),"origin":origin.isoformat(),"actual":actual,
                    "marketMixHistGb":values[0],"fullHistoryHistGb":values[1],"monthlyPooledHistGb":values[2],"directRidge":values[3],"persistence":values[4]})
    comparators=("fullHistoryHistGb","monthlyPooledHistGb","directRidge","persistence")
    comparisons=[]; coverage={}; cell_conclusion_inputs=[]
    for p in dashboard.PRODUCTS:
        for h in HORIZONS:
            eligible=eligible_by[(p,h)]
            issued={name:{t for t in eligible if _issued(name,m,p,h,t)} for name,m in method_maps.items()}
            cov={"eligibleObservationCount":len(eligible),"issuanceCounts":{k:len(v) for k,v in issued.items()},
                 "availabilityRates":{k:(len(v)/len(eligible) if eligible else None) for k,v in issued.items()},
                 "candidateCoverageLossDates":{c:sorted(t.isoformat() for t in issued[c]-issued["marketMixHistGb"]) for c in comparators},
                 "excludedDates":{}}
            feature_flags={"garak":{},"gangseo":{},"sameDay":{}}
            for target in sorted(eligible):
                origin=target-dt.timedelta(days=h); fv=_features(markets[p],origin)
                feature_flags["garak"][target.isoformat()]=bool(fv[3])
                feature_flags["gangseo"][target.isoformat()]=bool(fv[7])
                feature_flags["sameDay"][target.isoformat()]=bool(fv[11])
            cov["featureAvailability"]={"denominator":len(eligible),
                "availableCounts":{k:sum(v.values()) for k,v in feature_flags.items()},
                "availabilityRates":{k:(sum(v.values())/len(eligible) if eligible else None) for k,v in feature_flags.items()},
                "unavailableTargetDates":{k:[d for d,v in v.items() if not v] for k,v in feature_flags.items()}}
            for name,dates in issued.items():
                missing=eligible-dates
                cov["excludedDates"][name]={}
                for t in sorted(missing):
                    reason="reference_unavailable"
                    if name=="marketMixHistGb":
                        audit=audits.get(f"{p}/{h}/{(t-dt.timedelta(days=h)).year:04d}-{(t-dt.timedelta(days=h)).month:02d}",{})
                        reason="fit_failed" if audit.get("status")=="fit_failed" else "insufficient_training" if audit.get("status")=="insufficient_training" else "core_source_or_market_feature_unavailable" if (p,h,t) not in feature_available else "model_prediction_unavailable"
                    cov["excludedDates"][name][t.isoformat()]=[reason]
            coverage[f"{p}/{h}"]=cov
            results_for_cell=[]
            for comp in comparators:
                rows=[r for r in paired if r["product"]==p and r["horizon"]==h]
                for loss in ("mae","mape"):
                    pred=[r["marketMixHistGb"] for r in rows]; base=[r[comp] for r in rows]; actual=[r["actual"] for r in rows]
                    if loss=="mae": diff={dt.date.fromisoformat(r["target"]):abs(r["actual"]-r["marketMixHistGb"])-abs(r["actual"]-r[comp]) for r in rows}
                    else: diff={dt.date.fromisoformat(r["target"]):100*(abs(r["actual"]-r["marketMixHistGb"])-abs(r["actual"]-r[comp]))/r["actual"] for r in rows}
                    item={"product":p,"horizon":h,"comparator":comp,"loss":loss,"n":len(rows),
                        "candidateMetric":dh._metrics(actual,pred)[loss],"comparatorMetric":dh._metrics(actual,base)[loss],
                        "dm":{"9":dh._dm(diff,9),"28":dh._dm(diff,28)}}
                    comparisons.append(item);results_for_cell.append(item)
            cell_conclusion_inputs.append((p,h,results_for_cell,[cov["candidateCoverageLossDates"][c] for c in comparators]))
    for bw in (9,28):
        family=[r["dm"][str(bw)] for r in comparisons]
        if len(family)!=400: raise AssertionError("frozen Holm family must contain 400 comparisons")
        dh._holm(family)
    for key, cov in coverage.items():
        p,h=key.split("/"); rows=[r for r in paired if r["product"]==p and r["horizon"]==int(h)]
        cov["commonPairedCount"]=len(rows);cov["pairedDates"]=[r["target"] for r in rows]
    cell_conclusions=[{"product":p,"horizon":h,**_cell_conclusion(tests,losses)} for p,h,tests,losses in cell_conclusion_inputs]
    strata=[]
    for p in dashboard.PRODUCTS:
        for h in HORIZONS:
            eligible=eligible_by[(p,h)]
            for kind, fn in (("year",lambda d:d.year),("season",lambda d:"winter" if d.month in (12,1,2) else "spring" if d.month in (3,4,5) else "summer" if d.month in (6,7,8) else "autumn")):
                groups={}
                common={dt.date.fromisoformat(r["target"]):r for r in paired if r["product"]==p and r["horizon"]==h}
                for day in common: groups.setdefault(fn(day),set()).add(day)
                for label,days in sorted(groups.items()):
                    metrics={}
                    for name in method_maps:
                        actual=[common[d]["actual"] for d in days]
                        pred=[common[d][name] for d in days]
                        metrics[name]=dh._metrics(actual,pred)
                    strata.append({"product":p,"horizon":h,"stratum":kind,"value":label,
                        "fiveMethodCommonDateCount":len(days),"commonDates":[d.isoformat() for d in sorted(days)],"metrics":metrics})
    macro={}
    for method in method_maps:
        group_metrics=[]
        for p in dashboard.PRODUCTS:
            for h in HORIZONS:
                rows=[r for r in paired if r["product"]==p and r["horizon"]==h]
                group_metrics.append(dh._metrics([r["actual"] for r in rows],[r[method] for r in rows])["mape"] if rows else None)
        macro[method]=sum(group_metrics)/50 if len(group_metrics)==50 and all(x is not None for x in group_metrics) else None
    fit_summary={state:sum(1 for row in audits.values() if row.get("status")==state)
                 for state in ("fitted","fit_failed","insufficient_training")}
    return {"schemaVersion":1,"status":"complete","protocol":PROTOCOL,"hashes":hashes or {},"fitAudits":audits,
            "fitSummary":fit_summary,
            "pairedRows":paired,"comparisons":comparisons,"coverage":coverage,"strata":strata,
            "jointConclusions":cell_conclusions,"macroMetrics":macro,
            "headlineMetricScope":"five-method common target dates within each product/horizon",
            "macroMetricGroupCounts":{method:sum(any(r["product"]==p and r["horizon"]==h for r in paired)
                for p in dashboard.PRODUCTS for h in HORIZONS) for method in method_maps},
            "limitations":PROTOCOL["limitations"]}


def run(prices_path: Path = INPUT_PRICES, predictions_path: Path = INPUT_PREDICTIONS,
        ridge_path: Path = INPUT_RIDGE_REPORT, full_report_path: Path | None = None,
        db_path: Path = DB_PATH, *, allow_nonfrozen_inputs: bool = False):
    """Run only when explicitly invoked; all input checks precede DB reads and model fitting."""
    full_report_path = full_report_path or ROOT / "outputs/experiments/recent-window-20261006/report.json"
    price_bytes=prices_path.read_bytes(); prediction_bytes=predictions_path.read_bytes(); ridge_bytes=ridge_path.read_bytes(); full_bytes=full_report_path.read_bytes()
    hashes={"pricesSha256":_sha(price_bytes),"predictionsSha256":_sha(prediction_bytes),"directRidgeReportSha256":_sha(ridge_bytes),"recentWindowReportSha256":_sha(full_bytes)}
    expected={"pricesSha256":EXPECTED_PRICES_SHA256,"predictionsSha256":EXPECTED_PREDICTIONS_SHA256,"directRidgeReportSha256":EXPECTED_RIDGE_SHA256,"recentWindowReportSha256":EXPECTED_RECENT_REPORT_SHA256}
    if not allow_nonfrozen_inputs and hashes != expected: raise ValueError("input SHA-256 does not match frozen market-mix protocol")
    raw=json.loads(price_bytes); forecasts=json.loads(prediction_bytes); ridge=json.loads(ridge_bytes); full=json.loads(full_bytes)
    core=dh._observations(raw)
    market,provenance=extract_market_data(db_path,raw,core)
    result=build(raw,market,forecasts,ridge,full,hashes=hashes)
    result["inputs"]={"databasePath":str(db_path),"pricesPath":str(prices_path),"pricesSha256":hashes["pricesSha256"],
        "selectedRequestHashes":provenance["selected"]["selectedRequestHashes"],"selectedScopesSha256":provenance["selected"]["selectedScopesSha256"],
        "broadScopesSha256":provenance["broad"]["scopesSha256"],"reconciliation":provenance["reconciliation"]}
    return result
