"""Frozen retrospective garlic subclass mix experiment (read-only inputs)."""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import hashlib
import json
import math
import os
import sqlite3
import uuid
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from . import dashboard, direct_horizon as dh, market_extension as me, market_mix, modeling

ROOT = Path(__file__).resolve().parents[1]
PRODUCT = "garlic"
CLASSES = ("06", "16", "41", "42", "43")
MIN_TRAIN = 90
MAX_AGE = 14
HORIZONS = tuple(range(1, 11))
EVAL_START, EVAL_END = dt.date(2023, 1, 1), dt.date(2026, 9, 27)
OUTPUT = ROOT / "outputs/experiments/garlic-mix-20261007/report.json"
DB_PATH = ROOT / "data/agri.sqlite"
PROTOCOL = {
    "id": "garlic-subclass-mix-2026-10-07", "product": PRODUCT, "subclasses": list(CLASSES),
    "evaluation": [EVAL_START.isoformat(), EVAL_END.isoformat()], "horizons": list(HORIZONS),
    "features": "per subclass signed net kg shares over prior 7/28 calendar days; last positive subclass price / same-day positive aggregate garlic price within 14 days (log), age and presence; negative row-count ratios over 7/28 days",
    "shareMissing": "nonpositive signed aggregate denominator => zero plus explicit missing flag; signed source totals retained",
    "fit": "monthly expanding StandardScaler + Ridge(alpha=10); labels strictly before fit origin; 90 distinct target dates minimum; pinned direct Ridge fallback only",
    "comparisons": ["directRidge", "persistence"], "DM": {"twoSided": True, "calendarBartlettHacDays": [9, 28], "fixedHolmFamilySize": 40, "alpha": .05, "minimumPairs": 30},
    "availability": "selected revised historical responses; first publication time unknown"
}


def _sha(b): return hashlib.sha256(b).hexdigest()


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _decode_payload(value):
    """Return a mapping payload or None for empty, malformed, or JSON null responses."""
    try: payload=json.loads(value) if value else None
    except (TypeError, json.JSONDecodeError): return None
    return payload if isinstance(payload,dict) else None


def _market_subclass_totals(subclasses, market, day):
    """Sum without inserting absent market/date/class rows."""
    selected=subclasses.get(market,{}).get(day,{})
    amount=sum((selected.get(c,{}).get("amount",Decimal(0)) for c in CLASSES),Decimal(0))
    weight=sum((selected.get(c,{}).get("weight",Decimal(0)) for c in CLASSES),Decimal(0))
    return amount,weight,selected


def _assert_market_reconciled(subclasses, market, day, expected_amount, expected_weight):
    amount,weight,_=_market_subclass_totals(subclasses,market,day)
    if (amount,weight)!=(expected_amount,expected_weight):
        raise ValueError(f"garlic subclass signed totals do not reconcile for {market}/{day}")
    return amount,weight


def _candidate_subset(candidate, reference):
    return {key:value for key,value in candidate.items() if key in reference}


def _output_may_not_replace(output, *inputs):
    return output.resolve() not in {p.resolve() for p in inputs}


def _garlic_feature_index(rows, by_day=None):
    by_day=by_day if by_day is not None else _by_day(rows)
    aggregate_prices={}
    for day, day_rows in by_day.items():
        amount=sum((v["amount"] for v in day_rows.values()),Decimal(0)); weight=sum((v["weight"] for v in day_rows.values()),Decimal(0))
        if amount>0 and weight>0: aggregate_prices[day]=amount/weight
    return {"byDay":by_day,"classDates":{code:sorted(rows.get(code,{})) for code in CLASSES},"allDates":sorted(by_day),"aggregatePrices":aggregate_prices}


def _garlic_features(rows, origin, by_day=None, index=None):
    """Return 5 × (7d share, valid, 28d share, valid, log ratio, age, present), then negative ratios."""
    out = []
    index=index if index is not None else _garlic_feature_index(rows,by_day)
    by_day=index["byDay"]; dates_by_class=index["classDates"]; all_dates=index["allDates"]; aggregate_prices=index["aggregatePrices"]
    for code in CLASSES:
        for width in (7, 28):
            lo = origin - dt.timedelta(days=width)
            lo_idx=bisect.bisect_left(dates_by_class[code],lo); hi_idx=bisect.bisect_left(dates_by_class[code],origin)
            subset_days=dates_by_class[code][lo_idx:hi_idx]
            subset = [rows[code][d] for d in subset_days]
            numerator = sum((v["weight"] for v in subset), Decimal(0))
            day_lo=bisect.bisect_left(all_dates,lo); day_hi=bisect.bisect_left(all_dates,origin)
            denom = sum((sum((v["weight"] for v in by_day[d].values()),Decimal(0)) for d in all_dates[day_lo:day_hi]),Decimal(0))
            out.extend((float(numerator / denom) if denom > 0 else 0., float(denom > 0)))
        class_dates=dates_by_class[code]; index=bisect.bisect_right(class_dates,origin)-1; selected=None
        while index >= 0 and (origin-class_dates[index]).days <= MAX_AGE:
            d=class_dates[index]; value=rows[code][d]
            if value["price"] > 0 and d in aggregate_prices:
                selected=(d,value,aggregate_prices[d]); break
            index-=1
        if selected:
            d,v,aggregate=selected
            out.extend((math.log(float(v["price"] / aggregate)), float((origin-d).days), 1.))
        else: out.extend((0., 15., 0.))
    for width in (7, 28):
        lo = origin - dt.timedelta(days=width)
        lo_idx=bisect.bisect_left(all_dates,lo); hi_idx=bisect.bisect_left(all_dates,origin)
        records = [r for d in all_dates[lo_idx:hi_idx] for r in by_day[d].values()]
        n = sum(r["rowCount"] for r in records)
        neg = sum(r["negativeRows"] for r in records)
        out.append(neg / n if n else 0.)
    return out


def _by_day(rows):
    result = defaultdict(dict)
    for code, dates in rows.items():
        for day, value in dates.items(): result[day][code] = value
    return result


def _joint_conclusion(tests, candidate_metric_better, coverage_nondeclining):
    inferable = len(tests) == 4 and all(x["n"] >= 30 and x["dm"]["9"].get("status") in ("available", "tested") for x in tests)
    significant_better = inferable and all(x["dm"]["9"].get("holmSignificant", x["dm"]["9"].get("holmSignificantAt005", False)) and x["dm"]["9"].get("meanLossDifference", 0) < 0 for x in tests)
    significant_worse = any(x["n"] >= 30 and x["dm"]["9"].get("holmSignificant", x["dm"]["9"].get("holmSignificantAt005", False)) and x["dm"]["9"].get("meanLossDifference", 0) > 0 for x in tests)
    conclusion = "inference_unavailable" if not inferable else "improvement" if candidate_metric_better and significant_better and coverage_nondeclining else "worsening" if significant_worse else "inconclusive"
    return {"conclusion": conclusion, "inferenceAvailable": inferable, "allFourHac9HolmTestsImprove": significant_better,
            "pairedMetricsImprove": candidate_metric_better, "coverageNondeclining": coverage_nondeclining}


def extract_market_data(db_path, raw, core):
    """Extract selected response rows, preserving signed decimal subclass sums and audit hashes."""
    uri = "file:" + str(Path(db_path).resolve()) + "?mode=ro"
    db = sqlite3.connect(uri, uri=True); db.row_factory = sqlite3.Row; db.execute("PRAGMA query_only=ON")
    rows = defaultdict(lambda: defaultdict(lambda: {"amount": Decimal(0), "weight": Decimal(0), "rowCount": 0, "negativeRows": 0}))
    by_market = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"amount": Decimal(0), "weight": Decimal(0), "rowCount": 0, "negativeRows": 0})))
    selected_ids = set(); scope_manifest = []; selected_states = defaultdict(int)
    start = dt.date.fromisoformat(raw["meta"]["contextStart"]); end = dt.date.fromisoformat(raw["meta"]["contextEnd"])
    try:
        from . import coverage
        db.create_function("coverage_task_key", 3, coverage._request_task_key, deterministic=True)
        db.execute("BEGIN")
        cte = me._scoped_selection_cte(start, end, tuple(dashboard.MARKETS))
        scopes = list(db.execute(cte + "SELECT p.selected_job_id,p.selected_epoch_id,p.task_key,p.query_json,p.selected_state,p.latest_state FROM picked p WHERE p.dataset='prices_origin'"))
        specific = set()
        for job, epoch, task, qtext, state, latest_state in scopes:
            try: q=json.loads(qtext)
            except (TypeError, json.JSONDecodeError): continue
            dims=coverage.price_dimensions(q,"prices_origin")
            if not dims or dims[2] != PRODUCT: continue
            selected_states[str(state or "missing")] += 1
            if me._selected_specific_scope(q,state): specific.add(me._selected_specific_scope(q,state))
            scope_manifest.append({"jobId":job,"epochId":epoch,"taskKey":task,"query":q,"selectedState":state,"latestState":latest_state})
        sql = cte + dashboard.PRICE_SQL[len(coverage.SELECTION_CTE):]
        for rec in db.execute(sql, (start.isoformat(), end.isoformat(), *dashboard.MARKETS.keys())):
            try: q=json.loads(rec["query_json"])
            except (TypeError, json.JSONDecodeError): continue
            if dashboard._query_product(q) != PRODUCT: continue
            if rec["selected_state"] in ("complete","empty") and rec["request_id"] is not None:
                selected_ids.add(int(rec["request_id"]))
            payload=_decode_payload(rec["payload_json"])
            if payload is None: continue
            product=dashboard._query_product(q); day=dashboard._parse_date(q.get("cond[trd_clcln_ymd::EQ]")); market=str(q.get("cond[whsl_mrkt_cd::EQ]",""))
            code=str(payload.get("gds_sclsf_cd", q.get("cond[gds_sclsf_cd::EQ]", "")))
            if product != PRODUCT or market not in dashboard.MARKETS or day is None or code not in CLASSES or rec["selected_state"] not in ("complete","empty"): continue
            if not me._valid_payload_scope(payload,q,product,market)[0] or me._overlap_shadowed(product,day.isoformat(),market,str(q.get("cond[gds_sclsf_cd::EQ]","")),code,specific): continue
            if payload.get("trd_se") != "경매" or payload.get("unit_nm") != "kg": continue
            amount=dashboard._as_decimal(payload.get("totprc")); weight=dashboard._as_decimal(payload.get("unit_tot_qty"))
            if amount is None or weight is None: continue
            for target in (rows[code][day], by_market[market][day][code]):
                target["amount"]+=amount; target["weight"]+=weight; target["rowCount"]+=1
                target["negativeRows"] += int(weight < 0)
        hashes={str(r[0]):r[1] for rid in sorted(selected_ids) for r in db.execute("SELECT request_id,response_sha256 FROM source_requests WHERE request_id=?",(rid,)) if r[1]}
        # Reconcile under the same read snapshot used for row extraction.
        broad, provenance, diag = me._extract_core_market_prices(db,start,end,True)
        states, broadprov=me._core_broad_query_states(db,start)
        check=market_mix._reconcile(raw,core,states,diag,end)
        for code, dates in rows.items():
            for d, value in dates.items():
                value["price"] = value["amount"] / value["weight"] if value["weight"] > 0 and value["amount"] > 0 else Decimal(0)
        audit_totals = {}
        all_keys = {(PRODUCT,d,m) for d in set(day for market_rows in by_market.values() for day in market_rows) for m in dashboard.MARKETS}
        all_keys.update(k for k in diag if k[0] == PRODUCT)
        for _product, d, market in sorted(all_keys):
            expected = diag.get((PRODUCT,d,market), {})
            expect_a=expected.get("signedNetAmount",Decimal(0)); expect_w=expected.get("signedNetWeight",Decimal(0))
            total_a,total_w,subclasses=_market_subclass_totals(by_market,market,d)
            _assert_market_reconciled(by_market,market,d,expect_a,expect_w)
            audit_totals[f"{market}/{d.isoformat()}"]={"subclasses":{c:{"signedAmount":str(subclasses[c]["amount"]),"signedNetKg":str(subclasses[c]["weight"]),"rowCount":subclasses[c]["rowCount"],"negativeRows":subclasses[c]["negativeRows"]} for c in CLASSES if c in subclasses},"sumSignedAmount":str(total_a),"sumSignedNetKg":str(total_w),"expectedSignedAmount":str(expect_a),"expectedSignedNetKg":str(expect_w)}
        db.rollback()
    finally:
        if db.in_transaction: db.rollback()
        db.close()
    return rows, {"selectedRequestHashes":hashes,"selectedScopesSha256":_sha(json.dumps(scope_manifest,sort_keys=True,separators=(",",":")).encode()),"selectedScopeStates":dict(sorted(selected_states.items())),"selectedScopes":scope_manifest,"selectedRequestCount":len(hashes),"selectedQueryStateCounts":provenance.get("selectedQueryStates",{}),"broadScopesSha256":broadprov["scopesSha256"],"reconciliation":check,"signedSubclassTotals":audit_totals}


def build(raw, subclass_rows, forecasts, ridge_report, *, hashes=None):
    """Build walk-forward candidate and paired 40-test evidence from frozen snapshots."""
    core=dh._observations(raw); series=core[PRODUCT]
    _,_,persistence=dh._forecast_maps(forecasts); ridge, ridge_actual=__import__("agri_data.recent_window",fromlist=["_direct_ridge_reference"])._direct_ridge_reference(ridge_report)
    import numpy as np
    from threadpoolctl import threadpool_limits
    candidate={}; audits={}; feature= set(); cache={}; subclass_by_day=_by_day(subclass_rows); subclass_index=_garlic_feature_index(subclass_rows,subclass_by_day); origin_features={}
    for h in HORIZONS:
        samples=[]
        for target,(actual,_) in series.items():
            origin=target-dt.timedelta(days=h)
            if origin < dt.date(2014,1,1): continue
            base=modeling._feature_row(series,origin,target,h,cache)
            if base is None: continue
            feature.add((h,target))
            if origin not in origin_features: origin_features[origin]=tuple(_garlic_features(subclass_rows,origin,index=subclass_index))
            samples.append((target,origin,actual,tuple(base[1])+origin_features[origin],math.log(actual/series[base[0]][0])))
        samples.sort(); smap={(x[0],x[1]):x for x in samples}; months=defaultdict(list)
        for t in series:
            o=t-dt.timedelta(days=h)
            if EVAL_START <= t <= EVAL_END and o >= EVAL_START: months[(o.year,o.month)].append((t,o))
        for ym, targets in sorted(months.items()):
            fit_origin=min(o for _,o in targets); train=[r for r in samples if r[0] < fit_origin]; n=len({r[0] for r in train}); audit_key=f"{h}/{ym[0]:04d}-{ym[1]:02d}"
            if n<MIN_TRAIN: audits[audit_key]={"fitOrigin":fit_origin.isoformat(),"distinctTargets":n,"status":"insufficient_training"}; continue
            valid=[smap[x] for x in targets if x in smap]
            try:
                with threadpool_limits(limits=1):
                    model=modeling._new_estimator("ridge"); model.fit(np.asarray([r[3] for r in train]),np.asarray([r[4] for r in train])); preds=model.predict(np.asarray([r[3] for r in valid])) if valid else []
                for r,delta in zip(valid,preds):
                    src=modeling._feature_row(series,r[1],r[0],h,cache)[0]; pred,_=modeling._inverse_log_ratio(series[src][0],float(delta))
                    key=(PRODUCT,h,r[0])
                    if math.isfinite(pred) and pred>0 and key in ridge: candidate[key]=pred
                audits[audit_key]={"fitOrigin":fit_origin.isoformat(),"distinctTargets":n,"trainTargetEnd":max(r[0] for r in train).isoformat(),"status":"fitted"}
            except Exception as e: audits[audit_key]={"fitOrigin":fit_origin.isoformat(),"distinctTargets":n,"status":"fit_failed","errorType":type(e).__name__,"error":str(e)[:200]}
    # The frozen direct-Ridge reference is the required fallback for every eligible issuance.
    # Candidate issuance can never extend beyond the pinned direct-Ridge issue set.
    candidate = _candidate_subset(candidate,ridge)
    fallback_count = 0
    for h in HORIZONS:
        for target in series:
            key=(PRODUCT,h,target)
            if EVAL_START <= target <= EVAL_END and target-dt.timedelta(days=h) >= EVAL_START and key not in candidate and key in ridge:
                candidate[key]=ridge[key]; fallback_count += 1
    eligible={h:{t for t in series if EVAL_START<=t<=EVAL_END and t-dt.timedelta(days=h)>=EVAL_START} for h in HORIZONS}
    pairs=[]; comparisons=[]; coverage={}
    for h in HORIZONS:
        common=sorted(t for t in eligible[h] if (PRODUCT,h,t) in candidate and (PRODUCT,h,t) in ridge and (t,h) in persistence[PRODUCT])
        for t in common:
            actual=series[t][0]
            if (PRODUCT,h,t) in ridge_actual and not math.isclose(actual,ridge_actual[(PRODUCT,h,t)],rel_tol=1e-12): raise ValueError("pinned direct Ridge actual mismatch")
            pairs.append({"product":PRODUCT,"horizon":h,"target":t.isoformat(),"origin":(t-dt.timedelta(days=h)).isoformat(),"actual":actual,"garlicMix":candidate[(PRODUCT,h,t)],"directRidge":ridge[(PRODUCT,h,t)],"persistence":persistence[PRODUCT][(t,h)]})
        rows=[x for x in pairs if x["horizon"]==h]
        candidate_issued={t for t in eligible[h] if (PRODUCT,h,t) in candidate}
        direct_issued={t for t in eligible[h] if (PRODUCT,h,t) in ridge}
        persistence_issued={t for t in eligible[h] if (t,h) in persistence[PRODUCT]}
        coverage[str(h)]={"eligible":len(eligible[h]),"candidateIssued":len(candidate_issued),"directRidgeIssued":len(direct_issued),"persistenceIssued":len(persistence_issued),"commonPairs":len(rows),"dates":[x["target"] for x in rows],"candidateOutsideDirectRidgeDates":sorted(t.isoformat() for t in candidate_issued-direct_issued),"coverageLossDates":{"directRidge":sorted(t.isoformat() for t in direct_issued-candidate_issued),"persistence":sorted(t.isoformat() for t in persistence_issued-candidate_issued)}}
        for comp in ("directRidge","persistence"):
            for loss in ("mae","mape"):
                diff={dt.date.fromisoformat(x["target"]):(abs(x["actual"]-x["garlicMix"])-abs(x["actual"]-x[comp])) if loss=="mae" else 100*(abs(x["actual"]-x["garlicMix"])-abs(x["actual"]-x[comp]))/x["actual"] for x in rows}
                a=[x["actual"] for x in rows]; cp=[x["garlicMix"] for x in rows]; bp=[x[comp] for x in rows]
                comparisons.append({"horizon":h,"comparator":comp,"loss":loss,"n":len(rows),"candidateMetric":dh._metrics(a,cp)[loss] if rows else None,"comparatorMetric":dh._metrics(a,bp)[loss] if rows else None,"dm":{"9":dh._dm(diff,9),"28":dh._dm(diff,28)}})
    for bw in (9,28): dh._holm([x["dm"][str(bw)] for x in comparisons])
    joint=[]
    for h in HORIZONS:
        htests=[x for x in comparisons if x["horizon"]==h]
        metric_better=all(x["candidateMetric"] is not None and x["comparatorMetric"] is not None and x["candidateMetric"] < x["comparatorMetric"] for x in htests)
        joint.append({"horizon":h,**_joint_conclusion(htests,metric_better,not coverage[str(h)]["coverageLossDates"]["persistence"] and not coverage[str(h)]["coverageLossDates"]["directRidge"])})
    byyear=[]
    for h in HORIZONS:
        rows=[x for x in pairs if x["horizon"]==h]
        for field,fn in (("year",lambda d:str(d.year)),("season",lambda d:"winter" if d.month in (12,1,2) else "spring" if d.month in (3,4,5) else "summer" if d.month in (6,7,8) else "autumn")):
            groups=defaultdict(list)
            for x in rows: groups[fn(dt.date.fromisoformat(x["target"]))].append(x)
            for label,group in sorted(groups.items()): byyear.append({"horizon":h,"stratum":field,"value":label,"n":len(group),"garlicMix":dh._metrics([x["actual"] for x in group],[x["garlicMix"] for x in group]),"directRidge":dh._metrics([x["actual"] for x in group],[x["directRidge"] for x in group]),"persistence":dh._metrics([x["actual"] for x in group],[x["persistence"] for x in group])})
    return {"schemaVersion":1,"status":"complete","protocol":PROTOCOL,"hashes":hashes or {},"fitAudits":audits,"fallbackToDirectRidgeCount":fallback_count,"pairedRows":pairs,"comparisons":comparisons,"coverage":coverage,"jointConclusions":joint,"strata":byyear,"macroMetrics":{"garlicMixEqualHorizonMape":sum(dh._metrics([x["actual"] for x in pairs if x["horizon"]==h],[x["garlicMix"] for x in pairs if x["horizon"]==h])["mape"] for h in HORIZONS if any(x["horizon"]==h for x in pairs))/sum(bool(any(x["horizon"]==h for x in pairs)) for h in HORIZONS) if pairs else None},"limitations":PROTOCOL["availability"],"auxiliaryStaticBlend":{"status":"unavailable","reason":"frozen static-blend report contains aggregate results and selection summaries, not per-origin predictions required for common-pair evaluation"}}


def _atomic(path, data):
    path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as f: f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        try: temp.unlink()
        except FileNotFoundError: pass


def run(prices_path=None,predictions_path=None,ridge_path=None,db_path=DB_PATH,*,allow_nonfrozen_inputs=False):
    prices_path=prices_path or dh.INPUT_PRICES; predictions_path=predictions_path or dh.INPUT_PREDICTIONS; ridge_path=ridge_path or ROOT/"outputs/experiments/direct-horizon-20261006/report.json"
    pb=prices_path.read_bytes(); fb=predictions_path.read_bytes(); rb=ridge_path.read_bytes()
    hashes={"pricesSha256":_sha(pb),"predictionsSha256":_sha(fb),"directRidgeReportSha256":_sha(rb)}
    expected={"pricesSha256":"f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8","predictionsSha256":"c41ed5d8799306f151b1c5135399a606982f12623bf709a4438969d8cc389113","directRidgeReportSha256":"a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"}
    if not allow_nonfrozen_inputs and hashes!=expected: raise ValueError("input SHA-256 does not match frozen garlic mix protocol")
    raw=json.loads(pb); forecasts=json.loads(fb); ridge=json.loads(rb); core=dh._observations(raw)
    subclass_rows,prov=extract_market_data(db_path,raw,core)
    result=build(raw,subclass_rows,forecasts,ridge,hashes=hashes); result["inputs"]={**prov,"databasePath":str(db_path)}
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--prices",type=Path,default=dh.INPUT_PRICES); p.add_argument("--predictions",type=Path,default=dh.INPUT_PREDICTIONS); p.add_argument("--ridge",type=Path,default=ROOT/"outputs/experiments/direct-horizon-20261006/report.json"); p.add_argument("--db",type=Path,default=DB_PATH); p.add_argument("--output",type=Path,default=OUTPUT); p.add_argument("--allow-nonfrozen-inputs",action="store_true"); a=p.parse_args(argv)
    if not _output_may_not_replace(a.output,a.prices,a.predictions,a.ridge,a.db): p.error("output must not replace inputs or database")
    result=run(a.prices,a.predictions,a.ridge,a.db,allow_nonfrozen_inputs=a.allow_nonfrozen_inputs); _atomic(a.output,_json_bytes(result)); print(json.dumps({"output":str(a.output),"pairedRows":len(result["pairedRows"]),"hashes":result["hashes"]},ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
