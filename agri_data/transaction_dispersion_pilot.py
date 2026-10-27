"""Offline two-phase pilot for transaction-level auction price dispersion."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sqlite3
import uuid
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

from . import coverage, dashboard

PRICE_SHA256 = "eeb014d9c7487aae047eb1a0df153f18dc0d496a457a52c0125430401353926c"
PERIODS = {"2023": (dt.date(2023, 8, 8), dt.date(2023, 10, 31)), "2024": (dt.date(2024, 8, 8), dt.date(2024, 10, 31))}
HORIZONS = (1, 5, 10)
MIN_LOTS, MIN_RESIDUAL, CAP = 5, 30, .08
BANDWIDTHS, COMPARATORS = (9, 28), ("ablation", "direct", "persistence")
PRODUCTS = tuple(dashboard.PRODUCTS)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_immutable(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as f:
            f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")
            f.flush(); os.fsync(f.fileno())
        os.link(tmp, path)
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass


def _dec(x):
    try:
        d = Decimal(str(x)) if x not in (None, "") else None
        return d if d is not None and d.is_finite() else None
    except (InvalidOperation, ValueError, TypeError): return None


def _reconcile_signed_prices(calculated, frozen, totals=None):
    """Fail closed if local signed aggregation disagrees with pinned daily targets."""
    checked=[]
    for key,value in calculated.items():
        if key not in frozen: continue
        saved=frozen[key]
        expected=saved.get("actual") if isinstance(saved,dict) else saved
        if value is None or not math.isfinite(value) or not math.isclose(value,expected,rel_tol=1e-9,abs_tol=1e-8):
            raise ValueError(f"signed_price_mismatch:{key[0]}:{key[1]}")
        if isinstance(saved,dict) and totals is not None:
            amount,weight=totals.get(key,(None,None)); expected_weight=saved.get("volumeKg")
            expected_amount=expected*expected_weight if expected_weight is not None else None
            if (amount is None or weight is None or expected_weight is None
                    or not math.isclose(weight,expected_weight,rel_tol=1e-9,abs_tol=1e-8)
                    or not math.isclose(amount,expected_amount,rel_tol=1e-9,abs_tol=1e-8)):
                raise ValueError(f"signed_numerator_denominator_mismatch:{key[0]}:{key[1]}")
        checked.append(key)
    return checked


def _validate_frozen_manifest(frozen, manifest_path):
    if frozen.get("protocol")!="transaction-dispersion-pilot-2026-10-08" or frozen.get("status")!="extracted":
        raise ValueError("feature_manifest_protocol_or_status_mismatch")
    if frozen.get("pricesPinnedSha256")!=PRICE_SHA256 or frozen.get("pricesSha256")!=PRICE_SHA256:
        raise ValueError("feature_manifest_prices_hash_mismatch")
    code_sha=_sha(Path(__file__).read_bytes())
    if frozen.get("outputCodeSha256")!=code_sha: raise ValueError("feature_manifest_code_hash_mismatch")
    src=Path(manifest_path).with_name(frozen.get("sourceRowsArtifact", ""))
    if not src.is_file() or _sha(src.read_bytes())!=frozen.get("sourceRowsSha256"):
        raise ValueError("feature_manifest_source_artifact_hash_mismatch")
    expected={(y,d.isoformat(),p) for y,(lo,hi) in PERIODS.items() for d in _dates(lo,hi) for p in PRODUCTS}
    actual={}
    for row in frozen.get("features",[]):
        key=(str(row.get("year")),row.get("origin"),row.get("product"))
        if key in actual: raise ValueError("duplicate_feature_key")
        actual[key]=row
    if set(actual)!=expected: raise ValueError("feature_manifest_feature_keys_mismatch")
    for (year,origin_text,product),row in actual.items():
        origin=dt.date.fromisoformat(origin_text); vector=row.get("features"); audit=row.get("audit",{})
        if vector is None and audit.get("status")!="unavailable": raise ValueError("missing_feature_without_unavailable_state")
        if vector is not None and (not isinstance(vector,list) or len(vector)!=2 or not all(isinstance(x,(int,float)) and math.isfinite(x) for x in vector)):
            raise ValueError("invalid_feature_vector")
        if audit.get("sourceDate") is not None:
            source_day=dt.date.fromisoformat(audit["sourceDate"]); age=(origin-source_day).days
            if source_day>=origin or age<1 or audit.get("ageDays")!=age: raise ValueError("invalid_feature_source_date")
            if vector is not None and (age>14 or audit.get("positiveLotCount",0)<MIN_LOTS): raise ValueError("invalid_feature_source_date_or_eligibility")
            if age>14 and (vector is not None or audit.get("reason")!="source_older_than_14_days"): raise ValueError("invalid_stale_feature_state")
        elif vector is not None: raise ValueError("available_feature_missing_source_date")
        if vector is not None and audit.get("status")!="available": raise ValueError("feature_status_mismatch")
    return actual


def _preferred_garlic_scopes(records):
    scopes=set()
    for rec in records:
        q=json.loads(rec["query_json"])
        subclass=str(q.get("cond[gds_sclsf_cd::EQ]", ""))
        if (dashboard._query_product(q)=="garlic" and rec["selected_state"] in ("complete","empty")
                and subclass in dashboard.PEELED_GARLIC_LABELS):
            scopes.add((q.get("cond[trd_clcln_ymd::EQ]"),q.get("cond[whsl_mrkt_cd::EQ]"),subclass))
    return scopes


def _payload_category_exclusion(product, query, payload, preferred):
    if dashboard._product_id(payload)!=product: return "payload_category_mismatch"
    if product!="garlic": return None
    if not dashboard._garlic_class_is_verified(payload): return "unverified_garlic_class"
    subclass=str(payload.get("gds_sclsf_cd", "")); requested=str(query.get("cond[gds_sclsf_cd::EQ]", ""))
    if requested and subclass!=requested: return "payload_subclass_query_mismatch"
    scope=(query.get("cond[trd_clcln_ymd::EQ]"),query.get("cond[whsl_mrkt_cd::EQ]"),subclass)
    if not requested and scope in preferred: return "overlap_preferred_specific_query"
    return None


def weighted_quantile(rows, q):
    """Weighted inverse empirical CDF; positive amount and kg rows only."""
    pairs = sorted((Decimal(a)/Decimal(w), Decimal(w)) for a, w in rows if _dec(a) and _dec(w) and Decimal(a)>0 and Decimal(w)>0)
    if not pairs: return None
    total = sum((w for _, w in pairs), Decimal(0)); threshold = total*Decimal(str(q)); cum = Decimal(0)
    for value, weight in pairs:
        cum += weight
        if cum >= threshold: return float(value)
    return float(pairs[-1][0])


def dispersion_features(transactions, daily_prices, product, origin, unreconciled_days=()):
    """Use the most recent strictly prior trading day; never back off past it."""
    candidates = defaultdict(list); rejected = defaultdict(lambda: defaultdict(int))
    for r in transactions:
        day = dt.date.fromisoformat(r["tradeDate"])
        if r["product"] != product or day >= origin: continue
        amount, kg = _dec(r.get("amountRaw")), _dec(r.get("weightRaw"))
        if amount is None or kg is None: rejected[day]["invalid_amount_or_weight"] += 1
        elif amount <= 0: rejected[day]["nonpositive_amount"] += 1
        elif kg <= 0: rejected[day]["nonpositive_weight"] += 1
        else: candidates[day].append((r["amountRaw"], r["weightRaw"]))
    days=[dt.date.fromisoformat(r["tradeDate"]) for r in transactions if r["product"]==product and dt.date.fromisoformat(r["tradeDate"])<origin]
    if not days: return None, {"status":"unavailable","reason":"no_prior_trading_day","rejected":{d.isoformat():dict(v) for d,v in rejected.items()}}
    day=max(days); rows=candidates.get(day,[]); signed=daily_prices.get((product,day.isoformat())); age=(origin-day).days; n=len(rows)
    if age > 14: return None,{"status":"unavailable","reason":"source_older_than_14_days","sourceDate":day.isoformat(),"ageDays":age,"positiveLotCount":n}
    if (product,day.isoformat()) in unreconciled_days: return None,{"status":"unavailable","reason":"not_in_frozen_prices_snapshot","sourceDate":day.isoformat(),"ageDays":age,"positiveLotCount":n}
    if n<MIN_LOTS: return None,{"status":"unavailable","reason":"fewer_than_5_positive_lots","sourceDate":day.isoformat(),"ageDays":age,"positiveLotCount":n}
    if signed is None or not math.isfinite(signed) or signed<=0: return None,{"status":"unavailable","reason":"invalid_signed_daily_price","sourceDate":day.isoformat(),"ageDays":age,"positiveLotCount":n}
    p25,p50,p75=(weighted_quantile(rows,q) for q in (.25,.5,.75))
    if not all(v is not None and math.isfinite(v) and v>0 for v in (p25,p50,p75)) or p75<p25:
        return None,{"status":"unavailable","reason":"invalid_weighted_quantiles","sourceDate":day.isoformat(),"ageDays":age,"positiveLotCount":n}
    vector=[math.log(p50/signed),math.log(p75/p25)]
    if not all(math.isfinite(x) for x in vector): return None,{"status":"unavailable","reason":"nonfinite_feature","sourceDate":day.isoformat(),"ageDays":age,"positiveLotCount":n}
    return vector,{"status":"available","sourceDate":day.isoformat(),"ageDays":age,"positiveLotCount":n,"weightedQuantilesPerKg":{"p25":p25,"p50":p50,"p75":p75},"signedDailyPrice":signed,"rejected":{d.isoformat():dict(v) for d,v in rejected.items()}}


def extract(db_path, output):
    """Read selected response rows through canonical dashboard selection SQL."""
    db = coverage.open_readonly(str(db_path))
    rows=[]; source=defaultdict(lambda: {"requests":set(),"responseShas":set(),"excluded":defaultdict(int)})
    start=min(x[0] for x in PERIODS.values())-dt.timedelta(days=14); end=max(x[1] for x in PERIODS.values())
    try:
        selected=list(db.execute(dashboard.PRICE_SQL,(start.isoformat(),end.isoformat(),*dashboard.MARKETS.keys())))
        preferred_garlic_scopes=_preferred_garlic_scopes(selected)
        for rec in selected:
            query=json.loads(rec["query_json"]); payload=json.loads(rec["payload_json"]) if rec["payload_json"] is not None else None
            product=dashboard._query_product(query); day=query.get("cond[trd_clcln_ymd::EQ]"); market=query.get("cond[whsl_mrkt_cd::EQ]")
            key=(product,day,market,rec["task_key"])
            m=source[key]; m["selectedState"]=rec["selected_state"]
            m.update({"query":query,"taskKey":rec["task_key"],"sourceUrl":rec["source_url"],"latestState":rec["latest_state"],"latestJobId":rec["latest_job_id"],"latestEpochId":rec["latest_epoch_id"],"selectedJobId":rec["selected_job_id"],"selectedEpochId":rec["selected_epoch_id"],"selectedResponseAt":rec["selected_response_at"]})
            if rec["request_id"] is not None:
                reqs=db.execute("SELECT request_id,response_sha256,retrieved_at,state,query_json FROM source_requests WHERE request_id=?",(rec["request_id"],)).fetchall()
            else:
                job_id=rec["selected_job_id"] or rec["latest_job_id"]
                epoch_id=rec["selected_epoch_id"] or rec["latest_epoch_id"]
                reqs=db.execute("SELECT request_id,response_sha256,retrieved_at,state,query_json FROM source_requests WHERE job_id=? AND epoch_id=? AND dataset='prices_origin' AND source_url=? AND coverage_task_key(dataset,source_url,query_json)=?",(job_id,epoch_id,rec["source_url"],rec["task_key"])).fetchall() if job_id is not None else []
            for req in reqs:
                m["requests"].add(req["request_id"])
                m.setdefault("requestMeta",{})[str(req["request_id"]) ]={"responseSha256":req["response_sha256"],"retrievedAt":req["retrieved_at"],"state":req["state"],"query":json.loads(req["query_json"])}
            if rec["selected_state"] not in ("complete","empty") or payload is None: continue
            if not isinstance(payload,dict): continue
            exclusion=_payload_category_exclusion(product,query,payload,preferred_garlic_scopes)
            if exclusion: m["excluded"][exclusion]+=1; continue
            if payload.get("trd_se")!="경매": m["excluded"]["not_auction"]+=1; continue
            if payload.get("unit_nm")!="kg": m["excluded"]["unit_not_kg"]+=1; continue
            if market not in dashboard.MARKETS or payload.get("whsl_mrkt_cd")!=market: m["excluded"]["market_mismatch"]+=1; continue
            if payload.get("trd_clcln_ymd")!=day: m["excluded"]["date_mismatch"]+=1; continue
            rows.append({"product":product,"tradeDate":day,"market":market,"amountRaw":payload.get("totprc"),"weightRaw":payload.get("unit_tot_qty"),"requestId":rec["request_id"],"rowNo":rec["row_no"],"qualityFlags":rec["quality_flags"]})
    finally: db.close()
    prices_raw=Path(output).resolve().parent
    price_file=prices_raw/"prices.json"
    if not price_file.exists():
        # Snapshot path is fixed by the protocol and independent of output location.
        price_file=Path(__file__).resolve().parents[1]/"outputs/daily/2026-10-07/20261007T120015-e366c766/prices.json"
    if _sha(price_file.read_bytes())!=PRICE_SHA256: raise ValueError("prices_snapshot_hash_mismatch")
    daily={}; daily_saved={}
    for p in PRODUCTS:
        for obs in json.loads(price_file.read_bytes()).get("products",[]):
            if obs.get("id")==p:
                for o in obs.get("observations",[]):
                    k=(p,o["date"]);daily[k]=float(o["actual"]);daily_saved[k]={"actual":float(o["actual"]),"volumeKg":float(o["volumeKg"])}
    tx=[]; daily_transactions=defaultdict(list)
    for r in rows:
        a,w=_dec(r["amountRaw"]),_dec(r["weightRaw"])
        if a is not None and w is not None:
            k=(r["product"],r["tradeDate"]); old=daily_transactions[k]; old.extend([(a,w)])
    totals={k:(float(sum((a for a,w in v),Decimal(0))),float(sum((w for a,w in v),Decimal(0)))) for k,v in daily_transactions.items()}
    signed={k:(amount/weight if weight else None) for k,(amount,weight) in totals.items()}
    reconciled=_reconcile_signed_prices(signed,daily_saved,totals)
    unreconciled={(p,d) for p,d in signed if (p,d) not in daily_saved}
    reconciled_signed={k:signed[k] for k in reconciled}
    # Persist transaction-level provenance and features before any scoring.
    for r in rows: tx.append(r)
    feature_rows=[]
    for year,(lo,hi) in PERIODS.items():
        origin=lo
        while origin<=hi:
            for p in PRODUCTS:
                vec,audit=dispersion_features(tx,reconciled_signed,p,origin,unreconciled); feature_rows.append({"year":year,"origin":origin.isoformat(),"product":p,"features":vec,"audit":audit})
            origin+=dt.timedelta(days=1)
    source_summary=[{"product":k[0],"tradeDate":k[1],"market":k[2],"taskKey":k[3],"conditions":v.get("query"),"sourceUrl":v.get("sourceUrl"),"latestState":v.get("latestState"),"latestJobId":v.get("latestJobId"),"latestEpochId":v.get("latestEpochId"),"selectedState":v.get("selectedState"),"selectedJobId":v.get("selectedJobId"),"selectedEpochId":v.get("selectedEpochId"),"selectedResponseAt":v.get("selectedResponseAt"),"selectedRequestCount":len(v["requests"]),"requests":v.get("requestMeta",{}),"excludedRows":dict(v["excluded"])} for k,v in sorted(source.items())]
    source_artifact=Path(output).with_name(Path(output).stem+".source_rows.json")
    source_bytes=(json.dumps(rows,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    _write_immutable(source_artifact,rows)
    reconciliation_status="passed" if not unreconciled else ("no_overlapping_dates" if not reconciled else "partial_unreconciled_source_dates")
    manifest={"schemaVersion":1,"protocol":"transaction-dispersion-pilot-2026-10-08","status":"extracted","pricesPinnedSha256":PRICE_SHA256,"pricesSha256":PRICE_SHA256,"selection":"coverage.SELECTION_CTE + dashboard.PRICE_SQL","queryRange":[start.isoformat(),end.isoformat()],"sourceRows":len(rows),"sourceRowsArtifact":source_artifact.name,"sourceRowsSha256":_sha(source_bytes),"sourceRowsWithSignedAdjustmentsRetained":sum(1 for r in rows if (_dec(r["amountRaw"]) is not None and _dec(r["amountRaw"])<0) or (_dec(r["weightRaw"]) is not None and _dec(r["weightRaw"])<0)),"signedPriceReconciliation":{"sourceDates":len(signed),"overlappingDatesChecked":len(reconciled),"unreconciledSourceDates":[{"product":p,"date":d} for p,d in sorted(unreconciled)],"status":reconciliation_status},"sourceQuerySummary":source_summary,"features":feature_rows,"outputCodeSha256":_sha(Path(__file__).read_bytes())}
    _write_immutable(output,manifest); return manifest


def _dm(diffs,bw):
    dates=sorted(diffs); n=len(dates)
    if n<30:return {"status":"insufficient_pairs","n":n,"conclusion":"inference_unavailable"}
    mean=sum(diffs.values())/n; c={d:v-mean for d,v in diffs.items()}; s=sum(v*v for v in c.values())
    for lag in range(1,bw+1): s+=2*(1-lag/(bw+1))*sum(c[d]*c.get(d+dt.timedelta(days=lag),0) for d in dates)
    var=s/(n*n)
    if not math.isfinite(var) or var<=0:return {"status":"invalid_hac_variance","n":n,"conclusion":"inference_unavailable"}
    z=mean/math.sqrt(var); return {"status":"tested","n":n,"meanLossDifference":mean,"statistic":z,"pValue":math.erfc(abs(z)/math.sqrt(2)),"conclusion":"inconclusive"}


def _holm(family):
    eligible=sorted((x for x in family if x.get("status")=="tested"),key=lambda x:x["pValue"])
    running=0.; total=len(family)
    for i,x in enumerate(eligible):
        running=max(running,min(1.,x["pValue"]*(total-i)))
        x["holmAdjustedPValue"]=running;x["holmSignificant"]=running<.05
        if x["holmSignificant"]: x["conclusion"]="improvement" if x["meanLossDifference"]<0 else "worsening"
        else: x["conclusion"]="inconclusive"
    for x in family:
        if x.get("status")!="tested":x.update(holmAdjustedPValue=None,holmSignificant=False,conclusion="inference_unavailable")
    return family


def score(manifest_path, prices_path, output):
    """Score only frozen 2024 origins, using the immutable extracted features."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from . import kma_warning_pilot as kp, live_forecast
    raw=Path(prices_path).read_bytes()
    if _sha(raw)!=PRICE_SHA256: raise ValueError("prices_snapshot_hash_mismatch")
    manifest_raw=Path(manifest_path).read_bytes(); frozen=json.loads(manifest_raw)
    frozen_features=_validate_frozen_manifest(frozen,manifest_path)
    features={(r["product"],dt.date.fromisoformat(r["origin"])):r["features"] for r in frozen_features.values()}
    products={p["id"]:p for p in json.loads(raw)["products"]}; rows=[]
    eval_start,eval_end=PERIODS["2024"]
    for pid in PRODUCTS:
        series={dt.date.fromisoformat(r["date"]):(float(r["actual"]),float(r.get("volumeKg",0))) for r in products[pid].get("observations",[])}
        days=tuple(sorted(series)); logs=tuple(math.log(series[d][0]) for d in days)
        residuals={h:[] for h in HORIZONS}
        for origin in _dates(*PERIODS["2023"]):
            for h in HORIZONS:
                model,n=kp._fit_direct(series,origin,h,days,logs); target=origin+dt.timedelta(days=h); strict=live_forecast._strict_feature_row(series,origin,target,h,days,logs)
                if model is None or strict is None or target not in series: continue
                src,x=strict; pred=series[src][0]*math.exp(float(model.predict([x])[0])); vec=features.get((pid,origin))
                if vec is not None and math.isfinite(pred) and pred>0:residuals[h].append((target,vec,math.log(series[target][0]/pred)))
        for origin in _dates(eval_start,eval_end):
            vec=features.get((pid,origin)); target_rows=[]
            for h in HORIZONS:
                model,nd=kp._fit_direct(series,origin,h,days,logs); target=origin+dt.timedelta(days=h); strict=live_forecast._strict_feature_row(series,origin,target,h,days,logs)
                direct=None
                if model is not None and strict:
                    src,x=strict; direct=series[src][0]*math.exp(float(model.predict([x])[0]))
                train=[r for r in residuals[h] if r[0]<origin]; nres=len({r[0] for r in train})
                corr=None; mean=None
                if nres>=MIN_RESIDUAL:
                    corr=make_pipeline(StandardScaler(),Ridge(alpha=100,fit_intercept=True)).fit([r[1] for r in train],[r[2] for r in train]);mean=sum(r[2] for r in train)/len(train)
                candidate=ablation=None
                if vec is not None and corr is not None and direct and strict:
                    src,x=strict; base=math.log(direct/series[src][0]); candidate=series[src][0]*math.exp(base+max(-CAP,min(CAP,float(corr.predict([vec])[0])))); ablation=series[src][0]*math.exp(base+max(-CAP,min(CAP,mean)))
                prior=max((d for d in days if d<origin),default=None); persistence=series[prior][0] if prior and (origin-prior).days<=14 else None
                actual=series.get(target,(None,))[0]
                rows.append({"product":pid,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":h,"actual":actual,"candidate":candidate,"ablation":ablation,"direct":direct,"persistence":persistence,"candidateTrainingTargets":nres,"directTrainingTargets":nd})
    report=_summarize(rows); report.update({"protocol":"transaction-dispersion-pilot-2026-10-08","sourceManifestSha256":_sha(manifest_raw),"pricesSha256":_sha(raw),"limitations":["Retrospective revised transaction responses; first-publication time is unknown.","The target is the frozen two-market proxy."]})
    _write_immutable(output,report);return report


def _dates(start,end):
    for i in range((end-start).days+1):yield start+dt.timedelta(days=i)


def _summarize(rows):
    paired=[r for r in rows if r["actual"] is not None and all(r[m] is not None for m in ("candidate","ablation","direct","persistence"))]
    metrics={m:{"n":len(paired),"mae":sum(abs(r["actual"]-r[m]) for r in paired)/len(paired) if paired else None,"mape":100*sum(abs(r["actual"]-r[m])/r["actual"] for r in paired)/len(paired) if paired else None} for m in ("candidate","ablation","direct","persistence")}
    coverage={m:sum(r["actual"] is not None and r[m] is not None for r in rows) for m in ("candidate","ablation","direct","persistence")}
    families={}
    for bw in BANDWIDTHS:
        family=[]
        for pid in PRODUCTS:
            for h in HORIZONS:
                g=[r for r in paired if r["product"]==pid and r["horizon"]==h]
                for comp in COMPARATORS:
                    for loss in ("AE","APE"):
                        d={dt.date.fromisoformat(r["target"]):(abs(r["actual"]-r["candidate"])-abs(r["actual"]-r[comp]))*(100/r["actual"] if loss=="APE" else 1) for r in g}
                        family.append({"product":pid,"horizon":h,"comparator":comp,"loss":loss,**_dm(d,bw)})
        _holm(family)
        families[str(bw)]=family
    groups=[]; group_metrics={}
    for pid in PRODUCTS:
        for h in HORIZONS:
            g=[r for r in rows if r["product"]==pid and r["horizon"]==h];groups.append({"product":pid,"horizon":h,"planned":len(g),"targetObserved":sum(r["actual"] is not None for r in g),"commonPairs":sum(r in paired for r in g),"coverageByModel":{m:sum(r["actual"] is not None and r[m] is not None for r in g) for m in metrics}})
            pg=[r for r in g if r["actual"] is not None and all(r[m] is not None for m in ("candidate","ablation","direct","persistence"))]
            group_metrics[f"{pid}/h{h}"]={m:{"n":len(pg),"mae":sum(abs(r["actual"]-r[m]) for r in pg)/len(pg) if pg else None,"mape":100*sum(abs(r["actual"]-r[m])/r["actual"] for r in pg)/len(pg) if pg else None} for m in metrics}
    macro={m:(sum(group_metrics[f"{p}/h{h}"][m]["mape"] for p in PRODUCTS for h in HORIZONS)/(len(PRODUCTS)*len(HORIZONS)) if all(group_metrics[f"{p}/h{h}"][m]["mape"] is not None for p in PRODUCTS for h in HORIZONS) else None) for m in metrics}
    crop_limits={}
    for p in PRODUCTS:
        cm=[group_metrics[f"{p}/h{h}"]["candidate"]["mape"] for h in HORIZONS]; dm=[group_metrics[f"{p}/h{h}"]["direct"]["mape"] for h in HORIZONS]
        ca=[group_metrics[f"{p}/h{h}"]["candidate"]["mae"] for h in HORIZONS]; da=[group_metrics[f"{p}/h{h}"]["direct"]["mae"] for h in HORIZONS]
        crop_limits[p]={"mapeIncreasePp":sum(x-y for x,y in zip(cm,dm))/3 if all(x is not None for x in cm+dm) else None,"maeRelativeIncrease":(sum(ca)/sum(da)-1 if sum(da)>0 else None) if all(x is not None for x in ca+da) else None}
    gate={"allGroupsAtLeast30CommonPairs":all(g["commonPairs"]>=30 for g in groups),"candidateCoverageAtLeastDirect":all(g["coverageByModel"]["candidate"]>=g["coverageByModel"]["direct"] for g in groups),"candidateMacroMapeAtMost10":macro["candidate"] is not None and macro["candidate"]<=10,"candidateGainAtLeast0_5ppVsAblation":macro["ablation"] is not None and macro["candidate"] is not None and macro["ablation"]-macro["candidate"]>=.5,"candidateGainAtLeast0_5ppVsDirect":macro["direct"] is not None and macro["candidate"] is not None and macro["direct"]-macro["candidate"]>=.5,"cropWorseningWithinLimits":all(v["mapeIncreasePp"] is not None and v["mapeIncreasePp"]<=1 and v["maeRelativeIncrease"] is not None and v["maeRelativeIncrease"]<=.05 for v in crop_limits.values())}
    gate["passed"]=all(gate.values())
    return {"coverage":{"planned":1275,"rows":len(rows),"commonPairs":len(paired),"models":coverage,"groups":groups},"pairedMetrics":metrics,"groupPairedMetrics":group_metrics,"equalWeightMacroMape":macro,"cropWorseningVsDirect":crop_limits,"dmTestsByHacCalendarDays":families,"frozenExpansionGate":gate,"rows":rows}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    e=sub.add_parser("extract");e.add_argument("--db",type=Path,required=True);e.add_argument("--manifest",type=Path,required=True)
    s=sub.add_parser("score");s.add_argument("--manifest",type=Path,required=True);s.add_argument("--prices",type=Path,required=True);s.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv)
    if a.command=="extract":extract(a.db,a.manifest)
    else:score(a.manifest,a.prices,a.output)
    return 0

if __name__=="__main__":raise SystemExit(main())
