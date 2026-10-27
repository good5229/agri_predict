"""Offline, frozen KMA warning-count feature pilot for dashboard prices."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from . import dashboard, live_forecast, modeling
from .kma_warning_csv import parse_csv

PRICE_SHA256 = "eeb014d9c7487aae047eb1a0df153f18dc0d496a457a52c0125430401353926c"
REGIONS = ("강원특별자치도", "전남광주통합특별시")
HORIZONS = (1, 5, 10)
TRAIN_ORIGINS = (dt.date(2023, 8, 8), dt.date(2023, 10, 31))
EVAL_ORIGINS = (dt.date(2024, 8, 8), dt.date(2024, 10, 31))
MIN_DIRECT, MIN_RESIDUAL = 90, 30
CAP, BANDWIDTHS, COMPARATORS, LOSSES = .08, (9, 28), ("ablation", "direct", "persistence"), ("AE", "APE")
FAMILY_SIZE = len(dashboard.PRODUCTS) * len(HORIZONS) * len(COMPARATORS) * len(LOSSES)
METRIC_GROUP_COUNT = len(dashboard.PRODUCTS) * len(HORIZONS)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def warning_features(rows, origin: dt.date, *, query_start: dt.date, query_end: dt.date):
    """Four log1p counts. Return unavailable if either complete window is unobserved."""
    origin_dt = dt.datetime.combine(origin, dt.time(10))
    counts = {}
    for days in (1, 7):
        begin = origin - dt.timedelta(days=days)
        if begin < query_start or origin > query_end + dt.timedelta(days=1):
            return None
        for region in REGIONS:
            n = sum(r.region == region and origin_dt-dt.timedelta(days=days) <= r.issue_time < origin_dt for r in rows)
            counts[f"{region}_{days}d"] = n
    return [math.log1p(counts[f"{region}_{days}d"]) for days in (1, 7) for region in REGIONS]


def _positive(value):
    return value is not None and math.isfinite(value) and value > 0


def _fit_direct(series, origin, horizon, days, logs):
    X, y, targets = [], [], set()
    for target, (price, _volume) in series.items():
        if target >= origin:
            continue
        sample_origin = target - dt.timedelta(days=horizon)
        feature = live_forecast._strict_feature_row(series, sample_origin, target, horizon, days, logs)
        if feature is None:
            continue
        src, vector = feature
        label = math.log(price / series[src][0])
        if math.isfinite(label):
            X.append(vector); y.append(label); targets.add(target)
    if len(targets) < MIN_DIRECT:
        return None, len(targets)
    estimator=modeling._new_estimator("ridge")
    estimator.fit(X,y)
    return estimator, len(targets)


def _correction_fit(rows):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    # rows are (target, four warning features, log(actual/direct))
    if len({r[0] for r in rows}) < MIN_RESIDUAL:
        return None, None, len({r[0] for r in rows})
    X, y = [r[1] for r in rows], [r[2] for r in rows]
    return make_pipeline(StandardScaler(), Ridge(alpha=100, fit_intercept=True)).fit(X, y), sum(y)/len(y), len({r[0] for r in rows})


def corrected_prices(direct, source_price, feature, model, mean_residual):
    """Return candidate/ablation or unavailable values; never substitute direct."""
    if not (_positive(direct) and _positive(source_price) and feature is not None
            and model is not None and mean_residual is not None):
        return None, None
    try:
        direct_log=math.log(direct/source_price)
        candidate=source_price*math.exp(direct_log+max(-CAP,min(CAP,float(model.predict([feature])[0]))))
        ablation=source_price*math.exp(direct_log+max(-CAP,min(CAP,float(mean_residual))))
    except (ValueError,OverflowError,TypeError):
        return None,None
    return (candidate if _positive(candidate) else None,
            ablation if _positive(ablation) else None)


def _dm(diffs, bandwidth):
    n = len(diffs)
    if n < 30: return {"status":"insufficient_pairs","n":n,"meanLossDifference":None,"statistic":None,"pValue":1.0}
    mean = sum(diffs.values()) / n; dates = sorted(diffs)
    centered = {d:v-mean for d,v in diffs.items()}
    s = sum(v*v for v in centered.values())
    for lag in range(1, bandwidth+1):
        s += 2*(1-lag/(bandwidth+1))*sum(centered[d]*centered.get(d+dt.timedelta(days=lag),0) for d in dates)
    var = s/(n*n)
    if not math.isfinite(var) or var <= 0:
        return {"status":"invalid_hac_variance","n":n,"meanLossDifference":mean,"statistic":None,"pValue":None}
    z=mean/math.sqrt(var)
    return {"status":"tested","n":n,"meanLossDifference":mean,"statistic":z,"pValue":math.erfc(abs(z)/math.sqrt(2))}


def _holm(items):
    order=sorted((i for i,x in enumerate(items) if x["status"]=="tested"),key=lambda i:items[i]["pValue"])
    running=0.0; m=len(items)
    for rank,i in enumerate(order):
        adj=min(1.0,(m-rank)*items[i]["pValue"]); running=max(running,adj)
        items[i]["holmAdjustedPValue"]=running; items[i]["holmSignificant"]=running<.05
    for x in items:
        if x["status"]!="tested": x.update(holmAdjustedPValue=None,holmSignificant=False)


def _write_immutable(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x",encoding="utf-8") as f:
            f.write(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n"); f.flush(); os.fsync(f.fileno())
        os.link(tmp,path)
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass


def _load_warnings(path, year, expected_rows):
    rows,digest=parse_csv(path)
    start,end=dt.date(year,8,1),dt.date(year,10,31)
    if len(rows)!=expected_rows: raise ValueError(f"{year}_warning_row_count_mismatch")
    if any(not start <= r.issue_time.date() <= end for r in rows): raise ValueError("warning_row_outside_declared_query")
    if any(r.region not in REGIONS for r in rows): raise ValueError("warning_region_outside_frozen_set")
    return rows,digest


def build(prices_path, csv_2023, csv_2024, expected_rows_2023, expected_rows_2024, output_dir):
    raw=Path(prices_path).read_bytes()
    if _sha(raw)!=PRICE_SHA256: raise ValueError("prices_snapshot_hash_mismatch")
    prices=json.loads(raw)
    if prices.get("schemaVersion")!=1: raise ValueError("unsupported_prices_schema")
    products={p.get("id"):p for p in prices.get("products",[]) if isinstance(p,dict)}
    if set(products)!=set(dashboard.PRODUCTS): raise ValueError("prices_must_contain_dashboard_products")
    warnings23,hash23=_load_warnings(csv_2023,2023,expected_rows_2023)
    warnings24,hash24=_load_warnings(csv_2024,2024,expected_rows_2024)
    out=Path(output_dir)
    if out.exists() and any(out.iterdir()): raise ValueError("output_directory_must_be_new_or_empty")
    out.mkdir(parents=True,exist_ok=True)
    manifest={"schemaVersion":1,"protocol":"kma-warning-count-pilot-2026-10-08","status":"started","queryRanges":{"2023":["2023-08-01","2023-10-31"],"2024":["2024-08-01","2024-10-31"]},"allowedRegions":list(REGIONS),"expectedUiRowCounts":{"2023":expected_rows_2023,"2024":expected_rows_2024},"sourceHashes":{"pricesSha256":_sha(raw),"warnings2023Sha256":hash23,"warnings2024Sha256":hash24},"pricesPinnedSha256":PRICE_SHA256,"plannedRows":1275,"fitProtocol":{"minimumDirectTargets":MIN_DIRECT,"minimumResidualTargets":MIN_RESIDUAL,"candidate":"StandardScaler + Ridge(alpha=100, fit_intercept=True)","ablation":"mean log residual on identical training rows","clipLogCorrection":[-CAP,CAP],"hacCalendarDays":list(BANDWIDTHS),"holmFamilySizePerBandwidth":FAMILY_SIZE}}
    _write_immutable(out/"manifest.json",manifest)
    allrows=[]; fit_audits=[]
    for pid in dashboard.PRODUCTS:
        series={dt.date.fromisoformat(r["date"]):(float(r["actual"]),float(r.get("volumeKg",0))) for r in products[pid].get("observations",[])}
        if any(not _positive(v[0]) for v in series.values()): raise ValueError(f"invalid_price:{pid}")
        days=tuple(sorted(series)); logs=tuple(math.log(series[d][0]) for d in days)
        # Generate out-of-time 2023 direct forecasts, then use only those residuals.
        residuals={h:[] for h in HORIZONS}
        for origin in _dates(TRAIN_ORIGINS):
            feat=warning_features(warnings23,origin,query_start=dt.date(2023,8,1),query_end=dt.date(2023,10,31))
            for h in HORIZONS:
                model,ntrain=_fit_direct(series,origin,h,days,logs)
                target=origin+dt.timedelta(days=h); strict=live_forecast._strict_feature_row(series,origin,target,h,days,logs)
                direct=None
                if model is not None and strict:
                    src,x=strict
                    try: direct=series[src][0]*math.exp(float(model.predict([x])[0]))
                    except (ValueError,OverflowError): pass
                actual=series.get(target,(None,))[0]
                if feat is not None and _positive(direct) and _positive(actual): residuals[h].append((target,feat,math.log(actual/direct)))
        for origin in _dates(EVAL_ORIGINS):
            feat=warning_features(warnings24,origin,query_start=dt.date(2024,8,1),query_end=dt.date(2024,10,31))
            for h in HORIZONS:
                model,ndirect=_fit_direct(series,origin,h,days,logs)
                target=origin+dt.timedelta(days=h); strict=live_forecast._strict_feature_row(series,origin,target,h,days,logs)
                direct=None
                if model is not None and strict:
                    src,x=strict
                    try: direct=series[src][0]*math.exp(float(model.predict([x])[0]))
                    except (ValueError,OverflowError): pass
                corr,mean,nres=_correction_fit([r for r in residuals[h] if r[0] < origin])
                candidate=ablation=None; reason=None
                if feat is None: reason="warning_window_unavailable"
                elif corr is None: reason="fewer_than_30_distinct_residual_targets"
                elif not _positive(direct) or strict is None: reason="direct_unavailable"
                else:
                    src,x=strict
                    candidate,ablation=corrected_prices(direct,series[src][0],feat,corr,mean)
                    if candidate is None or ablation is None: reason="invalid_correction"
                actual=series.get(target,(None,))[0]
                srcday=max((d for d in days if d<origin),default=None)
                persistence=series[srcday][0] if srcday and (origin-srcday).days<=dashboard.MAX_SOURCE_AGE_DAYS else None
                values={"candidate":candidate,"ablation":ablation,"direct":direct,"persistence":persistence}
                available={k:_positive(v) for k,v in values.items()}
                allrows.append({"product":pid,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":h,"actual":actual if _positive(actual) else None,**values,"statuses":{k:"available" if v else "unavailable" for k,v in available.items()},"candidateUnavailableReason":None if available["candidate"] else reason or "model_or_input_unavailable","directTrainingTargetDays":ndirect,"residualTrainingTargetDays":nres,"pairedEvaluable":_positive(actual) and all(available.values())})
    report=_summarize(allrows)
    report.update({"schemaVersion":1,"protocol":manifest,"sourceManifestSha256":_sha((out/"manifest.json").read_bytes()),"codeSha256":_sha(Path(__file__).read_bytes()),"limitations":["Retrospective revised prices; 2024 was previously explored and is not independent validation.","Warning-row counts do not establish active warning state or first-publication time."]})
    _write_immutable(out/"report.json",report)
    manifest["status"]="complete"; manifest["reportSha256"]=_sha((out/"report.json").read_bytes())
    # Manifest is immutable by protocol: completion and report hash go in a separate seal.
    _write_immutable(out/"seal.json",manifest)
    return report


def _dates(bounds):
    start,end=bounds
    for i in range((end-start).days+1): yield start+dt.timedelta(days=i)


def _summarize(rows):
    paired=[r for r in rows if r["pairedEvaluable"]]
    metrics={}
    for model in ("candidate","ablation","direct","persistence"):
        by={}
        for pid in dashboard.PRODUCTS:
            for h in HORIZONS:
                g=[r for r in paired if r["product"]==pid and r["horizon"]==h]
                by[f"{pid}/h{h}"]={"n":len(g),"mae":sum(abs(r["actual"]-r[model]) for r in g)/len(g) if g else None,"mape":100*sum(abs(r["actual"]-r[model])/r["actual"] for r in g)/len(g) if g else None}
        metrics[model]=by
    # Equal weight across horizons within each crop, then across crops.
    macro={m:(sum(metrics[m][f"{p}/h{h}"]["mape"] for p in dashboard.PRODUCTS for h in HORIZONS)/METRIC_GROUP_COUNT if all(metrics[m][f"{p}/h{h}"]["mape"] is not None for p in dashboard.PRODUCTS for h in HORIZONS) else None) for m in metrics}
    tests={}
    for bw in BANDWIDTHS:
        family=[]
        for pid in dashboard.PRODUCTS:
            for h in HORIZONS:
                group=[r for r in paired if r["product"]==pid and r["horizon"]==h]
                for comp in COMPARATORS:
                    for loss in LOSSES:
                        diffs={dt.date.fromisoformat(r["target"]):(abs(r["actual"]-r["candidate"])-abs(r["actual"]-r[comp])) if loss=="AE" else 100*(abs(r["actual"]-r["candidate"])-abs(r["actual"]-r[comp]))/r["actual"] for r in group}
                        x=_dm(diffs,bw); x.update(product=pid,horizon=h,comparator=comp,loss=loss)
                        family.append(x)
        _holm(family)
        for x in family:
            if x["status"]!="tested": x["conclusion"]="inference_unavailable"
            elif not x["holmSignificant"] or x["meanLossDifference"]==0: x["conclusion"]="inconclusive"
            else: x["conclusion"]="improvement" if x["meanLossDifference"]<0 else "worsening"
        tests[str(bw)]=family
    groups=[]
    for pid in dashboard.PRODUCTS:
        for h in HORIZONS:
            g=[r for r in rows if r["product"]==pid and r["horizon"]==h]
            groups.append({"product":pid,"horizon":h,"planned":len(g),"targetObserved":sum(r["actual"] is not None for r in g),"commonPairs":sum(r["pairedEvaluable"] for r in g),"coverageByModel":{m:sum(r["actual"] is not None and r["statuses"][m]=="available" for r in g) for m in ("candidate","ablation","direct","persistence")}})
    crop_worsening={}
    for pid in dashboard.PRODUCTS:
        c=[metrics["candidate"][f"{pid}/h{h}"]["mape"] for h in HORIZONS]
        d=[metrics["direct"][f"{pid}/h{h}"]["mape"] for h in HORIZONS]
        cmae=[metrics["candidate"][f"{pid}/h{h}"]["mae"] for h in HORIZONS]
        dmae=[metrics["direct"][f"{pid}/h{h}"]["mae"] for h in HORIZONS]
        direct_crop_mae=sum(dmae)/3 if all(x is not None for x in dmae) else None
        candidate_crop_mae=sum(cmae)/3 if all(x is not None for x in cmae) else None
        relative_mae=(candidate_crop_mae/direct_crop_mae-1 if direct_crop_mae>0 else 0.0 if candidate_crop_mae==0 else None) if direct_crop_mae is not None and candidate_crop_mae is not None else None
        crop_worsening[pid]={"mapeIncreasePp":sum(a-b for a,b in zip(c,d))/3 if all(x is not None for x in c+d) else None,"maeRelativeIncrease":relative_mae,"candidateEqualHorizonMae":candidate_crop_mae,"directEqualHorizonMae":direct_crop_mae}
    gains={m:(macro["candidate"] is not None and macro[m] is not None and macro[m]-macro["candidate"]) for m in ("ablation","direct")}
    coverage_ok=all(g["coverageByModel"]["candidate"]>=g["coverageByModel"]["direct"] for g in groups)
    gate={"allCropHorizonGroupsAtLeast30CommonPairs":all(g["commonPairs"]>=30 for g in groups),"candidateCoverageAtLeastDirect":coverage_ok,"candidateMacroMapeAtMost10Percent":macro["candidate"] is not None and macro["candidate"]<=10,"candidateGainAtLeast0_5ppVsAblation":gains["ablation"] is not False and gains["ablation"]>=.5,"candidateGainAtLeast0_5ppVsDirect":gains["direct"] is not False and gains["direct"]>=.5,"cropWorseningWithinDirectLimits":all(v["mapeIncreasePp"] is not None and v["mapeIncreasePp"]<=1 and v["maeRelativeIncrease"] is not None and v["maeRelativeIncrease"]<=.05 for v in crop_worsening.values())}
    gate["passed"]=all(gate.values())
    return {"pairedMetrics":metrics,"equalWeightMacroMape":macro,"coverage":{"planned":1275,"rows":len(rows),"commonPairs":len(paired),"groups":groups,"byModel":{m:sum(r["statuses"][m]=="available" and r["actual"] is not None for r in rows) for m in ("candidate","ablation","direct","persistence")}},"cropWorseningVsDirect":crop_worsening,"frozenExpansionGate":gate,"dmTestsByHacCalendarDays":tests,"rows":rows}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prices",type=Path,required=True); p.add_argument("--warnings-2023",type=Path,required=True); p.add_argument("--warnings-2024",type=Path,required=True)
    p.add_argument("--expected-ui-rows-2023",type=int,required=True); p.add_argument("--expected-ui-rows-2024",type=int,required=True); p.add_argument("--output-dir",type=Path,required=True)
    a=p.parse_args(argv)
    build(a.prices,a.warnings_2023,a.warnings_2024,a.expected_ui_rows_2023,a.expected_ui_rows_2024,a.output_dir)
    return 0

if __name__=="__main__": raise SystemExit(main())
