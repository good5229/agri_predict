"""Frozen cross-crop pooled HistGB experiment with pre-origin feature sharing."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
import tempfile
from typing import Any, Callable

from . import dashboard, direct_horizon as dh, modeling

ROOT=Path(__file__).resolve().parents[1]
INPUT_PRICES=dh.INPUT_PRICES
INPUT_REPORT=ROOT/"outputs/experiments/direct-horizon-20261006/report.json"
PRICES_SHA256="f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
REPORT_SHA256="a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
START,END=dt.date(2025,9,25),dt.date(2026,9,24)
HORIZONS=tuple(range(1,11)); SOURCE_MAX_AGE=14; TRAIN_DAYS=1460; MIN_CROP_ROWS=90
BANDWIDTHS=(9,28); COMPARATORS=("directRidge","persistence"); LOSSES=("mae","mape")
FAMILY_SIZE=len(dashboard.PRODUCTS)*10*2*2
MODEL_PARAMS={"loss":"squared_error","max_iter":40,"max_leaf_nodes":15,"min_samples_leaf":20,
 "learning_rate":.08,"l2_regularization":1.0,"random_state":42,"early_stopping":False}
PRODUCTS=tuple(dashboard.PRODUCTS)


def _pos(value,label):
    if isinstance(value,bool): raise ValueError(f"{label} must be positive finite")
    try: x=float(value)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError(f"{label} must be positive finite") from exc
    if not math.isfinite(x) or x<=0: raise ValueError(f"{label} must be positive finite")
    return x


def _report_rows(report):
    if report.get("schemaVersion",1)!=1 or report.get("protocol",{}).get("id")!="direct-horizon-ridge-2026-10-06":
        raise ValueError("unexpected direct-horizon report schema/protocol")
    out={}
    for r in report.get("pairedRows",[]):
        if not isinstance(r,dict) or r.get("_summary"): continue
        p,h=r.get("product"),r.get("horizon")
        if p not in PRODUCTS or isinstance(h,bool) or not isinstance(h,int) or h not in HORIZONS: raise ValueError("invalid product/horizon")
        target=dt.date.fromisoformat(r["target"]); origin=dt.date.fromisoformat(r["origin"]) if r.get("origin") else target-dt.timedelta(days=h)
        if target-origin!=dt.timedelta(days=h): raise ValueError("origin/horizon mismatch")
        key=(p,origin,target,h)
        if key in out: raise ValueError(f"duplicate direct report identity {key}")
        out[key]={"actual":_pos(r["actual"],"actual"),"directRidge":_pos(r["direct"],"direct Ridge"),"persistence":_pos(r["persistence"],"persistence")}
    if not out: raise ValueError("direct report has no paired rows")
    return out


def _latest(series,days,cutoff):
    i=bisect_right(days,cutoff)-1
    if i<0 or (cutoff-days[i]).days>SOURCE_MAX_AGE: return None
    return days[i]


def _features(product,origin,target,h,series_map,caches,days_map):
    base=modeling._feature_row(series_map[product],origin,target,h,caches[product])
    if base is None: return None
    source_day,own=base; values=list(own); availability={}
    values.extend(1.0 if product==label else 0.0 for label in PRODUCTS)
    for other in PRODUCTS:
        if other==product:
            values.extend((0.0,0.0))
            availability[other]={"available":False,"currentDate":None,"priorDate":None,
                "reason":"own_crop_intentionally_masked"}
            continue
        series=series_map[other]; days=days_map[other]
        now=_latest(series,days,origin); cutoff=origin-dt.timedelta(days=7); prior=_latest(series,days,cutoff)
        if now is None or prior is None:
            values.extend((0.0,0.0)); availability[other]={"available":False,"currentDate":now.isoformat() if now else None,
                "priorDate":prior.isoformat() if prior else None,"reason":"current_or_prior_source_missing_or_stale"}
        else:
            change=math.log(series[now][0]/series[prior][0])
            if not math.isfinite(change): raise ValueError("nonfinite cross-crop log change")
            values.extend((change,1.0)); availability[other]={"available":True,"currentDate":now.isoformat(),"priorDate":prior.isoformat(),"logChange":change}
    if not all(math.isfinite(float(v)) for v in values): raise ValueError("nonfinite model feature")
    return source_day,tuple(values),availability


def _make_estimator():
    os.environ.setdefault("LOKY_MAX_CPU_COUNT","1")
    try: from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc: raise RuntimeError("scikit-learn is required") from exc
    return HistGradientBoostingRegressor(**MODEL_PARAMS)


def _season(day): return ("DJF","MAM","JJA","SON")[(day.month%12)//3]


def _metrics(rows,field): return dh._metrics([r["actual"] for r in rows],[r[field] for r in rows])


def build(prices,report,*,hashes=None,estimator_factory:Callable[[],Any]=_make_estimator):
    series_map=dh._observations(prices); indexed=_report_rows(report)
    rows=[dict(product=p,origin=o,target=t,horizon=h,**v) for (p,o,t,h),v in sorted(indexed.items()) if START<=t<=END]
    for r in rows:
        observed=series_map[r["product"]].get(r["target"])
        if observed is None or not math.isclose(observed[0],r["actual"],rel_tol=1e-12,abs_tol=0): raise ValueError(f"price/report actual mismatch at {r['product']}/{r['target']}")
    months=defaultdict(list)
    for r in rows: months[(r["origin"].year,r["origin"].month)].append(r)
    possible_fits=[min(r["origin"] for r in group) for group in months.values()]
    sample_start=min(possible_fits)-dt.timedelta(days=TRAIN_DAYS) if possible_fits else START-dt.timedelta(days=TRAIN_DAYS)
    sample_end=max(possible_fits) if possible_fits else START
    caches={p:{} for p in PRODUCTS}; days_map={p:tuple(series_map[p]) for p in PRODUCTS}; all_samples=[]
    # Generate supervised samples only from each product's observed target; every horizon is pooled.
    for product in PRODUCTS:
        series=series_map[product]
        for target,(actual,_) in series.items():
            if target<sample_start or target>=sample_end: continue
            for h in HORIZONS:
                origin=target-dt.timedelta(days=h)
                feature=_features(product,origin,target,h,series_map,caches,days_map)
                if feature is None: continue
                source_day,vector,cross=feature; source_price=series[source_day][0]
                label=math.log(actual/source_price)
                if not math.isfinite(label) or source_price<=0 or actual<=0: raise ValueError("invalid training label/source price")
                sample={"product":product,"target":target,"origin":origin,"horizon":h,"sourceDate":source_day,
                        "sourcePrice":source_price,"actual":actual,"label":label,"features":vector,"cross":cross}
                all_samples.append(sample)
    predictions={}; fit_audit=[]
    for month,eval_rows in sorted(months.items()):
        f=min(r["origin"] for r in eval_rows); cutoff=f-dt.timedelta(days=TRAIN_DAYS)
        eval_feature_map={}
        for r in eval_rows:
            feature=_features(r["product"],r["origin"],r["target"],r["horizon"],series_map,caches,days_map)
            if feature is None:
                r["candidateMissingReason"]="own_feature_unavailable_or_stale"
                r["ownSourceDate"]=None; r["crossCropFeatures"]={}
            else:
                eval_feature_map[(r["product"],r["origin"],r["target"],r["horizon"])]=feature
                r["ownSourceDate"]=feature[0]; r["crossCropFeatures"]=feature[2]
        training=[s for s in all_samples if cutoff<=s["target"]<f]
        per_crop={p:[s for s in training if s["product"]==p] for p in PRODUCTS}
        counts={p:len(per_crop[p]) for p in PRODUCTS}
        entry={"fitMonth":f"{month[0]:04d}-{month[1]:02d}","fitOrigin":f.isoformat(),
            "trainTargetStart":min((s["target"] for s in training),default=None).isoformat() if training else None,
            "trainTargetEnd":max((s["target"] for s in training),default=None).isoformat() if training else None,
            "trainingRowsByProduct":counts,"distinctTargetsByProduct":{p:len({s["target"] for s in per_crop[p]}) for p in PRODUCTS},
            "weightSumByProduct":{},"trainingRows":len(training),"status":None}
        if any(counts[p]<MIN_CROP_ROWS for p in PRODUCTS):
            entry["status"]="insufficient_crop_training"
            for r in eval_rows:
                if "candidateMissingReason" not in r: r["candidateMissingReason"]="insufficient_crop_training"
            fit_audit.append(entry); continue
        weights=[]
        for s in training: weights.append(len(training)/(len(PRODUCTS)*counts[s["product"]]))
        for p in PRODUCTS: entry["weightSumByProduct"][p]=sum(w for s,w in zip(training,weights) if s["product"]==p)
        try:
            import numpy as np
            model=estimator_factory()
            model.fit(np.asarray([s["features"] for s in training],dtype=float),np.asarray([s["label"] for s in training],dtype=float),sample_weight=np.asarray(weights,dtype=float))
        except Exception as exc:
            entry["status"]="fit_failed"; entry["errorType"]=type(exc).__name__; entry["error"]=str(exc)
            for r in eval_rows:
                if "candidateMissingReason" not in r: r["candidateMissingReason"]="fit_failed"
            fit_audit.append(entry); continue
        staged={}
        for r in eval_rows:
            feat=eval_feature_map.get((r["product"],r["origin"],r["target"],r["horizon"]))
            if feat is None: r["candidateMissingReason"]="own_feature_unavailable_or_stale"; continue
            source_day,vector,cross=feat; source_price=series_map[r["product"]][source_day][0]
            try:
                log_ratio=float(model.predict(np.asarray([vector],dtype=float))[0])
                if not math.isfinite(log_ratio): r["candidateMissingReason"]="nonfinite_prediction"; continue
                price,_=modeling._inverse_log_ratio(source_price,log_ratio)
                if not math.isfinite(price) or price<=0: r["candidateMissingReason"]="invalid_inverse_price"; continue
                staged[(r["product"],r["origin"],r["target"],r["horizon"]) ]={"prediction":price,"predictedLogRatio":log_ratio,
                    "sourceDate":source_day,"sourcePrice":source_price,"crossCropFeatures":cross,
                    "fitMonth":entry["fitMonth"],"fitOrigin":f,"fitTrainTargetEnd":entry["trainTargetEnd"],
                    "trainingRowsByProduct":counts,"weightSumByProduct":entry["weightSumByProduct"]}
            except Exception as exc: r["candidateMissingReason"]=f"prediction_failed:{type(exc).__name__}"
        predictions.update(staged); entry["status"]="fitted" if len(staged)==len(eval_rows) else "fitted_with_prediction_gaps"
        fit_audit.append(entry)
    for r in rows:
        pred=predictions.get((r["product"],r["origin"],r["target"],r["horizon"]))
        if pred: r.update(pred); r["candidatePrediction"]=pred["prediction"]
        else: r["candidatePrediction"]=None
    paired=[]; coverage={}; comparisons=[]; strata=[]; macro={}
    for product in PRODUCTS:
        for h in HORIZONS:
            group=[r for r in rows if r["product"]==product and r["horizon"]==h]
            comp=[r for r in group if r["candidatePrediction"] is not None]
            coverage[f"{product}/{h}"]={"eligibleDirectReportRows":len(group),"candidatePairedCount":len(comp),
                "candidateCoverageNondeclining":len(comp)==len(group),"coverageLossDates":[r["target"].isoformat() for r in group if r["candidatePrediction"] is None],
                "missingOriginsTargets":[{"origin":r["origin"].isoformat(),"target":r["target"].isoformat(),
                    "reason":r.get("candidateMissingReason","candidate_missing"),"ownSourceDate":r["ownSourceDate"].isoformat() if r.get("ownSourceDate") else None,
                    "crossCropFeatures":r.get("crossCropFeatures",{})} for r in group if r["candidatePrediction"] is None],
                "crossCropFeatureAvailability":{"availableCounts":{p:sum(bool(r.get("crossCropFeatures",{}).get(p,{}).get("available")) for r in group) for p in PRODUCTS if p!=product},
                    "eligibleDirectReportRows":len(group)}}
            for comparator in COMPARATORS:
                for loss in ("mae","mape"):
                    diff={r["target"]:(abs(r["actual"]-r["candidatePrediction"])-abs(r["actual"]-r[comparator])) if loss=="mae" else 100*(abs(r["actual"]-r["candidatePrediction"])/r["actual"]-abs(r["actual"]-r[comparator])/r["actual"]) for r in comp}
                    comparisons.append({"product":product,"horizon":h,"comparator":comparator,"loss":loss,"n":len(comp),
                        "candidateMetric":_metrics(comp,"candidatePrediction")[loss],"comparatorMetric":_metrics(comp,comparator)[loss],
                        "dm":{str(b):dh._dm(diff,b) for b in BANDWIDTHS}})
            for name,fn in (("year",lambda d:str(d.year)),("season",_season)):
                buckets=defaultdict(list)
                for r in comp: buckets[fn(r["target"])].append(r)
                for value,bucket in buckets.items(): strata.append({"product":product,"horizon":h,"stratum":name,"value":value,"n":len(bucket),
                    "candidate":_metrics(bucket,"candidatePrediction"),"directRidge":_metrics(bucket,"directRidge"),"persistence":_metrics(bucket,"persistence")})
        per_method={}
        for name,field in (("candidate","candidatePrediction"),("directRidge","directRidge"),("persistence","persistence")):
            per_h={str(h):_metrics([r for r in rows if r["product"]==product and r["horizon"]==h and r["candidatePrediction"] is not None],field) for h in HORIZONS}
            per_method[name]={"byHorizon":per_h,"equal10HorizonMape":sum(v["mape"] for v in per_h.values())/10 if all(v["mape"] is not None for v in per_h.values()) else None,
                "equal10HorizonMae":sum(v["mae"] for v in per_h.values())/10 if all(v["mae"] is not None for v in per_h.values()) else None}
        macro[product]=per_method
    for b in BANDWIDTHS: dh._holm([c["dm"][str(b)] for c in comparisons])
    ix={(c["product"],c["horizon"],c["comparator"],c["loss"]):c for c in comparisons}; conclusions=[]
    for p in PRODUCTS:
        for h in HORIZONS:
            family=[ix[(p,h,c,l)] for c in COMPARATORS for l in LOSSES]; tests=[x["dm"]["9"] for x in family]
            available=all(t["status"]=="tested" and t["n"]>=30 for t in tests)
            lower=all(x["candidateMetric"] is not None and x["comparatorMetric"] is not None and x["candidateMetric"]<x["comparatorMetric"] for x in family)
            sig=all(t.get("holmSignificant") and t.get("meanLossDifference") is not None and t["meanLossDifference"]<0 for t in tests)
            cov=coverage[f"{p}/{h}"]["candidateCoverageNondeclining"]
            adverse=any(t.get("holmSignificant") and t.get("meanLossDifference") is not None and t["meanLossDifference"]>0 for t in tests)
            status="inference_unavailable" if not available else "improvement" if lower and sig and cov else "worsening_or_mixed" if adverse else "inconclusive"
            conclusions.append({"product":p,"horizon":h,"conclusion":status,"inferenceAvailable":available,"allFourPairedMetricsImprove":lower,
                "allFourHac9HolmTestsImprove":sig,"coverageNondeclining":cov})
    goals={"below10PercentCount":sum(macro[p]["candidate"]["equal10HorizonMape"] is not None and macro[p]["candidate"]["equal10HorizonMape"]<10 for p in PRODUCTS),
        "below7PercentCount":sum(macro[p]["candidate"]["equal10HorizonMape"] is not None and macro[p]["candidate"]["equal10HorizonMape"]<7 for p in PRODUCTS)}
    goals["goalMet"]=goals["below10PercentCount"]>=3 or goals["below7PercentCount"]>=2
    paired_out=[]
    for r in rows:
        if r["candidatePrediction"] is None: continue
        paired_out.append({**r,"origin":r["origin"].isoformat(),"target":r["target"].isoformat(),"sourceDate":r["sourceDate"].isoformat(),
            "ownSourceDate":r["ownSourceDate"].isoformat(),"fitOrigin":r["fitOrigin"].isoformat(),
            "crossCropFeatures":{p:{**v} for p,v in r["crossCropFeatures"].items()}})
    return {"schemaVersion":1,"protocol":{"id":"cross-crop-pooled-origin-sharing-2026-10-07","inputHashes":{"prices":PRICES_SHA256,"directReport":REPORT_SHA256},
        "targetInterval":[START.isoformat(),END.isoformat()],"horizons":list(HORIZONS),"features":"own modeling._feature_row + product one-hot + other-crop 7-day log changes and availability flags",
        "otherCropSourceAgeDays":SOURCE_MAX_AGE,"trainingWindowDays":TRAIN_DAYS,"trainingTargetCutoff":"target < month first eligible test origin",
        "minimumTrainingRowsPerProduct":MIN_CROP_ROWS,"equalTotalProductWeighting":"total rows / (5 * product row count)","model":"HistGradientBoostingRegressor","parameters":MODEL_PARAMS,
        "comparators":list(COMPARATORS),"losses":list(LOSSES),"dmFamilySize":FAMILY_SIZE,"hacCalendarDays":list(BANDWIDTHS),"holmAlpha":.05,"minimumDmPairs":30},
        "hashes":hashes or {},"fitAudit":fit_audit,"pairedRows":paired_out,"coverage":coverage,"comparisons":comparisons,"conclusions":conclusions,
        "macroMetrics":macro,"yearSeasonMetrics":strata,"goals":goals,
        "limitation":"Retrospective evaluation of previously examined, revised historical data; not independent confirmation or official nationwide performance."}


def _read(path):
    b=Path(path).read_bytes()
    try: value=json.loads(b)
    except (UnicodeDecodeError,json.JSONDecodeError) as exc: raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value,dict): raise ValueError("input must be JSON object")
    return value,b


def _write_immutable(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f: fd=None; f.write(payload); f.flush(); os.fsync(f.fileno())
        os.link(tmp,path)
    except FileExistsError as exc: raise FileExistsError(f"refusing to overwrite immutable output: {path}") from exc
    finally:
        if fd is not None: os.close(fd)
        try: os.unlink(tmp)
        except FileNotFoundError: pass


def run(prices_path=INPUT_PRICES,report_path=INPUT_REPORT,output_path=None):
    if output_path is None: raise ValueError("output_path is required")
    prices,pb=_read(prices_path); report,rb=_read(report_path)
    hashes={"prices":hashlib.sha256(pb).hexdigest(),"directReport":hashlib.sha256(rb).hexdigest()}
    if hashes!={"prices":PRICES_SHA256,"directReport":REPORT_SHA256}: raise ValueError("input SHA-256 differs from frozen protocol")
    result=build(prices,report,hashes=hashes); payload=(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    _write_immutable(Path(output_path),payload); return {"output":str(output_path),"sha256":hashlib.sha256(payload).hexdigest()}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--prices",type=Path,default=INPUT_PRICES); p.add_argument("--report",type=Path,default=INPUT_REPORT); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv); run(a.prices,a.report,a.output); return 0


if __name__=="__main__": raise SystemExit(main())
