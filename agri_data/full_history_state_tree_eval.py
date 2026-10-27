"""Frozen nonlinear-state and leading-volume ablation experiment.

Protocol frozen before fitting: train actual target dates 2014-01-01..2025-12-31,
per crop and horizon (1..10), with two HistGradientBoostingRegressor arms using
loss=absolute_error, max_iter=160, learning_rate=.05, max_leaf_nodes=7,
min_samples_leaf=30, l2_regularization=1, early_stopping=False, random_state=0.
tree_price uses the prior full_history_return_eval relative-price features and
log(actual/p0) target; tree_volume adds p0-day volume and pre-p0 7/30-day
positive-volume medians, log ratios, counts and missing flags. All inputs stop
at origin-1. Planned 2026-01-01..09-27 rows with pre-2026 origins stay unissued.
The answer-free manifest is written and replayed before a separate score command
joins actuals. No API or database access occurs.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from . import full_history_return_eval as ridge

ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = ridge.PRICES_PATH
RIDGE_MANIFEST_PATH = ROOT / "outputs/experiments/full-history-return-20261009/forecast-manifest.json"
FIXED50_PATH = ridge.FIXED50_PATH
PRICES_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
RIDGE_MANIFEST_SHA256 = "86da40f5bcfb6669c5f0a4b34e9ad33c9c69d7f7492b85d58dd43bd90b119d45"
FIXED50_SHA256 = "1497ee362945fcab45353a7b2403be752af3bf63c5f584cc0aedb2cc111ceaa8"
RIDGE_HELPER_SHA256 = "276f389e7153e07ef02836183d55e9823a6eb7c9bf24acc882d663946c55c5c5"
PRODUCTS = ridge.PRODUCTS
HORIZONS = ridge.HORIZONS
TRAIN_START, TRAIN_END = ridge.TRAIN_START, ridge.TRAIN_END
EVAL_START, EVAL_END = ridge.EVAL_START, ridge.EVAL_END
MIN_TRAIN = ridge.MIN_TRAIN
BANDWIDTHS = ridge.BANDWIDTHS
ALPHA = ridge.ALPHA
COMPARISONS = ("tree_volume_vs_fixed50", "tree_volume_vs_tree_price", "tree_volume_vs_ridge", "tree_price_vs_ridge")
LOSSES = ("ae", "ape")


def _sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""): h.update(block)
    return h.hexdigest()


def _check_helper() -> None:
    path=Path(ridge.__file__)
    if _sha(path)!=RIDGE_HELPER_SHA256:
        raise ValueError("reviewed relative-Ridge helper hash mismatch")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def _volume_data(raw:dict[str,Any])->tuple[dict[str,dict[dt.date,float]],dict[str,Any]]:
    result={};audit={}
    for product in raw.get("products",[]):
        if not isinstance(product,dict) or product.get("id") not in PRODUCTS:continue
        pid=product["id"];values={};ambiguous=set();counts={"rows":0,"invalidDateRows":0,"nonpositiveOrInvalidVolumeRows":0,"duplicatePositiveVolumeDates":0}
        for row in product.get("observations",[]) if isinstance(product.get("observations"),list) else []:
            counts["rows"]+=1
            if not isinstance(row,dict):counts["invalidDateRows"]+=1;counts["nonpositiveOrInvalidVolumeRows"]+=1;continue
            day=ridge._date(row.get("date"));value=ridge._positive(row.get("volumeKg"))
            if day is None:counts["invalidDateRows"]+=1
            if value is None:counts["nonpositiveOrInvalidVolumeRows"]+=1
            if day is not None and value is not None:
                if day in values or day in ambiguous:
                    counts["duplicatePositiveVolumeDates"]+=1;values.pop(day,None);ambiguous.add(day)
                else:values[day]=value
        result[pid]=values;audit[pid]=counts
    if set(result)!=set(PRODUCTS):raise ValueError("volume source must contain all five products")
    return result,audit


def _median(rows:list[float])->float|None:
    return statistics.median(rows) if rows else None


def _volume_features(volumes:dict[dt.date,float],p0_date:dt.date,ordered:list[dt.date]|None=None)->list[float]:
    days=ordered if ordered is not None else sorted(volumes)
    end=bisect.bisect_left(days,p0_date)
    start7=bisect.bisect_left(days,p0_date-dt.timedelta(days=7),0,end)
    start30=bisect.bisect_left(days,p0_date-dt.timedelta(days=30),0,end)
    v0=volumes.get(p0_date); v7s=[volumes[d] for d in days[start7:end]];v30s=[volumes[d] for d in days[start30:end]]
    m7,m30=_median(v7s),_median(v30s)
    return [math.log(v0/m7) if v0 is not None and m7 else 0.,
            math.log(v0/m30) if v0 is not None and m30 else 0.,
            math.log(m7/m30) if m7 and m30 else 0.,
            float(v0 is not None and m7 is not None),float(v0 is not None and m30 is not None),float(m7 is not None and m30 is not None),
            float(len(v7s)),float(len(v30s))]


def _combined_features(prices:dict[dt.date,float],volumes:dict[dt.date,float],target:dt.date,horizon:int,
                       ordered_prices:list[dt.date]|None=None,ordered_volumes:list[dt.date]|None=None):
    price_features,p0,audit=ridge._features(prices,target,horizon,ordered_prices)
    if price_features is None or p0 is None:return None,None,None,audit
    p0_date=dt.date.fromisoformat(audit["p0Date"])
    volume_features=_volume_features(volumes,p0_date,ordered_volumes)
    return price_features,price_features+volume_features,p0,audit


def _grid()->list[tuple[str,dt.date,int]]:
    return [(p,EVAL_START+dt.timedelta(days=i),h) for p in PRODUCTS for i in range((EVAL_END-EVAL_START).days+1) for h in HORIZONS]


def _ridge_rows(raw:dict[str,Any])->dict[tuple[str,dt.date,int],dict[str,Any]]:
    if raw.get("phase")!="forecast_only" or raw.get("targetValuesAttached") is not False or not isinstance(raw.get("forecasts"),list):
        raise ValueError("pinned Ridge manifest must be answer-free")
    out={}
    for row in raw["forecasts"]:
        key=(row.get("product"),ridge._date(row.get("target")),row.get("horizon"))
        if key[0] not in PRODUCTS or key[1] is None or key[2] not in HORIZONS or key in out: raise ValueError("invalid or duplicate Ridge manifest key")
        out[key]=row
    return out


def _prior_ridge_coverage_audit(ridge_map:dict[tuple[str,dt.date,int],dict[str,Any]])->dict[str,Any]:
    out={}
    for product in PRODUCTS:
        missed={};count=0
        for (pid,day,h),row in ridge_map.items():
            if pid!=product or day is None or day-dt.timedelta(days=h)<EVAL_START:continue
            if ridge._positive(row.get("fixed50")) and not ridge._positive(row.get("prediction")):
                count+=1;missed[day.isoformat()]=row.get("reason") or "ridge_prediction_unavailable"
        out[product]={"eligibleFixed50RowsWithoutRidgePrediction":count,"uniqueTargetCount":len(missed),"targets":dict(sorted(missed.items()))}
    return out


def _fit_tree(X:list[list[float]],y:list[float]):
    from sklearn.ensemble import HistGradientBoostingRegressor
    model=HistGradientBoostingRegressor(loss="absolute_error",max_iter=160,learning_rate=.05,max_leaf_nodes=7,
        min_samples_leaf=30,l2_regularization=1,early_stopping=False,random_state=0)
    return model.fit(X,y)


def build(prices:dict[str,Any],ridge_manifest:dict[str,Any],fixed50:dict[str,Any],*,price_sha:str|None=None,ridge_sha:str|None=None,fixed50_sha:str|None=None)->dict[str,Any]:
    series=ridge._series(prices);volumes,volume_audit=_volume_data(prices)
    ridge_map=_ridge_rows(ridge_manifest); fixed_map=ridge._fixed50(fixed50)
    for key,row in ridge_map.items():
        if ridge._positive(row.get("fixed50"))!=fixed_map.get(key):raise ValueError("Ridge and separately pinned fixed50 rows disagree")
    rows=[]
    grid=_grid()
    for product in PRODUCTS:
        s=series[product]; ordered=sorted(s); volume=volumes[product]; ordered_volume=sorted(volume)
        for h in HORIZONS:
            xp=[]; xv=[]; ys=[]; train_dates=[]
            for day in sorted(s):
                if not TRAIN_START<=day<=TRAIN_END: continue
                pf,vf,p0,audit=_combined_features(s,volume,day,h,ordered,ordered_volume)
                if pf is None or p0 is None: continue
                target=s[day]
                xp.append(pf);xv.append(vf)
                ys.append(math.log(target/p0));train_dates.append(day)
            models={}; fit_reasons={}
            if len(ys)>=MIN_TRAIN:
                for arm,X in (("tree_price",xp),("tree_volume",xv)):
                    try: models[arm]=_fit_tree(X,ys)
                    except Exception as exc: fit_reasons[arm]=f"fit_failed:{type(exc).__name__}"
            pending=[]
            for prod,day,hh in (k for k in grid if k[0]==product and k[2]==h):
                pf,vf,p0,audit=_combined_features(s,volume,day,h,ordered,ordered_volume)
                origin=day-dt.timedelta(days=h); reason=None
                if origin<EVAL_START: reason="model_not_yet_fitted_at_origin"
                elif len(ys)<MIN_TRAIN: reason="fewer_than_90_valid_training_targets"
                elif pf is None or p0 is None: reason=audit.get("reason","stale_or_missing_price")
                pending.append({"product":product,"target":day.isoformat(),"origin":origin.isoformat(),"horizon":h,
                    "eligible":origin>=EVAL_START,"treePriceStatus":"not_issued" if reason or "tree_price" not in models else "issued",
                    "treePriceReason":reason or fit_reasons.get("tree_price"),"treePrice":None,
                    "treeVolumeStatus":"not_issued" if reason or "tree_volume" not in models else "issued",
                    "treeVolumeReason":reason or fit_reasons.get("tree_volume"),"treeVolume":None,
                    "persistence":p0 if audit.get("reason") is None else None,
                    "fixed50":fixed_map.get((product,day,h)),"ridge":(ridge_map.get((product,day,h)) or {}).get("prediction"),
                    "featureAudit":audit,"training":{"count":len(ys),"firstTarget":min(train_dates).isoformat() if train_dates else None,"lastTarget":max(train_dates).isoformat() if train_dates else None},
                    "_xPrice":pf,"_xVolume":vf,"_p0":p0})
            for arm,xkey,predkey,statuskey,reasonkey in (("tree_price","_xPrice","treePrice","treePriceStatus","treePriceReason"),("tree_volume","_xVolume","treeVolume","treeVolumeStatus","treeVolumeReason")):
                ready=[r for r in pending if r[statuskey]=="issued"]
                if not ready: continue
                try:
                    predictions=models[arm].predict([r[xkey] for r in ready])
                    if len(predictions)!=len(ready):raise ValueError("prediction_count_mismatch")
                    for row,logratio in zip(ready,predictions):
                        pred=row["_p0"]*math.exp(float(logratio))
                        if not math.isfinite(pred) or pred<=0: row[statuskey]="not_issued";row[reasonkey]="nonpositive_or_nonfinite_prediction"
                        else: row[predkey]=pred
                except Exception as exc:
                    for row in ready: row[statuskey]="not_issued";row[predkey]=None;row[reasonkey]=f"prediction_failed:{type(exc).__name__}"
            for row in pending:
                for key in ("_xPrice","_xVolume","_p0"): row.pop(key,None)
                rows.append(row)
    doc={"schemaVersion":1,"phase":"forecast_only","targetValuesAttached":False,
        "protocol":{"id":"full-history-state-tree-volume-ablation-v1","products":list(PRODUCTS),"horizons":list(HORIZONS),
          "trainTargets":[TRAIN_START.isoformat(),TRAIN_END.isoformat()],"evaluationTargets":[EVAL_START.isoformat(),EVAL_END.isoformat()],
          "minimumTrainingTargets":MIN_TRAIN,"arms":{"tree_price":{"features":"frozen full_history_return_eval price features","model":{"class":"HistGradientBoostingRegressor","loss":"absolute_error","max_iter":160,"learning_rate":.05,"max_leaf_nodes":7,"min_samples_leaf":30,"l2_regularization":1,"early_stopping":False,"random_state":0}},
             "tree_volume":{"features":"tree_price features plus p0-day volume and strictly earlier 7/30-day positive-volume medians, log ratios, counts, and masks","model":"identical parameters to tree_price"}},
          "target":"log(actual/p0)","featureCutoff":"origin-1 day; pre-2026 origins not issued","volumeWindows":"strictly before p0 date, calendar days [p0-7,p0) and [p0-30,p0)"},
        "inputSha256":{"prices":price_sha,"ridgeManifest":ridge_sha,"fixed50Manifest":fixed50_sha,"reviewedRidgeHelper":RIDGE_HELPER_SHA256},
        "sourceDataAudit":{"prices":ridge._source_audit(prices),"volume":volume_audit},"priorRidgeCoverageAudit":_prior_ridge_coverage_audit(ridge_map),"plannedForecastCount":len(rows),"forecasts":rows}
    doc["manifestSha256"]=hashlib.sha256(_canonical(doc)).hexdigest()
    return doc


def build_from_paths()->dict[str,Any]:
    _check_helper()
    for path,expected,label in ((PRICES_PATH,PRICES_SHA256,"prices"),(RIDGE_MANIFEST_PATH,RIDGE_MANIFEST_SHA256,"Ridge manifest"),(FIXED50_PATH,FIXED50_SHA256,"fixed50 manifest")):
        if _sha(path)!=expected: raise ValueError(f"pinned {label} hash mismatch")
    ridge_raw=json.loads(RIDGE_MANIFEST_PATH.read_text())
    if len(ridge_raw.get("forecasts",[]))!=len(_grid()): raise ValueError("pinned Ridge manifest does not contain the complete 13,500-row grid")
    return build(json.loads(PRICES_PATH.read_text()),ridge_raw,json.loads(FIXED50_PATH.read_text()),price_sha=PRICES_SHA256,ridge_sha=RIDGE_MANIFEST_SHA256,fixed50_sha=FIXED50_SHA256)


def verify_manifest(path:Path)->bool:
    raw=json.loads(path.read_text()); digest=raw.get("manifestSha256"); unsigned=dict(raw);unsigned.pop("manifestSha256",None)
    if digest!=hashlib.sha256(_canonical(unsigned)).hexdigest():return False
    return _canonical(raw)==_canonical(build_from_paths())


def _dm(diff:dict[dt.date,float],bandwidth:int)->dict[str,Any]:
    n=len(diff)
    if any(not math.isfinite(x) for x in diff.values()):return {"status":"nonfinite_loss_difference","reason":"nonfinite_loss_difference","n":n,"statistic":None,"meanLossDifference":None,"pValue":1.0}
    mean=math.fsum(diff.values())/n if n else None
    if n<30:return {"status":"insufficient_pairs","reason":"fewer_than_30_pairs","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.0}
    vals=list(diff.values());spread=max(vals)-min(vals);magnitude=max(abs(min(vals)),abs(max(vals)))
    if spread<=8*math.ulp(magnitude):return {"status":"zero_spread","reason":"constant_loss_difference_within_float_tolerance","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.0}
    centered={d:v-mean for d,v in diff.items()}; terms=[math.fsum(v*v for v in centered.values())]
    for lag in range(1,bandwidth+1):
        cross=math.fsum(centered[d]*centered.get(d+dt.timedelta(days=lag),0.) for d in centered)
        terms.append(2*(1-lag/(bandwidth+1))*cross)
    total=math.fsum(terms); tolerance=8*math.ulp(math.fsum(abs(x) for x in terms)) if any(terms) else 0.; variance=total/(n*n)
    if not math.isfinite(variance) or total<=tolerance:return {"status":"invalid_hac_variance","reason":"nonpositive_or_cancellation_dominated_hac_variance","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.0}
    stat=mean/math.sqrt(variance)
    return {"status":"tested","reason":None,"n":n,"statistic":stat,"meanLossDifference":mean,"pValue":math.erfc(abs(stat)/math.sqrt(2))}


def _holm(rows:list[dict[str,Any]])->None:
    ix=sorted(range(len(rows)),key=lambda i:rows[i]["pValue"]);running=0.;n=len(rows)
    for rank,i in enumerate(ix):running=max(running,min(1.,(n-rank)*rows[i]["pValue"]));rows[i]["holmPValue"]=running;rows[i]["holmSignificant"]=running<ALPHA


def _check_manifest(manifest:dict[str,Any])->None:
    if manifest.get("phase")!="forecast_only" or manifest.get("targetValuesAttached") is not False:raise ValueError("expected answer-free tree manifest")
    unsigned=dict(manifest);digest=unsigned.pop("manifestSha256",None)
    if digest!=hashlib.sha256(_canonical(unsigned)).hexdigest():raise ValueError("tree manifest hash mismatch")
    expected=set(_grid());seen=set()
    for row in manifest.get("forecasts",[]):
        key=(row.get("product"),ridge._date(row.get("target")),row.get("horizon"))
        if key not in expected or key in seen:raise ValueError("invalid or duplicate tree manifest key")
        day=key[1];h=key[2];origin=ridge._date(row.get("origin"))
        if origin!=day-dt.timedelta(days=h) or row.get("eligible") is not (origin>=EVAL_START):raise ValueError("forecast origin or eligibility mismatch")
        for pred_key,status_key in (("treePrice","treePriceStatus"),("treeVolume","treeVolumeStatus")):
            if row.get(status_key) not in ("issued","not_issued"):raise ValueError("invalid candidate status")
            if row[status_key]=="issued" and not ridge._positive(row.get(pred_key)):raise ValueError("issued forecast lacks a positive prediction")
            if row[status_key]=="not_issued" and row.get(pred_key) is not None:raise ValueError("not-issued forecast contains a prediction")
        seen.add(key)
    if seen!=expected:raise ValueError("incomplete tree forecast grid")


def score(manifest:dict[str,Any],prices:dict[str,Any])->dict[str,Any]:
    _check_manifest(manifest); series=ridge._series(prices)
    actuals={(p,d):v for p,rows in series.items() for d,v in rows.items() if EVAL_START<=d<=EVAL_END}
    rows=manifest["forecasts"]; lookup={(r["product"],ridge._date(r["target"]),r["horizon"]):r for r in rows}
    pairrows=[]; metrics=[]; dm=[];coverage={};temporal={}
    for p in PRODUCTS:
      for h in HORIZONS:
        cells=[lookup[(p,d,h)] for d in (EVAL_START+dt.timedelta(days=i) for i in range((EVAL_END-EVAL_START).days+1))]
        eligible=[r for r in cells if r["eligible"]]; fixed=[r for r in eligible if ridge._positive(r.get("fixed50"))]
        losses=[r["target"] for r in fixed if r["treeVolumeStatus"]!="issued" or not ridge._positive(r.get("treeVolume"))]
        common=[]
        for r in cells:
            day=ridge._date(r["target"]);actual=actuals.get((p,day))
            preds=[r.get("treeVolume") if r.get("treeVolumeStatus")=="issued" else None,r.get("treePrice") if r.get("treePriceStatus")=="issued" else None,r.get("ridge"),r.get("fixed50")]
            if r["eligible"] and actual is not None and all(ridge._positive(x) for x in preds):
                z={"product":p,"target":day.isoformat(),"horizon":h,"actual":actual,"treeVolume":preds[0],"treePrice":preds[1],"ridge":preds[2],"fixed50":preds[3]};common.append(z);pairrows.append(z)
        comps=(("tree_volume_vs_fixed50","treeVolume","fixed50"),("tree_volume_vs_tree_price","treeVolume","treePrice"),("tree_volume_vs_ridge","treeVolume","ridge"),("tree_price_vs_ridge","treePrice","ridge"))
        for name,a,b in comps:
            ys=[r["actual"] for r in common]
            def mt(k):
                if not common:return {"n":0,"mae":None,"mape":None}
                err=[abs(r[k]-r["actual"]) for r in common]
                return {"n":len(common),"mae":math.fsum(err)/len(err),"mape":100*math.fsum(e/r["actual"] for e,r in zip(err,common))/len(common)}
            metrics.append({"product":p,"horizon":h,"comparison":name,"a":mt(a),"b":mt(b)})
            for loss in LOSSES:
                diffs={ridge._date(r["target"]):(abs(r[a]-r["actual"])-abs(r[b]-r["actual"]) if loss=="ae" else abs(r[a]-r["actual"])/r["actual"]-abs(r[b]-r["actual"])/r["actual"]) for r in common}
                for bw in BANDWIDTHS:dm.append({"product":p,"horizon":h,"comparison":name,"loss":loss,"bandwidth":bw,**_dm(diffs,bw)})
        fixed_missing=sorted({r["target"] for r in eligible if not ridge._positive(r.get("fixed50"))})
        coverage[f"{p}/{h}"]={"planned":len(cells),"excludedPrefit":sum(not r["eligible"] for r in cells),"eligiblePlanned":len(eligible),
            "treePriceIssued":sum(bool(r["eligible"] and r["treePriceStatus"]=="issued" and ridge._positive(r.get("treePrice"))) for r in cells),
            "treeVolumeIssued":sum(bool(r["eligible"] and r["treeVolumeStatus"]=="issued" and ridge._positive(r.get("treeVolume"))) for r in cells),
            "fixed50AvailableEligible":len(fixed),"fixed50UnavailableEligibleTargetCount":len(fixed_missing),"fixed50UnavailableEligibleTargets":fixed_missing,
            "treeVolumeCoverageLossTargets":losses,"positiveActuals":sum((p,ridge._date(r["target"])) in actuals for r in eligible),"allFourCommonPairs":len(common)}
    for p in PRODUCTS:
      for period in ("month","quarter"):
        labels={}
        for i in range((EVAL_END-EVAL_START).days+1):
            day=EVAL_START+dt.timedelta(days=i);label=f"{day.year}-{day.month:02d}" if period=="month" else f"{day.year}-Q{(day.month-1)//3+1}"
            labels.setdefault(label,[]).append(day)
        for label,days in labels.items():
            cells=[lookup[(p,d,h)] for d in days for h in HORIZONS]
            temporal[f"{p}/{period}/{label}"]={"planned":len(cells),"excludedPrefit":sum(not r["eligible"] for r in cells),"eligiblePlanned":sum(r["eligible"] for r in cells),
                "treePriceIssued":sum(bool(r["eligible"] and r["treePriceStatus"]=="issued" and ridge._positive(r.get("treePrice"))) for r in cells),
                "treeVolumeIssued":sum(bool(r["eligible"] and r["treeVolumeStatus"]=="issued" and ridge._positive(r.get("treeVolume"))) for r in cells),
                "fixed50UnavailableEligibleTargetCount":len({r["target"] for r in cells if r["eligible"] and not ridge._positive(r.get("fixed50"))}),
                "positiveActualPairs":sum((p,d) in actuals for d in days)*len(HORIZONS),
                "fourModelFixed50Common":sum(1 for r in cells if r["eligible"] and (p,ridge._date(r["target"])) in actuals and all(ridge._positive(r.get(k)) for k in ("treeVolume","treePrice","ridge","fixed50")))}
    for bw in BANDWIDTHS:
        family=[r for r in dm if r["bandwidth"]==bw]
        if len(family)!=400: raise ValueError("frozen 400-test Holm family mismatch")
        _holm(family)
    macros=[];gates=[]
    for p in PRODUCTS:
        for name in COMPARISONS:
            group=[r for r in metrics if r["product"]==p and r["comparison"]==name and r["a"]["n"]]
            macros.append({"product":p,"comparison":name,"horizonsIncluded":len(group),"aEqualHorizonMae":math.fsum(r["a"]["mae"] for r in group)/len(group) if group else None,
                "bEqualHorizonMae":math.fsum(r["b"]["mae"] for r in group)/len(group) if group else None,"aEqualHorizonMape":math.fsum(r["a"]["mape"] for r in group)/len(group) if group else None,
                "bEqualHorizonMape":math.fsum(r["b"]["mape"] for r in group)/len(group) if group else None})
        group=[r for r in metrics if r["product"]==p and r["comparison"]=="tree_volume_vs_fixed50" and r["a"]["n"]]
        byh={r["horizon"]:r for r in group}; mape_a=math.fsum(r["a"]["mape"] for r in group)/len(group) if group else None;mape_b=math.fsum(r["b"]["mape"] for r in group)/len(group) if group else None
        mae_a=math.fsum(r["a"]["mae"] for r in group)/len(group) if group else None;mae_b=math.fsum(r["b"]["mae"] for r in group)/len(group) if group else None
        covloss_targets={}
        for h in HORIZONS:
            for target in coverage[f"{p}/{h}"]["treeVolumeCoverageLossTargets"]:
                source=lookup[(p,dt.date.fromisoformat(target),h)]
                covloss_targets[target]=source.get("treeVolumeReason") or "prediction_unavailable"
        covloss=bool(covloss_targets)
        no_harm=len(byh)==len(HORIZONS) and all(r["a"]["mape"]-r["b"]["mape"]<=1 for r in group)
        joint=False
        for h in HORIZONS:
            relevant={(r["loss"],r["bandwidth"]):r for r in dm if r["product"]==p and r["horizon"]==h and r["comparison"]=="tree_volume_vs_fixed50"}
            if all(relevant.get((loss,bw),{}).get("holmSignificant") and relevant[(loss,bw)].get("meanLossDifference",0)<0 for loss in LOSSES for bw in BANDWIDTHS):joint=True
        passes=bool(len(byh)==len(HORIZONS) and mape_a is not None and mape_b is not None and mape_a<=mape_b-.5 and mae_a<mae_b and not covloss and no_harm and joint)
        gates.append({"product":p,"candidate":"tree_volume","benchmark":"fixed50","all10HorizonsHaveCommonPairs":len(byh)==len(HORIZONS),"candidateEqualHorizonMape":mape_a,"fixed50EqualHorizonMape":mape_b,"mapeImprovementPctPoints":mape_b-mape_a if mape_a is not None and mape_b is not None else None,
            "candidateEqualHorizonMae":mae_a,"fixed50EqualHorizonMae":mae_b,"coverageNondeclining":not covloss,"candidateCoverageLossTargetCount":len(covloss_targets),"candidateCoverageLossTargetReasons":dict(sorted(covloss_targets.items())),"noHorizonMapeWorseByMoreThan1PctPoint":no_harm,
            "jointAeApeHolmImprovementAtSomeHorizonBothHac":joint,"decision":"eligible_for_prospective_confirmation" if passes else "retain_existing_method"})
    return {"schemaVersion":1,"phase":"scored","manifestSha256":manifest["manifestSha256"],"metrics":metrics,"macros":macros,"coverage":coverage,"temporalCoverage":temporal,"cropGates":gates,"dm":dm,"commonPairs":pairrows}


def write_exclusive(path:Path,value:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f:f.write(_canonical(value))


def main(argv:list[str]|None=None)->int:
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="cmd",required=True)
    b=sub.add_parser("build");b.add_argument("output",type=Path)
    v=sub.add_parser("verify");v.add_argument("manifest",type=Path)
    s=sub.add_parser("score");s.add_argument("manifest",type=Path);s.add_argument("output",type=Path)
    args=ap.parse_args(argv)
    if args.cmd=="build":write_exclusive(args.output,build_from_paths())
    elif args.cmd=="verify":
        if not verify_manifest(args.manifest):raise SystemExit("manifest replay mismatch")
    else:
        if not verify_manifest(args.manifest):raise SystemExit("manifest replay mismatch")
        write_exclusive(args.output,score(json.loads(args.manifest.read_text()),json.loads(PRICES_PATH.read_text())))
    return 0

if __name__=="__main__":raise SystemExit(main())
