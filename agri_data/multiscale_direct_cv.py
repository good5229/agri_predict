"""Chronological CV for a direct multiscale nonlinear 1–10 day price model."""
from __future__ import annotations
import argparse, datetime as dt, json, math, os, uuid
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from . import direct_horizon as dh, dashboard
from . import multiscale_residual as shared

ROOT=shared.ROOT
PRICES=shared.PRICES; DIRECT=shared.DIRECT; BLEND=shared.BLEND; BLEND_REPORT=shared.BLEND_REPORT
PRICES_SHA=shared.PRICES_SHA; DIRECT_SHA=shared.DIRECT_SHA; BLEND_SHA=shared.BLEND_SHA
BLEND_REPORT_SHA=shared.BLEND_REPORT_SHA
PRODUCTS=shared.PRODUCTS; HORIZONS=shared.HORIZONS; BANDWIDTHS=shared.BANDWIDTHS
TRAIN_END=dt.date(2025,2,28); VALIDATION_START=dt.date(2025,3,11); VALIDATION_END=dt.date(2025,8,31)
REFIT_END=dt.date(2025,8,31); EVAL_START=dt.date(2025,9,25); EVAL_END=dt.date(2026,9,24)
ALPHAS=(.001,.1,10.); HIDDEN=(24,12); SEED=1729; MAX_ITER=500; LOG_RATIO_LIMIT=.7
EVAL_PAIR_COUNT=12_508; MAX_SOURCE_AGE_DAYS=dashboard.MAX_SOURCE_AGE_DAYS
PROTOCOL="multiscale-direct-mlp-chronological-cv-2026-10-08"
METHODS=("candidate","ablation","direct","blend","persistence")

def _sha(raw:bytes)->str:return shared.sha(raw)
def _canonical(obj):return shared.canonical(obj)
def _code_sha():return _sha(Path(__file__).read_bytes())

def _feature(series,origin,target,h,season=True):
    history=sorted((day,row[0]) for day,row in series.items()
                   if day<=origin and row[0]>0 and math.isfinite(row[0]))
    if not history:return None
    source_day,anchor=history[-1]
    anchor_log=math.log(anchor)
    age=(origin-source_day).days
    values=[anchor_log]
    for window in (7,28,56):
        window_logs=[math.log(price) for day,price in history if 0<=(origin-day).days<window]
        values.extend((float(np.mean(window_logs)-anchor_log) if window_logs else 0.,
                       float(len(window_logs)),
                       float(np.std(window_logs)) if len(window_logs)>1 else 0.,
                       float(window_logs[-1]-window_logs[0]) if len(window_logs)>1 else 0.))
    weekday=target.weekday()*2*math.pi/7
    values.extend((float(age),float(h),math.sin(weekday),math.cos(weekday)))
    if season:
        year_length=366 if __import__("calendar").isleap(target.year) else 365
        annual=target.timetuple().tm_yday*2*math.pi/year_length
        values.extend((math.sin(annual),math.cos(annual)))
    return source_day,anchor,values,age

def _fit(X,y,alpha,season):
    matrix=np.asarray(X,dtype=float)
    if not season:matrix=matrix[:,:-2]
    model=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=HIDDEN,alpha=alpha,
        random_state=SEED,max_iter=MAX_ITER,early_stopping=False))
    model.fit(matrix,np.asarray(y,dtype=float))
    return model

def _predict(model,features,season):
    vector=np.asarray(features if season else features[:-2],dtype=float).reshape(1,-1)
    log_ratio=float(np.clip(model.predict(vector)[0],-LOG_RATIO_LIMIT,LOG_RATIO_LIMIT))
    return log_ratio

def _macro_horizon(rows,values,kind):
    groups={h:[] for h in HORIZONS}
    for row,value in zip(rows,values):groups[row[0][3]].append((row[1]["actual"],value))
    cell=[]
    for pairs in groups.values():
        if not pairs:continue
        if kind=="mape":cell.append(100*sum(abs(a-p)/a for a,p in pairs)/len(pairs))
        else:cell.append(sum(abs(a-p) for a,p in pairs)/len(pairs))
    return sum(cell)/len(cell) if cell else math.inf

def _select_alpha(train_rows,validation_rows):
    if not train_rows or not validation_rows:raise ValueError("empty_chronological_fit_or_validation_split")
    if {row[0][3] for row in train_rows}!=set(HORIZONS) or {row[0][3] for row in validation_rows}!=set(HORIZONS):
        raise ValueError("chronological_split_missing_horizon")
    X=[row[4] for row in train_rows]
    y=[math.log(row[1]["actual"]/row[3]) for row in train_rows]
    valX=[row[4] for row in validation_rows]
    results=[]
    for alpha in ALPHAS:
        model=_fit(X,y,alpha,True)
        pred=[anchor*math.exp(_predict(model,features,True)) for anchor,features in zip([r[3] for r in validation_rows],valX)]
        results.append({"alpha":alpha,"meanHorizonMape":_macro_horizon(validation_rows,pred,"mape"),
                        "meanHorizonMae":_macro_horizon(validation_rows,pred,"mae"),"validationRows":len(validation_rows)})
    selected=min(results,key=lambda r:(r["meanHorizonMape"],r["meanHorizonMae"],-r["alpha"]))
    return selected["alpha"],results

def build_forecast(prices_raw:bytes,direct_raw:bytes,blend_raw:bytes)->dict[str,Any]:
    if _sha(prices_raw)!=PRICES_SHA or _sha(direct_raw)!=DIRECT_SHA or _sha(blend_raw)!=BLEND_SHA:
        raise ValueError("pinned input SHA mismatch")
    raw=json.loads(prices_raw); report=json.loads(direct_raw); blend_doc=json.loads(blend_raw)
    series=dh._observations(raw); pairs=shared._pairs(report,include_all_actuals=False); blends=shared._source_rows(blend_doc)
    training={p:[] for p in PRODUCTS}; validation={p:[] for p in PRODUCTS}; fit_pool={p:[] for p in PRODUCTS}; evaluation=[]
    for k,row in pairs.items():
        product,origin,target,h=k
        if product not in series:continue
        feature=_feature(series[product],origin,target,h,True)
        record=None if feature is None else (k,row,*feature)
        if target<=REFIT_END and origin<target and row["actual"]>0 and record is not None:
            fit_pool[product].append(record)
        if target<=TRAIN_END and origin<target:
            if row["actual"]>0 and record is not None:training[product].append(record)
        elif VALIDATION_START<=target<=VALIDATION_END and origin>TRAIN_END:
            if row["actual"]>0 and record is not None:validation[product].append(record)
        elif EVAL_START<=target<=EVAL_END and k in blends:
            if feature is None:evaluation.append((k,row,None,None,None,None))
            else:evaluation.append(record)
    if len(evaluation)!=EVAL_PAIR_COUNT:raise ValueError(f"evaluation_pair_count_mismatch:{len(evaluation)}")
    alpha_meta={}; fitted={}; fitted_ablation={}; fit_counts={}
    for product in PRODUCTS:
        tr=training[product]; va=validation[product]
        if not tr or not va:raise ValueError(f"empty_cv_rows:{product}")
        alpha,cv=_select_alpha(tr,va); alpha_meta[product]={"selectedAlpha":alpha,"candidates":cv,
            "trainingRows":len(tr),"validationRows":len(va),"trainingTargetEnd":TRAIN_END.isoformat(),
            "validationTargetStart":VALIDATION_START.isoformat(),"validationTargetEnd":VALIDATION_END.isoformat()}
        refit=fit_pool[product]
        X=[r[4] for r in refit]; y=[math.log(r[1]["actual"]/r[3]) for r in refit]
        if not X:raise ValueError(f"empty_refit_rows:{product}")
        fitted[product]=_fit(X,y,alpha,True); fitted_ablation[product]=_fit(X,y,alpha,False)
        fit_counts[product]=len(refit)
    forecasts=[]
    for k,row,source_day,anchor,features,age in sorted(evaluation,key=lambda x:x[0]):
        fallback=age is None or age>MAX_SOURCE_AGE_DAYS
        if age is None:
            candidate=ablation=row["direct"]; reason="no_valid_asof_price_history"
        elif fallback:
            candidate=ablation=row["direct"]; reason="latest_source_older_than_14_days"
        else:
            candidate=anchor*math.exp(_predict(fitted[k[0]],features,True))
            ablation=anchor*math.exp(_predict(fitted_ablation[k[0]],features,False)); reason=None
        forecasts.append({"product":k[0],"origin":k[1].isoformat(),"target":k[2].isoformat(),"horizon":k[3],
            "predictions":{"candidate":candidate,"ablation":ablation,"direct":row["direct"],"blend":blends[k],"persistence":row["persistence"]},
            "lineage":{"sourceDate":source_day.isoformat() if source_day else None,"sourceAgeDays":age,"fallback":fallback,"fallbackReason":reason}})
    doc={"schemaVersion":1,"protocol":PROTOCOL,"phase":"forecast_only","targetValuesAttached":False,
        "inputSha256":{"prices":PRICES_SHA,"direct":DIRECT_SHA,"blendManifest":BLEND_SHA},"codeSha256":_code_sha(),
        "trainingTargetEnd":TRAIN_END.isoformat(),"refitTargetEnd":REFIT_END.isoformat(),
        "validation":{"start":VALIDATION_START.isoformat(),"end":VALIDATION_END.isoformat(),"earliestOriginAfterTraining":True},
        "model":{"type":"direct pooled-horizon MLP","features":"latest log price; causal 7/28/56-day deviations, changes, volatility, counts, source age; horizon, target weekday, annual phase",
            "hiddenLayerSizes":list(HIDDEN),"alphaCandidates":list(ALPHAS),"randomState":SEED,"maxIter":MAX_ITER,"earlyStopping":False,"logRatioLimit":LOG_RATIO_LIMIT},
        "alphaSelection":alpha_meta,"refitRowsByProduct":fit_counts,"fallbackCount":sum(x["lineage"]["fallback"] for x in forecasts),"forecasts":forecasts}
    doc["manifestSha256"]=_canonical(doc);return doc

def _verify_manifest(m,raw):
    if m.get("protocol")!=PROTOCOL or m.get("phase")!="forecast_only" or m.get("targetValuesAttached") is not False:
        raise ValueError("manifest_protocol_or_phase_invalid")
    if m.get("codeSha256")!=_code_sha():raise ValueError("manifest_code_hash_mismatch")
    clean=dict(m);saved=clean.pop("manifestSha256",None)
    if saved!=_canonical(clean) or _sha(raw)!=_sha((json.dumps(m,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()):
        raise ValueError("manifest_hash_or_bytes_mismatch")
    rows={}
    for row in m.get("forecasts",[]):
        if "actual" in row or "targetValue" in row:raise ValueError("answer_leakage")
        k=shared.key(row)
        if k in rows:raise ValueError("duplicate_manifest_identity")
        if set(row.get("predictions",{}))!=set(METHODS):raise ValueError("manifest_methods_invalid")
        for value in row["predictions"].values():
            if not math.isfinite(float(value)) or float(value)<=0:raise ValueError("manifest_prediction_invalid")
        rows[k]=row
    return rows

def score(manifest_raw:bytes,prices_raw:bytes,direct_raw:bytes,blend_raw:bytes,blend_report_raw:bytes)->dict[str,Any]:
    if _sha(prices_raw)!=PRICES_SHA or _sha(direct_raw)!=DIRECT_SHA or _sha(blend_raw)!=BLEND_SHA or _sha(blend_report_raw)!=BLEND_REPORT_SHA:
        raise ValueError("pinned input SHA mismatch")
    blend_report=json.loads(blend_report_raw); blend_doc=json.loads(blend_raw)
    if blend_report.get("manifestSha256")!=blend_doc.get("manifestSha256"):raise ValueError("blend_report_manifest_lineage_mismatch")
    manifest=json.loads(manifest_raw); forecasts=_verify_manifest(manifest,manifest_raw)
    expected=build_forecast(prices_raw,direct_raw,blend_raw)
    if manifest!=expected:raise ValueError("forecast_manifest_differs_from_deterministic_reconstruction")
    pairs=shared._pairs(json.loads(direct_raw)); blends=shared._source_rows(blend_doc); observations=dh._observations(json.loads(prices_raw))
    actuals={(p,day):value[0] for p,series in observations.items() for day,value in series.items() if value[0]>0 and math.isfinite(value[0])}
    common={}
    for k,row in forecasts.items():
        if k not in pairs or k not in blends:continue
        actual=pairs[k]["actual"]; snap=actuals.get((k[0],k[2]))
        if snap is None or not math.isclose(snap,actual,rel_tol=1e-12,abs_tol=1e-12):raise ValueError("actual_mismatch_against_frozen_prices")
        pred=row["predictions"]
        if not math.isclose(pred["direct"],pairs[k]["direct"],rel_tol=1e-12,abs_tol=1e-12) or not math.isclose(pred["blend"],blends[k],rel_tol=1e-12,abs_tol=1e-12):
            raise ValueError("baseline_prediction_mismatch")
        if not math.isclose(pred["persistence"],pairs[k]["persistence"],rel_tol=1e-12,abs_tol=1e-12):raise ValueError("persistence_prediction_mismatch")
        common[k]={"actual":actual,"predictions":pred}
    if len(common)!=EVAL_PAIR_COUNT:raise ValueError(f"common_pair_count_mismatch:{len(common)}")
    metrics={}
    for method in METHODS:
        cells={}
        for product in PRODUCTS:
            for horizon in HORIZONS:
                rows=[row for k,row in common.items() if k[0]==product and k[3]==horizon]; n=len(rows)
                cells[f"{product}|{horizon}"]={"n":n,"mae":sum(abs(r["actual"]-r["predictions"][method]) for r in rows)/n if n else None,
                    "mape":100*sum(abs(r["actual"]-r["predictions"][method])/r["actual"] for r in rows)/n if n else None}
        by_product={}
        for product in PRODUCTS:
            rows=[r for k,r in common.items() if k[0]==product]; n=len(rows)
            by_product[product]={"n":n,"mae":sum(abs(r["actual"]-r["predictions"][method]) for r in rows)/n if n else None,
                "mape":100*sum(abs(r["actual"]-r["predictions"][method])/r["actual"] for r in rows)/n if n else None}
        metrics[method]={"cells":cells,"byProduct":by_product,"macro50CellMae":sum(x["mae"] for x in cells.values())/50 if len(cells)==50 and all(x["mae"] is not None for x in cells.values()) else None,
            "macro50CellMape":sum(x["mape"] for x in cells.values())/50 if len(cells)==50 and all(x["mape"] is not None for x in cells.values()) else None,"coverage":len(common)}
    tests=[]
    for comparator in ("ablation","direct","blend","persistence"):
        for product in PRODUCTS:
            for horizon in HORIZONS:
                rows={k:r for k,r in common.items() if k[0]==product and k[3]==horizon}
                for loss in ("ae","ape"):
                    differences=shared._lossdiff(rows,"candidate",comparator,loss)
                    for bandwidth in BANDWIDTHS:
                        result=shared._dm(differences,bandwidth)
                        tests.append({"candidate":"candidate","comparator":comparator,"product":product,"horizon":horizon,"loss":loss,
                            "bandwidth":bandwidth,"meanLossDifference":sum(v for _,v in differences)/len(differences) if differences else None,**result})
    for bandwidth in BANDWIDTHS:shared._holm([x for x in tests if x["bandwidth"]==bandwidth])
    grouped={}
    for item in tests:grouped.setdefault((item["comparator"],item["product"],item["horizon"]),[]).append(item)
    conclusions=[]
    for (comparator,product,horizon),items in grouped.items():
        states=[item["conclusion"] for item in items]
        conclusion="improvement" if len(states)==4 and all(x=="improvement" for x in states) else "worsening" if len(states)==4 and all(x=="worsening" for x in states) else "inference_unavailable" if "inference_unavailable" in states else "inconclusive"
        conclusions.append({"comparator":comparator,"product":product,"horizon":horizon,"conclusion":conclusion})
    return {"protocol":PROTOCOL,"manifestSha256":manifest["manifestSha256"],"inputSha256":{"prices":PRICES_SHA,"direct":DIRECT_SHA,"blendManifest":BLEND_SHA,"blendReport":BLEND_REPORT_SHA},
        "coverage":{"planned":EVAL_PAIR_COUNT,"manifestForecasts":len(forecasts),"commonScoredPairs":len(common),"fallbackForecasts":manifest["fallbackCount"]},
        "metrics":metrics,"dmTests":tests,"cellConclusions":conclusions,
        "interpretation":"Retrospective, repeatedly inspected revised-price study; nonsignificance is not equivalence or proof of operational benefit."}

def _immutable(path:Path,doc):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x",encoding="utf-8") as stream:
            stream.write(json.dumps(doc,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n");stream.flush();os.fsync(stream.fileno())
        os.link(temp,path)
    finally:
        try:temp.unlink()
        except FileNotFoundError:pass

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("phase",choices=("forecast","score"))
    parser.add_argument("--prices",type=Path,default=PRICES);parser.add_argument("--direct",type=Path,default=DIRECT)
    parser.add_argument("--blend",type=Path,default=BLEND);parser.add_argument("--blend-report",type=Path,default=BLEND_REPORT)
    parser.add_argument("--manifest",type=Path,required=True);parser.add_argument("--report",type=Path)
    args=parser.parse_args(argv);prices=args.prices.read_bytes();direct=args.direct.read_bytes();blend=args.blend.read_bytes()
    if args.phase=="forecast":_immutable(args.manifest,build_forecast(prices,direct,blend))
    else:
        if not args.report:parser.error("--report required for score phase")
        _immutable(args.report,score(args.manifest.read_bytes(),prices,direct,blend,args.blend_report.read_bytes()))
    return 0
if __name__=="__main__":raise SystemExit(main())
