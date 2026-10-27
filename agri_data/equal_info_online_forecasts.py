"""Same-information monthly online price forecast experiment (answer-free).

Frozen protocol: use the pinned five-crop price snapshot and 2023/24 selected
71-feature Ridge alphas (cabbage 100; other crops 1000). For every product,
horizon 1..10 and monthly first eligible origin, fit both the 71-feature Ridge
and the narrow direct Ridge on labels with target < fit origin. Every feature
row uses only observations through its own origin-1. Refit once monthly; update
forecast features daily. Compare against a causally refit alpha-10 direct
Ridge, calendar SARIMA(1,1,1)(1,0,0)[7] on the preceding 730 calendar days
ending origin-1, its fixed 50:50 blend with the direct model, a 50:50 blend of
online71 with the same SARIMA, and persistence. SARIMA forecasts h+1 calendar
steps from origin-1; missing dates remain NaN. A failed SARIMA blend falls
back to its Ridge component with provenance. The 2025Q4 confirmation and
2026-01-01..09-27 evaluation full grids are answer-free. This module does not
read target answers for either forecast grid and never accesses an API.
"""
from __future__ import annotations
import argparse, copy, datetime as dt, hashlib, json, math
from pathlib import Path
from typing import Any, Callable

from . import full_history_return_eval as narrow
from . import full_history_xgb_rf_eval as rich
from . import full_history_feature_library as library
from . import full_history_hyperparam_forecasts as hp
from . import ten_strategy_benchmark as ten
from . import modeling, direct_horizon

ROOT=Path(__file__).resolve().parents[1]
PRICES_PATH=hp.PRICES_PATH
FIXED50_PATH=hp.FIXED50_PATH
HYPERPARAM_PATH=hp.OUT/"forecast-manifest.json"
OUT=ROOT/"outputs/experiments/equal-info-online-20261010"
PRICES_SHA256=hp.PRICES_SHA256
FIXED50_SHA256=hp.FIXED50_SHA256
HYPERPARAM_SHA256="9d08211d7ad3b207d9ae182360864ca06d98da17ad85e2b9bdffed3384eff0cc"
RICH_HELPER_SHA256=hp.RICH_HELPER_SHA256
NARROW_HELPER_SHA256=hp.NARROW_HELPER_SHA256
LIBRARY_SHA256=hp.FEATURE_LIBRARY_SHA256
TEN_HELPER_SHA256="e0518a826423953fbc8dee09a61c0463fdc1b469a97727a1f8f4ac4d5eb81460"
MODELING_SHA256="0ef149c231cc19f5990548822c10afecb16957863eddbd436d7d61a06c9963fc"
DIRECT_HELPER_SHA256="1cf3f8ce5455d9ce1f8d4eeb6e14f84e465acf4f5e6d512670b102115fb96d40"
PRODUCTS=narrow.PRODUCTS; HORIZONS=narrow.HORIZONS
CONF_START,CONF_END=dt.date(2025,10,1),dt.date(2025,12,31)
EVAL_START,EVAL_END=narrow.EVAL_START,narrow.EVAL_END
MIN_TRAIN=narrow.MIN_TRAIN; SARIMA_DAYS=730
SELECTED_ALPHA={"cabbage":100.0,"radish":1000.0,"onion":1000.0,"garlic":1000.0,"pepper":1000.0}
METHODS=("online71","causalDirect","causalSarima","causal50","online71Blend","persistence","fixed50")

def _canonical(v:Any)->bytes:return (json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def _sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def _positive(v:Any)->float|None:
    try:x=float(v)
    except (TypeError,ValueError,OverflowError):return None
    return x if math.isfinite(x) and x>0 else None
def _grid(start:dt.date,end:dt.date):
    return [(p,start+dt.timedelta(days=i),h) for p in PRODUCTS for i in range((end-start).days+1) for h in HORIZONS]
def _records(prices:dict[str,Any]):return rich._raw_maps(prices)

def _check_inputs()->None:
    for path,digest,label in ((PRICES_PATH,PRICES_SHA256,"prices"),(FIXED50_PATH,FIXED50_SHA256,"fixed50"),(HYPERPARAM_PATH,HYPERPARAM_SHA256,"hyperparameter manifest")):
        if _sha(path)!=digest:raise ValueError(f"pinned {label} SHA-256 mismatch")
    if (_sha(Path(rich.__file__))!=RICH_HELPER_SHA256 or _sha(Path(narrow.__file__))!=NARROW_HELPER_SHA256 or _sha(Path(library.__file__))!=LIBRARY_SHA256
        or _sha(Path(modeling.__file__))!=MODELING_SHA256 or _sha(Path(direct_horizon.__file__))!=DIRECT_HELPER_SHA256):
        raise ValueError("pinned feature helper hash mismatch")
    if _sha(Path(ten.__file__))!=TEN_HELPER_SHA256:raise ValueError("pinned SARIMA helper hash mismatch")
    try:import statsmodels
    except Exception as exc:raise RuntimeError("statsmodels==0.14.0 required") from exc
    if statsmodels.__version__!="0.14.0":raise RuntimeError("statsmodels==0.14.0 required")

def _alpha_from_manifest(raw):
    if raw.get("phase")!="forecast_only" or raw.get("targetValuesAttached") is not False:raise ValueError("invalid answer-free hyperparameter manifest")
    out={}
    for p in PRODUCTS:
        chosen=raw.get("selection",{}).get(p,{}).get("ridge",{}).get("chosenConfig")
        alpha=_positive(chosen.get("alpha")) if isinstance(chosen,dict) else None
        if alpha is None:raise ValueError(f"missing selected Ridge alpha for {p}")
        out[p]=alpha
    if out!=SELECTED_ALPHA:raise ValueError("frozen 2023/24 selected alpha values changed")
    return out

def _fit_ridge(X,y,alpha):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    return make_pipeline(StandardScaler(),Ridge(alpha=alpha,fit_intercept=True)).fit(X,y)

def _model_row(status,reason=None,prediction=None,training=None,provenance=None):
    return {"status":status,"reason":reason,"prediction":prediction,"training":training,"provenance":provenance or {}}

def _blend_ridge_sarima(ridge,sarima):
    if ridge["status"]!="issued":
        return _model_row("not_issued",ridge["reason"] or "ridge_component_unavailable",provenance={"weights":[.5,.5],"fallback":False})
    if sarima["status"]=="issued":
        return _model_row("issued",prediction=.5*ridge["prediction"]+.5*sarima["prediction"],provenance={"weights":[.5,.5],"fallback":False})
    return _model_row("issued",prediction=ridge["prediction"],provenance={"weights":[.5,.5],"fallback":True,"fallbackReason":sarima["reason"]})

def _direct_feature(series,target,h,cache):
    origin=target-dt.timedelta(days=h);cutoff=origin-dt.timedelta(days=1)
    found=modeling._feature_row(series,cutoff,target,h,cache)
    if found is None:return None,None,{"reason":"feature_unavailable_or_source_too_old","cutoff":cutoff.isoformat()}
    source,vector=found
    age=(origin-source).days
    if age>14:return None,None,{"reason":"p0_older_than_14_days","p0Date":source.isoformat(),"p0AgeDays":age,"cutoff":cutoff.isoformat()}
    return list(vector),series[source][0],{"p0Date":source.isoformat(),"p0AgeDays":age,"cutoff":cutoff.isoformat()}

def _latest_pre_origin_source(records,origin):
    prior=[day for day in records if day<=origin-dt.timedelta(days=1)]
    source=max(prior) if prior else None
    return source,(origin-source).days if source else None

def _period_fit(records,direct_series,product,h,fit_origin,alpha,kind):
    rec=records[product];ordered=sorted(rec);prices={d:r["price"] for d,r in rec.items()}
    if kind=="direct":
        series=direct_series[product];cache={};X=[];y=[];targets=[]
        for target,(actual,_) in series.items():
            if target>=fit_origin or target<narrow.TRAIN_START:continue
            found=modeling._feature_row(series,target-dt.timedelta(days=h+1),target,h,cache)
            if found is None:continue
            source,vector=found
            if (target-dt.timedelta(days=h)-source).days>14:continue
            X.append(list(vector));y.append(math.log(actual/series[source][0]));targets.append(target)
        audit={"count":len(X),"firstTarget":targets[0].isoformat() if targets else None,"lastTarget":targets[-1].isoformat() if targets else None,"fitOrigin":fit_origin.isoformat()}
        if len(X)<MIN_TRAIN:return None,audit,"fewer_than_90_valid_training_targets"
        try:return _fit_ridge(X,y,alpha),audit,None
        except Exception as exc:return None,audit,f"fit_failed:{type(exc).__name__}"
    X=[];y=[];targets=[]
    for target in ordered:
        if target>=fit_origin or target<narrow.TRAIN_START:continue
        feat,p0,audit=library.features(rec,target,h,ordered,prices)
        actual=prices[target]
        if feat is None or p0 is None:continue
        X.append(feat);y.append(math.log(actual/p0));targets.append(target)
    audit={"count":len(X),"firstTarget":targets[0].isoformat() if targets else None,"lastTarget":targets[-1].isoformat() if targets else None,"fitOrigin":fit_origin.isoformat()}
    if len(X)<MIN_TRAIN:return None,audit,"fewer_than_90_valid_training_targets"
    try:return _fit_ridge(X,y,alpha),audit,None
    except Exception as exc:return None,audit,f"fit_failed:{type(exc).__name__}"

def _fit_sarima(series,origin):
    start=origin-dt.timedelta(days=SARIMA_DAYS);end=origin-dt.timedelta(days=1)
    fit=ten._fit_calendar(series,start,end,"sarima",1)
    return fit,{"trainStart":start.isoformat(),"trainEnd":end.isoformat(),"trainingCalendarDays":SARIMA_DAYS,"order":[1,1,1],"seasonalOrder":[1,0,0,7],"selectedD":1}

def _forecast_origin_month(records,direct_series,fixed_map,alpha,phase,start,end,month,progress=None):
    grid=[k for k in _grid(start,end) if (k[1]-dt.timedelta(days=k[2])).year==month[0] and (k[1]-dt.timedelta(days=k[2])).month==month[1]];by_key={};total=0
    for product in PRODUCTS:
      rec=records[product];ordered=sorted(rec);prices={d:r["price"] for d,r in rec.items()};series={d:r["price"] for d,r in rec.items()}
      product_month_targets=[k for k in grid if k[0]==product]
      month_eligible_origins=[k[1]-dt.timedelta(days=k[2]) for k in product_month_targets if k[1]-dt.timedelta(days=k[2])>=start]
      month_first_origin=min(month_eligible_origins) if month_eligible_origins else None
      initial_sarima=None;month_sarima_audit={};month_sarima_error=None
      if month_first_origin is not None:
          try:initial_sarima,month_sarima_audit=_fit_sarima(series,month_first_origin)
          except Exception as exc:month_sarima_error=f"sarima_fit_failed:{type(exc).__name__}:{exc}"
      for h in HORIZONS:
        targets=[(p,d,hh) for p,d,hh in grid if p==product and hh==h]
        eligible=[(p,d,hh) for p,d,hh in targets if d-dt.timedelta(days=h)>=start]
        if not eligible:
            for _,day,_ in targets:
                origin=day-dt.timedelta(days=h);by_key[(product,day,h)]=_base_row(product,day,h,origin,start,phase)
            continue
        first_origin=min(k[1]-dt.timedelta(days=h) for k in eligible)
        fit71,audit71,err71=_period_fit(records,direct_series,product,h,first_origin,alpha[product],"online71")
        fitdirect,auditdirect,errdirect=_period_fit(records,direct_series,product,h,first_origin,10.0,"direct")
        direct_feature_cache={}
        fit_sarima=copy.deepcopy(initial_sarima) if initial_sarima is not None else None
        sarima_audit=month_sarima_audit;sarima_error=month_sarima_error;state_day=month_first_origin-dt.timedelta(days=1) if fit_sarima is not None else None
        for _,day,_ in targets:
            origin=day-dt.timedelta(days=h);eligible_origin=origin>=start
            key=(product,day,h)
            if not eligible_origin:
                by_key[key]=_base_row(product,day,h,origin,start,phase);continue
            # Advance SARIMA state only through origin-1, appending actual observation or NaN for each calendar day.
            state_error=None
            if fit_sarima is not None:
                while state_day is not None and state_day<origin-dt.timedelta(days=1):
                    d=state_day+dt.timedelta(days=1)
                    try:fit_sarima=fit_sarima.append(ten._calendar_values(series,d,d),refit=False);state_day=d
                    except Exception as exc:state_error=f"sarima_state_update_failed:{type(exc).__name__}";break
            feature71,p071,a71=library.features(rec,day,h,ordered,prices)
            featureD,p0d,aD=_direct_feature(direct_series[product],day,h,direct_feature_cache)
            source_date,age=_latest_pre_origin_source(rec,origin)
            training71={**audit71,"alpha":alpha[product],"featureCount":71}
            trainingD={**auditdirect,"alpha":10.0,"featureCount":len(modeling.FEATURE_NAMES)}
            pr71=_model_row("not_issued",err71 or a71.get("reason"),training=training71,provenance={"sourceDate":source_date.isoformat() if source_date else None,"sourceAgeDays":age,"featureAudit":a71})
            prD=_model_row("not_issued",errdirect or aD.get("reason"),training=trainingD,provenance={"sourceDate":source_date.isoformat() if source_date else None,"sourceAgeDays":age,"featureAudit":aD})
            persistence=_positive(p071 or p0d)
            if persistence is not None:
                prP=_model_row("issued",prediction=persistence,provenance={"sourceDate":source_date.isoformat() if source_date else None,"sourceAgeDays":age})
            else:prP=_model_row("not_issued","no_pre_origin_observation_or_source_too_old")
            for model,feat,p0,err,pr in ((fit71,feature71,p071,err71,pr71),(fitdirect,featureD,p0d,errdirect,prD)):
                if model is None or err or feat is None or p0 is None:continue
                try:
                    pred=p0*math.exp(float(model.predict([feat])[0]))
                    if math.isfinite(pred) and pred>0:pr.update(status="issued",reason=None,prediction=pred)
                    else:pr.update(reason="nonpositive_or_nonfinite_prediction")
                except Exception as exc:pr.update(reason=f"prediction_failed:{type(exc).__name__}")
            source_sarima=None; sarima_pred=None;sarima_reason=sarima_error or state_error
            if source_date is None:sarima_reason=sarima_reason or "no_source_observation"
            elif age>14:sarima_reason=sarima_reason or "latest_source_older_than_14_days"
            elif fit_sarima is not None and not state_error:
                try:
                    mean=fit_sarima.get_forecast(steps=h+1).predicted_mean
                    lp=float(mean.iloc[-1] if hasattr(mean,"iloc") else mean[-1]); candidate=math.exp(lp)
                    if math.isfinite(candidate) and candidate>0:sarima_pred=candidate
                    else:sarima_reason="invalid_sarima_prediction"
                except Exception as exc:sarima_reason=f"sarima_forecast_failed:{type(exc).__name__}"
            sarima=_model_row("issued" if sarima_pred else "not_issued",sarima_reason,sarima_pred,training={**sarima_audit,"fitOrigin":month_first_origin.isoformat()},provenance={"lastActualDate":source_date.isoformat() if source_date else None,"requestedCutoff":(origin-dt.timedelta(days=1)).isoformat(),"lastUpdatedDate":state_day.isoformat() if state_day else None})
            fixed=_positive(fixed_map.get((product,day,h)))
            by_key[key]={"product":product,"target":day.isoformat(),"origin":origin.isoformat(),"horizon":h,"eligible":True,"models":{"online71":pr71,"causalDirect":prD,"causalSarima":sarima,"causal50":_blend_ridge_sarima(prD,sarima),"online71Blend":_blend_ridge_sarima(pr71,sarima),"persistence":prP,"fixed50":_model_row("issued",prediction=fixed,provenance={"role":"descriptive_only_same_day_price_access"}) if fixed else _model_row("not_issued","fixed50_unavailable",provenance={"role":"descriptive_only_same_day_price_access"})}}
            total+=1
            if progress and total%500==0:progress(phase,total)
    return [by_key[k] for k in grid]

def _forecast_phase(records,direct_series,fixed_map,alpha,phase,start,end,progress=None):
    # Refit by the forecast origin's calendar month, not the target's month.
    months=sorted({((d-dt.timedelta(days=h)).year,(d-dt.timedelta(days=h)).month) for _,d,h in _grid(start,end)})
    rows=[]
    for month in months:
        rows.extend(_forecast_origin_month(records,direct_series,fixed_map,alpha,phase,start,end,month,progress))
    indexed={(r["product"],dt.date.fromisoformat(r["target"]),r["horizon"]):r for r in rows}
    return [indexed[k] for k in _grid(start,end)]

def _base_row(product,day,h,origin,start,phase):
    why="model_not_yet_fitted_at_origin"
    none={m:_model_row("not_issued",why) for m in METHODS if m!="fixed50"}
    none["fixed50"]=_model_row("not_issued","outside_fixed50_comparison_grid",provenance={"role":"descriptive_only_same_day_price_access"})
    return {"product":product,"target":day.isoformat(),"origin":origin.isoformat(),"horizon":h,"eligible":False,"models":none}

def build(prices,fixed50,hyperparam,*,price_sha=None,fixed50_sha=None,hyperparam_sha=None,progress=None):
    alpha=_alpha_from_manifest(hyperparam);records,audit=_records(prices);direct_series=direct_horizon._observations(prices);fixed_map=narrow._fixed50(fixed50)
    confirmation=_forecast_phase(records,direct_series,fixed_map,alpha,"confirmation",CONF_START,CONF_END,progress)
    evaluation=_forecast_phase(records,direct_series,fixed_map,alpha,"evaluation",EVAL_START,EVAL_END,progress)
    import sklearn,statsmodels
    doc={"schemaVersion":1,"phase":"forecast_only","targetValuesAttached":False,"protocol":{"id":"equal-info-online-20261010-v1","products":list(PRODUCTS),"horizons":list(HORIZONS),"confirmationTargets":[CONF_START.isoformat(),CONF_END.isoformat()],"evaluationTargets":[EVAL_START.isoformat(),EVAL_END.isoformat()],"alphaByProduct":alpha,"minimumTrainingTargets":MIN_TRAIN,"monthlyRefit":"first eligible origin per calendar month; target < fit origin","featureCutoff":"each historical and forecast row origin-1","online71FeatureNames":list(library.FEATURE_NAMES),"causalDirect":"original modeling.FEATURE_NAMES; positive price-and-volume observations; Ridge alpha 10","SARIMA":{"order":[1,1,1],"seasonalOrder":[1,0,0,7],"d":1,"windowCalendarDays":SARIMA_DAYS,"fitEnd":"fit origin-1","forecastSteps":"horizon+1","missingCalendarDays":"NaN; no interpolation","fallback":"blend uses available Ridge component if not issued"},"blendWeights":[.5,.5],"fixed50":"descriptive only; same-day price access differs","runtimeVersions":{"scikitLearn":sklearn.__version__,"statsmodels":statsmodels.__version__}},"inputSha256":{"prices":price_sha,"fixed50Manifest":fixed50_sha,"hyperparameterManifest":hyperparam_sha,"module":_sha(Path(__file__)),"featureLibrary":LIBRARY_SHA256,"richHelper":RICH_HELPER_SHA256,"narrowHelper":NARROW_HELPER_SHA256,"sarimaHelper":TEN_HELPER_SHA256,"modelingHelper":MODELING_SHA256,"directHorizonHelper":DIRECT_HELPER_SHA256},"sourceAudit":{"priceRows":audit},"plannedConfirmationCount":len(confirmation),"plannedEvaluationCount":len(evaluation),"confirmationForecasts":confirmation,"evaluationForecasts":evaluation}
    doc["manifestSha256"]=hashlib.sha256(_canonical(doc)).hexdigest();return doc

def build_from_paths(progress=None):
    _check_inputs()
    return build(json.loads(PRICES_PATH.read_text()),json.loads(FIXED50_PATH.read_text()),json.loads(HYPERPARAM_PATH.read_text()),price_sha=PRICES_SHA256,fixed50_sha=FIXED50_SHA256,hyperparam_sha=HYPERPARAM_SHA256,progress=progress)

def _validate(doc):
    if doc.get("phase")!="forecast_only" or doc.get("targetValuesAttached") is not False:raise ValueError("expected answer-free manifest")
    unsigned=dict(doc);digest=unsigned.pop("manifestSha256",None)
    if digest!=hashlib.sha256(_canonical(unsigned)).hexdigest():raise ValueError("manifest digest mismatch")
    for key,start,end in (("confirmationForecasts",CONF_START,CONF_END),("evaluationForecasts",EVAL_START,EVAL_END)):
        expected=set(_grid(start,end));rows=doc.get(key)
        if not isinstance(rows,list) or len(rows)!=len(expected):raise ValueError("forecast grid length mismatch")
        seen=set()
        for r in rows:
            d=narrow._date(r.get("target"));o=narrow._date(r.get("origin"));h=r.get("horizon");ident=(r.get("product"),d,h)
            if ident not in expected or ident in seen or d is None or o!=d-dt.timedelta(days=h):raise ValueError("invalid forecast key")
            if r.get("eligible") is not (o>=start):raise ValueError("eligibility mismatch")
            models=r.get("models")
            if not isinstance(models,dict) or set(models)!=set(METHODS):raise ValueError("method grid mismatch")
            for m,p in models.items():
                if p.get("status") not in ("issued","not_issued"):raise ValueError("invalid prediction status")
                v=_positive(p.get("prediction"))
                if p["status"]=="issued" and v is None:raise ValueError("invalid issued prediction")
                if p["status"]=="not_issued" and p.get("prediction") is not None:raise ValueError("not-issued prediction has value")
            seen.add(ident)
        if seen!=expected:raise ValueError("incomplete forecast grid")

def verify_manifest(path:Path)->bool:
    raw=json.loads(path.read_text());_validate(raw)
    return _canonical(raw)==_canonical(build_from_paths())

def write_exclusive(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f:f.write(_canonical(value))

def main(argv=None):
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="cmd",required=True);b=sub.add_parser("build");b.add_argument("output",type=Path);v=sub.add_parser("verify");v.add_argument("manifest",type=Path);args=ap.parse_args(argv)
    if args.cmd=="build":write_exclusive(args.output,build_from_paths(progress=lambda phase,n:print(f"{phase}: {n}",flush=True)))
    elif not verify_manifest(args.manifest):raise SystemExit("manifest replay mismatch")
    return 0
if __name__=="__main__":raise SystemExit(main())
