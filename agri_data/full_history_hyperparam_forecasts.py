"""Chronological hyperparameter selection and answer-free forecast manifests.

This module implements the preregistered 2026-10-10 protocol only. It uses the
fixed 71-feature library, no API access, and never reads target answers in the
forecast grids. Scoring is a separate module and must verify replay first.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math, random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from . import full_history_return_eval as narrow
from . import full_history_xgb_rf_eval as rich
from . import full_history_feature_library as library

ROOT=Path(__file__).resolve().parents[1]
PRICES_PATH=ROOT/"outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
FIXED50_PATH=ROOT/"outputs/experiments/five-crop-specialists-20261009/forecast-manifest.json"
OUT=ROOT/"outputs/experiments/full-history-hyperparam-20261010"
PRICES_SHA256="f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
FEATURE_LIBRARY_SHA256="58bcde98d11239494ae13cc6169fb4c7f84a49a15ad11cf5b3cf5c49b4f39389"
FIXED50_SHA256="1497ee362945fcab45353a7b2403be752af3bf63c5f584cc0aedb2cc111ceaa8"
RICH_HELPER_SHA256="1951b34bc019ccd9fe680776f4277105e1edb470d3d7d5a37802c781088d9245"
NARROW_HELPER_SHA256="276f389e7153e07ef02836183d55e9823a6eb7c9bf24acc882d663946c55c5c5"
XGBOOST_VERSION="2.1.4"
PRODUCTS=narrow.PRODUCTS; HORIZONS=narrow.HORIZONS
CV_FOLDS=((2023,dt.date(2014,1,1),dt.date(2022,12,31)),(2024,dt.date(2014,1,1),dt.date(2023,12,31)))
CONF_START,CONF_END=dt.date(2025,1,1),dt.date(2025,12,31)
EVAL_START,EVAL_END=narrow.EVAL_START,narrow.EVAL_END
MIN_CV=30; MIN_TRAIN=narrow.MIN_TRAIN
RIDGE_ALPHAS=(.01,.1,1.,10.,100.,1000.)
XGB_CONFIGS=(
 {"id":"X0","n_estimators":300,"max_depth":3,"min_child_weight":15,"learning_rate":.05,"reg_lambda":5.},
 {"id":"X1","n_estimators":200,"max_depth":2,"min_child_weight":30,"learning_rate":.05,"reg_lambda":10.},
 {"id":"X2","n_estimators":500,"max_depth":2,"min_child_weight":10,"learning_rate":.03,"reg_lambda":5.},
 {"id":"X3","n_estimators":200,"max_depth":4,"min_child_weight":10,"learning_rate":.1,"reg_lambda":5.},
 {"id":"X4","n_estimators":500,"max_depth":3,"min_child_weight":5,"learning_rate":.03,"reg_lambda":1.},
 {"id":"X5","n_estimators":300,"max_depth":4,"min_child_weight":30,"learning_rate":.05,"reg_lambda":10.},)
RF_CONFIGS=(
 {"id":"R0","max_depth":8,"min_samples_leaf":10,"max_features":.7},
 {"id":"R1","max_depth":5,"min_samples_leaf":20,"max_features":.5},
 {"id":"R2","max_depth":12,"min_samples_leaf":5,"max_features":.5},
 {"id":"R3","max_depth":8,"min_samples_leaf":5,"max_features":1.},
 {"id":"R4","max_depth":None,"min_samples_leaf":20,"max_features":.7},
 {"id":"R5","max_depth":5,"min_samples_leaf":10,"max_features":1.},)
FAMILIES=("ridge","xgb","rf")
DEFAULTS={"ridge":"ridge_a10","xgb":"X0","rf":"R0"}
TREE_FIT_PROGRESS_INTERVAL=100
TREE_CV_WORKERS=4
FEATURE_ROW_PROGRESS_INTERVAL=10000

def _canonical(v:Any)->bytes:return (json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def _sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()

def _load_xgb():
    try:
        import xgboost
        from xgboost import XGBRegressor
    except Exception as exc: raise RuntimeError("required xgboost==2.1.4 is unavailable") from exc
    if xgboost.__version__!=XGBOOST_VERSION:raise RuntimeError(f"xgboost=={XGBOOST_VERSION} required; found {xgboost.__version__}")
    return XGBRegressor

def _configurations(family:str)->tuple[dict[str,Any],...]:
    if family=="ridge":
        ids={.01:"ridge_a0p01",.1:"ridge_a0p1",1.:"ridge_a1",10.:"ridge_a10",100.:"ridge_a100",1000.:"ridge_a1000"}
        return tuple({"id":ids[a],"alpha":a} for a in RIDGE_ALPHAS)
    if family=="xgb":return XGB_CONFIGS
    if family=="rf":return RF_CONFIGS
    raise ValueError(f"unknown family: {family}")

def _fit(family:str,config:dict[str,Any],X,y):
    if family=="ridge":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge
        return make_pipeline(StandardScaler(),Ridge(alpha=float(config["alpha"]),fit_intercept=True)).fit(X,y)
    if family=="xgb":
        cls=_load_xgb()
        return cls(objective="reg:squarederror",tree_method="hist",subsample=.8,colsample_bytree=.8,n_jobs=1,random_state=0,
                   **{k:v for k,v in config.items() if k!="id"}).fit(X,y)
    if family=="rf":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(bootstrap=True,n_jobs=1,random_state=42,n_estimators=300,
                   **{k:v for k,v in config.items() if k!="id"}).fit(X,y)
    raise ValueError(f"unknown family: {family}")

def _maps(prices:dict[str,Any]):return rich._raw_maps(prices)

def _fixed_map(raw:dict[str,Any]):return narrow._fixed50(raw)

def _valid_rows(data:list[dict[str,Any]],start:dt.date,end:dt.date,*,origin_start:dt.date|None=None):
    return sorted((r for r in data if start<=r["target"]<=end and (origin_start is None or r["target"]-dt.timedelta(days=r["horizon"])>=origin_start)),key=lambda r:r["target"])

def _metrics(predictions:list[float],rows:list[dict[str,Any]])->dict[str,Any]:
    if not rows:return {"n":0,"mape":None,"mae":None}
    ae=[abs(float(p)-r["actual"]) for p,r in zip(predictions,rows)]
    return {"n":len(rows),"mape":100*math.fsum(e/r["actual"] for e,r in zip(ae,rows))/len(rows),"mae":math.fsum(ae)/len(ae)}

def _choose_family(family:str,cv:list[dict[str,Any]])->dict[str,Any]:
    configs=_configurations(family);ranked=[]
    for config in configs:
        cells=[r for r in cv if r["configId"]==config["id"]]
        supported=[r for r in cells if r["status"]=="scored"]
        # Both years must have all 10 supported horizon cells.
        valid=len(supported)==20 and {r["year"] for r in supported}=={2023,2024}
        mape=math.fsum(math.fsum(r["mape"] for r in supported if r["year"]==year)/10 for year in (2023,2024))/2 if valid else None
        mae=math.fsum(math.fsum(r["mae"] for r in supported if r["year"]==year)/10 for year in (2023,2024))/2 if valid else None
        ranked.append({"config":config,"supported":valid,"equalYearHorizonMape":mape,"equalYearHorizonMae":mae})
    eligible=[r for r in ranked if r["supported"]]
    default=next(r for r in ranked if r["config"]["id"]==DEFAULTS[family])
    def tie(config):
        if family=="ridge":return float(config["alpha"])
        return int(config["id"][1:])
    chosen=min(eligible,key=lambda r:(r["equalYearHorizonMape"],r["equalYearHorizonMae"],tie(r["config"])),default=None)
    if not default["supported"]:chosen=None
    reason=None if chosen else ("default_configuration_disqualified" if not default["supported"] else "no_configuration_supported_all_20_cv_cells")
    return {"chosenConfig":chosen["config"] if chosen else None,"reason":reason,"defaultSupported":default["supported"],"configurationSummary":ranked}

def _make_cache(records:dict[str,dict[dt.date,dict[str,Any]]],progress:Callable[[str,int],None]|None=None)->dict[tuple[str,int],list[dict[str,Any]]]:
    cache={};feature_rows=0
    for product in PRODUCTS:
        rec=records[product];ordered=sorted(rec);prices={d:r["price"] for d,r in rec.items()}
        for h in HORIZONS:
            rows=[]
            # Targets through the final evaluation calendar date; no answer values
            # are written into either forecast grid.
            day=dt.date(2014,1,1)
            while day<=EVAL_END:
                x,p0,audit=library.features(rec,day,h,ordered,prices)
                # Keep 2026 labels out of the answer-free build entirely.
                actual=rec.get(day,{}).get("price") if day<=narrow.TRAIN_END else None
                y=math.log(actual/p0) if actual is not None and p0 is not None else None
                rows.append({"target":day,"horizon":h,"features":x,"p0":p0,"actual":actual,"y":y,"audit":audit})
                feature_rows+=1
                if progress and feature_rows%FEATURE_ROW_PROGRESS_INTERVAL==0:progress("feature_rows",feature_rows)
                day+=dt.timedelta(days=1)
            cache[(product,h)]=rows
    return cache

def _fit_period(family,config,data,end):
    train=[r for r in data if dt.date(2014,1,1)<=r["target"]<=end and r["features"] is not None and r["y"] is not None]
    if len(train)<MIN_TRAIN:return None,{"count":len(train),"firstTarget":train[0]["target"].isoformat() if train else None,"lastTarget":train[-1]["target"].isoformat() if train else None},"fewer_than_90_valid_training_targets"
    try:model=_fit(family,config,[r["features"] for r in train],[r["y"] for r in train])
    except Exception as exc:return None,{"count":len(train),"firstTarget":train[0]["target"].isoformat(),"lastTarget":train[-1]["target"].isoformat()},f"fit_failed:{type(exc).__name__}"
    return model,{"count":len(train),"firstTarget":train[0]["target"].isoformat(),"lastTarget":train[-1]["target"].isoformat()},None

def _cv_cell(cache,product,family,config,year,train_start,train_end,h):
    data=cache[(product,h)]
    train=sorted((r for r in data if train_start<=r["target"]<=train_end and r["features"] is not None and r["y"] is not None),key=lambda r:r["target"])
    val=_valid_rows(data,dt.date(year,1,1),dt.date(year,12,31),origin_start=dt.date(year,1,1))
    val=[r for r in val if r["features"] is not None and r["actual"] is not None]
    base={"configId":config["id"],"year":year,"horizon":h,"trainCount":len(train),"trainFirst":train[0]["target"].isoformat() if train else None,"trainLast":train[-1]["target"].isoformat() if train else None,"validationCount":len(val),"validationFirst":val[0]["target"].isoformat() if val else None,"validationLast":val[-1]["target"].isoformat() if val else None}
    if len(val)<MIN_CV:return {**base,"status":"unsupported","reason":"fewer_than_30_validation_pairs","mape":None,"mae":None}
    if len(train)<MIN_TRAIN:return {**base,"status":"unsupported","reason":"fewer_than_90_valid_training_targets","mape":None,"mae":None}
    try:
        model=_fit(family,config,[r["features"] for r in train],[r["y"] for r in train])
        predicted=model.predict([r["features"] for r in val])
        price_predictions=[r["p0"]*math.exp(float(q)) for q,r in zip(predicted,val)]
        if len(price_predictions)!=len(val) or any(not math.isfinite(p) or p<=0 for p in price_predictions):raise ValueError("invalid_prediction_batch")
        metric=_metrics(price_predictions,[{"actual":r["actual"]} for r in val])
        return {**base,"status":"scored","reason":None,"mape":metric["mape"],"mae":metric["mae"]}
    except Exception as exc:return {**base,"status":"failed","reason":f"fit_or_predict_failed:{type(exc).__name__}","mape":None,"mae":None}

def _cross_validate(cache,progress:Callable[[str,int],None]|None=None,max_workers:int=TREE_CV_WORKERS):
    if max_workers<1:raise ValueError("max_workers must be positive")
    jobs=[]
    for product in PRODUCTS:
      for family in FAMILIES:
       for config in _configurations(family):
        for year,train_start,train_end in CV_FOLDS:
         for h in HORIZONS:jobs.append((product,family,config,year,train_start,train_end,h))
    results={};tree_jobs=[job for job in jobs if job[1] in ("xgb","rf")];ridge_jobs=[job for job in jobs if job[1]=="ridge"]
    for job in ridge_jobs:
        product,family,config,year,_,_,h=job
        results[(product,family,config["id"],year,h)]=_cv_cell(cache,product,family,config,year,job[4],job[5],h)
    if max_workers==1:
        tree_results=(_cv_cell(cache,*job[:3],job[3],job[4],job[5],job[6]) for job in tree_jobs)
        for i,(job,result) in enumerate(zip(tree_jobs,tree_results),1):
            product,family,config,year,_,_,h=job
            results[(product,family,config["id"],year,h)]=result
            if progress and i%TREE_FIT_PROGRESS_INTERVAL==0:progress("tree_cv_fits",i)
    else:
        with ThreadPoolExecutor(max_workers=max_workers,thread_name_prefix="hyperparam-tree-cv") as pool:
            futures=[pool.submit(_cv_cell,cache,product,family,config,year,train_start,train_end,h) for product,family,config,year,train_start,train_end,h in tree_jobs]
            for i,(job,future) in enumerate(zip(tree_jobs,futures),1):
                product,family,config,year,_,_,h=job
                results[(product,family,config["id"],year,h)]=future.result()
                if progress and i%TREE_FIT_PROGRESS_INTERVAL==0:progress("tree_cv_fits",i)
    cv_by_product={p:{f:[] for f in FAMILIES} for p in PRODUCTS}
    for product,family,config,year,_,_,h in jobs:
        cv_by_product[product][family].append(results[(product,family,config["id"],year,h)])
    return cv_by_product

def _default_config(family):
    return next(c for c in _configurations(family) if c["id"]==DEFAULTS[family])

def _grid(start,end):
    days=(end-start).days+1
    return [(p,start+dt.timedelta(days=i),h) for p in PRODUCTS for i in range(days) for h in HORIZONS]

def _forecast_arm(cache,job):
    product,h,phase,family,arm,config,default_supported,targets=job
    data=cache[(product,h)];by_day={r["target"]:r for r in data};end_fit=dt.date(2024,12,31) if phase=="confirmation" else dt.date(2025,12,31)
    if config is None or not default_supported:
        why="default_configuration_disqualified" if not default_supported else "no_supported_selected_configuration"
        return {day:{"status":"not_issued","reason":why,"prediction":None} for day,_ in targets}
    model,training,reason=_fit_period(family,config,data,end_fit);result={};ready=[]
    for day,eligible in targets:
        item=by_day.get(day);status_reason=None
        if not eligible:status_reason="model_not_yet_fitted_at_origin"
        elif item is None or item["features"] is None:status_reason=(item or {}).get("audit",{}).get("reason","missing_or_stale_price")
        elif reason:status_reason=reason
        if status_reason:result[day]={"status":"not_issued","reason":status_reason,"prediction":None,"training":training}
        else:ready.append((day,item))
    if ready:
        try:
            logpred=model.predict([item["features"] for day,item in ready])
            if len(logpred)!=len(ready):raise ValueError("prediction_count_mismatch")
            for (day,item),q in zip(ready,logpred):
                try:pred=item["p0"]*math.exp(float(q))
                except (OverflowError,ValueError):pred=float("inf")
                result[day]=({"status":"not_issued","reason":"nonpositive_or_nonfinite_prediction","prediction":None,"training":training} if not math.isfinite(pred) or pred<=0 else {"status":"issued","reason":None,"prediction":pred,"training":training})
        except Exception as exc:
            for day,item in ready:result[day]={"status":"not_issued","reason":f"prediction_failed:{type(exc).__name__}","prediction":None,"training":training}
    return result

def _emit_phase(cache,selection,phase,grid,fixed_map=None,progress=None,max_workers:int=TREE_CV_WORKERS):
    rows_by_key={};jobs=[]
    for product in PRODUCTS:
      for h in HORIZONS:
        keys=[(p,d,hh) for p,d,hh in grid if p==product and hh==h]
        for key in keys:
            origin=key[1]-dt.timedelta(days=h);minimum_origin=CONF_START if phase=="confirmation" else EVAL_START
            rows_by_key[key]={"product":product,"target":key[1].isoformat(),"origin":origin.isoformat(),"horizon":h,"eligible":origin>=minimum_origin,"models":{}}
        target_info=[(key[1],rows_by_key[key]["eligible"]) for key in keys]
        for family in FAMILIES:
          info=selection[product][family]
          for arm,config in ((family+"Tuned",info["chosenConfig"]),(family+"Default",_default_config(family))):
            jobs.append((product,h,phase,family,arm,config,info["defaultSupported"],target_info))
        if phase=="evaluation":
          for key in keys:
            pred=fixed_map.get((product,key[1],h)) if fixed_map else None
            rows_by_key[key]["fixed50"]={"status":"issued" if narrow._positive(pred) else "not_issued","reason":None if narrow._positive(pred) else "fixed50_unavailable","prediction":pred}
    tree_jobs=[j for j in jobs if j[3] in ("xgb","rf")];ridge_jobs=[j for j in jobs if j[3]=="ridge"];results={}
    for job in ridge_jobs:results[(job[0],job[1],job[3],job[4])]=_forecast_arm(cache,job)
    with ThreadPoolExecutor(max_workers=max_workers,thread_name_prefix="hyperparam-final-tree") as pool:
        futures=[pool.submit(_forecast_arm,cache,job) for job in tree_jobs]
        for i,(job,future) in enumerate(zip(tree_jobs,futures),1):
            results[(job[0],job[1],job[3],job[4])]=future.result()
            if progress and i%TREE_FIT_PROGRESS_INTERVAL==0:progress("final_tree_fits",i)
    for job in jobs:
        product,h,phase,family,arm,config,default_supported,targets=job
        predictions=results[(product,h,family,arm)]
        for day,_ in targets:rows_by_key[(product,day,h)]["models"][arm]=predictions[day]
    return [rows_by_key[k] for k in grid]

def build(prices:dict[str,Any],fixed50:dict[str,Any],*,price_sha:str|None=None,fixed50_sha:str|None=None,progress:Callable[[str,int],None]|None=None)->dict[str,Any]:
    if len(library.FEATURE_NAMES)!=71:raise ValueError("the fixed 71-feature schema is required")
    if _sha(Path(library.__file__))!=FEATURE_LIBRARY_SHA256:raise ValueError("pinned feature-library hash mismatch")
    if _sha(Path(rich.__file__))!=RICH_HELPER_SHA256 or _sha(Path(narrow.__file__))!=NARROW_HELPER_SHA256:raise ValueError("pinned prior feature helper hash mismatch")
    _load_xgb()  # Exact required dependency; fail closed before any fitting.
    import sklearn
    records,audit=_maps(prices);fixed_map=_fixed_map(fixed50)
    cache=_make_cache(records,progress)
    cv_by_product=_cross_validate(cache,progress)
    selection={}
    for product in PRODUCTS:
        selection[product]={}
        for family in FAMILIES:
            selected=_choose_family(family,cv_by_product[product][family])
            selection[product][family]={**selected,"defaultConfig":_default_config(family),"cvCells":cv_by_product[product][family]}
    conf_grid=_grid(CONF_START,CONF_END);eval_grid=_grid(EVAL_START,EVAL_END)
    confirmation=_emit_phase(cache,selection,"confirmation",conf_grid,progress=progress)
    evaluation=_emit_phase(cache,selection,"evaluation",eval_grid,fixed_map,progress)
    doc={"schemaVersion":1,"phase":"forecast_only","targetValuesAttached":False,
      "protocol":{"id":"full-history-hyperparam-20261010-v1","products":list(PRODUCTS),"horizons":list(HORIZONS),
        "confirmationTargets":[CONF_START.isoformat(),CONF_END.isoformat()],"evaluationTargets":[EVAL_START.isoformat(),EVAL_END.isoformat()],
        "cv":{"folds":[{"year":y,"trainTargets":[a.isoformat(),b.isoformat()],"validationOriginStart":dt.date(y,1,1).isoformat()} for y,a,b in CV_FOLDS],"minimumValidationPairs":MIN_CV,"minimumTrainingTargets":MIN_TRAIN,"originCutoff":"origin-1","ranking":"MAPE equal weight year then horizon; ties MAE then config order"},
        "featureNames":list(library.FEATURE_NAMES),"target":"log(actual/p0)","models":{"ridge":{"pipeline":["StandardScaler","Ridge(fit_intercept=True)"],"alphas":list(RIDGE_ALPHAS)},
        "xgb":{"version":XGBOOST_VERSION,"objective":"reg:squarederror","tree_method":"hist","subsample":.8,"colsample_bytree":.8,"n_jobs":1,"random_state":0,"configs":[dict(x) for x in XGB_CONFIGS]},
        "rf":{"bootstrap":True,"n_jobs":1,"random_state":42,"n_estimators":300,"configs":[dict(x) for x in RF_CONFIGS]},"defaultConfigIds":dict(DEFAULTS),"runtimeVersions":{"scikitLearn":sklearn.__version__}},
        "prefitOrigins":"not issued","2025ConfirmationIsPreviouslyViewed":True},
      "inputSha256":{"prices":price_sha,"fixed50Manifest":fixed50_sha,"featureLibrary":FEATURE_LIBRARY_SHA256,
         "previousRichFeatureBuilder":RICH_HELPER_SHA256,"narrowPriceFeatureHelper":NARROW_HELPER_SHA256},
      "sourceAudit":{"sourceRows":audit},"selection":selection,"plannedConfirmationCount":len(confirmation),"plannedEvaluationCount":len(evaluation),
      "confirmationForecasts":confirmation,"evaluationForecasts":evaluation}
    doc["manifestSha256"]=hashlib.sha256(_canonical(doc)).hexdigest();return doc

def build_from_paths(progress:Callable[[str,int],None]|None=None)->dict[str,Any]:
    if _sha(Path(library.__file__))!=FEATURE_LIBRARY_SHA256:raise ValueError("pinned feature-library hash mismatch")
    for path,expected,label in ((PRICES_PATH,PRICES_SHA256,"prices"),(FIXED50_PATH,FIXED50_SHA256,"fixed50 manifest")):
        if _sha(path)!=expected:raise ValueError(f"pinned {label} hash mismatch")
    return build(json.loads(PRICES_PATH.read_text()),json.loads(FIXED50_PATH.read_text()),price_sha=PRICES_SHA256,fixed50_sha=FIXED50_SHA256,progress=progress)

def _validate_manifest(raw:dict[str,Any])->None:
    if raw.get("phase")!="forecast_only" or raw.get("targetValuesAttached") is not False:raise ValueError("expected answer-free forecast manifest")
    unsigned=dict(raw);digest=unsigned.pop("manifestSha256",None)
    if digest!=hashlib.sha256(_canonical(unsigned)).hexdigest():raise ValueError("manifest digest mismatch")
    for key,start,end in (("confirmationForecasts",CONF_START,CONF_END),("evaluationForecasts",EVAL_START,EVAL_END)):
        rows=raw.get(key)
        expected=_grid(start,end)
        expected_set=set(expected)
        if not isinstance(rows,list) or len(rows)!=len(expected):raise ValueError(f"{key} has wrong grid size")
        seen=set()
        for r in rows:
            day=narrow._date(r.get("target"));h=r.get("horizon");product=r.get("product");origin=narrow._date(r.get("origin"))
            item=(product,day,h)
            if item not in expected_set or item in seen or day is None or origin!=day-dt.timedelta(days=h):raise ValueError(f"invalid row identity in {key}")
            if r.get("eligible") is not (origin>=(CONF_START if key=="confirmationForecasts" else EVAL_START)):raise ValueError("eligibility mismatch")
            if not isinstance(r.get("models"),dict):raise ValueError("missing model status map")
            for arm in ("ridgeTuned","ridgeDefault","xgbTuned","xgbDefault","rfTuned","rfDefault"):
                x=r["models"].get(arm)
                if not isinstance(x,dict) or x.get("status") not in ("issued","not_issued"):raise ValueError("invalid model status")
                if x["status"]=="issued" and not narrow._positive(x.get("prediction")):raise ValueError("issued arm lacks positive forecast")
                if x["status"]=="not_issued" and x.get("prediction") is not None:raise ValueError("not-issued arm contains prediction")
            if key=="evaluationForecasts":
                x=r.get("fixed50")
                if not isinstance(x,dict) or x.get("status") not in ("issued","not_issued"):raise ValueError("invalid fixed50 status")
                if x["status"]=="issued" and not narrow._positive(x.get("prediction")):raise ValueError("issued fixed50 row lacks positive forecast")
                if x["status"]=="not_issued" and x.get("prediction") is not None:raise ValueError("not-issued fixed50 row contains forecast")
            seen.add(item)
        if seen!=expected_set:raise ValueError(f"incomplete grid in {key}")

def verify_manifest(path:Path)->bool:
    raw=json.loads(path.read_text());_validate_manifest(raw)
    return _canonical(raw)==_canonical(build_from_paths())

def write_exclusive(path:Path,value:dict[str,Any])->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f:f.write(_canonical(value))

def main(argv:list[str]|None=None)->int:
    ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="cmd",required=True)
    b=sub.add_parser("build");b.add_argument("output",type=Path)
    v=sub.add_parser("verify");v.add_argument("manifest",type=Path)
    a=ap.parse_args(argv)
    if a.cmd=="build":write_exclusive(a.output,build_from_paths(progress=lambda kind,n:print(f"{kind}: {n}",flush=True)))
    elif not verify_manifest(a.manifest):raise SystemExit("manifest replay mismatch")
    return 0

if __name__=="__main__":raise SystemExit(main())
