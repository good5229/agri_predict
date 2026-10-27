"""Frozen XGBoost, Random Forest and richer-feature ablation experiment.

The fixed protocol trains target dates 2014-01-01..2025-12-31, separately for
five crops and horizons 1..10. tree_xgb and tree_rf plus rich Ridge use identical
rows and scale-free price/volume/row-count/market features; target is
log(actual/p0). At target t, horizon h, every input ends at origin-1 where
o=t-h. The 2026-01-01..09-27 full calendar grid is answer-free first; origins
before 2026-01-01 remain planned but unissued. A separate replay-verified score
step joins actuals. XGBoost must be exactly version 2.1.4 from the isolated
runtime; this module never substitutes another estimator. No API access occurs.
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

from . import full_history_return_eval as narrow

ROOT=Path(__file__).resolve().parents[1]
PRICES_PATH=ROOT/"outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
NARROW_PATH=ROOT/"outputs/experiments/full-history-return-20261009/forecast-manifest.json"
FIXED50_PATH=ROOT/"outputs/experiments/five-crop-specialists-20261009/forecast-manifest.json"
PRICES_SHA256="f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
NARROW_SHA256="86da40f5bcfb6669c5f0a4b34e9ad33c9c69d7f7492b85d58dd43bd90b119d45"
FIXED50_SHA256="1497ee362945fcab45353a7b2403be752af3bf63c5f584cc0aedb2cc111ceaa8"
NARROW_HELPER_SHA256="276f389e7153e07ef02836183d55e9823a6eb7c9bf24acc882d663946c55c5c5"
XGBOOST_VERSION="2.1.4"
PRODUCTS=narrow.PRODUCTS;HORIZONS=narrow.HORIZONS
TRAIN_START,TRAIN_END=narrow.TRAIN_START,narrow.TRAIN_END
EVAL_START,EVAL_END=narrow.EVAL_START,narrow.EVAL_END
MIN_TRAIN=narrow.MIN_TRAIN;BANDWIDTHS=narrow.BANDWIDTHS;ALPHA=narrow.ALPHA
MARKET_CODES=frozenset(("110001","110008"))
COMPARISONS=("xgb_vs_fixed50","xgb_vs_rich_ridge","rf_vs_fixed50","rf_vs_rich_ridge","rich_ridge_vs_narrow_ridge","xgb_vs_rf")
LOSSES=("ae","ape")


def _sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def _canonical(value:Any)->bytes:
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def _check_helper()->None:
    if _sha(Path(narrow.__file__))!=NARROW_HELPER_SHA256:raise ValueError("reviewed narrow-Ridge helper hash mismatch")


def _load_xgboost():
    try:
        import xgboost
        from xgboost import XGBRegressor
    except Exception as exc:
        raise RuntimeError("XGBoost 2.1.4 unavailable; set PYTHONPATH to an environment containing xgboost==2.1.4 before running") from exc
    if getattr(xgboost,"__version__",None)!=XGBOOST_VERSION:
        raise RuntimeError(f"XGBoost {XGBOOST_VERSION} required; found {getattr(xgboost,'__version__','unknown')}")
    return XGBRegressor


def _raw_maps(raw:dict[str,Any])->tuple[dict[str,dict[dt.date,dict[str,Any]]],dict[str,Any]]:
    if raw.get("schemaVersion")!=1 or not isinstance(raw.get("products"),list):raise ValueError("prices must use dashboard schemaVersion 1")
    maps={};audits={}
    for product in raw["products"]:
        pid=product.get("id") if isinstance(product,dict) else None
        if pid not in PRODUCTS or pid in maps or not isinstance(product.get("observations"),list):raise ValueError("unknown, duplicate, or malformed product")
        records={};audit={"rows":0,"invalidDate":0,"invalidOrNonpositivePrice":0,"invalidOrNonpositiveVolume":0,"invalidOrNonpositiveRowCount":0,"invalidMarketCodes":0,"duplicatePriceDates":0,"duplicateVolumeDates":0,"duplicateRowCountDates":0}
        seen_volume=set();seen_rows=set()
        for row in product["observations"]:
            audit["rows"]+=1
            if not isinstance(row,dict):audit["invalidDate"]+=1;audit["invalidOrNonpositivePrice"]+=1;continue
            day=narrow._date(row.get("date"));price=narrow._positive(row.get("actual"));vol=narrow._positive(row.get("volumeKg"));count=narrow._positive(row.get("rowCount"))
            if day is None:audit["invalidDate"]+=1
            if price is None:audit["invalidOrNonpositivePrice"]+=1
            if vol is None:audit["invalidOrNonpositiveVolume"]+=1
            if count is None:audit["invalidOrNonpositiveRowCount"]+=1
            markets=row.get("marketCodes")
            if not isinstance(markets,list) or any(not isinstance(x,str) for x in markets):audit["invalidMarketCodes"]+=1;market_set=frozenset()
            else:market_set=frozenset(markets)
            if day is None or price is None:continue
            if day in records:audit["duplicatePriceDates"]+=1;raise ValueError(f"duplicate positive price date {pid}/{day}")
            if vol is not None and day in seen_volume:audit["duplicateVolumeDates"]+=1
            if count is not None and day in seen_rows:audit["duplicateRowCountDates"]+=1
            if vol is not None:seen_volume.add(day)
            if count is not None:seen_rows.add(day)
            records[day]={"price":price,"volume":vol,"rowCount":count,"markets":market_set,"rawVolume":row.get("volumeKg"),"rawRowCount":row.get("rowCount"),"rawMarketCodes":markets}
        maps[pid]=records;audits[pid]=audit
    if set(maps)!=set(PRODUCTS):raise ValueError("prices must contain all five products")
    return maps,audits


def _input_coverage(records:dict[str,dict[dt.date,dict[str,Any]]])->dict[str,Any]:
    out={}
    windows={"train":(TRAIN_START,TRAIN_END),"evaluation":(EVAL_START,EVAL_END)}
    for product,rows in records.items():
        out[product]={}
        for label,(start,end) in windows.items():
            values=[r for d,r in rows.items() if start<=d<=end]
            out[product][label]={"priceObservationRows":len(values),"positiveVolumeRows":sum(r["volume"] is not None for r in values),
                "positiveRowCountRows":sum(r["rowCount"] is not None for r in values),"marketCodeRows":sum(bool(r["markets"]) for r in values),
                "dualMarketRows":sum(MARKET_CODES.issubset(r["markets"]) for r in values)}
    return out


def _median(values:list[float])->float|None:return statistics.median(values) if values else None


def _rich_features(records:dict[dt.date,dict[str,Any]],target:dt.date,horizon:int,ordered:list[dt.date]|None=None,prices:dict[dt.date,float]|None=None)->tuple[list[float]|None,float|None,dict[str,Any]]:
    price_values=prices if prices is not None else {d:r["price"] for d,r in records.items()}
    price_features,p0,audit=narrow._features(price_values,target,horizon,ordered)
    if price_features is None or p0 is None:return None,None,audit
    p0_date=dt.date.fromisoformat(audit["p0Date"]);end=bisect.bisect_right(ordered or sorted(records),p0_date)
    all_days=ordered or sorted(records)
    before_end=bisect.bisect_left(all_days,p0_date)
    extra=[]
    # Inclusive p0 day windows: exactly the prior 7/30/90 calendar dates plus p0.
    for window in (7,30,90):
        start=bisect.bisect_left(all_days,p0_date-dt.timedelta(days=window-1),0,end)
        vals=[records[d]["price"] for d in all_days[start:end] if narrow._positive(records[d]["price"]) ]
        logs=[math.log(v/p0) for v in vals]
        extra.extend([math.fsum(logs)/len(logs) if logs else 0.,statistics.pstdev(logs) if len(logs)>1 else 0.,
                      math.log(max(vals)/min(vals)) if vals else 0.,float(len(vals))])
    # Positive source volume rows strictly earlier than the p0 transaction date.
    for window in (7,30,90):
        start=bisect.bisect_left(all_days,p0_date-dt.timedelta(days=window),0,before_end)
        vals=[records[d]["volume"] for d in all_days[start:before_end] if records[d]["volume"] is not None]
        med=_median(vals);v0=records[p0_date]["volume"]
        extra.extend([math.log(v0/med) if v0 is not None and med is not None else 0.,float(len(vals)),float(v0 is not None and med is not None)])
    # Row count relative to prior-30-day median; market participation and coverage.
    start30=bisect.bisect_left(all_days,p0_date-dt.timedelta(days=30),0,before_end)
    prior_rows=[records[d]["rowCount"] for d in all_days[start30:before_end] if records[d]["rowCount"] is not None]
    row_med=_median(prior_rows);rows0=records[p0_date]["rowCount"]
    markets0=len(records[p0_date]["markets"] & MARKET_CODES)
    window_records=[records[d] for d in all_days[start30:before_end]]
    both=sum(MARKET_CODES.issubset(r["markets"]) for r in window_records)
    extra.extend([math.log(rows0/row_med) if rows0 is not None and row_med is not None else 0.,float(len(prior_rows)),float(rows0 is not None and row_med is not None),
                  float(markets0),float(both/len(window_records)) if window_records else 0.])
    return price_features+extra,p0,{"p0Date":p0_date.isoformat(),"p0AgeDays":audit["p0AgeDays"],"priceFeatureCount":len(price_features),"richFeatureCount":len(price_features)+len(extra)}


def _grid()->list[tuple[str,dt.date,int]]:
    return [(p,EVAL_START+dt.timedelta(days=i),h) for p in PRODUCTS for i in range((EVAL_END-EVAL_START).days+1) for h in HORIZONS]


def _narrow_rows(raw:dict[str,Any])->dict[tuple[str,dt.date,int],dict[str,Any]]:
    if raw.get("phase")!="forecast_only" or raw.get("targetValuesAttached") is not False or not isinstance(raw.get("forecasts"),list):raise ValueError("narrow Ridge manifest must be answer-free")
    rows={}
    for row in raw["forecasts"]:
        key=(row.get("product"),narrow._date(row.get("target")),row.get("horizon"))
        if key[0] not in PRODUCTS or key[1] is None or key[2] not in HORIZONS or key in rows:raise ValueError("invalid/duplicate narrow Ridge key")
        rows[key]=row
    return rows


def _fit_xgb(X,y):
    XGBRegressor=_load_xgboost()
    return XGBRegressor(objective="reg:squarederror",n_estimators=300,max_depth=3,learning_rate=.05,min_child_weight=15,
        subsample=.8,colsample_bytree=.8,reg_lambda=5,tree_method="hist",n_jobs=1,random_state=0).fit(X,y)


def _fit_rf(X,y):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=300,max_depth=8,min_samples_leaf=10,max_features=.7,bootstrap=True,n_jobs=1,random_state=42).fit(X,y)


def _fit_rich_ridge(X,y):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    return make_pipeline(StandardScaler(),Ridge(alpha=10.,fit_intercept=True)).fit(X,y)


def build(prices:dict[str,Any],narrow_manifest:dict[str,Any],fixed50:dict[str,Any],*,price_sha:str|None=None,narrow_sha:str|None=None,fixed50_sha:str|None=None)->dict[str,Any]:
    _check_helper()
    _load_xgboost()  # Hard fail before any fitting if the required isolated dependency is missing/wrong.
    records,audit=_raw_maps(prices);narrow_map=_narrow_rows(narrow_manifest);fixed_map=narrow._fixed50(fixed50)
    for key,row in narrow_map.items():
        if narrow._positive(row.get("fixed50"))!=fixed_map.get(key):raise ValueError("narrow Ridge and fixed50 manifests disagree")
    rows=[];grid=_grid()
    for product in PRODUCTS:
        rec=records[product];ordered=sorted(rec);price_values={d:r["price"] for d,r in rec.items()}
        for h in HORIZONS:
            X=[];y=[];train_dates=[]
            for day in ordered:
                if not TRAIN_START<=day<=TRAIN_END:continue
                feat,p0,_= _rich_features(rec,day,h,ordered,price_values)
                if feat is None or p0 is None:continue
                X.append(feat);y.append(math.log(rec[day]["price"]/p0));train_dates.append(day)
            models={};fit_reasons={}
            if len(y)>=MIN_TRAIN:
                for arm,fit in (("xgb",_fit_xgb),("rf",_fit_rf),("richRidge",_fit_rich_ridge)):
                    try:models[arm]=fit(X,y)
                    except Exception as exc:fit_reasons[arm]=f"fit_failed:{type(exc).__name__}"
            pending=[]
            for _,day,horizon in (k for k in grid if k[0]==product and k[2]==h):
                feat,p0,faudit=_rich_features(rec,day,h,ordered,price_values);origin=day-dt.timedelta(days=h);reason=None
                if origin<EVAL_START:reason="model_not_yet_fitted_at_origin"
                elif len(y)<MIN_TRAIN:reason="fewer_than_90_valid_training_targets"
                elif feat is None or p0 is None:reason=faudit.get("reason","stale_or_missing_price")
                baseline=narrow_map.get((product,day,h),{})
                row={"product":product,"target":day.isoformat(),"origin":origin.isoformat(),"horizon":h,"eligible":origin>=EVAL_START,
                    "xgbStatus":"not_issued" if reason or "xgb" not in models else "issued","xgbReason":reason or fit_reasons.get("xgb"),"xgb":None,
                    "rfStatus":"not_issued" if reason or "rf" not in models else "issued","rfReason":reason or fit_reasons.get("rf"),"rf":None,
                    "richRidgeStatus":"not_issued" if reason or "richRidge" not in models else "issued","richRidgeReason":reason or fit_reasons.get("richRidge"),"richRidge":None,
                    "narrowRidge":baseline.get("prediction"),"persistence":baseline.get("persistence"),"fixed50":fixed_map.get((product,day,h)),
                    "featureAudit":faudit,"training":{"count":len(y),"firstTarget":min(train_dates).isoformat() if train_dates else None,"lastTarget":max(train_dates).isoformat() if train_dates else None},"_features":feat,"_p0":p0}
                pending.append(row)
            for arm,statuskey,reasonkey in (("xgb","xgbStatus","xgbReason"),("rf","rfStatus","rfReason"),("richRidge","richRidgeStatus","richRidgeReason")):
                ready=[r for r in pending if r[statuskey]=="issued"]
                if not ready:continue
                try:
                    vals=models[arm].predict([r["_features"] for r in ready])
                    if len(vals)!=len(ready):raise ValueError("prediction_count_mismatch")
                    for row,logratio in zip(ready,vals):
                        pred=row["_p0"]*math.exp(float(logratio))
                        if not math.isfinite(pred) or pred<=0:row[statuskey]="not_issued";row[arm]=None;row[reasonkey]="nonpositive_or_nonfinite_prediction"
                        else:row[arm]=pred
                except Exception as exc:
                    for row in ready:row[statuskey]="not_issued";row[arm]=None;row[reasonkey]=f"prediction_failed:{type(exc).__name__}"
            for row in pending:row.pop("_features",None);row.pop("_p0",None);rows.append(row)
    doc={"schemaVersion":1,"phase":"forecast_only","targetValuesAttached":False,
        "protocol":{"id":"full-history-xgb-rf-rich-features-v1","products":list(PRODUCTS),"horizons":list(HORIZONS),"trainTargets":[TRAIN_START.isoformat(),TRAIN_END.isoformat()],"evaluationTargets":[EVAL_START.isoformat(),EVAL_END.isoformat()],"minimumTrainingTargets":MIN_TRAIN,
          "features":{"price":"narrow Ridge features plus 7/30/90d inclusive means log(price/p0), population std(log price), log(max/min), count","volume":"prior-p0 strict 7/30/90d positive volume medians, log(p0 volume/median), counts, masks","rowCount":"p0 rowCount/prior-30d positive median log ratio, count, mask","markets":"p0 market count and prior-30d dual-market observation share","cutoff":"origin-1 day; no filling"},
          "models":{"xgb":{"version":XGBOOST_VERSION,"objective":"reg:squarederror","n_estimators":300,"max_depth":3,"learning_rate":.05,"min_child_weight":15,"subsample":.8,"colsample_bytree":.8,"reg_lambda":5,"tree_method":"hist","n_jobs":1,"random_state":0},
             "rf":{"n_estimators":300,"max_depth":8,"min_samples_leaf":10,"max_features":.7,"bootstrap":True,"n_jobs":1,"random_state":42},"richRidge":{"pipeline":["StandardScaler","Ridge(alpha=10)"]}},"target":"log(actual/p0)","prefitOrigins":"not issued"},
        "inputSha256":{"prices":price_sha,"narrowRidgeManifest":narrow_sha,"fixed50Manifest":fixed50_sha,"reviewedNarrowHelper":NARROW_HELPER_SHA256},
        "sourceAudit":{"sourceRows":audit},"inputCoverage":_input_coverage(records),"plannedForecastCount":len(rows),"forecasts":rows}
    doc["manifestSha256"]=hashlib.sha256(_canonical(doc)).hexdigest();return doc


def build_from_paths()->dict[str,Any]:
    _check_helper()
    for path,expected,label in ((PRICES_PATH,PRICES_SHA256,"prices"),(NARROW_PATH,NARROW_SHA256,"narrow Ridge manifest"),(FIXED50_PATH,FIXED50_SHA256,"fixed50 manifest")):
        if _sha(path)!=expected:raise ValueError(f"pinned {label} hash mismatch")
    narrow_raw=json.loads(NARROW_PATH.read_text())
    if len(narrow_raw.get("forecasts",[]))!=len(_grid()):raise ValueError("pinned narrow Ridge manifest must contain 13,500 rows")
    return build(json.loads(PRICES_PATH.read_text()),narrow_raw,json.loads(FIXED50_PATH.read_text()),price_sha=PRICES_SHA256,narrow_sha=NARROW_SHA256,fixed50_sha=FIXED50_SHA256)


def verify_manifest(path:Path)->bool:
    raw=json.loads(path.read_text());digest=raw.get("manifestSha256");unsigned=dict(raw);unsigned.pop("manifestSha256",None)
    return digest==hashlib.sha256(_canonical(unsigned)).hexdigest() and _canonical(raw)==_canonical(build_from_paths())


def _dm(diff:dict[dt.date,float],bandwidth:int)->dict[str,Any]:
    n=len(diff)
    if any(not math.isfinite(v) for v in diff.values()):return {"status":"nonfinite_loss_difference","reason":"nonfinite_loss_difference","n":n,"statistic":None,"meanLossDifference":None,"pValue":1.}
    mean=math.fsum(diff.values())/n if n else None
    if n<30:return {"status":"insufficient_pairs","reason":"fewer_than_30_pairs","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.}
    values=list(diff.values());spread=max(values)-min(values);magnitude=max(abs(min(values)),abs(max(values)))
    if spread<=8*math.ulp(magnitude):return {"status":"zero_spread","reason":"constant_loss_difference_within_float_tolerance","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.}
    centered={d:v-mean for d,v in diff.items()};terms=[math.fsum(v*v for v in centered.values())]
    for lag in range(1,bandwidth+1):
        covariance=math.fsum(centered[d]*centered.get(d+dt.timedelta(days=lag),0.) for d in centered);terms.append(2*(1-lag/(bandwidth+1))*covariance)
    total=math.fsum(terms);tolerance=8*math.ulp(math.fsum(abs(v) for v in terms)) if any(terms) else 0.;variance=total/(n*n)
    if not math.isfinite(variance) or total<=tolerance:return {"status":"invalid_hac_variance","reason":"nonpositive_or_cancellation_dominated_hac_variance","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.}
    stat=mean/math.sqrt(variance);return {"status":"tested","reason":None,"n":n,"statistic":stat,"meanLossDifference":mean,"pValue":math.erfc(abs(stat)/math.sqrt(2))}


def _holm(rows:list[dict[str,Any]])->None:
    ix=sorted(range(len(rows)),key=lambda i:rows[i]["pValue"]);run=0.;m=len(rows)
    for rank,i in enumerate(ix):run=max(run,min(1.,(m-rank)*rows[i]["pValue"]));rows[i]["holmPValue"]=run;rows[i]["holmSignificant"]=run<ALPHA


def _check_manifest(manifest:dict[str,Any])->None:
    if manifest.get("phase")!="forecast_only" or manifest.get("targetValuesAttached") is not False:raise ValueError("expected answer-free manifest")
    unsigned=dict(manifest);digest=unsigned.pop("manifestSha256",None)
    if digest!=hashlib.sha256(_canonical(unsigned)).hexdigest():raise ValueError("manifest hash mismatch")
    expected=set(_grid());seen=set()
    for r in manifest.get("forecasts",[]):
        key=(r.get("product"),narrow._date(r.get("target")),r.get("horizon"))
        if key not in expected or key in seen:raise ValueError("invalid/duplicate manifest row")
        day=key[1];h=key[2];origin=narrow._date(r.get("origin"))
        if origin!=day-dt.timedelta(days=h) or r.get("eligible") is not (origin>=EVAL_START):raise ValueError("origin/eligibility mismatch")
        for pred,status in (("xgb","xgbStatus"),("rf","rfStatus"),("richRidge","richRidgeStatus")):
            if r.get(status) not in ("issued","not_issued"):raise ValueError("invalid prediction status")
            if r[status]=="issued" and not narrow._positive(r.get(pred)):raise ValueError("issued row lacks positive forecast")
            if r[status]=="not_issued" and r.get(pred) is not None:raise ValueError("not-issued row contains forecast")
        seen.add(key)
    if seen!=expected:raise ValueError("incomplete manifest grid")


def score(manifest:dict[str,Any],prices:dict[str,Any])->dict[str,Any]:
    _check_manifest(manifest);series=narrow._series(prices);actuals={(p,d):v for p,rows in series.items() for d,v in rows.items() if EVAL_START<=d<=EVAL_END}
    lookup={(r["product"],narrow._date(r["target"]),r["horizon"]):r for r in manifest["forecasts"]};metrics=[];dm=[];coverage={};temporal={};common_pairs=[]
    comparisons=(("xgb_vs_fixed50","xgb","fixed50"),("xgb_vs_rich_ridge","xgb","richRidge"),("rf_vs_fixed50","rf","fixed50"),("rf_vs_rich_ridge","rf","richRidge"),("rich_ridge_vs_narrow_ridge","richRidge","narrowRidge"),("xgb_vs_rf","xgb","rf"))
    planned_days=[EVAL_START+dt.timedelta(days=i) for i in range((EVAL_END-EVAL_START).days+1)]
    for p in PRODUCTS:
      for h in HORIZONS:
        cells=[lookup[(p,d,h)] for d in planned_days];eligible=[r for r in cells if r["eligible"]]
        allkeys=["xgb","rf","richRidge","narrowRidge","fixed50"]
        common=[]
        for r in eligible:
            day=narrow._date(r["target"]);actual=actuals.get((p,day))
            preds={"xgb":r["xgb"] if r["xgbStatus"]=="issued" else None,"rf":r["rf"] if r["rfStatus"]=="issued" else None,
                "richRidge":r["richRidge"] if r["richRidgeStatus"]=="issued" else None,"narrowRidge":r.get("narrowRidge"),"fixed50":r.get("fixed50")}
            if actual is not None and all(narrow._positive(preds[k]) for k in allkeys):
                row={"product":p,"target":day.isoformat(),"horizon":h,"actual":actual,**preds};common.append(row);common_pairs.append(row)
        for name,a,b in comparisons:
            def metric(k):
                if not common:return {"n":0,"mae":None,"mape":None}
                errs=[abs(r[k]-r["actual"]) for r in common]
                return {"n":len(common),"mae":math.fsum(errs)/len(errs),"mape":100*math.fsum(e/r["actual"] for e,r in zip(errs,common))/len(errs)}
            metrics.append({"product":p,"horizon":h,"comparison":name,"a":metric(a),"b":metric(b)})
            for loss in LOSSES:
                diffs={narrow._date(r["target"]):(abs(r[a]-r["actual"])-abs(r[b]-r["actual"]) if loss=="ae" else abs(r[a]-r["actual"])/r["actual"]-abs(r[b]-r["actual"])/r["actual"]) for r in common}
                for bw in BANDWIDTHS:dm.append({"product":p,"horizon":h,"comparison":name,"loss":loss,"bandwidth":bw,**_dm(diffs,bw)})
        cov={"planned":len(cells),"excludedPrefit":sum(not r["eligible"] for r in cells),"eligiblePlanned":len(eligible),"positiveActuals":sum((p,narrow._date(r["target"])) in actuals for r in eligible),"allFiveModelCommonPairs":len(common)}
        for arm,status in (("xgb","xgbStatus"),("rf","rfStatus"),("richRidge","richRidgeStatus")):
            cov[f"{arm}Issued"]=sum(bool(r[status]=="issued" and narrow._positive(r.get(arm))) for r in eligible)
            lost=[r["target"] for r in eligible if narrow._positive(r.get("fixed50")) and (r[status]!="issued" or not narrow._positive(r.get(arm)))]
            cov[f"{arm}CoverageLossTargets"]=lost;cov[f"{arm}CoverageLossReasons"]={r["target"]:r.get(arm+"Reason") or "prediction_unavailable" for r in eligible if r["target"] in lost}
        cov["fixed50AvailableEligible"]=sum(bool(narrow._positive(r.get("fixed50"))) for r in eligible);coverage[f"{p}/{h}"]=cov
    for p in PRODUCTS:
      for period in ("month","quarter"):
        groups={}
        for day in planned_days:
            label=f"{day.year}-{day.month:02d}" if period=="month" else f"{day.year}-Q{(day.month-1)//3+1}"
            groups.setdefault(label,[]).append(day)
        for label,days in groups.items():
            cells=[lookup[(p,d,h)] for d in days for h in HORIZONS];eligible=[r for r in cells if r["eligible"]]
            temporal[f"{p}/{period}/{label}"]={"planned":len(cells),"excludedPrefit":sum(not r["eligible"] for r in cells),"eligiblePlanned":len(eligible),
              "xgbIssued":sum(bool(r["eligible"] and r["xgbStatus"]=="issued" and narrow._positive(r.get("xgb"))) for r in cells),
              "rfIssued":sum(bool(r["eligible"] and r["rfStatus"]=="issued" and narrow._positive(r.get("rf"))) for r in cells),
              "richRidgeIssued":sum(bool(r["eligible"] and r["richRidgeStatus"]=="issued" and narrow._positive(r.get("richRidge"))) for r in cells),
              "fixed50Available":sum(bool(narrow._positive(r.get("fixed50"))) for r in eligible),
              "positiveActualTargets":sum((p,d) in actuals for d in days),"eligiblePositiveActualPairs":sum(r["eligible"] and (p,narrow._date(r["target"])) in actuals for r in cells),"allFiveModelCommonPairs":sum(1 for r in eligible if (p,narrow._date(r["target"])) in actuals and all(narrow._positive(r.get(k)) for k in ("xgb","rf","richRidge","narrowRidge","fixed50")))}
    for bw in BANDWIDTHS:
        family=[r for r in dm if r["bandwidth"]==bw]
        if len(family)!=600:raise ValueError("frozen 600-test Holm family mismatch")
        _holm(family)
    macros=[];gates=[]
    for p in PRODUCTS:
        for name,a,b in comparisons:
            cells=[r for r in metrics if r["product"]==p and r["comparison"]==name and r["a"]["n"]]
            macros.append({"product":p,"comparison":name,"horizonsIncluded":len(cells),"aEqualHorizonMae":math.fsum(r["a"]["mae"] for r in cells)/len(cells) if cells else None,"bEqualHorizonMae":math.fsum(r["b"]["mae"] for r in cells)/len(cells) if cells else None,"aEqualHorizonMape":math.fsum(r["a"]["mape"] for r in cells)/len(cells) if cells else None,"bEqualHorizonMape":math.fsum(r["b"]["mape"] for r in cells)/len(cells) if cells else None})
        for arm,comparison in (("xgb","xgb_vs_fixed50"),("rf","rf_vs_fixed50")):
            cells=[r for r in metrics if r["product"]==p and r["comparison"]==comparison and r["a"]["n"]];byh={r["horizon"]:r for r in cells}
            ma=math.fsum(r["a"]["mape"] for r in cells)/len(cells) if cells else None;mb=math.fsum(r["b"]["mape"] for r in cells)/len(cells) if cells else None
            ea=math.fsum(r["a"]["mae"] for r in cells)/len(cells) if cells else None;eb=math.fsum(r["b"]["mae"] for r in cells)/len(cells) if cells else None
            coverage_loss=any(coverage[f"{p}/{h}"][f"{arm}CoverageLossTargets"] for h in HORIZONS)
            no_harm=len(byh)==len(HORIZONS) and all(r["a"]["mape"]-r["b"]["mape"]<=1. for r in cells)
            joint=False
            for h in HORIZONS:
                q={(r["loss"],r["bandwidth"]):r for r in dm if r["product"]==p and r["horizon"]==h and r["comparison"]==comparison}
                if all(q.get((loss,bw),{}).get("holmSignificant") and q[(loss,bw)].get("meanLossDifference",0)<0 for loss in LOSSES for bw in BANDWIDTHS):joint=True
            passed=bool(len(byh)==len(HORIZONS) and ma is not None and mb is not None and ma<=mb-.5 and ea<eb and not coverage_loss and no_harm and joint)
            gates.append({"product":p,"candidate":arm,"benchmark":"fixed50","all10HorizonsHaveCommonPairs":len(byh)==len(HORIZONS),"candidateEqualHorizonMape":ma,"fixed50EqualHorizonMape":mb,"mapeImprovementPctPoints":mb-ma if ma is not None and mb is not None else None,"candidateEqualHorizonMae":ea,"fixed50EqualHorizonMae":eb,"coverageNondeclining":not coverage_loss,"noHorizonMapeWorseByMoreThan1PctPoint":no_harm,"jointAeApeHolmImprovementAtSomeHorizonBothHac":joint,"decision":"eligible_for_prospective_confirmation" if passed else "retain_existing_method"})
    selected=[]
    for p in PRODUCTS:
        candidates=[g for g in gates if g["product"]==p and g["decision"]=="eligible_for_prospective_confirmation"]
        candidates.sort(key=lambda g:(g["candidateEqualHorizonMape"],g["candidateEqualHorizonMae"],0 if g["candidate"]=="xgb" else 1))
        selected.append({"product":p,"selection":candidates[0]["candidate"] if candidates else "retain_existing_method"})
    dm_summary=[]
    for bw in BANDWIDTHS:
        family=[r for r in dm if r["bandwidth"]==bw]
        dm_summary.append({"bandwidth":bw,"familySize":len(family),
            "significantImprovement":sum(r["status"]=="tested" and r.get("holmSignificant") and r.get("meanLossDifference",0)<0 for r in family),
            "significantWorsening":sum(r["status"]=="tested" and r.get("holmSignificant") and r.get("meanLossDifference",0)>0 for r in family),
            "inconclusive":sum(r["status"]=="tested" and not r.get("holmSignificant") for r in family),
            "inferenceUnavailable":sum(r["status"]!="tested" for r in family)})
    return {"schemaVersion":1,"phase":"scored","manifestSha256":manifest["manifestSha256"],"metrics":metrics,"macros":macros,"coverage":coverage,"temporalCoverage":temporal,"candidateGates":gates,"selection":selected,"dmSummaryByBandwidth":dm_summary,"dm":dm,"commonPairs":common_pairs}


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
        if _sha(PRICES_PATH)!=PRICES_SHA256:raise SystemExit("pinned price hash mismatch")
        write_exclusive(args.output,score(json.loads(args.manifest.read_text()),json.loads(PRICES_PATH.read_text())))
    return 0

if __name__=="__main__":raise SystemExit(main())
