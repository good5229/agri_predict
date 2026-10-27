"""Frozen 10-strategy same-pair retrospective benchmark (forecast then score)."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math, os, uuid
from pathlib import Path
from collections import defaultdict
from bisect import bisect_left, bisect_right
from . import dashboard

PROTOCOL="ten-strategy-same-pair-2026-10-08"
ROOT=Path(__file__).resolve().parents[1]
PRICE_PATH=ROOT/"outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
DIRECT_PATH=ROOT/"outputs/experiments/direct-horizon-20261006/report.json"
HIST_PATH=ROOT/"outputs/experiments/simple-ensemble-20261007/report.json"
QUANTILE_PATH=ROOT/"outputs/experiments/mape-quantile-20261007/report.json"
PRICE_SHA="f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
DIRECT_SHA="a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
HIST_SHA="7003511400f79bf354b13f0d64d580eae7215eb12ef5b15ebab0f61762291ffb"
QUANTILE_SHA="600afb4a2197596687dc0ed2b2eeb2ff119a9193ba8aacdb58ef7f85b084619b"
PRODUCTS=tuple(dashboard.PRODUCTS)
HORIZONS=tuple(range(1,11))
TARGET_START,TARGET_END=dt.date(2025,9,25),dt.date(2026,9,24)
ORIGIN_START=TARGET_START-dt.timedelta(days=10)
CUTOFF=ORIGIN_START-dt.timedelta(days=1)
TRAIN_DAYS=730; VALIDATION_DAYS=60; MAXITER=100; MAX_SOURCE_AGE=14
ALPHAS=(.1,.25,.5); BANDWIDTHS=(9,28); ALPHA=.05
METHODS=("naive","weekly_seasonal_naive","sma7","ses","log_drift_365d","direct_ridge","full_history_histgb","mape_quantile","calendar_arima_111","calendar_sarima_111_100_7")
BASELINES=METHODS[:5]; COMPLEX=("calendar_arima_111","calendar_sarima_111_100_7")
PLANNED=12508; CELLS=len(PRODUCTS)*len(HORIZONS); DM_FAMILY=9*CELLS*2


def _sha(b): return hashlib.sha256(b).hexdigest()
def _pos(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x>0
def _finite(x):
    try:
        y=float(x); return y if math.isfinite(y) else None
    except (TypeError,ValueError,OverflowError): return None
def _iso(s): return dt.date.fromisoformat(s) if isinstance(s,str) else s
def _dates(a,b): return [a+dt.timedelta(days=i) for i in range((b-a).days+1)]
def _canonical_hash(obj): return _sha(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False).encode())
def _immutable(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x",encoding="utf-8") as f:
            f.write(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n"); f.flush(); os.fsync(f.fileno())
        os.link(temp,path)
    finally:
        try: temp.unlink()
        except FileNotFoundError: pass

def _price_series(doc):
    if doc.get("schemaVersion")!=1 or not isinstance(doc.get("products"),list): raise ValueError("prices_schema_invalid")
    out={}
    for p in doc["products"]:
        pid=p.get("id")
        if pid not in PRODUCTS or pid in out: raise ValueError("prices_product_invalid")
        s={}
        for row in p.get("observations",[]):
            d=_iso(row["date"]); v=float(row["actual"])
            if not _pos(v) or d in s: raise ValueError("prices_observation_invalid")
            s[d]=v
        out[pid]=s
    if set(out)!=set(PRODUCTS): raise ValueError("prices_products_incomplete")
    return out

def _key(row,label,*,require_origin=True):
    p=row.get("product"); h=row.get("horizon"); t=_iso(row.get("target"))
    if p not in PRODUCTS or isinstance(h,bool) or not isinstance(h,int) or h not in HORIZONS: raise ValueError(f"{label}_key_invalid")
    o=_iso(row.get("origin")) if row.get("origin") else t-dt.timedelta(days=h)
    if require_origin and t-o!=dt.timedelta(days=h): raise ValueError(f"{label}_origin_horizon_mismatch")
    return (p,o,t,h)
def _number(v,label,positive=True):
    x=_finite(v)
    if x is None or (positive and x<=0): raise ValueError(f"{label}_not_positive_finite")
    return x

def _maps(direct,hist,quantile):
    if direct.get("protocol",{}).get("id")!="direct-horizon-ridge-2026-10-06": raise ValueError("direct_protocol_mismatch")
    d={}; h={}; q={}
    for doc, dst, label in ((direct,d,"direct"),(hist,h,"histgb"),(quantile,q,"quantile")):
        rows=doc.get("pairedRows")
        if not isinstance(rows,list): raise ValueError(f"{label}_paired_rows_missing")
        for r in rows:
            if not isinstance(r,dict) or r.get("_summary"): continue
            k=_key(r,label)
            if k in dst: raise ValueError(f"duplicate_{label}_key")
            dst[k]=r
    if not d: raise ValueError("direct_rows_empty")
    expected={k:r for k,r in d.items() if TARGET_START<=k[2]<=TARGET_END}
    if len(expected)!=PLANNED: raise ValueError(f"evaluation_key_count_mismatch:{len(expected)}")
    expected_keys=set(expected)
    if set(h)!=expected_keys: raise ValueError("histgb_evaluation_key_set_mismatch")
    for k,row in expected.items():
        hr=h[k]
        for field,other in (("direct","direct_ridge"),("persistence","persistence")):
            if not math.isclose(_number(row.get(field),f"direct_{field}"),_number(hr.get(other),f"histgb_{other}"),rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError(f"histgb_direct_{field}_mismatch")
    for k,row in q.items():
        if k not in expected_keys: raise ValueError("quantile_key_not_in_evaluation_grid")
    if set(q)!=expected_keys: raise ValueError("quantile_evaluation_key_set_mismatch")
    return expected,h,q

# -- classical and state-space fits ------------------------------------------------
def _spec(kind,d): return ((1,d,1),(0,0,0),"c" if d==0 else "n") if kind=="arima" else ((1,d,1),(1,0,0,7),"c" if d==0 else "n")
def _calendar_values(series,start,end):
    try: import pandas as pd
    except ImportError as exc: raise RuntimeError("pandas_required") from exc
    idx=pd.date_range(start=start,end=end,freq="D")
    vals=[math.log(series[x.date()]) if x.date() in series else float("nan") for x in idx]
    return pd.Series(vals,index=idx,dtype=float)
def _fit_calendar(series,start,end,kind,d):
    """Fit one candidate; d0 carries constant, d1 has no drift."""
    import statsmodels
    if statsmodels.__version__!="0.14.0": raise RuntimeError("statsmodels_version_mismatch")
    y=_calendar_values(series,start,end); order,seasonal,trend=_spec(kind,d)
    if kind=="arima":
        from statsmodels.tsa.arima.model import ARIMA
        model=ARIMA(y,order=order,trend=trend)
    else:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        model=SARIMAX(y,order=order,seasonal_order=seasonal,trend=trend,enforce_stationarity=True,enforce_invertibility=True)
    if kind=="arima": fit=model.fit(method_kwargs={"maxiter":MAXITER})
    else: fit=model.fit(maxiter=MAXITER,disp=False)
    if not bool(fit.mle_retvals.get("converged",False)): raise ValueError("fit_not_converged")
    params=[_finite(x) for x in fit.params]
    if any(x is None for x in params): raise ValueError("fit_nonfinite_parameters")
    return fit

def _choose_difference(series,kind,train_start,cutoff):
    val_start=cutoff-dt.timedelta(days=VALIDATION_DAYS-1); fit_end=val_start-dt.timedelta(days=1)
    audit={"kind":kind,"validationStart":val_start.isoformat(),"validationEnd":cutoff.isoformat(),"validationObservedCount":sum(val_start<=d<=cutoff for d in series),"byDifference":{},"selectedD":None,"status":"unavailable"}
    scores={}
    for d in (0,1):
        item={"d":d,"status":"unavailable","error":None,"mape":None,"fitStart":train_start.isoformat(),"fitEnd":fit_end.isoformat()}
        try:
            fit=_fit_calendar(series,train_start,fit_end,kind,d); errors=[]
            item.update(fitStatus="converged",order=list(_spec(kind,d)[0]),seasonalOrder=list(_spec(kind,d)[1]),trend=_spec(kind,d)[2])
            for day in _dates(val_start,cutoff):
                pred_mean=fit.get_forecast(steps=1).predicted_mean
                logp=float(pred_mean.iloc[-1] if hasattr(pred_mean,"iloc") else pred_mean[-1])
                pred=math.exp(logp); actual=series.get(day)
                if actual is not None:
                    if not _pos(pred): raise ValueError("validation_prediction_invalid")
                    errors.append(abs(actual-pred)/actual)
                fit=fit.append(_calendar_values(series,day,day),refit=False)
            if not errors: raise ValueError("no_validation_actuals")
            item.update(status="validated",mape=100*sum(errors)/len(errors),validationPairs=len(errors)); scores[d]=item["mape"]
        except Exception as exc: item["error"]=type(exc).__name__+":"+str(exc)
        audit["byDifference"][str(d)]=item
    if scores:
        selected=min(scores,key=lambda d:(scores[d],d)); audit.update(selectedD=selected,status="selected",selectedValidationMape=scores[selected])
    return audit

def _fit_rolling_stats(series,cutoff,alpha,train_start):
    pre=sorted((d,v) for d,v in series.items() if train_start<=d<=cutoff-dt.timedelta(days=VALIDATION_DAYS))
    valstart=cutoff-dt.timedelta(days=VALIDATION_DAYS-1)
    if not pre: return None
    level=pre[0][1]
    for _,v in pre[1:]: level=alpha*v+(1-alpha)*level
    errors=[]
    for day in _dates(valstart,cutoff):
        if day in series:
            actual=series[day]; errors.append(abs(actual-level)/actual); level=alpha*actual+(1-alpha)*level
    return (100*sum(errors)/len(errors)) if errors else None

def _select_alpha(series,cutoff,train_start):
    scores={str(a):_fit_rolling_stats(series,cutoff,a,train_start) for a in ALPHAS}
    valid=[a for a in ALPHAS if scores[str(a)] is not None and math.isfinite(scores[str(a)])]
    return (min(valid,key=lambda a:(scores[str(a)],a)) if valid else None),{"mapeByAlpha":scores,"selectedAlpha":min(valid,key=lambda a:(scores[str(a)],a)) if valid else None}

def _naive_fallback(series,origin,ordered_dates=None):
    ordered_dates=ordered_dates or sorted(series)
    i=bisect_right(ordered_dates,origin); src=ordered_dates[i-1] if i else None
    return (series[src] if src and (origin-src).days<=MAX_SOURCE_AGE else None),src

def _baseline_predictions(series,origin,target,alpha,es_level,ordered_dates=None):
    ordered_dates=ordered_dates or sorted(series)
    end=bisect_right(ordered_dates,origin); recent=ordered_dates[:end]
    last,src=_naive_fallback(series,origin,ordered_dates)
    weekly_src=next((d for d in reversed(recent) if (origin-d).days<=56 and d.isoweekday()==target.isoweekday()),None)
    weekly=series[weekly_src] if weekly_src and last else None
    sma=sum(series[d] for d in recent[-7:])/7 if len(recent)>=7 and last else None
    ses=es_level if last and alpha is not None else None
    drift_start=bisect_left(recent,origin-dt.timedelta(days=365)); drifts=recent[drift_start:]
    drift=None
    if len(drifts)>=2 and last:
        first,lastd=drifts[0],drifts[-1]; elapsed=(lastd-first).days
        if elapsed>0:
            slope=(math.log(series[lastd])-math.log(series[first]))/elapsed
            try: drift=math.exp(math.log(series[lastd])+slope*(target-origin).days)
            except OverflowError: drift=None
    vals={"naive":last,"weekly_seasonal_naive":weekly,"sma7":sma,"ses":ses,"log_drift_365d":drift}
    meta={m:{"sourceDate":(src.isoformat() if m in ("naive","ses") and src else weekly_src.isoformat() if m=="weekly_seasonal_naive" and weekly_src else None),"nativeStatus":"available" if _pos(v) else "unavailable","fallbackReason":None if _pos(v) else ("fewer_than_7_trade_observations" if m=="sma7" else "no_same_target_weekday_trade_within_56_days" if m=="weekly_seasonal_naive" else "alpha_or_state_unavailable" if m=="ses" else "insufficient_365_day_log_drift_points" if m=="log_drift_365d" else "no_recent_trade_within_14_days")} for m,v in vals.items()}
    return vals,meta

def _fit_all(pseries,first_origin,cutoff):
    train_start=first_origin-dt.timedelta(days=TRAIN_DAYS)
    # Align to an exact 730-day calendar window ending on cutoff (inclusive).
    train_start=cutoff-dt.timedelta(days=TRAIN_DAYS-1)
    audit={}; fits={}; selected={}
    alpha,aa=_select_alpha(pseries,cutoff,train_start); audit["ses"]=aa; selected["alpha"]=alpha
    for kind,name in (("arima","calendar_arima_111"),("sarima","calendar_sarima_111_100_7")):
        da=_choose_difference(pseries,kind,train_start,cutoff); audit[name]={"selection":da,"fullFitStatus":"unavailable","fullFitError":None,"fullFitOrder":list(_spec(kind,da["selectedD"])[0]) if da["selectedD"] is not None else None,"fullFitSeasonalOrder":list(_spec(kind,da["selectedD"])[1]) if da["selectedD"] is not None else None,"fullFitTrend":_spec(kind,da["selectedD"])[2] if da["selectedD"] is not None else None}
        if da["selectedD"] is None: continue
        try:
            fits[name]=_fit_calendar(pseries,train_start,cutoff,kind,da["selectedD"])
            audit[name]["fullFitStatus"]="converged"; selected[name+"D"]=da["selectedD"]
        except Exception as exc: audit[name]["fullFitError"]=type(exc).__name__+":"+str(exc)
    audit["ses"]["trainingWindow"]={"start":train_start.isoformat(),"end":cutoff.isoformat()}
    return fits,audit,selected,alpha,train_start

def _validate_input_sha(prices_raw,direct_raw,hist_raw,quantile_raw):
    got={"prices":_sha(prices_raw),"direct":_sha(direct_raw),"histgb":_sha(hist_raw),"quantile":_sha(quantile_raw)}
    expected={"prices":PRICE_SHA,"direct":DIRECT_SHA,"histgb":HIST_SHA,"quantile":QUANTILE_SHA}
    if got!=expected: raise ValueError("pinned_input_hash_mismatch")
    return got

def build(prices_raw,direct_raw,hist_raw,quantile_raw):
    hashes=_validate_input_sha(prices_raw,direct_raw,hist_raw,quantile_raw)
    ps=_price_series(json.loads(prices_raw)); direct=json.loads(direct_raw); hist=json.loads(hist_raw); quant=json.loads(quantile_raw)
    eligible,hmap,qmap=_maps(direct,hist,quant)
    first_origin=min(k[1] for k in eligible); cutoff=first_origin-dt.timedelta(days=1)
    rows=[]; audits={}; products_state={}
    # Precompute fixed alpha/order and fit calendar state models once per product.
    for p in PRODUCTS:
        fits,audit,selected,alpha,train_start=_fit_all(ps[p],first_origin,cutoff)
        audits[p]={"selectionAndFit":audit,"selected":selected,"trainStart":train_start.isoformat(),"trainEnd":cutoff.isoformat()}
        products_state[p]={"fits":fits,"alpha":alpha,"esLevel":None,"esLast":None,"appendedThrough":cutoff,"failed":set(),"trainStart":train_start}
    # Iterate by product/origin so model states append each new calendar day once.
    for p in PRODUCTS:
        state=products_state[p]; psr=ps[p]
        ordered_dates=sorted(psr)
        pre=sorted((d,v) for d,v in psr.items() if state["trainStart"]<=d<=cutoff)
        es_level=pre[0][1] if pre else None
        if state["alpha"] is not None and es_level is not None:
            for _,v in pre[1:]: es_level=state["alpha"]*v+(1-state["alpha"])*es_level
        state["esLevel"]=es_level
        keys=sorted((k for k in eligible if k[0]==p),key=lambda k:(k[1],k[3],k[2]))
        current=cutoff
        for key in keys:
            _,origin,target,h=key
            latest_idx=bisect_right(ordered_dates,origin); latest=ordered_dates[latest_idx-1] if latest_idx else None
            if latest is None or (origin-latest).days>MAX_SOURCE_AGE: raise ValueError(f"fixed_grid_origin_price_stale:{p}:{origin}:{h}")
            while current<origin:
                day=current+dt.timedelta(days=1); actual=psr.get(day)
                for name,fit in list(state["fits"].items()):
                    if name in state["failed"]: continue
                    try: state["fits"][name]=fit.append(_calendar_values(psr,day,day),refit=False)
                    except Exception as exc:
                        state["failed"].add(name); audits[p]["selectionAndFit"][name]["stateUpdateError"]={"date":day.isoformat(),"type":type(exc).__name__}
                if actual is not None:
                    if state["esLevel"] is None: state["esLevel"]=actual
                    elif state["alpha"] is not None: state["esLevel"]=state["alpha"]*actual+(1-state["alpha"])*state["esLevel"]
                current=day
            baseline,base_meta=_baseline_predictions(psr,origin,target,state["alpha"],state["esLevel"],ordered_dates)
            native={**baseline,"direct_ridge":_pos(eligible[key].get("direct")) and float(eligible[key]["direct"]) or None}
            statuses={m:{**base_meta[m],"fallback":False} for m in BASELINES}
            statuses["direct_ridge"]={"nativeStatus":"available" if native["direct_ridge"] else "unavailable","fallback":False,"fallbackReason":None,"source":"direct_report"}
            hrow=hmap[key]; qrow=qmap.get(key)
            hv=_finite(hrow.get("full_history_histgb")); qv=_finite(qrow.get("candidatePrediction")) if qrow else None
            for method,value,src in (("full_history_histgb",hv,"simple_ensemble_report"),("mape_quantile",qv,"mape_quantile_report")):
                if not _pos(value): raise ValueError(f"source_report_prediction_invalid:{method}:{p}:{target}:{h}")
                native[method]=value
                statuses[method]={"nativeStatus":"available","fallback":False,"fallbackReason":None,"source":src}
            for name in COMPLEX:
                if name not in state["fits"] or name in state["failed"]:
                    native[name]=None; statuses[name]={"nativeStatus":"unavailable","fallback":False,"fallbackReason":audits[p]["selectionAndFit"].get(name,{}).get("fullFitError") or "fit_or_state_unavailable"}
                else:
                    try:
                        forecast=state["fits"][name].get_forecast(steps=h).predicted_mean
                        logv=float(forecast.iloc[-1] if hasattr(forecast,"iloc") else forecast[-1]); value=math.exp(logv)
                        native[name]=value if _pos(value) else None
                        statuses[name]={"nativeStatus":"available" if _pos(value) else "unavailable","fallback":False,"fallbackReason":None if _pos(value) else "nonpositive_or_nonfinite_forecast"}
                    except Exception as exc:
                        native[name]=None; statuses[name]={"nativeStatus":"unavailable","fallback":False,"fallbackReason":"forecast_error:"+type(exc).__name__}
            last,src=_naive_fallback(psr,origin)
            final={}; fallback={}
            for method in METHODS:
                v=native.get(method); reason=None
                if not _pos(v):
                    if method in BASELINES or method in COMPLEX:
                        v=last; reason=statuses[method].get("fallbackReason") or "method_unavailable"
                    elif method in ("full_history_histgb","mape_quantile"):
                        raise ValueError(f"source_report_prediction_unavailable:{method}:{p}:{target}:{h}")
                    if v is not None: statuses[method].update(fallback=True,fallbackReason=reason,fallbackValueSource="direct_ridge" if method in ("full_history_histgb","mape_quantile") else "last_price",fallbackSourceDate=src.isoformat() if src else None)
                if not _pos(v): raise ValueError(f"same_pair_forecast_unavailable:{p}:{target}:{h}:{method}")
                final[method]=float(v); fallback[method]=bool(statuses[method]["fallback"])
            rows.append({"product":p,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":h,
                "predictions":final,"nativePredictions":{m:(float(native[m]) if _pos(native.get(m)) else None) for m in METHODS},
                "methodStatus":statuses,"fallback":fallback,"fallbackLastPrice":last,"fallbackLastPriceDate":src.isoformat() if src else None})
    rows.sort(key=lambda r:(r["product"],r["target"],r["horizon"]))
    m={"schemaVersion":1,"protocol":PROTOCOL,"phase":"forecast_only","targetValuesAttached":False,
       "inputSha256":hashes,"codeSha256":_sha(Path(__file__).read_bytes()),"evaluation":{"targetStart":TARGET_START.isoformat(),"targetEnd":TARGET_END.isoformat(),"earliestOrigin":ORIGIN_START.isoformat(),"cutoff":cutoff.isoformat(),"plannedPairs":PLANNED,"products":list(PRODUCTS),"horizons":list(HORIZONS)},
       "methods":list(METHODS),"precomputedMethodLineage":{"direct_ridge":{"input":"directReport","reportProtocol":direct.get("protocol")},"full_history_histgb":{"input":"histgbReport","reportProtocol":hist.get("protocol")},"mape_quantile":{"input":"quantileReport","reportProtocol":quant.get("protocol")}},"protocolDetails":{"trainingCalendarDays":TRAIN_DAYS,"maximumSourceAgeDays":MAX_SOURCE_AGE,"sesAlphas":list(ALPHAS),"validationCalendarDays":VALIDATION_DAYS,"arima":"calendar log ARIMA(1,d,1); d selected on last 60 pre-evaluation calendar days; d0 trend c, d1 trend n","sarima":"calendar log SARIMA(1,d,1)(1,0,0)[7]; same d selection/trend","arimaFitMaxiter":MAXITER,"missingCalendarDays":"NaN; appended daily; never interpolated","futureInputs":"none; origin <= only"},"trainingAudits":audits,"forecasts":rows}
    m["manifestSha256"]=_canonical_hash(m)
    return m

# -- scoring -----------------------------------------------------------------------
def _manifest_check(m,prices_raw,direct_raw,hist_raw,quant_raw):
    hashes=_validate_input_sha(prices_raw,direct_raw,hist_raw,quant_raw)
    if m.get("protocol")!=PROTOCOL or m.get("phase")!="forecast_only" or m.get("targetValuesAttached") is not False: raise ValueError("manifest_not_forecast_only")
    if m.get("inputSha256")!=hashes or m.get("codeSha256")!=_sha(Path(__file__).read_bytes()): raise ValueError("manifest_input_or_code_hash_mismatch")
    clean=dict(m); saved=clean.pop("manifestSha256",None)
    if saved!=_canonical_hash(clean): raise ValueError("manifest_hash_mismatch")
    d=json.loads(direct_raw); eligible,hist_map,quant_map=_maps(d,json.loads(hist_raw),json.loads(quant_raw))
    expected={k for k in eligible if TARGET_START<=k[2]<=TARGET_END}; seen=set()
    if len(expected)!=PLANNED: raise ValueError("pinned_evaluation_grid_invalid")
    for r in m.get("forecasts",[]):
        k=_key(r,"manifest")
        if k not in expected or k in seen: raise ValueError("manifest_grid_or_identity_invalid")
        seen.add(k)
        if set(r.get("predictions",{}))!=set(METHODS) or set(r.get("nativePredictions",{}))!=set(METHODS): raise ValueError("manifest_methods_invalid")
        for method in METHODS:
            if not _pos(r["predictions"].get(method)): raise ValueError("manifest_prediction_invalid")
            native=r["nativePredictions"][method]
            if native is not None and not _pos(native): raise ValueError("manifest_native_prediction_invalid")
        if "actual" in r or "targetValue" in r: raise ValueError("manifest_contains_answer")
    if seen!=expected or len(seen)!=PLANNED: raise ValueError("manifest_grid_incomplete")
    return expected,eligible,hist_map,quant_map

def _loss_metrics(rows,method):
    n=len(rows)
    return {"n":n,"mae":sum(abs(r["actual"]-r["predictions"][method]) for r in rows)/n if n else None,
        "mape":100*sum(abs(r["actual"]-r["predictions"][method])/r["actual"] for r in rows)/n if n else None}
def _dm(diffs,lag):
    n=len(diffs); mean=sum(v for _,v in diffs)/n if n else None
    base={"n":n,"meanLossDifference":mean,"calendarHacMaxLagDays":lag}
    if n<30:return {**base,"status":"inference_unavailable_insufficient_pairs","statistic":None,"pValue":1.}
    by={d:v-mean for d,v in diffs}; variance=sum(x*x for x in by.values())
    for k in range(1,lag+1): variance+=2*(1-k/(lag+1))*sum(v*by.get(d-dt.timedelta(days=k),0.) for d,v in by.items())
    if not math.isfinite(variance) or variance<=0:return {**base,"status":"inference_unavailable_nonpositive_variance","statistic":None,"pValue":1.}
    z=mean/(math.sqrt(variance)/n)
    return {**base,"status":"tested","statistic":z,"pValue":math.erfc(abs(z)/math.sqrt(2))}
def _holm(tests):
    ordered=sorted(tests,key=lambda x:(x["pValue"],x["product"],x["horizon"],x["method"],x["loss"])); run=0.
    for i,t in enumerate(ordered):
        run=max(run,min(1.,(len(ordered)-i)*t["pValue"])); t["holmPValue"]=run;t["holmSignificant"]=t["status"]=="tested" and run<ALPHA

def score(m,prices_raw,direct_raw,hist_raw,quant_raw):
    expected,direct_map,hist_map,quant_map=_manifest_check(m,prices_raw,direct_raw,hist_raw,quant_raw)
    prices=_price_series(json.loads(prices_raw))
    rows=[]
    for f in m["forecasts"]:
        k=_key(f,"forecast"); p,o,t,h=k; report=direct_map[k]; hist=hist_map[k]; quant=quant_map[k]; actual=_number(report.get("actual"),"direct_actual")
        observed=prices[p].get(t)
        if not _pos(observed) or not math.isclose(observed,actual,rel_tol=1e-12,abs_tol=1e-12): raise ValueError("pinned_actual_reconciliation_failed")
        for label,row in (("histgb",hist),("quantile",quant)):
            report_actual=_number(row.get("actual"),f"{label}_actual")
            if not math.isclose(actual,report_actual,rel_tol=1e-12,abs_tol=1e-12): raise ValueError(f"{label}_actual_mismatch")
        if not math.isclose(f["predictions"]["direct_ridge"],_number(report.get("direct"),"direct_prediction"),rel_tol=1e-12,abs_tol=1e-12): raise ValueError("manifest_direct_prediction_mismatch")
        for label,row,baseline_name,prediction_name,method in (
            ("histgb",hist,"direct_ridge","full_history_histgb","full_history_histgb"),
            ("quantile",quant,"directRidge","candidatePrediction","mape_quantile"),
        ):
            if not math.isclose(_number(row.get(baseline_name),f"{label}_{baseline_name}"),_number(report.get("direct"),"direct_prediction"),rel_tol=1e-12,abs_tol=1e-12): raise ValueError(f"{label}_direct_baseline_mismatch")
            if not math.isclose(_number(row.get("persistence"),f"{label}_persistence"),_number(report.get("persistence"),"direct_persistence"),rel_tol=1e-12,abs_tol=1e-12): raise ValueError(f"{label}_persistence_mismatch")
            if not math.isclose(f["predictions"][method],_number(row.get(prediction_name),f"{label}_{prediction_name}"),rel_tol=1e-12,abs_tol=1e-12): raise ValueError(f"manifest_{method}_prediction_mismatch")
        rows.append({**f,"actual":actual})
    cell_results=[]; tests={bw:[] for bw in BANDWIDTHS}; macro_by={method_name:[] for method_name in METHODS}
    fallback_totals={m:sum(bool(r["fallback"][m]) for r in rows) for m in METHODS}
    native_totals={m:sum(_pos(r["nativePredictions"][m]) for r in rows) for m in METHODS}
    for p in PRODUCTS:
        for h in HORIZONS:
            group=[r for r in rows if r["product"]==p and r["horizon"]==h]
            metrics={m:_loss_metrics(group,m) for m in METHODS}
            for method_name in METHODS: macro_by[method_name].append(metrics[method_name])
            cell={"product":p,"horizon":h,"plannedPairs":len(group),"metrics":metrics,"fallbackCounts":{m:sum(bool(r["fallback"][m]) for r in group) for m in METHODS},"nativeCounts":{m:sum(_pos(r["nativePredictions"][m]) for r in group) for m in METHODS}}
            for method in METHODS:
                if method=="direct_ridge": continue
                for loss in ("AE","APE"):
                    diffs=[]
                    for r in group:
                        y=r["actual"]; a=r["predictions"][method]; b=r["predictions"]["direct_ridge"]
                        x=abs(y-a)-abs(y-b) if loss=="AE" else 100*(abs(y-a)/y-abs(y-b)/y)
                        diffs.append((dt.date.fromisoformat(r["target"]),x))
                    for bw in BANDWIDTHS:
                        test={"product":p,"horizon":h,"method":method,"comparator":"direct_ridge","loss":loss,**_dm(diffs,bw)}; tests[bw].append(test)
            cell_results.append(cell)
    for bw in BANDWIDTHS:_holm(tests[bw])
    ranking=[]; macro_summary={}
    for method in METHODS:
        mapes=[x["mape"] for x in macro_by[method]]; maes=[x["mae"] for x in macro_by[method]]
        macro_summary[method]={"meanOf50CellMape":sum(mapes)/len(mapes),"meanOf50CellMae":sum(maes)/len(maes),"cells":len(mapes),"below10PercentCells":sum(x<10 for x in mapes),"nativePredictions":native_totals[method],"fallbacks":fallback_totals[method]}
        ranking.append({"method":method,"meanOf50CellMape":macro_summary[method]["meanOf50CellMape"],"meanOf50CellMae":macro_summary[method]["meanOf50CellMae"],"below10PercentCells":macro_summary[method]["below10PercentCells"],"rankByMape":None})
    ranking.sort(key=lambda x:(x["meanOf50CellMape"],x["method"]))
    for i,r in enumerate(ranking,1):r["rankByMape"]=i
    conclusions=[]
    primary={(x["product"],x["horizon"],x["method"],x["loss"]):x for x in tests[9]}
    for cell in cell_results:
        for method in METHODS:
            if method=="direct_ridge":continue
            ae=primary[(cell["product"],cell["horizon"],method,"AE")]; ape=primary[(cell["product"],cell["horizon"],method,"APE")]
            all_tests=(ae["status"]=="tested" and ape["status"]=="tested")
            joint_improve=all_tests and ae["holmSignificant"] and ape["holmSignificant"] and ae["meanLossDifference"]<0 and ape["meanLossDifference"]<0
            joint_worsen=all_tests and ae["holmSignificant"] and ape["holmSignificant"] and ae["meanLossDifference"]>0 and ape["meanLossDifference"]>0
            any_adverse=(ae["holmSignificant"] and ae["meanLossDifference"]>0) or (ape["holmSignificant"] and ape["meanLossDifference"]>0)
            direction="improvement" if joint_improve else "worsening_or_mixed" if any_adverse or joint_worsen else "inconclusive" if all_tests else "inference_unavailable"
            conclusions.append({"product":cell["product"],"horizon":cell["horizon"],"method":method,"conclusionHAC9":direction,"aeHac9":ae,"apeHac9":ape})
    return {"schemaVersion":1,"protocol":PROTOCOL,"inputSha256":_validate_input_sha(prices_raw,direct_raw,hist_raw,quant_raw),"manifestSha256":m["manifestSha256"],
      "coverage":{"plannedPairs":PLANNED,"scoredPairs":len(rows),"byMethod":{method:{"native":native_totals[method],"fallback":fallback_totals[method],"forecasted":sum(_pos(r["predictions"][method]) for r in rows),"fallbackReasons":dict(__import__("collections").Counter(r["methodStatus"][method].get("fallbackReason") for r in rows if r["fallback"][method]))} for method in METHODS}},
      "cellMetrics":cell_results,"macroMetrics":macro_summary,"rankingByEqual50CellMape":ranking,"internal10PercentReference":{"thresholdPercent":10,"methodsBelowMacro10Percent":[m for m in METHODS if macro_summary[m]["meanOf50CellMape"]<10]},
      "dmTests":{"HAC9":tests[9],"HAC28":tests[28],"familySizePerBandwidth":DM_FAMILY,"holmAlpha":ALPHA},"comparisonConclusions":conclusions,
      "limitations":["retrospective revised historical prices and repeatedly inspected period; exploratory, not independent confirmation","10% is an internal proxy-target reference, not official contest score or demonstrated user benefit","existing Ridge/HistGB/quantile use precomputed report values; their fit cadence differs from online-updated naive and state-space methods","calendar ARIMA forecasts calendar h with missing days represented as NaN; no interpolation"]}


def main(argv=None):
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    for cmd in ("forecast","score"):
        p=sub.add_parser(cmd); p.add_argument("--prices",type=Path,default=PRICE_PATH); p.add_argument("--direct",type=Path,default=DIRECT_PATH); p.add_argument("--histgb",type=Path,default=HIST_PATH); p.add_argument("--quantile",type=Path,default=QUANTILE_PATH)
        if cmd=="forecast":p.add_argument("--manifest",type=Path,required=True)
        else:p.add_argument("--manifest",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(argv); pb=a.prices.read_bytes(); db=a.direct.read_bytes(); hb=a.histgb.read_bytes(); qb=a.quantile.read_bytes()
    if a.cmd=="forecast":
        m=build(pb,db,hb,qb);_immutable(a.manifest,m);print(json.dumps({"manifest":str(a.manifest),"pairs":len(m["forecasts"])}))
    else:
        mb=a.manifest.read_bytes();m=json.loads(mb);result=score(m,pb,db,hb,qb);_immutable(a.output,result);print(json.dumps({"report":str(a.output),"pairs":PLANNED}))
if __name__=="__main__":main()
