"""Frozen daily onion ARIMA transferability pilot; forecasts and scoring are separate."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math, os, uuid
from pathlib import Path
from typing import Any
from . import dashboard

PROTOCOL = "onion-arima-transfer-2026-10-08"
PRICE_PATH = Path("outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json")
DIRECT_PATH = Path("outputs/experiments/direct-horizon-20261006/report.json")
PRICE_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
DIRECT_SHA256 = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
START, END = dt.date(2024, 9, 1), dt.date(2024, 11, 30)
TRAIN_START = START - dt.timedelta(days=730)
HORIZON, MIN_TRAIN, MAXITER, MAX_SOURCE_AGE = 1, 90, 100, 14
BANDWIDTHS, ALPHA = (9, 28), .05
COMPARATORS, LOSSES = ("direct", "persistence"), ("AE", "APE")
PLANNED = 91 * len(dashboard.PRODUCTS)
FAMILY = len(dashboard.PRODUCTS) * len(COMPARATORS) * len(LOSSES)


def sha(raw: bytes) -> str: return hashlib.sha256(raw).hexdigest()
def _positive(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) and x > 0
def _finite_or_none(x):
    try:
        value=float(x)
        return value if math.isfinite(value) else None
    except (TypeError,ValueError,OverflowError): return None
def dates():
    for i in range((END-START).days+1): yield START+dt.timedelta(days=i)
def _immutable(path: Path, value):
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x",encoding="utf-8") as f:
            f.write(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n"); f.flush(); os.fsync(f.fileno())
        os.link(tmp,path)
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass

def _products(raw):
    if raw.get("schemaVersion") != 1 or not isinstance(raw.get("products"),list): raise ValueError("prices_schema_invalid")
    result={}
    for p in raw["products"]:
        pid=p.get("id")
        if pid not in dashboard.PRODUCTS or pid in result: raise ValueError("prices_products_invalid")
        series={}
        for r in p.get("observations",[]):
            d=dt.date.fromisoformat(r["date"]); v=float(r["actual"])
            if not _positive(v) or d in series: raise ValueError("prices_observations_invalid")
            series[d]=v
        result[pid]=series
    if set(result)!=set(dashboard.PRODUCTS): raise ValueError("prices_products_incomplete")
    return result

def _training_prices(series):
    """Only the frozen 730-day window before evaluation is eligible for fit selection."""
    return [(d,v) for d,v in sorted(series.items()) if TRAIN_START<=d<START]

def _diagnostic(series):
    """Train-only ADF/KPSS on observed trade sequence (no calendar interpolation)."""
    result={"n":len(series),"firstDate":series[0][0].isoformat() if series else None,
            "lastDate":series[-1][0].isoformat() if series else None,
            "medianGapDays":None,"level":None,"firstDifference":None}
    if len(series)>1:
        gaps=sorted((series[i][0]-series[i-1][0]).days for i in range(1,len(series)))
        result["medianGapDays"]=float(gaps[len(gaps)//2])
    if len(series)<MIN_TRAIN: result["status"]="fewer_than_90_training_observations"; return result
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
        y=[v for _,v in series]; diff=[y[i]-y[i-1] for i in range(1,len(y))]
        def tests(x):
            a=adfuller(x,autolag="AIC"); k=kpss(x,regression="c",nlags="auto")
            return {"adfStatistic":_finite_or_none(a[0]),"adfPValue":_finite_or_none(a[1]),"adfLags":int(a[2]),
                    "kpssStatistic":_finite_or_none(k[0]),"kpssPValue":_finite_or_none(k[1]),"kpssLags":int(k[2])}
        result["level"]=tests(y); result["firstDifference"]=tests(diff)
        lev=result["level"]; d1=result["firstDifference"]
        if any(v is None for group in (lev,d1) for v in group.values() if not isinstance(v,int)):
            result["status"]="diagnostic_nonfinite"; return result
        valid=(lev["adfPValue"]>=.05 and lev["kpssPValue"]<.05 and d1["adfPValue"]<.05 and d1["kpssPValue"]>=.05)
        result["status"]="d1_supported" if valid else "stationarity_gate_failed"
    except Exception as exc: result["status"]="diagnostic_error:"+type(exc).__name__
    return result

def _fit_selected(train):
    """Fit 9 fixed-order candidates to levels with d=1; select converged finite AIC/BIC."""
    diag=_diagnostic(train); audit={"diagnostics":diag,"candidates":[],"selectedOrder":None,"status":"unavailable"}
    if diag["status"]!="d1_supported": audit["reason"]=diag["status"]; return None,audit
    try:
        import statsmodels
        if statsmodels.__version__!="0.14.0": raise RuntimeError("statsmodels_version_mismatch")
        from statsmodels.tsa.arima.model import ARIMA
        y=[v for _,v in train]; candidate_fits={}
        for p in range(3):
            for q in range(3):
                entry={"order":[p,1,q],"converged":False,"aic":None,"bic":None,"reason":None}
                try:
                    fit=ARIMA(y,order=(p,1,q),trend="n").fit(method_kwargs={"maxiter":MAXITER})
                    entry["converged"]=bool(fit.mle_retvals.get("converged",False))
                    entry["aic"]=_finite_or_none(fit.aic); entry["bic"]=_finite_or_none(fit.bic)
                    if not entry["converged"]: entry["reason"]="not_converged"
                    elif entry["aic"] is None or entry["bic"] is None: entry["reason"]="nonfinite_information_criterion"
                    else: entry["reason"]=None; candidate_fits[(p,q)]=fit
                except Exception as exc: entry["reason"]="fit_error:"+type(exc).__name__
                audit["candidates"].append(entry)
        valid=[r for r in audit["candidates"] if r["converged"] and r["reason"] is None]
        if not valid: audit.update(status="unavailable",reason="no_converged_finite_candidate"); return None,audit
        best=min(valid,key=lambda r:(r["aic"],r["bic"],r["order"][0],r["order"][2]))
        fit=candidate_fits[(best["order"][0],best["order"][2])]
        audit.update(status="selected",reason=None,selectedOrder=best["order"],selectedAic=best["aic"],selectedBic=best["bic"],statsmodelsVersion=statsmodels.__version__)
        return fit,audit
    except Exception as exc: audit.update(status="unavailable",reason="fit_error:"+type(exc).__name__); return None,audit

def _forecast_next(fit):
    try:
        value=float(fit.forecast(steps=1)[0])
        return (value if _positive(value) else None), fit
    except Exception: return None,fit

def _walkforward_rows(pid, series, fit, audit, direct):
    """Forecast once at each calendar origin, appending only newly observed trades."""
    rows=[]
    last=max((d for d in series if d<START),default=None)
    for origin in dates():
        new=[(d,v) for d,v in sorted(series.items()) if (last is None or d>last) and d<=origin]
        if fit is not None and new:
            try: fit=fit.append([v for _,v in new],refit=False)
            except Exception: fit=None
        last=max((d for d in series if d<=origin),default=last)
        stale=last is None or (origin-last).days>MAX_SOURCE_AGE
        arima=None; reason=None
        if stale: reason="last_observation_older_than_14_days"
        elif fit is None: reason=audit.get("reason") or "state_update_or_fit_unavailable"
        else:
            arima,fit=_forecast_next(fit)
            if arima is None: reason="forecast_error_or_nonpositive"
        target=origin+dt.timedelta(days=1); dval=direct.get((pid,target)); persist=series.get(last) if last else None
        rows.append({"product":pid,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":1,
            "candidate":arima,"direct":dval,"persistence":persist if _positive(persist) and not stale else None,
            "persistenceSourceDate":last.isoformat() if last and not stale else None,
            "unavailableReasons":{"candidate":reason,"direct":None if dval is not None else "direct_pair_missing","persistence":None if persist is not None and not stale else "no_recent_price"}})
    return rows

def build(prices_raw: bytes,direct_raw: bytes):
    if sha(prices_raw)!=PRICE_SHA256 or sha(direct_raw)!=DIRECT_SHA256: raise ValueError("pinned_input_hash_mismatch")
    prices=_products(json.loads(prices_raw)); direct_doc=json.loads(direct_raw)
    if direct_doc.get("protocol",{}).get("id")!="direct-horizon-ridge-2026-10-06": raise ValueError("direct_report_protocol_mismatch")
    direct={}
    for row in direct_doc.get("pairedRows",[]):
        if row.get("horizon")!=1: continue
        key=(row.get("product"),dt.date.fromisoformat(row["target"]))
        if key in direct: raise ValueError("duplicate_direct_pair")
        value=row.get("direct"); direct[key]=value if _positive(value) else None
    rows=[]; audits={}
    for pid in dashboard.PRODUCTS:
        series=prices[pid]; train=_training_prices(series)
        fit,audit=_fit_selected(train); audits[pid]=audit
        rows.extend(_walkforward_rows(pid,series,fit,audit,direct))
    m={"schemaVersion":1,"protocol":PROTOCOL,"phase":"forecast_only","targetValuesAttached":False,
       "sourceHashes":{"prices":sha(prices_raw),"directReport":sha(direct_raw)},"pinnedSourceHashes":{"prices":PRICE_SHA256,"directReport":DIRECT_SHA256},
       "codeSha256":sha(Path(__file__).read_bytes()),"period":[START.isoformat(),END.isoformat()],"products":list(dashboard.PRODUCTS),"plannedRows":PLANNED,
       "candidateProtocol":{"model":"statsmodels 0.14.0 ARIMA(p,1,q)","orders":[[p,1,q] for p in range(3) for q in range(3)],"trend":"n","maxiter":MAXITER,"trainCalendarDays":730,"minimumTrainingObservations":MIN_TRAIN,"diagnostics":"level ADF>=.05/KPSS<.05 and difference ADF<.05/KPSS>=.05","missingDays":"trade-observation sequence; no interpolation","stateUpdate":"observations dated <= origin; append(refit=False); one next trade observation forecast","staleSourceDays":MAX_SOURCE_AGE},"fitAudits":audits,"forecasts":rows}
    m["manifestSha256"]=sha(json.dumps(m,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode())
    return m

def _dm(diff,bandwidth):
    n=len(diff)
    if n<30:return {"status":"insufficient_pairs","n":n,"meanLossDifference":None,"statistic":None,"pValue":1.0}
    mean=sum(diff.values())/n; c={d:x-mean for d,x in diff.items()}; var=sum(x*x for x in c.values())
    for lag in range(1,bandwidth+1):
        cross=sum(c[d]*c.get(d+dt.timedelta(days=lag),0) for d in c)
        var+=2*(1-lag/(bandwidth+1))*cross
    vm=var/(n*n)
    if not math.isfinite(vm) or vm<=0:return {"status":"zero_or_invalid_hac_variance","n":n,"meanLossDifference":mean,"statistic":None,"pValue":1.0}
    z=mean/math.sqrt(vm); return {"status":"tested","n":n,"meanLossDifference":mean,"statistic":z,"pValue":math.erfc(abs(z)/math.sqrt(2))}
def _holm(rows):
    order=sorted(range(len(rows)),key=lambda i:rows[i]["pValue"]); running=0
    for rank,i in enumerate(order): running=max(running,min(1,(len(rows)-rank)*rows[i]["pValue"])); rows[i]["holmPValue"]=running; rows[i]["holmSignificant"]=running<ALPHA
def _metric(actual,pred,loss):
    return sum((abs(y-x) if loss=="AE" else abs(y-x)/y) for y,x in zip(actual,pred))/len(actual) if actual else None

def _manifest_valid(m,mraw,prices_raw,direct_raw):
    if sha(prices_raw)!=PRICE_SHA256 or sha(direct_raw)!=DIRECT_SHA256: raise ValueError("pinned_input_hash_mismatch")
    if m.get("protocol")!=PROTOCOL or m.get("phase")!="forecast_only" or m.get("targetValuesAttached") is not False: raise ValueError("manifest_phase_invalid")
    if m.get("sourceHashes")!={"prices":sha(prices_raw),"directReport":sha(direct_raw)}: raise ValueError("manifest_source_hash_mismatch")
    if m.get("codeSha256")!=sha(Path(__file__).read_bytes()): raise ValueError("manifest_code_hash_mismatch")
    clean=dict(m); saved=clean.pop("manifestSha256",None)
    digest=sha(json.dumps(clean,ensure_ascii=False,sort_keys=True,separators=(",",":" )).encode())
    if saved!=digest: raise ValueError("manifest_hash_mismatch")
    expected={(p,o) for p in dashboard.PRODUCTS for o in dates()}; seen=set()
    for r in m.get("forecasts",[]):
        p=r.get("product"); o=dt.date.fromisoformat(r["origin"]); t=dt.date.fromisoformat(r["target"])
        if (p,o) not in expected or (p,o) in seen or t!=o+dt.timedelta(days=1) or r.get("horizon")!=1: raise ValueError("forecast_grid_invalid")
        seen.add((p,o))
        for name in ("candidate","direct","persistence"):
            v=r.get(name)
            if v is not None and not _positive(v): raise ValueError("forecast_value_invalid")
        if "actual" in r: raise ValueError("answer_attached")
    if seen!=expected or len(m.get("forecasts",[]))!=PLANNED: raise ValueError("forecast_grid_incomplete")

def score(m,prices_raw,direct_raw,mraw):
    _manifest_valid(m,mraw,prices_raw,direct_raw); series=_products(json.loads(prices_raw)); report=json.loads(direct_raw)
    direct_expected={}
    for r in report.get("pairedRows",[]):
        if r.get("horizon")==1:
            key=(r["product"],dt.date.fromisoformat(r["target"]))
            if key in direct_expected: raise ValueError("duplicate_direct_report_pair")
            direct_expected[key]=r
    scored=[]
    for r in m["forecasts"]:
        p=r["product"]; o=dt.date.fromisoformat(r["origin"]); t=dt.date.fromisoformat(r["target"])
        actual=series[p].get(t); report_pair=direct_expected.get((p,t))
        expected=report_pair.get("direct") if report_pair is not None else None
        if report_pair is None and r["direct"] is not None: raise ValueError("direct_value_not_in_report")
        if report_pair is not None:
            report_actual=report_pair.get("actual")
            if not _positive(report_actual): raise ValueError("direct_report_actual_missing_or_invalid")
            if not _positive(actual) or not math.isclose(float(report_actual),actual,rel_tol=1e-12,abs_tol=1e-12):
                raise ValueError("direct_report_actual_mismatch")
        if expected is None and r["direct"] is not None: raise ValueError("direct_value_not_in_report")
        if expected is not None and (r["direct"] is None or not math.isclose(float(expected),r["direct"],rel_tol=1e-12,abs_tol=1e-12)): raise ValueError("direct_value_mismatch")
        scored.append({**r,"actual":actual,"pairedActual":_positive(actual)})
    cells={}; all_dm=[]
    for p in dashboard.PRODUCTS:
        pr=[r for r in scored if r["product"]==p and _positive(r["actual"])]
        cell={"planned":91,"actualRows":len(pr),"candidateCoverage":sum(_positive(r["candidate"]) for r in pr),"directCoverage":sum(_positive(r["direct"]) for r in pr),"persistenceCoverage":sum(_positive(r["persistence"]) for r in pr),"metrics":{}}
        cell["forecastCoverage"]={m:{"planned":91,"withForecast":sum(_positive(r[m]) for r in scored if r["product"]==p),
            "fullGridFraction":sum(_positive(r[m]) for r in scored if r["product"]==p)/91,
            "observedTargets":len(pr),"withForecastAmongObserved":sum(_positive(r[m]) for r in pr),
            "observedTargetFraction":sum(_positive(r[m]) for r in pr)/len(pr) if pr else None} for m in ("candidate","direct","persistence")}
        triple=[r for r in pr if all(_positive(r[x]) for x in ("candidate","direct","persistence"))]
        for model in ("candidate","direct","persistence"):
            cell["metrics"][model]={"n":len(triple),"mae":_metric([r["actual"] for r in triple],[r[model] for r in triple],"AE"),"mape":100*_metric([r["actual"] for r in triple],[r[model] for r in triple],"APE") if triple else None}
        cells[p]=cell
        for comparator in COMPARATORS:
            common=triple
            for loss in LOSSES:
                dif={dt.date.fromisoformat(r["target"]):(abs(r["actual"]-r["candidate"]) - abs(r["actual"]-r[comparator]) if loss=="AE" else abs(r["actual"]-r["candidate"])/r["actual"]-abs(r["actual"]-r[comparator])/r["actual"]) for r in common}
                for bw in BANDWIDTHS: all_dm.append({"product":p,"comparator":comparator,"loss":loss,"bandwidth":bw,**_dm(dif,bw)})
    for bw in BANDWIDTHS: _holm([x for x in all_dm if x["bandwidth"]==bw])
    macros={}
    all_crop_common=all(cells[p]["metrics"]["candidate"]["n"]>0 for p in dashboard.PRODUCTS)
    for model in ("candidate","direct","persistence"):
        vals=[cells[p]["metrics"][model]["mape"] for p in dashboard.PRODUCTS if cells[p]["metrics"][model]["mape"] is not None]
        maes=[cells[p]["metrics"][model]["mae"] for p in dashboard.PRODUCTS if cells[p]["metrics"][model]["mae"] is not None]
        macros[model]={"mae":sum(maes)/len(maes) if all_crop_common and len(maes)==len(dashboard.PRODUCTS) else None,
                       "mape":sum(vals)/len(vals) if all_crop_common and len(vals)==len(dashboard.PRODUCTS) else None,
                       "productsWithCommonRows":len(vals),"available":all_crop_common and len(vals)==len(dashboard.PRODUCTS)}
    shared={model:sum(1 for r in scored if _positive(r["actual"]) and _positive(r[model])) for model in ("candidate","direct","persistence")}
    coverage_pass=all(cells[p]["candidateCoverage"]>=.9*cells[p]["directCoverage"] for p in dashboard.PRODUCTS)
    per_product={p:{"commonPairsWithDirect":sum(1 for r in scored if r["product"]==p and all(_positive(r[x]) for x in ("actual","candidate","direct"))),"directMaeWorseningPct":None,"directMapeWorseningPp":None} for p in dashboard.PRODUCTS}
    for p in dashboard.PRODUCTS:
        a=cells[p]["metrics"]["candidate"]; b=cells[p]["metrics"]["direct"]
        per_product[p]["directMaeWorseningPct"]=(100*(a["mae"]-b["mae"])/b["mae"] if a["mae"] is not None and b["mae"] else None)
        per_product[p]["directMapeWorseningPp"]=(a["mape"]-b["mape"] if a["mape"] is not None and b["mape"] is not None else None)
    macro_pass=all(macros[m]["available"] for m in ("candidate",*COMPARATORS)) and macros["candidate"]["mape"]<=10 and all(macros["candidate"]["mape"]<=macros[m]["mape"]-.5 for m in COMPARATORS)
    pair_pass=all(per_product[p]["commonPairsWithDirect"]>=30 for p in dashboard.PRODUCTS)
    worsening_pass=all(per_product[p]["directMaeWorseningPct"] is not None and per_product[p]["directMaeWorseningPct"]<=5 and per_product[p]["directMapeWorseningPp"]<=1 for p in dashboard.PRODUCTS)
    gate={"commonPairsAtLeast30Each":pair_pass,"candidateCoverageAtLeast90PctOfDirect":coverage_pass,"macroMapeAtMost10AndHalfPointBetter":macro_pass,"eachProductDirectWorseningWithinLimits":worsening_pass,"allPass":all((pair_pass,coverage_pass,macro_pass,worsening_pass))}
    comparisons=[]
    for p in dashboard.PRODUCTS:
        for comparator in COMPARATORS:
            for loss in LOSSES:
                tests=[x for x in all_dm if x["product"]==p and x["comparator"]==comparator and x["loss"]==loss]
                primary=next(x for x in tests if x["bandwidth"]==9)
                if primary["status"]!="tested" or primary["n"]<30:
                    conclusion="inference_unavailable"
                elif primary["holmSignificant"] and primary["meanLossDifference"]<0:
                    conclusion="improvement"
                elif primary["holmSignificant"] and primary["meanLossDifference"]>0:
                    conclusion="worsening"
                else: conclusion="inconclusive"
                comparisons.append({"product":p,"comparator":comparator,"loss":loss,"n":primary["n"],
                    "candidateMetric":cells[p]["metrics"]["candidate"]["mae" if loss=="AE" else "mape"],
                    "comparatorMetric":cells[p]["metrics"][comparator]["mae" if loss=="AE" else "mape"],
                    "hac9Holm":primary,"hac28Holm":next(x for x in tests if x["bandwidth"]==28),"conclusion":conclusion})
    global_coverage={}
    for model in ("candidate","direct","persistence"):
        n=sum(_positive(r[model]) for r in scored)
        nobs=sum(_positive(r["actual"]) for r in scored)
        nok=sum(_positive(r[model]) and _positive(r["actual"]) for r in scored)
        global_coverage[model]={"planned":PLANNED,"withForecast":n,"fullGridFraction":n/PLANNED,
            "observedTargets":nobs,"withForecastAmongObserved":nok,"observedTargetFraction":nok/nobs if nobs else None}
    return {"protocol":PROTOCOL,"sourceHashes":{"prices":sha(prices_raw),"directReport":sha(direct_raw)},"coverage":{"planned":PLANNED,"actualRows":sum(c["actualRows"] for c in cells.values()),"commonRows":shared,"byMethod":global_coverage,"byProduct":cells},"macroMetrics":macros,"productGates":per_product,"expansionGate":gate,"dmTests":all_dm,"comparisonConclusions":comparisons,"limitations":["retrospective period already explored; not independent confirmation","historical price/report publication times unknown","onion paper uses KAMIS single-market simple-average price, not this signed two-market proxy","other four crops are extrapolation","no operational adoption or user benefit established"]}

def main(argv=None):
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    f=sub.add_parser("forecast"); f.add_argument("--prices",type=Path,default=PRICE_PATH); f.add_argument("--direct-report",type=Path,default=DIRECT_PATH); f.add_argument("--output",type=Path,required=True)
    s=sub.add_parser("score"); s.add_argument("--prices",type=Path,default=PRICE_PATH); s.add_argument("--direct-report",type=Path,default=DIRECT_PATH); s.add_argument("--manifest",type=Path,required=True); s.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(argv); pb=a.prices.read_bytes(); db=a.direct_report.read_bytes()
    if a.cmd=="forecast":
        m=build(pb,db); _immutable(a.output,m); print(json.dumps({"manifest":str(a.output),"rows":len(m["forecasts"])}))
    else:
        mb=a.manifest.read_bytes(); result=score(json.loads(mb),pb,db,mb); _immutable(a.output,result); print(json.dumps({"score":str(a.output),"planned":PLANNED}))
if __name__=="__main__": main()
