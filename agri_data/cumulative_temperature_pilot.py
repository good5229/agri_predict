"""Frozen cumulative-temperature residual pilot; forecast and scoring are separate phases."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math, os, sqlite3, uuid
from pathlib import Path
from typing import Any

from . import dashboard, origin_linkage, state_space_pilot

ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
DIRECT_PATH = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
PRICE_SHA = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
DIRECT_SHA = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
PRODUCTS = tuple(dashboard.PRODUCTS)
STATIONS = ("100", "108", "261")
HORIZONS = (1, 5, 10)
TRAIN_END = dt.date(2024, 8, 7)
EVAL_START, EVAL_END = dt.date(2024, 8, 8), dt.date(2024, 10, 31)
BANDWIDTHS = (9, 28)
COMPARATORS = ("ablation", "direct", "persistence")
LOSSES = ("AE", "APE")
PLANNED = ((EVAL_END-EVAL_START).days+1)*len(PRODUCTS)*len(HORIZONS)
FAMILY = len(PRODUCTS)*len(HORIZONS)*len(COMPARATORS)*len(LOSSES)
PROTOCOL_ID = "cumulative-temperature-pilot-2026-10-08"
WINDOW = {"ablation": (7,), "candidate": (7, 30, 60)}

class PilotError(ValueError): pass

def sha(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def canon(x: Any) -> bytes: return (json.dumps(x, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
def finite(x): return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def positive(x): return finite(x) and x > 0
def dates(start,end): return (start+dt.timedelta(days=i) for i in range((end-start).days+1))

def product_series(doc):
    if doc.get("schemaVersion") != 1: raise PilotError("unexpected prices schema")
    out={}
    for p in doc.get("products",[]):
        pid=p.get("id")
        if pid not in PRODUCTS or pid in out: raise PilotError("unknown_or_duplicate_product")
        vals={}
        for r in p.get("observations",[]):
            try: day=dt.date.fromisoformat(r["date"]); val=float(r["actual"])
            except (KeyError,TypeError,ValueError): continue
            if math.isfinite(val) and val>0:
                if day in vals: raise PilotError("duplicate_price_actual")
                vals[day]=val
        out[pid]=vals
    if set(out)!=set(PRODUCTS): raise PilotError("prices_must_contain_five_products")
    return out

def direct_map(report):
    if report.get("protocol",{}).get("id")!="direct-horizon-ridge-2026-10-06": raise PilotError("direct_report_protocol_mismatch")
    out={}
    for r in report.get("pairedRows",[]):
        try:
            pid=r.get("product",r.get("productId")); target=dt.date.fromisoformat(r["target"]); h=r["horizon"]; pred=float(r["direct"])
        except (KeyError,TypeError,ValueError): continue
        if pid not in PRODUCTS or h not in HORIZONS or isinstance(h,bool) or not positive(pred): continue
        origin=target-dt.timedelta(days=h)
        if r.get("origin") and dt.date.fromisoformat(r["origin"])!=origin: raise PilotError("direct_origin_mismatch")
        key=(pid,h,target)
        if key in out: raise PilotError("duplicate_direct_prediction")
        # Answer deliberately not accessed in forecast phase.
        out[key]={"product":pid,"horizon":h,"target":target,"origin":origin,"direct":pred}
    if not out: raise PilotError("direct_report_has_no_predictions")
    return out

def weather_vectors(weather, start=EVAL_START, end=EVAL_END):
    """Return complete-window features and auditable per-station missing reasons."""
    vectors={}; audits={}
    for origin in dates(start,end):
        features={"candidate":[],"ablation":[]}; missing={}
        for station in STATIONS:
            series=weather.get(station,{})
            for length in (7,30,60):
                window=[origin-dt.timedelta(days=i) for i in range(length,0,-1)]
                vals=[series.get(day) for day in window]
                if any(not finite(v) for v in vals):
                    missing[f"{station}/{length}"]=sum(not finite(v) for v in vals)
                    value=None
                elif length==7: value=sum(vals)/7
                else: value=sum(max(v-10.,0.) for v in vals)
                if length==7:
                    features["candidate"].append(value); features["ablation"].append(value)
                else: features["candidate"].append(value)
        for mode,vals in features.items():
            if any(v is None for v in vals): features[mode]=None
        vectors[origin]=features
        audits[origin.isoformat()]={"missingWindows":missing,"ablationAvailable":features["ablation"] is not None,"candidateAvailable":features["candidate"] is not None}
    return vectors,audits

def _fit_model(rows, series, mode):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    usable=[r for r in rows if positive(series.get(r["product"],{}).get(r["target"])) and positive(r["direct"]) and r["features"] is not None]
    nday=len({r["target"] for r in usable}); audit={"trainingRows":len(usable),"distinctTrainingTargetDates":nday,"status":"insufficient_training" if nday<90 else "fitted"}
    if nday<90: return None,None,audit
    X=np.asarray([r["features"] for r in usable],dtype=float)
    y=np.asarray([math.log(series[r["product"]][r["target"]]/r["direct"]) for r in usable],dtype=float)
    if not np.isfinite(X).all() or not np.isfinite(y).all(): raise PilotError("nonfinite_training_data")
    scaler=StandardScaler().fit(X); scaler.scale_[scaler.scale_==0]=1
    model=Ridge(alpha=100,fit_intercept=True).fit(scaler.transform(X),y)
    audit.update(trainingTargetStart=min(r["target"] for r in usable).isoformat(),trainingTargetEnd=max(r["target"] for r in usable).isoformat())
    return scaler,model,audit

def build_forecast(report, prices, weather, input_hashes, provenance=None):
    series=product_series(prices); direct=direct_map(report)
    earliest=min((r["origin"] for r in direct.values() if r["target"]<=TRAIN_END),default=EVAL_START)
    vectors,weather_audit=weather_vectors(weather,min(earliest,EVAL_START),EVAL_END)
    models={}; training_audit={}
    # Train only on report target dates <= the frozen cutoff; features require windows strictly pre-origin.
    for pid in PRODUCTS:
        for h in HORIZONS:
            rows=[]
            for r in direct.values():
                if r["product"]!=pid or r["horizon"]!=h or r["target"]>TRAIN_END: continue
                feat=vectors.get(r["origin"],{}).get("candidate")
                if feat is not None: rows.append({**r,"features":feat})
            ab_rows=[]
            for r in rows:
                feat=vectors.get(r["origin"],{}).get("ablation")
                if feat is not None: ab_rows.append({**r,"features":feat})
            fitted={}
            for mode,train in (("candidate",rows),("ablation",ab_rows)):
                try: fitted[mode]=_fit_model(train,series,mode)
                except Exception as exc: fitted[mode]=(None,None,{"trainingRows":len(train),"distinctTrainingTargetDates":len({x["target"] for x in train}),"status":"fit_failed:"+type(exc).__name__})
            models[(pid,h)]=fitted; training_audit[f"{pid}/{h}"]={m:v[2] for m,v in fitted.items()}
    forecasts=[]
    for origin in dates(EVAL_START,EVAL_END):
        for pid in PRODUCTS:
            src=max((d for d in series[pid] if d<origin),default=None)
            persistence=series[pid][src] if src and (origin-src).days<=dashboard.MAX_SOURCE_AGE_DAYS else None
            for h in HORIZONS:
                target=origin+dt.timedelta(days=h); dr=direct.get((pid,h,target)); base=dr["direct"] if dr else None
                # The frozen protocol treats a pair absent from the direct report as absent for all methods.
                row_persistence,row_src=(persistence,src) if base is not None else (None,None)
                row={"product":pid,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":h,"direct":base,"persistence":row_persistence,"persistenceSourceDate":row_src.isoformat() if row_persistence is not None else None}
                for mode in ("candidate","ablation"):
                    scaler,model,audit=models[(pid,h)][mode]; vec=vectors[origin][mode]
                    reason=None; value=None
                    if base is None: reason="direct_prediction_missing"
                    elif model is None: reason=audit["status"]
                    elif vec is None: reason="weather_window_incomplete"
                    else:
                        try:
                            import numpy as np
                            corr=float(model.predict(scaler.transform(np.asarray([vec],dtype=float)))[0])
                            if not math.isfinite(corr): raise ValueError("nonfinite_correction")
                            value=base*math.exp(max(-.08,min(.08,corr)))
                            if not positive(value): raise ValueError("invalid_forecast")
                        except Exception as exc: reason="prediction_failed:"+type(exc).__name__; value=None
                    if reason and base is not None: value=base
                    row[mode]=value
                    if reason: row[mode+"FallbackReason"]=reason
                if base is None: row["directMissingReason"]="direct_prediction_missing"
                if row_persistence is None: row["persistenceMissingReason"]="direct_prediction_missing" if base is None else "no_prior_source_or_too_old"
                forecasts.append(row)
    if len(forecasts)!=PLANNED: raise PilotError("forecast_grid_size_mismatch")
    selected_records={s:{d.isoformat():{"requestId":v.get("requestId"),"jobId":v.get("jobId"),"epochId":v.get("epochId"),"responseSha256":v.get("responseSha256"),"rowNo":v.get("rowNo"),"retrievedAt":v.get("retrievedAt")} for d,v in sorted((provenance or {}).get(s,{}).items())} for s in STATIONS}
    return {"schemaVersion":1,"status":"answer_free_forecast_manifest","protocol":PROTOCOL,"inputs":input_hashes,
            "weatherSource":{"selector":"origin_linkage._select_weather","stations":list(STATIONS),"weatherAudit":weather_audit,"selectedRecordsByStationDate":selected_records},
            "trainingAudit":training_audit,"forecasts":forecasts}

PROTOCOL={"id":PROTOCOL_ID,"forecastPeriod":[EVAL_START.isoformat(),EVAL_END.isoformat()],"products":list(PRODUCTS),"horizons":list(HORIZONS),"plannedRows":PLANNED,"trainingTargetEnd":TRAIN_END.isoformat(),"minimumDistinctTrainingTargetDates":90,"target":"log(actual/direct)","estimator":"training-only StandardScaler + Ridge(alpha=100, fit_intercept=True)","correctionClip":[-.08,.08],"weatherStations":list(STATIONS),"ablationFeatures":"daily mean temperature [origin-7,origin-1] for each station","candidateFeatures":"ablation features plus station GDD sums max(avgTa-10C,0) over [origin-30,origin-1] and [origin-60,origin-1]","missingWeather":"no imputation; direct fallback, with reason","comparators":list(COMPARATORS),"dm":{"losses":list(LOSSES),"bandwidths":list(BANDWIDTHS),"familySize":FAMILY,"holmAlpha":.05}}

def validate_forecasts(doc):
    rows=doc.get("forecasts"); expected={(p,o.isoformat(),h,(o+dt.timedelta(days=h)).isoformat()) for o in dates(EVAL_START,EVAL_END) for p in PRODUCTS for h in HORIZONS}; seen=set()
    if not isinstance(rows,list) or len(rows)!=PLANNED: raise PilotError("forecast_grid_incomplete")
    for r in rows:
        if "actual" in r: raise PilotError("target_value_in_forecast_manifest")
        try: key=(r["product"],r["origin"],r["horizon"],r["target"]); o=dt.date.fromisoformat(r["origin"]); t=dt.date.fromisoformat(r["target"])
        except (KeyError,TypeError,ValueError): raise PilotError("invalid_forecast_identity")
        if key in seen or key not in expected or t!=o+dt.timedelta(days=r["horizon"]): raise PilotError("forecast_grid_or_date_invalid")
        seen.add(key)
        for mode in ("candidate","ablation","direct","persistence"):
            v=r.get(mode)
            if v is not None and not positive(v): raise PilotError("invalid_forecast_value")
        if r["candidate"] is None or r["ablation"] is None:
            # Missing methods are permitted only when direct itself is unavailable; fallback otherwise equals direct.
            if r["direct"] is not None and (r["candidate"]!=r["direct"] or r["ablation"]!=r["direct"]): raise PilotError("fallback_must_equal_direct")
        if r["direct"] is None and any(r.get(m) is not None for m in ("candidate","ablation","persistence")):
            raise PilotError("direct_report_pair_absent_all_methods_must_be_missing")
    if seen!=expected: raise PilotError("forecast_grid_incomplete")
    for r in rows:
        for mode in ("candidate","ablation"):
            reason=r.get(mode+"FallbackReason")
            if reason and r.get(mode)!=r.get("direct") and r.get("direct") is not None: raise PilotError("fallback_must_equal_direct")

def _dm(cell,candidate,benchmark,loss,bw):
    diff={dt.date.fromisoformat(r["target"]):(abs(r["actual"]-r[candidate])-abs(r["actual"]-r[benchmark]) if loss=="AE" else 100*(abs(r["actual"]-r[candidate])-abs(r["actual"]-r[benchmark]))/r["actual"]) for r in cell}
    return state_space_pilot._dm(diff,bw)

def score(manifest, prices, *, manifest_bytes=None, prices_bytes=None, report=None, report_bytes=None):
    if manifest.get("status")!="answer_free_forecast_manifest" or manifest.get("protocol")!=PROTOCOL: raise PilotError("manifest_protocol_mismatch")
    if manifest_bytes is not None:
        check=dict(manifest); stated=check.pop("manifestSha256",None)
        if sha(canon(check))!=stated: raise PilotError("manifest_hash_mismatch")
    if prices_bytes is not None and manifest.get("inputs",{}).get("prices")!=sha(prices_bytes): raise PilotError("price_hash_mismatch")
    if report_bytes is not None and manifest.get("inputs",{}).get("directReport")!=sha(report_bytes): raise PilotError("direct_report_hash_mismatch")
    series=product_series(prices); validate_forecasts(manifest)
    actual={}
    for p,vals in series.items(): actual.update({(p,d):v for d,v in vals.items()})
    # Reconcile target answers against the pinned direct report's paired rows when available.
    report_actual={}
    if report is not None:
        for r in report.get("pairedRows",[]):
            try: k=(r.get("product",r.get("productId")),dt.date.fromisoformat(r["target"])); y=float(r["actual"])
            except (KeyError,TypeError,ValueError): continue
            if k in report_actual and not math.isclose(y,report_actual[k],rel_tol=1e-10,abs_tol=1e-7): raise PilotError("direct_report_duplicate_actual_conflict")
            report_actual[k]=y
    observed_rows=[]; scored=[]; drop={"missingActual":0,"reportActualMissing":0,"reportActualMismatch":0}
    report_keys=set()
    models=("candidate","ablation","direct","persistence")
    if report is not None:
        for r in report.get("pairedRows",[]):
            try: report_keys.add((r.get("product",r.get("productId")),dt.date.fromisoformat(r["target"])))
            except (KeyError,TypeError,ValueError): continue
    for f in manifest["forecasts"]:
        k=(f["product"],dt.date.fromisoformat(f["target"])); y=actual.get(k)
        if y is None: drop["missingActual"]+=1; continue
        row={**f,"actual":y}; observed_rows.append(row)
        if report is not None and k in report_keys:
            ry=report_actual.get(k)
            if ry is None: drop["reportActualMissing"]+=1; raise PilotError("report_pair_missing_actual")
            if not math.isclose(y,ry,rel_tol=1e-10,abs_tol=1e-7): drop["reportActualMismatch"]+=1; raise PilotError("price_report_actual_mismatch")
        elif report is not None: drop["reportActualMissing"]+=1
        scored.append(row)
    common=[r for r in scored if all(positive(r.get(m)) for m in models)]
    def metrics(rows,m):
        return {"n":len(rows),"mae":sum(abs(r["actual"]-r[m]) for r in rows)/len(rows) if rows else None,"mape":100*sum(abs(r["actual"]-r[m])/r["actual"] for r in rows)/len(rows) if rows else None}
    cells={}; comparisons=[]
    for p in PRODUCTS:
        for h in HORIZONS:
            cell=[r for r in common if r["product"]==p and r["horizon"]==h]; cells[f"{p}/{h}"]={"n":len(cell),"metrics":{m:metrics(cell,m) for m in models}}
            for benchmark in COMPARATORS:
                for loss in LOSSES:
                    comparisons.append({"product":p,"horizon":h,"comparator":benchmark,"loss":loss,"n":len(cell),"dm":{str(b):_dm(cell,"candidate",benchmark,loss,b) for b in BANDWIDTHS}})
    for b in BANDWIDTHS: state_space_pilot._holm([x["dm"][str(b)] for x in comparisons])
    for item in comparisons:
        item["conclusion"]={}
        for b in BANDWIDTHS:
            test=item["dm"][str(b)]; mean=test.get("meanLossDifference")
            item["conclusion"][str(b)]=("inference_unavailable" if test.get("status")!="tested" else
                "improvement" if test.get("holmSignificant") and mean<0 else
                "worsening" if test.get("holmSignificant") and mean>0 else "inconclusive")
    macro={m:{metric: (sum(cells[f"{p}/{h}"]["metrics"][m][metric] for p in PRODUCTS for h in HORIZONS)/15 if all(cells[f"{p}/{h}"]["metrics"][m][metric] is not None for p in PRODUCTS for h in HORIZONS) else None) for metric in ("mae","mape")} for m in models}
    product_gate={}
    for p in PRODUCTS:
        product_gate[p]={"mapeDeltaPctPoints":sum(cells[f"{p}/{h}"]["metrics"]["candidate"]["mape"]-cells[f"{p}/{h}"]["metrics"]["direct"]["mape"] for h in HORIZONS)/3 if all(cells[f"{p}/{h}"]["metrics"]["candidate"]["mape"] is not None and cells[f"{p}/{h}"]["metrics"]["direct"]["mape"] is not None for h in HORIZONS) else None,"equalHorizonMaeWorseningPct":None}
        vals=[(cells[f"{p}/{h}"]["metrics"]["candidate"]["mae"],cells[f"{p}/{h}"]["metrics"]["direct"]["mae"]) for h in HORIZONS]
        if all(a is not None and b is not None for a,b in vals):
            ca=sum(a for a,_ in vals); dr=sum(b for _,b in vals)
            if dr>0: product_gate[p]["equalHorizonMaeWorseningPct"]=100*(ca-dr)/dr
    coverage={m:sum(positive(r.get(m)) for r in manifest["forecasts"]) for m in models}
    observed_coverage={m:sum(positive(r.get(m)) for r in observed_rows) for m in models}
    coverage_fraction={m:coverage[m]/PLANNED for m in models}
    observed_fraction={m:observed_coverage[m]/len(observed_rows) if observed_rows else None for m in models}
    gates={"all15CellsCommonPairsAtLeast30":all(cells[f"{p}/{h}"]["n"]>=30 for p in PRODUCTS for h in HORIZONS),"candidateCoverageAtLeastDirect":coverage["candidate"]>=coverage["direct"],"candidateMacroMapeAtMost10Pct":macro["candidate"]["mape"] is not None and macro["candidate"]["mape"]<=10,"candidateImprovesVsAblationBy0_5pp":macro["candidate"]["mape"] is not None and macro["ablation"]["mape"] is not None and macro["candidate"]["mape"]<=macro["ablation"]["mape"]-.5,"candidateImprovesVsDirectBy0_5pp":macro["candidate"]["mape"] is not None and macro["direct"]["mape"] is not None and macro["candidate"]["mape"]<=macro["direct"]["mape"]-.5,"productMapeWorseningAtMost1pp":all(v["mapeDeltaPctPoints"] is not None and v["mapeDeltaPctPoints"]<=1 for v in product_gate.values()),"productMaeWorseningAtMost5Pct":all(v["equalHorizonMaeWorseningPct"] is not None and v["equalHorizonMaeWorseningPct"]<=5 for v in product_gate.values())}
    gates["expand"]=all(gates.values())
    fallbacks={m:{} for m in ("candidate","ablation")}
    for row in manifest["forecasts"]:
        for m in fallbacks:
            reason=row.get(m+"FallbackReason")
            if reason: fallbacks[m][reason]=fallbacks[m].get(reason,0)+1
    return {"schemaVersion":1,"status":"scored","forecastManifestSha256":manifest.get("manifestSha256"),"pairedRows":len(common),"byProductHorizon":cells,"macroMetrics":macro,"productGates":product_gate,"gates":gates,"coverage":{"plannedRows":PLANNED,"forecastCoverage":coverage,"forecastCoverageFractionOfPlannedGrid":coverage_fraction,"forecastCoverageAmongTargetObservedRows":observed_coverage,"forecastCoverageFractionAmongTargetObservedRows":observed_fraction,"rowsWithActual":len(observed_rows),"scoreableCommonRows":len(common),"dropReasons":drop,"fallbackCounts":fallbacks},"dm":{"familySize":FAMILY,"comparisons":comparisons,"bandwidths":list(BANDWIDTHS)},"sampleLimits":{"cellsBelowThirty":[k for k,v in cells.items() if v["n"]<30],"dmInferenceMinPairs":30,"limitations":["retrospective repeated exploration","weather and price first-publication times unknown","three stations are not verified crop-origin representatives","10C is a calculation threshold, not crop-specific biology","no claim of independent validation or user benefit"]}}

def _write_immutable(path,raw):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f: f.write(raw)

def main(argv=None):
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest="phase",required=True)
    f=sp.add_parser("forecast"); f.add_argument("--prices",type=Path,default=PRICE_PATH); f.add_argument("--direct-report",type=Path,default=DIRECT_PATH); f.add_argument("--db",type=Path,default=ROOT/"data/agri.sqlite"); f.add_argument("--output",type=Path,required=True)
    s=sp.add_parser("score"); s.add_argument("--manifest",type=Path,required=True); s.add_argument("--prices",type=Path,default=PRICE_PATH); s.add_argument("--direct-report",type=Path,default=DIRECT_PATH); s.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(argv)
    try:
        if a.phase=="forecast":
            pb=a.prices.read_bytes(); rb=a.direct_report.read_bytes()
            if sha(pb)!=PRICE_SHA: raise PilotError("pinned_price_hash_mismatch")
            if sha(rb)!=DIRECT_SHA: raise PilotError("pinned_direct_report_hash_mismatch")
            # Weather range includes all possible training and evaluation windows.
            start,end=dt.date(2022,11,1),dt.date(2024,10,31)
            db=sqlite3.connect(f"file:{a.db.resolve()}?mode=ro",uri=True); db.row_factory=sqlite3.Row; db.execute("PRAGMA query_only=ON")
            try: weather,prov,meta=origin_linkage._select_weather(db,start,end)
            finally: db.close()
            temps={st:{day:vals.get("avgTa") for day,vals in rec.items()} for st,rec in weather.items() if st in STATIONS}
            manifest=build_forecast(json.loads(rb),json.loads(pb),temps,{"prices":sha(pb),"directReport":sha(rb),"code":sha(Path(__file__).read_bytes()),"weatherSelection":meta,"weatherProvenanceSha256":sha(canon({s:{d.isoformat():v for d,v in prov.get(s,{}).items()} for s in STATIONS}))},prov)
            manifest["manifestSha256"]=sha(canon(manifest)); _write_immutable(a.output,canon(manifest))
        else:
            mb=a.manifest.read_bytes(); pb=a.prices.read_bytes(); rb=a.direct_report.read_bytes()
            if sha(pb)!=PRICE_SHA or sha(rb)!=DIRECT_SHA: raise PilotError("pinned_input_hash_mismatch")
            m=json.loads(mb); report=json.loads(rb)
            if m.get("inputs",{}).get("code")!=sha(Path(__file__).read_bytes()): raise PilotError("code_hash_mismatch")
            result=score(m,json.loads(pb),manifest_bytes=mb,prices_bytes=pb,report=report,report_bytes=rb); _write_immutable(a.output,canon(result))
        return 0
    except (OSError,sqlite3.Error,ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
        ap.exit(2,f"cumulative_temperature_pilot: {exc}\n")

if __name__=="__main__": raise SystemExit(main())
