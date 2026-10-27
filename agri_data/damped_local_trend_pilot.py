"""Frozen damped local price-trend pilot with separated forecast and scoring phases."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math
from pathlib import Path
from typing import Any

from . import dashboard, state_space_pilot

ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
DIRECT_PATH = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
PRICE_SHA = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
DIRECT_SHA = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
PRODUCTS = tuple(dashboard.PRODUCTS)
HORIZONS = (1, 5, 10)
EVAL_START, EVAL_END = dt.date(2023, 9, 1), dt.date(2023, 11, 30)
BANDWIDTHS = (9, 28)
COMPARATORS = ("ablation", "direct", "persistence")
LOSSES = ("AE", "APE")
PLANNED = ((EVAL_END-EVAL_START).days+1)*len(PRODUCTS)*len(HORIZONS)
FAMILY = len(PRODUCTS)*len(HORIZONS)*len(COMPARATORS)*len(LOSSES)
PROTOCOL_ID = "damped-local-trend-pilot-2026-10-08"
LOOKBACK_DAYS, MIN_DATES, MIN_SPAN_DAYS, MAX_SOURCE_AGE = 28, 10, 7, 14
SLOPE_CLIP, PHI, CORRECTION_CLIP, BLEND = .02, .8, .08, .5

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
            if isinstance(r.get("actual"),bool): continue
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

def local_trend(series, origin):
    start=origin-dt.timedelta(days=LOOKBACK_DAYS)
    obs=sorted((d,p) for d,p in series.items() if start<=d<origin and positive(p))
    audit={"observationDates":[d.isoformat() for d,_ in obs],"observationCount":len(obs),"sourceDate":obs[-1][0].isoformat() if obs else None,"sourcePrice":obs[-1][1] if obs else None,"sourceAgeDays":(origin-obs[-1][0]).days if obs else None,"spanDays":(obs[-1][0]-obs[0][0]).days if len(obs)>=2 else 0,"rawMedianPairwiseLogSlope":None,"slope":None,"status":"unavailable"}
    if len(obs)<MIN_DATES: audit["reason"]="fewer_than_10_dates"; return None,audit
    span=(obs[-1][0]-obs[0][0]).days
    if span<MIN_SPAN_DAYS: audit["reason"]="span_under_7_days"; return None,audit
    age=(origin-obs[-1][0]).days
    if age>MAX_SOURCE_AGE: audit["reason"]="latest_observation_older_than_14_days"; return None,audit
    slopes=[]
    for i,(day_i,p_i) in enumerate(obs):
        for day_j,p_j in obs[i+1:]:
            gap=(day_j-day_i).days
            if gap>0: slopes.append((math.log(p_j)-math.log(p_i))/gap)
    if not slopes or any(not math.isfinite(x) for x in slopes): audit["reason"]="invalid_pairwise_slopes"; return None,audit
    raw=sorted(slopes)[len(slopes)//2] if len(slopes)%2 else (sorted(slopes)[len(slopes)//2-1]+sorted(slopes)[len(slopes)//2])/2
    slope=max(-SLOPE_CLIP,min(SLOPE_CLIP,raw))
    audit.update(rawMedianPairwiseLogSlope=raw,slope=slope,status="eligible",pairwiseSlopeCount=len(slopes),reason=None)
    return (obs[-1][0],obs[-1][1],slope),audit

def build_forecast(report, prices, input_hashes):
    series=product_series(prices); direct=direct_map(report)
    forecasts=[]
    for origin in dates(EVAL_START,EVAL_END):
        for pid in PRODUCTS:
            trend,audit=local_trend(series[pid],origin)
            for h in HORIZONS:
                target=origin+dt.timedelta(days=h); dr=direct.get((pid,h,target)); base=dr["direct"] if dr else None
                row={"product":pid,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":h,"candidate":None,"ablation":None,"direct":base,"persistence":None,"trendAudit":audit}
                if base is None:
                    row.update(candidateFallbackReason="direct_prediction_missing",ablationFallbackReason="direct_prediction_missing",directMissingReason="direct_prediction_missing",persistenceMissingReason="direct_prediction_missing")
                else:
                    if trend is None:
                        row["candidate"]=row["ablation"]=base
                        row["candidateFallbackReason"]=row["ablationFallbackReason"]=audit["reason"]
                    else:
                        source_day,source_price,slope=trend; gap=(target-source_day).days
                        damped=sum(PHI**j for j in range(1,gap+1))
                        candidate_residual=max(-CORRECTION_CLIP,min(CORRECTION_CLIP,slope*damped))
                        ablation_residual=max(-CORRECTION_CLIP,min(CORRECTION_CLIP,slope*gap))
                        candidate_trend=source_price*math.exp(candidate_residual); ablation_trend=source_price*math.exp(ablation_residual)
                        candidate=(1-BLEND)*base+BLEND*candidate_trend
                        ablation=(1-BLEND)*base+BLEND*ablation_trend
                        row.update(candidate=candidate,ablation=ablation,trendForecastGapDays=gap,dampedLogAdjustment=candidate_residual,undampedLogAdjustment=ablation_residual,candidateTrendForecast=candidate_trend,ablationTrendForecast=ablation_trend)
                        if not positive(candidate) or not positive(ablation):
                            row["candidate"]=row["ablation"]=base
                            row["candidateFallbackReason"]=row["ablationFallbackReason"]="invalid_blended_forecast"
                    source_day=max((d for d in series[pid] if d<origin),default=None)
                    if source_day is not None: row.update(persistence=series[pid][source_day],persistenceSourceDate=source_day.isoformat())
                    else: row["persistenceMissingReason"]="no_prior_price"
                forecasts.append(row)
    if len(forecasts)!=PLANNED: raise PilotError("forecast_grid_size_mismatch")
    return {"schemaVersion":1,"status":"answer_free_forecast_manifest","protocol":PROTOCOL,"inputs":input_hashes,"forecasts":forecasts}

PROTOCOL={"id":PROTOCOL_ID,"forecastPeriod":[EVAL_START.isoformat(),EVAL_END.isoformat()],"products":list(PRODUCTS),"horizons":list(HORIZONS),"plannedRows":PLANNED,"priceLookbackCalendarDays":LOOKBACK_DAYS,"eligibility":{"minimumDistinctPriceDates":MIN_DATES,"minimumSpanDays":MIN_SPAN_DAYS,"maximumLatestPriceAgeDays":MAX_SOURCE_AGE},"slope":"median all pairwise daily log-price slopes, calendar-day denominators, clipped to +/-0.02","dampingPhi":PHI,"correctionClip":[-CORRECTION_CLIP,CORRECTION_CLIP],"blend":{"directWeight":1-BLEND,"trendWeight":BLEND,"scale":"arithmetic price level"},"comparators":list(COMPARATORS),"dm":{"losses":list(LOSSES),"bandwidths":list(BANDWIDTHS),"familySize":FAMILY,"holmAlpha":.05}}

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
    for row in manifest["forecasts"]:
        pid=row["product"]; origin=dt.date.fromisoformat(row["origin"]); pred=row["persistence"]; source=row.get("persistenceSourceDate")
        if pred is None and source is not None: raise PilotError("null_persistence_has_source_date")
        if pred is not None:
            try: source_day=dt.date.fromisoformat(source)
            except (TypeError,ValueError): raise PilotError("invalid_persistence_source_date")
            if source_day>=origin or source_day not in series[pid] or not math.isclose(pred,series[pid][source_day],rel_tol=1e-12,abs_tol=1e-12): raise PilotError("persistence_source_mismatch")
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
    return {"schemaVersion":1,"status":"scored","forecastManifestSha256":manifest.get("manifestSha256"),"pairedRows":len(common),"byProductHorizon":cells,"macroMetrics":macro,"productGates":product_gate,"gates":gates,"coverage":{"plannedRows":PLANNED,"forecastCoverage":coverage,"forecastCoverageFractionOfPlannedGrid":coverage_fraction,"forecastCoverageAmongTargetObservedRows":observed_coverage,"forecastCoverageFractionAmongTargetObservedRows":observed_fraction,"rowsWithActual":len(observed_rows),"scoreableCommonRows":len(common),"dropReasons":drop,"fallbackCounts":fallbacks},"dm":{"familySize":FAMILY,"comparisons":comparisons,"bandwidths":list(BANDWIDTHS)},"sampleLimits":{"cellsBelowThirty":[k for k,v in cells.items() if v["n"]<30],"dmInferenceMinPairs":30,"limitations":["retrospective repeated exploration","snapshot price first-publication times are unknown","origin is conditioned on prior close and is not a 10 a.m. real-time forecast"]}}

def _write_immutable(path,raw):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f: f.write(raw)

def main(argv=None):
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest="phase",required=True)
    f=sp.add_parser("forecast"); f.add_argument("--prices",type=Path,default=PRICE_PATH); f.add_argument("--direct-report",type=Path,default=DIRECT_PATH); f.add_argument("--output",type=Path,required=True)
    s=sp.add_parser("score"); s.add_argument("--manifest",type=Path,required=True); s.add_argument("--prices",type=Path,default=PRICE_PATH); s.add_argument("--direct-report",type=Path,default=DIRECT_PATH); s.add_argument("--output",type=Path,required=True)
    a=ap.parse_args(argv)
    try:
        if a.phase=="forecast":
            pb=a.prices.read_bytes(); rb=a.direct_report.read_bytes()
            if sha(pb)!=PRICE_SHA: raise PilotError("pinned_price_hash_mismatch")
            if sha(rb)!=DIRECT_SHA: raise PilotError("pinned_direct_report_hash_mismatch")
            manifest=build_forecast(json.loads(rb),json.loads(pb),{"prices":sha(pb),"directReport":sha(rb),"code":sha(Path(__file__).read_bytes())})
            manifest["manifestSha256"]=sha(canon(manifest)); _write_immutable(a.output,canon(manifest))
        else:
            mb=a.manifest.read_bytes(); pb=a.prices.read_bytes(); rb=a.direct_report.read_bytes()
            if sha(pb)!=PRICE_SHA or sha(rb)!=DIRECT_SHA: raise PilotError("pinned_input_hash_mismatch")
            m=json.loads(mb); report=json.loads(rb)
            if m.get("inputs",{}).get("code")!=sha(Path(__file__).read_bytes()): raise PilotError("code_hash_mismatch")
            result=score(m,json.loads(pb),manifest_bytes=mb,prices_bytes=pb,report=report,report_bytes=rb); _write_immutable(a.output,canon(result))
        return 0
    except (OSError,ValueError,KeyError,TypeError,json.JSONDecodeError) as exc:
        ap.exit(2,f"damped_local_trend_pilot: {exc}\n")

if __name__=="__main__": raise SystemExit(main())
