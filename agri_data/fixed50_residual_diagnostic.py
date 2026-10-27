"""Read-only historical diagnostic for multiplicative fixed50 residual bias.

Protocol is frozen before reading outcomes: reconstruct historical fixed50 as
0.5*(Ridge alpha-10 + SARIMA d1/window730), then assess only 2025-03-01 through
2025-08-31. For each crop/horizon/origin, estimate an MAPE-optimal multiplier
from positive-actual fixed50 pairs whose target is in [origin-180d, origin-14d].
Require 30 unique target dates; otherwise use fixed50 unchanged. The MAPE
minimizer is the weighted median of actual/fixed50 with weights fixed50/actual;
ties choose the minimizer closest to 1, then lower. Shrink and clip it to
fixed50*(1+0.5*clip(c-1,-0.10,+0.10)). No data after 2025-08-31 is read as
outcomes, and no model is fitted. Report same-pair metrics, coverage, monthly
bias stability, and 100 paired calendar-HAC DM tests per bandwidth with Holm.
The selection gate is separate by crop: all ten horizons need 30 same pairs,
equal-horizon MAPE must improve by at least 0.5 percentage point with lower
MAE, issuance cannot decline, no horizon may worsen by more than 1 point, and
the same horizon must improve AE and APE with Holm significance at both
bandwidths. Global 50-cell macros are descriptive.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
COMPONENT_PATH=ROOT/"outputs/experiments/ridge-sarima-weight-cv-20261008/historical-components.json"
PRICES_PATH=ROOT/"outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
OUT=ROOT/"outputs/experiments/fixed50-residual-diagnostic-20261010/report.json"
COMPONENT_SHA256="6ade501d5818cca33ad973963d12349f51b5ee3bed27afcd2cc023ae972fed74"
PRICES_SHA256="f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
PRODUCTS=("cabbage","radish","onion","garlic","pepper");HORIZONS=tuple(range(1,11))
HIST_START,HIST_END=dt.date(2024,9,25),dt.date(2025,8,31)
VALID_START,VALID_END=dt.date(2025,3,1),dt.date(2025,8,31)
MIN_HISTORY=30;ROLLING_DAYS=180;MATURITY_DAYS=14;BANDWIDTHS=(9,28);ALPHA=.05
PLANNED_COMPONENT_ROWS=11_370

def _canonical(x:Any)->bytes:return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def _sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def _date(value):
    try:return dt.date.fromisoformat(value) if isinstance(value,str) else None
    except ValueError:return None
def _positive(value):
    if isinstance(value,bool):return None
    try:x=float(value)
    except (TypeError,ValueError,OverflowError):return None
    return x if math.isfinite(x) and x>0 else None

def _price_map(raw:dict[str,Any])->tuple[dict[str,dict[dt.date,float]],dict[str,int]]:
    if raw.get("schemaVersion")!=1 or not isinstance(raw.get("products"),list):raise ValueError("prices must use dashboard schemaVersion 1")
    result={};audit={}
    for product in raw["products"]:
        pid=product.get("id") if isinstance(product,dict) else None
        if pid not in PRODUCTS or pid in result or not isinstance(product.get("observations"),list):raise ValueError("unknown, duplicate, or malformed price product")
        values={};count=0
        for row in product["observations"]:
            if not isinstance(row,dict):continue
            day=_date(row.get("date"))
            if day is None or not HIST_START<=day<=HIST_END:continue
            count+=1;actual=_positive(row.get("actual"))
            if actual is None:continue
            if day in values:raise ValueError(f"duplicate positive actual {pid}/{day}")
            values[day]=actual
        result[pid]=values;audit[pid]=count
    if set(result)!=set(PRODUCTS):raise ValueError("prices must contain all five products")
    return result,audit

def _component_rows(raw:dict[str,Any],prices:dict[str,dict[dt.date,float]],*,expected_count:int|None=PLANNED_COMPONENT_ROWS)->list[dict[str,Any]]:
    if raw.get("phase")!="historical_components_only" or raw.get("targetValuesAttached") is not False or not isinstance(raw.get("forecasts"),list):raise ValueError("expected historical_components_only answer-free manifest")
    unsigned=dict(raw);digest=unsigned.pop("manifestSha256",None)
    computed=hashlib.sha256(_canonical(unsigned).rstrip(b"\n")).hexdigest()
    if digest!=computed:raise ValueError("component manifest digest mismatch")
    if raw.get("inputSha256",{}).get("prices")!=PRICES_SHA256:raise ValueError("component manifest price lineage mismatch")
    if expected_count is not None and len(raw["forecasts"])!=expected_count:raise ValueError("historical component row count mismatch")
    out=[];seen=set()
    for row in raw["forecasts"]:
        if not isinstance(row,dict) or row.get("product") not in PRODUCTS:raise ValueError("invalid component row")
        product=row["product"];target=_date(row.get("target"));origin=_date(row.get("origin"));h=row.get("horizon")
        if target is None or origin is None or h not in HORIZONS or origin!=target-dt.timedelta(days=h):raise ValueError("invalid historical component key")
        if not HIST_START<=target<=HIST_END:raise ValueError("component target outside frozen historical period")
        key=(product,target,h)
        if key in seen:raise ValueError("duplicate historical component key")
        seen.add(key)
        direct=_positive(row.get("direct"));sarima=_positive(row.get("sarima"))
        fixed50=(.5*direct+.5*sarima) if direct is not None and sarima is not None else None
        actual=prices.get(product,{}).get(target)
        out.append({"product":product,"target":target,"origin":origin,"horizon":h,"actual":actual,"fixed50":fixed50})
    return sorted(out,key=lambda r:(r["product"],r["horizon"],r["target"]))

def _weighted_median_multiplier(pairs:list[tuple[float,float]])->float:
    """MAPE minimizer for c*base: weighted median(actual/base), weight base/actual."""
    if not pairs:raise ValueError("weighted median requires pairs")
    vals=sorted((actual/base,base/actual) for actual,base in pairs)
    total=math.fsum(w for _,w in vals);half=total/2;cum=0.;lower=None
    for i,(ratio,weight) in enumerate(vals):
        cum=math.fsum((cum,weight))
        if cum>=half or math.isclose(cum,half,rel_tol=1e-14,abs_tol=1e-15):
            lower=ratio
            # At exact half every point to the next ratio minimizes weighted
            # absolute error; project 1 onto that full optimal interval.
            exact_half=math.isclose(cum,half,rel_tol=1e-14,abs_tol=1e-15)
            upper=vals[i+1][0] if exact_half and i+1<len(vals) else ratio
            return min(max(1.0,lower),upper)
    return vals[-1][0]

def _correct(row:dict[str,Any],history:dict[tuple[str,int],list[dict[str,Any]]])->dict[str,Any]:
    base=row["fixed50"]
    if base is None:return {**row,"candidate":None,"correction":{"status":"not_issued","reason":"fixed50_unavailable","n":0,"multiplier":None}}
    cutoff=row["origin"]-dt.timedelta(days=MATURITY_DAYS);start=row["origin"]-dt.timedelta(days=ROLLING_DAYS)
    prior=[r for r in history[(row["product"],row["horizon"])] if start<=r["target"]<=cutoff and r["actual"] is not None and r["fixed50"] is not None]
    if len({r["target"] for r in prior})!=len(prior):raise ValueError("duplicate calibration target date")
    if len(prior)<MIN_HISTORY:
        return {**row,"candidate":base,"correction":{"status":"fallback","reason":"fewer_than_30_mature_prior_pairs","n":len(prior),"multiplier":1.0,"appliedFactor":1.0}}
    multiplier=_weighted_median_multiplier([(r["actual"],r["fixed50"]) for r in prior])
    delta=.5*min(.10,max(-.10,multiplier-1.0));factor=1.0+delta
    return {**row,"candidate":base*factor,"correction":{"status":"corrected","reason":None,"n":len(prior),"multiplier":multiplier,"appliedFactor":factor}}

def _metrics(rows:list[dict[str,Any]],pred_field:str)->dict[str,Any]:
    valid=[r for r in rows if r.get("actual") is not None and r.get(pred_field) is not None]
    n=len(valid)
    if not n:return {"n":0,"mae":None,"mape":None}
    return {"n":n,"mae":math.fsum(abs(r["actual"]-r[pred_field]) for r in valid)/n,
            "mape":100*math.fsum(abs(r["actual"]-r[pred_field])/r["actual"] for r in valid)/n}

def _dm(diffs:dict[dt.date,float],bandwidth:int)->dict[str,Any]:
    n=len(diffs)
    if n<30:return {"status":"insufficient_pairs","reason":"fewer_than_30_unique_target_dates","n":n,"meanLossDifference":None,"statistic":None,"pValue":1.0}
    mean=math.fsum(diffs.values())/n;center={d:v-mean for d,v in diffs.items()};dates=sorted(center)
    variance_sum=math.fsum(v*v for v in center.values())
    for lag in range(1,bandwidth+1):
        cross=math.fsum(center[d]*center[d+dt.timedelta(days=lag)] for d in dates if d+dt.timedelta(days=lag) in center)
        variance_sum+=2*(1-lag/(bandwidth+1))*cross
    variance=variance_sum/(n*n)
    if not math.isfinite(variance) or variance<=0:return {"status":"zero_or_invalid_hac_variance","reason":"nonpositive_or_nonfinite_bartlett_variance","n":n,"meanLossDifference":mean,"statistic":None,"pValue":1.0}
    statistic=mean/math.sqrt(variance)
    return {"status":"tested","reason":None,"n":n,"meanLossDifference":mean,"statistic":statistic,"pValue":math.erfc(abs(statistic)/math.sqrt(2))}

def _holm(rows:list[dict[str,Any]])->None:
    order=sorted(range(len(rows)),key=lambda i:rows[i]["pValue"]);running=0.;m=len(rows)
    for rank,index in enumerate(order):
        running=max(running,min(1.,(m-rank)*rows[index]["pValue"]))
        rows[index]["holmPValue"]=running;rows[index]["holmSignificant"]=running<ALPHA

def _macro(rows,field):
    cells={(p,h):_metrics([r for r in rows if r["product"]==p and r["horizon"]==h],field) for p in PRODUCTS for h in HORIZONS}
    supported=[v for v in cells.values() if v["n"]]
    return {"validCells":len(supported),"equal50CellMae":math.fsum(v["mae"] for v in supported)/len(supported) if len(supported)==50 else None,
            "equal50CellMape":math.fsum(v["mape"] for v in supported)/len(supported) if len(supported)==50 else None,
            "cells":{f"{p}|{h}":v for (p,h),v in cells.items()}}

def _crop_metrics(rows,field):
    horizon_cells=[_metrics([r for r in rows if r["horizon"]==h],field) for h in HORIZONS]
    valid=[v for v in horizon_cells if v["n"]]
    return {"pooledRows":_metrics(rows,field),"equal10HorizonMape":math.fsum(v["mape"] for v in valid)/10 if len(valid)==10 else None,
      "equal10HorizonMae":math.fsum(v["mae"] for v in valid)/10 if len(valid)==10 else None,
      "supportedHorizons":len(valid),"byHorizon":{str(h):v for h,v in zip(HORIZONS,horizon_cells)}}

def _support_gate(cells,tests_by_bw,macro_fixed,macro_candidate,crop):
    by_crop={}
    for p in PRODUCTS:
        support_ok=all(cells[f"{p}|{h}"]["n"]>=30 for h in HORIZONS)
        f=crop[p]["fixed50"];c=crop[p]["candidate"]
        mape_gain=f["equal10HorizonMape"]-c["equal10HorizonMape"] if f["equal10HorizonMape"] is not None and c["equal10HorizonMape"] is not None else None
        mae_gain=f["equal10HorizonMae"]-c["equal10HorizonMae"] if f["equal10HorizonMae"] is not None and c["equal10HorizonMae"] is not None else None
        macro_pass=mape_gain is not None and mape_gain>=.5 and mae_gain is not None and mae_gain>0
        coverage_pass=crop[p]["candidateIssued"]>=crop[p]["fixed50Issued"]
        adverse_horizons=[h for h in HORIZONS if f["byHorizon"][str(h)]["mape"] is None or c["byHorizon"][str(h)]["mape"] is None or c["byHorizon"][str(h)]["mape"]>f["byHorizon"][str(h)]["mape"]+1.0]
        joint=[]
        if support_ok:
            for h in HORIZONS:
                improves=True
                for bw in BANDWIDTHS:
                    for loss in ("ae","ape"):
                        t=next(t for t in tests_by_bw[bw] if t["product"]==p and t["horizon"]==h and t["loss"]==loss)
                        improves &= t["status"]=="tested" and t["meanLossDifference"] is not None and t["meanLossDifference"]<0 and t["holmSignificant"]
                if improves:joint.append(h)
        reasons=[]
        if not support_ok:reasons.append("one_or_more_crop_horizons_has_fewer_than_30_pairs")
        if not macro_pass:reasons.append("crop_equal10_horizon_macro_gate_failed")
        if not coverage_pass:reasons.append("candidate_crop_coverage_below_fixed50")
        if adverse_horizons:reasons.append("one_or_more_horizons_mape_worsens_by_more_than_1pp")
        if not joint:reasons.append("no_same_crop_horizon_joint_ae_ape_holm_improvement_at_both_bandwidths")
        passed=not reasons
        by_crop[p]={"status":"candidate_passes_historical_selection" if passed else "inference_unavailable" if not support_ok else "candidate_fails_historical_selection",
          "passed":passed,"reasons":reasons,"all10HorizonsHaveAtLeast30Pairs":support_ok,"equal10HorizonMapeImprovementPp":mape_gain,"equal10HorizonMaeImprovement":mae_gain,
          "macroGatePass":macro_pass,"candidateCoverageNondeclining":coverage_pass,"horizonsWorseningMapeOver1pp":adverse_horizons,"jointImprovingHorizons":joint}
    eligible=[p for p in PRODUCTS if by_crop[p]["passed"]]
    return {"status":"one_or_more_crops_pass" if eligible else "no_crop_passes","eligibleCrops":eligible,"byCrop":by_crop,
      "global50CellMacroIsDescriptiveOnly":{"fixed50Mape":macro_fixed["equal50CellMape"],"candidateMape":macro_candidate["equal50CellMape"],"fixed50Mae":macro_fixed["equal50CellMae"],"candidateMae":macro_candidate["equal50CellMae"]},
      "rule":"separate per-crop gate: all 10 horizons >=30 pairs; equal-10-horizon MAPE improves >=0.5pp and MAE improves; crop coverage nondeclining; no horizon MAPE worsens >1pp; same crop/horizon improves AE and APE with Holm p<.05 at both HAC9 and HAC28. The 100-test Holm family includes unsupported cells with p=1."}

def _score(rows:list[dict[str,Any]])->dict[str,Any]:
    common=[r for r in rows if r["actual"] is not None and r["fixed50"] is not None and r["candidate"] is not None]
    cells={};tests_by_bw={bw:[] for bw in BANDWIDTHS}
    for p in PRODUCTS:
      for h in HORIZONS:
        group=[r for r in common if r["product"]==p and r["horizon"]==h]
        key=f"{p}|{h}";cells[key]={"n":len(group),"fixed50":_metrics(group,"fixed50"),"candidate":_metrics(group,"candidate")}
        for loss in ("ae","ape"):
            diffs={r["target"]:(abs(r["candidate"]-r["actual"])-abs(r["fixed50"]-r["actual"])) if loss=="ae" else 100*(abs(r["candidate"]-r["actual"])-abs(r["fixed50"]-r["actual"]))/r["actual"] for r in group}
            for bw in BANDWIDTHS:tests_by_bw[bw].append({"product":p,"horizon":h,"loss":loss,**_dm(diffs,bw)})
    for bw,tests in tests_by_bw.items():
        if len(tests)!=100:raise ValueError("DM multiplicity family must contain 100 tests")
        _holm(tests)
    crop={}
    for p in PRODUCTS:
        group=[r for r in common if r["product"]==p]
        crop[p]={"planned":sum(r["product"]==p for r in rows),"fixed50Issued":sum(r["product"]==p and r["fixed50"] is not None for r in rows),"actualPositive":sum(r["product"]==p and r["actual"] is not None for r in rows),"commonPairs":len(group),"fixed50":_crop_metrics(group,"fixed50"),"candidate":_crop_metrics(group,"candidate")}
    month=[]
    months=sorted({r["target"].strftime("%Y-%m") for r in rows if VALID_START<=r["target"]<=VALID_END})
    for ym in months:
        group=[r for r in common if r["target"].strftime("%Y-%m")==ym]
        cell_stability={}
        for p in PRODUCTS:
            for h in HORIZONS:
                cell=[r for r in group if r["product"]==p and r["horizon"]==h]
                cell_stability[f"{p}|{h}"]={"n":len(cell),"fixed50":_metrics(cell,"fixed50"),"candidate":_metrics(cell,"candidate"),
                    "meanSignedRelativeBiasPct":100*math.fsum((r["fixed50"]-r["actual"])/r["actual"] for r in cell)/len(cell) if cell else None,
                    "medianLogActualOverFixed50":_median([math.log(r["actual"]/r["fixed50"]) for r in cell]) if cell else None}
        month.append({"month":ym,"n":len(group),"fixed50":_metrics(group,"fixed50"),"candidate":_metrics(group,"candidate"),"cells":cell_stability})
    by_bw={str(k):v for k,v in tests_by_bw.items()}
    fixed_macro=_macro(common,"fixed50");candidate_macro=_macro(common,"candidate")
    coverage={"fixed50Issued":sum(r["fixed50"] is not None for r in rows),"candidateIssued":sum(r["candidate"] is not None for r in rows)}
    for p in PRODUCTS:crop[p]["candidateIssued"]=sum(r["product"]==p and r["candidate"] is not None for r in rows);crop[p]["fixed50Issued"]=sum(r["product"]==p and r["fixed50"] is not None for r in rows)
    gate=_support_gate(cells,tests_by_bw,fixed_macro,candidate_macro,crop)
    return {"planned":len(rows),"coverageBasis":"historical_components_only keys drawn from prior direct pairedRows; conditioned on source forecast/target-observation availability, not the full calendar opportunity grid","positiveActual":sum(r["actual"] is not None for r in rows),"fixed50Issued":coverage["fixed50Issued"],"candidateIssued":coverage["candidateIssued"],"commonPairs":len(common),"fixed50":fixed_macro,"candidate":candidate_macro,"crop":crop,"cropHorizon":cells,"monthlyBiasStability":month,"dmByBandwidth":by_bw,"supportGate":gate}

def _median(values):
    if not values:return None
    data=sorted(values);n=len(data);return data[n//2] if n%2 else (data[n//2-1]+data[n//2])/2

def diagnose(component_raw,prices_raw,*,component_sha=None,prices_sha=None):
    prices,price_audit=_price_map(prices_raw);historical=_component_rows(component_raw,prices)
    history={key:[r for r in historical if (r["product"],r["horizon"])==key] for key in ((p,h) for p in PRODUCTS for h in HORIZONS)}
    validation=[]
    for row in historical:
        if VALID_START<=row["target"]<=VALID_END:validation.append(_correct(row,history))
    result={"schemaVersion":1,"protocol":"fixed50-residual-bias-diagnostic-20261010-v1","phase":"historical_diagnostic_only","targetValuesAttached":True,
      "evaluation":{"historicalSourceTargets":[HIST_START.isoformat(),HIST_END.isoformat()],"validationTargets":[VALID_START.isoformat(),VALID_END.isoformat()],"plannedComponentRows":len(historical),"validationPlannedRows":len(validation)},
      "candidateRule":{"base":"0.5*ridge.alpha10+0.5*sarima.d1_w730","trainingWindowTarget":"[origin-180d,origin-14d] inclusive","minimumUniquePositivePairs":MIN_HISTORY,"multiplier":"weighted median(actual/fixed50, weight=fixed50/actual); MAPE minimizer tie closest to 1 then lower","applied":"1+0.5*clip(multiplier-1,-0.10,+0.10)","fallback":"fixed50 unchanged below minimum"},
      "inputSha256":{"historicalComponents":component_sha,"prices":prices_sha},"sourceAudit":{"priceRowsInHistoricalDateRange":price_audit,"componentRows":len(historical)},"results":_score(validation),
      "correctionAuditByMonth":[]}
    for ym in sorted({r["target"].strftime("%Y-%m") for r in validation}):
        group=[r for r in validation if r["target"].strftime("%Y-%m")==ym]
        result["correctionAuditByMonth"].append({"month":ym,"planned":len(group),"corrected":sum(r["correction"]["status"]=="corrected" for r in group),"fallback":sum(r["correction"]["status"]=="fallback" for r in group),"medianCalibrationPairs":_median([r["correction"]["n"] for r in group]),"medianMultiplier":_median([r["correction"]["multiplier"] for r in group if r["correction"]["status"]=="corrected"])})
    result["reportSha256"]=hashlib.sha256(_canonical(result).rstrip(b"\n")).hexdigest();return result

def _check_paths():
    if _sha(COMPONENT_PATH)!=COMPONENT_SHA256:raise ValueError("pinned historical component manifest SHA-256 mismatch")
    if _sha(PRICES_PATH)!=PRICES_SHA256:raise ValueError("pinned prices SHA-256 mismatch")

def diagnose_from_paths():
    _check_paths()
    return diagnose(json.loads(COMPONENT_PATH.read_text()),json.loads(PRICES_PATH.read_text()),component_sha=COMPONENT_SHA256,prices_sha=PRICES_SHA256)

def write_exclusive(path:Path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:stream.write(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False).encode()+b"\n")

def main(argv=None):
    ap=argparse.ArgumentParser();ap.add_argument("output",nargs="?",type=Path,default=OUT);args=ap.parse_args(argv)
    write_exclusive(args.output,diagnose_from_paths());return 0
if __name__=="__main__":raise SystemExit(main())
