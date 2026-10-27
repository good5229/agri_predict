"""Frozen retrospective one-day-lag guarded update counterfactual."""
from __future__ import annotations

import argparse
from bisect import bisect_right
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import direct_horizon, national_median_ensemble as inputs, national_price_selection as base

PROTOCOL_ID="national-lag-guard-counterfactual-2026-10-09"
PRODUCTS,HORIZONS=base.PRODUCTS,base.HORIZONS
WINDOWS=inputs.WINDOWS
BANDWIDTHS=(9,28);MIN_PAIRS=30;MIN_GATE_PAIRS=20
METHOD="guarded_update";RAW_METHOD="raw_update"
BASELINES=("fixed_blend_50_50","ridge_alpha_100","calendar_sarima","persistence")
DM_COMPARATORS=(RAW_METHOD,"fixed_blend_50_50","ridge_alpha_100","calendar_sarima","persistence")
GATE_LOOKBACK_DAYS=56
UPDATE_GAIN=0.5;CLIP_LOG_CHANGE=0.10;MAPE_IMPROVEMENT_PP=0.5;MAE_RATIO_MAX=1.05
PROJECT_ROOT=Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE,DEFAULT_COVERAGE=inputs.DEFAULT_ARCHIVE,inputs.DEFAULT_COVERAGE


def _sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()
def _hash(value:Any)->str:return _sha(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode())
def _positive(value:Any)->float|None:
    try:x=float(value)
    except (TypeError,ValueError,OverflowError):return None
    return x if math.isfinite(x) and x>0 else None
def _date(value:str)->dt.date:
    try:return dt.date.fromisoformat(value)
    except (TypeError,ValueError) as exc:raise ValueError("invalid ISO date") from exc


def _forecast_key(row:dict)->tuple:
    return inputs._forecast_identity(row)


def _all_input_forecasts(development:dict,fresh:dict)->dict[tuple,dict]:
    keyed={}
    for row in [*development.get("forecasts",[]),*fresh.get("forecasts",[])]:
        key=_forecast_key(row)
        if key in keyed:raise ValueError("duplicate baseline forecast key")
        if _positive(row.get("predictions",{}).get("fixed_blend_50_50")) is None:raise ValueError("baseline row lacks positive fixed blend")
        keyed[key]=row
    expected={(p,(target-dt.timedelta(days=h)).isoformat(),target.isoformat(),h)
              for target in (dt.date(2026,1,1)+dt.timedelta(days=i) for i in range(273)) for p in PRODUCTS for h in HORIZONS}
    if not expected.issubset(keyed):raise ValueError("frozen forecasts do not cover Jan-Sep planned grid")
    return keyed


def _source_price_state(source:dict,product:str,day:dt.date,prices:dict[str,dict[dt.date,float]])->dict:
    value=prices.get(product,{}).get(day)
    if value is not None:return {"state":"positive_price","date":day.isoformat(),"price":value}
    raw=source.get("observations",{}).get(product,{}).get(day.isoformat())
    if raw is None:return {"state":"no_observation","date":day.isoformat(),"price":None,"reason":None}
    return {"state":"observed_unusable","date":day.isoformat(),"price":None,"reason":raw.get("priceReason")}


def _p0(prices:dict[str,dict[dt.date,float]],product:str,cutoff:dt.date,price_days:dict[str,list[dt.date]]|None=None)->tuple[dt.date,float]|None:
    days=(price_days or {}).get(product)
    if days is None:days=sorted(d for d,v in prices.get(product,{}).items() if _positive(v) is not None)
    pos=bisect_right(days,cutoff)-1;day=days[pos] if pos>=0 else None
    return (day,float(prices[product][day])) if day is not None else None


def _raw_update(baseline:float|None,product:str,origin:dt.date,prices:dict[str,dict[dt.date,float]],source:dict,
                price_days:dict[str,list[dt.date]]|None=None)->dict:
    p0_cutoff=origin-dt.timedelta(days=2);p1_day=origin-dt.timedelta(days=1)
    p0=_p0(prices,product,p0_cutoff,price_days);p1state=_source_price_state(source,product,p1_day,prices)
    p0_record={"date":p0[0].isoformat(),"price":p0[1]} if p0 else None
    p1=_positive(p1state.get("price"))
    if baseline is None:return {"rawUpdate":None,"p0":p0_record,"p1":p1,"p1Date":p1_day.isoformat(),"p1State":p1state,"logChange":None,"reason":"invalid_baseline"}
    if p0 is None:return {"rawUpdate":baseline,"p0":None,"p1":p1,"p1Date":p1_day.isoformat(),"p1State":p1state,"logChange":None,"reason":"no_positive_p0_asof_o_minus_2"}
    if p1 is None:
        reason="p1_absent_exact_o_minus_1" if p1state["state"]=="no_observation" else "p1_observed_unusable_exact_o_minus_1"
        return {"rawUpdate":baseline,"p0":p0_record,"p1":None,"p1Date":p1_day.isoformat(),"p1State":p1state,"logChange":None,"reason":reason}
    delta=math.log(p1/p0[1]);clipped=max(-CLIP_LOG_CHANGE,min(CLIP_LOG_CHANGE,delta))
    value=baseline*math.exp(UPDATE_GAIN*clipped)
    if not math.isfinite(value) or value<=0:return {"rawUpdate":baseline,"p0":p0_record,"p1":p1,"p1Date":p1_day.isoformat(),"p1State":p1state,"logChange":delta,"reason":"nonpositive_or_nonfinite_raw_update"}
    return {"rawUpdate":value,"p0":p0_record,"p1":p1,
        "p1State":p1state,"p1Date":p1_day.isoformat(),"logChange":delta,"clippedLogChange":clipped,"reason":None}


def _history_gate(product:str,horizon:int,origin:dt.date,all_forecasts:dict[tuple,dict],
                  prices:dict[str,dict[dt.date,float]],source:dict,price_days:dict[str,list[dt.date]]|None=None)->dict:
    lower=origin-dt.timedelta(days=GATE_LOOKBACK_DAYS+1) # o-57 inclusive
    upper=origin-dt.timedelta(days=2)
    rows=[]
    for offset in range((upper-lower).days+1):
        target=lower+dt.timedelta(days=offset)
        key=(product,(target-dt.timedelta(days=horizon)).isoformat(),target.isoformat(),horizon)
        row=all_forecasts.get(key)
        if row is None:continue
        actual=_positive(prices.get(product,{}).get(target))
        baseline=_positive(row.get("predictions",{}).get("fixed_blend_50_50"))
        if actual is None or baseline is None:continue
        origin_past=_date(row["origin"])
        update=_raw_update(baseline,product,origin_past,prices,source,price_days)
        raw=_positive(update.get("rawUpdate"))
        if raw is None or update.get("p1") is None or update.get("p0") is None or update.get("reason") is not None:continue
        rows.append({"target":target,"actual":actual,"baseline":baseline,"raw":raw})
    rows.sort(key=lambda r:r["target"])
    n=len(rows)
    if n:
        actual=[r["actual"] for r in rows];b=[r["baseline"] for r in rows];u=[r["raw"] for r in rows]
        mb=direct_horizon._metrics(actual,b);mu=direct_horizon._metrics(actual,u)
        hist_max=max(r["target"] for r in rows).isoformat();hist_min=min(r["target"] for r in rows).isoformat()
        keys_hash=_hash([r["target"].isoformat() for r in rows])
    else:mb=mu={"mae":None,"mape":None};hist_min=hist_max=None;keys_hash=_hash([])
    mape_ok=n>=MIN_GATE_PAIRS and mu["mape"]<=mb["mape"]-MAPE_IMPROVEMENT_PP
    mae_ok=n>=MIN_GATE_PAIRS and mu["mae"]<=mb["mae"]*MAE_RATIO_MAX
    active=bool(n>=MIN_GATE_PAIRS and mape_ok and mae_ok)
    return {"active":active,"status":"active" if active else "insufficient_history_pairs" if n<MIN_GATE_PAIRS else "criteria_not_met",
        "historyStart":lower.isoformat(),"historyEnd":upper.isoformat(),"historyPairCount":n,
        "historyMinTargetDate":hist_min,"historyMaxTargetDate":hist_max,"historyPairKeysSha256":keys_hash,
        "meanApeBaselinePct":mb["mape"],"meanApeRawUpdatePct":mu["mape"],
        "meanAeBaseline":mb["mae"],"meanAeRawUpdate":mu["mae"],
        "criteria":{"minimumPairs":n>=MIN_GATE_PAIRS,"mapeImprovementAtLeast0_5pp":mape_ok,"meanAeAtMost105PctBaseline":mae_ok}}


def _build_manifest(prices:dict[str,dict[dt.date,float]],source:dict,source_digests:dict,
                    development:dict,fresh:dict)->dict:
    all_forecasts=_all_input_forecasts(development,fresh)
    price_days={p:sorted(d for d,v in prices[p].items() if _positive(v) is not None) for p in PRODUCTS}
    expected={(p,(target-dt.timedelta(days=h)).isoformat(),target.isoformat(),h)
              for target in (dt.date(2026,1,1)+dt.timedelta(days=i) for i in range(273)) for p in PRODUCTS for h in HORIZONS}
    forecasts=[]
    for key in sorted(expected):
        product,origin_text,target_text,horizon=key;row=all_forecasts[key]
        origin=_date(origin_text);target=_date(target_text);baseline=_positive(row["predictions"].get("fixed_blend_50_50"))
        gate=_history_gate(product,horizon,origin,all_forecasts,prices,source,price_days)
        update=_raw_update(baseline,product,origin,prices,source,price_days)
        use_update=gate["active"] and update.get("p1") is not None and update.get("p0") is not None and update.get("reason") is None
        final=update["rawUpdate"] if use_update else baseline
        forecasts.append({"product":product,"origin":origin_text,"target":target_text,"horizon":horizon,
            "cutoff":(origin-dt.timedelta(days=2)).isoformat(),
            "predictions":{"fixed_blend_50_50":baseline,"raw_update":update["rawUpdate"],"guarded_update":final,
                           "ridge_alpha_100":row.get("predictions",{}).get("ridge_alpha_100"),
                           "calendar_sarima":row.get("predictions",{}).get("calendar_sarima"),
                           "persistence":row.get("predictions",{}).get("persistence")},
            "lineage":{"p0":update.get("p0"),"p1":{"date":update.get("p1Date"),"price":update.get("p1"),
                    "state":update.get("p1State",{}).get("state"),"sourceReason":update.get("p1State",{}).get("reason")},
                "rawUpdateReason":update.get("reason"),"rawLogChange":update.get("logChange"),
                "clippedLogChange":update.get("clippedLogChange"),"updateGain":UPDATE_GAIN,"clipLogChange":CLIP_LOG_CHANGE,
                "gate":gate,"guardedUsesRawUpdate":bool(use_update),
                "baselineLineage":row.get("lineage",{})}})
    doc={"schemaVersion":1,"protocolId":PROTOCOL_ID,"phase":"forecast_only","targetValuesAttached":False,
        "counterfactual":True,"oMinusOneAvailabilityVerified":False,
        "sourceDigests":source_digests,"inputManifestSha256":dict(inputs.MANIFEST_SHA256),
        "inputManifestContentSha256":dict(inputs.MANIFEST_CONTENT_SHA256),
        "codeSha256":_sha(Path(__file__).read_bytes()),"products":list(PRODUCTS),"horizons":list(HORIZONS),
        "window":["2026-01-01","2026-09-30"],"plannedForecastCount":len(forecasts),"forecasts":forecasts}
    doc["manifestSha256"]=_hash(doc);return doc


def build_manifest(raw_bundles:dict)->dict:
    source,dev_source,fresh_source,development,fresh=_load_all(raw_bundles)
    return _build_manifest(base._prices(source),source,{"development":dev_source["sourceDigests"],"fresh":fresh_source["sourceDigests"]},development,fresh)


def _load_all(raw_bundles:dict):
    if not isinstance(raw_bundles,dict) or set(raw_bundles)!={"development","fresh"}:raise ValueError("raw_bundles must contain development and fresh")
    development=inputs._load_frozen_manifest("development")
    fresh=inputs._load_frozen_manifest("fresh")
    dev_source,fresh_source=inputs._load_sources(raw_bundles)
    if development.get("sourceDigests")!=dev_source.get("sourceDigests") or fresh.get("sourceDigests")!=fresh_source.get("sourceDigests"):
        raise ValueError("baseline manifest and raw source digests differ")
    # The adapter used here reparses the archive and fresh JSON source with the
    # same pinned validation path as the established national scorers.
    return fresh_source,dev_source,fresh_source,development,fresh


def verify_manifest(doc:dict)->None:
    if doc.get("protocolId")!=PROTOCOL_ID or doc.get("phase")!="forecast_only" or doc.get("targetValuesAttached") is not False:
        raise ValueError("not an answer-free lag-guard manifest")
    if doc.get("counterfactual") is not True or doc.get("oMinusOneAvailabilityVerified") is not False:
        raise ValueError("counterfactual availability caveat missing")
    if doc.get("codeSha256")!=_sha(Path(__file__).read_bytes()):raise ValueError("lag-guard code hash mismatch")
    body=dict(doc);saved=body.pop("manifestSha256",None)
    if saved!=_hash(body):raise ValueError("manifest content hash mismatch")
    rows=doc.get("forecasts")
    if not isinstance(rows,list) or len(rows)!=10920 or doc.get("plannedForecastCount")!=10920:raise ValueError("planned count mismatch")
    seen=set()
    for row in rows:
        key=_forecast_key(row)
        if key in seen:raise ValueError("duplicate forecast key")
        seen.add(key)
        if "actual" in row or "targetValue" in row:raise ValueError("current target answer leaked into manifest")
        if row.get("lineage",{}).get("gate",{}).get("historyMaxTargetDate"):
            if _date(row["lineage"]["gate"]["historyMaxTargetDate"])>_date(row["lineage"]["gate"]["historyEnd"]):
                raise ValueError("gate history extends past frozen cutoff")
        origin=_date(row["origin"]);lineage=row.get("lineage",{})
        if row.get("cutoff")!=(origin-dt.timedelta(days=2)).isoformat():raise ValueError("forecast cutoff mismatch")
        if lineage.get("p1",{}).get("date")!=(origin-dt.timedelta(days=1)).isoformat():raise ValueError("p1 must be exact origin minus one")
    expected={(p,(target-dt.timedelta(days=h)).isoformat(),target.isoformat(),h)
        for target in (dt.date(2026,1,1)+dt.timedelta(days=i) for i in range(273)) for p in PRODUCTS for h in HORIZONS}
    if seen!=expected:raise ValueError("forecast grid mismatch")


def _verify_replay(doc:dict,raw_bundles:dict)->None:
    verify_manifest(doc);rebuilt=build_manifest(raw_bundles)
    if doc.get("manifestSha256")!=rebuilt.get("manifestSha256") or doc.get("forecasts")!=rebuilt.get("forecasts"):
        raise ValueError("full lag-guard replay mismatch")


def _dm(diff:dict[dt.date,float],bandwidth:int)->dict:
    """Use the shared HAC test, but classify constant losses before float centering."""
    if diff and all(value==next(iter(diff.values())) for value in diff.values()):
        return {"status":"zero_or_invalid_hac_variance","n":len(diff),"meanLossDifference":float(np.mean(list(diff.values()))),
                "statistic":None,"pValue":1.0,"calendarHacMaxLagDays":bandwidth}
    result=direct_horizon._dm(diff,bandwidth)
    result["calendarHacMaxLagDays"]=bandwidth
    return result


def _period_report(selected:list[dict],actuals:dict[str,dict[dt.date,float]],source:dict,status_evidence:dict|None=None)->dict:
    common=[]
    for row in selected:
        actual=_positive(actuals.get(row["product"],{}).get(_date(row["target"])))
        pred={m:_positive(row["predictions"].get(m)) for m in (METHOD,RAW_METHOD,*BASELINES)}
        if actual is not None and all(v is not None for v in pred.values()):common.append({**row,"actual":actual,"predictions":pred})
    cells={};tests=[];paired=[]
    for product in PRODUCTS:
        for horizon in HORIZONS:
            cell=[r for r in common if r["product"]==product and r["horizon"]==horizon]
            metrics={method:direct_horizon._metrics([r["actual"] for r in cell],[r["predictions"][method] for r in cell]) for method in (METHOD,RAW_METHOD,*BASELINES)}
            cells[f"{product}/{horizon}"]={"n":len(cell),"metrics":metrics}
            for comparator in DM_COMPARATORS:
                for loss in ("mae","mape"):
                    diffs={_date(r["target"]):(abs(r["predictions"][METHOD]-r["actual"])-abs(r["predictions"][comparator]-r["actual"])) if loss=="mae" else 100*(abs(r["predictions"][METHOD]-r["actual"])-abs(r["predictions"][comparator]-r["actual"]))/r["actual"] for r in cell}
                    for bw in BANDWIDTHS:
                        dm=_dm(diffs,bw)
                        tests.append({"product":product,"horizon":horizon,"comparator":comparator,"loss":loss,"dm":dm,"family":"guarded_vs_comparators"})
            for loss in ("mae","mape"):
                diffs={_date(r["target"]):(abs(r["predictions"][RAW_METHOD]-r["actual"])-abs(r["predictions"]["fixed_blend_50_50"]-r["actual"])) if loss=="mae" else 100*(abs(r["predictions"][RAW_METHOD]-r["actual"])-abs(r["predictions"]["fixed_blend_50_50"]-r["actual"]))/r["actual"] for r in cell}
                for bw in BANDWIDTHS:
                    dm=_dm(diffs,bw)
                    tests.append({"product":product,"horizon":horizon,"comparator":"fixed_blend_50_50","loss":loss,"dm":dm,"family":"raw_vs_fixed50"})
            paired.append({"product":product,"horizon":horizon,"comparator":"raw_update_vs_fixed_blend_50_50","n":len(cell),
                "meanPairedMaeDifference":float(np.mean([abs(r["predictions"][RAW_METHOD]-r["actual"])-abs(r["predictions"]["fixed_blend_50_50"]-r["actual"]) for r in cell])) if cell else None,
                "meanPairedMapeDifferencePp":float(np.mean([100*(abs(r["predictions"][RAW_METHOD]-r["actual"])-abs(r["predictions"]["fixed_blend_50_50"]-r["actual"]))/r["actual"] for r in cell])) if cell else None})
            for comparator in DM_COMPARATORS:
                paired.append({"product":product,"horizon":horizon,"comparator":comparator,"n":len(cell),
                    "meanPairedMaeDifference":float(np.mean([abs(r["predictions"][METHOD]-r["actual"])-abs(r["predictions"][comparator]-r["actual"]) for r in cell])) if cell else None,
                    "meanPairedMapeDifferencePp":float(np.mean([100*(abs(r["predictions"][METHOD]-r["actual"])-abs(r["predictions"][comparator]-r["actual"]))/r["actual"] for r in cell])) if cell else None})
    # Holm families are fixed at 400 guarded comparisons plus 80 raw-vs-fixed tests.
    for bw in BANDWIDTHS:
        family=[r for r in tests if r["dm"]["calendarHacMaxLagDays"]==bw]
        for r in family:r["pValue"]=r["dm"]["pValue"]
        direct_horizon._holm(family)
        for r in family:
            adj=r.get("holmPValue",1.);r["dm"]["holmAdjustedPValue"]=adj;r["dm"]["holmSignificant"]=r.get("holmSignificant",False)
            mean=r["dm"].get("meanLossDifference")
            r["dm"]["conclusion"]="inference_unavailable" if r["dm"]["status"]!="tested" else "improvement" if r["dm"]["holmSignificant"] and mean<0 else "worsening" if r["dm"]["holmSignificant"] and mean>0 else "inconclusive"
    issued={m:sum(_positive(r["predictions"].get(m)) is not None for r in selected) for m in (METHOD,RAW_METHOD,*BASELINES)}
    coverage={m:{"planned":len(selected),"issued":issued[m],"issuedOnValidTargets":sum(_date(r["target"]) in actuals.get(r["product"],{}) and _positive(r["predictions"].get(m)) is not None for r in selected)} for m in issued}
    p1_count=sum(r["lineage"].get("p1",{}).get("state")=="positive_price" for r in selected)
    gate_count=sum(r["lineage"].get("gate",{}).get("active") is True for r in selected)
    used_count=sum(r["lineage"].get("guardedUsesRawUpdate") is True for r in selected)
    signal={"planned":len(selected),"p1Available":p1_count,"p1AvailableRate":p1_count/len(selected) if selected else None,
            "gateActive":gate_count,"gateActiveRate":gate_count/len(selected) if selected else None,
            "guardedUsesRawUpdate":used_count,"guardedUsesRawUpdateRate":used_count/len(selected) if selected else None,
            "p1States":dict(sorted(__import__("collections").Counter(r["lineage"].get("p1",{}).get("state","missing_state") for r in selected).items()))}
    complete=all(cells[f"{p}/{h}"]["n"]>=MIN_PAIRS for p in PRODUCTS for h in HORIZONS)
    macro={m:{loss:float(np.mean([cells[f"{p}/{h}"]["metrics"][m][loss] for p in PRODUCTS for h in HORIZONS])) if complete else None for loss in ("mae","mape")} for m in (METHOD,RAW_METHOD,*BASELINES)}
    radish={}
    for month in ("2026-01","2026-02","2026-03","2026-04","2026-05","2026-06","2026-07","2026-08","2026-09"):
        for horizon in (5,10):
            rows=[r for r in common if r["product"]=="radish" and r["target"].startswith(month) and r["horizon"]==horizon]
            if rows:radish[f"{month}/h{horizon}"]={"n":len(rows),"metrics":{m:direct_horizon._metrics([r["actual"] for r in rows],[r["predictions"][m] for r in rows]) for m in (METHOD,RAW_METHOD,"fixed_blend_50_50")}}
    common_keys={_forecast_key(r) for r in common}
    exclusions=inputs._exclusions(selected,source,actuals,status_evidence or {},common_keys)
    return {"planned":len(selected),"validCommonPairs":len(common),"exclusionsByState":exclusions,"coverage":coverage,"signalAndGateRates":signal,
            "complete40Cells":complete,"macro40EqualCell":macro,"cells":cells,"pairedCellLosses":paired,
            "dmComparisons":tests,"holmFamilySizePerBandwidth":len(tests)//len(BANDWIDTHS),"radishMonthly5and10Day":radish}


def _gate(period:dict)->dict:
    complete=period["complete40Cells"];macro=period["macro40EqualCell"];coverage=period["coverage"];cells=period["cells"]
    best=min(((macro[m]["mape"],m) for m in ("ridge_alpha_100","calendar_sarima","fixed_blend_50_50") if macro[m]["mape"] is not None),default=(None,None))
    gain=best[0]-macro[METHOD]["mape"] if complete and best[0] is not None else None
    crop_mape={};crop_mae={}
    for p in PRODUCTS:
        mape_u=[];mape_b=[];mae_u=[];mae_b=[]
        for h in HORIZONS:
            cell=cells[f"{p}/{h}"]["metrics"];mape_u.append(cell[METHOD]["mape"]);mape_b.append(cell["fixed_blend_50_50"]["mape"]);mae_u.append(cell[METHOD]["mae"]);mae_b.append(cell["fixed_blend_50_50"]["mae"])
        crop_mape[p]=float(np.mean(mape_u)-np.mean(mape_b)) if all(v is not None for v in mape_u+mape_b) else None
        crop_mae[p]=100*(float(np.mean(mae_u))-float(np.mean(mae_b)))/float(np.mean(mae_b)) if all(v is not None for v in mae_u+mae_b) and np.mean(mae_b)>0 else None
    indexes={(r["product"],r["horizon"],r["comparator"],r["loss"],r["dm"]["calendarHacMaxLagDays"],r["family"]):r for r in period["dmComparisons"]}
    jointly=sum(all(indexes[(p,h,"fixed_blend_50_50",loss,bw,"guarded_vs_comparators")]["dm"]["conclusion"]=="improvement" for loss in ("mae","mape") for bw in BANDWIDTHS) for p in PRODUCTS for h in HORIZONS)
    criteria={"macroMapeGainVsBestBaselineAtLeast0_5pp":gain is not None and gain>=.5,
        "issuedCoverageNotBelowBestBaseline":best[1] is not None and coverage[METHOD]["issued"]>=coverage[best[1]]["issued"],
        "eachCropMapeWorseningVsFixed50AtMost1pp":all(v is not None and v<=1 for v in crop_mape.values()),
        "eachCropMaeWorseningVsFixed50AtMost5pct":all(v is not None and v<=5 for v in crop_mae.values()),
        "atLeastOneCellJointImprovementVsFixed50":jointly>=1}
    return {"status":"unsupported" if not complete else "pass" if all(criteria.values()) else "fail","criteria":criteria,
        "macroMapeBestBaseline":best[1],"macroMapeGainPp":gain,"cropMapeChangePp":crop_mape,"cropMaeChangePct":crop_mae,
        "jointImprovedCells":jointly,"failureReasons":[] if complete and all(criteria.values()) else ["incomplete_40_cell_coverage"] if not complete else [k for k,v in criteria.items() if not v]}


def score_manifest(doc:dict,raw_bundles:dict)->dict:
    _verify_replay(doc,raw_bundles)
    source,_,_,_,_=_load_all(raw_bundles);actuals=base._prices(source)
    status_evidence=base._coverage_status_evidence(raw_bundles["fresh"])
    report={"protocolId":PROTOCOL_ID,"candidateManifestSha256":doc["manifestSha256"],"periods":{}}
    for start,end in WINDOWS:
        selected=[r for r in doc["forecasts"] if start<=_date(r["target"])<=end]
        key=f"{start.isoformat()}_{end.isoformat()}";report["periods"][key]=_period_report(selected,actuals,source,status_evidence)
    gates=[report["periods"][f"{a.isoformat()}_{b.isoformat()}"] for a,b in WINDOWS[1:]]
    for item in gates:item["predeclaredGate"]=_gate(item)
    report["twoWindowPredeclaredConclusion"]="pass" if all(g["predeclaredGate"]["status"]=="pass" for g in gates) else "fail" if any(g["predeclaredGate"]["status"]=="fail" for g in gates) else "unsupported"
    report["resultType"]="retrospective_counterfactual";report["oMinusOneAvailabilityVerified"]=False
    report["interpretation"]="The exact o-1 price's historical release before the forecast origin is unverified; this is counterfactual evidence, not operationally available signal or official contest performance. Nonsignificance is not equivalence."
    report["reportSha256"]=_hash(report);return report


def _write_json_create_only(path:Path,doc:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True);payload=(json.dumps(doc,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode()
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,"wb") as f:f.write(payload);f.flush();os.fsync(f.fileno())


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description="Frozen one-day information-lag guarded counterfactual")
    sub=parser.add_subparsers(dest="command",required=True)
    common=argparse.ArgumentParser(add_help=False);common.add_argument("--archive",default=str(DEFAULT_ARCHIVE));common.add_argument("--coverage-report",default=str(DEFAULT_COVERAGE))
    build=sub.add_parser("build",parents=[common]);build.add_argument("--output",required=True)
    score=sub.add_parser("score",parents=[common]);score.add_argument("--manifest",required=True);score.add_argument("--output",required=True)
    args=parser.parse_args(argv);bundles={"development":{"archivePath":args.archive},"fresh":{"archivePath":args.archive,"coverageReportPath":args.coverage_report}}
    result=build_manifest(bundles) if args.command=="build" else score_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8")),bundles)
    _write_json_create_only(Path(args.output),result);return 0


if __name__=="__main__":raise SystemExit(main())
