"""Frozen 364-day historical analog blends for the direct Ridge forecast."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
import tempfile
from typing import Any

from . import dashboard, direct_horizon as dh

ROOT=Path(__file__).resolve().parents[1]
INPUT_PRICES=dh.INPUT_PRICES
INPUT_REPORT=ROOT/"outputs/experiments/direct-horizon-20261006/report.json"
PRICES_SHA256="f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
REPORT_SHA256="a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
HORIZONS=tuple(range(1,11)); OFFSETS=(1,2,3); SOURCE_MAX_AGE=14
SELECTION_START,SELECTION_END=dt.date(2024,9,25),dt.date(2025,9,1)
TEST_START,TEST_END=dt.date(2025,9,25),dt.date(2026,9,24)
FIRST_TEST_ORIGIN=TEST_START-dt.timedelta(days=10); EMBARGO=14
MIXES=(.25,.50); COMPARATORS=("directRidge","persistence"); LOSSES=("mae","mape")
BANDWIDTHS=(9,28); MIN_PAIRS=30; FAMILY_SIZE=len(dashboard.PRODUCTS)*10*2*2
if SELECTION_END>FIRST_TEST_ORIGIN-dt.timedelta(days=EMBARGO): raise AssertionError("selection window leaks test outcomes")


def _pos(value,label):
    if isinstance(value,bool): raise ValueError(f"{label} must be positive finite")
    try: result=float(value)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError(f"{label} must be positive finite") from exc
    if not math.isfinite(result) or result<=0: raise ValueError(f"{label} must be positive finite")
    return result


def _report_rows(report):
    if report.get("schemaVersion",1)!=1 or report.get("protocol",{}).get("id")!="direct-horizon-ridge-2026-10-06":
        raise ValueError("unexpected direct-horizon report schema/protocol")
    indexed={}
    for row in report.get("pairedRows",[]):
        if not isinstance(row,dict) or row.get("_summary"): continue
        product,h=row.get("product"),row.get("horizon")
        if product not in dashboard.PRODUCTS or isinstance(h,bool) or not isinstance(h,int) or h not in HORIZONS: raise ValueError("invalid product/horizon")
        target=dt.date.fromisoformat(row["target"])
        origin=dt.date.fromisoformat(row["origin"]) if row.get("origin") else target-dt.timedelta(days=h)
        if target-origin!=dt.timedelta(days=h): raise ValueError("origin/horizon mismatch")
        key=(product,origin,target,h)
        if key in indexed: raise ValueError(f"duplicate forecast identity {key}")
        indexed[key]={"actual":_pos(row["actual"],"actual"),"directRidge":_pos(row["direct"],"direct Ridge"),
                      "persistence":_pos(row["persistence"],"persistence")}
    if not indexed: raise ValueError("direct report has no paired rows")
    return indexed


def _latest(series, days, cutoff):
    i=bisect_right(days,cutoff)-1
    if i<0 or (cutoff-days[i]).days>SOURCE_MAX_AGE: return None
    return days[i]


def _pair_row(product,origin,target,h,base,series):
    days=tuple(sorted(series)); current_day=_latest(series,days,origin)
    if current_day is None:
        return {**base,"product":product,"origin":origin,"target":target,"horizon":h,
            "currentSourceDate":None,"currentSourcePrice":None,"analogYears":[],"analogYearCount":0,
            "rawMedianLogChange":None,"clampedLogChange":None,"analogPrice":None,
            "fallbackReason":"current_source_missing_or_stale","mix25":base["directRidge"],"mix50":base["directRidge"]}
    source=series[current_day][0]
    if not math.isclose(source,base["persistence"],rel_tol=1e-12,abs_tol=0):
        raise ValueError(f"persistence/source mismatch at {product}/{origin}/h{h}")
    analogs=[]
    for k in OFFSETS:
        past_origin=origin-dt.timedelta(days=364*k); past_target=target-dt.timedelta(days=364*k)
        actual=series.get(past_target)
        source_day=_latest(series,days,past_origin)
        if actual is None or source_day is None: continue
        change=math.log(actual[0]/series[source_day][0])
        if not math.isfinite(change): raise ValueError("nonfinite seasonal log change")
        analogs.append({"k":k,"origin":past_origin,"target":past_target,"sourceDate":source_day,
                        "targetPrice":actual[0],"sourcePrice":series[source_day][0],"logChange":change})
    if analogs:
        raw=sorted(a["logChange"] for a in analogs)[len(analogs)//2] if len(analogs)%2 else sum(sorted(a["logChange"] for a in analogs)[len(analogs)//2-1:len(analogs)//2+1])/2
        clipped=min(.20,max(-.20,raw)); analog_price=source*math.exp(clipped); fallback=None
    else:
        raw=clipped=None; analog_price=None; fallback="no_eligible_historical_analog"
    result={**base,"product":product,"origin":origin,"target":target,"horizon":h,"currentSourceDate":current_day,
        "currentSourcePrice":source,"analogYears":analogs,"analogYearCount":len(analogs),"rawMedianLogChange":raw,
        "clampedLogChange":clipped,"analogPrice":analog_price,"fallbackReason":fallback}
    result["mix25"]=(.75*base["directRidge"]+.25*analog_price) if analog_price is not None else base["directRidge"]
    result["mix50"]=(.5*base["directRidge"]+.5*analog_price) if analog_price is not None else base["directRidge"]
    if not all(math.isfinite(result[m]) and result[m]>0 for m in ("mix25","mix50")): raise ValueError("invalid blended forecast")
    return result


def _metrics(rows,field):
    return dh._metrics([r["actual"] for r in rows],[r[field] for r in rows])


def _evidence_row(row,method):
    return {**row,"selectedMethod":method,"selectedPrediction":row[method],
        "origin":row["origin"].isoformat(),"target":row["target"].isoformat(),
        "currentSourceDate":row["currentSourceDate"].isoformat() if row["currentSourceDate"] else None,
        "analogYears":[{**a,"origin":a["origin"].isoformat(),"target":a["target"].isoformat(),
                         "sourceDate":a["sourceDate"].isoformat()} for a in row["analogYears"]]}


def _season(day): return ("DJF","MAM","JJA","SON")[(day.month%12)//3]


def _coverage(rows,product,horizon,start,end,field):
    subset=[r for r in rows if r["product"]==product and r["horizon"]==horizon and start<=r["target"]<=end]
    candidate={r["target"] for r in subset if r[field] is not None}
    controls={c:{r["target"] for r in subset if r[c] is not None} for c in COMPARATORS}
    return {"eligibleDirectReportRows":len(subset),"candidatePairedCount":len(candidate),
        "candidateCoverageNondeclining":all(not controls[c]-candidate for c in COMPARATORS),
        "coverageLossDates":{c:sorted(d.isoformat() for d in controls[c]-candidate) for c in COMPARATORS},
        "pairedDates":sorted(d.isoformat() for d in candidate)}


def _select(rows):
    chosen={}; audit={}
    for product in dashboard.PRODUCTS:
        product_rows=[r for r in rows if r["product"]==product and SELECTION_START<=r["target"]<=SELECTION_END]
        options=[]
        for order,(name,field) in enumerate((("mix25","mix25"),("mix50","mix50"))):
            per_h={}; valid=[]
            for h in HORIZONS:
                group=[r for r in product_rows if r["horizon"]==h and r[field] is not None]
                valid.extend(group); per_h[str(h)]=_metrics(group,field)
            ridge_by_h={str(h):_metrics([r for r in product_rows if r["horizon"]==h and r[field] is not None],"directRidge") for h in HORIZONS}
            mape=sum(v["mape"] for v in per_h.values())/10 if all(v["mape"] is not None for v in per_h.values()) else None
            mae=sum(v["mae"] for v in per_h.values())/10 if all(v["mae"] is not None for v in per_h.values()) else None
            ridge_mape=sum(v["mape"] for v in ridge_by_h.values())/10 if all(v["mape"] is not None for v in ridge_by_h.values()) else None
            ridge_mae=sum(v["mae"] for v in ridge_by_h.values())/10 if all(v["mae"] is not None for v in ridge_by_h.values()) else None
            coverage={str(h):_coverage(rows,product,h,SELECTION_START,SELECTION_END,field) for h in HORIZONS}
            cov_ok=all(v["candidateCoverageNondeclining"] for v in coverage.values())
            eligible=(mape is not None and ridge_mape is not None and mae is not None and ridge_mae is not None and cov_ok
                and mape<=ridge_mape-.10 and mae<=ridge_mae*1.01)
            options.append({"name":name,"mape":mape,"mae":mae,"ridgeMape":ridge_mape,"ridgeMae":ridge_mae,
                            "coverageNondeclining":cov_ok,"eligible":eligible,"coverage":coverage})
        eligible=[x for x in options if x["eligible"]]
        selected=min(eligible,key=lambda x:(x["mape"],x["mae"],0 if x["name"]=="mix25" else 1))["name"] if eligible else "directRidge"
        chosen[product]=selected; audit[product]={"selected":selected,"candidates":options,
            "fallback":"Ridge" if not eligible else None}
    return chosen,audit


def build(prices,report,*,hashes=None):
    series_map=dh._observations(prices); indexed=_report_rows(report); rows=[]
    for (product,origin,target,h),base in sorted(indexed.items()):
        if not (SELECTION_START<=target<=SELECTION_END or TEST_START<=target<=TEST_END): continue
        actual=series_map[product].get(target)
        if actual is None or not math.isclose(actual[0],base["actual"],rel_tol=1e-12,abs_tol=0):
            raise ValueError(f"pinned actual mismatch at {product}/{target}")
        rows.append(_pair_row(product,origin,target,h,base,series_map[product]))
    selected,selection_audit=_select(rows)
    test=[r for r in rows if TEST_START<=r["target"]<=TEST_END]
    paired=[]; comparisons=[]; coverage={}; strata=[]; macro={}
    for product in dashboard.PRODUCTS:
        method=selected[product]; field=method
        for h in HORIZONS:
            group=[r for r in test if r["product"]==product and r["horizon"]==h]
            comp=[r for r in group if r[field] is not None]
            for r in comp:
                paired.append(_evidence_row(r,method))
            cov=_coverage(test,product,h,TEST_START,TEST_END,field); coverage[f"{product}/{h}"]={**cov,"selectedMethod":method}
            for comparator in COMPARATORS:
                for loss in LOSSES:
                    diff={r["target"]:(abs(r["actual"]-r[field])-abs(r["actual"]-r[comparator])) if loss=="mae" else 100*(abs(r["actual"]-r[field])/r["actual"]-abs(r["actual"]-r[comparator])/r["actual"]) for r in comp}
                    comparisons.append({"product":product,"horizon":h,"comparator":comparator,"loss":loss,"n":len(comp),
                        "candidateMetric":_metrics(comp,field)[loss],"comparatorMetric":_metrics(comp,comparator)[loss],
                        "dm":{str(b):dh._dm(diff,b) for b in BANDWIDTHS}})
            for name,fn in (("year",lambda d:str(d.year)),("season",_season)):
                buckets=defaultdict(list)
                for r in comp: buckets[fn(r["target"])].append(r)
                for label,bucket in sorted(buckets.items()):
                    strata.append({"product":product,"horizon":h,"stratum":name,"value":label,"n":len(bucket),
                        "candidate":_metrics(bucket,field),"directRidge":_metrics(bucket,"directRidge"),"persistence":_metrics(bucket,"persistence")})
        per_method={}
        for name,field_name in ((method,method),("directRidge","directRidge"),("persistence","persistence")):
            values={str(h):_metrics([r for r in test if r["product"]==product and r["horizon"]==h and r[field] is not None],field_name) for h in HORIZONS}
            per_method[name]={"byHorizon":values,"equal10HorizonMape":sum(v["mape"] for v in values.values())/10 if all(v["mape"] is not None for v in values.values()) else None,
                "equal10HorizonMae":sum(v["mae"] for v in values.values())/10 if all(v["mae"] is not None for v in values.values()) else None}
        macro[product]=per_method
    for b in BANDWIDTHS: dh._holm([x["dm"][str(b)] for x in comparisons])
    indexed_comp={(r["product"],r["horizon"],r["comparator"],r["loss"]):r for r in comparisons}; conclusions=[]
    for product in dashboard.PRODUCTS:
        for h in HORIZONS:
            family=[indexed_comp[(product,h,c,l)] for c in COMPARATORS for l in LOSSES]; tests=[x["dm"]["9"] for x in family]
            enough=all(t["status"]=="tested" and t["n"]>=MIN_PAIRS for t in tests)
            lower=all(x["candidateMetric"] is not None and x["comparatorMetric"] is not None and x["candidateMetric"]<x["comparatorMetric"] for x in family)
            significant=all(t.get("holmSignificant") and t.get("meanLossDifference") is not None and t["meanLossDifference"]<0 for t in tests)
            covok=coverage[f"{product}/{h}"]["candidateCoverageNondeclining"]
            adverse=any(t.get("holmSignificant") and t.get("meanLossDifference") is not None and t["meanLossDifference"]>0 for t in tests)
            status="inference_unavailable" if not enough else "improvement" if lower and significant and covok else "worsening_or_mixed" if adverse else "inconclusive"
            conclusions.append({"product":product,"horizon":h,"conclusion":status,"inferenceAvailable":enough,
                "allFourPairedMetricsImprove":lower,"allFourHac9HolmTestsImprove":significant,"coverageNondeclining":covok})
    goals={"below10PercentCount":sum(v[selected[p]]["equal10HorizonMape"] is not None and v[selected[p]]["equal10HorizonMape"]<10 for p,v in macro.items()),
           "below7PercentCount":sum(v[selected[p]]["equal10HorizonMape"] is not None and v[selected[p]]["equal10HorizonMape"]<7 for p,v in macro.items())}
    goals["goalMet"]=goals["below10PercentCount"]>=3 or goals["below7PercentCount"]>=2
    return {"schemaVersion":1,"protocol":{"id":"seasonal-analog-ridge-blend-2026-10-07","inputHashes":{"prices":PRICES_SHA256,"directReport":REPORT_SHA256},
        "selectionTargetInterval":[SELECTION_START.isoformat(),SELECTION_END.isoformat()],"testTargetInterval":[TEST_START.isoformat(),TEST_END.isoformat()],
        "selectionEmbargoDays":EMBARGO,"firstTestOrigin":FIRST_TEST_ORIGIN.isoformat(),"analogOffsetsDays":[364,728,1092],
        "pastOriginSourceMaxAgeDays":SOURCE_MAX_AGE,"currentSourceMaxAgeDays":SOURCE_MAX_AGE,"medianLogChangeClamp":[-.20,.20],
        "mixes":{"mix25":[.75,.25],"mix50":[.5,.5]},"minimumMapeImprovementPercentagePoints":.10,
        "maximumMaeRegressionFraction":.01,"dmFamilySize":FAMILY_SIZE,"comparators":list(COMPARATORS),"losses":list(LOSSES),
        "hacCalendarDays":list(BANDWIDTHS),"holmAlpha":.05,"minimumDmPairs":MIN_PAIRS},"hashes":hashes or {},
        "selection":selected,"selectionAudit":selection_audit,
        "selectionRows":[_evidence_row(r,selected[r["product"]]) for r in rows if SELECTION_START<=r["target"]<=SELECTION_END],
        "pairedRows":paired,"coverage":coverage,"comparisons":comparisons,
        "conclusions":conclusions,"macroMetrics":macro,"yearSeasonMetrics":strata,"goals":goals,
        "limitation":"Retrospective exploration after inspecting recent errors; historical source publication times are unavailable; not independent or official nationwide performance."}


def _read(path):
    raw=Path(path).read_bytes()
    try: value=json.loads(raw)
    except (UnicodeDecodeError,json.JSONDecodeError) as exc: raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value,dict): raise ValueError("input must be a JSON object")
    return value,raw


def _write_immutable(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f: fd=None; f.write(payload); f.flush(); os.fsync(f.fileno())
        os.link(tmp,path)
    except FileExistsError as exc: raise FileExistsError(f"refusing to overwrite immutable output: {path}") from exc
    finally:
        if fd is not None: os.close(fd)
        try: os.unlink(tmp)
        except FileNotFoundError: pass


def run(prices_path=INPUT_PRICES,report_path=INPUT_REPORT,output_path=None):
    if output_path is None: raise ValueError("output_path is required")
    prices,pb=_read(prices_path); report,rb=_read(report_path)
    hashes={"prices":hashlib.sha256(pb).hexdigest(),"directReport":hashlib.sha256(rb).hexdigest()}
    if hashes!={"prices":PRICES_SHA256,"directReport":REPORT_SHA256}: raise ValueError("input SHA-256 differs from frozen protocol")
    result=build(prices,report,hashes=hashes)
    payload=(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    _write_immutable(Path(output_path),payload)
    return {"output":str(output_path),"sha256":hashlib.sha256(payload).hexdigest()}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--prices",type=Path,default=INPUT_PRICES)
    p.add_argument("--report",type=Path,default=INPUT_REPORT); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv); run(a.prices,a.report,a.output); return 0


if __name__=="__main__": raise SystemExit(main())
