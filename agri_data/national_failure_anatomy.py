"""Read-only monthly anatomy of frozen national nonlinear forecast failures."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import direct_horizon, national_median_ensemble as inputs, national_nonlinear_exact_mape as exact_model
from . import national_nonlinear_ratio as ratio_model, national_price_selection as base

PROJECT_ROOT=Path(__file__).resolve().parent.parent
RATIO_PATH=PROJECT_ROOT/"outputs/experiments/national-nonlinear-ratio-20261009/forecast-manifest.json"
RATIO_SHA256="1a459d61637af66f024af2c144326347769eed4a5978771bfb7c9b1710908f08"
EXACT_PATH=PROJECT_ROOT/"outputs/experiments/national-nonlinear-exact-mape-20261009/forecast-manifest.json"
EXACT_SHA256="2176502b1aefeaf87e1550653c4be4b30083a635773bcdfa5ecf4c75678a8685"
PRODUCTS=base.PRODUCTS
HORIZONS=(5,10)
METHODS=("ridge_alpha_100","fixed_blend_50_50",ratio_model.METHOD,exact_model.METHOD)
REQUIRED_METHODS=(exact_model.METHOD,*exact_model.COMPARATORS)
WINDOWS=((dt.date(2026,4,1),dt.date(2026,6,30)),(dt.date(2026,7,1),dt.date(2026,9,30)))


def _sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()


def _hash(value:Any)->str:return _sha(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode())


def _positive(value:Any)->float|None:
    try:x=float(value)
    except (TypeError,ValueError,OverflowError):return None
    return x if math.isfinite(x) and x>0 else None


def _date(value:str)->dt.date:return dt.date.fromisoformat(value)


def _load_pinned(path:Path,expected_sha:str,verifier)->dict:
    raw=path.read_bytes()
    if _sha(raw)!=expected_sha:raise ValueError(f"pinned manifest byte SHA-256 mismatch: {path.name}")
    doc=json.loads(raw);verifier(doc);return doc


def _forecast_maps(ratio_doc:dict,exact_doc:dict)->dict[tuple,dict]:
    rmap={};emap={}
    for row in ratio_doc.get("forecasts",[]):
        key=inputs._forecast_identity(row)
        if key in rmap:raise ValueError("duplicate ratio forecast identity")
        rmap[key]=row
    for row in exact_doc.get("forecasts",[]):
        key=inputs._forecast_identity(row)
        if key in emap:raise ValueError("duplicate exact-MAPE forecast identity")
        emap[key]=row
    expected={(p,(target-dt.timedelta(days=h)).isoformat(),target.isoformat(),h)
              for target in (dt.date(2026,1,1)+dt.timedelta(days=i) for i in range(273)) for p in PRODUCTS for h in range(1,11)}
    if set(rmap)!=expected or set(emap)!=expected:raise ValueError("pinned forecast manifests do not match the full Jan-Sep grid")
    merged={}
    for key in sorted(expected):
        rr,er=rmap[key],emap[key]
        exact_predictions=er.get("predictions",{})
        if not set(REQUIRED_METHODS).issubset(exact_predictions):raise ValueError("exact-MAPE manifest lacks a frozen common comparator")
        ratio_prediction=_positive(rr.get("predictions",{}).get(ratio_model.METHOD))
        if ratio_prediction is None or ratio_prediction!=_positive(exact_predictions.get(ratio_model.METHOD)):
            raise ValueError("pinned manifests disagree on the shared HGB-ratio forecast")
        merged[key]={"product":key[0],"origin":key[1],"target":key[2],"horizon":key[3],"cutoff":rr["cutoff"],
            "predictions":{m:exact_predictions.get(m) for m in REQUIRED_METHODS},
            "ratioLineage":rr.get("lineage",{}).get("candidate",{}),
            "exactLineage":er.get("lineage",{}).get("candidate",{})}
    return merged


def _common_rows(forecasts:dict[tuple,dict],actuals:dict[str,dict[dt.date,float]])->list[dict]:
    common=[]
    for key,row in forecasts.items():
        target=_date(row["target"]);actual=actuals[row["product"]].get(target)
        preds={method:_positive(row["predictions"].get(method)) for method in REQUIRED_METHODS}
        if _positive(actual) is None or any(value is None for value in preds.values()):continue
        ratio_asof=_positive(row["ratioLineage"].get("lastObservedPrice"))
        exact_asof=_positive(row["exactLineage"].get("lastObservedPrice"))
        if ratio_asof is None or exact_asof is None:continue
        if ratio_asof!=exact_asof or row["ratioLineage"].get("lastObservedDate")!=row["exactLineage"].get("lastObservedDate"):
            raise ValueError("ratio and exact candidate disagree on as-of price provenance")
        if row["ratioLineage"].get("featureCutoff")!=row["cutoff"] or row["exactLineage"].get("featureCutoff")!=row["cutoff"]:
            raise ValueError("candidate feature cutoff differs from forecast origin cutoff")
        common.append({**row,"actual":float(actual),"predictions":preds,"asofPrice":ratio_asof,
                       "asofDate":row["ratioLineage"].get("lastObservedDate"),
                       "actualLogChange":math.log(float(actual)/ratio_asof)})
    return common


def _metrics(rows:list[dict],method:str)->dict:
    actual=[r["actual"] for r in rows];pred=[r["predictions"][method] for r in rows]
    result=direct_horizon._metrics(actual,pred)
    result["signedBias"]=(float(np.mean(np.asarray(pred)-np.asarray(actual))) if rows else None)
    return result


def _distribution(values:list[float])->dict:
    if not values:return {k:None for k in ("mean","median","min","p25","p75","max")}
    a=np.asarray(values,dtype=float)
    return {"mean":float(np.mean(a)),"median":float(np.median(a)),"min":float(np.min(a)),
            "p25":float(np.quantile(a,.25)),"p75":float(np.quantile(a,.75)),"max":float(np.max(a))}


def _cell(rows:list[dict])->dict:
    return {"pairedN":len(rows),"metricsByMethod":{method:_metrics(rows,method) for method in METHODS},
            "actualTargetToLastAsOfLogPriceChange":_distribution([r["actualLogChange"] for r in rows])}


def _radish_top(rows:list[dict],limit:int=10)->list[dict]:
    radish=[r for r in rows if r["product"]=="radish"]
    scored=[]
    for r in radish:
        ratio_ape=100*abs(r["predictions"][ratio_model.METHOD]-r["actual"])/r["actual"]
        fixed_ape=100*abs(r["predictions"]["fixed_blend_50_50"]-r["actual"])/r["actual"]
        scored.append((ratio_ape-fixed_ape,r))
    scored.sort(key=lambda pair:(-pair[0],pair[1]["target"],pair[1]["horizon"]))
    return [{"target":r["target"],"origin":r["origin"],"horizon":r["horizon"],"ratioMinusFixed50ApePp":delta,
        "actual":r["actual"],"predictions":{m:r["predictions"][m] for m in METHODS},
        "asofPrice":r["asofPrice"],"asofDate":r["asofDate"],
        "provenanceFlags":{"ratioFallback":r["ratioLineage"].get("fallback"),"ratioFallbackReason":r["ratioLineage"].get("fallbackReason"),
            "ratioFeatureCutoff":r["ratioLineage"].get("featureCutoff"),"ratioFitCutoff":r["ratioLineage"].get("fitCutoff"),
            "exactFallback":r["exactLineage"].get("fallback"),"exactFallbackReason":r["exactLineage"].get("fallbackReason"),
            "exactFeatureCutoff":r["exactLineage"].get("featureCutoff"),"exactFitCutoff":r["exactLineage"].get("fitCutoff")}}
        for delta,r in scored[:limit]]


def _forecast_extrema(forecasts:dict[tuple,dict])->dict:
    output={}
    for product in PRODUCTS:
        for horizon in HORIZONS:
            rows=[r for r in forecasts.values() if r["product"]==product and r["horizon"]==horizon and WINDOWS[0][0].isoformat()<=r["target"]<=WINDOWS[1][1].isoformat()]
            for method in METHODS:
                vals=[(_positive(r["predictions"].get(method)),r) for r in rows]
                vals=[(value,row) for value,row in vals if value is not None]
                vals.sort(key=lambda x:(x[0],x[1]["target"]))
                output[f"{product}/{horizon}/{method}"]={"planned":len(rows),"issued":len(vals),
                    "minimum":{"value":vals[0][0],"target":vals[0][1]["target"]} if vals else None,
                    "maximum":{"value":vals[-1][0],"target":vals[-1][1]["target"]} if vals else None}
    return output


def build_diagnostic(ratio_doc:dict,exact_doc:dict,raw_bundles:dict)->dict:
    ratio_model.verify_manifest(ratio_doc);exact_model.verify_manifest(exact_doc)
    ratio_frozen=_load_pinned(RATIO_PATH,RATIO_SHA256,ratio_model.verify_manifest)
    exact_frozen=_load_pinned(EXACT_PATH,EXACT_SHA256,exact_model.verify_manifest)
    if ratio_doc!=ratio_frozen or exact_doc!=exact_frozen:raise ValueError("forecast manifests differ from pinned files")
    dev_source,fresh_source=inputs._load_sources(raw_bundles)
    expected_source_digests={"development":dev_source["sourceDigests"],"fresh":fresh_source["sourceDigests"]}
    if ratio_doc.get("sourceDigests")!=expected_source_digests:raise ValueError("ratio manifest source digest mismatch")
    if exact_doc.get("sourceDigests")!=ratio_doc.get("sourceDigests"):raise ValueError("forecast manifests disagree on input source hashes")
    actuals=base._prices(fresh_source)
    forecasts=_forecast_maps(ratio_doc,exact_doc);common=_common_rows(forecasts,actuals)
    groups={}
    for start,end in WINDOWS:
        months=[]
        cursor=dt.date(start.year,start.month,1)
        while cursor<=end:
            months.append(cursor.strftime("%Y-%m"))
            cursor=dt.date(cursor.year+int(cursor.month==12),cursor.month%12+1,1)
        for month in months:
            for product in PRODUCTS:
                for horizon in HORIZONS:
                    key=f"{month}/{product}/{horizon}"
                    rows=[r for r in common if r["target"].startswith(month) and r["product"]==product and r["horizon"]==horizon]
                    groups[key]=_cell(rows)
    top={}
    for month in ("2026-04","2026-05","2026-06","2026-07","2026-08","2026-09"):
        for horizon in HORIZONS:
            top[f"{month}/radish/{horizon}"]=_radish_top([r for r in common if r["target"].startswith(month) and r["horizon"]==horizon],10)
    report={"schemaVersion":1,"protocolId":"national-forecast-failure-anatomy-2026-10-09",
        "resultType":"read_only_descriptive_diagnostic","sourceDigests":expected_source_digests,
        "manifestSha256":{"ratio":ratio_doc["manifestSha256"],"exactMape":exact_doc["manifestSha256"]},
        "pairedCommonN":len(common),"validCommonActualRule":"positive reconstructed national target actual and positive finite Ridge100, SARIMA, fixed 50:50, persistence, unweighted ratio, weighted ratio, and exact-MAPE predictions; both HGB last-as-of prices must match",
        "signedBiasDefinition":"mean(forecast - actual); positive values indicate overprediction",
        "windows":[[a.isoformat(),b.isoformat()] for a,b in WINDOWS],"horizonsDays":list(HORIZONS),
        "perCropMonthHorizon":groups,"radishTopRatioMinusFixed50Ape":top,"forecastExtrema":_forecast_extrema(forecasts),
        "interpretation":"Descriptive error anatomy only; results do not attribute causes, establish significance, or tune a model."}
    report["reportSha256"]=_hash(report);return report


def _write_json_create_only(path:Path,doc:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(doc,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode()
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,"wb") as f:f.write(payload);f.flush();os.fsync(f.fileno())


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description="Read-only national forecast failure anatomy")
    parser.add_argument("--archive",default=str(inputs.DEFAULT_ARCHIVE));parser.add_argument("--coverage-report",default=str(inputs.DEFAULT_COVERAGE))
    parser.add_argument("--output",required=True);args=parser.parse_args(argv)
    ratio_doc=_load_pinned(RATIO_PATH,RATIO_SHA256,ratio_model.verify_manifest)
    exact_doc=_load_pinned(EXACT_PATH,EXACT_SHA256,exact_model.verify_manifest)
    bundles={"development":{"archivePath":args.archive},"fresh":{"archivePath":args.archive,"coverageReportPath":args.coverage_report}}
    _write_json_create_only(Path(args.output),build_diagnostic(ratio_doc,exact_doc,bundles));return 0


if __name__=="__main__":raise SystemExit(main())
