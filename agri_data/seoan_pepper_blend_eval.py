"""Frozen exploratory Seoan pepper price-blend evaluation.

Build an answer-free B prediction manifest first (no B actual is attached).
Only a separate score command may join B actuals and compute the frozen gate.
This retrospective study is exploratory and never establishes publication-time
availability, independent confirmation, or operational benefit.
"""
from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
from typing import Any
import zipfile

from . import five_crop_specialists as five
from . import seoan_pepper_collection as seoan

ROOT=Path(__file__).resolve().parents[1]
HISTORY_DIR=Path("data/raw/seoan_pepper_history_202407_202606")
HISTORY_SUMMARY=HISTORY_DIR/"summary.json"
FORECAST_MANIFEST=Path("outputs/experiments/five-crop-specialists-20261009/forecast-manifest.json")
DIRECT_REPORT=five.PATHS["direct"]
HISTORY_SUMMARY_SHA256="0ba2f0d85839f056f68a168b7bf830403e9170d3d305763bcf63601a46855ca3"
FORECAST_MANIFEST_SHA256="1497ee362945fcab45353a7b2403be752af3bf63c5f584cc0aedb2cc111ceaa8"
FORECAST_CONTENT_SHA256="188828df80788c56ab87e32848dc51a3cca91f92daa642e6ab16e03b365960f0"
OUTPUT_DIR=Path("outputs/experiments/seoan-pepper-blend-eval-20261009")
PREDICTION_MANIFEST=OUTPUT_DIR/"prediction-manifest.json"
SCORED_REPORT=OUTPUT_DIR/"score.json"
FIT_START=dt.date(2024,9,25)
FIT_END=dt.date(2025,8,31)
TEST_START=dt.date(2025,9,25)
TEST_END=dt.date(2026,6,30)
MAX_SOURCE_AGE_DAYS=3
BLEND_WEIGHT=Decimal("0.25")
BANDWIDTHS=(9,28)
ALPHA=0.05
FROZEN_NAME="화건(손꼭무)"
FALLBACK="fixed50"

PROTOCOL={"id":"seoan-pepper-fixed50-external-price-blend-2026-10-09",
 "sourceName":FROZEN_NAME,"dailySourceStatistic":"median of positive finite raw price_kg values; retain raw rows and exclusions",
 "sourceFreshness":"latest source date <= origin-1 calendar day and age <= 3 calendar days; otherwise fixed50 fallback",
 "fit":{"period":[FIT_START.isoformat(),FIT_END.isoformat()],"rows":"unique one-day-horizon direct-report target rows",
        "scale":"median(actual / selected daily source median)"},
 "test":{"period":[TEST_START.isoformat(),TEST_END.isoformat()],"rows":"all pepper horizons 1-10 in answer-free five-crop forecast manifest",
        "formula":"fixed50^0.75 * (scale * latest_source_price)^0.25; if source unavailable use fixed50"},
 "inference":"same-row paired MAE/MAPE; equal-weight 10-horizon macro; 20 two-sided AE/APE DM tests per calendar Bartlett HAC9/HAC28 family with separate Holm .05 correction",
 "gate":"macro MAPE improvement >= 0.5 percentage points, macro MAE lower, source used on >=80% of positive-actual common B rows, no horizon MAPE worsening >1 percentage point, and at least one horizon with both AE and APE Holm improvement for HAC9 and HAC28; otherwise stop",
 "limitation":"retrospective revised source and previously viewed B; exploratory only, not independent confirmation or operational benefit"}


class BlendError(ValueError):pass


def _sha(raw:bytes)->str:return hashlib.sha256(raw).hexdigest()
def _canonical_hash(value:Any)->str:return _sha(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode())


def _positive_decimal(value:Any)->Decimal|None:
    if value is None:return None
    try:n=Decimal(str(value).strip().replace(",",""))
    except (InvalidOperation,ValueError):return None
    return n if n.is_finite() and n>0 else None


def _positive_float(value:Any)->float|None:
    if isinstance(value,bool):return None
    try:n=float(value)
    except (TypeError,ValueError,OverflowError):return None
    return n if math.isfinite(n) and n>0 else None


def _actual_signature(value:Any)->tuple[str,str]:
    if value is None:return ("null","")
    try:n=Decimal(str(value).strip().replace(",",""))
    except (InvalidOperation,ValueError):return ("raw",str(value))
    return ("decimal",str(n.normalize())) if n.is_finite() else ("raw",str(value))


def _date(value:Any)->dt.date:
    if not isinstance(value,str):raise BlendError("date must be ISO text")
    try:return dt.date.fromisoformat(value)
    except ValueError as exc:raise BlendError("date must be ISO text") from exc


def _source_date(value:str)->dt.date|None:
    for fmt in ("%Y-%m-%d","%Y.%m.%d","%Y/%m/%d","%Y%m%d","%Y년 %m월 %d일"):
        try:return dt.datetime.strptime(value.strip(),fmt).date()
        except ValueError:continue
    return None


def _row_key(row:dict)->tuple[str,int,str,str]:
    target=_date(row["target"]);origin=_date(row["origin"]);h=row.get("horizon")
    if isinstance(h,bool) or not isinstance(h,int) or not 1<=h<=10 or target-origin!=dt.timedelta(days=h):
        raise BlendError("forecast origin/target/horizon mismatch")
    if row.get("product", "pepper")!="pepper":raise BlendError("unexpected crop in pepper forecast")
    return ("pepper",h,origin.isoformat(),target.isoformat())


def _load_history(root:Path)->tuple[dict[str,dict],list[dict],dict]:
    summary_path=root/HISTORY_SUMMARY
    summary_bytes=summary_path.read_bytes();summary=json.loads(summary_bytes)
    if _sha(summary_bytes)!=HISTORY_SUMMARY_SHA256:raise BlendError("pinned 24-month history summary SHA-256 mismatch")
    months=summary.get("months",[])
    if summary.get("complete") is not False or summary.get("allMonthsUsable") is not True or summary.get("allMonthsStructurallyReadable") is not True:
        raise BlendError("24-month Seoan history is not structurally usable")
    if len(months)!=24 or [m.get("requestId") for m in months]!=[x["requestId"] for x in _month_plan()]:
        raise BlendError("Seoan history summary does not contain the frozen 24 months")
    source_hashes={"directory":str(HISTORY_DIR),"summaryFile":HISTORY_SUMMARY.name,
                   "summarySha256":_sha(summary_bytes),"months":[]}
    daily:dict[str,list[dict]]={}
    for month in months:
        rid=month["requestId"];raw_name=month.get("rawFile");manifest_name=month.get("manifestFile")
        if month.get("state") not in {"parsed","normal_empty"} or not month.get("usable") or not raw_name or not manifest_name:
            raise BlendError(f"history month {rid} is not usable")
        mp=HISTORY_DIR/manifest_name;rp=HISTORY_DIR/raw_name
        if Path(raw_name).name!=raw_name or Path(manifest_name).name!=manifest_name:raise BlendError(f"history month {rid} has unsafe artifact names")
        mb= (root/mp).read_bytes();rb=(root/rp).read_bytes();meta=json.loads(mb)
        req=meta.get("request",{});expected=next(x for x in _month_plan() if x["requestId"]==rid)
        if (meta.get("kind")!="seoan_history_export" or meta.get("http_status")!=200 or
                meta.get("truncated") is not False or meta.get("transport_error") or req.get("method")!="POST" or
                req.get("url")!=seoan.POST_URL):
            raise BlendError(f"history month {rid} attempt manifest unusable")
        if req.get("params")!=expected["params"] or meta.get("raw_file")!=raw_name or meta.get("response_bytes")!=len(rb) or meta.get("sha256")!=_sha(rb):
            raise BlendError(f"history month {rid} provenance mismatch")
        if month.get("sha256")!=_sha(rb) or month.get("responseBytes")!=len(rb):
            raise BlendError(f"history month {rid} summary/raw mismatch")
        try:
            with zipfile.ZipFile(io.BytesIO(rb)) as archive:
                sheets=[n for n in archive.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
        except zipfile.BadZipFile as exc:raise BlendError(f"history month {rid} raw is not a valid XLSX") from exc
        if len(sheets)!=1:raise BlendError(f"history month {rid} must contain exactly one worksheet")
        parsed=seoan._xlsx_rows(rb)
        if not parsed or (len(parsed)<2 and month.get("state")!="normal_empty"):
            raise BlendError(f"history month {rid} XLSX has no data/header")
        cols=seoan._columns(parsed[0])
        if not {"date","product","price_kg"}.issubset(cols):raise BlendError(f"history month {rid} lacks required columns")
        if month.get("header")!=parsed[0] or month.get("schema")!=sorted(cols):
            raise BlendError(f"history month {rid} summary/schema differs from raw XLSX")
        rows_in_month=0
        for rownum,cells in enumerate(parsed[1:],start=2):
            if not any(c.strip() for c in cells):continue
            product=cells[cols["product"]].strip() if cols["product"]<len(cells) else None
            if product!=FROZEN_NAME:continue
            date_text=cells[cols["date"]].strip() if cols["date"]<len(cells) else ""
            parsed_day=_source_date(date_text)
            day=parsed_day.isoformat() if parsed_day else f"invalid_date::{rid}::{rownum}"
            if parsed_day and not dt.date.fromisoformat(expected["start"])<=parsed_day<=dt.date.fromisoformat(expected["end"]):
                day=f"out_of_month_date::{rid}::{rownum}"
            raw_price=cells[cols["price_kg"]].strip() if cols["price_kg"]<len(cells) else None
            parsed_price=_positive_decimal(raw_price)
            issues=[]
            if raw_price is None or not str(raw_price).strip():issues.append("missing_price_kg")
            elif parsed_price is None:
                try:v=Decimal(str(raw_price).strip().replace(",",""))
                except (InvalidOperation,ValueError):issues.append("invalid_price_kg")
                else:issues.append("nonpositive_or_nonfinite_price_kg" if v.is_finite() else "nonfinite_price_kg")
            rec={"date":day,"dateRaw":date_text,"product":product,"sourceRow":rownum,"priceKgRaw":raw_price,"issues":issues}
            daily.setdefault(day,[]).append(rec);rows_in_month+=1
        if rows_in_month!=month.get("frozenNameCounts",{}).get(FROZEN_NAME,{}).get("rowCount"):
            raise BlendError(f"history month {rid} frozen-name row count mismatch")
        attempt_hashes=[]
        outcomes=month.get("attemptOutcomes",[])
        if not outcomes or manifest_name not in {x.get("manifestFile") for x in outcomes}:
            raise BlendError(f"history month {rid} summary lacks selected-attempt provenance")
        for outcome in outcomes:
            attempt_manifest=outcome.get("manifestFile")
            if not attempt_manifest or Path(attempt_manifest).name!=attempt_manifest:
                raise BlendError(f"history month {rid} attempt list has missing manifest provenance")
            attempt_meta_bytes=(root/HISTORY_DIR/attempt_manifest).read_bytes()
            attempt_meta=json.loads(attempt_meta_bytes);attempt_raw=attempt_meta.get("raw_file")
            if not attempt_raw or Path(attempt_raw).name!=attempt_raw:
                raise BlendError(f"history month {rid} attempt has missing raw provenance")
            if (attempt_meta.get("http_status") is None or attempt_meta.get("truncated") is not False or
                    attempt_meta.get("transport_error") or (isinstance(attempt_meta.get("http_status"),int) and 400<=attempt_meta["http_status"]<500)):
                raise BlendError(f"history month {rid} has a non-usable archived attempt")
            attempt_raw_bytes=(root/HISTORY_DIR/attempt_raw).read_bytes()
            if _sha(attempt_raw_bytes)!=attempt_meta.get("sha256") or len(attempt_raw_bytes)!=attempt_meta.get("response_bytes"):
                raise BlendError(f"history month {rid} attempt raw integrity failure")
            if outcome.get("sha256")!=attempt_meta.get("sha256"):
                raise BlendError(f"history month {rid} attempt summary hash mismatch")
            attempt_hashes.append({"manifestFile":attempt_manifest,"manifestSha256":_sha(attempt_meta_bytes),
                                   "rawFile":attempt_raw,"rawSha256":_sha(attempt_raw_bytes)})
        source_hashes["months"].append({"requestId":rid,"selectedManifestFile":manifest_name,"selectedManifestSha256":_sha(mb),
                                        "selectedRawFile":raw_name,"selectedRawSha256":_sha(rb),"responseBytes":len(rb),
                                        "attempts":attempt_hashes})
    return summary,daily,source_hashes


def _month_plan()->list[dict]:
    result=[];year,month=2024,7
    while (year,month)<=(2026,6):
        start=dt.date(year,month,1)
        end=dt.date(year+1,1,1)-dt.timedelta(days=1) if month==12 else dt.date(year,month+1,1)-dt.timedelta(days=1)
        result.append({"requestId":f"{year:04d}-{month:02d}","start":start.isoformat(),"end":end.isoformat(),
                       "params":{"searchStartDt":start.strftime("%Y%m%d"),"searchEndDt":end.strftime("%Y%m%d"),"searchSpcs":""}})
        month+=1
        if month==13:year+=1;month=1
    return result


def _daily_source(daily:dict[str,list[dict]])->tuple[dict[str,Decimal],dict[str,dict]]:
    medians={};audit={}
    for day,rows in daily.items():
        positives=[_positive_decimal(r["priceKgRaw"]) for r in rows]
        valid=[x for x in positives if x is not None]
        audit[day]={"rowCount":len(rows),"positiveFiniteCount":len(valid),
                    "excludedCount":len(rows)-len(valid),"priceKgMedian":str(Decimal(str(statistics.median(valid)))) if valid else None,
                    "sourceRows":rows}
        if valid:medians[day]=statistics.median(valid)
    return medians,audit


def _latest_source(day_medians:dict[str,Decimal],origin:dt.date)->tuple[dt.date,Decimal]|None:
    cutoff=origin-dt.timedelta(days=1)
    candidates=[]
    for text,value in day_medians.items():
        day=_source_date(text)
        if day is None:continue
        age=(cutoff-day).days
        if day<=cutoff and 0<=age<MAX_SOURCE_AGE_DAYS:candidates.append((day,value))
    return max(candidates,key=lambda x:x[0]) if candidates else None


def _read_direct(root:Path)->tuple[dict,bytes]:
    path=root/DIRECT_REPORT;raw=path.read_bytes()
    if _sha(raw)!=five.INPUT_HASHES["direct"]:raise BlendError("pinned direct report SHA-256 mismatch")
    doc=json.loads(raw)
    # The build phase projects the direct report to A fit fields and prediction
    # keys only; held-out actuals are discarded before any fitting logic runs.
    for row in doc.get("pairedRows",[]):
        if row.get("product")=="pepper":
            try:target=_date(row.get("target"))
            except BlendError:continue
            if TEST_START<=target<=TEST_END:row.pop("actual",None)
    return doc,raw


def _fit_scale(direct:dict,medians:dict[str,Decimal])->dict:
    included=[];excluded={}
    rows=direct.get("pairedRows",[])
    seen=set()
    for row in rows:
        if row.get("product")!="pepper" or row.get("horizon")!=1:continue
        target=_date(row.get("target"))
        if not FIT_START<=target<=FIT_END:continue
        origin=target-dt.timedelta(days=1);key=target.isoformat()
        if key in seen:raise BlendError(f"duplicate A one-day target row: {key}")
        seen.add(key)
        actual=_positive_decimal(row.get("actual"))
        if actual is None:excluded["missing_or_nonpositive_actual"]=excluded.get("missing_or_nonpositive_actual",0)+1;continue
        source=_latest_source(medians,origin)
        if source is None:excluded["source_unavailable_within_3_days"]=excluded.get("source_unavailable_within_3_days",0)+1;continue
        ratio=actual/source[1]
        included.append({"target":key,"origin":origin.isoformat(),"sourceDate":source[0].isoformat(),
                         "sourcePriceKg":str(source[1]),"actualRaw":str(row.get("actual")),"ratio":str(ratio)})
    if not included:raise BlendError("A scale cannot be fit: no eligible unique one-day target pairs")
    scale=statistics.median([Decimal(x["ratio"]) for x in included])
    if not scale.is_finite() or scale<=0:raise BlendError("A fitted source scale is not positive finite")
    return {"scale":str(scale),"includedCount":len(included),"excludedCounts":excluded,"includedRows":included}


def _load_forecasts(root:Path)->tuple[list[dict],bytes]:
    path=root/FORECAST_MANIFEST;raw=path.read_bytes();doc=json.loads(raw)
    if _sha(raw)!=FORECAST_MANIFEST_SHA256:raise BlendError("pinned forecast manifest SHA-256 mismatch")
    if doc.get("phase")!="forecast_only" or doc.get("targetValuesAttached") is not False:
        raise BlendError("forecast manifest is not answer-free")
    body=dict(doc);expected=body.pop("manifestSha256",None)
    if expected!=_canonical_hash(body) or expected!=FORECAST_CONTENT_SHA256:raise BlendError("forecast manifest canonical hash mismatch")
    forecasts=[];seen=set()
    for row in doc.get("forecasts",[]):
        if row.get("product")!="pepper":continue
        key=_row_key(row);target=_date(row["target"])
        if not TEST_START<=target<=TEST_END:continue
        if key in seen:raise BlendError("duplicate B forecast key")
        seen.add(key)
        fixed=_positive_float(row.get("predictions",{}).get("fixed50"))
        if fixed is None:raise BlendError(f"fixed50 unavailable for B key {key}")
        forecasts.append({"key":list(key),"origin":row["origin"],"target":row["target"],"horizon":row["horizon"],
                          "fixed50":fixed,"sourceFallback":None})
    if not forecasts:raise BlendError("no eligible B pepper forecasts")
    return forecasts,raw


def _source_for_forecasts(rows:list[dict],medians:dict[str,Decimal],scale:Decimal)->tuple[list[dict],dict]:
    fallback_counts={"fixed50":0,"externalSource":0}
    output=[]
    for row in rows:
        origin=_date(row["origin"]);selected=_latest_source(medians,origin)
        if selected is None:
            candidate=row["fixed50"];reason="no_positive_source_within_3_days";source_day=None;source_price=None
            fallback_counts["fixed50"]+=1
        else:
            source_day,price=selected
            candidate=(Decimal(str(row["fixed50"]))**(Decimal(1)-BLEND_WEIGHT)*(scale*price)**BLEND_WEIGHT)
            candidate=float(candidate)
            if not math.isfinite(candidate) or candidate<=0:raise BlendError("blend generated a nonpositive/nonfinite prediction")
            reason=None;source_price=str(price);fallback_counts["externalSource"]+=1
        output.append({**row,"prediction":candidate,"sourceDate":source_day.isoformat() if source_day else None,
                       "sourcePriceKg":source_price,"sourceAgeDays":(origin-source_day).days if source_day else None,
                       "fallback":selected is None,"fallbackReason":reason})
    return output,fallback_counts


def build_manifest(root:Path=ROOT)->dict:
    history_summary,daily,history_hashes=_load_history(root)
    medians,source_audit=_daily_source(daily)
    direct,direct_raw=_read_direct(root)
    scale_audit=_fit_scale(direct,medians)
    forecasts,forecast_raw=_load_forecasts(root)
    predictions,fallback_counts=_source_for_forecasts(forecasts,medians,Decimal(scale_audit["scale"]))
    source_hashes={"history":history_hashes,"directReport":{"path":str(DIRECT_REPORT),"sha256":_sha(direct_raw)},
                   "fiveCropForecastManifest":{"path":str(FORECAST_MANIFEST),"sha256":_sha(forecast_raw),
                                               "canonicalManifestSha256":json.loads(forecast_raw).get("manifestSha256")}}
    doc={"schemaVersion":1,"protocol":PROTOCOL,"protocolSha256":_canonical_hash(PROTOCOL),
         "sourceHashes":source_hashes,"trainingA":{"period":[FIT_START.isoformat(),FIT_END.isoformat()],"scaleFit":scale_audit},
         "sourceDailyAudit":source_audit,"fallbackAudit":fallback_counts,"testB":{"period":[TEST_START.isoformat(),TEST_END.isoformat()],
         "plannedForecastRows":len(predictions),"publishedForecastRows":sum(_positive_float(r["fixed50"]) is not None for r in predictions),
         "targetValuesAttached":False,"forecasts":predictions},"complete":False,
         "limitation":"No B actual values are included; published count denotes available answer-free forecast rows, not proof of external publication."}
    doc["manifestSha256"]=_canonical_hash(doc)
    return doc


def write_manifest(path:Path=PREDICTION_MANIFEST,root:Path=ROOT)->dict:
    doc=build_manifest(root);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as f:f.write(json.dumps(doc,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return doc


def verify_manifest(path:Path=PREDICTION_MANIFEST,root:Path=ROOT)->dict:
    saved=json.loads(path.read_text(encoding="utf-8"))
    body=dict(saved);digest=body.pop("manifestSha256",None)
    if digest!=_canonical_hash(body):raise BlendError("saved answer-free manifest canonical hash mismatch")
    rebuilt=build_manifest(root)
    if rebuilt.get("manifestSha256")!=digest:raise BlendError("answer-free prediction manifest is not reproducible from pinned inputs")
    if saved.get("testB",{}).get("targetValuesAttached") is not False:raise BlendError("saved prediction manifest contains B target values")
    return {"verified":True,"manifestSha256":digest,"forecastCount":len(saved.get("testB",{}).get("forecasts",[]))}


def _dm(diffs:dict[dt.date,float],bandwidth:int)->dict:
    n=len(diffs)
    if any(not math.isfinite(x) for x in diffs.values()):
        return {"status":"inference_unavailable_nonfinite_loss_difference","n":n,
                "meanLossDifference":None,"statistic":None,"pValue":1.}
    mean=math.fsum(diffs.values())/n if n else None
    if n<30:return {"status":"inference_unavailable_insufficient_pairs","n":n,"meanLossDifference":mean,"statistic":None,"pValue":1.}
    values=list(diffs.values())
    spread=max(values)-min(values)
    magnitude=max(abs(min(values)),abs(max(values)))
    spread_tolerance=8*math.ulp(magnitude) if magnitude else 8*math.ulp(0.0)
    if spread<=spread_tolerance:
        return {"status":"inference_unavailable_zero_spread","n":n,"meanLossDifference":mean,
                "statistic":None,"pValue":1.}
    centered={d:v-mean for d,v in diffs.items()}
    gamma0=math.fsum(v*v for v in centered.values())
    terms=[gamma0]
    for lag in range(1,bandwidth+1):
        covariance=math.fsum(v*centered.get(day+dt.timedelta(days=lag),0.) for day,v in centered.items())
        terms.append(2*(1-lag/(bandwidth+1))*covariance)
    total=math.fsum(terms);variance=total/(n*n)
    cancellation_tolerance=8*math.ulp(math.fsum(abs(x) for x in terms)) if any(terms) else 0.
    if not math.isfinite(variance) or total<=cancellation_tolerance:
        return {"status":"inference_unavailable_nonpositive_variance","n":n,"meanLossDifference":mean,"statistic":None,"pValue":1.}
    statistic=mean/math.sqrt(variance)
    return {"status":"available","n":n,"meanLossDifference":mean,"statistic":statistic,"pValue":math.erfc(abs(statistic)/math.sqrt(2))}


def _holm(tests:list[dict])->None:
    for bw in BANDWIDTHS:
        family=[t["dm"][str(bw)] for t in tests]
        if len(family)!=20:raise BlendError("frozen 20-test Holm family mismatch")
        ordered=sorted(family,key=lambda x:x["pValue"]);running=0.
        for rank,row in enumerate(ordered):
            running=max(running,min(1.,(len(family)-rank)*row["pValue"]))
            row["holmAdjustedPValue"]=running;row["holmSignificant"]=row["status"]=="available" and running<ALPHA
            row["conclusion"]=("inference_unavailable" if row["status"]!="available" else
                               "improvement" if row["holmSignificant"] and row["meanLossDifference"]<0 else
                               "worsening" if row["holmSignificant"] and row["meanLossDifference"]>0 else "inconclusive")


def _scores(actual:dict[tuple,float],pred_manifest:dict)->dict:
    rows=pred_manifest["testB"]["forecasts"];scored=[];exclusions={};coverage={};
    for h in range(1,11):
        planned=[r for r in rows if r["horizon"]==h]
        valid=[]
        for r in planned:
            key=tuple(r["key"]);y=_positive_float(actual.get(key))
            if y is None:exclusions["missing_or_nonpositive_actual"]=exclusions.get("missing_or_nonpositive_actual",0)+1;continue
            valid.append((r,y));scored.append((r,y))
        coverage[str(h)]={"planned":len(planned),"published":sum(_positive_float(x["fixed50"]) is not None and _positive_float(x["prediction"]) is not None for x in planned),
                          "positiveActual":len(valid),"sourceUsedPlanned":sum(not r["fallback"] for r in planned),
                          "candidateUsedSource":sum(not r["fallback"] for r,_ in valid),
                          "sourceUseRateOnPositiveActual":sum(not r["fallback"] for r,_ in valid)/len(valid) if valid else None}
    tests=[];per_horizon={};
    for h in range(1,11):
        cell=[(r,y) for r,y in scored if r["horizon"]==h]
        def mae(which):return sum(abs(y-float(r[which])) for r,y in cell)/len(cell) if cell else None
        def mape(which):return sum(100*abs(y-float(r[which]))/y for r,y in cell)/len(cell) if cell else None
        per_horizon[str(h)]={"n":len(cell),"candidate":{"mae":mae("prediction"),"mape":mape("prediction")},
                             "fixed50":{"mae":mae("fixed50"),"mape":mape("fixed50")}}
        for loss in ("ae","ape"):
            diffs={}
            for r,y in cell:
                cand=abs(y-r["prediction"]);base=abs(y-r["fixed50"])
                diffs[_date(r["target"])]=cand-base if loss=="ae" else 100*(cand-base)/y
            tests.append({"horizon":h,"loss":loss,"n":len(diffs),"dm":{str(bw):_dm(diffs,bw) for bw in BANDWIDTHS}})
    _holm(tests)
    if all(v["n"]>0 for v in per_horizon.values()):
        macro={m:{loss:sum(per_horizon[str(h)][m][loss] for h in range(1,11))/10 for loss in ("mae","mape")} for m in ("candidate","fixed50")}
    else:macro={m:{loss:None for loss in ("mae","mape")} for m in ("candidate","fixed50")}
    source_rate=sum(v["candidateUsedSource"] for v in coverage.values())/sum(v["positiveActual"] for v in coverage.values()) if sum(v["positiveActual"] for v in coverage.values()) else None
    horizon_map={}
    for h in range(1,11):
        v=per_horizon[str(h)];horizon_map[str(h)]=None if v["candidate"]["mape"] is None else v["candidate"]["mape"]-v["fixed50"]["mape"]
    eligible=[t for t in tests if t["loss"]=="ae"]
    idx={(t["horizon"],t["loss"]):t for t in tests}
    bilateral=[]
    for h in range(1,11):
        pair=[idx[(h,loss)]["dm"][str(bw)]["conclusion"] for loss in ("ae","ape") for bw in BANDWIDTHS]
        if all(x=="improvement" for x in pair):bilateral.append(h)
    reasons=[]
    if macro["candidate"]["mape"] is None:reasons.append("at_least_one_horizon_has_no_positive_actual_pairs")
    if source_rate is None:reasons.append("no_positive_actual_common_pairs")
    gate=(not reasons and macro["candidate"]["mape"]<=macro["fixed50"]["mape"]-0.5 and
          macro["candidate"]["mae"]<macro["fixed50"]["mae"] and source_rate>=0.8 and
          all(v is not None and v<=1.0 for v in horizon_map.values()) and bool(bilateral))
    if not gate and not reasons:reasons.append("one_or_more_frozen_gate_conditions_failed")
    conclusion="inference_unavailable" if reasons and (macro["candidate"]["mape"] is None or source_rate is None) else "candidate_for_prospective_test_only" if gate else "stop_keep_fixed50"
    unavailable={str(bw):{status:sum(t["dm"][str(bw)]["status"]==status for t in tests)
                          for status in sorted({t["dm"][str(bw)]["status"] for t in tests if t["dm"][str(bw)]["status"]!="available"})}
                 for bw in BANDWIDTHS}
    return {"plannedPairs":len(rows),"publishedPairs":sum(_positive_float(r["fixed50"]) is not None and _positive_float(r["prediction"]) is not None for r in rows),
            "positiveActualCommonPairs":len(scored),"exclusionsByState":exclusions,"coverageByHorizon":coverage,
            "perHorizon":per_horizon,"macro10HorizonEqualWeight":macro,"macroMapeImprovementPoints":
                macro["fixed50"]["mape"]-macro["candidate"]["mape"] if macro["candidate"]["mape"] is not None else None,
            "macroMaeImproved":macro["candidate"]["mae"] is not None and macro["candidate"]["mae"]<macro["fixed50"]["mae"],
            "sourceUseRateOnPositiveActualCommonPairs":source_rate,"horizonMapeDifferencePointsCandidateMinusFixed50":horizon_map,
            "dmComparisons":tests,"holmFamilySizePerBandwidth":20,"bilateralImprovementHorizons":bilateral,
            "dmUnavailableCountsByBandwidth":unavailable,
            "publishedCoverageBasis":"answer-free forecast manifest rows with positive finite fixed50 predictions; does not establish external publication",
            "gatePassed":bool(gate),"conclusion":conclusion,"inferenceLimitations":reasons}


def _verify_prediction_rows(doc:dict)->None:
    scale=_positive_decimal(doc.get("trainingA",{}).get("scaleFit",{}).get("scale"))
    if scale is None:raise BlendError("prediction manifest has no positive finite A scale")
    seen=set()
    for row in doc.get("testB",{}).get("forecasts",[]):
        key=_row_key(row)
        if tuple(row.get("key",()))!=key:raise BlendError("prediction key does not match origin/target/horizon")
        if key in seen:raise BlendError("duplicate B prediction key")
        seen.add(key)
        fixed=_positive_decimal(row.get("fixed50"));prediction=_positive_decimal(row.get("prediction"))
        if fixed is None or prediction is None:raise BlendError("nonpositive or nonfinite prediction in answer-free manifest")
        if row.get("fallback") is True:
            if prediction!=fixed or row.get("sourceDate") is not None or row.get("sourcePriceKg") is not None:
                raise BlendError("fixed50 fallback row has inconsistent source/prediction fields")
        elif row.get("fallback") is False:
            day=_date(row.get("sourceDate"));price=_positive_decimal(row.get("sourcePriceKg"));origin=_date(row.get("origin"))
            age=origin-day
            if price is None or not 1<=age.days<=MAX_SOURCE_AGE_DAYS or row.get("sourceAgeDays")!=age.days:
                raise BlendError("external-source row violates frozen source freshness rule")
            expected=fixed**(Decimal(1)-BLEND_WEIGHT)*(scale*price)**BLEND_WEIGHT
            if not math.isclose(float(prediction),float(expected),rel_tol=1e-12,abs_tol=1e-12):
                raise BlendError("external-source prediction does not replay frozen blend formula")
        else:raise BlendError("source fallback state is missing or invalid")


def score_manifest(prediction_path:Path=PREDICTION_MANIFEST,output_path:Path=SCORED_REPORT,root:Path=ROOT)->dict:
    manifest_raw=prediction_path.read_bytes();doc=json.loads(manifest_raw)
    body=dict(doc);saved=body.pop("manifestSha256",None)
    if saved!=_canonical_hash(body):raise BlendError("answer-free prediction manifest hash mismatch")
    if doc.get("testB",{}).get("targetValuesAttached") is not False:raise BlendError("prediction manifest contains or claims attached B targets")
    _verify_prediction_rows(doc)
    direct_raw=(root/DIRECT_REPORT).read_bytes()
    if _sha(direct_raw)!=doc["sourceHashes"]["directReport"]["sha256"]:raise BlendError("B actual source differs from manifest source")
    direct=json.loads(direct_raw);actual={};actual_by_target={};
    for row in direct.get("pairedRows",[]):
        if row.get("product")!="pepper":continue
        target=_date(row.get("target"))
        if not TEST_START<=target<=TEST_END:continue
        h=row.get("horizon");key=("pepper",h,(target-dt.timedelta(days=h)).isoformat(),target.isoformat())
        if key in actual:raise BlendError(f"duplicate B actual direct row: {key}")
        raw_actual=row.get("actual");signature=_actual_signature(raw_actual)
        if target.isoformat() in actual_by_target and actual_by_target[target.isoformat()]!=signature:
            raise BlendError(f"duplicate target-day actual prices disagree: {target.isoformat()}")
        actual_by_target[target.isoformat()]=signature
        actual[key]=raw_actual
    scores=_scores(actual,doc)
    scores["predictionManifestSha256"]=_sha(manifest_raw)
    scores["actualSourceSha256"]=_sha(direct_raw)
    scores["complete"]=False
    scores["scoreSha256"]=_canonical_hash(scores)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open("x",encoding="utf-8") as f:f.write(json.dumps(scores,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return scores


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--build-manifest",action="store_true");mode.add_argument("--verify-manifest",action="store_true");mode.add_argument("--score",action="store_true")
    parser.add_argument("--manifest",type=Path,default=PREDICTION_MANIFEST);parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    if args.build_manifest:
        doc=write_manifest(args.manifest)
        print(json.dumps({"manifest":str(args.manifest),"manifestSha256":doc["manifestSha256"],"plannedB":doc["testB"]["plannedForecastRows"],"targetValuesAttached":False},ensure_ascii=False,sort_keys=True))
    elif args.verify_manifest:
        print(json.dumps(verify_manifest(args.manifest),ensure_ascii=False,sort_keys=True))
    else:
        output=args.output or SCORED_REPORT;result=score_manifest(args.manifest,output)
        print(json.dumps({k:result.get(k) for k in ("complete","plannedPairs","positiveActualCommonPairs","gatePassed","conclusion")},ensure_ascii=False,sort_keys=True))
    return 0


if __name__=="__main__":raise SystemExit(main())
