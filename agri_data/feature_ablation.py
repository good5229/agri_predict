"""Leakage-aware P/V/T weather feature POC with monthly expanding Ridge."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import coverage, dashboard, experiments, modeling, origin_linkage

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
INPUT_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
DB_PATH = ROOT / "data/agri.sqlite"
PRODUCTS = tuple(dashboard.PRODUCTS)
STATIONS = ("100", "108", "261")
HORIZONS = (1, 5, 10)
FEATURES = ("P", "V", "T")
COMPARISONS = (("P", "persistence"), ("V", "P"), ("V", "persistence"), ("T", "V"), ("T", "persistence"))
TRAIN_START = dt.date(2014, 1, 1)
EVAL_START, EVAL_END = dt.date(2024, 1, 1), dt.date(2026, 9, 27)
WINDOWS, LAGS = (7, 28, 90), (1, 7, 28, 90)
MIN_TRAIN_DAYS, MIN_PAIRS = 90, 30
PRIMARY_LAG, SENSITIVITY_LAG, ALPHA = 9, 28, .05
LOSSES = ("absolute_error", "absolute_percentage_error")
PROTOCOL = {
    "id": "pvt-weather-feature-poc-2026-10-06",
    "inputPricesSha256": INPUT_SHA256,
    "weatherStations": list(STATIONS),
    "weatherSource": "already-collected ASOS current revisions, selected read-only through origin_linkage validated selector; no API calls",
    "period": {"trainingTargetStart": TRAIN_START.isoformat(), "evaluationTargetStart": EVAL_START.isoformat(),
               "evaluationTargetEnd": EVAL_END.isoformat(), "horizonsCalendarDays": list(HORIZONS)},
    "origin": "forecast issued after 10:00; price and auction weight inputs strictly before origin; seven complete station days origin-7..origin-1",
    "features": {
        "P": "log previous price, 1/7/28/90-day available price lags, 7/28/90-day log-price summaries and coverage, known target calendar features",
        "V": "P plus log1p sums of observed signed net auction kg over preceding 7/28/90 calendar days; absent trade dates are not inserted as rows",
        "T": "V plus each station's mean avgTa over seven complete prior calendar dates; no temperature imputation"},
    "target": "log(target actual / last valid positive price strictly before origin)",
    "fit": {"model": "StandardScaler + Ridge", "alpha": 10, "refit": "first evaluation origin each month; expanding targets from 2014-01-01 strictly before fit origin",
            "minimumDistinctTrainingTargetDays": MIN_TRAIN_DAYS},
    "sample": "P, V, and T share the same weather-complete training and evaluation rows; report price-only-possible rows and weather dropouts separately",
    "weatherSnapshot": "deterministic date/station/avgTa export with selected source request IDs, response hashes, and retrieval/publication metadata is embedded in JSON for offline reproducibility",
    "comparisons": [f"{a} vs {b}" for a,b in COMPARISONS],
    "inference": {"losses": list(LOSSES), "DM": "two-sided asymptotic normal with calendar-distance Bartlett HAC",
        "primaryLagDays": PRIMARY_LAG, "sensitivityLagDays": SENSITIVITY_LAG,
        "minimumPairs": MIN_PAIRS, "holmFamily": "5 comparisons x 5 products x 3 horizons x 2 losses = 150, separately by lag", "alpha": ALPHA},
    "limitations": "retrospective pseudo-as-of revised prices and ASOS; station locations are not proven farm locations; no historical first-publication-time claim; exploratory POC only",
}


def _date(value, label="date"):
    if not isinstance(value,str): raise ValueError(f"{label} must be an ISO date")
    try: return dt.date.fromisoformat(value)
    except ValueError as exc: raise ValueError(f"{label} must be an ISO date") from exc


def _finite(value,label):
    if isinstance(value,bool): raise ValueError(f"{label} must be finite")
    try: result=float(value)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result): raise ValueError(f"{label} must be finite")
    return result


def _parse_prices(source):
    if not isinstance(source,dict) or source.get("schemaVersion")!=1:
        raise ValueError("prices input must be dashboard schemaVersion 1")
    products=source.get("products")
    if not isinstance(products,list) or len(products)!=len(PRODUCTS): raise ValueError("prices input must contain five products")
    series={}; exclusions={}
    for product in products:
        pid=product.get("id")
        if pid not in PRODUCTS or pid in series: raise ValueError("unknown or duplicate price product")
        obs={}; rejected=Counter()
        for row in product.get("observations",[]):
            day=_date(row.get("date"),f"{pid} observation date")
            try: price=_finite(row.get("actual"),"actual price"); volume=_finite(row.get("volumeKg"),"volumeKg")
            except ValueError:
                rejected["invalid_price_or_volume"]+=1;continue
            if price<=0:
                rejected["nonpositive_price"]+=1;continue
            if day in obs: raise ValueError(f"duplicate {pid} daily observation {day}")
            obs[day]={"price":price,"volumeKg":volume,"rowCount":row.get("rowCount")}
        series[pid]=dict(sorted(obs.items()));exclusions[pid]=dict(rejected)
    if set(series)!=set(PRODUCTS): raise ValueError("prices input missing product")
    return series,exclusions


def _weather_window(weather, provenance, origin):
    days=[origin-dt.timedelta(days=i) for i in range(1,8)]
    values=[]; used=[]
    for station in STATIONS:
        station_values=weather.get(station,{})
        station_prov=provenance.get(station,{})
        if any(day not in station_values or station_values[day].get("avgTa") is None for day in days): return None
        try: vals=[_finite(station_values[d]["avgTa"],f"{station}/{d} avgTa") for d in days]
        except ValueError: return None
        values.append(sum(vals)/7.0)
        used.extend(station_prov[d] for d in days if d in station_prov)
    return tuple(values),used


def _missing_weather(weather, origin):
    """Describe every unavailable station/date needed by this origin."""
    missing=[]
    for station in STATIONS:
        for offset in range(1,8):
            day=origin-dt.timedelta(days=offset)
            row=weather.get(station,{}).get(day)
            value=row.get("avgTa") if row is not None else None
            reason="missing_date" if row is None else "missing_avgTa"
            if value is not None:
                try: _finite(value,f"{station}/{day} avgTa")
                except ValueError: reason="invalid_avgTa"
                else: continue
            missing.append({"stationId":station,"date":day.isoformat(),"reason":reason})
    return missing


def _price_context(series):
    dates=list(series)
    logs=[math.log(series[day]["price"]) for day in dates]
    prefix=[0.0]
    for day in dates: prefix.append(prefix[-1]+series[day]["volumeKg"])
    return dates,logs,prefix


def _price_features(series, origin, target, horizon, context=None):
    dates,logs,prefix=context or _price_context(series)
    source_idx=bisect_left(dates,origin)-1
    if source_idx<0: return None
    source_day=dates[source_idx];age=(origin-source_day).days
    if age<=0 or age>dashboard.MAX_SOURCE_AGE_DAYS: return None
    source_price=series[source_day]["price"]
    features=[math.log(source_price),float(horizon),float(age)]
    for lag in LAGS:
        cutoff=origin-dt.timedelta(days=lag)
        idx=bisect_right(dates,cutoff)-1
        if idx<0: features.extend((0.0,float(lag+1),0.0))
        else: features.extend((logs[idx],float((cutoff-dates[idx]).days),1.0))
    for window in WINDOWS:
        begin=origin-dt.timedelta(days=window)
        lo=bisect_left(dates,begin);hi=bisect_left(dates,origin)
        selected=logs[lo:hi];n=len(selected)
        if n:
            mean=sum(selected)/n;var=sum((x-mean)**2 for x in selected)/n
            features.extend((mean,math.sqrt(max(0.0,var)),min(selected),max(selected),float(n),n/window))
        else: features.extend((0.0,0.0,0.0,0.0,0.0,0.0))
    weekday=2*math.pi*target.weekday()/7
    year_days=366 if _is_leap(target.year) else 365
    annual=2*math.pi*(target.timetuple().tm_yday-1)/year_days
    features.extend((math.sin(weekday),math.cos(weekday),math.sin(annual),math.cos(annual)))
    volume_features=[]
    for window in WINDOWS:
        begin=origin-dt.timedelta(days=window)
        lo=bisect_left(dates,begin);hi=bisect_left(dates,origin)
        kg=prefix[hi]-prefix[lo]
        volume_features.append(math.copysign(math.log1p(abs(kg)), kg) if kg else 0.0)
    return {"price":source_price,"sourceDate":source_day,"featuresP":tuple(features),
            "featuresV":tuple(features+volume_features),"volumeFeatures":tuple(volume_features)}


def _is_leap(year):
    return year%4==0 and (year%100!=0 or year%400==0)


def _make_samples(series, weather, weather_provenance, product, horizon):
    samples=[];price_possible_dates=[];weather_missing_dates=[];missing_weather_details=[];price_possible_eval_dates=[]
    prices=series[product]
    context=_price_context(prices)
    for target,obs in prices.items():
        if target<TRAIN_START or target>EVAL_END: continue
        origin=target-dt.timedelta(days=horizon)
        px=_price_features(prices,origin,target,horizon,context)
        if px is None: continue
        price_possible_dates.append(target.isoformat())
        in_eval=EVAL_START<=target<=EVAL_END
        if in_eval: price_possible_eval_dates.append(target.isoformat())
        wx=_weather_window(weather,weather_provenance,origin)
        if wx is None:
            weather_missing_dates.append(target.isoformat())
            missing_weather_details.append({"targetDate":target.isoformat(),"originDate":origin.isoformat(),
                "missingStationDays":_missing_weather(weather,origin)})
            continue
        wfeatures,used=wx
        samples.append({"productId":product,"target":target,"origin":origin,"actual":obs["price"],
            "sourcePrice":px["price"],"features":{"P":px["featuresP"],"V":px["featuresV"],
            "T":px["featuresV"]+wfeatures},"weatherSourceRows":used})
    samples.sort(key=lambda x:(x["origin"],x["target"]))
    eval_weather_complete=sum(EVAL_START<=x["target"]<=EVAL_END for x in samples)
    weather_complete_dates=[x["target"].isoformat() for x in samples]
    weather_missing_eval_dates=[d for d in weather_missing_dates if EVAL_START.isoformat()<=d<=EVAL_END.isoformat()]
    return samples,{"pricePossibleTargets":len(price_possible_dates),"weatherCompleteTargets":len(samples),
        "weatherMissingTargets":len(weather_missing_dates),"pricePossibleEvalTargets":len(price_possible_eval_dates),
        "weatherCompleteEvalTargets":eval_weather_complete,"weatherMissingEvalTargets":len(weather_missing_eval_dates),
        "pricePossibleTargetDates":price_possible_dates,"weatherCompleteTargetDates":weather_complete_dates,
        "weatherMissingTargetDates":weather_missing_dates,"pricePossibleEvalTargetDates":price_possible_eval_dates,
        "weatherMissingEvalTargetDates":weather_missing_eval_dates,"missingWeatherDetails":missing_weather_details}


def _dm_test(rows, candidate, comparator, loss, lag):
    n=len(rows)
    def point(row,model):
        err=abs(row[model]-row["actual"])
        return err if loss=="absolute_error" else err/row["actual"]*100
    differences=[point(row,candidate)-point(row,comparator) for row in rows]
    mean=sum(differences)/n if n else None
    if n<MIN_PAIRS:
        return {"n":n,"status":"unavailable_insufficient_pairs","meanLossDifference":mean,"pValue":None,"calendarHacMaxLagDays":lag}
    scale=max(1.0,max(abs(v) for v in differences))
    if max(differences)-min(differences)<=1e-12*scale:
        return {"n":n,"status":"unavailable_nonpositive_variance","meanLossDifference":mean,"pValue":None,"calendarHacMaxLagDays":lag}
    cand=[{"target":r["target"].isoformat(),"horizon":r["horizon"],"predicted":r[candidate]} for r in rows]
    comp=[{"target":r["target"].isoformat(),"horizon":r["horizon"],"predicted":r[comparator]} for r in rows]
    actuals={(r["target"].isoformat(),r["horizon"]):r["actual"] for r in rows}
    loss_name="absolute_error" if loss=="absolute_error" else "absolute_percentage_error"
    test=experiments._dm(cand,comp,actuals,rows[0]["horizon"],loss_name,max_lag=lag)
    if test.get("status")!="available":
        return {"n":n,"status":test.get("status","unavailable"),"meanLossDifference":mean,"pValue":None,"calendarHacMaxLagDays":lag}
    return {"n":n,"status":"available","meanLossDifference":mean,"statistic":test["statistic"],
            "pValue":test["pValue"],"calendarHacMaxLagDays":lag}


def _holm(tests,key):
    order=sorted(tests,key=lambda x:(x[key]["pValue"] if x[key]["pValue"] is not None else 1.0,
                                     x["productId"],x["horizon"],x["comparison"],x["loss"]))
    running=0.0;m=len(tests)
    for i,row in enumerate(order):
        t=row[key];p=t["pValue"] if t["pValue"] is not None else 1.0
        running=max(running,min(1.0,(m-i)*p));t["holmAdjustedPValue"]=running
        t["holmSignificantAt005"]=t["status"]=="available" and running<ALPHA


def _conclusion(ae,ape):
    if ae["status"]!="available" or ape["status"]!="available": return "inference_unavailable"
    neg=[x["meanLossDifference"]<0 and x["holmSignificantAt005"] for x in (ae,ape)]
    pos=[x["meanLossDifference"]>0 and x["holmSignificantAt005"] for x in (ae,ape)]
    if all(neg):return "improved"
    if all(pos):return "worsened"
    if any(neg) or any(pos):return "mixed_evidence"
    return "no_evidence"


def analyze(source, weather, weather_provenance=None, weather_selection=None, input_sha256=None):
    series,price_exclusions=_parse_prices(source)
    weather_provenance=weather_provenance or {}
    monthly_fits=[];sample_counts={};comparison_records=[];provenance_ids=set()
    for product in PRODUCTS:
        for horizon in HORIZONS:
            samples,counts=_make_samples(series,weather,weather_provenance,product,horizon)
            sample_counts[f"{product}|{horizon}"]=counts
            for sample in samples:
                for src in sample["weatherSourceRows"]:
                    if src.get("requestId") is not None:provenance_ids.add(src["requestId"])
            eval_samples=[x for x in samples if EVAL_START<=x["target"]<=EVAL_END]
            forecasts=[];fit_cache={}
            months=defaultdict(list)
            for sample in eval_samples:months[sample["origin"].replace(day=1)].append(sample)
            for month,month_samples in sorted(months.items()):
                fit_origin=min(x["origin"] for x in month_samples)
                train=[x for x in samples if x["target"]<fit_origin]
                distinct=len({x["target"] for x in train})
                fit_record={"productId":product,"horizon":horizon,"fitMonth":month.isoformat(),
                    "fitOrigin":fit_origin.isoformat(),"fitCutoff":(fit_origin-dt.timedelta(days=1)).isoformat(),
                    "trainingRows":len(train),"distinctTrainingTargetDays":distinct,
                    "status":"fitted" if distinct>=MIN_TRAIN_DAYS else "insufficient_training_dates"}
                if distinct<MIN_TRAIN_DAYS:
                    monthly_fits.append(fit_record);continue
                models={}
                try:
                    from threadpoolctl import threadpool_limits
                except ImportError as exc:raise RuntimeError("threadpoolctl is required for deterministic Ridge fitting") from exc
                with threadpool_limits(limits=1):
                    for feature in FEATURES:
                        estimator=modeling._new_estimator("ridge")
                        estimator.fit([x["features"][feature] for x in train],
                            [math.log(x["actual"]/x["sourcePrice"]) for x in train])
                        models[feature]=estimator
                fit_record["status"]="fitted";monthly_fits.append(fit_record)
                for sample in month_samples:
                    row={"productId":product,"horizon":horizon,"target":sample["target"],"origin":sample["origin"],
                        "actual":sample["actual"],"persistence":sample["sourcePrice"]}
                    for feature,estimator in models.items():
                        predicted_log=float(estimator.predict([sample["features"][feature]])[0])
                        predicted,clipped=modeling._inverse_log_ratio(sample["sourcePrice"],predicted_log)
                        row[feature]=predicted
                        row.setdefault("clippedPredictions",0);row["clippedPredictions"]+=int(clipped)
                    forecasts.append(row)
            forecast_map={(x["productId"],x["horizon"],x["target"]):x for x in forecasts}
            actual_eval=[x for x in eval_samples if (product,horizon,x["target"]) in forecast_map]
            issue_counts={"pricePossibleEvalTargets":counts["pricePossibleEvalTargets"],
                "weatherCompleteEvalTargets":counts["weatherCompleteEvalTargets"],
                "issuedCommonTargets":len(actual_eval),"weatherDropoutTargets":counts["weatherMissingEvalTargets"],
                "pricePossibleEvalTargetDates":counts["pricePossibleEvalTargetDates"],
                "weatherMissingEvalTargetDates":counts["weatherMissingEvalTargetDates"],
                "missingWeatherEvalDetails":[x for x in counts["missingWeatherDetails"]
                    if EVAL_START.isoformat()<=x["targetDate"]<=EVAL_END.isoformat()]}
            # Every contrast is computed on the same P/V/T/persistence target set.
            rows=[forecast_map[(product,horizon,x["target"])] for x in actual_eval]
            for candidate,comparator in COMPARISONS:
                metrics={}
                for model_name in (candidate,comparator):
                    mae=sum(abs(r[model_name]-r["actual"]) for r in rows)/len(rows) if rows else None
                    mape=sum(abs(r[model_name]-r["actual"])/r["actual"]*100 for r in rows)/len(rows) if rows else None
                    metrics[model_name]={"mae":mae,"mape":mape}
                paired_errors=[{"target":r["target"].isoformat(),"origin":r["origin"].isoformat(),
                    "actual":r["actual"],"candidatePrediction":r[candidate],"comparatorPrediction":r[comparator],
                    "absoluteErrorDifference":abs(r[candidate]-r["actual"])-abs(r[comparator]-r["actual"]),
                    "absolutePercentageErrorDifference":(abs(r[candidate]-r["actual"])-abs(r[comparator]-r["actual"])) / r["actual"] * 100}
                    for r in rows]
                tests={}
                for loss in LOSSES:
                    tests[loss]={"primaryHAC9":_dm_test(rows,candidate,comparator,loss,PRIMARY_LAG),
                                 "sensitivityHAC28":_dm_test(rows,candidate,comparator,loss,SENSITIVITY_LAG)}
                comparison_records.append({"productId":product,"horizon":horizon,"comparison":f"{candidate}_vs_{comparator}",
                    "candidate":candidate,"comparator":comparator,"sample":{**issue_counts,"pairedDates":len(rows),
                    "targetStart":min((r["target"].isoformat() for r in rows),default=None),
                    "targetEnd":max((r["target"].isoformat() for r in rows),default=None),
                    "yearCoverage":dict(sorted(Counter(str(r["target"].year) for r in rows).items())),
                    "seasonCoverage":_season_counts(rows)},"metrics":metrics,"pairedErrors":paired_errors,
                    "tests":tests})
    for lag_key in ("primaryHAC9","sensitivityHAC28"):
        flat=[{**t,"test":t["tests"][loss][lag_key],"loss":loss}
              for t in comparison_records for loss in LOSSES]
        _holm(flat,"test")
    for rec in comparison_records:
        rec["conclusion"]=_conclusion(rec["tests"]["absolute_error"]["primaryHAC9"],
            rec["tests"]["absolute_percentage_error"]["primaryHAC9"])
    # Report only compact test records in top-level fixed families; full pair metrics remain grouped.
    family={"primaryHAC9":[],"sensitivityHAC28":[]}
    for rec in comparison_records:
        for loss in LOSSES:
            for key in family:
                family[key].append({"productId":rec["productId"],"horizon":rec["horizon"],
                    "comparison":rec["comparison"],"loss":loss,**rec["tests"][loss][key]})
    weather_reqs=[]
    if weather_selection:
        weather_reqs=[x for x in weather_selection.get("selectedRequests",[]) if x.get("stationId") in STATIONS
            and x.get("requestId") in provenance_ids]
    return {"schemaVersion":1,"status":"complete","exploratoryPostHoc":True,
        "protocol":{**PROTOCOL,"sha256":hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()},
        "inputPricesSha256":input_sha256,"weatherSelection":{**(weather_selection or {}),"selectedRequests":weather_reqs,
            "selectedRequestIds":sorted({x["requestId"] for x in weather_reqs}),"selectedRequestCount":len(weather_reqs)},
        "weatherUsedRequestCount":len(provenance_ids),"weatherInputSnapshot":_weather_snapshot(weather,weather_provenance),
        "priceInputExclusions":price_exclusions,
        "sampleCounts":sample_counts,"monthlyFits":monthly_fits,"comparisons":comparison_records,
        "multipleTesting":{"familySize":150,"alpha":ALPHA,"byBandwidth":family}}


def _season_counts(rows):
    def season(d):
        return "winter" if d.month in (12,1,2) else "spring" if d.month in (3,4,5) else "summer" if d.month in (6,7,8) else "autumn"
    return dict(sorted(Counter(f"{r['target'].year}-{season(r['target'])}" for r in rows).items()))


def _weather_snapshot(weather, provenance):
    """Return a deterministic small input export, not a database or raw response archive."""
    rows=[]
    for station in STATIONS:
        for day,values in sorted(weather.get(station,{}).items()):
            prov=provenance.get(station,{}).get(day,{})
            rows.append({"stationId":station,"date":day.isoformat(),"avgTa":values.get("avgTa"),
                "requestId":prov.get("requestId"),"responseSha256":prov.get("responseSha256"),
                "retrievedAt":prov.get("retrievedAt"),"publishedAt":prov.get("publishedAt"),
                "versionPublishedAt":prov.get("versionPublishedAt"),"rowNo":prov.get("rowNo")})
    digest=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    return {"schemaVersion":1,"scope":"all selected rows for the three frozen stations and requested date window",
            "sha256":digest,"recordCount":len(rows),"records":rows}


def _select_three_stations(db,start,end):
    weather,prov,report=origin_linkage._select_weather(db,start,end)
    subset={s:weather.get(s,{}) for s in STATIONS}
    provenance={s:prov.get(s,{}) for s in STATIONS}
    # Narrow provenance metadata to the three frozen stations.
    report={**report,"queryScopes":[x for x in report.get("queryScopes",[]) if x.get("station") in STATIONS],
        "selectedRequests":[x for x in report.get("selectedRequests",[]) if x.get("stationId") in STATIONS]}
    report["selectedRequestIds"]=sorted({x["requestId"] for x in report["selectedRequests"]})
    report["selectedRequestCount"]=len(report["selectedRequests"])
    report["recordCount"]=sum(len(x) for x in subset.values())
    report["selectedQueryCount"]=sum(len(x.get("selectedRequestIds",[]))>0 for x in report["queryScopes"])
    return subset,provenance,report


def run(input_path=INPUT_PATH,db_path=DB_PATH):
    ip=Path(input_path).resolve();raw=ip.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    if digest!=INPUT_SHA256:raise ValueError(f"fixed prices SHA-256 mismatch: expected {INPUT_SHA256}, found {digest}")
    source=json.loads(raw)
    start=TRAIN_START-dt.timedelta(days=max(HORIZONS)+7);end=EVAL_END
    db=coverage.open_readonly(str(Path(db_path).resolve()))
    try:
        db.execute("BEGIN")
        weather,weather_prov,weather_report=_select_three_stations(db,start,end)
        result=analyze(source,weather,weather_prov,weather_report,digest)
        db.rollback();return result
    finally:db.close()


def _atomic(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def write_report(report,json_path,markdown_path):
    _atomic(json_path,json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    lines=["# 선행 거래량·기온 POC 예측 기여 비교","",
        "2024~2026년 이미 확인된 기간의 후향 pseudo-as-of 비교입니다. 기온 관측소를 농장 위치로 간주하지 않으며, 과거 첫 공개시각을 복원하지 않습니다. 운영모델을 자동 교체하거나 본선 성능을 입증하지 않습니다.","",
        "| 품목 | 거리 | 비교 | 공통일 | MAE 차이 | MAPE 차이 | HAC9 Holm p (AE/APЕ) | 판정 |","|---|---:|---|---:|---:|---:|---|---|"]
    for row in report["comparisons"]:
        t=row["tests"];ae=t["absolute_error"]["primaryHAC9"];ape=t["absolute_percentage_error"]["primaryHAC9"]
        def fmt(v):return f"{v:.3g}" if v is not None else "—"
        pae=fmt(ae.get("holmAdjustedPValue"));pape=fmt(ape.get("holmAdjustedPValue"))
        mae=row["metrics"][row["candidate"]]["mae"];base=row["metrics"][row["comparator"]]["mae"]
        mape=row["metrics"][row["candidate"]]["mape"];basep=row["metrics"][row["comparator"]]["mape"]
        mae_delta=mae-base if mae is not None and base is not None else None
        mape_delta=mape-basep if mape is not None and basep is not None else None
        lines.append(f"| {row['productId']} | {row['horizon']} | {row['comparison']} | {row['sample']['pairedDates']} | {fmt(mae_delta)} | {fmt(mape_delta)} | {pae} / {pape} | {row['conclusion']} |")
    lines += ["","P/V/T는 동일한 세 관측소 7일 완전 기온 표본으로 학습·평가합니다. 빠진 기온은 보간하지 않으며, 기온 때문에 탈락한 가격 가능 표본은 JSON의 sampleCounts에 기록합니다. 가락·강서 경매 순중량만 사용하고 이를 소비자 판매량·출하량으로 해석하지 않습니다.",
        "각 품목×거리에서 2014년 이후 target<fit origin 확장 표본으로 월초 재적합하며 서로 다른 학습 정답일 90일 미만이면 예측하지 않습니다. JSON에는 날짜별 짝지은 오차차와 재현용 세 관측소 일평균기온·원천 요청 해시 스냅샷도 포함합니다. 주 검정은 HAC9, 민감도는 HAC28이며 각 대역폭에서 고정 Holm150을 적용합니다. 두 손실 모두 낮고 두 보정 p<0.05일 때만 개선 근거입니다."]
    _atomic(markdown_path,"\n".join(lines)+"\n")


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--input",default=str(INPUT_PATH));ap.add_argument("--db",default=str(DB_PATH));ap.add_argument("--output",required=True);ap.add_argument("--report")
    args=ap.parse_args(argv);ip=Path(args.input).resolve();db=Path(args.db).resolve();out=Path(args.output).resolve();md=Path(args.report).resolve() if args.report else out.with_suffix(".md")
    if out==md or out in (ip,db) or md in (ip,db):raise ValueError("output paths must be distinct from inputs/database")
    write_report(run(ip,db),out,md);return 0

if __name__=="__main__":raise SystemExit(main())
