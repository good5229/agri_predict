"""Fixed, non-learning seasonal relative-price analog experiment (V4).

The real-data stages are intentionally gated on a frozen protocol digest. This
module only reuses the pinned V3 daily snapshot and saved forecasts; it does not
fit a model, call an API, or alter any prior experiment artifact.
"""
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

from . import period_proxy_experiment as proxy
from . import period_mape10 as v3
from . import period_mape10_score as v3score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/experiments/period-seasonal-relative-analog-v4-20261016"
RUN_ID="PERIOD_SEASONAL_RELATIVE_ANALOG_V4"
PROTOCOL=ROOT/"docs/period_seasonal_analog_protocol.md"
# Filled only after the protocol text is frozen and reviewed.
PROTOCOL_SHA="1f4742b36200a75a338d9e1766004915946df7d6718b2a339fca027ce8f4e42a"
V3_OUT=ROOT/"outputs/experiments/period-direct-mape-rolling-v3-20261015"
OLD_REPORT=ROOT/"outputs/experiments/period-proxy-20261015/report.json"
SOURCE_SHA="6ca1366d1dda618c96f48e62fea578193696b24dc893e68320bdeaae8e43e659"
PROXY_SHA="b6896eac8c43f1f324673e1a1251bd080469c517a80e99d051ceb50c7617fc1c"
V3_CODE_SHA="cabce6ad19104c6c71148b6c8ce1bd7c8b2efe941941deb2051ebb94405fb6f5"
V3_SCORE_SHA="ab2cc43784f4dc10509715c9c545ccc0e43a8706080a6dc6a5e021d19898c051"
V3_PROTOCOL_SHA="778de0205cea77591ff0e5b292fc427373e8b44d7d8ff2e9412755d074abae01"
V3_PROXY_PROTOCOL_SHA="d4a8b94fe6a056e22fefc3f44576018e2fd049397165ea6f856b7b10e2b4a3d4"
OLD_REPORT_SHA="be77991bc4b2b48a1aaf02d286eaedba1633b92d8b8c19c9b337f0184d72d112"
V3_PINNED_FILES={
    "preflight.json":"7e742c0ffaeb0746912b357bce996c83a45df47962d72be688de6d186e15a4ed",
    "resolved_config.json":"6d34b021f32bc75e5a4c9c29659ace03920a3b9d47f3cd9c6c4dd1b54cf66575",
    "selection.json":"fb030e723e3b46674c6555bf6356308246966f12af1f0662d5f541d764af4cae",
    "forecasts.parquet":"9bf20704883f396d13fcd5583eb1c9248a8f720ad51013723e76ad83b42f42a2",
    "forecasts_test.parquet":"697839423df8b9145290f5c2c3866f21c5bab05c2307829124614d98b744b712",
    "score_report.json":"4d8ded8817557239549d3a8f2331a375a1262e68e1c75fd3f1edddcde69be1f1",
}
STARTS={"replay":(dt.date(2018,1,1),dt.date(2021,12,31)),
        "develop":(dt.date(2022,1,1),dt.date(2023,12,31)),
        "test":(dt.date(2024,1,1),dt.date(2026,9,30))}
STAGES=("preflight","replay","develop","forecast","score")
K=8
SEASON_WINDOW=3
FEATURES=("trend_1","trend_2","vol_30","volume_ratio")
FORMAT_PARQUET="parquet"
FORMAT_JSONL="jsonl_fallback_no_parquet_engine"


def digest(path:Path)->str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def code_hash()->str:
    h=hashlib.sha256()
    for p in (Path(__file__).resolve(),Path(__file__).with_name("period_seasonal_analog_score.py")):
        h.update(p.name.encode());h.update(bytes.fromhex(digest(p)))
    return h.hexdigest()

def dependency_identity()->dict[str,str]:
    checks={"protocolSha256":PROTOCOL_SHA,"sourceSha256":proxy.SOURCE_SHA,
       "proxyImplementationSha256":digest(Path(proxy.__file__).resolve()),
       "v3ImplementationSha256":digest(Path(v3.__file__).resolve()),
       "v3ScoreImplementationSha256":digest(Path(v3score.__file__).resolve()),
       "oldProxyReportSha256":digest(OLD_REPORT),"v3ForecastAndSelection":{},
       "v3ProtocolSha256":digest(v3.PROTOCOL),"v3ProxyProtocolSha256":digest(proxy.PROTOCOL)}
    for name,expected in V3_PINNED_FILES.items():
        path=V3_OUT/name
        if not path.is_file() or digest(path)!=expected:
            raise ValueError(f"pinned V3 artifact differs: {path}")
        checks["v3ForecastAndSelection"][name]=expected
    expected={"proxyImplementationSha256":PROXY_SHA,"v3ImplementationSha256":V3_CODE_SHA,
              "v3ScoreImplementationSha256":V3_SCORE_SHA,"sourceSha256":SOURCE_SHA,
              "oldProxyReportSha256":OLD_REPORT_SHA,"v3ProtocolSha256":V3_PROTOCOL_SHA,
              "v3ProxyProtocolSha256":V3_PROXY_PROTOCOL_SHA}
    for k,val in expected.items():
        if checks[k]!=val:raise ValueError(f"pinned V3 dependency differs: {k}")
    if not PROTOCOL.is_file() or digest(PROTOCOL)!=PROTOCOL_SHA:
        raise ValueError("V4 protocol must be frozen and match its pinned digest")
    if not proxy.SOURCE.is_file() or digest(proxy.SOURCE)!=SOURCE_SHA:
        raise ValueError("pinned V3 source snapshot differs")
    return checks

def run_identity()->dict[str,Any]:
    return {"runId":RUN_ID,"dependencies":dependency_identity(),"analysisCodeSha256":code_hash()}

def upstream_hashes(stage:str,out:Path)->dict[str,str]:
    req={"preflight":(),"replay":("preflight",),"develop":("preflight","replay"),
         "forecast":("preflight","replay","develop"),"score":("preflight","forecast")}
    if stage not in req:raise ValueError(stage)
    result={}
    for n in req[stage]:
        p=out/f"{n}.json"
        if not p.is_file():raise FileNotFoundError(f"required upstream checkpoint missing: {p}")
        result[n]=digest(p)
    return result

def _assert_checkpoint(doc:dict,stage:str,out:Path)->None:
    if doc.get("runIdentity")!=run_identity():
        raise ValueError(f"{stage} checkpoint has stale code/protocol/source/dependency identity")
    if doc.get("upstreamCheckpointSha256")!=upstream_hashes(stage,out):
        raise ValueError(f"{stage} checkpoint upstream provenance mismatch")

def save_checkpoint(stage:str,payload:dict,out:Path=OUT)->dict:
    out.mkdir(parents=True,exist_ok=True)
    if stage not in STAGES:raise ValueError(stage)
    path=out/f"{stage}.json"
    if path.exists():
        old=json.loads(path.read_text(encoding="utf-8"));_assert_checkpoint(old,stage,out)
    doc={**payload,"runIdentity":run_identity(),"upstreamCheckpointSha256":upstream_hashes(stage,out)}
    temp=path.with_suffix(".json.tmp");temp.write_text(json.dumps(doc,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(temp,path)
    return doc

def load_checkpoint(stage:str,out:Path=OUT)->dict:
    p=out/f"{stage}.json"
    if not p.is_file():raise FileNotFoundError(f"required checkpoint missing: {p}")
    doc=json.loads(p.read_text(encoding="utf-8"));_assert_checkpoint(doc,stage,out)
    return doc

def write_rows(path:Path,rows:list[dict])->dict:
    path.parent.mkdir(parents=True,exist_ok=True)
    for r in rows:
        if "actual" in r or "targetObserved" in r:raise ValueError("forecast artifact must not contain realized labels")
    try:
        import pandas as pd
        import pyarrow  # noqa
        target=path.with_suffix(".parquet");pd.DataFrame(rows).to_parquet(target,index=False)
        return {"path":str(target),"format":FORMAT_PARQUET,"rows":len(rows),"sha256":digest(target)}
    except ImportError:
        target=path.with_suffix(".jsonl")
        with target.open("w",encoding="utf-8") as f:
            for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
        return {"path":str(target),"format":FORMAT_JSONL,"rows":len(rows),"sha256":digest(target)}

def read_artifact(meta:dict)->list[dict]:
    fmt=meta.get("format");path=Path(meta["path"])
    if fmt not in (FORMAT_PARQUET,FORMAT_JSONL):raise ValueError(f"unsupported artifact format: {fmt}")
    if not path.is_file() or digest(path)!=meta.get("sha256"):raise ValueError("artifact hash/path mismatch")
    if fmt==FORMAT_PARQUET:
        if path.suffix!=".parquet":raise ValueError("artifact format/path mismatch")
        import pandas as pd
        return pd.read_parquet(path).to_dict("records")
    if path.suffix!=".jsonl":raise ValueError("artifact format/path mismatch")
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s]

def artifact_identity(meta:dict)->dict:
    return {"path":meta["path"],"format":meta["format"],"sha256":meta["sha256"],"rows":meta["rows"]}


def period_key(p:tuple[dt.date,dt.date])->str:return f"{p[0].isoformat()}|{p[1].isoformat()}"

def cyclic_season_distance(a:tuple[dt.date,dt.date],b:tuple[dt.date,dt.date])->int:
    ai=proxy._period_index(a)%36;bi=proxy._period_index(b)%36
    d=abs(ai-bi);return min(d,36-d)

def latest_reference(series:dict[dt.date,tuple[float,float]],input_end:dt.date)->tuple[float|None,dt.date|None]:
    valid=[d for d in series if d<=input_end]
    if not valid:return None,None
    day=max(valid);price=float(series[day][0])
    return (price,day) if math.isfinite(price) and price>0 else (None,None)

def _calendar_windows(series,input_end):
    def window(days):
        lo=input_end-dt.timedelta(days=days-1)
        return [(d,float(series[d][0]),float(series[d][1])) for d in sorted(series)
                if lo<=d<=input_end]
    return window

def origin_features(series:dict[dt.date,tuple[float,float]],targets:dict,
                    origin:dt.date)->dict:
    """Calculate only features observable by input_end=origin-2, with gaps intact."""
    input_end=proxy.input_cutoff(origin)
    reference,reference_day=latest_reference(series,input_end)
    recent=proxy.dekad_bounds(input_end)
    if recent[1]>input_end:recent=proxy._next_period(recent,-1)
    previous=proxy._next_period(recent,-1)
    previous2=proxy._next_period(previous,-1)
    p0=targets.get(recent,{}).get("actual")
    p1=targets.get(previous,{}).get("actual")
    p2=targets.get(previous2,{}).get("actual")
    trend1=math.log(reference/p0) if reference and p0 and p0>0 else None
    trend2=math.log(p0/p1) if p0 and p1 and p0>0 and p1>0 else None
    logs=[math.log(float(series[d][0])) for d in sorted(series)
          if input_end-dt.timedelta(days=29)<=d<=input_end and series[d][0]>0]
    vol30=float(np.std(logs,ddof=0)) if len(logs)>=9 else None
    recent_days=[(d,float(series[d][1])) for d in sorted(series)
                 if input_end-dt.timedelta(days=9)<=d<=input_end]
    prior_days=[(d,float(series[d][1])) for d in sorted(series)
                if input_end-dt.timedelta(days=29)<=d<=input_end-dt.timedelta(days=10)]
    vratio=None
    if len(recent_days)>=3 and len(prior_days)>=6:
        recent_mean=float(np.mean([v for _,v in recent_days]));prior_mean=float(np.mean([v for _,v in prior_days]))
        if recent_mean>0 and prior_mean>0:vratio=math.log(recent_mean/prior_mean)
    values={"trend_1":trend1,"trend_2":trend2,"vol_30":vol30,"volume_ratio":vratio}
    return {"origin":origin.isoformat(),"inputEnd":input_end.isoformat(),
       "referencePrice":reference,"referenceDate":reference_day.isoformat() if reference_day else None,
       "recentCompletedPeriod":period_key(recent),"previousCompletedPeriod":period_key(previous),
       "previous2CompletedPeriod":period_key(previous2),"recentPeriodPrice":p0,
       "previousPeriodPrice":p1,"previous2PeriodPrice":p2,
       "recentPeriodObservedDays":int(targets.get(recent,{}).get("observedDays",0)),
       "previousPeriodObservedDays":int(targets.get(previous,{}).get("observedDays",0)),
       "previous2PeriodObservedDays":int(targets.get(previous2,{}).get("observedDays",0)),
       "vol30ObservedDays":len(logs),"recentVolumeObservedDays":len(recent_days),
       "priorVolumeObservedDays":len(prior_days),"features":values}

def weighted_ratio_median(ratios:list[float])->float:
    if not ratios or any(not math.isfinite(r) or r<=0 for r in ratios):raise ValueError("positive finite ratios required")
    ordered=sorted(float(r) for r in ratios)
    weights=[1./r for r in ordered];half=sum(weights)/2
    acc=0.
    for r,w in zip(ordered,weights):
        acc+=w
        if acc>=half:return r
    return ordered[-1]

def _historical_catalog(series,targets,horizon)->list[dict]:
    out=[]
    for target,value in sorted(targets.items()):
        actual=value.get("actual")
        if actual is None or not math.isfinite(float(actual)) or actual<=0:continue
        origin_period=proxy._previous_period_target(target,horizon)
        origin=proxy.origin_for(origin_period)
        if origin<dt.date(2014,1,1):continue
        input_end=proxy.input_cutoff(origin)
        reference,reference_day=latest_reference(series,input_end)
        if not reference:continue
        feats=origin_features(series,targets,origin)
        out.append({"crop":None,"horizon":horizon,"targetStart":target[0].isoformat(),
          "targetEnd":target[1].isoformat(),"targetKey":period_key(target),"origin":origin.isoformat(),
          "inputEnd":input_end.isoformat(),"referencePrice":reference,"referenceDate":reference_day.isoformat(),
          "ratio":float(actual)/reference,"features":feats["features"]})
    return out

def eligible_candidates(catalog:list[dict],crop:str,horizon:int,target:tuple[dt.date,dt.date],origin:dt.date)->list[dict]:
    by_target={}
    for c in catalog:
        c_origin=dt.date.fromisoformat(c["origin"])
        c_target=(dt.date.fromisoformat(c["targetStart"]),dt.date.fromisoformat(c["targetEnd"]))
        if c_origin>=origin or not proxy.label_matured(c_target,origin):continue
        if cyclic_season_distance(c_target,target)>SEASON_WINDOW:continue
        key=c["targetKey"]
        # Multiple legal issue origins for one historical target: nearest to target,
        # with earlier origin breaking an exact distance tie.
        rank=(abs((c_target[0]-c_origin).days),c_origin,key)
        if key not in by_target or rank<by_target[key][0]:by_target[key]=(rank,c)
    return sorted((v[1] for v in by_target.values()),key=lambda c:c["targetKey"])

def _stable_key(c:dict)->tuple:
    return (c["crop"] or "",int(c["horizon"]),c["targetKey"],c["origin"])

def select_analogs(current:dict,candidates:list[dict],crop:str,horizon:int,
                   target:tuple[dt.date,dt.date],k:int=K)->dict:
    # Selection metadata is origin-specific. Work on shallow copies so distances
    # cannot leak from a prior origin through the reusable historical catalog.
    candidates=[{key:value for key,value in c.items() if key not in ("_distance","_seasonDistance")} for c in candidates]
    q0_sorted=sorted(candidates,key=lambda c:(cyclic_season_distance(
        (dt.date.fromisoformat(c["targetStart"]),dt.date.fromisoformat(c["targetEnd"])),target),
        -dt.date.fromisoformat(c["origin"]).toordinal(),_stable_key(c)))
    q0_pool=len(q0_sorted)
    q0_selected=q0_sorted[:k] if q0_pool>=k else []
    valid=[c for c in candidates if all(c["features"].get(f) is not None and
               math.isfinite(float(c["features"][f])) for f in FEATURES)]
    x=current.get("features",{})
    x_valid=all(x.get(f) is not None and math.isfinite(float(x[f])) for f in FEATURES)
    scale={};mask={}
    if valid:
        arr=np.asarray([[float(c["features"][f]) for f in FEATURES] for c in valid],dtype=np.float64)
        med=np.median(arr,axis=0);q25=np.quantile(arr,.25,axis=0,method="linear");q75=np.quantile(arr,.75,axis=0,method="linear")
        iqr=q75-q25;zero=iqr==0
        for i,f in enumerate(FEATURES):scale[f]={"median":float(med[i]),"q25":float(q25[i]),
             "q75":float(q75[i]),"iqr":float(iqr[i]),"zeroIQR":bool(zero[i])}
        mask={f:bool(zero[i]) for i,f in enumerate(FEATURES)}
        if x_valid:
            for c in valid:
                delta=np.asarray([float(c["features"][f])-float(x[f]) for f in FEATURES])
                scaled=np.divide(delta,iqr,out=np.zeros_like(delta),where=~zero)
                c["_distance"] = float(np.sqrt(np.dot(scaled,scaled)))
                c["_seasonDistance"]=cyclic_season_distance((dt.date.fromisoformat(c["targetStart"]),
                    dt.date.fromisoformat(c["targetEnd"])),target)
            q1_sorted=sorted(valid,key=lambda c:(c["_distance"],c["_seasonDistance"],
                -dt.date.fromisoformat(c["origin"]).toordinal(),_stable_key(c)))
        else:q1_sorted=[]
    else:q1_sorted=[]
    q1_pool=len(valid)
    q1_selected=q1_sorted[:k] if q1_pool>=k and x_valid else []
    q0_fallback=None if q0_selected else "insufficient_analogs"
    q0_pred=weighted_ratio_median([c["ratio"] for c in q0_selected]) if q0_selected else None
    q1_fallback=None
    if not x_valid:q1_fallback="current_features_missing"
    elif q1_pool<k:q1_fallback="insufficient_q1_analogs"
    if q1_fallback:
        q1_selected=[]
        q1_pred=q0_pred
        q1_effective="Q0" if q0_pred is not None else "B0"
        if q0_pred is None:q1_fallback="q1_and_q0_insufficient_analogs"
    else:
        q1_pred=weighted_ratio_median([c["ratio"] for c in q1_selected]);q1_effective="Q1"
    q0_effective="Q0" if q0_pred is not None else "B0"
    return {"q0Candidates":q0_pool,"q1ValidCandidates":q1_pool,"currentFeaturesValid":x_valid,
      "scale":scale,"zeroIQRMask":mask,"q0Selected":q0_selected,"q1Selected":q1_selected,
      "q0Ratio":q0_pred,"q1Ratio":q1_pred,"q0FallbackReason":q0_fallback,
      "q1FallbackReason":q1_fallback,"q0Effective":q0_effective,"q1Effective":q1_effective,
      "oracleCandidateRatios":[float(c["ratio"]) for c in valid],"k":k}

def _add_price(prediction,reference):
    return None if prediction is None or reference is None else float(prediction)*float(reference)

def analog_forecasts(current,candidates,crop,horizon,target,origin,b0):
    selected=select_analogs(current,candidates,crop,horizon,target)
    ref=current.get("referencePrice")
    q0_direct=_add_price(selected["q0Ratio"],ref)
    q1_direct=_add_price(selected["q1Ratio"],ref)
    if selected["q0Effective"]=="B0":q0_direct=b0
    if selected["q1Effective"]=="B0":q1_direct=b0
    elif selected["q1Effective"]=="Q0":q1_direct=q0_direct
    q0={"predicted":q0_direct,"effectiveMethod":selected["q0Effective"],
        "fallbackReason":selected["q0FallbackReason"],"referencePrice":ref}
    q1={"predicted":q1_direct,"effectiveMethod":selected["q1Effective"],
        "fallbackReason":selected["q1FallbackReason"],"referencePrice":ref}
    analog_rows=[]
    for method,field in (("Q0","q0Selected"),("Q1","q1Selected")):
        for rank,c in enumerate(selected[field],1):
            analog_rows.append({"crop":crop,"horizon":horizon,"targetStart":target[0].isoformat(),
              "targetEnd":target[1].isoformat(),"origin":origin.isoformat(),"method":method,"rank":rank,
              "candidateTargetKey":c["targetKey"],"candidateOrigin":c["origin"],"candidateYear":int(c["targetStart"][:4]),
              "seasonDistance":cyclic_season_distance((dt.date.fromisoformat(c["targetStart"]),
                    dt.date.fromisoformat(c["targetEnd"])),target),"distance":c.get("_distance"),
              "ratio":c["ratio"],"referencePrice":c["referencePrice"],"targetStartCandidate":c["targetStart"],
              "targetEndCandidate":c["targetEnd"],"candidateFeatures":c["features"]})
    feature_row={"crop":crop,"horizon":horizon,"targetStart":target[0].isoformat(),
        "targetEnd":target[1].isoformat(),**current,"seasonPoolQ0":selected["q0Candidates"],
        "seasonPoolQ1":selected["q1ValidCandidates"],"q0FallbackReason":selected["q0FallbackReason"],
        "q1FallbackReason":selected["q1FallbackReason"],"q1Scale":selected["scale"],
        "zeroIQRMask":selected["zeroIQRMask"],"oraclePoolMinRatio":min(selected["oracleCandidateRatios"])
           if selected["oracleCandidateRatios"] else None,
        "oraclePoolMaxRatio":max(selected["oracleCandidateRatios"])
           if selected["oracleCandidateRatios"] else None,
        "oracleCandidateCount":len(selected["oracleCandidateRatios"]),
        "oracleEligible":selected["currentFeaturesValid"] and selected["q1ValidCandidates"]>0}
    forecast={"crop":crop,"horizon":horizon,"targetStart":target[0].isoformat(),
        "targetEnd":target[1].isoformat(),"targetKey":period_key(target),"origin":origin.isoformat(),
        "inputCutoff":proxy.input_cutoff(origin).isoformat(),"policyUnavailable":False,
        "forecasts":{"Q0":q0,"Q1":q1}}
    return feature_row,analog_rows,forecast,selected


def _v3_rows(name:str)->list[dict]:
    path=V3_OUT/name
    expected=V3_PINNED_FILES[name]
    if not path.is_file() or digest(path)!=expected:raise ValueError(f"V3 artifact integrity failure: {path}")
    import pandas as pd
    return pd.read_parquet(path).to_dict("records")

def _row_key(row):
    return (row["crop"],int(row["horizon"]),period_key((dt.date.fromisoformat(row["targetStart"]),
            dt.date.fromisoformat(row["targetEnd"]))))

def _prediction(row,method):
    f=row.get("forecasts",{}).get(method)
    if not f:return None
    val=f.get("predicted")
    if val is None:return None
    val=float(val)
    if not math.isfinite(val) or val<=0:raise ValueError(f"invalid frozen {method} forecast in {_row_key(row)}")
    return val

def _assert_metrics_match(actual,expected,where):
    if expected.get("n") is not None and actual.get("n")!=expected.get("n"):
        raise ValueError(f"{where} support differs: {actual.get('n')} vs {expected.get('n')}")
    for k in ("mae","mape"):
        if actual.get(k) is None or expected.get(k) is None or not math.isclose(
                float(actual[k]),float(expected[k]),rel_tol=0,abs_tol=1e-9):
            raise ValueError(f"{where} {k} reaggregation differs")

def preflight(out:Path=OUT)->dict:
    identity=run_identity()
    out.mkdir(parents=True,exist_ok=True)
    existing=out/"preflight.json"
    if existing.exists():
        old=json.loads(existing.read_text(encoding="utf-8"));_assert_checkpoint(old,"preflight",out)
        config=out/"resolved_config.json"
        if not config.is_file():raise ValueError("compatible preflight lacks resolved_config.json")
        if json.loads(config.read_text(encoding="utf-8")).get("runIdentity")!=identity:
            raise ValueError("resolved config identity mismatch")
        return old
    _,raw=proxy.validate_inputs();series_by_crop=proxy.series_from_raw(raw)
    baseline=json.loads((V3_OUT/"score_report.json").read_text(encoding="utf-8"))
    test_rows=_v3_rows("forecasts_test.parquet")
    targets={c:proxy._target_periods(s) for c,s in series_by_crop.items()}
    by_key={_row_key(r):r for r in test_rows}
    if len(by_key)!=len(test_rows):raise ValueError("duplicate V3 test forecast keys")
    if len(test_rows)!=len(proxy.CROPS)*len(proxy.HORIZONS)*99:
        raise ValueError("V3 test calendar does not contain all 792 planned keys")
    cells={};all_pairs={"P":[],"B0":[]};policy_unavailable=0
    for crop in proxy.CROPS:
      for horizon in proxy.HORIZONS:
        schedule=proxy.planned_origins(*STARTS["test"],horizon)
        expected={period_key(s["target"]):(s["origin"],s["inputCutoff"]) for s in schedule}
        cellrows=[r for r in test_rows if r["crop"]==crop and int(r["horizon"])==horizon]
        if len(cellrows)!=99 or {_row_key(r)[2] for r in cellrows}!=set(expected):
            raise ValueError(f"V3 test key set differs for {crop}/{horizon}")
        p_pairs=[];b_pairs=[];pkeys=set();bkeys=set();cell_unavailable=0
        for r in cellrows:
            target=(dt.date.fromisoformat(r["targetStart"]),dt.date.fromisoformat(r["targetEnd"]))
            key=period_key(target)
            origin,cutoff=expected[key]
            if dt.date.fromisoformat(r["origin"])!=origin or dt.date.fromisoformat(r["inputCutoff"])!=cutoff:
                raise ValueError(f"V3 test origin/cutoff differs at {crop}/{horizon}/{key}")
            unavailable=bool(r.get("policyUnavailable",False));cell_unavailable+=unavailable
            actual=targets[crop].get(target,{}).get("actual")
            p=_prediction(r,"P");b=_prediction(r,"B0")
            if p is not None:pkeys.add(key)
            if b is not None:bkeys.add(key)
            if p is not None and b is not None:
                if actual is None:raise ValueError(f"scored V3 key has no source actual: {crop}/{horizon}/{key}")
                p_pairs.append((float(actual),p));b_pairs.append((float(actual),b))
                all_pairs["P"].append((float(actual),p));all_pairs["B0"].append((float(actual),b))
        expected_n=98 if horizon==2 else 97
        if pkeys!=bkeys or len(pkeys)!=expected_n or len(p_pairs)!=expected_n:
            raise ValueError(f"V3 P/B0 support differs for {crop}/{horizon}: P={len(pkeys)}, B0={len(bkeys)}")
        if cell_unavailable!=(99-expected_n):raise ValueError("V3 policy-unavailable rows differ from frozen 12-row schedule")
        policy_unavailable+=cell_unavailable
        pm=v3score.metrics(p_pairs);bm=v3score.metrics(b_pairs)
        _assert_metrics_match(pm,baseline["metricsByCell"][crop][str(horizon)]["P"],f"V3 P {crop}/{horizon}")
        _assert_metrics_match(bm,baseline["metricsByCell"][crop][str(horizon)]["B0"],f"V3 B0 {crop}/{horizon}")
        cells[f"{crop}:{horizon}"]={"keys":len(pkeys),"P":pm,"B0":bm,"policyUnavailable":cell_unavailable}
    if policy_unavailable!=12:raise ValueError("V3 unavailable-policy total must remain 12")
    equal={m:{"mae":float(np.mean([cells[f"{c}:{h}"][m]["mae"] for c in proxy.CROPS for h in proxy.HORIZONS])),
        "mape":float(np.mean([cells[f"{c}:{h}"][m]["mape"] for c in proxy.CROPS for h in proxy.HORIZONS])),
        "n":sum(cells[f"{c}:{h}"][m]["n"] for c in proxy.CROPS for h in proxy.HORIZONS)} for m in ("P","B0")}
    for m in ("P","B0"):
        _assert_metrics_match(equal[m],baseline["metrics8EqualCell"][m],f"V3 macro {m}")
    checks={"runId":RUN_ID,"stage":"preflight","complete":True,"dependencyIdentity":identity["dependencies"],
      "v3Reaggregation":{"metrics8EqualCell":equal,"cells":cells,"policyUnavailableRows":policy_unavailable,
        "plannedRows":len(test_rows),"pairedScoredRows":sum(x["keys"] for x in cells.values())},
      "oldReportSha256":OLD_REPORT_SHA,"v3ScoreReportSha256":V3_PINNED_FILES["score_report.json"],
      "v3TestForecastSha256":V3_PINNED_FILES["forecasts_test.parquet"],"status":"verified"}
    doc=save_checkpoint("preflight",checks,out)
    selection=json.loads((V3_OUT/"selection.json").read_text(encoding="utf-8"))
    old=json.loads(OLD_REPORT.read_text(encoding="utf-8"))
    config={"runId":RUN_ID,"runIdentity":identity,"protocolSha256":PROTOCOL_SHA,
      "sourceSha256":SOURCE_SHA,"dependencies":identity["dependencies"],"k":K,
      "seasonWindowDekads":SEASON_WINDOW,"features":list(FEATURES),
      "featureRules":{"inputEnd":"origin-2days","trendPriorCompletedDekadMinimumObservedDays":3,
       "vol30InclusiveDays":30,"vol30MinimumObservedDays":9,"vol30Ddof":0,
       "recentVolumeWindowInclusiveDays":10,"priorVolumeWindowInclusiveDays":20,
       "recentVolumeMinimumObservedDays":3,"priorVolumeMinimumObservedDays":6,
       "volumeSignedFiniteValues":True,"nonpositiveMeanMakesMissing":True,
       "scale":"candidate-only median/IQR; numpy linear quantiles; zero-IQR coordinate contributes zero"},
      "candidateRules":{"firstOrigin":"2014-01-01","targetLabelMaturity":"targetEnd+2days<=currentOrigin",
       "deduplicate":"same target nearest legal origin to target; ties earlier origin, then fixed key",
       "seasonDistance":"36-dekad circular distance <=3","candidateSources":"same crop and horizon only"},
      "fallbacks":{"q1":"current features missing or fewer than eight Q1 candidates => Q0",
       "q0":"fewer than eight Q0 candidates => B0","baselineUnavailable":"preserve missing"},
      "ratioAggregation":"float64 weighted median; weight=1/ratio; ratio ascending; first cumulative weight >= half",
      "priorP":{"selection":selection["P"],"source":"saved V3 selected R/R/R/W development forecasts"},
      "priorB0Development":{"source":"reconstructed from pinned old baseline code and baselineDevelopmentSelection",
        "selection":{f"{c}:{h}":old["crops"][c][str(h)]["baselineDevelopmentSelection"]
          for c in proxy.CROPS for h in proxy.HORIZONS},
        "retrospectiveSelectionAssumption":True},
      "fixedNoLearning":True,"noModelFits":True}
    (out/"resolved_config.json").write_text(json.dumps(config,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return doc

def _frozen_source_context():
    protocol,raw=proxy.validate_inputs()
    series=proxy.series_from_raw(raw)
    targets={c:proxy._target_periods(s) for c,s in series.items()}
    catalog={(c,h):_historical_catalog(series[c],targets[c],h) for c in proxy.CROPS for h in proxy.HORIZONS}
    return {"protocol":protocol,"raw":raw,"series":series,"targets":targets,"catalog":catalog}

def _v3_test_map()->dict:
    return {_row_key(r):r for r in _v3_rows("forecasts_test.parquet")}

def _v3_dev_maps()->tuple[dict,dict,dict]:
    selection=json.loads((V3_OUT/"selection.json").read_text(encoding="utf-8"))
    dev={_row_key(r):r for r in _v3_rows("forecasts.parquet")}
    old=json.loads(OLD_REPORT.read_text(encoding="utf-8"))
    base_selection={(c,h):old["crops"][c][str(h)]["baselineDevelopmentSelection"]
                    for c in proxy.CROPS for h in proxy.HORIZONS}
    return dev,selection["P"],base_selection

def _baseline_replay(series,targets,target,origin,baseline_id):
    cutoff=proxy.input_cutoff(origin)
    if baseline_id=="B0a":
        p,_=latest_reference(series,cutoff);return p
    if baseline_id=="B0b":
        rows=[(tp[1],float(v["actual"])) for tp,v in targets.items()
              if v.get("actual") is not None and proxy.label_matured(tp,origin)]
        return max(rows,key=lambda x:x[0])[1] if rows else None
    if baseline_id=="B0c":
        previous=proxy._next_period(target,-36)
        v=targets.get(previous,{})
        return float(v["actual"]) if v.get("actual") is not None and proxy.label_matured(previous,origin) else None
    raise ValueError(f"unknown frozen B0 selection {baseline_id}")

def build_split(context:dict,split:str,*,v3_test:dict|None=None,v3_dev:dict|None=None,
                p_policy:dict|None=None,b0_policy:dict|None=None)->tuple[list[dict],list[dict],list[dict]]:
    if split not in STARTS:raise ValueError(split)
    start,end=STARTS[split];feature_rows=[];analog_rows=[];forecast_rows=[]
    for crop in proxy.CROPS:
      series=context["series"][crop];targets=context["targets"][crop]
      for horizon in proxy.HORIZONS:
        catalog=context["catalog"][(crop,horizon)]
        v3base=(b0_policy or {}).get((crop,horizon))
        for schedule in proxy.planned_origins(start,end,horizon):
            target=schedule["target"];origin=schedule["origin"];key=period_key(target)
            current=origin_features(series,targets,origin)
            candidates=eligible_candidates(catalog,crop,horizon,target,origin)
            b0_reconstructed=_baseline_replay(series,targets,target,origin,v3base) if v3base else None
            if split=="test":
                base=(v3_test or {}).get((crop,horizon,key))
                if base is None:raise ValueError(f"missing pinned V3 test row: {crop}/{horizon}/{key}")
                pv=_prediction(base,"P");bv=_prediction(base,"B0")
                unavailable=bool(base.get("policyUnavailable",False))
                if dt.date.fromisoformat(base["origin"])!=origin or dt.date.fromisoformat(base["inputCutoff"])!=schedule["inputCutoff"]:
                    raise ValueError("pinned V3 test origin/cutoff changed after preflight")
                p_obj={"predicted":pv,"source":"V3_saved_test_P"} if pv is not None else {"predicted":None,"source":"V3_policy_unavailable"}
                b0_obj={"predicted":bv,"source":"V3_saved_test_B0"} if bv is not None else {"predicted":None,"source":"V3_policy_unavailable"}
                fallback_b0=bv if bv is not None else b0_reconstructed
            elif split=="develop":
                base=(v3_dev or {}).get((crop,horizon,key))
                if base is None:raise ValueError(f"missing pinned V3 development row: {crop}/{horizon}/{key}")
                method=(p_policy or {}).get(crop)
                if isinstance(method,dict):method=method["method"]
                pv=_prediction(base,method) if method else None
                if pv is None:raise ValueError(f"missing saved V3 P dev forecast: {crop}/{horizon}/{key}/{method}")
                p_obj={"predicted":pv,"source":"V3_saved_development_P","method":method}
                b0_obj={"predicted":b0_reconstructed,"source":"V3_baseline_reconstructed",
                         "baselineRule":v3base,"retrospectiveSelectionAssumption":True}
                fallback_b0=b0_reconstructed;unavailable=False
            else:
                p_obj=None;b0_obj=None;fallback_b0=None;unavailable=False
            frow,arows,new,selected=analog_forecasts(current,candidates,crop,horizon,target,origin,fallback_b0)
            frow["split"]=split;feature_rows.append(frow);analog_rows.extend(arows)
            if split!="replay":
                new["policyUnavailable"]=unavailable
                if p_obj is not None:new["forecasts"]["P"]=p_obj
                if b0_obj is not None:new["forecasts"]["B0"]=b0_obj
                new["fallbackBaseline"]={"predicted":fallback_b0,"source":"technical B0 fallback"}
                new["fallbackBaselineReconstructed"]=split=="develop" or (split=="test" and _prediction(base,"B0") is None)
                new["q0CandidateCount"]=selected["q0Candidates"]
                new["q1ValidCandidateCount"]=selected["q1ValidCandidates"]
                new["forecasts"]["Q0"].update({"ratio":selected["q0Ratio"],"candidateCount":selected["q0Candidates"]})
                new["forecasts"]["Q1"].update({"ratio":selected["q1Ratio"],"candidateCount":selected["q1ValidCandidates"]})
                forecast_rows.append(new)
    return feature_rows,analog_rows,forecast_rows

def replay(out:Path=OUT)->dict:
    load_checkpoint("preflight",out)
    ctx=_frozen_source_context();features,analogs,_=build_split(ctx,"replay")
    fm=write_rows(out/"origin_features_replay.parquet",features)
    am=write_rows(out/"analogs_replay.parquet",analogs)
    return save_checkpoint("replay",{"runId":RUN_ID,"stage":"replay","complete":True,
       "originFeatures":fm,"analogs":am,"origins":len(features),"selectedAnalogs":len(analogs)},out)

def develop(out:Path=OUT)->dict:
    load_checkpoint("preflight",out);load_checkpoint("replay",out)
    ctx=_frozen_source_context();dev,p_policy,b0_policy=_v3_dev_maps()
    features,analogs,forecasts=build_split(ctx,"develop",v3_dev=dev,p_policy=p_policy,b0_policy=b0_policy)
    fm=write_rows(out/"origin_features_develop.parquet",features)
    am=write_rows(out/"analogs_develop.parquet",analogs)
    ff=write_rows(out/"forecasts_develop.parquet",forecasts)
    return save_checkpoint("develop",{"runId":RUN_ID,"stage":"develop","complete":True,
       "originFeatures":fm,"analogs":am,"forecastArtifact":ff,"origins":len(features),
       "forecastRows":len(forecasts),"developmentB0RetrospectiveSelectionAssumption":True},out)

def forecast_test(out:Path=OUT)->dict:
    load_checkpoint("preflight",out);load_checkpoint("replay",out);devdoc=load_checkpoint("develop",out)
    ctx=_frozen_source_context();old=_v3_test_map()
    oldreport=json.loads(OLD_REPORT.read_text(encoding="utf-8"))
    b0_policy={(c,h):oldreport["crops"][c][str(h)]["selectedSimpleBaseline"]
               for c in proxy.CROPS for h in proxy.HORIZONS}
    features,analogs,forecasts=build_split(ctx,"test",v3_test=old,b0_policy=b0_policy)
    if len(forecasts)!=len(proxy.CROPS)*len(proxy.HORIZONS)*99:raise ValueError("test output must retain all 792 planned keys")
    if sum(r["policyUnavailable"] for r in forecasts)!=12:raise ValueError("test output must retain exactly 12 early unavailable rows")
    fm=write_rows(out/"origin_features_test.parquet",features)
    am=write_rows(out/"analogs_test.parquet",analogs)
    test_meta=write_rows(out/"forecasts_test.parquet",forecasts)
    # Deliver the plan's canonical complete feature/candidate/prediction tables as well.
    all_features=(read_artifact(_artifact_meta(out,"replay.json",["originFeatures"]))+
        read_artifact(devdoc["originFeatures"])+features)
    all_analogs=(read_artifact(_artifact_meta(out,"replay.json",["analogs"]))+
        read_artifact(devdoc["analogs"])+analogs)
    all_forecasts=read_artifact(devdoc["forecastArtifact"])+forecasts
    canonical={"originFeatures":write_rows(out/"origin-features.parquet",all_features),
       "analogs":write_rows(out/"analogs.parquet",all_analogs),
       "forecasts":write_rows(out/"forecasts.parquet",all_forecasts)}
    (out/"fit_timeline.json").write_text(json.dumps({"fitCount":0,"nonLearningSearches":len(all_features)},indent=2)+"\n")
    return save_checkpoint("forecast",{"runId":RUN_ID,"stage":"forecast","complete":True,
      "originFeaturesTest":fm,"analogsTest":am,"forecastArtifact":test_meta,
      "canonicalArtifacts":canonical,"plannedRows":len(forecasts),
      "policyUnavailableRows":sum(r["policyUnavailable"] for r in forecasts),
      "developmentCheckpointSha256":digest(out/"develop.json"),
      "modelFits":0},out)

def _artifact_meta(out,checkpoint,field):
    cp=json.loads((out/checkpoint).read_text(encoding="utf-8"))
    meta=cp
    for key in field:meta=meta[key]
    if not Path(meta["path"]).is_file() or digest(Path(meta["path"]))!=meta["sha256"]:
        raise ValueError(f"upstream artifact integrity failure: {meta['path']}")
    return meta

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("stage",choices=STAGES+("run",));p.add_argument("--output-dir",type=Path,default=OUT)
    args=p.parse_args(argv)
    actions={"preflight":preflight,"replay":replay,"develop":develop,"forecast":forecast_test}
    out=args.output_dir;stages=STAGES if args.stage=="run" else (args.stage,);results=[]
    for stage in stages:
        if stage=="score":
            from . import period_seasonal_analog_score as score
            result=score.score_saved(out)
        else:
            cp=out/f"{stage}.json"
            if cp.exists():
                result=load_checkpoint(stage,out)
                if not result.get("complete"):raise ValueError(f"incomplete checkpoint: {cp}")
            else:result=actions[stage](out)
        results.append(result)
    final=results[-1];print(json.dumps({"stage":final.get("stage"),"complete":final.get("complete"),
       "outputDir":str(out),"plannedRows":final.get("plannedRows")},ensure_ascii=False))

if __name__=="__main__":main()

