"""Staged implementation of PERIOD_DIRECT_MAPE_ROLLING_V3.

This module is deliberately separate from the earlier proxy experiment.  Its
forecast records contain no realized target values; scoring is a later stage.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linprog
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from . import period_proxy_experiment as proxy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/period-direct-mape-rolling-v3-20261015"
RUN_ID = "PERIOD_DIRECT_MAPE_ROLLING_V3"
METHODS = ("R", "W", "S", "M", "MW")
LAMBDA_GRID = (0.001, 0.01, 0.1)
HALF_LIFE_PERIODS = 72.0
FLOOR = 0.05
PROTOCOL = ROOT / "docs/period_mape10_protocol.md"
PROTOCOL_SHA = "778de0205cea77591ff0e5b292fc427373e8b44d7d8ff2e9412755d074abae01"
OLD_REPORT = ROOT / "outputs/experiments/period-proxy-20261015/report.json"
OLD_REPORT_SHA = "be77991bc4b2b48a1aaf02d286eaedba1633b92d8b8c19c9b337f0184d72d112"
PROXY_IMPLEMENTATION_SHA = "b6896eac8c43f1f324673e1a1251bd080469c517a80e99d051ceb50c7617fc1c"
STAGES = ("preflight", "pilot", "tune", "develop", "test", "score")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

def analysis_code_hash() -> str:
    h=hashlib.sha256()
    for p in (Path(__file__).resolve(),Path(__file__).with_name("period_mape10_score.py")):
        h.update(p.name.encode());h.update(bytes.fromhex(digest(p)))
    return h.hexdigest()

def run_identity() -> dict[str,str]:
    """Inputs that define a run; a checkpoint cannot be reused across any change."""
    if not PROTOCOL.is_file() or digest(PROTOCOL)!=PROTOCOL_SHA:
        raise ValueError("frozen V3 protocol digest differs")
    if not OLD_REPORT.is_file() or digest(OLD_REPORT)!=OLD_REPORT_SHA:
        raise ValueError("pinned prior report is absent or differs from authorized digest")
    proxy_path=Path(proxy.__file__).resolve()
    proxy_hash=digest(proxy_path)
    if proxy_hash!=PROXY_IMPLEMENTATION_SHA:
        raise ValueError("pinned proxy implementation digest differs")
    if not proxy.SOURCE.is_file() or digest(proxy.SOURCE)!=proxy.SOURCE_SHA:
        raise ValueError("pinned source snapshot digest differs")
    return {"runId":RUN_ID,"protocolSha256":PROTOCOL_SHA,"sourceSha256":proxy.SOURCE_SHA,
            "oldReportSha256":OLD_REPORT_SHA,"proxyImplementationSha256":proxy_hash,
            "analysisCodeSha256":analysis_code_hash()}

def _upstream_hashes(stage:str,out:Path) -> dict[str,str]:
    upstream={"preflight":(),"pilot":("preflight",),"tune":("preflight","pilot"),
              "develop":("preflight","tune"),"test":("preflight","tune","develop"),
              "score":("preflight","test")}
    if stage not in upstream: raise ValueError(stage)
    hashes={}
    for name in upstream[stage]:
        p=out/f"{name}.json"
        if not p.is_file(): raise FileNotFoundError(f"required upstream checkpoint missing: {p}")
        hashes[name]=digest(p)
    return hashes

def _assert_identity(doc:dict,stage:str,out:Path):
    expected=run_identity()
    if doc.get("runIdentity")!=expected:
        raise ValueError(f"{stage} checkpoint belongs to different frozen inputs/code; restart in a new output directory")
    if doc.get("upstreamCheckpointSha256",{})!=_upstream_hashes(stage,out):
        raise ValueError(f"{stage} checkpoint upstream provenance mismatch")


def weighted_l1_fit(x: Sequence[Sequence[float]], ratio: Sequence[float], *, lam: float,
                    weights: Sequence[float] | None = None) -> dict[str, Any]:
    """Solve weighted ratio-MAPE plus L1(beta); intercept is unpenalized.

    The LP has one nonnegative residual variable per row and coefficient
    positive/negative parts.  The inverse ratio appears exactly once.
    """
    X = np.asarray(x, dtype=float); y = np.asarray(ratio, dtype=float)
    if X.ndim != 2 or len(X) != len(y) or not len(y):
        raise ValueError("X and ratio must be nonempty aligned arrays")
    if not np.isfinite(X).all() or not np.isfinite(y).all() or np.any(y <= 0):
        raise ValueError("LP training data must be finite with positive price ratios")
    if lam < 0: raise ValueError("lambda must be nonnegative")
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != y.shape or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("weights must be finite and strictly positive")
    # Standardize outside the optimizer. Parameters are [intercept, beta+, beta-, slack].
    n, p = X.shape; denom = float(w.sum())
    c = np.r_[0., np.full(p, lam), np.full(p, lam), w / (denom * y)]
    design=np.zeros((n, 1 + 2*p + n)); design[:, 0] = 1.
    design[:, 1:1+p] = X; design[:, 1+p:1+2*p] = -X
    residual=np.zeros_like(design); residual[:, :1+2*p]=design[:, :1+2*p]
    residual[:, 1+2*p:] = -np.eye(n)
    neg_residual=np.zeros_like(design); neg_residual[:, :1+2*p]=-design[:, :1+2*p]
    neg_residual[:, 1+2*p:] = -np.eye(n)
    # f >= floor; slack >= |y-f|, represented by three inequalities.
    floor_constraint=np.zeros_like(design)
    floor_constraint[:, :1+2*p]=-design[:, :1+2*p]
    A_ub=np.vstack([floor_constraint,residual,neg_residual])
    b_ub=np.r_[-np.full(n,FLOOR),y,-y]
    bounds = [(None, None)] + [(0, None)]*(2*p) + [(0, None)]*n
    result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"weighted L1 LP failed: {result.message}")
    v = result.x; beta = v[1:1+p] - v[1+p:1+2*p]
    f = v[0] + X @ beta
    residuals=np.abs(y-f)
    objective=float(np.sum(w*residuals/y)/denom + lam*np.abs(beta).sum())
    residual_slack=float(np.max(residuals-v[1+2*p:]))
    if objective < -1e-12 or abs(objective-float(result.fun)) > 1e-7*max(1.,abs(objective)):
        raise RuntimeError("LP objective replay verification failed")
    if float(f.min()) < FLOOR-1e-8 or residual_slack > 1e-7:
        raise RuntimeError("LP primal feasibility verification failed")
    return {"intercept": float(v[0]), "coef": beta.tolist(), "objective": objective,
            "solverObjective":float(result.fun),"maxConstraintViolation":float(max(0., FLOOR - float(f.min()))),
            "maxResidualSlackViolation":max(0.,residual_slack),
            "trainingPredictions": f.tolist(), "n": n, "p": p,
            "inverseRatioAppliedOnce": True}


def recency_weights(target_periods, fit_origin: dt.date) -> np.ndarray:
    """Mean-one decay based on completed dekads since target end."""
    mature_through=fit_origin-dt.timedelta(days=2)
    completed=proxy._period_index(proxy.dekad_bounds(mature_through))
    ages=[max(0,completed-proxy._period_index(p)) for p in target_periods]
    w=np.power(2.0, -np.asarray(ages, dtype=float)/HALF_LIFE_PERIODS)
    return w / w.mean()


def fit_ratio_model(method: str, x, y_ratio, *, alpha: float, lam: float | None = None,
                    sample_weights=None) -> dict[str, Any]:
    """Fit standardized ratio target and return a portable parameter bundle."""
    if method not in METHODS: raise ValueError(f"unknown method {method}")
    X=np.asarray(x, dtype=float); y=np.asarray(y_ratio, dtype=float)
    if X.ndim != 2 or len(X)!=len(y) or not len(y): raise ValueError("invalid training arrays")
    scaler=StandardScaler().fit(X); Z=scaler.transform(X)
    if method in ("R", "W"):
        model=Ridge(alpha=alpha, fit_intercept=True)
        if method == "W":
            if sample_weights is None: raise ValueError("W requires normalized recency weights")
            w=np.asarray(sample_weights,dtype=float)
            if not np.isclose(w.mean(),1.,rtol=0,atol=1e-10): raise ValueError("W weights must have mean one")
            model.fit(Z,y,sample_weight=w)
        else: model.fit(Z,y)
        inter=float(model.intercept_); coef=np.asarray(model.coef_,dtype=float)
        details={}
    elif method == "S":
        model=Ridge(alpha=alpha, fit_intercept=True).fit(Z,y)
        inter=float(model.intercept_); coef=np.asarray(model.coef_,dtype=float); details={}
    else:
        if lam is None: raise ValueError("M/MW require lambda")
        lp=weighted_l1_fit(Z,y,lam=lam,weights=sample_weights if method=="MW" else None)
        inter=lp["intercept"]; coef=np.asarray(lp["coef"]); details={k:v for k,v in lp.items() if k!="trainingPredictions"}
    scaler_mean=scaler.mean_.tolist(); scaler_scale=scaler.scale_.tolist()
    scaler_id=hashlib.sha256(json.dumps({"mean":scaler_mean,"scale":scaler_scale,"n":len(y)},
        sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"method":method,"alpha":float(alpha),"lambda":lam,"scalerMean":scaler_mean,
            "scalerScale":scaler_scale,"scalerIdentity":scaler_id,"intercept":inter,"coef":coef.tolist(),
            "fitN":len(y),"lp":details}


def predict_ratio(bundle: dict, x) -> tuple[float, bool]:
    x=np.asarray(x,dtype=float); mean_=np.asarray(bundle["scalerMean"]); scale=np.asarray(bundle["scalerScale"])
    z=(x-mean_)/scale
    raw=float(bundle["intercept"] + np.dot(np.asarray(bundle["coef"]),z))
    if bundle["method"] in ("R","W"):
        ratio=math.exp(raw)
        if not math.isfinite(ratio) or ratio<=0: raise ValueError("nonpositive or nonfinite inverse-log-ratio prediction")
        return ratio,False
    return max(FLOOR,raw), raw < FLOOR


def verify_reload(bundle: dict, x: Sequence[float], expected: float, atol: float=1e-10):
    got,_=predict_ratio(bundle,x)
    if not math.isclose(got,expected,rel_tol=0,abs_tol=atol):
        raise ValueError("serialized model reload did not reproduce prediction")


def choose_global_tuning(cell_scores: dict[str, dict[float, dict]], *, key="mape") -> float:
    """Equal-cell candidate selection; all eight cells must score all candidates."""
    cells=tuple(f"{c}:{h}" for c in proxy.CROPS for h in proxy.HORIZONS)
    if set(cell_scores)!=set(cells): raise ValueError("tuning requires exactly all eight cells")
    candidates=set.intersection(*(set(cell_scores[c]) for c in cells))
    if not candidates: raise ValueError("no candidate scored in every cell")
    def rank(param):
        rows=[cell_scores[c][param] for c in cells]
        if any(r.get(key) is None or r.get("mae") is None for r in rows): return (math.inf,math.inf,-param)
        return (float(np.mean([r[key] for r in rows])),float(np.mean([r["mae"] for r in rows])),-param)
    return min(candidates,key=rank)


def select_crop_policies(scores: dict, f_mae: dict) -> dict:
    """Select one method per crop from frozen 2022/23 development score tables."""
    order=("M","MW","R","W","S"); out={}
    for crop in proxy.CROPS:
        candidates=[]
        for method in order:
            rows=[scores[crop][str(y)][str(h)][method] for y in (2022,2023) for h in proxy.HORIZONS]
            if any(r.get("mape") is None or r.get("mae") is None for r in rows): continue
            eligible=all(np.mean([scores[crop][str(y)][str(h)][method]["mae"]
                                  for h in proxy.HORIZONS]) <= 1.05*f_mae[crop][str(y)] + 1e-12
                         for y in (2022,2023))
            candidates.append((float(np.mean([r["mape"] for r in rows])),order.index(method),method,eligible,
                               float(np.mean([r["mae"] for r in rows]))))
        if not candidates: raise ValueError(f"no complete candidate scores for {crop}")
        allowed=[x for x in candidates if x[3]]; pool=allowed or candidates
        best=min(x[0] for x in pool)
        tied=[x for x in pool if x[0]<=best+1e-10]
        chosen=min(tied,key=lambda x:x[1])
        out[crop]={"method":chosen[2],"riskCandidate":not bool(allowed),"developmentMAPE":chosen[0],
                   "developmentMAE":chosen[4],"maeGuardPassed":bool(allowed)}
    return out


def write_forecasts(path: Path, rows: list[dict]) -> dict:
    """Write predictions in Parquet when available, otherwise explicit JSONL."""
    path.parent.mkdir(parents=True,exist_ok=True)
    # Reject accidental label leakage at the persistence boundary.
    forbidden={"actual","targetActual","label","realized","targetObserved"}
    for row in rows:
        if forbidden.intersection(row): raise ValueError("forecast rows must not contain actual labels")
    try:
        import pandas as pd
        import pyarrow  # noqa: F401
        target=path.with_suffix(".parquet")
        pd.DataFrame(rows).to_parquet(target,index=False)
        return {"path":str(target),"format":"parquet","rows":len(rows),"sha256":digest(target)}
    except ImportError:
        target=path.with_suffix(".jsonl")
        with target.open("w",encoding="utf-8") as f:
            for row in rows: f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
        return {"path":str(target),"format":"jsonl_fallback_no_parquet_engine","rows":len(rows),"sha256":digest(target)}


def save_checkpoint(stage: str, payload: dict, out: Path=OUT):
    if stage not in STAGES: raise ValueError(stage)
    out.mkdir(parents=True,exist_ok=True)
    p=out/f"{stage}.json"; tmp=p.with_suffix(".json.tmp")
    payload={**payload,"runIdentity":run_identity(),"upstreamCheckpointSha256":_upstream_hashes(stage,out)}
    tmp.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    os.replace(tmp,p)


def load_checkpoint(stage: str, out: Path=OUT) -> dict:
    p=out/f"{stage}.json"
    if not p.is_file(): raise FileNotFoundError(f"required checkpoint missing: {p}")
    doc=json.loads(p.read_text(encoding="utf-8"))
    if doc.get("runId")!=RUN_ID: raise ValueError(f"checkpoint identity mismatch: {p}")
    _assert_identity(doc,stage,out)
    return doc


def preflight(out: Path=OUT) -> dict:
    identity=run_identity()
    out.mkdir(parents=True,exist_ok=True)
    existing=out/"preflight.json"
    if existing.is_file():
        old_pf=json.loads(existing.read_text(encoding="utf-8"))
        _assert_identity(old_pf,"preflight",out)
        config_path=out/"resolved_config.json"
        if not config_path.is_file(): raise ValueError("compatible preflight is missing resolved_config.json")
        config=json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("runIdentity")!=identity:
            raise ValueError("resolved configuration provenance mismatch; refusing preflight overwrite")
        return old_pf
    protocol,raw=proxy.validate_inputs()
    if not PROTOCOL.is_file(): raise FileNotFoundError("frozen docs/period_mape10_protocol.md is required")
    if digest(PROTOCOL) != PROTOCOL_SHA: raise ValueError("frozen V3 protocol digest differs")
    if not OLD_REPORT.is_file() or digest(OLD_REPORT)!=OLD_REPORT_SHA:
        raise ValueError("pinned prior report is absent or differs from authorized digest")
    old=json.loads(OLD_REPORT.read_text(encoding="utf-8"))
    checks={"protocolId":protocol["id"],"sourceSha256":proxy.SOURCE_SHA,
            "oldReportSha256":OLD_REPORT_SHA,"priorCells":{},"protocolPath":str(PROTOCOL),
            "protocolSha256":digest(PROTOCOL),"forecastFormat":"parquet_if_engine_else_jsonl"}
    checks["analysisCodeSha256"]=analysis_code_hash()
    actuals={c:proxy._target_periods(s) for c,s in proxy.series_from_raw(raw).items()}
    macro={"F":[],"B0":[]}; cell_recheck={}
    for crop in proxy.CROPS:
        for h in proxy.HORIZONS:
            cell=old["crops"][crop][str(h)]
            f_by={}; b_by={}
            for r in cell["predictions"]:
                tp=(dt.date.fromisoformat(r["targetStart"]),dt.date.fromisoformat(r["targetEnd"]))
                actual=actuals[crop].get(tp,{}).get("actual")
                if actual is None: continue
                if "B1" in r["forecasts"]: f_by[tp]=float(r["forecasts"]["B1"]["predicted"])
                if "selectedB0" in r["forecasts"]: b_by[tp]=float(r["forecasts"]["selectedB0"]["predicted"])
            common=set(f_by)&set(b_by)
            exp_n=98 if h==2 else 97
            if len(f_by)!=exp_n or len(b_by)!=exp_n or len(common)!=exp_n:
                raise ValueError(f"prior F/B0 support mismatch {crop}/{h}: F={len(f_by)} B0={len(b_by)} paired={len(common)} expected={exp_n}")
            fpairs=[(actuals[crop][tp]["actual"],f_by[tp]) for tp in sorted(common)]
            bpairs=[(actuals[crop][tp]["actual"],b_by[tp]) for tp in sorted(common)]
            fm=proxy._metrics(fpairs);bm=proxy._metrics(bpairs)
            macro["F"].append(fm);macro["B0"].append(bm)
            cell_recheck[f"{crop}:{h}"]={"commonTargets":len(common),"F":fm,"B0":bm}
            checks["priorCells"][f"{crop}:{h}"]={"alpha":cell["selectedAlpha"],"b0":cell["selectedSimpleBaseline"],
                "F_test_forecasts":sum("B1" in r["forecasts"] for r in cell["predictions"]),
                "pairedReaggregation":cell_recheck[f"{crop}:{h}"],
                "F_dev_reconstruction":"R-identical alpha=1; synthesized only where absent",
                "F_fitEvidence":{"selectedAlpha":cell["selectedAlpha"],
                  "fitAuditEntries":cell["fitAudit"].get("B1",[]),
                  "storedModelObjects":False,"evidenceLimit":"saved report/logs only; no serialized historical scaler or coefficient object"}}
    reaggregated={m:{"MAPE8":float(np.mean([x["mape"] for x in ms])),"MAE8":float(np.mean([x["mae"] for x in ms]))}
                  for m,ms in macro.items()}
    expected={"F":{"MAPE8":14.460809093332058,"MAE8":232.31041756515413},
              "B0":{"MAPE8":15.732254045793303,"MAE8":221.2327823696221}}
    for m in expected:
        for metric,value in expected[m].items():
            if not math.isclose(reaggregated[m][metric],value,rel_tol=0,abs_tol=1e-8):
                raise ValueError(f"prior {m} {metric} reaggregation differs from pinned report value")
    checks.update({"baselineReaggregation":reaggregated,"baselineExpected":expected,
                   "runId":RUN_ID,"stage":"preflight","complete":True})
    save_checkpoint("preflight",checks,out)
    identity=run_identity()
    config={"runId":RUN_ID,"runIdentity":identity,"protocolSha256":checks["protocolSha256"],"sourceSha256":proxy.SOURCE_SHA,
        "priorReportSha256":OLD_REPORT_SHA,"analysisCodeSha256":checks["analysisCodeSha256"],
        "crops":list(proxy.CROPS),"horizons":list(proxy.HORIZONS),"methods":list(METHODS),
        "ridgeAlphasInheritedByCell":{f"{c}:{h}":checks["priorCells"][f"{c}:{h}"]["alpha"] for c in proxy.CROPS for h in proxy.HORIZONS},
        "tuningGrid":{"S":list(proxy.ALPHAS),"M":list(LAMBDA_GRID),"MW":list(LAMBDA_GRID)},
        "forecastFloor":FLOOR,"recencyHalfLifeCompletedDekads":HALF_LIFE_PERIODS,"dataSource":"pinned daily price snapshot only"}
    (out/"resolved_config.json").write_text(json.dumps(config,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return checks


def _train_arrays(series, targets, features, horizon, fit_origin):
    train_x=[]; ratios=[]; periods=[]; refs=[]
    valid=sorted(tp for tp,val in targets.items() if val.get("actual") is not None)
    for tp in valid:
        if not proxy.label_matured(tp,fit_origin): continue
        issue=proxy.origin_for(proxy._previous_period_target(tp,horizon))
        if issue>=fit_origin: continue
        row=features.row(issue,tp)
        if row is None: continue
        x,ref=row
        if not math.isfinite(ref) or ref<=0: continue
        ratio=targets[tp]["actual"]/ref
        if ratio<=0 or not math.isfinite(ratio): continue
        train_x.append(x); ratios.append(ratio); periods.append(tp); refs.append(ref)
    return np.asarray(train_x,dtype=float),np.asarray(ratios,dtype=float),periods


def _issue_fit(series, targets, features, horizon, fit_origin, method, alpha, lam=None):
    t0=time.perf_counter()
    X,y,periods=_train_arrays(series,targets,features,horizon,fit_origin)
    if len(y)<2: raise ValueError(f"insufficient matured training labels at {fit_origin}")
    weights=recency_weights(periods,fit_origin) if method in ("W","MW") else None
    target=np.log(y) if method in ("R","W") else y
    bundle=fit_ratio_model(method,X,target,alpha=alpha,lam=lam,sample_weights=weights)
    key_doc=[[p[0].isoformat(),p[1].isoformat()] for p in periods]
    key_hash=hashlib.sha256(json.dumps(key_doc,separators=(",",":")).encode()).hexdigest()
    bundle["trainingLabelKeysSha256"]=key_hash
    bundle["scalerIdentity"]=hashlib.sha256((bundle["scalerIdentity"]+key_hash).encode()).hexdigest()
    return bundle,periods,time.perf_counter()-t0


def _forecast_one(series, targets, features, horizon, target, origin, method, alpha, lam, model_store):
    bundle,train_periods,fit_seconds=_issue_fit(series,targets,features,horizon,origin,method,alpha,lam)
    row=features.row(origin,target)
    if row is None: raise ValueError(f"missing forecast feature row {origin} {target}")
    x,ref=row
    if ref<=0 or not math.isfinite(ref): raise ValueError("invalid forecast reference price")
    ratio,clipped=predict_ratio(bundle,x)
    predicted=ref*ratio
    raw_ratio=float(bundle["intercept"]+np.dot(np.asarray(bundle["coef"]),(np.asarray(x)-bundle["scalerMean"])/bundle["scalerScale"]))
    if method in ("R","W"):
        raw_ratio=math.exp(raw_ratio)
    model_identity={"bundle":bundle,"fitOrigin":origin.isoformat(),"target":target[0].isoformat()}
    model_id=hashlib.sha256(json.dumps(model_identity,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    model_store[model_id]={"modelId":model_id,"fitOrigin":origin.isoformat(),"method":method,
        "inputCutoff":proxy.input_cutoff(origin).isoformat(),"scalerIdentity":bundle["scalerIdentity"],"fitSeconds":fit_seconds,
        "lastTrainingTargetEnd":max((p[1].isoformat() for p in train_periods),default=None),
        "trainLabels":len(train_periods),"trainingTargets":[[p[0].isoformat(),p[1].isoformat()] for p in train_periods],
        "forecastInput":list(map(float,x)),"referencePrice":float(ref),"expectedRatio":float(ratio),
        "expectedPrediction":float(predicted),"bundle":bundle}
    return {"predicted":predicted,"referencePrice":ref,"ratioRaw":raw_ratio,"ratioUsed":ratio,
            "clippedAtFloor":clipped,"modelId":model_id,"trainLabels":len(train_periods),
            "scalerIdentity":bundle["scalerIdentity"],"inputCutoff":proxy.input_cutoff(origin).isoformat(),
            "fitSeconds":fit_seconds,
            "lastTrainingTargetEnd":max((p[1].isoformat() for p in train_periods),default=None),
            "alpha":alpha,"lambda":lam}


def _schedule_by_cell(start, end, horizon):
    return proxy.planned_origins(start,end,horizon)


def _calendar_block(start, end, horizon):
    return proxy.planned_origins(start,end,horizon)


def _score_rows(rows, actuals, method):
    pairs=[(actuals[r["targetKey"]],r["forecasts"][method]["predicted"])
           for r in rows if r["targetKey"] in actuals and method in r["forecasts"]]
    return {"mae":proxy._metrics(pairs)["mae"],"mape":proxy._metrics(pairs)["mape"],"n":len(pairs)}


def _save_models(out, models, stage):
    path=out/f"models_{stage}.jsonl"
    with path.open("w",encoding="utf-8") as f:
        for model in models.values(): f.write(json.dumps(model,ensure_ascii=False,allow_nan=False)+"\n")
    # Verify the persisted representation from disk using each fit's real forecast row.
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            model=json.loads(line)
            ratio,_=predict_ratio(model["bundle"],model["forecastInput"])
            prediction=model["referencePrice"]*ratio
            if not math.isclose(ratio,model["expectedRatio"],rel_tol=0,abs_tol=1e-10) or not math.isclose(
                    prediction,model["expectedPrediction"],rel_tol=0,abs_tol=1e-8):
                raise ValueError(f"on-disk model replay failed: {model['modelId']}")
    timeline=out/f"fit_timeline_{stage}.json"
    timeline_rows=[{k:model[k] for k in ("modelId","fitOrigin","inputCutoff","method","lastTrainingTargetEnd",
        "trainLabels","scalerIdentity","fitSeconds")} for model in models.values()]
    timeline.write_text(json.dumps(timeline_rows,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return {"path":str(path),"sha256":digest(path),"models":len(models),
            "timelinePath":str(timeline),"timelineSha256":digest(timeline)}

def _verify_artifact(meta):
    if not meta:return
    p=Path(meta["path"])
    if not p.is_file() or (meta.get("sha256") and digest(p)!=meta["sha256"]):
        raise ValueError(f"checkpoint artifact missing or digest mismatch: {p}")
    timeline=meta.get("timelinePath")
    if timeline and (not Path(timeline).is_file() or digest(Path(timeline))!=meta.get("timelineSha256")):
        raise ValueError(f"checkpoint timeline missing or digest mismatch: {timeline}")

def _verify_stage_checkpoint(stage,out):
    doc=load_checkpoint(stage,out)
    if stage=="pilot":_verify_artifact(doc.get("modelArtifact"))
    elif stage=="tune":_verify_artifact(doc.get("tuneModelArtifact"))
    elif stage=="develop":
        _verify_artifact(doc.get("forecastArtifact"));_verify_artifact(doc.get("modelArtifact"))
        p=out/"fit_timeline.json"
        if doc.get("fitTimelineSha256") and (not p.is_file() or digest(p)!=doc.get("fitTimelineSha256")):raise ValueError("development timeline digest mismatch")
        if doc.get("selectionSha256") and (not (out/"selection.json").is_file() or digest(out/"selection.json")!=doc.get("selectionSha256")):raise ValueError("selection digest mismatch")
    elif stage=="test":
        _verify_artifact(doc.get("forecastArtifact"));_verify_artifact(doc.get("modelArtifact"))
        p=out/"fit_timeline_test.json"
        if doc.get("fitTimelineSha256") and (not p.is_file() or digest(p)!=doc.get("fitTimelineSha256")):raise ValueError("test timeline digest mismatch")
        if doc.get("selectionSha256") and (not (out/"selection.json").is_file() or digest(out/"selection.json")!=doc.get("selectionSha256")):raise ValueError("test selection digest mismatch")
    return doc


def _verified_inputs(out):
    pf=load_checkpoint("preflight",out)
    return pf


def _load_data():
    protocol,raw=proxy.validate_inputs()
    return protocol,raw,proxy.series_from_raw(raw)


def pilot(out=OUT):
    _verified_inputs(out)
    _,raw,series_by_crop=_load_data()
    crop,h=proxy.CROPS[0],proxy.HORIZONS[0]
    series=series_by_crop[crop]; targets=proxy._target_periods(series); features=proxy.FeatureBuilder(series,targets)
    schedule=_tuning_schedules(h)
    if not schedule: raise ValueError("no pilot issue origins")
    chosen=schedule[0]; model_store={}; t0=time.perf_counter()
    fit=_forecast_one(series,targets,features,h,chosen["target"],chosen["origin"],"M",1.,LAMBDA_GRID[0],model_store)
    elapsed=time.perf_counter()-t0
    entry={"runId":RUN_ID,"stage":"pilot","complete":True,"fitSeconds":elapsed,
           "sampleFit":{"crop":crop,"horizon":h,"targetStart":chosen["target"][0].isoformat(),
              "targetEnd":chosen["target"][1].isoformat(),"origin":chosen["origin"].isoformat(),"modelId":fit["modelId"],"prediction":fit},
           "modelArtifact":_save_models(out,model_store,"pilot"),"pilotReusableFit":True,
           "implementation":"one LP example; remaining count estimated in tuning stage"}
    save_checkpoint("pilot",entry,out); return entry


def _make_tune_grid(series_by_crop):
    grid={}
    for crop in proxy.CROPS:
        series=series_by_crop[crop]; targets=proxy._target_periods(series); features=proxy.FeatureBuilder(series,targets)
        for h in proxy.HORIZONS:
            schedule=_tuning_schedules(h)
            seen=set(); picks=[]
            for s in schedule:
                q=proxy._quarter_key(s["origin"])
                if q not in seen: seen.add(q); picks.append(s)
            grid[(crop,h)]=(series,targets,features,picks)
    return grid

def _tuning_schedules(horizon):
    # Quarter is assigned by the publication/fit origin, never by target date.
    return [s for s in proxy.planned_origins(dt.date(2020,1,1),dt.date(2021,12,31),horizon)
            if dt.date(2020,1,1)<=s["origin"]<=dt.date(2021,12,31)]

def _tuning_evaluation_maturity(grid):
    ends=[s["target"][1] for _,targets,_,schedules in grid.values() for s in schedules
          if targets.get(s["target"],{}).get("actual") is not None]
    return max(ends)+dt.timedelta(days=2) if ends else None


def tune(out=OUT):
    _verified_inputs(out); pilot_doc=_verify_stage_checkpoint("pilot",out)
    _,_,series_by_crop=_load_data(); grid=_make_tune_grid(series_by_crop)
    params={"S":proxy.ALPHAS,"M":LAMBDA_GRID,"MW":LAMBDA_GRID}
    scores={m:{p:{} for p in values} for m,values in params.items()}
    models={}; fit_count=0; lp_seconds=[]; cache={}
    progress_path=out/"tune_progress.json"
    done_cells=set()
    if progress_path.is_file():
        progress=json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("runIdentity")!=run_identity() or progress.get("upstreamCheckpointSha256")!=_upstream_hashes("tune",out):
            raise ValueError("tuning progress provenance mismatch")
        done_cells=set(progress.get("completedCells",[]));fit_count=int(progress.get("fitCount",0));lp_seconds=progress.get("fitSeconds",[])
        for method,byparam in progress.get("scores",{}).items():
            for param,cells in byparam.items():scores[method][float(param)].update(cells)
        for item in progress.get("predictions",[]):
            key=(item["crop"],int(item["horizon"]),item["method"],float(item["parameter"]),
                 dt.date.fromisoformat(item["origin"]),(dt.date.fromisoformat(item["targetStart"]),dt.date.fromisoformat(item["targetEnd"])))
            cache[key]=item["prediction"]
        model_path=out/"models_tune_progress.jsonl"
        if progress.get("modelsSha256") and (not model_path.is_file() or digest(model_path)!=progress["modelsSha256"]):
            raise ValueError("tuning progress model checkpoint digest mismatch")
        if model_path.is_file():
            for line in model_path.read_text(encoding="utf-8").splitlines():
                item=json.loads(line);models[item["modelId"]]=item
    # Reuse the measured pilot LP as the first matching lambda=0.001 tuning fit.
    pilot_models_path=Path(pilot_doc["modelArtifact"]["path"])
    for line in pilot_models_path.read_text(encoding="utf-8").splitlines():
        item=json.loads(line); models[item["modelId"]]=item
    ps=pilot_doc["sampleFit"]
    pilot_key=(ps["crop"],int(ps["horizon"]),"M",LAMBDA_GRID[0],dt.date.fromisoformat(ps["origin"]),
               (dt.date.fromisoformat(ps["targetStart"]),dt.date.fromisoformat(ps["targetEnd"])))
    cache.setdefault(pilot_key,ps["prediction"])
    for (crop,h),(series,targets,features,schedules) in grid.items():
        cell_id=f"{crop}:{h}"
        if cell_id in done_cells:continue
        for method,values in params.items():
            for param in values:
                ypred=[]; actuals=[]
                for s in schedules:
                    cache_key=(crop,h,method,param,s["origin"],s["target"])
                    if cache_key not in cache:
                        t0=time.perf_counter()
                        pred=_forecast_one(series,targets,features,h,s["target"],s["origin"],method,
                            (proxy.ALPHAS[0] if method in ("M","MW") else param),
                            param if method in ("M","MW") else None,models)
                        lp_seconds.append(time.perf_counter()-t0); fit_count+=1; cache[cache_key]=pred
                    actual=targets.get(s["target"],{}).get("actual")
                    if actual is not None:
                        ypred.append((actual,cache[cache_key]["predicted"])); actuals.append(s["target"])
                metrics=proxy._metrics(ypred)
                scores[method][param][f"{crop}:{h}"]=metrics
        done_cells.add(cell_id)
        mp=out/"models_tune_progress.jsonl"
        with mp.open("w",encoding="utf-8") as f:
            for model in models.values():f.write(json.dumps(model,allow_nan=False)+"\n")
        progress={"runId":RUN_ID,"runIdentity":run_identity(),"upstreamCheckpointSha256":_upstream_hashes("tune",out),
          "analysisCodeSha256":analysis_code_hash(),"protocolSha256":PROTOCOL_SHA,
          "modelsSha256":digest(mp),
          "completedCells":sorted(done_cells),"fitCount":fit_count,"fitSeconds":lp_seconds,
          "scores":{m:{str(p):v for p,v in by.items()} for m,by in scores.items()},
          "predictions":[{"crop":k[0],"horizon":k[1],"method":k[2],"parameter":k[3],"origin":k[4].isoformat(),
              "targetStart":k[5][0].isoformat(),"targetEnd":k[5][1].isoformat(),"prediction":v} for k,v in cache.items()]}
        tmp=progress_path.with_suffix(".json.tmp");tmp.write_text(json.dumps(progress,allow_nan=False)+"\n",encoding="utf-8");os.replace(tmp,progress_path)
    selected={}
    for method,byparam in scores.items():
        cells={cell:{param:byparam[param][cell] for param in byparam} for cell in next(iter(byparam.values()))}
        selected[method]=choose_global_tuning(cells)
    tune_jobs=sum(len(sched)*(len(LAMBDA_GRID)*2+len(proxy.ALPHAS)) for _,_,_,sched in grid.values())
    dev_jobs=sum(len(proxy.planned_origins(dt.date(2022,1,1),dt.date(2023,12,31),h))
        for c in proxy.CROPS for h in proxy.HORIZONS)*len(METHODS)
    test_jobs=sum(sum(s['origin']>=proxy.POLICY_AVAILABLE_FROM for s in proxy.planned_origins(
        dt.date(2024,1,1),dt.date(2026,9,30),h)) for c in proxy.CROPS for h in proxy.HORIZONS)*len(METHODS)
    projected=tune_jobs+dev_jobs+test_jobs
    mean_sec=float(np.mean(lp_seconds)) if lp_seconds else None
    max_end=max((m.get("lastTrainingTargetEnd") for m in models.values() if m.get("lastTrainingTargetEnd")),default=None)
    maturity_date=_tuning_evaluation_maturity(grid)
    maturity=maturity_date.isoformat() if maturity_date else None
    doc={"runId":RUN_ID,"stage":"tune","complete":True,"selectedGlobalParameters":selected,
         "scores":scores,"originGrid":"first lawful issue in each calendar quarter of 2020-2021",
         "fitCount":fit_count,"meanObservedFitSeconds":mean_sec,"estimatedRemainingFits":max(0,projected-fit_count),
         "estimatedRemainingSeconds":None if mean_sec is None else mean_sec*max(0,projected-1-fit_count),
         "pilotFitSeconds":pilot_doc["fitSeconds"],"tuneModelArtifact":_save_models(out,models,"tune"),
         "maxTuningTargetMaturity":maturity,
         "tuningPredictions":[{"crop":k[0],"horizon":k[1],"method":k[2],"parameter":k[3],
              "origin":k[4].isoformat(),"targetStart":k[5][0].isoformat(),"targetEnd":k[5][1].isoformat(),
              "prediction":v} for k,v in cache.items()]}
    save_checkpoint("tune",doc,out)
    progress_path.unlink(missing_ok=True);(out/"models_tune_progress.jsonl").unlink(missing_ok=True)
    return doc


def _cell_rows(series_by_crop, crop, h, start, end, policy=None, old=None, models=None):
    series=series_by_crop[crop]; targets=proxy._target_periods(series); features=proxy.FeatureBuilder(series,targets)
    rows=[]
    for s in proxy.planned_origins(start,end,h):
        target=s["target"]; origin=s["origin"]
        row={"crop":crop,"horizon":h,"targetStart":target[0].isoformat(),"targetEnd":target[1].isoformat(),
             "targetKey":target[0].isoformat(),"origin":origin.isoformat(),
             "inputCutoff":s["inputCutoff"].isoformat(),
             "forecasts":{}}
        if policy is not None:
            row["policyMethod"]=policy
        rows.append((s,row,series,targets,features))
    return rows


def _load_previous():
    if digest(OLD_REPORT)!=OLD_REPORT_SHA: raise ValueError("old frozen report digest changed")
    return json.loads(OLD_REPORT.read_text(encoding="utf-8"))


def _year_actual_maps(series_by_crop, crop, h, year):
    targets=proxy._target_periods(series_by_crop[crop])
    return {tp[0].isoformat():v["actual"] for tp,v in targets.items()
            if v.get("actual") is not None and tp[0].year==year}


def develop(out=OUT):
    _verified_inputs(out); tune_doc=_verify_stage_checkpoint("tune",out)
    _,_,series_by_crop=_load_data(); old=_load_previous(); models={}; rows=[]; audit=[]
    maturity=dt.date.fromisoformat(tune_doc["maxTuningTargetMaturity"])
    per_cell={}; f_mae={}
    for crop in proxy.CROPS:
        per_cell[crop]={}
        for h in proxy.HORIZONS:
            result=old["crops"][crop][str(h)]
            alpha=float(result["selectedAlpha"])
            cellrows=[x for x in _cell_rows(series_by_crop,crop,h,dt.date(2022,1,1),dt.date(2023,12,31))
                      if x[0]["origin"]>maturity]
            for s,row,series,targets,features in cellrows:
                for method in METHODS:
                    param=float(tune_doc["selectedGlobalParameters"][method]) if method in ("S","M","MW") else alpha
                    lam=param if method in ("M","MW") else None
                    a=1. if method in ("M","MW") else param
                    t0=time.perf_counter()
                    pred=_forecast_one(series,targets,features,h,s["target"],s["origin"],method,a,lam,models)
                    audit.append({"crop":crop,"horizon":h,"method":method,"fitOrigin":s["origin"].isoformat(),
                        "targetStart":s["target"][0].isoformat(),"lastTrainingTargetEnd":pred["lastTrainingTargetEnd"],
                        "inputCutoff":pred["inputCutoff"],"scalerIdentity":pred["scalerIdentity"],
                        "trainLabels":pred["trainLabels"],"elapsedSeconds":pred["fitSeconds"],"modelId":pred["modelId"]})
                    row["forecasts"][method]=pred
                # At development cadence, F is the exact R fit reconstructed from saved R predictions.
                row["forecasts"]["F"]={**row["forecasts"]["R"],"reconstructedFrom":"R_identical_policy","notIndependentFit":True}
            rows.extend(x[1] for x in cellrows)
            # Exact same-key metrics: all planned observed target keys must exist in every method.
            actuals=_year_actual_maps(series_by_crop,crop,h,2022)|_year_actual_maps(series_by_crop,crop,h,2023)
            for y in (2022,2023):
                per_cell[crop].setdefault(str(y),{})
                subset=[r for r in rows if r["crop"]==crop and r["horizon"]==h and dt.date.fromisoformat(r["targetStart"]).year==y]
                keys={r["targetKey"] for r in subset if r["targetKey"] in actuals}
                for method in METHODS+ ("F",):
                    method_keys={r["targetKey"] for r in subset if r["targetKey"] in actuals and method in r["forecasts"]}
                    if method_keys != keys: raise ValueError(f"development missing paired keys {crop}/{h}/{y}/{method}")
                    per_cell[crop][str(y)].setdefault(str(h),{})[method]=proxy._metrics([
                        (actuals[k],next(r["forecasts"][method]["predicted"] for r in subset if r["targetKey"]==k)) for k in sorted(keys)])
            if len(cellrows)!=len(proxy.planned_origins(dt.date(2022,1,1),dt.date(2023,12,31),h)):
                raise ValueError("development origin grid was truncated")
    # MAE guard is computed from exact paired R/F reconstructed values.
    for crop in proxy.CROPS:
        f_mae[crop]={}
        for y in (2022,2023):
            vals=[per_cell[crop][str(y)][str(h)]["F"]["mae"] for h in proxy.HORIZONS]
            f_mae[crop][str(y)]=float(np.mean(vals))
    policies=select_crop_policies(per_cell,f_mae)
    # P_global selects one method on the exact 8-cell mean development MAPE.
    global_scores={m:float(np.mean([np.mean([per_cell[c][str(y)][str(h)][m]["mape"]
        for c in proxy.CROPS for h in proxy.HORIZONS]) for y in (2022,2023)])) for m in METHODS}
    best_global=min(global_scores.values())
    pglobal=min((m for m,v in global_scores.items() if v<=best_global+1e-10),
                key=lambda m:("M","MW","R","W","S").index(m))
    selection={"P":policies,"Pglobal":{"method":pglobal,"developmentMAPE8":global_scores[pglobal],"diagnosticOnly":True},
               "globalMethodScores":global_scores,"developmentScores":per_cell,"fMaeGuard":f_mae,
               "tuningMaturityCutoff":maturity.isoformat(),"alphaRetrospectiveCaveat":True}
    (out/"selection.json").write_text(json.dumps(selection,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    config=json.loads((out/"resolved_config.json").read_text(encoding="utf-8"))
    config.update({"selectedGlobalParameters":tune_doc["selectedGlobalParameters"],"P":policies,
                   "Pglobal":selection["Pglobal"],"tuningMaturityCutoff":maturity.isoformat()})
    (out/"resolved_config.json").write_text(json.dumps(config,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    forecast_meta=write_forecasts(out/"forecasts.parquet",rows)
    (out/"fit_timeline.json").write_text(json.dumps(audit,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    model_meta=_save_models(out,models,"develop")
    doc={"runId":RUN_ID,"stage":"develop","complete":True,"forecastArtifact":forecast_meta,
         "fitTimelineSha256":digest(out/"fit_timeline.json"),"modelArtifact":model_meta,
         "selectionSha256":digest(out/"selection.json"),"rows":len(rows),"fitCount":len(audit),
         "yearlyDevelopmentScores":per_cell,"selection":selection}
    save_checkpoint("develop",doc,out); return doc


def test_stage(out=OUT):
    _verified_inputs(out); dev=_verify_stage_checkpoint("develop",out)
    _,_,series_by_crop=_load_data(); old=_load_previous(); selection=json.loads((out/"selection.json").read_text(encoding="utf-8"))
    allrows=[]; models={}; fit_timeline=[]
    start,end=dt.date(2024,1,1),dt.date(2026,9,30)
    for crop in proxy.CROPS:
        for h in proxy.HORIZONS:
            series=series_by_crop[crop]; targets=proxy._target_periods(series); features=proxy.FeatureBuilder(series,targets)
            oldcell=old["crops"][crop][str(h)]
            saved={ (r["targetStart"],r["targetEnd"]):r for r in oldcell["predictions"] }
            planned=proxy.planned_origins(start,end,h)
            if len(planned)!=99: raise ValueError(f"unexpected test target grid size {crop}/{h}: {len(planned)}")
            selected=selection["P"][crop]["method"]; global_method=selection["Pglobal"]["method"]
            for s in planned:
                tp=s["target"]; issue=s["origin"]
                target_key=tp[0].isoformat(); oldrow=saved.get((target_key,tp[1].isoformat()))
                row={"crop":crop,"horizon":h,"targetStart":target_key,"targetEnd":tp[1].isoformat(),
                    "targetKey":target_key,"origin":issue.isoformat(),"inputCutoff":s["inputCutoff"].isoformat(),
                    "policyUnavailable":issue<proxy.POLICY_AVAILABLE_FROM,
                    "policyUnavailableReason":"development_labels_mature_2024-01-02" if issue<proxy.POLICY_AVAILABLE_FROM else None,
                    "forecasts":{}}
                if oldrow:
                    if "B1" in oldrow["forecasts"]:
                        row["forecasts"]["F"]={"predicted":oldrow["forecasts"]["B1"]["predicted"],
                            "source":"saved_prior_quarterly_F","fitAt":oldrow["forecasts"]["B1"].get("fitAt")}
                    if "selectedB0" in oldrow["forecasts"]:
                        row["forecasts"]["B0"]={"predicted":oldrow["forecasts"]["selectedB0"]["predicted"],
                            "baseline":oldcell["selectedSimpleBaseline"],"source":"saved_prior_selected_B0"}
                if issue>=proxy.POLICY_AVAILABLE_FROM:
                    for method in METHODS:
                        # Fixed before test: R/W inherited per-cell prior alpha; S/M/MW use one global tuned value.
                        if method in ("R","W"): param=float(oldcell["selectedAlpha"])
                        else: param=float(load_checkpoint("tune",out)["selectedGlobalParameters"][method])
                        pred=_forecast_one(series,targets,features,h,tp,issue,method,
                            1. if method in ("M","MW") else param,
                            param if method in ("M","MW") else None,models)
                        row["forecasts"][method]=pred
                        fit_timeline.append({"crop":crop,"horizon":h,"method":method,"fitOrigin":issue.isoformat(),
                            "targetStart":target_key,"lastTrainingTargetEnd":pred["lastTrainingTargetEnd"],
                            "inputCutoff":pred["inputCutoff"],"scalerIdentity":pred["scalerIdentity"],
                            "trainLabels":pred["trainLabels"],"elapsedSeconds":pred["fitSeconds"],"modelId":pred["modelId"]})
                    row["forecasts"]["P"]={**row["forecasts"][selected],"selectedMethod":selected}
                    row["forecasts"]["Pglobal"]={**row["forecasts"][global_method],"selectedMethod":global_method}
                allrows.append(row)
            expected_f=98 if h==2 else 97
            fkeys={r["targetKey"] for r in allrows if r["crop"]==crop and r["horizon"]==h and r["forecasts"].get("F")}
            pkeys={r["targetKey"] for r in allrows if r["crop"]==crop and r["horizon"]==h and r["forecasts"].get("P")}
            if len(fkeys)!=expected_f or pkeys!=fkeys:
                raise ValueError(f"test support mismatch: expected F/P={expected_f}, got F={len(fkeys)}, P={len(pkeys)} for {crop}/{h}")
    if len(allrows)!=len(proxy.CROPS)*2*99: raise ValueError("test forecast calendar lost planned targets")
    if any("actual" in r for r in allrows): raise ValueError("test forecasts leaked labels")
    forecast_meta=write_forecasts(out/"forecasts_test.parquet",allrows)
    (out/"fit_timeline_test.json").write_text(json.dumps(fit_timeline,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    model_meta=_save_models(out,models,"test")
    doc={"runId":RUN_ID,"stage":"test","complete":True,"rows":len(allrows),"forecastArtifact":forecast_meta,
         "policyUnavailableCount":sum(r["policyUnavailable"] for r in allrows),"fitCount":len(fit_timeline),
         "fitTimelineSha256":digest(out/"fit_timeline_test.json"),"modelArtifact":model_meta,
         "developmentArtifactSha256":digest(Path(dev["forecastArtifact"]["path"])),
         "selectionSha256":digest(out/"selection.json")}
    save_checkpoint("test",doc,out); return doc


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=STAGES+("run",))
    p.add_argument("--output-dir",type=Path,default=OUT)
    args=p.parse_args(argv)
    actions={"preflight":preflight,"pilot":pilot,"tune":tune,"develop":develop,"test":test_stage}
    results=[]
    stages=STAGES if args.stage=="run" else (args.stage,)
    for stage in stages:
        if stage!="preflight": _verified_inputs(args.output_dir)
        if stage=="score":
            from . import period_mape10_score
            result=period_mape10_score.score_saved(args.output_dir)
        else:
            checkpoint=args.output_dir/f"{stage}.json"
            if stage!="preflight" and checkpoint.is_file():
                cached=_verify_stage_checkpoint(stage,args.output_dir)
                if not cached.get("complete"): raise ValueError(f"incomplete checkpoint: {checkpoint}")
                result=cached
            else: result=actions[stage](args.output_dir)
        results.append(result)
    final=results[-1]
    summary={k:final[k] for k in ("stage","complete","fitCount","rows","forecastArtifact",
        "policyUnavailableCount","baselineReaggregation","selectedGlobalParameters") if k in final}
    summary["outputDir"]=str(args.output_dir)
    print(json.dumps(summary,ensure_ascii=False,default=str))


if __name__=="__main__": main()
