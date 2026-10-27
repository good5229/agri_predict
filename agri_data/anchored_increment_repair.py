"""V2 low-dimensional calibration of saved anchored forecast increments.

The calibration command reads only V target dates through 2026-06-20. T
forecast generation consumes the frozen calibration without labels; T scoring
is a separate command which verifies that forecast and calibration first.
No base model is fitted or invoked here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import crop_specialist_assembly_score as inference
from . import full_history_return_eval as price_helper
from . import research_progress_diagnostic as diagnostics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/anchored-increment-repair-v2-20261013"
PROTOCOL_SHA256 = "2d2b59bc5992995f6e5d598d494a3b6fd3e4996ad2368f5bc4db124d0b3e831f"
PRICE_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
K1_PATHS = {
    "V": "outputs/experiments/anchored-increment-diagnostic-v1-20261013/V/forecasts.json",
    "T": "outputs/experiments/anchored-increment-diagnostic-v1-20261013/T/forecasts.json",
}
K1_FILE_SHA256 = {
    "V": "c0af6b4a20b4354046f9c02b102563244928e9929ae01e3ae43be89304576b09",
    "T": "6ca9daf6f0b1c2fe787c202bd2f03b697603bb1abc38a6ebded894db202d6afd",
}
K1_SCORE_PATH = "outputs/experiments/anchored-increment-diagnostic-v1-20261013/T/score-report.json"
K1_SCORE_FILE_SHA256 = "de36ee0aa41ee7afc9ea80347b191751b18a6dc9d49fdf7ea3ce89b83c19c2c8"
PRODUCTS = ("cabbage", "radish", "onion", "garlic")
CALIBRATED_PRODUCTS = ("radish", "onion")
HORIZONS = tuple(range(1, 11))
PERIODS = {
    "V": (dt.date(2026, 4, 2), dt.date(2026, 6, 20), dt.date(2026, 4, 1)),
    "T": (dt.date(2026, 7, 11), dt.date(2026, 9, 27), dt.date(2026, 7, 1)),
}
C_START, C_END = dt.date(2026, 4, 2), dt.date(2026, 5, 31)
DEV_START, DEV_END, DEV_ORIGIN_START = dt.date(2026, 6, 2), dt.date(2026, 6, 20), dt.date(2026, 6, 1)
ALPHA_BOUNDS = (-1.0, 1.0)
PRICE_FLOOR_RATIO = 1e-8
SOLVER_METHOD = "highs"
SOLVER_OPTIONS = {"presolve": True, "primal_feasibility_tolerance": 1e-9,
                  "dual_feasibility_tolerance": 1e-9, "ipm_optimality_tolerance": 1e-10}
OBJECTIVE_FACE_TOL = 1e-10  # weighted fractional MAPE units; never treated as gain
VERIFY_ABS_TOL = 1e-6  # weighted MAE currency/kg feasibility check
SCORE_MAPE_TOL_PP = 1e-8
SCORE_MAE_TOL = 1e-7
SIGN_SUPPORT_DAYS = 10
_DATE_MIN = dt.date.min


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _document_sha(doc: Mapping[str, Any], field: str) -> str:
    unsigned=dict(doc); expected=unsigned.pop(field,None); actual=_sha(_canonical(unsigned))
    if expected != actual: raise ValueError(f"{field} mismatch")
    return actual


def _path(value: str | Path) -> Path:
    p=Path(value)
    return (ROOT/p).resolve() if not p.is_absolute() else p.resolve()


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    try: fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    except FileExistsError: raise FileExistsError(f"refusing to overwrite frozen artifact: {path}") from None
    with os.fdopen(fd,"wb") as f:
        f.write(payload); f.flush(); os.fsync(f.fileno())


def _read(path: Path) -> dict[str, Any]:
    value=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value,dict): raise ValueError(f"expected JSON object: {path}")
    return value


def _number(value: Any, label: str, *, positive: bool=False) -> float:
    if isinstance(value,bool): raise ValueError(f"{label} cannot be boolean")
    try: x=float(value)
    except (TypeError,ValueError,OverflowError) as exc: raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(x) or (positive and x<=0): raise ValueError(f"{label} must be finite"+(" and positive" if positive else ""))
    return x


def _key(row: Mapping[str,Any]) -> tuple[str,str,str,int]:
    try:
        p=row["product"]; origin=dt.date.fromisoformat(row["origin"]); target=dt.date.fromisoformat(row["target"]); h=row["horizon"]
    except (KeyError,TypeError,ValueError) as exc: raise ValueError("malformed forecast key") from exc
    if p not in PRODUCTS or h not in HORIZONS or origin != target-dt.timedelta(days=h):
        raise ValueError("forecast origin/target/horizon key mismatch")
    return p,origin.isoformat(),target.isoformat(),h


def _expected_keys(phase: str) -> set[tuple]:
    start,end,origin_start=PERIODS[phase]
    return {(p,(t-dt.timedelta(days=h)).isoformat(),t.isoformat(),h)
            for p in PRODUCTS for h in HORIZONS
            for t in (start+dt.timedelta(days=i) for i in range((end-start).days+1))
            if t-dt.timedelta(days=h)>=origin_start}


def _load_k1(phase: str, supplied_path: str | Path | None=None) -> tuple[dict[str,Any],dict[tuple,dict[str,Any]],str]:
    path=_path(supplied_path or K1_PATHS[phase]); raw=path.read_bytes(); digest=_sha(raw)
    if digest != K1_FILE_SHA256[phase]: raise ValueError(f"pinned K1/{phase} forecast file hash mismatch")
    doc=_read(path); _document_sha(doc,"manifestSha256")
    if doc.get("experiment")!="ANCHORED_INCREMENT_DIAGNOSTIC_V1" or doc.get("phase")!=phase or doc.get("targetValuesAttached") is not False:
        raise ValueError("expected immutable answer-free K1 V/T artifact")
    if doc.get("targetSnapshotSha256")!=PRICE_SHA256: raise ValueError("K1 target snapshot version differs from protocol")
    if _sha((OUT/"protocol.md").read_bytes())!=PROTOCOL_SHA256: raise ValueError("V2 protocol file hash changed")
    rows=doc.get("forecasts")
    if not isinstance(rows,list) or _sha(_canonical(rows))!=doc.get("forecastRowsSha256"):
        raise ValueError("K1 forecast rows digest mismatch")
    indexed={}
    for row in rows:
        k=_key(row)
        if k in indexed: raise ValueError("duplicate K1 forecast key")
        if "actual" in row or "actualPrice" in row or "targetValue" in row: raise ValueError("K1 forecast unexpectedly contains target labels")
        b=_number(row.get("baselineB"),"baselineB",positive=True)
        delta=_number(row.get("delta"),"delta")
        k1=_number(row.get("prediction"),"K1 prediction",positive=True)
        indexed[k]={**row,"_key":k,"_b":b,"_delta":delta,"_k1":k1,"_sourceRow":row}
    if set(indexed)!=_expected_keys(phase):
        raise ValueError(f"K1/{phase} grid mismatch: missing={len(_expected_keys(phase)-set(indexed))}, extra={len(set(indexed)-_expected_keys(phase))}")
    return doc,indexed,digest


def _limited_actuals(prices_path: str | Path, through: dt.date) -> tuple[dict[tuple[str,str],float],str]:
    path=_path(prices_path); raw=path.read_bytes(); digest=_sha(raw)
    if digest!=PRICE_SHA256: raise ValueError("pinned price snapshot file hash mismatch")
    series=price_helper._series(json.loads(raw))
    # Deliberately expose only the protocol-allowed V dates to calibration code.
    out={}
    for product,values in series.items():
        for day,value in values.items():
            if day<=through and value>0:
                out[(product,day.isoformat())]=value
    return out,digest


def _join(indexed: Mapping[tuple,dict[str,Any]], actuals: Mapping[tuple[str,str],float],
          start: dt.date, end: dt.date, origin_start: dt.date) -> list[dict[str,Any]]:
    output=[]
    for k,row in sorted(indexed.items()):
        p,o,t,h=k; target=dt.date.fromisoformat(t); origin=dt.date.fromisoformat(o)
        if target<start or target>end or origin<origin_start: continue
        actual=actuals.get((p,t))
        if actual is None: continue
        output.append({"product":p,"origin":o,"target":t,"horizon":h,"actual":float(actual),
                       "B":row["_b"],"delta":row["_delta"],"K1":row["_k1"],"sourceRow":row})
    return output


def _weights(records: list[dict[str,Any]]) -> list[float]:
    counts=defaultdict(int)
    for row in records: counts[row["horizon"]]+=1
    if any(counts[h]==0 for h in HORIZONS): raise ValueError("calibration period lacks positive-target support for a horizon")
    return [1.0/(10*counts[r["horizon"]]) for r in records]


def _policy_prediction(row: Mapping[str,Any], coeffs: Mapping[str,Any], *, structure: str) -> dict[str,Any]:
    p=row["product"]; b=_number(row.get("baselineB",row.get("B")),"B",positive=True)
    d=_number(row.get("delta"),"delta")
    if p not in CALIBRATED_PRODUCTS: alpha=0.0
    elif structure=="R1": alpha=float(coeffs[p])
    elif structure=="R2":
        by_sign=coeffs.get(p)
        if not isinstance(by_sign,Mapping): alpha=float(coeffs.get("r1Fallback",{}).get(p,0.0))
        elif d>0: alpha=float(by_sign["positive"])
        elif d<0: alpha=float(by_sign["negative"])
        else: alpha=0.0
    else: raise ValueError(f"unknown coefficient structure: {structure}")
    if not math.isfinite(alpha) or not -1.0<=alpha<=1.0: raise ValueError("frozen alpha is outside [-1,1]")
    raw=b+alpha*d
    reason=None
    if not math.isfinite(raw): reason="raw_overflow"
    elif raw<=0: reason="raw_nonpositive"
    value=b if reason else raw
    return {"prediction":value,"rawPrediction":raw if math.isfinite(raw) else "nonfinite",
            "alpha":alpha,"fallbackReason":reason,"appliedDelta":value-b}


def _fit_lp(records: list[dict[str,Any]], *, signs: bool=False, fit_name: str="R1") -> dict[str,Any]:
    """One low-dimensional constrained fit, with a deterministic |alpha| tie pass."""
    from scipy.optimize import linprog
    if not records: raise ValueError("cannot fit calibration on empty records")
    weights=_weights(records); keys=("positive","negative") if signs else ("all",)
    n=len(records); k=len(keys); alpha_col={name:i for i,name in enumerate(keys)}
    base_mae=math.fsum(w*abs(r["actual"]-r["B"]) for r,w in zip(records,weights))
    # LP variables are [alpha(1|2), absolute residual(n)].
    m=k+n; objective=np.zeros(m); A=[]; rhs=[]
    for i,(r,w) in enumerate(zip(records,weights)):
        d=r["delta"]; err=r["actual"]-r["B"]
        if signs:
            group="positive" if d>0 else "negative" if d<0 else None
            ai=alpha_col.get(group) if group else None
        else: ai=0
        if ai is None:  # delta==0 has no effect and no alpha constraint
            ai=0; d=0.0
        left=np.zeros(m); left[ai]=-d; left[k+i]=-1.0; A.append(left); rhs.append(-err)
        right=np.zeros(m); right[ai]=d; right[k+i]=-1.0; A.append(right); rhs.append(err)
        floor=np.zeros(m); floor[ai]=-d; A.append(floor); rhs.append((1.0-PRICE_FLOOR_RATIO)*r["B"])
        objective[k+i]=w/r["actual"]
    mae=np.zeros(m); mae[k:]=weights; A.append(mae); rhs.append(base_mae)
    bounds=[ALPHA_BOUNDS]*k+[(0,None)]*n
    result=linprog(objective,A_ub=np.asarray(A),b_ub=np.asarray(rhs),bounds=bounds,
                   method=SOLVER_METHOD,options=dict(SOLVER_OPTIONS))
    if not result.success or result.x is None: raise RuntimeError(f"{fit_name} primary LP failed: {result.status} {result.message}")
    primary_obj=float(result.fun)
    # Lexicographic second LP: minimize sum |alpha| without leaving the primary optimum face.
    m2=k+n+k; objective2=np.zeros(m2); objective2[k+n:]=1.0
    A2=[]; rhs2=[]
    for oldrow,oldrhs in zip(A,rhs):
        new=np.zeros(m2); new[:m]=oldrow; A2.append(new); rhs2.append(oldrhs)
    face=np.zeros(m2); face[k:k+n]=objective[k:]; A2.append(face); rhs2.append(primary_obj+OBJECTIVE_FACE_TOL)
    for j in range(k):
        pos=np.zeros(m2); pos[j]=1; pos[k+n+j]=-1; A2.append(pos); rhs2.append(0.0)
        neg=np.zeros(m2); neg[j]=-1; neg[k+n+j]=-1; A2.append(neg); rhs2.append(0.0)
    bounds2=[ALPHA_BOUNDS]*k+[(0,None)]*n+[(0,None)]*k
    tie=linprog(objective2,A_ub=np.asarray(A2),b_ub=np.asarray(rhs2),bounds=bounds2,
                method=SOLVER_METHOD,options=dict(SOLVER_OPTIONS))
    if not tie.success or tie.x is None: raise RuntimeError(f"{fit_name} tie-break LP failed: {tie.status} {tie.message}")
    alpha=[float(x) for x in tie.x[:k]]
    mae_new=0.; objective_new=0.; floor_margin=math.inf
    for r,w in zip(records,weights):
        if signs:
            group="positive" if r["delta"]>0 else "negative" if r["delta"]<0 else None
            a=alpha[alpha_col[group]] if group else 0.0
        else: a=alpha[0]
        err=r["actual"]-r["B"]-a*r["delta"]
        mae_new+=w*abs(err)
        objective_new+=w*abs(err)/r["actual"]
        floor_margin=min(floor_margin,(r["B"]+a*r["delta"])/r["B"]-PRICE_FLOOR_RATIO)
    mae_residual=mae_new-base_mae; objective_gap=objective_new-primary_obj
    if mae_residual>VERIFY_ABS_TOL: raise RuntimeError(f"{fit_name} MAE feasibility residual {mae_residual} exceeds tolerance")
    if floor_margin < -SOLVER_OPTIONS["primal_feasibility_tolerance"]*max(1.,PRICE_FLOOR_RATIO):
        raise RuntimeError(f"{fit_name} positivity constraint failed by {floor_margin}")
    if objective_gap < -1e-9 or objective_gap>OBJECTIVE_FACE_TOL+1e-9:
        raise RuntimeError(f"{fit_name} tie-break left the recorded primary optimum face: {objective_gap}")
    return {"fitName":fit_name,"logicalFitCount":1,"alphas":dict(zip(keys,alpha)),
            "records":n,"uniqueTargetDays":len({r["target"] for r in records}),
            "horizonCounts":{str(h):sum(r["horizon"]==h for r in records) for h in HORIZONS},
            "baselineWeightedMae":base_mae,"fittedWeightedMae":mae_new,
            "maeFeasibilityResidual":mae_residual,"primaryWeightedMapeFraction":primary_obj,
            "tieBreakWeightedMapeFraction":objective_new,"primaryObjectiveGap":objective_gap,
            "minPriceFloorMarginRatio":floor_margin,"primarySolver":{"success":bool(result.success),"status":int(result.status),"message":result.message,"method":SOLVER_METHOD},
            "tieBreakSolver":{"success":bool(tie.success),"status":int(tie.status),"message":tie.message,"method":SOLVER_METHOD},
            "numerics":{"solverOptions":SOLVER_OPTIONS,"objectiveFaceToleranceFraction":OBJECTIVE_FACE_TOL,
                        "maeVerificationAbsTolerance":VERIFY_ABS_TOL,"scoreMapeTolerancePercentagePoints":SCORE_MAPE_TOL_PP,
                        "scoreMaeTolerance":SCORE_MAE_TOL,"priceFloorRatio":PRICE_FLOOR_RATIO}}


def _apply(records: list[dict[str,Any]], coeffs: Mapping[str,Any], structure: str) -> list[dict[str,Any]]:
    output=[]
    for row in records:
        result=_policy_prediction(row,coeffs,structure=structure)
        output.append({**row,structure:result["prediction"],f"{structure}Raw":result["rawPrediction"],
                       f"{structure}Alpha":result["alpha"],f"{structure}Fallback":result["fallbackReason"]})
    return output


def _combined_r2_candidate(records: list[dict[str,Any]], r2_coeffs: Mapping[str,Any],
                           r1_coeffs: Mapping[str,float]) -> list[dict[str,Any]]:
    """Build the one R2 policy: every eligible crop sign-splits; all others use R1."""
    eligible=set(r2_coeffs)
    if not eligible or not eligible.issubset(CALIBRATED_PRODUCTS):
        raise ValueError("R2 candidate requires at least one eligible calibrated crop")
    if set(r1_coeffs)!=set(CALIBRATED_PRODUCTS):
        raise ValueError("R2 candidate requires the complete R1 fallback map")
    return _apply(records,{**r2_coeffs,"r1Fallback":r1_coeffs},"R2")


def _global_r2_gate(r1_records: list[dict[str,Any]], r2_records: list[dict[str,Any]]) -> dict[str,Any]:
    """Make exactly one aggregate 40-cell R2-versus-R1 decision."""
    key_fields=("product","origin","target","horizon")
    r1_keys={tuple(r[f] for f in key_fields) for r in r1_records}
    r2_keys={tuple(r[f] for f in key_fields) for r in r2_records}
    if r1_keys!=r2_keys: raise ValueError("R2 global gate requires the same exact development keys as R1")
    r1_metrics=_metrics(r1_records,("R1",)); r2_metrics=_metrics(r2_records,("R2",))
    complete=bool(r1_metrics["complete40Cells"] and r2_metrics["complete40Cells"])
    mape_gain=bool(complete and r2_metrics["metrics"]["R2Mape"] <
                   r1_metrics["metrics"]["R1Mape"]-SCORE_MAPE_TOL_PP)
    mae_nonworse=bool(complete and r2_metrics["metrics"]["R2Mae"] <=
                      r1_metrics["metrics"]["R1Mae"]+SCORE_MAE_TOL)
    return {"r1Metrics":r1_metrics,"r2Metrics":r2_metrics,"complete40Cells":complete,
            "mapeImprovementBeyondTolerance":mape_gain,
            "maeNonworseningWithinTolerance":mae_nonworse,
            "selectedStructure":"R2" if mape_gain and mae_nonworse else "R1",
            "scope":"one combined R2 candidate, one global 40-cell comparison; no post-score crop subset search"}


def _metrics(records: list[dict[str,Any]], models: tuple[str,...]) -> dict[str,Any]:
    cells=[]
    for p in PRODUCTS:
        for h in HORIZONS:
            cell=[r for r in records if r["product"]==p and r["horizon"]==h]
            result={"product":p,"horizon":h,"n":len(cell)}
            for model in ("B",)+models:
                available=bool(cell) and all(isinstance(r.get(model),(int,float)) and math.isfinite(r[model]) for r in cell)
                if available:
                    result[model+"Mae"]=math.fsum(abs(r[model]-r["actual"]) for r in cell)/len(cell)
                    result[model+"Mape"]=100*math.fsum(abs(r[model]-r["actual"])/r["actual"] for r in cell)/len(cell)
                else: result[model+"Mae"]=result[model+"Mape"]=None
            cells.append(result)
    complete=all(x["n"]>0 for x in cells)
    metrics={}
    for model in ("B",)+models:
        available=complete and all(c[model+"Mae"] is not None and c[model+"Mape"] is not None for c in cells)
        metrics[model+"Mae"]=math.fsum(c[model+"Mae"] for c in cells)/40 if available else None
        metrics[model+"Mape"]=math.fsum(c[model+"Mape"] for c in cells)/40 if available else None
    return {"cells":cells,"complete40Cells":complete,"pairedRows":sum(c["n"] for c in cells),"metrics":metrics}


def _contributions(records: list[dict[str,Any]], models: tuple[str,...]) -> dict[str,Any]:
    counts=defaultdict(int)
    for r in records: counts[(r["product"],r["horizon"])]+=1
    groups=defaultdict(lambda:defaultdict(float)); sign_groups=defaultdict(lambda:defaultdict(float))
    for r in records:
        p,h=r["product"],r["horizon"]; w=1/(40*counts[(p,h)])
        month=r["target"][:7]; sign="positive" if r["delta"]>0 else "negative" if r["delta"]<0 else "zero"
        for model in models:
            loss=abs(r[model]-r["actual"])-abs(r["B"]-r["actual"])
            ape=100*loss/r["actual"]
            monthly=groups[(model,p,h,month)]; signed=sign_groups[(model,p,sign)]
            monthly["n"]+=1; signed["n"]+=1
            for family,key,val in ((monthly,"mae",loss*w),(monthly,"mape",ape*w),
                                   (signed,"mae",loss*w),(signed,"mape",ape*w)):
                family[key+"Net"]+=val
                family[key+("Gain" if val<0 else "Loss")]+=abs(val)
    paired_metrics=_metrics(records,models)
    net_by_model={}
    for model in models:
        mae=sum(v["maeNet"] for k,v in groups.items() if k[0]==model)
        mape=sum(v["mapeNet"] for k,v in groups.items() if k[0]==model)
        expected_mae=(paired_metrics["metrics"][model+"Mae"]-paired_metrics["metrics"]["BMae"]
                      if paired_metrics["complete40Cells"] else None)
        expected_mape=(paired_metrics["metrics"][model+"Mape"]-paired_metrics["metrics"]["BMape"]
                       if paired_metrics["complete40Cells"] else None)
        net_by_model[model]={"maeNet":mae,"mapeNet":mape,"expectedMaeDifference":expected_mae,
                             "expectedMapeDifference":expected_mape,
                             "reconciles":(expected_mae is not None and expected_mape is not None and
                                            math.isclose(mae,expected_mae,rel_tol=1e-9,abs_tol=1e-9) and
                                            math.isclose(mape,expected_mape,rel_tol=1e-9,abs_tol=1e-9))}
    return {"productHorizonMonth":[{"model":k[0],"product":k[1],"horizon":k[2],"month":k[3],**v} for k,v in sorted(groups.items())],
            "byDeltaSign":[{"model":k[0],"product":k[1],"deltaSign":k[2],**v} for k,v in sorted(sign_groups.items())],
            "netReconciliation":net_by_model}


def _loss_class(actual: float, baseline: float, prediction: float) -> str:
    if prediction==baseline: return "exact_B_unchanged"
    needed=actual-baseline; shift=prediction-baseline
    if needed==0: return "changed_exact_B"
    if shift*needed<0: return "opposite_direction"
    if prediction==actual: return "hit_actual"
    if abs(shift)>abs(needed): return "overshot_actual"
    return "toward_actual"


def _residual_loss_diagnosis(records: list[dict[str,Any]], models: tuple[str,...]) -> dict[str,Any]:
    counts=defaultdict(lambda:defaultdict(float)); cell_n=defaultdict(int)
    for r in records: cell_n[(r["product"],r["horizon"])]+=1
    for r in records:
        key=(r["product"],r["horizon"]); w=1/(40*cell_n[key])
        for model in models:
            classification=_loss_class(r["actual"],r["B"],r[model])
            g=counts[(model,classification)]; g["n"]+=1
            mae=abs(r[model]-r["actual"])-abs(r["B"]-r["actual"])
            mape=100*mae/r["actual"]
            g["maeNet"]+=mae*w; g["mapeNet"]+=mape*w
            g["maeGain"]+=max(-mae,0)*w; g["maeLoss"]+=max(mae,0)*w
            g["mapeGain"]+=max(-mape,0)*w; g["mapeLoss"]+=max(mape,0)*w
    return {"tolerance":"classifications use exact realized predictions; losses are unrounded",
            "byPolicyAndResidualClass":[{"policy":k[0],"residualClass":k[1],**v} for k,v in sorted(counts.items())]}


def _action_summary(rows: list[dict[str,Any]], paired_keys: set[tuple]) -> dict[str,Any]:
    summaries={}
    policies=sorted({p for r in rows for p in r.get("policies",{})})
    for policy in policies:
        full=defaultdict(int); paired=defaultdict(int); full_sign=defaultdict(int); pair_sign=defaultdict(int)
        for r in rows:
            obj=r.get("policies",{}).get(policy)
            if not isinstance(obj,Mapping): continue
            if obj.get("status")=="not_fitted": continue
            b=_number(r["B"],"B",positive=True); pred=_number(obj["prediction"],policy+" prediction",positive=True)
            delta=_number(r["delta"],"delta"); shift=pred-b
            sign="positive" if delta>0 else "negative" if delta<0 else "zero"
            key=_key(r); is_pair=key in paired_keys
            full_sign[sign]+=1
            if is_pair: pair_sign[sign]+=1
            for counts in (full,)+( (paired,) if is_pair else () ):
                counts["issuedRows"]+=1
                counts["requestedNonzeroDeltaRows"]+=int(delta!=0)
                counts["realizedNonzeroShiftRows"]+=int(shift!=0)
                counts["aboveReplayToleranceShiftRows"]+=int(not math.isclose(pred,b,rel_tol=1e-12,abs_tol=1e-12))
                reason=obj.get("fallbackReason")
                if reason: counts["fallback:"+str(reason)]+=1
                raw=obj.get("rawPrediction")
                if isinstance(raw,str) and raw in ("nonfinite","raw_overflow"): counts["rawNonfiniteRows"]+=1
                counts["exactBUnchangedRows"]+=int(pred==b and not reason)
        n=full["issuedRows"]; pn=paired["issuedRows"]
        summaries[policy]={"fullForecastGrid":{"denominator":n,"counts":dict(sorted(full.items())),
                                                  "deltaSignCounts":dict(sorted(full_sign.items())),
                                                  "realizedInterventionRate":full["realizedNonzeroShiftRows"]/n if n else None},
                           "positiveActualPairedGrid":{"denominator":pn,"counts":dict(sorted(paired.items())),
                                                        "deltaSignCounts":dict(sorted(pair_sign.items())),
                                                        "realizedInterventionRate":paired["realizedNonzeroShiftRows"]/pn if pn else None}}
    return summaries


def _final_price_decision(metrics: Mapping[str,Any], *, all_alpha_zero: bool,
                          baseline_identical: bool) -> tuple[bool,str]:
    m=metrics["metrics"]
    passed=bool(metrics.get("complete40Cells") and not all_alpha_zero and not baseline_identical and
                m["FINALMape"]<m["BMape"]-SCORE_MAPE_TOL_PP and
                m["FINALMae"]<=m["BMae"]+SCORE_MAE_TOL)
    if all_alpha_zero or baseline_identical: return False,"completed_baseline_identical_no_effective_calibration"
    return passed,"completed_exploratory_gain" if passed else "completed_no_incremental_gain"


def _stage_diagnostic_rows(indexed: Mapping[tuple,dict[str,Any]], actuals: Mapping[tuple[str,str],float],
                           start: dt.date, end: dt.date, origin_start: dt.date,
                           r1_coeffs: Mapping[str,float]) -> list[dict[str,Any]]:
    rows=[]
    for k,src in sorted(indexed.items()):
        p,o,t,h=k; day=dt.date.fromisoformat(t)
        if not (start<=day<=end and dt.date.fromisoformat(o)>=origin_start): continue
        r1=_policy_prediction(src,r1_coeffs,structure="R1")
        rows.append({"product":p,"origin":o,"target":t,"horizon":h,"B":src["_b"],"delta":src["_delta"],
                     "actual":actuals.get((p,t)),"policies":{
                         "K1":{"prediction":src["_k1"],"rawPrediction":src.get("kRaw",src["_k1"]),
                               "fallbackReason":src.get("fallbackReason")},
                         "R1":{"prediction":r1["prediction"],"rawPrediction":r1["rawPrediction"],
                               "fallbackReason":r1["fallbackReason"]}}})
    return rows


def _sign_support(records: list[dict[str,Any]], product: str, prediction_field: str) -> dict[str,Any]:
    result={}
    for sign in ("positive","negative"):
        group=[r for r in records if r["product"]==product and ((r["delta"]>0)==(sign=="positive")) and r["delta"]!=0]
        cells=[]
        for h in HORIZONS:
            c=[r for r in group if r["horizon"]==h]
            if c:
                cells.append({"horizon":h,"n":len(c),"maeDifference":math.fsum(abs(r[prediction_field]-r["actual"])-abs(r["B"]-r["actual"]) for r in c)/len(c),
                              "mapeDifferencePctPoints":100*math.fsum((abs(r[prediction_field]-r["actual"])-abs(r["B"]-r["actual"]))/r["actual"] for r in c)/len(c)})
        support=len({r["target"] for r in group})
        result[sign]={"uniqueTargetDays":support,"rows":len(group),"supportedHorizons":len(cells),
                      "maeDifference":math.fsum(c["maeDifference"] for c in cells)/len(cells) if cells else None,
                      "mapeDifferencePctPoints":math.fsum(c["mapeDifferencePctPoints"] for c in cells)/len(cells) if cells else None,
                      "byHorizon":cells}
    return result


def _opposed_sign_effects(support: Mapping[str,Any]) -> bool:
    pos=support["positive"]; neg=support["negative"]
    if min(pos["uniqueTargetDays"],neg["uniqueTargetDays"])<SIGN_SUPPORT_DAYS: return False
    tol_m= SCORE_MAE_TOL; tol_p=SCORE_MAPE_TOL_PP
    pos_improves=pos["maeDifference"] < -tol_m and pos["mapeDifferencePctPoints"] < -tol_p
    pos_worsens=pos["maeDifference"] > tol_m and pos["mapeDifferencePctPoints"] > tol_p
    neg_improves=neg["maeDifference"] < -tol_m and neg["mapeDifferencePctPoints"] < -tol_p
    neg_worsens=neg["maeDifference"] > tol_m and neg["mapeDifferencePctPoints"] > tol_p
    return (pos_improves and neg_worsens) or (neg_improves and pos_worsens)


def calibrate(v_forecast_path: str|Path|None=None, prices_path: str|Path|None=None,
              output: str|Path|None=None) -> dict[str,Any]:
    """Use C and June development only, select R1/R2, then refit once on all V."""
    source,indexed,source_sha=_load_k1("V",v_forecast_path)
    actuals,price_sha=_limited_actuals(prices_path or "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json",DEV_END)
    c=_join(indexed,actuals,C_START,C_END,PERIODS["V"][2])
    dev=_join(indexed,actuals,DEV_START,DEV_END,DEV_ORIGIN_START)
    full=_join(indexed,actuals,PERIODS["V"][0],PERIODS["V"][1],PERIODS["V"][2])
    if any(not rows for rows in (c,dev,full)): raise ValueError("V calibration periods have no eligible positive targets")
    r1fit={p:_fit_lp([r for r in c if r["product"]==p],fit_name="R1_C_"+p) for p in CALIBRATED_PRODUCTS}
    r1={p:v["alphas"]["all"] for p,v in r1fit.items()}
    r1_dev=_apply(dev,r1,"R1")
    dev_metrics_r1=_metrics(r1_dev,("R1",))
    sign_support={p:_sign_support(r1_dev,p,"R1") for p in CALIBRATED_PRODUCTS}
    trigger={p:_opposed_sign_effects(sign_support[p]) for p in CALIBRATED_PRODUCTS}
    c_sign_support={p:{s:len({r["target"] for r in c if r["product"]==p and r["delta"]!=0 and ((r["delta"]>0)==(s=="positive"))}) for s in ("positive","negative")} for p in CALIBRATED_PRODUCTS}
    r2fit={}; r2_coeff={}
    for p in CALIBRATED_PRODUCTS:
        if trigger[p] and min(c_sign_support[p].values())>=SIGN_SUPPORT_DAYS:
            r2fit[p]=_fit_lp([r for r in c if r["product"]==p],signs=True,fit_name="R2_C_"+p)
            r2_coeff[p]=r2fit[p]["alphas"]
    if r2_coeff:
        r2_dev=_combined_r2_candidate(dev,r2_coeff,r1)
        gate=_global_r2_gate(r1_dev,r2_dev)
        dev_metrics_r2=gate["r2Metrics"]
        r2selected=gate["selectedStructure"]=="R2"
    else:
        r2_dev=[]; dev_metrics_r2=None; r2selected=False
        gate={"r1Metrics":dev_metrics_r1,"r2Metrics":None,"complete40Cells":False,
              "mapeImprovementBeyondTolerance":False,"maeNonworseningWithinTolerance":False,
              "selectedStructure":"R1","scope":"no crop met the pre-score R2 eligibility rule"}
    chosen=gate["selectedStructure"]
    final_fit={}; final_coeff={}
    if chosen=="R1":
        for p in CALIBRATED_PRODUCTS:
            final_fit[p]=_fit_lp([r for r in full if r["product"]==p],fit_name="R1_fullV_"+p)
            final_coeff[p]=final_fit[p]["alphas"]["all"]
    else:
        for p in CALIBRATED_PRODUCTS:
            if p in r2_coeff:
                final_fit[p]=_fit_lp([r for r in full if r["product"]==p],signs=True,fit_name="R2_fullV_"+p)
                final_coeff[p]=final_fit[p]["alphas"]
            else:
                final_fit[p]=_fit_lp([r for r in full if r["product"]==p],fit_name="R1_fullV_"+p)
                final_coeff[p]=final_fit[p]["alphas"]["all"]
    final_v=_apply(full,final_coeff,chosen) if chosen=="R1" else _apply(full,{**{p:v for p,v in final_coeff.items() if isinstance(v,Mapping)},"r1Fallback":{p:v for p,v in final_coeff.items() if not isinstance(v,Mapping)}},"R2")
    full_metrics=_metrics(final_v,(chosen,))
    c_scored=_apply(c,r1,"R1"); dev_scored=r1_dev
    records={"C":_metrics(c_scored,("K1","R1")),"dev_R1":_metrics(dev_scored,("K1","R1")),
             "dev_R2":dev_metrics_r2,"fullVRefitTraining":full_metrics}
    c_issued=_stage_diagnostic_rows(indexed,actuals,C_START,C_END,PERIODS["V"][2],r1)
    dev_issued=_stage_diagnostic_rows(indexed,actuals,DEV_START,DEV_END,DEV_ORIGIN_START,r1)
    def paired_diagnostic_rows(stage_rows):
        return [{**r,"K1":r["policies"]["K1"]["prediction"],"R1":r["policies"]["R1"]["prediction"]}
                for r in stage_rows if r["actual"] is not None]
    c_pairs=paired_diagnostic_rows(c_issued); dev_pairs=paired_diagnostic_rows(dev_issued)
    payload={"schemaVersion":1,"experiment":"ANCHORED_INCREMENT_REPAIR_V2","stage":"calibration_frozen",
             "protocolSha256":PROTOCOL_SHA256,"targetSnapshotSha256":price_sha,
             "source":{"path":K1_PATHS["V"],"fileSha256":source_sha,"manifestSha256":source["manifestSha256"],
                       "targetValuesAttached":False},
             "targetUse":{"calibrationIntervals":{"C":[C_START.isoformat(),C_END.isoformat()],
                                                    "dev":[DEV_START.isoformat(),DEV_END.isoformat()]},
                          "VLabelReadThrough":DEV_END.isoformat(),"TLabelsUsed":False,
                          "explanation":"The source snapshot physically contains later dates. The calibration path exposes only positive V targets through June 20 to the fitting and selection functions; no T target enters any fit, gate, or coefficient choice."},
             "calibratorSha256":_sha(Path(__file__).read_bytes()),
             "solver":{"name":"scipy.optimize.linprog","scipyVersion":__import__("scipy").__version__,
                       "method":SOLVER_METHOD,"options":SOLVER_OPTIONS,
                       "lexicographicTieBreak":"second LP minimizes sum(abs(alpha)) on primary objective <= optimum + recorded tolerance",
                       "objectiveFaceToleranceFraction":OBJECTIVE_FACE_TOL,"fitCountExcludesTieBreakPasses":True,
                       "logicalFitCount":sum(x["logicalFitCount"] for x in list(r1fit.values())+list(r2fit.values())+list(final_fit.values()))},
             "R1_CFits":r1fit,"R1_CAlpha":r1,"R1_CMetrics":records["C"],
             "development":{"R1Metrics":dev_metrics_r1,"signSupportAndEffects":sign_support,
                            "R2DevTriggerByProduct":trigger,"R2_CSignUniqueTargets":c_sign_support,
                            "R2Fits":r2fit,"R2DevelopmentMetrics":dev_metrics_r2,
                            "R2vsR1Gate":{**{k:v for k,v in gate.items() if k not in ("r1Metrics","r2Metrics")},
                                          "interpretation":"June is development/selection data, not an independent validation."}},
             "finalRefit":{"selectedStructure":chosen,"coefficients":final_coeff,"fitRecords":final_fit,
                           "trainingPerformanceFullV":full_metrics,
                           "fitTargetThrough":PERIODS["V"][1].isoformat(),"TLabelsUsed":False},
             "fitRecords":records,"behaviorDiagnostics":{
                 "C":{"actions":_action_summary(c_issued,{_key(r) for r in c_pairs}),
                      "weightedResidualLoss":_residual_loss_diagnosis(c_pairs,("K1","R1"))},
                 "JuneDevelopment":{"actions":_action_summary(dev_issued,{_key(r) for r in dev_pairs}),
                                    "weightedResidualLoss":_residual_loss_diagnosis(dev_pairs,("K1","R1"))}},
             "contributions":{"C_K1_R1":_contributions(c_scored,("K1","R1")),
                                                      "dev_K1_R1":_contributions(dev_scored,("K1","R1")),
                                                      "dev_R2":_contributions(r2_dev,("R2",)) if r2_dev else None,
                                                      "fullVRefitTraining":_contributions(final_v,(chosen,))},
             "logicalFitCount":sum(x["logicalFitCount"] for x in list(r1fit.values())+list(r2fit.values())+list(final_fit.values())),
             "maximumLogicalFits":6,"noBaseModelFitOrInference":True}
    if payload["logicalFitCount"]>6: raise RuntimeError("calibration exceeded the protocol fit budget")
    payload["calibrationSha256"]=_sha(_canonical(payload))
    if output is not None: _write_once(_path(output),_canonical(payload))
    return payload


def _row_view(row: Mapping[str,Any], r1: Mapping[str,Any], r2: Mapping[str,Any]|None,
              final: Mapping[str,Any], *, r2_status: str) -> dict[str,Any]:
    b=_number(row.get("baselineB",row.get("B")),"B",positive=True)
    k1=_number(row.get("prediction",row.get("K1")),"K1",positive=True)
    result={"product":row["product"],"origin":row["origin"],"target":row["target"],"horizon":row["horizon"],
            "B":{"prediction":b,"status":"issued"},
            "K1":{"prediction":k1,"status":"issued","rawPrediction":row.get("kRaw",k1),
                  "fallbackReason":row.get("fallbackReason")},
            "R1":r1,"R2":r2 if r2 is not None else {"status":"not_fitted","prediction":None},
            "R2Status":r2_status,"final":final,"delta":row["delta"]}
    return result


def generate_t(calibration_path: str|Path, calibration_sha256: str,
               output: str|Path|None=None, v_forecast_path: str|Path|None=None,
               t_forecast_path: str|Path|None=None) -> dict[str,Any]:
    """Build answer-free C/dev/full-V/T forecast views from frozen coefficients."""
    cp=_path(calibration_path); raw=cp.read_bytes(); ch=_sha(raw)
    if ch!=calibration_sha256: raise ValueError("caller-pinned calibration file hash mismatch")
    cal=_read(cp); _document_sha(cal,"calibrationSha256")
    if cal.get("stage")!="calibration_frozen" or cal.get("experiment")!="ANCHORED_INCREMENT_REPAIR_V2":
        raise ValueError("calibration is not a frozen V2 calibration artifact")
    if cal.get("targetUse",{}).get("TLabelsUsed") is not False: raise ValueError("calibration claims T label use")
    vdoc,vidx,vsha=_load_k1("V",v_forecast_path); tdoc,tidx,tsha=_load_k1("T",t_forecast_path)
    r1_dev=cal["R1_CAlpha"]; r2_dev=cal["development"]["R2Fits"]
    r2_dev_alpha={p:fit["alphas"] for p,fit in r2_dev.items()}
    dev_structure="R2" if cal["development"]["R2vsR1Gate"]["selectedStructure"]=="R2" else "R1"
    final_coeff=cal["finalRefit"]["coefficients"]; final_structure=cal["finalRefit"]["selectedStructure"]
    r1_t=[]; r2_t=[]; final_t=[]; crows=[]; devrows=[]; vfull=[]
    for idx,period_name,source_doc,source_file_sha in ((vidx,"V",vdoc,vsha),(tidx,"T",tdoc,tsha)):
        start,end,origin_start=PERIODS[period_name]
        for k,row in sorted(idx.items()):
            p,o,t,h=k; day=dt.date.fromisoformat(t); origin=dt.date.fromisoformat(o)
            if period_name=="V":
                role="C" if C_START<=day<=C_END else "dev" if origin>=DEV_ORIGIN_START and DEV_START<=day<=DEV_END else "otherV"
            else: role="T"
            r1pred=_policy_prediction(row,r1_dev,structure="R1")
            r1obj={"prediction":r1pred["prediction"],"rawPrediction":r1pred["rawPrediction"],"alpha":r1pred["alpha"],"fallbackReason":r1pred["fallbackReason"]}
            r2obj=None
            if r2_dev_alpha:
                r2pred=_policy_prediction(row,{**r2_dev_alpha,"r1Fallback":r1_dev},structure="R2")
                r2obj={"prediction":r2pred["prediction"],"rawPrediction":r2pred["rawPrediction"],"alpha":r2pred["alpha"],"fallbackReason":r2pred["fallbackReason"]}
            if final_structure=="R1": fpred=_policy_prediction(row,final_coeff,structure="R1")
            else:
                f_alpha={p:v for p,v in final_coeff.items() if isinstance(v,Mapping)}
                f_fallback={p:v for p,v in final_coeff.items() if not isinstance(v,Mapping)}
                fpred=_policy_prediction(row,{**f_alpha,"r1Fallback":f_fallback},structure="R2")
            fobj={"prediction":fpred["prediction"],"rawPrediction":fpred["rawPrediction"],"alpha":fpred["alpha"],"fallbackReason":fpred["fallbackReason"]}
            out=_row_view(row,r1obj,r2obj,fobj,r2_status="development candidate" if r2obj else "not fitted")
            source_row=row.get("_sourceRow")
            if not isinstance(source_row,Mapping) or _key(source_row)!=(p,o,t,h):
                raise ValueError("source row provenance is missing or does not match the issued key")
            source_row_sha=_sha(_canonical(source_row))
            out["sourceProvenance"]={"sourcePhase":period_name,"sourceForecastFileSha256":source_file_sha,
                                     "sourceForecastManifestSha256":source_doc["manifestSha256"],
                                     "sourceRowSha256":source_row_sha,"sourceKey":{"product":p,"origin":o,"target":t,"horizon":h},
                                     "targetSnapshotSha256":PRICE_SHA256,
                                     "informationBoundary":{"forecastOrigin":o,"requiredAvailableThrough":
                                         (origin-dt.timedelta(days=1)).isoformat()}}
            if period_name=="V":
                # This is an overlapping stage view: every eligible V key is part of the
                # full-V refit-training predictions, and C/dev receive their own labels.
                vfull.append(out)
                if role=="C": crows.append(out)
                elif role=="dev": devrows.append(out)
            else: final_t.append(out)
    if len(final_t)!=len(tidx): raise ValueError("T source forecast grid lost keys during answer-free generation")
    # T variants are directly materialized into the final answer-free view; no T targets are loaded.
    payload={"schemaVersion":1,"experiment":"ANCHORED_INCREMENT_REPAIR_V2","phase":"V-and-T-answer-free",
             "targetValuesAttached":False,"protocolSha256":PROTOCOL_SHA256,
             "calibration":{"path":str(cp.relative_to(ROOT)),"fileSha256":ch,"calibrationSha256":cal["calibrationSha256"]},
             "targetSnapshotSha256":PRICE_SHA256,"sourceArtifacts":{"K1_V":vsha,"K1_T":tsha},
             "sourceManifestHashes":{"K1_V":vdoc["manifestSha256"],"K1_T":tdoc["manifestSha256"]},
             "generatorSha256":_sha(Path(__file__).read_bytes()),"finalStructure":final_structure,
             "phaseRows":{"C_training":{"count":len(crows),"forecasts":crows},
                          "June_development":{"count":len(devrows),"forecasts":devrows},
                          "fullV_refit_training":{"count":len(vfull),"forecasts":vfull},
                          "T_final_evaluation":{"count":len(final_t),"forecasts":final_t}}
             }
    # Main T rows include the required unchanged B, K1 and calibrated variants.
    payload["forecastKeyCount"]={k:v["count"] for k,v in payload["phaseRows"].items()}
    payload["forecastsSha256"]=_sha(_canonical(payload["phaseRows"]["T_final_evaluation"]["forecasts"]))
    payload["manifestSha256"]=_sha(_canonical(payload))
    if output is not None: _write_once(_path(output),_canonical(payload))
    return payload


def score_t(forecasts_path: str|Path, forecasts_sha256: str, calibration_path: str|Path,
            calibration_sha256: str, prices_path: str|Path, output: str|Path|None=None) -> dict[str,Any]:
    fp=_path(forecasts_path); raw=fp.read_bytes(); actual_forecast_sha=_sha(raw)
    if actual_forecast_sha!=forecasts_sha256: raise ValueError("caller-pinned forecasts file hash mismatch")
    doc=_read(fp); _document_sha(doc,"manifestSha256")
    if doc.get("experiment")!="ANCHORED_INCREMENT_REPAIR_V2" or doc.get("targetValuesAttached") is not False:
        raise ValueError("expected frozen answer-free V2 forecasts")
    cpath=_path(calibration_path); cbytes=cpath.read_bytes(); cfile_sha=_sha(cbytes)
    if cfile_sha!=calibration_sha256 or cfile_sha!=doc.get("calibration",{}).get("fileSha256"):
        raise ValueError("caller-pinned calibration hash mismatch")
    calibration=_read(cpath); _document_sha(calibration,"calibrationSha256")
    if calibration.get("calibrationSha256")!=doc.get("calibration",{}).get("calibrationSha256"):
        raise ValueError("forecast artifact is not bound to this calibration")
    # Validate the old frozen K1 report without recomputing its already-reported inferential results.
    k1score_path=_path(K1_SCORE_PATH); k1score_raw=k1score_path.read_bytes()
    if _sha(k1score_raw)!=K1_SCORE_FILE_SHA256: raise ValueError("pinned K1 T score report hash mismatch")
    k1score=_read(k1score_path); _document_sha(k1score,"reportSha256")
    if (k1score.get("forecastFileSha256")!=doc.get("sourceArtifacts",{}).get("K1_T")
            or k1score.get("targetSnapshotSha256")!=PRICE_SHA256):
        raise ValueError("saved K1 statistics are not bound to the frozen K1 T forecast/target source")
    target_path=_path(prices_path); target_bytes=target_path.read_bytes()
    if _sha(target_bytes)!=PRICE_SHA256 or doc.get("targetSnapshotSha256")!=PRICE_SHA256:
        raise ValueError("pinned T price target snapshot hash mismatch")
    actuals,_=_limited_actuals(prices_path,PERIODS["T"][1])
    rows=doc.get("phaseRows",{}).get("T_final_evaluation",{}).get("forecasts")
    if not isinstance(rows,list) or _sha(_canonical(rows))!=doc.get("forecastsSha256"):
        raise ValueError("T answer-free forecast row digest mismatch")
    indexed={}
    for row in rows:
        k=_key(row)
        if k in indexed: raise ValueError("duplicate final T key")
        indexed[k]=row
    if set(indexed)!=_expected_keys("T"): raise ValueError("T final forecast key grid mismatch")
    joined=[]
    for k,row in sorted(indexed.items()):
        actual=actuals.get((k[0],k[2]))
        if actual is None: continue
        pairs={}
        for name in ("B","K1","R1","R2","final"):
            obj=row.get(name)
            if isinstance(obj,Mapping) and obj.get("status")=="not_fitted": continue
            if not isinstance(obj,Mapping) or obj.get("status","issued")!="issued": raise ValueError(f"missing issued prediction {name}")
            pairs[name]=_number(obj.get("prediction"),f"{name} prediction",positive=True)
        if "R2" not in pairs: pairs["R2"]=None
        joined.append({"product":k[0],"origin":k[1],"target":k[2],"horizon":k[3],"actual":actual,
                       "delta":_number(row.get("delta"),"delta"),"FINAL":pairs["final"],**pairs})
    if len(joined)!=2470: raise ValueError(f"T positive-target coverage expected 2470, found {len(joined)}")
    # K1 point estimates are taken from the completed V1 report, already scored on these exact keys.
    old_metrics=k1score.get("metrics40Cell",{})
    if old_metrics.get("pairedRows")!=len(joined) or not old_metrics.get("complete40Cells"):
        raise ValueError("saved K1 T statistics do not have the expected full common key coverage")
    metrics=_metrics([{"product":r["product"],"horizon":r["horizon"],"actual":r["actual"],"B":r["B"],"K1":r["K1"],"R1":r["R1"],"R2":r["R2"],"FINAL":r["final"]} for r in joined],("K1","R1","R2","FINAL"))
    # Exact same target keys and cells across policies; R2 may be absent when its branch did not fit.
    comparisons={}
    for name in ("K1","R1","R2","FINAL"):
        vals=[r for r in metrics["cells"] if r["n"]>0]
        if name=="R2" and all(r["R2Mae"] is None for r in vals): continue
        comparisons[name]={"mae":metrics["metrics"][name+"Mae"],"mape":metrics["metrics"][name+"Mape"]}
    if not math.isclose(metrics["metrics"]["BMae"],old_metrics["baselineMae"],rel_tol=1e-10,abs_tol=1e-10) or not math.isclose(metrics["metrics"]["BMape"],old_metrics["baselineMape"],rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError("rejoined T baseline does not match existing K1 score report")
    if not math.isclose(metrics["metrics"]["K1Mae"],old_metrics["mae"],rel_tol=1e-10,abs_tol=1e-10) or not math.isclose(metrics["metrics"]["K1Mape"],old_metrics["mape"],rel_tol=1e-10,abs_tol=1e-10):
        raise ValueError("K1 T point metrics differ from immutable prior score (K1 evidence not reused)")
    infer_pairs=[{"product":r["product"],"horizon":r["horizon"],"origin":r["origin"],"target":r["target"],"actual":r["actual"],"a":r["final"],"b":r["B"]} for r in joined]
    dm=inference._dm(infer_pairs,PRODUCTS,"FINAL_vs_B")
    contrib=inference._contributions(infer_pairs)
    bootstrap=[]
    for metric in ("ae","ape"):
        for block in (14,28):
            bootstrap.append({"metric":metric,"blockDays":block,**diagnostics.circular_block_bootstrap(
                infer_pairs,a="a",b="b",metric=metric,block=block,replicates=2000,seed=0,
                calendar_start=dt.date(2026,7,11),calendar_end=dt.date(2026,9,27))})
    # Existing K1 DM/Holm/bootstrap remains linked by its report hash.
    f=metrics["metrics"]
    t_issued=[{"product":r["product"],"origin":r["origin"],"target":r["target"],"horizon":r["horizon"],
                "B":r["B"]["prediction"],"delta":r["delta"],
                "policies":{name:r.get(name) for name in ("K1","R1","R2","final")}}
               for r in rows]
    paired_keys={_key(r) for r in joined}
    loss_diagnosis=_residual_loss_diagnosis(joined,("K1","R1","FINAL"))
    action_diagnosis=_action_summary(t_issued,paired_keys)
    all_final_alpha_zero=all(_number(r["final"].get("alpha"),"final alpha")==0.0 for r in rows)
    baseline_identical=all(r["final"]["prediction"]==r["B"]["prediction"] for r in rows)
    price_gate,status=_final_price_decision(metrics,all_alpha_zero=all_final_alpha_zero,
                                            baseline_identical=baseline_identical)
    result={"schemaVersion":1,"experiment":"ANCHORED_INCREMENT_REPAIR_V2","phase":"T_final",
            "forecastManifestSha256":doc["manifestSha256"],"forecastsFileSha256":actual_forecast_sha,
            "calibrationFileSha256":cfile_sha,"calibrationSha256":calibration["calibrationSha256"],
            "targetSnapshotSha256":PRICE_SHA256,"positiveActualPairs":len(joined),"complete40Cells":metrics["complete40Cells"],
            "policyMetrics":{**comparisons,"B":{"mae":f["BMae"],"mape":f["BMape"]}},
            "savedK1Evidence":{"path":K1_SCORE_PATH,"fileSha256":K1_SCORE_FILE_SHA256,
                               "reportSha256":k1score["reportSha256"],"metrics40Cell":old_metrics,
                               "inference":k1score.get("inference"),"interpretation":"reused from completed V1; not recomputed"},
            "finalVsB":{"metrics40Cell":metrics,"dmHAC9and28":dm,"holmFamilies":"existing scorer; one 80-test family per bandwidth",
                        "jointCalendarBlockBootstrap":bootstrap,"contributions":contrib,
                        "priceGatePassed":price_gate,"coveragePreserved":True,
                        "meaningfulImprovementTolerance":{"mapePercentagePoints":SCORE_MAPE_TOL_PP,"maeCurrencyPerKg":SCORE_MAE_TOL},
                        "allFinalAlphaZero":all_final_alpha_zero,"baselineIdentical":baseline_identical},
            "behaviorDiagnostics":{"actionsFullIssuedAndPositivePaired":action_diagnosis,
                                    "weightedResidualLossK1R1Final":loss_diagnosis},
            "calibrationDevelopment":{"selectedStructure":calibration["finalRefit"]["selectedStructure"],
                                      "devR1":calibration["development"]["R1Metrics"],
                                      "devR2":calibration["development"]["R2DevelopmentMetrics"],
                                      "fitCount":calibration["logicalFitCount"]},
            "inferenceLimitations":"T was repeatedly inspected before this after-the-fact design; this is not independent confirmation. No T label influenced the frozen coefficients.",
            "status":status}
    result["reportSha256"]=_sha(_canonical(result))
    if output is not None: _write_once(_path(output),_canonical(result))
    return result


def main(argv: list[str]|None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="command",required=True)
    c=sub.add_parser("calibrate"); c.add_argument("--v-forecasts",default=K1_PATHS["V"]); c.add_argument("--prices",default="outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"); c.add_argument("--output",required=True)
    g=sub.add_parser("generate-t"); g.add_argument("--calibration",required=True); g.add_argument("--calibration-sha256",required=True); g.add_argument("--v-forecasts",default=K1_PATHS["V"]); g.add_argument("--t-forecasts",default=K1_PATHS["T"]); g.add_argument("--output",required=True)
    s=sub.add_parser("score-t"); s.add_argument("--forecasts",required=True); s.add_argument("--forecasts-sha256",required=True); s.add_argument("--calibration",required=True); s.add_argument("--calibration-sha256",required=True); s.add_argument("--prices",required=True); s.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    if args.command=="calibrate": result=calibrate(args.v_forecasts,args.prices,args.output); digest=result["calibrationSha256"]
    elif args.command=="generate-t": result=generate_t(args.calibration,args.calibration_sha256,args.output,args.v_forecasts,args.t_forecasts); digest=result["manifestSha256"]
    else: result=score_t(args.forecasts,args.forecasts_sha256,args.calibration,args.calibration_sha256,args.prices,args.output); digest=result["reportSha256"]
    print(json.dumps({"command":args.command,"artifactSha256":digest,"output":args.output},ensure_ascii=False))
    return 0


if __name__=="__main__": raise SystemExit(main())
