"""Immutable implementation of the retrospective E106-FHS experiment.

The source uses the frozen RHS issue grid, reconstructs its training rows, and
fits only the new functional-sharing (F) joint ridge system. B/C predictions
and all other crops are replayed from saved artifacts. This module is not run
against real data until the full source and synthetic suite have Astra CLEAR.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from . import radish_horizon_sharing as rhs

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/experiments/radish-functional-sharing-v1-20261012"
SOURCE_OUT=rhs.OUT
SOURCE_PHASES={"V":("validation-manifest.json","validation-predictions.csv","validation-report.json","validation-terminal.json","validation-claim.json"),
               "T":("test-manifest.json","test-predictions.csv","test-report.json","test-terminal.json","test-claim.json")}
CONDITIONS=("B","C","F","A")
PHASE_GRID={"V":3240,"T":3160}
PHASE_ISSUED={"V":755,"T":790}
PHASE_PAIRED={"V":651,"T":650}
ALPHA=10.0
LAMBDA_MAX_BRACKET=64
LAMBDA_MAX_BISECTIONS=120
PENALTY_PSD_FACTOR=64.0
A_ZERO_FACTOR=64.0
PRICE_ABS_TOL=1e-6
PRICE_REL_TOL=1e-8
SOLVE_REL_TOL=1e-8
EDF_ABS_FACTOR=1e-6
LOG_REPLAY_TOL=1e-11
TRAIN_A_RMS_FACTOR=64.0
CHAIN=np.zeros((10,10),dtype=np.float64)
for _i in range(9):
    CHAIN[_i,_i]+=1; CHAIN[_i+1,_i+1]+=1
    CHAIN[_i,_i+1]-=1; CHAIN[_i+1,_i]-=1


def canonical(value:Any)->bytes:
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode("utf-8")


def sha256_bytes(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def _json(path:Path)->Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_exclusive(path:Path,data:bytes)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,"wb") as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())


def _write_derived(path:Path,data:bytes)->None:
    """Create deterministic derived artifacts once; resume only if bytes match."""
    if path.exists():
        if path.read_bytes()!=data: raise ValueError(f"existing derived artifact differs: {path}")
        return
    _write_exclusive(path,data)


def _source_paths()->dict[str,Path]:
    from . import equal_info_online_forecasts as base
    from . import radish_functional_sharing_score as scorer
    paths={"rhs":Path(rhs.__file__),"featureForecasts":Path(base.__file__),"modeling":Path(base.modeling.__file__),
           "directHorizon":Path(base.direct_horizon.__file__),"functionalSharing":Path(__file__),"scorer":Path(scorer.__file__)}
    return paths


def _input_paths()->dict[str,Path]:
    from . import equal_info_online_forecasts as base
    paths={"rhsProtocol":SOURCE_OUT/"protocol.json","rhsVManifest":SOURCE_OUT/"validation-manifest.json",
           "rhsVForecasts":SOURCE_OUT/"validation-predictions.csv","rhsVReport":SOURCE_OUT/"validation-report.json",
           "rhsVFitDiagnostics":SOURCE_OUT/"fit-diagnostics.json","rhsVBaselineReproduction":SOURCE_OUT/"baseline-reproduction.json",
           "rhsTManifest":SOURCE_OUT/"test-manifest.json","rhsTForecasts":SOURCE_OUT/"test-predictions.csv",
           "rhsTReport":SOURCE_OUT/"test-report.json","rhsTFitDiagnostics":SOURCE_OUT/"test-fit-diagnostics.json",
           "rhsTBaselineReproduction":SOURCE_OUT/"test-baseline-reproduction.json",
           "baselineManifest":rhs.BASELINE_PATH,"prices":base.PRICES_PATH,"fixed50":base.FIXED50_PATH}
    for phase,month in (("V","04"),("V","05"),("V","06"),("T","07"),("T","08"),("T","09")):
        fit_id=f"radish-2026-{month}-01"; prefix=f"{phase.lower()}-{fit_id}"
        for suffix in (".json",".claim.json",".terminal.json"):
            name=f"fit-checkpoints/{prefix}{suffix}"
            paths[name]=SOURCE_OUT/name
    return paths


def _protocol_body()->dict[str,Any]:
    from . import equal_info_online_forecasts as base
    import scipy,sklearn,statsmodels
    from . import radish_functional_sharing_score as scorer
    paths=_input_paths(); missing=[name for name,path in paths.items() if not path.is_file()]
    if missing: raise FileNotFoundError(f"pinned RHS artifacts are incomplete: {missing}")
    # The original source experiment's read-only preflight pins its underlying data and helpers.
    for phase in ("V","T"): rhs._preflight(phase,SOURCE_OUT)
    return {"schemaVersion":1,"experimentId":"radish-functional-sharing-v1","candidateId":"E106-FHS",
            "preservedBaseline":"E106-RHS/C","phases":{"V":{"gridRows":PHASE_GRID["V"],"eligibleRadish":PHASE_ISSUED["V"],"pairedPositiveRadish":PHASE_PAIRED["V"]},
                       "T":{"gridRows":PHASE_GRID["T"],"eligibleRadish":PHASE_ISSUED["T"],"pairedPositiveRadish":PHASE_PAIRED["T"]}},
            "sourceProtocolSha256":_json(SOURCE_OUT/"protocol.json")["protocolSha256"],
            "inputsSha256":{name:file_sha(path) for name,path in paths.items()},
            "sourcesSha256":{name:file_sha(path) for name,path in _source_paths().items()},
            "runtime":{"python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,
                       "scikitLearn":sklearn.__version__,"statsmodels":statsmodels.__version__},
            "design":{"features":"existing 40-column order; horizonDays coefficient fixed zero; 39 slopes per horizon",
                      "objective":"sum_h ||y_h-X_h beta_h-b_h||^2 + 10 sum_h ||beta_h||^2 + lambda beta.T P_F beta",
                      "coordinate":"D_h=diag(sigma_ref/sigma_h), q_h=D_h beta_h, U_h=Xcenter_h D_h^-1",
                      "SigmaBar":"(1/10) sum_h U_h.T U_h/n_h, preserve each horizon's original rows",
                      "penalty":"P_F=Dcoord.T@(L_chain kron SigmaBar)@Dcoord; replace C graph penalty; all 39 directions; no added ridge or truncation",
                      "edfTarget":"recompute RHS C target from training X and verify saved C target; fit F to that exact target",
                      "intercepts":"ten independent, unpenalized intercepts; record their EDF=10 separately from slope EDF",
                      "edfTolerance":"1e-6*max(1,df_B); root via float64 Cholesky EDF, initial lambda ||Q0||2/||P_F||2, double bracket <=64, bisection <=120",
                      "penaltyEigenZeroRule":"tau=64*eps64*390*||P_F||2; use |eig|<=tau only to construct nullspace N for df_inf; preserve raw spectrum and use raw P_F in every finite-lambda solve; fail eig < -tau; never clip, shift, or truncate P_F",
                      "AZeroRule":"per h s=max(1,max|rB_train,h|,max|rC_train,h|), tau_h=64*eps64*39*s_h, E_C_zero_tol=mean_h(tau_h^2); A unavailable iff E_C<=threshold; no denominator epsilon or rho cap",
                      "linearSolveRelativeTolerance":SOLVE_REL_TOL,
                      "baselinePriceReplayTolerance":{"absolute":PRICE_ABS_TOL,"relative":PRICE_REL_TOL},
                      "AEqualRmsTolerance":"64*eps64*39*max(1,sqrt(E_C),sqrt(E_F)); compare RMS values",
                      "validationGate":"unrounded radish F-C equal-10-horizon blend MAPE<0, MAE<=0, and no added F coverage loss; A does not gate",
                      "fitBudget":{"FSystemsPerPhase":3,"maxFSystemsTotal":6,"B_C_D_PRefits":0,"SARIMARefits":0,"ExternalCollections":0}},
            "scoreProtocol":{"comparisons":["F-C","F-A","F-B"],"horizons":10,"losses":["AE","APE"],"familySize":60,
                             "HAC":[9,28],"HolmAlpha":0.05,"minimumPairs":30,"calendarBlocks":[14,28],"replicates":2000,"seed":0}}


def freeze_protocol(output_dir:Path=OUT)->dict[str,Any]:
    output_dir.mkdir(parents=True,exist_ok=True)
    path=output_dir/"protocol.json"; body=_protocol_body()
    doc={"protocol":body,"protocolSha256":sha256_bytes(canonical(body))}
    _write_exclusive(path,canonical(doc)); return doc


def _load_protocol(output_dir:Path=OUT)->dict[str,Any]:
    doc=_json(output_dir/"protocol.json")
    if doc.get("protocolSha256")!=sha256_bytes(canonical(doc.get("protocol"))): raise ValueError("FHS protocol digest mismatch")
    if doc.get("protocol")!=_protocol_body(): raise ValueError("FHS source or input differs from frozen protocol")
    return doc


def _array(value:Any,name:str,ndim:int|None=None)->np.ndarray:
    out=np.asarray(value,dtype=np.float64)
    if (ndim is not None and out.ndim!=ndim) or not np.isfinite(out).all(): raise ValueError(f"{name} must be finite {ndim or 'numeric'}-D data")
    return out


def _edf(G:np.ndarray,Q0:np.ndarray,P:np.ndarray,lam:float)->float:
    Q=(Q0+lam*P);Q=(Q+Q.T)*0.5
    factor=cho_factor(Q,lower=True,check_finite=True)
    solved=cho_solve(factor,G,check_finite=True)
    value=float(np.trace(solved))
    if not math.isfinite(value): raise ArithmeticError("nonfinite EDF")
    return value


def _edf_floor(P:np.ndarray,G:np.ndarray,Q0:np.ndarray)->dict[str,Any]:
    P=(P+P.T)*0.5; values,vectors=np.linalg.eigh(P); norm=float(np.linalg.norm(P,2))
    tau=PENALTY_PSD_FACTOR*np.finfo(np.float64).eps*P.shape[0]*norm
    if not math.isfinite(norm): raise ArithmeticError("functional penalty has nonfinite spectral scale")
    if float(values[0]) < -tau: raise ArithmeticError(f"functional penalty has material negative eigenvalue {values[0]} below {-tau}")
    mask=np.abs(values)<=tau; N=vectors[:,mask]
    if N.size:
        orth=float(np.max(np.abs(N.T@N-np.eye(N.shape[1]))))
        pn=float(np.linalg.norm(P@N,2)/max(norm, np.finfo(float).tiny))
        null_residual_tolerance=max(10*tau/norm,2e-12) if norm>0 else 2e-12
        if orth>2e-12 or pn>null_residual_tolerance: raise ArithmeticError("functional penalty nullspace basis failed verification")
        A=N.T@G@N; B=N.T@Q0@N; factor=cho_factor((B+B.T)*0.5,lower=True,check_finite=True)
        lower=float(np.trace(cho_solve(factor,A,check_finite=True)))
    else:
        orth=0.0;pn=0.0;lower=0.0
    return {"dfInfinity":lower,"penaltySpectralNorm":norm,"numericZeroTolerance":tau,
            "rawEigenvalues":values.tolist(),"rawMinimumEigenvalue":float(values[0]),"rawMaximumEigenvalue":float(values[-1]),
            "nullity":int(N.shape[1]),"negativeRoundoffCount":int(np.sum((values<0)&(values>=-tau))),
            "nullspaceOrthogonalityMaxError":orth,"relativePenaltyNullResidual":pn}


def _solve_f(Xstd_by_h:Sequence[np.ndarray],y_by_h:Sequence[np.ndarray],fit:Mapping[str,Any],*,phase:str)->dict[str,Any]:
    if len(Xstd_by_h)!=10 or len(y_by_h)!=10: raise ValueError("F requires ten original horizon supports")
    means=_array(fit["scalerXMean"],"saved centering means",2); scales=_array(fit["scalerScale"],"saved scales",2)
    if means.shape!=(10,39) or scales.shape!=(10,39) or np.any(scales<=0): raise ValueError("saved FHS scaler state must be 10x39 with positive scales")
    centered=[]; ymeans=[]; ycentered=[]
    for hi,(xraw,yraw) in enumerate(zip(Xstd_by_h,y_by_h)):
        x=_array(xraw,f"standardized X h={hi+1}",2)[:,list(rhs.ACTIVE)]
        y=_array(yraw,f"target h={hi+1}",1)
        if x.shape!=(len(y),39) or len(y)<90: raise ValueError(f"invalid original training support h={hi+1}")
        if not np.allclose(x.mean(0),means[hi],rtol=0,atol=2e-12): raise ValueError(f"saved centering mean mismatch h={hi+1}")
        ym=float(math.fsum(y.tolist())/len(y))
        if abs(ym-float(fit["scalerYMean"][hi]))>1e-12: raise ValueError(f"saved target mean mismatch h={hi+1}")
        centered.append(x-means[hi]);ymeans.append(ym);ycentered.append(y-ym)
    ref=scales[0]; Dblocks=[np.diag(ref/scales[h]) for h in range(10)]
    U=[centered[h]*(scales[h]/ref) for h in range(10)]
    sigma=sum((u.T@u)/len(u) for u in U)/10.0
    sigma=(sigma+sigma.T)*0.5
    G=np.zeros((390,390),dtype=np.float64); b=np.zeros(390,dtype=np.float64)
    for h,(x,y) in enumerate(zip(centered,ycentered)):
        sl=slice(39*h,39*(h+1));G[sl,sl]=x.T@x;b[sl]=x.T@y
    Q0=G+ALPHA*np.eye(390,dtype=np.float64)
    dcoord=np.zeros_like(G)
    for h,D in enumerate(Dblocks): dcoord[39*h:39*(h+1),39*h:39*(h+1)]=D
    penalty_metric=np.kron(CHAIN,sigma)
    P=dcoord.T@penalty_metric@dcoord;P=(P+P.T)*0.5
    c_pen=rhs.graph_penalties(scales)["C"]
    saved_c=float(fit["edf"]["dfTarget"])
    c_lambda=float(fit["lambdas"]["C"])
    c_edf=_edf(G,Q0,c_pen,c_lambda)
    saved_c_actual=float(fit["diagnostics"]["C"]["actualEdf"])
    df0=_edf(G,Q0,np.zeros_like(G),0.0)
    c_shared_limit=rhs._shared_limit(G,Q0,scales)
    recomputed_target=c_shared_limit+rhs.EDF_FRACTION*(df0-c_shared_limit)
    edf_tolerance=EDF_ABS_FACTOR*max(1.0,df0)
    if (abs(saved_c_actual-c_edf)>edf_tolerance or abs(saved_c-recomputed_target)>edf_tolerance or
            abs(float(fit["edf"]["df0"])-df0)>edf_tolerance or abs(float(fit["edf"]["dfInf"])-c_shared_limit)>edf_tolerance):
        raise ArithmeticError("recomputed original C EDF differs from the saved monthly target")
    floor=_edf_floor(P,G,Q0);df_inf=floor["dfInfinity"]
    if not (math.isfinite(df_inf) and -edf_tolerance<=df_inf<=df0+edf_tolerance): raise ArithmeticError("F EDF lower bound outside [0,df_B]")
    if saved_c<df_inf-edf_tolerance or saved_c>df0+edf_tolerance or saved_c<=df_inf:
        return {"status":"edf_target_unreachable","fitId":fit["fitId"],"reason":"saved_C_target_outside_F_reachable_edf_interval",
                "dfB":df0,"dfInfinityF":df_inf,"dfTargetC":saved_c,"edfTolerance":edf_tolerance,"penaltyAudit":floor}
    # Monotone root in lambda; raw P is always used, even for roundoff eigenvalues.
    norm_q=float(np.linalg.norm(Q0,2));norm_p=floor["penaltySpectralNorm"]
    lo=0.0;hi=norm_q/norm_p
    if not math.isfinite(hi) or hi<=0: raise ArithmeticError("invalid initial lambda bracket")
    df_lo=df0;df_hi=_edf(G,Q0,P,hi);doublings=0
    while df_hi>saved_c and doublings<LAMBDA_MAX_BRACKET:
        hi*=2.0;doublings+=1
        if not math.isfinite(hi): break
        df_hi=_edf(G,Q0,P,hi)
    if not math.isfinite(df_hi) or df_hi>saved_c+edf_tolerance:
        return {"status":"numerical_failure","fitId":fit["fitId"],"reason":"could_not_bracket_edf_target",
                "dfB":df0,"dfInfinityF":df_inf,"dfTargetC":saved_c,"edfAtUpperBracket":df_hi,"bracketDoublings":doublings}
    target_lam=None;df_actual=None;bisect=0
    while bisect<LAMBDA_MAX_BISECTIONS:
        mid=lo+(hi-lo)*0.5; value=_edf(G,Q0,P,mid)
        if value>df_lo+edf_tolerance or value<df_hi-edf_tolerance: raise ArithmeticError("F EDF solver is not monotone in its bracket")
        target_lam=mid;df_actual=value
        if abs(value-saved_c)<=edf_tolerance: break
        if value>saved_c: lo=mid;df_lo=value
        else: hi=mid;df_hi=value
        bisect+=1
    if target_lam is None or df_actual is None or abs(df_actual-saved_c)>edf_tolerance or target_lam<=0 or not df_actual<df0:
        return {"status":"numerical_failure","fitId":fit["fitId"],"reason":"edf_root_did_not_converge",
                "dfB":df0,"dfInfinityF":df_inf,"dfTargetC":saved_c,"dfActualF":df_actual,"lambda":target_lam}
    Q=Q0+target_lam*P;Q=(Q+Q.T)*0.5;factor=cho_factor(Q,lower=True,check_finite=True)
    beta_flat=cho_solve(factor,b,check_finite=True)
    residual=Q@beta_flat-b;denom=float(np.linalg.norm(Q,2)*np.linalg.norm(beta_flat)+np.linalg.norm(b))
    rel_res=float(np.linalg.norm(residual)/denom) if denom else float(np.linalg.norm(residual))
    if not math.isfinite(rel_res) or rel_res>SOLVE_REL_TOL: raise ArithmeticError(f"F linear solve residual {rel_res} exceeds {SOLVE_REL_TOL}")
    beta=beta_flat.reshape(10,39); intercept=[]; training=[]
    for h in range(10):
        intercept.append(float(ymeans[h]-means[h]@beta[h]))
        training.append((centered[h]@beta[h]+ymeans[h]).tolist())
    return {"status":"ok","fitId":fit["fitId"],"phase":phase,"lambda":float(target_lam),"dfB":df0,"dfIntercepts":10.0,
            "dfInfinityF":df_inf,"dfTargetC":saved_c,"dfActualF":df_actual,"edfTolerance":edf_tolerance,
            "dfActualFTotalIncludingIntercepts":df_actual+10.0,
            "bracketDoublings":doublings,"bisections":bisect,"relativeSolveResidual":rel_res,
            "scalerXMean":means.tolist(),"scalerYMean":ymeans,"scalerScale":scales.tolist(),
            "fitCoefficients":beta.tolist(),"fitIntercepts":intercept,"trainingPredictions":{"F":training},
            "penaltyDiagnostics":{"SigmaBar":sigma.tolist(),"SigmaBarEigenvalues":np.linalg.eigvalsh(sigma).tolist(),
                                  "PFSpectralNorm":norm_p,"PFSymmetryMaxError":float(np.max(np.abs(P-P.T))),
                                  **floor},"trainingCounts":[len(y) for y in y_by_h]}


def _a_audit(fit:Mapping[str,Any],Xstd_by_h:Sequence[np.ndarray],y_by_h:Sequence[np.ndarray],f_beta:np.ndarray)->dict[str,Any]:
    means=np.asarray(fit["scalerXMean"],dtype=np.float64); ymeans=np.asarray(fit["scalerYMean"],dtype=np.float64)
    beta_b=np.asarray(fit["fitCoefficients"]["B"],dtype=np.float64);beta_c=np.asarray(fit["fitCoefficients"]["C"],dtype=np.float64)
    rb=[];rc=[];rf=[];svals=[];taus=[]
    for h in range(10):
        x=np.asarray(Xstd_by_h[h])[:,list(rhs.ACTIVE)]-means[h]
        rb_h=x@beta_b[h]+ymeans[h];rc_h=x@beta_c[h]+ymeans[h];rf_h=x@f_beta[h]+ymeans[h]
        scale=max(1.0,float(np.max(np.abs(rb_h))),float(np.max(np.abs(rc_h))))
        tau=A_ZERO_FACTOR*np.finfo(np.float64).eps*39*scale
        rb.append(rb_h);rc.append(rc_h);rf.append(rf_h);svals.append(scale);taus.append(tau)
    ec=math.fsum(float(np.mean((rc[h]-rb[h])**2)) for h in range(10))/10.0
    ef=math.fsum(float(np.mean((rf[h]-rb[h])**2)) for h in range(10))/10.0
    tol=math.fsum(t*t for t in taus)/10.0
    if not (math.isfinite(ec) and math.isfinite(ef) and ec>=0 and ef>=0): return {"status":"numerical_failure","reason":"nonfinite_or_negative_training_change_energy","E_C":ec,"E_F":ef,"E_C_zeroTolerance":tol}
    if ec<=tol:
        return {"status":"unsupported","reason":"training_C_change_numerically_zero","E_C":ec,"E_F":ef,
                "E_C_zeroTolerance":tol,"perHorizonScale":svals,"perHorizonZeroTolerance":taus}
    rho=math.sqrt(ef/ec)
    if not math.isfinite(rho): return {"status":"numerical_failure","reason":"nonfinite_rho","E_C":ec,"E_F":ef,"rho":rho}
    ra=[rb[h]+rho*(rc[h]-rb[h]) for h in range(10)]
    ea=math.fsum(float(np.mean((ra[h]-rb[h])**2)) for h in range(10))/10.0
    rms_tol=TRAIN_A_RMS_FACTOR*np.finfo(np.float64).eps*39*max(1.0,math.sqrt(ec),math.sqrt(ef))
    rms_delta=abs(math.sqrt(ea)-math.sqrt(ef))
    if not math.isfinite(ea) or rms_delta>rms_tol:
        return {"status":"numerical_failure","reason":"A_training_rms_does_not_match_F","E_C":ec,"E_F":ef,"E_A":ea,"rho":rho,"rmsTolerance":rms_tol,"rmsDifference":rms_delta}
    return {"status":"ok","E_C":ec,"E_F":ef,"E_A":ea,"rho":rho,"E_C_zeroTolerance":tol,
            "perHorizonScale":svals,"perHorizonZeroTolerance":taus,"trainingARmsTolerance":rms_tol,
            "trainingRms":{"B":math.sqrt(math.fsum(float(np.mean(rb[h]**2)) for h in range(10))/10),
                           "C_minus_B":math.sqrt(ec),"F_minus_B":math.sqrt(ef),"A_minus_B":math.sqrt(ea)},
            "trainingVectors":{"B":rb,"C":rc,"F":rf,"A":ra}}


def _score_training_records(fit:Mapping[str,Any],ys:Sequence[np.ndarray],targets:Sequence[Sequence[dt.date]],
                            f_fit:Mapping[str,Any],a_audit:Mapping[str,Any])->list[dict[str,Any]]:
    out=[]; pred_b=fit["trainingPredictions"]["B"];pred_c=fit["trainingPredictions"]["C"];pred_f=f_fit["trainingPredictions"]["F"]
    for h in range(10):
        a_values=a_audit.get("trainingVectors",{}).get("A",[None]*10)[h] if a_audit.get("status")=="ok" else None
        rows=[]
        for i,target in enumerate(targets[h]):
            rows.append({"targetDate":target.isoformat(),"actualLogReturn":float(ys[h][i]),"B":float(pred_b[h][i]),
                         "C":float(pred_c[h][i]),"F":float(pred_f[h][i]),"A":(float(a_values[i]) if a_values is not None else None)})
        audit={k:v for k,v in a_audit.items() if k!="trainingVectors"}
        out.append({"fitId":fit["fitId"],"fitOrigin":fit["fitId"].removeprefix("radish-"),"horizon":h+1,
                    "trainingCount":len(rows),"trainingRows":rows,**audit})
    return out


def _phase_files(phase:str)->tuple[str,str,str,str]:
    if phase not in SOURCE_PHASES: raise ValueError("phase must be V or T")
    m,c,r,t,_=SOURCE_PHASES[phase]
    return m,c,r,t


def _eligible_fit_origins(source_by:Mapping[Any,Any],phase:str)->list[dt.date]:
    cutoff=rhs.PHASES[phase][2]
    return sorted({dt.date.fromisoformat(key[1]).replace(day=1) for key in source_by
                   if key[0]=="radish" and dt.date.fromisoformat(key[1])>=cutoff})


def _phase_claim(output_dir:Path,phase:str,protocol_sha:str)->str:
    """Create or safely resume the unique phase claim after preflight."""
    path=output_dir/("validation-claim.json" if phase=="V" else "test-claim.json")
    if path.exists():
        prior=_json(path)
        if prior.get("phase")!=phase or prior.get("protocolSha256")!=protocol_sha or prior.get("status")!="running":
            raise ValueError("existing FHS phase claim cannot be resumed under this protocol")
        attempt=prior.get("attemptId")
        if not isinstance(attempt,str) or not attempt: raise ValueError("existing phase claim lacks attempt ID")
        return attempt
    attempt=str(uuid.uuid4())
    _write_exclusive(path,canonical({"schemaVersion":1,"phase":phase,"status":"running",
                                     "attemptId":attempt,"protocolSha256":protocol_sha}))
    return attempt


def _record_fit_failure(output_dir:Path,phase:str,fit_id:str,attempt_id:str,protocol_sha:str,
                        failure:Mapping[str,Any])->None:
    failure_path=output_dir/("validation-fit-failure.json" if phase=="V" else "test-fit-failure.json")
    terminal_path=output_dir/_phase_files(phase)[3]
    doc={"schemaVersion":1,"phase":phase,"fitId":fit_id,**dict(failure),"protocolSha256":protocol_sha}
    _write_exclusive(failure_path,canonical(doc))
    _write_exclusive(terminal_path,canonical({"phase":phase,"status":doc.get("status","numerical_failure"),
        "attemptId":attempt_id,"protocolSha256":protocol_sha,"failureSha256":file_sha(failure_path)}))


def _load_rhs_phase(phase:str,protocol:Mapping[str,Any])->tuple[dict[str,Any],dict[tuple[str,str,str,int],Any],dict[str,dict[str,Any]]]:
    """Load the immutable RHS phase and check its saved byte-level links."""
    manifest_name,csv_name,report_name,_=_phase_files(phase)
    manifest_path=SOURCE_OUT/manifest_name; csv_path=SOURCE_OUT/csv_name
    if not manifest_path.is_file() or not csv_path.is_file(): raise FileNotFoundError(f"missing pinned RHS {phase} forecast artifacts")
    doc=_json(manifest_path)
    rhs_protocol=_json(SOURCE_OUT/"protocol.json")
    if (doc.get("phase")!=phase or doc.get("protocolSha256")!=rhs_protocol.get("protocolSha256") or
            doc.get("inputSha256")!=rhs_protocol.get("protocol",{}).get("inputSha256")):
        raise ValueError("RHS manifest phase/protocol/input binding mismatch")
    if file_sha(csv_path)!=doc.get("forecastFileSha256"):
        raise ValueError("saved RHS forecast CSV digest mismatch")
    forecasts=doc.get("forecasts")
    if not isinstance(forecasts,list): raise ValueError("saved RHS manifest has no forecast grid")
    by={}
    for row in forecasts:
        key=(row.get("product"),row.get("origin"),row.get("target"),row.get("horizon"))
        if key in by: raise ValueError(f"duplicate RHS forecast key: {key}")
        by[key]=row
    if len(by)!=PHASE_GRID[phase]: raise ValueError("RHS forecast grid size differs from frozen phase")
    fits=doc.get("jointFitDiagnostics")
    if not isinstance(fits,list) or len(fits)!=3: raise ValueError("RHS must contain exactly three saved monthly fit states")
    fitmap={f.get("fitId"):f for f in fits}
    if len(fitmap)!=3: raise ValueError("duplicate RHS monthly fit IDs")
    return doc,by,fitmap


def _rhs_baseline_ref(row:Mapping[str,Any])->dict[str,Any]:
    models=row.get("models",{})
    direct=models.get("causalDirect",{}); blend=models.get("causal50",{})
    status=("fallback" if blend.get("provenance",{}).get("fallback") else "issued") if blend.get("status")=="issued" else "not_issued"
    return {"status":status,
            "prediction":blend.get("prediction"),"directPrediction":direct.get("prediction"),
            "sarimaPrediction":models.get("causalSarima",{}).get("prediction")}


def _condition_from_source(item:Mapping[str,Any])->dict[str,Any]:
    return {"status":item.get("status","not_issued"),
            "directLogPrediction":item.get("directLogPrediction"),
            "directPrediction":item.get("directPrediction"),"blendPrediction":item.get("blendPrediction"),
            "fitId":item.get("fitId"),"cutoff":item.get("cutoff"),
            "p0":item.get("p0"),"p0Date":item.get("p0Date"),
            "sarimaPrediction":item.get("sarimaPrediction"),
            "fallbackReason":item.get("fallbackReason")}


def _condition_skeleton(source_conditions:Mapping[str,Any],product:str,eligible:bool)->dict[str,dict[str,Any]]:
    """Create exactly B/C/F/A keys while preserving source controls and crop compositions."""
    candidate=source_conditions.get("C",source_conditions.get("B",{}))
    copied=_condition_from_source(candidate)
    if product=="radish" and eligible:
        return {"B":_condition_from_source(source_conditions["B"]),
                "C":_condition_from_source(source_conditions["C"])}
    if product=="radish":
        baseline=_condition_from_source(source_conditions["B"])
        comparison=_condition_from_source(source_conditions["C"])
    else:
        baseline=dict(copied);comparison=dict(copied)
    return {"B":baseline,"C":comparison,"F":dict(copied),"A":dict(copied)}


def _forecast_csv(rows:Sequence[Mapping[str,Any]])->bytes:
    import io
    buffer=io.StringIO(newline="")
    fields=("product","eligible","originDate","targetDate","horizon","fitId","cutoff","p0","p0Date","sarimaPrediction","condition","status","directLogPrediction","directPrediction","blendPrediction","fallbackReason")
    writer=csv.DictWriter(buffer,fieldnames=fields,lineterminator="\n");writer.writeheader()
    for row in rows:
        for condition in CONDITIONS:
            item=row["conditions"][condition]
            writer.writerow({"product":row["product"],"eligible":row["eligible"],"originDate":row["originDate"],"targetDate":row["targetDate"],"horizon":row["horizon"],"fitId":row.get("fitId"),"cutoff":row.get("cutoff"),"p0":row.get("p0"),"p0Date":row.get("p0Date"),"sarimaPrediction":row.get("sarimaPrediction"),"condition":condition,**item})
    return buffer.getvalue().encode("utf-8")


def _baseline_rows_for_scorer(phase:str)->dict[tuple[str,str,str,int],dict[str,Any]]:
    baseline=_json(rhs.BASELINE_PATH); raw=rhs._baseline_index(baseline,phase)
    return {k:_rhs_baseline_ref(v) for k,v in raw.items()}


def _prices()->dict[str,Any]:
    from . import equal_info_online_forecasts as base
    return _json(base.PRICES_PATH)


def _fit_f_bundle(phase:str,fit_id:str,source_fit:Mapping[str,Any],xstd:Sequence[np.ndarray],
                  y:Sequence[np.ndarray],targets:Sequence[Sequence[dt.date]],
                  protocol:Mapping[str,Any])->dict[str,Any]:
    try:
        return _fit_f_bundle_unchecked(phase,fit_id,source_fit,xstd,y,targets,protocol)
    except (ArithmeticError,np.linalg.LinAlgError,OverflowError) as exc:
        # Returning a terminal bundle lets _fit_checkpoint persist the failed attempt and reason.
        return {"status":"numerical_failure","fitId":fit_id,
                "f":{"status":"numerical_failure","fitId":fit_id,
                     "reason":f"{type(exc).__name__}: {exc}"}}


def _fit_f_bundle_unchecked(phase:str,fit_id:str,source_fit:Mapping[str,Any],xstd:Sequence[np.ndarray],
                            y:Sequence[np.ndarray],targets:Sequence[Sequence[dt.date]],
                            protocol:Mapping[str,Any])->dict[str,Any]:
    f_fit=_solve_f(xstd,y,source_fit,phase=phase)
    if f_fit.get("status")!="ok": return {"status":f_fit.get("status"),"f":f_fit,"fitId":fit_id}
    a_audit=_a_audit(source_fit,xstd,y,np.asarray(f_fit["fitCoefficients"],dtype=np.float64))
    f_fit["aAudit"]={k:v for k,v in a_audit.items() if k!="trainingVectors"}
    f_fit["trainingFirstTargetByHorizon"]=[v[0].isoformat() for v in targets]
    f_fit["trainingLastTargetByHorizon"]=[v[-1].isoformat() for v in targets]
    f_fit["trainingRows"]=[]
    for h in range(10):
        for i,target in enumerate(targets[h]):
            f_fit["trainingRows"].append({"horizon":h+1,"targetDate":target.isoformat(),"actualLogReturn":float(y[h][i]),
                "B":float(source_fit["trainingPredictions"]["B"][h][i]),"C":float(source_fit["trainingPredictions"]["C"][h][i]),
                "F":float(f_fit["trainingPredictions"]["F"][h][i]),
                "A":(float(a_audit["trainingVectors"]["A"][h][i]) if a_audit.get("status")=="ok" else None)})
    f_fit["trainingDiagnostics"]=_score_training_records(source_fit,y,targets,f_fit,a_audit)
    f_fit["sourceRhsFitSha256"]=sha256_bytes(canonical(source_fit))
    f_fit["protocolSha256"]=protocol["protocolSha256"]
    return {"status":"ok","fitId":fit_id,"f":f_fit,"a":{k:v for k,v in a_audit.items() if k!="trainingVectors"}}


def _f_fit_hashes(source_fit:Mapping[str,Any],protocol:Mapping[str,Any])->dict[str,str]:
    return {"sourceRhsFit":sha256_bytes(canonical(source_fit)),
            "sourceTrainingMatrix":str(source_fit.get("inputHashes",{}).get("trainingMatrix","")),
            "sourceTrainingTarget":str(source_fit.get("inputHashes",{}).get("trainingTarget","")),
            "protocolSha256":protocol["protocolSha256"],
            "functionalSourceSha256":file_sha(Path(__file__))}


def _verify_f_fit_replay(fit:Mapping[str,Any],source_fit:Mapping[str,Any],
                         xstd:Sequence[np.ndarray],y_by_h:Sequence[np.ndarray])->dict[str,Any]:
    """Replay a saved F coefficient state without repeating the EDF root solve."""
    means=np.asarray(fit["scalerXMean"],dtype=np.float64); ymeans=np.asarray(fit["scalerYMean"],dtype=np.float64)
    scales=np.asarray(fit["scalerScale"],dtype=np.float64)
    if means.shape!=(10,39) or scales.shape!=(10,39) or ymeans.shape!=(10,): raise ValueError("saved F scaler state shape mismatch")
    centered=[]; yc=[]
    for h,(x,y) in enumerate(zip(xstd,y_by_h)):
        xa=np.asarray(x,dtype=np.float64)[:,list(rhs.ACTIVE)];ya=np.asarray(y,dtype=np.float64)
        if not np.allclose(xa.mean(0),means[h],rtol=0,atol=2e-12) or abs(float(ya.mean())-ymeans[h])>1e-12:
            raise ValueError(f"saved F centering state differs from reconstructed support h={h+1}")
        centered.append(xa-means[h]);yc.append(ya-ymeans[h])
    G=np.zeros((390,390));b=np.zeros(390)
    for h,(x,y) in enumerate(zip(centered,yc)):
        sl=slice(39*h,39*(h+1));G[sl,sl]=x.T@x;b[sl]=x.T@y
    Q0=G+ALPHA*np.eye(390)
    D=np.zeros_like(Q0);ref=scales[0]
    for h in range(10): D[39*h:39*(h+1),39*h:39*(h+1)]=np.diag(ref/scales[h])
    U=[centered[h]*(scales[h]/ref) for h in range(10)]
    sigma=sum(u.T@u/len(u) for u in U)/10;sigma=(sigma+sigma.T)*0.5
    P=D.T@np.kron(CHAIN,sigma)@D;P=(P+P.T)*0.5
    lam=float(fit["lambda"]);beta=np.asarray(fit["fitCoefficients"],dtype=np.float64).reshape(390)
    q=Q0+lam*P;q=(q+q.T)*0.5;res=q@beta-b
    denom=float(np.linalg.norm(q,2)*np.linalg.norm(beta)+np.linalg.norm(b))
    rel=float(np.linalg.norm(res)/denom) if denom else float(np.linalg.norm(res))
    if not math.isfinite(rel) or rel>SOLVE_REL_TOL: raise ValueError("saved F coefficients fail normal-equation replay")
    train=fit.get("trainingPredictions",{}).get("F",[])
    if len(train)!=10: raise ValueError("saved F training prediction grid mismatch")
    for h,x in enumerate(centered):
        pred=x@np.asarray(fit["fitCoefficients"][h])+ymeans[h]
        if len(pred)!=len(train[h]) or not np.allclose(pred,train[h],rtol=0,atol=LOG_REPLAY_TOL):
            raise ValueError(f"saved F training predictions fail coefficient replay h={h+1}")
    edf=_edf(G,Q0,P,lam);target=float(source_fit["edf"]["dfTarget"])
    tol=EDF_ABS_FACTOR*max(1.0,_edf(G,Q0,np.zeros_like(G),0.0))
    if abs(edf-target)>tol or abs(float(fit["dfActualF"])-edf)>tol or abs(float(fit["dfTargetC"])-target)>tol:
        raise ValueError("saved F EDF does not match the original C target")
    return {"relativeSolveResidual":rel,"dfActualF":edf,"edfTolerance":tol}


def _build_phase(phase:str,output_dir:Path)->dict[str,Any]:
    from . import equal_info_online_forecasts as base
    if phase not in SOURCE_PHASES: raise ValueError("phase must be V or T")
    protocol=_load_protocol(output_dir)
    manifest_name,csv_name,report_name,terminal_name=_phase_files(phase)
    manifest_path=output_dir/manifest_name; csv_path=output_dir/csv_name
    if manifest_path.exists():
        existing=verify_phase(phase,output_dir,allow_missing_terminal=True)
        terminal_path=output_dir/terminal_name
        if not terminal_path.exists():
            claim_path=output_dir/("validation-claim.json" if phase=="V" else "test-claim.json")
            claim=_json(claim_path)
            _write_exclusive(terminal_path,canonical({"phase":phase,"status":"complete_answer_free",
                "attemptId":claim["attemptId"],"protocolSha256":protocol["protocolSha256"],"manifestSha256":file_sha(manifest_path),
                "recoveredFromVerifiedManifest":True}))
            verify_phase(phase,output_dir)
        return existing
    if (output_dir/terminal_name).exists(): raise ValueError("phase is terminal but its manifest is absent")
    v_report=score_phase("V",output_dir) if phase=="T" else None
    if phase=="T" and not bool(v_report.get("vGateMetrics",{}).get("passed")):
        marker={"schemaVersion":1,"phase":"T","status":"not_run_validation_failed","reason":"V F-vs-C price/coverage gate failed","protocolSha256":protocol["protocolSha256"]}
        marker["vGateReportSha256"]=file_sha(output_dir/"validation-report.json")
        _write_derived(output_dir/"test-manifest.json",canonical(marker))
        _write_derived(output_dir/"test-report.json",canonical({**marker,"forecastRows":0,"jointFits":0}))
        _write_derived(output_dir/"test-terminal.json",canonical({"phase":"T","status":marker["status"],"protocolSha256":protocol["protocolSha256"],"manifestSha256":sha256_bytes(canonical(marker))}))
        return marker
    preflight=rhs._preflight(phase,SOURCE_OUT)
    # The saved control must still pass its original coefficient, input, and forecast replay.
    rhs.verify_phase(phase,SOURCE_OUT)
    source_doc,source_by,source_fits=_load_rhs_phase(phase,protocol)
    source_protocol=rhs._load_protocol(SOURCE_OUT)
    raw_prices=_prices(); direct_series=base.direct_horizon._observations(raw_prices)
    attempt_id=_phase_claim(output_dir,phase,protocol["protocolSha256"])
    origin_dates=_eligible_fit_origins(source_by,phase)
    if len(origin_dates)!=3: raise ValueError("eligible RHS forecast origins must map to three monthly fits")
    fit_docs=[]; fit_map={}
    for fit_origin in origin_dates:
        fit_id=f"radish-{fit_origin.isoformat()}"; source_fit=source_fits.get(fit_id)
        if source_fit is None: raise ValueError(f"missing saved RHS fit {fit_id}")
        _,xraw,y,targets=rhs._reconstruct_training_fold(fit_origin,raw_prices,base)
        xstd=rhs._verify_fit_state(source_fit,phase,source_protocol,xraw,y,targets)
        input_hashes=_f_fit_hashes(source_fit,protocol)
        try:
            bundle=rhs._fit_checkpoint(output_dir,phase,fit_id,input_hashes,
                lambda:_fit_f_bundle(phase,fit_id,source_fit,xstd,y,targets,protocol))
        except ValueError as exc:
            # A prior claim without a matching complete checkpoint is never silently refit.
            _record_fit_failure(output_dir,phase,fit_id,attempt_id,protocol["protocolSha256"],
                                {"status":"checkpoint_resume_refused","reason":str(exc)})
            raise
        if bundle.get("status")!="ok":
            f_fit=bundle.get("f",{})
            # This is an explicit computational reachability/numerical failure, not a scored model result.
            _record_fit_failure(output_dir,phase,fit_id,attempt_id,protocol["protocolSha256"],f_fit)
            raise ArithmeticError(f"{fit_id}: {f_fit.get('status')} ({f_fit.get('reason')})")
        f_fit=bundle["f"]
        a_audit=_a_audit(source_fit,xstd,y,np.asarray(f_fit["fitCoefficients"],dtype=np.float64))
        _verify_f_fit_replay(f_fit,source_fit,xstd,y)
        if canonical({k:v for k,v in a_audit.items() if k!="trainingVectors"})!=canonical(f_fit.get("aAudit")):
            raise ValueError(f"saved A training rule does not replay for {fit_id}")
        fit_map[fit_id]={"f":f_fit,"a":a_audit,"xstd":xstd,"targets":targets,"y":y,"source":source_fit,"origin":fit_origin}
        fit_docs.append(f_fit)
    forecasts=[]; baseline_reproduction=[]
    raw_baseline=rhs._baseline_index(_json(rhs.BASELINE_PATH),phase)
    for key in sorted(source_by,key=lambda k:(k[0],k[2],k[3])):
        old=source_by[key]; product,origin,target,h=key; eligible=bool(old.get("eligible"))
        conditions=_condition_skeleton(old["conditions"],product,eligible)
        # RHS saved B and C are the original E106 references; non-radish E106 composition is unchanged.
        if product=="radish" and eligible:
            fit_id=old["conditions"]["C"].get("fitId") or old["conditions"]["B"].get("fitId")
            if fit_id not in fit_map: raise ValueError(f"eligible radish key references absent monthly fit: {key}")
            state=fit_map[fit_id]; fit=state["f"]; a_audit=state["a"]
            feat,p0,audit=base._direct_feature(direct_series["radish"],dt.date.fromisoformat(target),h,{})
            if feat is None or p0 is None: raise ValueError(f"source feature unavailable for eligible key {key}")
            active=np.asarray(feat,dtype=np.float64)[list(rhs.ACTIVE)]
            source_fit=state["source"]
            raw_mean=np.asarray(source_fit["standardScalerMean"][h-1],dtype=np.float64)
            raw_scale=np.asarray(source_fit["standardScalerScale"][h-1],dtype=np.float64)
            z=(active-raw_mean)/raw_scale
            xmean=np.asarray(fit["scalerXMean"][h-1]); ymean=float(fit["scalerYMean"][h-1])
            beta=np.asarray(fit["fitCoefficients"][h-1])
            f_log=float((z-xmean)@beta+ymean)
            b_log=old["conditions"]["B"].get("directLogPrediction")
            c_log=old["conditions"]["C"].get("directLogPrediction")
            a_log=(float(b_log)+float(a_audit["rho"])*(float(c_log)-float(b_log)) if a_audit.get("status")=="ok" else None)
            sarima=old.get("sarimaPrediction",old["conditions"]["C"].get("sarimaPrediction"))
            status=old["conditions"]["C"]["status"]
            common={"fitId":fit_id,"cutoff":audit.get("cutoff"),"p0":float(p0),"p0Date":audit.get("p0Date"),"sarimaPrediction":sarima}
            for name,logpred in (("F",f_log),("A",a_log)):
                if logpred is None:
                    conditions[name]={**common,"status":"not_issued","directLogPrediction":None,"directPrediction":None,"blendPrediction":None,"fallbackReason":"A unsupported: "+str(a_audit.get("reason"))}
                else:
                    direct=float(p0*math.exp(logpred)); blend=0.5*(direct+float(sarima)) if sarima is not None else direct
                    conditions[name]={**common,"status":status,"directLogPrediction":logpred,"directPrediction":direct,"blendPrediction":blend,
                                      "fallbackReason":old["conditions"]["C"].get("fallbackReason")}
            # Restore only the requested B/C labels; no refit or rounding is allowed.
            conditions["B"]=_condition_from_source(old["conditions"]["B"])
            conditions["C"]=_condition_from_source(old["conditions"]["C"])
        # Full scorer contract includes common row-level source state.
        p0=old.get("p0")
        sarima=old.get("sarimaPrediction",conditions.get("C",{}).get("sarimaPrediction"))
        if product!="radish" or not eligible:
            template=conditions["C"]
            p0=template.get("p0",p0);sarima=template.get("sarimaPrediction",sarima)
            for name in CONDITIONS:
                conditions[name].setdefault("p0",p0);conditions[name].setdefault("sarimaPrediction",sarima)
        row={"product":product,"eligible":eligible,"originDate":origin,"targetDate":target,"horizon":int(h),
             "fitId":(conditions.get("F",{}).get("fitId") or conditions.get("C",{}).get("fitId")),
             "cutoff":conditions.get("F",{}).get("cutoff",conditions.get("C",{}).get("cutoff")),
             "p0":p0,"p0Date":old.get("p0Date"),"sarimaPrediction":sarima,"conditions":conditions}
        forecasts.append(row)
        if product=="radish" and eligible:
            e106=raw_baseline[key]
            for component,expected,actual in (("direct",e106["models"]["causalDirect"].get("prediction"),conditions["B"].get("directPrediction")),
                                              ("blend",e106["models"]["causal50"].get("prediction"),conditions["B"].get("blendPrediction"))):
                if expected is None or actual is None:
                    if expected is not actual: raise ValueError(f"saved B {component} availability differs from E106 for {key}")
                    delta=None
                else:
                    delta=abs(float(expected)-float(actual))
                    if delta>PRICE_ABS_TOL+PRICE_REL_TOL*abs(float(expected)): raise ValueError(f"saved B {component} fails E106 replay at {key}")
                baseline_reproduction.append({"key":list(key),"component":component,"expected":expected,"reproduced":actual,"absoluteDifference":delta})
    if len(forecasts)!=PHASE_GRID[phase]: raise ValueError("FHS phase grid row count mismatch")
    payload=_forecast_csv(forecasts);_write_derived(csv_path,payload)
    diag=[r for fit in fit_docs for r in fit["trainingDiagnostics"]]
    manifest={"schemaVersion":1,"phase":phase,"status":"answer_free_locked","protocolSha256":protocol["protocolSha256"],
              "inputSha256":protocol["protocol"]["inputsSha256"],"forecastFile":csv_name,"forecastFileSha256":sha256_bytes(payload),
              "fitDiagnostics":diag,"fitDiagnosticsSha256":sha256_bytes(canonical(diag)),"jointFitDiagnostics":fit_docs,
              "baselineReproduction":{"status":"passed","phase":phase,"checkedValues":len(baseline_reproduction),"rows":baseline_reproduction,
                                       "priceTolerance":{"absolute":PRICE_ABS_TOL,"relative":PRICE_REL_TOL}},
              "preflight":preflight,"forecasts":forecasts,"modelFitCounts":{"FJointSystems":len(fit_docs),"BCRefits":0,"SARIMARefits":0,"ASeparateFits":0}}
    if phase=="T": manifest["vGateReportSha256"]=file_sha(output_dir/"validation-report.json")
    _write_derived(output_dir/("fit-diagnostics.json" if phase=="V" else "test-fit-diagnostics.json"),canonical({"jointFits":fit_docs,"fitDiagnostics":diag}))
    _write_exclusive(manifest_path,canonical(manifest))
    _write_exclusive(output_dir/terminal_name,canonical({"phase":phase,"status":"complete_answer_free","attemptId":attempt_id,
                                                         "protocolSha256":protocol["protocolSha256"],"manifestSha256":file_sha(manifest_path)}))
    return manifest


def verify_phase(phase:str,output_dir:Path=OUT,*,allow_missing_terminal:bool=False)->dict[str,Any]:
    """Verify frozen answer-free grid, all source pins and stored-coefficient replay."""
    protocol=_load_protocol(output_dir)
    manifest_name,csv_name,_,terminal_name=_phase_files(phase)
    path=output_dir/manifest_name;doc=_json(path)
    if doc.get("status")=="not_run_validation_failed":
        if phase!="T" or not (output_dir/"test-report.json").is_file() or not (output_dir/terminal_name).is_file():
            raise ValueError("invalid T not-run marker")
        terminal=_json(output_dir/terminal_name)
        if (terminal.get("status")!="not_run_validation_failed" or terminal.get("protocolSha256")!=protocol["protocolSha256"] or
                terminal.get("manifestSha256")!=file_sha(path)):
            raise ValueError("T not-run terminal does not bind frozen protocol")
        if doc.get("vGateReportSha256")!=file_sha(output_dir/"validation-report.json"):
            raise ValueError("T not-run marker does not bind the scored V gate")
        return doc
    if doc.get("phase")!=phase or doc.get("protocolSha256")!=protocol["protocolSha256"]: raise ValueError("FHS manifest protocol/phase binding mismatch")
    if phase=="T":
        v_report_path=output_dir/"validation-report.json"
        if not v_report_path.is_file() or doc.get("vGateReportSha256")!=file_sha(v_report_path) or not _json(v_report_path).get("vGateMetrics",{}).get("passed"):
            raise ValueError("T was produced without a hash-bound passing V gate")
    if file_sha(output_dir/csv_name)!=doc.get("forecastFileSha256") or (output_dir/csv_name).read_bytes()!=_forecast_csv(doc.get("forecasts",[])):
        raise ValueError("FHS forecast CSV does not replay from immutable manifest")
    if doc.get("baselineReproduction",{}).get("status")!="passed": raise ValueError("FHS B reproduction has not passed")
    if len(doc.get("forecasts",[]))!=PHASE_GRID[phase]: raise ValueError("FHS forecast grid size mismatch")
    start,end,origin_start=rhs.PHASES[phase]
    expected={(p,(start+dt.timedelta(days=i)-dt.timedelta(days=h)).isoformat(),
               (start+dt.timedelta(days=i)).isoformat(),h)
              for p in ("cabbage","radish","onion","garlic")
              for i in range((end-start).days+1) for h in range(1,11)}
    keys={(r.get("product"),r.get("originDate"),r.get("targetDate"),r.get("horizon")) for r in doc["forecasts"]}
    if keys!=expected: raise ValueError("FHS forecast manifest does not preserve the exact four-crop grid")
    if len(doc.get("jointFitDiagnostics",[]))!=3: raise ValueError("FHS must preserve three monthly F fits")
    claim_path=output_dir/("validation-claim.json" if phase=="V" else "test-claim.json")
    terminal_path=output_dir/terminal_name
    if not claim_path.is_file() or (not terminal_path.is_file() and not allow_missing_terminal): raise ValueError("FHS phase claim or terminal is missing")
    claim=_json(claim_path)
    if claim.get("protocolSha256")!=protocol["protocolSha256"]: raise ValueError("FHS phase claim protocol mismatch")
    if terminal_path.is_file():
        terminal=_json(terminal_path)
        if (terminal.get("manifestSha256")!=file_sha(path) or terminal.get("protocolSha256")!=protocol["protocolSha256"] or
                terminal.get("status")!="complete_answer_free" or terminal.get("attemptId")!=claim.get("attemptId")):
            raise ValueError("FHS phase claim/terminal does not bind the forecast manifest")
    # Prove the original RHS B/C and unchanged crop controls still replay from pinned inputs.
    rhs.verify_phase(phase,SOURCE_OUT)
    source_doc,source_by,source_fits=_load_rhs_phase(phase,protocol)
    from . import equal_info_online_forecasts as base
    raw_prices=_prices();direct_series=base.direct_horizon._observations(raw_prices)
    output_by={(r["product"],r["originDate"],r["targetDate"],r["horizon"]):r for r in doc["forecasts"]}
    f_fits={f["fitId"]:f for f in doc["jointFitDiagnostics"]}
    source_protocol=rhs._load_protocol(SOURCE_OUT)
    for fit_id,fit in f_fits.items():
        fit_origin=dt.date.fromisoformat(fit_id.removeprefix("radish-"));source_fit=source_fits.get(fit_id)
        if source_fit is None: raise ValueError(f"F fit has no pinned RHS fit: {fit_id}")
        _,xraw,y,targets=rhs._reconstruct_training_fold(fit_origin,raw_prices,base)
        xstd=rhs._verify_fit_state(source_fit,phase,source_protocol,xraw,y,targets)
        _verify_f_fit_replay(fit,source_fit,xstd,y)
        a_audit=_a_audit(source_fit,xstd,y,np.asarray(fit["fitCoefficients"],dtype=np.float64))
        if canonical({k:v for k,v in a_audit.items() if k!="trainingVectors"})!=canonical(fit.get("aAudit")):
            raise ValueError(f"saved A training rule does not replay for {fit_id}")
        if fit.get("trainingDiagnostics")!=_score_training_records(source_fit,y,targets,fit,a_audit):
            raise ValueError(f"saved monthly training diagnostics do not replay for {fit_id}")
    expected_diag=[r for fit in doc["jointFitDiagnostics"] for r in fit["trainingDiagnostics"]]
    if doc.get("fitDiagnostics")!=expected_diag or doc.get("fitDiagnosticsSha256")!=sha256_bytes(canonical(expected_diag)):
        raise ValueError("flat scorer training diagnostics digest/replay mismatch")
    diag_path=output_dir/("fit-diagnostics.json" if phase=="V" else "test-fit-diagnostics.json")
    if not diag_path.is_file() or canonical(_json(diag_path))!=canonical({"jointFits":doc["jointFitDiagnostics"],"fitDiagnostics":expected_diag}):
        raise ValueError("persisted F fit diagnostics differ from answer-free manifest")
    for key,source in source_by.items():
        row=output_by[key]
        if row.get("eligible")!=bool(source.get("eligible")): raise ValueError(f"eligibility changed at {key}")
        if key[0]!="radish":
            for cond in CONDITIONS:
                _compare_condition(row["conditions"][cond],_condition_from_source(source["conditions"].get("C",source["conditions"].get("B",{}))),key,cond)
            continue
        if not row["eligible"]:
            _compare_condition(row["conditions"]["B"],_condition_from_source(source["conditions"]["B"]),key,"B")
            _compare_condition(row["conditions"]["C"],_condition_from_source(source["conditions"]["C"]),key,"C")
            for cond in ("F","A"):
                _compare_condition(row["conditions"][cond],_condition_from_source(source["conditions"]["C"]),key,cond)
            continue
        fit_id=source["conditions"]["C"].get("fitId") or source["conditions"]["B"].get("fitId")
        fit=f_fits.get(fit_id)
        if fit is None or row.get("fitId")!=fit_id: raise ValueError(f"F forecast fit link mismatch at {key}")
        for cond in ("B","C"):
            _compare_condition(row["conditions"][cond],_condition_from_source(source["conditions"][cond]),key,cond)
        feat,p0,audit=base._direct_feature(direct_series["radish"],dt.date.fromisoformat(key[2]),key[3],{})
        if feat is None or p0 is None: raise ValueError(f"source feature unavailable during FHS replay at {key}")
        source_fit=source_fits[fit_id]; h=key[3]-1
        x=(np.asarray(feat,dtype=np.float64)[list(rhs.ACTIVE)]-np.asarray(source_fit["standardScalerMean"][h]))/np.asarray(source_fit["standardScalerScale"][h])
        f_log=float((x-np.asarray(fit["scalerXMean"][h]))@np.asarray(fit["fitCoefficients"][h])+float(fit["scalerYMean"][h]))
        _compare_condition(row["conditions"]["F"],_compose_new_condition(source["conditions"]["C"],f_log,p0,audit,fit_id,None),key,"F")
        rho=(fit.get("aAudit") or {}).get("rho") if (fit.get("aAudit") or {}).get("status")=="ok" else None
        _compare_condition(row["conditions"]["A"],_compose_new_condition(source["conditions"]["C"],None,p0,audit,fit_id,rho,
                           b_log=source["conditions"]["B"].get("directLogPrediction"),
                           c_log=source["conditions"]["C"].get("directLogPrediction"),
                           a_reason=(fit.get("aAudit") or {}).get("reason")),key,"A")
    # Checkpoint hashes and their claim/terminal pair are part of the immutable fit lineage.
    for fit_id,fit in f_fits.items():
        prefix=f"{phase.lower()}-{fit_id}"; checkpoint_path=output_dir/"fit-checkpoints"/f"{prefix}.json"
        claim_fit_path=output_dir/"fit-checkpoints"/f"{prefix}.claim.json"
        terminal_fit_path=output_dir/"fit-checkpoints"/f"{prefix}.terminal.json"
        if not (checkpoint_path.is_file() and claim_fit_path.is_file() and terminal_fit_path.is_file()):
            raise ValueError(f"F monthly checkpoint lineage incomplete: {fit_id}")
        checkpoint=_json(checkpoint_path);claim_fit=_json(claim_fit_path);terminal_fit=_json(terminal_fit_path)
        source_fit=source_fits[fit_id]; expected_hashes=_f_fit_hashes(source_fit,protocol)
        input_sha=sha256_bytes(canonical(expected_hashes))
        bundle={"status":"ok","fitId":fit_id,"f":fit,"a":fit.get("aAudit")}
        # The fit checkpoint stores the full bundle produced by _fit_f_bundle.
        if (checkpoint.get("fit",{}).get("fitId")!=fit_id or checkpoint.get("inputSha256")!=input_sha or
                checkpoint.get("checkpointSha256")!=sha256_bytes(canonical(checkpoint.get("fit"))) or
                checkpoint.get("fit",{}).get("status")!="ok" or checkpoint.get("fit",{}).get("f")!=fit or
                checkpoint.get("fit",{}).get("a")!=fit.get("aAudit") or checkpoint.get("phase")!=phase or
                terminal_fit.get("inputSha256")!=input_sha or terminal_fit.get("checkpointSha256")!=checkpoint.get("checkpointSha256") or
                claim_fit.get("inputSha256")!=input_sha or claim_fit.get("inputHashes")!=expected_hashes):
            raise ValueError(f"F monthly checkpoint binding mismatch: {fit_id}")
    return doc


def _compare_condition(actual:Mapping[str,Any],expected:Mapping[str,Any],key:Any,name:str)->None:
    for field in ("status","fitId","cutoff","p0Date","fallbackReason"):
        if actual.get(field)!=expected.get(field): raise ValueError(f"{name} {field} differs from saved source at {key}")
    for field in ("directLogPrediction","directPrediction","blendPrediction","p0","sarimaPrediction"):
        a=actual.get(field);e=expected.get(field)
        if a is None or e is None:
            if a is not e: raise ValueError(f"{name} {field} availability differs at {key}")
        else:
            tol=LOG_REPLAY_TOL if field=="directLogPrediction" else PRICE_REL_TOL
            atol=LOG_REPLAY_TOL if field=="directLogPrediction" else PRICE_ABS_TOL
            if not math.isclose(float(a),float(e),rel_tol=tol,abs_tol=atol):
                raise ValueError(f"{name} {field} fails replay at {key}")


def _compose_new_condition(source:Mapping[str,Any],logpred:float|None,p0:float,audit:Mapping[str,Any],fit_id:str,
                           rho:float|None,*,b_log:float|None=None,c_log:float|None=None,a_reason:str|None=None)->dict[str,Any]:
    sarima=source.get("sarimaPrediction");status=source.get("status")
    if logpred is None and rho is not None and b_log is not None and c_log is not None:
        logpred=float(b_log)+rho*(float(c_log)-float(b_log))
    if logpred is None:
        return {"status":"not_issued","directLogPrediction":None,"directPrediction":None,"blendPrediction":None,
                "fitId":fit_id,"cutoff":audit.get("cutoff"),"p0":float(p0),"p0Date":audit.get("p0Date"),
                "sarimaPrediction":sarima,"fallbackReason":"A unsupported: "+str(a_reason)}
    direct=float(p0*math.exp(logpred));blend=0.5*(direct+float(sarima)) if sarima is not None else direct
    return {"status":status,"directLogPrediction":logpred,"directPrediction":direct,"blendPrediction":blend,
            "fitId":fit_id,"cutoff":audit.get("cutoff"),"p0":float(p0),"p0Date":audit.get("p0Date"),
            "sarimaPrediction":sarima,"fallbackReason":source.get("fallbackReason")}


def _validation_gate_passed(output_dir:Path=OUT)->bool:
    return bool(score_phase("V",output_dir).get("vGateMetrics",{}).get("passed"))


def score_phase(phase:str,output_dir:Path=OUT)->dict[str,Any]:
    """Connect actual prices only after the immutable answer-free phase verifies."""
    from . import radish_functional_sharing_score as scorer
    from . import equal_info_online_forecasts as base
    doc=verify_phase(phase,output_dir)
    if doc.get("status")=="not_run_validation_failed": return _json(output_dir/"test-report.json")
    actuals=rhs._actuals_by_product_target(_json(base.PRICES_PATH))
    report=scorer.score_manifest(doc,baseline_rows=_baseline_rows_for_scorer(phase),actuals=actuals,
                                 fit_diagnostics=doc["fitDiagnostics"],phase=phase)
    name="validation-report.json" if phase=="V" else "test-report.json";path=output_dir/name
    raw=canonical(report)
    if path.exists():
        if path.read_bytes()!=raw: raise ValueError("existing FHS score report differs from replay")
        return _json(path)
    _write_exclusive(path,raw);return report


def run_phase(phase:str,output_dir:Path=OUT)->dict[str,Any]:
    doc=_build_phase(phase,output_dir)
    if doc.get("status")=="not_run_validation_failed": return doc
    verify_phase(phase,output_dir)
    report=score_phase(phase,output_dir)
    if phase=="V" and not report.get("vGateMetrics",{}).get("passed"):
        marker={"schemaVersion":1,"phase":"T","status":"not_run_validation_failed","reason":"V F-vs-C price/coverage gate failed","protocolSha256":doc["protocolSha256"],"vGateReportSha256":file_sha(output_dir/"validation-report.json")}
        _write_derived(output_dir/"test-manifest.json",canonical(marker));_write_derived(output_dir/"test-report.json",canonical({**marker,"forecastRows":0,"jointFits":0}))
        _write_derived(output_dir/"test-terminal.json",canonical({"phase":"T","status":marker["status"],"protocolSha256":doc["protocolSha256"],"manifestSha256":file_sha(output_dir/"test-manifest.json")}))
    return {"phase":phase,"manifest":doc,"report":report}


def main(argv:Sequence[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("freeze","build","verify","score","run"))
    parser.add_argument("--phase",choices=("V","T"));parser.add_argument("--output-dir",type=Path,default=OUT)
    args=parser.parse_args(argv)
    if args.command=="freeze": result=freeze_protocol(args.output_dir)
    elif args.phase is None: parser.error("--phase required for this command")
    elif args.command=="build": result=_build_phase(args.phase,args.output_dir)
    elif args.command=="verify": result=verify_phase(args.phase,args.output_dir)
    elif args.command=="score": result=score_phase(args.phase,args.output_dir)
    else: result=run_phase(args.phase,args.output_dir)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False,default=str))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
