"""Single fixed R/M/H hidden-representation experiment for garlic E106.

This module is intentionally inert on import.  Its numerical core is shared by
the answer-free V/T builders; it contains no parameter search or refit-on-score
path.  See docs/PLAN.md for the frozen experiment contract.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import e106_mlp16 as mlp

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/e106-rmh-20261011"
SCHEMA = "e106-rmh-v1"
PHASES = {"validation": (dt.date(2026, 4, 1), dt.date(2026, 6, 20)),
          "final": (dt.date(2026, 7, 11), dt.date(2026, 9, 27))}
ORIGIN_START = {"validation": dt.date(2026, 4, 1), "final": dt.date(2026, 7, 1)}
RCOND_RULE = "s > eps(float64) * max(A.shape) * s_max"
LOSS_BOUND_ABS = 1e-12
LOSS_BOUND_REL = 1e-10
DM_RETAIN_REL = 1e-12
R_REPRO_ABS = 1e-8
R_REPRO_REL = 1e-8
FOUR_CROP_PRODUCTS = ("cabbage", "radish", "onion", "garlic")
EXPECTED_RUNTIME = {"python": "3.12.14", "numpy": "2.5.3", "scikitLearn": "1.9.1",
                    "scipy": "1.18.1", "joblib": "1.6.0"}
PINNED_MLP = {
    "manifest": "174f76e25c79e9fff2973dbc42f6cd869742222d7ddd93e6483451bb9e0607bf",
    "report": "4778f0a8555d4971deaca6385dcc76ce37032cc98ffae038f1bb6ca7b10883f9",
    "trainingDiagnostic": "68d2f49cb9b2633aa5b01b1449f354d442e68e539156b1adb71f76d4ee742e2d",
    "module": "5855f4c40b0f3ef60c24e9ba48ee7fe985ea5eaf25d3aea147690645e79c7f6f",
}


def _runtime(require: bool = True) -> dict[str, str]:
    import platform
    import sklearn
    import scipy
    import joblib
    current = {"python": platform.python_version(), "numpy": np.__version__,
               "scikitLearn": sklearn.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__}
    if require and current != EXPECTED_RUNTIME:
        raise RuntimeError(f"RMH requires frozen runtime {EXPECTED_RUNTIME}; found {current}")
    return current


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                        allow_nan=False) + "\n").encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(doc: dict[str, Any], field: str = "sha256") -> str:
    body = dict(doc)
    body.pop(field, None)
    return sha_bytes(canonical(body).rstrip(b"\n"))


def _finite_array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite values")
    return array


def hidden_design(x_scaled: Any, coefs: list[Any], intercepts: list[Any]) -> np.ndarray:
    """Build exactly [1, ReLU(XW+b)] from the persisted MLP16 input layer."""
    x = _finite_array(x_scaled, "scaled training matrix", 2)
    w = _finite_array(coefs[0], "first-layer coefficients", 2)
    b = _finite_array(intercepts[0], "first-layer intercept", 1)
    if w.shape != (x.shape[1], 16) or b.shape != (16,):
        raise ValueError("expected frozen 106-to-16 MLP first layer")
    z = np.maximum(0.0, x @ w + b)
    a = np.column_stack((np.ones(len(x), dtype=np.float64), z))
    if not np.isfinite(a).all():
        raise ValueError("hidden design is nonfinite")
    return a


def _least_squares_svd(a: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """One float64 SVD with the predeclared numerical rank rule."""
    u, singular, vt = np.linalg.svd(a, full_matrices=False)
    threshold = float(np.finfo(np.float64).eps * max(a.shape) * singular[0]) if len(singular) else 0.0
    keep = singular > threshold
    if not keep.any():
        raise ValueError("SVD retained no singular values")
    beta = vt[keep].T @ ((u[:, keep].T @ y) / singular[keep])
    audit = {"singularValues": singular.tolist(), "threshold": threshold,
             "retainedRank": int(keep.sum()), "designShape": list(a.shape),
             "rcondRule": RCOND_RULE, "retainedMask": keep.tolist()}
    return beta, audit, u, singular, vt


def decompose_training(x: Any, y: Any, scaler_mean: Any, scaler_scale: Any,
                       coefs: list[Any], intercepts: list[Any]) -> dict[str, Any]:
    """Fit R and unpenalized H, and exactly decompose original M output.

    R and H use the same float64 rows and M's already-fitted training-only
    scaler.  No target clipping or prediction fallback enters these losses.
    """
    from sklearn.linear_model import Ridge

    raw = _finite_array(x, "training matrix", 2)
    target = _finite_array(y, "training target", 1)
    mean = _finite_array(scaler_mean, "scaler mean", 1)
    scale = _finite_array(scaler_scale, "scaler scale", 1)
    if len(raw) != len(target) or raw.shape[1] != 106 or mean.shape != (106,) or scale.shape != (106,):
        raise ValueError("training rows or scaler do not match the frozen 106-column fit")
    if np.any(scale <= 0):
        raise ValueError("scaler scales must be positive")
    xs = (raw - mean) / scale
    if not np.isfinite(xs).all():
        raise ValueError("scaled training matrix is nonfinite")

    ridge = Ridge(alpha=10.0, fit_intercept=True, solver="cholesky")
    ridge.fit(xs, target)
    r_prediction = ridge.predict(xs)
    r_coef = np.concatenate(([float(ridge.intercept_)], ridge.coef_.astype(float))).tolist()
    a = hidden_design(xs, coefs, intercepts)
    beta_h, svd_audit, u, singular, vt = _least_squares_svd(a, target)
    h_prediction = a @ beta_h

    m_w1 = _finite_array(coefs[0], "MLP first-layer coefficients", 2)
    m_w2 = _finite_array(coefs[1], "MLP output coefficients", 2)
    m_b1 = _finite_array(intercepts[0], "MLP first-layer intercept", 1)
    m_b2 = _finite_array(intercepts[1], "MLP output intercept", 1)
    if m_w2.shape != (16, 1) or m_b2.shape != (1,):
        raise ValueError("expected frozen MLP16 output layer")
    m_prediction = (a[:, 1:] @ m_w2[:, 0]) + m_b2[0]

    # Reuse the same SVD for target, M-output and R-output projections.
    keep = singular > svd_audit["threshold"]
    def projection_gap(values: np.ndarray) -> tuple[float, list[float]]:
        coefficients = u[:, keep].T @ values
        projected = u[:, keep] @ coefficients
        residual = values - projected
        return float(np.mean(residual * residual)), coefficients.tolist()

    d_m, m_projected_coef = projection_gap(m_prediction)
    d_r, r_projected_coef = projection_gap(r_prediction)
    lr = float(np.mean((target - r_prediction) ** 2))
    lm = float(np.mean((target - m_prediction) ** 2))
    # This is the loss from the actual SVD-fitted H coefficients used later.
    lh = float(np.mean((target - h_prediction) ** 2))
    projection_h = vt[keep].T @ ((u[:, keep].T @ target) / singular[keep])
    projected_fit = a @ projection_h
    h_reconstruction_error = float(np.max(np.abs(projected_fit - h_prediction)))
    target_projection = u[:, keep] @ (u[:, keep].T @ target)
    fitted_projection = a @ projection_h
    projection_error = float(np.max(np.abs(target_projection - fitted_projection)))
    output_gap, representation_gap = lm - lh, lh - lr
    eq_error = abs((output_gap + representation_gap) - (lm - lr))
    m_rms = float(np.mean(m_prediction * m_prediction))
    m_preserved_approx = d_m <= DM_RETAIN_REL * max(1.0, m_rms)
    upper_m = (math.sqrt(lm) + math.sqrt(d_m)) ** 2
    upper_r = (math.sqrt(lr) + math.sqrt(d_r)) ** 2
    slack = LOSS_BOUND_ABS + LOSS_BOUND_REL * max(1.0, lr, lm, lh, upper_m, upper_r)
    if (h_reconstruction_error > 1e-9 or projection_error > 1e-9
            or eq_error > 1e-10 * max(1.0, lr, lm, lh)):
        raise ArithmeticError("H reconstruction or loss decomposition identity failed")
    if lh > upper_m + slack or lh > upper_r + slack:
        raise ArithmeticError("H loss violates its reconstruction-aware RMS bound")
    return {"n": len(target), "L_R": lr, "L_M": lm, "L_H": lh,
            "outputGap": output_gap, "representationGap": representation_gap,
            "decompositionIdentityError": eq_error, "D_M": d_m, "D_R": d_r,
            "MOutputRetainedApprox": m_preserved_approx,
            "interpretation": "approximate_retention" if m_preserved_approx else "not_retained_in_hidden_span",
            "HUpperBoundFromM": upper_m, "HUpperBoundFromR": upper_r,
            "boundSlack": slack, "hReconstructionMaxAbsError": h_reconstruction_error,
            "hProjectionMaxAbsError": projection_error,
            "ridgeCoefficientIncludingIntercept": r_coef,
            "hCoefficientIncludingIntercept": beta_h.tolist(),
            "hTrainingPrediction": h_prediction.tolist(),
            "mTrainingPrediction": m_prediction.tolist(),
            "rTrainingPrediction": r_prediction.tolist(),
            "mProjectionCoefficient": m_projected_coef,
            "rProjectionCoefficient": r_projected_coef,
            "svd": svd_audit,
            # Persist factors once so later lineage verification never refits H.
            "svdFactors": {"u": u.tolist(), "vt": vt.tolist()}}


def _is_h_output(row: dict[str, Any]) -> bool:
    return row.get("usedH") is True


def _assert_exact_fallback(row: dict[str, Any], base_row: dict[str, Any], key: Any) -> None:
    expected = base_row["candidate"]
    for field in ("status", "prediction", "reason"):
        if row.get(field) != expected.get(field):
            raise ValueError(f"fallback differs from exact E106 at {key}")
    if row.get("usedH") is True or not row.get("fallbackReason"):
        raise ValueError(f"fallback provenance is inconsistent at {key}")


def _assert_r_replay(row:dict[str,Any],new:float,old:float,key:Any)->float:
    if not ridge_reproduces(new,old):
        raise ValueError(f"stored R does not reproduce original R106 at {key}")
    gap=abs(new-old)
    if row.get("RDirectPrediction") is None or row.get("r106ReplayAbsError") is None:
        raise ValueError(f"forecast omitted the required R replay fields at {key}")
    if (not math.isclose(float(row["RDirectPrediction"]),new,rel_tol=1e-10,abs_tol=1e-8)
            or not math.isclose(float(row["r106ReplayAbsError"]),gap,rel_tol=1e-10,abs_tol=1e-8)):
        raise ValueError(f"serialized R replay fails at {key}")
    return gap


def _four_crop_cells(candidate_all: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[tuple[str,int]], set[tuple[str,int]]]:
    expected={(product,h) for product in FOUR_CROP_PRODUCTS for h in mlp.HORIZONS}
    observed={(x["product"],x["horizon"]) for x in candidate_all if x["product"] in FOUR_CROP_PRODUCTS}
    cells=[]
    for product,h in sorted(expected):
        xs=[x for x in candidate_all if x["product"]==product and x["horizon"]==h]
        if xs:
            cells.append({"product":product,"horizon":h,
                "candidateMae":math.fsum(abs(x["candidate"]-x["actual"]) for x in xs)/len(xs),
                "E106Mae":math.fsum(abs(x["E106"]-x["actual"]) for x in xs)/len(xs),
                "candidateMape":100*math.fsum(abs(x["candidate"]-x["actual"])/x["actual"] for x in xs)/len(xs),
                "E106Mape":100*math.fsum(abs(x["E106"]-x["actual"])/x["actual"] for x in xs)/len(xs)})
    return cells, expected, observed


def _portfolio_macro(cells:list[dict[str,Any]],complete:bool)->dict[str,float|None]:
    out={k:(math.fsum(x[k] for x in cells)/40 if complete else None)
         for k in ("candidateMae","E106Mae","candidateMape","E106Mape")}
    out["deltaMae"]=out["candidateMae"]-out["E106Mae"] if out["candidateMae"] is not None else None
    out["deltaMape"]=out["candidateMape"]-out["E106Mape"] if out["candidateMape"] is not None else None
    return out


def _common_component_row(row:dict[str,Any],fields:tuple[str,...],names:tuple[str,...],
                          *,month:str,actual:float)->dict[str,Any]|None:
    if any(row.get(field) is None for field in fields):
        return None
    return {"month":month,"actual":actual,**{name:float(row[field]) for name,field in zip(names,fields)}}


def _label_dm_units(result:dict[str,Any],loss:str)->dict[str,Any]:
    out=dict(result)
    out["lossUnit"]="price" if loss=="ae" else "fractional_absolute_percentage_error"
    if loss=="ape" and out.get("meanLossDifference") is not None:
        out["meanLossDifferencePercentPoints"]=100*float(out["meanLossDifference"])
    return out


def _conclusion(macro: dict[str, float | None]) -> str:
    mae, mape = macro.get("deltaMae"), macro.get("deltaMape")
    if mae is None or mape is None or not math.isfinite(float(mae)) or not math.isfinite(float(mape)):
        return "unsupported_insufficient_fixed_cells"
    if mae <= 0 and mape < 0:
        return "improved"
    if mae > 0 and mape > 0:
        return "worsened"
    return "mixed_or_inconclusive"


def _fit_count_summary(phase: str) -> dict[str,Any]:
    hidden=0 if phase=="validation" else 30
    return {"newHiddenFits":hidden,"fitCounts":{"newHidden":hidden,"R":30,"H":30,
            "M":0 if phase=="validation" else 30}}


def _saved_svd(a: np.ndarray, y: np.ndarray, fit: dict[str, Any]) -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """Verify persisted SVD factors and replay H using those factors only."""
    audit=fit["svd"]
    factors=fit["svdFactors"]
    singular=_finite_array(audit["singularValues"],"saved singular values",1)
    u=_finite_array(factors["u"],"saved SVD U",2)
    vt=_finite_array(factors["vt"],"saved SVD Vt",2)
    rank=len(singular)
    if u.shape!=(len(y),rank) or vt.shape!=(rank,a.shape[1]):
        raise ValueError("saved SVD factor shapes do not match the training design")
    threshold=np.finfo(np.float64).eps*max(a.shape)*(singular[0] if len(singular) else 0.0)
    keep=singular>threshold
    if (not len(singular) or np.any(singular<0) or np.any(np.diff(singular)>0)
            or audit.get("rcondRule")!=RCOND_RULE
            or audit.get("designShape")!=list(a.shape)
            or audit.get("retainedMask")!=keep.tolist()
            or audit.get("retainedRank")!=int(keep.sum())
            or not math.isclose(float(audit.get("threshold",math.inf)),threshold,rel_tol=1e-12,abs_tol=1e-14)
            or not np.allclose((u*singular[None,:])@vt,a,rtol=1e-10,atol=1e-10)
            or not np.allclose(u.T@u,np.eye(rank),rtol=1e-9,atol=1e-9)
            or not np.allclose(vt@vt.T,np.eye(rank),rtol=1e-9,atol=1e-9)):
        raise ValueError("saved SVD factors fail reconstruction or orthogonality checks")
    if not keep.any():
        raise ValueError("saved SVD retained no singular values")
    beta=vt[keep].T@((u[:,keep].T@y)/singular[keep])
    return beta,u,singular,vt


def price_prediction(p0: float, log_ratio: float) -> float:
    if not math.isfinite(p0) or p0 <= 0 or not math.isfinite(log_ratio):
        raise ValueError("nonpositive anchor or nonfinite log-ratio")
    try:
        result = p0 * math.exp(log_ratio)
    except OverflowError as exc:
        raise ValueError("price reconstruction is nonfinite or nonpositive") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError("price reconstruction is nonfinite or nonpositive")
    return result


def ridge_reproduces(new: float, original: float) -> bool:
    return math.isfinite(new) and math.isfinite(original) and abs(new-original) <= R_REPRO_ABS + R_REPRO_REL * abs(original)


def _manifest_path(phase: str) -> Path:
    return OUT / f"{phase}-manifest.json"


def _protocol_path() -> Path:
    return OUT / "protocol.json"


def freeze_protocol() -> dict[str, Any]:
    """Save the executable, input and runtime pins before any model fit."""
    runtime = _runtime()
    mlp._check_pinned_inputs()
    saved_m = {"manifest": mlp.file_sha(mlp.OUT / "confirmation-manifest.json"),
               "report": mlp.file_sha(mlp.OUT / "confirmation-report.json"),
               "trainingDiagnostic": mlp.file_sha(mlp.OUT / "training-diagnostic.json"),
               "module": mlp.file_sha(Path(mlp.__file__))}
    if saved_m != PINNED_MLP:
        raise ValueError("frozen saved MLP source pins do not match the completed diagnostics")
    doc = {"schema": SCHEMA, "frozenBeforeFit": True,
           "moduleSha256": sha_bytes(Path(__file__).read_bytes()),
           "inputs": {"prices": mlp.file_sha(mlp.PRICES), "r106": mlp.file_sha(mlp.R106),
                      "helper": mlp.file_sha(Path(mlp.pf.__file__)),
                      "e106Confirmation": mlp.file_sha(mlp.CONF_BASE), "e106Final": mlp.file_sha(mlp.FINAL_BASE),
                      "savedMlpManifest": saved_m["manifest"],"savedMlpReport":saved_m["report"],
                      "trainingDiagnostic":saved_m["trainingDiagnostic"],"mlpModule":saved_m["module"]},
           "runtime": runtime,
           "protocol": {"phases": {k: [v[0].isoformat(), v[1].isoformat()] for k,v in PHASES.items()},
                        "originStart": {k:v.isoformat() for k,v in ORIGIN_START.items()},
                        "model": {"R": {"alpha":10.0,"fitIntercept":True,"solver":"cholesky"},
                                  "M": mlp.MODEL_PARAMS,
                                  "H": "least squares on intercept plus frozen 16-unit ReLU activations"},
                        "scaler": "saved M fitted StandardScaler; T refits once on own training rows",
                        "target": "log(target/p0)", "blend": "0.5*direct-price + 0.5*same saved causal SARIMA",
                        "SVD": RCOND_RULE, "DMRetainedApproxRelative": DM_RETAIN_REL,
                        "lossBoundSlack": {"absolute":LOSS_BOUND_ABS,"relative":LOSS_BOUND_REL},
                        "ridgeReproductionTolerance": {"absolute":R_REPRO_ABS,"relative":R_REPRO_REL},
                        "newHiddenFits": 0,
                        "validationFitBudget": {"R":30,"H":30,"M":0},
                        "conditionalFinalFitBudget": {"R":30,"H":30,"M":30},
                        "validationGate": {"garlic10hDeltaMapeLt":0,"deltaMaeLe":0,
                                           "noAdditionalFallbackOrUnissued":True,"minimumPairsPerHorizon":30},
                        "statistics": {"fixedTests":20,"calendarHacDays":[9,28],"holmAlpha":0.05,
                                       "jointCalendarBootstrapBlocks":[14,28],"replicates":2000,"seed":20261011}}}
    doc["protocolSha256"] = digest(doc, "protocolSha256")
    path = _protocol_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(doc))
    return doc


def verify_protocol() -> dict[str, Any]:
    path = _protocol_path()
    doc = json.loads(path.read_text())
    if doc.get("protocolSha256") != digest(doc, "protocolSha256"):
        raise ValueError("frozen RMH protocol digest mismatch")
    if doc.get("moduleSha256") != sha_bytes(Path(__file__).read_bytes()):
        raise ValueError("RMH source changed after protocol freeze")
    if doc.get("runtime") != _runtime():
        raise RuntimeError("RMH runtime changed after protocol freeze")
    mlp._check_pinned_inputs()
    expected = {"prices": mlp.file_sha(mlp.PRICES), "r106": mlp.file_sha(mlp.R106),
                "helper": mlp.file_sha(Path(mlp.pf.__file__)),
                "e106Confirmation": mlp.file_sha(mlp.CONF_BASE), "e106Final": mlp.file_sha(mlp.FINAL_BASE),
                "savedMlpManifest":mlp.file_sha(mlp.OUT/"confirmation-manifest.json"),
                "savedMlpReport":mlp.file_sha(mlp.OUT/"confirmation-report.json"),
                "trainingDiagnostic":mlp.file_sha(mlp.OUT/"training-diagnostic.json"),
                "mlpModule":mlp.file_sha(Path(mlp.__file__))}
    if doc.get("inputs") != expected:
        raise ValueError("RMH source input changed after protocol freeze")
    return doc


def preflight(phase: str) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError("phase must be validation or final")
    protocol = verify_protocol()
    source_phase = "confirmation" if phase == "validation" else "final"
    base = mlp._base_doc(source_phase)
    if phase == "validation":
        _fit_from_saved_m("validation")
    else:
        _verify_validation_gate()
    _, features, by_h = mlp._make_features(source_phase, mlp._idx(base["forecasts"]), mlp._r106_index())
    support = {str(h): {"forecastKeys": len(by_h[h]), "featureRows": len(features[h])} for h in mlp.HORIZONS}
    if any(v["forecastKeys"] == 0 for v in support.values()):
        raise ValueError("incomplete horizon grid")
    return {"phase": phase, "protocolSha256": protocol["protocolSha256"], "fitCount": 30,
            "fitSupport": support, "status": "ready_no_fit_performed"}


def _claim_fit(phase: str, fit_id: str) -> Path:
    """Reserve each permitted fit exactly once before starting it."""
    path = OUT / ".fit-claims" / phase / f"{fit_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    protocol = verify_protocol()
    doc = {"schema": SCHEMA, "phase": phase, "fitId": fit_id,
           "protocolSha256": protocol["protocolSha256"],
           "runtime": protocol["runtime"], "inputHashes": protocol["inputs"],
           "moduleSha256": sha_bytes(Path(__file__).read_bytes()),
           "reservedAtUtc": dt.datetime.now(dt.timezone.utc).isoformat()}
    with path.open("xb") as stream:
        stream.write(canonical(doc))
    return path


def _verify_fit_claim(path: Path, phase: str, fit_id: str, protocol: dict[str,Any]) -> dict[str,Any]:
    claim=json.loads(path.read_text())
    expected={"schema":SCHEMA,"phase":phase,"fitId":fit_id,
              "protocolSha256":protocol["protocolSha256"],"runtime":protocol["runtime"],
              "inputHashes":protocol["inputs"],"moduleSha256":protocol["moduleSha256"]}
    if any(claim.get(k)!=v for k,v in expected.items()) or not claim.get("reservedAtUtc"):
        raise ValueError(f"fit claim does not bind the frozen attempt {fit_id}")
    return claim


def _verify_checkpoint(state_doc: dict[str,Any], *, phase: str, fit_id: str,
                       fit_origin: dt.date, training: list[dt.date], x: np.ndarray, y: np.ndarray,
                       train_audit: dict[str,Any], protocol: dict[str,Any],
                       saved_m: dict[str,dict[str,Any]]) -> dict[str,Any]:
    if (state_doc.get("stateSha256")!=digest(state_doc,"stateSha256")
            or state_doc.get("schema")!=SCHEMA or state_doc.get("phase")!=phase
            or state_doc.get("fitId")!=fit_id
            or state_doc.get("protocolSha256")!=protocol["protocolSha256"]
            or state_doc.get("inputHashes")!=protocol["inputs"]
            or state_doc.get("moduleSha256")!=protocol["moduleSha256"]
            or state_doc.get("runtime")!=protocol["runtime"]
            or state_doc.get("trainingKeysSha256")!=sha_bytes(canonical([d.isoformat() for d in training]))):
        raise ValueError(f"checkpoint protocol/source/runtime binding failed for {fit_id}")
    fit=state_doc.get("fitAudit",{})
    checks={"fitId":fit_id,"fitOrigin":fit_origin.isoformat(),
            "trainingKeysSha256":sha_bytes(canonical([d.isoformat() for d in training])),
            "trainingMatrixSha256":sha_bytes(x.tobytes(order="C")),
            "trainingTargetSha256":sha_bytes(y.tobytes(order="C"))}
    if any(fit.get(k)!=v for k,v in checks.items()) or fit.get("training")!=train_audit:
        raise ValueError(f"checkpoint no longer matches pinned training support for {fit_id}")
    if (fit.get("mlpParameters")!=mlp.MODEL_PARAMS
            or fit.get("newHiddenFits")!=(0 if phase=="validation" else 1)):
        raise ValueError(f"checkpoint fit budget or MLP parameters changed for {fit_id}")
    _verify_fit_claim(OUT/".fit-claims"/phase/(fit_id+"-RH.json"),phase,fit_id+"-RH",protocol)
    if phase=="validation":
        source=saved_m.get(fit_id)
        if (source is None or fit.get("mlpStateSource")!="saved_validation_fit"
                or fit.get("scalerMean")!=source.get("scalerMean")
                or fit.get("scalerScale")!=source.get("scalerScale")
                or fit.get("mlpCoefs")!=source.get("coefs")
                or fit.get("mlpCoefShapes")!=source.get("coefShapes")
                or fit.get("mlpIntercepts")!=source.get("intercepts")):
            raise ValueError(f"checkpoint M state differs from pinned validation fit for {fit_id}")
    else:
        if fit.get("mlpStateSource")!="fixed_authorized_refit":
            raise ValueError(f"checkpoint M source is not the authorized final refit for {fit_id}")
        _verify_fit_claim(OUT/".fit-claims"/phase/(fit_id+"-M.json"),phase,fit_id+"-M",protocol)
    return fit


def _fit_from_saved_m(phase: str) -> dict[str, dict[str, Any]]:
    path = mlp.OUT / ("confirmation-manifest.json" if phase == "validation" else "final-manifest.json")
    report_path = mlp.OUT / ("confirmation-report.json" if phase == "validation" else "final-report.json")
    diagnostic_path = mlp.OUT / "training-diagnostic.json"
    if phase == "validation":
        pins = {"manifest": mlp.file_sha(path), "report": mlp.file_sha(report_path),
                "trainingDiagnostic": mlp.file_sha(diagnostic_path), "module": mlp.file_sha(Path(mlp.__file__))}
        if pins != PINNED_MLP:
            raise ValueError("saved MLP confirmation/report/diagnostic/module pins changed")
    doc = json.loads(path.read_text())
    mlp.validate(doc, "confirmation" if phase == "validation" else "final")
    mlp.verify_sources(doc, "confirmation" if phase == "validation" else "final")
    return {r["fitId"]: r for r in doc["fitAudits"]}


def _training_state(phase: str, horizon: int, month: tuple[int, int], training: list[dt.date],
                    x: np.ndarray, y: np.ndarray, saved_m: dict[str, dict[str, Any]]):
    """Return M training state (saved in V, fixed refit in authorized T) and R/H."""
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from . import preserved_feature_forecasts as pf
    import warnings

    fit_id = f"garlic-h{horizon}-{month[0]}-{month[1]:02d}"
    if len(training) < pf.MIN_TRAIN:
        raise ValueError(f"{fit_id}: fewer than frozen minimum training targets")
    source_phase = "confirmation" if phase == "validation" else "final"
    if phase == "validation":
        m = saved_m.get(fit_id)
        if m is None or m.get("fitFailure"):
            raise ValueError(f"saved M state unavailable for {fit_id}")
        if m.get("trainingKeysSha256") != sha_bytes(canonical([t.isoformat() for t in training])):
            raise ValueError(f"saved M training key mismatch for {fit_id}")
        if m.get("trainingMatrixSha256") != sha_bytes(x.tobytes(order="C")):
            raise ValueError(f"saved M training matrix mismatch for {fit_id}")
        if m.get("trainingTargetSha256") != sha_bytes(y.tobytes(order="C")):
            raise ValueError(f"saved M training target mismatch for {fit_id}")
        mean = np.asarray(m["scalerMean"], dtype=np.float64)
        scale = np.asarray(m["scalerScale"], dtype=np.float64)
        coefs = [np.asarray(v, dtype=np.float64).reshape(shape) for v, shape in zip(m["coefs"], m["coefShapes"])]
        intercepts = [np.asarray(v, dtype=np.float64) for v in m["intercepts"]]
        fit_seconds = 0.0
        m_fit_source = "saved_validation_fit"
        model_audit = {"nIter": m.get("nIter"), "finalLoss": m.get("finalLoss"),
                       "lossCurve": m.get("lossCurve", []), "warnings": m.get("warnings", [])}
    else:
        m_claim = OUT / ".fit-claims" / phase / f"{fit_id}-M.json"
        if m_claim.exists():
            raise RuntimeError(f"interrupted M fit attempt already consumed for {fit_id}; refusing refit")
        _claim_fit(phase, fit_id + "-M")
        start = time.monotonic()
        scaler = StandardScaler()
        scaler.fit(x)
        mean, scale = scaler.mean_.astype(float), scaler.scale_.astype(float)
        model = MLPRegressor(**mlp.MODEL_PARAMS)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit((x - mean) / scale, y)
        coefs = [np.asarray(w, dtype=np.float64) for w in model.coefs_]
        intercepts = [np.asarray(v, dtype=np.float64) for v in model.intercepts_]
        model_state = [*mean,*scale,*model.loss_curve_,float(model.loss_),
                       *(v for layer in coefs for v in layer.ravel()),
                       *(v for layer in intercepts for v in layer.ravel())]
        if not all(math.isfinite(float(v)) for v in model_state):
            raise FloatingPointError(f"nonfinite MLP fitted state for {fit_id}")
        fit_seconds = time.monotonic() - start
        m_fit_source = "fixed_authorized_refit"
        model_audit = {"nIter": int(model.n_iter_), "finalLoss": float(model.loss_),
                       "lossCurve": [float(v) for v in model.loss_curve_],
                       "warnings": [f"{type(w.message).__name__}: {w.message}" for w in caught]}
    if mean.shape != (106,) or scale.shape != (106,) or np.any(scale <= 0):
        raise ValueError(f"invalid M scaler for {fit_id}")
    decomposition = decompose_training(x, y, mean, scale, coefs, intercepts)
    decomposition.update({"fitId": fit_id, "fitOrigin": training[-1].isoformat(),
                          "trainingRows": len(training), "trainingKeysSha256": sha_bytes(canonical([d.isoformat() for d in training])),
                          "trainingMatrixSha256": sha_bytes(x.tobytes(order="C")),
                          "trainingTargetSha256": sha_bytes(y.tobytes(order="C")),
                          "scalerMean": mean.tolist(), "scalerScale": scale.tolist(),
                          "mlpCoefs": [c.ravel().tolist() for c in coefs],
                          "mlpCoefShapes": [list(c.shape) for c in coefs],
                          "mlpIntercepts": [v.tolist() for v in intercepts],
                          "mlpStateSource": m_fit_source, "mlpFitSeconds": fit_seconds,
                          "mlpFitAudit": model_audit, "mlpParameters": mlp.MODEL_PARAMS,
                          "newHiddenFits": 0 if phase == "validation" else 1})
    return decomposition


def build_phase(phase: str, output: Path | None = None) -> dict[str, Any]:
    """Build and immutably save a complete answer-free phase forecast manifest."""
    from . import preserved_feature_forecasts as pf
    if phase not in PHASES:
        raise ValueError("phase must be validation or final")
    frozen_protocol = verify_protocol()
    runtime = _runtime()
    fit_count_summary=_fit_count_summary(phase)
    phase_start = time.monotonic()
    mlp._check_pinned_inputs()
    if phase == "final":
        _verify_validation_gate()
    src_phase = "confirmation" if phase == "validation" else "final"
    baseline = mlp._base_doc(src_phase)
    base_index = mlp._idx(baseline["forecasts"])
    r106 = mlp._r106_index()
    rows = mlp._identity_rows(src_phase, base_index, r106)
    series, features, by_h = mlp._make_features(src_phase, base_index, r106)
    m_audits = _fit_from_saved_m("validation") if phase == "validation" else {}
    row_index = {(r["product"], dt.date.fromisoformat(r["target"]), r["horizon"]): r for r in rows}
    fit_audits = []
    for h in mlp.HORIZONS:
        monthly: dict[tuple[int, int], list[tuple[str, dt.date, int]]] = {}
        for k in by_h[h]:
            origin = k[1] - dt.timedelta(days=h)
            monthly.setdefault((origin.year, origin.month), []).append(k)
        if len(monthly) != 3:
            raise ValueError(f"expected three monthly fits for garlic/h{h}")
        for month, month_keys in sorted(monthly.items()):
            frozen, fit_origin = mlp._frozen_fit_audit(r106, h, month_keys)
            # Keep each original fit origin, even for T's July 11 scoring start.
            training = [t for t in series if t >= pf.online.narrow.TRAIN_START and t < fit_origin and features[h].get(t) is not None]
            train_audit = pf._fit_audit(training, fit_origin)
            pf._assert_training_support(train_audit, frozen, "garlic", h, month)
            x = np.asarray([features[h][t]["preserved106"] for t in training], dtype=np.float64)
            y = np.asarray([math.log(series[t][0] / features[h][t]["p0"]) for t in training], dtype=np.float64)
            state_path = OUT / ".fit-states" / phase / f"garlic-h{h}-{month[0]}-{month[1]:02d}.json"
            fit_id=f"garlic-h{h}-{month[0]}-{month[1]:02d}"
            if state_path.exists():
                state_doc = json.loads(state_path.read_text())
                fit = _verify_checkpoint(state_doc,phase=phase,fit_id=fit_id,fit_origin=fit_origin,
                    training=training,x=x,y=y,train_audit=train_audit,protocol=frozen_protocol,saved_m=m_audits)
                fit["checkpointFileSha256"] = state_doc["stateSha256"]
                fit["checkpointPath"] = str(state_path.relative_to(ROOT))
            else:
                rh_claim = OUT / ".fit-claims" / phase / f"garlic-h{h}-{month[0]}-{month[1]:02d}-RH.json"
                if rh_claim.exists():
                    raise RuntimeError(f"interrupted R/H attempt consumed for {h}/{month}; refusing refit")
                _claim_fit(phase, f"garlic-h{h}-{month[0]}-{month[1]:02d}-RH")
                rh_start = time.monotonic()
                fit = _training_state(phase, h, month, training, x, y, m_audits)
                fit["ridgeAndHiddenFitSeconds"] = time.monotonic() - rh_start
                fit["training"] = train_audit
                fit["fitOrigin"] = fit_origin.isoformat()
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_doc = {"schema": SCHEMA, "phase": phase, "fitId": fit["fitId"],
                             "protocolSha256": frozen_protocol["protocolSha256"],
                             "inputHashes": frozen_protocol["inputs"],
                             "moduleSha256":frozen_protocol["moduleSha256"],"runtime":frozen_protocol["runtime"],
                             "trainingKeysSha256": fit["trainingKeysSha256"], "fitAudit": fit}
                state_doc["stateSha256"] = digest(state_doc, "stateSha256")
                with state_path.open("xb") as stream:
                    stream.write(canonical(state_doc))
                fit["checkpointFileSha256"] = state_doc["stateSha256"]
                fit["checkpointPath"] = str(state_path.relative_to(ROOT))
            fit_audits.append(fit)
            for key in month_keys:
                _, day, _ = key
                out = row_index[key]
                fr = features[h].get(day)
                old = base_index[key]["candidate"]
                if fr is None:
                    out.update({**mlp._fallback(old, "feature_unavailable_or_source_too_old"), "eligible": True})
                    continue
                old_direct = r106[key]["models"]["preserved106"]
                sarima = r106[key]["models"]["causalSarima"]
                # R reproduction is a correctness gate on every issued direct key.
                z = (np.asarray(fr["preserved106"], dtype=np.float64) - fit["scalerMean"]) / fit["scalerScale"]
                r_log = float(fit["ridgeCoefficientIncludingIntercept"][0] + z @ np.asarray(fit["ridgeCoefficientIncludingIntercept"][1:]))
                r_price = price_prediction(float(fr["p0"]), r_log)
                old_prediction = old_direct.get("prediction")
                if old_direct.get("status") == "issued" and (old_prediction is None or not ridge_reproduces(r_price, float(old_prediction))):
                    raise ArithmeticError(f"R106 direct prediction reproduction failed for {key}")
                r_error = abs(r_price - float(old_prediction)) if old_prediction is not None else None
                sarima_price = float(sarima["prediction"]) if sarima.get("status") == "issued" else None
                xs = z.reshape(1, -1)
                try:
                    a = hidden_design(xs, [np.asarray(v).reshape(s) for v,s in zip(fit["mlpCoefs"],fit["mlpCoefShapes"])],
                                      [np.asarray(v) for v in fit["mlpIntercepts"]])
                    m_log = float(a[0, 1:] @ np.asarray(fit["mlpCoefs"][1]).reshape(16, 1)[:, 0] + fit["mlpIntercepts"][1][0])
                    h_log = float(a[0] @ np.asarray(fit["hCoefficientIncludingIntercept"]))
                    m_price, h_price = price_prediction(float(fr["p0"]), m_log), price_prediction(float(fr["p0"]), h_log)
                    m_blend = .5 * (m_price + sarima_price) if sarima_price is not None else None
                    h_blend = .5 * (h_price + sarima_price) if sarima_price is not None else None
                except (ArithmeticError, FloatingPointError, OverflowError, ValueError) as exc:
                    out.update({**mlp._fallback(old, f"nonfinite_or_invalid_prediction:{type(exc).__name__}"),
                                "eligible": True, "training": train_audit, "fitId": fit["fitId"],
                                "featureVector":fr["preserved106"],"p0":fr["p0"],
                                "p0Date":fr["p0Date"].isoformat(),"RDirectPrediction":r_price,
                                "r106ReplayAbsError":r_error})
                    continue
                if phase == "validation":
                    saved_m_path = mlp.OUT / "confirmation-manifest.json"
                    saved_m_doc = json.loads(saved_m_path.read_text())
                    saved_m_row = mlp._idx(saved_m_doc["forecasts"])[key]
                    if saved_m_row.get("usedMLP"):
                        if not math.isclose(m_price, float(saved_m_row["mlpStandalonePrediction"]), rel_tol=1e-10, abs_tol=1e-8):
                            raise ArithmeticError(f"saved M inference replay failed for {key}")
                        if m_blend is not None and not math.isclose(m_blend, float(saved_m_row["prediction"]), rel_tol=1e-10, abs_tol=1e-8):
                            raise ArithmeticError(f"saved M blend replay failed for {key}")
                if sarima_price is None:
                    out.update({**mlp._fallback(old, "sarima_unavailable"), "eligible": True,
                                "fitId":fit["fitId"],"training":train_audit,
                                "featureVector":fr["preserved106"],"p0":fr["p0"],
                                "p0Date":fr["p0Date"].isoformat(),"RDirectPrediction":r_price,
                                "r106ReplayAbsError":r_error,"MStandalonePrediction":m_price,
                                "HStandalonePrediction":h_price})
                    continue
                out.update({"status": "issued", "prediction": h_blend, "reason": None,
                            "usedMLP": True, "usedH": True, "fallbackReason": None,
                            "training": train_audit, "fitId": fit["fitId"], "featureVector": fr["preserved106"],
                            "p0": fr["p0"], "p0Date": fr["p0Date"].isoformat(),
                            "RDirectPrediction": r_price, "RBlendPrediction": .5*(r_price+sarima_price),
                            "r106ReplayAbsError":r_error,
                            "MStandalonePrediction": m_price, "MBlendPrediction": m_blend,
                            "HStandalonePrediction": h_price,
                            "sarimaPrediction": sarima_price})
    planned = len(rows)
    doc = {"schemaVersion": 1, "phase": "forecast_only", "phaseName": src_phase,
           "rmhPhase": phase, "targetValuesAttached": False,
           "protocol": {"id": SCHEMA, "protocolSha256": frozen_protocol["protocolSha256"], "svdRankRule": RCOND_RULE,
                        "originStart": ORIGIN_START[phase].isoformat(),
                        "R": "Ridge(alpha=10, fit_intercept=True, solver=cholesky)",
                        "H": "unpenalized SVD on [1, ReLU(X_scaled W1+b1)]",
                        "M": mlp.MODEL_PARAMS, "newHiddenFits": fit_count_summary["newHiddenFits"],
                        "runtime": runtime,
                        "inputHashes": {"module": sha_bytes(Path(__file__).read_bytes()),
                                        "prices": mlp.file_sha(mlp.PRICES), "r106": mlp.file_sha(mlp.R106),
                                        "e106": mlp.file_sha(mlp.CONF_BASE if src_phase == "confirmation" else mlp.FINAL_BASE),
                                        "mlpManifest": mlp.file_sha(mlp.OUT / (src_phase + "-manifest.json")) if (mlp.OUT / (src_phase + "-manifest.json")).exists() else None}},
           "plannedCount": planned, "forecasts": rows, "fitAudits": fit_audits,
           "fitCounts": fit_count_summary["fitCounts"],
           "buildElapsedSeconds": time.monotonic() - phase_start,
           "infoTimeLimitation": "Pinned retrospective prices have observation dates but no proven row-level first-publication/availability timestamp; causal as-of-observation-date reconstruction only."}
    doc["manifestSha256"] = mlp.digest(doc)
    dest = output or _manifest_path(phase)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as stream:
        stream.write(mlp.canonical(doc))
    return doc


def verify_manifest(doc: dict[str, Any], phase: str, path: Path | None = None) -> None:
    """Fit-free verification of every key, fitted state and serialized prediction."""
    from . import preserved_feature_forecasts as pf
    protocol = verify_protocol()
    src_phase = "confirmation" if phase == "validation" else "final"
    mlp.validate(doc, src_phase)
    if doc.get("rmhPhase") != phase or doc.get("protocol", {}).get("protocolSha256") != protocol["protocolSha256"]:
        raise ValueError("RMH phase/protocol binding mismatch")
    counts=_fit_count_summary(phase)
    if (doc.get("protocol",{}).get("newHiddenFits")!=counts["newHiddenFits"]
            or doc.get("fitCounts")!=counts["fitCounts"]):
        raise ValueError("RMH phase fit counts do not match the frozen budget")
    if path is not None and doc.get("manifestSha256") != mlp.digest(doc):
        raise ValueError("RMH answer-free manifest digest mismatch")
    if any(r.get("eligible") is not True for r in doc.get("forecasts", [])):
        raise ValueError("RMH manifest dropped or disabled a fixed forecast key")
    mlp._check_pinned_inputs()
    expected_hashes = {"module":sha_bytes(Path(__file__).read_bytes()),"prices":mlp.file_sha(mlp.PRICES),
                       "r106":mlp.file_sha(mlp.R106),
                       "e106":mlp.file_sha(mlp.CONF_BASE if src_phase=="confirmation" else mlp.FINAL_BASE),
                       "mlpManifest":mlp.file_sha(mlp.OUT/(src_phase+"-manifest.json")) if (mlp.OUT/(src_phase+"-manifest.json")).exists() else None}
    if doc["protocol"].get("inputHashes") != expected_hashes:
        raise ValueError("RMH manifest source pins changed")
    saved_m = _fit_from_saved_m("validation") if phase == "validation" else {}
    base_doc = mlp._base_doc(src_phase)
    base = mlp._idx(base_doc["forecasts"])
    saved_m_rows = (mlp._idx(json.loads((mlp.OUT/"confirmation-manifest.json").read_text())["forecasts"])
                    if phase == "validation" else {})
    r106 = mlp._r106_index()
    series, features, by_h = mlp._make_features(src_phase, base, r106)
    expected_keys = set(mlp.keys(src_phase))
    forecast = mlp._idx(doc["forecasts"])
    if set(forecast) != expected_keys:
        raise ValueError("RMH forecast key coverage differs from fixed E106 grid")
    expected_ids = set()
    for h in mlp.HORIZONS:
        months = sorted({((k[1]-dt.timedelta(days=h)).year, (k[1]-dt.timedelta(days=h)).month)
                         for k in by_h[h]})
        expected_ids.update(f"garlic-h{h}-{y}-{m:02d}" for y,m in months)
    fits = {f.get("fitId"):f for f in doc.get("fitAudits", [])}
    if len(fits) != 30 or set(fits) != expected_ids:
        raise ValueError("RMH must contain exactly the 30 fixed monthly fit states")
    prices = json.loads(mlp.PRICES.read_text())
    direct, _ = pf.direct_horizon._observations(prices), None
    for fit_id, fit in fits.items():
        if fit.get("newHiddenFits")!=(0 if phase=="validation" else 1):
            raise ValueError(f"per-fit hidden-fit count changed for {fit_id}")
        h = int(fit_id.split("-")[1][1:])
        ymonth = tuple(int(v) for v in fit_id.rsplit("-",2)[-2:])
        month_keys = [k for k in by_h[h] if ((k[1]-dt.timedelta(days=h)).year,
                                              (k[1]-dt.timedelta(days=h)).month) == ymonth]
        frozen, origin = mlp._frozen_fit_audit(r106, h, month_keys)
        training = [t for t in series if t >= pf.online.narrow.TRAIN_START and t < origin and features[h].get(t) is not None]
        audit = pf._fit_audit(training, origin)
        pf._assert_training_support(audit, frozen, "garlic", h, ymonth)
        x = np.asarray([features[h][t]["preserved106"] for t in training], dtype=np.float64)
        y = np.asarray([math.log(series[t][0] / features[h][t]["p0"]) for t in training], dtype=np.float64)
        checks = {"trainingKeysSha256":sha_bytes(canonical([t.isoformat() for t in training])),
                  "trainingMatrixSha256":sha_bytes(x.tobytes(order="C")),
                  "trainingTargetSha256":sha_bytes(y.tobytes(order="C"))}
        if any(fit.get(k) != v for k,v in checks.items()) or fit.get("training") != audit or fit.get("fitOrigin") != origin.isoformat():
            raise ValueError(f"RMH fit training lineage mismatch for {fit_id}")
        mean = _finite_array(fit.get("scalerMean"), "stored scaler mean", 1)
        scale = _finite_array(fit.get("scalerScale"), "stored scaler scale", 1)
        coefs = [np.asarray(v,dtype=np.float64).reshape(s) for v,s in zip(fit["mlpCoefs"],fit["mlpCoefShapes"])]
        intercepts = [np.asarray(v,dtype=np.float64) for v in fit["mlpIntercepts"]]
        if mean.shape != (106,) or scale.shape != (106,) or np.any(scale <= 0):
            raise ValueError(f"bad scaler state for {fit_id}")
        xs = (x-mean)/scale
        a = hidden_design(xs, coefs, intercepts)
        beta_h = _finite_array(fit.get("hCoefficientIncludingIntercept"), "stored H coefficients", 1)
        beta_r = _finite_array(fit.get("ridgeCoefficientIncludingIntercept"), "stored R coefficients", 1)
        if beta_h.shape != (17,) or beta_r.shape != (107,):
            raise ValueError(f"bad R/H coefficient shape for {fit_id}")
        r_prediction = beta_r[0] + xs @ beta_r[1:]
        m_prediction = (a[:,1:] @ coefs[1][:,0]) + intercepts[1][0]
        h_prediction = a @ beta_h
        lr, lm, lh = (float(np.mean((y-v)**2)) for v in (r_prediction,m_prediction,h_prediction))
        tol = 1e-10 * max(1.0,lr,lm,lh)
        if any(abs(fit[k]-v)>tol for k,v in (("L_R",lr),("L_M",lm),("L_H",lh))):
            raise ValueError(f"stored training losses do not replay for {fit_id}")
        # Replay the one frozen SVD from its persisted orthogonal factors; never refit H here.
        beta_expected,u,singular,vt = _saved_svd(a,y,fit)
        keep = singular > np.finfo(np.float64).eps * max(a.shape) * singular[0]
        if not np.allclose(beta_h,beta_expected,rtol=1e-9,atol=1e-10):
            raise ValueError(f"saved H coefficients are not the fixed SVD solution for {fit_id}")
        projected_m=u[:,keep]@(u[:,keep].T@m_prediction)
        projected_r=u[:,keep]@(u[:,keep].T@r_prediction)
        projected_target=u[:,keep]@(u[:,keep].T@y)
        projected_h=a@beta_expected
        projection_error=float(np.max(np.abs(projected_target-projected_h)))
        if projection_error>1e-9 or abs(float(fit.get("hProjectionMaxAbsError",math.inf))-projection_error)>1e-9:
            raise ValueError(f"H target projection does not replay for {fit_id}")
        dm=float(np.mean((m_prediction-projected_m)**2)); dr=float(np.mean((r_prediction-projected_r)**2))
        upper_m=(math.sqrt(lm)+math.sqrt(dm))**2; upper_r=(math.sqrt(lr)+math.sqrt(dr))**2
        slack=LOSS_BOUND_ABS+LOSS_BOUND_REL*max(1.0,lr,lm,lh,upper_m,upper_r)
        if (abs(float(fit.get("D_M",math.inf))-dm)>tol or abs(float(fit.get("D_R",math.inf))-dr)>tol
                or lh>upper_m+slack or lh>upper_r+slack):
            raise ValueError(f"RMS reconstruction bound or output gap mismatch for {fit_id}")
        cp_path=fit.get("checkpointPath")
        expected_cp=Path("outputs/experiments/e106-rmh-20261011/.fit-states")/phase/(fit_id+".json")
        if cp_path != str(expected_cp): raise ValueError(f"unsafe/wrong checkpoint binding for {fit_id}")
        checkpoint=json.loads((ROOT/expected_cp).read_text())
        manifest_fit={k:v for k,v in fit.items() if k not in ("checkpointPath","checkpointFileSha256")}
        resumed=_verify_checkpoint(checkpoint,phase=phase,fit_id=fit_id,fit_origin=origin,
            training=training,x=x,y=y,train_audit=audit,protocol=protocol,saved_m=saved_m)
        if (checkpoint.get("stateSha256")!=fit.get("checkpointFileSha256")
                or resumed!=manifest_fit):
            raise ValueError(f"fit checkpoint does not bind the displayed fitted state for {fit_id}")
        if phase=="validation":
            source=saved_m.get(fit_id)
            if source is None: raise ValueError(f"missing pinned M fit {fit_id}")
            if (mean.tolist()!=source["scalerMean"] or scale.tolist()!=source["scalerScale"]
                    or fit["mlpCoefShapes"]!=source["coefShapes"] or fit["mlpCoefs"]!=source["coefs"]
                    or fit["mlpIntercepts"]!=source["intercepts"]):
                raise ValueError(f"saved M state differs from immutable MLP manifest {fit_id}")
        normal_error = float(np.max(np.abs(a.T @ (y-h_prediction))))
        if normal_error > 1e-8 * max(1.0,float(np.max(np.abs(a.T@y)))):
            raise ValueError(f"H least-squares normal-equation check failed for {fit_id}")
    for key,row in forecast.items():
        base_row=base[key]
        if row["product"] != "garlic":
            for field in ("status","prediction","reason"):
                if row.get(field) != base_row["candidate"].get(field):
                    raise ValueError(f"unchanged non-garlic output changed at {key}")
            continue
        used_h=_is_h_output(row)
        if not used_h:
            _assert_exact_fallback(row,base_row,key)
        elif row.get("status")!="issued":
            raise ValueError(f"H output is not issued at {key}")
        feature_for_key=features[key[2]].get(key[1])
        if feature_for_key is not None:
            origin=key[1]-dt.timedelta(days=key[2])
            expected_fit_id=f"garlic-h{key[2]}-{origin.year}-{origin.month:02d}"
            if row.get("fitId")!=expected_fit_id:
                raise ValueError(f"fixed fit assignment is missing or changed at {key}")
        elif row.get("fitId") is not None:
            raise ValueError(f"forecast claims a fit without an available feature row at {key}")
        if not row.get("fitId"):
            if used_h: raise ValueError(f"H output has no fit state at {key}")
            continue
        fit=fits.get(row.get("fitId"))
        if fit is None: raise ValueError(f"forecast lacks fit state at {key}")
        d=key[1]; fr=features[key[2]].get(d)
        if fr is None or row.get("featureVector") != fr["preserved106"]:
            raise ValueError(f"feature row does not replay at {key}")
        if not math.isclose(row.get("p0"),fr["p0"],rel_tol=0,abs_tol=1e-12) or row.get("p0Date") != fr["p0Date"].isoformat():
            raise ValueError(f"p0 does not replay at {key}")
        z=(np.asarray(fr["preserved106"],dtype=np.float64)-fit["scalerMean"])/fit["scalerScale"]
        r_price=price_prediction(float(fr["p0"]),float(fit["ridgeCoefficientIncludingIntercept"][0]+z@np.asarray(fit["ridgeCoefficientIncludingIntercept"][1:])))
        old_r=r106[key]["models"]["preserved106"]
        if old_r.get("status")!="issued" or old_r.get("prediction") is None:
            raise ValueError(f"original R106 prediction unavailable at {key}")
        replay_gap=_assert_r_replay(row,r_price,float(old_r["prediction"]),key)
        a=hidden_design(z.reshape(1,-1),[np.asarray(v).reshape(s) for v,s in zip(fit["mlpCoefs"],fit["mlpCoefShapes"])],
                        [np.asarray(v) for v in fit["mlpIntercepts"]])
        m_log=float(a[0,1:]@np.asarray(fit["mlpCoefs"][1]).reshape(16) + fit["mlpIntercepts"][1][0])
        h_log=float(a[0]@fit["hCoefficientIncludingIntercept"])
        try: m=price_prediction(float(fr["p0"]),m_log)
        except (ArithmeticError,FloatingPointError,OverflowError,ValueError): m=None
        try: h_price=price_prediction(float(fr["p0"]),h_log)
        except (ArithmeticError,FloatingPointError,OverflowError,ValueError): h_price=None
        if phase=="validation":
            mrow=saved_m_rows.get(key)
            if mrow is None: raise ValueError(f"missing saved M forecast at {key}")
            if mrow.get("usedMLP"):
                if m is None: raise ValueError(f"V M forecast became invalid at {key}")
                if not math.isclose(m,float(mrow["mlpStandalonePrediction"]),rel_tol=1e-10,abs_tol=1e-8):
                    raise ValueError(f"V saved M forecast row does not replay at {key}")
        sarima=r106[key]["models"]["causalSarima"]
        if not used_h:
            direct_valid=m is not None and h_price is not None
            if direct_valid and sarima.get("status")!="issued":
                if row.get("fallbackReason")!="sarima_unavailable":
                    raise ValueError(f"SARIMA fallback cause changed at {key}")
            elif direct_valid:
                raise ValueError(f"unexplained H fallback despite finite direct forecasts at {key}")
            elif not str(row.get("fallbackReason","")).startswith("nonfinite_or_invalid_prediction:"):
                raise ValueError(f"invalid-prediction fallback cause changed at {key}")
            for field,value in (("MStandalonePrediction",m),("HStandalonePrediction",h_price)):
                if row.get(field) is not None and (value is None or not math.isclose(
                        float(row[field]),value,rel_tol=1e-10,abs_tol=1e-8)):
                    raise ValueError(f"serialized fallback diagnostic {field} fails replay at {key}")
            continue
        if sarima.get("status")!="issued":
            raise ValueError(f"issued H lacks frozen SARIMA at {key}")
        if m is None or h_price is None: raise ValueError(f"issued H model prediction became invalid at {key}")
        sarima_value=float(sarima["prediction"])
        if phase=="validation" and mrow.get("usedMLP"):
            m_blend=.5*(m+sarima_value)
            if not math.isclose(m_blend,float(mrow["prediction"]),rel_tol=1e-10,abs_tol=1e-8):
                raise ValueError(f"V saved M blend row does not replay at {key}")
        expected={"RDirectPrediction":r_price,"MStandalonePrediction":m,"MBlendPrediction":.5*(m+sarima_value),
                  "RBlendPrediction":.5*(r_price+sarima_value),"HStandalonePrediction":h_price,
                  "prediction":.5*(h_price+sarima_value),
                  "sarimaPrediction":sarima_value,"r106ReplayAbsError":replay_gap}
        for field,value in expected.items():
            if not math.isclose(float(row.get(field)),value,rel_tol=1e-10,abs_tol=1e-8):
                raise ValueError(f"stored forecast {field} fails source replay at {key}")
    return None


def _verify_validation_gate() -> dict[str, Any]:
    """Recompute the V gate from its validated manifest before authorizing T."""
    report_path=OUT/"validation-report.json"
    manifest_path=_manifest_path("validation")
    report=json.loads(report_path.read_text())
    manifest=json.loads(manifest_path.read_text())
    if report.get("reportSha256")!=digest(report,"reportSha256"):
        raise ValueError("V report digest mismatch")
    if report.get("manifestFileSha256")!=mlp.file_sha(manifest_path):
        raise ValueError("V report does not bind the saved V manifest")
    verify_manifest(manifest,"validation",manifest_path)
    baseline=mlp._idx(mlp._base_doc("confirmation")["forecasts"])
    prices=json.loads(mlp.PRICES.read_text())
    from . import preserved_feature_forecasts as pf
    actuals=pf.direct_horizon._observations(prices)
    r106=mlp._r106_index()
    by_h={h:[] for h in mlp.HORIZONS}
    extra=[]
    replay_gaps=[]
    replay_pass=[]
    for row in manifest["forecasts"]:
        if row["product"]!="garlic": continue
        if row.get("r106ReplayAbsError") is not None:
            gap=float(row["r106ReplayAbsError"])
            replay_gaps.append(gap)
            key=(row["product"],dt.date.fromisoformat(row["target"]),row["horizon"])
            old_value=r106[key]["models"]["preserved106"].get("prediction")
            replay_pass.append(old_value is not None and gap<=R_REPRO_ABS+R_REPRO_REL*abs(float(old_value)))
        key=(row["product"],dt.date.fromisoformat(row["target"]),row["horizon"])
        e=baseline[key]["candidate"]
        if e.get("status")=="issued" and (row.get("fallbackReason") or row.get("status")!="issued"):
            extra.append(key)
        act=actuals.get("garlic",{}).get(key[1])
        cp=row.get("prediction") if row.get("status")=="issued" else e.get("prediction")
        if act is not None and e.get("prediction") is not None and cp is not None:
            by_h[key[2]].append((float(cp),float(e["prediction"]),float(act[0])))
    if any(not by_h[h] for h in mlp.HORIZONS):
        raise ValueError("V cannot authorize T with an empty fixed horizon cell")
    cm=math.fsum(math.fsum(abs(a-y) for a,e,y in by_h[h])/len(by_h[h]) for h in mlp.HORIZONS)/10
    em=math.fsum(math.fsum(abs(e-y) for a,e,y in by_h[h])/len(by_h[h]) for h in mlp.HORIZONS)/10
    cp=math.fsum(100*math.fsum(abs(a-y)/y for a,e,y in by_h[h])/len(by_h[h]) for h in mlp.HORIZONS)/10
    ep=math.fsum(100*math.fsum(abs(e-y)/y for a,e,y in by_h[h])/len(by_h[h]) for h in mlp.HORIZONS)/10
    minimum=min(len(v) for v in by_h.values())
    expected_issued=sum(1 for k,v in baseline.items() if k[0]=="garlic" and v["candidate"].get("status")=="issued")
    got_issued=sum(1 for r in manifest["forecasts"] if r["product"]=="garlic" and r.get("status")=="issued")
    checks={"deltaMapeNegative":(cp-ep)<0,"deltaMaeNonpositive":(cm-em)<=0,
            "allEligibleE106IssuedKeysRetained":got_issued>=expected_issued and not extra,
            "atLeast30PairsAtEveryHorizon":minimum>=30,
            "r106ReproductionPassed":bool(replay_gaps) and all(replay_pass),
            "minimumSquaresChecksPassed":len(manifest["fitAudits"])==30}
    checks["passes"]=all(checks.values())
    old=report.get("primaryDecision",{})
    if checks!=old:
        raise ValueError("V report gate does not reproduce from its bound manifest")
    expected_macro={"candidateMae":cm,"E106Mae":em,"candidateMape":cp,"E106Mape":ep,
                    "deltaMae":cm-em,"deltaMape":cp-ep}
    for k,v in expected_macro.items():
        if not math.isclose(float(report["primaryGarlic10h"][k]),v,rel_tol=0,abs_tol=1e-10):
            raise ValueError(f"V report primary {k} does not replay")
    if not checks["passes"]:
        raise ValueError("V primary gate failed; T is not authorized")
    return report


def score_phase(phase: str, manifest_path: Path | None = None) -> dict[str, Any]:
    """Attach pinned truths once, complete the fixed statistics, and save report."""
    from . import equal_info_online_score as stats
    from . import preserved_feature_forecasts as pf
    from .research_progress_diagnostic import circular_block_bootstrap
    if phase not in PHASES:
        raise ValueError("phase must be validation or final")
    src_phase = "confirmation" if phase == "validation" else "final"
    path = manifest_path or _manifest_path(phase)
    doc = json.loads(path.read_text())
    verify_manifest(doc, phase, path)
    if doc.get("rmhPhase") != phase or doc.get("protocol", {}).get("id") != SCHEMA:
        raise ValueError("wrong RMH phase manifest")
    mlp._check_pinned_inputs()
    baseline = mlp._idx(mlp._base_doc(src_phase)["forecasts"])
    prices = json.loads(mlp.PRICES.read_text())
    actuals = pf.direct_horizon._observations(prices)
    r106 = mlp._r106_index()
    pairs_by_h: dict[int, list[dict[str, Any]]] = {h: [] for h in mlp.HORIZONS}
    hmetrics, r_direct, m_direct, h_direct = [], [], [], []
    r_blends, m_blends, h_blends = [], [], []
    common_blends = {h:[] for h in mlp.HORIZONS}
    common_direct = {h:[] for h in mlp.HORIZONS}
    direct_log_errors = {h: {m: [] for m in ("R","M","H")} for h in mlp.HORIZONS}
    monthly_garlic = {}
    monthly_components: dict[tuple[str,str,int],list[tuple[float,float]]] = {}
    coverage = {h: {"eligible": 0, "E106Issued": 0, "HIssued": 0, "fallback": 0} for h in mlp.HORIZONS}
    fallback_additions = []
    candidate_all = []
    truth_rows = []
    replay_gaps = []
    replay_pass = []
    for row in doc["forecasts"]:
        if not row.get("eligible"):
            continue
        key = (row["product"], dt.date.fromisoformat(row["target"]), row["horizon"])
        product, target, horizon = key
        base = baseline[key]["candidate"]
        target_row = actuals.get(product, {}).get(target)
        base_pred = base.get("prediction")
        cand_pred = row.get("prediction") if row.get("status") == "issued" else base_pred
        if product == "garlic" and base.get("status") == "issued" and row.get("status") != "issued":
            fallback_additions.append({"product": product, "target": target.isoformat(), "horizon": horizon,
                                       "reason": row.get("fallbackReason")})
        if product == "garlic" and base.get("status") == "issued" and row.get("status") == "issued" and row.get("fallbackReason"):
            fallback_additions.append({"product": product, "target": target.isoformat(), "horizon": horizon,
                                       "reason": row.get("fallbackReason")})
        if product == "garlic":
            if row.get("r106ReplayAbsError") is not None:
                gap=float(row["r106ReplayAbsError"])
                replay_gaps.append(gap)
                old_value=r106[key]["models"]["preserved106"].get("prediction")
                replay_pass.append(old_value is not None and gap<=R_REPRO_ABS+R_REPRO_REL*abs(float(old_value)))
            coverage[horizon]["eligible"] += 1
            coverage[horizon]["E106Issued"] += int(base.get("status") == "issued")
            coverage[horizon]["HIssued"] += int(_is_h_output(row))
            coverage[horizon]["fallback"] += int(bool(row.get("fallbackReason")))
        if target_row is None or base_pred is None or cand_pred is None:
            continue
        actual = float(target_row[0])
        if not math.isfinite(actual) or actual <= 0:
            continue
        item = {"product": product, "target": target.isoformat(), "horizon": horizon,
                "actual": actual, "candidate": float(cand_pred), "E106": float(base_pred)}
        candidate_all.append(item)
        truth_rows.append({"product": product, "target": target.isoformat(), "horizon": horizon,
                           "actual": actual, "prediction": float(cand_pred), "E106": float(base_pred)})
        if product == "garlic":
            pairs_by_h[horizon].append(item)
            month = target.strftime("%Y-%m")
            monthly_garlic.setdefault((month, horizon), []).append(item)
            def add_component(name, prediction):
                if prediction is not None:
                    monthly_components.setdefault((name,month,horizon),[]).append((float(prediction),actual))
            add_component("R",row.get("RDirectPrediction"))
            add_component("M",row.get("MStandalonePrediction"))
            add_component("H",row.get("HStandalonePrediction"))
            add_component("RBlend",row.get("RBlendPrediction"))
            add_component("MBlend",row.get("MBlendPrediction"))
            add_component("HBlend",row.get("prediction") if _is_h_output(row) else None)
            direct_common=_common_component_row(row,
                ("RDirectPrediction","MStandalonePrediction","HStandalonePrediction"),
                ("R","M","H"),month=month,actual=actual)
            if direct_common: common_direct[horizon].append(direct_common)
            if _is_h_output(row):
                blend_common=_common_component_row(row,
                    ("RBlendPrediction","MBlendPrediction","prediction"),("R","M","H"),
                    month=month,actual=actual)
                if blend_common: common_blends[horizon].append(blend_common)
            old_direct = r106[key]["models"]["preserved106"].get("prediction")
            if old_direct is not None and row.get("RDirectPrediction") is not None:
                r_direct.append((horizon, abs(float(row["RDirectPrediction"])-actual), abs(float(old_direct)-actual),
                                 abs(float(row["RDirectPrediction"])-actual)/actual,
                                 abs(float(old_direct)-actual)/actual))
                direct_log_errors[horizon]["R"].append(math.log(float(row["RDirectPrediction"])/actual)**2)
            if row.get("MStandalonePrediction") is not None:
                m_direct.append((horizon, abs(float(row["MStandalonePrediction"])-actual),
                                 100*abs(float(row["MStandalonePrediction"])-actual)/actual,
                                 abs(float(old_direct)-actual)/actual if old_direct is not None else None))
                direct_log_errors[horizon]["M"].append(math.log(float(row["MStandalonePrediction"])/actual)**2)
            if row.get("HStandalonePrediction") is not None:
                h_direct.append((horizon, abs(float(row["HStandalonePrediction"])-actual),
                                 100*abs(float(row["HStandalonePrediction"])-actual)/actual))
                direct_log_errors[horizon]["H"].append(math.log(float(row["HStandalonePrediction"])/actual)**2)
            for col, target_list in (("RBlendPrediction",r_blends),("MBlendPrediction",m_blends)):
                if row.get(col) is not None:
                    target_list.append((horizon,abs(float(row[col])-actual),100*abs(float(row[col])-actual)/actual))
            if _is_h_output(row):
                h_blends.append((horizon,abs(float(row["prediction"])-actual),100*abs(float(row["prediction"])-actual)/actual))
    for h in mlp.HORIZONS:
        xs = pairs_by_h[h]
        def avg(values): return math.fsum(values)/len(values) if values else None
        mae_c = avg([abs(x["candidate"]-x["actual"]) for x in xs])
        mae_b = avg([abs(x["E106"]-x["actual"]) for x in xs])
        mape_c = avg([100*abs(x["candidate"]-x["actual"])/x["actual"] for x in xs])
        mape_b = avg([100*abs(x["E106"]-x["actual"])/x["actual"] for x in xs])
        hmetrics.append({"horizon": h, "n": len(xs), "uniqueTargets": len({x["target"] for x in xs}),
                         "candidateMae": mae_c, "E106Mae": mae_b, "deltaMae": mae_c-mae_b if xs else None,
                         "candidateMape": mape_c, "E106Mape": mape_b, "deltaMape": mape_c-mape_b if xs else None})
    family = []
    for h, xs in pairs_by_h.items():
        for loss in ("ae", "ape"):
            diff = {}
            for x in xs:
                e = abs(x["candidate"]-x["actual"])-abs(x["E106"]-x["actual"])
                if loss == "ape": e /= x["actual"]
                diff[dt.date.fromisoformat(x["target"])] = e
            for bw in (9, 28):
                result=_label_dm_units(stats._dm(diff,bw),loss)
                family.append({"horizon": h, "loss": loss, "bandwidth": bw, **result})
    for bw in (9, 28):
        stats._holm([x for x in family if x["bandwidth"] == bw])
    dm_summary = []
    for bw in (9, 28):
        xs = [x for x in family if x["bandwidth"] == bw]
        dm_summary.append({"bandwidth": bw, "improvement": sum(x.get("holmSignificant") and x.get("meanLossDifference",0)<0 for x in xs),
                           "worsening": sum(x.get("holmSignificant") and x.get("meanLossDifference",0)>0 for x in xs),
                           "inconclusive": sum(not x.get("holmSignificant",False) or x.get("status") != "tested" for x in xs),
                           "unsupported": sum(x.get("status") != "tested" for x in xs), "familySize": 20})
    n_min = min((m["n"] for m in hmetrics), default=0)
    complete_10h = len(hmetrics)==10 and all(x["n"]>0 for x in hmetrics)
    macro = {k:(math.fsum(x[k] for x in hmetrics)/10 if complete_10h else None)
             for k in ("candidateMae","E106Mae","candidateMape","E106Mape")}
    macro["deltaMae"] = macro["candidateMae"]-macro["E106Mae"] if complete_10h else None
    macro["deltaMape"] = macro["candidateMape"]-macro["E106Mape"] if complete_10h else None
    base_keys = mlp._idx(mlp._base_doc(src_phase)["forecasts"])
    expected_issued = sum(1 for k,v in base_keys.items() if k[0] == "garlic" and v["candidate"].get("status") == "issued")
    observed_issued = sum(1 for r in doc["forecasts"] if r["product"] == "garlic" and r.get("eligible") and r.get("status") == "issued")
    decision = {"deltaMapeNegative": macro["deltaMape"] is not None and macro["deltaMape"] < 0,
                "deltaMaeNonpositive": macro["deltaMae"] is not None and macro["deltaMae"] <= 0,
                "allEligibleE106IssuedKeysRetained": observed_issued >= expected_issued and not fallback_additions,
                "atLeast30PairsAtEveryHorizon": n_min >= 30,
                "r106ReproductionPassed": bool(replay_gaps) and all(replay_pass),
                "minimumSquaresChecksPassed": len(doc["fitAudits"]) == 30 and all(
                    f["L_H"] <= f["HUpperBoundFromM"] + f["boundSlack"] and
                    f["L_H"] <= f["HUpperBoundFromR"] + f["boundSlack"] for f in doc["fitAudits"])}
    decision["passes"] = all(decision.values())
    boot = {}
    garlic_rows = [x for x in candidate_all if x["product"] == "garlic"]
    four_crop_rows=[x for x in candidate_all if x["product"] in FOUR_CROP_PRODUCTS]
    portfolio_cells,expected_cells,observed_cells=_four_crop_cells(candidate_all)
    complete_40=expected_cells==observed_cells
    complete_garlic=all(pairs_by_h[h] for h in mlp.HORIZONS)
    for block in (14, 28):
        boot[str(block)] = {}
        for metric in ("ape", "ae"):
            args = dict(a="candidate", b="E106", metric=metric, block=block,
                        replicates=2000, seed=20261011, calendar_start=PHASES[phase][0],
                        calendar_end=PHASES[phase][1])
            boot[str(block)]["fortyCell_" + metric] = circular_block_bootstrap(
                four_crop_rows, a="candidate", b="E106", metric=metric, block=block,
                replicates=2000, seed=20261011, calendar_start=PHASES[phase][0],
                calendar_end=PHASES[phase][1]) if complete_40 else {"interval95":None,"status":"unsupported_missing_fixed_cells","missingCells":sorted([list(c) for c in expected_cells-observed_cells])}
            boot[str(block)]["garlic10H_" + metric] = circular_block_bootstrap(garlic_rows, **args) if complete_garlic else {"interval95":None,"status":"unsupported_missing_fixed_horizon_cells"}
    r_mae = math.fsum(x[1] for x in r_direct)/len(r_direct) if r_direct else None
    r106_mae = math.fsum(x[2] for x in r_direct)/len(r_direct) if r_direct else None
    r_mape = 100*math.fsum(x[3] for x in r_direct)/len(r_direct) if r_direct else None
    r106_mape = 100*math.fsum(x[4] for x in r_direct)/len(r_direct) if r_direct else None
    m_mae = math.fsum(x[1] for x in m_direct)/len(m_direct) if m_direct else None
    m_mape = math.fsum(x[2] for x in m_direct)/len(m_direct) if m_direct else None
    m_r106_mape = 100*math.fsum(x[3] for x in m_direct if x[3] is not None)/sum(x[3] is not None for x in m_direct) if any(x[3] is not None for x in m_direct) else None
    h_mae = math.fsum(x[1] for x in h_direct)/len(h_direct) if h_direct else None
    h_mape = math.fsum(x[2] for x in h_direct)/len(h_direct) if h_direct else None
    monthly_metrics=[]
    for (month,h), xs in sorted(monthly_garlic.items()):
        monthly_metrics.append({"month":month,"horizon":h,"n":len(xs),
            "candidateMae":math.fsum(abs(x["candidate"]-x["actual"]) for x in xs)/len(xs),
            "E106Mae":math.fsum(abs(x["E106"]-x["actual"]) for x in xs)/len(xs),
                "candidateMape":100*math.fsum(abs(x["candidate"]-x["actual"])/x["actual"] for x in xs)/len(xs),
                "E106Mape":100*math.fsum(abs(x["E106"]-x["actual"])/x["actual"] for x in xs)/len(xs)})
    monthly_component_metrics=[]
    for (name,month,h),xs in sorted(monthly_components.items()):
        monthly_component_metrics.append({"model":name,"month":month,"horizon":h,"n":len(xs),
            "logMse":math.fsum(math.log(p/y)**2 for p,y in xs)/len(xs),
            "mae":math.fsum(abs(p-y) for p,y in xs)/len(xs),
            "mape":100*math.fsum(abs(p-y)/y for p,y in xs)/len(xs)})
    direct_by_h=[]
    common_blend_by_h=[]
    common_direct_by_h=[]
    for h in mlp.HORIZONS:
        rr=[x for x in r_direct if x[0]==h]
        mm=[x for x in m_direct if x[0]==h]
        hh=[x for x in h_direct if x[0]==h]
        cb=common_blends[h]
        cd=common_direct[h]
        common_direct_by_h.append({"horizon":h,"n":len(cd),**{
            model:{"logMse":math.fsum(math.log(x[model]/x["actual"])**2 for x in cd)/len(cd) if cd else None,
                   "mae":math.fsum(abs(x[model]-x["actual"]) for x in cd)/len(cd) if cd else None,
                   "mape":100*math.fsum(abs(x[model]-x["actual"])/x["actual"] for x in cd)/len(cd) if cd else None}
            for model in ("R","M","H")}})
        common_blend_by_h.append({"horizon":h,"n":len(cb),**{
            model:{"mae":math.fsum(abs(x[model]-x["actual"]) for x in cb)/len(cb) if cb else None,
                   "mape":100*math.fsum(abs(x[model]-x["actual"])/x["actual"] for x in cb)/len(cb) if cb else None}
            for model in ("R","M","H")}})
        direct_by_h.append({"horizon":h,
            "R":{"n":len(rr),"logMse":math.fsum(direct_log_errors[h]["R"])/len(direct_log_errors[h]["R"]) if direct_log_errors[h]["R"] else None,"mae":math.fsum(x[1] for x in rr)/len(rr) if rr else None,"mape":100*math.fsum(x[3] for x in rr)/len(rr) if rr else None},
            "M":{"n":len(mm),"logMse":math.fsum(direct_log_errors[h]["M"])/len(direct_log_errors[h]["M"]) if direct_log_errors[h]["M"] else None,"mae":math.fsum(x[1] for x in mm)/len(mm) if mm else None,"mape":math.fsum(x[2] for x in mm)/len(mm) if mm else None},
            "H":{"n":len(hh),"logMse":math.fsum(direct_log_errors[h]["H"])/len(direct_log_errors[h]["H"]) if direct_log_errors[h]["H"] else None,"mae":math.fsum(x[1] for x in hh)/len(hh) if hh else None,"mape":math.fsum(x[2] for x in hh)/len(hh) if hh else None},
            "RBlend":{"n":sum(x[0]==h for x in r_blends),"mae":math.fsum(x[1] for x in r_blends if x[0]==h)/sum(x[0]==h for x in r_blends) if any(x[0]==h for x in r_blends) else None,"mape":math.fsum(x[2] for x in r_blends if x[0]==h)/sum(x[0]==h for x in r_blends) if any(x[0]==h for x in r_blends) else None},
            "MBlend":{"n":sum(x[0]==h for x in m_blends),"mae":math.fsum(x[1] for x in m_blends if x[0]==h)/sum(x[0]==h for x in m_blends) if any(x[0]==h for x in m_blends) else None,"mape":math.fsum(x[2] for x in m_blends if x[0]==h)/sum(x[0]==h for x in m_blends) if any(x[0]==h for x in m_blends) else None},
            "HBlend":{"n":sum(x[0]==h for x in h_blends),
                      "mae":math.fsum(x[1] for x in h_blends if x[0]==h)/sum(x[0]==h for x in h_blends) if any(x[0]==h for x in h_blends) else None,
                      "mape":math.fsum(x[2] for x in h_blends if x[0]==h)/sum(x[0]==h for x in h_blends) if any(x[0]==h for x in h_blends) else None}})
    complete_blend_h=all(common_blends[h] for h in mlp.HORIZONS)
    common_blend_primary={model:{"mae":math.fsum(v[model]["mae"] for v in common_blend_by_h)/10 if complete_blend_h else None,
                                "mape":math.fsum(v[model]["mape"] for v in common_blend_by_h)/10 if complete_blend_h else None}
                         for model in ("R","M","H")}
    common_blend_primary["nMinHorizon"]=min((len(common_blends[h]) for h in mlp.HORIZONS),default=0)
    common_blend_primary["complete10H"]=complete_blend_h
    complete_direct_h=all(common_direct[h] for h in mlp.HORIZONS)
    common_direct_primary={model:{metric:math.fsum(v[model][metric] for v in common_direct_by_h)/10
                                  if complete_direct_h else None
                                  for metric in ("logMse","mae","mape")}
                           for model in ("R","M","H")}
    common_direct_primary["nMinHorizon"]=min((len(common_direct[h]) for h in mlp.HORIZONS),default=0)
    common_direct_primary["complete10H"]=complete_direct_h
    monthly_common={}
    for h,rows in common_blends.items():
        for x in rows: monthly_common.setdefault((x["month"],h),[]).append(x)
    monthly_common_metrics=[]
    for (month,h),rows in sorted(monthly_common.items()):
        monthly_common_metrics.append({"month":month,"horizon":h,"n":len(rows),**{
            model:{"mae":math.fsum(abs(x[model]-x["actual"]) for x in rows)/len(rows),
                   "mape":100*math.fsum(abs(x[model]-x["actual"])/x["actual"] for x in rows)/len(rows)}
            for model in ("R","M","H")}})
    monthly_common_direct={}
    for h,rows in common_direct.items():
        for x in rows: monthly_common_direct.setdefault((x["month"],h),[]).append(x)
    monthly_common_direct_metrics=[]
    for (month,h),rows in sorted(monthly_common_direct.items()):
        monthly_common_direct_metrics.append({"month":month,"horizon":h,"n":len(rows),**{
            model:{"logMse":math.fsum(math.log(x[model]/x["actual"])**2 for x in rows)/len(rows),
                   "mae":math.fsum(abs(x[model]-x["actual"]) for x in rows)/len(rows),
                   "mape":100*math.fsum(abs(x[model]-x["actual"])/x["actual"] for x in rows)/len(rows)}
            for model in ("R","M","H")}})
    portfolio = _portfolio_macro(portfolio_cells,complete_40)
    report = {"schema": SCHEMA, "phase": phase, "manifestFileSha256": mlp.file_sha(path),
              "targetValuesAttached": True, "pairedRows": sum(x["n"] for x in hmetrics),
              "primaryGarlic10h": macro, "primaryDecision": decision,
              "garlicHorizonMetrics": hmetrics, "dm": family,
              "dmSummary": dm_summary, "coverageByHorizon": coverage,
              "monthlyGarlicMetrics": monthly_metrics,
              "monthlyDirectAndBlendMetrics": monthly_component_metrics,
              "monthlyCommonBlendMetrics":monthly_common_metrics,
              "monthlyCommonDirectMetrics":monthly_common_direct_metrics,
              "r106Replay": {"n":len(replay_gaps),"maxAbsPriceDifference":max(replay_gaps) if replay_gaps else None,
                             "absoluteTolerance":R_REPRO_ABS,"relativeTolerance":R_REPRO_REL},
              "directAndBlendByHorizon": direct_by_h,
              "commonBlendByHorizon":common_blend_by_h,
              "commonDirectByHorizon":common_direct_by_h,
              "commonBlendPrimary10h":common_blend_primary,
              "commonDirectPrimary10h":common_direct_primary,
              "fortyCellAuxiliary": {"cells": len(portfolio_cells),"completeFixedGrid":complete_40,
                                     "unsupportedReason":None if complete_40 else "one or more fixed product×horizon cells have no positive paired target",
                                     **portfolio,
                                     "interpretation": "auxiliary; garlic is only one quarter of the equal-cell mean"},
              "fortyCellByProductHorizon": portfolio_cells,
              "dmProtocol": {"familySize": 20, "bandwidths": [9, 28], "holmAlpha": .05,
                             "losses": ["ae", "ape"], "minimumPairs": 30},
              "fortyCellAndGarlicBlockBootstrap": boot,
              "directComponentDiagnostics": {"R106Refit": {"n": len(r_direct), "mae": r_mae,
                  "originalR106Mae": r106_mae, "mape": r_mape, "originalR106Mape": r106_mape},
                  "MStandalone": {"n": len(m_direct), "mae": m_mae, "mape": m_mape,
                                  "originalR106Mape": m_r106_mape},
                  "HStandalone": {"n": len(h_direct), "mae": h_mae, "mape": h_mape}},
              "fallbackAdditions": fallback_additions,
              "fitAuditSummary": {"fitCount": len(doc["fitAudits"]),
                                  "L_R_mean": math.fsum(f["L_R"] for f in doc["fitAudits"])/30,
                                  "L_M_mean": math.fsum(f["L_M"] for f in doc["fitAudits"])/30,
                                  "L_H_mean": math.fsum(f["L_H"] for f in doc["fitAudits"])/30,
                                  "outputGapMean": math.fsum(f["outputGap"] for f in doc["fitAudits"])/30,
                                  "representationGapMean": math.fsum(f["representationGap"] for f in doc["fitAudits"])/30,
                                  "newHiddenFits": sum(f["newHiddenFits"] for f in doc["fitAudits"]),
                                  "ridgeAndHiddenSeconds": math.fsum(f.get("ridgeAndHiddenFitSeconds", 0) for f in doc["fitAudits"]),
                                  "mFitSeconds": math.fsum(f.get("mlpFitSeconds", 0) for f in doc["fitAudits"])},
              "trainingFitDiagnostics": [{"fitId":f["fitId"],"fitOrigin":f["fitOrigin"],"n":f["n"],
                  "L_R":f["L_R"],"L_M":f["L_M"],"L_H":f["L_H"],"outputGap":f["outputGap"],
                  "representationGap":f["representationGap"],"D_M":f["D_M"],"D_R":f["D_R"],
                  "retainedRank":f["svd"]["retainedRank"],"R_H_seconds":f.get("ridgeAndHiddenFitSeconds"),
                  "M_seconds":f.get("mlpFitSeconds",0)} for f in doc["fitAudits"]],
              "conclusion": _conclusion(macro),
              "inferenceLimit": "Nested retrospective fit diagnostics are descriptive; DM/bootstrap assess paired forecast losses only."}
    report["reportSha256"] = digest(report, "reportSha256")
    report_path = OUT / ("validation-report.json" if phase == "validation" else "final-report.json")
    with report_path.open("xb") as stream:
        stream.write(canonical(report))
    if phase == "validation" and not decision["passes"]:
        skipped = {"schema": SCHEMA, "phase": "final", "status": "not_run_validation_failed",
                   "validationReportSha256": report["reportSha256"], "modelFits": 0, "forecasts": []}
        skipped["manifestSha256"] = digest(skipped, "manifestSha256")
        with (OUT / "final-not-run-manifest.json").open("xb") as stream:
            stream.write(canonical(skipped))
        skipped_report = {"schema": SCHEMA, "phase": "final", "status": "not_run_validation_failed",
                          "validationReportSha256": report["reportSha256"]}
        skipped_report["reportSha256"] = digest(skipped_report, "reportSha256")
        with (OUT / "final-not-run-report.json").open("xb") as stream:
            stream.write(canonical(skipped_report))
    return report


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze", help="freeze source, input and runtime pins before fits")
    pre = sub.add_parser("preflight", help="check frozen lineage/support without fitting")
    pre.add_argument("phase", choices=tuple(PHASES))
    build = sub.add_parser("build", help="build one answer-free phase manifest")
    build.add_argument("phase", choices=tuple(PHASES))
    score = sub.add_parser("score", help="attach pinned truths and complete phase inference")
    score.add_argument("phase", choices=tuple(PHASES))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        doc = freeze_protocol()
        print(json.dumps({"status":"frozen","protocolSha256":doc["protocolSha256"],"path":str(_protocol_path())}))
    elif args.command == "preflight":
        print(json.dumps(preflight(args.phase), ensure_ascii=False))
    elif args.command == "build":
        doc = build_phase(args.phase)
        print(json.dumps({"status":"manifest_saved","phase":args.phase,"path":str(_manifest_path(args.phase)),
                          "manifestSha256":doc["manifestSha256"],"fitCounts":doc["fitCounts"]}, ensure_ascii=False))
    elif args.command == "score":
        doc = score_phase(args.phase)
        print(json.dumps({"status":"scored","phase":args.phase,"path":str(OUT / ("validation-report.json" if args.phase=="validation" else "final-report.json")),
                          "reportSha256":doc["reportSha256"],"primaryDecision":doc["primaryDecision"],
                          "primaryGarlic10h":doc["primaryGarlic10h"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
