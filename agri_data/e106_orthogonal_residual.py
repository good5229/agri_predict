"""Frozen orthogonal residual correction to the R106 direct forecast.

The numerical core is deliberately independent of project inputs: it consumes
already aligned, finite float64 arrays.  Forecast orchestration is kept behind
explicit CLI phases and imports; importing this module never reads data or fits.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import e106_mlp16 as mlp
from . import e106_rmh as rmh

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/e106-orthogonal-residual-20261011"
SCHEMA = "e106-orthogonal-residual-v1"
PHASES = {"validation": (dt.date(2026, 4, 1), dt.date(2026, 6, 20)),
          "final": (dt.date(2026, 7, 11), dt.date(2026, 9, 27))}
ORIGIN_START = {"validation": dt.date(2026, 4, 1), "final": dt.date(2026, 7, 1)}
TAU_A_RULE = "eps*max(n,107)*smax"
TAU_RESIDUAL_RULE = "max(eps*max(n,16)*tmax,64*eps*max(n,107,16)*max(1,||Z||F,||A||F*||B||F))"
ORTHOGONALITY_TOL = 1e-10
RECONSTRUCTION_TOL = 1e-9
LOSS_ABS_TOL = 1e-12
LOSS_REL_TOL = 1e-10


class ValidationGateFailed(ValueError):
    """A validly recomputed V result that does not authorize conditional T."""


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                        allow_nan=False) + "\n").encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(doc: dict[str, Any], field: str = "sha256") -> str:
    body = dict(doc)
    body.pop(field, None)
    return sha_bytes(canonical(body).rstrip(b"\n"))


def _finite(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains nonfinite values")
    return out


def fit_residual(x_scaled: Any, z: Any, r: Any, r_r: Any) -> dict[str, Any]:
    """Fit B and gamma on one fold; R itself is never modified.

    A includes the intercept and all standardized input columns.  A second
    truncated SVD fits the correction without an intercept.  The SVD factors
    are stored so later verification can replay the solve without decomposing
    again or fitting a model.
    """
    x = _finite(x_scaled, "scaled training matrix", 2)
    hidden = _finite(z, "hidden design", 2)
    target = _finite(r, "training log ratio", 1)
    ridge = _finite(r_r, "R training prediction", 1)
    n = len(target)
    if x.shape != (n, 106) or hidden.shape != (n, 16) or ridge.shape != target.shape:
        raise ValueError("training arrays must align as n×106, n×16 and n-vectors")
    a = np.column_stack((np.ones(n, dtype=np.float64), x))
    eps = np.finfo(np.float64).eps
    u_a, s_a, vt_a = np.linalg.svd(a, full_matrices=False)
    tau_a = float(eps * max(a.shape) * s_a[0])
    keep_a = s_a > tau_a
    b = vt_a[keep_a].T @ ((u_a[:, keep_a].T @ hidden) / s_a[keep_a, None])
    a_svd = {"u": u_a.tolist(), "singularValues": s_a.tolist(), "vt": vt_a.tolist(),
             "retainedMask": keep_a.tolist(), "retainedRank": int(keep_a.sum()),
             "threshold": tau_a, "shape": list(a.shape)}
    z_perp = hidden - a @ b
    z_norm = float(np.linalg.norm(hidden, ord="fro"))
    ab_norm = float(np.linalg.norm(a, ord="fro") * np.linalg.norm(b, ord="fro"))
    u_z, s_z, vt_z = np.linalg.svd(z_perp, full_matrices=False)
    tmax = float(s_z[0]) if len(s_z) else 0.0
    q_round = max(1.0, z_norm, ab_norm)
    tau_round = float(64 * eps * max(n, a.shape[1], hidden.shape[1]) * q_round)
    tau_relative = float(eps * max(n, hidden.shape[1]) * tmax)
    tau_zp = max(tau_relative, tau_round)
    keep_z = s_z > tau_zp
    gamma = (vt_z[keep_z].T @ ((u_z[:, keep_z].T @ (target - ridge)) / s_z[keep_z])
             if keep_z.any() else np.zeros(16, dtype=np.float64))
    z_svd = {"u": u_z.tolist(), "singularValues": s_z.tolist(), "vt": vt_z.tolist(),
             "retainedMask": keep_z.tolist(), "retainedRank": int(keep_z.sum()),
             "threshold": float(tau_zp), "shape": list(z_perp.shape)}
    if gamma.ndim != 1:
        raise ValueError("residual solve returned unexpected coefficient shape")
    correction = z_perp @ gamma
    s_pred = ridge + correction
    e_r = target - ridge
    l_r = float(np.mean(e_r * e_r))
    l_s = float(np.mean((target - s_pred) ** 2))
    loss_slack = LOSS_ABS_TOL + LOSS_REL_TOL * max(1.0, l_r, l_s)
    orth_a = float(np.linalg.norm(a.T @ z_perp, ord="fro") /
                    max(1.0, np.linalg.norm(a, ord="fro") * np.linalg.norm(z_perp, ord="fro")))
    u_z = np.asarray(z_svd["u"], dtype=np.float64)
    kept = np.asarray(z_svd["retainedMask"], dtype=bool)
    u_a_keep = u_a[:, keep_a]
    q_u = float(np.linalg.norm(u_a_keep.T @ z_perp, ord="fro") /
                max(1.0, z_norm, float(np.linalg.norm(a @ b, ord="fro"))))
    # qA is diagnostic; this rank-aware residual bound is the gate because
    # discarded singular directions and cancellation contribute roundoff.
    orth_a_limit = float(tau_a * z_norm + np.linalg.norm(a, ord="fro") * tau_round
                         + ORTHOGONALITY_TOL * max(1.0, np.linalg.norm(a, ord="fro") *
                                                   np.linalg.norm(z_perp, ord="fro")))
    if float(np.linalg.norm(a.T @ z_perp, ord="fro")) > orth_a_limit or q_u > ORTHOGONALITY_TOL:
        raise ArithmeticError("orthogonal residual failed its fixed numerical check")
    r_projection = u_a_keep @ (u_a_keep.T @ ridge) if keep_a.any() else np.zeros_like(ridge)
    d_ra = float(np.mean((ridge - r_projection) ** 2))
    r_projection_error = float(np.max(np.abs(ridge - r_projection)))
    r_projection_limit = RECONSTRUCTION_TOL * max(1.0, float(np.max(np.abs(ridge))))
    if r_projection_error > r_projection_limit:
        raise ArithmeticError("R training output is not reproduced by retained A space")
    e_r = target - ridge
    e_projection = u_z[:, kept] @ (u_z[:, kept].T @ e_r) if kept.any() else np.zeros_like(e_r)
    correction_projection_error = float(np.max(np.abs(correction - e_projection)))
    correction_projection_limit = RECONSTRUCTION_TOL * max(1.0, float(np.max(np.abs(e_r))))
    if correction_projection_error > correction_projection_limit:
        raise ArithmeticError("stored correction differs from retained Zperp residual projection")
    if l_s > l_r + loss_slack:
        raise ArithmeticError("least-squares correction increased training MSE beyond tolerance")
    return {"n": n, "B": b.tolist(), "gamma": gamma.tolist(),
            "A_svd": a_svd, "Zperp_svd": z_svd,
            "tauARule": TAU_A_RULE, "tauZperpRule": TAU_RESIDUAL_RULE,
            "tauA": tau_a, "tauZperp": tau_zp, "tauZperpRelative": tau_relative,
            "tauZperpRoundoff": tau_round, "qRoundoff": q_round,
            "rankA": a_svd["retainedRank"], "rankZperp": z_svd["retainedRank"],
            "singularValuesA": a_svd["singularValues"], "singularValuesZperp": z_svd["singularValues"],
            "orthogonalityNormalizedFro": orth_a, "orthogonalityAbsoluteFro": float(np.linalg.norm(a.T @ z_perp, ord="fro")),
            "orthogonalityLimit": orth_a_limit, "qU": q_u,
            "zeroResidualRepresentation": not bool(z_svd["retainedRank"]),
            "L_R": l_r, "L_S": l_s, "trainingDeltaMse": l_s - l_r,
            "D_RA": d_ra, "rRetainedProjectionMaxAbsError": r_projection_error,
            "rRetainedProjectionLimit": r_projection_limit,
            "correctionProjectionMaxAbsError": correction_projection_error,
            "correctionProjectionLimit": correction_projection_limit,
            "trainingLossSlack": loss_slack, "trainingPrediction": s_pred.tolist(),
            "rTrainingPrediction": ridge.tolist(), "trainingCorrection": correction.tolist()}


def predict_residual(x_scaled: Any, z: Any, r_r: Any, fit: dict[str, Any]) -> np.ndarray:
    x = _finite(x_scaled, "scaled prediction matrix", 2)
    hidden = _finite(z, "prediction hidden features", 2)
    ridge = _finite(r_r, "R prediction", 1)
    if x.shape != (len(ridge), 106) or hidden.shape != (len(ridge), 16):
        raise ValueError("prediction arrays have incompatible shapes")
    b = _finite(fit["B"], "saved B", 2)
    gamma = _finite(fit["gamma"], "saved gamma", 1)
    if b.shape != (107, 16) or gamma.shape != (16,):
        raise ValueError("saved residual state has invalid shapes")
    a = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    z_perp = hidden - a @ b
    out = ridge + z_perp @ gamma
    if not np.isfinite(out).all():
        raise ValueError("residual prediction is nonfinite")
    return out


def residual_identity(actual_log: Any, r_log: Any, s_log: Any) -> dict[str, float]:
    """Exact same-row MSE difference decomposition, for diagnostics only."""
    y = _finite(actual_log, "actual log ratio", 1)
    r = _finite(r_log, "R log prediction", 1)
    s = _finite(s_log, "S log prediction", 1)
    if y.shape != r.shape or y.shape != s.shape or not len(y):
        raise ValueError("identity inputs must be nonempty aligned vectors")
    e_r, d = y - r, s - r
    mse_delta = float(np.mean((y - s) ** 2) - np.mean((y - r) ** 2))
    rhs = float(np.mean(d * d) - 2.0 * np.mean(e_r * d))
    return {"mseSMinusR": mse_delta, "correctionSquared": float(np.mean(d * d)),
            "residualAlignmentOffset": float(-2.0 * np.mean(e_r * d)),
            "identityRhs": rhs, "identityError": abs(mse_delta - rhs)}


def direct_price(p0: float, log_ratio: float) -> float:
    """Restore the direct forecast without clipping or implicit fallback."""
    return rmh.price_prediction(float(p0), float(log_ratio))


def _rmh_coefs_in_s_shape(source_fit: dict[str, Any]) -> list[Any] | None:
    values, shapes = source_fit.get("mlpCoefs", []), source_fit.get("mlpCoefShapes", [])
    if not isinstance(values, list) or not isinstance(shapes, list) or len(values) != len(shapes):
        return None
    try:
        return [np.asarray(value, dtype=np.float64).reshape(tuple(shape)).tolist()
                for value, shape in zip(values, shapes)]
    except (TypeError, ValueError):
        return None


def _ridge_training_output(x_scaled: Any, beta_value: Any, saved_value: Any) -> np.ndarray:
    x = _finite(x_scaled, "R scaled training matrix", 2)
    beta = _finite(beta_value, "R training coefficients", 1)
    saved = _finite(saved_value, "R training prediction", 1)
    if x.shape[1] != 106 or beta.shape != (107,) or saved.shape != (len(x),):
        raise ValueError("R training output state has inconsistent shapes")
    replayed = beta[0] + x @ beta[1:]
    if not np.allclose(saved, replayed, rtol=1e-10, atol=1e-10):
        raise ValueError("S R training prediction does not replay from stored coefficients")
    return replayed


def _s_forecast_candidate(x_scaled: np.ndarray, r_log: float, r_price: float,
                         p0: float, sarima_source: dict[str, Any],
                         fit: dict[str, Any]) -> dict[str, Any]:
    """Compute S only; numerical/component failures become exact-E106 fallback evidence."""
    sarima_price = None
    s_log = s_price = r_blend = s_blend = None
    try:
        coefs = [np.asarray(v, dtype=np.float64) for v in fit["mlpCoefs"]]
        intercepts = [np.asarray(v, dtype=np.float64) for v in fit["mlpIntercepts"]]
        hidden = rmh.hidden_design(x_scaled, coefs, intercepts)[:, 1:]
        s_log = float(predict_residual(x_scaled, hidden, np.asarray([r_log]), fit)[0])
        s_price = direct_price(p0, s_log)
        if not math.isfinite(s_log) or not math.isfinite(s_price) or s_price <= 0:
            raise FloatingPointError("S direct forecast is nonfinite or nonpositive")
    except (ArithmeticError, FloatingPointError, OverflowError, ValueError) as exc:
        s_failure: str | None = type(exc).__name__
        if s_log is not None and not math.isfinite(float(s_log)):
            s_log = None
        if s_price is not None and (not math.isfinite(float(s_price)) or float(s_price) <= 0):
            s_price = None
    else:
        s_failure = None
    try:
        if sarima_source.get("status") != "issued" or sarima_source.get("prediction") is None:
            raise ValueError("saved SARIMA component is unavailable")
        sarima_price = float(sarima_source["prediction"])
        if not math.isfinite(sarima_price) or sarima_price <= 0:
            sarima_price = None
            raise FloatingPointError("saved SARIMA component is nonfinite or nonpositive")
    except (ArithmeticError, FloatingPointError, OverflowError, ValueError) as exc:
        sarima_failure = type(exc).__name__
    else:
        sarima_failure = None
    if sarima_price is not None:
        r_blend = .5 * (r_price + sarima_price)
        if not math.isfinite(r_blend) or r_blend <= 0:
            r_blend = None
            sarima_failure = "FloatingPointError"
    if sarima_price is not None and s_price is not None:
        s_blend = .5 * (s_price + sarima_price)
        if not math.isfinite(s_blend) or s_blend <= 0:
            s_blend = None
            s_failure = "FloatingPointError"
    failure_type = s_failure or sarima_failure
    failure = f"S_invalid_or_component_unavailable:{failure_type}" if failure_type else None
    return {"SLogPrediction": s_log, "SDirectPrediction": s_price,
            "RBlendPrediction": r_blend, "SBlendPrediction": s_blend,
            "sarimaPrediction": sarima_price, "failureReason": failure}


def _file_sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def _protocol_path() -> Path:
    return OUT / "protocol.json"


def _manifest_path(phase: str) -> Path:
    return OUT / f"{phase}-manifest.json"


def _report_path(phase: str) -> Path:
    return OUT / f"{phase}-report.json"


def freeze_protocol() -> dict[str, Any]:
    """Freeze executable, reviewed source dependencies, runtime, and inputs."""
    from . import e106_orthogonal_residual_score as score
    runtime = rmh._runtime()
    mlp._check_pinned_inputs()
    rmh_manifest = rmh._manifest_path("validation")
    rmh_report = rmh.OUT / "validation-report.json"
    if not rmh_manifest.is_file() or not rmh_report.is_file():
        raise FileNotFoundError("completed, cleared RMH validation artifacts are required before S freeze")
    deps = {"module": _file_sha(Path(__file__)), "rmhModule": _file_sha(Path(rmh.__file__)),
            "mlpModule": _file_sha(Path(mlp.__file__)), "prices": _file_sha(mlp.PRICES),
            "r106": _file_sha(mlp.R106), "preservedFeatureHelper": _file_sha(Path(mlp.pf.__file__)),
            "e106Confirmation": _file_sha(mlp.CONF_BASE), "e106Final": _file_sha(mlp.FINAL_BASE),
            "savedMlpManifest": _file_sha(mlp.OUT / "confirmation-manifest.json"),
            "savedMlpReport": _file_sha(mlp.OUT / "confirmation-report.json"),
            "trainingDiagnostic": _file_sha(mlp.OUT / "training-diagnostic.json"),
            "scorerModule": _file_sha(Path(score.__file__)),
            "statisticsModule": _file_sha(Path(score.stats.__file__)),
            "bootstrapModule": _file_sha(Path(score.__file__).with_name("research_progress_diagnostic.py")),
            "rmhValidationManifest": _file_sha(rmh_manifest), "rmhValidationReport": _file_sha(rmh_report)}
    doc = {"schema": SCHEMA, "frozenBeforeFit": True, "inputs": deps, "runtime": runtime,
           "protocol": {"phases": {k: [v[0].isoformat(), v[1].isoformat()] for k, v in PHASES.items()},
                        "originStart": {k: v.isoformat() for k, v in ORIGIN_START.items()},
                        "definition": "A=[1,X]; B=argmin||Z-A B||²; Zperp=Z-A B; gamma=argmin||r-rR-Zperp gamma||²; rS=rR+Zperp gamma",
                        "scalerAndHidden": "reuse saved M scaler and 16-unit ReLU hidden layer in V; fixed M config in T",
                        "tauARule": TAU_A_RULE, "tauResidualRule": TAU_RESIDUAL_RULE,
                        "orthogonality": {"qAResidualLimit": ORTHOGONALITY_TOL,
                                          "qUResidualLimit": ORTHOGONALITY_TOL,
                                          "rankAwareGate": "||A.T@Zperp||F <= tauA||Z||F + ||A||F*tauRound + 1e-10*max(1,||A||F||Zperp||F)"},
                        "rRetainedProjectionMaxAbsScale": RECONSTRUCTION_TOL,
                        "correctionProjectionMaxAbsScale": RECONSTRUCTION_TOL,
                        "trainingMseSlack": {"absolute": LOSS_ABS_TOL, "relative": LOSS_REL_TOL},
                        "r106ReplayTolerance": {"absolute": rmh.R_REPRO_ABS, "relative": rmh.R_REPRO_REL},
                        "VFitBudget": {"M": 0, "R": 0, "B": 30, "gamma": 30},
                        "conditionalTFitBudget": {"M": 30, "R": 30, "B": 30, "gamma": 30, "H": 0},
                        "TRequiresVerifiedVPass": True, "VAllZeroCorrectionStops": True,
                        "priceBlend": "0.5*S direct price + 0.5*same saved causal SARIMA price",
                        "otherCrops": "exact saved E106 policy",
                        "statistics": {"fixedTests": 20, "HACCalendarDays": [9, 28], "holmAlpha": 0.05,
                                       "calendarBlocksDays": [14, 28], "replicates": 2000, "seed": 20261011},
                        "identity": "mean(d^2)-2*mean(eR*d), raw direct log forecasts only; same pairs/aggregation"}}
    doc["protocolSha256"] = digest(doc, "protocolSha256")
    path = _protocol_path()
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("S output directory already contains files; refusing to freeze over prior state")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical(doc))
    return doc


def verify_protocol() -> dict[str, Any]:
    from . import e106_orthogonal_residual_score as score
    path = _protocol_path()
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("protocolSha256") != digest(doc, "protocolSha256"):
        raise ValueError("S protocol digest mismatch")
    if doc.get("inputs", {}).get("module") != _file_sha(Path(__file__)):
        raise ValueError("S source changed after protocol freeze")
    if doc.get("runtime") != rmh._runtime():
        raise RuntimeError("S runtime changed after protocol freeze")
    mlp._check_pinned_inputs()
    expected = {"module": _file_sha(Path(__file__)), "rmhModule": _file_sha(Path(rmh.__file__)),
                "mlpModule": _file_sha(Path(mlp.__file__)), "prices": _file_sha(mlp.PRICES),
                "r106": _file_sha(mlp.R106), "preservedFeatureHelper": _file_sha(Path(mlp.pf.__file__)),
                "e106Confirmation": _file_sha(mlp.CONF_BASE), "e106Final": _file_sha(mlp.FINAL_BASE),
                "savedMlpManifest": _file_sha(mlp.OUT / "confirmation-manifest.json"),
                "savedMlpReport": _file_sha(mlp.OUT / "confirmation-report.json"),
                "trainingDiagnostic": _file_sha(mlp.OUT / "training-diagnostic.json"),
                "scorerModule": _file_sha(Path(score.__file__)),
                "statisticsModule": _file_sha(Path(score.stats.__file__)),
                "bootstrapModule": _file_sha(Path(score.__file__).with_name("research_progress_diagnostic.py")),
                "rmhValidationManifest": _file_sha(rmh._manifest_path("validation")),
                "rmhValidationReport": _file_sha(rmh.OUT / "validation-report.json")}
    if doc.get("inputs") != expected:
        raise ValueError("frozen S source or input hashes changed")
    return doc


def _fit_claim(phase: str, fit_id: str) -> Path:
    protocol = verify_protocol()
    path = OUT / ".fit-claims" / phase / f"{fit_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    claim = {"schema": SCHEMA, "phase": phase, "fitId": fit_id,
             "protocolSha256": protocol["protocolSha256"], "inputs": protocol["inputs"],
             "moduleSha256": protocol["inputs"]["module"], "runtime": protocol["runtime"],
             "reservedAtUtc": dt.datetime.now(dt.timezone.utc).isoformat()}
    with path.open("xb") as stream:
        stream.write(canonical(claim))
    return path


def _verify_claim(path: Path, phase: str, fit_id: str, protocol: dict[str, Any]) -> None:
    claim = json.loads(path.read_text(encoding="utf-8"))
    expected = {"schema": SCHEMA, "phase": phase, "fitId": fit_id,
                "protocolSha256": protocol["protocolSha256"], "inputs": protocol["inputs"],
                "moduleSha256": protocol["inputs"]["module"], "runtime": protocol["runtime"]}
    if any(claim.get(k) != v for k, v in expected.items()) or not claim.get("reservedAtUtc"):
        raise ValueError(f"S fit claim does not bind frozen attempt {fit_id}")


def _state_path(phase: str, fit_id: str) -> Path:
    return OUT / ".fit-states" / phase / f"{fit_id}.json"


def _save_fit_state(phase: str, fit: dict[str, Any], protocol: dict[str, Any]) -> None:
    path = _state_path(phase, fit["fitId"])
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"schema": SCHEMA, "phase": phase, "fitId": fit["fitId"],
             "protocolSha256": protocol["protocolSha256"], "inputs": protocol["inputs"],
             "moduleSha256": protocol["inputs"]["module"], "runtime": protocol["runtime"],
             "trainingKeysSha256": fit["trainingKeysSha256"], "fitAudit": fit}
    state["stateSha256"] = digest(state, "stateSha256")
    with path.open("xb") as stream:
        stream.write(canonical(state))


def _load_fit_state(phase: str, fit_id: str, expected: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any] | None:
    path = _state_path(phase, fit_id)
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    if (state.get("stateSha256") != digest(state, "stateSha256")
            or state.get("schema") != SCHEMA or state.get("phase") != phase
            or state.get("fitId") != fit_id or state.get("protocolSha256") != protocol["protocolSha256"]
            or state.get("inputs") != protocol["inputs"] or state.get("moduleSha256") != protocol["inputs"]["module"]
            or state.get("runtime") != protocol["runtime"]):
        raise ValueError(f"S fit state source binding changed for {fit_id}")
    fit = state.get("fitAudit", {})
    if any(fit.get(k) != v for k, v in expected.items()):
        raise ValueError(f"S fit state training lineage changed for {fit_id}")
    _verify_claim(OUT / ".fit-claims" / phase / f"{fit_id}-B-gamma.json", phase,
                  f"{fit_id}-B-gamma", protocol)
    return fit


def _component_path(phase: str, fit_id: str, component: str) -> Path:
    return OUT / ".fit-components" / phase / f"{fit_id}-{component}.json"


def _save_component(phase: str, fit_id: str, component: str, value: dict[str, Any],
                    expected: dict[str, Any], protocol: dict[str, Any]) -> None:
    path = _component_path(phase, fit_id, component)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc = {"schema": SCHEMA, "phase": phase, "fitId": fit_id, "component": component,
           "protocolSha256": protocol["protocolSha256"], "inputs": protocol["inputs"],
           "moduleSha256": protocol["inputs"]["module"], "runtime": protocol["runtime"],
           "expected": expected, "value": value}
    doc["componentSha256"] = digest(doc, "componentSha256")
    with path.open("xb") as stream:
        stream.write(canonical(doc))


def _load_component(phase: str, fit_id: str, component: str, expected: dict[str, Any],
                    protocol: dict[str, Any]) -> dict[str, Any] | None:
    path = _component_path(phase, fit_id, component)
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if (doc.get("componentSha256") != digest(doc, "componentSha256")
            or doc.get("schema") != SCHEMA or doc.get("phase") != phase
            or doc.get("fitId") != fit_id or doc.get("component") != component
            or doc.get("protocolSha256") != protocol["protocolSha256"]
            or doc.get("inputs") != protocol["inputs"]
            or doc.get("moduleSha256") != protocol["inputs"]["module"]
            or doc.get("runtime") != protocol["runtime"] or doc.get("expected") != expected):
        raise ValueError(f"S component checkpoint binding failed for {fit_id}/{component}")
    _verify_claim(OUT / ".fit-claims" / phase / f"{fit_id}-{component}.json", phase,
                  f"{fit_id}-{component}", protocol)
    return doc.get("value")


def _training_support(phase: str, h: int, month: tuple[int, int], series: dict,
                      features: dict, frozen_r106: dict) -> tuple[list[dt.date], dt.date, dict[str, Any], np.ndarray, np.ndarray]:
    from . import preserved_feature_forecasts as pf
    month_keys = [key for key in frozen_r106 if key[0] == "garlic" and key[2] == h
                  and (key[1] - dt.timedelta(days=h)).year == month[0]
                  and (key[1] - dt.timedelta(days=h)).month == month[1]]
    frozen, fit_origin = mlp._frozen_fit_audit(frozen_r106, h, month_keys)
    training = [day for day in series if day >= pf.online.narrow.TRAIN_START and day < fit_origin
                and features[h].get(day) is not None]
    train_audit = pf._fit_audit(training, fit_origin)
    pf._assert_training_support(train_audit, frozen, "garlic", h, month)
    x = np.asarray([features[h][day]["preserved106"] for day in training], dtype=np.float64)
    y = np.asarray([math.log(series[day][0] / features[h][day]["p0"]) for day in training], dtype=np.float64)
    return training, fit_origin, train_audit, x, y


def preflight(phase: str = "validation") -> dict[str, Any]:
    """Check fixed source and training-key coverage; performs no fit."""
    if phase not in PHASES:
        raise ValueError("phase must be validation or final")
    protocol = verify_protocol()
    if phase == "final":
        _verify_validation_gate()
    src = "confirmation" if phase == "validation" else "final"
    base = mlp._base_doc(src)
    base_index = mlp._idx(base["forecasts"])
    r106 = mlp._r106_index()
    series, features, by_h = mlp._make_features(src, base_index, r106)
    support = {}
    for h in mlp.HORIZONS:
        months = sorted({((key[1] - dt.timedelta(days=h)).year,
                          (key[1] - dt.timedelta(days=h)).month) for key in by_h[h]})
        if len(months) != 3:
            raise ValueError(f"expected 3 frozen monthly fit groups for garlic/h{h}")
        support[str(h)] = {"forecastKeys": len(by_h[h]), "featureRows": len(features[h]),
                           "monthlyFitIds": [f"garlic-h{h}-{y}-{m:02d}" for y, m in months]}
    return {"schema": SCHEMA, "phase": phase, "protocolSha256": protocol["protocolSha256"],
            "support": support, "status": "ready_no_fit_performed"}


def _fit_state(phase: str, fit_id: str, h: int, month: tuple[int, int], training: list[dt.date],
               fit_origin: dt.date, train_audit: dict[str, Any], x: np.ndarray, y: np.ndarray,
               saved_rmh_fit: dict[str, Any] | None, protocol: dict[str, Any]) -> dict[str, Any]:
    """Load V R/M, or make the one conditional T R/M fit, then fit S once."""
    expected = {"fitId": fit_id, "fitOrigin": fit_origin.isoformat(),
                "trainingKeysSha256": sha_bytes(canonical([d.isoformat() for d in training])),
                "trainingMatrixSha256": sha_bytes(x.tobytes(order="C")),
                "trainingTargetSha256": sha_bytes(y.tobytes(order="C")), "training": train_audit}
    resumed = _load_fit_state(phase, fit_id, expected, protocol)
    if resumed is not None:
        verify_fit_free((x - np.asarray(resumed["scalerMean"])) / np.asarray(resumed["scalerScale"]),
                        _hidden_training(x, resumed), y,
                        np.asarray(resumed["rTrainingPrediction"], dtype=np.float64), resumed)
        return resumed

    if phase == "validation":
        if saved_rmh_fit is None:
            raise ValueError(f"missing cleared RMH V state for {fit_id}")
        mean = np.asarray(saved_rmh_fit["scalerMean"], dtype=np.float64)
        scale = np.asarray(saved_rmh_fit["scalerScale"], dtype=np.float64)
        coefs = [np.asarray(v, dtype=np.float64).reshape(shape) for v, shape in
                 zip(saved_rmh_fit["mlpCoefs"], saved_rmh_fit["mlpCoefShapes"])]
        intercepts = [np.asarray(v, dtype=np.float64) for v in saved_rmh_fit["mlpIntercepts"]]
        beta_r = np.asarray(saved_rmh_fit["ridgeCoefficientIncludingIntercept"], dtype=np.float64)
        xs = (x - mean) / scale
        r_train = beta_r[0] + xs @ beta_r[1:]
        if not np.allclose(r_train, saved_rmh_fit["rTrainingPrediction"], rtol=1e-10, atol=1e-10):
            raise ValueError(f"pinned RMH R training output does not replay for {fit_id}")
        model_audit = {"MSource": "pinned_RMH_validation_saved_M", "RSource": "pinned_RMH_validation_saved_R",
                       "MFitSeconds": 0.0, "RFitSeconds": 0.0}
    else:
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        import warnings
        component_ids = ("R", "M")
        component_data = {}
        for component in component_ids:
            component_expected = {**expected, "model": component,
                                  "parameters": mlp.MODEL_PARAMS if component == "M" else
                                  {"alpha": 10.0, "fit_intercept": True, "solver": "cholesky"}}
            prior = _load_component(phase, fit_id, component, component_expected, protocol)
            if prior is None:
                claim_path = OUT / ".fit-claims" / phase / f"{fit_id}-{component}.json"
                if claim_path.exists():
                    raise RuntimeError(f"interrupted {component} fit attempt consumed for {fit_id}; refusing refit")
                _fit_claim(phase, f"{fit_id}-{component}")
                start = __import__("time").monotonic()
                if component == "R":
                    scaler = StandardScaler().fit(x)
                    mean, scale = scaler.mean_.astype(float), scaler.scale_.astype(float)
                    model = Ridge(alpha=10.0, fit_intercept=True, solver="cholesky").fit((x - mean) / scale, y)
                    value = {"scalerMean": mean.tolist(), "scalerScale": scale.tolist(),
                             "ridgeCoefficientIncludingIntercept": np.concatenate(([model.intercept_], model.coef_)).astype(float).tolist(),
                             "fitSeconds": __import__("time").monotonic() - start}
                else:
                    if "R" not in component_data:
                        component_data["R"] = _load_component(phase, fit_id, "R",
                            {**expected, "model": "R", "parameters": {"alpha":10.0,"fit_intercept":True,"solver":"cholesky"}}, protocol)
                    r_saved = component_data["R"]
                    if r_saved is None:
                        raise RuntimeError("R component must be checkpointed before M fit")
                    mean, scale = np.asarray(r_saved["scalerMean"]), np.asarray(r_saved["scalerScale"])
                    model = MLPRegressor(**mlp.MODEL_PARAMS)
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        model.fit((x - mean) / scale, y)
                    value = {"scalerMean": mean.tolist(), "scalerScale": scale.tolist(),
                             "mlpCoefs": [c.astype(float).tolist() for c in model.coefs_],
                             "mlpIntercepts": [v.astype(float).tolist() for v in model.intercepts_],
                             "lossCurve": [float(v) for v in model.loss_curve_], "nIter": int(model.n_iter_),
                             "finalLoss": float(model.loss_),
                             "warnings": [f"{type(w.message).__name__}: {w.message}" for w in caught],
                             "fitSeconds": __import__("time").monotonic() - start}
                if not np.isfinite(np.asarray(value["scalerMean"])).all() or not np.isfinite(np.asarray(value["scalerScale"])).all():
                    raise FloatingPointError(f"nonfinite scaler state for {fit_id}/{component}")
                _save_component(phase, fit_id, component, value, component_expected, protocol)
                component_data[component] = value
            else:
                component_data[component] = prior
        r_saved, m_saved = component_data["R"], component_data["M"]
        if r_saved["scalerMean"] != m_saved["scalerMean"] or r_saved["scalerScale"] != m_saved["scalerScale"]:
            raise ValueError(f"T R and M scaler state differs for {fit_id}")
        mean, scale = np.asarray(r_saved["scalerMean"], dtype=np.float64), np.asarray(r_saved["scalerScale"], dtype=np.float64)
        beta_r = np.asarray(r_saved["ridgeCoefficientIncludingIntercept"], dtype=np.float64)
        coefs = [np.asarray(v, dtype=np.float64) for v in m_saved["mlpCoefs"]]
        intercepts = [np.asarray(v, dtype=np.float64) for v in m_saved["mlpIntercepts"]]
        xs = (x - mean) / scale
        r_train = beta_r[0] + xs @ beta_r[1:]
        model_audit = {"MSource": "fixed_authorized_T_MLP16_fit", "RSource": "fixed_authorized_T_Ridge_fit",
                       "MFitSeconds": m_saved["fitSeconds"], "RFitSeconds": r_saved["fitSeconds"],
                       "M": {k: v for k, v in m_saved.items() if k not in ("scalerMean", "scalerScale", "mlpCoefs", "mlpIntercepts")}}
    if mean.shape != (106,) or scale.shape != (106,) or np.any(scale <= 0):
        raise ValueError(f"invalid scaler for {fit_id}")
    xs = (x - mean) / scale
    hidden = rmh.hidden_design(xs, coefs, intercepts)[:, 1:]
    if phase == "validation":
        claim_path = OUT / ".fit-claims" / phase / f"{fit_id}-B-gamma.json"
        if claim_path.exists():
            raise RuntimeError(f"interrupted S coefficient fit consumed for {fit_id}; refusing refit")
        _fit_claim(phase, f"{fit_id}-B-gamma")
    else:
        claim_path = OUT / ".fit-claims" / phase / f"{fit_id}-B-gamma.json"
        if claim_path.exists():
            raise RuntimeError(f"interrupted S coefficient fit consumed for {fit_id}; refusing refit")
        _fit_claim(phase, f"{fit_id}-B-gamma")
    start = __import__("time").monotonic()
    fitted = fit_residual(xs, hidden, y, r_train)
    fitted.update(expected)
    fitted.update({"scalerMean": mean.tolist(), "scalerScale": scale.tolist(),
                   "mlpCoefs": [v.tolist() for v in coefs], "mlpIntercepts": [v.tolist() for v in intercepts],
                   "ridgeCoefficientIncludingIntercept": beta_r.tolist(),
                   "rTrainingPrediction": r_train.tolist(), "mlpStateSource": model_audit["MSource"],
                   "modelFitAudit": model_audit, "SFitSeconds": __import__("time").monotonic() - start,
                   "trainingRows": int(len(training)), "newHiddenFits": 0 if phase == "validation" else 1})
    _save_fit_state(phase, fitted, protocol)
    return fitted


def _hidden_training(x: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    xs = (x - np.asarray(fit["scalerMean"], dtype=np.float64)) / np.asarray(fit["scalerScale"], dtype=np.float64)
    coefs = [np.asarray(v, dtype=np.float64) for v in fit["mlpCoefs"]]
    intercepts = [np.asarray(v, dtype=np.float64) for v in fit["mlpIntercepts"]]
    return rmh.hidden_design(xs, coefs, intercepts)[:, 1:]


def build_phase(phase: str = "validation", output: Path | None = None) -> dict[str, Any]:
    """Build the immutable answer-free S manifest for V or an authorized T."""
    import time
    if phase not in PHASES:
        raise ValueError("phase must be validation or final")
    protocol = verify_protocol()
    preflight(phase)
    if phase == "final":
        _verify_validation_gate()
    started = time.monotonic()
    src_phase = "confirmation" if phase == "validation" else "final"
    base_doc = mlp._base_doc(src_phase)
    base_index = mlp._idx(base_doc["forecasts"])
    r106 = mlp._r106_index()
    rows = mlp._identity_rows(src_phase, base_index, r106)
    for row in rows:
        row.pop("usedMLP", None)
        row["usedS"] = False
        row["actualLogRatio"] = None
        row.update({"RLogPrediction": None, "SLogPrediction": None,
                    "RDirectPrediction": None, "SDirectPrediction": None,
                    "RBlendPrediction": None, "SBlendPrediction": None,
                    "sarimaPrediction": None, "fitId": None,
                    "featureVector": None, "p0": None, "p0Date": None,
                    "r106ReplayAbsError": None})
    series, features, by_h = mlp._make_features(src_phase, base_index, r106)

    saved_rmh: dict[str, dict[str, Any]] = {}
    if phase == "validation":
        rmh_path = rmh._manifest_path("validation")
        rmh_doc = json.loads(rmh_path.read_text(encoding="utf-8"))
        rmh.verify_manifest(rmh_doc, "validation", rmh_path)
        saved_rmh = {fit["fitId"]: fit for fit in rmh_doc["fitAudits"]}

    fit_audits = []
    by_fit = {}
    for h in mlp.HORIZONS:
        months = sorted({((key[1] - dt.timedelta(days=h)).year,
                          (key[1] - dt.timedelta(days=h)).month) for key in by_h[h]})
        if len(months) != 3:
            raise ValueError(f"expected exactly 3 fit months for garlic/h{h}")
        for month in months:
            fit_id = f"garlic-h{h}-{month[0]}-{month[1]:02d}"
            training, origin, train_audit, x, y = _training_support(phase, h, month, series, features, r106)
            fit = _fit_state(phase, fit_id, h, month, training, origin, train_audit, x, y,
                             saved_rmh.get(fit_id), protocol)
            fit_audits.append(fit)
            by_fit[fit_id] = fit

    fallback_rows = []
    replay_gaps = []
    all_forecasts = []
    for row in rows:
        key = (row["product"], dt.date.fromisoformat(row["target"]), int(row["horizon"]))
        product, target, h = key
        base = base_index[key]["candidate"]
        if product != "garlic":
            # Other three crops retain the exact saved E106 policy record.
            row.update({"usedS": False, "fallbackReason": "non_garlic_exact_E106",
                        "actualLogRatio": None, "RLogPrediction": None, "SLogPrediction": None,
                        "RDirectPrediction": None, "SDirectPrediction": None,
                        "RBlendPrediction": None, "SBlendPrediction": None,
                        "sarimaPrediction": None, "fitId": None})
            all_forecasts.append(row)
            continue
        origin = target - dt.timedelta(days=h)
        fit_id = f"garlic-h{h}-{origin.year}-{origin.month:02d}"
        fit = by_fit.get(fit_id)
        fr = features[h].get(target)
        r_source = r106[key]["models"]["preserved106"]
        sar_source = r106[key]["models"]["causalSarima"]
        if fit is None or fr is None:
            reason = "feature_unavailable_or_source_too_old"
            row.update({"usedS": False, "fallbackReason": reason, "fitId": None,
                        "prediction": base.get("prediction"), "status": base.get("status"), "reason": base.get("reason"),
                        "actualLogRatio": None, "RLogPrediction": None, "SLogPrediction": None,
                        "RDirectPrediction": None, "SDirectPrediction": None,
                        "RBlendPrediction": None, "SBlendPrediction": None, "sarimaPrediction": None})
            fallback_rows.append({"key": [product, target.isoformat(), h], "reason": reason})
            all_forecasts.append(row)
            continue
        vector = np.asarray(fr["preserved106"], dtype=np.float64)
        mean = np.asarray(fit["scalerMean"], dtype=np.float64)
        scale = np.asarray(fit["scalerScale"], dtype=np.float64)
        xs = (vector - mean) / scale
        beta_r = np.asarray(fit["ridgeCoefficientIncludingIntercept"], dtype=np.float64)
        r_log = float(beta_r[0] + xs @ beta_r[1:])
        p0 = float(fr["p0"])
        r_price = direct_price(p0, r_log)
        old_r = r_source.get("prediction")
        if r_source.get("status") != "issued" or old_r is None:
            raise ValueError(f"original R106 direct output is unavailable at {key}")
        if not rmh.ridge_reproduces(r_price, float(old_r)):
            raise ArithmeticError(f"S source R106 replay failed at {key}")
        replay_gap = abs(r_price - float(old_r))
        replay_gaps.append(replay_gap)
        candidate = _s_forecast_candidate(xs.reshape(1, -1), r_log, r_price,
                                          p0, sar_source, fit)
        s_log, s_price = candidate["SLogPrediction"], candidate["SDirectPrediction"]
        r_blend, s_blend = candidate["RBlendPrediction"], candidate["SBlendPrediction"]
        sarima_price = candidate["sarimaPrediction"]
        if candidate["failureReason"]:
            reason = candidate["failureReason"]
            row.update({"usedS": False, "fallbackReason": reason, "fitId": fit_id,
                        "featureVector": fr["preserved106"], "p0": p0,
                        "p0Date": fr["p0Date"].isoformat(), "prediction": base.get("prediction"),
                        "status": base.get("status"), "reason": base.get("reason"),
                        "actualLogRatio": None, "RLogPrediction": r_log, "SLogPrediction": s_log,
                        "RDirectPrediction": r_price, "SDirectPrediction": s_price,
                        "RBlendPrediction": r_blend, "SBlendPrediction": s_blend,
                        "sarimaPrediction": sarima_price, "r106ReplayAbsError": replay_gap})
            fallback_rows.append({"key": [product, target.isoformat(), h], "reason": reason})
            all_forecasts.append(row)
            continue
        row.update({"status": "issued", "prediction": s_blend, "reason": None,
                    "usedS": True, "fallbackReason": None, "fitId": fit_id,
                    "featureVector": fr["preserved106"], "p0": p0, "p0Date": fr["p0Date"].isoformat(),
                    "RLogPrediction": r_log, "SLogPrediction": s_log,
                    "actualLogRatio": None,
                    "RDirectPrediction": r_price, "SDirectPrediction": s_price,
                    "RBlendPrediction": r_blend, "SBlendPrediction": s_blend,
                    "sarimaPrediction": sarima_price, "r106ReplayAbsError": replay_gap})
        all_forecasts.append(row)

    all_zero = all(fit["rankZperp"] == 0 or not any(abs(float(v)) > 0 for v in fit["gamma"])
                   for fit in fit_audits)
    doc = {"schema": SCHEMA, "phase": phase, "targetValuesAttached": False,
           "protocolSha256": protocol["protocolSha256"],
           "protocol": {"id": SCHEMA, "protocolSha256": protocol["protocolSha256"],
                        "inputHashes": protocol["inputs"], "runtime": protocol["runtime"]},
           "sourcePhase": src_phase, "inputHashes": protocol["inputs"],
           "runtime": protocol["runtime"],
           "fitCounts": {"M": 0 if phase == "validation" else 30,
                         "R": 0 if phase == "validation" else 30,
                         "B": len(fit_audits), "gamma": len(fit_audits), "H": 0},
           "newHiddenFits": 0 if phase == "validation" else 30,
           "plannedCount": len(all_forecasts), "forecasts": all_forecasts,
           "fitAudits": fit_audits, "fallbackRows": fallback_rows,
           "allCorrectionsZero": all_zero,
           "r106Replay": {"n": len(replay_gaps), "maxAbsError": max(replay_gaps) if replay_gaps else None,
                          "passed": len(replay_gaps) > 0},
           "buildElapsedSeconds": time.monotonic() - started,
           "informationLimitation": "Pinned retrospective prices have observation dates but no proven row-level first-publication/availability timestamp; causal as-of-observation-date reconstruction only."}
    doc["manifestSha256"] = digest(doc, "manifestSha256")
    dest = output or _manifest_path(phase)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as stream:
        stream.write(canonical(doc))
    return doc


def _verify_validation_gate() -> dict[str, Any]:
    manifest = _manifest_path("validation")
    report_path = _report_path("validation")
    if not manifest.is_file() or not report_path.is_file():
        raise ValueError("verified S validation manifest/report required before conditional T")
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    verify_manifest(doc, "validation", manifest)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("manifestFileSha256") != _file_sha(manifest)
            or report.get("manifestSha256") != doc.get("manifestSha256")
            or report.get("protocolSha256") != doc.get("protocolSha256")
            or report.get("reportSha256") != digest(report, "reportSha256")):
        raise ValueError("S validation report does not bind the verified answer-free manifest")
    recomputed = _score_object(doc, "validation")
    if report.get("score") != recomputed or report.get("primaryDecision") != recomputed.get("primaryDecision"):
        raise ValueError("S validation score does not recompute from its pinned truth/source rows")
    gate = recomputed.get("primaryDecision", {})
    if gate.get("passes") is not True:
        raise ValidationGateFailed("S validation gate failed; conditional T is not authorized")
    return report


def _score_object(doc: dict[str, Any], phase: str) -> dict[str, Any]:
    """Call the pure, source-free scoring helper after the manifest verifier."""
    from . import e106_orthogonal_residual_score as score
    from . import preserved_feature_forecasts as pf
    src = "confirmation" if phase == "validation" else "final"
    base = mlp._idx(mlp._base_doc(src)["forecasts"])
    prices = json.loads(mlp.PRICES.read_text(encoding="utf-8"))
    actuals = pf.direct_horizon._observations(prices)
    corrections_zero = bool(doc.get("allCorrectionsZero"))
    numeric = (doc.get("r106Replay", {}).get("passed") is True
               and len(doc.get("fitAudits", [])) == 30
               and all(float(f.get("qU", math.inf)) <= ORTHOGONALITY_TOL
                       and float(f.get("orthogonalityAbsoluteFro", math.inf)) <= float(f.get("orthogonalityLimit", -math.inf))
                       and float(f.get("rRetainedProjectionMaxAbsError", math.inf)) <= float(f.get("rRetainedProjectionLimit", -math.inf))
                       and float(f.get("correctionProjectionMaxAbsError", math.inf)) <= float(f.get("correctionProjectionLimit", -math.inf))
                       and float(f.get("L_S", math.inf)) <= float(f.get("L_R", -math.inf)) + float(f.get("trainingLossSlack", 0))
                       for f in doc.get("fitAudits", [])))
    report = score.score_manifest(doc, base_forecasts=base, actuals=actuals, phase=phase,
                                  all_corrections_zero=corrections_zero,
                                  numerical_checks_passed=numeric)
    if not isinstance(report, dict) or not isinstance(report.get("primaryDecision"), dict):
        raise ValueError("S scoring helper returned an invalid report contract")
    return report


def score_phase(phase: str = "validation", manifest_path: Path | None = None) -> dict[str, Any]:
    """Verify a frozen forecast manifest, then score once against pinned truths."""
    if phase not in PHASES:
        raise ValueError("phase must be validation or final")
    if phase == "final":
        _verify_validation_gate()
    path = manifest_path or _manifest_path(phase)
    doc = json.loads(path.read_text(encoding="utf-8"))
    verify_manifest(doc, phase, path)
    report = _score_object(doc, phase)
    full = {"schema": SCHEMA + "-report-v1", "phase": phase,
            "protocolSha256": doc["protocolSha256"], "manifestSha256": doc["manifestSha256"],
            "manifestFileSha256": _file_sha(path), "primaryDecision": report["primaryDecision"],
            "score": report, "scoredManifestFile": str(path.relative_to(ROOT))}
    full["reportSha256"] = digest(full, "reportSha256")
    dest = _report_path(phase)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("xb") as stream:
        stream.write(canonical(full))
    return full


def _write_not_run_final(reason: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Persist immutable T nonexecution artifacts when verified V does not pass."""
    protocol = verify_protocol()
    manifest = {"schema": SCHEMA, "phase": "final", "status": "not_run_validation_failed",
                "targetValuesAttached": False, "protocolSha256": protocol["protocolSha256"],
                "protocol": {"id": SCHEMA, "protocolSha256": protocol["protocolSha256"],
                             "inputHashes": protocol["inputs"], "runtime": protocol["runtime"]},
                "inputHashes": protocol["inputs"], "runtime": protocol["runtime"],
                "fitCounts": {"M": 0, "R": 0, "B": 0, "gamma": 0, "H": 0},
                "newHiddenFits": 0, "plannedCount": 0, "forecasts": [], "fitAudits": [],
                "reason": reason}
    manifest["manifestSha256"] = digest(manifest, "manifestSha256")
    mpath = _manifest_path("final")
    mpath.parent.mkdir(parents=True, exist_ok=True)
    with mpath.open("xb") as stream:
        stream.write(canonical(manifest))
    report = {"schema": SCHEMA + "-report-v1", "phase": "final",
              "status": "not_run_validation_failed", "reason": reason,
              "protocolSha256": protocol["protocolSha256"],
              "manifestSha256": manifest["manifestSha256"],
              "manifestFileSha256": _file_sha(mpath),
              "primaryDecision": {"passes": False, "interpretation": "not_run_validation_failed"}}
    report["reportSha256"] = digest(report, "reportSha256")
    rpath = _report_path("final")
    with rpath.open("xb") as stream:
        stream.write(canonical(report))
    return manifest, report


def run_final_if_eligible() -> dict[str, Any]:
    """Conditionally build/score T; otherwise save explicit not-run artifacts."""
    try:
        _verify_validation_gate()
    except ValidationGateFailed as exc:
        _, report = _write_not_run_final(f"validation did not authorize T: {exc}")
        return report
    manifest = build_phase("final")
    verify_manifest(manifest, "final", _manifest_path("final"))
    return score_phase("final")


def verify_manifest(doc: dict[str, Any], phase: str, path: Path | None = None) -> None:
    """Fit-free verification of state, lineage, predictions, and complete key grid."""
    if phase not in PHASES or doc.get("schema") != SCHEMA or doc.get("phase") != phase:
        raise ValueError("wrong S schema or phase")
    protocol = verify_protocol()
    if doc.get("protocolSha256") != protocol["protocolSha256"] or doc.get("inputHashes") != protocol["inputs"]:
        raise ValueError("S manifest protocol/source binding mismatch")
    if path is not None and doc.get("manifestSha256") != digest(doc, "manifestSha256"):
        raise ValueError("S manifest digest mismatch")
    src_phase = "confirmation" if phase == "validation" else "final"
    base_doc = mlp._base_doc(src_phase)
    base = mlp._idx(base_doc["forecasts"])
    r106 = mlp._r106_index()
    series, features, by_h = mlp._make_features(src_phase, base, r106)
    expected_keys = set(mlp.keys(src_phase))
    forecast = mlp._idx(doc.get("forecasts", []))
    if set(forecast) != expected_keys or any(row.get("eligible") is not True for row in forecast.values()):
        raise ValueError("S manifest does not retain the complete eligible E106 key grid")
    fit_by_id = {f.get("fitId"): f for f in doc.get("fitAudits", [])}
    expected_ids = {f"garlic-h{h}-{y}-{m:02d}" for h in mlp.HORIZONS
                    for y, m in sorted({((key[1] - dt.timedelta(days=h)).year,
                                         (key[1] - dt.timedelta(days=h)).month) for key in by_h[h]})}
    if len(fit_by_id) != 30 or set(fit_by_id) != expected_ids:
        raise ValueError("S manifest must have exactly 30 fixed monthly coefficient states")
    expected_counts = {"M": 0 if phase == "validation" else 30,
                       "R": 0 if phase == "validation" else 30, "B": 30, "gamma": 30, "H": 0}
    if doc.get("fitCounts") != expected_counts or doc.get("newHiddenFits") != (0 if phase == "validation" else 30):
        raise ValueError("S phase fit budget was exceeded or misreported")
    source_rmh_fits = {}
    if phase == "validation":
        rmh_path = rmh._manifest_path("validation")
        rmh_doc = json.loads(rmh_path.read_text(encoding="utf-8"))
        rmh.verify_manifest(rmh_doc, "validation", rmh_path)
        source_rmh_fits = {f["fitId"]: f for f in rmh_doc["fitAudits"]}
    claim_dir = OUT / ".fit-claims" / phase
    expected_claims = {f"{fit_id}-B-gamma.json" for fit_id in expected_ids}
    if phase == "final":
        expected_claims |= {f"{fit_id}-{component}.json" for fit_id in expected_ids for component in ("R", "M")}
    actual_claims = {p.name for p in claim_dir.glob("*.json")} if claim_dir.exists() else set()
    if actual_claims != expected_claims:
        raise ValueError("S fit claims do not match the fixed phase budget")
    zero_corrections = True
    for fit_id, fit in fit_by_id.items():
        h = int(fit_id.split("-")[1][1:])
        year, month = map(int, fit_id.rsplit("-", 2)[-2:])
        keys = [k for k in by_h[h] if (k[1] - dt.timedelta(days=h)).year == year
                and (k[1] - dt.timedelta(days=h)).month == month]
        training, origin, audit, x, y = _training_support(phase, h, (year, month), series, features, r106)
        if (fit.get("fitOrigin") != origin.isoformat() or fit.get("training") != audit
                or fit.get("trainingRows") != len(training)
                or fit.get("trainingKeysSha256") != sha_bytes(canonical([d.isoformat() for d in training]))
                or fit.get("trainingMatrixSha256") != sha_bytes(x.tobytes(order="C"))
                or fit.get("trainingTargetSha256") != sha_bytes(y.tobytes(order="C"))):
            raise ValueError(f"S training lineage mismatch for {fit_id}")
        xs = (x - np.asarray(fit["scalerMean"], dtype=np.float64)) / np.asarray(fit["scalerScale"], dtype=np.float64)
        z = _hidden_training(x, fit)
        r_train = np.asarray(fit["rTrainingPrediction"], dtype=np.float64)
        beta_r = _finite(fit.get("ridgeCoefficientIncludingIntercept"),
                         f"{fit_id} R coefficients", 1)
        _ridge_training_output(xs, beta_r, r_train)
        verify_fit_free(xs, z, y, r_train, fit)
        if fit.get("newHiddenFits") != (0 if phase == "validation" else 1):
            raise ValueError(f"S hidden-fit count changed for {fit_id}")
        zero_corrections &= bool(np.all(np.asarray(fit.get("trainingCorrection", []), dtype=np.float64) == 0.0))
        if phase == "validation":
            source_fit = source_rmh_fits.get(fit_id)
            source_coefs = _rmh_coefs_in_s_shape(source_fit) if source_fit is not None else None
            if (source_fit is None or fit.get("scalerMean") != source_fit.get("scalerMean")
                    or fit.get("scalerScale") != source_fit.get("scalerScale")
                    or fit.get("mlpCoefs") != source_coefs
                    or fit.get("mlpIntercepts") != source_fit.get("mlpIntercepts")
                    or fit.get("ridgeCoefficientIncludingIntercept") != source_fit.get("ridgeCoefficientIncludingIntercept")):
                raise ValueError(f"S V did not reuse exact saved R/M state for {fit_id}")
        else:
            common = {"fitId": fit_id, "fitOrigin": origin.isoformat(),
                      "trainingKeysSha256": fit["trainingKeysSha256"],
                      "trainingMatrixSha256": fit["trainingMatrixSha256"],
                      "trainingTargetSha256": fit["trainingTargetSha256"], "training": audit}
            r_expected = {**common, "model": "R", "parameters": {"alpha": 10.0, "fit_intercept": True, "solver": "cholesky"}}
            m_expected = {**common, "model": "M", "parameters": mlp.MODEL_PARAMS}
            r_component = _load_component(phase, fit_id, "R", r_expected, protocol)
            m_component = _load_component(phase, fit_id, "M", m_expected, protocol)
            if (r_component is None or m_component is None
                    or r_component.get("scalerMean") != fit.get("scalerMean")
                    or r_component.get("scalerScale") != fit.get("scalerScale")
                    or r_component.get("ridgeCoefficientIncludingIntercept") != fit.get("ridgeCoefficientIncludingIntercept")
                    or m_component.get("mlpCoefs") != fit.get("mlpCoefs")
                    or m_component.get("mlpIntercepts") != fit.get("mlpIntercepts")):
                raise ValueError(f"S T R/M component checkpoints do not bind {fit_id}")
        cp = json.loads(_state_path(phase, fit_id).read_text(encoding="utf-8"))
        if cp.get("stateSha256") != digest(cp, "stateSha256") or cp.get("fitAudit") != fit:
            raise ValueError(f"S fit state checkpoint does not match manifest for {fit_id}")
        _verify_claim(OUT / ".fit-claims" / phase / f"{fit_id}-B-gamma.json", phase,
                      f"{fit_id}-B-gamma", protocol)
    if bool(doc.get("allCorrectionsZero")) != zero_corrections:
        raise ValueError("S all-zero correction diagnostic does not replay")
    replay_gaps = []

    def check_num(row: dict[str, Any], field: str, expected_value: float | None, key: Any) -> None:
        if field not in row:
            raise ValueError(f"S serialized forecast omits {field} at {key}")
        observed = row[field]
        if expected_value is None:
            if observed is not None:
                raise ValueError(f"S serialized forecast must leave {field} unavailable at {key}")
        elif (observed is None or not math.isfinite(float(observed))
              or not math.isclose(float(observed), float(expected_value), rel_tol=1e-10, abs_tol=1e-8)):
            raise ValueError(f"S serialized forecast does not replay at {key}/{field}")

    for key, row in forecast.items():
        product, target, h = key
        expected = base[key]["candidate"]
        if product != "garlic":
            for field in ("status", "prediction", "reason"):
                if row.get(field) != expected.get(field):
                    raise ValueError(f"non-garlic E106 output changed at {key}")
            if row.get("usedS") or row.get("fallbackReason") != "non_garlic_exact_E106":
                raise ValueError(f"non-garlic provenance changed at {key}")
            continue

        fr = features[h].get(target)
        if fr is None:
            if (row.get("usedS") is not False or row.get("fitId") is not None
                    or row.get("fallbackReason") != "feature_unavailable_or_source_too_old"):
                raise ValueError(f"S unavailable-feature fallback changed at {key}")
            for field in ("status", "prediction", "reason"):
                if row.get(field) != expected.get(field):
                    raise ValueError(f"S unavailable-feature fallback differs from E106 at {key}")
            for field in ("RLogPrediction", "SLogPrediction", "RDirectPrediction", "SDirectPrediction",
                          "RBlendPrediction", "SBlendPrediction", "sarimaPrediction", "p0", "p0Date",
                          "featureVector", "r106ReplayAbsError"):
                if field not in row or row[field] is not None:
                    raise ValueError(f"S unavailable-feature diagnostic {field} changed at {key}")
            if row.get("actualLogRatio") is not None:
                raise ValueError(f"answer was attached to answer-free S manifest at {key}")
            continue

        origin = target - dt.timedelta(days=h)
        fit_id = f"garlic-h{h}-{origin.year}-{origin.month:02d}"
        fit = fit_by_id[fit_id]
        if row.get("fitId") != fit_id or row.get("featureVector") != fr["preserved106"]:
            raise ValueError(f"S feature/fit assignment mismatch at {key}")
        if row.get("p0Date") != fr["p0Date"].isoformat():
            raise ValueError(f"S p0 date does not replay at {key}")
        check_num(row, "p0", float(fr["p0"]), key)
        xs = ((np.asarray(fr["preserved106"], dtype=np.float64) - fit["scalerMean"])
              / fit["scalerScale"]).reshape(1, -1)
        beta_r = np.asarray(fit["ridgeCoefficientIncludingIntercept"], dtype=np.float64)
        rlog = float(beta_r[0] + xs[0] @ beta_r[1:])
        rprice = direct_price(float(fr["p0"]), rlog)
        rsource = r106[key]["models"]["preserved106"]
        old_r = rsource.get("prediction")
        if rsource.get("status") != "issued" or old_r is None or not rmh.ridge_reproduces(rprice, float(old_r)):
            raise ValueError(f"S R106 replay failed for {key}")
        replay_gap = abs(rprice - float(old_r))
        replay_gaps.append(replay_gap)
        sarima_source = r106[key]["models"]["causalSarima"]
        candidate = _s_forecast_candidate(xs, rlog, rprice, float(fr["p0"]), sarima_source, fit)
        failure = candidate["failureReason"]
        should_use_s = failure is None
        if row.get("usedS") is not should_use_s:
            raise ValueError(f"S fallback decision does not replay at {key}")
        expected_fallback = failure
        if should_use_s:
            if row.get("fallbackReason") is not None:
                raise ValueError(f"issued S forecast has a fallback reason at {key}")
            expected_prediction, expected_status, expected_reason = candidate["SBlendPrediction"], "issued", None
        else:
            if row.get("fallbackReason") != expected_fallback:
                raise ValueError(f"S fallback cause does not replay at {key}")
            expected_prediction, expected_status, expected_reason = expected.get("prediction"), expected.get("status"), expected.get("reason")
        for field, value in (("status", expected_status), ("prediction", expected_prediction),
                             ("reason", expected_reason)):
            observed = row.get(field)
            if value is None:
                if observed is not None:
                    raise ValueError(f"S serialized {field} changed at {key}")
            elif field == "prediction":
                check_num(row, field, float(value), key)
            elif observed != value:
                raise ValueError(f"S serialized {field} changed at {key}")
        for field, value in (("RLogPrediction", rlog), ("SLogPrediction", candidate["SLogPrediction"]),
                             ("RDirectPrediction", rprice), ("SDirectPrediction", candidate["SDirectPrediction"]),
                             ("RBlendPrediction", candidate["RBlendPrediction"]),
                             ("SBlendPrediction", candidate["SBlendPrediction"]),
                             ("sarimaPrediction", candidate["sarimaPrediction"]),
                             ("r106ReplayAbsError", replay_gap)):
            check_num(row, field, value, key)
        if row.get("actualLogRatio") is not None:
            raise ValueError(f"answer was attached to answer-free S manifest at {key}")
    if (doc.get("r106Replay", {}).get("n") != len(replay_gaps)
            or not replay_gaps
            or not math.isclose(float(doc["r106Replay"].get("maxAbsError", math.inf)), max(replay_gaps), rel_tol=1e-8, abs_tol=1e-12)
            or doc["r106Replay"].get("passed") is not True):
        raise ValueError("S R106 replay summary does not match verified forecast rows")


def _saved_factors_replay(a: np.ndarray, y: np.ndarray, audit: dict[str, Any]) -> np.ndarray:
    """Validate saved SVD evidence and replay LS without calling SVD."""
    u = _finite(audit["u"], "saved U", 2)
    singular = _finite(audit["singularValues"], "saved singular values", 1)
    vt = _finite(audit["vt"], "saved Vt", 2)
    keep = np.asarray(audit["retainedMask"], dtype=bool)
    if u.shape != (a.shape[0], len(singular)) or vt.shape != (len(singular), a.shape[1]) or keep.shape != singular.shape:
        raise ValueError("saved SVD shapes are inconsistent")
    if (audit.get("shape") != list(a.shape) or np.any(singular < 0)
            or np.any(np.diff(singular) > 0) or int(keep.sum()) != audit.get("retainedRank")
            or not np.allclose((u * singular[None, :]) @ vt, a, rtol=1e-9, atol=1e-9)
            or not np.allclose(u.T @ u, np.eye(len(singular)), rtol=1e-8, atol=1e-8)
            or not np.allclose(vt @ vt.T, np.eye(len(singular)), rtol=1e-8, atol=1e-8)):
        raise ValueError("saved SVD factors failed fit-free reconstruction checks")
    if keep.any():
        projected = u[:, keep].T @ y
        divisor = singular[keep] if projected.ndim == 1 else singular[keep, None]
        return vt[keep].T @ (projected / divisor)
    return np.zeros((a.shape[1],) + y.shape[1:], dtype=np.float64)


def verify_fit_free(x_scaled: Any, z: Any, r: Any, r_r: Any, fit: dict[str, Any]) -> None:
    """Verify B/gamma and stored training evidence without refitting or SVD."""
    x = _finite(x_scaled, "scaled training matrix", 2)
    hidden = _finite(z, "training hidden features", 2)
    target = _finite(r, "training log ratio", 1)
    ridge = _finite(r_r, "R training prediction", 1)
    a = np.column_stack((np.ones(len(x), dtype=np.float64), x))
    b = _saved_factors_replay(a, hidden, fit["A_svd"])
    z_perp = hidden - a @ b
    gamma = _saved_factors_replay(z_perp, target - ridge, fit["Zperp_svd"])
    if (not np.allclose(b, fit["B"], rtol=1e-10, atol=1e-10)
            or not np.allclose(gamma, fit["gamma"], rtol=1e-10, atol=1e-10)):
        raise ValueError("saved orthogonal residual coefficients do not replay")
    pred = ridge + z_perp @ gamma
    if not np.allclose(pred, fit["trainingPrediction"], rtol=1e-10, atol=1e-10):
        raise ValueError("saved training residual prediction does not replay")
    correction = pred - ridge
    saved_correction = _finite(fit.get("trainingCorrection"), "saved training correction", 1)
    if saved_correction.shape != correction.shape or not np.allclose(
            saved_correction, correction, rtol=1e-10, atol=1e-12):
        raise ValueError("saved training correction does not replay from fitted coefficients")
    eps = np.finfo(np.float64).eps
    s_a = np.asarray(fit["A_svd"]["singularValues"], dtype=np.float64)
    tau_a = float(eps * max(a.shape) * (s_a[0] if len(s_a) else 0.0))
    q_round = max(1.0, float(np.linalg.norm(hidden, ord="fro")),
                  float(np.linalg.norm(a, ord="fro") * np.linalg.norm(b, ord="fro")))
    s_z = np.asarray(fit["Zperp_svd"]["singularValues"], dtype=np.float64)
    tau_relative = float(eps * max(len(target), hidden.shape[1]) * (s_z[0] if len(s_z) else 0.0))
    tau_round = float(64 * eps * max(len(target), a.shape[1], hidden.shape[1]) * q_round)
    expected_tau_z = max(tau_relative, tau_round)
    if (fit.get("tauARule") != TAU_A_RULE or fit.get("tauZperpRule") != TAU_RESIDUAL_RULE
            or not math.isclose(float(fit.get("tauA", math.inf)), tau_a, rel_tol=1e-12, abs_tol=1e-15)
            or not math.isclose(float(fit["A_svd"].get("threshold", math.inf)), tau_a, rel_tol=1e-12, abs_tol=1e-15)
            or not math.isclose(float(fit.get("tauZperpRoundoff", math.inf)), tau_round, rel_tol=1e-12, abs_tol=1e-15)
            or not math.isclose(float(fit.get("tauZperpRelative", math.inf)), tau_relative, rel_tol=1e-12, abs_tol=1e-15)
            or not math.isclose(float(fit.get("tauZperp", math.inf)), expected_tau_z, rel_tol=1e-12, abs_tol=1e-15)
            or not math.isclose(float(fit["Zperp_svd"].get("threshold", math.inf)), expected_tau_z, rel_tol=1e-12, abs_tol=1e-15)):
        raise ValueError("saved SVD cutoff does not match the frozen rank rules")
    keep_a = np.asarray(fit["A_svd"]["retainedMask"], dtype=bool)
    keep_z = np.asarray(fit["Zperp_svd"]["retainedMask"], dtype=bool)
    if (not np.array_equal(keep_a, s_a > tau_a)
            or not np.array_equal(keep_z, s_z > expected_tau_z)):
        raise ValueError("saved SVD retained masks do not match frozen cutoffs")
    u_a = np.asarray(fit["A_svd"]["u"], dtype=np.float64)
    u_z = np.asarray(fit["Zperp_svd"]["u"], dtype=np.float64)
    z_norm = float(np.linalg.norm(hidden, ord="fro"))
    ab_norm = float(np.linalg.norm(a @ b, ord="fro"))
    orth_abs = float(np.linalg.norm(a.T @ z_perp, ord="fro"))
    orth_lim = float(tau_a * z_norm + np.linalg.norm(a, ord="fro") * tau_round
                     + ORTHOGONALITY_TOL * max(1.0, np.linalg.norm(a, ord="fro") * np.linalg.norm(z_perp, ord="fro")))
    q_u = float(np.linalg.norm(u_a[:, keep_a].T @ z_perp, ord="fro") /
                max(1.0, z_norm, ab_norm)) if keep_a.any() else 0.0
    if (orth_abs > orth_lim or q_u > ORTHOGONALITY_TOL
            or not math.isclose(float(fit.get("orthogonalityAbsoluteFro", math.inf)), orth_abs, rel_tol=1e-8, abs_tol=1e-12)
            or not math.isclose(float(fit.get("orthogonalityLimit", math.inf)), orth_lim, rel_tol=1e-8, abs_tol=1e-12)
            or not math.isclose(float(fit.get("qU", math.inf)), q_u, rel_tol=1e-8, abs_tol=1e-12)):
        raise ValueError("saved orthogonality evidence fails the rank-aware check")
    r_projection = u_a[:, keep_a] @ (u_a[:, keep_a].T @ ridge) if keep_a.any() else np.zeros_like(ridge)
    r_error = float(np.max(np.abs(ridge - r_projection)))
    r_limit = RECONSTRUCTION_TOL * max(1.0, float(np.max(np.abs(ridge))))
    correction = z_perp @ gamma
    e_r = target - ridge
    e_projection = u_z[:, keep_z] @ (u_z[:, keep_z].T @ e_r) if keep_z.any() else np.zeros_like(e_r)
    correction_error = float(np.max(np.abs(correction - e_projection)))
    correction_limit = RECONSTRUCTION_TOL * max(1.0, float(np.max(np.abs(e_r))))
    lr, ls = float(np.mean(e_r * e_r)), float(np.mean((target - pred) ** 2))
    slack = LOSS_ABS_TOL + LOSS_REL_TOL * max(1.0, lr, ls)
    if (r_error > r_limit or correction_error > correction_limit or ls > lr + slack
            or not math.isclose(float(fit.get("D_RA", math.inf)), float(np.mean((ridge-r_projection)**2)), rel_tol=1e-8, abs_tol=1e-12)
            or not math.isclose(float(fit.get("L_R", math.inf)), lr, rel_tol=1e-10, abs_tol=1e-12)
            or not math.isclose(float(fit.get("L_S", math.inf)), ls, rel_tol=1e-10, abs_tol=1e-12)):
        raise ValueError("saved retained-space reconstruction or training loss evidence failed")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "preflight", "build", "verify", "score", "final"))
    parser.add_argument("--phase", choices=tuple(PHASES), default="validation")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze":
        result = freeze_protocol()
    elif args.command == "preflight":
        result = preflight(args.phase)
    elif args.command == "build":
        result = build_phase(args.phase)
    elif args.command == "verify":
        path = _manifest_path(args.phase)
        doc = json.loads(path.read_text(encoding="utf-8"))
        verify_manifest(doc, args.phase, path)
        result = {"phase": args.phase, "status": "verified_fit_free", "manifestSha256": doc.get("manifestSha256")}
    elif args.command == "score":
        result = score_phase(args.phase)
    else:
        result = run_final_if_eligible()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    main()
