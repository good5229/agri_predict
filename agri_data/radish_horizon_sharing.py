"""Frozen implementation for the radish horizon-sharing retrospective.

The numerical core is deliberately source-free.  The command line separates
protocol freezing, input preflight, answer-free forecast construction and
answer-bearing scoring so those stages can be reviewed independently.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/radish-horizon-sharing-v1-20261011"
BASELINE_PATH = ROOT / "outputs/experiments/equal-info-online-20261010/forecast-manifest.json"
PRICES_PATH = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
FIXED50_PATH = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/fixed50-manifest.json"
E106_COMPOSITION_PATHS = {
    "V": ROOT / "outputs/experiments/preserved-improvement-composition-20261011/confirmation-manifest.json",
    "T": ROOT / "outputs/experiments/preserved-improvement-composition-20261011/final-manifest.json",
}
E106_COMPOSITION_SHA256 = {
    "V": "1c32c11f0db65c1f06724cf8ae4ae02bd3455eedb1bafcdf827c9b0fef7e473b",
    "T": "c45cc99de60cc9df752194e694c4de8984d424c4edc03de2e283c23ddea9bbe2",
}
PHASES = {"V": (dt.date(2026, 4, 1), dt.date(2026, 6, 20), dt.date(2026, 4, 1)),
          "T": (dt.date(2026, 7, 11), dt.date(2026, 9, 27), dt.date(2026, 7, 1))}
HORIZONS = tuple(range(1, 11))
CONDITIONS = ("B", "D", "C", "P")
FEATURES = ("lastPriceLog", "horizonDays", "sourceAgeDays") + tuple(
    x for lag in (1, 7, 28, 90) for x in (f"lag{lag}PriceLog", f"lag{lag}AgeDays", f"lag{lag}Available")
) + tuple(x for w in (7, 28, 90) for x in (f"window{w}{suffix}" for suffix in
    ("MeanLogPrice", "StdLogPrice", "MinLogPrice", "MaxLogPrice", "ObservedDays", "Coverage", "LogVolumeKg"))) + (
    "targetWeekdaySin", "targetWeekdayCos", "targetYearDaySin", "targetYearDayCos"
)
assert len(FEATURES) == 40
ACTIVE = tuple(i for i, name in enumerate(FEATURES) if name != "horizonDays")
N_FEATURE = len(ACTIVE)
N_HORIZON = 10
N_COEF = N_FEATURE * N_HORIZON
ALPHA = 10.0
EDF_FRACTION = 0.75
EDF_USER_TOL = 1e-6
SOLVE_REL_TOL = 1e-8
PRICE_ABS_TOL = 1e-6
PRICE_REL_TOL = 1e-8
EDF_MAX_BRACKET = 64
EDF_MAX_ITER = 100
PINNED_BASELINE_SHA256 = "d7d74823754c2a48d33cc4ddaeab674c686601fb927d399d84c5dbcd235184df"
PINNED_EQUAL_INFO_SHA256 = "84e68a9a3eb362e59a336440a0bf9254eacb68dfb881e6cf3abdeeba4a215471"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_array(value: Any, *, ndim: int | None = None, name: str = "array") -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _path_laplacian(order: Sequence[int]) -> np.ndarray:
    if sorted(order) != list(range(N_HORIZON)):
        raise ValueError("path order must be a permutation of horizons 1..10")
    lap = np.zeros((N_HORIZON, N_HORIZON), dtype=np.float64)
    for a, b in zip(order[:-1], order[1:]):
        lap[a, a] += 1.0
        lap[b, b] += 1.0
        lap[a, b] -= 1.0
        lap[b, a] -= 1.0
    return lap


PATH_C = _path_laplacian(tuple(range(10)))
PATH_P = _path_laplacian((0, 5, 1, 6, 2, 7, 3, 8, 4, 9))


def graph_penalties(scales: Sequence[Sequence[float]]) -> dict[str, np.ndarray]:
    """Return transformed D/C/P penalties in the h=1 standardized coordinate."""
    s = _finite_array(scales, ndim=2, name="scales")
    if s.shape != (N_HORIZON, N_FEATURE) or np.any(s <= 0):
        raise ValueError("scales must be positive (10, 39)")
    d = np.zeros((N_COEF, N_COEF), dtype=np.float64)
    for h in range(N_HORIZON):
        sl = slice(h * N_FEATURE, (h + 1) * N_FEATURE)
        d[sl, sl] = np.diag(s[0] / s[h])
    identity = np.eye(N_FEATURE, dtype=np.float64)
    pc = d.T @ np.kron(PATH_C, identity) @ d
    pp = d.T @ np.kron(PATH_P, identity) @ d
    pd = np.diag(np.diag(pc))
    return {"D": pd, "C": pc, "P": pp}


def _check_psd(matrix: np.ndarray, *, name: str) -> dict[str, float]:
    matrix = _finite_array(matrix, ndim=2, name=name)
    if matrix.shape != (N_COEF, N_COEF) or not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} must be a symmetric 390x390 matrix")
    norm = float(np.linalg.norm(matrix, 2))
    min_eigen = float(np.linalg.eigvalsh(matrix)[0])
    tol = 64 * np.finfo(np.float64).eps * N_COEF * max(1.0, norm)
    if min_eigen < -tol:
        raise ValueError(f"{name} is not positive semidefinite")
    return {"spectralNorm": norm, "minimumEigenvalue": min_eigen, "psdTolerance": tol}


def _spd_solve(matrix: np.ndarray, rhs: np.ndarray, *, name: str) -> np.ndarray:
    try:
        return cho_solve(cho_factor(matrix, lower=True, check_finite=True), rhs, check_finite=True)
    except Exception as exc:
        raise ArithmeticError(f"{name} is not numerically positive definite") from exc


def _edf(g: np.ndarray, q: np.ndarray, factor: Any | None = None) -> float:
    cf = factor if factor is not None else cho_factor(q, lower=True, check_finite=True)
    return float(np.trace(cho_solve(cf, g, check_finite=True)))


def solve_edf_lambda(g: np.ndarray, q0: np.ndarray, penalty: np.ndarray, target: float,
                     *, df0: float, df_inf: float, user_tolerance: float, q0_factor: Any | None = None) -> dict[str, Any]:
    """Solve the monotone training-only EDF equation with float64 Cholesky."""
    g = _finite_array(g, ndim=2, name="G")
    q0 = _finite_array(q0, ndim=2, name="Q0")
    p = _finite_array(penalty, ndim=2, name="penalty")
    if g.shape != (N_COEF, N_COEF) or q0.shape != g.shape or p.shape != g.shape:
        raise ValueError("EDF matrices must be 390x390")
    if not (math.isfinite(target) and math.isfinite(df0) and math.isfinite(df_inf)):
        raise ValueError("EDF values must be finite")
    user_tol = float(user_tolerance)
    gap = df0 - df_inf
    if gap <= user_tol:
        return {"status": "intervention_not_identifiable", "df0": df0, "dfInf": df_inf,
                "dfTarget": target, "edfTolerance": user_tol}
    if target >= df0 or target <= df_inf:
        return {"status": "numerical_failure", "reason": "target_not_inside_edf_limits", "df0": df0,
                "dfInf": df_inf, "dfTarget": target, "edfTolerance": user_tol}
    try:
        q0factor = q0_factor if q0_factor is not None else cho_factor(q0, lower=True, check_finite=True)
        actual0 = _edf(g, q0, q0factor)
        pnorm = float(np.linalg.norm(p, 2))
        qnorm = float(np.linalg.norm(q0, 2))
        if not all(math.isfinite(v) for v in (actual0, pnorm, qnorm)) or pnorm <= 0 or qnorm <= 0:
            raise ArithmeticError("invalid endpoint norms")
        if abs(actual0 - df0) > user_tol:
            raise ArithmeticError("df0 factorization check failed")
        lo = 0.0
        hi = qnorm / pnorm
        if not math.isfinite(hi) or hi <= 0:
            raise ArithmeticError("invalid initial EDF bracket")
        def at(lam: float) -> tuple[float, Any]:
            q = q0 + lam * p
            cf = cho_factor(q, lower=True, check_finite=True)
            return _edf(g, q, cf), cf
        ehi, _ = at(hi)
        expansions = 0
        while ehi > target and expansions < EDF_MAX_BRACKET:
            hi *= 2.0
            if not math.isfinite(hi):
                raise ArithmeticError("EDF bracket overflow")
            ehi, _ = at(hi)
            expansions += 1
        if ehi > target:
            raise ArithmeticError("EDF root not bracketed")
        internal_tol = min(user_tol, 1e-3 * 0.25 * gap)
        lam = None
        actual = None
        for iterations in range(1, EDF_MAX_ITER + 1):
            mid = lo + (hi - lo) * 0.5
            if mid <= 0 or mid == lo or mid == hi:
                break
            emid, _ = at(mid)
            if emid > target:
                lo = mid
            else:
                hi = mid
            lam = mid
            actual = emid
            if abs(emid - target) <= internal_tol:
                break
        if lam is None:
            lam = hi
            actual, _ = at(lam)
            iterations = 1
        if abs(actual - target) > user_tol or lam <= 0 or actual >= df0:
            raise ArithmeticError("EDF root failed final positive-intervention check")
        return {"status": "ok", "lambda": float(lam), "df0": float(df0), "dfInf": float(df_inf),
                "dfTarget": float(target), "actualDf": float(actual), "edfError": float(actual-target),
                "edfTolerance": user_tol, "internalTolerance": internal_tol, "bracketExpansions": expansions,
                "iterations": iterations, "lambdaBracket": [float(lo), float(hi)]}
    except Exception as exc:
        return {"status": "numerical_failure", "reason": f"{type(exc).__name__}:{exc}", "df0": df0,
                "dfInf": df_inf, "dfTarget": target, "edfTolerance": user_tol}


def _shared_limit(g: np.ndarray, q0: np.ndarray, scales: np.ndarray) -> float:
    # Tshared stacks D_h^-1 = diag(scale_h / scale_ref), in the specified order.
    t = np.concatenate([np.diag(scales[h] / scales[0]) for h in range(N_HORIZON)], axis=0)
    gt = t.T @ g @ t
    qt = t.T @ q0 @ t
    return float(np.trace(_spd_solve(qt, gt, name="shared nullspace system")))


def _solve_beta(g: np.ndarray, q0: np.ndarray, penalties: Mapping[str, np.ndarray],
                lambdas: Mapping[str, float], rhs: np.ndarray, q0_factor: Any | None = None) -> dict[str, Any]:
    out = {}
    for name in CONDITIONS:
        q = q0 if name == "B" else q0 + float(lambdas[name]) * penalties[name]
        beta = cho_solve(q0_factor, rhs, check_finite=True) if name == "B" and q0_factor is not None else _spd_solve(q, rhs, name=f"Q_{name}")
        residual = q @ beta - rhs
        denominator = float(np.linalg.norm(q, 2) * np.linalg.norm(beta) + np.linalg.norm(rhs))
        relative = float(np.linalg.norm(residual) / denominator) if denominator else float(np.linalg.norm(residual))
        if not np.isfinite(beta).all() or not math.isfinite(relative) or relative > SOLVE_REL_TOL:
            raise ArithmeticError(f"{name} solve residual exceeds tolerance")
        out[name] = {"beta": beta, "relativeResidual": relative, "Q": q}
    return out


def fit_joint(X_by_h: Sequence[Any], y_by_h: Sequence[Any], scales_by_h: Sequence[Any], *,
              fit_id: str, feature_names: Sequence[str] = FEATURES) -> dict[str, Any]:
    """Fit B/D/C/P for one monthly fold using each horizon's own rows."""
    if len(X_by_h) != N_HORIZON or len(y_by_h) != N_HORIZON or len(scales_by_h) != N_HORIZON:
        raise ValueError("exactly ten horizon datasets are required")
    if tuple(feature_names) != tuple(FEATURES):
        raise ValueError("the frozen 40-feature order changed")
    xs, ys, means, ymeans, scales = [], [], [], [], []
    for h in range(N_HORIZON):
        x = _finite_array(X_by_h[h], ndim=2, name=f"X[{h+1}]")
        y = _finite_array(y_by_h[h], ndim=1, name=f"y[{h+1}]")
        if x.shape != (len(y), 40):
            raise ValueError(f"h={h+1} matrix violates frozen 40-column feature schema")
        if len(y) < 90:
            raise ValueError(f"h={h+1} has fewer than 90 training rows")
        s = _finite_array(scales_by_h[h], ndim=1, name=f"scale[{h+1}]")
        if s.size != 40 or np.any(s <= 0):
            raise ValueError("scaler scale must contain 40 positive values")
        x = x[:, list(ACTIVE)]
        s = s[list(ACTIVE)]
        # Caller supplies StandardScaler-transformed X and its scale_.
        xm = x.mean(axis=0)
        ym = float(y.mean())
        xs.append(x - xm)
        ys.append(y - ym)
        means.append(xm)
        ymeans.append(ym)
        scales.append(s)
    scales_a = np.asarray(scales, dtype=np.float64)
    g = np.zeros((N_COEF, N_COEF), dtype=np.float64)
    rhs = np.zeros(N_COEF, dtype=np.float64)
    for h, (x, y) in enumerate(zip(xs, ys)):
        sl = slice(h*N_FEATURE, (h+1)*N_FEATURE)
        g[sl, sl] = x.T @ x
        rhs[sl] = x.T @ y
    q0 = g + ALPHA * np.eye(N_COEF, dtype=np.float64)
    penalties = graph_penalties(scales_a)
    psd = {name: _check_psd(penalties[name], name=f"P_{name}") for name in ("D", "C", "P")}
    if not np.allclose(np.linalg.eigvalsh(PATH_C), np.linalg.eigvalsh(PATH_P), atol=1e-12, rtol=0):
        raise AssertionError("C/P path spectra differ")
    q0_factor=cho_factor(q0,lower=True,check_finite=True)
    df0 = _edf(g, q0, q0_factor)
    dfinf = _shared_limit(g, q0, scales_a)
    edf_bound_tol=1e-10*max(1.0,abs(df0),abs(dfinf))
    if not (math.isfinite(df0) and math.isfinite(dfinf) and -edf_bound_tol <= dfinf <= df0+edf_bound_tol):
        return {"status":"numerical_failure","fitId":fit_id,"reason":"dfInf_outside_[0,df0]",
                "df0":df0,"dfInf":dfinf,"edfBoundTolerance":edf_bound_tol}
    gap = df0 - dfinf
    tolerance = EDF_USER_TOL * max(1.0, df0)
    target = dfinf + EDF_FRACTION * gap
    if gap <= tolerance:
        return {"status": "intervention_not_identifiable", "fitId": fit_id,
                "df0": df0, "dfInf": dfinf, "dfTarget": target, "edfGap": gap,
                "edfTolerance": tolerance}
    lambdas: dict[str, float] = {"B": 0.0}
    edf_audits = {}
    for name in ("D", "C", "P"):
        audit = solve_edf_lambda(g, q0, penalties[name], target, df0=df0, df_inf=dfinf,
                                 user_tolerance=tolerance,q0_factor=q0_factor)
        edf_audits[name] = audit
        if audit["status"] != "ok":
            return {"status": audit["status"], "fitId": fit_id, "reason": audit.get("reason"),
                    "edf": {"df0": df0, "dfInf": dfinf, "dfTarget": target, "gap": gap,
                            "tolerance": tolerance, "conditions": edf_audits}}
        lambdas[name] = audit["lambda"]
    solutions = _solve_beta(g, q0, penalties, lambdas, rhs, q0_factor)
    # B is the stored lambda-zero Q0 block solve; verify it through the system
    # residual rather than refitting or building a second estimator.
    lambda0_residual=q0@solutions["B"]["beta"]-rhs
    lambda0_denominator=float(np.linalg.norm(q0,2)*np.linalg.norm(solutions["B"]["beta"])+np.linalg.norm(rhs))
    lambda0_relative=float(np.linalg.norm(lambda0_residual)/lambda0_denominator) if lambda0_denominator else float(np.linalg.norm(lambda0_residual))
    if not math.isfinite(lambda0_relative) or lambda0_relative>SOLVE_REL_TOL:
        raise ArithmeticError("lambda-zero joint system check failed")
    prediction = {}
    for name in CONDITIONS:
        values = []
        beta = solutions[name]["beta"].reshape(N_HORIZON, N_FEATURE)
        for h, x in enumerate(xs):
            values.append((x @ beta[h] + ymeans[h]).tolist())
        prediction[name] = values
    beta_by_condition = {name: solutions[name]["beta"].reshape(N_HORIZON, N_FEATURE) for name in CONDITIONS}
    coordinate_ratio = scales_a[0][None, :] / scales_a
    qcoord = {name: coordinate_ratio * beta_by_condition[name] for name in CONDITIONS}
    def f0_value(beta_flat: np.ndarray) -> float:
        bmat = beta_flat.reshape(N_HORIZON, N_FEATURE)
        return math.fsum(float(np.sum((y - xs[h] @ bmat[h]) ** 2)) for h, y in enumerate(ys)) + ALPHA * float(beta_flat @ beta_flat)
    f0_b = f0_value(solutions["B"]["beta"])
    diagnostics = {}
    edf_by_horizon = {}
    penalty_at_beta = {penalty_name: {beta_name: float(solutions[beta_name]["beta"] @ penalties[penalty_name] @ solutions[beta_name]["beta"])
                                       for beta_name in CONDITIONS} for penalty_name in ("D", "C", "P")}
    objective_at_beta = {}
    comparison_checks = {}
    for name in ("D", "C", "P"):
        beta_a = solutions[name]["beta"]
        j_a = f0_value(beta_a) + lambdas[name] * penalty_at_beta[name][name]
        j_b = f0_b + lambdas[name] * penalty_at_beta[name]["B"]
        slack_pen = 1e-12 + 1e-10 * max(1.0, abs(penalty_at_beta[name][name]), abs(penalty_at_beta[name]["B"]))
        slack_obj = 1e-12 + 1e-10 * max(1.0, abs(j_a), abs(j_b))
        slack_f0 = 1e-12 + 1e-10 * max(1.0, abs(f0_b), abs(f0_value(beta_a)))
        if penalty_at_beta[name][name] > penalty_at_beta[name]["B"] + slack_pen:
            raise ArithmeticError(f"{name} added penalty is not lower than B within tolerance")
        if j_a > j_b + slack_obj:
            raise ArithmeticError(f"{name} objective does not improve over B")
        if f0_b > f0_value(beta_a) + slack_f0:
            raise ArithmeticError(f"B base objective is not minimal")
        objective_at_beta[name] = {"J_at_candidate": j_a, "J_at_B": j_b, "F0_at_B": f0_b, "F0_at_candidate": f0_value(beta_a)}
        comparison_checks[name] = {"penaltyAtCandidate": penalty_at_beta[name][name], "penaltyAtB": penalty_at_beta[name]["B"],
                                   "penaltySlack": slack_pen, "objectiveSlack": slack_obj, "baseObjectiveSlack": slack_f0}
    for name in CONDITIONS:
        beta = solutions[name]["beta"]
        extra = float(beta @ penalties[name] @ beta) if name != "B" else 0.0
        base_pen = float(ALPHA * beta @ beta)
        sse = math.fsum(float(np.sum((y - x @ beta[h*N_FEATURE:(h+1)*N_FEATURE])**2))
                        for h, (x, y) in enumerate(zip(xs, ys)))
        diagnostics[name] = {"lambda": lambdas[name], "actualEdf": df0 if name == "B" else edf_audits[name]["actualDf"],
                             "totalEdfIncludingIntercepts": (df0 if name == "B" else edf_audits[name]["actualDf"]) + 10,
                             "relativeSolveResidual": solutions[name]["relativeResidual"],
                             "basePenalty": base_pen, "extraPenalty": extra, "objective": sse + base_pen + lambdas[name]*extra,
                             "F0": sse + base_pen,
                             "trainLogMseByHorizon": [float(np.mean((y - xs[h] @ beta[h*N_FEATURE:(h+1)*N_FEATURE])**2)) for h,y in enumerate(ys)]}
        q=q0 if name=="B" else q0+lambdas[name]*penalties[name]
        q_factor=cho_factor(q,lower=True,check_finite=True)
        qinv_g=cho_solve(q_factor,g,check_finite=True)
        by_h=[float(np.trace(qinv_g[h*N_FEATURE:(h+1)*N_FEATURE,h*N_FEATURE:(h+1)*N_FEATURE])) for h in range(N_HORIZON)]
        if abs(math.fsum(by_h)-diagnostics[name]["actualEdf"]) > 1e-8*max(1.0,diagnostics[name]["actualEdf"]):
            raise ArithmeticError(f"{name} per-horizon EDF does not sum to total EDF")
        edf_by_horizon[name]=by_h
    return {"status": "ok", "fitId": fit_id, "featureNames": [FEATURES[i] for i in ACTIVE],
            "horizonDaysCoefficient": 0.0, "scalerXMean": [m.tolist() for m in means],
            "scalerYMean": ymeans, "scalerScale": scales_a.tolist(), "trainingCounts": [len(y) for y in ys],
            "fitCoefficients": {name: solutions[name]["beta"].reshape(N_HORIZON, N_FEATURE).tolist() for name in CONDITIONS},
            "fitIntercepts": {name: [float(ymeans[h] - means[h] @ beta_by_condition[name][h]) for h in range(N_HORIZON)] for name in CONDITIONS},
            "sharedCoordinateCoefficients": {name: qcoord[name].tolist() for name in CONDITIONS},
            "sharedCoordinateChangesVsB": {name: (qcoord[name] - qcoord["B"]).tolist() for name in ("D", "C", "P")},
            "adjacentCoefficientDifferenceSq": {name: [float(np.sum((qcoord[name][h] - qcoord[name][h-1])**2)) for h in range(1,N_HORIZON)] for name in CONDITIONS},
            "lambdas": lambdas, "edf": {"df0": df0, "dfInf": dfinf, "dfTarget": target,
                    "gap": gap, "tolerance": tolerance, "boundTolerance":edf_bound_tol,
                    "conditions": edf_audits,"actualByHorizon":edf_by_horizon},
            "graph": {"degreeC": np.diag(PATH_C).astype(int).tolist(), "degreeP": np.diag(PATH_P).astype(int).tolist(),
                      "laplacianC": PATH_C.tolist(), "laplacianP": PATH_P.tolist(), "penaltyChecks": psd},
            "diagnostics": diagnostics, "penaltyAtBeta": penalty_at_beta, "objectiveChecks": objective_at_beta,
            "comparisonToleranceChecks": comparison_checks, "lambda0Check":{"status":"passed","relativeResidual":lambda0_relative,"reusesQ0Factorization":True},
            "trainingPredictions": prediction}


def predict_saved(fit: Mapping[str, Any], X_by_h: Sequence[Any]) -> dict[str, list[float]]:
    if fit.get("status") != "ok":
        raise ValueError("fit state is not usable")
    if len(X_by_h) != 10:
        raise ValueError("ten evaluation matrices are required")
    result = {name: [] for name in CONDITIONS}
    means = np.asarray(fit["scalerXMean"], dtype=np.float64)
    ymeans = np.asarray(fit["scalerYMean"], dtype=np.float64)
    for h, raw in enumerate(X_by_h):
        x = _finite_array(raw, ndim=2, name=f"forecastX[{h+1}]")
        if x.shape[1] != 40:
            raise ValueError("forecast matrix must preserve the 40-column feature order")
        x = x[:, ACTIVE] - means[h]
        for name in CONDITIONS:
            beta = np.asarray(fit["fitCoefficients"][name], dtype=np.float64)[h]
            result[name].extend((x @ beta + ymeans[h]).tolist())
    return result


def _source_protocol() -> dict[str, Any]:
    import sklearn, statsmodels
    from . import equal_info_online_forecasts as base
    from . import equal_info_online_score as inference
    from . import radish_horizon_sharing_score as scorer
    if (platform.python_version(), sklearn.__version__, statsmodels.__version__) != ("3.11.4", "1.3.0", "0.14.0"):
        raise RuntimeError("RHS must use the pinned E106 Anaconda Python 3.11.4/sklearn 1.3.0/statsmodels 0.14.0 runtime")
    if tuple(base.modeling.FEATURE_NAMES) != tuple(FEATURES):
        raise ValueError("RHS feature declaration differs from the frozen modeling helper")
    return {"id": "radish-horizon-sharing-v1", "candidate": "E106-RHS/C", "products": ["cabbage", "radish", "onion", "garlic"],
            "horizons": list(HORIZONS), "phases": {k: {"targetStart": v[0].isoformat(), "targetEnd": v[1].isoformat(), "originStart": v[2].isoformat()} for k,v in PHASES.items()},
            "conditions": {"B": "E106 direct Ridge alpha=10", "D": "diagonal of C graph penalty", "C": "adjacent path 1..10", "P": "path 1,6,2,7,3,8,4,9,5,10"},
            "fitBudgetByPhase": {"jointMonthlySystems": 3, "BIndependentHorizonCoefficientSolutions": 30,
                                 "DCPSharedConditionSystems": 9, "sklearnRidgeEstimatorFitCalls": 0},
            "alpha": ALPHA, "target": "log(actual/p0)", "features": list(base.modeling.FEATURE_NAMES),
            "constantFeature": "horizonDays coefficient fixed zero; preserve 40-column feature schema", "minimumTrainingRows": 90,
            "refit": "first eligible origin each calendar month; targets strictly before fit origin; feature cutoff is each row origin-1",
            "coordinate": "D_h=diag(scale_ref/scale_h), scale_ref from horizon 1; independent intercepts",
            "edf": {"target": "dfInf + 0.75*(df0-dfInf)", "tolerance": "1e-6*max(1,df0)", "Q0": "G+10I", "solver": "float64 Cholesky; EDF=trace(solve(Q,G))", "maxBracketDoublings": EDF_MAX_BRACKET, "maxBisections": EDF_MAX_ITER},
            "validationGate": "C macro radish 10-horizon same-SARIMA blend MAPE strictly lower than E106/B, MAE no higher, no coverage loss; T only after V pass",
            "inference": {"familySizePerBandwidth": 60, "comparisons": ["C-B", "C-D", "C-P"], "horizons": 10, "losses": ["AE", "APE"], "HAC": [9,28], "HolmAlpha": .05, "minPairs": 30, "bootstrapBlocks": [14,28], "bootstrapReplicates": 2000, "seed": 0},
            "inputSha256": {"baselineManifest": PINNED_BASELINE_SHA256, "prices": base.PRICES_SHA256, "fixed50": base.FIXED50_SHA256,
                            "e106CompositionV": E106_COMPOSITION_SHA256["V"], "e106CompositionT": E106_COMPOSITION_SHA256["T"],
                            "source": file_sha(Path(__file__)), "equalInfoModule": PINNED_EQUAL_INFO_SHA256,
                            "modeling": base.MODELING_SHA256, "narrow": base.NARROW_HELPER_SHA256,
                            "rich": base.RICH_HELPER_SHA256, "featureLibrary": base.LIBRARY_SHA256,
                            "directHorizon": base.DIRECT_HELPER_SHA256, "sarima": base.TEN_HELPER_SHA256,
                            "scorer": file_sha(Path(scorer.__file__)),
                            "inferenceHelper": file_sha(Path(inference.__file__))},
            "runtime": {"python": platform.python_version(), "numpy": np.__version__, "scikitLearn": sklearn.__version__, "statsmodels": statsmodels.__version__}}


def freeze_protocol(output_dir: Path = OUT) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "protocol.json"
    doc = _source_protocol()
    payload = {"protocol": doc, "protocolSha256": sha256_bytes(canonical(doc))}
    raw = canonical(payload)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return payload


def _load_protocol(output_dir: Path = OUT) -> dict[str, Any]:
    doc = _json(output_dir / "protocol.json")
    if doc.get("protocolSha256") != sha256_bytes(canonical(doc.get("protocol"))):
        raise ValueError("frozen protocol digest mismatch")
    current = _source_protocol()
    if doc["protocol"] != current:
        raise ValueError("frozen protocol differs from current code, inputs, or runtime")
    return doc


def _composition_index(phase: str) -> dict[tuple[str, str, str, int], Mapping[str, Any]]:
    """Load the immutable composed E106 control for the same evaluation phase."""
    if phase not in PHASES:
        raise ValueError("phase must be V or T")
    path = E106_COMPOSITION_PATHS[phase]
    if file_sha(path) != E106_COMPOSITION_SHA256[phase]:
        raise ValueError(f"pinned composed E106 {phase} manifest changed")
    doc = _json(path)
    unsigned = dict(doc)
    stored = unsigned.pop("manifestSha256", None)
    if stored != sha256_bytes(canonical(unsigned).rstrip(b"\n")) or doc.get("targetValuesAttached") is not False:
        raise ValueError("composed E106 manifest digest/answer-free check failed")
    expected_phase = "preserved_improvement_confirmation_forecast_only" if phase == "V" else "preserved_improvement_final_forecast_only"
    if doc.get("phase") != expected_phase:
        raise ValueError("composed E106 manifest phase mismatch")
    start, end, origin_start = PHASES[phase]
    output = {}
    for row in doc.get("forecasts", []):
        product, origin, target, horizon = row.get("product"), row.get("origin"), row.get("target"), row.get("horizon")
        if product not in ("cabbage", "radish", "onion", "garlic") or type(horizon) is not int or horizon not in HORIZONS:
            raise ValueError("invalid composed E106 key")
        od, td = dt.date.fromisoformat(origin), dt.date.fromisoformat(target)
        if td - od != dt.timedelta(days=horizon) or not start <= td <= end:
            raise ValueError("composed E106 row lies outside the fixed forecast grid")
        key = (product, origin, target, horizon)
        if key in output:
            raise ValueError("duplicate composed E106 key")
        if row.get("eligible") is not (od >= origin_start):
            raise ValueError("composed E106 eligibility mismatch")
        if not isinstance(row.get("candidate"), dict) or row["candidate"].get("status") not in ("issued", "not_issued"):
            raise ValueError("invalid composed E106 candidate row")
        output[key] = row
    expected = {(p, (start + dt.timedelta(days=i) - dt.timedelta(days=h)).isoformat(),
                 (start + dt.timedelta(days=i)).isoformat(), h)
                for p in ("cabbage", "radish", "onion", "garlic")
                for i in range((end-start).days+1) for h in HORIZONS
                if start + dt.timedelta(days=i) - dt.timedelta(days=h) >= origin_start}
    if set(output) != expected:
        raise ValueError("composed E106 manifest does not contain the exact phase grid")
    return output


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _claim_run(output_dir: Path, phase: str, protocol_sha: str) -> dict[str, Any]:
    path=output_dir/("validation-claim.json" if phase=="V" else "test-claim.json")
    terminal=output_dir/("validation-terminal.json" if phase=="V" else "test-terminal.json")
    if terminal.exists():
        term=_json(terminal)
        if term.get("status")=="complete" and (output_dir/("validation-manifest.json" if phase=="V" else "test-manifest.json")).exists():
            return _json(path)
        raise ValueError("phase already has an immutable terminal record")
    if path.exists():
        claim=_json(path)
        if claim.get("protocolSha256")!=protocol_sha or claim.get("phase")!=phase:
            raise ValueError("existing phase claim is bound to another protocol")
        return claim
    claim={"schemaVersion":1,"phase":phase,"attemptId":str(uuid.uuid4()),"protocolSha256":protocol_sha,
           "startedAtKst":dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(),"status":"running"}
    _write_exclusive(path,canonical(claim))
    return claim


def _terminal_run(output_dir: Path, phase: str, status: str, *, manifest_sha: str | None = None,
                  artifact_sha: str | None = None, reason: str | None = None) -> None:
    claim_path=output_dir/("validation-claim.json" if phase=="V" else "test-claim.json")
    terminal_path=output_dir/("validation-terminal.json" if phase=="V" else "test-terminal.json")
    claim=_json(claim_path)
    payload={"schemaVersion":1,"phase":phase,"attemptId":claim["attemptId"],"protocolSha256":claim["protocolSha256"],
             "status":status,"manifestSha256":manifest_sha,"artifactSha256":artifact_sha,"reason":reason,
             "finishedAtKst":dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()}
    _write_exclusive(terminal_path,canonical(payload))


def _fit_checkpoint(output_dir: Path, phase: str, fit_id: str, input_hashes: Mapping[str,str], fit_factory) -> dict[str, Any]:
    fits_dir=output_dir/"fit-checkpoints"; fits_dir.mkdir(parents=True,exist_ok=True)
    safe=fit_id.replace(":","-")
    claim_path=fits_dir/f"{phase.lower()}-{safe}.claim.json"
    checkpoint_path=fits_dir/f"{phase.lower()}-{safe}.json"
    terminal_path=fits_dir/f"{phase.lower()}-{safe}.terminal.json"
    input_hash=sha256_bytes(canonical(input_hashes))
    if checkpoint_path.exists():
        doc=_json(checkpoint_path)
        if doc.get("inputSha256")!=input_hash or doc.get("fitId")!=fit_id or doc.get("phase")!=phase:
            raise ValueError(f"fit checkpoint input binding mismatch: {fit_id}")
        content=doc.get("fit")
        if sha256_bytes(canonical(content))!=doc.get("checkpointSha256"):
            raise ValueError(f"fit checkpoint digest mismatch: {fit_id}")
        if claim_path.exists():
            claim=_json(claim_path)
            if claim.get("inputSha256")!=input_hash or claim.get("fitId")!=fit_id or claim.get("phase")!=phase:
                raise ValueError(f"fit claim/checkpoint binding mismatch: {fit_id}")
        if not terminal_path.exists():
            _write_exclusive(terminal_path,canonical({"phase":phase,"fitId":fit_id,"status":content.get("status","complete_recovered_from_checkpoint"),
                                                       "inputSha256":input_hash,"checkpointSha256":doc["checkpointSha256"]}))
        return content
    if claim_path.exists():
        claim=_json(claim_path)
        raise ValueError(f"fit has a prior running claim but no checkpoint; refusing silent refit: {fit_id}")
    claim={"schemaVersion":1,"phase":phase,"fitId":fit_id,"status":"running","inputSha256":input_hash,
           "inputHashes":dict(input_hashes),
           "trainingKeySha256":input_hashes.get("trainingKeys"),"trainingMatrixSha256":input_hashes.get("trainingMatrix"),
           "trainingTargetSha256":input_hashes.get("trainingTarget"),"claimedAtKst":dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()}
    _write_exclusive(claim_path,canonical(claim))
    fit=fit_factory()
    digest=sha256_bytes(canonical(fit))
    _write_exclusive(checkpoint_path,canonical({"phase":phase,"fitId":fit_id,"inputSha256":input_hash,"checkpointSha256":digest,"fit":fit}))
    _write_exclusive(terminal_path,canonical({"phase":phase,"fitId":fit_id,"status":fit.get("status","complete"),"inputSha256":input_hash,"checkpointSha256":digest}))
    return fit


def _baseline_index(manifest: Mapping[str, Any], phase: str) -> dict[tuple[str, str, str, int], Mapping[str, Any]]:
    if manifest.get("schemaVersion") != 1 or manifest.get("protocol", {}).get("id") != "equal-info-online-20261010-v1":
        raise ValueError("unexpected pinned baseline manifest")
    forecasts = manifest.get("evaluationForecasts")
    if not isinstance(forecasts, list):
        raise ValueError("baseline has no forecast grid")
    start, end, origin_start = PHASES[phase]
    output = {}
    for row in forecasts:
        if row.get("product") not in ("cabbage", "radish", "onion", "garlic"):
            continue
        target = dt.date.fromisoformat(row["target"])
        origin = dt.date.fromisoformat(row["origin"])
        if start <= target <= end:
            key = (row["product"], row["origin"], row["target"], int(row["horizon"]))
            if key in output: raise ValueError("duplicate baseline key")
            output[key] = row
    return output


def _preflight(phase: str, output_dir: Path = OUT) -> dict[str, Any]:
    _load_protocol(output_dir)
    from . import equal_info_online_forecasts as base
    for path, expected, label in ((BASELINE_PATH, _source_protocol()["inputSha256"]["baselineManifest"], "baseline"),
                                  (base.PRICES_PATH, base.PRICES_SHA256, "prices"),
                                  (base.FIXED50_PATH, base.FIXED50_SHA256, "fixed50")):
        if file_sha(path) != expected:
            raise ValueError(f"pinned {label} hash changed")
    _composition_index(phase)
    expected_modules={base.__file__:PINNED_EQUAL_INFO_SHA256,base.modeling.__file__:base.MODELING_SHA256,
                      base.narrow.__file__:base.NARROW_HELPER_SHA256,base.rich.__file__:base.RICH_HELPER_SHA256,
                      base.library.__file__:base.LIBRARY_SHA256,base.direct_horizon.__file__:base.DIRECT_HELPER_SHA256,
                      base.ten.__file__:base.TEN_HELPER_SHA256}
    for path, expected in expected_modules.items():
        if file_sha(Path(path)) != expected:
            raise ValueError(f"pinned forecast helper hash changed: {Path(path).name}")
    baseline = _json(BASELINE_PATH)
    rows = _baseline_index(baseline, phase)
    radish = [r for k,r in rows.items() if k[0] == "radish"]
    eligible = [r for r in radish if dt.date.fromisoformat(r["origin"]) >= PHASES[phase][2]]
    if len(eligible) not in (755, 790):
        raise ValueError(f"unexpected radish eligible key count: {len(eligible)}")
    return {"phase": phase, "baselineRows": len(rows), "radishEligibleRows": len(eligible),
            "radishIssued": sum(r["models"]["causal50"]["status"] == "issued" for r in eligible),
            "baselineManifestSha256": file_sha(BASELINE_PATH)}


def _status_row(base_row: Mapping[str, Any], fit_id: str | None = None) -> dict[str, Any]:
    direct = base_row.get("models", {}).get("causalDirect", {})
    sarima = base_row.get("models", {}).get("causalSarima", {})
    blend = base_row.get("models", {}).get("causal50", {})
    p0 = None
    feature_audit = direct.get("provenance", {}).get("featureAudit", {})
    # E manifest stores p0 inside causal direct provenance's feature source; reconstruct from source feature logic later.
    return {"status": "issued" if direct.get("status") == "issued" else "not_issued",
            "directLogPrediction": None, "directPrediction": direct.get("prediction"),
            "blendPrediction": blend.get("prediction"), "fallbackReason": blend.get("provenance", {}).get("fallbackReason"),
            "fitId": fit_id, "cutoff": feature_audit.get("cutoff"), "p0": p0,
            "p0Date": feature_audit.get("p0Date"), "sarimaPrediction": sarima.get("prediction")}


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    import io
    buffer = io.StringIO(newline="")
    fields = ["product", "origin", "target", "originDate", "targetDate", "horizon", "eligible", "p0", "p0Date", "fitId", "condition", "status", "directLogPrediction", "directPrediction", "sarimaPrediction", "blendPrediction", "fallbackReason", "cutoff"]
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        for name in CONDITIONS:
            item = row["conditions"][name]
            writer.writerow({"product": row["product"], "origin": row.get("origin", row["originDate"]), "target": row.get("target", row["targetDate"]), "originDate": row["originDate"], "targetDate": row["targetDate"], "horizon": row["horizon"],
                             "eligible": row["eligible"], "p0": row.get("p0"), "p0Date": row.get("p0Date"), "condition": name, **item})
    return buffer.getvalue().encode()


def _actuals_by_product_target(prices: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    from . import direct_horizon
    obs = direct_horizon._observations(dict(prices))
    return {(p, d.isoformat()): float(v[0]) for p, values in obs.items() for d, v in values.items() if p in ("cabbage", "radish", "onion", "garlic") and math.isfinite(float(v[0])) and float(v[0]) > 0}


def _gate_evidence(report: Mapping[str, Any], manifest: Mapping[str, Any], baseline: Mapping[Any, Any]) -> dict[str, Any]:
    per_h = report.get("radishPrimary", {}).get("perHorizon", [])
    hmap = {row.get("horizon"): row for row in per_h}
    cells = [hmap.get(h) for h in HORIZONS]
    complete = all(c is not None and c.get("commonPairs", 0) > 0 and
                   c.get("CBlendMape") is not None and c.get("BBlendMape") is not None and
                   c.get("CBlendMae") is not None and c.get("BBlendMae") is not None for c in cells)
    c_mape = math.fsum(c["CBlendMape"] for c in cells) / 10 if complete else None
    b_mape = math.fsum(c["BBlendMape"] for c in cells) / 10 if complete else None
    c_mae = math.fsum(c["CBlendMae"] for c in cells) / 10 if complete else None
    b_mae = math.fsum(c["BBlendMae"] for c in cells) / 10 if complete else None
    rows = [r for r in manifest.get("forecasts", []) if r.get("product") == "radish" and r.get("eligible")]
    by_key = {(r["product"], r["origin"], r["target"], r["horizon"]): r for r in rows}
    baseline_radish = {k: v for k, v in baseline.items() if k[0] == "radish" and
                       dt.date.fromisoformat(k[1]) >= (PHASES[manifest["phase"]][2])}
    key_coverage_complete = set(by_key) == set(baseline_radish)
    extra_fallback = extra_withheld = 0
    changed = False
    for key in set(by_key) & set(baseline_radish):
        candidate = by_key[key]["conditions"]["C"]
        old = baseline_radish[key]
        old_fallback = bool(old.get("fallback", False))
        old_status = old.get("status")
        if candidate.get("status") == "fallback" and not old_fallback: extra_fallback += 1
        if candidate.get("status") not in ("issued", "fallback") and old_status in ("issued", "fallback"):
            extra_withheld += 1
        b_prediction = by_key[key]["conditions"]["B"].get("directLogPrediction")
        c_prediction = candidate.get("directLogPrediction")
        if b_prediction is not None and c_prediction is not None and float(b_prediction) != float(c_prediction): changed = True
    coverage_ok = key_coverage_complete and extra_fallback == 0 and extra_withheld == 0
    passed = bool(complete and coverage_ok and changed and c_mape < b_mape and c_mae <= b_mae)
    return {"allTenHorizonsSupported": complete, "candidateEqualHorizonBlendMape": c_mape,
            "baselineEqualHorizonBlendMape": b_mape, "candidateEqualHorizonBlendMae": c_mae,
            "baselineEqualHorizonBlendMae": b_mae, "strictMapeImprovement": bool(complete and c_mape < b_mape),
            "maeNotWorse": bool(complete and c_mae <= b_mae), "keyCoverageComplete": key_coverage_complete,
            "additionalFallbackKeys": extra_fallback, "additionalWithheldKeys": extra_withheld,
            "coverageNoLoss": coverage_ok, "candidateDiffersFromB": changed, "passed": passed,
            "scope": "unrounded radish C-vs-B same-SARIMA blend; ten horizons equally weighted; V is retrospective"}


def _test_gate_passed(output_dir: Path = OUT) -> bool:
    report_path = output_dir / "validation-report.json"
    manifest_path = output_dir / "validation-manifest.json"
    if not report_path.exists() or not manifest_path.exists():
        raise ValueError("V evidence is incomplete; T cannot be classified as a performance-gate failure")
    verification=verify_phase("V",output_dir)
    report = _json(report_path)
    recomputed=_compute_score_report("V",output_dir,verification)
    if canonical(report)!=canonical(recomputed):
        raise ValueError("V report does not replay from the verified manifest, baseline and pinned actual prices")
    return bool(recomputed["primaryGateEvidence"]["passed"])


def _four_crop_auxiliary(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics=report.get("allFourCommonMetrics",[])
    lookup={(r.get("product"),r.get("horizon"),r.get("condition")):r for r in metrics}
    cells=[lookup.get((p,h,c)) for p in ("cabbage","radish","onion","garlic") for h in HORIZONS for c in CONDITIONS]
    complete=all(row is not None and row.get("n",0)>0 and row.get("blendMape") is not None and row.get("blendMae") is not None for row in cells)
    if not complete:
        return {"nCellsExpected":40,"nCellsComplete":sum(row is not None and row.get("n",0)>0 for row in cells),
                "equalCellMacroByCondition":None,"candidateMinusBaseline":None}
    by_condition={c:{"blendMape":math.fsum(lookup[(p,h,c)]["blendMape"] for p in ("cabbage","radish","onion","garlic") for h in HORIZONS)/40,
                     "blendMae":math.fsum(lookup[(p,h,c)]["blendMae"] for p in ("cabbage","radish","onion","garlic") for h in HORIZONS)/40}
                  for c in CONDITIONS}
    return {"nCellsExpected":40,"nCellsComplete":40,"equalCellMacroByCondition":by_condition,
            "candidateMinusBaseline":{"blendMape":by_condition["C"]["blendMape"]-by_condition["B"]["blendMape"],
                                       "blendMae":by_condition["C"]["blendMae"]-by_condition["B"]["blendMae"]},
            "scope":"fixed 4 crop x 10 horizon equal cell weighting; auxiliary only"}


def build_phase(phase: str, output_dir: Path = OUT) -> dict[str, Any]:
    """Build and lock answer-free forecast grid; requires protocol and preflight."""
    from . import equal_info_online_forecasts as base
    build_started=time.perf_counter()
    if phase not in PHASES: raise ValueError("phase must be V or T")
    protocol = _load_protocol(output_dir)
    if phase == "T" and not _test_gate_passed(output_dir):
        _claim_run(output_dir,phase,protocol["protocolSha256"])
        marker = {"schemaVersion": 1, "phase": "T", "status": "not_run_validation_failed",
                  "reason": "V primary gate did not pass", "protocolSha256": protocol["protocolSha256"]}
        _write_exclusive(output_dir / "test-manifest.json", canonical(marker))
        _write_exclusive(output_dir / "test-report.json", canonical({**marker, "fits": 0, "forecastRows": 0, "scoreStatus": "not_run"}))
        _terminal_run(output_dir,phase,"not_run_validation_failed",manifest_sha=file_sha(output_dir/"test-manifest.json"),reason=marker["reason"])
        return marker
    claim=_claim_run(output_dir,phase,protocol["protocolSha256"])
    manifest_path_existing=output_dir/("validation-manifest.json" if phase=="V" else "test-manifest.json")
    terminal_path=output_dir/("validation-terminal.json" if phase=="V" else "test-terminal.json")
    if manifest_path_existing.exists():
        verify_phase(phase,output_dir,allow_missing_terminal=not terminal_path.exists())
        if not terminal_path.exists():
            _terminal_run(output_dir,phase,"complete",manifest_sha=file_sha(manifest_path_existing))
        return _json(manifest_path_existing)
    if terminal_path.exists(): raise ValueError("phase run already terminated")
    preflight = _preflight(phase, output_dir)
    baseline = _json(BASELINE_PATH)
    source_rows = _baseline_index(baseline, phase)
    composition_rows = _composition_index(phase)
    start, end, origin_start = PHASES[phase]
    # Keep baseline forecast keys/status/SARIMA exact; later fit only the radish direct rows.
    # The forecast loop intentionally uses the same source ordering and cutoffs as E106.
    from . import full_history_return_eval as narrow
    from . import full_history_xgb_rf_eval as rich
    raw_prices = _json(PRICES_PATH)
    records = rich._raw_maps(raw_prices)[0]
    direct_series = base.direct_horizon._observations(raw_prices)
    all_keys = sorted(source_rows, key=lambda k:(k[0],k[2],k[3]))
    # Construct each monthly radish fit once. Exact feature construction is delegated to the frozen E106 helper.
    radish_forecasts: dict[tuple[str,str,str,int], dict[str,Any]] = {}
    fit_docs=[]; fit_ids={}
    radish_keys=[k for k in all_keys if k[0]=="radish" and dt.date.fromisoformat(k[1])>=origin_start]
    fit_groups={}
    fit_origin_by_key={}
    for key in radish_keys:
        h=key[3]
        training=source_rows[key]["models"]["causalDirect"].get("training") or {}
        if not training.get("fitOrigin"):
            raise ValueError(f"eligible E106 key lacks actual monthly fit origin: {key}")
        fit_origin=dt.date.fromisoformat(training["fitOrigin"])
        origin=dt.date.fromisoformat(key[1])
        if (fit_origin.year,fit_origin.month)!=(origin.year,origin.month) or fit_origin>origin:
            raise ValueError(f"E106 fit origin is not the eligible month origin: {key}")
        fit_origin_by_key[key]=fit_origin
        fit_groups.setdefault(fit_origin, set()).add(h)
    for fit_origin in sorted(fit_groups):
        fold_started=time.perf_counter()
        # The task requires all h to share actual monthly fit origin.
        if fit_groups[fit_origin] != set(HORIZONS): raise ValueError("mismatched horizon monthly fit schedule")
        fit_id=f"radish-{fit_origin.isoformat()}"
        Xh=[];Xraw_by_h=[];yh=[];scales=[];standardizer_means=[];standardizer_scales=[];standardizer_means_full=[];train_rows=[];targets_by_h=[]
        cache={}
        for h in HORIZONS:
            series=direct_series["radish"]
            X=[];y=[];targets=[]
            for target,(actual,_) in sorted(series.items()):
                if target < narrow.TRAIN_START or target >= fit_origin: continue
                found=base.modeling._feature_row(series,target-dt.timedelta(days=h+1),target,h,cache)
                if found is None: continue
                source,vector=found
                if (target-dt.timedelta(days=h)-source).days >14: continue
                X.append(vector); y.append(math.log(actual/series[source][0])); targets.append(target)
            if len(X)<90: raise ValueError(f"{fit_id} h={h} has fewer than 90 baseline training rows")
            from sklearn.preprocessing import StandardScaler
            scaler=StandardScaler().fit(np.asarray(X,dtype=float))
            Xh.append(scaler.transform(np.asarray(X,dtype=float)))
            Xraw_by_h.append(np.asarray(X,dtype=float))
            yh.append(np.asarray(y,dtype=float));scales.append(scaler.scale_.tolist());standardizer_means.append(scaler.mean_[list(ACTIVE)].tolist());standardizer_scales.append(scaler.scale_[list(ACTIVE)].tolist());standardizer_means_full.append(scaler.mean_.tolist());targets_by_h.append(targets)
            baseline_audits=[source_rows[k]["models"]["causalDirect"].get("training") or {}
                             for k in radish_keys if k[3]==h and
                             (source_rows[k]["models"]["causalDirect"].get("training") or {}).get("fitOrigin")==fit_origin.isoformat()]
            if not baseline_audits:
                raise ValueError(f"no saved E106 training audit for {fit_id} h={h}")
            audit0=baseline_audits[0]
            expected_first=targets[0].isoformat()
            expected_last=targets[-1].isoformat()
            if (int(audit0.get("count", -1)) != len(X) or audit0.get("firstTarget") != expected_first or
                    audit0.get("lastTarget") != expected_last or float(audit0.get("alpha", -1)) != ALPHA or
                    int(audit0.get("featureCount", -1)) != 40):
                raise ValueError(f"training support/scaler/alpha lineage differs from E106 at {fit_id} h={h}")
        # B/D/C/P fit once over ten horizon datasets.
        input_hashes={"trainingKeys":sha256_bytes(canonical([[d.isoformat() for d in ts] for ts in targets_by_h])),
                      "trainingMatrix":sha256_bytes(b"".join(np.asarray(x,dtype="<f8").tobytes(order="C") for x in Xh)),
                      "rawTrainingMatrix":sha256_bytes(b"".join(np.asarray(x,dtype="<f8").tobytes(order="C") for x in Xraw_by_h)),
                      "trainingTarget":sha256_bytes(b"".join(np.asarray(y,dtype="<f8").tobytes(order="C") for y in yh)),
                      "trainingCounts":sha256_bytes(canonical([len(x) for x in yh])),
                      "scalerState":sha256_bytes(canonical({"mean":standardizer_means_full,"scale":scales})),
                      "protocolSha256":protocol["protocolSha256"],
                      "protocolInputsSha256":sha256_bytes(canonical(protocol["protocol"]["inputSha256"])),
                      "runtimeSha256":sha256_bytes(canonical(protocol["protocol"]["runtime"]))}
        def make_fit():
            try:
                result=fit_joint(Xh,yh,scales,fit_id=fit_id)
            except (ArithmeticError, ValueError, np.linalg.LinAlgError) as exc:
                result={"status":"numerical_failure",
                        "reason":f"fit_joint raised {type(exc).__name__}: {exc}",
                        "fitId":fit_id}
            result["standardScalerMean"] = standardizer_means
            result["standardScalerScale"] = standardizer_scales
            result["standardScalerMeanFull"] = standardizer_means_full
            result["standardScalerScaleFull"] = scales
            result["inputHashes"] = input_hashes
            result["protocolSha256"] = protocol["protocolSha256"]
            result["runtime"] = protocol["protocol"]["runtime"]
            if result["status"] != "ok":
                return result
            beta={name:np.asarray(result["fitCoefficients"][name],dtype=float) for name in CONDITIONS}
            for hi,h in enumerate(HORIZONS):
                for j,target in enumerate(targets_by_h[hi]):
                    x=np.asarray(Xh[hi][j],dtype=float)[list(ACTIVE)]-np.asarray(result["scalerXMean"][hi])
                    row={"fitId":fit_id,"horizon":h,"fitOrigin":fit_origin.isoformat(),"targetDate":target.isoformat(),"actualLogReturn":float(yh[hi][j])}
                    for name in CONDITIONS: row[name]=float(x@beta[name][hi]+result["scalerYMean"][hi])
                    train_rows.append(row)
            result["trainingRows"]=train_rows
            result["trainingFirstTargetByHorizon"]=[ts[0].isoformat() for ts in targets_by_h]
            result["trainingLastTargetByHorizon"]=[ts[-1].isoformat() for ts in targets_by_h]
            result["fitAndDiagnosticElapsedSeconds"]=time.perf_counter()-fold_started
            return result
        fit=_fit_checkpoint(output_dir,phase,fit_id,input_hashes,make_fit)
        if fit.get("status") != "ok":
            failure={"schemaVersion":1,"phase":phase,"fitId":fit_id,"status":fit.get("status"),
                     "reason":fit.get("reason"),"edf":fit.get("edf"),"protocolSha256":protocol["protocolSha256"],
                     "inputHashes":input_hashes}
            failure_path=output_dir/("validation-fit-failure.json" if phase=="V" else "test-fit-failure.json")
            _write_exclusive(failure_path,canonical(failure))
            _terminal_run(output_dir,phase,str(fit.get("status")),artifact_sha=file_sha(failure_path),reason=fit.get("reason"))
            raise ArithmeticError(f"{fit_id}: {fit.get('status')} (failure terminal saved)")
        fit_ids[fit_id]=fit
        fit_docs.append(fit)
        # Prediction keys for this fit origin, with source E106 SARIMA forecasts reused exactly.
        for key in radish_keys:
            if fit_origin_by_key.get(key) != fit_origin: continue
            baseline_row=source_rows[key]
            # Existing fit audit must identify same fit origin, including all ten h.
            direct=baseline_row["models"]["causalDirect"]
            if (direct.get("training") or {}).get("fitOrigin") != fit_origin.isoformat():
                continue
            h=key[3]
            # Resolve exact E106 feature vector and p0 from the frozen helper.
            target=dt.date.fromisoformat(key[2]);origin=dt.date.fromisoformat(key[1])
            feat,p0,audit=base._direct_feature(direct_series["radish"],target,h,{})
            if feat is None or p0 is None: raise ValueError(f"E106 feature unavailable for {key}")
            Xstd=np.asarray([feat],dtype=float)
            scaler_mean=np.asarray(fit["scalerXMean"][h-1],dtype=float)
            # Reconstruct scaler transform from saved baseline fit inputs above is represented by scaler means/scale.
            # The joint fit's stored x means are on standardized rows; use the fold scaler statistics below.
            # store scaler values in fit metadata while training matrix was constructed.
            raw_scale=np.asarray(fit["standardScalerScale"][h-1],dtype=float)
            raw_mean=np.asarray(fit["standardScalerMean"][h-1],dtype=float)
            z=(Xstd[:,list(ACTIVE)]-raw_mean)/raw_scale
            pred={}
            for name in CONDITIONS:
                coef=np.asarray(fit["fitCoefficients"][name][h-1],dtype=float)
                pred[name]=float(((z-fit["scalerXMean"][h-1])@coef+fit["scalerYMean"][h-1]).item())
            sarima=baseline_row["models"]["causalSarima"]
            s_pred=sarima.get("prediction") if sarima.get("status")=="issued" else None
            direct_price={n:float(p0*math.exp(pred[n])) for n in CONDITIONS}
            conditions={}
            for name in CONDITIONS:
                status="issued" if s_pred is not None else "fallback"
                blend=.5*direct_price[name]+.5*float(s_pred) if s_pred is not None else direct_price[name]
                conditions[name]={"status":status,"directLogPrediction":pred[name],"directPrediction":direct_price[name],"blendPrediction":blend,
                                  "fallbackReason":None if s_pred is not None else sarima.get("reason"),"fitId":fit_id,"cutoff":audit.get("cutoff"),"p0":p0,
                                  "p0Date":audit.get("p0Date"),"sarimaPrediction":s_pred}
            radish_forecasts[key]={"product":"radish","origin":key[1],"target":key[2],"originDate":key[1],"targetDate":key[2],"horizon":h,"eligible":True,"p0":p0,"p0Date":audit.get("p0Date"),"featureVector":feat,"conditions":conditions}
    forecasts=[]
    if len(radish_forecasts) != len(radish_keys):
        raise ValueError(f"radish forecast coverage mismatch: expected {len(radish_keys)}, got {len(radish_forecasts)}")
    for key, base_row in source_rows.items():
        if key[0]=="radish" and key in radish_forecasts:
            forecasts.append(radish_forecasts[key]);continue
        conditions={}
        target=dt.date.fromisoformat(key[2]); h=key[3]
        feat,p0,audit=base._direct_feature(direct_series.get(key[0],{}),target,h,{})
        for name in CONDITIONS:
            model=base_row["models"]["causal50"] if key[0]=="radish" or key not in composition_rows else composition_rows[key]["candidate"]
            d=base_row["models"]["causalDirect"]
            s=base_row["models"]["causalSarima"]
            dprice=d.get("prediction")
            dlog=math.log(float(dprice)/float(p0)) if dprice and p0 else None
            fallback=bool(model.get("provenance",{}).get("fallback"))
            if key[0] != "radish" and key in composition_rows and composition_rows[key].get("candidateSource") == "E":
                fallback = fallback or bool(base_row["models"]["causal50"].get("provenance",{}).get("fallback"))
            conditions[name]={"status":("fallback" if fallback else "issued") if model.get("status")=="issued" else "not_issued",
                              "directLogPrediction":dlog,"directPrediction":dprice,"blendPrediction":model.get("prediction"),
                              "fallbackReason":model.get("provenance",{}).get("fallbackReason") or base_row["models"]["causal50"].get("provenance",{}).get("fallbackReason"),"fitId":None,
                              "cutoff":d.get("provenance",{}).get("featureAudit",{}).get("cutoff"),"p0":p0,
                              "p0Date":audit.get("p0Date") if audit else None,"sarimaPrediction":s.get("prediction")}
        forecasts.append({"product":key[0],"origin":key[1],"target":key[2],"originDate":key[1],"targetDate":key[2],"horizon":key[3],"eligible":dt.date.fromisoformat(key[1])>=origin_start,"p0":p0,"p0Date":audit.get("p0Date") if audit else None,"conditions":conditions})
    forecasts.sort(key=lambda r:(r["product"],r["targetDate"],r["horizon"]))
    reproduction=[]
    for row in forecasts:
        if row["product"] != "radish" or not row["eligible"]: continue
        key=(row["product"],row["origin"],row["target"],row["horizon"])
        old=source_rows[key]
        for component, old_model, candidate_field in (("direct", "causalDirect", "directPrediction"),
                                                       ("blend", "causal50", "blendPrediction")):
            existing=old["models"][old_model].get("prediction")
            reproduced=row["conditions"]["B"].get(candidate_field)
            if existing is None or reproduced is None:
                if existing is not reproduced:
                    raise ValueError(f"B {component} status mismatch for {key}")
                delta=None
            else:
                delta=abs(float(reproduced)-float(existing))
                if delta > PRICE_ABS_TOL + PRICE_REL_TOL*abs(float(existing)):
                    raise ValueError(f"B {component} reproduction exceeds tolerance for {key}")
            reproduction.append({"key":list(key),"component":component,"existing":existing,
                                 "reproduced":reproduced,"absoluteDifference":delta})
    baseline_reproduction={"phase":phase,"status":"passed","priceTolerance":{"absolute":PRICE_ABS_TOL,"relative":PRICE_REL_TOL},
                           "checkedValues":len(reproduction),"maximumAbsoluteDifference":max((r["absoluteDifference"] or 0.0 for r in reproduction),default=0.0),
                           "rows":reproduction}
    _write_exclusive(output_dir/("baseline-reproduction.json" if phase=="V" else "test-baseline-reproduction.json"),canonical(baseline_reproduction))
    payload=_csv_bytes(forecasts)
    csv_path=output_dir/("validation-predictions.csv" if phase=="V" else "test-predictions.csv")
    _write_exclusive(csv_path,payload)
    score_fits=[]
    for fit in fit_docs:
        for h in HORIZONS:
            train=[r for r in fit["trainingRows"] if r["horizon"]==h]
            score_fits.append({"fitId":f"{fit['fitId']}-h{h:02d}","jointFitId":fit["fitId"],"horizon":h,
                               "fitOrigin":fit["fitId"].removeprefix("radish-"),"trainingCount":len(train),
                               "trainingRows":[{"targetDate":r["targetDate"],"actualLogReturn":r["actualLogReturn"],**{c:r[c] for c in CONDITIONS}} for r in train]})
    manifest={"schemaVersion":1,"phase":phase,"protocolSha256":protocol["protocolSha256"],"inputSha256":protocol["protocol"]["inputSha256"],
              "forecastFile":csv_path.name,"forecastFileSha256":sha256_bytes(payload),"fitDiagnosticsSha256":sha256_bytes(canonical(score_fits)),
              "fitDiagnostics":score_fits,"jointFitDiagnostics":fit_docs,"baselineReproduction":baseline_reproduction,
              "modelFitCounts":{"jointMonthlySystems":len(fit_docs),"baselineHorizonCoefficientSolutions":len(fit_docs)*10,
                                "DCPSharedConditionSystems":len(fit_docs)*3,"sklearnRidgeEstimatorFitCalls":0,
                                "StandardScalerFits":len(fit_docs)*10},
              "preflight":preflight,"forecasts":forecasts,"buildElapsedSeconds":time.perf_counter()-build_started,
              "createdAtKst":dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()}
    manifest_path=output_dir/("validation-manifest.json" if phase=="V" else "test-manifest.json")
    _write_exclusive(manifest_path,canonical(manifest))
    _write_exclusive(output_dir/("fit-diagnostics.json" if phase=="V" else "test-fit-diagnostics.json"),canonical({"jointFits":fit_docs,"scoringFits":score_fits}))
    _terminal_run(output_dir,phase,"complete",manifest_sha=file_sha(manifest_path))
    return manifest


def _reconstruct_training_fold(fit_origin: dt.date, prices: Mapping[str, Any], base: Any):
    from . import full_history_return_eval as narrow
    direct_series=base.direct_horizon._observations(dict(prices))
    series=direct_series["radish"]
    Xraw_by_h=[]; y_by_h=[]; targets_by_h=[]; cache={}
    for h in HORIZONS:
        X=[];y=[];targets=[]
        for target,(actual,_) in sorted(series.items()):
            if target<narrow.TRAIN_START or target>=fit_origin: continue
            found=base.modeling._feature_row(series,target-dt.timedelta(days=h+1),target,h,cache)
            if found is None: continue
            source,vector=found
            if (target-dt.timedelta(days=h)-source).days>14: continue
            X.append(vector);y.append(math.log(actual/series[source][0]));targets.append(target)
        if len(X)<90: raise ValueError(f"reconstructed fold {fit_origin} h={h} has fewer than 90 rows")
        Xraw_by_h.append(np.asarray(X,dtype=np.float64));y_by_h.append(np.asarray(y,dtype=np.float64));targets_by_h.append(targets)
    return direct_series,Xraw_by_h,y_by_h,targets_by_h


def _verify_fit_state(fit: Mapping[str, Any], phase: str, protocol: Mapping[str, Any],
                      Xraw_by_h: Sequence[np.ndarray], y_by_h: Sequence[np.ndarray],
                      targets_by_h: Sequence[Sequence[dt.date]]) -> list[np.ndarray]:
    if fit.get("status")!="ok" or fit.get("phase") not in (None,phase):
        raise ValueError("invalid saved monthly fit state")
    if fit.get("protocolSha256")!=protocol["protocolSha256"] or fit.get("runtime")!=protocol["protocol"]["runtime"]:
        raise ValueError("fit state is not bound to the frozen protocol/runtime")
    means=_finite_array(fit.get("standardScalerMeanFull"),ndim=2,name="saved StandardScaler mean")
    scales=_finite_array(fit.get("standardScalerScaleFull"),ndim=2,name="saved StandardScaler scale")
    if means.shape!=(10,40) or scales.shape!=(10,40) or np.any(scales<=0):
        raise ValueError("saved StandardScaler state must be finite 10x40")
    if (not np.allclose(fit.get("standardScalerMean"),means[:,list(ACTIVE)],rtol=0,atol=0) or
            not np.allclose(fit.get("standardScalerScale"),scales[:,list(ACTIVE)],rtol=0,atol=0)):
        raise ValueError("active-feature scaler arrays differ from the full saved scaler state")
    Xstd=[(x-means[i])/scales[i] for i,x in enumerate(Xraw_by_h)]
    input_hashes=fit.get("inputHashes")
    if not isinstance(input_hashes,dict): raise ValueError("fit state lacks immutable input hashes")
    expected={"trainingKeys":sha256_bytes(canonical([[d.isoformat() for d in ts] for ts in targets_by_h])),
              "rawTrainingMatrix":sha256_bytes(b"".join(np.asarray(x,dtype="<f8").tobytes(order="C") for x in Xraw_by_h)),
              "trainingMatrix":sha256_bytes(b"".join(np.asarray(x,dtype="<f8").tobytes(order="C") for x in Xstd)),
              "trainingTarget":sha256_bytes(b"".join(np.asarray(y,dtype="<f8").tobytes(order="C") for y in y_by_h)),
              "trainingCounts":sha256_bytes(canonical([len(y) for y in y_by_h])),
              "scalerState":sha256_bytes(canonical({"mean":means.tolist(),"scale":scales.tolist()})),
              "protocolSha256":protocol["protocolSha256"],
              "protocolInputsSha256":sha256_bytes(canonical(protocol["protocol"]["inputSha256"])),
              "runtimeSha256":sha256_bytes(canonical(protocol["protocol"]["runtime"]))}
    if input_hashes!=expected:
        raise ValueError(f"reconstructed monthly fit inputs differ: {fit.get('fitId')}")
    if fit.get("trainingCounts")!=[len(y) for y in y_by_h]:
        raise ValueError("saved monthly training row counts differ from reconstructed support")
    return Xstd


def _verify_fit_numerics(fit: Mapping[str, Any], Xstd_by_h: Sequence[np.ndarray],
                         y_by_h: Sequence[np.ndarray], targets_by_h: Sequence[Sequence[dt.date]]) -> None:
    xs=[];ys=[];means=[];ymeans=[]
    for xfull,y in zip(Xstd_by_h,y_by_h):
        x=xfull[:,list(ACTIVE)]; xm=x.mean(axis=0); ym=float(y.mean())
        xs.append(x-xm);ys.append(y-ym);means.append(xm);ymeans.append(ym)
    if not np.allclose(np.asarray(fit["scalerXMean"]),means,rtol=0,atol=2e-13) or not np.allclose(fit["scalerYMean"],ymeans,rtol=0,atol=2e-13):
        raise ValueError("saved joint centering values do not match reconstructed training data")
    g=np.zeros((N_COEF,N_COEF));rhs=np.zeros(N_COEF)
    for h,(x,y) in enumerate(zip(xs,ys)):
        sl=slice(h*N_FEATURE,(h+1)*N_FEATURE);g[sl,sl]=x.T@x;rhs[sl]=x.T@y
    q0=g+ALPHA*np.eye(N_COEF); scales=np.asarray(fit["scalerScale"],dtype=float); penalties=graph_penalties(scales)
    df0=_edf(g,q0);dfinf=_shared_limit(g,q0,scales);edf=fit["edf"]
    bound_tol=1e-10*max(1.0,abs(df0),abs(dfinf))
    if dfinf < -bound_tol or dfinf > df0+bound_tol:
        raise ValueError("reconstructed shared EDF limit is outside [0,df0]")
    target=dfinf+EDF_FRACTION*(df0-dfinf)
    psd={name:_check_psd(penalties[name],name=f"reconstructed P_{name}") for name in ("D","C","P")}
    for name,record in psd.items():
        saved=fit.get("graph",{}).get("penaltyChecks",{}).get(name,{})
        if abs(saved.get("minimumEigenvalue",float("nan"))-record["minimumEigenvalue"])>1e-8*max(1.0,abs(record["minimumEigenvalue"])):
            raise ValueError("saved penalty PSD diagnostic differs from reconstruction")
    for field,actual in (("df0",df0),("dfInf",dfinf),("dfTarget",target)):
        if abs(float(edf.get(field))-actual)>1e-8*max(1.0,abs(actual)):
            raise ValueError(f"saved {field} differs from reconstructed joint design")
    by_h=edf.get("actualByHorizon",{})
    for condition in CONDITIONS:
        beta=_finite_array(fit["fitCoefficients"][condition],ndim=2,name=f"beta {condition}")
        if beta.shape!=(10,39): raise ValueError("saved condition coefficients have the wrong shape")
        flat=beta.reshape(-1);lam=0.0 if condition=="B" else float(fit["lambdas"][condition])
        if condition=="B" and lam!=0.0 or condition!="B" and (not math.isfinite(lam) or lam<=0):
            raise ValueError("saved lambda violates the frozen intervention rule")
        p=np.zeros_like(q0) if condition=="B" else penalties[condition]
        q=q0+lam*p
        residual=q@flat-rhs;den=float(np.linalg.norm(q,2)*np.linalg.norm(flat)+np.linalg.norm(rhs))
        rel=float(np.linalg.norm(residual)/den) if den else float(np.linalg.norm(residual))
        if rel>SOLVE_REL_TOL or abs(rel-fit["diagnostics"][condition]["relativeSolveResidual"])>1e-12:
            raise ValueError(f"saved {condition} coefficient state fails normal-equation replay")
        total=_edf(g,q)
        qinv_g=np.linalg.solve(q,g)
        parts=[float(np.trace(qinv_g[h*39:(h+1)*39,h*39:(h+1)*39])) for h in range(10)]
        saved=by_h.get(condition)
        if not isinstance(saved,list) or len(saved)!=10 or not np.allclose(saved,parts,rtol=1e-9,atol=1e-9):
            raise ValueError(f"saved {condition} per-horizon EDF differs from source reconstruction")
        if abs(math.fsum(parts)-total)>1e-8*max(1.0,total): raise ValueError("per-horizon EDF does not sum to total EDF")
        stored_total=float(fit["diagnostics"][condition]["actualEdf"])
        if abs(total-stored_total)>1e-8*max(1.0,total): raise ValueError(f"saved {condition} total EDF mismatch")
        if condition!="B":
            audit=edf.get("conditions",{}).get(condition,{})
            if (audit.get("status")!="ok" or abs(float(audit.get("lambda"))-lam)>1e-14*max(1.0,lam) or
                    abs(float(audit.get("actualDf"))-total)>1e-8*max(1.0,total)):
                raise ValueError(f"saved {condition} EDF root audit is inconsistent")
        if condition!="B" and (abs(total-target)>float(edf["tolerance"]) or total>=df0):
            raise ValueError(f"saved {condition} EDF misses the frozen positive-intervention target")
        diag=fit["diagnostics"][condition]
        extra=0.0 if condition=="B" else float(flat@penalties[condition]@flat)
        base_pen=float(ALPHA*flat@flat)
        sse=math.fsum(float(np.sum((y-xs[h]@beta[h])**2)) for h,y in enumerate(ys))
        f0=sse+base_pen;objective=f0+lam*extra
        train_mse=[float(np.mean((y-xs[h]@beta[h])**2)) for h,y in enumerate(ys)]
        for field,value in (("basePenalty",base_pen),("extraPenalty",extra),("F0",f0),("objective",objective)):
            if abs(float(diag.get(field))-value)>1e-9*max(1.0,abs(value)):
                raise ValueError(f"saved {condition} objective diagnostic {field} mismatch")
        if not np.allclose(diag.get("trainLogMseByHorizon"),train_mse,rtol=1e-9,atol=1e-11):
            raise ValueError(f"saved {condition} training loss diagnostics mismatch")
        predictions=fit["trainingPredictions"][condition]
        for h,(x,y) in enumerate(zip(xs,ys)):
            pred=x@beta[h]+ymeans[h]
            if not np.allclose(predictions[h],pred,rtol=0,atol=1e-11):
                raise ValueError(f"saved {condition} training predictions failed reconstruction")


def verify_phase(phase: str, output_dir: Path = OUT, *, allow_missing_terminal: bool = False) -> dict[str, Any]:
    """Fit-free integrity, coefficient replay, grid, and baseline verification."""
    if phase not in PHASES: raise ValueError("phase must be V or T")
    protocol = _load_protocol(output_dir)
    _preflight(phase,output_dir)
    if phase == "V":
        manifest_path = output_dir / "validation-manifest.json"
        csv_path = output_dir / "validation-predictions.csv"
        diag_path = output_dir / "fit-diagnostics.json"
        baseline_path = output_dir / "baseline-reproduction.json"
    else:
        manifest_path = output_dir / "test-manifest.json"
        csv_path = output_dir / "test-predictions.csv"
        diag_path = output_dir / "test-fit-diagnostics.json"
        baseline_path = output_dir / "test-baseline-reproduction.json"
    manifest = _json(manifest_path)
    if manifest.get("protocolSha256") != protocol["protocolSha256"]:
        raise ValueError("forecast manifest is not bound to frozen protocol")
    if manifest.get("phase") != phase:
        raise ValueError("forecast manifest phase mismatch")
    if manifest.get("inputSha256") != protocol["protocol"].get("inputSha256"):
        raise ValueError("forecast manifest input pins differ from frozen protocol")
    if file_sha(csv_path) != manifest.get("forecastFileSha256"):
        raise ValueError("forecast CSV digest mismatch")
    diagnostics = _json(diag_path)
    if sha256_bytes(canonical(manifest.get("fitDiagnostics"))) != manifest.get("fitDiagnosticsSha256"):
        raise ValueError("scorer training diagnostics digest mismatch")
    if canonical(diagnostics) != canonical({"jointFits": manifest.get("jointFitDiagnostics"),
                                           "scoringFits": manifest.get("fitDiagnostics")}):
        raise ValueError("persisted fit diagnostics differ from manifest")
    baseline_repro = _json(baseline_path)
    if baseline_repro != manifest.get("baselineReproduction") or baseline_repro.get("status") != "passed":
        raise ValueError("baseline reproduction artifact mismatch")
    start, end, origin_start = PHASES[phase]
    expected = {(p, (start + dt.timedelta(days=i) - dt.timedelta(days=h)).isoformat(),
                 (start + dt.timedelta(days=i)).isoformat(), h)
                for p in ("cabbage", "radish", "onion", "garlic")
                for i in range((end-start).days+1) for h in HORIZONS}
    rows = manifest.get("forecasts")
    keys = {(r.get("product"), r.get("origin"), r.get("target"), r.get("horizon")) for r in rows or []}
    if keys != expected or len(rows or []) != len(expected):
        raise ValueError("forecast manifest does not preserve complete phase key grid")
    if csv_path.read_bytes()!=_csv_bytes(rows):
        raise ValueError("forecast CSV rows do not exactly replay from the answer-free manifest")
    baseline = _baseline_index(_json(BASELINE_PATH), phase)
    composition=_composition_index(phase)
    from . import equal_info_online_forecasts as base
    raw_prices=_json(PRICES_PATH)
    direct_series=base.direct_horizon._observations(raw_prices)
    expected_fit_ids={f"radish-{y:04d}-{m:02d}-01" for y,m in ((dt.date(start.year+(start.month-1+i)//12,(start.month-1+i)%12+1,1).year,dt.date(start.year+(start.month-1+i)//12,(start.month-1+i)%12+1,1).month) for i in range(3))}
    joint_fits=manifest.get("jointFitDiagnostics",[])
    fit_map={fit.get("fitId"):fit for fit in joint_fits}
    if len(fit_map)!=3 or set(fit_map)!=expected_fit_ids:
        raise ValueError("monthly joint fit IDs do not match the three frozen phase months")
    expected_counts={"jointMonthlySystems":3,"baselineHorizonCoefficientSolutions":30,"DCPSharedConditionSystems":9,
                     "sklearnRidgeEstimatorFitCalls":0,"StandardScalerFits":30}
    if manifest.get("modelFitCounts")!=expected_counts:
        raise ValueError("fit counts differ from the frozen RHS budget")
    for fit_id,fit in fit_map.items():
        fit_origin=dt.date.fromisoformat(fit_id.removeprefix("radish-"))
        _,Xraw,y,targets=_reconstruct_training_fold(fit_origin,raw_prices,base)
        Xstd=_verify_fit_state(fit,phase,protocol,Xraw,y,targets)
        _verify_fit_numerics(fit,Xstd,y,targets)
        source_audits={key[3]:(r["models"]["causalDirect"].get("training") or {}) for key,r in baseline.items()
                       if key[0]=="radish" and
                       (r["models"]["causalDirect"].get("training") or {}).get("fitOrigin")==fit_origin.isoformat()}
        if set(source_audits)!=set(HORIZONS):
            raise ValueError("frozen E106 source lacks exactly ten horizon fit audits for a monthly RHS fit")
        for h in HORIZONS:
            audit=source_audits[h]
            if (int(audit.get("count",-1))!=len(targets[h-1]) or
                    audit.get("firstTarget")!=targets[h-1][0].isoformat() or
                    audit.get("lastTarget")!=targets[h-1][-1].isoformat() or
                    float(audit.get("alpha",-1))!=ALPHA or int(audit.get("featureCount",-1))!=40):
                raise ValueError(f"reconstructed RHS training support does not match E106 audit for {fit_id} h={h}")
        if fit.get("trainingFirstTargetByHorizon")!=[ts[0].isoformat() for ts in targets] or fit.get("trainingLastTargetByHorizon")!=[ts[-1].isoformat() for ts in targets]:
            raise ValueError("saved monthly first/last training keys differ from reconstructed support")
        saved_rows=fit.get("trainingRows",[])
        expected_n=sum(len(ts) for ts in targets)
        if len(saved_rows)!=expected_n:
            raise ValueError("saved monthly training loss rows do not cover reconstructed support")
        cursor=0
        for hi,h in enumerate(HORIZONS):
            beta={c:np.asarray(fit["fitCoefficients"][c][hi],dtype=float) for c in CONDITIONS}
            mean=np.asarray(fit["scalerXMean"][hi]);ym=float(fit["scalerYMean"][hi])
            for j,target in enumerate(targets[hi]):
                saved=saved_rows[cursor];cursor+=1
                if saved.get("horizon")!=h or saved.get("targetDate")!=target.isoformat() or abs(float(saved.get("actualLogReturn"))-float(y[hi][j]))>1e-13:
                    raise ValueError("saved training row key/actual differs from reconstructed support")
                x=Xstd[hi][j,list(ACTIVE)]-mean
                for c in CONDITIONS:
                    pred=float(x@beta[c]+ym)
                    if abs(float(saved[c])-pred)>1e-11:
                        raise ValueError("saved training fit loss prediction differs from coefficient replay")
        checkpoint_dir=output_dir/"fit-checkpoints";prefix=f"{phase.lower()}-{fit_id}"
        claim_path=checkpoint_dir/f"{prefix}.claim.json";checkpoint_path=checkpoint_dir/f"{prefix}.json";fit_terminal_path=checkpoint_dir/f"{prefix}.terminal.json"
        claim=_json(claim_path);checkpoint=_json(checkpoint_path);fit_terminal=_json(fit_terminal_path)
        input_sha=sha256_bytes(canonical(fit["inputHashes"]))
        if (checkpoint.get("fit")!=fit or checkpoint.get("checkpointSha256")!=sha256_bytes(canonical(fit)) or
                checkpoint.get("inputSha256")!=input_sha or claim.get("inputSha256")!=input_sha or claim.get("inputHashes")!=fit["inputHashes"] or
                fit_terminal.get("inputSha256")!=input_sha or fit_terminal.get("checkpointSha256")!=checkpoint.get("checkpointSha256") or
                fit_terminal.get("status")!="ok"):
            raise ValueError("monthly fit claim/checkpoint/terminal binding failed")
    expected_score_fits=[]
    for fit in joint_fits:
        for h in HORIZONS:
            train=[r for r in fit["trainingRows"] if r["horizon"]==h]
            expected_score_fits.append({"fitId":f"{fit['fitId']}-h{h:02d}","jointFitId":fit["fitId"],"horizon":h,
                "fitOrigin":fit["fitId"].removeprefix("radish-"),"trainingCount":len(train),
                "trainingRows":[{"targetDate":r["targetDate"],"actualLogReturn":r["actualLogReturn"],**{c:r[c] for c in CONDITIONS}} for r in train]})
    if manifest.get("fitDiagnostics")!=expected_score_fits or len(expected_score_fits)!=30:
        raise ValueError("per-horizon scorer training diagnostics do not replay from monthly fit state")
    terminal_path=output_dir/("validation-terminal.json" if phase=="V" else "test-terminal.json")
    run_claim_path=output_dir/("validation-claim.json" if phase=="V" else "test-claim.json")
    run_claim=_json(run_claim_path)
    if terminal_path.exists():
        run_terminal=_json(terminal_path)
        if (run_terminal.get("status")!="complete" or run_terminal.get("attemptId")!=run_claim.get("attemptId") or
                run_terminal.get("protocolSha256")!=protocol["protocolSha256"] or run_terminal.get("manifestSha256")!=file_sha(manifest_path)):
            raise ValueError("phase claim/terminal does not bind the completed forecast manifest")
    elif not allow_missing_terminal:
        raise ValueError("phase terminal is missing; recovery must first verify all saved artifacts")
    for row in rows:
        key=(row["product"],row["origin"],row["target"],row["horizon"])
        source=baseline.get(key)
        if source is None: raise ValueError(f"forecast key is absent from E106 baseline: {key}")
        target=dt.date.fromisoformat(row["target"]);h=row["horizon"]
        feature,p0,audit=base._direct_feature(direct_series.get(row["product"],{}),target,h,{})
        if row.get("p0")!=p0 or row.get("p0Date")!=(audit.get("p0Date") if audit else None):
            raise ValueError(f"saved p0 state differs from the frozen source at {key}")
        if row["product"]=="radish" and row["eligible"]:
            if row.get("featureVector")!=feature:
                raise ValueError(f"saved radish feature vector differs from source reconstruction at {key}")
            audit_training=source["models"]["causalDirect"].get("training") or {}
            fit_id=f"radish-{audit_training.get('fitOrigin')}"
            fit=fit_map.get(fit_id)
            if fit is None or dt.date.fromisoformat(fit["fitId"].removeprefix("radish-"))!=dt.date.fromisoformat(row["origin"]).replace(day=1):
                raise ValueError("radish forecast is linked to the wrong monthly fit")
            if any(row["conditions"][c].get("fitId")!=fit_id for c in CONDITIONS):
                raise ValueError("condition fit IDs differ for eligible radish key")
        else:
            candidate=composition[key]["candidate"] if row["product"]!="radish" and key in composition else source["models"]["causal50"]
            fallback=bool(candidate.get("provenance",{}).get("fallback"))
            if key in composition and composition[key].get("candidateSource")=="E":
                fallback=fallback or bool(source["models"]["causal50"].get("provenance",{}).get("fallback"))
            expected_status=("fallback" if fallback else "issued") if candidate.get("status")=="issued" else "not_issued"
            for condition in CONDITIONS:
                item=row["conditions"][condition]
                if (item.get("status")!=expected_status or item.get("blendPrediction")!=candidate.get("prediction") or
                        item.get("directPrediction")!=source["models"]["causalDirect"].get("prediction") or
                        item.get("sarimaPrediction")!=source["models"]["causalSarima"].get("prediction")):
                    raise ValueError(f"non-radish/ineligible E106 composed forecast changed at {key}")
        for condition in CONDITIONS:
            item=row["conditions"][condition]
            expected_cutoff=audit.get("cutoff") if audit else None
            if item.get("p0")!=p0 or item.get("p0Date")!=(audit.get("p0Date") if audit else None) or item.get("cutoff")!=expected_cutoff:
                raise ValueError(f"saved forecast cutoff/p0 audit differs at {key}")
            sarima=source["models"]["causalSarima"]
            if item.get("sarimaPrediction")!=sarima.get("prediction"):
                raise ValueError(f"saved SARIMA prediction differs from E106 at {key}")
            if row["product"]=="radish" and row["eligible"]:
                direct=float(item["directPrediction"]);sarima_value=sarima.get("prediction")
                expected_blend=0.5*direct+0.5*float(sarima_value) if sarima.get("status")=="issued" else direct
                expected_status="issued" if sarima.get("status")=="issued" else "fallback"
                if abs(float(item["blendPrediction"])-expected_blend)>PRICE_ABS_TOL+PRICE_REL_TOL*abs(expected_blend) or item.get("status")!=expected_status:
                    raise ValueError(f"saved {condition} same-SARIMA blend/status failed replay at {key}")
    repro_rows={tuple(item["key"])+(item["component"],):item for item in baseline_repro.get("rows",[])}
    checked=0
    for row in rows:
        if row["product"]!="radish" or not row["eligible"]: continue
        key=(row["product"],row["origin"],row["target"],row["horizon"])
        source=baseline[key]
        for component,model_name,pred_name in (("direct","causalDirect","directPrediction"),("blend","causal50","blendPrediction")):
            item=repro_rows.get(key+(component,))
            if item is None: raise ValueError(f"missing baseline reproduction row for {key}/{component}")
            old=source["models"][model_name].get("prediction")
            actual=row["conditions"]["B"].get(pred_name)
            if item.get("existing")!=old or item.get("reproduced")!=actual:
                raise ValueError(f"baseline reproduction values changed for {key}/{component}")
            if old is None or actual is None:
                if old is not actual: raise ValueError(f"baseline issue state changed for {key}/{component}")
            elif abs(float(actual)-float(old))>PRICE_ABS_TOL+PRICE_REL_TOL*abs(float(old)):
                raise ValueError(f"baseline price reproduction failed at {key}/{component}")
            checked+=1
    if checked!=baseline_repro.get("checkedValues"):
        raise ValueError("baseline reproduction count does not match eligible B values")
    fit_map = {fit.get("fitId"): fit for fit in manifest.get("jointFitDiagnostics", [])}
    replay_count = 0
    max_log_error = 0.0
    for row in rows:
        key=(row["product"],row["origin"],row["target"],row["horizon"])
        eligible = dt.date.fromisoformat(row["origin"]) >= origin_start
        if row.get("eligible") is not eligible:
            raise ValueError(f"eligibility mismatch for {key}")
        if row["product"] != "radish" or not eligible:
            continue
        fit = fit_map.get(row.get("conditions", {}).get("B", {}).get("fitId"))
        if fit is None or fit.get("status") != "ok" or not isinstance(row.get("featureVector"), list):
            raise ValueError(f"missing fitted state or feature vector for {key}")
        h = int(row["horizon"]) - 1
        raw = np.asarray(row["featureVector"], dtype=np.float64)[list(ACTIVE)]
        std = (raw - np.asarray(fit["standardScalerMean"][h])) / np.asarray(fit["standardScalerScale"][h])
        centered = std - np.asarray(fit["scalerXMean"][h])
        for condition in CONDITIONS:
            beta = np.asarray(fit["fitCoefficients"][condition][h], dtype=np.float64)
            expected_log = float(centered @ beta + fit["scalerYMean"][h])
            item = row["conditions"][condition]
            delta = abs(expected_log - float(item["directLogPrediction"]))
            max_log_error = max(max_log_error, delta)
            if delta > 1e-12 + 1e-10 * max(1.0, abs(expected_log)):
                raise ValueError(f"stored {condition} log prediction replay failed for {key}")
            p0 = float(item["p0"])
            price = p0 * math.exp(expected_log)
            if abs(price-float(item["directPrediction"])) > PRICE_ABS_TOL + PRICE_REL_TOL*abs(float(item["directPrediction"])):
                raise ValueError(f"stored {condition} direct price replay failed for {key}")
        replay_count += 1
    return {"phase":phase,"status":"passed","manifestSha256":file_sha(manifest_path),
            "forecastCsvSha256":file_sha(csv_path),"forecastKeys":len(rows),"coefficientReplayRows":replay_count,
            "maximumLogReplayDifference":max_log_error,"baselineReproductionStatus":baseline_repro["status"],
            "protocolSha256":protocol["protocolSha256"]}


def _actuals_for_score() -> dict[tuple[str,str],float]:
    return _actuals_by_product_target(_json(PRICES_PATH))


def _scoring_baseline_rows(phase: str) -> dict[tuple[str,str,str,int],dict[str,Any]]:
    source=_baseline_index(_json(BASELINE_PATH),phase);composition=_composition_index(phase);out={}
    for key,row in source.items():
        d=row["models"]["causalDirect"];s=row["models"]["causalSarima"]
        policy=row["models"]["causal50"] if key[0]=="radish" or key not in composition else composition[key]["candidate"]
        fallback=bool(policy.get("provenance",{}).get("fallback"))
        if key[0]!="radish" and key in composition and composition[key].get("candidateSource")=="E":
            fallback=fallback or bool(row["models"]["causal50"].get("provenance",{}).get("fallback"))
        out[key]={"status":"fallback" if fallback and policy.get("status")=="issued" else policy.get("status"),
                  "fallback":fallback,"prediction":policy.get("prediction"),"reason":policy.get("reason"),
                  "directStatus":d.get("status"),"directPrediction":d.get("prediction"),
                  "sarimaStatus":s.get("status"),"sarimaPrediction":s.get("prediction"),
                  "fallbackReason":policy.get("provenance",{}).get("fallbackReason") or row["models"]["causal50"].get("provenance",{}).get("fallbackReason")}
    return out


def _compute_score_report(phase: str, output_dir: Path, verification: Mapping[str,Any]) -> dict[str,Any]:
    from . import radish_horizon_sharing_score as scorer
    manifest_path=output_dir/("validation-manifest.json" if phase=="V" else "test-manifest.json")
    manifest=_json(manifest_path)
    baseline_rows=_scoring_baseline_rows(phase)
    actuals=_actuals_for_score()
    report=scorer.score_manifest(manifest,baseline_rows=baseline_rows,actuals=actuals,fit_diagnostics=manifest["fitDiagnostics"],phase=phase)
    report["primaryGateEvidence"]=_gate_evidence(report,manifest,baseline_rows) if phase=="V" else None
    report["fourCrop40CellAuxiliary"]=_four_crop_auxiliary(report)
    report["validationManifestSha256" if phase=="V" else "testManifestSha256"]=file_sha(manifest_path)
    report["verification"]=verification
    return report


def _score_phase(phase: str, output_dir: Path = OUT) -> dict[str, Any]:
    _load_protocol(output_dir)
    if phase == "T":
        marker_path=output_dir/"test-manifest.json"
        if marker_path.exists():
            marker=_json(marker_path)
            if marker.get("status")=="not_run_validation_failed":
                return _json(output_dir/"test-report.json")
    verification=verify_phase(phase,output_dir)
    report=_compute_score_report(phase,output_dir,verification)
    report_path=output_dir/("validation-report.json" if phase=="V" else "test-report.json")
    raw=canonical(report)
    if report_path.exists():
        if report_path.read_bytes()!=raw:
            raise ValueError("existing score report differs from verified recomputation")
        return _json(report_path)
    _write_exclusive(report_path,raw)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("freeze","preflight","build","verify","score"))
    parser.add_argument("--phase",choices=("V","T"))
    parser.add_argument("--output-dir",type=Path,default=OUT)
    args=parser.parse_args(argv)
    if args.command=="freeze":
        frozen=freeze_protocol(args.output_dir)
        result={"status":"frozen","protocolSha256":frozen["protocolSha256"],"path":str(args.output_dir/"protocol.json")}
    elif args.command=="preflight":
        if args.phase is None: parser.error("--phase required")
        result=_preflight(args.phase,args.output_dir)
    elif args.command=="build":
        if args.phase is None: parser.error("--phase required")
        built=build_phase(args.phase,args.output_dir)
        result={"phase":args.phase,"status":built.get("status","built"),"forecastRows":len(built.get("forecasts",[])),
                "monthlyJointFits":len(built.get("jointFitDiagnostics",[])),"path":str(args.output_dir/("validation-manifest.json" if args.phase=="V" else "test-manifest.json"))}
    elif args.command=="verify":
        if args.phase is None: parser.error("--phase required")
        result=verify_phase(args.phase,args.output_dir)
    else:
        if args.phase is None: parser.error("--phase required")
        report=_score_phase(args.phase,args.output_dir)
        result={"phase":args.phase,"status":report.get("status","scored"),
                "primaryGateEvidence":report.get("primaryGateEvidence"),
                "reportPath":str(args.output_dir/("validation-report.json" if args.phase=="V" else "test-report.json"))}
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
