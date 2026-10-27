"""Fit-free coefficient-to-prediction-to-price transmission diagnostic for RHS.

This module consumes only the already-frozen E106-RHS V/T artifacts. It never
fits a model or chooses a new candidate. The diagnostic protocol must be frozen
before ``run`` is allowed to inspect forecast movements or actual prices.
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
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import radish_horizon_sharing as rhs

ROOT = Path(__file__).resolve().parents[1]
SOURCE_OUT = ROOT / "outputs/experiments/radish-horizon-sharing-v1-20261011"
OUT = ROOT / "outputs/experiments/rhs-transmission-diagnostic-20261012"
PHASE_FORECASTS = {"V": 755, "T": 790}
PHASE_PAIRED = {"V": 651, "T": 650}
PHASE_GRID_FORECASTS = {"V": 3240, "T": 3160}
CONDITIONS = ("D", "C", "P")
RANKS = (10, 20, 30, 39)
EIGEN_TIE_REL_TOL = 1e-8
PRICE_ABS_TOL = 1e-6
PRICE_REL_TOL = 1e-8
LOG_ABS_TOL = 1e-12
LOG_REL_TOL = 1e-10
INTERCEPT_CENTER_TOL = 2e-11

ORIGINAL_FILES = (
    "protocol.json",
    "validation-claim.json", "validation-terminal.json", "validation-manifest.json",
    "validation-predictions.csv", "fit-diagnostics.json", "baseline-reproduction.json",
    "validation-report.json",
    "test-claim.json", "test-terminal.json", "test-manifest.json",
    "test-predictions.csv", "test-fit-diagnostics.json", "test-baseline-reproduction.json",
    "test-report.json",
)


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _original_artifact_paths() -> dict[str, Path]:
    paths = {name: SOURCE_OUT / name for name in ORIGINAL_FILES}
    for phase, month in (("V", "04"), ("V", "05"), ("V", "06"),
                         ("T", "07"), ("T", "08"), ("T", "09")):
        fit_id = f"radish-2026-{month}-01"
        prefix = f"{phase.lower()}-{fit_id}"
        for suffix in (".claim.json", ".json", ".terminal.json"):
            name = f"fit-checkpoints/{prefix}{suffix}"
            paths[name] = SOURCE_OUT / name
    paths["baselineManifest"] = rhs.BASELINE_PATH
    paths["prices"] = rhs.PRICES_PATH
    return paths


def _source_hashes() -> dict[str, str]:
    from . import equal_info_online_forecasts as base
    from . import equal_info_online_score as inference
    from . import radish_horizon_sharing_score as scorer

    named = {
        "rhs": Path(rhs.__file__),
        "equalInfo": Path(base.__file__),
        "modeling": Path(base.modeling.__file__),
        "inference": Path(inference.__file__),
        "scorer": Path(scorer.__file__),
        "narrow": Path(__import__("agri_data.full_history_return_eval", fromlist=["__file__"]).__file__),
        "rich": Path(__import__("agri_data.full_history_xgb_rf_eval", fromlist=["__file__"]).__file__),
        "featureLibrary": Path(__import__("agri_data.full_history_feature_library", fromlist=["__file__"]).__file__),
        "directHorizon": Path(base.direct_horizon.__file__),
    }
    return {name: file_sha(path) for name, path in named.items()}


def _protocol_body() -> dict[str, Any]:
    original_protocol = _json(SOURCE_OUT / "protocol.json")
    if original_protocol.get("protocolSha256") != sha256_bytes(canonical(original_protocol.get("protocol"))):
        raise ValueError("original RHS protocol digest is invalid")
    paths = _original_artifact_paths()
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"RHS source artifacts are incomplete: {missing}")
    import matplotlib
    import scipy

    return {
        "schemaVersion": 1,
        "id": "rhs-transmission-diagnostic-20261012-v1",
        "sourceExperiment": "radish-horizon-sharing-v1-20261011",
        "sourceProtocolSha256": original_protocol["protocolSha256"],
        "inputSha256": {name: file_sha(path) for name, path in paths.items()},
        "sourceSha256": _source_hashes(),
        "diagnosticSourceSha256": file_sha(Path(__file__)),
        "priceMovementSourceSha256": _price_movement_source_sha(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "support": {"V": {"allIssuedRadish": PHASE_FORECASTS["V"], "pairedCommonPositiveActual": PHASE_PAIRED["V"]},
                    "T": {"allIssuedRadish": PHASE_FORECASTS["T"], "pairedCommonPositiveActual": PHASE_PAIRED["T"]}},
        "geometry": {
            "coordinates": "Xstd=(Xraw-standardScalerMeanFull)/standardScalerScaleFull; Xcentered=Xstd[:,ACTIVE]-scalerXMean[h]; D_h=diag(scale_ref/scale_h); U=Xcentered@D_h^-1; q=D_h beta",
            "eigensystem": "float64 symmetric eigh of U_train.T@U_train/n, ascending eigenvalue order; retain all 39 directions",
            "negativeEigenvalueTolerance": "64*eps64*39*max(1,||U.T@U/n||_2); preserve raw values; clip only values within tolerance to zero for directional-energy summation; larger negatives fail",
            "nearDegenerateEigenvalueRelativeTolerance": EIGEN_TIE_REL_TOL,
            "rankCheckpoints": list(RANKS),
            "degenerateEigenvalues": "individual eigenvector contributions within repeated or numerically near-degenerate eigenspaces depend on the selected basis; all-rank curves are descriptive, not feature or causal attributions",
            "intercept": "use saved fitIntercepts; c=(b_A-b_B)+scalerXMean[h]@(beta_A-beta_B), report and verify near zero without forcing it to zero",
            "evaluation": "use training mean and eigensystem unchanged; evaluation full movement includes intercept c and all cross-direction terms; never re-center evaluation inputs",
        },
        "priceMovement": {"comparisons": ["D-B", "C-B", "P-B"], "layers": ["direct", "blend"],
                          "support": "one common four-condition positive-actual radish key set per phase; equal weighting across ten horizons",
                          "forecastIssueCounts": PHASE_FORECASTS, "pairedCounts": PHASE_PAIRED,
                          "noAdditionalScoresOrInference": True},
        "forbidden": ["model fitting", "new candidate selection", "new statistical tests", "log-error decomposition", "future data"],
    }


def _price_movement_source_sha() -> str:
    path = ROOT / "agri_data/rhs_price_movement.py"
    if not path.is_file():
        raise FileNotFoundError("the shared RHS price movement helper is not yet available")
    return file_sha(path)


def freeze_protocol(output_dir: Path = OUT) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "protocol.json"
    body = _protocol_body()
    doc = {"protocol": body, "protocolSha256": sha256_bytes(canonical(body))}
    _write_exclusive(path, canonical(doc))
    return doc


def _load_protocol(output_dir: Path = OUT) -> dict[str, Any]:
    doc = _json(output_dir / "protocol.json")
    if doc.get("protocolSha256") != sha256_bytes(canonical(doc.get("protocol"))):
        raise ValueError("diagnostic protocol digest mismatch")
    if doc["protocol"] != _protocol_body():
        raise ValueError("diagnostic inputs or sources differ from the frozen protocol")
    return doc


def _movement_module():
    from . import rhs_price_movement
    return rhs_price_movement


def _array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if (ndim is not None and result.ndim != ndim) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {ndim or 'numeric'}-dimensional array")
    return result


def _tolerance(eigenvalues: np.ndarray) -> float:
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    return 64.0 * np.finfo(np.float64).eps * 39.0 * scale


def _fit_geometry(fit: Mapping[str, Any], Xraw_by_h: Sequence[np.ndarray],
                  *, phase: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, np.ndarray]]:
    """Return transmission rows, curves, and training eigenvectors by horizon."""
    fit_id = str(fit["fitId"])
    means_full = _array(fit["standardScalerMeanFull"], "saved full scaler mean", 2)
    scales_full = _array(fit["standardScalerScaleFull"], "saved full scaler scale", 2)
    means = _array(fit["scalerXMean"], "saved training centering mean", 2)
    scales = _array(fit["scalerScale"], "saved fit scale", 2)
    if means_full.shape != (10, 40) or scales_full.shape != (10, 40) or means.shape != (10, 39) or scales.shape != (10, 39):
        raise ValueError(f"saved coordinate arrays have the wrong shape for {fit_id}")
    ref = scales[0]
    transmission: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    training_basis: dict[int, np.ndarray] = {}
    for hi in range(10):
        h = hi + 1
        raw = _array(Xraw_by_h[hi], f"training raw X {fit_id} h={h}", 2)
        if raw.shape[1] != 40:
            raise ValueError("reconstructed training X must preserve all 40 features")
        xstd = (raw - means_full[hi]) / scales_full[hi]
        xcenter = xstd[:, list(rhs.ACTIVE)] - means[hi]
        inv_d = scales[hi] / ref
        U = xcenter * inv_d
        gram = U.T @ U / len(U)
        gram = (gram + gram.T) * 0.5
        eigenvalues, vectors = np.linalg.eigh(gram)
        training_basis[h] = vectors.copy()
        tol = _tolerance(eigenvalues)
        if float(eigenvalues[0]) < -tol:
            raise ArithmeticError(f"materially negative training Gram eigenvalue at {fit_id} h={h}: {eigenvalues[0]} < {-tol}")
        eig_for_energy = np.where(eigenvalues < 0.0, 0.0, eigenvalues)
        tie_tol=EIGEN_TIE_REL_TOL*max(1.0,float(np.max(np.abs(eigenvalues))))
        tie_group=np.zeros(39,dtype=np.int64)
        next_group=0
        tie_group[0]=next_group
        for rank_i in range(1,39):
            if eigenvalues[rank_i]-eigenvalues[rank_i-1]>tie_tol:
                next_group+=1
            tie_group[rank_i]=next_group
        group_size={int(group):int(np.sum(tie_group==group)) for group in np.unique(tie_group)}
        if not np.allclose(vectors.T @ vectors, np.eye(39), rtol=0, atol=2e-12):
            raise ArithmeticError(f"training eigensystem is not orthonormal at {fit_id} h={h}")
        row_condition_values = {}
        for condition in ("B", *CONDITIONS):
            beta = _array(fit["fitCoefficients"][condition][hi], f"{condition} beta", 1)
            if beta.shape != (39,):
                raise ValueError(f"saved coefficient vector shape mismatch at {fit_id} h={h}")
            intercept = float(fit["fitIntercepts"][condition][hi])
            if not math.isfinite(intercept):
                raise ValueError("saved fit intercept must be finite")
            row_condition_values[condition] = (beta, intercept)
        beta_b, intercept_b = row_condition_values["B"]
        q_b = _array(fit["sharedCoordinateCoefficients"]["B"][hi], "B shared coordinate coefficient", 1)
        if not np.allclose(q_b, ref / scales[hi] * beta_b, rtol=1e-12, atol=1e-12):
            raise ValueError(f"saved B shared-coordinate coefficients do not reconstruct at {fit_id} h={h}")
        for condition in CONDITIONS:
            beta_a, intercept_a = row_condition_values[condition]
            saved_training=_array(fit["trainingPredictions"][condition][hi],f"saved {condition} training predictions",1)
            full_training=xcenter @ beta_a + float(_array(fit["scalerYMean"],"saved training response means",1)[hi])
            if saved_training.shape!=full_training.shape or not np.allclose(saved_training,full_training,rtol=0,atol=1e-11):
                raise ValueError(f"saved {condition} training predictions do not replay from reconstructed X at {fit_id} h={h}")
            q_a = _array(fit["sharedCoordinateCoefficients"][condition][hi], f"{condition} shared coordinate coefficient", 1)
            delta_beta = beta_a - beta_b
            delta_q = q_a - q_b
            expected_delta_q = ref / scales[hi] * delta_beta
            if not np.allclose(delta_q, expected_delta_q, rtol=1e-12, atol=1e-12):
                raise ValueError(f"saved {condition} shared-coordinate change does not reconstruct at {fit_id} h={h}")
            intercept_delta = intercept_a - intercept_b
            centering_delta = float(means[hi] @ delta_beta)
            c = intercept_delta + centering_delta
            if abs(c) > INTERCEPT_CENTER_TOL * max(1.0, abs(intercept_delta), abs(centering_delta)):
                raise ArithmeticError(f"intercept/centering restoration does not cancel at {fit_id} h={h} {condition}: {c}")
            projections = vectors.T @ delta_q
            coefficient_energy = projections ** 2
            training_direction_energy = eig_for_energy * coefficient_energy
            movement_spectral = float(math.fsum(training_direction_energy.tolist()))
            train_delta_saved = xstd[:, list(rhs.ACTIVE)] @ delta_beta + intercept_delta
            train_delta_coord = U @ delta_q + c
            if not np.allclose(train_delta_saved, train_delta_coord, rtol=LOG_REL_TOL, atol=LOG_ABS_TOL):
                raise ArithmeticError(f"training coefficient/intercept replay differs from U delta-q at {fit_id} h={h} {condition}")
            movement_direct = float(np.mean(train_delta_saved ** 2))
            train_mean_u = U.mean(axis=0)
            mean_energy_with_intercept = movement_spectral + 2.0 * c * float(train_mean_u @ delta_q) + c * c
            if abs(movement_direct - mean_energy_with_intercept) > 5e-12 + 5e-10 * max(1.0, movement_direct):
                raise ArithmeticError(f"spectral training movement does not replay at {fit_id} h={h} {condition}")
            norm_sq = float(delta_q @ delta_q)
            weighted_eigenvalue = movement_spectral / norm_sq if norm_sq > 0 else None
            row_condition_values[condition] = row_condition_values[condition] + (delta_q, projections, coefficient_energy,
                                                                                 training_direction_energy)
            transmission.append({
                "phase": phase, "fitId": fit_id, "fitOrigin": fit_id.removeprefix("radish-"), "horizon": h,
                "condition": condition, "trainingRows": len(U), "coordinateDimension": 39,
                "coefficientChangeSquared": norm_sq, "coefficientChangeNorm": math.sqrt(norm_sq),
                "trainingPredictionChangeMSE": movement_direct,
                "trainingPredictionChangeRms": math.sqrt(max(0.0, movement_direct)),
                "spectralTrainingPredictionChangeMSE": movement_spectral,
                "spectralReplayDifference": movement_direct - mean_energy_with_intercept,
                "weightedEigenvalue": weighted_eigenvalue, "interceptDifference": intercept_delta,
                "centeringContribution": centering_delta, "restoredInterceptAndCenteringResidual": c,
                "trainingCenteredCoordinateMeanNorm": float(np.linalg.norm(train_mean_u)),
                "negativeEigenvalueTolerance": tol,
                "rawMinimumEigenvalue": float(eigenvalues[0]),
                "clippedRoundoffNegativeCount": int(np.sum(eigenvalues < 0.0)),
            })
            total_coef = float(math.fsum(coefficient_energy.tolist()))
            total_energy = movement_spectral
            for rank in range(1, 40):
                curves.append({
                    "phase": phase, "fitId": fit_id, "fitOrigin": fit_id.removeprefix("radish-"),
                    "horizon": h, "condition": condition, "ascendingRank": rank,
                    "eigenvalueRaw": float(eigenvalues[rank-1]),
                    "eigenvalueForEnergy": float(eig_for_energy[rank-1]),
                    "eigenvalueTolerance": tol, "projection": float(projections[rank-1]),
                    "nearDegenerateGroup": int(tie_group[rank-1]),
                    "nearDegenerateGroupSize": group_size[int(tie_group[rank-1])],
                    "nearDegenerateWithPrevious": bool(rank>1 and tie_group[rank-1]==tie_group[rank-2]),
                    "coefficientChangeContribution": float(coefficient_energy[rank-1]),
                    "coefficientChangeCumulative": float(math.fsum(coefficient_energy[:rank].tolist())),
                    "coefficientChangeCumulativeShare": (float(math.fsum(coefficient_energy[:rank].tolist()) / total_coef) if total_coef > 0 else None),
                    "trainingPredictionEnergyContribution": float(training_direction_energy[rank-1]),
                    "trainingPredictionEnergyCumulative": float(math.fsum(training_direction_energy[:rank].tolist())),
                    "trainingPredictionEnergyCumulativeShare": (float(math.fsum(training_direction_energy[:rank].tolist()) / total_energy) if total_energy > 0 else None),
                    "totalCoefficientChangeSquared": total_coef,
                    "totalTrainingPredictionChangeMSE": total_energy,
                    "degenerateSpectrumCaveat": "individual eigenvector contribution is basis-dependent within repeated or near-repeated eigenvalue spaces",
                })
    return transmission, curves, training_basis


def _eval_replay(phase: str, manifest: Mapping[str, Any], fit_map: Mapping[str, Mapping[str, Any]],
                training_bases: Mapping[str, Mapping[int, np.ndarray]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    replay_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for row in manifest["forecasts"]:
        if row.get("product") != "radish" or not row.get("eligible"):
            continue
        item = row["conditions"]["B"]
        fit_id = item.get("fitId")
        if fit_id not in fit_map or not isinstance(row.get("featureVector"), list):
            raise ValueError("eligible radish forecast lacks its saved monthly fit or feature vector")
        fit = fit_map[fit_id]
        h = int(row["horizon"])
        hi = h - 1
        raw = _array(row["featureVector"], "evaluation feature vector", 1)
        if raw.shape != (40,):
            raise ValueError("evaluation feature vector must preserve 40 columns")
        z = (raw - _array(fit["standardScalerMeanFull"][hi], "evaluation scaler mean", 1)) / _array(fit["standardScalerScaleFull"][hi], "evaluation scaler scale", 1)
        z_active = z[list(rhs.ACTIVE)]
        row_key = (row["originDate"], row["targetDate"], h)
        for condition in ("B", *CONDITIONS):
            beta = _array(fit["fitCoefficients"][condition][hi], f"{condition} evaluation beta", 1)
            intercept = float(fit["fitIntercepts"][condition][hi])
            log_prediction = float(z_active @ beta + intercept)
            saved = row["conditions"][condition]
            if abs(log_prediction - float(saved["directLogPrediction"])) > LOG_ABS_TOL + LOG_REL_TOL * max(1.0, abs(log_prediction)):
                raise ValueError(f"saved evaluation log prediction fails replay: {row_key} {condition}")
            price_prediction = float(row["p0"] * math.exp(log_prediction))
            stored_price = float(saved["directPrediction"])
            if abs(price_prediction-stored_price) > PRICE_ABS_TOL + PRICE_REL_TOL*abs(stored_price):
                raise ValueError(f"saved evaluation direct price fails replay: {row_key} {condition}")
            replay_rows.append({"phase": phase, "fitId": fit_id, "originDate": row["originDate"], "targetDate": row["targetDate"],
                                "horizon": h, "condition": condition, "savedLogPrediction": float(saved["directLogPrediction"]),
                                "replayedLogPrediction": log_prediction, "logDifference": log_prediction-float(saved["directLogPrediction"]),
                                "savedDirectPrediction": stored_price, "replayedDirectPrediction": price_prediction,
                                "directPriceDifference": price_prediction-stored_price})
    if len(replay_rows) != PHASE_FORECASTS[phase] * 4:
        raise ValueError(f"expected {PHASE_FORECASTS[phase]} eligible radish forecasts in phase {phase}")
    # Decompose evaluation movement in the unchanged training eigenbasis.
    # Forecast keys are origin-specific; aggregate per fit/horizon across those issue keys.
    for fit_id, fit in fit_map.items():
        for h in range(1, 11):
            selected = [r for r in manifest["forecasts"] if r.get("product") == "radish" and r.get("eligible") and
                        r["conditions"]["B"]["fitId"] == fit_id and int(r["horizon"]) == h]
            if not selected:
                raise ValueError(f"no evaluation rows for {fit_id} h={h}")
            hi = h-1
            m = _array(fit["scalerXMean"][hi], "evaluation centering mean", 1)
            scales = _array(fit["scalerScale"], "evaluation scale", 2)
            U=[]
            for row in selected:
                z = (_array(row["featureVector"], "evaluation raw feature", 1) -
                     _array(fit["standardScalerMeanFull"][hi], "evaluation full mean", 1)) / \
                    _array(fit["standardScalerScaleFull"][hi], "evaluation full scale", 1)
                U.append((z[list(rhs.ACTIVE)] - m) * (scales[hi]/scales[0]))
            ueval=np.asarray(U,dtype=np.float64)
            vec=_array(training_bases[fit_id][h],"saved training eigenvectors",2)
            if vec.shape!=(39,39) or not np.allclose(vec.T@vec,np.eye(39),rtol=0,atol=2e-12):
                raise ValueError(f"invalid training eigenbasis for evaluation: {fit_id} h={h}")
            for condition in CONDITIONS:
                beta_a=_array(fit["fitCoefficients"][condition][hi], "evaluation candidate beta", 1)
                beta_b=_array(fit["fitCoefficients"]["B"][hi], "evaluation B beta", 1)
                q_a=_array(fit["sharedCoordinateCoefficients"][condition][hi], "evaluation candidate q", 1)
                q_b=_array(fit["sharedCoordinateCoefficients"]["B"][hi], "evaluation B q", 1)
                delta_q=q_a-q_b
                delta_beta=beta_a-beta_b
                c=(float(fit["fitIntercepts"][condition][hi])-float(fit["fitIntercepts"]["B"][hi]))+float(m@delta_beta)
                components=(ueval@vec)*(vec.T@delta_q)[None,:]
                delta_coordinate=ueval@delta_q+c
                projection_sum=components.sum(axis=1)+c
                if not np.allclose(delta_coordinate,projection_sum,rtol=LOG_REL_TOL,atol=LOG_ABS_TOL):
                    raise ArithmeticError(f"evaluation eigen-direction replay failed: {fit_id} h={h} {condition}")
                diagonal=float(np.mean(np.sum(components**2,axis=1)))
                full=float(np.mean(delta_coordinate**2))
                cross=float(full-diagonal)
                # Cross terms can be negative. Verify the full quadratic identity directly.
                matrix_mean=(components.T@components)/len(components)
                expanded=float(np.trace(matrix_mean)+2.0*math.fsum(float(matrix_mean[j,k]) for j in range(39) for k in range(j+1,39))+
                               2.0*c*float(components.sum(axis=1).mean())+c*c)
                if abs(full-expanded)>5e-12+5e-10*max(1.0,full):
                    raise ArithmeticError(f"evaluation diagonal/cross-term identity failed: {fit_id} h={h} {condition}")
                direction_rows.append({"phase":phase,"fitId":fit_id,"fitOrigin":fit_id.removeprefix("radish-"),"horizon":h,
                                       "condition":condition,"evaluationIssueRows":len(selected),"evaluationLogMovementMSE":full,
                                       "evaluationLogMovementRms":math.sqrt(max(0.0,full)),
                                       "diagonalDirectionalEnergy":diagonal,"crossDirectionAndInterceptTerms":cross,
                                       "interceptCenteringResidual":c,"fullMinusDiagonal":full-diagonal,
                                       "directionalIdentityDifference":full-expanded,
                                       "evaluationRecentered":False,"basisSource":"trainingGramEigenvectors",
                                       "trainingBasisOrthogonalityMaxError":float(np.max(np.abs(vec.T@vec-np.eye(39))))})
    if len(direction_rows)!=90:
        raise ValueError("evaluation direction diagnostics must cover all six month-by-h fits and three candidates")
    return replay_rows,direction_rows


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    import io
    fields=[]
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    buf=io.StringIO(newline="")
    writer=csv.DictWriter(buf,fieldnames=fields,lineterminator="\n",extrasaction="raise")
    writer.writeheader();writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _rank_summary(curves: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup={(r["phase"],r["fitId"],r["horizon"],r["condition"],r["ascendingRank"]):r for r in curves}
    out=[]
    groups=sorted({(r["phase"],r["fitId"],r["horizon"],r["condition"]) for r in curves})
    for phase,fit_id,horizon,condition in groups:
        for rank in RANKS:
            row=lookup[(phase,fit_id,horizon,condition,rank)]
            out.append({"phase":phase,"fitId":fit_id,"horizon":horizon,"condition":condition,"rank":rank,
                        "cumulativeCoefficientChangeShare":row["coefficientChangeCumulativeShare"],
                        "cumulativeTrainingPredictionEnergyShare":row["trainingPredictionEnergyCumulativeShare"],
                        "interpretation":"descriptive checkpoint only; not a selected rank cutoff"})
    return out


def _plot_curves(curves: Sequence[Mapping[str, Any]], output_dir: Path) -> list[dict[str,str]]:
    import matplotlib.pyplot as plt
    outputs=[]
    for metric,title,suffix in (("coefficientChangeCumulativeShare","Cumulative coefficient-change share","coefficient"),
                                ("trainingPredictionEnergyCumulativeShare","Cumulative training prediction-energy share","training-energy")):
        fig,axes=plt.subplots(4,5,figsize=(16,13),sharex=True,sharey=True)
        for phase_i,phase in enumerate(("V","T")):
            phase_rows=[r for r in curves if r["phase"]==phase]
            for hi in range(1,11):
                ax=axes[phase_i*2+(hi-1)//5,(hi-1)%5]
                for condition,color in zip(CONDITIONS,("#2878B5","#D95319","#2B8C56")):
                    month_curves=[]
                    for fit_id in sorted({r["fitId"] for r in phase_rows}):
                        vals=sorted((r for r in phase_rows if r["fitId"]==fit_id and r["horizon"]==hi and r["condition"]==condition),key=lambda r:r["ascendingRank"])
                        if vals:
                            ys=[v[metric] for v in vals]
                            if all(v is not None for v in ys):
                                month_curves.append(ys)
                                ax.plot(range(1,40),ys,color=color,alpha=.22,linewidth=.8)
                    if month_curves:
                        ax.plot(range(1,40),np.mean(np.asarray(month_curves),axis=0),color=color,linewidth=2,label=condition)
                ax.set_title(f"{phase}, h={hi}");ax.grid(alpha=.2)
                if hi in (1,6): ax.set_ylabel("Cumulative share")
                if hi>5: ax.set_xlabel("Ascending eigenvalue rank")
                if phase_i==0 and hi==1: ax.legend(title="Condition",fontsize=8)
        fig.suptitle(f"RHS V/T: {title} by horizon\nThin lines are monthly folds; thick lines are the unweighted mean of three normalized monthly curves (not a pooled-energy curve)")
        fig.tight_layout()
        path=output_dir/f"{suffix}-curves-VT.png"
        buffer=__import__("io").BytesIO()
        fig.savefig(buffer,format="png",dpi=160)
        plt.close(fig)
        _write_exclusive(path,buffer.getvalue())
        outputs.append({"file":path.name,"sha256":file_sha(path)})
    return outputs


def _verify_price_against_original(phase: str, movement: Mapping[str, Any]) -> None:
    report_path=SOURCE_OUT/("validation-report.json" if phase=="V" else "test-report.json")
    report=_json(report_path)
    # The helper is descriptive, but verify its fixed support and every
    # all-four comparison against the originally saved scorer cells.
    aggregates=movement.get("aggregates",[])
    expected_horizon={(int(r["horizon"]),r["condition"]):r for r in report["allFourCommonMetrics"] if r["product"]=="radish"}
    candidates=[r for r in aggregates if r.get("scope")=="horizon"]
    if len(candidates)!=60:
        raise ValueError("price movement helper must return all 30 all-four horizon comparisons in both price layers")
    for row in candidates:
        original=expected_horizon[(int(row["horizon"]),row["condition"])]
        other=expected_horizon[(int(row["horizon"]),"B")]
        condition=original
        mae_key="blendMae" if row.get("layer")=="blend" else "directMae"
        mape_key="blendMape" if row.get("layer")=="blend" else "directMape"
        checks=((row.get("n"),original["n"]),(row.get("baselineMae"),other[mae_key]),(row.get("conditionMae"),condition[mae_key]),
                (row.get("baselineMapePct"),other[mape_key]),(row.get("conditionMapePct"),condition[mape_key]))
        for observed,want in checks:
            if observed is None or abs(float(observed)-float(want))>2e-10*max(1.0,abs(float(want))):
                raise ValueError(f"price movement {row.get('comparison')} {row.get('layer')} h={row.get('horizon')} differs from saved all-four report")

    saved_month={(r["targetMonth"],int(r["horizon"]),r["condition"]):r for r in report["allFourMonthlyMetrics"] if r["product"]=="radish"}
    monthly=[r for r in aggregates if r.get("scope")=="target_month"]
    months=sorted({key[0] for key in saved_month})
    if len(monthly)!=len(months)*6:
        raise ValueError("price movement helper month aggregates do not cover each all-four comparison/layer/month")
    for row in monthly:
        month=row["targetMonth"]
        per_h=[saved_month[(month,h,row["condition"])] for h in range(1,11)]
        per_h_baseline=[saved_month[(month,h,"B")] for h in range(1,11)]
        if not row.get("hasAllTenHorizons") or any(x["n"]<=0 for x in per_h):
            raise ValueError(f"saved all-four monthly report lacks complete positive support for {month}")
        layer=row["layer"]
        mae_key="blendMae" if layer=="blend" else "directMae"
        mape_key="blendMape" if layer=="blend" else "directMape"
        equal=row["recomputedEqualHorizonMonthMean"]
        checks=((row.get("n"),sum(x["n"] for x in per_h)),(equal.get("baselineMae"),math.fsum(x[mae_key] for x in per_h_baseline)/10),
                (equal.get("conditionMae"),math.fsum(x[mae_key] for x in per_h)/10),
                (equal.get("baselineMapePct"),math.fsum(x[mape_key] for x in per_h_baseline)/10),
                (equal.get("conditionMapePct"),math.fsum(x[mape_key] for x in per_h)/10))
        for observed,want in checks:
            if observed is None or abs(float(observed)-float(want))>2e-10*max(1.0,abs(float(want))):
                raise ValueError(f"price movement {row.get('comparison')} {layer} target month {month} differs from saved all-four report")


def _validate_phase_manifest_metadata(phase: str, manifest: Mapping[str, Any], fit_map: Mapping[str, Any]) -> None:
    months=("04","05","06") if phase=="V" else ("07","08","09") if phase=="T" else None
    if months is None:
        raise ValueError("phase must be V or T")
    expected_fit_ids={f"radish-2026-{month}-01" for month in months}
    if len(fit_map)!=3 or set(fit_map)!=expected_fit_ids:
        raise ValueError(f"original RHS monthly fit IDs differ from the frozen {phase} phase")
    forecasts=manifest.get("forecasts")
    if not isinstance(forecasts,list) or len(forecasts)!=PHASE_GRID_FORECASTS[phase]:
        raise ValueError(f"original RHS full forecast grid differs from the frozen {phase} phase count {PHASE_GRID_FORECASTS[phase]}")


def _expected_paired_keys(radish_rows: Sequence[Mapping[str, Any]], actuals: Mapping[tuple[str,str],float], phase: str) -> set[tuple[str,str,str,int]]:
    expected=set()
    for row in radish_rows:
        key=row["key"]
        actual=actuals.get(("radish",key["targetDate"]))
        if actual is not None and math.isfinite(float(actual)) and float(actual)>0 and all(
                row["conditions"][c]["status"]=="issued" for c in ("B","D","C","P")):
            expected.add((key["product"],key["originDate"],key["targetDate"],int(key["horizon"])))
    if len(expected)!=PHASE_PAIRED[phase]:
        raise ValueError(f"four-condition positive-actual paired support differs from pinned {PHASE_PAIRED[phase]} for {phase}")
    return expected


def _phase_diagnostic(phase: str, raw_prices: Mapping[str, Any]) -> dict[str, Any]:
    from . import equal_info_online_forecasts as base
    if phase not in ("V","T"):
        raise ValueError("phase must be V or T")
    # Recheck the previously frozen RHS source pins, then verify the existing
    # phase artifacts by their frozen bytes and internal links. No old score,
    # fit, or 390x390 numerical audit is rerun here.
    preflight=rhs._preflight(phase,SOURCE_OUT)
    manifest_path=SOURCE_OUT/("validation-manifest.json" if phase=="V" else "test-manifest.json")
    manifest=_json(manifest_path)
    protocol=rhs._load_protocol(SOURCE_OUT)
    if manifest.get("protocolSha256")!=protocol.get("protocolSha256") or manifest.get("inputSha256")!=protocol["protocol"].get("inputSha256") or manifest.get("phase")!=phase:
        raise ValueError("original RHS manifest protocol/phase binding failed")
    csv_path=SOURCE_OUT/("validation-predictions.csv" if phase=="V" else "test-predictions.csv")
    diagnostics_path=SOURCE_OUT/("fit-diagnostics.json" if phase=="V" else "test-fit-diagnostics.json")
    baseline_path=SOURCE_OUT/("baseline-reproduction.json" if phase=="V" else "test-baseline-reproduction.json")
    if file_sha(csv_path)!=manifest.get("forecastFileSha256"):
        raise ValueError("original RHS CSV digest does not match its saved manifest")
    if csv_path.read_bytes()!=rhs._csv_bytes(manifest.get("forecasts",[])):
        raise ValueError("original RHS CSV rows do not serialize from its frozen manifest")
    diagnostics=_json(diagnostics_path)
    if (sha256_bytes(rhs.canonical(manifest.get("fitDiagnostics")))!=manifest.get("fitDiagnosticsSha256") or
            diagnostics!= {"jointFits":manifest.get("jointFitDiagnostics"),"scoringFits":manifest.get("fitDiagnostics")}):
        raise ValueError("original RHS fit-diagnostics link/digest failed")
    if not _json(baseline_path).get("status")=="passed" or _json(baseline_path)!=manifest.get("baselineReproduction"):
        raise ValueError("original RHS baseline reproduction status is not passed")
    claim= _json(SOURCE_OUT/("validation-claim.json" if phase=="V" else "test-claim.json"))
    terminal= _json(SOURCE_OUT/("validation-terminal.json" if phase=="V" else "test-terminal.json"))
    if (terminal.get("status")!="complete" or terminal.get("attemptId")!=claim.get("attemptId") or
            terminal.get("protocolSha256")!=protocol.get("protocolSha256") or terminal.get("manifestSha256")!=file_sha(manifest_path)):
        raise ValueError("original RHS terminal/claim/manifest binding failed")
    fit_map={fit["fitId"]:fit for fit in manifest["jointFitDiagnostics"]}
    if len(fit_map)!=3:
        raise ValueError(f"phase {phase} must contain exactly three monthly saved fits")
    _validate_phase_manifest_metadata(phase,manifest,fit_map)
    transmission=[];curves=[];training_support=[];training_bases={}
    for fit_id,fit in sorted(fit_map.items()):
        fit_origin=dt.date.fromisoformat(fit_id.removeprefix("radish-"))
        series,Xraw,y,targets=rhs._reconstruct_training_fold(fit_origin,raw_prices,base)
        # Recheck the fit's raw/scaled/key/target lineage without fitting.
        Xstd=rhs._verify_fit_state(fit,phase,protocol,Xraw,y,targets)
        checkpoint_prefix=f"{phase.lower()}-{fit_id}"
        checkpoint_dir=SOURCE_OUT/"fit-checkpoints"
        checkpoint=_json(checkpoint_dir/f"{checkpoint_prefix}.json")
        fit_claim=_json(checkpoint_dir/f"{checkpoint_prefix}.claim.json")
        fit_terminal=_json(checkpoint_dir/f"{checkpoint_prefix}.terminal.json")
        input_sha=sha256_bytes(canonical(fit.get("inputHashes")))
        if (checkpoint.get("fit")!=fit or checkpoint.get("inputSha256")!=input_sha or
                checkpoint.get("checkpointSha256")!=sha256_bytes(canonical(fit)) or
                fit_claim.get("inputSha256")!=input_sha or fit_claim.get("inputHashes")!=fit.get("inputHashes") or
                fit_terminal.get("status")!="ok" or fit_terminal.get("inputSha256")!=input_sha or
                fit_terminal.get("checkpointSha256")!=checkpoint.get("checkpointSha256")):
            raise ValueError(f"original RHS fit checkpoint chain failed: {fit_id}")
        training_support.append({"fitId":fit_id,"trainingCounts":[len(a) for a in Xraw],
                                 "trainingKeyCounts":[len(a) for a in targets],
                                 "trainingTargetSha256":fit["inputHashes"]["trainingTarget"],
                                 "rawTrainingMatrixSha256":fit["inputHashes"]["rawTrainingMatrix"],
                                 "standardizedTrainingMatrixSha256":fit["inputHashes"]["trainingMatrix"]})
        tr,cr,basis=_fit_geometry(fit,Xraw,phase=phase)
        transmission.extend(tr);curves.extend(cr)
        training_bases[fit_id]=basis
    replay,directions=_eval_replay(phase,manifest,fit_map,training_bases)
    radish_rows=[]
    for row in manifest["forecasts"]:
        if row.get("product")!="radish" or not row.get("eligible"):
            continue
        radish_rows.append({"key":{"product":"radish","originDate":row["originDate"],"targetDate":row["targetDate"],"horizon":int(row["horizon"])},
                            "eligible":True,
                            "conditions":{c:{k:row["conditions"][c].get(k) for k in ("status","directPrediction","blendPrediction","sarimaPrediction")} for c in ("B","D","C","P")}})
    if len(radish_rows)!=PHASE_FORECASTS[phase]:
        raise ValueError(f"eligible radish issue support differs from pinned count for {phase}")
    actuals=rhs._actuals_by_product_target(raw_prices)
    expected_keys=_expected_paired_keys(radish_rows,actuals,phase)
    movement_module=_movement_module()
    movement=movement_module.diagnose_price_movement(radish_rows,actuals,phase=phase,
                                                     expected_comparison_keys=expected_keys)
    _verify_price_against_original(phase,movement)
    support=movement.get("support",{})
    if support.get("allIssuedKeys")!=PHASE_FORECASTS[phase] or support.get("commonPositiveActualKeys")!=PHASE_PAIRED[phase]:
        raise ValueError(f"price movement helper did not preserve the frozen issue/paired support for {phase}")
    return {"phase":phase,"originalPreflight":preflight,"trainingSupport":training_support,"issueCount":len(radish_rows),
            "pairedCommonPositiveActualCount":len(expected_keys),"fitCount":len(fit_map),
            "fitTransmission":transmission,"eigenCurves":curves,"evaluationReplay":replay,
            "evaluationDirections":directions,"priceMovement":movement}


def run_diagnostic(output_dir: Path = OUT) -> dict[str, Any]:
    """Verify and diagnose the frozen V/T artifacts, with no fitting or rescoring."""
    protocol=_load_protocol(output_dir)
    claim_path=output_dir/"claim.json";terminal_path=output_dir/"terminal.json"
    if terminal_path.exists():
        raise FileExistsError("diagnostic attempt already has an immutable terminal")
    attempt_id=str(uuid.uuid4())
    claim={"schemaVersion":1,"attemptId":attempt_id,"protocolSha256":protocol["protocolSha256"],
           "status":"running","startedAtKst":dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()}
    _write_exclusive(claim_path,canonical(claim))
    try:
        raw_prices=_json(rhs.PRICES_PATH)
        phase_docs={phase:_phase_diagnostic(phase,raw_prices) for phase in ("V","T")}
        all_transmission=[r for d in phase_docs.values() for r in d["fitTransmission"]]
        all_curves=[r for d in phase_docs.values() for r in d["eigenCurves"]]
        all_replay=[r for d in phase_docs.values() for r in d["evaluationReplay"]]
        all_directions=[r for d in phase_docs.values() for r in d["evaluationDirections"]]
        if len(all_transmission)!=180 or len(all_curves)!=7020 or len(all_replay)!=(755+790)*4 or len(all_directions)!=180:
            raise ValueError("diagnostic fit/horizon/direction coverage is incomplete")
        rank_rows=_rank_summary(all_curves)
        figures=_plot_curves(all_curves,output_dir)
        artifacts={
            "fit-transmission.csv":_csv_bytes(all_transmission),
            "eigen-curves.csv":_csv_bytes(all_curves),
            "rank-checkpoints.csv":_csv_bytes(rank_rows),
            "evaluation-replay.csv":_csv_bytes(all_replay),
            "evaluation-direction-energy.csv":_csv_bytes(all_directions),
        }
        for phase,doc in phase_docs.items():
            artifacts[f"{phase.lower()}-price-movement-keys.csv"]=_csv_bytes(doc["priceMovement"].get("keyRows",[]))
            artifacts[f"{phase.lower()}-price-movement-aggregates.csv"]=_csv_bytes(doc["priceMovement"].get("aggregates",[]))
        written=[]
        for name,payload in artifacts.items():
            path=output_dir/name
            _write_exclusive(path,payload)
            written.append({"file":name,"sha256":sha256_bytes(payload),"bytes":len(payload)})
        summary={"schemaVersion":1,"protocolSha256":protocol["protocolSha256"],"attemptId":attempt_id,
                 "phaseSummary":{phase:{"issued":doc["issueCount"],"paired":doc["pairedCommonPositiveActualCount"],
                                        "fits":doc["fitCount"],"fitTransmissionRows":len(doc["fitTransmission"]),
                                        "eigenCurveRows":len(doc["eigenCurves"]),"evaluationRows":len(doc["evaluationReplay"]),
                                        "evaluationDirectionRows":len(doc["evaluationDirections"]),
                                        "priceMovementAggregates":doc["priceMovement"].get("aggregates",[])}
                                for phase,doc in phase_docs.items()},
                 "diagnosticDefinitions":protocol["protocol"]["geometry"],"plots":figures,"artifacts":written,
                 "claimSha256":file_sha(claim_path),"completedAtKst":dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat(),
                 "interpretationLimits":["retrospective diagnostic on the existing V/T samples; not a new validation",
                                         "eigen-direction rank contributions are basis-dependent within repeated or near-repeated eigenspaces",
                                         "training/evaluation energy is geometric movement, not estimation variance or causal noise reduction",
                                         "paired evaluation and all-issued movement are separate supports"]}
        summary_path=output_dir/"diagnostic-summary.json"
        _write_exclusive(summary_path,canonical(summary))
        terminal={"schemaVersion":1,"attemptId":attempt_id,"protocolSha256":protocol["protocolSha256"],
                  "status":"complete","summarySha256":file_sha(summary_path),"artifactsSha256":sha256_bytes(canonical(written)),
                  "finishedAtKst":dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()}
        _write_exclusive(terminal_path,canonical(terminal))
        return summary
    except Exception as exc:
        terminal={"schemaVersion":1,"attemptId":attempt_id,"protocolSha256":protocol["protocolSha256"],
                  "status":"failed","errorType":type(exc).__name__,"reason":str(exc),
                  "finishedAtKst":dt.datetime.now(dt.timezone(dt.timedelta(hours=9))).isoformat()}
        _write_exclusive(terminal_path,canonical(terminal))
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("freeze","run"))
    parser.add_argument("--output-dir",type=Path,default=OUT)
    args=parser.parse_args(argv)
    if args.command=="freeze":
        doc=freeze_protocol(args.output_dir)
        result={"status":"frozen","protocolSha256":doc["protocolSha256"],"path":str(args.output_dir/"protocol.json")}
    else:
        summary=run_diagnostic(args.output_dir)
        phase_summary={phase:{key:value for key,value in metrics.items()
                              if key in {"issued","paired","fits","fitTransmissionRows","eigenCurveRows","evaluationRows","evaluationDirectionRows"}}
                       for phase,metrics in summary["phaseSummary"].items()}
        result={"status":"complete","protocolSha256":summary["protocolSha256"],"path":str(args.output_dir/"diagnostic-summary.json"),
                "phaseSummary":phase_summary}
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
