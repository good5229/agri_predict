"""One-time, no-refit in-sample diagnostic for saved confirmation MLP16 fits."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import e106_mlp16 as experiment

ROOT = Path(__file__).resolve().parents[1]
OUT = experiment.OUT / "training-diagnostic.json"
MANIFEST = experiment.OUT / "confirmation-manifest.json"
REPORT = experiment.OUT / "confirmation-report.json"
MANIFEST_SHA256 = "174f76e25c79e9fff2973dbc42f6cd869742222d7ddd93e6483451bb9e0607bf"
REPORT_SHA256 = "4778f0a8555d4971deaca6385dcc76ce37032cc98ffae038f1bb6ca7b10883f9"
MANIFEST_DIGEST = "b5e850da3956e1e8deb119c92fcd2e297ef77f47b45d026d5d3bfd9c88e50fd9"
EXPERIMENT_SOURCE_SHA256 = "5855f4c40b0f3ef60c24e9ba48ee7fe985ea5eaf25d3aea147690645e79c7f6f"
PRICES_SHA256 = experiment.PINNED_SHA256["prices"]
R106_SHA256 = experiment.PINNED_SHA256["r106"]
FEATURE_HELPER_SHA256 = experiment.PINNED_SHA256["preservedFeatureHelper"]
PRODUCT = "garlic"
SCHEMA = "e106-mlp16-training-diagnostic-v1"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _finite_float(value: Any) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("nonfinite diagnostic input")
    return result


def _scaler_state_without_fit(matrix):
    """Reconstruct sklearn 1.9.1 StandardScaler dense state without calling fit."""
    import numpy as np
    from sklearn.utils.extmath import _incremental_mean_and_var
    from sklearn.preprocessing._data import _handle_zeros_in_scale, _is_constant_feature

    n_features = matrix.shape[1]
    mean, variance, count = _incremental_mean_and_var(
        matrix, 0.0, 0.0, np.zeros(n_features, dtype=np.int64)
    )
    constant = _is_constant_feature(variance, mean, count)
    scale = _handle_zeros_in_scale(np.sqrt(variance), copy=False, constant_mask=constant)
    return (np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64),
            np.asarray(variance, dtype=np.float64))


def _forward(fit: dict[str, Any], vector: list[float]) -> float:
    """Pure 106→16 ReLU→1 forward pass; no estimator construction or fitting."""
    if len(vector) != 106 or fit.get("coefShapes") != [[106, 16], [16, 1]]:
        raise ValueError("saved MLP architecture does not match MLP16")
    means, scales = fit["scalerMean"], fit["scalerScale"]
    z = [(_finite_float(v) - means[i]) / scales[i] for i, v in enumerate(vector)]
    w0, w1 = fit["coefs"]
    hidden = [max(0.0, math.fsum(z[i] * w0[i * 16 + j] for i in range(106))
                  + fit["intercepts"][0][j]) for j in range(16)]
    return math.fsum(hidden[j] * w1[j] for j in range(16)) + fit["intercepts"][1][0]


def _losses(rows: list[dict[str, float]]) -> dict[str, Any]:
    """Pure fixture-friendly loss calculation; each row has r, rhat, p0, actual."""
    if not rows:
        raise ValueError("empty training rows")
    mlp_sq = [(_finite_float(r["r"]) - _finite_float(r["rhat"])) ** 2 for r in rows]
    p0_sq = [_finite_float(r["r"]) ** 2 for r in rows]
    l_mlp = math.fsum(mlp_sq) / len(rows)
    l_p0 = math.fsum(p0_sq) / len(rows)
    ratio = l_mlp / l_p0 if l_p0 > 0 else None
    actual = [_finite_float(r["actual"]) for r in rows]
    predicted = [_finite_float(r["p0"]) * math.exp(_finite_float(r["rhat"])) for r in rows]
    mae = math.fsum(abs(p - y) for p, y in zip(predicted, actual)) / len(rows)
    mape = 100.0 * math.fsum(abs(p - y) / y for p, y in zip(predicted, actual)) / len(rows)
    return {"n": len(rows), "L_MLP": l_mlp, "L_p0": l_p0, "Q_train": ratio,
            "Q_trainUnavailableReason": "L_p0_zero" if l_p0 == 0 else None,
            "trainingPriceMae": mae, "trainingPriceMapePercent": mape}


def _actual_price(value: Any) -> float:
    """Read the price component returned by the pinned direct-horizon source."""
    if isinstance(value, (list, tuple)):
        value = value[0]
    return _finite_float(value)


def _price_metrics(rows: list[dict[str, Any]], prediction_field: str) -> dict[str, Any]:
    pairs = []
    for row in rows:
        prediction = row.get(prediction_field)
        if prediction is None:
            continue
        prediction = _finite_float(prediction)
        actual = _finite_float(row["actual"])
        if actual <= 0:
            raise ValueError("development1 actual price must be positive for MAPE")
        pairs.append((row["target"], prediction, actual))
    if not pairs:
        return {"n": 0, "targetKeys": [], "mae": None, "mapePercent": None}
    errors = [abs(prediction - actual) for _, prediction, actual in pairs]
    return {"n": len(pairs), "targetKeys": [target for target, _, _ in pairs],
            "mae": math.fsum(errors) / len(errors),
            "mapePercent": 100.0 * math.fsum(error / actual for error, (_, _, actual) in zip(errors, pairs)) / len(errors)}


def _assigned_evaluation(fit_id: str, horizon: int, forecast_rows: list[dict[str, Any]],
                         actuals: dict[dt.date, Any]) -> dict[str, Any]:
    assigned = []
    for row in forecast_rows:
        if row.get("product") != PRODUCT or row.get("horizon") != horizon or row.get("fitId") != fit_id:
            continue
        target = dt.date.fromisoformat(row["target"])
        if target not in actuals:
            continue
        assigned.append({"target": target.isoformat(), "actual": _actual_price(actuals[target]),
                         "mlpStandalonePrediction": row.get("mlpStandalonePrediction"),
                         "prediction": row.get("prediction")})
    assigned.sort(key=lambda row: row["target"])
    return {"assignedRowCount": len(assigned), "assignedTargetKeys": [row["target"] for row in assigned],
            "mlpStandalone": _price_metrics(assigned, "mlpStandalonePrediction"),
            "savedMLP16Policy": _price_metrics(assigned, "prediction")}


def _diagnose_fit(fit: dict[str, Any], train_targets: list[dt.date], feature_rows: dict[dt.date, dict[str, Any]],
                  series: dict[dt.date, Any], forecast_rows: list[dict[str, Any]],
                  actuals: dict[dt.date, Any]) -> dict[str, Any]:
    """Validate frozen audit hashes, compute train loss, and score its assigned saved rows."""
    import numpy as np
    fit_id, h = fit["fitId"], int(fit["horizon"])
    if fit.get("fitFailure"):
        raise ValueError(f"saved fit failed; no trained state to diagnose: {fit_id}/{fit['fitFailure']}")
    if not train_targets:
        raise ValueError(f"empty reconstructed training rows: {fit_id}")
    keys_raw = experiment.canonical([d.isoformat() for d in train_targets])
    if _sha(keys_raw) != fit["trainingKeysSha256"]:
        raise ValueError(f"training key fingerprint mismatch: {fit_id}")
    x = np.asarray([feature_rows[d]["preserved106"] for d in train_targets], dtype=np.float64)
    y = np.asarray([math.log(series[d][0] / feature_rows[d]["p0"]) for d in train_targets], dtype=np.float64)
    if _sha(x.tobytes(order="C")) != fit["trainingMatrixSha256"]:
        raise ValueError(f"training feature matrix fingerprint mismatch: {fit_id}")
    if _sha(y.tobytes(order="C")) != fit["trainingTargetSha256"]:
        raise ValueError(f"training target fingerprint mismatch: {fit_id}")
    mean, scale, variance = _scaler_state_without_fit(x)
    scaler_fingerprint = _sha(mean.tobytes() + scale.tobytes() + variance.tobytes())
    if scaler_fingerprint != fit["scalerFingerprintSha256"]:
        raise ValueError(f"scaler fingerprint mismatch: {fit_id}")
    if not np.array_equal(mean, np.asarray(fit["scalerMean"])) or not np.array_equal(scale, np.asarray(fit["scalerScale"])):
        raise ValueError(f"saved scaler state mismatch: {fit_id}")
    state = {key: fit[key] for key in ("scalerMean", "scalerScale", "coefs", "coefShapes", "intercepts")}
    if fit.get("inferenceStateSha256") != _sha(experiment.canonical(state)):
        raise ValueError(f"saved inference state fingerprint mismatch: {fit_id}")
    training_rows = []
    for d, target in zip(train_targets, y.tolist()):
        fr = feature_rows[d]
        rhat = _forward(fit, fr["preserved106"])
        training_rows.append({"r": target, "rhat": rhat, "p0": fr["p0"], "actual": series[d][0]})
    losses = _losses(training_rows)
    assigned = _assigned_evaluation(fit_id, h, forecast_rows, actuals)
    return {"fitId": fit_id, "month": fit["month"], "horizon": h,
            "fitOrigin": fit["training"]["fitOrigin"], "trainingRows": len(train_targets),
            "uniqueOrigins": len({d - dt.timedelta(days=h) for d in train_targets}),
            "uniqueTargets": len(set(train_targets)), "parameters": 1729,
            "parametersPerTrainingRow": 1729 / len(train_targets), "losses": losses,
            "development1AssignedEvaluation": assigned}


def _verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    # This includes the complete transitive preserved-feature helper stack and
    # the answer-free E106/R106 lineage, in addition to the local file pins below.
    experiment._check_pinned_inputs()
    if _file_sha(MANIFEST) != MANIFEST_SHA256:
        raise ValueError("saved confirmation MLP manifest SHA-256 mismatch")
    if _file_sha(REPORT) != REPORT_SHA256:
        raise ValueError("saved confirmation score report SHA-256 mismatch")
    if _file_sha(Path(experiment.__file__)) != EXPERIMENT_SOURCE_SHA256:
        raise ValueError("saved MLP builder source SHA-256 mismatch")
    if (_file_sha(experiment.PRICES) != PRICES_SHA256 or _file_sha(experiment.R106) != R106_SHA256
            or _file_sha(Path(experiment.pf.__file__)) != FEATURE_HELPER_SHA256):
        raise ValueError("frozen training inputs or feature helper SHA-256 mismatch")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if manifest.get("manifestSha256") != MANIFEST_DIGEST or experiment.digest(manifest) != MANIFEST_DIGEST:
        raise ValueError("confirmation manifest internal digest mismatch")
    if report.get("phase") != "confirmation" or report.get("manifestFileSha256") != MANIFEST_SHA256:
        raise ValueError("score report is not linked to the pinned confirmation manifest")
    if manifest.get("protocol", {}).get("inputHashes", {}).get("experimentModule") != EXPERIMENT_SOURCE_SHA256:
        raise ValueError("confirmation manifest source lineage mismatch")
    if manifest.get("targetValuesAttached") is not False or len(manifest.get("fitAudits", [])) != 30:
        raise ValueError("confirmation manifest fit/grid schema mismatch")
    return manifest, report, json.loads(experiment.PRICES.read_text(encoding="utf-8"))


def run_diagnostic() -> dict[str, Any]:
    """Reconstruct the exact training rows and forward-pass saved weights once."""
    import sklearn
    if sklearn.__version__ != "1.9.1":
        raise RuntimeError(f"requires scikit-learn 1.9.1, found {sklearn.__version__}")
    manifest, report, prices = _verify_frozen_inputs()
    # Reuse the already completed saved-prediction/report hash linkage. Do not
    # rerun development evaluation, DM tests, bootstrap, or large-error analysis.
    base_index = experiment._idx(experiment._base_doc("confirmation")["forecasts"])
    r106 = experiment._r106_index()
    series, features, _ = experiment._make_features("confirmation", base_index, r106)
    fits = {x["fitId"]: x for x in manifest["fitAudits"]}
    if len(fits) != 30:
        raise ValueError("expected 30 saved MLP fit states")
    forecast_rows = manifest["forecasts"]
    actuals = experiment.pf.direct_horizon._observations(prices)[PRODUCT]
    result_rows = []
    for fit_id, fit in sorted(fits.items(), key=lambda item:(item[1]["month"],item[1]["horizon"])):
        h = int(fit["horizon"])
        fit_origin = dt.date.fromisoformat(fit["training"]["fitOrigin"])
        train_targets = [d for d in series if d >= experiment.pf.online.narrow.TRAIN_START
                         and d < fit_origin and features[h].get(d) is not None]
        if len(train_targets) != fit.get("trainingRows") or len(train_targets) != fit.get("training",{}).get("count"):
            raise ValueError(f"training row count mismatch: {fit_id}")
        row = _diagnose_fit(fit, train_targets, features[h], series, forecast_rows, actuals)
        row["fitOrigin"] = fit_origin.isoformat()
        row["savedReplayLink"] = {"manifestSha256": MANIFEST_SHA256,
                                   "scoreReportSha256": REPORT_SHA256,
                                   "scoreReportManifestSha256": report["manifestFileSha256"]}
        result_rows.append(row)
    q_values=[r["losses"]["Q_train"] for r in result_rows if r["losses"]["Q_train"] is not None]
    return {"schemaVersion":1,"kind":SCHEMA,"status":"complete","refitPerformed":False,
            "sourceLinkage":{"confirmationManifestFileSha256":MANIFEST_SHA256,"confirmationReportFileSha256":REPORT_SHA256,
                             "reportManifestFileSha256":report["manifestFileSha256"],
                             "savedPredictionReplay":"reused the pinned report produced by the pinned score path after verify_sources; no score or DM rerun"},
            "scikitLearnVersion":sklearn.__version__,"fits":result_rows,
            "summary":{"fitCount":len(result_rows),"qTrainBelowOne":sum(q<1 for q in q_values),
                       "qTrainAtOrAboveOne":sum(q>=1 for q in q_values),"qTrainUnavailable":sum(r["losses"]["Q_train"] is None for r in result_rows),
                       "interpretation":"In-sample dependent training statistics; not independent inference. Q_train does not establish convergence or overfitting, and this diagnostic does not reactivate the stopped MLP candidate."}}


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    run=sub.add_parser("run",help="reconstruct frozen training rows and forward-pass saved MLP weights once")
    run.add_argument("--output",type=Path,default=OUT)
    args=parser.parse_args(argv)
    if args.output.exists(): raise FileExistsError(f"immutable diagnostic output exists: {args.output}")
    doc=run_diagnostic()
    doc["diagnosticSha256"]=_sha(experiment.canonical(doc).rstrip(b"\n"))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("xb") as stream: stream.write(experiment.canonical(doc))
    print(f"wrote {args.output} sha256={_file_sha(args.output)}")
    return 0


if __name__=="__main__": raise SystemExit(main())
