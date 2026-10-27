import ast
import datetime as dt
import hashlib
import math
from pathlib import Path

import numpy as np
import pytest

from agri_data import e106_training_diagnostic as diagnostic


def _zero_fit():
    return {
        "coefShapes": [[106, 16], [16, 1]],
        "scalerMean": [0.0] * 106,
        "scalerScale": [1.0] * 106,
        "coefs": [[0.0] * (106 * 16), [0.0] * 16],
        "intercepts": [[0.0] * 16, [0.25]],
    }


def test_forward_pass_uses_saved_weights_without_estimators():
    fit = _zero_fit()
    fit["coefs"][0][0] = 1.0
    fit["coefs"][1][0] = 2.0
    fit["intercepts"][0][0] = 0.5
    fit["intercepts"][1][0] = -0.25

    vector = [1.0] + [0.0] * 105
    result = diagnostic._forward(fit, vector)
    assert result == pytest.approx(2.75)
    standalone, _ = diagnostic.experiment._replay_prediction(fit, vector, 1.0, 1.0)
    assert result == pytest.approx(math.log(standalone))


def test_forward_rejects_wrong_saved_architecture():
    with pytest.raises(ValueError, match="architecture"):
        diagnostic._forward({**_zero_fit(), "coefShapes": [[1, 16], [16, 1]]}, [0.0] * 106)


def test_losses_report_q_and_price_metrics():
    rows = [
        {"r": 1.0, "rhat": 0.5, "p0": 10.0, "actual": 10.0 * np.exp(1.0)},
        {"r": -1.0, "rhat": -0.5, "p0": 10.0, "actual": 10.0 * np.exp(-1.0)},
    ]

    result = diagnostic._losses(rows)

    assert result["n"] == 2
    assert result["L_MLP"] == pytest.approx(0.25)
    assert result["L_p0"] == pytest.approx(1.0)
    assert result["Q_train"] == pytest.approx(0.25)
    assert result["Q_trainUnavailableReason"] is None
    assert result["trainingPriceMae"] > 0
    assert result["trainingPriceMapePercent"] > 0


def test_losses_mark_zero_baseline_loss_unavailable():
    result = diagnostic._losses([{"r": 0.0, "rhat": 0.1, "p0": 10.0, "actual": 10.0}])

    assert result["L_p0"] == 0.0
    assert result["Q_train"] is None
    assert result["Q_trainUnavailableReason"] == "L_p0_zero"


def test_scaler_reconstruction_matches_population_state_without_fit():
    matrix = np.array([[1.0, 4.0], [3.0, 4.0], [5.0, 4.0]])

    mean, scale, variance = diagnostic._scaler_state_without_fit(matrix)

    np.testing.assert_array_equal(mean, [3.0, 4.0])
    np.testing.assert_array_equal(scale, [np.sqrt(8.0 / 3.0), 1.0])
    np.testing.assert_array_equal(variance, [8.0 / 3.0, 0.0])


def test_module_has_no_estimator_fit_calls():
    source = Path(diagnostic.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]

    assert not any(isinstance(node.func, ast.Attribute) and node.func.attr == "fit" for node in calls)


def _synthetic_fit(fit_id, month, horizon):
    days = [dt.date(2020, 1, 1), dt.date(2020, 1, 2)]
    feature_rows = {}
    series = {}
    targets = []
    for i, day in enumerate(days):
        vector = [float(i + j) / 100.0 for j in range(106)]
        p0 = 10.0
        target = 0.1 * (i + 1)
        feature_rows[day] = {"preserved106": vector, "p0": p0}
        series[day] = (p0 * math.exp(target), 1.0)
        targets.append(day)

    x = np.asarray([feature_rows[d]["preserved106"] for d in targets], dtype=np.float64)
    y = np.asarray([math.log(series[d][0] / feature_rows[d]["p0"]) for d in targets], dtype=np.float64)
    mean, scale, variance = diagnostic._scaler_state_without_fit(x)
    fit = _zero_fit()
    fit.update({
        "fitId": fit_id, "month": month, "horizon": horizon,
        "training": {"fitOrigin": f"{month}-01"},
        "trainingRows": len(targets), "trainingKeysSha256": diagnostic._sha(
            diagnostic.experiment.canonical([d.isoformat() for d in targets])),
        "trainingMatrixSha256": hashlib.sha256(x.tobytes(order="C")).hexdigest(),
        "trainingTargetSha256": hashlib.sha256(y.tobytes(order="C")).hexdigest(),
        "scalerMean": mean.tolist(), "scalerScale": scale.tolist(),
        "scalerFingerprintSha256": hashlib.sha256(mean.tobytes() + scale.tobytes() + variance.tobytes()).hexdigest(),
    })
    state = {key: fit[key] for key in ("scalerMean", "scalerScale", "coefs", "coefShapes", "intercepts")}
    fit["inferenceStateSha256"] = diagnostic._sha(diagnostic.experiment.canonical(state))
    return fit, targets, feature_rows, series


def test_30_fit_diagnostics_use_saved_fit_ids_and_cross_month_targets():
    forecasts = []
    actuals = {}
    fit_specs = []
    for month_number, month in enumerate(("2026-04", "2026-05", "2026-06"), start=4):
        for horizon in range(1, 11):
            fit_id = f"garlic-h{horizon}-2026-{month_number:02d}"
            fit, targets, features, series = _synthetic_fit(fit_id, month, horizon)
            fit_specs.append((fit, targets, features, series))
            target_day = dt.date(2026, month_number, 28) + dt.timedelta(days=horizon)
            actual = 20.0 + horizon
            actuals[target_day] = (actual, 1.0)
            forecasts.append({"product": "garlic", "horizon": horizon, "fitId": fit_id,
                              "target": target_day.isoformat(), "mlpStandalonePrediction": actual + 1.0,
                              "prediction": actual + 2.0})

    diagnostics = [diagnostic._diagnose_fit(fit, targets, features, series, forecasts, actuals)
                   for fit, targets, features, series in fit_specs]

    assert len(diagnostics) == 30
    april_h10 = next(row for row in diagnostics if row["fitId"] == "garlic-h10-2026-04")
    assert april_h10["development1AssignedEvaluation"]["assignedRowCount"] == 1
    assert april_h10["development1AssignedEvaluation"]["assignedTargetKeys"] == ["2026-05-08"]
    assert april_h10["development1AssignedEvaluation"]["mlpStandalone"]["mae"] == pytest.approx(1.0)
    assert april_h10["development1AssignedEvaluation"]["savedMLP16Policy"]["mae"] == pytest.approx(2.0)


def test_assigned_evaluation_uses_date_keyed_actuals_and_fit_identity():
    target = dt.date(2026, 5, 2)
    rows = [
        {"product": "garlic", "horizon": 2, "fitId": "assigned", "target": target.isoformat(),
         "mlpStandalonePrediction": 12.0, "prediction": 13.0},
        {"product": "garlic", "horizon": 2, "fitId": "other", "target": target.isoformat(),
         "mlpStandalonePrediction": 1000.0, "prediction": 1000.0},
    ]

    result = diagnostic._assigned_evaluation("assigned", 2, rows, {target: (10.0, 1.0)})

    assert result["assignedRowCount"] == 1
    assert result["mlpStandalone"]["targetKeys"] == [target.isoformat()]
    assert result["mlpStandalone"]["mae"] == 2.0


def test_diagnostic_fit_rejects_malformed_saved_hashes():
    fit, targets, features, series = _synthetic_fit("fit", "2026-04", 1)
    fit["trainingMatrixSha256"] = "bad"

    with pytest.raises(ValueError, match="training feature matrix fingerprint mismatch"):
        diagnostic._diagnose_fit(fit, targets, features, series, [], {})


def test_frozen_input_validation_fails_closed_when_transitive_pin_fails(monkeypatch):
    def reject_pins():
        raise ValueError("pinned feature helper SHA-256 mismatch")

    monkeypatch.setattr(diagnostic.experiment, "_check_pinned_inputs", reject_pins)
    with pytest.raises(ValueError, match="pinned feature helper SHA-256 mismatch"):
        diagnostic._verify_frozen_inputs()
