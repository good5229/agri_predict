import numpy as np
import csv
import pytest
from sklearn.preprocessing import StandardScaler

from agri_data import radish_horizon_sharing as rhs


def _synthetic():
    rng = np.random.default_rng(7)
    x_by_h, y_by_h, scales = [], [], []
    for h in rhs.HORIZONS:
        raw = rng.normal(size=(100 + h, 40))
        raw[:, 1] = h
        raw[:, 38] = 2.0  # retain another constant column as scaler scale=1
        target = 0.2 * raw[:, 0] - 0.1 * raw[:, 3] + rng.normal(scale=0.2, size=len(raw))
        scaler = StandardScaler().fit(raw)
        x_by_h.append(scaler.transform(raw))
        y_by_h.append(target)
        scales.append(scaler.scale_)
    return x_by_h, y_by_h, scales


def test_graph_paths_and_diagonal_penalty_have_expected_structure():
    p = rhs.graph_penalties(np.ones((10, 39)))
    assert np.diag(rhs.PATH_C).tolist() == [1, 2, 2, 2, 2, 2, 2, 2, 2, 1]
    assert sorted(np.diag(rhs.PATH_P).tolist()) == [1, 1] + [2] * 8
    assert np.allclose(rhs.PATH_C, rhs.PATH_P) is False
    assert np.allclose(p["D"], np.diag(np.diag(p["C"])))
    assert rhs._check_psd(p["C"], name="C")["minimumEigenvalue"] >= -1e-10


def test_lambda_solver_matches_target_and_requires_positive_intervention():
    g = np.eye(rhs.N_COEF, dtype=float)
    q0 = g + 10 * np.eye(rhs.N_COEF)
    penalty = np.eye(rhs.N_COEF)
    df0 = rhs._edf(g, q0)
    target = 0.9 * df0
    result = rhs.solve_edf_lambda(g, q0, penalty, target, df0=df0, df_inf=0.0,
                                  user_tolerance=rhs.EDF_USER_TOL * max(1, df0))
    assert result["status"] == "ok"
    assert result["lambda"] > 0
    assert abs(result["actualDf"] - target) <= result["edfTolerance"]
    blocked = rhs.solve_edf_lambda(g, q0, np.zeros_like(g), target, df0=df0, df_inf=0.0,
                                   user_tolerance=rhs.EDF_USER_TOL * max(1, df0))
    assert blocked["status"] == "numerical_failure"


def test_joint_fit_replays_saved_coefficients_and_keeps_horizon_constant_zero():
    x, y, scale = _synthetic()
    fit = rhs.fit_joint(x, y, scale, fit_id="synthetic-2026-01")
    assert fit["status"] == "ok"
    assert fit["horizonDaysCoefficient"] == 0.0
    assert len(fit["fitCoefficients"]["C"]) == 10
    assert all(len(row) == 39 for row in fit["fitCoefficients"]["C"])
    assert fit["lambdas"]["C"] > 0 and fit["lambdas"]["D"] > 0 and fit["lambdas"]["P"] > 0
    assert max(fit["diagnostics"][name]["relativeSolveResidual"] for name in rhs.CONDITIONS) <= rhs.SOLVE_REL_TOL
    predicted = rhs.predict_saved(fit, x)
    assert set(predicted) == set(rhs.CONDITIONS)
    assert all(len(v) == sum(len(z) for z in x) for v in predicted.values())
    assert np.allclose(predicted["B"], np.concatenate(fit["trainingPredictions"]["B"]), atol=1e-11)


def test_fit_requires_40_column_preserving_schema_and_minimum_rows():
    x, y, scale = _synthetic()
    with pytest.raises(ValueError, match="40-column"):
        rhs.fit_joint([z[:, :39] for z in x], y, scale, fit_id="bad")
    y_bad = [z[:89] for z in y]
    x_bad = [z[:89] for z in x]
    with pytest.raises(ValueError, match="fewer than 90"):
        rhs.fit_joint(x_bad, y_bad, scale, fit_id="short")


def test_fit_checkpoint_reuses_completed_result_and_rejects_changed_inputs(tmp_path):
    calls = []
    def fit_factory():
        calls.append("fit")
        return {"status": "ok", "value": [1.0, 2.0]}
    hashes = {"trainingKeys": "keys", "trainingMatrix": "matrix", "trainingTarget": "target"}
    first = rhs._fit_checkpoint(tmp_path, "V", "radish-2026-04", hashes, fit_factory)
    second = rhs._fit_checkpoint(tmp_path, "V", "radish-2026-04", hashes, fit_factory)
    assert first == second
    assert calls == ["fit"]
    with pytest.raises(ValueError, match="input binding"):
        rhs._fit_checkpoint(tmp_path, "V", "radish-2026-04", {**hashes, "trainingTarget": "changed"}, fit_factory)


def test_fit_checkpoint_preserves_distinct_failure_terminal(tmp_path):
    calls = []
    hashes = {"trainingKeys": "keys", "trainingMatrix": "matrix", "trainingTarget": "target"}

    def fit_factory():
        calls.append("fit")
        return {"status": "intervention_not_identifiable", "reason": "edf_gap_below_tolerance"}

    fit = rhs._fit_checkpoint(tmp_path, "V", "radish-2026-04", hashes, fit_factory)
    terminal = rhs._json(tmp_path / "fit-checkpoints" / "v-radish-2026-04.terminal.json")
    reused = rhs._fit_checkpoint(tmp_path, "V", "radish-2026-04", hashes, fit_factory)
    assert fit == reused
    assert calls == ["fit"]
    assert terminal["status"] == "intervention_not_identifiable"


def test_frozen_protocol_pins_scoring_and_inference_helpers(monkeypatch):
    from agri_data import equal_info_online_score, radish_horizon_sharing_score
    monkeypatch.setattr(rhs.platform, "python_version", lambda: "3.11.4")
    protocol = rhs._source_protocol()
    assert protocol["inputSha256"]["scorer"] == rhs.file_sha(rhs.Path(radish_horizon_sharing_score.__file__))
    assert protocol["inputSha256"]["inferenceHelper"] == rhs.file_sha(rhs.Path(equal_info_online_score.__file__))




def test_synthetic_build_verify_score_and_t_gate_are_bound_and_source_free(tmp_path, monkeypatch):
    import datetime as dt
    import json
    import numpy as np
    from agri_data import equal_info_online_forecasts as base
    from agri_data import full_history_return_eval as narrow
    from agri_data import full_history_xgb_rf_eval as rich

    phase = "V"
    start, end, origin_start = rhs.PHASES[phase]
    train_start = dt.date(2025, 12, 30)
    monkeypatch.setattr(narrow, "TRAIN_START", train_start)
    days = []
    d = train_start - dt.timedelta(days=20)
    while d <= end:
        days.append(d)
        d += dt.timedelta(days=1)
    series = {d: (100.0, 100.0) for d in days}
    all_series = {p: dict(series) for p in ("cabbage", "radish", "onion", "garlic")}

    def feature_row(_series, cutoff, target, h, _cache):
        seed = target.toordinal() * 37 + h * 991
        x = np.random.default_rng(seed).normal(size=40)
        x[1] = h
        return cutoff, x.tolist()

    def direct_feature(_series, target, h, cache):
        origin = target - dt.timedelta(days=h)
        cutoff = origin - dt.timedelta(days=1)
        _, vector = feature_row(_series, cutoff, target, h, cache)
        return vector, 100.0, {"cutoff": cutoff.isoformat(), "p0Date": cutoff.isoformat(), "p0AgeDays": 1}

    monkeypatch.setattr(base.modeling, "_feature_row", feature_row)
    monkeypatch.setattr(base, "_direct_feature", direct_feature)
    monkeypatch.setattr(base.direct_horizon, "_observations", lambda _prices: all_series)
    monkeypatch.setattr(rich, "_raw_maps", lambda _prices: ({}, {}))

    baseline_path = tmp_path / "baseline.json"
    prices_path = tmp_path / "prices.json"
    prices_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(rhs, "BASELINE_PATH", baseline_path)
    monkeypatch.setattr(rhs, "PRICES_PATH", prices_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    input_pins = {"baseline": "synthetic", "prices": "synthetic", "composition": "synthetic"}
    runtime = {"python": "synthetic", "numpy": "synthetic", "scikitLearn": "synthetic", "statsmodels": "synthetic"}
    protocol = {"protocolSha256": "synthetic-protocol", "protocol": {"inputSha256": input_pins, "runtime": runtime}}
    monkeypatch.setattr(rhs, "_load_protocol", lambda _out: protocol)
    monkeypatch.setattr(rhs, "_preflight", lambda *_args: {"phase": "V", "synthetic": True})

    baseline_rows = []
    composition = {}
    actuals = {(p, day.isoformat()): 100.0 for p, daily in all_series.items() for day in daily}
    for offset in range((end-start).days + 1):
        target = start + dt.timedelta(days=offset)
        for h in rhs.HORIZONS:
            origin = target - dt.timedelta(days=h)
            fit_origin = origin.replace(day=1)
            eligible = origin >= origin_start
            for product in ("cabbage", "radish", "onion", "garlic"):
                key = (product, origin.isoformat(), target.isoformat(), h)
                training = {"count": max(0, (fit_origin-train_start).days),
                            "firstTarget": train_start.isoformat(), "lastTarget": (fit_origin-dt.timedelta(days=1)).isoformat(),
                            "fitOrigin": fit_origin.isoformat(), "alpha": 10.0, "featureCount": 40}
                model = lambda prediction, provenance=None, training=None: {
                    "status": "issued", "prediction": prediction, "reason": None,
                    "provenance": provenance or {"fallback": False}, "training": training}
                direct_model = model(100.0, {"featureAudit": {"cutoff": (origin-dt.timedelta(days=1)).isoformat() }}, training)
                row = {"product": product, "origin": origin.isoformat(), "target": target.isoformat(), "horizon": h,
                       "eligible": eligible, "models": {"causalDirect": direct_model,
                       "causalSarima": model(100.0), "causal50": model(100.0)}}
                baseline_rows.append(row)
                if eligible:
                    candidate_prediction = 101.0 if product == "cabbage" else 102.0 if product == "garlic" else 100.0
                    candidate_source = "preserved106Blend" if product == "garlic" else "E"
                    composition[key] = {"candidateSource": candidate_source,
                                        "candidate": model(candidate_prediction), "eligible": True}
    baseline_path.write_bytes(rhs.canonical({"schemaVersion": 1,
        "protocol": {"id": "equal-info-online-20261010-v1"}, "evaluationForecasts": baseline_rows}))
    monkeypatch.setattr(rhs, "_composition_index", lambda _phase: composition)
    monkeypatch.setattr(rhs, "_actuals_for_score", lambda: actuals)

    fit_calls = []
    original_fit_joint = rhs.fit_joint
    def counted_fit(*args, **kwargs):
        fit_calls.append(kwargs["fit_id"])
        return original_fit_joint(*args, **kwargs)
    monkeypatch.setattr(rhs, "fit_joint", counted_fit)

    manifest = rhs.build_phase("V", output_dir)
    assert len(fit_calls) == 3
    assert manifest["modelFitCounts"]["jointMonthlySystems"] == 3
    assert len(manifest["forecasts"]) == 3240
    assert len(list(csv.DictReader((output_dir / "validation-predictions.csv").open()))) == 4 * len(manifest["forecasts"])
    cabbage_row = next(r for r in manifest["forecasts"] if r["product"] == "cabbage" and r["eligible"])
    garlic_row = next(r for r in manifest["forecasts"] if r["product"] == "garlic" and r["eligible"])
    assert cabbage_row["conditions"]["B"]["blendPrediction"] == 101.0
    assert garlic_row["conditions"]["B"]["blendPrediction"] == 102.0
    verification = rhs.verify_phase("V", output_dir)
    assert verification["status"] == "passed"
    assert rhs.build_phase("V", output_dir)["protocolSha256"] == protocol["protocolSha256"]
    assert len(fit_calls) == 3  # completed checkpoint reuse does not fit again

    # A crash after manifest persistence is recovered only after fit-free verification.
    terminal_path = output_dir / "validation-terminal.json"
    saved_terminal = terminal_path.read_bytes()
    terminal_path.unlink()
    def forbidden_refit(*_args, **_kwargs):
        raise AssertionError("recovery must not fit")
    monkeypatch.setattr(rhs, "fit_joint", forbidden_refit)
    assert rhs.build_phase("V", output_dir)["protocolSha256"] == protocol["protocolSha256"]
    assert rhs._json(terminal_path)["status"] == "complete"
    terminal_path.write_bytes(saved_terminal)
    monkeypatch.setattr(rhs, "fit_joint", counted_fit)

    # Raised numerical errors are checkpointed and terminate the phase explicitly.
    failed_dir = tmp_path / "numerical-failure"
    failed_dir.mkdir()
    def raised_fit(*_args, **_kwargs):
        raise ArithmeticError("synthetic Cholesky failure")
    monkeypatch.setattr(rhs, "fit_joint", raised_fit)
    with pytest.raises(ArithmeticError, match="numerical_failure"):
        rhs.build_phase("V", failed_dir)
    failed_fit = rhs._json(failed_dir / "fit-checkpoints" / "v-radish-2026-04-01.json")
    failed_fit_terminal = rhs._json(failed_dir / "fit-checkpoints" / "v-radish-2026-04-01.terminal.json")
    phase_terminal = rhs._json(failed_dir / "validation-terminal.json")
    assert failed_fit["fit"]["status"] == failed_fit_terminal["status"] == "numerical_failure"
    assert phase_terminal["status"] == "numerical_failure"
    monkeypatch.setattr(rhs, "fit_joint", counted_fit)

    report = rhs._score_phase("V", output_dir)
    assert report["primaryGateEvidence"]["passed"] is False
    report_path = output_dir / "validation-report.json"
    original_report = report_path.read_bytes()
    assert rhs._score_phase("V", output_dir) == report  # verified report is idempotent on resume
    tampered = json.loads(original_report)
    tampered["primaryGateEvidence"]["passed"] = True
    report_path.write_bytes(rhs.canonical(tampered))
    with pytest.raises(ValueError, match="report does not replay"):
        rhs._test_gate_passed(output_dir)
    report_path.write_bytes(original_report)

    manifest_path = output_dir / "validation-manifest.json"
    csv_path = output_dir / "validation-predictions.csv"
    original_manifest = manifest_path.read_bytes(); original_csv = csv_path.read_bytes()
    changed = json.loads(original_manifest)
    row = next(r for r in changed["forecasts"] if r["product"] == "radish" and r["eligible"])
    row["featureVector"][0] += 1.0
    changed_csv = rhs._csv_bytes(changed["forecasts"])
    changed["forecastFileSha256"] = rhs.sha256_bytes(changed_csv)
    manifest_path.write_bytes(rhs.canonical(changed)); csv_path.write_bytes(changed_csv)
    terminal_path = output_dir / "validation-terminal.json"
    original_terminal = terminal_path.read_bytes()
    terminal = json.loads(original_terminal)
    terminal["manifestSha256"] = rhs.file_sha(manifest_path)
    terminal_path.write_bytes(rhs.canonical(terminal))
    with pytest.raises(ValueError, match="feature vector differs"):
        rhs.verify_phase("V", output_dir)
    manifest_path.write_bytes(original_manifest); csv_path.write_bytes(original_csv); terminal_path.write_bytes(original_terminal)

    changed = json.loads(original_manifest)
    row = next(r for r in changed["forecasts"] if r["product"] == "radish" and r["eligible"])
    row["conditions"]["C"]["blendPrediction"] += 1.0
    changed_csv = rhs._csv_bytes(changed["forecasts"])
    changed["forecastFileSha256"] = rhs.sha256_bytes(changed_csv)
    manifest_path.write_bytes(rhs.canonical(changed)); csv_path.write_bytes(changed_csv)
    terminal = json.loads(original_terminal); terminal["manifestSha256"] = rhs.file_sha(manifest_path)
    terminal_path.write_bytes(rhs.canonical(terminal))
    with pytest.raises(ValueError, match="same-SARIMA blend/status"):
        rhs.verify_phase("V", output_dir)
    manifest_path.write_bytes(original_manifest); csv_path.write_bytes(original_csv); terminal_path.write_bytes(original_terminal)

    marker = rhs.build_phase("T", output_dir)
    assert marker["status"] == "not_run_validation_failed"
    assert rhs._json(output_dir / "test-report.json")["fits"] == 0
