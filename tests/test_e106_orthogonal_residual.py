import inspect
import datetime as dt
import json

import numpy as np
import pytest

from agri_data import e106_orthogonal_residual as s


def arrays(seed=4, n=190):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 106))
    w = rng.normal(scale=.08, size=(106, 16))
    b = rng.normal(scale=.1, size=16)
    z = np.maximum(0, x @ w + b)
    r_r = .2 + x[:, :4] @ np.array([.03, -.02, .01, .04])
    target = r_r + .15 * (z[:, 0] - np.mean(z[:, 0])) + rng.normal(scale=.03, size=n)
    return x, z, target, r_r


def test_fit_projects_hidden_features_and_improves_training_residual():
    x, z, target, r_r = arrays()
    fit = s.fit_residual(x, z, target, r_r)
    assert fit["A_svd"]["retainedRank"] <= 107
    assert fit["Zperp_svd"]["retainedRank"] <= 16
    assert fit["qU"] <= s.ORTHOGONALITY_TOL
    assert fit["orthogonalityAbsoluteFro"] <= fit["orthogonalityLimit"]
    assert fit["L_S"] <= fit["L_R"] + fit["trainingLossSlack"]
    np.testing.assert_allclose(
        np.asarray(fit["trainingPrediction"]),
        s.predict_residual(x, z, r_r, fit), rtol=1e-12, atol=1e-12,
    )


def test_rank_zero_residual_sets_exact_zero_correction():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(80, 106))
    z = np.ones((80, 16)) * np.arange(1, 17)
    r = .4 + .02 * x[:, 0] + rng.normal(scale=.1, size=80)
    r_r = .4 + .02 * x[:, 0]
    fit = s.fit_residual(x, z, r, r_r)
    assert fit["rankZperp"] == 0
    assert fit["zeroResidualRepresentation"] is True
    np.testing.assert_array_equal(fit["gamma"], np.zeros(16))
    np.testing.assert_array_equal(s.predict_residual(x, z, r_r, fit), r_r)


def test_training_state_verifies_from_saved_factors_without_svd(monkeypatch):
    x, z, target, r_r = arrays(seed=8, n=120)
    fit = s.fit_residual(x, z, target, r_r)
    monkeypatch.setattr(np.linalg, "svd", lambda *a, **k: pytest.fail("fit-free verifier recomputed SVD"))
    s.verify_fit_free(x, z, target, r_r, fit)


def test_fit_uses_exactly_two_svd_decompositions(monkeypatch):
    x, z, target, r_r = arrays(seed=10, n=120)
    original = np.linalg.svd
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", counted)
    s.fit_residual(x, z, target, r_r)
    assert len(calls) == 2


def test_fit_free_verifier_rejects_mutated_gamma():
    x, z, target, r_r = arrays(seed=13, n=120)
    fit = s.fit_residual(x, z, target, r_r)
    fit["gamma"][0] += .01
    with pytest.raises(ValueError, match="coefficients"):
        s.verify_fit_free(x, z, target, r_r, fit)


def test_fit_free_verifier_rejects_changed_cutoff():
    x, z, target, r_r = arrays(seed=16, n=120)
    fit = s.fit_residual(x, z, target, r_r)
    fit["tauZperp"] *= 2
    with pytest.raises(ValueError, match="cutoff"):
        s.verify_fit_free(x, z, target, r_r, fit)


def test_fit_free_verifier_binds_training_correction():
    x, z, target, r_r = arrays(seed=17, n=120)
    fit = s.fit_residual(x, z, target, r_r)
    fit["trainingCorrection"][0] += .01
    with pytest.raises(ValueError, match="training correction"):
        s.verify_fit_free(x, z, target, r_r, fit)


def test_validation_hidden_state_compares_reshaped_rmh_coefficients():
    source = {"mlpCoefs": [[1, 2, 3, 4], [5, 6]],
              "mlpCoefShapes": [[2, 2], [2, 1]]}
    assert s._rmh_coefs_in_s_shape(source) == [[[1.0, 2.0], [3.0, 4.0]], [[5.0], [6.0]]]
    source["mlpCoefShapes"] = [[3, 2], [2, 1]]
    assert s._rmh_coefs_in_s_shape(source) is None


def test_ridge_training_prediction_is_recomputed_from_saved_coefficients():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(120, 106))
    beta = rng.normal(size=107)
    y = beta[0] + x @ beta[1:]
    np.testing.assert_allclose(s._ridge_training_output(x, beta, y), y)
    y[0] += .01
    with pytest.raises(ValueError, match="does not replay"):
        s._ridge_training_output(x, beta, y)


def test_s_numerical_failure_is_reported_for_exact_fallback(monkeypatch):
    x, z, target, r_r = arrays(seed=21, n=120)
    fit = s.fit_residual(x, z, target, r_r)
    fit["mlpCoefs"] = [[[0.0]]]
    fit["mlpIntercepts"] = [[0.0]]

    def overflow(*args, **kwargs):
        raise OverflowError("synthetic S price overflow")

    monkeypatch.setattr(s, "direct_price", overflow)
    monkeypatch.setattr(s.rmh, "hidden_design", lambda *args, **kwargs: np.ones((1, 17)))
    # The helper performs only S conversion; R price remains the saved baseline input.
    result = s._s_forecast_candidate(x[:1], .2, 120.0, 100.0,
                                     {"status": "issued", "prediction": 140.0}, fit)
    assert result["failureReason"] == "S_invalid_or_component_unavailable:OverflowError"
    assert result["RBlendPrediction"] == pytest.approx(130.0)
    assert result["SDirectPrediction"] is None
    assert result["SBlendPrediction"] is None


def test_r_output_outside_retained_input_space_is_a_hard_numerical_stop():
    x, z, target, _ = arrays(seed=18, n=150)
    bad_r = np.random.default_rng(19).normal(size=len(target))
    with pytest.raises(ArithmeticError, match="R training output"):
        s.fit_residual(x, z, target, bad_r)


def test_residual_identity_is_exact_on_shared_rows():
    actual = np.array([.1, -.3, .2, .5])
    r = np.array([.2, -.2, .1, .3])
    s_pred = np.array([.18, -.4, .3, .45])
    result = s.residual_identity(actual, r, s_pred)
    assert result["identityError"] < 1e-15
    assert result["mseSMinusR"] == pytest.approx(
        result["correctionSquared"] + result["residualAlignmentOffset"]
    )


def test_invalid_alignment_or_nonfinite_values_are_rejected():
    x, z, target, r_r = arrays(seed=14, n=120)
    with pytest.raises(ValueError, match="align"):
        s.fit_residual(x[:-1], z, target, r_r)
    bad = z.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        s.fit_residual(x, bad, target, r_r)


def test_cli_phases_include_full_frozen_workflow():
    assert set(s.PHASES) == {"validation", "final"}
    assert "run_final_if_eligible" in inspect.getsource(s.main)


def test_final_scoring_rechecks_validation_gate(monkeypatch, tmp_path):
    called = []

    def gate():
        called.append(True)
        raise ValueError("validation did not authorize T")

    monkeypatch.setattr(s, "_verify_validation_gate", gate)
    with pytest.raises(ValueError, match="did not authorize"):
        s.score_phase("final", tmp_path / "not-read.json")
    assert called == [True]


def test_fold_checkpoint_requires_source_claim_binding(monkeypatch, tmp_path):
    monkeypatch.setattr(s, "OUT", tmp_path)
    protocol = {"protocolSha256": "p", "inputs": {"prices": "x"},
                "runtime": {"python": "3.12"}}
    fit = {"fitId": "garlic-h1-2026-04", "trainingKeysSha256": "keys",
           "trainingRows": 10, "trainingMatrixSha256": "matrix",
           "trainingTargetSha256": "target"}
    claim_path = tmp_path / ".fit-claims" / "validation" / f"{fit['fitId']}-B-gamma.json"
    claim_path.parent.mkdir(parents=True)
    protocol["inputs"]["module"] = "module"
    claim_path.write_bytes(s.canonical({"schema": s.SCHEMA, "phase": "validation",
        "fitId": f"{fit['fitId']}-B-gamma", "protocolSha256": "p", "inputs": protocol["inputs"],
        "moduleSha256": "module", "runtime": {"python": "3.12"}, "reservedAtUtc": "synthetic"}))
    s._save_fit_state("validation", fit, protocol)
    assert s._load_fit_state("validation", fit["fitId"], fit, protocol) == fit
    protocol["protocolSha256"] = "changed"
    with pytest.raises(ValueError, match="state source binding"):
        s._load_fit_state("validation", fit["fitId"], fit, protocol)


def test_synthetic_build_verify_score_resume_and_validation_failed_final(monkeypatch, tmp_path):
    """Exercise the production phase wiring with only deterministic synthetic inputs."""
    monkeypatch.setattr(s, "OUT", tmp_path)
    monkeypatch.setattr(s, "ROOT", tmp_path)
    monkeypatch.setattr(s, "verify_protocol", lambda: {
        "protocolSha256": "synthetic-protocol", "inputs": {"module": "synthetic-module"},
        "runtime": {"python": "synthetic"}})
    monkeypatch.setattr(s.rmh, "hidden_design",
                        lambda x, coefs, intercepts: np.column_stack((np.ones(len(x)), np.sin(x[:, :16]))))
    monkeypatch.setattr(s.rmh, "ridge_reproduces", lambda a, b: abs(a - b) <= 1e-9 * max(1., abs(a), abs(b)))
    monkeypatch.setattr(s.rmh, "verify_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(s, "_score_object", lambda doc, phase: {
        "primaryDecision": {"passes": False, "interpretation": "synthetic_validation_failure"},
        "syntheticScore": True})

    horizons = tuple(range(1, 11))
    months = ((2026, 4), (2026, 5), (2026, 6))
    keys, base_rows, r106_rows = [], [], {}
    feature_rows, by_h = {h: {} for h in horizons}, {h: [] for h in horizons}
    source_audits = []
    training_data = {}
    rng = np.random.default_rng(7301)
    for h in horizons:
        for year, month in months:
            origin = dt.date(year, month, 10)
            target = origin + dt.timedelta(days=h)
            key = ("garlic", target, h)
            keys.append(key)
            by_h[h].append(key)
            vector = rng.normal(size=106)
            fr = {"preserved106": vector.tolist(), "p0": 100.0 + h,
                  "p0Date": target - dt.timedelta(days=1)}
            feature_rows[h][target] = fr
            beta = np.linspace(-.001, .001, 107)
            rlog = float(beta[0] + vector @ beta[1:])
            old_r = s.direct_price(fr["p0"], rlog)
            base_rows.append({"product": "garlic", "target": target.isoformat(), "horizon": h,
                              "candidate": {"status": "issued", "prediction": 103.0,
                                            "reason": None}})
            r106_rows[key] = {"models": {
                "preserved106": {"status": "issued", "prediction": old_r},
                "causalSarima": {"status": "issued", "prediction": 105.0}}}
            x = rng.normal(size=(120, 106))
            y = .01 + x[:, :3] @ np.array([.02, -.01, .015]) + rng.normal(scale=.01, size=120)
            training = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(len(x))]
            audit = {"fitOrigin": origin.isoformat(), "trainingRows": len(x)}
            training_data[f"garlic-h{h}-{year}-{month:02d}"] = (training, origin, audit, x, y)
            source_audits.append({"fitId": f"garlic-h{h}-{year}-{month:02d}",
                "scalerMean": np.zeros(106).tolist(), "scalerScale": np.ones(106).tolist(),
                "mlpCoefs": [np.zeros((106, 16)).ravel().tolist(), np.zeros((16, 1)).ravel().tolist()],
                "mlpCoefShapes": [[106, 16], [16, 1]],
                "mlpIntercepts": [np.zeros(16).tolist(), np.zeros(1).tolist()],
                "ridgeCoefficientIncludingIntercept": beta.tolist(),
                "rTrainingPrediction": (beta[0] + x @ beta[1:]).tolist()})

    def base_doc(phase):
        return {"forecasts": base_rows}

    def index(rows):
        return {(r["product"], dt.date.fromisoformat(r["target"]), int(r["horizon"])): r
                for r in rows}

    def make_features(*args):
        return {}, feature_rows, by_h

    def identity_rows(phase, base, r106):
        return [{"product": p, "target": d.isoformat(), "origin": (d-dt.timedelta(days=h)).isoformat(),
                 "horizon": h, "eligible": True,
                 "status": base[(p,d,h)]["candidate"]["status"],
                 "prediction": base[(p,d,h)]["candidate"]["prediction"], "reason": None,
                 "usedMLP": False, "fallbackReason": None} for p,d,h in keys]

    def training_support(phase, h, month, series, features, frozen):
        year, month_number = month
        fit_id = f"garlic-h{h}-{year}-{month_number:02d}"
        return training_data[fit_id]

    monkeypatch.setattr(s.mlp, "HORIZONS", horizons)
    monkeypatch.setattr(s.mlp, "keys", lambda phase: keys)
    monkeypatch.setattr(s.mlp, "_base_doc", base_doc)
    monkeypatch.setattr(s.mlp, "_idx", index)
    monkeypatch.setattr(s.mlp, "_r106_index", lambda: r106_rows)
    monkeypatch.setattr(s.mlp, "_make_features", make_features)
    monkeypatch.setattr(s.mlp, "_identity_rows", identity_rows)
    monkeypatch.setattr(s, "_training_support", training_support)
    monkeypatch.setattr(s.rmh, "_manifest_path", lambda phase: tmp_path / "rmh-validation.json")
    rmh_manifest = {"fitAudits": source_audits}
    (tmp_path / "rmh-validation.json").write_text(json.dumps(rmh_manifest), encoding="utf-8")

    first = s.build_phase("validation", tmp_path / "validation-manifest.json")
    s.verify_manifest(first, "validation", tmp_path / "validation-manifest.json")
    assert first["fitCounts"] == {"M": 0, "R": 0, "B": 30, "gamma": 30, "H": 0}
    assert len(list((tmp_path / ".fit-claims" / "validation").glob("*-B-gamma.json"))) == 30

    def no_refit(*args, **kwargs):
        pytest.fail("S coefficient refit occurred while resuming a saved fold")

    monkeypatch.setattr(s, "fit_residual", no_refit)
    second = s.build_phase("validation", tmp_path / "validation-resumed.json")
    s.verify_manifest(second, "validation", tmp_path / "validation-resumed.json")
    report = s.score_phase("validation", tmp_path / "validation-manifest.json")
    assert report["score"]["syntheticScore"] is True

    monkeypatch.setattr(s, "_verify_validation_gate", lambda: (_ for _ in ()).throw(s.ValidationGateFailed("synthetic V failed")))
    final_report = s.run_final_if_eligible()
    assert final_report["status"] == "not_run_validation_failed"
    assert final_report["primaryDecision"]["passes"] is False
    assert not list((tmp_path / ".fit-claims" / "final").glob("*.json"))

    broken_out = tmp_path / "broken-validation"
    broken_out.mkdir()
    monkeypatch.setattr(s, "OUT", broken_out)
    monkeypatch.setattr(s, "_verify_validation_gate",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("synthetic missing validation report")))
    with pytest.raises(FileNotFoundError, match="missing validation report"):
        s.run_final_if_eligible()
    assert not (broken_out / "final-manifest.json").exists()
    assert not (broken_out / "final-report.json").exists()
