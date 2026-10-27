import numpy as np
import pytest
import datetime as dt
import json
from scipy.linalg import cho_solve, cho_factor

from agri_data import radish_functional_sharing as fhs
from agri_data import radish_horizon_sharing as rhs
from agri_data import radish_functional_sharing_score as scorer


def test_edf_floor_uses_numerical_nullspace_without_modifying_penalty():
    p = np.diag([0.0, 1.0, 2.0])
    g = np.diag([2.0, 1.0, 3.0])
    q0 = g + 10 * np.eye(3)
    audit = fhs._edf_floor(p, g, q0)
    assert audit["nullity"] == 1
    assert audit["dfInfinity"] == pytest.approx(2 / 12)
    assert audit["numericZeroTolerance"] == pytest.approx(
        64 * np.finfo(np.float64).eps * 3 * np.linalg.norm(p, 2)
    )
    assert audit["rawEigenvalues"] == pytest.approx([0.0, 1.0, 2.0])


def test_edf_floor_classifies_only_roundoff_negative_and_rejects_material_negative():
    tau = 64 * np.finfo(np.float64).eps * 2
    p = np.diag([-tau / 2, 1.0])
    g = np.eye(2)
    audit = fhs._edf_floor(p, g, g + 10 * np.eye(2))
    assert audit["nullity"] == 1
    assert audit["negativeRoundoffCount"] == 1
    with pytest.raises(ArithmeticError, match="material negative eigenvalue"):
        fhs._edf_floor(np.diag([-1e-5, 1.0]), g, g + 10 * np.eye(2))


def test_zero_penalty_is_valid_and_reports_the_unreachable_edf_floor():
    g = np.eye(3)
    q0 = g + 10 * np.eye(3)
    audit = fhs._edf_floor(np.zeros((3, 3)), g, q0)
    assert audit["nullity"] == 3
    assert audit["dfInfinity"] == pytest.approx(rhs._edf(g, q0))


def _a_inputs():
    rng = np.random.default_rng(16)
    xs = [rng.normal(size=(95 + h, 40)) for h in range(10)]
    for h, x in enumerate(xs):
        x[:, 1] = h + 1
    means = [x[:, list(rhs.ACTIVE)].mean(axis=0).tolist() for x in xs]
    fit = {
        "scalerXMean": means,
        "scalerYMean": [0.25] * 10,
        "fitCoefficients": {
            "B": (np.zeros((10, 39))).tolist(),
            "C": (np.ones((10, 39)) * 0.02).tolist(),
        },
    }
    ys = [np.zeros(len(x)) for x in xs]
    return fit, xs, ys


def test_a_rule_has_explicit_unsupported_state_for_zero_c_change():
    fit, xs, ys = _a_inputs()
    fit["fitCoefficients"]["C"] = fit["fitCoefficients"]["B"]
    result = fhs._a_audit(fit, xs, ys, np.ones((10, 39)) * 0.03)
    assert result["status"] == "unsupported"
    assert result["reason"] == "training_C_change_numerically_zero"
    assert result["E_C"] == 0
    assert result["E_C_zeroTolerance"] > 0


def test_a_rule_matches_f_training_rms_with_one_month_scalar():
    fit, xs, ys = _a_inputs()
    result = fhs._a_audit(fit, xs, ys, np.ones((10, 39)) * 0.06)
    assert result["status"] == "ok"
    assert result["rho"] == pytest.approx(3.0)
    assert result["E_A"] == pytest.approx(result["E_F"], rel=1e-13)
    assert len(result["perHorizonScale"]) == len(result["perHorizonZeroTolerance"]) == 10


def test_f_joint_solver_reaches_saved_c_edf_and_returns_finite_coefficients():
    rng = np.random.default_rng(106)
    x_by_h, y_by_h, centers, centered_y = [], [], [], []
    for h in range(10):
        x = rng.normal(size=(92 + h, 40))
        x[:, 1] = h + 1
        y = 0.15 * x[:, 0] - 0.08 * x[:, 5] + rng.normal(scale=0.1, size=len(x))
        center, ym = x[:, list(rhs.ACTIVE)].mean(0), y.mean()
        x_by_h.append(x)
        y_by_h.append(y)
        centers.append(center)
        centered_y.append(y - ym)
    g = np.zeros((390, 390))
    b = np.zeros(390)
    for h, (x, y) in enumerate(zip(x_by_h, centered_y)):
        xc = x[:, list(rhs.ACTIVE)] - centers[h]
        sl = slice(39 * h, 39 * (h + 1))
        g[sl, sl] = xc.T @ xc
        b[sl] = xc.T @ y
    q0 = g + 10 * np.eye(390)
    scales = np.ones((10, 39))
    c_penalty = rhs.graph_penalties(scales)["C"]
    df0 = rhs._edf(g, q0)
    df_inf = rhs._shared_limit(g, q0, scales)
    target = df_inf + rhs.EDF_FRACTION * (df0 - df_inf)
    c_solution = rhs.solve_edf_lambda(g, q0, c_penalty, target, df0=df0, df_inf=df_inf,
                                      user_tolerance=rhs.EDF_USER_TOL * max(1.0, df0))
    assert c_solution["status"] == "ok"
    c_lam = c_solution["lambda"]
    c_beta = cho_solve(cho_factor(q0 + c_lam * c_penalty, lower=True), b).reshape(10, 39)
    b_beta = cho_solve(cho_factor(q0, lower=True), b).reshape(10, 39)
    ymeans = [float(y.mean()) for y in y_by_h]
    training = {"B": [], "C": []}
    for h, x in enumerate(x_by_h):
        xc = x[:, list(rhs.ACTIVE)] - centers[h]
        training["B"].append((xc @ b_beta[h] + ymeans[h]).tolist())
        training["C"].append((xc @ c_beta[h] + ymeans[h]).tolist())
    fit = {"fitId":"synthetic-radish-2026-01-01","scalerXMean": [v.tolist() for v in centers], "scalerScale": scales.tolist(),
           "scalerYMean": ymeans, "fitCoefficients": {"B": b_beta.tolist(), "C": c_beta.tolist()},
           "trainingPredictions": training, "lambdas": {"C": c_lam},
           "edf": {"dfTarget": target, "df0": df0, "dfInf": df_inf},
           "diagnostics": {"C": {"actualEdf": c_solution["actualDf"]}}}
    result = fhs._solve_f(x_by_h, y_by_h, fit, phase="V")
    assert result["status"] == "ok"
    assert abs(result["dfActualF"] - target) <= result["edfTolerance"]
    assert result["lambda"] > 0
    assert result["relativeSolveResidual"] <= fhs.SOLVE_REL_TOL
    assert np.isfinite(np.asarray(result["fitCoefficients"])).all()


def test_scorer_csv_is_deterministic_and_carries_all_four_conditions():
    row = {
        "product": "radish", "eligible": True, "originDate": "2026-04-01",
        "targetDate": "2026-04-02", "horizon": 1, "fitId": "radish-2026-04-01",
        "cutoff": "2026-04-01", "p0": 100.0, "p0Date": "2026-04-01",
        "sarimaPrediction": 101.0,
        "conditions": {c: {"status": "issued", "directLogPrediction": 0.0,
                           "directPrediction": 100.0, "blendPrediction": 100.5,
                           "fallbackReason": None} for c in fhs.CONDITIONS},
    }
    a, b = fhs._forecast_csv([row]), fhs._forecast_csv([row])
    assert a == b
    assert sum(line.startswith(b"radish,") for line in a.splitlines()) == 4
    assert b"condition,status,directLogPrediction" in a


def test_eligible_source_condition_adapter_emits_exact_bcfa_keys_and_sarima():
    source = {c: {"status":"issued", "directLogPrediction":0.0, "directPrediction":100.0,
                  "blendPrediction":100.5, "sarimaPrediction":101.0} for c in ("B","C","D","P")}
    conditions = fhs._condition_skeleton(source,"radish",True)
    conditions["F"] = dict(conditions["C"])
    conditions["A"] = dict(conditions["C"])
    assert set(conditions) == {"B","C","F","A"}
    assert "D" not in conditions and "P" not in conditions
    models = {"causalDirect":{"status":"issued","prediction":100.0},
              "causal50":{"status":"issued","prediction":100.5,"provenance":{"fallback":False}},
              "causalSarima":{"status":"issued","prediction":101.0}}
    reference = fhs._rhs_baseline_ref({"models":models})
    assert reference == {"status":"issued","prediction":100.5,"directPrediction":100.0,"sarimaPrediction":101.0}


def test_fit_origin_selection_uses_phase_cutoff_day_not_month_as_day():
    source = {("radish","2026-06-30","2026-07-01",1):{},
              ("radish","2026-07-01","2026-07-02",1):{},
              ("radish","2026-08-01","2026-08-02",1):{},
              ("radish","2026-09-01","2026-09-02",1):{}}
    assert fhs._eligible_fit_origins(source,"T")==[dt.date(2026,7,1),dt.date(2026,8,1),dt.date(2026,9,1)]


def test_phase_claim_resumes_same_protocol_and_rejects_changed_protocol(tmp_path):
    first=fhs._phase_claim(tmp_path,"V","protocol-a")
    assert fhs._phase_claim(tmp_path,"V","protocol-a")==first
    with pytest.raises(ValueError,match="cannot be resumed"):
        fhs._phase_claim(tmp_path,"V","protocol-b")


def test_fit_solver_numerical_exception_is_persisted_as_explicit_phase_failure(monkeypatch,tmp_path):
    def fail(*args,**kwargs): raise np.linalg.LinAlgError("synthetic singular system")
    monkeypatch.setattr(fhs,"_solve_f",fail)
    bundle=fhs._fit_f_bundle("V","synthetic-fit",{},[],[],[],{"protocolSha256":"p"})
    assert bundle["status"]=="numerical_failure"
    assert "synthetic singular system" in bundle["f"]["reason"]
    fhs._record_fit_failure(tmp_path,"V","synthetic-fit","attempt-1","p",bundle["f"])
    terminal=json.loads((tmp_path/"validation-terminal.json").read_text())
    assert terminal["status"]=="numerical_failure"
    assert terminal["failureSha256"]==fhs.file_sha(tmp_path/"validation-fit-failure.json")


def test_verified_manifest_recovers_missing_phase_terminal_without_refitting(monkeypatch,tmp_path):
    protocol={"protocolSha256":"p"}
    monkeypatch.setattr(fhs,"_load_protocol",lambda _path:protocol)
    calls=[]
    def verify(phase,output_dir,*,allow_missing_terminal=False):
        calls.append(allow_missing_terminal)
        return {"phase":phase,"status":"answer_free_locked"}
    monkeypatch.setattr(fhs,"verify_phase",verify)
    manifest=tmp_path/"validation-manifest.json";manifest.write_bytes(b"verified elsewhere")
    claim={"phase":"V","status":"running","attemptId":"attempt-2","protocolSha256":"p"}
    (tmp_path/"validation-claim.json").write_bytes(fhs.canonical(claim))
    result=fhs._build_phase("V",tmp_path)
    assert result["status"]=="answer_free_locked"
    assert calls==[True,False]
    terminal=json.loads((tmp_path/"validation-terminal.json").read_text())
    assert terminal["recoveredFromVerifiedManifest"] is True
    assert terminal["manifestSha256"]==fhs.file_sha(manifest)


def test_interrupted_derived_artifacts_resume_only_when_identical(tmp_path):
    # Simulate a crash after writing CSV/diagnostics but before the answer-free manifest.
    csv=tmp_path/"validation-predictions.csv"
    diagnostics=tmp_path/"fit-diagnostics.json"
    csv_bytes=b"synthetic deterministic rows\n"
    diagnostics_bytes=b'{"jointFits":[],"fitDiagnostics":[]}\n'
    fhs._write_derived(csv,csv_bytes)
    fhs._write_derived(diagnostics,diagnostics_bytes)
    assert not (tmp_path/"validation-manifest.json").exists()
    fhs._write_derived(csv,csv_bytes)
    fhs._write_derived(diagnostics,diagnostics_bytes)
    assert csv.read_bytes()==csv_bytes and diagnostics.read_bytes()==diagnostics_bytes
    with pytest.raises(ValueError,match="existing derived artifact differs"):
        fhs._write_derived(csv,b"tampered rows\n")
    with pytest.raises(ValueError,match="existing derived artifact differs"):
        fhs._write_derived(diagnostics,b'{"tampered":true}\n')


def test_generated_bcfa_grid_satisfies_scorer_manifest_contract(monkeypatch):
    first,last=dt.date(2026,4,11),dt.date(2026,4,12)
    monkeypatch.setitem(scorer.WINDOWS,"V",(first,last))
    monkeypatch.setitem(scorer.ORIGIN_CUTOFF,"V",dt.date(2026,4,1))
    monkeypatch.setattr(scorer,"BLOCKS",())
    forecasts=[];baseline={};actuals={}
    for product in scorer.PRODUCTS:
        for offset in range((last-first).days+1):
            target=first+dt.timedelta(days=offset)
            actuals[product,target.isoformat()]=100.0
            for h in scorer.HORIZONS:
                origin=target-dt.timedelta(days=h); key=(product,origin.isoformat(),target.isoformat(),h)
                source={"status":"issued","directLogPrediction":0.0,"directPrediction":100.0,
                        "blendPrediction":100.5,"sarimaPrediction":101.0,"fitId":None,"cutoff":"2026-04-01",
                        "p0":100.0,"p0Date":"2026-04-10","fallbackReason":None}
                conditions={c:dict(source) for c in fhs.CONDITIONS}
                forecasts.append({"product":product,"eligible":True,"originDate":origin.isoformat(),
                                  "targetDate":target.isoformat(),"horizon":h,"p0":100.0,
                                  "sarimaPrediction":101.0,"conditions":conditions})
                if product=="radish": baseline[key]={"status":"issued","prediction":100.5,"directPrediction":100.0,"sarimaPrediction":101.0}
    doc={"forecasts":forecasts}
    scored=scorer.score_manifest(doc,baseline_rows=baseline,actuals=actuals,fit_diagnostics=[],phase="V")
    assert scored["vGateMetrics"]["passed"] is False  # exact tie is not a V improvement
    assert len(forecasts)==80
