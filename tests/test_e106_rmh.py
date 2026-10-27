import numpy as np
import pytest
import inspect

from agri_data import e106_rmh as rmh


def fixture_arrays(seed=7, n=180):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 106))
    mean = rng.normal(size=106)
    scale = np.exp(rng.normal(scale=.1, size=106))
    xs = (x - mean) / scale
    w1 = rng.normal(scale=.08, size=(106, 16))
    b1 = rng.normal(scale=.1, size=16)
    w2 = rng.normal(scale=.1, size=(16, 1))
    b2 = np.array([.03])
    y = .2 * np.sin(xs[:, 0]) + rng.normal(scale=.05, size=n)
    return x, y, mean, scale, [w1, w2], [b1, b2]


def test_hidden_design_includes_intercept_and_relu():
    x, _, mean, scale, coefs, intercepts = fixture_arrays(n=10)
    a = rmh.hidden_design((x-mean)/scale, coefs, intercepts)
    np.testing.assert_array_equal(a[:, 0], np.ones(10))
    assert a.shape == (10, 17)
    assert np.all(a[:, 1:] >= 0)


def test_decomposition_records_actual_h_loss_and_reconstruction_bounds():
    x, y, mean, scale, coefs, intercepts = fixture_arrays()
    result = rmh.decompose_training(x, y, mean, scale, coefs, intercepts)
    assert result["n"] == len(y)
    assert result["svd"]["retainedRank"] <= 17
    assert result["L_H"] == pytest.approx(
        np.mean((y - np.asarray(result["hTrainingPrediction"])) ** 2), abs=1e-14
    )
    assert result["L_M"] == pytest.approx(
        np.mean((y - np.asarray(result["mTrainingPrediction"])) ** 2), abs=1e-14
    )
    assert result["L_R"] == pytest.approx(
        np.mean((y - np.asarray(result["rTrainingPrediction"])) ** 2), abs=1e-14
    )
    assert result["D_M"] >= 0 and result["D_R"] >= 0
    assert result["L_H"] <= result["HUpperBoundFromM"] + result["boundSlack"]
    assert result["L_H"] <= result["HUpperBoundFromR"] + result["boundSlack"]
    assert result["decompositionIdentityError"] <= 1e-10
    assert result["hProjectionMaxAbsError"] <= 1e-9


def test_invalid_width_or_nonfinite_state_is_rejected():
    x, y, mean, scale, coefs, intercepts = fixture_arrays()
    with pytest.raises(ValueError, match="106-column"):
        rmh.decompose_training(x[:, :105], y, mean[:105], scale[:105], coefs, intercepts)
    bad = [coefs[0].copy(), coefs[1].copy()]
    bad[0][0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        rmh.decompose_training(x, y, mean, scale, bad, intercepts)


def test_reproduction_tolerance_is_fixed_and_asymmetric_on_reference_scale():
    assert rmh.ridge_reproduces(100.0000005, 100.0)
    assert not rmh.ridge_reproduces(100.0000011, 100.0)
    assert rmh.ridge_reproduces(2e-8, 0.0) is False


def test_price_reconstruction_has_no_implicit_clipping():
    assert rmh.price_prediction(100.0, 0.0) == 100.0
    with pytest.raises(ValueError):
        rmh.price_prediction(100.0, 1000.0)


def test_saved_svd_replay_never_refits_or_redecomposes(monkeypatch):
    x, y, mean, scale, coefs, intercepts = fixture_arrays(seed=14, n=60)
    fit = rmh.decompose_training(x, y, mean, scale, coefs, intercepts)
    xs = (x-mean)/scale
    a = rmh.hidden_design(xs, coefs, intercepts)
    monkeypatch.setattr(np.linalg, "svd", lambda *a, **k: pytest.fail("verification recomputed SVD"))
    beta, _, _, _ = rmh._saved_svd(a,y,fit)
    np.testing.assert_allclose(beta,fit["hCoefficientIncludingIntercept"],rtol=1e-9,atol=1e-10)


def test_saved_svd_accepts_rank_deficiency_from_dead_hidden_units(monkeypatch):
    x,y,mean,scale,coefs,intercepts=fixture_arrays(seed=21,n=60)
    coefs[0][:]=0.0
    intercepts[0][:]=-1.0
    fit=rmh.decompose_training(x,y,mean,scale,coefs,intercepts)
    assert fit["svd"]["retainedRank"]==1
    assert any(s==0.0 for s in fit["svd"]["singularValues"])
    a=rmh.hidden_design((x-mean)/scale,coefs,intercepts)
    monkeypatch.setattr(np.linalg,"svd",lambda *a,**k:pytest.fail("verification recomputed SVD"))
    beta,_,_,_=rmh._saved_svd(a,y,fit)
    np.testing.assert_allclose(beta,fit["hCoefficientIncludingIntercept"],rtol=1e-9,atol=1e-10)


def test_saved_svd_rejects_mutated_factor():
    x,y,mean,scale,coefs,intercepts=fixture_arrays(seed=18,n=60)
    fit=rmh.decompose_training(x,y,mean,scale,coefs,intercepts)
    fit["svdFactors"]["u"][0][0]+=0.2
    with pytest.raises(ValueError,match="SVD factors"):
        rmh._saved_svd(rmh.hidden_design((x-mean)/scale,coefs,intercepts),y,fit)


def test_four_crop_grid_is_exactly_forty_cells_and_excludes_pepper():
    rows=[]
    for product in (*rmh.FOUR_CROP_PRODUCTS,"pepper"):
        for h in range(1,11):
            rows.append({"product":product,"horizon":h,"actual":10.0,"candidate":9.0,"E106":10.0})
    cells,expected,observed=rmh._four_crop_cells(rows)
    macro=rmh._portfolio_macro(cells,expected==observed)
    assert len(cells)==40 and len(expected)==40 and expected==observed
    assert {p for p,_ in expected}==set(rmh.FOUR_CROP_PRODUCTS)
    assert macro["candidateMae"]==pytest.approx(1.0)
    assert macro["deltaMape"]==pytest.approx(10.0)
    rows=[x for x in rows if not (x["product"]=="garlic" and x["horizon"]==1)]
    cells,expected,observed=rmh._four_crop_cells(rows)
    assert len(cells)==39 and expected!=observed
    assert rmh._portfolio_macro(cells,False)["candidateMae"] is None


def test_four_crop_block_bootstrap_uses_complete_forty_cell_grid():
    from agri_data.research_progress_diagnostic import circular_block_bootstrap
    rows=[]
    for day in range(1,31):
        for product in rmh.FOUR_CROP_PRODUCTS:
            for h in range(1,11):
                rows.append({"product":product,"target":f"2026-04-{day:02d}","horizon":h,
                             "actual":100.0,"candidate":99.0,"E106":100.0})
    cells,expected,observed=rmh._four_crop_cells(
        [{**x,"actual":x["actual"],"candidate":x["candidate"],"E106":x["E106"]} for x in rows])
    assert len(cells)==40 and expected==observed
    result=circular_block_bootstrap(rows,a="candidate",b="E106",metric="ape",block=7,
        replicates=20,seed=3,calendar_start=rmh.dt.date(2026,4,1),calendar_end=rmh.dt.date(2026,4,30))
    assert result["valid"]==20 and result["interval95"]==pytest.approx([1.0,1.0])


def test_fallback_preserves_exact_issued_e106_without_claiming_h():
    base={"candidate":{"status":"issued","prediction":101.0,"reason":None}}
    fallback={"status":"issued","prediction":101.0,"reason":None,
              "usedH":False,"fallbackReason":"sarima_unavailable"}
    rmh._assert_exact_fallback(fallback,base,"synthetic")
    fallback["prediction"]=102.0
    with pytest.raises(ValueError,match="exact E106"):
        rmh._assert_exact_fallback(fallback,base,"synthetic")


def test_serialized_r_replay_mutation_is_rejected():
    row={"RDirectPrediction":100.0,"r106ReplayAbsError":0.0}
    assert rmh._assert_r_replay(row,100.0,100.0,"synthetic")==0.0
    row["RDirectPrediction"]=100.01
    with pytest.raises(ValueError,match="serialized R replay"):
        rmh._assert_r_replay(row,100.0,100.0,"synthetic")


def test_common_direct_metrics_exclude_rows_missing_any_model():
    row={"RDirectPrediction":10.0,"MStandalonePrediction":11.0,"HStandalonePrediction":12.0}
    result=rmh._common_component_row(row,tuple(row),("R","M","H"),month="2026-04",actual=10.5)
    assert result=={"month":"2026-04","actual":10.5,"R":10.0,"M":11.0,"H":12.0}
    row["HStandalonePrediction"]=None
    assert rmh._common_component_row(row,tuple(row),("R","M","H"),month="2026-04",actual=10.5) is None


def test_null_primary_conclusion_is_explicitly_unsupported():
    assert rmh._conclusion({"deltaMae":None,"deltaMape":None})=="unsupported_insufficient_fixed_cells"
    assert rmh._conclusion({"deltaMae":1.0,"deltaMape":2.0})=="worsened"


def test_fit_count_summary_reports_conditional_hidden_refits():
    assert rmh._fit_count_summary("validation")=={
        "newHiddenFits":0,"fitCounts":{"newHidden":0,"R":30,"H":30,"M":0}}
    assert rmh._fit_count_summary("final")=={
        "newHiddenFits":30,"fitCounts":{"newHidden":30,"R":30,"H":30,"M":30}}


def test_final_build_calls_verified_validation_gate_before_fit_work(monkeypatch,tmp_path):
    called=[]
    monkeypatch.setattr(rmh,"verify_protocol",lambda:{"protocolSha256":"p"})
    monkeypatch.setattr(rmh,"_runtime",lambda:{})
    monkeypatch.setattr(rmh.mlp,"_check_pinned_inputs",lambda:None)
    def gate():
        called.append(True)
        raise ValueError("synthetic gate stop")
    monkeypatch.setattr(rmh,"_verify_validation_gate",gate)
    with pytest.raises(ValueError,match="synthetic gate stop"):
        rmh.build_phase("final",tmp_path/"final.json")
    assert called==[True]


def test_validation_gate_rejects_manifest_report_binding_mismatch(monkeypatch,tmp_path):
    manifest=tmp_path/"validation-manifest.json"
    manifest.write_text("{}\n",encoding="utf-8")
    report={"manifestFileSha256":"0"*64}
    report["reportSha256"]=rmh.digest(report,"reportSha256")
    (tmp_path/"validation-report.json").write_bytes(rmh.canonical(report))
    monkeypatch.setattr(rmh,"OUT",tmp_path)
    monkeypatch.setattr(rmh,"_manifest_path",lambda phase:manifest)
    with pytest.raises(ValueError,match="does not bind"):
        rmh._verify_validation_gate()


def test_manifest_verifier_uses_saved_svd_factors_without_fit_calls():
    source=inspect.getsource(rmh.verify_manifest)
    assert "np.linalg.svd" not in source
    assert "_saved_svd(a,y,fit)" in source
    assert ".fit(" not in source


def test_resume_checkpoint_requires_all_frozen_bindings(monkeypatch,tmp_path):
    monkeypatch.setattr(rmh,"OUT",tmp_path)
    phase,fit_id="validation","garlic-h1-2026-04"
    training=[rmh.dt.date(2026,3,1),rmh.dt.date(2026,3,2)]
    x=np.ones((2,106)); y=np.array([.1,.2])
    protocol={"protocolSha256":"p","inputs":{"prices":"x"},"moduleSha256":"m","runtime":{"python":"3.12"}}
    fit={"fitId":fit_id,"fitOrigin":training[-1].isoformat(),
         "trainingKeysSha256":rmh.sha_bytes(rmh.canonical([d.isoformat() for d in training])),
         "trainingMatrixSha256":rmh.sha_bytes(x.tobytes(order="C")),
         "trainingTargetSha256":rmh.sha_bytes(y.tobytes(order="C")),"training":{"n":2},
         "mlpStateSource":"saved_validation_fit","mlpParameters":rmh.mlp.MODEL_PARAMS,"newHiddenFits":0,
         "scalerMean":[0.0]*106,"scalerScale":[1.0]*106,"mlpCoefs":[[1.0]],"mlpCoefShapes":[[1,1]],
         "mlpIntercepts":[[0.0]]}
    claim={"schema":rmh.SCHEMA,"phase":phase,"fitId":fit_id+"-RH",
           "protocolSha256":"p","runtime":{"python":"3.12"},"inputHashes":{"prices":"x"},
           "moduleSha256":"m","reservedAtUtc":"synthetic"}
    claim_path=tmp_path/".fit-claims"/phase/(fit_id+"-RH.json")
    claim_path.parent.mkdir(parents=True); claim_path.write_bytes(rmh.canonical(claim))
    state={"schema":rmh.SCHEMA,"phase":phase,"fitId":fit_id,"protocolSha256":"p",
           "inputHashes":{"prices":"x"},"moduleSha256":"m","runtime":{"python":"3.12"},
           "trainingKeysSha256":fit["trainingKeysSha256"],"fitAudit":fit}
    state["stateSha256"]=rmh.digest(state,"stateSha256")
    saved={fit_id:{"scalerMean":[0.0]*106,"scalerScale":[1.0]*106,"coefs":[[1.0]],
                   "coefShapes":[[1,1]],"intercepts":[[0.0]]}}
    assert rmh._verify_checkpoint(state,phase=phase,fit_id=fit_id,fit_origin=training[-1],
        training=training,x=x,y=y,train_audit={"n":2},protocol=protocol,saved_m=saved)==fit
    claim["inputHashes"]={"prices":"changed"}
    claim_path.write_bytes(rmh.canonical(claim))
    with pytest.raises(ValueError,match="fit claim"):
        rmh._verify_checkpoint(state,phase=phase,fit_id=fit_id,fit_origin=training[-1],
            training=training,x=x,y=y,train_audit={"n":2},protocol=protocol,saved_m=saved)
    claim["inputHashes"]={"prices":"x"}
    claim_path.write_bytes(rmh.canonical(claim))
    state["moduleSha256"]="changed"; state["stateSha256"]=rmh.digest(state,"stateSha256")
    with pytest.raises(ValueError,match="binding"):
        rmh._verify_checkpoint(state,phase=phase,fit_id=fit_id,fit_origin=training[-1],
            training=training,x=x,y=y,train_audit={"n":2},protocol=protocol,saved_m=saved)


def test_dm_ape_report_labels_fraction_and_percentage_points():
    # A -0.02 fractional APE difference equals -2 percentage points.
    result=rmh._label_dm_units({"meanLossDifference":-0.02},"ape")
    assert result["lossUnit"]=="fractional_absolute_percentage_error"
    assert result["meanLossDifferencePercentPoints"]==-2.0
