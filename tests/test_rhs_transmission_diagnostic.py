import numpy as np
import pytest

from agri_data import radish_horizon_sharing as rhs
from agri_data import rhs_transmission_diagnostic as diagnostic


def _synthetic_fit(fit_id="radish-2026-04-01", seed=17, *, zero_change=False):
    rng=np.random.default_rng(seed)
    means_full=np.zeros((10,40))
    scales_full=np.ones((10,40))
    means=[];fit_scales=[];betas={c:[] for c in ("B","D","C","P")}
    intercepts={c:[] for c in ("B","D","C","P")}
    qcoords={c:[] for c in ("B","D","C","P")}
    predictions={c:[] for c in ("B","D","C","P")}
    raw_by_h=[]
    for hi in range(10):
        x=rng.normal(size=(110,40))
        x[:,1]=hi+1
        raw_by_h.append(x)
        means.append(x[:,list(rhs.ACTIVE)].mean(axis=0))
        scale=np.full(39,1.0+0.03*hi)
        fit_scales.append(scale)
        ymean=0.02+0.001*hi
        base=rng.normal(scale=.03,size=39)
        changes={"B":np.zeros(39) if zero_change else rng.normal(scale=.002,size=39),
                 "D":np.zeros(39) if zero_change else rng.normal(scale=.002,size=39),
                 "C":np.zeros(39) if zero_change else rng.normal(scale=.002,size=39),
                 "P":np.zeros(39) if zero_change else rng.normal(scale=.002,size=39)}
        for c in ("B","D","C","P"):
            beta=base+changes[c]
            betas[c].append(beta)
            intercept=ymean-float(means[hi]@beta)
            intercepts[c].append(intercept)
            qcoords[c].append((fit_scales[0]/scale)*beta)
            predictions[c].append(((x[:,list(rhs.ACTIVE)]-means[hi])@beta+ymean).tolist())
    fit={"fitId":fit_id,"standardScalerMeanFull":means_full.tolist(),"standardScalerScaleFull":scales_full.tolist(),
         "scalerXMean":np.asarray(means).tolist(),"scalerScale":np.asarray(fit_scales).tolist(),
         "scalerYMean":[0.02+0.001*i for i in range(10)],
         "fitCoefficients":{c:np.asarray(v).tolist() for c,v in betas.items()},
         "fitIntercepts":intercepts,"sharedCoordinateCoefficients":{c:np.asarray(v).tolist() for c,v in qcoords.items()},
         "trainingPredictions":predictions}
    return fit,raw_by_h


def test_fit_geometry_reconstructs_all_directions_and_rank_summaries():
    fit,raw=_synthetic_fit()
    transmission,curves,_basis=diagnostic._fit_geometry(fit,raw,phase="V")
    assert len(transmission)==30
    assert len(curves)==30*39
    lookup={(r["condition"],r["horizon"]):r for r in transmission}
    assert set(lookup)=={(c,h) for c in diagnostic.CONDITIONS for h in range(1,11)}
    assert all(r["coefficientChangeSquared"]>0 for r in transmission)
    assert max(abs(r["restoredInterceptAndCenteringResidual"]) for r in transmission)<1e-12
    for condition in diagnostic.CONDITIONS:
        for horizon in range(1,11):
            rows=sorted((r for r in curves if r["condition"]==condition and r["horizon"]==horizon),key=lambda r:r["ascendingRank"])
            assert len(rows)==39
            assert rows[-1]["coefficientChangeCumulativeShare"]==pytest.approx(1.0)
            assert rows[-1]["trainingPredictionEnergyCumulativeShare"]==pytest.approx(1.0)
    checkpoints=diagnostic._rank_summary(curves)
    assert len(checkpoints)==30*4
    assert {r["rank"] for r in checkpoints}==set(diagnostic.RANKS)
    assert all("not a selected rank cutoff" in r["interpretation"] for r in checkpoints)


def test_zero_coefficient_change_has_null_directional_shares():
    fit,raw=_synthetic_fit(zero_change=True)
    transmission,curves,_basis=diagnostic._fit_geometry(fit,raw,phase="T")
    assert all(r["coefficientChangeSquared"]==0 for r in transmission)
    assert all(r["weightedEigenvalue"] is None for r in transmission)
    assert all(r["coefficientChangeCumulativeShare"] is None for r in curves)
    assert all(r["trainingPredictionEnergyCumulativeShare"] is None for r in curves)


def test_eval_replay_uses_saved_intercepts_and_keeps_cross_direction_terms(monkeypatch):
    rng=np.random.default_rng(123)
    fits={};rows=[];training_bases={}
    for month_i,month in enumerate(("04","05","06")):
        fit_id=f"radish-2026-{month}-01"
        fit,_=_synthetic_fit(fit_id,seed=month_i+2)
        _tr,_cr,training_bases[fit_id]=diagnostic._fit_geometry(fit,_,phase="V")
        fits[fit_id]=fit
        # Force a known two-axis coefficient change in the training basis and
        # a nonzero restored intercept. The evaluation support below has a
        # known off-diagonal covariance in those same training directions.
        basis=training_bases[fit_id][1]
        delta_q=basis[:,0]+basis[:,1]
        scale=np.asarray(fit["scalerScale"][0]);ref=scale[0]
        fit["fitCoefficients"]["C"][0]=(np.asarray(fit["fitCoefficients"]["B"][0])+delta_q*scale/ref).tolist()
        fit["sharedCoordinateCoefficients"]["C"][0]=(np.asarray(fit["sharedCoordinateCoefficients"]["B"][0])+delta_q).tolist()
        fit["fitIntercepts"]["C"][0]=float(fit["fitIntercepts"]["B"][0])+0.03
        for h in range(1,11):
            u_points=([np.array([1.,1.]),np.array([-1.,-1.]),np.array([1.,0.]),np.array([-1.,0.])]
                      if h==1 else [None])
            for rep,u2 in enumerate(u_points):
                z=rng.normal(size=40)
                z[1]=h
                if h==1:
                    u=np.zeros(39);u[:2]=u2
                    active=np.asarray(fit["scalerXMean"][0])+(basis@u)*ref/scale
                    z[list(rhs.ACTIVE)]=active
                conditions={}
                for c in ("B","D","C","P"):
                    beta=np.asarray(fit["fitCoefficients"][c][h-1])
                    intercept=fit["fitIntercepts"][c][h-1]
                    log_prediction=float(z[list(rhs.ACTIVE)]@beta+intercept)
                    direct=100.0*np.exp(log_prediction)
                    conditions[c]={"fitId":fit_id,"status":"issued","directLogPrediction":log_prediction,
                                   "directPrediction":direct,"blendPrediction":0.5*(direct+100.0),
                                   "sarimaPrediction":100.0,"p0":100.0}
                target=(np.datetime64("2026-04-01")+np.timedelta64(len(rows)+1,"D")).astype("datetime64[D]").astype(str)
                origin=(np.datetime64(target)-np.timedelta64(h,"D")).astype("datetime64[D]").astype(str)
                rows.append({"product":"radish","eligible":True,"originDate":origin,"targetDate":target,
                             "horizon":h,"featureVector":z.tolist(),"p0":100.0,"conditions":conditions})
    monkeypatch.setitem(diagnostic.PHASE_FORECASTS,"V",len(rows))
    replay,directions=diagnostic._eval_replay("V",{"forecasts":rows},fits,training_bases)
    assert len(replay)==len(rows)*4
    assert len(directions)==3*10*3
    assert max(abs(r["logDifference"]) for r in replay)<1e-12
    assert all(r["evaluationRecentered"] is False for r in directions)
    assert all(abs(r["directionalIdentityDifference"])<1e-11 for r in directions)
    assert all(r["fullMinusDiagonal"]==pytest.approx(r["crossDirectionAndInterceptTerms"]) for r in directions)
    selected=next(r for r in directions if r["fitId"]=="radish-2026-04-01" and r["horizon"]==1 and r["condition"]=="C")
    expected_c=(float(fits["radish-2026-04-01"]["fitIntercepts"]["C"][0])-float(fits["radish-2026-04-01"]["fitIntercepts"]["B"][0])+
                np.asarray(fits["radish-2026-04-01"]["scalerXMean"][0]) @
                (np.asarray(fits["radish-2026-04-01"]["fitCoefficients"]["C"][0])-np.asarray(fits["radish-2026-04-01"]["fitCoefficients"]["B"][0])))
    assert selected["basisSource"]=="trainingGramEigenvectors"
    assert selected["interceptCenteringResidual"]==pytest.approx(expected_c,abs=1e-12)
    assert abs(expected_c)>0.03
    assert selected["fullMinusDiagonal"]==pytest.approx(1.0+expected_c**2,abs=1e-11)
    assert selected["diagonalDirectionalEnergy"]==pytest.approx(1.5,abs=1e-11)
    assert selected["evaluationLogMovementMSE"]==pytest.approx(1.5+1.0+expected_c**2,abs=1e-11)


def test_paired_support_is_fixed_to_four_issued_conditions_and_positive_actuals(monkeypatch):
    monkeypatch.setitem(diagnostic.PHASE_PAIRED,"V",1)
    rows=[]
    for day,status in (("2026-04-01","issued"),("2026-04-02","issued"),("2026-04-03","fallback")):
        rows.append({"key":{"product":"radish","originDate":"2026-03-20","targetDate":day,"horizon":1},
                     "conditions":{c:{"status":status} for c in ("B","D","C","P")}})
    actuals={("radish","2026-04-01"):100.,("radish","2026-04-02"):0.,("radish","2026-04-03"):100.}
    result=diagnostic._expected_paired_keys(rows,actuals,"V")
    assert result=={("radish","2026-03-20","2026-04-01",1)}


def test_phase_manifest_metadata_uses_phase_specific_full_grid_counts():
    v_fits={f"radish-2026-{month}-01":{} for month in ("04","05","06")}
    t_fits={f"radish-2026-{month}-01":{} for month in ("07","08","09")}
    diagnostic._validate_phase_manifest_metadata("V",{"forecasts":[None]*3240},v_fits)
    diagnostic._validate_phase_manifest_metadata("T",{"forecasts":[None]*3160},t_fits)
    with pytest.raises(ValueError,match="T phase count 3160"):
        diagnostic._validate_phase_manifest_metadata("T",{"forecasts":[None]*3240},t_fits)


def test_saved_all_four_price_cells_and_equal_horizon_months_are_verified(monkeypatch):
    condition_offset={"B":0.,"D":1.,"C":2.,"P":3.}
    def metrics(condition,h,month_offset=0.):
        x=h+condition_offset[condition]+month_offset
        return {"n":2,"directMae":x,"directMape":10*x,"blendMae":2*x,"blendMape":20*x}
    horizon_rows=[];monthly_rows=[]
    for h in range(1,11):
        for condition in ("B","D","C","P"):
            horizon_rows.append({"product":"radish","horizon":h,"condition":condition,**metrics(condition,h)})
    for month_i,month in enumerate(("2026-04","2026-05","2026-06")):
        for h in range(1,11):
            for condition in ("B","D","C","P"):
                monthly_rows.append({"product":"radish","targetMonth":month,"horizon":h,"condition":condition,
                                     **metrics(condition,h,month_i)})
    report={"allFourCommonMetrics":horizon_rows,"allFourMonthlyMetrics":monthly_rows}
    monkeypatch.setattr(diagnostic,"_json",lambda _path:report)
    aggregates=[]
    for new in ("D","C","P"):
        for layer in ("direct","blend"):
            mae_key=f"{layer}Mae";mape_key=f"{layer}Mape"
            for h in range(1,11):
                base=metrics("B",h);cond=metrics(new,h)
                aggregates.append({"scope":"horizon","comparison":f"{new}-B","condition":new,"layer":layer,"horizon":h,
                                  "n":2,"baselineMae":base[mae_key],"conditionMae":cond[mae_key],
                                  "baselineMapePct":base[mape_key],"conditionMapePct":cond[mape_key]})
            for month_i,month in enumerate(("2026-04","2026-05","2026-06")):
                base=[metrics("B",h,month_i) for h in range(1,11)]
                cond=[metrics(new,h,month_i) for h in range(1,11)]
                aggregates.append({"scope":"target_month","comparison":f"{new}-B","condition":new,"layer":layer,
                                  "targetMonth":month,"n":20,"hasAllTenHorizons":True,
                                  "recomputedEqualHorizonMonthMean":{"baselineMae":sum(x[mae_key] for x in base)/10,
                                  "conditionMae":sum(x[mae_key] for x in cond)/10,
                                  "baselineMapePct":sum(x[mape_key] for x in base)/10,
                                  "conditionMapePct":sum(x[mape_key] for x in cond)/10}})
    diagnostic._verify_price_against_original("V",{"aggregates":aggregates})
    aggregates[0]["conditionMae"]+=1
    with pytest.raises(ValueError,match="saved all-four report"):
        diagnostic._verify_price_against_original("V",{"aggregates":aggregates})


def test_protocol_uses_exclusive_creation_and_hash_links(tmp_path, monkeypatch):
    body={"schemaVersion":1,"inputSha256":{"synthetic":"a"},"diagnosticSourceSha256":"b"}
    monkeypatch.setattr(diagnostic,"_protocol_body",lambda:body)
    first=diagnostic.freeze_protocol(tmp_path)
    assert first["protocolSha256"]==diagnostic.sha256_bytes(diagnostic.canonical(body))
    with pytest.raises(FileExistsError):
        diagnostic.freeze_protocol(tmp_path)
    path=tmp_path/"protocol.json"
    doc=diagnostic._load_protocol(tmp_path)
    assert doc==first
    changed=diagnostic._json(path)
    changed["protocol"]["inputSha256"]["synthetic"]="changed"
    path.write_bytes(diagnostic.canonical(changed))
    with pytest.raises(ValueError,match="digest mismatch"):
        diagnostic._load_protocol(tmp_path)
