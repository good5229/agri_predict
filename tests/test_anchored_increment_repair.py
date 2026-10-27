from __future__ import annotations

import datetime as dt
import hashlib
import json

import pytest

from agri_data import anchored_increment_repair as air


def _development_rows():
    rows=[]
    for product in air.PRODUCTS:
        for horizon in air.HORIZONS:
            if product == "radish": actual=110.0
            elif product == "onion": actual=90.0
            else: actual=100.0
            target=f"2026-06-{horizon+1:02d}"
            rows.append({"product":product,"origin":f"2026-06-{1:02d}","target":target,
                         "horizon":horizon,"actual":actual,"B":100.0,
                         "baselineB":100.0,"delta":10.0})
    return rows


def test_two_eligible_crops_form_one_r2_candidate_and_one_global_gate():
    dev=_development_rows()
    r1_coeffs={"radish":0.0,"onion":0.0}
    r1=air._apply(dev,r1_coeffs,"R1")
    # Both crops passed the pre-score eligibility screen. The assembled policy
    # must retain both sign-split fits together; it cannot keep only radish
    # after seeing the development score. Cabbage and garlic stay at R1/B.
    r2=air._combined_r2_candidate(
        dev,
        {"radish":{"positive":1.0,"negative":-1.0},
         "onion":{"positive":1.0,"negative":-1.0}},
        r1_coeffs,
    )
    assert len(r2)==len(dev)==40
    for row in r2:
        if row["product"] in ("radish","onion"):
            assert row["R2Alpha"]==1.0
        else:
            assert row["R2Alpha"]==0.0
            assert row["R2"]==row["B"]
    gate=air._global_r2_gate(r1,r2)
    assert gate["complete40Cells"]
    assert gate["selectedStructure"]=="R1"
    assert "one combined R2 candidate" in gate["scope"]


def test_r2_sign_alpha_weights_follow_full_crop_horizon_weights():
    rows=[]
    # One horizon has a 9:1 sign imbalance. The R2 objective gives that
    # horizon total weight 1/10, preserving the crop's actual sign mix.
    for i in range(9):
        rows.append({"horizon":1,"delta":1.0,"target":f"2026-04-{i+1:02d}"})
    rows.append({"horizon":1,"delta":-1.0,"target":"2026-04-10"})
    for h in range(2,11):
        rows.append({"horizon":h,"delta":-1.0,"target":f"2026-04-{h+10:02d}"})
    weights=air._weights(rows)
    pos=sum(w for r,w in zip(rows,weights) if r["horizon"]==1 and r["delta"]>0)
    neg=sum(w for r,w in zip(rows,weights) if r["horizon"]==1 and r["delta"]<0)
    assert pos==0.09
    assert neg==0.01


def test_two_alpha_fit_obeys_one_joint_crop_mae_constraint():
    rows=[]
    for h in air.HORIZONS:
        target=f"2026-04-{h+1:02d}"
        rows.append({"product":"radish","target":target,"horizon":h,
                     "actual":110.0,"B":100.0,"delta":10.0})
        rows.append({"product":"radish","target":target,"horizon":h,
                     "actual":90.0,"B":100.0,"delta":-10.0})
    fit=air._fit_lp(rows,signs=True,fit_name="synthetic-joint-r2")
    assert fit["logicalFitCount"]==1
    assert fit["fittedWeightedMae"]<=fit["baselineWeightedMae"]+air.VERIFY_ABS_TOL
    assert abs(fit["alphas"]["positive"]-1.0)<1e-8
    assert abs(fit["alphas"]["negative"]-1.0)<1e-8


def test_zero_feasible_fit_uses_minimum_absolute_alpha_tie_break():
    rows=[]
    for h in air.HORIZONS:
        rows.append({"target":f"2026-04-{h+1:02d}","horizon":h,
                     "actual":100.0,"B":100.0,"delta":float(h)})
    fit=air._fit_lp(rows,fit_name="synthetic-zero-tie")
    assert fit["alphas"]["all"]==0.0
    assert fit["primaryObjectiveGap"]<=air.OBJECTIVE_FACE_TOL


def test_r2_global_gate_rejects_key_mismatch_and_unfitted_metric_stays_null():
    r1=air._apply(_development_rows(),{"radish":0.0,"onion":0.0},"R1")
    r2=air._apply(_development_rows(),{"radish":0.0,"onion":0.0},"R2")
    with pytest.raises(ValueError,match="same exact development keys"):
        air._global_r2_gate(r1,r2[:-1])
    records=[{"product":p,"horizon":h,"actual":100.0,"B":100.0,"R2":None}
             for p in air.PRODUCTS for h in air.HORIZONS]
    report=air._metrics(records,("R2",))
    assert report["complete40Cells"]
    assert report["metrics"]["R2Mae"] is None
    assert report["metrics"]["R2Mape"] is None


def test_calibration_join_excludes_post_june_targets_and_pre_june_origin_dev_rows():
    indexed={}
    for p in air.PRODUCTS:
        for h in air.HORIZONS:
            for target in ("2026-06-05","2026-06-21"):
                day=dt.date.fromisoformat(target)
                origin=(day-dt.timedelta(days=h)).isoformat()
                key=(p,origin,target,h)
                indexed[key]={"_b":100.0,"_delta":1.0,"_k1":101.0}
    actuals={(p,t):100.0 for p in air.PRODUCTS for t in ("2026-06-05","2026-06-21")}
    dev=air._join(indexed,actuals,air.DEV_START,air.DEV_END,air.DEV_ORIGIN_START)
    assert dev
    assert all(r["target"]<="2026-06-20" and r["origin"]>="2026-06-01" for r in dev)


def test_policy_prediction_falls_back_to_b_on_nonpositive_raw_price():
    row={"product":"radish","baselineB":10.0,"delta":-20.0}
    result=air._policy_prediction(row,{"radish":1.0},structure="R1")
    assert result["rawPrediction"]==-10.0
    assert result["prediction"]==10.0
    assert result["fallbackReason"]=="raw_nonpositive"


def test_contribution_counts_are_rows_not_metric_updates_and_reconcile():
    rows=[]
    for product in air.PRODUCTS:
        for horizon in air.HORIZONS:
            rows.append({"product":product,"horizon":horizon,"target":"2026-06-02",
                         "actual":110.0,"B":100.0,"K1":105.0,"R1":107.0,"delta":1.0})
    report=air._contributions(rows,("K1","R1"))
    monthly=report["productHorizonMonth"]
    assert monthly and all(item["n"]==1 for item in monthly)
    assert all(v["reconciles"] for v in report["netReconciliation"].values())


def test_action_and_residual_summaries_separate_issued_and_paired_rows():
    rows=[]
    for i,(delta,pred,reason) in enumerate(((2.0,102.0,None),(-2.0,100.0,"raw_nonpositive"))):
        origin=dt.date(2026,7,1)+dt.timedelta(days=i)
        target=origin+dt.timedelta(days=1)
        rows.append({"product":"radish","origin":origin.isoformat(),"target":target.isoformat(),
                     "horizon":1,"actual":101.0 if i==0 else None,"B":100.0,"delta":delta,
                     "policies":{"R1":{"prediction":pred,"rawPrediction":pred,"fallbackReason":reason}},
                     "R1":pred})
    paired={air._key(rows[0])}
    actions=air._action_summary(rows,paired)["R1"]
    assert actions["fullForecastGrid"]["denominator"]==2
    assert actions["positiveActualPairedGrid"]["denominator"]==1
    assert actions["fullForecastGrid"]["counts"]["fallback:raw_nonpositive"]==1
    loss=air._residual_loss_diagnosis([rows[0]],("R1",))
    assert loss["byPolicyAndResidualClass"][0]["residualClass"]=="overshot_actual"


def test_generate_t_keeps_all_v_keys_in_overlapping_full_v_view(tmp_path,monkeypatch):
    monkeypatch.setattr(air,"ROOT",tmp_path)
    cal={"stage":"calibration_frozen","experiment":"ANCHORED_INCREMENT_REPAIR_V2",
         "targetUse":{"TLabelsUsed":False},"R1_CAlpha":{"radish":0.0,"onion":0.0},
         "development":{"R2Fits":{},"R2vsR1Gate":{"selectedStructure":"R1"}},
         "finalRefit":{"selectedStructure":"R1","coefficients":{"radish":0.0,"onion":0.0}}}
    cal["calibrationSha256"]=air._sha(air._canonical(cal))
    cpath=tmp_path/"calibration.json"; raw=air._canonical(cal); cpath.write_bytes(raw)
    def source_row(product,target,horizon):
        day=dt.date.fromisoformat(target); origin=(day-dt.timedelta(days=horizon)).isoformat()
        return {"product":product,"origin":origin,"target":target,"horizon":horizon,
                "baselineB":100.0,"prediction":101.0,"kRaw":101.0,"fallbackReason":None,"delta":1.0}
    def indexed(rows):
        out={}
        for row in rows:
            key=air._key(row)
            out[key]={**row,"_key":key,"_b":100.0,"_delta":1.0,"_k1":101.0,"_sourceRow":row}
        return out
    vrows=[source_row("radish","2026-04-02",1),source_row("radish","2026-06-01",1),
           source_row("radish","2026-06-02",1)]
    trows=[source_row("radish","2026-07-11",1)]
    vdoc={"manifestSha256":"a"*64}; tdoc={"manifestSha256":"b"*64}
    def fake_load(phase,path=None):
        return (vdoc,indexed(vrows),"c"*64) if phase=="V" else (tdoc,indexed(trows),"d"*64)
    monkeypatch.setattr(air,"_load_k1",fake_load)
    generated=air.generate_t(cpath,hashlib.sha256(raw).hexdigest())
    phases=generated["phaseRows"]
    assert phases["C_training"]["count"]==1
    assert phases["June_development"]["count"]==1
    assert phases["fullV_refit_training"]["count"]==3
    assert phases["T_final_evaluation"]["count"]==1
    for row in phases["fullV_refit_training"]["forecasts"]:
        matching=next(source for source in vrows if air._key(source)==air._key(row))
        assert row["sourceProvenance"]["sourceRowSha256"]==air._sha(air._canonical(matching))
        assert row["sourceProvenance"]["sourceForecastManifestSha256"]=="a"*64


def test_final_price_gate_uses_frozen_meaningful_gain_tolerances_and_zero_alpha_status():
    metrics={"complete40Cells":True,"metrics":{"FINALMape":10.0-0.5*air.SCORE_MAPE_TOL_PP,
                                                    "BMape":10.0,"FINALMae":5.0,
                                                    "BMae":5.0}}
    passed,status=air._final_price_decision(metrics,all_alpha_zero=False,baseline_identical=False)
    assert not passed and status=="completed_no_incremental_gain"
    metrics["metrics"]["FINALMape"]=10.0-2*air.SCORE_MAPE_TOL_PP
    passed,status=air._final_price_decision(metrics,all_alpha_zero=False,baseline_identical=False)
    assert passed and status=="completed_exploratory_gain"
    passed,status=air._final_price_decision(metrics,all_alpha_zero=True,baseline_identical=True)
    assert not passed and status=="completed_baseline_identical_no_effective_calibration"
