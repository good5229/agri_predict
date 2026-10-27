import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pytest

from agri_data import period_mape10 as m
from agri_data import period_mape10_score as score


def test_lp_intercept_only_inverse_ratio_weighted_median_and_exact_objective():
    # The ordinary median is 2; the inverse-ratio weighted median is 1.
    fit=m.weighted_l1_fit([[0.],[0.],[0.]],[1.,2.,4.],lam=0.)
    assert fit["intercept"] == pytest.approx(1.)
    assert fit["coef"] == pytest.approx([0.])
    assert fit["objective"] == pytest.approx((0 + .5 + .75)/3)
    assert min(fit["trainingPredictions"]) >= .05


def test_lp_hand_calculated_mape_and_single_inverse_ratio():
    fit=m.weighted_l1_fit([[0.],[0.]],[1.,3.],lam=0.)
    expected=min((abs(fit["intercept"]-1)+abs(fit["intercept"]-3)/3)/2,
                 (abs(1.-1)+abs(1.-3)/3)/2)
    assert fit["objective"] == pytest.approx(expected)
    assert fit["inverseRatioAppliedOnce"]


def test_model_bundle_replays_clipped_prediction():
    b=m.fit_ratio_model("M",[[0.,1.],[1.,0.],[2.,1.]],[1.,2.,1.5],alpha=1.,lam=.01)
    x=[8.,-2.]
    pred,_=m.predict_ratio(b,x)
    m.verify_reload(b,x,pred)

def test_model_artifact_reloads_from_disk_and_replays_forecast_row(tmp_path):
    x=[8.,-2.]
    b=m.fit_ratio_model("M",[[0.,1.],[1.,0.],[2.,1.]],[1.,2.,1.5],alpha=1.,lam=.01)
    ratio,_=m.predict_ratio(b,x); prediction=100*ratio
    model={"modelId":"synthetic","fitOrigin":"2020-01-01","method":"M","inputCutoff":"2019-12-30",
        "lastTrainingTargetEnd":"2019-12-20","trainLabels":3,"scalerIdentity":b["scalerIdentity"],"fitSeconds":.1,
        "bundle":b,"forecastInput":x,"referencePrice":100.,"expectedRatio":ratio,"expectedPrediction":prediction}
    meta=m._save_models(tmp_path,{"synthetic":model},"disk_replay")
    loaded=json.loads(Path(meta["path"]).read_text().splitlines()[0])
    assert loaded["forecastInput"]==x and loaded["expectedPrediction"]==pytest.approx(prediction)


def test_log_ratio_ridge_prediction_is_exponentiated_without_floor():
    b={"method":"R","scalerMean":[0.],"scalerScale":[1.],"intercept":-4.,"coef":[0.]}
    ratio,clipped=m.predict_ratio(b,[1.])
    assert ratio==pytest.approx(np.exp(-4.))
    assert not clipped


def test_recency_age_uses_calendar_dekads_including_gaps():
    periods=[(dt.date(2020,1,1),dt.date(2020,1,10)),
             (dt.date(2020,1,21),dt.date(2020,1,31))]
    weights=m.recency_weights(periods,dt.date(2020,2,5))
    assert weights.mean()==pytest.approx(1.)
    assert weights[1]>weights[0]
    idx0=m.proxy._period_index(periods[0]); idx1=m.proxy._period_index(periods[1])
    assert weights[1]/weights[0]==pytest.approx(2**((idx1-idx0)/m.HALF_LIFE_PERIODS))


def test_persisted_forecast_rejects_realized_labels(tmp_path):
    with pytest.raises(ValueError,match="actual"):
        m.write_forecasts(tmp_path/"forecasts.parquet",[{"actual":1,"predicted":1}])
    with pytest.raises(ValueError,match="actual"):
        m.write_forecasts(tmp_path/"forecasts.parquet",[{"targetObserved":True,"predicted":1}])

def test_jsonl_fallback_and_scorer_use_checkpoint_artifact_not_stale_parquet(tmp_path,monkeypatch):
    monkeypatch.setitem(sys.modules,"pyarrow",None)
    rows=[{"crop":"cabbage","horizon":2,"forecasts":{}}]
    meta=m.write_forecasts(tmp_path/"forecasts_test.parquet",rows)
    assert meta["format"]=="jsonl_fallback_no_parquet_engine"
    (tmp_path/"stale.parquet").write_text("not parquet")
    read,path=score.read_rows(meta)
    assert path==Path(meta["path"]) and read==rows

def test_checkpoint_identity_upstream_provenance_and_preflight_preservation(tmp_path,monkeypatch):
    pf=m.save_checkpoint("preflight",{"runId":m.RUN_ID,"complete":True},tmp_path)
    identity=m.run_identity()
    (tmp_path/"resolved_config.json").write_text(json.dumps({"runIdentity":identity,"P":{"cabbage":"M"}}))
    monkeypatch.setattr(m.proxy,"validate_inputs",lambda:pytest.fail("compatible preflight must be preserved"))
    result=m.preflight(tmp_path)
    assert result["runIdentity"]==identity
    assert json.loads((tmp_path/"resolved_config.json").read_text())["P"]=={"cabbage":"M"}
    m.save_checkpoint("pilot",{"runId":m.RUN_ID,"complete":True},tmp_path)
    doc=json.loads((tmp_path/"preflight.json").read_text());doc["complete"]=False
    (tmp_path/"preflight.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="upstream provenance"):
        m.load_checkpoint("pilot",tmp_path)
    doc["runIdentity"]["sourceSha256"]="stale"
    (tmp_path/"preflight.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="different frozen inputs"):
        m.preflight(tmp_path)

def test_run_identity_enforces_pinned_proxy_implementation(monkeypatch):
    original=m.digest
    def changed_proxy(path):
        if Path(path).resolve()==Path(m.proxy.__file__).resolve():return "0"*64
        return original(path)
    monkeypatch.setattr(m,"digest",changed_proxy)
    with pytest.raises(ValueError,match="proxy implementation"):
        m.run_identity()

def test_tuning_maturity_uses_evaluation_targets_not_train_labels():
    training_max=dt.date(2020,4,30)
    evaluation_target=(dt.date(2020,2,1),dt.date(2020,2,10))
    grid={("cabbage",2):({}, {evaluation_target:{"actual":123.}}, None,
        [{"target":evaluation_target,"lastTrainingTargetEnd":training_max}])}
    assert m._tuning_evaluation_maturity(grid)==dt.date(2020,2,12)


def test_synthetic_saved_forecasts_score_end_to_end_and_keep_full_families(tmp_path,monkeypatch):
    periods=m.proxy.dekads(dt.date(2024,1,1),dt.date(2026,9,30))
    raw={"products":[]}
    actual_by_crop={}
    for ci,crop in enumerate(m.proxy.CROPS):
        obs=[];actual_by_crop[crop]={}
        for i,(lo,hi) in enumerate(periods):
            value=100.+ci*30+i%9*2
            actual_by_crop[crop][lo.isoformat()]=value
            for day in (lo,lo+dt.timedelta(days=1),lo+dt.timedelta(days=2)):
                obs.append({"date":day.isoformat(),"actual":str(value+i%3),"volumeKg":"10"})
        raw["products"].append({"id":crop,"observations":obs})
    monkeypatch.setattr(score.proxy,"validate_inputs",lambda:( {},raw ))
    m.save_checkpoint("preflight",{"runId":m.RUN_ID,"complete":True},tmp_path)
    rows=[]
    methods=("P","Pglobal","F","B0","R","W","S","M","MW")
    for crop in m.proxy.CROPS:
        ci=m.proxy.CROPS.index(crop)
        for h in m.proxy.HORIZONS:
            for i,(lo,hi) in enumerate(periods):
                actual=actual_by_crop[crop][lo.isoformat()]
                factors={"P":1.01,"Pglobal":1.02,"F":1.05,"B0":.96,"R":1.03,
                         "W":1.025,"S":1.04,"M":1.015,"MW":1.012}
                forecasts={k:{"predicted":actual*factors[k]+(i%5)*.1,"referencePrice":actual*.98}
                           for k in methods}
                unavailable=(i<1 if h==2 else i<2)
                if unavailable: forecasts.pop("P");forecasts.pop("F")
                rows.append({"crop":crop,"horizon":h,"targetStart":lo.isoformat(),"targetEnd":hi.isoformat(),
                    "targetKey":lo.isoformat(),"origin":(lo-dt.timedelta(days=1)).isoformat(),
                    "inputCutoff":(lo-dt.timedelta(days=2)).isoformat(),"policyUnavailable":unavailable,"forecasts":forecasts})
    artifact=m.write_forecasts(tmp_path/"forecasts_test.parquet",rows)
    for stage in ("pilot","tune","develop"):
        m.save_checkpoint(stage,{"runId":m.RUN_ID,"complete":True},tmp_path)
    m.save_checkpoint("test",{"runId":m.RUN_ID,"complete":True,"forecastArtifact":artifact},tmp_path)
    if artifact["format"]=="parquet":
        import pandas as pd
        assert "actual" not in pd.read_parquet(artifact["path"]).columns
    else:
        assert "actual" not in json.loads(Path(artifact["path"]).read_text().splitlines()[0])
    report=score.score_saved(tmp_path)
    assert len(report["main32"])==32 and len(report["mechanism64"])==64
    assert report["hypothesisCounts"]=={"mainPerHACBandwidth":32,"mechanismPerHACBandwidth":64}
    assert all("hac3" in x and "hac6" in x for x in report["main32"]+report["mechanism64"])
    assert set(report["jointTargetDekadBlockBootstrap"]["P-F"])=={"6","12"}
    assert len(report["metricsByCell"])==4
    assert report["MAPE10"]["cellsAtOrBelow10"]==8
    assert (tmp_path/"result.md").exists()


def test_all_stages_resume_artifacts_and_score_on_synthetic_data(tmp_path,monkeypatch,capsys):
    periods=m.proxy.dekads(dt.date(2014,1,1),dt.date(2026,9,30))
    raw={"products":[]};series_by_crop={}
    for ci,crop in enumerate(m.proxy.CROPS):
        obs={}
        for i,(lo,hi) in enumerate(periods):
            for j,day in enumerate((lo,lo+dt.timedelta(days=1),lo+dt.timedelta(days=2))):
                value=250.+ci*80+15*math.sin(i/13)+j
                obs[day]=(value,100+j)
        series_by_crop[crop]=obs
        raw["products"].append({"id":crop,"observations":[{"date":d.isoformat(),"actual":str(v[0]),"volumeKg":str(v[1])} for d,v in obs.items()]})
    original_schedule=m.proxy.planned_origins
    def short_schedule(start,end,h):
        rows=original_schedule(start,end,h)
        if start==dt.date(2020,1,1):
            first={}
            for s in rows:
                if 2020<=s["origin"].year<=2021:first.setdefault(m.proxy._quarter_key(s["origin"]),s)
            return list(first.values())
        if start==dt.date(2022,1,1):
            by_year={}
            for s in rows:by_year[dt.date.fromisoformat(s["target"][0].isoformat()).year]=s
            return [by_year[y] for y in sorted(by_year)]
        return rows
    monkeypatch.setattr(m.proxy,"planned_origins",short_schedule)
    monkeypatch.setattr(m,"_verified_inputs",lambda out:None)
    monkeypatch.setattr(m,"_load_data",lambda:({},raw,series_by_crop))
    monkeypatch.setattr(score.proxy,"validate_inputs",lambda:({},raw))
    train_cache={}
    def fake_forecast(series,targets,features,h,target,origin,method,alpha,lam,model_store):
        row=features.row(origin,target)
        if row is None:raise ValueError("synthetic feature unavailable")
        _,reference=row
        ratio={"R":1.09,"W":1.06,"S":1.04,"M":1.01,"MW":1.025}.get(method,1.01)
        if lam is not None:ratio+=float(lam)*.1
        elif method=="S":ratio+=float(alpha)*.0001
        predicted=reference*ratio
        train_key=(id(targets),origin)
        if train_key not in train_cache:
            train_cache[train_key]=[[p[0].isoformat(),p[1].isoformat()] for p,v in targets.items()
                if v.get("actual") is not None and m.proxy.label_matured(p,origin)]
        target_ids=train_cache[train_key]
        model_id=hashlib.sha256(f"{method}:{origin}:{target[0]}:{alpha}:{lam}".encode()).hexdigest()
        raw_for_bundle=math.log(ratio) if method in ("R","W") else ratio
        x_array=list(map(float,row[0]))
        model_store[model_id]={"modelId":model_id,"fitOrigin":origin.isoformat(),"method":method,
            "inputCutoff":m.proxy.input_cutoff(origin).isoformat(),"scalerIdentity":"synthetic-scaler","fitSeconds":0.0001,
            "lastTrainingTargetEnd":target_ids[-1][1] if target_ids else None,"trainingTargets":target_ids,
            "trainLabels":len(target_ids),"forecastInput":x_array,"referencePrice":reference,
            "expectedRatio":ratio,"expectedPrediction":predicted,
            "bundle":{"method":method,"scalerMean":[0.]*len(x_array),"scalerScale":[1.]*len(x_array),
                "intercept":raw_for_bundle,"coef":[0.]*len(x_array)}}
        return {"predicted":predicted,"referencePrice":reference,"ratioRaw":ratio,"ratioUsed":ratio,
            "clippedAtFloor":False,"modelId":model_id,"trainLabels":len(target_ids),
            "scalerIdentity":"synthetic-scaler","inputCutoff":m.proxy.input_cutoff(origin).isoformat(),
            "fitSeconds":0.0001,"lastTrainingTargetEnd":target_ids[-1][1] if target_ids else None,"alpha":alpha,"lambda":lam}
    monkeypatch.setattr(m,"_forecast_one",fake_forecast)
    def fake_old_report():
        result={"crops":{}}
        for crop in m.proxy.CROPS:
            result["crops"][crop]={}
            targets=m.proxy._target_periods(series_by_crop[crop])
            for h in m.proxy.HORIZONS:
                predictions=[]
                for s in original_schedule(dt.date(2024,1,1),dt.date(2026,9,30),h):
                    row={"targetStart":s["target"][0].isoformat(),"targetEnd":s["target"][1].isoformat(),"origin":s["origin"].isoformat(),"forecasts":{}}
                    if s["origin"]>=m.proxy.POLICY_AVAILABLE_FROM:
                        ref=series_by_crop[crop][max(d for d in series_by_crop[crop] if d<=s["inputCutoff"])][0]
                        row["forecasts"]={"B1":{"predicted":ref*1.1},"selectedB0":{"predicted":ref*.98}}
                    predictions.append(row)
                result["crops"][crop][str(h)]={"selectedAlpha":1.,"selectedSimpleBaseline":"B0b","predictions":predictions}
        return result
    monkeypatch.setattr(m,"_load_previous",fake_old_report)
    # A minimal preflight makes every subsequent stage exercise its true checkpoint contract.
    m.save_checkpoint("preflight",{"runId":m.RUN_ID,"protocolSha256":m.PROTOCOL_SHA,"analysisCodeSha256":m.analysis_code_hash(),"complete":True},tmp_path)
    (tmp_path/"resolved_config.json").write_text("{}\n")
    pilot=m.pilot(tmp_path);tune=m.tune(tmp_path);dev=m.develop(tmp_path);test=m.test_stage(tmp_path)
    assert pilot["pilotReusableFit"] and tune["selectedGlobalParameters"]
    assert dev["rows"]>0 and dev["selectionSha256"]
    assert test["rows"]==4*2*99 and test["policyUnavailableCount"]==12
    report=score.score_saved(tmp_path)
    assert len(report["main32"])==32 and len(report["mechanism64"])==64
    assert json.loads((tmp_path/"preflight.json").read_text())["analysisCodeSha256"]==m.analysis_code_hash()
    assert all((tmp_path/f"{stage}.json").exists() for stage in ("pilot","tune","develop","test","score"))
    monkeypatch.setattr(m,"_forecast_one",lambda *a,**k:pytest.fail("completed tune checkpoint must resume without refitting"))
    m.main(["tune","--output-dir",str(tmp_path)])
    assert json.loads(capsys.readouterr().out)["complete"] is True
