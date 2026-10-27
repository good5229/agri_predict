import datetime as dt
import json
import pytest

from agri_data import temperature_extremes_pilot as pilot


def test_six_completed_max_temperature_days_build_paired_features():
    origin=dt.date(2024,9,1)
    vals=(5.,10.,30.,35.,9.,31.)
    weather={"100":{origin-dt.timedelta(days=i):v for i,v in zip(range(6,0,-1),vals)},
             "108":{origin-dt.timedelta(days=i):20. for i in range(1,7)},
             "261":{origin-dt.timedelta(days=i):20. for i in range(1,7)}}
    weather["100"][origin]=1000. # Must not be read.
    candidate,ablation,audit=pilot._weather_features(weather,origin)
    assert ablation==[20.,20.,20.]
    assert candidate==pytest.approx([20.,1.,1.,20.,0.,0.,20.,0.,0.])
    assert audit["window"]==["2024-08-26","2024-08-31"]


def test_one_missing_weather_day_invalidates_both_feature_sets():
    origin=dt.date(2024,9,1)
    weather={s:{origin-dt.timedelta(days=i):20. for i in range(1,7)} for s in pilot.STATIONS}
    del weather["261"][origin-dt.timedelta(days=4)]
    del weather["100"][origin-dt.timedelta(days=5)]
    candidate,ablation,audit=pilot._weather_features(weather,origin)
    assert candidate is None and ablation is None
    assert audit["status"]=="missing_weather"
    assert audit["missingDaysByStation"]["261"]==["2024-08-28"]
    assert audit["missingDaysByStation"]["100"]==["2024-08-27"]


def test_monthly_training_target_window_is_strict_and_rows_are_shared():
    fit=dt.date(2024,9,1); h=1
    target_dates=[fit-dt.timedelta(days=365),fit-dt.timedelta(days=100),fit-dt.timedelta(days=1),fit]
    direct={}; series={"pepper":{}}
    weather={s:{} for s in pilot.STATIONS}
    for i,target in enumerate(target_dates):
        origin=target-dt.timedelta(days=h); series["pepper"][target]=100.+i
        direct[("pepper",h,target)]={"product":"pepper","horizon":h,"target":target,"origin":origin,"direct":99.}
        for station in pilot.STATIONS:
            for lag in range(1,7): weather[station][origin-dt.timedelta(days=lag)]=30.+i
    rows,missing=pilot._training_rows(direct,series,weather,"pepper",h,fit)
    assert missing==0 and len(rows)==3
    assert min(r["target"] for r in rows)==fit-dt.timedelta(days=365)
    assert max(r["target"] for r in rows)<fit
    assert all(len(r["candidateFeature"])==9 and len(r["ablationFeature"])==3 for r in rows)
    assert all([r["candidateFeature"][i] for i in (0,3,6)]==r["ablationFeature"] for r in rows)


def test_both_ridges_require_60_distinct_targets_and_same_rows():
    rows=[{"target":dt.date(2024,1,1)+dt.timedelta(days=i),"candidateFeature":[float(i)]*9,"ablationFeature":[float(i)]*3,"label":i*.001} for i in range(60)]
    cf,cm,ca=pilot._fit_model(rows,"candidateFeature","pepper",1,dt.date(2024,9,1))
    af,am,aa=pilot._fit_model(rows,"ablationFeature","pepper",1,dt.date(2024,9,1))
    assert cf is not None and af is not None
    assert ca["trainingRows"]==aa["trainingRows"]==60
    assert ca["distinctTrainingTargetDates"]==aa["distinctTrainingTargetDates"]==60
    small,_,audit=pilot._fit_model(rows[:59],"candidateFeature","pepper",1,dt.date(2024,9,1))
    assert small is None and audit["status"]=="insufficient_training"


def _fixture():
    origin=pilot.EVAL_START; target=origin+dt.timedelta(days=1); product=pilot.PRODUCTS[0]
    prices={"schemaVersion":1,"products":[]}
    for p in pilot.PRODUCTS:
        obs=[]
        if p==product:
            obs=[{"date":(origin-dt.timedelta(days=i)).isoformat(),"actual":100.+i} for i in range(1,7)]
            obs.append({"date":target.isoformat(),"actual":110.})
        prices["products"].append({"id":p,"observations":obs})
    paired=[{"product":product,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":1,"direct":105.,"actual":110.}]
    report={"protocol":{"id":"direct-horizon-ridge-2026-10-06"},"pairedRows":paired}
    weather={s:{origin-dt.timedelta(days=i):20. for i in range(1,7)} for s in pilot.STATIONS}
    prov={s:{origin-dt.timedelta(days=1):{"requestId":1,"jobId":2,"epochId":"e1","responseSha256":"x","rowNo":1,"retrievedAt":"2024-01-01T00:00:00Z"}} for s in pilot.STATIONS}
    return report,prices,weather,prov,origin,target,product


def test_build_manifest_is_complete_answer_free_and_carries_weather_provenance(monkeypatch):
    report,prices,weather,prov,origin,target,product=_fixture()
    # Keep the synthetic path focused on grid, features and fallback behavior.
    def nofit(rows,feature_name,pid,h,fit):
        return None,None,{"product":pid,"horizon":h,"fitDate":fit.isoformat(),"trainingRows":len(rows),"distinctTrainingTargetDates":len({r["target"] for r in rows}),"status":"insufficient_training"}
    monkeypatch.setattr(pilot,"_fit_model",nofit)
    manifest=pilot.build_forecast(report,prices,weather,{"prices":"p","directReport":"r"},prov)
    assert len(manifest["forecasts"])==1365
    assert all("actual" not in r for r in manifest["forecasts"])
    selected=manifest["weatherSource"]["selectedRecordsByStationDate"]
    assert selected["100"][(origin-dt.timedelta(days=1)).isoformat()]["responseSha256"]=="x"
    row=next(r for r in manifest["forecasts"] if r["product"]==product and r["target"]==target.isoformat())
    assert row["candidate"]==row["ablation"]==row["direct"]==105.
    assert row["candidateFallbackReason"]=="insufficient_training"
    missing=next(r for r in manifest["forecasts"] if r["product"]==product and r["origin"]==origin.isoformat() and r["horizon"]==5)
    assert all(missing[m] is None for m in ("candidate","ablation","direct","persistence"))
    pilot.validate_forecasts(manifest)
    row["actual"]=110
    with pytest.raises(pilot.PilotError,match="target_value"):
        pilot.validate_forecasts(manifest)


def test_score_reconciles_source_actuals_and_handles_empty_cells(monkeypatch):
    report,prices,weather,prov,origin,target,product=_fixture()
    def nofit(rows,feature_name,pid,h,fit):
        return None,None,{"product":pid,"horizon":h,"fitDate":fit.isoformat(),"trainingRows":len(rows),"distinctTrainingTargetDates":len({r["target"] for r in rows}),"status":"insufficient_training"}
    monkeypatch.setattr(pilot,"_fit_model",nofit)
    manifest=pilot.build_forecast(report,prices,weather,{},prov)
    scored=pilot.score(manifest,prices,report=report)
    assert scored["coverage"]["rowsWithActual"]==1
    assert scored["macroMetrics"]["candidate"]["mae"] is None
    report["pairedRows"][0]["actual"]=999.
    with pytest.raises(pilot.PilotError,match="price_report_actual_mismatch"):
        pilot.score(manifest,prices,report=report)


def test_manifest_byte_hash_tampering_rejected():
    manifest={"status":"answer_free_forecast_manifest","protocol":pilot.PROTOCOL,"manifestSha256":"0"*64}
    raw=(json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode()
    with pytest.raises(pilot.PilotError,match="manifest_hash"):
        pilot.score(manifest,{"schemaVersion":1,"products":[]},manifest_bytes=raw)
