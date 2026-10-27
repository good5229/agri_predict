import datetime as dt
import pytest

from agri_data import cumulative_temperature_pilot as pilot


def test_features_are_strictly_pre_origin_and_missing_weather_stays_missing():
    origin=dt.date(2024,8,8)
    weather={s:{origin-dt.timedelta(days=i):15.0 for i in range(1,61)} for s in pilot.STATIONS}
    vectors,_=pilot.weather_vectors(weather,start=origin,end=origin)
    assert vectors[origin]["ablation"]==[15.0]*3
    assert vectors[origin]["candidate"]==[15.0,150.0,300.0]*3
    # An observation on the origin cannot fill the preceding-day gap.
    weather[pilot.STATIONS[0]][origin]=99.0
    del weather[pilot.STATIONS[0]][origin-dt.timedelta(days=1)]
    vectors,audit=pilot.weather_vectors(weather,start=origin,end=origin)
    assert vectors[origin]["candidate"] is None
    assert audit[origin.isoformat()]["candidateAvailable"] is False
    assert audit[origin.isoformat()]["missingWindows"]["100/7"]==1


def test_forecast_train_rows_stop_at_cutoff_and_insufficient_cell_falls_back():
    # Exercise the train-only selection directly with one post-cutoff high target.
    series={p:{} for p in pilot.PRODUCTS}
    rows=[]
    start=dt.date(2024,1,1)
    for i in range(100):
        target=start+dt.timedelta(days=i); series["pepper"][target]=100+i
        rows.append({"product":"pepper","target":target,"direct":100.0,"features":[float(i),1.0]})
    rows.append({"product":"pepper","target":pilot.TRAIN_END+dt.timedelta(days=1),"direct":1.0,"features":[1e9,1e9]})
    filtered=[r for r in rows if r["target"]<=pilot.TRAIN_END]
    scaler,model,audit=pilot._fit_model(filtered,series,"candidate")
    assert audit["distinctTrainingTargetDates"]==100
    assert max(r["target"] for r in filtered)==start+dt.timedelta(days=99)
    assert model is not None and scaler is not None
    # The frozen threshold is distinct days, never row count.
    _,_,small=pilot._fit_model(rows[:1],series,"candidate")
    assert small["status"]=="insufficient_training"


def test_manifest_grid_and_answer_tamper_rejected():
    rows=[]
    for origin in pilot.dates(pilot.EVAL_START,pilot.EVAL_END):
        for product in pilot.PRODUCTS:
            for h in pilot.HORIZONS:
                rows.append({"product":product,"origin":origin.isoformat(),"target":(origin+dt.timedelta(days=h)).isoformat(),"horizon":h,
                    "candidate":10.0,"ablation":10.0,"direct":10.0,"persistence":10.0})
    manifest={"forecasts":rows}
    pilot.validate_forecasts(manifest)
    rows[0]["actual"]=10.0
    with pytest.raises(pilot.PilotError,match="target_value"):
        pilot.validate_forecasts(manifest)


def test_manifest_identity_tampering_rejected():
    rows=[]
    for origin in pilot.dates(pilot.EVAL_START,pilot.EVAL_END):
        for product in pilot.PRODUCTS:
            for h in pilot.HORIZONS:
                rows.append({"product":product,"origin":origin.isoformat(),"target":(origin+dt.timedelta(days=h)).isoformat(),"horizon":h,
                    "candidate":10.0,"ablation":10.0,"direct":10.0,"persistence":10.0})
    rows[-1]["target"]=(dt.date.fromisoformat(rows[-1]["target"])+dt.timedelta(days=1)).isoformat()
    with pytest.raises(pilot.PilotError,match="grid_or_date"):
        pilot.validate_forecasts({"forecasts":rows})


def test_manifest_hash_tampering_rejected_before_scoring():
    import json
    manifest={"status":"answer_free_forecast_manifest","protocol":pilot.PROTOCOL,"manifestSha256":"0"*64}
    raw=(json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode()
    with pytest.raises(pilot.PilotError,match="manifest_hash"):
        pilot.score(manifest,{"schemaVersion":1,"products":[]},manifest_bytes=raw)


def _tiny_build_inputs():
    origin=pilot.EVAL_START; h=1; target=origin+dt.timedelta(days=h); product=pilot.PRODUCTS[0]
    report={"protocol":{"id":"direct-horizon-ridge-2026-10-06"},"pairedRows":[{"product":product,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":h,"direct":100.0}]}
    prices={"schemaVersion":1,"products":[{"id":p,"observations":[]} for p in pilot.PRODUCTS]}
    prices["products"][0]["observations"]=[{"date":origin.isoformat(),"actual":99.0},{"date":target.isoformat(),"actual":101.0}]
    provenance={s:{origin-dt.timedelta(days=1):{"requestId":17,"jobId":3,"epochId":"epoch-x","responseSha256":"abc","rowNo":2,"retrievedAt":"2024-08-01T00:00:00Z"}} for s in pilot.STATIONS}
    return report,prices,origin,target,product,provenance


def test_build_path_accepts_provenance_and_keeps_selected_weather_mapping():
    report,prices,origin,target,product,provenance=_tiny_build_inputs()
    manifest=pilot.build_forecast(report,prices,{s:{} for s in pilot.STATIONS},{"prices":"p","directReport":"r"},provenance)
    selected=manifest["weatherSource"]["selectedRecordsByStationDate"]
    ref=selected[pilot.STATIONS[0]][(origin-dt.timedelta(days=1)).isoformat()]
    assert ref=={"requestId":17,"jobId":3,"epochId":"epoch-x","responseSha256":"abc","rowNo":2,"retrievedAt":"2024-08-01T00:00:00Z"}
    row=next(r for r in manifest["forecasts"] if r["product"]==product and r["target"]==target.isoformat())
    assert row["direct"]==100.0 and row["candidate"]==100.0 and row["ablation"]==100.0


def test_absent_direct_pair_requires_all_methods_missing():
    report,prices,origin,target,product,provenance=_tiny_build_inputs()
    manifest=pilot.build_forecast(report,prices,{s:{} for s in pilot.STATIONS},{},provenance)
    row=next(r for r in manifest["forecasts"] if r["product"]==product and r["target"]==target.isoformat())
    row["direct"]=None; row["directMissingReason"]="direct_prediction_missing"
    row["candidate"]=row["ablation"]=None
    with pytest.raises(pilot.PilotError,match="direct_report_pair_absent"):
        row["persistence"]=100.0
        pilot.validate_forecasts(manifest)


def test_score_keeps_observed_rows_when_report_pair_is_absent_and_handles_empty_cells():
    report,prices,origin,target,product,provenance=_tiny_build_inputs()
    other_target=origin+dt.timedelta(days=5)
    report["pairedRows"]=[{"product":product,"origin":origin.isoformat(),"target":other_target.isoformat(),"horizon":5,"direct":100.0}]
    manifest=pilot.build_forecast(report,prices,{s:{} for s in pilot.STATIONS},{},provenance)
    result=pilot.score(manifest,prices,report=report)
    coverage=result["coverage"]
    assert coverage["rowsWithActual"]==1
    assert coverage["forecastCoverageAmongTargetObservedRows"]["candidate"]==0
    assert coverage["forecastCoverageFractionAmongTargetObservedRows"]["candidate"]==0.0
    assert result["macroMetrics"]["candidate"]["mae"] is None
    assert result["productGates"][product]["equalHorizonMaeWorseningPct"] is None


def test_missing_horizon_does_not_clear_persistence_for_another_horizon():
    report,prices,origin,target,product,provenance=_tiny_build_inputs()
    prices["products"][0]["observations"].append({"date":(origin-dt.timedelta(days=1)).isoformat(),"actual":98.0})
    h5_target=origin+dt.timedelta(days=5)
    report["pairedRows"]=[{"product":product,"origin":origin.isoformat(),"target":h5_target.isoformat(),"horizon":5,"direct":100.0}]
    prices["products"][0]["observations"].append({"date":h5_target.isoformat(),"actual":102.0})
    manifest=pilot.build_forecast(report,prices,{s:{} for s in pilot.STATIONS},{},provenance)
    h1=next(r for r in manifest["forecasts"] if r["product"]==product and r["origin"]==origin.isoformat() and r["horizon"]==1)
    h5=next(r for r in manifest["forecasts"] if r["product"]==product and r["origin"]==origin.isoformat() and r["horizon"]==5)
    assert h1["direct"] is None and h1["persistence"] is None
    assert h5["direct"]==100.0 and h5["persistence"]==98.0
    assert h5["persistenceSourceDate"]==(origin-dt.timedelta(days=1)).isoformat()
