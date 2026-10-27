import datetime as dt
import json
import math
import pytest

from agri_data import bagged_forest_pilot as pilot


def _fixture(start=dt.date(2023,9,1),end=pilot.EVAL_END):
    obs=[]
    for i in range((end-start).days+1):
        day=start+dt.timedelta(days=i)
        obs.append({"date":day.isoformat(),"actual":100.0+0.05*i,"volumeKg":1000.0+i})
    prices={"schemaVersion":1,"products":[{"id":p,"observations":list(obs)} for p in pilot.PRODUCTS]}
    origin=pilot.EVAL_START; h=1; target=origin+dt.timedelta(days=h); product=pilot.PRODUCTS[0]
    actual=next(float(r["actual"]) for r in obs if r["date"]==target.isoformat())
    report={"protocol":{"id":"direct-horizon-ridge-2026-10-06"},"pairedRows":[{"product":product,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":h,"direct":actual-2,"actual":actual}]}
    return report,prices,origin,target,product


def test_monthly_training_uses_strict_730_day_targets_and_same_rows(monkeypatch):
    import sklearn.ensemble,sklearn.linear_model
    fit=dt.date(2025,9,1); first=fit-dt.timedelta(days=pilot.TRAIN_WINDOW_DAYS)
    series={first+dt.timedelta(days=i):(100+i*.1,1000+i) for i in range(pilot.TRAIN_WINDOW_DAYS)}
    fitted_rows=[]
    class FakeForest:
        def __init__(self,**kwargs): self.kwargs=kwargs
        def fit(self,X,y): fitted_rows.append(("forest",len(X))); return self
        def predict(self,X): return [0.0]*len(X)
    class FakeRidge:
        def __init__(self,**kwargs): self.kwargs=kwargs
        def fit(self,X,y): fitted_rows.append(("ridge",len(X))); return self
        def predict(self,X): return [0.0]*len(X)
    monkeypatch.setattr(sklearn.ensemble,"RandomForestRegressor",FakeForest)
    monkeypatch.setattr(sklearn.linear_model,"Ridge",FakeRidge)
    fit_pair,rows,audit=pilot._train_month(series,1,fit)
    assert audit["trainingWindow"]==[first.isoformat(),(fit-dt.timedelta(days=1)).isoformat(),"lower_and_upper_inclusive"]
    assert max(r["target"] for r in rows)<fit
    assert all(r["origin"]<fit and r["sourceDate"]<=r["origin"] for r in rows)
    assert audit["sameTrainingRows"] is True
    assert fitted_rows==[("forest",len(rows)),("ridge",len(rows))]
    assert fit_pair is not None


def test_monthly_fit_below_90_target_days_is_unavailable():
    fit=dt.date(2025,9,1); first=fit-dt.timedelta(days=pilot.TRAIN_WINDOW_DAYS)
    series={first+dt.timedelta(days=i):(100+i,1000+i) for i in range(80)}
    models,rows,audit=pilot._train_month(series,1,fit)
    assert models is None
    assert audit["distinctTrainingTargetDates"]<90
    assert audit["forestStatus"]==audit["ridgeStatus"]=="insufficient_training"


def test_fit_date_selects_correct_month_and_missing_direct_means_all_methods_missing(monkeypatch):
    report,prices,origin,target,product=_fixture()
    def no_fit(series,h,fit_date):
        return None,None,{"fitDate":fit_date.isoformat(),"trainingRows":0,"distinctTrainingTargetDates":0,"sameTrainingRows":True,"forestStatus":"insufficient_training","ridgeStatus":"insufficient_training"}
    monkeypatch.setattr(pilot,"_train_month",no_fit)
    manifest=pilot.build_forecast(report,prices,{"prices":"p","directReport":"r"})
    assert len(manifest["forecasts"])==1365
    row=next(r for r in manifest["forecasts"] if r["product"]==product and r["origin"]==origin.isoformat() and r["horizon"]==1)
    assert row["fitDate"]=="2025-09-01"
    assert row["candidate"]==row["ablation"]==row["direct"]
    missing=next(r for r in manifest["forecasts"] if r["product"]==product and r["origin"]==origin.isoformat() and r["horizon"]==5)
    assert all(missing[m] is None for m in ("candidate","ablation","direct","persistence"))
    pilot.validate_forecasts(manifest)


def test_new_predictions_use_source_price_times_exp_log_ratio(monkeypatch):
    report,prices,origin,target,product=_fixture()
    class FakeForest:
        def predict(self,X): return [math.log(1.1)]
    class FakeScaler:
        def transform(self,X): return X
    class FakeRidge:
        def predict(self,X): return [math.log(.9)]
    def fit(series,h,fit_date):
        return (FakeForest(),(FakeScaler(),FakeRidge())),[],{"fitDate":fit_date.isoformat(),"trainingRows":100,"distinctTrainingTargetDates":100,"sameTrainingRows":True,"forestStatus":"fitted","ridgeStatus":"fitted"}
    monkeypatch.setattr(pilot,"_train_month",fit)
    manifest=pilot.build_forecast(report,prices,{"prices":"p","directReport":"r"})
    row=next(r for r in manifest["forecasts"] if r["product"]==product and r["origin"]==origin.isoformat() and r["horizon"]==1)
    assert row["candidate"]==pytest.approx(row["sourcePrice"]*1.1)
    assert row["ablation"]==pytest.approx(row["sourcePrice"]*.9)


def test_forecast_manifest_answer_and_hash_tampering_rejected(monkeypatch):
    report,prices,_,_,_=_fixture()
    manifest={"status":"answer_free_forecast_manifest","protocol":pilot.PROTOCOL,"manifestSha256":"0"*64}
    raw=(json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode()
    with pytest.raises(pilot.PilotError,match="manifest_hash"):
        pilot.score(manifest,prices,manifest_bytes=raw)
    def no_fit(series,h,fit_date): return None,None,{"fitDate":fit_date.isoformat(),"trainingRows":0,"distinctTrainingTargetDates":0,"sameTrainingRows":True,"forestStatus":"insufficient_training","ridgeStatus":"insufficient_training"}
    monkeypatch.setattr(pilot,"_train_month",no_fit)
    m=pilot.build_forecast(report,prices,{})
    m["forecasts"][0]["actual"]=1
    with pytest.raises(pilot.PilotError,match="target_value"):
        pilot.validate_forecasts(m)


def test_score_reconciles_actuals_and_returns_null_for_empty_cells(monkeypatch):
    report,prices,_,_,product=_fixture()
    def no_fit(series,h,fit_date): return None,None,{"fitDate":fit_date.isoformat(),"trainingRows":0,"distinctTrainingTargetDates":0,"sameTrainingRows":True,"forestStatus":"insufficient_training","ridgeStatus":"insufficient_training"}
    monkeypatch.setattr(pilot,"_train_month",no_fit)
    manifest=pilot.build_forecast(report,prices,{"prices":"p","directReport":"r"})
    result=pilot.score(manifest,prices,report=report)
    assert result["coverage"]["rowsWithActual"]==1285
    assert result["macroMetrics"]["candidate"]["mae"] is None
    assert result["productGates"][product]["equalHorizonMaeWorseningPct"] is None
    row=next(r for r in manifest["forecasts"] if r["product"]==product and r["target"]==report["pairedRows"][0]["target"])
    original=row["direct"]; row["direct"]=original+1; row["candidate"]+=1; row["ablation"]+=1
    with pytest.raises(pilot.PilotError,match="forecast_direct_report_value_mismatch"):
        pilot.score(manifest,prices,report=report)
    row["direct"]=original; row["candidate"]=row["ablation"]=original
    report["pairedRows"][0]["actual"]+=50
    with pytest.raises(pilot.PilotError,match="price_report_actual_mismatch"):
        pilot.score(manifest,prices,report=report)
