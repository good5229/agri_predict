import datetime as dt
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from agri_data import dashboard
from agri_data import onion_arima_transfer_pilot as pilot


def test_training_window_excludes_evaluation_future_and_is_exact_calendar_window():
    series={pilot.TRAIN_START-dt.timedelta(days=1):10.,pilot.TRAIN_START:11.,pilot.START-dt.timedelta(days=1):12.,pilot.START:999.,pilot.END+dt.timedelta(days=1):10000.}
    assert pilot._training_prices(series)==[(pilot.TRAIN_START,11.),(pilot.START-dt.timedelta(days=1),12.)]


def test_calendar_forecast_and_trade_gap_state_update_use_only_prices_at_origin(monkeypatch):
    start=dt.date(2024,9,1)
    monkeypatch.setattr(pilot,"START",start)
    monkeypatch.setattr(pilot,"dates",lambda: iter([start,start+dt.timedelta(days=1),start+dt.timedelta(days=2)]))
    class State:
        def __init__(self): self.updates=[]; self.calls=0
        def append(self,values,refit):
            assert refit is False; self.updates.extend(values); return self
        def forecast(self,steps):
            assert steps==1; self.calls+=1; return [100+self.calls]
    state=State(); series={start-dt.timedelta(days=1):90.,start:91.,start+dt.timedelta(days=2):93.}
    rows=pilot._walkforward_rows("onion",series,state,{"reason":None},{})
    assert state.updates==[91.,93.]
    assert [r["target"] for r in rows]==["2024-09-02","2024-09-03","2024-09-04"]
    assert [r["candidate"] for r in rows]==[101,102,103]
    assert rows[1]["persistence"]==91 and rows[1]["persistenceSourceDate"]=="2024-09-01"
    assert rows[2]["persistence"]==93 and rows[2]["persistenceSourceDate"]=="2024-09-03"


def test_arima_candidate_nonconvergence_is_not_selected():
    class Nonconverged:
        mle_retvals={"converged":False}; aic=1.; bic=2.
    class Model:
        def __init__(self,*args,**kwargs): pass
        def fit(self,**kwargs):
            assert kwargs["method_kwargs"]=={"maxiter":100}; return Nonconverged()
    supported={"n":120,"status":"d1_supported","level":{},"firstDifference":{}}
    with patch.object(pilot,"_diagnostic",return_value=supported), patch("statsmodels.tsa.arima.model.ARIMA",Model):
        fit,audit=pilot._fit_selected([(dt.date(2024,1,1)+dt.timedelta(days=i),100+i) for i in range(120)])
    assert fit is None and audit["status"]=="unavailable"
    assert audit["reason"]=="no_converged_finite_candidate" and len(audit["candidates"])==9
    assert all(x["reason"]=="not_converged" for x in audit["candidates"])


def _synthetic_bundle(monkeypatch, *, missing_product=None, omit_persistence=None, report_actual_edit=None):
    start=dt.date(2024,9,1); end=start+dt.timedelta(days=30)
    monkeypatch.setattr(pilot,"START",start); monkeypatch.setattr(pilot,"END",end)
    monkeypatch.setattr(pilot,"PLANNED",31*len(dashboard.PRODUCTS))
    prices={"schemaVersion":1,"products":[]}; report={"protocol":{"id":"direct-horizon-ridge-2026-10-06"},"pairedRows":[]}
    for p in dashboard.PRODUCTS:
        obs=[] if p==missing_product else [{"date":(start+dt.timedelta(days=i+1)).isoformat(),"actual":100.+i} for i in range(31)]
        prices["products"].append({"id":p,"observations":obs})
        for item in obs:
            actual=item["actual"]
            if report_actual_edit and p==report_actual_edit[0] and item["date"]==report_actual_edit[1]: actual+=1
            report["pairedRows"].append({"product":p,"target":item["date"],"horizon":1,"actual":actual,"direct":100.})
    pb=json.dumps(prices,sort_keys=True).encode(); db=json.dumps(report,sort_keys=True).encode()
    ph=hashlib.sha256(pb).hexdigest(); dh=hashlib.sha256(db).hexdigest()
    monkeypatch.setattr(pilot,"PRICE_SHA256",ph); monkeypatch.setattr(pilot,"DIRECT_SHA256",dh)
    rows=[]
    for p in dashboard.PRODUCTS:
        for origin in pilot.dates():
            target=origin+dt.timedelta(days=1); has_actual=p!=missing_product
            rows.append({"product":p,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":1,
                "candidate":101.,"direct":100. if has_actual else None,
                "persistence":(99. if has_actual and (p,target.isoformat())!=omit_persistence else None)})
    m={"schemaVersion":1,"protocol":pilot.PROTOCOL,"phase":"forecast_only","targetValuesAttached":False,
       "sourceHashes":{"prices":ph,"directReport":dh},"codeSha256":hashlib.sha256(Path(pilot.__file__).read_bytes()).hexdigest(),"forecasts":rows}
    m["manifestSha256"]=hashlib.sha256(json.dumps(m,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    raw=json.dumps(m,ensure_ascii=False,sort_keys=True).encode()
    return m,pb,db,raw


def test_score_rejects_forecast_tampering_before_scoring(monkeypatch):
    m,pb,db,raw=_synthetic_bundle(monkeypatch); m["forecasts"][0]["candidate"]=999.
    with pytest.raises(ValueError,match="manifest_hash_mismatch"): pilot.score(m,pb,db,raw)


@pytest.mark.parametrize("edit,reason",[(None,"direct_report_actual_missing_or_invalid"),(1,"direct_report_actual_mismatch")])
def test_report_paired_actual_must_reconcile_to_pinned_prices(monkeypatch,edit,reason):
    m,pb,db,raw=_synthetic_bundle(monkeypatch)
    report=json.loads(db); report["pairedRows"][0]["actual"]=edit
    db=json.dumps(report,sort_keys=True).encode(); dh=hashlib.sha256(db).hexdigest()
    monkeypatch.setattr(pilot,"DIRECT_SHA256",dh); m["sourceHashes"]["directReport"]=dh
    m["manifestSha256"]=hashlib.sha256(json.dumps({k:v for k,v in m.items() if k!="manifestSha256"},ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    raw=json.dumps(m,ensure_ascii=False,sort_keys=True).encode()
    with pytest.raises(ValueError,match=reason): pilot.score(m,pb,db,raw)


def test_macro_is_unavailable_if_any_crop_has_no_common_rows_and_gate_fails(monkeypatch):
    m,pb,db,raw=_synthetic_bundle(monkeypatch,missing_product=list(dashboard.PRODUCTS)[-1])
    result=pilot.score(m,pb,db,raw)
    assert result["macroMetrics"]["candidate"]=={"mae":None,"mape":None,"productsWithCommonRows":len(dashboard.PRODUCTS)-1,"available":False}
    assert result["expansionGate"]["macroMapeAtMost10AndHalfPointBetter"] is False
    assert result["expansionGate"]["allPass"] is False


def test_persistence_dm_and_metrics_use_same_three_method_common_rows(monkeypatch):
    start=dt.date(2024,9,1); missing=(list(dashboard.PRODUCTS)[0],(start+dt.timedelta(days=1)).isoformat())
    m,pb,db,raw=_synthetic_bundle(monkeypatch,omit_persistence=missing)
    result=pilot.score(m,pb,db,raw); p=list(dashboard.PRODUCTS)[0]
    assert result["coverage"]["byProduct"][p]["metrics"]["candidate"]["n"]==30
    n_dm={(x["comparator"],x["loss"]):x["n"] for x in result["dmTests"] if x["product"]==p and x["bandwidth"]==9}
    assert n_dm[("direct","AE")]==30 and n_dm[("persistence","AE")]==30
    assert all(x["n"]==30 for x in result["comparisonConclusions"] if x["product"]==p)


def test_full_grid_forecast_coverage_and_comparison_conclusions_are_reported(monkeypatch):
    m,pb,db,raw=_synthetic_bundle(monkeypatch); result=pilot.score(m,pb,db,raw)
    assert result["coverage"]["byMethod"]["candidate"]["planned"]==pilot.PLANNED
    assert result["coverage"]["byMethod"]["candidate"]["withForecast"]==pilot.PLANNED
    assert len(result["comparisonConclusions"])==len(dashboard.PRODUCTS)*2*2
    assert all(x["conclusion"] in {"improvement","worsening","inconclusive","inference_unavailable"} for x in result["comparisonConclusions"])
    assert result["macroMetrics"]["candidate"]["mae"] is not None


def test_nonfinite_information_criteria_are_null_with_reason_and_json_safe():
    class Result:
        mle_retvals={"converged":True}; aic=float("inf"); bic=3.
    class Model:
        def __init__(self,*args,**kwargs): pass
        def fit(self,**kwargs): return Result()
    supported={"n":120,"status":"d1_supported","level":{},"firstDifference":{}}
    with patch.object(pilot,"_diagnostic",return_value=supported),patch("statsmodels.tsa.arima.model.ARIMA",Model):
        fit,audit=pilot._fit_selected([(dt.date(2024,1,1)+dt.timedelta(days=i),100+i) for i in range(120)])
    assert fit is None
    assert audit["candidates"][0]["aic"] is None
    assert audit["candidates"][0]["reason"]=="nonfinite_information_criterion"
    json.dumps(audit,allow_nan=False)


def test_nonfinite_stationarity_diagnostic_is_null_and_ineligible(monkeypatch):
    from statsmodels.tsa import stattools
    monkeypatch.setattr(stattools,"adfuller",lambda *a,**k:(float("inf"),float("nan"),1,100,{},0.))
    monkeypatch.setattr(stattools,"kpss",lambda *a,**k:(float("inf"),float("nan"),1,{}))
    train=[(dt.date(2023,1,1)+dt.timedelta(days=i),100.+i) for i in range(100)]
    audit=pilot._diagnostic(train)
    assert audit["status"]=="diagnostic_nonfinite"
    assert audit["level"]["adfStatistic"] is None and audit["level"]["adfPValue"] is None
    json.dumps(audit,allow_nan=False)
