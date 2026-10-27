import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import pytest

from agri_data import dashboard
from agri_data import ten_strategy_benchmark as bench


def test_naive_family_uses_only_origin_information_and_marks_missing_native_values():
    origin=dt.date(2025,9,15); target=origin+dt.timedelta(days=1)
    series={origin-dt.timedelta(days=1):100.,origin-dt.timedelta(days=2):90.}
    vals,meta=bench._baseline_predictions(series,origin,target,.25,100.)
    assert vals["naive"]==100. and vals["sma7"] is None and meta["sma7"]["fallbackReason"]=="fewer_than_7_trade_observations"
    assert vals["weekly_seasonal_naive"] is None or vals["weekly_seasonal_naive"] in series.values()
    assert vals["log_drift_365d"] is not None


def test_calendar_specs_and_d_selection_score_only_observed_validation_days(monkeypatch):
    assert bench._spec("arima",0)==((1,0,1),(0,0,0),"c")
    assert bench._spec("arima",1)==((1,1,1),(0,0,0),"n")
    assert bench._spec("sarima",0)==((1,0,1),(1,0,0,7),"c")
    cutoff=dt.date(2025,9,14); start=cutoff-dt.timedelta(days=59); train_start=cutoff-dt.timedelta(days=729)
    observed={start+dt.timedelta(days=i):100. for i in range(0,60,2)}
    class Result:
        def __init__(self,d): self.d=d; self.appends=[]
        def get_forecast(self,steps): return type("Forecast",(),{"predicted_mean":[math.log(100. if self.d else 90.)]})()
        def append(self,values,refit):
            assert refit is False; self.appends.append(values); return self
    fits=[]
    def fake_fit(series,a,b,kind,d):
        fit=Result(d); fits.append(fit); return fit
    monkeypatch.setattr(bench,"_fit_calendar",fake_fit)
    monkeypatch.setattr(bench,"_calendar_values",lambda s,a,b:[math.log(s[a])] if a in s else [float("nan")])
    audit=bench._choose_difference(observed,"arima",train_start,cutoff)
    assert audit["selectedD"]==1 and audit["validationObservedCount"]==30
    assert audit["byDifference"]["1"]["mape"]==pytest.approx(0.)
    assert len(fits[0].appends)==60 and len(fits[1].appends)==60
    assert any(math.isnan(v[0]) for v in fits[0].appends)


def test_alpha_selection_is_train_only_and_ties_choose_smallest():
    cutoff=dt.date(2025,9,14); start=cutoff-dt.timedelta(days=729)
    series={start+dt.timedelta(days=i):100. for i in range(730)}
    alpha,audit=bench._select_alpha(series,cutoff,start)
    assert alpha==.1 and audit["selectedAlpha"]==.1
    assert set(audit["mapeByAlpha"])=={"0.1","0.25","0.5"}
    series[cutoff+dt.timedelta(days=1)]=1e9
    assert bench._select_alpha(series,cutoff,start)[0]==alpha


def test_ses_without_selected_alpha_is_unavailable_with_specific_fallback_reason():
    origin=dt.date(2025,9,15)
    vals,meta=bench._baseline_predictions({origin-dt.timedelta(days=1):100.},origin,origin+dt.timedelta(days=1),None,100.)
    assert vals["ses"] is None
    assert meta["ses"]["nativeStatus"]=="unavailable"
    assert meta["ses"]["fallbackReason"]=="alpha_or_state_unavailable"


def test_actual_statsmodels_fit_calls_use_arima_and_sarimax_supported_signatures(monkeypatch):
    import statsmodels
    if statsmodels.__version__ != "0.14.0":
        pytest.skip("frozen protocol requires statsmodels 0.14.0")
    monkeypatch.setattr(bench,"MAXITER",1)
    start=dt.date(2024,1,1)
    series={start+dt.timedelta(days=i):100.+i*.1+math.sin(i/5) for i in range(100)}
    for kind in ("arima","sarima"):
        # One iteration commonly cannot converge, but reaching this controlled
        # result proves the real library accepted the respective fit kwargs.
        try:
            bench._fit_calendar(series,start,start+dt.timedelta(days=99),kind,0)
        except ValueError as exc:
            assert str(exc)=="fit_not_converged"


def _bundle(monkeypatch):
    start=dt.date(2025,9,25); end=start+dt.timedelta(days=1); hs=(1,2)
    monkeypatch.setattr(bench,"TARGET_START",start); monkeypatch.setattr(bench,"TARGET_END",end)
    monkeypatch.setattr(bench,"HORIZONS",hs); monkeypatch.setattr(bench,"PLANNED",len(dashboard.PRODUCTS)*2*2)
    monkeypatch.setattr(bench,"CELLS",len(dashboard.PRODUCTS)*2); monkeypatch.setattr(bench,"DM_FAMILY",9*bench.CELLS*2)
    price_doc={"schemaVersion":1,"products":[]}; direct={"protocol":{"id":"direct-horizon-ridge-2026-10-06"},"pairedRows":[]}; hist={"pairedRows":[]}; quant={"pairedRows":[]}
    for p in dashboard.PRODUCTS:
        obs=[]
        begin=start-dt.timedelta(days=40)
        for i in range(43): obs.append({"date":(begin+dt.timedelta(days=i)).isoformat(),"actual":100.+i%5})
        price_doc["products"].append({"id":p,"observations":obs})
        values={dt.date.fromisoformat(r["date"]):r["actual"] for r in obs}
        for t in (start,end):
            for h in hs:
                o=t-dt.timedelta(days=h); actual=values[t]; dr=actual+1.; pers=actual-1.; key={"product":p,"origin":o.isoformat(),"target":t.isoformat(),"horizon":h}
                direct["pairedRows"].append({**key,"actual":actual,"direct":dr,"persistence":pers})
                hist["pairedRows"].append({**key,"actual":actual,"direct_ridge":dr,"persistence":pers,"full_history_histgb":actual+2.})
                quant["pairedRows"].append({**key,"actual":actual,"directRidge":dr,"persistence":pers,"candidatePrediction":actual+3.})
    blobs=[json.dumps(x,sort_keys=True).encode() for x in (price_doc,direct,hist,quant)]
    names=("PRICE_SHA","DIRECT_SHA","HIST_SHA","QUANTILE_SHA")
    hashes=[hashlib.sha256(b).hexdigest() for b in blobs]
    for n,h in zip(names,hashes): monkeypatch.setattr(bench,n,h)
    def fake_fit(series,first_origin,cutoff):
        train_start=cutoff-dt.timedelta(days=bench.TRAIN_DAYS-1)
        return {},{"synthetic":{"status":"unavailable"}},{},.25,train_start
    monkeypatch.setattr(bench,"_fit_all",fake_fit)
    return blobs


def test_forecast_manifest_is_full_answer_free_and_tracks_native_vs_fallback(monkeypatch):
    pb,db,hb,qb=_bundle(monkeypatch)
    manifest=bench.build(pb,db,hb,qb)
    assert len(manifest["forecasts"])==bench.PLANNED
    assert manifest["targetValuesAttached"] is False
    assert all("actual" not in row for row in manifest["forecasts"])
    assert all(set(row["predictions"])==set(bench.METHODS) for row in manifest["forecasts"])
    assert all(row["methodStatus"]["calendar_arima_111"]["fallback"] for row in manifest["forecasts"])
    assert all(row["methodStatus"]["mape_quantile"]["nativeStatus"]=="available" for row in manifest["forecasts"])


def test_score_reconciles_inputs_ranks_50_cells_and_keeps_900_test_family(monkeypatch):
    pb,db,hb,qb=_bundle(monkeypatch); m=bench.build(pb,db,hb,qb)
    result=bench.score(m,pb,db,hb,qb)
    assert result["coverage"]["scoredPairs"]==bench.PLANNED
    assert len(result["cellMetrics"])==bench.CELLS
    assert len(result["rankingByEqual50CellMape"])==10
    assert result["dmTests"]["familySizePerBandwidth"]==bench.DM_FAMILY
    assert len(result["dmTests"]["HAC9"])==bench.DM_FAMILY
    assert all(t["status"].startswith("inference_unavailable") for t in result["dmTests"]["HAC9"])
    assert result["coverage"]["byMethod"]["calendar_arima_111"]["fallback"]==bench.PLANNED
    assert "methodsBelowInAllCells" not in result["internal10PercentReference"]
    assert result["internal10PercentReference"]["methodsBelowMacro10Percent"]==[m for m in bench.METHODS if result["macroMetrics"][m]["meanOf50CellMape"]<10]


def test_scoring_rejects_manifest_prediction_tampering(monkeypatch):
    pb,db,hb,qb=_bundle(monkeypatch); m=bench.build(pb,db,hb,qb)
    m["forecasts"][0]["predictions"]["sma7"]+=7
    with pytest.raises(ValueError,match="manifest_hash_mismatch"):
        bench.score(m,pb,db,hb,qb)


def test_missing_or_invalid_published_quantile_and_histgb_forecasts_abort(monkeypatch):
    pb,db,hb,qb=_bundle(monkeypatch)
    quant=json.loads(qb); quant["pairedRows"].pop()
    qbad=json.dumps(quant,sort_keys=True).encode()
    monkeypatch.setattr(bench,"QUANTILE_SHA",hashlib.sha256(qbad).hexdigest())
    with pytest.raises(ValueError,match="quantile_evaluation_key_set_mismatch"):
        bench.build(pb,db,hb,qbad)
    hist=json.loads(hb); hist["pairedRows"][0]["full_history_histgb"]=None
    hbad=json.dumps(hist,sort_keys=True).encode(); monkeypatch.setattr(bench,"HIST_SHA",hashlib.sha256(hbad).hexdigest()); monkeypatch.setattr(bench,"QUANTILE_SHA",hashlib.sha256(qb).hexdigest())
    with pytest.raises(ValueError,match="source_report_prediction_invalid:full_history_histgb"):
        bench.build(pb,db,hbad,qb)


def test_stale_origin_price_cannot_be_used_for_fixed_grid_fallback(monkeypatch):
    pb,db,hb,qb=_bundle(monkeypatch)
    prices=json.loads(pb)
    prices["products"][0]["observations"]=[r for r in prices["products"][0]["observations"] if r["date"]<="2025-09-01" or r["date"]>="2025-09-25"]
    bad=json.dumps(prices,sort_keys=True).encode(); monkeypatch.setattr(bench,"PRICE_SHA",hashlib.sha256(bad).hexdigest())
    with pytest.raises(ValueError,match="fixed_grid_origin_price_stale"):
        bench.build(bad,db,hb,qb)


def test_build_is_answer_free_and_score_rejects_report_actual_mismatch(monkeypatch):
    pb,db,hb,qb=_bundle(monkeypatch); m=bench.build(pb,db,hb,qb)
    first=m["forecasts"][0]; first["predictions"]["direct_ridge"]+=5
    m["manifestSha256"]=bench._canonical_hash({k:v for k,v in m.items() if k!="manifestSha256"})
    with pytest.raises(ValueError,match="manifest_direct_prediction_mismatch"):
        bench.score(m,pb,db,hb,qb)
    direct=json.loads(db); hist=json.loads(hb); quant=json.loads(qb)
    key=(direct["pairedRows"][0]["product"],direct["pairedRows"][0]["target"],direct["pairedRows"][0]["horizon"])
    direct["pairedRows"][0]["actual"]+=2
    for report,field in ((hist,"actual"),(quant,"actual")):
        row=next(r for r in report["pairedRows"] if (r["product"],r["target"],r["horizon"])==key); row[field]+=2
    blobs=[json.dumps(x,sort_keys=True).encode() for x in (direct,hist,quant)]
    for name,blob in zip(("DIRECT_SHA","HIST_SHA","QUANTILE_SHA"),blobs): monkeypatch.setattr(bench,name,hashlib.sha256(blob).hexdigest())
    # Forecast construction uses report keys and published predictions only.
    monkeypatch.setattr(bench,"DIRECT_SHA",hashlib.sha256(blobs[0]).hexdigest())
    monkeypatch.setattr(bench,"HIST_SHA",hashlib.sha256(blobs[1]).hexdigest())
    monkeypatch.setattr(bench,"QUANTILE_SHA",hashlib.sha256(blobs[2]).hexdigest())
    changed=bench.build(pb,*blobs)
    assert changed["targetValuesAttached"] is False
    with pytest.raises(ValueError,match="pinned_actual_reconciliation_failed"):
        bench.score(changed,pb,*blobs)
