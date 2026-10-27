import datetime as dt
import hashlib
import json
from unittest.mock import patch

from agri_data import equal_info_online_forecasts as mod


def test_alpha_is_read_only_from_answer_free_selected_ridge():
    raw={"phase":"forecast_only","targetValuesAttached":False,"selection":{
        p:{"ridge":{"chosenConfig":{"alpha":a}}} for p,a in mod.SELECTED_ALPHA.items()}}
    assert mod._alpha_from_manifest(raw)==mod.SELECTED_ALPHA
    raw["selection"]["pepper"]["ridge"]["chosenConfig"]["alpha"]=10
    try:
        mod._alpha_from_manifest(raw)
    except ValueError as exc:
        assert "alpha values changed" in str(exc)
    else:
        raise AssertionError("unexpected alpha was accepted")


def test_origin_month_groups_refit_at_first_eligible_origin():
    # Use two synthetic targets with origins in consecutive months; fit setup
    # is stubbed so this tests the online schedule without training real models.
    targets=[("cabbage",dt.date(2025,10,2),1),("cabbage",dt.date(2025,10,5),2),("cabbage",dt.date(2025,11,2),1)]
    fit_origins=[]
    state_appends=[]
    class Model:
        def predict(self,rows): return [0.0]*len(rows)
    class Fit:
        def append(self,values,refit=False): state_appends.extend(values);return self
        def get_forecast(self,steps): return type("Forecast",(),{"predicted_mean":[0.0]*steps})()
    def fake_period(records,direct_series,product,h,origin,alpha,kind):
        fit_origins.append((product,h,origin,kind,alpha))
        return Model(),{"count":100,"firstTarget":"2014-01-01","lastTarget":"2025-01-01","fitOrigin":origin.isoformat()},None
    with patch.object(mod,"_grid",return_value=targets), \
         patch.object(mod,"_period_fit",side_effect=fake_period), \
         patch.object(mod,"_fit_sarima",return_value=(Fit(),{"trainEnd":"2025-09-30"})), \
         patch.object(mod.ten,"_calendar_values",side_effect=lambda series,start,end:[start]), \
         patch.object(mod.library,"features",return_value=([0.0]*71,100.0,{"p0Date":"2025-10-01"})), \
         patch.object(mod,"_direct_feature",return_value=([0.0]*len(mod.modeling.FEATURE_NAMES),100.0,{"p0Date":"2025-10-01"})):
        records={p:{} for p in mod.PRODUCTS};records["cabbage"]={dt.date(2025,10,1):{"price":100.0}}
        rows=mod._forecast_phase(records,{p:{} for p in mod.PRODUCTS},{},mod.SELECTED_ALPHA,"confirmation",mod.CONF_START,mod.CONF_END)
    assert [(x[2],x[3]) for x in fit_origins if x[0]=="cabbage"]==[
        (dt.date(2025,10,1),"online71"),(dt.date(2025,10,1),"direct"),
        (dt.date(2025,10,3),"online71"),(dt.date(2025,10,3),"direct"),
        (dt.date(2025,11,1),"online71"),(dt.date(2025,11,1),"direct")]
    assert [r["target"] for r in rows]==["2025-10-02","2025-10-05","2025-11-02"]
    assert state_appends==[dt.date(2025,10,1),dt.date(2025,10,2)]
    assert rows[1]["models"]["causalSarima"]["training"]["fitOrigin"]=="2025-10-01"
    assert rows[1]["models"]["causalSarima"]["provenance"]["lastUpdatedDate"]=="2025-10-02"
    assert all(set(r["models"])==set(mod.METHODS) for r in rows)
    assert all(r["models"]["online71Blend"]["status"]=="issued" for r in rows)
    sarima=rows[1]["models"]["causalSarima"]
    assert sarima["provenance"]["lastActualDate"]=="2025-10-01"
    assert sarima["provenance"]["requestedCutoff"]=="2025-10-02"
    assert sarima["provenance"]["lastUpdatedDate"]=="2025-10-02"


def _signed_manifest():
    def make_rows(start,end):
        rows=[]
        for product,day,h in mod._grid(start,end):
            origin=day-dt.timedelta(days=h)
            models={m:{"status":"not_issued","reason":"synthetic","prediction":None,"training":None,"provenance":{}} for m in mod.METHODS}
            rows.append({"product":product,"target":day.isoformat(),"origin":origin.isoformat(),"horizon":h,"eligible":origin>=start,"models":models})
        return rows
    result={"phase":"forecast_only","targetValuesAttached":False,
            "confirmationForecasts":make_rows(mod.CONF_START,mod.CONF_END),
            "evaluationForecasts":make_rows(mod.EVAL_START,mod.EVAL_END)}
    result["manifestSha256"]=hashlib.sha256(mod._canonical(result)).hexdigest()
    return result


def test_manifest_validation_rejects_digest_or_grid_mutation():
    raw=_signed_manifest()
    mod._validate(raw)
    raw["confirmationForecasts"].pop()
    try:
        mod._validate(raw)
    except ValueError:
        pass
    else:
        raise AssertionError("incomplete grid accepted")


def test_blend_never_replaces_missing_ridge_with_sarima():
    sarima={"status":"issued","reason":None,"prediction":20.0}
    absent={"status":"not_issued","reason":"ridge_fit_failed","prediction":None}
    blended=mod._blend_ridge_sarima(absent,sarima)
    assert blended["status"]=="not_issued"
    assert blended["reason"]=="ridge_fit_failed"
    ridge={"status":"issued","reason":None,"prediction":10.0}
    fallback=mod._blend_ridge_sarima(ridge,{"status":"not_issued","reason":"sarima_failed","prediction":None})
    assert fallback["status"]=="issued" and fallback["prediction"]==10.0
    assert fallback["provenance"]["fallback"] is True


def test_direct_feature_uses_each_row_origin_minus_one():
    target=dt.date(2025,10,4);h=2;seen=[]
    series={dt.date(2025,10,1):(100.0,12.0)}
    def feature_row(actual_series,origin,target_arg,horizon,cache):
        seen.append((origin,target_arg,horizon))
        return dt.date(2025,10,1),(1.0,2.0)
    with patch.object(mod.modeling,"_feature_row",side_effect=feature_row):
        vec,source,audit=mod._direct_feature(series,target,h,{})
    assert seen==[(dt.date(2025,10,1),target,h)]
    assert vec==[1.0,2.0] and source==100.0


def test_direct_feature_applies_true_origin_fourteen_day_boundary():
    target=dt.date(2025,1,17);h=1;source_day=dt.date(2025,1,1)
    series={source_day:(100.0,12.0)}
    with patch.object(mod.modeling,"_feature_row",return_value=(source_day,(1.0,2.0))):
        vec,price,audit=mod._direct_feature(series,target,h,{})
    assert vec is None and price is None
    assert audit["reason"]=="p0_older_than_14_days"
    assert audit["p0AgeDays"]==15


def test_latest_source_is_audited_independently_of_feature_issuance():
    records={dt.date(2025,9,1):{},dt.date(2025,9,20):{}}
    assert mod._latest_pre_origin_source(records,dt.date(2025,10,1))==(dt.date(2025,9,20),11)
    assert mod._latest_pre_origin_source({},dt.date(2025,10,1))==(None,None)
