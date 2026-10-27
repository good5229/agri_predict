import datetime as dt
import hashlib
import json
import math
import sys
from types import SimpleNamespace

import pytest

from agri_data import ridge_sarima_component_cv as cv


def _canon(obj):
    return cv._canonical(obj)


def _fixture(monkeypatch):
    product = "cabbage"
    monkeypatch.setattr(cv, "PRODUCTS", (product,))
    monkeypatch.setattr(cv, "HORIZONS", (1,))
    monkeypatch.setattr(cv, "PLANNED_HIST", 60)
    monkeypatch.setattr(cv, "PLANNED_TEST", 30)
    baseline_rows=[]
    monkeypatch.setattr(cv, "_source_hashes", lambda *args: {"prices": "p", "direct": "d", "tenManifest": "tm",
        "tenReport": "tr", "fixedManifest": "fm", "fixedReport": "fr", "historicalBaseline":"hb"})
    monkeypatch.setattr(cv, "_code_sha", lambda: "code")

    hist_targets = [cv.B1_START + dt.timedelta(days=i) for i in range(30)] + [cv.B2_START + dt.timedelta(days=i) for i in range(30)]
    test_targets = [cv.TEST_START + dt.timedelta(days=i) for i in range(30)]
    targets = hist_targets + test_targets
    actuals = {day: 100. + (i % 3) for i, day in enumerate(targets)}
    direct_rows = {}
    ten_rows = {}
    fixed_rows = {}
    components = []
    for i, target in enumerate(targets):
        origin = target - dt.timedelta(days=1)
        key = (product, origin, target, 1)
        actual = actuals[target]
        direct_rows[key] = {"actual": actual if target not in test_targets else "MUST_NOT_READ", "direct": 100., "persistence": 99.}
        ten_rows[key] = {"predictions": {"direct_ridge": 100., "naive": 99., "calendar_sarima_111_100_7":100.},
                         "nativePredictions": {"calendar_sarima_111_100_7": 100.}}
        fixed_rows[key] = {"predictions": {"blend": 100., "direct_ridge": 100., "persistence": 99.}}
        ridge = {"1": 100. + (i % 4), "10": 100., "100": 101.}
        sarima = {name: 100. + offset for name, offset in zip(cv.SARIMA_KEYS, (3., 2., 1., 0.))}
        sarima["d1_w730"] = 100.
        components.append({"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                           "horizon": 1, "period": "test" if target in test_targets else "selection",
                           "ridge": ridge, "sarima": sarima, "sarimaStatus": {}})
        if target not in test_targets:
            baseline_rows.append({"product":product,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":1,
                                  "direct":100.,"sarima":100.,"status":"available"})
    monkeypatch.setattr(cv, "_maps", lambda *args: (direct_rows, {}, ten_rows, {}, {}, fixed_rows, {}))
    price_series = {product: actuals}
    monkeypatch.setattr(cv, "_price_series", lambda raw: price_series)
    price_raw = b"prices"; direct_raw = b"direct"; ten_raw = b"ten"; ten_report_raw = b"ten-report"
    fixed_raw = b"fixed"; fixed_report_raw = b"fixed-report"
    baseline={"schemaVersion":1,"protocol":"ridge-sarima-historical-weight-cv-2026-10-08","phase":"historical_components_only",
              "targetValuesAttached":False,"inputSha256":{"prices":cv.PRICES_SHA,"direct":cv.DIRECT_SHA},
              "audits":{"cabbage":{"selectedD":1}},"forecasts":baseline_rows}
    baseline["manifestSha256"]=_canon(baseline)
    baseline_raw=(json.dumps(baseline,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode()
    baseline_sha=hashlib.sha256(baseline_raw).hexdigest()
    monkeypatch.setattr(cv,"HISTORICAL_BASELINE_SHA",baseline_sha)
    forecast = {"schemaVersion": 1, "protocol": cv.PROTOCOL, "phase": "components_forecast_only",
                "targetValuesAttached": False, "inputSha256": {"prices": "p", "direct": "d", "tenManifest": "tm",
                "tenReport": "tr", "fixedManifest": "fm", "fixedReport": "fr","historicalBaseline":"hb"}, "codeSha256": "code",
                "forecasts": components}
    forecast["manifestSha256"] = _canon(forecast)
    forecast_raw = (json.dumps(forecast, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    monkeypatch.setattr(cv, "build_forecasts", lambda *args: forecast)
    return forecast_raw, (price_raw, direct_raw, ten_raw, ten_report_raw, fixed_raw, fixed_report_raw,baseline_raw), direct_rows, actuals, ten_rows, fixed_rows


def test_phase_two_freezes_test_predictions_without_reading_test_truth(monkeypatch):
    forecast_raw, sources, direct_rows, actuals, ten_rows, fixed_rows = _fixture(monkeypatch)
    selected = cv.build_selection(forecast_raw, *sources)
    assert selected["phase"] == "selected_test_forecast_only"
    assert selected["targetValuesAttached"] is False
    assert len(selected["forecasts"]) == 30
    assert all("actual" not in row for row in selected["forecasts"])
    assert selected["forecasts"][0]["predictions"]["a"] == pytest.approx(100.)
    assert selected["forecasts"][0]["predictions"]["fixed50"] == pytest.approx(100.)


def test_score_reconstructs_frozen_manifest_and_reports_unavailable_small_sample_dm(monkeypatch):
    forecast_raw, sources, direct_rows, actuals, ten_rows, fixed_rows = _fixture(monkeypatch)
    selected = cv.build_selection(forecast_raw, *sources)
    for i in range(30):
        target=cv.TEST_START+dt.timedelta(days=i)
        direct_rows["cabbage", target-dt.timedelta(days=1), target, 1]["actual"]=actuals[target]
    prices_raw, direct_raw, ten_raw, ten_report_raw, fixed_raw, fixed_report_raw, _historical_baseline_raw = sources
    actual_values=[actuals[cv.TEST_START+dt.timedelta(days=i)] for i in range(30)]
    # Scoring verifies report lineage and checks the published source-cell summaries.
    ten_report,fixed_report=_source_reports(actual_values)
    monkeypatch.setattr(cv, "_maps", lambda *args: (direct_rows, {}, ten_rows, ten_report, {}, fixed_rows, fixed_report))
    report = cv.score(selected_raw(selected), forecast_raw, *sources)
    assert report["coverage"]["scoredPairs"] == 30
    assert report["coverage"]["commonSixMethodPairs"] == 30
    assert report["dmTests"]["familySizePerBandwidth"] == 300
    assert all(t["n"] == 30 for t in report["dmTests"]["HAC9"])


def test_score_common_six_method_sample_excludes_missing_sarima_but_reports_full_grid(monkeypatch):
    forecast_raw,sources,direct_rows,actuals,ten_rows,fixed_rows=_fixture(monkeypatch)
    selected=cv.build_selection(forecast_raw,*sources)
    for i in range(30):
        target=cv.TEST_START+dt.timedelta(days=i)
        direct_rows["cabbage",target-dt.timedelta(days=1),target,1]["actual"]=actuals[target]
    row=selected["forecasts"][0]
    row["predictions"]["sarima"]=None
    row["predictions"]["a"]=row["predictions"]["fixed50"]
    row["predictions"]["b"]=row["predictions"]["fixed50"]
    row["lineage"].update(sarimaNative=False,componentFallback=True,fallbackReason="synthetic_missing_sarima")
    selected["manifestSha256"]=cv._canonical({k:v for k,v in selected.items() if k!="manifestSha256"})
    monkeypatch.setattr(cv,"build_selection",lambda *args:selected)
    actual_values=[actuals[cv.TEST_START+dt.timedelta(days=i)] for i in range(30)]
    ten_report,fixed_report=_source_reports(actual_values)
    monkeypatch.setattr(cv,"_maps",lambda *args:(direct_rows,{},ten_rows,ten_report,{},fixed_rows,fixed_report))
    report=cv.score(selected_raw(selected),forecast_raw,*sources)
    cell=report["cellMetrics"][0]
    assert report["coverage"]["commonSixMethodPairs"]==29
    assert report["coverage"]["adoptionPairs"]==30
    assert cell["metrics"]["a"]["n"]==30
    assert cell["sixMethodCommonMetrics"]["a"]["n"]==cell["sixMethodCommonMetrics"]["sarima"]["n"]==29
    assert cell["fullGridDiagnostics"]["a"]["n"]==30
    assert cell["fullGridDiagnostics"]["sarima"]["n"]==29
    assert report["dmTests"]["HAC9"][0]["n"]==30
    assert report["expansionGate"]["a"]["coverageKeysMatchFixed50"] is True


def test_historical_sarima_baseline_mismatch_is_rejected(monkeypatch):
    forecast_raw,sources,*_=_fixture(monkeypatch)
    baseline=json.loads(sources[-1])
    baseline["forecasts"][0]["sarima"]=baseline["forecasts"][0]["sarima"]+5.
    baseline["manifestSha256"]=_canon({k:v for k,v in baseline.items() if k!="manifestSha256"})
    bad_raw=(json.dumps(baseline,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode()
    monkeypatch.setattr(cv,"HISTORICAL_BASELINE_SHA",hashlib.sha256(bad_raw).hexdigest())
    with pytest.raises(ValueError,match="historical_baseline_sarima_mismatch"):
        cv.build_selection(forecast_raw,*sources[:-1],bad_raw)


def test_direct_ridge_fit_survives_broken_threadpoolctl_with_monthly_cutoff(monkeypatch):
    monkeypatch.setattr(cv,"PRODUCTS",("cabbage",));monkeypatch.setattr(cv,"HORIZONS",(1,))
    monkeypatch.setattr(cv.direct,"EVAL_START",dt.date(2024,2,1));monkeypatch.setattr(cv.direct,"EVAL_END",dt.date(2024,3,31))
    monkeypatch.setattr(cv.direct,"MIN_TRAIN_DAYS",30)
    class Estimator:
        def set_params(self,**kwargs):return self
        def fit(self,x,y):self.mean=sum(y)/len(y);self.samples=len(y);return self
        def predict(self,x):return [self.mean]
    monkeypatch.setattr(cv.modeling,"_new_estimator",lambda _model:Estimator())
    def broken_threadpool_limits(*,limits):raise RuntimeError("threadpool introspection failed")
    monkeypatch.setitem(sys.modules,"threadpoolctl",SimpleNamespace(threadpool_limits=broken_threadpool_limits))
    start=dt.date(2024,1,1);series={"cabbage":{start+dt.timedelta(days=i):(100.+i,.0+1000.) for i in range(91)}}
    target=dt.date(2024,2,10);origin=target-dt.timedelta(days=1);key=("cabbage",origin,target,1)
    predictions,audits=cv._direct_fits(series,{key},10.)
    assert predictions[key][0] is not None
    audit=audits[("cabbage",1,2024,2)]
    assert audit["status"]=="fitted"
    assert audit["fitOrigin"]=="2024-02-01"
    assert audit["fitSamples"]==31
    assert audit["threadpoolLimitFallback"].startswith("RuntimeError:threadpool introspection failed")


def test_historical_ridge_fit_failure_is_distinct_from_numeric_mismatch(monkeypatch):
    forecast_raw,sources,*_=_fixture(monkeypatch)
    components=json.loads(forecast_raw)["forecasts"]
    components[0]["ridge"]["10"]=None
    components[0]["ridgeStatus"]={"10":"ridge_fit_failed"}
    with pytest.raises(ValueError,match="historical_baseline_ridge_fit_unavailable:ridge_fit_failed"):
        cv._verify_historical_baseline(components,sources[-1])


def selected_raw(manifest):
    return (json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()


def _source_reports(actual_values):
    def metric(prediction):
        n=len(actual_values)
        return {"n":n,"mae":sum(abs(y-prediction) for y in actual_values)/n,
                "mape":100*sum(abs(y-prediction)/y for y in actual_values)/n}
    perfect=metric(100.);persistence=metric(99.)
    cell={"product":"cabbage","horizon":1,"metrics":{"direct_ridge":perfect,
        "calendar_sarima_111_100_7":perfect,"naive":persistence,"persistence":persistence,"blend":perfect}}
    return {"cellMetrics":[cell]},{"cellMetrics":[cell]}


def test_component_phase_rejects_answer_attached_even_when_rehashed(monkeypatch):
    forecast_raw, sources, *_ = _fixture(monkeypatch)
    doc = json.loads(forecast_raw)
    doc["forecasts"][0]["actual"] = 100.
    doc["manifestSha256"] = _canon({k: v for k, v in doc.items() if k != "manifestSha256"})
    raw = (json.dumps(doc, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode()
    with pytest.raises(ValueError, match="phase1_answer_leakage"):
        cv.build_selection(raw, *sources)


def test_a_selection_marks_partial_component_configuration_ineligible(monkeypatch):
    monkeypatch.setattr(cv, "PRODUCTS", ("cabbage",))
    monkeypatch.setattr(cv, "HORIZONS", (1,))
    rows=[]
    for i in range(60):
        target=cv.B1_START+dt.timedelta(days=i) if i<30 else cv.B2_START+dt.timedelta(days=i-30)
        row={"product":"cabbage","target":target,"origin":target-dt.timedelta(days=1),"horizon":1,"actual":100.+i%3}
        for a in cv.RIDGE_KEYS:row[f"ridge{a}"]=100.
        for s in cv.SARIMA_KEYS:row[f"sarima{s}"]=100.
        rows.append(row)
    rows[0]["sarimad0_w365"]=None
    selected=cv._select_a(rows)
    failed=next(r for r in selected["candidateResults"] if r["d"]==0 and r["trainingCalendarDays"]==365)
    assert failed["eligible"] is False
    assert failed["ineligibleReason"]=="sarima_fit_or_native_coverage_incomplete"


def test_b_positive_shrink_refits_weights_on_adopted_components(monkeypatch):
    monkeypatch.setattr(cv,"PRODUCTS",("cabbage",))
    monkeypatch.setattr(cv,"HORIZONS",(1,2,3))
    rows=[]
    for period_start in (cv.B1_START,cv.B2_START):
        for i in range(30):
            target=period_start+dt.timedelta(days=i)
            for horizon in (1,2,3):
                row={"product":"cabbage","origin":target-dt.timedelta(days=horizon),"target":target,"horizon":horizon,"actual":100.}
                for alpha in cv.RIDGE_KEYS:row[f"ridge{alpha}"]=100.
                for sarima in cv.SARIMA_KEYS:row[f"sarima{sarima}"]=120.
                rows.append(row)
    result=cv._select_b(rows,{"alpha":10.,"d":1,"trainingCalendarDays":730})
    group=result["groups"]["cabbage|1-3"]
    assert result["adoptedShrink"]>0
    assert group["rawWeight"]==pytest.approx(1.)
    assert group["testWeight"]==pytest.approx(.75)
    assert result["refitB1B2Weights"]["cabbage|1-3"]["pairs"]==180


@pytest.mark.parametrize("first_origin,window",[(dt.date(2024,9,15),365),(dt.date(2025,9,15),730)])
def test_sarima_fit_cutoff_is_period_wide_with_delayed_product_key(monkeypatch,first_origin,window):
    delayed=first_origin+dt.timedelta(days=5)
    append_log={"cabbage":[],"radish":[]};fit_calls=[]
    class Fit:
        def __init__(self,product):self.product=product
        def append(self,values,refit=False):
            append_log[self.product].append(values);return self
        def get_forecast(self,steps):return SimpleNamespace(predicted_mean=[math.log(100.)]*steps)
    def fit_calendar(series,start,end,kind,d):
        product="cabbage" if first_origin in series else "radish";fit_calls.append((product,start,end))
        return Fit(product)
    monkeypatch.setattr(cv.ten,"_fit_calendar",fit_calendar)
    monkeypatch.setattr(cv.ten,"_calendar_values",lambda series,start,end:start)
    series={"cabbage":{first_origin:100.},"radish":{delayed:100.}}
    keys=[("cabbage",first_origin,first_origin+dt.timedelta(days=1),1),
          ("radish",delayed,delayed+dt.timedelta(days=1),1)]
    predictions,audits=cv._sarima_config(series,keys,d=1,window=window)
    expected_end=first_origin-dt.timedelta(days=1)
    assert {end for _,_,end in fit_calls}=={expected_end}
    assert {audit["periodFirstOrigin"] for audit in audits.values()}=={first_origin.isoformat()}
    assert predictions[keys[0]][0]==pytest.approx(100.)
    assert predictions[keys[1]][0]==pytest.approx(100.)
    assert append_log["radish"][-1]==delayed


def test_calendar_dm_counts_actual_target_dates_and_holm_family():
    dates = [(dt.date(2025, 1, 1) + dt.timedelta(days=40*i), -1. if i % 2 else 1.) for i in range(30)]
    result = cv._dm(dates, 9)
    assert result["n"] == 30
    assert result["status"] == "tested"
    tests = [{"pValue": p, "status": "tested", "product": "cabbage", "horizon": 1,
              "candidate": "a", "comparator": "fixed50", "loss": "AE"} for p in (.01, .04, .5)]
    cv._holm(tests)
    assert tests[0]["holmPValue"] == pytest.approx(.03)
