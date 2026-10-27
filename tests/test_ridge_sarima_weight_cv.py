import datetime as dt
import hashlib
import json
import math
from types import SimpleNamespace

import pytest

from agri_data import multiscale_residual as stats
from agri_data import ridge_sarima_weight_cv as cv
from agri_data import ten_strategy_benchmark as ten


class FakeFit:
    def append(self, values, refit=False):
        return self
    def get_forecast(self, steps):
        return SimpleNamespace(predicted_mean=[math.log(100.)] * steps)


def _fixture(monkeypatch):
    product = "cabbage"
    monkeypatch.setattr(cv, "PRODUCTS", (product,))
    monkeypatch.setattr(cv, "HORIZONS", (1,))
    monkeypatch.setattr(stats, "PRODUCTS", (product,))
    monkeypatch.setattr(cv, "MIN_CELL_PAIRS", 1)
    monkeypatch.setattr(cv, "TEST_COUNT", 1)
    monkeypatch.setattr(cv, "HIST_START", dt.date(2024, 9, 25))
    monkeypatch.setattr(cv, "FIRST_HIST_ORIGIN", dt.date(2024, 9, 24))
    monkeypatch.setattr(cv, "CUTOFF", dt.date(2024, 9, 23))
    monkeypatch.setattr(ten, "_price_series", lambda raw: {product: prices})
    monkeypatch.setattr(ten, "_choose_difference", lambda *args: {"selectedD": 0, "status": "selected"})
    monkeypatch.setattr(ten, "_fit_calendar", lambda *args: FakeFit())
    monkeypatch.setattr(ten, "_calendar_values", lambda *args: None)

    prices = {}
    day = dt.date(2022, 9, 1)
    while day <= dt.date(2025, 9, 25):
        i = (day - dt.date(2022, 9, 1)).days
        prices[day] = 120. + math.sin(i / 13.)
        day += dt.timedelta(days=1)
    hist = []
    for i in range(30):
        target = dt.date(2024, 9, 25) + dt.timedelta(days=10 * i)
        origin = target - dt.timedelta(days=1)
        hist.append({"product": product, "target": target.isoformat(), "horizon": 1,
                     "direct": 150., "persistence": 120., "actual": prices[target]})
    test_target = cv.TEST_START
    test_origin = test_target - dt.timedelta(days=1)
    direct_row = {"product": product, "target": test_target.isoformat(), "horizon": 1,
                  "direct": 150., "persistence": 120., "actual": prices[test_target]}
    direct_raw = json.dumps({"pairedRows": hist + [direct_row]}).encode()
    prices_raw = json.dumps({"schemaVersion": 1, "products": []}).encode()
    ten_forecast = {"product": product, "origin": test_origin.isoformat(), "target": test_target.isoformat(),
        "horizon": 1,"predictions":{"direct_ridge":150.,"calendar_sarima_111_100_7":100.,"naive":120.},
        "nativePredictions":{"calendar_sarima_111_100_7":100.},
        "methodStatus":{"calendar_sarima_111_100_7":{"fallback":False,"fallbackReason":None}},"fallback":{}}
    ten_doc={"protocol":"ten-strategy-same-pair-2026-10-08","phase":"forecast_only","targetValuesAttached":False,
             "manifestSha256":"ten-internal","forecasts":[ten_forecast]}
    fixed_forecast={"product":product,"origin":test_origin.isoformat(),"target":test_target.isoformat(),"horizon":1,
       "predictions":{"blend":125.,"direct_ridge":150.,"calendar_sarima_111_100_7":100.,"persistence":120.},
       "lineage":{"sarimaFallback":False,"sarimaFallbackReason":None,"sarimaNativePrediction":100.,
                  "blendRule":"direct fallback if SARIMA fallback; else arithmetic 0.5/0.5"}}
    fixed_doc={"protocol":"ridge-sarima-fixed-50-50-2026-10-08","phase":"forecast_only","targetValuesAttached":False,
               "sourceManifestSha256":"ten-hash","manifestSha256":"fixed-internal","forecasts":[fixed_forecast]}
    ten_raw=json.dumps(ten_doc).encode();fixed_raw=json.dumps(fixed_doc).encode()
    ten_doc["manifestSha256"]="ten-internal";fixed_doc["manifestSha256"]="fixed-internal"
    test_key=(product,test_origin,test_target,1)
    ten_rows={test_key:ten_forecast};fixed_rows={test_key:fixed_forecast}
    monkeypatch.setattr(cv,"_source_manifests",lambda *args:(ten_doc,ten_rows,fixed_doc,fixed_rows))
    price_hash=hashlib.sha256(prices_raw).hexdigest();direct_hash=hashlib.sha256(direct_raw).hexdigest()
    ten_hash=hashlib.sha256(ten_raw).hexdigest();fixed_hash=hashlib.sha256(fixed_raw).hexdigest()
    actual=prices[test_target]
    def metric(prediction):
        return {"n":1,"mae":abs(actual-prediction),"mape":100*abs(actual-prediction)/actual}
    ten_metrics={"direct_ridge":metric(150.),"calendar_sarima_111_100_7":metric(100.),"naive":metric(120.)}
    ten_report=json.dumps({"protocol":ten.PROTOCOL,"manifestSha256":"ten-internal",
        "inputSha256":{"direct":direct_hash,"prices":price_hash},
        "cellMetrics":[{"product":product,"horizon":1,"metrics":ten_metrics}]}).encode()
    ten_report_hash=hashlib.sha256(ten_report).hexdigest()
    fixed_metrics={**ten_metrics,"persistence":metric(120.),"blend":metric(125.)}
    fixed_report=json.dumps({"protocol":"ridge-sarima-fixed-50-50-2026-10-08","manifestSha256":"fixed-internal",
        "inputSha256":{"directReport":direct_hash,"tenStrategyManifest":ten_hash,"tenStrategyReport":ten_report_hash},
        "cellMetrics":[{"product":product,"horizon":1,"metrics":fixed_metrics}]}).encode()
    for name,value in (("PRICES_SHA",price_hash),("DIRECT_SHA",direct_hash),("TEN_MANIFEST_SHA",ten_hash),
                       ("TEN_REPORT_SHA",ten_report_hash),("FIXED_MANIFEST_SHA",fixed_hash),
                       ("FIXED_REPORT_SHA",hashlib.sha256(fixed_report).hexdigest())):
        monkeypatch.setattr(cv,name,value)
    return prices_raw,direct_raw,ten_raw,ten_report,fixed_raw,fixed_report,prices


def test_phase_a_is_answer_free_and_rebuilds_calendar_components(monkeypatch):
    prices_raw,direct_raw,*rest=_fixture(monkeypatch)
    doc=cv.build_historical(prices_raw,direct_raw)
    assert doc["phase"]=="historical_components_only"
    assert doc["audits"]["cabbage"]["trainEnd"]==cv.CUTOFF.isoformat()
    assert doc["audits"]["cabbage"]["trainStart"]==(cv.CUTOFF-dt.timedelta(days=729)).isoformat()
    assert len(doc["forecasts"])==30
    assert all("actual" not in row for row in doc["forecasts"])
    assert all(row["sarima"]==pytest.approx(100.) for row in doc["forecasts"])
    bad=json.loads(direct_raw)
    for row in bad["pairedRows"]:row["actual"]="unreadable-phase-a-label"
    badraw=json.dumps(bad).encode();monkeypatch.setattr(cv,"DIRECT_SHA",hashlib.sha256(badraw).hexdigest())
    assert len(cv.build_historical(prices_raw,badraw)["forecasts"])==30


def test_global_weight_selection_and_ties(monkeypatch):
    _fixture(monkeypatch)
    rows=[{"product":"cabbage","horizon":1,"actual":100.,"direct":150.,"sarima":100.}]
    weight,scores=cv._weight_scores(rows)
    assert weight==0.
    assert len(scores)==5
    equal=[{"product":"cabbage","horizon":1,"actual":100.,"direct":100.,"sarima":100.}]
    assert cv._weight_scores(equal)[0]==.5


def test_phase_b_saves_weight_and_answer_free_test_manifest_then_scores(monkeypatch):
    prices_raw,direct_raw,ten_raw,ten_report,fixed_raw,fixed_report,prices=_fixture(monkeypatch)
    historical=cv.build_historical(prices_raw,direct_raw)
    historical_raw=(json.dumps(historical,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode()
    selected=cv.build_selection(historical_raw,prices_raw,direct_raw,ten_raw,fixed_raw)
    assert selected["selection"]["selectedWeight"]==.5
    assert len(selected["forecasts"])==1 and "actual" not in selected["forecasts"][0]
    test_raw=(json.dumps(selected,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode()
    result=cv.score(test_raw,historical_raw,prices_raw,direct_raw,ten_raw,ten_report,fixed_raw,fixed_report)
    assert result["coverage"]["commonScoredPairs"]==1
    assert len(result["dmTests"])==4*1*1*2*2
    assert all(x["status"]=="inference_unavailable_insufficient_pairs" for x in result["dmTests"])


@pytest.mark.parametrize("tamper",["candidate","weight","extra","missing"])
def test_phase_c_rejects_rehashed_test_manifest_forgery(monkeypatch,tamper):
    prices_raw,direct_raw,ten_raw,ten_report,fixed_raw,fixed_report,_=_fixture(monkeypatch)
    hist=cv.build_historical(prices_raw,direct_raw)
    hist_raw=(json.dumps(hist,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode()
    manifest=cv.build_selection(hist_raw,prices_raw,direct_raw,ten_raw,fixed_raw)
    if tamper=="candidate":manifest["forecasts"][0]["predictions"]["candidate"]+=1
    elif tamper=="weight":manifest["selection"]["selectedWeight"]=.25
    elif tamper=="extra":manifest["forecasts"].append(dict(manifest["forecasts"][0]))
    else:manifest["forecasts"].pop()
    manifest.pop("manifestSha256");manifest["manifestSha256"]=cv._canonical(manifest)
    raw=(json.dumps(manifest,sort_keys=True,ensure_ascii=False,indent=2)+"\n").encode()
    with pytest.raises(ValueError):cv.score(raw,hist_raw,prices_raw,direct_raw,ten_raw,ten_report,fixed_raw,fixed_report)


def test_calendar_hac_numeric_scale_and_holm_family():
    rows=[(dt.date(2025,1,1)+dt.timedelta(days=40*i),0. if i%2==0 else 2.) for i in range(100)]
    assert stats._dm(rows,9)["statistic"]==pytest.approx(10.)
    tests=[{"pValue":p,"status":"tested","meanLossDifference":-1.,"product":"cabbage","horizon":1,
            "comparator":"direct","loss":"mae"} for p in (.01,.04,.5)]
    stats._holm(tests)
    assert tests[0]["holmPValue"]==pytest.approx(.03)
    assert sum(x["conclusion"]=="improvement" for x in tests)==1
