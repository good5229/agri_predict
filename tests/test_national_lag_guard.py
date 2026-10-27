import datetime as dt
import copy
import math

import pytest

from agri_data import national_lag_guard as lag


def _source(prices):
    return {"observations":{"radish":{d.isoformat():{"price":str(v),"priceReason":None} for d,v in prices.items()}}}


def test_raw_update_uses_exact_o_minus_one_and_fixed_clip():
    origin=dt.date(2026,1,10);p0day=origin-dt.timedelta(days=2);p1day=origin-dt.timedelta(days=1)
    prices={"radish":{p0day:100.0,p1day:120.0,origin:9999.0}}
    result=lag._raw_update(80.0,"radish",origin,prices,_source(prices["radish"]))
    assert result["p0"]=={"date":p0day.isoformat(),"price":100.0}
    assert result["p1Date"]==p1day.isoformat()
    assert result["clippedLogChange"]==pytest.approx(.10)
    assert result["rawUpdate"]==pytest.approx(80*math.exp(.05))


def test_missing_exact_p1_does_not_backfill_from_origin_day():
    origin=dt.date(2026,1,10);prices={"radish":{origin-dt.timedelta(days=2):100.0,origin:150.0}}
    result=lag._raw_update(80.0,"radish",origin,prices,_source(prices["radish"]))
    assert result["rawUpdate"]==80.0
    assert result["p1"] is None
    assert result["p1Date"]==(origin-dt.timedelta(days=1)).isoformat()
    assert result["reason"]=="p1_absent_exact_o_minus_1"


def test_history_gate_uses_only_targets_through_origin_minus_two(monkeypatch):
    origin=dt.date(2026,5,10);h=5;upper=origin-dt.timedelta(days=2);lower=origin-dt.timedelta(days=57)
    all_rows={};prices={"radish":{}}
    for i in range(25):
        target=upper-dt.timedelta(days=i);prior_origin=target-dt.timedelta(days=h)
        key=("radish",prior_origin.isoformat(),target.isoformat(),h)
        all_rows[key]={"product":"radish","origin":prior_origin.isoformat(),"target":target.isoformat(),"horizon":h,
                       "predictions":{"fixed_blend_50_50":110.0}}
        prices["radish"][target]=100.0
    for target in (lower-dt.timedelta(days=1),upper+dt.timedelta(days=1),origin+dt.timedelta(days=1)):
        prior_origin=target-dt.timedelta(days=h);key=("radish",prior_origin.isoformat(),target.isoformat(),h)
        all_rows[key]={"product":"radish","origin":prior_origin.isoformat(),"target":target.isoformat(),"horizon":h,
                       "predictions":{"fixed_blend_50_50":110.0}}
        prices["radish"][target]=100.0
    monkeypatch.setattr(lag,"_raw_update",lambda b,p,o,pr,s,pd=None:{"rawUpdate":100.0,"p0":{"date":"2026-01-01","price":100.0},"p1":90.0,"reason":None})
    gate=lag._history_gate("radish",h,origin,all_rows,prices,_source(prices["radish"]))
    assert gate["historyPairCount"]==25
    assert gate["historyMinTargetDate"]==(upper-dt.timedelta(days=24)).isoformat()
    assert gate["historyMaxTargetDate"]==upper.isoformat()
    assert gate["active"] is True


def _synthetic_manifest():
    start=dt.date(2025,10,1);end=dt.date(2026,9,30);prices={p:{} for p in lag.PRODUCTS};observations={p:{} for p in lag.PRODUCTS};forecasts=[]
    for offset in range((end-start).days+1):
        day=start+dt.timedelta(days=offset)
        for i,p in enumerate(lag.PRODUCTS):
            price=100.0+i+offset*.01;prices[p][day]=price
            observations[p][day.isoformat()]={"price":str(price),"priceReason":None}
    for offset in range((end-start).days+1):
        target=start+dt.timedelta(days=offset)
        for p in lag.PRODUCTS:
            for h in lag.HORIZONS:
                origin=target-dt.timedelta(days=h);cutoff=origin-dt.timedelta(days=2)
                forecasts.append({"product":p,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":h,"cutoff":cutoff.isoformat(),
                    "predictions":{"fixed_blend_50_50":100.0,"ridge_alpha_100":100.0,"calendar_sarima":100.0,"persistence":100.0},"lineage":{}})
    source={"observations":observations}
    dev={"forecasts":[r for r in forecasts if r["target"]<"2026-07-01"]};fresh={"forecasts":[r for r in forecasts if r["target"]>="2026-07-01"]}
    doc=lag._build_manifest(prices,source,{"synthetic":True},dev,fresh)
    return doc


@pytest.fixture(scope="module")
def synthetic_manifest():
    return _synthetic_manifest()


def test_manifest_is_answer_free_and_records_counterfactual_provenance(synthetic_manifest):
    doc=synthetic_manifest
    lag.verify_manifest(doc)
    assert doc["plannedForecastCount"]==10920
    assert doc["counterfactual"] is True and doc["oMinusOneAvailabilityVerified"] is False
    assert all("actual" not in row and "targetValue" not in row for row in doc["forecasts"])
    row=next(r for r in doc["forecasts"] if r["product"]=="radish" and r["target"]=="2026-07-01" and r["horizon"]==5)
    assert row["lineage"]["gate"]["historyMaxTargetDate"]<=row["lineage"]["gate"]["historyEnd"]
    assert row["lineage"]["p1"]["date"]==(dt.date(2026,7,1)-dt.timedelta(days=6)).isoformat()


def test_manifest_content_hash_detects_forecast_tampering(synthetic_manifest):
    doc=copy.deepcopy(synthetic_manifest);doc["forecasts"][0]["predictions"][lag.METHOD]=123.0
    with pytest.raises(ValueError,match="content hash"):
        lag.verify_manifest(doc)


def test_replay_mismatch_fails_before_scoring(synthetic_manifest,monkeypatch):
    doc=synthetic_manifest
    rebuilt=copy.deepcopy(doc);rebuilt["forecasts"][0]["predictions"][lag.METHOD]+=1
    rebuilt["manifestSha256"]=lag._hash({k:v for k,v in rebuilt.items() if k!="manifestSha256"})
    monkeypatch.setattr(lag,"build_manifest",lambda bundles:rebuilt)
    with pytest.raises(ValueError,match="full lag-guard replay mismatch"):
        lag._verify_replay(doc,{})


def test_period_report_has_the_frozen_480_comparison_family():
    d=dt.date(2026,4,1);origin=d-dt.timedelta(days=5);row={"product":"radish","origin":origin.isoformat(),"target":d.isoformat(),"horizon":5,"cutoff":(origin-dt.timedelta(days=2)).isoformat(),
        "predictions":{lag.METHOD:102.0,lag.RAW_METHOD:103.0,**{m:100.0 for m in lag.BASELINES}},
        "lineage":{"p1":{"state":"positive_price"},"gate":{"active":True},"guardedUsesRawUpdate":True}}
    actuals={p:{} for p in lag.PRODUCTS};actuals["radish"][d]=100.0
    report=lag._period_report([row],actuals,{})
    assert report["holmFamilySizePerBandwidth"]==480
    assert len(report["dmComparisons"])==960


@pytest.mark.parametrize("bandwidth",lag.BANDWIDTHS)
def test_constant_decimal_dm_differences_are_unavailable(bandwidth):
    diffs={dt.date(2026,1,1)+dt.timedelta(days=i):0.1 for i in range(40)}
    result=lag._dm(diffs,bandwidth)
    assert result["status"]=="zero_or_invalid_hac_variance"
    assert result["pValue"]==1.0
    assert result["statistic"] is None


def _complete_period_for_gate():
    metrics={m:{"mae":1.0,"mape":1.0} for m in (lag.METHOD,lag.RAW_METHOD,*lag.BASELINES)}
    cells={f"{p}/{h}":{"n":30,"metrics":copy.deepcopy(metrics)} for p in lag.PRODUCTS for h in lag.HORIZONS}
    dm=[]
    for p in lag.PRODUCTS:
        for h in lag.HORIZONS:
            for loss in ("mae","mape"):
                for bw in lag.BANDWIDTHS:
                    dm.append({"product":p,"horizon":h,"comparator":"fixed_blend_50_50","loss":loss,"family":"guarded_vs_comparators",
                        "dm":{"calendarHacMaxLagDays":bw,"conclusion":"inconclusive"}})
    coverage={m:{"issued":1} for m in (lag.METHOD,*lag.BASELINES)}
    return {"complete40Cells":True,"macro40EqualCell":copy.deepcopy(metrics),"coverage":coverage,"cells":cells,"dmComparisons":dm}


def test_score_path_runs_predeclared_gate_without_name_error(synthetic_manifest,monkeypatch):
    monkeypatch.setattr(lag,"_verify_replay",lambda doc,bundles:None)
    monkeypatch.setattr(lag,"_load_all",lambda bundles:({},None,None,None,None))
    monkeypatch.setattr(lag.base,"_prices",lambda source:{})
    monkeypatch.setattr(lag.base,"_coverage_status_evidence",lambda fresh:{})
    monkeypatch.setattr(lag,"_period_report",lambda *args:_complete_period_for_gate())
    report=lag.score_manifest(synthetic_manifest,{"fresh":{}})
    assert report["periods"]
    first_gate=report["periods"][f"{lag.WINDOWS[1][0].isoformat()}_{lag.WINDOWS[1][1].isoformat()}"]["predeclaredGate"]
    assert first_gate["jointImprovedCells"]==0
    assert first_gate["criteria"]["atLeastOneCellJointImprovementVsFixed50"] is False
