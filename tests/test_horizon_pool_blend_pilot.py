import copy
import datetime as dt

import pytest

from agri_data import horizon_pool_blend_pilot as pilot


def fixture(missing=None, actual=100.0):
    prices={"schemaVersion":1,"products":[]}
    report={"schemaVersion":1,"protocol":{"id":"direct-horizon-ridge-2026-10-06"},"pairedRows":[]}
    for pi,p in enumerate(pilot.PRODUCTS):
        obs=[]
        start=pilot.START-dt.timedelta(days=20)
        for i in range(130):
            d=start+dt.timedelta(days=i)
            obs.append({"date":d.isoformat(),"actual":actual,"volumeKg":1000+pi})
        prices["products"].append({"id":p,"observations":obs})
        for i in range(85):
            origin=pilot.START+dt.timedelta(days=i)
            for h in pilot.HORIZONS:
                if missing==(p,origin.isoformat(),h):continue
                target=origin+dt.timedelta(days=h)
                report["pairedRows"].append({"product":p,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":h,
                    "actual":actual,"direct":actual*1.02,"monthlyPooledRidge":actual*1.0,"persistence":actual})
    return prices,report


def test_forecast_is_exact_answer_free_50_50_manifest():
    prices,report=fixture()
    forecast=pilot.build_forecast(report,{"directReport":"r","code":"c"})
    assert len(forecast["forecasts"])==1275
    assert all("actual" not in r for r in forecast["forecasts"])
    row=forecast["forecasts"][0]
    assert row["candidate"]==pytest.approx(.5*row["direct"]+.5*row["monthlyPooledRidge"])
    assert row["candidate"]==pytest.approx(101.0)
    # Report answers are ignored in the answer-free forecast stage.
    original=[r.copy() for r in forecast["forecasts"]]
    for source in report["pairedRows"]:source["actual"]=-1e99
    assert pilot.build_forecast(report)["forecasts"]==original


def test_missing_direct_source_keeps_the_planned_row_and_scores_common_sample():
    p=pilot.PRODUCTS[0]; missing=(p,pilot.START.isoformat(),1)
    prices,report=fixture(missing=missing)
    forecast=pilot.build_forecast(report)
    row=next(r for r in forecast["forecasts"] if r["product"]==p and r["origin"]==missing[1] and r["horizon"]==1)
    assert row["direct"] is row["candidate"] is None
    assert row["missingSourceReason"]=="no_direct_report_pair"
    assert forecast["missingCounts"]["candidate"]["no_direct_report_pair"]==1
    scored=pilot.score(forecast,prices,report)
    assert scored["fourWayCommonRows"]==1274
    assert len(scored["dm"]["comparisons"])==90
    assert scored["coverage"]["forecastValidRows"]["candidate"]==1274
    assert scored["byProductHorizon"][f"{p}/1"]["n"]==84


def test_duplicate_identity_same_length_and_answer_key_are_rejected():
    prices,report=fixture();forecast=pilot.build_forecast(report)
    duplicate=copy.deepcopy(forecast);duplicate["forecasts"][2]=copy.deepcopy(duplicate["forecasts"][1])
    with pytest.raises(pilot.PilotError,match="duplicate"):
        pilot.score(duplicate,prices,report)
    leaked=copy.deepcopy(forecast);leaked["forecasts"][0]["actual"]=100
    with pytest.raises(pilot.PilotError,match="answer field"):
        pilot.score(leaked,prices,report)


def test_score_checks_report_prediction_provenance_and_actual_price_agreement():
    prices,report=fixture();forecast=pilot.build_forecast(report)
    tampered=copy.deepcopy(forecast);tampered["forecasts"][0]["monthlyPooledRidge"]+=1
    tampered["forecasts"][0]["candidate"]=.5*tampered["forecasts"][0]["direct"]+.5*tampered["forecasts"][0]["monthlyPooledRidge"]
    with pytest.raises(pilot.PilotError,match="differs from frozen report"):
        pilot.score(tampered,prices,report)
    report["pairedRows"][0]["actual"]+=1
    with pytest.raises(pilot.PilotError,match="does not agree"):
        pilot.score(forecast,prices,report)


def test_gate_metrics_use_three_horizon_then_five_product_macro_and_thresholds():
    prices,report=fixture();forecast=pilot.build_forecast(report);result=pilot.score(forecast,prices,report)
    assert len(result["byProductHorizon"])==15
    assert result["macroMetrics"]["candidate"]["mape"]==pytest.approx(1.0)
    assert result["macroMetrics"]["monthlyPooledRidge"]["mape"]==pytest.approx(0.0)
    assert result["macroMetrics"]["direct"]["mape"]==pytest.approx(2.0)
    assert result["gates"]["all15CellsCommonPairsAtLeast30"]
    assert result["gates"]["candidateCoverageAtLeastDirect"]
    assert result["gates"]["candidateFiveProductMacroMapeAtMost10Pct"]
    assert result["gates"]["candidateMapeImprovesVsDirectByAtLeast0_5PctPoints"]
    assert not result["gates"]["candidateMapeImprovesVsPooledByAtLeast0_5PctPoints"]
    assert not result["gates"]["expand"]
