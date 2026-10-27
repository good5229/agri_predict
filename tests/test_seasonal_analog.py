import datetime as dt
import json

import pytest

from agri_data import dashboard, seasonal_analog as sa


def _inputs(days=30):
    start,end=dt.date(2022,1,1),dt.date(2026,10,1)
    dates=[start+dt.timedelta(days=i) for i in range((end-start).days+1)]
    products=[]; report_rows=[]; maps={}
    for pi,p in enumerate(dashboard.PRODUCTS):
        prices={d:100+pi*10+i*.025+3*__import__('math').sin(i/70) for i,d in enumerate(dates)}
        maps[p]=prices
        products.append({"id":p,"observations":[{"date":d.isoformat(),"actual":v,"volumeKg":1000} for d,v in prices.items()]})
        targets=[dt.date(2024,9,25)+dt.timedelta(days=i) for i in range(days)]
        targets += [dt.date(2025,9,25)+dt.timedelta(days=i) for i in range(days)]
        for target in targets:
            for h in sa.HORIZONS:
                origin=target-dt.timedelta(days=h); actual=prices[target]
                report_rows.append({"product":p,"origin":origin.isoformat(),"target":target.isoformat(),"horizon":h,
                    "actual":actual,"direct":actual*1.04,"persistence":prices[origin]})
    return {"schemaVersion":1,"products":products},{"protocol":{"id":"direct-horizon-ridge-2026-10-06"},"pairedRows":report_rows},maps


def test_frozen_selection_test_coverage_and_dm_family():
    prices,report,_=_inputs()
    result=sa.build(prices,report)
    assert result["selection"] and set(result["selection"])==set(dashboard.PRODUCTS)
    assert len(result["comparisons"])==200
    assert all(len(row["dm"])==2 for row in result["comparisons"])
    assert all("holmPValue" in row["dm"]["9"] and "holmPValue" in row["dm"]["28"] for row in result["comparisons"])
    assert all(row["n"]==30 for row in result["comparisons"])
    assert len(result["pairedRows"])==5*10*30
    assert len(result["selectionRows"])==5*10*30
    assert all(row["origin"]<=row["target"] for row in result["selectionRows"])
    assert all(value["candidateCoverageNondeclining"] for value in result["coverage"].values())
    first=result["pairedRows"][0]
    assert first["analogYearCount"]==3
    assert [a["target"] for a in first["analogYears"]]==[
        (dt.date.fromisoformat(first["target"])-dt.timedelta(days=364*k)).isoformat() for k in (1,2,3)]
    assert all(dt.date.fromisoformat(a["sourceDate"])<=dt.date.fromisoformat(a["origin"]) for a in first["analogYears"])
    assert first["mix25"]==pytest.approx(.75*first["directRidge"]+.25*first["analogPrice"])
    json.dumps(result,allow_nan=False)


def test_no_eligible_analog_falls_back_to_direct_ridge():
    origin=dt.date(2025,9,24); target=origin+dt.timedelta(days=1)
    series={origin:(100.,500.),target:(101.,500.)}
    row=sa._pair_row("onion",origin,target,1,{"actual":101.,"directRidge":105.,"persistence":100.},series)
    assert row["analogYearCount"]==0
    assert row["fallbackReason"]=="no_eligible_historical_analog"
    assert row["mix25"]==row["mix50"]==105.


def test_missing_current_source_is_explicit_ridge_fallback():
    origin,target=dt.date(2025,9,24),dt.date(2025,9,25)
    series={origin-dt.timedelta(days=15):(90.,1.)}
    row=sa._pair_row("onion",origin,target,1,{"actual":101.,"directRidge":105.,"persistence":90.},series)
    assert row["currentSourceDate"] is None
    assert row["fallbackReason"]=="current_source_missing_or_stale"
    assert row["mix25"]==row["mix50"]==105.


def test_clamps_median_log_change_and_rejects_persistence_source_mismatch():
    origin,target=dt.date(2025,9,24),dt.date(2025,9,25)
    series={origin:(100.,1.),target:(100.,1.)}
    for k in (1,2,3):
        po=origin-dt.timedelta(days=364*k); pt=target-dt.timedelta(days=364*k)
        series[po]=(100.,1.); series[pt]=(1000.,1.)
    row=sa._pair_row("onion",origin,target,1,{"actual":100.,"directRidge":105.,"persistence":100.},series)
    assert row["rawMedianLogChange"]>0.2
    assert row["clampedLogChange"]==0.2
    with pytest.raises(ValueError,match="persistence/source mismatch"):
        sa._pair_row("onion",origin,target,1,{"actual":100.,"directRidge":105.,"persistence":99.},series)


def test_selection_embargo_and_invalid_duplicate_identity():
    assert sa.FIRST_TEST_ORIGIN==dt.date(2025,9,15)
    assert sa.SELECTION_END<=sa.FIRST_TEST_ORIGIN-dt.timedelta(days=14)
    prices,report,_=_inputs(days=1)
    report["pairedRows"].append(dict(report["pairedRows"][0]))
    with pytest.raises(ValueError,match="duplicate forecast identity"):
        sa.build(prices,report)


def test_immutable_writer_refuses_overwrite(tmp_path):
    path=tmp_path/"audit.json"; path.write_bytes(b"existing")
    with pytest.raises(FileExistsError): sa._write_immutable(path,b"replacement")
    assert path.read_bytes()==b"existing"
