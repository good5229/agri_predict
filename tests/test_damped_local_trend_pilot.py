import datetime as dt
import json
import math
import pytest

from agri_data import damped_local_trend_pilot as pilot


def test_pairwise_median_slope_and_input_window_are_strictly_pre_origin():
    origin=dt.date(2023,9,1)
    obs={origin-dt.timedelta(days=i):100*math.exp(.01*(28-i)) for i in range(1,29)}
    # Outlier remains in-window, but pairwise median remains robust.
    obs[origin-dt.timedelta(days=10)]*=4
    obs[origin]=1e200
    result,audit=pilot.local_trend(obs,origin)
    assert result is not None
    assert audit["observationCount"]==28
    assert audit["status"]=="eligible"
    assert abs(result[2]-.01)<.003
    assert max(dt.date.fromisoformat(d) for d in audit["observationDates"])<origin


@pytest.mark.parametrize("kind,reason",[("few","fewer_than_10_dates"),("stale","latest_observation_older_than_14_days")])
def test_trend_eligibility_failures(kind,reason):
    origin=dt.date(2023,9,1)
    if kind=="few": obs={origin-dt.timedelta(days=i):100. for i in range(1,10)}
    else: obs={origin-dt.timedelta(days=15+i):100. for i in range(10)}
    result,audit=pilot.local_trend(obs,origin)
    assert result is None
    assert audit["reason"]==reason


def test_slope_and_log_corrections_clip_at_frozen_limits():
    origin=dt.date(2023,9,1)
    obs={origin-dt.timedelta(days=i):math.exp(10*i) for i in range(1,29)}
    trend,audit=pilot.local_trend(obs,origin)
    assert trend is not None and abs(trend[2])==pilot.SLOPE_CLIP
    # Output correction is clipped even when distant undamped projection is much larger.
    p=trend[1]*math.exp(pilot.CORRECTION_CLIP)
    assert p>0


def _fixture():
    origin=pilot.EVAL_START; product=pilot.PRODUCTS[0]
    prices={"schemaVersion":1,"products":[]}; rows=[]
    obs=[]
    for i in range(1,29):
        day=origin-dt.timedelta(days=i); obs.append({"date":day.isoformat(),"actual":100*math.exp(.01*(28-i))})
    target=origin+dt.timedelta(days=1)
    obs.append({"date":target.isoformat(),"actual":105.0})
    for p in pilot.PRODUCTS:
        prices["products"].append({"id":p,"observations":obs if p==product else []})
    paired=[]
    for h in pilot.HORIZONS:
        t=origin+dt.timedelta(days=h)
        paired.append({"product":product,"origin":origin.isoformat(),"target":t.isoformat(),"horizon":h,"direct":100.0,"actual":105.0})
    report={"protocol":{"id":"direct-horizon-ridge-2026-10-06"},"pairedRows":paired}
    return report,prices,origin,product


def test_blended_candidate_and_ablation_use_damped_and_undamped_horizon_sums():
    report,prices,origin,product=_fixture()
    manifest=pilot.build_forecast(report,prices,{"prices":"p","directReport":"r"})
    row=next(r for r in manifest["forecasts"] if r["product"]==product and r["origin"]==origin.isoformat() and r["horizon"]==5)
    slope=row["trendAudit"]["slope"]
    gap=row["trendForecastGapDays"]
    source=row["trendAudit"]["sourcePrice"]
    expected_damped=sum(pilot.PHI**j for j in range(1,gap+1))
    cand=source*math.exp(max(-.08,min(.08,slope*expected_damped)))
    undamped=source*math.exp(max(-.08,min(.08,slope*gap)))
    assert row["candidate"]==pytest.approx(.5*100+.5*cand)
    assert row["ablation"]==pytest.approx(.5*100+.5*undamped)
    assert row["persistence"]==source
    assert row["persistenceSourceDate"]==row["trendAudit"]["sourceDate"]


def test_full_manifest_is_target_free_and_absent_direct_pair_is_all_missing():
    report,prices,origin,product=_fixture()
    report["pairedRows"]=report["pairedRows"][:1]
    manifest=pilot.build_forecast(report,prices,{"prices":"p","directReport":"r"})
    assert len(manifest["forecasts"])==1365
    assert all("actual" not in r for r in manifest["forecasts"])
    missing=next(r for r in manifest["forecasts"] if r["product"]==product and r["horizon"]==5)
    assert all(missing[k] is None for k in ("candidate","ablation","direct","persistence"))
    pilot.validate_forecasts(manifest)
    missing["actual"]=1
    with pytest.raises(pilot.PilotError,match="target_value"):
        pilot.validate_forecasts(manifest)


def test_ineligible_trend_falls_back_to_direct_with_reason():
    report,prices,origin,product=_fixture()
    prices["products"][0]["observations"]=[{"date":(origin-dt.timedelta(days=1)).isoformat(),"actual":99.0},
                                             {"date":(origin+dt.timedelta(days=1)).isoformat(),"actual":105.0}]
    manifest=pilot.build_forecast(report,prices,{"prices":"p","directReport":"r"})
    row=next(r for r in manifest["forecasts"] if r["product"]==product and r["origin"]==origin.isoformat() and r["horizon"]==1)
    assert row["candidate"]==row["ablation"]==row["direct"]==100.0
    assert row["candidateFallbackReason"]==row["ablationFallbackReason"]=="fewer_than_10_dates"


def test_score_handles_empty_cells_and_paired_target_reconciliation():
    report,prices,origin,product=_fixture()
    manifest=pilot.build_forecast(report,prices,{"prices":"p","directReport":"r"})
    result=pilot.score(manifest,prices,report=report)
    assert result["coverage"]["rowsWithActual"]==1
    assert result["macroMetrics"]["candidate"]["mae"] is None
    assert result["productGates"][product]["equalHorizonMaeWorseningPct"] is None
    report["pairedRows"][0]["actual"]=999
    with pytest.raises(pilot.PilotError,match="price_report_actual_mismatch"):
        pilot.score(manifest,prices,report=report)


def test_manifest_hash_tampering_rejected():
    manifest={"status":"answer_free_forecast_manifest","protocol":pilot.PROTOCOL,"manifestSha256":"0"*64}
    raw=(json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode()
    with pytest.raises(pilot.PilotError,match="manifest_hash"):
        pilot.score(manifest,{"schemaVersion":1,"products":[]},manifest_bytes=raw)


def test_persistence_source_must_match_prior_price_snapshot():
    report,prices,origin,product=_fixture()
    manifest=pilot.build_forecast(report,prices,{"prices":"p","directReport":"r"})
    row=next(r for r in manifest["forecasts"] if r["product"]==product and r["origin"]==origin.isoformat() and r["horizon"]==1)
    row["persistenceSourceDate"]=origin.isoformat()
    with pytest.raises(pilot.PilotError,match="persistence_source_mismatch"):
        pilot.score(manifest,prices,report=report)
