import datetime as dt
import math

import pytest

from agri_data import period_proxy_experiment as ex


def test_dekad_bounds_and_target_requires_three_observed_days():
    assert ex.dekad_bounds(dt.date(2024, 2, 29)) == (dt.date(2024, 2, 21), dt.date(2024, 2, 29))
    series = {
        dt.date(2024, 1, 1): (10., 1.), dt.date(2024, 1, 4): (20., 2.),
        dt.date(2024, 1, 10): (30., 3.), dt.date(2024, 1, 11): (999., 1.),
        dt.date(2024, 1, 12): (999., 1.), dt.date(2024, 2, 1): (800., 1.),
    }
    targets = ex.aggregate_targets(series)
    assert targets[(dt.date(2024, 1, 1), dt.date(2024, 1, 10))]["actual"] == 20.
    assert targets[(dt.date(2024, 1, 1), dt.date(2024, 1, 10))]["observedDays"] == 3
    short=(dt.date(2024, 1, 11), dt.date(2024, 1, 20))
    assert targets[short]["actual"] is None
    assert targets[short]["observedDays"]==2
    assert targets[short]["targetStatus"]=="insufficient_target_observations"
    empty=(dt.date(2024, 1, 21), dt.date(2024, 1, 31))
    assert targets[empty]["actual"] is None
    assert targets[empty]["observedDays"]==0
    p = (dt.date(2024, 1, 11), dt.date(2024, 1, 20))
    assert ex._next_period(p, 2) == (dt.date(2024, 2, 1), dt.date(2024, 2, 10))
    assert ex._next_period(p, -2) == (dt.date(2023, 12, 21), dt.date(2023, 12, 31))


def test_period_step_is_constant_time_calendar_arithmetic(monkeypatch):
    periods=ex.dekads(dt.date(2019,1,1),dt.date(2029,12,31))
    for i in range(40,len(periods)-40,37):
        for offset in (-36,-3,-1,0,1,3,36):
            assert ex._next_period(periods[i],offset)==periods[i+offset]
    monkeypatch.setattr(ex,"dekads",lambda *_: (_ for _ in ()).throw(AssertionError("period stepping rebuilt a grid")))
    assert ex._next_period((dt.date(2024,2,21),dt.date(2024,2,29)),36)==(dt.date(2025,2,21),dt.date(2025,2,28))


def test_features_use_only_prices_available_at_origin_and_matured_past_dekad():
    origin = dt.date(2024, 1, 13)
    series = {dt.date(2024, 1, 5): (100., 5.), dt.date(2024, 1, 11): (101., 6.),
              dt.date(2024, 1, 12): (10000., 5.), dt.date(2024, 1, 14): (20000., 5.)}
    targets = {(dt.date(2024, 1, 1), dt.date(2024, 1, 10)):
               {"actual": 90., "observedDays": 3}}
    got = ex.feature_row(series, targets, origin,
                         (dt.date(2024, 1, 21), dt.date(2024, 1, 31)))
    assert got is not None
    assert math.isclose(got[0][0], math.log(101.))
    assert got[0][1] == 2
    assert got[0][2] == pytest.approx(math.log(100.5))
    assert got[0][3] == pytest.approx(math.log1p(11.))
    assert got[0][4] == 2.
    assert got[0][11] == math.log(90.)
    assert got[2] == 101.
    assert ex.input_cutoff(origin) == dt.date(2024, 1, 11)


def test_b0_and_market_features_respect_origin_minus_two_cutoff():
    origin_period=(dt.date(2024,1,11),dt.date(2024,1,20))
    origin=ex.origin_for(origin_period)
    target=ex._next_period(origin_period,2)
    series={dt.date(2024,1,20):(50.,1.),dt.date(2024,1,21):(9999.,1.)}
    baseline=ex.baseline_predictions(series,{},2,target[0],target[1])
    assert baseline["B0a"][target]["predicted"]==50.
    assert baseline["B0a"][target]["sourceDate"]=="2024-01-20"
    market={"g":{dt.date(2024,1,6):{"price":100.,"weight":1.},dt.date(2024,1,20):{"price":100.,"weight":1.},dt.date(2024,1,21):{"price":10000.,"weight":1.}},
            "j":{dt.date(2024,1,6):{"price":80.,"weight":1.},dt.date(2024,1,20):{"price":80.,"weight":1.},dt.date(2024,1,21):{"price":1.,"weight":1.}}}
    got=ex.market_features(market,ex.input_cutoff(origin))
    assert got[1]==2


def test_label_maturity_and_prediction_origin_are_enforced():
    series = {dt.date(2024, 1, 1): (10., 1.), dt.date(2024, 1, 20): (11., 1.)}
    targets = {
        (dt.date(2024, 1, 1), dt.date(2024, 1, 10)): {"actual": 10., "observedDays": 3},
        (dt.date(2024, 1, 11), dt.date(2024, 1, 20)): {"actual": 11., "observedDays": 3},
    }
    early = dt.date(2024, 1, 21)
    late = dt.date(2024, 1, 22)
    first = (dt.date(2024, 1, 1), dt.date(2024, 1, 10))
    second = (dt.date(2024, 1, 11), dt.date(2024, 1, 20))
    assert ex.label_matured(first, early)
    assert not ex.label_matured(second, early)
    assert ex.label_matured(second, late)


def test_leap_dekad_previous_year_and_january_origins_are_calendar_safe():
    leap=(dt.date(2024,2,21),dt.date(2024,2,29))
    assert ex._previous_year_period(leap)==(dt.date(2023,2,21),dt.date(2023,2,28))
    jan=(dt.date(2024,1,1),dt.date(2024,1,10))
    h2=ex.planned_origins(jan[0],jan[1],2)[0]
    h3=ex.planned_origins(jan[0],jan[1],3)[0]
    assert h2["originPeriod"]==(dt.date(2023,12,11),dt.date(2023,12,20))
    assert h2["origin"]==dt.date(2023,12,22)
    assert h3["originPeriod"]==(dt.date(2023,12,1),dt.date(2023,12,10))


def test_quarterly_rolling_fit_uses_only_matured_labels_and_valid_forecast_targets():
    series = {}
    day = dt.date(2020, 1, 1)
    end = dt.date(2024, 3, 31)
    while day <= end:
        series[day] = (100. + (day-day.replace(month=1, day=1)).days / 100., 10.)
        day += dt.timedelta(days=1)
    targets = ex.aggregate_targets(series)
    predictions, audits = ex.rolling_predictions(series, targets, 2, 10., start=dt.date(2024,1,1),
                                                end=dt.date(2024,3,31), cadence="quarter")
    assert predictions
    assert len(audits) == 2
    audit = audits[-1]
    assert audit["status"] == "fitted"
    assert dt.date.fromisoformat(audit["latestMaturedTargetEnd"]) + dt.timedelta(days=2) <= dt.date.fromisoformat(audit["fitOrigin"])
    for period, prediction in predictions.items():
        origin_period = ex._next_period(period, -2)
        assert dt.date.fromisoformat(prediction["origin"]) == ex.origin_for(origin_period)
        assert period in targets


def test_missing_future_label_does_not_change_planned_schedule_or_suppress_forecast():
    series={}
    day=dt.date(2020,1,1); end=dt.date(2024,3,31)
    while day<=end:
        series[day]=(100.+day.day,10.); day+=dt.timedelta(days=1)
    targets=ex.aggregate_targets(series)
    planned=ex.planned_origins(dt.date(2024,1,1),dt.date(2024,3,31),2)
    absent=planned[3]["target"]
    reduced={k:v for k,v in targets.items() if k!=absent}
    control,control_audit=ex.rolling_predictions(series,targets,2,10.,start=dt.date(2024,1,1),end=dt.date(2024,3,31),cadence="quarter")
    forecast,forecast_audit=ex.rolling_predictions(series,reduced,2,10.,start=dt.date(2024,1,1),end=dt.date(2024,3,31),cadence="quarter")
    assert [x["fitOrigin"] for x in forecast_audit]==[x["fitOrigin"] for x in control_audit]
    assert absent in forecast and absent in control
    assert ex.label_matured(absent,dt.date.fromisoformat(forecast_audit[0]["fitOrigin"])) is False


def test_january_policy_is_unavailable_until_development_labels_mature():
    series={}
    day=dt.date(2020,1,1); end=dt.date(2024,1,31)
    while day<=end:
        series[day]=(100.+day.day,10.); day+=dt.timedelta(days=1)
    targets=ex.aggregate_targets(series)
    early=(dt.date(2024,1,1),dt.date(2024,1,10))
    later=(dt.date(2024,1,11),dt.date(2024,1,20))
    b1,audit=ex.rolling_predictions(series,targets,2,10.,start=dt.date(2024,1,1),end=dt.date(2024,1,31),
                                    cadence="quarter",policy_available_from=ex.POLICY_AVAILABLE_FROM)
    assert early not in b1
    assert later in b1
    assert any(a["status"]=="policy_unavailable_before_development_labels_mature" and
               dt.date.fromisoformat(a["fitOrigin"])<ex.POLICY_AVAILABLE_FROM for a in audit)
    raw=ex.baseline_predictions(series,targets,2,dt.date(2024,1,1),dt.date(2024,1,31))["B0a"]
    selected=ex._policy_available_predictions(raw)
    assert early in raw and early not in selected
    assert selected[later]["origin"]>=ex.POLICY_AVAILABLE_FROM.isoformat()


def test_baseline_selection_enforces_coverage_and_common_key_rule():
    actuals = {i: float(i + 1) for i in range(100)}
    maps = {name: {i: {"predicted": actuals[i] * 1.2} for i in actuals} for name in ("B0a", "B0b", "B0c")}
    # Better standalone errors cannot qualify below 90% of B0a-valid development keys.
    maps["B0c"] = {i: {"predicted": actuals[i]} for i in range(89)}
    assert ex._choose_baseline(actuals, maps) == "B0a"
    maps["B0a"] = {i: {"predicted": actuals[i] * 1.2} for i in range(10, 100)}
    maps["B0b"] = {i: {"predicted": actuals[i] * 1.01} for i in range(100)}
    maps["B0c"] = {i: {"predicted": actuals[i]} for i in range(90)}
    # B0c has 90% total availability but only 80/90 keys shared with B0a.
    assert ex._choose_baseline(actuals, maps) == "B0b"
    maps["B0a"] = {i: {"predicted": actuals[i] * 1.2} for i in range(100)}
    maps["B0b"] = {i: {"predicted": actuals[i] * 1.01} for i in actuals}
    maps["B0c"] = {i: {"predicted": actuals[i] * 1.03} for i in range(90)}
    assert ex._choose_baseline(actuals, maps) == "B0b"


def test_hac_and_holm_keep_inferentially_unavailable_items():
    pairs = [((dt.date(2024, 1, 1) + dt.timedelta(days=10*i),
               dt.date(2024, 1, 10) + dt.timedelta(days=10*i)), 100., 100. + (i % 7 + 1), 100.) for i in range(35)]
    good = ex._hac_dm(pairs, "ae", 3)
    bad = ex._hac_dm([], "ae", 3)
    adjusted = ex.holm([good, bad])
    assert adjusted[0]["status"] == "tested"
    assert adjusted[1]["holmP"] is None
    assert adjusted[1]["holmSignificant"] is False


def test_inverse_logratio_uses_latest_eligible_price_not_historical_maximum():
    assert ex._inverse_log_ratio(100.,math.log(1.1))==pytest.approx(110.)
    series={dt.date(2024,1,1):(900.,1.),dt.date(2024,1,10):(100.,1.),dt.date(2024,1,12):(5000.,1.)}
    row=ex.feature_row(series,{},dt.date(2024,1,12),(dt.date(2024,1,21),dt.date(2024,1,31)))
    assert row[2]==100.


def test_feature_builder_reuses_rows_and_market_windows_across_refits(monkeypatch):
    series={dt.date(2024,1,d):(float(100+d),float(d)) for d in range(1,29)}
    targets={(dt.date(2024,1,1),dt.date(2024,1,10)):{"actual":105.,"observedDays":3}}
    market={"m1":{},"m2":{}}
    calls=[]
    original=ex.market_features
    def counted(data,cutoff):
        calls.append(cutoff)
        return original(data,cutoff)
    monkeypatch.setattr(ex,"market_features",counted)
    builder=ex.FeatureBuilder(series,targets,market)
    origin=dt.date(2024,1,30); target=(dt.date(2024,2,1),dt.date(2024,2,10))
    first=builder.row(origin,target)
    cache_size=len(builder.cache)
    second=builder.row(origin,target)
    builder.row(origin,(dt.date(2024,2,11),dt.date(2024,2,20)))
    assert first is second
    assert len(builder.cache)==cache_size+1
    assert calls==[ex.input_cutoff(origin)]


def test_sparse_history_empty_lookbacks_preserve_zero_sentinel_and_count():
    origin=dt.date(2024,2,2)  # as-of cutoff is January 31
    target=(dt.date(2024,2,11),dt.date(2024,2,20))
    cases=((dt.date(2024,1,21), (2,3,4)),
           (dt.date(2024,1,11), (2,3,4,5,6,7)),
           (dt.date(2023,12,27), (2,3,4,5,6,7,8,9,10)))
    for last_day,empty_indices in cases:
        builder=ex.FeatureBuilder({last_day:(123.,4.)},{})
        row=builder.row(origin,target)
        assert row is not None
        for index in empty_indices:
            assert row[0][index]==0.
        assert row[1]==123.


def test_c1_missing_feature_uses_fixed_selected_b1_alpha_and_fallback_only_is_unavailable():
    targets=[("target-a",), ("target-b",)]
    candidate={targets[0]:{"predicted":130.,"alpha":100.}}
    fixed_selected_b1={targets[0]:{"predicted":120.,"alpha":10.},targets[1]:{"predicted":80.,"alpha":10.}}
    merged=ex._with_b1_fallback(candidate,fixed_selected_b1,targets)
    assert merged[targets[1]]["predicted"]==80.
    assert merged[targets[1]]["alpha"]==10.
    assert merged[targets[1]]["fallback"] is True
    paired=[1,2]
    assert ex._c1_inference_status({"status":"tested"},paired,0,True)["status"]=="fallback_only"
    assert ex._c1_inference_status({"status":"tested"},paired,0,False)["status"]=="gate_not_open"


def test_conclusions_separate_significant_worsening_mixed_and_inconclusive():
    def cell(left, right, differences, significant):
        return {"pairedMetrics":{"x":{"n":40,"left":left,"right":right}},"comparisons":[
            {"comparison":"x","lag":3,"status":"tested","meanLossDifference":differences[0],"holmSignificant":significant[0]},
            {"comparison":"x","lag":3,"status":"tested","meanLossDifference":differences[1],"holmSignificant":significant[1]}]}
    metric={"mae":12.,"mape":12.}
    base={"mae":10.,"mape":10.}
    assert ex._conclusion(cell(metric,base,[1.,2.],[True,True]),"x")["statistical"]=="significant_worsening"
    assert ex._conclusion(cell(metric,base,[1.,-1.],[True,False]),"x")["statistical"]=="mixed"
    assert ex._conclusion(cell(base,base,[.1,.2],[False,False]),"x")["statistical"]=="inconclusive"


def test_protocol_and_source_hash_fail_closed(tmp_path):
    wrong = tmp_path / "wrong.json"
    wrong.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="protocol hash"):
        ex.validate_inputs(wrong, ex.SOURCE)
    with pytest.raises(ValueError,match="context end"):
        ex.validate_snapshot_meta({"meta":{"contextEnd":"2026-10-10"}})
    ex.validate_snapshot_meta({"meta":{"contextEnd":"2026-10-09"}})


def test_run_end_to_end_on_synthetic_single_crop_both_horizons(monkeypatch):
    monkeypatch.setattr(ex,"CROPS",("cabbage",))
    monkeypatch.setattr(ex,"ALPHAS",(10.,))
    observations=[]
    day=dt.date(2020,1,1); end=dt.date(2026,9,30)
    while day<=end:
        price=100.+(day.timetuple().tm_yday%29)+((day.year-2020)*.2)
        observations.append({"date":day.isoformat(),"actual":price,"volumeKg":10.})
        day+=dt.timedelta(days=1)
    raw={"products":[{"id":"cabbage","observations":observations}]}
    report=ex.run(raw,{"id":"synthetic-test-protocol"})
    for horizon,unavailable_count in ((2,1),(3,2)):
        cell=report["crops"]["cabbage"][str(horizon)]
        first_row=next(r for r in cell["predictions"] if r["targetStart"]=="2024-01-01")
        assert len(cell["predictions"])==len(ex.dekads(dt.date(2024,1,1),dt.date(2026,9,30)))
        assert first_row["learnedPolicyAvailable"] is False
        assert "B0a" in first_row["forecasts"]
        assert "selectedB0" not in first_row["forecasts"]
        assert "B1" not in first_row["forecasts"]
        assert cell["policyUnavailableOrigins"]==unavailable_count
        assert cell["coverage"]["B1"]["scoredForecasts"] < cell["coverage"]["B1"]["plannedOrigins"]
        assert "conclusions" in cell
