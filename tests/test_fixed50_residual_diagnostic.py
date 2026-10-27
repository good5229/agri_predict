import datetime as dt
import hashlib

from agri_data import fixed50_residual_diagnostic as diag


def _component_manifest(rows):
    raw={"phase":"historical_components_only","targetValuesAttached":False,
         "inputSha256":{"prices":diag.PRICES_SHA256},"forecasts":rows}
    raw["manifestSha256"]=hashlib.sha256(diag._canonical(raw).rstrip(b"\n")).hexdigest()
    return raw


def test_weighted_median_projects_one_onto_exact_optimal_interval():
    # Ratios .8 and 1.2 have weights 1.25 and .8333; duplicate the latter
    # five times to make the lower and upper sides each carry equal total weight.
    pairs=[(80.0,100.0)]*2+[(120.0,100.0)]*3
    assert diag._weighted_median_multiplier(pairs)==1.0
    # A singleton minimizer is the weighted median observation.
    assert diag._weighted_median_multiplier([(120.0,100.0)]*3)==1.2


def test_rolling_correction_obeys_window_maturity_and_minimum_support():
    origin=dt.date(2025,2,28)
    prior=[]
    for i in range(30):
        day=dt.date(2025,1,15)+dt.timedelta(days=i)
        prior.append({"product":"cabbage","horizon":1,"target":day,"actual":120.0,"fixed50":100.0})
    prior.extend([
        {"product":"cabbage","horizon":1,"target":origin-dt.timedelta(days=13),"actual":200.0,"fixed50":100.0},
        {"product":"cabbage","horizon":1,"target":origin-dt.timedelta(days=181),"actual":20.0,"fixed50":100.0},
    ])
    row={"product":"cabbage","horizon":1,"target":dt.date(2025,3,1),"origin":origin,"actual":110.0,"fixed50":100.0}
    history={(p,h):[] for p in diag.PRODUCTS for h in diag.HORIZONS}
    history[("cabbage",1)]=prior
    corrected=diag._correct(row,history)
    assert corrected["correction"]["n"]==30
    assert corrected["correction"]["multiplier"]==1.2
    assert corrected["correction"]["appliedFactor"]==1.05
    assert corrected["candidate"]==105.0
    history[("cabbage",1)]=prior[:29]
    fallback=diag._correct(row,history)
    assert fallback["correction"]["status"]=="fallback"
    assert fallback["candidate"]==row["fixed50"]


def test_components_reconstruct_only_fixed50_and_join_historical_actuals():
    target=dt.date(2025,3,1);h=2;origin=target-dt.timedelta(days=h)
    row={"product":"cabbage","target":target.isoformat(),"origin":origin.isoformat(),"horizon":h,
         "direct":100.0,"sarima":120.0}
    raw=_component_manifest([row])
    rows=diag._component_rows(raw,{"cabbage":{target:110.0}},expected_count=None)
    assert rows[0]["fixed50"]==110.0
    assert rows[0]["actual"]==110.0


def test_dm_uses_unique_calendar_dates_and_invalid_variance_is_not_significant():
    values={dt.date(2025,1,1)+dt.timedelta(days=i):1.0 for i in range(30)}
    result=diag._dm(values,9)
    assert result["status"]=="zero_or_invalid_hac_variance"
    assert result["pValue"]==1.0
    assert diag._dm(dict(list(values.items())[:29]),9)["status"]=="insufficient_pairs"


def test_prices_after_historical_cutoff_are_not_parsed_as_actuals():
    raw={"schemaVersion":1,"products":[{"id":p,"observations":[
        {"date":"2025-08-31","actual":"123"},{"date":"2025-09-01","actual":"not-an-answer"}]} for p in diag.PRODUCTS]}
    prices,audit=diag._price_map(raw)
    assert all(list(v)==[dt.date(2025,8,31)] for v in prices.values())
    assert all(x==1 for x in audit.values())


def test_holm_family_size_is_100_and_joint_gate_does_not_mix_cells():
    rows=[]
    for p in diag.PRODUCTS:
      for h in diag.HORIZONS:
       for i in range(30):
        day=dt.date(2025,3,1)+dt.timedelta(days=i)
        base=100.0;actual=110.0;candidate=base if (p,h)!=("cabbage",1) else 108.0
        rows.append({"product":p,"horizon":h,"target":day,"origin":day-dt.timedelta(days=h),"actual":actual,"fixed50":base,"candidate":candidate})
    score=diag._score(rows)
    assert len(score["dmByBandwidth"]["9"])==100
    assert len(score["dmByBandwidth"]["28"])==100
    assert all(score["supportGate"]["byCrop"][p]["all10HorizonsHaveAtLeast30Pairs"] for p in diag.PRODUCTS)
    assert score["supportGate"]["eligibleCrops"]==[]


def test_joint_gate_requires_same_cell_to_improve_both_losses_at_both_bandwidths():
    cells={f"{p}|{h}":{"n":30} for p in diag.PRODUCTS for h in diag.HORIZONS}
    metrics=lambda mape,mae:{"n":30,"mape":mape,"mae":mae}
    crop_metrics=lambda mape,mae:{"equal10HorizonMape":mape,"equal10HorizonMae":mae,"byHorizon":{str(h):metrics(10.0,10.0) for h in diag.HORIZONS}}
    fixed_macro={"equal50CellMape":10.0,"equal50CellMae":10.0}
    candidate_macro={"equal50CellMape":9.0,"equal50CellMae":9.0}
    crop={p:{"fixed50":crop_metrics(10.0,10.0),"candidate":crop_metrics(9.0,9.0),"fixed50Issued":300,"candidateIssued":300} for p in diag.PRODUCTS}
    tests={bw:[{"product":p,"horizon":h,"loss":loss,"status":"tested","meanLossDifference":1.0,"holmSignificant":False}
               for p in diag.PRODUCTS for h in diag.HORIZONS for loss in ("ae","ape")] for bw in diag.BANDWIDTHS}
    # Split positive evidence across different cells: this must not pass.
    for bw in diag.BANDWIDTHS:
        for t in tests[bw]:
            if t["product"]=="cabbage" and t["horizon"]==1 and t["loss"]=="ae":t.update(meanLossDifference=-1.0,holmSignificant=True)
            if t["product"]=="radish" and t["horizon"]==2 and t["loss"]=="ape":t.update(meanLossDifference=-1.0,holmSignificant=True)
    result=diag._support_gate(cells,tests,fixed_macro,candidate_macro,crop)
    assert result["byCrop"]["cabbage"]["jointImprovingHorizons"]==[]
    # Completing both losses for one cell across both families passes.
    for bw in diag.BANDWIDTHS:
        for t in tests[bw]:
            if t["product"]=="cabbage" and t["horizon"]==1 and t["loss"]=="ape":t.update(meanLossDifference=-1.0,holmSignificant=True)
    gate=diag._support_gate(cells,tests,fixed_macro,candidate_macro,crop)
    assert gate["byCrop"]["cabbage"]["jointImprovingHorizons"]==[1]
    assert gate["byCrop"]["cabbage"]["passed"] is True
    cells["pepper|1"]["n"]=29
    unsupported=diag._support_gate(cells,tests,fixed_macro,candidate_macro,crop)
    assert unsupported["byCrop"]["pepper"]["status"]=="inference_unavailable"
    assert "pepper" not in unsupported["eligibleCrops"]
