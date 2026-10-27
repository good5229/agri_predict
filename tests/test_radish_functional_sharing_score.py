import datetime as dt
import copy
import math

import pytest

from agri_data import radish_functional_sharing_score as score


def fixture(monkeypatch, a_status="not_issued"):
    start = dt.date(2026, 5, 20)
    monkeypatch.setattr(score, "WINDOWS", {"V": (start, start)})
    monkeypatch.setattr(score, "ORIGIN_CUTOFF", {"V": start-dt.timedelta(days=20)})
    monkeypatch.setattr(score, "REPS", 8)
    rows, baseline, actuals = [], {}, {}
    for product in score.PRODUCTS:
        actuals[(product, start)] = 11.0
        for h in score.HORIZONS:
            origin = start-dt.timedelta(days=h)
            key = (product, origin, start, h)
            conditions = {}
            for c, prediction in (("B", 10.0), ("C", 10.5), ("F", 10.25), ("A", 10.3)):
                if product != "radish": prediction = 10.0
                status = a_status if c == "A" else "issued"
                conditions[c] = {"status": status, "directPrediction": prediction,
                                 "directLogPrediction": math.log(prediction/10),
                                 "blendPrediction": (prediction+10)/2 if product=="radish" else 17.0,
                                 "sarimaPrediction": 10.0}
            row = {"product": product, "originDate": origin.isoformat(), "targetDate": start.isoformat(),
                   "horizon": h, "eligible": True, "p0": 10.0, "sarimaPrediction": 10.0,
                   "conditions": conditions}
            rows.append(row)
            if product == "radish":
                baseline[key] = {"status":"issued", "prediction":10.0, "directPrediction":10.0,
                                 "sarimaPrediction":10.0}
    return {"forecasts": rows}, baseline, actuals


def test_a_unavailable_does_not_shrink_primary_fc_support_or_fixed_family(monkeypatch):
    doc, baseline, actuals = fixture(monkeypatch)
    original=copy.deepcopy(doc)
    result = score.score_manifest(doc, baseline_rows=baseline, actuals=actuals,
                                  fit_diagnostics=[], phase="V")
    assert doc==original
    assert result["comparisons"]["F_vs_C"]["n"] == 10
    assert result["comparisons"]["F_vs_C"]["complete10HorizonSupport"] is True
    assert result["comparisons"]["F_vs_A"]["n"] == 0
    assert len(result["dmTests"]) == 120
    for bw in (9, 28):
        cells = [x for x in result["dmTests"] if x["bandwidth"] == bw]
        assert len(cells) == 60
        assert sum(x["comparison"] == "F_vs_A" for x in cells) == 20
        assert all(x["status"] == "insufficient_pairs" for x in cells)
    assert result["coverage"]["F_missing_when_C_available"] == []
    assert len(result["allFourFCCommonMetrics"]) == 40
    radish_cell = next(x for x in result["allFourFCCommonMetrics"] if x["product"]=="radish" and x["horizon"]==1)
    assert radish_cell["n"]==1
    assert radish_cell["F_minus_C"]["mae"] is not None
    assert result["fourCrop40CellMacro"]["complete40Cells"] is True
    assert result["allFourConditionCommonMetrics"][0]["n"] == 0
    assert result["vGateMetrics"]["passed"] is False


def test_saved_baseline_replay_is_enforced(monkeypatch):
    doc, baseline, actuals = fixture(monkeypatch)
    baseline[next(iter(baseline))]["prediction"] = 100.0
    with pytest.raises(ValueError, match="replay"):
        score.score_manifest(doc, baseline_rows=baseline, actuals=actuals,
                             fit_diagnostics=[], phase="V")


def test_only_radish_candidate_is_required_to_use_50_50_blend(monkeypatch):
    doc, baseline, actuals = fixture(monkeypatch,a_status="issued")
    # The unchanged E106 cabbage policy keeps its pinned blend composition.
    result=score.score_manifest(doc,baseline_rows=baseline,actuals=actuals,fit_diagnostics=[],phase="V")
    assert result["allFourConditionCommonMetrics"][0]["n"]==1
    # The actual FHS radish candidate still must use the frozen direct/SARIMA blend.
    row=next(r for r in doc["forecasts"] if r["product"]=="radish")
    row["conditions"]["F"]["blendPrediction"]+=1.0
    with pytest.raises(ValueError,match="radish candidate blend"):
        score.score_manifest(doc,baseline_rows=baseline,actuals=actuals,fit_diagnostics=[],phase="V")


def test_forecast_grid_and_cutoff_are_fixed(monkeypatch):
    doc, baseline, actuals = fixture(monkeypatch)
    doc["forecasts"].pop()
    with pytest.raises(ValueError, match="full four-crop grid"):
        score.score_manifest(doc, baseline_rows=baseline, actuals=actuals,
                             fit_diagnostics=[], phase="V")
    doc, baseline, actuals = fixture(monkeypatch)
    doc["forecasts"][0]["eligible"] = False
    with pytest.raises(ValueError, match="eligibility"):
        score.score_manifest(doc, baseline_rows=baseline, actuals=actuals,
                             fit_diagnostics=[], phase="V")


def test_ineligible_source_issued_rows_are_preserved_but_not_scored(monkeypatch):
    doc, baseline, actuals = fixture(monkeypatch)
    cutoff=dt.date(2026,5,20)
    monkeypatch.setattr(score,"ORIGIN_CUTOFF",{"V":cutoff})
    for row in doc["forecasts"]:
        row["eligible"]=False
    result=score.score_manifest(doc,baseline_rows=baseline,actuals=actuals,fit_diagnostics=[],phase="V")
    assert result["comparisons"]["F_vs_C"]["n"]==0
    assert result["coverage"]["F"]["eligibleKeys"]==0
    assert all(row["conditions"]["B"]["status"]=="issued" for row in doc["forecasts"])


def test_training_diagnostics_preserve_unavailable_a():
    fits = [{"fitId":"may-h1","fitOrigin":"2026-05-01","horizon":1,
             "trainingCount":1,"rho":1.2,"E_C":0.25,"E_F":0.36,
             "aStatus":"unavailable","aReason":"zero_E_C",
             "trainingRms":{"B":0.1,"C":0.2,"F":0.3,"A":None},
             "movementCosine":{"F_B":0.9,"C_B":0.8,"A_B":None},
             "trainingRows":[{"targetDate":"2026-04-01","actualLogReturn":0.1,
                              "B":0.0,"C":0.1,"F":0.15,"A":None}]}]
    result = score._training(fits)
    row = result["fitDiagnostics"][0]
    assert row["aStatus"] == "unavailable"
    assert row["trainingRms"]["A"] is None
    assert "A" not in row["recomputedTrainingRms"]


def test_movement_uses_actual_denominator_overshoot_gain_harm_and_month_support():
    rows=[]; by={}
    # Unequal horizon counts: h1 has three keys, all other horizons two.
    for h in score.HORIZONS:
        for month,day in ((4,10+h),(5,10+h)):
            target=dt.date(2026,month,day); actual=100.0
            cond={}
            for c,pred in (("B",98.0),("C",98.0),("F",100.0+0.5*h),("A",98.5)):
                cond[c]={"status":"issued","directPrediction":pred,"blendPrediction":pred,
                         "directLogPrediction":math.log(pred/100)}
            key=("radish",target-dt.timedelta(days=h),target,h)
            row={"product":"radish","origin":key[1],"target":target,"horizon":h,
                 "eligible":True,"issued":set(score.CONDITIONS),"conditions":cond,"actual":actual,"actualLogReturn":0.0}
            rows.append(row);by[key]=row
        if h==1:
            target=dt.date(2026,4,25);actual=100.0;cond={}
            for c,pred in (("B",98.0),("C",98.0),("F",100.0+0.5*h),("A",98.5)):
                cond[c]={"status":"issued","directPrediction":pred,"blendPrediction":pred,"directLogPrediction":math.log(pred/100)}
            key=("radish",target-dt.timedelta(days=h),target,h)
            row={"product":"radish","origin":key[1],"target":target,"horizon":h,"eligible":True,
                 "issued":set(score.CONDITIONS),"conditions":cond,"actual":actual,"actualLogReturn":0.0}
            rows.append(row);by[key]=row
    result=score._movement_diagnostics(rows,by,"V")
    fvc=next(x for x in result["interval"] if x["comparison"]=="F_vs_C" and x["layer"]=="direct")
    assert fvc["n"]==21
    assert fvc["movementMae"]==pytest.approx(4.75)
    assert fvc["movementMapePct"]==pytest.approx(4.75)  # denominator is actual = 100
    assert fvc["improvementMae"]==pytest.approx(-0.75)
    assert fvc["lossReducingMae"]==pytest.approx(2.0)
    assert fvc["lossIncreasingMae"]==pytest.approx(2.75)
    assert fvc["efficiencyMae"]==pytest.approx(-0.75/4.75)
    apr=next(x for x in result["targetMonth"] if x["comparison"]=="F_vs_C" and x["layer"]=="direct" and x["targetMonth"]=="2026-04")
    assert apr["hasAllTenHorizons"] is True
    assert apr["horizonCounts"]["1"]==2
    assert apr["recomputedEqualHorizonMonthMean"]["movementMae"]==pytest.approx(4.75)
    assert apr["fixedGlobalContribution"]["movementMae"] != pytest.approx(4.75)
    dates=result["targetDate"]
    assert math.fsum(x["fixedGlobalContribution"]["improvementMae"] for x in dates if x["comparison"]=="F_vs_C" and x["layer"]=="direct")==pytest.approx(fvc["improvementMae"])
    logs=score._log_error_report(rows,by,"V")["comparisons"]["F_vs_C"]
    assert logs["equalHorizonOverall"]["complete10HorizonSupport"] is True
    assert len(logs["byTargetMonth"])==3


def test_incomplete_horizon_interval_is_null_and_all_issued_is_radish_only():
    rows=[]; by={}
    for h in score.HORIZONS[:-1]:
        target=dt.date(2026,4,10+h); cond={}
        for c,pred in (("B",90.0),("C",91.0),("F",92.0),("A",90.5)):
            cond[c]={"status":"issued","directPrediction":pred,"blendPrediction":pred,"directLogPrediction":math.log(pred/100)}
        key=("radish",target-dt.timedelta(days=h),target,h)
        row={"product":"radish","origin":key[1],"target":target,"horizon":h,"eligible":True,"issued":set(score.CONDITIONS),"conditions":cond,"actual":100.0}
        rows.append(row);by[key]=row
    # A missing-actual radish issue belongs to all-issued RMS, not paired metrics.
    target=dt.date(2026,6,1); h=10; cond={c:{"status":"issued","directPrediction":200.0 if c=="F" else 100.0,"blendPrediction":200.0 if c=="F" else 100.0,"directLogPrediction":math.log(2 if c=="F" else 1)} for c in score.CONDITIONS}
    key=("radish",target-dt.timedelta(days=h),target,h); by[key]={"product":"radish","origin":key[1],"target":target,"horizon":h,"eligible":True,"issued":set(score.CONDITIONS),"conditions":cond}
    # Non-radish issues are present in the full grid and must not dilute radish RMS.
    for h0 in score.HORIZONS:
        target0=dt.date(2026,4,10+h0);cond0={c:{"status":"issued","directPrediction":10000.0 if c=="F" else 100.0,"blendPrediction":10000.0 if c=="F" else 100.0,"directLogPrediction":0.0} for c in score.CONDITIONS}
        key0=("cabbage",target0-dt.timedelta(days=h0),target0,h0);by[key0]={"product":"cabbage","origin":key0[1],"target":target0,"horizon":h0,"eligible":True,"issued":set(score.CONDITIONS),"conditions":cond0}
    result=score._movement_diagnostics(rows,by,"V")
    fvc=next(x for x in result["interval"] if x["comparison"]=="F_vs_C" and x["layer"]=="direct")
    assert fvc["complete10HorizonSupport"] is False
    assert fvc["movementMae"] is None
    assert fvc["availableFixedGlobalContribution"]["movementMae"] is not None
    assert fvc["allIssued"]["n"]==9+1


def test_missing_a_horizon_does_not_inherit_other_comparison_support():
    rows=[]; by={}
    for h in score.HORIZONS:
        target=dt.date(2026,4,10+h); cond={}
        for c,pred in (("B",100.0),("C",101.0),("F",102.0),("A",103.0)):
            status="not_issued" if c=="A" and h==10 else "issued"
            cond[c]={"status":status,"directPrediction":pred,"blendPrediction":pred,"directLogPrediction":math.log(pred/100)}
        key=("radish",target-dt.timedelta(days=h),target,h)
        row={"product":"radish","origin":key[1],"target":target,"horizon":h,"eligible":True,"issued":{c for c in score.CONDITIONS if cond[c]["status"]=="issued"},"conditions":cond,"actual":100.0}
        rows.append(row);by[key]=row
    result=score._movement_diagnostics(rows,by,"V")
    fc=next(x for x in result["interval"] if x["comparison"]=="F_vs_C" and x["layer"]=="direct")
    fa=next(x for x in result["interval"] if x["comparison"]=="F_vs_A" and x["layer"]=="direct")
    assert fc["complete10HorizonSupport"] is True and fc["movementMae"]==pytest.approx(1.0)
    assert fa["complete10HorizonSupport"] is False and fa["movementMae"] is None
    assert fa["availableFixedGlobalContribution"]["movementMae"] is not None
