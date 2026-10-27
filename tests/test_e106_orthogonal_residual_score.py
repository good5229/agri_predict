import datetime as dt
import math

import pytest

from agri_data import e106_orthogonal_residual_score as scorer


def _fixture(phase="validation", *, unissued_garlic_h1=False, empty_actual_block=False,
             extra_fallback=False):
    start, end, origin_start = scorer.PHASES[phase]
    base, forecasts = {}, []
    actuals = {product: {} for product in scorer.PRODUCTS}
    for product in scorer.PRODUCTS:
        for i in range((end - start).days + 1):
            day = start + dt.timedelta(days=i)
            if not (empty_actual_block and product == "garlic"
                    and start + dt.timedelta(days=14) <= day <= start + dt.timedelta(days=27)):
                actuals[product][day] = (100.0, 25.0, 1, ["market-a"])
        for horizon in scorer.HORIZONS:
            for i in range(horizon, (end - start).days + 1):
                target = start + dt.timedelta(days=i)
                origin = target - dt.timedelta(days=horizon)
                key = (product, target, horizon)
                not_issued = unissued_garlic_h1 and product == "garlic" and horizon == 1
                candidate = {"status": "not_issued" if not_issued else "issued",
                             "prediction": None if not_issued else 105.0,
                             "reason": "synthetic_unissued" if not_issued else None}
                base[key] = {"product": product, "target": target.isoformat(), "origin": origin.isoformat(),
                             "horizon": horizon, "candidate": candidate}
                if product != "garlic":
                    row = {"usedS": False, "fallbackReason": "non_garlic_exact_E106",
                           "status": candidate["status"], "prediction": candidate["prediction"],
                           "reason": candidate["reason"]}
                elif not_issued:
                    row = {"usedS": False, "fallbackReason": "synthetic_unissued",
                           "status": candidate["status"], "prediction": candidate["prediction"],
                           "reason": candidate["reason"]}
                elif extra_fallback and i == horizon and horizon == 2:
                    row = {"usedS": False, "fallbackReason": "synthetic_S_fallback",
                           "status": candidate["status"], "prediction": candidate["prediction"],
                           "reason": candidate["reason"]}
                else:
                    row = {"usedS": True, "fallbackReason": None,
                           "status": "issued", "prediction": 100.0, "reason": None}
                row.update({"product": product, "target": target.isoformat(), "origin": origin.isoformat(),
                    "horizon": horizon, "eligible": origin >= origin_start,
                    "actualLogRatio": None, "p0": 90.0 if product == "garlic" else None,
                    "p0Date": "synthetic" if product == "garlic" else None,
                    "RLogPrediction": math.log(105 / 90) if product == "garlic" else None,
                    "SLogPrediction": math.log(100 / 90) if product == "garlic" else None,
                    "RDirectPrediction": 105.0 if product == "garlic" else None,
                    "SDirectPrediction": 100.0 if product == "garlic" else None,
                    "sarimaPrediction": 102.0 if product == "garlic" else None,
                    "RBlendPrediction": 103.5 if product == "garlic" else None,
                    "SBlendPrediction": 101.0 if product == "garlic" and row["usedS"] else None,
                    "r106ReplayAbsError": 0.0 if product == "garlic" else None,
                    "fitId": f"fit-h{horizon}" if product == "garlic" else None,
                    "featureVector": [0.0] * 106 if product == "garlic" else None})
                forecasts.append(row)
    doc = {"schema": "e106-orthogonal-residual-v1", "phase": phase,
        "targetValuesAttached": False, "manifestSha256": "manifest-hash",
        "protocol": {"protocolSha256": "protocol-hash"}, "forecasts": forecasts}
    return doc, base, actuals


def _score(doc, base, actuals, **kwargs):
    options = {"all_corrections_zero": False, "numerical_checks_passed": True}
    options.update(kwargs)
    return scorer.score_manifest(doc, base_forecasts=base, actuals=actuals,
        phase=doc["phase"], **options)


def test_fixed_20_dm_families_40_cell_macro_and_primary_decision(monkeypatch):
    calls = []
    def fake_bootstrap(rows, **kwargs):
        calls.append((len(rows), kwargs["metric"], kwargs["block"], kwargs["calendar_start"], kwargs["calendar_end"]))
        return {"blockDays": kwargs["block"], "calendarDays": 81, "requested": kwargs["replicates"],
                "valid": kwargs["replicates"], "invalid": 0, "seed": kwargs["seed"],
                "interval95": [0.0, 0.0]}
    monkeypatch.setattr(scorer, "circular_block_bootstrap", fake_bootstrap)
    doc, base, actuals = _fixture()
    report = _score(doc, base, actuals)
    assert len(report["dmTests"]) == 40
    assert all(sum(row["bandwidthDays"] == bw for row in report["dmTests"]) == 20
               for bw in scorer.BANDWIDTHS)
    assert all(row["units"] == "percentage_points" for row in report["dmTests"] if row["loss"] == "ape")
    assert report["wholePolicy"]["primaryGarlic10Horizon"]["status"] == "complete_fixed_cell_macro"
    assert report["wholePolicy"]["auxiliaryFourCrop40Cell"]["expectedCells"] == 40
    assert report["primaryDecision"]["passes"] is True
    assert len(calls) == 8
    assert all(call[3:] == (dt.date(2026, 4, 1), dt.date(2026, 6, 20)) for call in calls)
    assert report["manifestSha256"] == "manifest-hash"
    assert report["protocolSha256"] == "protocol-hash"
    assert doc["forecasts"][0]["actualLogRatio"] is None


def test_empty_fixed_horizon_cell_stays_null_and_dm_family_keeps_unsupported_p1(monkeypatch):
    monkeypatch.setattr(scorer, "circular_block_bootstrap", lambda *a, **k: pytest.fail("incomplete cells must not bootstrap"))
    doc, base, actuals = _fixture(unissued_garlic_h1=True)
    report = _score(doc, base, actuals)
    primary = report["wholePolicy"]["primaryGarlic10Horizon"]
    assert primary["status"] == "incomplete_fixed_cell_support"
    assert primary["mapeDifferencePercentagePoints"] is None
    assert "garlic/1" in primary["missingCells"]
    h1 = [r for r in report["dmTests"] if r["horizon"] == 1]
    assert all(r["n"] == 0 and r["pValue"] == 1 for r in h1)
    assert report["primaryDecision"]["passes"] is False
    assert report["blockBootstrap"]["garlic10HorizonPrimary"]["status"] == "unsupported_missing_fixed_cells"


def test_fallback_counts_only_garlic_and_blocks_gate(monkeypatch):
    monkeypatch.setattr(scorer, "circular_block_bootstrap", lambda *a, **k: {
        "interval95": [0.0, 0.0], "valid": 1, "invalid": 0})
    doc, base, actuals = _fixture(extra_fallback=True)
    report = _score(doc, base, actuals)
    coverage = report["coverage"]
    assert coverage["garlicFallbackReasonCounts"]["synthetic_S_fallback"] == 1
    assert len(coverage["garlicAdditionalFallbackKeys"]) == 1
    assert "non_garlic_exact_E106" not in coverage["garlicFallbackReasonCounts"]
    assert report["primaryDecision"]["noAdditionalGarlicFallback"] is False
    assert report["primaryDecision"]["passes"] is False


def test_raw_log_identity_months_and_calendar_blocks_are_anchored_with_empty_block(monkeypatch):
    monkeypatch.setattr(scorer, "circular_block_bootstrap", lambda *a, **k: {
        "interval95": [0.0, 0.0], "valid": 1, "invalid": 0})
    doc, base, actuals = _fixture(empty_actual_block=True)
    report = _score(doc, base, actuals)
    identity = report["rawLogRatioIdentity"]
    assert identity["overall"]["status"] == "complete_fixed_cell_macro"
    assert identity["overall"]["identityError"] < 1e-14
    assert "2026-04" in identity["byMonth"]
    blocks = identity["nonoverlapCalendarBlocks"]["14"]
    assert blocks[0]["start"] == "2026-04-01"
    assert blocks[1]["start"] == "2026-04-15"
    assert blocks[1]["status"] == "empty_block"
    assert blocks[-1]["end"] == "2026-06-20"
    assert identity["missingCalendarDatesWereNotCompressed"] is True


def test_unknown_actual_state_is_not_treated_as_nonpositive_and_zero_correction_fails(monkeypatch):
    monkeypatch.setattr(scorer, "circular_block_bootstrap", lambda *a, **k: {
        "interval95": [0.0, 0.0], "valid": 1, "invalid": 0})
    doc, base, actuals = _fixture()
    del actuals["garlic"][dt.date(2026, 4, 5)]
    report = _score(doc, base, actuals, all_corrections_zero=True)
    coverage = report["coverage"]["eligibleByProduct"]["garlic"]
    assert coverage["noSourceObservation"] > 0
    assert report["primaryDecision"]["notAllCorrectionsZero"] is False
    assert report["primaryDecision"]["passes"] is False


def test_empty_phase_month_is_retained_with_null_metrics_and_distinct_date_coverage(monkeypatch):
    monkeypatch.setattr(scorer, "circular_block_bootstrap", lambda *a, **k: {
        "interval95": [0.0, 0.0], "valid": 1, "invalid": 0})
    doc, base, actuals = _fixture()
    for day in list(actuals["garlic"]):
        if day.month == 5:
            del actuals["garlic"][day]
    for product in ("cabbage", "radish", "onion"):
        for day in list(actuals[product]):
            if day.month == 5:
                del actuals[product][day]
    report = _score(doc, base, actuals)
    policy_month = report["wholePolicy"]["byMonth"]["2026-05"]
    assert policy_month["garlic10H"]["status"] == "incomplete_fixed_cell_support"
    assert policy_month["garlic10H"]["mapeDifferencePercentagePoints"] is None
    assert policy_month["fourCrop40"]["status"] == "incomplete_fixed_cell_support"
    component_month = report["rawGarlicComponents"]["garlic"]["byMonth"]["2026-05"]
    assert component_month["byHorizon"]["1"]["direct"]["n"] == 0
    assert component_month["directFixed10Horizon"]["aMae"] is None
    identity_month = report["rawLogRatioIdentity"]["byMonth"]["2026-05"]
    assert identity_month["status"] == "incomplete_fixed_cell_support"
    coverage_month = report["coverage"]["eligibleByMonth"]["2026-05"]["eligibleByProductAndHorizon"]
    assert coverage_month["garlic/1"]["eligibleKeys"] > 0
    assert coverage_month["garlic/1"]["positiveActual"] == 0
    assert coverage_month["garlic/1"]["noSourceObservation"] > 0
    h1 = report["coverage"]["eligibleByProductHorizon"]["garlic/1"]
    garlic_all_h = report["coverage"]["eligibleByProduct"]["garlic"]
    assert h1["scheduledUniqueOriginDates"] == h1["eligibleKeys"]
    assert garlic_all_h["scheduledUniqueTargetDates"] < garlic_all_h["eligibleKeys"]
    assert garlic_all_h["scheduledUniqueOriginDates"] < garlic_all_h["eligibleKeys"]


def test_exact_manifest_key_binding_and_phase_window(monkeypatch):
    monkeypatch.setattr(scorer, "circular_block_bootstrap", lambda *a, **k: {
        "interval95": [0.0, 0.0], "valid": 1, "invalid": 0})
    doc, base, actuals = _fixture("final")
    report = _score(doc, base, actuals)
    assert report["phaseTargetWindow"] == {"start": "2026-07-11", "end": "2026-09-27"}
    assert report["primaryDecision"]["passes"] is True
    doc["forecasts"].pop()
    with pytest.raises(ValueError, match="complete forecast grid"):
        _score(doc, base, actuals)
