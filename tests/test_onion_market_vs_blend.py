import datetime as dt

import pytest

from agri_data import onion_market_vs_blend as onion


def _synthetic_inputs():
    report_rows, a_rows, b_rows = [], [], []
    for name, (start, _) in onion.WINDOWS.items():
        count = onion.EXPECTED_COUNTS[name] // 2
        for i in range(count):
            target = start + dt.timedelta(days=i)
            for h in onion.HORIZONS:
                origin = target-dt.timedelta(days=h)
                key = {"product": onion.PRODUCT, "horizon": h, "origin": origin.isoformat(), "target": target.isoformat()}
                report_rows.append({**key, "fallback": (name == "A" and i == 0 and h == 1),
                                    "fallbackReasons": ["original_frozen_fallback"] if name == "A" and i == 0 and h == 1 else [],
                                    "candidatePrediction": 11., "baselinePrediction": 10., "actual": 999999.})
                if name == "A":
                    a_rows.append({**key, "status": "available", "direct": 10., "sarima": 14.})
                else:
                    b_rows.append({**key, "predictions": {"direct": 10., "sarima": 14., "fixed50": 12.}})
    return {
        str(onion.REPORT_PATH): {"originEvidence": report_rows},
        str(onion.HIST_PATH): {"forecasts": a_rows},
        str(onion.B_PATH): {"forecasts": b_rows},
    }


def test_compose_rows_uses_exact_window_sources_without_truth_or_new_fallback():
    docs = _synthetic_inputs()
    rows = onion._compose_rows(docs)
    assert len(rows) == onion.EXPECTED_COUNTS["A"]+onion.EXPECTED_COUNTS["B"]
    a = next(r for r in rows if onion._window_of(r) == "A")
    b = next(r for r in rows if onion._window_of(r) == "B")
    assert a["predictions"] == {onion.CANDIDATE: 11., onion.DIRECT: 10., onion.SARIMA: 14., onion.FIXED: 12.}
    assert b["predictions"][onion.FIXED] == 12.
    assert a["lineage"]["marketFallback"] is True
    assert a["lineage"]["marketFallbackReasons"] == ["original_frozen_fallback"]
    assert "actual" not in a and "actual" not in b


def test_composition_fails_on_key_or_direct_mismatch_and_nonpositive_inputs():
    docs = _synthetic_inputs()
    docs[str(onion.HIST_PATH)]["forecasts"][0]["direct"] = 9.
    with pytest.raises(ValueError, match="direct Ridge equality"):
        onion._compose_rows(docs)
    docs = _synthetic_inputs()
    docs[str(onion.B_PATH)]["forecasts"].pop()
    with pytest.raises(ValueError, match="missing exact market key|key-set mismatch"):
        onion._compose_rows(docs)
    docs = _synthetic_inputs()
    docs[str(onion.REPORT_PATH)]["originEvidence"][0]["candidatePrediction"] = 0.
    with pytest.raises(ValueError, match="unavailable/nonpositive"):
        onion._compose_rows(docs)


def test_manifest_verifier_rejects_answer_and_duplicate_key():
    row = {"product": onion.PRODUCT, "horizon": 1, "origin": "2024-09-24", "target": "2024-09-25",
           "predictions": {m: 10. for m in onion.METHODS}, "lineage": {}}
    doc = {"protocol": onion.PROTOCOL, "phase": "forecast_only", "targetValuesAttached": False,
           "inputSha256": onion.INPUT_HASHES, "plannedForecastCount": 1, "forecasts": [row]}
    doc["manifestSha256"] = onion._hash(doc)
    onion.verify_manifest(doc)
    doc["forecasts"] = [row, dict(row)]
    doc["plannedForecastCount"] = 2
    doc["manifestSha256"] = onion._hash({k:v for k,v in doc.items() if k != "manifestSha256"})
    with pytest.raises(ValueError, match="duplicate"):
        onion.verify_manifest(doc)
    doc["forecasts"][0]["actual"] = 10
    doc["manifestSha256"] = onion._hash({k:v for k,v in doc.items() if k != "manifestSha256"})
    with pytest.raises(ValueError, match="answer-bearing"):
        onion.verify_manifest(doc)


def test_period_score_reports_paired_metrics_calendar_dm_and_fallback_counts():
    planned, actuals = [], {}
    start = onion.WINDOWS["A"][0]
    for i in range(40):
        target = start+dt.timedelta(days=i)
        origin = target-dt.timedelta(days=1)
        row = {"product": onion.PRODUCT, "horizon": 1, "origin": origin.isoformat(), "target": target.isoformat(),
               "predictions": {onion.CANDIDATE: 10.+i*.01, onion.DIRECT: 10.1+i*.005,
                               onion.SARIMA: 10.2+i*.003, onion.FIXED: 10.15+i*.004},
               "lineage": {"marketFallback": i == 0, "marketFallbackReasons": ["old_fallback"] if i == 0 else []}}
        planned.append(row)
        actuals[onion._key(row)] = 10.+i*.02
    report = onion._score_window(planned, actuals, "A")
    assert report["dmFamilySizePerBandwidth"] == 12
    assert len(report["dmComparisons"]) == 12  # fixed family retains the empty horizon tests
    assert report["marketFallbackCount"] == 1
    assert report["marketFallbackReasonCounts"] == {"old_fallback": 1}
    assert report["byHorizon"][0]["metrics"][onion.CANDIDATE]["mae"] is not None
    assert report["dmComparisons"][0]["dm"]["9"]["n"] == 40


def test_holm_unavailable_and_joint_status_remain_explicit():
    records = [{"status": "inference_unavailable_insufficient_pairs", "pValue": 1.},
               {"status": "available", "pValue": .001, "meanLossDifference": -1.}]
    onion._holm(records)
    assert records[0]["conclusion"] == "inference_unavailable"
    assert onion._joint(["improvement"]*4) == "joint_improvement"
    assert onion._joint(["inference_unavailable", "inconclusive", "inconclusive", "inconclusive"]) == "inference_unavailable"
    assert onion._joint(["improvement", "worsening", "inconclusive", "inconclusive"]) == "mixed"


def test_gate_uses_frozen_tie_order_and_requires_both_periods():
    def period(gain):
        methods = {m: {"mae": 10., "mape": 5.} for m in onion.METHODS}
        methods[onion.CANDIDATE] = {"mae": 9., "mape": 5.-gain}
        by_h = [{"horizon": h, "metrics": methods} for h in onion.HORIZONS]
        joint = [{"horizon": 1, "comparator": onion.FIXED, "jointConclusion": "joint_improvement"}]
        return {"macroTwoHorizonEqualWeight": methods, "byHorizon": by_h,
                "coverage": {m: {"issued": 100} for m in onion.METHODS},
                "completeBothHorizonCellsAt30Pairs": True, "jointConclusions": joint}
    gate = onion._gate({"A": period(.3), "B": period(.3)})
    assert gate["criteriaByWindow"]["A"]["strongestBaseline"] == onion.FIXED
    assert gate["status"] == "pass"
    assert onion._gate({"A": period(.3)})["status"] == "unsupported"


def test_horizon_coverage_tracks_asymmetric_planned_valid_issued_and_common_counts():
    planned, actuals = [], {}
    start = onion.WINDOWS["A"][0]
    for horizon, offsets in ((1, (0, 1, 2)), (2, (0,))):
        for offset in offsets:
            target = start+dt.timedelta(days=offset)
            origin = target-dt.timedelta(days=horizon)
            predictions = {m: 10. for m in onion.METHODS}
            if horizon == 1 and offset == 2:
                predictions[onion.FIXED] = None
            row = {"product": onion.PRODUCT, "horizon": horizon, "origin": origin.isoformat(), "target": target.isoformat(),
                   "predictions": predictions, "lineage": {"marketFallback": False, "marketFallbackReasons": []}}
            planned.append(row)
            if not (horizon == 1 and offset == 1):
                actuals[onion._key(row)] = 12.
    report = onion._score_window(planned, actuals, "A")
    h1, h2 = report["coverageByHorizon"]["1"], report["coverageByHorizon"]["2"]
    assert (h1["planned"], h1["validPositiveTargets"], h1["issued"][onion.CANDIDATE],
            h1["issued"][onion.FIXED], h1["commonPositiveTargets"]) == (3, 2, 3, 2, 1)
    assert (h2["planned"], h2["validPositiveTargets"], h2["commonPositiveTargets"]) == (1, 1, 1)
