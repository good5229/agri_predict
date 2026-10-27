import datetime as dt
import math

import pytest

from agri_data import final_ensemble_onion_recovery as recovery


def _row(value=100.0):
    return {
        "product": "onion", "origin": "2025-12-22", "target": "2025-12-23", "horizon": 1,
        "B": value, "sourceStatus": "issued", "BSourceStatus": "issued", "sourceReason": None,
        "fitId": "onion-2025-12-ridge-h01", "fitBoundary": "2025-12-01",
        "cutoff": "2025-12-21", "fitModel": "causalDirectRidge+SARIMA-causal50-v1",
        "sourcePhase": "generated", "direct": {"status": "issued", "prediction": 99.0, "reason": None},
        "sarima": {"status": "issued", "prediction": 101.0, "reason": None,
                    "fit": {"trainStart": "2023-12-02", "trainEnd": "2025-11-30",
                            "fitOrigin": "2025-12-01", "order": [1, 1, 1],
                            "seasonalOrder": [1, 0, 0, 7],
                            "trainingCalendarDays": 730, "selectedD": 1}},
        "provenance": {"sourceDate": "2025-12-20", "sourceAgeDays": 2,
                       "stateThrough": "2025-12-21"},
    }


def test_overlap_audit_accepts_bounded_binary64_drift_and_reports_exact_contract():
    source = _row()
    replay = _row(math.nextafter(math.nextafter(100.0, math.inf), math.inf))
    result = recovery.audit_overlap({("onion", "2025-12-22", "2025-12-23", 1): source}, [replay])
    assert result["overlapRows"] == 1
    assert result["numericToleranceUlps"] == 8
    assert result["allStatusesAndTimeContractsExact"] is True
    assert result["sourceRowsUnchanged"] is True
    assert source["B"] == 100.0


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(cutoff="2025-12-20"),
    lambda r: r["sarima"].update(status="not_issued"),
    lambda r: r["provenance"].update(stateThrough="2025-12-20"),
])
def test_overlap_audit_rejects_contract_or_numeric_mismatch(mutation):
    source, replay = _row(), _row()
    mutation(replay)
    with pytest.raises(ValueError, match="overlap mismatch"):
        recovery.audit_overlap({("onion", "2025-12-22", "2025-12-23", 1): source}, [replay])


def test_overlap_audit_rejects_numeric_difference_over_eight_ulps():
    source, replay = _row(), _row()
    for _ in range(9):
        replay["B"] = math.nextafter(replay["B"], math.inf)
    with pytest.raises(ValueError, match="overlap mismatch"):
        recovery.audit_overlap({("onion", "2025-12-22", "2025-12-23", 1): source}, [replay])


def test_overlap_audit_rejects_duplicate_replay_keys():
    row = _row()
    key = ("onion", "2025-12-22", "2025-12-23", 1)
    with pytest.raises(ValueError, match="duplicate replay overlap"):
        recovery.audit_overlap({key: row}, [row, row])


def test_sarima_component_prediction_is_required_to_match_exactly():
    source, replay = _row(), _row()
    replay["sarima"]["prediction"] = math.nextafter(101.0, math.inf)
    with pytest.raises(ValueError, match="sarima.prediction"):
        recovery.audit_overlap({("onion", "2025-12-22", "2025-12-23", 1): source}, [replay])


def test_equal_info_source_component_fields_normalize_without_fit_identity_aliasing():
    source = {
        "B": 100.0, "sourceStatus": "issued", "BSourceStatus": "fallback",
        "sourceReason": None, "fitBoundary": "2025-12-01", "cutoff": "2025-12-21",
        "sourceDirectStatus": "issued", "sourceDirectPrediction": 99.0,
        "sourceDirectReason": None,
        "sourceDirectProvenance": {"sourceAgeDays": 2},
        "sourceSarima": {"status": "not_issued", "prediction": None,
                          "reason": "state-error",
                          "training": {"trainStart": "2023-12-02", "trainEnd": "2025-11-30",
                                       "fitOrigin": "2025-12-01", "order": [1, 1, 1],
                                       "seasonalOrder": [1, 0, 0, 7],
                                       "trainingCalendarDays": 730, "selectedD": 1},
                          "provenance": {"lastActualDate": "2025-12-20",
                                         "lastUpdatedDate": "2025-12-21"}},
    }
    normalized = recovery._normalize_source_row(source)
    assert normalized["direct"] == {"status": "issued", "prediction": 99.0, "reason": None}
    assert normalized["sarima"]["fit"]["trainStart"] == "2023-12-02"
    assert normalized["sarima"]["fit"]["fitOrigin"] == "2025-12-01"
    assert normalized["provenance"] == {"sourceDate": "2025-12-20", "sourceAgeDays": 2,
                                        "stateThrough": "2025-12-21"}


def test_equal_info_source_schema_passes_component_comparison_after_normalization():
    source = {
        "B": 100.0, "sourceStatus": "issued", "BSourceStatus": "issued",
        "sourceReason": None, "fitBoundary": "2025-12-01", "cutoff": "2025-12-21",
        "sourceDirectStatus": "issued", "sourceDirectPrediction": 99.0,
        "sourceDirectReason": None, "sourceDirectProvenance": {"sourceAgeDays": 2},
        "sourceSarima": {"status": "issued", "prediction": 101.0, "reason": None,
                          "training": {"trainStart": "2023-12-02", "trainEnd": "2025-11-30",
                                       "fitOrigin": "2025-12-01", "order": [1, 1, 1],
                                       "seasonalOrder": [1, 0, 0, 7],
                                       "trainingCalendarDays": 730, "selectedD": 1},
                          "provenance": {"lastActualDate": "2025-12-20",
                                         "lastUpdatedDate": "2025-12-21"}},
    }
    replay = _row()
    replay["B"] = math.nextafter(100.0, math.inf)
    key = ("onion", "2025-12-22", "2025-12-23", 1)
    result = recovery.audit_overlap({key: source}, [replay], exact_expected_keys={key})
    assert result["overlapRows"] == 1
    assert result["changedValueCounts"]["B"] == 1
    assert result["sarimaPredictionExact"] is True


def test_expected_missing_tail_is_exactly_55_triangular_keys():
    rows = []
    start = dt.date(2021, 1, 1)
    end = dt.date(2026, 3, 31)
    for offset in range((end - start).days + 1):
        origin = start + dt.timedelta(days=offset)
        for h in range(1, 11):
            target = origin + dt.timedelta(days=h)
            if (dt.date(2025, 12, 22) <= origin <= dt.date(2025, 12, 31)
                    and target >= dt.date(2026, 1, 1)):
                continue
            rows.append({"product": "onion", "origin": origin.isoformat(),
                         "target": target.isoformat(), "horizon": h})
    missing = recovery._expected_missing(rows)
    assert len(missing) == 55
    assert {origin for _, origin, _, _ in missing} == {
        (dt.date(2025, 12, 22) + dt.timedelta(days=i)).isoformat() for i in range(10)
    }


def test_reconstituted_december_fit_inventory_records_unknown_elapsed_and_no_new_fits():
    evidence = []
    for horizon in range(1, 11):
        fit_id = f"onion-2025-12-ridge-h{horizon:02d}"
        evidence.append({"fitId": fit_id, "artifacts": [{"path": f"models/{fit_id}.joblib",
                                                              "sha256": "a" * 64}]})
    evidence.append({"fitId": "onion-2025-12-sarima", "artifacts": [
        {"path": "models/onion-2025-12-sarima-initial.joblib", "sha256": "b" * 64},
        {"path": "checkpoints/onion-2025-12-sarima-through-2025-12-31.joblib", "sha256": "c" * 64},
    ]})
    counts = (3617, 3617, 3616, 3615, 3614, 3613, 3612, 3611, 3611, 3610)
    contracts = {"ridge": [{"fitId": f"onion-2025-12-ridge-h{h:02d}",
                             "trainingCount": counts[h - 1],
                             "firstTrainingTarget": "2014-01-01",
                             "lastTrainingTarget": "2025-11-29"}
                            for h in range(1, 11)]}
    rows = recovery._december_fit_records(evidence, contracts)
    assert len(rows) == 11
    assert sum(row["model"] == "causalDirectRidge" for row in rows) == 10
    assert sum(row["model"] == "SARIMA" for row in rows) == 1
    assert all(row["fitElapsedMs"] is None for row in rows)
    assert rows[-1]["stateCheckpoints"][0]["stateThrough"] == "2025-12-30"
    assert [row["trainingCount"] for row in rows[:10]] == list(counts)


def test_saved_fit_audits_are_read_from_confirmation_phase():
    doc = {"confirmationForecasts": [], "evaluationForecasts": []}
    counts = (3617, 3617, 3616, 3615, 3614, 3613, 3612, 3611, 3611, 3610)
    for h, count in enumerate(counts, 1):
        doc["confirmationForecasts"].append({
            "product": "onion", "origin": "2025-12-21", "horizon": h, "eligible": True,
            "models": {"causalDirect": {"training": {
                "count": count, "firstTarget": "2014-01-01", "lastTarget": "2025-11-29",
                "fitOrigin": "2025-12-01", "alpha": 10.0, "featureCount": 40,
            }}},
        })
    saved = recovery._saved_training_audits(doc)
    assert len(saved) == 10
    assert [saved[h]["count"] for h in range(1, 11)] == list(counts)
