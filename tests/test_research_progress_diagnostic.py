import datetime as dt

import pytest

from agri_data.research_progress_diagnostic import (
    _verify_saved_metrics,
    circular_block_bootstrap,
    classify_countertrend,
    decompose_losses,
    write_report,
)


def test_decomposition_respects_unequal_cell_sizes_and_reconstructs_macro():
    rows = [
        {"product": "a", "horizon": 1, "target": "2026-01-01", "actual": 10, "a": 12, "b": 10},
        {"product": "a", "horizon": 1, "target": "2026-01-03", "actual": 10, "a": 10, "b": 12},
        {"product": "b", "horizon": 1, "target": "2026-01-01", "actual": 10, "a": 14, "b": 10},
    ]
    result = decompose_losses(rows, metric="ae")
    # Cell A's two rows average to zero difference; cell B contributes +4,
    # and the original two-cell equal-weight macro is therefore +2.
    assert result["meanDifference"] == pytest.approx(2.0)
    assert sum(row["contribution"] for row in result["byTarget"]) == pytest.approx(2.0)
    assert result["accountingVerified"] is True


def test_calendar_bootstrap_keeps_dates_joint_and_does_not_impute_missing_days():
    rows = [
        {"product": p, "horizon": 1, "target": d, "actual": 10, "a": a, "b": 10}
        for p, d, a in (("a", "2026-01-01", 12), ("b", "2026-01-01", 11),
                        ("a", "2026-01-03", 9), ("b", "2026-01-03", 10))
    ]
    result = circular_block_bootstrap(rows, block=1, replicates=200, seed=8)
    assert result["valid"] + result["invalid"] == 200
    assert result["valid"] > 0
    # No target rows are synthesized for January 2; both cells share the same
    # sampled calendar block by construction.
    assert {r["target"] for r in rows} == {"2026-01-01", "2026-01-03"}


def test_full_calendar_block_has_known_interval_across_missing_day():
    rows = [
        {"product": "a", "horizon": 1, "target": "2026-01-01", "actual": 10, "a": 12, "b": 10},
        {"product": "a", "horizon": 1, "target": "2026-01-03", "actual": 10, "a": 10, "b": 12},
    ]
    result = circular_block_bootstrap(rows, block=3, replicates=1, seed=4,
                                      calendar_start=dt.date(2026, 1, 1),
                                      calendar_end=dt.date(2026, 1, 3))
    assert result["calendarDays"] == 3
    assert result["valid"] == 1 and result["invalid"] == 0
    assert result["interval95"] == [0.0, 0.0]


def test_empty_cell_in_bootstrap_replicate_is_invalid_not_filled():
    rows = [{"product": "a", "horizon": 1, "target": "2026-01-01", "actual": 10, "a": 11, "b": 10},
            {"product": "b", "horizon": 1, "target": "2026-01-02", "actual": 10, "a": 12, "b": 10}]
    result = circular_block_bootstrap(rows, block=1, replicates=100, seed=2)
    assert result["invalid"] > 0
    assert result["valid"] + result["invalid"] == 100


def test_leave_out_keeps_original_cell_family_and_price_subgroups_reconcile():
    rows = [
        {"product": "a", "horizon": 1, "target": "2026-01-01", "actual": 1, "a": 2, "b": 1},
        {"product": "a", "horizon": 1, "target": "2026-02-01", "actual": 10, "a": 10, "b": 11},
        {"product": "b", "horizon": 1, "target": "2026-01-01", "actual": 2, "a": 4, "b": 2},
    ]
    result = decompose_losses(rows, metric="ae")
    assert result["bottomPriceDecile"]["accountingVerified"] is True
    assert result["leaveOut"]["month"]["2026-01"] == {
        "difference": None, "reason": "at_least_one_original_cell_empty"}


def test_saved_metric_reproduction_checks_counts_and_values():
    pairs = [{"product": "a", "horizon": 1, "target": "2026-01-01", "actual": 10, "a": 12, "b": 10}]
    saved = [{"product": "a", "horizon": 1, "n": 1, "aMae": 2, "aMape": 20,
              "bMae": 0, "bMape": 0}]
    assert _verify_saved_metrics(pairs, saved)["reproduced"] is True
    with pytest.raises(ValueError, match="paired count mismatch"):
        _verify_saved_metrics([], saved)


def test_countertrend_separates_direction_and_magnitude_and_sums_gain_harm():
    rows = [
        {"guardApplied": True, "p0": 100, "baselinePrediction": 80,
         "model": {"prediction": 110}, "actual": 120, "product": "a", "horizon": 1},
        {"guardApplied": True, "p0": 100, "baselinePrediction": 110,
         "model": {"prediction": 120}, "actual": 80, "product": "a", "horizon": 1},
        {"guardApplied": True, "p0": 100, "baselinePrediction": 80,
         "model": {"prediction": 110}, "actual": 120, "product": "b", "horizon": 1},
    ]
    result = classify_countertrend(rows, expected_cell_count=2,
                                   expected_cells={("a", 1), ("b", 1)})
    gain_label = "E:wrong/exact|C:correct/under"
    harm_label = "E:wrong/under|C:wrong/exact"
    grouped = result["weightedGainHarmByDirectionMagnitude"]
    assert result["counts"][gain_label] == 2
    assert grouped[gain_label]["aeGain"] == pytest.approx(22.5)
    assert grouped[gain_label]["aeHarm"] == pytest.approx(0)
    assert grouped[harm_label]["aeGain"] == pytest.approx(0)
    assert grouped[harm_label]["aeHarm"] == pytest.approx(2.5)
    assert grouped[gain_label]["aeNet"] == pytest.approx(22.5)


def test_output_write_is_create_only(tmp_path):
    report = {"reportSha256": "abc", "diagnostics": {
        "lossStructure": {}, "circularBlockBootstrap": {}, "countertrendMagnitude": {"counts": {}},
        "componentComplementarity": {"cells": []},
        "pathMlpTrainingDiagnostic": {"monthlyTrainingAudits": [], "storedLearningCurves": False,
                                       "storedValidationCurves": False},
        "priceAge": {"summary": {"cellCount": 0, "rowCount": 0,
                                   "equalCellMeanPositivePriceDaysInPrior28": 0,
                                   "equalCellMeanPositiveVolumeDaysInPrior28": 0, "byPhase": {}}},
    }}
    write_report(report, tmp_path)
    before = (tmp_path / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_report({"reportSha256": "changed"}, tmp_path)
    assert (tmp_path / "report.json").read_bytes() == before
