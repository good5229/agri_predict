import csv

import numpy as np
import pytest

from agri_data import period_supply_scenario_score as m


def _row(index, *, actual=100.0, base=100.0, d0=99.0, d1=98.0, crop="cabbage", horizon=2):
    return {"targetIndex": index, "actual": actual, "base": base, "d0": d0, "d1": d1,
            "crop": crop, "horizon": horizon}


def test_dm_hac_uses_target_calendar_gaps_instead_of_compressing_them():
    sparse = [(0, 1.0), (1, 2.0), (20, 3.0), (21, 4.0), (40, 5.0), (41, 6.0)]
    result = m._dm_test(sparse, lag=3)
    values = np.asarray([1, 2, 3, 4, 5, 6], dtype=float)
    centered = values - values.mean()
    n = len(values)
    expected_lrv = float(np.sum(centered**2) / n)
    expected_lrv += 2 * (1 - 1 / 4) * float((centered[0] * centered[1] + centered[2] * centered[3] + centered[4] * centered[5]) / n)
    expected_stat = values.mean() / np.sqrt(expected_lrv / n)
    assert result["status"] == "computed"
    assert result["n"] == 6
    assert result["statistic"] == pytest.approx(expected_stat)


def test_dm_marks_insufficient_temporal_support_and_holm_keeps_full_family():
    unsupported = m._dm_test([(0, 1.0), (1, 2.0), (12, 3.0)], lag=3)
    assert unsupported["status"] == "unsupported"
    family = [{"hypothesisId": f"h{i}", "pRaw": 0.001 if i == 0 else None} for i in range(16)]
    m._holm(family, 16)
    assert len(family) == 16
    assert family[0]["pHolm"] == pytest.approx(0.016)
    assert all(row["pHolm"] == 1.0 for row in family[1:])


def test_dm_family_has_16_slots_for_both_hac_lags_including_unsupported_cells():
    rows = [_row(i, d0=100 + (i % 7), d1=99 + (i % 5), base=101 + (i % 4)) for i in range(40)]
    rows += [_row(i, crop="radish", horizon=3) for i in range(5)]
    result = m.dm_family(rows)
    assert len(result) == 32
    assert {row["hacLagDekads"] for row in result} == {3, 6}
    for lag in (3, 6):
        assert sum(row["hacLagDekads"] == lag for row in result) == 16
        assert any(row["status"] == "unsupported" and row["crop"] == "radish" and row["horizon"] == 3 for row in result)


def test_calendar_bootstrap_block_draws_retain_missing_dekad_positions():
    calendar = list(range(0, 13))
    draw = m._block_resample_indices(calendar, 6, np.random.default_rng(20261016))
    assert len(draw) == len(calendar)
    # The input calendar can contain positions without any target row; the
    # resampler still samples those absolute dekad positions.
    gapped_rows = [_row(i) for i in (0, 1, 10, 11)]
    mean = m._mean_for_boot(gapped_rows, "d1", "base", "mae", [0, 1, 2, 3, 10, 11])
    assert mean is not None
    assert mean == pytest.approx(2.0)


def test_metrics_are_paired_and_equal_cell_aggregate_weights_cells_equally():
    rows = [_row(0, actual=100, d0=90, d1=80, base=100),
            _row(1, actual=100, d0=90, d1=80, base=100),
            _row(0, actual=100, d0=0, d1=100, base=100, crop="radish", horizon=2)]
    differences = m._comparison_differences(rows, "d1", "d0", "mae")
    assert [value for _, value in differences] == [10.0, 10.0, -100.0]
    first = m._cell_metric(rows[:2], "d1")
    second = m._cell_metric(rows[2:], "d1")
    assert first["mae"] == 20.0
    assert second["mae"] == 0.0


def test_matched_fact_counts_preserve_numeric_and_update_group_support(tmp_path):
    path = tmp_path / "matched-facts.csv"
    fields = ["origin", "crop", "horizon", "targetStart", "targetEnd", "availability", "factId",
              "value", "valueLower", "valueUpper", "tieConflict", "updateGroupId"]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"origin": "2023-01-01", "crop": "cabbage", "horizon": 2,
            "targetStart": "2023-01-11", "targetEnd": "2023-01-20", "availability": "matched",
            "factId": "f1", "value": "", "valueLower": "100", "valueUpper": "", "tieConflict": "false", "updateGroupId": "u1"})
        writer.writerow({"origin": "2023-01-01", "crop": "cabbage", "horizon": 2,
            "targetStart": "2023-01-11", "targetEnd": "2023-01-20", "availability": "matched",
            "factId": "f2", "value": "200", "valueLower": "", "valueUpper": "", "tieConflict": "true", "updateGroupId": "u1"})
        writer.writerow({"origin": "2023-01-01", "crop": "cabbage", "horizon": 2,
            "targetStart": "2023-01-11", "targetEnd": "2023-01-20", "availability": "none", "factId": ""})
    counts = m._matched_fact_counts(tmp_path)
    row = counts["cabbage|2|2023-01-11|2023-01-20|2023-01-01"]
    assert row == {"matchedFactCount": 2, "matchedNumericFactCount": 1, "matchedUpdateGroupIds": ["u1"]}


def test_observed_improvement_requires_both_mape_gains_mae_nonworsening_and_coverage():
    rows = [{"aggregate": "supply_4cell", "split": "primary_evaluation", "prediction": method,
             "mape_pct": mape, "mae": mae, "nRowsSum": 400, "cellsWithScores": 4}
            for method, mape, mae in (("base", 12.0, 100.0), ("d0", 11.0, 90.0), ("d1", 10.0, 90.0))]
    assert m._observed_improvement(rows)["status"] == "observed_improvement"
    rows[-1]["nRowsSum"] = 399
    result = m._observed_improvement(rows)
    assert result["status"] == "criterion_not_met"
    assert not result["checks"]["samePairedRowCoverage"]


def test_support_diagnostics_split_by_year_and_record_update_reuse(tmp_path):
    rows = [
        {**_row(2023 * 36, actual=100, base=100, d0=99, d1=98), "targetStart": "2023-01-01",
         "targetEnd": "2023-01-10", "origin": "2022-12-20", "split": "development",
         "rowKey": "key-1",
         "selectedFactIds": ["f1"], "selectedUpdateGroups": ["u1"], "matchedFactCount": 1,
         "matchedNumericFactCount": 1, "matchedUpdateGroupIds": ["u1"], "supportedCurrentNumericColumns": ["v1"],
         "trainingOrigins": 12, "fallbackReason": None, "targetStatus": "observed"},
        {**_row(2023 * 36 + 1, actual=100, base=100, d0=99, d1=98), "targetStart": "2023-01-11",
         "targetEnd": "2023-01-20", "origin": "2022-12-30", "split": "development",
         "rowKey": "key-2",
         "selectedFactIds": ["f1"], "selectedUpdateGroups": ["u1"], "matchedFactCount": 1,
         "matchedNumericFactCount": 1, "matchedUpdateGroupIds": ["u1"], "supportedCurrentNumericColumns": ["v1"],
         "trainingOrigins": 12, "fallbackReason": None, "targetStatus": "observed"},
    ]
    m._mark_new_updates(rows)
    diagnostics = m._diagnostics(rows)
    assert len(diagnostics) == 1
    assert diagnostics[0]["targetYear"] == 2023
    assert diagnostics[0]["firstNewUpdateRows"] == 1
    assert diagnostics[0]["repeatedUpdateGroupOriginAppearances"] == 1
    m._write_csv(tmp_path / "diagnostics.csv", diagnostics)
    assert (tmp_path / "diagnostics.csv").stat().st_size > 0


def test_primary_inference_selector_excludes_development_even_if_effects_are_large():
    evaluation = [_row(i, d0=100.0, d1=99.0, base=100.0) | {"split": "evaluation"} for i in range(40)]
    development = [_row(i, d0=0.0, d1=250.0, base=0.0) | {"split": "development"} for i in range(40)]
    selected = m.primary_evaluation_rows([*evaluation, *development])
    assert selected == evaluation
    assert m.dm_family(selected) == m.dm_family(evaluation)
    assert m.block_bootstrap(selected, replicates=2) == m.block_bootstrap(evaluation, replicates=2)
