import datetime as dt
import json

import numpy as np
import pytest

from agri_data import final_ensemble_onion as onion


def test_weighted_loss_is_equal_horizon_and_masks_labels():
    records = [
        {"targets": [10.0] * 10, "targetMask": [True] * 10},
        {"targets": [20.0] * 10, "targetMask": [True] + [False] * 9},
    ]
    pred = [[11.0] * 10, [20.0] * 10]
    # h1: mean(|e|/y + |e|/q); other horizons only have 1 row.
    assert onion.weighted_price_loss(pred, records, 10.0, 1.0) == pytest.approx(
        (0.5 * (0.1 + 0.1) + 9 * (0.1 + 0.1)) / 10
    )


def test_rank_prefers_raw_eligible_then_dimensionless_ratio_then_earlier_epoch():
    baseline = {"MAPE": 10.0, "MAE": 100.0}
    eligible = {"MAPE": 10.0 - 1e-14, "MAE": 100.0, "epoch": 9}
    ineligible = {"MAPE": 8.0, "MAE": 101.0, "epoch": 1}
    assert onion._rank_cmp(eligible, ineligible, baseline) < 0
    better = {"MAPE": 9.0, "MAE": 99.0, "epoch": 3}
    worse = {"MAPE": 10.0, "MAE": 110.0, "epoch": 2}
    assert onion._rank_cmp(better, worse, baseline) < 0
    tie_a = {"MAPE": 12.0, "MAE": 120.0, "epoch": 4}
    tie_b = {"MAPE": 12.0, "MAE": 120.0, "epoch": 5}
    assert onion._rank_cmp(tie_a, tie_b, baseline) < 0


def test_b_alignment_requires_every_valid_training_label():
    origin = dt.date(2025, 1, 1)
    rec = {"origin": origin, "targets": [1.0] * 10, "targetMask": [True] * 10}
    with pytest.raises(ValueError, match="missing eligible B"):
        onion._with_b([rec], {})
    aligned = onion._with_b([rec], {(origin, h): float(h) for h in range(1, 11)})
    assert aligned[0]["B"] == list(map(float, range(1, 11)))


def test_t26_uses_july_first_origin_for_the_monthly_fit_grid(tmp_path, monkeypatch):
    monkeypatch.setattr(onion, "OUT", tmp_path)
    row = {"product": "onion", "origin": "2026-07-01", "target": "2026-07-11",
        "horizon": 10, "B": 100.0, "BSourceStatus": "issued"}
    (tmp_path / "baseline-cache.json").write_text(json.dumps({"rows": [row]}))
    selected = onion._phase_baseline_rows("T26")
    assert len(selected) == 1
    assert selected[0]["originDate"] == dt.date(2026, 7, 1)


def test_optimizer_factor_two_changes_only_f_objective_scale():
    records = [{"targets": [100.0]*10, "targetMask": [True]*10}]
    pred = [[90.0]*10]
    b = [[100.0]*10]
    ld = onion.weighted_price_loss(pred, records, 100.0)
    lf = onion.weighted_price_loss(pred, records, 100.0, blend=[[95.0]*10])
    assert ld == pytest.approx(0.20)
    assert lf == pytest.approx(0.10)
    assert 2*lf == pytest.approx(ld)


def test_cross_loss_and_positive_representation_floor_diagnostics():
    records = [{"targets": [40.0]*10, "targetMask": [True]*10,
                "B": [100.0]*10}]
    d, f = [[60.0]*10], [[2.0]*10]
    report = onion.cross_loss_diagnostics(records, d, f, 100.0, 1.0)
    assert report["D"]["LD"] == pytest.approx(0.7)
    assert report["D"]["price"]["MAE"] == pytest.approx(40.0)
    assert report["B"]["MAE"] == pytest.approx(60.0)
    assert report["F"]["LF"] < report["D"]["LF"]
    floor = onion.representation_floor(records, f)
    assert floor["strictlyAboveHalfBCount"] == 10
    assert floor["targetBelowHalfBCount"] == 10
    assert floor["sumTargetBelowHalfBShortfall"] == pytest.approx(100.0)


def test_d25_lambda_selection_is_conditional_and_defaults_to_one_on_ties(tmp_path, monkeypatch):
    monkeypatch.setattr(onion, "OUT", tmp_path)
    monkeypatch.setattr(onion, "validate_config", lambda: {"configSha256": "test-config"})
    forecast = tmp_path / "forecast.json"
    forecast.write_text("{}")
    base = {"phase": "D25", "lambda": 1, "configSha256": "test-config", "pairedPriceCoverage": True,
            "forecastPath": str(forecast), "forecastFileSha256": onion.file_sha256(forecast),
            "metrics": {"H_F_MAE": 11.0, "B_MAE": 10.0, "H_F_MAPE": 9.0, "B_MAPE": 10.0}}
    repair = {"phase": "D25", "lambda": 3, "configSha256": "test-config", "pairedPriceCoverage": True,
              "forecastPath": str(forecast), "forecastFileSha256": onion.file_sha256(forecast),
              "metrics": {"H_F_MAE": 10.0, "B_MAE": 10.0, "H_F_MAPE": 8.0, "B_MAPE": 10.0}}
    (tmp_path / "score-D25-L1.json").write_text(json.dumps(base))
    (tmp_path / "score-D25-L3.json").write_text(json.dumps(repair))
    assert onion.select_d25_lambda()["selectedLambda"] == 3
    repair["metrics"]["H_F_MAPE"] = 9.0
    repair["metrics"]["H_F_MAE"] = 11.0
    (tmp_path / "D25-selection.json").unlink()
    (tmp_path / "score-D25-L3.json").write_text(json.dumps(repair))
    assert onion.select_d25_lambda()["selectedLambda"] == 1

