from __future__ import annotations

import datetime as dt

import pytest

from agri_data import chronos2_cabbage_followup as f


def _fixture(u=60.0, m=60.0, *, invalid_u_key=None, h_override=None, base_status="issued"):
    forecasts, baseline, actuals = [], [], {}
    for h in range(1, 11):
        for i in range(2):
            origin = dt.date(2026, 4, 1) + dt.timedelta(days=i + h)
            target = origin + dt.timedelta(days=h)
            key = {"product": "cabbage", "originDate": origin.isoformat(),
                   "targetDate": target.isoformat(), "horizon": h}
            b, mv, uv, actual = 100.0, float(m), float(u), 80.0
            hval = 0.5 * b + 0.5 * mv
            if h_override is not None:
                hval += h_override
            actuals[("cabbage", target)] = actual
            baseline.append({**key, "conditions": {"C": {"status": base_status,
                "blendPrediction": b, "fallbackReason": "synthetic-fallback" if base_status == "fallback" else None}}})
            u_item = {"status": "issued", "prediction": uv}
            if invalid_u_key == (h, i):
                u_item = {"status": "invalid_prediction", "prediction": None}
            forecasts.append({"key": key, "context": {"cutoff": (origin-dt.timedelta(days=1)).isoformat()},
                "conditions": {"B": {"status": "issued", "prediction": b,
                                      "baselineSourceStatus": base_status,
                                      "reason": "synthetic-fallback" if base_status == "fallback" else None},
                               "U": u_item,
                               "M": {"status": "issued", "prediction": mv},
                               "H": {"status": "issued", "prediction": hval}}})
    cells = []
    for h in range(1, 11):
        cells.append({"product": "cabbage", "horizon": h,
                      "leftMae": abs(80 - (100 + m) / 2), "rightMae": 20.0,
                      "leftMape": 100*abs(80 - (100 + m) / 2)/80, "rightMape": 25.0})
    return forecasts, baseline, actuals, {"cells": cells}


def test_selects_only_candidate_passing_both_losses_with_same_sample():
    forecasts, baseline, actuals, report = _fixture(u=50, m=140)
    result = f.calculate_selection(forecasts, baseline, actuals, report, expected_pairs=20)
    assert result["selected"] == "HU"
    assert result["eligibleCandidates"] == ["HU"]
    assert result["positiveActualPairs"] == 20
    assert result["metrics"]["HU"]["equalHorizonMae"] == 5
    assert result["metrics"]["HM"]["equalHorizonMae"] == 40
    assert all(result["metrics"][name]["positiveActualPairs"] == 20 for name in ("B", "HU", "HM"))


def test_exact_mape_tie_prefers_simpler_hu():
    forecasts, baseline, actuals, report = _fixture(u=60, m=60)
    result = f.calculate_selection(forecasts, baseline, actuals, report, expected_pairs=20)
    assert result["eligibleCandidates"] == ["HU", "HM"]
    assert result["selected"] == "HU"


def test_candidate_coverage_loss_makes_it_ineligible_without_shrinking_b():
    forecasts, baseline, actuals, report = _fixture(u=50, m=150, invalid_u_key=(1, 0))
    result = f.calculate_selection(forecasts, baseline, actuals, report, expected_pairs=20)
    assert result["coverage"]["HU"]["issued"] == 19
    assert result["coverage"]["HU"]["evaluation"] == 19
    assert result["metrics"]["HU"]["noCoverageLoss"] is False
    assert result["selected"] is None
    assert result["status"] == "NO_ELIGIBLE_CABBAGE_CANDIDATE"


def test_source_baseline_fallback_status_is_preserved_and_h_replay_is_checked():
    forecasts, baseline, actuals, report = _fixture(u=50, m=150, base_status="fallback")
    result = f.calculate_selection(forecasts, baseline, actuals, report, expected_pairs=20)
    assert result["coverage"]["B"]["issued"] == 20
    forecasts[0]["conditions"]["B"]["reason"] = None
    with pytest.raises(ValueError, match="fallback reason"):
        f.calculate_selection(forecasts, baseline, actuals, report, expected_pairs=20)
    forecasts[0]["conditions"]["B"]["reason"] = "synthetic-fallback"
    forecasts[0]["conditions"]["H"]["prediction"] += 0.01
    with pytest.raises(ValueError, match="original H"):
        f.calculate_selection(forecasts, baseline, actuals, report, expected_pairs=20)


def test_rejects_missing_or_duplicate_source_keys():
    forecasts, baseline, actuals, report = _fixture()
    with pytest.raises(ValueError, match="exactly match"):
        f.calculate_selection(forecasts[:-1], baseline, actuals, report, expected_pairs=20)
    with pytest.raises(ValueError, match="duplicate"):
        f.calculate_selection(forecasts + forecasts[:1], baseline, actuals, report, expected_pairs=20)


def test_rejects_wrong_positive_pair_count_and_report_reproduction():
    forecasts, baseline, actuals, report = _fixture()
    with pytest.raises(ValueError, match="positive B/actual pairs"):
        f.calculate_selection(forecasts, baseline, actuals, report, expected_pairs=651)
    report["cells"][0]["leftMae"] += 2.0
    with pytest.raises(ValueError, match="reproduce original report"):
        f.calculate_selection(forecasts, baseline, actuals, report, expected_pairs=20)


def test_actual_extraction_filters_before_numeric_conversion_and_excludes_out_of_window():
    raw = {"schemaVersion": 1, "products": [
        {"id": "cabbage", "observations": [
            {"date": "2026-06-20", "actual": "80", "volumeKg": "2"},
            {"date": "2026-06-21", "actual": "T_POISON", "volumeKg": "bad"},
            {"date": "2026-07-11", "actual": "T_POISON", "volumeKg": "bad"}]},
        {"id": "radish", "observations": [
            {"date": "2026-04-01", "actual": "OTHER_CROP_POISON", "volumeKg": "bad"}]},
        {"id": "onion", "observations": []}, {"id": "garlic", "observations": []},
        {"id": "pepper", "observations": []}]}

    actuals = f._stage_a_actuals(raw)

    assert actuals == {("cabbage", dt.date(2026, 6, 20)): (80.0, 2.0)}
