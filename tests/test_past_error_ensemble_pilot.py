import datetime as dt

import pytest

from agri_data import past_error_ensemble_pilot as pilot


def row(product="cabbage", horizon=1, target=dt.date(2026, 8, 1), *, actual=100.0,
        direct=100.0, fixed=100.0, seasonal=100.0, market=100.0, persistence=100.0):
    return {"product": product, "origin": target - dt.timedelta(days=horizon), "target": target,
            "horizon": horizon, "actual": actual, "directRidge": direct, "fixedBlend": fixed,
            "seasonalAnalog": seasonal, "marketRecompose": market, "persistence": persistence,
            "baselineAvailable": True, "fixedBlendStatus": None, "seasonalAnalogStatus": None,
            "marketRecomposeStatus": None}


def report(rows, candidate_field, direct_field="directRidge"):
    value_field = {"candidate": "fixedBlend", "selectedPrediction": "seasonalAnalog",
                   "candidatePrediction": "marketRecompose"}[candidate_field]
    out = []
    for r in rows:
        out.append({"product": r["product"], "origin": r["origin"].isoformat(), "target": r["target"].isoformat(),
                    "horizon": r["horizon"], "actual": r["actual"], direct_field: r["directRidge"],
                    candidate_field: r[value_field],
                    "persistence": r["persistence"]})
    return {"pairedRows": out}


def reports(rows):
    return (report(rows, "candidate", "direct_ridge"), report(rows, "selectedPrediction"),
            report(rows, "candidatePrediction"))


def keyed(rows):
    return {(r["product"], r["origin"], r["target"], r["horizon"]): r for r in rows}


def test_validate_reports_requires_duplicate_free_identity_and_consistent_values():
    base = row()
    a, b, c = reports([base])
    assert list(pilot.validate_reports(a, b, c).values())[0]["marketRecompose"] == 100

    a["pairedRows"].append(a["pairedRows"][0])
    with pytest.raises(ValueError, match="duplicate"):
        pilot.validate_reports(a, b, c)

    a, b, c = reports([base])
    c["pairedRows"][0]["actual"] = 101
    with pytest.raises(ValueError, match="disagree on actual"):
        pilot.validate_reports(a, b, c)

    a, b, c = reports([base])
    b["pairedRows"][0]["origin"] = "2026-07-30"
    with pytest.raises(ValueError, match="origin/horizon"):
        pilot.validate_reports(a, b, c)

    a, b, c = reports([base])
    c["pairedRows"].clear()
    validated = pilot.validate_reports(a, b, c)
    one = next(iter(validated.values()))
    assert one["baselineAvailable"] is True
    assert one["marketRecompose"] is None
    assert one["marketRecomposeStatus"] == "missing_report_key"

    a, b, c = reports([base])
    b["pairedRows"][0]["selectedPrediction"] = 0
    validated = pilot.validate_reports(a, b, c)
    one = next(iter(validated.values()))
    assert one["seasonalAnalog"] is None
    assert one["seasonalAnalogStatus"] == "invalid_forecast"


def test_weights_use_historical_mean_ape_points_and_normalize():
    history = [row(actual=100, fixed=110, seasonal=120, market=100),
               row(actual=100, fixed=110, seasonal=120, market=100)]
    means, result = pilot.weights(history)
    assert means == {"directRidge": 0, "fixedBlend": 10, "seasonalAnalog": 20, "marketRecompose": 0}
    assert sum(result.values()) == pytest.approx(1)
    assert result["directRidge"] == result["marketRecompose"]
    assert result["marketRecompose"] > result["fixedBlend"] > result["seasonalAnalog"]


def test_history_filter_excludes_embargo_boundary_and_older_than_365_days():
    origin = dt.date(2026, 8, 10)
    horizon = 3
    current = row(target=origin + dt.timedelta(days=horizon), horizon=horizon)
    rows = {("cabbage", current["origin"], current["target"], horizon): current}
    # Exactly 20 targets are eligible: [origin-365, origin-14).
    for i in range(20):
        target = origin - dt.timedelta(days=15 + i)
        old = row(target=target, horizon=horizon, actual=100, fixed=110, seasonal=120, market=100)
        rows[(old["product"], old["origin"], old["target"], horizon)] = old
    boundary = row(target=origin - dt.timedelta(days=14), horizon=horizon)
    rows[(boundary["product"], boundary["origin"], boundary["target"], horizon)] = boundary
    too_old = row(target=origin - dt.timedelta(days=366), horizon=horizon)
    rows[(too_old["product"], too_old["origin"], too_old["target"], horizon)] = too_old

    result = pilot._forecast(rows, current)
    assert result["eligibleHistoryTargets"] == 20
    assert result["fallback"] is False
    assert result["weights"]["marketRecompose"] > result["weights"]["seasonalAnalog"]


def test_candidate_weights_all_four_including_direct_then_applies_fixed_half_mix():
    origin = dt.date(2026, 8, 10)
    current = row(target=origin + dt.timedelta(days=1), direct=100, fixed=120, seasonal=130, market=140)
    rows = {("cabbage", current["origin"], current["target"], 1): current}
    for i in range(20):
        past = row(target=origin - dt.timedelta(days=15 + i), actual=100,
                   direct=100, fixed=110, seasonal=120, market=130)
        rows[(past["product"], past["origin"], past["target"], 1)] = past
    means, w = pilot.weights([r for r in rows.values() if r is not current])
    expected = 0.5 * current["directRidge"] + 0.5 * sum(w[name] * current[name] for name in pilot.METHODS)
    result = pilot._forecast(rows, current)
    assert set(result["weights"]) == {"directRidge", "fixedBlend", "seasonalAnalog", "marketRecompose"}
    assert result["historicalMeanApe"] == means
    assert result["candidate"] == pytest.approx(expected)


def test_current_missing_forecast_falls_back_even_with_sufficient_history():
    origin = dt.date(2026, 8, 10)
    current = row(target=origin + dt.timedelta(days=1), direct=93, market=100)
    current["seasonalAnalog"] = None
    current["seasonalAnalogStatus"] = "invalid_forecast"
    rows = {("cabbage", current["origin"], current["target"], 1): current}
    for i in range(20):
        past = row(target=origin - dt.timedelta(days=15 + i))
        rows[(past["product"], past["origin"], past["target"], 1)] = past
    result = pilot._forecast(rows, current)
    assert result["candidate"] == 93
    assert result["fallback"] is True
    assert result["missingMethods"] == ["seasonalAnalog"]


def test_under_minimum_history_falls_back_to_direct_with_count_and_reason():
    target = dt.date(2026, 8, 1)
    current = row(target=target, direct=93)
    rows = {("cabbage", current["origin"], target, 1): current}
    result = pilot._forecast(rows, current)
    assert result["candidate"] == 93
    assert result["fallback"] is True
    assert result["eligibleHistoryTargets"] == 0
    assert result["fallbackReason"] == "fewer_than_20_causal_history_targets"


def test_macro_requires_all_cells_for_five_crop_equal_weight_denominator():
    cells = []
    for p in pilot.PRODUCTS:
        for h in pilot.HORIZONS:
            metrics = {name: {"n": 1, "mae": 2.0, "mape": 4.0} for name in ("candidate", "directRidge", "persistence")}
            cells.append({"product": p, "horizon": h, "metrics": metrics})
    full = pilot._macro(cells)
    assert full["fiveProductEqualWeight"]["candidate"] == {
        "mape": 4, "mae": 2, "availableProducts": 5, "totalProducts": 5}
    cells.pop()
    partial = pilot._macro(cells)
    assert partial["fiveProductEqualWeight"]["candidate"]["mape"] is None
    assert partial["byProduct"]["pepper"]["candidate"]["availableHorizons"] == 9


def test_expansion_gate_uses_within_crop_equal_horizon_mae_ratio():
    cells, coverage, macro = [], {}, {"byProduct": {}, "fiveProductEqualWeight": {}}
    for p in pilot.PRODUCTS:
        macro["byProduct"][p] = {
            "candidate": {"equalHorizonMape": 8.0, "equalHorizonMae": 10.5},
            "directRidge": {"equalHorizonMape": 8.6, "equalHorizonMae": 10.0},
        }
        for h in pilot.HORIZONS:
            cells.append({"product": p, "horizon": h, "n": 35,
                          "coverage": {"nonfallbackAllFourCount": 30}})
            coverage[f"{p}/{h}"] = {"candidateAvailableCount": 35, "directRidgeAvailableCount": 35,
                                    "persistenceAvailableCount": 35}
    macro["fiveProductEqualWeight"] = {
        "candidate": {"mape": 8.0}, "directRidge": {"mape": 8.6}}
    gate = pilot._expansion_gate(cells, macro, coverage)
    assert gate["byProduct"]["cabbage"]["maeWorseningFraction"] == pytest.approx(0.05)
    assert gate["checks"]["eachProductEqualHorizonMaeWorseningAtMost5Percent"] is True
    macro["byProduct"]["cabbage"]["candidate"]["equalHorizonMae"] = 10.51
    assert pilot._expansion_gate(cells, macro, coverage)["checks"]["eachProductEqualHorizonMaeWorseningAtMost5Percent"] is False


def test_calendar_dm_and_holm_keep_unavailable_and_directional_conclusions():
    rows = []
    for i in range(35):
        actual = 100.0
        rows.append({"target": dt.date(2026, 1, 1) + dt.timedelta(days=i), "actual": actual,
                     "candidate": 100.0 + (i % 2), "directRidge": 110.0, "persistence": 110.0})
    improved = pilot._dm(rows, "directRidge", "mae", 9)
    assert improved["status"] == "available"
    assert improved["meanLossDifference"] < 0
    family = []
    for i in range(pilot.FAMILY_SIZE):
        family.append({"product": "cabbage", "horizon": i + 1, "comparator": "directRidge", "loss": "mae",
                       "status": "available", "meanLossDifference": -1, "pValue": 0.00001})
    family[0].update(status="inference_unavailable_insufficient_pairs", meanLossDifference=None, pValue=1.0)
    pilot._holm(family)
    assert family[0]["conclusion"] == "inference_unavailable"
    assert all(r["conclusion"] == "improvement" for r in family[1:])

    worsened = [{"product": "pepper", "horizon": 1, "comparator": "directRidge", "loss": "mae",
                 "status": "available", "meanLossDifference": 1, "pValue": 0.00001}]
    pilot._holm(worsened)
    assert worsened[0]["conclusion"] == "worsening"


def test_end_to_end_analyze_handles_date_dm_holm_and_full_coverage_accounting():
    rows = {}
    for product in pilot.PRODUCTS:
        for horizon in pilot.HORIZONS:
            for i in range(50):
                target = dt.date(2026, 5, 1) + dt.timedelta(days=i)
                actual = 100.0 + (i % 7)
                history = row(product, horizon, target, actual=actual, direct=actual + 4,
                              fixed=actual + 5, seasonal=actual + 6, market=actual + 7,
                              persistence=actual + 3)
                rows[(product, history["origin"], target, horizon)] = history
            for i in range(35):
                target = dt.date(2026, 8, 1) + dt.timedelta(days=i)
                actual = 100.0 + (i % 7)
                current = row(product, horizon, target, actual=actual,
                              direct=actual + (4 if i % 2 else 5), fixed=actual + 2,
                              seasonal=actual + 3, market=actual + 4, persistence=actual + 5)
                rows[(product, current["origin"], target, horizon)] = current
    report_out = pilot.analyze(rows)
    assert report_out["coverage"]["plannedTargetGridCount"] == 55 * 5 * 10
    assert report_out["coverage"]["targetObservedCount"] == 35 * 5 * 10
    assert report_out["coverage"]["allFourForecastsAvailableCount"] == 35 * 5 * 10
    assert report_out["coverage"]["nonfallbackAllFourCount"] == 35 * 5 * 10
    assert len(report_out["multipleTesting"]["HAC9"]) == pilot.FAMILY_SIZE
    assert len(report_out["multipleTesting"]["HAC28"]) == pilot.FAMILY_SIZE
    assert all(cell["coverage"]["nonfallbackAllFourCount"] == 35 for cell in report_out["cells"])
    assert report_out["pairedRows"][0]["target"] == "2026-08-01"
