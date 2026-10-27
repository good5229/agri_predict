import datetime as dt
import math

from agri_data import supply_pressure_reentry_pilot as pilot


def _dense_series():
    series = {}
    day = dt.date(2016, 12, 1)
    while day <= dt.date(2017, 1, 10):
        series[day] = (100.0 + (day - dt.date(2016, 12, 1)).days, 10.0)
        day += dt.timedelta(days=1)
    return series


def test_training_samples_exclude_cutoff_and_future_targets(monkeypatch):
    series = _dense_series()
    seen = []

    def fake_candidate(_series, origin, target, horizon):
        seen.append((origin, target, horizon))
        earlier = [day for day in _series if day < origin]
        if not earlier:
            return None
        source = max(earlier)
        return ((float(horizon),), source, False)

    def fake_strict(_series, origin, target, horizon, _days, _logs):
        earlier = [day for day in _series if day < origin]
        if not earlier:
            return None
        return max(earlier), (float(horizon),)

    monkeypatch.setattr(pilot.reentry, "issue_feature", fake_candidate)
    monkeypatch.setattr(pilot.live_forecast, "_strict_feature_row", fake_strict)
    days = tuple(sorted(series))
    logs = tuple(0.0 for _ in days)
    cutoff = dt.date(2017, 1, 8)
    candidate, direct = pilot._training_samples(series, cutoff, {}, {}, days, logs)
    assert candidate and all(target < cutoff for target, _, _ in candidate)
    assert all(target < cutoff for rows in direct.values() for target, _, _ in rows)
    assert all(target < cutoff for _, target, _ in seen)


def test_macro_metrics_equal_weight_horizons_then_products():
    rows = []
    errors_a = {1: 1.0, 5: 3.0, 10: 8.0}
    for product in ("a", "b"):
        for horizon in pilot.HORIZONS:
            error = errors_a[horizon] if product == "a" else 5.0
            rows.append({"productId": product, "horizon": horizon, "actual": 100.0,
                         "candidatePrice": 100.0 + error, "directPrice": 100.0,
                         "persistencePrice": 100.0, "pairedEvaluable": True})
    result = pilot._aggregate_metrics(rows, products=("a", "b"))
    assert math.isclose(result["byModelAndProduct"]["candidatePrice"]["a"]["mape"], 4.0)
    assert math.isclose(result["macro"]["candidatePrice"]["mape"], 4.5)


def test_coverage_reports_product_horizon_decrease():
    rows = []
    for product in ("a", "b"):
        for horizon in pilot.HORIZONS:
            rows.append({"productId": product, "horizon": horizon, "targetStatus": "present",
                         "candidateStatus": "unavailable" if (product, horizon) == ("a", 5) else "available",
                         "directStatus": "available"})
    result = pilot._coverage_gates(rows, products=("a", "b"))
    assert not result["candidateAtLeastDirectEveryProductHorizon"]
    group = next(item for item in result["groups"] if item["productId"] == "a" and item["horizon"] == 5)
    assert group["candidateCoverage"] == 0.0 and group["directCoverage"] == 1.0


def test_growth_gate_combines_point_estimates_and_coverage():
    rows = [{"productId": product, "horizon": horizon, "actual": 100.0,
             "candidatePrice": 99.0, "directPrice": 103.0, "persistencePrice": 103.0,
             "targetStatus": "present", "candidateStatus": "available",
             "directStatus": "available", "pairedEvaluable": True}
            for product in ("a", "b") for horizon in pilot.HORIZONS]
    aggregate = pilot._aggregate_metrics(rows, products=("a", "b"))
    gates = pilot._goal_gates(aggregate, rows, products=("a", "b"))
    assert gates["candidateMacroMapeAtMost10Percent"]
    assert gates["candidateImprovesDirectByAtLeast0_5pp"]
    assert gates["allProductsWithinMapeAndMaeWorseningLimits"]
    assert gates["candidateCoverageAtLeastDirect"]
    assert not gates["allProductHorizonsMeetThreeModelPairMinimum"]
    assert not gates["allGrowthGatesPassed"]
    assert {item["threeModelPairedRows"] for item in gates["threeModelPairedRowsByProductHorizon"]} == {1}


def test_dm_holm_family_has_60_tests_and_marks_small_samples_unavailable():
    tests = pilot._dm_family([], bandwidth=9)
    assert len(tests) == 60
    assert {item["n"] for item in tests} == {0}
    assert {item["conclusion"] for item in tests} == {"inference_unavailable"}
    assert all("holmPValue" in item for item in tests)

    paired = []
    start = dt.date(2024, 1, 1)
    for offset in range(30):
        target = (start + dt.timedelta(days=offset)).isoformat()
        paired.append({"productId": pilot.PRODUCTS[0], "horizon": pilot.HORIZONS[0],
                       "target": target, "actual": 100.0, "candidatePrice": 100.0,
                       "directPrice": 101.0, "persistencePrice": 101.0, "pairedEvaluable": True})
    tested = pilot._dm_family(paired, bandwidth=28)
    group = [item for item in tested if item["productId"] == pilot.PRODUCTS[0]
             and item["horizon"] == pilot.HORIZONS[0]]
    assert len(tested) == 60
    assert all(item["n"] == 30 and item["conclusion"] == "inference_unavailable" for item in group)
    constant = next(item for item in group if item["comparator"] == "direct" and item["loss"] == "absolute")
    assert constant["status"] == "zero_or_invalid_hac_variance"
    assert constant["conclusion"] == "inference_unavailable"
    combined = pilot._combined_dm_conclusions(tested)
    assert combined[0]["conclusion"] == "inference_unavailable"


def test_immutable_output_rejects_replacement(tmp_path):
    output = tmp_path / "pilot.json"
    pilot._write_immutable(output, {"ok": True})
    try:
        pilot._write_immutable(output, {"ok": False})
    except FileExistsError:
        pass
    else:
        raise AssertionError("immutable output unexpectedly replaced")
    assert output.read_text().find('"ok": true') >= 0
