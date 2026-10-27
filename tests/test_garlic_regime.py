import datetime as dt
import json

import pytest

from agri_data import adaptive_selector
from agri_data import garlic_regime as regime


def _observed_series(origin, recent_values, prior_values, *, newest_age=1):
    result = {}
    for offset, value in zip((1, 2, 3), recent_values):
        result[origin - dt.timedelta(days=offset)] = value
    for offset, value in zip(range(8, 17), prior_values):
        result[origin - dt.timedelta(days=offset)] = value
    if newest_age > 3:
        result.pop(origin - dt.timedelta(days=1), None)
        result[origin - dt.timedelta(days=newest_age)] = recent_values[0]
    return result


def _source(n=40):
    issues = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    target_dates = {origin + dt.timedelta(days=h) for origin in issues for h in regime.HORIZONS}
    target_dates.update(dt.date(2023, 12, 4) + dt.timedelta(days=i) for i in range(28))
    observations = [{"date": day.isoformat(), "actual": 100.0 + (day.toordinal() % 7)} for day in sorted(target_dates)]
    baseline, ridge, hist = [], [], []
    actuals = {dt.date.fromisoformat(row["date"]): row["actual"] for row in observations}
    for origin in issues:
        issue_index = (origin - issues[0]).days
        for h in regime.HORIZONS:
            target = origin + dt.timedelta(days=h)
            actual = actuals[target]
            baseline.append({"origin": origin.isoformat(), "target": target.isoformat(), "horizon": h, "predicted": actual + 4.0 + issue_index % 4})
            ridge_error = (-2.0, 1.0, 4.0, 2.0, -1.0)[issue_index % 5]
            ridge.append({"origin": origin.isoformat(), "target": target.isoformat(), "horizon": h, "predicted": actual + ridge_error})
            hist.append({"origin": origin.isoformat(), "target": target.isoformat(), "horizon": h, "predicted": actual + 3.0})
    products = []
    for product in adaptive_selector.PRODUCTS:
        products.append({"id": product, "observations": observations if product == "garlic" else [],
            "predictions": baseline if product == "garlic" else [],
            "runs": [{"modelId": "ridge", "cadence": "daily", "predictions": ridge if product == "garlic" else []},
                     {"modelId": "hist_gb", "cadence": "daily", "predictions": hist if product == "garlic" else []}]})
    return {"schemaVersion": 2, "products": products}


def _selection():
    return {h: {"model": "ridge", "weight": regime.FROZEN_WEIGHTS[h]} for h in regime.HORIZONS}


def test_regime_rule_uses_observed_calendar_windows_and_fixed_boundary():
    origin = dt.date(2025, 4, 1)
    stable = regime.classify_regime(_observed_series(origin, [100, 100, 100], [100] * 9), origin)
    volatile = regime.classify_regime(_observed_series(origin, [104, 104, 104], [100] * 9), origin)
    assert stable["regime"] == "stable" and stable["weight"] == .5
    assert volatile["regime"] == "volatile" and volatile["weight"] == 1.0
    assert stable["latestObservedDate"] < origin.isoformat()
    ratio = __import__("math").nextafter(__import__("math").exp(.03), 0.0)
    exact_boundary = _observed_series(origin, [100 * ratio] * 3, [100] * 9)
    exact = regime.classify_regime(exact_boundary, origin)
    with_future = {**exact_boundary, origin + dt.timedelta(days=1): 1e9}
    assert exact["regime"] == "stable"
    assert regime.classify_regime(with_future, origin)["weight"] == exact["weight"]


def test_regime_fallbacks_for_sparse_or_stale_observations():
    origin = dt.date(2025, 4, 1)
    sparse = regime.classify_regime({origin - dt.timedelta(days=1): 100.0}, origin)
    stale = regime.classify_regime({origin - dt.timedelta(days=15): 100.0}, origin)
    no_prior = regime.classify_regime({origin - dt.timedelta(days=d):100.0 for d in (1,2,3)}, origin)
    assert sparse["weight"] is None and sparse["fallbackReason"] == "insufficient_recent_observations"
    assert stale["weight"] is None and stale["fallbackReason"] == "source_stale_or_missing"
    assert no_prior["fallbackReason"] == "insufficient_prior_observations"


def test_revise_falls_back_to_frozen_static_for_invalid_candidate():
    static = regime._revise(1e308, 1e-308, .75, {"weight": 1.0, "regime": "volatile", "fallbackReason": None})
    assert static["prediction"] == pytest.approx(2.5e307)
    assert static["regimePrediction"] is None
    assert static["fallbackReason"] == "invalid_regime_candidate"


def test_assessment_builds_fixed_families_pairs_and_serializable_compact_report():
    source = _source()
    result = regime.build_assessment(source, _selection())
    assert len(result["comparisons"]) == 60
    assert len(result["pairedMetrics"]) == 30
    assert all(row["n"] == 40 for row in result["comparisons"])
    assert all(len(result["multipleTesting"][f"HAC{lag}"]) == 60 for lag in (9, 28))
    assert all(t["pValue"] is not None for lag in (9, 28) for t in result["multipleTesting"][f"HAC{lag}"])
    assert all(len({(t["horizon"], t["comparator"], t["loss"]) for t in result["multipleTesting"][f"HAC{lag}"]}) == 60 for lag in (9, 28))
    assert all(t["status"] == "tested" and "holmPValue" in t and "holmSignificant" in t
               for lag in (9, 28) for t in result["multipleTesting"][f"HAC{lag}"])
    h1 = [row for row in result["pairedMetrics"] if row["horizon"] == 1]
    by_comparator = {row["comparatorName"]: row["comparator"]["mae"] for row in h1}
    assert len(set(by_comparator.values())) == 3
    garlic = next(product for product in source["products"] if product["id"] == "garlic")
    actuals = {dt.date.fromisoformat(row["date"]): row["actual"] for row in garlic["observations"]}
    baselines = {(dt.date.fromisoformat(row["origin"]), dt.date.fromisoformat(row["target"])): row["predicted"]
                 for row in garlic["predictions"] if row["horizon"] == 1}
    ridges = {(dt.date.fromisoformat(row["origin"]), dt.date.fromisoformat(row["target"])): row["predicted"]
              for row in garlic["runs"][0]["predictions"] if row["horizon"] == 1}
    common = [(origin, target) for origin, target in baselines if origin >= regime.EVAL_START and regime.EVAL_START <= target <= regime.EVAL_END and (origin, target) in ridges and target in actuals]
    expected_static_mae = sum(abs(actuals[target] - (baselines[origin, target] + .75 * (ridges[origin, target] - baselines[origin, target]))) for origin, target in common) / len(common)
    static_row = next(row for row in h1 if row["comparatorName"] == "static")
    assert static_row["comparator"]["mae"] == pytest.approx(expected_static_mae)
    assert result["coverage"]["1"]["candidate"] == result["coverage"]["1"]["static"] == result["coverage"]["1"]["persistence"] == result["coverage"]["1"]["daily_ridge"] == 40
    json.dumps(result, allow_nan=False)


def test_unavailable_tests_stay_in_fixed_families_with_p_one():
    source = _source(n=10)
    result = regime.build_assessment(source, _selection())
    assert len(result["comparisons"]) == 60
    assert all(t["n"] == 10 for t in result["comparisons"])
    assert all(t["dm"]["9"]["pValue"] == 1.0 and t["dm"]["28"]["pValue"] == 1.0 for t in result["comparisons"])
    assert all(c["conclusion"] == "inference_unavailable" for c in result["conclusions"])


def test_frozen_static_weights_are_enforced():
    bad = _selection(); bad[3] = {"model": "ridge", "weight": .75}
    with pytest.raises(ValueError, match="frozen weights"):
        regime.build_assessment(_source(), bad)
    with pytest.raises(ValueError, match="input SHA-256"):
        regime.build_assessment(_source(), _selection(), input_sha256="0" * 64)


def test_candidate_coverage_gate_compares_independent_baseline_coverage():
    source = _source()
    garlic = next(product for product in source["products"] if product["id"] == "garlic")
    ridge_run = next(run for run in garlic["runs"] if run["modelId"] == "ridge")
    ridge_run["predictions"].pop()
    result = regime.build_assessment(source, _selection())
    horizon = 10
    conclusion = next(row for row in result["conclusions"] if row["horizon"] == horizon)
    assert result["coverage"][str(horizon)]["candidate"] < result["coverage"][str(horizon)]["persistence"]
    assert conclusion["coverageNondeclining"] is False


def test_build_assessment_keeps_sparse_classifier_fallback_and_applied_weight_separate():
    source = _source()
    garlic = next(product for product in source["products"] if product["id"] == "garlic")
    cutoff = dt.date(2024, 1, 1)
    garlic["observations"] = [row for row in garlic["observations"]
        if dt.date.fromisoformat(row["date"]) >= cutoff or dt.date.fromisoformat(row["date"]) in {
            cutoff - dt.timedelta(days=1), cutoff - dt.timedelta(days=2)}]
    result = regime.build_assessment(source, _selection())
    audit = next(row for row in result["horizonSummary"] if row["horizon"] == 1)["decisionAudit"]
    sparse = next(row for row in audit if row["regime"] == "fallback" and row["fallbackReason"] == "insufficient_recent_observations")
    assert sparse["appliedWeight"] == .75
    assert sparse["fallbackReason"] == "insufficient_recent_observations"
    assert sparse["classificationFallbackReason"] == "insufficient_recent_observations"


def test_build_assessment_preserves_invalid_regime_candidate_reason_with_static_fallback():
    source = _source()
    garlic = next(product for product in source["products"] if product["id"] == "garlic")
    origin = dt.date(2024, 1, 10)
    for row in garlic["observations"]:
        day = dt.date.fromisoformat(row["date"])
        if origin - dt.timedelta(days=7) <= day < origin:
            row["actual"] = 200.0
        elif origin - dt.timedelta(days=28) <= day < origin - dt.timedelta(days=7):
            row["actual"] = 100.0
    target = origin + dt.timedelta(days=1)
    baseline_row = next(row for row in garlic["predictions"] if row["origin"] == origin.isoformat() and row["target"] == target.isoformat() and row["horizon"] == 1)
    ridge_row = next(row for row in garlic["runs"][0]["predictions"] if row["origin"] == origin.isoformat() and row["target"] == target.isoformat() and row["horizon"] == 1)
    baseline_row["predicted"] = 1e308
    ridge_row["predicted"] = 1e-308
    result = regime.build_assessment(source, _selection())
    audit = next(row for row in result["horizonSummary"] if row["horizon"] == 1)["decisionAudit"]
    invalid = next(row for row in audit if row["fallbackReason"] == "invalid_regime_candidate")
    assert invalid["regime"] == "volatile"
    assert invalid["appliedWeight"] == .75
    assert invalid["classificationFallbackReason"] is None
