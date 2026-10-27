import datetime as dt
import hashlib
import copy

import pytest
import sklearn.ensemble

from agri_data import ecmwf_nonlinear_pilot as pilot
from agri_data import ecmwf_price_eval as base
from tests.test_ecmwf_price_eval import feature_fixture


def synthetic_inputs():
    features, manifest = feature_fixture()
    prices = {"schemaVersion": 1, "products": []}
    report = {"protocol": {"id": "direct-horizon-ridge-2026-10-06"}, "pairedRows": []}
    for product in pilot.PRODUCTS:
        obs = {}
        for i in range(220):
            day = pilot.TRAIN_START - dt.timedelta(days=10) + dt.timedelta(days=i)
            obs[day.isoformat()] = 100 + i / 10
        for i in range(61 + 10):
            day = pilot.EVAL_START + dt.timedelta(days=i)
            obs[day.isoformat()] = 120 + i / 10
        prices["products"].append({"id": product, "observations": [{"date": d, "actual": v} for d, v in obs.items()]})
        for horizon in range(1, 11):
            for i in range(120):
                target = pilot.TRAIN_START + dt.timedelta(days=i)
                report["pairedRows"].append({"product": product, "horizon": horizon,
                    "target": target.isoformat(), "origin": (target-dt.timedelta(days=horizon)).isoformat(),
                    "direct": obs[target.isoformat()] * .98, "actual": 1e100})
            for i in range(61):
                origin = pilot.EVAL_START + dt.timedelta(days=i)
                target = origin + dt.timedelta(days=horizon)
                report["pairedRows"].append({"product": product, "horizon": horizon,
                    "target": target.isoformat(), "origin": origin.isoformat(),
                    "direct": obs[target.isoformat()] * .98, "actual": 1e200})
    return prices, report, features, manifest


def test_report_parser_never_carries_actual_and_requires_unique_rows():
    raw = {"protocol": {"id": "direct-horizon-ridge-2026-10-06"}, "pairedRows": [
        {"product": pilot.PRODUCTS[0], "horizon": 1, "target": "2026-06-02", "direct": 100, "actual": 1e300}]}
    row = pilot._report_map(raw)[(pilot.PRODUCTS[0], 1, dt.date(2026, 6, 2))]
    assert "actual" not in row
    raw["pairedRows"].append(raw["pairedRows"][0])
    with pytest.raises(pilot.PilotError, match="duplicate"):
        pilot._report_map(raw)


def test_forecast_is_full_answer_free_grid_with_training_date_gate():
    prices, report, features, manifest = synthetic_inputs()
    result = pilot.build_forecast(prices, report, features, manifest)
    assert len(result["forecasts"]) == 915
    assert all("actual" not in row for row in result["forecasts"])
    assert all(v["distinctTrainingTargetDates"] >= 90 for v in result["trainingAudit"].values())
    assert result["forecasts"][0]["nonlinear"] > 0
    # Malicious evaluation actuals in the direct report have no effect on predictions.
    for row in report["pairedRows"]:
        if row["target"] >= pilot.EVAL_START.isoformat():
            row["actual"] = -999999
    changed = pilot.build_forecast(prices, report, features, manifest)
    assert [r["nonlinear"] for r in result["forecasts"]] == [r["nonlinear"] for r in changed["forecasts"]]


def test_missing_direct_is_kept_as_null_and_models_fail_independently(monkeypatch):
    prices, report, features, manifest = synthetic_inputs()
    missing = next(r for r in report["pairedRows"] if r["target"] == "2026-06-02" and r["horizon"] == 1)
    report["pairedRows"].remove(missing)

    class BrokenHGB:
        def __init__(self, **kwargs): pass
        def fit(self, X, y): raise ArithmeticError("synthetic nonlinear failure")

    monkeypatch.setattr(sklearn.ensemble, "HistGradientBoostingRegressor", BrokenHGB)
    result = pilot.build_forecast(prices, report, features, manifest)
    assert len(result["forecasts"]) == 915
    absent = next(r for r in result["forecasts"] if r["product"] == pilot.PRODUCTS[0] and r["origin"] == "2026-06-01" and r["horizon"] == 1)
    assert absent["direct"] is absent["nonlinear"] is absent["linear"] is None
    assert absent["directMissingReason"] == "direct_prediction_missing"
    present = next(r for r in result["forecasts"] if r["product"] == pilot.PRODUCTS[1] and r["origin"] == "2026-06-01" and r["horizon"] == 1)
    assert present["nonlinear"] == present["direct"]
    assert present["nonlinearFallbackReason"].startswith("fit_failed:")
    assert present["linear"] is not None and present.get("linearFallbackReason") is None
    later = next(r for r in result["forecasts"] if r["product"] == pilot.PRODUCTS[1] and r["origin"] == "2026-06-02" and r["horizon"] == 1)
    assert dt.date.fromisoformat(later["persistenceSourceDate"]) < dt.date.fromisoformat(later["origin"])


def test_score_checks_manifest_digest_and_keeps_fixed_dm_family():
    prices, report, features, manifest = synthetic_inputs()
    f = pilot.build_forecast(prices, report, features, manifest, {"prices": "x"})
    f["manifestSha256"] = pilot.sha(pilot.canon(f))
    raw = pilot.canon(f)
    out = pilot.score(f, prices, raw)
    assert len(out["dm"]["comparisons"]) == 90
    assert len(out["byProductHorizon"]) == 15
    f["forecasts"][0]["direct"] += 1
    with pytest.raises(pilot.PilotError, match="SHA"):
        pilot.score(f, prices, raw)


def test_score_rejects_duplicate_grid_and_uses_four_way_common_set():
    prices, report, features, manifest = synthetic_inputs()
    f = pilot.build_forecast(prices, report, features, manifest, {"prices": "x"})
    # Forecast availability is counted over the planned rows independently of
    # whether the scoring join can find an answer.
    prices["products"][0]["observations"] = [r for r in prices["products"][0]["observations"] if r["date"] != "2026-08-10"]
    row = next(r for r in f["forecasts"] if r["persistence"] is not None)
    row["persistence"] = None
    row["persistenceSourceDate"] = None
    row["persistenceMissingReason"] = "synthetic_missing"
    out = pilot.score(f, prices)
    assert out["coverage"]["fourWayCommonRows"] == 898
    assert out["coverage"]["validForecastRows"]["nonlinear"] == 915
    assert out["coverage"]["scoreableValidForecastRows"]["nonlinear"] == 914
    assert len(out["dm"]["comparisons"]) == 90
    assert set(out["gates"]) >= {"all15CellsCommonPairsAtLeast30", "nonlinearCoverageAtLeastDirect", "expand"}
    duplicate = copy.deepcopy(f)
    duplicate["forecasts"][1] = copy.deepcopy(duplicate["forecasts"][0])
    with pytest.raises(pilot.PilotError, match="duplicate"):
        pilot.score(duplicate, prices)

    answer_leak = copy.deepcopy(f)
    answer_leak["forecasts"][0]["actual"] = None
    with pytest.raises(pilot.PilotError, match="actual key"):
        pilot.score(answer_leak, prices)

    for name, reason in (("direct", "directMissingReason"), ("persistence", "persistenceMissingReason")):
        contradictory = copy.deepcopy(f)
        valid = next(r for r in contradictory["forecasts"] if r[name] is not None)
        valid[reason] = "contradictory"
        with pytest.raises(pilot.PilotError, match="contradicts"):
            pilot.score(contradictory, prices)

    contradictory = copy.deepcopy(f)
    valid = next(r for r in contradictory["forecasts"] if r["nonlinear"] is not None)
    valid["nonlinearFallbackReason"] = "fit_failed:synthetic"
    with pytest.raises(pilot.PilotError, match="fallback must equal"):
        pilot.score(contradictory, prices)
    assert pilot.PROTOCOL["expansionGates"]["perProductGateBenchmark"] == "direct"
