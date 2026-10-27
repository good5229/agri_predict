import copy
import datetime as dt
import math

import pytest

from agri_data import analog_residual_pilot as pilot


def synthetic_inputs(constant=False):
    prices = {"schemaVersion": 1, "products": []}
    report = {"schemaVersion": 1, "protocol": {"id": "direct-horizon-ridge-2026-10-06"}, "pairedRows": []}
    for pi, product in enumerate(pilot.PRODUCTS):
        observations = []
        day = pilot.TRAIN_END - dt.timedelta(days=350)
        for i in range(460):
            d = day + dt.timedelta(days=i)
            value = 100.0 if constant else 100 + pi * 3 + i * .03 + math.sin(i / 11)
            observations.append({"date": d.isoformat(), "actual": value, "volumeKg": 1000 + i % 17})
        prices["products"].append({"id": product, "observations": observations})
        # 110 distinct pre-cutoff target dates per product and horizon.
        for h in pilot.HORIZONS:
            for i in range(110):
                target = dt.date(2024, 1, 1) + dt.timedelta(days=i)
                actual = 100.0 if constant else 100 + pi * 3 + i * .03 + math.sin(i / 11)
                direct = actual * (1.02 if constant else 1.01 + .001 * (i % 4))
                report["pairedRows"].append({"product": product, "origin": (target-dt.timedelta(days=h)).isoformat(),
                    "target": target.isoformat(), "horizon": h, "direct": direct, "actual": -987654321})
        for i in range(85):
            origin = pilot.EVAL_START + dt.timedelta(days=i)
            for h in pilot.HORIZONS:
                target = origin + dt.timedelta(days=h)
                actual = 100.0 if constant else 100 + pi * 3 + (target-pilot.TRAIN_END).days * .03 + math.sin((target-pilot.TRAIN_END).days / 11)
                direct = actual * (1.02 if constant else 1.01 + .001 * (i % 4))
                report["pairedRows"].append({"product": product, "origin": origin.isoformat(),
                    "target": target.isoformat(), "horizon": h, "direct": direct, "actual": -1e250})
    return prices, report


def test_state_features_use_only_strict_pre_origin_valid_observations():
    origin = dt.date(2024, 10, 1)
    series = {origin-dt.timedelta(days=i): (100.0+i, 20.0+i) for i in range(1, 11)}
    values, audit = pilot.state_features(series, origin)
    assert audit["status"] == "available"
    assert audit["latestSourceDate"] == (origin-dt.timedelta(days=1)).isoformat()
    assert values[0] == pytest.approx(math.log(101/103))
    assert values[1] == pytest.approx(math.log(101/110))
    assert values[2] == pytest.approx(math.log((sum([21,22,23])/3)/(sum([24,25,26,27,28,29,30])/7)))
    assert len(values) == 5
    missing, _ = pilot.state_features({origin-dt.timedelta(days=25+i):(100+i, 1) for i in range(10)}, origin)
    assert missing is None


def test_forecast_is_1275_answer_free_and_eval_answers_do_not_change_it():
    prices, report = synthetic_inputs()
    forecast = pilot.build_forecast(prices, report)
    assert len(forecast["forecasts"]) == 1275
    assert all("actual" not in row for row in forecast["forecasts"])
    before = forecast["forecasts"]
    for row in report["pairedRows"]:
        if row["target"] >= pilot.EVAL_START.isoformat():
            row["actual"] = 1e250
    after = pilot.build_forecast(prices, report)["forecasts"]
    assert before == after
    assert all(forecast["trainingAudit"][key]["distinctTrainingTargetDates"] >= 90 for key in forecast["trainingAudit"])


def test_missing_direct_is_preserved_in_fixed_grid_and_score_uses_common_four_way_set():
    prices, report = synthetic_inputs()
    # Remove one evaluation direct forecast; the planned cell remains present.
    missing = next(r for r in report["pairedRows"] if r["origin"] == "2024-08-08" and r["horizon"] == 1)
    report["pairedRows"].remove(missing)
    forecast = pilot.build_forecast(prices, report)
    row = next(r for r in forecast["forecasts"] if r["product"] == missing["product"] and r["origin"] == missing["origin"] and r["horizon"] == 1)
    assert row["direct"] is row["candidate"] is row["meanAblation"] is None
    assert row["directMissingReason"] == "direct_prediction_missing"
    scored = pilot.score(forecast, prices)
    assert scored["fourWayCommonRows"] == 1274
    assert scored["coverage"]["forecastValidRows"]["direct"] == 1274
    assert len(scored["dm"]["comparisons"]) == 90
    assert len(scored["byProductHorizon"]) == 15


def test_neighbor_ties_choose_earlier_target_date_and_uses_same_rows_mean_ablation():
    import numpy as np
    dates = [dt.date(2024, 1, 1)+dt.timedelta(days=i) for i in range(25)]
    rows = [{"target": d, "origin": d-dt.timedelta(days=1)} for d in dates]
    fit = {"audit": {"status": "fitted"}, "train": rows, "meanCorrection": .01,
        "neighborFit": {"means": [0]*5, "scales": [1]*5, "Z": np.zeros((25,5)),
                        "residuals": [i/1000 for i in range(25)], "rows": rows}}
    result = pilot._predict({"direct": 100.0}, [0]*5, fit)
    assert [x["target"] for x in result["neighbors"]] == [d.isoformat() for d in dates[:20]]
    assert result["meanAblation"] == pytest.approx(100*math.exp(.01))
    assert result["candidate"] == pytest.approx(100*math.exp(sum(i/1000 for i in range(20))/20))


def test_neighbor_fallback_does_not_suppress_mean_ablation():
    import numpy as np
    date = dt.date(2024, 1, 1)
    rows = [{"target": date+dt.timedelta(days=i), "origin": date+dt.timedelta(days=i-1)} for i in range(10)]
    fit = {"audit": {"status": "fitted"}, "train": rows, "meanCorrection": .02,
        "neighborFit": {"means": [0]*5, "scales": [1]*5, "Z": np.zeros((10,5)),
                        "residuals": [.02]*10, "rows": rows}}
    result = pilot._predict({"direct": 100.0}, [0]*5, fit)
    assert result["candidate"] == 100.0
    assert result["candidateFallbackReason"] == "fewer_than_20_neighbors"
    assert result["meanAblation"] == pytest.approx(100*math.exp(.02))
    assert result["meanFallbackReason"] is None


def test_missing_evaluation_feature_only_falls_back_neighbor_candidate():
    fit = {"audit": {"status": "fitted"}, "train": [], "meanCorrection": .03, "neighborFit": None}
    result = pilot._predict({"direct": 100.0}, None, fit)
    assert result["candidate"] == 100.0
    assert result["candidateFallbackReason"] == "feature_unavailable"
    assert result["meanAblation"] == pytest.approx(100*math.exp(.03))
    assert result["meanFallbackReason"] is None


def test_same_length_duplicate_grid_is_rejected_and_frozen_expansion_gates_are_applied():
    prices, report = synthetic_inputs(constant=True)
    forecast = pilot.build_forecast(prices, report)
    dup = copy.deepcopy(forecast)
    dup["forecasts"][1] = copy.deepcopy(dup["forecasts"][0])
    with pytest.raises(pilot.PilotError, match="duplicate"):
        pilot.score(dup, prices)
    # Synthetic perfect candidate; score-only mutation demonstrates the frozen gates.
    for row in forecast["forecasts"]:
        row["candidate"] = 100.0
        row["meanAblation"] = row["direct"]
    scored = pilot.score(forecast, prices)
    assert scored["gates"]["all15CellsCommonPairsAtLeast30"]
    assert scored["gates"]["candidateCoverageAtLeastDirect"]
    assert scored["gates"]["candidateFiveProductMacroMapeAtMost10Pct"]
    assert scored["gates"]["candidateMapeImprovesVsMeanByAtLeast0_5PctPoints"]
    assert scored["gates"]["candidateMapeImprovesVsDirectByAtLeast0_5PctPoints"]
    assert scored["gates"]["eachProductMapeWorseningAtMost1PctPoint"]
    assert scored["gates"]["eachProductEqualThreeHorizonMaeWorseningAtMost5Pct"]
    assert scored["gates"]["expand"]


def test_significant_adverse_evidence_survives_other_unavailable_dm_tests():
    prices, report = synthetic_inputs(constant=True)
    forecast = pilot.build_forecast(prices, report)
    p, h = pilot.PRODUCTS[0], 1
    index = 0
    for row in forecast["forecasts"]:
        if row["product"] == p and row["horizon"] == h:
            row["candidate"] = 110.0
            row["meanAblation"] = 110.0
            row["direct"] = 100.0 if index % 2 == 0 else 102.0
            index += 1
    scored = pilot.score(forecast, prices)
    target = next(r for r in scored["conclusions"] if r["product"] == p and r["horizon"] == h)
    mean_tests = [r for r in scored["dm"]["comparisons"] if r["product"] == p and r["horizon"] == h and r["comparator"] == "meanAblation"]
    direct_tests = [r for r in scored["dm"]["comparisons"] if r["product"] == p and r["horizon"] == h and r["comparator"] == "direct"]
    assert any(r["dm"]["9"]["status"] != "tested" for r in mean_tests)
    assert any(r["dm"]["9"].get("holmSignificant") and r["dm"]["9"]["meanLossDifference"] > 0 for r in direct_tests)
    assert target["conclusion"] == "worsening_or_mixed"
    assert target["inferenceIncomplete"]
    assert target["significantAdverseHac9Tests"] >= 1
