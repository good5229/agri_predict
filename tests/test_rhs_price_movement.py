from __future__ import annotations

import math

import pytest

from agri_data.rhs_price_movement import diagnose_price_movement


def _fixture():
    rows = []
    actuals = {}
    expected = set()
    for horizon in range(1, 11):
        counts = 3 if horizon == 1 else 2
        for index in range(counts):
            if index == 1:
                target = f"2026-05-{10 + horizon:02d}"
            else:
                target = f"2026-04-{10 + horizon + (10 if index == 2 else 0):02d}"
            origin = f"2026-03-{horizon:02d}"
            key = ("radish", origin, target, horizon)
            expected.add(key)
            y = float(100 + 5 * horizon + index)
            actuals[("radish", target)] = y
            baseline_direct = y - 2.0
            sarima = y + 4.0
            direct = {
                "B": baseline_direct,
                "D": y - 1.0,
                "C": y + 1.0 if horizon == 1 else y - 1.0,
                "P": baseline_direct,
            }
            rows.append({
                "key": {"product": key[0], "originDate": key[1], "targetDate": key[2], "horizon": key[3]},
                "eligible": True,
                "conditions": {
                    condition: {
                        "status": "issued",
                        "directPrediction": prediction,
                        "sarimaPrediction": sarima,
                        "blendPrediction": (prediction + sarima) / 2.0,
                    }
                    for condition, prediction in direct.items()
                },
            })
    # An issued forecast outside the mature actual sample contributes only to
    # the separately labeled all-issued relative-change RMS.
    extra = ("radish", "2026-06-01", "2026-06-11", 1)
    rows.append({
        "key": {"product": extra[0], "originDate": extra[1], "targetDate": extra[2], "horizon": extra[3]},
        "eligible": True,
        "conditions": {
            "B": {"status": "issued", "directPrediction": 100.0, "sarimaPrediction": 110.0, "blendPrediction": 105.0},
            "D": {"status": "issued", "directPrediction": 100.0, "sarimaPrediction": 110.0, "blendPrediction": 105.0},
            "C": {"status": "issued", "directPrediction": 150.0, "sarimaPrediction": 110.0, "blendPrediction": 130.0},
            "P": {"status": "issued", "directPrediction": 100.0, "sarimaPrediction": 110.0, "blendPrediction": 105.0},
        },
    })
    return rows, actuals, expected


def _aggregate(result, comparison, layer, scope="interval", **scope_keys):
    return next(
        row for row in result["aggregates"]
        if row["comparison"] == comparison and row["layer"] == layer and row["scope"] == scope
        and all(row.get(key) == value for key, value in scope_keys.items())
    )


def test_movement_efficiency_gain_harm_and_blend_identity():
    rows, actuals, expected = _fixture()
    result = diagnose_price_movement(rows, actuals, phase="synthetic", expected_comparison_keys=expected)
    direct = _aggregate(result, "C-B", "direct")
    blend = _aggregate(result, "C-B", "blend")

    assert direct["n"] == 21
    assert direct["movementMae"] == pytest.approx(1.2)
    assert direct["baselineMae"] - direct["conditionMae"] == pytest.approx(direct["improvementMae"])
    assert direct["baselineMapePct"] - direct["conditionMapePct"] == pytest.approx(direct["improvementMapePp"])
    assert direct["improvementMae"] == pytest.approx(1.0)
    assert direct["efficiencyMae"] == pytest.approx(1.0 / 1.2)
    assert direct["lossReducingMae"] + direct["lossIncreasingMae"] == pytest.approx(direct["movementMae"])
    assert direct["lossReducingMae"] - direct["lossIncreasingMae"] == pytest.approx(direct["improvementMae"])
    assert blend["movementMae"] == pytest.approx(0.5 * direct["movementMae"])
    assert blend["improvementMae"] == pytest.approx(-0.6)
    assert blend["efficiencyMae"] == pytest.approx(-1.0)
    assert blend["lossReducingMae"] == pytest.approx(0.0)
    assert blend["lossIncreasingMae"] == pytest.approx(blend["movementMae"])

    crossed = next(row for row in result["keyRows"] if row["comparison"] == "C-B" and row["layer"] == "direct"
                   and row["horizon"] == 1)
    assert crossed["crossesActual"] is True
    assert crossed["direction"] == 1
    assert crossed["weightedMovementMae"] == pytest.approx(3.0 / (10 * 3))


def test_fixed_horizon_weights_and_month_date_contributions_reconstruct_interval():
    rows, actuals, expected = _fixture()
    result = diagnose_price_movement(rows, actuals, phase="synthetic", expected_comparison_keys=expected)
    interval = _aggregate(result, "C-B", "direct")
    horizon_rows = [r for r in result["aggregates"] if r["comparison"] == "C-B" and r["layer"] == "direct"
                    and r["scope"] == "horizon"]
    assert sum(r["movementMae"] for r in horizon_rows) / 10 == pytest.approx(interval["movementMae"])
    pooled = sum(row["movementMae"] for row in result["keyRows"] if row["comparison"] == "C-B"
                 and row["layer"] == "direct") / len(expected)
    assert pooled != pytest.approx(interval["movementMae"])

    months = [r for r in result["aggregates"] if r["comparison"] == "C-B" and r["layer"] == "direct"
              and r["scope"] == "target_month"]
    assert {r["targetMonth"] for r in months} == {"2026-04", "2026-05"}
    assert all(r["hasAllTenHorizons"] for r in months)
    assert sum(r["fixedGlobalContribution"]["improvementMae"] for r in months) == pytest.approx(
        interval["improvementMae"]
    )
    dates = [r for r in result["aggregates"] if r["comparison"] == "C-B" and r["layer"] == "direct"
             and r["scope"] == "target_date"]
    assert sum(r["fixedGlobalContribution"]["improvementMae"] for r in dates) == pytest.approx(
        interval["improvementMae"]
    )


def test_zero_movement_efficiency_is_null_and_all_issued_rms_is_separate():
    rows, actuals, expected = _fixture()
    result = diagnose_price_movement(rows, actuals, phase="synthetic", expected_comparison_keys=expected)
    p_direct = _aggregate(result, "P-B", "direct")
    assert p_direct["movementMae"] == 0
    assert p_direct["efficiencyMae"] is None
    assert p_direct["relativePriceChangeRmsFixed10hMacro"] == 0
    assert p_direct["allIssuedN"] == 22
    assert p_direct["allIssuedKeyPooledRelativePriceChangeRms"] == 0

    c_direct = _aggregate(result, "C-B", "direct")
    assert c_direct["allIssuedN"] == 22
    assert c_direct["allIssuedKeyPooledRelativePriceChangeRms"] > c_direct["relativePriceChangeRmsFixed10hMacro"]
    assert c_direct["allIssuedFixed10hMacroRelativePriceChangeRms"] > c_direct["relativePriceChangeRmsFixed10hMacro"]
    concentration = c_direct["dateContributionConcentration"]["improvementMae"]
    assert concentration["positiveDateCount"] > 0
    assert concentration["negativeDateCount"] == 0
    assert 0 < concentration["topFiveAbsoluteNetDateShare"] <= 1


def test_date_concentration_keeps_gain_and_harm_and_uses_absolute_net_denominator():
    rows, actuals, expected = _fixture()
    changed = []
    for row in rows:
        new_row = {**row, "conditions": {name: dict(state) for name, state in row["conditions"].items()}}
        if row["key"]["horizon"] == 2:
            y = actuals[("radish", row["key"]["targetDate"])]
            new_direct = y - 4.0
            new_row["conditions"]["C"]["directPrediction"] = new_direct
            new_row["conditions"]["C"]["blendPrediction"] = (
                new_direct + new_row["conditions"]["C"]["sarimaPrediction"]
            ) / 2.0
        changed.append(new_row)
    result = diagnose_price_movement(changed, actuals, phase="synthetic", expected_comparison_keys=expected)
    interval = _aggregate(result, "C-B", "direct")
    date_rows = [row for row in result["aggregates"] if row["comparison"] == "C-B"
                 and row["layer"] == "direct" and row["scope"] == "target_date"]
    contributions = [row["fixedGlobalContribution"]["improvementMae"] for row in date_rows]
    concentration = interval["dateContributionConcentration"]["improvementMae"]
    assert concentration["positiveDateCount"] > 0
    assert concentration["negativeDateCount"] > 0
    assert concentration["grossPositiveContribution"] > 0
    assert concentration["grossNegativeContributionAbs"] > 0
    expected_top_share = sum(sorted((abs(x) for x in contributions), reverse=True)[:5]) / sum(abs(x) for x in contributions)
    assert concentration["topFiveAbsoluteNetDateShare"] == pytest.approx(expected_top_share)
    assert sum(contributions) == pytest.approx(interval["improvementMae"])


def test_rejects_support_shrink_invalid_predictions_sarima_mismatch_and_bad_blend():
    rows, actuals, expected = _fixture()
    with pytest.raises(ValueError, match="support differs"):
        diagnose_price_movement(rows, actuals, phase="synthetic", expected_comparison_keys=set(list(expected)[1:]))

    bad = [dict(row) for row in rows]
    bad[0] = {**rows[0], "conditions": {**rows[0]["conditions"], "C": {**rows[0]["conditions"]["C"],
                 "directPrediction": -1.0}}}
    with pytest.raises(ValueError, match="positive"):
        diagnose_price_movement(bad, actuals, phase="synthetic", expected_comparison_keys=expected)

    bad = [dict(row) for row in rows]
    bad[0] = {**rows[0], "conditions": {**rows[0]["conditions"], "C": {**rows[0]["conditions"]["C"],
                 "sarimaPrediction": 1000.0}}}
    with pytest.raises(ValueError, match="50:50"):
        diagnose_price_movement(bad, actuals, phase="synthetic", expected_comparison_keys=expected)

    bad = [dict(row) for row in rows]
    bad[0] = {**rows[0], "conditions": {**rows[0]["conditions"], "P": {**rows[0]["conditions"]["P"],
                 "sarimaPrediction": rows[0]["conditions"]["P"]["sarimaPrediction"] + 1}}}
    with pytest.raises(ValueError, match="50:50"):
        diagnose_price_movement(bad, actuals, phase="synthetic", expected_comparison_keys=expected)

    bad_actuals = dict(actuals)
    bad_actuals["radish", "2026-04-11"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        diagnose_price_movement(rows, bad_actuals, phase="synthetic", expected_comparison_keys=expected)


def test_production_support_counts_are_pinned_and_horizon_must_be_integer():
    rows, actuals, expected = _fixture()
    with pytest.raises(ValueError, match="651 keys"):
        diagnose_price_movement(rows, actuals, phase="V", expected_comparison_keys=expected)

    bad = [dict(row) for row in rows]
    bad[0] = {**rows[0], "key": {**rows[0]["key"], "horizon": True}}
    with pytest.raises(ValueError, match="integer"):
        diagnose_price_movement(bad, actuals, phase="synthetic", expected_comparison_keys=expected)


def test_rejects_nonfinite_values():
    rows, actuals, expected = _fixture()
    bad = [dict(row) for row in rows]
    bad[0] = {**rows[0], "conditions": {**rows[0]["conditions"], "C": {**rows[0]["conditions"]["C"],
                 "directPrediction": math.inf}}}
    with pytest.raises(ValueError, match="finite"):
        diagnose_price_movement(bad, actuals, phase="synthetic", expected_comparison_keys=expected)
