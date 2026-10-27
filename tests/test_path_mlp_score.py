from __future__ import annotations

import datetime as dt

import pytest

from agri_data import path_mlp_score as scorer


def _forecast_row(crop, day, horizon, *, missing_arm=None):
    row = {"crop": crop, "target_date": day.isoformat(), "origin_date": (day - dt.timedelta(days=horizon)).isoformat(),
           "horizon": horizon, "eligible": True, "fixed50KeyPresent": True}
    for arm, prediction in (("orderedBlend", 110.0), ("summaryBlend", 111.0), ("scrambledBlend", 112.0), ("fixed50", 120.0)):
        unavailable = arm == missing_arm
        row[arm] = {"status": "not_issued" if unavailable else "issued",
                    "reason": "synthetic_missing" if unavailable else None,
                    "prediction": None if unavailable else prediction}
    return row


def test_dm_uses_calendar_spacing_for_hac_covariance():
    start = dt.date(2025, 1, 1)
    losses = [(-1.0 if index % 2 else 1.0) + index / 100 for index in range(50)]
    consecutive = {start + dt.timedelta(days=index): value for index, value in enumerate(losses)}
    every_other_day = {start + dt.timedelta(days=2 * index): value for index, value in enumerate(losses)}

    dense = scorer._dm(consecutive, 9)
    sparse = scorer._dm(every_other_day, 9)

    assert dense["status"] == sparse["status"] == "tested"
    assert dense["statistic"] != sparse["statistic"]


def test_holm_correction_is_monotone_and_keeps_family_size():
    rows = [{"pValue": value} for value in (.01, .02, .6)]

    scorer._holm(rows)

    assert [row["holmPValue"] for row in rows] == pytest.approx([.03, .04, .6])
    assert [row["holmSignificant"] for row in rows] == [True, True, False]


def test_any_arm_losing_a_fixed50_key_holds_all_crop_candidates():
    day = dt.date(2025, 1, 1)
    rows = []
    for crop in scorer.PRODUCTS:
        for horizon in scorer.HORIZONS:
            rows.append(_forecast_row(crop, day, horizon,
                                      missing_arm="orderedBlend" if crop == "cabbage" and horizon == 1 else None))
    macro = []
    for crop in scorer.PRODUCTS:
        for comparison in ("ordered_vs_fixed50", "summary_vs_fixed50", "scrambled_vs_fixed50"):
            macro.append({"crop": crop, "comparison": comparison, "horizonsIncluded": 10,
                          "aEqualHorizonMape": 1.0, "bEqualHorizonMape": 2.0,
                          "aEqualHorizonMae": 1.0, "bEqualHorizonMae": 2.0,
                          "horizons": [{"horizon": horizon, "a": {"n": 30, "mape": 1.0, "mae": 1.0},
                                        "b": {"n": 30, "mape": 2.0, "mae": 2.0}}
                                       for horizon in scorer.HORIZONS]})

    gates = scorer._screen_gates(rows, {}, macro, [])
    cabbage = [gate for gate in gates if gate["crop"] == "cabbage"]

    assert len(cabbage) == 3
    assert all(gate["anyArmLostFixed50IssuedKey"] for gate in cabbage)
    assert all(gate["decision"] == "retain_fixed50" for gate in cabbage)


def test_failed_2026_gate_does_not_rewrite_screen_selected_portfolio():
    eligible_day = dt.date(2026, 1, 20)
    preperiod_day = dt.date(2026, 1, 2)
    eligible = _forecast_row("cabbage", eligible_day, 1)
    eligible["orderedBlend"]["prediction"] = 500.0  # Deliberately worse than fixed50.
    preperiod = _forecast_row("cabbage", preperiod_day, 5)
    preperiod["eligible"] = False
    preperiod["orderedBlend"] = {"status": "not_issued", "reason": "origin_before_phase_start", "prediction": None}
    actuals = {("cabbage", eligible_day): 100.0, ("cabbage", preperiod_day): 100.0}

    policy = scorer._whole_policy_rows([eligible, preperiod], actuals,
        {"cabbage": "orderedBlend"}, {"cabbage": "retain_fixed50"})

    assert policy[0]["policy"] == 500.0
    assert policy[0]["evaluationGateDecision"] == "retain_fixed50"
    assert policy[0]["fallback"] is False
    assert policy[1]["policy"] == 120.0
    assert policy[1]["fallback"] is True
    assert policy[1]["preperiodOriginFallback"] is True


def test_forecast_replay_runs_before_target_join(monkeypatch):
    events = []

    def validate(*args, **kwargs):
        events.append("replay")
        raise ValueError("synthetic replay mismatch")

    def actuals(*args, **kwargs):
        events.append("target_join")
        raise AssertionError("target join must not run after replay failure")

    monkeypatch.setattr(scorer.forecasts, "validate_manifest_bytes", validate)
    monkeypatch.setattr(scorer, "_actuals", actuals)
    manifest = {"phaseName": "screen", "forecasts": []}

    with pytest.raises(ValueError, match="synthetic replay mismatch"):
        scorer.score_screen(manifest, {}, {})

    assert events == ["replay"]


def test_synthetic_screen_scores_full_frozen_family_without_fitting(monkeypatch):
    monkeypatch.setattr(scorer.forecasts, "validate_manifest_bytes", lambda *args, **kwargs: None)
    rows = []
    for crop in scorer.PRODUCTS:
        for offset in range((scorer.SCREEN_END - scorer.SCREEN_START).days + 1):
            day = scorer.SCREEN_START + dt.timedelta(days=offset)
            for horizon in scorer.HORIZONS:
                rows.append(_forecast_row(crop, day, horizon))
    manifest = {"phaseName": "screen", "forecasts": rows}
    prices = {"schemaVersion": 1, "products": [
        {"id": crop, "observations": [
            {"date": (scorer.SCREEN_START + dt.timedelta(days=offset)).isoformat(),
             "actual": 100.0 + offset / 10}
            for offset in range((scorer.SCREEN_END - scorer.SCREEN_START).days + 1)]}
        for crop in scorer.PRODUCTS]}

    report = scorer.score_screen(manifest, prices, {})

    assert len(rows) == 17_050
    assert len(report["metrics"]) == 5 * 5 * 10
    assert len(report["dm"]) == 5 * 5 * 10 * 2 * 2
    assert all(item["familySize"] == 500 for item in report["dmSummaryByBandwidth"])
    assert report["coverage"]["cabbage/1"]["planned"] == 341
    manifest_bytes = scorer._canonical(manifest)
    price_bytes = scorer._canonical(prices)
    component_bytes = scorer._canonical({})
    report_bytes = scorer._canonical(report)
    decision = scorer._decision_for_validated_screen(manifest, report,
        screen_manifest_bytes=manifest_bytes, screen_report_bytes=report_bytes)
    unsigned = dict(decision)
    saved = unsigned.pop("decisionSha256")
    assert saved == scorer._sha_bytes(scorer._canonical(unsigned))
    binding = scorer.validate_screen_artifacts(manifest_bytes, price_bytes, component_bytes,
                                               report_bytes, scorer._canonical(decision))
    assert binding["selection"] == report["selection"]


def test_evaluation_keeps_fixed_family_and_falls_back_on_missing_selected_arm(monkeypatch):
    monkeypatch.setattr(scorer.forecasts, "validate_manifest_bytes", lambda *args, **kwargs: None)
    monkeypatch.setattr(scorer, "validate_screen_artifacts", lambda *args, **kwargs: {
        "selection": {crop: "orderedBlend" for crop in scorer.PRODUCTS}})
    rows = []
    for crop in scorer.PRODUCTS:
        for offset in range((scorer.EVAL_END - scorer.EVAL_START).days + 1):
            day = scorer.EVAL_START + dt.timedelta(days=offset)
            for horizon in scorer.HORIZONS:
                row = _forecast_row(crop, day, horizon, missing_arm="orderedBlend")
                row["eligible"] = day - dt.timedelta(days=horizon) >= scorer.EVAL_START
                rows.append(row)
    manifest = {"phaseName": "evaluation", "forecasts": rows}
    prices = {"schemaVersion": 1, "products": [
        {"id": crop, "observations": [
            {"date": (scorer.EVAL_START + dt.timedelta(days=offset)).isoformat(),
             "actual": 100.0 + offset / 10}
            for offset in range((scorer.EVAL_END - scorer.EVAL_START).days + 1)]}
        for crop in scorer.PRODUCTS]}
    screen_manifest = {"phaseName": "screen", "forecasts": []}
    screen_manifest_bytes = scorer._canonical(screen_manifest)
    gates = [{"crop": crop, "candidate": "orderedBlend", "decision": "eligible_for_selection",
              "candidateEqualHorizonMape": 1.0, "candidateEqualHorizonMae": 1.0}
             for crop in scorer.PRODUCTS]
    screen_report = {"phase": "screen_scored", "manifestSha256": scorer._sha_bytes(screen_manifest_bytes),
                     "candidateGates": gates,
                     "selection": {crop: "orderedBlend" for crop in scorer.PRODUCTS}}
    screen_report_bytes = scorer._canonical(screen_report)
    decision = scorer._decision_for_validated_screen(screen_manifest, screen_report,
        screen_manifest_bytes=screen_manifest_bytes, screen_report_bytes=screen_report_bytes)

    report = scorer.score_evaluation(manifest, prices, screen_manifest, screen_report, decision, {},
        screen_manifest_bytes=screen_manifest_bytes, screen_report_bytes=screen_report_bytes,
        screen_components_bytes=b"{}")

    assert len(report["dm"]) == 500 * 2
    assert all(row["pValue"] == 1.0 for row in report["dm"])
    assert all(item["familySize"] == 500 for item in report["dmSummaryByBandwidth"])
    cabbage = [row for row in report["evaluationGates"] if row["crop"] == "cabbage"][0]
    assert cabbage["decision"] == "retain_fixed50"
    assert report["policyMetrics"][0]["fallbackCount"] > 0
    assert report["policyMetrics"][0]["policy"]["mae"] == report["policyMetrics"][0]["fixed50"]["mae"]
