"""Offline regressions for the frozen countertrend guard."""
from __future__ import annotations

import datetime as dt
import math

import pytest

from agri_data import countertrend_guard as guard


def test_guard_applies_fixed_log_geometric_midpoint_only_on_countertrend():
    result, applied, reason = guard.apply_guard("cabbage", 4, 100.0, 64.0, 0.1, 0.2)
    assert applied and reason is None
    assert result == pytest.approx(80.0)


def test_guard_uses_log_difference_when_ratio_underflows():
    result, applied, reason = guard.apply_guard("radish", 4, 1e308, 1e-308, 0.1, 0.2)
    assert applied and reason is None
    assert result == pytest.approx(1.0)


@pytest.mark.parametrize(
    "product,horizon,p0,e,m7,m28,reason",
    [
        ("onion", 4, 100, 64, 0.1, 0.2, "product_out_of_scope"),
        ("cabbage", 3, 100, 64, 0.1, 0.2, "horizon_out_of_scope"),
        ("radish", 10, 100, 64, 0.1, 0.2, None),
        ("cabbage", 4, 100, 64, 0.1, -0.2, "momentum_not_strictly_aligned"),
        ("cabbage", 4, 100, 64, 0.0, 0.2, "momentum_not_strictly_aligned"),
        ("cabbage", 4, 100, 110, 0.1, 0.2, "forecast_not_countertrend"),
        ("cabbage", 4, None, 64, 0.1, 0.2, "momentum_unavailable"),
        ("cabbage", 4, 100, 64, float("nan"), 0.2, "momentum_unavailable"),
    ],
)
def test_guard_scope_and_fallbacks(product, horizon, p0, e, m7, m28, reason):
    value, applied, actual_reason = guard.apply_guard(product, horizon, p0, e, m7, m28)
    if reason is None:
        assert applied and actual_reason is None
        assert value == pytest.approx(80)
    elif reason == "product_out_of_scope" or reason == "horizon_out_of_scope" or reason == "forecast_not_countertrend" or reason == "momentum_not_strictly_aligned":
        assert value == e and not applied and actual_reason == reason
    else:
        assert value == e and not applied and actual_reason == reason


def test_missing_baseline_is_not_fabricated():
    assert guard.apply_guard("cabbage", 4, 100, None, 0.1, 0.2) == (None, False, "baseline_not_issued")
    assert guard.apply_guard("cabbage", 4, 100, float("inf"), 0.1, 0.2) == (None, False, "baseline_not_issued")


def _score_rows():
    rows = []
    actuals = {}
    for pi, product in enumerate(guard.PRODUCTS):
        for day in range(1, 6):
            target = dt.date(2025, 3, day)
            actual = 100.0 + pi * 20 + day
            actuals[(product, target)] = actual
            for horizon in guard.HORIZONS:
                e = actual * 1.2
                # Candidate improves only on selected cabbage rows.
                c = actual * (1.0 if product == "cabbage" else 1.2)
                rows.append({"product": product, "target": target.isoformat(), "horizon": horizon,
                             "origin": (target-dt.timedelta(days=horizon)).isoformat(),
                             "baselinePrediction": e, "model": {"prediction": c}, "guardApplied": c != e})
    return rows, actuals


def test_score_has_fixed_four_by_ten_families_holm_and_unselected_p_one(monkeypatch):
    rows, actuals = _score_rows()
    monkeypatch.setattr(guard.prior, "_actuals", lambda *_args: actuals)
    result = guard._score(rows, {}, "confirmation", {"cabbage"})
    assert len(result["metrics"]) == 40
    # The 40-cell reporting grid is retained across both HAC bandwidths.
    assert len(result["dm"]) == len(guard.PRODUCTS) * len(guard.HORIZONS) * 2 * len(guard.BANDWIDTHS)
    assert all(x["pValue"] == 1.0 and x["status"] == "not_tested_by_protocol"
               for x in result["dm"] if x["product"] not in guard.GUARD_PRODUCTS or x["product"] != "cabbage")
    assert set(result["coverage"]) == {f"{p}/{h}" for p in guard.PRODUCTS for h in guard.HORIZONS}
    assert len(result["monthly"]) == 4 * 10 * 1
    assert result["collisionCounts"]["all"] == 50


def test_final_gate_stops_when_no_crop_selected(monkeypatch):
    # The builder must stop before final forecasts if the replayed confirmation
    # decision selects no crop.
    result = {"reportSha256": "same", "inputsSha256": {},
        "decisions": [{"product": p, "decision": "retain_prior_E"} for p in guard.GUARD_PRODUCTS]}
    monkeypatch.setattr(guard, "score_confirmation", lambda *args: result)
    monkeypatch.setattr(guard, "build_confirmation_manifest", lambda *args: {})
    monkeypatch.setattr(guard, "_verify_sources", lambda *args: ({}, {}, {}, {}, {}, {"finalForecasts": []}))
    monkeypatch.setattr(guard, "_period_rows", lambda *_args: pytest.fail("final rows must not be opened"))
    with pytest.raises(ValueError, match="passed for no crop"):
        guard.build_final_manifest(b"", b"", b"", b"", b"", b"", result)


def test_guard_manifest_tampering_is_detected():
    doc = {"phase": "countertrend_confirmation_forecast_only", "confirmationForecasts": []}
    doc["manifestSha256"] = guard.sha(guard.canonical(doc))
    doc["confirmationForecasts"].append({"product": "cabbage"})
    with pytest.raises(ValueError, match="manifestSha256 digest mismatch"):
        guard._validate_guard_manifest(doc, "confirmation")


def test_pinned_source_bytes_reject_forged_confirmation_report():
    # Even a self-consistent fabricated report cannot substitute for the
    # reviewed, content-pinned confirmation result.
    with pytest.raises(ValueError, match="pinned source artifact SHA-256 mismatch"):
        guard._verify_sources(b"baseline", b"prices", b"selection", b"confirmation", b"forged report", b"final")


def test_confirmation_manifest_replays_complete_causal_grid_without_actuals(monkeypatch):
    rows = []
    for product in guard.PRODUCTS:
        for offset in range((guard.CONF_END - guard.CONF_START).days + 1):
            target = guard.CONF_START + dt.timedelta(days=offset)
            for horizon in guard.HORIZONS:
                origin = target - dt.timedelta(days=horizon)
                if origin < guard.CONF_START:
                    continue
                rows.append({"product": product, "target": target.isoformat(), "origin": origin.isoformat(),
                    "horizon": horizon, "eligible": True,
                    "model": {"prediction": 64.0},
                    "features": {"p0": 100.0, "momentum7Log": 0.1, "momentum28Log": 0.2}})
    monkeypatch.setattr(guard, "_verify_sources", lambda *_args: ({}, {}, {}, {"confirmationForecasts": rows}, {}, {}))
    built = guard.build_confirmation_manifest(b"m", b"p", b"s", b"c", b"r", b"f")
    expected_n = sum(1 for row in rows)
    assert len(built["confirmationForecasts"]) == expected_n
    assert built["targetValuesAttached"] is False
    assert all("actual" not in row for row in built["confirmationForecasts"])
    assert all(row["model"]["prediction"] == pytest.approx(80) and row["guardApplied"]
               for row in built["confirmationForecasts"] if row["product"] in guard.GUARD_PRODUCTS and row["horizon"] >= 4)
    assert guard._validate_guard_manifest(built, "confirmation") == built["confirmationForecasts"]


def test_final_manifest_keeps_unselected_crop_at_exact_prior_e(monkeypatch):
    prior_rows = []
    for product in guard.PRODUCTS:
        for offset in range((guard.FINAL_END - guard.FINAL_START).days + 1):
            target = guard.FINAL_START + dt.timedelta(days=offset)
            for horizon in guard.HORIZONS:
                origin = target - dt.timedelta(days=horizon)
                if origin < dt.date(2026, 7, 1):
                    continue
                prior_rows.append({"product": product, "target": target.isoformat(), "origin": origin.isoformat(),
                    "horizon": horizon, "model": {"prediction": 64.0},
                    "features": {"p0": 100.0, "momentum7Log": 0.1, "momentum28Log": 0.2}})
    result = {"reportSha256": "r", "inputsSha256": {}, "decisions": [
        {"product": "cabbage", "decision": "freeze_for_final_period"},
        {"product": "radish", "decision": "retain_prior_E"}]}
    monkeypatch.setattr(guard, "score_confirmation", lambda *_args: result)
    monkeypatch.setattr(guard, "build_confirmation_manifest", lambda *_args: {})
    monkeypatch.setattr(guard, "_verify_sources", lambda *_args: ({}, {}, {}, {}, {}, {"finalForecasts": prior_rows}))
    built = guard.build_final_manifest(b"m", b"p", b"s", b"c", b"r", b"f", result)
    assert built["frozenCrops"] == ["cabbage"]
    for row in built["finalForecasts"]:
        if row["product"] == "radish":
            assert row["model"]["prediction"] == row["baselinePrediction"]
            assert not row["guardApplied"]
            assert row["fallbackReason"] == "crop_not_selected_by_confirmation"


def test_failed_final_retention_does_not_rewrite_frozen_policy_score(monkeypatch):
    final = {"schemaVersion": 1, "phase": "countertrend_final_forecast_only", "targetValuesAttached": False,
             "frozenCrops": ["cabbage"], "finalForecasts": []}
    final["manifestSha256"] = guard.sha(guard.canonical(final))
    confirmation = {"reportSha256": "confirmation"}
    scored_policy = {"metrics": [{"product": "cabbage", "horizon": h, "n": 30,
                                  "eMae": 10.0, "cMae": 12.0, "eMape": 10.0, "cMape": 12.0}
                                 for h in guard.HORIZONS],
                    "macro": [{"product": p, "cells": 10, "eMae": 10.0, "cMae": 12.0,
                               "eMape": 10.0, "cMape": 12.0} for p in guard.PRODUCTS],
                    "coverage": {f"{p}/{h}": {"planned": 30, "eIssued": 30, "cIssued": 30,
                                                "coveragePreserved": True}
                                 for p in guard.PRODUCTS for h in guard.HORIZONS},
                    "dm": [], "monthly": {}, "collisionCounts": {"all": 0, "byProduct": {p: 0 for p in guard.PRODUCTS}}}
    monkeypatch.setattr(guard, "build_final_manifest", lambda *_args: final)
    monkeypatch.setattr(guard, "_verify_sources", lambda *_args: ({}, {}, {}, {}, {}, {}))
    monkeypatch.setattr(guard, "_score", lambda *_args: scored_policy)
    result = guard.score_final(b"m", b"p", b"s", b"c", b"r", b"f", confirmation, final)
    cabbage_macro = next(x for x in result["macro"] if x["product"] == "cabbage")
    retention = next(x for x in result["nextResearchCandidateRetention"] if x["product"] == "cabbage")
    assert cabbage_macro["cMae"] == 12.0 and cabbage_macro["eMae"] == 10.0
    assert cabbage_macro["cMape"] == 12.0 and cabbage_macro["eMape"] == 10.0
    assert retention["retainedCandidate"] == "E"
    assert "do not rewrite" in result["policyForecastInterpretation"]
