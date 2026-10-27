from __future__ import annotations

import datetime as dt

import pytest

from agri_data import chronos2_market_context_score as score


def _manifest(*, phase="V", u_status="issued", h_status="issued", h_value=90.0):
    target = dt.date(2026, 4, 15) if phase == "V" else dt.date(2026, 7, 20)
    rows = []
    actuals = {}
    for product in score.PRODUCTS:
        for horizon in score.HORIZONS:
            origin = target - dt.timedelta(days=horizon)
            rows.append({
                "key": {"product": product, "originDate": origin.isoformat(),
                        "targetDate": target.isoformat(), "horizon": horizon},
                "eligible": True,
                "conditions": {
                    "B": {"status": "issued", "prediction": 100.0},
                    "U": {"status": u_status, "prediction": 101.0 if u_status == "issued" else None,
                          "reason": None if u_status == "issued" else "input_unavailable"},
                    "M": {"status": "issued", "prediction": 80.0},
                    "H": {"status": h_status, "prediction": h_value if h_status == "issued" else None,
                          "reason": None if h_status == "issued" else "model_output_invalid"},
                },
            })
            actuals[(product, target)] = 90.0
    return {"forecasts": rows}, actuals


def test_fixed_hb_family_and_gate_ignore_u_failures(monkeypatch):
    monkeypatch.setattr(score, "BOOTSTRAP_REPS", 5)
    doc, actuals = _manifest(u_status="unavailable")
    out = score.score_manifest(doc, actuals=actuals, phase="V")

    assert len(out["fixedDMFamily"]["tests"]) == 160
    assert all(r["comparison"] == "H_vs_B" for r in out["fixedDMFamily"]["tests"])
    assert all(r["status"] == "insufficient_pairs" for r in out["fixedDMFamily"]["tests"])
    assert all(r["pValue"] == 1 for r in out["fixedDMFamily"]["tests"])
    assert len(out["cells"]) == 40
    assert out["fortyCellEqualWeight"]["mapeDifference"] < 0
    assert out["fortyCellEqualWeight"]["maeDifference"] == -10.0
    assert out["validationGate"]["passed"]
    assert all(row["HIssued"] and not row["UIssued"] for row in out["primaryCoverage"])
    assert out["descriptiveComparisons"][0]["pairedKeys"] == 0
    assert out["descriptiveComparisons"][1]["pairedKeys"] == 0
    assert out["descriptiveComparisons"][2]["pairedKeys"] == 40
    assert all(b["comparisons"]["H_vs_B"]["ae"]["requested"] == 5 for b in out["bootstrap"])


def test_h_must_be_exact_frozen_blend_and_cannot_fallback():
    doc, actuals = _manifest(h_value=89.0)
    with pytest.raises(ValueError, match="0.5 B \\+ 0.5 M"):
        score.score_manifest(doc, actuals=actuals, phase="V")

    doc, actuals = _manifest()
    doc["forecasts"][0]["conditions"]["M"] = {"status": "unavailable", "prediction": None}
    with pytest.raises(ValueError, match="H is issued despite M being unavailable"):
        score.score_manifest(doc, actuals=actuals, phase="V")


def test_missing_h_reduces_coverage_and_fails_gate_not_the_fixed_family(monkeypatch):
    monkeypatch.setattr(score, "BOOTSTRAP_REPS", 5)
    doc, actuals = _manifest(h_status="unavailable")
    out = score.score_manifest(doc, actuals=actuals, phase="V")
    assert len(out["fixedDMFamily"]["tests"]) == 160
    assert all(r["status"] == "insufficient_pairs" for r in out["fixedDMFamily"]["tests"])
    assert all(c["issuedCoverageLoss"] == 1 and c["evaluationCoverageLoss"] == 1 for c in out["cells"])
    assert not out["validationGate"]["passed"]
    assert all(b["comparisons"]["H_vs_B"]["ae"]["status"] == "unavailable_missing_fixed_cell"
               for b in out["bootstrap"])


def test_actual_coverage_loss_is_separate_and_t_phase_uses_july_origin_cutoff(monkeypatch):
    monkeypatch.setattr(score, "BOOTSTRAP_REPS", 2)
    doc, actuals = _manifest(phase="T")
    first = doc["forecasts"][0]
    # H can issue without a positive actual, but the evaluation sample coverage is lost.
    actuals.pop((first["key"]["product"], dt.date.fromisoformat(first["key"]["targetDate"])))
    out = score.score_manifest(doc, actuals=actuals, phase="T")
    assert out["originCutoff"] == "2026-07-01"
    cell = next(c for c in out["cells"] if c["product"] == first["key"]["product"]
                and c["horizon"] == first["key"]["horizon"])
    assert cell["issuedCoverageLoss"] == 0
    assert cell["bPositiveActualKeys"] == cell["hPositiveActualKeys"] == 0
    assert cell["evaluationCoverageLoss"] == 0
    assert not out["validationGate"]["passed"]


def test_phase_domain_and_bad_key_are_rejected():
    doc, actuals = _manifest()
    with pytest.raises(ValueError, match="phase must be V or T"):
        score.score_manifest(doc, actuals=actuals, phase="X")
    doc["forecasts"][0]["key"]["targetDate"] = "2026-04-14"
    with pytest.raises(ValueError, match="origin/target/horizon"):
        score.score_manifest(doc, actuals=actuals, phase="V")


def test_hac_cell_uses_unique_calendar_targets_and_family_stays_80(monkeypatch):
    monkeypatch.setattr(score, "BOOTSTRAP_REPS", 3)
    rows = []
    actuals = {}
    product, horizon = "radish", 4
    start = dt.date(2026, 4, 6)
    for i in range(30):
        target = start + dt.timedelta(days=i)
        origin = target - dt.timedelta(days=horizon)
        rows.append({"key": {"product": product, "originDate": origin.isoformat(),
                             "targetDate": target.isoformat(), "horizon": horizon},
                     "conditions": {"B": {"status": "issued", "prediction": 100.0},
                                    "U": {"status": "unavailable"},
                                    "M": {"status": "issued", "prediction": 90.0},
                                    "H": {"status": "issued", "prediction": 95.0}}})
        actuals[(product, target)] = 100.0 + (i % 5)
    out = score.score_manifest({"forecasts": rows}, actuals=actuals, phase="V")
    family = out["fixedDMFamily"]["tests"]
    target_tests = [r for r in family if (r["product"], r["horizon"]) == (product, horizon)]
    assert len(family) == 160
    assert len(target_tests) == 4
    assert all(r["n"] == 30 for r in target_tests)
    assert {r["loss"]: r["status"] for r in target_tests if r["bandwidth"] == 9} == {
        "ae": "zero_spread", "ape": "tested"}
    assert sum(r["status"] in ("tested", "zero_spread") for r in family) == 4
    assert sum(r["status"] == "insufficient_pairs" for r in family) == 156


def test_primary_month_and_product_horizon_summaries_are_equal_cell_not_pooled(monkeypatch):
    monkeypatch.setattr(score, "BOOTSTRAP_REPS", 2)
    rows = []
    actuals = {}
    special = ("cabbage", 1)
    for product in score.PRODUCTS:
        for horizon in score.HORIZONS:
            is_special = (product, horizon) == special
            targets = ([dt.date(2026, 4, 1) + dt.timedelta(days=horizon+i) for i in range(29)]
                       if is_special else [dt.date(2026, 4, 1) + dt.timedelta(days=horizon+10)])
            targets.append(dt.date(2026, 5, 10) + dt.timedelta(days=horizon))
            for target in targets:
                origin = target - dt.timedelta(days=horizon)
                b = 100.0 if is_special else 101.0
                h = 110.0 if is_special else 100.0
                m = 2*h-b
                rows.append({"key": {"product": product, "originDate": origin.isoformat(),
                                     "targetDate": target.isoformat(), "horizon": horizon},
                             "conditions": {"B": {"status": "issued", "prediction": b},
                                            "U": {"status": "unavailable"},
                                            "M": {"status": "issued", "prediction": m},
                                            "H": {"status": "issued", "prediction": h}}})
                actuals[(product, target)] = 100.0
    out = score.score_manifest({"forecasts": rows}, actuals=actuals, phase="V")
    april = next(r for r in out["primaryEqualCellStrata"]
                 if r["scope"] == "target_month_40_equal_cell" and r["targetMonth"] == "2026-04")
    may = next(r for r in out["primaryEqualCellStrata"]
               if r["scope"] == "target_month_40_equal_cell" and r["targetMonth"] == "2026-05")
    cell_april = next(r for r in out["primaryEqualCellStrata"]
                      if r["scope"] == "product_horizon_target_month_cell"
                      and r["product"] == "cabbage" and r["horizon"] == 1
                      and r["targetMonth"] == "2026-04")
    assert april["completeSupport"] and may["completeSupport"]
    assert april["mapeDifference"] < 0 and may["mapeDifference"] < 0
    assert cell_april["mapeDifference"] > 0
    # A pooled April score reverses sign because one worse cell has many more dates.
    april_rows = [r for r in rows if r["key"]["targetDate"].startswith("2026-04")]
    pooled = sum(abs(actuals[(r["key"]["product"], dt.date.fromisoformat(r["key"]["targetDate"]))]
                     - r["conditions"]["H"]["prediction"])
                 - abs(actuals[(r["key"]["product"], dt.date.fromisoformat(r["key"]["targetDate"]))]
                       - r["conditions"]["B"]["prediction"])
                 for r in april_rows) / len(april_rows)
    assert pooled > 0 > april["maeDifference"]
