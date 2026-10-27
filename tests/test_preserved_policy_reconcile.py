from __future__ import annotations

import datetime as dt
import hashlib
import json

from agri_data import preserved_policy_reconcile as reconcile


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _fixture(monkeypatch, baseline_rows, evaluation_rows, actual_days):
    decision = {"schemaVersion": 1, "kind": "preserved_feature_confirmation_gate", "selectedCrops": ["garlic"]}
    decision["decisionSha256"] = hashlib.sha256(_canonical(decision)).hexdigest()
    prices = {"schemaVersion": 1, "products": [
        {"id": crop, "observations": [{"date": day.isoformat(), "actual": 120.0} for day in actual_days]}
        for crop in reconcile.PRODUCTS]}
    price_hash = hashlib.sha256(_canonical(prices)).hexdigest()
    evaluation = {"phase": "forecast_only", "phaseName": "evaluation", "targetValuesAttached": False,
                  "selectedCrops": ["garlic"], "manifestSha256": "synthetic-evaluation-sha",
                  "inputSha256": {"confirmationDecisionFile": "decision-file-sha", "prices": price_hash},
                  "protocol": {"evaluationSelection": ["garlic"]}, "forecasts": evaluation_rows}
    report = {"phase": "scored", "scoredPhase": "evaluation", "selectedCrops": ["garlic"],
              "confirmationDecisionSha256": decision["decisionSha256"],
              "forecastManifestSha256": evaluation["manifestSha256"], "priceInputSha256": price_hash,
              "candidateGates": [{"product": "garlic", "decision": "retain_fixed50_policy"}]}
    report["reportSha256"] = hashlib.sha256(_canonical(report)).hexdigest()
    baseline = {"phase": "forecast_only", "targetValuesAttached": False, "forecasts": baseline_rows}
    docs = {"fixed50Manifest": baseline, "evaluationManifest": evaluation,
            "confirmationDecision": decision, "evaluationReport": report, "prices": prices}
    raw = {name: _canonical(doc) for name, doc in docs.items()}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    evaluation["inputSha256"]["confirmationDecisionFile"] = hashes["confirmationDecision"]
    raw = {name: _canonical(doc) for name, doc in docs.items()}
    hashes = {name: hashlib.sha256(value).hexdigest() for name, value in raw.items()}
    monkeypatch.setattr(reconcile, "EXPECTED_SHA256", hashes)
    return reconcile.reconcile_bytes(raw)


def _baseline_row(crop, day, horizon, fixed=100.0):
    return {"product": crop, "target": day.isoformat(), "origin": (day - dt.timedelta(days=horizon)).isoformat(),
            "horizon": horizon, "predictions": {"fixed50": fixed}}


def _eval_row(crop, day, horizon, *, issued=False, prediction=None, eligible=None):
    origin = day - dt.timedelta(days=horizon)
    if eligible is None:
        eligible = origin >= reconcile.EVAL_START
    record = {"status": "issued" if issued else "not_issued", "prediction": prediction,
              "reason": None if issued else "model_not_yet_fitted_at_origin"}
    return {"product": crop, "target": day.isoformat(), "origin": origin.isoformat(), "horizon": horizon,
            "eligible": eligible, "selectedCrop": crop == "garlic", "models": {"preserved106Blend": record}}


def test_reconciliation_includes_114_preperiod_keys_and_falls_back(monkeypatch):
    all_preperiod_rows = []
    start = reconcile.EVAL_START
    for crop in reconcile.PRODUCTS:
        for offset in range(10):
            day = start + dt.timedelta(days=offset)
            for horizon in reconcile.HORIZONS:
                if day - dt.timedelta(days=horizon) < start:
                    all_preperiod_rows.append(_baseline_row(crop, day, horizon))
    garlic = [row for row in all_preperiod_rows if row["product"] == "garlic"]
    others = [row for row in all_preperiod_rows if row["product"] != "garlic"]
    baseline_rows = garlic[:23] + others[:91]
    assert len(baseline_rows) == 114
    target_days = sorted({_day for _row in baseline_rows for _day in [dt.date.fromisoformat(_row["target"])]})

    result = _fixture(monkeypatch, baseline_rows, [], target_days)

    assert result["coverage"]["originalFixed50IssuedKeys"] == 114
    assert result["coverage"]["eligibleOriginKeys"] == 0
    assert result["coverage"]["preperiodOriginKeys"] == 114
    garlic_keys = sum(row["product"] == "garlic" for row in baseline_rows)
    assert garlic_keys == 23
    assert result["coverage"]["preperiodOriginFallbacks"] == garlic_keys
    assert all(row["fallback"] == "fixed50" for row in result["candidateFailureTargets"])


def test_failed_final_gate_does_not_reselect_or_erase_candidate_loss(monkeypatch):
    target = dt.date(2026, 1, 3)
    baseline_rows = [_baseline_row("garlic", target, 1)]
    evaluation_rows = [_eval_row("garlic", target, 1, issued=True, prediction=200.0)]

    result = _fixture(monkeypatch, baseline_rows, evaluation_rows, [target])

    assert result["selection"]["selectedCrops"] == ["garlic"]
    assert result["selection"]["evaluationGate"][0]["decision"] == "retain_fixed50_policy"
    cell = next(row for row in result["cellMetrics"] if row["product"] == "garlic" and row["horizon"] == 1)
    assert cell["policyMae"] == 80.0
    assert cell["fixed50Mae"] == 20.0
    assert cell["candidateFailures"] == 0
