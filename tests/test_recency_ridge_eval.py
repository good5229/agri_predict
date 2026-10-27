import datetime as dt
import json
import contextlib
import sys
import types
from unittest.mock import patch

import pytest

from agri_data import recency_ridge_eval as ev


def test_three_year_weights_are_mean_normalized_and_strictly_pre_origin():
    origin = dt.date(2025, 1, 1)
    got = ev.weights(origin, [origin - dt.timedelta(days=d) for d in (1, 1095, 2190)])
    assert sum(got) / len(got) == pytest.approx(1.0)
    assert got[0] > got[1] > got[2]
    assert got[0] / got[1] == pytest.approx(2 ** ((1094) / 1095))
    with pytest.raises(ValueError, match="strictly before"):
        ev.weights(origin, [origin])


def _tiny_reports():
    paired = []
    date = ev.START
    for product in ev.PRODUCTS:
        for horizon in ev.HORIZONS:
            origin = date - dt.timedelta(days=horizon)
            paired.append({"product": product, "target": date.isoformat(),
                           "horizon": horizon, "actual": 100., "direct": 102., "persistence": 104.})
    paired.append({"product": ev.PRODUCTS[0], "target": "2023-01-01", "horizon": 1,
                   "actual": 100., "direct": 102., "persistence": 104.})
    return {"protocol": {"id": "direct-horizon-ridge-2026-10-06"}, "pairedRows": paired}


def test_pair_identity_requires_all_crop_horizon_cells_and_unique_keys(monkeypatch):
    monkeypatch.setattr(ev, "EXPECTED_PAIRS", 50)
    report = _tiny_reports()
    assert len(ev._pairs(report)) == 50
    assert all(row["target"] >= ev.START.isoformat() for row in ev._pairs(report).values())
    report["pairedRows"].append(dict(report["pairedRows"][0]))
    with pytest.raises(ValueError, match="duplicate"):
        ev._pairs(report)


def test_build_preserves_fixed_pair_coverage_and_direct_fallback(monkeypatch):
    monkeypatch.setattr(ev, "EXPECTED_PAIRS", 50)
    prices = {"schemaVersion": 1, "products": [
        {"id": p, "observations": [{"date": ev.START.isoformat(), "actual": 100., "volumeKg": 10.}]}
        for p in ev.PRODUCTS]}
    def no_fit(series, h, fit_origin, rows, cache):
        return {}, {"fitOrigin": fit_origin.isoformat(), "trainingRows": 0,
                    "distinctTrainingTargets": 0, "status": "insufficient_training"}
    with patch.object(ev, "_fit_predict", side_effect=no_fit):
        result = ev.build(prices, _tiny_reports())
    assert result["coverage"]["n"] == 50
    assert result["coverage"]["fallbackCount"] == 50
    assert len(result["cells"]) == 50
    assert len(result["multipleTesting"]["HAC9"]) == 200
    assert all(row["candidate"] == row["directRidge"] and row["fallback"]
               for row in result["pairedRows"])
    assert result["macro"]["fiveProductEqualWeightMape"] == pytest.approx(2.0)
    assert all("sourceAgeDays" in row and "sourceDate" in row for row in result["pairedRows"])
    assert set(result["coverage"]["bySeason"]) == {"3"}
    assert result["macro"]["fiveProductEqualWeight"]["directRidge"]["mape"] == pytest.approx(2.0)


def test_build_rejects_actual_value_that_does_not_match_pinned_prices(monkeypatch):
    monkeypatch.setattr(ev, "EXPECTED_PAIRS", 50)
    prices = {"schemaVersion": 1, "products": [
        {"id": p, "observations": [{"date": ev.START.isoformat(), "actual": 100., "volumeKg": 10.}]}
        for p in ev.PRODUCTS]}
    report = _tiny_reports()
    report["pairedRows"][0]["actual"] = 101.
    with pytest.raises(ValueError, match="does not match"):
        ev.build(prices, report)


def test_manifest_creation_is_immutable(tmp_path):
    path = tmp_path / "manifest.json"
    payload = {"status": "frozen_before_scoring", "protocol": ev.PROTOCOL}
    digest = ev._immutable_json(path, payload)
    assert len(digest) == 64
    assert json.loads(path.read_text())["status"] == "frozen_before_scoring"
    with pytest.raises(FileExistsError):
        ev._immutable_json(path, payload)


def test_fit_passes_mean_normalized_recency_weights_and_rejects_nonfinite_raw_output():
    start = dt.date(2024, 1, 1)
    series = {start + dt.timedelta(days=i): (100. + i / 10, 10.) for i in range(120)}
    captured = {}

    class Estimator:
        def fit(self, x, y, **kwargs):
            captured["weights"] = kwargs["ridge__sample_weight"]
            captured["n"] = len(y)
            return self
        def predict(self, x):
            return [float("nan")]

    eval_target = start + dt.timedelta(days=121)
    eval_row = {"targetDate": eval_target, "originDate": eval_target - dt.timedelta(days=1)}
    fit_origin = eval_row["originDate"]
    def fake_feature(s, o, t, h, c):
        eligible = [day for day in s if day <= o]
        return (max(eligible), (float(t.toordinal()),)) if eligible else None
    threadpool_stub = types.SimpleNamespace(threadpool_limits=lambda limits: contextlib.nullcontext())
    with patch.dict(sys.modules, {"threadpoolctl": threadpool_stub}):
      with patch.object(ev.modeling, "_feature_row", side_effect=fake_feature):
        with patch.object(ev.modeling, "_new_estimator", return_value=Estimator()):
            preds, audit = ev._fit_predict(series, 1, fit_origin, [(("cabbage", eval_target, 1), eval_row)], {})
    assert captured["n"] == 119
    assert sum(captured["weights"]) / len(captured["weights"]) == pytest.approx(1.0)
    assert audit["status"] == "fitted"
    assert audit["invalidPredictionTargets"] == [eval_target.isoformat()]
    assert preds == {}
