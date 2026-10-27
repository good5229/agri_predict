import datetime as dt
import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from agri_data import dashboard, modeling, onion_prospective as op

SELECTED = {"selectedScopeCount": 4, "selectedScopesSha256": "c" * 64,
            "selectedRequestHashes": {"r1": "d" * 64}}


def fixture(issue=dt.date(2026, 10, 7), n=150):
    start = issue - dt.timedelta(days=n)
    observations = []
    markets = {m: {} for m in op.MARKETS}
    for i in range(n):
        day = start + dt.timedelta(days=i)
        price = 100 + i * 0.3 + (i % 7)
        observations.append({"date": day.isoformat(), "actual": price, "volumeKg": 1000 + i})
        markets[op.MARKETS[0]][day] = {"price": price * 1.02, "weight": 60 + i % 5}
        markets[op.MARKETS[1]][day] = {"price": price * 0.98, "weight": 40 + i % 3}
    raw = {"schemaVersion": 1, "products": [
        {"id": pid, "observations": observations if pid == "onion" else []}
        for pid in dashboard.PRODUCTS
    ]}
    return raw, {"onion": markets}


def test_fit_uses_only_pre_issue_labels_and_features():
    issue = dt.date(2026, 10, 7)
    raw, _ = fixture(issue)
    series = {dt.date.fromisoformat(r["date"]): (r["actual"], r["volumeKg"])
              for r in raw["products"][2]["observations"]}
    result, state = op._fit_predict(series, issue)
    for row in result.values():
        assert row["trainingTargetEnd"] < issue.isoformat()
        assert row["sourceDate"] < issue.isoformat()
        assert row["trainingTargetDays"] >= op.MIN_TRAIN_DAYS
    assert all(state["byHorizon"][str(h)]["fitOrigin"] == issue.isoformat() for h in op.HORIZONS)
    # Forecast features use the issue date as origin while the input prefix ends yesterday.
    assert all(result[h]["sourceAgeDays"] == 1 for h in op.HORIZONS)


def test_saved_month_state_matches_sklearn_pipeline_prediction():
    issue = dt.date(2026, 10, 7)
    raw, _ = fixture(issue)
    series = {dt.date.fromisoformat(r["date"]): (r["actual"], r["volumeKg"])
              for r in raw["products"][2]["observations"]}
    forecasts, _ = op._fit_predict(series, issue)
    h = 1
    cache = {}
    samples = []
    for target, (actual, _) in sorted(series.items()):
        origin = target - dt.timedelta(days=h)
        feature = modeling._feature_row(series, origin, target, h, cache)
        if feature:
            samples.append((target, feature[1], np.log(actual / series[feature[0]][0])))
    train = [r for r in samples if r[0] < issue]
    model = modeling._new_estimator("ridge")
    model.fit(np.asarray([r[1] for r in train]), np.asarray([r[2] for r in train]))
    predict_feature = modeling._feature_row(series, issue, issue + dt.timedelta(days=h), h, cache)
    delta = float(model.predict(np.asarray([predict_feature[1]]))[0])
    expected, _ = modeling._inverse_log_ratio(series[predict_feature[0]][0], delta)
    assert forecasts[h]["directRidgePrediction"] == pytest.approx(expected, rel=1e-11)


def test_share_and_no_share_fallback():
    issue = dt.date(2026, 10, 7)
    raw, markets = fixture(issue)
    out = op.build(raw, markets, issue, dt.datetime(2026, 10, 7, 8, tzinfo=op.KST), "a" * 64, SELECTED)
    forecasts = out["products"][0]["forecasts"]
    assert all(r["method"] == "market_ridge_28d_median_weight_share" for r in forecasts)
    assert all(0 < r["share"] < 1 for r in forecasts)
    for row in markets["onion"].values():
        row.clear()
    out = op.build(raw, markets, issue, dt.datetime(2026, 10, 7, 8, tzinfo=op.KST), "a" * 64, SELECTED)
    assert all(r["method"] == "direct_ridge_fallback" for r in out["products"][0]["forecasts"])
    assert all("28_day_window" in r["fallbackReason"] for r in out["products"][0]["forecasts"])


def test_immutable_output_refuses_overwrite(tmp_path):
    path = tmp_path / "issue.json"
    op._immutable(path, b'{"first":true}\n')
    with pytest.raises(FileExistsError):
        op._immutable(path, b'{"second":true}\n')
    assert json.loads(path.read_text()) == {"first": True}


def test_nonfinite_raw_delta_is_never_issued(monkeypatch):
    issue = dt.date(2026, 10, 7)
    raw, markets = fixture(issue)
    class Step:
        pass
    class Estimator:
        def fit(self, x, y):
            self.named_steps = {"standardscaler": Step(), "ridge": Step()}
            self.named_steps["standardscaler"].mean_ = np.zeros(x.shape[1])
            self.named_steps["standardscaler"].scale_ = np.ones(x.shape[1])
            self.named_steps["ridge"].coef_ = np.full(x.shape[1], 1e308)
            self.named_steps["ridge"].intercept_ = 0.0
    monkeypatch.setattr(modeling, "_new_estimator", lambda _: Estimator())
    result, _ = op._fit_predict({dt.date.fromisoformat(r["date"]): (r["actual"], r["volumeKg"])
                                 for r in raw["products"][2]["observations"]}, issue)
    assert all(r["status"] == "unavailable" and "non_finite_raw_log_delta" in r["reason"]
               for r in result.values())


def test_month_state_reuse_and_month_guard():
    issue = dt.date(2026, 10, 7)
    raw, _ = fixture(issue)
    series = {dt.date.fromisoformat(r["date"]): (r["actual"], r["volumeKg"])
              for r in raw["products"][2]["observations"]}
    _, state = op._fit_predict(series, issue)
    next_issue = issue + dt.timedelta(days=1)
    later_predictions, reused = op._fit_predict(series, next_issue, state)
    assert reused["byHorizon"] == state["byHorizon"]
    assert all(r["trainingTargetEnd"] == state["byHorizon"][str(h)]["trainingTargetEnd"]
               for h, r in later_predictions.items())
    with pytest.raises(ValueError, match="month mismatch"):
        op._fit_predict(series, dt.date(2026, 11, 1), state)


def test_state_file_hash_and_month_are_checked(tmp_path):
    issue = dt.date(2026, 10, 7)
    raw, markets = fixture(issue)
    payload = op.build(raw, markets, issue, dt.datetime(2026, 10, 7, 8, tzinfo=op.KST),
                       "a" * 64, SELECTED)
    path = tmp_path / "first-issue.json"
    path.write_bytes(op._json_bytes(payload))
    states, reference = op._load_month_state(path, issue + dt.timedelta(days=1))
    assert states == payload["modelState"]
    assert reference["sha256"] == op._sha(path.read_bytes())
    with pytest.raises(ValueError, match="month mismatch"):
        op._load_month_state(path, dt.date(2026, 11, 1))
    with pytest.raises(ValueError, match="month mismatch"):
        op._load_month_state(path, issue)
    with pytest.raises(ValueError, match="month mismatch"):
        op._load_month_state(path, dt.date(2026, 10, 6))
    payload["modelStateSha256"] = "0" * 64
    path.write_bytes(op._json_bytes(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        op._load_month_state(path, issue + dt.timedelta(days=1))


def test_state_loader_rejects_partial_and_invalid_scaler(tmp_path):
    issue = dt.date(2026, 10, 7)
    raw, markets = fixture(issue)
    payload = op.build(raw, markets, issue, dt.datetime(2026, 10, 7, 8, tzinfo=op.KST),
                       "a" * 64, SELECTED)
    path = tmp_path / "first-issue.json"
    payload["modelState"].pop("110008")
    payload["modelStateSha256"] = op._sha(op._json_bytes(payload["modelState"]))
    path.write_bytes(op._json_bytes(payload))
    with pytest.raises(ValueError, match="incomplete"):
        op._load_month_state(path, issue + dt.timedelta(days=1))
    payload = op.build(raw, markets, issue, dt.datetime(2026, 10, 7, 8, tzinfo=op.KST),
                       "a" * 64, SELECTED)
    payload["modelState"]["direct"]["byHorizon"]["1"]["scale"][0] = -1.0
    payload["modelStateSha256"] = op._sha(op._json_bytes(payload["modelState"]))
    path.write_bytes(op._json_bytes(payload))
    with pytest.raises(ValueError, match="parameters are invalid"):
        op._load_month_state(path, issue + dt.timedelta(days=1))


def test_unavailable_saved_model_keeps_first_issue_training_audit():
    issue = dt.date(2026, 10, 7)
    raw, markets = fixture(issue)
    first = op.build(raw, markets, issue, dt.datetime(2026, 10, 7, 8, tzinfo=op.KST),
                     "a" * 64, SELECTED)
    state = first["modelState"]
    frozen = state["direct"]["byHorizon"]["1"]
    frozen.update(available=False, reason="frozen_unavailable", trainingTargetDays=91,
                  trainingSampleCount=97, trainingTargetEnd="2026-07-01")
    for key in ("mean", "scale", "coef", "intercept"):
        frozen.pop(key, None)
    later = op.build(raw, markets, issue + dt.timedelta(days=1),
                     dt.datetime(2026, 10, 8, 8, tzinfo=op.KST), "b" * 64,
                     SELECTED, model_states=state, state_reference={"sha256": "e" * 64})
    row = later["products"][0]["forecasts"][0]
    assert row["trainingTargetDays"] == 91
    assert row["trainingSampleCount"] == 97
    assert row["trainingTargetEnd"] == "2026-07-01"
    assert "frozen_unavailable" in row["fallbackReason"]


def test_real_run_rejects_historical_issue_date(tmp_path):
    issue = dt.date(2026, 10, 6)
    with pytest.raises(ValueError, match="current KST date"):
        op.run(tmp_path / "no-prices.json", tmp_path / "no-db.sqlite", tmp_path / "out.json",
               issue_date=issue, now=dt.datetime(2026, 10, 7, 8, tzinfo=op.KST))


def test_default_horizon_parameter_keeps_first_artifact_and_fixture_output_unchanged():
    issue = dt.date(2026, 10, 7)
    raw, markets = fixture(issue)
    stamp = dt.datetime(2026, 10, 7, 8, tzinfo=op.KST)
    default = op.build(raw, markets, issue, stamp, "a" * 64, SELECTED)
    explicit = op.build(raw, markets, issue, stamp, "a" * 64, SELECTED, horizons=(1, 2))
    assert op._json_bytes(default) == op._json_bytes(explicit)
    first = Path(op.ROOT) / "outputs/prospective-onion/2026-10/first-2026-10-07.json"
    assert hashlib.sha256(first.read_bytes()).hexdigest() == "bf9be149f83062966a5c223eb9c2e914c5a9a0a8291bd52cb984eecefb4ee8c8"
