import datetime as dt
import json
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path

import pytest
import numpy as np

import agri_data.garlic_prospective as gp
from agri_data import modeling


def _selection_report():
    return {"results": [
        {"productId": "garlic", "horizon": h,
         "selection": {"selectedModel": "ridge", "weight": 1.0 if h in (3, 4) else 0.75}}
        for h in range(1, 11)
    ]}


def _raw(end="2026-10-06"):
    return {"meta": {"contextEnd": end}, "products": [{"id": "garlic", "observations": [
        {"date": "2026-10-03", "actual": 100.0, "volume": 10.0},
    ]}]}


def _stub_fit(monkeypatch, n=100, source_day=dt.date(2026, 10, 3)):
    targets = [dt.date(2026, 6, 1) + dt.timedelta(days=i) for i in range(n)]
    samples = [(day, day - dt.timedelta(days=1), 1, (float(i),), .01) for i, day in enumerate(targets)]
    monkeypatch.setattr(modeling, "_supervised_samples", lambda series, cutoff, cache: (samples, set(targets)))
    monkeypatch.setattr(modeling, "_feature_row", lambda series, origin, target, horizon, cache: (source_day, (1.0,)))


def test_build_validates_pinned_weights_uses_strict_cutoff_and_records_forecasts(monkeypatch):
    raw = _raw()
    series = {dt.date.fromisoformat(row["date"]): (row["actual"], row["volume"]) for row in raw["products"][0]["observations"]}
    monkeypatch.setattr(gp.dh, "_observations", lambda _raw: {"garlic": series})
    _stub_fit(monkeypatch)
    monkeypatch.setattr("threadpoolctl.threadpool_limits", lambda limits: nullcontext())
    issued = dt.datetime(2026, 10, 7, 10, 0, tzinfo=gp.KST)
    result = gp.build(raw, _selection_report(), gp.ISSUE_DATE, issued, gp.PRICES_SHA256,
                      gp.STATIC_SHA256, {"selectedRequestHashes": {"1": "abc"}, "selectedRequestCount": 1})
    assert result["status"] == "available"
    assert result["fit"]["trainingTargetDays"] == 100
    assert result["fit"]["trainingTargetEnd"] < result["fit"]["issueCutoffExclusive"]
    assert len(result["forecasts"]) == 10
    assert all(row["sourceDate"] == "2026-10-03" and row["sourceAgeDays"] == 4 for row in result["forecasts"])
    assert result["forecasts"][0]["weight"] == .75
    assert result["forecasts"][2]["weight"] == 1.0
    assert result["forecasts"][0]["persistencePrediction"] == 100.0
    assert result["forecasts"][0]["blendPrediction"] == pytest.approx(
        result["forecasts"][0]["persistencePrediction"] + .75 * (result["forecasts"][0]["ridgePrediction"] - result["forecasts"][0]["persistencePrediction"]))
    json.dumps(result, allow_nan=False)


def test_fit_survives_threadpoolctl_blas_configuration_attribute_error(monkeypatch):
    raw = _raw()
    series = {dt.date.fromisoformat(row["date"]): (row["actual"], row["volume"])
              for row in raw["products"][0]["observations"]}
    monkeypatch.setattr(gp.dh, "_observations", lambda _raw: {"garlic": series})
    _stub_fit(monkeypatch)

    class FakePipeline:
        named_steps = {
            "standardscaler": SimpleNamespace(mean_=np.array([0.0]), scale_=np.array([1.0])),
            "ridge": SimpleNamespace(coef_=np.array([0.0]), intercept_=0.0, alpha=10.0),
        }
        def fit(self, _x, _y): return self
        def predict(self, _x): return [0.0]

    monkeypatch.setattr(modeling, "_new_estimator", lambda _kind: FakePipeline())
    def broken_threadpool_limits(*, limits):
        raise AttributeError("'NoneType' object has no attribute 'split'")
    monkeypatch.setattr("threadpoolctl.threadpool_limits", broken_threadpool_limits)
    result = gp.build(raw, _selection_report(), gp.ISSUE_DATE,
                      dt.datetime(2026, 10, 7, 10, tzinfo=gp.KST), gp.PRICES_SHA256,
                      gp.STATIC_SHA256, {})
    assert result["fit"]["status"] == "fitted"
    assert result["fit"]["reason"] is None
    assert result["forecastCount"] == 10


def test_build_holds_when_less_than_90_target_days_or_stale_source(monkeypatch):
    raw = _raw()
    series = {dt.date(2026, 9, 1): (100.0, 1.0)}
    monkeypatch.setattr(gp.dh, "_observations", lambda _raw: {"garlic": series})
    _stub_fit(monkeypatch, n=89, source_day=dt.date(2026, 9, 1))
    monkeypatch.setattr("threadpoolctl.threadpool_limits", lambda limits: nullcontext())
    result = gp.build(raw, _selection_report(), gp.ISSUE_DATE, dt.datetime(2026, 10, 7, 10, tzinfo=gp.KST),
                      gp.PRICES_SHA256, gp.STATIC_SHA256, {})
    assert result["fit"]["status"] == "held"
    assert result["fit"]["reason"] == "fewer_than_90_distinct_training_target_days"
    assert result["fit"]["sourceHoldReason"] == "latest_source_older_than_14_days"
    assert all(row["status"] == "held" and row["ridgePrediction"] is None for row in result["forecasts"])

    _stub_fit(monkeypatch, n=100, source_day=dt.date(2026, 9, 1))
    monkeypatch.setattr(modeling, "_feature_row", lambda *args: None)
    stale = gp.build(raw, _selection_report(), gp.ISSUE_DATE, dt.datetime(2026, 10, 7, 10, tzinfo=gp.KST),
                     gp.PRICES_SHA256, gp.STATIC_SHA256, {})
    assert stale["fit"]["sourceHoldReason"] == "latest_source_older_than_14_days"
    assert all(row["status"] == "held" for row in stale["forecasts"])


def test_invalid_report_and_issue_cutoff_fail_closed(monkeypatch):
    report = _selection_report(); report["results"][2]["selection"]["weight"] = .75
    with pytest.raises(ValueError, match="selection mismatch"):
        gp._validate_weights(report, gp.STATIC_SHA256)
    with pytest.raises(ValueError, match="contextEnd"):
        gp.build(_raw("2026-10-07"), _selection_report(), gp.ISSUE_DATE,
                 dt.datetime(2026, 10, 7, 10, tzinfo=gp.KST), gp.PRICES_SHA256, gp.STATIC_SHA256, {})


def test_follow_on_build_accepts_latest_snapshot_and_records_its_path(monkeypatch):
    issue=dt.date(2026,10,8); raw=_raw("2026-10-07")
    raw["products"][0]["observations"].append({"date":"2026-10-07","actual":102.0,"volume":12.0})
    series={dt.date.fromisoformat(r["date"]):(r["actual"],r["volume"]) for r in raw["products"][0]["observations"]}
    monkeypatch.setattr(gp.dh,"_observations",lambda _raw:{"garlic":series})
    _stub_fit(monkeypatch,source_day=issue-dt.timedelta(days=1))
    monkeypatch.setattr("threadpoolctl.threadpool_limits",lambda limits:nullcontext())
    snapshot=Path("/snapshots/2026-10-07/prices.json"); digest="a"*64
    result=gp.build(raw,_selection_report(),issue,dt.datetime(2026,10,8,9,tzinfo=gp.KST),digest,gp.STATIC_SHA256,{},prices_path=snapshot)
    assert result["issueDateKst"]=="2026-10-08"
    assert result["input"]["snapshotCutoff"]=="2026-10-07"
    assert result["input"]["pricesSha256"]==digest
    assert result["input"]["pricesPath"]==str(snapshot)
    assert all(row["sourceDate"]=="2026-10-07" and row["sourceAgeDays"]==1 for row in result["forecasts"])


def test_follow_on_is_current_date_only_and_checks_all_products_cutoff(monkeypatch):
    with pytest.raises(ValueError,match="retrospective"):
        gp.build(_raw(),_selection_report(),dt.date(2026,10,6),dt.datetime(2026,10,6,9,tzinfo=gp.KST),gp.PRICES_SHA256,gp.STATIC_SHA256,{})
    raw=_raw(); core={"garlic":{dt.date(2026,10,3):(100.0,10.0)},"cabbage":{dt.date(2026,10,7):(200.0,20.0)}}
    monkeypatch.setattr(gp.dh,"_observations",lambda _raw:core)
    with pytest.raises(ValueError,match="all|observations extend"):
        gp.build(raw,_selection_report(),gp.ISSUE_DATE,dt.datetime(2026,10,7,9,tzinfo=gp.KST),gp.PRICES_SHA256,gp.STATIC_SHA256,{})


def test_follow_on_output_path_is_per_issue_date():
    assert gp._daily_output_path(dt.date(2026,10,8)).name=="2026-10-08.json"


def test_run_requires_follow_on_snapshot_and_rejects_wrong_date_output_before_io(tmp_path):
    oct8=dt.datetime(2026,10,8,9,tzinfo=gp.KST)
    with pytest.raises(ValueError,match="snapshot must be supplied"):
        gp.run(now=oct8)
    with pytest.raises(ValueError,match="actual KST issue date"):
        gp.run(prices_path=tmp_path/"prices.json",output_path=gp.OUTPUT,now=oct8)


@pytest.mark.parametrize("bad_delta", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_model_delta_is_held_before_inverse_transform(monkeypatch, bad_delta):
    raw = _raw()
    series = {dt.date.fromisoformat(row["date"]): (row["actual"], row["volume"]) for row in raw["products"][0]["observations"]}
    monkeypatch.setattr(gp.dh, "_observations", lambda _raw: {"garlic": series})
    _stub_fit(monkeypatch)
    monkeypatch.setattr("threadpoolctl.threadpool_limits", lambda limits: nullcontext())

    class FakePipeline:
        named_steps = {
            "standardscaler": SimpleNamespace(mean_=np.array([0.0]), scale_=np.array([1.0])),
            "ridge": SimpleNamespace(coef_=np.array([0.0]), intercept_=0.0, alpha=10.0),
        }
        def fit(self, _x, _y): return self
        def predict(self, _x): return [bad_delta]

    monkeypatch.setattr(modeling, "_new_estimator", lambda _kind: FakePipeline())
    issued = dt.datetime(2026, 10, 7, 10, tzinfo=gp.KST)
    result = gp.build(raw, _selection_report(), gp.ISSUE_DATE, issued, gp.PRICES_SHA256,
                      gp.STATIC_SHA256, {})
    assert result["fit"]["status"] == "fitted"
    assert all(row["status"] == "held" for row in result["forecasts"])
    assert all(row["ridgePrediction"] is None and row["blendPrediction"] is None for row in result["forecasts"])
    assert all(row["holdReason"] == "prediction_failed:non_finite_log_ratio_prediction" for row in result["forecasts"])


def test_immutable_output_refuses_replacement(tmp_path):
    output = tmp_path / "issue.json"
    gp._immutable(output, b'{"first":1}')
    with pytest.raises(FileExistsError):
        gp._immutable(output, b'{"second":2}')
    assert output.read_bytes() == b'{"first":1}'
