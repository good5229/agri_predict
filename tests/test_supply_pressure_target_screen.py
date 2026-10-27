import datetime as dt
import hashlib
import pytest

from agri_data import dashboard, supply_pressure_target_screen as screen


def _inputs(tmp_path, monkeypatch, observations):
    products = {pid: {"id": pid, "observations": observations(pid)} for pid in dashboard.PRODUCTS}
    status = {"snapshotSha256": "a" * 64}
    provenance = tmp_path / "provenance.json"
    provenance.write_bytes(b"provenance")
    monkeypatch.setattr(screen, "validate_inputs", lambda *_: (
        products, status, provenance, b"prices", b"status", b"provenance"))
    return tmp_path / "prices.json", tmp_path / "status.json"


def test_target_aligned_counts_gate_and_provenance(tmp_path, monkeypatch):
    observations = []
    day = dt.date(2017, 1, 1)
    while day <= dt.date(2024, 12, 31):
        observations.append({"date": day.isoformat(), "actual": 100, "volumeKg": 10})
        day += dt.timedelta(days=1)
    prices, status = _inputs(tmp_path, monkeypatch, lambda _: observations)
    payload = screen.build(prices, status, tmp_path / "out.json")
    assert payload["plannedRows"] == 420
    assert len(payload["rows"]) == 420
    assert payload["origins"] == [d.isoformat() for d in screen.ORIGINS]
    assert {r["targetStatus"] for r in payload["rows"]} == {"present"}
    assert {r["featureStatus"] for r in payload["rows"]} == {"available"}
    assert payload["gate"]["passed"]
    assert payload["noModelFit"] and not payload["mapeComputed"]
    assert payload["input"]["pricesSha256"] == hashlib.sha256(b"prices").hexdigest()
    assert payload["input"]["provenanceSha256"] == hashlib.sha256(b"provenance").hexdigest()
    assert payload["products"][0]["byHorizon"][0]["pairedEvaluable"] == 28
    with pytest.raises(FileExistsError):
        screen.build(prices, status, tmp_path / "out.json")


def test_missing_target_is_not_feature_failure_and_gates_are_distinct(tmp_path, monkeypatch):
    observations = []
    day = dt.date(2017, 1, 1)
    while day <= dt.date(2024, 12, 31):
        observations.append({"date": day.isoformat(), "actual": 100, "volumeKg": 10})
        day += dt.timedelta(days=1)
    # Remove some target dates without labeling them as market closures.
    removed = {"2024-08-16", "2024-10-02"}
    observations = [row for row in observations if row["date"] not in removed]
    prices, status = _inputs(tmp_path, monkeypatch,
                             lambda pid: observations if pid == dashboard.PRODUCTS[0] else [])
    # This isolates counting behavior: available features can coexist with missing targets.
    monkeypatch.setattr(screen.prospective, "_feature", lambda *_: ((1.0,), dt.date(2024, 1, 1)))
    payload = screen.build(prices, status, tmp_path / "out.json")
    first = payload["products"][0]
    assert first["planned"] == 84
    assert first["targetPresent"] < first["planned"]
    assert all(r["targetStatus"] == "missing_observation" for r in payload["rows"]
               if r["productId"] == dashboard.PRODUCTS[0] and r["target"] in removed)
    assert all(r["featureStatus"] == "available" for r in payload["rows"])
    assert not payload["gate"]["allProductHorizonsHaveMinimumPaired"]
    assert not payload["gate"]["allProductsMeetConditionalCoverage"]


def test_coverage_gate_applies_to_each_product_horizon(tmp_path, monkeypatch):
    observations = []
    day = dt.date(2017, 1, 1)
    while day <= dt.date(2024, 12, 31):
        observations.append({"date": day.isoformat(), "actual": 100, "volumeKg": 10})
        day += dt.timedelta(days=1)
    prices, status = _inputs(tmp_path, monkeypatch, lambda _: observations)
    unavailable_origins = {dt.date(2024, 8, d) for d in range(15, 19)} | {
        dt.date(2024, 10, d) for d in range(1, 5)
    }
    monkeypatch.setattr(
        screen.prospective, "_feature",
        lambda _series, origin, _target, horizon: (
            None if horizon == 1 and origin in unavailable_origins else ((1.0,), origin - dt.timedelta(days=1))
        ),
    )
    payload = screen.build(prices, status, tmp_path / "out.json")
    product = payload["products"][0]
    h1, h5, h10 = product["byHorizon"]
    assert (h1["targetPresent"], h1["pairedEvaluable"], h1["planned"]) == (28, 20, 28)
    assert (h5["pairedEvaluable"], h5["planned"]) == (28, 28)
    assert (h10["pairedEvaluable"], h10["planned"]) == (28, 28)
    assert product["pairedEvaluable"] / product["planned"] > 0.90
    assert h1["conditionalFeatureCoverage"] < 0.90
    assert not h1["conditionalCoveragePassed"]
    assert h5["conditionalCoveragePassed"] and h10["conditionalCoveragePassed"]
    assert not payload["gate"]["allProductsMeetConditionalCoverage"]
    assert not payload["gate"]["passed"]


def test_feature_failure_reason_is_separate_from_target_status(tmp_path, monkeypatch):
    prices, status = _inputs(tmp_path, monkeypatch, lambda _: [])
    monkeypatch.setattr(screen.prospective, "_feature", lambda *_: None)
    monkeypatch.setattr(screen, "_failure_reason", lambda *_: "missing_past_history")
    payload = screen.build(prices, status, tmp_path / "out.json")
    row = payload["rows"][0]
    assert row["targetStatus"] == "missing_observation"
    assert row["featureStatus"] == "unavailable"
    assert row["featureFailureReason"] == "missing_past_history"
    assert row["pairedEvaluable"] is False
    assert payload["products"][0]["conditionalFeatureCoverage"] is None
