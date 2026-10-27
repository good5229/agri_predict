import datetime as dt
import json
from pathlib import Path

import pytest

from agri_data import dashboard, supply_pressure_screen as screen


def _inputs(tmp_path, monkeypatch, observations):
    products = {pid: {"id": pid, "observations": observations(pid)} for pid in dashboard.PRODUCTS}
    status = {"snapshotSha256": "f" * 64}
    monkeypatch.setattr(screen, "validate_inputs", lambda *_: ({}, products, status,
        tmp_path / "provenance.json", b"prices", b"status", b"provenance"))
    return tmp_path / "prices.json", tmp_path / "status.json"


def test_planned_population_no_fitting_future_exclusion_and_gate(tmp_path, monkeypatch):
    # Produce enough history for features; a future observation must be excluded.
    base = []
    day = dt.date(2017, 1, 1)
    while day <= dt.date(2024, 12, 31):
        base.append({"date": day.isoformat(), "actual": 100, "volumeKg": 10})
        day += dt.timedelta(days=1)
    def obs(pid):
        values = list(base)
        if pid == "cabbage":
            values = [r for r in values if not (dt.date.fromisoformat(r["date"]).year in (2021, 2022, 2023)
                                                   and dt.date.fromisoformat(r["date"]).month == 1)]
        values.append({"date": "2025-01-01", "actual": 999999, "volumeKg": 999999})
        return values
    prices, status = _inputs(tmp_path, monkeypatch, obs)
    seen_ends = []
    original_feature = screen.prospective._feature
    def record_end(series, *args):
        seen_ends.append(max(series))
        return original_feature(series, *args)
    monkeypatch.setattr(screen.prospective, "_feature", record_end)
    payload = screen.build(prices, status, tmp_path / "screen.json")
    assert payload["plannedRows"] == 840
    assert len(payload["rows"]) == 840
    assert payload["noModelFit"] and not payload["mapeComputed"]
    assert payload["products"][0]["available"] < 168
    assert payload["products"][1]["available"] == 168
    assert all(r["target"][:4] == "2024" for r in payload["rows"])
    assert max(seen_ends) == dt.date(2024, 12, 31)
    assert all(p["gatePassed"] == (p["available"] / 168 >= .9) for p in payload["products"])


def test_failure_reasons_and_immutable_output(tmp_path, monkeypatch):
    empty = lambda _: []
    prices, status = _inputs(tmp_path, monkeypatch, empty)
    output = tmp_path / "screen.json"
    payload = screen.build(prices, status, output)
    assert {r["reason"] for r in payload["rows"]} == {"missing_past_price_or_volume"}
    with pytest.raises(FileExistsError):
        screen.build(prices, status, output)


def test_seasonal_failure_is_distinct():
    # Recent and rolling history exist, while the three prior seasonal years do not.
    series = {}
    start = dt.date(2023, 12, 1)
    while start <= dt.date(2024, 1, 14):
        series[start] = (100.0, 10.0)
        start += dt.timedelta(days=1)
    assert screen._failure_reason(series, dt.date(2024, 1, 15), dt.date(2024, 1, 16), 1) == "seasonal_3_year_failure"


def test_frozen_snapshot_cutoff_is_october_6(tmp_path, monkeypatch):
    prices = tmp_path / "prices.json"
    status = tmp_path / "run-status.json"
    prices.write_text("{}")
    status.write_text(json.dumps({"cutoff": "2026-10-06"}))
    def reached_snapshot_validation(*_):
        raise RuntimeError("passed frozen cutoff validation")
    monkeypatch.setattr(screen.prospective, "validate_snapshot", reached_snapshot_validation)
    with pytest.raises(RuntimeError, match="passed frozen cutoff"):
        screen.validate_inputs(prices, status)
    status.write_text(json.dumps({"cutoff": "2026-10-07"}))
    with pytest.raises(ValueError, match="2026-10-06"):
        screen.validate_inputs(prices, status)
