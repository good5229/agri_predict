import datetime as dt
import json

import pytest

from agri_data import onion_horizon_extension as ext
from agri_data import onion_prospective as op
from test_onion_prospective import SELECTED, fixture


def test_extension_build_and_model_state_are_exactly_horizons_3_through_10(tmp_path):
    issue = dt.date(2026, 10, 7)
    raw, markets = fixture(issue)
    payload = op.build(raw, markets, issue, dt.datetime(2026, 10, 7, 8, tzinfo=op.KST),
                       "a" * 64, SELECTED, horizons=ext.HORIZONS)
    assert tuple(r["horizon"] for r in payload["products"][0]["forecasts"]) == ext.HORIZONS
    assert set(payload["modelState"]["direct"]["byHorizon"]) == {str(h) for h in ext.HORIZONS}
    state_path = tmp_path / "first-extension.json"
    state_path.write_bytes(op._json_bytes(payload))
    state, reference = op._load_month_state(state_path, issue + dt.timedelta(days=1), horizons=ext.HORIZONS)
    assert set(state["direct"]["byHorizon"]) == {str(h) for h in ext.HORIZONS}
    assert reference["sha256"] == op._sha(state_path.read_bytes())
    with pytest.raises(ValueError, match="horizons are incomplete"):
        op._load_month_state(state_path, issue + dt.timedelta(days=1), horizons=(1, 2))


def test_extension_followup_reuses_only_extension_state(tmp_path):
    issue = dt.date(2026, 10, 7)
    raw, markets = fixture(issue)
    first = op.build(raw, markets, issue, dt.datetime(2026, 10, 7, 8, tzinfo=op.KST),
                     "a" * 64, SELECTED, horizons=ext.HORIZONS)
    first_path = tmp_path / "first-extension.json"
    first_path.write_bytes(op._json_bytes(first))
    second_issue = issue + dt.timedelta(days=1)
    prior, reference = op._load_month_state(first_path, second_issue,
                                             dt.datetime(2026, 10, 8, 8, tzinfo=op.KST), ext.HORIZONS)
    second = op.build(raw, markets, second_issue, dt.datetime(2026, 10, 8, 8, tzinfo=op.KST),
                      "b" * 64, SELECTED, model_states=prior, state_reference=reference,
                      horizons=ext.HORIZONS)
    assert tuple(r["horizon"] for r in second["products"][0]["forecasts"]) == ext.HORIZONS
    assert second["modelState"] is None
    assert second["modelStateReference"]["sha256"] == reference["sha256"]
    assert all(second["modelStateReference"]["sha256"] == op._sha(first_path.read_bytes()) for _ in (0,))
    json.dumps(second, allow_nan=False)


def test_wrapper_delegates_frozen_horizon_tuple_and_has_separate_output(monkeypatch):
    calls = {}
    def fake(*args, **kwargs):
        calls.update(kwargs)
        return {"output": str(args[2]), "availableHorizons": 8}
    monkeypatch.setattr(op, "run", fake)
    result = ext.run(tmp_path_path("prices.json"), tmp_path_path("db.sqlite"),
                     ext.default_output(dt.date(2026, 10, 7)), issue_date=dt.date(2026, 10, 7),
                     first_month_issue=True)
    assert result["availableHorizons"] == 8
    assert calls["horizons"] == tuple(range(3, 11))
    assert "horizons-3-10" in str(ext.default_output(dt.date(2026, 10, 7)))
    assert ext.HORIZONS == (3, 4, 5, 6, 7, 8, 9, 10)


def tmp_path_path(name):
    from pathlib import Path
    return Path("/synthetic") / name


def test_wrapper_cli_requires_exactly_one_state_mode(tmp_path):
    with pytest.raises(SystemExit):
        ext.main(["--prices", str(tmp_path / "prices.json"), "--output", str(tmp_path / "out.json")])
