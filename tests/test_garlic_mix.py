import datetime as dt
import json
from contextlib import nullcontext
from decimal import Decimal
from pathlib import Path

import agri_data.garlic_mix as gm
from agri_data import direct_horizon as dh, modeling
from agri_data.garlic_mix import (CLASSES, _candidate_subset, _decode_payload, _garlic_features,
                                  _assert_market_reconciled, _joint_conclusion, _market_subclass_totals,
                                  _output_may_not_replace)


def test_garlic_features_use_signed_shares_and_asof_relative_price():
    origin = dt.date(2024, 2, 1)
    rows = {code: {} for code in CLASSES}
    # The denominator includes negative adjustments and all selected subclasses.
    rows["06"][origin - dt.timedelta(days=1)] = {
        "amount": Decimal("120"), "weight": Decimal("2"), "rowCount": 2, "negativeRows": 1,
        "price": Decimal("60"),
    }
    rows["16"][origin - dt.timedelta(days=2)] = {
        "amount": Decimal("80"), "weight": Decimal("2"), "rowCount": 1, "negativeRows": 0,
        "price": Decimal("40"),
    }
    features = _garlic_features(rows, origin)
    assert len(features) == len(CLASSES) * 7 + 2
    # First subclass's signed share is 2/(2+2) over either window.
    assert features[0:4] == [0.5, 1.0, 0.5, 1.0]
    # Its relative price is against same-day positive garlic aggregate price 60.
    assert abs(features[4]) < 1e-12
    assert features[5:7] == [1.0, 1.0]
    # Unobserved class carries the explicit missing sentinels.
    assert features[14:21] == [0.0, 1.0, 0.0, 1.0, 0.0, 15.0, 0.0]
    assert features[-2:] == [1 / 3, 1 / 3]


def test_garlic_features_exclude_origin_from_composition_windows():
    origin = dt.date(2024, 2, 1)
    rows = {code: {} for code in CLASSES}
    rows["06"][origin] = {"amount": Decimal("1000"), "weight": Decimal("100"), "rowCount": 1, "negativeRows": 0, "price": Decimal("10")}
    features = _garlic_features(rows, origin)
    assert features[0:4] == [0.0, 0.0, 0.0, 0.0]
    assert features[4:7] == [0.0, 0.0, 1.0]
    assert features[7:35] == [0.0, 0.0, 0.0, 0.0, 0.0, 15.0, 0.0] * 4
    assert features[-2:] == [0.0, 0.0]


def test_share_denominator_flag_is_zero_when_signed_total_is_nonpositive():
    origin = dt.date(2024, 2, 1)
    rows = {code: {} for code in CLASSES}
    rows["06"][origin - dt.timedelta(days=1)] = {"amount": Decimal("1"), "weight": Decimal("-1"), "rowCount": 1, "negativeRows": 1, "price": Decimal(0)}
    features = _garlic_features(rows, origin)
    assert features[0:4] == [0.0, 0.0, 0.0, 0.0]


def test_null_payload_is_ignored_without_mapping_access():
    assert _decode_payload(None) is None
    assert _decode_payload("null") is None
    assert _decode_payload("not-json") is None
    assert _decode_payload('{"trd_se":"경매"}') == {"trd_se": "경매"}


def test_signed_reconciliation_does_not_create_phantom_rows_and_is_market_scoped():
    day = dt.date(2024, 1, 1)
    source = {"M1": {day: {"06": {"amount": Decimal("12"), "weight": Decimal("2")}}}}
    a, w, rows = _market_subclass_totals(source, "M1", day)
    assert (a, w) == (Decimal("12"), Decimal("2"))
    assert "16" not in rows
    assert _market_subclass_totals(source, "M2", day)[:2] == (Decimal(0), Decimal(0))
    _assert_market_reconciled(source, "M1", day, Decimal("12"), Decimal("2"))
    try:
        _assert_market_reconciled(source, "M2", day, Decimal("12"), Decimal("2"))
    except ValueError:
        pass
    else:
        raise AssertionError("a cross-market mismatch must fail")


def test_candidate_issue_set_is_subset_of_pinned_reference():
    candidate = {("garlic", 1, dt.date(2024, 1, 1)): 1.0, ("garlic", 1, dt.date(2024, 1, 2)): 2.0}
    pinned = {("garlic", 1, dt.date(2024, 1, 1)): 0.9}
    assert _candidate_subset(candidate, pinned) == {("garlic", 1, dt.date(2024, 1, 1)): 1.0}


def test_joint_gate_requires_four_significant_pairs_and_coverage():
    tests = [{"n": 40, "dm": {"9": {"status": "available", "holmSignificant": True, "meanLossDifference": -1}}} for _ in range(4)]
    assert _joint_conclusion(tests, True, True)["conclusion"] == "improvement"
    assert _joint_conclusion(tests[:3], True, True)["conclusion"] == "inference_unavailable"
    assert _joint_conclusion(tests, True, False)["conclusion"] == "inconclusive"
    worse = [dict(t, dm={"9": {"status": "available", "holmSignificant": True, "meanLossDifference": 1}}) for t in tests]
    assert _joint_conclusion(worse, False, True)["conclusion"] == "worsening"


def test_cli_output_guard_includes_database_path():
    assert not _output_may_not_replace(Path("/tmp/data.sqlite"), Path("/tmp/prices.json"), Path("/tmp/f.json"), Path("/tmp/r.json"), Path("/tmp/data.sqlite"))


def test_build_serializes_month_audits_and_applies_holm_before_joint_gate(monkeypatch):
    monkeypatch.setattr(gm, "EVAL_START", dt.date(2023, 1, 1))
    monkeypatch.setattr(gm, "EVAL_END", dt.date(2023, 2, 10))
    monkeypatch.setattr(gm, "HORIZONS", (1,))
    dates = [dt.date(2022, 7, 1) + dt.timedelta(days=i) for i in range((dt.date(2023, 2, 10) - dt.date(2022, 7, 1)).days + 1)]
    series = {day: (100.0 + i, 1.0) for i, day in enumerate(dates)}
    monkeypatch.setattr(dh, "_observations", lambda _raw: {"garlic": series})
    persistence = {"garlic": {(day, 1): series[day][0] * 1.1 for day in series if day >= dt.date(2023, 1, 2)}}
    monkeypatch.setattr(dh, "_forecast_maps", lambda _forecast: (None, None, persistence))
    ridge = {("garlic", 1, day): series[day][0] * 1.1 for day in series if day >= dt.date(2023, 1, 2)}
    ridge_actual = {key: series[key[2]][0] for key in ridge}
    monkeypatch.setattr("agri_data.recent_window._direct_ridge_reference", lambda _report: (ridge, ridge_actual))
    def feature_row(values, origin, target, horizon, cache):
        source=target-dt.timedelta(days=horizon)
        return (source,[1.0]) if source in values else None
    monkeypatch.setattr(modeling, "_feature_row", feature_row)
    monkeypatch.setattr("threadpoolctl.threadpool_limits", lambda limits: nullcontext())

    monkeypatch.setattr(dh, "_dm", lambda _diff, _bandwidth: {"status": "available", "pValue": .001, "meanLossDifference": -1.0})

    def fake_holm(family):
        for item in family:
            item["holmPValue"] = item.get("pValue", 1.0)
            item["holmSignificant"] = item["holmPValue"] < .05

    monkeypatch.setattr(dh, "_holm", fake_holm)
    result = gm.build({}, {}, {}, {})
    assert result["fitAudits"]
    assert all(isinstance(key, str) for key in result["fitAudits"])
    json.dumps(result, allow_nan=False)
    assert result["jointConclusions"][0]["conclusion"] == "improvement"
