from __future__ import annotations

import datetime as dt

from agri_data import direct_price_path_data as data


def _series(values):
    return {dt.date(2026, 1, 1) + dt.timedelta(days=i): {"price": value}
            for i, value in enumerate(values)}


def test_cutoff_selector_uses_only_trailing_year_support_and_lexical_ties():
    end = dt.date(2026, 6, 30)
    # Equal qualifying support sorts by code, while a channel supported only
    # before the trailing year and one supported only after cutoff are excluded.
    first = {end - dt.timedelta(days=29) + dt.timedelta(days=i): {"price": 100 + i}
             for i in range(30)}
    second = dict(first)
    before = {end - dt.timedelta(days=500) + dt.timedelta(days=i): {"price": 200 + i}
              for i in range(60)}
    future = {end + dt.timedelta(days=i): {"price": 300 + i} for i in range(40)}
    selected = data.select_channels_for_cutoff({
        ("110001", "02", "A"): second,
        ("110001", "01", "B"): first,
        ("110001", "03", "C"): before,
        ("110001", "04", "D"): future,
    }, end)
    assert [r["channelId"] for r in selected] == [["110001", "01", "B"], ["110001", "02", "A"]]
    assert all(r["supportDaysInLookback"] == 30 for r in selected)


def test_input_window_is_padded_masked_and_cutoff_causal():
    origin = dt.date(2026, 6, 20)
    rep = {origin - dt.timedelta(days=i): 1000. - i for i in range(1, 100)}
    active_key = ("110001", "01", "A")
    active_series = {origin - dt.timedelta(days=i): {"price": 500. + i}
                     for i in range(1, 20)}
    stale_series = {origin - dt.timedelta(days=30): {"price": 700.}}
    channels = {active_key: active_series, ("110008",): stale_series}
    selected = [{"channelId": list(active_key)}]
    channels[("110001",)] = active_series
    m = data.build_input_window("cabbage", origin, 84, "M", rep, channels, [])
    u = data.build_input_window("cabbage", origin, 84, "U", rep, channels, [])
    assert m is not None and u is not None
    assert m["cutoff"] == (origin - dt.timedelta(days=1)).isoformat()
    assert len(m["columns"]) == len(u["columns"]) == 11
    assert m["activeAuxCount"] == 1
    assert m["columns"][1]["active"] is True
    assert m["columns"][2]["active"] is False
    assert all(x == 0 for x in m["columns"][2]["observedMask"])
    assert u["activeAuxCount"] == 0
    assert all(not c["active"] for c in u["columns"][1:])
    assert all(len(c["normalized"]) == 84 for c in m["columns"])
    # Future values for any selected channel do not enter the same origin window.
    extended = {k: dict(v) for k, v in channels.items()}
    extended[("110001",)][origin + dt.timedelta(days=1)] = {"price": 1e9}
    again = data.build_input_window("cabbage", origin, 84, "M", rep, extended, [])
    assert again == m


def test_representative_freshness_is_calendar_days_and_target_version_pinned():
    origin = dt.date(2026, 6, 20)
    rep = {origin - dt.timedelta(days=14): 100.}
    assert data.build_input_window("cabbage", origin, 84, "U", rep, {}, []) is not None
    stale = {origin - dt.timedelta(days=15): 100.}
    assert data.build_input_window("cabbage", origin, 84, "U", stale, {}, []) is None


def test_representative_loader_ignores_unrelated_snapshot_products():
    doc = {"schemaVersion": 1, "products": [
        {"id": product, "observations": [{"date": "2026-01-01", "actual": "100"}]}
        for product in (*data.PRODUCTS, "pepper")
    ]}
    got = data.representative_series(doc)
    assert set(got) == set(data.PRODUCTS)
    assert all(values == {dt.date(2026, 1, 1): 100.0} for values in got.values())
