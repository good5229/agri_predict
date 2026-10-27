from __future__ import annotations

import datetime as dt
import math

import numpy as np

from agri_data import period_v9_correction as v9


def row(start="2024-01-01", end="2024-01-10", origin="2023-12-20", *, crop="cabbage", horizon=2, base=100., pref=100.):
    r = {"crop": crop, "horizon": horizon, "targetStart": start, "targetEnd": end,
         "origin": origin, "inputCutoff": "2023-12-18", "p_ref_restored": pref,
         "base": base, "BASE": base, "month_sin": 0., "month_cos": 1.}
    r["_rowkey"] = "|".join(map(str, v9._key(r)))
    return r


def fact(value, group, released="2022-01-01"):
    if len(released) == 10:
        released += "T09:59:00+09:00"
    return {"value": value, "updateGroupId": group, "publishEligibleAt": released,
            "ageDays": 0, "overlap": "known_nonoverlap"}


def test_label_maturity_includes_exact_target_end_plus_two_day_boundary():
    r = row(end="2024-01-10")
    assert v9._mature(r, dt.date(2024, 1, 12))
    assert not v9._mature(r, dt.date(2024, 1, 11))


def test_log_ratio_splits_positive_and_negative_parts():
    up = row(base=120., pref=100.)
    down = row(base=80., pref=100.)
    x_up, _ = v9._base_features(up, 100.)
    x_down, _ = v9._base_features(down, 100.)
    assert x_up[:2] == [math.log(1.2), 0.]
    assert x_down[:2] == [0., math.log(.8)]


def test_shared_ridge_uses_only_mature_training_rows_and_clip_guardrail():
    train = [row(start=f"2022-01-{d:02d}", end=f"2022-01-{d+9:02d}", origin=f"2022-01-{d:02d}", base=100.+d)
             for d in (1, 11, 21)]
    for i, r in enumerate(train):
        r["_rowkey"] = f"train-{i}"
    labels = {r["_rowkey"]: v for r, v in zip(train, (-.3, .05, .3))}
    refs = {r["_rowkey"]: 100. for r in train}
    cur = row(start="2023-01-01", end="2023-01-10", origin="2022-12-20", base=100.)
    cur["_rowkey"] = "current"
    bundle = v9._fit_model(train, labels, refs, {}, with_supply=False)
    pred = v9._predict_model(bundle, cur, {**refs, "current": 100.}, {}, with_supply=False)
    assert bundle is not None
    assert bundle["audit"]["trainRowKeys"] == ["train-0", "train-1", "train-2"]
    assert .8*cur["BASE"] <= pred["prediction"] <= 1.2*cur["BASE"]
    assert bundle["model"].solver == "svd"


def test_supply_value_mean_imputation_mask_and_mean_centered_direct_effect():
    train = []
    supply = {}
    vals = (20., None, 40.)
    groups = ("g1", None, "g2")
    for i, (value, group) in enumerate(zip(vals, groups)):
        r = row(start=f"2022-0{i+1}-01", end=f"2022-0{i+1}-10", origin=f"2022-0{i+1}-15", crop="radish")
        r["_rowkey"] = f"train-{i}"
        train.append(r)
        if value is not None:
            supply[v9._key(r)] = fact(value, group, r["origin"])
    labels = {r["_rowkey"]: .02*i for i, r in enumerate(train)}
    refs = {r["_rowkey"]: 100. for r in train}
    cur = row(start="2024-01-01", end="2024-01-10", origin="2023-12-20", crop="radish")
    cur["_rowkey"] = "current"
    supply[v9._key(cur)] = fact(30., "g3", cur["origin"])
    bundle = v9._fit_model(train, labels, refs, supply, with_supply=True)
    assert bundle is not None
    assert bundle["audit"]["supplyDistinctTrainingGroups"] == 2
    assert bundle["supplyMean"] == 30.
    # Missing numeric training value is imputed to the observed mean, not zero.
    assert bundle["means"][6] == 30.
    pred = v9._predict_model(bundle, cur, {**refs, "current": 100.}, supply, with_supply=True)
    assert pred["prediction"] is not None
    assert pred["directSupplyLogContribution"] == 0.


def test_e_requires_two_independent_nonconstant_groups_and_current_numeric_value():
    train = [row(start=f"2022-0{i+1}-01", end=f"2022-0{i+1}-10", origin=f"2022-0{i+1}-15", crop="radish") for i in range(2)]
    for i, r in enumerate(train):
        r["_rowkey"] = f"train-{i}"
    labels = {r["_rowkey"]: .01*i for i, r in enumerate(train)}
    refs = {r["_rowkey"]: 100. for r in train}
    constant = {v9._key(train[0]): fact(20., "g1", train[0]["origin"]),
                v9._key(train[1]): fact(20., "g2", train[1]["origin"])}
    assert v9._fit_model(train, labels, refs, constant, with_supply=True) is None
    varied = {v9._key(train[0]): fact(20., "g1", train[0]["origin"]),
              v9._key(train[1]): fact(30., "g2", train[1]["origin"])}
    bundle = v9._fit_model(train, labels, refs, varied, with_supply=True)
    assert bundle is not None
    cur = row(crop="radish")
    cur["_rowkey"] = "current"
    # No current numeric value makes E unavailable; caller must use exact C.
    assert v9._predict_model(bundle, cur, {**refs, "current": 100.}, varied, with_supply=True)["prediction"] is None


def test_e_feature_does_not_turn_missing_fact_into_zero_quantity():
    cur = row(crop="radish")
    features, fact = v9._supply_features(cur, {})
    assert fact is None
    assert features == [0., 0., 0., 0., 1.]


def test_overlap_feature_marks_knownness_separately_from_overlap_value():
    current = row(crop="radish")
    current["origin"] = "2024-01-10"
    known = {v9._key(current): {**fact(20., "g1", "2024-01-01"), "overlap": "known_nonoverlap"}}
    unknown = {v9._key(current): {**fact(20., "g1", "2024-01-01"), "overlap": "unknown"}}
    assert v9._supply_features(current, known)[0][2] == 1.
    assert v9._supply_features(current, unknown)[0][2] == 0.


def test_training_set_cache_reuses_identical_fit_and_does_not_use_current_rows():
    train = [row(start="2022-01-01", end="2022-01-10", origin="2022-01-15"),
             row(start="2022-02-01", end="2022-02-10", origin="2022-02-15")]
    for i, r in enumerate(train):
        r["_rowkey"] = f"train-{i}"
    labels = {"train-0": .01, "train-1": -.02}
    refs = {"train-0": 100., "train-1": 100.}
    cache = {}
    first = v9._fit_cached(cache, train, labels, refs, {}, with_supply=False)
    second = v9._fit_cached(cache, list(reversed(train)), labels, refs, {}, with_supply=False)
    assert first is second
    assert len(cache) == 1
    assert first["audit"]["trainRowKeys"] == ["train-0", "train-1"]


def test_dm_holm_keeps_all_sixteen_slots_per_lag_and_signed_direction():
    rows = []
    target = (dt.date(2024, 1, 1), dt.date(2024, 1, 10))
    for i in range(15):
        if i:
            target = v9.proxy._next_period(target, 1)
        for crop in v9.CROPS:
            for horizon in v9.HORIZONS:
                actual = 100. + i
                rows.append({"crop": crop, "horizon": horizon, "targetStart": target[0].isoformat(),
                             "actual": actual, "BASE": actual+3., "C": actual+(i % 3), "E": actual+(i % 3)})
    result = v9._dm_results(rows, "C", "BASE")
    assert len(result) == 32
    for lag in v9.DM_LAGS:
        family = [r for r in result if r["hacLagDekads"] == lag]
        assert len(family) == 16
        assert all(r["holmFamilySize"] == 16 for r in family)
        assert all(r["nDistinctTargetDekads"] == 15 for r in family)
        assert all(r["meanDifference"] < 0 for r in family)


def test_calendar_block_bootstrap_reports_per_cell_and_macro_direction():
    rows = []
    target = (dt.date(2024, 1, 1), dt.date(2024, 1, 10))
    for i in range(99):
        if i:
            target = v9.proxy._next_period(target, 1)
        for crop in v9.CROPS:
            for horizon in v9.HORIZONS:
                actual = 100. + i
                rows.append({"crop": crop, "horizon": horizon, "targetStart": target[0].isoformat(),
                             "actual": actual, "BASE": actual+2., "C": actual+1., "E": actual+1.})
    c = v9._block_bootstrap(rows, "C", "BASE")
    e = v9._block_bootstrap(rows, "E", "C")
    assert len(c) == 2 * 2 * (8 + 1)
    macro = [r for r in c if r["aggregate"] == "all_8cell_equal_weight"]
    assert len(macro) == 4
    assert all(r["direction"] == "improvement" and r["observedDifference"] < 0 for r in macro)
    assert all(r["replicatesUsable"] == v9.BOOTSTRAP_REPLICATES for r in macro)
    assert all(r["direction"] == "inconclusive" for r in e)



def test_supply_fact_expiration_uses_exact_c_fallback_after_30_days():
    current = row(start="2024-02-01", origin="2024-02-01", crop="radish")
    stale = {v9._key(current): fact(20., "g1", released="2024-01-01")}
    features, matched = v9._supply_features(current, stale)
    assert matched is None
    assert features == [0., 0., 0., 0., 1.]


def test_publication_timestamp_must_precede_origin_10_kst():
    current = row(start="2024-01-10", origin="2024-01-10", crop="radish")
    accepted = {v9._key(current): fact(20., "g1", "2024-01-10T09:59:00+09:00")}
    rejected = {v9._key(current): fact(20., "g1", "2024-01-10T10:01:00+09:00")}
    date_only_same_day = {v9._key(current): {**fact(20., "g1", "2024-01-10"),
                                            "publishEligibleAt": "2024-01-10"}}
    previous_date_only = {v9._key(current): {**fact(20., "g1", "2024-01-09"),
                                                  "publishEligibleAt": "2024-01-09"}}
    assert v9._supply_features(current, accepted)[1] is not None
    assert v9._supply_features(current, rejected)[1] is None
    assert v9._supply_features(current, date_only_same_day)[1] is None
    assert v9._supply_features(current, previous_date_only)[1] is not None


def test_month_fallback_matches_v5_calendar_convention():
    current = row(start="2024-01-01")
    current["month_sin"] = None
    current["month_cos"] = None
    cont, _ = v9._base_features(current, 100.)
    assert cont[3:] == [0., 1.]
