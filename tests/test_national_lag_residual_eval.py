import datetime as dt

from agri_data import dashboard, direct_horizon as dh, national_lag_residual_eval as ev


def _row(price, quantity=10):
    return {"price": str(price), "quantitySum": str(quantity)}


def test_feature_observations_respect_origin_lag_and_age_cutoffs():
    origin = dt.date(2025, 1, 20)
    obs = {
        "2025-01-18": _row(100),  # o-2, selected
        "2025-01-19": _row(900),  # too recent
        "2025-01-11": _row(80),   # o-9, lag reference
        "2025-01-20": _row(999),  # future information at origin
    }
    # Exercise the actual direct-horizon observation interface: each date maps
    # to (price, weight), not a scalar price.
    prices = {"schemaVersion": 1, "products": []}
    for product in dashboard.PRODUCTS:
        observations = [{"date": "2025-01-20", "actual": "50", "volumeKg": "7"}]
        prices["products"].append({"id": product, "observations": observations})
    parsed_core = dh._observations(prices)["cabbage"]
    core_series = {day.isoformat(): value for day, value in parsed_core.items()}
    features, audit = ev._feature(origin, obs, core_series)
    assert audit["nationalDate"] == "2025-01-18"
    assert audit["twoMarketDate"] == "2025-01-20"
    assert audit["lagDate"] == "2025-01-11"
    assert features[0] == __import__("math").log(2)
    assert features[1] == __import__("math").log(10)
    assert features[2:] == [2.0, __import__("math").log(1.25), 1.0]


def test_stale_or_incomplete_national_observations_fall_back():
    origin = dt.date(2025, 2, 1)
    obs = {"2025-01-01": _row(100)}
    vector, audit = ev._feature(origin, obs, {"2025-02-01": "50"})
    assert vector is None
    assert audit["status"] == "national_missing_or_stale"
    obs["2025-01-20"] = {"price": "20", "quantitySum": None}
    vector, audit = ev._feature(origin, obs, {"2025-02-01": "50"})
    assert vector is None
    assert audit["status"] == "national_missing_or_stale"


def test_fixed_holm_family_retains_unavailable_tests():
    family = [{"pValue": 0.001}, {"pValue": 1.0}, {"pValue": 0.03}]
    ev._holm(family)
    assert family[0]["holmSignificant"]
    assert family[1]["holmPValue"] == 1.0
    assert not family[1]["holmSignificant"]


def test_nonfinite_residual_uses_direct_fallback():
    assert ev._adjusted_prediction(100.0, float("nan")) is None
    assert ev._adjusted_prediction(100.0, float("inf")) is None
    assert ev._adjusted_prediction(100.0, 0.0) == 100.0
