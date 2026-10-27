import datetime as dt
import math
from contextlib import nullcontext
from types import SimpleNamespace
import sys

from agri_data import dashboard
from agri_data import market_recompose as mr


def _fixture():
    day = dt.date(2023, 1, 20)
    products = []
    markets = {}
    ridge = {}
    persistence = {}
    market_predictions = {}
    for p in dashboard.PRODUCTS:
        obs = []
        markets[p] = {m: {} for m in mr.INPUT_MARKETS}
        for back in range(35, -1, -1):
            d = day - dt.timedelta(days=back)
            for m, price, weight in ((mr.GARAK, 100 + back, 3 + back), (mr.GANGSEO, 80 + back, 2 + back)):
                markets[p][m][d] = {"price": price, "weight": weight}
            if back == 0:
                obs.append({"date": d.isoformat(), "actual": 100.0, "volumeKg": 5.0})
        products.append({"id": p, "observations": obs})
        for h in mr.HORIZONS:
            target = day
            origin = target - dt.timedelta(days=h)
            ridge[(p, h, target)] = 100.0
            persistence[(p, h, target)] = 101.0
            market_predictions[(p, mr.GARAK, h, origin)] = 110.0
            market_predictions[(p, mr.GANGSEO, h, origin)] = 90.0
    raw = {"products": products}
    return raw, markets, {"__market__": market_predictions, "__persistence__": persistence}, {"predictions": []}, ridge


def test_share_requires_five_recent_cotrades_and_uses_median():
    _, markets, _, _, _ = _fixture()
    data = mr._series(markets)
    share, days, last = mr._share(data, mr.PRODUCTS[0], dt.date(2023, 1, 19))
    assert len(days) == 28
    assert last == dt.date(2023, 1, 19)
    assert 0 < share < 1
    share2, days2, _ = mr._share(data, mr.PRODUCTS[0], dt.date(2023, 1, 19) + dt.timedelta(days=15))
    assert len(days2) == 14
    assert share2 is not None  # Fourteen-day-old co-trade remains within frozen staleness limit.


def test_build_emits_full_fixed_family_and_fallback_evidence(monkeypatch):
    raw, markets, forecasts, ridge_report, ridge = _fixture()
    monkeypatch.setattr(mr.rw, "_direct_ridge_reference", lambda report: (ridge, {}))
    monkeypatch.setattr(mr.dh, "_observations", lambda raw: {
        product["id"]: {dt.date.fromisoformat(r["date"]): (r["actual"], r["volumeKg"]) for r in product["observations"]}
        for product in raw["products"]
    })
    report = mr.build(raw, markets, forecasts, ridge_report, hashes={"synthetic": "yes"})
    assert len(report["comparisons"]) == 200
    assert all(len(report["coverage"][f"{p}/{h}"]) for p in dashboard.PRODUCTS for h in mr.HORIZONS)
    garlic = next(r for r in report["originEvidence"] if r["product"] == list(dashboard.PRODUCTS)[3])
    assert garlic["fallback"] is True
    assert "always_fallback_product" in garlic["fallbackReasons"]
    assert garlic["candidatePrediction"] == 100.0
    cabbage = next(r for r in report["originEvidence"] if r["product"] == list(dashboard.PRODUCTS)[0] and r["horizon"] == 1)
    assert cabbage["candidatePrediction"] != 100.0
    assert "priceContribution" in cabbage
    assert abs(cabbage["identityResidual"]) < 1e-9


def test_non_cotrade_target_stays_paired_and_does_not_trigger_fallback(monkeypatch):
    raw, markets, forecasts, ridge_report, ridge = _fixture()
    target = dt.date(2023, 1, 20)
    p = mr.PRODUCTS[0]
    monkeypatch.setattr(mr.rw, "_direct_ridge_reference", lambda report: (ridge, {}))
    monkeypatch.setattr(mr.dh, "_observations", lambda raw: {
        product["id"]: {dt.date.fromisoformat(r["date"]): (r["actual"], r["volumeKg"]) for r in product["observations"]}
        for product in raw["products"]
    })
    full_report = mr.build(raw, markets, forecasts, ridge_report)
    del markets[p][mr.GARAK][target]
    report = mr.build(raw, markets, forecasts, ridge_report)
    full_row = next(r for r in full_report["originEvidence"] if r["product"] == p and r["horizon"] == 1)
    row = next(r for r in report["originEvidence"] if r["product"] == p and r["horizon"] == 1)
    assert row["targetCoTrade"] is False
    assert row["fallback"] is False
    assert full_row["fallback"] is False
    assert row["candidatePrediction"] == full_row["candidatePrediction"]
    assert row["candidatePrediction"] == 100.0 + (row["share"] * 10.0 - (1 - row["share"]) * 10.0)
    assert report["coverage"][f"{p}/1"]["paired"] == 1
    assert report["coverage"][f"{p}/1"]["candidateCoverageLossDates"] == {"directRidge": [], "persistence": []}
    assert row["decompositionExcludedReason"] == "target_not_positive_cotrade"
    assert report["jointConclusions"][0]["conclusion"] == "inference_unavailable"
    assert len(report["comparisons"]) == 200


def test_market_forecast_does_not_depend_on_target_market_trade(monkeypatch):
    """A target-date transaction is not part of the ex-ante forecast calendar/features."""
    p = mr.PRODUCTS[0]
    target = dt.date(2023, 1, 20)
    market = mr.GARAK
    rows = {}
    for i in range(120):
        d = dt.date(2014, 1, 1) + dt.timedelta(days=i)
        rows[d] = {"price": 100 + i * 0.1, "weight": 20 + i}
    rows[target - dt.timedelta(days=2)] = {"price": 130.0, "weight": 40.0}
    rows[target] = {"price": 131.0, "weight": 41.0}
    all_market_data = {product: {m: {} for m in mr.INPUT_MARKETS} for product in mr.PRODUCTS}
    all_market_data[p][market] = rows
    calendar = {(p, 1): [target]}
    class FixedModel:
        def fit(self, x, y): return self
        def predict(self, x): return [0.0] * len(x)
    monkeypatch.setitem(sys.modules, "threadpoolctl", SimpleNamespace(threadpool_limits=lambda **kwargs: nullcontext()))
    monkeypatch.setattr(mr.modeling, "_new_estimator", lambda model_id: FixedModel())
    first, _, _ = mr._market_forecasts(all_market_data, calendar)
    del all_market_data[p][market][target]
    second, _, _ = mr._market_forecasts(all_market_data, calendar)
    key = (p, market, 1, target - dt.timedelta(days=1))
    assert key in first and key in second
    assert math.isclose(first[key], second[key], rel_tol=0, abs_tol=1e-12)


def test_monthly_fit_uses_first_market_eligible_origin(monkeypatch):
    p, market = mr.PRODUCTS[0], mr.GARAK
    jan1, jan10, target1, target2 = (dt.date(2023, 1, 1), dt.date(2023, 1, 10),
                                     dt.date(2023, 1, 2), dt.date(2023, 1, 11))
    rows = {dt.date(2014, 1, 1) + dt.timedelta(days=i): {"price": 100 + i * .1, "weight": 20 + i} for i in range(120)}
    rows[dt.date(2023, 1, 9)] = {"price": 130.0, "weight": 40.0}  # Jan 1 source is stale; Jan 10 source is fresh.
    data = {product: {m: {} for m in mr.INPUT_MARKETS} for product in mr.PRODUCTS}
    data[p][market] = rows
    calendar = {(p, 1): [target1, target2]}
    original_feature = mr.modeling._feature_row

    def feature(series, origin, target, horizon, cache):
        if origin == jan1:
            return None
        return original_feature(series, origin, target, horizon, cache)

    class FixedModel:
        def fit(self, x, y): return self
        def predict(self, x): return [0.0] * len(x)

    monkeypatch.setitem(sys.modules, "threadpoolctl", SimpleNamespace(threadpool_limits=lambda **kwargs: nullcontext()))
    monkeypatch.setattr(mr.modeling, "_new_estimator", lambda model_id: FixedModel())
    monkeypatch.setattr(mr.modeling, "_feature_row", feature)
    predictions, audits, _ = mr._market_forecasts(data, calendar)
    key = f"{p}/{market}/1/2023-01"
    assert audits[key]["status"] == "fitted"
    assert audits[key]["fitOrigin"] == jan10.isoformat()
    assert (p, market, 1, jan10) in predictions
    assert target1 not in data[p][market] and target2 not in data[p][market]
