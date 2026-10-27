import datetime as dt
import math

from agri_data import dashboard, market_lag


def test_three_market_features_use_distinct_prior_cutoffs_and_no_filling():
    origin = dt.date(2024, 4, 15)
    series = {p: {market_lag.GARAK: {}, market_lag.GANGSEO: {}} for p in dashboard.PRODUCTS}
    d0, d7 = origin - dt.timedelta(days=2), origin - dt.timedelta(days=9)
    product = next(iter(dashboard.PRODUCTS))
    series[product][market_lag.GARAK].update({d0: 120.0, d7: 100.0})
    series[product][market_lag.GANGSEO].update({d0: 90.0, d7: 100.0})
    features = market_lag._market_features(series, product, origin)
    assert features == (math.log(1.2), math.log(0.9), math.log(120 / 90))
    # A stale p7 observation is unavailable even when a recent p0 exists.
    series[product][market_lag.GANGSEO].pop(d7)
    assert market_lag._market_features(series, product, origin) is None


def test_nonfinite_estimator_delta_falls_back_before_inverse_clipping(monkeypatch):
    inverse_calls = []

    class FakeEstimator:
        def predict(self, _features):
            return [float("inf")]

    def inverse(source_price, delta):
        inverse_calls.append((source_price, delta))
        return 123.0, 0.0

    monkeypatch.setattr(market_lag.modeling, "_inverse_log_ratio", inverse)
    assert market_lag._prediction_from_estimator(FakeEstimator(), [[1.0]], 100.0) is None
    assert inverse_calls == []


def test_joint_conclusion_retains_significant_worsening_when_other_inference_unavailable():
    def comparison(status, mean, significant):
        return {"n": 40 if status == "tested" else 0, "candidateMetric": 10.0,
                "comparatorMetric": 9.0, "dm": {"9": {"status": status,
                    "meanLossDifference": mean, "holmSignificant": significant}}}

    result = market_lag._joint_decision([
        comparison("tested", 1.0, True),
        comparison("zero_or_invalid_hac_variance", None, False),
    ], coverage_ok=True)

    assert result["inferenceAvailable"] is False
    assert result["significantWorseningEvidence"] is True
    assert result["conclusion"] == "worsening_or_mixed"


def test_build_preserves_full_comparison_family_and_prior_only_fallbacks():
    start = dt.date(2022, 12, 20)
    observations, markets, ridge_rows, recompose_rows, persistence_products = [], {}, [], [], []
    for product_index, product in enumerate(dashboard.PRODUCTS):
        rows = []
        market_rows = {market_lag.GARAK: {}, market_lag.GANGSEO: {}}
        prices = {}
        for offset in range(60):
            day = start + dt.timedelta(days=offset)
            price = 100.0 + product_index * 10 + offset
            prices[day] = price
            # Omit the first product's origin day for product 2. A cache shared
            # across products would return that absent source date and fail.
            if not (product_index == 1 and day == dt.date(2023, 2, 1)):
                rows.append({"date": day.isoformat(), "actual": price, "volumeKg": 1000.0})
            for market, multiplier in ((market_lag.GARAK, 1.2), (market_lag.GANGSEO, 0.9)):
                market_rows[market][day] = {"price": price * multiplier, "weight": 500.0}
        observations.append({"id": product, "observations": rows})
        markets[product] = market_rows
        base_day = dt.date(2023, 2, 1)
        for horizon in market_lag.HORIZONS:
            target = base_day + dt.timedelta(days=horizon)
            actual = prices[target]
            direct = actual * 1.01
            ridge_rows.append({"product": product, "horizon": horizon, "target": target.isoformat(),
                               "actual": actual, "direct": direct})
            recompose_rows.append({"product": product, "horizon": horizon, "target": target.isoformat(),
                                   "candidatePrediction": actual * 1.02})
        persistence_products.append({"id": product, "runs": [], "predictions": [
            {"target": (base_day + dt.timedelta(days=h)).isoformat(), "origin": base_day.isoformat(),
             "horizon": h, "predicted": prices[base_day + dt.timedelta(days=h)] * 1.03}
            for h in market_lag.HORIZONS]})
    raw = {"schemaVersion": 1, "products": observations}
    forecasts = {"schemaVersion": 2, "products": persistence_products}
    ridge = {"protocol": {"id": "direct-horizon-ridge-2026-10-06"}, "pairedRows": ridge_rows}
    recompose = {"originEvidence": recompose_rows}

    report = market_lag.build(raw, markets, forecasts, ridge, recompose)

    assert len(report["comparisons"]) == 300
    assert len(report["originTrace"]) == 50
    assert len(report["jointConclusions"]) == 50
    assert report["fitSummary"]["insufficient_training"] > 0
    assert all(row["fallback"] and row["candidatePrediction"] == row["directRidge"]
               for row in report["originTrace"])
    assert report["originTrace"][0]["marketFeatures"] is not None
    second_product = tuple(dashboard.PRODUCTS)[1]
    second_rows = [r for r in report["originTrace"] if r["product"] == second_product]
    assert second_rows and all(r["baseFeatureSourceDate"] == "2023-01-31" for r in second_rows)
    assert all(row["dm"]["9"]["status"] == "insufficient_pairs" for row in report["comparisons"])
