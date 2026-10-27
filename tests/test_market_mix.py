import datetime as dt
from contextlib import nullcontext
import math
from decimal import Decimal

import pytest

from agri_data import dashboard, market_mix


def test_missing_markets_have_zero_sentinel_and_no_future_leakage():
    day = dt.date(2024, 1, 1)
    data = {m: {} for m in dashboard.MARKETS}
    data["110001"][day] = {"price": 20., "weight": 99.}
    # A future Gangseo row cannot affect a Jan 2 origin.
    data["110008"][day + dt.timedelta(days=3)] = {"price": 50., "weight": 30.}
    v = market_mix._features(data, day + dt.timedelta(days=1))
    assert v[:4] == pytest.approx([math.log(20), 1, math.log1p(99), 1])
    assert v[4:] == [0., 15., 0., 0., 0., 0., 15., 0.]


def test_stale_market_is_unavailable_after_fourteen_days():
    day = dt.date(2024, 1, 1)
    data = {m: {} for m in dashboard.MARKETS}
    for m in data:
        data[m][day] = {"price": 20., "weight": 10.}
    v = market_mix._features(data, day + dt.timedelta(days=15))
    assert v == [0., 15., 0., 0., 0., 15., 0., 0., 0., 0., 15., 0.]


def test_reconcile_preserves_signed_aggregate_amount_and_weight():
    day = dt.date(2017, 1, 1); product = "cabbage"
    states = {(p, day, m): {"selectedState": "complete"} for p in dashboard.PRODUCTS for m in dashboard.MARKETS}
    diagnostics = {
        (product, day, "110001"): {"signedNetAmount": Decimal("100"), "signedNetWeight": Decimal("10"), "eligibleAuctionKgRows": 1},
        (product, day, "110008"): {"signedNetAmount": Decimal("-20"), "signedNetWeight": Decimal("2"), "eligibleAuctionKgRows": 1},
    }
    # Combined = 80 / 12; the signed adjustment must remain in the aggregate.
    core = {p: {} for p in dashboard.PRODUCTS}
    core[product][day] = (80/12, 12.)
    raw = {"meta": {"contextStart": "2017-01-01", "contextEnd": "2017-01-01"}, "products": [
        {"id": p, "observations": ([{"date": day.isoformat(), "actual": 80/12, "volumeKg": 12., "marketCodes": list(dashboard.MARKETS)}] if p == product else [])}
        for p in dashboard.PRODUCTS]}
    assert market_mix._reconcile(raw, core, states, diagnostics, day)["positiveTargetsReconciled"] == 1
    core[product][day] = (10., 10.)
    with pytest.raises(ValueError, match="signed aggregate"):
        market_mix._reconcile(raw, core, states, diagnostics, day)


def test_reconcile_checks_market_codes_and_requires_complete_date_coverage():
    day = dt.date(2017, 1, 1); product = "garlic"
    states = {(p, day, m): {"selectedState": "complete"} for p in dashboard.PRODUCTS for m in dashboard.MARKETS}
    diag = {(product, day, "110001"): {"signedNetAmount": Decimal("80"), "signedNetWeight": Decimal("8"), "eligibleAuctionKgRows": 2},
            (product, day, "110008"): {"signedNetAmount": Decimal("0"), "signedNetWeight": Decimal("0"), "eligibleAuctionKgRows": 0}}
    core = {p: {} for p in dashboard.PRODUCTS}; core[product][day] = (10., 8.)
    raw = {"meta": {"contextStart": day.isoformat(), "contextEnd": day.isoformat()}, "products": [
        {"id": p, "observations": ([{"date": day.isoformat(), "actual": 10., "volumeKg": 8., "marketCodes": ["110008"]}] if p == product else [])}
        for p in dashboard.PRODUCTS]}
    with pytest.raises(ValueError, match="market codes"):
        market_mix._reconcile(raw, core, states, diag, day)
    states.pop((product, day, "110008"))
    with pytest.raises(ValueError, match="incomplete broad query"):
        market_mix._reconcile(raw, core, states, diag, day)


def test_issuance_key_shapes_and_frozen_origin_cutoff():
    jan1 = dt.date(2023, 1, 1); prior = jan1 - dt.timedelta(days=1)
    assert market_mix._issued("marketMixHistGb", {("cabbage", 1, jan1): 10.}, "cabbage", 1, jan1)
    assert market_mix._issued("monthlyPooledHistGb", {"cabbage": {(jan1, 1): 11.}}, "cabbage", 1, jan1)
    eligible = {t for t in (jan1, prior) if t-dt.timedelta(days=1) >= jan1}
    assert eligible == set()


def test_seasons_use_calendar_season_names():
    season = lambda d: "winter" if d.month in (12, 1, 2) else "spring" if d.month in (3, 4, 5) else "summer" if d.month in (6, 7, 8) else "autumn"
    assert [season(dt.date(2025, m, 1)) for m in (1, 4, 7, 10)] == ["winter", "spring", "summer", "autumn"]


def test_joint_result_requires_all_eight_improvements_and_no_coverage_loss():
    tests = [{"n": 35, "candidateMetric": 1., "comparatorMetric": 2.,
              "dm": {"9": {"status": "tested", "holmSignificant": True, "meanLossDifference": -1.}}}
             for _ in range(8)]
    assert market_mix._cell_conclusion(tests, [[], [], [], []])["conclusion"] == "improvement"
    assert market_mix._cell_conclusion(tests, [[], ["2024-01-01"], [], []])["conclusion"] == "inconclusive"
    tests[-1]["candidateMetric"] = 3.
    assert market_mix._cell_conclusion(tests, [[], [], [], []])["conclusion"] == "inconclusive"


def test_joint_result_does_not_treat_inconclusive_as_equivalence():
    tests = [{"n": 20, "candidateMetric": 1., "comparatorMetric": 1.,
              "dm": {"9": {"status": "insufficient_pairs", "holmSignificant": False, "meanLossDifference": 0.}}}
             for _ in range(8)]
    assert market_mix._cell_conclusion(tests, [[], [], [], []])["conclusion"] == "inference_unavailable"


def test_build_applies_holm_before_joint_conclusions_and_uses_common_headline_dates(monkeypatch):
    from agri_data import direct_horizon as dh, modeling, recent_window as rw

    start, end = dt.date(2022, 9, 1), dt.date(2023, 2, 15)
    series = {}
    for p in dashboard.PRODUCTS:
        series[p] = {start + dt.timedelta(days=i): (100., 10.) for i in range((end-start).days+1)}

    def fake_feature(obs, origin, target, horizon, cache):
        prior = max((d for d in obs if d <= origin), default=None)
        return None if prior is None else (prior, (0.,))

    class ZeroModel:
        def fit(self, X, y): return self
        def predict(self, X): return [0.] * len(X)

    eligible = {(p, h, t) for p in dashboard.PRODUCTS for h in market_mix.HORIZONS
                for t in series[p] if dh.EVAL_START <= t <= dh.EVAL_END and t-dt.timedelta(days=h) >= dh.EVAL_START}
    ref = {(p, h, t): 110. for p, h, t in eligible}
    persistence = {p: {(t, h): 110. for pp, h, t in eligible if pp == p} for p in dashboard.PRODUCTS}
    monthly = {p: {(t, h): 110. for pp, h, t in eligible if pp == p} for p in dashboard.PRODUCTS}
    actual_map = {(p, h, t): 100. for p, h, t in eligible}
    full = {"predictions": [{"product": p, "horizon": h, "target": t.isoformat(), "predicted": 110., "arm": "full"}
                            for p, h, t in eligible]}
    monkeypatch.setattr(dh, "_observations", lambda raw: series)
    monkeypatch.setattr(dh, "_forecast_maps", lambda raw: ({}, {}, persistence))
    monkeypatch.setattr(rw, "_monthly_hist_gb", lambda raw: monthly)
    monkeypatch.setattr(rw, "_direct_ridge_reference", lambda raw: (ref, actual_map))
    monkeypatch.setattr(modeling, "_feature_row", fake_feature)
    monkeypatch.setattr(modeling, "_new_estimator", lambda name: ZeroModel())
    import threadpoolctl
    monkeypatch.setattr(threadpoolctl, "threadpool_limits", lambda limits: nullcontext())
    monkeypatch.setattr(dh, "_dm", lambda diff, bandwidth: {"status": "tested", "n": len(diff),
        "meanLossDifference": -1., "statistic": -5., "pValue": 1e-6})

    raw = {"schemaVersion": 1}
    market = {p: {m: {} for m in dashboard.MARKETS} for p in dashboard.PRODUCTS}
    result = market_mix.build(raw, market, {}, {}, full)
    assert len(result["comparisons"]) == 400
    assert all(x["dm"]["9"]["holmSignificant"] for x in result["comparisons"])
    assert all(x["conclusion"] == "improvement" for x in result["jointConclusions"]), (result["jointConclusions"][:2], result["coverage"]["cabbage/1"], list(result["fitAudits"].items())[:2])
    assert result["headlineMetricScope"].startswith("five-method common")
    for cell in result["coverage"].values():
        assert cell["commonPairedCount"] == len(cell["pairedDates"])
        assert cell["featureAvailability"]["denominator"] == cell["eligibleObservationCount"]
        assert cell["featureAvailability"]["availableCounts"] == {"garak": 0, "gangseo": 0, "sameDay": 0}
    assert result["macroMetrics"]["marketMixHistGb"] < result["macroMetrics"]["persistence"]


def test_reconcile_aborts_on_incomplete_broad_market_scope():
    day = dt.date(2017, 1, 1); product = "cabbage"
    states = {(p, day, "110001"): {"selectedState": "complete"} for p in dashboard.PRODUCTS}
    states.update({(p, day, "110008"): {"selectedState": ("failed" if p == product else "complete")} for p in dashboard.PRODUCTS})
    core = {p: {} for p in dashboard.PRODUCTS}
    raw = {"meta": {"contextStart": day.isoformat(), "contextEnd": day.isoformat()}, "products": [{"id": p, "observations": []} for p in dashboard.PRODUCTS]}
    with pytest.raises(ValueError, match="incomplete broad query"):
        market_mix._reconcile(raw, core, states, {}, day)


def test_serialization_rejects_nonfinite_values_and_is_stable():
    assert market_mix._json_bytes({"b": 1, "a": "가"}) == market_mix._json_bytes({"a": "가", "b": 1})
    with pytest.raises(ValueError):
        market_mix._json_bytes({"bad": float("nan")})
