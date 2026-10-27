import datetime as dt
import hashlib
import json
import math

import pytest

from agri_data import multiscale_direct_cv as cv
from agri_data import multiscale_residual as shared


def _fixture(monkeypatch, stale=False):
    product = "cabbage"
    monkeypatch.setattr(cv, "PRODUCTS", (product,))
    monkeypatch.setattr(shared, "PRODUCTS", (product,))
    monkeypatch.setattr(cv, "EVAL_PAIR_COUNT", 1)
    series = {}
    day = dt.date(2024, 12, 1)
    while day <= dt.date(2025, 9, 25):
        if not stale or day <= dt.date(2025, 9, 8) or day == dt.date(2025, 9, 25):
            i = (day - dt.date(2024, 12, 1)).days
            series[day] = (100. + .08 * i + math.sin(i / 8), 1.)
        day += dt.timedelta(days=1)
    monkeypatch.setattr(cv.dh, "_observations", lambda raw: {product: series})
    rows = []
    for target in (dt.date(2025, 1, 20), dt.date(2025, 2, 20)):
        for horizon in cv.HORIZONS:
            origin = target - dt.timedelta(days=horizon)
            anchor = series[origin][0]
            rows.append({"product": product, "target": target.isoformat(), "horizon": horizon,
                         "actual": anchor * math.exp(.01 * (horizon - 4)), "direct": anchor * 1.01,
                         "persistence": anchor})
    for target in (dt.date(2025, 3, 11), dt.date(2025, 8, 20)):
        for horizon in cv.HORIZONS:
            origin = target - dt.timedelta(days=horizon)
            anchor = series[origin][0]
            rows.append({"product": product, "target": target.isoformat(), "horizon": horizon,
                         "actual": anchor * math.exp(.02 * (horizon - 3)), "direct": anchor * 1.02,
                         "persistence": anchor})
    target = dt.date(2025, 3, 5)
    origin = target - dt.timedelta(days=1)
    anchor = series[origin][0]
    rows.append({"product": product, "target": target.isoformat(), "horizon": 1,
                 "actual": anchor * 1.01, "direct": anchor * 1.02, "persistence": anchor})
    target = cv.EVAL_START
    origin = target - dt.timedelta(days=1)
    actual = series[target][0]
    anchor = series[max(day for day in series if day <= origin)][0]
    rows.append({"product": product, "target": target.isoformat(), "horizon": 1,
                 "actual": actual, "direct": anchor * 1.02, "persistence": anchor})
    direct_raw = json.dumps({"pairedRows": rows}).encode()
    blend_row = {"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                 "horizon": 1, "predictions": {"blend": anchor * 1.01, "direct_ridge": anchor * 1.02,
                                                  "persistence": anchor}}
    blend_raw = json.dumps({"manifestSha256": "synthetic-blend", "forecasts": [blend_row]}).encode()
    prices_raw = json.dumps({"schemaVersion": 1, "products": []}).encode()
    for name, raw in (("PRICES_SHA", prices_raw), ("DIRECT_SHA", direct_raw), ("BLEND_SHA", blend_raw)):
        monkeypatch.setattr(cv, name, hashlib.sha256(raw).hexdigest())
    report_raw = json.dumps({"manifestSha256": "synthetic-blend"}).encode()
    monkeypatch.setattr(cv, "BLEND_REPORT_SHA", hashlib.sha256(report_raw).hexdigest())
    return prices_raw, direct_raw, blend_raw, report_raw, series


def test_direct_features_use_only_origin_history_and_keep_anchor_and_season_ablation(monkeypatch):
    prices, direct, blend, report, series = _fixture(monkeypatch)
    origin = dt.date(2025, 8, 1); target = origin + dt.timedelta(days=3)
    full = cv._feature(series, origin, target, 3, True)
    no_season = cv._feature(series, origin, target, 3, False)
    assert full[1] == series[full[0]][0]
    assert len(full[2]) == len(no_season[2]) + 2
    changed = dict(series); changed[origin + dt.timedelta(days=1)] = (99999., 1.)
    assert cv._feature(changed, origin, target, 3, True) == full


def test_alpha_choice_uses_validation_scores_only_with_frozen_tie_rule(monkeypatch):
    train = [(('cabbage', dt.date(2025, 2, 1), dt.date(2025, 2, 2), h),
              {"actual": 101.}, None, 100., [0.], 0) for h in cv.HORIZONS]
    validation = [(('cabbage', dt.date(2025, 3, 10), dt.date(2025, 3, 11), h),
                   {"actual": 100 * math.exp(.2)}, None, 100., [0.], 0) for h in cv.HORIZONS]
    monkeypatch.setattr(cv, "_fit", lambda X, y, alpha, season: alpha)
    corrections = {.001: 0., .1: .2, 10.: .1}
    monkeypatch.setattr(cv, "_predict", lambda model, X, season: corrections[model])
    selected, scores = cv._select_alpha(train, validation)
    assert selected == .1
    assert [row["alpha"] for row in scores] == [.001, .1, 10.]
    assert scores[1]["meanHorizonMape"] == pytest.approx(0.)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_chronological_split_manifest_and_score(monkeypatch):
    prices, direct, blend, blend_report, _ = _fixture(monkeypatch)
    manifest = cv.build_forecast(prices, direct, blend)
    assert manifest["alphaSelection"]["cabbage"]["trainingRows"] == 20
    assert manifest["alphaSelection"]["cabbage"]["validationRows"] == 20
    assert manifest["refitRowsByProduct"]["cabbage"] == 41
    assert manifest["trainingTargetEnd"] == "2025-02-28"
    assert manifest["validation"]["start"] == "2025-03-11"
    assert len(manifest["forecasts"]) == 1
    assert "actual" not in manifest["forecasts"][0]
    raw = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    result = cv.score(raw, prices, direct, blend, blend_report)
    assert result["coverage"]["commonScoredPairs"] == 1
    assert len(result["dmTests"]) == 4 * 1 * 10 * 2 * 2
    direct_cell = result["metrics"]["direct"]["cells"]["cabbage|1"]
    assert direct_cell["n"] == 1 and direct_cell["mae"] > 0 and direct_cell["mape"] > 0
    assert all(test["status"] == "inference_unavailable_insufficient_pairs" for test in result["dmTests"])


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_stale_asof_price_falls_back_to_direct_with_reason(monkeypatch):
    prices, direct, blend, _, _ = _fixture(monkeypatch, stale=True)
    manifest = cv.build_forecast(prices, direct, blend)
    row = manifest["forecasts"][0]
    assert row["lineage"]["fallback"] is True
    assert row["lineage"]["fallbackReason"] == "latest_source_older_than_14_days"
    assert row["predictions"]["candidate"] == row["predictions"]["direct"]


@pytest.mark.parametrize("mutation", ["candidate", "ablation", "direct", "blend", "persistence", "protocol", "alpha", "extra", "missing"])
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_score_rejects_rehashed_manifest_forgery(monkeypatch, mutation):
    prices, direct, blend, blend_report, _ = _fixture(monkeypatch)
    manifest = cv.build_forecast(prices, direct, blend)
    if mutation in ("candidate", "ablation", "direct", "blend", "persistence"):
        manifest["forecasts"][0]["predictions"][mutation] += 1.
    elif mutation == "protocol":
        manifest["protocol"] = "forged"
    elif mutation == "alpha":
        manifest["alphaSelection"]["cabbage"]["selectedAlpha"] = 999.
    elif mutation == "extra":
        row = dict(manifest["forecasts"][0])
        row["target"] = (dt.date.fromisoformat(row["target"]) + dt.timedelta(days=1)).isoformat()
        row["origin"] = (dt.date.fromisoformat(row["origin"]) + dt.timedelta(days=1)).isoformat()
        manifest["forecasts"].append(row)
    else:
        manifest["forecasts"].pop()
    manifest.pop("manifestSha256")
    manifest["manifestSha256"] = cv._canonical(manifest)
    raw = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    with pytest.raises(ValueError, match="protocol|deterministic_reconstruction"):
        cv.score(raw, prices, direct, blend, blend_report)


def test_multiscale_change_features_use_half_open_calendar_windows():
    origin = dt.date(2025, 3, 1)
    observations = {origin - dt.timedelta(days=age): (price, 1.) for age, price in
                    ((0, 20.), (6, 10.), (7, 8.), (27, 4.), (28, 2.), (55, 1.), (56, .5))}
    feature = cv._feature(observations, origin, origin + dt.timedelta(days=1), 1)
    values = feature[2]
    # Indices 1..4, 5..8, and 9..12 are 7/28/56-day
    # deviation, count, volatility, and chronological first-to-last change.
    assert values[2] == 2
    assert values[6] == 4
    assert values[10] == 6
    assert values[4] == pytest.approx(math.log(20.) - math.log(10.))
    assert values[8] == pytest.approx(math.log(20.) - math.log(4.))
    assert values[12] == pytest.approx(math.log(20.) - math.log(1.))
    assert values[8] != 0.  # the 28-day change has its own represented feature
