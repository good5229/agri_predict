import datetime as dt
import hashlib
import json

import numpy as np
import pytest

from agri_data import multiscale_residual as mr


def _fixture(monkeypatch):
    product = "cabbage"
    monkeypatch.setattr(mr, "PRODUCTS", (product,))
    monkeypatch.setattr(mr, "EVAL_PAIR_COUNT", 1)
    series = {dt.date(2025, 1, 1) + dt.timedelta(days=i): (100 + i * .1, 1.) for i in range(280)}
    monkeypatch.setattr(mr.dh, "_observations", lambda raw: {product: series})
    paired = []
    for i in range(40):
        target = dt.date(2025, 6, 1) + dt.timedelta(days=i)
        paired.append({"product": product, "target": target.isoformat(), "horizon": 1,
                       "actual": 120 + i * .2, "direct": 119 + i * .2, "persistence": 118.})
    target = mr.EVAL_START
    series[target] = (130., 1.)
    origin = target - dt.timedelta(days=1)
    paired.append({"product": product, "target": target.isoformat(), "horizon": 1,
                   "actual": 130., "direct": 129., "persistence": 128.})
    direct_raw = json.dumps({"pairedRows": paired}).encode()
    blend_row = {"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                 "horizon": 1, "predictions": {"blend": 127., "direct_ridge": 129., "persistence": 128.}}
    blend_raw = json.dumps({"manifestSha256": "test-blend", "forecasts": [blend_row]}).encode()
    prices_raw = json.dumps({"schemaVersion": 1, "products": []}).encode()
    for name, raw in (("PRICES_SHA", prices_raw), ("DIRECT_SHA", direct_raw), ("BLEND_SHA", blend_raw)):
        monkeypatch.setattr(mr, name, hashlib.sha256(raw).hexdigest())
    report_raw = json.dumps({"manifestSha256": json.loads(blend_raw)["manifestSha256"]}).encode()
    monkeypatch.setattr(mr, "BLEND_REPORT_SHA", hashlib.sha256(report_raw).hexdigest())
    return prices_raw, direct_raw, blend_raw, report_raw, series, paired, blend_row


def test_multiscale_features_are_causal_and_ablation_only_removes_season(monkeypatch):
    _, _, _, _, series, _, _ = _fixture(monkeypatch)
    origin = dt.date(2025, 8, 1); target = origin + dt.timedelta(days=3)
    full = mr._features(series, origin, target, 3, True)
    no_season = mr._features(series, origin, target, 3, False)
    assert len(full) == len(no_season) + 2
    changed = dict(series); changed[origin + dt.timedelta(days=1)] = (9999., 1.)
    assert mr._features(changed, origin, target, 3, True) == full


def test_forecast_split_answer_free_manifest_and_tamper_check(monkeypatch):
    args = _fixture(monkeypatch)
    manifest = mr.build_forecast(*args[:3])
    assert manifest["phase"] == "forecast_only" and manifest["targetValuesAttached"] is False
    assert len(manifest["forecasts"]) == 1
    assert "actual" not in manifest["forecasts"][0]
    assert manifest["trainingTargetEnd"] == mr.TRAIN_END.isoformat()
    raw = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    assert len(mr._verify_manifest(manifest, raw)) == 1
    broken = dict(manifest); broken["forecasts"] = [dict(manifest["forecasts"][0], actual=130.)]
    broken_raw = (json.dumps(broken, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    with pytest.raises(ValueError, match="tampered"):
        mr._verify_manifest(broken, broken_raw)


def test_score_checks_frozen_actual_agreement_and_computes_four_contrast_family(monkeypatch):
    prices, direct, blend, blend_report, series, paired, blend_row = _fixture(monkeypatch)
    manifest = mr.build_forecast(prices, direct, blend)
    manifest_raw = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    result = mr.score(manifest_raw, prices, direct, blend, blend_report)
    assert result["coverage"]["commonScoredPairs"] == 1
    assert len(result["dmTests"]) == 4 * 1 * 10 * 2 * 2
    assert set(result["metrics"]) == set(mr.METHODS)
    wrong = dict(series); wrong[dt.date.fromisoformat(blend_row["target"])] = (131., 1.)
    monkeypatch.setattr(mr.dh, "_observations", lambda raw: {"cabbage": wrong})
    with pytest.raises(ValueError, match="actual mismatch"):
        mr.score(manifest_raw, prices, direct, blend, blend_report)


def test_dm_calendar_hac_uses_mean_over_sqrt_variance_divided_by_n():
    spaced = [(dt.date(2025, 1, 1) + dt.timedelta(days=40 * i), 0. if i % 2 == 0 else 2.)
              for i in range(100)]
    result = mr._dm(spaced, 9)
    assert result["statistic"] == pytest.approx(10.)

    correlated = [(dt.date(2025, 1, 1) + dt.timedelta(days=2 * i), float((i % 4)))
                  for i in range(100)]
    mean = sum(value for _, value in correlated) / len(correlated)
    centered = {day: value - mean for day, value in correlated}
    variance = sum(value * value for value in centered.values())
    for lag in range(1, 4):
        variance += 2 * (1 - lag / 4) * sum(
            value * centered.get(day - dt.timedelta(days=lag), 0.)
            for day, value in centered.items())
    expected = mean / (variance ** .5 / len(correlated))
    observed = mr._dm(correlated, 3)
    assert observed["statistic"] == pytest.approx(expected)
    assert observed["statistic"] != pytest.approx(mean / (variance / len(correlated)) ** .5)


@pytest.mark.parametrize("mutation", ["candidate", "ablation", "persistence", "protocol", "metadata", "extra", "missing"])
def test_score_rejects_rehashed_forged_manifest_before_scoring(monkeypatch, mutation):
    prices, direct, blend, blend_report, _, _, _ = _fixture(monkeypatch)
    manifest = mr.build_forecast(prices, direct, blend)
    if mutation in ("candidate", "ablation", "persistence"):
        manifest["forecasts"][0]["predictions"][mutation] += 1.
    elif mutation == "protocol":
        manifest["protocol"] = "forged-protocol"
    elif mutation == "metadata":
        manifest["trainingRowsByProduct"]["cabbage"] += 1
    elif mutation == "extra":
        row = dict(manifest["forecasts"][0])
        row["target"] = (dt.date.fromisoformat(row["target"]) + dt.timedelta(days=1)).isoformat()
        row["origin"] = (dt.date.fromisoformat(row["origin"]) + dt.timedelta(days=1)).isoformat()
        manifest["forecasts"].append(row)
    else:
        manifest["forecasts"].pop()
    manifest.pop("manifestSha256")
    manifest["manifestSha256"] = mr.canonical(manifest)
    raw = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    with pytest.raises(ValueError, match="deterministic reconstruction"):
        mr.score(raw, prices, direct, blend, blend_report)
