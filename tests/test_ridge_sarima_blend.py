import datetime as dt
import copy
import hashlib
import json
import math

import pytest

from agri_data import ridge_sarima_blend as blend


def _raw(doc):
    return json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=2).encode()


def _source_rows():
    rows = []
    for day, direct, sarima, persistence, fallback in (
        ("2025-09-25", 100.0, 120.0, 90.0, False),
        ("2025-09-26", 200.0, 220.0, 190.0, True),
    ):
        target = dt.date.fromisoformat(day)
        rows.append({"product": "cabbage", "origin": (target-dt.timedelta(days=1)).isoformat(),
                     "target": day, "horizon": 1,
                     "predictions": {"direct_ridge": direct,
                                     "calendar_sarima_111_100_7": sarima,
                                     "naive": persistence},
                     "fallback": {"calendar_sarima_111_100_7": fallback},
                     "nativePredictions": {"calendar_sarima_111_100_7": None if fallback else sarima},
                     "methodStatus": {"calendar_sarima_111_100_7": {"fallback": fallback,
                                                                       "fallbackReason": "fit_unavailable" if fallback else None}}})
    return rows


def _source():
    return {"protocol": blend.TEN_PROTOCOL, "phase": "forecast_only",
            "targetValuesAttached": False, "forecasts": _source_rows()}


def _patch_tiny(monkeypatch, source_raw):
    monkeypatch.setattr(blend, "TEN_MANIFEST_SHA", hashlib.sha256(source_raw).hexdigest())
    monkeypatch.setattr(blend, "PLANNED", 2)
    monkeypatch.setattr(blend, "PRODUCTS", ("cabbage",))
    monkeypatch.setattr(blend, "HORIZONS", (1,))


def test_forecast_phase_is_answer_free_and_respects_fixed_fallback(monkeypatch):
    raw = _raw(_source())
    _patch_tiny(monkeypatch, raw)
    result = blend.build_forecast(raw)
    assert result["phase"] == "forecast_only"
    assert result["targetValuesAttached"] is False
    assert len(result["forecasts"]) == 2
    assert result["forecasts"][0]["predictions"]["blend"] == 110.0
    assert result["forecasts"][1]["predictions"]["blend"] == 200.0
    assert all("actual" not in row and "targetValue" not in row for row in result["forecasts"])


def test_forecast_rejects_key_mismatch_duplicate_and_truth_leakage(monkeypatch):
    for mutate, expected in (
        (lambda doc: doc["forecasts"].pop(), "key_count_mismatch"),
        (lambda doc: doc["forecasts"].append(dict(doc["forecasts"][0])), "duplicate_key"),
        (lambda doc: doc["forecasts"][0].update(actual=100), "answer_leakage"),
    ):
        doc = _source()
        mutate(doc)
        raw = _raw(doc)
        _patch_tiny(monkeypatch, raw)
        with pytest.raises(ValueError, match=expected):
            blend.build_forecast(raw)


def test_score_checks_pinned_lineage_key_set_and_concordance(monkeypatch):
    source_raw = _raw(_source())
    _patch_tiny(monkeypatch, source_raw)
    forecast = blend.build_forecast(source_raw)
    manifest_raw = _raw(forecast) + b"\n"
    actuals = {"2025-09-25": 105.0, "2025-09-26": 210.0}
    direct_rows = []
    by_method = {method: [] for method in blend.METHODS}
    for source_row in forecast["forecasts"]:
        target = source_row["target"]
        pred = source_row["predictions"]
        direct_rows.append({"product": "cabbage", "target": target, "origin": source_row["origin"],
                            "horizon": 1, "actual": actuals[target],
                            "direct": pred["direct_ridge"], "persistence": pred["persistence"]})
        for method in blend.METHODS:
            by_method[method].append({"actual": actuals[target], "predictions": pred})
    cell = {"product": "cabbage", "horizon": 1,
            "metrics": {"direct_ridge": blend._loss(by_method["direct_ridge"], "direct_ridge"),
                        "calendar_sarima_111_100_7": blend._loss(by_method["calendar_sarima_111_100_7"], "calendar_sarima_111_100_7"),
                        "naive": blend._loss(by_method["persistence"], "persistence")}}
    ten_report_raw = _raw({"protocol": blend.TEN_PROTOCOL, "cellMetrics": [cell]})
    direct_raw = _raw({"pairedRows": direct_rows})
    monkeypatch.setattr(blend, "TEN_REPORT_SHA", hashlib.sha256(ten_report_raw).hexdigest())
    monkeypatch.setattr(blend, "DIRECT_REPORT_SHA", hashlib.sha256(direct_raw).hexdigest())
    result = blend.score(forecast, manifest_raw, source_raw, ten_report_raw, direct_raw)
    assert result["coverage"]["scoredPairs"] == 2
    assert result["dmTests"]["familySizePerBandwidth"] == 6
    assert result["dmTests"]["HAC9"][0]["status"] == "inference_unavailable_insufficient_pairs"

    bad_rows = [dict(row) for row in direct_rows]
    bad_rows[0]["direct"] += 1
    bad_direct = _raw({"pairedRows": bad_rows})
    monkeypatch.setattr(blend, "DIRECT_REPORT_SHA", hashlib.sha256(bad_direct).hexdigest())
    with pytest.raises(ValueError, match="actual_direct_concordance_failed"):
        blend.score(forecast, manifest_raw, source_raw, ten_report_raw, bad_direct)

    tampered = copy.deepcopy(forecast)
    tampered["forecasts"][0]["predictions"]["blend"] += 1
    content = {key: value for key, value in tampered.items() if key != "manifestSha256"}
    tampered["manifestSha256"] = blend._canonical_hash(content)
    tampered_raw = _raw(tampered) + b"\n"
    with pytest.raises(ValueError, match="manifest_blend_rule_mismatch"):
        blend.score(tampered, tampered_raw, source_raw, ten_report_raw, direct_raw)


def test_dm_uses_actual_calendar_lags_and_holm_family():
    start = dt.date(2025, 1, 1)
    diffs = [(start + dt.timedelta(days=2*i), -1.0 if i % 2 == 0 else -2.0) for i in range(40)]
    result = blend._dm(diffs, 9)
    assert result["status"] == "tested"
    assert result["n"] == 40
    mean = sum(value for _, value in diffs) / len(diffs)
    centered = {day: value - mean for day, value in diffs}
    reference_variance = sum(value * value for value in centered.values())
    for lag in range(1, 10):
        covariance = sum(value * centered.get(day-dt.timedelta(days=lag), 0.0)
                         for day, value in centered.items())
        reference_variance += 2 * (1 - lag / 10) * covariance
    reference_statistic = mean / (math.sqrt(reference_variance) / len(diffs))
    assert result["statistic"] == pytest.approx(reference_statistic)
    assert result["pValue"] == pytest.approx(math.erfc(abs(reference_statistic) / math.sqrt(2)))
    tests = [{"pValue": .01, "product": "cabbage", "horizon": 1, "comparator": "direct", "loss": "AE", "status": "tested"},
             {"pValue": .04, "product": "cabbage", "horizon": 1, "comparator": "direct", "loss": "APE", "status": "tested"}]
    blend._holm(tests)
    assert all("holmPValue" in item for item in tests)
    assert sorted(item["holmPValue"] for item in tests) == [.02, .04]
