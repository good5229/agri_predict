import datetime as dt
import hashlib
import json
import math

import pytest

from agri_data import trend_response_correction as trend


def _prices(origin, values):
    """Dense synthetic observed price and volume series ending before origin."""
    return {origin - dt.timedelta(days=age): (value, 100.0)
            for age, value in values.items()}


def test_price_features_use_strict_past_p0_and_lagged_price_anchors():
    origin = dt.date(2026, 5, 20)
    series = {"cabbage": {origin - dt.timedelta(days=age): (100.0 + age, 500.0)
                          for age in range(1, 50)}}
    got = trend._price_features(series, "cabbage", origin)
    assert got["p0Date"] == (origin - dt.timedelta(days=1)).isoformat()
    assert got["p7Date"] == (origin - dt.timedelta(days=8)).isoformat()
    assert got["p28Date"] == (origin - dt.timedelta(days=29)).isoformat()
    assert got["cutoff"] == (origin - dt.timedelta(days=1)).isoformat()
    assert got["p0Date"] < origin.isoformat()


def test_regimes_distinguish_aligned_conflicting_and_zero_momentum():
    origin = dt.date(2026, 6, 20)
    aligned = {"cabbage": {origin - dt.timedelta(days=d): (p, 100)
                            for d, p in [(1, 120), (8, 110), (29, 100)]}}
    conflict = {"cabbage": {origin - dt.timedelta(days=d): (p, 100)
                             for d, p in [(1, 100), (8, 110), (29, 90)]}}
    flat = {"cabbage": {origin - dt.timedelta(days=d): (100, 100)
                         for d in (1, 8, 29)}}
    assert trend._price_features(aligned, "cabbage", origin)["regime"] == "aligned"
    assert trend._price_features(conflict, "cabbage", origin)["regime"] == "conflicting"
    assert trend._price_features(flat, "cabbage", origin)["regime"] == "neutral"


def test_missing_or_stale_anchors_trigger_explicit_fallback():
    origin = dt.date(2026, 6, 20)
    sparse = {"cabbage": {origin - dt.timedelta(days=d): (100, 100) for d in (1, 3, 9)}}
    got = trend._price_features(sparse, "cabbage", origin)
    assert got["reason"] == "momentum_anchor_missing_or_older_than_14_days"
    assert got["p0Date"] == (origin - dt.timedelta(days=1)).isoformat()


def test_correction_scales_existing_baseline_movement_and_is_bounded():
    assert trend._scale_movement(110, 100, 1.0) == 110
    assert trend._scale_movement(110, 100, 0.75) == pytest.approx(100 * 1.1**0.75)
    assert trend._scale_movement(90, 100, 1.25) == pytest.approx(100 * 0.9**1.25)
    assert trend._scale_movement(1e308, 1e-308, 1.25) is None
    with pytest.raises(ValueError, match="frozen bounded"):
        trend._scale_movement(110, 100, 1.5)


def test_unavailable_or_overflowing_scaled_correction_falls_back_to_exact_baseline(monkeypatch):
    baseline = 1e308
    row = {"product": "cabbage", "origin": "2026-06-01", "target": "2026-06-05",
           "horizon": 4, "models": {"causal50": {"status": "issued", "prediction": baseline}}}
    features = {"reason": None, "regime": "aligned", "p0": 1e-308}
    monkeypatch.setattr(trend, "_price_features", lambda *a, **k: features)
    record = trend._make_record(row, {}, {}, scale=1.25)
    assert record["baseline"] == baseline
    assert record["candidate"] == baseline
    assert record["reason"] == "nonfinite_scaled_forecast_fallback_baseline"
    features["reason"] = "momentum_anchor_missing_or_older_than_14_days"
    missing = trend._make_record(row, {}, {}, scale=1.25)
    assert missing["candidate"] == baseline
    assert missing["reason"] == features["reason"]


def test_train_selector_uses_supported_pooled_group_and_forbids_mae_worsening(monkeypatch):
    records = []
    for h in trend.LONG_HORIZONS:
        for i in range(45):
            records.append({"origin": dt.date(2025, 10, 1) + dt.timedelta(days=i),
                            "target": dt.date(2025, 10, 1) + dt.timedelta(days=i + h),
                            "horizon": h, "baseline": 110.0,
                            "features": {"p0": 100.0}, "actual": 106.0})
    monkeypatch.setattr(trend, "_training_records",
                        lambda _manifest, _series, _actuals, crop, regime: records
                        if crop == "cabbage" and regime == "aligned" else [])
    chosen = trend._select({}, {}, {}, {})
    cabbage_aligned = next(g for g in chosen["groups"] if g["product"] == "cabbage" and g["regime"] == "aligned")
    assert cabbage_aligned["supported"] is True
    assert cabbage_aligned["scale"] == 0.75
    assert all(x["maeConstraintSatisfied"] for x in cabbage_aligned["trials"] if x["scale"] in (0.75, 1.0))
    unsupported = next(g for g in chosen["groups"] if g["product"] == "radish" and g["regime"] == "conflicting")
    assert unsupported["scale"] == 1.0
    assert unsupported["reason"] == "unsupported_group_fallback_scale_one"


def test_correction_periods_preserve_exact_target_and_origin_gates():
    conf = trend._period_rows({"evaluationForecasts": [
        {"product": "cabbage", "target": "2026-04-01", "origin": "2026-03-31", "horizon": 1, "eligible": True},
        {"product": "cabbage", "target": "2026-04-02", "origin": "2026-04-01", "horizon": 1, "eligible": True},
        {"product": "cabbage", "target": "2026-06-20", "origin": "2026-06-10", "horizon": 10, "eligible": True},
        {"product": "cabbage", "target": "2026-06-21", "origin": "2026-06-11", "horizon": 10, "eligible": True}]}, "confirmation")
    assert [r["target"] for r in conf] == ["2026-04-02", "2026-06-20"]
    final = trend._period_rows({"evaluationForecasts": [
        {"product": "cabbage", "target": "2026-07-11", "origin": "2026-07-01", "horizon": 10, "eligible": True},
        {"product": "cabbage", "target": "2026-07-11", "origin": "2026-07-10", "horizon": 1, "eligible": True},
        {"product": "cabbage", "target": "2026-09-27", "origin": "2026-09-17", "horizon": 10, "eligible": True}]}, "final")
    assert len(final) == 3


def test_confirmation_fixed_40_dm_family_and_macro_gate_no_extra_hard_rules():
    rows = []
    actuals = {}
    start, end = trend.CONF_START, trend.CONF_END
    period_days = (end - start).days + 1
    for crop in trend.ALL_PRODUCTS:
        for h in trend.HORIZONS:
            for i in range(h, period_days):
                target = start + dt.timedelta(days=i)
                actual = 100.0 + i / 10
                actuals[(crop, target)] = actual
                baseline = actual + 10.0
                candidate = (baseline if h <= 3 or crop != "cabbage" else actual + 5.0)
                rows.append({"product": crop, "target": target.isoformat(), "horizon": h,
                             "model": {"status": "issued", "prediction": candidate},
                             "baselinePrediction": baseline})
    scored = trend._score_period(rows, actuals, trend.PRODUCTS)
    assert [r["familySize"] for r in scored["dmSummaryByBandwidth"]] == [40, 40]
    assert all(r["pValue"] == 1.0 for r in scored["dm"] if r["horizon"] <= 3)
    decisions = {r["product"]: r for r in trend._gate(scored)}
    assert decisions["cabbage"]["decision"] == "freeze_for_final_period"
    assert decisions["radish"]["decision"] == "stop_candidate_work"


def test_tampered_correction_manifest_is_rejected():
    doc = {"schemaVersion": 1, "phase": "correction_forecast_only", "targetValuesAttached": False,
           "manifestSha256": "bad", "confirmationForecasts": [], "finalForecasts": []}
    with pytest.raises(ValueError, match="digest mismatch"):
        trend._validate_correction_manifest(doc)


def test_pinned_baseline_identity_checks_file_digest_and_schema_without_refit(monkeypatch):
    raw_manifest = trend.MANIFEST.read_bytes()
    raw_prices = trend.PRICES.read_bytes()
    embedded = json.loads(raw_manifest)["manifestSha256"]
    assert hashlib.sha256(raw_manifest).hexdigest() == trend.MANIFEST_SHA256
    assert embedded != trend.MANIFEST_SHA256
    monkeypatch.setattr(trend.forecasts, "_fit_ridge", lambda *a, **k: pytest.fail("identity check refit Ridge"))
    monkeypatch.setattr(trend.forecasts, "_fit_sarima", lambda *a, **k: pytest.fail("identity check refit SARIMA"))
    manifest, prices = trend._input_identity(raw_manifest, raw_prices)
    assert manifest["manifestSha256"] == embedded
    assert isinstance(prices, dict)


def _synthetic_diagnostic_inputs():
    first, last = dt.date(2025, 7, 1), dt.date(2026, 3, 20)
    series = {}
    actuals = {}
    for product_index, product in enumerate(trend.PRODUCTS):
        product_prices = {}
        day, i = first, 0
        while day <= last:
            value = 100.0 + 30 * product_index + 0.08 * i + 0.0007 * i * i
            product_prices[day] = (value, 1000.0)
            actuals[(product, day)] = value
            day += dt.timedelta(days=1); i += 1
        series[product] = product_prices
    rows = []
    day = trend.TRAIN_START
    while day <= trend.TRAIN_END:
        for product in trend.PRODUCTS:
            actual = actuals[(product, day)]
            for horizon in trend.LONG_HORIZONS:
                rows.append({"product": product, "target": day.isoformat(),
                    "origin": (day - dt.timedelta(days=horizon)).isoformat(),
                    "horizon": horizon, "eligible": True,
                    "models": {"causal50": {"status": "issued", "prediction": actual * 1.04}}})
        day += dt.timedelta(days=1)
    return {"confirmationForecasts": rows, "evaluationForecasts": []}, series, actuals


def test_diagnostics_keep_product_block_regime_horizon_and_month_slices():
    manifest, series, actuals = _synthetic_diagnostic_inputs()
    report = trend._diagnose(manifest, series, actuals)
    assert len(report) == len(trend.PRODUCTS) * 2 * 2
    assert {r["product"] for r in report} == set(trend.PRODUCTS)
    assert {r["trainBlock"] for r in report} == {"train_2025q4", "train_2026jan_mar20"}
    assert {r["regime"] for r in report} == {"aligned", "conflicting"}
    for item in report:
        assert len(item["byHorizon"]) == 7
        assert all(row["horizon"] in trend.LONG_HORIZONS for row in item["byHorizon"])
        if item["regime"] == "aligned":
            assert item["n"] > 0
            assert item["distinctTargets"] > 0
            assert item["byMonth"]
            assert all("n" in values and "baselineMae" in values
                       for values in item["byMonth"].values())


def _selection_document():
    unsigned = {"schemaVersion": 1, "groups": []}
    return {**unsigned, "selectionSha256": trend._sha(trend._canonical(unsigned))}


def test_initial_manifest_is_confirmation_only_and_excludes_pepper(monkeypatch):
    target = dt.date(2026, 4, 2)
    products = (*trend.ALL_PRODUCTS, "pepper")
    base_rows = [{"product": product, "target": target.isoformat(),
                  "origin": (target - dt.timedelta(days=1)).isoformat(), "horizon": 1,
                  "eligible": True,
                  "models": {"causal50": {"status": "issued", "prediction": 100.0 + i}}}
                 for i, product in enumerate(products)]
    manifest = {"evaluationForecasts": base_rows}
    prices = {}
    day0 = dt.date(2025, 12, 1)
    for i, product in enumerate(trend.ALL_PRODUCTS):
        prices[product] = {day0 + dt.timedelta(days=d): (100.0 + i * 20 + d * .1, 500.0)
                           for d in range((dt.date(2026, 3, 31) - day0).days + 1)}
    selection = _selection_document()
    monkeypatch.setattr(trend, "_input_identity", lambda *_: (manifest, {}))
    monkeypatch.setattr(trend, "diagnose_and_select", lambda *_: ({}, selection))
    monkeypatch.setattr(trend.direct_horizon, "_observations", lambda _prices: prices)
    correction = trend.build_correction_manifest(b"synthetic-baseline", b"synthetic-prices", selection)
    assert correction["phase"] == "correction_forecast_only"
    assert correction["targetValuesAttached"] is False
    assert "confirmationForecasts" in correction
    assert "finalForecasts" not in correction
    assert {row["product"] for row in correction["confirmationForecasts"]} == set(trend.ALL_PRODUCTS)
    assert "pepper" not in {row["product"] for row in correction["confirmationForecasts"]}
    assert len(correction["confirmationForecasts"]) == 4
    # h1 is unchanged, as are the other two in-scope crops.
    assert all(row["model"]["prediction"] == row["baselinePrediction"] for row in correction["confirmationForecasts"])


def test_confirmation_manifest_validator_requires_exact_four_crop_grid(monkeypatch):
    day = dt.date(2026, 4, 2)
    monkeypatch.setattr(trend, "CONF_START", day - dt.timedelta(days=1))
    monkeypatch.setattr(trend, "CONF_END", day)
    monkeypatch.setattr(trend, "HORIZONS", (1,))
    rows = [{"product": crop, "target": day.isoformat(), "origin": (day-dt.timedelta(days=1)).isoformat(),
             "horizon": 1, "eligible": True,
             "model": {"status": "issued", "prediction": 100.0}, "baselinePrediction": 100.0,
             "scale": 1.0, "correctionApplied": False, "fallbackReason": "short_horizon_unchanged"}
            for crop in trend.ALL_PRODUCTS]
    unsigned = {"phase": "correction_forecast_only", "targetValuesAttached": False,
                "confirmationForecasts": rows}
    doc = {**unsigned, "manifestSha256": trend._sha(trend._canonical(unsigned))}
    assert trend._validate_correction_manifest(doc) is None
    forged = {**doc, "confirmationForecasts": [*rows, {**rows[0], "product": "pepper"}]}
    forged_unsigned = {key: value for key, value in forged.items() if key != "manifestSha256"}
    forged["manifestSha256"] = trend._sha(trend._canonical(forged_unsigned))
    with pytest.raises(ValueError, match="invalid correction forecast grid"):
        trend._validate_correction_manifest(forged)


def test_zero_confirmation_pass_stops_before_final_generation_and_forged_decision_rejected(monkeypatch):
    selection = _selection_document()
    expected_report = {"decisions": [{"product": crop, "decision": "stop_candidate_work"}
                                     for crop in trend.PRODUCTS],
                       "reportSha256": "synthetic-report-digest"}
    monkeypatch.setattr(trend, "diagnose_and_select", lambda *_: ({}, selection))
    monkeypatch.setattr(trend, "_input_identity", lambda *_: ({}, {}))
    monkeypatch.setattr(trend, "build_correction_manifest", lambda *_: {"phase": "correction_forecast_only"})
    monkeypatch.setattr(trend, "score_confirmation", lambda *_: expected_report)
    original_period_rows = trend._period_rows
    periods = []
    def track_period(manifest, period):
        periods.append(period)
        assert period != "final", "final-period grid generated before any confirmation crop passed"
        return original_period_rows(manifest, period)
    monkeypatch.setattr(trend, "_period_rows", track_period)
    with pytest.raises(ValueError, match="no crop"):
        trend.build_final_manifest(b"synthetic-baseline", b"synthetic-prices", selection, expected_report)
    assert "final" not in periods

    forged = {"decisions": [{"product": crop,
                              "decision": "freeze_for_final_period" if crop == trend.PRODUCTS[0] else "stop_candidate_work"}
                             for crop in trend.PRODUCTS]}
    forged["reportSha256"] = trend._sha(trend._canonical(forged))
    with pytest.raises(ValueError, match="confirmation report replay mismatch"):
        trend.build_final_manifest(b"synthetic-baseline", b"synthetic-prices", selection, forged)
    assert "final" not in periods


def test_final_period_score_keeps_four_crop_40_cells_and_fixed40_dm_families():
    rows, actuals = [], {}
    start = dt.date(2026, 7, 11)
    for product_index, product in enumerate(trend.ALL_PRODUCTS):
        for horizon in trend.HORIZONS:
            for offset in range(30):
                target = start + dt.timedelta(days=offset)
                actual = 100.0 + product_index * 50 + offset * .75
                if offset % 6:
                    actuals[(product, target)] = actual
                rows.append({"product": product, "target": target.isoformat(), "horizon": horizon,
                    "model": {"status": "issued", "prediction": actual * 1.03},
                    "baselinePrediction": actual * 1.2})
    scored = trend._score_period(rows, actuals, trend.PRODUCTS, frozen_crops=("cabbage",))
    assert len(scored["metrics"]) == 40
    assert len(scored["macro"]) == 4
    assert {r["product"] for r in scored["macro"]} == set(trend.ALL_PRODUCTS)
    assert len(scored["coverage"]) == 40
    sample = scored["coverage"]["cabbage/4"]
    assert sample["planned"] == sample["baselineIssued"] == sample["candidateIssued"] == 30
    assert sample["commonPairs"] == 25
    for bandwidth in trend.BANDWIDTHS:
        summary = next(row for row in scored["dmSummaryByBandwidth"] if row["bandwidth"] == bandwidth)
        assert summary["familySize"] == 40
        family = [r for r in scored["dm"] if r["bandwidth"] == bandwidth]
        assert len(family) == 4 * 10 * 2
        radish = [r for r in family if r["product"] == "radish"]
        assert len(radish) == 20
        assert all(r["pValue"] == 1.0 and r["reason"] == "crop_not_selected_by_confirmation"
                   for r in radish)
        onion_garlic = [r for r in family if r["product"] in {"onion", "garlic"}]
        assert all(r["pValue"] == 1.0 and r["reason"] == "crop_not_in_correction_family"
                   for r in onion_garlic)
    assert "pepper" not in trend.ALL_PRODUCTS
