import datetime as dt
import hashlib
import json

import pytest

from agri_data import preserved_feature_forecasts as mod
from agri_data import direct_horizon, full_history_feature_library as library, modeling


def _synthetic_feature_inputs(days=180):
    start = dt.date(2024, 1, 1)
    all_days = [start + dt.timedelta(days=i) for i in range(days)]
    series = {"cabbage": {d: (100.0 + i * 0.2, 1000.0 + i * 3.0) for i, d in enumerate(all_days)}}
    records = {"cabbage": {d: {"price": p, "volume": v, "rowCount": 10 + i % 4,
                                "markets": frozenset(library.MARKET_CODES)}
                           for i, (d, (p, v)) in enumerate(series["cabbage"].items())}}
    prices = {"cabbage": {d: pair[0] for d, pair in series["cabbage"].items()}}
    ordered = {"cabbage": all_days}
    return series, records, prices, ordered


def test_new_vector_is_exact_40_feature_prefix_plus_66_named_additions():
    series, records, prices, ordered = _synthetic_feature_inputs()
    target = ordered["cabbage"][-1] + dt.timedelta(days=5)
    horizon = 3
    result = mod._feature_pair("cabbage", target, horizon, series, records,
                               {"cabbage": {}}, ordered, prices)
    assert result is not None
    assert len(result["original40"]) == 40
    assert len(result["preserved106"]) == 106
    assert result["preserved106"][:40] == result["original40"]
    assert len([n for n in library.FEATURE_NAMES if n not in mod.OMITTED_LIBRARY_FEATURES]) == 66
    assert result["p0Date"] == dt.date.fromisoformat(result["featureAudit"]["p0Date"])
    expected = modeling._feature_row(series["cabbage"], target - dt.timedelta(days=horizon + 1), target, horizon, {})
    assert expected[0] == result["p0Date"]
    full_expanded, _, _ = library.features(records["cabbage"], target, horizon,
                                           ordered["cabbage"], prices["cabbage"])
    expected_additions = [value for name, value in zip(library.FEATURE_NAMES, full_expanded)
                          if name not in mod.OMITTED_LIBRARY_FEATURES]
    assert result["preserved106"][40:] == expected_additions


def test_candidate_feature_failure_or_p0_mismatch_fails_closed(monkeypatch):
    series, records, prices, ordered = _synthetic_feature_inputs()
    target = ordered["cabbage"][-1] + dt.timedelta(days=5)
    monkeypatch.setattr(library, "features", lambda *args: (None, None, {"reason": "synthetic_missing"}))
    with pytest.raises(ValueError, match="expanded feature row missing"):
        mod._feature_pair("cabbage", target, 3, series, records, {"cabbage": {}}, ordered, prices)


def test_candidate_p0_value_and_date_must_match_direct_feature_row(monkeypatch):
    series, records, prices, ordered = _synthetic_feature_inputs()
    target = ordered["cabbage"][-1] + dt.timedelta(days=5)
    original = library.features

    def wrong_p0(*args, **kwargs):
        vector, p0, audit = original(*args, **kwargs)
        return vector, p0 + 1.0, audit

    monkeypatch.setattr(library, "features", wrong_p0)
    with pytest.raises(ValueError, match="p0 mismatch"):
        mod._feature_pair("cabbage", target, 3, series, records, {"cabbage": {}}, ordered, prices)


def test_phase_price_trim_excludes_future_answers_before_value_parsing():
    end = dt.date(2025, 12, 31)

    class MustNotParse:
        def __float__(self):
            raise AssertionError("future actual was parsed")

    products = []
    for product in mod.PRODUCTS:
        products.append({"id": product, "observations": [
            {"date": "2025-12-31", "actual": "12", "volumeKg": "2"},
            {"date": "2026-01-01", "actual": MustNotParse(), "volumeKg": MustNotParse()},
            {"date": "bad", "actual": MustNotParse(), "volumeKg": MustNotParse()},
        ]})
    trimmed, audit = mod._trim_price_snapshot({"schemaVersion": 1, "products": products}, end)
    assert all([row["date"] for row in product["observations"]] == ["2025-12-31"]
               for product in trimmed["products"])
    assert all(item == {"includedThroughPhaseEnd": 1, "skippedLaterDates": 1,
                        "skippedInvalidDateRows": 1} for item in audit.values())


def test_monthly_training_support_is_exact_and_mismatch_stops():
    targets = [dt.date(2025, 1, 1) + dt.timedelta(days=i) for i in range(6)]
    audit = mod._fit_audit(targets, dt.date(2025, 1, 5))
    assert audit["count"] == 4
    assert audit["firstTarget"] == "2025-01-01"
    assert audit["lastTarget"] == "2025-01-04"
    expected = {k: audit[k] for k in ("count", "firstTarget", "lastTarget", "fitOrigin")}
    mod._assert_training_support(audit, expected, "cabbage", 2, (2025, 1))
    expected["count"] += 1
    with pytest.raises(ValueError, match="training support mismatch"):
        mod._assert_training_support(audit, expected, "cabbage", 2, (2025, 1))


def test_frozen_direct_replay_requires_numeric_baseline_match():
    key = ("cabbage", dt.date(2025, 10, 2), 1)
    actual = {"status": "issued", "prediction": 123.0}
    expected = {"status": "issued", "prediction": 123.0 + 1e-10}
    mod._assert_prediction_matches(actual, expected, "causalDirect", key)
    expected["prediction"] = 124.0
    with pytest.raises(ValueError, match="differs"):
        mod._assert_prediction_matches(actual, expected, "causalDirect", key)


def test_unselected_policy_uses_frozen_fixed50_and_selected_policy_uses_blend():
    fixed = {"status": "issued", "prediction": 90.0}
    candidate = {"status": "issued", "prediction": 100.0, "provenance": {"fallback": False}}
    assert mod._policy_forecast(False, mod._not_selected(), fixed)["prediction"] == 90.0
    assert mod._policy_forecast(False, mod._not_selected(), fixed)["method"] == "fixed50"
    assert mod._policy_forecast(True, candidate, fixed)["prediction"] == 100.0
    missing = {"status": "not_issued", "prediction": None}
    fallback = mod._policy_forecast(True, missing, fixed)
    assert fallback["prediction"] == 90.0 and fallback["method"] == "fixed50"


def test_selected_evaluation_rows_fall_back_for_pre_phase_origins():
    key = ("cabbage", dt.date(2026, 1, 1), 1)
    baseline_models = {name: {"status": "issued", "prediction": 10.0}
                       for name in ("causalDirect", "causalSarima", "causal50", "fixed50")}
    row = mod._unissued_grid_row(key, "evaluation", {"cabbage"}, "origin_before_phase_start",
                                {"models": baseline_models})
    assert row["eligible"] is False
    assert row["models"]["preserved106Blend"]["status"] == "not_issued"
    assert row["policyForecast"]["method"] == "fixed50"
    assert row["policyForecast"]["prediction"] == 10.0


def test_unselected_evaluation_row_inherits_all_baselines_without_fit():
    key = ("radish", dt.date(2026, 1, 10), 3)
    baseline_models = {name: {"status": "issued", "prediction": float(i + 1)}
                       for i, name in enumerate(("causalDirect", "causalSarima", "causal50", "fixed50"))}
    row = mod._unselected_policy_row(key, {"models": baseline_models})
    assert row["eligible"] is True
    assert row["selectedCrop"] is False
    assert row["models"]["causalDirect"] == baseline_models["causalDirect"]
    assert row["models"]["causalSarima"] == baseline_models["causalSarima"]
    assert row["models"]["causal50"] == baseline_models["causal50"]
    assert row["models"]["fixed50"] == baseline_models["fixed50"]
    assert row["models"]["preserved106"]["status"] == "not_selected"
    assert row["policyForecast"]["method"] == "fixed50"
    assert row["policyForecast"]["prediction"] == baseline_models["fixed50"]["prediction"]


def test_month_grouping_initializes_ineligible_rows_even_with_eligible_months():
    start = dt.date(2025, 10, 1)
    keys = [("cabbage", dt.date(2025, 10, 1), 1),
            ("cabbage", dt.date(2025, 10, 2), 1)]
    no_model = {"status": "not_issued", "reason": "synthetic", "prediction": None,
                "training": None, "provenance": {}}
    baseline = {key: {"models": {name: dict(no_model) for name in
                                 ("causalDirect", "causalSarima", "causal50", "fixed50")}}
                for key in keys}
    output = {}
    months = mod._group_monthly_keys(keys, start, "confirmation", set(mod.PRODUCTS), baseline, output)
    assert list(months) == [(2025, 10)]
    assert keys[0] in output
    assert keys[1] not in output


@pytest.mark.parametrize("model,reason", [
    (type("Raises", (), {"predict": lambda self, _: (_ for _ in ()).throw(RuntimeError())})(),
     "prediction_failed:RuntimeError"),
    (type("Overflow", (), {"predict": lambda self, _: [10000.0]})(), "prediction_failed:OverflowError"),
])
def test_prediction_failures_become_not_issued_rows_and_allow_policy_fallback(model, reason):
    candidate = mod._predict_row(model, [1.0], 100.0, {"count": 100}, {"featureCount": 106})
    assert candidate["status"] == "not_issued"
    assert candidate["reason"] == reason
    fixed = {"status": "issued", "prediction": 95.0}
    assert mod._policy_forecast(True, candidate, fixed)["prediction"] == 95.0


def test_evaluation_with_no_passing_confirmation_crop_is_stopped_before_build():
    with pytest.raises(ValueError, match="no crop passed confirmation"):
        mod.build({}, {}, "evaluation", selected_crops=[], decision_hashes={})


def test_decision_digest_binds_confirmation_manifest_and_eligible_crop_list(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "PHASES", {"confirmation": (dt.date(2025, 10, 1), dt.date(2025, 10, 1)),
                                         "evaluation": (dt.date(2026, 1, 1), dt.date(2026, 1, 1))})
    monkeypatch.setattr(mod, "_expected_keys", lambda phase: [("cabbage", dt.date(2025, 10, 1), 1)])
    baseline = {"status": "not_issued", "reason": "synthetic", "prediction": None, "training": None, "provenance": {}}
    forecast = {"product": "cabbage", "target": "2025-10-01", "origin": "2025-09-30", "horizon": 1,
                "eligible": False, "selectedCrop": None,
                "models": {m: dict(baseline) for m in ("causalDirect", "causalSarima", "causal50", "fixed50", "preserved106", "preserved106Blend")},
                "policyForecast": {"method": "preserved106Blend", "status": "not_issued", "prediction": None, "reason": "synthetic", "provenance": {}}}
    confirmation = {"schemaVersion": 1, "phase": "forecast_only", "phaseName": "confirmation", "targetValuesAttached": False,
                    "plannedCount": 1, "forecasts": [forecast]}
    confirmation["manifestSha256"] = hashlib.sha256(mod._canonical(confirmation).rstrip(b"\n")).hexdigest()
    confirmation_path = tmp_path / "confirmation.json"
    confirmation_bytes = mod._canonical(confirmation)
    confirmation_path.write_bytes(confirmation_bytes)
    decision = {"schemaVersion": 1, "kind": "preserved_feature_confirmation_gate",
                "confirmationForecastManifestSha256": hashlib.sha256(confirmation_bytes).hexdigest(),
                "confirmationInternalManifestSha256": confirmation["manifestSha256"],
                "confirmationEvaluationSha256": "a" * 64, "scorerSha256": "b" * 64,
                "selectedCrops": ["cabbage"],
                "cropDecisions": [{"product": p, "eligible": p == "cabbage", "reasons": []} for p in mod.PRODUCTS]}
    decision["decisionSha256"] = mod._unsigned_sha(decision, "decisionSha256")
    selected, hashes = mod._decision_fields(decision, confirmation_path)
    assert selected == ["cabbage"]
    assert hashes["confirmationInternalManifest"] == confirmation["manifestSha256"]
    decision["selectedCrops"] = []
    decision["decisionSha256"] = mod._unsigned_sha(decision, "decisionSha256")
    with pytest.raises(ValueError, match="disagrees"):
        mod._decision_fields(decision, confirmation_path)


def test_manifest_digest_uses_same_newline_free_unsigned_bytes(monkeypatch):
    date = dt.date(2025, 10, 1)
    monkeypatch.setattr(mod, "PHASES", {"confirmation": (date, date)})
    monkeypatch.setattr(mod, "_expected_keys", lambda phase: [("cabbage", date, 1)])
    model = {"status": "not_issued", "reason": "synthetic", "prediction": None, "training": None, "provenance": {}}
    row = {"product": "cabbage", "target": date.isoformat(), "origin": "2025-09-30", "horizon": 1,
           "models": {m: dict(model) for m in ("causalDirect", "causalSarima", "causal50", "fixed50", "preserved106", "preserved106Blend")}}
    doc = {"phase": "forecast_only", "phaseName": "confirmation", "targetValuesAttached": False,
           "plannedCount": 1, "forecasts": [row]}
    doc["manifestSha256"] = mod._unsigned_sha(doc, "manifestSha256")
    mod._validate_manifest(doc, "confirmation")
    doc["plannedCount"] = 2
    with pytest.raises(ValueError, match="digest"):
        mod._validate_manifest(doc, "confirmation")
