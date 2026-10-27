import datetime as dt
import json
import math
from pathlib import Path
from array import array

import pytest

from agri_data import path_mlp_forecasts as mod


def _synthetic_domain(n=240):
    origin = dt.date(2024, 9, 1)
    series = {}
    records = {}
    for i in range(n):
        day = origin - dt.timedelta(days=n - i)
        price = 100.0 + 0.7 * i + (i % 11) * 0.4
        volume = 1000.0 + (i % 7) * 19.0
        series[day] = (price, volume)
        records[day] = {"price": price, "volume": volume,
                        "rowCount": 8 + i % 5,
                        "markets": frozenset(mod.library.MARKET_CODES)}
    return origin, {"cabbage": series}, {"cabbage": records}


def test_feature_pair_has_106_summaries_and_91_calendar_values_with_mask():
    origin, series, records = _synthetic_domain()
    ordered = {"cabbage": sorted(series["cabbage"])}
    prices = {"cabbage": {d: r["price"] for d, r in records["cabbage"].items()}}
    sample = mod._feature_pair("cabbage", origin + dt.timedelta(days=1), 1,
                               series, records, {"cabbage": {}}, ordered, prices)
    assert sample is not None
    assert len(sample["preserved106"]) == 106
    assert len(sample["pathValues"]) == len(sample["pathMask"]) == len(sample["scrambledValues"]) == 91
    assert sum(sample["pathMask"]) == 91
    assert sample["p0Date"] == origin - dt.timedelta(days=1)
    assert sample["p0AgeDays"] == 1
    assert max((abs(math.log(series["cabbage"][d][0] / sample["p0"]) - value)
                for d, value in zip(sorted(series["cabbage"])[-91:], sample["pathValues"])), default=0.) < 1e-12


def test_feature_pair_never_passes_later_rows_into_feature_helpers():
    class MustNotRead:
        def __float__(self):
            raise AssertionError("future source value was read by a feature helper")

    origin, series, records = _synthetic_domain()
    later = origin + dt.timedelta(days=1)
    series["cabbage"][later] = (MustNotRead(), MustNotRead())
    records["cabbage"][later] = {"price": MustNotRead(), "volume": MustNotRead(),
                                  "rowCount": MustNotRead(), "markets": MustNotRead()}
    ordered = {"cabbage": sorted(series["cabbage"])}
    prices = {"cabbage": {d: row["price"] for d, row in records["cabbage"].items()}}
    sample = mod._feature_pair("cabbage", origin + dt.timedelta(days=1), 1,
                               series, records, {"cabbage": {}}, ordered, prices)
    assert sample is not None


def test_scramble_is_deterministic_for_crop_origin_and_preserves_observation_membership():
    origin, series, _ = _synthetic_domain()
    days = sorted(series["cabbage"])
    p0_day = days[-1]
    path_days = [origin - dt.timedelta(days=91-i) for i in range(91)]
    a, audit_a = mod._scramble_path("cabbage", origin, p0_day, path_days, series["cabbage"], seed=0)
    b, audit_b = mod._scramble_path("cabbage", origin, p0_day, path_days, series["cabbage"], seed=0)
    assert a == b
    assert audit_a == audit_b
    assert set(a).issubset(set(path_days))
    assert len(a) == audit_a["permutableObservedDates"]
    assert audit_a["changedPriceDates"] <= len(a)


def test_equal_cell_weights_give_each_supported_cell_equal_total_mass_and_mean_one():
    weights, audit = mod._equal_cell_weights({("cabbage", 1): 100,
                                               ("radish", 1): 200,
                                               ("onion", 1): 89})
    assert weights == {("cabbage", 1): 1.5, ("radish", 1): .75}
    assert 100 * weights[("cabbage", 1)] == 200 * weights[("radish", 1)]
    assert audit["supportedCells"] == 2
    assert audit["trainingRows"] == 300
    assert audit["meanRowWeight"] == 1


def test_arm_packing_has_exactly_293_inputs_and_summary_blanks_path_and_mask_after_scaling():
    scaled = [float(i) for i in range(mod.CONTINUOUS_DIM)]
    mask = [1] * mod.PATH_DAYS
    scrambled = [float(i + 500) for i in range(mod.CONTINUOUS_DIM)]
    ordered = mod._pack_features(scaled, mask, "onion", "ordered")
    summary = mod._pack_features(scaled, mask, "onion", "summary")
    shuffled = mod._pack_features(scaled, mask, "onion", "scrambled", scrambled)
    assert all(len(row) == mod.FEATURE_DIM for row in (ordered, summary, shuffled))
    assert ordered[:106] == summary[:106] == shuffled[:106] == scaled[:106]
    assert ordered[106:197] == scaled[106:197]
    assert summary[106:288] == [0.] * 182
    assert shuffled[106:197] == scrambled[106:197]
    assert shuffled[197:288] == [1.] * 91
    assert ordered[288:] == summary[288:] == shuffled[288:] == [0., 0., 1., 0., 0.]
    with pytest.raises(ValueError):
        mod._pack_features(scaled[:-1], mask, "onion", "ordered")


def test_month_first_fit_origins_follow_first_eligible_origin_in_each_phase_month():
    screen = mod._monthly_fit_origins("screen")
    evaluation = mod._monthly_fit_origins("evaluation")
    assert len(screen) == 12
    assert list(screen.items())[0] == ((2024, 9), dt.date(2024, 9, 15))
    assert screen[(2024, 10)] == dt.date(2024, 10, 1)
    assert len(evaluation) == 9
    assert evaluation[(2026, 1)] == dt.date(2026, 1, 1)


def test_snapshot_trimmer_does_not_read_values_after_phase_end():
    class MustNotRead:
        def __float__(self):
            raise AssertionError("later phase values were accessed")

    raw = {"schemaVersion": 1, "products": [
        {"id": crop, "observations": [
            {"date": "2025-08-31", "actual": "100", "volumeKg": "10"},
            {"date": "2025-09-01", "actual": MustNotRead(), "volumeKg": MustNotRead()},
        ]} for crop in mod.PRODUCTS
    ]}
    trimmed, audit = mod._trim_prices(raw, dt.date(2025, 8, 31))
    assert all(len(product["observations"]) == 1 for product in trimmed["products"])
    assert all(value["skippedLaterDates"] == 1 for value in audit.values())


def test_feature_helper_caches_are_bounded_across_historical_cutoffs():
    origin, series, records = _synthetic_domain(n=300)
    for offset in range(31):
        day = origin + dt.timedelta(days=offset)
        series["cabbage"][day] = (500.0 + offset, 900.0 + offset)
        records["cabbage"][day] = {"price": 500.0 + offset, "volume": 900.0 + offset,
                                   "rowCount": 8, "markets": frozenset(mod.library.MARKET_CODES)}
    ordered = {"cabbage": sorted(series["cabbage"])}
    prices = {"cabbage": {day: row["price"] for day, row in records["cabbage"].items()}}
    caches = {"cabbage": {}}
    for offset in range(1, 31):
        target = origin + dt.timedelta(days=offset + 1)
        sample = mod._feature_pair("cabbage", target, 1, series, records, caches,
                                   ordered, prices)
        assert sample is not None
        assert len(caches["cabbage"]) <= mod.MAX_CACHED_CUTOFFS_PER_CROP
    assert len(caches["cabbage"]) == mod.MAX_CACHED_CUTOFFS_PER_CROP


def test_compact_build_cache_preserves_uncached_sample_values_bitwise():
    origin, series, records = _synthetic_domain()
    ordered = {"cabbage": sorted(series["cabbage"])}
    prices = {"cabbage": {day: row["price"] for day, row in records["cabbage"].items()}}
    target = origin + dt.timedelta(days=1)
    key = ("cabbage", target, 1)
    uncached = mod._prepare_sample("cabbage", target, 1, series, records,
                                   {"cabbage": {}}, ordered, prices)
    cache = {}
    cached = mod._cached_feature_sample(cache, key, lambda: mod._prepare_sample(
        "cabbage", target, 1, series, records, {"cabbage": {}}, ordered, prices))
    assert uncached is not None and cached is not None
    for field in ("preserved106", "pathValues", "scrambledValues"):
        assert array("d", uncached[field]).tobytes() == cached[field].tobytes()
    assert list(cached["pathMask"]) == uncached["pathMask"]
    for field in ("p0", "p0Date", "p0AgeDays", "permutation"):
        assert cached[field] == uncached[field]
    assert mod._cached_feature_sample(cache, key, lambda: pytest.fail("cache miss for exact key")) is cached


def test_compact_build_cache_keys_target_horizon_and_unavailable_rows_exactly():
    cache = {}
    calls = []

    def available():
        calls.append("available")
        return {"preserved106": [1.0], "pathValues": [2.0], "pathMask": [1],
                "scrambledValues": [3.0], "p0": 4.0, "p0Date": dt.date(2024, 1, 1),
                "p0AgeDays": 1, "permutation": {"identityPermutation": True}}

    day = dt.date(2024, 2, 1)
    first = ("cabbage", day, 1)
    second = ("cabbage", day, 2)
    one = mod._cached_feature_sample(cache, first, available)
    two = mod._cached_feature_sample(cache, second, available)
    assert len(calls) == 2
    assert one is not two
    miss_key = ("radish", day, 1)
    assert mod._cached_feature_sample(cache, miss_key, lambda: None) is None
    assert mod._cached_feature_sample(cache, miss_key, lambda: pytest.fail("unavailable row was recomputed")) is None


def test_generator_decision_digest_matches_scorer_signed_newline_contract():
    from agri_data import path_mlp_score as scorer

    screen_manifest = {"phaseName": "screen"}
    screen_report = {"selection": {crop: "retain_fixed50" for crop in mod.PRODUCTS}}
    manifest_bytes = scorer._canonical(screen_manifest)
    report_bytes = scorer._canonical(screen_report)
    decision_bytes = scorer._canonical(scorer._decision_for_validated_screen(
        screen_manifest, screen_report, screen_manifest_bytes=manifest_bytes,
        screen_report_bytes=report_bytes))
    selections, hashes = mod._validate_decision(decision_bytes, manifest_bytes, report_bytes)
    assert selections == screen_report["selection"]
    assert hashes["screenDecision"] == json.loads(decision_bytes)["decisionSha256"]


def test_evaluation_replays_screen_against_current_scorer_before_accepting_binding(monkeypatch,
                                                                                   tmp_path):
    from agri_data import path_mlp_score as scorer

    screen_components = tmp_path / "screen-components.json"
    screen_components.write_bytes(b"synthetic pinned components")
    monkeypatch.setattr(mod, "SCREEN_COMPONENTS_PATH", screen_components)
    manifest = b"manifest"
    report = b"report"
    decision = b"decision"
    selections = {crop: "retain_fixed50" for crop in mod.PRODUCTS}
    hashes = {"screenScorer": mod._sha_path(Path(scorer.__file__)),
              "screenManifest": mod._sha_bytes(manifest), "screenReport": mod._sha_bytes(report),
              "screenDecision": mod._sha_bytes(decision)}
    captured = {}

    def replay(*args):
        captured["args"] = args
        return {"selection": selections, "manifestSha256": hashes["screenManifest"],
                "screenReportSha256": hashes["screenReport"],
                "decisionSha256": hashes["screenDecision"]}

    monkeypatch.setattr(scorer, "validate_screen_artifacts", replay)
    mod._replay_screen_for_evaluation(b"prices", decision, manifest, report,
                                      selections, hashes)
    assert captured["args"] == (manifest, b"prices", screen_components.read_bytes(), report, decision)


def test_small_synthetic_build_serialization_round_trips_and_replays(monkeypatch, tmp_path):
    target = dt.date(2024, 9, 2)
    origin = target - dt.timedelta(days=1)
    monkeypatch.setattr(mod, "PHASES", {"screen": (target, target, origin)})
    monkeypatch.setattr(mod, "_phase_grid", lambda phase: [("cabbage", target, 1)])
    monkeypatch.setattr(mod, "_monthly_fit_origins", lambda phase: {(origin.year, origin.month): origin})
    monkeypatch.setattr(mod, "_verify_input_hashes", lambda *args: {"synthetic": "input"})
    monkeypatch.setattr(mod, "_check_environment", lambda: {"synthetic": "environment"})
    monkeypatch.setattr(mod, "_trim_prices", lambda raw, end: (raw, {"synthetic": {"rows": 0}}))
    monkeypatch.setattr(mod, "_source_rows", lambda raw: (
        {crop: {} for crop in mod.PRODUCTS}, {crop: {} for crop in mod.PRODUCTS}))
    monkeypatch.setattr(mod, "_component_rows", lambda raw, phase: {
        ("cabbage", target, 1): {"direct": 100.0, "sarima": 100.0,
                                 "fixed50": 100.0, "sarimaFallback": False}})

    def synthetic_fit(phase, fit_origin, active_arms, *args):
        key = ("cabbage", target, 1)
        forecasts = {arm: mod._not_issued("synthetic_empty_model") for arm in active_arms}
        audit = {"fitOrigin": fit_origin.isoformat(), "trainingRows": 0,
                 "supportedCells": 0,
                 "fitDiagnostics": {arm: {"status": "not_fit"} for arm in active_arms}}
        return {key: forecasts}, audit, {}

    monkeypatch.setattr(mod, "_fit_month", synthetic_fit)
    raw_prices = b'{"schemaVersion":1,"products":[]}'
    raw_components = b'{"synthetic":true}'
    doc = mod.build(raw_prices, raw_components, "screen")
    assert isinstance(doc["protocol"]["activeArms"], list)
    path = tmp_path / "synthetic-screen-manifest.json"
    mod._write_exclusive(path, doc)
    saved = path.read_bytes()
    parsed = json.loads(saved)
    assert parsed == doc
    replayed = mod.validate_manifest_bytes(saved, raw_prices, raw_components, "screen")
    assert replayed == parsed
    tampered = dict(parsed)
    tampered["protocol"] = dict(parsed["protocol"])
    tampered["protocol"]["activeArms"] = list(parsed["protocol"]["activeArms"])
    tampered["protocol"]["activeArms"][0] = "different-arm"
    unsigned = dict(tampered)
    unsigned.pop("manifestSha256")
    tampered["manifestSha256"] = mod._sha_bytes(mod._canonical(unsigned))
    tampered_bytes = (json.dumps(tampered, ensure_ascii=False, sort_keys=True,
                                 indent=2, allow_nan=False) + "\n").encode("utf-8")
    with pytest.raises(ValueError, match=r"\$\.protocol\.activeArms\[0\]"):
        mod.validate_manifest_bytes(tampered_bytes, raw_prices, raw_components, "screen")


def test_replay_mismatch_reports_bounded_first_path_type_and_values():
    # The real build/replay round-trip is covered above; this directly guards
    # the diagnostic format for mismatches after schema/digest validation.
    assert mod._first_difference({"a": [1]}, {"a": [2]}) == {
        "path": "$.a[0]", "actualType": "int", "expectedType": "int",
        "actual": "1", "expected": "2"}
    preview = mod._preview_json_value("x" * 300, 20)
    assert len(preview) == 20 and preview.endswith("...")
