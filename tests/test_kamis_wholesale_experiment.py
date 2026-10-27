import datetime as dt
import hashlib
import json

import pytest

from agri_data import kamis_wholesale_experiment as experiment


def _forecast(crop, day, horizon, *, actual=1000.0, fixed=1.06, control=1.04, candidate=1.02,
              eligible=True, fixed_present=True):
    def issued(multiplier):
        return {"status": "issued", "prediction": actual * multiplier, "reason": None}
    return {"crop": crop, "target_date": day.isoformat(),
        "origin_date": (day - dt.timedelta(days=horizon)).isoformat(), "horizon": horizon,
        "eligible": eligible, "fixed50KeyPresent": fixed_present,
        "fixed50": issued(fixed), "blend106": issued(control), "blend113": issued(candidate),
        "blend106Raw": issued(control), "blend113Raw": issued(candidate)}


def _prices(dates):
    return {"schemaVersion": 1, "products": [
        {"id": crop, "observations": [{"date": day.isoformat(), "actual": 1000 + i * 7}
                                        for i, day in enumerate(dates)]}
        for crop in experiment.CROPS]}


def _manifest(rows):
    gates = {crop: {"eligibleForLaterReview": True} for crop in experiment.CROPS}
    return {"phaseName": "screen", "selectedModels": {crop: "retain_fixed50" for crop in experiment.CROPS},
        "inputSha256": {"archivePlan": "a", "archiveSourceManifest": "b", "diagnostic": "c"},
        "diagnosticIdentity": {"perCropGates": gates}, "forecasts": rows}


def test_daily_inference_uses_only_complete_same_date_ten_horizon_three_way_intersections():
    day = experiment.SCREEN_START
    actuals = {("cabbage", day): 1000.0}
    rows = [_forecast("cabbage", day, h) for h in experiment.HORIZONS]
    rows[-1]["blend106"] = {"status": "not_issued", "prediction": None}
    scored = experiment._daily_losses(rows, actuals, day, day)
    assert scored["cabbage"]["common"] == []
    assert sum(len(v) for v in scored["cabbage"]["byHorizon"].values()) == 9


def test_daily_inference_averages_ten_horizons_to_one_target_day():
    day = experiment.SCREEN_START
    rows = [_forecast("cabbage", day, h) for h in experiment.HORIZONS]
    scored = experiment._daily_losses(rows, {("cabbage", day): 1000.0}, day, day)
    assert len(scored["cabbage"]["common"]) == 1
    one_day = scored["cabbage"]["common"][0]
    assert one_day["blend113Ae"] == pytest.approx(20.0)
    assert one_day["blend113Ape"] == pytest.approx(.02)


def test_screen_keeps_fixed_twenty_hypothesis_family_and_unsupported_p1():
    day = experiment.SCREEN_START
    rows = [_forecast("cabbage", day, h) for h in experiment.HORIZONS]
    manifest = _manifest(rows)
    prices = _prices([day])
    report = experiment._score(manifest, prices, b"m", b"p", b"c", "screen")
    assert len(report["dm"]) == 20 * 2
    for bandwidth in experiment.BANDWIDTHS:
        family = [r for r in report["dm"] if r["bandwidth"] == bandwidth]
        assert len(family) == 20
        assert all(r["pValue"] == 1.0 and r["holmPValue"] == 1.0 for r in family)
        assert all(r["status"] == "insufficient_pairs" for r in family)


def test_screen_gate_rejects_any_candidate_loss_of_a_fixed50_key():
    days = [experiment.SCREEN_START + dt.timedelta(days=i) for i in range(35)]
    rows = [_forecast(crop, day, h, actual=1000 + i * 7)
            for crop in experiment.CROPS for i, day in enumerate(days) for h in experiment.HORIZONS]
    lost = next(r for r in rows if r["crop"] == "cabbage" and r["target_date"] == days[0].isoformat() and r["horizon"] == 10)
    lost["blend113"] = {"status": "not_issued", "prediction": None, "reason": "synthetic_coverage_loss"}
    prices = _prices(days)
    report = experiment._score(_manifest(rows), prices, b"m", b"p", b"c", "screen")
    cabbage = next(g for g in report["candidateGates"] if g["crop"] == "cabbage")
    assert cabbage["candidateLostFixed50Keys"]
    assert not cabbage["coverageNoLoss"] and not cabbage["screenPass"]


def test_joint_significance_requires_both_losses_and_both_hac_bands():
    dm = []
    for loss in experiment.LOSSES:
        for bandwidth in experiment.BANDWIDTHS:
            dm.append({"crop": "cabbage", "comparison": "ridge113_vs_fixed50", "loss": loss,
                "bandwidth": bandwidth, "status": "tested", "holmSignificant": True,
                "meanLossDifference": -1.0})
    assert experiment._joint_dm(dm, "cabbage", "ridge113_vs_fixed50")
    dm[-1]["holmSignificant"] = False
    assert not experiment._joint_dm(dm, "cabbage", "ridge113_vs_fixed50")


def test_hac_uses_calendar_lags_not_row_order():
    values = [-1.0 + (i % 2) * .1 for i in range(40)]
    first = dt.date(2025, 1, 1)
    contiguous = {first + dt.timedelta(days=i): value for i, value in enumerate(values)}
    spaced = {first + dt.timedelta(days=2 * i): value for i, value in enumerate(values)}
    day_hac = experiment.stats._dm(contiguous, 9)
    two_day_hac = experiment.stats._dm(spaced, 9)
    assert day_hac["status"] == two_day_hac["status"] == "tested"
    assert day_hac["n"] == two_day_hac["n"] == 40
    assert day_hac["statistic"] != two_day_hac["statistic"]


def test_manifest_replay_rejects_tampering_before_scoring(monkeypatch):
    expected = {"phase": "forecast_only", "phaseName": "screen", "targetValuesAttached": False,
                "forecasts": [], "manifestSha256": "expected"}
    monkeypatch.setattr(experiment, "_build_impl", lambda *a, **k: expected)
    raw = experiment._pretty(expected)
    assert experiment.validate_manifest_bytes(raw, b"p", b"c", "screen", diagnostic_raw=b"d") == expected
    tampered = json.loads(raw); tampered["forecasts"].append({"actual": 100})
    with pytest.raises(ValueError, match="differs from full deterministic replay"):
        experiment.validate_manifest_bytes(experiment._pretty(tampered), b"p", b"c", "screen", diagnostic_raw=b"d")


def test_target_values_must_match_the_source_bytes_bound_to_the_manifest():
    prices = _prices([experiment.SCREEN_START])
    raw = experiment._canonical(prices)
    assert experiment._bound_prices(prices, raw) == prices
    prices["products"][0]["observations"][0]["actual"] = 999
    with pytest.raises(ValueError, match="differ from the bound source bytes"):
        experiment._bound_prices(prices, raw)
    manifest = {"phaseName": "screen", "forecasts": []}
    manifest_raw = experiment._canonical(manifest)
    assert experiment._bound_manifest(manifest, manifest_raw) == manifest
    manifest["forecasts"].append({"actual": 100})
    with pytest.raises(ValueError, match="forecast manifest differs"):
        experiment._bound_manifest(manifest, manifest_raw)


def test_evaluation_with_no_screen_selection_stops_before_any_fit(monkeypatch):
    selection = {crop: "retain_fixed50" for crop in experiment.CROPS}
    monkeypatch.setattr(experiment, "validate_screen_artifacts", lambda *a, **k: {"selection": selection})
    monkeypatch.setattr(experiment, "_fit", lambda *a, **k: pytest.fail("fit called before no-selection gate"))
    with pytest.raises(ValueError, match="no screen-selected crop"):
        experiment.build(b"prices", b"components", "evaluation", diagnostic_raw=b"diagnostic",
            decision_raw=b"decision", screen_manifest_raw=b"screen", screen_report_raw=b"report",
            _diagnostic_override=({}, {}))


def test_public_replay_entrypoints_do_not_accept_unverified_selection_binding():
    with pytest.raises(TypeError, match="verified_replay_binding"):
        experiment.build(b"p", b"c", "evaluation", verified_replay_binding={"selection": {}})
    with pytest.raises(TypeError, match="verified_replay_binding"):
        experiment.validate_manifest_bytes(b"m", b"p", b"c", "evaluation", diagnostic_raw=b"d",
            verified_replay_binding={"selection": {}})


def test_diagnostic_gate_is_recomputed_and_editable_gate_claim_is_rejected(monkeypatch):
    gates = {crop: {"eligibleForLaterReview": crop != "pepper", "jointBaseAndQ0MonthlyOrigins": 31,
                    "logQ0OverP0Nonconstant": True, "reasons": []} for crop in experiment.CROPS}
    monkeypatch.setattr(experiment, "_recompute_diagnostic_gates", lambda *a: gates)
    archive = {"collectionPlanSha256": "plan", "sourceManifestSha256": "source"}
    payload = {"archiveValidation": {"collectionPlanSha256": "plan", "sourceManifestSha256": "source"},
               "auctionSource": {"sha256": hashlib.sha256(b"prices").hexdigest()},
               "perCrop": {crop: {"entryGate": dict(gate)} for crop, gate in gates.items()}}
    raw = experiment._pretty(payload)
    monkeypatch.setattr(experiment, "DIAGNOSTIC_SHA256", hashlib.sha256(raw).hexdigest())
    assert experiment._validate_diagnostic(b"prices", archive, raw)[1] == gates
    payload["perCrop"]["pepper"]["entryGate"]["eligibleForLaterReview"] = True
    tampered = experiment._pretty(payload)
    monkeypatch.setattr(experiment, "DIAGNOSTIC_SHA256", hashlib.sha256(tampered).hexdigest())
    with pytest.raises(ValueError, match="differs from original sources for pepper"):
        experiment._validate_diagnostic(b"prices", archive, tampered)
    payload["auctionSource"]["sha256"] = "0" * 64
    rebound = experiment._pretty(payload)
    monkeypatch.setattr(experiment, "DIAGNOSTIC_SHA256", hashlib.sha256(rebound).hexdigest())
    with pytest.raises(ValueError, match="source identity mismatch"):
        experiment._validate_diagnostic(b"prices", archive, rebound)


def test_feature_cache_hides_all_post_cutoff_source_values(monkeypatch):
    target = dt.date(2025, 1, 20)
    horizon = 3
    cutoff = target - dt.timedelta(days=horizon + 1)
    past = cutoff - dt.timedelta(days=1)
    future = cutoff + dt.timedelta(days=1)

    class Poison:
        def __getitem__(self, _key): raise AssertionError("future source value was inspected")

    seen = {}
    def helper(crop, records, direct, surveys, requested_target, requested_horizon):
        seen.update(records=records, direct=direct, surveys=surveys)
        return [0.0] * 113, 100.0, {"featureCutoff": cutoff.isoformat()}
    monkeypatch.setattr(experiment.survey_features, "features", helper)
    cache = {}
    vector, p0, _ = experiment._features(cache, "cabbage", target, horizon,
        {"cabbage": {past: {"price": 1}, future: Poison()}},
        {"cabbage": {past: (100.0, 1.0), future: Poison()}},
        {"cabbage": {past: {"meanRaw": "10"}, future: Poison()}})
    assert len(vector) == 113 and p0 == 100
    assert all(future not in seen[k] for k in ("records", "direct", "surveys"))


def test_failed_2026_gate_does_not_rewrite_screen_selected_policy():
    day = experiment.EVAL_START
    row = _forecast("cabbage", day, 1)
    row["fixed50KeyPresent"] = True
    row["origin_date"] = experiment.EVAL_ORIGIN_START.isoformat()
    row["eligible"] = True
    gates = [{"crop": "cabbage", "evaluationGate": "selected_candidate_fails_2026_gate_policy_retained"}]
    policy = experiment._policy_rows([row], {("cabbage", day): 1000.0},
        {"cabbage": "ridge113"}, gates)
    assert policy[0]["screenSelection"] == "ridge113"
    assert policy[0]["evaluationGateDecision"] == "selected_candidate_fails_2026_gate_policy_retained"
    assert policy[0]["policy"] == pytest.approx(1020.0)
    assert policy[0]["policy"] != policy[0]["fixed50"]


def test_tiny_evaluation_build_preserves_preperiod_rows_and_mixed_crop_selection(monkeypatch):
    crops = ("cabbage", "onion")
    dates = [dt.date(2026, 1, day) for day in (1, 2, 3)]
    monkeypatch.setattr(experiment, "CROPS", crops)
    monkeypatch.setattr(experiment, "HORIZONS", (1, 2))
    monkeypatch.setattr(experiment, "_phase", lambda phase: (dates[0], dates[-1], dates[0]))
    monkeypatch.setattr(experiment, "_grid", lambda phase: [
        (crop, day, h) for crop in crops for day in dates for h in experiment.HORIZONS])
    monkeypatch.setattr(experiment, "_code_hashes", lambda: {"test": "bound"})
    prices_raw, components_raw, diagnostic_raw = b"synthetic prices", b"synthetic components", b"diagnostic"
    monkeypatch.setattr(experiment, "PRICES_SHA256", hashlib.sha256(prices_raw).hexdigest())
    monkeypatch.setattr(experiment, "EVALUATION_COMPONENTS_SHA256", hashlib.sha256(components_raw).hexdigest())
    monkeypatch.setattr(experiment.source, "_check_environment", lambda: {"numpy": "synthetic"})
    def fit(rows, width):
        return {"width": width}, {"status": "fitted", "distinctTargets": 1, "samples": 1}
    def predict(model, vector, p0, width):
        return experiment._model("issued", 1000 + width)
    monkeypatch.setattr(experiment, "_fit", fit)
    monkeypatch.setattr(experiment, "_predict", predict)
    monkeypatch.setattr(experiment, "_features", lambda *args: ([0.0] * 113, 1000.0, {"status": "ok"}))
    components = {(crop, day, h): {"fixed50": 1000.0, "sarima": 1000.0}
                  for crop in crops for day in dates for h in experiment.HORIZONS}
    archive = {"collectionPlanSha256": "plan", "sourceManifestSha256": "source"}
    loaded = ({}, {crop: {} for crop in crops}, {}, {crop: {} for crop in crops}, components, archive, {})
    monkeypatch.setattr(experiment, "_component_map", lambda *_args: components)
    gates = {crop: {"eligibleForLaterReview": True} for crop in crops}
    manifest = experiment._run_build(prices_raw, components_raw, "evaluation", experiment.ARCHIVE_DIR,
        diagnostic_raw, selections={"cabbage": "ridge113", "onion": "retain_fixed50"},
        diagnostic_override=({}, gates), loaded=loaded)
    assert len(manifest["forecasts"]) == 12
    preperiod = [r for r in manifest["forecasts"] if r["origin_date"] < dates[0].isoformat()]
    assert len(preperiod) == 6
    assert all(r["eligible"] is False and r["featureAudit"]["status"] == "origin_before_phase_start"
               and r["blend106"]["reason"] == "origin_before_phase_start"
               and r["blend113"]["reason"] == "origin_before_phase_start"
               and r["fixed50"]["status"] == "issued" and r["fixed50"]["prediction"] == 1000.0
               and r["sarima"]["status"] == "issued" for r in preperiod)
    policy = experiment._policy_rows(preperiod, {}, {"cabbage": "ridge113", "onion": "retain_fixed50"}, [])
    assert len(policy) == 6 and all(r["preperiodOriginFallback"] and r["fallback"] for r in policy)
    eligible = [r for r in manifest["forecasts"] if r["eligible"]]
    cabbage = [r for r in eligible if r["crop"] == "cabbage"]
    onion = [r for r in eligible if r["crop"] == "onion"]
    assert cabbage and all(r["blend113"]["status"] == "issued" for r in cabbage)
    assert onion and all(r["blend113"]["reason"] == "crop_not_selected_by_frozen_screen" for r in onion)


def test_evaluation_uses_each_crop_selection_and_control_coverage_loss_fails_gate():
    day = experiment.EVAL_START
    rows = [_forecast(crop, day, h) for crop in experiment.CROPS for h in experiment.HORIZONS]
    row = next(r for r in rows if r["crop"] == "cabbage" and r["horizon"] == 1)
    row["blend106"] = {"status": "not_issued", "prediction": None}
    report = experiment._score(_manifest(rows), _prices([day]), b"m", b"p", b"c", "evaluation",
                               active_crops={"cabbage"})
    gates = {r["crop"]: r for r in report["candidateGates"]}
    assert gates["cabbage"]["controlLostFixed50Keys"]
    assert gates["cabbage"]["evaluationGate"] == "selected_candidate_fails_2026_gate_policy_retained"
    assert gates["onion"]["evaluationGate"] == "not_selected"
