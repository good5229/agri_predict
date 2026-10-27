import datetime as dt
import math

import pytest

from agri_data import national_price_selection as cv


def _archive():
    days = {}
    day = dt.date(2026, 4, 1)
    for i in range(100):
        d = day + dt.timedelta(days=i)
        days[d.isoformat()] = {"price": str(100 + i), "priceReason": None}
    return {"archiveSha256": "a" * 64, "sourceRowCount": len(days), "periods": [],
            "observations": {p: dict(days) for p in cv.PRODUCTS}}


def _source(archive):
    return {**archive, "schemaVersion": 1,
            "sourceDigests": {"archiveSha256": "a" * 64, "coverageReportSha256": "b" * 64,
                              "combinedObservationsSha256": cv._canonical_hash(archive["observations"])}}


def test_features_respect_origin_minus_two_cutoff():
    cutoff = dt.date(2026, 4, 10)
    target = cutoff + dt.timedelta(days=4)
    series = {cutoff - dt.timedelta(days=i): 100 + i for i in range(20)}
    original = cv._feature(series, target, 2)
    series[target] = 999999
    series[cutoff + dt.timedelta(days=1)] = 888888
    assert cv._feature(series, target, 2) == original


def test_precollection_raw_bundle_parses_only_the_pinned_archive(tmp_path, monkeypatch):
    raw = b"synthetic national archive bytes"
    archive_path = tmp_path / "archive.zip"
    archive_path.write_bytes(raw)
    parsed = _archive()
    digest = cv._sha(raw)
    monkeypatch.setattr(cv.national_transfer, "ARCHIVE_SHA256", digest)
    monkeypatch.setattr(cv.national_transfer, "parse_archive", lambda data, expected_sha256: parsed)
    source = cv._load_source({"archivePath": archive_path})
    assert source["sourceDigests"]["archiveSha256"] == digest
    assert source["sourceDigests"]["combinedObservationsSha256"] == cv._canonical_hash(parsed["observations"])


def test_ridge_fit_ignores_labels_after_monthly_cutoff():
    cutoff = dt.date(2026, 4, 1)
    series = {cutoff - dt.timedelta(days=i): 100 + (i % 13) for i in range(160)}
    fit_a, _, max_label_a = cv._ridge_fit(series, 3, cutoff, 100.0)
    changed = dict(series)
    changed[cutoff + dt.timedelta(days=1)] = 1e12
    changed[cutoff + dt.timedelta(days=2)] = 1e-12
    fit_b, _, max_label_b = cv._ridge_fit(changed, 3, cutoff, 100.0)
    assert fit_a.predict([[1.0] * 19]) == pytest.approx(fit_b.predict([[1.0] * 19]))
    assert max_label_a <= cutoff and max_label_b <= cutoff


def test_ridge_candidates_have_separate_alpha_fits_and_asof_cutoffs():
    first = dt.date(2025, 1, 1)
    series = {first + dt.timedelta(days=i): 100 * math.exp(.002 * i + .12 * math.sin(i / 5)) for i in range(300)}
    target = first + dt.timedelta(days=304)
    states = {}
    pred10, audit10 = cv._forecast_one(series, "cabbage", target, 3, states, "ridge_alpha_10")
    pred100, audit100 = cv._forecast_one(series, "cabbage", target, 3, states, "ridge_alpha_100")
    assert pred10 != pytest.approx(pred100)
    assert "ridge_10" in next(iter(states.values())) and "ridge_100" in next(iter(states.values()))
    cutoff = (target - dt.timedelta(days=5)).isoformat()
    assert audit10["fitCutoff"] <= cutoff and audit100["fitCutoff"] <= cutoff
    assert audit10["maxTrainingTargetDate"] <= cutoff and audit100["maxTrainingTargetDate"] <= cutoff


def test_sarima_fit_failure_is_cached_for_the_crop_month(monkeypatch):
    calls = []
    def failed_fit(series, cutoff):
        calls.append(cutoff)
        raise ValueError("synthetic_fit_failure")
    monkeypatch.setattr(cv, "_sarima_fit", failed_fit)
    first = dt.date(2026, 4, 1)
    series = {first + dt.timedelta(days=i): 100.0 + i for i in range(30)}
    states = {}
    _, first_audit = cv._forecast_one(series, "cabbage", first + dt.timedelta(days=20), 1, states, "calendar_sarima")
    _, second_audit = cv._forecast_one(series, "cabbage", first + dt.timedelta(days=21), 1, states, "calendar_sarima")
    assert calls == [first + dt.timedelta(days=17)]
    assert first_audit["fallbackReason"] == second_audit["fallbackReason"] == "ValueError:synthetic_fit_failure"


def test_forecast_manifest_is_answer_free_and_preserves_full_grid(monkeypatch):
    source = _source(_archive())

    def fake_forecast(series, product, target, horizon, states, method):
        return 100.0 + horizon, {"nativeStatus": "available", "fallback": False,
                                 "fallbackReason": None, "fitCutoff": (target - dt.timedelta(days=horizon + 2)).isoformat()}

    monkeypatch.setattr(cv, "_forecast_one", fake_forecast)
    monkeypatch.setattr(cv, "_load_source", lambda _: source)
    doc = cv.build_manifest({"archivePath": "synthetic.zip", "coverageReportPath": "synthetic.json"},
                            start=dt.date(2026, 4, 1), end=dt.date(2026, 4, 2))
    assert doc["phase"] == "forecast_only"
    assert doc["targetValuesAttached"] is False
    assert len(doc["forecasts"]) == 2 * len(cv.PRODUCTS) * len(cv.HORIZONS)
    assert all("actual" not in row and "targetValue" not in row for row in doc["forecasts"])
    assert all(row["cutoff"] == (dt.date.fromisoformat(row["target"]) - dt.timedelta(days=row["horizon"] + 2)).isoformat()
               for row in doc["forecasts"])


def test_scoring_rebuilds_forecasts_from_source_and_rejects_rehashed_forgery(monkeypatch):
    source = _source(_archive())

    def fake_forecast(series, product, target, horizon, states, method):
        return 100.0 + horizon, {"nativeStatus": "available", "fallback": False, "fallbackReason": None}

    monkeypatch.setattr(cv, "_load_source", lambda _: source)
    monkeypatch.setattr(cv, "_forecast_one", fake_forecast)
    raw_bundle = {"archivePath": "synthetic.zip", "coverageReportPath": "synthetic.json"}
    manifest = cv.build_manifest(raw_bundle, start=dt.date(2026, 4, 1), end=dt.date(2026, 4, 2))
    result = cv.score_manifest(manifest, raw_bundle, status_evidence={})
    assert result["commonScoredPairs"] == 80
    assert result["selection"]["status"] == "unsupported_default_ridge_alpha_100"
    tampered = {**manifest, "forecasts": [dict(r) for r in manifest["forecasts"]]}
    tampered["forecasts"][0]["predictions"] = dict(tampered["forecasts"][0]["predictions"])
    tampered["forecasts"][0]["predictions"]["ridge_alpha_10"] += 1
    tampered.pop("manifestSha256")
    tampered["manifestSha256"] = cv._canonical_hash(tampered)
    with pytest.raises(ValueError, match="forecast-only rows do not rebuild"):
        cv.score_manifest(tampered, raw_bundle, status_evidence={})


def test_frozen_development_selection_can_drive_synthetic_fresh_holdout(monkeypatch):
    source = _source(_archive())

    def fake_forecast(series, product, target, horizon, states, method):
        return 100.0 + horizon, {"nativeStatus": "available", "fallback": False, "fallbackReason": None}

    monkeypatch.setattr(cv, "_load_source", lambda _: source)
    monkeypatch.setattr(cv, "_forecast_one", fake_forecast)
    bundle = {"archivePath": "synthetic.zip", "coverageReportPath": "synthetic.json"}
    development = cv.build_manifest(bundle, start=dt.date(2025, 10, 1), end=dt.date(2026, 6, 30))
    selection_report = cv.score_manifest(development, bundle, status_evidence={})
    assert selection_report["selection"]["status"] == "selected"
    holdout = cv.build_manifest(bundle, start=dt.date(2026, 7, 1), end=dt.date(2026, 7, 2))
    scored = cv.score_manifest(holdout, bundle, status_evidence={}, selection_report=selection_report)
    assert scored["freshHoldout"]["selectedCandidate"] == selection_report["selection"]["selected"]
    assert len(scored["freshHoldout"]["comparisonsSelectedVsRidgeAndPersistence"]) == 160
    assert scored["freshHoldout"]["holmFamilySizePerBandwidth"] == 160
    assert scored["freshHoldout"]["coverage"]["total"][selection_report["selection"]["selected"]]["planned"] == 80
    assert set(scored["freshHoldout"]["perCropMape"]) == set(cv.PRODUCTS)
    assert scored["freshHoldout"]["overallGate"]["status"] in ("pass", "fail", "unsupported")
    assert len(selection_report["development"]["selection"]["comparisonsAllCandidatesVsRidgeAndPersistence"]) == 640


def test_missing_truth_is_unknown_absent_and_explicit_closure_is_distinct():
    archive = {"archiveSha256": "a" * 64, "periods": [], "sourceRowCount": 0,
               "observations": {p: {} for p in cv.PRODUCTS}}
    archive["schemaVersion"] = 1
    archive["sourceDigests"] = {"archiveSha256": "a" * 64, "coverageReportSha256": "b" * 64,
                                "combinedObservationsSha256": cv._canonical_hash(archive["observations"])}
    target = dt.date(2026, 4, 1)
    f = {"product": "cabbage", "origin": (target - dt.timedelta(days=1)).isoformat(),
         "target": target.isoformat(), "horizon": 1, "cutoff": (target - dt.timedelta(days=3)).isoformat(),
         "predictions": {**{m: 100.0 for m in cv.METHODS}, "persistence": 100.0}, "lineage": {}}
    manifest = {"protocol": cv.PROTOCOL, "phase": "forecast_only", "targetValuesAttached": False,
                "codeSha256": cv._sha(__import__("pathlib").Path(cv.__file__).read_bytes()),
                "sourceDigests": archive["sourceDigests"], "sourceBundleSha256": cv._canonical_hash(archive["sourceDigests"]),
                "window": [target.isoformat(), target.isoformat()], "forecasts": [f]}
    manifest["manifestSha256"] = cv._canonical_hash(manifest)
    _, absent = cv._pair_rows(manifest, archive)
    _, closed = cv._pair_rows(manifest, archive, {("cabbage", target.isoformat()): "market_closed"})
    assert absent == {"unknown_absent": 1}
    assert closed == {"market_closed": 1}


def test_dm_family_has_frozen_160_tests_and_reports_unavailable_samples():
    rows = []
    start = dt.date(2026, 4, 1)
    for p_i, product in enumerate(cv.PRODUCTS):
        for h in cv.HORIZONS:
            for i in range(30):
                actual = 100 + i % 5
                rows.append({"product": product, "horizon": h, "target": (start + dt.timedelta(days=i)).isoformat(),
                    "actual": actual, "predictions": {"ridge_alpha_10": actual + 2 + i % 2,
                        "ridge_alpha_100": actual + 1, "calendar_sarima": actual + 3,
                        "fixed_blend_50_50": actual + 2, "persistence": actual + 4}})
    results = cv._dm_family(rows, "ridge_alpha_10")
    assert len(results) == 160
    assert {r["comparator"] for r in results} == {"ridge_alpha_100", "persistence"}
    assert all(set(r["dm"]) == {"9", "28"} for r in results)
    assert all("holmPValue" in r["dm"]["9"] and "conclusion" in r["dm"]["28"] for r in results)


def test_dm_family_retains_self_comparison_cells_when_ridge100_is_selected():
    rows = []
    for product in cv.PRODUCTS:
        for h in cv.HORIZONS:
            for i in range(30):
                y = 100 + i % 3
                rows.append({"product": product, "horizon": h, "target": (dt.date(2026, 4, 1) + dt.timedelta(days=i)).isoformat(),
                    "actual": y, "predictions": {**{m: y + 1 for m in cv.METHODS}, "persistence": y + 2}})
    results = cv._dm_family(rows, "ridge_alpha_100")
    assert len(results) == 160
    self_comparisons = [r for r in results if r["comparator"] == "ridge_alpha_100"]
    assert len(self_comparisons) == 80
    assert all(r["n"] == 30 and r["dm"]["9"]["pValue"] == 1 for r in self_comparisons)


def test_dm_unavailable_and_joint_mixed_direction_are_explicit():
    rows = [{"product": "cabbage", "horizon": 1, "target": "2026-04-01", "actual": 100.0,
             "predictions": {"ridge_alpha_10": 99.0, "ridge_alpha_100": 100.0,
                             "calendar_sarima": 100.0, "fixed_blend_50_50": 100.0, "persistence": 100.0}}]
    unavailable = cv._dm_family(rows, "ridge_alpha_10")
    assert all(r["dm"]["9"]["conclusion"] == "inference_unavailable" for r in unavailable)
    synthetic = []
    for loss, conclusion in (("mae", "improvement"), ("mape", "worsening")):
        synthetic.append({"product": "cabbage", "horizon": 1, "candidate": "ridge_alpha_10",
            "comparator": "ridge_alpha_100", "loss": loss, "n": 30, "conclusion": conclusion,
            "dm": {"9": {"status": "tested", "conclusion": conclusion},
                   "28": {"status": "tested", "conclusion": conclusion}}})
    joined = cv._joint_cell_verdicts(synthetic)[0]
    assert joined["verdictByBandwidth"]["9"]["joint"] == "inconclusive"
    assert joined["verdictByBandwidth"]["9"]["mixedDirection"] is True
    assert joined["verdictByBandwidth"]["28"]["directionPattern"] == "mae=improvement;mape=worsening"


def test_fresh_gate_requires_all_frozen_metrics_and_joint_hac9_improvement():
    rows = []
    start = dt.date(2026, 7, 1)
    for product in cv.PRODUCTS:
        for horizon in cv.HORIZONS:
            for i in range(40):
                actual = 100 + (i % 7)
                rows.append({"product": product, "horizon": horizon, "target": (start + dt.timedelta(days=i)).isoformat(),
                    "actual": actual, "predictions": {"ridge_alpha_10": actual + 1 + i % 2,
                        "ridge_alpha_100": actual + 5 + i % 3, "calendar_sarima": actual + 7,
                        "fixed_blend_50_50": actual + 2 + i % 2, "persistence": actual + 8}})
    metrics = cv._macro_40(rows)
    coverage = {"total": {m: {"issued": 1600 if m == "ridge_alpha_10" else 1500, "planned": 1600}
                           for m in (*cv.METHODS, "persistence")}}
    crop_mape = cv._per_crop_mape(rows, "ridge_alpha_10")
    tests = cv._dm_family(rows, "ridge_alpha_10")
    joint = cv._joint_cell_verdicts(tests)
    gate = cv._fresh_gate(metrics, coverage, crop_mape, joint, "ridge_alpha_10")
    assert gate["status"] == "pass"
    assert gate["criteria"]["mapeImprovementAtLeast0_5pp"] is True
    assert gate["criteria"]["atLeastOneCellJointlyImprovedAtHac9AndHac28"] is True
    lower_coverage = {"total": {**coverage["total"], "ridge_alpha_10": {"issued": 1499, "planned": 1600}}}
    failed = cv._fresh_gate(metrics, lower_coverage, crop_mape, joint, "ridge_alpha_10")
    assert failed["status"] == "fail"
    assert failed["criteria"]["issuedCoverageNotDecreased"] is False
    hac9_only = [dict(row, verdictByBandwidth={**row["verdictByBandwidth"],
        "28": {**row["verdictByBandwidth"]["28"], "joint": "inconclusive"}}) for row in joint]
    failed_two_bandwidth_gate = cv._fresh_gate(metrics, coverage, crop_mape, hac9_only, "ridge_alpha_10")
    assert failed_two_bandwidth_gate["criteria"]["atLeastOneCellJointlyImprovedAtHac9AndHac28"] is False


def test_selection_uses_macro_mape_and_frozen_tie_order():
    rows = []
    for product in cv.PRODUCTS:
        for h in cv.HORIZONS:
            for i in range(30):
                y = 100.0 + i % 4
                rows.append({"product": product, "horizon": h, "target": (dt.date(2026, 4, 1) + dt.timedelta(days=i)).isoformat(),
                    "actual": y, "predictions": {"ridge_alpha_10": y + 1, "ridge_alpha_100": y + 1,
                    "calendar_sarima": y + 1, "fixed_blend_50_50": y + 1, "persistence": y + 2}})
    coverage = {m: {f"{p}/{h}": 30 for p in cv.PRODUCTS for h in cv.HORIZONS}
                for m in (*cv.METHODS, "persistence")}
    result = cv._select(rows, coverage)
    assert result["complete40Cells"] is True
    assert result["selected"] == "ridge_alpha_100"
    assert result["eligible"] == list(cv.METHODS)
    assert result["macroMae"]["ridge_alpha_100"] == pytest.approx(1.0)
