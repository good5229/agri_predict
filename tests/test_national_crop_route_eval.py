import datetime as dt

import pytest

from agri_data import national_crop_route_eval as route


def _source_row(product, target, horizon):
    origin = target - dt.timedelta(days=horizon)
    return {"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
            "horizon": horizon, "cutoff": (origin-dt.timedelta(days=2)).isoformat(),
            "predictions": {m: 10.0 for m in ("ridge_alpha_10", "ridge_alpha_100", "calendar_sarima", "fixed_blend_50_50", "persistence")},
            "lineage": {}}


def test_route_is_fixed_and_answer_free_protocol():
    assert route.ROUTE == {"cabbage": "calendar_sarima", "radish": "ridge_alpha_100",
                           "onion": "ridge_alpha_100", "pepper": "ridge_alpha_100"}
    assert route.INPUT_HASHES[str(route.DEV_PATH)] == "8209689fbaddd33ffdfe7f30a87b200fb1fa881d5788e38ffebb13de8974b2e8"
    assert route.INPUT_HASHES[str(route.FRESH_PATH)] == "925ac27d569a1ca299451f7f4a3073bcece99b7e0901d9bbc2cf3d1a089f3fa4"
    assert "actual" not in route.PROTOCOL


def test_exact_fixed_grid_and_cutoff_validation():
    start = dt.date(2026, 1, 1)
    rows = [_source_row(p, start + dt.timedelta(days=i), h)
            for i in range(273) for p in route.PRODUCTS for h in route.HORIZONS]
    assert len(route._all_rows(({"forecasts": rows}, {"forecasts": []}))) == 10_920
    bad = [dict(r) for r in rows]
    bad[0] = {**bad[0], "cutoff": "2025-09-29"}
    with pytest.raises(ValueError, match="cutoff"):
        route._all_rows(({"forecasts": bad}, {"forecasts": []}))
    with pytest.raises(ValueError, match="10920"):
        route._all_rows(({"forecasts": rows[:-1]}, {"forecasts": []}))


def test_window_metrics_keep_same_pair_and_fixed_dm_family():
    rows = []
    start = dt.date(2026, 4, 1)
    for i in range(40):
        target = start + dt.timedelta(days=i)
        row = _source_row("cabbage", target, 1)
        row["predictions"] = {"crop_route": 9., "ridge_alpha_100": 10., "calendar_sarima": 11., "fixed_blend_50_50": 10.5}
        row["actual"] = 10.
        rows.append(row)
    report = route._window_report(rows, rows, {})
    assert report["dmFamilySizePerBandwidth"] == 240
    assert len(report["dmComparisons"]) == 240
    assert report["macro40CellEqualWeight"]["crop_route"]["mae"] is None  # 40-cell requirement
    assert report["commonPairs"] == 40
    dm = report["dmComparisons"][0]["hac9"]
    assert dm["n"] == 40 and "holmAdjustedPValue" in dm


def test_manifest_verifier_rejects_answer_bearing_rows():
    body = {"protocol": route.PROTOCOL, "phase": "forecast_only", "targetValuesAttached": False,
            "sourceManifestSha256": route.INPUT_HASHES, "plannedForecastCount": 10_920,
            "forecasts": [{"actual": 1}]}
    body["manifestSha256"] = route._canonical_hash(body)
    with pytest.raises(ValueError, match="answer-bearing"):
        route._verify_route_manifest(body)


def test_create_only_output_checks_existing_and_input_paths(tmp_path):
    existing = tmp_path / "existing.json"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        route._exclusive_output(existing, ())
    with pytest.raises(ValueError, match="overwrite"):
        route._exclusive_output(existing, (existing,))
    assert existing.read_text(encoding="utf-8") == "keep"


def test_target_bundle_requires_pinned_combined_observation_and_coverage_digests(monkeypatch, tmp_path):
    archive = route.ROOT / route.selection.national_transfer.INPUT_ARCHIVE.relative_to(route.ROOT)
    coverage = route.ROOT / "data/raw/nongnet_national_fresh_holdout_20260701-20260930/coverage-report.json"
    bundle = {"archivePath": str(archive), "coverageReportPath": str(coverage)}
    expected = {"archiveSha256": "a" * 64, "developmentObservationsSha256": "b" * 64,
                "freshObservationsSha256": "c" * 64, "freshCoverageReportSha256": "d" * 64}
    parsed = {"sourceDigests": {"archiveSha256": "a"*64, "combinedObservationsSha256": "c"*64,
                                "coverageReportSha256": "d"*64}}
    monkeypatch.setattr(route.selection, "_load_source", lambda _: parsed)
    monkeypatch.setattr(route, "_load_inputs", lambda _: (
        {"sourceDigests": {"combinedObservationsSha256": "b"*64}},
        {"sourceDigests": {"combinedObservationsSha256": "c"*64, "coverageReportSha256": "d"*64}}))
    route._validate_target_bundle(route.ROOT, bundle, expected)
    altered = {"sourceDigests": {**parsed["sourceDigests"], "combinedObservationsSha256": "e"*64}}
    monkeypatch.setattr(route.selection, "_load_source", lambda _: altered)
    with pytest.raises(ValueError, match="observation or fresh coverage"):
        route._validate_target_bundle(route.ROOT, bundle, expected)
    with pytest.raises(ValueError, match="requires the pinned"):
        route._validate_target_bundle(route.ROOT, None, expected)


def test_incomplete_gate_is_unsupported_and_crop_baseline_uses_lowest_mape_method():
    incomplete = route._gate({"apr_jun": {}, "jul_sep": {}})
    assert incomplete["status"] == "fail_or_unsupported"
    assert incomplete["criteria"]["apr_jun"]["complete40Cells"] is False

    report = {"complete40CellsAt30Pairs": True,
              "macro40CellEqualWeight": {m: {"mae": 10., "mape": 5.} for m in route.METHODS},
              "perCrop10HorizonEqualWeight": {}, "issuedCoverage": {m: {"issued": 100} for m in route.METHODS}}
    # Lowest MAPE ties resolve by COMPARATORS order, so Ridge100 supplies both
    # the crop MAPE and crop MAE comparison baseline.
    for crop in route.PRODUCTS:
        report["perCrop10HorizonEqualWeight"][crop] = {
            "crop_route": {"mae": 120., "mape": 5.5},
            "ridge_alpha_100": {"mae": 100., "mape": 4.},
            "calendar_sarima": {"mae": 200., "mape": 4.},
            "fixed_blend_50_50": {"mae": 90., "mape": 6.},
        }
    gate = route._gate({"apr_jun": report, "jul_sep": report})
    crop = gate["criteria"]["apr_jun"]["cropChanges"]["cabbage"]
    assert crop["chosenComparator"] == "ridge_alpha_100"
    assert crop["mapeWorsePp"] == pytest.approx(1.5)
    assert crop["maeWorsePct"] == pytest.approx(20.)


def test_joint_classification_keeps_unavailable_mixed_and_inconclusive_distinct():
    assert route._joint_classification(["improvement"] * 4) == "joint_improvement"
    assert route._joint_classification(["worsening"] * 4) == "joint_worsening"
    assert route._joint_classification(["improvement", "worsening", "inconclusive", "inconclusive"]) == "mixed"
    assert route._joint_classification(["inconclusive"] * 4) == "inconclusive"
    assert route._joint_classification(["improvement", "inference_unavailable", "inconclusive", "inconclusive"]) == "inference_unavailable"


def test_window_report_exposes_crop_macros_and_identical_prediction_flags():
    rows = []
    start = dt.date(2026, 4, 1)
    for i in range(40):
        target = start + dt.timedelta(days=i)
        for crop in route.PRODUCTS:
            for horizon in route.HORIZONS:
                row = _source_row(crop, target, horizon)
                row["predictions"] = {"crop_route": 10., "ridge_alpha_100": 10.,
                                      "calendar_sarima": 11., "fixed_blend_50_50": 10.5}
                row["actual"] = 10.
                rows.append(row)
    report = route._window_report(rows, rows, {})
    assert report["perCrop10HorizonEqualWeight"]["cabbage"]["crop_route"]["mae"] == 0
    cabbage_cell = next(c for c in report["cells"] if c["product"] == "cabbage" and c["horizon"] == 1)
    assert cabbage_cell["identicalPredictionsToComparator"]["ridge_alpha_100"] is True
    assert cabbage_cell["identicalPredictionsToComparator"]["calendar_sarima"] is False
    assert "ridge_alpha_100" in report["atLeastOneJointImprovement"]


def test_cli_reserves_or_rejects_output_before_build(monkeypatch, tmp_path):
    existing = tmp_path / "occupied.json"
    existing.write_text("untouched", encoding="utf-8")
    monkeypatch.setattr(route, "build_manifest", lambda: pytest.fail("expensive build ran before output check"))
    with pytest.raises(FileExistsError):
        route.main(["build", str(existing)])
    assert existing.read_text(encoding="utf-8") == "untouched"
    with pytest.raises(ValueError, match="overwrite"):
        route.main(["build", str(route.ROOT / route.DEV_PATH)])


def test_failed_build_does_not_leave_create_only_artifact(monkeypatch, tmp_path):
    destination = tmp_path / "failed-build.json"
    monkeypatch.setattr(route, "build_manifest", lambda: (_ for _ in ()).throw(ValueError("replay failed")))
    with pytest.raises(ValueError, match="replay failed"):
        route.main(["build", str(destination)])
    assert not destination.exists()
