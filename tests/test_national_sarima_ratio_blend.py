import datetime as dt

import pytest

from agri_data import national_sarima_ratio_blend as blend


def _component_doc():
    rows = []
    start = dt.date(2026, 1, 1)
    for i in range(273):
        target = start + dt.timedelta(days=i)
        for crop in blend.PRODUCTS:
            for h in blend.HORIZONS:
                origin = target - dt.timedelta(days=h)
                rows.append({"product": crop, "origin": origin.isoformat(), "target": target.isoformat(), "horizon": h,
                             "cutoff": (origin-dt.timedelta(days=2)).isoformat(),
                             "predictions": {blend.RATIO: 12., blend.SARIMA: 8., blend.BASELINE_BLEND: 9.,
                                             "ridge_alpha_100": 10., "persistence": 10.},
                             "lineage": {"componentSources": {blend.SARIMA: {"nativeStatus": "available"}}}})
    doc = {"schemaVersion": 1, "protocolId": blend.INPUT_PROTOCOL, "phase": "forecast_only", "targetValuesAttached": False,
           "window": ["2026-01-01", "2026-09-30"], "products": list(blend.PRODUCTS), "horizons": list(blend.HORIZONS),
           "methods": [blend.RATIO, "ridge_alpha_100", blend.SARIMA, blend.BASELINE_BLEND, "persistence"],
           "sourceDigests": {"development": {"archiveSha256": blend.ARCHIVE_SHA256, "combinedObservationsSha256": "c"*64},
                             "fresh": {"archiveSha256": blend.ARCHIVE_SHA256,
                                       "combinedObservationsSha256": "a"*64, "coverageReportSha256": "b"*64}},
           "plannedForecastCount": len(rows), "forecasts": rows}
    doc["manifestSha256"] = blend._hash(doc)
    return doc


def test_validate_and_blend_fixed_complete_answer_free_grid():
    source = _component_doc()
    checked = blend.validate_component_manifest(source)
    result = blend._build_from_component(source, blend.INPUT_SHA256)
    assert len(checked) == len(result["forecasts"]) == 10_920
    row = result["forecasts"][0]
    assert row["predictions"][blend.CANDIDATE] == 10.
    assert row["predictions"][blend.SARIMA] == 8.
    assert row["predictions"][blend.RATIO] == 12.
    assert result["targetValuesAttached"] is False
    blend.verify_manifest(result)


def test_component_grid_cutoff_prediction_and_answer_leak_are_rejected():
    source = _component_doc()
    source["forecasts"][0]["cutoff"] = "2025-12-28"
    source["manifestSha256"] = blend._hash({k: v for k, v in source.items() if k != "manifestSha256"})
    with pytest.raises(ValueError, match="cutoff"):
        blend.validate_component_manifest(source)

    source = _component_doc()
    source["forecasts"][0]["predictions"][blend.RATIO] = 0
    source["manifestSha256"] = blend._hash({k: v for k, v in source.items() if k != "manifestSha256"})
    with pytest.raises(ValueError, match="not positive finite"):
        blend.validate_component_manifest(source)

    source = _component_doc()
    source["forecasts"][0]["actual"] = 9
    source["manifestSha256"] = blend._hash({k: v for k, v in source.items() if k != "manifestSha256"})
    with pytest.raises(ValueError, match="answer-bearing"):
        blend.validate_component_manifest(source)


def test_duplicate_or_missing_forecast_key_is_rejected():
    source = _component_doc()
    source["forecasts"][-1] = dict(source["forecasts"][0])
    source["manifestSha256"] = blend._hash({k: v for k, v in source.items() if k != "manifestSha256"})
    with pytest.raises(ValueError, match="duplicate"):
        blend.validate_component_manifest(source)


def test_target_source_is_bound_to_fresh_observations_and_coverage(monkeypatch):
    source = {"sourceDigests": {"archiveSha256": blend.ARCHIVE_SHA256,
                                "combinedObservationsSha256": "a"*64, "coverageReportSha256": "b"*64}}
    bundle = {"archivePath": "prices.zip", "coverageReportPath": "coverage.json"}
    monkeypatch.setattr(blend.selection, "_load_source", lambda _: source if "coverageReportPath" in _ else {"sourceDigests": {"combinedObservationsSha256": "c"*64}})
    monkeypatch.setattr(blend.selection, "_coverage_status_evidence", lambda _: {})
    expected = {"development": {"combinedObservationsSha256": "c"*64},
                "fresh": {"combinedObservationsSha256": "a"*64, "coverageReportSha256": "b"*64}}
    assert blend._target_source(bundle, expected)[0] == source
    changed = {"sourceDigests": {**source["sourceDigests"], "coverageReportSha256": "d"*64}}
    monkeypatch.setattr(blend.selection, "_load_source", lambda _: changed if "coverageReportPath" in _ else {"sourceDigests": {"combinedObservationsSha256": "c"*64}})
    with pytest.raises(ValueError, match="digests differ"):
        blend._target_source(bundle, expected)
    with pytest.raises(ValueError, match="requires the pinned"):
        blend._target_source({}, expected)


def test_period_report_keeps_fixed_dm_family_and_pair_metrics():
    planned = []
    scored = []
    start = dt.date(2026, 4, 1)
    for i in range(40):
        target = start + dt.timedelta(days=i)
        for crop in blend.PRODUCTS:
            for h in blend.HORIZONS:
                origin = target - dt.timedelta(days=h)
                preds = {blend.CANDIDATE: 10.+i*.01, blend.SARIMA: 10.1+i*.005,
                         blend.BASELINE_BLEND: 10.2+i*.003, blend.RATIO: 10.3+i*.002}
                row = {"product": crop, "origin": origin.isoformat(), "target": target.isoformat(), "horizon": h,
                       "cutoff": (origin-dt.timedelta(days=2)).isoformat(), "predictions": preds,
                       "_hasValidTarget": True, "actual": 10.+i*.02}
                planned.append(row)
                scored.append(row)
    report = blend._period_report(planned, scored, {})
    assert report["complete40CellsAt30Pairs"] is True
    assert len(report["pairedDmComparisons"]) == 240
    assert report["holmFamilySizePerBandwidth"] == 240
    assert report["macro40CellEqualWeight"][blend.CANDIDATE]["mae"] is not None
    assert report["perCrop10HorizonEqualWeight"]["cabbage"][blend.CANDIDATE]["mape"] is not None
    assert report["issuedCoverage"][blend.CANDIDATE]["issuedOnValidTargets"] == 1_600
    assert len(report["jointDmConclusions"]) == 120


def test_joint_classification_and_gate_unsupported_are_explicit():
    assert blend._joint_classification(["improvement"]*4) == "joint_improvement"
    assert blend._joint_classification(["worsening"]*4) == "joint_worsening"
    assert blend._joint_classification(["improvement", "worsening", "inconclusive", "inconclusive"]) == "mixed"
    assert blend._joint_classification(["inference_unavailable", "inconclusive", "inconclusive", "inconclusive"]) == "inference_unavailable"
    report = blend._gate({name: {"complete40CellsAt30Pairs": False, "macro40CellEqualWeight": {},
                                 "perCrop10HorizonEqualWeight": {}, "issuedCoverage": {}, "jointDmConclusions": []}
                           for name in blend.WINDOWS})
    assert report["status"] == "unsupported"


def test_output_path_rejects_existing_and_input(tmp_path):
    existing = tmp_path/"exists.json"
    existing.write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        blend._exclusive_output(existing, ())
    with pytest.raises(ValueError, match="overwrite"):
        blend._exclusive_output(existing, (existing,))
    assert existing.read_text(encoding="utf-8") == "preserve"


def test_gate_uses_lower_mape_comparator_and_fixed_tolerances():
    periods = {}
    for idx, name in enumerate(blend.WINDOWS):
        candidate_mape = 7.7 if idx != 1 else 8.1
        per_crop = {}
        for crop in blend.PRODUCTS:
            per_crop[crop] = {
                blend.CANDIDATE: {"mae": 100., "mape": 7.5},
                blend.SARIMA: {"mae": 100., "mape": 8.},
                blend.BASELINE_BLEND: {"mae": 120., "mape": 9.},
                blend.RATIO: {"mae": 130., "mape": 10.},
            }
        periods[name] = {
            "complete40CellsAt30Pairs": True,
            "macro40CellEqualWeight": {
                blend.CANDIDATE: {"mae": 100., "mape": candidate_mape},
                blend.SARIMA: {"mae": 100., "mape": 8.},
                blend.BASELINE_BLEND: {"mae": 120., "mape": 9.},
                blend.RATIO: {"mae": 130., "mape": 10.},
            },
            "perCrop10HorizonEqualWeight": per_crop,
            "issuedCoverage": {m: {"issued": 100} for m in blend.METHODS},
            "jointDmConclusions": [{"product": "cabbage", "horizon": 1, "comparator": blend.SARIMA,
                                    "jointConclusion": "joint_improvement"}],
        }
    gate = blend._gate(periods)
    assert gate["criteriaByQuarter"]["jan_mar"]["strongestComparator"] == blend.SARIMA
    assert gate["criteriaByQuarter"]["apr_jun"]["macroGainAtLeast0_25pp"] is False
    assert gate["status"] == "pass"


def test_failed_cli_build_leaves_no_output(monkeypatch, tmp_path):
    output = tmp_path/"unbuilt.json"
    monkeypatch.setattr(blend, "build_manifest", lambda: (_ for _ in ()).throw(ValueError("synthetic replay failure")))
    with pytest.raises(ValueError, match="synthetic replay failure"):
        blend.main(["build", str(output)])
    assert not output.exists()


def test_valid_target_unissued_exclusions_use_private_target_flag():
    target = dt.date(2026, 4, 1)
    origin = target-dt.timedelta(days=1)
    planned = [{"product": "cabbage", "origin": origin.isoformat(), "target": target.isoformat(),
                "horizon": 1, "cutoff": (origin-dt.timedelta(days=2)).isoformat(),
                "predictions": {m: 10. for m in blend.METHODS}, "_hasValidTarget": True}]
    report = blend._period_report(planned, [], {})
    assert report["exclusionsByState"]["valid_positive_target_unissued_or_incomplete"] == 1
    planned[0]["_hasValidTarget"] = False
    report = blend._period_report(planned, [], {})
    assert report["exclusionsByState"]["valid_positive_target_unissued_or_incomplete"] == 0


def test_method_declaration_must_match_pinned_order_exactly():
    source = _component_doc()
    assert source["methods"] == [blend.RATIO, "ridge_alpha_100", blend.SARIMA, blend.BASELINE_BLEND, "persistence"]
    source["methods"] = [blend.RATIO, blend.SARIMA, "ridge_alpha_100", blend.BASELINE_BLEND, "persistence"]
    source["manifestSha256"] = blend._hash({k: v for k, v in source.items() if k != "manifestSha256"})
    with pytest.raises(ValueError, match="method declaration"):
        blend.validate_component_manifest(source)
