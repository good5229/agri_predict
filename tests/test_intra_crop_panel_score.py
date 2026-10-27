import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import platform
from pathlib import Path

import pytest

from agri_data import intra_crop_panel_score as scorer


def _manifest():
    forecasts = []
    for product in scorer.PRODUCTS:
        for horizon in scorer.HORIZONS:
            phase_start, phase_end = scorer.PERIODS["V"]
            for offset in range(horizon, (phase_end-phase_start).days + 1):
                target = phase_start + dt.timedelta(days=offset)
                origin = target - dt.timedelta(days=horizon)
                selected = target == dt.date(2026, 5, 1) + dt.timedelta(days=horizon)
                forecasts.append({
                "product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                "horizon": horizon, "fitMonth": origin.strftime("%Y-%m"),
                "B": {
                    "E106_RHS": {"status": "issued" if selected else "not_issued",
                                 "prediction": 100.0 if selected else None},
                    "E106": {"status": "issued" if selected else "not_issued",
                             "prediction": 101.0 if selected else None},
                },
                "A_old": {
                    "CTRL_LGB": {"status": "issued", "prediction": 100.0},
                    "CAND_A": {"status": "issued", "prediction": 100.0},
                },
                "models": {
                    "L": {"status": "issued" if selected else "not_issued",
                          "prediction": 100.0 if selected else None, "logRatio": 0.0 if selected else None,
                          "p0": 100.0, "p0Date": origin.isoformat()},
                    "P": {"status": "issued", "prediction": 110.0, "logRatio": math.log(1.1),
                          "p0": 100.0, "p0Date": origin.isoformat(), "modelUsed": True,
                          "rawModelStatus": "issued", "rawModelPrediction": 110.0,
                          "rawModelLogRatio": math.log(1.1)},
                    "Q": {"status": "issued" if selected else "not_issued",
                          "prediction": 100.0 if selected else None, "logRatio": 0.0 if selected else None,
                          "p0": 100.0, "p0Date": origin.isoformat(), "modelUsed": True},
                },
                })
    fits = []
    for product in scorer.PRODUCTS:
        for horizon in scorer.HORIZONS:
            for month in ("2026-04", "2026-05", "2026-06"):
                for variant in ("L", "P", "Q"):
                    fit = {
                        "product": product, "horizon": horizon, "fitMonth": month, "variant": variant,
                        "representativeKeyDigest": f"{product}-{horizon}-{month}",
                        "status": "complete", "timeIntegrity": {
                            "allTrainingTargetsBeforeFitOrigin": True,
                            "allFeatureDatesAtOrBeforeCutoff": True,
                        },
                        "representativeTrainDiagnostics": {"n": 1, "logMse": 0.0,
                                                           "priceMAE": 0.0, "priceMAPE": 0.0},
                        "representativeTrainLogDecomposition": {},
                        "support": [{"seriesId": 0, "rows": 1, "distinctTargets": 1, "weightSum": 1.0}],
                        "originWeightDiagnostics": [{"origin": "2026-03-01", "target": "2026-03-02",
                                                     "K": 0, "totalWeight": 1.0,
                                                     "representativeWeight": 1.0, "auxiliaryWeight": 0.0}],
                    }
                    if variant == "Q":
                        fit["pqIdentity"] = {
                            "sameRepresentativeKeyDigest": True, "sameInputWeightDigest": True,
                            "sameBinningIdentity": True,
                        }
                        fit["permutationDiagnostics"] = {"blockCount": 0, "unchangedBlocks": 0,
                                                         "changedRows": 0, "auxiliaryRows": 0,
                                                         "changedAuxiliaryWeight": 0.0,
                                                         "auxiliaryWeight": 0.0,
                                                         "changedWeightedShare": 0.0}
                        fit["representativeTrainLogDecomposition"] = {
                            "P-Q": {"n": 1, "deltaSquaredSum": 0.0, "crossTermSum": 0.0,
                                    "squaredErrorDifferenceSum": 0.0, "identityMaxAbsError": 0.0},
                        }
                    if variant == "P":
                        fit["representativeTrainLogDecomposition"] = {
                            "P-L": {"n": 1, "deltaSquaredSum": 0.0, "crossTermSum": 0.0,
                                    "squaredErrorDifferenceSum": 0.0, "identityMaxAbsError": 0.0},
                        }
                    fits.append(fit)
    doc = {
        "schemaVersion": 1, "experiment": "INTRA_CROP_PANEL_CAND_B_V1", "phase": "V",
        "targetValuesAttached": False, "forecasts": forecasts, "fits": fits,
        "technicalCompletion": {product: {"status": "complete", "reason": None} for product in scorer.PRODUCTS},
        "integrityChecks": {
            "sourceCacheHashesVerified": True, "baselineKeysAndActualsVerified": True,
            "forecastReplayVerified": True, "resolvedConfigVerified": True,
        },
    }
    doc["manifestSha256"] = hashlib.sha256(scorer._canonical(doc)).hexdigest()
    return doc


def _prices():
    observations = []
    for day in range(1, 11):
        observations.append({"date": (dt.date(2026, 5, 1) + dt.timedelta(days=day)).isoformat(),
                             "actual": 110.0})
    products = [{"id": product, "observations": observations} for product in scorer.PRODUCTS]
    products.append({"id": "pepper", "observations": observations})
    return {"schemaVersion": 1, "products": products}


def _fast_bootstrap(*args, block, metric, **kwargs):
    return {"blockDays": block, "metric": metric, "requested": 2000, "valid": 0,
            "invalid": 2000, "interval95": None, "seed": 0}


def _external_evidence_fixture(tmp_path, monkeypatch):
    from agri_data import chronos2_market_context as rhs
    from agri_data import external_reference_v1 as source
    from agri_data import intra_crop_panel as producer

    output_dir = tmp_path / "panel-output"
    phase_dir = output_dir / "T"
    phase_dir.mkdir(parents=True)
    start, end = scorer.PERIODS["T"]
    forecasts = []
    observations = []
    for day_offset in range((end-start).days + 1):
        day = start + dt.timedelta(days=day_offset)
        observations.append({"date": day.isoformat(), "actual": 110.0})
    prices = {"schemaVersion": 1, "products": [
        {"id": product, "observations": observations} for product in scorer.PRODUCTS]}
    prices_path = tmp_path / "prices.json"
    prices_path.write_bytes(scorer._canonical(prices))
    monkeypatch.setattr(scorer.price_helper, "PRICES_SHA256", scorer._sha(prices_path.read_bytes()))
    actuals = source._actuals_from_snapshot(prices)
    positive_keys = sorted((product, day.isoformat()) for (product, day), value in actuals.items()
                           if scorer._positive(value) is not None)
    first_legal = max(start, scorer.ORIGIN_STARTS["T"] + dt.timedelta(days=1))
    expected_targets = sorted((product, (first_legal + dt.timedelta(days=offset)).isoformat())
                              for product in scorer.PRODUCTS
                              for offset in range((end-first_legal).days + 1))
    phase_positive = [(product, day.isoformat()) for (product, day), value in actuals.items()
                      if start <= day <= end and scorer._positive(value) is not None]

    for product in scorer.PRODUCTS:
        for horizon in scorer.HORIZONS:
            for offset in range((end-start).days + 1):
                target = start + dt.timedelta(days=offset)
                origin = target - dt.timedelta(days=horizon)
                row = {"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                       "horizon": horizon, "fitMonth": origin.strftime("%Y-%m"),
                       "B": {name: {"status": "issued", "prediction": 100.0, "sourceStatus": "issued"}
                             for name in ("E106_RHS", "E106")},
                       "models": {
                           "L": {"status": "issued", "prediction": 100.0, "logRatio": 0.0,
                                 "p0": 100.0, "p0Date": origin.isoformat()},
                           "P": {"status": "issued", "prediction": 110.0, "logRatio": math.log(1.1),
                                 "rawModelLogRatio": math.log(1.1), "p0": 100.0,
                                 "p0Date": origin.isoformat(), "modelUsed": True,
                                 "rawModelStatus": "issued", "rawModelPrediction": 110.0},
                       }}
                forecasts.append(row)

    def baseline_doc(phase, rows):
        return {"phase": phase, "forecasts": [{
            "product": row["product"], "origin": row["origin"], "target": row["target"],
            "horizon": row["horizon"], "conditions": {
                "C": {"status": "issued", "blendPrediction": row["B"]["E106_RHS"]["prediction"]},
                "B": {"status": "issued", "blendPrediction": row["B"]["E106"]["prediction"]},
            }} for row in rows]}

    v_baseline_path = tmp_path / "rhs-v.json"
    v_baseline_path.write_bytes(scorer._canonical(baseline_doc("V", [{
        "product": "cabbage", "origin": "2026-04-01", "target": "2026-04-02", "horizon": 1,
        "B": {"E106_RHS": {"prediction": 100.0}, "E106": {"prediction": 101.0}}}])))
    t_baseline_path = tmp_path / "rhs-t.json"
    t_baseline_path.write_bytes(scorer._canonical(baseline_doc("T", forecasts)))
    baseline_paths = {"V": v_baseline_path, "T": t_baseline_path}
    baseline_hashes = {key: scorer._sha(path.read_bytes()) for key, path in baseline_paths.items()}
    monkeypatch.setattr(rhs, "BASELINE_PATHS", baseline_paths)
    monkeypatch.setattr(rhs, "BASELINE_SHA256", baseline_hashes)

    cache = {"cacheSha256": "test-cache-internal", "channels": []}
    cache_path = tmp_path / "source-cache.json"
    cache_path.write_bytes(scorer._canonical(cache))
    monkeypatch.setattr(producer, "SOURCE_CACHE", cache_path)
    monkeypatch.setattr(producer, "SOURCE_CACHE_INTERNAL_SHA256", cache["cacheSha256"])
    config = {"status": "resolved", "sourceManifestSha256": producer.SOURCE_V_MANIFEST_SHA256,
              "fitParams": {}, "inherited": {"max_bin": 255},
              "serializedCanonicalParameters": {"cat_l2": 10.0, "cat_smooth": 10.0,
                                                "max_cat_threshold": 32, "max_cat_to_onehot": 4,
                                                "min_data_per_group": 100, "max_bin": 255},
              "environment": {"python": platform.python_version(),
                              **{name: importlib.metadata.version(name)
                                 for name in ("lightgbm", "numpy", "joblib")}}}
    config["configSha256"] = scorer._sha(scorer._canonical(config))
    (output_dir / "resolved-config.json").write_bytes(scorer._canonical(config))
    execution = producer._execution_identity(config, scorer._sha(cache_path.read_bytes()))
    execution_sha = scorer._sha(scorer._canonical(execution))
    baseline_counts = {key: len(source._baseline_index(json.loads(path.read_text()), key))
                       for key, path in baseline_paths.items()}
    preflight = {"status": "preflight_complete", "sourceCacheInternalSha256": cache["cacheSha256"],
                 "sourceCacheFileSha256": scorer._sha(cache_path.read_bytes()),
                 "resolvedConfigSha256": config["configSha256"], "baselineHashes": baseline_hashes,
                 "baselineKeysByPhase": baseline_counts,
                 "targetSnapshotSha256": scorer._sha(prices_path.read_bytes()),
                 "executionIdentity": execution, "sourceCacheHashesVerified": True,
                 "resolvedConfigVerified": True, "baselineKeysVerified": True,
                 "targetSnapshotVerified": True, "representativeKeyInheritanceVerified": True}
    preflight["preflightSha256"] = scorer._sha(scorer._canonical(preflight))
    (output_dir / "preflight.json").write_bytes(scorer._canonical(preflight))

    manifest = {"phase": "T", "targetValuesAttached": False, "forecasts": forecasts, "fits": [],
                "sourceCacheFileSha256": scorer._sha(cache_path.read_bytes()),
                "sourceCacheSha256": cache["cacheSha256"],
                "baselineManifestSha256": baseline_hashes["T"],
                "targetSnapshotSha256": scorer._sha(prices_path.read_bytes()),
                "resolvedConfigSha256": config["configSha256"],
                "executionIdentity": execution, "executionIdentitySha256": execution_sha,
                "positiveActualKeyCount": len(positive_keys),
                "positiveActualKeysSha256": scorer._sha(scorer._canonical(positive_keys)),
                "phaseActualCoverage": {
                    "status": "pass", "phaseStart": start.isoformat(), "phaseEnd": end.isoformat(),
                    "originFloor": scorer.ORIGIN_STARTS["T"].isoformat(),
                    "firstLegalTarget": first_legal.isoformat(),
                    "expectedLegalTargetPairCount": len(expected_targets),
                    "forecastTargetPairCount": len(expected_targets),
                    "positiveActualKeyCountInPhase": len(phase_positive),
                    "coveredPositiveActualKeyCount": len(phase_positive),
                    "excludedPositiveActualKeyCount": 0, "excludedPositiveActualKeys": [],
                    "legalTargetGridSha256": scorer._sha(scorer._canonical(expected_targets)),
                },
                "preflightArtifactPath": "preflight.json", "preflightSha256": preflight["preflightSha256"],
                "forecastReplayAuditPath": "T/replay-verification.json"}
    replay = {"phase": "T", "status": "pass", "preReplayManifestSha256": "pretend-pre-manifest",
              "forecastRowsSha256": scorer._sha(scorer._canonical(forecasts)),
              "sourceCacheFileSha256": manifest["sourceCacheFileSha256"],
              "baselineManifestSha256": baseline_hashes["T"],
              "resolvedConfigSha256": config["configSha256"],
              "executionIdentitySha256": execution_sha, "checkpointEvidence": [],
              "baselineCopiedValuesVerified": True, "AOldCopiedValuesVerified": True}
    replay["replayVerificationSha256"] = scorer._sha(scorer._canonical(replay))
    replay_path = phase_dir / "replay-verification.json"
    replay_path.write_bytes(scorer._canonical(replay))
    manifest["forecastReplayAuditSha256"] = scorer._sha(replay_path.read_bytes())
    manifest["replayVerificationSha256"] = replay["replayVerificationSha256"]
    manifest["manifestSha256"] = scorer._sha(scorer._canonical(manifest))
    manifest_path = phase_dir / "forecast-manifest.json"
    manifest_path.write_bytes(scorer._canonical(manifest))
    return manifest, prices, manifest_path, prices_path, baseline_paths, baseline_hashes, replay_path


def test_score_uses_fixed_families_and_macro_accounting(monkeypatch):
    monkeypatch.setattr(scorer.shared_diagnostics, "circular_block_bootstrap", _fast_bootstrap)
    score, diagnostic, decision = scorer.score(_manifest(), _prices())

    assert score["comparisons"]["P_vs_B"]["macro"]["pairedRows"] == 40
    assert score["comparisons"]["P_vs_B"]["macro"]["aMape"] < score["comparisons"]["P_vs_B"]["macro"]["bMape"]
    assert score["directModelOnlyPvsB"]["macro"]["pairedRows"] == 40
    assert len(score["dm"]) == 480
    assert [x["familySize"] for x in score["dmSummaryByBandwidth"]] == [240, 240]
    assert all(score["priceGainLossContributions"][name]["accountingVerified"] == {"mae": True, "mape": True}
               for name in ("P_vs_B", "P_vs_L", "P_vs_Q"))
    assert diagnostic["timeIntegrityVerified"] is True
    assert diagnostic["representativeTrainLogDecomposition"][0]["identityVerified"] is True
    # Self-asserted manifest booleans cannot authorize a passing gate; the
    # production CLI supplies paths for independent artifact verification.
    assert diagnostic["informationIntegrityVerified"] is False
    assert decision["gate"]["passed"] is False
    assert decision["status"] == "technical_or_integrity_block"


def test_missing_q_does_not_shrink_primary_pair_set(monkeypatch):
    monkeypatch.setattr(scorer.shared_diagnostics, "circular_block_bootstrap", _fast_bootstrap)
    doc = _manifest()
    for row in doc["forecasts"]:
        row["models"]["Q"] = {"status": "not_issued", "prediction": None, "modelUsed": False}
    doc.pop("manifestSha256")
    doc["manifestSha256"] = hashlib.sha256(scorer._canonical(doc)).hexdigest()

    score, _, _ = scorer.score(doc, _prices())

    assert score["comparisons"]["P_vs_B"]["macro"]["pairedRows"] == 40
    assert score["directModelOnlyPvsB"]["macro"]["pairedRows"] == 40
    assert score["comparisons"]["P_vs_Q"]["macro"]["pairedRows"] == 0
    assert len([row for row in score["dm"] if row["comparison"] == "P_vs_Q"]) == 160
    assert all(row["status"] == "unsupported_missing_frozen_cell"
               for row in score["jointCalendarBlockBootstrap"]["P_vs_Q"])


def test_fallback_is_reported_separately_and_preserves_policy(monkeypatch):
    monkeypatch.setattr(scorer.shared_diagnostics, "circular_block_bootstrap", _fast_bootstrap)
    doc = _manifest()
    p = next(row for row in doc["forecasts"] if row["product"] == "cabbage"
             and row["horizon"] == 1 and row["target"] == "2026-05-02")["models"]["P"]
    p.update({"status": "issued", "prediction": 100.0, "modelUsed": False,
              "rawModelStatus": "not_issued", "rawModelPrediction": None,
              "rawModelLogRatio": None, "fallbackReason": "fixture"})
    doc.pop("manifestSha256")
    doc["manifestSha256"] = hashlib.sha256(scorer._canonical(doc)).hexdigest()

    score, _, decision = scorer.score(doc, _prices())

    assert score["coverage"]["cabbage/1"]["candidateFallback"] == 1
    assert score["coverage"]["cabbage/1"]["candidateModelIssued"] == 79
    assert score["comparisons"]["P_vs_B"]["macro"]["pairedRows"] == 40
    assert score["directModelOnlyPvsB"]["macro"]["pairedRows"] == 39
    assert decision["gate"]["noBaselineCoverageLoss"] is True


def test_incomplete_product_blocks_even_with_perfect_fallback(monkeypatch):
    monkeypatch.setattr(scorer.shared_diagnostics, "circular_block_bootstrap", _fast_bootstrap)
    doc = _manifest()
    doc["technicalCompletion"]["cabbage"] = {"status": "failed", "reason": "fixture"}
    # Make P equal to the baseline everywhere, so price failure does not mask the technical block.
    for row in doc["forecasts"]:
        p = row["models"]["P"]
        p.update({"prediction": 100.0, "rawModelPrediction": 100.0, "logRatio": 0.0,
                  "rawModelLogRatio": 0.0})
    doc.pop("manifestSha256")
    doc["manifestSha256"] = hashlib.sha256(scorer._canonical(doc)).hexdigest()

    _, _, decision = scorer.score(doc, _prices())

    assert decision["status"] == "technical_or_integrity_block"
    assert decision["gate"]["allFourProductsTechnicallyComplete"] is False
    assert decision["gate"]["passed"] is False


def test_manifest_rejects_embedded_answers():
    doc = _manifest()
    doc["forecasts"][0]["actual"] = 110.0
    doc.pop("manifestSha256")
    doc["manifestSha256"] = hashlib.sha256(scorer._canonical(doc)).hexdigest()
    with pytest.raises(ValueError, match="answer value"):
        scorer._validate_manifest(doc)


def test_manifest_claimed_integrity_does_not_replace_external_evidence():
    doc = _manifest()
    result = scorer._fit_diagnostics(doc, "V")
    assert result["informationIntegrityChecks"] == {
        "sourceCacheHashesVerified": False,
        "baselineKeysAndActualsVerified": False,
        "forecastReplayVerified": False,
        "resolvedConfigVerified": False,
    }
    assert result["informationIntegrityVerified"] is False


def test_required_fit_diagnostics_fail_closed():
    doc = _manifest()
    p_fit = next(row for row in doc["fits"] if row["variant"] == "P")
    p_fit.pop("support")
    q_fit = next(row for row in doc["fits"] if row["variant"] == "Q")
    q_fit.pop("permutationDiagnostics")
    q_fit["representativeTrainLogDecomposition"].pop("P-Q")
    diagnostics, permutation, decomposition = scorer._fit_diagnostic_integrity(doc["fits"], "V")
    assert diagnostics is False
    assert permutation is False
    assert decomposition is False


def test_not_run_test_phase_q_null_decomposition_is_skipped():
    assert scorer._train_log_decomposition([{
        "status": "not_run", "variant": "Q",
        "representativeTrainLogDecomposition": {"P-Q": None},
    }]) == []


def _no_support_fit(variant, status, *, phase="V"):
    fit = {"product": "cabbage", "horizon": 1, "fitMonth": "2026-04", "variant": variant,
           "status": status, "representativeKeyDigest": "representative-key-digest",
           "timeIntegrity": {"allTrainingTargetsBeforeFitOrigin": True,
                             "allFeatureDatesAtOrBeforeCutoff": True},
           "representativeTrainDiagnostics": {"n": 1, "logMse": 0.0, "priceMAE": 0.0, "priceMAPE": 0.0},
           "support": [{"seriesId": 0, "rows": 1, "distinctTargets": 1, "weightSum": 1.0},
                       {"seriesId": 1, "rows": 0, "distinctTargets": 0, "weightSum": 0.0}],
           "originWeightDiagnostics": [{"origin": "2026-04-01", "target": "2026-04-02", "K": 0,
                                        "totalWeight": 1.0, "representativeWeight": 1.0,
                                        "auxiliaryWeight": 0.0}]}
    if variant == "P":
        fit["representativeTrainLogDecomposition"] = {"P-L": {
            "n": 1, "deltaSquaredSum": 0.0, "crossTermSum": 0.0,
            "squaredErrorDifferenceSum": 0.0, "identityMaxAbsError": 0.0}}
    if variant == "Q" and phase == "V":
        fit["pqIdentity"] = {"sameRepresentativeKeyDigest": True, "sameInputWeightDigest": True,
                             "sameBinningIdentity": True}
        fit["permutationDiagnostics"] = {"blockCount": 0, "unchangedBlocks": 0, "changedRows": 0,
                                         "auxiliaryRows": 0, "changedAuxiliaryWeight": 0.0,
                                         "auxiliaryWeight": 0.0, "changedWeightedShare": 0.0}
        fit["representativeTrainLogDecomposition"] = {"P-Q": {
            "n": 1, "deltaSquaredSum": 0.0, "crossTermSum": 0.0,
            "squaredErrorDifferenceSum": 0.0, "identityMaxAbsError": 0.0}}
    if variant == "Q" and phase == "T":
        fit.update({"status": "not_run", "representativeTrainLogDecomposition": {"P-Q": None}})
        for key in ("representativeTrainDiagnostics", "support", "originWeightDiagnostics", "timeIntegrity"):
            fit.pop(key)
    return fit


def test_producer_shaped_no_auxiliary_v_fits_pass_fit_diagnostic_checks():
    fits = [_no_support_fit("L", "complete"), _no_support_fit("P", "aliased_no_support"),
            _no_support_fit("Q", "aliased_no_support")]
    diagnostic_ok, permutation_ok, decomposition_ok = scorer._fit_diagnostic_integrity(fits, "V")
    assert diagnostic_ok is True
    assert permutation_ok is True
    assert decomposition_ok is True
    assert scorer._representative_key_identity(fits, "V") is True
    assert scorer._origin_weight_integrity(fits) is True
    assert len(scorer._train_log_decomposition(fits)) == 2


def test_producer_shaped_t_fits_with_not_run_q_pass_fit_diagnostic_checks():
    fits = [_no_support_fit("L", "complete", phase="T"),
            _no_support_fit("P", "aliased_no_support", phase="T"),
            _no_support_fit("Q", "not_run", phase="T")]
    diagnostic_ok, permutation_ok, decomposition_ok = scorer._fit_diagnostic_integrity(fits, "T")
    assert diagnostic_ok is True
    assert permutation_ok is True
    assert decomposition_ok is True
    assert scorer._representative_key_identity(fits, "T") is True
    assert scorer._origin_weight_integrity(fits) is True
    assert len(scorer._train_log_decomposition(fits)) == 1


def test_v_phase_excluded_april_first_actual_survives_json_roundtrip():
    from agri_data import intra_crop_panel as producer

    start, end = scorer.PERIODS["V"]
    first_legal = scorer.ORIGIN_STARTS["V"] + dt.timedelta(days=1)
    forecasts = [{"product": product, "target": (first_legal + dt.timedelta(days=offset)).isoformat()}
                 for product in scorer.PRODUCTS
                 for offset in range((end-first_legal).days + 1)]
    actuals = {("cabbage", start): 100.0, ("cabbage", first_legal): 101.0}
    coverage = scorer._expected_phase_actual_coverage("V", forecasts, actuals)
    coverage.pop("_gridMatches")
    producer_coverage = producer._verify_phase_actual_coverage("V", forecasts, actuals)
    assert scorer._canonical(coverage) == scorer._canonical(producer_coverage)
    serialized = json.loads(scorer._canonical(coverage))
    assert serialized["excludedPositiveActualKeys"] == [["cabbage", "2026-04-01"]]
    assert scorer._canonical(serialized) == scorer._canonical(producer_coverage)


def test_external_evidence_chain_accepts_bound_artifacts(tmp_path, monkeypatch):
    fixture = _external_evidence_fixture(tmp_path, monkeypatch)
    manifest, prices, manifest_path, prices_path = fixture[:4]
    checks = scorer._verify_external_artifacts(manifest, prices,
                                               manifest_path=manifest_path, prices_path=prices_path)
    assert checks == {
        "sourceCacheHashesVerified": True,
        "baselineKeysAndActualsVerified": True,
        "forecastReplayVerified": True,
        "resolvedConfigVerified": True,
    }


def test_external_evidence_chain_rejects_changed_forecast_rows(tmp_path, monkeypatch):
    manifest, prices, manifest_path, prices_path, _, _, _ = _external_evidence_fixture(tmp_path, monkeypatch)
    manifest["forecasts"][0]["models"]["P"]["rawModelPrediction"] = 111.0
    manifest["forecasts"][0]["models"]["P"]["prediction"] = 111.0
    manifest["manifestSha256"] = scorer._sha(scorer._canonical(
        {key: value for key, value in manifest.items() if key != "manifestSha256"}))
    manifest_path.write_bytes(scorer._canonical(manifest))
    with pytest.raises(ValueError, match="replay audit does not verify"):
        scorer._verify_external_artifacts(manifest, prices,
                                          manifest_path=manifest_path, prices_path=prices_path)


def test_external_evidence_chain_rejects_pinned_baseline_value_change(tmp_path, monkeypatch):
    manifest, prices, manifest_path, prices_path, baseline_paths, baseline_hashes, _ = \
        _external_evidence_fixture(tmp_path, monkeypatch)
    doc = json.loads(baseline_paths["T"].read_text())
    doc["forecasts"][0]["conditions"]["C"]["blendPrediction"] = 99.0
    baseline_paths["T"].write_bytes(scorer._canonical(doc))
    baseline_hashes["T"] = scorer._sha(baseline_paths["T"].read_bytes())
    from agri_data import chronos2_market_context as rhs
    monkeypatch.setattr(rhs, "BASELINE_SHA256", baseline_hashes)
    manifest["baselineManifestSha256"] = baseline_hashes["T"]
    manifest["manifestSha256"] = scorer._sha(scorer._canonical(
        {key: value for key, value in manifest.items() if key != "manifestSha256"}))
    manifest_path.write_bytes(scorer._canonical(manifest))
    with pytest.raises(ValueError, match="stored E106_RHS baseline differs"):
        scorer._verify_external_artifacts(manifest, prices,
                                          manifest_path=manifest_path, prices_path=prices_path)


def test_external_evidence_chain_rejects_config_identity_change(tmp_path, monkeypatch):
    manifest, prices, manifest_path, prices_path, _, _, _ = _external_evidence_fixture(tmp_path, monkeypatch)
    manifest["resolvedConfigSha256"] = "0" * 64
    manifest["manifestSha256"] = scorer._sha(scorer._canonical(
        {key: value for key, value in manifest.items() if key != "manifestSha256"}))
    manifest_path.write_bytes(scorer._canonical(manifest))
    with pytest.raises(ValueError, match="resolved configuration or source-cache"):
        scorer._verify_external_artifacts(manifest, prices,
                                          manifest_path=manifest_path, prices_path=prices_path)


def test_log_delta_decomposition_has_p_minus_l_direction():
    actual = 110.0
    p0 = 100.0
    l_log = 0.0
    p_log = math.log(1.1)
    row = {"actual": actual, "models": {
        "L": {"logRatio": l_log, "p0": p0},
        "P": {"rawModelLogRatio": p_log, "p0": p0},
    }}

    result = scorer._log_decomposition([row], "L", "P_raw")
    direct_difference = (math.log(actual/p0)-p_log)**2 - (math.log(actual/p0)-l_log)**2

    assert result["leftModel"] == "L"
    assert result["rightModel"] == "P_raw"
    assert result["squaredErrorDifference"] == pytest.approx(direct_difference)
    assert result["squaredErrorDifference"] < 0
    assert result["deltaSquared"] + result["minusTwoResidualDelta"] == pytest.approx(direct_difference)


def test_test_phase_keeps_july_first_target_for_all_horizons():
    start, end = scorer.PERIODS["T"]
    rows = []
    for product in scorer.PRODUCTS:
        for horizon in scorer.HORIZONS:
            for offset in range((end-start).days + 1):
                target = start + dt.timedelta(days=offset)
                origin = target - dt.timedelta(days=horizon)
                rows.append({
                    "product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                    "horizon": horizon, "fitMonth": origin.strftime("%Y-%m"),
                    "B": {
                        "E106_RHS": {"status": "issued", "prediction": 100.0},
                        "E106": {"status": "issued", "prediction": 100.0},
                    },
                    "models": {
                        "L": {"status": "issued", "prediction": 100.0, "logRatio": 0.0,
                              "p0": 100.0, "p0Date": origin.isoformat()},
                        "P": {"status": "issued", "prediction": 100.0, "logRatio": 0.0,
                              "rawModelLogRatio": 0.0, "p0": 100.0, "p0Date": origin.isoformat(),
                              "modelUsed": True, "rawModelStatus": "issued", "rawModelPrediction": 100.0},
                    },
                })
    doc = {"phase": "T", "targetValuesAttached": False, "forecasts": rows}
    doc["manifestSha256"] = hashlib.sha256(scorer._canonical(doc)).hexdigest()

    assert scorer._validate_manifest(doc) == "T"
