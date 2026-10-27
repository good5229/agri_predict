import copy
import datetime as dt
import math
import json
from pathlib import Path
import shutil
import tempfile

import pytest

from agri_data import crop_specialist_assembly_score as score


def _seal(doc, field):
    unsigned = dict(doc)
    unsigned.pop(field, None)
    doc[field] = score._sha(score._canonical(unsigned))
    return doc


@pytest.fixture
def bundle():
    selected = {"cabbage": "B", "radish": "P", "onion": "A", "garlic": "B"}
    planned = {"cabbage": ["L"], "radish": ["L", "P"],
               "onion": ["A", "CTRL"], "garlic": []}
    selection = {"experiment": "CROP_SPECIALIST_ASSEMBLY_V1", "stage": "V",
                 "targetVersion": "target-snapshot-sha", "targetSnapshot": {"path": "prices.json", "sha256": "target-snapshot-sha"},
                 "products": {p: {"selectedModelId": model} for p, model in selected.items()},
                 "sources": {name: {"path": f"{name}.json", "sha256": f"{name}-sha"} for name in ("A", "B", "L", "P")}}
    _seal(selection, "selectionSha256")
    rows, products, missing_actual = [], [], None
    obs = {p: {} for p in score.PRODUCTS}
    for i in range((score.END-score.START).days + 1):
        target = score.START + dt.timedelta(days=i)
        for product in score.PRODUCTS:
            value = 100.0 + i
            obs[product][target.isoformat()] = value
    missing_actual = ("radish", score.START.isoformat())
    del obs[missing_actual[0]][missing_actual[1]]
    for product in score.PRODUCTS:
        products.append({"id": product, "observations": [
            {"date": day, "actual": value} for day, value in sorted(obs[product].items())]})
    prices = {"schemaVersion": 1, "products": products}
    for product in score.PRODUCTS:
        for horizon in score.HORIZONS:
            for i in range((score.END-score.START).days + 1):
                target = score.START + dt.timedelta(days=i)
                origin = target - dt.timedelta(days=horizon)
                if origin < score.ORIGIN_START:
                    continue
                base = 100.0 + i
                b = {"status": "issued", "prediction": base}
                direct_prediction = base + 1.0
                models = {"P": {"status": "issued", "prediction": base * math.exp(0.01), "rawModelStatus": "issued",
                                "logRatio": 0.01, "p0": base, "p0Date": (origin-dt.timedelta(days=1)).isoformat()},
                          "L": {"status": "issued", "prediction": base * math.exp(-0.01), "logRatio": -0.01,
                                "p0": base, "p0Date": (origin-dt.timedelta(days=1)).isoformat()},
                          "A": {"status": "issued", "prediction": base * math.exp(0.02), "logRatio": 0.02,
                                "p0": base, "p0Date": (origin-dt.timedelta(days=1)).isoformat()},
                          "CTRL": {"status": "issued", "prediction": base * math.exp(-0.02), "logRatio": -0.02,
                                   "p0": base, "p0Date": (origin-dt.timedelta(days=1)).isoformat()}}
                if product == "radish" and i == 0 and horizon == 1:
                    p_obj = {"status": "not_issued", "prediction": None, "rawModelStatus": "not_issued"}
                    models["P"] = p_obj
                    g = {"status": "fallback", "prediction": base, "selectedModelId": "P",
                         "rawModelStatus": "not_issued", "rawModelPrediction": None,
                         "fallbackTo": "B", "fallbackReason": "numerical_output_unavailable",
                         "modelUsed": False}
                else:
                    direct_prediction = base if selected[product] == "B" else models[selected[product]]["prediction"]
                    g = {"status": "issued", "prediction": direct_prediction,
                         "selectedModelId": selected[product],
                         "rawModelStatus": "not_applicable" if selected[product] == "B" else "issued",
                         "rawModelPrediction": None if selected[product] == "B" else direct_prediction,
                         "fallbackTo": None, "modelUsed": selected[product] != "B"}
                rows.append({"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                             "horizon": horizon, "target_version": "target-snapshot-sha",
                             "fit_month": origin.strftime("%Y-%m"),
                             "cutoff": (origin-dt.timedelta(days=1)).isoformat(),
                             "selected_model_id": selected[product],
                             "selected_policy_id": f"G:{product}:{selected[product]}",
                             "B": b, "models": {name: models[name] for name in planned[product]}, "G": g})
    forecast = {"experiment": "CROP_SPECIALIST_ASSEMBLY_V1", "stage": "T",
                "targetValuesAttached": False, "targetVersion": "target-snapshot-sha",
                "plannedModelIdsByProduct": planned,
                "executionIdentitySha256": "execution-sha", "forecasts": rows}
    forecast["forecastRowsSha256"] = score._sha(score._canonical(rows))
    _seal(forecast, "manifestSha256")
    replay = {"status": "pass", "forecastsFileSha256": score._sha(score._canonical(forecast)),
              "forecastRowsSha256": forecast["forecastRowsSha256"],
              "selectionSha256": selection["selectionSha256"],
              "executionIdentitySha256": "execution-sha"}
    _seal(replay, "replayVerificationSha256")
    return selection, forecast, replay, prices, missing_actual


def test_fixed_families_cover_80_and_40_tests_and_report_counts(bundle):
    selection, forecast, replay, prices, _ = bundle
    result = score.score_documents(selection, forecast, replay, prices,
                                   forecast_file_sha=replay["forecastsFileSha256"])
    assert result["inference"]["primary"]["testsPerBandwidth"] == 80
    assert result["inference"]["secondary"]["testsPerBandwidth"] == 40
    assert all(len([r for r in result["inference"]["primary"]["rows"] if r["bandwidth"] == b]) == 80
               for b in score.BANDWIDTHS)
    assert all(len([r for r in result["inference"]["secondary"]["rows"] if r["bandwidth"] == b]) == 40
               for b in score.BANDWIDTHS)
    assert result["counts"]["forecastPairs"] == 3160
    assert result["counts"]["uniqueOrigins"] > 0
    assert result["counts"]["uniqueTargets"] == 79
    assert result["secondaryDiagnostics"]["comparisonDirections"] == {
        "radish": "P minus L", "onion": "A minus CTRL"}
    assert result["jointCalendarBlockBootstrap"]["metrics"] == ["MAE", "MAPE"]
    assert set(result["jointCalendarBlockBootstrap"]["results"]) == {"MAE", "MAPE"}
    assert set(result["primaryGvsB"]["byProduct10HorizonMacro"]) == set(score.PRODUCTS)
    assert result["primaryGvsB"]["directSelectedVsB"]["selectedProducts"] == ["radish", "onion"]
    assert result["noOperationalPromotion"] is True
    assert len(result["targetEvaluation"]) == 3160
    assert result["targetEvaluation"][0]["targetVersion"] == selection["targetVersion"]
    assert result["targetEvaluation"][0]["targetObservationStatus"] == "positive_actual"
    assert {row["targetObservationStatus"] for row in result["targetEvaluation"]} >= {"no_observation"}


def test_fallback_and_missing_actual_preserve_issuance_coverage(bundle):
    selection, forecast, replay, prices, missing = bundle
    result = score.score_documents(selection, forecast, replay, prices,
                                   forecast_file_sha=replay["forecastsFileSha256"])
    radish = result["coverage"]["radish"]
    assert radish["fallback"] == 1
    assert radish["baselineIssued"] == radish["policyIssued"]
    assert radish["positiveActualBaselineIssued"] == radish["baselineIssued"] - 10
    assert radish["positiveActualPolicyIssued"] == radish["positiveActualBaselineIssued"]
    assert result["counts"]["positiveActualPairs"] == 3150
    assert missing[1] == "2026-07-11"


def test_malformed_status_prediction_and_replay_binding_fail_closed(bundle):
    selection, forecast, replay, prices, _ = bundle
    broken = copy.deepcopy(forecast)
    broken["forecasts"][0]["B"] = {"status": "not_issued", "prediction": 100.0}
    broken["forecastRowsSha256"] = score._sha(score._canonical(broken["forecasts"]))
    _seal(broken, "manifestSha256")
    with pytest.raises(ValueError, match="not_issued status"):
        score.score_documents(selection, broken, replay, prices,
                              forecast_file_sha=score._sha(score._canonical(broken)))
    replay2 = copy.deepcopy(replay)
    replay2["forecastRowsSha256"] = "wrong"
    _seal(replay2, "replayVerificationSha256")
    with pytest.raises(ValueError, match="replay audit"):
        score.score_documents(selection, forecast, replay2, prices,
                              forecast_file_sha=replay["forecastsFileSha256"])


def test_secondary_price_evidence_must_be_causal_and_reconstruct_prediction(bundle):
    selection, forecast, replay, prices, _ = bundle
    broken = copy.deepcopy(forecast)
    row = next(r for r in broken["forecasts"] if r["product"] == "radish"
               and r["models"]["P"]["status"] == "issued")
    row["models"]["P"]["p0Date"] = row["origin"]
    broken["forecastRowsSha256"] = score._sha(score._canonical(broken["forecasts"]))
    _seal(broken, "manifestSha256")
    replay2 = {**replay, "forecastRowsSha256": broken["forecastRowsSha256"],
               "forecastsFileSha256": score._sha(score._canonical(broken))}
    _seal(replay2, "replayVerificationSha256")
    with pytest.raises(ValueError, match="causal log-price evidence"):
        score.score_documents(selection, broken, replay2, prices,
                              forecast_file_sha=score._sha(score._canonical(broken)))


def test_unexplained_coverage_loss_is_a_policy_integrity_failure(bundle):
    selection, forecast, replay, prices, _ = bundle
    broken = copy.deepcopy(forecast)
    row = next(r for r in broken["forecasts"] if r["product"] == "cabbage")
    row["G"] = {"status": "not_issued", "prediction": None, "selectedModelId": "B",
                "rawModelStatus": "not_applicable", "fallbackTo": None, "modelUsed": False}
    broken["forecastRowsSha256"] = score._sha(score._canonical(broken["forecasts"]))
    _seal(broken, "manifestSha256")
    replay2 = {**replay, "forecastsFileSha256": score._sha(score._canonical(broken)),
               "forecastRowsSha256": score._sha(score._canonical(broken["forecasts"]))}
    _seal(replay2, "replayVerificationSha256")
    with pytest.raises(ValueError, match="B-selected G policy differs"):
        score.score_documents(selection, broken, replay2, prices,
                                   forecast_file_sha=replay2["forecastsFileSha256"])


def test_artifact_reference_is_rehashed_from_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "ROOT", tmp_path)
    path = tmp_path / "pinned.json"
    path.write_text('{"value": 1}\n', encoding="utf-8")
    digest = score._sha(path.read_bytes())
    assert score._verify_ref({"path": "pinned.json", "fileSha256": digest}) == path
    with pytest.raises(ValueError, match="file hash mismatch"):
        score._verify_ref({"path": "pinned.json", "fileSha256": "self-asserted-wrong"})


def test_target_observation_states_preserve_missing_null_and_negative_values():
    states = score._target_observations({"schemaVersion": 1, "products": [{"id": "radish", "observations": [
        {"date": "2026-07-11", "actual": None},
        {"date": "2026-07-12", "actual": -2.5},
        {"date": "2026-07-13", "actual": 0},
        {"date": "2026-07-14", "actual": "12.30"},
        {"date": "2026-07-15"},
    ]}]})
    assert states[("radish", "2026-07-11")][0] == {"status": "null_actual", "actualRaw": None}
    assert states[("radish", "2026-07-12")][0] == {"status": "negative_actual", "actualRaw": -2.5}
    assert states[("radish", "2026-07-13")][0]["status"] == "zero_actual"
    assert states[("radish", "2026-07-14")][0] == {"status": "positive_actual", "actualRaw": "12.30"}
    assert states[("radish", "2026-07-15")][0]["status"] == "actual_field_missing"


def test_score_files_binds_execution_paths_and_native_sidecars(bundle, monkeypatch):
    from agri_data import chronos2_market_context as rhs
    from agri_data import crop_specialist_assembly as producer
    from agri_data import external_reference_v1 as source_a
    from agri_data import full_history_return_eval as price_helper
    from agri_data import intra_crop_panel as source_b

    selection_path = producer.OUT / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = {p: selection["products"][p]["selectedModelId"] for p in score.PRODUCTS}
    monkeypatch.setattr(score, "_validate_selection", lambda *_args, **_kwargs: selected)
    _, forecast, _, _, _ = bundle
    forecast = copy.deepcopy(forecast)
    forecast["targetVersion"] = selection["targetVersion"]
    forecast["plannedModelIdsByProduct"] = {
        p: selection["plannedFitsByProduct"][p]["models"] for p in score.PRODUCTS}
    for row in forecast["forecasts"]:
        row["target_version"] = selection["targetVersion"]
        product = row["product"]
        model_id = selected[product]
        row["selected_model_id"] = model_id
        row["selected_policy_id"] = f"G:{product}:{model_id}"
        if model_id != "B":
            model = row["models"][model_id]
            if model.get("status") == "issued":
                row["G"] = {"status": "issued", "prediction": model["prediction"],
                            "selectedModelId": model_id, "rawModelStatus": "issued",
                            "rawModelPrediction": model["prediction"], "fallbackTo": None,
                            "fallbackReason": None, "modelUsed": True}
            else:
                row["G"] = {"status": "fallback", "prediction": row["B"]["prediction"],
                            "selectedModelId": model_id, "rawModelStatus": "not_issued",
                            "rawModelPrediction": None, "fallbackTo": "B",
                            "fallbackReason": "offline fixture selected output unavailable",
                            "modelUsed": False}

    runtime_paths = {
        "targetSnapshot": Path(price_helper.PRICES_PATH),
        "resolvedConfig": source_b.OUT / "resolved-config.json",
        "sourceCache": source_b.SOURCE_CACHE,
        "preflight": source_b.OUT / "preflight.json",
        "pinnedRhsBaseline": Path(rhs.BASELINE_PATHS["T"]),
    }
    for name, path in runtime_paths.items():
        ref = {"path": str(path), "fileSha256": score._sha(path.read_bytes())}
        if name == "targetSnapshot":
            forecast[name] = ref
        elif name == "pinnedRhsBaseline":
            ref["pinnedSha256"] = rhs.BASELINE_SHA256["T"]
            forecast[name] = ref
        elif name == "resolvedConfig":
            ref["configSha256"] = json.loads(path.read_text())["configSha256"]
            forecast[name] = ref
        elif name == "sourceCache":
            ref["cacheSha256"] = source_b.SOURCE_CACHE_INTERNAL_SHA256
            forecast[name] = ref
        else:
            ref["preflightSha256"] = json.loads(path.read_text())["preflightSha256"]
            forecast[name] = ref
    source_paths = {
        "A": source_a.OUT / "V/forecast-manifest.json",
        "B": source_b.OUT / "V/forecast-manifest.json",
        "L": source_b.OUT / "V/forecast-manifest.json",
        "P": source_b.OUT / "V/forecast-manifest.json",
    }
    forecast["sources"] = {}
    for name, path in source_paths.items():
        source_doc = json.loads(path.read_text(encoding="utf-8"))
        forecast["sources"][name] = {"path": str(path), "fileSha256": score._sha(path.read_bytes()),
                                      "manifestSha256": source_doc["manifestSha256"]}
    forecast["selection"] = {"path": str(selection_path),
                              "fileSha256": score._sha(selection_path.read_bytes()),
                              "selectionSha256": selection["selectionSha256"]}
    config_doc = json.loads((source_b.OUT / "resolved-config.json").read_text(encoding="utf-8"))
    forecast["executionIdentity"] = producer._execution_identity(
        config_doc, score._sha(source_b.SOURCE_CACHE.read_bytes()), selection["selectionSha256"])
    forecast["executionIdentitySha256"] = forecast["executionIdentity"]["executionIdentitySha256"]

    # Build tiny, deterministic placeholder checkpoints for each planned fold. This
    # exercises persisted path/hash/identity validation without fitting any model.
    with tempfile.TemporaryDirectory(prefix=".score-files-test-", dir=score.ROOT / "outputs") as temp:
        run_root = Path(temp)
        artifact_by_fold = {}
        fits = []
        fold_months = sorted({(r["product"], r["horizon"], r["fit_month"])
                              for r in forecast["forecasts"]})
        for product, horizon, month in fold_months:
            models = forecast["plannedModelIdsByProduct"][product]
            for model_id in models:
                artifact_key = (product, horizon, month,
                                "A/CTRL" if product == "onion" else model_id)
                artifact = artifact_by_fold.get(artifact_key)
                if artifact is None:
                    suffix = ".joblib" if artifact_key[-1] == "A/CTRL" else ".txt"
                    rel = f"T/models/{product}-h{horizon:02d}-{month}-{artifact_key[-1]}{suffix}"
                    artifact_path = run_root / rel
                    artifact_path.parent.mkdir(parents=True, exist_ok=True)
                    content = f"offline-checkpoint:{artifact_key}".encode()
                    artifact_path.write_bytes(content)
                    identity_doc = {"artifactKey": list(artifact_key)}
                    identity_sha = score._sha(score._canonical(identity_doc))
                    identity_path = artifact_path.with_suffix(".identity.json")
                    sidecar = {"identity": identity_doc, "identitySha256": identity_sha,
                               "modelTextSha256" if suffix == ".txt" else "checkpointFileSha256":
                                   score._sha(content)}
                    identity_path.write_bytes(score._canonical(sidecar))
                    artifact = {"product": product, "horizon": horizon, "fitMonth": month,
                                "path": rel, "fileSha256": score._sha(content),
                                "identityPath": str(identity_path.relative_to(run_root)),
                                "identityFileSha256": score._sha(identity_path.read_bytes()),
                                "identitySha256": identity_sha, "checkpointIdentity": identity_doc}
                    if artifact_key[-1] == "A/CTRL":
                        artifact["modelIds"] = ["A", "CTRL"]
                    else:
                        artifact["modelId"] = model_id
                    artifact_by_fold[artifact_key] = artifact
                fit = {"product": product, "modelId": model_id, "horizon": horizon,
                       "fitMonth": month, "fitOrigin": month + "-01",
                       "fitTargetMax": (dt.date.fromisoformat(month + "-01")-dt.timedelta(days=1)).isoformat(),
                       "trainingTargetsBeforeFitOrigin": True, "allFeatureDatesAtOrBeforeCutoff": True,
                       "fitStatus": "reused", "fitTargetCount": 10,
                       "fitTargetSha256": "a"*64,
                       "checkpointFileSha256": artifact["fileSha256"],
                       "checkpointIdentitySha256": artifact["identitySha256"],
                       "representativeTrainDiagnostics": {"n": 10, "logMse": 0.0, "logMae": 0.0,
                                                           "priceMAE": 0.0, "priceMAPE": 0.0},
                       "representativeTrainLogDecomposition": {}}
                if product == "radish" and model_id == "P":
                    fit["representativeTrainLogDecomposition"] = {"P-L": {
                        "n": 10, "deltaSquaredSum": 0.0, "crossTermSum": 0.0,
                        "squaredErrorDifferenceSum": 0.0, "identityMaxAbsError": 0.0}}
                if product == "onion" and model_id == "A":
                    fit["representativeTrainLogDecomposition"] = {"A-CTRL": {
                        "n": 10, "deltaSquaredSum": 0.0, "crossTermSum": 0.0,
                        "squaredErrorDifferenceSum": 0.0, "identityMaxAbsError": 0.0}}
                fits.append(fit)
        artifacts = sorted(artifact_by_fold.values(), key=lambda a: (a["product"], a["fitMonth"], a["horizon"],
                                                                         a.get("modelId", "/".join(a.get("modelIds", [])))))
        forecast["fitDiagnostics"] = fits
        forecast["modelArtifacts"] = artifacts
        forecast["plannedFitCount"] = len(fits)
        forecast["actualEstimatorFitCount"] = len(fits)
        forecast["newEstimatorFitCount"] = 0
        forecast["forecastKeyCount"] = len(forecast["forecasts"])
        forecast["replayVerificationPath"] = "T/replay-verification.json"
        forecast["forecastRowsSha256"] = score._sha(score._canonical(forecast["forecasts"]))
        _seal(forecast, "manifestSha256")
        forecast_path = run_root / "T/forecasts.json"
        forecast_path.parent.mkdir(parents=True, exist_ok=True)
        forecast_path.write_bytes(score._canonical(forecast))
        forecast_file_sha = score._sha(forecast_path.read_bytes())
        replay = {"status": "pass", "forecastsFileSha256": forecast_file_sha,
                  "manifestSha256": forecast["manifestSha256"],
                  "forecastRowsSha256": forecast["forecastRowsSha256"],
                  "selectionFileSha256": score._sha(selection_path.read_bytes()),
                  "selectionSha256": selection["selectionSha256"],
                  "executionIdentitySha256": forecast["executionIdentitySha256"],
                  "sourceCacheFileSha256": forecast["sourceCache"]["fileSha256"],
                  "resolvedConfigFileSha256": forecast["resolvedConfig"]["fileSha256"],
                  "baselineFileSha256": forecast["pinnedRhsBaseline"]["fileSha256"],
                  "targetSnapshotFileSha256": forecast["targetSnapshot"]["fileSha256"],
                  "sourceManifestHashes": {name: ref["fileSha256"] for name, ref in forecast["sources"].items()},
                  "checkpointArtifacts": [{key: a.get(key) for key in ("path", "fileSha256", "identityPath",
                                                                      "identityFileSha256", "identitySha256")}
                                          for a in artifacts]}
        _seal(replay, "replayVerificationSha256")
        replay_path = run_root / "T/replay-verification.json"
        replay_path.write_bytes(score._canonical(replay))
        result = score.score_files(selection_path, forecast_path, replay_path)
        assert result["status"] == "completed"
        assert result["noOperationalPromotion"] is True
        assert result["primaryGvsB"]["directSelectedVsB"]["selectedProducts"] == ["cabbage", "radish", "onion"]
        assert result["inputProvenance"]["forecasts"]["fileSha256"] == forecast_file_sha
        assert result["inputProvenance"]["replay"]["fileSha256"] == score._sha(replay_path.read_bytes())
        assert result["inputProvenance"]["priceSnapshot"]["fileSha256"] == price_helper.PRICES_SHA256
        assert len(result["targetEvaluation"]) == len(forecast["forecasts"])

