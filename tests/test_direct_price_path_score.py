import datetime as dt

import pytest

from agri_data import direct_price_path_score as scorer


def _row(product="cabbage", origin=dt.date(2026, 1, 1), horizon=1):
    target = origin + dt.timedelta(days=horizon)
    return {
        "product": product, "origin": origin, "target": target,
        "horizon": horizon, "target_version": "synthetic-target-v1",
        "selectedInput": "U", "neuralUsed": True,
        "B": {"status": "issued", "prediction": 10.0, "sourceStatus": "issued"},
        "U": {"status": "issued", "prediction": 11.0,
              "rawPrediction": 11.0, "rawNumericKind": "issued_positive", "rawStatus": "issued", "rawReason": None},
        "M": {"status": "issued", "prediction": 9.0,
              "rawPrediction": 9.0, "rawNumericKind": "issued_positive", "rawStatus": "issued", "rawReason": None},
        "N": {"status": "issued", "prediction": 11.0,
              "rawPrediction": 11.0, "rawNumericKind": "issued_positive", "rawStatus": "issued", "rawReason": None},
        "H": {"status": "issued", "prediction": 10.5, "rawPrediction": 10.5, "rawNumericKind": "issued_positive", "rawStatus": "issued", "rawReason": None},
        "H_U": {"status": "issued", "prediction": 10.5, "rawPrediction": 10.5, "rawNumericKind": "issued_positive", "rawStatus": "issued", "rawReason": None},
        "H_M": {"status": "issued", "prediction": 9.5, "rawPrediction": 9.5, "rawNumericKind": "issued_positive", "rawStatus": "issued", "rawReason": None},
    }


def _small_grid(monkeypatch):
    monkeypatch.setattr(scorer, "PRODUCTS", ("cabbage",))
    monkeypatch.setattr(scorer, "HORIZONS", (1,))
    day = dt.date(2026, 1, 2)
    monkeypatch.setattr(scorer, "PHASES", {"V": (day, day, dt.date(2026, 1, 1))})
    row = _row()
    baseline = {("cabbage", day, 1): {"origin": dt.date(2026, 1, 1),
                                      "target": day, "E106_RHS": 10.0, "rhsStatus": "issued"}}
    return [row], baseline


def test_validates_exact_saved_baseline_key_and_blends(monkeypatch):
    rows, baseline = _small_grid(monkeypatch)
    scorer.validate_phase_rows("V", rows, baseline, "synthetic-target-v1")


def test_rejects_missing_key_and_wrong_origin(monkeypatch):
    rows, baseline = _small_grid(monkeypatch)
    with pytest.raises(ValueError, match="row count"):
        scorer.validate_phase_rows("V", [], baseline, "synthetic-target-v1")
    rows[0]["origin"] = dt.date(2026, 1, 2)
    with pytest.raises(ValueError, match="unexpected or duplicate"):
        scorer.validate_phase_rows("V", rows, baseline, "synthetic-target-v1")


def test_rejects_hybrid_arithmetic_and_missing_fallback_provenance(monkeypatch):
    rows, baseline = _small_grid(monkeypatch)
    rows[0]["H"]["prediction"] = 10.4
    with pytest.raises(ValueError, match="50/50 blend"):
        scorer.validate_phase_rows("V", rows, baseline, "synthetic-target-v1")
    rows, baseline = _small_grid(monkeypatch)
    rows[0]["U"] = {"status": "fallback", "prediction": 10.0, "fallbackTo": "B", "reason": "invalid raw output",
                     "rawPrediction": -2.0, "rawNumericKind": "negative_finite", "rawStatus": "not_issued", "rawReason": "invalid raw output"}
    rows[0]["N"] = dict(rows[0]["U"])
    rows[0]["neuralUsed"] = False
    rows[0]["H"] = {"status": "fallback", "prediction": 10.0,
                     "fallbackTo": "B", "reason": "invalid raw output",
                     "rawPrediction": -2.0, "rawNumericKind": "negative_finite", "rawStatus": "not_issued", "rawReason": "invalid raw output"}
    rows[0]["H_U"] = dict(rows[0]["H"])
    scorer.validate_phase_rows("V", rows, baseline, "synthetic-target-v1")
    rows[0]["H"]["fallbackTo"] = "wrong"
    with pytest.raises(ValueError, match="fallback provenance"):
        scorer.validate_phase_rows("V", rows, baseline, "synthetic-target-v1")
    rows, baseline = _small_grid(monkeypatch)
    failed_nan = {"status": "fallback", "prediction": 10.0, "fallbackTo": "B",
                  "reason": "nonfinite_prediction", "rawPrediction": None,
                  "rawNumericKind": "nan", "rawStatus": "not_issued",
                  "rawReason": "nonfinite_prediction"}
    rows[0]["U"] = dict(failed_nan)
    rows[0]["N"] = dict(failed_nan)
    rows[0]["H"] = dict(failed_nan)
    rows[0]["H_U"] = dict(failed_nan)
    rows[0]["neuralUsed"] = False
    scorer.validate_phase_rows("V", rows, baseline, "synthetic-target-v1")
    rows[0]["U"]["rawNumericKind"] = "negative_finite"
    rows[0]["N"]["rawNumericKind"] = "negative_finite"
    rows[0]["H"]["rawNumericKind"] = "negative_finite"
    rows[0]["H_U"]["rawNumericKind"] = "negative_finite"
    with pytest.raises(ValueError, match="lacks a rawPrediction"):
        scorer.validate_phase_rows("V", rows, baseline, "synthetic-target-v1")


def test_canonical_digests_are_stable_for_synthetic_rows():
    row = _row()
    first = scorer._row_digest([row])
    assert first == scorer._row_digest([dict(row)])
    assert len(first) == 64
    assert len(scorer._key_digest([row])) == 64


def test_target_snapshot_bytes_and_version_must_both_match_pin(monkeypatch):
    pinned = "a" * 64
    monkeypatch.setattr(scorer.price_helper, "PRICES_SHA256", pinned)
    scorer._validate_pinned_target_ref({"sha256": pinned, "targetVersion": pinned})
    with pytest.raises(ValueError, match="both match"):
        scorer._validate_pinned_target_ref({"sha256": "b" * 64, "targetVersion": pinned})
    with pytest.raises(ValueError, match="both match"):
        scorer._validate_pinned_target_ref({"sha256": pinned, "targetVersion": "b" * 64})


def test_fit_grid_requires_all_24_fits_and_first_eligible_origin(tmp_path, monkeypatch):
    products = ("cabbage", "radish", "onion", "garlic")
    monkeypatch.setattr(scorer, "PRODUCTS", products)
    rows = []
    fit_grid = []
    models_by_fit = {}
    months = ((4, 3), (5, 5), (6, 2))
    for product in products:
        for month, first_day in months:
            fit_month = f"2026-{month:02d}"
            origin = dt.date(2026, month, first_day)
            fit_ids = {kind: f"V-2026{month:02d}-{product}-{kind}" for kind in ("U", "M")}
            fit_records = {}
            for kind, fit_id in fit_ids.items():
                model_rel = f"models/{fit_id}.pt"
                checkpoint_rel = f"checkpoints/{fit_id}.pt"
                model_path = tmp_path / model_rel
                checkpoint_path = tmp_path / checkpoint_rel
                model_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                model_path.write_bytes(f"model:{fit_id}".encode())
                checkpoint_path.write_bytes(f"checkpoint:{fit_id}".encode())
                model_sha = scorer.sha256(model_path.read_bytes())
                checkpoint_sha = scorer.sha256(checkpoint_path.read_bytes())
                fit = {"fitId": fit_id, "product": product, "inputType": kind,
                       "fitMonth": fit_month, "fitBoundary": origin.isoformat(),
                       "bestEpoch": 3, "fixedEpoch": 3, "modelPath": model_rel,
                       "modelSha256": model_sha, "checkpointPath": checkpoint_rel,
                       "checkpointSha256": checkpoint_sha, "contextDays": 30,
                       "repairSetting": "base", "selectedChannels": []}
                fit_grid.append(fit)
                fit_records[kind] = fit
                models_by_fit[fit_id] = {"pathBase": "experimentRoot", "path": model_rel,
                    "sha256": model_sha, "checkpointPath": checkpoint_rel,
                    "checkpointSha256": checkpoint_sha}
            rows.append({"product": product, "origin": origin, "target": origin + dt.timedelta(days=1),
                "horizon": 1, "selectedInput": "U", "fitMonth": fit_month,
                "fitId": fit_ids["U"], "activeAuxCount": 0,
                "U": {"status": "issued", "prediction": 11.0, "rawPrediction": 11.0,
                      "rawNumericKind": "issued_positive", "rawStatus": "issued", "rawReason": None},
                "M": {"status": "issued", "prediction": 9.0, "rawPrediction": 9.0,
                      "rawNumericKind": "issued_positive", "rawStatus": "issued", "rawReason": None},
                "outputEvidence": {role: {"fitId": fit_ids[kind],
                    "modelSha256": fit_records[kind]["modelSha256"]}
                    for role, kind in (("U", "U"), ("M", "M"), ("N", "U"))}})
    replay_by_fit = {}
    for fit in fit_grid:
        fit_id = fit["fitId"]
        payload_records = [{"product": fit["product"], "origin": rows[
            next(i for i, row in enumerate(rows) if row["product"] == fit["product"] and row["fitMonth"] == fit["fitMonth"])
        ]["origin"].isoformat(), "window": {"x": [1.0]}, "calendar": {"x": [0.0]},
            "targetDays": ["2026-01-01"] * 10}]
        payload_records[0]["targetDays"] = [
            (dt.date.fromisoformat(payload_records[0]["origin"]) + dt.timedelta(days=h)).isoformat()
            for h in range(1, 11)]
        schema = {"contextDays": fit["contextDays"], "inputType": fit["inputType"],
                  "repair": fit["repairSetting"], "selectedInputChannels": fit["selectedChannels"]}
        payload = {"fitId": fit_id, "modelSha256": fit["modelSha256"], "schema": schema,
            "schemaSha256": scorer.sha256(scorer.canonical(schema)), "records": payload_records,
            "rowPayloadSha256": scorer._inference_payload_digest(payload_records), "rowPayloadCount": 1}
        rel = f"V/replay-inputs/{fit_id}.json"
        payload_path = tmp_path / rel
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_bytes(scorer.canonical(payload))
        evidence = {"fitId": fit_id, "modelPath": fit["modelPath"], "modelSha256": fit["modelSha256"],
            "schemaSha256": payload["schemaSha256"],
            "rowPayloadPath": rel, "rowPayloadPathBase": "experimentRoot",
            "rowPayloadFileSha256": scorer.sha256(payload_path.read_bytes()),
            "rowPayloadSha256": payload["rowPayloadSha256"], "rowPayloadCount": 1,
            "originalPredictionSha256": "d" * 64, "predictionSha256": "e" * 64,
            "reloadedStateDict": True, "stateAgreement": True, "stateKeyAgreement": True,
            "stateTensorCount": 1, "allRowsAgree": True, "predictionSource": "reloaded_saved_state_dict_and_persisted_input_schema",
            "issueClassAgreement": True,
            "inferenceRowCount": 1, "horizonCount": 10, "absoluteTolerance": 1e-6,
            "relativeTolerance": 1e-6, "maxOriginalMagnitude": 1.0,
            "maxAbsoluteDifference": 0.0, "maxRelativeDifference": 0.0,
            "forecastRowsSha256": scorer._forecast_fit_rows_digest(rows, fit)}
        evidence["replaySha256"] = scorer.sha256(scorer.canonical(evidence))
        fit["replayVerification"] = evidence
        replay_by_fit[fit_id] = evidence
    manifest = {"fitGrid": fit_grid, "modelsByFitId": models_by_fit,
                "replayVerification": {"schemaVersion": 1,
                    "method": "reload_saved_state_dict_then_infer_before_baseline_row_join",
                    "byFitId": replay_by_fit}}
    scorer._validate_fit_grid("V", manifest, rows, tmp_path)
    first_fit = fit_grid[0]
    first_evidence = first_fit["replayVerification"]
    first_evidence["issueClassAgreement"] = False
    unsigned = dict(first_evidence)
    unsigned.pop("replaySha256")
    first_evidence["replaySha256"] = scorer.sha256(scorer.canonical(unsigned))
    with pytest.raises(ValueError, match="prove reloaded-model agreement"):
        scorer._validate_fit_grid("V", manifest, rows, tmp_path)
    first_evidence["issueClassAgreement"] = True
    unsigned = dict(first_evidence)
    unsigned.pop("replaySha256")
    first_evidence["replaySha256"] = scorer.sha256(scorer.canonical(unsigned))
    saved_max_difference = first_evidence["maxAbsoluteDifference"]
    first_evidence["maxAbsoluteDifference"] = 1.0
    unsigned = dict(first_evidence)
    unsigned.pop("replaySha256")
    first_evidence["replaySha256"] = scorer.sha256(scorer.canonical(unsigned))
    with pytest.raises(ValueError, match="exceeds its declared tolerance"):
        scorer._validate_fit_grid("V", manifest, rows, tmp_path)
    first_evidence["maxAbsoluteDifference"] = saved_max_difference
    unsigned = dict(first_evidence)
    unsigned.pop("replaySha256")
    first_evidence["replaySha256"] = scorer.sha256(scorer.canonical(unsigned))
    scorer._validate_fit_grid("V", manifest, rows, tmp_path)
    with pytest.raises(ValueError, match="cover all 24"):
        scorer._validate_fit_grid("V", {**manifest, "fitGrid": fit_grid[:-1]}, rows, tmp_path)
    wrong_month = [dict(row) for row in rows]
    wrong_month[0]["fitId"] = "V-202605-cabbage-U"
    with pytest.raises(ValueError, match="selected monthly fit"):
        scorer._validate_fit_grid("V", manifest, wrong_month, tmp_path)
    wrong_boundary = [dict(fit) for fit in fit_grid]
    wrong_boundary[0]["fitBoundary"] = "2026-04-01"
    with pytest.raises(ValueError, match="first eligible origin"):
        scorer._validate_fit_grid("V", {**manifest, "fitGrid": wrong_boundary}, rows, tmp_path)


def test_transition_validation_binds_preserved_code_config_policy_dev_and_fits(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    out = repo / "outputs/experiments/direct-price-path-attention-v1-20261014"
    out.mkdir(parents=True)
    old_path = "agri_data/direct_price_path.py"
    new_path = "agri_data/direct_price_path_v2.py"
    data_path = "agri_data/direct_price_path_data.py"
    source_code_files = {}
    for rel, content in ((old_path, b"frozen-old-main"), (new_path, b"reviewed-v2-main"),
                         (data_path, b"frozen-data-source")):
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        source_code_files[rel] = scorer.sha256(content)
    config = {"sourceCodeFiles": {old_path: source_code_files[old_path],
                                   data_path: source_code_files[data_path]}}
    config["resolvedConfigSha256"] = scorer.sha256(scorer.canonical(config))
    config_raw = scorer.canonical(config)
    (out / "resolved-config.json").write_bytes(config_raw)
    policy = {"modelPolicy": "frozen"}
    policy["modelPolicySha256"] = scorer.sha256(scorer.canonical(policy))
    policy_raw = scorer.canonical(policy)
    (out / "model_policy.json").write_bytes(policy_raw)
    fit_ids = [f"fit-{i:03d}" for i in range(104)]
    inventory = []
    development_fits = []
    for fit_id in fit_ids:
        paths = {}
        for artifact, rel in (("weightPath", f"models/{fit_id}.pt"),
                              ("checkpointPath", f"checkpoints/{fit_id}.pt")):
            path = out / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            content = f"{fit_id}:{artifact}".encode()
            path.write_bytes(content)
            inventory.append({"fitId": fit_id, "artifact": artifact, "path": rel,
                              "sha256": scorer.sha256(content)})
            paths[artifact] = (rel, scorer.sha256(content))
        development_fits.append({"fitId": fit_id,
            "weightPath": paths["weightPath"][0], "weightSha256": paths["weightPath"][1],
            "checkpointPath": paths["checkpointPath"][0], "checkpointSha256": paths["checkpointPath"][1]})
    inventory_sha = scorer.sha256(scorer.canonical(inventory))
    development = {"scores": "preserved", "fitArtifacts": development_fits}
    development["developmentScoresSha256"] = scorer.sha256(scorer.canonical(development))
    development_raw = scorer.canonical(development)
    (out / "development-scores.json").write_bytes(development_raw)
    pins = {"transitionId": "nested-forecast-manifest-with-hashed-csv-companion-v1",
        "configFileSha256": scorer.sha256(config_raw),
        "configInternalSha256": config["resolvedConfigSha256"],
        "originalMainPath": old_path, "originalMainSha256": source_code_files[old_path],
        "policyFileSha256": scorer.sha256(policy_raw), "policyInternalSha256": policy["modelPolicySha256"],
        "developmentFileSha256": scorer.sha256(development_raw),
        "developmentInternalSha256": development["developmentScoresSha256"],
        "trainingLedgerFileSha256": "4" * 64,
        "completedFitCount": 104}
    monkeypatch.setattr(scorer, "TRANSITION_PINS", pins)
    monkeypatch.setattr(scorer.design, "BASELINE_SHA256", {"V": "v" * 64, "T": "t" * 64})
    transition = {"schemaVersion": 1, "experiment": "DIRECT_PRICE_PATH_ATTENTION_V1",
        "transitionId": pins["transitionId"],
        "originalResolvedConfigPath": "outputs/experiments/direct-price-path-attention-v1-20261014/resolved-config.json",
        "originalResolvedConfigFileSha256": pins["configFileSha256"],
        "originalConfigInternalSha256": pins["configInternalSha256"],
        "originalMainPath": old_path, "originalMainSha256": pins["originalMainSha256"],
        "originalTrainingLedgerFileSha256": pins["trainingLedgerFileSha256"],
        "revisedMainPath": new_path, "revisedMainSha256": source_code_files[new_path],
        "dataSourcePath": data_path, "dataSourceSha256": source_code_files[data_path],
        "baselineManifestSha256": {"V": "v" * 64, "T": "t" * 64},
        "modelPolicyFileSha256": pins["policyFileSha256"],
        "developmentScoresFileSha256": pins["developmentFileSha256"],
        "preservedArtifacts": {"resolvedConfigFileSha256": pins["configFileSha256"],
            "modelPolicyFileSha256": pins["policyFileSha256"],
            "modelPolicyInternalSha256": pins["policyInternalSha256"],
            "developmentScoresFileSha256": pins["developmentFileSha256"],
            "developmentScoresInternalSha256": pins["developmentInternalSha256"],
            "trainingLedgerFileSha256": pins["trainingLedgerFileSha256"],
            "completedFitCount": 104, "completedFitIds": fit_ids,
            "completedFitInventorySha256": inventory_sha},
        "completedFitIds": fit_ids, "completedFitInventory": inventory,
        "completedFitInventorySha256": inventory_sha}
    transition["transitionSha256"] = scorer.sha256(scorer.canonical(transition))
    transition_raw = scorer.canonical(transition)
    (out / "implementation-transition.json").write_bytes(transition_raw)
    ref = {"path": "implementation-transition.json", "pathBase": "experimentRoot",
           "fileSha256": scorer.sha256(transition_raw), "transitionSha256": transition["transitionSha256"]}
    assert scorer._validate_transition_refs(ref, ref, source_code_files, out, repo) == transition
    with pytest.raises(ValueError, match="references differ"):
        scorer._validate_transition_refs(ref, {**ref, "fileSha256": "0" * 64}, source_code_files, out, repo)
    (out / "model_policy.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="policy.json"):
        scorer._validate_transition_refs(ref, ref, source_code_files, out, repo)


def test_cell_month_gross_gain_loss_reconciles_to_equal_cell_mean(monkeypatch):
    monkeypatch.setattr(scorer, "PRODUCTS", ("cabbage",))
    monkeypatch.setattr(scorer, "HORIZONS", (1,))
    rows = [
        {"product": "cabbage", "horizon": 1, "target": dt.date(2026, 1, 2),
         "U": {"status": "issued", "prediction": 11.0}, "B": {"status": "issued", "prediction": 8.0}},
        {"product": "cabbage", "horizon": 1, "target": dt.date(2026, 1, 3),
         "U": {"status": "issued", "prediction": 13.0}, "B": {"status": "issued", "prediction": 11.0}},
    ]
    actuals = {("cabbage", dt.date(2026, 1, 2)): 10.0,
               ("cabbage", dt.date(2026, 1, 3)): 10.0}
    report = scorer._comparison(rows, actuals, "U", "B", inference=False)
    contributions = report["cellMonthContributions"]
    assert report["contributionAccounting"]["passed"] is True
    assert sum(x["grossImprovementMae"] for x in contributions) == pytest.approx(0.5)
    assert sum(x["grossWorseningMae"] for x in contributions) == pytest.approx(1.0)
    assert sum(x["netMaeContribution"] for x in contributions) == pytest.approx(0.5)


def test_combined_synthetic_scoring_emits_fixed_inference_families(monkeypatch):
    products = ("cabbage", "radish", "onion", "garlic")
    horizons = tuple(range(1, 11))
    target = dt.date(2026, 1, 31)
    monkeypatch.setattr(scorer, "PRODUCTS", products)
    monkeypatch.setattr(scorer, "HORIZONS", horizons)
    phase = (target, target, target - dt.timedelta(days=10))
    monkeypatch.setattr(scorer, "PHASES", {"V": phase, "T": phase})
    baseline = {}
    rows = []
    for product in products:
        for horizon in horizons:
            origin = target - dt.timedelta(days=horizon)
            baseline[(product, target, horizon)] = {
                "origin": origin, "target": target, "E106_RHS": 10.0, "rhsStatus": "issued"
            }
            rows.append({
                "product": product, "origin": origin, "target": target, "horizon": horizon,
                "target_version": "synthetic-target-v1", "selectedInput": "U", "neuralUsed": True,
                "B": {"status": "issued", "prediction": 10.0, "sourceStatus": "issued"},
                "U": {"status": "issued", "prediction": 10.0, "rawPrediction": 10.0, "rawNumericKind": "issued_positive", "rawStatus": "issued", "rawReason": None},
                "M": {"status": "issued", "prediction": 10.0, "rawPrediction": 10.0, "rawNumericKind": "issued_positive", "rawStatus": "issued", "rawReason": None},
                "N": {"status": "issued", "prediction": 10.0, "rawPrediction": 10.0, "rawNumericKind": "issued_positive", "rawStatus": "issued", "rawReason": None},
                "H": {"status": "issued", "prediction": 10.0, "rawPrediction": 10.0, "rawNumericKind": "issued_positive", "rawStatus": "issued", "rawReason": None},
                "H_U": {"status": "issued", "prediction": 10.0, "rawPrediction": 10.0, "rawNumericKind": "issued_positive", "rawStatus": "issued", "rawReason": None},
                "H_M": {"status": "issued", "prediction": 10.0, "rawPrediction": 10.0, "rawNumericKind": "issued_positive", "rawStatus": "issued", "rawReason": None},
                "activeAuxCount": 0, "fitId": "synthetic-fit", "fitMonth": "2025-12",
            })
    price_doc = {"schemaVersion": 1, "products": [
        {"id": p, "observations": [{"date": target.isoformat(), "actual": 10.0}]}
        for p in (*products, "pepper")
    ]}
    manifest = {"modelPolicy": {"selectedInputByProduct": {p: "U" for p in products}}}
    result = scorer.score_documents(manifest, rows, manifest, rows,
        {"V": baseline, "T": baseline}, price_doc, "synthetic-target-v1")
    assert result["status"] == "completed_goal_not_met"
    assert len(result["V"]["comparisons"]["H_minus_B"]["dm"]) == 160
    assert all(x["familySize"] == 80 for x in result["T"]["comparisons"]["M_minus_U"]["dmSummaryByBandwidth"])
    assert result["V"]["priceChangeRowsVsB"]["H"] == 0
    assert result["V"]["neuralUseRowsByProduct"]["cabbage"] == 10
    assert result["V"]["comparisons"]["H_minus_B"]["contributionAccounting"]["passed"] is True
    for row in rows:
        if row["product"] != "cabbage":
            continue
        failed = {"status": "fallback", "prediction": 10.0, "rawPrediction": None,
                  "rawNumericKind": "no_prediction", "rawStatus": "not_issued",
                  "rawReason": "model_not_issued", "fallbackTo": "B", "reason": "model_not_issued"}
        row["U"] = dict(failed)
        row["N"] = dict(failed)
        row["H"] = dict(failed)
        row["H_U"] = dict(failed)
        row["neuralUsed"] = False
    incomplete = scorer.score_documents(manifest, rows, manifest, rows,
        {"V": baseline, "T": baseline}, price_doc, "synthetic-target-v1")
    assert incomplete["status"] == "blocked_integrity"
    assert incomplete["statusReason"] == ["V:cabbage:no_selected_neural_issues",
                                            "T:cabbage:no_selected_neural_issues"]
