import json
import math
import importlib.metadata
import platform
from pathlib import Path

import pytest
import numpy as np

from agri_data import intra_crop_panel as panel


class _FakeBooster:
    def __init__(self, params):
        self.params = params

    def current_iteration(self):
        return 5

    def feature_name(self):
        return list(panel.FEATURE_NAMES)

    def model_to_string(self):
        values = {"num_iterations": 5, "learning_rate": 0.05, "num_leaves": 7,
                  "max_depth": -1, "min_data_in_leaf": 10,
                  "min_sum_hessian_in_leaf": 0.001, "lambda_l1": 0.0,
                  "lambda_l2": 0.0, "min_gain_to_split": 0.0,
                  "max_bin": 31, "use_missing": 1, "zero_as_missing": 0,
                  "seed": 17, "cat_l2": 10, "cat_smooth": 10,
                  "max_cat_threshold": 32, "max_cat_to_onehot": 4,
                  "min_data_per_group": 100}
        serialized = "\n".join(f"[{key}: {value}]" for key, value in values.items())
        return f"tree\nend of trees\n\nparameters:\n{serialized}\n"


class _FakeModel:
    def __init__(self, booster, params):
        self.booster_ = booster
        self._params = params

    def get_params(self, deep=False):
        return dict(self._params)


def _serialized_params():
    return {
        "num_iterations": 5,
        "learning_rate": 0.05,
        "num_leaves": 7,
        "max_depth": -1,
        "min_data_in_leaf": 10,
        "min_sum_hessian_in_leaf": 0.001,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "min_gain_to_split": 0.0,
        "max_bin": 31,
        "use_missing": True,
        "zero_as_missing": False,
        "objective": "regression",
        "seed": 17,
    }


def test_required_saved_parameter_rejects_missing_and_conflicting_values():
    assert panel._required_saved_parameter({"num_iterations": 5}, {},
                                          ("num_iterations", "n_estimators"), "rounds") == 5
    with pytest.raises(ValueError, match="does not serialize"):
        panel._required_saved_parameter({}, {}, ("num_iterations",), "rounds")
    with pytest.raises(ValueError, match="conflicting aliases"):
        panel._required_saved_parameter({"max_depth": 3}, {"max_depth": 4},
                                        ("max_depth",), "depth")


def test_price_snapshot_hash_is_read_from_its_real_owner_module():
    assert hasattr(panel.online_source, "PRICES_SHA256")
    assert panel.PRICE_SNAPSHOT_SHA256 == panel.online_source.PRICES_SHA256
    assert not hasattr(panel.source_exp, "PRICES_SHA256")


def test_config_extraction_reads_only_serialized_ctrl_metadata(tmp_path, monkeypatch):
    source_out = tmp_path / "source"
    model_rel = "V/models/ctrl.joblib"
    checkpoint = source_out / model_rel
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    params = {"num_iterations": 5, "learning_rate": 0.05, "reg_alpha": 0.0,
              "reg_lambda": 0.0, "min_split_gain": 0.0, "min_child_samples": 10}
    booster = _FakeBooster(params)
    model = _FakeModel(booster, {"n_estimators": 5})
    manifest_path = source_out / "V" / "forecast-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "phase": "V", "targetValuesAttached": False,
        "sourceCodeSha256": panel.source_exp._file_sha(Path(panel.source_exp.__file__)),
        "fitRecords": [{"modelStatus": "fit", "modelFile": model_rel,
                        "modelFileSha256": panel._file_sha(checkpoint), "product": "cabbage",
                        "horizon": 1, "fitMonth": "2026-07"}],
    }
    manifest["manifestSha256"] = panel._sha(panel._canonical(manifest))
    manifest_raw = json.dumps(manifest).encode()
    manifest_path.write_bytes(manifest_raw)
    monkeypatch.setattr(panel, "SOURCE_OUT", source_out)
    monkeypatch.setattr(panel, "SOURCE_V_MANIFEST_SHA256", panel._sha(manifest_raw))
    monkeypatch.setattr(panel, "SOURCE_V_MANIFEST_CANONICAL_SHA256", manifest["manifestSha256"])
    monkeypatch.setattr("joblib.load", lambda _: {"CTRL_LGB": model})

    output = tmp_path / "resolved.json"
    config = panel.extract_saved_ctrl_config(source_manifest_path=manifest_path,
                                             output_path=output)
    assert config["status"] == "resolved"
    assert config["inherited"]["num_boost_round"] == 5
    assert config["fitParams"]["num_threads"] == 1
    assert config["fitParams"]["feature_fraction"] == 1.0
    assert config["fitParams"]["lambda_l1"] == 0.0
    assert config["fitParams"]["min_data_in_leaf"] == 10
    assert panel._fit_parameters(config)["cat_l2"] == 10
    assert panel._fit_parameters(config)["max_cat_threshold"] == 32
    assert "[max_bin: 31]" in config["serializedBoosterParameterText"]
    assert config["serializedCanonicalParameters"]["use_missing"] is True
    assert config["serializedCanonicalParameters"]["zero_as_missing"] is False
    assert json.loads(output.read_text(encoding="utf-8"))["configSha256"] == config["configSha256"]
    second = panel.extract_saved_ctrl_config(source_manifest_path=manifest_path,
                                            output_path=output)
    assert second["configSha256"] == config["configSha256"]


def test_booster_checkpoint_is_raw_model_text_and_bound_to_identity(tmp_path):
    import lightgbm as lgb
    x = np.asarray([[float(i % 3), float(i % 5)] for i in range(24)])
    y = [float(i % 7) for i in range(24)]
    model = lgb.train({"objective": "regression", "verbosity": -1,
                       "num_threads": 1, "min_data_in_leaf": 2},
                      lgb.Dataset(x, label=y), num_boost_round=2)
    raw_parameter_block, parsed = panel._parse_booster_parameter_block(model.model_to_string())
    assert raw_parameter_block.startswith("parameters:\n[boosting: gbdt]")
    assert parsed["max_bin"] == 255
    assert parsed["use_missing"] is True
    assert parsed["zero_as_missing"] is False
    path = tmp_path / "fold.txt"
    identity = {"product": "cabbage", "variant": "L", "fitOrigin": "2026-07-01"}
    model_hash, identity_hash = panel._save_booster(model, path, identity)
    assert path.read_text(encoding="utf-8") == model.model_to_string()
    assert model_hash == panel._file_sha(path)
    assert identity_hash == panel._sha(panel._canonical(identity))
    loaded, loaded_hash, loaded_identity = panel._load_booster(
        path, path.with_suffix(".identity.json"), identity)
    assert isinstance(loaded, lgb.Booster)
    assert loaded_hash == model_hash
    assert loaded_identity == identity_hash
    with pytest.raises(ValueError, match="identity mismatch"):
        panel._load_booster(path, path.with_suffix(".identity.json"), {**identity, "variant": "P"})


def test_basic40_transform_is_series_relative_and_preserves_masks():
    raw = [0.0] * len(panel.BASE_FEATURE_NAMES)
    index = {name: panel.BASE_FEATURE_NAMES.index(name) for name in panel.BASE_FEATURE_NAMES}
    raw[index["lastPriceLog"]] = 5.0
    raw[index["lag1PriceLog"]] = 4.5
    raw[index["lag1Available"]] = 1.0
    raw[index["lag7PriceLog"]] = 0.0  # unavailable sentinel stays exactly zero
    raw[index["lag7Available"]] = 0.0
    raw[index["window7MeanLogPrice"]] = 4.0
    raw[index["window7MinLogPrice"]] = 3.5
    raw[index["window7MaxLogPrice"]] = 4.5
    raw[index["window7ObservedDays"]] = 3.0
    raw[index["window90ObservedDays"]] = 40.0
    raw[index["window7LogVolumeKg"]] = 2.0
    raw[index["window90LogVolumeKg"]] = 4.0
    result = panel._transform_basic40(raw)
    assert result[index["lag1PriceLog"]] == pytest.approx(-0.5)
    assert result[index["lag7PriceLog"]] == 0.0
    assert result[index["window7MeanLogPrice"]] == pytest.approx(-1.0)
    assert result[index["window7MinLogPrice"]] == pytest.approx(-1.5)
    assert result[index["window7MaxLogPrice"]] == pytest.approx(-0.5)
    assert result[index["window7LogVolumeKg"]] == pytest.approx(-2.0)
    assert result[index["window90LogVolumeKg"]] == 4.0


def test_feature_freshness_is_14_days_from_origin_and_cutoff_is_origin_minus_one():
    origin = panel.dt.date(2026, 4, 20)
    target = origin + panel.dt.timedelta(days=1)
    age14 = {origin - panel.dt.timedelta(days=14): (100.0, 10.0), target: (101.0, 11.0)}
    age15 = {origin - panel.dt.timedelta(days=15): (100.0, 10.0), target: (101.0, 11.0)}
    valid = panel._series_feature(age14, origin, target, 1, {})
    assert valid is not None
    assert valid["sourceDate"] == origin - panel.dt.timedelta(days=14)
    assert valid["cutoff"] == origin - panel.dt.timedelta(days=1)
    assert valid["ageFromOrigin"] == 14
    assert panel._series_feature(age15, origin, target, 1, {}) is None


def test_future_price_mutation_cannot_change_as_of_features():
    origin = panel.dt.date(2026, 4, 20)
    target = origin + panel.dt.timedelta(days=1)
    history = {panel.dt.date(2026, 1, 1) + panel.dt.timedelta(days=i):
               (100.0 + i / 10, 200.0 + i) for i in range((origin - panel.dt.date(2026, 1, 1)).days)}
    history[target] = (500.0, 999.0)
    changed_future = dict(history)
    changed_future[target] = (1.0, 0.0)
    before = panel._series_feature(history, origin, target, 1, {})
    after = panel._series_feature(changed_future, origin, target, 1, {})
    assert before == after


def test_phase_coverage_excludes_april_first_without_a_legal_origin(monkeypatch):
    start = panel.dt.date(2026, 4, 1)
    monkeypatch.setattr(panel, "PRODUCTS", ("cabbage",))
    monkeypatch.setattr(panel, "PHASES", {"V": (start, start + panel.dt.timedelta(days=2), start)})
    audit = panel._verify_phase_actual_coverage(
        "V", [{"product": "cabbage", "target": "2026-04-02"},
              {"product": "cabbage", "target": "2026-04-03"}],
        {("cabbage", start): 100.0,
         ("cabbage", start + panel.dt.timedelta(days=1)): 101.0,
         ("cabbage", start - panel.dt.timedelta(days=1)): 99.0})
    assert audit["expectedLegalTargetPairCount"] == 2
    assert audit["excludedPositiveActualKeyCount"] == 1
    assert audit["excludedPositiveActualKeys"] == [("cabbage", "2026-04-01")]
    with pytest.raises(ValueError, match="forecast target grid"):
        panel._verify_phase_actual_coverage(
            "V", [{"product": "cabbage", "target": "2026-04-02"}],
            {("cabbage", start + panel.dt.timedelta(days=1)): 101.0})


def test_representative_inheritance_matches_saved_checkpoint_target_digest(tmp_path, monkeypatch):
    import joblib

    monkeypatch.setattr(panel, "PRODUCTS", ("cabbage",))
    monkeypatch.setattr(panel, "HORIZONS", (1,))
    monkeypatch.setattr(panel, "SOURCE_CACHE_INTERNAL_SHA256", "synthetic-cache")
    monkeypatch.setattr(panel, "SOURCE_OUT", tmp_path)
    selection = {"cabbage": {"selected": [{"channelId": ["wholesale", "A"],
                                               "grain": "market_subclass"}]}}
    start = panel.dt.date(2026, 1, 1)
    observations = []
    series = {}
    channel_observations = {}
    for offset in range(182):
        day = start + panel.dt.timedelta(days=offset)
        price, volume = 100.0 + offset / 5, 50.0 + offset
        series[day] = (price, volume)
        observations.append({"date": day.isoformat(), "actual": price, "volumeKg": volume})
        channel_observations[day] = {"price": 85.0 + offset / 7, "volumeKgSigned": 1.0}
    channels = {"cabbage": {("wholesale", "A"): channel_observations}}
    prices = {"products": [{"id": "cabbage", "observations": observations}]}
    model_dir = tmp_path / "V/models"
    model_dir.mkdir(parents=True)
    bundles = {}
    records = []
    channel_selection_sha = panel.source_exp._sha(panel.source_exp._canonical(selection))
    for month_text in ("2026-04", "2026-05", "2026-06"):
        fit_origin = panel.dt.date.fromisoformat(month_text + "-01")
        channel_series = panel.source_exp._channel_series(channels, selection, "cabbage")
        cache = {}
        train_rows = []
        for target in sorted(series):
            if target < panel.source_exp.TRAIN_START or target >= fit_origin:
                continue
            feature = panel.source_exp._feature_row(
                series, channel_series, target - panel.dt.timedelta(days=1), target, 1,
                cache, include_path_audit=False)
            if feature is not None:
                train_rows.append({**feature, "target": target,
                                   "label": math.log(series[target][0] / feature["p0"])})
        targets_sha = panel.source_exp._sha(panel.source_exp._canonical([
            {"target": row["target"].isoformat(), "label": row["label"],
             "basic40": row["basic40"], "candidate": row["candidate"]}
            for row in train_rows]))
        identity = {"product": "cabbage", "horizon": 1, "fitOrigin": fit_origin.isoformat(),
                    "sourceCacheSha256": "synthetic-cache",
                    "channelSelectionSha256": channel_selection_sha,
                    "trainingRows": len(train_rows), "trainingTargetsSha256": targets_sha,
                    "modelConfig": panel.source_exp.LIGHTGBM_CONFIG,
                    "executionIdentity": {"fixture": True}}
        relative = f"V/models/cabbage-h01-{month_text}.joblib"
        checkpoint = tmp_path / relative
        checkpoint.write_bytes(f"synthetic checkpoint {month_text}".encode())
        identity_sha = panel.source_exp._sha(panel.source_exp._canonical(identity))
        bundles[str(checkpoint)] = {"checkpointIdentity": identity}
        records.append({"product": "cabbage", "horizon": 1, "fitMonth": fit_origin.isoformat(),
                        "fitOrigin": fit_origin.isoformat(), "modelStatus": "fit",
                        "modelFile": relative,
                        "modelFileSha256": panel._file_sha(checkpoint),
                        "checkpointIdentitySha256": identity_sha,
                        "trainingRows": len(train_rows),
                        "trainingFirstTarget": train_rows[0]["target"].isoformat(),
                        "trainingLastTarget": train_rows[-1]["target"].isoformat(),
                        "trainingTargetsStrictlyBeforeFitOrigin": True})
    monkeypatch.setattr(joblib, "load", lambda path: bundles[str(path)])
    result = panel.verify_representative_key_inheritance(
        {"fitRecords": records}, prices, channels, selection)
    assert result["allMatched"] is True
    assert result["foldCount"] == 3
    assert all(row["exactSavedKeysAndLabelsMatch"] for row in result["checks"])
    assert all(row["exactCANDAToCANDBKeysAndLabelsMatch"] for row in result["checks"])


def test_training_rows_keep_l_weights_separate_from_panel_weights():
    start = panel.dt.date(2026, 1, 1)
    representative, auxiliary = {}, {}
    for offset in range(100):
        day = start + panel.dt.timedelta(days=offset)
        representative[day] = (100.0 + offset, 50.0 + offset)
        if offset % 5 != 0:
            auxiliary[day] = (200.0 + 2 * offset, 20.0 + offset)
    l_rows, panel_rows, origins = panel._training_rows(
        {0: representative, 1: auxiliary}, panel.dt.date(2026, 4, 1), 1)
    assert len(l_rows) == len(origins) > 0
    assert all(row["weight"] == 1.0 for row in l_rows)
    assert l_rows[0] is not panel_rows[0]
    for key, totals in origins.items():
        rows = [r for r in panel_rows if (r["origin"], r["target"]) == key]
        assert totals["K"] in (0, 1)
        assert math.fsum(r["weight"] for r in rows) == pytest.approx(1.0)
        rep = next(row for row in rows if row["seriesId"] == 0)
        assert rep["weight"] == pytest.approx(0.5 if totals["K"] else 1.0)
        assert math.fsum(r["weight"] for r in rows if r["seriesId"] != 0) == pytest.approx(
            0.5 if totals["K"] else 0.0)
    assert all(row["weight"] == 1.0 for row in l_rows)


def test_q_fit_uses_permuted_auxiliary_rows_and_shared_bins(tmp_path, monkeypatch):
    import math

    cache = tmp_path / "source-cache.json"
    cache.write_text("synthetic cache identity", encoding="utf-8")
    monkeypatch.setattr(panel, "SOURCE_CACHE", cache)
    origin0 = panel.dt.date(2026, 1, 1)
    rep_rows, panel_rows, per_origin = [], [], {}
    for offset in range(36):
        target = origin0 + panel.dt.timedelta(days=offset + 1)
        origin = target - panel.dt.timedelta(days=1)
        source_date = origin - panel.dt.timedelta(days=1)
        features = [float((offset * (j + 1)) % 13) / 10 for j in range(len(panel.BASE_FEATURE_NAMES))]
        label = math.sin(offset / 2)
        common = {"target": target, "origin": origin, "features": features,
                  "sourceDate": source_date, "cutoff": origin - panel.dt.timedelta(days=1),
                  "p0": 100.0 + offset, "targetPrice": (100.0 + offset) * math.exp(label)}
        rep = {**common, "seriesId": 0, "label": label, "weight": 1.0}
        aux_label = math.cos(offset / 3)
        rep_rows.append(rep)
        panel_rows.append({**rep, "weight": 0.5})
        panel_rows.append({**common, "seriesId": 1, "label": aux_label, "weight": 0.5})
        per_origin[(origin, target)] = {"K": 1, "representativeWeight": 0.5,
                                        "auxiliaryWeightSum": 0.5, "totalWeight": 1.0,
                                        "auxiliarySeriesIds": [1]}
    params = {"objective": "regression", "num_threads": 1, "verbosity": -1,
              "num_leaves": 7, "learning_rate": 0.1, "min_data_in_leaf": 2,
              "max_bin": 15, "data_random_seed": 38, "use_missing": True,
              "zero_as_missing": False}
    config = {"configSha256": "synthetic-config", "fitParams": params,
              "inherited": {"num_boost_round": 3, "max_bin": 15},
              "serializedCanonicalParameters": {"max_bin": 15, "min_data_in_bin": 1,
                  "bin_construct_sample_cnt": 200, "data_random_seed": 38,
                  "feature_pre_filter": True, "is_enable_sparse": True,
                  "enable_bundle": True, "use_missing": True, "zero_as_missing": False,
                  "cat_l2": 10, "cat_smooth": 10, "max_cat_threshold": 32,
                  "max_cat_to_onehot": 4, "min_data_per_group": 100}}
    record, models = panel._train_fold(
        "cabbage", 1, panel.dt.date(2026, 2, 1), panel.dt.date(2026, 2, 1),
        rep_rows, panel_rows, per_origin,
        [{"seriesId": 0, "channelId": ["representative"]},
         {"seriesId": 1, "channelId": ["detail", "A"]}], config, tmp_path / "V",
        include_q=True, execution_identity_sha="synthetic-execution")
    q_diag = record["permutationDiagnostics"]
    assert q_diag["auxiliaryRows"] == sum(row["seriesId"] != 0 for row in panel_rows)
    assert q_diag["changedRows"] <= q_diag["auxiliaryRows"]
    assert record["pqIdentity"]["sameInputWeightDigest"] is True
    assert record["pqIdentity"]["sameBinningIdentity"] is True
    assert set(models) == {"L", "P", "Q"}

    t_record, t_models = panel._train_fold(
        "cabbage", 1, panel.dt.date(2026, 3, 1), panel.dt.date(2026, 3, 1),
        rep_rows, panel_rows, per_origin,
        [{"seriesId": 0, "channelId": ["representative"]},
         {"seriesId": 1, "channelId": ["detail", "A"]}], config, tmp_path / "T",
        include_q=False, execution_identity_sha="synthetic-execution")
    assert t_record["modelVariants"]["Q"]["status"] == "not_run"
    assert "Q" not in t_models

    no_aux_origins = {key: {"K": 0, "representativeWeight": 1.0,
                            "auxiliaryWeightSum": 0.0, "totalWeight": 1.0,
                            "auxiliarySeriesIds": []} for key in per_origin}
    alias_record, alias_models = panel._train_fold(
        "radish", 1, panel.dt.date(2026, 4, 1), panel.dt.date(2026, 4, 1),
        rep_rows, rep_rows, no_aux_origins,
        [{"seriesId": 0, "channelId": ["representative"]}], config, tmp_path / "alias",
        include_q=True, execution_identity_sha="synthetic-execution")
    assert set(alias_models) == {"L", "P", "Q"}
    assert alias_record["modelVariants"]["Q"]["status"] == "aliased_no_support"
    assert alias_record["permutationDiagnostics"]["changedAuxiliaryWeight"] == 0.0
    assert alias_record["permutationDiagnostics"]["auxiliaryWeight"] == 0.0


def test_synthetic_forecast_manifest_replays_from_saved_checkpoint(tmp_path, monkeypatch):
    import hashlib

    phase = "T"
    target = panel.dt.date(2026, 8, 15)
    origin = target - panel.dt.timedelta(days=1)
    prices_path = tmp_path / "prices.json"
    observations = []
    first = panel.dt.date(2026, 1, 1)
    for offset in range((origin - first).days + 1):
        day = first + panel.dt.timedelta(days=offset)
        observations.append({"date": day.isoformat(), "actual": 80.0 + offset / 2,
                             "volumeKg": 100.0 + offset})
    prices_path.write_text(json.dumps({"products": [{"id": "cabbage", "observations": observations}]}),
                           encoding="utf-8")
    baseline_path = tmp_path / "rhs.json"
    baseline_path.write_text("{}", encoding="utf-8")
    baseline_hash = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    expected_baseline = {("cabbage", target, 1): {
        "product": "cabbage", "origin": origin, "target": target, "horizon": 1,
        "E106_RHS": 100.0, "E106": 99.0, "rhsStatus": "issued", "e106Status": "issued"}}
    cache_path = tmp_path / "source-cache.json"
    cache_path.write_text(json.dumps({"cacheSha256": "synthetic-cache"}), encoding="utf-8")
    config = {"schemaVersion": 1, "experiment": "INTRA_CROP_PANEL_CAND_B_V1", "status": "resolved",
              "sourceManifestSha256": panel.SOURCE_V_MANIFEST_SHA256,
              "environment": {"python": platform.python_version(),
                             **{name: importlib.metadata.version(name)
                                for name in ("lightgbm", "numpy", "joblib")}},
              "configSha256": None, "inherited": {"num_boost_round": 3, "max_bin": 15},
              "fitParams": {"objective": "regression", "verbosity": -1, "num_threads": 1,
                            "num_leaves": 7, "learning_rate": 0.1, "min_data_in_leaf": 2,
                            "max_bin": 15, "data_random_seed": 38,
                            "use_missing": True, "zero_as_missing": False},
              "serializedCanonicalParameters": {"max_bin": 15, "min_data_in_bin": 1,
                  "bin_construct_sample_cnt": 200, "data_random_seed": 38,
                  "feature_pre_filter": True, "is_enable_sparse": True,
                  "enable_bundle": True, "use_missing": True, "zero_as_missing": False,
                  "cat_l2": 10, "cat_smooth": 10, "max_cat_threshold": 32,
                  "max_cat_to_onehot": 4, "min_data_per_group": 100}}
    config["configSha256"] = panel._sha(panel._canonical({k: v for k, v in config.items() if k != "configSha256"}))
    output_dir = tmp_path / "experiment"
    output_dir.mkdir()
    (output_dir / "resolved-config.json").write_bytes(panel._canonical(config))

    monkeypatch.setattr(panel, "SOURCE_CACHE", cache_path)
    monkeypatch.setattr(panel, "SOURCE_CACHE_INTERNAL_SHA256", "synthetic-cache")
    monkeypatch.setattr(panel, "PRICES_PATH", prices_path)
    monkeypatch.setattr(panel, "PRODUCTS", ("cabbage",))
    monkeypatch.setattr(panel, "HORIZONS", (1,))
    monkeypatch.setattr(panel, "PHASES", {phase: (target, target, panel.dt.date(2026, 7, 1))})
    monkeypatch.setattr(panel, "BASELINE_PATHS", {phase: baseline_path})
    monkeypatch.setattr(panel, "BASELINE_SHA256", {phase: baseline_hash})
    monkeypatch.setattr(panel.source_exp, "load_source_cache", lambda _: ({}, {"cabbage": {"selected": []}}))
    monkeypatch.setattr(panel.source_exp, "_baseline_index", lambda _doc, _phase: expected_baseline)

    manifest = panel.build_forecasts(phase, output_dir=output_dir)
    assert manifest["targetValuesAttached"] is False
    assert len(manifest["forecasts"]) == 1
    row = manifest["forecasts"][0]
    assert row["models"]["L"]["status"] == "issued"
    assert row["models"]["P"]["modelUsed"] is True
    assert manifest["technicalCompletion"]["cabbage"]["status"] == "complete"
    assert all("representativeTrainLogDecomposition" not in fit
               for fit in manifest["fits"] if fit["variant"] == "Q")

    replay = panel.verify_saved_model_replay(phase, manifest, output_dir=output_dir)
    assert replay["status"] == "pass"
    assert replay["forecastKeysVerified"] == 1
    assert replay["directPredictionsReplayed"] >= 1
    assert replay["baselineCopiedValuesVerified"] is True
