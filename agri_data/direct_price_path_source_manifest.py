"""Build the protocol-required, hash-pinned source manifest for the direct-price run.

This is an offline inventory operation. It does not open the price database, fit a
model, or run inference. It refuses to emit a manifest until the immutable inputs,
all 152 fit artifacts, and both final forecast phases agree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

EXPERIMENT = "DIRECT_PRICE_PATH_ATTENTION_V1"
EXPERIMENT_DIR = Path("outputs/experiments/direct-price-path-attention-v1-20261014")
RHS_DIR = Path("outputs/experiments/radish-horizon-sharing-v1-20261011")
PROTOCOL_SHA256 = "11bccf3da0abf8c221c6740639fd44efd309a186cf174fa891ec0b06c7389799"
BASELINE_MANIFEST_SHA256 = {
    "V": "632e4968123715612738145cfab35f679cbf02ee9f733c2d470c9b991423beb6",
    "T": "bc4b453df2ded47b8968fc7907fecad024e7da7fa0e2a5e730099a233848cc76",
}
TARGET_SNAPSHOT_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"expected JSON object: {path}")
    return doc


def _verify_self_hash(doc: Mapping[str, Any], field: str, label: str) -> None:
    unsigned = dict(doc)
    actual = unsigned.pop(field, None)
    if actual != sha256_bytes(canonical_json(unsigned)):
        raise ValueError(f"{label} internal {field} mismatch")


def _verify_file(root: Path, relative: str, expected: str, label: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise ValueError(f"missing {label}: {relative}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: {relative}")
    return path


def _model_metadata(root: Path, fit: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    model_rel = fit.get("weightPath", fit.get("modelPath"))
    model_sha = fit.get("weightSha256", fit.get("modelSha256"))
    checkpoint_rel = fit.get("checkpointPath")
    checkpoint_sha = fit.get("checkpointSha256")
    if not all(isinstance(x, str) and x for x in (model_rel, model_sha, checkpoint_rel, checkpoint_sha)):
        raise ValueError(f"fit lacks model/checkpoint paths or hashes: {fit.get('fitId')}")
    model_path = _verify_file(root, model_rel, model_sha, "saved model")
    checkpoint_path = _verify_file(root, checkpoint_rel, checkpoint_sha, "optimizer checkpoint")
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fit_id = fit.get("fitId")
    if not isinstance(model, dict) or not isinstance(checkpoint, dict):
        raise ValueError(f"model/checkpoint payload is not a mapping: {fit_id}")
    for label, metadata in (("model", model), ("checkpoint", checkpoint)):
        if not isinstance(metadata, dict) or metadata.get("fitId") != fit_id:
            raise ValueError(f"{label} metadata fitId mismatch: {fit_id}")
        for key in ("contextDays", "inputType"):
            if key in fit and metadata.get(key) != fit[key]:
                raise ValueError(f"{label} metadata {key} mismatch: {fit_id}")
        expected_repair = fit.get("repair", fit.get("repairSetting"))
        if expected_repair is not None and metadata.get("repair") != expected_repair:
            raise ValueError(f"{label} metadata repair mismatch: {fit_id}")
    if checkpoint.get("product") != fit.get("product"):
        raise ValueError(f"checkpoint metadata product mismatch: {fit_id}")
    if fit.get("bestEpoch") is not None and model.get("bestEpoch") != fit["bestEpoch"]:
        raise ValueError(f"saved model bestEpoch mismatch: {fit_id}")
    fit_limits = fit.get("fitLimits", {})
    fixed_epochs = fit.get("fixedEpoch", fit_limits.get("fixedEpochs"))
    if fixed_epochs is not None:
        checkpoint_limits = checkpoint.get("fitLimits", {})
        if (checkpoint_limits.get("fixedEpochs") != fixed_epochs
                or checkpoint.get("epoch") != fixed_epochs):
            raise ValueError(f"checkpoint fixed-epoch terminal state mismatch: {fit_id}")
    elif fit.get("bestEpoch") is not None and checkpoint.get("bestEpoch") != fit["bestEpoch"]:
        raise ValueError(f"checkpoint bestEpoch mismatch: {fit_id}")
    if model.get("experiment") != EXPERIMENT or not isinstance(model.get("stateDict"), dict):
        raise ValueError(f"saved model schema/experiment mismatch: {fit_id}")
    model_channels = model.get("selectedInputChannels")
    if not isinstance(model_channels, list):
        raise ValueError(f"saved model lacks selectedInputChannels: {fit_id}")
    checkpoint_channels = checkpoint.get("fitIdentity")
    if not isinstance(checkpoint_channels, str) or not checkpoint_channels:
        raise ValueError(f"checkpoint lacks fitIdentity: {fit_id}")
    return {
        "modelPath": model_rel, "modelSha256": model_sha,
        "checkpointPath": checkpoint_rel, "checkpointSha256": checkpoint_sha,
        "checkpointFitIdentity": checkpoint_channels,
        "selectedChannels": model_channels,
    }


def _completed_fit_records(cache: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    records = list(cache.get("completedFits", []))
    if cache.get("completedFit") is not None:
        records.append(cache["completedFit"])
    return records


def _cache_channel_catalog(experiment_root: Path, score_doc: Mapping[str, Any],
                           torch: Any, *, protocol_sha: str, config_file_sha: str,
                           source_cache_sha: str) -> dict[str, dict[str, Any]]:
    """Join 104 score artifacts to their hash-checked cache and saved model."""
    score_fits = score_doc.get("fitArtifacts")
    if not isinstance(score_fits, list) or len(score_fits) != 104:
        raise ValueError("development-scores must contain exactly 104 fitArtifacts")
    dev_cache_dir = experiment_root / "development-cache"
    cache_fits: dict[str, dict[str, Any]] = {}
    for cache_path in sorted(dev_cache_dir.glob("*.json")):
        cache = _read_json(cache_path)
        _verify_self_hash(cache, "cacheSha256", f"development cache {cache_path.name}")
        identity = cache.get("identity")
        if not isinstance(identity, dict):
            raise ValueError(f"development cache lacks identity: {cache_path.name}")
        if (identity.get("protocolSha256") != protocol_sha
                or identity.get("resolvedConfigSha256") != config_file_sha
                or identity.get("sourceCacheSha256") != source_cache_sha):
            raise ValueError(f"development cache references another protocol/config/source: {cache_path.name}")
        selected = identity.get("selectedChannels")
        if not isinstance(selected, list):
            raise ValueError(f"development cache lacks selected channel list: {cache_path.name}")
        for completed in _completed_fit_records(cache):
            fit_id = completed.get("fitId")
            if not isinstance(fit_id, str) or fit_id in cache_fits:
                raise ValueError(f"duplicate/missing fitId in development caches: {fit_id}")
            if any(completed.get(k) != identity.get(k) for k in
                   ("product", "contextDays", "inputType", "repair")):
                raise ValueError(f"cache identity mismatch for {fit_id}")
            fit = next((x for x in score_fits if x.get("fitId") == fit_id), None)
            if fit is None:
                raise ValueError(f"cache fit absent from development-scores: {fit_id}")
            for field in ("weightPath", "weightSha256", "checkpointPath", "checkpointSha256"):
                if completed.get(field) != fit.get(field):
                    raise ValueError(f"development cache/score {field} mismatch for {fit_id}")
            model = _model_metadata(experiment_root, fit, torch)
            if model["selectedChannels"] != selected:
                raise ValueError(f"saved model channels differ from development cache: {fit_id}")
            # Bind the cache identity and its content hash without embedding its rows.
            cache_fits[fit_id] = {
                "cachePath": str(cache_path.relative_to(experiment_root)),
                "cacheSha256": sha256_file(cache_path),
                "cacheInternalSha256": cache["cacheSha256"],
                "fitId": fit_id, "product": identity["product"],
                "contextDays": identity["contextDays"], "inputType": identity["inputType"],
                "repair": identity["repair"], "fold": identity.get("fold", "Y2025"),
                "selectedChannels": selected,
                "model": {k: model[k] for k in ("modelPath", "modelSha256", "checkpointPath",
                                                  "checkpointSha256", "checkpointFitIdentity")},
            }
    if len(cache_fits) != 104 or set(cache_fits) != {x["fitId"] for x in score_fits}:
        raise ValueError("development cache fit coverage does not match 104 scored fits")
    return cache_fits


def _baseline_source(project_root: Path, phase: str, expected_manifest_sha: str,
                     loader: Callable[[str], tuple[dict[str, Any], list[dict[str, Any]]]] | None = None
                     ) -> dict[str, Any]:
    manifest_rel = RHS_DIR / ("validation-manifest.json" if phase == "V" else "test-manifest.json")
    manifest_path = project_root / manifest_rel
    if not manifest_path.is_file() or sha256_file(manifest_path) != expected_manifest_sha:
        raise ValueError(f"pinned RHS {phase} baseline manifest absent or changed")
    if loader is None:
        # Reuse the already reviewed producer loader for its nested/flat row
        # alignment and exact condition semantics; this inventory records only
        # the pinned source references and counts returned by that validator.
        from . import direct_price_path_v2
        manifest, eligible_rows = direct_price_path_v2._baseline_phase(phase)
    else:
        manifest, eligible_rows = loader(phase)
    forecasts = manifest.get("forecasts")
    if not isinstance(forecasts, list) or not forecasts:
        raise ValueError(f"RHS {phase} manifest does not contain nested forecasts")
    loader_manifest_path = manifest.get("path")
    if (not isinstance(loader_manifest_path, Path)
            or loader_manifest_path.resolve() != manifest_path.resolve()
            or sha256_file(loader_manifest_path) != expected_manifest_sha):
        raise ValueError(f"reviewed RHS {phase} loader returned another manifest")
    forecast_path = manifest.get("forecastCsvPath")
    forecast_sha = manifest.get("forecastFileSha256")
    if not isinstance(forecast_path, Path) or not isinstance(forecast_sha, str):
        raise ValueError(f"RHS {phase} baseline missing CSV companion reference/hash")
    _verify_file(project_root, str(forecast_path.relative_to(project_root)), forecast_sha,
                 f"RHS {phase} baseline CSV")
    expected_eligible = {"V": 3020, "T": 3160}[phase]
    if len(eligible_rows) != expected_eligible:
        raise ValueError(f"RHS {phase} baseline eligible coverage changed")
    forecast_file = manifest.get("forecastFile")
    if not isinstance(forecast_file, str):
        raise ValueError(f"RHS {phase} baseline missing CSV companion filename")
    return {"manifestPath": str(manifest_rel), "manifestSha256": expected_manifest_sha,
            "forecastFile": forecast_file, "forecastFileSha256": forecast_sha,
            "nestedForecasts": len(forecasts), "conditionRows": len(forecasts) * 4,
            "eligibleKeys": len(eligible_rows), "conditions": ["B", "C", "D", "P"]}


def build_source_manifest(project_root: Path, torch: Any | None = None) -> dict[str, Any]:
    """Build and validate the single source manifest; no writes are performed here."""
    project_root = project_root.resolve()
    exp = project_root / EXPERIMENT_DIR
    config_path = exp / "resolved-config.json"
    protocol_path = exp / "protocol.md"
    policy_path = exp / "model_policy.json"
    scores_path = exp / "development-scores.json"
    transition_path = exp / "implementation-transition.json"
    cache_path = exp / "source-cache.json"
    config, policy = _read_json(config_path), _read_json(policy_path)
    scores, transition, source_cache = (_read_json(scores_path), _read_json(transition_path),
                                         _read_json(cache_path))
    _verify_self_hash(config, "resolvedConfigSha256", "resolved config")
    _verify_self_hash(policy, "modelPolicySha256", "model policy")
    _verify_self_hash(scores, "developmentScoresSha256", "development scores")
    _verify_self_hash(transition, "transitionSha256", "implementation transition")
    _verify_self_hash(source_cache, "sourceCacheSha256", "source cache")
    _verify_file(project_root, str(protocol_path.relative_to(project_root)), PROTOCOL_SHA256,
                 "frozen protocol")
    if config.get("experiment") != EXPERIMENT or config.get("protocolSha256") != PROTOCOL_SHA256:
        raise ValueError("resolved config experiment/protocol mismatch")
    if policy.get("sourceCacheSha256") != source_cache.get("sourceCacheSha256"):
        raise ValueError("policy does not bind current source cache")
    if transition.get("experiment") != EXPERIMENT:
        raise ValueError("implementation transition experiment mismatch")
    if transition.get("baselineManifestSha256") != BASELINE_MANIFEST_SHA256:
        raise ValueError("transition does not bind the pinned RHS V/T baselines")
    # Revalidate immutable source inputs named by the cache/config.
    for rel, expected, label in (
        ("outputs/experiments/direct-price-path-attention-v1-20261014/source-cache.json",
         sha256_file(cache_path), "source cache"),
        (source_cache["source"]["sourceCodePath"], source_cache["source"]["sourceCodeSha256"], "query source"),
        (source_cache["source"]["dataCodePath"], source_cache["source"]["dataCodeSha256"], "data source"),
        (source_cache["source"]["priceSnapshotPath"], TARGET_SNAPSHOT_SHA256, "pinned target snapshot"),
    ):
        _verify_file(project_root, rel, expected, label)
    if (source_cache["source"].get("priceSnapshotSha256") != TARGET_SNAPSHOT_SHA256
            or config.get("targetSnapshotSha256") != TARGET_SNAPSHOT_SHA256
            or config.get("targetSnapshotPath") != source_cache["source"].get("priceSnapshotPath")):
        raise ValueError("pinned target snapshot reference changed")
    for code_path, digest in config.get("sourceCodeFiles", {}).items():
        _verify_file(project_root, code_path, digest, "frozen original source code")
    _verify_file(project_root, transition["originalMainPath"], transition["originalMainSha256"],
                 "preserved original model producer")
    _verify_file(project_root, transition["revisedMainPath"], transition["revisedMainSha256"],
                 "transition model producer")
    if sha256_file(config_path) != transition.get("originalResolvedConfigFileSha256"):
        raise ValueError("original config file bytes differ from transition")
    if sha256_file(policy_path) != transition.get("modelPolicyFileSha256"):
        raise ValueError("policy bytes differ from implementation transition")
    if sha256_file(scores_path) != transition.get("developmentScoresFileSha256"):
        raise ValueError("development score bytes differ from implementation transition")
    if torch is None:
        import torch as torch_module  # imported only at actual export/runtime
        torch = torch_module

    dev = _cache_channel_catalog(
        exp, scores, torch, protocol_sha=PROTOCOL_SHA256,
        config_file_sha=sha256_file(config_path),
        source_cache_sha=source_cache["sourceCacheSha256"])
    dev_artifacts = {fit["fitId"]: fit for fit in scores["fitArtifacts"]}
    ledger_path = exp / "training-ledger.json"
    ledger = _read_json(ledger_path)
    _verify_self_hash(ledger, "ledgerSha256", "training ledger")
    ledger_fits = ledger.get("fits", {})
    if len(ledger_fits) != 152:
        raise ValueError(f"expected 152 completed training-ledger fits, found {len(ledger_fits)}")
    if set(dev) - set(ledger_fits):
        raise ValueError("training ledger omits development fits")
    # Match every development score record to the transition inventory and live files.
    inventory = transition.get("completedFitInventory", [])
    if (transition.get("completedFitInventorySha256") != sha256_bytes(canonical_json(inventory))
            or len(transition.get("completedFitIds", [])) != 104):
        raise ValueError("transition's preserved fit inventory is incomplete or inconsistent")
    inventory_by_fit: dict[str, dict[str, str]] = {}
    for item in inventory:
        inventory_by_fit.setdefault(item["fitId"], {})[item["artifact"]] = item["sha256"]
    if set(dev) != set(dev_artifacts):
        raise ValueError("development scored fit ids differ from cache inventory")
    for fit_id, details in dev.items():
        score_fit, ledger_fit = dev_artifacts[fit_id], ledger_fits[fit_id]
        for field in ("weightPath", "weightSha256", "checkpointPath", "checkpointSha256"):
            if score_fit.get(field) != ledger_fit.get(field):
                raise ValueError(f"development score/ledger {field} mismatch for {fit_id}")
        inventory = inventory_by_fit.get(fit_id, {})
        for artifact, path_field, hash_field in (
            ("weightPath", "weightPath", "weightSha256"),
            ("checkpointPath", "checkpointPath", "checkpointSha256"),
        ):
            if inventory.get(artifact) != score_fit.get(hash_field):
                raise ValueError(f"transition inventory {artifact} hash mismatch for {fit_id}")
            expected_path = next((x.get("path") for x in transition["completedFitInventory"]
                                  if x.get("fitId") == fit_id and x.get("artifact") == artifact), None)
            if expected_path != score_fit.get(path_field):
                raise ValueError(f"transition inventory {artifact} path mismatch for {fit_id}")
        details["fitBoundary"] = _development_boundary(score_fit, config)
        details["selectedChannels"] = dev_artifacts[fit_id].get("selectedChannels", details["selectedChannels"])
        details["model"]["selectedChannels"] = details["selectedChannels"]
        details["stage"] = "year2025" if fit_id.startswith("year2025-") else "development"

    final_fits: dict[str, dict[str, Any]] = {}
    baseline = {}
    for phase in ("V", "T"):
        baseline[phase] = _baseline_source(project_root, phase, BASELINE_MANIFEST_SHA256[phase])
        manifest_path = exp / phase / "forecast-manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"missing finalized {phase} forecast manifest")
        manifest = _read_json(manifest_path)
        _verify_self_hash(manifest, "manifestSha256", f"{phase} forecast manifest")
        if manifest.get("stage") != phase:
            raise ValueError(f"forecast manifest phase mismatch: {phase}")
        if manifest.get("implementationTransition", {}).get("transitionSha256") != transition["transitionSha256"]:
            raise ValueError(f"{phase} forecast manifest transition reference mismatch")
        if manifest.get("sourceCache", {}).get("sourceCacheSha256") != source_cache["sourceCacheSha256"]:
            raise ValueError(f"{phase} forecast manifest source-cache reference mismatch")
        if manifest.get("sourceCache", {}).get("fileSha256") != sha256_file(cache_path):
            raise ValueError(f"{phase} forecast manifest source-cache file hash mismatch")
        if manifest.get("modelPolicy", {}).get("modelPolicySha256") != policy["modelPolicySha256"]:
            raise ValueError(f"{phase} forecast manifest policy reference mismatch")
        if manifest.get("targetSnapshot", {}).get("sha256") != TARGET_SNAPSHOT_SHA256:
            raise ValueError(f"{phase} forecast manifest target snapshot reference mismatch")
        baseline_ref = manifest.get("baselineSource", {})
        if baseline_ref.get("sha256") != BASELINE_MANIFEST_SHA256[phase]:
            raise ValueError(f"{phase} forecast manifest baseline hash mismatch")
        if baseline_ref.get("csvSha256") != baseline[phase]["forecastFileSha256"]:
            raise ValueError(f"{phase} forecast manifest baseline CSV hash mismatch")
        source_code_files = manifest.get("sourceCodeFiles", {})
        if not isinstance(source_code_files, dict):
            raise ValueError(f"{phase} manifest lacks sourceCodeFiles")
        required_code = {
            transition["revisedMainPath"]: transition["revisedMainSha256"],
            transition["dataSourcePath"]: transition["dataSourceSha256"],
            transition["originalMainPath"]: transition["originalMainSha256"],
        }
        if any(source_code_files.get(path) != digest for path, digest in required_code.items()):
            raise ValueError(f"{phase} manifest source-code hash inventory mismatch")
        for code_path, digest in required_code.items():
            _verify_file(project_root, code_path, digest, "forecast source code")
        phase_source = {"manifestPath": str(manifest_path.relative_to(project_root)),
                        "manifestFileSha256": sha256_file(manifest_path),
                        "manifestInternalSha256": manifest["manifestSha256"],
                        "forecastArtifact": manifest.get("forecastArtifact"),
                        "sourceCodeFiles": source_code_files,
                        "replayVerificationMethod": manifest.get("replayVerification", {}).get("method")}
        fit_grid = manifest.get("fitGrid")
        if not isinstance(fit_grid, list) or len(fit_grid) != 24:
            raise ValueError(f"{phase} manifest must contain exactly 24 fitGrid entries")
        for fit in fit_grid:
            fit_id = fit.get("fitId")
            if not isinstance(fit_id, str) or fit_id in final_fits:
                raise ValueError(f"duplicate/missing final fitId: {fit_id}")
            model = _model_metadata(exp, fit, torch)
            selected = fit.get("selectedChannels")
            if selected != model["selectedChannels"]:
                raise ValueError(f"manifest/model selected channel mismatch: {fit_id}")
            record = dict(fit)
            record["phase"] = phase
            record["phaseArtifact"] = phase_source
            record["modelMetadata"] = {k: model[k] for k in
                                        ("checkpointFitIdentity", "selectedChannels")}
            final_fits[fit_id] = record
            ledger_fit = ledger_fits.get(fit_id)
            if ledger_fit is None:
                raise ValueError(f"final fit absent from training ledger: {fit_id}")
            for key in ("weightPath", "weightSha256", "checkpointPath", "checkpointSha256"):
                manifest_key = {"weightPath": "modelPath", "weightSha256": "modelSha256"}.get(key, key)
                if fit.get(manifest_key, fit.get(key)) != ledger_fit.get(key):
                    raise ValueError(f"final manifest/ledger {key} mismatch: {fit_id}")
    if (len(final_fits) != 48 or len(dev) + len(final_fits) != 152
            or set(dev) & set(final_fits)
            or set(ledger_fits) != (set(dev) | set(final_fits))):
        raise ValueError("fit inventory is not exactly 104 development + 48 final fits")

    source = source_cache["source"]
    channels = source_cache.get("channels", {})
    channel_catalog = {}
    for product, channel_rows in channels.items():
        channel_catalog[product] = [
            {k: row.get(k) for k in ("channelId", "grain", "label", "sourceLabel") if k in row}
            for row in channel_rows
        ]
    doc: dict[str, Any] = {
        "schemaVersion": 1, "experiment": EXPERIMENT, "protocolSha256": PROTOCOL_SHA256,
        "allowedTrainingPeriods": {
            "F23": {"innerFitBoundary": config["folds"]["F23"]["innerFitBoundary"],
                    "innerChannelCutoff": "2022-09-30", "prefixRefitBoundary": config["folds"]["F23"]["refitBoundary"],
                    "prefixRefitChannelRule": "retain inner cutoff-selected list"},
            "F24": {"innerFitBoundary": config["folds"]["F24"]["innerFitBoundary"],
                    "innerChannelCutoff": "2023-09-30", "prefixRefitBoundary": config["folds"]["F24"]["refitBoundary"],
                    "prefixRefitChannelRule": "retain inner cutoff-selected list"},
            "year2025": {"fitBoundary": "2025-01-01", "channelCutoff": "2024-12-31"},
            "V": {"fitBoundary": "per product/month: first eligible origin", "originRange": [config["phases"]["V"]["originStart"], "2026-06-20"]},
            "T": {"fitBoundary": "per product/month: first eligible origin", "originRange": [config["phases"]["T"]["originStart"], config["phases"]["T"]["targetEnd"]]},
        },
        "inputs": {
            "protocol": {"path": str(protocol_path.relative_to(project_root)),
                         "sha256": PROTOCOL_SHA256},
            "sourceCache": {"path": str(EXPERIMENT_DIR / "source-cache.json"),
                            "fileSha256": sha256_file(cache_path),
                            "internalSha256": source_cache["sourceCacheSha256"],
                            "sourceVersion": source_cache.get("schemaVersion"),
                            "source": {k: source.get(k) for k in
                                       ("sourceCodePath", "sourceCodeSha256", "dataCodePath", "dataCodeSha256",
                                        "range", "selectedRowsSha256", "readOnly", "targetReconciliation")}},
            "sourceCodeVersions": {
                "querySource": {"path": source["sourceCodePath"], "sha256": source["sourceCodeSha256"]},
                "dataSource": {"path": source["dataCodePath"], "sha256": source["dataCodeSha256"]},
                "originalProducer": {"path": transition["originalMainPath"], "sha256": transition["originalMainSha256"]},
                "revisedProducer": {"path": transition["revisedMainPath"], "sha256": transition["revisedMainSha256"]},
            },
            "targetSnapshot": {"path": source["priceSnapshotPath"], "sha256": TARGET_SNAPSHOT_SHA256,
                               "vintageLimitation": "Pinned revised snapshot; first publication time is not established."},
            "pinnedRhsBaselines": baseline,
            "runtimeInputRules": {**config["dates"], **config["inputAvailability"],
                                  "marketSlots": config["inputAvailability"]["marketSlots"],
                                  "auxiliaryMasking": "U masks auxiliary values and support; M retains support/momentum masks; masked/padded slots stay explicit."},
            "channelCatalog": channel_catalog,
        },
        "configuration": {
            "resolvedConfig": {"path": str(EXPERIMENT_DIR / "resolved-config.json"), "fileSha256": sha256_file(config_path), "internalSha256": config["resolvedConfigSha256"]},
            "modelPolicy": {"path": str(EXPERIMENT_DIR / "model_policy.json"), "fileSha256": sha256_file(policy_path), "internalSha256": policy["modelPolicySha256"]},
            "developmentScores": {"path": str(EXPERIMENT_DIR / "development-scores.json"), "fileSha256": sha256_file(scores_path), "internalSha256": scores["developmentScoresSha256"]},
            "implementationTransition": {"path": str(EXPERIMENT_DIR / "implementation-transition.json"), "fileSha256": sha256_file(transition_path), "internalSha256": transition["transitionSha256"]},
        },
        "fits": {"development": sorted(dev.values(), key=lambda x: x["fitId"]),
                 "final": sorted(final_fits.values(), key=lambda x: x["fitId"]),
                 "count": 152, "trainingLedgerFileSha256": sha256_file(ledger_path),
                 "trainingLedgerInternalSha256": ledger["ledgerSha256"]},
        "sourceManifestSha256": "",
    }
    doc["sourceManifestSha256"] = sha256_bytes(canonical_json({k: v for k, v in doc.items() if k != "sourceManifestSha256"}))
    return doc


def write_source_manifest(project_root: Path) -> Path:
    doc = build_source_manifest(project_root)
    output = project_root / EXPERIMENT_DIR / "source-manifest.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_bytes(canonical_json(doc))
    os.replace(temporary, output)
    return output


def _development_boundary(fit: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    fit_id = str(fit.get("fitId", ""))
    if fit_id.startswith("year2025-"):
        return "2025-01-01"
    fold = fit_id.rsplit("-", 1)[-1]
    if fold not in {"inner", "refit"}:
        raise ValueError(f"cannot infer development fold from fitId {fit_id}")
    fold_name = "F23" if "-F23-" in fit_id else "F24" if "-F24-" in fit_id else None
    if fold_name is None:
        raise ValueError(f"cannot infer development fold from fitId {fit_id}")
    return config["folds"][fold_name]["innerFitBoundary" if fold == "inner" else "refitBoundary"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    output = write_source_manifest(args.project_root)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
