"""Reviewed V1 producer revision for the pinned nested-manifest/CSV baseline format.

The frozen producer remains available at ``direct_price_path.py``. This module
preserves its protocol, model and completed-fit identities while correcting the
baseline reader and recording the source relocation. Training/inference use the
existing PyTorch runtime; Parquet export uses the existing host runtime.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import chronos2_market_context as baseline_source
from . import direct_price_path_data as data

ROOT = Path(__file__).resolve().parents[1]
OUT = data.OUT
EXPERIMENT_ID = data.EXPERIMENT_ID
PRICE_PATH = data.PRICE_PATH
PRICE_SHA256 = data.PRICE_SHA256
PROTOCOL_PATH = OUT / "protocol.md"
PROTOCOL_SHA256 = "11bccf3da0abf8c221c6740639fd44efd309a186cf174fa891ec0b06c7389799"
CONFIG_PATH = OUT / "resolved-config.json"
TRANSITION_PATH = OUT / "implementation-transition.json"
POLICY_PATH = OUT / "model_policy.json"
DEVELOPMENT_PATH = OUT / "development-scores.json"
LEDGER_PATH = OUT / "training-ledger.json"
PROGRESS_PATH = OUT / "training-progress.json"
HORIZONS = tuple(range(1, 11))
CONTEXTS = (84, 168)
FIT_SEED = 42
MAX_BASE_EPOCHS = 200
MAX_EXTENDED_EPOCHS = 400
PATIENCE = 25
MIN_DELTA = 1e-5
BATCH_SIZE = 64
CLIP_NORM = 1.0
INITIAL_LR = 0.001
INITIAL_WEIGHT_DECAY = 0.0001
INITIAL_DROPOUT = 0.1
INITIAL_AUX_DROPOUT = 0.1
GRADIENT_NORM_ABNORMAL = 50.0
OVERFIT_MIN_VAL_RISE = 0.02
OVERFIT_MIN_TRAIN_FALL = 0.02
REPAIR_M_CONTEXT_AUX_REQUIRED = True
RELOAD_REPLAY_ATOL = 1e-6
RELOAD_REPLAY_RTOL = 1e-6
V_PHASE = (dt.date(2026, 4, 1), dt.date(2026, 6, 20), dt.date(2026, 4, 1))
T_PHASE = (dt.date(2026, 7, 11), dt.date(2026, 9, 27), dt.date(2026, 7, 1))
_SOURCE_CACHE_INTERNAL_SHA: str | None = None
_FROZEN_MAIN_PATH = ROOT / "agri_data/direct_price_path.py"
_ORIGINAL_CONFIG_FILE_SHA256 = "236003f2e844e30eaece7a30123e56151c84fcb695e406164d15b222c7d7af93"
_ORIGINAL_CONFIG_INTERNAL_SHA256 = "ed6f49c7ecd0331e1cfef137c9f3c453d1f1d5a2361fd4dfdc43f3cba383aa12"
_ORIGINAL_MAIN_SHA256 = "0a69eea3f55b72574d869e6d17b563b332d18a3cf12d5dd2f4bb41da1a932401"
_ORIGINAL_POLICY_FILE_SHA256 = "dc42a386878360cde117795393478b0e836b9f538749acafabc08967db1c26f2"
_ORIGINAL_POLICY_INTERNAL_SHA256 = "e0cc3d73d2592bf05cce65b4b2e63bf12abb5c182bacbc45435a667d3ef4ca64"
_ORIGINAL_DEVELOPMENT_FILE_SHA256 = "89e9839d11f6e23636e7b92b6d188d8ae8ec166b8ec48e221bf5fd3a378d350b"
_ORIGINAL_DEVELOPMENT_INTERNAL_SHA256 = "8298477ac10b43097f635acecb3a1e35482fff065ff5fd3caad0b5e95c539d64"
_ORIGINAL_TRAINING_LEDGER_FILE_SHA256 = "4b1272de58c3a27af3c019e2f789bcb7c17d9762cfe1af5ca18b365574e01616"
_ORIGINAL_COMPLETED_FITS = 104
_BASELINE_CSV_FIELDS = ["product", "origin", "target", "originDate", "targetDate", "horizon",
                        "eligible", "p0", "p0Date", "fitId", "condition", "status",
                        "directLogPrediction", "directPrediction", "sarimaPrediction",
                        "blendPrediction", "fallbackReason", "cutoff"]
BASELINE_EXPECTED_ELIGIBLE = {"V": 3020, "T": 3160}
BASELINE_PROTOCOL_SHA256 = "953c18f63f79fc427d54c07a138cd6ad7c24e5a50dc0dcafd2c0fd01ebe91430"


class OptimizerInstabilityError(FloatingPointError):
    """A nonfinite update from finite, validated model inputs/targets."""


class FitDataIntegrityError(ValueError):
    """Malformed or nonfinite fit data; never eligible for model repair budget."""
DEV_FOLDS = {
    "F23": {"innerFitBoundary": dt.date(2022, 10, 1),
            "validationEnd": dt.date(2022, 12, 31),
            "refitBoundary": dt.date(2023, 1, 1),
            "outerStart": dt.date(2023, 1, 1),
            "outerEnd": dt.date(2023, 12, 31)},
    "F24": {"innerFitBoundary": dt.date(2023, 10, 1),
            "validationEnd": dt.date(2023, 12, 31),
            "refitBoundary": dt.date(2024, 1, 1),
            "outerStart": dt.date(2024, 1, 1),
            "outerEnd": dt.date(2024, 12, 31)},
}


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _file_sha(path: Path) -> str:
    return _sha(path.read_bytes())


def _assert_price_snapshot() -> None:
    if not PRICE_PATH.is_file() or _file_sha(PRICE_PATH) != PRICE_SHA256:
        raise ValueError("pinned representative-price snapshot is missing or changed")


def _write_once(path: Path, raw: bytes) -> None:
    if path.exists():
        if path.read_bytes() != raw:
            raise FileExistsError(f"refusing to overwrite frozen artifact {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(raw)
    temp.replace(path)


def _write_replace(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(raw)
    temp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _self_hash(doc: Mapping[str, Any], field: str, label: str) -> None:
    unsigned = dict(doc)
    value = unsigned.pop(field, None)
    if value != _sha(_canonical(unsigned)):
        raise ValueError(f"{label} self-hash mismatch")


def _source_cache_identity() -> str | None:
    global _SOURCE_CACHE_INTERNAL_SHA
    if _SOURCE_CACHE_INTERNAL_SHA is not None:
        return _SOURCE_CACHE_INTERNAL_SHA
    if not data.SOURCE_CACHE_PATH.exists():
        return None
    doc = _read_json(data.SOURCE_CACHE_PATH)
    _self_hash(doc, "sourceCacheSha256", "direct-price source cache")
    _SOURCE_CACHE_INTERNAL_SHA = str(doc["sourceCacheSha256"])
    return _SOURCE_CACHE_INTERNAL_SHA


def _resolved_config() -> dict[str, Any]:
    """All model and protocol choices, including prefit repair thresholds."""
    source_files = [Path(__file__), Path(data.__file__)]
    return {
        "schemaVersion": 1, "experiment": EXPERIMENT_ID,
        "protocolPath": str(PROTOCOL_PATH.relative_to(ROOT)),
        "protocolSha256": PROTOCOL_SHA256,
        "targetSnapshotPath": str(PRICE_PATH.relative_to(ROOT)),
        "targetSnapshotSha256": PRICE_SHA256,
        "sourceCachePath": str(data.SOURCE_CACHE_PATH.relative_to(ROOT)),
        "sourceCacheInternalShaField": "sourceCacheSha256",
        "sourceCodeFiles": {str(p.relative_to(ROOT)): _file_sha(p) for p in source_files},
        "fitSeed": FIT_SEED, "contextsCalendarDays": list(CONTEXTS),
        "outputHorizonsCalendarDays": list(HORIZONS),
        "inputChannels": {"representative": 1, "markets": list(data.MARKETS),
                          "maximumTrainingSelectedDetails": data.MAX_DETAIL_CHANNELS,
                          "paddedChannels": 11},
        "dates": {"inputCutoff": "origin_minus_1_calendar_day",
                  "targetForOutputJ0ToJ9": "origin_plus_J_plus_1_calendar_day",
                  "trainingLabelRule": "target_strictly_before_fit_boundary",
                  "firstTargetDayAndHorizon1BothEqualOriginPlus1": True},
        "inputAvailability": {"representativeMaxAgeAtOriginDays": data.MAX_REP_AGE_DAYS,
                               "auxMinimumInputWindowObservations": data.MIN_AUX_WINDOW_DAYS,
                               "auxMaximumLastObservationAgeAtOriginDays": data.MAX_AUX_AGE_DAYS,
                               "windowNormalization": "current-window-observed-log-price-population-mean-std",
                               "sigmaFloor": 0.01,
                               "candidateSelectionLookbackCalendarDays": data.DETAIL_LOOKBACK_DAYS,
                               "candidateMinPositiveDays": data.MIN_DETAIL_DAYS,
                               "candidateMaxLastObservationAgeDays": data.DETAIL_MAX_AGE_DAYS,
                               "candidateOrder": "support_desc_then_channel_id_lexicographic",
                               "channelListOnPrefixRefit": "retain_inner_training_cutoff_list",
                               "marketSlots": "fixed_code_slots; absent rows are padded/masked"},
        "architecture": {"trendKernel": 7, "trendPadding": "replicate_values_and_masks",
                          "momentumLag": 7, "attentionAxis": "channel",
                          "heads": 4, "attentionDropout": INITIAL_DROPOUT,
                          "readoutDropout": INITIAL_DROPOUT,
                          "metadataEncoder": "Linear(4,L,bias=True)",
                          "maskEncoderBottleneck": 8,
                          "channelRoleOrder": ["representative", "gareak", "gangseo", "detail"],
                          "directReadoutPaths": ["trend", "seasonal", "attention", "representative_observed_mask"],
                          "calendarFeatures": ["weekday_sin", "weekday_cos", "year_position_sin", "year_position_cos"],
                          "calendarYearPositionUsesActual365Or366": True,
                          "zeroInitializeAllFinalOutputReadouts": True},
        "training": {"loss": "equal_horizon_mean(abs_error/actual + mae_weight*abs_error/q_train)",
                     "qStatistic": "median_unique_representative_observation_dates_before_fit_boundary",
                     "evenQMedian": "arithmetic_mean_middle_pair_no_rounding",
                     "globalValidLabelCountsPerHorizon": True,
                     "minibatchGlobalWeight": "num_training_examples/minibatch_size * sum_valid_horizon_loss/n_h; last partial batch uses actual partial size",
                     "validLabelGatherBeforeLoss": True,
                     "optimizer": "AdamW", "initialLearningRate": INITIAL_LR,
                     "initialWeightDecay": INITIAL_WEIGHT_DECAY,
                     "batchSize": BATCH_SIZE, "dropLast": False,
                     "patience": PATIENCE, "minDelta": MIN_DELTA,
                     "readoutDropoutPlacement": "independent dropout on each direct feature path immediately before its zero-initialized Linear(L,10); shared CalendarHead uses raw target-calendar features",
                     "attentionOutputDropoutPlacement": "dropout on the attended representative vector immediately before its zero-initialized Linear(L,10)",
                     "auxTokenDropoutPlacement": "independently mask active auxiliary tokens per training forward; preserve representative token; apply mask to token values and key-padding mask",
                     "oomOnceOnlyAction": "one retry with batch32 gradient accumulation2, recompute each microbatch contribution using full effective-batch denominator and same effective64 update; persist batch32 mode for later training/evaluation/inference and never silently skip an update",
                     "clipGlobalNorm": CLIP_NORM,
                     "abnormalGradientNormBeforeClip": GRADIENT_NORM_ABNORMAL,
                     "auxTokenDropoutBySetting": {"initial": INITIAL_AUX_DROPOUT, "repair": 0.3},
                     "seed": FIT_SEED, "dtype": "float32",
                     "predictionExpAndPriceBlend": "float64_outside_device",
                     "extension": {"atEpoch": 200, "recentEpochWindow": 25,
                                   "recentBestMustUpdate": True,
                                   "relativeImprovementAgainstPrecedingBest": 0.005,
                                   "maxEpoch": 400, "sameOptimizerState": True},
                     "checkpoint": "atomic_state_model_optimizer_rng_diagnostics_and_exact_terminal_extension_decision_each_epoch",
                     "checkpointIdentity": "fit_id_plus_data_prefix_plus_source_config_plus_fixed_epochs_max_epochs_and_extension_allowed",
                     "fitLedger": "immutable_completed_fit_id_plus_started_checkpoint_identity_and_epoch_wall_clock_update_counts",
                     "trainingHistoryArtifact": "host_pandas_pyarrow_export_of_complete_fit_ledger_with_roundtrip_hash_check"},
        "folds": {k: {name: value.isoformat() for name, value in fold.items()}
                  for k, fold in DEV_FOLDS.items()},
            "selection": {"J": "equal_mean_over_2023_2024_of_half(MAPE/model_p0_MAPE+MAE/model_p0_MAE)",
                      "contextScore": "0.5*(J_U+J_M)", "contextTie": "shorter_context",
                      "allInfiniteFallback": "mean_unscaled_MAPE_then_mean_unscaled_MAE_then_short_context",
                      "2025InputTie": "U", "2025EpochMedianRounding": "half_up",
                      "NSelectionUnit": "one_input_type_per_product_for_all_10_horizons"},
        "repairPolicy": {
            "order": ["lower_learning_rate", "regularization_package", "increase_mae_loss_weight", "auxiliary_dropout"],
            "perProductMaximumSettings": 1,
            "settingsRunPairedAcrossU_MAndF23_F24": True,
            "lrTrigger": {"id": "repeated_finite_validation_oscillation", "numericalEvents": "nonfinite loss/gradient aborts the fit as an implementation/runtime failure and never consumes repair budget", "finiteOscillation": "at_least_3 local validation rebound/recovery cycles; each rebound >5% above preceding best; recovery >=2% within 5 epochs; gradient norms finite and <=50 before clipping", "applyWhen": "any base U/M selected-context fold meets the finite-oscillation criterion"},
            "regularizationTrigger": {"id": "repeatable_overfit", "perCurve": "post-best final 10-epoch mean validation loss >= best*1.02 AND final 10-epoch mean train loss <= train loss at best*0.98", "aggregate": "same input U or M shows perCurve in both F23 and F24 at selected context"},
            "maeWeightTrigger": {"id": "repeatable_mape_mae_conflict", "criterion": "same U or M has MAPE below p0 and MAE above p0 in both outer years, using exact paired issued/positive-label keys"},
            "auxDropoutTrigger": {"id": "M_worse_than_U_with_active_aux", "criterion": "M J exceeds U J in each outer year and M has >=1 active auxiliary channel in an evaluated key in each outer year"},
            "conflictMultipleTriggerResolution": "first applicable item in order; only one per crop",
            "noTrigger": "do not fit a repair setting; preserve diagnosis and continue 2025/final required fits"},
        "diagnosticDefinitions": {"activeAuxCoverage": "count/fraction of rows with >=1 active selected auxiliary input; also per-role means",
                                  "devP0Baseline": "last positive representative price at/before cutoff with <=14 day age, same rows as model",
                                  "directionAndContributions": "scorer-defined; all forecast keys preserved",
                                  "nonfinitePolicy": "abort fit as implementation/runtime integrity error with checkpoint retained; do not skip minibatch or consume repair budget",
                                  "gradientInstability": "record finite preclip gradient norm; classify a norm >50 as a declared optimizer instability event; record clipping at global norm 1"},
        "phases": {"V": {"targetStart": V_PHASE[0].isoformat(), "targetEnd": V_PHASE[1].isoformat(), "originStart": V_PHASE[2].isoformat()},
                   "T": {"targetStart": T_PHASE[0].isoformat(), "targetEnd": T_PHASE[1].isoformat(), "originStart": T_PHASE[2].isoformat()}},
        "budget": {"baseFits": 64, "repairMaxFits": 32, "year2025Fits": 8,
                   "final2026Fits": 48, "maximumFits": 152},
        "runtime": _runtime_identity(include_device=True),
        "savedModelReplay": {"absoluteTolerance": RELOAD_REPLAY_ATOL,
                             "relativeTolerance": RELOAD_REPLAY_RTOL,
                             "stateDictAgreement": "bitwise_equal_cpu_tensors",
                             "predictionAgreement": "all_raw_values_with_numpy_allclose_equal_nan"},
    }


def freeze_resolved_config() -> dict[str, Any]:
    if _file_sha(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValueError("the frozen protocol file hash differs from this implementation")
    doc = _resolved_config()
    doc["resolvedConfigSha256"] = _sha(_canonical(doc))
    _write_once(CONFIG_PATH, _canonical(doc))
    return doc


def validate_resolved_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    doc = _read_json(path)
    _self_hash(doc, "resolvedConfigSha256", "resolved config")
    current = _resolved_config()
    if doc.get("protocolSha256") != PROTOCOL_SHA256:
        raise ValueError("resolved configuration is bound to a different protocol")
    current.pop("resolvedConfigSha256", None)
    prior = dict(doc)
    prior.pop("resolvedConfigSha256", None)
    if prior != current:
        _validate_implementation_transition(doc, current)
    return doc


def _completed_fit_inventory(ledger: Mapping[str, Any]) -> list[dict[str, str]]:
    fits = ledger.get("fits")
    if not isinstance(fits, Mapping):
        raise ValueError("training ledger has no completed-fit map")
    inventory = []
    for fit_id, fit in sorted(fits.items()):
        for path_key, sha_key in (("weightPath", "weightSha256"),
                                  ("checkpointPath", "checkpointSha256")):
            path, digest = fit.get(path_key), fit.get(sha_key)
            if not isinstance(path, str) or not isinstance(digest, str) or len(digest) != 64:
                raise ValueError(f"completed fit lacks immutable {path_key}/{sha_key}: {fit_id}")
            inventory.append({"fitId": str(fit_id), "artifact": path_key,
                              "path": path, "sha256": digest})
    return inventory


def _load_implementation_transition(*, verify_native_runtime: bool = True) -> dict[str, Any]:
    if not TRANSITION_PATH.is_file():
        raise ValueError("approved baseline-format transition is missing; run the transition-record command after review")
    transition = _read_json(TRANSITION_PATH)
    _self_hash(transition, "transitionSha256", "implementation transition")
    if (transition.get("schemaVersion") != 1
            or transition.get("experiment") != EXPERIMENT_ID
            or transition.get("transitionId") != "nested-forecast-manifest-with-hashed-csv-companion-v1"
            or transition.get("originalResolvedConfigFileSha256") != _file_sha(CONFIG_PATH)
            or transition.get("originalResolvedConfigFileSha256") != _ORIGINAL_CONFIG_FILE_SHA256
            or transition.get("originalMainSha256") != _ORIGINAL_MAIN_SHA256
            or _file_sha(_FROZEN_MAIN_PATH) != _ORIGINAL_MAIN_SHA256
            or transition.get("revisedMainPath") != str(Path(__file__).relative_to(ROOT))
            or transition.get("revisedMainSha256") != _file_sha(Path(__file__))
            or transition.get("dataSourceSha256") != _file_sha(Path(data.__file__))):
        raise ValueError("implementation transition does not match the frozen/current producer sources")
    preserved = transition.get("preservedArtifacts", {})
    if not isinstance(preserved, Mapping):
        raise ValueError("implementation transition lacks its preserved-artifact inventory")
    for label, path, expected in (
            ("resolvedConfig", CONFIG_PATH, transition.get("originalResolvedConfigFileSha256")),
            ("modelPolicy", POLICY_PATH, preserved.get("modelPolicyFileSha256")),
            ("developmentScores", DEVELOPMENT_PATH, preserved.get("developmentScoresFileSha256"))):
        if not path.is_file() or _file_sha(path) != expected:
            raise ValueError(f"preserved {label} artifact changed after compatibility transition")
    if preserved.get("modelPolicyFileSha256") != _ORIGINAL_POLICY_FILE_SHA256:
        raise ValueError("transition does not preserve the reviewed model policy bytes")
    if preserved.get("developmentScoresFileSha256") != _ORIGINAL_DEVELOPMENT_FILE_SHA256:
        raise ValueError("transition does not preserve the completed development result bytes")
    if (transition.get("originalConfigInternalSha256") != _ORIGINAL_CONFIG_INTERNAL_SHA256
            or transition.get("modelPolicyFileSha256") != preserved.get("modelPolicyFileSha256")
            or transition.get("developmentScoresFileSha256") != preserved.get("developmentScoresFileSha256")
            or preserved.get("modelPolicyInternalSha256") != _ORIGINAL_POLICY_INTERNAL_SHA256
            or preserved.get("developmentScoresInternalSha256") != _ORIGINAL_DEVELOPMENT_INTERNAL_SHA256):
        raise ValueError("transition's preserved artifact hashes are inconsistent")
    if (transition.get("originalTrainingLedgerFileSha256") != _ORIGINAL_TRAINING_LEDGER_FILE_SHA256
            or preserved.get("trainingLedgerFileSha256") != _ORIGINAL_TRAINING_LEDGER_FILE_SHA256):
        raise ValueError("transition does not pin the completed development training ledger")
    baseline_hashes = transition.get("baselineManifestSha256")
    if not isinstance(baseline_hashes, Mapping):
        raise ValueError("implementation transition lacks pinned baseline hashes")
    for phase, expected in baseline_source.BASELINE_SHA256.items():
        path = baseline_source.BASELINE_PATHS[phase]
        if baseline_hashes.get(phase) != expected or _file_sha(path) != expected:
            raise ValueError(f"{phase} pinned baseline changed after compatibility transition")
    ledger = _read_json(LEDGER_PATH)
    _self_hash(ledger, "ledgerSha256", "training ledger")
    fits = ledger.get("fits", {})
    preserved_ids = transition.get("completedFitIds", [])
    if (not isinstance(preserved_ids, list)
            or any(not isinstance(fit_id, str) for fit_id in preserved_ids)
            or len(preserved_ids) != _ORIGINAL_COMPLETED_FITS
            or len(set(preserved_ids)) != _ORIGINAL_COMPLETED_FITS
            or preserved.get("completedFitCount") != _ORIGINAL_COMPLETED_FITS
            or preserved.get("completedFitIds") != preserved_ids
            or not set(preserved_ids).issubset(fits)):
        raise ValueError("completed development fit inventory is incomplete after transition")
    current_inventory = [item for item in _completed_fit_inventory({
        "fits": {fit_id: fits[fit_id] for fit_id in preserved_ids}})]
    if (_sha(_canonical(current_inventory)) != transition.get("completedFitInventorySha256")
            or current_inventory != transition.get("completedFitInventory")):
        raise ValueError("preserved completed development fit inventory changed")
    doc = _read_json(CONFIG_PATH)
    current = _resolved_config()
    prior_sources = dict(doc.get("sourceCodeFiles", {}))
    current_sources = dict(current.get("sourceCodeFiles", {}))
    old_path = "agri_data/direct_price_path.py"
    new_path = str(Path(__file__).relative_to(ROOT))
    if (prior_sources.pop(old_path, None) != _ORIGINAL_MAIN_SHA256
            or new_path in prior_sources
            or current_sources.get(new_path) != transition.get("revisedMainSha256")
            or prior_sources != {k: v for k, v in current_sources.items() if k != new_path}
            or _file_sha(Path(data.__file__)) != prior_sources.get(str(Path(data.__file__).relative_to(ROOT)))):
        raise ValueError("compatibility transition allows only the frozen main-to-revised-module source relocation")
    current.pop("sourceCodeFiles", None)
    prior_doc = dict(doc)
    prior_doc.pop("resolvedConfigSha256", None)
    prior_doc.pop("sourceCodeFiles", None)
    if not verify_native_runtime:
        current.pop("runtime", None)
        prior_doc.pop("runtime", None)
    if prior_doc != current:
        raise ValueError("compatibility transition changed non-source frozen configuration")
    return transition


def _validate_implementation_transition(original_config: Mapping[str, Any],
                                       current_config: Mapping[str, Any], *,
                                       verify_native_runtime: bool = True) -> None:
    # Compare all frozen fields; the only permitted delta is the source path/hash
    # relocation from the immutable approved module to this reviewed revision.
    transition = _load_implementation_transition(verify_native_runtime=verify_native_runtime)
    if transition.get("originalConfigInternalSha256") != original_config.get("resolvedConfigSha256"):
        raise ValueError("compatibility transition does not reference this resolved configuration")
    prior_sources = dict(original_config.get("sourceCodeFiles", {}))
    current_sources = dict(current_config.get("sourceCodeFiles", {}))
    old_path = "agri_data/direct_price_path.py"
    new_path = str(Path(__file__).relative_to(ROOT))
    if (prior_sources.pop(old_path, None) != _ORIGINAL_MAIN_SHA256
            or new_path in prior_sources
            or current_sources.get(new_path) != transition.get("revisedMainSha256")
            or prior_sources != {k: v for k, v in current_sources.items() if k != new_path}):
        raise ValueError("compatibility transition source path/hash delta is not exactly declared")
    old_config = dict(original_config)
    old_config.pop("resolvedConfigSha256", None)
    old_config.pop("sourceCodeFiles", None)
    new_config = dict(current_config)
    new_config.pop("sourceCodeFiles", None)
    if not verify_native_runtime:
        old_config.pop("runtime", None)
        new_config.pop("runtime", None)
    if old_config != new_config:
        raise ValueError("compatibility transition changes frozen configuration beyond source relocation")


def record_implementation_transition() -> dict[str, Any]:
    """Record the one-file producer relocation without rewriting config or completed work."""
    config = _read_json(CONFIG_PATH)
    _self_hash(config, "resolvedConfigSha256", "resolved config")
    if (_file_sha(CONFIG_PATH) != _ORIGINAL_CONFIG_FILE_SHA256
            or config.get("resolvedConfigSha256") != _ORIGINAL_CONFIG_INTERNAL_SHA256
            or config.get("sourceCodeFiles", {}).get("agri_data/direct_price_path.py") != _ORIGINAL_MAIN_SHA256
            or _file_sha(_FROZEN_MAIN_PATH) != _ORIGINAL_MAIN_SHA256):
        raise ValueError("original config or approved producer differs from the pinned transition source")
    policy = _read_json(POLICY_PATH)
    _self_hash(policy, "modelPolicySha256", "model policy")
    development = _read_json(DEVELOPMENT_PATH)
    _self_hash(development, "developmentScoresSha256", "development scores")
    if (_file_sha(POLICY_PATH) != _ORIGINAL_POLICY_FILE_SHA256
            or policy.get("modelPolicySha256") != _ORIGINAL_POLICY_INTERNAL_SHA256
            or _file_sha(DEVELOPMENT_PATH) != _ORIGINAL_DEVELOPMENT_FILE_SHA256
            or development.get("developmentScoresSha256") != _ORIGINAL_DEVELOPMENT_INTERNAL_SHA256):
        raise ValueError("completed policy or development results differ from the reviewed frozen artifacts")
    ledger = _read_json(LEDGER_PATH)
    if _file_sha(LEDGER_PATH) != _ORIGINAL_TRAINING_LEDGER_FILE_SHA256:
        raise ValueError("completed development training ledger differs from its frozen 104-fit snapshot")
    _self_hash(ledger, "ledgerSha256", "training ledger")
    fits = ledger.get("fits", {})
    if len(fits) != _ORIGINAL_COMPLETED_FITS:
        raise ValueError(f"expected exactly {_ORIGINAL_COMPLETED_FITS} completed fits before V/T; found {len(fits)}")
    inventory = _completed_fit_inventory({"fits": fits})
    for item in inventory:
        artifact = OUT / item["path"]
        if not artifact.is_file() or _file_sha(artifact) != item["sha256"]:
            raise ValueError(f"completed model/checkpoint artifact is missing or changed: {item['fitId']}/{item['artifact']}")
    transition = {
        "schemaVersion": 1, "experiment": EXPERIMENT_ID,
        "transitionId": "nested-forecast-manifest-with-hashed-csv-companion-v1",
        "reason": "The pinned baseline manifest stores nested forecasts; forecastFile names its SHA-checked flattened CSV companion.",
        "originalResolvedConfigPath": str(CONFIG_PATH.relative_to(ROOT)),
        "originalResolvedConfigFileSha256": _file_sha(CONFIG_PATH),
        "originalConfigInternalSha256": config["resolvedConfigSha256"],
        "originalMainPath": str(_FROZEN_MAIN_PATH.relative_to(ROOT)),
        "originalMainSha256": _file_sha(_FROZEN_MAIN_PATH),
        "revisedMainPath": str(Path(__file__).relative_to(ROOT)),
        "revisedMainSha256": _file_sha(Path(__file__)),
        "dataSourcePath": str(Path(data.__file__).relative_to(ROOT)),
        "dataSourceSha256": _file_sha(Path(data.__file__)),
        "baselineManifestSha256": dict(baseline_source.BASELINE_SHA256),
        "modelPolicyFileSha256": _file_sha(POLICY_PATH),
        "developmentScoresFileSha256": _file_sha(DEVELOPMENT_PATH),
        "originalTrainingLedgerFileSha256": _file_sha(LEDGER_PATH),
        "preservedArtifacts": {
            "resolvedConfigFileSha256": _file_sha(CONFIG_PATH),
            "modelPolicyFileSha256": _file_sha(POLICY_PATH),
            "modelPolicyInternalSha256": policy["modelPolicySha256"],
            "developmentScoresFileSha256": _file_sha(DEVELOPMENT_PATH),
            "developmentScoresInternalSha256": development["developmentScoresSha256"],
            "trainingLedgerFileSha256": _file_sha(LEDGER_PATH),
            "completedFitCount": len(fits),
            "completedFitIds": sorted(fits),
            "completedFitInventorySha256": _sha(_canonical(inventory)),
        },
        "completedFitIds": sorted(fits),
        "completedFitInventory": inventory,
        "completedFitInventorySha256": _sha(_canonical(inventory)),
    }
    transition["transitionSha256"] = _sha(_canonical(transition))
    _write_once(TRANSITION_PATH, _canonical(transition))
    _validate_implementation_transition(config, _resolved_config())
    return transition


def _torch_runtime() -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        import torch
        from torch import nn
        from torch.nn import functional as F
    except ImportError as exc:
        raise RuntimeError("this stage requires the existing PyTorch 2.5.1 runtime") from exc
    if torch.__version__ != "2.5.1":
        raise RuntimeError(f"refusing unpinned PyTorch version {torch.__version__}")
    return np, torch, (nn, F)


def _runtime_identity(include_device: bool) -> dict[str, Any]:
    out = {"python": platform.python_version(), "platform": platform.platform(),
           "machine": platform.machine(), "implementation": platform.python_implementation()}
    if importlib.util.find_spec("torch"):
        import torch
        out["torch"] = torch.__version__
        if include_device:
            out["mpsAvailable"] = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
            out["device"] = "mps" if out["mpsAvailable"] else "cpu"
            out["dtype"] = "float32"
            out["threadCount"] = torch.get_num_threads()
    else:
        out["torch"] = None
    return out


def calendar_features(target_days: Sequence[dt.date]) -> list[list[float]]:
    """Return target-day weekday and actual-year-position Fourier features."""
    result = []
    for day in target_days:
        year_days = 366 if _is_leap(day.year) else 365
        position = (day.timetuple().tm_yday - 1) / year_days
        result.append([math.sin(2 * math.pi * day.weekday() / 7),
                       math.cos(2 * math.pi * day.weekday() / 7),
                       math.sin(2 * math.pi * position),
                       math.cos(2 * math.pi * position)])
    return result


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _decompose_numpy(z: Any, observed: Any, lag: int = 7, kernel: int = 7) -> tuple[Any, Any, Any, Any, Any]:
    """Reference decomposition for tests: channels×time, mask-weighted trend."""
    import numpy as np
    z = np.asarray(z, dtype=np.float64)
    observed = np.asarray(observed, dtype=bool)
    if z.ndim != 2 or z.shape != observed.shape or kernel % 2 != 1:
        raise ValueError("decomposition expects matching [channels,time] arrays and odd kernel")
    pad = kernel // 2
    zpad = np.pad(z, ((0, 0), (pad, pad)), mode="edge")
    mpad = np.pad(observed.astype(np.float64), ((0, 0), (pad, pad)), mode="edge")
    sums = np.zeros_like(z)
    counts = np.zeros_like(z)
    for offset in range(kernel):
        sums += zpad[:, offset:offset + z.shape[1]] * mpad[:, offset:offset + z.shape[1]]
        counts += mpad[:, offset:offset + z.shape[1]]
    trend_support = counts > 0
    trend = np.divide(sums, counts, out=np.zeros_like(sums), where=trend_support)
    seasonal = np.where(observed, z - trend, 0.0)
    momentum_mask = np.zeros_like(observed)
    momentum = np.zeros_like(z)
    if z.shape[1] > lag:
        momentum_mask[:, lag:] = observed[:, lag:] & observed[:, :-lag]
        momentum[:, lag:] = np.where(momentum_mask[:, lag:], z[:, lag:] - z[:, :-lag], 0.0)
    return trend, trend_support, seasonal, momentum, momentum_mask


def build_model(context_days: int, repair: str = "base", *, torch: Any | None = None,
                nn: Any | None = None, F: Any | None = None) -> Any:
    """Create the independently specified channel-attention network."""
    if context_days not in CONTEXTS:
        raise ValueError("context must be 84 or 168 calendar days")
    if repair not in ("base", "lr", "regularization", "mae_weight", "aux_dropout"):
        raise ValueError("unknown repair setting")
    if torch is None or nn is None or F is None:
        _, torch, (nn, F) = _torch_runtime()
    class DirectPricePath(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = context_days
            self.mask_encoder = nn.Sequential(nn.Linear(width, 8, bias=False),
                                              nn.Linear(8, width, bias=False))
            self.momentum_mask_encoder = nn.Sequential(nn.Linear(width, 8, bias=False),
                                                       nn.Linear(8, width, bias=False))
            self.role_embedding = nn.Embedding(4, width)
            self.metadata_encoder = nn.Linear(4, width)
            self.attention = nn.MultiheadAttention(width, 4, dropout=INITIAL_DROPOUT,
                                                   batch_first=True)
            self.attention_output_dropout = nn.Dropout(INITIAL_DROPOUT)
            self.readout_dropout = nn.Dropout(0.2 if repair == "regularization" else INITIAL_DROPOUT)
            self.trend_head = nn.Linear(width, 10)
            self.seasonal_head = nn.Linear(width, 10)
            self.attention_head = nn.Linear(width, 10)
            self.observed_mask_head = nn.Linear(width, 10)
            self.calendar_head = nn.Linear(4, 1)
            for head in (self.trend_head, self.seasonal_head, self.attention_head,
                         self.observed_mask_head, self.calendar_head):
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
            if repair == "regularization":
                self.attention.dropout = 0.2
                self.attention_output_dropout.p = 0.2
            self.aux_dropout = 0.3 if repair == "aux_dropout" else INITIAL_AUX_DROPOUT

        def forward(self, z: Any, observed: Any, active: Any, roles: Any,
                    metadata: Any, calendars: Any, *, return_attention: bool = False) -> Any:
            # z/observed are [B,C,L]; roles are integer [B,C].
            mask = observed.to(dtype=z.dtype)
            if self.training and self.aux_dropout > 0:
                keep = torch.rand(active.shape, device=active.device) >= self.aux_dropout
                keep[:, 0] = True
                dropped = active & keep
                active = dropped
                mask = mask * dropped.unsqueeze(-1).to(dtype=mask.dtype)
                z = z * dropped.unsqueeze(-1).to(dtype=z.dtype)
            # Replicate-padding both values and masks gives the edge trend support
            # rule without reading beyond this origin's input window.
            padded_z = F.pad(z * mask, (3, 3), mode="replicate")
            padded_m = F.pad(mask, (3, 3), mode="replicate")
            sums = F.avg_pool1d(padded_z.reshape(-1, 1, context_days + 6), 7, stride=1)
            counts = F.avg_pool1d(padded_m.reshape(-1, 1, context_days + 6), 7, stride=1)
            trend = (sums / counts.clamp_min(1e-8)).reshape_as(z)
            trend = torch.where(counts.reshape_as(z) > 0, trend, torch.zeros_like(trend))
            seasonal = torch.where(mask > 0, z - trend, torch.zeros_like(z))
            momentum_mask = torch.zeros_like(mask)
            momentum = torch.zeros_like(z)
            momentum_mask[:, :, 7:] = mask[:, :, 7:] * mask[:, :, :-7]
            momentum[:, :, 7:] = torch.where(momentum_mask[:, :, 7:] > 0,
                                              z[:, :, 7:] - z[:, :, :-7], 0.0)
            encoded = (self.mask_encoder(mask) + self.momentum_mask_encoder(momentum_mask)
                       + self.role_embedding(roles) + self.metadata_encoder(metadata))
            query = trend[:, 0:1, :] + encoded[:, 0:1, :]
            key = momentum + encoded
            value = seasonal + encoded
            key_padding = ~active.bool()
            # Representative channel is guaranteed active by the data builder.
            attended, weights = self.attention(query, key, value,
                                               key_padding_mask=key_padding,
                                               need_weights=return_attention,
                                               average_attn_weights=False)
            attended = self.attention_output_dropout(attended[:, 0, :])
            tr = self.readout_dropout(trend[:, 0, :])
            se = self.readout_dropout(seasonal[:, 0, :])
            observed_path = self.readout_dropout(mask[:, 0, :])
            delta = (self.trend_head(tr) + self.seasonal_head(se)
                     + self.attention_head(attended)
                     + self.observed_mask_head(observed_path))
            delta = delta + self.calendar_head(calendars).squeeze(-1)
            if return_attention:
                return delta, weights, momentum_mask, counts.reshape_as(z) > 0
            return delta
    return DirectPricePath()


def _as_tensor_batch(records: Sequence[Mapping[str, Any]], labels: Any, label_mask: Any,
                     device: Any, torch: Any) -> tuple[Any, ...]:
    import numpy as np
    z = np.asarray([[c["normalized"] for c in r["window"]["columns"]] for r in records], dtype=np.float32)
    mask = np.asarray([[c["observedMask"] for c in r["window"]["columns"]] for r in records], dtype=np.float32)
    active = np.asarray([[c["active"] for c in r["window"]["columns"]] for r in records], dtype=bool)
    roles = np.asarray([[c["role"] for c in r["window"]["columns"]] for r in records], dtype=np.int64)
    meta = np.asarray([[c["metadata"] for c in r["window"]["columns"]] for r in records], dtype=np.float32)
    return (torch.as_tensor(z, device=device), torch.as_tensor(mask, device=device),
            torch.as_tensor(active, device=device), torch.as_tensor(roles, device=device),
            torch.as_tensor(meta, device=device),
            torch.as_tensor(np.asarray([r["calendar"] for r in records], dtype=np.float32), device=device),
            torch.as_tensor(np.asarray(labels, dtype=np.float64), device="cpu"),
            torch.as_tensor(np.asarray(label_mask, dtype=bool), device="cpu"))


def _date_range(start: dt.date, end_exclusive: dt.date):
    day = start
    while day < end_exclusive:
        yield day
        day += dt.timedelta(days=1)


def _records_for_period(product: str, representative: Mapping[dt.date, float],
                        product_channels: Mapping[tuple[str, ...], Mapping[dt.date, Any]],
                        context_days: int, input_type: str,
                        selected_details: Sequence[Mapping[str, Any]],
                        start: dt.date, end_exclusive: dt.date,
                        label_before: dt.date | None,
                        label_through: dt.date | None = None) -> list[dict[str, Any]]:
    """Build daily calendar-origin rows; label lookups are bounded before access."""
    if not representative:
        return []
    first_origin = min(representative) + dt.timedelta(days=1)
    records = []
    for origin in _date_range(max(first_origin, start), end_exclusive):
        window = data.build_input_window(product, origin, context_days, input_type,
                                         representative, product_channels, selected_details)
        if window is None:
            continue
        target_days = [origin + dt.timedelta(days=h) for h in HORIZONS]
        target_labels = []
        target_mask = []
        for target in target_days:
            # Enforce temporal eligibility before consulting the label mapping.
            allowed = ((label_before is None or target < label_before)
                       and (label_through is None or target <= label_through))
            value = representative.get(target) if allowed else None
            valid = value is not None and math.isfinite(float(value)) and float(value) > 0
            target_labels.append(float(value) if valid else 0.0)
            target_mask.append(bool(valid))
        if label_before is not None and not any(target_mask):
            continue
        records.append({"product": product, "origin": origin, "window": window,
                        "calendar": calendar_features(target_days),
                        "targets": target_labels, "targetMask": target_mask,
                        "targetDays": [d.isoformat() for d in target_days]})
    return records


def _q_scale(representative: Mapping[dt.date, float], fit_boundary: dt.date) -> float:
    values = [float(v) for day, v in representative.items()
              if day < fit_boundary and math.isfinite(float(v)) and float(v) > 0]
    if not values:
        raise ValueError("fit prefix has no positive representative observations")
    q = float(statistics.median(values))
    if not math.isfinite(q) or q <= 0:
        raise ValueError("training median q must be finite and positive")
    return q


def _seed_everything(torch: Any, seed: int = FIT_SEED) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _device(torch: Any) -> Any:
    return torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
                        else "cpu")


def _batch_loss(model: Any, batch_records: Sequence[Mapping[str, Any]], device: Any,
                torch: Any, *, q_scale: float, mae_weight: float,
                horizon_counts: Sequence[int], population: int,
                effective_batch: int | None = None) -> tuple[Any, Any, Any]:
    import numpy as np
    labels = [r["targets"] for r in batch_records]
    masks = [r["targetMask"] for r in batch_records]
    for record in batch_records:
        window = record["window"]
        if (not math.isfinite(float(window["p0"])) or float(window["p0"]) <= 0
                or not math.isfinite(float(window["representativeSigma"]))
                or float(window["representativeSigma"]) <= 0):
            raise FitDataIntegrityError("fit record has invalid representative normalization")
        for column in window["columns"]:
            if (not all(math.isfinite(float(v)) for v in column["normalized"])
                    or not all(v in (0, 1, False, True) for v in column["observedMask"])
                    or not all(math.isfinite(float(v)) for v in column["metadata"])):
                raise FitDataIntegrityError("fit record has malformed or nonfinite input features")
        for value, valid in zip(record["targets"], record["targetMask"]):
            if valid and (not math.isfinite(float(value)) or float(value) <= 0):
                raise FitDataIntegrityError("fit record has invalid observed target label")
    z, observed, active, roles, meta, calendar, y_cpu, mask_cpu = _as_tensor_batch(
        batch_records, labels, masks, device, torch)
    delta = model(z, observed, active, roles, meta, calendar)
    p0 = torch.as_tensor(np.asarray([r["window"]["p0"] for r in batch_records], dtype=np.float64), device="cpu")
    sigma = torch.as_tensor(np.asarray([r["window"]["representativeSigma"] for r in batch_records], dtype=np.float64), device="cpu")
    # The copy remains differentiable; exp and restoration deliberately run on CPU float64.
    delta_cpu = delta.to(device="cpu").to(dtype=torch.float64)
    raw_prediction = torch.exp(torch.log(p0).unsqueeze(1) + sigma.unsqueeze(1) * delta_cpu)
    actuals = y_cpu
    valid = mask_cpu
    batch_size = len(batch_records)
    denominator = effective_batch or batch_size
    scale = population / denominator / 10.0
    pieces = []
    for h in range(10):
        selected = valid[:, h]
        if bool(selected.any()):
            pred_h = raw_prediction[selected, h]
            actual_h = actuals[selected, h]
            err = (pred_h - actual_h).abs()
            term = err / actual_h + mae_weight * err / q_scale
            pieces.append(term.sum() / float(horizon_counts[h]))
    if not pieces:
        raise ValueError("minibatch has no valid target labels")
    loss = torch.stack(pieces).sum() * scale
    if not bool(torch.isfinite(loss)):
        raise OptimizerInstabilityError("nonfinite loss from finite fit inputs; fit aborted without skipping an update")
    return loss, raw_prediction.detach(), (actuals, valid)


def _full_loss(model: Any, records: Sequence[Mapping[str, Any]], device: Any,
               torch: Any, *, q_scale: float, mae_weight: float,
               batch_size: int = BATCH_SIZE) -> float:
    import numpy as np
    if not records:
        return float("inf")
    y = np.asarray([r["targets"] for r in records], dtype=np.float64)
    m = np.asarray([r["targetMask"] for r in records], dtype=bool)
    n_h = m.sum(axis=0)
    denom = np.where(n_h > 0, n_h, 1)
    losses = np.zeros(10, dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            z, observed, active, roles, meta, calendar, _, _ = _as_tensor_batch(
                batch, [r["targets"] for r in batch], [r["targetMask"] for r in batch], device, torch)
            delta = model(z, observed, active, roles, meta, calendar).to(device="cpu").to(dtype=torch.float64)
            p0 = torch.as_tensor([r["window"]["p0"] for r in batch], dtype=torch.float64)
            sigma = torch.as_tensor([r["window"]["representativeSigma"] for r in batch], dtype=torch.float64)
            pred = torch.exp(torch.log(p0).unsqueeze(1) + sigma.unsqueeze(1) * delta)
            actual = torch.as_tensor(np.asarray([r["targets"] for r in batch], dtype=np.float64))
            valid = torch.as_tensor(np.asarray([r["targetMask"] for r in batch], dtype=bool))
            err = (pred - actual).abs()
            term = err / actual.clamp_min(1e-300) + mae_weight * err / q_scale
            term = torch.where(valid, term, torch.zeros_like(term))
            losses += term.sum(dim=0).numpy()
    return float(np.sum(losses / denom) / 10.0)


def _metric_summary(records: Sequence[Mapping[str, Any]], predictions: Sequence[Sequence[float]]) -> dict[str, Any]:
    import numpy as np
    actual = np.asarray([r["targets"] for r in records], dtype=np.float64)
    mask = np.asarray([r["targetMask"] for r in records], dtype=bool)
    pred = np.asarray(predictions, dtype=np.float64)
    persistence = np.asarray([[r["window"]["p0"]] * 10 for r in records], dtype=np.float64)
    horizon_n = mask.sum(axis=0).astype(int)
    model_mape, model_mae, p0_mape, p0_mae = [], [], [], []
    model_coverage, horizon_coverage = 0, []
    for h in range(10):
        key_mask = mask[:, h] & (actual[:, h] > 0) & np.isfinite(actual[:, h])
        candidate_valid = key_mask & np.isfinite(pred[:, h]) & (pred[:, h] > 0)
        baseline_valid = key_mask & np.isfinite(persistence[:, h]) & (persistence[:, h] > 0)
        common = candidate_valid & baseline_valid
        count = int(common.sum())
        target_count = int(key_mask.sum())
        model_coverage += count
        horizon_coverage.append({"targetKeyCount": target_count,
                                 "commonPairCount": count,
                                 "missingModelPredictionCount": target_count - count})
        if target_count == 0:
            model_mape.append(None); model_mae.append(None)
            p0_mape.append(None); p0_mae.append(None)
            continue
        if count == 0:
            model_mape.append(None); model_mae.append(None)
            p0_mape.append(None); p0_mae.append(None)
            continue
        model_error = np.abs(pred[common, h] - actual[common, h])
        baseline_error = np.abs(persistence[common, h] - actual[common, h])
        model_mape.append(float(np.mean(model_error / actual[common, h]) * 100))
        model_mae.append(float(np.mean(model_error)))
        p0_mape.append(float(np.mean(baseline_error / actual[common, h]) * 100))
        p0_mae.append(float(np.mean(baseline_error)))
    supported_h = [i for i, n in enumerate(horizon_n) if n > 0]
    complete_coverage = bool(supported_h and all(
        horizon_coverage[h]["commonPairCount"] == horizon_n[h] for h in supported_h))
    average = lambda values: (float(np.mean([values[h] for h in supported_h]))
                              if supported_h and all(values[h] is not None for h in supported_h) else None)
    result: dict[str, Any] = {"n": int(mask.sum()),
                              "horizonN": horizon_n.tolist(),
                              "commonPairCount": int(model_coverage),
                              "coverageComplete": complete_coverage,
                              "horizonCoverage": horizon_coverage,
                              "metricAveraging": "arithmetic_mean_of_supported_horizon_pair_means",
                              "model": {"MAPE": average(model_mape), "MAE": average(model_mae),
                                        "horizonMAPE": model_mape, "horizonMAE": model_mae,
                                        "commonPairCount": int(model_coverage),
                                        "requiredPairCount": int(mask.sum()),
                                        "coverageComplete": complete_coverage},
                              "p0": {"MAPE": average(p0_mape), "MAE": average(p0_mae),
                                     "horizonMAPE": p0_mape, "horizonMAE": p0_mae,
                                     "commonPairCount": int(model_coverage),
                                     "requiredPairCount": int(mask.sum()),
                                     "coverageComplete": complete_coverage}}
    return result


def _predict_records(model: Any, records: Sequence[Mapping[str, Any]], device: Any,
                     torch: Any, *, batch_size: int = BATCH_SIZE) -> list[list[float]]:
    import numpy as np
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start:start + batch_size]
            z, observed, active, roles, meta, calendar, _, _ = _as_tensor_batch(
                batch, [r["targets"] for r in batch], [r["targetMask"] for r in batch], device, torch)
            delta = model(z, observed, active, roles, meta, calendar).to(device="cpu").to(dtype=torch.float64)
            p0 = torch.as_tensor([r["window"]["p0"] for r in batch], dtype=torch.float64)
            sigma = torch.as_tensor([r["window"]["representativeSigma"] for r in batch], dtype=torch.float64)
            pred = torch.exp(torch.log(p0).unsqueeze(1) + sigma.unsqueeze(1) * delta)
            arr = pred.numpy()
            predictions.extend([[float(v) for v in row] for row in arr])
    return predictions


def _save_checkpoint(path: Path, payload: Mapping[str, Any], torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temp)
    temp.replace(path)


def _save_frozen_checkpoint(path: Path, payload: Mapping[str, Any], torch: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".candidate")
    torch.save(dict(payload), temp)
    if path.exists():
        if _file_sha(path) != _file_sha(temp):
            temp.unlink()
            raise FileExistsError(f"refusing to replace immutable fit artifact {path}")
        temp.unlink()
        return
    temp.replace(path)


def fit_model(product: str, context_days: int, input_type: str, repair: str,
              train_records: Sequence[Mapping[str, Any]], validation_records: Sequence[Mapping[str, Any]],
              q_scale: float, fit_id: str, checkpoint_path: Path,
              *, max_epochs: int = MAX_BASE_EPOCHS, extension_allowed: bool = True,
              fixed_epochs: int | None = None) -> dict[str, Any]:
    """Fit one crop model; nonfinite updates abort rather than being skipped."""
    import copy
    import numpy as np
    np_runtime, torch, (nn, F) = _torch_runtime()
    if not train_records or not validation_records:
        raise ValueError("fit requires nonempty training and validation records")
    ymask = np.asarray([r["targetMask"] for r in train_records], dtype=bool)
    horizon_counts = ymask.sum(axis=0).astype(int).tolist()
    if any(n <= 0 for n in horizon_counts):
        raise ValueError("training prefix lacks a valid label for at least one horizon")
    seed = FIT_SEED
    _seed_everything(torch, seed)
    device = _device(torch)
    model = build_model(context_days, repair, torch=torch, nn=nn, F=F)
    model.to(device=device, dtype=torch.float32)
    learning_rate = 0.0003 if repair == "lr" else INITIAL_LR
    weight_decay = 0.001 if repair == "regularization" else INITIAL_WEIGHT_DECAY
    mae_weight = 3.0 if repair == "mae_weight" else 1.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if fixed_epochs is not None:
        if type(fixed_epochs) is not int or fixed_epochs < 1:
            raise ValueError("fixed_epochs must be a positive integer")
        max_epochs = fixed_epochs
    fit_limits = {"maxEpochs": int(max_epochs), "extensionAllowed": bool(extension_allowed),
                  "fixedEpochs": fixed_epochs}
    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    grad_norm_events = 0
    oom_fallback_used = False
    small_batch_mode = False
    terminal_reason: str | None = None
    started_at = time.monotonic()
    elapsed_prior = 0.0
    base_limit = min(max_epochs, MAX_BASE_EPOCHS)
    final_limit = max_epochs
    train_population = len(train_records)
    epoch = 0
    fit_identity = _sha(_canonical({"fitId": fit_id, "product": product,
                                    "contextDays": context_days, "inputType": input_type,
                                    "repair": repair, "qScale": q_scale,
                                    "trainingRows": train_population,
                                    "validationRows": len(validation_records),
                                    "trainingOriginFirst": str(train_records[0]["origin"]),
                                    "trainingOriginLast": str(train_records[-1]["origin"]),
                                    "fitLimits": fit_limits,
                                    "resolvedConfigSha256": _file_sha(CONFIG_PATH) if CONFIG_PATH.exists() else None,
                                    "sourceCacheSha256": _source_cache_identity(),
                                    "implementationTransitionSha256": _active_transition_sha256()}))
    if checkpoint_path.exists():
        try:
            saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        except TypeError:
            saved = torch.load(checkpoint_path, map_location=device)
        if saved.get("fitIdentity") != fit_identity or saved.get("fitId") != fit_id:
            raise ValueError(f"checkpoint identity mismatch; refusing to reuse {checkpoint_path}")
        model.load_state_dict(saved["modelState"])
        optimizer.load_state_dict(saved["optimizerState"])
        history = list(saved["history"])
        epoch = int(saved["epoch"])
        best_epoch = int(saved["bestEpoch"])
        best_loss = float(saved["bestValidationLoss"])
        best_state = saved.get("bestState")
        no_improve = int(saved["noImprove"])
        grad_norm_events = int(saved.get("gradientNormAbnormalEvents", 0))
        oom_fallback_used = bool(saved.get("oomFallbackUsed", False))
        small_batch_mode = bool(saved.get("smallBatchMode", False))
        terminal_reason = saved.get("terminalReason")
        elapsed_prior = float(saved.get("elapsedSeconds", 0.0))
        if saved.get("fitLimits") != fit_limits:
            raise ValueError(f"checkpoint stopping/extension identity mismatch: {checkpoint_path}")
        final_limit = int(saved.get("finalLimit", final_limit))
        random.setstate(saved["pythonRandomState"])
        np_runtime.random.set_state(saved["numpyRandomState"])
        torch.set_rng_state(saved["torchRandomState"].cpu())
        if device.type == "mps" and saved.get("mpsRandomState") is not None:
            torch.mps.set_rng_state(saved["mpsRandomState"].cpu())
    while epoch < final_limit and terminal_reason is None:
        epoch_started = time.monotonic()
        model.train()
        rng = random.Random(seed + epoch)
        order = list(range(train_population))
        rng.shuffle(order)
        epoch_loss = 0.0
        updates = 0
        observed_norms: list[float] = []
        for start in range(0, train_population, BATCH_SIZE):
            ids = order[start:start + BATCH_SIZE]
            batch = [train_records[i] for i in ids]
            optimizer.zero_grad(set_to_none=True)
            try:
                if small_batch_mode:
                    raise MemoryError("out of memory: persisted small-batch mode")
                loss, _, _ = _batch_loss(model, batch, device, torch, q_scale=q_scale,
                                         mae_weight=mae_weight, horizon_counts=horizon_counts,
                                         population=train_population)
                loss.backward()
            except (RuntimeError, MemoryError) as exc:
                text = str(exc).lower()
                is_oom = "out of memory" in text or "mps backend out of memory" in text
                if not is_oom or (oom_fallback_used and not small_batch_mode):
                    raise RuntimeError(f"fit {fit_id} failed before update {updates + 1}: {exc}") from exc
                # No optimizer step occurred, so the same shuffled effective batch
                # can be recomputed as two microbatches without dropping examples.
                oom_fallback_used = True
                small_batch_mode = True
                optimizer.zero_grad(set_to_none=True)
                if device.type == "mps":
                    torch.mps.empty_cache()
                for micro_start in range(0, len(batch), 32):
                    micro = batch[micro_start:micro_start + 32]
                    micro_loss, _, _ = _batch_loss(model, micro, device, torch,
                                                   q_scale=q_scale, mae_weight=mae_weight,
                                                   horizon_counts=horizon_counts,
                                                   population=train_population,
                                                   effective_batch=len(batch))
                    micro_loss.backward()
                    epoch_loss += float(micro_loss.detach().cpu())
                loss = None
            if loss is not None:
                epoch_loss += float(loss.detach().cpu())
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            norm_value = float(norm.detach().cpu())
            if not math.isfinite(norm_value):
                optimizer.zero_grad(set_to_none=True)
                raise OptimizerInstabilityError(f"fit {fit_id} produced a nonfinite gradient norm before update")
            observed_norms.append(norm_value)
            if norm_value > GRADIENT_NORM_ABNORMAL:
                grad_norm_events += 1
            optimizer.step()
            updates += 1
        train_loss = _full_loss(model, train_records, device, torch,
                                q_scale=q_scale, mae_weight=mae_weight,
                                batch_size=32 if small_batch_mode else BATCH_SIZE)
        val_loss = _full_loss(model, validation_records, device, torch,
                              q_scale=q_scale, mae_weight=mae_weight,
                              batch_size=32 if small_batch_mode else BATCH_SIZE)
        if not math.isfinite(train_loss) or not math.isfinite(val_loss):
            raise OptimizerInstabilityError(f"fit {fit_id} emitted nonfinite epoch loss; checkpoint retained")
        epoch += 1
        improved = val_loss < best_loss - MIN_DELTA
        if improved:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        history.append({"epoch": epoch, "trainLoss": train_loss, "validationLoss": val_loss,
                        "improved": improved, "updates": updates,
                        "optimizerUpdatesCumulative": sum(int(r.get("updates", 0)) for r in history) + updates,
                        "epochWallClockSeconds": time.monotonic() - epoch_started,
                        "gradientNormMaxBeforeClip": max(observed_norms, default=0.0),
                        "gradientNormAbnormalEvents": sum(n > GRADIENT_NORM_ABNORMAL for n in observed_norms)})
        # Persist the exact terminal/extension decision with the epoch so a
        # restart cannot take another optimizer step after reaching a stop.
        if fixed_epochs is not None and epoch >= fixed_epochs:
            terminal_reason = "fixed_epoch_complete"
        elif fixed_epochs is None and no_improve >= PATIENCE:
            terminal_reason = "early_stopping_patience"
        elif fixed_epochs is None and epoch == base_limit and extension_allowed and max_epochs >= MAX_EXTENDED_EPOCHS:
            recent = history[-25:]
            preceding = history[-50:-25]
            recent_best = min((r["validationLoss"] for r in recent), default=math.inf)
            preceding_best = min((r["validationLoss"] for r in preceding), default=math.inf)
            recent_updated = any(r["improved"] for r in recent)
            if (recent_updated and math.isfinite(preceding_best)
                    and recent_best <= preceding_best * (1.0 - 0.005)):
                final_limit = MAX_EXTENDED_EPOCHS
            else:
                terminal_reason = "extension_rule_not_met"
        elif epoch >= final_limit:
            terminal_reason = "maximum_epoch_complete"
        _save_checkpoint(checkpoint_path, {
            "schemaVersion": 1, "fitId": fit_id, "fitIdentity": fit_identity, "product": product,
            "contextDays": context_days, "inputType": input_type, "repair": repair,
            "epoch": epoch, "bestEpoch": best_epoch, "bestValidationLoss": best_loss,
            "bestState": best_state,
            "noImprove": no_improve, "modelState": model.state_dict(),
            "optimizerState": optimizer.state_dict(), "history": history,
            "pythonRandomState": random.getstate(), "torchRandomState": torch.get_rng_state(),
            "numpyRandomState": np_runtime.random.get_state(), "device": str(device),
            "mpsRandomState": torch.mps.get_rng_state() if device.type == "mps" else None,
            "gradientNormAbnormalEvents": grad_norm_events,
            "oomFallbackUsed": oom_fallback_used,
            "smallBatchMode": small_batch_mode, "fitLimits": fit_limits,
            "finalLimit": final_limit, "terminalReason": terminal_reason,
            "elapsedSeconds": elapsed_prior + time.monotonic() - started_at,
        }, torch)
        _write_replace(PROGRESS_PATH, _canonical({
            "schemaVersion": 1, "experiment": EXPERIMENT_ID, "fitId": fit_id,
            "status": "running" if terminal_reason is None else "fit_complete",
            "epoch": epoch, "finalLimit": final_limit, "fixedEpochs": fixed_epochs,
            "terminalReason": terminal_reason,
            "optimizerUpdatesThisFit": sum(int(row.get("updates", 0)) for row in history),
            "oomFallbackUsed": oom_fallback_used, "smallBatchMode": small_batch_mode,
            "elapsedSeconds": elapsed_prior + time.monotonic() - started_at,
            "checkpointPath": str(checkpoint_path.relative_to(OUT)),
            "checkpointSha256": _file_sha(checkpoint_path)}))
    if fixed_epochs is not None:
        # Fixed-epoch refits use the final prescribed update, not an epoch
        # selected by looking again at the training loss.
        best_epoch = epoch
        best_loss = history[-1]["validationLoss"] if history else float("inf")
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError(f"fit {fit_id} did not produce a validation checkpoint")
    model.load_state_dict(best_state)
    return {"fitId": fit_id, "product": product, "contextDays": context_days,
            "inputType": input_type, "repair": repair, "bestEpoch": best_epoch,
            "bestValidationLoss": best_loss, "history": history,
            "extensionUsed": len(history) > MAX_BASE_EPOCHS,
            "extraEpochs": max(0, len(history) - MAX_BASE_EPOCHS),
            "gradientNormAbnormalEvents": grad_norm_events,
            "oomFallbackUsed": oom_fallback_used, "device": str(device),
            "smallBatchMode": small_batch_mode, "terminalReason": terminal_reason,
            "fitLimits": fit_limits, "wallClockSeconds": elapsed_prior + time.monotonic() - started_at,
            "optimizerUpdates": sum(int(row.get("updates", 0)) for row in history),
            "model": model, "learningRate": learning_rate,
            "weightDecay": weight_decay, "maeWeight": mae_weight,
            "qScale": q_scale, "checkpointPath": str(checkpoint_path),
            "checkpointSha256": _file_sha(checkpoint_path)}


def _safe_number(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(k): _safe_number(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_number(v) for v in value]
    return value


def _save_fit_result(fit: Mapping[str, Any], selected_details: Sequence[Mapping[str, Any]],
                     fit_id: str) -> dict[str, Any]:
    _, torch, _ = _torch_runtime()
    models = OUT / "models"
    models.mkdir(parents=True, exist_ok=True)
    model_path = models / f"{fit_id}.pt"
    _save_frozen_checkpoint(model_path, {"schemaVersion": 1, "fitId": fit_id,
                                        "experiment": EXPERIMENT_ID,
                                        "contextDays": fit["contextDays"],
                                        "inputType": fit["inputType"], "repair": fit["repair"],
                                        "bestEpoch": fit["bestEpoch"],
                                        "selectedInputChannels": list(selected_details),
                                        "stateDict": {k: v.detach().cpu() for k, v in fit["model"].state_dict().items()}}, torch)
    record = {k: v for k, v in fit.items() if k != "model"}
    record["weightPath"] = str(model_path.relative_to(OUT))
    record["weightSha256"] = _file_sha(model_path)
    record["checkpointPath"] = str(Path(record["checkpointPath"]).relative_to(OUT))
    ledger = _read_json(LEDGER_PATH) if LEDGER_PATH.exists() else {
        "schemaVersion": 1, "experiment": EXPERIMENT_ID,
        "protocolSha256": PROTOCOL_SHA256, "fits": {}}
    prior = ledger.setdefault("fits", {}).get(fit_id)
    if prior is not None and prior.get("weightSha256") != record["weightSha256"]:
        raise FileExistsError(f"fit ledger already contains a different completed fit: {fit_id}")
    ledger["fits"][fit_id] = _safe_number(record)
    ledger.pop("ledgerSha256", None)
    ledger["ledgerSha256"] = _sha(_canonical(ledger))
    _write_replace(LEDGER_PATH, _canonical(ledger))
    _write_replace(PROGRESS_PATH, _canonical({
        "schemaVersion": 1, "experiment": EXPERIMENT_ID, "fitId": fit_id,
        "status": "completed", "completedFits": len(ledger["fits"]),
        "checkpointPath": record["checkpointPath"],
        "checkpointSha256": record["checkpointSha256"],
        "modelPath": record["weightPath"], "modelSha256": record["weightSha256"],
        "wallClockSeconds": record.get("wallClockSeconds"),
        "optimizerUpdates": record.get("optimizerUpdates"),
        "terminalReason": record.get("terminalReason")}))
    result = {k: v for k, v in record.items() if k in (
        "fitId", "product", "contextDays", "inputType", "repair", "bestEpoch",
        "bestValidationLoss", "extensionUsed", "extraEpochs", "gradientNormAbnormalEvents",
        "oomFallbackUsed", "smallBatchMode", "terminalReason", "fitLimits",
        "wallClockSeconds", "optimizerUpdates", "device", "learningRate", "weightDecay", "maeWeight",
        "qScale", "weightPath", "weightSha256", "checkpointPath", "checkpointSha256")}
    result["history"] = fit["history"]
    return result


def _load_saved_fit_model(model_path: Path, *, expected_fit_id: str,
                          expected_details: Sequence[Mapping[str, Any]]) -> tuple[Any, dict[str, Any]]:
    """Reload only the immutable saved weights and their persisted input schema."""
    _, torch, (nn, F) = _torch_runtime()
    if not model_path.is_file():
        raise FileNotFoundError(f"saved model missing before replay: {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    if (checkpoint.get("experiment") != EXPERIMENT_ID
            or checkpoint.get("fitId") != expected_fit_id
            or checkpoint.get("contextDays") not in CONTEXTS
            or checkpoint.get("inputType") not in ("U", "M")
            or checkpoint.get("selectedInputChannels") != list(expected_details)):
        raise ValueError(f"saved model input schema does not match the final fit: {expected_fit_id}")
    model = build_model(int(checkpoint["contextDays"]), str(checkpoint["repair"]),
                        torch=torch, nn=nn, F=F)
    model.load_state_dict(checkpoint["stateDict"], strict=True)
    model.to(_device(torch))
    model.eval()
    return model, checkpoint


def _inference_payload_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Hash the full origin-bounded model inputs and calendar payload in order."""
    payload = [{"product": r["product"], "origin": _iso_date(r["origin"]),
                "window": r["window"], "calendar": r["calendar"]}
               for r in records]
    return _sha(_canonical(_safe_number(payload)))


def _iso_date(value: dt.date | str) -> str:
    return value.isoformat() if isinstance(value, dt.date) else str(value)


def _persist_replay_payload(phase: str, fit_id: str, model_sha256: str,
                            schema: Mapping[str, Any],
                            records: Sequence[Mapping[str, Any]]) -> tuple[Path, dict[str, Any]]:
    """Persist exact answer-free inputs and schema, then verify the readback."""
    payload_records = [{"product": r["product"], "origin": _iso_date(r["origin"]),
                        "window": r["window"], "calendar": r["calendar"],
                        "targetDays": r["targetDays"]}
                       for r in records]
    schema_doc = dict(schema)
    schema_sha = _sha(_canonical(schema_doc))
    payload = {"schemaVersion": 1, "fitId": fit_id,
               "modelSha256": model_sha256, "schema": schema_doc,
               "schemaSha256": schema_sha, "records": payload_records,
               "rowPayloadSha256": _inference_payload_digest(payload_records),
               "rowPayloadCount": len(payload_records)}
    path = OUT / phase / "replay-inputs" / f"{fit_id}.json"
    _write_once(path, _canonical(payload))
    loaded = _read_json(path)
    if (loaded != payload
            or loaded.get("rowPayloadSha256") != _inference_payload_digest(loaded.get("records", []))
            or loaded.get("schemaSha256") != _sha(_canonical(loaded.get("schema")))):
        raise ValueError(f"persisted replay input payload failed readback validation: {fit_id}")
    return path, loaded


def _records_from_replay_payload(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    if payload.get("schemaVersion") != 1 or not isinstance(payload.get("records"), list):
        raise ValueError("unsupported or malformed persisted replay payload")
    records = []
    for row in payload["records"]:
        try:
            origin = dt.date.fromisoformat(row["origin"])
            if row["product"] not in data.PRODUCTS:
                raise ValueError("unknown product")
            records.append({**row, "origin": origin,
                            "targets": [0.0] * len(HORIZONS),
                            "targetMask": [False] * len(HORIZONS)})
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed persisted replay forecast row") from exc
    return records


def _replay_prediction_digest(records: Sequence[Mapping[str, Any]],
                              predictions: Sequence[Sequence[float]]) -> str:
    payload = [{"product": r["product"], "origin": _iso_date(r["origin"]),
                "predictions": [_raw_prediction_value(v) for v in pred]}
               for r, pred in zip(records, predictions)]
    return _sha(_canonical(payload))


def _prediction_class(value: float) -> str:
    value = float(value)
    if math.isnan(value):
        return "nan"
    if value == math.inf:
        return "positive_infinity"
    if value == -math.inf:
        return "negative_infinity"
    if value > 0:
        return "issued_positive"
    if value == 0:
        return "zero"
    return "negative_finite"


def _raw_prediction_value(value: float) -> float | str:
    value = float(value)
    return value if math.isfinite(value) else _prediction_class(value)


def _compare_saved_model_replay(original: Sequence[Sequence[float]],
                                reloaded: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Compare every raw horizon prediction and report bounded numeric error."""
    import numpy as np
    original_values = np.asarray(original, dtype=np.float64)
    reloaded_values = np.asarray(reloaded, dtype=np.float64)
    if original_values.shape != reloaded_values.shape:
        raise ValueError("saved-model replay prediction shape changed")
    finite_pair = np.isfinite(original_values) & np.isfinite(reloaded_values)
    original_classes = np.vectorize(_prediction_class)(original_values)
    reloaded_classes = np.vectorize(_prediction_class)(reloaded_values)
    matching_class = original_classes == reloaded_classes
    issue_class_agreement = bool(np.all(matching_class))
    equal_nonfinite = (~np.isfinite(original_values)) & (~np.isfinite(reloaded_values)) & matching_class
    numerical_agreement = bool(np.all(np.isclose(original_values[finite_pair], reloaded_values[finite_pair],
                                                atol=RELOAD_REPLAY_ATOL,
                                                rtol=RELOAD_REPLAY_RTOL)))
    all_agree = numerical_agreement and issue_class_agreement
    if np.any(finite_pair):
        absolute = np.abs(original_values[finite_pair] - reloaded_values[finite_pair])
        relative = absolute / np.maximum(np.abs(original_values[finite_pair]), np.finfo(np.float64).tiny)
        max_abs = float(np.max(absolute))
        max_rel = float(np.max(relative))
        max_original_magnitude = float(np.max(np.abs(original_values[finite_pair])))
    else:
        max_abs = max_rel = max_original_magnitude = 0.0
    return {"predictionValueCount": int(original_values.size),
            "inferenceRowCount": int(original_values.shape[0]),
            "horizonCount": int(original_values.shape[1]) if original_values.ndim == 2 else None,
            "predictionArrayShape": list(original_values.shape),
            "allRowsAgree": all_agree,
            "issueClassAgreement": issue_class_agreement,
            "finitePairCount": int(finite_pair.sum()),
            "matchingNonfinitePairCount": int(equal_nonfinite.sum()),
            "maxOriginalMagnitude": max_original_magnitude,
            "maxAbsoluteDifference": max_abs,
            "maxRelativeDifference": max_rel,
            "absoluteTolerance": RELOAD_REPLAY_ATOL,
            "relativeTolerance": RELOAD_REPLAY_RTOL}


def _state_dict_agreement(original_model: Any, reloaded_model: Any) -> dict[str, Any]:
    _, torch, _ = _torch_runtime()
    original = original_model.state_dict()
    reloaded = reloaded_model.state_dict()
    same_keys = set(original) == set(reloaded)
    same = same_keys and all(torch.equal(original[name].detach().cpu(),
                                         reloaded[name].detach().cpu())
                             for name in original)
    return {"stateAgreement": bool(same), "stateTensorCount": len(original),
            "stateKeyAgreement": bool(same_keys)}


def _forecast_rows_for_fit_digest(rows: Sequence[Mapping[str, Any]], fit: Mapping[str, Any]) -> str:
    """Bind a replay to the final keyed U/M forecast values derived from it."""
    input_type = str(fit["inputType"])
    month = str(fit["fitMonth"])
    product = str(fit["product"])
    payload = []
    for row in rows:
        if row["product"] != product or str(row["origin"])[:7] != month:
            continue
        output = row[input_type]
        payload.append({"product": row["product"], "origin": row["origin"],
                        "target": row["target"], "horizon": int(row["horizon"]),
                        "inputType": input_type, "prediction": output.get("rawPrediction"),
                        "status": output.get("rawStatus"),
                        "numericKind": output.get("rawNumericKind"),
                        "rawReason": output.get("rawReason")})
    payload.sort(key=lambda r: (r["product"], r["origin"], r["target"], r["horizon"]))
    return _sha(_canonical(_safe_number(payload)))


def _selected_channels(channels: Mapping[str, Any], product: str,
                       fit_boundary: dt.date) -> list[dict[str, Any]]:
    return data.select_channels_for_cutoff(channels.get(product, {}),
                                           fit_boundary - dt.timedelta(days=1))


def _train_and_evaluate(product: str, context: int, input_type: str, repair: str,
                        fit_boundary: dt.date, validation_start: dt.date,
                        validation_end: dt.date, representative: Mapping[dt.date, float],
                        channels: Mapping[str, Any], *, fit_id: str,
                        selected_details: Sequence[Mapping[str, Any]] | None = None,
                        max_epochs: int = MAX_BASE_EPOCHS,
                        fixed_epochs: int | None = None) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if selected_details is None:
        selected_details = _selected_channels(channels, product, fit_boundary)
    train = _records_for_period(product, representative, channels.get(product, {}), context,
                                input_type, selected_details, min(representative), fit_boundary,
                                label_before=fit_boundary)
    val = (train if fixed_epochs is not None else
           _records_for_period(product, representative, channels.get(product, {}), context,
                               input_type, selected_details, validation_start, validation_end,
                               label_before=None, label_through=validation_end))
    if not train or not val:
        raise ValueError(f"empty training/validation set in {fit_id}")
    q = _q_scale(representative, fit_boundary)
    checkpoint = OUT / "checkpoints" / f"{fit_id}.pt"
    fit = fit_model(product, context, input_type, repair, train, val, q, fit_id,
                    checkpoint, max_epochs=max_epochs,
                    extension_allowed=fixed_epochs is None, fixed_epochs=fixed_epochs)
    summary = {"fitId": fit_id, "qScale": q, "trainRows": len(train),
               "validationRows": len(val), "bestEpoch": fit["bestEpoch"],
               "bestValidationLoss": fit["bestValidationLoss"],
               "abnormalGradientEvents": fit["gradientNormAbnormalEvents"],
               "history": fit["history"]}
    return fit, summary, train


def _outer_evaluate(fit: Mapping[str, Any], product: str, context: int, input_type: str,
                    selected_details: Sequence[Mapping[str, Any]],
                    representative: Mapping[dt.date, float], channels: Mapping[str, Any],
                    start: dt.date, end: dt.date, fit_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _, torch, _ = _torch_runtime()
    device = _device(torch)
    records = _records_for_period(product, representative, channels.get(product, {}), context,
                                  input_type, selected_details, start, end,
                                  label_before=None, label_through=end - dt.timedelta(days=1))
    predictions = _predict_records(fit["model"], records, device, torch,
                                   batch_size=32 if fit.get("smallBatchMode") else BATCH_SIZE)
    metrics = _metric_summary(records, predictions)
    metrics["activeAuxRows"] = sum(r["window"]["activeAuxCount"] > 0 for r in records)
    metrics["activeAuxCoverage"] = (metrics["activeAuxRows"] / len(records) if records else 0.0)
    metrics["activeAuxCountMean"] = (statistics.mean(r["window"]["activeAuxCount"] for r in records)
                                    if records else 0.0)
    return metrics, records


def _ratio(numerator: Any, denominator: Any) -> float:
    if numerator is None or denominator is None:
        return math.inf
    numerator, denominator = float(numerator), float(denominator)
    if denominator == 0:
        return 1.0 if numerator == 0 else math.inf
    return numerator / denominator


def _candidate_j(metrics: Mapping[str, Any]) -> float:
    model, p0 = metrics.get("model", {}), metrics.get("p0", {})
    if model.get("coverageComplete") is False or p0.get("coverageComplete") is False:
        return math.inf
    return 0.5 * (_ratio(model.get("MAPE"), p0.get("MAPE"))
                  + _ratio(model.get("MAE"), p0.get("MAE")))


def _select_2025_input(scores: Mapping[str, float]) -> str:
    if set(scores) != {"U", "M"}:
        raise ValueError("2025 input selection requires exactly U and M")
    if all(not math.isfinite(float(scores[k])) for k in ("U", "M")):
        return "U"
    return min(("U", "M"), key=lambda it: (float(scores[it]), 0 if it == "U" else 1))


def _finite_oscillation(history: Sequence[Mapping[str, Any]]) -> bool:
    events = []
    for i in range(len(history)):
        previous_best = min(float(x["validationLoss"]) for x in history[:i + 1])
        loss = float(history[i]["validationLoss"])
        if loss <= previous_best * 1.05:
            continue
        recovery = next((j for j in range(i + 1, min(len(history), i + 6))
                         if float(history[j]["validationLoss"]) <= loss * 0.98), None)
        if recovery is None:
            continue
        curve = history[i:recovery + 1]
        if any(float(x.get("gradientNormMaxBeforeClip", math.inf)) > GRADIENT_NORM_ABNORMAL
               for x in curve):
            continue
        if not events or i - events[-1] >= 5:
            events.append(i)
    return len(events) >= 3


def _trigger_repair(base_results: Mapping[str, Any], product: str, context: int) -> str | None:
    # A single issue must recur at the declared fold/product level before any
    # additional setting consumes the capped repair budget.
    rows = {(r["fold"], r["inputType"]): r for r in base_results["outer"]
            if r["product"] == product and r["contextDays"] == context}
    fits = {(r["fold"], r["inputType"]): r for r in base_results["inner"]
            if r["product"] == product and r["contextDays"] == context}
    if any(_finite_oscillation(fits[(f, input_type)]["history"])
           for f in ("F23", "F24") for input_type in ("U", "M")):
        return "lr"
    for input_type in ("U", "M"):
        overfit = True
        for fold in ("F23", "F24"):
            history = fits[(fold, input_type)]["history"]
            if not history:
                overfit = False
                break
            best_i = min(range(len(history)), key=lambda i: history[i]["validationLoss"])
            tail = history[-10:]
            train_at_best = history[best_i]["trainLoss"]
            overfit &= (statistics.mean(x["validationLoss"] for x in tail) >= history[best_i]["validationLoss"] * 1.02
                        and statistics.mean(x["trainLoss"] for x in tail) <= train_at_best * 0.98)
        if overfit:
            return "regularization"
    for input_type in ("U", "M"):
        if all(rows[(f, input_type)]["metrics"]["model"].get("coverageComplete") is True
               and rows[(f, input_type)]["metrics"]["p0"].get("coverageComplete") is True
               and rows[(f, input_type)]["metrics"]["model"]["MAPE"] is not None
               and rows[(f, input_type)]["metrics"]["p0"]["MAPE"] is not None
               and rows[(f, input_type)]["metrics"]["model"]["MAE"] is not None
               and rows[(f, input_type)]["metrics"]["p0"]["MAE"] is not None
               and rows[(f, input_type)]["metrics"]["model"]["MAPE"] < rows[(f, input_type)]["metrics"]["p0"]["MAPE"]
               and rows[(f, input_type)]["metrics"]["model"]["MAE"] > rows[(f, input_type)]["metrics"]["p0"]["MAE"]
               for f in ("F23", "F24")):
            return "mae_weight"
    if all(_candidate_j(rows[(f, "M")]["metrics"]) > _candidate_j(rows[(f, "U")]["metrics"])
           and rows[(f, "M")]["activeAuxCoverage"] > 0 for f in ("F23", "F24")):
        return "aux_dropout"
    return None


def run_development() -> dict[str, Any]:
    """Run the frozen F23/F24 selection, capped repair and 2025 input decision."""
    _assert_price_snapshot()
    if _file_sha(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValueError("protocol changed after implementation review")
    validate_resolved_config()
    source, source_doc = data.load_source_cache()
    global _SOURCE_CACHE_INTERNAL_SHA
    _SOURCE_CACHE_INTERNAL_SHA = str(source_doc["sourceCacheSha256"])
    prices = _read_json(PRICE_PATH)
    representatives = data.representative_series(prices)
    folds = DEV_FOLDS
    base: dict[str, Any] = {"inner": [], "outer": []}
    all_fits: list[dict[str, Any]] = []

    def fit_fold(product: str, context: int, input_type: str, repair: str,
                 fold_name: str, *, fit_prefix: str | None = None,
                 selected: Sequence[Mapping[str, Any]] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        fold = folds[fold_name]
        suffix = f"{product}-L{context}-{input_type}-{repair}-{fold_name}"
        inner_id = f"{fit_prefix or 'dev'}-{suffix}-inner"
        details = list(selected) if selected is not None else _selected_channels(source, product, fold["innerFitBoundary"])
        cache_path = OUT / "development-cache" / f"{fit_prefix or 'base'}-{suffix}.json"
        cache_identity = {"fold": fold_name, "product": product, "contextDays": context,
                          "inputType": input_type, "repair": repair,
                          "selectedChannels": details,
                          "protocolSha256": PROTOCOL_SHA256,
                          "sourceCacheSha256": source_doc["sourceCacheSha256"],
                          "resolvedConfigSha256": _file_sha(CONFIG_PATH)}
        if cache_path.is_file():
            cached = _read_json(cache_path)
            _self_hash(cached, "cacheSha256", f"development fold cache {cache_path.name}")
            if cached.get("identity") != cache_identity:
                raise ValueError(f"development fold cache identity mismatch: {cache_path}")
            for fit_record in cached.get("completedFits", []):
                weight_path = OUT / fit_record["weightPath"]
                checkpoint_path = OUT / fit_record["checkpointPath"]
                if (not weight_path.is_file() or _file_sha(weight_path) != fit_record["weightSha256"]
                        or not checkpoint_path.is_file()
                        or _file_sha(checkpoint_path) != fit_record["checkpointSha256"]):
                    raise ValueError(f"development fit artifact missing/changed for {fit_record['fitId']}")
            all_fits.extend(cached["completedFits"])
            return cached["innerSummary"], cached["outerSummary"]
        inner_fit, inner_summary, _ = _train_and_evaluate(
            product, context, input_type, repair, fold["innerFitBoundary"],
            fold["innerFitBoundary"], fold["validationEnd"], representatives[product], source,
            fit_id=inner_id, selected_details=details, max_epochs=MAX_EXTENDED_EPOCHS)
        inner_summary.update({"fold": fold_name, "product": product,
                              "contextDays": context, "inputType": input_type,
                              "repair": repair, "selectedChannels": details})
        all_fits.append(_save_fit_result(inner_fit, details, inner_id))

        refit_id = f"{fit_prefix or 'dev'}-{suffix}-refit"
        refit_fit, refit_summary, _ = _train_and_evaluate(
            product, context, input_type, repair, fold["refitBoundary"],
            fold["refitBoundary"], fold["refitBoundary"] + dt.timedelta(days=1),
            representatives[product], source, fit_id=refit_id,
            selected_details=details, fixed_epochs=inner_fit["bestEpoch"])
        refit_summary.update({"fold": fold_name, "product": product,
                              "contextDays": context, "inputType": input_type,
                              "repair": repair, "selectedChannels": details,
                              "innerBestEpoch": inner_fit["bestEpoch"]})
        all_fits.append(_save_fit_result(refit_fit, details, refit_id))
        score, _ = _outer_evaluate(refit_fit, product, context, input_type, details,
                                   representatives[product], source,
                                   fold["outerStart"], fold["outerEnd"] + dt.timedelta(days=1),
                                   refit_id)
        outer_summary = {"fold": fold_name, "product": product,
                         "contextDays": context, "inputType": input_type,
                         "repair": repair, "fitId": refit_id,
                         "innerBestEpoch": inner_fit["bestEpoch"],
                         "metrics": score, "activeAuxCoverage": score["activeAuxCoverage"],
                         "selectedChannels": details}
        cache = {"schemaVersion": 1, "identity": cache_identity,
                 "completedFits": [all_fits[-2], all_fits[-1]],
                 "innerSummary": inner_summary, "outerSummary": outer_summary}
        cache["cacheSha256"] = _sha(_canonical(cache))
        _write_once(cache_path, _canonical(cache))
        return inner_summary, outer_summary

    # Paired input types and contexts use identical calendar keys and fold rules.
    for product in data.PRODUCTS:
        for context in CONTEXTS:
            for input_type in ("U", "M"):
                for fold_name in ("F23", "F24"):
                    inner, outer = fit_fold(product, context, input_type, "base", fold_name)
                    base["inner"].append(inner)
                    base["outer"].append(outer)
    if len(all_fits) != 64:
        raise AssertionError("frozen base-fit budget must be exactly 64 completed fits")

    selected_context: dict[str, int] = {}
    base_context_scores: dict[str, Any] = {}
    for product in data.PRODUCTS:
        candidates = {}
        for context in CONTEXTS:
            context_rows = [r for r in base["outer"] if r["product"] == product
                            and r["contextDays"] == context and r["repair"] == "base"]
            vals = [_candidate_j(r["metrics"]) for r in context_rows]
            candidates[context] = statistics.mean(vals)
        if all(not math.isfinite(v) for v in candidates.values()):
            def fallback_key(c: int) -> tuple[float, float, int]:
                rows_for_context = [r for r in base["outer"] if r["product"] == product
                                    and r["contextDays"] == c and r["repair"] == "base"]
                metrics = [r["metrics"]["model"] for r in rows_for_context]
                mape = statistics.mean(float(m["MAPE"]) if m["MAPE"] is not None
                                       and m.get("coverageComplete") else math.inf for m in metrics)
                mae = statistics.mean(float(m["MAE"]) if m["MAE"] is not None
                                      and m.get("coverageComplete") else math.inf for m in metrics)
                return mape, mae, c
            best = min(CONTEXTS, key=fallback_key)
        else:
            best = min(candidates, key=lambda c: (candidates[c], c))
        selected_context[product] = best
        base_context_scores[product] = {"scores": _safe_number({str(k): v for k, v in candidates.items()}),
                                        "selected": best,
                                        "allInfinite": all(not math.isfinite(v) for v in candidates.values())}

    selected_settings = {product: "base" for product in data.PRODUCTS}
    repair_summary: dict[str, Any] = {}
    for product in data.PRODUCTS:
        context = selected_context[product]
        trigger = _trigger_repair(base, product, context)
        repair_summary[product] = {"selectedContext": context, "trigger": trigger,
                                   "fitCount": 0, "recovered": False}
        if trigger is None:
            continue
        start_count = len(all_fits)
        repaired = {"inner": [], "outer": []}
        for input_type in ("U", "M"):
            for fold_name in ("F23", "F24"):
                inner, outer = fit_fold(product, context, input_type, trigger, fold_name,
                                        fit_prefix="repair")
                repaired["inner"].append(inner)
                repaired["outer"].append(outer)
        count = len(all_fits) - start_count
        if count != 8:
            raise AssertionError("one repair setting requires exactly eight paired fit stages")
        repair_summary[product]["fitCount"] = count
        base_score = statistics.mean(_candidate_j(r["metrics"]) for r in base["outer"]
                                     if r["product"] == product and r["contextDays"] == context
                                     and r["repair"] == "base")
        repair_score = statistics.mean(_candidate_j(r["metrics"]) for r in repaired["outer"])
        repair_summary[product].update({"baseCommonScore": _safe_number(base_score),
                                        "repairCommonScore": _safe_number(repair_score),
                                        "recovered": math.isfinite(repair_score) and repair_score < base_score})
        base["inner"].extend(repaired["inner"])
        base["outer"].extend(repaired["outer"])
        if repair_score < base_score:
            selected_settings[product] = trigger
    repair_fit_count = len(all_fits) - 64
    if repair_fit_count > 32:
        raise AssertionError("global repair fit budget exceeds 32")

    scores_2025: dict[str, Any] = {}
    policy: dict[str, Any] = {"schemaVersion": 1, "experiment": EXPERIMENT_ID,
                              "protocolSha256": PROTOCOL_SHA256,
                              "sourceCacheSha256": source_doc["sourceCacheSha256"],
                              "targetSnapshotSha256": PRICE_SHA256,
                              "selectionRule": "single input type per product across all 10 horizons",
                              "hybridWeightB": 0.5, "hybridWeightN": 0.5,
                              "selectedInputByProduct": {}, "productPolicy": {}}
    for product in data.PRODUCTS:
        context = selected_context[product]
        setting = selected_settings[product]
        by_input = {}
        for input_type in ("U", "M"):
            inner_rows = [r for r in base["inner"] if r["product"] == product
                          and r["contextDays"] == context and r["inputType"] == input_type
                          and r["repair"] == setting and r["fold"] in ("F23", "F24")]
            if len(inner_rows) != 2:
                raise ValueError(f"incomplete selected-setting epoch record: {product}/{input_type}")
            fixed_epoch = int(math.floor(statistics.median(r["bestEpoch"] for r in inner_rows) + 0.5))
            fit_id = f"year2025-{product}-L{context}-{input_type}-{setting}"
            boundary = dt.date(2025, 1, 1)
            details = _selected_channels(source, product, boundary)
            y25_cache_path = OUT / "development-cache" / f"{fit_id}.json"
            y25_identity = {"fitId": fit_id, "product": product, "contextDays": context,
                            "inputType": input_type, "repair": setting, "fixedEpoch": fixed_epoch,
                            "selectedChannels": details, "protocolSha256": PROTOCOL_SHA256,
                            "sourceCacheSha256": source_doc["sourceCacheSha256"],
                            "resolvedConfigSha256": _file_sha(CONFIG_PATH)}
            if y25_cache_path.is_file():
                cached = _read_json(y25_cache_path)
                _self_hash(cached, "cacheSha256", f"2025 development cache {fit_id}")
                if cached.get("identity") != y25_identity:
                    raise ValueError(f"2025 development cache identity mismatch: {fit_id}")
                saved_fit = cached["completedFit"]
                weight_path = OUT / saved_fit["weightPath"]
                checkpoint_path = OUT / saved_fit["checkpointPath"]
                if (not weight_path.is_file() or _file_sha(weight_path) != saved_fit["weightSha256"]
                        or not checkpoint_path.is_file()
                        or _file_sha(checkpoint_path) != saved_fit["checkpointSha256"]):
                    raise ValueError(f"2025 development fit artifact missing/changed: {fit_id}")
                summary, metric = cached["fitSummary"], cached["metrics"]
                all_fits.append(saved_fit)
            else:
                fit, summary, _ = _train_and_evaluate(
                    product, context, input_type, setting, boundary,
                    boundary, dt.date(2026, 1, 1), representatives[product], source,
                    fit_id=fit_id, selected_details=details, fixed_epochs=fixed_epoch)
                saved_fit = _save_fit_result(fit, details, fit_id)
                all_fits.append(saved_fit)
                metric, _ = _outer_evaluate(fit, product, context, input_type, details,
                                            representatives[product], source,
                                            boundary, dt.date(2026, 1, 1), fit_id)
                y25_cache = {"schemaVersion": 1, "identity": y25_identity,
                             "completedFit": saved_fit, "fitSummary": summary, "metrics": metric}
                y25_cache["cacheSha256"] = _sha(_canonical(y25_cache))
                _write_once(y25_cache_path, _canonical(y25_cache))
            years = {}
            for year, fold_name in (("2023", "F23"), ("2024", "F24")):
                year_row = next(r for r in base["outer"] if r["product"] == product
                                and r["contextDays"] == context and r["inputType"] == input_type
                                and r["repair"] == setting and r["fold"] == fold_name)
                years[year] = _safe_number(_candidate_j(year_row["metrics"]))
            years["2025"] = _safe_number(_candidate_j(metric))
            by_input[input_type] = {"fitId": fit_id, "fixedEpoch": fixed_epoch,
                                    "metrics": metric, "developmentJ": years,
                                    "developmentJStatus": {year: ("positive_infinity" if not math.isfinite(float(value)) else "finite")
                                                          for year, value in {
                                                              "2023": _candidate_j(next(r["metrics"] for r in base["outer"]
                                                                  if r["product"] == product and r["contextDays"] == context
                                                                  and r["inputType"] == input_type and r["repair"] == setting
                                                                  and r["fold"] == "F23")),
                                                              "2024": _candidate_j(next(r["metrics"] for r in base["outer"]
                                                                  if r["product"] == product and r["contextDays"] == context
                                                                  and r["inputType"] == input_type and r["repair"] == setting
                                                                  and r["fold"] == "F24")),
                                                              "2025": _candidate_j(metric)}.items()},
                                    "fitSummary": summary}
        scores = {it: statistics.mean(by_input[it]["developmentJ"][year]
                                      if by_input[it]["developmentJ"][year] is not None else math.inf
                                      for year in ("2023", "2024", "2025")) for it in ("U", "M")}
        selected_input = _select_2025_input(scores)
        policy_scores_status = {it: ("positive_infinity" if not math.isfinite(scores[it]) else "finite")
                                for it in ("U", "M")}
        policy["selectedInputByProduct"][product] = selected_input
        policy["productPolicy"][product] = {
            "contextDays": context, "repairSetting": setting,
            "selectedInput": selected_input,
            "fixedEpochByInput": {it: by_input[it]["fixedEpoch"] for it in ("U", "M")},
            "learningRate": 0.0003 if setting == "lr" else INITIAL_LR,
            "weightDecay": 0.001 if setting == "regularization" else INITIAL_WEIGHT_DECAY,
            "maeLossWeight": 3.0 if setting == "mae_weight" else 1.0,
            "auxiliaryDropout": 0.3 if setting == "aux_dropout" else INITIAL_AUX_DROPOUT,
            "developmentScores": by_input,
            "developmentScoreStatus": policy_scores_status,
            "selectedContextScores": base_context_scores[product],
            "repair": repair_summary[product]}
        scores_2025[product] = by_input
    policy["modelPolicySha256"] = _sha(_canonical(policy))
    _write_once(POLICY_PATH, _canonical(policy))
    dev_doc = {"schemaVersion": 1, "experiment": EXPERIMENT_ID,
               "protocolSha256": PROTOCOL_SHA256,
               "sourceCacheSha256": source_doc["sourceCacheSha256"],
               "baseFits": 64, "repairFits": repair_fit_count,
               "year2025Fits": 8, "totalCompletedFits": len(all_fits),
               "base": _safe_number(base), "repair": _safe_number(repair_summary),
               "year2025": _safe_number(scores_2025),
               "selectedInputByProduct": policy["selectedInputByProduct"],
               "modelPolicySha256": policy["modelPolicySha256"],
               "fitArtifacts": [{k: v for k, v in item.items() if k != "history"}
                                 for item in all_fits]}
    dev_doc["developmentScoresSha256"] = _sha(_canonical(dev_doc))
    _write_once(OUT / "development-scores.json", _canonical(dev_doc))
    return dev_doc


def _baseline_phase(phase: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if phase not in ("V", "T"):
        raise ValueError("phase must be V or T")
    manifest_path = baseline_source.BASELINE_PATHS[phase]
    if _file_sha(manifest_path) != baseline_source.BASELINE_SHA256[phase]:
        raise ValueError(f"pinned {phase} baseline manifest hash mismatch")
    manifest = _read_json(manifest_path)
    forecast_path = manifest_path.parent / manifest["forecastFile"]
    if _file_sha(forecast_path) != manifest.get("forecastFileSha256"):
        raise ValueError(f"pinned {phase} baseline forecast file hash mismatch")
    if (manifest.get("phase") != phase
            or manifest.get("protocolSha256") != BASELINE_PROTOCOL_SHA256):
        raise ValueError(f"{phase} pinned baseline phase/protocol mismatch")
    nested = manifest.get("forecasts")
    if not isinstance(nested, list) or not nested:
        raise ValueError(f"{phase} baseline manifest has no nested forecasts")
    expected_header = list(_BASELINE_CSV_FIELDS)
    with forecast_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected_header:
            raise ValueError(f"{phase} baseline CSV header differs from the pinned schema")
        flat_rows = list(reader)
    flat: dict[tuple[str, str, str, int, str], dict[str, str]] = {}
    for row in flat_rows:
        try:
            key = (str(row["product"]), str(row["originDate"]), str(row["targetDate"]),
                   int(row["horizon"]), str(row["condition"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"malformed {phase} baseline CSV row") from exc
        if key in flat:
            raise ValueError(f"duplicate {phase} baseline CSV key {key}")
        if row.get("origin") != key[1] or row.get("target") != key[2]:
            raise ValueError(f"baseline CSV display dates disagree with key dates: {key}")
        flat[key] = row
    expected_flat_keys: set[tuple[str, str, str, int, str]] = set()
    nested_keys: set[tuple[str, str, str, int]] = set()
    normalized: list[dict[str, Any]] = []
    start, end, origin_start = baseline_source.PHASES[phase]
    def optional_float(value: Any, key: tuple[str, str, str, int, str], field: str) -> float | None:
        if value is None or value == "":
            return None
        try:
            out = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid {field} in pinned {phase} baseline row {key}") from exc
        if not math.isfinite(out):
            raise ValueError(f"nonfinite {field} in pinned {phase} baseline row {key}")
        return out

    for source_row in nested:
        try:
            product = str(source_row["product"])
            origin = dt.date.fromisoformat(str(source_row["originDate"]))
            target = dt.date.fromisoformat(str(source_row["targetDate"]))
            horizon = int(source_row["horizon"])
            conditions = source_row["conditions"]
            eligible = source_row["eligible"]
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"malformed nested {phase} baseline forecast row") from exc
        if (product not in data.PRODUCTS or horizon not in HORIZONS
                or target - origin != dt.timedelta(days=horizon)
                or source_row.get("origin") != origin.isoformat()
                or source_row.get("target") != target.isoformat()
                or type(eligible) is not bool
                or not isinstance(conditions, Mapping)
                or set(conditions) != {"B", "C", "D", "P"}):
            raise ValueError(f"invalid nested {phase} baseline key/schema: {product}/{origin}/{target}/{horizon}")
        ident = (product, origin.isoformat(), target.isoformat(), horizon)
        if ident in nested_keys:
            raise ValueError(f"duplicate nested {phase} baseline key {ident}")
        nested_keys.add(ident)
        for condition, state in conditions.items():
            key = (*ident, condition)
            expected_flat_keys.add(key)
            flat_row = flat.get(key)
            if flat_row is None:
                raise ValueError(f"hashed CSV companion is missing condition row {key}")
            if (flat_row.get("eligible", "").casefold() != str(eligible).casefold()
                    or (flat_row.get("p0Date") or None) != source_row.get("p0Date")):
                raise ValueError(f"nested/CSV eligibility or p0-date mismatch for {key}")
            nested_p0 = optional_float(source_row.get("p0"), key, "nested p0")
            csv_p0 = optional_float(flat_row.get("p0"), key, "CSV p0")
            if nested_p0 != csv_p0:
                raise ValueError(f"nested/CSV p0 mismatch for {key}")
            if not isinstance(state, Mapping):
                raise ValueError(f"malformed nested condition {condition} for {ident}")
            for field in ("fitId", "status", "fallbackReason", "cutoff"):
                csv_value = flat_row.get(field) or None
                nested_value = state.get(field) or None
                if csv_value != nested_value:
                    raise ValueError(f"nested/CSV {field} mismatch for {key}")
            for field in ("directLogPrediction", "directPrediction", "sarimaPrediction", "blendPrediction"):
                if optional_float(state.get(field), key, field) != optional_float(flat_row.get(field), key, field):
                    raise ValueError(f"nested/CSV {field} mismatch for {key}")
        # This experiment maps saved condition C to its B reference; saved condition B
        # is separate diagnostic evidence and may legitimately differ.
        c_state = conditions["C"]
        if not (eligible and start <= target <= end and origin >= origin_start):
            continue
        state = c_state
        status = state.get("status")
        if status not in ("issued", "fallback"):
            continue
        b = optional_float(state.get("blendPrediction"), (*ident, "C"), "C blendPrediction")
        if b is None or b <= 0:
            raise ValueError(f"invalid pinned B price at {ident}")
        normalized.append({"product": product, "origin": origin, "target": target,
                           "horizon": horizon, "B": b, "BSourceStatus": status,
                           "baselineReason": state.get("fallbackReason")})
    if expected_flat_keys != set(flat):
        missing = len(expected_flat_keys - set(flat))
        extra = len(set(flat) - expected_flat_keys)
        raise ValueError(f"nested/CSV baseline key coverage mismatch: missing={missing}, extra={extra}")
    expected_eligible = BASELINE_EXPECTED_ELIGIBLE[phase]
    if len(normalized) != expected_eligible:
        raise ValueError(f"{phase} pinned baseline eligible coverage changed: expected {expected_eligible}, got {len(normalized)}")
    return {**manifest, "path": manifest_path, "forecastCsvPath": forecast_path}, sorted(normalized,
            key=lambda r: (r["product"], r["origin"], r["horizon"]))


def baseline_preflight() -> dict[str, Any]:
    """Check the pinned answer-free B/C baseline schema and row coverage only."""
    phases = {}
    for phase in ("V", "T"):
        manifest, rows = _baseline_phase(phase)
        phases[phase] = {
            "manifestPath": str(manifest["path"].relative_to(ROOT)),
            "manifestSha256": baseline_source.BASELINE_SHA256[phase],
            "csvPath": str(manifest["forecastCsvPath"].relative_to(ROOT)),
            "csvSha256": manifest["forecastFileSha256"],
            "nestedForecastCount": len(manifest["forecasts"]),
            "eligibleKeyCount": len(rows),
            "normalizedBCoverageSha256": _sha(_canonical([
                {**row, "origin": row["origin"].isoformat(), "target": row["target"].isoformat()}
                for row in rows]))}
    return {"schemaVersion": 1, "experiment": EXPERIMENT_ID,
            "protocolSha256": PROTOCOL_SHA256, "phases": phases}


def _active_transition_sha256() -> str | None:
    if not TRANSITION_PATH.is_file():
        return None
    return str(_load_implementation_transition()["transitionSha256"])


def forecast_phase(phase: str) -> dict[str, Any]:
    """Train monthly U/M models and stage keyed forecasts without scored target values."""
    if phase not in ("V", "T"):
        raise ValueError("phase must be V or T")
    validate_resolved_config()
    _assert_price_snapshot()
    policy = _read_json(POLICY_PATH)
    _self_hash(policy, "modelPolicySha256", "model policy")
    source, source_doc = data.load_source_cache()
    global _SOURCE_CACHE_INTERNAL_SHA
    _SOURCE_CACHE_INTERNAL_SHA = str(source_doc["sourceCacheSha256"])
    representatives = data.representative_series(_read_json(PRICE_PATH))
    base_manifest, baseline_rows = _baseline_phase(phase)
    transition = _load_implementation_transition()
    if not baseline_rows:
        raise ValueError(f"no eligible pinned baseline rows for phase {phase}")
    by_month: dict[tuple[str, dt.date], list[dict[str, Any]]] = defaultdict(list)
    for row in baseline_rows:
        month_start = row["origin"].replace(day=1)
        by_month[(row["product"], month_start)].append(row)
    phase_months = (range(4, 7) if phase == "V" else range(7, 10))
    for product in data.PRODUCTS:
        for month in phase_months:
            by_month.setdefault((product, dt.date(2026, month, 1)), [])

    fit_grid, rows = [], []
    model_by_fit = {}
    replay_by_fit = {}
    runtime = _runtime_identity(include_device=True)
    for (product, fit_month), month_rows in sorted(by_month.items()):
        product_policy = policy["productPolicy"][product]
        context = int(product_policy["contextDays"])
        setting = str(product_policy["repairSetting"])
        eligible_origins = sorted({r["origin"] for r in month_rows})
        if not eligible_origins:
            raise ValueError(f"cannot establish first eligible pinned B origin for {product}/{fit_month:%Y-%m}")
        fit_boundary = eligible_origins[0]
        details = _selected_channels(source, product, fit_boundary)
        forecasts_by_type: dict[str, dict[dt.date, dict[str, Any]]] = {}
        model_paths: dict[str, Path] = {}
        for input_type in ("U", "M"):
            fit_id = f"{phase}-{fit_month:%Y%m}-{product}-{input_type}"
            fixed_epoch = int(product_policy["fixedEpochByInput"][input_type])
            train = _records_for_period(product, representatives[product], source.get(product, {}),
                                        context, input_type, details, min(representatives[product]),
                                        fit_boundary, label_before=fit_boundary)
            if not train:
                raise ValueError(f"final monthly fit has no eligible training rows: {fit_id}")
            q = _q_scale(representatives[product], fit_boundary)
            fit = fit_model(product, context, input_type, setting, train, train, q,
                            fit_id, OUT / "checkpoints" / f"{fit_id}.pt",
                            fixed_epochs=fixed_epoch, extension_allowed=False)
            saved = _save_fit_result(fit, details, fit_id)
            model_path = OUT / saved["weightPath"]
            model_paths[input_type] = model_path
            model_by_fit[fit_id] = saved
            reloaded_model, persisted_schema = _load_saved_fit_model(
                model_path, expected_fit_id=fit_id, expected_details=details)
            replay_details = persisted_schema["selectedInputChannels"]
            records = []
            seen_origins = sorted({r["origin"] for r in month_rows})
            for origin in seen_origins:
                # This inference record contains no targets and never reads a
                # price on or after origin. Only the feature window is supplied.
                window = data.build_input_window(product, origin, context, input_type,
                                                 representatives[product], source.get(product, {}), replay_details)
                if window is None:
                    continue
                target_days = [origin + dt.timedelta(days=h) for h in HORIZONS]
                records.append({"product": product, "origin": origin, "window": window,
                                "calendar": calendar_features(target_days),
                                "targets": [0.0] * 10, "targetMask": [False] * 10,
                                "targetDays": [d.isoformat() for d in target_days]})
            if not records:
                raise ValueError(f"monthly fit has no eligible origin-only inference rows: {fit_id}")
            _, replay_torch, _ = _torch_runtime()
            device = _device(replay_torch)
            replay_schema = {"contextDays": persisted_schema["contextDays"],
                             "inputType": persisted_schema["inputType"],
                             "repair": persisted_schema["repair"],
                             "selectedInputChannels": replay_details}
            replay_path, replay_payload = _persist_replay_payload(
                phase, fit_id, saved["weightSha256"], replay_schema, records)
            replay_records = _records_from_replay_payload(replay_payload)
            predict_batch_size = 32 if fit.get("smallBatchMode") else BATCH_SIZE
            original_values = _predict_records(fit["model"], replay_records, device, replay_torch,
                                               batch_size=predict_batch_size)
            values = _predict_records(reloaded_model, replay_records, device, replay_torch,
                                      batch_size=predict_batch_size)
            agreement = _compare_saved_model_replay(original_values, values)
            state_agreement = _state_dict_agreement(fit["model"], reloaded_model)
            if not agreement["allRowsAgree"] or not state_agreement["stateAgreement"]:
                raise ValueError(f"saved-model prediction replay differs for {fit_id}: {agreement}; {state_agreement}")
            original_prediction_sha = _replay_prediction_digest(replay_records, original_values)
            prediction_sha = _replay_prediction_digest(replay_records, values)
            schema_sha = replay_payload["schemaSha256"]
            replay_record = {"fitId": fit_id, "modelPath": saved["weightPath"],
                             "modelSha256": saved["weightSha256"],
                             "schemaSha256": schema_sha,
                             "rowPayloadPath": str(replay_path.relative_to(OUT)),
                             "rowPayloadPathBase": "experimentRoot",
                             "rowPayloadFileSha256": _file_sha(replay_path),
                             "rowPayloadSha256": replay_payload["rowPayloadSha256"],
                             "rowPayloadCount": replay_payload["rowPayloadCount"],
                             "originalPredictionSha256": original_prediction_sha,
                             "predictionSha256": prediction_sha,
                             **agreement, **state_agreement,
                             "reloadedStateDict": True,
                             "predictionSource": "reloaded_saved_state_dict_and_persisted_input_schema"}
            replay_record["replaySha256"] = _sha(_canonical(replay_record))
            replay_by_fit[fit_id] = replay_record
            forecasts_by_type[input_type] = {r["origin"]: {"values": pred,
                                                            "window": r["window"]}
                                             for r, pred in zip(replay_records, values)}
            fit_grid.append({"fitId": fit_id, "product": product,
                             "inputType": input_type, "fitMonth": fit_month.strftime("%Y-%m"),
                             "fitBoundary": fit_boundary.isoformat(),
                             "bestEpoch": fit["bestEpoch"], "fixedEpoch": fixed_epoch,
                             "contextDays": context, "repairSetting": setting,
                             "modelPath": saved["weightPath"],
                             "modelSha256": saved["weightSha256"],
                             "replayVerification": replay_record,
                             "checkpointPath": saved["checkpointPath"],
                             "checkpointSha256": saved["checkpointSha256"],
                             "selectedChannels": details,
                             "activeAuxFitRows": sum(r["window"]["activeAuxCount"] > 0 for r in train),
                             "trainRows": len(train), "qScale": q,
                             "runtime": runtime})

        selected_input = policy["selectedInputByProduct"][product]
        for baseline_row in month_rows:
            origin = baseline_row["origin"]
            outputs: dict[str, dict[str, Any]] = {}
            for input_type in ("U", "M"):
                predicted = forecasts_by_type[input_type].get(origin)
                b = baseline_row["B"]
                if predicted is None:
                    raw_value, raw_state, reason, active_aux = None, "not_issued", "input_window_unavailable", 0
                    raw_numeric_kind = "no_prediction"
                else:
                    raw_value = predicted["values"][baseline_row["horizon"] - 1]
                    active_aux = int(predicted["window"]["activeAuxCount"])
                    raw_numeric_kind = _prediction_class(raw_value)
                    raw_state = "issued" if math.isfinite(raw_value) and raw_value > 0 else "not_issued"
                    reason = None if raw_state == "issued" else (
                        "nonfinite_prediction" if not math.isfinite(raw_value) else "nonpositive_prediction")
                if raw_state == "issued":
                    outputs[input_type] = {"status": "issued", "prediction": raw_value,
                                           "rawPrediction": raw_value, "rawStatus": "issued",
                                           "rawNumericKind": raw_numeric_kind,
                                           "rawReason": None, "fallbackTo": None,
                                           "reason": None}
                elif b is not None:
                    outputs[input_type] = {"status": "fallback", "prediction": b,
                                           "rawPrediction": raw_value if raw_value is not None and math.isfinite(raw_value) else None,
                                           "rawNumericKind": raw_numeric_kind,
                                           "rawStatus": "not_issued", "rawReason": reason,
                                           "fallbackTo": "B", "reason": reason}
                else:
                    outputs[input_type] = {"status": "not_issued", "prediction": None,
                                           "rawPrediction": raw_value if raw_value is not None and math.isfinite(raw_value) else None,
                                           "rawNumericKind": raw_numeric_kind,
                                           "rawStatus": "not_issued", "rawReason": reason,
                                           "fallbackTo": None, "reason": reason}
            neural = dict(outputs[selected_input])
            outputs["N"] = neural
            b = baseline_row["B"]
            outputs["B"] = {"status": "issued" if baseline_row["BSourceStatus"] == "issued" else "fallback",
                             "prediction": b, "sourceStatus": baseline_row["BSourceStatus"],
                             "fallbackTo": None, "reason": baseline_row["baselineReason"]}
            for blend, candidate in (("H", "N"), ("H_U", "U"), ("H_M", "M")):
                n = outputs[candidate]["prediction"]
                if outputs[candidate]["status"] == "fallback" and b is not None:
                    outputs[blend] = {"status": "fallback", "prediction": b,
                                      "rawPrediction": outputs[candidate]["rawPrediction"],
                                      "rawNumericKind": outputs[candidate]["rawNumericKind"],
                                      "rawStatus": outputs[candidate]["rawStatus"],
                                      "rawReason": outputs[candidate].get("rawReason"),
                                      "fallbackTo": "B", "reason": outputs[candidate]["reason"]}
                elif b is not None and n is not None:
                    value = 0.5 * b + 0.5 * n
                    outputs[blend] = {"status": "issued", "prediction": value,
                                      "rawPrediction": value, "rawStatus": "issued",
                                      "rawNumericKind": "issued_positive",
                                      "rawReason": None,
                                      "fallbackTo": None, "reason": None}
                elif b is not None:
                    outputs[blend] = {"status": "fallback", "prediction": b,
                                      "rawPrediction": None, "rawStatus": "not_issued",
                                      "rawNumericKind": "no_prediction",
                                      "rawReason": "candidate_and_baseline_state",
                                      "fallbackTo": "B", "reason": "candidate_and_baseline_state"}
                else:
                    outputs[blend] = {"status": "not_issued", "prediction": None,
                                      "rawPrediction": None, "rawStatus": "not_issued",
                                      "rawNumericKind": "no_prediction",
                                      "rawReason": "baseline_unavailable",
                                      "fallbackTo": None, "reason": "baseline_unavailable"}
            chosen_info = forecasts_by_type.get(selected_input, {}).get(origin)
            active_aux = int(chosen_info["window"]["activeAuxCount"]) if chosen_info else 0
            fit_id = f"{phase}-{origin:%Y%m}-{product}-{selected_input}"
            output_evidence = {name: {"fitId": f"{phase}-{origin:%Y%m}-{product}-{it}",
                                      "modelSha256": model_by_fit[f"{phase}-{origin:%Y%m}-{product}-{it}"]["weightSha256"]}
                               for name, it in (("U", "U"), ("M", "M"), ("N", selected_input))}
            rows.append({"product": product, "origin": origin.isoformat(),
                         "target": baseline_row["target"].isoformat(),
                         "horizon": baseline_row["horizon"], "target_version": PRICE_SHA256,
                         "selectedInput": selected_input, "fitId": fit_id,
                         "fitMonth": fit_month.strftime("%Y-%m"), "neuralUsed": outputs["N"]["status"] == "issued",
                         "activeAuxCount": active_aux, "outputEvidence": output_evidence,
                         **outputs})
    rows.sort(key=lambda r: (r["product"], r["origin"], r["target"], r["horizon"]))
    for fit in fit_grid:
        fit["replayVerification"]["forecastRowsSha256"] = _forecast_rows_for_fit_digest(rows, fit)
        unsigned_replay = dict(fit["replayVerification"])
        unsigned_replay.pop("replaySha256", None)
        fit["replayVerification"]["replaySha256"] = _sha(_canonical(unsigned_replay))
    staging = {"schemaVersion": 1, "experiment": EXPERIMENT_ID, "stage": phase,
               "targetValuesAttached": False,
               "temporalBoundary": {
                   "monthlyFitBoundary": "first_eligible_pinned_B_origin_for_product_and_month",
                   "trainingLabels": "representative_targets_strictly_before_each_fitBoundary",
                   "inferenceFeatures": "source_values_through_origin_minus_one_calendar_day",
                   "targetLabelsAttachedToForecastRows": False},
               "targetSnapshot": {"path": str(PRICE_PATH.relative_to(ROOT)),
                                  "pathBase": "repository", "sha256": PRICE_SHA256,
                                  "targetVersion": PRICE_SHA256},
               "sourceCache": {"path": str(data.SOURCE_CACHE_PATH.relative_to(OUT)),
                               "pathBase": "experimentRoot",
                               "fileSha256": _file_sha(data.SOURCE_CACHE_PATH),
                               "sourceCacheSha256": source_doc["sourceCacheSha256"]},
               "modelPolicy": {"path": str(POLICY_PATH.relative_to(OUT)),
                               "pathBase": "experimentRoot", "fileSha256": _file_sha(POLICY_PATH),
                               "modelPolicySha256": policy["modelPolicySha256"],
                               "selectedInputByProduct": policy["selectedInputByProduct"]},
               "implementationTransition": {
                   "path": str(TRANSITION_PATH.relative_to(OUT)), "pathBase": "experimentRoot",
                   "fileSha256": _file_sha(TRANSITION_PATH),
                   "transitionSha256": transition["transitionSha256"]},
               "baselineSource": {"path": str(base_manifest["path"].relative_to(ROOT)),
                                  "pathBase": "repository",
                                  "sha256": baseline_source.BASELINE_SHA256[phase],
                                  "protocolSha256": base_manifest["protocolSha256"],
                                  "loaderContract": "nested-forecast-manifest-with-hashed-csv-companion-v1",
                                  "csvPath": str(base_manifest["forecastCsvPath"].relative_to(ROOT)),
                                  "csvSha256": base_manifest["forecastFileSha256"],
                                  "nestedForecastCount": len(base_manifest["forecasts"]),
                                  "eligibleKeyCount": len(baseline_rows),
                                  "normalizedBCoverageSha256": _sha(_canonical([
                                      {**row, "origin": row["origin"].isoformat(),
                                       "target": row["target"].isoformat()}
                                      for row in baseline_rows]))},
               "sourceCodeFiles": {str(Path(__file__).relative_to(ROOT)): _file_sha(Path(__file__)),
                                   str(Path(data.__file__).relative_to(ROOT)): _file_sha(Path(data.__file__)),
                                   str(_FROZEN_MAIN_PATH.relative_to(ROOT)): _file_sha(_FROZEN_MAIN_PATH)},
               "executionIdentity": {**runtime, "experiment": EXPERIMENT_ID,
                                     "phase": phase, "sourceCacheSha256": source_doc["sourceCacheSha256"]},
               "fitGrid": fit_grid,
               "replayVerification": {"schemaVersion": 1,
                                     "method": "reload_saved_state_dict_then_infer_before_baseline_row_join",
                                     "byFitId": {item["fitId"]: item["replayVerification"]
                                                 for item in fit_grid}},
               "rows": rows}
    staging_path = OUT / phase / "forecast-staging.json"
    _write_once(staging_path, _canonical(staging))
    return {"stage": phase, "rowCount": len(rows), "fitCount": len(fit_grid),
            "stagingPath": str(staging_path), "targetValuesAttached": False}


def _row_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    ordered = sorted((json.loads(_canonical(_safe_number(r))) for r in rows),
                     key=lambda r: (r["product"], r["origin"], r["target"], int(r["horizon"])))
    return _sha(_canonical(ordered))


def _key_set_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    keys = sorted([[r["product"], r["origin"], r["target"], int(r["horizon"]), r["target_version"]]
                   for r in rows])
    return _sha(_canonical(keys))


def export_parquet(phase: str) -> dict[str, Any]:
    """Host-runtime writer: seal a forecast staging file into the score schema."""
    if phase not in ("V", "T"):
        raise ValueError("phase must be V or T")
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("Parquet export needs the existing host Python with pandas and pyarrow") from exc
    staging_path = OUT / phase / "forecast-staging.json"
    staging = _read_json(staging_path)
    if staging.get("targetValuesAttached") is not False or staging.get("stage") != phase:
        raise ValueError("forecast staging is not an answer-free phase artifact")
    transition = _load_implementation_transition(verify_native_runtime=False)
    transition_ref = staging.get("implementationTransition", {})
    if (transition_ref.get("path") != str(TRANSITION_PATH.relative_to(OUT))
            or transition_ref.get("pathBase") != "experimentRoot"
            or transition_ref.get("fileSha256") != _file_sha(TRANSITION_PATH)
            or transition_ref.get("transitionSha256") != transition["transitionSha256"]):
        raise ValueError("forecast staging is not bound to the preserved implementation transition")
    policy = _read_json(POLICY_PATH)
    _self_hash(policy, "modelPolicySha256", "model policy")
    if staging["modelPolicy"].get("fileSha256") != _file_sha(POLICY_PATH):
        raise ValueError("policy changed after answer-free forecast staging")
    if staging["sourceCache"].get("fileSha256") != _file_sha(data.SOURCE_CACHE_PATH):
        raise ValueError("source cache changed after forecast staging")
    rows = staging.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("forecast staging has no rows")
    fit_grid = staging.get("fitGrid")
    if not isinstance(fit_grid, list) or len(fit_grid) != 24:
        raise ValueError("each phase requires exactly 24 product/input/month fits")
    fit_ids = [str(x["fitId"]) for x in fit_grid]
    if len(set(fit_ids)) != 24:
        raise ValueError("phase fit grid has duplicate fit identifiers")
    months = ("2026-04", "2026-05", "2026-06") if phase == "V" else ("2026-07", "2026-08", "2026-09")
    expected_grid = {(p, it, month) for p in data.PRODUCTS for it in ("U", "M") for month in months}
    actual_grid = {(x.get("product"), x.get("inputType"), x.get("fitMonth")) for x in fit_grid}
    if actual_grid != expected_grid:
        raise ValueError("phase fit grid does not cover each product/input/month exactly once")
    replay_doc = staging.get("replayVerification")
    if (not isinstance(replay_doc, Mapping)
            or replay_doc.get("method") != "reload_saved_state_dict_then_infer_before_baseline_row_join"
            or set(replay_doc.get("byFitId", {})) != set(fit_ids)):
        raise ValueError("phase is missing the complete saved-model replay verification grid")
    for item in fit_grid:
        evidence = replay_doc["byFitId"][item["fitId"]]
        expected_payload_path = f"{phase}/replay-inputs/{item['fitId']}.json"
        row_payload_path = OUT / expected_payload_path
        expected_schema = {"contextDays": item.get("contextDays"),
                           "inputType": item.get("inputType"),
                           "repair": item.get("repairSetting"),
                           "selectedInputChannels": item.get("selectedChannels")}
        replay_payload = _read_json(row_payload_path) if row_payload_path.is_file() else {}
        unsigned = dict(evidence)
        replay_sha = unsigned.pop("replaySha256", None)
        if (evidence.get("fitId") != item["fitId"]
                or evidence.get("modelPath") != item.get("modelPath")
                or evidence.get("modelSha256") != item.get("modelSha256")
                or evidence.get("rowPayloadPath") != expected_payload_path
                or evidence.get("rowPayloadPathBase") != "experimentRoot"
                or evidence.get("rowPayloadFileSha256") != (_file_sha(row_payload_path) if row_payload_path.is_file() else None)
                or replay_payload.get("fitId") != item["fitId"]
                or replay_payload.get("modelSha256") != item.get("modelSha256")
                or replay_payload.get("schema") != expected_schema
                or replay_payload.get("schemaSha256") != _sha(_canonical(expected_schema))
                or replay_payload.get("rowPayloadSha256") != _inference_payload_digest(replay_payload.get("records", []))
                or replay_payload.get("rowPayloadSha256") != evidence.get("rowPayloadSha256")
                or replay_payload.get("rowPayloadCount") != evidence.get("rowPayloadCount")
                or evidence.get("reloadedStateDict") is not True
                or evidence.get("stateAgreement") is not True
                or evidence.get("allRowsAgree") is not True
                or evidence.get("issueClassAgreement") is not True
                or evidence.get("inferenceRowCount") != evidence.get("rowPayloadCount")
                or evidence.get("horizonCount") != len(HORIZONS)
                or evidence.get("absoluteTolerance") != RELOAD_REPLAY_ATOL
                or evidence.get("relativeTolerance") != RELOAD_REPLAY_RTOL
                or evidence.get("maxAbsoluteDifference", math.inf) > RELOAD_REPLAY_ATOL + RELOAD_REPLAY_RTOL * evidence.get("maxOriginalMagnitude", -math.inf)
                or evidence.get("predictionSource") != "reloaded_saved_state_dict_and_persisted_input_schema"
                or int(evidence.get("rowPayloadCount", 0)) <= 0
                or replay_sha != _sha(_canonical(unsigned))
                or evidence.get("forecastRowsSha256") != _forecast_rows_for_fit_digest(rows, item)):
            raise ValueError(f"invalid saved-model replay chain for {item['fitId']}")
    for source_path, expected_sha in staging.get("sourceCodeFiles", {}).items():
        if _file_sha(ROOT / source_path) != expected_sha:
            raise ValueError(f"source code changed after forecast staging: {source_path}")
    models = {}
    for item in fit_grid:
        model_path = OUT / "models" / f"{item['fitId']}.pt"
        if not model_path.is_file() or _file_sha(model_path) != item["modelSha256"]:
            raise ValueError(f"phase model file is missing or changed: {item['fitId']}")
        checkpoint_path = OUT / item["checkpointPath"]
        if not checkpoint_path.is_file() or _file_sha(checkpoint_path) != item["checkpointSha256"]:
            raise ValueError(f"phase optimizer checkpoint is missing or changed: {item['fitId']}")
        models[item["fitId"]] = {"path": str(model_path.relative_to(OUT)),
                                 "pathBase": "experimentRoot",
                                 "sha256": item["modelSha256"],
                                 "checkpointPath": item["checkpointPath"],
                                 "checkpointSha256": item["checkpointSha256"]}
    parquet_path = OUT / phase / "forecasts.parquet"
    temp_path = parquet_path.with_suffix(".parquet.tmp")
    pd.DataFrame.from_records(rows).to_parquet(temp_path, index=False, engine="pyarrow")
    if parquet_path.exists():
        if _file_sha(temp_path) != _file_sha(parquet_path):
            temp_path.unlink()
            raise FileExistsError(f"refusing to replace immutable forecast artifact {parquet_path}")
        temp_path.unlink()
    else:
        temp_path.replace(parquet_path)
    # Hash the exact decoded table that the scorer will receive, including nested structs.
    roundtrip = pd.read_parquet(parquet_path, engine="pyarrow").to_dict(orient="records")
    roundtrip_rows = [_safe_number(row) for row in roundtrip]
    if _row_digest(rows) != _row_digest(roundtrip_rows):
        raise ValueError("Parquet round-trip changed forecast rows")
    if _key_set_digest(rows) != _key_set_digest(roundtrip_rows):
        raise ValueError("Parquet round-trip changed forecast keys")
    for item in fit_grid:
        if replay_doc["byFitId"][item["fitId"]]["forecastRowsSha256"] != _forecast_rows_for_fit_digest(roundtrip_rows, item):
            raise ValueError(f"Parquet round-trip broke replay-to-row binding for {item['fitId']}")
    manifest = {k: v for k, v in staging.items() if k != "rows"}
    manifest.update({"stage": phase, "createdAtKst": dt.datetime.now().astimezone().isoformat(),
                     "forecastArtifact": {"path": f"{phase}/forecasts.parquet",
                                          "pathBase": "experimentRoot",
                                          "sha256": _file_sha(parquet_path),
                                          "rowsSha256": _row_digest(roundtrip_rows),
                                          "keySetSha256": _key_set_digest(roundtrip_rows),
                                          "rowCount": len(roundtrip_rows)},
                     "coverage": {"rowCount": len(roundtrip_rows),
                                  "uniqueKeyCount": len(roundtrip_rows),
                                  "expectedKeyCount": len(rows)},
                     "modelsByFitId": models,
                     "replayVerification": replay_doc,
                     "fitGrid": fit_grid,
                     "executionIdentity": {**staging["executionIdentity"],
                                           "parquetWriter": {"python": platform.python_version(),
                                                             "pandas": pd.__version__,
                                                             "pyarrow": importlib.metadata.version("pyarrow")}}})
    manifest["executionIdentitySha256"] = _sha(_canonical(manifest["executionIdentity"]))
    manifest["manifestSha256"] = _sha(_canonical(manifest))
    manifest_path = OUT / phase / "forecast-manifest.json"
    _write_once(manifest_path, _canonical(manifest))
    staging_path.unlink()
    return {"manifestPath": str(manifest_path), "manifestSha256": manifest["manifestSha256"],
            "forecastPath": str(parquet_path), "forecastSha256": manifest["forecastArtifact"]["sha256"],
            "rowCount": len(roundtrip_rows), "fitCount": len(fit_grid)}


def export_training_history() -> dict[str, Any]:
    """Export the append-only per-epoch ledger through the existing host Arrow runtime."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("training-history export needs the existing host Python with pandas and pyarrow") from exc
    ledger = _read_json(LEDGER_PATH)
    unsigned = dict(ledger)
    ledger_sha = unsigned.pop("ledgerSha256", None)
    if ledger_sha != _sha(_canonical(unsigned)):
        raise ValueError("training ledger self-hash mismatch")
    rows = []
    for fit_id, fit in sorted(ledger.get("fits", {}).items()):
        for epoch in fit.get("history", []):
            rows.append({"fitId": fit_id, "product": fit.get("product"),
                         "contextDays": fit.get("contextDays"), "inputType": fit.get("inputType"),
                         "repair": fit.get("repair"), "fitTerminalReason": fit.get("terminalReason"),
                         "fitWallClockSeconds": fit.get("wallClockSeconds"),
                         "fitOptimizerUpdates": fit.get("optimizerUpdates"),
                         "oomFallbackUsed": fit.get("oomFallbackUsed"),
                         "smallBatchMode": fit.get("smallBatchMode"), **epoch})
    if not rows:
        raise ValueError("training ledger contains no completed epoch rows")
    path = OUT / "training-history.parquet"
    temp = path.with_suffix(".parquet.tmp")
    pd.DataFrame.from_records(rows).to_parquet(temp, index=False, engine="pyarrow")
    if path.exists():
        if _file_sha(path) != _file_sha(temp):
            temp.unlink()
            raise FileExistsError(f"refusing to replace immutable training history {path}")
        temp.unlink()
    else:
        temp.replace(path)
    roundtrip = pd.read_parquet(path, engine="pyarrow").to_dict(orient="records")
    expected = sorted(rows, key=lambda r: (r["fitId"], r["epoch"]))
    actual = sorted((_safe_number(row) for row in roundtrip), key=lambda r: (r["fitId"], r["epoch"]))
    if _canonical(expected) != _canonical(actual):
        raise ValueError("training-history Parquet round-trip changed ledger values")
    return {"path": str(path.relative_to(OUT)), "sha256": _file_sha(path),
            "rows": len(rows), "fitCount": len(ledger.get("fits", {})),
            "ledgerSha256": ledger_sha}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DIRECT_PRICE_PATH_ATTENTION_V1 implementation")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze-config")
    sub.add_parser("validate-config")
    sub.add_parser("record-transition")
    sub.add_parser("baseline-preflight")
    sub.add_parser("develop")
    for phase in ("V", "T"):
        p = sub.add_parser(f"forecast-{phase.lower()}")
        p.set_defaults(phase=phase)
        e = sub.add_parser(f"export-{phase.lower()}")
        e.set_defaults(phase=phase)
    sub.add_parser("export-training-history")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "freeze-config":
        doc = freeze_resolved_config()
        print(json.dumps({"resolvedConfigSha256": doc["resolvedConfigSha256"],
                          "path": str(CONFIG_PATH)}, sort_keys=True))
    elif args.command == "validate-config":
        doc = validate_resolved_config()
        print(json.dumps({"resolvedConfigSha256": doc["resolvedConfigSha256"], "valid": True}, sort_keys=True))
    elif args.command == "record-transition":
        doc = record_implementation_transition()
        print(json.dumps({"transitionSha256": doc["transitionSha256"],
                          "path": str(TRANSITION_PATH)}, sort_keys=True))
    elif args.command == "baseline-preflight":
        print(json.dumps(baseline_preflight(), sort_keys=True))
    elif args.command == "develop":
        result = run_development()
        print(json.dumps({"developmentScoresSha256": result["developmentScoresSha256"],
                          "fitCount": result["totalCompletedFits"],
                          "modelPolicySha256": result["modelPolicySha256"]}, sort_keys=True))
    elif args.command.startswith("forecast-"):
        print(json.dumps(forecast_phase(args.phase), sort_keys=True))
    elif args.command.startswith("export-"):
        if args.command == "export-training-history":
            print(json.dumps(export_training_history(), sort_keys=True))
        else:
            print(json.dumps(export_parquet(args.phase), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
