"""No-fit recovery of the 55 missing December 2025 onion B cache keys.

This helper only loads reviewed December fit artifacts. It never calls a fit
routine. Recovery is explicit via ``--recover``; ``--audit`` is read-only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

from . import equal_info_online_forecasts as equal_info
from . import final_ensemble_onion_baseline as baseline

ROOT = Path(__file__).resolve().parents[1]
OUT = baseline.OUT
FROZEN_BASELINE_SHA256 = "d42787b08a1d0620e0dcdb186c22c2068bae5d9ac3fa2ad3f574339c2b7b15b3"
RECOVERY_ID = "december-2025-no-fit-recovery-v1"
AUDIT_NAME = "baseline-recovery-audit.json"
MISSING_ORIGIN_START = dt.date(2025, 12, 22)
MISSING_ORIGIN_END = dt.date(2025, 12, 31)
MAX_ULPS = 8


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return baseline.canonical(value)


def _ulp_close(left: Any, right: Any, max_ulps: int = MAX_ULPS) -> bool:
    """Compare finite binary64 values within a small representation error."""
    if left is None or right is None:
        return left is right
    a, b = float(left), float(right)
    if not (math.isfinite(a) and math.isfinite(b)):
        return a == b
    return abs(a - b) <= max_ulps * max(math.ulp(a), math.ulp(b))


def audit_overlap(source_rows: dict[tuple[str, str, str, int], dict[str, Any]],
                  replay_rows: list[dict[str, Any]], *,
                  exact_expected_keys: set[tuple[str, str, str, int]] | None = None) -> dict[str, Any]:
    """Audit replay pairs without changing source rows or accepting time drift.

    Status, direct/SARIMA provenance, and temporal fields are exact. Only the
    floating point direct and blended values receive an eight-ULP allowance.
    """
    replay = {}
    for row in replay_rows:
        key = baseline._key(row["product"], baseline._date(row["origin"]),
                            baseline._date(row["target"]), int(row["horizon"]))
        if key in replay:
            raise ValueError(f"duplicate replay overlap {key}")
        replay[key] = row
    common = set(source_rows) & set(replay)
    if exact_expected_keys is not None and (common != exact_expected_keys or set(replay) != exact_expected_keys):
        raise ValueError(f"overlap key set differs: expected={len(exact_expected_keys)} "
                         f"common={len(common)} replay={len(replay)}")
    fields_exact = ("sourceStatus", "BSourceStatus", "sourceReason", "fitBoundary", "cutoff")
    fields_float = ("B", "direct.prediction")
    nested_exact = ("direct.status", "direct.reason", "sarima.status", "sarima.reason",
                    "sarima.fit.trainStart", "sarima.fit.trainEnd", "sarima.fit.fitOrigin",
                    "sarima.fit.order", "sarima.fit.seasonalOrder",
                    "sarima.fit.trainingCalendarDays", "sarima.fit.selectedD",
                    "provenance.sourceDate", "provenance.sourceAgeDays",
                    "provenance.stateThrough")

    def value_at(row: dict[str, Any], path: str) -> Any:
        value: Any = row
        for part in path.split("."):
            value = value.get(part) if isinstance(value, dict) else None
        return value
    mismatches = []
    max_abs = {name: 0.0 for name in fields_float}
    max_ulps_seen = {name: 0 for name in fields_float}
    numeric_changed = {name: 0 for name in fields_float + ("sarima.prediction",)}
    for key in sorted(common):
        old = _normalize_source_row(source_rows[key])
        new = replay[key]
        for name in fields_exact + nested_exact:
            a, b = value_at(old, name), value_at(new, name)
            if a != b:
                mismatches.append({"key": key, "field": name,
                                   "source": a, "replay": b})
        for name in fields_float:
            a, b = value_at(old, name), value_at(new, name)
            if a != b:
                numeric_changed[name] += 1
                if a is not None and b is not None and math.isfinite(float(a)) and math.isfinite(float(b)):
                    max_abs[name] = max(max_abs[name], abs(float(a) - float(b)))
                    ulp = max(math.ulp(float(a)), math.ulp(float(b)))
                    max_ulps_seen[name] = max(max_ulps_seen[name],
                                              math.ceil(abs(float(a) - float(b)) / ulp))
            if not _ulp_close(a, b):
                mismatches.append({"key": key, "field": name,
                                   "source": a, "replay": b})
        a, b = value_at(old, "sarima.prediction"), value_at(new, "sarima.prediction")
        if a != b:
            numeric_changed["sarima.prediction"] += 1
            mismatches.append({"key": key, "field": "sarima.prediction", "source": a, "replay": b})
    if mismatches:
        raise ValueError(f"replay overlap mismatch count {len(mismatches)}; first={mismatches[0]}")
    return {"overlapRows": len(common), "numericToleranceUlps": MAX_ULPS,
            "numericFieldsWithUlpTolerance": list(fields_float),
            "sarimaPredictionExact": True, "maxAbsoluteDifference": max_abs,
            "maxUlpsDifference": max_ulps_seen, "changedValueCounts": numeric_changed,
            "exactFields": list(fields_exact + nested_exact),
            "allStatusesAndTimeContractsExact": True,
            "sourceRowsUnchanged": True,
            "keySha256": _sha(_canonical([list(k) for k in sorted(common)]))}


def _normalize_source_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map equal-info saved component fields to the recovery replay schema."""
    if isinstance(row.get("direct"), dict):
        return row
    sarima = row.get("sourceSarima") or {}
    training = sarima.get("training") or {}
    blend_provenance = row.get("sourceBlendProvenance") or {}
    direct_provenance = (row.get("sourceDirectProvenance") or {})
    sarima_provenance = sarima.get("provenance") or {}
    return {
        **row,
        "direct": {"status": row.get("sourceDirectStatus"),
                   "prediction": row.get("sourceDirectPrediction"),
                   "reason": row.get("sourceDirectReason")},
        "sarima": {"status": sarima.get("status"), "prediction": sarima.get("prediction"),
                   "reason": sarima.get("reason"),
                   "fit": {"trainStart": training.get("trainStart"),
                           "trainEnd": training.get("trainEnd"),
                           "fitOrigin": training.get("fitOrigin"),
                           "order": training.get("order"),
                           "seasonalOrder": training.get("seasonalOrder"),
                           "trainingCalendarDays": training.get("trainingCalendarDays"),
                           "selectedD": training.get("selectedD")}},
        "provenance": {"sourceDate": sarima_provenance.get("lastActualDate"),
                       "sourceAgeDays": direct_provenance.get("sourceAgeDays"),
                       "stateThrough": sarima_provenance.get("lastUpdatedDate")},
    }


def _validate_pair(out_dir: Path) -> tuple[dict[str, Any], dict[str, Any], bytes, bytes]:
    manifest_path, cache_path = out_dir / baseline.MANIFEST_NAME, out_dir / baseline.CACHE_NAME
    manifest_raw, cache_raw = manifest_path.read_bytes(), cache_path.read_bytes()
    manifest, cache = json.loads(manifest_raw), json.loads(cache_raw)
    unsigned = dict(manifest)
    recorded = unsigned.pop("manifestSha256", None)
    if recorded != _sha(_canonical(unsigned)):
        raise ValueError("baseline manifest self-digest mismatch")
    if manifest.get("complete") is not False or manifest.get("targetValuesAttached") is not False:
        raise ValueError("recovery requires the incomplete answer-free cache")
    if manifest.get("cacheFileSha256") != _sha(cache_raw):
        raise ValueError("baseline cache byte digest mismatch")
    if manifest.get("cacheRowsSha256") != _sha(_canonical(cache.get("rows", []))):
        raise ValueError("baseline cache row digest mismatch")
    if manifest.get("sourceCodeSha256", {}).get("baselineCacheModule") != FROZEN_BASELINE_SHA256:
        raise ValueError("frozen baseline producer hash changed")
    if _sha(Path(baseline.__file__).read_bytes()) != FROZEN_BASELINE_SHA256:
        raise ValueError("baseline producer file no longer matches reviewed bytes")
    if cache.get("phase") != "answer_free" or cache.get("targetValuesAttached") is not False:
        raise ValueError("baseline cache is not answer-free")
    return manifest, cache, manifest_raw, cache_raw


def _expected_missing(rows: list[dict[str, Any]]) -> set[tuple[str, str, str, int]]:
    expected = baseline.expected_origin_keys()
    present = {baseline._key(row["product"], baseline._date(row["origin"]),
                              baseline._date(row["target"]), int(row["horizon"]))
               for row in rows if row.get("product") == "onion"}
    missing = expected - present
    wanted = {baseline._key("onion", origin, origin + dt.timedelta(days=h), h)
              for offset in range((MISSING_ORIGIN_END - MISSING_ORIGIN_START).days + 1)
              for origin in (MISSING_ORIGIN_START + dt.timedelta(days=offset),)
              for h in equal_info.HORIZONS
              if origin + dt.timedelta(days=h) >= dt.date(2026, 1, 1)}
    if missing != wanted or len(missing) != 55:
        raise ValueError(f"missing key set is not the reviewed 55-key December tail: {len(missing)}")
    return missing


def _fit_by_id(manifest: dict[str, Any], out_dir: Path, fit_id: str) -> dict[str, Any]:
    matches = [fit for fit in manifest.get("fitInventory", []) if fit.get("fitId") == fit_id]
    if len(matches) > 1:
        raise ValueError(f"duplicate fit records for {fit_id}")
    if matches:
        fit = matches[0]
        if fit.get("fitStatus") != "issued" or fit.get("reloadValidated") is not True:
            raise ValueError(f"fit artifact is not issued and reload-validated: {fit_id}")
        path_value, digest = fit.get("modelPath"), fit.get("modelSha256")
    else:
        # The failed December progress commit omitted December fit records, but
        # the parent-approved orphaned artifacts remain under their fixed names.
        path_value = (f"models/{fit_id}-initial.joblib" if fit_id.endswith("sarima")
                      else f"models/{fit_id}.joblib")
        digest = _sha((out_dir / path_value).read_bytes()) if (out_dir / path_value).is_file() else None
    import joblib
    if not path_value or not digest:
        raise ValueError(f"missing fit artifact evidence for {fit_id}")
    path = out_dir / path_value
    if not path.is_file() or _sha(path.read_bytes()) != digest:
        raise ValueError(f"fit artifact hash mismatch: {fit_id}")
    model = joblib.load(path)
    if fit_id.endswith("sarima") and not all(hasattr(model, name) for name in ("get_forecast", "append")):
        raise ValueError(f"loaded saved SARIMA artifact is malformed: {fit_id}")
    if not fit_id.endswith("sarima") and not hasattr(model, "predict"):
        raise ValueError(f"loaded saved Ridge artifact is malformed: {fit_id}")
    return model


def _december_fit_evidence(manifest: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    """Return hash-verified evidence for the 10 Ridge, SARIMA, and saved state files."""
    result = []
    fits = {row.get("fitId"): row for row in manifest.get("fitInventory", [])}
    ids = [baseline._fit_id("2025-12", h) for h in equal_info.HORIZONS]
    ids.append(baseline._fit_id("2025-12"))
    for fit_id in ids:
        fit = fits.get(fit_id)
        artifacts = []
        model_rel = (f"models/{fit_id}-initial.joblib" if fit_id.endswith("sarima")
                     else f"models/{fit_id}.joblib")
        model_path = out_dir / model_rel
        if not model_path.is_file():
            raise ValueError(f"missing December fit model artifact: {fit_id}")
        artifacts.append({"path": model_rel, "sha256": _sha(model_path.read_bytes())})
        for checkpoint in (fit or {}).get("stateCheckpoints", []):
            if not checkpoint.get("path") or not checkpoint.get("sha256"):
                continue
            path = out_dir / checkpoint["path"]
            if not path.is_file() or _sha(path.read_bytes()) != checkpoint["sha256"]:
                raise ValueError(f"December checkpoint hash mismatch for {fit_id}")
            artifacts.append({"path": checkpoint["path"], "sha256": checkpoint["sha256"]})
        if fit_id.endswith("sarima"):
            checkpoint_path = out_dir / "checkpoints/onion-2025-12-sarima-through-2025-12-31.joblib"
            if not checkpoint_path.is_file():
                raise ValueError("missing saved December SARIMA end-state checkpoint")
            record = {"path": "checkpoints/onion-2025-12-sarima-through-2025-12-31.joblib",
                      "sha256": _sha(checkpoint_path.read_bytes())}
            if record not in artifacts:
                artifacts.append(record)
        result.append({"fitId": fit_id, "artifacts": artifacts})
    if len(result) != 11 or not any(len(row["artifacts"]) > 1 for row in result):
        raise ValueError("December fit evidence must include 11 fits and SARIMA state checkpoint")
    return result


def _december_fit_records(evidence: list[dict[str, Any]],
                          contracts: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstitute only auditable December fields for formerly orphaned fits."""
    artifact_by_fit = {row["fitId"]: row["artifacts"] for row in evidence}
    records = []
    ridge_contracts = {int(row["fitId"].rsplit("h", 1)[-1]): row for row in contracts["ridge"]}
    for horizon in equal_info.HORIZONS:
        fit_id = baseline._fit_id("2025-12", horizon)
        model = artifact_by_fit[fit_id][0]
        fit_contract = ridge_contracts[horizon]
        records.append({"fitId": fit_id, "product": "onion", "model": "causalDirectRidge",
                        "horizon": horizon, "fitMonth": "2025-12",
                        "fitBoundary": "2025-12-01", "fitOrigin": "2025-12-01",
                        "alpha": 10.0, "featureNames": list(equal_info.modeling.FEATURE_NAMES),
                        "trainingCount": fit_contract["trainingCount"],
                        "trainingFirstTarget": fit_contract["firstTrainingTarget"],
                        "trainingLastTarget": fit_contract["lastTrainingTarget"],
                        "trainingTargetsBeforeFitOrigin": True,
                        "fitStatus": "issued", "fitReason": None,
                        "modelPath": model["path"], "modelSha256": model["sha256"],
                        "fitElapsedMs": None, "fitElapsedUnavailableReason": "not recorded",
                        "reloadValidated": True, "recoveryArtifactRecord": True})
    fit_id = baseline._fit_id("2025-12")
    artifacts = artifact_by_fit[fit_id]
    model = next(row for row in artifacts if row["path"].endswith("-initial.joblib"))
    checkpoint = next(row for row in artifacts if row["path"].endswith("through-2025-12-31.joblib"))
    records.append({"fitId": fit_id, "product": "onion", "model": "SARIMA",
                    "fitMonth": "2025-12", "fitBoundary": "2025-12-01",
                    "fitOrigin": "2025-12-01", "order": [1, 1, 1],
                    "seasonalOrder": [1, 0, 0, 7], "windowCalendarDays": 730,
                    "fitStart": "2023-12-02", "fitEnd": "2025-11-30",
                    "fitStatus": "issued", "fitReason": None,
                    "modelPath": model["path"], "modelSha256": model["sha256"],
                    "fitElapsedMs": None, "fitElapsedUnavailableReason": "not recorded",
                    "reloadValidated": True,
                    "stateCheckpoints": [{"origin": "2025-12-31", "stateThrough": "2025-12-30",
                                          "path": checkpoint["path"], "sha256": checkpoint["sha256"]}],
                    "recoveryArtifactRecord": True})
    return records


def _saved_training_audits(source_doc: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Get the frozen December confirmation-phase direct-fit audits by horizon."""
    saved = {}
    for horizon in equal_info.HORIZONS:
        audits = []
        for source in source_doc.get("confirmationForecasts", []):
            if source.get("product") != "onion" or source.get("eligible") is not True:
                continue
            if not dt.date(2025, 12, 1) <= baseline._date(source["origin"]) <= dt.date(2025, 12, 31):
                continue
            if int(source["horizon"]) != horizon:
                continue
            training = (source.get("models", {}).get("causalDirect") or {}).get("training") or {}
            audits.append({name: training.get(name) for name in
                           ("count", "firstTarget", "lastTarget", "fitOrigin", "alpha", "featureCount")})
        unique = {json.dumps(row, sort_keys=True) for row in audits}
        if not audits or len(unique) != 1:
            raise ValueError(f"saved December confirmation fit audit is missing/inconsistent for h{horizon}")
        saved[horizon] = audits[0]
    return saved


def _validate_december_model_contracts(manifest: dict[str, Any], out_dir: Path,
                                      source_doc: dict[str, Any]) -> dict[str, Any]:
    """Check saved model structure and fixed fit-time boundaries without fitting."""
    from . import modeling
    from . import direct_horizon

    prices = json.loads(equal_info.PRICES_PATH.read_text(encoding="utf-8"))
    direct_series = direct_horizon._observations(prices)["onion"]
    training_audits = {}
    expected_counts = (3617, 3617, 3616, 3615, 3614, 3613, 3612, 3611, 3611, 3610)
    saved_training_audits = _saved_training_audits(source_doc)
    for horizon in equal_info.HORIZONS:
        feature_cache: dict[dt.date, tuple[Any, ...]] = {}
        targets = []
        for target in direct_series:
            if target >= dt.date(2025, 12, 1) or target < equal_info.narrow.TRAIN_START:
                continue
            found = modeling._feature_row(direct_series, target - dt.timedelta(days=horizon + 1),
                                          target, horizon, feature_cache)
            if found is None:
                continue
            source, _vector = found
            origin = target - dt.timedelta(days=horizon)
            if (origin - source).days > 14:
                continue
            targets.append(target)
        expected_count = expected_counts[horizon - 1]
        if len(targets) != expected_count or max(targets).isoformat() != "2025-11-29":
            raise ValueError(f"December Ridge training-row audit failed for horizon {horizon}")
        reconstructed = {"count": len(targets), "firstTarget": min(targets).isoformat(),
                         "lastTarget": max(targets).isoformat(), "fitOrigin": "2025-12-01",
                         "alpha": 10.0, "featureCount": len(modeling.FEATURE_NAMES)}
        if reconstructed != saved_training_audits[horizon]:
            raise ValueError(f"reconstructed/saved Ridge training audit differs for horizon {horizon}")
        training_audits[horizon] = reconstructed

    ridge_contracts = []
    for horizon in equal_info.HORIZONS:
        fit_id = baseline._fit_id("2025-12", horizon)
        model = _fit_by_id(manifest, out_dir, fit_id)
        steps = getattr(model, "named_steps", {})
        ridge = next((value for name, value in steps.items() if name.lower() == "ridge"), None)
        scaler = next((value for name, value in steps.items() if "scaler" in name.lower()), None)
        feature_count = len(modeling.FEATURE_NAMES)
        scaler_samples = getattr(scaler, "n_samples_seen_", None)
        if hasattr(scaler_samples, "item"):
            scaler_samples = scaler_samples.item()
        if (ridge is None or scaler is None or float(ridge.alpha) != 10.0
                or getattr(scaler, "n_features_in_", None) != feature_count
                or getattr(ridge, "n_features_in_", None) != feature_count
                or int(scaler_samples) != training_audits[horizon]["count"]):
            raise ValueError(f"saved Ridge structure/alpha/features failed validation: {fit_id}")
        ridge_contracts.append({"fitId": fit_id, "alpha": float(ridge.alpha),
                                "featureCount": feature_count,
                                "trainingCount": training_audits[horizon]["count"],
                                "firstTrainingTarget": training_audits[horizon]["firstTarget"],
                                "lastTrainingTarget": training_audits[horizon]["lastTarget"],
                                "fitOrigin": "2025-12-01"})
    sarima_id = baseline._fit_id("2025-12")
    sarima = _fit_by_id(manifest, out_dir, sarima_id)
    smodel = getattr(sarima, "model", None)
    labels = getattr(getattr(smodel, "data", None), "row_labels", None)
    first_label = str(labels[0])[:10] if labels is not None and len(labels) else None
    last_label = str(labels[-1])[:10] if labels is not None and len(labels) else None
    nobs = int(getattr(smodel, "nobs", -1)) if smodel is not None else -1
    if (getattr(smodel, "order", None) != (1, 1, 1)
            or getattr(smodel, "seasonal_order", None) != (1, 0, 0, 7)
            or first_label != "2023-12-02" or last_label != "2025-11-30" or nobs != 730):
        raise ValueError("saved SARIMA structure/window failed validation")
    checkpoint_path = out_dir / "checkpoints/onion-2025-12-sarima-through-2025-12-31.joblib"
    import joblib
    checkpoint = joblib.load(checkpoint_path)
    checkpoint_model = getattr(checkpoint, "model", None)
    checkpoint_labels = getattr(getattr(checkpoint_model, "data", None), "row_labels", None)
    checkpoint_last = str(checkpoint_labels[-1])[:10] if checkpoint_labels is not None and len(checkpoint_labels) else None
    checkpoint_nobs = int(getattr(checkpoint_model, "nobs", -1)) if checkpoint_model is not None else -1
    if (checkpoint_last != "2025-12-30" or checkpoint_nobs != 760
            or getattr(checkpoint_model, "order", None) != (1, 1, 1)
            or getattr(checkpoint_model, "seasonal_order", None) != (1, 0, 0, 7)):
        raise ValueError("saved SARIMA Dec31 checkpoint is not actually advanced through Dec30")
    import numpy as np
    if not np.array_equal(np.asarray(sarima.params), np.asarray(checkpoint.params)):
        raise ValueError("SARIMA Dec31 checkpoint parameters differ from the initial fit")
    return {"ridge": ridge_contracts,
            "sarima": {"fitId": sarima_id, "order": [1, 1, 1],
                       "seasonalOrder": [1, 0, 0, 7], "fitStart": first_label,
                       "fitEnd": last_label, "nobs": nobs,
                       "sourceStateThroughBeforeRecovery": "2025-11-30",
                       "checkpointPath": "checkpoints/onion-2025-12-sarima-through-2025-12-31.joblib",
                       "checkpointLastStateLabel": checkpoint_last,
                       "checkpointNobs": checkpoint_nobs,
                       "checkpointParametersEqualInitial": True}}


def audit_saved_overlaps(out_dir: Path = OUT) -> dict[str, Any]:
    """No-fit full-December component audit, bound to the current cache pair."""
    import datetime as _dt

    equal_info._check_inputs()
    manifest, cache, manifest_raw, cache_raw = _validate_pair(out_dir)
    source_raw = baseline.EXISTING_MANIFEST.read_bytes()
    if _sha(source_raw) != baseline.EQUAL_INFO_MANIFEST_SHA256:
        raise ValueError("equal-info source manifest byte hash differs from its reviewed pin")
    source_doc = json.loads(source_raw)
    baseline._verify_equal_info_manifest(source_doc)
    source_rows = baseline._extract_reusable_rows(source_doc)
    for field in ("confirmationForecasts", "evaluationForecasts"):
        for source in source_doc.get(field, []):
            if source.get("product") != "onion" or source.get("eligible") is not True:
                continue
            key = baseline._key("onion", baseline._date(source["origin"]),
                                baseline._date(source["target"]), int(source["horizon"]))
            if key in source_rows:
                direct = source.get("models", {}).get("causalDirect") or {}
                source_rows[key]["sourceDirectProvenance"] = direct.get("provenance") or {}
    start, end = dt.date(2025, 12, 1), dt.date(2025, 12, 30)
    selected = {key for key in source_rows
                if start <= _dt.date.fromisoformat(key[1]) <= end}
    if len(selected) != 255:
        raise ValueError(f"expected 255 eligible full-month source overlap rows, found {len(selected)}")
    replay = _replay_keys(manifest, cache, out_dir, selected, start, end)
    report = audit_overlap(source_rows, replay, exact_expected_keys=selected)
    original_attempt_keys = {baseline._key("onion", origin,
                              origin + dt.timedelta(days=h), h)
                             for day_offset in range(10)
                             for origin in (MISSING_ORIGIN_START + dt.timedelta(days=day_offset),)
                             for h in equal_info.HORIZONS}
    original_overlap = selected & original_attempt_keys
    if len(original_overlap) != 45:
        raise ValueError(f"expected 45 overlaps in original 100 emitted rows, found {len(original_overlap)}")
    contracts = _validate_december_model_contracts(manifest, out_dir, source_doc)
    fit_artifacts = _december_fit_evidence(manifest, out_dir)
    report.update({"status": "passed", "experiment": "FINAL_ENSEMBLE_LOSS_ONION_V2",
                   "recoveryId": RECOVERY_ID,
                   "fullMonthReplayOverlaps": 255,
                   "originalAttemptEmittedRows": 100,
                   "originalAttemptExistingOverlaps": 45,
                   "originalAttemptMissingRows": 55,
                   "sourceManifestSha256": _sha(source_raw),
                   "sourceCacheFileSha256": _sha(cache_raw),
                   "sourceCacheManifestFileSha256": _sha(manifest_raw),
                   "recoveryCodeSha256": _sha(Path(__file__).read_bytes()),
                   "modelContracts": contracts,
                   "decemberFitArtifacts": fit_artifacts,
                   "sourceRowsUnchanged": True,
                   "targetValuesAttached": False,
                   "newModelFits": 0,
                   "perDecemberFitElapsedMs": None,
                   "perDecemberFitElapsedUnknown": True})
    unsigned = dict(report)
    report["auditSha256"] = _sha(_canonical(unsigned))
    audit_path = out_dir / AUDIT_NAME
    temp_path = out_dir / (AUDIT_NAME + ".next")
    temp_path.write_bytes(_canonical(report))
    temp_path.replace(audit_path)
    return report


def _validated_audit_binding(out_dir: Path, cache_raw: bytes,
                             manifest_raw: bytes) -> dict[str, Any]:
    path = out_dir / AUDIT_NAME
    if not path.is_file():
        raise ValueError("successful saved overlap audit is required before recovery")
    report = json.loads(path.read_bytes())
    unsigned = dict(report)
    audit_sha = unsigned.pop("auditSha256", None)
    if audit_sha != _sha(_canonical(unsigned)) or report.get("status") != "passed":
        raise ValueError("saved recovery audit is invalid or failed")
    if (report.get("recoveryCodeSha256") != _sha(Path(__file__).read_bytes())
            or report.get("sourceCacheFileSha256") != _sha(cache_raw)
            or report.get("sourceCacheManifestFileSha256") != _sha(manifest_raw)):
        raise ValueError("saved recovery audit does not bind current code and cache pair")
    equal_info._check_inputs()
    source_raw = baseline.EXISTING_MANIFEST.read_bytes()
    if (_sha(source_raw) != baseline.EQUAL_INFO_MANIFEST_SHA256
            or report.get("sourceManifestSha256") != _sha(source_raw)):
        raise ValueError("saved recovery audit does not bind current pinned source manifest")
    current_manifest = json.loads(manifest_raw)
    current_fits = _december_fit_evidence(current_manifest, out_dir)
    if report.get("decemberFitArtifacts") != current_fits:
        raise ValueError("December model/checkpoint hashes changed after the saved audit")
    if (report.get("fullMonthReplayOverlaps") != 255
            or report.get("originalAttemptExistingOverlaps") != 45
            or report.get("originalAttemptMissingRows") != 55
            or report.get("newModelFits") != 0
            or report.get("targetValuesAttached") is not False
            or report.get("allStatusesAndTimeContractsExact") is not True
            or report.get("sarimaPredictionExact") is not True):
        raise ValueError("saved recovery audit does not satisfy the reviewed overlap criteria")
    if any(value > MAX_ULPS for value in report.get("maxUlpsDifference", {}).values()):
        raise ValueError("saved recovery audit exceeded the numeric replay tolerance")
    return report


def _replay_keys(manifest: dict[str, Any], cache: dict[str, Any], out_dir: Path,
                 selected_keys: set[tuple[str, str, str, int]],
                 origin_start: dt.date, origin_end: dt.date) -> list[dict[str, Any]]:
    from . import direct_horizon

    prices = json.loads(equal_info.PRICES_PATH.read_text(encoding="utf-8"))
    records, _ = equal_info._records(prices)
    direct_series = direct_horizon._observations(prices)
    source_records = records["onion"]
    series = {day: item["price"] for day, item in source_records.items()}
    month = "2025-12"
    # Load the existing initial SARIMA fit and advance it through Dec 21.
    sarima_id = baseline._fit_id(month)
    initial = _fit_by_id(manifest, out_dir, sarima_id)
    # Fit boundary is Dec 1; reset the initial state cursor to Nov 30.
    state_day = dt.date(2025, 11, 30)
    ridge = {h: _fit_by_id(manifest, out_dir, baseline._fit_id(month, h))
             for h in equal_info.HORIZONS}
    cache_features: dict[dt.date, tuple[Any, ...]] = {}
    rows = []
    state_error = None
    missing = _expected_missing(cache["rows"])
    for day_offset in range((origin_end - origin_start).days + 1):
        origin = origin_start + dt.timedelta(days=day_offset)
        initial, state_day, state_error = baseline._advance_sarima_state(
            initial, state_day, origin, series)
        for horizon in equal_info.HORIZONS:
            target = origin + dt.timedelta(days=horizon)
            key = baseline._key("onion", origin, target, horizon)
            if key not in selected_keys:
                continue
            feature, p0, audit = equal_info._direct_feature(
                direct_series["onion"], target, horizon, cache_features)
            direct, direct_prediction = baseline._ridge_output(
                ridge[horizon], None, feature, p0)
            source_date, source_age = equal_info._latest_pre_origin_source(source_records, origin)
            sarima_audit = {"trainStart": "2023-12-02", "trainEnd": "2025-11-30",
                            "fitOrigin": "2025-12-01",
                            "trainingCalendarDays": 730, "order": [1, 1, 1],
                            "seasonalOrder": [1, 0, 0, 7], "selectedD": 1}
            sarima = baseline._sarima_output(initial, None, source_date, source_age,
                                             horizon, state_error, sarima_audit)
            if direct["status"] == "issued" and sarima["status"] == "issued":
                value, status, fallback, reason = .5 * direct_prediction + .5 * sarima["prediction"], "issued", False, None
            elif direct["status"] == "issued":
                value, status, fallback, reason = direct_prediction, "issued", True, None
            else:
                value, status, fallback, reason = None, "not_issued", False, direct.get("reason")
            rows.append({"product": "onion", "origin": origin.isoformat(),
                         "target": target.isoformat(), "horizon": horizon,
                         "B": value, "sourceStatus": status,
                         "BSourceStatus": "fallback" if fallback else status,
                         "sourceReason": reason, "fitId": baseline._fit_id(month, horizon),
                         "fitBoundary": "2025-12-01", "cutoff": (origin - dt.timedelta(days=1)).isoformat(),
                         "fitModel": "causalDirectRidge+SARIMA-causal50-v1",
                         "sourcePhase": "generated", "reuse": False,
                         "direct": direct, "sarima": sarima, "fallback": fallback,
                         "fallbackReason": sarima.get("reason") if fallback else None,
                         "provenance": {"p0": p0, "p0Date": audit.get("p0Date"),
                                        "sourceDate": source_date.isoformat() if source_date else None,
                                        "sourceAgeDays": source_age,
                                        "stateThrough": state_day.isoformat() if state_day else None,
                                        "targetValuesAttached": False,
                                        "recoveryId": RECOVERY_ID}})
    return rows


def _replay_missing(manifest: dict[str, Any], cache: dict[str, Any], out_dir: Path) -> list[dict[str, Any]]:
    missing = _expected_missing(cache["rows"])
    rows = _replay_keys(manifest, cache, out_dir, missing,
                        dt.date(2025, 12, 1), MISSING_ORIGIN_END)
    if len(rows) != 55:
        raise ValueError(f"replay generated {len(rows)} rows; expected exactly 55")
    return rows


def recover(out_dir: Path = OUT) -> dict[str, Any]:
    """Append the reviewed 55 missing rows and atomically update cache provenance."""
    started = time.perf_counter()
    manifest, cache, old_manifest_raw, old_cache_raw = _validate_pair(out_dir)
    audit_binding = _validated_audit_binding(out_dir, old_cache_raw, old_manifest_raw)
    old_rows = cache["rows"]
    missing = _expected_missing(old_rows)
    expected_present = int(manifest.get("expectedHistoricalOnionKeys", -1)) - 55
    if (manifest.get("historicalOnionKeysPresent") != expected_present
            or manifest.get("historicalOnionKeysMissing") != 55):
        raise ValueError("cache counts differ from reviewed December failure state")
    replay = _replay_missing(manifest, cache, out_dir)
    december_fit_evidence = _december_fit_evidence(manifest, out_dir)
    december_fit_records = _december_fit_records(december_fit_evidence,
                                                 audit_binding["modelContracts"])
    replay_by_key = {baseline._key(r["product"], baseline._date(r["origin"]),
                                   baseline._date(r["target"]), r["horizon"]): r for r in replay}
    if set(replay_by_key) != missing:
        raise ValueError("replayed keys differ from exact missing key set")
    original_by_key = {baseline._key(r["product"], baseline._date(r["origin"]),
                                     baseline._date(r["target"]), int(r["horizon"])): r for r in old_rows}
    if len(original_by_key) != len(old_rows):
        raise ValueError("existing cache contains duplicate keys")
    preserved_digest = _sha(_canonical(old_rows))
    new_rows = sorted(old_rows + [replay_by_key[k] for k in sorted(missing)],
                      key=lambda r: (r["product"], r["origin"], r["target"], int(r["horizon"])))
    if [r for r in new_rows if baseline._key(r["product"], baseline._date(r["origin"]),
            baseline._date(r["target"]), int(r["horizon"])) in original_by_key] != old_rows:
        # Original rows must remain byte/value-identical and in their original order in source cache.
        raise ValueError("existing rows changed during append-only recovery")
    cache["rows"] = new_rows
    recovery = {"recoveryId": RECOVERY_ID,
                "recoveryCodeSha256": _sha(Path(__file__).read_bytes()),
                "sourceCacheFileSha256": _sha(old_cache_raw),
                "sourceManifestFileSha256": _sha(old_manifest_raw),
                "successfulAuditSha256": audit_binding["auditSha256"],
                "fullMonthReplayOverlaps": audit_binding["fullMonthReplayOverlaps"],
                "originalAttemptExistingOverlaps": audit_binding["originalAttemptExistingOverlaps"],
                "numericToleranceUlps": audit_binding["numericToleranceUlps"],
                "preservedExistingRows": len(old_rows),
                "preservedExistingRowsCanonicalSha256": preserved_digest,
                "appendedRows": len(replay),
                "appendedKeysSha256": _sha(_canonical([list(k) for k in sorted(missing)])),
                "decemberFitArtifacts": december_fit_evidence,
                "priorDecemberFits": {"ridge": 10, "sarima": 1,
                                      "fitElapsedMs": None,
                                      "elapsedUnavailableReason": "not recorded in committed fit inventory"},
                "fitCounts": {"priorRunRidge": 580, "priorRunSarima": 58,
                              "recoveryRidge": 0, "recoverySarima": 0},
                "recoveryElapsedMs": (time.perf_counter() - started) * 1000,
                "targetValuesAttached": False}
    manifest["complete"] = True
    manifest["historicalOnionKeysPresent"] = manifest["expectedHistoricalOnionKeys"]
    manifest["historicalOnionKeysMissing"] = 0
    manifest["totalCacheRowCount"] = len(new_rows)
    manifest["generatedRows"] = sum(row.get("reuse") is False for row in new_rows)
    manifest["onionStatusCounts"] = {
        status: sum(row.get("sourceStatus") == status for row in new_rows if row.get("product") == "onion")
        for status in ("issued", "not_issued")
    }
    manifest["fitCount"]["ridge"] = 580
    manifest["fitCount"]["sarima"] = 58
    existing_fit_ids = {row.get("fitId") for row in manifest.get("fitInventory", [])}
    if existing_fit_ids.intersection(row["fitId"] for row in december_fit_records):
        raise ValueError("December recovery fit record already exists; refusing duplicate inventory")
    manifest["fitInventory"] = manifest.get("fitInventory", []) + december_fit_records
    manifest["recoveryProvenance"] = recovery
    return baseline._commit_cache(out_dir, cache, manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true", help="read-only integrity and missing-key audit")
    parser.add_argument("--recover", action="store_true", help="append exactly 55 rows using saved models")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.audit == args.recover:
        parser.error("select exactly one of --audit or --recover")
    _manifest, cache, _, _ = _validate_pair(args.out_dir)
    if args.audit:
        print(json.dumps(audit_saved_overlaps(args.out_dir), indent=2))
        return 0
    result = recover(args.out_dir)
    print(json.dumps({"status": "recovered", "complete": result["complete"],
                      "recoveryProvenance": result["recoveryProvenance"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
