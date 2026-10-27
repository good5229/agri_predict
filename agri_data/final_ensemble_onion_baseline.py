"""Answer-free onion causal50 baseline cache for FINAL_ENSEMBLE_LOSS_ONION_V2.

The first command exposed by this module is a read-only inventory. Building a
new cache requires an explicit ``--build`` flag; ordinary preflight never fits
models or attaches target answers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

from . import equal_info_online_forecasts as equal_info

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/final-ensemble-loss-onion-v2-20261014"
EXISTING_MANIFEST = equal_info.OUT / "forecast-manifest.json"
MANIFEST_NAME = "baseline-cache-manifest.json"
CACHE_NAME = "baseline-cache.json"
EQUAL_INFO_MANIFEST_SHA256 = "d7d74823754c2a48d33cc4ddaeab674c686601fb927d399d84c5dbcd235184df"
PRODUCT = "onion"
ORIGIN_START = dt.date(2021, 1, 1)
ORIGIN_END = dt.date(2026, 3, 31)
EXPECTED_STATS_MODELS = "0.14.0"


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _date(value: Any) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def _key(product: str, origin: dt.date, target: dt.date, horizon: int) -> tuple[str, str, str, int]:
    return product, origin.isoformat(), target.isoformat(), int(horizon)


def expected_origin_keys(start: dt.date = ORIGIN_START,
                         end: dt.date = ORIGIN_END) -> set[tuple[str, str, str, int]]:
    """Return the complete onion origin/horizon key set, before status filtering."""
    if end < start:
        raise ValueError("origin end precedes origin start")
    return {
        _key(PRODUCT, origin, origin + dt.timedelta(days=h), h)
        for day_offset in range((end - start).days + 1)
        for origin in (start + dt.timedelta(days=day_offset),)
        for h in equal_info.HORIZONS
    }


def _verify_equal_info_manifest(doc: dict[str, Any]) -> None:
    equal_info._validate(doc)
    input_hashes = doc.get("inputSha256", {})
    expected_hashes = {
        "prices": equal_info.PRICES_SHA256,
        "fixed50Manifest": equal_info.FIXED50_SHA256,
        "hyperparameterManifest": equal_info.HYPERPARAM_SHA256,
        "module": file_sha256(Path(equal_info.__file__)),
        "featureLibrary": equal_info.LIBRARY_SHA256,
        "richHelper": equal_info.RICH_HELPER_SHA256,
        "narrowHelper": equal_info.NARROW_HELPER_SHA256,
        "sarimaHelper": equal_info.TEN_HELPER_SHA256,
        "modelingHelper": equal_info.MODELING_SHA256,
        "directHorizonHelper": equal_info.DIRECT_HELPER_SHA256,
    }
    for key, expected in expected_hashes.items():
        if input_hashes.get(key) != expected:
            raise ValueError(f"equal-info cache pinned source mismatch: {key}")
    if doc.get("manifestSha256") != "3d714acd76282f25830d2f789688d9f1b1128142f9fe46a1fcf4edc5a6a3b5a3":
        raise ValueError("equal-info canonical manifest digest differs from the reviewed saved run")
    versions = doc.get("protocol", {}).get("runtimeVersions", {})
    if versions.get("statsmodels") != EXPECTED_STATS_MODELS:
        raise ValueError("equal-info cache statsmodels version is not the pinned 0.14.0")
    import sklearn
    import statsmodels
    if statsmodels.__version__ != versions.get("statsmodels"):
        raise ValueError("installed statsmodels differs from the saved equal-info runtime")
    if sklearn.__version__ != versions.get("scikitLearn"):
        raise ValueError("installed scikit-learn differs from the saved equal-info runtime")


def _extract_reusable_rows(doc: dict[str, Any]) -> dict[tuple[str, str, str, int], dict[str, Any]]:
    """Extract only exact onion causal50 rows from validated saved phases."""
    _verify_equal_info_manifest(doc)
    out: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    for phase, field in (("confirmation", "confirmationForecasts"),
                         ("evaluation", "evaluationForecasts")):
        for row in doc[field]:
            if row.get("product") != PRODUCT:
                continue
            target, origin = _date(row["target"]), _date(row["origin"])
            horizon = int(row["horizon"])
            key = _key(PRODUCT, origin, target, horizon)
            model = row.get("models", {}).get("causal50")
            direct = row.get("models", {}).get("causalDirect")
            sarima = row.get("models", {}).get("causalSarima")
            if not isinstance(model, dict) or not isinstance(direct, dict):
                raise ValueError(f"missing saved causal baseline for {key}")
            if row.get("eligible") is not True:
                # These keys were deliberately outside the source phase's fit window;
                # their placeholder not-issued state is not a reusable forecast attempt.
                continue
            if model.get("status") not in ("issued", "not_issued"):
                raise ValueError(f"invalid causal50 status for {key}")
            pred = model.get("prediction")
            if model["status"] == "issued":
                try:
                    pred = float(pred)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(f"invalid issued causal50 value for {key}") from exc
                if not math.isfinite(pred) or pred <= 0:
                    raise ValueError(f"invalid issued causal50 value for {key}")
            elif pred is not None:
                raise ValueError(f"not-issued causal50 row has a prediction for {key}")
            if key in out:
                raise ValueError(f"duplicate saved onion causal50 key: {key}")
            training = direct.get("training") or {}
            fit_origin = training.get("fitOrigin")
            fit_month = str(fit_origin)[:7] if fit_origin else origin.strftime("%Y-%m")
            out[key] = {
                "product": PRODUCT,
                "origin": origin.isoformat(),
                "target": target.isoformat(),
                "horizon": horizon,
                "B": pred,
                "sourceStatus": model["status"],
                "BSourceStatus": ("fallback" if model["status"] == "issued"
                                  and (model.get("provenance") or {}).get("fallback") is True
                                  else model["status"]),
                "sourceReason": model.get("reason"),
                "fitId": _fit_id(fit_month, horizon),
                "fitBoundary": fit_origin,
                "fitMonth": fit_month,
                "cutoff": (origin - dt.timedelta(days=1)).isoformat(),
                "fitModel": "equal-info-online-20261010.causal50-v1",
                "sourcePhase": phase,
                "sourceManifestSha256": doc["manifestSha256"],
                "sourceModel": "causal50",
                "sourceDirectStatus": direct.get("status"),
                "sourceDirectReason": direct.get("reason"),
                "sourceDirectPrediction": direct.get("prediction"),
                "sourceSarima": sarima,
                "sourceBlendProvenance": model.get("provenance"),
                "sourceEligible": True,
                "reuse": True,
            }
    return out


def compare_with_rhs(source_rows: dict[tuple[str, str, str, int], dict[str, Any]],
                     rhs_rows: list[dict[str, Any]], phase: str,
                     *, expected_count: int) -> dict[str, Any]:
    """Check onion equal-info B against the pinned phase RHS, without labels."""
    if phase not in ("V", "T"):
        raise ValueError("phase must be V or T")
    rhs = {}
    for row in rhs_rows:
        if row.get("product") != PRODUCT:
            continue
        key = _key(PRODUCT, _date(row["origin"]), _date(row["target"]), int(row["horizon"]))
        if key in rhs:
            raise ValueError(f"duplicate {phase} RHS onion key: {key}")
        rhs[key] = row
    common = set(source_rows) & set(rhs)
    if len(rhs) != expected_count or len(common) != expected_count:
        raise ValueError(f"{phase} RHS onion key coverage mismatch: rhs={len(rhs)}, common={len(common)}")
    mismatches = []
    digest_rows = []
    for key in sorted(common):
        left, right = source_rows[key], rhs[key]
        left_status = left.get("BSourceStatus")
        right_status = right.get("BSourceStatus", right.get("status"))
        left_value, right_value = left.get("B"), right.get("B", right.get("prediction"))
        if left_status != right_status or left_value != right_value:
            mismatches.append({"key": key, "leftStatus": left_status,
                               "rightStatus": right_status,
                               "leftValue": left_value, "rightValue": right_value})
        digest_rows.append([*key, left_status, left_value])
    if mismatches:
        raise ValueError(f"{phase} onion B differs from pinned RHS on {len(mismatches)} keys")
    return {"phase": phase, "commonOnionKeys": len(common), "mismatchCount": 0,
            "equalBStatusAndPrice": True,
            "comparisonSha256": sha256_bytes(canonical(digest_rows))}


def compare_saved_vt_baselines(source_rows: dict[tuple[str, str, str, int], dict[str, Any]]) -> dict[str, Any]:
    """Use the reviewed direct-path baseline reader to compare exact pinned V/T keys."""
    from . import direct_price_path_v2

    comparisons = {}
    for phase, count in (("V", 755), ("T", 790)):
        _manifest, rhs_rows = direct_price_path_v2._baseline_phase(phase)
        comparisons[phase] = compare_with_rhs(source_rows, rhs_rows, phase, expected_count=count)
    return comparisons


def _phase_reference_rows(existing_manifest: Path = EXISTING_MANIFEST) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from . import direct_price_path_v2

    rows = []
    comparison = {}
    for phase, count in (("V", 755), ("T", 790)):
        _manifest, phase_rows = direct_price_path_v2._baseline_phase(phase)
        comparison[phase] = compare_with_rhs(
            _extract_reusable_rows(json.loads(existing_manifest.read_text())),
            phase_rows, phase, expected_count=count)
        for source in phase_rows:
            rows.append({"product": source["product"],
                         "origin": source["origin"].isoformat(),
                         "target": source["target"].isoformat(),
                         "horizon": int(source["horizon"]),
                         "B": source["B"],
                         "sourceStatus": "issued",
                         "BSourceStatus": source["BSourceStatus"],
                         "sourceReason": source.get("baselineReason"),
                         "fitModel": "pinned-RHS-condition-C-v1",
                         "sourcePhase": phase, "reuse": True,
                         "baselineReason": source.get("baselineReason")})
    return rows, comparison


def _load_previous_cache(out_dir: Path) -> tuple[dict[tuple[str, str, str, int], dict[str, Any]], list[dict[str, Any]]]:
    cache_path = out_dir / CACHE_NAME
    manifest_path = out_dir / MANIFEST_NAME
    if not cache_path.exists() and not manifest_path.exists():
        return {}, []
    errors = []
    candidates = [(cache_path, manifest_path),
                  (out_dir / (CACHE_NAME + ".previous"), out_dir / (MANIFEST_NAME + ".previous")),
                  (out_dir / (CACHE_NAME + ".next"), out_dir / (MANIFEST_NAME + ".next"))]
    for cache_candidate, manifest_candidate in candidates:
        if not cache_candidate.is_file() or not manifest_candidate.is_file():
            continue
        try:
            cache_raw, manifest_raw = cache_candidate.read_bytes(), manifest_candidate.read_bytes()
            cache, manifest = json.loads(cache_raw), json.loads(manifest_raw)
            check = dict(manifest)
            recorded = check.pop("manifestSha256", None)
            if recorded != sha256_bytes(canonical(check)):
                raise ValueError("prior baseline cache manifest self-digest mismatch")
            if manifest.get("cacheFileSha256") != sha256_bytes(cache_raw):
                raise ValueError("prior baseline cache file hash mismatch")
            if manifest.get("phase") != "answer_free" or manifest.get("targetValuesAttached") is not False:
                raise ValueError("prior baseline cache is not answer-free")
            if manifest.get("equalInfoManifestFileSha256") != EQUAL_INFO_MANIFEST_SHA256:
                raise ValueError("prior baseline cache source pin mismatch")
            if manifest.get("sourceCodeSha256", {}).get("baselineCacheModule") != file_sha256(Path(__file__)):
                raise ValueError("prior baseline cache producer code hash changed")
            if manifest.get("sourceCodeSha256", {}).get("equalInfoModule") != file_sha256(Path(equal_info.__file__)):
                raise ValueError("prior baseline cache equal-info code hash changed")
            if manifest.get("sourceCodeSha256", {}).get("directPricePathV2") != file_sha256(ROOT / "agri_data/direct_price_path_v2.py"):
                raise ValueError("prior baseline cache RHS reader code hash changed")
            rows = cache.get("rows")
            if not isinstance(rows, list):
                raise ValueError("prior baseline cache row payload is malformed")
            if manifest.get("cacheRowsSha256") != sha256_bytes(canonical(rows)):
                raise ValueError("prior baseline cache rows hash mismatch")
            row_index = {}
            for row in rows:
                key = _key(row["product"], _date(row["origin"]), _date(row["target"]), int(row["horizon"]))
                if key in row_index:
                    raise ValueError(f"duplicate prior baseline cache key {key}")
                row_index[key] = row
            fits = manifest.get("fitInventory", [])
            for fit in fits:
                for path_field, hash_field in (("modelPath", "modelSha256"),
                                               ("checkpointPath", "checkpointSha256")):
                    if not fit.get(path_field) or not fit.get(hash_field):
                        continue
                    path = out_dir / fit[path_field]
                    if not path.is_file() or file_sha256(path) != fit[hash_field]:
                        raise ValueError(f"prior fitted model artifact mismatch: {fit.get('fitId')}")
                for checkpoint in fit.get("stateCheckpoints", []):
                    if not checkpoint.get("path") or not checkpoint.get("sha256"):
                        continue
                    path = out_dir / checkpoint["path"]
                    if not path.is_file() or file_sha256(path) != checkpoint["sha256"]:
                        raise ValueError(f"prior SARIMA state checkpoint mismatch: {fit.get('fitId')}")
            return row_index, fits
        except Exception as exc:
            errors.append(str(exc))
    if (out_dir / ".baseline-cache-commit-pending").is_file():
        fatal_markers = ("self-digest mismatch", "file hash mismatch", "rows hash mismatch",
                         "source pin mismatch", "producer code hash changed",
                         "equal-info code hash changed", "RHS reader code hash changed",
                         "artifact mismatch", "not answer-free")
        if any(any(marker in error for marker in fatal_markers) for error in errors):
            raise ValueError("pending cache commit failed integrity validation: " + "; ".join(errors))
        # A first-ever write can be interrupted before either complete pair is
        # durable. Only malformed/partial files may be discarded for a clean rerun.
        return {}, []
    if errors:
        raise ValueError("; ".join(errors))
    return {}, []


def _commit_cache(out_dir: Path, cache_doc: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    """Commit a progress snapshot with a previous-pair recovery point."""
    cache_path, manifest_path = out_dir / CACHE_NAME, out_dir / MANIFEST_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_raw = canonical(cache_doc)
    manifest["cacheFile"] = CACHE_NAME
    manifest["cacheFileSha256"] = sha256_bytes(cache_raw)
    manifest["cacheRowsSha256"] = sha256_bytes(canonical(cache_doc["rows"]))
    manifest.pop("manifestSha256", None)
    manifest["manifestSha256"] = sha256_bytes(canonical(manifest))
    cache_tmp, manifest_tmp = out_dir / (CACHE_NAME + ".next"), out_dir / (MANIFEST_NAME + ".next")
    cache_tmp.write_bytes(cache_raw)
    manifest_tmp.write_bytes(canonical(manifest))
    if cache_path.is_file() and manifest_path.is_file():
        (out_dir / (CACHE_NAME + ".previous")).write_bytes(cache_path.read_bytes())
        (out_dir / (MANIFEST_NAME + ".previous")).write_bytes(manifest_path.read_bytes())
    marker = out_dir / ".baseline-cache-commit-pending"
    marker.write_text(sha256_bytes(cache_raw) + "\n", encoding="utf-8")
    cache_tmp.replace(cache_path)
    manifest_tmp.replace(manifest_path)
    marker.unlink(missing_ok=True)
    return manifest


def build_cache(*, out_dir: Path = OUT, pilot_month: str | None = None,
                existing_manifest: Path = EXISTING_MANIFEST,
                origin_start: dt.date = ORIGIN_START,
                origin_end: dt.date = ORIGIN_END) -> dict[str, Any]:
    """Build answer-free onion B rows; call only after external review approval.

    ``pilot_month`` fits one missing origin month and writes a deliberately
    incomplete cache. A later unrestricted run validates and reuses its rows
    and model artifacts before filling the remaining exact keys.
    """
    equal_info._check_inputs()
    existing_raw = existing_manifest.read_bytes()
    if sha256_bytes(existing_raw) != EQUAL_INFO_MANIFEST_SHA256:
        raise ValueError("equal-info forecast-manifest.json file-byte pin mismatch")
    saved_doc = json.loads(existing_raw)
    source_rows = _extract_reusable_rows(saved_doc)
    expected = expected_origin_keys(origin_start, origin_end)
    prior_rows, prior_fits = _load_previous_cache(out_dir)
    # Keep every eligible saved onion row, including 2026 rows after March;
    # later monthly final fits need those rows as paired training history.
    all_rows = dict(source_rows)
    for key, row in prior_rows.items():
        if key in expected and row.get("sourcePhase") == "generated":
            all_rows[key] = row

    phase_rows, phase_comparison = _phase_reference_rows(existing_manifest)
    phase_by_key = {}
    for row in phase_rows:
        key = _key(row["product"], _date(row["origin"]), _date(row["target"]), row["horizon"])
        if key in phase_by_key:
            raise ValueError(f"duplicate pinned phase reference key {key}")
        phase_by_key[key] = row
    for key, row in phase_by_key.items():
        if key in all_rows:
            existing = all_rows[key]
            if existing.get("BSourceStatus") != row["BSourceStatus"] or existing.get("B") != row["B"]:
                raise ValueError(f"phase reference disagrees with cache row {key}")
            existing["phaseReferences"] = sorted(set(existing.get("phaseReferences", []))
                                                   | {row["sourcePhase"]})
        else:
            all_rows[key] = row

    fit_inventory = list(prior_fits)
    out_dir.mkdir(parents=True, exist_ok=True)

    def commit_progress(build_mode: str) -> dict[str, Any]:
        expected_union = set(expected) | set(phase_by_key)
        missing_union = expected_union - set(all_rows)
        ordered_rows = [all_rows[key] for key in sorted(all_rows)]
        cache_doc = {"schemaVersion": 1, "experiment": "FINAL_ENSEMBLE_LOSS_ONION_V2",
                     "phase": "answer_free", "targetValuesAttached": False,
                     "rows": ordered_rows}
        source_counts = {status: sum(row.get("sourceStatus") == status for row in ordered_rows
                                     if row.get("product") == PRODUCT)
                         for status in ("issued", "not_issued")}
        manifest = {
            "schemaVersion": 1,
            "experiment": "FINAL_ENSEMBLE_LOSS_ONION_V2",
            "phase": "answer_free",
            "targetValuesAttached": False,
            "complete": not missing_union,
            "requestedOriginRange": [origin_start.isoformat(), origin_end.isoformat()],
            "expectedHistoricalOnionKeys": len(expected),
            "historicalOnionKeysPresent": len(expected & set(all_rows)),
            "historicalOnionKeysMissing": len(missing_union & expected),
            "savedEligibleOnionRowsRetained": len(source_rows),
            "totalCacheRowCount": len(ordered_rows),
            "onionStatusCounts": source_counts,
            "equalInfoManifestPath": str(existing_manifest.relative_to(ROOT)),
            "equalInfoManifestFileSha256": sha256_bytes(existing_raw),
            "equalInfoManifestSha256": saved_doc["manifestSha256"],
            "pinnedPriceSnapshotSha256": equal_info.PRICES_SHA256,
            "sourceCodeSha256": {"equalInfoModule": file_sha256(Path(equal_info.__file__)),
                                  "directPricePathV2": file_sha256(ROOT / "agri_data/direct_price_path_v2.py"),
                                  "baselineCacheModule": file_sha256(Path(__file__))},
            "phaseBaselineComparisons": phase_comparison,
            "phaseReferenceKeyCounts": {phase: sum(phase in (row.get("phaseReferences") or [])
                                                      or row.get("sourcePhase") == phase
                                                      for row in ordered_rows)
                                         for phase in ("V", "T")},
            "fitCount": {"ridge": sum(row.get("model") == "causalDirectRidge" for row in fit_inventory),
                         "sarima": sum(row.get("model") == "SARIMA" for row in fit_inventory),
                         "newRidge": sum(row.get("model") == "causalDirectRidge" for row in new_fits),
                         "newSarima": sum(row.get("model") == "SARIMA" for row in new_fits)},
            "fitInventory": fit_inventory,
            "fitCeiling": {"ridge": 630, "sarima": 63},
            "reusedSourceRows": sum(row.get("reuse") is True for row in ordered_rows),
            "generatedRows": sum(row.get("reuse") is False for row in ordered_rows),
            "buildMode": build_mode,
        }
        return _commit_cache(out_dir, cache_doc, manifest)

    missing = expected - set(all_rows)
    by_month: dict[str, list[dt.date]] = {}
    for _, origin_text, _, _ in sorted(missing):
        origin = _date(origin_text)
        by_month.setdefault(origin.strftime("%Y-%m"), []).append(origin)
    month_names = sorted(by_month)
    if pilot_month is not None:
        try:
            dt.date.fromisoformat(pilot_month + "-01")
        except ValueError as exc:
            raise ValueError("pilot_month must use YYYY-MM") from exc
        if pilot_month not in by_month:
            raise ValueError(f"pilot month {pilot_month} has no missing origin keys")
        month_names = [pilot_month]

    new_fits = []
    prices = json.loads(equal_info.PRICES_PATH.read_text())
    for month in month_names:
        origins = sorted(set(by_month[month]))
        fit_origin = _month_first_eligible_origin(month, origin_start)
        generated, month_fits = _origin_month_rows(prices, month, origins, out_dir,
                                                   fit_origin=fit_origin)
        for row in generated:
            key = _key(row["product"], _date(row["origin"]), _date(row["target"]), row["horizon"])
            if key in source_rows:
                saved = source_rows[key]
                if row["sourceStatus"] != saved["sourceStatus"] or row["B"] != saved["B"]:
                    raise ValueError(f"generated replay disagrees with saved causal50 row {key}")
                row = saved
            if key in all_rows and all_rows[key].get("sourcePhase") == "generated":
                if row["sourceStatus"] != all_rows[key].get("sourceStatus") or row["B"] != all_rows[key].get("B"):
                    raise ValueError(f"generated replay disagrees with resumed cache row {key}")
            all_rows[key] = row
        new_fits.extend(month_fits)
        fit_inventory.extend(month_fits)
        commit_progress(f"pilot:{pilot_month}" if pilot_month else f"month:{month}")

    return commit_progress(f"pilot:{pilot_month}" if pilot_month else "complete_or_resumed")


def _month_counts(keys: set[tuple[str, str, str, int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, origin, _, _ in keys:
        month = origin[:7]
        counts[month] = counts.get(month, 0) + 1
    return dict(sorted(counts.items()))


def _fit_id(month: str, horizon: int | None = None) -> str:
    suffix = "sarima" if horizon is None else f"ridge-h{horizon:02d}"
    return f"onion-{month}-{suffix}"


def _month_first_eligible_origin(month: str, range_start: dt.date) -> dt.date:
    return max(dt.date.fromisoformat(month + "-01"), range_start)


def _model_prediction(model: Any, feature: list[float], p0: float) -> tuple[float | None, str | None]:
    try:
        value = p0 * math.exp(float(model.predict([feature])[0]))
    except Exception as exc:
        return None, f"prediction_failed:{type(exc).__name__}"
    if not math.isfinite(value) or value <= 0:
        return None, "nonpositive_or_nonfinite_prediction"
    return value, None


def _numeric_sequence(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (int, float)):
        value = [value]
    return [float(item) for item in value]


def _assert_native_replay(before: Any, after: Any, fit_id: str) -> None:
    left, right = _numeric_sequence(before), _numeric_sequence(after)
    if len(left) != len(right) or any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12)
                                       for a, b in zip(left, right)):
        raise ValueError(f"native reload prediction replay mismatch for {fit_id}")


def _ridge_output(model: Any, fit_error: str | None, feature: list[float] | None,
                  p0: Any) -> tuple[dict[str, Any], float | None]:
    positive = equal_info._positive(p0)
    if fit_error:
        return {"status": "not_issued", "prediction": None, "reason": fit_error}, None
    if model is None:
        return {"status": "not_issued", "prediction": None,
                "reason": "ridge_model_unavailable"}, None
    if feature is None or positive is None:
        return {"status": "not_issued", "prediction": None,
                "reason": "feature_or_p0_unavailable"}, None
    prediction, error = _model_prediction(model, feature, positive)
    if error:
        return {"status": "not_issued", "prediction": None, "reason": error}, None
    return {"status": "issued", "prediction": prediction, "reason": None}, prediction


def _sarima_output(fit: Any, fit_error: str | None, source_date: dt.date | None,
                   age: int | None, horizon: int, state_error: str | None,
                   audit: dict[str, Any]) -> dict[str, Any]:
    reason = fit_error or state_error
    if source_date is None:
        reason = reason or "no_source_observation"
    elif age is not None and age > 14:
        reason = reason or "latest_source_older_than_14_days"
    elif fit is not None and not state_error:
        try:
            mean = fit.get_forecast(steps=horizon + 1).predicted_mean
            log_prediction = float(mean.iloc[-1] if hasattr(mean, "iloc") else mean[-1])
            prediction = math.exp(log_prediction)
            if math.isfinite(prediction) and prediction > 0:
                return {"status": "issued", "prediction": prediction, "reason": None,
                        "fit": audit}
            reason = "invalid_sarima_prediction"
        except Exception as exc:
            reason = f"sarima_forecast_failed:{type(exc).__name__}"
    return {"status": "not_issued", "prediction": None,
            "reason": reason or "sarima_model_unavailable", "fit": audit}


def _advance_sarima_state(fit_state: Any, state_day: dt.date | None,
                          origin: dt.date, series: dict[dt.date, float]) -> tuple[Any, dt.date | None, str | None]:
    """Advance through this origin's cutoff; a failure is retried next origin."""
    from . import ten_strategy_benchmark as ten

    state_error = None
    while fit_state is not None and state_day is not None and state_day < origin - dt.timedelta(days=1):
        day = state_day + dt.timedelta(days=1)
        try:
            fit_state = fit_state.append(ten._calendar_values(series, day, day), refit=False)
            state_day = day
        except Exception as exc:
            state_error = f"sarima_state_update_failed:{type(exc).__name__}"
            break
    return fit_state, state_day, state_error


def _origin_month_rows(prices: dict[str, Any], month: str,
                       origins: list[dt.date], output_dir: Path,
                       *, fit_origin: dt.date | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit one onion month and issue all requested daily origins answer-free."""
    import joblib
    from . import direct_horizon

    records, _ = equal_info._records(prices)
    direct_series = direct_horizon._observations(prices)
    month_first_origin = fit_origin or min(origins)
    grid_rows: list[dict[str, Any]] = []
    fit_rows: list[dict[str, Any]] = []
    rec = records[PRODUCT]
    ordered = sorted(rec)
    direct_cache: dict[dt.date, tuple[Any, ...]] = {}

    ridge_models: dict[int, Any] = {}
    ridge_errors: dict[int, str | None] = {}
    for horizon in equal_info.HORIZONS:
        t0 = __import__("time").perf_counter()
        model, audit, error = equal_info._period_fit(
            records, direct_series, PRODUCT, horizon, month_first_origin, 10.0, "direct")
        fit_id = _fit_id(month, horizon)
        model_path = None
        model_sha = None
        if model is not None:
            rel = Path("models") / f"{fit_id}.joblib"
            dest = output_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            probe = [[0.0] * len(equal_info.modeling.FEATURE_NAMES)]
            before = model.predict(probe)
            joblib.dump(model, dest)
            restored = joblib.load(dest)
            if restored is None:
                raise ValueError(f"ridge native reload smoke check failed for {fit_id}")
            _assert_native_replay(before, restored.predict(probe), fit_id)
            model_path, model_sha = rel.as_posix(), file_sha256(dest)
            model = restored
        elapsed_ms = (__import__("time").perf_counter() - t0) * 1000
        ridge_models[horizon], ridge_errors[horizon] = model, error
        fit_rows.append({"fitId": fit_id, "product": PRODUCT, "model": "causalDirectRidge",
                         "horizon": horizon, "fitMonth": month,
                         "fitBoundary": month_first_origin.isoformat(),
                         "fitOrigin": month_first_origin.isoformat(), "alpha": 10.0,
                         "featureNames": list(equal_info.modeling.FEATURE_NAMES),
                         "trainingCount": audit.get("count", 0),
                         "trainingFirstTarget": audit.get("firstTarget"),
                         "trainingLastTarget": audit.get("lastTarget"),
                         "trainingTargetsBeforeFitOrigin": True,
                         "fitStatus": "issued" if model is not None else "not_issued",
                         "fitReason": error, "modelPath": model_path,
                         "modelSha256": model_sha, "fitElapsedMs": elapsed_ms,
                         "reloadValidated": model is not None})

    sarima_t0 = __import__("time").perf_counter()
    series = {day: item["price"] for day, item in rec.items()}
    initial_sarima = None
    sarima_audit: dict[str, Any] = {}
    sarima_error = None
    try:
        initial_sarima, sarima_audit = equal_info._fit_sarima(series, month_first_origin)
    except Exception as exc:
        sarima_error = f"sarima_fit_failed:{type(exc).__name__}:{exc}"
    sarima_fit_id = _fit_id(month)
    sarima_model_path = None
    sarima_model_sha = None
    if initial_sarima is not None:
        rel = Path("models") / f"{sarima_fit_id}-initial.joblib"
        dest = output_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        before = initial_sarima.get_forecast(steps=1).predicted_mean
        joblib.dump(initial_sarima, dest)
        initial_sarima = joblib.load(dest)
        if not hasattr(initial_sarima, "get_forecast") or not hasattr(initial_sarima, "append"):
            raise ValueError(f"SARIMA native reload smoke check failed for {sarima_fit_id}")
        after = initial_sarima.get_forecast(steps=1).predicted_mean
        _assert_native_replay(before, after, sarima_fit_id)
        sarima_model_path, sarima_model_sha = rel.as_posix(), file_sha256(dest)
    fit_rows.append({"fitId": sarima_fit_id, "product": PRODUCT, "model": "SARIMA",
                     "fitMonth": month, "fitBoundary": month_first_origin.isoformat(),
                     "fitOrigin": month_first_origin.isoformat(), "order": [1, 1, 1],
                     "seasonalOrder": [1, 0, 0, 7], "windowCalendarDays": 730,
                     "fitStart": sarima_audit.get("trainStart"),
                     "fitEnd": sarima_audit.get("trainEnd"),
                     "fitStatus": "issued" if initial_sarima is not None else "not_issued",
                     "fitReason": sarima_error, "modelPath": sarima_model_path,
                     "modelSha256": sarima_model_sha,
                     "fitElapsedMs": (__import__("time").perf_counter() - sarima_t0) * 1000,
                     "reloadValidated": initial_sarima is not None})

    fit_state = initial_sarima
    state_day = month_first_origin - dt.timedelta(days=1) if fit_state is not None else None
    state_checkpoints: list[dict[str, Any]] = []
    for origin in sorted(origins):
        fit_state, state_day, state_error = _advance_sarima_state(
            fit_state, state_day, origin, series)
        if origin.strftime("%Y-%m") != month:
            raise ValueError("origin escaped its fit month")
        if origin.day == 1 or origin == max(origins):
            checkpoint_rel = None
            checkpoint_sha = None
            if fit_state is not None:
                rel = Path("checkpoints") / f"{sarima_fit_id}-through-{origin.isoformat()}.joblib"
                dest = output_dir / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                before = fit_state.get_forecast(steps=1).predicted_mean
                joblib.dump(fit_state, dest)
                reloaded = joblib.load(dest)
                if not hasattr(reloaded, "get_forecast"):
                    raise ValueError(f"SARIMA checkpoint reload smoke check failed for {sarima_fit_id}")
                after = reloaded.get_forecast(steps=1).predicted_mean
                _assert_native_replay(before, after, f"{sarima_fit_id}:{origin.isoformat()}")
                checkpoint_rel, checkpoint_sha = rel.as_posix(), file_sha256(dest)
            state_checkpoints.append({"origin": origin.isoformat(),
                                      "stateThrough": state_day.isoformat() if state_day else None,
                                      "path": checkpoint_rel, "sha256": checkpoint_sha})

        for horizon in equal_info.HORIZONS:
            target = origin + dt.timedelta(days=horizon)
            feature, p0, feature_audit = equal_info._direct_feature(
                direct_series[PRODUCT], target, horizon, direct_cache)
            direct, direct_prediction = _ridge_output(
                ridge_models[horizon], ridge_errors[horizon], feature, p0)
            source_date, source_age = equal_info._latest_pre_origin_source(rec, origin)
            sarima = _sarima_output(fit_state, sarima_error, source_date, source_age,
                                    horizon, state_error, sarima_audit)
            if direct["status"] == "issued" and sarima["status"] == "issued":
                blend = {"status": "issued", "prediction": .5 * direct_prediction + .5 * sarima["prediction"],
                         "reason": None, "fallback": False, "fallbackReason": None}
            elif direct["status"] == "issued":
                blend = {"status": "issued", "prediction": direct_prediction,
                         "reason": None, "fallback": True, "fallbackReason": sarima["reason"]}
            else:
                blend = {"status": "not_issued", "prediction": None,
                         "reason": direct.get("reason") or "ridge_component_unavailable",
                         "fallback": False, "fallbackReason": None}
            grid_rows.append({"product": PRODUCT, "origin": origin.isoformat(),
                              "target": target.isoformat(), "horizon": horizon,
                              "B": blend["prediction"], "sourceStatus": blend["status"],
                              "BSourceStatus": ("fallback" if blend["fallback"] else blend["status"]),
                              "sourceReason": blend["reason"], "fitId": _fit_id(month, horizon),
                              "fitBoundary": month_first_origin.isoformat(),
                              "cutoff": (origin - dt.timedelta(days=1)).isoformat(),
                              "fitModel": "causalDirectRidge+SARIMA-causal50-v1",
                              "sourcePhase": "generated", "reuse": False,
                              "direct": direct, "sarima": sarima,
                              "fallback": blend["fallback"],
                              "fallbackReason": blend["fallbackReason"],
                              "provenance": {"p0": p0, "p0Date": feature_audit.get("p0Date"),
                                             "sourceDate": source_date.isoformat() if source_date else None,
                                             "sourceAgeDays": source_age,
                                             "stateThrough": state_day.isoformat() if state_day else None,
                                             "targetValuesAttached": False}})

    # A month-ending checkpoint records the advanced state needed to audit resumption.
    fit_rows[-1]["stateCheckpoints"] = state_checkpoints
    return grid_rows, fit_rows


def inventory(existing_manifest: Path = EXISTING_MANIFEST,
              origin_start: dt.date = ORIGIN_START,
              origin_end: dt.date = ORIGIN_END) -> dict[str, Any]:
    """Read-only cache and environment preflight; does not fit or write."""
    equal_info._check_inputs()
    raw = existing_manifest.read_bytes()
    if sha256_bytes(raw) != EQUAL_INFO_MANIFEST_SHA256:
        raise ValueError("equal-info forecast-manifest.json file-byte pin mismatch")
    doc = json.loads(raw)
    reusable = _extract_reusable_rows(doc)
    source_presence: set[tuple[str, str, str, int]] = set()
    genuine_failed: set[tuple[str, str, str, int]] = set()
    phase_ineligible: set[tuple[str, str, str, int]] = set()
    for field in ("confirmationForecasts", "evaluationForecasts"):
        for row in doc[field]:
            if row.get("product") != PRODUCT:
                continue
            key = _key(PRODUCT, _date(row["origin"]), _date(row["target"]), int(row["horizon"]))
            source_presence.add(key)
            if row.get("eligible") is not True:
                phase_ineligible.add(key)
            elif row.get("models", {}).get("causal50", {}).get("status") == "not_issued":
                genuine_failed.add(key)
    expected = expected_origin_keys(origin_start, origin_end)
    reused = expected & set(reusable)
    missing = expected - reused
    import statsmodels
    import numpy
    import pandas
    import sklearn
    import joblib
    if statsmodels.__version__ != EXPECTED_STATS_MODELS:
        raise RuntimeError(f"statsmodels {EXPECTED_STATS_MODELS} required; found {statsmodels.__version__}")
    return {
        "status": "preflight_only",
        "targetValuesAttached": False,
        "existingManifest": str(existing_manifest),
        "existingManifestFileSha256": sha256_bytes(raw),
        "existingManifestSha256": doc["manifestSha256"],
        "pinnedPriceSnapshotSha256": equal_info.PRICES_SHA256,
        "runtime": {"python": sys.executable, "pythonVersion": sys.version.split()[0],
                    "statsmodels": statsmodels.__version__,
                    "scikitLearn": sklearn.__version__, "numpy": numpy.__version__,
                    "pandas": pandas.__version__, "joblib": joblib.__version__},
        "requestedOriginRange": [origin_start.isoformat(), origin_end.isoformat()],
        "expectedOnionKeys": len(expected),
        "sourceKeyPresence": len(expected & source_presence),
        "reusableExactKeys": len(reused),
        "genuineFailedForecastAttempts": len(expected & genuine_failed),
        "phaseIneligibleSourcePlaceholders": len(expected & phase_ineligible),
        "sourceKeysOutsideRequestedOriginRange": len(source_presence - expected),
        "missingKeys": len(missing),
        "reusableOriginMonthKeyCounts": _month_counts(reused),
        "missingOriginMonthKeyCounts": _month_counts(missing),
        "reuseByPhase": {
            phase: sum(1 for row in reusable.values()
                       if row["sourcePhase"] == phase and
                       _key(PRODUCT, _date(row["origin"]), _date(row["target"]), row["horizon"]) in expected)
            for phase in ("confirmation", "evaluation")
        },
        "phaseBaselineComparisons": compare_saved_vt_baselines(reusable),
        "fitCeiling": {"ridgeFits": 630, "sarimaFits": 63},
        "modelFitsPerformed": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", action="store_true", help="run the read-only cache/environment preflight")
    parser.add_argument("--build", action="store_true", help="build an answer-free baseline cache after review approval")
    parser.add_argument("--pilot-month", help="fit only one missing origin month and mark the cache incomplete")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--manifest", type=Path, default=EXISTING_MANIFEST)
    args = parser.parse_args(argv)
    if args.inventory and (args.build or args.pilot_month):
        parser.error("--inventory cannot be combined with build options")
    if args.pilot_month and not args.build:
        parser.error("--pilot-month requires --build")
    if args.inventory:
        result = inventory(args.manifest)
    elif args.build:
        result = build_cache(out_dir=args.out_dir, pilot_month=args.pilot_month,
                             existing_manifest=args.manifest)
    else:
        parser.error("select --inventory or --build")
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
