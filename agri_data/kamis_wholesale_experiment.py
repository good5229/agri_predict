"""Frozen, replay-gated experiment for causal KAMIS wholesale survey inputs.

This module keeps the auction-price target unchanged. It creates answer-free
forecast manifests, verifies deterministic replay before attaching outcomes,
and retains the fixed50 policy unless a candidate passes the frozen screen.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

from . import kamis_wholesale_features as survey_features
from . import path_mlp_forecasts as source
from . import path_mlp_score as stats

ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = source.PRICES_PATH
SCREEN_COMPONENTS_PATH = source.SCREEN_COMPONENTS_PATH
EVALUATION_COMPONENTS_PATH = source.EVALUATION_COMPONENTS_PATH
ARCHIVE_DIR = survey_features.ARCHIVE_DIR
DIAGNOSTIC_PATH = ROOT / "outputs/experiments/kamis-wholesale-feature-20261010/diagnostic.json"
OUT_DIR = ROOT / "outputs/experiments/kamis-wholesale-experiment-20261010"
SCREEN_START, SCREEN_END, SCREEN_ORIGIN_START = dt.date(2024, 9, 25), dt.date(2025, 8, 31), dt.date(2024, 9, 15)
EVAL_START, EVAL_END, EVAL_ORIGIN_START = dt.date(2026, 1, 1), dt.date(2026, 9, 27), dt.date(2026, 1, 1)
TRAIN_START = dt.date(2014, 1, 1)
MIN_TRAIN = 90
BANDWIDTHS = (9, 28)
ALPHA = 0.05
CROPS = tuple(survey_features.CROPS)
HORIZONS = tuple(range(1, 11))
MODELS = ("ridge106", "ridge113")
COMPARISONS = (("ridge113_vs_fixed50", "blend113", "fixed50"),
               ("ridge113_vs_ridge106", "blend113", "blend106"))
LOSSES = ("ae", "ape")
PRICES_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
SCREEN_COMPONENTS_SHA256 = "6ade501d5818cca33ad973963d12349f51b5ee3bed27afcd2cc023ae972fed74"
EVALUATION_COMPONENTS_SHA256 = "1497ee362945fcab45353a7b2403be752af3bf63c5f584cc0aedb2cc111ceaa8"
ARCHIVE_PLAN_SHA256 = "11e5655adcf51ef49400cf00abb150e96b962a505325308af5090f77ea41d26a"
ARCHIVE_SOURCE_MANIFEST_SHA256 = "ad97dcfc1809f9f27fdd1938330146208ac8afb57d29314d3fbef972c9099d88"
DIAGNOSTIC_SHA256 = "db2489fe711ab719e5ce8109cf8a98895ff4c0099d0fd941b1471791f6cd05ed"


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _code_hashes() -> dict[str, str]:
    modules = {"experiment": sys.modules[__name__], "kamisFeatures": survey_features,
        "priceFeatureBaseline": survey_features.baseline,
        "forecastSource": source, "directHorizon": source.direct_horizon,
        "narrowFeatures": source.narrow, "richFeatures": source.rich,
        "featureLibrary": source.library, "modeling": source.modeling,
        "statistics": stats}
    return {name: _sha(Path(module.__file__).read_bytes()) for name, module in modules.items()}


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_pretty(value))


def _positive(value: Any) -> float | None:
    if isinstance(value, bool): return None
    try: x = float(value)
    except (TypeError, ValueError, OverflowError): return None
    return x if math.isfinite(x) and x > 0 else None


def _date(value: Any) -> dt.date | None:
    if not isinstance(value, str): return None
    try: return dt.date.fromisoformat(value)
    except ValueError: return None


def _phase(phase: str):
    if phase == "screen": return SCREEN_START, SCREEN_END, SCREEN_ORIGIN_START
    if phase == "evaluation": return EVAL_START, EVAL_END, EVAL_ORIGIN_START
    raise ValueError("phase must be screen or evaluation")


def _grid(phase: str):
    start, end, _ = _phase(phase)
    return [(crop, start + dt.timedelta(days=i), horizon)
            for crop in CROPS for i in range((end - start).days + 1) for horizon in HORIZONS]


def _model(status: str, prediction: float | None = None, reason: str | None = None, **extra):
    return {"status": status, "prediction": prediction, "reason": reason, **extra}


def _component_map(raw: dict[str, Any], phase: str):
    return source._component_rows(raw, phase)


def _recompute_diagnostic_gates(prices_raw: bytes, archive: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Recompute the diagnostic eligibility gate from the pinned source rows.

    No target values are joined here. The monthly origin sample and cutoff are
    the same as the documented read-only diagnostic; only the eligibility
    statistic (joint base/q0 origins and nonconstant q0/p0) is returned.
    """
    prices = json.loads(prices_raw)
    trimmed, _ = source._trim_prices(prices, survey_features.DIAGNOSTIC_MAX_SOURCE_DATE)
    direct, records = source._source_rows(trimmed)
    start, end = survey_features.DIAGNOSTIC_START, survey_features.DIAGNOSTIC_END
    origins = [dt.date(y, m, 1) for y in range(start.year, end.year + 1)
               for m in range(1, 13) if dt.date(y, m, 1) <= end]
    gates = {}
    for crop in CROPS:
        surveys = archive["recordsByCrop"][crop]
        ratios = []
        for origin in origins:
            target = origin + dt.timedelta(days=1)
            vector, _, audit = survey_features.features(crop, records[crop], direct[crop], surveys, target, 1)
            if vector is None: continue
            if vector[:106] is None or len(vector) != 113:
                raise ValueError("diagnostic source feature schema mismatch")
            if audit["anchors"]["0"]["available"]:
                ratios.append(float(vector[107]))
        gates[crop] = survey_features._entry_gate(ratios)
    return gates


def _validate_diagnostic(prices_raw: bytes, archive: dict[str, Any], diagnostic_raw: bytes | None):
    if diagnostic_raw is None:
        raise ValueError("pinned KAMIS availability diagnostic is required before fitting")
    if _sha(diagnostic_raw) != DIAGNOSTIC_SHA256:
        raise ValueError("pinned KAMIS diagnostic SHA-256 mismatch")
    diagnostic = json.loads(diagnostic_raw)
    if diagnostic.get("archiveValidation", {}).get("collectionPlanSha256") != archive["collectionPlanSha256"] or \
       diagnostic.get("archiveValidation", {}).get("sourceManifestSha256") != archive["sourceManifestSha256"] or \
       diagnostic.get("auctionSource", {}).get("sha256") != _sha(prices_raw):
        raise ValueError("availability diagnostic source identity mismatch")
    recomputed = _recompute_diagnostic_gates(prices_raw, archive)
    for crop in CROPS:
        claimed = diagnostic.get("perCrop", {}).get(crop, {}).get("entryGate")
        if claimed != recomputed[crop]:
            raise ValueError(f"diagnostic eligibility gate differs from original sources for {crop}")
    return diagnostic, recomputed


def _features(cache: dict, crop, target, horizon, records, direct, surveys):
    key = (crop, target, horizon)
    if key not in cache:
        expected_cutoff = target - dt.timedelta(days=horizon + 1)
        # Filter keys before retrieving their values, so post-cutoff values are
        # not even handed to the feature helper.
        causal_records = {day: records[crop][day] for day in records[crop] if day <= expected_cutoff}
        causal_direct = {day: direct[crop][day] for day in direct[crop] if day <= expected_cutoff}
        causal_surveys = {day: surveys[crop][day] for day in surveys[crop] if day <= expected_cutoff}
        vector, p0, audit = survey_features.features(crop, causal_records, causal_direct, causal_surveys, target, horizon)
        if audit.get("featureCutoff") != expected_cutoff.isoformat():
            raise ValueError("KAMIS feature helper violated origin-minus-one cutoff")
        if vector is not None and (len(vector) != 113 or _positive(p0) is None):
            raise ValueError("KAMIS feature helper returned malformed features/p0")
        cache[key] = (vector, p0, audit)
    return cache[key]


def _fit(training, feature_count):
    distinct = len({row[0] for row in training})
    if distinct < MIN_TRAIN:
        return None, {"status": "insufficient_training", "distinctTargets": distinct, "samples": len(training)}
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits
    x = np.asarray([row[2][:feature_count] for row in training], dtype=float)
    y = np.asarray([math.log(row[1] / row[3]) for row in training], dtype=float)
    if x.shape != (len(training), feature_count) or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None, {"status": "invalid_training_values", "distinctTargets": distinct, "samples": len(training)}
    estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    with threadpool_limits(limits=1): estimator.fit(x, y)
    return estimator, {"status": "fitted", "distinctTargets": distinct, "samples": len(training),
                       "maxTrainingTarget": max(row[0] for row in training).isoformat(), "featureCount": feature_count}


def _predict(estimator, vector, p0, feature_count):
    if estimator is None: return _model("not_issued", reason="model_not_fitted")
    if vector is None or _positive(p0) is None: return _model("not_issued", reason="feature_or_p0_unavailable")
    import numpy as np
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1): result = float(estimator.predict(np.asarray([vector[:feature_count]], dtype=float))[0])
    try: value = float(p0) * math.exp(result)
    except (OverflowError, ValueError): value = math.nan
    if not math.isfinite(value) or value <= 0: return _model("not_issued", reason="inverse_transform_nonpositive_or_nonfinite")
    return _model("issued", value)


def _load_inputs(prices_raw, components_raw, phase, archive_dir):
    start, end, _ = _phase(phase)
    price_doc = json.loads(prices_raw)
    trimmed, trim_audit = source._trim_prices(price_doc, end)
    direct, records = source._source_rows(trimmed)
    component_doc = json.loads(components_raw)
    components = _component_map(component_doc, phase)
    archive = survey_features.validate_archive(archive_dir)
    if archive["collectionPlanSha256"] != ARCHIVE_PLAN_SHA256 or archive["sourceManifestSha256"] != ARCHIVE_SOURCE_MANIFEST_SHA256:
        raise ValueError("pinned KAMIS archive identity mismatch")
    return price_doc, direct, records, archive["recordsByCrop"], components, archive, trim_audit


def _run_build(prices_raw: bytes, components_raw: bytes, phase: str, archive_dir: Path,
               diagnostic_raw: bytes, *, selections=None, replay_binding=None,
               diagnostic_override=None, loaded=None, progress=None):
    start, end, origin_start = _phase(phase)
    prices, direct, records, surveys, components, archive, trim_audit = loaded or _load_inputs(
        prices_raw, components_raw, phase, archive_dir)
    if _sha(prices_raw) != PRICES_SHA256: raise ValueError("pinned auction price SHA-256 mismatch")
    expected_comp = SCREEN_COMPONENTS_SHA256 if phase == "screen" else EVALUATION_COMPONENTS_SHA256
    if _sha(components_raw) != expected_comp: raise ValueError("pinned component source SHA-256 mismatch")
    diag, gates = diagnostic_override or _validate_diagnostic(prices_raw, archive, diagnostic_raw)
    selected = selections or {crop: "retain_fixed50" for crop in CROPS}
    active_crops = set(CROPS) if phase == "screen" else {crop for crop in CROPS if selected.get(crop) == "ridge113"}
    if phase == "evaluation" and not active_crops:
        raise ValueError("no screen-selected crop; evaluation fitting is blocked")
    # Check the numerical environment before starting any fit.
    env = {**source._check_environment(), "python": sys.version.split()[0], "blasThreads": 1}
    feature_cache = {}
    forecast = {}
    fit_audit = []
    phase_grid = _grid(phase)
    monthly = {}
    for key in phase_grid:
        crop, target, h = key
        origin = target - dt.timedelta(days=h)
        if origin >= origin_start: monthly.setdefault((origin.year, origin.month), []).append(key)
    price_maps = {crop: {day: pair[0] for day, pair in rows.items()} for crop, rows in direct.items()}
    for (year, month), month_keys in sorted(monthly.items()):
        fit_origin = min(target - dt.timedelta(days=h) for _, target, h in month_keys)
        for crop in CROPS:
            for horizon in HORIZONS:
                keys = [key for key in month_keys if key[0] == crop and key[2] == horizon]
                if not keys: continue
                diag_eligible = gates[crop]["eligibleForLaterReview"]
                fit_here = crop in active_crops and diag_eligible
                training = []
                support = {"positiveTargetsBeforeFitOrigin": 0, "featureAvailable": 0, "kamisQ0Available": 0}
                if fit_here:
                    for target, actual in sorted(price_maps[crop].items()):
                        if target < TRAIN_START or target >= fit_origin: continue
                        support["positiveTargetsBeforeFitOrigin"] += 1
                        vector, p0, audit = _features(feature_cache, crop, target, horizon, records, direct, surveys)
                        if vector is not None and _positive(p0) is not None:
                            training.append((target, actual, vector, p0, audit))
                            support["featureAvailable"] += 1
                            support["kamisQ0Available"] += int(audit["anchors"]["0"]["available"])
                models, audits = {}, {}
                for name, width in (("ridge106", 106), ("ridge113", 113)):
                    enabled = fit_here and (phase == "screen" or crop in active_crops)
                    if not enabled:
                        models[name] = None
                        reason = "diagnostic_gate_ineligible" if not diag_eligible else "not_selected_for_evaluation"
                        audits[name] = {"status": reason, "distinctTargets": 0, "samples": 0}
                    else:
                        models[name], audits[name] = _fit(training, width)
                        audits[name]["trainingSupport"] = dict(support)
                for crop0, target, h in keys:
                    origin = target - dt.timedelta(days=h)
                    key = (crop0, target, h)
                    component = components.get(key)
                    fixed = _positive(component.get("fixed50")) if component else None
                    if origin < origin_start:
                        vector, p0, audit = None, None, {"status": "origin_before_phase_start", "featureCutoff": (origin-dt.timedelta(days=1)).isoformat()}
                    else:
                        vector, p0, audit = _features(feature_cache, crop, target, h, records, direct, surveys)
                    if crop not in active_crops:
                        pred106 = _model("not_issued", reason="crop_not_selected_by_frozen_screen")
                        pred113 = _model("not_issued", reason="crop_not_selected_by_frozen_screen")
                    elif not diag_eligible:
                        pred106 = _model("not_issued", reason="diagnostic_gate_ineligible")
                        pred113 = _model("not_issued", reason="diagnostic_gate_ineligible")
                    else:
                        pred106 = _predict(models["ridge106"], vector, p0, 106)
                        pred113 = _predict(models["ridge113"], vector, p0, 113)
                    sarima = _positive(component.get("sarima")) if component else None
                    def blend(model_row):
                        value = _positive(model_row.get("prediction"))
                        if value is None: return _model("not_issued", reason=model_row.get("reason"))
                        return _model("issued", value if sarima is None else .5 * value + .5 * sarima,
                                      "sarima_missing_ridge_only_fallback" if sarima is None else None,
                                      sarimaFallback=sarima is None)
                    forecast[key] = {"crop": crop, "target_date": target.isoformat(), "origin_date": origin.isoformat(),
                        "horizon": h, "eligible": origin >= origin_start, "fitOrigin": fit_origin.isoformat() if origin >= origin_start else None,
                        "fixed50KeyPresent": component is not None, "featureAudit": audit,
                        "ridge106": pred106, "ridge113": pred113, "blend106": blend(pred106), "blend113": blend(pred113),
                        "sarima": _model("issued", sarima) if sarima is not None else _model("not_issued", reason="source_sarima_missing"),
                        "fixed50": _model("issued", fixed) if fixed is not None else _model("not_issued", reason="source_fixed50_unavailable")}
                for name in MODELS:
                    fit_audit.append({"month": f"{year:04d}-{month:02d}", "crop": crop, "horizon": horizon,
                                      "fitOrigin": fit_origin.isoformat(), "model": name, **audits[name]})
            if progress: progress({"event": "monthly_crop_complete", "phase": phase, "month": f"{year:04d}-{month:02d}", "crop": crop})
    # Keep every planned target/horizon in the manifest. Origins before the
    # phase's frozen start were not eligible to issue any arm in this phase.
    for crop, target, horizon in phase_grid:
        origin = target - dt.timedelta(days=horizon)
        if origin >= origin_start:
            continue
        component = components.get((crop, target, horizon))
        reason = "origin_before_phase_start"
        not_issued = _model("not_issued", reason=reason)
        sarima = _positive(component.get("sarima")) if component else None
        fixed = _positive(component.get("fixed50")) if component else None
        forecast[(crop, target, horizon)] = {
            "crop": crop, "target_date": target.isoformat(), "origin_date": origin.isoformat(),
            "horizon": horizon, "eligible": False, "fitOrigin": None,
            "fixed50KeyPresent": component is not None,
            "featureAudit": {"status": reason, "featureCutoff": (origin-dt.timedelta(days=1)).isoformat()},
            "ridge106": not_issued, "ridge113": not_issued,
            "blend106": not_issued, "blend113": not_issued,
            "sarima": _model("issued", sarima) if sarima is not None else _model("not_issued", reason=reason),
            "fixed50": (_model("issued", fixed) if fixed is not None else
                        _model("not_issued", reason="source_fixed50_unavailable")),
        }
    diag_identity = {"sha256": _sha(diagnostic_raw), "archiveSourceManifestSha256": archive["sourceManifestSha256"],
                     "eligibilityRecomputedFromSources": True, "perCropGates": gates}
    manifest = {"schemaVersion": 1, "phase": "forecast_only", "phaseName": phase,
        "targetValuesAttached": False, "selectedModels": selected,
        "inputSha256": {"prices": _sha(prices_raw), "components": _sha(components_raw),
            "archivePlan": archive["collectionPlanSha256"], "archiveSourceManifest": archive["sourceManifestSha256"],
            "diagnostic": _sha(diagnostic_raw), "code": _code_hashes(),
            **({"screenManifest": replay_binding["manifestSha256"], "screenReport": replay_binding["reportSha256"],
                "screenDecision": replay_binding["decisionSha256"]} if replay_binding else {})},
        "protocol": {"trainingStart": TRAIN_START.isoformat(), "minimumDistinctTargets": MIN_TRAIN,
            "features": "same 106 auction prefix plus seven causal KAMIS survey features through origin-1; anchors <=14 days",
            "ridgeAlpha": 10.0, "target": "log(actual/p0)", "inverse": "p0*exp(predicted_log_return)",
            "models": {"ridge106": 106, "ridge113": 113},
            "blend": "50:50 with same-key SARIMA; ridge-only when SARIMA absent; no clipping",
            "diagnosticEntryGate": "recomputed 2014-01..2024-12 monthly origins; >=30 joint base/q0 and nonconstant log(q0/p0)",
            "selection": selected},
        "environment": env, "sourceAudit": trim_audit, "diagnosticIdentity": diag_identity,
        "fitAudit": fit_audit, "forecasts": [forecast[key] for key in phase_grid]}
    manifest["manifestSha256"] = _sha(_canonical(manifest))
    return manifest


def build(prices_raw: bytes, components_raw: bytes, phase: str, *, archive_dir=ARCHIVE_DIR,
          diagnostic_raw: bytes | None = None, decision_raw: bytes | None = None,
          screen_manifest_raw: bytes | None = None, screen_report_raw: bytes | None = None,
          progress: Callable[[dict[str, Any]], None] | None = None, _diagnostic_override=None) -> dict[str, Any]:
    return _build_impl(prices_raw, components_raw, phase, archive_dir=archive_dir,
        diagnostic_raw=diagnostic_raw, decision_raw=decision_raw, screen_manifest_raw=screen_manifest_raw,
        screen_report_raw=screen_report_raw, progress=progress, _diagnostic_override=_diagnostic_override)


def _build_impl(prices_raw: bytes, components_raw: bytes, phase: str, *, archive_dir=ARCHIVE_DIR,
          diagnostic_raw: bytes | None = None, decision_raw: bytes | None = None,
          screen_manifest_raw: bytes | None = None, screen_report_raw: bytes | None = None,
          progress: Callable[[dict[str, Any]], None] | None = None, _diagnostic_override=None,
          verified_replay_binding=None) -> dict[str, Any]:
    """Build a full answer-free phase manifest; evaluation replays screen before fitting."""
    _, _, origin_start = _phase(phase)
    replay_binding, selection = None, {crop: "retain_fixed50" for crop in CROPS}
    if phase == "evaluation":
        if not all((diagnostic_raw, decision_raw, screen_manifest_raw, screen_report_raw)):
            raise ValueError("evaluation requires pinned diagnostic and complete screen artifacts")
        # This full replay/rescore happens before evaluation calls _fit.
        replay_binding = verified_replay_binding or validate_screen_artifacts(
            screen_manifest_raw, prices_raw, SCREEN_COMPONENTS_PATH.read_bytes(),
            screen_report_raw, decision_raw, diagnostic_raw=diagnostic_raw, archive_dir=archive_dir)
        selection = replay_binding["selection"]
        if not any(value == "ridge113" for value in selection.values()):
            raise ValueError("no screen-selected crop; stop before evaluation fitting")
    elif any(value is not None for value in (decision_raw, screen_manifest_raw, screen_report_raw)):
        raise ValueError("screen build cannot accept evaluation-selection artifacts")
    if diagnostic_raw is None: raise ValueError("pinned KAMIS availability diagnostic is required")
    loaded = _load_inputs(prices_raw, components_raw, phase, archive_dir)
    archive = loaded[5]
    if _sha(prices_raw) != PRICES_SHA256:
        raise ValueError("pinned auction price SHA-256 mismatch")
    expected_comp = SCREEN_COMPONENTS_SHA256 if phase == "screen" else EVALUATION_COMPONENTS_SHA256
    if _sha(components_raw) != expected_comp:
        raise ValueError("pinned component source SHA-256 mismatch")
    if _diagnostic_override is None:
        # Recompute each crop's gate from the validated original source rows before any fit.
        diagnostic_override = _validate_diagnostic(prices_raw, archive, diagnostic_raw)
    return _run_build(prices_raw, components_raw, phase, Path(archive_dir), diagnostic_raw,
                      selections=selection, replay_binding=replay_binding, diagnostic_override=diagnostic_override,
                      loaded=loaded,
                      progress=progress)


def validate_manifest_bytes(manifest_raw: bytes, prices_raw: bytes, components_raw: bytes, phase: str, *,
                            archive_dir=ARCHIVE_DIR, diagnostic_raw: bytes,
                            decision_raw: bytes | None = None, screen_manifest_raw: bytes | None = None,
                            screen_report_raw: bytes | None = None, _diagnostic_override=None) -> dict[str, Any]:
    return _validate_manifest_bytes_impl(manifest_raw, prices_raw, components_raw, phase,
        archive_dir=archive_dir, diagnostic_raw=diagnostic_raw, decision_raw=decision_raw,
        screen_manifest_raw=screen_manifest_raw, screen_report_raw=screen_report_raw,
        _diagnostic_override=_diagnostic_override)


def _validate_manifest_bytes_impl(manifest_raw: bytes, prices_raw: bytes, components_raw: bytes, phase: str, *,
                                  archive_dir=ARCHIVE_DIR, diagnostic_raw: bytes,
                                  decision_raw: bytes | None = None, screen_manifest_raw: bytes | None = None,
                                  screen_report_raw: bytes | None = None, _diagnostic_override=None,
                                  verified_replay_binding=None) -> dict[str, Any]:
    manifest = json.loads(manifest_raw)
    if (manifest.get("phase") != "forecast_only" or manifest.get("phaseName") != phase
            or manifest.get("targetValuesAttached") is not False):
        raise ValueError("KAMIS forecast manifest must be answer-free for requested phase")
    expected = _build_impl(prices_raw, components_raw, phase, archive_dir=archive_dir, diagnostic_raw=diagnostic_raw,
                     decision_raw=decision_raw, screen_manifest_raw=screen_manifest_raw,
                     screen_report_raw=screen_report_raw, _diagnostic_override=_diagnostic_override,
                     verified_replay_binding=verified_replay_binding)
    if manifest != expected: raise ValueError("KAMIS forecast manifest differs from full deterministic replay")
    if manifest_raw != _pretty(expected): raise ValueError("KAMIS forecast manifest is not canonical JSON")
    return manifest


def _prediction(row, field):
    item = row.get(field)
    return _positive(item.get("prediction")) if isinstance(item, dict) and item.get("status") == "issued" else None


def _key(row):
    crop, day, h = row.get("crop"), _date(row.get("target_date")), row.get("horizon")
    if crop not in CROPS or day is None or isinstance(h, bool) or h not in HORIZONS:
        raise ValueError("invalid KAMIS forecast key")
    return crop, day, h


def _actuals(prices, start, end):
    raw, _ = source._trim_prices(prices, end)
    series = source.narrow._series(raw)
    return {(crop, day): value for crop, rows in series.items() for day, value in rows.items()
            if start <= day <= end}


def _bound_prices(prices, prices_raw):
    if prices != json.loads(prices_raw):
        raise ValueError("parsed auction prices differ from the bound source bytes")
    return prices


def _bound_manifest(manifest, manifest_raw):
    if manifest != json.loads(manifest_raw):
        raise ValueError("parsed forecast manifest differs from the replay-bound source bytes")
    return manifest


def _daily_losses(rows, actuals, start, end):
    by = {}
    for row in rows:
        if row.get("eligible") is not True: continue
        crop, day, h = _key(row)
        if not start <= day <= end: continue
        by.setdefault((crop, day), {})[h] = row
    result = {crop: {"common": [], "byHorizon": {h: [] for h in HORIZONS}} for crop in CROPS}
    for crop in CROPS:
        for (c, day), horizon_rows in sorted(by.items()):
            if c != crop: continue
            for h, row in horizon_rows.items():
                actual = actuals.get((crop, day))
                values = {"fixed50": _prediction(row, "fixed50"), "blend106": _prediction(row, "blend106"),
                          "blend113": _prediction(row, "blend113")}
                if actual is not None and all(v is not None for v in values.values()):
                    result[crop]["byHorizon"][h].append({"date": day, "actual": actual, **values})
            complete = len(horizon_rows) == len(HORIZONS) and set(horizon_rows) == set(HORIZONS)
            if not complete: continue
            daily = {"date": day}
            for arm in ("fixed50", "blend106", "blend113"):
                errors, ape = [], []
                for h in HORIZONS:
                    row = horizon_rows[h]
                    actual = actuals.get((crop, day)); pred = _prediction(row, arm)
                    if actual is None or pred is None: break
                    errors.append(abs(pred - actual)); ape.append(abs(pred - actual) / actual)
                if len(errors) != 10: break
                daily[f"{arm}Ae"] = math.fsum(errors) / 10
                daily[f"{arm}Ape"] = math.fsum(ape) / 10
            if len(daily) == 7: result[crop]["common"].append(daily)
    return result


def _metrics(rows, arm):
    if not rows: return {"n": 0, "mae": None, "mape": None}
    errors = [abs(row[arm] - row["actual"]) for row in rows]
    return {"n": len(rows), "mae": math.fsum(errors) / len(rows),
            "mape": 100 * math.fsum(e / r["actual"] for e, r in zip(errors, rows)) / len(rows)}


def _holm20(rows):
    if len(rows) != 20: raise ValueError(f"fixed DM family must contain20 hypotheses, got {len(rows)}")
    stats._holm(rows)


def _score(manifest, prices, manifest_raw, prices_raw, components_raw, phase, *, active_crops=None):
    start, end, _ = _phase(phase)
    actuals = _actuals(prices, start, end)
    forecasts = manifest["forecasts"]
    daily = _daily_losses(forecasts, actuals, start, end)
    dm = []
    selected = manifest.get("selectedModels", {})
    for crop in CROPS:
        active = phase == "screen" or (active_crops is not None and crop in active_crops)
        dates = daily[crop]["common"]
        for comparison, a, b in COMPARISONS:
            for loss in LOSSES:
                for bandwidth in BANDWIDTHS:
                    if not active:
                        test = {"status": "not_selected", "reason": "crop_not_selected_by_frozen_screen", "n": 0,
                                "statistic": None, "meanLossDifference": None, "pValue": 1.0}
                    else:
                        diff = {row["date"]: row[f"{a}{'Ae' if loss == 'ae' else 'Ape'}"] - row[f"{b}{'Ae' if loss == 'ae' else 'Ape'}"] for row in dates}
                        test = stats._dm(diff, bandwidth)
                    dm.append({"crop": crop, "comparison": comparison, "loss": loss, "bandwidth": bandwidth, **test})
    for bandwidth in BANDWIDTHS: _holm20([r for r in dm if r["bandwidth"] == bandwidth])
    metrics, coverage, temporal = [], {}, {}
    gates = []
    for crop in CROPS:
        active = phase == "screen" or (active_crops is not None and crop in active_crops)
        triplets = daily[crop]["common"]
        macro = {arm: {"mae": math.fsum(r[f"{arm}Ae"] for r in triplets) / len(triplets) if triplets else None,
                       "mape": 100 * math.fsum(r[f"{arm}Ape"] for r in triplets) / len(triplets) if triplets else None,
                       "n": len(triplets)} for arm in ("fixed50", "blend106", "blend113")}
        for comparison, a, b in COMPARISONS:
            metrics.append({"crop": crop, "comparison": comparison, "unit": "mean10HorizonLossPerTargetDate",
                            "a": macro[a], "b": macro[b], "targetDateRange": [min((r["date"] for r in triplets), default=None).isoformat() if triplets else None,
                                max((r["date"] for r in triplets), default=None).isoformat() if triplets else None]})
        horizon_metrics = []
        for h in HORIZONS:
            pairrows = daily[crop]["byHorizon"][h]
            common = [r for r in pairrows if all(_positive(r.get(arm)) is not None for arm in ("fixed50", "blend106", "blend113"))]
            m = {arm: _metrics(common, arm) for arm in ("fixed50", "blend106", "blend113")}
            planned_cell = [r for r in forecasts if _key(r)[0] == crop and _key(r)[2] == h]
            eligible_cell = [r for r in planned_cell if r.get("eligible") is True]
            actual_cell = [r for r in eligible_cell if (crop, _key(r)[1]) in actuals]
            horizon_metrics.append({"horizon": h, "planned": len(planned_cell), "eligible": len(eligible_cell),
                "positiveActualPairs": len(actual_cell), "fixed50Issued": sum(_prediction(r, "fixed50") is not None for r in eligible_cell),
                "blend106Issued": sum(_prediction(r, "blend106") is not None for r in eligible_cell),
                "blend113Issued": sum(_prediction(r, "blend113") is not None for r in eligible_cell),
                "commonThreeWayPairs": len(common), "commonThreeWayUniqueTargets": len({r["date"] for r in common}), **m})
        eligible_rows = [r for r in forecasts if _key(r)[0] == crop and r.get("eligible") is True]
        lost106 = any(_prediction(r, "fixed50") is not None and _prediction(r, "blend106") is None for r in eligible_rows)
        lost113 = any(_prediction(r, "fixed50") is not None and _prediction(r, "blend113") is None for r in eligible_rows)
        no_horizon_harm = len(horizon_metrics) == 10 and all(
            item["blend113"]["mape"] is not None and item["fixed50"]["mape"] is not None and
            item["blend113"]["mape"] <= item["fixed50"]["mape"] + 1.0 for item in horizon_metrics)
        count_gate = len(triplets) >= 30
        macro_gate = (macro["blend113"]["mape"] is not None and macro["fixed50"]["mape"] is not None and
                      macro["blend113"]["mape"] <= macro["fixed50"]["mape"] - 0.5 and
                      macro["blend113"]["mae"] < macro["fixed50"]["mae"])
        joint_fixed = _joint_dm(dm, crop, "ridge113_vs_fixed50")
        pass_screen = all((count_gate, macro_gate, not lost106, not lost113, no_horizon_harm, joint_fixed))
        feature_effect = (macro["blend113"]["mae"] is not None and macro["blend106"]["mae"] is not None and
                          macro["blend113"]["mape"] is not None and macro["blend106"]["mape"] is not None and
                          macro["blend113"]["mae"] < macro["blend106"]["mae"] and
                          macro["blend113"]["mape"] < macro["blend106"]["mape"] and
                          _joint_dm(dm, crop, "ridge113_vs_ridge106"))
        selected_model = "ridge113" if phase == "screen" and pass_screen else selected.get(crop, "retain_fixed50")
        gates.append({"crop": crop, "candidate": "ridge113", "diagnosticEligible": manifest["diagnosticIdentity"]["perCropGates"][crop]["eligibleForLaterReview"],
            "commonCompleteTargetDates": len(triplets), "minimumTargetDates": 30, "coverageNoLoss": not (lost106 or lost113),
            "candidateLostFixed50Keys": lost113, "controlLostFixed50Keys": lost106,
            "candidateEqualDateMeanTenHorizonMae": macro["blend113"]["mae"], "fixed50EqualDateMeanTenHorizonMae": macro["fixed50"]["mae"],
            "candidateEqualDateMeanTenHorizonMape": macro["blend113"]["mape"], "fixed50EqualDateMeanTenHorizonMape": macro["fixed50"]["mape"],
            "mapeImprovementPctPoints": (macro["fixed50"]["mape"] - macro["blend113"]["mape"] if macro["fixed50"]["mape"] is not None and macro["blend113"]["mape"] is not None else None),
            "minimumGainAndMaeGate": macro_gate, "noHorizonWorseByMoreThan1PctPoint": no_horizon_harm,
            "fixed50JointAeApeBothHacImprovement": joint_fixed, "additionalFeatureBothLossesAndHacImprovedOver106": feature_effect,
            "screenPass": pass_screen if phase == "screen" else None,
            "evaluationGate": _evaluation_gate(dm, crop, macro, horizon_metrics, lost106 or lost113) if phase == "evaluation" and active else
                ("not_selected" if phase == "evaluation" else None), "screenSelection": selected_model})
        coverage[crop] = {"planned": sum(_key(r)[0] == crop for r in forecasts), "eligible": len(eligible_rows),
            "fixed50IssuedEligible": sum(_prediction(r, "fixed50") is not None for r in eligible_rows),
            "blend106IssuedEligible": sum(_prediction(r, "blend106") is not None for r in eligible_rows),
            "blend113IssuedEligible": sum(_prediction(r, "blend113") is not None for r in eligible_rows),
            "fixed50KeysLostBy106": sum(_prediction(r, "fixed50") is not None and _prediction(r, "blend106") is None for r in eligible_rows),
            "fixed50KeysLostBy113": sum(_prediction(r, "fixed50") is not None and _prediction(r, "blend113") is None for r in eligible_rows),
            "commonCompleteTargetDates": len(triplets), "horizonMetrics": horizon_metrics}
        temporal.update(_temporal_coverage(forecasts, actuals, start, end, crop))
    selection = {crop: ("ridge113" if phase == "screen" and next(g for g in gates if g["crop"] == crop)["screenPass"] else "retain_fixed50")
                 if phase == "screen" else selected.get(crop, "retain_fixed50") for crop in CROPS}
    report = {"schemaVersion": 1, "phase": f"{phase}_scored", "manifestSha256": _sha(manifest_raw),
        "pricesSha256": _sha(prices_raw), "componentsSha256": _sha(components_raw),
        "archivePlanSha256": manifest["inputSha256"]["archivePlan"],
        "archiveSourceManifestSha256": manifest["inputSha256"]["archiveSourceManifest"],
        "diagnosticSha256": manifest["inputSha256"]["diagnostic"],
        "scorerSha256": _sha(Path(__file__).read_bytes()), "metrics": metrics, "dm": dm,
        "dmSummaryByBandwidth": _dm_summary(dm), "coverage": coverage, "temporalCoverage": temporal,
        "candidateGates": gates, "selection": selection}
    if phase == "evaluation":
        report.update(_evaluation_policy(forecasts, actuals, selection, gates))
    return report


def _joint_dm(dm, crop, comparison):
    tests = {(r["loss"], r["bandwidth"]): r for r in dm if r["crop"] == crop and r["comparison"] == comparison}
    return all(tests.get((loss, bandwidth), {}).get("holmSignificant") and
               tests[(loss, bandwidth)].get("meanLossDifference") is not None and
               tests[(loss, bandwidth)]["meanLossDifference"] < 0 for loss in LOSSES for bandwidth in BANDWIDTHS)


def _evaluation_gate(dm, crop, macro, horizons, coverage_loss):
    # Same locked criterion as screening, applied only to previously selected crops.
    cand, base = macro["blend113"], macro["fixed50"]
    all30 = cand["n"] >= 30 and len(horizons) == 10 and all(r["blend113"]["n"] >= 30 for r in horizons)
    gain = cand["mape"] is not None and base["mape"] is not None and cand["mape"] <= base["mape"] - .5 and cand["mae"] < base["mae"]
    no_harm = len(horizons) == 10 and all(r["blend113"]["mape"] is not None and r["fixed50"]["mape"] is not None and r["blend113"]["mape"] <= r["fixed50"]["mape"] + 1 for r in horizons)
    return "selected_candidate_passes_2026_gate" if all((all30, gain, no_harm, not coverage_loss,
        _joint_dm(dm, crop, "ridge113_vs_fixed50"))) else "selected_candidate_fails_2026_gate_policy_retained"


def _periods(start, end):
    months, quarters = {}, {}
    day = start
    while day <= end:
        months.setdefault(day.strftime("%Y-%m"), set()).add(day)
        quarters.setdefault(f"{day.year}-Q{(day.month-1)//3+1}", set()).add(day)
        day += dt.timedelta(days=1)
    return {"month": months, "quarter": quarters}


def _temporal_coverage(rows, actuals, start, end, crop):
    output = {}
    for grain, periods in _periods(start, end).items():
        for label, days in periods.items():
            planned = [r for r in rows if _key(r)[0] == crop and _key(r)[1] in days]
            eligible = [r for r in planned if r.get("eligible") is True]
            key = f"{crop}/{grain}/{label}"
            output[key] = {"planned": len(planned), "eligible": len(eligible),
                "positiveActualPairs": sum((crop, _key(r)[1]) in actuals for r in eligible),
                "uniquePositiveActualTargets": len({ _key(r)[1] for r in eligible if (crop, _key(r)[1]) in actuals}),
                "commonThreeWayPairs": sum((crop, _key(r)[1]) in actuals and all(_prediction(r, arm) is not None
                    for arm in ("fixed50", "blend106", "blend113")) for r in eligible),
                "commonThreeWayUniqueTargets": len({_key(r)[1] for r in eligible if (crop, _key(r)[1]) in actuals
                    and all(_prediction(r, arm) is not None for arm in ("fixed50", "blend106", "blend113"))}),
                **{f"{arm}Issued": sum(_prediction(r, arm) is not None for r in eligible)
                   for arm in ("fixed50", "blend106", "blend113")}}
    return output


def _dm_summary(dm):
    out = []
    for bandwidth in BANDWIDTHS:
        family = [r for r in dm if r["bandwidth"] == bandwidth]
        out.append({"bandwidth": bandwidth, "hypotheses": len(family),
            "tested": sum(r["status"] == "tested" for r in family),
            "inferenceUnavailable": sum(r["status"] not in ("tested", "not_selected") for r in family),
            "notSelected": sum(r["status"] == "not_selected" for r in family),
            "holmSignificantImprovement": sum(r["status"] == "tested" and r.get("holmSignificant") and r["meanLossDifference"] < 0 for r in family),
            "holmSignificantWorsening": sum(r["status"] == "tested" and r.get("holmSignificant") and r["meanLossDifference"] > 0 for r in family),
            "inconclusive": sum(r["status"] == "tested" and not r.get("holmSignificant") for r in family)})
    return out


def _score_screen_validated(manifest, prices, manifest_raw, prices_raw, components_raw):
    return _score(manifest, prices, manifest_raw, prices_raw, components_raw, "screen")


def score_screen(manifest: dict[str, Any], prices: dict[str, Any], *, manifest_raw: bytes,
                 prices_raw: bytes, components_raw: bytes, diagnostic_raw: bytes,
                 archive_dir=ARCHIVE_DIR, _diagnostic_override=None):
    manifest = _bound_manifest(manifest, manifest_raw)
    prices = _bound_prices(prices, prices_raw)
    validate_manifest_bytes(manifest_raw, prices_raw, components_raw, "screen", archive_dir=archive_dir,
                            diagnostic_raw=diagnostic_raw, _diagnostic_override=_diagnostic_override)
    report = _score_screen_validated(manifest, prices, manifest_raw, prices_raw, components_raw)
    return report


def make_decision(manifest_raw: bytes, report_raw: bytes, prices_raw: bytes, components_raw: bytes,
                  diagnostic_raw: bytes, *, archive_dir=ARCHIVE_DIR, _diagnostic_override=None):
    manifest, report = json.loads(manifest_raw), json.loads(report_raw)
    recomputed = score_screen(manifest, json.loads(prices_raw), manifest_raw=manifest_raw, prices_raw=prices_raw,
                              components_raw=components_raw, diagnostic_raw=diagnostic_raw,
                              archive_dir=archive_dir, _diagnostic_override=_diagnostic_override)
    if _pretty(recomputed) != report_raw:
        raise ValueError("screen report differs from full replay and score")
    return _make_decision_from_validated(manifest_raw, report_raw, recomputed)


def _make_decision_from_validated(manifest_raw: bytes, report_raw: bytes,
                                  recomputed: dict[str, Any]) -> dict[str, Any]:
    """Construct a decision when this invocation has already replayed inputs."""
    unsigned = {"schemaVersion": 1, "phase": "selection_decision", "screenManifestSha256": _sha(manifest_raw),
        "screenReportSha256": _sha(report_raw), "scorerSha256": _sha(Path(__file__).read_bytes()),
        "selections": recomputed["selection"]}
    unsigned["decisionSha256"] = _sha(_canonical(unsigned))
    return unsigned


def validate_screen_artifacts(manifest_raw, prices_raw, components_raw, report_raw, decision_raw, *,
                              diagnostic_raw, archive_dir=ARCHIVE_DIR, _diagnostic_override=None):
    manifest, report, decision = json.loads(manifest_raw), json.loads(report_raw), json.loads(decision_raw)
    recomputed = score_screen(manifest, json.loads(prices_raw), manifest_raw=manifest_raw,
        prices_raw=prices_raw, components_raw=components_raw, diagnostic_raw=diagnostic_raw,
        archive_dir=archive_dir, _diagnostic_override=_diagnostic_override)
    if _pretty(recomputed) != report_raw: raise ValueError("screen report differs from complete deterministic replay/rescore")
    unsigned = {"schemaVersion": 1, "phase": "selection_decision", "screenManifestSha256": _sha(manifest_raw),
        "screenReportSha256": _sha(report_raw), "scorerSha256": _sha(Path(__file__).read_bytes()),
        "selections": recomputed["selection"]}
    expected = {**unsigned, "decisionSha256": _sha(_canonical(unsigned))}
    if decision != expected: raise ValueError("screen decision differs from replayed screen selection")
    return {"selection": expected["selections"], "manifestSha256": _sha(manifest_raw),
            "reportSha256": _sha(report_raw), "decisionSha256": expected["decisionSha256"]}


def _policy_rows(rows, actuals, selection, gates):
    result = []
    decision = {g["crop"]: g.get("evaluationGate") for g in gates}
    for row in rows:
        if not row.get("fixed50KeyPresent"): continue
        crop, day, h = _key(row)
        fixed, candidate = _prediction(row, "fixed50"), _prediction(row, "blend113") if selection.get(crop) == "ridge113" else None
        if fixed is None: continue
        fallback = selection.get(crop) != "ridge113" or candidate is None
        result.append({"crop": crop, "target_date": day.isoformat(), "origin_date": row["origin_date"], "horizon": h,
            "actual": actuals.get((crop, day)), "screenSelection": selection.get(crop, "retain_fixed50"),
            "evaluationGateDecision": decision.get(crop), "candidate": candidate, "fixed50": fixed,
            "policy": fixed if fallback else candidate, "fallback": fallback,
            "preperiodOriginFallback": bool(fallback and row.get("eligible") is not True),
            "candidateMissingReason": row["blend113"].get("reason") if selection.get(crop) == "ridge113" and candidate is None else None})
    return result


def _evaluation_policy(rows, actuals, selection, gates):
    policy = _policy_rows(rows, actuals, selection, gates)
    metrics, fallback = [], {"sourceFixed50Keys": len(policy), "positiveActualKeys": sum(r["actual"] is not None for r in policy),
        "policyIssued": sum(_positive(r["policy"]) is not None for r in policy), "fixed50Issued": len(policy),
        "fallbackKeys": sum(r["fallback"] for r in policy),
        "preperiodOriginFallbackKeys": sum(r["preperiodOriginFallback"] for r in policy),
        "eligibleCandidateFailureKeys": sum(r["fallback"] and not r["preperiodOriginFallback"] and r["screenSelection"] == "ridge113" for r in policy)}
    for crop in CROPS:
        for h in HORIZONS:
            subset = [r for r in policy if r["crop"] == crop and r["horizon"] == h and r["actual"] is not None]
            metrics.append({"crop": crop, "horizon": h, "sourceFixed50Keys": sum(r["crop"] == crop and r["horizon"] == h for r in policy),
                "policy": _metrics([{"actual": r["actual"], "policy": r["policy"]} for r in subset], "policy"),
                "fixed50": _metrics([{"actual": r["actual"], "fixed50": r["fixed50"]} for r in subset], "fixed50"),
                "fallbackCount": sum(r["fallback"] for r in policy if r["crop"] == crop and r["horizon"] == h),
                "preperiodOriginFallbackCount": sum(r["preperiodOriginFallback"] for r in policy if r["crop"] == crop and r["horizon"] == h),
                "eligibleCandidateFailureCount": sum(r["fallback"] and not r["preperiodOriginFallback"] and r["screenSelection"] == "ridge113" for r in policy if r["crop"] == crop and r["horizon"] == h)})
    included = [r for r in metrics if r["policy"]["n"] and r["fixed50"]["n"]]
    complete = len(included) == 50
    summary = {"cellsPossible": 50, "cellsIncluded": len(included), "complete50CellCoverage": complete,
        **{f"equalCell{arm}{metric.title()}": math.fsum(r[arm][metric] for r in included) / len(included) if included else None
           for arm in ("policy", "fixed50") for metric in ("mae", "mape")},
        **{f"all50Cell{arm}{metric.title()}": math.fsum(r[arm][metric] for r in included) / len(included) if complete else None
           for arm in ("policy", "fixed50") for metric in ("mae", "mape")},
        "inference": "descriptive_only_no_added_statistical_family"}
    return {"policyCoverage": fallback, "policyMetrics": metrics,
            "policyPortfolioEqualWeightAcross50Cells": summary, "policyRows": policy}


def score_evaluation(manifest: dict[str, Any], prices: dict[str, Any], *, manifest_raw: bytes,
                     prices_raw: bytes, components_raw: bytes, screen_manifest_raw: bytes,
                     screen_report_raw: bytes, decision_raw: bytes, diagnostic_raw: bytes,
                     archive_dir=ARCHIVE_DIR, _diagnostic_override=None):
    manifest = _bound_manifest(manifest, manifest_raw)
    prices = _bound_prices(prices, prices_raw)
    binding = validate_screen_artifacts(screen_manifest_raw, prices_raw, SCREEN_COMPONENTS_PATH.read_bytes(),
        screen_report_raw, decision_raw, diagnostic_raw=diagnostic_raw, archive_dir=archive_dir,
        _diagnostic_override=_diagnostic_override)
    replayed = _validate_manifest_bytes_impl(manifest_raw, prices_raw, components_raw, "evaluation", archive_dir=archive_dir,
        diagnostic_raw=diagnostic_raw, decision_raw=decision_raw, screen_manifest_raw=screen_manifest_raw,
        screen_report_raw=screen_report_raw, _diagnostic_override=_diagnostic_override,
        verified_replay_binding=binding)
    selected = {crop for crop, arm in binding["selection"].items() if arm == "ridge113"}
    start, end, _ = _phase("evaluation")
    actuals = _actuals(prices, start, end)
    scored = _score(replayed, prices, manifest_raw, prices_raw, components_raw, "evaluation", active_crops=selected)
    return {**scored, "screenManifestSha256": binding["manifestSha256"], "screenReportSha256": binding["reportSha256"],
            "screenDecisionSha256": binding["decisionSha256"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    build_p = subs.add_parser("build")
    build_p.add_argument("phase", choices=("screen", "evaluation")); build_p.add_argument("output", type=Path)
    build_p.add_argument("--prices", type=Path, default=PRICES_PATH); build_p.add_argument("--components", type=Path)
    build_p.add_argument("--archive-dir", type=Path, default=ARCHIVE_DIR); build_p.add_argument("--diagnostic", type=Path, default=DIAGNOSTIC_PATH)
    build_p.add_argument("--decision", type=Path); build_p.add_argument("--screen-manifest", type=Path); build_p.add_argument("--screen-report", type=Path)
    ss = subs.add_parser("score-screen")
    for name in ("manifest", "report", "decision"): ss.add_argument(name, type=Path)
    ss.add_argument("--prices", type=Path, default=PRICES_PATH); ss.add_argument("--components", type=Path, default=SCREEN_COMPONENTS_PATH)
    ss.add_argument("--archive-dir", type=Path, default=ARCHIVE_DIR); ss.add_argument("--diagnostic", type=Path, default=DIAGNOSTIC_PATH)
    se = subs.add_parser("score-evaluation")
    for name in ("manifest", "screen_manifest", "screen_report", "decision", "report"): se.add_argument(name, type=Path)
    se.add_argument("--prices", type=Path, default=PRICES_PATH); se.add_argument("--components", type=Path, default=EVALUATION_COMPONENTS_PATH)
    se.add_argument("--archive-dir", type=Path, default=ARCHIVE_DIR)
    se.add_argument("--diagnostic", type=Path, default=DIAGNOSTIC_PATH)
    args = parser.parse_args(argv)
    if args.command == "build":
        comp = args.components or (SCREEN_COMPONENTS_PATH if args.phase == "screen" else EVALUATION_COMPONENTS_PATH)
        diagnostic = args.diagnostic.read_bytes()
        manifest = build(args.prices.read_bytes(), comp.read_bytes(), args.phase, archive_dir=args.archive_dir,
            diagnostic_raw=diagnostic, decision_raw=args.decision.read_bytes() if args.decision else None,
            screen_manifest_raw=args.screen_manifest.read_bytes() if args.screen_manifest else None,
            screen_report_raw=args.screen_report.read_bytes() if args.screen_report else None,
            progress=lambda e: print(json.dumps(e, sort_keys=True), file=sys.stderr))
        _write_exclusive(args.output, manifest)
    elif args.command == "score-screen":
        mr, pr, cr, dr = args.manifest.read_bytes(), args.prices.read_bytes(), args.components.read_bytes(), args.diagnostic.read_bytes()
        manifest, prices = json.loads(mr), json.loads(pr)
        report = score_screen(manifest, prices, manifest_raw=mr, prices_raw=pr, components_raw=cr,
                              diagnostic_raw=dr, archive_dir=args.archive_dir)
        rr = _pretty(report); decision = _make_decision_from_validated(mr, rr, report)
        _write_exclusive(args.report, report); _write_exclusive(args.decision, decision)
    else:
        mr, pr, cr = args.manifest.read_bytes(), args.prices.read_bytes(), args.components.read_bytes()
        sm, sr, sd, dr = args.screen_manifest.read_bytes(), args.screen_report.read_bytes(), args.decision.read_bytes(), args.diagnostic.read_bytes()
        report = score_evaluation(json.loads(mr), json.loads(pr), manifest_raw=mr, prices_raw=pr, components_raw=cr,
            screen_manifest_raw=sm, screen_report_raw=sr, decision_raw=sd, diagnostic_raw=dr, archive_dir=args.archive_dir)
        _write_exclusive(args.report, report)
    return 0


if __name__ == "__main__": raise SystemExit(main())
