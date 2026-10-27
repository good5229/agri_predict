"""Frozen publication-scenario supply overlay for V5 G-B (exploratory only).

The module matches directly reviewed Phase-A/evaluation facts at the contract
origin time, then makes paired D0 (availability) and D1 (availability + typed
numeric supply values) online residual forecasts. It never changes V5 files or
claims strict historical-vintage evidence for today's captured documents.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.linear_model import Ridge
from . import period_proxy_experiment as proxy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/period-supply-publication-scenario-v8-20261016"
PROTOCOL = OUT / "protocol.md"
PHASE_A = ROOT / "outputs/experiments/period-supply-document-signal-v8/phase-A/reviewed-final-preclassification-fix/facts.parquet"
EVAL = ROOT / "outputs/experiments/period-supply-document-signal-v8/eval-audit-20261016"
V5 = ROOT / "outputs/experiments/period-pooled-direct-lgbm-v5-20261016"
PRICE_SOURCE = proxy.SOURCE
ALPHA = 10.0
MIN_ORIGINS = 12
MIN_UPDATE_GROUPS = 3
MAX_FITS = 2000
CORRECTION_CAP = math.log(1.25)
TZ = dt.timezone(dt.timedelta(hours=9))
PROTOCOL_SHA = "ae116887854bf3c84dd5c5f9d21b2828a04db362f75bc23906fc2cd510451af8"
EXPECTED_SOURCE_SHA = {
    "phaseAFacts": "83007a6c7fc656149c6593e4c44982cda2584e08987ab320f9ba966a0b8130cb",
    "evalReviewedFacts": "4428343ec089fe19146341336f200458590bf3a81e1a8ace5510ffb79d6934f8",
    "developmentForecasts": "26a1a7192dd0b42f43794301e2cd9758c95b4a86138ebccc497763e7f4ca3115",
    "evaluationForecasts": "1a9847d08121aee8d3d808961d116bf5abf0151f74957a1b49639e1d3c9ac0c7",
    "priceTargetSource": "6ca1366d1dda618c96f48e62fea578193696b24dc893e68320bdeaae8e43e659",
    "priceProxyCode": "b6896eac8c43f1f324673e1a1251bd080469c517a80e99d051ceb50c7617fc1c",
    "priceProxyProtocol": "d4a8b94fe6a056e22fefc3f44576018e2fd049397165ea6f856b7b10e2b4a3d4",
}
EXPECTED_PLANNED_ROWS = 1368
EXPECTED_CELLS = {(crop, h) for crop in ("cabbage", "radish", "onion", "garlic") for h in (2, 3)}
SCENARIO_ELIGIBLE = {"retrospective_publication_scenario", "strict_candidate"}
FORBIDDEN_ORIGIN = {"explicit_krei_citation", "explicit_krei_original", "krei_derived_fact", "origin_source_hold"}
FORBIDDEN_SERIES = {"historical_comparator_not_current_supply", "reiteration_not_proven_new_measurement"}
CELL_CROPS = {"cabbage", "radish"}
HORIZONS = {2, 3}
SOURCE_ALIASES = {"농림축산식품부": "MAFRA", "MAFRA": "MAFRA"}
# Quantity labels are intentionally not merged: observed release, planned
# subset, additional release, reserve-in-progress, reserve, and capacity are
# different states/scopes even when their units match.
QUANTITY_KIND_ALIASES: dict[str, str] = {}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _date(value: Any) -> dt.date | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _scenario_time(origin: Any) -> dt.datetime:
    day = _date(origin)
    if day is None:
        raise ValueError("origin is required")
    return dt.datetime.combine(day, dt.time(10, 0), TZ)


def _published_at(value: Any) -> dt.datetime | None:
    if value is None or value == "":
        return None
    if isinstance(value, dt.datetime):
        parsed = value
    else:
        s = str(value).strip()
        if len(s) == 10:
            parsed = dt.datetime.combine(dt.date.fromisoformat(s) + dt.timedelta(days=1), dt.time.min)
        else:
            try:
                parsed = dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
            except ValueError:
                parsed = dt.datetime.combine(dt.date.fromisoformat(s[:10]) + dt.timedelta(days=1), dt.time.min)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=TZ)
    return parsed.astimezone(TZ)


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _is_krei(row: dict[str, Any]) -> bool:
    source = str(row.get("source_org", row.get("sourceOrg", "")) or "").strip().lower()
    origin = str(row.get("provenance_origin", row.get("provenanceOrigin", "")) or "").strip().lower()
    review = str(row.get("origin_review_status", row.get("originReviewStatus", "")) or "").strip().lower()
    known_sources = {"krei", "한국농촌경제연구원", "농업관측센터"}
    positive_origins = {"krei_original", "krei_derived_fact", "explicit_krei_citation", "explicit_krei_original"}
    return source in known_sources or origin in positive_origins or review in positive_origins


def fact_exclusion_reason(row: dict[str, Any]) -> str | None:
    """Apply the frozen reviewed-fact candidate gate without dropping audit provenance."""
    eligible = row.get("asof_eligibility", row.get("asofEligibility"))
    if eligible not in SCENARIO_ELIGIBLE:
        state = str(eligible or "").lower()
        if "pending" in state:
            return "pending_asof"
        if state == "ineligible":
            return "asof_ineligible"
        return "unknown_asof_status"
    if _is_krei(row):
        return "positive_krei_source_or_attribution"
    origin_status = str(row.get("origin_review_status", row.get("originReviewStatus")) or "").lower()
    if origin_status in FORBIDDEN_ORIGIN or "hold" in origin_status or "pending" in origin_status:
        return "origin_hold_or_pending"
    series = str(row.get("series_admissibility", row.get("seriesAdmissibility")) or "")
    if series.startswith("hold_") or series in FORBIDDEN_SERIES:
        return "series_hold_or_historical_comparator"
    if str(row.get("evidence_status", row.get("evidenceStatus")) or "").lower() in {"pending", "pending_source_review"}:
        return "pending_evidence"
    if str(row.get("target_mapping_status", row.get("targetMappingStatus")) or "").lower() in {"pending", "unmapped", "out_of_scope"}:
        return "pending_or_unmapped_target"
    if str(row.get("horizon_relevance", row.get("horizonRelevance")) or "").lower() == "historical_comparator_not_current_supply":
        return "historical_comparator"
    if _published_at(row.get("publish_eligible_at_scenario", row.get("publishEligibleAtScenario"))) is None:
        return "missing_publication_boundary"
    return None


def fact_is_eligible(row: dict[str, Any]) -> bool:
    return fact_exclusion_reason(row) is None


def fact_id(row: dict[str, Any]) -> str:
    return str(row.get("fact_id", row.get("factId", "")))


def fact_group_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Typed series key; separate source, geography, crop type, quantity and unit."""
    crop = str(row.get("crop") or "").strip()
    source = str(row.get("source_org", row.get("sourceOrg")) or "").strip()
    source = SOURCE_ALIASES.get(source, source)
    region = str(row.get("region") or "").strip()
    quantity = str(row.get("quantity_kind", row.get("quantityKind")) or "").strip()
    quantity = QUANTITY_KIND_ALIASES.get(quantity, quantity)
    unit = str(row.get("unit") or "").strip()
    variety = str(row.get("variety_or_crop_type", row.get("varietyOrCropType")) or "").strip()
    return (crop, source, region, quantity, unit, variety)


def _fact_value(row: dict[str, Any], snake: str, camel: str) -> Any:
    return row.get(snake, row.get(camel))


def fact_match_state(row: dict[str, Any], forecast: dict[str, Any]) -> dict[str, Any] | None:
    """Return point-in-time matching state at origin 10:00 KST, never price cutoff."""
    if not fact_is_eligible(row):
        return None
    if str(_fact_value(row, "crop", "crop") or "") != str(forecast.get("crop") or ""):
        return None
    eligible_at = _published_at(_fact_value(row, "publish_eligible_at_scenario", "publishEligibleAtScenario"))
    issued_at = _scenario_time(forecast.get("origin"))
    if eligible_at is None or eligible_at > issued_at:
        return None
    age = (issued_at - eligible_at).total_seconds() / 86400.0
    effective_start = _date(_fact_value(row, "effective_start", "effectiveStart"))
    effective_end = _date(_fact_value(row, "effective_end", "effectiveEnd"))
    dated_plan = effective_start is not None and effective_end is not None
    plan_like = any(token in str(_fact_value(row, "quantity_kind", "quantityKind") or "").lower()
                    for token in ("planned", "plan", "reserve", "release", "capacity", "contract"))
    target_start, target_end = _date(forecast.get("targetStart")), _date(forecast.get("targetEnd"))
    if target_start is None or target_end is None:
        raise ValueError("targetStart and targetEnd are required")
    if dated_plan:
        if issued_at.date() > effective_end:
            return None
        overlaps = target_start <= effective_end and target_end >= effective_start
        overlap_state = "known_overlap" if overlaps else "known_nonoverlap"
        validity = "dated_plan"
    else:
        if age > 30.0:
            return None
        overlap_state = "unknown"
        validity = "unknown" if plan_like else "not_applicable_unknown_period"
    return {
        "fact": row,
        "eligibleAt": eligible_at.isoformat(),
        "scenarioAt": issued_at.isoformat(),
        "ageDays": age,
        "validity": validity,
        "overlap": overlap_state,
        "effectiveStart": effective_start.isoformat() if effective_start else None,
        "effectiveEnd": effective_end.isoformat() if effective_end else None,
        "numeric": {
            "value": _number(_fact_value(row, "value", "value")),
            "lower": _number(_fact_value(row, "value_lower", "valueLower")),
            "upper": _number(_fact_value(row, "value_upper", "valueUpper")),
        },
    }


def select_facts(facts: Iterable[dict[str, Any]], forecast: dict[str, Any]) -> list[dict[str, Any]]:
    """Select latest eligible state by typed series; conflicting same-time rows remain explicit."""
    eligible = [s for fact in facts if (s := fact_match_state(fact, forecast)) is not None]
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for state in eligible:
        groups.setdefault(fact_group_key(state["fact"]), []).append(state)
    selected = []
    for key, states in sorted(groups.items()):
        latest = max(s["eligibleAt"] for s in states)
        tied = [s for s in states if s["eligibleAt"] == latest]
        signatures = {tuple(s["numeric"].get(k) for k in ("value", "lower", "upper")) for s in tied}
        if len(signatures) > 1:
            # Do not choose a winner among contradictory same-vintage facts.
            for state in tied:
                selected.append({**state, "tieConflict": True})
            continue
        chosen = min(tied, key=lambda s: fact_id(s["fact"]))
        selected.append({**chosen, "tieConflict": False,
                         "duplicateFactIds": sorted(fact_id(s["fact"]) for s in tied)})
    return sorted(selected, key=lambda s: fact_group_key(s["fact"]))


def _feature_key(key: tuple[str, ...]) -> str:
    return "|".join(part if part else "_" for part in key)


def feature_schema(facts: Iterable[dict[str, Any]]) -> tuple[list[str], list[str]]:
    groups = sorted({_feature_key(fact_group_key(f)) for f in facts if fact_is_eligible(f)})
    masks = []
    numeric = []
    for group in groups:
        masks.extend((f"supply::{group}::present", f"supply::{group}::age_days",
                      f"supply::{group}::dated_plan", f"supply::{group}::unknown_validity",
                      f"supply::{group}::known_overlap", f"supply::{group}::known_nonoverlap",
                      f"supply::{group}::tie_conflict"))
        numeric.extend(f"value::{group}::{bound}" for bound in ("value", "lower", "upper"))
    return masks, numeric


def row_features(forecast: dict[str, Any], selected: list[dict[str, Any]], masks: list[str], numeric: list[str]) -> tuple[dict[str, float], dict[str, float | None]]:
    """Construct D0 and D1 columns using a frozen typed feature map."""
    d0 = {name: 0.0 for name in masks}
    d1 = {name: None for name in numeric}
    by_group: dict[str, dict[str, Any]] = {}
    for state in selected:
        group = _feature_key(fact_group_key(state["fact"]))
        if group not in by_group or state.get("tieConflict"):
            by_group[group] = state
    for group, state in by_group.items():
        values = {
            "present": 1.0,
            "age_days": float(state["ageDays"]),
            "dated_plan": float(state["validity"] == "dated_plan"),
            "unknown_validity": float(state["validity"] in {"unknown", "not_applicable_unknown_period"}),
            "known_overlap": float(state["overlap"] == "known_overlap"),
            "known_nonoverlap": float(state["overlap"] == "known_nonoverlap"),
            "tie_conflict": float(bool(state.get("tieConflict"))),
        }
        for suffix, value in values.items():
            name = f"supply::{group}::{suffix}"
            if name in d0:
                d0[name] = value
        for bound, value in ({} if state.get("tieConflict") else state["numeric"]).items():
            name = f"value::{group}::{bound}"
            if name in d1:
                d1[name] = value
    start = _date(forecast.get("targetStart"))
    dekad = (start.month - 1) * 3 + (1 if start.day <= 10 else 2 if start.day <= 20 else 3)
    angle = 2 * math.pi * (dekad - 1) / 36.0
    base = _number(forecast.get("base"))
    pref = _number(forecast.get("p_ref"))
    d0.update({"calendar::dekad_sin": math.sin(angle), "calendar::dekad_cos": math.cos(angle),
               "price::log_base_over_pref": math.log(base / pref) if base and pref and base > 0 and pref > 0 else 0.0,
               "price::ratio_missing": float(not (base and pref and base > 0 and pref > 0))})
    # Keep the two explicit baseline/calendar terms in D0 and in D1.
    return d0, d1


def _scaled_matrix(rows: list[dict[str, Any]], names: list[str], *, fit: bool,
                   means: np.ndarray | None = None, scales: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray([[np.nan if r.get("features", {}).get(n) is None else float(r["features"][n]) for n in names]
                      for r in rows], dtype=float)
    if fit:
        means = np.nanmean(raw, axis=0) if raw.shape[0] else np.zeros(len(names))
        means = np.where(np.isfinite(means), means, 0.0)
        filled = np.where(np.isfinite(raw), raw, means)
        scales = np.std(filled, axis=0, ddof=0)
        scales = np.where((scales > 0) & np.isfinite(scales), scales, 1.0)
    if means is None or scales is None:
        raise ValueError("training scaler missing")
    filled = np.where(np.isfinite(raw), raw, means)
    return (filled - means) / scales, means, scales


def _origin_day(row: dict[str, Any]) -> dt.date:
    return _date(row["origin"])


def _mature_at(training_row: dict[str, Any], fit_origin: dt.date) -> bool:
    end = _date(training_row.get("targetEnd"))
    return end is not None and end + dt.timedelta(days=2) <= fit_origin


def _has_numeric(state: dict[str, Any]) -> bool:
    return any(v is not None for s in state.get("selectedFacts", []) if not s.get("tieConflict") for v in s["numeric"].values())


def _features_from_row(row: dict[str, Any], masks: list[str], numeric: list[str], d1: bool) -> dict[str, float | None]:
    d0, values = row_features(row, row.get("selectedFacts", []), masks, numeric)
    result: dict[str, float | None] = dict(d0)
    result.update({f"missing::{k}": float(v is None) for k, v in values.items()})
    if d1:
        result.update(values)
    return result


def _forecast_one(row: dict[str, Any], all_rows: list[dict[str, Any]], masks: list[str], numeric: list[str],
                  counters: dict[str, int]) -> dict[str, Any]:
    base = _number(row.get("base"))
    output = {k: row.get(k) for k in ("crop", "horizon", "targetStart", "targetEnd", "targetKey", "origin", "inputCutoff", "p_ref", "status")}
    output.update({"base": base, "d0": base, "d1": base, "d0Status": "base_fallback", "d1Status": "base_fallback",
                   "fallbackReason": None, "selectedFactIds": [fact_id(s["fact"]) for s in row["selectedFacts"]],
                   "selectedFeatureGroups": sorted({_feature_key(fact_group_key(s["fact"])) for s in row["selectedFacts"]}),
                   "selectedUpdateGroups": sorted({str(s["fact"].get("update_group_id", s["fact"].get("updateGroupId", ""))) for s in row["selectedFacts"] if s["fact"].get("update_group_id", s["fact"].get("updateGroupId"))}),
                   "trainingOrigins": 0, "trainingUpdateGroups": 0, "trainingRowKeys": []})
    output.update({"d0CorrectionLog": None, "d1CorrectionLog": None, "d1NumericNeutral": base,
                   "d1NuisanceRefitDeltaLog": None, "d1NumericValueContributionLog": None})
    output["d0TrainingRowKeys"] = []
    output["d1TrainingRowKeys"] = []
    if row.get("crop") not in CELL_CROPS or row.get("horizon") not in HORIZONS:
        output["fallbackReason"] = "outside_supply_scenario_cells"
        return output
    if base is None or base <= 0:
        output["fallbackReason"] = "baseline_unavailable"
        return output
    if not row["selectedFacts"]:
        output["fallbackReason"] = "no_fresh_or_valid_supply_fact"
        return output
    if not _has_numeric(row):
        output["fallbackReason"] = "matched_facts_without_numeric_values"
        return output
    origin = _origin_day(row)
    train = []
    for candidate in all_rows:
        if candidate.get("crop") != row.get("crop") or candidate.get("horizon") != row.get("horizon"):
            continue
        if _origin_day(candidate) >= origin or not candidate.get("selectedFacts") or not _mature_at(candidate, origin):
            continue
        target = _number(candidate.get("actual"))
        cand_base = _number(candidate.get("base"))
        if target is None or cand_base is None or target <= 0 or cand_base <= 0:
            continue
        if not _has_numeric(candidate):
            continue
        train.append(candidate)
    update_groups = {str(group) for candidate in train for group in candidate["updateGroups"] if group}
    unique_origins = {_origin_day(candidate) for candidate in train}
    output["trainingOrigins"] = len(unique_origins)
    output["trainingUpdateGroups"] = len(update_groups)
    output["trainingRowKeys"] = [candidate["rowKey"] for candidate in train]
    output["d0TrainingRowKeys"] = list(output["trainingRowKeys"])
    output["d1TrainingRowKeys"] = list(output["trainingRowKeys"])
    current_numeric_groups = {_feature_key(fact_group_key(s["fact"])) for s in row["selectedFacts"]
                              if not s.get("tieConflict") and any(v is not None for v in s["numeric"].values())}
    all_numeric_groups = sorted({name.split("::", 2)[1] for name in numeric if name.startswith("value::")})
    supported_numeric_columns: set[str] = set()
    for group in all_numeric_groups:
        for bound in ("value", "lower", "upper"):
            name = f"value::{group}::{bound}"
            observed = [_number(c["featureRows"]["d1"].get(name)) for c in train]
            observed = [v for v in observed if v is not None]
            if len(observed) >= 2 and float(np.std(observed, ddof=0)) > 0.0:
                supported_numeric_columns.add(name)
    supported_numeric_groups = {name.split("::", 2)[1] for name in supported_numeric_columns}
    output["trainingNonconstantNumericGroups"] = len(supported_numeric_groups)
    output["trainingNonconstantNumericGroupKeys"] = sorted(supported_numeric_groups)
    output["trainingNonconstantNumericColumns"] = sorted(supported_numeric_columns)
    output["currentNumericGroups"] = sorted(current_numeric_groups)
    current_numeric_columns = {f"value::{group}::{bound}" for group in current_numeric_groups for bound in ("value", "lower", "upper")
                               if any(s["numeric"].get(bound) is not None and _feature_key(fact_group_key(s["fact"])) == group
                                      for s in row["selectedFacts"] if not s.get("tieConflict"))}
    output["supportedCurrentNumericColumns"] = sorted(current_numeric_columns & supported_numeric_columns)
    output["unsupportedCurrentNumericColumns"] = sorted(current_numeric_columns - set(output["supportedCurrentNumericColumns"]))
    output["currentQuantityKindInventory"] = sorted({str(s["fact"].get("quantity_kind", s["fact"].get("quantityKind", "")) or "")
        for s in row["selectedFacts"]})
    if len(unique_origins) < MIN_ORIGINS or len(update_groups) < MIN_UPDATE_GROUPS:
        output["fallbackReason"] = "minimum_mature_origin_or_update_group_support"
        return output
    if counters["fits"] + 2 > MAX_FITS:
        raise RuntimeError("frozen maximum fit attempt count exceeded")
    counters["fits"] += 2
    for variant in ("d0", "d1"):
        feature_names = row["featureNames"][variant]
        train_variant = []
        for candidate in train:
            candidate["features"] = candidate["featureRows"][variant]
            train_variant.append(candidate)
        row["features"] = row["featureRows"][variant]
        x_train, means, scales = _scaled_matrix(train_variant, feature_names, fit=True)
        x_current, _, _ = _scaled_matrix([row], feature_names, fit=False, means=means, scales=scales)
        y = np.asarray([math.log(float(c["actual"]) / float(c["base"])) for c in train], dtype=float)
        model = Ridge(alpha=ALPHA, fit_intercept=True, solver="lsqr")
        model.fit(x_train, y)
        residual_raw = float(model.predict(x_current)[0])
        residual = float(np.clip(residual_raw, -CORRECTION_CAP, CORRECTION_CAP))
        output[variant] = float(base * math.exp(residual))
        output[f"{variant}Status"] = "ridge_prequential"
        output[f"{variant}CorrectionLog"] = residual
        if variant == "d1":
            neutral = dict(row["features"])
            # Hold all D0 columns and numeric missing masks fixed. Replacing only
            # typed values by their train-only means isolates current-value effect.
            for idx, name in enumerate(feature_names):
                if name.startswith("value::"):
                    neutral[name] = float(means[idx])
            neutral_row = {"features": neutral}
            x_neutral, _, _ = _scaled_matrix([neutral_row], feature_names, fit=False, means=means, scales=scales)
            neutral_raw = float(model.predict(x_neutral)[0])
            neutral_correction = float(np.clip(neutral_raw, -CORRECTION_CAP, CORRECTION_CAP))
            output["d1NumericNeutral"] = float(base * math.exp(neutral_correction))
            output["d1NuisanceRefitDeltaLog"] = neutral_correction - float(output["d0CorrectionLog"])
            output["d1NumericValueContributionLog"] = residual - neutral_correction
    return output


def _read_facts() -> list[dict[str, Any]]:
    import pandas as pd
    phase = [{**r, "_inputCohort": "phaseA"} for r in pd.read_parquet(PHASE_A).to_dict("records")]
    eval_rows: list[dict[str, Any]] = []
    for name in ("reviewed-facts.json",):
        raw = json.loads((EVAL / name).read_text(encoding="utf-8"))
        eval_rows.extend({**r, "_inputCohort": "evaluation"} for r in (raw if isinstance(raw, list) else raw.get("facts", [])))
    # The normalization and the aggregate manual file are alternate views of the
    # same 41 documents. Use the canonical reviewed-facts list once only.
    by_id: dict[str, dict[str, Any]] = {}
    for row in [*phase, *eval_rows]:
        fid = fact_id(row)
        if fid:
            if fid in by_id and by_id[fid] != row:
                raise ValueError(f"conflicting duplicate fact id: {fid}")
            by_id[fid] = row
    return list(by_id.values())


def _read_forecasts() -> list[dict[str, Any]]:
    import pandas as pd
    _, raw = proxy.validate_inputs()
    series = proxy.series_from_raw(raw)
    target_catalog = {crop: proxy.aggregate_targets(values) for crop, values in series.items()}
    forecast_rows = []
    keys_seen: set[tuple[str, int, str, str, str]] = set()
    for split, path in (("development", V5 / "development-forecasts.parquet"), ("evaluation", V5 / "forecasts.parquet")):
        for r in pd.read_parquet(path).to_dict("records"):
            key = (str(r["crop"]), int(r["horizon"]), str(r["targetStart"]), str(r["targetEnd"]), str(r["origin"]))
            if key in keys_seen:
                raise ValueError(f"duplicate frozen V5 row key: {key}")
            keys_seen.add(key)
            target = (dt.date.fromisoformat(key[2]), dt.date.fromisoformat(key[3]))
            label = target_catalog.get(key[0], {}).get(target)
            if label is None:
                raise ValueError(f"frozen price snapshot has no target row for V5 key: {key}")
            pred = (r.get("predictions") or {}).get("G-B") or {}
            base = _v5_gb_prediction(r)
            unavailable = _v5_gb_unavailable(r)
            row = {"split": split, "crop": key[0], "horizon": key[1], "targetStart": key[2], "targetEnd": key[3],
                   "targetKey": str(r.get("targetKey", key[2])), "origin": key[4], "inputCutoff": str(r.get("inputCutoff")),
                   "p_ref": _number(r.get("p_ref")), "base": base,
                   "actual": _number(label.get("actual")), "targetObservedDays": label.get("observedDays"),
                   "targetStatus": label.get("targetStatus"),
                   "status": pred.get("status") or ("policy_unavailable" if unavailable else "missing_baseline"),
                   "rowKey": "|".join(map(str, key))}
            forecast_rows.append(row)
    cells = {(r["crop"], r["horizon"]) for r in forecast_rows}
    if len(forecast_rows) != EXPECTED_PLANNED_ROWS or cells != EXPECTED_CELLS:
        raise ValueError(f"incomplete frozen forecast grid: rows={len(forecast_rows)} cells={sorted(cells)}")
    return forecast_rows


def _v5_gb_prediction(row: dict[str, Any]) -> float | None:
    pred = (row.get("predictions") or {}).get("G-B") or {}
    unavailable = _v5_gb_unavailable(row)
    return _number(pred.get("predicted")) if pred.get("status") in {"ok", "available"} and not unavailable else None


def _v5_gb_unavailable(row: dict[str, Any]) -> bool:
    unavailable_raw = row.get("policyUnavailable")
    return bool(unavailable_raw.get("G-B")) if isinstance(unavailable_raw, dict) else bool(unavailable_raw)


def _hashes() -> dict[str, str]:
    paths = {"protocol": PROTOCOL, "phaseAFacts": PHASE_A, "evalReviewedFacts": EVAL / "reviewed-facts.json",
             "developmentForecasts": V5 / "development-forecasts.parquet", "evaluationForecasts": V5 / "forecasts.parquet",
             "priceTargetSource": PRICE_SOURCE, "priceProxyCode": Path(proxy.__file__), "priceProxyProtocol": proxy.PROTOCOL}
    hashes = {name: sha256(path) for name, path in paths.items()}
    if hashes["protocol"] != PROTOCOL_SHA:
        raise ValueError("publication scenario protocol differs from its frozen digest")
    for name, expected in EXPECTED_SOURCE_SHA.items():
        if hashes.get(name) != expected:
            raise ValueError(f"pinned V5/V8 source differs: {name}")
    return hashes


def _csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def run_forecast(out: Path = OUT) -> dict[str, Any]:
    """Fit the frozen exploratory overlay and persist labels separately from predictions."""
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    # Confirm the V5 daily snapshot/protocol and all immutable source identities
    # before writing the prefit manifest or fitting any candidate.
    proxy.validate_inputs()
    facts = _read_facts()
    masks, numeric = feature_schema(facts)
    raw_groups: dict[tuple[str, ...], set[str]] = {}
    for f in facts:
        if fact_is_eligible(f):
            raw_groups.setdefault(fact_group_key(f), set()).add(str(f.get("source_org", f.get("sourceOrg")) or ""))
    group_details = [{"typedGroup": list(key), "sourceLabelAliases": sorted(raw_groups[key])} for key in sorted(raw_groups)]
    cohort_support = {}
    for cohort in ("phaseA", "evaluation"):
        cohort_facts = [f for f in facts if f.get("_inputCohort") == cohort]
        cohort_eligible = [f for f in cohort_facts if fact_is_eligible(f)]
        cohort_support[cohort] = {"inputFacts": len(cohort_facts), "eligibleFacts": len(cohort_eligible),
            "eligibleTypedGroups": len({fact_group_key(f) for f in cohort_eligible})}
    excluded_facts = [{"factId": fact_id(f), "cohort": f.get("_inputCohort"),
                       "reason": fact_exclusion_reason(f)} for f in facts if not fact_is_eligible(f)]
    negated_krei_retained = [fact_id(f) for f in facts if fact_is_eligible(f)
        and "no krei attribution" in str(f.get("provenance_origin", f.get("provenanceOrigin", ""))).lower()]
    schema_payload = {"availabilityColumns": masks, "numericColumns": numeric,
        "typedGroups": group_details, "sourceAliases": SOURCE_ALIASES,
        "quantityKindAliases": QUANTITY_KIND_ALIASES,
        "quantityKindInventory": sorted({str(f.get("quantity_kind", f.get("quantityKind")) or "") for f in facts if fact_is_eligible(f)})}
    schema_hash = hashlib.sha256(json.dumps(schema_payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    raw_rows = _read_forecasts()
    out.mkdir(parents=True, exist_ok=True)
    (out / "feature-map.json").write_text(json.dumps({"schema": schema_payload, "sha256": schema_hash}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prefit = {"status": "frozen_before_fit", "inputSha256": _hashes(),
        "implementationSha256": sha256(Path(__file__)), "protocolSha256": sha256(PROTOCOL),
        "featureMapSha256": schema_hash, "plannedForecastRows": len(raw_rows),
        "maxFitAttempts": MAX_FITS, "alpha": ALPHA, "minUniqueMatureOrigins": MIN_ORIGINS,
        "minIndependentUpdateGroups": MIN_UPDATE_GROUPS, "docAvailabilityBoundary": "origin 10:00 Asia/Seoul",
        "priceFeatureBoundary": "V5 saved inputCutoff unchanged", "labelMaturity": "targetEnd + 2 days <= fit origin date",
        "fitSchedule": "expanding prequential, per crop x horizon", "trainingPairing": "identical D0/D1 rows",
        "typedFeatureGroups": len(group_details), "quantityKindAliases": QUANTITY_KIND_ALIASES,
        "sourceAliases": SOURCE_ALIASES, "eligibleFactCount": sum(fact_is_eligible(f) for f in facts),
        "eligibleFactsByCohort": cohort_support, "excludedFactCount": len(excluded_facts),
        "excludedFacts": excluded_facts,
        "retainedNegatedKreiNoteFactIds": sorted(negated_krei_retained)}
    prefit_path = out / "prefit-manifest.json"
    prefit_path.write_text(json.dumps(prefit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prefit_hash = sha256(prefit_path)
    source_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    for row in raw_rows:
        selected = select_facts(facts, row)
        row["selectedFacts"] = selected
        row["updateGroups"] = sorted({str(s["fact"].get("update_group_id", s["fact"].get("updateGroupId", ""))) for s in selected if s["fact"].get("update_group_id", s["fact"].get("updateGroupId"))})
        common = ["calendar::dekad_sin", "calendar::dekad_cos", "price::log_base_over_pref", "price::ratio_missing", *masks, *[f"missing::{x}" for x in numeric]]
        row["featureNames"] = {"d0": common, "d1": [*common, *numeric]}
        row["featureRows"] = {"d0": _features_from_row(row, masks, numeric, False),
                              "d1": _features_from_row(row, masks, numeric, True)}
        source_rows.append(row)
        if selected:
            for s in selected:
                f = s["fact"]
                match_rows.append({"origin": row["origin"], "crop": row["crop"], "horizon": row["horizon"],
                    "targetStart": row["targetStart"], "targetEnd": row["targetEnd"], "availability": "matched",
                    "quantityKind": f.get("quantity_kind", f.get("quantityKind")), "value": f.get("value"),
                    "valueLower": f.get("value_lower", f.get("valueLower")), "valueUpper": f.get("value_upper", f.get("valueUpper")),
                    "unit": f.get("unit"), "sourceOrg": f.get("source_org", f.get("sourceOrg")), "region": f.get("region"),
                    "varietyOrCropType": f.get("variety_or_crop_type", f.get("varietyOrCropType")), "factId": fact_id(f),
                    "updateGroupId": f.get("update_group_id", f.get("updateGroupId")), "publishEligibleAt": s["eligibleAt"],
                    "ageDays": s["ageDays"], "validity": s["validity"], "overlap": s["overlap"],
                    "effectiveStart": s["effectiveStart"], "effectiveEnd": s["effectiveEnd"], "tieConflict": s.get("tieConflict", False),
                    "evidenceSpan": f.get("evidence_span", f.get("evidenceSpan")), "sourceUrl": f.get("source_url", f.get("sourceUrl"))})
        else:
            match_rows.append({"origin": row["origin"], "crop": row["crop"], "horizon": row["horizon"],
                "targetStart": row["targetStart"], "targetEnd": row["targetEnd"], "availability": "none", "factId": None})
    counters = {"fits": 0}
    forecasts = [_forecast_one(row, source_rows, masks, numeric, counters) for row in source_rows]
    import pandas as pd
    forecast_path = out / "forecasts.parquet"
    # Keep the prediction artifact target-free. actual values are used only for
    # prequential training and are written to an isolated scoring input by caller.
    pd.DataFrame(forecasts).to_parquet(forecast_path, index=False)
    _csv_write(out / "matched-facts.csv", match_rows)
    manifest = {"status": "forecast_complete", "prefitManifestSha256": prefit_hash,
        "fitAttempts": counters["fits"], "maxFitAttempts": MAX_FITS,
        "featureMapSha256": schema_hash, "inputSha256": _hashes(), "implementationSha256": sha256(Path(__file__)),
        "protocolSha256": sha256(PROTOCOL), "forecastSha256": sha256(forecast_path),
        "matchCsvSha256": sha256(out / "matched-facts.csv"), "plannedRows": len(forecasts),
        "forecastRows": len(forecasts), "labelFreeForecastArtifact": True,
        "scoringLabelsFrom": str(PRICE_SOURCE), "trainedCells": ["cabbage/2", "cabbage/3", "radish/2", "radish/3"],
        "baseOnlyCells": ["onion/2", "onion/3", "garlic/2", "garlic/3"]}
    (out / "forecast-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("forecast",))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.stage == "forecast":
        print(json.dumps(run_forecast(args.out), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
