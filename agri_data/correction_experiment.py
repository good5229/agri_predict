"""Frozen retrospective correction-layer experiment; never mutates production forecasts."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from . import significance
from . import experiments

FROZEN_INPUT = Path("outputs/daily/2026-09-30/20260930T122342-844989ce/predictions.json")
FROZEN_INPUT_SHA256 = "9d6ac4d6a00c581821e97b4c4de074d129af6f06b8c8b38fe7ecbbf398c277f7"
OUTPUT_DEFAULT = Path("outputs/experiments/correction-20260930")
PRODUCT_IDS = ("cabbage", "radish")
CANDIDATES = ("fixed_half", "adaptive", "correction_regime")
COMPARATORS = ("ridge", "persistence")
CALIBRATION_START = dt.date(2017, 1, 1)
CALIBRATION_END = dt.date(2018, 12, 31)
EVALUATION_START = dt.date(2019, 1, 1)
EVALUATION_END = dt.date(2020, 2, 18)
WINDOW_DAYS = 180
GLOBAL_MIN_N = 60
REGIME_MIN_N = 30
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
PRIMARY_FAMILY_SIZE = len(PRODUCT_IDS) * 10 * len(CANDIDATES) * len(COMPARATORS) * 2


class ExperimentError(RuntimeError):
    pass


def _day(value: Any, label: str) -> dt.date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be positive and finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _quantile75(values: list[float]) -> float:
    """Type-7 linear-interpolation 75th percentile, fixed in the protocol."""
    if not values:
        raise ValueError("cannot compute regime threshold without observations")
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.75
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _losses(rows: list[dict[str, Any]], weight: float) -> tuple[float, float]:
    if not rows:
        return math.inf, math.inf
    errors = []
    for row in rows:
        if weight <= 0:
            predicted = row["baseline"]
        elif weight >= 1:
            predicted = row["ridge"]
        else:
            predicted = row["baseline"] + weight * (row["ridge"] - row["baseline"])
        errors.append(predicted - row["actual"])
    mae = sum(abs(error) for error in errors) / len(errors)
    mape = sum(abs(error) / row["actual"] * 100 for error, row in zip(errors, rows)) / len(rows)
    return mae, mape


def _select_weight(rows: list[dict[str, Any]], min_n: int) -> dict[str, Any]:
    if len(rows) < min_n:
        return {"weight": 0.0, "trainN": len(rows), "status": "fallback_insufficient_pairs",
                "fallbackReason": f"n<{min_n}"}
    baseline_mae, baseline_mape = _losses(rows, 0.0)
    if baseline_mae <= 0 or baseline_mape <= 0:
        return {"weight": 0.0, "trainN": len(rows), "status": "fallback_zero_baseline_loss",
                "fallbackReason": "baseline MAE or MAPE is zero"}
    eligible = []
    for weight in WEIGHTS:
        mae, mape = _losses(rows, weight)
        if weight > 0 and not (mae < baseline_mae and mape < baseline_mape):
            continue
        score = ((mae / baseline_mae) + (mape / baseline_mape)) / 2
        eligible.append((score, weight, mae, mape))
    if not eligible:
        return {"weight": 0.0, "trainN": len(rows), "status": "fallback_no_eligible_weight",
                "fallbackReason": "no weight passed both strict baseline gates"}
    score, weight, mae, mape = min(eligible, key=lambda row: (row[0], row[1]))
    return {"weight": weight, "trainN": len(rows), "status": "fit", "selectionScore": score,
            "trainMAE": mae, "trainMAPE_percent": mape,
            "baselineTrainMAE": baseline_mae, "baselineTrainMAPE_percent": baseline_mape}


def _signal(row: dict[str, Any]) -> float:
    return abs(math.log(row["ridge"] / row["baseline"]))


def _fit_regimes(rows: list[dict[str, Any]], common_fit: dict[str, Any]) -> dict[str, Any]:
    if len(rows) < GLOBAL_MIN_N:
        return {"status": "fallback_global_insufficient_pairs", "threshold75": None,
                "commonWeight": common_fit["weight"], "regimes": {}}
    threshold = _quantile75([_signal(row) for row in rows])
    buckets = {"low": [row for row in rows if _signal(row) <= threshold],
               "high": [row for row in rows if _signal(row) > threshold]}
    regimes = {}
    for name, subset in buckets.items():
        if len(subset) < REGIME_MIN_N:
            regimes[name] = {"weight": common_fit["weight"], "trainN": len(subset),
                             "status": "fallback_common_weight", "fallbackReason": f"regime n<{REGIME_MIN_N}"}
        else:
            regimes[name] = _select_weight(subset, REGIME_MIN_N)
    return {"status": "fit", "threshold75": threshold, "thresholdMethod": "type-7 linear interpolation",
            "regimeRule": "low <= threshold; high > threshold", "commonWeight": common_fit["weight"],
            "regimes": regimes}


def _metrics(rows: list[dict[str, Any]], prediction_key: str) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "MAE": None, "MAPE_percent": None, "RMSE": None,
                "sumAE": 0.0, "sumAPE_percent": 0.0, "sumSquaredError": 0.0}
    errors = [row[prediction_key] - row["actual"] for row in rows]
    absolute = [abs(error) for error in errors]
    ape = [value / row["actual"] * 100 for value, row in zip(absolute, rows)]
    squared = [error * error for error in errors]
    return {"n": len(rows), "MAE": sum(absolute) / len(rows),
            "MAPE_percent": sum(ape) / len(ape), "RMSE": math.sqrt(sum(squared) / len(squared)),
            "sumAE": sum(absolute), "sumAPE_percent": sum(ape), "sumSquaredError": sum(squared)}


def _sensitivity_status(ae: dict[str, Any], ape: dict[str, Any]) -> str:
    tests = (ae["sensitivity"]["test"], ape["sensitivity"]["test"])
    if any(row["status"] != "available" for row in tests):
        return "unavailable"
    if all(row["meanLossDifference"] < 0 and row["holmAdjustedPValue"] < 0.05 for row in tests):
        return "improved"
    if any(row["meanLossDifference"] > 0 and row["holmAdjustedPValue"] < 0.05 for row in tests):
        return "worsened_or_mixed"
    return "no_evidence"


def _month_start(day: dt.date) -> dt.date:
    return dt.date(day.year, day.month, 1)


def _training_rows(baseline: dict[tuple[str, int], dict[str, Any]],
                   ridge: dict[tuple[str, int], dict[str, Any]], actuals: dict[str, float],
                   horizon: int, fit_origin: dt.date) -> list[dict[str, Any]]:
    lower = fit_origin - dt.timedelta(days=WINDOW_DAYS)
    keys = set(baseline) & set(ridge)
    rows = []
    for target, h in sorted(keys):
        if h != horizon:
            continue
        target_day = _day(target, "training target")
        if lower <= target_day < fit_origin and target in actuals:
            rows.append({"target": target, "actual": actuals[target],
                         "baseline": baseline[(target, h)]["predicted"],
                         "ridge": ridge[(target, h)]["predicted"]})
    return rows


def _fit_month(product_id: str, horizon: int, fit_origin: dt.date,
               baseline: dict[tuple[str, int], dict[str, Any]],
               ridge: dict[tuple[str, int], dict[str, Any]], actuals: dict[str, float]) -> dict[str, Any]:
    rows = _training_rows(baseline, ridge, actuals, horizon, fit_origin)
    common = _select_weight(rows, GLOBAL_MIN_N)
    regimes = _fit_regimes(rows, common)
    targets = [row["target"] for row in rows]
    return {"productId": product_id, "horizon": horizon, "fitOrigin": fit_origin.isoformat(),
            "trainingTargetStart": min(targets) if targets else None,
            "trainingTargetEnd": max(targets) if targets else None,
            "trainingTargetStrictlyBeforeFitOrigin": True,
            "windowDays": WINDOW_DAYS, "globalN": len(rows), "adaptive": common,
            "regime": regimes}


def _corrected_row(source_row: dict[str, Any], baseline_row: dict[str, Any], ridge_row: dict[str, Any],
                   weight: float, fit: dict[str, Any], regime: str | None) -> dict[str, Any]:
    if weight <= 0:
        predicted = baseline_row["predicted"]
    elif weight >= 1:
        predicted = ridge_row["predicted"]
    else:
        predicted = baseline_row["predicted"] + weight * (ridge_row["predicted"] - baseline_row["predicted"])
    return {"target": source_row["target"], "origin": source_row["origin"],
            "horizon": source_row["horizon"], "predicted": predicted,
            "sourceDate": source_row["sourceDate"], "fitCutoff": fit["fitOrigin"],
            "weight": weight, "regime": regime,
            "fitOrigin": fit["fitOrigin"], "trainingTargetStart": fit["trainingTargetStart"],
            "trainingTargetEnd": fit["trainingTargetEnd"], "trainingN": fit["globalN"],
            "regimeTrainingN": (fit["regime"].get("regimes", {}).get(regime, {}).get("trainN")
                                if regime else None)}


def _previous_actuals(observations: list[dict[str, Any]]) -> dict[str, tuple[str, float] | None]:
    ordered = sorted((_day(row["date"], "observation date"), _finite(row["actual"], "actual"))
                     for row in observations)
    previous = {}
    prior = None
    for day, actual in ordered:
        previous[day.isoformat()] = None if prior is None else (prior[0].isoformat(), prior[1])
        prior = (day, actual)
    return previous


def _spike_targets(target_rows: list[dict[str, Any]], previous: dict[str, tuple[str, float] | None]) -> tuple[set[str], dict[str, Any]]:
    eligible = []
    gaps: dict[str, int] = {"gap1": 0, "gapGreaterThanOne": 0}
    for row in target_rows:
        prior = previous.get(row["target"])
        if prior is None:
            continue
        gap = (_day(row["target"], "target") - _day(prior[0], "previous actual date")).days
        gaps["gap1" if gap == 1 else "gapGreaterThanOne"] += 1
        eligible.append((abs(math.log(row["actual"] / prior[1])), row["target"], gap))
    eligible.sort(key=lambda item: (-item[0], item[1]))
    count = math.ceil(len(eligible) * 0.10) if eligible else 0
    chosen = eligible[:count]
    return {day for _, day, _ in chosen}, {"eligibleN": len(eligible), "spikeN": count,
        "cutoffAbsLogChange": chosen[-1][0] if chosen else None,
        "eligiblePriorObservationGaps": gaps,
        "spikePriorObservationGaps": {"gap1": sum(row[2] == 1 for row in chosen),
                                      "gapGreaterThanOne": sum(row[2] > 1 for row in chosen)},
        "selection": "top ceil(10%) by abs(log(actual/prior observed actual)); ties by target date"}


def _season(day: dt.date) -> str:
    if day.month in (12, 1, 2):
        return "winter"
    if day.month in (3, 4, 5):
        return "spring"
    if day.month in (6, 7, 8):
        return "summer"
    return "autumn"


def _gate_metrics(rows: list[dict[str, Any]], candidate: dict[tuple[str, int], dict[str, Any]],
                  horizon: int, spike_dates: set[str]) -> dict[str, Any]:
    candidate_rows = []
    for row in rows:
        prediction = candidate.get((row["target"], horizon))
        if prediction is not None:
            candidate_rows.append({**row, "candidate": prediction["predicted"]})
    eligible = [row for row in candidate_rows if row["previousActualDate"] is not None]
    spike = [row for row in eligible if row["target"] in spike_dates]
    quiet = [row for row in eligible if row["target"] not in spike_dates]
    ridge_quiet = _metrics(quiet, "ridge")
    candidate_quiet = _metrics(quiet, "candidate")
    ridge_spike = _metrics(spike, "ridge")
    candidate_spike = _metrics(spike, "candidate")
    quiet_pass = bool(quiet and candidate_quiet["MAE"] < ridge_quiet["MAE"]
                      and candidate_quiet["MAPE_percent"] < ridge_quiet["MAPE_percent"])
    spike_pass = bool(spike and candidate_spike["MAE"] <= ridge_spike["MAE"]
                      and candidate_spike["MAPE_percent"] <= ridge_spike["MAPE_percent"])
    return {"eligibleN": len(eligible), "quietN": len(quiet), "spikeN": len(spike),
            "unclassifiedNoPreviousActualN": len(candidate_rows) - len(eligible),
            "quietMetrics": {"ridge": ridge_quiet, "candidate": candidate_quiet},
            "spikeMetrics": {"ridge": ridge_spike, "candidate": candidate_spike},
            "quietBothLossesStrictlyImprove": quiet_pass,
            "spikeBothLossesNonWorsening": spike_pass,
            "descriptiveGatePass": quiet_pass and spike_pass}


def _monthly_season_metrics(rows: list[dict[str, Any]], candidate: dict[tuple[str, int], dict[str, Any]],
                            horizon: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = row["target"][:7]
        groups.setdefault(key, []).append(row)
    monthly = {}
    seasonal_groups: dict[str, list[dict[str, Any]]] = {}
    for month, subset in sorted(groups.items()):
        season = _season(_day(month + "-01", "month"))
        candidate_subset = [{**row, "candidate": candidate[(row["target"], horizon)]["predicted"]}
                            for row in subset if (row["target"], horizon) in candidate]
        ridge_subset = [{**row, "candidate": row["ridge"]} for row in subset]
        monthly[month] = {"n": len(subset), "ridge": _metrics(ridge_subset, "candidate"),
                          "candidate": _metrics(candidate_subset, "candidate")}
        seasonal_groups.setdefault(season, []).extend(subset)
    seasonal = {}
    for name, subset in sorted(seasonal_groups.items()):
        candidate_subset = [{**row, "candidate": candidate[(row["target"], horizon)]["predicted"]}
                            for row in subset if (row["target"], horizon) in candidate]
        ridge_subset = [{**row, "candidate": row["ridge"]} for row in subset]
        seasonal[name] = {"n": len(subset), "ridge": _metrics(ridge_subset, "candidate"),
                           "candidate": _metrics(candidate_subset, "candidate")}
    return {"monthly": monthly, "seasonal": seasonal}


def _build_candidate_predictions(product_id: str, candidate_id: str,
                                 baseline: dict[tuple[str, int], dict[str, Any]],
                                 ridge: dict[tuple[str, int], dict[str, Any]],
                                 actuals: dict[str, float]) -> tuple[dict[tuple[str, int], dict[str, Any]], list[dict[str, Any]]]:
    corrected = {}
    fits = []
    for horizon in range(1, 11):
        keys = set(baseline) & set(ridge)
        eval_rows = []
        for target, h in keys:
            if h == horizon and EVALUATION_START <= _day(target, "target") <= EVALUATION_END and target in actuals:
                b, r = baseline[(target, h)], ridge[(target, h)]
                origin = _day(r["origin"], "origin")
                eval_rows.append({"target": target, "origin": origin.isoformat(), "horizon": h,
                                  "actual": actuals[target], "baseline": b["predicted"], "ridge": r["predicted"],
                                  "ridgeRow": r, "baselineRow": b})
        month_fits: dict[str, dict[str, Any]] = {}
        for row in sorted(eval_rows, key=lambda item: item["origin"]):
            origin = _day(row["origin"], "origin")
            month_key = origin.strftime("%Y-%m")
            if month_key not in month_fits:
                fit_origin = _month_start(origin)
                if fit_origin > origin:
                    raise ExperimentError("month-start fit is after the forecast origin")
                if candidate_id == "fixed_half":
                    month_fits[month_key] = {"productId": product_id, "horizon": horizon,
                        "fitOrigin": fit_origin.isoformat(), "trainingTargetStart": None,
                        "trainingTargetEnd": None, "trainingTargetStrictlyBeforeFitOrigin": True,
                        "windowDays": WINDOW_DAYS, "globalN": 0,
                        "adaptive": {"weight": 0.5, "status": "fixed_not_fitted"},
                        "regime": {"status": "not_fitted", "threshold75": None, "commonWeight": 0.5,
                                   "regimes": {}}}
                else:
                    month_fits[month_key] = _fit_month(product_id, horizon, fit_origin, baseline, ridge, actuals)
                fits.append(month_fits[month_key])
            fit = month_fits[month_key]
            if candidate_id == "fixed_half":
                weight, regime = 0.5, None
            elif candidate_id == "adaptive":
                weight, regime = fit["adaptive"]["weight"], None
            else:
                regime_fit = fit["regime"]
                if regime_fit["status"] != "fit":
                    regime = "fallback_common"
                    weight = regime_fit["commonWeight"]
                else:
                    current_signal = abs(math.log(row["ridge"] / row["baseline"]))
                    regime = "low" if current_signal <= regime_fit["threshold75"] else "high"
                    weight = regime_fit["regimes"][regime]["weight"]
            corrected[(row["target"], horizon)] = _corrected_row(row["ridgeRow"], row["baselineRow"],
                                                                   row["ridgeRow"], weight, fit, regime)
    return corrected, fits


def _metrics_rows(product: dict[str, Any], baseline: dict[tuple[str, int], dict[str, Any]],
                  ridge: dict[tuple[str, int], dict[str, Any]], horizon: int) -> list[dict[str, Any]]:
    observations = {row["date"]: _finite(row["actual"], "actual") for row in product.get("observations", [])}
    result = []
    for (target, h) in sorted(set(baseline) & set(ridge)):
        day = _day(target, "target")
        if h != horizon or not EVALUATION_START <= day <= EVALUATION_END or target not in observations:
            continue
        result.append({"target": target, "actual": observations[target],
                       "baseline": baseline[(target, h)]["predicted"],
                       "ridge": ridge[(target, h)]["predicted"]})
    return result


def _report_conclusions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {(row["productId"], row["candidateId"], row["comparatorId"], row["horizon"], row["loss"]): row
              for row in records}
    conclusions = []
    for product in PRODUCT_IDS:
        for candidate in CANDIDATES:
            for horizon in range(1, 11):
                comparator_conclusions = {}
                for comparator in COMPARATORS:
                    ae = by_key[(product, candidate, comparator, horizon, "absolute_error")]
                    ape = by_key[(product, candidate, comparator, horizon, "absolute_percentage_error")]
                    outcome = significance._conclusion(ae["primary"], ape["primary"])
                    outcome["sensitivityStatus"] = _sensitivity_status(ae, ape)
                    comparator_conclusions[comparator] = outcome
                conclusions.append({"productId": product, "candidateId": candidate, "horizon": horizon,
                                    "comparators": comparator_conclusions,
                                    "primaryBothComparatorsImprove": all(row["status"] == "improved"
                                                                          for row in comparator_conclusions.values()),
                                    "sensitivityBothComparatorsImprove": all(row["sensitivityStatus"] == "improved"
                                                                              for row in comparator_conclusions.values())})
    return conclusions


def analyze(source: dict[str, Any], source_sha256: str, *, expected_sha256: str = FROZEN_INPUT_SHA256) -> dict[str, Any]:
    if source_sha256 != expected_sha256:
        raise ExperimentError("input bytes do not match the frozen prediction archive hash")
    try:
        products, forecasts, eval_start, eval_end = significance._validate_input(source)
    except (TypeError, ValueError, KeyError) as exc:
        raise ExperimentError(f"invalid archived prediction input: {exc}") from exc
    if eval_start > EVALUATION_START or eval_end < EVALUATION_END:
        raise ExperimentError("frozen prediction input does not cover the fixed evaluation window")
    if len(PRODUCT_IDS) * 10 * len(CANDIDATES) * len(COMPARATORS) * len(significance.LOSSES) != 240:
        raise ExperimentError("fixed Holm family definition changed")
    actual_maps = {}
    corrected_maps: dict[tuple[str, str], dict[tuple[str, int], dict[str, Any]]] = {}
    all_fits = []
    product_details = []
    for product_id in PRODUCT_IDS:
        product = products[product_id]
        observations = product["observations"]
        actuals = {_day(row["date"], "actual date").isoformat(): _finite(row["actual"], "actual")
                   for row in observations}
        actual_maps[product_id] = actuals
        baseline = forecasts[(product_id, "baseline")]
        ridge = forecasts[(product_id, "ridge", "daily")]
        previous = _previous_actuals(observations)
        candidate_summaries = {}
        product_candidate_maps = {}
        for candidate_id in CANDIDATES:
            corrected, fits = _build_candidate_predictions(product_id, candidate_id, baseline, ridge, actuals)
            product_candidate_maps[candidate_id] = corrected
            all_fits.extend({"candidateId": candidate_id, **fit} for fit in fits)
        for horizon in range(1, 11):
            eligible_rows = _metrics_rows(product, baseline, ridge, horizon)
            for row in eligible_rows:
                prior = previous.get(row["target"])
                row["previousActualDate"] = None if prior is None else prior[0]
                row["previousActual"] = None if prior is None else prior[1]
            spikes, spike_info = _spike_targets(eligible_rows, previous)
            for candidate_id in CANDIDATES:
                corrected = product_candidate_maps[candidate_id]
                corrected_maps[(product_id, candidate_id)] = corrected
                candidate_summaries[f"{candidate_id}/h{horizon}"] = {
                    "candidateId": candidate_id, "horizon": horizon,
                    "spikeDefinition": spike_info,
                    "monthlySeasonalMetrics": {}}
                model_rows = []
                for row in eligible_rows:
                    predicted = corrected.get((row["target"], horizon))
                    if predicted is not None:
                        model_rows.append({**row, "candidate": predicted["predicted"]})
                candidate_summaries[f"{candidate_id}/h{horizon}"]["matchedMetrics"] = {
                    "persistence": _metrics(model_rows, "baseline"),
                    "ridge": _metrics(model_rows, "ridge"),
                    "candidate": _metrics(model_rows, "candidate")}
                candidate_summaries[f"{candidate_id}/h{horizon}"]["spikeQuietGate"] = _gate_metrics(
                    eligible_rows, corrected, horizon, spikes)
                candidate_summaries[f"{candidate_id}/h{horizon}"]["monthlySeasonalMetrics"] = _monthly_season_metrics(
                    eligible_rows, corrected, horizon)
        product_details.append({"productId": product_id, "unit": "원/kg", "candidateHorizons": candidate_summaries})

    records = []
    for product_id in PRODUCT_IDS:
        baseline = forecasts[(product_id, "baseline")]
        ridge = forecasts[(product_id, "ridge", "daily")]
        actuals = {day: value for day, value in actual_maps[product_id].items()
                   if EVALUATION_START.isoformat() <= day <= EVALUATION_END.isoformat()}
        for candidate_id in CANDIDATES:
            candidate = corrected_maps[(product_id, candidate_id)]
            for horizon in range(1, 11):
                for comparator_id, comparator in (("persistence", baseline), ("ridge", ridge)):
                    for loss in significance.LOSSES:
                        records.append({"productId": product_id, "candidateId": candidate_id,
                                        "comparatorId": comparator_id, "horizon": horizon, "loss": loss,
                                        "primary": significance._paired_comparison(
                                            candidate, comparator, actuals, horizon, loss,
                                            significance.PRIMARY_LAG_DAYS),
                                        "sensitivity": significance._paired_comparison(
                                            candidate, comparator, actuals, horizon, loss,
                                            significance.SENSITIVITY_LAG_DAYS)})
    if len(records) != 240:
        raise ExperimentError(f"internal family size mismatch: expected 240, got {len(records)}")
    significance._apply_fixed_holm(records, "primary")
    significance._apply_fixed_holm(records, "sensitivity")
    conclusions = _report_conclusions(records)
    if len(conclusions) != 60:
        raise ExperimentError(f"internal conclusion count mismatch: expected 60, got {len(conclusions)}")
    adoption = []
    for conclusion in conclusions:
        detail = next(row for row in product_details if row["productId"] == conclusion["productId"])
        gate = detail["candidateHorizons"][f"{conclusion['candidateId']}/h{conclusion['horizon']}"]["spikeQuietGate"]
        adoption.append({**conclusion,
                         "quietBothLossesStrictlyImproveVsRidge": gate["quietBothLossesStrictlyImprove"],
                         "spikeBothLossesNonWorseningVsRidge": gate["spikeBothLossesNonWorsening"],
                         "adoptableWithinThisRetrospectiveExperiment": bool(
                             conclusion["primaryBothComparatorsImprove"]
                             and conclusion["sensitivityBothComparatorsImprove"]
                             and gate["quietBothLossesStrictlyImprove"]
                             and gate["spikeBothLossesNonWorsening"])})
    fit_summary = {}
    for candidate_id in CANDIDATES:
        candidate_fits = [row for row in all_fits if row["candidateId"] == candidate_id]
        adaptive_fits = [row for row in candidate_fits if row.get("adaptive", {}).get("status") != "fixed_not_fitted"]
        adaptive_fallbacks = sum(row.get("adaptive", {}).get("status", "").startswith("fallback")
                                 for row in adaptive_fits)
        regime_fallback_fits = [row for row in candidate_fits if row.get("regime", {}).get("status") != "not_fitted"]
        regime_global_fallbacks = sum(row.get("regime", {}).get("status") == "fallback_global_insufficient_pairs"
                                      for row in regime_fallback_fits)
        regime_bucket_fallbacks = sum(
            bucket.get("status", "").startswith("fallback")
            for row in regime_fallback_fits
            for bucket in row.get("regime", {}).get("regimes", {}).values())
        regime_fallbacks = 2 * regime_global_fallbacks + regime_bucket_fallbacks
        regime_denominator = 2 * len(regime_fallback_fits)
        fit_summary[candidate_id] = {"fitCount": len(candidate_fits),
                                     "adaptiveFitCount": len(adaptive_fits),
                                     "adaptiveFallbackCount": adaptive_fallbacks,
                                     "adaptiveFallbackRate": (adaptive_fallbacks / len(adaptive_fits)
                                                               if adaptive_fits else None),
                                     "regimeGlobalFallbackFitCount": regime_global_fallbacks,
                                     "regimeBucketFallbackCount": regime_bucket_fallbacks,
                                     "regimeFallbackCount": regime_fallbacks,
                                     "regimeFallbackRate": (regime_fallbacks / regime_denominator
                                                             if regime_denominator and candidate_id == "correction_regime"
                                                             else None)}
    corrected_predictions = []
    for (product_id, candidate_id), rows in sorted(corrected_maps.items()):
        corrected_predictions.extend({"productId": product_id, "candidateId": candidate_id, **row}
                                     for _, row in sorted(rows.items()))
    return {"schemaVersion": 1,
            "protocol": {"name": "frozen monthly prediction correction layer",
                         "inputSha256": source_sha256, "evaluationStart": EVALUATION_START.isoformat(),
                         "evaluationEnd": EVALUATION_END.isoformat(),
                         "calibrationData": "archived predictions with matured target labels from 2017-01-01 onward",
                         "fitSchedule": "calendar month-start for each horizon and origin-month; independent of target availability",
                         "fitWindow": "target date in [fitOrigin-180 calendar days, fitOrigin); strict target < fitOrigin",
                         "formula": "persistence + weight × (Ridge - persistence)",
                         "candidates": {"fixed_half": {"weight": 0.5},
                                         "adaptive": {"weights": list(WEIGHTS), "minimumGlobalPairs": GLOBAL_MIN_N},
                                         "correction_regime": {"weights": list(WEIGHTS),
                                             "minimumGlobalPairs": GLOBAL_MIN_N,
                                             "minimumRegimePairs": REGIME_MIN_N,
                                             "signal": "abs(log(Ridge/persistence))",
                                             "threshold": "type-7 linear-interpolation 75th percentile from training rows",
                                             "regimes": "low <= threshold; high > threshold",
                                             "fallback": "regime below minimum uses adaptive common weight; global below 60 uses zero"}},
                         "weightSelection": "w>0 must strictly improve training MAE and MAPE against persistence; minimize mean normalized MAE/MAPE; ties choose smaller w; no eligible fit uses w=0",
                         "primaryInference": {"method": "paired DM asymptotic normal, two-sided, calendar Bartlett HAC",
                             "maxLagDays": significance.PRIMARY_LAG_DAYS, "familySize": 240,
                             "family": "2 products × 10 horizons × 3 candidates × 2 comparators × 2 losses",
                             "holmUnavailableP": 1.0, "alpha": 0.05},
                         "sensitivityInference": {"maxLagDays": significance.SENSITIVITY_LAG_DAYS,
                             "familySize": 240, "holmUnavailableP": 1.0},
                         "interpretation": "retrospective exploratory reuse of 2019-2020 evaluation; not untouched confirmation; no causal claim",
                         "limitations": ["관측가격은 사후 수정본이며 최초 공표시각을 알 수 없습니다.",
                                         "미기각은 동등성의 증거가 아니며 DM 검정은 점근적입니다.",
                                         "월·계절·급변/평상시 조건은 기술·채택 게이트이며 별도 유의성 검정을 추가하지 않습니다.",
                                         "가중치 보정층 결과는 원 Ridge 모델 재학습 결과가 아닙니다."]},
            "fits": all_fits, "fitSummary": fit_summary,
            "correctedPredictions": corrected_predictions,
            "productDiagnostics": product_details,
            "comparisons": records, "conclusions": adoption}


def markdown_report(report: dict[str, Any]) -> str:
    protocol = report["protocol"]
    lines = ["# 예측 보정층 실험", "", f"입력 SHA-256: `{protocol['inputSha256']}`",
             f"평가: {protocol['evaluationStart']}~{protocol['evaluationEnd']}",
             "2019–2020은 기존 탐색에서 사용한 회고 기간이며 미관측 확증 holdout이 아닙니다.",
             "원 Ridge 재학습이 아닌 월별 보정층 비교입니다. 최초 가격 공표시각은 미확인입니다.", "",
             "## 일별 1일 주표", "", "| 품목 | 후보 | 비교대상 | n | 후보 MAE | 비교 MAE | 후보 MAPE% | 비교 MAPE% | 후보 RMSE | 비교 RMSE |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    rows = {(row["productId"], row["candidateId"], row["horizon"]): row for row in report["conclusions"]}
    details = {row["productId"]: row for row in report["productDiagnostics"]}
    for product in PRODUCT_IDS:
        for candidate in CANDIDATES:
            row = rows[(product, candidate, 1)]
            metrics = details[product]["candidateHorizons"][f"{candidate}/h1"]["matchedMetrics"]
            candidate_metrics = metrics["candidate"]
            for comparator in COMPARATORS:
                compare_metrics = metrics[comparator]
                fmt = lambda value: "—" if value is None else f"{value:.4g}"
                lines.append(f"| {product} | {candidate} | {comparator} | {candidate_metrics['n']} | "
                             f"{fmt(candidate_metrics['MAE'])} | {fmt(compare_metrics['MAE'])} | "
                             f"{fmt(candidate_metrics['MAPE_percent'])} | {fmt(compare_metrics['MAPE_percent'])} | "
                             f"{fmt(candidate_metrics['RMSE'])} | {fmt(compare_metrics['RMSE'])} |")
    record_map = {(r["productId"], r["candidateId"], r["comparatorId"], r["horizon"], r["loss"]): r
                  for r in report["comparisons"]}
    lines += ["", "## 일별 1일 DM 보정 p값과 채택 게이트", "",
              "| 품목 | 후보 | 비교대상 | primary AE/APE Holm p | 28일 AE/APE Holm p | primary 판정 | 민감도 판정 | 평상시 개선 | 급변 비악화 | 내부 채택 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for product in PRODUCT_IDS:
        for candidate in CANDIDATES:
            conclusion = rows[(product, candidate, 1)]
            for comparator in COMPARATORS:
                ae = record_map[(product, candidate, comparator, 1, "absolute_error")]
                ape = record_map[(product, candidate, comparator, 1, "absolute_percentage_error")]
                def ptext(item, band):
                    test = item[band]["test"]
                    value = test.get("holmAdjustedPValue")
                    return "—" if test.get("status") != "available" or value is None else f"{value:.4g}"
                primary = conclusion["comparators"][comparator]["status"]
                sensitivity = conclusion["comparators"][comparator]["sensitivityStatus"]
                quiet = "통과" if conclusion["quietBothLossesStrictlyImproveVsRidge"] else "미통과"
                spike = "통과" if conclusion["spikeBothLossesNonWorseningVsRidge"] else "미통과"
                adopt = "통과" if conclusion["adoptableWithinThisRetrospectiveExperiment"] else "미통과"
                lines.append(f"| {product} | {candidate} | {comparator} | {ptext(ae, 'primary')}/{ptext(ape, 'primary')} | "
                             f"{ptext(ae, 'sensitivity')}/{ptext(ape, 'sensitivity')} | {primary} | {sensitivity} | "
                             f"{quiet} | {spike} | {adopt} |")
    primary_pass = sum(row["primaryBothComparatorsImprove"] for row in report["conclusions"])
    adopted = sum(row["adoptableWithinThisRetrospectiveExperiment"] for row in report["conclusions"])
    fallback_lines = []
    for key, value in report["fitSummary"].items():
        adaptive = ("미학습" if value["adaptiveFallbackRate"] is None else
                    f"{value['adaptiveFallbackCount']}/{value['adaptiveFitCount']} ({value['adaptiveFallbackRate']:.1%})")
        regime = ("해당 없음" if value["regimeFallbackRate"] is None else
                  f"{value['regimeFallbackCount']} (global fit fallback {value['regimeGlobalFallbackFitCount']}, "
                  f"bucket fallback {value['regimeBucketFallbackCount']}, rate {value['regimeFallbackRate']:.1%})")
        fallback_lines.append(f"{key}: adaptive fallback {adaptive}; regime fallback {regime}")
    lines += ["", "후보별 10개 horizon, 240개 고정 Holm 비교, 월별·계절별 기술통계와 fit 기록은 JSON에 있습니다.",
              f"전체 60개 조합 중 primary 양 비교대상 개선 통과: {primary_pass}; 보수적 내부 채택 게이트 통과: {adopted}.",
              "Fallback 현황:", *[f"- {line}" for line in fallback_lines],
              "내부 채택 조건은 양 비교대상(primary 및 28일 민감도) 통과, 평상시 두 손실 개선, 급변일 두 손실 비악화를 함께 요구합니다.", ""]
    return "\n".join(lines)


def _exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(FROZEN_INPUT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DEFAULT))
    args = parser.parse_args(argv)
    try:
        source_path = Path(args.input)
        raw = source_path.read_bytes()
        source = json.loads(raw.decode("utf-8"))
        if not isinstance(source, dict):
            raise ValueError("input must be a JSON object")
        source_hash = hashlib.sha256(raw).hexdigest()
        report = analyze(source, source_hash)
        report["protocol"]["sourcePath"] = str(source_path)
        report["protocol"]["codeHashes"] = {
            "correction_experiment.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "significance.py": hashlib.sha256(Path(significance.__file__).read_bytes()).hexdigest(),
            "experiments.py": hashlib.sha256(Path(experiments.__file__).read_bytes()).hexdigest()}
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=False)
        _exclusive_write(out / "correction-experiment.json",
                         (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())
        _exclusive_write(out / "report.md", markdown_report(report).encode("utf-8"))
    except (OSError, json.JSONDecodeError, ValueError, ExperimentError, TypeError) as exc:
        print(f"correction experiment failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": "complete", "outputDirectory": str(out),
                      "comparisonCount": len(report["comparisons"]),
                      "conclusionCount": len(report["conclusions"]),
                      "inputSha256": source_hash}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
