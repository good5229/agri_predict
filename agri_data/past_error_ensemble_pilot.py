"""Frozen small pilot for combining existing forecasts by causal past error.

This module reads only the three pinned retrospective reports. It has no API,
database, or training access. The all-period expansion is intentionally absent;
the only scoring window is the small target interval frozen in docs/PLAN.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from . import dashboard, direct_horizon as dh

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "fixedBlend": ROOT / "outputs/experiments/simple-ensemble-20261007/report.json",
    "seasonalAnalog": ROOT / "outputs/experiments/seasonal-analog-20261007/report.json",
    "marketRecompose": ROOT / "outputs/experiments/market-recompose-20261007/report.json",
}
EXPECTED_SHA256 = {
    "fixedBlend": "7003511400f79bf354b13f0d64d580eae7215eb12ef5b15ebab0f61762291ffb",
    "seasonalAnalog": "2f754c4ac3cc17b3894441f313e322dc8ca3ca91a9b5274a2d63201e7a896089",
    "marketRecompose": "e5f23f1012dad2acc774c83f995e2ee2851409af913ee82483aa3d953ee87581",
}
PRODUCTS = tuple(dashboard.PRODUCTS)
HORIZONS = tuple(range(1, 11))
START, END = dt.date(2026, 8, 1), dt.date(2026, 9, 24)
EXPANSION_START, EXPANSION_END = dt.date(2025, 9, 25), dt.date(2026, 9, 24)
MIN_HISTORY, MIN_PAIRS = 20, 30
BANDWIDTHS = (9, 28)
ALPHA = 0.05
COMPARATORS = ("directRidge", "persistence")
LOSSES = ("mae", "mape")
FAMILY_SIZE = len(PRODUCTS) * len(HORIZONS) * len(COMPARATORS) * len(LOSSES)
REPORT_INPUTS = ("fixedBlend", "seasonalAnalog", "marketRecompose")
METHODS = ("directRidge", *REPORT_INPUTS)

PROTOCOL = {
    "id": "past-error-weighted-existing-forecasts-small-pilot-2026-10-08",
    "inputSha256": EXPECTED_SHA256,
    "identity": "the fixed-blend report defines direct baseline keys; retain the union of rows for observation and method availability; every overlapping key must agree on product, origin, target, horizon, actual, and direct Ridge",
    "history": "same product and horizon; target in [origin-365 calendar days, origin-14 calendar days), requiring valid direct Ridge, fixed blend, selected seasonal analog, and market recomposition forecasts; >=20 target pairs",
    "weight": "normalize exp(-0.25 * (method mean historical APE percentage points - minimum method mean historical APE percentage points))",
    "candidate": "0.5 * direct Ridge + 0.5 * weighted mean of direct Ridge, fixed blend, selected seasonal analog, and market recomposition",
    "fallback": "if any current forecast among the four methods is missing or invalid, use direct Ridge and record the method; fewer than 20 eligible all-four history pairs also uses direct Ridge",
    "evaluationTargets": [START.isoformat(), END.isoformat()],
    "macro": "paired MAE/MAPE first by product and horizon; 50-cell equal-weight macro average only when all 50 cell metrics exist; product MAPE equal-weights its 10 horizons; product MAE gate equal-weights its 10 horizon MAEs",
    "dm": {"twoSided": True, "calendarBartlettHacMaxLagDays": list(BANDWIDTHS), "holmAlpha": ALPHA,
           "familySizePerBandwidth": FAMILY_SIZE, "minimumPairs": MIN_PAIRS,
           "unavailableRule": "fewer than 30 paired targets or invalid/nonpositive HAC variance remains in the Holm family with p=1"},
    "expansionGate": {
        "all50ProductHorizonCellsNonfallbackAllFourPairsAtLeast": MIN_PAIRS,
        "candidateCoverageAtLeastDirectAndPersistence": True,
        "candidateMacroMapeAtMostPercent": 10,
        "candidateMacroMapeImprovementVsDirectAtLeastPercentagePoints": 0.5,
        "eachProductMapeWorseningVsDirectAtMostPercentagePoints": 1,
        "eachProductEqualHorizonMaeWorseningVsDirectAtMostFraction": 0.05,
        "allConditionsRequired": True,
    },
    "scope": "No full-period scoring command is implemented. This pilot is retrospective, overlaps previously inspected data, and does not establish independent performance or official nationwide scores.",
}


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be positive and finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _identity(row: Any, label: str) -> tuple[str, dt.date, dt.date, int]:
    if not isinstance(row, dict):
        raise ValueError(f"{label} row must be an object")
    product, horizon = row.get("product"), row.get("horizon")
    if product not in PRODUCTS or isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
        raise ValueError(f"{label} has invalid product/horizon")
    try:
        target = dt.date.fromisoformat(row["target"])
        origin = dt.date.fromisoformat(row["origin"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} has invalid origin/target") from exc
    if target - origin != dt.timedelta(days=horizon):
        raise ValueError(f"{label} origin/horizon mismatch")
    return product, origin, target, horizon


def _optional_positive(value: Any) -> tuple[float | None, str | None]:
    if value is None:
        return None, "missing_forecast"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None, "invalid_forecast"
    if isinstance(value, bool) or not math.isfinite(number) or number <= 0:
        return None, "invalid_forecast"
    return number, None


def _index(report: Any, label: str, prediction_field: str) -> dict[tuple, dict]:
    if not isinstance(report, dict) or not isinstance(report.get("pairedRows"), list):
        raise ValueError(f"{label} report must contain pairedRows")
    result = {}
    for i, row in enumerate(report["pairedRows"]):
        key = _identity(row, f"{label} row {i}")
        if key in result:
            raise ValueError(f"duplicate {label} key {key}")
        candidate, reason = _optional_positive(row.get(prediction_field))
        result[key] = {
            "product": key[0], "origin": key[1], "target": key[2], "horizon": key[3],
            "actual": _positive(row.get("actual"), f"{label} actual"),
            "directRidge": _positive(row.get("direct_ridge", row.get("directRidge")), f"{label} direct Ridge"),
            "candidate": candidate, "candidateStatus": reason,
        }
        if label == "fixedBlend":
            persistence, persistence_reason = _optional_positive(row.get("persistence"))
            result[key]["persistence"] = persistence
            result[key]["persistenceStatus"] = persistence_reason
    return result


def validate_reports(fixed: dict, seasonal: dict, market: dict) -> dict[tuple, dict]:
    """Retain baseline keys and union provenance while validating all overlaps."""
    indexed = {
        "fixedBlend": _index(fixed, "fixedBlend", "candidate"),
        "seasonalAnalog": _index(seasonal, "seasonalAnalog", "selectedPrediction"),
        "marketRecompose": _index(market, "marketRecompose", "candidatePrediction"),
    }
    all_keys = set().union(*(set(values) for values in indexed.values()))
    if not all_keys:
        raise ValueError("pinned reports have no forecast rows")
    out = {}
    for key in all_keys:
        sources = [(name, values[key]) for name, values in indexed.items() if key in values]
        reference = sources[0][1]
        for name, other in sources[1:]:
            for field in ("actual", "directRidge"):
                if not math.isclose(reference[field], other[field], rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f"pinned reports disagree on {field} at {key} ({name})")
        baseline = indexed["fixedBlend"].get(key)
        row = {"product": key[0], "origin": key[1], "target": key[2], "horizon": key[3],
               "actual": reference["actual"], "directRidge": reference["directRidge"],
               "baselineAvailable": baseline is not None,
               "persistence": baseline["persistence"] if baseline else None,
               "persistenceStatus": baseline["persistenceStatus"] if baseline else "missing_baseline_key",
               "sourceReports": [name for name, _ in sources]}
        for name in REPORT_INPUTS:
            source = indexed[name].get(key)
            row[name] = source["candidate"] if source else None
            row[f"{name}Status"] = source["candidateStatus"] if source else "missing_report_key"
        out[key] = row
    return out


def weights(history: list[dict]) -> tuple[dict[str, float], dict[str, float]]:
    """Calculate softmax-like fixed weights from historical mean APE points."""
    if not history:
        raise ValueError("weights require eligible historical rows")
    means = {name: sum(abs(r[name] - r["actual"]) / r["actual"] * 100 for r in history) / len(history)
             for name in METHODS}
    best = min(means.values())
    raw = {name: math.exp(-0.25 * (means[name] - best)) for name in METHODS}
    total = sum(raw.values())
    return means, {name: value / total for name, value in raw.items()}


def _forecast(rows: dict[tuple, dict], row: dict) -> dict:
    origin, target = row["origin"], row["target"]
    cutoff, earliest = origin - dt.timedelta(days=14), origin - dt.timedelta(days=365)
    past = [r for r in rows.values() if r["product"] == row["product"] and r["horizon"] == row["horizon"]
            and earliest <= r["target"] < cutoff and all(r.get(name) is not None for name in METHODS)]
    audit = {"eligibleHistoryTargets": len(past), "historyTargetStart": min((r["target"] for r in past), default=None),
             "historyTargetEnd": max((r["target"] for r in past), default=None), "historicalMeanApe": None,
             "weights": None, "fallback": False, "fallbackReason": None, "missingMethods": []}
    missing = [name for name in METHODS if row.get(name) is None]
    if missing:
        audit.update(fallback=True, missingMethods=missing,
                     fallbackReason="current_forecast_unavailable:" + ",".join(
                         f"{name}:{row.get(f'{name}Status') or 'invalid_forecast'}" for name in missing))
        return {**audit, "candidate": row["directRidge"]}
    if len(past) < MIN_HISTORY:
        audit.update(fallback=True, fallbackReason="fewer_than_20_causal_history_targets")
        return {**audit, "candidate": row["directRidge"]}
    try:
        means, weight_map = weights(past)
        blended = sum(weight_map[name] * row[name] for name in METHODS)
        candidate = 0.5 * row["directRidge"] + 0.5 * blended
        if not math.isfinite(candidate) or candidate <= 0:
            raise ValueError("weighted candidate is not positive and finite")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        audit.update(fallback=True, fallbackReason=f"candidate_unavailable:{type(exc).__name__}")
        return {**audit, "candidate": row["directRidge"]}
    audit.update(historicalMeanApe=means, weights=weight_map)
    return {**audit, "candidate": candidate}


def _metrics(rows: list[dict], field: str) -> dict:
    n = len(rows)
    return {"n": n, "mae": sum(abs(r[field] - r["actual"]) for r in rows) / n if n else None,
            "mape": sum(abs(r[field] - r["actual"]) / r["actual"] * 100 for r in rows) / n if n else None}


def _dm(rows: list[dict], comparator: str, loss: str, lag: int) -> dict:
    differences = []
    for r in rows:
        ce, be = abs(r["candidate"] - r["actual"]), abs(r[comparator] - r["actual"])
        if loss == "mape":
            ce, be = ce / r["actual"] * 100, be / r["actual"] * 100
        differences.append((r["target"], ce - be))
    n = len(differences)
    mean = sum(v for _, v in differences) / n if n else None
    base = {"n": n, "meanLossDifference": mean, "calendarHacMaxLagDays": lag, "statistic": None, "pValue": 1.0}
    if n < MIN_PAIRS:
        return {**base, "status": "inference_unavailable_insufficient_pairs"}
    by_day = {day: value - mean for day, value in differences}
    variance_sum = sum(v * v for v in by_day.values())
    for k in range(1, lag + 1):
        cross = sum(v * by_day.get(day - dt.timedelta(days=k), 0.0) for day, v in by_day.items())
        variance_sum += 2 * (1 - k / (lag + 1)) * cross
    variance_mean = variance_sum / (n * n)
    if not math.isfinite(variance_mean) or variance_mean <= 0:
        return {**base, "status": "inference_unavailable_invalid_hac_variance"}
    statistic = mean / math.sqrt(variance_mean)
    return {**base, "status": "available", "statistic": statistic, "pValue": math.erfc(abs(statistic) / math.sqrt(2))}


def _holm(records: list[dict]) -> None:
    ordered = sorted(range(len(records)), key=lambda i: (records[i]["pValue"], records[i]["product"], records[i]["horizon"], records[i]["comparator"], records[i]["loss"]))
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, (len(records) - rank) * records[index]["pValue"]))
        records[index]["holmAdjustedPValue"] = running
        rec = records[index]
        rec["holmSignificant"] = rec["status"] == "available" and running < ALPHA
        rec["conclusion"] = ("inference_unavailable" if rec["status"] != "available" else
                             "improvement" if rec["holmSignificant"] and rec["meanLossDifference"] < 0 else
                             "worsening" if rec["holmSignificant"] and rec["meanLossDifference"] > 0 else "inconclusive")


def _cell_conclusion(tests: list[dict]) -> str:
    if any(t["status"] != "available" for t in tests):
        return "inference_unavailable"
    if all(t["conclusion"] == "improvement" for t in tests):
        return "improvement"
    if any(t["conclusion"] == "worsening" for t in tests):
        return "worsening"
    return "inconclusive"


def _macro(cells: list[dict]) -> dict:
    result = {"byProduct": {}, "fiveProductEqualWeight": {}}
    for product in PRODUCTS:
        subset = [c for c in cells if c["product"] == product]
        result["byProduct"][product] = {}
        for model in ("candidate", "directRidge", "persistence"):
            ms = [c["metrics"][model] for c in subset if c["metrics"][model]["mape"] is not None]
            result["byProduct"][product][model] = {
                "equalHorizonMape": sum(m["mape"] for m in ms) / len(ms) if ms else None,
                "equalHorizonMae": sum(m["mae"] for m in ms) / len(ms) if ms else None,
                "availableHorizons": len(ms), "totalHorizons": len(HORIZONS)}
    for model in ("candidate", "directRidge", "persistence"):
        values = [result["byProduct"][p][model]["equalHorizonMape"] for p in PRODUCTS]
        mae_values = [result["byProduct"][p][model]["equalHorizonMae"] for p in PRODUCTS]
        complete = (len(values) == len(PRODUCTS) and all(v is not None for v in values)
                    and all(result["byProduct"][p][model]["availableHorizons"] == len(HORIZONS) for p in PRODUCTS))
        mae_complete = (len(mae_values) == len(PRODUCTS) and all(v is not None for v in mae_values)
                        and all(result["byProduct"][p][model]["availableHorizons"] == len(HORIZONS) for p in PRODUCTS))
        result["fiveProductEqualWeight"][model] = {
            "mape": sum(values) / len(values) if complete else None,
            "mae": sum(mae_values) / len(mae_values) if mae_complete else None,
            "availableProducts": sum(v is not None for v in values), "totalProducts": len(PRODUCTS)}
    return result


def _expansion_gate(cells: list[dict], macro: dict, coverage: dict) -> dict:
    cand, direct = macro["fiveProductEqualWeight"]["candidate"], macro["fiveProductEqualWeight"]["directRidge"]
    product_gates = {}
    for p in PRODUCTS:
        c, d = macro["byProduct"][p]["candidate"], macro["byProduct"][p]["directRidge"]
        product_gates[p] = {"mapeWorseningPercentagePoints": c["equalHorizonMape"] - d["equalHorizonMape"] if c["equalHorizonMape"] is not None and d["equalHorizonMape"] is not None else None,
                            "maeWorseningFraction": (c["equalHorizonMae"] / d["equalHorizonMae"] - 1) if c["equalHorizonMae"] is not None and d["equalHorizonMae"] else None}
    checks = {
        "all50CellsAtLeast30NonfallbackAllFourPairs": len(cells) == 50 and all(
            c["coverage"]["nonfallbackAllFourCount"] >= MIN_PAIRS for c in cells),
        "coverageNondeclining": all(v["candidateAvailableCount"] >= v["directRidgeAvailableCount"]
                                     and v["candidateAvailableCount"] >= v["persistenceAvailableCount"]
                                     for v in coverage.values()),
        "macroMapeAtMost10Percent": cand["mape"] is not None and cand["mape"] <= 10,
        "macroMapeImprovesDirectByAtLeast0_5Points": cand["mape"] is not None and direct["mape"] is not None and direct["mape"] - cand["mape"] >= 0.5,
        "eachProductMapeWorseningAtMost1Point": all(v["mapeWorseningPercentagePoints"] is not None and v["mapeWorseningPercentagePoints"] <= 1 for v in product_gates.values()),
        "eachProductEqualHorizonMaeWorseningAtMost5Percent": all(v["maeWorseningFraction"] is not None and v["maeWorseningFraction"] <= 0.05 + 1e-12 for v in product_gates.values()),
    }
    return {"checks": checks, "byProduct": product_gates, "passed": all(checks.values()),
            "action": "full_period_expansion_allowed_by_frozen_gate" if all(checks.values()) else "stop_no_full_period_run"}


def analyze(rows: dict[tuple, dict], hashes: dict[str, str] | None = None) -> dict:
    planned = (END - START).days + 1
    grid = []
    for p in PRODUCTS:
        for h in HORIZONS:
            day = START
            while day <= END:
                key = (p, day - dt.timedelta(days=h), day, h)
                source = rows.get(key)
                if source is None:
                    grid.append({"product": p, "horizon": h, "origin": key[1], "target": day,
                                 "targetObserved": False, "baselineAvailable": False,
                                 "actual": None, "directRidge": None, "persistence": None,
                                 "candidate": None, "allFourForecastsAvailable": False,
                                 "nonfallbackAllFour": False, "fallbackReason": "target_unobserved_in_pinned_reports"})
                else:
                    forecast = _forecast(rows, source)
                    all_four = all(source.get(name) is not None for name in METHODS)
                    row = {"product": p, "horizon": h, "origin": key[1], "target": day,
                           "targetObserved": True, "baselineAvailable": source["baselineAvailable"],
                           "actual": source["actual"],
                           "directRidge": source["directRidge"] if source["baselineAvailable"] else None,
                           "persistence": source["persistence"] if source["baselineAvailable"] else None,
                           "candidate": forecast["candidate"] if source["baselineAvailable"] else None,
                           "allFourForecastsAvailable": all_four,
                           "nonfallbackAllFour": all_four and not forecast["fallback"] and source["baselineAvailable"],
                           "forecastAudit": forecast if source["baselineAvailable"] else None,
                           "fallbackReason": forecast["fallbackReason"] if source["baselineAvailable"] else "direct_baseline_unavailable"}
                    grid.append(row)
                day += dt.timedelta(days=1)

    evaluated = [r for r in grid if r["targetObserved"] and r["baselineAvailable"]
                 and r["directRidge"] is not None and r["persistence"] is not None]
    # Paired scoring uses target-observed rows with direct and persistence baselines.
    comparisons, tests_by_lag, cells, coverage = [], {lag: [] for lag in BANDWIDTHS}, [], {}
    for p in PRODUCTS:
        for h in HORIZONS:
            cell = [r for r in evaluated if r["product"] == p and r["horizon"] == h]
            metrics = {model: _metrics(cell, model) for model in ("candidate", "directRidge", "persistence")}
            planned_cell = [r for r in grid if r["product"] == p and r["horizon"] == h]
            cov = {
                "plannedTargetCount": len(planned_cell),
                "targetObservedCount": sum(r["targetObserved"] for r in planned_cell),
                "directRidgeAvailableCount": sum(r["directRidge"] is not None for r in planned_cell),
                "persistenceAvailableCount": sum(r["persistence"] is not None for r in planned_cell),
                "candidateAvailableCount": sum(r["candidate"] is not None for r in planned_cell),
                "allFourForecastsAvailableCount": sum(r["allFourForecastsAvailable"] for r in planned_cell),
                "nonfallbackAllFourCount": sum(r["nonfallbackAllFour"] for r in planned_cell),
                "pairedScoredCount": len(cell),
                "unobservedCount": sum(not r["targetObserved"] for r in planned_cell),
                "observedWithoutDirectBaselineCount": sum(r["targetObserved"] and not r["baselineAvailable"] for r in planned_cell),
                "missingForecastCounts": {
                    name: sum(r["targetObserved"] and r.get("forecastAudit") is not None
                              and name in r["forecastAudit"].get("missingMethods", []) for r in planned_cell)
                    for name in REPORT_INPUTS
                },
            }
            coverage[f"{p}/{h}"] = cov
            for comparator in COMPARATORS:
                for loss in LOSSES:
                    for lag in BANDWIDTHS:
                        test = {"product": p, "horizon": h, "comparator": comparator, "loss": loss,
                                **_dm(cell, comparator, loss, lag)}
                        tests_by_lag[lag].append(test)
            cells.append({"product": p, "horizon": h, "n": len(cell), "metrics": metrics,
                          "coverage": cov, "conclusionHAC9": None})
    for lag, family in tests_by_lag.items():
        if len(family) != FAMILY_SIZE:
            raise AssertionError(f"expected fixed Holm family of {FAMILY_SIZE}, got {len(family)}")
        _holm(family)
    # Re-link conclusions after Holm adjustment, which is required for directional results.
    for cell in cells:
        relevant = [t for t in tests_by_lag[9] if t["product"] == cell["product"] and t["horizon"] == cell["horizon"]]
        cell["conclusionHAC9"] = _cell_conclusion(relevant)
    for cell in cells:
        relevant = [t for t in tests_by_lag[9] if t["product"] == cell["product"] and t["horizon"] == cell["horizon"]]
        cell["conclusionHAC9"] = _cell_conclusion(relevant)
    macro = _macro(cells)
    gate = _expansion_gate(cells, macro, coverage)
    paired_rows = []
    for r in evaluated:
        audit = r["forecastAudit"] or {}
        paired_rows.append({"product": r["product"], "origin": r["origin"].isoformat(),
                            "target": r["target"].isoformat(), "horizon": r["horizon"],
                            "actual": r["actual"], "candidate": r["candidate"],
                            "directRidge": r["directRidge"], "persistence": r["persistence"],
                            "allFourForecastsAvailable": r["allFourForecastsAvailable"],
                            "nonfallbackAllFour": r["nonfallbackAllFour"], "history": {
                                **audit,
                                "historyTargetStart": audit.get("historyTargetStart").isoformat() if audit.get("historyTargetStart") else None,
                                "historyTargetEnd": audit.get("historyTargetEnd").isoformat() if audit.get("historyTargetEnd") else None}})
    planned_grid = [{"product": r["product"], "horizon": r["horizon"], "origin": r["origin"].isoformat(),
                     "target": r["target"].isoformat(), "targetObserved": r["targetObserved"],
                     "baselineAvailable": r["baselineAvailable"], "candidateAvailable": r["candidate"] is not None,
                     "allFourForecastsAvailable": r["allFourForecastsAvailable"],
                     "nonfallbackAllFour": r["nonfallbackAllFour"], "fallbackReason": r["fallbackReason"]}
                    for r in grid]
    total_target_slots = planned * len(PRODUCTS) * len(HORIZONS)
    return {"schemaVersion": 1, "status": "complete", "protocol": PROTOCOL, "inputSha256": hashes or {},
            "coverage": {"byProductHorizon": coverage, "plannedTargetGridCount": total_target_slots,
                         "targetObservedCount": sum(r["targetObserved"] for r in grid),
                         "directRidgeAvailableCount": sum(r["directRidge"] is not None for r in grid),
                         "persistenceAvailableCount": sum(r["persistence"] is not None for r in grid),
                         "allFourForecastsAvailableCount": sum(r["allFourForecastsAvailable"] for r in grid),
                         "nonfallbackAllFourCount": sum(r["nonfallbackAllFour"] for r in grid),
                         "pairedScoredCount": len(evaluated),
                         "unobservedCount": sum(not r["targetObserved"] for r in grid)},
            "fallbackCount": sum(bool(r.get("forecastAudit") and r["forecastAudit"]["fallback"]) for r in grid),
            "plannedTargetGrid": planned_grid, "pairedRows": paired_rows, "cells": cells, "macro": macro,
            "multipleTesting": {f"HAC{lag}": family for lag, family in tests_by_lag.items()},
            "expansionGate": gate,
            "conclusionCounts": {f"HAC{lag}": {label: sum(t["conclusion"] == label for t in family)
                                                   for label in ("improvement", "worsening", "inconclusive", "inference_unavailable")}
                                 for lag, family in tests_by_lag.items()},
            "limitations": PROTOCOL["scope"]}


def _immutable_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if path.read_bytes() != raw:
            raise FileExistsError(f"refusing to replace different immutable output: {path}")
        return hashlib.sha256(raw).hexdigest()
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def run(*, input_paths: dict[str, Path] = INPUTS, manifest_path: Path, output_path: Path,
        execute: bool = False) -> dict:
    payloads, hashes = {}, {}
    for name in REPORT_INPUTS:
        raw = Path(input_paths[name]).read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        if hashes[name] != EXPECTED_SHA256[name]:
            raise ValueError(f"pinned report SHA-256 mismatch for {name}")
        payloads[name] = json.loads(raw)
    rows = validate_reports(payloads["fixedBlend"], payloads["seasonalAnalog"], payloads["marketRecompose"])
    manifest = {"schemaVersion": 1, "status": "frozen_before_scoring", "inputSha256": hashes, "protocol": PROTOCOL}
    manifest_hash = _immutable_json(Path(manifest_path), manifest)
    if not execute:
        return {"status": "manifest_frozen", "manifest": str(manifest_path), "manifestSha256": manifest_hash}
    result = analyze(rows, hashes)
    result["manifestSha256"] = manifest_hash
    report_hash = _immutable_json(Path(output_path), result)
    return {"status": "complete", "manifest": str(manifest_path), "manifestSha256": manifest_hash,
            "report": str(output_path), "reportSha256": report_hash,
            "expansionGatePassed": result["expansionGate"]["passed"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixed-blend", type=Path, default=INPUTS["fixedBlend"])
    parser.add_argument("--seasonal-analog", type=Path, default=INPUTS["seasonalAnalog"])
    parser.add_argument("--market-recompose", type=Path, default=INPUTS["marketRecompose"])
    parser.add_argument("--manifest", type=Path, default=ROOT / "outputs/experiments/past-error-ensemble-20261008/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/past-error-ensemble-20261008/report.json")
    parser.add_argument("--execute", action="store_true", help="score only the frozen 2026-08-01..09-24 pilot")
    args = parser.parse_args(argv)
    result = run(input_paths={"fixedBlend": args.fixed_blend, "seasonalAnalog": args.seasonal_analog,
                              "marketRecompose": args.market_recompose}, manifest_path=args.manifest,
                 output_path=args.output, execute=args.execute)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
