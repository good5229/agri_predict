"""Pure scorer for the frozen Chronos-2 market-context comparison.

The caller supplies an answer-free forecast manifest and separately validated
actual prices.  This module performs no collection, inference, model loading,
fitting, or artifact writes.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import Counter, defaultdict
from typing import Any, Mapping

from . import dashboard, equal_info_online_score as inference
from .research_progress_diagnostic import circular_block_bootstrap

PRODUCTS = tuple(dashboard.PRODUCTS)[:4]
HORIZONS = tuple(range(1, 11))
CONDITIONS = ("B", "U", "M", "H")
LOSSES = ("ae", "ape")
BANDWIDTHS = (9, 28)
BOOTSTRAP_BLOCKS = (14, 28)
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 0
TARGET_WINDOWS = {
    "V": (dt.date(2026, 4, 1), dt.date(2026, 6, 20)),
    "T": (dt.date(2026, 7, 11), dt.date(2026, 9, 27)),
}
ORIGIN_CUTOFFS = {"V": dt.date(2026, 4, 1), "T": dt.date(2026, 7, 1)}
COMPARISONS = (("U_vs_B", "U", "B"), ("M_vs_U", "M", "U"), ("M_vs_B", "M", "B"))


def _day(value: Any) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        return dt.date.fromisoformat(value)
    raise ValueError(f"invalid forecast date: {value!r}")


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _key(row: Mapping[str, Any]) -> tuple[str, dt.date, dt.date, int]:
    try:
        ident = row.get("key", row)
        product = ident["product"]
        origin = _day(ident.get("originDate", ident.get("origin")))
        target = _day(ident.get("targetDate", ident.get("target")))
        horizon = ident["horizon"]
    except (KeyError, TypeError) as exc:
        raise ValueError("forecast row lacks a complete product/origin/target/horizon key") from exc
    if product not in PRODUCTS or type(horizon) is not int or horizon not in HORIZONS:
        raise ValueError("forecast key is outside the fixed four-product/ten-horizon domain")
    if target - origin != dt.timedelta(days=horizon):
        raise ValueError("forecast origin/target/horizon do not agree")
    return product, origin, target, horizon


def _prediction(item: Any) -> tuple[str, float | None, str | None]:
    if not isinstance(item, Mapping):
        return "missing", None, "missing_condition_record"
    status = str(item.get("status", "missing"))
    reason = item.get("reason")
    if status != "issued":
        return status, None, str(reason) if reason is not None else None
    value = _positive(item.get("prediction"))
    if value is None:
        return "invalid_prediction", None, str(reason) if reason is not None else "issued_value_not_finite_positive"
    return status, value, str(reason) if reason is not None else None


def _month_labels(phase: str) -> list[str]:
    start, end = TARGET_WINDOWS[phase]
    labels = []
    day = start.replace(day=1)
    while day <= end:
        labels.append(day.strftime("%Y-%m"))
        day = (day.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return labels


def _loss(actual: float, prediction: float, loss: str) -> float:
    error = abs(actual - prediction)
    return error if loss == "ae" else 100.0 * error / actual


def _metrics(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "leftMae": None, "rightMae": None, "maeDifference": None,
                "leftMape": None, "rightMape": None, "mapeDifference": None}
    mae_l = math.fsum(abs(r[left] - r["actual"]) for r in rows) / len(rows)
    mae_r = math.fsum(abs(r[right] - r["actual"]) for r in rows) / len(rows)
    mape_l = math.fsum(100 * abs(r[left] - r["actual"]) / r["actual"] for r in rows) / len(rows)
    mape_r = math.fsum(100 * abs(r[right] - r["actual"]) / r["actual"] for r in rows) / len(rows)
    return {"n": len(rows), "leftMae": mae_l, "rightMae": mae_r, "maeDifference": mae_l - mae_r,
            "leftMape": mape_l, "rightMape": mape_r, "mapeDifference": mape_l - mape_r}


def _metrics_by(rows: list[dict[str, Any]], left: str, right: str, *, phase: str) -> list[dict[str, Any]]:
    """Descriptive paired means by overall, product, horizon, and target month."""
    out = [{"scope": "overall", **_metrics(rows, left, right)}]
    for product in PRODUCTS:
        part = [r for r in rows if r["product"] == product]
        out.append({"scope": "product", "product": product, **_metrics(part, left, right)})
    for horizon in HORIZONS:
        part = [r for r in rows if r["horizon"] == horizon]
        out.append({"scope": "horizon", "horizon": horizon, **_metrics(part, left, right)})
    for product in PRODUCTS:
        for horizon in HORIZONS:
            part = [r for r in rows if r["product"] == product and r["horizon"] == horizon]
            out.append({"scope": "product_horizon", "product": product, "horizon": horizon,
                        **_metrics(part, left, right)})
    for month in _month_labels(phase):
        part = [r for r in rows if r["target"].strftime("%Y-%m") == month]
        out.append({"scope": "target_month", "targetMonth": month, **_metrics(part, left, right)})
        for product in PRODUCTS:
            local = [r for r in part if r["product"] == product]
            out.append({"scope": "product_target_month", "product": product, "targetMonth": month,
                        **_metrics(local, left, right)})
        for horizon in HORIZONS:
            local = [r for r in part if r["horizon"] == horizon]
            out.append({"scope": "horizon_target_month", "horizon": horizon, "targetMonth": month,
                        **_metrics(local, left, right)})
    return out


def _primary_strata(paired: Mapping[tuple[str, int], list[dict[str, Any]]], phase: str) -> list[dict[str, Any]]:
    """H-B summaries average within cell, then equally across the fixed cell set."""
    all_cells = [(p, h) for p in PRODUCTS for h in HORIZONS]
    months = _month_labels(phase)
    result = []

    def emit(scope: str, cells: list[tuple[str, int]], *, month: str | None = None,
             product: str | None = None, horizon: int | None = None) -> None:
        per_cell = {}
        for cell in cells:
            rows = paired.get(cell, [])
            if month is not None:
                rows = [r for r in rows if r["target"].strftime("%Y-%m") == month]
            per_cell[cell] = _metrics(rows, "H", "B")
        supported = {cell: item for cell, item in per_cell.items() if item["n"] > 0}
        complete = len(supported) == len(cells) and bool(cells)
        fields = ("leftMae", "rightMae", "maeDifference", "leftMape", "rightMape", "mapeDifference")
        aggregate = {name: (math.fsum(supported[c][name] for c in cells) / len(cells)
                            if complete else None) for name in fields}
        row = {"scope": scope, "expectedCellCount": len(cells), "supportedCellCount": len(supported),
               "pairedKeyCount": sum(item["n"] for item in per_cell.values()),
               "completeSupport": complete, **aggregate,
               "cellPairCounts": {f"{p}/{h}": per_cell[(p, h)]["n"] for p, h in cells}}
        if product is not None:
            row["product"] = product
        if horizon is not None:
            row["horizon"] = horizon
        if month is not None:
            row["targetMonth"] = month
        result.append(row)

    emit("overall_40_equal_cell", all_cells)
    for product in PRODUCTS:
        emit("product_10_equal_horizon", [(product, h) for h in HORIZONS], product=product)
    for horizon in HORIZONS:
        emit("horizon_4_equal_product", [(p, horizon) for p in PRODUCTS], horizon=horizon)
    for month in months:
        emit("target_month_40_equal_cell", all_cells, month=month)
        for product in PRODUCTS:
            emit("product_target_month_10_equal_horizon", [(product, h) for h in HORIZONS],
                 month=month, product=product)
        for horizon in HORIZONS:
            emit("horizon_target_month_4_equal_product", [(p, horizon) for p in PRODUCTS],
                 month=month, horizon=horizon)
        for product, horizon in all_cells:
            emit("product_horizon_target_month_cell", [(product, horizon)], month=month,
                 product=product, horizon=horizon)
    return result


def _forecast_rows(doc: Mapping[str, Any], phase: str) -> dict[tuple[str, dt.date, dt.date, int], dict[str, Any]]:
    if phase not in TARGET_WINDOWS:
        raise ValueError("phase must be V or T")
    forecasts = doc.get("forecasts")
    if not isinstance(forecasts, list) or not forecasts:
        raise ValueError("manifest requires nonempty forecasts list")
    start, end = TARGET_WINDOWS[phase]
    cutoff = ORIGIN_CUTOFFS[phase]
    by_key = {}
    for row in forecasts:
        if not isinstance(row, Mapping):
            raise ValueError("forecast row must be an object")
        key = _key(row)
        product, origin, target, horizon = key
        if target < start or target > end or origin < cutoff:
            raise ValueError(f"forecast key outside frozen {phase} target/origin window: {key}")
        if key in by_key:
            raise ValueError(f"duplicate B-eligible issue key: {key}")
        conditions = row.get("conditions")
        if not isinstance(conditions, Mapping):
            raise ValueError(f"forecast row has no conditions map: {key}")
        # Each row is explicitly the frozen B-eligible universe; never infer it from U/M/H.
        b_status, b_value, _ = _prediction(conditions.get("B"))
        if b_status != "issued" or b_value is None:
            raise ValueError(f"B-eligible key is not a valid issued B forecast: {key}")
        values = {"B": b_value}
        states = {"B": b_status}
        reasons = {"B": None}
        for condition in ("U", "M", "H"):
            status, value, reason = _prediction(conditions.get(condition))
            values[condition], states[condition], reasons[condition] = value, status, reason
        m_value, h_value = values["M"], values["H"]
        if states["H"] == "issued":
            if m_value is None:
                raise ValueError(f"H is issued despite M being unavailable: {key}")
            expected = 0.5 * b_value + 0.5 * m_value
            if h_value is None or abs(h_value - expected) > 1e-8 + 1e-10 * abs(expected):
                raise ValueError(f"H is not the frozen 0.5 B + 0.5 M price blend: {key}")
        elif m_value is None and states["H"] == "issued":
            raise ValueError(f"H fallback is prohibited when M is unavailable: {key}")
        by_key[key] = {"key": key, "product": product, "origin": origin, "target": target,
                       "horizon": horizon, "values": values, "states": states, "reasons": reasons}
    return by_key


def score_manifest(doc: Mapping[str, Any], *, actuals: Mapping[tuple[str, Any], Any], phase: str) -> dict[str, Any]:
    """Score saved B/U/M/H forecasts; actuals must be separately source-validated."""
    by_key = _forecast_rows(doc, phase)
    actual_map: dict[tuple[str, dt.date], float] = {}
    for (product, target), value in actuals.items():
        if product not in PRODUCTS:
            continue
        price = _positive(value)
        if price is not None:
            actual_map[(product, _day(target))] = price

    keys_by_cell: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    paired: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    coverage = []
    for row in by_key.values():
        product, target, horizon = row["product"], row["target"], row["horizon"]
        cell = product, horizon
        keys_by_cell[cell].append(row)
        actual = actual_map.get((product, target))
        row["actual"] = actual
        if actual is not None:
            if row["values"]["B"] is not None and row["values"]["H"] is not None:
                paired[cell].append({"product": product, "origin": row["origin"], "target": target,
                                     "horizon": horizon, "actual": actual,
                                     "B": row["values"]["B"], "H": row["values"]["H"]})
        statuses = Counter(row["states"][c] for c in ("U", "M", "H"))
        actual_available = actual is not None
        coverage.append({"product": product, "originDate": row["origin"].isoformat(),
                         "targetDate": target.isoformat(), "horizon": horizon,
                         "actualAvailable": actual_available, "BIssued": True,
                         "UIssued": row["values"]["U"] is not None,
                         "MIssued": row["values"]["M"] is not None,
                         "HIssued": row["values"]["H"] is not None,
                         "UStatus": row["states"]["U"], "MStatus": row["states"]["M"],
                         "HStatus": row["states"]["H"], "HReason": row["reasons"]["H"],
                         "statusCounts": dict(statuses)})

    cells = []
    dm = []
    fixed_cells = {(p, h) for p in PRODUCTS for h in HORIZONS}
    for product in PRODUCTS:
        for horizon in HORIZONS:
            cell = (product, horizon)
            universe = keys_by_cell.get(cell, [])
            rows = paired.get(cell, [])
            issued_h = [r for r in universe if r["values"]["H"] is not None]
            b_eval = [r for r in universe if r["actual"] is not None]
            h_eval = [r for r in universe if r["actual"] is not None and r["values"]["H"] is not None]
            metrics = _metrics(rows, "H", "B")
            cells.append({"product": product, "horizon": horizon,
                          "bEligibleIssuedKeys": len(universe), "bIssuedKeys": len(universe),
                          "hIssuedKeys": len(issued_h), "issuedCoverageLoss": len(universe)-len(issued_h),
                          "bPositiveActualKeys": len(b_eval), "hPositiveActualKeys": len(h_eval),
                          "evaluationCoverageLoss": len(b_eval)-len(h_eval), **metrics})
            identical = bool(rows) and all(r["H"] == r["B"] for r in rows)
            for loss in LOSSES:
                diffs = {r["target"]: _loss(r["actual"], r["H"], loss) - _loss(r["actual"], r["B"], loss)
                         for r in rows}
                if len(diffs) != len(rows):
                    raise ValueError("duplicate product/horizon target makes calendar HAC ambiguous")
                for bandwidth in BANDWIDTHS:
                    if identical and len(diffs) >= 30:
                        test = {"status": "identical_predictions", "reason": "H_equals_B_for_every_common_pair",
                                "n": len(diffs), "meanLossDifference": math.fsum(diffs.values())/len(diffs),
                                "statistic": None, "pValue": 1.0}
                    else:
                        test = inference._dm(diffs, bandwidth)
                    dm.append({"comparison": "H_vs_B", "product": product, "horizon": horizon,
                               "loss": loss, "unit": "price_per_kg" if loss == "ae" else "percentage_points",
                               "bandwidth": bandwidth, "samePrediction": identical, **test})
    for bandwidth in BANDWIDTHS:
        family = [r for r in dm if r["bandwidth"] == bandwidth]
        if len(family) != 80:
            raise AssertionError("fixed Chronos H-B family must contain 80 tests per bandwidth")
        inference._holm(family)

    present_all_cells = set(paired) == fixed_cells and all(paired.get(c) for c in fixed_cells)
    gate_support = all(paired.get(c) for c in fixed_cells)
    macro = None
    if gate_support:
        macro = {name: math.fsum(c[name] for c in cells if c[name] is not None) / 40
                 for name in ("leftMae", "rightMae", "maeDifference", "leftMape", "rightMape", "mapeDifference")}
    coverage_ok = all(c["issuedCoverageLoss"] == 0 and c["evaluationCoverageLoss"] == 0 for c in cells)
    gate = {"status": "evaluated" if gate_support else "inference_or_metric_support_incomplete",
            "all40CellsSupported": gate_support,
            "hMapeStrictlyLower": bool(gate_support and macro["leftMape"] < macro["rightMape"]),
            "hMaeNoHigher": bool(gate_support and macro["leftMae"] <= macro["rightMae"]),
            "noIssuedOrEvaluationCoverageLoss": coverage_ok,
            "passed": bool(gate_support and coverage_ok and macro["leftMape"] < macro["rightMape"]
                            and macro["leftMae"] <= macro["rightMae"]),
            "thresholds": {"equalWeightCells": 40, "mape": "H < B, unrounded", "mae": "H <= B, unrounded",
                            "coverageLossAllowed": 0}}

    primary_strata = _primary_strata(paired, phase)
    descriptive = []
    for comparison, left, right in COMPARISONS:
        rows = []
        for row in by_key.values():
            a = row["values"].get(left)
            b = row["values"].get(right)
            actual = actual_map.get((row["product"], row["target"]))
            if a is not None and b is not None and actual is not None:
                rows.append({"product": row["product"], "horizon": row["horizon"],
                             "target": row["target"], "actual": actual, left: a, right: b})
        descriptive.append({"comparison": comparison,
                            "aggregation": "paired-row-pooled for overall/month; cell-specific for product-horizon",
                            "paired": _metrics_by(rows, left, right, phase=phase),
                            "pairedKeys": len(rows), "scope": "descriptive_only_no_inference_family"})

    start, end = TARGET_WINDOWS[phase]
    bootstrap_rows = [dict(r, target=r["target"].isoformat())
                      for cell_rows in paired.values() for r in cell_rows]
    bootstrap = []
    for block in BOOTSTRAP_BLOCKS:
        estimates = {}
        for loss in LOSSES:
            if not present_all_cells:
                result = {"blockDays": block, "requested": BOOTSTRAP_REPS, "valid": 0,
                          "invalid": BOOTSTRAP_REPS, "interval95": None,
                          "status": "unavailable_missing_fixed_cell"}
            else:
                raw = circular_block_bootstrap(bootstrap_rows, a="H", b="B", metric=loss,
                    block=block, replicates=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED,
                    calendar_start=start, calendar_end=end)
                result = {**raw, "status": "available" if raw.get("interval95") is not None else "all_replicates_invalid"}
            estimates[loss] = result
        bootstrap.append({"blockDays": block, "seed": BOOTSTRAP_SEED, "replicates": BOOTSTRAP_REPS,
                          "comparisons": {"H_vs_B": estimates},
                          "scope": "joint circular calendar-date draws on fixed 40-cell H-B support; missing cells never change the grid"})

    return {"schemaVersion": 1, "phase": phase, "targetWindow": [start.isoformat(), end.isoformat()],
            "originCutoff": ORIGIN_CUTOFFS[phase].isoformat(),
            "primaryComparison": "H_vs_B", "candidate": "H=0.5*B+0.5*M",
            "forecastKeyCount": len(by_key), "primaryCoverage": coverage,
            "cells": cells, "fortyCellEqualWeight": macro, "validationGate": gate,
            "primaryEqualCellStrata": primary_strata,
            "fixedDMFamily": {"hypothesisCountPerBandwidth": 80, "comparisons": ["H_vs_B"],
                              "products": list(PRODUCTS), "horizons": list(HORIZONS),
                              "losses": list(LOSSES), "bandwidths": list(BANDWIDTHS), "holmAlpha": 0.05,
                              "tests": dm, "summary": [{"bandwidth": bw,
                                  "tested": sum(r["status"] == "tested" and r["bandwidth"] == bw for r in dm),
                                  "unavailable": sum(r["status"] != "tested" and r["bandwidth"] == bw for r in dm),
                                  "holmSignificantImprovement": sum(r["bandwidth"] == bw and r["status"] == "tested" and r.get("holmSignificant") and r["meanLossDifference"] < 0 for r in dm),
                                  "holmSignificantWorsening": sum(r["bandwidth"] == bw and r["status"] == "tested" and r.get("holmSignificant") and r["meanLossDifference"] > 0 for r in dm)} for bw in BANDWIDTHS]},
            "bootstrap": bootstrap, "descriptiveComparisons": descriptive,
            "limitations": ["V/T are repeatedly inspected retrospective development periods, not independent or future performance evidence.",
                            "U-B and M-U/M-B are descriptive diagnostics; they do not isolate pretraining or market causality."]}
