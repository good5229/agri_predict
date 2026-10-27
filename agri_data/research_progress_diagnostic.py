"""Retrospective error-structure diagnostics on frozen forecast artifacts.

This module never fits a model or accesses a network.  It reads only pinned
forecast manifests, their existing scored summaries, and the frozen price
snapshot.  Its bootstrap is descriptive and does not replace the original DM
families.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

from . import equal_info_online_forecasts as online
from . import equal_info_online_score as scorer
from . import countertrend_guard as countertrend_protocol
from . import path_mlp_forecasts as path_mlp_protocol
from . import preserved_feature_forecasts as preserved_protocol
from . import preserved_feature_score as preserved_scorer
from . import trend_response_correction as trend_protocol
from . import four_crop_forecast as four_crop_protocol

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/research-progress-diagnostic-20261011"
PRICE_PATH = online.PRICES_PATH
PRODUCTS = ("cabbage", "radish", "onion", "garlic")

SOURCES = {
    "online_manifest": ROOT / "outputs/experiments/equal-info-online-20261010/forecast-manifest.json",
    "online_report": ROOT / "outputs/experiments/equal-info-online-20261010/evaluation.json",
    "trend_confirmation_manifest": ROOT / "outputs/experiments/trend-response-correction-20261010/confirmation-manifest.json",
    "trend_confirmation_report": ROOT / "outputs/experiments/trend-response-correction-20261010/confirmation-report.json",
    "trend_final_manifest": ROOT / "outputs/experiments/trend-response-correction-20261010/final-manifest.json",
    "trend_final_report": ROOT / "outputs/experiments/trend-response-correction-20261010/final-report.json",
    "preserved_confirmation_manifest": ROOT / "outputs/experiments/preserved-feature-20261010/confirmation-manifest.json",
    "preserved_confirmation_report": ROOT / "outputs/experiments/preserved-feature-20261010/confirmation-report.json",
    "preserved_evaluation_manifest": ROOT / "outputs/experiments/preserved-feature-20261010/evaluation-manifest.json",
    "preserved_evaluation_report": ROOT / "outputs/experiments/preserved-feature-20261010/evaluation-report.json",
    "countertrend_confirmation_manifest": ROOT / "outputs/experiments/countertrend-guard-20261010/confirmation-manifest.json",
    "countertrend_confirmation_report": ROOT / "outputs/experiments/countertrend-guard-20261010/confirmation-report.json",
    "four_crop_forecast": ROOT / "outputs/experiments/four-crop-forecast-20261010/forecast.json",
    "four_crop_fixed_prices": four_crop_protocol.LIVE_PRICES,
    "path_mlp_screen_manifest": ROOT / "outputs/experiments/path-mlp-20261010/screen-manifest.json",
}

FROZEN_FILE_SHA256 = {
    "online_manifest": "d7d74823754c2a48d33cc4ddaeab674c686601fb927d399d84c5dbcd235184df",
    "online_report": "da9a2f5c69c7ae9da503e60e10f0501ff7f5eb18ec60e9d5596f05e981fc7412",
    "trend_confirmation_manifest": "bb929cadbdee4862b71ac93f97a0ecbed9ea7ecb3ff78ac5251e281bdea5d74b",
    "trend_confirmation_report": "9f7d90fc2ef4b748577fe3637ee255da6e21ead859a735702e9665f46e1a2f55",
    "trend_final_manifest": "7ec6453660e020298bf407b829fdda9c8807c739769b5e58abc4d04631662502",
    "trend_final_report": "a2254f9fb590620f0855cae8187619f993863cfb336286bb4178d5d1d8070b8a",
    "preserved_confirmation_manifest": "e7686eace6a5500964823f781dd8faf93ff0fc17005c9b76513e1f0198d8b04c",
    "preserved_confirmation_report": "9c76e2e46013e434400fe3a9036fe91dcdf3af4816d923cf1c2adec045833241",
    "preserved_evaluation_manifest": "35164a11143768f26e7567e80643ec323c7279083ac5e6ef166a7fe8d185aa20",
    "preserved_evaluation_report": "d4fbb06a983e95159aeae5fef095a280f996543d022ecf65c462709ebbcb4708",
    "countertrend_confirmation_manifest": "57d28f2e80783fa8beb5400fd5ca45ffdbd9cd5468c2937cf07580de595b0ae0",
    "countertrend_confirmation_report": "5e5e1ab9c3140badeb399cbddd62250c1bcaa3198b1be455073acb534e80a0bf",
    "four_crop_forecast": "e86cc3577b3d2d078badbfee1c903841d3f936ff751d8765935a12bdc019ba4b",
    "four_crop_fixed_prices": "3e9eca5e1da69812fb3e82894dc1c143af1bc752a33740cc03850d1439792691",
    "path_mlp_screen_manifest": "a4c093520d9412a1f892c64583e8cda3bcfdc11f594e03ccf55e20df148052a8",
    "prices": "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8",
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _positive(value):
    if isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _loss(row, key, metric):
    error = abs(row[key] - row["actual"])
    return error if metric == "ae" else 100.0 * error / row["actual"]


def _cell_means(rows, a, b, metric, expected_cells=None):
    """Equal weight for original cells, equal weight within each cell."""
    cells = defaultdict(list)
    for row in rows:
        cells[(row["product"], row["horizon"])].append(row)
    expected_cells = set(expected_cells) if expected_cells is not None else set(cells)
    if not expected_cells or set(cells) != expected_cells or any(not values for values in cells.values()):
        return None
    per_cell = {}
    for cell, values in cells.items():
        per_cell[cell] = {
            "n": len(values),
            "a": math.fsum(_loss(v, a, metric) for v in values) / len(values),
            "b": math.fsum(_loss(v, b, metric) for v in values) / len(values),
        }
    ncell = len(per_cell)
    return {"cells": per_cell,
            "a": math.fsum(v["a"] for v in per_cell.values()) / ncell,
            "b": math.fsum(v["b"] for v in per_cell.values()) / ncell,
            "difference": math.fsum(v["a"] - v["b"] for v in per_cell.values()) / ncell}


def decompose_losses(rows, a="a", b="b", metric="ae", bottom_price_decile=True, expected_cells=None):
    """Return contribution decomposition, concentration and leave-period sensitivity.

    A contribution is signed loss(A)-loss(B), scaled by the original cell
    count and that cell's paired sample size. Thus all target contributions
    sum exactly to the original equal-cell mean loss difference.
    """
    if metric not in ("ae", "ape"):
        raise ValueError("metric must be ae or ape")
    cell_counts = defaultdict(int)
    for r in rows:
        cell_counts[(r["product"], r["horizon"])] += 1
    if not cell_counts:
        return {"n": 0, "meanDifference": None, "byTarget": [], "concentration": {}, "leaveOut": {}}
    n_cells = len(expected_cells or cell_counts)
    if expected_cells is not None and set(cell_counts) != set(expected_cells):
        raise ValueError("paired comparison does not cover the frozen original cell family")
    contributions = defaultdict(float)
    raw = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        day = dt.date.fromisoformat(r["target"])
        delta = _loss(r, a, metric) - _loss(r, b, metric)
        weight = 1.0 / (n_cells * cell_counts[(r["product"], r["horizon"])])
        contributions[day] += delta * weight
        raw[day][0] += max(0.0, delta) * weight
        raw[day][1] += max(0.0, -delta) * weight
    loss_a = defaultdict(float)
    loss_b = defaultdict(float)
    target_price = {}
    for r in rows:
        day = dt.date.fromisoformat(r["target"])
        w = 1.0 / (n_cells * cell_counts[(r["product"], r["horizon"])])
        loss_a[day] += _loss(r, a, metric) * w
        loss_b[day] += _loss(r, b, metric) * w
        target_price[(r["product"], day)] = r["actual"]
    by_target = [{"target": d.isoformat(), "lossA": loss_a[d], "lossB": loss_b[d],
                  "contribution": contributions[d], "positive": raw[d][0], "negative": raw[d][1]}
                 for d in sorted(contributions)]
    total = math.fsum(contributions.values())
    ranked = sorted(by_target, key=lambda x: (-abs(x["contribution"]), x["target"]))
    concentration = {}
    for top in (1, 5, 10):
        chosen = ranked[:top]
        concentration[str(top)] = {
            "targets": [r["target"] for r in chosen],
            "netContribution": math.fsum(r["contribution"] for r in chosen),
            "positiveContribution": math.fsum(r["positive"] for r in chosen),
            "negativeContribution": math.fsum(r["negative"] for r in chosen),
        }
    for grain, keyfn in (("month", lambda d: d.strftime("%Y-%m")),
                         ("isoWeek", lambda d: f"{d.isocalendar().year}-W{d.isocalendar().week:02d}")):
        buckets = defaultdict(lambda: {"lossA": 0.0, "lossB": 0.0, "contribution": 0.0})
        for d, value in contributions.items():
            label = keyfn(d)
            buckets[label]["lossA"] += loss_a[d]
            buckets[label]["lossB"] += loss_b[d]
            buckets[label]["contribution"] += value
        concentration[grain] = [{"period": k, **buckets[k]} for k in sorted(buckets)]
    leave = {"month": {}, "isoWeek": {}}
    for grain, keyfn in (("month", lambda d: d.strftime("%Y-%m")),
                         ("isoWeek", lambda d: f"{d.isocalendar().year}-W{d.isocalendar().week:02d}")):
        for label in sorted({keyfn(d) for d in contributions}):
            selected = [r for r in rows if keyfn(dt.date.fromisoformat(r["target"])) != label]
            estimate = _cell_means(selected, a, b, metric, expected_cells=cell_counts)
            leave[grain][label] = ({"difference": None, "reason": "at_least_one_original_cell_empty"}
                                   if estimate is None else {"difference": estimate["difference"], "reason": None})
    # Accounting invariant: this decomposition must recover the cell macro.
    macro = _cell_means(rows, a, b, metric)
    if not math.isclose(total, macro["difference"], rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError("target contributions do not sum to original cell macro")
    bottom = None
    if bottom_price_decile:
        prices_by_product = defaultdict(dict)
        for (product, day), value in target_price.items():
            prices_by_product[product][day] = value
        cutoffs = {}
        event_dates = {}
        for product, values in prices_by_product.items():
            ordered = sorted((v, d) for d, v in values.items())
            count = max(1, math.ceil(.10 * len(ordered)))
            threshold = ordered[count-1][0]
            cutoffs[product] = threshold
            event_dates[product] = {d for v, d in ordered[:count]}
        subgroup = {"bottomDecileByProduct": {}}
        for label, predicate in (("bottom10Percent", lambda r: dt.date.fromisoformat(r["target"]) in event_dates.get(r["product"], set())),
                                 ("remaining90Percent", lambda r: dt.date.fromisoformat(r["target"]) not in event_dates.get(r["product"], set()))):
            sums = {"lossA": 0.0, "lossB": 0.0, "difference": 0.0, "n": 0}
            for r in rows:
                if not predicate(r):
                    continue
                d = dt.date.fromisoformat(r["target"])
                w = 1.0 / (n_cells * cell_counts[(r["product"], r["horizon"])])
                sums["lossA"] += _loss(r, a, metric) * w
                sums["lossB"] += _loss(r, b, metric) * w
                sums["difference"] += (_loss(r, a, metric) - _loss(r, b, metric)) * w
                sums["n"] += 1
            subgroup["bottomDecileByProduct"][label] = sums
        subgroup["thresholdsByProduct"] = cutoffs
        subgroup["accountingVerified"] = math.isclose(
            subgroup["bottomDecileByProduct"]["bottom10Percent"]["difference"] +
            subgroup["bottomDecileByProduct"]["remaining90Percent"]["difference"], total,
            rel_tol=1e-10, abs_tol=1e-10)
        bottom = subgroup
    return {"n": len(rows), "cellCount": n_cells, "meanDifference": total,
            "byTarget": by_target, "concentration": concentration,
            "leaveOut": leave, "accountingVerified": True, "bottomPriceDecile": bottom}


def circular_block_bootstrap(rows, *, a="a", b="b", metric="ae", block=14,
                             replicates=2000, seed=20261011, calendar_start=None,
                             calendar_end=None):
    """Joint-date circular calendar-block bootstrap; no missing-date filling."""
    if block < 1 or replicates < 1:
        raise ValueError("block and replicates must be positive")
    import random
    dates = sorted({dt.date.fromisoformat(r["target"]) for r in rows})
    if not dates:
        return {"blockDays": block, "calendarDays": 0, "requested": replicates, "valid": 0,
                "invalid": replicates, "interval95": None}
    by_date = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for row in rows:
        day = dt.date.fromisoformat(row["target"])
        cell = (row["product"], row["horizon"])
        by_date[day][cell][0] += _loss(row, a, metric) - _loss(row, b, metric)
        by_date[day][cell][1] += 1
    cells = {(r["product"], r["horizon"]) for r in rows}
    rng = random.Random(seed)
    estimates = []
    invalid = 0
    base = calendar_start or dates[0]
    end = calendar_end or dates[-1]
    span = (end - base).days + 1
    if span < 1 or dates[0] < base or dates[-1] > end:
        raise ValueError("paired targets fall outside the frozen bootstrap calendar")
    # Sampling calendar dates joint across all cells; absent calendar dates
    # remain absent and any replicate missing a cell is invalid.
    for _ in range(replicates):
        totals = defaultdict(float)
        counts = defaultdict(int)
        selected_count = 0
        while selected_count < span:
            start = rng.randrange(span)
            for offset in range(block):
                d = base + dt.timedelta(days=(start + offset) % span)
                selected_count += 1
                if d in by_date:
                    for cell, (subtotal, n) in by_date[d].items():
                        totals[cell] += subtotal
                        counts[cell] += n
                if selected_count >= span:
                    break
        if set(counts) != cells or any(counts[c] == 0 for c in cells):
            invalid += 1
            continue
        estimate = math.fsum(totals[c] / counts[c] for c in cells) / len(cells)
        if not math.isfinite(estimate):
            invalid += 1
        else:
            estimates.append(estimate)
    if estimates:
        ordered = sorted(estimates)
        lo = ordered[max(0, math.ceil(.025 * len(ordered)) - 1)]
        hi = ordered[max(0, math.ceil(.975 * len(ordered)) - 1)]
        interval = [lo, hi]
    else:
        interval = None
    return {"blockDays": block, "calendarDays": span, "requested": replicates, "valid": len(estimates),
            "invalid": invalid, "seed": seed, "interval95": interval,
            "interpretation": "descriptive retrospective sensitivity only"}


def _verify_saved_metrics(pairs, saved_metrics, *, comparison=None, product=None,
                          a_fields=("aMae", "aMape"), b_fields=("bMae", "bMape")):
    """Fail closed unless exact paired cell counts and metrics reproduce."""
    expected_rows = [m for m in saved_metrics
                     if (comparison is None or m.get("comparison") == comparison)
                     and (product is None or m.get("product") == product)]
    if not expected_rows:
        raise ValueError("required saved metric cells are missing")
    pairs_by_cell = defaultdict(list)
    for row in pairs:
        pairs_by_cell[(row["product"], row["horizon"])].append(row)
    seen = set()
    for saved in expected_rows:
        cell = (saved.get("product"), saved.get("horizon"))
        if cell in seen:
            raise ValueError("saved metric file contains duplicate comparison cells")
        seen.add(cell)
        values = pairs_by_cell.get(cell, [])
        if saved.get("n") != len(values):
            raise ValueError(f"paired count mismatch for frozen cell {cell}")
        if not values:
            raise ValueError(f"frozen source has an empty comparison cell {cell}")
        mae_a = math.fsum(abs(r["a"]-r["actual"]) for r in values)/len(values)
        mae_b = math.fsum(abs(r["b"]-r["actual"]) for r in values)/len(values)
        mape_a = 100*math.fsum(abs(r["a"]-r["actual"])/r["actual"] for r in values)/len(values)
        mape_b = 100*math.fsum(abs(r["b"]-r["actual"])/r["actual"] for r in values)/len(values)
        for key, expected, observed in ((a_fields[0], saved.get(a_fields[0]), mae_a),
                                        (b_fields[0], saved.get(b_fields[0]), mae_b),
                                        (a_fields[1], saved.get(a_fields[1]), mape_a),
                                        (b_fields[1], saved.get(b_fields[1]), mape_b)):
            if expected is None or not math.isclose(expected, observed, rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError(f"saved {key} mismatch for frozen cell {cell}")
    if set(pairs_by_cell) != seen:
        raise ValueError("reconstructed paired set has cells outside saved comparison family")
    return {"cells": len(seen), "pairs": len(pairs), "reproduced": True}


def classify_countertrend(rows, expected_cell_count=None, expected_cells=None):
    """Classify applied shrink rows by direction and movement magnitude."""
    counts = defaultdict(int)
    sums = defaultdict(lambda: {"aeNet": 0.0, "aeGain": 0.0, "aeHarm": 0.0,
                                "apeNet": 0.0, "apeGain": 0.0, "apeHarm": 0.0, "weight": 0.0})
    cell_n = defaultdict(int)
    all_losses = {"ae": {"E": 0.0, "C": 0.0}, "ape": {"E": 0.0, "C": 0.0}}
    active_totals = {"ae": 0.0, "ape": 0.0}
    for r in rows:
        if r.get("product") and r.get("horizon"):
            cell_n[(r["product"], r["horizon"])] += 1
    cell_count = expected_cell_count or len(cell_n)
    if expected_cells is not None and set(cell_n) != set(expected_cells):
        raise ValueError("countertrend paired cohort does not cover its frozen cell family")
    for r in rows:
        p0, e, c, actual = map(_positive, (r.get("p0"), r.get("baselinePrediction"),
                                           r.get("model", {}).get("prediction"), r.get("actual")))
        if None in (e, c, actual):
            counts["unscorable"] += 1
            continue
        cell = (r["product"], r["horizon"])
        weight = 1.0 / (cell_count * cell_n[cell]) if cell_count and cell_n.get(cell) else 0.0
        before_ae, after_ae = abs(e-actual), abs(c-actual)
        before_ape, after_ape = 100*before_ae/actual, 100*after_ae/actual
        all_losses["ae"]["E"] += before_ae * weight
        all_losses["ae"]["C"] += after_ae * weight
        all_losses["ape"]["E"] += before_ape * weight
        all_losses["ape"]["C"] += after_ape * weight
        if p0 is None:
            counts["direction_unavailable"] += 1
            continue
        if not r.get("guardApplied"):
            counts["unchanged" if c == e else "candidate_changed_without_guard_flag"] += 1
            continue
        e_move, c_move, true_move = e - p0, c - p0, actual - p0
        def direction(move):
            return "zero" if true_move == 0 or move == 0 else ("correct" if move * true_move > 0 else "wrong")
        def magnitude(move):
            return "zero" if true_move == 0 and move == 0 else ("over" if true_move == 0 or abs(move) > abs(true_move) else
                    "under" if abs(move) < abs(true_move) else "exact")
        e_class = f"{direction(e_move)}/{magnitude(e_move)}"
        c_class = f"{direction(c_move)}/{magnitude(c_move)}"
        label = f"E:{e_class}|C:{c_class}"
        counts[label] += 1
        active_totals["ae"] += (before_ae-after_ae)*weight
        active_totals["ape"] += (before_ape-after_ape)*weight
        for loss_name, gain in (("ae", before_ae-after_ae), ("ape", before_ape-after_ape)):
            sums[label][loss_name + "Net"] += gain * weight
            sums[label][loss_name + "Gain"] += max(0.0, gain) * weight
            sums[label][loss_name + "Harm"] += max(0.0, -gain) * weight
        sums[label]["weight"] += weight
        counts[label + ("/gain" if before_ae > after_ae else "/harm" if before_ae < after_ae else "/tie")] += 1
    return {"counts": dict(sorted(counts.items())), "weightedGainHarmByDirectionMagnitude": dict(sorted(sums.items())),
            "allPairedCellWeightedLoss": {"E": all_losses["ae"]["E"], "C": all_losses["ae"]["C"],
                "E_MAPE": all_losses["ape"]["E"], "C_MAPE": all_losses["ape"]["C"],
                "CMinusE_AE": all_losses["ae"]["C"]-all_losses["ae"]["E"],
                "CMinusE_MAPE": all_losses["ape"]["C"]-all_losses["ape"]["E"]},
            "activeWeightedGain": {"ae": active_totals["ae"], "ape": active_totals["ape"]},
            "cellCount": cell_count, "weighting": "original product×horizon equal weight; within-cell paired count denominator"}


def _component_complementarity(rows):
    cells = defaultdict(list)
    for r in rows:
        cells[(r["product"], r["horizon"])].append(r)
    details = []
    for (product, horizon), group in sorted(cells.items()):
        n = len(group)
        if not n:
            continue
        errors = {name: [r[name] - r["actual"] for r in group]
                  for name in ("ridge", "sarima", "blend")}
        abs_ape = {name: [100 * abs(e) / r["actual"] for e, r in zip(vals, group)]
                   for name, vals in errors.items()}
        k = max(1, math.ceil(.10 * n))
        sorted_ridge = sorted(range(n), key=lambda i: (-abs_ape["ridge"][i], group[i]["target"]))
        sorted_sarima = sorted(range(n), key=lambda i: (-abs_ape["sarima"][i], group[i]["target"]))
        top_ridge, top_sarima = set(sorted_ridge[:k]), set(sorted_sarima[:k])
        between = sum(min(r["ridge"], r["sarima"]) <= r["actual"] <= max(r["ridge"], r["sarima"]) for r in group)
        oracle_component = [min(abs(r["ridge"]-r["actual"]), abs(r["sarima"]-r["actual"])) for r in group]
        oracle_interval = [max(0.0, min(r["ridge"], r["sarima"])-r["actual"], r["actual"]-max(r["ridge"], r["sarima"])) for r in group]
        details.append({"product": product, "horizon": horizon, "n": n,
                        "sameDirectionResidualFraction": sum(x*y > 0 for x,y in zip(errors["ridge"], errors["sarima"])) / n,
                        "top10PercentLargeAPEOverlap": len(top_ridge & top_sarima),
                        "top10PercentLargeAPECountPerComponent": k,
                        "actualBetweenComponentForecastsFraction": between/n,
                        "oracleBestComponentMae": math.fsum(oracle_component)/n,
                        "oracleBestComponentMape": 100*math.fsum(v/r["actual"] for v,r in zip(oracle_component,group))/n,
                        "oracleProjectionIntervalMae": math.fsum(oracle_interval)/n,
                        "oracleProjectionIntervalMape": 100*math.fsum(v/r["actual"] for v,r in zip(oracle_interval,group))/n})
    return {"cells": details,
            "interpretation": "Per-row realized-target oracles are unattainable hindsight diagnostics, not candidate forecasts."}


def _age_summary(rows):
    phases = sorted({row["phase"] for row in rows})
    return {**_age_summary_single(rows),
            "byPhase": {phase: _age_summary_single([r for r in rows if r["phase"] == phase])
                        for phase in phases}}


def _age_summary_single(rows):
    buckets = {"1": 0, "2": 0, "3-4": 0, "5-7": 0, "8-14": 0, "outside14_or_missing": 0}
    by_cell = defaultdict(list)
    fallback = defaultdict(int)
    for r in rows:
        age = r.get("ageDays")
        if not isinstance(age, int) or age < 1 or age > 14:
            bucket = "outside14_or_missing"
        elif age <= 2:
            bucket = str(age)
        elif age <= 4:
            bucket = "3-4"
        elif age <= 7:
            bucket = "5-7"
        else:
            bucket = "8-14"
        buckets[bucket] += 1
        by_cell[(r["product"], r["horizon"])].append(r)
        fallback[(r["phase"], "fallback" if r["fallback"] else "noFallback")] += 1
    # Each product×horizon cell gets equal weight in the age distribution.
    total = sum(len(v) for v in by_cell.values())
    cell_count = len(by_cell)
    weighted = {key: (math.fsum(sum(1 for r in values if r.get("ageDays") == age) / len(values)
                               for values in by_cell.values()) / cell_count if cell_count else 0.0)
                for key, age in (("1",1), ("2",2))}
    for label, lo, hi in (("3-4",3,4), ("5-7",5,7), ("8-14",8,14)):
        weighted[label] = (math.fsum(sum(1 for r in vals if isinstance(r.get("ageDays"), int) and lo <= r["ageDays"] <= hi) / len(vals)
                                     for vals in by_cell.values()) / cell_count if cell_count else 0.0)
    weighted["outside14_or_missing"] = max(0.0, 1.0 - math.fsum(weighted.values())) if cell_count else 0.0
    price28_by_cell = [math.fsum(r["positivePriceObservationDaysInPrior28"] for r in values) / len(values)
                       for values in by_cell.values()]
    volume28_by_cell = [math.fsum(r["positiveVolumeObservationDaysInPrior28"] for r in values) / len(values)
                        for values in by_cell.values()]
    return {"rowCounts": buckets, "equalCellWeightedFractions": weighted,
            "fallbackCountsByPhase": {f"{p}/{s}": n for (p,s),n in sorted(fallback.items())},
            "equalCellMeanPositivePriceDaysInPrior28": math.fsum(price28_by_cell)/cell_count if cell_count else None,
            "equalCellMeanPositiveVolumeDaysInPrior28": math.fsum(volume28_by_cell)/cell_count if cell_count else None,
            "equalCellZeroPriceCoverageFraction": math.fsum(sum(r["positivePriceObservationDaysInPrior28"] == 0 for r in values)/len(values) for values in by_cell.values())/cell_count if cell_count else None,
            "rowCount": total, "cellCount": cell_count}


def _mlp_diagnostic(doc):
    audits = doc.get("monthlyTrainingAudit", [])
    compact = []
    for month in audits:
        fd = month.get("fitDiagnostics", {})
        compact.append({"fitOrigin": month.get("fitOrigin"),
                        "models": {name: {"nIter": info.get("nIter"), "finalLoss": info.get("finalLoss"),
                                          "warnings": info.get("warnings", [])}
                                   for name, info in fd.items()},
                        "scrambleTrainingAudit": month.get("scrambleTrainingAudit"),
                        "equalCellWeight": month.get("equalCellWeight"),
                        "cellTrainingTargetCounts": month.get("cellTrainingTargetCounts")})
    return {"storedLearningCurves": False, "storedValidationCurves": False,
            "interpretation": "40 iterations with n_iter_no_change=41 does not by itself establish underfitting. The saved artifact contains final loss, iteration count and warning, but no epoch loss or validation curves; plateau cannot be assessed from it.",
            "monthlyTrainingAudits": compact,
            "sampleConfoundAssessment": "The implemented scramble audit is at row/date level. It permutes observed path prices on the causal prefix, preserves the latest p0 anchor, retains the observation mask and records changed versus identity rows; the saved counts should be interpreted with the equal-cell training weights. This is a controlled representation perturbation, not independent resampling of observations."}


def _read_sources():
    raw = {key: path.read_bytes() for key, path in SOURCES.items()}
    raw["prices"] = PRICE_PATH.read_bytes()
    if {key: sha256(value) for key, value in raw.items()} != FROZEN_FILE_SHA256:
        raise ValueError("one or more pinned source file SHA-256 values changed")
    docs = {key: json.loads(value) for key, value in raw.items() if key != "prices"}
    docs["prices"] = json.loads(raw["prices"])
    online._validate(docs["online_manifest"])
    if docs["online_report"].get("manifestSha256") != docs["online_manifest"].get("manifestSha256"):
        raise ValueError("online report is not bound to the frozen forecast manifest")
    expected_links = {"prices": scorer.forecasts.PRICES_SHA256,
                      "baseline": sha256(raw["online_manifest"])}
    for key, phase in (("preserved_confirmation_manifest", "confirmation"),
                       ("preserved_evaluation_manifest", "evaluation")):
        preserved_protocol._validate_manifest(docs[key], phase)
        links = docs[key].get("inputSha256", {})
        if links.get("prices") != expected_links["prices"] or links.get("equalInfoBaseline") != expected_links["baseline"]:
            raise ValueError(f"{key} does not reference the frozen price/baseline inputs")
    for key in ("preserved_confirmation_report", "preserved_evaluation_report"):
        if preserved_scorer._unsigned_digest(docs[key], "reportSha256") != docs[key].get("reportSha256"):
            raise ValueError(f"{key} report digest mismatch")
        phase = "confirmation" if "confirmation" in key else "evaluation"
        manifest = docs[f"preserved_{phase}_manifest"]
        report = docs[key]
        if (report.get("forecastManifestSha256") != manifest.get("manifestSha256") or
                report.get("forecastInputSha256") != manifest.get("inputSha256") or
                report.get("priceInputSha256") != scorer.forecasts.PRICES_SHA256):
            raise ValueError(f"{key} report is not bound to its saved forecast and price inputs")
    for key, digest_key, phase in (("trend_confirmation_manifest", "manifestSha256", "correction_forecast_only"),
                                   ("trend_final_manifest", "manifestSha256", "final_forecast_only")):
        if phase == "correction_forecast_only":
            trend_protocol._validate_correction_manifest(docs[key])
        else:
            unsigned = dict(docs[key]); digest = unsigned.pop(digest_key, None)
            if digest != sha256(trend_protocol._canonical(unsigned)) or docs[key].get("phase") != phase:
                raise ValueError("trend final manifest digest or phase mismatch")
        links = docs[key].get("inputSha256", {})
        if links.get("prices") != expected_links["prices"] or links.get("baselineManifest") != expected_links["baseline"]:
            raise ValueError(f"{key} does not reference the frozen price/baseline inputs")
    for key in ("trend_confirmation_report", "trend_final_report"):
        unsigned = dict(docs[key]); digest = unsigned.pop("reportSha256", None)
        if digest != sha256(trend_protocol._canonical(unsigned)):
            raise ValueError(f"{key} report digest mismatch")
        if docs[key].get("inputSha256") != {"baselineManifest": expected_links["baseline"], "prices": expected_links["prices"]}:
            raise ValueError(f"{key} report does not bind the frozen inputs")
    countertrend_protocol._validate_digest(docs["countertrend_confirmation_manifest"], "manifestSha256")
    if docs["countertrend_confirmation_manifest"].get("inputSha256", {}).get("baselineManifest") != expected_links["baseline"]:
        raise ValueError("countertrend source does not reference the frozen baseline")
    countertrend_protocol._validate_digest(docs["countertrend_confirmation_report"], "reportSha256", "countertrend_confirmation_scored")
    if docs["countertrend_confirmation_report"].get("inputsSha256") != docs["countertrend_confirmation_manifest"].get("inputSha256"):
        raise ValueError("countertrend report does not bind its frozen inputs")
    path_mlp_protocol._validate_manifest(docs["path_mlp_screen_manifest"], "screen")
    forecast = docs["four_crop_forecast"]
    unsigned = dict(forecast); digest = unsigned.pop("forecastSha256", None)
    if digest != sha256(four_crop_protocol._canonical(unsigned)):
        raise ValueError("four-crop forecast digest mismatch")
    if forecast.get("inputSha256", {}).get("prices") != four_crop_protocol.LIVE_PRICES_SHA256:
        raise ValueError("four-crop forecast does not reference its frozen live price snapshot")
    four_crop_protocol._validate_snapshot(docs["four_crop_fixed_prices"], raw["four_crop_fixed_prices"], four_crop_protocol.LIVE_PRICES)
    return raw, docs


def _validate_unique(rows, key_fields):
    seen = set()
    for row in rows:
        key = tuple(row.get(k) for k in key_fields)
        if key in seen:
            raise ValueError(f"duplicate source key: {key}")
        seen.add(key)
    return len(seen)


def build_report():
    """Build report solely from saved predictions/summaries and pinned prices."""
    raw, docs = _read_sources()
    actuals = scorer.price_helper._series(docs["prices"])
    actual_map = {(p, d): v for p, vals in actuals.items() for d, v in vals.items()}
    source_hashes = {k: sha256(v) for k, v in raw.items()}
    # Reuse the already completed statistical families verbatim with explicit provenance.
    dm = {
        "equalInfoConfirmation": {"source": str(SOURCES["online_report"]), "family": docs["online_report"]["confirmation2025"]["dmSummaryByBandwidth"]},
        "equalInfoEvaluation": {"source": str(SOURCES["online_report"]), "family": docs["online_report"]["evaluation2026"]["dmSummaryByBandwidth"]},
        "preservedConfirmation": {"source": str(SOURCES["preserved_confirmation_report"]), "family": docs["preserved_confirmation_report"]["dmSummaryByBandwidth"]},
        "preservedEvaluation": {"source": str(SOURCES["preserved_evaluation_report"]), "family": docs["preserved_evaluation_report"]["dmSummaryByBandwidth"]},
        "trendConfirmation": {"source": str(SOURCES["trend_confirmation_report"]), "family": docs["trend_confirmation_report"]["dmSummaryByBandwidth"]},
        "trendFinal": {"source": str(SOURCES["trend_final_report"]), "family": docs["trend_final_report"]["dmSummaryByBandwidth"]},
        "countertrendConfirmation": {"source": str(ROOT / "outputs/experiments/countertrend-guard-20261010/confirmation-report.json"), "family": None},
    }
    c_report = json.loads((ROOT / "outputs/experiments/countertrend-guard-20261010/confirmation-report.json").read_text())
    dm["countertrendConfirmation"]["family"] = c_report.get("dmSummaryByBandwidth")

    def make_pairs(rows, method_a, method_b, target_key="target", model_getter=None):
        out = []
        for r in rows:
            if r.get("eligible") is not True:
                continue
            p, d, h = r.get("product"), dt.date.fromisoformat(r[target_key]), r.get("horizon")
            actual = actual_map.get((p, d))
            if actual is None:
                continue
            if model_getter:
                pa, pb = model_getter(r, method_a), model_getter(r, method_b)
            else:
                pa, pb = _positive(r.get(method_a)), _positive(r.get(method_b))
            if pa and pb:
                out.append({"product": p, "target": d.isoformat(), "horizon": h,
                            "actual": actual, "a": pa, "b": pb})
        return out

    online_rows = docs["online_manifest"]["evaluationForecasts"]
    _validate_unique(online_rows, ("product", "target", "horizon"))
    decomposition = {}
    reproduction = {}
    bootstrap_pairsets = {}
    comp_rows = []
    for r in online_rows:
        if not r.get("eligible"):
            continue
        p, d, h = r["product"], dt.date.fromisoformat(r["target"]), r["horizon"]
        actual = actual_map.get((p, d)); models = r.get("models", {})
        vals = {name: _positive(models.get(key, {}).get("prediction"))
                for name, key in (("ridge", "causalDirect"), ("sarima", "causalSarima"), ("blend", "causal50"))}
        if p in PRODUCTS and actual and all(vals.values()):
            comp_rows.append({"product": p, "target": d.isoformat(), "horizon": h, "actual": actual, **vals})
    if len({(r["product"], r["horizon"]) for r in comp_rows}) != 40:
        raise ValueError("component complementarity lost the frozen four-crop 40-cell family")
    complementarity = _component_complementarity(comp_rows)

    # E confirmation/final and preserved garlic periods retain the original
    # product×horizon cell weighting and paired keys.
    def trend_pairs(doc, phase):
        key = "confirmationForecasts" if phase == "confirmation" else "finalForecasts"
        result = []
        for r in doc[key]:
            p, d, h = r["product"], dt.date.fromisoformat(r["target"]), r["horizon"]
            actual = actual_map.get((p, d)); a = _positive(r.get("model", {}).get("prediction")); b = _positive(r.get("baselinePrediction"))
            if p in PRODUCTS and actual and a and b:
                result.append({"product": p, "target": d.isoformat(), "horizon": h, "actual": actual, "a": a, "b": b})
        return result
    for label, doc, phase in (("trendConfirmation", docs["trend_confirmation_manifest"], "confirmation"),
                              ("trendFinal", docs["trend_final_manifest"], "final")):
        pairs = trend_pairs(doc, phase)
        bootstrap_pairsets[label] = pairs
        saved = docs["trend_confirmation_report" if phase == "confirmation" else "trend_final_report"]
        reproduction[label] = _verify_saved_metrics(pairs, saved["metrics"],
            a_fields=("aMae", "aMape"), b_fields=("bMae", "bMape"))
        expected_cells = {(m["product"], m["horizon"]) for m in saved["metrics"] if m["n"] > 0}
        decomposition[label] = {metric: decompose_losses(pairs, metric=metric, expected_cells=expected_cells) for metric in ("ae", "ape")}
        macro = saved.get("overall40CellMacro")
        if not isinstance(macro, dict) or len(expected_cells) != 40:
            raise ValueError(f"{label} lost a required original 40-cell metric family")
        # Preserve and verify the per-cell cohort arithmetic against stored report cells.
        for metric, key_a, key_b in (("ae", "candidateMae", "baselineMae"), ("ape", "candidateMape", "baselineMape")):
            rebuilt = _cell_means(pairs, "a", "b", metric)
            if macro.get(key_a) is None or rebuilt is None or not math.isclose(rebuilt["a"], macro[key_a], rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError(f"{label} paired candidate metric does not reproduce the saved macro")
            if macro.get(key_b) is None or rebuilt is None or not math.isclose(rebuilt["b"], macro[key_b], rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError(f"{label} paired baseline metric does not reproduce the saved macro")

    preserved = {}
    for key, phase in (("preserved_confirmation_manifest", "confirmation"), ("preserved_evaluation_manifest", "evaluation")):
        doc = docs[key]
        rows = doc["forecasts"]
        _validate_unique(rows, ("product", "target", "horizon"))
        for a_name, b_name, name in (("preserved106Blend", "causal50", "blend_vs_causal50"),
                                     ("preserved106", "causalDirect", "direct106_vs_direct40")):
            pairs = make_pairs(rows, a_name, b_name,
                model_getter=lambda r, method: _positive(r.get("models", {}).get(method, {}).get("prediction")))
            rows_by_key = {(r["product"], r["target"], r["horizon"]): r for r in rows}
            pairs = [r for r in pairs if r["product"] == "garlic" and
                     all(_positive(rows_by_key[(r["product"], r["target"], r["horizon"])].get("models", {}).get(method, {}).get("prediction")) is not None
                         for method in ("preserved106Blend", "causal50", "fixed50"))]
            saved_report = docs["preserved_confirmation_report" if phase == "confirmation" else "preserved_evaluation_report"]
            comparison = "preserved106Blend_vs_causal50" if a_name == "preserved106Blend" else "preserved106_vs_causalDirect"
            reproduction[f"{phase}_{name}"] = _verify_saved_metrics(pairs, saved_report["metrics"],
                comparison=comparison, product="garlic", a_fields=("maeA", "mapeA"), b_fields=("maeB", "mapeB"))
            expected_cells = {(m["product"], m["horizon"]) for m in saved_report["metrics"]
                              if m.get("comparison") == comparison and m.get("product") == "garlic" and m["n"] > 0}
            if len(expected_cells) != 10:
                raise ValueError("preserved garlic comparison lost its frozen 10-cell family")
            preserved[f"{phase}_{name}"] = {metric: decompose_losses(pairs, metric=metric, expected_cells=expected_cells) for metric in ("ae", "ape")}
            report_metric = next((m for m in saved_report.get("equalHorizonMetrics", [])
                                  if m.get("product") == "garlic" and m.get("comparison") == comparison), None)
            rebuilt = _cell_means(pairs, "a", "b", "ae")
            if report_metric is None or rebuilt is None or report_metric.get("aEqualHorizonMae") is None or not math.isclose(rebuilt["a"], report_metric["aEqualHorizonMae"], rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError(f"{phase} garlic preserved candidate cohort does not reproduce saved MAE")
            if report_metric.get("bEqualHorizonMae") is None or not math.isclose(rebuilt["b"], report_metric["bEqualHorizonMae"], rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError(f"{phase} garlic preserved baseline cohort does not reproduce saved MAE")
            if not math.isclose(_cell_means(pairs, "a", "b", "ape")["a"], report_metric["aEqualHorizonMape"], rel_tol=1e-9, abs_tol=1e-8) or not math.isclose(_cell_means(pairs, "a", "b", "ape")["b"], report_metric["bEqualHorizonMape"], rel_tol=1e-9, abs_tol=1e-8):
                raise ValueError(f"{phase} garlic preserved cohort does not reproduce saved MAPE")
            if name == "blend_vs_causal50":
                bootstrap_pairsets[f"preservedGarlic_{phase}"] = pairs
                decomposition[f"preservedGarlic_{phase}"] = preserved[f"{phase}_{name}"]
                if phase == "evaluation":
                    by_cell = defaultdict(list)
                    for pair in pairs:
                        by_cell[(pair["product"], pair["horizon"])].append(pair)
                    event_keys = set()
                    for cell, group in by_cell.items():
                        ranked = sorted(group, key=lambda r: (-100*abs(r["b"]-r["actual"])/r["actual"], r["target"]))
                        event = ranked[:max(1, math.ceil(.10*len(group)))]
                        event_keys.update((cell, r["target"]) for r in event)
                    event_summary = {}
                    cell_counts = {cell: len(group) for cell, group in by_cell.items()}
                    for metric in ("ae", "ape"):
                        subtotals = {group: {"gain": 0.0, "n": 0, "positiveGainCount": 0, "harmCount": 0}
                                     for group in ("largeErrorEvents", "otherTargets", "allTargets")}
                        for pair in pairs:
                            cell = (pair["product"], pair["horizon"])
                            group = "largeErrorEvents" if (cell, pair["target"]) in event_keys else "otherTargets"
                            weight = 1.0 / (10 * cell_counts[cell])
                            gain = (_loss(pair, "b", metric)-_loss(pair, "a", metric))*weight
                            for name_group in (group, "allTargets"):
                                slot = subtotals[name_group]
                                slot["gain"] += gain
                                slot["n"] += 1
                                slot["positiveGainCount"] += int(gain > 0)
                                slot["harmCount"] += int(gain < 0)
                        total_gain = subtotals["allTargets"]["gain"]
                        event_gain = subtotals["largeErrorEvents"]["gain"]
                        subtotals["largeErrorEvents"]["shareOfTotalWeightedGain"] = event_gain/total_gain if total_gain else None
                        event_summary[metric] = subtotals
                    complementarity["garlic106BlendGainOnCausal50LargeErrorEvents"] = {
                        "definition": "within each product×horizon cell, top ceil(10%) paired targets by causal50 absolute percentage error",
                        "weighting": "frozen 10-cell equal weight and within-cell paired denominator",
                        "subgroups": event_summary}

    boot = {}
    bootstrap_ranges = {"trendConfirmation": (dt.date(2026,4,1), dt.date(2026,6,20)),
                        "trendFinal": (dt.date(2026,7,11), dt.date(2026,9,27)),
                        "preservedGarlic_confirmation": (dt.date(2025,10,1), dt.date(2025,12,31)),
                        "preservedGarlic_evaluation": (dt.date(2026,1,1), dt.date(2026,9,27))}
    for label, pairset in bootstrap_pairsets.items():
        start, end = bootstrap_ranges[label]
        boot[label] = {metric: {str(block): circular_block_bootstrap(pairset, metric=metric, block=block, replicates=2000,
                                                                       calendar_start=start, calendar_end=end)
                                for block in (14, 28)} for metric in ("ae", "ape")}
    # Magnitude and direction on the frozen candidate confirmation rows.
    ct_rows = docs["countertrend_confirmation_manifest"]["confirmationForecasts"]
    ct_actual = []
    for r in ct_rows:
        actual = actual_map.get((r["product"], dt.date.fromisoformat(r["target"])))
        if actual and _positive(r.get("baselinePrediction")) and _positive(r.get("model", {}).get("prediction")):
            ct_actual.append({**r, "actual": actual})
    countertrend = classify_countertrend(ct_actual, expected_cell_count=40,
        expected_cells={(product, horizon) for product in PRODUCTS for horizon in range(1, 11)})

    # Source ages are taken from each row's frozen origin-time feature audit;
    # issued 2026-10-10 rows are kept as their own period.
    age_rows = []
    price_records, _price_audit = online.rich._raw_maps(docs["prices"])
    live_price_records, _live_price_audit = online.rich._raw_maps(docs["four_crop_fixed_prices"])
    age_sources = [("online2026", docs["online_manifest"]["evaluationForecasts"]),
                   ("Econfirmation", docs["trend_confirmation_manifest"]["confirmationForecasts"]),
                   ("Efinal", docs["trend_final_manifest"]["finalForecasts"]),
                   ("preservedConfirmation", docs["preserved_confirmation_manifest"]["forecasts"]),
                   ("preservedEvaluation", docs["preserved_evaluation_manifest"]["forecasts"]),
                   ("fourCrop2026-10-10", docs["four_crop_forecast"]["forecasts"])]
    frozen_baseline_rows = docs["online_manifest"]["confirmationForecasts"] + docs["online_manifest"]["evaluationForecasts"]
    frozen_baseline = {(r["product"], r["target"], r["horizon"]): r for r in frozen_baseline_rows}
    for phase, rows in age_sources:
        for r in rows:
            if r.get("eligible") is False:
                continue
            origin = dt.date.fromisoformat(r.get("origin") or r.get("fitOrigin"))
            h, product = r.get("horizon"), r.get("product")
            if product not in PRODUCTS:
                continue
            models = r.get("models", {})
            feat = models.get("causalDirect", {}).get("provenance", {}).get("featureAudit", {})
            if not feat:
                feat = models.get("causalDirect", {}).get("provenance", {})
            if not feat and isinstance(r.get("features"), dict):
                feat = r["features"]
            if phase.startswith("fourCrop"):
                feat = models.get("causalDirect", {}).get("provenance", {}).get("featureAudit", {})
                if not feat:
                    feat = models.get("causal50", {}).get("provenance", {})
            age = feat.get("p0AgeDays", feat.get("sourceAgeDays"))
            source_date = feat.get("p0Date", feat.get("sourceDate"))
            cutoff = origin - dt.timedelta(days=1)
            records = (live_price_records if phase.startswith("fourCrop") else price_records).get(product, {})
            latest_price = max((day for day, rec in records.items() if day <= cutoff and _positive(rec.get("price"))), default=None)
            latest_volume = max((day for day, rec in records.items() if day <= cutoff and _positive(rec.get("volume"))), default=None)
            computed_age = (origin-latest_price).days if latest_price else None
            if source_date and latest_price and dt.date.fromisoformat(source_date) != latest_price:
                raise ValueError(f"saved p0 source date disagrees with frozen input for {product}/{origin}")
            if isinstance(age, int) and computed_age is not None and age != computed_age:
                raise ValueError(f"saved source age disagrees with frozen input for {product}/{origin}")
            if age is None:
                age = computed_age
            price28 = sum(1 for day, rec in records.items() if cutoff-dt.timedelta(days=27) <= day <= cutoff and _positive(rec.get("price")))
            volume28 = sum(1 for day, rec in records.items() if cutoff-dt.timedelta(days=27) <= day <= cutoff and _positive(rec.get("volume")))
            base_row = frozen_baseline.get((product, r.get("target"), h), {})
            issue_model = models.get("causal50") or base_row.get("models", {}).get("causal50") or r.get("model") or {}
            issue_prediction = issue_model.get("prediction") if isinstance(issue_model, dict) else None
            fallback_model = models.get("causal50") or base_row.get("models", {}).get("causal50")
            fallback = (fallback_model.get("provenance", {}).get("fallback", fallback_model.get("fallback", False))
                        if fallback_model else False)
            age_rows.append({"phase": phase, "product": product, "origin": origin.isoformat(),
                             "target": r.get("target"), "horizon": h, "ageDays": age,
                             "targetGapDaysFromLatestPrice": ((dt.date.fromisoformat(r["target"])-latest_price).days if latest_price else None),
                             "latestPositiveVolumeAgeDays": ((origin-latest_volume).days if latest_volume else None),
                             "positivePriceObservationDaysInPrior28": price28,
                             "positiveVolumeObservationDaysInPrior28": volume28,
                             "sourceDate": source_date, "issued": _positive(issue_prediction) is not None,
                             "unissuedReason": issue_model.get("reason") if isinstance(issue_model, dict) else None,
                             "fallback": fallback})
    report = {"schemaVersion": 1, "phase": "retrospective_diagnostic",
              "inputSha256": source_hashes,
              "diagnostics": {"lossStructure": decomposition,
                              "metricReproduction": reproduction,
                              "preservedFeaturePairs": preserved,
                              "countertrendMagnitude": countertrend,
                              "priceAge": {"rows": age_rows, "summary": _age_summary(age_rows),
                                           "selectionIsFrozenInputSnapshot": True,
                                           "publicationDelayCannotBeInferred": True},
                              "componentComplementarity": complementarity,
                              "pathMlpTrainingDiagnostic": _mlp_diagnostic(docs["path_mlp_screen_manifest"]),
                              "originalDmFamilies": dm,
                              "circularBlockBootstrap": boot,
                              "complementarity": "See stored component predictions; oracle statistics are reported as unattainable diagnostics only."},
              "limitations": ["Repeatedly inspected retrospective periods; bootstrap is descriptive, not confirmatory.",
                              "No archived first-publication time is inferred from observation or modification dates.",
                              "Missing targets are never imputed; all comparisons use their saved paired cohorts.",
                              "Oracle selection/projection uses realized targets and cannot be deployed."],
              "sourceFiles": {k: str(v) for k, v in SOURCES.items()} | {"prices": str(PRICE_PATH)}}
    report["reportSha256"] = sha256((json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode())
    return report


def write_report(report, output_dir=OUT):
    """Create-only output; existing diagnostics are immutable."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = output_dir / "report.json", output_dir / "report.md"
    if json_path.exists() or md_path.exists():
        raise FileExistsError("diagnostic outputs already exist; refusing overwrite")
    json_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    details = report["diagnostics"]
    summary = ["# 저장 예측 오차 구조 진단", "", "고정 저장 예측과 원가격을 사용한 후향 진단이다. 새 적합·수집은 없다.", "",
               "## paired 오차 구조", "", "| 비교 | 공통쌍 | MAE 차이 (A−B) | MAPE 차이 (A−B, %p) |",
               "|---|---:|---:|---:|"]
    for name, metric_set in details["lossStructure"].items():
        ae, ape = metric_set["ae"], metric_set["ape"]
        summary.append(f"| {name} | {ae['n']} | {ae['meanDifference']:.4f} | {ape['meanDifference']:.5f} |")
    summary.extend(["", "## 고정 추론 및 보조 bootstrap", "", "기존 DM/HAC/Holm 판정은 저장 보고서의 고정 검정 가족을 그대로 인용한다. 순환 블록 구간은 반복 열람 자료의 기술적 민감도다.", ""])
    for name, metrics in details["circularBlockBootstrap"].items():
        for metric, bands in metrics.items():
            for band, result in bands.items():
                interval = result["interval95"]
                text = "무효 반복만 존재" if interval is None else f"[{interval[0]:.5f}, {interval[1]:.5f}]"
                summary.append(f"- {name}, {metric}, {band}일 블록: 유효 {result['valid']}/{result['requested']}, 차이 95% 구간 {text}.")
    ages = details["priceAge"]["summary"]
    active = {k: v for k, v in details["countertrendMagnitude"]["counts"].items() if "/" in k and not k.endswith(("/gain", "/harm", "/tie"))}
    component_cells = details["componentComplementarity"]["cells"]
    component_n = sum(r["n"] for r in component_cells)
    between = math.fsum(r["actualBetweenComponentForecastsFraction"] * r["n"] for r in component_cells) / component_n if component_n else 0.0
    mlp = details["pathMlpTrainingDiagnostic"]
    summary.extend(["", "## 실제 입력 나이", "",
                    "기간별로 분리해 원점×거리 셀 동일가중으로 요약했다."])
    for phase, phase_age in ages["byPhase"].items():
        summary.append(f"- {phase}: {phase_age['rowCount']} rows/{phase_age['cellCount']} cells; price age buckets {json.dumps(phase_age['equalCellWeightedFractions'], ensure_ascii=False, sort_keys=True)}; prior-28-day positive price/weight observation mean {phase_age['equalCellMeanPositivePriceDaysInPrior28']:.2f}/{phase_age['equalCellMeanPositiveVolumeDaysInPrior28']:.2f} days; fallback/issue state counts {json.dumps(phase_age['fallbackCountsByPhase'], ensure_ascii=False, sort_keys=True)}.")
    summary.extend([
                    "10월 10일 발행은 해당 발행에 고정된 가격 스냅숏만 사용한다. 관측일로부터 최초 공개 지연을 추정하지 않는다.",
                    "", "## 모델 보완성과 역추세 크기 진단", "",
                    f"동일정보 Ridge/SARIMA 공통쌍 {component_n}개 중 실제값이 두 예측 사이에 놓인 비율은 {between:.1%}다. 두 oracle 수치는 hindsight 진단으로만 JSON에 기록했다.",
                    f"역추세 후보의 활성 행 방향·크기 분류: `{json.dumps(active, ensure_ascii=False, sort_keys=True)}`.",
                    f"저장 MLP 적합 감사는 {len(mlp['monthlyTrainingAudits'])}개 월 원점이다. 학습·검증 곡선 저장 여부: {mlp['storedLearningCurves']}/{mlp['storedValidationCurves']}; 따라서 반복 한계 경고만으로 수렴 부족을 판정하지 않는다.",
                    "", "## 한계", "",
                    "- 반복 열람한 후향 자료이며 bootstrap은 새 유의성이나 채택 근거가 아니다.",
                    "- 방향·크기 oracle은 실현값을 사용하므로 실행 후보가 아니다.",
                    "- 기존 DM/HAC/Holm 검정 가족은 재계산하거나 변경하지 않았다.",
                    "", f"JSON SHA-256: `{report['reportSha256']}`", "",
                    "입력 파일별 해시와 날짜·월·ISO주 분해, subgroup, coverage, source age 상세는 report.json에 보존했다."])
    md_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
    return json_path, md_path


if __name__ == "__main__":
    write_report(build_report())
