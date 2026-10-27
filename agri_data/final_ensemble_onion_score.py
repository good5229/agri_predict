"""Integrity-checked scoring for FINAL_ENSEMBLE_LOSS_ONION_V2.

Forecast artifacts stay answer-free. This module joins the frozen price
snapshot only after checking the producer manifests, baseline cache, and exact
forecast key sets.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import equal_info_online_score as stats
from . import full_history_return_eval as price_helper
from . import research_progress_diagnostic as diagnostics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/final-ensemble-loss-onion-v2-20261014"
PRODUCTS = ("cabbage", "radish", "onion", "garlic")
ONION = "onion"
HORIZONS = tuple(range(1, 11))
PHASES = {
    "D25": (dt.date(2025, 4, 1), dt.date(2025, 6, 20), dt.date(2025, 4, 1)),
    "E25": (dt.date(2025, 7, 11), dt.date(2025, 9, 27), dt.date(2025, 7, 1)),
    "V26": (dt.date(2026, 4, 1), dt.date(2026, 6, 20), dt.date(2026, 4, 1)),
    "T26": (dt.date(2026, 7, 11), dt.date(2026, 9, 27), dt.date(2026, 7, 1)),
}
BANDWIDTHS = (9, 28)
REPLICATES = 2000
SEED = 0


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def _date(value: Any) -> dt.date:
    if isinstance(value, dt.date):
        return value
    try:
        return dt.date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"invalid ISO date {value!r}") from exc


def _node(row: Mapping[str, Any], name: str) -> tuple[float | None, str, Mapping[str, Any]]:
    """Accept the producer's typed output object and the compact test form."""
    raw = row.get(name)
    if isinstance(raw, Mapping):
        return positive(raw.get("prediction")), str(raw.get("status", "missing")), raw
    aliases = {"B": ("BStatus", "sourceStatus"), "ND": ("NDStatus", "statusD"),
               "NF": ("NFStatus", "statusF"), "HD": ("HDStatus", "statusD"),
               "HF": ("HFStatus", "statusF")}
    status = next((row.get(field) for field in aliases[name] if row.get(field) is not None), None)
    if name == "B" and status is None and row.get("BSourceStatus") is not None:
        # BSourceStatus describes which baseline component issued B; fallback
        # is still a valid B forecast row when its value is present.
        status = "not_issued" if row.get("BSourceStatus") == "not_issued" else "issued"
    if status is None:
        status = "issued" if positive(raw) is not None else "not_issued"
    suffix = {"ND": "D", "NF": "F", "HD": "D", "HF": "F"}.get(name, name)
    return positive(raw), str(status), {"prediction": raw, "status": status,
        "reason": row.get("reason" + suffix),
        "fallbackTo": row.get("fallbackTo" + suffix),
        "rawStatus": row.get("rawStatus" + suffix)}


def _expected_keys(phase: str, products: Sequence[str]) -> set[tuple[str, str, str, int]]:
    start, end, first_origin = PHASES[phase]
    return {
        (product, (target - dt.timedelta(days=h)).isoformat(), target.isoformat(), h)
        for product in products
        for offset in range((end - start).days + 1)
        for target in (start + dt.timedelta(days=offset),)
        for h in HORIZONS
        if target - dt.timedelta(days=h) >= first_origin
    }


def _normalized_rows(phase: str, rows: Sequence[Mapping[str, Any]], *,
                     products: Sequence[str] | None = None,
                     horizons: Sequence[int] | None = None) -> list[dict[str, Any]]:
    products = tuple(products or PRODUCTS)
    horizons = tuple(horizons or HORIZONS)
    expected = _expected_keys(phase, products)
    if tuple(horizons) != HORIZONS:
        start, end, first_origin = PHASES[phase]
        expected = {
            (product, (target - dt.timedelta(days=h)).isoformat(), target.isoformat(), h)
            for product in products
            for offset in range((end - start).days + 1)
            for target in (start + dt.timedelta(days=offset),)
            for h in horizons
            if target - dt.timedelta(days=h) >= first_origin
        }
    out = []
    seen = set()
    for source in rows:
        product = source.get("product")
        target, origin = _date(source.get("target")), _date(source.get("origin"))
        try:
            horizon = int(source.get("horizon"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("forecast has an invalid horizon") from exc
        key = (product, origin.isoformat(), target.isoformat(), horizon)
        if key not in expected or key in seen or target - origin != dt.timedelta(days=horizon):
            raise ValueError(f"unexpected or duplicate {phase} forecast key: {key}")
        seen.add(key)
        values = {}
        states = {}
        nodes = {}
        for name in ("B", "ND", "NF", "HD", "HF"):
            value, status, node = _node(source, name)
            values[name], states[name], nodes[name] = value, status, node
        # The producer persists both neural outputs and the selected 50:50
        # hybrids; derive H_D when it is represented by statusD + N_D only.
        for nn, hybrid in (("ND", "HD"), ("NF", "HF")):
            if product == ONION and states[nn] in ("fallback", "technical_fallback"):
                supplied_hybrid = values[hybrid]
                if supplied_hybrid is not None and supplied_hybrid != values["B"]:
                    raise ValueError(f"{hybrid} technical fallback does not equal B")
                suffix = "D" if nn == "ND" else "F"
                hybrid_raw_status = source.get("statusH" + suffix)
                if hybrid_raw_status is None and isinstance(source.get(hybrid), Mapping):
                    hybrid_raw_status = source[hybrid].get("status")
                if hybrid_raw_status not in ("fallback", "technical_fallback"):
                    raise ValueError(f"{hybrid} did not preserve the neural fallback status")
                hybrid_reason = source.get("reasonH" + suffix)
                if hybrid_reason is None and isinstance(source.get(hybrid), Mapping):
                    hybrid_reason = source[hybrid].get("reason")
                if hybrid_reason is not None and hybrid_reason != nodes[nn].get("reason"):
                    raise ValueError(f"{hybrid} fallback reason differs from neural failure")
                values[hybrid], states[hybrid] = values["B"], "fallback"
                reason = nodes[nn].get("reason")
                if not reason:
                    raise ValueError(f"{nn} technical fallback lacks a preserved reason")
                nodes[hybrid] = {"status": "fallback", "prediction": values["B"],
                                 "fallbackTo": "B", "reason": reason}
            elif source.get(hybrid) is None and product == ONION:
                if states[nn] == "issued" and values[nn] is not None and values["B"] is not None:
                    values[hybrid] = 0.5 * values["B"] + 0.5 * values[nn]
                    states[hybrid] = "issued"
                    nodes[hybrid] = {"status": "issued", "prediction": values[hybrid]}
        if product != ONION:
            for name in ("ND", "NF", "HD", "HF"):
                if values[name] is None:
                    values[name], states[name], nodes[name] = values["B"], states["B"], nodes["B"]
                elif states[name] == "baseline_copy":
                    if values[name] != values["B"]:
                        raise ValueError(f"non-onion {name} baseline_copy differs from B")
                    states[name] = "issued"
                    nodes[name] = {"status": "issued", "prediction": values[name]}
        out.append({**dict(source), "product": product, "origin": origin,
                    "target": target, "horizon": horizon, "_values": values,
                    "_states": states, "_nodes": nodes})
    if seen != expected:
        raise ValueError(f"{phase} forecast grid mismatch: missing={len(expected-seen)} extra={len(seen-expected)}")
    return sorted(out, key=lambda r: (r["product"], r["origin"], r["target"], r["horizon"]))


def _check_fallback(node: Mapping[str, Any], prediction: float | None,
                    baseline: float | None, label: str, key: tuple[Any, ...]) -> None:
    if node.get("status") == "fallback":
        if prediction is None or baseline is None or prediction != baseline:
            raise ValueError(f"{label} fallback does not equal B at {key}")
        if node.get("fallbackTo") != "B" or not node.get("reason"):
            raise ValueError(f"{label} fallback lacks B destination/reason at {key}")
    elif node.get("status") == "issued":
        if prediction is None:
            raise ValueError(f"{label} issued without finite positive prediction at {key}")
    else:
        raise ValueError(f"{label} has nonissuance on eligible key {key}")


def _validate_semantics(rows: Sequence[dict[str, Any]], *, phase: str) -> None:
    start = PHASES[phase][2]
    for row in rows:
        if row["origin"] < start:
            continue
        key = (row["product"], row["origin"].isoformat(), row["target"].isoformat(), row["horizon"])
        values, states, nodes = row["_values"], row["_states"], row["_nodes"]
        b = values["B"]
        if states["B"] == "issued" and b is None:
            raise ValueError(f"B issue has no positive value at {key}")
        if row["product"] != ONION:
            for name in ("ND", "NF", "HD", "HF"):
                if name in row and values[name] is not None and values[name] != b:
                    raise ValueError(f"non-onion {name} is not an exact B copy at {key}")
            continue
        if states["B"] != "issued":
            if any(states[n] not in ("not_issued", "fallback") for n in ("HD", "HF")):
                raise ValueError(f"hybrid is issued without an issuable B at {key}")
            continue
        for name in ("ND", "NF", "HD", "HF"):
            if name in ("ND", "NF") and states[name] == "technical_fallback":
                if values[name] is not None or not nodes[name].get("reason"):
                    raise ValueError(f"{name} technical fallback is malformed at {key}")
            else:
                _check_fallback(nodes[name], values[name], b, name, key)
        for nn, hybrid in (("ND", "HD"), ("NF", "HF")):
            if states[nn] == "fallback":
                if states[hybrid] != "fallback" or values[hybrid] != b:
                    raise ValueError(f"{hybrid} did not preserve neural failure fallback at {key}")
            elif states[hybrid] == "issued":
                expected = 0.5 * b + 0.5 * values[nn]
                if values[hybrid] != expected:
                    raise ValueError(f"{hybrid} is not the frozen 50:50 blend at {key}")


def _actual_map(price_doc: Mapping[str, Any], *, products: Sequence[str] = PRODUCTS) -> dict[tuple[str, dt.date], float]:
    all_prices = price_helper._series(dict(price_doc))
    return {(product, day): value for product in products
            for day, value in all_prices[product].items()}


def _score_rows(rows: Sequence[dict[str, Any]], actuals: Mapping[tuple[str, dt.date], float],
                *, products: Sequence[str], use_full_policy: bool) -> dict[str, Any]:
    scored = []
    positive_pairs = 0
    missing_actual_rows = []
    for row in rows:
        actual = actuals.get((row["product"], row["target"]))
        b = row["_values"]["B"]
        if row["_states"]["B"] == "issued" and (
                actual is None or not math.isfinite(float(actual)) or float(actual) <= 0):
            missing_actual_rows.append({"product": row["product"], "origin": row["origin"].isoformat(),
                                        "target": row["target"].isoformat(), "horizon": row["horizon"],
                                        "reason": "missing_or_invalid_actual"})
        if actual is None or b is None or row["_states"]["B"] != "issued":
            continue
        if not math.isfinite(float(actual)) or float(actual) <= 0:
            continue
        positive_pairs += 1
        candidate_names = ("B", "HD", "HF") if row["product"] == ONION else ("B",)
        if use_full_policy and row["product"] != ONION:
            candidate_names = ("B", "HD", "HF")
        for name in candidate_names:
            value = row["_values"][name]
            state = row["_states"][name]
            if value is None or state not in ("issued", "fallback"):
                raise ValueError(f"missing eligible {name} output at {row['product']}/{row['target']}/h{row['horizon']}")
        scored.append({**row, "actual": actual})
    if not scored:
        raise ValueError("no paired positive actual/B-issued keys")

    scope_products = tuple(products) if use_full_policy else (ONION,)
    cells = [(product, h) for product in scope_products for h in HORIZONS]
    by_cell: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        if row["product"] in scope_products:
            by_cell[(row["product"], row["horizon"])].append(row)
    missing_cells = [cell for cell in cells if not by_cell.get(cell)]
    if missing_cells:
        raise ValueError(f"paired scoring lacks frozen cells: {missing_cells}")

    comparisons = (("HF_vs_B", "HF", "B"), ("HF_vs_HD", "HF", "HD"),
                   ("NF_vs_ND", "NF", "ND"))
    metrics = {}
    decompositions = {}
    for cname, a, b in comparisons:
        comparison_cells = ([(ONION, h) for h in HORIZONS] if cname == "NF_vs_ND" else cells)
        common = [r for r in scored if r["product"] in scope_products
                  and (cname != "NF_vs_ND" or r["product"] == ONION)
                  and r["_values"].get(a) is not None and r["_values"].get(b) is not None
                  and (cname != "NF_vs_ND" or
                       r["_states"].get(a) == "issued" and r["_states"].get(b) == "issued")]
        per_cell = {}
        per_cell_rows = {}
        for cell in comparison_cells:
            values = [r for r in common if (r["product"], r["horizon"]) == cell]
            per_cell_rows[cell] = values
            if any(r["_values"].get(a) is None or r["_values"].get(b) is None for r in values):
                raise ValueError(f"incomplete paired {cname} coverage for {cell}")
            n = len(values)
            if n == 0:
                per_cell[f"{cell[0]}:h{cell[1]}"] = {"n": 0, "aMae": None, "bMae": None,
                                                       "aMape": None, "bMape": None}
                continue
            abs_a = [abs(r["_values"][a] - r["actual"]) for r in values]
            abs_b = [abs(r["_values"][b] - r["actual"]) for r in values]
            per_cell[f"{cell[0]}:h{cell[1]}"] = {
                "n": n,
                "aMae": math.fsum(abs_a) / n, "bMae": math.fsum(abs_b) / n,
                "aMape": 100 * math.fsum(e / r["actual"] for e, r in zip(abs_a, values)) / n,
                "bMape": 100 * math.fsum(e / r["actual"] for e, r in zip(abs_b, values)) / n,
            }
        n_cells = len(comparison_cells)
        nonempty_cells = [v for v in per_cell.values() if v["n"] > 0]
        onion_cell_values = [per_cell[f"{ONION}:h{h}"] for h in HORIZONS
                             if f"{ONION}:h{h}" in per_cell and per_cell[f"{ONION}:h{h}"]["n"] > 0]
        onion_complete = len(onion_cell_values) == len(HORIZONS)
        metrics[cname] = {
            "scope": ("common_issued_neural_onion_rows" if cname == "NF_vs_ND" else
                      "original_40_cell_policy" if use_full_policy else "onion_10_cells"),
            "cells": per_cell,
            "aEqualCellMae": (math.fsum(v["aMae"] for v in per_cell.values() if v["n"] > 0) / n_cells
                              if len(nonempty_cells) == n_cells else None),
            "bEqualCellMae": (math.fsum(v["bMae"] for v in per_cell.values() if v["n"] > 0) / n_cells
                              if len(nonempty_cells) == n_cells else None),
            "aEqualCellMape": (math.fsum(v["aMape"] for v in per_cell.values() if v["n"] > 0) / n_cells
                               if len(nonempty_cells) == n_cells else None),
            "bEqualCellMape": (math.fsum(v["bMape"] for v in per_cell.values() if v["n"] > 0) / n_cells
                               if len(nonempty_cells) == n_cells else None),
            "cellCount": n_cells, "commonIssuedPairs": len(common),
            "onion10Cell": {
                "cellCount": len(HORIZONS),
                "aEqualCellMae": math.fsum(v["aMae"] for v in onion_cell_values) / len(HORIZONS)
                    if onion_complete else None,
                "bEqualCellMae": math.fsum(v["bMae"] for v in onion_cell_values) / len(HORIZONS)
                    if onion_complete else None,
                "aEqualCellMape": math.fsum(v["aMape"] for v in onion_cell_values) / len(HORIZONS)
                    if onion_complete else None,
                "bEqualCellMape": math.fsum(v["bMape"] for v in onion_cell_values) / len(HORIZONS)
                    if onion_complete else None,
            },
        }
        for metric, suffix in (("ae", "Mae"), ("ape", "Mape")):
            prepared = [{"product": r["product"], "horizon": r["horizon"],
                         "target": r["target"].isoformat(), "actual": r["actual"],
                         "a": r["_values"][a], "b": r["_values"][b]} for r in common]
            key = f"{cname}:{metric}"
            if len(nonempty_cells) == n_cells:
                dec = diagnostics.decompose_losses(prepared, a="a", b="b", metric=metric,
                                                   bottom_price_decile=False, expected_cells=comparison_cells)
                decompositions[key] = dec
                counted = math.fsum(v[f"a{suffix}"] - v[f"b{suffix}"] for v in per_cell.values()) / n_cells
                if not math.isclose(counted, dec["meanDifference"], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f"{key} contribution total does not reconcile")
            else:
                decompositions[key] = {"n": len(prepared), "cellCount": len(nonempty_cells),
                                        "expectedCellCount": n_cells,
                                        "reason": "common neural output support omits one or more horizons"}
    nn_rows = [r for r in scored if r["product"] == ONION]
    movement = {}
    for policy in ("ND", "NF", "HD", "HF"):
        operational = [(r, r["_values"]["B"] if r["_states"][policy] in ("fallback", "technical_fallback")
                        else r["_values"][policy]) for r in nn_rows]
        movement[policy] = {
            "eligibleOnionPairs": len(nn_rows),
            "outputChangedFromB": sum(value != row["_values"]["B"] for row, value in operational),
            "outputSameAsB": sum(value == row["_values"]["B"] for row, value in operational),
            "technicalFallbackToB": sum(r["_states"][policy] in ("fallback", "technical_fallback") for r in nn_rows),
            "issuedSameAsB": sum(r["_states"][policy] == "issued"
                                  and r["_values"][policy] == r["_values"]["B"] for r in nn_rows),
            "issuedDistinctFromB": sum(r["_states"][policy] == "issued"
                                       and r["_values"][policy] != r["_values"]["B"] for r in nn_rows),
            "changedPriceRate": (sum(value != row["_values"]["B"] for row, value in operational) / len(nn_rows)
                                 if nn_rows else None),
        }
    floor_rows = []
    for row in nn_rows:
        floor = max(0.5 * row["_values"]["B"] - row["actual"], 0.0)
        floor_rows.append({"product": ONION, "horizon": row["horizon"],
                           "target": row["target"].isoformat(), "actual": row["actual"],
                           "a": row["actual"] + floor, "b": row["actual"]})
    floor = diagnostics.decompose_losses(floor_rows, a="a", b="b", metric="ae",
                                        bottom_price_decile=False,
                                        expected_cells=[(ONION, h) for h in HORIZONS])
    floor_summary = {"rowsBelowHalfBaseline": sum(r["actual"] < 0.5 * r["_values"]["B"] for r in nn_rows),
                     "eligibleOnionPairs": len(nn_rows),
                     "macroMaeLowerBound": floor["meanDifference"],
                     "equallyWeightedTotalAbsoluteFloor": floor["meanDifference"] * len(nn_rows)}
    coverage = {"plannedEligibleRows": len(rows),
                "BIssuedRows": sum(r["_states"]["B"] == "issued" for r in rows),
                "BNotIssuedRows": sum(r["_states"]["B"] != "issued" for r in rows),
                "positiveActualOnBIssuedRows": positive_pairs,
                "missingOrInvalidActualOnBIssuedRows": len(missing_actual_rows)}
    for model in ("ND", "NF", "HD", "HF"):
        coverage[model + "IssuedOnBIssuedRows"] = sum(
            r["_states"]["B"] == "issued" and r["_states"][model] in ("issued", "fallback")
            for r in rows)
    return {"positiveBIssuedPairs": positive_pairs, "scoredPairs": len(scored),
            "coverage": coverage,
            "cellCount": len(cells), "metrics": metrics,
            "priceLossContributions": decompositions, "outputMovement": movement,
            "representationFloor": floor_summary, "scoredRows": scored,
            "missingActualRows": missing_actual_rows}


def _inference(scored_rows: Sequence[Mapping[str, Any]], phase: str,
               scope_products: Sequence[str]) -> dict[str, Any]:
    fixed_cells = [(ONION, h) for h in HORIZONS]
    family_rows = []
    for comparison, a, b in (("HF_vs_B", "HF", "B"), ("HF_vs_HD", "HF", "HD")):
        for horizon in HORIZONS:
            cell = [r for r in scored_rows if r["product"] == ONION and r["horizon"] == horizon]
            for loss in ("ae", "ape"):
                diff = {}
                for row in cell:
                    x, y = row["_values"][a], row["_values"][b]
                    delta = abs(x - row["actual"]) - abs(y - row["actual"])
                    if loss == "ape":
                        delta /= row["actual"]
                    day = row["target"]
                    if day in diff:
                        raise ValueError(f"duplicate {comparison}/{horizon}/{day} inference date")
                    diff[day] = delta
                for bandwidth in BANDWIDTHS:
                    family_rows.append({"comparison": comparison, "horizon": horizon,
                                        "loss": loss, "bandwidthDays": bandwidth,
                                        **stats._dm(diff, bandwidth)})
    for bandwidth in BANDWIDTHS:
        family = [r for r in family_rows if r["bandwidthDays"] == bandwidth]
        if len(family) != 40:
            raise ValueError(f"frozen {phase} inferential family must contain 40 tests")
        stats._holm(family)
    scope = set(scope_products)
    cells = [(p, h) for p in PRODUCTS if p in scope for h in HORIZONS]
    bootstrap = {}
    for comparison, a, b in (("HF_vs_B", "HF", "B"), ("HF_vs_HD", "HF", "HD")):
        pair_rows = [{"product": r["product"], "horizon": r["horizon"],
                      "target": r["target"].isoformat(), "actual": r["actual"],
                      "a": r["_values"][a], "b": r["_values"][b]}
                     for r in scored_rows if r["product"] in scope]
        bootstrap[comparison] = {}
        start, end, _ = PHASES[phase]
        for loss in ("ae", "ape"):
            bootstrap[comparison][loss] = {
                str(block): diagnostics.circular_block_bootstrap(
                    pair_rows, a="a", b="b", metric=loss, block=block,
                    replicates=REPLICATES, seed=SEED,
                    calendar_start=start, calendar_end=end)
                for block in (14, 28)
            }
    return {"familyDefinition": "(HF-B, HF-HD) x 10 onion horizons x (AE, APE)",
            "familyTestCountPerBandwidth": 40, "alpha": 0.05,
            "holmSeparatelyByBandwidth": True, "tests": family_rows,
            "meanDifferenceBlockBootstrap": bootstrap,
            "bootstrapResamplingUnit": "joint target-calendar date blocks; fixed original cell weights",
            "nonsignificanceInterpretation": "not evidence of equivalence"}


def score_phase(phase: str, rows: Sequence[Mapping[str, Any]],
                actuals: Mapping[tuple[str, dt.date], float], *,
                products: Sequence[str] = PRODUCTS,
                horizons: Sequence[int] = HORIZONS) -> dict[str, Any]:
    """Pure scoring core. Synthetic tests call this without saved data or training."""
    normalized = _normalized_rows(phase, rows, products=products, horizons=horizons)
    _validate_semantics(normalized, phase=phase)
    start, end, first_origin = PHASES[phase]
    eligible = [r for r in normalized if r["origin"] >= first_origin]
    use_full_policy = phase in ("V26", "T26") and set(products) == set(PRODUCTS)
    result = _score_rows(eligible, actuals, products=products, use_full_policy=use_full_policy)
    scored_rows = result.pop("scoredRows")
    missing_actual_rows = result.pop("missingActualRows")
    result["inference"] = (_inference(scored_rows, phase, products)
                           if phase in ("E25", "V26", "T26") else None)
    result["phase"] = phase
    result["targetCoverage"] = {
        "positiveActualBIssuedPairs": result["positiveBIssuedPairs"],
        "missingOrInvalidActualOnBIssuedRows": result["coverage"]["missingOrInvalidActualOnBIssuedRows"],
        "missingRows": missing_actual_rows,
    }
    result["dateRange"] = {"targetStart": start.isoformat(), "targetEnd": end.isoformat(),
                           "firstOrigin": first_origin.isoformat()}
    result["forecastGrid"] = {"rows": len(normalized), "eligibleRows": len(eligible),
                              "uniqueOrigins": len({r["origin"] for r in eligible}),
                              "uniqueTargets": len({r["target"] for r in eligible}),
                              "uniqueOriginTargetPairs": len({(r["origin"], r["target"]) for r in eligible})}
    return result


def score_documents(*, forecasts_by_phase: Mapping[str, Sequence[Mapping[str, Any]]],
                    targets: Mapping[tuple[str, dt.date], float],
                    d25_by_lambda: Mapping[int, Sequence[Mapping[str, Any]]] | None = None,
                    selected_lambda: int | None = None,
                    products: Sequence[str] = PRODUCTS) -> dict[str, Any]:
    """Score the four phases and recompute the one permitted D25 decision."""
    required = ("D25", "E25", "V26", "T26")
    if set(forecasts_by_phase) != set(required):
        raise ValueError("all four frozen forecast phases are required")
    d25_rows = (d25_by_lambda or {1: forecasts_by_phase["D25"]})
    if 1 not in d25_rows or set(d25_rows) - {1, 3}:
        raise ValueError("D25 must include lambda=1 and may include only lambda=3")
    base_d = score_phase("D25", d25_rows[1], targets, products=(ONION,), horizons=HORIZONS)
    d25_hf = base_d["metrics"]["HF_vs_B"]
    base_improves = (d25_hf["aEqualCellMape"] < d25_hf["bEqualCellMape"]
                     and d25_hf["aEqualCellMae"] <= d25_hf["bEqualCellMae"])
    should_have_repair = d25_hf["aEqualCellMae"] > d25_hf["bEqualCellMae"]
    if should_have_repair != (3 in d25_rows):
        raise ValueError("D25 lambda=3 artifact presence violates the one-shot MAE trigger")
    scored_d = {1: base_d}
    if 3 in d25_rows:
        scored_d[3] = score_phase("D25", d25_rows[3], targets, products=(ONION,), horizons=HORIZONS)
    chosen = _choose_lambda(scored_d)
    if selected_lambda is not None and int(selected_lambda) != chosen:
        raise ValueError("D25 selection artifact disagrees with recomputed frozen rule")
    phase_scores = {"D25": scored_d[chosen]}
    for phase in ("E25", "V26", "T26"):
        phase_products = (ONION,) if phase == "E25" else tuple(products)
        phase_scores[phase] = score_phase(phase, forecasts_by_phase[phase], targets,
                                          products=phase_products)
    return _assemble_report(phase_scores, chosen, base_improves,
                            d25_lambda_scores={str(k): v for k, v in scored_d.items()})


def _choose_lambda(scored: Mapping[int, Mapping[str, Any]]) -> int:
    eligible = []
    ranks = {}
    for lam, phase in scored.items():
        metric = phase["metrics"]["HF_vs_B"]
        mae_a, mae_b = metric["aEqualCellMae"], metric["bEqualCellMae"]
        mape_a, mape_b = metric["aEqualCellMape"], metric["bEqualCellMape"]
        if mape_a < mape_b and mae_a <= mae_b:
            eligible.append(lam)
        ratios = (mape_a / mape_b if mape_b else (1.0 if mape_a == 0 else math.inf),
                  mae_a / mae_b if mae_b else (1.0 if mae_a == 0 else math.inf))
        ranks[lam] = max(ratios)
    if eligible:
        return int(min(eligible, key=lambda lam: (
            scored[lam]["metrics"]["HF_vs_B"]["aEqualCellMape"], lam)))
    if 1 not in scored:
        raise ValueError("base lambda score is missing")
    if 3 not in scored:
        return 1
    # The base setting wins worst-ratio ties; there is no secondary criterion.
    if ranks[3] < ranks[1] - 1e-8:
        return 3
    return 1


def _assemble_report(phase_scores: Mapping[str, Mapping[str, Any]],
                     selected_lambda: int, base_d_improves: bool,
                     d25_lambda_scores: Mapping[str, Any]) -> dict[str, Any]:
    e, v, t = (phase_scores[name] for name in ("E25", "V26", "T26"))
    v_hf, t_hf = v["metrics"]["HF_vs_B"], t["metrics"]["HF_vs_B"]
    v_cov = v["coverage"]["HFIssuedOnBIssuedRows"] == v["coverage"]["BIssuedRows"]
    t_cov = t["coverage"]["HFIssuedOnBIssuedRows"] == t["coverage"]["BIssuedRows"]
    v_ok = (v_hf["aEqualCellMape"] < v_hf["bEqualCellMape"]
            and v_hf["aEqualCellMae"] <= v_hf["bEqualCellMae"] and v_cov)
    t_ok = (t_hf["aEqualCellMape"] < t_hf["bEqualCellMape"]
            and t_hf["aEqualCellMae"] <= t_hf["bEqualCellMae"] and t_cov)
    target_reduction_pp = t_hf["bEqualCellMape"] - t_hf["aEqualCellMape"]
    onion_reduction_pp = (t_hf["onion10Cell"]["bEqualCellMape"]
                          - t_hf["onion10Cell"]["aEqualCellMape"])
    if v_ok and t_ok and target_reduction_pp >= 0.125:
        status = "completed_price_gain"
    elif v_ok and t_ok:
        status = "completed_small_gain"
    else:
        hf_hd = phase_scores["T26"]["metrics"]["HF_vs_HD"]
        v_hf_hd = v["metrics"]["HF_vs_HD"]
        v_recovered = (v_hf_hd["aEqualCellMape"] < v_hf_hd["bEqualCellMape"]
                       and v_hf_hd["aEqualCellMae"] <= v_hf_hd["bEqualCellMae"] and v_cov)
        t_recovered = (hf_hd["aEqualCellMape"] < hf_hd["bEqualCellMape"]
                       and hf_hd["aEqualCellMae"] <= hf_hd["bEqualCellMae"] and t_cov)
        v_changed = v["outputMovement"]["NF"]["issuedDistinctFromB"] > 0
        t_changed = t["outputMovement"]["NF"]["issuedDistinctFromB"] > 0
        status = ("completed_recovery_only" if v_recovered and t_recovered
                  and v_changed and t_changed else "completed_no_recovery")
    return {"schemaVersion": 1, "experiment": "FINAL_ENSEMBLE_LOSS_ONION_V2",
            "status": status, "selectedLambda": selected_lambda,
            "selection": {"D25BaseHFImprovesB": base_d_improves,
                          "scoresByLambda": d25_lambda_scores,
                          "rule": "lambda3 is eligible only if base D25 HF MAE > B; select eligible by HF MAPE, otherwise worst-relative-ratio rank; tie to base"},
            "phases": dict(phase_scores),
            "finalGoal": {"V26PriceGate": v_ok, "T26PriceGate": t_ok,
                          "V26CoverageMaintained": v_cov, "T26CoverageMaintained": t_cov,
                          "T26HfMapeReductionPercentagePoints": target_reduction_pp,
                          "T26OnionHfMapeReductionPercentagePoints": onion_reduction_pp,
                          "requiredT26MapeReductionPercentagePoints": 0.125,
                          "requiredT26OnionMapeReductionPercentagePoints": 0.5,
                          "E25DirectionAndMetrics": e["metrics"]["HF_vs_B"]}}


def _read_json(path: Path) -> dict[str, Any]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"expected JSON object at {path}")
    return doc


def _validate_config(out_dir: Path) -> dict[str, Any]:
    config_path = out_dir / "resolved-config.json"
    config = _read_json(config_path)
    unsigned = dict(config)
    recorded = unsigned.pop("configSha256", None)
    if recorded != sha256_bytes(canonical(unsigned)):
        raise ValueError("resolved config self-digest mismatch")
    if config.get("experiment") != "FINAL_ENSEMBLE_LOSS_ONION_V2":
        raise ValueError("resolved config experiment identity mismatch")
    for rel, digest in config.get("producerCodeSha256", {}).items():
        path = ROOT / rel
        if not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"resolved config source code hash mismatch: {rel}")
    scorer_rel = "agri_data/final_ensemble_onion_score.py"
    if config.get("producerCodeSha256", {}).get(scorer_rel) != file_sha256(Path(__file__)):
        raise ValueError("resolved config does not pin this scorer version")
    protocol_path = ROOT / config["protocolPath"]
    if file_sha256(protocol_path) != config.get("protocolSha256"):
        raise ValueError("frozen protocol file hash mismatch")
    price_path = ROOT / config["priceSnapshotPath"]
    if not price_path.is_file() or file_sha256(price_path) != config.get("priceSnapshotSha256"):
        raise ValueError("pinned target price snapshot hash mismatch")
    if price_path.resolve() != price_helper.PRICES_PATH.resolve():
        raise ValueError("target snapshot path differs from the registered price source")
    prior_path = ROOT / config["priorResolvedConfigPath"]
    if not prior_path.is_file() or file_sha256(prior_path) != config.get("priorResolvedConfigSha256"):
        raise ValueError("prior frozen model config hash mismatch")
    source_path = ROOT / config["sourceCachePath"]
    if not source_path.is_file() or file_sha256(source_path) != config.get("sourceCacheFileSha256"):
        raise ValueError("source cache file hash mismatch")
    return config


def _validate_baseline_cache(out_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any], str]:
    cache_path, manifest_path = out_dir / "baseline-cache.json", out_dir / "baseline-cache-manifest.json"
    cache_raw, manifest_raw = cache_path.read_bytes(), manifest_path.read_bytes()
    cache, manifest = json.loads(cache_raw), json.loads(manifest_raw)
    unsigned = dict(manifest)
    recorded = unsigned.pop("manifestSha256", None)
    if recorded != sha256_bytes(canonical(unsigned)):
        raise ValueError("baseline cache manifest self-digest mismatch")
    if manifest.get("phase") != "answer_free" or manifest.get("targetValuesAttached") is not False:
        raise ValueError("baseline cache is not answer-free")
    if manifest.get("complete") is not True:
        raise ValueError("baseline cache is incomplete")
    if manifest.get("cacheFileSha256") != sha256_bytes(cache_raw):
        raise ValueError("baseline cache raw file hash mismatch")
    if manifest.get("cacheRowsSha256") != sha256_bytes(canonical(cache.get("rows"))):
        raise ValueError("baseline cache row digest mismatch")
    if cache.get("phase") != "answer_free" or cache.get("targetValuesAttached") is not False:
        raise ValueError("baseline cache payload contains answers or wrong phase")
    index = {}
    for row in cache.get("rows", []):
        key = (row["product"], str(row["origin"]), str(row["target"]), int(row["horizon"]))
        if key in index:
            raise ValueError(f"duplicate baseline cache row key {key}")
        index[key] = row
    return index, manifest, sha256_bytes(manifest_raw)


def _key(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    return (str(row["product"]), _date(row["origin"]).isoformat(),
            _date(row["target"]).isoformat(), int(row["horizon"]))


def _full_policy_rows(phase: str, onion_rows: Sequence[Mapping[str, Any]],
                      baseline_index: Mapping[tuple[str, str, str, int], Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Rebuild V/T's 40-cell policy from saved onion forecasts and pinned B copies."""
    start, end, first_origin = PHASES[phase]
    onion = {_key(row): dict(row) for row in onion_rows}
    result = []
    for key, base in sorted(baseline_index.items()):
        product, origin_s, target_s, horizon = key
        origin, target = _date(origin_s), _date(target_s)
        if product not in PRODUCTS or origin < first_origin or not start <= target <= end:
            continue
        if base.get("BSourceStatus") not in ("issued", "fallback") or positive(base.get("B")) is None:
            continue
        if product == ONION:
            row = onion.get(key)
            if row is None or positive(row.get("B")) != positive(base.get("B")):
                raise ValueError(f"{phase} onion row does not match pinned B at {key}")
            result.append(row)
        else:
            b = base["B"]
            result.append({"product": product, "origin": origin_s, "target": target_s,
                           "horizon": horizon, "B": b, "BStatus": "issued",
                           "ND": b, "NF": b, "HD": b, "HF": b,
                           "NDStatus": "issued", "NFStatus": "issued",
                           "HDStatus": "issued", "HFStatus": "issued",
                           "BSourceStatus": base["BSourceStatus"]})
    expected = _expected_keys(phase, PRODUCTS)
    found = {_key(row) for row in result}
    if found != expected:
        raise ValueError(f"{phase} full 40-cell policy grid differs from pinned baseline: missing={len(expected-found)} extra={len(found-expected)}")
    return result


def _verify_phase_doc(doc: Mapping[str, Any], phase: str, penalty: int,
                      baseline_index: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
                      baseline_manifest_sha: str, config_sha: str) -> list[dict[str, Any]]:
    if doc.get("phase") != phase or doc.get("targetValuesAttached") is not False:
        raise ValueError(f"{phase} forecast must be persisted answer-free")
    if doc.get("configSha256") != config_sha:
        raise ValueError(f"{phase} forecast config identity mismatch")
    if phase != "D25" and not isinstance(doc.get("d25SelectionSha256"), str):
        raise ValueError(f"{phase} forecast lacks its frozen D25 selection pin")
    if doc.get("baselineCacheManifestSha256") != baseline_manifest_sha:
        raise ValueError(f"{phase} forecast baseline manifest pin mismatch")
    rows = doc.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{phase} forecast rows are malformed")
    expected = _expected_keys(phase, (ONION,))
    if len(rows) != len(expected):
        raise ValueError(f"{phase} onion forecast row count differs from frozen grid")
    found = set()
    for row in rows:
        key = _key(row)
        if key not in expected or key in found:
            raise ValueError(f"unexpected or duplicate {phase} key {key}")
        found.add(key)
        if row.get("lambda") != penalty:
            raise ValueError(f"{phase} row lambda differs from the selected setting")
        if row.get("cutoff") != (_date(row["origin"]) - dt.timedelta(days=1)).isoformat():
            raise ValueError(f"{phase} forecast cutoff mismatch at {key}")
        cached = baseline_index.get(key)
        if cached is None or row.get("B") != cached.get("B") or row.get("BSourceStatus") != cached.get("BSourceStatus"):
            raise ValueError(f"{phase} B value/status differs from the answer-free cache at {key}")
        if row.get("replayAgreement") is not True:
            raise ValueError(f"{phase} forecast lacks successful reload replay at {key}")
        for suffix in ("D", "F"):
            relpath, expected_hash = row.get("weightPath" + suffix), row.get("weightSha256" + suffix)
            if not isinstance(relpath, str) or not isinstance(expected_hash, str):
                raise ValueError(f"{phase} row lacks {suffix} checkpoint provenance at {key}")
            weight_path = (ROOT / relpath).resolve()
            if not weight_path.is_relative_to(ROOT.resolve()) or not weight_path.is_file():
                raise ValueError(f"{phase} {suffix} checkpoint path is unsafe or missing at {key}")
            if file_sha256(weight_path) != expected_hash:
                raise ValueError(f"{phase} {suffix} checkpoint digest mismatch at {key}")
        if not row.get("reasonD") and row.get("statusD") == "technical_fallback":
            raise ValueError(f"{phase} D numerical fallback lacks a reason at {key}")
        if not row.get("reasonF") and row.get("statusF") == "technical_fallback":
            raise ValueError(f"{phase} F numerical fallback lacks a reason at {key}")
    if found != expected:
        raise ValueError(f"{phase} forecast grid is incomplete")
    return rows


def _parquet_value_equal(left: Any, right: Any) -> bool:
    a, b = positive(left), positive(right)
    if a is None or b is None:
        return left is None or (isinstance(left, float) and math.isnan(left)) if b is None else False
    return a == b


def _validate_parquet(out_dir: Path, manifest: Mapping[str, Any],
                      phases: Mapping[str, Sequence[Mapping[str, Any]]],
                      baseline_index: Mapping[tuple[str, str, str, int], Mapping[str, Any]],
                      selected_lambda: int) -> dict[str, Any]:
    import pandas as pd

    parquet_path = ROOT / manifest["path"]
    if not parquet_path.is_file() or file_sha256(parquet_path) != manifest.get("fileSha256"):
        raise ValueError("final Parquet file hash mismatch")
    if manifest.get("targetValuesAttached") is not False:
        raise ValueError("final Parquet artifact unexpectedly contains targets")
    frame = pd.read_parquet(parquet_path)
    records = frame.to_dict(orient="records")
    key_fields = ("phase", "product", "origin", "target", "horizon")
    actual_keys = [(str(r["phase"]), str(r["product"]), str(r["origin"]),
                    str(r["target"]), int(r["horizon"])) for r in records]
    if len(actual_keys) != len(set(actual_keys)):
        raise ValueError("final Parquet key uniqueness failed")
    if len(records) != manifest.get("rows"):
        raise ValueError("final Parquet row count differs from manifest")
    canonical_keys = sorted(actual_keys)
    if manifest.get("keyDigest") != sha256_bytes(canonical(canonical_keys)):
        raise ValueError("final Parquet canonical key digest mismatch")
    map_records = {tuple(str(r[k]) if k != "horizon" else int(r[k]) for k in key_fields): r
                   for r in records}
    expected_keys = set()
    expected_source: dict[tuple[str, str, str, str, int], Mapping[str, Any]] = {}
    for phase, rows in phases.items():
        forecast_map = {_key(r): r for r in rows}
        if phase in ("D25", "E25"):
            for key, source in forecast_map.items():
                full = (phase, *key)
                expected_keys.add(full)
                expected_source[full] = source
        else:
            target_start, target_end, first_origin = PHASES[phase]
            for key, base in baseline_index.items():
                product, origin_s, target_s, horizon = key
                origin, target = _date(origin_s), _date(target_s)
                if product not in PRODUCTS or origin < first_origin or not target_start <= target <= target_end:
                    continue
                if base.get("BSourceStatus") not in ("issued", "fallback") or positive(base.get("B")) is None:
                    continue
                full = (phase, *key)
                expected_keys.add(full)
                expected_source[full] = forecast_map.get(key, base)
    if set(map_records) != expected_keys:
        raise ValueError(f"final Parquet forecast grid mismatch: missing={len(expected_keys-set(map_records))} extra={len(set(map_records)-expected_keys)}")
    for key, source in expected_source.items():
        saved = map_records[key]
        bval = source.get("B")
        if not _parquet_value_equal(saved.get("B"), bval):
            raise ValueError(f"Parquet B copy differs from source row {key}")
        if key[1] == ONION:
            for field in ("ND", "NF", "HD", "HF"):
                if not _parquet_value_equal(saved.get(field), source.get(field)):
                    raise ValueError(f"Parquet {field} differs from phase source row {key}")
            for field in ("BSourceStatus", "statusD", "statusF", "statusHD", "statusHF",
                          "reasonD", "reasonF"):
                if saved.get(field) != source.get(field):
                    raise ValueError(f"Parquet {field} differs from phase source row {key}")
            if saved.get("finalStatus") != source.get("statusF"):
                raise ValueError(f"Parquet finalStatus differs from source statusF {key}")
            if not _parquet_value_equal(saved.get("finalH"), source.get("HF")):
                raise ValueError(f"Parquet finalH differs from the frozen HF blend {key}")
        else:
            for field in ("ND", "NF"):
                if not _parquet_value_equal(saved.get(field), None):
                    raise ValueError(f"non-onion Parquet {field} must remain unissued {key}")
            for field in ("HD", "HF", "finalH"):
                if not _parquet_value_equal(saved.get(field), bval):
                    raise ValueError(f"non-onion Parquet {field} differs from exact B copy {key}")
            for field in ("statusD", "statusF", "statusHD", "statusHF", "finalStatus"):
                if saved.get(field) != "baseline_copy":
                    raise ValueError(f"non-onion Parquet {field} is not baseline_copy {key}")
            if saved.get("BSourceStatus") != source.get("BSourceStatus"):
                raise ValueError(f"non-onion Parquet B source status differs from pinned cache {key}")
    return {"rows": len(records), "keyDigest": manifest["keyDigest"],
            "countsByPhaseProduct": manifest.get("countsByPhaseProduct", {})}


def score_saved_run(out_dir: Path = OUT) -> dict[str, Any]:
    """Validate immutable producer artifacts, join targets, and recompute scores.

    This entry point is intentionally fail-closed. It is not called by the
    offline scorer tests and must run only after the authorized producer stage.
    """
    config = _validate_config(out_dir)
    baseline_index, baseline_manifest, baseline_manifest_sha = _validate_baseline_cache(out_dir)
    if config.get("baselineCacheManifestSha256") is not None and config["baselineCacheManifestSha256"] != baseline_manifest_sha:
        raise ValueError("resolved config baseline manifest pin mismatch")
    forecast_manifest = _read_json(out_dir / "forecast-manifest.json")
    if forecast_manifest.get("configSha256") != config["configSha256"]:
        raise ValueError("forecast manifest config identity mismatch")
    entries = {}
    for row in forecast_manifest.get("forecasts", []):
        entry_key = (str(row["phase"]), int(float(row["lambda"])))
        if entry_key in entries:
            raise ValueError(f"duplicate forecast manifest entry {entry_key}")
        entries[entry_key] = row
    selection = _read_json(out_dir / "D25-selection.json")
    chosen = int(float(selection.get("selectedLambda")))
    if (chosen not in (1, 3) or selection.get("downstreamLambda") != chosen
            or selection.get("phase") != "D25" or selection.get("configSha256") != config["configSha256"]):
        raise ValueError("D25 selection artifact is malformed")
    wanted = {("D25", 1)}
    if ("D25", 3) in entries:
        wanted.add(("D25", 3))
    wanted.update((phase, chosen) for phase in ("E25", "V26", "T26"))
    if not wanted.issubset(entries):
        raise ValueError("forecast manifest omits a required phase/lambda source")
    expected_score_refs = {f"L{lam}" for phase, lam in wanted if phase == "D25"}
    refs = selection.get("inputScoreSha256", {})
    if set(refs) != expected_score_refs:
        raise ValueError("D25 selection score references do not match its candidate inventory")
    for phase, lam in sorted((key for key in wanted if key[0] == "D25")):
        score_path = out_dir / f"score-D25-L{lam}.json"
        if not score_path.is_file() or file_sha256(score_path) != refs[f"L{lam}"]:
            raise ValueError(f"D25 selection input score pin mismatch for lambda={lam}")
        score_doc = _read_json(score_path)
        if (score_doc.get("configSha256") != config["configSha256"]
                or score_doc.get("phase") != "D25" or int(float(score_doc.get("lambda", -1))) != lam):
            raise ValueError(f"D25 score document is bound to the wrong run: lambda={lam}")
        if score_doc.get("forecastFileSha256") != entries[("D25", lam)].get("fileSha256"):
            raise ValueError(f"D25 score document forecast hash mismatch: lambda={lam}")
    phase_docs = {}
    phase_rows = {}
    for phase, penalty in sorted(wanted):
        entry = entries[(phase, penalty)]
        if entry.get("targetValuesAttached") is not False:
            raise ValueError(f"{phase} manifest entry is not answer-free")
        path = ROOT / entry["path"]
        if not path.is_file() or file_sha256(path) != entry.get("fileSha256"):
            raise ValueError(f"{phase} phase forecast file hash mismatch")
        doc = _read_json(path)
        if phase != "D25" and doc.get("d25SelectionSha256") != file_sha256(out_dir / "D25-selection.json"):
            raise ValueError(f"{phase} forecast D25 selection pin mismatch")
        rows = _verify_phase_doc(doc, phase, penalty, baseline_index,
                                 baseline_manifest_sha, config["configSha256"])
        keys = [_key(r) for r in rows]
        if len(rows) != entry.get("rows") or sha256_bytes(canonical(keys)) != entry.get("keyDigest"):
            raise ValueError(f"{phase} forecast manifest row/key digest mismatch")
        phase_docs[(phase, penalty)] = doc
        phase_rows[(phase, penalty)] = rows
    d25_by_lambda = {lam: phase_rows[("D25", lam)] for phase, lam in wanted if phase == "D25"}
    forecasts_by_phase = {"D25": phase_rows[("D25", chosen)], "E25": phase_rows[("E25", chosen)],
                          "V26": _full_policy_rows("V26", phase_rows[("V26", chosen)], baseline_index),
                          "T26": _full_policy_rows("T26", phase_rows[("T26", chosen)], baseline_index)}
    price_path = ROOT / config["priceSnapshotPath"]
    targets = _actual_map(_read_json(price_path))
    report = score_documents(forecasts_by_phase=forecasts_by_phase, targets=targets,
                             d25_by_lambda=d25_by_lambda, selected_lambda=chosen,
                             products=PRODUCTS)
    for lam in d25_by_lambda:
        saved = _read_json(out_dir / f"score-D25-L{lam}.json")
        recomputed = report["selection"]["scoresByLambda"][str(lam)]
        if saved.get("priceSnapshotSha256") != config.get("priceSnapshotSha256"):
            raise ValueError(f"D25 saved score target snapshot pin mismatch for lambda={lam}")
        if (saved.get("pairedRows") != recomputed["positiveBIssuedPairs"]
                or saved.get("uniqueForecastKeys") != recomputed["positiveBIssuedPairs"]):
            raise ValueError(f"D25 saved score pair count differs from recomputed paired support for lambda={lam}")
        saved_horizons = {int(r["horizon"]): r for r in saved.get("horizons", [])}
        if set(saved_horizons) != set(HORIZONS):
            raise ValueError(f"D25 saved score horizon support is incomplete for lambda={lam}")
        for horizon in HORIZONS:
            cell = recomputed["metrics"]["HF_vs_B"]["cells"][f"{ONION}:h{horizon}"]
            hd = recomputed["metrics"]["HF_vs_HD"]["cells"][f"{ONION}:h{horizon}"]
            saved_cell = saved_horizons[horizon]
            if saved_cell.get("n") != cell["n"]:
                raise ValueError(f"D25 saved score paired keys differ at lambda={lam}, h={horizon}")
            pairs = (("H_F_MAE", "aMae", cell, 1.0), ("B_MAE", "bMae", cell, 1.0),
                     ("H_D_MAE", "bMae", hd, 1.0),
                     ("H_F_MAPE", "aMape", cell, 100.0),
                     ("B_MAPE", "bMape", cell, 100.0),
                     ("H_D_MAPE", "bMape", hd, 100.0))
            for source_name, metric_name, current, scale in pairs:
                if not math.isclose(float(saved_cell[source_name]) * scale,
                                    float(current[metric_name]), rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError(f"D25 saved metric differs from same-pair recomputation: lambda={lam}, h={horizon}, metric={source_name}")
    base_d25 = report["selection"]["scoresByLambda"]["1"]["metrics"]["HF_vs_B"]
    required_l3 = base_d25["aEqualCellMae"] > base_d25["bEqualCellMae"]
    if bool(selection.get("lambda3Required")) != required_l3:
        raise ValueError("D25 selection lambda3 trigger differs from recomputed target pairs")
    parquet_manifest = _read_json(out_dir / "forecasts-parquet-manifest.json")
    if parquet_manifest.get("configSha256") != config["configSha256"]:
        raise ValueError("Parquet manifest config hash mismatch")
    if parquet_manifest.get("baselineCacheManifestSha256") != baseline_manifest_sha:
        raise ValueError("Parquet manifest baseline cache pin mismatch")
    if file_sha256(out_dir / "D25-selection.json") != parquet_manifest.get("D25SelectionSha256"):
        raise ValueError("Parquet manifest D25 selection pin mismatch")
    for phase in ("D25", "E25", "V26", "T26"):
        source = parquet_manifest.get("forecastSourceFiles", {}).get(phase)
        entry = entries[(phase, chosen)]
        if not source or source.get("sha256") != entry.get("fileSha256"):
            raise ValueError(f"Parquet manifest {phase} source pin mismatch")
    report["parquetValidation"] = _validate_parquet(out_dir, parquet_manifest,
        {phase: phase_rows[(phase, chosen)] for phase in ("D25", "E25", "V26", "T26")},
        baseline_index, chosen)
    report["integrity"] = {"configSha256": config["configSha256"],
        "baselineCacheManifestSha256": baseline_manifest_sha,
        "forecastManifestSha256": file_sha256(out_dir / "forecast-manifest.json"),
        "parquetManifestSha256": file_sha256(out_dir / "forecasts-parquet-manifest.json"),
        "targetSnapshotSha256": config["priceSnapshotSha256"],
        "allSourceForecastsAnswerFree": True}
    result_path = out_dir / "score-report.json"
    raw = canonical(report)
    if result_path.exists() and result_path.read_bytes() != raw:
        raise FileExistsError("refusing to replace a different frozen score report")
    if not result_path.exists():
        tmp = result_path.with_suffix(".json.tmp")
        tmp.write_bytes(raw); tmp.replace(result_path)
    return report


def main() -> None:
    """Validate the frozen default run and print a compact saved-report summary."""
    report = score_saved_run(OUT)
    summary = {
        "scoreReportPath": str(OUT / "score-report.json"),
        "status": report["status"],
        "selectedLambda": report["selectedLambda"],
        "finalGoal": report["finalGoal"],
        "phaseMetrics": {
            phase: {
                "HFvsB": scores["metrics"]["HF_vs_B"],
                "HFvsHD": scores["metrics"]["HF_vs_HD"],
                "coverage": scores["coverage"],
            }
            for phase, scores in report["phases"].items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
