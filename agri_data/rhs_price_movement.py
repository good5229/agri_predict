"""Pure price-movement and error-transfer diagnostics for RHS forecasts.

The module consumes already-produced B/D/C/P forecast rows and actual prices.
It never fits a model, chooses a candidate, or performs statistical inference.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from typing import Any


CONDITIONS = ("B", "D", "C", "P")
COMPARISONS = (("D", "B"), ("C", "B"), ("P", "B"))
LAYERS = ("direct", "blend")
HORIZONS = tuple(range(1, 11))
PHASE_COUNTS = {"V": {"paired": 651, "issued": 755}, "T": {"paired": 650, "issued": 790}}
PHASE_MONTHS = {
    "V": ("2026-04", "2026-05", "2026-06"),
    "T": ("2026-07", "2026-08", "2026-09"),
}
_REL_TOL = 1e-9
_ABS_TOL = 1e-6


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return value


def _key_from_row(row: Mapping[str, Any]) -> tuple[str, str, str, int]:
    key = row.get("key")
    if not isinstance(key, Mapping):
        raise ValueError("forecast row must contain a key mapping")
    product = key.get("product")
    origin = key.get("originDate")
    target = key.get("targetDate")
    horizon = key.get("horizon")
    if not all(isinstance(x, str) and x for x in (product, origin, target)):
        raise ValueError("forecast key product, originDate, and targetDate must be nonempty strings")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
        raise ValueError("forecast key horizon must be an integer from 1 through 10")
    return product, origin, target, horizon


def _key_from_expected(key: Any) -> tuple[str, str, str, int]:
    if not isinstance(key, (tuple, list)) or len(key) != 4:
        raise ValueError("expected comparison keys must be (product, origin, target, horizon) tuples")
    product, origin, target, horizon = key
    if not all(isinstance(x, str) and x for x in (product, origin, target)):
        raise ValueError("expected key product, origin, and target must be nonempty strings")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
        raise ValueError("expected key horizon must be an integer from 1 through 10")
    return product, origin, target, horizon


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=_REL_TOL, abs_tol=_ABS_TOL + 1e-8 * abs(expected))


def _validate_condition(row: Mapping[str, Any], key: tuple[str, str, str, int], condition: str) -> dict[str, float]:
    state = row["conditions"].get(condition)
    if not isinstance(state, Mapping):
        raise ValueError(f"{key}: missing {condition} condition")
    if state.get("status") != "issued":
        raise ValueError(f"{key}: {condition} status must be issued; fallbacks are not in this diagnostic")
    direct = _number(state.get("directPrediction"), f"{key} {condition} directPrediction", positive=True)
    blend = _number(state.get("blendPrediction"), f"{key} {condition} blendPrediction", positive=True)
    sarima = _number(state.get("sarimaPrediction"), f"{key} {condition} sarimaPrediction", positive=True)
    expected_blend = 0.5 * (direct + sarima)
    if not _close(blend, expected_blend):
        raise ValueError(f"{key}: {condition} blend is not the saved 50:50 direct/SARIMA combination")
    return {"direct": direct, "blend": blend, "sarima": sarima}


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(value * weight for value, weight in zip(values, weights))


def _metric_record(records: Sequence[Mapping[str, Any]], weights: Sequence[float]) -> dict[str, Any]:
    if not records:
        return {
            "n": 0,
            "baselineMae": None,
            "conditionMae": None,
            "baselineMapePct": None,
            "conditionMapePct": None,
            "movementMae": None,
            "movementMapePct": None,
            "improvementMae": None,
            "improvementMapePp": None,
            "efficiencyMae": None,
            "efficiencyMape": None,
            "lossReducingMae": None,
            "lossIncreasingMae": None,
            "lossReducingMapePp": None,
            "lossIncreasingMapePp": None,
        }

    def mean(field: str) -> float:
        result = _weighted_mean([float(item[field]) for item in records], weights)
        assert result is not None
        return result

    move_mae = mean("movementMae")
    move_mape = mean("movementMapePct")
    baseline_mae = mean("baselineError")
    condition_mae = mean("conditionError")
    baseline_mape = mean("baselineMapePct")
    condition_mape = mean("conditionMapePct")
    improve_mae = mean("improvementMae")
    improve_mape = mean("improvementMapePp")
    if not _close(mean("lossReducingMae") + mean("lossIncreasingMae"), move_mae):
        raise ValueError("MAE reducing/increasing movement components do not sum to movement")
    if not _close(mean("lossReducingMae") - mean("lossIncreasingMae"), improve_mae):
        raise ValueError("MAE reducing/increasing movement components do not reproduce improvement")
    if not _close(mean("lossReducingMapePp") + mean("lossIncreasingMapePp"), move_mape):
        raise ValueError("MAPE reducing/increasing movement components do not sum to movement")
    if not _close(mean("lossReducingMapePp") - mean("lossIncreasingMapePp"), improve_mape):
        raise ValueError("MAPE reducing/increasing movement components do not reproduce improvement")
    if not _close(baseline_mae - condition_mae, improve_mae):
        raise ValueError("baseline/candidate MAE do not reproduce improvement")
    if not _close(baseline_mape - condition_mape, improve_mape):
        raise ValueError("baseline/candidate MAPE do not reproduce improvement")
    eff_mae = improve_mae / move_mae if move_mae > 0 else None
    eff_mape = improve_mape / move_mape if move_mape > 0 else None
    for label, value in (("MAE", eff_mae), ("MAPE", eff_mape)):
        if value is not None and abs(value) > 1.0 + 1e-8:
            raise ValueError(f"{label} improvement/movement efficiency fell outside [-1, 1]")
    return {
        "n": len(records),
        "baselineMae": baseline_mae,
        "conditionMae": condition_mae,
        "baselineMapePct": baseline_mape,
        "conditionMapePct": condition_mape,
        "movementMae": move_mae,
        "movementMapePct": move_mape,
        "improvementMae": improve_mae,
        "improvementMapePp": improve_mape,
        "efficiencyMae": eff_mae,
        "efficiencyMape": eff_mape,
        "lossReducingMae": mean("lossReducingMae"),
        "lossIncreasingMae": mean("lossIncreasingMae"),
        "lossReducingMapePp": mean("lossReducingMapePp"),
        "lossIncreasingMapePp": mean("lossIncreasingMapePp"),
    }


def _rms(changes: Sequence[float]) -> float | None:
    if not changes:
        return None
    return math.sqrt(sum(value * value for value in changes) / len(changes))


def _date_concentration(date_rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = [float(row["fixedGlobalContribution"][field]) for row in date_rows]
    absolute = [abs(value) for value in values]
    total_absolute = sum(absolute)
    return {
        "positiveDateCount": sum(value > 0 for value in values),
        "negativeDateCount": sum(value < 0 for value in values),
        "zeroDateCount": sum(value == 0 for value in values),
        "grossPositiveContribution": sum(value for value in values if value > 0),
        "grossNegativeContributionAbs": sum(-value for value in values if value < 0),
        "sumAbsoluteNetDateContribution": total_absolute,
        "topFiveAbsoluteNetDateShare": (
            sum(sorted(absolute, reverse=True)[:5]) / total_absolute if total_absolute > 0 else None
        ),
    }


def diagnose_price_movement(
    manifest_rows: Sequence[Mapping[str, Any]],
    actuals: Mapping[tuple[str, str], float],
    *,
    phase: str,
    expected_comparison_keys: Collection[tuple[str, str, str, int]],
) -> dict[str, Any]:
    """Summarize fixed-support B/D/C/P movement and its error consequences.

    ``manifest_rows`` contains radish forecast rows for one RHS phase. The
    expected keys are the original common positive-actual B/D/C/P support and
    are shared by all three comparisons and both direct/blend layers. For the
    production phases, V/T key counts are pinned at 651/650 paired keys and
    755/790 all-issued keys. Use ``phase='synthetic'`` for small unit fixtures.

    All aggregate movement/improvement values use equal weight over the ten
    horizons, with key weight ``1 / (10 * n_h)``. ``allIssuedRms`` is instead
    a pooled key-level diagnostic over every issued forecast, without actuals.
    """
    if phase not in {"V", "T", "synthetic"}:
        raise ValueError("phase must be V, T, or synthetic")
    if not isinstance(manifest_rows, Sequence) or isinstance(manifest_rows, (str, bytes)):
        raise ValueError("manifest_rows must be a sequence of forecast rows")
    expected_list = [_key_from_expected(key) for key in expected_comparison_keys]
    expected = set(expected_list)
    if len(expected) != len(expected_list):
        raise ValueError("expected comparison keys contain duplicates")
    if phase in PHASE_COUNTS and len(expected) != PHASE_COUNTS[phase]["paired"]:
        raise ValueError(f"{phase} expected comparison support must contain {PHASE_COUNTS[phase]['paired']} keys")

    rows_by_key: dict[tuple[str, str, str, int], Mapping[str, Any]] = {}
    values_by_key: dict[tuple[str, str, str, int], dict[str, dict[str, float]]] = {}
    for row in manifest_rows:
        if not isinstance(row, Mapping):
            raise ValueError("each manifest row must be a mapping")
        key = _key_from_row(row)
        if key in rows_by_key:
            raise ValueError(f"duplicate forecast row for {key}")
        if row.get("eligible") is not True:
            raise ValueError(f"{key}: RHS movement row must be eligible")
        conditions = row.get("conditions")
        if not isinstance(conditions, Mapping):
            raise ValueError(f"{key}: missing conditions mapping")
        parsed = {condition: _validate_condition(row, key, condition) for condition in CONDITIONS}
        sarima_values = [parsed[c]["sarima"] for c in CONDITIONS]
        if any(not _close(value, sarima_values[0]) for value in sarima_values[1:]):
            raise ValueError(f"{key}: B/D/C/P must share the same saved SARIMA prediction")
        rows_by_key[key] = row
        values_by_key[key] = parsed

    if not rows_by_key:
        raise ValueError("manifest_rows must not be empty")
    products = {key[0] for key in rows_by_key}
    if phase in PHASE_COUNTS and products != {"radish"}:
        raise ValueError("production RHS movement input must contain radish rows only")

    all_issued_keys = set(rows_by_key)
    if phase in PHASE_COUNTS and len(all_issued_keys) != PHASE_COUNTS[phase]["issued"]:
        raise ValueError(f"{phase} issued forecast support must contain {PHASE_COUNTS[phase]['issued']} keys")

    paired_keys: set[tuple[str, str, str, int]] = set()
    actual_by_key: dict[tuple[str, str, str, int], float] = {}
    for key in all_issued_keys:
        actual_key = (key[0], key[2])
        if actual_key not in actuals:
            continue
        actual = _number(actuals[actual_key], f"actual {actual_key}", positive=True)
        paired_keys.add(key)
        actual_by_key[key] = actual
    if paired_keys != expected:
        missing = len(expected - paired_keys)
        unexpected = len(paired_keys - expected)
        raise ValueError(f"common positive-actual support differs from frozen keys (missing={missing}, unexpected={unexpected})")

    horizon_counts = Counter(key[3] for key in expected)
    if set(horizon_counts) != set(HORIZONS):
        raise ValueError("expected comparison support must include all ten horizons")
    key_weights = {key: 1.0 / (10.0 * horizon_counts[key[3]]) for key in expected}
    if not math.isclose(sum(key_weights.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("frozen key weights do not sum to one")

    key_rows: list[dict[str, Any]] = []
    records_by_comparison_layer: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    all_issued_rms: dict[tuple[str, str, str], dict[str, Any]] = {}
    for new, baseline in COMPARISONS:
        for layer in LAYERS:
            issue_changes: list[float] = []
            issue_by_h: dict[int, list[float]] = defaultdict(list)
            for key in sorted(all_issued_keys):
                p_base = values_by_key[key][baseline][layer]
                p_new = values_by_key[key][new][layer]
                change = p_new / p_base - 1.0
                issue_changes.append(change)
                issue_by_h[key[3]].append(change)
            issue_h_macro = [
                sum(x * x for x in issue_by_h[h]) / len(issue_by_h[h])
                for h in HORIZONS
                if issue_by_h.get(h)
            ]
            if len(issue_h_macro) != 10:
                raise ValueError("all-issued relative-change RMS must have support at every horizon")
            all_issued_rms[(new, baseline, layer)] = {
                "allIssuedKeyPooledRms": _rms(issue_changes),
                "allIssuedFixed10hMacroRms": math.sqrt(sum(issue_h_macro) / 10.0),
                "n": len(issue_changes),
                "horizonCounts": {str(h): len(issue_by_h[h]) for h in HORIZONS},
            }

    for key in sorted(expected):
        y = actual_by_key[key]
        weight = key_weights[key]
        item = rows_by_key[key]
        for new, baseline in COMPARISONS:
            bvals = values_by_key[key][baseline]
            avals = values_by_key[key][new]
            for layer in LAYERS:
                pb = bvals[layer]
                pa = avals[layer]
                movement = abs(pa - pb)
                improvement = abs(pb - y) - abs(pa - y)
                movement_mape = 100.0 * movement / y
                improvement_mape = 100.0 * improvement / y
                reducing_mae = (movement + improvement) / 2.0
                increasing_mae = (movement - improvement) / 2.0
                reducing_mape = 100.0 * reducing_mae / y
                increasing_mape = 100.0 * increasing_mae / y
                if min(reducing_mae, increasing_mae, reducing_mape, increasing_mape) < -1e-10:
                    raise ValueError(f"{key} {new}-{baseline} {layer}: negative gain/harm component")
                direct_sign = math.copysign(1.0, avals["direct"] - bvals["direct"]) if avals["direct"] != bvals["direct"] else 0.0
                expected_blend_move = 0.5 * abs(avals["direct"] - bvals["direct"])
                if not _close(abs(avals["blend"] - bvals["blend"]), expected_blend_move):
                    raise ValueError(f"{key} {new}-{baseline}: blend movement is not half direct movement")
                if not _close(avals["blend"] - bvals["blend"], 0.5 * (avals["direct"] - bvals["direct"])):
                    raise ValueError(f"{key} {new}-{baseline}: signed blend shift is not half direct shift")
                if layer == "blend" and expected_blend_move > 0:
                    blend_sign = math.copysign(1.0, avals["blend"] - bvals["blend"])
                    if blend_sign != direct_sign:
                        raise ValueError(f"{key} {new}-{baseline}: direct/blend movement direction differs")
                overshoot = (pb - y) * (pa - y) < 0
                record = {
                    "product": key[0],
                    "originDate": key[1],
                    "targetDate": key[2],
                    "targetMonth": key[2][:7],
                    "horizon": key[3],
                    "actual": y,
                    "comparison": f"{new}-{baseline}",
                    "condition": new,
                    "baseline": baseline,
                    "layer": layer,
                    "baselinePrediction": pb,
                    "conditionPrediction": pa,
                    "baselineError": abs(pb - y),
                    "conditionError": abs(pa - y),
                    "baselineMapePct": 100.0 * abs(pb - y) / y,
                    "conditionMapePct": 100.0 * abs(pa - y) / y,
                    "movementMae": movement,
                    "movementMapePct": movement_mape,
                    "improvementMae": improvement,
                    "improvementMapePp": improvement_mape,
                    "lossReducingMae": reducing_mae,
                    "lossIncreasingMae": increasing_mae,
                    "lossReducingMapePp": reducing_mape,
                    "lossIncreasingMapePp": increasing_mape,
                    "relativePriceChange": pa / pb - 1.0,
                    "crossesActual": overshoot,
                    "direction": 0 if pa == pb else (1 if pa > pb else -1),
                    "weight": weight,
                    "weightedMovementMae": weight * movement,
                    "weightedMovementMapePp": weight * movement_mape,
                    "weightedImprovementMae": weight * improvement,
                    "weightedImprovementMapePp": weight * improvement_mape,
                    "weightedLossReducingMae": weight * reducing_mae,
                    "weightedLossIncreasingMae": weight * increasing_mae,
                    "weightedLossReducingMapePp": weight * reducing_mape,
                    "weightedLossIncreasingMapePp": weight * increasing_mape,
                }
                key_rows.append(record)
                records_by_comparison_layer[(new, baseline, layer)].append(record)

    aggregate_rows: list[dict[str, Any]] = []
    for new, baseline in COMPARISONS:
        for layer in LAYERS:
            comparison = f"{new}-{baseline}"
            full_records = records_by_comparison_layer[(new, baseline, layer)]
            full_weights = [key_weights[(r["product"], r["originDate"], r["targetDate"], r["horizon"])] for r in full_records]
            overall = _metric_record(full_records, full_weights)
            relative_rms_h = []
            for h in HORIZONS:
                hr = [r for r in full_records if r["horizon"] == h]
                vals = [float(r["relativePriceChange"]) for r in hr]
                if not vals:
                    raise ValueError(f"{comparison}/{layer}: missing paired horizon {h}")
                relative_rms_h.append(sum(x * x for x in vals) / len(vals))
            overall["relativePriceChangeRmsFixed10hMacro"] = math.sqrt(sum(relative_rms_h) / 10.0)
            issue_rms = all_issued_rms[(new, baseline, layer)]
            overall["allIssuedKeyPooledRelativePriceChangeRms"] = issue_rms["allIssuedKeyPooledRms"]
            overall["allIssuedFixed10hMacroRelativePriceChangeRms"] = issue_rms["allIssuedFixed10hMacroRms"]
            overall["allIssuedN"] = issue_rms["n"]
            date_rows = []
            for target_date in sorted({r["targetDate"] for r in full_records}):
                day_records = [r for r in full_records if r["targetDate"] == target_date]
                day_weights = [key_weights[(r["product"], r["originDate"], r["targetDate"], r["horizon"])]
                               for r in day_records]
                date_rows.append({"fixedGlobalContribution": _metric_record(day_records, day_weights)})
            overall["dateContributionConcentration"] = {
                "improvementMae": _date_concentration(date_rows, "improvementMae"),
                "improvementMapePp": _date_concentration(date_rows, "improvementMapePp"),
            }
            aggregate_rows.append({"phase": phase, "comparison": comparison, "condition": new, "baseline": baseline,
                                   "layer": layer, "scope": "interval", "horizon": None, "targetMonth": None,
                                   "targetDate": None, **overall})

            for h in HORIZONS:
                records = [r for r in full_records if r["horizon"] == h]
                weights = [1.0 / len(records)] * len(records)
                metrics = _metric_record(records, weights)
                aggregate_rows.append({"phase": phase, "comparison": comparison, "condition": new, "baseline": baseline,
                                       "layer": layer, "scope": "horizon", "horizon": h, "targetMonth": None,
                                       "targetDate": None, **metrics})

            months = sorted(set(PHASE_MONTHS.get(phase, ())) | {r["targetMonth"] for r in full_records})
            for month in months:
                records = [r for r in full_records if r["targetMonth"] == month]
                contribution_weights = [key_weights[(r["product"], r["originDate"], r["targetDate"], r["horizon"])]
                                        for r in records]
                contribution = _metric_record(records, contribution_weights)
                month_h_counts = Counter(r["horizon"] for r in records)
                if set(month_h_counts) == set(HORIZONS):
                    within_month_weights = [1.0 / (10.0 * month_h_counts[r["horizon"]]) for r in records]
                    within_month = _metric_record(records, within_month_weights)
                    complete_horizons = True
                else:
                    within_month = _metric_record([], [])
                    complete_horizons = False
                aggregate_rows.append({"phase": phase, "comparison": comparison, "condition": new, "baseline": baseline,
                                       "layer": layer, "scope": "target_month", "horizon": None,
                                       "targetMonth": month, "targetDate": None,
                                       "n": len(records), "horizonCounts": {str(h): month_h_counts.get(h, 0) for h in HORIZONS},
                                       "hasAllTenHorizons": complete_horizons,
                                       "fixedGlobalContribution": contribution,
                                       "recomputedEqualHorizonMonthMean": within_month})

            dates = sorted({r["targetDate"] for r in full_records})
            for target_date in dates:
                records = [r for r in full_records if r["targetDate"] == target_date]
                contribution_weights = [key_weights[(r["product"], r["originDate"], r["targetDate"], r["horizon"])]
                                        for r in records]
                contribution = _metric_record(records, contribution_weights)
                aggregate_rows.append({"phase": phase, "comparison": comparison, "condition": new, "baseline": baseline,
                                       "layer": layer, "scope": "target_date", "horizon": None,
                                       "targetMonth": target_date[:7], "targetDate": target_date,
                                       "n": len(records), "fixedGlobalContribution": contribution})

    # Verify all reported date and month contributions reconstruct the interval macro.
    metric_fields = (
        "baselineMae", "conditionMae", "baselineMapePct", "conditionMapePct",
        "movementMae", "movementMapePct", "improvementMae", "improvementMapePp",
        "lossReducingMae", "lossIncreasingMae", "lossReducingMapePp", "lossIncreasingMapePp",
    )
    for new, baseline in COMPARISONS:
        for layer in LAYERS:
            rows = [r for r in aggregate_rows if r["comparison"] == f"{new}-{baseline}" and r["layer"] == layer]
            interval = next(r for r in rows if r["scope"] == "interval")
            for scope in ("target_month", "target_date"):
                contributions = [r["fixedGlobalContribution"] for r in rows if r["scope"] == scope]
                for field in metric_fields:
                    total = sum(float(x[field]) for x in contributions if x[field] is not None)
                    if not _close(total, float(interval[field])):
                        raise ValueError(f"{scope} contributions do not sum to interval {field} for {new}-{baseline}/{layer}")

    return {
        "schemaVersion": 1,
        "phase": phase,
        "scope": "radish RHS B/D/C/P forecast movement and price-error transfer; descriptive only",
        "support": {
            "allIssuedKeys": len(all_issued_keys),
            "commonPositiveActualKeys": len(expected),
            "commonPositiveActualHorizonCounts": {str(h): horizon_counts[h] for h in HORIZONS},
            "uniqueTargetDates": len({key[2] for key in expected}),
            "uniqueOrigins": len({key[1] for key in expected}),
            "products": sorted(products),
            "sameKeysForAllComparisonsAndLayers": True,
        },
        "definitions": {
            "keyWeight": "1 / (10 * paired-key count at that horizon)",
            "improvement": "baseline absolute error minus condition absolute error",
            "movementMapePct": "100 * absolute price movement / actual; measured in percent",
            "improvementMapePp": "100 * (baseline absolute error - condition absolute error) / actual; percentage points",
            "gainAndHarm": "(movement +/- improvement) / 2; nonnegative and sum/difference reconstruct movement/improvement",
            "efficiency": "improvement / movement, null when movement is zero; bounded by [-1, 1]",
            "relativePriceChangeRmsFixed10hMacro": "sqrt(equal-horizon mean of mean((conditionPrice / baselinePrice - 1)^2)) on paired actual support",
            "allIssuedKeyPooledRelativePriceChangeRms": "pooled RMS of price relative changes over every issued B/condition pair, without actuals",
            "allIssuedFixed10hMacroRelativePriceChangeRms": "equal-horizon RMS over every issued B/condition pair, without actuals",
            "crossesActual": "true only when the forecast move crosses from one side of the actual price to the other",
        },
        "aggregates": aggregate_rows,
        "keyRows": key_rows,
    }
