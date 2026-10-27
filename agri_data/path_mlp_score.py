"""Replay-gated scoring for the frozen daily-price-path MLP experiment.

Screening uses the exact four-way common population. The later evaluation
keeps the fixed five-contrast family, but activates only the crop's frozen
selected-arm versus fixed50 test; all other entries remain explicit p=1.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import full_history_return_eval as price_helper
from . import path_mlp_forecasts as forecasts

PRODUCTS = price_helper.PRODUCTS
HORIZONS = price_helper.HORIZONS
SCREEN_START, SCREEN_END = dt.date(2024, 9, 25), dt.date(2025, 8, 31)
EVAL_START, EVAL_END = dt.date(2026, 1, 1), dt.date(2026, 9, 27)
BANDWIDTHS = (9, 28)
ALPHA = 0.05
ARMS = ("orderedBlend", "summaryBlend", "scrambledBlend")
COMPARISONS = (
    ("ordered_vs_fixed50", "orderedBlend", "fixed50"),
    ("summary_vs_fixed50", "summaryBlend", "fixed50"),
    ("scrambled_vs_fixed50", "scrambledBlend", "fixed50"),
    ("ordered_vs_summary", "orderedBlend", "summaryBlend"),
    ("ordered_vs_scrambled", "orderedBlend", "scrambledBlend"),
)
LOSSES = ("ae", "ape")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _dm(differences: dict[dt.date, float], bandwidth: int) -> dict[str, Any]:
    n = len(differences)
    if any(not math.isfinite(value) for value in differences.values()):
        return {"status": "nonfinite_loss_difference", "reason": "nonfinite_loss_difference", "n": n, "statistic": None, "meanLossDifference": None, "pValue": 1.0}
    mean = math.fsum(differences.values()) / n if n else None
    if n < 30:
        return {"status": "insufficient_pairs", "reason": "fewer_than_30_pairs", "n": n, "statistic": None, "meanLossDifference": mean, "pValue": 1.0}
    values = list(differences.values())
    spread = max(values) - min(values)
    magnitude = max(abs(min(values)), abs(max(values)))
    if spread <= 8 * math.ulp(magnitude):
        return {"status": "zero_spread", "reason": "constant_loss_difference_within_float_tolerance", "n": n, "statistic": None, "meanLossDifference": mean, "pValue": 1.0}
    centered = {day: value - mean for day, value in differences.items()}
    terms = [math.fsum(value * value for value in centered.values())]
    for lag in range(1, bandwidth + 1):
        covariance = math.fsum(centered[day] * centered.get(day + dt.timedelta(days=lag), 0.0) for day in centered)
        terms.append(2 * (1 - lag / (bandwidth + 1)) * covariance)
    total = math.fsum(terms)
    tolerance = 8 * math.ulp(math.fsum(abs(value) for value in terms)) if any(terms) else 0.0
    if not math.isfinite(total) or total <= tolerance:
        return {"status": "invalid_hac_variance", "reason": "nonpositive_or_cancellation_dominated_hac_variance", "n": n, "statistic": None, "meanLossDifference": mean, "pValue": 1.0}
    statistic = mean / math.sqrt(total / (n * n))
    return {"status": "tested", "reason": None, "n": n, "statistic": statistic, "meanLossDifference": mean, "pValue": math.erfc(abs(statistic) / math.sqrt(2))}


def _holm(rows: list[dict[str, Any]]) -> None:
    running = 0.0
    size = len(rows)
    for rank, index in enumerate(sorted(range(size), key=lambda i: rows[i]["pValue"])):
        running = max(running, min(1.0, (size - rank) * rows[index]["pValue"]))
        rows[index]["holmPValue"] = running
        rows[index]["holmSignificant"] = running < ALPHA


def _actuals(raw: dict[str, Any], start: dt.date, end: dt.date) -> dict[tuple[str, dt.date], float]:
    series = price_helper._series(raw)
    return {(product, day): value for product, rows in series.items() for day, value in rows.items() if start <= day <= end}


def _rows(manifest: dict[str, Any], prices_bytes: bytes, components_bytes: bytes, *, manifest_bytes: bytes | None = None,
          screen_manifest_bytes: bytes | None = None, screen_report_bytes: bytes | None = None,
          decision_bytes: bytes | None = None) -> tuple[list[dict[str, Any]], str]:
    """Replay and integrity-check before callers can join target values."""
    raw_bytes = manifest_bytes if manifest_bytes is not None else _canonical(manifest)
    # The generator owns complete deterministic replay and source bindings.
    # Pass canonical bytes rather than trusting a caller-provided validation bit.
    forecasts.validate_manifest_bytes(raw_bytes, prices_bytes, components_bytes,
        manifest.get("phaseName"), decision_bytes=decision_bytes,
        screen_manifest_bytes=screen_manifest_bytes, screen_report_bytes=screen_report_bytes)
    rows = manifest.get("forecasts")
    if not isinstance(rows, list):
        raise ValueError("forecast manifest has no forecast rows")
    return rows, _sha_bytes(raw_bytes)


def _key(row: dict[str, Any]) -> tuple[str, dt.date, int]:
    crop = row.get("crop")
    day = price_helper._date(row.get("target_date"))
    horizon = row.get("horizon")
    if crop not in PRODUCTS or day is None or horizon not in HORIZONS:
        raise ValueError("invalid forecast key")
    return crop, day, horizon


def _prediction(row: dict[str, Any], arm: str) -> float | None:
    record = row.get(arm)
    if not isinstance(record, dict):
        return None
    return _positive(record.get("prediction")) if record.get("status") == "issued" else None


def _metrics(pairs: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "mae": None, "mape": None}
    errors = [abs(pair[arm] - pair["actual"]) for pair in pairs]
    return {"n": len(pairs), "mae": math.fsum(errors) / len(errors), "mape": 100 * math.fsum(error / pair["actual"] for error, pair in zip(errors, pairs)) / len(errors)}


def _metrics_and_tests(pairs: list[dict[str, Any]], comparisons=COMPARISONS) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics, dm = [], []
    for comparison, a, b in comparisons:
        metrics.append({"comparison": comparison, "a": _metrics(pairs, a), "b": _metrics(pairs, b)})
        for loss in LOSSES:
            differences = {}
            for pair in pairs:
                day = dt.date.fromisoformat(pair["target_date"])
                ae_a, ae_b = abs(pair[a] - pair["actual"]), abs(pair[b] - pair["actual"])
                differences[day] = ae_a - ae_b if loss == "ae" else (ae_a - ae_b) / pair["actual"]
            for bandwidth in BANDWIDTHS:
                dm.append({"comparison": comparison, "loss": loss, "bandwidth": bandwidth, **_dm(differences, bandwidth)})
    return metrics, dm


def _period_groups(start: dt.date, end: dt.date):
    days = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    groups = {"month": {}, "quarter": {}}
    for day in days:
        groups["month"].setdefault(f"{day.year}-{day.month:02d}", set()).add(day)
        groups["quarter"].setdefault(f"{day.year}-Q{(day.month - 1) // 3 + 1}", set()).add(day)
    return days, groups


def _coverage(rows, actuals, start, end, *, selected_arms=None):
    days, groups = _period_groups(start, end)
    coverage, temporal = {}, {}
    bykey = {_key(row): row for row in rows}
    for crop in PRODUCTS:
        for horizon in HORIZONS:
            cells = [bykey[(crop, day, horizon)] for day in days]
            eligible = [row for row in cells if row.get("eligible") is True]
            issued = {arm: sum(_prediction(row, arm) is not None for row in eligible) for arm in (*ARMS, "fixed50")}
            actual_pairs = sum((crop, day) in actuals for day in days if any(_key(row) == (crop, day, horizon) and row.get("eligible") is True for row in cells))
            required = (*ARMS, "fixed50") if selected_arms is None else ((selected_arms[crop], "fixed50") if selected_arms[crop] in ARMS else ("fixed50",))
            common = 0
            for row in eligible:
                day = price_helper._date(row["target_date"])
                if (crop, day) in actuals and all(_prediction(row, arm) is not None for arm in required):
                    common += 1
            coverage[f"{crop}/{horizon}"] = {"planned": len(cells), "preFitExcluded": sum(row.get("eligible") is not True for row in cells), "eligible": len(eligible), "positiveActualPairs": actual_pairs, "commonPairs": common,
                "fixed50SourceKeyPresent": sum(row.get("fixed50KeyPresent") is True for row in eligible),
                "positiveActualUniqueTargets": len({price_helper._date(row["target_date"]) for row in eligible if (crop, price_helper._date(row["target_date"])) in actuals}),
                "commonUniqueTargets": len({price_helper._date(row["target_date"]) for row in eligible if (crop, price_helper._date(row["target_date"])) in actuals and all(_prediction(row, arm) is not None for arm in required)}),
                **{f"{arm}Issued": count for arm, count in issued.items()},
                **{f"{arm}CoverageLossTargets": [row["target_date"] for row in eligible if _prediction(row, "fixed50") is not None and _prediction(row, arm) is None] for arm in ARMS}}
        for grain, labels in groups.items():
            for label, period_days in labels.items():
                cells = [row for row in rows if _key(row)[0] == crop and _key(row)[1] in period_days]
                eligible = [row for row in cells if row.get("eligible") is True]
                info = {"planned": len(cells), "preFitExcluded": sum(row.get("eligible") is not True for row in cells), "eligible": len(eligible),
                    "positiveActualTargets": sum((crop, day) in actuals for day in period_days),
                    "eligiblePositiveActualPairs": sum((crop, price_helper._date(row["target_date"])) in actuals for row in eligible),
                    "eligiblePositiveActualUniqueTargets": len({price_helper._date(row["target_date"]) for row in eligible if (crop, price_helper._date(row["target_date"])) in actuals}),
                    "commonPairs": sum((crop, price_helper._date(row["target_date"])) in actuals and all(_prediction(row, arm) is not None for arm in ((*ARMS, "fixed50") if selected_arms is None else ((selected_arms[crop], "fixed50") if selected_arms[crop] in ARMS else ("fixed50",)))) for row in eligible),
                    "commonUniqueTargets": len({price_helper._date(row["target_date"]) for row in eligible if (crop, price_helper._date(row["target_date"])) in actuals and all(_prediction(row, arm) is not None for arm in ((*ARMS, "fixed50") if selected_arms is None else ((selected_arms[crop], "fixed50") if selected_arms[crop] in ARMS else ("fixed50",))))})}
                for arm in (*ARMS, "fixed50"):
                    info[f"{arm}Issued"] = sum(_prediction(row, arm) is not None for row in eligible)
                temporal[f"{crop}/{grain}/{label}"] = info
    return coverage, temporal


def _score_common(rows, actuals, start, end, *, all_four=True):
    days, _ = _period_groups(start, end)
    pairs = []
    for row in rows:
        if row.get("eligible") is not True:
            continue
        crop, day, horizon = _key(row)
        if not start <= day <= end:
            continue
        actual = actuals.get((crop, day))
        preds = {arm: _prediction(row, arm) for arm in (*ARMS, "fixed50")}
        required = (*ARMS, "fixed50") if all_four else ()
        if actual is not None and all(preds[arm] is not None for arm in required):
            pairs.append({"crop": crop, "target_date": day.isoformat(), "horizon": horizon, "actual": actual, **preds})
    return pairs


def _equal_horizon(pairs, comparisons=COMPARISONS):
    result = []
    for crop in PRODUCTS:
        for name, a, b in comparisons:
            cells = []
            for horizon in HORIZONS:
                subset = [pair for pair in pairs if pair["crop"] == crop and pair["horizon"] == horizon]
                if subset:
                    cells.append({"horizon": horizon, "a": _metrics(subset, a), "b": _metrics(subset, b)})
            result.append({"crop": crop, "comparison": name, "horizonsIncluded": len(cells),
                "aEqualHorizonMae": math.fsum(cell["a"]["mae"] for cell in cells) / len(cells) if cells else None,
                "bEqualHorizonMae": math.fsum(cell["b"]["mae"] for cell in cells) / len(cells) if cells else None,
                "aEqualHorizonMape": math.fsum(cell["a"]["mape"] for cell in cells) / len(cells) if cells else None,
                "bEqualHorizonMape": math.fsum(cell["b"]["mape"] for cell in cells) / len(cells) if cells else None,
                "horizons": cells})
    return result


def score_screen(manifest: dict[str, Any], prices: dict[str, Any], components: dict[str, Any], *,
                 manifest_bytes: bytes | None = None, prices_bytes: bytes | None = None,
                 components_bytes: bytes | None = None) -> dict[str, Any]:
    manifest_bytes = manifest_bytes if manifest_bytes is not None else _canonical(manifest)
    prices_bytes = prices_bytes if prices_bytes is not None else _canonical(prices)
    components_bytes = components_bytes if components_bytes is not None else _canonical(components)
    # Replay and source bindings are verified before prices can enter `_actuals`.
    rows, digest = _rows(manifest, prices_bytes, components_bytes, manifest_bytes=manifest_bytes)
    if manifest.get("phaseName") != "screen":
        raise ValueError("screen scoring requires screen forecast manifest")
    return _score_screen_rows(manifest, prices, rows, digest)


def _score_screen_rows(manifest: dict[str, Any], prices: dict[str, Any], rows: list[dict[str, Any]], digest: str) -> dict[str, Any]:
    actuals = _actuals(prices, SCREEN_START, SCREEN_END)
    coverage, temporal = _coverage(rows, actuals, SCREEN_START, SCREEN_END)
    pairs = _score_common(rows, actuals, SCREEN_START, SCREEN_END, all_four=True)
    metrics, dm = [], []
    for crop in PRODUCTS:
        for horizon in HORIZONS:
            subset = [pair for pair in pairs if pair["crop"] == crop and pair["horizon"] == horizon]
            cell_metrics, cell_dm = _metrics_and_tests(subset)
            metrics.extend({"crop": crop, "horizon": horizon, **metric} for metric in cell_metrics)
            dm.extend({"crop": crop, "horizon": horizon, **test} for test in cell_dm)
    for bandwidth in BANDWIDTHS:
        family = [item for item in dm if item["bandwidth"] == bandwidth]
        if len(family) != 500:
            raise ValueError(f"screen Holm family must contain 500 tests per bandwidth, got {len(family)}")
        _holm(family)
    macros = _equal_horizon(pairs)
    gates = _screen_gates(rows, coverage, macros, dm)
    selection = _select(gates)
    return {"schemaVersion": 1, "phase": "screen_scored", "manifestSha256": digest, "metrics": metrics,
        "equalHorizonMetrics": macros, "coverage": coverage, "temporalCoverage": temporal,
        "candidateGates": gates, "selection": selection, "dm": dm,
        "dmSummaryByBandwidth": _dm_summary(dm), "commonPairs": pairs}


def _screen_gates(rows, coverage, macros, dm):
    out = []
    macro = {(item["crop"], item["comparison"]): item for item in macros}
    for crop in PRODUCTS:
        crop_rows = [row for row in rows if _key(row)[0] == crop and row.get("eligible") is True]
        fixed_keys = {_key(row) for row in crop_rows if _prediction(row, "fixed50") is not None}
        lost = {arm: any(_key(row) in fixed_keys and _prediction(row, arm) is None for row in crop_rows) for arm in ARMS}
        any_arm_lost_fixed_key = any(lost.values())
        for arm, comp in zip(ARMS, ("ordered_vs_fixed50", "summary_vs_fixed50", "scrambled_vs_fixed50")):
            item = macro[(crop, comp)]
            horizon_items = item["horizons"]
            no_harm = len(horizon_items) == len(HORIZONS) and all(cell["a"]["mape"] - cell["b"]["mape"] <= 1.0 for cell in horizon_items)
            joint = False
            for horizon in HORIZONS:
                tests = {(test["loss"], test["bandwidth"]): test for test in dm if test["crop"] == crop and test["horizon"] == horizon and test["comparison"] == comp}
                if all(tests.get((loss, bandwidth), {}).get("holmSignificant") and tests[(loss, bandwidth)]["meanLossDifference"] < 0 for loss in LOSSES for bandwidth in BANDWIDTHS):
                    joint = True
            all30 = len(horizon_items) == len(HORIZONS) and all(cell["a"]["n"] >= 30 for cell in horizon_items)
            gain = item["bEqualHorizonMape"] is not None and item["aEqualHorizonMape"] <= item["bEqualHorizonMape"] - 0.5
            mae = item["aEqualHorizonMae"] is not None and item["aEqualHorizonMae"] < item["bEqualHorizonMae"]
            passed = all((all30, gain, mae, not any_arm_lost_fixed_key, no_harm, joint))
            out.append({"crop": crop, "candidate": arm, "benchmark": "fixed50", "all10HorizonsAtLeast30CommonPairs": all30,
                "candidateEqualHorizonMape": item["aEqualHorizonMape"], "fixed50EqualHorizonMape": item["bEqualHorizonMape"],
                "mapeImprovementPctPoints": item["bEqualHorizonMape"] - item["aEqualHorizonMape"] if item["aEqualHorizonMape"] is not None and item["bEqualHorizonMape"] is not None else None,
                "candidateEqualHorizonMae": item["aEqualHorizonMae"], "fixed50EqualHorizonMae": item["bEqualHorizonMae"],
                "coverageNondeclining": not any_arm_lost_fixed_key, "anyArmLostFixed50IssuedKey": any_arm_lost_fixed_key,
                "noHorizonMapeWorseByMoreThan1PctPoint": no_harm,
                "jointAeApeHolmImprovementAtSomeHorizonBothHac": joint,
                "decision": "eligible_for_selection" if passed else "retain_fixed50"})
    return out


def _select(gates):
    selections = {}
    for crop in PRODUCTS:
        passed = [row for row in gates if row["crop"] == crop and row["decision"] == "eligible_for_selection"]
        passed.sort(key=lambda row: (row["candidateEqualHorizonMape"], row["candidateEqualHorizonMae"], ARMS.index(row["candidate"])))
        selections[crop] = passed[0]["candidate"] if passed else "retain_fixed50"
    return selections


def _dm_summary(dm):
    result = []
    for bandwidth in BANDWIDTHS:
        family = [row for row in dm if row["bandwidth"] == bandwidth]
        result.append({"bandwidth": bandwidth, "familySize": len(family),
            "significantImprovement": sum(row["status"] == "tested" and row.get("holmSignificant") and row.get("meanLossDifference", 0) < 0 for row in family),
            "significantWorsening": sum(row["status"] == "tested" and row.get("holmSignificant") and row.get("meanLossDifference", 0) > 0 for row in family),
            "inconclusive": sum(row["status"] == "tested" and not row.get("holmSignificant") for row in family),
            "inferenceUnavailable": sum(row["status"] != "tested" for row in family)})
    return result


def make_decision(screen_manifest: dict[str, Any], screen_report: dict[str, Any], *,
                  screen_manifest_bytes: bytes, screen_report_bytes: bytes,
                  prices_bytes: bytes, components_bytes: bytes) -> dict[str, Any]:
    # The screen manifest is replay-validated; artifact hashes bind its decision.
    rows, manifest_digest = _rows(screen_manifest, prices_bytes, components_bytes, manifest_bytes=screen_manifest_bytes)
    if screen_report.get("phase") != "screen_scored" or screen_report.get("manifestSha256") != manifest_digest:
        raise ValueError("screen report does not bind to validated screen manifest")
    expected_selection = _select(screen_report.get("candidateGates", []))
    if screen_report.get("selection") != expected_selection:
        raise ValueError("screen report selection does not match its candidate gates")
    return _decision_for_validated_screen(screen_manifest, screen_report,
        screen_manifest_bytes=screen_manifest_bytes, screen_report_bytes=screen_report_bytes)


def _decision_for_validated_screen(screen_manifest: dict[str, Any], screen_report: dict[str, Any], *,
                                   screen_manifest_bytes: bytes, screen_report_bytes: bytes) -> dict[str, Any]:
    unsigned = {"schemaVersion": 1, "phase": "selection_decision",
        "screenManifestSha256": _sha_bytes(screen_manifest_bytes),
        "screenReportSha256": _sha_bytes(screen_report_bytes),
        "scorerSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selections": screen_report["selection"]}
    unsigned["decisionSha256"] = _sha_bytes(_canonical(unsigned))
    return unsigned


def validate_screen_artifacts(screen_manifest_bytes: bytes, prices_bytes: bytes,
                              screen_components_bytes: bytes, screen_report_bytes: bytes,
                              decision_bytes: bytes) -> dict[str, Any]:
    """Replay and rescore the locked screen before any evaluation fit is allowed."""
    screen_manifest = json.loads(screen_manifest_bytes)
    prices = json.loads(prices_bytes)
    screen_components = json.loads(screen_components_bytes)
    screen_report = json.loads(screen_report_bytes)
    decision = json.loads(decision_bytes)
    rows, digest = _rows(screen_manifest, prices_bytes, screen_components_bytes,
                         manifest_bytes=screen_manifest_bytes)
    if screen_manifest.get("phaseName") != "screen":
        raise ValueError("screen artifact is not the frozen screening phase")
    recomputed = _score_screen_rows(screen_manifest, prices, rows, digest)
    if _canonical(recomputed) != screen_report_bytes:
        raise ValueError("screen report differs from replayed forecast and pinned outcomes")
    expected_decision = _decision_for_validated_screen(screen_manifest, screen_report,
        screen_manifest_bytes=screen_manifest_bytes, screen_report_bytes=screen_report_bytes)
    if decision != expected_decision or decision.get("selections") != _select(recomputed.get("candidateGates", [])):
        raise ValueError("screen decision does not match replayed screen selection")
    return {"selection": decision["selections"], "manifestSha256": digest,
            "screenReportSha256": _sha_bytes(screen_report_bytes),
            "decisionSha256": decision["decisionSha256"]}


def score_evaluation(manifest: dict[str, Any], prices: dict[str, Any], screen_manifest: dict[str, Any],
                     screen_report: dict[str, Any], decision: dict[str, Any], components: dict[str, Any], *,
                     manifest_bytes: bytes | None = None, prices_bytes: bytes | None = None,
                     components_bytes: bytes | None = None, screen_manifest_bytes: bytes | None = None,
                     screen_report_bytes: bytes | None = None, decision_bytes: bytes | None = None,
                     screen_components_bytes: bytes | None = None) -> dict[str, Any]:
    manifest_bytes = manifest_bytes if manifest_bytes is not None else _canonical(manifest)
    prices_bytes = prices_bytes if prices_bytes is not None else _canonical(prices)
    components_bytes = components_bytes if components_bytes is not None else _canonical(components)
    screen_manifest_bytes = screen_manifest_bytes if screen_manifest_bytes is not None else _canonical(screen_manifest)
    screen_report_bytes = screen_report_bytes if screen_report_bytes is not None else _canonical(screen_report)
    decision_bytes = decision_bytes if decision_bytes is not None else _canonical(decision)
    if screen_components_bytes is None:
        raise ValueError("pinned screen component bytes are required to verify the selection protocol")
    replayed_screen = validate_screen_artifacts(screen_manifest_bytes, prices_bytes, screen_components_bytes,
                                                 screen_report_bytes, decision_bytes)
    rows, digest = _rows(manifest, prices_bytes, components_bytes, manifest_bytes=manifest_bytes, screen_manifest_bytes=screen_manifest_bytes,
                         screen_report_bytes=screen_report_bytes, decision_bytes=decision_bytes)
    if manifest.get("phaseName") != "evaluation":
        raise ValueError("evaluation scoring requires evaluation forecast manifest")
    if decision.get("selections") != replayed_screen["selection"]:
        raise ValueError("selection decision is not bound to the supplied validated screen artifacts")
    actuals = _actuals(prices, EVAL_START, EVAL_END)
    selections = decision["selections"]
    coverage, temporal = _coverage(rows, actuals, EVAL_START, EVAL_END, selected_arms=selections)
    candidate_pairs = []
    metrics, dm = [], []
    # Keep all five comparisons in the 500-test family. Only the selected
    # crop's candidate-vs-fixed50 contrast is active; others are explicit p=1.
    for crop in PRODUCTS:
        candidate = selections[crop]
        for horizon in HORIZONS:
            cell_rows = [row for row in rows if _key(row)[0] == crop and _key(row)[2] == horizon and row.get("eligible") is True]
            pairs = []
            for row in cell_rows:
                _, day, _ = _key(row)
                actual = actuals.get((crop, day))
                if actual is None:
                    continue
                values = {arm: _prediction(row, arm) for arm in ARMS}
                values["fixed50"] = _prediction(row, "fixed50")
                if candidate in ARMS and values[candidate] is not None and values["fixed50"] is not None:
                    pair = {"crop": crop, "target_date": day.isoformat(), "horizon": horizon, "actual": actual, **values}
                    pairs.append(pair)
                    candidate_pairs.append(pair)
            active = candidate if candidate in ARMS else None
            for name, a, b in COMPARISONS:
                # Comparison IDs use ordered/summary/scrambled names.
                key = {"orderedBlend": "ordered_vs_fixed50", "summaryBlend": "summary_vs_fixed50", "scrambledBlend": "scrambled_vs_fixed50"}.get(active)
                enabled = bool(key == name)
                if enabled:
                    cell_metrics, cell_dm = _metrics_and_tests(pairs, ((name, a, b),))
                    metric = cell_metrics[0]
                    metrics.append({"crop": crop, "horizon": horizon, **metric})
                    dm.extend({"crop": crop, "horizon": horizon, **test} for test in cell_dm)
                else:
                    metrics.append({"crop": crop, "horizon": horizon, "comparison": name,
                        "a": {"n": 0, "mae": None, "mape": None}, "b": {"n": 0, "mae": None, "mape": None},
                        "status": "not_selected", "reason": "frozen_screen_decision_did_not_select_comparison"})
                    for loss in LOSSES:
                        for bandwidth in BANDWIDTHS:
                            dm.append({"crop": crop, "horizon": horizon, "comparison": name, "loss": loss,
                                "bandwidth": bandwidth, "status": "not_selected", "reason": "frozen_screen_decision_did_not_select_comparison",
                                "n": 0, "statistic": None, "meanLossDifference": None, "pValue": 1.0,
                                "holmPValue": 1.0, "holmSignificant": False})
    for bandwidth in BANDWIDTHS:
        family = [row for row in dm if row["bandwidth"] == bandwidth]
        if len(family) != 500:
            raise ValueError(f"evaluation Holm family must contain 500 tests per bandwidth, got {len(family)}")
        _holm(family)
    gates = _evaluation_gates(rows, actuals, dm, metrics, selections)
    gate_decisions = {gate["crop"]: gate["decision"] for gate in gates}
    # The evaluated whole policy is fixed by the earlier screen selection.
    # The 2026 gate is reported separately and cannot retroactively erase poor
    # candidate predictions from the selected policy's paired outcomes.
    raw_pairs = _whole_policy_rows(rows, actuals, selections, gate_decisions)
    # Full-policy 50-cell descriptive metrics on original fixed50 keys.
    policy_metrics = []
    for crop in PRODUCTS:
        for horizon in HORIZONS:
            subset = [row for row in raw_pairs if row["crop"] == crop and row["horizon"] == horizon]
            policy_issued = [row for row in subset if row["actual"] is not None and row["policy"] is not None]
            fixed_issued = [row for row in subset if row["actual"] is not None and row["fixed50"] is not None]
            policy_metrics.append({"crop": crop, "horizon": horizon,
                "originalFixed50SourceKeys": len(subset),
                "originalFixed50KeysWithActual": sum(row["actual"] is not None for row in subset),
                "eligibleOriginalFixed50SourceKeys": sum(row["eligible"] for row in subset),
                "preperiodOriginOriginalFixed50SourceKeys": sum(not row["eligible"] for row in subset),
                "policy": _metrics([{**row, "policyPred": row["policy"]} for row in policy_issued], "policyPred"),
                "fixed50": _metrics(fixed_issued, "fixed50"), "fallbackCount": sum(row["fallback"] for row in subset),
                "preperiodOriginFallbackCount": sum(row["preperiodOriginFallback"] for row in subset),
                "candidateMissingFallbackCount": sum(row["fallback"] and not row["preperiodOriginFallback"] for row in subset),
                "policyUnavailableOriginalFixed50Keys": sum(row["policy"] is None for row in subset),
                "fixed50UnavailableOriginalKeys": sum(row["fixed50"] is None for row in subset)})
    policy_coverage = {
        "originalFixed50SourceKeys": len(raw_pairs),
        "keysWithPositiveActual": sum(row["actual"] is not None for row in raw_pairs),
        "eligibleOriginKeys": sum(row["eligible"] for row in raw_pairs),
        "preperiodOriginKeys": sum(not row["eligible"] for row in raw_pairs),
        "selectedCandidateIssued": sum(row["candidatePrediction"] is not None for row in raw_pairs),
        "policyIssued": sum(row["policy"] is not None for row in raw_pairs),
        "fixed50Issued": sum(row["fixed50"] is not None for row in raw_pairs),
        "fallbackCount": sum(row["fallback"] for row in raw_pairs),
        "preperiodOriginFallbackCount": sum(row["preperiodOriginFallback"] for row in raw_pairs),
    }
    supported_cells = [cell for cell in policy_metrics if cell["policy"]["n"] and cell["fixed50"]["n"]]
    complete_policy_cells = len(supported_cells) == len(PRODUCTS) * len(HORIZONS)
    policy_portfolio = {
        "cellsPossible": len(PRODUCTS) * len(HORIZONS),
        "cellsIncluded": len(supported_cells),
        "complete50CellCoverage": complete_policy_cells,
        "equalCellPolicyMae": math.fsum(cell["policy"]["mae"] for cell in supported_cells) / len(supported_cells) if supported_cells else None,
        "equalCellFixed50Mae": math.fsum(cell["fixed50"]["mae"] for cell in supported_cells) / len(supported_cells) if supported_cells else None,
        "equalCellPolicyMape": math.fsum(cell["policy"]["mape"] for cell in supported_cells) / len(supported_cells) if supported_cells else None,
        "equalCellFixed50Mape": math.fsum(cell["fixed50"]["mape"] for cell in supported_cells) / len(supported_cells) if supported_cells else None,
        "all50CellPolicyMae": math.fsum(cell["policy"]["mae"] for cell in supported_cells) / len(supported_cells) if complete_policy_cells else None,
        "all50CellFixed50Mae": math.fsum(cell["fixed50"]["mae"] for cell in supported_cells) / len(supported_cells) if complete_policy_cells else None,
        "all50CellPolicyMape": math.fsum(cell["policy"]["mape"] for cell in supported_cells) / len(supported_cells) if complete_policy_cells else None,
        "all50CellFixed50Mape": math.fsum(cell["fixed50"]["mape"] for cell in supported_cells) / len(supported_cells) if complete_policy_cells else None,
        "supportedCellMeansAreDescriptiveOnly": not complete_policy_cells,
    }
    equal_horizon_metrics = []
    for crop in PRODUCTS:
        candidate = selections[crop]
        if candidate not in ARMS:
            equal_horizon_metrics.append({"crop": crop, "comparison": "retain_fixed50", "horizonsIncluded": 0,
                "candidateEqualHorizonMae": None, "fixed50EqualHorizonMae": None,
                "candidateEqualHorizonMape": None, "fixed50EqualHorizonMape": None})
            continue
        cells = []
        for horizon in HORIZONS:
            subset = [pair for pair in candidate_pairs if pair["crop"] == crop and pair["horizon"] == horizon]
            if subset:
                cells.append({"candidate": _metrics(subset, candidate), "fixed50": _metrics(subset, "fixed50")})
        equal_horizon_metrics.append({"crop": crop, "comparison": f"{candidate}_vs_fixed50", "horizonsIncluded": len(cells),
            "candidateEqualHorizonMae": math.fsum(cell["candidate"]["mae"] for cell in cells) / len(cells) if cells else None,
            "fixed50EqualHorizonMae": math.fsum(cell["fixed50"]["mae"] for cell in cells) / len(cells) if cells else None,
            "candidateEqualHorizonMape": math.fsum(cell["candidate"]["mape"] for cell in cells) / len(cells) if cells else None,
            "fixed50EqualHorizonMape": math.fsum(cell["fixed50"]["mape"] for cell in cells) / len(cells) if cells else None})
    return {"schemaVersion": 1, "phase": "evaluation_scored", "manifestSha256": digest,
        "screenManifestSha256": decision["screenManifestSha256"], "screenReportSha256": decision["screenReportSha256"],
        "decisionSha256": decision["decisionSha256"], "selections": selections, "metrics": metrics,
        "evaluationGates": gates, "equalHorizonMetrics": equal_horizon_metrics, "coverage": coverage, "temporalCoverage": temporal,
        "dm": dm, "dmSummaryByBandwidth": _dm_summary(dm), "policyCoverage": policy_coverage,
        "policyMetrics": policy_metrics, "policyPortfolioEqualWeightAcross50Cells": policy_portfolio,
        "candidatePairs": candidate_pairs, "policyPairs": raw_pairs}


def _evaluation_gates(rows, actuals, dm, metrics, selections):
    gates = []
    for crop in PRODUCTS:
        arm = selections[crop]
        if arm == "retain_fixed50":
            gates.append({"crop": crop, "candidate": arm, "decision": "retain_fixed50", "reason": "no_screen_candidate_selected"})
            continue
        comp = {"orderedBlend": "ordered_vs_fixed50", "summaryBlend": "summary_vs_fixed50", "scrambledBlend": "scrambled_vs_fixed50"}[arm]
        cells = [row for row in metrics if row["crop"] == crop and row["comparison"] == comp and row.get("a", {}).get("n", 0)]
        byh = {row["horizon"]: row for row in cells}
        coverage_loss = any(row.get("eligible") is True and _prediction(row, "fixed50") is not None and _prediction(row, arm) is None for row in rows if _key(row)[0] == crop)
        no_harm = len(byh) == len(HORIZONS) and all(row["a"]["mape"] - row["b"]["mape"] <= 1 for row in cells)
        macro_a = math.fsum(row["a"]["mape"] for row in cells) / len(cells) if cells else None
        macro_b = math.fsum(row["b"]["mape"] for row in cells) / len(cells) if cells else None
        mae_a = math.fsum(row["a"]["mae"] for row in cells) / len(cells) if cells else None
        mae_b = math.fsum(row["b"]["mae"] for row in cells) / len(cells) if cells else None
        joint = False
        for horizon in HORIZONS:
            tests = {(row["loss"], row["bandwidth"]): row for row in dm if row["crop"] == crop and row["horizon"] == horizon and row["comparison"] == comp}
            if all(tests.get((loss, bandwidth), {}).get("holmSignificant") and tests[(loss, bandwidth)]["meanLossDifference"] < 0 for loss in LOSSES for bandwidth in BANDWIDTHS):
                joint = True
        all30 = len(byh) == len(HORIZONS) and all(row["a"]["n"] >= 30 for row in cells)
        passed = all30 and macro_a is not None and macro_b is not None and macro_a <= macro_b - .5 and mae_a < mae_b and not coverage_loss and no_harm and joint
        gates.append({"crop": crop, "candidate": arm, "benchmark": "fixed50", "all10HorizonsAtLeast30CommonPairs": all30,
            "candidateEqualHorizonMape": macro_a, "fixed50EqualHorizonMape": macro_b,
            "candidateEqualHorizonMae": mae_a, "fixed50EqualHorizonMae": mae_b, "coverageNondeclining": not coverage_loss,
            "noHorizonMapeWorseByMoreThan1PctPoint": no_harm, "jointAeApeHolmImprovementAtSomeHorizonBothHac": joint,
            "decision": "retain_candidate_for_policy" if passed else "retain_fixed50"})
    return gates


def _whole_policy_rows(rows, actuals, selections, gate_decisions):
    """Apply the screen-frozen crop choice to every source fixed50 key.

    The 2026 gate is annotation only here; only an unavailable chosen forecast
    falls back to that key's inherited fixed50 prediction.
    """
    result = []
    for row in rows:
        if row.get("fixed50KeyPresent") is not True:
            continue
        crop, day, horizon = _key(row)
        if not EVAL_START <= day <= EVAL_END:
            continue
        candidate = selections[crop]
        candidate_prediction = _prediction(row, candidate) if candidate in ARMS else None
        fixed_prediction = _prediction(row, "fixed50")
        fallback = candidate in ARMS and candidate_prediction is None
        result.append({"crop": crop, "target_date": day.isoformat(), "horizon": horizon,
            "origin_date": row["origin_date"], "eligible": row.get("eligible") is True,
            "actual": actuals.get((crop, day)), "selectedArm": candidate,
            "candidatePrediction": candidate_prediction,
            "policy": fixed_prediction if candidate == "retain_fixed50" or fallback else candidate_prediction,
            "fixed50": fixed_prediction, "fallback": fallback,
            "fallbackReason": ("preperiod_origin_candidate_not_issued" if not row.get("eligible") else "candidate_not_issued") if fallback else None,
            "preperiodOriginFallback": bool(fallback and not row.get("eligible")),
            "evaluationGateDecision": gate_decisions[crop]})
    return result


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(_canonical(value))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    screen = sub.add_parser("score-screen")
    screen.add_argument("forecast", type=Path); screen.add_argument("report", type=Path); screen.add_argument("decision", type=Path)
    screen.add_argument("--prices", type=Path, default=forecasts.PRICES_PATH)
    screen.add_argument("--components", type=Path, default=forecasts.SCREEN_COMPONENTS_PATH)
    evaluation = sub.add_parser("score-evaluation")
    evaluation.add_argument("forecast", type=Path); evaluation.add_argument("screen_manifest", type=Path)
    evaluation.add_argument("screen_report", type=Path); evaluation.add_argument("decision", type=Path); evaluation.add_argument("report", type=Path)
    evaluation.add_argument("--prices", type=Path, default=forecasts.PRICES_PATH)
    evaluation.add_argument("--components", type=Path, default=forecasts.EVALUATION_COMPONENTS_PATH)
    evaluation.add_argument("--screen-components", type=Path, default=forecasts.SCREEN_COMPONENTS_PATH)
    args = parser.parse_args(argv)
    if args.command == "score-screen":
        forecast_bytes=args.forecast.read_bytes();price_bytes=args.prices.read_bytes();component_bytes=args.components.read_bytes()
        manifest = json.loads(forecast_bytes); prices = json.loads(price_bytes); components=json.loads(component_bytes)
        report = score_screen(manifest, prices, components, manifest_bytes=forecast_bytes, prices_bytes=price_bytes, components_bytes=component_bytes)
        report_bytes=_canonical(report)
        decision = _decision_for_validated_screen(manifest, report,
            screen_manifest_bytes=forecast_bytes, screen_report_bytes=report_bytes)
        write_exclusive(args.report, report); write_exclusive(args.decision, decision)
    else:
        forecast_bytes=args.forecast.read_bytes();price_bytes=args.prices.read_bytes();component_bytes=args.components.read_bytes()
        screen_manifest_bytes=args.screen_manifest.read_bytes();screen_report_bytes=args.screen_report.read_bytes();decision_bytes=args.decision.read_bytes()
        screen_component_bytes=args.screen_components.read_bytes()
        manifest=json.loads(forecast_bytes);prices=json.loads(price_bytes);components=json.loads(component_bytes)
        screen_manifest=json.loads(screen_manifest_bytes);screen_report=json.loads(screen_report_bytes);decision=json.loads(decision_bytes)
        report=score_evaluation(manifest,prices,screen_manifest,screen_report,decision,components,manifest_bytes=forecast_bytes,
            prices_bytes=price_bytes,components_bytes=component_bytes,screen_manifest_bytes=screen_manifest_bytes,
            screen_report_bytes=screen_report_bytes,decision_bytes=decision_bytes,screen_components_bytes=screen_component_bytes)
        write_exclusive(args.report, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
