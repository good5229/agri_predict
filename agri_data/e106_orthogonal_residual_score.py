"""Pure, frozen scoring for the E106 orthogonal-residual experiment.

The caller verifies source files, manifests, truth inputs, and numerical
replays before calling :func:`score_manifest`.  This module performs no file
access, fitting, input mutation, or report writes.
"""
from __future__ import annotations

import datetime as dt
import math
from collections import Counter, defaultdict
from typing import Any

from . import equal_info_online_score as stats
from .research_progress_diagnostic import circular_block_bootstrap

PRODUCTS = ("cabbage", "radish", "onion", "garlic")
HORIZONS = tuple(range(1, 11))
PHASES = {
    "validation": (dt.date(2026, 4, 1), dt.date(2026, 6, 20), dt.date(2026, 4, 1)),
    "final": (dt.date(2026, 7, 11), dt.date(2026, 9, 27), dt.date(2026, 7, 1)),
}
BANDWIDTHS = (9, 28)
BLOCKS = (14, 28)
REPLICATES = 2000
SEED = 20261011


def _date(value: Any, label: str) -> dt.date:
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return value
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _positive(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _actual_record(actuals: dict[str, dict[dt.date, Any]], product: str,
                   target: dt.date) -> tuple[str, float | None]:
    rows = actuals.get(product, {})
    record = rows.get(target)
    if record is None:
        record = rows.get(target.isoformat())
    if record is None:
        return "no_source_observation", None
    if isinstance(record, dict):
        value = record.get("actual")
    elif isinstance(record, (tuple, list)) and record:
        value = record[0]
    else:
        value = record
    actual = _positive(value)
    return ("positive_actual", actual) if actual is not None else ("nonpositive_or_invalid_actual", None)


def _mean(values: list[float]) -> float | None:
    return math.fsum(values) / len(values) if values else None


def _phase_months(start: dt.date, end: dt.date) -> list[str]:
    month = dt.date(start.year, start.month, 1)
    last = dt.date(end.year, end.month, 1)
    result = []
    while month <= last:
        result.append(month.strftime("%Y-%m"))
        month = dt.date(month.year + (month.month == 12), 1 if month.month == 12 else month.month + 1, 1)
    return result


def _price_summary(rows: list[dict[str, Any]], a: str, b: str) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "uniqueOriginDates": 0, "uniqueTargetDates": 0, "aMae": None, "bMae": None,
                "maeDifference": None, "aMape": None, "bMape": None,
                "mapeDifferencePercentagePoints": None, "aLogMse": None, "bLogMse": None,
                "logMseDifference": None}
    ae_a, ae_b, ape_a, ape_b, le_a, le_b = [], [], [], [], [], []
    for row in rows:
        actual, pred_a, pred_b = row["actual"], row[a], row[b]
        if actual <= 0 or pred_a <= 0 or pred_b <= 0:
            raise ValueError("paired price metrics require positive actual and forecasts")
        ae_a.append(abs(pred_a - actual))
        ae_b.append(abs(pred_b - actual))
        ape_a.append(100.0 * abs(pred_a - actual) / actual)
        ape_b.append(100.0 * abs(pred_b - actual) / actual)
        le_a.append(math.log(pred_a / actual) ** 2)
        le_b.append(math.log(pred_b / actual) ** 2)
    a_mae, b_mae = _mean(ae_a), _mean(ae_b)
    a_mape, b_mape = _mean(ape_a), _mean(ape_b)
    a_log, b_log = _mean(le_a), _mean(le_b)
    return {"n": len(rows),
            "uniqueOriginDates": len({r["origin"] for r in rows if r.get("origin") is not None}),
            "uniqueTargetDates": len({r["target"] for r in rows}),
            "aMae": a_mae, "bMae": b_mae, "maeDifference": a_mae - b_mae,
            "aMape": a_mape, "bMape": b_mape,
            "mapeDifferencePercentagePoints": a_mape - b_mape,
            "aLogMse": a_log, "bLogMse": b_log, "logMseDifference": a_log - b_log}


def _fixed_cells(rows: list[dict[str, Any]], a: str, b: str,
                 cells: set[tuple[str, int]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["product"], row["horizon"])].append(row)
    by_cell = {f"{p}/{h}": _price_summary(grouped.get((p, h), []), a, b)
               for p, h in sorted(cells)}
    missing = sorted(f"{p}/{h}" for p, h in cells if not grouped.get((p, h)))
    result: dict[str, Any] = {"status": "incomplete_fixed_cell_support" if missing else "complete_fixed_cell_macro",
        "expectedCells": len(cells), "cellsIncluded": len(cells) - len(missing),
        "missingCells": missing, "byCell": by_cell}
    fields = ("aMae", "bMae", "maeDifference", "aMape", "bMape",
              "mapeDifferencePercentagePoints", "aLogMse", "bLogMse", "logMseDifference")
    if missing:
        result.update({field: None for field in fields})
    else:
        for field in fields:
            result[field] = math.fsum(float(by_cell[f"{p}/{h}"][field]) for p, h in sorted(cells)) / len(cells)
    return result


def _coverage_counts(rows: list[dict[str, Any]], actual_state: dict[tuple[str, dt.date, int], str],
                     baseline_prediction: dict[tuple[str, dt.date, int], float | None],
                     policy_prediction: dict[tuple[str, dt.date, int], float | None],
                     start: dt.date, end: dt.date) -> dict[str, Any]:
    by_cell: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    by_product: dict[str, list[dict[str, Any]]] = defaultdict(list)
    extra_fallbacks: list[dict[str, Any]] = []
    fallback_reasons: Counter[str] = Counter()
    coverage_losses: list[dict[str, Any]] = []
    coverage_gains: list[dict[str, Any]] = []
    for row in rows:
        if not row["eligible"]:
            continue
        key = row["key"]
        product = row["product"]
        by_cell[(product, row["horizon"])].append(row)
        by_product[product].append(row)
        if product == "garlic":
            reason = row.get("fallbackReason")
            if isinstance(reason, str) and reason:
                fallback_reasons[reason] += 1
            if baseline_prediction[key] is not None and not row["usedS"]:
                extra_fallbacks.append({"product": product, "target": key[1].isoformat(),
                                        "horizon": key[2], "reason": reason})
        if baseline_prediction[key] is not None and policy_prediction[key] is None:
            coverage_losses.append({"product": product, "target": key[1].isoformat(), "horizon": key[2]})
        if baseline_prediction[key] is None and policy_prediction[key] is not None:
            coverage_gains.append({"product": product, "target": key[1].isoformat(), "horizon": key[2]})

    def counts(values: list[dict[str, Any]]) -> dict[str, int]:
        output = Counter()
        baseline_issued = [r for r in values if baseline_prediction[r["key"]] is not None]
        policy_issued = [r for r in values if policy_prediction[r["key"]] is not None]
        positive_pairs = [r for r in values if actual_state[r["key"]] == "positive_actual"
                          and baseline_prediction[r["key"]] is not None
                          and policy_prediction[r["key"]] is not None]
        def distinct(rows: list[dict[str, Any]], field: str) -> int:
            return len({r[field] for r in rows})
        output["scheduledUniqueOriginDates"] = distinct(values, "originDate")
        output["scheduledUniqueTargetDates"] = distinct(values, "targetDate")
        output["baselineIssuedUniqueOriginDates"] = distinct(baseline_issued, "originDate")
        output["baselineIssuedUniqueTargetDates"] = distinct(baseline_issued, "targetDate")
        output["policyIssuedUniqueOriginDates"] = distinct(policy_issued, "originDate")
        output["policyIssuedUniqueTargetDates"] = distinct(policy_issued, "targetDate")
        output["positivePairedUniqueOriginDates"] = distinct(positive_pairs, "originDate")
        output["positivePairedUniqueTargetDates"] = distinct(positive_pairs, "targetDate")
        for row in values:
            output["eligibleKeys"] += 1
            output["baselineIssued"] += int(baseline_prediction[row["key"]] is not None)
            output["policyIssued"] += int(policy_prediction[row["key"]] is not None)
            output["baselineWithheldNotIssued"] += int(baseline_prediction[row["key"]] is None)
            output["policyWithheldNotIssued"] += int(policy_prediction[row["key"]] is None)
            output["usedS"] += int(row["product"] == "garlic" and row["usedS"])
            output["garlicFallbackToE106"] += int(row["product"] == "garlic" and not row["usedS"])
            output["positiveActual"] += int(actual_state[row["key"]] == "positive_actual")
            output["noSourceObservation"] += int(actual_state[row["key"]] == "no_source_observation")
            output["nonpositiveOrInvalidActual"] += int(actual_state[row["key"]] == "nonpositive_or_invalid_actual")
        return dict(output)

    by_product_horizon = {f"{p}/{h}": counts(by_cell.get((p, h), [])) for p in PRODUCTS for h in HORIZONS}
    product_counts = {p: counts(by_product.get(p, [])) for p in PRODUCTS}
    by_month: dict[str, Any] = {}
    for month in _phase_months(start, end):
        month_rows = [r for r in rows if r["eligible"] and r["targetDate"].strftime("%Y-%m") == month]
        month_cells: dict[str, Any] = {}
        for product in PRODUCTS:
            product_rows = [r for r in month_rows if r["product"] == product]
            month_cells[product] = counts(product_rows)
            for horizon in HORIZONS:
                cell_rows = [r for r in product_rows if r["horizon"] == horizon]
                month_cells[f"{product}/{horizon}"] = counts(cell_rows)
        by_month[month] = {"eligibleByProductAndHorizon": month_cells}
    return {"eligibleByProduct": product_counts, "eligibleByProductHorizon": by_product_horizon,
            "eligibleByMonth": by_month,
            "identityCountNote": "distinct dates are calendar identities; overlapping horizons are not independent observations",
            "garlicFallbackReasonCounts": dict(sorted(fallback_reasons.items())),
            "garlicAdditionalFallbackKeys": extra_fallbacks,
            "coverageLossKeys": coverage_losses, "coverageGainKeys": coverage_gains,
            "coveragePreserved": not coverage_losses,
            "noAdditionalGarlicFallback": not extra_fallbacks}


def _dm_family(pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tests = []
    for horizon in HORIZONS:
        cell = [r for r in pairs if r["product"] == "garlic" and r["horizon"] == horizon]
        for loss in ("ae", "ape"):
            differences: dict[dt.date, float] = {}
            for row in cell:
                day = row["targetDate"]
                if day in differences:
                    raise ValueError("duplicate garlic target date within a horizon DM cell")
                error_a = abs(row["policy"] - row["actual"])
                error_b = abs(row["baseline"] - row["actual"])
                differences[day] = error_a - error_b if loss == "ae" else 100.0 * (error_a - error_b) / row["actual"]
            for bandwidth in BANDWIDTHS:
                tests.append({"comparison": "SBlend_minus_E106", "product": "garlic",
                    "horizon": horizon, "loss": loss,
                    "units": "percentage_points" if loss == "ape" else "price_units_per_kg",
                    "bandwidthDays": bandwidth, **stats._dm(dict(sorted(differences.items())), bandwidth)})
    for bandwidth in BANDWIDTHS:
        family = [r for r in tests if r["bandwidthDays"] == bandwidth]
        if len(family) != 20:
            raise AssertionError("the frozen garlic DM family must contain exactly 20 tests per bandwidth")
        stats._holm(family)
    summaries = []
    for bandwidth in BANDWIDTHS:
        family = [r for r in tests if r["bandwidthDays"] == bandwidth]
        for loss in ("ae", "ape"):
            loss_rows = [r for r in family if r["loss"] == loss]
            improvements = sum(r["holmSignificant"] and r["meanLossDifference"] is not None
                               and r["meanLossDifference"] < 0 for r in loss_rows)
            worsenings = sum(r["holmSignificant"] and r["meanLossDifference"] is not None
                             and r["meanLossDifference"] > 0 for r in loss_rows)
            summaries.append({"bandwidthDays": bandwidth, "loss": loss, "familySize": len(loss_rows),
                "significantImprovements": improvements, "significantWorsenings": worsenings,
                "inconclusiveOrUnsupported": len(loss_rows) - improvements - worsenings,
                "interpretation": "percentage_point_difference" if loss == "ape" else "price_units_per_kg"})
    return tests, summaries


def _bootstrap(rows: list[dict[str, Any]], cells: set[tuple[str, int]], *,
               start: dt.date, end: dt.date, label: str) -> dict[str, Any]:
    observed = {(r["product"], r["horizon"]) for r in rows}
    grouped = Counter((r["product"], r["horizon"]) for r in rows)
    missing = sorted(f"{p}/{h}" for p, h in cells if grouped[(p, h)] == 0)
    if observed != cells or missing:
        return {"status": "unsupported_missing_fixed_cells", "scope": label,
                "expectedCells": len(cells), "missingCells": missing,
                "byMetric": {metric: {str(block): {"interval95": None, "status": "unsupported_missing_fixed_cells"}
                                       for block in BLOCKS} for metric in ("ae", "ape")}}
    result: dict[str, Any] = {"status": "complete_fixed_cell_support", "scope": label,
        "expectedCells": len(cells), "missingCells": [], "byMetric": {}}
    for metric in ("ae", "ape"):
        result["byMetric"][metric] = {}
        for block in BLOCKS:
            estimate = circular_block_bootstrap(rows, a="policy", b="baseline", metric=metric,
                block=block, replicates=REPLICATES, seed=SEED,
                calendar_start=start, calendar_end=end)
            estimate["units"] = "percentage_points" if metric == "ape" else "price_units_per_kg"
            result["byMetric"][metric][str(block)] = estimate
    return result


def _component_summary(rows: list[dict[str, Any]], r_field: str, s_field: str,
                       *, direct: bool) -> dict[str, Any]:
    pairs = []
    for row in rows:
        r_pred, s_pred = _positive(row.get(r_field)), _positive(row.get(s_field))
        if r_pred is None or s_pred is None:
            continue
        pairs.append({"target": row["target"], "origin": row.get("origin"),
                      "actual": row["actual"], "R": r_pred, "S": s_pred})
    summary = _price_summary(pairs, "S", "R")
    if not direct:
        summary.pop("aLogMse", None)
        summary.pop("bLogMse", None)
        summary.pop("logMseDifference", None)
    return summary


def _component_metrics(raw_rows: list[dict[str, Any]], phase_start: dt.date, phase_end: dt.date) -> dict[str, Any]:
    garlic = [row for row in raw_rows if row["product"] == "garlic" and row["actual"] is not None]

    def summarize_segment(rows: list[dict[str, Any]]) -> dict[str, Any]:
        by_h = {str(h): {"direct": _component_summary([r for r in rows if r["horizon"] == h],
                    "RDirectPrediction", "SDirectPrediction", direct=True),
                "sameSarimaBlend": _component_summary([r for r in rows if r["horizon"] == h],
                    "RBlendPrediction", "SBlendPrediction", direct=False)} for h in HORIZONS}
        expected = {("garlic", h) for h in HORIZONS}
        direct_rows = []
        blend_rows = []
        for row in rows:
            actual = row["actual"]
            rd, sd = _positive(row.get("RDirectPrediction")), _positive(row.get("SDirectPrediction"))
            rb, sb = _positive(row.get("RBlendPrediction")), _positive(row.get("SBlendPrediction"))
            if rd is not None and sd is not None:
                direct_rows.append({"product": "garlic", "horizon": row["horizon"], "target": row["target"],
                                    "origin": row["origin"],
                                    "actual": actual, "S": sd, "R": rd})
            if rb is not None and sb is not None:
                blend_rows.append({"product": "garlic", "horizon": row["horizon"], "target": row["target"],
                                   "origin": row["origin"],
                                   "actual": actual, "S": sb, "R": rb})
        direct_macro = _fixed_cells(direct_rows, "S", "R", expected)
        blend_macro = _fixed_cells(blend_rows, "S", "R", expected)
        # Keep the paired direct log-MSE table separate from the fixed-cell
        # price macro, whose score fields are all price errors.
        direct_macro["logMse"] = None
        if direct_macro["status"] == "complete_fixed_cell_macro":
            direct_macro["logMse"] = math.fsum(by_h[str(h)]["direct"]["aLogMse"]
                                                 for h in HORIZONS) / len(HORIZONS)
        month_keys = _phase_months(phase_start, phase_end)
        monthly = {}
        for month in month_keys:
            month_rows = [r for r in rows if r["targetDate"].strftime("%Y-%m") == month]
            month_by_h = {str(h): {"direct": _component_summary([r for r in month_rows if r["horizon"] == h],
                    "RDirectPrediction", "SDirectPrediction", direct=True),
                "sameSarimaBlend": _component_summary([r for r in month_rows if r["horizon"] == h],
                    "RBlendPrediction", "SBlendPrediction", direct=False)} for h in HORIZONS}
            direct_mrows, blend_mrows = [], []
            for row in month_rows:
                rd, sd = _positive(row.get("RDirectPrediction")), _positive(row.get("SDirectPrediction"))
                rb, sb = _positive(row.get("RBlendPrediction")), _positive(row.get("SBlendPrediction"))
                if rd is not None and sd is not None:
                    direct_mrows.append({"product": "garlic", "horizon": row["horizon"], "target": row["target"],
                                         "origin": row["origin"],
                                         "actual": row["actual"], "S": sd, "R": rd})
                if rb is not None and sb is not None:
                    blend_mrows.append({"product": "garlic", "horizon": row["horizon"], "target": row["target"],
                                        "origin": row["origin"],
                                        "actual": row["actual"], "S": sb, "R": rb})
            monthly[month] = {"byHorizon": month_by_h,
                "directFixed10Horizon": _fixed_cells(direct_mrows, "S", "R", expected),
                "sameSarimaBlendFixed10Horizon": _fixed_cells(blend_mrows, "S", "R", expected)}
        return {"byHorizon": by_h, "fixed10HorizonDirect": direct_macro,
                "fixed10HorizonSameSarimaBlend": blend_macro, "byMonth": monthly}

    return {"garlic": summarize_segment(garlic),
            "sourceRowsWithPositiveActual": len(garlic),
            "scope": "raw R/S component diagnostics; separate from E106 fallback policy coverage"}


def _identity_one(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    er, delta, actual_r = [], [], []
    for row in rows:
        y = row["actualLogRatio"]
        r = row["RLogPrediction"]
        s = row["SLogPrediction"]
        er.append(y - r)
        delta.append(s - r)
        actual_r.append(y)
    mse_s = math.fsum((y - s) ** 2 for y, s in
                      ((row["actualLogRatio"], row["SLogPrediction"]) for row in rows)) / len(rows)
    mse_r = math.fsum(v * v for v in er) / len(rows)
    squared = math.fsum(v * v for v in delta) / len(rows)
    offset = -2.0 * math.fsum(e * d for e, d in zip(er, delta)) / len(rows)
    rhs = squared + offset
    return {"n": len(rows), "uniqueTargetDates": len({r["target"] for r in rows}),
            "mseSMinusR": mse_s - mse_r, "correctionSquared": squared,
            "residualAlignmentOffset": offset, "identityRhs": rhs,
            "identityError": abs((mse_s - mse_r) - rhs)}


def _identity_fixed(rows: list[dict[str, Any]], cells: set[tuple[str, int]]) -> dict[str, Any]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["product"], row["horizon"])].append(row)
    by_cell = {f"{p}/{h}": _identity_one(grouped.get((p, h), [])) for p, h in sorted(cells)}
    missing = sorted(key for key, val in by_cell.items() if val is None)
    out = {"status": "incomplete_fixed_cell_support" if missing else "complete_fixed_cell_macro",
           "expectedCells": len(cells), "missingCells": missing, "byCell": by_cell}
    fields = ("mseSMinusR", "correctionSquared", "residualAlignmentOffset", "identityRhs")
    if missing:
        out.update({field: None for field in fields})
        out["identityError"] = None
        out["n"] = sum(int(x["n"]) for x in by_cell.values() if x)
        out["uniqueTargetDates"] = len({r["target"] for r in rows})
    else:
        for field in fields:
            out[field] = math.fsum(float(by_cell[k][field]) for k in sorted(by_cell)) / len(cells)
        out["identityError"] = abs(out["mseSMinusR"] - out["identityRhs"])
        out["n"] = sum(int(x["n"]) for x in by_cell.values())
        out["uniqueTargetDates"] = len({r["target"] for r in rows})
    return out


def _raw_identity(raw_rows: list[dict[str, Any]], start: dt.date, end: dt.date) -> dict[str, Any]:
    rows = [r for r in raw_rows if r["product"] == "garlic" and r.get("actualLogRatio") is not None]
    cells = {("garlic", h) for h in HORIZONS}
    by_horizon = {str(h): _identity_one([r for r in rows if r["horizon"] == h]) for h in HORIZONS}
    months = {}
    for month in _phase_months(start, end):
        part = [r for r in rows if r["targetDate"].strftime("%Y-%m") == month]
        months[month] = _identity_fixed(part, cells)
    blocks = {}
    for width in BLOCKS:
        items = []
        offset = 0
        while start + dt.timedelta(days=offset) <= end:
            left = start + dt.timedelta(days=offset)
            right = min(end, left + dt.timedelta(days=width - 1))
            part = [r for r in rows if left <= r["targetDate"] <= right]
            summary = _identity_fixed(part, cells) if part else None
            items.append({"start": left.isoformat(), "end": right.isoformat(),
                          "calendarDays": (right - left).days + 1,
                          "n": len(part), "uniqueTargetDates": len({r["target"] for r in part}),
                          "status": "empty_block" if not part else summary["status"],
                          "identity": summary})
            offset += width
        blocks[str(width)] = items
    return {"scope": "garlic raw S versus R log-ratio identity; same positive-actual component pairs",
            "overall": _identity_fixed(rows, cells), "byHorizon": by_horizon,
            "byMonth": months, "nonoverlapCalendarBlocks": blocks,
            "blockCalendarAnchor": start.isoformat(), "blockCalendarEnd": end.isoformat(),
            "missingCalendarDatesWereNotCompressed": True}


def score_manifest(doc: dict[str, Any], *, base_forecasts: dict[tuple[str, dt.date, int], dict[str, Any]],
                   actuals: dict[str, dict[dt.date, Any]], phase: str,
                   all_corrections_zero: bool, numerical_checks_passed: bool) -> dict[str, Any]:
    """Score the fixed S policy against E106 using only supplied verified objects.

    The first positional input is a truth-free forecast manifest.  `actuals`
    is joined in memory by product/date only after the caller has verified its
    pinned source.  Missing score cells remain null and fail the primary gate.
    """
    if phase not in PHASES:
        raise ValueError("phase must be validation or final")
    if type(all_corrections_zero) is not bool or type(numerical_checks_passed) is not bool:
        raise ValueError("zero-correction and numerical-check flags must be booleans")
    if not isinstance(doc, dict) or doc.get("phase") != phase or doc.get("targetValuesAttached") is not False:
        raise ValueError("scorer requires a matching truth-free phase manifest")
    target_start, target_end, origin_start = PHASES[phase]
    expected = set(base_forecasts)
    if not expected:
        raise ValueError("verified E106 baseline grid is empty")
    for product, target, horizon in expected:
        target = _date(target, "E106 baseline target")
        if product not in PRODUCTS or type(horizon) is not int or horizon not in HORIZONS:
            raise ValueError("E106 baseline key is outside the frozen product/horizon grid")
        if not target_start <= target <= target_end:
            raise ValueError("E106 baseline target is outside the selected phase window")
    manifest_rows = doc.get("forecasts")
    if not isinstance(manifest_rows, list) or len(manifest_rows) != len(expected):
        raise ValueError("manifest does not contain the frozen complete forecast grid")
    indexed: dict[tuple[str, dt.date, int], dict[str, Any]] = {}
    for row in manifest_rows:
        if not isinstance(row, dict):
            raise ValueError("manifest forecast row must be an object")
        product = row.get("product")
        target = _date(row.get("target"), "forecast target")
        origin = _date(row.get("origin"), "forecast origin")
        horizon = row.get("horizon")
        key = (product, target, horizon)
        if type(horizon) is not int or key not in expected or key in indexed:
            raise ValueError("manifest has a duplicate or out-of-grid forecast key")
        if origin != target - dt.timedelta(days=horizon):
            raise ValueError("forecast origin and horizon do not align")
        eligible = origin >= origin_start
        if type(row.get("eligible")) is not bool or row["eligible"] != eligible:
            raise ValueError("manifest eligibility differs from frozen phase origin rule")
        if type(row.get("usedS")) is not bool:
            raise ValueError("manifest row must state whether S was used")
        if row.get("status") not in ("issued", "not_issued"):
            raise ValueError("manifest row status is invalid")
        pred = _positive(row.get("prediction"))
        if (row["status"] == "issued") != (pred is not None):
            raise ValueError("manifest row status and prediction disagree")
        base_row = base_forecasts.get(key)
        if not isinstance(base_row, dict) or not isinstance(base_row.get("candidate"), dict):
            raise ValueError(f"missing verified E106 baseline row for {key}")
        base_candidate = base_row["candidate"]
        base_pred = _positive(base_candidate.get("prediction"))
        if base_candidate.get("status") not in ("issued", "not_issued") or (
                (base_candidate.get("status") == "issued") != (base_pred is not None)):
            raise ValueError("E106 baseline status and prediction disagree")
        if product != "garlic":
            if row["usedS"] or row.get("fallbackReason") != "non_garlic_exact_E106":
                raise ValueError("non-garlic policy must explicitly retain E106")
            if row["status"] != base_candidate["status"] or pred != base_pred:
                raise ValueError("non-garlic policy differs from exact E106 baseline")
        elif row["usedS"]:
            if not eligible or row["status"] != "issued" or pred is None or row.get("fallbackReason"):
                raise ValueError("successful S policy row is malformed or outside eligible origins")
        else:
            if row["status"] != base_candidate["status"] or pred != base_pred:
                raise ValueError("garlic fallback must copy the E106 candidate status and prediction")
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError("manifest forecast grid is incomplete")

    score_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    actual_state: dict[tuple[str, dt.date, int], str] = {}
    baseline_prediction: dict[tuple[str, dt.date, int], float | None] = {}
    policy_prediction: dict[tuple[str, dt.date, int], float | None] = {}
    for key in sorted(expected):
        row = indexed[key]
        product, target, horizon = key
        base_candidate = base_forecasts[key]["candidate"]
        base_pred = _positive(base_candidate.get("prediction"))
        s_pred = _positive(row.get("prediction"))
        policy_pred = s_pred if product == "garlic" and row["usedS"] else base_pred
        state, actual = _actual_record(actuals, product, target)
        actual_state[key] = state
        baseline_prediction[key] = base_pred
        policy_prediction[key] = policy_pred
        origin = _date(row["origin"], "forecast origin")
        if row["eligible"] and actual is not None and base_pred is not None and policy_pred is not None:
            score_rows.append({"key": key, "product": product, "target": target.isoformat(),
                "targetDate": target, "origin": origin, "horizon": horizon, "actual": actual,
                "policy": policy_pred, "baseline": base_pred})
        raw_item = {"key": key, "product": product, "target": target.isoformat(), "targetDate": target,
                    "origin": origin,
                    "horizon": horizon, "actual": actual}
        if row["eligible"] and product == "garlic" and actual is not None:
            p0 = _positive(row.get("p0"))
            r_log, s_log = _number(row.get("RLogPrediction")), _number(row.get("SLogPrediction"))
            if p0 is not None and r_log is not None and s_log is not None:
                raw_item.update({"actualLogRatio": math.log(actual / p0),
                    "RLogPrediction": r_log, "SLogPrediction": s_log})
        raw_item.update({field: row.get(field) for field in
            ("RDirectPrediction", "SDirectPrediction", "RBlendPrediction", "SBlendPrediction", "sarimaPrediction")})
        raw_rows.append(raw_item)

    coverage_rows = [{**row, "key": (row["product"], _date(row["target"], "target"), row["horizon"]),
                      "targetDate": _date(row["target"], "target"),
                      "originDate": _date(row["origin"], "origin")}
                     for row in indexed.values()]
    coverage = _coverage_counts(coverage_rows, actual_state, baseline_prediction, policy_prediction,
                                target_start, target_end)

    garlic_cells = {("garlic", h) for h in HORIZONS}
    all_cells = {(p, h) for p in PRODUCTS for h in HORIZONS}
    policy_garlic = _fixed_cells(score_rows, "policy", "baseline", garlic_cells)
    policy_all = _fixed_cells(score_rows, "policy", "baseline", all_cells)
    per_product_horizon = {f"{p}/{h}": _price_summary(
        [r for r in score_rows if r["product"] == p and r["horizon"] == h], "policy", "baseline")
        for p in PRODUCTS for h in HORIZONS}
    month_names = _phase_months(target_start, target_end)
    monthly_policy = {}
    for month in month_names:
        month_rows = [r for r in score_rows if r["targetDate"].strftime("%Y-%m") == month]
        monthly_policy[month] = {"garlic10H": _fixed_cells(month_rows, "policy", "baseline", garlic_cells),
                                 "fourCrop40": _fixed_cells(month_rows, "policy", "baseline", all_cells)}

    dm_tests, dm_summary = _dm_family(score_rows)
    bootstrap_rows = [{"product": r["product"], "horizon": r["horizon"], "target": r["target"],
                       "actual": r["actual"], "policy": r["policy"], "baseline": r["baseline"]}
                      for r in score_rows]
    boot_garlic = _bootstrap([r for r in bootstrap_rows if r["product"] == "garlic"], garlic_cells,
                             start=target_start, end=target_end, label="garlic 10 horizons")
    boot_all = _bootstrap(bootstrap_rows, all_cells, start=target_start, end=target_end,
                          label="four crops x 10 horizons; auxiliary macro, not four independent successes")
    components = _component_metrics(raw_rows, target_start, target_end)
    raw_identity = _raw_identity(raw_rows, target_start, target_end)

    minimum_horizon_pairs = min(policy_garlic["byCell"][f"garlic/{h}"]["n"] for h in HORIZONS)
    delta_mape = policy_garlic["mapeDifferencePercentagePoints"]
    delta_mae = policy_garlic["maeDifference"]
    complete_primary = policy_garlic["status"] == "complete_fixed_cell_macro"
    checks = {
        "completeFixed10HorizonPairs": complete_primary,
        "mapeImproves": complete_primary and delta_mape is not None and delta_mape < 0,
        "maeDoesNotWorsen": complete_primary and delta_mae is not None and delta_mae <= 0,
        "coveragePreserved": coverage["coveragePreserved"],
        "noAdditionalGarlicFallback": coverage["noAdditionalGarlicFallback"],
        "notAllCorrectionsZero": not all_corrections_zero,
        "numericalChecksPass": numerical_checks_passed,
    }
    decision = {**checks, "passes": all(checks.values()),
        "status": "pass" if all(checks.values()) else "fail",
        "deltaMapePercentagePoints": delta_mape if complete_primary else None,
        "deltaMaePerKg": delta_mae if complete_primary else None,
        "minimumPositivePairedTargetsAcrossGarlicHorizons": minimum_horizon_pairs,
        "failureReasons": [key for key, value in checks.items() if not value],
        "interpretation": "pass" if all(checks.values()) else "retain_E106_stop_S_family"}

    eligible_key_count = sum(bool(row["eligible"]) for row in indexed.values())
    return {"schema": "e106-orthogonal-residual-score-v1", "phase": phase,
        "phaseTargetWindow": {"start": target_start.isoformat(), "end": target_end.isoformat()},
        "manifestKeyCount": len(indexed), "eligibleKeyCount": eligible_key_count,
        "ineligibleKeyCount": len(indexed) - eligible_key_count,
        "manifestSha256": doc.get("manifestSha256"),
        "protocolSha256": (doc.get("protocol") or {}).get("protocolSha256"),
        "targetValuesAttachedToManifest": doc.get("targetValuesAttached") is True,
        "scoredPositivePairCount": len(score_rows), "coverage": coverage,
        "wholePolicy": {"candidate": "S/SARIMA for usedS garlic keys; exact E106 fallback and other crops",
            "primaryGarlic10Horizon": policy_garlic, "auxiliaryFourCrop40Cell": policy_all,
            "byProductHorizon": per_product_horizon, "byMonth": monthly_policy},
        "rawGarlicComponents": components,
        "dmTests": dm_tests, "dmSummary": dm_summary,
        "blockBootstrap": {"garlic10HorizonPrimary": boot_garlic, "fourCrop40CellAuxiliary": boot_all,
            "blockDays": list(BLOCKS), "replicates": REPLICATES, "seed": SEED,
            "targetDatesResampledJointly": True},
        "rawLogRatioIdentity": raw_identity,
        "primaryDecision": decision}
