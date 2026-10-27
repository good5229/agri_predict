"""Frozen garlic price-regime shrinkage assessment on pinned historical forecasts."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics

from . import adaptive_selector as source_schema
from . import direct_horizon as dh
from . import garlic_prospective

INPUT_PATH = source_schema.INPUT_PATH
INPUT_SHA256 = source_schema.INPUT_SHA256
STATIC_REPORT_PATH = garlic_prospective.STATIC_REPORT
STATIC_REPORT_SHA256 = garlic_prospective.STATIC_SHA256
PRODUCT = "garlic"
HORIZONS = tuple(range(1, 11))
COMPARATORS = ("static", "persistence", "daily_ridge")
LOSSES = (("mae", "absolute_error"), ("mape", "absolute_percentage_error"))
HAC = (9, 28)
EVAL_START = dt.date(2024, 1, 1)
EVAL_END = dt.date(2026, 9, 27)
MIN_PAIRS = 30
ALPHA = 0.05
STABLE_LOG_RATIO_MAX = 0.03
STABLE_WEIGHT = 0.50
VOLATILE_WEIGHT = 1.00
RECENT_DAYS = 7
PRIOR_DAYS = 21
MIN_RECENT = 3
MIN_PRIOR = 9
MAX_AGE_DAYS = 14
FROZEN_WEIGHTS = {h: (1.0 if h in (3, 4) else 0.75) for h in HORIZONS}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _positive(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _season(day: dt.date) -> str:
    if day.month in (12, 1, 2): return "winter"
    if day.month in (3, 4, 5): return "spring"
    if day.month in (6, 7, 8): return "summer"
    return "autumn"


def classify_regime(actuals: dict[dt.date, float], origin: dt.date) -> dict:
    """Classify from observed prices strictly before origin; never fill missing dates."""
    observed = {day: value for day, raw in actuals.items()
                if day < origin and (value := _positive(raw)) is not None}
    recent_start = origin - dt.timedelta(days=RECENT_DAYS)
    prior_start = origin - dt.timedelta(days=RECENT_DAYS + PRIOR_DAYS)
    recent = [value for day, value in observed.items() if recent_start <= day < origin]
    prior = [value for day, value in observed.items() if prior_start <= day < recent_start]
    latest = max(observed, default=None)
    recent_median = statistics.median(recent) if recent else None
    prior_median = statistics.median(prior) if prior else None
    result = {"regime": None, "weight": None, "fallbackReason": None,
              "recentObservedDays": len(recent), "priorObservedDays": len(prior),
              "recentMedian": recent_median, "priorMedian": prior_median,
              "latestObservedDate": latest.isoformat() if latest else None,
              "latestAgeDays": (origin - latest).days if latest else None,
              "absoluteLogMedianRatio": None}
    if latest is None or (origin - latest).days > MAX_AGE_DAYS:
        result["fallbackReason"] = "source_stale_or_missing"
        return result
    if len(recent) < MIN_RECENT:
        result["fallbackReason"] = "insufficient_recent_observations"
        return result
    if len(prior) < MIN_PRIOR:
        result["fallbackReason"] = "insufficient_prior_observations"
        return result
    ratio = recent_median / prior_median
    if not math.isfinite(ratio) or ratio <= 0:
        result["fallbackReason"] = "invalid_log_median_ratio"
        return result
    log_ratio = math.log(ratio)
    if not math.isfinite(log_ratio):
        result["fallbackReason"] = "invalid_log_median_ratio"
        return result
    value = abs(log_ratio)
    result["absoluteLogMedianRatio"] = value
    result["regime"] = "stable" if value <= STABLE_LOG_RATIO_MAX else "volatile"
    result["weight"] = STABLE_WEIGHT if result["regime"] == "stable" else VOLATILE_WEIGHT
    return result


def _revise(persistence, ridge, static_weight, regime):
    static = persistence + static_weight * (ridge - persistence)
    if _positive(static) is None:
        return {"prediction": None, "regimePrediction": None, "weight": static_weight,
                "regime": regime["regime"], "fallbackReason": "invalid_static_fallback"}
    if regime["weight"] is None:
        return {"prediction": static, "regimePrediction": None, "weight": static_weight,
                "regime": None, "fallbackReason": regime["fallbackReason"]}
    candidate = persistence + regime["weight"] * (ridge - persistence)
    if _positive(candidate) is None:
        return {"prediction": static, "regimePrediction": None, "weight": static_weight,
                "regime": regime["regime"], "fallbackReason": "invalid_regime_candidate"}
    return {"prediction": candidate, "regimePrediction": candidate, "weight": regime["weight"],
            "regime": regime["regime"], "fallbackReason": None}


def _blend(persistence, ridge, weight):
    persistence = _positive(persistence)
    ridge = _positive(ridge)
    if persistence is None or ridge is None:
        return None
    value = persistence + weight * (ridge - persistence)
    return value if _positive(value) is not None else None


def _metric(actuals, predictions):
    return dh._metrics(actuals, predictions)


def _pair_metrics(rows, comparator):
    actual = [r["actual"] for r in rows]
    candidate = [r["candidate"] for r in rows]
    other = [r[comparator] for r in rows]
    candidate_stats = _metric(actual, candidate)
    comparator_stats = _metric(actual, other)
    return {"n": len(rows), "candidate": candidate_stats, "comparator": comparator_stats,
            "meanLossDifference": {"mae": candidate_stats["mae"] - comparator_stats["mae"] if rows else None,
                                  "mape": candidate_stats["mape"] - comparator_stats["mape"] if rows else None}}


def _test(crop, horizon, comparator, loss, rows):
    metric = "mae" if loss == "absolute_error" else "mape"
    differences = {}
    for row in rows:
        y = row["actual"]
        a = abs(y - row["candidate"])
        b = abs(y - row[comparator])
        differences[row["target"]] = a - b if metric == "mae" else 100 * (a - b) / y
    result = dh._dm(differences, 9)
    sensitivity = dh._dm(differences, 28)
    for dm in (result, sensitivity):
        dm.update({"productId": crop, "horizon": horizon, "comparator": comparator, "loss": loss})
    return {"productId": crop, "horizon": horizon, "comparator": comparator, "loss": loss,
            "n": len(rows), "candidateMetric": _metric([r["actual"] for r in rows], [r["candidate"] for r in rows])[metric],
            "comparatorMetric": _metric([r["actual"] for r in rows], [r[comparator] for r in rows])[metric],
            "meanLossDifference": (sum(differences.values()) / len(differences)) if differences else None,
            "dm": {"9": result, "28": sensitivity}}


def _conclusion(tests, candidate_nondeclining):
    enough = all(t["n"] >= MIN_PAIRS and t["dm"]["9"].get("status") == "tested" for t in tests)
    pointwise_better = all(t["candidateMetric"] is not None and t["comparatorMetric"] is not None and
                           t["candidateMetric"] < t["comparatorMetric"] for t in tests)
    significant = all(t["dm"]["9"].get("holmSignificant", False) and
                      t["dm"]["9"].get("meanLossDifference", 0) < 0 for t in tests)
    adverse = any(t["dm"]["9"].get("holmSignificant", False) and
                  t["dm"]["9"].get("meanLossDifference", 0) > 0 for t in tests)
    conclusion = ("inference_unavailable" if not enough else
                  "improvement" if pointwise_better and significant and candidate_nondeclining else
                  "worsening" if adverse else "inconclusive")
    return {"conclusion": conclusion, "inferenceAvailable": enough,
            "pointwiseLossesLower": pointwise_better, "allSixHac9TestsImprove": significant,
            "coverageNondeclining": candidate_nondeclining,
            "hac28Sensitivity": "available" if all(t["dm"]["28"].get("status") == "tested" for t in tests) else "unavailable"}


def build_assessment(source, selection, *, input_sha256=None, static_sha256=None):
    """Pure evaluator; caller supplies parsed immutable predictions and validated selections."""
    products = source_schema._validate(source)
    if set(selection) != set(HORIZONS) or any(selection[h].get("model") != "ridge" or
                                              selection[h].get("weight") != FROZEN_WEIGHTS[h] for h in HORIZONS):
        raise ValueError("garlic static-blend selections do not match frozen weights")
    if input_sha256 is not None and input_sha256 != INPUT_SHA256:
        raise ValueError("predictions input SHA-256 does not match frozen input")
    if static_sha256 is not None and static_sha256 != STATIC_REPORT_SHA256:
        raise ValueError("static-blend report SHA-256 does not match frozen selection")
    data = products[PRODUCT]
    actuals = data["actuals"]
    forecasts = data["forecasts"]
    baseline = forecasts["baseline"]
    ridge = forecasts["ridge"]
    candidates = {}
    candidate_states = {}
    fallback_counts = {}
    for key in set(baseline) & set(ridge):
        origin, target, horizon = key
        if horizon not in HORIZONS:
            continue
        regime = classify_regime(actuals, origin)
        revised = _revise(baseline[key], ridge[key], FROZEN_WEIGHTS[horizon], regime)
        candidate_states[key] = {"decision": revised, "classification": regime}
        if (revised["fallbackReason"] and origin >= EVAL_START and EVAL_START <= target <= EVAL_END and target in actuals):
            fallback_counts[revised["fallbackReason"]] = fallback_counts.get(revised["fallbackReason"], 0) + 1
        if revised["prediction"] is not None:
            candidates[key] = {"decision": revised, "classification": regime}
    rows_by_horizon = {h: [] for h in HORIZONS}
    counts = {h: {"eligibleActualTargets": 0, "candidate": 0, "static": 0, "persistence": 0, "daily_ridge": 0, "common": 0}
              for h in HORIZONS}
    for h in HORIZONS:
        for target, actual in actuals.items():
            origin = target - dt.timedelta(days=h)
            if not (EVAL_START <= target <= EVAL_END and origin >= EVAL_START and _positive(actual) is not None):
                continue
            counts[h]["eligibleActualTargets"] += 1
            key = (origin, target, h)
            if _positive(baseline.get(key)) is not None:
                counts[h]["persistence"] += 1
            if _blend(baseline.get(key), ridge.get(key), FROZEN_WEIGHTS[h]) is not None:
                counts[h]["static"] += 1
            if _positive(ridge.get(key)) is not None:
                counts[h]["daily_ridge"] += 1
            if key in candidates and _positive(candidates[key]["decision"]["prediction"]) is not None:
                counts[h]["candidate"] += 1
    for key, revised in candidates.items():
        origin, target, horizon = key
        if not (EVAL_START <= target <= EVAL_END and origin >= EVAL_START):
            continue
        actual = actuals.get(target)
        if actual is None or _positive(actual) is None:
            continue
        values = {"static": _blend(baseline.get(key), ridge.get(key), FROZEN_WEIGHTS[horizon]),
                  "persistence": baseline.get(key), "daily_ridge": ridge.get(key)}
        if all(_positive(value) is not None for value in values.values()):
            decision = revised["decision"]
            classification = revised["classification"]
            rows_by_horizon[horizon].append({"origin": origin, "target": target, "actual": float(actual),
                "candidate": decision["prediction"], "static": values["static"], "persistence": values["persistence"],
                "daily_ridge": values["daily_ridge"], "regime": classification["regime"] or "fallback",
                "weight": decision["weight"], "fallbackReason": decision["fallbackReason"],
                "regimeFallbackReason": classification["fallbackReason"]})
            counts[horizon]["common"] += 1

    comparisons = []
    strata = []
    summary = []
    horizon_summary = []
    coverage_gate = {}
    all_tests = {lag: [] for lag in HAC}
    for h in HORIZONS:
        rows = sorted(rows_by_horizon[h], key=lambda r: (r["target"], r["origin"]))
        for comp in COMPARATORS:
            pair = _pair_metrics(rows, comp)
            summary.append({"horizon": h, "comparatorName": comp, **pair})
            for _, loss in LOSSES:
                test = _test(PRODUCT, h, comp, loss, rows)
                comparisons.append(test)
                for lag in HAC:
                    all_tests[lag].append(test["dm"][str(lag)])
            for key_type, labeler in (("year", lambda r: str(r["target"].year)),
                                      ("season", lambda r: _season(r["target"])),
                                      ("regime", lambda r: r["regime"])):
                groups = {}
                for row in rows:
                    groups.setdefault(labeler(row), []).append(row)
                for label, group in sorted(groups.items()):
                    stats = _pair_metrics(group, comp)
                    strata.append({"horizon": h, "comparatorName": comp, "stratum": key_type,
                                   "value": label, **stats})
        common_n = len(rows)
        nondeclining = all(counts[h]["candidate"] >= counts[h][comp] for comp in COMPARATORS)
        decision_groups = {}
        for row in rows:
            key = (row["regime"], row["weight"], row["fallbackReason"], row["regimeFallbackReason"])
            decision_groups[key] = decision_groups.get(key, 0) + 1
        decision_audit = [{"regime": regime, "appliedWeight": weight, "fallbackReason": reason,
                           "classificationFallbackReason": classifier_reason, "n": n}
                          for (regime, weight, reason, classifier_reason), n in sorted(decision_groups.items(), key=lambda item: str(item[0]))]
        summary_row = {"horizon": h, "pairedN": common_n,
                       "coverage": counts[h],
                       "decisionAudit": decision_audit,
                       "equalHorizonMape": {name: _metric([r["actual"] for r in rows], [r[name] if name != "candidate" else r["candidate"] for r in rows])["mape"] for name in ("candidate", *COMPARATORS)},
                       "fallbackReasons": {reason: count for reason in sorted(fallback_counts)
                           if (count := sum(1 for key, value in candidate_states.items() if key[2] == h and key[0] >= EVAL_START and EVAL_START <= key[1] <= EVAL_END and key[1] in actuals and value["decision"]["fallbackReason"] == reason)) > 0}}
        horizon_summary.append(summary_row)
        coverage_gate[h] = nondeclining

    for lag in HAC:
        dh._holm(all_tests[lag])
    # Holm mutates shared DM records, so the comparison entries now contain adjusted p-values.
    conclusions = []
    for h in HORIZONS:
        tests = [t for t in comparisons if t["horizon"] == h]
        primary = _conclusion(tests, coverage_gate[h])
        primary["horizon"] = h
        for lag in HAC:
            lag_tests = [t["dm"][str(lag)] for t in tests]
            if not all(test.get("status") == "tested" and test.get("n", 0) >= MIN_PAIRS for test in lag_tests):
                lag_conclusion = "inference_unavailable"
            elif all(test.get("holmSignificant", False) and test.get("meanLossDifference", 0) < 0 for test in lag_tests) and all(
                    t["candidateMetric"] < t["comparatorMetric"] for t in tests) and coverage_gate[h]:
                lag_conclusion = "improvement"
            elif any(test.get("holmSignificant", False) and test.get("meanLossDifference", 0) > 0 for test in lag_tests):
                lag_conclusion = "worsening"
            else:
                lag_conclusion = "inconclusive"
            primary[f"hac{lag}Conclusion"] = lag_conclusion
        conclusions.append(primary)

    macro = {}
    for name in ("candidate", *COMPARATORS):
        per_h = [next((x["equalHorizonMape"][name] for x in horizon_summary if x["horizon"] == h), None) for h in HORIZONS]
        macro[name] = sum(per_h) / len(HORIZONS) if all(value is not None for value in per_h) else None
    return {"schemaVersion": 1, "status": "complete", "exploratoryPostHoc": True,
        "inputSha256": input_sha256, "staticReportSha256": static_sha256,
        "protocol": {"inputSha256": INPUT_SHA256, "staticReportSha256": STATIC_REPORT_SHA256,
            "evaluationTargets": [EVAL_START.isoformat(), EVAL_END.isoformat()],
            "evaluationOriginMinimum": EVAL_START.isoformat(), "weights": {str(h): FROZEN_WEIGHTS[h] for h in HORIZONS},
            "regime": {"recentCalendarDays": RECENT_DAYS, "priorCalendarDays": PRIOR_DAYS,
                "minimumRecentObservations": MIN_RECENT, "minimumPriorObservations": MIN_PRIOR,
                "maximumLatestAgeDays": MAX_AGE_DAYS, "stableAbsoluteLogRatioMaximum": STABLE_LOG_RATIO_MAX,
                "stableWeight": STABLE_WEIGHT, "volatileWeight": VOLATILE_WEIGHT},
            "familyPerLag": 60, "bandwidthsCalendarDays": list(HAC), "minimumPairs": MIN_PAIRS,
            "alpha": ALPHA, "holmSeparatelyByLag": True,
            "limitations": "Post-hoc re-evaluation on revised actuals; exploratory only; no equivalence claim or prospective promotion."},
        "coverage": {str(h): counts[h] for h in HORIZONS}, "fallbackReasonCounts": fallback_counts,
        "pairedMetrics": summary, "horizonSummary": horizon_summary, "comparisons": comparisons,
        "multipleTesting": {f"HAC{lag}": all_tests[lag] for lag in HAC}, "conclusions": conclusions,
        "strata": strata, "equalHorizonMape": macro,
        "under10PointEstimate": {"candidate": macro["candidate"] < 10 if macro["candidate"] is not None else None,
            "eligible": all(len(rows_by_horizon[h]) >= MIN_PAIRS for h in HORIZONS),
            "reason": None if all(len(rows_by_horizon[h]) >= MIN_PAIRS for h in HORIZONS) else "fewer_than_30_pairs_at_one_or_more_horizons"}}


def run(input_path=INPUT_PATH, static_report_path=STATIC_REPORT_PATH):
    source_path = Path(input_path).resolve()
    report_path = Path(static_report_path).resolve()
    source_bytes = source_path.read_bytes()
    report_bytes = report_path.read_bytes()
    source_sha = _sha(source_bytes)
    report_sha = _sha(report_bytes)
    if source_sha != INPUT_SHA256:
        raise ValueError("pinned predictions.json SHA-256 mismatch")
    if report_sha != STATIC_REPORT_SHA256:
        raise ValueError("pinned static-blend report SHA-256 mismatch")
    source = json.loads(source_bytes)
    report = json.loads(report_bytes)
    if report.get("schemaVersion") != 1 or report.get("inputSha256") != INPUT_SHA256:
        raise ValueError("pinned static-blend report metadata does not match predictions input")
    selection = garlic_prospective._validate_weights(report, report_sha)
    result = build_assessment(source, selection, input_sha256=source_sha, static_sha256=report_sha)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH)
    parser.add_argument("--static-report", type=Path, default=STATIC_REPORT_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output in (Path(args.input).resolve(), Path(args.static_report).resolve()):
        raise ValueError("output must not overwrite either pinned input")
    result = run(args.input, args.static_report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": result["status"], "inputSha256": result["inputSha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
