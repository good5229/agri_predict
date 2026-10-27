"""Frozen direct-per-horizon Ridge experiment on dashboard price exports.

This module has no API or database access. It consumes frozen prices and forecast
JSON files and writes a standalone result bundle.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import dashboard, modeling

ROOT = Path(__file__).resolve().parents[1]
INPUT_PRICES = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
INPUT_PREDICTIONS = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/predictions.json"
PRICES_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
PREDICTIONS_SHA256 = "c41ed5d8799306f151b1c5135399a606982f12623bf709a4438969d8cc389113"
EVAL_START = dt.date(2023, 1, 1)
EVAL_END = dt.date(2026, 9, 27)
HORIZONS = tuple(range(1, 11))
MIN_TRAIN_DAYS = 90
HAC_BANDWIDTHS = (9, 28)
ALPHA = 0.05
COMPARATORS = ("monthly_pooled_ridge", "persistence")
LOSSES = ("mae", "mape")
FAMILY_SIZE = 5 * 10 * 2 * 2


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _observations(raw: dict[str, Any]) -> dict[str, dict[dt.date, tuple[float, float]]]:
    if raw.get("schemaVersion") != 1 or not isinstance(raw.get("products"), list):
        raise ValueError("prices must use dashboard schemaVersion 1")
    result = {}
    for product in raw["products"]:
        pid = product.get("id")
        if pid not in dashboard.PRODUCTS or pid in result:
            raise ValueError("unknown or duplicate product in prices")
        result[pid] = modeling._observed_series(product)
    if set(result) != set(dashboard.PRODUCTS):
        raise ValueError("prices must contain every dashboard product")
    return result


def _forecast_maps(raw: dict[str, Any]) -> tuple[dict[str, dict[tuple[dt.date, int], float]], dict[str, dict[tuple[dt.date, int], float]], dict[str, dict[tuple[dt.date, int], float]]]:
    if not isinstance(raw, dict) or raw.get("schemaVersion") != 2 or not isinstance(raw.get("products"), list):
        raise ValueError("predictions must use frozen dashboard schemaVersion 2 with a products list")
    pooled: dict[str, dict[tuple[dt.date, int], float]] = {}
    daily: dict[str, dict[tuple[dt.date, int], float]] = {}
    persistence: dict[str, dict[tuple[dt.date, int], float]] = {}
    for product in raw["products"]:
        if not isinstance(product, dict):
            raise ValueError("each prediction product must be an object")
        pid = product.get("id")
        if pid not in dashboard.PRODUCTS or pid in pooled:
            raise ValueError("unknown or duplicate product in predictions")
        runs = product.get("runs")
        if not isinstance(runs, list):
            raise ValueError(f"prediction product {pid} must contain a runs list")
        baseline_rows = product.get("predictions")
        if not isinstance(baseline_rows, list):
            raise ValueError(f"prediction product {pid} must contain top-level persistence predictions")
        pooled[pid] = {}
        daily[pid] = {}
        persistence[pid] = {}
        for row in baseline_rows:
            if not isinstance(row, dict):
                raise ValueError(f"top-level persistence prediction rows for {pid} must be objects")
            day = modeling._parse_day(row.get("target"))
            horizon = row.get("horizon")
            origin = modeling._parse_day(row.get("origin"))
            if day is None or isinstance(horizon, bool) or horizon not in HORIZONS:
                raise ValueError(f"invalid persistence target or horizon for {pid}")
            if origin != day - dt.timedelta(days=horizon):
                raise ValueError(f"persistence forecast origin mismatch for {pid}/{day}/h{horizon}")
            try:
                pred = float(row.get("predicted"))
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid persistence prediction for {pid}/{day}/h{horizon}") from exc
            if not math.isfinite(pred) or pred <= 0:
                raise ValueError(f"persistence prediction must be positive and finite for {pid}/{day}/h{horizon}")
            key = (day, horizon)
            if key in persistence[pid]:
                raise ValueError(f"duplicate persistence prediction for {pid}/{day}/h{horizon}")
            persistence[pid][key] = pred
        for run in runs:
            if not isinstance(run, dict):
                raise ValueError(f"prediction runs for {pid} must be objects")
            if run.get("modelId") not in ("ridge", "baseline"):
                continue
            cadence_kind = ("daily_pooled_ridge" if run.get("cadence") == "daily" else "monthly_pooled_ridge" if run.get("cadence") == "monthly" else None) if run.get("modelId") == "ridge" else None
            target_map = daily[pid] if cadence_kind == "daily_pooled_ridge" else pooled[pid] if cadence_kind == "monthly_pooled_ridge" else None
            if target_map is None:
                continue
            predictions = run.get("predictions")
            if not isinstance(predictions, list):
                raise ValueError(f"selected {run.get('modelId')} run for {pid} must contain a predictions list")
            for row in predictions:
                if not isinstance(row, dict):
                    raise ValueError(f"prediction rows for {pid} must be objects")
                day = modeling._parse_day(row.get("target")); horizon = row.get("horizon")
                origin = modeling._parse_day(row.get("origin"))
                if day is None or isinstance(horizon, bool) or horizon not in HORIZONS:
                    continue
                if origin != day - dt.timedelta(days=horizon):
                    raise ValueError(f"forecast origin mismatch for {pid}/{day}/h{horizon}")
                try: pred = float(row.get("predicted"))
                except (TypeError, ValueError, OverflowError): continue
                if not math.isfinite(pred) or pred <= 0:
                    continue
                target_map[(day, horizon)] = pred
    if set(pooled) != set(dashboard.PRODUCTS):
        raise ValueError("predictions must contain every dashboard product")
    return pooled, daily, persistence


def _normal_two_sided(stat: float) -> float:
    return math.erfc(abs(stat) / math.sqrt(2.0))


def _dm(diff: dict[dt.date, float], bandwidth: int) -> dict[str, Any]:
    """Two-sided DM test with Bartlett HAC covariance on actual calendar lags."""
    n = len(diff)
    if n < 30:
        return {"status": "insufficient_pairs", "n": n, "meanLossDifference": None, "statistic": None, "pValue": 1.0}
    mean = sum(diff.values()) / n
    centered = {day: value - mean for day, value in diff.items()}
    dates = sorted(centered)
    variance_sum = sum(value * value for value in centered.values())
    for lag in range(1, bandwidth + 1):
        cross = 0.0
        for day in dates:
            other = day + dt.timedelta(days=lag)
            if other in centered:
                cross += centered[day] * centered[other]
        variance_sum += 2.0 * (1.0 - lag / (bandwidth + 1.0)) * cross
    variance_mean = variance_sum / (n * n)
    if not math.isfinite(variance_mean) or variance_mean <= 0:
        return {"status": "zero_or_invalid_hac_variance", "n": n, "meanLossDifference": mean, "statistic": None, "pValue": 1.0}
    stat = mean / math.sqrt(variance_mean)
    return {"status": "tested", "n": n, "meanLossDifference": mean, "statistic": stat, "pValue": _normal_two_sided(stat)}


def _holm(rows: list[dict[str, Any]], p_field: str = "pValue") -> None:
    ordered = sorted(range(len(rows)), key=lambda i: rows[i][p_field])
    running = 0.0
    m = len(rows)
    for rank, idx in enumerate(ordered):
        adjusted = min(1.0, (m - rank) * rows[idx][p_field])
        running = max(running, adjusted)
        rows[idx]["holmPValue"] = running
        rows[idx]["holmSignificant"] = running < ALPHA


def _metrics(actual: list[float], pred: list[float]) -> dict[str, float]:
    n = len(actual)
    if not n:
        return {"mae": None, "mape": None}
    return {"mae": sum(abs(y - p) for y, p in zip(actual, pred)) / n,
            "mape": 100.0 * sum(abs(y - p) / y for y, p in zip(actual, pred)) / n}


def _joint_conclusions(summaries: list[dict[str, Any]], coverage: dict[str, Any]) -> list[dict[str, Any]]:
    """Apply the frozen joint gate to each product/horizon using the HAC9 family."""
    indexed = {(r["product"], r["horizon"], r["comparator"], r["loss"]): r for r in summaries}
    conclusions = []
    for product in dashboard.PRODUCTS:
        for horizon in HORIZONS:
            key = f"{product}/{horizon}"
            cov = coverage[key]
            required = [indexed[(product, horizon, comparator, loss)]
                        for comparator in COMPARATORS for loss in LOSSES]
            tests = [r["dm"]["9"] for r in required]
            inference_available = all(t["status"] == "tested" and t["n"] >= 30 for t in tests)
            all_losses_lower = all(r["directMetric"] is not None and r["comparatorMetric"] is not None
                                   and r["directMetric"] < r["comparatorMetric"] for r in required)
            all_tests_significant = all(t["holmSignificant"] and t["meanLossDifference"] is not None
                                        and t["meanLossDifference"] < 0 for t in tests)
            coverage_pass = all(not cov["candidateCoverageLossDates"][comparator]
                                for comparator in COMPARATORS)
            adverse_evidence = any(
                (r["directMetric"] is not None and r["comparatorMetric"] is not None
                 and r["directMetric"] > r["comparatorMetric"])
                or (t["status"] == "tested" and t["holmSignificant"]
                    and t["meanLossDifference"] is not None and t["meanLossDifference"] > 0)
                for r, t in zip(required, tests))
            if not inference_available:
                conclusion = "inference_unavailable"
            elif all_losses_lower and all_tests_significant and coverage_pass:
                conclusion = "improvement"
            elif adverse_evidence:
                conclusion = "worsening_or_mixed"
            else:
                conclusion = "inconclusive"
            conclusions.append({"product": product, "horizon": horizon, "conclusion": conclusion,
                "inferenceAvailable": inference_available, "allFourHac9HolmTestsImprove": all_tests_significant,
                "allFourPairedMetricsImprove": all_losses_lower, "coverageNondeclining": coverage_pass,
                "candidateCoverageLossDates": cov["candidateCoverageLossDates"],
                "requiredTests": [{"comparator": r["comparator"], "loss": r["loss"], "hac9": r["dm"]["9"]}
                    for r in required]})
    return conclusions


def build(prices: dict[str, Any], forecasts: dict[str, Any], *, hashes: dict[str, str] | None = None) -> dict[str, Any]:
    series_by_product = _observations(prices)
    pooled, daily, persistence_maps = _forecast_maps(forecasts)
    prediction_rows: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    fit_audits: dict[str, dict[str, Any]] = {}
    candidate_status: dict[tuple[str, int, dt.date], str] = {}
    for product in dashboard.PRODUCTS:
        series = series_by_product[product]
        log_cache: dict[Any, Any] = {}
        for horizon in HORIZONS:
            key = f"{product}/{horizon}"
            # One cached sample row per historical target. It is shared across monthly fits.
            rows = []
            for target, (actual, _) in series.items():
                origin = target - dt.timedelta(days=horizon)
                if origin < dt.date(2014, 1, 1) or origin > EVAL_END:
                    continue
                feature = modeling._feature_row(series, origin, target, horizon, log_cache)
                if feature is None:
                    continue
                source_day, vector = feature
                rows.append((target, origin, source_day, actual, vector,
                             math.log(actual) - math.log(series[source_day][0])))
            rows.sort(key=lambda r: (r[1], r[0]))
            # Refit at the first eligible origin in every origin-calendar month,
            # including months whose first origin cannot form features.
            eligible = [(target, target - dt.timedelta(days=horizon)) for target in series
                        if EVAL_START <= target <= EVAL_END
                        and target - dt.timedelta(days=horizon) >= EVAL_START]
            by_month: dict[tuple[int, int], list[tuple[dt.date, dt.date]]] = defaultdict(list)
            for target, origin in eligible:
                by_month[(origin.year, origin.month)].append((target, origin))
            product_fit_audit = {"fitAttempts": 0, "fitSuccesses": 0, "months": [],
                                 "insufficientTrainingMonths": [], "insufficientTrainingOrigins": []}
            for month_key in sorted(by_month):
                eligible_month = by_month[month_key]
                fit_origin = min(origin for _, origin in eligible_month)
                month_rows = [(r[0], r[1], r[2], r[3], r[4]) for r in rows
                              if r[1].year == month_key[0] and r[1].month == month_key[1]
                              and EVAL_START <= r[0] <= EVAL_END and r[1] >= EVAL_START]
                training = [r for r in rows if r[0] <= fit_origin]
                distinct_targets = len({r[0] for r in training})
                fit_entry = {"originMonth": f"{month_key[0]:04d}-{month_key[1]:02d}",
                    "fitOrigin": fit_origin.isoformat(),
                    "eligibleOrigins": sorted({origin.isoformat() for _, origin in eligible_month}),
                    "eligibleTargets": sorted({target.isoformat() for target, _ in eligible_month}),
                    "fitDistinctTargets": distinct_targets, "fitSamples": len(training),
                    "fitAttempted": False, "fitSuccessful": False, "status": None}
                if distinct_targets < MIN_TRAIN_DAYS:
                    fit_entry["status"] = "insufficient_training"
                    product_fit_audit["insufficientTrainingMonths"].append(fit_entry["originMonth"])
                    product_fit_audit["insufficientTrainingOrigins"].extend(fit_entry["eligibleOrigins"])
                    product_fit_audit["months"].append(fit_entry)
                    for target, _ in eligible_month:
                        candidate_status[(product, horizon, target)] = "insufficient_training"
                    continue
                product_fit_audit["fitAttempts"] += 1
                fit_entry["fitAttempted"] = True
                try:
                    import numpy as np
                    from threadpoolctl import threadpool_limits
                except ImportError as exc:
                    raise RuntimeError("numpy and threadpoolctl are required") from exc
                estimator = modeling._new_estimator("ridge")
                x_train = np.asarray([r[4] for r in training], dtype=float)
                y_train = np.asarray([r[5] for r in training], dtype=float)
                try:
                    with threadpool_limits(limits=1):
                        estimator.fit(x_train, y_train)
                        predictions_for_month = []
                        for target, origin, source_day, actual, vector in month_rows:
                            predicted, clipped = modeling._inverse_log_ratio(series[source_day][0], float(estimator.predict(np.asarray([vector], dtype=float))[0]))
                            predictions_for_month.append({"product": product, "target": target.isoformat(), "origin": origin.isoformat(),
                                "horizon": horizon, "sourceDate": source_day.isoformat(), "actual": actual,
                                "predicted": predicted, "fitOrigin": fit_origin.isoformat(), "fitTrainTargetEnd": max(r[0] for r in training).isoformat(),
                                "fitDistinctTargets": distinct_targets, "fitSamples": len(training), "clipped": clipped})
                    prediction_rows.extend(predictions_for_month)
                    product_fit_audit["fitSuccesses"] += 1
                    fit_entry["fitSuccessful"] = True
                    fit_entry["status"] = "fitted"
                    for target, _ in eligible_month:
                        candidate_status[(product, horizon, target)] = (
                            "candidate_available" if any(r["target"] == target.isoformat() for r in predictions_for_month)
                            else "feature_unavailable_or_source_too_old")
                except Exception as exc:
                    fit_entry["status"] = "fit_failed"
                    fit_entry["errorType"] = type(exc).__name__
                    for target, _ in eligible_month:
                        candidate_status[(product, horizon, target)] = "fit_failed"
                product_fit_audit["months"].append(fit_entry)
            fit_audits[key] = product_fit_audit

    pred_map = {(r["product"], dt.date.fromisoformat(r["target"]), r["horizon"]): r for r in prediction_rows}
    test_rows = []
    coverage: dict[str, dict[str, Any]] = {}
    for product in dashboard.PRODUCTS:
        series = series_by_product[product]
        for horizon in HORIZONS:
            audit_key = f"{product}/{horizon}"
            target_window_dates = [EVAL_START + dt.timedelta(days=i)
                for i in range((EVAL_END - EVAL_START).days + 1)
                if EVAL_START + dt.timedelta(days=i) - dt.timedelta(days=horizon) >= EVAL_START]
            eligible_dates = sorted(target for target in series if EVAL_START <= target <= EVAL_END
                                    and target - dt.timedelta(days=horizon) >= EVAL_START)
            eligible_strings = [day.isoformat() for day in eligible_dates]
            actual_missing_dates = sorted(day.isoformat() for day in set(target_window_dates) - set(eligible_dates))
            direct_dates = sorted(day for (pid, day, h) in pred_map if pid == product and h == horizon
                                  and day in set(eligible_dates))
            monthly_dates = sorted(day for day in eligible_dates if (day, horizon) in pooled[product])
            persistence_dates = sorted(day for day in eligible_dates if (day, horizon) in persistence_maps[product])
            candidates = []
            exclusion_reasons: dict[str, list[str]] = {}
            for target, (actual, _) in series.items():
                if not EVAL_START <= target <= EVAL_END or target - dt.timedelta(days=horizon) < EVAL_START:
                    continue
                key = (target, horizon)
                cand = pred_map.get((product, target, horizon))
                baseline = pooled[product].get(key)
                frozen_persistence = persistence_maps[product].get(key)
                day_direct = daily[product].get(key)
                source_day = max((d for d in series if d <= target - dt.timedelta(days=horizon)), default=None)
                reasons = []
                if cand is None:
                    reasons.append(candidate_status.get((product, horizon, target), "candidate_unavailable"))
                if baseline is None:
                    reasons.append("monthly_control_unavailable")
                if frozen_persistence is None:
                    reasons.append("persistence_unavailable")
                stale_source = source_day is None or (target - dt.timedelta(days=horizon) - source_day).days > dashboard.MAX_SOURCE_AGE_DAYS
                if stale_source:
                    reasons.append("source_older_than_max_age")
                if cand is not None and baseline is not None and frozen_persistence is not None and source_day is not None:
                    if not all(math.isfinite(v) and v > 0 for v in (actual, cand["predicted"], baseline, frozen_persistence)):
                        reasons.append("nonpositive_or_nonfinite_paired_value")
                if reasons:
                    exclusion_reasons[target.isoformat()] = reasons
                    continue
                persistence = frozen_persistence
                candidates.append((target, actual, cand["predicted"], baseline, persistence, day_direct))
            # Common actual/date sample for the direct candidate and both primary comparators.
            for row in candidates:
                target, actual, direct, monthly, persistence, direct_day = row
                test_rows.append({"product": product, "target": target.isoformat(), "horizon": horizon,
                    "actual": actual, "direct": direct, "monthlyPooledRidge": monthly,
                    "persistence": persistence, "dailyPooledRidge": direct_day})
            for comparator in COMPARATORS:
                for loss in LOSSES:
                    ds = {}
                    actuals = []; direct_preds = []; comparator_preds = []
                    for target, actual, direct, monthly, persistence, _ in candidates:
                        other = monthly if comparator == "monthly_pooled_ridge" else persistence
                        d = abs(actual - direct) - abs(actual - other) if loss == "mae" else 100.0 * (abs(actual - direct) / actual - abs(actual - other) / actual)
                        ds[target] = d
                        actuals.append(actual); direct_preds.append(direct); comparator_preds.append(other)
                    test_rows.append({"_summary": True, "product": product, "horizon": horizon, "comparator": comparator,
                        "loss": loss, "n": len(ds), "directMetric": _metrics(actuals, direct_preds)[loss],
                        "comparatorMetric": _metrics(actuals, comparator_preds)[loss],
                        "dailyPooledRidgeDescriptive": _metrics(
                            [r[1] for r in candidates if r[5] is not None],
                            [r[5] for r in candidates if r[5] is not None])[loss],
                        "dailyPooledRidgeCoverage": sum(r[5] is not None for r in candidates),
                        "meanLossDifference": sum(ds.values()) / len(ds) if ds else None,
                        "dm": {str(bw): _dm(ds, bw) for bw in HAC_BANDWIDTHS}})
            common_strings = sorted(row[0].isoformat() for row in candidates)
            comparator_dates = {"monthly_pooled_ridge": [d.isoformat() for d in monthly_dates],
                                "persistence": [d.isoformat() for d in persistence_dates]}
            direct_set = set(d.isoformat() for d in direct_dates)
            coverage[audit_key] = {"targetWindowDates": [day.isoformat() for day in target_window_dates],
                "eligibleEvaluationDates": eligible_strings, "noActualObservationDates": actual_missing_dates,
                "candidateAvailableDates": sorted(direct_set), "comparatorAvailableDates": comparator_dates,
                "commonPairedDates": common_strings,
                "candidateCoverageLossDates": {comp: sorted(set(dates) - direct_set)
                    for comp, dates in comparator_dates.items()},
                "exclusionReasons": exclusion_reasons,
                "eligibleDateCount": len(eligible_strings), "candidateAvailableCount": len(direct_set),
                "monthlyControlAvailableCount": len(monthly_dates), "persistenceAvailableCount": len(persistence_dates),
                "commonPairedCount": len(common_strings)}
    for comparator in COMPARATORS:
        comparator_coverage = sum(len(row["comparatorAvailableDates"][comparator]) for row in coverage.values())
        comparator_pairs = sum(row["n"] for row in test_rows
            if row.get("_summary") and row["comparator"] == comparator and row["loss"] == "mae")
        if comparator_coverage == 0 or comparator_pairs == 0:
            raise ValueError(f"frozen comparator {comparator} has zero global coverage or common paired dates")
    for bandwidth in HAC_BANDWIDTHS:
        family = [r["dm"][str(bandwidth)] for r in test_rows if r.get("_summary")]
        if len(family) != FAMILY_SIZE:
            raise AssertionError(f"fixed Holm family expected {FAMILY_SIZE}, got {len(family)}")
        _holm(family)
    for row in test_rows:
        if row.get("_summary"):
            for bw in HAC_BANDWIDTHS:
                item = row["dm"][str(bw)]
                mean_diff = item.get("meanLossDifference")
                item["conclusion"] = ("improvement" if mean_diff is not None and mean_diff < 0 and item["holmSignificant"]
                    else "worsening" if mean_diff is not None and mean_diff > 0 and item["holmSignificant"]
                    else "inconclusive")
    # Seasonal/year coverage and metrics are descriptive; no extra tests.
    strata = []
    for product in dashboard.PRODUCTS:
        for horizon in HORIZONS:
            rows = [r for r in test_rows if not r.get("_summary") and r["product"] == product and r["horizon"] == horizon]
            for field, key_fn in (("year", lambda d: d.year), ("season", lambda d: (d.month - 1) // 3 + 1)):
                groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
                for row in rows: groups[key_fn(dt.date.fromisoformat(row["target"]))].append(row)
                for key, group in sorted(groups.items()):
                    actual = [r["actual"] for r in group]
                    strata.append({"product": product, "horizon": horizon, "stratum": field, "value": key, "n": len(group),
                        "direct": _metrics(actual, [r["direct"] for r in group]),
                        "monthlyPooledRidge": _metrics(actual, [r["monthlyPooledRidge"] for r in group]),
                        "persistence": _metrics(actual, [r["persistence"] for r in group])})
    summaries = [r for r in test_rows if r.get("_summary")]
    joint = _joint_conclusions(summaries, coverage)
    return {"protocol": {"id": "direct-horizon-ridge-2026-10-06", "evaluation": [EVAL_START.isoformat(), EVAL_END.isoformat()],
            "horizons": list(HORIZONS), "monthlyRefit": "first evaluation origin in each origin-calendar month; evaluation origin must be 2023-01-01 or later",
                "minimumDistinctTrainingTargets": MIN_TRAIN_DAYS, "trainTargetsAtOrBeforeFitOrigin": True,
                "maxSourceAgeDays": dashboard.MAX_SOURCE_AGE_DAYS, "model": {"pipeline": ["StandardScaler", "Ridge"], "alpha": 10.0},
                "holmFamilySizePerBandwidth": FAMILY_SIZE, "bandwidthsCalendarDays": list(HAC_BANDWIDTHS), "alpha": ALPHA,
                "comparisons": list(COMPARATORS), "losses": list(LOSSES), "dailyPooledRidge": "descriptive-only; no inference",
                "limitations": "Revised historical prices; first publication times unknown. Retrospective study, not operational or contest proof."},
            "hashes": hashes or {}, "coverage": coverage,
            "fitAudits": fit_audits,
            "fitTotals": {"fitAttempts": sum(a["fitAttempts"] for a in fit_audits.values()),
                          "fitSuccesses": sum(a["fitSuccesses"] for a in fit_audits.values()),
                          "insufficientTrainingMonths": sum(len(a["insufficientTrainingMonths"]) for a in fit_audits.values()),
                          "insufficientTrainingOrigins": sum(len(a["insufficientTrainingOrigins"]) for a in fit_audits.values())},
            "jointConclusions": joint,
            "predictions": prediction_rows, "pairedRows": [r for r in test_rows if not r.get("_summary")], "comparisons": summaries, "strata": strata}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=INPUT_PRICES)
    parser.add_argument("--predictions", type=Path, default=INPUT_PREDICTIONS)
    parser.add_argument("--output", type=Path, required=True, help="new output JSON path; no input is overwritten")
    parser.add_argument("--allow-nonfrozen-inputs", action="store_true", help="permit hashes other than the frozen defaults")
    args = parser.parse_args(argv)
    if args.output.resolve() in (args.prices.resolve(), args.predictions.resolve()):
        parser.error("output must not replace an input")
    hashes = {"pricesSha256": _sha(args.prices), "predictionsSha256": _sha(args.predictions)}
    if not args.allow_nonfrozen_inputs and (hashes["pricesSha256"] != PRICES_SHA256 or hashes["predictionsSha256"] != PREDICTIONS_SHA256):
        parser.error("input SHA-256 does not match frozen protocol")
    result = build(_read(args.prices), _read(args.predictions), hashes=hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(args.output)
    print(json.dumps({"output": str(args.output), "hashes": hashes, "coverage": result["coverage"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
