"""Frozen offline 730-calendar-day training-window comparison (no API access)."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import dashboard, direct_horizon as dh, direct_tree, modeling

ROOT = Path(__file__).resolve().parents[1]
INPUT_PRICES = dh.INPUT_PRICES
INPUT_PREDICTIONS = dh.INPUT_PREDICTIONS
INPUT_RIDGE_REPORT = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
PRICES_SHA256 = dh.PRICES_SHA256
PREDICTIONS_SHA256 = dh.PREDICTIONS_SHA256
RIDGE_REPORT_SHA256 = direct_tree.RIDGE_REPORT_SHA256
PRODUCTS = tuple(dashboard.PRODUCTS)
HORIZONS = tuple(range(1, 11))
EVAL_START, EVAL_END = dh.EVAL_START, dh.EVAL_END
MIN_TRAIN_DAYS = 90
WINDOW_DAYS = 730
COMPARATORS = ("full_history_direct_hist_gb", "monthly_pooled_hist_gb", "direct_ridge", "persistence")
LOSSES = ("mae", "mape")
BANDWIDTHS = (9, 28)
FAMILY_SIZE = len(PRODUCTS) * len(HORIZONS) * len(COMPARATORS) * len(LOSSES)
FIELDS = {"full_history_direct_hist_gb": "fullHistoryHistGb", "monthly_pooled_hist_gb": "monthlyPooledHistGb",
          "direct_ridge": "directRidge", "persistence": "persistence"}


def _direct_ridge_reference(report: dict[str, Any]):
    if report.get("protocol", {}).get("id") != "direct-horizon-ridge-2026-10-06":
        raise ValueError("unexpected direct-horizon Ridge report")
    predictions, actuals = {}, {}
    for row in report.get("pairedRows", []):
        key = (row["product"], int(row["horizon"]), dt.date.fromisoformat(row["target"]))
        if key in predictions: raise ValueError(f"duplicate direct Ridge row for {key}")
        predictions[key] = float(row["direct"])
        actuals[key] = float(row["actual"])
    return predictions, actuals


def _monthly_hist_gb(raw):
    if raw.get("schemaVersion") != 2 or not isinstance(raw.get("products"), list):
        raise ValueError("predictions must use dashboard schemaVersion 2")
    result = {}
    for product in raw["products"]:
        pid = product.get("id")
        if pid not in PRODUCTS or pid in result:
            raise ValueError("unknown or duplicate prediction product")
        runs = [r for r in product.get("runs", []) if r.get("modelId") == "hist_gb" and r.get("cadence") == "monthly"]
        if len(runs) != 1:
            raise ValueError(f"expected one monthly pooled HistGB run for {pid}")
        values = {}
        for row in runs[0].get("predictions", []):
            day = modeling._parse_day(row.get("target")); origin = modeling._parse_day(row.get("origin")); h = row.get("horizon")
            if day is None or isinstance(h, bool) or h not in HORIZONS:
                continue
            if origin != day - dt.timedelta(days=h):
                raise ValueError(f"monthly HistGB origin mismatch for {pid}/{day}/h{h}")
            try: pred = float(row.get("predicted"))
            except (TypeError, ValueError, OverflowError): continue
            if not math.isfinite(pred) or pred <= 0: continue
            if (day, h) in values: raise ValueError(f"duplicate monthly HistGB prediction for {pid}/{day}/h{h}")
            values[(day, h)] = pred
        result[pid] = values
    if set(result) != set(PRODUCTS): raise ValueError("monthly HistGB predictions must cover every product")
    return result


def _metrics(actual, pred): return dh._metrics(actual, pred)
def _dm(diff, bandwidth): return dh._dm(diff, bandwidth)


def _training_rows(samples, fit_origin, *, recent):
    """Select target rows under the frozen strict cutoff and optional 730-day window."""
    start = fit_origin - dt.timedelta(days=WINDOW_DAYS) if recent else None
    return [row for row in samples if row[0] < fit_origin and (start is None or row[0] >= start)]


def _coverage_loss(control_dates, candidate_dates):
    return [value.isoformat() if isinstance(value, dt.date) else value
            for value in sorted(set(control_dates) - set(candidate_dates))]


def _macro_metrics(rows):
    fields = ("recentWindowHistGb", "fullHistoryHistGb", "monthlyPooledHistGb", "directRidge", "persistence")
    metrics, counts = {}, {}
    for field in fields:
        groups = []
        for product in PRODUCTS:
            for horizon in HORIZONS:
                subset = [r for r in rows if r["product"] == product and r["horizon"] == horizon]
                groups.append(_metrics([r["actual"] for r in subset], [r[field] for r in subset])["mape"])
        counts[field] = sum(x is not None for x in groups)
        metrics[field] = sum(groups) / 50 if len(groups) == 50 and all(x is not None for x in groups) else None
    return metrics, counts


def _joint_conclusions(comparisons, coverage):
    indexed = {(r["product"], r["horizon"], r["comparator"], r["loss"]): r for r in comparisons}
    conclusions = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            cov = coverage[f"{product}/{horizon}"]
            required = [indexed[(product, horizon, c, loss)] for c in COMPARATORS for loss in LOSSES]
            tests = [r["dm"]["9"] for r in required]
            enough = all(t["status"] == "tested" and t["n"] >= 30 for t in tests)
            lower = all(r["candidateMetric"] is not None and r["comparatorMetric"] is not None and r["candidateMetric"] < r["comparatorMetric"] for r in required)
            sig = all(t.get("holmSignificant") and t.get("meanLossDifference") is not None and t["meanLossDifference"] < 0 for t in tests)
            coverage_ok = all(not cov["candidateCoverageLossDates"][c] for c in COMPARATORS)
            adverse = any(t["status"] == "tested" and t.get("holmSignificant") and t.get("meanLossDifference") is not None and t["meanLossDifference"] > 0 for t in tests)
            status = "inference_unavailable" if not enough else "improvement" if lower and sig and coverage_ok else "worsening_or_mixed" if adverse else "inconclusive"
            conclusions.append({"product": product, "horizon": horizon, "conclusion": status, "inferenceAvailable": enough,
                "allEightHac9HolmTestsImprove": sig, "allEightPairedMetricsImprove": lower, "coverageNondeclining": coverage_ok,
                "requiredTests": [{"comparator": r["comparator"], "loss": r["loss"], "hac9": r["dm"]["9"]} for r in required]})
    return conclusions


def build(prices, forecasts, ridge_report, *, hashes=None):
    series_by = dh._observations(prices)
    pooled, _, persistence = dh._forecast_maps(forecasts)
    pooled_hist = _monthly_hist_gb(forecasts)
    ridge, ridge_actuals = _direct_ridge_reference(ridge_report)
    predictions, fit_audits, statuses = [], {}, {}
    for product in PRODUCTS:
        series = series_by[product]
        cache = {}
        for horizon in HORIZONS:
            samples = []
            for target, (actual, _) in series.items():
                origin = target - dt.timedelta(days=horizon)
                if origin < dt.date(2014, 1, 1) or origin > EVAL_END: continue
                feature = modeling._feature_row(series, origin, target, horizon, cache)
                if feature is None: continue
                source, vector = feature
                samples.append((target, origin, source, actual, vector, math.log(actual) - math.log(series[source][0])))
            samples.sort(key=lambda r: r[0])
            sample_map = {(r[0], r[1]): r for r in samples}
            evals = [(t, t-dt.timedelta(days=horizon)) for t in series if EVAL_START <= t <= EVAL_END and t-dt.timedelta(days=horizon) >= EVAL_START]
            months = defaultdict(list)
            for target, origin in evals: months[(origin.year, origin.month)].append((target, origin))
            audit = {"fitAttempts": {"recent": 0, "full": 0}, "fitSuccesses": {"recent": 0, "full": 0}, "months": []}
            for (year, month), targets in sorted(months.items()):
                fit_origin = min(o for _, o in targets)
                full_train = _training_rows(samples, fit_origin, recent=False)
                recent_start = fit_origin - dt.timedelta(days=WINDOW_DAYS)
                recent_train = _training_rows(samples, fit_origin, recent=True)
                entry = {"originMonth": f"{year:04d}-{month:02d}", "fitOrigin": fit_origin.isoformat(),
                    "recentWindowStart": recent_start.isoformat(), "trainTargetCutoffExclusive": fit_origin.isoformat(),
                    "fullHistoryDistinctTargets": len({r[0] for r in full_train}), "fullHistorySamples": len(full_train),
                    "fullHistoryTargetStart": min((r[0] for r in full_train), default=None).isoformat() if full_train else None,
                    "fullHistoryTargetEnd": max((r[0] for r in full_train), default=None).isoformat() if full_train else None,
                    "recentWindowDistinctTargets": len({r[0] for r in recent_train}), "recentWindowSamples": len(recent_train),
                    "recentWindowTargetStart": min((r[0] for r in recent_train), default=None).isoformat() if recent_train else None,
                    "recentWindowTargetEnd": max((r[0] for r in recent_train), default=None).isoformat() if recent_train else None,
                    "arms": {}}
                valid = [sample_map[(t, o)] for t, o in targets if (t, o) in sample_map]
                for arm, train in (("full", full_train), ("recent", recent_train)):
                    distinct = len({r[0] for r in train})
                    arm_audit = {"distinctTargets": distinct, "samples": len(train), "status": None}
                    entry["arms"][arm] = arm_audit
                    if distinct < MIN_TRAIN_DAYS:
                        arm_audit["status"] = "insufficient_training"
                        for t, _ in targets: statuses[(product, horizon, t, arm)] = "insufficient_training"
                        continue
                    audit["fitAttempts"][arm] += 1
                    try:
                        import numpy as np
                        from threadpoolctl import threadpool_limits
                        with direct_tree._thread_limit_context(threadpool_limits):
                            model = modeling._new_estimator("hist_gb")
                            model.fit(np.asarray([r[4] for r in train]), np.asarray([r[5] for r in train]))
                            deltas = model.predict(np.asarray([r[4] for r in valid])) if valid else []
                        built = []
                        for sample, delta in zip(valid, deltas):
                            price, clipped = modeling._inverse_log_ratio(series[sample[2]][0], float(delta))
                            built.append({"product": product, "horizon": horizon, "target": sample[0].isoformat(),
                                "origin": sample[1].isoformat(), "actual": sample[3], "predicted": price, "clipped": clipped,
                                "fitOrigin": fit_origin.isoformat(), "trainTargetEnd": max(r[0] for r in train).isoformat(),
                                "distinctTrainingTargets": distinct, "trainingSamples": len(train), "arm": arm})
                        predictions.extend(built)
                        audit["fitSuccesses"][arm] += 1; arm_audit["status"] = "fitted"
                        for t, _ in targets: statuses[(product, horizon, t, arm)] = "candidate_available" if (t, t-dt.timedelta(days=horizon)) in sample_map else "feature_unavailable_or_source_too_old"
                    except Exception as exc:
                        arm_audit.update(status="fit_failed", errorType=type(exc).__name__, error=str(exc)[:200])
                        for t, _ in targets: statuses[(product, horizon, t, arm)] = "fit_failed"
                audit["months"].append(entry)
            fit_audits[f"{product}/{horizon}"] = audit

    predmap = {(r["product"], r["horizon"], dt.date.fromisoformat(r["target"]), r["arm"]): r for r in predictions}
    paired, comparisons, coverage, strata = [], [], {}, []
    for product in PRODUCTS:
        series = series_by[product]
        obs_days = tuple(series)
        for horizon in HORIZONS:
            eligible = sorted(t for t in series if EVAL_START <= t <= EVAL_END and t-dt.timedelta(days=horizon) >= EVAL_START)
            rows, reasons = [], {}
            control_dates = {c: set() for c in COMPARATORS}
            candidate_dates = {"recent": set(), "full": set()}
            for target in eligible:
                key = (target, horizon); origin = target-dt.timedelta(days=horizon)
                recent = predmap.get((product, horizon, target, "recent")); full = predmap.get((product, horizon, target, "full"))
                month = pooled_hist[product].get(key); ridge_pred = ridge.get((product, horizon, target)); persist = persistence[product].get(key)
                controls = {COMPARATORS[0]: full["predicted"] if full else None, COMPARATORS[1]: month,
                    COMPARATORS[2]: ridge_pred, COMPARATORS[3]: persist}
                for c, value in controls.items():
                    if value is not None: control_dates[c].add(target)
                for arm, value in (("recent", recent), ("full", full)):
                    if value: candidate_dates[arm].add(target)
                source_idx = bisect_right(obs_days, origin)-1
                source = obs_days[source_idx] if source_idx >= 0 else None
                missing = []
                if recent is None: missing.append("recent_candidate_unavailable:" + statuses.get((product,horizon,target,"recent"), "no_fit_record"))
                if full is None: missing.append("full_history_comparator_unavailable:" + statuses.get((product,horizon,target,"full"), "no_fit_record"))
                for c, value in controls.items():
                    if value is None: missing.append(c + "_unavailable")
                if source is None or (origin-source).days > dashboard.MAX_SOURCE_AGE_DAYS: missing.append("source_older_than_max_age")
                if ridge_pred is not None and not math.isclose(series[target][0], ridge_actuals[(product, horizon, target)], rel_tol=1e-12, abs_tol=0.0):
                    raise ValueError(f"direct Ridge actual mismatch for {product}/{target}/h{horizon}")
                if missing:
                    reasons[target.isoformat()] = missing; continue
                row = {"product": product, "horizon": horizon, "target": target.isoformat(), "origin": origin.isoformat(),
                    "actual": series[target][0], "recentWindowHistGb": recent["predicted"], "fullHistoryHistGb": full["predicted"],
                    "monthlyPooledHistGb": month, "directRidge": ridge_pred, "persistence": persist}
                rows.append(row)
            paired.extend(rows)
            ck = f"{product}/{horizon}"
            coverage[ck] = {"eligibleDateCount": len(eligible), "eligibleDates": [d.isoformat() for d in eligible],
                "recentCandidateAvailableCount": len(candidate_dates["recent"]), "fullComparatorAvailableCount": len(candidate_dates["full"]),
                "commonPairedCount": len(rows), "pairedDates": [r["target"] for r in rows], "exclusionReasons": reasons,
                "exclusionReasonCounts": {reason: sum(reason in values for values in reasons.values()) for reason in sorted({x for v in reasons.values() for x in v})},
                "candidateCoverageLossDates": {c: _coverage_loss(ds, candidate_dates["recent"]) for c, ds in control_dates.items()},
                "controlAvailableCounts": {c: len(ds) for c, ds in control_dates.items()}}
            for comparator in COMPARATORS:
                field = FIELDS[comparator]
                for loss in LOSSES:
                    actual = [r["actual"] for r in rows]; cand = [r["recentWindowHistGb"] for r in rows]; comp = [r[field] for r in rows]
                    diffs = {dt.date.fromisoformat(r["target"]): (abs(r["actual"]-r["recentWindowHistGb"])-abs(r["actual"]-r[field])) if loss == "mae" else 100*(abs(r["actual"]-r["recentWindowHistGb"])/r["actual"]-abs(r["actual"]-r[field])/r["actual"]) for r in rows}
                    comparisons.append({"product": product, "horizon": horizon, "comparator": comparator, "loss": loss, "n": len(rows),
                        "candidateMetric": _metrics(actual, cand)[loss], "comparatorMetric": _metrics(actual, comp)[loss],
                        "dm": {str(b): _dm(diffs, b) for b in BANDWIDTHS}})
            for label, keyfn in (("year", lambda d: d.year), ("season", lambda d: (d.month-1)//3+1)):
                groups = defaultdict(list)
                for row in rows: groups[keyfn(dt.date.fromisoformat(row["target"]))].append(row)
                for val, group in sorted(groups.items()):
                    actual = [r["actual"] for r in group]
                    strata.append({"product": product, "horizon": horizon, "stratum": label, "value": val, "n": len(group),
                        **{f: _metrics(actual, [r[f] for r in group]) for f in ("recentWindowHistGb", "fullHistoryHistGb", "monthlyPooledHistGb", "directRidge", "persistence")}})
    for bandwidth in BANDWIDTHS:
        family = [r["dm"][str(bandwidth)] for r in comparisons]
        if len(family) != FAMILY_SIZE: raise AssertionError("frozen Holm family size mismatch")
        dh._holm(family)
    macro, macro_counts = _macro_metrics(paired)
    return {"schemaVersion": 1, "protocol": {"id": "recent-730-day-direct-hist-gb-2026-10-06", "evaluation": [EVAL_START.isoformat(), EVAL_END.isoformat()],
        "products": list(PRODUCTS), "horizons": list(HORIZONS), "featureSet": "modeling._feature_row existing features only", "windowCalendarDays": WINDOW_DAYS,
        "recentTargetInterval": "fitOrigin-730 days <= target < fitOrigin", "fullHistoryTargetInterval": "all available targets < fitOrigin",
        "monthlyRefit": "first eligible evaluation origin per origin month", "minimumDistinctTrainingTargets": MIN_TRAIN_DAYS,
        "strictTargetBeforeFitOrigin": True, "maxSourceAgeDays": dashboard.MAX_SOURCE_AGE_DAYS, "estimator": "modeling._new_estimator('hist_gb')",
        "comparators": list(COMPARATORS), "losses": list(LOSSES), "dm": {"calendarBartlettHACDays": list(BANDWIDTHS), "twoSided": True,
        "holmAlpha": .05, "familySizePerBandwidth": FAMILY_SIZE}, "macroMAPE": "equal-weight mean of 50 product/horizon groups, null unless all groups are available"},
        "hashes": hashes or {}, "fitAudits": fit_audits, "fitTotals": {key: {arm: sum(v[key][arm] for v in fit_audits.values()) for arm in ("recent", "full")} for key in ("fitAttempts", "fitSuccesses")},
        "coverage": coverage, "macroMetrics": macro, "macroMetricGroupCounts": macro_counts, "jointConclusions": _joint_conclusions(comparisons, coverage),
        "limitations": "Retrospective use of revised prices; historical first publication times are unknown. This is not prospective or official-target evidence.",
        "predictions": predictions, "pairedRows": paired, "comparisons": comparisons, "strata": strata}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=INPUT_PRICES); parser.add_argument("--predictions", type=Path, default=INPUT_PREDICTIONS)
    parser.add_argument("--ridge-report", type=Path, default=INPUT_RIDGE_REPORT); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-nonfrozen-inputs", action="store_true"); args = parser.parse_args(argv)
    if args.output.resolve() in (args.prices.resolve(), args.predictions.resolve(), args.ridge_report.resolve()): parser.error("output must not replace input")
    hashes = {"pricesSha256": dh._sha(args.prices), "predictionsSha256": dh._sha(args.predictions), "directRidgeReportSha256": dh._sha(args.ridge_report)}
    expected = {"pricesSha256": PRICES_SHA256, "predictionsSha256": PREDICTIONS_SHA256, "directRidgeReportSha256": RIDGE_REPORT_SHA256}
    if not args.allow_nonfrozen_inputs and hashes != expected: parser.error("input SHA-256 does not match frozen protocol")
    result = build(dh._read(args.prices), dh._read(args.predictions), dh._read(args.ridge_report), hashes=hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True); tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"); tmp.replace(args.output)
    return 0


if __name__ == "__main__": raise SystemExit(main())
