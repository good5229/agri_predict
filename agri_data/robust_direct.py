"""Frozen retrospective Huber-vs-direct Ridge experiment; no API access."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import HuberRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from . import dashboard, direct_horizon, modeling

ROOT = Path(__file__).resolve().parents[1]
INPUT_PRICES = direct_horizon.INPUT_PRICES
INPUT_REPORT = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
PRICES_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
REPORT_SHA256 = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
HORIZONS = (1, 5, 10)
EVAL_START, EVAL_END = direct_horizon.EVAL_START, direct_horizon.EVAL_END
MIN_TRAIN_DAYS = 90
COMPARATORS = ("direct_ridge", "monthly_pooled_ridge", "persistence")
LOSSES = ("mae", "mape")
HAC = (9, 28)
FAMILY_SIZE = len(dashboard.PRODUCTS) * len(HORIZONS) * len(COMPARATORS) * len(LOSSES)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dm(diff: dict[dt.date, float], bandwidth: int) -> dict[str, Any]:
    return direct_horizon._dm(diff, bandwidth)


def holm(rows: list[dict[str, Any]]) -> None:
    direct_horizon._holm(rows)


def metrics(actual: list[float], pred: list[float]) -> dict[str, float | None]:
    return direct_horizon._metrics(actual, pred)


def _series(prices: dict[str, Any]):
    return direct_horizon._observations(prices)


def _reference(report: dict[str, Any]):
    if report.get("protocol", {}).get("id") != "direct-horizon-ridge-2026-10-06":
        raise ValueError("unexpected frozen direct-horizon report")
    paired = {}
    for row in report.get("pairedRows", []):
        key = (row["product"], int(row["horizon"]), dt.date.fromisoformat(row["target"]))
        paired[key] = {"actual": float(row["actual"]), "direct_ridge": float(row["direct"]),
                       "monthly_pooled_ridge": float(row["monthlyPooledRidge"]),
                       "persistence": float(row["persistence"])}
    report_coverage = report.get("coverage")
    if not isinstance(report_coverage, dict):
        raise ValueError("direct-horizon report is missing frozen availability coverage")
    availability = {}
    for key, row in report_coverage.items():
        if not isinstance(row, dict):
            raise ValueError(f"invalid direct-horizon coverage for {key}")
        direct = row.get("candidateAvailableDates")
        controls = row.get("comparatorAvailableDates")
        if not isinstance(direct, list) or not isinstance(controls, dict):
            raise ValueError(f"incomplete direct-horizon availability coverage for {key}")
        availability[key] = {"direct_ridge": {dt.date.fromisoformat(day) for day in direct},
            "monthly_pooled_ridge": {dt.date.fromisoformat(day) for day in controls.get("monthly_pooled_ridge", [])},
            "persistence": {dt.date.fromisoformat(day) for day in controls.get("persistence", [])}}
    return paired, availability


def _coverage_losses(control_dates: dict[str, set[dt.date]], candidate_dates: set[dt.date]) -> dict[str, int]:
    return {name: len(days - candidate_dates) for name, days in control_dates.items()}


def _new_model():
    return make_pipeline(StandardScaler(), HuberRegressor(epsilon=1.35, alpha=0.0001,
        max_iter=1000, tol=1e-5))


def build(prices: dict[str, Any], report: dict[str, Any], hashes: dict[str, str] | None = None) -> dict[str, Any]:
    series_by_product = _series(prices)
    reference, reference_availability = _reference(report)
    predictions, fits, statuses = [], {}, {}
    for product in dashboard.PRODUCTS:
        series = series_by_product[product]
        cache: dict[Any, Any] = {}
        for horizon in HORIZONS:
            key = f"{product}/{horizon}"
            rows = []
            for target, (actual, _) in series.items():
                origin = target - dt.timedelta(days=horizon)
                if origin < dt.date(2014, 1, 1) or origin > EVAL_END:
                    continue
                feat = modeling._feature_row(series, origin, target, horizon, cache)
                if feat is None:
                    continue
                source, vector = feat
                rows.append((target, origin, source, actual, vector,
                             math.log(actual) - math.log(series[source][0])))
            rows.sort(key=lambda x: (x[1], x[0]))
            months: dict[tuple[int, int], list[tuple[dt.date, dt.date]]] = defaultdict(list)
            for target in series:
                origin = target - dt.timedelta(days=horizon)
                if EVAL_START <= target <= EVAL_END and origin >= EVAL_START:
                    months[(origin.year, origin.month)].append((target, origin))
            audit = {"fitAttempts": 0, "fitSuccesses": 0, "fitFailures": 0,
                     "insufficientMonths": 0, "months": []}
            for mk, eligible in sorted(months.items()):
                fit_origin = min(o for _, o in eligible)
                train = [r for r in rows if r[0] <= fit_origin]
                ndays = len({r[0] for r in train})
                month_rows = [r for r in rows if (r[1].year, r[1].month) == mk
                              and EVAL_START <= r[0] <= EVAL_END and r[1] >= EVAL_START]
                entry = {"originMonth": f"{mk[0]:04d}-{mk[1]:02d}", "fitOrigin": fit_origin.isoformat(),
                         "trainTargetEnd": max((r[0] for r in train), default=None).isoformat() if train else None,
                         "distinctTrainTargets": ndays, "fitSamples": len(train), "status": "insufficient_training"}
                if ndays < MIN_TRAIN_DAYS:
                    audit["insufficientMonths"] += 1
                    for target, _ in eligible: statuses[(product, horizon, target)] = "insufficient_training"
                    audit["months"].append(entry); continue
                audit["fitAttempts"] += 1
                try:
                    model = _new_model()
                    with threadpool_limits(limits=1), warnings.catch_warnings():
                        warnings.simplefilter("error", ConvergenceWarning)
                        model.fit(np.asarray([r[4] for r in train]), np.asarray([r[5] for r in train]))
                        pred_rows = []
                        for target, origin, source, actual, vector, _ in month_rows:
                            ratio = float(model.predict(np.asarray([vector]))[0])
                            if not math.isfinite(ratio):
                                raise FloatingPointError("Huber prediction ratio is nonfinite")
                            predicted, clipped = modeling._inverse_log_ratio(series[source][0], ratio)
                            if not math.isfinite(predicted) or predicted <= 0:
                                raise FloatingPointError("nonpositive or nonfinite prediction")
                            pred_rows.append({"product": product, "horizon": horizon, "target": target.isoformat(),
                                "origin": origin.isoformat(), "actual": actual, "predicted": predicted,
                                "sourceDate": source.isoformat(), "fitOrigin": fit_origin.isoformat(),
                                "fitTrainTargetEnd": max(r[0] for r in train).isoformat(),
                                "fitDistinctTargets": ndays, "fitSamples": len(train), "clipped": clipped})
                        # sklearn exposes convergence through n_iter_; fail closed at the configured limit.
                        huber = model[-1]
                        if huber.n_iter_ >= 1000:
                            raise RuntimeError("HuberRegressor reached max_iter; convergence unverified")
                    predictions.extend(pred_rows); audit["fitSuccesses"] += 1; entry["status"] = "fitted"
                    for target, _ in eligible:
                        statuses[(product, horizon, target)] = "candidate_available" if any(
                            r["target"] == target.isoformat() for r in pred_rows) else "feature_unavailable_or_source_too_old"
                except Exception as exc:
                    audit["fitFailures"] += 1; entry["status"] = "fit_failed"; entry["errorType"] = type(exc).__name__; entry["error"] = str(exc)[:200]
                    for target, _ in eligible: statuses[(product, horizon, target)] = "fit_failed"
                audit["months"].append(entry)
            fits[key] = audit

    pred_map = {(r["product"], r["horizon"], dt.date.fromisoformat(r["target"])): r for r in predictions}
    coverage, paired, summaries = {}, [], []
    strata = []
    for product in dashboard.PRODUCTS:
        series = series_by_product[product]
        for horizon in HORIZONS:
            key = f"{product}/{horizon}"
            eligible = sorted(d for d in series if EVAL_START <= d <= EVAL_END and d-dt.timedelta(days=horizon) >= EVAL_START)
            rows, exclusions = [], defaultdict(int)
            for day in eligible:
                ref = reference.get((product, horizon, day)); candidate = pred_map.get((product, horizon, day))
                if not ref or not candidate:
                    reason = statuses.get((product, horizon, day), "candidate_unavailable") if not candidate else "reference_unavailable"
                    exclusions[reason] += 1; continue
                source = max((d for d in series if d <= day-dt.timedelta(days=horizon)), default=None)
                if source is None or (day-dt.timedelta(days=horizon)-source).days > dashboard.MAX_SOURCE_AGE_DAYS:
                    exclusions["source_older_than_max_age"] += 1; continue
                row = {"product": product, "horizon": horizon, "target": day.isoformat(),
                       "origin": (day-dt.timedelta(days=horizon)).isoformat(), "actual": ref["actual"],
                       "huber": candidate["predicted"], **{c: ref[c] for c in COMPARATORS}}
                paired.append(row); rows.append(row)
            control_days = reference_availability.get(key, {c: set() for c in COMPARATORS})
            cand_days = {d for (p,h,d) in pred_map if p == product and h == horizon}
            coverage[key] = {"eligibleCount": len(eligible), "candidateAvailableCount": len(cand_days),
                "commonPairedCount": len(rows), "pairedDates": [r["target"] for r in rows],
                "candidateCoverageLossCounts": _coverage_losses(control_days, cand_days),
                "exclusionCounts": dict(exclusions)}
            for comparator in COMPARATORS:
                for loss in LOSSES:
                    diffs, actuals, hp, cp = {}, [], [], []
                    for row in rows:
                        y, a, b = row["actual"], row["huber"], row[comparator]
                        diffs[dt.date.fromisoformat(row["target"])] = abs(y-a)-abs(y-b) if loss == "mae" else 100*(abs(y-a)/y-abs(y-b)/y)
                        actuals.append(y); hp.append(a); cp.append(b)
                    summaries.append({"product": product, "horizon": horizon, "comparator": comparator, "loss": loss,
                        "n": len(rows), "huberMetric": metrics(actuals, hp)[loss], "comparatorMetric": metrics(actuals, cp)[loss],
                        "dm": {str(bw): dm(diffs, bw) for bw in HAC}})
            for mode, groupfn in (("year", lambda d: d.year), ("season", lambda d: (d.month-1)//3+1)):
                groups = defaultdict(list)
                for r in rows: groups[groupfn(dt.date.fromisoformat(r["target"]))].append(r)
                for value, group in sorted(groups.items()):
                    actual = [r["actual"] for r in group]
                    strata.append({"product": product, "horizon": horizon, "stratum": mode, "value": value,
                        "n": len(group), "huber": metrics(actual, [r["huber"] for r in group]),
                        **{c: metrics(actual, [r[c] for r in group]) for c in COMPARATORS}})
    for bw in HAC:
        fam = [r["dm"][str(bw)] for r in summaries]
        if len(fam) != FAMILY_SIZE: raise AssertionError("fixed Holm family size mismatch")
        holm(fam)
    conclusions = joint_conclusions(summaries, coverage)
    return {"protocol": {"id": "robust-direct-huber-2026-10-06", "evaluation": [EVAL_START.isoformat(), EVAL_END.isoformat()],
        "products": list(dashboard.PRODUCTS), "horizons": list(HORIZONS), "minimumDistinctTrainingTargets": MIN_TRAIN_DAYS,
        "trainTargetsAtOrBeforeFitOrigin": True, "maxSourceAgeDays": dashboard.MAX_SOURCE_AGE_DAYS,
        "model": {"pipeline": ["StandardScaler", "HuberRegressor"], "epsilon": 1.35, "alpha": 0.0001, "maxIter": 1000, "tol": 1e-5},
        "comparators": list(COMPARATORS), "losses": list(LOSSES), "holmFamilySizePerBandwidth": FAMILY_SIZE,
        "bandwidthsCalendarDays": list(HAC), "alpha": 0.05,
        "limitations": "Retrospective exploratory study on revised historical prices; publication times and official target equivalence are unknown."},
        "hashes": hashes or {}, "fitAudits": fits,
        "fitTotals": {k: sum(a[k] for a in fits.values()) for k in ("fitAttempts", "fitSuccesses", "fitFailures", "insufficientMonths")},
        "coverage": coverage, "predictions": predictions, "pairedRows": paired, "comparisons": summaries, "strata": strata,
        "jointConclusions": conclusions}


def joint_conclusions(summaries: list[dict[str, Any]], coverage: dict[str, Any]) -> list[dict[str, Any]]:
    conclusions = []
    for product in dashboard.PRODUCTS:
        for horizon in HORIZONS:
            req = [r for r in summaries if r["product"] == product and r["horizon"] == horizon]
            cov = coverage[f"{product}/{horizon}"]
            inference = all(r["dm"]["9"]["status"] == "tested" and r["n"] >= 30 for r in req)
            metrics_pass = all(r["huberMetric"] is not None and r["comparatorMetric"] is not None and r["huberMetric"] < r["comparatorMetric"] for r in req)
            sig_pass = all(r["dm"]["9"]["holmSignificant"] and r["dm"]["9"]["meanLossDifference"] < 0 for r in req)
            coverage_pass = all(cov["candidateCoverageLossCounts"][c] == 0 for c in COMPARATORS)
            if inference and metrics_pass and sig_pass and coverage_pass: outcome = "improvement"
            elif not inference: outcome = "inference_unavailable"
            elif any(r["huberMetric"] is not None and r["comparatorMetric"] is not None and r["huberMetric"] > r["comparatorMetric"] for r in req): outcome = "worsening_or_mixed"
            else: outcome = "inconclusive"
            conclusions.append({"product": product, "horizon": horizon, "conclusion": outcome,
                "inferenceAvailable": inference, "allSixHac9HolmTestsImprove": sig_pass,
                "allSixPairedMetricsImprove": metrics_pass, "coverageNondeclining": coverage_pass})
    return conclusions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=INPUT_PRICES)
    parser.add_argument("--report", type=Path, default=INPUT_REPORT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-nonfrozen-inputs", action="store_true")
    args = parser.parse_args(argv)
    if args.output.resolve() in (args.prices.resolve(), args.report.resolve()): parser.error("output must not replace an input")
    hashes = {"pricesSha256": sha(args.prices), "directHorizonReportSha256": sha(args.report)}
    if not args.allow_nonfrozen_inputs and (hashes["pricesSha256"] != PRICES_SHA256 or hashes["directHorizonReportSha256"] != REPORT_SHA256):
        parser.error("input SHA-256 does not match frozen protocol")
    result = build(json.loads(args.prices.read_text()), json.loads(args.report.read_text()), hashes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(args.output)
    print(json.dumps({"output": str(args.output), "hashes": hashes, "fitTotals": result["fitTotals"],
                      "conclusions": result["jointConclusions"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
