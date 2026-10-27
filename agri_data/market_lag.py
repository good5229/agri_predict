"""Frozen offline cross-market lag feature experiment (no API access)."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from . import dashboard, direct_horizon as dh, market_mix, modeling, recent_window as rw

ROOT = Path(__file__).resolve().parents[1]
INPUT_PRICES = dh.INPUT_PRICES
INPUT_PREDICTIONS = dh.INPUT_PREDICTIONS
INPUT_RIDGE_REPORT = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
INPUT_RECOMPOSE_REPORT = ROOT / "outputs/experiments/market-recompose-20261007/report.json"
OUTPUT = ROOT / "outputs/experiments/market-lag-20261007/report.json"
GARAK, GANGSEO = dashboard.MARKETS
PRODUCTS = tuple(dashboard.PRODUCTS)
HORIZONS = tuple(range(1, 11))
EVAL_START, EVAL_END = dt.date(2023, 1, 1), dt.date(2026, 9, 27)
MIN_TRAIN = 90
MAX_AGE = 14
COMPARATORS = ("directRidge", "marketRecompose", "persistence")
LOSSES = ("mae", "mape")
FAMILY_SIZE = 5 * 10 * 3 * 2
EXPECTED = {
    "pricesSha256": "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8",
    "predictionsSha256": "c41ed5d8799306f151b1c5135399a606982f12623bf709a4438969d8cc389113",
    "directRidgeReportSha256": "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67",
    "marketRecomposeReportSha256": "e5f23f1012dad2acc774c83f995e2ee2851409af913ee82483aa3d953ee87581",
}
PROTOCOL = {
    "id": "external-evidence-market-lag-direct-ridge-2026-10-07",
    "features": ["garak log(p0/p7)", "gangseo log(p0/p7)", "log(garak p0/gangseo p0)"],
    "featureRule": "latest positive actual price at or before origin (p0) and origin-7 calendar days (p7); each source age <=14 calendar days; no filling",
    "baseFeatures": "modeling._feature_row price/volume feature vector, followed by the three market features",
    "fit": "per product and horizon; first eligible origin in each evaluation month; expanding Ridge(alpha=10) with training standardization; target < fit origin; >=90 distinct targets",
    "eligibility": "evaluation target must have pinned direct Ridge issuance; targets 2023-01-01..2026-09-27 and origins >=2023-01-01",
    "fallback": "feature unavailable, insufficient training, fit/predict failure or invalid prediction => pinned direct Ridge; garlic and dried red pepper always pinned direct Ridge",
    "comparison": "same issued candidate/direct Ridge/market recomposition/persistence target dates; 5 products x 10 horizons x 3 comparators x 2 losses = 300 tests per bandwidth",
    "inference": "two-sided DM; actual-calendar Bartlett HAC9 primary and HAC28 sensitivity; Holm alpha .05; fewer than 30 pairs or invalid variance retained with p=1",
    "limitations": "retrospective revised observations; historical first publication time unknown; exploratory historical evaluation",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _protocol_sha() -> str:
    return _sha(json.dumps(PROTOCOL, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _market_series(markets):
    out = {p: {m: {} for m in dashboard.MARKETS} for p in PRODUCTS}
    for p in PRODUCTS:
        for m in dashboard.MARKETS:
            for day, row in markets.get(p, {}).get(m, {}).items():
                day = day if isinstance(day, dt.date) else dt.date.fromisoformat(day)
                try:
                    price, weight = float(row["price"]), float(row["weight"])
                except (KeyError, TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(price) and price > 0 and math.isfinite(weight) and weight > 0:
                    out[p][m][day] = price
    return out


def _market_features(series, product: str, origin: dt.date) -> tuple[float, float, float] | None:
    detail = _market_feature_detail(series, product, origin)
    return detail[0] if detail is not None else None


def _market_feature_detail(series, product: str, origin: dt.date):
    pairs, sources = [], {}
    for market in (GARAK, GANGSEO):
        obs = series[product][market]
        p0_days = [d for d in obs if d <= origin and (origin - d).days <= MAX_AGE]
        cutoff = origin - dt.timedelta(days=7)
        p7_days = [d for d in obs if d <= cutoff and (cutoff - d).days <= MAX_AGE]
        if not p0_days or not p7_days:
            return None
        p0_day, p7_day = max(p0_days), max(p7_days)
        pairs.append((obs[p0_day], obs[p7_day]))
        sources[market] = {"p0Date": p0_day.isoformat(), "p0AgeDays": (origin-p0_day).days,
                           "p7Date": p7_day.isoformat(), "p7AgeDays": (cutoff-p7_day).days}
    g0, g7 = pairs[0]
    j0, j7 = pairs[1]
    return (math.log(g0 / g7), math.log(j0 / j7), math.log(g0 / j0)), sources


def _recompose_map(report):
    result = {}
    for row in report.get("originEvidence", []):
        key = (row.get("product"), int(row.get("horizon")), dt.date.fromisoformat(row["target"]))
        if key in result:
            raise ValueError(f"duplicate market-recomposition target: {key}")
        value = row.get("candidatePrediction")
        if value is not None:
            value = float(value)
            if math.isfinite(value) and value > 0:
                result[key] = value
    return result


def _persistence_map(raw):
    _, _, maps = dh._forecast_maps(raw)
    return {(p, h, target): value for p, rows in maps.items() for (target, h), value in rows.items()}


def _valid_prediction(value):
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _prediction_from_estimator(estimator, vector, source_price):
    """Predict a log ratio, rejecting nonfinite deltas before inverse clipping."""
    import numpy as np

    delta = float(estimator.predict(np.asarray(vector))[0])
    if not math.isfinite(delta):
        return None
    value, _ = modeling._inverse_log_ratio(source_price, delta)
    return _valid_prediction(value)


def _joint_decision(tests, coverage_ok):
    inference_available = all(row["n"] >= 30 and row["dm"]["9"]["status"] == "tested" for row in tests)
    metrics_better = all(row["candidateMetric"] is not None and row["comparatorMetric"] is not None
                         and row["candidateMetric"] < row["comparatorMetric"] for row in tests)
    significant = all(row["dm"]["9"].get("holmSignificant")
                      and row["dm"]["9"].get("meanLossDifference", 0) < 0 for row in tests)
    adverse = any(row["dm"]["9"]["status"] == "tested" and row["dm"]["9"].get("holmSignificant")
                  and row["dm"]["9"].get("meanLossDifference", 0) > 0 for row in tests)
    conclusion = ("worsening_or_mixed" if adverse else "inference_unavailable" if not inference_available
                  else "improvement" if metrics_better and significant and coverage_ok else "inconclusive")
    return {"conclusion": conclusion, "inferenceAvailable": inference_available,
            "allSixPairedMetricsImprove": metrics_better, "allSixHac9HolmTestsImprove": significant,
            "coverageNondeclining": coverage_ok, "significantWorseningEvidence": adverse}


def build(raw, markets, predictions, ridge_report, recompose_report, *, hashes=None):
    """Build the single fixed candidate, fallback trace, and paired inference."""
    import numpy as np
    from threadpoolctl import threadpool_limits

    actuals = dh._observations(raw)
    ridge, ridge_actual = rw._direct_ridge_reference(ridge_report)
    recompose, persistence = _recompose_map(recompose_report), _persistence_map(predictions)
    for (p, h, target), expected in ridge_actual.items():
        if target in actuals.get(p, {}) and not math.isclose(actuals[p][target][0], expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"pinned Ridge actual differs from frozen prices: {p}/{h}/{target}")
    market = _market_series(markets)
    candidate, traces, audits = {}, [], {}

    for p in PRODUCTS:
        series = actuals[p]
        # modeling._feature_row cache keys omit the product/series, so it is
        # shared across horizons only and must be reset for each product.
        feature_cache = {}
        for h in HORIZONS:
            # Training examples use the same target and direct-Ridge price features.
            samples = []
            for target, (actual, _) in series.items():
                origin = target - dt.timedelta(days=h)
                if origin < dt.date(2014, 1, 1):
                    continue
                base = modeling._feature_row(series, origin, target, h, feature_cache)
                extra = _market_features(market, p, origin)
                if base is None or extra is None:
                    continue
                label = math.log(actual / series[base[0]][0])
                samples.append((target, origin, tuple(base[1]) + extra, label, base[0]))
            samples.sort(key=lambda r: r[0])
            eligible_by_month = {}
            for target in series:
                key = (p, h, target)
                origin = target - dt.timedelta(days=h)
                if key not in ridge or not (EVAL_START <= target <= EVAL_END and origin >= EVAL_START):
                    continue
                base = modeling._feature_row(series, origin, target, h, feature_cache)
                extra = _market_features(market, p, origin)
                if base is not None and extra is not None:
                    eligible_by_month.setdefault((origin.year, origin.month), []).append(
                        (target, origin, tuple(base[1]) + extra, base[0]))
            monthly_models = {}
            for ym, rows in (sorted(eligible_by_month.items()) if p in PRODUCTS[:3] else []):
                fit_origin = min(row[1] for row in rows)
                train = [r for r in samples if r[0] < fit_origin]
                n_targets = len({r[0] for r in train})
                audit_key = f"{p}/{h}/{ym[0]:04d}-{ym[1]:02d}"
                audit = {"fitOrigin": fit_origin.isoformat(), "distinctTrainingTargets": n_targets,
                         "eligibleOrigins": len(rows), "status": "pending"}
                if n_targets < MIN_TRAIN:
                    audit["status"] = "insufficient_training"
                    monthly_models[ym] = (None, audit)
                    audits[audit_key] = audit
                    continue
                try:
                    with threadpool_limits(limits=1):
                        estimator = modeling._new_estimator("ridge")
                        estimator.fit(np.asarray([r[2] for r in train]), np.asarray([r[3] for r in train]))
                    monthly_models[ym] = (estimator, audit)
                    audit.update(status="fitted", trainingTargetEnd=max(r[0] for r in train).isoformat())
                except Exception as exc:
                    monthly_models[ym] = (None, audit)
                    audit.update(status="fit_failed", errorType=type(exc).__name__, error=str(exc)[:200])
                audits[audit_key] = audit

            for target, (actual, _) in series.items():
                key = (p, h, target)
                if key not in ridge or not (EVAL_START <= target <= EVAL_END):
                    continue
                origin = target - dt.timedelta(days=h)
                if origin < EVAL_START:
                    continue
                base_direct = float(ridge[key])
                feature_detail = _market_feature_detail(market, p, origin)
                extra = feature_detail[0] if feature_detail is not None else None
                base = modeling._feature_row(series, origin, target, h, feature_cache)
                ym = (origin.year, origin.month)
                reasons = []
                prediction = None
                trace_error = None
                if p not in PRODUCTS[:3]:
                    reasons.append("always_fallback_product")
                elif base is None:
                    reasons.append("base_feature_unavailable")
                elif extra is None:
                    reasons.append("market_feature_unavailable")
                else:
                    fitted = monthly_models.get(ym)
                    if fitted is None:
                        reasons.append("no_eligible_month_fit")
                    elif fitted[0] is None:
                        state = fitted[1]["status"]
                        reasons.append("training_insufficient" if state == "insufficient_training" else "fit_failed")
                    else:
                        try:
                            vector = np.asarray([tuple(base[1]) + extra])
                            with threadpool_limits(limits=1):
                                prediction = _prediction_from_estimator(fitted[0], vector, series[base[0]][0])
                            if prediction is None:
                                reasons.append("invalid_candidate_prediction")
                        except Exception as exc:
                            reasons.append("prediction_failed")
                            trace_error = {"type": type(exc).__name__, "message": str(exc)[:200]}
                        else:
                            trace_error = None
                if prediction is None:
                    prediction = base_direct
                else:
                    trace_error = None
                candidate[key] = prediction
                trace = {"product": p, "horizon": h, "origin": origin.isoformat(), "target": target.isoformat(),
                         "actual": actual, "directRidge": base_direct,
                         "marketRecompose": recompose.get(key), "persistence": persistence.get(key),
                         "marketFeatures": list(extra) if extra is not None else None,
                         "marketFeatureSources": feature_detail[1] if feature_detail is not None else None,
                         "baseFeatureSourceDate": base[0].isoformat() if base is not None else None,
                         "candidatePrediction": prediction, "fallback": bool(reasons),
                         "fallbackReasons": reasons, "fitAuditKey": f"{p}/{h}/{origin.year:04d}-{origin.month:02d}"}
                if trace_error is not None:
                    trace["predictionError"] = trace_error
                traces.append(trace)

    paired, comparisons, coverage = [], [], {}
    for p in PRODUCTS:
        for h in HORIZONS:
            traces_cell = [r for r in traces if r["product"] == p and r["horizon"] == h]
            candidate_dates = {r["target"] for r in traces_cell if r["candidatePrediction"] is not None}
            eligible_dates = {t.isoformat() for t in actuals[p]
                if EVAL_START <= t <= EVAL_END and t - dt.timedelta(days=h) >= EVAL_START}
            maps = {"candidate": {r["target"]: r["candidatePrediction"] for r in traces_cell},
                    "directRidge": {t.isoformat(): v for (pp, hh, t), v in ridge.items() if pp == p and hh == h and t.isoformat() in eligible_dates},
                    "marketRecompose": {t.isoformat(): v for (pp, hh, t), v in recompose.items() if pp == p and hh == h and t.isoformat() in eligible_dates},
                    "persistence": {t.isoformat(): v for (pp, hh, t), v in persistence.items() if pp == p and hh == h and t.isoformat() in eligible_dates}}
            common = set.intersection(*(set(maps[name]) for name in maps)) if maps else set()
            rows = []
            for r in traces_cell:
                if r["target"] not in common:
                    continue
                row = dict(r)
                rows.append(row)
                paired.append(row)
            for comp in COMPARATORS:
                for loss in LOSSES:
                    diffs = {}
                    for r in rows:
                        day = dt.date.fromisoformat(r["target"])
                        ca, cb = abs(r["actual"] - r["candidatePrediction"]), abs(r["actual"] - r[comp])
                        diffs[day] = ca - cb if loss == "mae" else 100.0 * (ca - cb) / r["actual"]
                    ca = [r["candidatePrediction"] for r in rows]
                    cb = [r[comp] for r in rows]
                    actual = [r["actual"] for r in rows]
                    comparisons.append({"product": p, "horizon": h, "comparator": comp, "loss": loss,
                        "n": len(rows), "candidateMetric": dh._metrics(actual, ca)[loss],
                        "comparatorMetric": dh._metrics(actual, cb)[loss],
                        "dm": {str(bw): dh._dm(diffs, bw) for bw in (9, 28)}})
            issue_counts = {name: len(values) for name, values in maps.items()}
            coverage[f"{p}/{h}"] = {"eligibleTargets": len(eligible_dates), "eligibleDirectRidgeTargets": len(traces_cell),
                "issuanceCounts": issue_counts,
                "availabilityRates": {name: len(values) / len(eligible_dates) if eligible_dates else None for name, values in maps.items()},
                "unissuedDates": {name: sorted(eligible_dates - set(values)) for name, values in maps.items()},
                "commonPairedCount": len(rows), "pairedDates": [r["target"] for r in rows],
                "fallbackCount": sum(r["fallback"] for r in traces_cell),
                "fallbackReasons": {reason: sum(reason in r["fallbackReasons"] for r in traces_cell)
                    for reason in sorted({x for r in traces_cell for x in r["fallbackReasons"]})},
                "candidateCoverageLossDates": {comp: sorted(set(maps[comp]) - candidate_dates) for comp in COMPARATORS}}

    if len(comparisons) != FAMILY_SIZE:
        raise AssertionError(f"frozen comparison family must contain {FAMILY_SIZE} comparisons")
    for bw in (9, 28):
        family = [r["dm"][str(bw)] for r in comparisons]
        dh._holm(family)
        for row, dm in zip(comparisons, family):
            row["dm"][str(bw)] = dm

    strata = []
    for p in PRODUCTS:
        for h in HORIZONS:
            rows = [r for r in paired if r["product"] == p and r["horizon"] == h]
            eligible_rows = [r for r in traces if r["product"] == p and r["horizon"] == h]
            for kind, fn in (("year", lambda d: str(d.year)), ("season", lambda d: "winter" if d.month in (12, 1, 2) else "spring" if d.month in (3, 4, 5) else "summer" if d.month in (6, 7, 8) else "autumn")):
                groups = {}
                eligible_groups = {}
                for row in rows:
                    d = dt.date.fromisoformat(row["target"])
                    groups.setdefault(fn(d), []).append(row)
                for row in eligible_rows:
                    d = dt.date.fromisoformat(row["target"])
                    eligible_groups.setdefault(fn(d), []).append(row)
                for label, group in sorted(groups.items()):
                    actual = [r["actual"] for r in group]
                    egroup = eligible_groups[label]
                    strata.append({"product": p, "horizon": h, "stratum": kind, "value": label,
                        "n": len(group), "eligibleCandidateOrigins": len(egroup),
                        "fallbackCount": sum(r["fallback"] for r in egroup),
                        "fallbackRate": sum(r["fallback"] for r in egroup) / len(egroup) if egroup else None,
                        "metrics": {name: dh._metrics(actual, [r[field] for r in group])
                            for name, field in (("candidate", "candidatePrediction"), ("directRidge", "directRidge"),
                                                ("marketRecompose", "marketRecompose"), ("persistence", "persistence"))}})
    macros = {}
    for method, field in (("candidate", "candidatePrediction"), ("directRidge", "directRidge"),
                          ("marketRecompose", "marketRecompose"), ("persistence", "persistence")):
        values = []
        for p in PRODUCTS:
            for h in HORIZONS:
                rows = [r for r in paired if r["product"] == p and r["horizon"] == h]
                values.append(dh._metrics([r["actual"] for r in rows], [r[field] for r in rows])["mape"] if rows else None)
        macros[method] = sum(values) / 50 if len(values) == 50 and all(v is not None for v in values) else None
    joint = []
    for p in PRODUCTS:
        for h in HORIZONS:
            tests = [r for r in comparisons if r["product"] == p and r["horizon"] == h]
            cov_ok = all(not coverage[f"{p}/{h}"]["candidateCoverageLossDates"][c] for c in COMPARATORS)
            joint.append({"product": p, "horizon": h, **_joint_decision(tests, cov_ok)})
    return {"schemaVersion": 1, "status": "complete", "protocol": PROTOCOL, "protocolSha256": _protocol_sha(), "hashes": hashes or {},
        "fitAudits": audits, "fitSummary": {state: sum(a["status"] == state for a in audits.values()) for state in ("fitted", "fit_failed", "insufficient_training")},
        "originTrace": traces, "pairedRows": paired, "comparisons": comparisons, "coverage": coverage,
        "yearSeasonMetrics": strata, "macroCommonDateMAPE": macros, "jointConclusions": joint,
        "limitations": PROTOCOL["limitations"]}


def _atomic_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        tmp.write_bytes(payload)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()


def run(prices_path=INPUT_PRICES, predictions_path=INPUT_PREDICTIONS, ridge_path=INPUT_RIDGE_REPORT,
        recompose_path=INPUT_RECOMPOSE_REPORT, db_path=market_mix.DB_PATH, output=OUTPUT,
        *, allow_nonfrozen_inputs=False):
    paths = {"pricesSha256": Path(prices_path), "predictionsSha256": Path(predictions_path),
             "directRidgeReportSha256": Path(ridge_path), "marketRecomposeReportSha256": Path(recompose_path)}
    blobs = {key: path.read_bytes() for key, path in paths.items()}
    hashes = {key: _sha(blob) for key, blob in blobs.items()}
    if not allow_nonfrozen_inputs and hashes != EXPECTED:
        raise ValueError("input SHA-256 does not match frozen market-lag protocol")
    raw, predictions, ridge, recompose = (json.loads(blobs[key]) for key in paths)
    core = dh._observations(raw)
    markets, provenance = market_mix.extract_market_data(Path(db_path), raw, core)
    report = build(raw, markets, predictions, ridge, recompose, hashes=hashes)
    report["inputs"] = {"selectedRequestHashes": provenance["selected"]["selectedRequestHashes"],
        "selectedScopesSha256": provenance["selected"]["selectedScopesSha256"],
        "broadScopesSha256": provenance["broad"]["scopesSha256"], "reconciliation": provenance["reconciliation"]}
    _atomic_json(Path(output), report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(OUTPUT))
    args = parser.parse_args(argv)
    run(output=Path(args.output))


if __name__ == "__main__":
    main()
