"""Frozen three-year half-life weighted monthly Ridge evaluation."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import dashboard, modeling

START, END = dt.date(2025, 9, 25), dt.date(2026, 9, 24)
PRODUCTS = tuple(dashboard.PRODUCTS)
HORIZONS = tuple(range(1, 11))
EXPECTED_PAIRS = 12_508
HALF_LIFE_DAYS = 1095
MIN_TRAIN_TARGETS = 90
BANDWIDTHS = (9, 28)
FAMILY_SIZE = 5 * 10 * 2 * 2
PRICE_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
DIRECT_SHA256 = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
PROTOCOL = {
    "id": "recency-weighted-direct-ridge-2026-10-07",
    "evaluationTargets": [START.isoformat(), END.isoformat()],
    "expectedEvaluationPairs": EXPECTED_PAIRS,
    "halfLifeCalendarDays": HALF_LIFE_DAYS,
    "weight": "2^(-(fitOrigin-target)/1095), normalized to mean 1 per fit",
    "fit": "monthly StandardScaler + Ridge(alpha=10); fit only samples with target < fitOrigin",
    "minimumDistinctTrainingTargets": MIN_TRAIN_TARGETS,
    "fallback": "use frozen direct Ridge for unavailable features, insufficient history, fit failure, or invalid output",
    "dm": {"twoSided": True, "calendarBartlettHacDays": list(BANDWIDTHS), "holmAlpha": 0.05,
           "familySizePerBandwidth": FAMILY_SIZE, "unavailablePValue": 1.0},
}


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _num(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} must be {'positive and ' if positive else ''}finite")
    return result


def _observations(raw: dict[str, Any]):
    if raw.get("schemaVersion") != 1 or not isinstance(raw.get("products"), list):
        raise ValueError("prices must use dashboard schemaVersion 1")
    data = {}
    for product in raw["products"]:
        pid = product.get("id")
        if pid not in PRODUCTS or pid in data:
            raise ValueError("unknown or duplicate product in prices")
        data[pid] = modeling._observed_series(product)
    if set(data) != set(PRODUCTS):
        raise ValueError("prices must contain every product")
    return data


def _pairs(report: dict[str, Any]) -> dict[tuple[str, dt.date, int], dict[str, Any]]:
    if report.get("protocol", {}).get("id") != "direct-horizon-ridge-2026-10-06":
        raise ValueError("direct report protocol mismatch")
    rows = report.get("pairedRows")
    if not isinstance(rows, list):
        raise ValueError("direct report pairedRows missing")
    result = {}
    for row in rows:
        pid = row.get("product"); target = modeling._parse_day(row.get("target")); h = row.get("horizon")
        if pid not in PRODUCTS or target is None or isinstance(h, bool) or h not in HORIZONS:
            raise ValueError("invalid direct paired-row identity")
        if not START <= target <= END:
            continue
        origin = target - dt.timedelta(days=h)
        supplied_origin = modeling._parse_day(row.get("origin"))
        if supplied_origin is not None and supplied_origin != origin:
            raise ValueError("direct paired-row origin or target mismatch")
        key = (pid, target, h)
        if key in result:
            raise ValueError("duplicate direct paired-row identity")
        result[key] = {**row, "actual": _num(row.get("actual"), "actual", positive=True),
                       "direct": _num(row.get("direct"), "direct Ridge", positive=True),
                       "persistence": _num(row.get("persistence"), "persistence", positive=True),
                       "originDate": origin}
    if len(result) != EXPECTED_PAIRS:
        raise ValueError(f"evaluation pair count mismatch: expected {EXPECTED_PAIRS}, found {len(result)}")
    if {(p, h) for p, _, h in result} != {(p, h) for p in PRODUCTS for h in HORIZONS}:
        raise ValueError("direct paired rows do not cover every crop and horizon")
    return result


def weights(fit_origin: dt.date, targets: list[dt.date]) -> list[float]:
    if any(target >= fit_origin for target in targets):
        raise ValueError("training target must be strictly before fit origin")
    raw = [2.0 ** (-((fit_origin - target).days / HALF_LIFE_DAYS)) for target in targets]
    mean = sum(raw) / len(raw)
    return [w / mean for w in raw]


def _fit_predict(series, horizon, fit_origin, eval_rows, cache):
    import numpy as np
    from threadpoolctl import threadpool_limits

    samples = []
    for target, (actual, _) in series.items():
        if target >= fit_origin:
            continue
        origin = target - dt.timedelta(days=horizon)
        if origin < dt.date(2014, 1, 1):
            continue
        feature = modeling._feature_row(series, origin, target, horizon, cache)
        if feature is None:
            continue
        source_day, vector = feature
        samples.append((target, source_day, vector, math.log(actual) - math.log(series[source_day][0])))
    distinct = {sample[0] for sample in samples}
    audit = {"fitOrigin": fit_origin.isoformat(), "trainingTargetEnd": max(distinct).isoformat() if distinct else None,
             "trainingRows": len(samples), "distinctTrainingTargets": len(distinct), "weightNormalizationMean": 1.0,
             "weightMin": None, "weightMax": None, "status": None}
    if len(distinct) < MIN_TRAIN_TARGETS:
        audit["status"] = "insufficient_training"
        return {}, audit
    w = weights(fit_origin, [r[0] for r in samples])
    audit.update(weightMin=min(w), weightMax=max(w))
    x = np.asarray([r[2] for r in samples], dtype=float); y = np.asarray([r[3] for r in samples], dtype=float)
    try:
        estimator = modeling._new_estimator("ridge")
        with threadpool_limits(limits=1):
            estimator.fit(x, y, ridge__sample_weight=np.asarray(w, dtype=float))
            predictions = {}
            unavailable = []
            invalid_outputs = []
            for key, row in eval_rows:
                try:
                    feat = modeling._feature_row(series, row["originDate"], row["targetDate"], horizon, cache)
                    if feat is None:
                        unavailable.append(row["targetDate"].isoformat())
                        continue
                    source_day, vector = feat
                    source_price = series[source_day][0]
                    raw_prediction = float(estimator.predict(np.asarray([vector]))[0])
                    if not math.isfinite(raw_prediction):
                        raise ValueError("nonfinite_raw_prediction")
                    pred, _ = modeling._inverse_log_ratio(source_price, raw_prediction)
                    if not math.isfinite(pred) or pred <= 0:
                        raise ValueError("invalid_price_prediction")
                    predictions[key] = pred
                except Exception:
                    invalid_outputs.append(row["targetDate"].isoformat())
        audit["status"] = "fitted"
        audit["evaluationFeatureUnavailableTargets"] = unavailable
        audit["invalidPredictionTargets"] = invalid_outputs
        return predictions, audit
    except Exception as exc:
        audit.update(status="fit_failed", errorType=type(exc).__name__, errorMessage=str(exc)[:240])
        return {}, audit


def _metrics(rows, field):
    if not rows:
        return {"n": 0, "mae": None, "mape": None}
    return {"n": len(rows), "mae": sum(abs(r["actual"] - r[field]) for r in rows) / len(rows),
            "mape": 100 * sum(abs(r["actual"] - r[field]) / r["actual"] for r in rows) / len(rows)}


def _dm(diff: dict[dt.date, float], lag: int):
    n = len(diff)
    if n < 30:
        return {"status": "insufficient_pairs", "n": n, "meanLossDifference": sum(diff.values()) / n if n else None,
                "statistic": None, "pValue": 1.0}
    mean = sum(diff.values()) / n
    centered = {day: value - mean for day, value in diff.items()}
    variance = sum(v * v for v in centered.values())
    for k in range(1, lag + 1):
        cross = sum(v * centered.get(day + dt.timedelta(days=k), 0.0) for day, v in centered.items())
        variance += 2 * (1 - k / (lag + 1)) * cross
    variance /= n * n
    if not math.isfinite(variance) or variance <= 0:
        return {"status": "zero_or_invalid_hac_variance", "n": n, "meanLossDifference": mean,
                "statistic": None, "pValue": 1.0}
    stat = mean / math.sqrt(variance)
    return {"status": "tested", "n": n, "meanLossDifference": mean, "statistic": stat,
            "pValue": math.erfc(abs(stat) / math.sqrt(2))}


def _holm(rows):
    running = 0.0
    for rank, idx in enumerate(sorted(range(len(rows)), key=lambda i: rows[i]["pValue"])):
        running = max(running, min(1.0, (len(rows) - rank) * rows[idx]["pValue"]))
        rows[idx]["holmPValue"] = running; rows[idx]["holmSignificant"] = running < .05


def build(prices, report, *, hashes=None, code_sha256=None):
    series_by_product = _observations(prices); pairs = _pairs(report)
    for (product, target, _), row in pairs.items():
        observed = series_by_product[product].get(target)
        if observed is None or observed[0] != row["actual"]:
            raise ValueError("direct paired actual does not match the pinned price observations")
    for key, row in pairs.items():
        row["targetDate"] = key[1]
    candidate = {}; audits = {}
    for product in PRODUCTS:
        for h in HORIZONS:
            series = series_by_product[product]; cache = {}
            months = defaultdict(list)
            for key, row in pairs.items():
                if key[0] == product and key[2] == h:
                    months[(row["originDate"].year, row["originDate"].month)].append((key, row))
            for month, eval_rows in sorted(months.items()):
                fit_origin = min(row["originDate"] for _, row in eval_rows)
                preds, audit = _fit_predict(series, h, fit_origin, eval_rows, cache)
                audit["originMonth"] = f"{month[0]:04d}-{month[1]:02d}"
                audits[f"{product}/{h}/{audit['originMonth']}"] = audit
                for key, row in eval_rows:
                    pred = preds.get(key)
                    if pred is not None and math.isfinite(pred) and pred > 0:
                        status = "weighted_ridge"
                    elif audit["status"] == "fitted":
                        target_text = row["targetDate"].isoformat()
                        if target_text in audit.get("evaluationFeatureUnavailableTargets", []):
                            status = "feature_unavailable_or_source_too_old"
                        elif target_text in audit.get("invalidPredictionTargets", []):
                            status = "invalid_candidate_prediction"
                        else:
                            status = "candidate_prediction_missing"
                    else:
                        status = audit["status"] or "invalid_prediction"
                    if status != "weighted_ridge":
                        pred = row["direct"]
                    candidate[key] = (pred, status)
    paired = []
    for key in sorted(pairs):
        row = pairs[key]; pred, status = candidate[key]
        series = series_by_product[key[0]]
        feature = modeling._feature_row(series, row["originDate"], row["targetDate"], key[2], {})
        source_day = feature[0] if feature is not None else max((day for day in series if day <= row["originDate"]), default=None)
        source_age = (row["originDate"] - source_day).days if source_day is not None else None
        if feature is None and status == "fitted":
            status = "feature_unavailable_or_source_too_old"
            pred = row["direct"]
        paired.append({"product": key[0], "target": key[1].isoformat(), "origin": row["originDate"].isoformat(),
                       "horizon": key[2], "actual": row["actual"], "candidate": pred,
                       "directRidge": row["direct"], "persistence": row["persistence"],
                       "sourceDate": source_day.isoformat() if source_day else None, "sourceAgeDays": source_age,
                       "fallback": status != "weighted_ridge", "fallbackReason": None if status == "weighted_ridge" else status})
    tests_by_lag = {lag: [] for lag in BANDWIDTHS}; cells = []
    grouped = defaultdict(list)
    for row in paired: grouped[(row["product"], row["horizon"])].append(row)
    for product in PRODUCTS:
        for h in HORIZONS:
            rows = grouped[(product, h)]
            metrics = {m: _metrics(rows, m) for m in ("candidate", "directRidge", "persistence")}
            comps = {}
            for comparator in ("directRidge", "persistence"):
                comps[comparator] = {"candidate": metrics["candidate"], "comparator": metrics[comparator]}
                for loss in ("mae", "mape"):
                    diff = {}
                    for r in rows:
                        a, b = abs(r["actual"] - r["candidate"]), abs(r["actual"] - r[comparator])
                        if loss == "mape": a, b = a / r["actual"] * 100, b / r["actual"] * 100
                        diff[dt.date.fromisoformat(r["target"])] = a - b
                    for lag in BANDWIDTHS:
                        tests_by_lag[lag].append({"product": product, "horizon": h, "comparator": comparator,
                                                  "loss": loss, **_dm(diff, lag)})
            cells.append({"product": product, "horizon": h, "n": len(rows), "metrics": metrics,
                          "comparisons": comps, "fallbackCount": sum(r["fallback"] for r in rows)})
    for tests in tests_by_lag.values():
        if len(tests) != FAMILY_SIZE: raise AssertionError("fixed Holm family shape mismatch")
        _holm(tests)
    hac9 = {(t["product"], t["horizon"], t["comparator"], t["loss"]): t
            for t in tests_by_lag[9]}
    for cell in cells:
        required = [hac9[(cell["product"], cell["horizon"], comp, loss)]
                    for comp in ("directRidge", "persistence") for loss in ("mae", "mape")]
        all_better = all(cell["comparisons"][t["comparator"]]["candidate"][t["loss"]]
                         < cell["comparisons"][t["comparator"]]["comparator"][t["loss"]]
                         for t in required)
        all_significant = all(t["status"] == "tested" and t["holmSignificant"]
                              and t["meanLossDifference"] < 0 for t in required)
        adverse = any(t["status"] == "tested" and t["holmSignificant"]
                      and t["meanLossDifference"] > 0 for t in required)
        cell["conclusionHAC9"] = ("improvement" if all_better and all_significant
                                  else "worsening_or_mixed" if adverse else "inconclusive")
        cell["coverageNondeclining"] = True
        cell["allFourHac9HolmTestsImprove"] = all_significant
        cell["allFourPairedMetricsImprove"] = all_better
    by_product = {}
    for product in PRODUCTS:
        by_product[product] = {}
        for name in ("candidate", "directRidge", "persistence"):
            vals = [c["metrics"][name]["mape"] for c in cells if c["product"] == product and c["metrics"][name]["mape"] is not None]
            by_product[product][name] = sum(vals) / len(vals) if len(vals) == len(HORIZONS) else None
    overall = {}
    for name in ("candidate", "directRidge", "persistence"):
        vals = [by_product[p][name] for p in PRODUCTS]
        overall[name] = {"mape": sum(vals) / len(vals) if all(v is not None for v in vals) else None,
                         "availableProducts": sum(v is not None for v in vals), "totalProducts": len(PRODUCTS)}
    macro = {"equalHorizonMapeByProduct": by_product,
             "fiveProductEqualWeight": overall,
             "fiveProductEqualWeightMape": overall["candidate"]["mape"]}
    by_season = defaultdict(list)
    for row in paired:
        month = dt.date.fromisoformat(row["target"]).month
        season = (month - 1) // 3 + 1
        by_season[str(season)].append(row)
    seasonal = {season: {"n": len(rows), "fallbackCount": sum(r["fallback"] for r in rows),
                         "fallbackRate": sum(r["fallback"] for r in rows) / len(rows),
                         "metrics": {name: _metrics(rows, name) for name in ("candidate", "directRidge", "persistence")}}
                for season, rows in sorted(by_season.items())}
    return {"schemaVersion": 1, "status": "complete", "protocol": PROTOCOL, "hashes": hashes or {},
            "codeSha256": code_sha256, "coverage": {"n": len(paired), "expected": EXPECTED_PAIRS,
                "byYear": {str(y): sum(r["target"].startswith(str(y)) for r in paired) for y in range(2025, 2027)},
                "bySeason": seasonal,
                "fallbackCount": sum(r["fallback"] for r in paired)}, "macro": macro,
            "cells": cells, "fitAudits": audits, "pairedRows": paired,
            "multipleTesting": {f"HAC{lag}": tests for lag, tests in tests_by_lag.items()}}


def _immutable_json(path: Path, data: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())
    return _sha_bytes(raw)


def run(prices_path: Path, report_path: Path, manifest_path: Path, output_path: Path, *, execute=False):
    pb, rb = prices_path.read_bytes(), report_path.read_bytes()
    hashes = {"pricesSha256": _sha_bytes(pb), "directReportSha256": _sha_bytes(rb)}
    if hashes != {"pricesSha256": PRICE_SHA256, "directReportSha256": DIRECT_SHA256}:
        raise ValueError("input SHA-256 differs from frozen protocol")
    code_hash = _sha_bytes(Path(__file__).read_bytes())
    prices, report = json.loads(pb), json.loads(rb)
    _observations(prices); _pairs(report)
    manifest = {"schemaVersion": 1, "status": "frozen_before_scoring", "createdAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "inputSha256": hashes, "codeSha256": code_hash, "protocol": PROTOCOL}
    manifest_hash = _immutable_json(manifest_path, manifest)
    if not execute:
        return {"status": "dryrun", "manifestSha256": manifest_hash, "manifest": str(manifest_path)}
    result = build(prices, report, hashes=hashes, code_sha256=code_hash)
    result["manifestSha256"] = manifest_hash
    output_hash = _immutable_json(output_path, result)
    return {"status": "complete", "manifestSha256": manifest_hash, "reportSha256": output_hash,
            "manifest": str(manifest_path), "report": str(output_path)}


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, default=root / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json")
    parser.add_argument("--direct", type=Path, default=root / "outputs/experiments/direct-horizon-20261006/report.json")
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.prices, args.direct, args.manifest, args.output, execute=args.execute), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
