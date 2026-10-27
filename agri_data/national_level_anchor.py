"""Frozen retrospective national-level ratio anchor for paired forecasts.

Consumes one hash-pinned national-transfer JSON report only. It performs no
network, database, archive, or model-training work.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import uuid

from . import direct_horizon as dh

ROOT = Path(__file__).resolve().parents[1]
INPUT_REPORT = ROOT / "outputs/experiments/national-transfer-20261007/report.json"
OUTPUT_DIR = ROOT / "outputs/experiments/national-level-anchor-20261007"
REPORT_SHA256 = "4a22a69e3e39aed56064641a9f5d5eed255057153ee705bc7c40bb22d338817d"
NATIONAL_ARCHIVE_SHA256 = "b9a38d4a62477054758e7576a7320131bc5acc8129a4079a33977f9ed876d7d7"
MARKET_RECOMPOSE_SHA256 = "e5f23f1012dad2acc774c83f995e2ee2851409af913ee82483aa3d953ee87581"
NATIONAL_TRANSFER_PROTOCOL = "national-wholesale-target-transfer-2026-10-07"
PRODUCTS = ("cabbage", "radish", "onion", "pepper")
HORIZONS = tuple(range(1, 11))
EVAL_START = dt.date(2026, 1, 1)
EVAL_END = dt.date(2026, 6, 30)
ANCHOR_START_OFFSET = 15
ANCHOR_END_OFFSET = 2
MIN_ANCHOR_DAYS = 3
MIN_PAIRS = 30
HAC = (9, 28)
FAMILY_SIZE = len(PRODUCTS) * len(HORIZONS) * 3 * 2
PROTOCOL = {
    "id": "national-level-ratio-anchor-2026-10-07",
    "inputReportSha256": REPORT_SHA256,
    "historySource": "pinned national-transfer pairedRows; unique product/target positive national and two-market actuals",
    "anchorWindow": "origin-15 through origin-2 inclusive; median nationalActual/twoMarketActual; >=3 distinct dates, else ratio=1",
    "evaluation": "target 2026-01-01..2026-06-30; all same-origin/target/horizon common pairs in input report",
    "candidate": "market-recompose candidatePrediction * origin anchor ratio",
    "comparators": ["candidatePrediction", "directRidge * anchor ratio", "persistence * anchor ratio"],
    "comparisonFamily": "4 products x 10 horizons x 3 comparators x 2 losses = 240 two-sided DM tests per bandwidth; calendar Bartlett HAC9/HAC28; Holm alpha .05",
    "minimumPairs": MIN_PAIRS,
    "limitations": "retrospective revised national prices are treated as if available before the origin; first-publication time is unknown; not official R01 performance",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _date(value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label} must be ISO date")
    try:
        result = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO date") from exc
    if result.isoformat() != value:
        raise ValueError(f"{label} must be canonical ISO date")
    return result


def _positive(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be positive finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be positive finite") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be positive finite")
    return number


def _same_actual(prior, current, label):
    if not math.isclose(prior, current, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError(f"inconsistent repeated target-date actual: {label}")


def _validate_report(report):
    if not isinstance(report, dict) or report.get("schemaVersion") != 1 or report.get("status") != "complete":
        raise ValueError("national-transfer report is not a complete schemaVersion 1 report")
    protocol = report.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("id") != NATIONAL_TRANSFER_PROTOCOL:
        raise ValueError("national-transfer report protocol is not the frozen target-transfer experiment")
    hashes = report.get("hashes")
    if (not isinstance(hashes, dict) or hashes.get("archiveSha256") != NATIONAL_ARCHIVE_SHA256 or
            protocol.get("archiveSha256") != NATIONAL_ARCHIVE_SHA256 or
            hashes.get("marketRecomposeSha256") != MARKET_RECOMPOSE_SHA256):
        raise ValueError("national-transfer report input hashes are missing or inconsistent")

    target_audit = report.get("targetAudit")
    if not isinstance(target_audit, dict) or set(target_audit) != set(PRODUCTS):
        raise ValueError("national-transfer targetAudit must contain exactly the four frozen products")
    national_history = {}
    for product in PRODUCTS:
        values = target_audit[product]
        if not isinstance(values, dict):
            raise ValueError(f"national-transfer targetAudit is invalid for {product}")
        for day_text, row in values.items():
            day = _date(day_text, f"targetAudit {product} date")
            if day < dt.date(2024, 7, 8) or day > dt.date(2026, 6, 30):
                raise ValueError(f"targetAudit date outside pinned archive range: {product}/{day}")
            if not isinstance(row, dict) or "price" not in row:
                raise ValueError(f"national-transfer targetAudit row is invalid: {product}/{day}")
            price = row["price"]
            if price is None:
                continue
            try:
                parsed = float(price)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"national-transfer targetAudit price is invalid: {product}/{day}") from exc
            if not math.isfinite(parsed):
                raise ValueError(f"national-transfer targetAudit price is non-finite: {product}/{day}")
            if parsed > 0:
                national_history[(product, day)] = parsed

    source_rows = report.get("pairedRows")
    if not isinstance(source_rows, list):
        raise ValueError("national-transfer pairedRows must be a list")
    seen_keys = set()
    national_by_date = {}
    local_by_date = {}
    validated_rows = []
    for i, row in enumerate(source_rows):
        if not isinstance(row, dict):
            raise ValueError(f"national-transfer paired row {i} is invalid")
        product = row.get("product")
        if product not in PRODUCTS:
            raise ValueError(f"national-transfer paired row has unknown product: {product!r}")
        horizon = row.get("horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
            raise ValueError(f"national-transfer paired horizon is invalid at row {i}")
        origin = _date(row.get("origin"), f"pairedRows[{i}].origin")
        target = _date(row.get("target"), f"pairedRows[{i}].target")
        if target - origin != dt.timedelta(days=horizon):
            raise ValueError(f"paired row origin/target/horizon mismatch at row {i}")
        if target < dt.date(2024, 7, 8) or target > dt.date(2026, 6, 30):
            raise ValueError(f"paired target outside pinned archive range at row {i}")
        key = (product, origin, target, horizon)
        if key in seen_keys:
            raise ValueError(f"duplicate national-transfer paired key: {key}")
        seen_keys.add(key)
        national = _positive(row.get("nationalActual"), f"pairedRows[{i}].nationalActual")
        local = _positive(row.get("twoMarketActual"), f"pairedRows[{i}].twoMarketActual")
        candidate = _positive(row.get("candidatePrediction"), f"pairedRows[{i}].candidatePrediction")
        ridge = _positive(row.get("directRidge"), f"pairedRows[{i}].directRidge")
        persistence = _positive(row.get("persistence"), f"pairedRows[{i}].persistence")
        date_key = (product, target)
        if date_key in national_by_date:
            _same_actual(national_by_date[date_key], national, f"{product}/{target}/national")
            _same_actual(local_by_date[date_key], local, f"{product}/{target}/two-market")
        else:
            national_by_date[date_key] = national
            local_by_date[date_key] = local
        audit_actual = national_history.get(date_key)
        if audit_actual is None:
            raise ValueError(f"paired national actual has no positive targetAudit value: {product}/{target}")
        _same_actual(audit_actual, national, f"{product}/{target}/targetAudit")
        validated_rows.append({"product": product, "origin": origin, "target": target, "horizon": horizon,
                               "nationalActual": national, "twoMarketActual": local,
                               "candidatePrediction": candidate, "directRidge": ridge, "persistence": persistence})
    return national_history, validated_rows


def _anchor(product, origin, national_history, local_history):
    first = origin - dt.timedelta(days=ANCHOR_START_OFFSET)
    last = origin - dt.timedelta(days=ANCHOR_END_OFFSET)
    ratios = []
    anchor_dates = []
    for day in sorted(set(national_history) & set(local_history)):
        if day[0] != product:
            continue
        date = day[1]
        if first <= date <= last:
            national, local = national_history[day], local_history[day]
            if national > 0 and local > 0:
                ratios.append(national / local)
                anchor_dates.append(date)
    if len(ratios) < MIN_ANCHOR_DAYS:
        return {"product": product, "origin": origin.isoformat(), "windowStart": first.isoformat(),
                "windowEnd": last.isoformat(), "anchorDates": [d.isoformat() for d in anchor_dates],
                "eligibleDateCount": len(ratios), "ratio": 1.0, "fallback": True,
                "fallbackReason": "fewer_than_3_eligible_anchor_dates"}
    return {"product": product, "origin": origin.isoformat(), "windowStart": first.isoformat(),
            "windowEnd": last.isoformat(), "anchorDates": [d.isoformat() for d in anchor_dates],
            "eligibleDateCount": len(ratios), "ratio": float(statistics.median(ratios)),
            "fallback": False, "fallbackReason": None}


def _loss_difference(actual, candidate, comparator, loss):
    if loss == "mae":
        return abs(actual - candidate) - abs(actual - comparator)
    return 100.0 * (abs(actual - candidate) - abs(actual - comparator)) / actual


def _season(day):
    if day.month in (12, 1, 2):
        return "winter"
    if day.month in (3, 4, 5):
        return "spring"
    if day.month in (6, 7, 8):
        return "summer"
    return "autumn"


def build_assessment(report, *, report_sha256=None):
    """Build the fixed paired level-anchor evaluation from a parsed report."""
    national_history, source_rows = _validate_report(report)
    local_history = {}
    for row in source_rows:
        key = (row["product"], row["target"])
        value = row["twoMarketActual"]
        if key in local_history:
            _same_actual(local_history[key], value, f"{key[0]}/{key[1]}/two-market")
        else:
            local_history[key] = value

    eval_rows = [row for row in source_rows if EVAL_START <= row["target"] <= EVAL_END]
    anchor_cache = {}
    anchor_rows = []
    paired_rows = []
    fallback_count = 0
    fallback_reasons = {}
    for row in eval_rows:
        anchor_key = (row["product"], row["origin"])
        if anchor_key not in anchor_cache:
            anchor_cache[anchor_key] = _anchor(row["product"], row["origin"], national_history, local_history)
        anchor = anchor_cache[anchor_key]
        ratio = anchor["ratio"]
        output_row = {"product": row["product"], "origin": row["origin"].isoformat(), "target": row["target"].isoformat(),
            "horizon": row["horizon"], "nationalActual": row["nationalActual"], "twoMarketActual": row["twoMarketActual"],
            "anchorRatio": ratio, "anchorDateCount": anchor["eligibleDateCount"], "anchorDates": anchor["anchorDates"],
            "anchorFallback": anchor["fallback"], "anchorFallbackReason": anchor["fallbackReason"],
            "originalCandidate": row["candidatePrediction"], "candidatePrediction": row["candidatePrediction"] * ratio,
            "directRidge": row["directRidge"], "scaledDirectRidge": row["directRidge"] * ratio,
            "persistence": row["persistence"], "scaledPersistence": row["persistence"] * ratio}
        paired_rows.append(output_row)
        if anchor["fallback"]:
            fallback_count += 1
            reason = anchor["fallbackReason"]
            fallback_reasons[reason] = fallback_reasons.get(reason, 0) + 1
    for key in sorted(anchor_cache):
        anchor_rows.append(anchor_cache[key])

    comparators = (("originalCandidate", "originalCandidate"), ("scaledDirectRidge", "scaledDirectRidge"),
                   ("scaledPersistence", "scaledPersistence"))
    comparisons = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            rows = [r for r in paired_rows if r["product"] == product and r["horizon"] == horizon]
            actual = [r["nationalActual"] for r in rows]
            candidate = [r["candidatePrediction"] for r in rows]
            for comparator, field in comparators:
                baseline = [r[field] for r in rows]
                for loss in ("mae", "mape"):
                    diff = {_date(r["target"], "paired target"): _loss_difference(r["nationalActual"], r["candidatePrediction"], r[field], loss) for r in rows}
                    metric = "mae" if loss == "mae" else "mape"
                    comparisons.append({"product": product, "horizon": horizon, "comparator": comparator,
                        "loss": loss, "n": len(rows), "candidateMetric": dh._metrics(actual, candidate)[metric],
                        "comparatorMetric": dh._metrics(actual, baseline)[metric],
                        "dm": {str(bw): dh._dm(diff, bw) for bw in HAC}})
    if len(comparisons) != FAMILY_SIZE:
        raise AssertionError(f"frozen national-anchor DM family must contain {FAMILY_SIZE} comparisons")
    for bw in HAC:
        dh._holm([r["dm"][str(bw)] for r in comparisons])

    coverage = {}
    conclusions = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            rows = [r for r in paired_rows if r["product"] == product and r["horizon"] == horizon]
            issued = len(rows)
            coverage[f"{product}/{horizon}"] = {"eligibleCommonPairs": issued, "candidateIssued": issued,
                "originalCandidateIssued": issued, "scaledDirectRidgeIssued": issued, "scaledPersistenceIssued": issued,
                "candidateCoverageLoss": {name: 0 for name, _ in comparators},
                "anchorFallbackCount": sum(r["anchorFallback"] for r in rows)}
            tests = [r for r in comparisons if r["product"] == product and r["horizon"] == horizon]
            inference = all(r["n"] >= MIN_PAIRS and r["dm"]["9"]["status"] == "tested" for r in tests)
            metrics_better = all(r["candidateMetric"] is not None and r["comparatorMetric"] is not None and r["candidateMetric"] < r["comparatorMetric"] for r in tests)
            tests_better = all(r["dm"]["9"].get("holmSignificant") and r["dm"]["9"].get("meanLossDifference") is not None and r["dm"]["9"]["meanLossDifference"] < 0 for r in tests)
            adverse = any(r["dm"]["9"].get("holmSignificant") and r["dm"]["9"].get("meanLossDifference", 0) > 0 for r in tests)
            conclusion = "inference_unavailable" if not inference else "improvement" if metrics_better and tests_better else "worsening_or_mixed" if adverse else "inconclusive"
            conclusions.append({"product": product, "horizon": horizon, "conclusion": conclusion,
                "inferenceAvailable": inference, "allSixPointMetricsImprove": metrics_better,
                "allSixHac9HolmTestsImprove": tests_better, "coverageNondeclining": True})

    strata = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            rows = [r for r in paired_rows if r["product"] == product and r["horizon"] == horizon]
            for stratum, fn in (("year", lambda d: str(d.year)), ("season", _season)):
                groups = {}
                for row in rows:
                    target = _date(row["target"], "paired target")
                    groups.setdefault(fn(target), []).append(row)
                for label, group in sorted(groups.items()):
                    strata.append({"product": product, "horizon": horizon, "stratum": stratum, "value": label, "n": len(group),
                        **{name: dh._metrics([r["nationalActual"] for r in group], [r[field] for r in group])
                           for name, field in (("candidate", "candidatePrediction"), ("originalCandidate", "originalCandidate"),
                                               ("scaledDirectRidge", "scaledDirectRidge"), ("scaledPersistence", "scaledPersistence"))}})

    macro = {}
    for product in PRODUCTS:
        rows_by_h = {h: [r for r in paired_rows if r["product"] == product and r["horizon"] == h] for h in HORIZONS}
        mape = {str(h): dh._metrics([r["nationalActual"] for r in rows_by_h[h]], [r["candidatePrediction"] for r in rows_by_h[h]])["mape"] for h in HORIZONS}
        macro[product] = {"equalHorizonMape": sum(mape.values()) / len(HORIZONS) if all(v is not None for v in mape.values()) else None,
                          "mapeByHorizon": mape, "pairedCountsByHorizon": {str(h): len(rows_by_h[h]) for h in HORIZONS},
                          "pointEstimateAvailable": all(v is not None for v in mape.values()),
                          "interpretation": "descriptive_point_estimate_not_equivalence_or_success_evidence"}

    return {"schemaVersion": 1, "status": "complete", "protocol": PROTOCOL,
        "hashes": {"nationalTransferReportSha256": report_sha256}, "coverage": coverage,
        "fallback": {"rowCount": fallback_count, "originCount": sum(row["fallback"] for row in anchor_rows),
                     "reasonCounts": fallback_reasons,
                     "reason": "fewer_than_3_eligible_anchor_dates"},
        "originAnchors": anchor_rows, "pairedRows": paired_rows, "comparisons": comparisons,
        "jointConclusions": conclusions, "yearSeasonMetrics": strata, "macro10EqualHorizonMape": macro,
        "limitations": PROTOCOL["limitations"]}


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_new(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def evaluate_bytes(data: bytes, *, expected_sha256: str | None = REPORT_SHA256):
    digest = _sha(data)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("national-transfer report SHA-256 differs from frozen protocol")
    try:
        def unique_pairs(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"duplicate JSON key: {key}")
                result[key] = value
            return result
        report = json.loads(data, object_pairs_hook=unique_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("national-transfer report is invalid JSON") from exc
    return build_assessment(report, report_sha256=digest)


def run(report_path=INPUT_REPORT, *, output_dir=OUTPUT_DIR):
    data = Path(report_path).read_bytes()
    result = evaluate_bytes(data)
    output = Path(output_dir) / "report.json"
    payload = _json_bytes(result)
    _write_new(output, payload)
    return {"output": str(output), "sha256": _sha(payload), "pairedRows": len(result["pairedRows"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=INPUT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.report, output_dir=args.output_dir), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
