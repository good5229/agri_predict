"""Frozen descriptive screen for conditional lead signal in signed national volume.

This is a retrospective, offline screening protocol. It does not train or score a
forecasting model and does not fetch data.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from . import national_transfer

ROOT = Path(__file__).resolve().parents[1]
INPUT_ARCHIVE = ROOT / "data/raw/national-wholesale-20240708-20260630.zip"
OUTPUT_PATH = ROOT / "outputs/experiments/national-signed-volume-screen-20261009/report.json"
ARCHIVE_SHA256 = national_transfer.ARCHIVE_SHA256
PRODUCTS = tuple(national_transfer.PRODUCT_CODES)
PRODUCT_NAMES = national_transfer.PRODUCT_NAMES
HORIZONS = (5, 10)
TARGET_START = dt.date(2024, 9, 1)
TARGET_END = dt.date(2026, 6, 30)
BLOCKS = (
    (dt.date(2024, 9, 1), dt.date(2025, 3, 31)),
    (dt.date(2025, 4, 1), dt.date(2025, 9, 30)),
    (dt.date(2025, 10, 1), dt.date(2026, 3, 31)),
    (dt.date(2026, 4, 1), dt.date(2026, 6, 30)),
)
MIN_RECENT_DAYS = 4
MIN_PRIOR_DAYS = 12
MIN_PAIRS = 30
MIN_COVERAGE = 0.70
MIN_NEGATIVE_BLOCKS = 3
NEGATIVE_CUTOFF = -0.10
POSITIVE_CEILING = 0.05
PROTOCOL_ID = "national-signed-volume-conditional-lead-screen-2026-10-09"
PROTOCOL = {
    "id": PROTOCOL_ID,
    "inputArchiveSha256": ARCHIVE_SHA256,
    "input": "national_transfer.parse_archive; four immutable period CSVs; signed amount and quantity retained; no forward fill",
    "targets": [TARGET_START.isoformat(), TARGET_END.isoformat()],
    "horizonsCalendarDays": list(HORIZONS),
    "origin": "target minus horizon; cutoff = origin minus 2 calendar days",
    "priceReferences": "p0 is latest positive price on/before cutoff, at most 14 days old; p7 is latest positive price on/before cutoff-7, at most 14 days old relative to cutoff-7",
    "volumeWindows": {"recent": "origin-8 through origin-2 inclusive; at least 4 valid observed days", "prior": "origin-29 through origin-9 inclusive; at least 12 valid observed days", "aggregation": "signed quantity average per observed valid trading day; no missing-date fill"},
    "features": {"x": "log(recent signed-volume mean / prior signed-volume mean), both means positive finite", "z": "log(p0 / p7)", "y": "log(target positive price / p0)"},
    "conditionalSpearman": "midranks of x,y,z within each product/horizon/block; regress x and y ranks separately on intercept, z rank and target calendar-month fixed effects; Pearson correlation of residuals",
    "blocks": [[a.isoformat(), b.isoformat()] for a, b in BLOCKS],
    "threshold": {"eachBlockMinimumPairs": MIN_PAIRS, "eachBlockMinimumCoverage": MIN_COVERAGE, "coverageDenominator": "positive-price target observation days in product/block", "minimumBlocksCoefficientAtMost": {"count": MIN_NEGATIVE_BLOCKS, "coefficient": NEGATIVE_CUTOFF, "mustIncludeLatestBlock": True}, "noBlockCoefficientAbove": POSITIVE_CEILING},
    "interpretation": "A pass only retains a feature for future model review; it is not DM inference, predictive-performance evidence, or a causal supply effect. Historical first-publication time is unknown.",
}


def _day(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def _positive(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _usable_days(parsed: dict, product: str) -> dict[dt.date, dict]:
    result = {}
    for text, row in parsed["observations"].get(product, {}).items():
        price = _positive(row.get("price"))
        try:
            quantity = float(row["quantitySum"])
        except (KeyError, TypeError, ValueError, OverflowError):
            quantity = math.nan
        if (row.get("priceReason") is None and row.get("productNames") == [PRODUCT_NAMES[product]]
                and row.get("nameConflictCount") == 0 and price is not None and math.isfinite(quantity)):
            result[_day(text)] = {**row, "_price": price, "_quantity": quantity}
    return result


def _latest(days: dict[dt.date, dict], end: dt.date, max_age: int) -> tuple[dt.date, dict] | None:
    candidates = [day for day in days if day <= end and (end - day).days <= max_age]
    if not candidates:
        return None
    day = max(candidates)
    return day, days[day]


def _range_stats(days: dict[dt.date, dict], start: dt.date, end: dt.date) -> dict:
    selected = [(d, r) for d, r in days.items() if start <= d <= end]
    negative_days = sum(r["_quantity"] < 0 for _, r in selected)
    return {
        "start": start.isoformat(), "end": end.isoformat(), "observedDays": len(selected),
        "negativeSignedQuantityDays": negative_days,
        "negativeQuantitySourceRows": sum(int(r.get("negativeQuantityCount", 0)) for _, r in selected),
        "signedQuantityMean": float(np.mean([r["_quantity"] for _, r in selected])) if selected else None,
        "dates": [d.isoformat() for d, _ in sorted(selected)],
    }


def _make_pair(days: dict[dt.date, dict], product: str, target: dt.date, horizon: int) -> dict:
    origin = target - dt.timedelta(days=horizon)
    cutoff = origin - dt.timedelta(days=2)
    recent_start, recent_end = origin - dt.timedelta(days=8), origin - dt.timedelta(days=2)
    prior_start, prior_end = origin - dt.timedelta(days=29), origin - dt.timedelta(days=9)
    row = {"product": product, "target": target.isoformat(), "horizon": horizon,
           "origin": origin.isoformat(), "cutoff": cutoff.isoformat(), "state": None}
    target_row = days.get(target)
    p0 = _latest(days, cutoff, 14)
    p7_reference = cutoff - dt.timedelta(days=7)
    p7 = _latest(days, p7_reference, 14)
    recent, prior = _range_stats(days, recent_start, recent_end), _range_stats(days, prior_start, prior_end)
    row.update({"targetPrice": target_row["_price"] if target_row else None,
                "p0": {"date": p0[0].isoformat(), "price": p0[1]["_price"]} if p0 else None,
                "p7ReferenceDate": p7_reference.isoformat(),
                "p7": {"date": p7[0].isoformat(), "price": p7[1]["_price"]} if p7 else None,
                "recentWindow": recent, "priorWindow": prior})
    negatives = {d for d, r in days.items() if r["_quantity"] < 0}
    row["negativeSignedQuantityDaysInFeatureWindows"] = len(negatives.intersection(
        {d for d in days if recent_start <= d <= recent_end or prior_start <= d <= prior_end}))
    if target_row is None:
        row["state"] = "target_price_unavailable"
        return row
    if p0 is None:
        row["state"] = "p0_unavailable_or_stale"
        return row
    if p7 is None:
        row["state"] = "p7_unavailable_or_stale"
        return row
    if recent["observedDays"] < MIN_RECENT_DAYS:
        row["state"] = "recent_window_below_minimum_days"
        return row
    if prior["observedDays"] < MIN_PRIOR_DAYS:
        row["state"] = "prior_window_below_minimum_days"
        return row
    recent_mean, prior_mean = recent["signedQuantityMean"], prior["signedQuantityMean"]
    if recent_mean is None or prior_mean is None or recent_mean <= 0 or prior_mean <= 0:
        row["state"] = "nonpositive_signed_window_mean"
        return row
    x = math.log(recent_mean / prior_mean)
    z = math.log(p0[1]["_price"] / p7[1]["_price"])
    y = math.log(target_row["_price"] / p0[1]["_price"])
    if not all(math.isfinite(v) for v in (x, y, z)):
        row["state"] = "nonfinite_feature"
        return row
    row.update({"state": "paired", "x": x, "y": y, "z": z,
                "recentSignedMean": recent_mean, "priorSignedMean": prior_mean})
    return row


def _midranks(values: list[float]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    return ranks


def _conditional_spearman(rows: list[dict]) -> dict:
    n = len(rows)
    if n < MIN_PAIRS:
        return {"status": "insufficient_pairs", "n": n, "coefficient": None}
    rx, ry, rz = (_midranks([r[k] for r in rows]) for k in ("x", "y", "z"))
    months = [_day(r["target"]).month for r in rows]
    month_levels = sorted(set(months))
    columns = [np.ones(n), rz]
    columns.extend(np.asarray([float(m == level) for m in months]) for level in month_levels[1:])
    design = np.column_stack(columns)
    residual_x = rx - design @ np.linalg.lstsq(design, rx, rcond=None)[0]
    residual_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    sx, sy = float(np.std(residual_x)), float(np.std(residual_y))
    # Rank values are O(n); least-squares residuals for an exactly explained
    # rank vector can still contain roundoff at roughly eps * n * rank-scale.
    rank_scale = max(float(np.std(rx)), float(np.std(ry)), float(np.std(rz)), 1.0)
    zero_tolerance = np.finfo(float).eps * max(design.shape) * rank_scale * 16.0
    if sx <= zero_tolerance or sy <= zero_tolerance or not math.isfinite(sx * sy):
        return {"status": "zero_residual_variance", "n": n, "coefficient": None,
                "residualStdX": sx, "residualStdY": sy, "residualZeroTolerance": zero_tolerance,
                "monthLevels": month_levels}
    coefficient = float(np.corrcoef(residual_x, residual_y)[0, 1])
    return {"status": "computed", "n": n, "coefficient": coefficient,
            "residualStdX": sx, "residualStdY": sy, "monthLevels": month_levels,
            "controlColumns": ["intercept", "zMidrank", *[f"targetMonth={m}" for m in month_levels[1:]]]}


def _block_pass(blocks: list[dict]) -> tuple[bool, dict]:
    latest_key = BLOCKS[-1][0].isoformat()
    all_availability = all(b["pairCount"] >= MIN_PAIRS and b["coverage"] is not None and b["coverage"] >= MIN_COVERAGE for b in blocks)
    negative_blocks = [b for b in blocks if b["conditionalSpearman"]["coefficient"] is not None
                       and b["conditionalSpearman"]["coefficient"] <= NEGATIVE_CUTOFF]
    latest_included = any(b["blockStart"] == latest_key for b in negative_blocks)
    no_large_positive = all(b["conditionalSpearman"]["coefficient"] is not None
                             and b["conditionalSpearman"]["coefficient"] <= POSITIVE_CEILING for b in blocks)
    criteria = {"allBlocksAtLeast30PairsAnd70pctCoverage": all_availability,
                "atLeastThreeNegativeBlocksIncludingLatest": len(negative_blocks) >= MIN_NEGATIVE_BLOCKS and latest_included,
                "noBlockCoefficientAbovePlus0_05": no_large_positive}
    return all(criteria.values()), {"criteria": criteria, "negativeBlockCount": len(negative_blocks),
                                    "latestBlockIncluded": latest_included}


def screen(parsed: dict) -> dict:
    observations = parsed.get("observations")
    if not isinstance(observations, dict) or set(observations) != set(PRODUCTS):
        raise ValueError("parsed national archive has incomplete product coverage")
    cell_reports = {}
    for product in PRODUCTS:
        days = _usable_days(parsed, product)
        for horizon in HORIZONS:
            block_reports = []
            for start, end in BLOCKS:
                targets = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
                positive_targets = [d for d in targets if d in days]
                candidates = [_make_pair(days, product, target, horizon) for target in targets]
                paired = [r for r in candidates if r["state"] == "paired"]
                result = _conditional_spearman(paired)
                coverage = len(paired) / len(positive_targets) if positive_targets else None
                state_counts: dict[str, int] = {}
                for row in candidates:
                    state_counts[row["state"]] = state_counts.get(row["state"], 0) + 1
                negative_days = set()
                for row in candidates:
                    for window in (row["recentWindow"], row["priorWindow"]):
                        negative_days.update(d for d in window["dates"]
                                             if days[_day(d)]["_quantity"] < 0)
                block = {"blockStart": start.isoformat(), "blockEnd": end.isoformat(),
                         "targetPositivePriceDays": len(positive_targets), "pairCount": len(paired),
                         "coverage": coverage, "stateCounts": dict(sorted(state_counts.items())),
                         "negativeSignedQuantityDaysInCandidateWindows": len(negative_days),
                         "conditionalSpearman": result}
                block_reports.append(block)
            passes, details = _block_pass(block_reports)
            cell_reports[f"{product}/h{horizon}"] = {"blocks": block_reports,
                "retainForFutureModelReview": passes, "thresholdAssessment": details}
    report = {"protocolId": PROTOCOL_ID, "protocol": PROTOCOL,
              "source": {"archiveSha256": parsed.get("archiveSha256"), "archiveByteCount": parsed.get("archiveByteCount"),
                         "sourceRowCount": parsed.get("sourceRowCount"), "csvCount": parsed.get("csvCount"),
                         "exactDuplicateRowCount": parsed.get("exactDuplicateRowCount"), "periods": parsed.get("periods")},
              "productAudit": {p: {"observedDays": len(parsed["observations"][p]),
                    "invalidPriceStateCounts": _count_invalid(parsed["observations"][p]),
                    "negativeSignedQuantityDays": sum(_is_negative_quantity(r) for r in parsed["observations"][p].values()),
                    "negativeQuantitySourceRows": sum(int(r.get("negativeQuantityCount", 0)) for r in parsed["observations"][p].values())}
                    for p in PRODUCTS},
              "cells": cell_reports,
              "interpretation": "Retrospective descriptive screening only. First-publication timing is unknown; same-source aggregate volume and price share transactions. Passing only permits a separately frozen future model review; it is not predictive or causal evidence."}
    report["reportSha256"] = _hash(report)
    return report


def _count_invalid(rows: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows.values():
        state = row.get("priceReason") or "valid_positive_price"
        counts[state] = counts.get(state, 0) + 1
    return dict(sorted(counts.items()))


def _is_negative_quantity(row: dict) -> bool:
    try:
        return float(row["quantitySum"]) < 0
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _hash(document: dict) -> str:
    payload = json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def build_report(archive_bytes: bytes, *, expected_sha256: str | None = ARCHIVE_SHA256) -> dict:
    parsed = national_transfer.parse_archive(archive_bytes, expected_sha256=expected_sha256)
    return screen(parsed)


def write_create_only(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(report, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="parse and screen the pinned archive (offline only)")
    build.add_argument("--archive", type=Path, default=INPUT_ARCHIVE)
    build.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    if args.command == "build":
        report = build_report(args.archive.read_bytes())
        write_create_only(args.output, report)
        print(json.dumps({"output": str(args.output), "reportSha256": report["reportSha256"],
                          "cellCount": len(report["cells"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
