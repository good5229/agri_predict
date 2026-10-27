"""Immutable read-only availability screen for the frozen target-aligned pilot."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import dashboard, daily_training, supply_pressure_prospective as prospective

START = dt.date(2017, 1, 1)
END = dt.date(2024, 12, 31)
ORIGINS = tuple(dt.date(2024, 8, day) for day in range(15, 29)) + tuple(
    dt.date(2024, 10, day) for day in range(1, 15)
)
HORIZONS = (1, 5, 10)
FROZEN_CUTOFF = dt.date(2026, 10, 6)
MIN_PAIRED = 10
MIN_CONDITIONAL_COVERAGE = 0.90


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_inputs(prices_path: Path, status_path: Path):
    """Validate the frozen completed snapshot and its exact source provenance."""
    pb, sb = prices_path.read_bytes(), status_path.read_bytes()
    status = json.loads(sb)
    cutoff = dt.date.fromisoformat(status["cutoff"])
    if cutoff != FROZEN_CUTOFF:
        raise ValueError("target screen requires the frozen 2026-10-06 snapshot cutoff")
    prices, products = prospective.validate_snapshot(pb, sb, dt.date(2026, 10, 8), cutoff)
    run_dir = Path(status["runDirectory"])
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    run_dir = run_dir.resolve()
    if prices_path.resolve().parent != run_dir or status_path.resolve().parent != run_dir:
        raise ValueError("prices and run-status must be from the declared same run directory")
    provenance_path = Path(status["selectedProvenance"])
    if not provenance_path.is_absolute():
        provenance_path = Path.cwd() / provenance_path
    if provenance_path.resolve().parent != run_dir:
        raise ValueError("selected provenance must be from the declared same run directory")
    provenance_bytes = provenance_path.read_bytes()
    provenance = json.loads(provenance_bytes)
    if not isinstance(provenance, list):
        raise ValueError("selected provenance is invalid")
    seen, finished = set(), prospective._parse_dt(status["finishedAtKst"])
    for row in provenance:
        if not isinstance(row, dict):
            raise ValueError("selected provenance row is invalid")
        day = dt.date.fromisoformat(row.get("date", ""))
        response = prospective._parse_dt(row.get("responseAt", ""))
        if day > cutoff or response > finished or response.astimezone(prospective.KST).date() >= dt.date(2026, 10, 8):
            raise ValueError("selected provenance exceeds cutoff or run timing")
        scope = (row.get("date"), row.get("market"), row.get("product"))
        if None in scope or scope in seen:
            raise ValueError("selected provenance has missing or duplicate scope")
        seen.add(scope)
    expected = {(day.isoformat(), market, product)
                for i in range((cutoff - daily_training.CONTEXT_START).days + 1)
                for day in (daily_training.CONTEXT_START + dt.timedelta(days=i),)
                for market, product in daily_training.EXPECTED_SCOPES}
    if seen != expected:
        raise ValueError("selected provenance does not cover every date and market/product scope through cutoff")
    if daily_training._pipeline_fingerprint(prices, provenance) != status.get("snapshotSha256"):
        raise ValueError("run-status fingerprint does not match prices and selected provenance")
    return products, status, provenance_path, pb, sb, provenance_bytes


def _failure_reason(series: dict[dt.date, tuple[float, float]], origin: dt.date,
                    target: dt.date, horizon: int) -> str:
    """Classify why the frozen feature builder cannot form an origin row."""
    from bisect import bisect_left

    days = tuple(sorted(series))
    hi = bisect_left(days, origin)
    if not hi or (origin - days[hi - 1]).days > dashboard.MAX_SOURCE_AGE_DAYS:
        return "missing_past_price_or_volume"
    for window in (7, 28):
        idx = bisect_left(days, origin - dt.timedelta(days=window), 0, hi) - 1
        if idx < 0 or (origin - dt.timedelta(days=window) - days[idx]).days > 14:
            return "missing_past_history"
        eligible = days[bisect_left(days, origin - dt.timedelta(days=window), 0, hi):hi]
        if not eligible or not any(series[d][1] > 0 for d in eligible):
            return "missing_past_price_or_volume"
    seasonal = prospective._seasonal(series, target, origin)
    if seasonal is None or seasonal <= 0:
        return "seasonal_3_year_failure"
    return "feature_unavailable"


def build(prices_path: Path, status_path: Path, output: Path) -> dict[str, Any]:
    products, status, provenance_path, pb, sb, provb = validate_inputs(prices_path, status_path)
    rows = []
    for pid in dashboard.PRODUCTS:
        raw_series = prospective._series(products[pid])
        series = {d: value for d, value in raw_series.items() if START <= d <= END}
        for origin in ORIGINS:
            for horizon in HORIZONS:
                target = origin + dt.timedelta(days=horizon)
                feature = prospective._feature(series, origin, target, horizon)
                target_value = series.get(target)
                target_present = target_value is not None and math.isfinite(target_value[0]) and target_value[0] > 0
                rows.append({
                    "productId": pid, "origin": origin.isoformat(), "horizon": horizon,
                    "target": target.isoformat(), "targetStatus": "present" if target_present else "missing_observation",
                    "featureStatus": "available" if feature is not None else "unavailable",
                    "featureFailureReason": None if feature is not None else _failure_reason(series, origin, target, horizon),
                    "pairedEvaluable": bool(target_present and feature is not None),
                })

    product_summaries = []
    for pid in dashboard.PRODUCTS:
        subset = [r for r in rows if r["productId"] == pid]
        target_count = sum(r["targetStatus"] == "present" for r in subset)
        paired = sum(r["pairedEvaluable"] for r in subset)
        conditional = paired / target_count if target_count else None
        by_horizon = []
        for horizon in HORIZONS:
            hrows = [r for r in subset if r["horizon"] == horizon]
            htargets = sum(r["targetStatus"] == "present" for r in hrows)
            hpairs = sum(r["pairedEvaluable"] for r in hrows)
            by_horizon.append({"horizon": horizon, "planned": len(hrows),
                               "targetPresent": htargets,
                               "featureAvailable": sum(r["featureStatus"] == "available" for r in hrows),
                               "pairedEvaluable": hpairs,
                               "conditionalFeatureCoverage": hpairs / htargets if htargets else None,
                               "conditionalCoveragePassed": htargets > 0 and hpairs / htargets >= MIN_CONDITIONAL_COVERAGE,
                               "minimumPairedPassed": hpairs >= MIN_PAIRED})
        product_summaries.append({"productId": pid, "planned": len(subset),
                                  "targetPresent": target_count,
                                  "featureAvailable": sum(r["featureStatus"] == "available" for r in subset),
                                  "pairedEvaluable": paired,
                                  "conditionalFeatureCoverage": conditional,
                                  "conditionalCoveragePassed": all(h["conditionalCoveragePassed"] for h in by_horizon),
                                  "byHorizon": by_horizon})
    all_pairs_gate = all(h["pairedEvaluable"] >= MIN_PAIRED
                         for p in product_summaries for h in p["byHorizon"])
    all_coverage_gate = all(h["conditionalCoveragePassed"]
                            for p in product_summaries for h in p["byHorizon"])
    payload = {"schemaVersion": 1, "experiment": "supply_pressure_target_aligned_availability_screen",
               "inputWindow": {"start": START.isoformat(), "end": END.isoformat()},
               "origins": [d.isoformat() for d in ORIGINS], "horizons": HORIZONS,
               "plannedRows": len(rows), "minimumPairedPerProductHorizon": MIN_PAIRED,
               "minimumConditionalFeatureCoverage": MIN_CONDITIONAL_COVERAGE,
               "gate": {"allProductHorizonsHaveMinimumPaired": all_pairs_gate,
                        "allProductsMeetConditionalCoverage": all_coverage_gate,
                        "passed": all_pairs_gate and all_coverage_gate},
               "noModelFit": True, "mapeComputed": False,
               "input": {"pricesPath": str(prices_path.resolve()), "pricesSha256": _sha(pb),
                         "runStatusPath": str(status_path.resolve()), "runStatusSha256": _sha(sb),
                         "snapshotFingerprint": status["snapshotSha256"],
                         "provenancePath": str(provenance_path.resolve()), "provenanceSha256": _sha(provb)},
               "codeSha256": _sha(Path(__file__).read_bytes()),
               "products": product_summaries, "rows": rows}
    prospective._write_immutable(output, json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                                    indent=2, allow_nan=False) + "\n")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices", type=Path, required=True)
    parser.add_argument("--run-status", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    build(args.prices, args.run_status, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
