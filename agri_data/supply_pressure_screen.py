"""Immutable stage-0 availability screen for the frozen supply-pressure features.

This module counts feature availability only. It never fits a model or evaluates
target outcomes.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from . import dashboard, daily_training, supply_pressure_prospective as prospective

START = dt.date(2017, 1, 1)
END = dt.date(2024, 12, 31)
ORIGINS = tuple(dt.date(2024, month, day) for month in (1, 4, 7, 10) for day in range(1, 15))
HORIZONS = (1, 5, 10)
MIN_COVERAGE = 0.90
FROZEN_CUTOFF = dt.date(2026, 10, 6)


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_inputs(prices_path: Path, status_path: Path):
    """Validate run identity, completeness, cutoff, provenance, and fingerprint."""
    pb, sb = prices_path.read_bytes(), status_path.read_bytes()
    prices, status = json.loads(pb), json.loads(sb)
    cutoff = dt.date.fromisoformat(status["cutoff"])
    if cutoff != FROZEN_CUTOFF:
        raise ValueError("stage-0 screen requires the frozen 2026-10-06 snapshot cutoff")
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
    return prices, products, status, provenance_path, pb, sb, provenance_bytes


def _failure_reason(series: dict[dt.date, tuple[float, float]], origin: dt.date,
                    target: dt.date, horizon: int) -> str:
    """Separate historical price/volume sufficiency from seasonal-year sufficiency."""
    days = tuple(sorted(series))
    import bisect
    hi = bisect.bisect_left(days, origin)
    if not hi or (origin - days[hi - 1]).days > dashboard.MAX_SOURCE_AGE_DAYS:
        return "missing_past_price_or_volume"
    for window in (7, 28):
        idx = bisect.bisect_left(days, origin - dt.timedelta(days=window), 0, hi) - 1
        if idx < 0 or (origin - dt.timedelta(days=window) - days[idx]).days > 14:
            return "missing_past_history"
        eligible = days[bisect.bisect_left(days, origin - dt.timedelta(days=window), 0, hi):hi]
        if not eligible or not any(series[d][1] > 0 for d in eligible):
            return "missing_past_price_or_volume"
    seasonal = prospective._seasonal(series, target, origin)
    if seasonal is None or seasonal <= 0:
        return "seasonal_3_year_failure"
    return "feature_unavailable"


def build(prices_path: Path, status_path: Path, output: Path) -> dict[str, Any]:
    prices, products, status, provenance_path, pb, sb, provb = validate_inputs(prices_path, status_path)
    rows = []
    for pid in dashboard.PRODUCTS:
        raw_series = prospective._series(products[pid])
        # Freeze the screen's observation window before any feature calculation.
        series = {d: value for d, value in raw_series.items() if START <= d <= END}
        for origin in ORIGINS:
            for horizon in HORIZONS:
                target = origin + dt.timedelta(days=horizon)
                result = prospective._feature(series, origin, target, horizon)
                rows.append({"productId": pid, "origin": origin.isoformat(), "horizon": horizon,
                             "target": target.isoformat(), "status": "available" if result else "unavailable",
                             "reason": None if result else _failure_reason(series, origin, target, horizon)})
    product_summaries = []
    for pid in dashboard.PRODUCTS:
        subset = [r for r in rows if r["productId"] == pid]
        available = sum(r["status"] == "available" for r in subset)
        product_summaries.append({"productId": pid, "planned": len(subset), "available": available,
                                  "coverage": available / len(subset), "gatePassed": available / len(subset) >= MIN_COVERAGE,
                                  "byHorizon": [{"horizon": h, "planned": 56,
                                                 "available": sum(r["status"] == "available" and r["horizon"] == h for r in subset)}
                                                for h in HORIZONS]})
    payload = {"schemaVersion": 1, "experiment": "supply_pressure_stage0_availability_screen",
               "inputWindow": {"start": START.isoformat(), "end": END.isoformat()},
               "origins": [d.isoformat() for d in ORIGINS], "horizons": HORIZONS,
               "plannedRows": len(rows), "minimumCoverage": MIN_COVERAGE,
               "noModelFit": True, "mapeComputed": False,
               "input": {"pricesPath": str(prices_path.resolve()), "pricesSha256": _sha(pb),
                         "runStatusPath": str(status_path.resolve()), "runStatusSha256": _sha(sb),
                         "snapshotFingerprint": status["snapshotSha256"],
                         "provenancePath": str(provenance_path.resolve()), "provenanceSha256": _sha(provb)},
               "codeSha256": _sha(Path(__file__).read_bytes()), "products": product_summaries, "rows": rows}
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
