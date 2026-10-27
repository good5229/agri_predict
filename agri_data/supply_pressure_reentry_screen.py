"""Immutable read-only screen for exploratory 28-day history reentry features."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import dashboard, modeling, supply_pressure_prospective as prospective
from . import supply_pressure_target_screen as target_screen

START = target_screen.START
END = target_screen.END
ORIGINS = target_screen.ORIGINS
HORIZONS = target_screen.HORIZONS
FROZEN_CUTOFF = target_screen.FROZEN_CUTOFF
MIN_PAIRED = 10
MIN_CONDITIONAL_COVERAGE = 0.90
MIN_PEPPER_REENTRY_TRAIN_TARGETS = 30
PEPPER_ID = "pepper"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _log_mean_volume7(series: dict[dt.date, tuple[float, float]], origin: dt.date) -> float | None:
    """Return log1p mean positive volume over observed pre-origin 7-day rows."""
    lo, hi = origin - dt.timedelta(days=7), origin
    eligible = [value[1] for day, value in series.items() if lo <= day < hi and value[1] > 0]
    if not eligible:
        return None
    mean = sum(eligible) / len(eligible)
    if not math.isfinite(mean) or mean <= 0:
        return None
    result = math.log1p(mean)
    return result if math.isfinite(result) else None


def _reentry_feature(series: dict[dt.date, tuple[float, float]], origin: dt.date,
                     target: dt.date, horizon: int) -> tuple[tuple[float, ...], dt.date] | None:
    """Build frozen features only when the full builder fails at its 28-day checks.

    The 13-value feature order remains prospective.FEATURES. The four fields
    specific to 28-day history are neutralized to zero; callers append a 1
    missing-history indicator.
    """
    days = tuple(sorted(series))
    hi = bisect_left(days, origin)
    if not hi:
        return None
    src = days[hi - 1]
    if (origin - src).days > dashboard.MAX_SOURCE_AGE_DAYS:
        return None

    refs, eligible_by_window = {}, {}
    for window in (7, 28):
        lo = bisect_left(days, origin - dt.timedelta(days=window), 0, hi)
        eligible = days[lo:hi]
        ref_idx = lo - 1
        ref = days[ref_idx] if ref_idx >= 0 else None
        ref_missing = ref is None or (origin - dt.timedelta(days=window) - ref).days > 14
        # Reentry is allowed only if the 28-day feature fails at this check.
        if window == 28:
            missing_28 = not eligible or ref_missing
        elif not eligible or ref_missing:
            return None
        refs[window], eligible_by_window[window] = (None if ref_missing else ref), eligible
    if not missing_28:
        return None

    eligible7 = eligible_by_window[7]
    volumes7 = [series[d][1] for d in eligible7 if series[d][1] > 0]
    if not volumes7:
        return None
    mean7 = sum(volumes7) / len(eligible7)
    if not math.isfinite(mean7) or mean7 <= 0:
        return None
    seasonal = prospective._seasonal(series, target, origin)
    if seasonal is None or seasonal <= 0:
        return None

    price0 = series[src][0]
    price7 = math.log(price0 / series[refs[7]][0])
    seasonal_gap = math.log(seasonal / price0)
    weekday = 2 * math.pi * target.weekday() / 7
    year_days = 366 if modeling._is_leap(target.year) else 365
    year_angle = 2 * math.pi * (target.timetuple().tm_yday - 1) / year_days
    coverage7 = len(eligible7) / 7
    row = (float(horizon), math.sin(weekday), math.cos(weekday), math.sin(year_angle), math.cos(year_angle),
           price7, 0.0, 0.0, seasonal_gap, 0.0, horizon * seasonal_gap, coverage7, 0.0)
    if not all(math.isfinite(value) for value in row):
        return None
    return row, src


def issue_feature(series: dict[dt.date, tuple[float, float]], origin: dt.date,
                  target: dt.date, horizon: int) -> tuple[tuple[float, ...], dt.date, bool] | None:
    """Return 13 frozen features, log mean 7-day volume, and missing28 indicator."""
    volume7 = _log_mean_volume7(series, origin)
    if volume7 is None:
        return None
    full = prospective._feature(series, origin, target, horizon)
    if full is not None:
        return full[0] + (volume7, 0.0), full[1], False
    fallback = _reentry_feature(series, origin, target, horizon)
    if fallback is None:
        return None
    return fallback[0] + (volume7, 1.0), fallback[1], True


def prior_reentry_target_days(series: dict[dt.date, tuple[float, float]],
                              issue_origin: dt.date,
                              feature_cache: dict[tuple[dt.date, int], Any] | None = None) -> set[dt.date]:
    """Distinct earlier target days with a reentry feature and observed target."""
    found: set[dt.date] = set()
    day = START
    while day < issue_origin:
        for horizon in HORIZONS:
            target = day + dt.timedelta(days=horizon)
            if target >= issue_origin or target not in series:
                continue
            key = (day, horizon)
            if feature_cache is not None and key in feature_cache:
                result = feature_cache[key]
            else:
                result = issue_feature(series, day, target, horizon)
                if feature_cache is not None:
                    feature_cache[key] = result
            if result is not None and result[2]:
                found.add(target)
        day += dt.timedelta(days=1)
    return found


def pepper_reentry_history_by_origin(rows: list[dict[str, Any]],
                                    pepper: dict[dt.date, tuple[float, float]]) -> list[dict[str, Any]]:
    """Apply the training-history check only to target-present reentry pairs."""
    results = []
    feature_cache: dict[tuple[dt.date, int], Any] = {}
    for origin in ORIGINS:
        origin_rows = [row for row in rows if row["productId"] == PEPPER_ID
                       and row["origin"] == origin.isoformat()]
        used = any(row["reentryBranch"] and row["targetStatus"] == "present" for row in origin_rows)
        if not used:
            continue
        prior_targets = prior_reentry_target_days(pepper, origin, feature_cache)
        results.append({"origin": origin.isoformat(), "distinctPriorReentryTargets": len(prior_targets),
                        "minimum": MIN_PEPPER_REENTRY_TRAIN_TARGETS,
                        "passed": len(prior_targets) >= MIN_PEPPER_REENTRY_TRAIN_TARGETS})
    return results


def validate_inputs(prices_path: Path, status_path: Path):
    """Reuse the target screen's frozen snapshot and provenance validation."""
    return target_screen.validate_inputs(prices_path, status_path)


def build(prices_path: Path, status_path: Path, output: Path) -> dict[str, Any]:
    products, status, provenance_path, pb, sb, provb = validate_inputs(prices_path, status_path)
    rows = []
    series_by_product = {}
    for pid in dashboard.PRODUCTS:
        raw = prospective._series(products[pid])
        series = {d: value for d, value in raw.items() if START <= d <= END}
        series_by_product[pid] = series
        for origin in ORIGINS:
            for horizon in HORIZONS:
                target = origin + dt.timedelta(days=horizon)
                feature = issue_feature(series, origin, target, horizon)
                target_value = series.get(target)
                target_present = target_value is not None and math.isfinite(target_value[0]) and target_value[0] > 0
                rows.append({"productId": pid, "origin": origin.isoformat(), "horizon": horizon,
                             "target": target.isoformat(),
                             "targetStatus": "present" if target_present else "missing_observation",
                             "featureStatus": "available" if feature is not None else "unavailable",
                             "reentryBranch": bool(feature is not None and feature[2]),
                             "featureFailureReason": None if feature is not None else "feature_requirements_unmet",
                             "pairedEvaluable": bool(target_present and feature is not None)})

    product_summaries = []
    for pid in dashboard.PRODUCTS:
        subset = [row for row in rows if row["productId"] == pid]
        by_horizon = []
        for horizon in HORIZONS:
            hrows = [row for row in subset if row["horizon"] == horizon]
            targets = sum(row["targetStatus"] == "present" for row in hrows)
            pairs = sum(row["pairedEvaluable"] for row in hrows)
            by_horizon.append({"horizon": horizon, "planned": len(hrows), "targetPresent": targets,
                               "featureAvailable": sum(row["featureStatus"] == "available" for row in hrows),
                               "pairedEvaluable": pairs,
                               "conditionalFeatureCoverage": pairs / targets if targets else None,
                               "conditionalCoveragePassed": targets > 0 and pairs / targets >= MIN_CONDITIONAL_COVERAGE,
                               "minimumPairedPassed": pairs >= MIN_PAIRED})
        product_summaries.append({"productId": pid, "planned": len(subset),
                                  "targetPresent": sum(row["targetStatus"] == "present" for row in subset),
                                  "featureAvailable": sum(row["featureStatus"] == "available" for row in subset),
                                  "pairedEvaluable": sum(row["pairedEvaluable"] for row in subset),
                                  "byHorizon": by_horizon})

    pepper = series_by_product.get(PEPPER_ID, {})
    pepper_history = pepper_reentry_history_by_origin(rows, pepper)

    all_pairs_gate = all(h["pairedEvaluable"] >= MIN_PAIRED
                         for product in product_summaries for h in product["byHorizon"])
    all_coverage_gate = all(h["conditionalCoveragePassed"]
                            for product in product_summaries for h in product["byHorizon"])
    pepper_history_gate = all(item["passed"] for item in pepper_history)
    payload = {"schemaVersion": 1, "experiment": "supply_pressure_seasonal_reentry_availability_screen",
               "inputWindow": {"start": START.isoformat(), "end": END.isoformat()},
               "origins": [day.isoformat() for day in ORIGINS], "horizons": HORIZONS,
               "plannedRows": len(rows), "minimumPairedPerProductHorizon": MIN_PAIRED,
               "minimumConditionalCoveragePerProductHorizon": MIN_CONDITIONAL_COVERAGE,
               "minimumPepperPriorReentryTargetDays": MIN_PEPPER_REENTRY_TRAIN_TARGETS,
               "gate": {"allProductHorizonsMeetPairedMinimum": all_pairs_gate,
                        "allProductHorizonsMeetConditionalCoverage": all_coverage_gate,
                        "allPepperReentryOriginsMeetPriorTargetMinimum": pepper_history_gate,
                        "pepperReentryOriginsChecked": len(pepper_history),
                        "passed": all_pairs_gate and all_coverage_gate and pepper_history_gate},
               "pepperReentryHistoryByOrigin": pepper_history,
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
