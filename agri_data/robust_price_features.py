"""Causal robust price summaries appended after the frozen 106-feature input.

This module builds features and read-only historical availability diagnostics;
it does not fit a model, request an API, or attach target answers.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from . import direct_horizon
from . import full_history_feature_library as library
from . import full_history_return_eval as narrow
from . import full_history_xgb_rf_eval as rich
from . import modeling

ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
PRICES_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
PRODUCTS = tuple(direct_horizon.dashboard.PRODUCTS)
WINDOWS = (7, 28, 90)
MIN_WINDOW_OBSERVATIONS = 3
MAX_SOURCE_AGE_DAYS = 14
SAMPLE_START = dt.date(2014, 1, 1)
SAMPLE_END = dt.date(2024, 12, 1)
MAX_READ_DATE = SAMPLE_END - dt.timedelta(days=1)
HORIZON = 1
OMITTED_LIBRARY_FEATURES = frozenset({
    "p0_age_days", "target_weekday_sin", "target_weekday_cos",
    "target_annual_sin", "target_annual_cos",
})
ORIGINAL_106_NAMES = tuple(modeling.FEATURE_NAMES) + tuple(
    name for name in library.FEATURE_NAMES if name not in OMITTED_LIBRARY_FEATURES)
ROBUST_FEATURE_NAMES = tuple(
    name for window in WINDOWS for name in (
        f"price_log_p0_to_median_{window}d",
        f"price_log_iqr_{window}d",
        f"price_log_mad_{window}d"))
FEATURE_NAMES = ORIGINAL_106_NAMES + ROBUST_FEATURE_NAMES
assert len(ORIGINAL_106_NAMES) == 106 and len(ROBUST_FEATURE_NAMES) == 9


def _linear_quantile(values: list[float], quantile: float) -> float:
    """NumPy-style linear quantile at (n-1)*q, without a runtime dependency."""
    if not values or not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile requires nonempty values and q in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])


def _date(value: Any) -> dt.date | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _causal_inputs(records: dict[dt.date, dict[str, Any]],
                   direct_series: dict[dt.date, tuple[float, float]], cutoff: dt.date
                   ) -> tuple[dict[dt.date, dict[str, Any]],
                              dict[dt.date, tuple[float, float]]]:
    # Filter date keys before touching any value in a later observation row.
    causal_records = {day: row for day, row in records.items() if day <= cutoff}
    causal_direct = {day: value for day, value in direct_series.items() if day <= cutoff}
    if set(causal_direct) - set(causal_records):
        raise ValueError("positive price-volume source dates lack rich metadata rows")
    for day, (price, volume) in causal_direct.items():
        if (not math.isfinite(price) or price <= 0
                or not math.isfinite(volume) or volume <= 0):
            raise ValueError(f"direct source is outside the positive finite price-volume domain at {day}")
        row = causal_records[day]
        row_price = row.get("price")
        row_volume = row.get("volume")
        if (not isinstance(row_price, (int, float)) or isinstance(row_price, bool)
                or not math.isclose(float(row_price), price, rel_tol=0.0, abs_tol=1e-12)):
            raise ValueError(f"price source mismatch at {day}")
        if (row_volume is None or isinstance(row_volume, bool)
                or not math.isclose(float(row_volume), volume, rel_tol=0.0, abs_tol=1e-12)):
            raise ValueError(f"volume source mismatch at {day}")
    return causal_records, causal_direct


def features(records: dict[dt.date, dict[str, Any]],
             direct_series: dict[dt.date, tuple[float, float]],
             target: dt.date, horizon: int
             ) -> tuple[list[float] | None, float | None, dict[str, Any]]:
    """Return the exact original 106-value prefix plus nine causal summaries.

    The original feature helpers see only source rows through origin-1. Robust
    values use positive price-and-volume observations on those same source days.
    Unsupported windows emit zeros while preserving their count/status in audit.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("horizon must be positive")
    origin = target - dt.timedelta(days=horizon)
    cutoff = origin - dt.timedelta(days=1)
    causal_records, causal_direct = _causal_inputs(records, direct_series, cutoff)
    causal_days = sorted(causal_direct)
    causal_prices = {day: pair[0] for day, pair in causal_direct.items()}

    source_day = max(causal_days, default=None)
    if source_day is None:
        return None, None, {"status": "no_p0_source", "featureCutoff": cutoff.isoformat()}
    source_age = (origin - source_day).days
    if source_age < 0 or source_age > MAX_SOURCE_AGE_DAYS:
        return None, None, {"status": "stale_p0", "featureCutoff": cutoff.isoformat(),
                            "p0Date": source_day.isoformat(), "p0AgeDays": source_age}

    found = modeling._feature_row(causal_direct, cutoff, target, horizon, {})
    if found is None:
        return None, None, {"status": "original_features_unavailable", "featureCutoff": cutoff.isoformat()}
    feature_source_day, original40 = found
    if feature_source_day != source_day:
        raise ValueError("modeling p0 selection differs from the positive price-volume domain")

    expanded, p0, library_audit = library.features(causal_records, target, horizon,
                                                   causal_days, causal_prices)
    if expanded is None or p0 is None:
        return None, None, {"status": "original_features_unavailable", "featureCutoff": cutoff.isoformat(),
                            "libraryAudit": library_audit}
    if (len(original40) != 40 or len(expanded) != 71
            or library_audit.get("p0Date") != source_day.isoformat()
            or not math.isclose(p0, causal_direct[source_day][0], rel_tol=0.0, abs_tol=1e-12)):
        raise ValueError("the exact frozen 106-feature prefix or common p0 anchor changed")
    appended = [float(value) for name, value in zip(library.FEATURE_NAMES, expanded)
                if name not in OMITTED_LIBRARY_FEATURES]
    original106 = [float(value) for value in (*original40, *appended)]
    if len(original106) != 106 or not all(math.isfinite(value) for value in original106):
        raise ValueError("frozen 106-feature prefix is invalid")

    robust: list[float] = []
    window_audit: dict[str, Any] = {}
    for window in WINDOWS:
        low = cutoff - dt.timedelta(days=window - 1)
        window_prices = [causal_direct[day][0] for day in causal_days
                         if low <= day <= cutoff]
        count = len(window_prices)
        supported = count >= MIN_WINDOW_OBSERVATIONS
        if supported:
            logs = [math.log(value) for value in window_prices]
            log_median = statistics.median(logs)
            median_raw_price = statistics.median(window_prices)
            robust.extend((math.log(p0 / median_raw_price),
                           _linear_quantile(logs, 0.75) - _linear_quantile(logs, 0.25),
                           statistics.median([abs(value - log_median) for value in logs])))
        else:
            robust.extend((0.0, 0.0, 0.0))
        window_audit[str(window)] = {"positivePriceVolumeDays": count,
                                     "minimumRequired": MIN_WINDOW_OBSERVATIONS,
                                     "supported": supported}
    vector = original106 + robust
    if len(vector) != 115 or not all(math.isfinite(value) for value in vector):
        raise ValueError("robust price feature vector is invalid")
    return vector, p0, {"status": "available", "featureCutoff": cutoff.isoformat(),
                        "p0Date": source_day.isoformat(), "p0AgeDays": source_age,
                        "originalFeatureCount": 106, "addedFeatureCount": 9,
                        "featureCount": 115, "windowAudit": window_audit,
                        "libraryAudit": library_audit}


def _trim_snapshot(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    """Remove post-cap and malformed-date rows before any value access."""
    if raw.get("schemaVersion") != 1 or not isinstance(raw.get("products"), list):
        raise ValueError("prices must use dashboard schemaVersion 1")
    products = []
    audit: dict[str, dict[str, int]] = {}
    for product in raw["products"]:
        if not isinstance(product, dict) or product.get("id") not in PRODUCTS:
            raise ValueError("unknown or malformed product in price snapshot")
        observations = product.get("observations")
        if not isinstance(observations, list):
            raise ValueError("product observations must be a list")
        kept = []
        later = invalid = 0
        for row in observations:
            if not isinstance(row, dict):
                invalid += 1
                continue
            day = _date(row.get("date"))
            if day is None:
                invalid += 1
            elif day > MAX_READ_DATE:
                later += 1
            else:
                kept.append(row)
        products.append({**product, "observations": kept})
        audit[product["id"]] = {"includedRowsThroughCap": len(kept),
                                 "skippedLaterDates": later,
                                 "skippedInvalidDateRows": invalid}
    if len(products) != len(PRODUCTS) or {item["id"] for item in products} != set(PRODUCTS):
        raise ValueError("prices must contain every product exactly once")
    return {**raw, "products": products}, audit


def _sample_origins() -> list[dt.date]:
    return [dt.date(year, month, 1)
            for year in range(SAMPLE_START.year, SAMPLE_END.year + 1)
            for month in range(1, 13)]


def _population_variance(values: list[float]) -> float | None:
    if not values:
        return None
    mean = math.fsum(values) / len(values)
    return math.fsum((value - mean) ** 2 for value in values) / len(values)


def analyze_synthetic_witness() -> dict[str, Any]:
    """Construct two ordered histories colliding in all old inputs but not new robust features."""
    cutoff = dt.date(2024, 6, 30)
    p0 = 100.0
    left_logs = [0, 1, 1, 4, 4, 6]
    right_logs = [0, 0, 3, 3, 4, 6]
    left: dict[dt.date, dict[str, Any]] = {}
    right: dict[dt.date, dict[str, Any]] = {}
    for offset in range(121):
        day = cutoff - dt.timedelta(days=120 - offset)
        price = p0 * math.exp(0.002 * (offset - 120))
        row = {"price": price, "volume": 1000.0 + 3.0 * (offset % 7),
               "rowCount": 8 + offset % 4,
               "markets": frozenset(library.MARKET_CODES)}
        left[day] = dict(row)
        right[day] = dict(row)
    # Put the common 90-day observations on both sides of the witness values,
    # so their changed log order changes a robust median. The old 90-day
    # sum/sum-of-squares/range stay fixed; other old windows and anchors do too.
    witness_days = [cutoff - dt.timedelta(days=35 + index) for index in range(6)]
    witness_set = set(witness_days)
    remaining_window_days = [cutoff - dt.timedelta(days=offset)
                             for offset in range(89, -1, -1)
                             if cutoff - dt.timedelta(days=offset) not in witness_set
                             and cutoff - dt.timedelta(days=offset) != cutoff]
    for index, day in enumerate(remaining_window_days):
        level = -2 if index < 42 else 8
        left[day]["price"] = p0 * math.exp(0.01 * level)
        right[day]["price"] = p0 * math.exp(0.01 * level)
    for day, a, b in zip(witness_days, left_logs, right_logs):
        left[day]["price"] = p0 * math.exp(0.01 * a)
        right[day]["price"] = p0 * math.exp(0.01 * b)
    direct_left = {day: (row["price"], row["volume"]) for day, row in left.items()}
    direct_right = {day: (row["price"], row["volume"]) for day, row in right.items()}
    target = cutoff + dt.timedelta(days=2)
    vector_left, p0_left, audit_left = features(left, direct_left, target, 1)
    vector_right, p0_right, audit_right = features(right, direct_right, target, 1)
    if vector_left is None or vector_right is None:
        raise AssertionError("synthetic witness lacks feature support")
    old_deltas = [abs(a - b) for a, b in zip(vector_left[:106], vector_right[:106])]
    new_deltas = [abs(a - b) for a, b in zip(vector_left[106:], vector_right[106:])]
    if max(old_deltas) > 1e-12 or max(new_deltas) <= 1e-12:
        raise AssertionError("synthetic histories did not isolate a robust-feature distinction")
    return {"status": "synthetic_representation_witness", "noFit": True,
            "realDataRead": False, "target": target.isoformat(), "horizon": 1,
            "featureCutoff": cutoff.isoformat(), "sourceDay": cutoff.isoformat(),
            "sameP0": p0_left == p0_right, "oldFeatureCount": 106,
            "newFeatureCount": 115, "oldFeatureMaxAbsDelta": max(old_deltas),
            "robustFeatureNames": list(ROBUST_FEATURE_NAMES),
            "robustFeatureDeltas": {name: abs(a - b) for name, a, b in
                                    zip(ROBUST_FEATURE_NAMES, vector_left[106:], vector_right[106:])},
            "windowAudit": audit_left["windowAudit"],
            "otherWindowAuditSame": audit_left["windowAudit"] == audit_right["windowAudit"],
            "interpretation": "The added robust summaries distinguish this synthetic pair while the frozen 106 inputs collide; this is representation evidence, not predictive-benefit evidence."}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _diagnostic_from_raw(raw: dict[str, Any], source_sha256: str,
                         source_path: str = str(PRICES_PATH)) -> dict[str, Any]:
    trimmed, trim_audit = _trim_snapshot(raw)
    direct_by_product = direct_horizon._observations(trimmed)
    raw_maps, raw_audit = rich._raw_maps(trimmed)
    origins = _sample_origins()
    feature_values: dict[str, dict[str, list[float]]] = {
        crop: {name: [] for name in ROBUST_FEATURE_NAMES} for crop in PRODUCTS}
    available_counts = {crop: 0 for crop in PRODUCTS}
    stale_counts = {crop: 0 for crop in PRODUCTS}
    unavailable_counts = {crop: 0 for crop in PRODUCTS}
    window_counts = {crop: {str(window): {"samples": 0, "supported": 0,
                                          "unsupported": 0, "observedDays": 0}
                            for window in WINDOWS} for crop in PRODUCTS}
    p0_ages: dict[str, list[int]] = {crop: [] for crop in PRODUCTS}
    planned = len(PRODUCTS) * len(origins)
    for crop in PRODUCTS:
        domain = direct_by_product[crop]
        if not set(domain).issubset(raw_maps[crop]):
            raise ValueError(f"positive price-volume day lacks metadata: {crop}")
        records = {day: raw_maps[crop][day] for day in domain}
        for origin in origins:
            target = origin + dt.timedelta(days=HORIZON)
            vector, _p0, audit = features(records, domain, target, HORIZON)
            if vector is None:
                if audit.get("status") == "stale_p0":
                    stale_counts[crop] += 1
                else:
                    unavailable_counts[crop] += 1
                continue
            available_counts[crop] += 1
            p0_ages[crop].append(audit["p0AgeDays"])
            for window in WINDOWS:
                info = audit["windowAudit"][str(window)]
                record = window_counts[crop][str(window)]
                record["samples"] += 1
                record["observedDays"] += info["positivePriceVolumeDays"]
                record["supported" if info["supported"] else "unsupported"] += 1
                names = [f"price_log_p0_to_median_{window}d", f"price_log_iqr_{window}d",
                         f"price_log_mad_{window}d"]
                for name, value in zip(names, vector[106 + WINDOWS.index(window) * 3:
                                                       106 + WINDOWS.index(window) * 3 + 3]):
                    if info["supported"]:
                        feature_values[crop][name].append(value)
    per_crop: dict[str, Any] = {}
    for crop in PRODUCTS:
        robust_summary = {}
        for name, values in feature_values[crop].items():
            robust_summary[name] = {"supportedSamples": len(values),
                                    "populationVariance": _population_variance(values),
                                    "distinctValues": len(set(values))}
        per_crop[crop] = {"plannedOrigins": len(origins),
                          "availableOriginal106AndRobust115": available_counts[crop],
                          "staleP0": stale_counts[crop],
                          "otherUnavailable": unavailable_counts[crop],
                          "p0AgeDays": {"min": min(p0_ages[crop]) if p0_ages[crop] else None,
                                        "max": max(p0_ages[crop]) if p0_ages[crop] else None},
                          "windowSupport": window_counts[crop],
                          "robustFeatureVarianceOnSupportedWindows": robust_summary}
    expected_grid = planned
    if sum(available_counts.values()) + sum(stale_counts.values()) + sum(unavailable_counts.values()) != expected_grid:
        raise AssertionError("diagnostic sample counts do not cover fixed origin/product grid")
    return {"schemaVersion": 1, "status": "read_only_robust_price_feature_diagnostic",
            "noFit": True, "noApi": True, "targetValuesOrLossesAttached": False,
            "protocol": {"originGrid": "first calendar day monthly 2014-01 through 2024-12",
                         "horizon": HORIZON, "featureCutoff": "origin-1; strict before helper calls",
                         "maximumReadDate": MAX_READ_DATE.isoformat(),
                         "positivePriceVolumeDomain": True,
                         "windowsCalendarDaysEndingAtCutoff": list(WINDOWS),
                         "minimumObservationsForRobustBlock": MIN_WINDOW_OBSERVATIONS,
                         "unsupportedValues": "all three robust values zero; existing 106 count/coverage and audit counts identify support",
                         "quantiles": "linear interpolation at (n-1)*q; q=0.25, 0.75",
                         "addedNames": list(ROBUST_FEATURE_NAMES),
                         "inputDimensions": {"frozenPrefix": 106, "added": 9, "total": 115}},
            "source": {"path": source_path, "sha256": source_sha256},
            "inputSha256": {"generator": _hash(Path(__file__)),
                            "modeling": _hash(Path(modeling.__file__)),
                            "featureLibrary": _hash(Path(library.__file__)),
                            "narrowFeatureHelper": _hash(Path(narrow.__file__)),
                            "richFeatureHelper": _hash(Path(rich.__file__)),
                            "directHorizonHelper": _hash(Path(direct_horizon.__file__))},
            "sourceRowAuditThroughMaximumReadDate": trim_audit,
            "rawFeatureSourceAuditThroughMaximumReadDate": raw_audit,
            "plannedSamples": planned, "perCrop": per_crop,
            "syntheticWitness": analyze_synthetic_witness()}


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, sort_keys=True,
                                 indent=2, allow_nan=False) + "\n").encode("utf-8"))


def run_diagnostic(prices_path: Path, output_path: Path) -> dict[str, Any]:
    raw_bytes = prices_path.read_bytes()
    source_sha = hashlib.sha256(raw_bytes).hexdigest()
    if source_sha != PRICES_SHA256:
        raise ValueError("pinned prices SHA-256 mismatch")
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict):
        raise ValueError("price snapshot root must be an object")
    result = _diagnostic_from_raw(raw, source_sha, str(prices_path))
    _write_exclusive(output_path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("synthetic", help="run only the synthetic feature distinction witness")
    diag = subs.add_parser("diagnostic", help="summarize pinned historical feature availability")
    diag.add_argument("--prices", type=Path, default=PRICES_PATH)
    diag.add_argument("--output", type=Path, required=True,
                      help="new immutable JSON output path (must not already exist)")
    args = parser.parse_args()
    if args.command == "synthetic":
        print(json.dumps(analyze_synthetic_witness(), ensure_ascii=False, indent=2))
    else:
        summary = run_diagnostic(args.prices, args.output)
        print(json.dumps({"status": summary["status"], "output": str(args.output),
                          "plannedSamples": summary["plannedSamples"],
                          "perCrop": summary["perCrop"]}, ensure_ascii=False))
