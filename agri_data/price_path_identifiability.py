"""Pure synthetic check for price-path collisions in the frozen 106 inputs.

This diagnostic constructs no market data and performs no fitting. It swaps two
interior, pre-origin prices that stay inside the same aggregate windows and
leaves volume and metadata fixed, then compares the frozen feature vector with
an ordered calendar-day representation.
"""
from __future__ import annotations

import datetime as dt
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import direct_horizon
from . import full_history_feature_library as library
from . import full_history_return_eval as narrow
from . import full_history_xgb_rf_eval as rich
from . import modeling

OMITTED_LIBRARY_FEATURES = frozenset({
    "p0_age_days", "target_weekday_sin", "target_weekday_cos",
    "target_annual_sin", "target_annual_cos",
})
ROUND_OFF_TOLERANCE = 1e-12
ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
PRICES_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
PRODUCTS = tuple(direct_horizon.dashboard.PRODUCTS)
SAMPLE_START = dt.date(2014, 1, 1)
SAMPLE_END = dt.date(2024, 12, 1)
MAX_READ_DATE = SAMPLE_END - dt.timedelta(days=1)
HORIZON = 1


def _history(cutoff: dt.date) -> dict[dt.date, dict[str, Any]]:
    records: dict[dt.date, dict[str, Any]] = {}
    for i in range(121):
        day = cutoff - dt.timedelta(days=120 - i)
        records[day] = {
            "price": 100.0 + 0.25 * i,
            "volume": 1000.0 + 7.0 * (i % 13),
            "rowCount": 8 + (i % 5),
            "markets": frozenset(library.MARKET_CODES),
        }
    # Distinct, deliberately non-symmetric values make the order change clear.
    records[cutoff - dt.timedelta(days=5)]["price"] = 111.0
    records[cutoff - dt.timedelta(days=6)]["price"] = 147.0
    return records


class FeatureUnavailable(Exception):
    """A required causal feature vector cannot be formed for this sample."""


def _feature_vector(records: dict[dt.date, dict[str, Any]], target: dt.date,
                    horizon: int) -> tuple[list[float], dt.date, float]:
    days = sorted(records)
    prices = {day: row["price"] for day, row in records.items()}
    series = {day: (row["price"], row["volume"]) for day, row in records.items()}
    found = modeling._feature_row(series, target - dt.timedelta(days=horizon + 1),
                                  target, horizon, {})
    if found is None:
        raise FeatureUnavailable("direct 40-feature row unavailable")
    source_day, original40 = found
    expanded71, p0, audit = library.features(records, target, horizon, days, prices)
    if expanded71 is None or p0 is None:
        raise FeatureUnavailable("full-history feature row unavailable")
    if len(original40) != 40 or len(expanded71) != 71:
        raise AssertionError("unexpected frozen feature schema")
    if audit.get("p0Date") != source_day.isoformat() or not math.isclose(
            p0, series[source_day][0], rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError("direct and rich synthetic p0 anchors differ")
    appended = [value for name, value in zip(library.FEATURE_NAMES, expanded71)
                if name not in OMITTED_LIBRARY_FEATURES]
    if len(appended) != 66:
        raise AssertionError("expected 66 preserved library additions")
    return list(original40) + appended, source_day, p0


def _ordered_calendar_vector(records: dict[dt.date, dict[str, Any]],
                             cutoff: dt.date, days: int = 91,
                             reference_price: float | None = None) -> list[tuple[float, int]]:
    """Encode each calendar date as (log(price/p0), observed mask), no fill."""
    p0 = reference_price if reference_price is not None else records[cutoff]["price"]
    output = []
    for offset in range(days - 1, -1, -1):
        day = cutoff - dt.timedelta(days=offset)
        row = records.get(day)
        if row is None or not math.isfinite(row["price"]) or row["price"] <= 0:
            output.append((0.0, 0))
        else:
            output.append((math.log(row["price"] / p0), 1))
    return output


def analyze_synthetic_collision() -> dict[str, Any]:
    horizon = 10
    target = dt.date(2026, 6, 10)
    cutoff = target - dt.timedelta(days=horizon + 1)
    first_day, second_day = cutoff - dt.timedelta(days=5), cutoff - dt.timedelta(days=6)
    left = _history(cutoff)
    right = {day: dict(row) for day, row in left.items()}
    right[first_day]["price"], right[second_day]["price"] = (
        right[second_day]["price"], right[first_day]["price"])

    frozen_left, p0_day_left, p0_left = _feature_vector(left, target, horizon)
    frozen_right, p0_day_right, p0_right = _feature_vector(right, target, horizon)
    deltas = [abs(a - b) for a, b in zip(frozen_left, frozen_right)]
    changed = [i for i, (a, b) in enumerate(zip(frozen_left, frozen_right))
               if abs(a - b) > ROUND_OFF_TOLERANCE]

    ordered_left = _ordered_calendar_vector(left, cutoff)
    ordered_right = _ordered_calendar_vector(right, cutoff)
    ordered_changed = [cutoff - dt.timedelta(days=90 - i)
                       for i, (a, b) in enumerate(zip(ordered_left, ordered_right)) if a != b]
    direct_lag_anchors = {f"lag{lag}": cutoff - dt.timedelta(days=lag)
                          for lag in modeling.LAGS}
    library_base_anchors = {
        "p1PreviousObservation": cutoff - dt.timedelta(days=1),
        "p7": cutoff - dt.timedelta(days=7),
        "p30": cutoff - dt.timedelta(days=30),
    }
    expanded_anchors = {f"anchor{window}d": cutoff - dt.timedelta(days=window)
                        for window in (3, 14, 60)}
    slope_endpoints = {f"slope{window}dStart": cutoff - dt.timedelta(days=window - 1)
                       for window in (14, 60)}
    slope_endpoints.update({f"slope{window}dEnd": cutoff for window in (14, 60)})
    anchors = {"direct40Lags": direct_lag_anchors,
               "libraryBaseAnchors": library_base_anchors,
               "expandedAnchors": expanded_anchors,
               "expandedSlopeEndpoints": slope_endpoints}
    if p0_day_left != p0_day_right or not math.isclose(p0_left, p0_right, rel_tol=0, abs_tol=1e-12):
        raise AssertionError("swap changed p0")
    protected_dates = {day for group in anchors.values() for day in group.values()}
    if first_day in protected_dates or second_day in protected_dates:
        raise AssertionError("swap touched a direct/library lag, anchor, or slope endpoint")
    if len(frozen_left) != 106 or len(frozen_right) != 106:
        raise AssertionError("expected 106-dimensional frozen representation")
    return {
        "status": "synthetic_identifiability_check",
        "noFit": True,
        "realDataRead": False,
        "target": target.isoformat(),
        "horizon": horizon,
        "featureCutoff": cutoff.isoformat(),
        "swappedDays": [first_day.isoformat(), second_day.isoformat()],
        "swappedPrices": [left[first_day]["price"], left[second_day]["price"]],
        "causalHistoryComplete": True,
        "allSyntheticPricesVolumesMetadataPositive": True,
        "p0": {"date": p0_day_left.isoformat(), "price": p0_left,
               "sameAcrossHistories": True},
        "unchangedAnchors": {key: {name: day.isoformat() for name, day in value.items()}
                             for key, value in anchors.items()},
        "frozenFeatureCount": 106,
        "frozenFeatureExactEquality": frozen_left == frozen_right,
        "frozenFeatureMaxAbsDelta": max(deltas, default=0.0),
        "frozenFeaturesChangedAboveTolerance": changed,
        "roundOffTolerance": ROUND_OFF_TOLERANCE,
        "orderedCalendarVectorLength": len(ordered_left),
        "orderedCalendarMaskChanges": [day.isoformat() for day in ordered_changed
                                        if ordered_left[(day - (cutoff - dt.timedelta(days=90))).days][1]
                                        != ordered_right[(day - (cutoff - dt.timedelta(days=90))).days][1]],
        "orderedCalendarValueChanges": [day.isoformat() for day in ordered_changed],
        "orderedCalendarVectorDistinguishesHistories": ordered_left != ordered_right,
        "existingSimilarTrajectoryRepresentation": {
            "usesExactOrderedDailyPath": False,
            "evidence": "analog_residual_pilot.state_features uses two endpoint price ratios over the last 10 observations, one volume ratio, and annual sine/cosine; it does not retain the interior daily order."
        },
        "interpretation": "The frozen 106 summaries identify these two different ordered paths as the same input; the ordered calendar vector retains the difference. This is representation evidence only, not predictive-benefit evidence.",
    }


def _date(value: Any) -> dt.date | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _trim_snapshot(raw: dict[str, Any], max_read_date: dt.date = MAX_READ_DATE) -> tuple[dict[str, Any], dict[str, dict[str, int]]]:
    """Drop future/malformed date rows before any price, volume, or metadata read."""
    if raw.get("schemaVersion") != 1 or not isinstance(raw.get("products"), list):
        raise ValueError("prices must use dashboard schemaVersion 1")
    trimmed_products = []
    audits: dict[str, dict[str, int]] = {}
    for product in raw["products"]:
        if not isinstance(product, dict) or product.get("id") not in PRODUCTS:
            raise ValueError("unknown or malformed product in price snapshot")
        observations = product.get("observations")
        if not isinstance(observations, list):
            raise ValueError("price product observations must be a list")
        kept = []
        future = invalid_date = 0
        for row in observations:
            if not isinstance(row, dict):
                invalid_date += 1
                continue
            day = _date(row.get("date"))
            if day is None:
                invalid_date += 1
            elif day > max_read_date:
                future += 1
            else:
                kept.append(row)
        trimmed_products.append({"id": product["id"], "observations": kept})
        audits[product["id"]] = {"keptThroughMaxReadDate": len(kept),
                                  "skippedLaterDates": future,
                                  "skippedInvalidDateRows": invalid_date}
    if {p["id"] for p in trimmed_products} != set(PRODUCTS):
        raise ValueError("prices must contain every product exactly once")
    return {"schemaVersion": 1, "products": trimmed_products}, audits


def _sample_origins() -> list[dt.date]:
    return [dt.date(year, month, 1)
            for year in range(SAMPLE_START.year, SAMPLE_END.year + 1)
            for month in range(1, 13)]


def _swap_status(history: dict[dt.date, dict[str, Any]], source_day: dt.date) -> tuple[str, dt.date, dt.date]:
    first_day, second_day = source_day - dt.timedelta(days=5), source_day - dt.timedelta(days=6)
    first, second = history.get(first_day), history.get(second_day)
    if first is None or second is None:
        return "missing_pair", first_day, second_day
    first_price, second_price = first.get("price"), second.get("price")
    if first_price == second_price:
        return "equal_price", first_day, second_day
    return "eligible", first_day, second_day


def _source_rows(raw: dict[str, Any]) -> tuple[dict[str, dict[dt.date, tuple[float, float]]],
                                                dict[str, dict[dt.date, dict[str, Any]]],
                                                dict[str, Any]]:
    direct = direct_horizon._observations(raw)
    raw_maps, source_audit = rich._raw_maps(raw)
    records: dict[str, dict[dt.date, dict[str, Any]]] = {}
    for product in PRODUCTS:
        domain = direct[product]
        if not set(domain).issubset(raw_maps[product]):
            raise ValueError(f"direct positive price-volume date missing from rich rows: {product}")
        records[product] = {day: dict(raw_maps[product][day]) for day in domain}
        for day, (price, volume) in domain.items():
            row = records[product][day]
            if not math.isclose(row["price"], price, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"price anchor mismatch: {product}/{day}")
            if row.get("volume") is None or not math.isclose(row["volume"], volume, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"volume anchor mismatch: {product}/{day}")
    return direct, records, source_audit


def _sample_record(product: str, origin: dt.date,
                   domain: dict[dt.date, tuple[float, float]],
                   records: dict[dt.date, dict[str, Any]]) -> dict[str, Any]:
    cutoff = origin - dt.timedelta(days=1)
    target = origin + dt.timedelta(days=HORIZON)
    history = {day: row for day, row in records.items() if day <= cutoff}
    source_day = max(history, default=None)
    base = {"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
            "horizon": HORIZON, "featureCutoff": cutoff.isoformat(),
            "swapDay5": None, "swapDay6": None,
            "p0SourceDate": source_day.isoformat() if source_day else None,
            "sourceAgeDays": (origin - source_day).days if source_day else None,
            "swapStatus": None, "frozen106MaxAbsDelta": None,
            "frozen106ChangedDimensionCount": None,
            "ordered91ChangedCalendarDates": [], "ordered91Changed": None}
    if source_day is None:
        base["swapStatus"] = "no_source"
        return base
    if (origin - source_day).days > 14:
        base["swapStatus"] = "stale_source"
        return base
    status, first_day, second_day = _swap_status(history, source_day)
    base["swapDay5"], base["swapDay6"] = first_day.isoformat(), second_day.isoformat()
    base["swapStatus"] = status
    if status != "eligible":
        return base

    try:
        target_features, source_before, p0 = _feature_vector(history, target, HORIZON)
    except FeatureUnavailable:
        base["swapStatus"] = "feature_unavailable"
        return base
    counterfactual = {day: dict(row) for day, row in history.items()}
    counterfactual[first_day]["price"], counterfactual[second_day]["price"] = (
        counterfactual[second_day]["price"], counterfactual[first_day]["price"])
    try:
        counter_features, source_after, p0_after = _feature_vector(counterfactual, target, HORIZON)
    except FeatureUnavailable as exc:
        raise AssertionError("price-only swap changed feature availability") from exc
    if source_before != source_after or not math.isclose(p0, p0_after, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(f"counterfactual swap changed p0 for {product}/{origin}")
    deltas = [abs(left - right) for left, right in zip(target_features, counter_features)]
    base["frozen106MaxAbsDelta"] = max(deltas, default=0.0)
    base["frozen106ChangedDimensionCount"] = sum(delta > ROUND_OFF_TOLERANCE for delta in deltas)
    ordered_original = _ordered_calendar_vector(history, cutoff, reference_price=p0)
    ordered_counter = _ordered_calendar_vector(counterfactual, cutoff, reference_price=p0)
    base["ordered91ChangedCalendarDates"] = [
        (cutoff - dt.timedelta(days=90 - i)).isoformat()
        for i, (left, right) in enumerate(zip(ordered_original, ordered_counter)) if left != right]
    base["ordered91Changed"] = ordered_original != ordered_counter
    return base


def build_snapshot_diagnostic(raw: dict[str, Any], *, source_sha256: str,
                              source_path: str = str(PRICES_PATH)) -> dict[str, Any]:
    """Summarize monthly historical price-path collisions without targets or fitting."""
    trimmed, trim_audit = _trim_snapshot(raw)
    direct, records, source_audit = _source_rows(trimmed)
    origins = _sample_origins()
    counts = {product: {"planned": len(origins), "eligible": 0, "noSource": 0,
                        "staleSource": 0, "featureUnavailable": 0,
                        "missingPair": 0, "equalPrice": 0,
                        "changedFeature": 0, "preservedFeatureOrderChanged": 0,
                        "orderedPathChanged": 0,
                        "maxFrozen106AbsDelta": None, "maxChangedFeatureDimensions": 0}
              for product in PRODUCTS}
    samples = []
    for product in PRODUCTS:
        for origin in origins:
            row = _sample_record(product, origin, direct[product], records[product])
            samples.append(row)
            summary = counts[product]
            status = row["swapStatus"]
            if status == "no_source":
                summary["noSource"] += 1
            elif status == "stale_source":
                summary["staleSource"] += 1
            elif status == "feature_unavailable":
                summary["featureUnavailable"] += 1
            elif status == "missing_pair":
                summary["missingPair"] += 1
            elif status == "equal_price":
                summary["equalPrice"] += 1
            elif status == "eligible":
                summary["eligible"] += 1
                delta = row["frozen106MaxAbsDelta"]
                summary["maxFrozen106AbsDelta"] = max(summary["maxFrozen106AbsDelta"] or 0.0, delta)
                summary["maxChangedFeatureDimensions"] = max(
                    summary["maxChangedFeatureDimensions"], row["frozen106ChangedDimensionCount"])
                if delta > ROUND_OFF_TOLERANCE:
                    summary["changedFeature"] += 1
                if row["ordered91Changed"]:
                    summary["orderedPathChanged"] += 1
                    if delta <= ROUND_OFF_TOLERANCE:
                        summary["preservedFeatureOrderChanged"] += 1
            else:
                raise AssertionError(f"unknown swap status: {status}")
    expected = len(PRODUCTS) * len(origins)
    if len(samples) != expected or any(sum(counts[p][key] for key in
            ("noSource", "staleSource", "featureUnavailable", "missingPair", "equalPrice", "eligible")) != len(origins)
            for p in PRODUCTS):
        raise AssertionError("sample grid or per-crop status accounting is incomplete")
    return {
        "schemaVersion": 1,
        "status": "read_only_price_path_identifiability_diagnostic",
        "noFit": True, "targetValuesOrLossesAttached": False,
        "counterfactualsAreObservedFutures": False,
        "sampleProtocol": {"originGrid": "first calendar day of every month 2014-01 through 2024-12",
                           "horizonForFeatureCalendarOnly": HORIZON,
                           "products": list(PRODUCTS), "plannedSamples": expected,
                           "sourceCutoff": "origin-1; only positive price and positive volume dates",
                           "maximumReadDate": MAX_READ_DATE.isoformat(),
                           "maximumSourceAgeDays": 14,
                           "swapDays": ["p0SourceDate-5 calendar days", "p0SourceDate-6 calendar days"],
                           "swapCondition": "both dates in positive price-volume feature domain and prices differ",
                           "sourceAge": "origin minus p0SourceDate in calendar days; maximum 14",
                           "counterfactual": "exchange only the two prices on a copy; preserve all other fields",
                           "orderedPath": "91 calendar-day log(price/p0) values with a positive-price-and-volume observation mask; missing dates use (0,0), no forward fill",
                           "featureChangeTolerance": ROUND_OFF_TOLERANCE},
        "inputSha256": {"prices": source_sha256,
                        "pricePathDiagnosticModule": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        "modeling": hashlib.sha256(Path(modeling.__file__).read_bytes()).hexdigest(),
                        "featureLibrary": hashlib.sha256(Path(library.__file__).read_bytes()).hexdigest(),
                        "narrowReturnFeatureHelper": hashlib.sha256(Path(narrow.__file__).read_bytes()).hexdigest(),
                        "richFeatureHelper": hashlib.sha256(Path(rich.__file__).read_bytes()).hexdigest(),
                        "directHorizonHelper": hashlib.sha256(Path(direct_horizon.__file__).read_bytes()).hexdigest()},
        "source": {"path": source_path, "sha256": source_sha256},
        "sourceRowAuditThroughMaxReadDate": trim_audit,
        "positivePriceVolumeRecordAuditThroughMaxReadDate": source_audit,
        "perCropCounts": counts,
        "samples": samples,
        "interpretation": "This measures whether the current summaries discard interior price ordering on historical inputs. It does not estimate model performance or benefit.",
    }


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=False, sort_keys=True,
                                 indent=2, allow_nan=False) + "\n").encode("utf-8"))


def run_snapshot_cli(prices_path: Path, output_path: Path) -> dict[str, Any]:
    raw_bytes = prices_path.read_bytes()
    source_sha = _sha256(raw_bytes)
    if source_sha != PRICES_SHA256:
        raise ValueError("pinned prices SHA-256 mismatch")
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict):
        raise ValueError("price snapshot must have an object root")
    report = build_snapshot_diagnostic(raw, source_sha256=source_sha,
                                       source_path=str(prices_path))
    _write_exclusive(output_path, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("synthetic", help="run only the synthetic identifiability check")
    snapshot = subparsers.add_parser("snapshot", help="analyze the pinned historical snapshot read-only")
    snapshot.add_argument("--prices", type=Path, default=PRICES_PATH)
    snapshot.add_argument("--output", type=Path, required=True,
                          help="new immutable JSON output path (must not already exist)")
    args = parser.parse_args()
    if args.command == "synthetic":
        print(json.dumps(analyze_synthetic_collision(), ensure_ascii=False, indent=2))
    else:
        report = run_snapshot_cli(args.prices, args.output)
        print(json.dumps({"status": report["status"], "output": str(args.output),
                          "plannedSamples": report["sampleProtocol"]["plannedSamples"],
                          "perCropCounts": report["perCropCounts"]},
                         ensure_ascii=False, indent=2))
