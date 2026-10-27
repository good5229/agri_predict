"""Read-only forensic diagnostics for repeated price-forecast failures.

This module does not fit, select, or tune models. It reconstructs the already
completed fixed50 residual correction, compares archived 2026 predictions on
an exact common-key subset, and inventories the named feature schemas. All
metrics are descriptive; the 2025 cell oracle is explicitly hindsight-only.
"""
from __future__ import annotations

import argparse
import bisect
import datetime as dt
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from . import fixed50_residual_diagnostic as fixed
from . import full_history_feature_library as library
from . import modeling

ROOT = Path(__file__).resolve().parents[1]
COMPONENT_PATH = ROOT / "outputs/experiments/ridge-sarima-weight-cv-20261008/historical-components.json"
PRICES_PATH = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
FIXED_REPORT_PATH = ROOT / "outputs/experiments/fixed50-residual-diagnostic-20261010/report.json"
ONLINE_MANIFEST_PATH = ROOT / "outputs/experiments/equal-info-online-20261010/forecast-manifest.json"
OUT = ROOT / "outputs/experiments/failure-mechanism-diagnostic-20261010/report.json"

INPUT_SHA256 = {
    "historicalComponents": "6ade501d5818cca33ad973963d12349f51b5ee3bed27afcd2cc023ae972fed74",
    "prices": "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8",
    "fixed50ResidualReport": "62a4936e83f89490e48950b483c36b87f67a229446b1453b61f436a271981c28",
    "equalInfoOnlineManifest": "d7d74823754c2a48d33cc4ddaeab674c686601fb927d399d84c5dbcd235184df",
}
PRODUCTS = fixed.PRODUCTS
HORIZONS = fixed.HORIZONS
VAL_START, VAL_END = dt.date(2025, 3, 1), dt.date(2025, 8, 31)
ONLINE_START, ONLINE_END = dt.date(2026, 1, 1), dt.date(2026, 9, 27)
ONLINE_METHODS = ("fixed50", "causal50", "online71Blend", "online71", "causalDirect")
NEAR_ZERO_LOG_RESIDUAL = 0.01
SIGN_EPS = 1e-12


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _date(value: Any) -> dt.date | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def _sign(value: float, eps: float = SIGN_EPS) -> int:
    return 1 if value > eps else -1 if value < -eps else 0


def _all_price_maps(raw: dict[str, Any], end: dt.date) -> dict[str, dict[dt.date, float]]:
    """Read positive answers only through end; later answer fields are untouched."""
    if raw.get("schemaVersion") != 1 or not isinstance(raw.get("products"), list):
        raise ValueError("prices must use dashboard schemaVersion 1")
    out: dict[str, dict[dt.date, float]] = {}
    for product in raw["products"]:
        pid = product.get("id") if isinstance(product, dict) else None
        if pid not in PRODUCTS or pid in out or not isinstance(product.get("observations"), list):
            raise ValueError("unknown, duplicate, or malformed price product")
        values: dict[dt.date, float] = {}
        for row in product["observations"]:
            if not isinstance(row, dict):
                continue
            day = _date(row.get("date"))
            if day is None or day > end:
                continue
            value = _positive(row.get("actual"))
            if value is None:
                continue
            if day in values:
                raise ValueError(f"duplicate positive actual {pid}/{day}")
            values[day] = value
        out[pid] = values
    if set(out) != set(PRODUCTS):
        raise ValueError("prices must contain all five products")
    return out


def _historical_rows(components: dict[str, Any], prices: dict[str, dict[dt.date, float]]) -> list[dict[str, Any]]:
    rows = fixed._component_rows(components, prices)
    history = {(p, h): [r for r in rows if (r["product"], r["horizon"]) == (p, h)]
               for p in PRODUCTS for h in HORIZONS}
    result = []
    for row in rows:
        if VAL_START <= row["target"] <= VAL_END:
            result.append(fixed._correct(row, history))
    return result


def _error_deltas(rows: list[dict[str, Any]], pred_field: str) -> dict[str, Any]:
    ae_delta: list[float] = []
    ape_delta_pp: list[float] = []
    helpful = harmful = unchanged = 0
    for row in rows:
        actual, base, pred = row["actual"], row["fixed50"], row[pred_field]
        if actual is None or base is None or pred is None:
            continue
        before = abs(base - actual)
        after = abs(pred - actual)
        delta = after - before
        if delta < -1e-12:
            helpful += 1
        elif delta > 1e-12:
            harmful += 1
        else:
            unchanged += 1
        ae_delta.append(delta)
        ape_delta_pp.append(100.0 * delta / actual)
    n = len(ae_delta)
    return {
        "n": n,
        "helpful": helpful,
        "harmful": harmful,
        "unchanged": unchanged,
        "sumAbsoluteErrorChange": math.fsum(ae_delta),
        "meanAbsoluteErrorChange": math.fsum(ae_delta) / n if n else None,
        "sumAbsolutePercentageErrorChangePp": math.fsum(ape_delta_pp),
        "meanAbsolutePercentageErrorChangePp": math.fsum(ape_delta_pp) / n if n else None,
    }


def _hindsight_oracle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [r for r in rows if r["actual"] is not None and r["fixed50"] is not None]
    if not usable:
        return {"n": 0, "multiplier": None, "appliedFactor": None, "mape": None, "mae": None}
    multiplier = fixed._weighted_median_multiplier([(r["actual"], r["fixed50"]) for r in usable])
    # Constrained MAPE optimum over all scalar factors in [0.95, 1.05].
    # This is intentionally stronger than the deployable rolling rule.
    factor = min(1.05, max(0.95, multiplier))
    corrected = [{**r, "oracle": r["fixed50"] * factor} for r in usable]
    metrics = fixed._metrics(corrected, "oracle")
    base_metrics = fixed._metrics(corrected, "fixed50")
    return {
        "n": len(usable),
        "multiplier": multiplier,
        "appliedFactor": factor,
        "mape": metrics["mape"],
        "mae": metrics["mae"],
        "fixed50Mape": base_metrics["mape"],
        "fixed50Mae": base_metrics["mae"],
        "mapeChangePp": metrics["mape"] - base_metrics["mape"],
        "maeChange": metrics["mae"] - base_metrics["mae"],
        "label": "HINDSIGHT_ORACLE_NOT_DEPLOYABLE; constrained MAPE-optimal factor in [0.95,1.05] uses this evaluation cell's own actuals",
    }


def _signed_direction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    corrected = [r for r in rows if r["correction"]["status"] == "corrected" and r["actual"] and r["fixed50"]]
    training = [ _sign(math.log(r["correction"]["multiplier"])) for r in corrected ]
    realized = [ _sign(math.log(r["actual"] / r["fixed50"])) for r in corrected ]
    agree = sum(a == b and a != 0 for a, b in zip(training, realized))
    opposite = sum(a == -b and a != 0 and b != 0 for a, b in zip(training, realized))
    train_neutral = sum(a == 0 for a in training)
    realized_neutral = sum(b == 0 for b in realized)
    both_nonzero = sum(a != 0 and b != 0 for a, b in zip(training, realized))
    return {
        "nCorrectedWithPositiveActual": len(corrected),
        "sameNonzeroDirection": agree,
        "oppositeNonzeroDirection": opposite,
        "trainingDirectionNeutral": train_neutral,
        "realizedLogResidualDirectionNeutral": realized_neutral,
        "sameDirectionRateAmongBothNonzero": agree / both_nonzero if both_nonzero else None,
        "signDefinitions": {"training": "sign(log(training weighted-median multiplier))", "realized": "sign(log(actual/fixed50))"},
    }


def _group_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in rows if r["actual"] is not None and r["fixed50"] is not None]
    clipping = [r for r in valid if r["correction"]["status"] == "corrected"]
    fallback = [r for r in rows if r["correction"]["status"] == "fallback"]
    applied = [r for r in rows if r.get("candidate") is not None and r.get("fixed50") is not None]
    output = {
        "planned": len(rows), "positiveActualAndFixed50": len(valid),
        "corrected": len(clipping), "fallback": len(fallback),
        "notIssued": sum(r["correction"]["status"] == "not_issued" for r in rows),
        "fallbackRate": len(fallback) / len(rows) if rows else None,
        "clipped": sum(abs(r["correction"]["multiplier"] - 1.0) > 0.10 + 1e-12 for r in clipping),
        "clippedRateAmongCorrected": sum(abs(r["correction"]["multiplier"] - 1.0) > 0.10 + 1e-12 for r in clipping) / len(clipping) if clipping else None,
        "appliedAtCap": sum(abs(abs(r["correction"]["appliedFactor"] - 1.0) - 0.05) < 1e-12 for r in clipping),
        "nearZeroRealizedLogResidual": sum(abs(math.log(r["actual"] / r["fixed50"])) <= NEAR_ZERO_LOG_RESIDUAL for r in valid),
        "nearZeroRealizedRate": sum(abs(math.log(r["actual"] / r["fixed50"])) <= NEAR_ZERO_LOG_RESIDUAL for r in valid) / len(valid) if valid else None,
        "issuedForAppliedCorrectionRate": len(applied),
        "nearZeroAppliedCorrection": sum(abs(r["correction"].get("appliedFactor", 1.0) - 1.0) <= SIGN_EPS for r in applied),
        "nearZeroAppliedCorrectionRate": sum(abs(r["correction"].get("appliedFactor", 1.0) - 1.0) <= SIGN_EPS for r in applied) / len(applied) if applied else None,
        "trainingVsRealizedDirection": _signed_direction(rows),
        "correctionChange": _error_deltas(rows, "candidate"),
        "hindsightOracle": _hindsight_oracle(rows),
    }
    return output


def _price_quintiles(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for p in PRODUCTS:
        group = sorted((r for r in rows if r["product"] == p and r["actual"] and r["fixed50"] and r["candidate"] is not None), key=lambda r:(r["actual"],r["target"],r["horizon"]))
        buckets: dict[int, list[dict[str, Any]]] = {i: [] for i in range(1, 6)}
        for i, row in enumerate(group):
            bucket = min(5, (i * 5) // max(1, len(group)) + 1)
            buckets[bucket].append(row)
        out[p] = {}
        for q, values in buckets.items():
            out[p][str(q)] = {
                "targetPriceRange": [min((r["actual"] for r in values), default=None), max((r["actual"] for r in values), default=None)],
                **_error_deltas(values, "candidate"),
                "interpretation": "sum AE change contributes to pooled MAE numerator; sum APE change contributes to pooled MAPE numerator",
            }
    return out


def _diagnose_2025(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell: dict[str, Any] = {}
    crop_summary = {}
    for p in PRODUCTS:
        crop_rows = [r for r in rows if r["product"] == p]
        crop_summary[p] = _group_stats(crop_rows)
        by_cell[p] = {}
        for month in sorted({r["target"].strftime("%Y-%m") for r in rows if r["product"] == p}):
            by_cell[p][month] = {}
            for h in HORIZONS:
                group = [r for r in rows if r["product"] == p and r["target"].strftime("%Y-%m") == month and r["horizon"] == h]
                by_cell[p][month][str(h)] = _group_stats(group)
    return {
        "period": [VAL_START.isoformat(), VAL_END.isoformat()],
        "overall": _group_stats(rows),
        "byCropSummary": crop_summary,
        "byCropMonthHorizon": by_cell,
        "targetPriceQuintileContributionByCrop": _price_quintiles(rows),
        "definitions": {
            "correctionDirection": "sign(log(training weighted-median multiplier)) vs sign(log(actual/fixed50))",
            "helpfulHarmful": "candidate absolute error lower/higher than fixed50 by more than 1e-12; otherwise unchanged",
            "clipped": "abs(training multiplier-1)>0.10; applied factor cap is ±5%",
            "nearZeroRealizedLogResidual": f"abs(log(actual/fixed50)) <= {NEAR_ZERO_LOG_RESIDUAL}",
            "nearZeroAppliedCorrection": f"abs(appliedFactor-1) <= {SIGN_EPS}; fallback is unchanged",
            "oracle": "within each crop-month-horizon, choose the weighted-median MAPE-optimal factor constrained directly to [0.95,1.05] using that cell's own evaluation actuals; hindsight-only representation-capacity bound, not deployable rule",
        },
    }


def _manifest_rows(raw: dict[str, Any]) -> dict[tuple[str, dt.date, int], dict[str, Any]]:
    if raw.get("phase") != "forecast_only" or raw.get("targetValuesAttached") is not False or not isinstance(raw.get("evaluationForecasts"), list):
        raise ValueError("expected answer-free equal-info evaluation manifest")
    out = {}
    for row in raw["evaluationForecasts"]:
        if not isinstance(row, dict) or row.get("product") not in PRODUCTS:
            raise ValueError("invalid online forecast row")
        day, origin, h = _date(row.get("target")), _date(row.get("origin")), row.get("horizon")
        if day is None or origin is None or h not in HORIZONS or origin != day - dt.timedelta(days=h):
            raise ValueError("invalid online forecast key")
        key = (row["product"], day, h)
        if key in out:
            raise ValueError("duplicate online forecast key")
        out[key] = row
    return out


def _pred(row: dict[str, Any], method: str) -> float | None:
    model = row.get("models", {}).get(method, {})
    if not isinstance(model, dict) or model.get("status") != "issued":
        return None
    return _positive(model.get("prediction"))


def _age(row: dict[str, Any]) -> int | None:
    model = row["models"].get("causalDirect", {})
    provenance = model.get("provenance", {})
    value = provenance.get("sourceAgeDays", provenance.get("featureAudit", {}).get("p0AgeDays"))
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _volatility(prices: dict[dt.date, float] | list[tuple[dt.date, float]], origin: dt.date) -> tuple[float | None, int]:
    cutoff = origin - dt.timedelta(days=1)
    start = cutoff - dt.timedelta(days=29)
    if isinstance(prices, dict):
        points = sorted(prices.items())
    else:
        points = prices
    lo = bisect.bisect_left(points, (start, -math.inf))
    hi = bisect.bisect_right(points, (cutoff, math.inf))
    points = points[lo:hi]
    if len(points) < 2:
        return None, len(points)
    squared = [math.log(points[i][1] / points[i - 1][1]) ** 2 / (points[i][0] - points[i - 1][0]).days
               for i in range(1, len(points)) if (points[i][0] - points[i - 1][0]).days > 0]
    return (math.sqrt(math.fsum(squared) / len(squared)), len(points)) if squared else (None, len(points))


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    data = sorted(values)
    pos = (len(data) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def _vol_bin(value: float | None, cuts: tuple[float, float] | None) -> str:
    if value is None or cuts is None:
        return "unavailable"
    if value <= cuts[0]:
        return "low_tertile"
    if value <= cuts[1]:
        return "middle_tertile"
    return "high_tertile"


def _metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result = {"nPairedKeys": len(rows), "uniqueTargetDates": len({r["target"] for r in rows}), "byMethod": {}}
    for method in ONLINE_METHODS:
        vals = [(r["actual"], r["predictions"][method]) for r in rows]
        if any(p is None for _, p in vals):
            raise ValueError("metric group escaped exact common-key subset")
        ae = [abs(a - p) for a, p in vals]
        result["byMethod"][method] = {
            "n": len(vals),
            "mae": math.fsum(ae) / len(vals) if vals else None,
            "mape": 100 * math.fsum(e / a for e, (a, _) in zip(ae, vals)) / len(vals) if vals else None,
        }
    base = result["byMethod"]["fixed50"]
    result["pairedDeltaVsFixed50"] = {
        m: {"maeChange": result["byMethod"][m]["mae"] - base["mae"] if rows else None,
            "mapeChangePp": result["byMethod"][m]["mape"] - base["mape"] if rows else None}
        for m in ONLINE_METHODS if m != "fixed50"
    }
    return result


def _diagnose_2026(manifest: dict[str, Any], prices: dict[str, dict[dt.date, float]]) -> dict[str, Any]:
    source = _manifest_rows(manifest)
    sorted_prices = {p: sorted(values.items()) for p, values in prices.items()}
    paired = []
    planned = 0
    for (p, day, h), row in source.items():
        if not ONLINE_START <= day <= ONLINE_END:
            continue
        planned += 1
        actual = prices[p].get(day)
        preds = {m: _pred(row, m) for m in ONLINE_METHODS}
        if actual is None or any(v is None for v in preds.values()):
            continue
        origin = day - dt.timedelta(days=h)
        age = _age(row)
        volatility, vol_obs = _volatility(sorted_prices[p], origin)
        paired.append({"product": p, "target": day, "origin": origin, "horizon": h, "actual": actual,
                       "predictions": preds, "p0AgeDays": age, "volatility30d": volatility,
                       "volatilityPriceObservations": vol_obs})
    cuts = {}
    for p in PRODUCTS:
        vals = [r["volatility30d"] for r in paired if r["product"] == p and r["volatility30d"] is not None]
        cuts[p] = (_quantile(vals, 1/3), _quantile(vals, 2/3)) if vals else None
    strata = {"crop": {}, "horizonBand": {}, "p0AgeBand": {}, "prior30dVolatilityTertile": {}}
    groups: dict[str, dict[str, list[dict[str, Any]]]] = {k: {} for k in strata}
    for r in paired:
        groups["crop"].setdefault(r["product"], []).append(r)
        hband = "1-3" if r["horizon"] <= 3 else "4-10"
        groups["horizonBand"].setdefault(hband, []).append(r)
        age = r["p0AgeDays"]
        ageband = "0-3d" if age is not None and age <= 3 else "4-7d" if age is not None and age <= 7 else "8-14d" if age is not None and age <= 14 else "unavailable"
        groups["p0AgeBand"].setdefault(ageband, []).append(r)
        vbin = _vol_bin(r["volatility30d"], cuts[r["product"]])
        groups["prior30dVolatilityTertile"].setdefault(f"{r['product']}:{vbin}", []).append(r)
    for dim, dim_groups in groups.items():
        for name, values in sorted(dim_groups.items()):
            stats = _metric_rows(values)
            if dim == "prior30dVolatilityTertile":
                crop, bin_name = name.split(":", 1)
                stats["tertileCutpoints"] = cuts[crop]
                stats["volatilityDefinition"] = "RMS log return per sqrt(calendar day), adjacent positive price observations within origin-30d..origin-1d"
            strata[dim][name] = stats
    all_metrics = _metric_rows(paired)
    # Sequential paired deltas expose, descriptively, timing-protocol gap and
    # remaining model gap on exactly the same keys.
    baseline = all_metrics["byMethod"]["fixed50"]
    protocol = all_metrics["byMethod"]["causal50"]
    all_metrics["descriptiveGapDecomposition"] = {
        "causal50MinusFixed50": {"maeChange": protocol["mae"] - baseline["mae"] if paired else None,
                                  "mapeChangePp": protocol["mape"] - baseline["mape"] if paired else None,
                                  "interpretation": "paired protocol/timing difference: historical same-day fixed50 vs causalized 50:50; not causal attribution"},
        "remainingCandidatesMinusCausal50": {
            m: {"maeChange": all_metrics["byMethod"][m]["mae"] - protocol["mae"] if paired else None,
                "mapeChangePp": all_metrics["byMethod"][m]["mape"] - protocol["mape"] if paired else None}
            for m in ("online71Blend", "online71", "causalDirect")
        },
        "interpretation": "descriptive same-key arithmetic only; cannot isolate causal effects of model changes",
    }
    return {
        "period": [ONLINE_START.isoformat(), ONLINE_END.isoformat()],
        "plannedManifestKeysInPeriod": planned,
        "exactCommonKeysWithPositiveActualAndAllFiveIssued": len(paired),
        "uniqueTargetDates": len({r["target"] for r in paired}),
        "metrics": all_metrics,
        "strata": strata,
        "features": {
            "p0AgeBands": {"0-3d": [0,3], "4-7d": [4,7], "8-14d": [8,14], "unavailable": None},
            "volatilityTertilesByCrop": {p: cuts[p] for p in PRODUCTS},
            "volatilityUsesOnlyObservationsThroughOriginMinus1": True,
            "pairedSubsetRule": "same product/target/horizon keys with positive target actual and issued positive predictions from all five methods",
        },
    }


def _feature_inventory() -> dict[str, Any]:
    original = list(modeling.FEATURE_NAMES)
    expanded = list(library.FEATURE_NAMES)
    if len(original) != 40 or len(expanded) != 71:
        raise ValueError("unexpected feature schema size")
    overlap = sorted(set(original) & set(expanded))
    return {
        "originalDirectFeatureCount": len(original),
        "fullHistoryFeatureCount": len(expanded),
        "sharedNames": overlap,
        "originalFeatureNamesNotReused": [x for x in original if x not in set(expanded)],
        "fullHistoryLibraryFeatures": expanded,
        "confirmedRepresentationChange": "The 71-feature library is a different 40-feature relative-price/availability/age/calendar/window/volume base plus 31 additions; it is not original 40 plus 31. None of the 40 original feature names are reused, but this name-level difference does not mean every signal was deleted: weekday/season and source-age concepts are represented in both schemas.",
        "omittedSignalGroups": [
            {"group": "absolute log price levels", "features": [x for x in original if x == "lastPriceLog" or "PriceLog" in x or "MeanLogPrice" in x or "MinLogPrice" in x or "MaxLogPrice" in x], "replacement": "ratios/relative logs anchored to latest source price; no explicit absolute log-price level"},
            {"group": "calendar-origin lag levels and availability", "features": [x for x in original if x.startswith("lag")], "replacement": "relative source-anchored lookbacks at p0-1/7/30 plus p0 age; expanded windows are p0-anchored, not origin-calendar lag cutoffs"},
            {"group": "absolute log volume by calendar window", "features": [x for x in original if x.endswith("LogVolumeKg")], "replacement": "7/14/30/60/90-day source-anchored relative volume ratios to prior medians; absolute scale omitted"},
            {"group": "calendar-window price level summaries", "features": [x for x in original if x.startswith("window")], "replacement": "source-anchored relative-price distribution statistics over p0 calendar windows"},
        ],
        "targetScalingAnchor": "The 71-feature return function cannot condition on absolute price/volume scale, but the final forecast is p0*exp(predicted return), so p0 price level remains the forecast scaling anchor.",
        "horizonFeatureNote": "Horizon is constant within each separately fitted horizon model in the direct setup; its absence from the 71-vector alone is not treated as meaningful information loss.",
        "remedyGuidance": "Preserve the original 40 direct features as the base and add the 31-feature relative/source-anchored block as a separately attributable expansion. A future comparison should use a preregistered equal-information split and an untouched period; do not infer that restoring a feature will improve results from this inventory alone.",
        "interpretationLimit": "This code comparison confirms representation changes, not that an omitted signal caused any measured forecast failure.",
        "syntheticScaleInvarianceAudit": _scale_invariance_audit(),
        "sourceSha256": {"originalDirectFeatures": _sha(Path(modeling.__file__)), "fullHistoryFeatureLibrary": _sha(Path(library.__file__))},
    }


def _scale_invariance_audit() -> dict[str, Any]:
    """Show which feature vectors can distinguish price/volume unit scaling."""
    start = dt.date(2024, 1, 1)
    days = [start + dt.timedelta(days=i) for i in range(180)]
    prices = {d: 100.0 + 0.35 * i + 5.0 * math.sin(i / 7.0) for i, d in enumerate(days)}
    records = {d: {"price": prices[d], "volume": 1000.0 + 11.0 * i,
                   "rowCount": 20 + i % 7, "markets": frozenset(library.MARKET_CODES)}
               for i, d in enumerate(days)}
    price_scale, volume_scale = 3.0, 4.0
    scaled_prices = {d: v * price_scale for d, v in prices.items()}
    scaled_records = {d: {**r, "price": r["price"] * price_scale,
                          "volume": r["volume"] * volume_scale} for d, r in records.items()}
    target, horizon = days[-1] + dt.timedelta(days=5), 3
    x, _, _ = library.features(records, target, horizon, days, prices)
    x_scaled, _, _ = library.features(scaled_records, target, horizon, days, scaled_prices)
    _, original = modeling._feature_row({d: (r["price"], r["volume"]) for d, r in records.items()}, target-dt.timedelta(days=horizon), target, horizon, {})
    _, original_scaled = modeling._feature_row({d: (r["price"], r["volume"]) for d, r in scaled_records.items()}, target-dt.timedelta(days=horizon), target, horizon, {})
    if x is None or x_scaled is None or original is None or original_scaled is None:
        raise ValueError("synthetic feature invariance audit failed to build vectors")
    library_max_delta = max(abs(a-b) for a,b in zip(x, x_scaled))
    original_max_delta = max(abs(a-b) for a,b in zip(original, original_scaled))
    return {
        "fixture": "synthetic smooth positive prices/volumes; no observed outcomes or fitting",
        "priceMultiplier": price_scale,
        "volumeMultiplier": volume_scale,
        "fullHistory71MaxAbsoluteFeatureDelta": library_max_delta,
        "originalDirect40MaxAbsoluteFeatureDelta": original_max_delta,
        "fullHistory71InvariantToJointScale": library_max_delta <= 1e-12,
        "originalDirect40ChangesUnderScale": original_max_delta > 1e-6,
        "interpretation": "The relative 71-feature representation cannot distinguish this uniform price-and-volume scale shift; original 40 includes explicit absolute log-price and absolute log-volume coordinates. This is a representational fact, not evidence that the shift caused forecast error.",
    }


def diagnose(components: dict[str, Any], prices_raw: dict[str, Any], prior_report: dict[str, Any], online_manifest: dict[str, Any], *, input_hashes: dict[str, str] | None = None) -> dict[str, Any]:
    prices_historical, _ = fixed._price_map(prices_raw)
    hist_rows = _historical_rows(components, prices_historical)
    if prior_report.get("reportSha256") is None or prior_report.get("results", {}).get("commonPairs") != sum(r["actual"] is not None and r["fixed50"] is not None for r in hist_rows):
        raise ValueError("fixed50 residual report does not match reconstructed 2025 paired count")
    all_prices = _all_price_maps(prices_raw, ONLINE_END)
    result = {
        "schemaVersion": 1,
        "protocol": "failure-mechanism-diagnostic-20261010-v1",
        "phase": "read_only_forensic_diagnostic",
        "modelFitting": False,
        "newApiAccess": False,
        "inputSha256": input_hashes or {},
        "sectionA_fixed50Correction": _diagnose_2025(hist_rows),
        "sectionB_exactCommon2026": _diagnose_2026(online_manifest, all_prices),
        "sectionC_featureRepresentation": _feature_inventory(),
        "limitations": [
            "Section A is a retrospective diagnostic over a previously examined 2025 period; its cell oracle uses each cell's own outcomes and is hindsight-only.",
            "Section B conditions on exact common keys, positive actuals, and all five issued forecasts; it is not coverage over the full calendar opportunity grid.",
            "Section B comparisons are descriptive and do not add DM inference or causal claims; consult the already completed evaluation reports for the prior inferential assessment.",
            "Section C reports source-code representation differences only and does not establish causal importance.",
        ],
    }
    return result


def _check_inputs() -> None:
    paths = {
        "historicalComponents": COMPONENT_PATH,
        "prices": PRICES_PATH,
        "fixed50ResidualReport": FIXED_REPORT_PATH,
        "equalInfoOnlineManifest": ONLINE_MANIFEST_PATH,
    }
    for key, path in paths.items():
        if _sha(path) != INPUT_SHA256[key]:
            raise ValueError(f"pinned {key} SHA-256 mismatch")


def diagnose_from_paths() -> dict[str, Any]:
    _check_inputs()
    return diagnose(
        json.loads(COMPONENT_PATH.read_text()), json.loads(PRICES_PATH.read_text()),
        json.loads(FIXED_REPORT_PATH.read_text()), json.loads(ONLINE_MANIFEST_PATH.read_text()),
        input_hashes=INPUT_SHA256,
    )


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=OUT)
    args = parser.parse_args(argv)
    write_exclusive(args.output, diagnose_from_paths())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
