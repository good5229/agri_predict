"""Frozen offline exploratory national residual correction using two-market gaps.

Forecast generation and scoring are separate. Candidate manifests contain no target
values; scoring rebuilds every feature and forecast from the pinned source bundle.
"""
from __future__ import annotations

import datetime as dt
import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import direct_horizon, national_price_selection as base

PROTOCOL_ID = "national-gap-residual-2026-10-09"
PRODUCTS = base.PRODUCTS
HORIZONS = base.HORIZONS
WINDOWS = ((dt.date(2026, 1, 1), dt.date(2026, 3, 31)),
           (dt.date(2026, 4, 1), dt.date(2026, 6, 30)),
           (dt.date(2026, 7, 1), dt.date(2026, 9, 30)))
METHODS = ("gap_residual_full", "gap_residual_no_market", "ridge_alpha_100", "calendar_sarima", "persistence")
COMPARATORS = ("ridge_alpha_100", "calendar_sarima", "gap_residual_no_market", "persistence")
BANDWIDTHS = (9, 28)
MIN_PAIRS = 30
MIN_LABEL_DAYS = 60
ALPHA = 100.0
CLIP = 0.15
MARKET_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
MANIFEST_SHA256 = {
    "development": "8209689fbaddd33ffdfe7f30a87b200fb1fa881d5788e38ffebb13de8974b2e8",
    "fresh": "925ac27d569a1ca299451f7f4a3073bcece99b7e0901d9bbc2cf3d1a089f3fa4",
}
MANIFEST_CONTENT_SHA256 = {
    "development": "955d6748e52137c52242ec764288f75e7bc07ce984463ec42093b5553d11f16a",
    "fresh": "d49bcc8ce1ae91f0093980e13bb899b2b6a8dd7d3e6291915e26e5a5be2e719a",
}
MANIFEST_PATHS = {
    "development": "outputs/experiments/national-price-selection-20261009/development-manifest.json",
    "fresh": "outputs/experiments/national-price-selection-20261009/fresh-manifest.json",
}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = PROJECT_ROOT / "data/raw/national-wholesale-20240708-20260630.zip"
DEFAULT_COVERAGE = PROJECT_ROOT / "data/raw/nongnet_national_fresh_holdout_20260701-20260930/coverage-report.json"
DEFAULT_MARKET_SNAPSHOT = PROJECT_ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hash(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def _positive(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def _date(text: str) -> dt.date:
    return dt.date.fromisoformat(text)


def _verify_input_manifest(doc: dict, kind: str, raw_bundle: dict) -> dict:
    if kind not in MANIFEST_SHA256:
        raise ValueError("input manifest hash mismatch or unsupported kind")
    path = Path(__file__).resolve().parent.parent / MANIFEST_PATHS[kind]
    raw_manifest = path.read_bytes()
    if _sha(raw_manifest) != MANIFEST_SHA256[kind]:
        raise ValueError(f"pinned {kind} manifest file SHA-256 mismatch")
    pinned_doc = json.loads(raw_manifest)
    if doc != pinned_doc or doc.get("manifestSha256") != MANIFEST_CONTENT_SHA256[kind]:
        raise ValueError(f"{kind} manifest differs from the frozen input")
    if _hash({k: v for k, v in doc.items() if k != "manifestSha256"}) != doc.get("manifestSha256"):
        raise ValueError("input manifest canonical hash mismatch")
    base._verify_manifest(doc)
    source = base._load_source(raw_bundle)
    if doc.get("sourceDigests") != source.get("sourceDigests"):
        raise ValueError(f"{kind} source digests mismatch")
    lo, hi = (_date(x) for x in doc["window"])
    rebuilt = base._build_manifest(source, source["sourceDigests"], start=lo, end=hi)
    if doc["forecasts"] != rebuilt["forecasts"]:
        raise ValueError(f"{kind} forecasts do not replay")
    return source


def _verified_sources(development: dict, fresh: dict, raw_bundles: dict) -> tuple[dict, dict]:
    """Verify each frozen forecast against its own provenance bundle."""
    if not isinstance(raw_bundles, dict) or set(raw_bundles) != {"development", "fresh"}:
        raise ValueError("raw_bundles must contain development and fresh bundles")
    development_bundle, fresh_bundle = raw_bundles["development"], raw_bundles["fresh"]
    if set(development_bundle) != {"archivePath"}:
        raise ValueError("development bundle must be archive-only")
    if set(fresh_bundle) != {"archivePath", "coverageReportPath"}:
        raise ValueError("fresh bundle must include archive and coverage report")
    development_source = _verify_input_manifest(development, "development", development_bundle)
    fresh_source = _verify_input_manifest(fresh, "fresh", fresh_bundle)
    if development_source["sourceDigests"].get("archiveSha256") != fresh_source["sourceDigests"].get("archiveSha256"):
        raise ValueError("development and fresh inputs do not share the pinned national archive")
    return development_source, fresh_source


def _market_series(path: str | Path) -> tuple[dict[str, dict[dt.date, float]], str]:
    raw = Path(path).read_bytes()
    digest = _sha(raw)
    if digest != MARKET_SHA256:
        raise ValueError("two-market snapshot SHA-256 mismatch")
    doc = json.loads(raw)
    rows = doc.get("products")
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError("two-market snapshot products invalid")
    ids = [r.get("id") for r in rows]
    if len(ids) != len(set(ids)) or not set(PRODUCTS).issubset(ids):
        raise ValueError("two-market snapshot product keys mismatch")
    result = {}
    for product in PRODUCTS:
        p = next(r for r in rows if r["id"] == product)
        series = {}
        for obs in p.get("observations", []):
            day = _date(obs.get("date"))
            price = _positive(obs.get("actual"))
            if day in series:
                raise ValueError("duplicate two-market product/date")
            if price is not None:
                series[day] = price
        result[product] = series
    return result, digest


def _delta(series: dict[dt.date, float], cutoff: dt.date) -> tuple[float | None, str | None, str | None]:
    prior = sorted((d, v) for d, v in series.items() if cutoff - dt.timedelta(days=20) <= d <= cutoff - dt.timedelta(days=7))
    recent = sorted((d, v) for d, v in series.items() if cutoff - dt.timedelta(days=13) <= d <= cutoff)
    if not prior or not recent:
        return None, prior[-1][0].isoformat() if prior else None, recent[-1][0].isoformat() if recent else None
    return math.log(recent[-1][1] / prior[-1][1]), prior[-1][0].isoformat(), recent[-1][0].isoformat()


def _features(product: str, row: dict, national: dict[dt.date, float], market: dict[dt.date, float]) -> tuple[list[float], list[float], dict]:
    origin, target = _date(row["origin"]), _date(row["target"])
    cutoff = origin - dt.timedelta(days=2)
    overlap = sorted(d for d in national.keys() & market.keys() if cutoff - dt.timedelta(days=13) <= d <= cutoff)
    gap = math.log(market[overlap[-1]] / national[overlap[-1]]) if overlap else None
    nd, nprior, nrecent = _delta(national, cutoff)
    md, mprior, mrecent = _delta(market, cutoff)
    ridge = _positive(row["predictions"].get("ridge_alpha_100"))
    sarima = _positive(row["predictions"].get("calendar_sarima"))
    h = int(row["horizon"])
    component_ratio = math.log(sarima / ridge) if ridge is not None and sarima is not None else None
    common = [nd, component_ratio, float(h)]
    full = [gap, md, *common]
    audit = {"cutoff": cutoff.isoformat(), "gapDate": overlap[-1].isoformat() if overlap else None,
             "nationalChangeReferenceDate": nprior, "nationalChangeRecentDate": nrecent,
             "marketChangeReferenceDate": mprior, "marketChangeRecentDate": mrecent,
             "nationalChange": nd, "nationalChangeAvailable": nd is not None,
             "componentRatio": component_ratio, "componentRatioAvailable": component_ratio is not None,
             "commonFeaturesAvailable": nd is not None and component_ratio is not None,
             "commonFeaturesReason": ("national_change_unavailable" if nd is None else
                                       "ridge_or_sarima_not_positive_finite" if component_ratio is None else None),
             "marketChange": md, "marketFeaturesAvailable": gap is not None and md is not None}
    return full, common, audit


def _merge_manifests(development: dict, fresh: dict) -> list[dict]:
    rows = development["forecasts"] + fresh["forecasts"]
    keyed = {}
    for row in rows:
        key = (row["product"], row["origin"], row["target"], row["horizon"])
        if key in keyed:
            if keyed[key] != row:
                raise ValueError("input manifest key collision with unequal rows")
            continue
        if _date(row["target"]) - _date(row["origin"]) != dt.timedelta(days=row["horizon"]):
            raise ValueError("input forecast key/horizon mismatch")
        if row["cutoff"] != (_date(row["origin"]) - dt.timedelta(days=2)).isoformat():
            raise ValueError("input forecast cutoff mismatch")
        keyed[key] = row
    return sorted(keyed.values(), key=lambda r: (r["product"], r["origin"], r["horizon"]))


def _row_key(row: dict) -> tuple:
    return (row["product"], row["origin"], row["target"], row["horizon"])


def _fit_month(rows: list[dict], product: str, month_start: dt.date,
               national: dict[dt.date, float], market: dict[dt.date, float],
               feature_cache: dict[tuple, tuple[list, list, dict]] | None = None) -> dict:
    """Fit using only mature targets. Kept pure enough for boundary tests."""
    fit_cutoff = month_start - dt.timedelta(days=2)
    xs_full, y_full, full_days = [], [], set()
    xs_common, y_common, common_days = [], [], set()
    for hist in rows:
        if hist["product"] != product or _date(hist["target"]) > fit_cutoff:
            continue
        actual = _positive(national.get(_date(hist["target"])))
        ridge = _positive(hist["predictions"].get("ridge_alpha_100"))
        if actual is None or ridge is None:
            continue
        hf, hc, feature_audit = (feature_cache[_row_key(hist)] if feature_cache is not None
                                 else _features(product, hist, national, market))
        if not feature_audit["commonFeaturesAvailable"]:
            continue
        residual = math.log(actual / ridge)
        xs_common.append(hc); y_common.append(residual); common_days.add(hist["target"])
        if feature_audit["marketFeaturesAvailable"]:
            xs_full.append(hf); y_full.append(residual); full_days.add(hist["target"])
    entry = {"fitCutoff": fit_cutoff.isoformat(), "trainingRows": len(y_common), "uniqueTargetDays": len(common_days),
             "maxLabelDate": max(common_days) if common_days else None, "fullTrainingRows": len(y_full),
             "fullUniqueTargetDays": len(full_days), "fullMaxLabelDate": max(full_days) if full_days else None,
             "fullModel": None, "noMarketModel": None, "fallbackReason": None,
             "fullFallbackReason": None, "noMarketFallbackReason": None}
    if len(common_days) < MIN_LABEL_DAYS:
        entry["fallbackReason"] = "fewer_than_60_unique_mature_target_days"
        entry["fullFallbackReason"] = entry["fallbackReason"]
        entry["noMarketFallbackReason"] = entry["fallbackReason"]
    else:
        if len(full_days) < MIN_LABEL_DAYS:
            entry["fullFallbackReason"] = "fewer_than_60_unique_mature_market_feature_target_days"
        else:
            try:
                entry["fullModel"] = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(np.asarray(xs_full), np.asarray(y_full))
            except Exception as exc:
                entry["fullFallbackReason"] = "fit_failed:" + type(exc).__name__
        try:
            entry["noMarketModel"] = make_pipeline(StandardScaler(), Ridge(alpha=ALPHA)).fit(np.asarray(xs_common), np.asarray(y_common))
        except Exception as exc:
            entry["noMarketFallbackReason"] = "fit_failed:" + type(exc).__name__
    return entry


def _correct(model: Any, features: list[float | None], ridge: Any,
             fallback_reason: str | None = None) -> tuple[float | None, dict]:
    """Apply a finite capped log correction or return the exact Ridge fallback."""
    base_price = _positive(ridge)
    if base_price is None:
        return None, {"fallback": False, "fallbackReason": "ridge100_not_positive_finite",
                      "rawLogCorrection": None, "appliedLogCorrection": None}
    reason = fallback_reason
    if reason:
        return base_price, {"fallback": True, "fallbackReason": reason,
                            "rawLogCorrection": None, "appliedLogCorrection": 0.0}
    try:
        raw = float(model.predict(np.asarray([features]))[0])
        if not math.isfinite(raw):
            raise ValueError("nonfinite_log_correction")
        applied = min(CLIP, max(-CLIP, raw))
        value = base_price * math.exp(applied)
        if not math.isfinite(value) or value <= 0:
            raise ValueError("corrected_prediction_not_positive_finite")
        return value, {"fallback": False, "fallbackReason": None,
                       "rawLogCorrection": raw, "appliedLogCorrection": applied}
    except Exception as exc:
        reason = str(exc) if isinstance(exc, ValueError) else "correction_failed:" + type(exc).__name__
        return base_price, {"fallback": True, "fallbackReason": reason,
                            "rawLogCorrection": None, "appliedLogCorrection": 0.0}


def build_manifest(development: dict, fresh: dict, raw_bundles: dict, market_snapshot_path: str | Path) -> dict:
    """Build an answer-free residual-correction manifest; no actual targets are attached."""
    _, source = _verified_sources(development, fresh, raw_bundles)
    markets, market_sha = _market_series(market_snapshot_path)
    national = base._prices(source)
    rows = _merge_manifests(development, fresh)
    feature_cache = {_row_key(r): _features(r["product"], r, national[r["product"]], markets[r["product"]]) for r in rows}
    by_product_month: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        if dt.date(2026, 1, 1) <= _date(r["target"]) <= dt.date(2026, 9, 30):
            by_product_month[(r["product"], _date(r["origin"]).strftime("%Y-%m"))].append(r)
    cached: dict[tuple[str, str, tuple], tuple[Any, dict]] = {}
    out = []
    for row in rows:
        product, origin = row["product"], _date(row["origin"])
        if not dt.date(2026, 1, 1) <= _date(row["target"]) <= dt.date(2026, 9, 30):
            continue
        full, common, feature_audit = feature_cache[_row_key(row)]
        key = (product, origin.strftime("%Y-%m"))
        if key not in cached:
            month_rows = by_product_month[key]
            month_start = min(_date(r["origin"]) for r in month_rows)
            entry = _fit_month(rows, product, month_start, national[product], markets[product], feature_cache)
            cached[key] = (entry, {"trainingRows": entry["trainingRows"], "uniqueTargetDays": entry["uniqueTargetDays"]})
        fit, _ = cached[key]
        preds, lineage = {}, {}
        ridge = _positive(row["predictions"].get("ridge_alpha_100"))
        for method, model, x, market_required, reason_key in (("gap_residual_full", fit["fullModel"], full, True, "fullFallbackReason"),
                                                              ("gap_residual_no_market", fit["noMarketModel"], common, False, "noMarketFallbackReason")):
            reason = fit[reason_key] or fit["fallbackReason"]
            if ridge is None: reason = "ridge100_not_positive_finite"
            elif not feature_audit["commonFeaturesAvailable"]: reason = feature_audit["commonFeaturesReason"]
            elif market_required and not feature_audit["marketFeaturesAvailable"]: reason = "market_features_unavailable"
            value, correction_audit = _correct(model, x, ridge, reason or ("model_fit_unavailable" if model is None else None))
            preds[method] = value
            lineage[method] = {**correction_audit, "fitCutoff": fit["fitCutoff"],
                "trainingRows": fit["trainingRows"], "uniqueTargetDays": fit["uniqueTargetDays"],
                "maxLabelDate": fit["maxLabelDate"], "fullTrainingRows": fit["fullTrainingRows"],
                "fullUniqueTargetDays": fit["fullUniqueTargetDays"]}
        out.append({**{k: row[k] for k in ("product", "origin", "target", "horizon", "cutoff")},
                    "predictions": {**row["predictions"], **preds}, "features": feature_audit,
                    "fit": {k: fit[k] for k in ("fitCutoff", "trainingRows", "uniqueTargetDays", "maxLabelDate",
                        "fullTrainingRows", "fullUniqueTargetDays", "fullMaxLabelDate",
                        "fallbackReason", "fullFallbackReason", "noMarketFallbackReason")},
                    "lineage": lineage})
    body = {"schemaVersion": 1, "protocolId": PROTOCOL_ID, "phase": "forecast_only", "targetValuesAttached": False,
            "inputManifestSha256": {"development": development["manifestSha256"], "fresh": fresh["manifestSha256"]},
            "sourceDigests": source["sourceDigests"], "marketSnapshotSha256": market_sha,
            "codeSha256": _sha(Path(__file__).read_bytes()), "products": list(PRODUCTS), "horizons": list(HORIZONS),
            "window": ["2026-01-01", "2026-09-30"], "methods": list(METHODS), "plannedForecastCount": len(out),
            "forecasts": out}
    body["manifestSha256"] = _hash(body)
    return body


def verify_manifest(doc: dict) -> None:
    if doc.get("protocolId") != PROTOCOL_ID or doc.get("phase") != "forecast_only" or doc.get("targetValuesAttached") is not False:
        raise ValueError("not an answer-free national gap manifest")
    if doc.get("codeSha256") != _sha(Path(__file__).read_bytes()):
        raise ValueError("candidate code hash mismatch")
    saved = doc.get("manifestSha256")
    if saved != _hash({k: v for k, v in doc.items() if k != "manifestSha256"}):
        raise ValueError("candidate manifest hash mismatch")
    if len(doc.get("forecasts", [])) != doc.get("plannedForecastCount"):
        raise ValueError("candidate manifest count mismatch")
    seen = set()
    for r in doc["forecasts"]:
        key = (r["product"], r["origin"], r["target"], r["horizon"])
        if key in seen: raise ValueError("duplicate candidate key")
        seen.add(key)
        if any(k in r for k in ("actual", "targetValue")): raise ValueError("answer field in candidate manifest")


def _verify_replay(doc: dict, development: dict, fresh: dict, raw_bundles: dict,
                   market_snapshot_path: str | Path) -> None:
    verify_manifest(doc)
    rebuilt = build_manifest(development, fresh, raw_bundles, market_snapshot_path)
    if doc.get("forecasts") != rebuilt.get("forecasts") or doc.get("manifestSha256") != rebuilt.get("manifestSha256"):
        raise ValueError("candidate manifest replay mismatch")


def _dm(diffs: dict[dt.date, float], bandwidth: int) -> dict:
    keys = sorted(diffs)
    n = len(keys)
    mean = float(np.mean([diffs[d] for d in keys])) if n else None
    if n < MIN_PAIRS:
        return {"n": n, "status": "inference_unavailable_insufficient_pairs", "meanLossDifference": mean, "statistic": None, "pValue": 1.0}
    centered = {d: diffs[d] - mean for d in keys}
    variance = sum(v*v for v in centered.values())
    for lag in range(1, bandwidth + 1):
        cov = sum(v * centered.get(d - dt.timedelta(days=lag), 0.0) for d, v in centered.items())
        variance += 2 * (1 - lag / (bandwidth + 1)) * cov
    if not math.isfinite(variance) or variance <= 0:
        return {"n": n, "status": "inference_unavailable_nonpositive_variance", "meanLossDifference": mean, "statistic": None, "pValue": 1.0}
    stat = mean / (math.sqrt(variance) / n)
    return {"n": n, "status": "available", "meanLossDifference": mean, "statistic": stat,
            "pValue": math.erfc(abs(stat) / math.sqrt(2)), "calendarHacMaxLagDays": bandwidth}


def _holm(items: list[dict]) -> None:
    ranked = sorted(range(len(items)), key=lambda i: items[i]["pValue"])
    running = 0.0
    for rank, idx in enumerate(ranked):
        running = max(running, min(1.0, (len(items) - rank) * items[idx]["pValue"]))
        items[idx]["holmAdjustedPValue"] = running
        raw = items[idx]
        raw["holmSignificant"] = raw["status"] == "available" and running < .05
        raw["conclusion"] = ("inference_unavailable" if raw["status"] != "available" else
                             "improvement" if raw["holmSignificant"] and raw["meanLossDifference"] < 0 else
                             "worsening" if raw["holmSignificant"] and raw["meanLossDifference"] > 0 else "inconclusive")


def score_manifest(doc: dict, development: dict, fresh: dict, raw_bundles: dict, market_snapshot_path: str | Path) -> dict:
    """Replay the frozen answer-free rows, then attach labels and assess fixed families."""
    _verify_replay(doc, development, fresh, raw_bundles, market_snapshot_path)
    _, source = _verified_sources(development, fresh, raw_bundles)
    actuals = base._prices(source)
    status_evidence = base._coverage_status_evidence(raw_bundles["fresh"])
    allrows = doc["forecasts"]
    report = {"protocolId": PROTOCOL_ID, "candidateManifestSha256": doc["manifestSha256"], "periods": {}}
    for lo, hi in WINDOWS:
        selected = [r for r in allrows if lo <= _date(r["target"]) <= hi]
        scored = []
        for r in selected:
            actual = actuals[r["product"]].get(_date(r["target"]))
            if actual is not None and all(_positive(r["predictions"].get(m)) for m in METHODS):
                scored.append({**r, "actual": actual})
        scored_keys = {_row_key(r) for r in scored}
        cells = {}
        for p in PRODUCTS:
            for h in HORIZONS:
                cell = [r for r in scored if r["product"] == p and r["horizon"] == h]
                cells[f"{p}/{h}"] = {"n": len(cell), "metrics": {m: direct_horizon._metrics(
                    [r["actual"] for r in cell], [r["predictions"][m] for r in cell]) for m in METHODS}}
        complete = all(c["n"] >= MIN_PAIRS for c in cells.values())
        macro = {m: {loss: (sum(c["metrics"][m][loss] for c in cells.values()) / 40 if complete else None)
                     for loss in ("mae", "mape")} for m in METHODS}
        coverage = {m: {"planned": len(selected),
                        "issued": sum(_positive(r["predictions"].get(m)) is not None for r in selected),
                        "issuedOnValidTargets": sum(_date(r["target"]) in actuals[r["product"]]
                                                    and _positive(r["predictions"].get(m)) is not None for r in selected)}
                    for m in METHODS}
        per_product = {}
        for p in PRODUCTS:
            per_product[p] = {}
            for m in METHODS:
                product_rows = [r for r in scored if r["product"] == p]
                pooled = direct_horizon._metrics([r["actual"] for r in product_rows], [r["predictions"][m] for r in product_rows])
                horizon_mapes = [cells[f"{p}/{h}"]["metrics"][m]["mape"] for h in HORIZONS
                                 if cells[f"{p}/{h}"]["metrics"][m]["mape"] is not None]
                horizon_maes = [cells[f"{p}/{h}"]["metrics"][m]["mae"] for h in HORIZONS
                                if cells[f"{p}/{h}"]["metrics"][m]["mae"] is not None]
                per_product[p][m] = {**pooled,
                    "equalHorizonMape": float(np.mean(horizon_mapes)) if len(horizon_mapes) == 10 else None,
                    "equalHorizonMae": float(np.mean(horizon_maes)) if len(horizon_maes) == 10 else None}
        tests = []
        for p in PRODUCTS:
            for h in HORIZONS:
                cell = [r for r in scored if r["product"] == p and r["horizon"] == h]
                for comp in COMPARATORS:
                    for loss in ("mae", "mape"):
                        diffs = {}
                        for r in cell:
                            a, b, y = r["predictions"]["gap_residual_full"], r["predictions"][comp], r["actual"]
                            diffs[_date(r["target"])] = (abs(a-y)-abs(b-y) if loss == "mae" else 100*(abs(a-y)-abs(b-y))/y)
                        item = {"product": p, "horizon": h, "comparator": comp, "loss": loss, "n": len(diffs), "dm": {}}
                        for bw in BANDWIDTHS: item["dm"][str(bw)] = _dm(diffs, bw)
                        tests.append(item)
        for bw in BANDWIDTHS: _holm([r["dm"][str(bw)] for r in tests])
        per_cell = []
        for p in PRODUCTS:
            for h in HORIZONS:
                cell = [r for r in scored if r["product"] == p and r["horizon"] == h]
                for comp in COMPARATORS:
                    per_cell.append({"product": p, "horizon": h, "comparator": comp, "n": len(cell),
                        "meanPairedMaeDifference": float(np.mean([abs(r["predictions"]["gap_residual_full"]-r["actual"])-abs(r["predictions"][comp]-r["actual"]) for r in cell])) if cell else None,
                        "meanPairedMapeDifferencePp": float(np.mean([100*(abs(r["predictions"]["gap_residual_full"]-r["actual"])-abs(r["predictions"][comp]-r["actual"]))/r["actual"] for r in cell])) if cell else None})
        joint = _joint_verdicts(tests)
        gate = _period_gate(complete, macro, per_product, coverage, joint) if lo.month in (4, 7) else None
        report["periods"][f"{lo.isoformat()}_{hi.isoformat()}"] = {"planned": len(selected), "issuedCoverage": coverage,
            "validCommonPairs": len(scored), "exclusionsByState": _exclusion_states(selected, source, actuals, status_evidence, scored_keys),
            "complete40Cells": complete, "cells": cells, "macro40EqualCell": macro, "perProduct": per_product,
            "pairedCellLosses": per_cell, "dmComparisons": tests, "jointCellVerdicts": joint,
            "predeclaredGate": gate, "holmFamilySizePerBandwidth": len(tests)}
    later = [v.get("predeclaredGate") for k, v in report["periods"].items() if k.startswith("2026-04") or k.startswith("2026-07")]
    report["twoWindowPredeclaredConclusion"] = ("pass" if len(later) == 2 and all(x and x["status"] == "pass" for x in later) else
                                                 "fail" if len(later) == 2 and any(x and x["status"] == "fail" for x in later) else "unsupported")
    report["resultType"] = "post_hoc_exploratory"
    report["thresholdsMetInBothWindows"] = report["twoWindowPredeclaredConclusion"] == "pass"
    report["independentConfirmation"] = False
    report["interpretation"] = "07-09 comparison is post hoc exploratory; thresholds met do not establish independent confirmation; nonsignificance is not equivalence."
    report["reportSha256"] = _hash(report)
    return report


def _exclusion_states(rows: list[dict], source: dict, actuals: dict, evidence: dict,
                      scored_keys: set[tuple]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in rows:
        key = _row_key(r)
        if _date(r["target"]) not in actuals[r["product"]]:
            state, _ = base._status(source, r["product"], _date(r["target"]), evidence)
            counts[state] += 1
        elif key not in scored_keys:
            counts["incomplete_common_candidate_forecast"] += 1
    return dict(counts)


def _apply_family_holm(records: list[dict], bandwidth: int) -> None:
    """Apply a fixed-size Holm family; unavailable tests stay as p=1."""
    tests = [record["dm"][str(bandwidth)] for record in records]
    _holm(tests)


def _joint_verdicts(tests: list[dict]) -> list[dict]:
    indexed = {(r["product"], r["horizon"], r["comparator"], r["loss"]): r for r in tests}
    out = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            for comparator in COMPARATORS:
                ae = indexed[(product, horizon, comparator, "mae")]
                ape = indexed[(product, horizon, comparator, "mape")]
                item = {"product": product, "horizon": horizon, "comparator": comparator,
                        "nMae": ae["n"], "nMape": ape["n"], "verdictByBandwidth": {}}
                for bw in BANDWIDTHS:
                    a, p = ae["dm"][str(bw)], ape["dm"][str(bw)]
                    av, pv = a["conclusion"], p["conclusion"]
                    joint = (av if av == pv and av in ("improvement", "worsening") else
                             "inference_unavailable" if "inference_unavailable" in (av, pv) else "inconclusive")
                    item["verdictByBandwidth"][str(bw)] = {"mae": av, "mape": pv, "joint": joint}
                out.append(item)
    return out


def _period_gate(complete: bool, macro: dict, per_product: dict, coverage: dict,
                 joint: list[dict]) -> dict:
    candidate, comparator = "gap_residual_full", "ridge_alpha_100"
    cm, bm = macro[candidate], macro[comparator]
    gain = bm["mape"] - cm["mape"] if cm["mape"] is not None and bm["mape"] is not None else None
    mae_change = (100 * (cm["mae"] - bm["mae"]) / bm["mae"]
                  if cm["mae"] is not None and bm["mae"] not in (None, 0) else None)
    crop_mape_change = {p: per_product[p][candidate]["equalHorizonMape"] - per_product[p][comparator]["equalHorizonMape"]
                        if per_product[p][candidate]["equalHorizonMape"] is not None and per_product[p][comparator]["equalHorizonMape"] is not None else None
                        for p in PRODUCTS}
    crop_mae_change = {p: (100 * (per_product[p][candidate]["equalHorizonMae"] - per_product[p][comparator]["equalHorizonMae"])
                           / per_product[p][comparator]["equalHorizonMae"]
                           if per_product[p][candidate]["equalHorizonMae"] is not None
                           and per_product[p][comparator]["equalHorizonMae"] not in (None, 0) else None)
                       for p in PRODUCTS}
    joint_improved = {(r["product"], r["horizon"]) for r in joint if r["comparator"] == comparator
                      and r["verdictByBandwidth"]["9"]["joint"] == "improvement"
                      and r["verdictByBandwidth"]["28"]["joint"] == "improvement"}
    issued_ok = coverage[candidate]["issued"] >= coverage[comparator]["issued"]
    criteria = {"macroMapeGainAtLeast0_5pp": gain is not None and gain >= 0.5,
                "issuedCoverageNotDecreased": issued_ok,
                "eachProductMapeWorseningAtMost1pp": all(v is not None and v <= 1.0 for v in crop_mape_change.values()),
                "eachProductMaeWorseningAtMost5pct": all(v is not None and v <= 5.0 for v in crop_mae_change.values()),
                "macroMaeWorseningAtMost5pct": mae_change is not None and mae_change <= 5.0,
                "atLeastOneCellJointlyImprovedAtHac9AndHac28": bool(joint_improved)}
    return {"status": "unsupported" if not complete else "pass" if all(criteria.values()) else "fail",
            "criteria": criteria, "thresholds": {"macroMapeGainPpMinimum": 0.5,
                "productMapeWorseningPpMaximum": 1.0, "productMaeWorseningPctMaximum": 5.0,
                "macroMaeWorseningPctMaximum": 5.0,
                "requiredJointImprovementBandwidthDays": [9, 28], "requiredJointImprovementCells": 1},
            "macroMapeGainPp": gain, "macroMaeWorseningPct": mae_change,
            "productMapeChangePp": crop_mape_change,
            "productMaeChangePct": crop_mae_change,
            "jointImprovedCells": len(joint_improved),
            "failureReasons": [] if complete and all(criteria.values()) else
                (["incomplete_40_cell_coverage"] if not complete else [name for name, passed in criteria.items() if not passed])}


def _write_json_create_only(path: str | Path, doc: dict) -> None:
    """Write a complete JSON artifact only if its destination does not exist."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass
        raise


def _cli_sources(args) -> tuple[dict, dict, dict]:
    development = json.loads(Path(args.development_manifest).read_text(encoding="utf-8"))
    fresh = json.loads(Path(args.fresh_manifest).read_text(encoding="utf-8"))
    bundles = {"development": {"archivePath": args.archive},
               "fresh": {"archivePath": args.archive, "coverageReportPath": args.coverage_report}}
    return development, fresh, bundles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen offline national two-market residual experiment")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--development-manifest", default=str(PROJECT_ROOT / MANIFEST_PATHS["development"]))
    common.add_argument("--fresh-manifest", default=str(PROJECT_ROOT / MANIFEST_PATHS["fresh"]))
    common.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    common.add_argument("--coverage-report", default=str(DEFAULT_COVERAGE))
    common.add_argument("--market-snapshot", default=str(DEFAULT_MARKET_SNAPSHOT))
    build = sub.add_parser("build", parents=[common], help="write an answer-free candidate manifest")
    build.add_argument("--output", required=True)
    score = sub.add_parser("score", parents=[common], help="replay and score a saved candidate manifest")
    score.add_argument("--manifest", required=True)
    score.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    development, fresh, bundles = _cli_sources(args)
    if args.command == "build":
        result = build_manifest(development, fresh, bundles, args.market_snapshot)
    else:
        candidate = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        result = score_manifest(candidate, development, fresh, bundles, args.market_snapshot)
    _write_json_create_only(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
