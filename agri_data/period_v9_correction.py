"""V9 fixed walk-forward residual correction for the saved V5 G-B forecasts.

This module uses only pinned local snapshots and saved forecast artifacts. It
does not collect data or rewrite any earlier experiment. Live fitting is
deliberately a CLI action after the separate review gate has cleared.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import period_proxy_experiment as proxy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/experiments/period-v9-run-now-correction"
PROTOCOL = OUT / "protocol.md"
PROTOCOL_SHA = "a63c235410d8a7f6e0f5790efe6b158d8f7708a003f88791ccefa4cb22458d8b"
V5_OUT = ROOT / "outputs/experiments/period-pooled-direct-lgbm-v5-20261016"
V3_OUT = ROOT / "outputs/experiments/period-direct-mape-rolling-v3-20261015"
V8_OUT = ROOT / "outputs/experiments/period-supply-publication-scenario-v8-20261016"
V8_MATCHED = V8_OUT / "matched-facts.csv"
CROPS = ("cabbage", "radish", "onion", "garlic")
HORIZONS = (2, 3)
ALPHA = 10.0
LOWER, UPPER = math.log(.80), math.log(1.20)
SUPPLY = ("radish", "제주특별자치도농업기술원", "Jeju", "harvest_progress", "percent", "winter_radish")
FIT_CAP = 200
DM_LAGS = (3, 6)
BLOCK_LENGTHS = (6, 12)
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261016
PINNED_INPUT_SHA256 = {
    "protocol": PROTOCOL_SHA,
    "priceSnapshot": "6ca1366d1dda618c96f48e62fea578193696b24dc893e68320bdeaae8e43e659",
    "v5Forecasts": "1a9847d08121aee8d3d808961d116bf5abf0151f74957a1b49639e1d3c9ac0c7",
    "v5DevelopmentForecasts": "26a1a7192dd0b42f43794301e2cd9758c95b4a86138ebccc497763e7f4ca3115",
    "v5ScoreReport": "54dbad25c91fe5463716ea8c2cbb53cc55d196adc72f0495f0facaa50bfef8ab",
    "v3PForecasts": "697839423df8b9145290f5c2c3866f21c5bab05c2307829124614d98b744b712",
    "v8MatchedFacts": "3554c2095587f65acd8238a465ece73760d8c792cc56b5e6f1e1bc3dba687744",
    "v8FeatureMap": "cbbe4d7f4497ad2244345b898a7fd5dc05a441179a4d20f30078b4865201c1e7",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iso_day(value: Any) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value)[:10])


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    return pd.read_parquet(path).to_dict(orient="records")


def _targets() -> tuple[dict, dict]:
    raw = _json(proxy.SOURCE)
    series = proxy.series_from_raw(raw)
    targets = {crop: proxy.aggregate_targets(rows) for crop, rows in series.items()}
    return series, targets


def _v5_rows(split: str) -> list[dict[str, Any]]:
    name = "development-forecasts.parquet" if split == "develop" else "forecasts.parquet"
    rows = _read_parquet(V5_OUT / name)
    result = []
    for raw in rows:
        preds = raw.get("predictions") or {}
        gb = preds.get("G-B") or {}
        feat = raw.get("asOfFeatures") or {}
        meta = raw.get("featureMetadata") or {}
        result.append({
            "crop": str(raw["crop"]), "horizon": int(raw["horizon"]),
            "targetStart": str(raw["targetStart"]), "targetEnd": str(raw["targetEnd"]),
            "targetKey": str(raw.get("targetKey") or ""), "origin": str(raw["origin"]),
            "inputCutoff": str(raw["inputCutoff"]), "p_ref": raw.get("p_ref"),
            "featureReferencePrice": meta.get("referencePrice"),
            "month_sin": feat.get("month_sin"), "month_cos": feat.get("month_cos"),
            "base": gb.get("predicted"), "baseStatus": gb.get("status"),
            "policyUnavailable": bool(raw.get("policyUnavailable", False)),
            "rowStatus": str(raw.get("status", "")),
        })
    return result


def _baseline_v3() -> tuple[dict, dict]:
    """Read the saved V3 test comparator forecasts without re-fitting them."""
    rows = _read_parquet(V3_OUT / "forecasts_test.parquet")
    pmap, bmap = {}, {}
    for r in rows:
        key = (str(r["crop"]), int(r["horizon"]), str(r["targetStart"]), str(r["targetEnd"]), str(r["origin"]))
        fs = r.get("forecasts") or {}
        for name, out in (("P", pmap), ("B0", bmap)):
            val = (fs.get(name) or {}).get("predicted")
            if val is not None and math.isfinite(float(val)) and float(val) > 0:
                out[key] = float(val)
    return pmap, bmap


def _key(r: dict[str, Any]) -> tuple:
    return (r["crop"], int(r["horizon"]), r["targetStart"], r["targetEnd"], r["origin"])


def _input_paths() -> dict[str, Path]:
    return {"protocol": PROTOCOL, "priceSnapshot": proxy.SOURCE,
            "v5Forecasts": V5_OUT/"forecasts.parquet", "v5DevelopmentForecasts": V5_OUT/"development-forecasts.parquet",
            "v5ScoreReport": V5_OUT/"score-report.json", "v3PForecasts": V3_OUT/"forecasts_test.parquet",
            "v8MatchedFacts": V8_MATCHED, "v8FeatureMap": V8_OUT/"feature-map.json"}


def _verify_pinned_hashes() -> dict[str, str]:
    paths = _input_paths()
    hashes = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        expected = PINNED_INPUT_SHA256[name]
        if digest != expected:
            raise ValueError(f"pinned input hash mismatch for {name}: expected {expected}, got {digest}")
        hashes[name] = digest
    return hashes


def _label(row: dict[str, Any], targets: dict) -> float | None:
    t = (_iso_day(row["targetStart"]), _iso_day(row["targetEnd"]))
    obj = targets[row["crop"]].get(t, {})
    v = obj.get("actual")
    return float(v) if v is not None and math.isfinite(float(v)) and float(v) > 0 else None


def _reference(row: dict, series: dict) -> tuple[float | None, str]:
    """Restore p_ref from the frozen daily series and audit the V5 saved value."""
    cutoff = _iso_day(row["inputCutoff"])
    candidates = [d for d in series[row["crop"]] if d <= cutoff and series[row["crop"]][d][0] > 0]
    if not candidates:
        return None, "no_positive_price_before_cutoff"
    expected = float(series[row["crop"]][max(candidates)][0])
    def finite(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None
    saved, meta = finite(row.get("p_ref")), finite(row.get("featureReferencePrice"))
    if saved is not None and math.isclose(saved, expected, rel_tol=1e-10, abs_tol=1e-8):
        if meta is not None and not math.isclose(meta, expected, rel_tol=1e-10, abs_tol=1e-8):
            return expected, "restored_daily_snapshot_metadata_mismatch"
        return expected, "verified_saved_p_ref"
    if meta is not None and math.isclose(meta, expected, rel_tol=1e-10, abs_tol=1e-8):
        return expected, "restored_from_feature_metadata"
    if saved is None:
        return expected, "restored_from_daily_snapshot"
    return expected, "restored_daily_snapshot_saved_value_mismatch"


def _origin_day(row: dict) -> dt.date:
    return _iso_day(row["origin"])


def _publication_age_days(published: Any, origin_day: dt.date) -> float | None:
    """Return age at 10:00 KST only if availability by that exact instant is provable."""
    origin_at = dt.datetime.combine(origin_day, dt.time(10), tzinfo=dt.timezone(dt.timedelta(hours=9)))
    raw = str(published or "").strip()
    if not raw:
        return None
    value = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        published_at = dt.datetime.fromisoformat(value)
    except ValueError:
        # A date-only value has no same-day release-time evidence. Prior days
        # are conservatively treated as if published at the last second of day.
        try:
            published_day = dt.date.fromisoformat(raw[:10])
        except ValueError:
            return None
        if published_day >= origin_day:
            return None
        published_at = dt.datetime.combine(published_day, dt.time(23, 59, 59),
                                            tzinfo=dt.timezone(dt.timedelta(hours=9)))
    else:
        if published_at.tzinfo is None:
            # Naive timestamps are treated as date-only, conservatively.
            published_day = published_at.date()
            if published_day >= origin_day:
                return None
            published_at = dt.datetime.combine(published_day, dt.time(23, 59, 59),
                                                tzinfo=dt.timezone(dt.timedelta(hours=9)))
        else:
            published_at = published_at.astimezone(dt.timezone(dt.timedelta(hours=9)))
    age = (origin_at - published_at).total_seconds() / 86400.
    return age if age >= 0. else None


def _mature(row: dict, asof: dt.date) -> bool:
    return _iso_day(row["targetEnd"]) + dt.timedelta(days=2) <= asof


def _base_features(row: dict, p_ref: float) -> tuple[list[float], list[float]]:
    base = float(row["base"])
    if not (math.isfinite(base) and base > 0 and math.isfinite(p_ref) and p_ref > 0):
        raise ValueError("C features require positive BASE and p_ref")
    lr = math.log(base / p_ref)
    crop = row["crop"]
    h = int(row["horizon"])
    try:
        ms, mc = float(row["month_sin"]), float(row["month_cos"])
    except (TypeError, ValueError):
        month = _iso_day(row["targetStart"]).month
        ms, mc = math.sin(2 * math.pi * (month-1) / 12), math.cos(2 * math.pi * (month-1) / 12)
    cont = [max(lr, 0.0), min(lr, 0.0), math.log(p_ref), ms, mc]
    cats = [float(crop == c) for c in CROPS] + [float(h == hh) for hh in HORIZONS]
    return cont, cats


def _supply_map() -> dict[tuple, dict]:
    out = {}
    with V8_MATCHED.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["crop"], int(row["horizon"]), row["targetStart"], row["targetEnd"], row["origin"])
            if row.get("availability") == "matched" and row.get("factId") and row.get("tieConflict", "false").lower() != "true" \
                    and row.get("sourceOrg") == SUPPLY[1] and row.get("region") == SUPPLY[2] \
                    and row.get("quantityKind") == SUPPLY[3] and row.get("unit") == SUPPLY[4] \
                    and row.get("varietyOrCropType") == SUPPLY[5]:
                # Use only the already-reviewed match whose publication was eligible by the saved scenario.
                val = row.get("value")
                pub = row.get("publishEligibleAt")
                if val and pub and row.get("validity") != "invalid":
                    try:
                        numeric = float(val)
                    except ValueError:
                        continue
                    if math.isfinite(numeric):
                        out[key] = {"value": numeric, "factId": row.get("factId"),
                                    "updateGroupId": row.get("updateGroupId"), "ageDays": row.get("ageDays"),
                                    "publishEligibleAt": pub, "overlap": row.get("overlap")}
    return out


def _supply_features(row: dict, mapping: dict) -> tuple[list[float], dict | None]:
    f = mapping.get(_key(row))
    if f is None:
        return [0., 0., 0., 0., 1.], None
    # Never clamp stale reports into an apparently fresh feature. Publication
    # must be provably available by the issue's 10:00 KST boundary.
    age_num = _publication_age_days(f["publishEligibleAt"], _origin_day(row))
    if age_num is None or age_num > 30:
        return [0., 0., 0., 0., 1.], None
    overlap_state = str(f.get("overlap", "")).strip().lower()
    overlap_known = overlap_state not in {"", "unknown", "none", "not_assessed", "null"}
    return [1., age_num, float(overlap_known), float(f["value"]), 0.], f


def _assemble(row: dict, p_ref: float, supply: dict, with_supply: bool) -> tuple[np.ndarray, dict | None]:
    cont, cats = _base_features(row, p_ref)
    if with_supply:
        extra, fact = _supply_features(row, supply)
        cont += [extra[1], extra[3]]
        cats += [extra[0], extra[2], extra[4]]
        return np.asarray(cont + cats, dtype=float), fact
    return np.asarray(cont + cats, dtype=float), None


def _fit_model(train: list[dict], labels: dict, refs: dict, supply_map: dict,
               *, with_supply: bool) -> dict | None:
    """Fit one estimator for a unique training-key set; callers share it by origin."""
    rows = [r for r in train if r["_rowkey"] in labels and r["_rowkey"] in refs]
    if not rows:
        return None
    usable_supply = {r["_rowkey"]: _supply_features(r, supply_map)[1] for r in rows}
    groups = {f.get("updateGroupId") for f in usable_supply.values()
              if f is not None and f.get("updateGroupId")}
    observed = [f["value"] for f in usable_supply.values() if f is not None]
    if with_supply and (len(groups) < 2 or len(set(observed)) < 2):
        return None
    built = [_assemble(r, refs[r["_rowkey"]], supply_map, with_supply)[0] for r in rows]
    X = np.vstack(built)
    y = np.asarray([labels[r["_rowkey"]] for r in rows], dtype=float)
    ncont = 7 if with_supply else 5
    value_mean = float(np.mean(observed)) if observed else 0.0
    if with_supply:
        for i, r in enumerate(rows):
            if usable_supply[r["_rowkey"]] is None:
                X[i, 6] = value_mean
    means = X[:, :ncont].mean(axis=0)
    scales = X[:, :ncont].std(axis=0)
    scales[scales < 1e-12] = 1.
    Z = X.copy()
    Z[:, :ncont] = (X[:, :ncont]-means)/scales
    model = Ridge(alpha=ALPHA, fit_intercept=True, solver="svd")
    model.fit(Z, y)
    audit = {"n": int(len(y)), "trainRowKeys": [r["_rowkey"] for r in rows],
             "lastMatureEnd": max((r["targetEnd"] for r in rows), default=None),
             "trainMaxMaturityAt": (max((_iso_day(r["targetEnd"])+dt.timedelta(days=2) for r in rows)).isoformat()+"T10:00:00+09:00" if rows else None),
             "scalerMean": means.tolist(), "scalerScale": scales.tolist(),
             "intercept": float(model.intercept_), "coefficients": model.coef_.tolist(),
             "supplyIdentifiable": bool(with_supply), "supplyObservedMean": value_mean,
             "supplyDistinctTrainingGroups": len(groups)}
    return {"model": model, "means": means, "scales": scales, "ncont": ncont,
            "supplyMean": value_mean, "audit": audit, "rows": rows}


def _predict_model(bundle: dict | None, current: dict, refs: dict, supply_map: dict,
                   *, with_supply: bool, numeric_neutral: bool = False) -> dict:
    if bundle is None:
        return {"prediction": None, "reason": "supply_or_training_not_identifiable" if with_supply else "no_mature_training_labels", "fit": None}
    x, fact = _assemble(current, refs[current["_rowkey"]], supply_map, with_supply)
    if with_supply and fact is None:
        return {"prediction": None, "reason": "no_current_numeric_supply_value", "fit": None}
    if with_supply and fact is not None:
        x[6] = bundle["supplyMean"] if numeric_neutral else float(fact["value"])
    ncont = bundle["ncont"]
    x[:ncont] = (x[:ncont]-bundle["means"])/bundle["scales"]
    raw = float(bundle["model"].predict(x.reshape(1, -1))[0])
    clipped = min(UPPER, max(LOWER, raw))
    audit = {**bundle["audit"], "supplyCurrentFact": fact}
    direct = 0.0
    if with_supply and fact is not None:
        direct = float(bundle["model"].coef_[6] *
                       ((float(fact["value"])-bundle["supplyMean"])/bundle["scales"][6]))
    return {"prediction": float(current["BASE"])*math.exp(clipped), "rawAdjustment": raw,
            "usedAdjustment": clipped, "clipped": clipped != raw, "reason": "ridge",
            "fit": audit, "directSupplyLogContribution": direct,
            "numericNeutral": numeric_neutral}


def _fit_cached(cache: dict, train: list[dict], labels: dict, refs: dict,
                supply_map: dict, *, with_supply: bool,
                other_cache: dict | None = None) -> dict | None:
    """Reuse only an exact ordered training-key set with the same feature contract."""
    key = tuple(sorted(labels))
    if key not in cache:
        if sum(bundle is not None for bucket in (cache, other_cache or {}) for bundle in bucket.values()) >= FIT_CAP:
            raise ValueError(f"Ridge fit cap {FIT_CAP} would be exceeded before fitting")
        cache[key] = _fit_model(train, labels, refs, supply_map, with_supply=with_supply)
    return cache[key]


def _prepare(split: str, targets: dict, series: dict, pmap: dict | None = None, bmap: dict | None = None) -> list[dict]:
    rows = _v5_rows(split)
    seen = set()
    result = []
    for r in rows:
        k = _key(r)
        if k in seen:
            raise ValueError(f"duplicate V5 forecast key: {k}")
        seen.add(k)
        r["_rowkey"] = "|".join(map(str, k))
        r["actual"] = _label(r, targets)
        ref, status = _reference(r, series)
        r["p_ref_restored"] = ref
        r["p_ref_status"] = status
        r["BASE"] = float(r["base"]) if r["base"] is not None and math.isfinite(float(r["base"])) and float(r["base"]) > 0 else None
        r["P"] = (pmap or {}).get(k)
        r["B0"] = (bmap or {}).get(k)
        result.append(r)
    return result


def _evaluable(r: dict) -> bool:
    return r["BASE"] is not None and r["actual"] is not None


def preflight() -> dict:
    """Read-only input/key/availability checks; this function never fits a model."""
    input_hashes = _verify_pinned_hashes()
    series, targets = _targets()
    pmap, bmap = _baseline_v3()
    dev = _prepare("develop", targets, series)
    test = _prepare("test", targets, series, pmap, bmap)
    if len(dev) != 576 or len(test) != 792:
        raise ValueError(f"planned-row mismatch: development={len(dev)} evaluation={len(test)}")
    valid = [r for r in test if _evaluable(r)]
    if len(valid) != 780:
        raise ValueError(f"expected 780 BASE/actual keys, found {len(valid)}")
    for method in ("P", "B0"):
        missing = [_key(r) for r in valid if r.get(method) is None]
        if missing:
            raise ValueError(f"{method} lacks {len(missing)} same-key comparator forecasts")
    reproduction = {name: _score(valid, name) for name in ("BASE", "P", "B0")}
    expected_metrics = {"BASE": (212.88623129313441, 13.478028329407538),
                        "P": (225.4959168858682, 14.327882691082277),
                        "B0": (221.2327823696221, 15.732254045793303)}
    for name, (mae, mape) in expected_metrics.items():
        observed = reproduction[name]
        if (not math.isclose(observed["mae8"], mae, rel_tol=0, abs_tol=1e-7)
                or not math.isclose(observed["mape8_pct"], mape, rel_tol=0, abs_tol=1e-7)):
            raise ValueError(f"same-key {name} score does not reproduce frozen baseline")
    by_cell = {f"{c}:{h}": sum(r["crop"] == c and int(r["horizon"]) == h for r in valid)
               for c in CROPS for h in HORIZONS}
    expected = {f"{c}:{h}": 98 if h == 2 else 97 for c in CROPS for h in HORIZONS}
    if by_cell != expected:
        raise ValueError(f"same-key per-cell support differs from V9 contract: {by_cell}")
    matched = _supply_map()
    groups = {f.get("updateGroupId") for f in matched.values() if f.get("updateGroupId")}
    return {"status": "preflight_passed_no_fit", "protocolSha256": PROTOCOL_SHA,
            "inputHashes": input_hashes,
            "baselineReproduction": reproduction,
            "plannedRows": {"development": len(dev), "evaluation": len(test)},
            "targetYears": {str(y): sum(r["actual"] is not None and _iso_day(r["targetStart"]).year == y for r in test)
                            for y in (2024, 2025, 2026)},
            "validSameKeys": len(valid), "validRowsByCell": by_cell,
            "pRefStatus": {s: sum(r["p_ref_status"] == s for r in test) for s in sorted({r["p_ref_status"] for r in test})},
            "pRefVerifiedRows": sum(r["p_ref_status"] == "verified_saved_p_ref" for r in test),
            "pRefRestoredRows": sum(r["p_ref_status"].startswith("restored_") for r in test),
            "sameKeyComparators": {"P": sum(r["P"] is not None for r in valid), "B0": sum(r["B0"] is not None for r in valid)},
            "selectedNumericSupplyRows": len(matched), "selectedNumericSupplyGroups": len(groups),
            "forecastOriginCounts": {"development2023": len({r["origin"] for r in dev if _iso_day(r["targetStart"]).year == 2023}),
                                     "evaluation": len({r["origin"] for r in test})},
            "modelFits": 0}


def _score(rows: list[dict], pred_key: str) -> dict:
    cells = {}
    for c in CROPS:
        for h in HORIZONS:
            cell = [r for r in rows if r["crop"] == c and int(r["horizon"]) == h
                    and r.get("actual") is not None and r.get(pred_key) is not None]
            if not cell:
                cells[f"{c}:{h}"] = {"n": 0, "mae": None, "mape_pct": None}
                continue
            a = np.asarray([r["actual"] for r in cell], dtype=float)
            p = np.asarray([r[pred_key] for r in cell], dtype=float)
            cells[f"{c}:{h}"] = {"n": int(len(cell)), "mae": float(np.mean(np.abs(p-a))),
                                   "mape_pct": float(100*np.mean(np.abs((p-a)/a)))}
    return {"cells": cells,
            "mae8": float(np.mean([cells[f"{c}:{h}"]["mae"] for c in CROPS for h in HORIZONS if cells[f"{c}:{h}"]["mae"] is not None])) if any(v["mae"] is not None for v in cells.values()) else None,
            "mape8_pct": float(np.mean([cells[f"{c}:{h}"]["mape_pct"] for c in CROPS for h in HORIZONS if cells[f"{c}:{h}"]["mape_pct"] is not None])) if any(v["mape_pct"] is not None for v in cells.values()) else None}


def _log_ratio_sd(rows: list[dict], pred_key: str) -> dict:
    pairs = [(r.get("actual"), r.get(pred_key), r.get("p_ref_restored")) for r in rows]
    pairs = [(float(a), float(p), float(ref)) for a, p, ref in pairs
             if a is not None and p is not None and ref is not None and float(a) > 0 and float(p) > 0 and float(ref) > 0]
    if not pairs:
        return {"n": 0, "actualLogRatioSd": None, "forecastLogRatioSd": None}
    actual = np.asarray([math.log(a/ref) for a, _, ref in pairs])
    predicted = np.asarray([math.log(p/ref) for _, p, ref in pairs])
    return {"n": len(pairs), "actualLogRatioSd": float(np.std(actual)),
            "forecastLogRatioSd": float(np.std(predicted))}


def _postfit_diagnostics(rows: list[dict]) -> dict:
    usable = [r for r in rows if r.get("actual") is not None and r.get("p_ref_restored") is not None
              and float(r["actual"]) > 0 and float(r["p_ref_restored"]) > 0]
    move = {r["_rowkey"]: float(r["actual"])/float(r["p_ref_restored"])-1. for r in usable}
    sudden = [r for r in usable if abs(move[r["_rowkey"]]) >= .20]
    rising = [r for r in usable if move[r["_rowkey"]] >= .20]
    falling = [r for r in usable if move[r["_rowkey"]] <= -.20]
    methods = ("BASE", "P", "B0", "C", "E")
    def subset_report(group):
        return {"n": len(group), "scores": {name: _score(group, name) for name in methods}}
    garlic2026 = [r for r in rows if r["crop"] == "garlic" and _iso_day(r["targetStart"]).year == 2026]
    changed_c = [r for r in rows if r.get("C") is not None and r.get("BASE") is not None
                 and not math.isclose(float(r["C"]), float(r["BASE"]), rel_tol=0, abs_tol=1e-12)]
    changed_e = [r for r in rows if r.get("E") is not None and r.get("C") is not None
                 and not math.isclose(float(r["E"]), float(r["C"]), rel_tol=0, abs_tol=1e-12)]
    e_supported = [r for r in rows if r.get("E_reason") == "ridge"]
    return {"rapidActualMoveThreshold": .20,
            "rapidActualMove": subset_report(sudden),
            "rapidActualMoveUp": subset_report(rising),
            "rapidActualMoveDown": subset_report(falling),
            "2026Garlic": {"plannedRows": len(garlic2026), "validLabels": sum(r.get("actual") is not None for r in garlic2026),
                           "scores": {name: _score(garlic2026, name) for name in methods}},
            "logRatioStandardDeviations": {"all": {name: _log_ratio_sd(rows, name) for name in ("BASE", "C", "E")},
                                           "byCell": {f"{c}:h{h}": {name: _log_ratio_sd([r for r in rows if r["crop"] == c and int(r["horizon"]) == h], name)
                                                                     for name in ("BASE", "C", "E")}
                                                       for c in CROPS for h in HORIZONS},
            },
            "correctionCoverage": {"CChangedFromBaseRows": len(changed_c), "CRidgeRows": sum(r.get("C_reason") == "ridge" for r in rows),
                                   "CExactBaseFallbackRows": sum(r.get("C_reason") != "ridge" for r in rows),
                                   "EChangedFromCRows": len(changed_e), "ERidgeRows": len(e_supported),
                                   "EExactCFallbackRows": len(rows)-len(e_supported),
                                   "EClippedRows": sum(bool(r.get("E_clipped")) for r in e_supported),
                                   "ENumericNeutralComparedRows": sum(r.get("E_numericNeutralPrediction") is not None for r in e_supported),
                                   "meanDirectNumericLogContributionPreClip": float(np.mean([r["E_numeric_direct_raw_log_contribution"] for r in e_supported])) if e_supported else None,
                                   "meanDirectNumericUsedLogDelta": float(np.mean([r["E_numeric_direct_log_contribution"] for r in e_supported])) if e_supported else None,
                                   "meanNuisanceRefitLogDelta": float(np.mean([r["E_nuisance_refit_log_delta"] for r in e_supported])) if e_supported else None}}


def _dekad_index(day: str) -> int:
    d = _iso_day(day)
    slot = 1 if d.day <= 10 else 2 if d.day <= 20 else 3
    return d.year * 36 + (d.month - 1) * 3 + slot - 1


def _paired_errors(rows: list[dict], candidate: str, comparator: str, metric: str) -> list[tuple[int, float]]:
    out = []
    for r in rows:
        actual, p1, p0 = r.get("actual"), r.get(candidate), r.get(comparator)
        if actual is None or p1 is None or p0 is None or float(actual) <= 0:
            continue
        e1, e0 = abs(float(actual)-float(p1)), abs(float(actual)-float(p0))
        diff = e1-e0 if metric == "mae" else 100. * (e1-e0)/float(actual)
        out.append((_dekad_index(r["targetStart"]), float(diff)))
    return out


def _dm_hac(differences: list[tuple[int, float]], lag: int) -> dict:
    """One-sided-origin target-dekad HAC variance, with absent dekads retained as gaps."""
    grouped: dict[int, list[float]] = {}
    for index, value in differences:
        grouped.setdefault(int(index), []).append(float(value))
    values = {i: float(np.mean(v)) for i, v in grouped.items()}
    n = len(values)
    if n < lag + 2:
        return {"nPairedRows": len(differences), "nDistinctTargetDekads": n, "meanDifference": float(np.mean(list(values.values()))) if values else None,
                "statistic": None, "pRaw": None, "status": "unsupported", "reason": "insufficient distinct target dekads for HAC lag"}
    mean = float(np.mean(list(values.values())))
    centered = {i: v-mean for i, v in values.items()}
    lrv = sum(v*v for v in centered.values())/n
    for h in range(1, lag+1):
        cross = sum(centered[i]*centered[i+h] for i in centered if i+h in centered)/n
        lrv += 2.*(1.-h/(lag+1.))*cross
    if not math.isfinite(lrv) or lrv <= 1e-15:
        return {"nPairedRows": len(differences), "nDistinctTargetDekads": n, "meanDifference": mean, "statistic": None,
                "pRaw": None, "status": "unsupported", "reason": "HAC long-run variance is zero or non-finite"}
    statistic = mean/math.sqrt(lrv/n)
    return {"nPairedRows": len(differences), "nDistinctTargetDekads": n, "meanDifference": mean, "statistic": statistic,
            "pRaw": math.erfc(abs(statistic)/math.sqrt(2.)), "status": "computed", "reason": None}


def _dm_results(rows: list[dict], candidate: str, comparator: str) -> list[dict]:
    output = []
    for lag in DM_LAGS:
        family = []
        for crop in CROPS:
            for h in HORIZONS:
                cell = [r for r in rows if r["crop"] == crop and int(r["horizon"]) == h]
                for metric in ("mae", "ape"):
                    test = _dm_hac(_paired_errors(cell, candidate, comparator, metric), lag)
                    family.append({"hypothesisId": f"{candidate}-{comparator}|{crop}|h{h}|{metric}|hac{lag}",
                                   "crop": crop, "horizon": h, "metric": metric,
                                   "candidate": candidate, "comparator": comparator,
                                   "hacLagDekads": lag, **test})
        if len(family) != 16:
            raise AssertionError("V9 DM family must contain 16 hypotheses")
        ordered = sorted(family, key=lambda x: (x["pRaw"] if x["pRaw"] is not None else 1., x["hypothesisId"]))
        running = 0.
        for rank, item in enumerate(ordered):
            raw = item["pRaw"] if item["pRaw"] is not None else 1.
            running = max(running, min(1., (16-rank)*raw))
            item["pHolm"] = running
            item["holmFamilySize"] = 16
            item["holmSignificant5pct"] = item["pRaw"] is not None and running < .05
            item["inferenceDirection"] = ("improvement" if item["meanDifference"] is not None and item["meanDifference"] < 0
                                           else "worsening" if item["meanDifference"] is not None and item["meanDifference"] > 0
                                           else "inconclusive") if item["holmSignificant5pct"] else "inconclusive"
        output.extend(family)
    return output


def _resample_indices(calendar: list[int], length: int, rng: np.random.Generator) -> list[int]:
    if len(calendar) < length:
        return []
    starts = np.arange(calendar[0], calendar[-1]-length+2)
    picked: list[int] = []
    while len(picked) < len(calendar):
        start = int(rng.choice(starts))
        picked.extend(range(start, start+length))
    return picked[:len(calendar)]


def _bootstrap_mean(rows: list[dict], candidate: str, comparator: str, metric: str,
                    sampled_calendar: list[int]) -> float | None:
    by_index: dict[int, list[dict]] = {}
    for r in rows:
        by_index.setdefault(_dekad_index(r["targetStart"]), []).append(r)
    diffs = []
    for index in sampled_calendar:
        for r in by_index.get(index, []):
            a, p1, p0 = r.get("actual"), r.get(candidate), r.get(comparator)
            if a is None or p1 is None or p0 is None or float(a) <= 0:
                continue
            e1, e0 = abs(float(a)-float(p1)), abs(float(a)-float(p0))
            diffs.append(e1-e0 if metric == "mae" else 100.*(e1-e0)/float(a))
    return float(np.mean(diffs)) if diffs else None


def _block_bootstrap(rows: list[dict], candidate: str, comparator: str) -> list[dict]:
    all_idx = sorted({_dekad_index(r["targetStart"]) for r in rows})
    if not all_idx:
        return []
    calendar = list(range(all_idx[0], all_idx[-1]+1))
    output = []
    cells = [(crop, h) for crop in CROPS for h in HORIZONS]
    arrays_by_metric: dict[str, dict[tuple[str, int], np.ndarray]] = {"mae": {}, "ape": {}}
    for metric in ("mae", "ape"):
        for crop, h in cells:
            cell_rows = [r for r in rows if r["crop"] == crop and int(r["horizon"]) == h]
            grouped: dict[int, list[float]] = {}
            for idx, diff in _paired_errors(cell_rows, candidate, comparator, metric):
                grouped.setdefault(idx, []).append(diff)
            values = np.full(len(calendar), np.nan, dtype=float)
            for idx, diffs in grouped.items():
                if calendar[0] <= idx <= calendar[-1]:
                    values[idx-calendar[0]] = float(np.mean(diffs))
            arrays_by_metric[metric][(crop, h)] = values
    for length in BLOCK_LENGTHS:
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        starts = np.arange(0, len(calendar)-length+1)
        draws = np.empty((BOOTSTRAP_REPLICATES, len(calendar)), dtype=np.int32)
        for i in range(BOOTSTRAP_REPLICATES):
            positions = []
            while len(positions) < len(calendar):
                start = int(rng.choice(starts))
                positions.extend(range(start, start+length))
            draws[i, :] = positions[:len(calendar)]
        for metric in ("mae", "ape"):
            samples_by_cell = {}
            for crop, h in cells:
                arr = arrays_by_metric[metric][(crop, h)]
                sample_matrix = arr[draws]
                counts = np.isfinite(sample_matrix).sum(axis=1)
                samples_full = np.divide(np.nansum(sample_matrix, axis=1), counts,
                                         out=np.full(len(counts), np.nan), where=counts > 0)
                samples = samples_full[np.isfinite(samples_full)]
                samples_by_cell[(crop, h)] = samples_full
                observed = float(np.nanmean(arr)) if np.isfinite(arr).any() else None
                output.append(_bootstrap_record(length, candidate, comparator, metric, f"{crop}:h{h}", samples, observed))
            cell_sample_matrix = np.vstack([samples_by_cell[cell] for cell in cells])
            # All cells share the exact same draw calendar and replicate count.
            counts = np.isfinite(cell_sample_matrix).sum(axis=0)
            all_samples_full = np.divide(np.nansum(cell_sample_matrix, axis=0), counts,
                                         out=np.full(len(counts), np.nan), where=counts > 0)
            all_samples = all_samples_full[np.isfinite(all_samples_full)]
            cell_observed = [float(np.nanmean(arrays_by_metric[metric][cell]))
                             for cell in cells if np.isfinite(arrays_by_metric[metric][cell]).any()]
            observed = float(np.mean(cell_observed)) if len(cell_observed) == 8 else None
            output.append(_bootstrap_record(length, candidate, comparator, metric,
                                            "all_8cell_equal_weight", all_samples, observed))
    return output


def _bootstrap_record(length: int, candidate: str, comparator: str, metric: str,
                      aggregate: str, samples: list[float], observed: float | None) -> dict:
    samples = np.asarray(samples, dtype=float)
    lower = float(np.quantile(samples, .025)) if len(samples) else None
    upper = float(np.quantile(samples, .975)) if len(samples) else None
    direction = "improvement" if upper is not None and upper < 0 else "worsening" if lower is not None and lower > 0 else "inconclusive"
    return {"blockLengthDekads": length, "candidate": candidate, "comparator": comparator,
            "metric": metric, "aggregate": aggregate, "replicatesRequested": BOOTSTRAP_REPLICATES,
            "replicatesUsable": len(samples), "seed": BOOTSTRAP_SEED,
            "ci95Lower": lower, "ci95Upper": upper, "observedDifference": observed,
            "meanDifference": float(np.mean(samples)) if len(samples) else None,
            "direction": direction,
            "status": "computed" if len(samples) else "unsupported_no_paired_scores"}


def run(out: Path = OUT) -> dict:
    """Perform the fixed V9 development and walk-forward forecasts."""
    input_paths = {**_input_paths(), "implementation": Path(__file__).resolve()}
    input_hashes_before = {name: sha256(path) for name, path in input_paths.items()}
    preflight_record = preflight()
    if any(input_hashes_before[name] != preflight_record["inputHashes"][name] for name in preflight_record["inputHashes"]):
        raise ValueError("pinned input changed between run hash and preflight")
    out.mkdir(parents=True, exist_ok=True)
    preflight_record["protocolBoundsVerified"] = True
    preflight_record["fitCap"] = FIT_CAP
    (out/"preflight.json").write_text(json.dumps(preflight_record, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    series, targets = _targets()
    pmap, bmap = _baseline_v3()
    dev = _prepare("develop", targets, series)
    test = _prepare("test", targets, series, pmap, bmap)
    if len(dev) != 576 or len(test) != 792:
        raise ValueError(f"planned-row mismatch: development={len(dev)} evaluation={len(test)}")
    if sum(r["BASE"] is not None and r["actual"] is not None for r in test) != 780:
        raise ValueError("V5 same-key evaluation support differs from frozen 780-key contract")
    for r in test:
        if r["BASE"] is not None and r["actual"] is not None and (r["P"] is None or r["B0"] is None):
            raise ValueError(f"P/B0 comparator missing for paired key {_key(r)}")
    supply_map = _supply_map()
    forecasts = []
    dev_out = []
    c_fit_cache: dict[tuple, dict | None] = {}
    e_fit_cache: dict[tuple, dict | None] = {}
    dev_groups: dict[str, list[dict]] = {}
    for r in dev:
        if _iso_day(r["targetStart"]).year == 2023:
            dev_groups.setdefault(r["origin"], []).append(r)
    # Each 2023 development origin fits once with only matured 2022 labels.
    for origin_text in sorted(dev_groups):
        group = dev_groups[origin_text]
        origin = _iso_day(origin_text)
        train = [r for r in dev if _iso_day(r["targetStart"]).year == 2022 and _origin_day(r) < origin
                 and _mature(r, origin) and r["BASE"] is not None and r["actual"] is not None
                 and r["p_ref_restored"] is not None]
        labels = {r["_rowkey"]: math.log(r["actual"]/r["BASE"]) for r in train}
        refs = {r["_rowkey"]: r["p_ref_restored"] for r in train}
        c_bundle = _fit_cached(c_fit_cache, train, labels, refs, supply_map, with_supply=False, other_cache=e_fit_cache)
        e_bundle = _fit_cached(e_fit_cache, train, labels, refs, supply_map, with_supply=True, other_cache=c_fit_cache) if any(_supply_features(r, supply_map)[1] is not None for r in group) else None
        for current in group:
            rowrefs = {**refs, current["_rowkey"]: current["p_ref_restored"]}
            c = _predict_model(c_bundle, current, rowrefs, supply_map, with_supply=False) if current["BASE"] is not None and current["p_ref_restored"] is not None else {"prediction": None, "reason": "baseline_or_reference_unavailable", "fit": None}
            current["C"] = c["prediction"] if c["prediction"] is not None else current["BASE"]
            current["C_reason"] = c["reason"]
            current["C_fit"] = c["fit"]
            current["C_raw_adjustment"] = c.get("rawAdjustment")
            current["C_used_adjustment"] = c.get("usedAdjustment")
            e = _predict_model(e_bundle, current, rowrefs, supply_map, with_supply=True) if current["BASE"] is not None and current["p_ref_restored"] is not None else {"prediction": None, "reason": "baseline_or_reference_unavailable", "fit": None}
            current["E"] = e["prediction"] if e["prediction"] is not None else current["C"]
            current["E_reason"] = e["reason"] if e["prediction"] is not None else "exact_C_fallback:" + e["reason"]
            current["E_fit"] = e.get("fit")
            current["E_raw_adjustment"] = e.get("rawAdjustment")
            current["E_used_adjustment"] = e.get("usedAdjustment")
            current["E_clipped"] = e.get("clipped", False)
            if e.get("prediction") is not None:
                neutral = _predict_model(e_bundle, current, rowrefs, supply_map, with_supply=True, numeric_neutral=True)
                current["E_numericNeutralPrediction"] = neutral["prediction"]
                current["E_numericNeutralAdjustment"] = neutral["usedAdjustment"]
                current["E_numeric_direct_raw_log_contribution"] = e.get("directSupplyLogContribution", 0.)
                current["E_numeric_direct_log_contribution"] = e["usedAdjustment"]-neutral["usedAdjustment"]
                current["E_nuisance_refit_log_delta"] = neutral["usedAdjustment"]-c["usedAdjustment"] if c.get("usedAdjustment") is not None else None
            else:
                current["E_numericNeutralPrediction"] = None
                current["E_numericNeutralAdjustment"] = None
                current["E_numeric_direct_raw_log_contribution"] = 0.
                current["E_numeric_direct_log_contribution"] = 0.
                current["E_nuisance_refit_log_delta"] = 0.
            dev_out.append(current.copy())
    dev_score = {k: _score(dev_out, k) for k in ("BASE", "C")}
    dev_fit_count = sum(bundle is not None for bucket in (c_fit_cache, e_fit_cache) for bundle in bucket.values())
    (out/"development.json").write_text(json.dumps({"protocolSha256": PROTOCOL_SHA, "year": 2023,
      "scores": dev_score, "rows": len(dev_out), "ridgeFitCountBeforeEvaluation": dev_fit_count,
      "originFitAudits": {origin: {
          "C": next((r.get("C_fit") for r in dev_out if r["origin"] == origin and r.get("C_fit")), None),
          "E": next((r.get("E_fit") for r in dev_out if r["origin"] == origin and r.get("E_fit")), None)}
                           for origin in sorted({r["origin"] for r in dev_out})},
      "forecasts": [{k: r.get(k) for k in ("crop", "horizon", "origin", "targetStart", "targetEnd", "actual", "BASE", "C", "C_reason")}
                    for r in dev_out]},
      ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    # Seed the online label pool with V5 development observations, then append
    # each previous evaluation label only after the targetEnd+2 maturity rule.
    prior = [r for r in dev if r["BASE"] is not None and r["actual"] is not None and r["p_ref_restored"] is not None]
    origin_groups: dict[str, list[dict]] = {}
    for r in test:
        origin_groups.setdefault(r["origin"], []).append(r)
    for origin_text in sorted(origin_groups):
        group = origin_groups[origin_text]
        origin = _iso_day(origin_text)
        eligible = [r for r in prior if _origin_day(r) < origin and _mature(r, origin)
                    and r["BASE"] is not None and r["actual"] is not None and r["p_ref_restored"] is not None]
        # Fit once per unique origin; every crop/horizon row shares the same model.
        labels = {r["_rowkey"]: math.log(r["actual"]/r["BASE"]) for r in eligible}
        refs = {r["_rowkey"]: r["p_ref_restored"] for r in eligible}
        c_bundle = _fit_cached(c_fit_cache, eligible, labels, refs, supply_map, with_supply=False, other_cache=e_fit_cache)
        e_bundle = _fit_cached(e_fit_cache, eligible, labels, refs, supply_map, with_supply=True, other_cache=c_fit_cache) if any(_supply_features(r, supply_map)[1] is not None for r in group) else None
        for r in group:
            # Compute a row-specific adjustment from the shared fitted estimator.
            if c_bundle is None or r["BASE"] is None or r["p_ref_restored"] is None:
                c = {"prediction": r["BASE"], "reason": "baseline_or_reference_unavailable", "fit": None}
            else:
                c = _predict_model(c_bundle, r, {**refs, r["_rowkey"]: r["p_ref_restored"]}, supply_map, with_supply=False)
            r["C"] = c["prediction"] if c["prediction"] is not None else r["BASE"]
            r["C_reason"] = c["reason"]
            r["C_fit"] = c.get("fit")
            r["C_raw_adjustment"] = c.get("rawAdjustment")
            r["C_used_adjustment"] = c.get("usedAdjustment")
            # E remains exactly C unless its one fixed scalar is identifiable.
            e = _predict_model(e_bundle, r, {**refs, r["_rowkey"]: r["p_ref_restored"]}, supply_map, with_supply=True) if e_bundle is not None and r["BASE"] is not None and r["p_ref_restored"] is not None else {"prediction": None, "reason": "numeric_supply_not_identifiable_or_unavailable", "fit": None}
            if e["prediction"] is None:
                r["E"] = r["C"]
                r["E_reason"] = e["reason"]
                r["E_numeric_direct_log_contribution"] = 0.
                r["E_numeric_direct_raw_log_contribution"] = 0.
                r["E_numericNeutralPrediction"] = None
                r["E_numericNeutralAdjustment"] = None
                r["E_nuisance_refit_log_delta"] = 0.
                r["E_raw_adjustment"] = None
                r["E_used_adjustment"] = None
                r["E_clipped"] = False
            else:
                r["E"] = e["prediction"]
                r["E_reason"] = e["reason"]
                r["E_fit"] = e["fit"]
                neutral = _predict_model(e_bundle, r, {**refs, r["_rowkey"]: r["p_ref_restored"]}, supply_map, with_supply=True, numeric_neutral=True)
                r["E_numericNeutralPrediction"] = neutral["prediction"]
                r["E_numericNeutralAdjustment"] = neutral["usedAdjustment"]
                r["E_numeric_direct_raw_log_contribution"] = e.get("directSupplyLogContribution", 0.)
                r["E_numeric_direct_log_contribution"] = e["usedAdjustment"]-neutral["usedAdjustment"]
                r["E_nuisance_refit_log_delta"] = neutral["usedAdjustment"]-c["usedAdjustment"] if c.get("usedAdjustment") is not None else None
                r["E_raw_adjustment"] = e.get("rawAdjustment")
                r["E_used_adjustment"] = e.get("usedAdjustment")
                r["E_clipped"] = e.get("clipped", False)
            forecasts.append(r.copy())
        # after forecasts at this origin are generated, mature current labels join future fits
        prior.extend(r for r in group if r["BASE"] is not None and r["actual"] is not None and r["p_ref_restored"] is not None)
    fit_count = sum(bundle is not None for bundle in c_fit_cache.values()) + sum(bundle is not None for bundle in e_fit_cache.values())
    if fit_count > FIT_CAP:
        raise ValueError(f"Ridge fits {fit_count} exceed frozen cap {FIT_CAP}")
    if len({r["_rowkey"] for r in forecasts}) != len(forecasts) or len(forecasts) != 792:
        raise ValueError("evaluation forecasts do not preserve the full planned key set")
    scores = {name: _score(forecasts, name) for name in ("BASE", "P", "B0", "C", "E")}
    supported_e = [r for r in forecasts if r.get("E_reason") == "ridge" and r.get("actual") is not None]
    scores["E_numericSupport"] = {"n": len(supported_e), "C": _score(supported_e, "C"), "E": _score(supported_e, "E")}
    scores["byTargetYear"] = {str(year): {name: _score([r for r in forecasts if _iso_day(r["targetStart"]).year == year], name)
                                                for name in ("BASE", "P", "B0", "C", "E")}
                              for year in (2024, 2025, 2026)}
    paired_coverage = {f"{c}:{h}": sum(r["crop"] == c and int(r["horizon"]) == h and r["actual"] is not None and r["BASE"] is not None
                                        and r.get("C") is not None and r.get("E") is not None for r in forecasts)
                       for c in CROPS for h in HORIZONS}
    practical = {
        "CMapeBelowBase": scores["C"]["mape8_pct"] is not None and scores["C"]["mape8_pct"] < scores["BASE"]["mape8_pct"],
        "CMaeAtMostBase": scores["C"]["mae8"] is not None and scores["C"]["mae8"] <= scores["BASE"]["mae8"],
        "CMaeAtMostB0": scores["C"]["mae8"] is not None and scores["B0"]["mae8"] is not None and scores["C"]["mae8"] <= scores["B0"]["mae8"],
        "CPracticalCriteriaPass": (scores["C"]["mape8_pct"] is not None and scores["C"]["mape8_pct"] < scores["BASE"]["mape8_pct"]
                                   and scores["C"]["mae8"] is not None and scores["C"]["mae8"] <= scores["BASE"]["mae8"]),
        "tenPercentGoalAchieved": scores["C"]["mape8_pct"] is not None and scores["C"]["mape8_pct"] <= 10.,
        "remainingMappPctPointsTo10": max(0., float(scores["C"]["mape8_pct"])-10.) if scores["C"]["mape8_pct"] is not None else None,
        "cells": {},
    }
    for c in CROPS:
        for h in HORIZONS:
            key = f"{c}:{h}"
            base_cell, c_cell = scores["BASE"]["cells"][key], scores["C"]["cells"][key]
            practical["cells"][key] = {"maeDifference": c_cell["mae"]-base_cell["mae"] if c_cell["mae"] is not None and base_cell["mae"] is not None else None,
                "mapeDifferencePctPoints": c_cell["mape_pct"]-base_cell["mape_pct"] if c_cell["mape_pct"] is not None and base_cell["mape_pct"] is not None else None,
                "maeDirection": "improvement" if c_cell["mae"] is not None and base_cell["mae"] is not None and c_cell["mae"] < base_cell["mae"] else "worsening" if c_cell["mae"] is not None and base_cell["mae"] is not None and c_cell["mae"] > base_cell["mae"] else "inconclusive",
                "mapeDirection": "improvement" if c_cell["mape_pct"] is not None and base_cell["mape_pct"] is not None and c_cell["mape_pct"] < base_cell["mape_pct"] else "worsening" if c_cell["mape_pct"] is not None and base_cell["mape_pct"] is not None and c_cell["mape_pct"] > base_cell["mape_pct"] else "inconclusive"}
    dm = {"C_vs_BASE": _dm_results(forecasts, "C", "BASE"),
          "E_vs_C": _dm_results(forecasts, "E", "C")}
    blocks = {"C_vs_BASE": _block_bootstrap(forecasts, "C", "BASE"),
              "E_vs_C": _block_bootstrap(forecasts, "E", "C")}
    group_uses: dict[str, int] = {}
    for f in supply_map.values():
        gid = f.get("updateGroupId")
        if gid:
            group_uses[gid] = group_uses.get(gid, 0)+1
    # Preserve an exact reproduction receipt, not a V5 overwrite.
    _write_parquet(forecasts, out/"forecasts.parquet")
    report = {"status": "completed_practical_criteria_met" if practical["CPracticalCriteriaPass"] else "completed_no_gain", "protocolSha256": PROTOCOL_SHA,
              "plannedRows": len(forecasts), "validBaseActualKeys": sum(_evaluable(r) for r in forecasts),
              "fitCount": fit_count, "fitCap": FIT_CAP, "development": dev_score, "evaluation": scores,
              "B0MaeProtected": bool(scores["C"]["mae8"] is not None and scores["B0"]["mae8"] is not None and scores["C"]["mae8"] <= scores["B0"]["mae8"]),
              "practicalAssessment": practical,
              "fixedDiagnostics": _postfit_diagnostics(forecasts),
              "sampleCoverage": {"pairedBaseCRows": sum(r["actual"] is not None and r["BASE"] is not None and r.get("C") is not None for r in forecasts),
                                 "pairedCENumericSupportRows": len(supported_e), "pairedCoverageByCell": paired_coverage,
                                 "testPlannedRows": len(forecasts), "testValidLabelRows": sum(r["actual"] is not None for r in forecasts),
                                 "validRowsByCell": {k: scores["BASE"]["cells"][k]["n"] for k in scores["BASE"]["cells"]}},
              "statisticalAssessment": {"families": {"C_vs_BASE": {"hypothesesPerLag": 16, "holmAlpha": .05, "dmLagsDekads": list(DM_LAGS)},
                                                        "E_vs_C": {"hypothesesPerLag": 16, "holmAlpha": .05, "dmLagsDekads": list(DM_LAGS)}},
                                        "dieboldMarianoHacHolm": dm,
                                        "sharedCalendarBlockBootstrap": {"lengthsDekads": list(BLOCK_LENGTHS), "replicates": BOOTSTRAP_REPLICATES,
                                                                          "seed": BOOTSTRAP_SEED, "results": blocks}},
              "evidence": {"matchedNumericSupplyRows": len(supply_map),
                           "distinctSupplyUpdateGroups": len({v.get("updateGroupId") for v in supply_map.values() if v.get("updateGroupId")}),
                           "repeatedUpdateGroupRowAppearances": sum(max(0, n-1) for n in group_uses.values()),
                           "supplyTypedGroup": list(SUPPLY),
                           "rows2026": sum(_iso_day(r["targetStart"]).year == 2026 for r in forecasts),
                           "rows2026ExactEEqualsC": sum(_iso_day(r["targetStart"]).year == 2026 and r["E"] == r["C"] for r in forecasts),
                           "rows2026AllExactEEqualsC": all(r["E"] == r["C"] for r in forecasts if _iso_day(r["targetStart"]).year == 2026)},
              "retraining": {"alpha": ALPHA, "solver": "svd", "clipLogAdjustment": [LOWER, UPPER], "maxTrainingLabelMaturityAt": max((r["C_fit"]["trainMaxMaturityAt"] for r in forecasts if r.get("C_fit")), default=None)},
              "caveats": ["Retrospective re-used test period; no independent validation.", "Saved V5 G-B development forecasts were retrospectively selected.", "Daily snapshot is revised historical data and does not prove first-publication vintages."]}
    (out/"score-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    input_hashes_after = {name: sha256(path) for name, path in input_paths.items()}
    if input_hashes_before != input_hashes_after:
        raise ValueError("a pinned input changed during the run")
    manifest = _source_manifest(out, fit_count, len(forecasts), input_hashes_before, input_hashes_after)
    (out/"source-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return report


def _write_parquet(rows: list[dict], path: Path) -> None:
    # Remove internal-only objects/keys and ensure stable Arrow-friendly columns.
    clean = []
    for r in rows:
        item = {k: v for k, v in r.items() if not k.startswith("_")}
        for k in ("C_fit", "E_fit"):
            if item.get(k) is not None:
                item[k] = json.dumps(item[k], ensure_ascii=False, sort_keys=True, allow_nan=False)
        clean.append(item)
    pd.DataFrame(clean).to_parquet(path, index=False)


def _source_manifest(out: Path, fits: int, rows: int, before: dict, after: dict) -> dict:
    outputs = {name: out/name for name in ("forecasts.parquet", "development.json", "score-report.json", "preflight.json")}
    input_paths = {**_input_paths(), "implementation": Path(__file__).resolve()}
    return {"status": "forecast_complete", "reproductionCommand": "python -m agri_data.period_v9_correction run",
            "protocolSha256": PROTOCOL_SHA,
            "inputs": {name: {"path": str(path.relative_to(ROOT)), "sha256Before": before[name], "sha256After": after[name],
                              "sha256Pinned": PINNED_INPUT_SHA256.get(name)}
                       for name, path in input_paths.items()},
            "outputs": {name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)} for name, path in outputs.items()},
            "outputRows": rows, "ridgeFitCount": fits, "fitCap": FIT_CAP,
            "successfulFitCacheEntries": fits,
            "runtime": {"python": __import__("sys").version.split()[0], "numpy": np.__version__, "pandas": pd.__version__}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run"))
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.command == "preflight":
        print(json.dumps(preflight(), ensure_ascii=False))
        return 0
    report = run(args.out)
    print(json.dumps({"status": report["status"], "rows": report["plannedRows"], "fitCount": report["fitCount"],
                      "base": report["evaluation"]["BASE"], "C": report["evaluation"]["C"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
