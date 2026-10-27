"""Answer-free pooled MLP contrasts for ordered, summary and scrambled paths.

The frozen experiment uses one small shared model per arm and monthly fit,
pooling supported crop/horizon cells. This module never calls an API and never
attaches target answers to its forecast manifests.
"""
from __future__ import annotations

import argparse
from array import array
import copy
import datetime as dt
import hashlib
import json
import math
import random
import sys
import warnings
from pathlib import Path
from typing import Any

from . import direct_horizon, full_history_feature_library as library
from . import full_history_return_eval as narrow
from . import full_history_xgb_rf_eval as rich
from . import modeling

ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
SCREEN_COMPONENTS_PATH = ROOT / "outputs/experiments/ridge-sarima-weight-cv-20261008/historical-components.json"
EVALUATION_COMPONENTS_PATH = ROOT / "outputs/experiments/five-crop-specialists-20261009/forecast-manifest.json"
OUT_DIR = ROOT / "outputs/experiments/path-mlp-20261010"
SCREEN_OUT = OUT_DIR / "screen-manifest.json"
EVALUATION_OUT = OUT_DIR / "evaluation-manifest.json"

PRICES_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
SCREEN_COMPONENTS_SHA256 = "6ade501d5818cca33ad973963d12349f51b5ee3bed27afcd2cc023ae972fed74"
EVALUATION_COMPONENTS_SHA256 = "1497ee362945fcab45353a7b2403be752af3bf63c5f584cc0aedb2cc111ceaa8"
LIBRARY_SHA256 = "58bcde98d11239494ae13cc6169fb4c7f84a49a15ad11cf5b3cf5c49b4f39389"
RICH_HELPER_SHA256 = "1951b34bc019ccd9fe680776f4277105e1edb470d3d7d5a37802c781088d9245"
MODELING_SHA256 = "0ef149c231cc19f5990548822c10afecb16957863eddbd436d7d61a06c9963fc"
DIRECT_HELPER_SHA256 = "1cf3f8ce5455d9ce1f8d4eeb6e14f84e465acf4f5e6d512670b102115fb96d40"
NARROW_HELPER_SHA256 = "276f389e7153e07ef02836183d55e9823a6eb7c9bf24acc882d663946c55c5c5"

PRODUCTS = ("cabbage", "radish", "onion", "garlic", "pepper")
HORIZONS = tuple(range(1, 11))
TRAIN_START = dt.date(2014, 1, 1)
PHASES = {
    "screen": (dt.date(2024, 9, 25), dt.date(2025, 8, 31), dt.date(2024, 9, 15)),
    "evaluation": (dt.date(2026, 1, 1), dt.date(2026, 9, 27), dt.date(2026, 1, 1)),
}
MIN_CELL_TARGETS = 90
PATH_DAYS = 91
FEATURE_DIM = 293
CONTINUOUS_DIM = 197  # 106 preserved features plus 91 ordered log-price values.
MAX_CACHED_CUTOFFS_PER_CROP = 10
ARM_FIELDS = ("orderedBlend", "summaryBlend", "scrambledBlend")
ARM_TO_REP = {"orderedBlend": "ordered", "summaryBlend": "summary", "scrambledBlend": "scrambled"}
ENVIRONMENT = {"scikitLearn": "1.9.1", "numpy": "2.5.3", "scipy": "1.18.1", "threadpoolctl": "3.7.0"}
MLP_PARAMETERS = {
    "hidden_layer_sizes": [16], "activation": "relu", "solver": "adam", "alpha": 10.0,
    "batch_size": 256, "learning_rate_init": 0.001, "max_iter": 40, "tol": 0.0,
    "n_iter_no_change": 41, "early_stopping": False, "shuffle": True, "random_state": 0,
}
OMITTED_LIBRARY_FEATURES = frozenset({
    "p0_age_days", "target_weekday_sin", "target_weekday_cos",
    "target_annual_sin", "target_annual_cos",
})


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha_path(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _date(value: Any) -> dt.date | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _phase_grid(phase: str) -> list[tuple[str, dt.date, int]]:
    if phase not in PHASES:
        raise ValueError("phase must be screen or evaluation")
    start, end, _ = PHASES[phase]
    return [(crop, start + dt.timedelta(days=i), horizon)
            for crop in PRODUCTS for i in range((end - start).days + 1) for horizon in HORIZONS]


def _trim_prices(raw: dict[str, Any], phase_end: dt.date) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove later/malformed dates before inspecting their values."""
    if raw.get("schemaVersion") != 1 or not isinstance(raw.get("products"), list):
        raise ValueError("prices must use dashboard schemaVersion 1")
    products = []
    audit: dict[str, Any] = {}
    for product in raw["products"]:
        if not isinstance(product, dict) or product.get("id") not in PRODUCTS:
            raise ValueError("unknown or malformed product in price snapshot")
        observations = product.get("observations")
        if not isinstance(observations, list):
            raise ValueError("price product observations must be a list")
        kept = []
        later = invalid = 0
        for row in observations:
            if not isinstance(row, dict):
                invalid += 1
                continue
            day = _date(row.get("date"))
            if day is None:
                invalid += 1
            elif day > phase_end:
                later += 1
            else:
                kept.append(row)
        products.append({**product, "observations": kept})
        audit[product["id"]] = {"includedRowsThroughPhaseEnd": len(kept),
                                 "skippedLaterDates": later, "skippedInvalidDateRows": invalid}
    if len(products) != len(PRODUCTS) or {row["id"] for row in products} != set(PRODUCTS):
        raise ValueError("prices must contain every product exactly once")
    return {**raw, "products": products}, audit


def _check_hash(path: Path, expected: str, label: str) -> None:
    if _sha_path(path) != expected:
        raise ValueError(f"pinned {label} SHA-256 mismatch")


def _check_environment() -> dict[str, str]:
    import numpy
    import scipy
    import sklearn
    import threadpoolctl
    observed = {"scikitLearn": sklearn.__version__, "numpy": numpy.__version__,
                "scipy": scipy.__version__, "threadpoolctl": threadpoolctl.__version__}
    if observed != ENVIRONMENT:
        raise RuntimeError(f"pinned Python environment mismatch: {observed}")
    return observed


def _verify_input_hashes(phase: str, prices_raw: bytes, components_raw: bytes) -> dict[str, str]:
    if _sha_bytes(prices_raw) != PRICES_SHA256:
        raise ValueError("pinned price snapshot SHA-256 mismatch")
    component_sha = SCREEN_COMPONENTS_SHA256 if phase == "screen" else EVALUATION_COMPONENTS_SHA256
    if _sha_bytes(components_raw) != component_sha:
        raise ValueError("pinned component forecast SHA-256 mismatch")
    paths = {
        "fullHistoryFeatureLibrary": (Path(library.__file__), LIBRARY_SHA256),
        "richFeatureHelper": (Path(rich.__file__), RICH_HELPER_SHA256),
        "modelingHelper": (Path(modeling.__file__), MODELING_SHA256),
        "directHorizonHelper": (Path(direct_horizon.__file__), DIRECT_HELPER_SHA256),
        "narrowFeatureHelper": (Path(narrow.__file__), NARROW_HELPER_SHA256),
    }
    for label, (path, expected) in paths.items():
        _check_hash(path, expected, label)
    return {"prices": PRICES_SHA256, "components": component_sha,
            **{label: expected for label, (_, expected) in paths.items()},
            "generator": _sha_path(Path(__file__))}


def _component_rows(raw: dict[str, Any], phase: str) -> dict[tuple[str, dt.date, int], dict[str, Any]]:
    if raw.get("phase") != ("historical_components_only" if phase == "screen" else "forecast_only") or raw.get("targetValuesAttached") is not False:
        raise ValueError("component source must be answer-free in its pinned phase")
    key_rows = raw.get("forecasts")
    if not isinstance(key_rows, list):
        raise ValueError("component forecast rows are missing")
    result = {}
    for row in key_rows:
        if not isinstance(row, dict) or row.get("product") not in PRODUCTS:
            raise ValueError("malformed component forecast row")
        target, horizon = _date(row.get("target")), row.get("horizon")
        if target is None or isinstance(horizon, bool) or horizon not in HORIZONS:
            raise ValueError("invalid component forecast key")
        key = (row["product"], target, int(horizon))
        if key in result:
            raise ValueError("duplicate component forecast key")
        if "actual" in row or "targetValue" in row:
            raise ValueError("answer leakage in component manifest")
        if phase == "screen":
            direct, sarima = _positive(row.get("direct")), _positive(row.get("sarima"))
            # The frozen blend falls back to direct only when the source SARIMA
            # component is absent. Keep the missing SARIMA state explicit.
            fixed50 = (direct if sarima is None else .5 * direct + .5 * sarima) if direct is not None else None
        else:
            predictions = row.get("predictions")
            if not isinstance(predictions, dict):
                raise ValueError("specialist component predictions are missing")
            direct, sarima, fixed50 = (_positive(predictions.get("direct")),
                                       _positive(predictions.get("sarima")),
                                       _positive(predictions.get("fixed50")))
            expected = (direct if sarima is None else .5 * direct + .5 * sarima) if direct is not None else None
            if expected is None and fixed50 is not None or expected is not None and (fixed50 is None or not math.isclose(expected, fixed50, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError("pinned specialist fixed50 does not match its original components")
        result[key] = {"direct": direct, "sarima": sarima, "fixed50": fixed50,
                       "sarimaFallback": sarima is None}
    return result


def _source_rows(raw: dict[str, Any]) -> tuple[dict[str, dict[dt.date, tuple[float, float]]],
                                                dict[str, dict[dt.date, dict[str, Any]]]]:
    direct_series = direct_horizon._observations(raw)
    raw_maps, _audit = rich._raw_maps(raw)
    records: dict[str, dict[dt.date, dict[str, Any]]] = {}
    for crop in PRODUCTS:
        domain = direct_series[crop]
        if not set(domain).issubset(raw_maps[crop]):
            raise ValueError(f"positive price-volume source rows do not align for {crop}")
        records[crop] = {day: dict(raw_maps[crop][day]) for day in domain}
        for day, (price, volume) in domain.items():
            row = records[crop][day]
            if not math.isclose(row["price"], price, rel_tol=0., abs_tol=1e-12) or row.get("volume") is None or not math.isclose(row["volume"], volume, rel_tol=0., abs_tol=1e-12):
                raise ValueError(f"positive price-volume anchor mismatch for {crop}/{day}")
    return direct_series, records


def _feature_pair(crop: str, target: dt.date, horizon: int,
                  direct_series: dict[str, dict[dt.date, tuple[float, float]]],
                  records: dict[str, dict[dt.date, dict[str, Any]]],
                  caches: dict[str, dict[Any, Any]], ordered_days: dict[str, list[dt.date]],
                  price_maps: dict[str, dict[dt.date, float]],
                  scramble_cache: dict[tuple[str, dt.date], tuple[dict[dt.date, float], dict[str, Any]]] | None = None,
                  invariant_cache: dict[tuple[str, dt.date], tuple[list[float], list[float]]] | None = None) -> dict[str, Any] | None:
    origin = target - dt.timedelta(days=horizon)
    cutoff = origin - dt.timedelta(days=1)
    # Pass only dated source rows at or before the causal cutoff. Several
    # helpers build cached arrays from their entire input mapping before
    # selecting a bounded slice, so relying only on an internal bisect would
    # still read later target outcomes.
    causal_series = {day: value for day, value in direct_series[crop].items() if day <= cutoff}
    causal_records = {day: value for day, value in records[crop].items() if day <= cutoff}
    causal_days = [day for day in ordered_days[crop] if day <= cutoff]
    causal_prices = {day: value for day, value in price_maps[crop].items() if day <= cutoff}
    crop_caches = caches[crop]
    cache_key = ("causal_cutoff", cutoff)
    if cache_key not in crop_caches and len(crop_caches) >= MAX_CACHED_CUTOFFS_PER_CROP:
        # `_feature_row` stores a full dated/log-price prefix in each cache.
        # Retain a small rolling set instead of one O(history) pair per sample.
        del crop_caches[next(iter(crop_caches))]
    feature_cache = crop_caches.setdefault(cache_key, {})
    found = modeling._feature_row(causal_series, cutoff, target, horizon, feature_cache)
    if found is None:
        return None
    source_day, original40 = found
    p0_age = (origin - source_day).days
    if p0_age < 0 or p0_age > 14:
        return None
    expanded, p0, audit = library.features(causal_records, target, horizon,
                                           causal_days, causal_prices)
    if expanded is None or p0 is None:
        raise ValueError(f"expanded 71-feature library unavailable where direct row exists: {crop}/{target}/h{horizon}")
    if _date(audit.get("p0Date")) != source_day or not math.isclose(p0, direct_series[crop][source_day][0], rel_tol=0., abs_tol=1e-12):
        raise ValueError(f"direct/rich p0 differs: {crop}/{target}/h{horizon}")
    appended = [value for name, value in zip(library.FEATURE_NAMES, expanded)
                if name not in OMITTED_LIBRARY_FEATURES]
    if len(original40) != 40 or len(appended) != 66:
        raise ValueError("frozen feature schema must be exact 40+66")
    preserved = [float(value) for value in (*original40, *appended)]
    if not all(math.isfinite(value) for value in preserved):
        raise ValueError("nonfinite 106-feature vector")
    start = origin - dt.timedelta(days=PATH_DAYS)
    path_days = [start + dt.timedelta(days=i) for i in range(PATH_DAYS)]
    mask = [int(day in direct_series[crop]) for day in path_days]
    values = [math.log(direct_series[crop][day][0] / p0) if mask[i] else 0.
              for i, day in enumerate(path_days)]
    cache_key = (crop, origin)
    if scramble_cache is not None and cache_key in scramble_cache:
        scrambled, permutation_audit = scramble_cache[cache_key]
        if not math.isclose(direct_series[crop][source_day][0], p0, rel_tol=0., abs_tol=1e-12):
            raise AssertionError("cached scramble p0 mismatch")
    else:
        scrambled, permutation_audit = _scramble_path(crop, origin, source_day, path_days,
                                                       direct_series[crop], seed=0)
        if scramble_cache is not None:
            scramble_cache[cache_key] = (scrambled, permutation_audit)
    # The path permutation is keyed by crop/origin, so it is shared across
    # horizons. Verify the complete 106-vector on its first use. Other horizon
    # rows at this origin differ only at direct horizon/target-calendar inputs;
    # validate their invariant projection against the cached checked vectors.
    checked = invariant_cache.get(cache_key) if invariant_cache is not None else None
    invariant_indices = [i for i in range(106) if i not in {1, 36, 37, 38, 39}]
    if checked is None:
        counter_series = dict(direct_series[crop])
        for day, price in scrambled.items():
            counter_series[day] = (price, counter_series[day][1])
        counter_records = {day: dict(row) for day, row in records[crop].items()}
        counter_prices = dict(price_maps[crop])
        for day, price in scrambled.items():
            counter_records[day]["price"] = price
            counter_prices[day] = price
        counter_causal_series = {day: value for day, value in counter_series.items() if day <= cutoff}
        counter_causal_records = {day: value for day, value in counter_records.items() if day <= cutoff}
        counter_causal_prices = {day: value for day, value in counter_prices.items() if day <= cutoff}
        counter_causal_days = [day for day in ordered_days[crop] if day <= cutoff]
        counter_found = modeling._feature_row(counter_causal_series, cutoff, target, horizon, {})
        if counter_found is None:
            raise AssertionError(f"scramble removed direct features: {crop}/{target}/h{horizon}")
        counter40 = counter_found[1]
        counter71, counter_p0, _ = library.features(counter_causal_records, target, horizon,
                                                     counter_causal_days, counter_causal_prices)
        if counter71 is None or counter_p0 is None:
            raise AssertionError(f"scramble removed rich features: {crop}/{target}/h{horizon}")
        counter_appended = [value for name, value in zip(library.FEATURE_NAMES, counter71)
                            if name not in OMITTED_LIBRARY_FEATURES]
        counter106 = [float(value) for value in (*counter40, *counter_appended)]
        if len(counter106) != 106 or max((abs(a-b) for a, b in zip(preserved, counter106)), default=0.) > 1e-12:
            raise AssertionError(f"scrambled price path changed a preserved 106 feature: {crop}/{target}/h{horizon}")
        if counter_found[0] != source_day or not math.isclose(counter_p0, p0, rel_tol=0., abs_tol=1e-12):
            raise AssertionError(f"scrambled price path changed p0: {crop}/{target}/h{horizon}")
        if invariant_cache is not None:
            invariant_cache[cache_key] = ([preserved[i] for i in invariant_indices],
                                          [counter106[i] for i in invariant_indices])
    else:
        original_projection, counter_projection = checked
        if any(abs(a-b) > 1e-12 for a, b in zip(original_projection,
                [preserved[i] for i in invariant_indices])) or any(
                abs(a-b) > 1e-12 for a, b in zip(original_projection, counter_projection)):
            raise AssertionError(f"cached scramble invariant changed across horizons: {crop}/{target}/h{horizon}")
    counter_values = [math.log(scrambled.get(day, direct_series[crop][day][0]) / p0) if mask[i] else 0.
                     for i, day in enumerate(path_days)]
    if any(mask[i] != int(day in direct_series[crop]) for i, day in enumerate(path_days)):
        raise AssertionError("scramble changed a calendar observation mask")
    return {"preserved106": preserved, "p0": p0, "p0Date": source_day,
            "p0AgeDays": p0_age, "pathValues": values, "pathMask": mask,
            "scrambledValues": counter_values, "permutation": permutation_audit}


def _scramble_path(crop: str, origin: dt.date, p0_day: dt.date,
                   path_days: list[dt.date],
                   series: dict[dt.date, tuple[float, float]], seed: int = 0) -> tuple[dict[dt.date, float], dict[str, Any]]:
    """Permute positive prices within equal summary-membership buckets only."""
    cutoff = origin - dt.timedelta(days=1)
    p0 = p0_day
    protected: set[dt.date] = {p0}
    # Exact observed lag/anchor dates are held fixed.
    for lag in modeling.LAGS:
        cutoff_lag = cutoff - dt.timedelta(days=lag)
        anchors = [day for day in series if day <= cutoff_lag]
        if anchors:
            protected.add(max(anchors))
    def last_at_or_before(day: dt.date, tolerance: int | None = None) -> dt.date | None:
        candidates = [candidate for candidate in series if candidate <= day]
        if not candidates:
            return None
        found = max(candidates)
        return found if tolerance is None or (day-found).days <= tolerance else None
    prior = [day for day in series if day < p0]
    if prior and (p0 - max(prior)).days <= 7:
        protected.add(max(prior))
    for offset, tolerance in ((7, 7), (30, 14)):
        anchor = last_at_or_before(p0 - dt.timedelta(days=offset), tolerance)
        if anchor is not None:
            protected.add(anchor)
    for window in (3, 14, 60):
        high = p0 - dt.timedelta(days=window)
        low = high - dt.timedelta(days=window)
        candidates = [day for day in series if low <= day <= high]
        if candidates:
            protected.add(max(candidates))
    for window in (14, 60):
        days = [day for day in series if p0 - dt.timedelta(days=window-1) <= day <= p0]
        if days:
            protected.update((min(days), max(days)))

    windows: list[tuple[dt.date, dt.date]] = []
    for width in (7, 28, 90):
        windows.append((cutoff-dt.timedelta(days=width-1), cutoff))
    for width in (7, 30, 90):
        windows.append((p0-dt.timedelta(days=width-1), p0))
    for width in (3, 14, 60):
        windows.append((p0-dt.timedelta(days=2*width), p0-dt.timedelta(days=width)))
    for width in (14, 60):
        windows.append((p0-dt.timedelta(days=width-1), p0))
    buckets: dict[tuple[int, ...], list[dt.date]] = {}
    for day in path_days:
        if day not in series or day in protected:
            continue
        signature = tuple(int(low <= day <= high) for low, high in windows)
        buckets.setdefault(signature, []).append(day)
    material = hashlib.sha256(f"{seed}|{crop}|{origin.isoformat()}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(material, "big"))
    changed: dict[dt.date, float] = {}
    eligible = total_changed = multi_buckets = 0
    for days in buckets.values():
        if len(days) < 2:
            continue
        multi_buckets += 1
        prices = [series[day][0] for day in days]
        eligible += len(days)
        rng.shuffle(prices)
        for day, price in zip(days, prices):
            changed[day] = price
            if price != series[day][0]:
                total_changed += 1
    return changed, {"permutableObservedDates": eligible, "changedPriceDates": total_changed,
                     "changedFractionOfAllPathObservations": None,
                     "multiDateBuckets": multi_buckets,
                     "identityPermutation": total_changed == 0}


def _prepare_sample(crop: str, target: dt.date, horizon: int,
                    direct_series: dict[str, dict[dt.date, tuple[float, float]]],
                    records: dict[str, dict[dt.date, dict[str, Any]]],
                    caches: dict[str, dict[Any, Any]], ordered_days: dict[str, list[dt.date]],
                    price_maps: dict[str, dict[dt.date, float]],
                    scramble_cache: dict[tuple[str, dt.date], tuple[dict[dt.date, float], dict[str, Any]]] | None = None,
                    invariant_cache: dict[tuple[str, dt.date], tuple[list[float], list[float]]] | None = None) -> dict[str, Any] | None:
    sample = _feature_pair(crop, target, horizon, direct_series, records, caches, ordered_days, price_maps,
                           scramble_cache, invariant_cache)
    if sample is None:
        return None
    nmask = sum(sample["pathMask"])
    sample["permutation"]["changedFractionOfAllPathObservations"] = (
        sample["permutation"]["changedPriceDates"] / nmask if nmask else None)
    return sample


def _compact_feature_sample(sample: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep exact doubles in packed arrays instead of Python float lists."""
    if sample is None:
        return None
    return {"preserved106": array("d", sample["preserved106"]),
            "pathValues": array("d", sample["pathValues"]),
            "pathMask": bytes(sample["pathMask"]),
            "scrambledValues": array("d", sample["scrambledValues"]),
            "p0": sample["p0"], "p0Date": sample["p0Date"],
            "p0AgeDays": sample["p0AgeDays"],
            "permutation": sample["permutation"]}


def _cached_feature_sample(cache: dict[tuple[str, dt.date, int], dict[str, Any] | None],
                           key: tuple[str, dt.date, int],
                           compute: Any) -> dict[str, Any] | None:
    """Cache by the complete answer key, including explicitly unavailable rows."""
    if key not in cache:
        cache[key] = _compact_feature_sample(compute())
    return cache[key]


def _equal_cell_weights(cell_target_counts: dict[tuple[str, int], int]) -> tuple[dict[tuple[str, int], float], dict[str, Any]]:
    supported = {cell: count for cell, count in cell_target_counts.items()
                 if count >= MIN_CELL_TARGETS}
    total_rows = sum(supported.values())
    cell_count = len(supported)
    weights = {cell: total_rows / (cell_count * count)
               for cell, count in supported.items()} if cell_count else {}
    row_mean = (sum(count * weights[cell] for cell, count in supported.items()) / total_rows
                if total_rows else None)
    if row_mean is not None and not math.isclose(row_mean, 1.0, rel_tol=0., abs_tol=1e-12):
        raise AssertionError("equal-cell row weights must have mean one")
    return weights, {"supportedCells": cell_count, "trainingRows": total_rows,
                     "meanRowWeight": row_mean,
                     "minimumCellWeight": min(weights.values()) if weights else None,
                     "maximumCellWeight": max(weights.values()) if weights else None}


def _pack_features(scaled: list[float], mask: list[int], crop: str, representation: str,
                   scrambled_scaled: list[float] | None = None) -> list[float]:
    if len(scaled) != CONTINUOUS_DIM or len(mask) != PATH_DAYS or representation not in {"ordered", "summary", "scrambled"}:
        raise ValueError("invalid MLP feature pack")
    path = list(scaled[106:197])
    path_mask = list(mask)
    if representation == "summary":
        path = [0.] * PATH_DAYS
        path_mask = [0.] * PATH_DAYS
    elif representation == "scrambled":
        if scrambled_scaled is None or len(scrambled_scaled) != CONTINUOUS_DIM:
            raise ValueError("scrambled representation requires a transformed path")
        path = list(scrambled_scaled[106:197])
    one_hot = [float(crop == item) for item in PRODUCTS]
    out = list(scaled[:106]) + path + path_mask + one_hot
    if len(out) != FEATURE_DIM:
        raise AssertionError("MLP feature pack must have 293 columns")
    return out


def _component_fields(key: tuple[str, dt.date, int], source: dict[tuple[str, dt.date, int], dict[str, Any]]) -> dict[str, Any]:
    value = source.get(key)
    if value is None:
        missing = {"status": "not_issued", "reason": "component_key_absent", "prediction": None}
        return {"fixed50KeyPresent": False, "direct": copy.deepcopy(missing), "sarima": copy.deepcopy(missing),
                "fixed50": copy.deepcopy(missing)}
    fields = {}
    for name in ("direct", "sarima", "fixed50"):
        prediction = _positive(value.get(name))
        reason = None if prediction is not None else ("sarima_component_unavailable" if name == "sarima" else "component_unavailable")
        fields[name] = {"status": "issued" if prediction is not None else "not_issued",
                        "reason": reason, "prediction": prediction}
    return {"fixed50KeyPresent": True, **fields,
            "sarimaFallback": value["sarimaFallback"]}


def _not_issued(reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "not_issued", "reason": reason, "prediction": None, **extra}


def _unselected_arm() -> dict[str, Any]:
    return _not_issued("arm_not_selected_for_crop")


def _fit_month(
    phase: str,
    fit_origin: dt.date,
    active_arms: list[str],
    direct_series: dict[str, dict[dt.date, tuple[float, float]]],
    records: dict[str, dict[dt.date, dict[str, Any]]],
    ordered_days: dict[str, list[dt.date]],
    price_maps: dict[str, dict[dt.date, float]],
    forecast_keys: list[tuple[str, dt.date, int]],
    scramble_cache: dict[tuple[str, dt.date], tuple[dict[dt.date, float], dict[str, Any]]],
    invariant_cache: dict[tuple[str, dt.date], tuple[list[float], list[float]]],
    feature_sample_cache: dict[tuple[str, dt.date, int], dict[str, Any] | None],
) -> tuple[dict[tuple[str, dt.date, int], dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from threadpoolctl import threadpool_limits

    cache_by_crop = {crop: {} for crop in PRODUCTS}
    samples: dict[tuple[str, dt.date, int], dict[str, Any] | None] = {}
    cells: dict[tuple[str, int], list[tuple[str, dt.date, int]]] = {}
    for crop in PRODUCTS:
        for horizon in HORIZONS:
            cell = (crop, horizon)
            keys = []
            for target in direct_series[crop]:
                if target < TRAIN_START or target >= fit_origin:
                    continue
                key = (crop, target, horizon)
                sample = _cached_feature_sample(feature_sample_cache, key,
                    lambda: _prepare_sample(crop, target, horizon, direct_series, records,
                                             cache_by_crop, ordered_days, price_maps,
                                             scramble_cache, invariant_cache))
                if sample is None:
                    continue
                samples[key] = sample
                keys.append(key)
            cells[cell] = keys
    cell_counts = {f"{crop}|{horizon}": len(keys) for (crop, horizon), keys in cells.items()}
    target_counts = {cell: len({key[1] for key in keys}) for cell, keys in cells.items()}
    cell_weights, weight_audit = _equal_cell_weights(target_counts)
    supported = {cell: keys for cell, keys in cells.items() if cell in cell_weights}
    support_keys = [key for cell in supported.values() for key in cell]
    n, s = len(support_keys), len(supported)
    weights_by_key: dict[tuple[str, dt.date, int], float] = {}
    for cell, keys in supported.items():
        weight = cell_weights[cell]
        for key in keys:
            weights_by_key[key] = weight
    if support_keys and not math.isclose(sum(weights_by_key.values()) / len(weights_by_key), 1.0, rel_tol=0., abs_tol=1e-12):
        raise AssertionError("equal-cell row weights must have mean one")

    fit_results: dict[str, dict[str, Any]] = {}
    scaler = None
    models: dict[str, Any] = {}
    fit_diagnostics: dict[str, Any] = {}
    if n:
        from numpy import asarray, float64
        continuous = [samples[key]["preserved106"] + samples[key]["pathValues"] for key in support_keys]
        y = [math.log(direct_series[key[0]][key[1]][0] / samples[key]["p0"]) for key in support_keys]
        weights = [weights_by_key[key] for key in support_keys]
        with threadpool_limits(limits=1):
            scaler = StandardScaler()
            scaler.fit(asarray(continuous, dtype=float64), sample_weight=asarray(weights, dtype=float64))
            scaled = scaler.transform(asarray(continuous, dtype=float64))
            for arm in active_arms:
                rep = ARM_TO_REP[arm]
                matrix = []
                scrambled_scaled = None
                if rep == "scrambled":
                    scrambled_raw = [samples[key]["preserved106"] + samples[key]["scrambledValues"]
                                     for key in support_keys]
                    scrambled_scaled = scaler.transform(asarray(scrambled_raw, dtype=float64))
                for i, key in enumerate(support_keys):
                    matrix.append(_pack_features(list(scaled[i]), samples[key]["pathMask"], key[0], rep,
                                                 list(scrambled_scaled[i]) if scrambled_scaled is not None else None))
                fit_warnings: list[str] = []
                model = MLPRegressor(hidden_layer_sizes=(16,), activation="relu", solver="adam",
                                     alpha=10, batch_size=256, learning_rate_init=.001,
                                     max_iter=40, tol=0, n_iter_no_change=41,
                                     early_stopping=False, shuffle=True, random_state=0)
                try:
                    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
                        warnings.simplefilter("always")
                        model.fit(asarray(matrix, dtype=float64), asarray(y, dtype=float64),
                                  sample_weight=asarray(weights, dtype=float64))
                    fit_warnings = [f"{warning.category.__name__}:{str(warning.message)}" for warning in caught]
                    models[arm] = model
                    fit_diagnostics[arm] = {"status": "fit", "nIter": int(model.n_iter_),
                                            "finalLoss": float(model.loss_), "warnings": fit_warnings}
                except Exception as exc:
                    fit_diagnostics[arm] = {"status": "fit_failed", "reason": f"fit_failed:{type(exc).__name__}",
                                            "warnings": fit_warnings}
    else:
        fit_diagnostics = {arm: {"status": "not_fit", "reason": "no_supported_crop_horizon_cells"}
                           for arm in active_arms}

    # Prepare all monthly prediction features once, with complete audits.
    target_samples: dict[tuple[str, dt.date, int], dict[str, Any] | None] = {}
    for key in forecast_keys:
        crop, target, horizon = key
        target_samples[key] = _cached_feature_sample(feature_sample_cache, key,
            lambda: _prepare_sample(crop, target, horizon, direct_series, records,
                                    cache_by_crop, ordered_days, price_maps,
                                    scramble_cache, invariant_cache))
    predictions: dict[tuple[str, dt.date, int], dict[str, Any]] = {}
    forecast_permutations: dict[str, dict[str, Any]] = {}
    for key in forecast_keys:
        crop, target, horizon = key
        cell = (crop, horizon)
        sample = target_samples[key]
        row: dict[str, Any] = {}
        for arm in active_arms:
            reason = None
            if len({k[1] for k in cells[cell]}) < MIN_CELL_TARGETS:
                row[arm] = _not_issued("fewer_than_90_training_targets_in_crop_horizon_cell",
                                       trainingTargetCount=cell_counts[f"{crop}|{horizon}"])
                continue
            if fit_diagnostics.get(arm, {}).get("status") != "fit":
                row[arm] = _not_issued(fit_diagnostics.get(arm, {}).get("reason", "model_not_fit"))
                continue
            if sample is None:
                row[arm] = _not_issued("feature_row_unavailable_or_p0_older_than_14_days")
                continue
            rep = ARM_TO_REP[arm]
            try:
                with threadpool_limits(limits=1):
                    ordered_continuous = scaler.transform([sample["preserved106"] + sample["pathValues"]])[0]
                    scrambled_continuous = None
                    if rep == "scrambled":
                        scrambled_continuous = scaler.transform([sample["preserved106"] + sample["scrambledValues"]])[0]
                    vector = _pack_features(list(ordered_continuous), sample["pathMask"], crop, rep,
                                            list(scrambled_continuous) if scrambled_continuous is not None else None)
                    log_return = float(models[arm].predict([vector])[0])
                pred = sample["p0"] * math.exp(log_return)
                if not math.isfinite(pred) or pred <= 0:
                    raise ValueError("nonpositive_or_nonfinite_prediction")
                row[arm] = {"status": "issued", "reason": None, "prediction": pred,
                            "trainingTargetCount": cell_counts[f"{crop}|{horizon}"],
                            "p0Date": sample["p0Date"].isoformat(), "p0AgeDays": sample["p0AgeDays"],
                            "scrambleAuditId": f"{crop}|{(target-dt.timedelta(days=horizon)).isoformat()}" if rep == "scrambled" else None}
            except (OverflowError, ValueError) as exc:
                row[arm] = _not_issued(f"prediction_invalid:{type(exc).__name__}",
                                       trainingTargetCount=cell_counts[f"{crop}|{horizon}"])
            except Exception as exc:
                row[arm] = _not_issued(f"prediction_failed:{type(exc).__name__}",
                                       trainingTargetCount=cell_counts[f"{crop}|{horizon}"])
        predictions[key] = row
        if sample is not None:
            crop, target, horizon = key
            audit_id = f"{crop}|{(target-dt.timedelta(days=horizon)).isoformat()}"
            audit_value = sample["permutation"]
            prior = forecast_permutations.get(audit_id)
            if prior is not None and prior != audit_value:
                raise AssertionError("same crop/origin permutation differs across horizons")
            forecast_permutations[audit_id] = audit_value
    training_perm = [sample["permutation"] for sample in samples.values()]
    training_scramble_audit = {"featureRows": len(training_perm),
                               "changedRows": sum(item["changedPriceDates"] > 0 for item in training_perm),
                               "identityRows": sum(item["identityPermutation"] for item in training_perm),
                               "permutableObservedDates": sum(item["permutableObservedDates"] for item in training_perm),
                               "changedPriceDates": sum(item["changedPriceDates"] for item in training_perm)}
    audit = {"fitOrigin": fit_origin.isoformat(), "supportedCells": s, "trainingRows": n,
             "cellTrainingTargetCounts": cell_counts,
             "equalCellWeight": {"formula": "N/(S*Ni)", "mean": weight_audit["meanRowWeight"],
                                 "minimum": min(weights_by_key.values()) if weights_by_key else None,
                                 "maximum": max(weights_by_key.values()) if weights_by_key else None},
             "fitDiagnostics": fit_diagnostics,
             "scrambleTrainingAudit": training_scramble_audit}
    return predictions, audit, forecast_permutations


def _monthly_fit_origins(phase: str) -> dict[tuple[int, int], dt.date]:
    start, end, earliest = PHASES[phase]
    grid = _phase_grid(phase)
    result: dict[tuple[int, int], dt.date] = {}
    for crop, target, horizon in grid:
        origin = target - dt.timedelta(days=horizon)
        if origin < earliest:
            continue
        month = (origin.year, origin.month)
        result[month] = min(result.get(month, origin), origin)
    return dict(sorted(result.items()))


def _validate_decision(decision_raw: bytes, screen_manifest_raw: bytes,
                       screen_report_raw: bytes) -> tuple[dict[str, str], dict[str, str]]:
    decision = json.loads(decision_raw)
    expected_fields = {"schemaVersion", "phase", "screenManifestSha256", "screenReportSha256",
                       "scorerSha256", "selections", "decisionSha256"}
    if not isinstance(decision, dict) or set(decision) != expected_fields:
        raise ValueError("screen selection decision schema mismatch")
    if decision.get("schemaVersion") != 1 or decision.get("phase") != "selection_decision":
        raise ValueError("screen selection decision type mismatch")
    unsigned = dict(decision)
    saved = unsigned.pop("decisionSha256")
    # The scorer signs compact, sorted JSON followed by exactly one newline.
    if _sha_bytes(_canonical(unsigned) + b"\n") != saved:
        raise ValueError("screen decision digest mismatch")
    if decision["screenManifestSha256"] != _sha_bytes(screen_manifest_raw):
        raise ValueError("screen manifest does not match evaluation decision")
    if decision["screenReportSha256"] != _sha_bytes(screen_report_raw):
        raise ValueError("screen report does not match evaluation decision")
    selections = decision.get("selections")
    if not isinstance(selections, dict) or set(selections) != set(PRODUCTS):
        raise ValueError("screen decision must list every crop")
    allowed = set(ARM_FIELDS) | {"retain_fixed50"}
    if any(value not in allowed for value in selections.values()):
        raise ValueError("screen decision has an unknown selected arm")
    hashes = {"screenManifest": decision["screenManifestSha256"],
              "screenReport": decision["screenReportSha256"],
              "screenScorer": decision["scorerSha256"],
              "screenDecision": saved, "screenDecisionFile": _sha_bytes(decision_raw)}
    return selections, hashes


def _replay_screen_for_evaluation(prices_raw: bytes, decision_raw: bytes,
                                  screen_manifest_raw: bytes,
                                  screen_report_raw: bytes,
                                  selections: dict[str, str],
                                  decision_hashes: dict[str, str]) -> None:
    """Require the screen forecast, score and decision to replay before evaluation fitting."""
    from . import path_mlp_score

    actual_scorer_sha = _sha_path(Path(path_mlp_score.__file__))
    if decision_hashes.get("screenScorer") != actual_scorer_sha:
        raise ValueError("selection decision was not signed by the current scorer source")
    replay = path_mlp_score.validate_screen_artifacts(
        screen_manifest_raw, prices_raw, SCREEN_COMPONENTS_PATH.read_bytes(),
        screen_report_raw, decision_raw)
    if (replay.get("selection") != selections
            or replay.get("manifestSha256") != decision_hashes.get("screenManifest")
            or replay.get("screenReportSha256") != decision_hashes.get("screenReport")
            or replay.get("decisionSha256") != decision_hashes.get("screenDecision")):
        raise ValueError("screen replay binding differs from the evaluation decision")


def _validate_manifest(doc: dict[str, Any], phase: str) -> None:
    if doc.get("phase") != "forecast_only" or doc.get("phaseName") != phase or doc.get("targetValuesAttached") is not False:
        raise ValueError("path-MLP manifest must be answer-free for its phase")
    rows = doc.get("forecasts")
    expected = _phase_grid(phase)
    if not isinstance(rows, list) or len(rows) != len(expected) or doc.get("plannedCount") != len(expected):
        raise ValueError("path-MLP forecast grid size mismatch")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("malformed path-MLP forecast row")
        crop, target, horizon = row.get("crop"), _date(row.get("target_date")), row.get("horizon")
        if crop not in PRODUCTS or target is None or isinstance(horizon, bool) or horizon not in HORIZONS:
            raise ValueError("invalid path-MLP row key")
        key = (crop, target, int(horizon))
        if key in seen:
            raise ValueError("duplicate path-MLP row key")
        seen.add(key)
        if "actual" in row or "targetValue" in row:
            raise ValueError("target answer leaked into forecast manifest")
        if row.get("origin_date") != (target - dt.timedelta(days=horizon)).isoformat():
            raise ValueError("origin/horizon mismatch")
        for arm in ARM_FIELDS:
            item = row.get(arm)
            if not isinstance(item, dict) or item.get("status") not in {"issued", "not_issued"}:
                raise ValueError("invalid candidate-arm state")
            prediction = _positive(item.get("prediction"))
            if item["status"] == "issued" and prediction is None:
                raise ValueError("issued candidate must be finite and positive")
            if item["status"] == "not_issued" and item.get("prediction") is not None:
                raise ValueError("not-issued candidate cannot carry a prediction")
        for field in ("fixed50", "direct", "sarima"):
            item = row.get(field)
            if not isinstance(item, dict) or item.get("status") not in {"issued", "not_issued"}:
                raise ValueError("invalid inherited component state")
    if seen != set(expected):
        raise ValueError("path-MLP forecast grid key mismatch")
    unsigned = dict(doc)
    saved = unsigned.pop("manifestSha256", None)
    if saved != _sha_bytes(_canonical(unsigned)):
        raise ValueError("path-MLP manifest digest mismatch")


def _preview_json_value(value: Any, limit: int = 120) -> str:
    preview = repr(value)
    return preview if len(preview) <= limit else preview[:limit - 3] + "..."


def _first_difference(actual: Any, expected: Any, path: str = "$") -> dict[str, str] | None:
    """Return a bounded, JSON-safe description of the first replay mismatch."""
    if type(actual) is not type(expected):
        return {"path": path, "actualType": type(actual).__name__,
                "expectedType": type(expected).__name__,
                "actual": _preview_json_value(actual),
                "expected": _preview_json_value(expected)}
    if isinstance(actual, dict):
        actual_keys, expected_keys = set(actual), set(expected)
        for key in sorted(actual_keys | expected_keys,
                          key=lambda item: (item in {"manifestSha256", "decisionSha256"}, str(item))):
            child = f"{path}.{key}"
            if key not in actual:
                return {"path": child, "actualType": "missing",
                        "expectedType": type(expected[key]).__name__,
                        "actual": "<missing>", "expected": _preview_json_value(expected[key])}
            if key not in expected:
                return {"path": child, "actualType": type(actual[key]).__name__,
                        "expectedType": "missing",
                        "actual": _preview_json_value(actual[key]), "expected": "<missing>"}
            found = _first_difference(actual[key], expected[key], child)
            if found is not None:
                return found
        return None
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return {"path": path, "actualType": "list", "expectedType": "list",
                    "actual": f"length={len(actual)}", "expected": f"length={len(expected)}"}
        for index, (left, right) in enumerate(zip(actual, expected)):
            found = _first_difference(left, right, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if actual != expected:
        return {"path": path, "actualType": type(actual).__name__,
                "expectedType": type(expected).__name__,
                "actual": _preview_json_value(actual),
                "expected": _preview_json_value(expected)}
    return None


def build(prices_raw: bytes, components_raw: bytes, phase: str,
          *, decision_raw: bytes | None = None, screen_manifest_raw: bytes | None = None,
          screen_report_raw: bytes | None = None) -> dict[str, Any]:
    """Build one frozen phase from pinned byte inputs; no output is written."""
    if phase not in PHASES:
        raise ValueError("phase must be screen or evaluation")
    if phase == "evaluation":
        if decision_raw is None or screen_manifest_raw is None or screen_report_raw is None:
            raise ValueError("evaluation requires the verified screen manifest, report and selection decision")
        selections, decision_hashes = _validate_decision(decision_raw, screen_manifest_raw, screen_report_raw)
        _replay_screen_for_evaluation(prices_raw, decision_raw, screen_manifest_raw,
                                      screen_report_raw, selections, decision_hashes)
        if not any(value in ARM_FIELDS for value in selections.values()):
            raise ValueError("no candidate selected; evaluation phase must not fit")
        screen_doc = json.loads(screen_manifest_raw)
        _validate_manifest(screen_doc, "screen")
    else:
        if any(value is not None for value in (decision_raw, screen_manifest_raw, screen_report_raw)):
            raise ValueError("screen phase must not receive a selection decision")
        selections, decision_hashes = {}, {}

    input_hashes = _verify_input_hashes(phase, prices_raw, components_raw)
    env = _check_environment()
    phase_end = PHASES[phase][1]
    price_doc = json.loads(prices_raw)
    if not isinstance(price_doc, dict):
        raise ValueError("price snapshot root must be an object")
    trimmed, trim_audit = _trim_prices(price_doc, phase_end)
    direct_series, records = _source_rows(trimmed)
    ordered_days = {crop: sorted(records[crop]) for crop in PRODUCTS}
    price_maps = {crop: {day: row["price"] for day, row in records[crop].items()} for crop in PRODUCTS}
    component_doc = json.loads(components_raw)
    component_map = _component_rows(component_doc, phase)
    grid = _phase_grid(phase)
    start, end, min_origin = PHASES[phase]
    by_month: dict[tuple[int, int], list[tuple[str, dt.date, int]]] = {}
    rows: dict[tuple[str, dt.date, int], dict[str, Any]] = {}
    for key in grid:
        crop, target, horizon = key
        origin = target - dt.timedelta(days=horizon)
        if origin >= min_origin:
            by_month.setdefault((origin.year, origin.month), []).append(key)
        else:
            component = _component_fields(key, component_map)
            row = {"crop": crop, "target_date": target.isoformat(), "origin_date": origin.isoformat(),
                   "horizon": horizon, "eligible": False,
                   "candidateStatusReason": "origin_before_phase_start",
                   **component,
                   "ordered": _not_issued("origin_before_phase_start"),
                   "summary": _not_issued("origin_before_phase_start"),
                   "scrambled": _not_issued("origin_before_phase_start"),
                   "orderedBlend": _not_issued("origin_before_phase_start"),
                   "summaryBlend": _not_issued("origin_before_phase_start"),
                   "scrambledBlend": _not_issued("origin_before_phase_start")}
            rows[key] = row

    monthly_audits = []
    shared_scramble_cache: dict[tuple[str, dt.date], tuple[dict[dt.date, float], dict[str, Any]]] = {}
    shared_invariant_cache: dict[tuple[str, dt.date], tuple[list[float], list[float]]] = {}
    shared_feature_sample_cache: dict[tuple[str, dt.date, int], dict[str, Any] | None] = {}
    forecast_permutation_audit: dict[str, dict[str, Any]] = {}
    fit_origins = _monthly_fit_origins(phase)
    for fit_index, (month, fit_origin) in enumerate(fit_origins.items(), 1):
        forecast_keys = by_month.get(month, [])
        if not forecast_keys:
            continue
        print(f"path_mlp {phase} fit {fit_index}/{len(fit_origins)} "
              f"month={month[0]:04d}-{month[1]:02d} fitOrigin={fit_origin.isoformat()} start",
              file=sys.stderr, flush=True)
        if phase == "screen":
            active_arms = list(ARM_FIELDS)
        else:
            active_arms = list(dict.fromkeys(value for value in selections.values() if value in ARM_FIELDS))
        # The same global network per arm pools all supported crop × horizon cells.
        arm_predictions, training_audit, month_permutations = _fit_month(
            phase, fit_origin, active_arms, direct_series, records, ordered_days,
            price_maps, forecast_keys, shared_scramble_cache, shared_invariant_cache,
            shared_feature_sample_cache)
        monthly_audits.append(training_audit)
        fitted_arms = [arm for arm, value in training_audit["fitDiagnostics"].items()
                       if value.get("status") == "fit"]
        print(f"path_mlp {phase} fit {fit_index}/{len(fit_origins)} "
              f"month={month[0]:04d}-{month[1]:02d} fitOrigin={fit_origin.isoformat()} "
              f"end trainingRows={training_audit['trainingRows']} "
              f"supportedCells={training_audit['supportedCells']} fittedArms={len(fitted_arms)} "
              f"cachedFeatureKeys={len(shared_feature_sample_cache)}",
              file=sys.stderr, flush=True)
        for audit_id, audit_value in month_permutations.items():
            existing = forecast_permutation_audit.get(audit_id)
            if existing is not None and existing != audit_value:
                raise AssertionError("crop/origin scramble changed across monthly fits")
            forecast_permutation_audit[audit_id] = audit_value
        audit_index = len(monthly_audits) - 1
        for key in forecast_keys:
            crop, target, horizon = key
            component = _component_fields(key, component_map)
            origin = target - dt.timedelta(days=horizon)
            available_arms = set(active_arms)
            raw_models: dict[str, Any] = {}
            blended: dict[str, Any] = {}
            for rep, blend_name in (("ordered", "orderedBlend"), ("summary", "summaryBlend"), ("scrambled", "scrambledBlend")):
                if phase == "evaluation" and selections.get(crop) != blend_name:
                    model = _unselected_arm()
                    blend = _unselected_arm()
                elif blend_name not in available_arms:
                    model = _unselected_arm()
                    blend = _unselected_arm()
                else:
                    model = copy.deepcopy(arm_predictions[key][blend_name])
                    if model["status"] != "issued":
                        blend = _not_issued(model.get("reason") or "model_not_issued")
                    else:
                        sarima = component["sarima"]["prediction"]
                        blend_value = (model["prediction"] if sarima is None
                                       else .5 * model["prediction"] + .5 * sarima)
                        if not math.isfinite(blend_value) or blend_value <= 0:
                            blend = _not_issued("blend_prediction_invalid")
                        else:
                            blend = {"status": "issued", "reason": None,
                                     "prediction": blend_value,
                                     "sarimaFallback": sarima is None,
                                     "sarimaFallbackReason": component["sarima"]["reason"] if sarima is None else None}
                raw_models[rep] = model
                raw_models[blend_name] = blend
                blended[blend_name] = blend
            rows[key] = {"crop": crop, "target_date": target.isoformat(),
                         "origin_date": origin.isoformat(), "horizon": horizon,
                         "eligible": True, "fitOrigin": fit_origin.isoformat(),
                         "trainingAuditIndex": audit_index, **component, **raw_models,
                         "selectedArm": selections.get(crop) if phase == "evaluation" else None,
                         "fixed50KeyPresent": component["fixed50KeyPresent"]}

    ordered_rows = [rows[key] for key in grid]
    perm_counts: dict[str, dict[str, int]] = {crop: {"forecastOrigins": 0, "changedOrigins": 0,
                                                   "identityOrigins": 0, "permutableObservedDates": 0,
                                                   "changedPriceDates": 0}
                                               for crop in PRODUCTS}
    permutation_rows = []
    for audit_id, info in sorted(forecast_permutation_audit.items()):
        crop, origin = audit_id.split("|", 1)
        c = perm_counts[crop]
        c["forecastOrigins"] += 1
        c["permutableObservedDates"] += info["permutableObservedDates"]
        c["changedPriceDates"] += info["changedPriceDates"]
        if info["changedPriceDates"]:
            c["changedOrigins"] += 1
        else:
            c["identityOrigins"] += 1
        permutation_rows.append({"crop": crop, "origin_date": origin, **info})

    doc = {"schemaVersion": 1, "phase": "forecast_only", "phaseName": phase,
           "targetValuesAttached": False,
           "protocol": {"id": "path-mlp-ordered-summary-scrambled-2026-10-10-v1",
                        "phase": phase, "products": list(PRODUCTS), "horizons": list(HORIZONS),
                        "targetDates": [start.isoformat(), end.isoformat()],
                        "firstEligibleOrigin": min_origin.isoformat(),
                        "trainingStart": TRAIN_START.isoformat(),
                        "refit": "one global model per active arm at each month first eligible origin, pooling all supported crop-horizon cells; labels target < fitOrigin",
                        "minimumUniqueTargetsPerCell": MIN_CELL_TARGETS,
                        "featureCutoff": "origin-1 for every training and prediction row; p0 source must be no older than 14 calendar days at origin",
                        "inputDimension": FEATURE_DIM,
                        "featureLayout": {"preserved": 106, "pathValues": 91, "pathMask": 91, "cropOneHot": 5},
                        "scaler": "one weighted StandardScaler per monthly fit over 106 preserved plus 91 ordered log-price columns, shared by arms; masks/crop one-hot remain binary; summary zeros path and mask after scaling",
                        "scramble": "deterministic seed0 plus sha256(crop,origin); permute price values within identical window-membership buckets, keeping p0, exact lag/anchor/slope endpoints and mask fixed; verify all 106 features and p0 per feature row",
                        "trainingWeight": "N/(S*Ni) per row across S supported cells; mean weight one",
                        "target": "log(actual/p0); positive finite inverse p0*exp(output); no clipping",
                        "mlp": MLP_PARAMETERS, "activeArms": list(ARM_FIELDS) if phase == "screen" else list(active_arms),
                        "blend": "0.5*MLP + 0.5*inherited SARIMA; if SARIMA is unavailable, MLP alone",
                        "evaluationSelection": selections if phase == "evaluation" else None,
                        "evaluationDecisionHashes": decision_hashes},
           "inputSha256": input_hashes,
           "sourceAudit": {"priceRows": trim_audit,
                           "positivePriceVolumeDays": {crop: len(direct_series[crop]) for crop in PRODUCTS},
                           "componentKeyCount": len(component_map)},
           "environment": {**env, "python": sys.version.split()[0], "blasThreads": 1},
           "monthlyTrainingAudit": monthly_audits,
           "scrambleAudit": {"forecastByCrop": perm_counts, "forecastOrigins": permutation_rows},
           "plannedCount": len(ordered_rows), "forecasts": ordered_rows}
    doc["manifestSha256"] = _sha_bytes(_canonical(doc))
    _validate_manifest(doc, phase)
    return doc


def validate_manifest_bytes(manifest_raw: bytes, prices_raw: bytes, components_raw: bytes,
                            phase: str, *, decision_bytes: bytes | None = None,
                            decision_raw: bytes | None = None,
                            screen_manifest_bytes: bytes | None = None,
                            screen_report_bytes: bytes | None = None,
                            screen_manifest_raw: bytes | None = None,
                            screen_report_raw: bytes | None = None) -> dict[str, Any]:
    """Validate digest/schema and replay the complete deterministic forecast manifest."""
    doc = json.loads(manifest_raw)
    _validate_manifest(doc, phase)
    if decision_bytes is not None and decision_raw is not None:
        raise ValueError("pass only one decision byte argument")
    if screen_manifest_bytes is not None and screen_manifest_raw is not None:
        raise ValueError("pass only one screen-manifest byte argument")
    if screen_report_bytes is not None and screen_report_raw is not None:
        raise ValueError("pass only one screen-report byte argument")
    decision_raw = decision_bytes if decision_bytes is not None else decision_raw
    screen_manifest_raw = screen_manifest_bytes if screen_manifest_bytes is not None else screen_manifest_raw
    screen_report_raw = screen_report_bytes if screen_report_bytes is not None else screen_report_raw
    expected = build(prices_raw, components_raw, phase, decision_raw=decision_raw,
                     screen_manifest_raw=screen_manifest_raw,
                     screen_report_raw=screen_report_raw)
    if doc != expected:
        mismatch = _first_difference(doc, expected)
        detail = json.dumps(mismatch, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        raise ValueError(f"path-MLP manifest replay mismatch: {detail}")
    file_expected = json.dumps(expected, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    if manifest_raw != file_expected:
        raise ValueError("path-MLP manifest file bytes are not canonical")
    return doc


def _write_exclusive(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")


def build_from_paths(phase: str, output: Path, *, decision_path: Path | None = None,
                     screen_manifest_path: Path | None = None,
                     screen_report_path: Path | None = None,
                     prices_path: Path = PRICES_PATH,
                     components_path: Path | None = None) -> dict[str, Any]:
    if phase == "screen":
        if any(item is not None for item in (decision_path, screen_manifest_path, screen_report_path)):
            raise ValueError("screen phase does not accept evaluation selection inputs")
        source_path = SCREEN_COMPONENTS_PATH
    elif phase == "evaluation":
        if not all(item is not None for item in (decision_path, screen_manifest_path, screen_report_path)):
            raise ValueError("evaluation requires decision, screen manifest and screen report")
        source_path = EVALUATION_COMPONENTS_PATH
    else:
        raise ValueError("phase must be screen or evaluation")
    if components_path is not None:
        source_path = components_path
    doc = build(prices_path.read_bytes(), source_path.read_bytes(), phase,
                decision_raw=decision_path.read_bytes() if decision_path else None,
                screen_manifest_raw=screen_manifest_path.read_bytes() if screen_manifest_path else None,
                screen_report_raw=screen_report_path.read_bytes() if screen_report_path else None)
    _write_exclusive(output, doc)
    return doc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build", help="create one answer-free forecast manifest")
    build_parser.add_argument("--phase", choices=tuple(PHASES), required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--decision", type=Path)
    build_parser.add_argument("--screen-manifest", type=Path)
    build_parser.add_argument("--screen-report", type=Path)
    build_parser.add_argument("--prices", type=Path, default=PRICES_PATH)
    build_parser.add_argument("--components", type=Path)
    verify = sub.add_parser("verify", help="replay and verify an existing manifest")
    verify.add_argument("--phase", choices=tuple(PHASES), required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--decision", type=Path)
    verify.add_argument("--screen-manifest", type=Path)
    verify.add_argument("--screen-report", type=Path)
    verify.add_argument("--prices", type=Path, default=PRICES_PATH)
    verify.add_argument("--components", type=Path)
    args = parser.parse_args()
    if args.command == "build":
        doc = build_from_paths(args.phase, args.output, decision_path=args.decision,
                               screen_manifest_path=args.screen_manifest,
                               screen_report_path=args.screen_report,
                               prices_path=args.prices, components_path=args.components)
        print(json.dumps({"status": "forecast_manifest_written", "phase": args.phase,
                          "output": str(args.output), "plannedCount": doc["plannedCount"],
                          "manifestSha256": doc["manifestSha256"],
                          "fitCount": len(doc["monthlyTrainingAudit"])}, ensure_ascii=False))
        return 0
    component_path = args.components or (SCREEN_COMPONENTS_PATH if args.phase == "screen" else EVALUATION_COMPONENTS_PATH)
    raw = validate_manifest_bytes(args.manifest.read_bytes(), args.prices.read_bytes(),
                                  component_path.read_bytes(), args.phase,
                                  decision_bytes=args.decision.read_bytes() if args.decision else None,
                                  screen_manifest_raw=args.screen_manifest.read_bytes() if args.screen_manifest else None,
                                  screen_report_raw=args.screen_report.read_bytes() if args.screen_report else None)
    print(json.dumps({"status": "manifest_verified", "phase": args.phase,
                      "manifestSha256": raw["manifestSha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
