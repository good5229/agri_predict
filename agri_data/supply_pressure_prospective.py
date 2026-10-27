"""Offline builder for the frozen supply-pressure/seasonal-error forecast."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo
import uuid

from . import dashboard, live_forecast, modeling

KST = ZoneInfo("Asia/Seoul")
HORIZONS = tuple(range(1, 11))
FEATURES = ("horizon", "targetWeekdaySin", "targetWeekdayCos", "targetYearSin", "targetYearCos",
            "priceChange7", "priceChange28", "logMeanVolume7Minus28", "seasonalLogGap",
            "horizonTimesLogMeanVolume7Minus28", "horizonTimesSeasonalLogGap",
            "coverage7", "coverage28")
_INDEXES: dict[int, tuple[Any, tuple[dt.date, ...], dict[Any, float | None]]] = {}


def _index(series):
    key = id(series)
    cached = _INDEXES.get(key)
    if cached is None or cached[0] is not series:
        days = tuple(sorted(series))
        seasonal = {}
        _INDEXES[key] = (series, days, seasonal)
        return days, seasonal
    return cached[1], cached[2]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_dt(value: str) -> dt.datetime:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def _series(product: dict[str, Any]) -> dict[dt.date, tuple[float, float]]:
    result = {}
    for row in product.get("observations", []):
        day = dt.date.fromisoformat(row["date"])
        price, volume = float(row["actual"]), float(row["volumeKg"])
        if day in result or not math.isfinite(price) or price <= 0 or not math.isfinite(volume) or volume <= 0:
            raise ValueError(f"invalid or duplicate observed row for {product.get('id')}")
        result[day] = (price, volume)
    return dict(sorted(result.items()))


def _seasonal(series: dict[dt.date, tuple[float, float]], target: dt.date,
              origin: dt.date | None = None) -> float | None:
    days, cache = _index(series)
    anchor_year = (origin or target).year
    cache_key = (target, anchor_year)
    if cache_key in cache:
        return cache[cache_key]
    vals = []
    for year in range(anchor_year - 3, anchor_year):
        try:
            center = dt.date(year, target.month, target.day)
        except ValueError:  # Feb 29 is matched around Feb 28 in non-leap years.
            center = dt.date(year, 2, 28)
        lo = max(center - dt.timedelta(days=14), dt.date(year, 1, 1))
        hi = min(center + dt.timedelta(days=14), dt.date(year, 12, 31))
        a, b = bisect_left(days, lo), bisect_right(days, hi)
        year_vals = [series[day][0] for day in days[a:b]]
        if not year_vals:
            cache[cache_key] = None
            return None
        vals.extend(year_vals)
    if not vals:
        cache[cache_key] = None
        return None
    vals.sort()
    mid = len(vals) // 2
    result = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2
    cache[cache_key] = result
    return result


def _feature(series: dict[dt.date, tuple[float, float]], origin: dt.date,
             target: dt.date, horizon: int) -> tuple[tuple[float, ...], dt.date] | None:
    all_days, _ = _index(series)
    hi = bisect_left(all_days, origin)
    if hi == 0:
        return None
    src = all_days[hi - 1]
    if (origin - src).days > dashboard.MAX_SOURCE_AGE_DAYS:
        return None
    p0 = series[src][0]
    observations = {}
    for window in (7, 28):
        eligible_lo = bisect_left(all_days, origin - dt.timedelta(days=window), 0, hi)
        eligible = all_days[eligible_lo:hi]
        if not eligible:
            return None
        # Change compares the latest pre-origin price with latest at/before origin-window.
        ref_idx = bisect_left(all_days, origin - dt.timedelta(days=window), 0, hi) - 1
        ref = all_days[ref_idx] if ref_idx >= 0 else None
        if ref is None or (origin - dt.timedelta(days=window) - ref).days > 14:
            return None
        observations[window] = (eligible, ref)
    seasonal = _seasonal(series, target, origin)
    if seasonal is None or seasonal <= 0:
        return None
    price7 = math.log(p0 / series[observations[7][1]][0])
    price28 = math.log(p0 / series[observations[28][1]][0])
    volmeans = {}
    coverages = {}
    for window in (7, 28):
        eligible = observations[window][0]
        positive = [series[d][1] for d in eligible if series[d][1] > 0]
        if not positive:
            return None
        volmeans[window] = sum(positive) / len(eligible)
        coverages[window] = len(eligible) / window
    if volmeans[7] <= 0 or volmeans[28] <= 0:
        return None
    pressure = math.log1p(volmeans[7]) - math.log1p(volmeans[28])
    seasonal_gap = math.log(seasonal / p0)
    weekday = 2 * math.pi * target.weekday() / 7
    year_days = 366 if modeling._is_leap(target.year) else 365
    year_angle = 2 * math.pi * (target.timetuple().tm_yday - 1) / year_days
    row = (float(horizon), math.sin(weekday), math.cos(weekday), math.sin(year_angle), math.cos(year_angle),
           price7, price28, pressure, seasonal_gap, horizon * pressure, horizon * seasonal_gap,
           coverages[7], coverages[28])
    if not all(math.isfinite(x) for x in row):
        return None
    return row, src


def _fit_direct_ridge(rows):
    """Fit the per-horizon comparator only with 90 distinct target days."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len({target for target, _, _ in rows}) < modeling.MIN_TRAIN_TARGET_DAYS:
        raise ValueError("fewer than 90 distinct direct-Ridge training target days")
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit([x for _, x, _ in rows], [y for _, _, y in rows])
    return model


def _write_immutable(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def validate_snapshot(prices_bytes: bytes, status_bytes: bytes, issue: dt.date, cutoff: dt.date):
    prices = json.loads(prices_bytes)
    status = json.loads(status_bytes)
    if cutoff >= issue:
        raise ValueError("cutoff must be strictly before issue date")
    if prices.get("schemaVersion") != 1 or status.get("status") != "complete":
        raise ValueError("prices snapshot must be schemaVersion 1 from a complete run")
    if status.get("cutoff") != cutoff.isoformat() or prices.get("meta", {}).get("contextEnd") != cutoff.isoformat():
        raise ValueError("prices/run-status cutoff mismatch")
    if status.get("collectionStatus", {}).get("status") != "complete":
        raise ValueError("collection status is not complete")
    finished_at = _parse_dt(status.get("finishedAtKst", ""))
    if finished_at.astimezone(KST).date() >= issue:
        raise ValueError("daily run completed on or after issue date")
    # Daily status paths are project-relative; bind the claimed run to the supplied file.
    declared_run = Path(status.get("runDirectory", ""))
    if not declared_run.is_absolute():
        declared_run = Path.cwd() / declared_run
    if not declared_run.resolve().is_dir():
        raise ValueError("run-status runDirectory is unavailable")
    if _parse_dt(prices["generatedAt"]).astimezone(KST).date() >= issue:
        raise ValueError("prices snapshot was generated on or after issue date")
    products = {p.get("id"): p for p in prices.get("products", []) if isinstance(p, dict)}
    if set(products) != set(dashboard.PRODUCTS):
        raise ValueError("snapshot must contain exactly the five target products")
    for product in products.values():
        for row in product.get("observations", []):
            day = dt.date.fromisoformat(row["date"])
            if day > cutoff:
                raise ValueError("observation exceeds cutoff")
            if (not math.isfinite(float(row["actual"])) or float(row["actual"]) <= 0
                    or not math.isfinite(float(row["volumeKg"])) or float(row["volumeKg"]) <= 0):
                raise ValueError("snapshot contains invalid price or volume")
    return prices, products


def build(prices_path: Path, status_path: Path, issue: dt.date, cutoff: dt.date, output: Path) -> dict[str, Any]:
    prices_bytes, status_bytes = prices_path.read_bytes(), status_path.read_bytes()
    prices, products = validate_snapshot(prices_bytes, status_bytes, issue, cutoff)
    status = json.loads(status_bytes)
    run_dir = Path(status["runDirectory"])
    if not run_dir.is_absolute():
        run_dir = Path.cwd() / run_dir
    if prices_path.resolve().parent != run_dir.resolve() or status_path.resolve().parent != run_dir.resolve():
        raise ValueError("prices and run-status must be from the declared same run directory")
    provenance_path = Path(status["selectedProvenance"])
    if not provenance_path.is_absolute():
        provenance_path = Path.cwd() / provenance_path
    provenance = json.loads(provenance_path.read_bytes())
    if not isinstance(provenance, list):
        raise ValueError("selected provenance is invalid")
    seen_scopes = set()
    response_times = []
    finished_at = _parse_dt(status["finishedAtKst"])
    for row in provenance:
        if not isinstance(row, dict):
            raise ValueError("selected provenance row is invalid")
        day = dt.date.fromisoformat(row.get("date", ""))
        if day > cutoff:
            raise ValueError("selected provenance exceeds cutoff")
        response_at = _parse_dt(row.get("responseAt", ""))
        if response_at.astimezone(KST).date() >= issue:
            raise ValueError("selected source response occurred on or after issue date")
        if response_at > finished_at:
            raise ValueError("selected source response occurred after run completion")
        response_times.append(response_at)
        scope = (row.get("date"), row.get("market"), row.get("product"))
        if None in scope or scope in seen_scopes:
            raise ValueError("selected provenance has missing or duplicate scope")
        seen_scopes.add(scope)
    from . import daily_training
    expected_scopes = {
        (day.isoformat(), market, product)
        for day in (daily_training.CONTEXT_START + dt.timedelta(days=i)
                    for i in range((cutoff - daily_training.CONTEXT_START).days + 1))
        for market, product in daily_training.EXPECTED_SCOPES
    }
    if seen_scopes != expected_scopes:
        raise ValueError("selected provenance does not cover every date and market/product scope through cutoff")
    if daily_training._pipeline_fingerprint(prices, provenance) != status.get("snapshotSha256"):
        raise ValueError("run-status fingerprint does not match prices and selected provenance")
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    results = []
    for pid in dashboard.PRODUCTS:
        series = _series(products[pid])
        series_days = tuple(series)
        series_logs = tuple(math.log(series[d][0]) for d in series_days)
        latest = max((d for d in series if d <= cutoff), default=None)
        item = {"productId": pid, "trainingTargetDays": 0, "trainingSampleCount": 0,
                "trainingEndTargetDate": None,
                "sourceDate": latest.isoformat() if latest else None, "predictions": []}
        if latest is None or (issue - latest).days > 14:
            item["reason"] = "no valid current price within 14 days"
            item["predictions"] = [{"horizon": h, "targetDate": (issue + dt.timedelta(days=h)).isoformat(),
                                    "status": "deferred", "reason": item["reason"],
                                    "persistenceStatus": "deferred", "directRidgeStatus": "deferred",
                                    "directRidgeReason": "no fresh strict source feature"} for h in HORIZONS]
            results.append(item); continue
        samples = []
        direct_samples: dict[int, list[tuple[dt.date, tuple[float, ...], float]]] = {h: [] for h in HORIZONS}
        targets = set()
        for target, (target_price, _) in series.items():
            if target >= issue:
                continue
            for h in HORIZONS:
                origin = target - dt.timedelta(days=h)
                strict = live_forecast._strict_feature_row(series, origin, target, h, series_days, series_logs)
                if strict is not None:
                    source_day, direct_x = strict
                    direct_samples[h].append((target, direct_x, math.log(target_price / series[source_day][0])))
                feat = _feature(series, origin, target, h)
                if feat is not None:
                    x, source = feat
                    label = math.log(target_price / series[source][0])
                    samples.append((target, h, x, label))
                    targets.add(target)
        item["trainingTargetDays"], item["trainingSampleCount"] = len(targets), len(samples)
        item["trainingEndTargetDate"] = max(targets).isoformat() if targets else None
        predictions = []
        pooled = make_pipeline(StandardScaler(), Ridge(alpha=10.0)) if len(targets) >= modeling.MIN_TRAIN_TARGET_DAYS else None
        fit_error = None
        if pooled is not None:
            try:
                pooled.fit([x for _, _, x, _ in samples], [y for _, _, _, y in samples])
            except Exception as exc:
                pooled = None
                fit_error = f"pooled model fit failed: {type(exc).__name__}"
        if pooled is not None:
            estimator = pooled[-1]
            item["coefficients"] = {"featureNames": list(FEATURES), "means": pooled[0].mean_.tolist(),
                                    "scales": pooled[0].scale_.tolist(), "ridge": estimator.coef_.tolist(),
                                    "intercept": float(estimator.intercept_)}
        for h in HORIZONS:
            target = issue + dt.timedelta(days=h)
            feat = _feature(series, issue, target, h)
            pred = {"horizon": h, "targetDate": target.isoformat(), "status": "deferred"}
            pred["persistencePrice"] = series[latest][0]
            strict_live = live_forecast._strict_feature_row(series, issue, target, h, series_days, series_logs)
            if strict_live is not None:
                pred["directRidgeStatus"] = "deferred"
                try:
                    direct = _fit_direct_ridge(direct_samples[h])
                    direct_log = float(direct.predict([strict_live[1]])[0])
                    direct_price = series[strict_live[0]][0] * math.exp(direct_log)
                    if not math.isfinite(direct_price) or direct_price <= 0:
                        raise ValueError("direct-Ridge prediction is non-finite or non-positive")
                    pred.update(directRidgeStatus="forecast", directRidgePrice=direct_price,
                                directSourceDate=strict_live[0].isoformat())
                except Exception as exc:
                    pred["directRidgeReason"] = f"direct Ridge unavailable: {type(exc).__name__}"
            else:
                pred.update(directRidgeStatus="deferred", directRidgeReason="no fresh strict source feature")
            if len(targets) < modeling.MIN_TRAIN_TARGET_DAYS:
                pred["reason"] = "fewer than 90 distinct training target days"
            elif fit_error:
                pred["reason"] = fit_error
            elif feat is None:
                pred["reason"] = "required current feature or seasonal baseline unavailable"
            else:
                try:
                    x, src = feat
                    cand_log = float(pooled.predict([x])[0])
                    cand_price = series[src][0] * math.exp(cand_log)
                    if not math.isfinite(cand_price) or cand_price <= 0:
                        pred["reason"] = "prediction is non-finite or non-positive"
                    else:
                        pred.update(status="forecast", sourceDate=src.isoformat(), candidatePrice=cand_price)
                except Exception as exc:
                    pred["reason"] = f"prediction failed: {type(exc).__name__}"
            predictions.append(pred)
        item["predictions"] = predictions
        if item.get("coefficients") is None:
            item["coefficients"] = None
        results.append(item)
    payload = {"schemaVersion": 1, "experiment": "supply_pressure_seasonal_error_prospective",
               "issueDate": issue.isoformat(), "cutoff": cutoff.isoformat(),
               "createdAtKst": dt.datetime.now(KST).isoformat(timespec="seconds"),
                "input": {"pricesPath": str(prices_path.resolve()), "pricesSha256": _sha(prices_bytes),
                          "runStatusPath": str(status_path.resolve()), "runStatusSha256": _sha(status_bytes),
                         "snapshotGeneratedAt": prices["generatedAt"],
                         "runFinishedAtKst": status["finishedAtKst"],
                         "runFingerprint": status["snapshotSha256"],
                         "selectedProvenancePath": str(provenance_path.resolve()),
                         "selectedProvenanceSha256": _sha(provenance_path.read_bytes()),
                         "selectedResponseAtMin": min(response_times).isoformat(),
                         "selectedResponseAtMax": max(response_times).isoformat()},
               "codeSha256": _sha(Path(__file__).read_bytes()), "featureNames": FEATURES,
               "model": {"type": "StandardScaler+Ridge", "alpha": 10, "fit": "one pooled model per product, horizons 1-10"},
               "products": results}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    _write_immutable(output, raw)
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prices", type=Path, required=True)
    p.add_argument("--run-status", type=Path, required=True)
    p.add_argument("--issue-date", required=True)
    p.add_argument("--cutoff", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    issue, cutoff = dt.date.fromisoformat(a.issue_date), dt.date.fromisoformat(a.cutoff)
    if issue != dt.datetime.now(KST).date():
        raise SystemExit("issue date must be the current KST date")
    build(a.prices, a.run_status, issue, cutoff, a.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
