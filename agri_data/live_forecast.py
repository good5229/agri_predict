"""Issue an immutable, current-day Ridge forecast from yesterday's verified data."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import uuid
from typing import Any, Callable
from zoneinfo import ZoneInfo

from . import coverage, dashboard, daily_training, modeling, volume_guard

KST = ZoneInfo("Asia/Seoul")


class LiveForecastError(RuntimeError):
    pass


def _strict_feature_row(series: dict[dt.date, tuple[float, float]], origin: dt.date,
                        target: dt.date, horizon: int,
                        days: tuple[dt.date, ...], log_prices: tuple[float, ...]):
    """Modeling feature row with the eligible prefix ending strictly before origin."""
    from .modeling import LAGS, WINDOWS

    hi = bisect_left(days, origin)
    source_idx = hi - 1
    if source_idx < 0:
        return None
    source_day = days[source_idx]
    source_age = (origin - source_day).days
    if source_age > dashboard.MAX_SOURCE_AGE_DAYS:
        return None
    values: list[float] = [modeling.math.log(series[source_day][0]), float(horizon), float(source_age)]
    for lag in LAGS:
        lag_cutoff = origin - dt.timedelta(days=lag)
        lag_idx = bisect_right(days, lag_cutoff, 0, hi) - 1
        if lag_idx < 0:
            values.extend((0.0, float(lag + 1), 0.0))
        else:
            lag_day = days[lag_idx]
            values.extend((modeling.math.log(series[lag_day][0]),
                           float((lag_cutoff - lag_day).days), 1.0))
    for window in WINDOWS:
        first_day = origin - dt.timedelta(days=window - 1)
        lo = bisect_left(days, first_day, 0, hi)
        selected_logs = log_prices[lo:hi]
        selected_volumes = [series[day][1] for day in days[lo:hi]]
        count = len(selected_logs)
        if count:
            mean = sum(selected_logs) / count
            variance = sum((value - mean) ** 2 for value in selected_logs) / count
            std = modeling.math.sqrt(max(0.0, variance))
            minimum, maximum = min(selected_logs), max(selected_logs)
            volume_log = modeling.math.log1p(max(0.0, sum(selected_volumes)))
        else:
            mean = std = minimum = maximum = volume_log = 0.0
        values.extend((mean, std, minimum, maximum, float(count), count / window, volume_log))
    weekday_angle = 2 * modeling.math.pi * target.weekday() / 7
    year_days = 366 if modeling._is_leap(target.year) else 365
    year_angle = 2 * modeling.math.pi * (target.timetuple().tm_yday - 1) / year_days
    values.extend((modeling.math.sin(weekday_angle), modeling.math.cos(weekday_angle),
                   modeling.math.sin(year_angle), modeling.math.cos(year_angle)))
    if not all(modeling.math.isfinite(value) for value in values):
        raise ValueError("feature generation produced a non-finite value")
    return source_day, tuple(values)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _atomic_replace(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _strict_samples(series: dict[dt.date, tuple[float, float]], cutoff: dt.date):
    """Build Ridge samples with all feature observations strictly before each origin."""
    tasks: dict[dt.date, list[tuple[dt.date, int, float]]] = {}
    ordered_tasks: list[tuple[dt.date, int, float]] = []
    for target, (target_price, _) in series.items():
        if target > cutoff:
            continue
        for horizon in modeling.HORIZONS:
            origin = target - dt.timedelta(days=horizon)
            tasks.setdefault(origin, []).append((target, horizon, target_price))
            ordered_tasks.append((target, horizon, target_price))

    # Reuse indexed price data and bound every lookup at the true historical origin.
    # This is equivalent to rebuilding a prefix dict for each origin, without that work.
    days = tuple(series)
    log_prices = tuple(modeling.math.log(series[day][0]) for day in days)
    features: dict[tuple[dt.date, int], tuple[tuple[float, ...], float] | None] = {}
    for origin in sorted(tasks):
        for target, horizon, target_price in tasks[origin]:
            feature = _strict_feature_row(series, origin, target, horizon, days, log_prices)
            if feature is None:
                features[(target, horizon)] = None
                continue
            source_day, vector = feature
            features[(target, horizon)] = (
                vector, modeling.math.log(target_price / series[source_day][0]))

    samples = []
    target_days: set[dt.date] = set()
    for target, horizon, _ in ordered_tasks:
        feature = features[(target, horizon)]
        if feature is None:
            continue
        vector, label = feature
        samples.append((target, vector, label))
        target_days.add(target)
    return samples, target_days


def _forecast_input_series(product: dict[str, Any], cutoff: dt.date
                           ) -> tuple[dict[dt.date, tuple[float, float]],
                                      dict[dt.date, tuple[float, float]], dict[str, Any]]:
    """Return raw as-of-cutoff and model-eligible series with quality provenance."""
    all_observations = modeling._observed_series(product)
    raw_as_of = {day: value for day, value in all_observations.items() if day <= cutoff}
    later_days = sorted(day for day in all_observations if day > cutoff)
    enabled = product.get("id") == "radish"
    flags = volume_guard.quarantine_dates(raw_as_of, price_ratio_threshold=4.0) if enabled else []
    eligible = volume_guard.quality_series(raw_as_of, flags) if enabled else dict(raw_as_of)
    raw_latest = max(raw_as_of, default=None)
    eligible_latest = max(eligible, default=None)
    metadata = {
        "enabled": enabled,
        "ruleVersion": "radish-low-volume-extreme-price-v1" if enabled else None,
        "rule": ("radish: at least 14 prior observed days; prior 28 observed-day net-volume median; "
                 "volume <1% of median AND price >=4x or <=0.25x prior 28 observed-day median"
                 if enabled else None),
        "priceRatioThreshold": 4.0 if enabled else None,
        "sourceCutoffDate": cutoff.isoformat(),
        "rawObservationCountThroughCutoff": len(raw_as_of),
        "observationsAfterCutoffIgnoredCount": len(later_days),
        "latestObservationAfterCutoffIgnored": later_days[-1].isoformat() if later_days else None,
        "lastRawObservationDate": raw_latest.isoformat() if raw_latest else None,
        "lastRawObservedPrice": raw_as_of[raw_latest][0] if raw_latest else None,
        "lastEligibleSourceDate": eligible_latest.isoformat() if eligible_latest else None,
        "lastEligiblePrice": eligible[eligible_latest][0] if eligible_latest else None,
        "flaggedDateCount": len(flags),
        "flaggedDates": flags,
    }
    return raw_as_of, eligible, metadata


def _coverage_states(db: sqlite3.Connection, day: dt.date) -> list[dict[str, Any]]:
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for row in daily_training._coverage_query(db, day, day):
        try:
            query = json.loads(row["query_json"])
        except (json.JSONDecodeError, TypeError) as exc:
            raise LiveForecastError("malformed price-query scope in coverage manifest") from exc
        dims = coverage.price_dimensions(query, row["dataset"])
        if not dims:
            continue
        _, market, product, source, scope = dims
        if source != "origin" or scope != "all_subclasses" or (market, product) not in daily_training.EXPECTED_SCOPES:
            continue
        key = (market, product)
        if key in found:
            raise LiveForecastError(f"duplicate selected broad query scope on {day.isoformat()}")
        selected = row["selected_state"]
        found[key] = {"date": day.isoformat(), "market": market, "product": product,
                      "selectedState": selected, "latestState": row["latest_state"],
                      "selectedOutcome": "validated_empty_response" if selected == "empty" else
                          "validated_nonempty_response" if selected == "complete" else "no_valid_selected_response",
                      "selectedJobId": row["selected_job_id"], "selectedEpochId": row["selected_epoch_id"],
                      "responseAt": row["selected_response_at"], "rowCount": row["selected_row_count"],
                      "selectedPrevious": bool(row["selected_previous"])}
    for market, product in sorted(daily_training.EXPECTED_SCOPES):
        found.setdefault((market, product), {"date": day.isoformat(), "market": market, "product": product,
                          "selectedState": None, "latestState": "unavailable",
                          "selectedOutcome": "no_manifest_scope", "selectedJobId": None,
                          "selectedEpochId": None, "responseAt": None, "rowCount": None,
                          "selectedPrevious": False})
    return [found[key] for key in sorted(found)]


def _forecast_product(product: dict[str, Any], issue_date: dt.date, cutoff: dt.date) -> dict[str, Any]:
    _, series, quality_meta = _forecast_input_series(product, cutoff)
    samples, target_days = _strict_samples(series, cutoff)
    latest_day = max(series, default=None)
    source_age = (issue_date - latest_day).days if latest_day else None
    fresh = latest_day is not None and 0 <= source_age <= dashboard.MAX_SOURCE_AGE_DAYS
    result: dict[str, Any] = {"productId": product["id"], "status": "unavailable_no_fresh_price",
                              "sourceDate": latest_day.isoformat() if latest_day else None,
                              "lastObservedPriceDate": latest_day.isoformat() if latest_day else None,
                              "lastObservedPrice": series[latest_day][0] if latest_day else None,
                              "qualityGuard": quality_meta,
                              "sourceAgeDays": source_age,
                              "freshnessState": "fresh" if fresh else "stale" if latest_day else "no_observation",
                              "trainingTargetDays": len(target_days), "trainingSampleCount": len(samples),
                              "fitFeaturePolicy": "price_and_volume_observations_strictly_before_historical_origin",
                              "labelCutoff": cutoff.isoformat(), "predictions": []}
    if not fresh:
        result["reason"] = "no valid observed price within the 14-day source-age limit"
        return result
    feature_cache: dict[Any, Any] = {}
    feature_origin = issue_date
    feature = modeling._feature_row(series, feature_origin, issue_date + dt.timedelta(days=1), 1,
                                    feature_cache)
    source_day = max(series)
    used_ridge = len(target_days) >= modeling.MIN_TRAIN_TARGET_DAYS and bool(samples) and feature is not None
    if used_ridge:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError as exc:
            raise LiveForecastError("threadpoolctl is required for bounded Ridge inference") from exc
        estimator = modeling._new_estimator("ridge")
        matrix = [row[1] for row in samples]
        labels = [row[2] for row in samples]
        with threadpool_limits(limits=1):
            estimator.fit(matrix, labels)
            method = "ridge"
            for horizon in modeling.HORIZONS:
                target = issue_date + dt.timedelta(days=horizon)
                current = modeling._feature_row(series, feature_origin, target, horizon,
                                                feature_cache)
                if current is None:
                    continue
                source_day, vector = current
                ratio = float(estimator.predict([vector])[0])
                predicted, clipped = modeling._inverse_log_ratio(series[source_day][0], ratio)
                result["predictions"].append({"target": target.isoformat(), "horizonDays": horizon,
                                              "predictedPrice": predicted, "sourceDate": source_day.isoformat(),
                                              "clippedLogPrice": clipped})
    else:
        method = "persistence_fallback"
        for horizon in modeling.HORIZONS:
            result["predictions"].append({"target": (issue_date + dt.timedelta(days=horizon)).isoformat(),
                                          "horizonDays": horizon, "predictedPrice": series[source_day][0],
                                          "sourceDate": source_day.isoformat(), "clippedLogPrice": False})
    result["method"] = method
    result["status"] = "available" if result["predictions"] else "unavailable_no_valid_feature"
    if method == "persistence_fallback":
        result["reason"] = (f"fewer than {modeling.MIN_TRAIN_TARGET_DAYS} distinct eligible training target days "
                            "or no valid current Ridge feature")
    return result


def _as_kst(value: dt.datetime, label: str) -> dt.datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise LiveForecastError(f"{label} must include a timezone")
    return value.astimezone(KST)


def _validate_publication_window(ready_at: dt.datetime, issue_date: dt.date) -> None:
    if ready_at.date() != issue_date:
        raise LiveForecastError("KST date rolled over before live forecast publication")
    if ready_at.time() < dt.time(10, 0):
        raise LiveForecastError("live forecast publication is allowed from 10:00 KST onward")


def issue_live_forecast(*, root: str | Path = ".", db_path: str | Path = "data/agri.sqlite",
                        today: dt.date | None = None,
                        issue_time: dt.datetime | None = None,
                        clock: Callable[[], dt.datetime] | None = None) -> dict[str, Any]:
    project = Path(root).resolve()
    database = Path(db_path)
    if not database.is_absolute():
        database = project / database
    database = database.resolve()
    now = clock or (lambda: dt.datetime.now(KST))
    started_at = _as_kst(issue_time or now(), "start timestamp")
    issue_date = today or started_at.date()
    if issue_date != started_at.date():
        raise LiveForecastError("issue date must match the KST issue timestamp")
    if started_at.time() < dt.time(10, 0):
        raise LiveForecastError("live forecast publication is allowed from 10:00 KST onward")
    if not database.is_file():
        raise LiveForecastError("collector database does not exist")
    cutoff = issue_date - dt.timedelta(days=1)

    lock_path = project / "runs" / "daily-collection.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise LiveForecastError("collection is active; live forecast snapshot deferred") from exc
        db = coverage.open_readonly(str(database))
        try:
            db.execute("BEGIN")
            days, provenance = daily_training._coverage_days(db, daily_training.CONTEXT_START, cutoff)
            contiguous, first_missing = daily_training.contiguous_cutoff(
                days, daily_training.CONTEXT_START, cutoff)
            if contiguous is None:
                raise LiveForecastError(f"no contiguous verified price scope; first gap={first_missing}")
            snapshot = dashboard.build_dashboard(
                db, str(database), generated_at=started_at.isoformat(timespec="seconds"),
                context_start=daily_training.CONTEXT_START, context_end=contiguous,
                evaluation_start=daily_training.EVALUATION_START, evaluation_end=contiguous)
            scope_states = _coverage_states(db, cutoff)
            selected_provenance = [row for row in provenance if row["date"] <= contiguous.isoformat()]
            snapshot_at = _as_kst(now(), "snapshot timestamp")
        finally:
            db.rollback()
            db.close()
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    products = [_forecast_product(product, issue_date, contiguous) for product in snapshot["products"]]
    ready_at = _as_kst(now(), "publication timestamp")
    _validate_publication_window(ready_at, issue_date)
    source_fingerprint = _canonical_sha256({"snapshot": snapshot, "selectedProvenance": selected_provenance})
    artifact = {
        "schemaVersion": 1,
        "startedAtKst": started_at.isoformat(timespec="seconds"),
        "snapshotCapturedAtKst": snapshot_at.isoformat(timespec="seconds"),
        "issueAtKst": ready_at.isoformat(timespec="seconds"),
        "issueDateKst": issue_date.isoformat(),
        "sourceCutoffDate": contiguous.isoformat(),
        "requestedSourceCapDate": cutoff.isoformat(),
        "firstMissingDateAfterContiguousPrefix": first_missing,
        "sourceSnapshotSha256": source_fingerprint,
        "selectedQueryCount": len(selected_provenance),
        "selectedQueryProvenance": {
            "path": None,
            "sha256": _canonical_sha256(selected_provenance),
            "snapshotSha256": source_fingerprint,
        },
        "cutoffDayQueryStates": scope_states,
        "dataVintage": "latest_verified_response; first_publication_time_unknown",
        "sourceObservationsAreRetrospective": True,
        "emptyQueryInterpretation": "a validated zero-row response means no matching trades were returned; it does not prove market closure",
        "targetRule": "all training target dates are on or before source cutoff and strictly before issue date",
        "featureRule": "training price and volume observations must be strictly before historical origin; live inputs end at source cutoff",
        "products": products,
    }
    run_id = ready_at.strftime("%Y%m%dT%H%M%S%z") + "-" + uuid.uuid4().hex[:8]
    output = project / "outputs" / "live-forecasts" / issue_date.isoformat() / f"{run_id}.json"
    provenance_output = output.with_name(output.stem + "-selected-provenance.json")
    artifact["selectedQueryProvenance"]["path"] = str(provenance_output.relative_to(project))
    _write_immutable(provenance_output, json.dumps(selected_provenance, ensure_ascii=False,
                                                   sort_keys=True, indent=2, allow_nan=False) + "\n")
    artifact_payload = json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    _write_immutable(output, artifact_payload)
    artifact_hash = hashlib.sha256(artifact_payload.encode("utf-8")).hexdigest()
    _validate_publication_window(_as_kst(now(), "publication timestamp"), issue_date)
    pointer = {"schemaVersion": 1, "runId": run_id, "issueAtKst": ready_at.isoformat(timespec="seconds"),
               "artifact": str(output.relative_to(project / "outputs" / "live-forecasts")),
               "artifactSha256": artifact_hash,
               "sourceCutoffDate": contiguous.isoformat(), "sourceSnapshotSha256": source_fingerprint}
    _atomic_replace(project / "outputs" / "live-forecasts" / "latest.json",
                    json.dumps(pointer, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return {"status": "complete", "artifact": str(output.relative_to(project)),
            "issueDateKst": issue_date.isoformat(), "sourceCutoffDate": contiguous.isoformat(),
            "sourceSnapshotSha256": source_fingerprint, "artifactSha256": artifact_hash,
            "availableProducts": sum(product["status"] == "available" for product in products),
            "products": [{"productId": p["productId"], "status": p["status"], "method": p.get("method")}
                         for p in products]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--db", default="data/agri.sqlite")
    args = parser.parse_args(argv)
    try:
        result = issue_live_forecast(root=args.root, db_path=args.db)
    except (LiveForecastError, OSError, sqlite3.Error, ValueError, RuntimeError) as exc:
        print(f"live forecast failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
