"""Leakage-aware walk-forward training for dashboard schema v1 price exports."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
import uuid
from bisect import bisect_left, bisect_right
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from .dashboard import HORIZONS, MAX_SOURCE_AGE_DAYS, write_json_atomic


MIN_TRAIN_TARGET_DAYS = 90
EVALUATION_START = dt.date(2015, 1, 1)
EVALUATION_END = dt.date(2015, 12, 31)
FIRST_ORIGIN = EVALUATION_START - dt.timedelta(days=max(HORIZONS))
LAST_ORIGIN = EVALUATION_END - dt.timedelta(days=1)
CADENCES = ("daily", "weekly", "monthly", "seasonal")
MODEL_IDS = ("ridge", "hist_gb")
WINDOWS = (7, 28, 90)
LAGS = (1, 7, 28, 90)

FEATURE_NAMES = ["lastPriceLog", "horizonDays", "sourceAgeDays"]
for _lag in LAGS:
    FEATURE_NAMES.extend((f"lag{_lag}PriceLog", f"lag{_lag}AgeDays", f"lag{_lag}Available"))
for _window in WINDOWS:
    FEATURE_NAMES.extend((f"window{_window}MeanLogPrice", f"window{_window}StdLogPrice",
                          f"window{_window}MinLogPrice", f"window{_window}MaxLogPrice",
                          f"window{_window}ObservedDays", f"window{_window}Coverage",
                          f"window{_window}LogVolumeKg"))
FEATURE_NAMES.extend(("targetWeekdaySin", "targetWeekdayCos", "targetYearDaySin",
                      "targetYearDayCos"))

MODEL_NAMES = {
    "baseline": "직전 관측가격 지속성 기준선",
    "ridge": "Ridge 회귀",
    "hist_gb": "히스토그램 기반 그래디언트 부스팅 회귀",
}


def _parse_day(value: Any) -> dt.date | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return None


def _finite_positive(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def _observed_series(product: dict[str, Any]) -> dict[dt.date, tuple[float, float]]:
    result: dict[dt.date, tuple[float, float]] = {}
    for row in product.get("observations", []):
        day = _parse_day(row.get("date"))
        price = _finite_positive(row.get("actual"))
        volume = _finite_positive(row.get("volumeKg"))
        if day is None or price is None or volume is None:
            raise ValueError(f"invalid observed price row for product {product.get('id')!r}")
        if day in result:
            raise ValueError(f"duplicate observation date {day.isoformat()} for product {product.get('id')!r}")
        result[day] = (price, volume)
    return dict(sorted(result.items()))


def _feature_row(
    series: dict[dt.date, tuple[float, float]],
    origin: dt.date,
    target: dt.date,
    horizon: int,
    cache: dict[Any, Any],
) -> tuple[dt.date, tuple[float, ...]] | None:
    key = (origin, target, horizon)
    if key in cache:
        return cache[key]
    days = cache.setdefault(("series_days",), tuple(series))
    log_prices = cache.setdefault(("series_log_prices",), tuple(math.log(series[day][0]) for day in days))
    source_idx = bisect_right(days, origin) - 1
    if source_idx < 0:
        cache[key] = None
        return None
    source_day = days[source_idx]
    source_age = (origin - source_day).days
    if source_age < 0 or source_age > MAX_SOURCE_AGE_DAYS:
        cache[key] = None
        return None
    source_price = series[source_day][0]
    values: list[float] = [math.log(source_price), float(horizon), float(source_age)]

    for lag in LAGS:
        lag_cutoff = origin - dt.timedelta(days=lag)
        lag_idx = bisect_right(days, lag_cutoff) - 1
        if lag_idx < 0:
            values.extend((0.0, float(lag + 1), 0.0))
        else:
            lag_day = days[lag_idx]
            values.extend((math.log(series[lag_day][0]), float((lag_cutoff - lag_day).days), 1.0))

    for window in WINDOWS:
        first_day = origin - dt.timedelta(days=window - 1)
        lo = bisect_left(days, first_day)
        hi = bisect_right(days, origin)
        selected_logs = log_prices[lo:hi]
        selected_volumes = [series[day][1] for day in days[lo:hi]]
        count = len(selected_logs)
        if count:
            mean = sum(selected_logs) / count
            variance = sum((value - mean) ** 2 for value in selected_logs) / count
            std = math.sqrt(max(0.0, variance))
            minimum, maximum = min(selected_logs), max(selected_logs)
            volume_log = math.log1p(max(0.0, sum(selected_volumes)))
        else:
            mean = std = minimum = maximum = volume_log = 0.0
        values.extend((mean, std, minimum, maximum, float(count), count / window, volume_log))

    weekday_angle = 2 * math.pi * target.weekday() / 7
    year_days = 366 if _is_leap(target.year) else 365
    year_angle = 2 * math.pi * (target.timetuple().tm_yday - 1) / year_days
    values.extend((math.sin(weekday_angle), math.cos(weekday_angle),
                   math.sin(year_angle), math.cos(year_angle)))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("feature generation produced a non-finite value")
    result = (source_day, tuple(values))
    cache[key] = result
    return result


def _is_leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _supervised_samples(
    series: dict[dt.date, tuple[float, float]],
    cutoff: dt.date,
    cache: dict[Any, Any] | None = None,
) -> tuple[list[tuple[dt.date, dt.date, int, tuple[float, ...], float]], set[dt.date]]:
    feature_cache = cache if cache is not None else {}
    samples: list[tuple[dt.date, dt.date, int, tuple[float, ...], float]] = []
    target_days: set[dt.date] = set()
    for target, (target_price, _) in series.items():
        if target > cutoff:
            continue
        for horizon in HORIZONS:
            origin = target - dt.timedelta(days=horizon)
            if origin > cutoff:
                continue
            feature = _feature_row(series, origin, target, horizon, feature_cache)
            if feature is None:
                continue
            source_day, vector = feature
            source_price = series[source_day][0]
            samples.append((target, origin, horizon, vector,
                            math.log(target_price) - math.log(source_price)))
            target_days.add(target)
    return samples, target_days


def _cadence_key(cadence: str, origin: dt.date) -> dt.date:
    if cadence == "daily":
        return origin
    if cadence == "weekly":
        return origin - dt.timedelta(days=origin.weekday())
    if cadence == "monthly":
        return origin.replace(day=1)
    if cadence == "seasonal":
        if origin.month in (12, 1, 2):
            return dt.date(origin.year if origin.month == 12 else origin.year - 1, 12, 1)
        if origin.month in (3, 4, 5):
            return dt.date(origin.year, 3, 1)
        if origin.month in (6, 7, 8):
            return dt.date(origin.year, 6, 1)
        return dt.date(origin.year, 9, 1)
    raise ValueError(f"unsupported cadence: {cadence}")


def _new_estimator(model_id: str):
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("model dependencies are missing; install requirements-modeling.txt") from exc
    if model_id == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    if model_id == "hist_gb":
        return HistGradientBoostingRegressor(
            max_iter=40,
            max_leaf_nodes=15,
            learning_rate=0.08,
            l2_regularization=1.0,
            random_state=42,
            early_stopping=False,
        )
    raise ValueError(f"unknown model: {model_id}")


def _model_parameters(model_id: str) -> dict[str, Any]:
    if model_id == "ridge":
        return {"pipeline": ["StandardScaler", "Ridge"], "alpha": 10.0}
    return {"max_iter": 40, "max_leaf_nodes": 15, "learning_rate": 0.08,
            "l2_regularization": 1.0, "random_state": 42, "early_stopping": False}


def _inverse_log_ratio(last_price: float, predicted_log_ratio: float) -> tuple[float, bool]:
    log_price = math.log(last_price) + predicted_log_ratio
    bounded = min(700.0, max(-700.0, log_price))
    predicted = math.exp(bounded)
    if not math.isfinite(predicted) or predicted <= 0:
        raise ValueError("model generated a non-finite or non-positive price")
    return predicted, bounded != log_price


def _fit_cadence(
    series: dict[dt.date, tuple[float, float]],
    model_id: str,
    cadence: str,
    origin_start: dt.date,
    origin_end: dt.date,
    eval_actuals: dict[dt.date, float],
    sample_pool: list[tuple[dt.date, dt.date, int, tuple[float, ...], float]],
    feature_cache: dict[Any, Any],
    training_matrix: Any,
    training_labels: Any,
) -> tuple[dict[str, Any], Any | None]:
    runs_predictions: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    last_estimator = None
    last_cutoff: dt.date | None = None
    last_period: dt.date | None = None
    diagnostics: Counter = Counter()
    issued = matched = 0
    sample_targets = [sample[0] for sample in sample_pool]
    unique_targets = sorted(set(sample_targets))
    sample_cursor = 0
    origins: list[dt.date] = []
    day = origin_start
    while day <= origin_end:
        origins.append(day)
        day += dt.timedelta(days=1)

    try:
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise RuntimeError("threadpoolctl is required; install requirements-modeling.txt") from exc

    with threadpool_limits(limits=1):
        for origin in origins:
            period = _cadence_key(cadence, origin)
            while sample_cursor < len(sample_pool) and sample_pool[sample_cursor][0] <= origin:
                sample_cursor += 1
            if period != last_period:
                last_period = period
                train_samples = sample_pool[:sample_cursor]
                target_days = unique_targets[:bisect_right(unique_targets, origin)]
                if len(target_days) < MIN_TRAIN_TARGET_DAYS:
                    fits.append({
                        "cutoff": origin.isoformat(),
                        "trainStart": target_days[0].isoformat() if target_days else None,
                        "trainEnd": target_days[-1].isoformat() if target_days else None,
                        "samples": len(train_samples),
                        "targetDays": len(target_days),
                        "status": "insufficient_data",
                    })
                    diagnostics["insufficientFitAttempts"] += 1
                else:
                    estimator = _new_estimator(model_id)
                    estimator.fit(training_matrix[:sample_cursor], training_labels[:sample_cursor])
                    last_estimator = estimator
                    last_cutoff = origin
                    fits.append({
                        "cutoff": origin.isoformat(),
                        "trainStart": target_days[0].isoformat(),
                        "trainEnd": target_days[-1].isoformat(),
                        "samples": len(train_samples),
                        "targetDays": len(target_days),
                        "status": "fitted",
                    })
                    diagnostics["successfulFits"] += 1

            if last_estimator is None or last_cutoff is None:
                diagnostics["originsWithoutFittedModel"] += 1
                continue
            for horizon in HORIZONS:
                target = origin + dt.timedelta(days=horizon)
                feature = _feature_row(series, origin, target, horizon, feature_cache)
                if feature is None:
                    diagnostics["issuedSkippedNoFreshSource"] += 1
                    continue
                source_day, vector = feature
                prediction_log_ratio = float(last_estimator.predict([vector])[0])
                if not math.isfinite(prediction_log_ratio):
                    diagnostics["issuedSkippedNonfiniteModelOutput"] += 1
                    continue
                predicted, clipped = _inverse_log_ratio(series[source_day][0], prediction_log_ratio)
                if clipped:
                    diagnostics["clippedLogPricePredictions"] += 1
                issued += 1
                actual = eval_actuals.get(target)
                if EVALUATION_START <= target <= EVALUATION_END and actual is not None:
                    matched += 1
                    runs_predictions.append({
                        "origin": origin.isoformat(),
                        "target": target.isoformat(),
                        "horizon": horizon,
                        "predicted": predicted,
                        "sourceDate": source_day.isoformat(),
                        "fitCutoff": last_cutoff.isoformat(),
                    })
                else:
                    diagnostics["issuedWithoutEvaluationActual"] += 1

    runs_predictions.sort(key=lambda row: (row["target"], row["horizon"], row["origin"]))
    run_status = "available" if diagnostics["successfulFits"] and issued else "insufficient_data"
    reason = None
    if diagnostics["successfulFits"] == 0:
        reason = f"서로 다른 훈련 목표 날짜가 최소 {MIN_TRAIN_TARGET_DAYS}일에 미달합니다."
    elif issued == 0:
        reason = "유효한 원자료가 14일 이내에 없어 예측을 발행하지 못했습니다."
    run = {
        "modelId": model_id,
        "cadence": cadence,
        "status": run_status,
        **({"reason": reason} if reason else {}),
        "predictions": runs_predictions,
        "fits": fits,
        "features": list(FEATURE_NAMES),
        "diagnostics": {
            "issuedPredictionCount": issued,
            "matchedPredictionCount": matched,
            "issuedWithoutEvaluationActualCount": diagnostics["issuedWithoutEvaluationActual"],
            "successfulFitCount": diagnostics["successfulFits"],
            "insufficientFitAttemptCount": diagnostics["insufficientFitAttempts"],
            "originsWithoutFittedModel": diagnostics["originsWithoutFittedModel"],
            "issuedSkippedNoFreshSource": diagnostics["issuedSkippedNoFreshSource"],
            "issuedSkippedNonfiniteModelOutput": diagnostics["issuedSkippedNonfiniteModelOutput"],
            "clippedLogPricePredictions": diagnostics["clippedLogPricePredictions"],
            "lastFitCutoff": last_cutoff.isoformat() if last_cutoff else None,
        },
        "targetTransform": "ln(목표 실제가격 / 예측 원점의 마지막 관측가격)",
        "hyperparameters": _model_parameters(model_id),
    }
    return run, last_estimator


def _save_artifacts(
    artifacts_dir: str | Path,
    artifacts: list[dict[str, Any]],
    versions: dict[str, str],
    price_export_sha256: str,
) -> dict[str, Any]:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError("joblib is required to save model artifacts; install requirements-modeling.txt") from exc
    root = Path(artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".run-staging-", dir=root))
    run_name = "run-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    final_dir = root / run_name
    manifest_models: list[dict[str, Any]] = []
    try:
        for entry in artifacts:
            estimator = entry["estimator"]
            if estimator is None:
                continue
            name = f"{entry['modelId']}__{entry['productId']}__{entry['cadence']}.joblib"
            output = staging / name
            joblib.dump(estimator, output)
            digest = hashlib.sha256(output.read_bytes()).hexdigest()
            manifest_models.append({
                "modelId": entry["modelId"],
                "productId": entry["productId"],
                "cadence": entry["cadence"],
                "fitCutoff": entry["fitCutoff"],
                "file": name,
                "sha256": digest,
                "features": list(FEATURE_NAMES),
                "targetTransform": "ln(목표 실제가격 / 예측 원점의 마지막 관측가격)",
                "hyperparameters": _model_parameters(entry["modelId"]),
            })
        manifest = {"schemaVersion": 1, "environment": versions,
                    "priceExportSha256": price_export_sha256, "models": manifest_models}
        write_json_atomic(staging / "manifest.json", manifest)
        os.replace(staging, final_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"directory": str(final_dir), "modelArtifactCount": len(manifest_models),
            "manifest": str(final_dir / "manifest.json")}


def build_model_dashboard(
    source: dict[str, Any],
    artifacts_dir: str | Path | None = None,
    *,
    origin_start: dt.date = FIRST_ORIGIN,
    origin_end: dt.date = LAST_ORIGIN,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if source.get("schemaVersion") != 1 or not isinstance(source.get("products"), list):
        raise ValueError("input must be a dashboard schemaVersion 1 JSON price export")
    if origin_start > origin_end:
        raise ValueError("origin_start must not be after origin_end")
    products = source["products"]
    if {product.get("id") for product in products} != {"cabbage", "radish", "onion", "garlic", "pepper"}:
        raise ValueError("input must include exactly the five dashboard products")

    output = json.loads(json.dumps(source, ensure_ascii=False))
    source_sha256 = hashlib.sha256(
        json.dumps(source, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    output["schemaVersion"] = 2
    meta = output.setdefault("meta", {})
    meta["modelKind"] = "comparison"
    meta["modelName"] = "지속성 기준선과 학습모델 비교"
    meta["modelDescription"] = "동일한 관측 가격 정의에서 직전 관측가격 기준선과 고정 설정 Ridge·HistGradientBoosting 모델을 비교합니다."
    meta["models"] = [
        {"id": "baseline", "name": MODEL_NAMES["baseline"], "kind": "baseline",
         "description": "원점 이전 또는 당일의 가장 최근 유효 가격을 그대로 예측합니다."},
        {"id": "ridge", "name": MODEL_NAMES["ridge"], "kind": "trained",
         "description": "표준화 후 alpha=10인 Ridge 회귀. 설정은 사전 고정했으며 튜닝·우월성은 주장하지 않습니다."},
        {"id": "hist_gb", "name": MODEL_NAMES["hist_gb"], "kind": "trained",
         "description": "고정 반복횟수의 HistGradientBoosting 회귀. 설정은 사전 고정했으며 튜닝·우월성은 주장하지 않습니다."},
    ]
    meta["training"] = {
        "initialPeriod": "2014년 자료를 초기 학습 창으로 사용하며, warm-up 평가 원점은 2014-12-22부터 시작합니다.",
        "evaluationPolicy": "원점마다 1~10일 뒤의 달력 날짜를 모두 예측한 뒤, 2015년 실제 관측일에만 결과를 연결합니다.",
        "cutoffPolicy": "각 fit cutoff는 그 fit을 사용하는 원점 이하이며 학습 목표일도 cutoff 이하입니다. 미래 목표 값은 특징·재학습 일정·예측 발행 여부에 사용하지 않습니다.",
        "refitCadence": {"daily": "매일", "weekly": "월요일 달력 주 시작", "monthly": "매월 1일", "seasonal": "3·6·9·12월 1일; 겨울은 12~2월"},
        "targetTransform": "ln(목표 실제가격 / 예측 원점의 마지막 관측가격)",
        "minimumTrainingTargetDays": MIN_TRAIN_TARGET_DAYS,
        "hyperparametersWereTuned": False,
        "performanceImprovementClaimed": False,
        "priceExportSha256": source_sha256,
        "featureNames": list(FEATURE_NAMES),
        "modelParameters": {model_id: _model_parameters(model_id) for model_id in MODEL_IDS},
    }
    meta["limitations"] = [
        text for text in meta.get("limitations", [])
        if "학습하지 않은 지속성 기준선" not in text and "미래 운영용 예측은 없습니다" not in text
    ]
    meta["limitations"].append("모델 설정은 사전 고정했으며 이 비교만으로 성능 개선, 인과 효과 또는 대회 제출 적합성을 주장하지 않습니다.")

    try:
        import numpy as np
        import sklearn
        import scipy
        import joblib
        import threadpoolctl
    except ImportError as exc:
        raise RuntimeError("model dependencies are missing; install requirements-modeling.txt") from exc
    versions = {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                "numpy": np.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__,
                "threadpoolctl": threadpoolctl.__version__}
    estimator_artifacts: list[dict[str, Any]] = []
    for product in output["products"]:
        series = _observed_series(product)
        eval_actuals = {day: price for day, (price, _) in series.items()
                        if EVALUATION_START <= day <= EVALUATION_END}
        feature_cache: dict[Any, Any] = {}
        sample_pool = _supervised_samples(series, max(series) if series else EVALUATION_START, feature_cache)[0]
        training_matrix = np.asarray([sample[3] for sample in sample_pool], dtype=np.float64)
        training_labels = np.asarray([sample[4] for sample in sample_pool], dtype=np.float64)
        runs = []
        for model_id in MODEL_IDS:
            for cadence in CADENCES:
                run_started = time.perf_counter()
                run, estimator = _fit_cadence(series, model_id, cadence, origin_start, origin_end, eval_actuals,
                                               sample_pool, feature_cache, training_matrix, training_labels)
                runs.append(run)
                estimator_artifacts.append({
                    "modelId": model_id,
                    "productId": product["id"],
                    "cadence": cadence,
                    "fitCutoff": run["diagnostics"]["lastFitCutoff"],
                    "estimator": estimator,
                })
                if progress_callback is not None:
                    progress_callback({
                        "productId": product["id"], "modelId": model_id, "cadence": cadence,
                        "fitCount": run["diagnostics"]["successfulFitCount"],
                        "matchedPredictionCount": run["diagnostics"]["matchedPredictionCount"],
                        "elapsedSeconds": time.perf_counter() - run_started,
                    })
        product["runs"] = runs
        product["notes"] = [note for note in product.get("notes", []) if "학습하지 않은 지속성 기준선" not in note]
        product["notes"].append("Ridge·HistGradientBoosting 결과는 고정 설정 walk-forward 비교이며, 성능 우위를 입증하지 않습니다.")

    if artifacts_dir is not None:
        meta["training"]["artifacts"] = _save_artifacts(artifacts_dir, estimator_artifacts, versions, source_sha256)
    else:
        meta["training"]["artifacts"] = {"saved": False}
    meta["training"]["environment"] = versions
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train fixed walk-forward models from a schema v1 price export.")
    parser.add_argument("--input", default="outputs/prices-dashboard.json", help="schema v1 price-only dashboard JSON")
    parser.add_argument("--output", default="outputs/dashboard.json", help="schema v2 comparison dashboard JSON")
    parser.add_argument("--artifacts-dir", default="outputs/models", help="directory for last-fit joblib files and manifest")
    args = parser.parse_args(argv)
    try:
        if Path(args.input).resolve() == Path(args.output).resolve():
            raise ValueError("input and output paths must differ")
        artifact_root = Path(args.artifacts_dir).resolve() if args.artifacts_dir else None
        input_path = Path(args.input).resolve()
        output_path = Path(args.output).resolve()
        if artifact_root and (input_path.is_relative_to(artifact_root) or output_path.is_relative_to(artifact_root)):
            raise ValueError("input/output JSON paths must be outside the model artifact directory")
        source = json.loads(Path(args.input).read_text(encoding="utf-8"))
        progress = lambda info: print(
            f"training progress product={info['productId']} model={info['modelId']} cadence={info['cadence']} "
            f"fits={info['fitCount']} matched_pairs={info['matchedPredictionCount']} elapsed_s={info['elapsedSeconds']:.1f}",
            file=sys.stderr, flush=True)
        result = build_model_dashboard(source, args.artifacts_dir, progress_callback=progress)
        try:
            write_json_atomic(args.output, result)
        except Exception:
            artifacts = result.get("meta", {}).get("training", {}).get("artifacts", {})
            new_dir = Path(artifacts.get("directory", ""))
            if artifact_root and new_dir.parent.resolve() == artifact_root and new_dir.name.startswith("run-"):
                shutil.rmtree(new_dir, ignore_errors=True)
            raise
    except (OSError, json.JSONDecodeError, RuntimeError, ValueError) as exc:
        print(f"model dashboard generation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
