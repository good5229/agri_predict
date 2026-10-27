"""Immutable offline performance pilot for the exploratory seasonal reentry features."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import dashboard, live_forecast, modeling, supply_pressure_prospective as prospective
from . import supply_pressure_reentry_screen as reentry
from . import supply_pressure_target_screen as target_screen
from .kma_residual_eval import _dm, _holm, _metrics

START, END = target_screen.START, target_screen.END
ORIGINS, HORIZONS = target_screen.ORIGINS, target_screen.HORIZONS
FROZEN_CUTOFF = target_screen.FROZEN_CUTOFF
PRODUCTS = tuple(dashboard.PRODUCTS)
BANDWIDTHS = (9, 28)
MIN_TRAIN_TARGET_DAYS = modeling.MIN_TRAIN_TARGET_DAYS
MIN_INFERENCE_PAIRS = 30
MIN_THREE_MODEL_PAIRED_ROWS = 10
COMPARATORS = (("direct", "directPrice"), ("persistence", "persistencePrice"))
LOSSES = ("absolute", "absolute_percentage")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _price_from_log(source_price: float, log_delta: float) -> float | None:
    try:
        value = source_price * math.exp(log_delta)
    except (OverflowError, ValueError):
        return None
    return value if _finite_positive(value) else None


def _training_samples(series: dict[dt.date, tuple[float, float]], cutoff: dt.date,
                      feature_cache: dict[tuple[dt.date, int], Any],
                      strict_cache: dict[tuple[dt.date, int], Any],
                      days: tuple[dt.date, ...], log_prices: tuple[float, ...]):
    """Build samples using only actual target dates strictly before cutoff."""
    candidate, direct = [], {h: [] for h in HORIZONS}
    for target, (target_price, _) in series.items():
        if target < START or target >= cutoff:
            continue
        for horizon in HORIZONS:
            origin = target - dt.timedelta(days=horizon)
            key = (target, horizon)
            if key not in feature_cache:
                feature_cache[key] = reentry.issue_feature(series, origin, target, horizon)
            candidate_result = feature_cache[key]
            if candidate_result is not None:
                vector, source, _branch = candidate_result
                label = math.log(target_price) - math.log(series[source][0])
                if math.isfinite(label):
                    candidate.append((target, vector, label))

            if key not in strict_cache:
                strict_cache[key] = live_forecast._strict_feature_row(series, origin, target, horizon,
                                                                      days, log_prices)
            strict_result = strict_cache[key]
            if strict_result is not None:
                source, vector = strict_result
                label = math.log(target_price) - math.log(series[source][0])
                if math.isfinite(label):
                    direct[horizon].append((target, vector, label))
    return candidate, direct


def _fit_direct_model(rows):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len({target for target, _, _ in rows}) < MIN_TRAIN_TARGET_DAYS:
        raise ValueError("fewer than 90 distinct strict training target days")
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    model.fit([values for _, values, _ in rows], [label for _, _, label in rows])
    return model


def _fit_origin_models(candidate_rows, direct_rows):
    models = {"candidate": None, "candidateReason": None, "candidateTargetDays": sorted({r[0] for r in candidate_rows}),
              "direct": {}, "directReasons": {}, "directTargetDays": {}}
    if len(models["candidateTargetDays"]) < MIN_TRAIN_TARGET_DAYS:
        models["candidateReason"] = "fewer_than_90_distinct_training_target_days"
    else:
        try:
            models["candidate"] = prospective._fit_direct_ridge(candidate_rows)
        except Exception as exc:
            models["candidateReason"] = f"fit_failed:{type(exc).__name__}"
    for horizon in HORIZONS:
        rows = direct_rows[horizon]
        targets = sorted({r[0] for r in rows})
        models["directTargetDays"][horizon] = targets
        if len(targets) < MIN_TRAIN_TARGET_DAYS:
            models["directReasons"][horizon] = "fewer_than_90_distinct_training_target_days"
            models["direct"][horizon] = None
            continue
        try:
            models["direct"][horizon] = _fit_direct_model(rows)
            models["directReasons"][horizon] = None
        except Exception as exc:
            models["direct"][horizon] = None
            models["directReasons"][horizon] = f"fit_failed:{type(exc).__name__}"
    return models


def _aggregate_metrics(rows: list[dict[str, Any]], products=PRODUCTS, horizons=HORIZONS):
    """Equal-weight horizons within crop, then crops, for every forecast."""
    by_model = {model: {} for model in ("candidatePrice", "directPrice", "persistencePrice")}
    for product in products:
        p_rows = [row for row in rows if row["productId"] == product and row["pairedEvaluable"]]
        for field in by_model:
            h_metrics = [_metrics([row for row in p_rows if row["horizon"] == h], field) for h in horizons]
            crop_metric = {}
            for metric in ("mae", "mape"):
                vals = [entry[metric] for entry in h_metrics]
                crop_metric[metric] = sum(vals) / len(vals) if vals and all(v is not None for v in vals) else None
            by_model[field][product] = crop_metric
    macro = {}
    for field, crop_values in by_model.items():
        macro[field] = {metric: (sum(v[metric] for v in crop_values.values()) / len(products)
                                 if products and all(v[metric] is not None for v in crop_values.values()) else None)
                        for metric in ("mae", "mape")}
    return {"aggregation": "unweighted mean over 1/5/10-day horizons within each crop, then unweighted mean over five crops",
            "byModelAndProduct": by_model, "macro": macro}


def _loss(actual: float, predicted: float, loss: str) -> float:
    error = abs(actual - predicted)
    return error if loss == "absolute" else 100.0 * error / actual


def _dm_family(rows: list[dict[str, Any]], bandwidth: int) -> list[dict[str, Any]]:
    tests = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            paired = [row for row in rows if row["productId"] == product and row["horizon"] == horizon
                      and row["pairedEvaluable"]]
            for comparator, comparator_field in COMPARATORS:
                for loss in LOSSES:
                    diff = {dt.date.fromisoformat(row["target"]):
                            _loss(row["actual"], row["candidatePrice"], loss)
                            - _loss(row["actual"], row[comparator_field], loss)
                            for row in paired}
                    result = _dm(diff, bandwidth)
                    result.update({"productId": product, "horizon": horizon,
                                   "comparator": comparator, "loss": loss,
                                   "conclusion": "inference_unavailable" if (result["status"] != "tested"
                                                                              or result["n"] < MIN_INFERENCE_PAIRS)
                                   else "inconclusive"})
                    tests.append(result)
    _holm(tests)
    for item in tests:
        if item["status"] != "tested" or item["n"] < MIN_INFERENCE_PAIRS:
            item["conclusion"] = "inference_unavailable"
        elif item.get("holmSignificant") and item["meanLossDifference"] < 0:
            item["conclusion"] = "improvement"
        elif item.get("holmSignificant") and item["meanLossDifference"] > 0:
            item["conclusion"] = "worsening"
        else:
            item["conclusion"] = "inconclusive"
    return tests


def _combined_dm_conclusions(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            for comparator, _field in COMPARATORS:
                items = [item for item in tests if item["productId"] == product
                         and item["horizon"] == horizon and item["comparator"] == comparator]
                if any(item["conclusion"] == "inference_unavailable" for item in items):
                    conclusion = "inference_unavailable"
                elif all(item["conclusion"] == "improvement" for item in items):
                    conclusion = "improvement"
                elif all(item["conclusion"] == "worsening" for item in items):
                    conclusion = "worsening"
                else:
                    conclusion = "inconclusive"
                results.append({"productId": product, "horizon": horizon,
                                "comparator": comparator, "conclusion": conclusion})
    return results


def _coverage_gates(rows: list[dict[str, Any]], products=PRODUCTS, horizons=HORIZONS):
    groups = []
    for product in products:
        for horizon in horizons:
            subset = [row for row in rows if row["productId"] == product and row["horizon"] == horizon]
            targets = sum(row["targetStatus"] == "present" for row in subset)
            candidate = sum(row["targetStatus"] == "present" and row["candidateStatus"] == "available"
                            for row in subset)
            direct = sum(row["targetStatus"] == "present" and row["directStatus"] == "available"
                         for row in subset)
            groups.append({"productId": product, "horizon": horizon, "targetPresent": targets,
                           "candidatePredictions": candidate, "directPredictions": direct,
                           "candidateCoverage": candidate / targets if targets else None,
                           "directCoverage": direct / targets if targets else None,
                           "candidateAtLeastDirect": candidate >= direct})
    return {"groups": groups,
            "candidateAtLeastDirectEveryProductHorizon": all(g["candidateAtLeastDirect"] for g in groups),
            "candidatePredictionsOnTargets": sum(g["candidatePredictions"] for g in groups),
            "directPredictionsOnTargets": sum(g["directPredictions"] for g in groups)}


def _paired_minimum_gate(rows: list[dict[str, Any]], products=PRODUCTS, horizons=HORIZONS):
    groups = []
    for product in products:
        for horizon in horizons:
            count = sum(row["productId"] == product and row["horizon"] == horizon
                        and row["pairedEvaluable"] for row in rows)
            groups.append({"productId": product, "horizon": horizon,
                           "threeModelPairedRows": count, "minimum": MIN_THREE_MODEL_PAIRED_ROWS,
                           "passed": count >= MIN_THREE_MODEL_PAIRED_ROWS})
    return {"groups": groups, "allMeetMinimum": all(group["passed"] for group in groups)}


def _goal_gates(aggregate: dict[str, Any], rows: list[dict[str, Any]], products=PRODUCTS):
    by_product = aggregate["byModelAndProduct"]
    candidate_macro = aggregate["macro"]["candidatePrice"]
    direct_macro = aggregate["macro"]["directPrice"]
    per_product = {}
    for product in products:
        candidate, direct = by_product["candidatePrice"][product], by_product["directPrice"][product]
        mae_limit = direct["mae"] * 1.05 if direct["mae"] is not None else None
        per_product[product] = {
            "candidateMape": candidate["mape"], "directMape": direct["mape"],
            "candidateMae": candidate["mae"], "directMae": direct["mae"],
            "mapeWorseningAtMost1pp": (candidate["mape"] is not None and direct["mape"] is not None
                                        and candidate["mape"] <= direct["mape"] + 1.0),
            "maeWorseningAtMost5Percent": (candidate["mae"] is not None and mae_limit is not None
                                            and candidate["mae"] <= mae_limit),
        }
    coverage = _coverage_gates(rows, products)
    paired_minimum = _paired_minimum_gate(rows, products)
    return {"candidateMacroMapeAtMost10Percent": candidate_macro["mape"] is not None and candidate_macro["mape"] <= 10.0,
            "candidateMacroMape": candidate_macro["mape"], "directMacroMape": direct_macro["mape"],
            "candidateImprovesDirectByAtLeast0_5pp": (candidate_macro["mape"] is not None
                and direct_macro["mape"] is not None and candidate_macro["mape"] <= direct_macro["mape"] - 0.5),
            "allProductsWithinMapeAndMaeWorseningLimits": all(
                value["mapeWorseningAtMost1pp"] and value["maeWorseningAtMost5Percent"]
                for value in per_product.values()),
            "candidateCoverageAtLeastDirect": coverage["candidateAtLeastDirectEveryProductHorizon"],
            "coverage": coverage,
            "minimumThreeModelPairsPerProductHorizon": MIN_THREE_MODEL_PAIRED_ROWS,
            "threeModelPairedRowsByProductHorizon": paired_minimum["groups"],
            "allProductHorizonsMeetThreeModelPairMinimum": paired_minimum["allMeetMinimum"],
            "allGrowthGatesPassed": (candidate_macro["mape"] is not None and candidate_macro["mape"] <= 10.0
                                      and direct_macro["mape"] is not None
                                      and candidate_macro["mape"] <= direct_macro["mape"] - 0.5
                                      and all(value["mapeWorseningAtMost1pp"] and value["maeWorseningAtMost5Percent"]
                                              for value in per_product.values())
                                      and coverage["candidateAtLeastDirectEveryProductHorizon"]
                                      and paired_minimum["allMeetMinimum"]),
            "perProduct": per_product}


def _write_immutable(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    prospective._write_immutable(path, serialized)


def validate_inputs(prices_path: Path, status_path: Path):
    return target_screen.validate_inputs(prices_path, status_path)


def build(prices_path: Path, status_path: Path, output: Path) -> dict[str, Any]:
    products_data, status, provenance_path, prices_bytes, status_bytes, provenance_bytes = validate_inputs(prices_path, status_path)
    rows, fit_audits = [], []
    for pid in PRODUCTS:
        raw_series = prospective._series(products_data[pid])
        series = {day: pair for day, pair in raw_series.items() if START <= day <= END}
        days = tuple(sorted(series))
        log_prices = tuple(math.log(series[day][0]) for day in days)
        candidate_cache, strict_cache = {}, {}
        feature_cache = {}
        for issue_origin in ORIGINS:
            candidate_train, direct_train = _training_samples(series, issue_origin, candidate_cache,
                                                              strict_cache, days, log_prices)
            model_pack = _fit_origin_models(candidate_train, direct_train)
            fit_audits.append({"productId": pid, "origin": issue_origin.isoformat(),
                               "trainingTargetCutoffExclusive": issue_origin.isoformat(),
                               "candidateDistinctTargetDays": len(model_pack["candidateTargetDays"]),
                               "candidateFitReason": model_pack["candidateReason"],
                               "candidateTrainingTargetDays": [day.isoformat() for day in model_pack["candidateTargetDays"]],
                               "directByHorizon": {str(h): {
                                   "distinctTargetDays": len(model_pack["directTargetDays"][h]),
                                   "fitReason": model_pack["directReasons"][h],
                                   "trainingTargetDays": [day.isoformat() for day in model_pack["directTargetDays"][h]]}
                                   for h in HORIZONS}})
            source_idx = bisect_left(days, issue_origin) - 1
            source_day = days[source_idx] if source_idx >= 0 else None
            source_price = series[source_day][0] if source_day is not None else None
            source_ok = source_day is not None and (issue_origin - source_day).days <= dashboard.MAX_SOURCE_AGE_DAYS
            persistence = source_price if source_ok else None
            for horizon in HORIZONS:
                target = issue_origin + dt.timedelta(days=horizon)
                feature_key = (target, horizon)
                if feature_key not in feature_cache:
                    feature_cache[feature_key] = reentry.issue_feature(series, issue_origin, target, horizon)
                feature = feature_cache[feature_key]
                target_value = series.get(target)
                actual = target_value[0] if target_value else None

                candidate_price = None
                candidate_source = None
                candidate_reason = model_pack["candidateReason"]
                if model_pack["candidate"] is not None and feature is not None:
                    candidate_source = feature[1]
                    log_pred = float(model_pack["candidate"].predict([feature[0]])[0])
                    candidate_price = _price_from_log(series[candidate_source][0], log_pred)
                    if candidate_price is None:
                        candidate_reason = "nonpositive_or_nonfinite_prediction"
                elif model_pack["candidate"] is not None:
                    candidate_reason = "candidate_feature_unavailable"

                direct_price = None
                direct_source = None
                direct_reason = model_pack["directReasons"][horizon]
                strict_key = (target, horizon)
                if strict_key not in strict_cache:
                    strict_cache[strict_key] = live_forecast._strict_feature_row(series, issue_origin, target,
                                                                                 horizon, days, log_prices)
                strict_feature = strict_cache[strict_key]
                direct_model = model_pack["direct"][horizon]
                if direct_model is not None and strict_feature is not None:
                    direct_source, vector = strict_feature
                    log_pred = float(direct_model.predict([vector])[0])
                    direct_price = _price_from_log(series[direct_source][0], log_pred)
                    if direct_price is None:
                        direct_reason = "nonpositive_or_nonfinite_prediction"
                elif direct_model is not None:
                    direct_reason = "strict_feature_unavailable"

                target_present = _finite_positive(actual)
                candidate_present, direct_present = _finite_positive(candidate_price), _finite_positive(direct_price)
                persistence_present = _finite_positive(persistence)
                rows.append({"productId": pid, "origin": issue_origin.isoformat(), "horizon": horizon,
                             "target": target.isoformat(), "targetStatus": "present" if target_present else "missing_observation",
                             "actual": actual if target_present else None,
                             "candidatePrice": candidate_price, "candidateSourceDate": candidate_source.isoformat() if candidate_source else None,
                             "candidateStatus": "available" if candidate_present else "unavailable",
                             "candidateFailureReason": candidate_reason,
                             "candidateFeatureAvailable": feature is not None,
                             "candidateFeatureFailureReason": None if feature is not None else "feature_unavailable",
                             "candidateReentryBranch": bool(feature is not None and feature[2]),
                             "directPrice": direct_price, "directSourceDate": direct_source.isoformat() if direct_source else None,
                             "directStatus": "available" if direct_present else "unavailable",
                             "directFailureReason": direct_reason,
                             "persistencePrice": persistence if persistence_present else None,
                             "persistenceSourceDate": source_day.isoformat() if source_ok else None,
                             "persistenceStatus": "available" if persistence_present else "unavailable",
                             "persistenceFailureReason": None if persistence_present else "no_fresh_pre_origin_price",
                             "pairedEvaluable": bool(target_present and candidate_present and direct_present and persistence_present),
                             "trainingTargetCutoffExclusive": issue_origin.isoformat(),
                             "candidateTrainingDistinctTargetDays": len(model_pack["candidateTargetDays"]),
                             "directTrainingDistinctTargetDays": len(model_pack["directTargetDays"][horizon])})

    products_summary = []
    for pid in PRODUCTS:
        subset = [row for row in rows if row["productId"] == pid]
        by_horizon = []
        for horizon in HORIZONS:
            hr = [row for row in subset if row["horizon"] == horizon]
            targets = sum(row["targetStatus"] == "present" for row in hr)
            by_horizon.append({"horizon": horizon, "planned": len(hr), "targetPresent": targets,
                               "candidateFeatureAvailable": sum(row["targetStatus"] == "present" and row["candidateFeatureAvailable"] for row in hr),
                               "candidateReentryBranch": sum(row["targetStatus"] == "present" and row["candidateReentryBranch"] for row in hr),
                               "candidatePredictions": sum(row["candidateStatus"] == "available" for row in hr),
                               "directPredictions": sum(row["directStatus"] == "available" for row in hr),
                               "pairedEvaluable": sum(row["pairedEvaluable"] for row in hr),
                               "candidateFailureReasons": {reason: sum(row["candidateFailureReason"] == reason for row in hr)
                                                           for reason in sorted({row["candidateFailureReason"] for row in hr if row["candidateFailureReason"]})},
                               "directFailureReasons": {reason: sum(row["directFailureReason"] == reason for row in hr)
                                                        for reason in sorted({row["directFailureReason"] for row in hr if row["directFailureReason"]})},
                               "candidateConditionalCoverage": sum(row["targetStatus"] == "present" and row["candidateStatus"] == "available" for row in hr) / targets if targets else None,
                               "directConditionalCoverage": sum(row["targetStatus"] == "present" and row["directStatus"] == "available" for row in hr) / targets if targets else None,
                               "metrics": {field: _metrics([row for row in hr if row["pairedEvaluable"]], field)
                                           for field in ("candidatePrice", "directPrice", "persistencePrice")}})
        products_summary.append({"productId": pid, "planned": len(subset),
                                 "targetPresent": sum(row["targetStatus"] == "present" for row in subset),
                                 "pairedEvaluable": sum(row["pairedEvaluable"] for row in subset),
                                 "byHorizon": by_horizon})

    aggregate = _aggregate_metrics(rows)
    gates = _goal_gates(aggregate, rows)
    dm = {str(bandwidth): _dm_family(rows, bandwidth) for bandwidth in BANDWIDTHS}
    dm_conclusions = {band: _combined_dm_conclusions(tests) for band, tests in dm.items()}
    payload = {"schemaVersion": 1, "experiment": "supply_pressure_seasonal_reentry_small_performance_pilot",
               "inputWindow": {"start": START.isoformat(), "end": END.isoformat()},
               "origins": [day.isoformat() for day in ORIGINS], "horizons": HORIZONS,
               "plannedRows": len(rows), "minimumTrainingDistinctTargetDays": MIN_TRAIN_TARGET_DAYS,
               "minimumInferencePairs": MIN_INFERENCE_PAIRS, "dmHacBandwidthsCalendarDays": BANDWIDTHS,
               "dmTestsPerBandwidth": len(PRODUCTS) * len(HORIZONS) * len(COMPARATORS) * len(LOSSES),
               "candidateFeatureNames": list(prospective.FEATURES) + ["logMeanVolume7", "missing28"],
               "noFutureTargetTraining": True,
               "goalAndGrowthGates": gates,
               "pairedAggregateMetrics": aggregate,
               "products": products_summary, "fitAudits": fit_audits, "rows": rows,
               "dmTestsByBandwidth": dm,
               "dmConclusionsByBandwidth": dm_conclusions,
               "limitations": ["availability-selected exploratory dates; not independent validation",
                               "unavailable or ineligible predictions remain missing; no fallback prediction substituted"],
               "input": {"pricesPath": str(prices_path.resolve()), "pricesSha256": _sha(prices_bytes),
                         "runStatusPath": str(status_path.resolve()), "runStatusSha256": _sha(status_bytes),
                         "snapshotFingerprint": status["snapshotSha256"],
                         "provenancePath": str(provenance_path.resolve()), "provenanceSha256": _sha(provenance_bytes)},
               "codeSha256": _sha(Path(__file__).read_bytes())}
    _write_immutable(output, payload)
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
