"""Frozen offline ECMWF-feature correction evaluation for fixed price pairs.

Dry-run is the default. Execution reads only the pinned local prices, direct
forecast report, and a complete feature snapshot; it makes no network calls.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from . import dashboard, direct_horizon as dh, ecmwf_field_collection as collection
from . import ecmwf_point_features as point_features
from . import kma_residual_eval as common

ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = dh.INPUT_PRICES
DIRECT_REPORT_PATH = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
FEATURE_DIR = ROOT / "outputs/experiments/ecmwf-point-features-20261007"
OUTPUT_DIR = ROOT / "outputs/experiments/ecmwf-price-eval-20261007"
PRICE_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
DIRECT_REPORT_SHA256 = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
PRODUCTS = tuple(dashboard.PRODUCTS)
HORIZONS = tuple(range(1, 11))
FEATURE_ORDER = tuple(point_features.FEATURE_ORDER)
TRAIN_START = dt.date(2024, 3, 16)
TRAIN_END = dt.date(2025, 9, 1)
EVAL_START = dt.date(2025, 9, 25)
EVAL_END = dt.date(2026, 9, 24)
MIN_TRAIN_DAYS = 90
MIN_EVAL_PAIRS = 30
MATURITY_GAP_DAYS = 24
CORRECTION_CLIP = 0.20
BANDWIDTHS = (9, 28)
ALPHA = 0.05
COMPARATORS = ("direct", "persistence")
LOSSES = ("mae", "mape")
FAMILY_SIZE = len(PRODUCTS) * len(HORIZONS) * len(COMPARATORS) * len(LOSSES)
EXPECTED_EVAL_PAIRS = 12_508

PROTOCOL = {
    "id": "ecmwf-point-features-price-correction-2026-10-07",
    "sourceScopeHash": collection.SCOPE_HASH,
    "trainingTargetDates": [TRAIN_START.isoformat(), TRAIN_END.isoformat()],
    "evaluationTargetDates": [EVAL_START.isoformat(), EVAL_END.isoformat()],
    "trainingPairing": "same frozen direct report product/horizon/origin/target/actual; fully finite 18 ECMWF features only",
    "featureOrder": list(FEATURE_ORDER),
    "minimumDistinctTrainingTargetDays": MIN_TRAIN_DAYS,
    "standardization": "fit mean and population standard deviation on training rows only; zero standard deviation uses scale 1",
    "model": {"estimator": "sklearn.linear_model.Ridge", "alpha": 100.0, "fit_intercept": True,
              "scaler": "sklearn.preprocessing.StandardScaler, training rows only"},
    "target": "log(actual / direct_ridge_prediction)",
    "evaluationMaturityGapCalendarDays": MATURITY_GAP_DAYS,
    "candidate": "direct * exp(clip(predicted_log_correction, -0.20, 0.20))",
    "fallback": "use direct prediction for absent/nonfinite feature, unavailable/nonfinite model, immature training label, or invalid candidate",
    "evaluationPairDenominator": EXPECTED_EVAL_PAIRS,
    "comparisons": list(COMPARATORS), "losses": list(LOSSES),
    "dm": {"twoSided": True, "calendarBartlettHacDays": list(BANDWIDTHS),
           "fixedHolmFamilySizePerBandwidth": FAMILY_SIZE, "holmAlpha": ALPHA,
           "minimumPairs": MIN_EVAL_PAIRS, "unavailablePValue": 1.0,
           "jointImprovementGate": "both paired MAE and MAPE lower than each comparator; all four HAC9 Holm tests favor candidate; coverage nondeclining"},
    "aggregation": "equal weight over ten horizons within each product, then equal weight over five products",
    "availabilityClaim": "Last-Modified is not proof of first publication; retrospective price revisions and market proxy limitations remain",
}


class EvaluationError(ValueError):
    pass


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"input unreadable or invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"input root must be an object: {path}")
    return value, raw


def _origin_dates() -> list[str]:
    start = dt.date(2024, 3, 16)
    end = dt.date(2026, 9, 23)
    return [(start + dt.timedelta(days=i)).isoformat() for i in range((end - start).days + 1)]


def validate_feature_inputs(feature_doc: dict[str, Any], manifest: dict[str, Any], *,
                            feature_bytes: bytes | None = None) -> dict[str, list[float] | None]:
    """Validate the full 922-day feature artifact and return ordered vectors."""
    if manifest.get("source_scope_hash") != collection.SCOPE_HASH:
        raise EvaluationError("feature manifest source scope hash mismatch")
    if manifest.get("feature_order") != list(FEATURE_ORDER) or manifest.get("feature_count") != 18:
        raise EvaluationError("feature order/count differs from frozen 18-feature protocol")
    if manifest.get("row_count") != 922:
        raise EvaluationError("feature manifest does not cover all 922 run dates")
    hashes = manifest.get("source_checkpoint_hashes")
    expected_run_dates = {(dt.date.fromisoformat(day) - dt.timedelta(days=1)).strftime("%Y%m%d")
                          for day in _origin_dates()}
    if not isinstance(hashes, dict) or set(hashes) != expected_run_dates:
        raise EvaluationError("feature snapshot is missing one or more of the 922 source checkpoints")
    if any(not isinstance(value, str) or len(value) != 64
           or any(ch not in "0123456789abcdef" for ch in value.lower()) for value in hashes.values()):
        raise EvaluationError("source checkpoint hashes are malformed")
    if feature_bytes is not None and _sha(feature_bytes) != manifest.get("feature_sha256"):
        raise EvaluationError("feature data hash does not match feature manifest")
    if manifest.get("feature_file") is not None:
        expected_name = manifest["feature_file"]
        if not isinstance(expected_name, str) or Path(expected_name).name != expected_name:
            raise EvaluationError("feature file reference in manifest is invalid")
    rows = feature_doc.get("rows")
    if feature_doc.get("schema_version") != 1 or not isinstance(rows, list) or len(rows) != 922:
        raise EvaluationError("feature data must contain exactly 922 rows")
    vectors: dict[str, list[float] | None] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("origin_date"), str):
            raise EvaluationError("feature row identity is malformed")
        origin = row["origin_date"]
        if origin in vectors:
            raise EvaluationError(f"duplicate feature origin date: {origin}")
        try:
            dt.date.fromisoformat(origin)
        except ValueError as exc:
            raise EvaluationError(f"invalid feature origin date: {origin}") from exc
        if not (dt.date(2024, 3, 16) <= dt.date.fromisoformat(origin) <= dt.date(2026, 9, 23)):
            raise EvaluationError(f"feature origin date outside frozen source window: {origin}")
        raw_values = row.get("features")
        vector = None
        if (row.get("candidate_valid") is True and isinstance(raw_values, dict)
                and set(raw_values) == set(FEATURE_ORDER)):
            candidate = [raw_values.get(name) for name in FEATURE_ORDER]
            if all(_finite(value) for value in candidate):
                vector = [float(value) for value in candidate]
        vectors[origin] = vector
    if set(vectors) != set(_origin_dates()):
        raise EvaluationError("feature rows do not exactly cover fixed origin-date interval")
    return vectors


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _training_rows(direct_rows: dict[tuple[str, int, dt.date], dict[str, Any]],
                   product: str, horizon: int) -> list[dict[str, Any]]:
    return sorted((row for row in direct_rows.values()
                   if row["product"] == product and row["horizon"] == horizon
                   and TRAIN_START <= row["target"] <= TRAIN_END),
                  key=lambda row: (row["target"], row["origin"]))


def _fit_model(training: list[dict[str, Any]], vectors: dict[str, list[float] | None]) -> tuple[dict[str, Any], Any | None]:
    eligible = [row for row in training if vectors.get(row["origin"].isoformat()) is not None
                and _finite(row["actual"]) and row["actual"] > 0
                and _finite(row["direct"]) and row["direct"] > 0]
    targets = sorted({row["target"] for row in eligible})
    audit: dict[str, Any] = {"pairedTrainingRows": len(training), "featureAvailableTrainingRows": len(eligible),
                             "distinctTrainingTargetDays": len(targets),
                             "trainingTargetStart": targets[0].isoformat() if targets else None,
                             "trainingTargetEnd": targets[-1].isoformat() if targets else None,
                             "status": "insufficient_training"}
    paired_payload = [{"target": row["target"].isoformat(), "origin": row["origin"].isoformat(),
                       "actual": row["actual"], "direct": row["direct"]}
                      for row in sorted(training, key=lambda item: (item["target"], item["origin"]))]
    fit_payload = [{**record, "features": vectors[record["origin"]]} for record in [
        {"target": row["target"].isoformat(), "origin": row["origin"].isoformat(),
         "actual": row["actual"], "direct": row["direct"]} for row in eligible]]
    audit["pairedTrainingInputSha256"] = _sha(_json_bytes(paired_payload))
    audit["eligibleFitSampleSha256"] = _sha(_json_bytes(fit_payload))
    if len(targets) < MIN_TRAIN_DAYS:
        return audit, None
    try:
        import numpy as np
        import sklearn
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        audit["status"] = "model_dependency_unavailable"
        audit["dependencyError"] = type(exc).__name__
        return audit, None
    x = np.asarray([vectors[row["origin"].isoformat()] for row in eligible], dtype=float)
    y = np.asarray([math.log(row["actual"] / row["direct"]) for row in eligible], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        audit["status"] = "nonfinite_training_data"
        return audit, None
    scaler = StandardScaler().fit(x)
    scaler.scale_[scaler.scale_ == 0] = 1.0
    z = scaler.transform(x)
    try:
        model = Ridge(alpha=100, fit_intercept=True).fit(z, y)
        if not np.isfinite(model.coef_).all() or not _finite(model.intercept_):
            audit["status"] = "nonfinite_fit"
            return audit, None
    except (ValueError, FloatingPointError, ArithmeticError, np.linalg.LinAlgError):
        audit["status"] = "fit_failed"
        return audit, None
    audit.update({"status": "fitted", "scalerMean": scaler.mean_.tolist(),
                  "scalerScale": scaler.scale_.tolist(),
                  "ridgeCoefficients": model.coef_.tolist(), "ridgeIntercept": float(model.intercept_),
                  "numpyVersion": np.__version__, "sklearnVersion": sklearn.__version__,
                  "trainingTargetDates": [day.isoformat() for day in targets],
                  "trainingTargetDatesSha256": _sha("\n".join(day.isoformat() for day in targets).encode())})
    return audit, (scaler, model)


def _predict(direct: float, origin: dt.date, fitted: tuple[Any, Any] | None,
             vector: list[float] | None, *, train_end: dt.date = TRAIN_END) -> tuple[float, str | None, float | None]:
    if fitted is None:
        return direct, "model_unavailable", None
    if vector is None:
        return direct, "feature_vector_missing_or_nonfinite", None
    if (origin - train_end).days < MATURITY_GAP_DAYS:
        return direct, "training_label_maturity_gap", None
    scaler, model = fitted
    try:
        import numpy as np
        prediction = float(model.predict(scaler.transform(np.asarray([vector], dtype=float)))[0])
    except (ValueError, FloatingPointError, ArithmeticError, ImportError):
        return direct, "nonfinite_model_prediction", None
    if not math.isfinite(prediction):
        return direct, "nonfinite_model_prediction", None
    clipped = max(-CORRECTION_CLIP, min(CORRECTION_CLIP, prediction))
    candidate = direct * math.exp(clipped)
    if not math.isfinite(candidate) or candidate <= 0:
        return direct, "invalid_candidate_prediction", prediction
    return candidate, None, prediction


def _dm_calendar(differences: dict[dt.date, float], bandwidth: int) -> dict[str, Any]:
    """Two-sided DM normal test using Bartlett HAC on actual calendar lags."""
    if len(differences) < MIN_EVAL_PAIRS:
        return {"status": "insufficient_pairs", "n": len(differences), "meanLossDifference": None,
                "statistic": None, "pValue": 1.0}
    return dh._dm(differences, bandwidth)


def _holm(rows: list[dict[str, Any]]) -> None:
    if len(rows) != FAMILY_SIZE:
        raise AssertionError(f"fixed Holm family has {len(rows)}, expected {FAMILY_SIZE}")
    order = sorted(range(len(rows)), key=lambda i: rows[i]["pValue"])
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(rows) - rank) * rows[index]["pValue"]))
        rows[index]["holmPValue"] = running
        rows[index]["holmSignificant"] = running < ALPHA


def _metrics(rows: list[dict[str, Any]], prediction: str) -> dict[str, float | None]:
    if not rows:
        return {"mae": None, "mape": None}
    return {"mae": sum(abs(row["actual"] - row[prediction]) for row in rows) / len(rows),
            "mape": 100.0 * sum(abs(row["actual"] - row[prediction]) / row["actual"] for row in rows) / len(rows)}


def _loss_diff(rows: list[dict[str, Any]], comparator: str, loss: str) -> dict[dt.date, float]:
    if loss == "mae":
        return {r["target"]: abs(r["actual"] - r["candidate"]) - abs(r["actual"] - r[comparator]) for r in rows}
    return {r["target"]: 100.0 * (abs(r["actual"] - r["candidate"]) - abs(r["actual"] - r[comparator])) / r["actual"] for r in rows}


def _season(day: dt.date) -> str:
    return {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
            6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}[day.month]


def build(prices: dict[str, Any], report: dict[str, Any], feature_doc: dict[str, Any],
          feature_manifest: dict[str, Any], *, feature_bytes: bytes | None = None,
          hashes: dict[str, str] | None = None,
          expected_eval_pairs: int | None = EXPECTED_EVAL_PAIRS) -> dict[str, Any]:
    vectors = validate_feature_inputs(feature_doc, feature_manifest, feature_bytes=feature_bytes)
    direct_rows = common._direct_rows(report, prices)
    eval_rows = [row for row in direct_rows.values() if EVAL_START <= row["target"] <= EVAL_END]
    if len(eval_rows) != EXPECTED_EVAL_PAIRS:
        raise EvaluationError(f"frozen direct eval denominator mismatch: {len(eval_rows)} != {EXPECTED_EVAL_PAIRS}")
    if expected_eval_pairs is not None and len(eval_rows) != expected_eval_pairs:
        raise EvaluationError("direct eligible evaluation rows differ from frozen denominator")

    audits, fitted = {}, {}
    for product in PRODUCTS:
        for horizon in HORIZONS:
            key = f"{product}/{horizon}"
            training = _training_rows(direct_rows, product, horizon)
            audit, model = _fit_model(training, vectors)
            audits[key] = audit
            if model is not None:
                fitted[key] = model

    predictions, grouped = [], defaultdict(list)
    feature_counts = Counter()
    for row in sorted(eval_rows, key=lambda r: (r["product"], r["horizon"], r["target"])):
        feature = vectors.get(row["origin"].isoformat())
        candidate, fallback, correction = _predict(row["direct"], row["origin"],
                                                    fitted.get(f"{row['product']}/{row['horizon']}"), feature)
        if fallback == "model_unavailable" and audits[f"{row['product']}/{row['horizon']}"].get("status") == "insufficient_training":
            fallback = "insufficient_training"
        if fallback == "model_unavailable" and audits[f"{row['product']}/{row['horizon']}"].get("status") != "fitted":
            fallback = audits[f"{row['product']}/{row['horizon']}"]["status"]
        feature_counts["available" if feature is not None else "missing_or_nonfinite"] += 1
        if fallback:
            feature_counts[f"fallback:{fallback}"] += 1
        item = {**row, "candidate": candidate, "candidateCorrectionLog": correction,
                "fallbackReason": fallback, "featureOrigin": row["origin"].isoformat(),
                "featureAvailable": feature is not None}
        predictions.append(item)
        grouped[(row["product"], row["horizon"])].append(item)

    comparisons = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            rows = grouped[(product, horizon)]
            for comparator in COMPARATORS:
                for loss in LOSSES:
                    difference = _loss_diff(rows, comparator, loss)
                    tests = {str(bandwidth): _dm_calendar(difference, bandwidth) for bandwidth in BANDWIDTHS}
                    comparisons.append({"product": product, "horizon": horizon,
                                        "comparator": comparator, "loss": loss,
                                        "n": len(rows),
                                        "candidateMetric": _metrics(rows, "candidate")[loss],
                                        "comparatorMetric": _metrics(rows, comparator)[loss],
                                        "dm": tests})
    for bandwidth in BANDWIDTHS:
        family = [item["dm"][str(bandwidth)] for item in comparisons]
        _holm(family)

    conclusions = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            needed = [item for item in comparisons if item["product"] == product and item["horizon"] == horizon]
            point_better = all(item["candidateMetric"] is not None and item["comparatorMetric"] is not None
                               and item["candidateMetric"] < item["comparatorMetric"] for item in needed)
            primary = [item["dm"]["9"] for item in needed]
            favorable = [test for test in primary if test["holmSignificant"] and test.get("meanLossDifference") is not None
                         and test["meanLossDifference"] < 0]
            adverse = [test for test in primary if test["holmSignificant"] and test.get("meanLossDifference") is not None
                       and test["meanLossDifference"] > 0]
            all_available = all(test.get("status") == "tested" for test in primary)
            gate = (point_better and all_available and len(favorable) == 4 and not adverse)
            if gate:
                conclusion = "improvement"
            elif favorable and adverse:
                conclusion = "mixed"
            elif adverse:
                conclusion = "worsening"
            else:
                conclusion = "inconclusive"
            conclusions.append({"product": product, "horizon": horizon,
                                "conclusion": conclusion, "inferenceAvailable": all_available,
                                "bothLossesLowerAgainstBothComparators": point_better,
                                "hac9HolmFavorableTests": len(favorable),
                                "hac9HolmAdverseTests": len(adverse),
                                "coverageNondeclining": len(grouped[(product, horizon)]) ==
                                    sum(1 for row in eval_rows if row["product"] == product and row["horizon"] == horizon)})

    product_summary, horizon_summary, seasonal, yearly = {}, {}, {}, {}
    equal_horizon_mape = {}
    all_candidates = [row for rows in grouped.values() for row in rows]
    for product in PRODUCTS:
        horizon_mapes = {name: [] for name in ("candidate", "direct", "persistence")}
        product_rows = [row for row in all_candidates if row["product"] == product]
        for horizon in HORIZONS:
            rows = grouped[(product, horizon)]
            metrics = {name: _metrics(rows, name) for name in ("candidate", "direct", "persistence")}
            horizon_summary[f"{product}/{horizon}"] = {"n": len(rows), "metrics": metrics,
                "featureAvailableRows": sum(row["featureAvailable"] for row in rows),
                "fallbackRows": sum(row["fallbackReason"] is not None for row in rows),
                "fallbackReasons": dict(Counter(row["fallbackReason"] for row in rows if row["fallbackReason"]))}
            for name in horizon_mapes:
                horizon_mapes[name].append(metrics[name]["mape"])
        equal_horizon_mape[product] = {
            name: (sum(values) / len(HORIZONS) if all(value is not None for value in values) else None)
            for name, values in horizon_mapes.items()}
        product_summary[product] = {"n": len(product_rows),
            "metrics": {name: _metrics(product_rows, name) for name in ("candidate", "direct", "persistence")},
            "equalHorizonMape": equal_horizon_mape[product]}
    macro_mape = {name: (sum(equal_horizon_mape[product][name] for product in PRODUCTS) / len(PRODUCTS)
                         if all(equal_horizon_mape[product][name] is not None for product in PRODUCTS) else None)
                  for name in ("candidate", "direct", "persistence")}
    for key_name, keyer in (("year", lambda r: str(r["target"].year)),
                            ("season", lambda r: _season(r["target"]))):
        output = yearly if key_name == "year" else seasonal
        keys = sorted({keyer(row) for row in all_candidates})
        for group_key in keys:
            rows = [row for row in all_candidates if keyer(row) == group_key]
            output[group_key] = {"n": len(rows),
                "metrics": {name: _metrics(rows, name) for name in ("candidate", "direct", "persistence")},
                "fallbackRows": sum(row["fallbackReason"] is not None for row in rows),
                "featureAvailableRows": sum(row["featureAvailable"] for row in rows)}

    report_out = {"schema_version": 1, "status": "ecmwf_feature_price_evaluation_complete",
                  "protocol": PROTOCOL, "input_hashes": hashes or {},
                  "feature_manifest": {"feature_sha256": feature_manifest.get("feature_sha256"),
                                       "feature_order": list(FEATURE_ORDER),
                                       "source_scope_hash": feature_manifest.get("source_scope_hash"),
                                       "source_checkpoint_count": len(feature_manifest.get("source_checkpoint_hashes", {}))},
                  "evaluation": {"target_start": EVAL_START.isoformat(), "target_end": EVAL_END.isoformat(),
                                 "expected_pairs": EXPECTED_EVAL_PAIRS, "observed_pairs": len(eval_rows),
                                 "coverage": len(predictions) / EXPECTED_EVAL_PAIRS,
                                 "pairedEvalSampleSha256": _sha(_json_bytes([
                                     {"product": row["product"], "horizon": row["horizon"],
                                      "target": row["target"].isoformat(), "origin": row["origin"].isoformat(),
                                      "actual": row["actual"], "direct": row["direct"],
                                      "persistence": row["persistence"],
                                      "features": vectors.get(row["origin"].isoformat())}
                                     for row in sorted(eval_rows, key=lambda r: (r["product"], r["horizon"], r["target"]))])),
                                 "fallback_rows": sum(row["fallbackReason"] is not None for row in predictions),
                                 "fallback_reasons": dict(Counter(row["fallbackReason"] for row in predictions if row["fallbackReason"]))},
                  "training": {"target_start": TRAIN_START.isoformat(), "target_end": TRAIN_END.isoformat(),
                               "models": audits},
                  "metrics": {"by_product_horizon": horizon_summary, "by_product": product_summary,
                              "equal_horizon_mape_by_product": equal_horizon_mape,
                              "five_product_equal_weight_equal_horizon_mape": macro_mape,
                              "by_year": yearly, "by_season": seasonal},
                  "dm": {"family_size_per_bandwidth": FAMILY_SIZE, "comparisons": comparisons,
                         "bandwidths_calendar_days": list(BANDWIDTHS), "two_sided": True,
                         "holm_alpha": ALPHA, "unavailable_p_value": 1.0},
                  "conclusions": conclusions,
                  "paired_predictions": [{k: v.isoformat() if isinstance(v, dt.date) else v for k, v in row.items()}
                                         for row in predictions]}
    return report_out


def _write_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError:
        if path.read_bytes() != content:
            raise
    finally:
        tmp.unlink(missing_ok=True)


def _freeze_prefit(output_dir: Path, hashes: dict[str, str]) -> tuple[str, str]:
    frozen = {"schema_version": 1, "protocol": PROTOCOL, "input_hashes": hashes,
              "frozen_before_fit": True}
    raw = _json_bytes(frozen)
    digest = _sha(raw)
    name = f"prefit_{digest}.json"
    _write_immutable(output_dir / name, raw)
    return name, digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="run the frozen offline evaluation")
    parser.add_argument("--prices", default=str(PRICE_PATH))
    parser.add_argument("--direct-report", default=str(DIRECT_REPORT_PATH))
    parser.add_argument("--feature-file", required=False)
    parser.add_argument("--feature-manifest", required=False)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "prices": args.prices,
                          "prices_sha256": PRICE_SHA256, "direct_report": args.direct_report,
                          "direct_report_sha256": DIRECT_REPORT_SHA256,
                          "feature_file_required": True, "feature_manifest_required": True,
                          "expected_eval_pairs": EXPECTED_EVAL_PAIRS, "inputs_read": False,
                          "network_requests": 0, "model_fit": False}, ensure_ascii=False))
        return 0
    if not args.feature_file or not args.feature_manifest:
        print("feature file and manifest are required for execution", file=sys.stderr)
        return 2
    try:
        price_path, report_path = Path(args.prices), Path(args.direct_report)
        feature_path, feature_manifest_path = Path(args.feature_file), Path(args.feature_manifest)
        prices, price_bytes = _read_json(price_path)
        direct_report, direct_bytes = _read_json(report_path)
        feature_doc, feature_bytes = _read_json(feature_path)
        feature_manifest, feature_manifest_bytes = _read_json(feature_manifest_path)
        actual_hashes = {"prices": _sha(price_bytes), "direct_report": _sha(direct_bytes),
                         "feature_file": _sha(feature_bytes), "feature_manifest": _sha(feature_manifest_bytes)}
        if actual_hashes["prices"] != PRICE_SHA256:
            raise EvaluationError("frozen prices SHA-256 mismatch")
        if actual_hashes["direct_report"] != DIRECT_REPORT_SHA256:
            raise EvaluationError("frozen direct report SHA-256 mismatch")
        if feature_manifest.get("feature_file") != feature_path.name:
            raise EvaluationError("feature data filename differs from feature manifest reference")
        if feature_manifest.get("feature_sha256") != actual_hashes["feature_file"]:
            raise EvaluationError("feature data SHA-256 mismatch")
        validate_feature_inputs(feature_doc, feature_manifest, feature_bytes=feature_bytes)
        output_dir = Path(args.output_dir)
        prefit_name, prefit_hash = _freeze_prefit(output_dir, actual_hashes)
        report = build(prices, direct_report, feature_doc, feature_manifest,
                       feature_bytes=feature_bytes, hashes=actual_hashes)
        report["prefit_manifest"] = {"file": prefit_name, "sha256": prefit_hash}
        raw = _json_bytes(report)
        digest = _sha(raw)
        report_name = f"report_{digest}.json"
        _write_immutable(output_dir / report_name, raw)
        print(json.dumps({"report": report_name, "report_sha256": digest,
                          "prefit_manifest": prefit_name, "prefit_sha256": prefit_hash,
                          "output_dir": str(output_dir), **report["evaluation"]}, ensure_ascii=False))
        return 0
    except (EvaluationError, OSError, ValueError, TypeError) as exc:
        print(f"ECMWF price evaluation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
