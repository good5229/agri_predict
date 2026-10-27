"""Frozen direct-Ridge lognormal MAPE calibration, with separate forecast/score phases."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any
import uuid

from agri_data import dashboard, direct_horizon

ROOT = Path(__file__).resolve().parents[1]
DIRECT_REPORT_PATH = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
BLEND_DIR = ROOT / "outputs/experiments/ridge-sarima-blend-20261008"
BLEND_MANIFEST_PATH = BLEND_DIR / "forecast-manifest.json"
BLEND_REPORT_PATH = BLEND_DIR / "report.json"
FORECAST_OUTPUT = ROOT / "outputs/experiments/lognormal-mape-calibration-20261008/forecast-manifest.json"
SCORE_OUTPUT = ROOT / "outputs/experiments/lognormal-mape-calibration-20261008/report.json"

DIRECT_SHA256 = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
BLEND_MANIFEST_SHA256 = "e73cd6b3c47b163000d020dbf60373af67d4edb79309af3a91b349167d945d40"
BLEND_REPORT_SHA256 = "423c70ce3c34dc4eaf80c3b3fb3367fa6eedc73745d2098bf66c8f62c1a5a822"
BLEND_MACRO_MAPE_PCT = 11.743  # Frozen rounded published benchmark from the pinned report.
PRODUCTS = tuple(dashboard.PRODUCTS)
HORIZONS = tuple(range(1, 11))
EVAL_START = dt.date(2025, 9, 25)
EVAL_END = dt.date(2026, 9, 24)
EXPECTED_EVAL_PAIRS = 12_508
LOOKBACK_DAYS = 365
ORIGIN_GAP_DAYS = 14
MIN_RESIDUALS = 30
MAX_ABS_LOG_CORRECTION = 0.10
COMPARATORS = ("direct", "blend", "persistence")
LOSSES = ("AE", "APE")
BANDWIDTHS = (9, 28)
FAMILY_SIZE = len(PRODUCTS) * len(HORIZONS) * len(COMPARATORS) * len(LOSSES)
PROTOCOL = {
    "id": "direct-ridge-lognormal-mape-calibration-2026-10-08",
    "trainingSource": {"directReportSha256": DIRECT_SHA256, "actuals": "direct report historical pairedRows only"},
    "historyWindow": "unique same-product/same-horizon target dates in [origin-365d, origin-14d], inclusive",
    "correction": "direct * exp(clip(mean(log(actual/direct))-population_variance(log(actual/direct)), -0.10, +0.10))",
    "minimumResiduals": MIN_RESIDUALS,
    "fallback": "direct prediction unchanged when fewer than 30 eligible residuals",
    "evaluationTargets": [EVAL_START.isoformat(), EVAL_END.isoformat()],
    "expectedPairs": EXPECTED_EVAL_PAIRS,
    "comparisonFamilyPerBandwidth": FAMILY_SIZE,
    "comparators": list(COMPARATORS), "losses": list(LOSSES),
    "dm": {"twoSided": True, "calendarBartlettHacDays": list(BANDWIDTHS), "holmAlpha": 0.05,
           "minimumPairs": 30, "unavailablePValue": 1.0},
    "limitations": "Retrospective revised report; not prospective or official-score evidence.",
}


class CalibrationError(ValueError):
    pass


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def _positive(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise CalibrationError(f"{label} must be positive finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalibrationError(f"{label} must be positive finite") from exc
    if not math.isfinite(number) or number <= 0:
        raise CalibrationError(f"{label} must be positive finite")
    return number


def _direct_rows(report: dict[str, Any], *, validate_actual: bool = True) -> dict[tuple[str, int, dt.date, dt.date], dict[str, Any]]:
    if report.get("schemaVersion", 1) != 1 or report.get("protocol", {}).get("id") != "direct-horizon-ridge-2026-10-06":
        raise CalibrationError("unexpected pinned direct report schema or protocol")
    rows = {}
    for index, row in enumerate(report.get("pairedRows", [])):
        if not isinstance(row, dict):
            raise CalibrationError(f"direct pairedRows[{index}] must be an object")
        product, horizon = row.get("product"), row.get("horizon")
        if product not in PRODUCTS or isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
            raise CalibrationError(f"invalid direct row product/horizon at index {index}")
        try:
            target = dt.date.fromisoformat(row["target"])
            origin = dt.date.fromisoformat(row.get("origin") or (target - dt.timedelta(days=horizon)).isoformat())
        except (KeyError, TypeError, ValueError) as exc:
            raise CalibrationError(f"invalid direct row date at index {index}") from exc
        if target - origin != dt.timedelta(days=horizon):
            raise CalibrationError(f"direct row origin/target/horizon mismatch at index {index}")
        actual = _positive(row.get("actual"), "scoring actual") if validate_actual else None
        direct = _positive(row.get("direct"), "direct prediction")
        persistence = _positive(row.get("persistence"), "persistence prediction")
        key = (product, horizon, origin, target)
        if key in rows:
            raise CalibrationError(f"duplicate direct key: {key}")
        record = {"product": product, "horizon": horizon, "origin": origin,
                  "target": target, "actual": actual, "direct": direct,
                  "persistence": persistence}
        if not validate_actual:
            record["sourceRow"] = row
        rows[key] = record
    if not rows:
        raise CalibrationError("direct report has no paired rows")
    return rows


def _correction(prior: list[dict[str, Any]], origin: dt.date, direct: float) -> dict[str, Any]:
    cutoff = origin - dt.timedelta(days=ORIGIN_GAP_DAYS)
    start = origin - dt.timedelta(days=LOOKBACK_DAYS)
    eligible = [r for r in prior if start <= r["target"] <= cutoff]
    by_target: dict[dt.date, dict[str, Any]] = {}
    for row in eligible:
        if row["target"] in by_target:
            raise CalibrationError(f"duplicate calibration target date: {row['target']}")
        by_target[row["target"]] = row
    ordered = [by_target[day] for day in sorted(by_target)]
    # Convert actuals only after applying the strict per-origin temporal filter.
    residuals = [math.log(_positive(r.get("sourceRow", {}).get("actual", r.get("actual")),
                                   "eligible historical actual") / r["direct"])
                 for r in ordered]
    if not all(math.isfinite(x) for x in residuals):
        raise CalibrationError("nonfinite historical log residual")
    n = len(residuals)
    if n < MIN_RESIDUALS:
        return {"prediction": direct, "fallback": True, "fallbackReason": "fewer_than_30_eligible_residuals",
                "windowStart": ordered[0]["target"].isoformat() if ordered else None,
                "windowEnd": ordered[-1]["target"].isoformat() if ordered else None,
                "residualCount": n, "meanLogResidual": None, "populationVariance": None,
                "unclippedLogCorrection": None, "logCorrection": 0.0, "multiplicativeFactor": 1.0}
    mean = sum(residuals) / n
    variance = sum((value - mean) ** 2 for value in residuals) / n
    raw_correction = mean - variance
    correction = max(-MAX_ABS_LOG_CORRECTION, min(MAX_ABS_LOG_CORRECTION, raw_correction))
    factor = math.exp(correction)
    prediction = direct * factor
    if (not math.isfinite(mean) or not math.isfinite(variance) or variance < 0 or
            not all(math.isfinite(x) and x > 0 for x in (factor, prediction))):
        raise CalibrationError("calibration produced nonfinite or nonpositive value")
    return {"prediction": prediction, "fallback": False, "fallbackReason": None,
            "windowStart": ordered[0]["target"].isoformat(), "windowEnd": ordered[-1]["target"].isoformat(),
            "residualCount": n, "meanLogResidual": mean, "populationVariance": variance,
            "unclippedLogCorrection": raw_correction, "logCorrection": correction,
            "multiplicativeFactor": factor}


def build_forecast_manifest(report: dict[str, Any], *, direct_report_sha256: str,
                            code_sha256: str, expected_pairs: int | None = EXPECTED_EVAL_PAIRS) -> dict[str, Any]:
    """Create target-value-free forecasts from the direct report alone."""
    if direct_report_sha256 != DIRECT_SHA256:
        raise CalibrationError("direct report SHA-256 differs from frozen protocol")
    rows = _direct_rows(report, validate_actual=False)
    eval_rows = sorted((r for r in rows.values() if EVAL_START <= r["target"] <= EVAL_END),
                       key=lambda r: (r["target"], r["product"], r["horizon"]))
    if expected_pairs is not None and len(eval_rows) != expected_pairs:
        raise CalibrationError(f"evaluation pair count differs from frozen count: {len(eval_rows)}")
    forecasts = []
    for row in eval_rows:
        prior = [candidate for candidate in rows.values()
                 if candidate["product"] == row["product"] and
                 candidate["horizon"] == row["horizon"] and
                 candidate["target"] < row["target"]]
        audit = _correction(prior, row["origin"], row["direct"])
        forecasts.append({"product": row["product"], "horizon": row["horizon"],
                          "origin": row["origin"].isoformat(), "target": row["target"].isoformat(),
                          "direct": row["direct"], "persistence": row["persistence"],
                          "candidate": audit["prediction"], "fallback": audit["fallback"],
                          "fallbackReason": audit["fallbackReason"],
                          "calibration": {k: v for k, v in audit.items() if k != "prediction"}})
    manifest = {"schemaVersion": 1, "status": "answer_free_forecast_manifest",
                "protocol": PROTOCOL, "inputs": {"directReportSha256": direct_report_sha256,
                                                     "codeSha256": code_sha256},
                "targetValuesAttached": False, "expectedPairCount": EXPECTED_EVAL_PAIRS,
                "forecastCount": len(forecasts), "forecasts": forecasts}
    manifest["manifestSha256"] = sha256(canonical(manifest))
    return manifest


def _key(row: dict[str, Any]) -> tuple[str, int, dt.date, dt.date]:
    try:
        p, h = row["product"], row["horizon"]
        o, t = dt.date.fromisoformat(row["origin"]), dt.date.fromisoformat(row["target"])
    except (KeyError, TypeError, ValueError) as exc:
        raise CalibrationError("malformed forecast identity") from exc
    if p not in PRODUCTS or isinstance(h, bool) or not isinstance(h, int) or h not in HORIZONS or t - o != dt.timedelta(days=h):
        raise CalibrationError("forecast identity is outside frozen product/horizon protocol")
    return p, h, o, t


def _validate_forecast(manifest: dict[str, Any], *, direct_report_sha256: str, code_sha256: str,
                       expected_pairs: int | None = EXPECTED_EVAL_PAIRS) -> dict[tuple[str, int, dt.date, dt.date], dict[str, Any]]:
    if manifest.get("status") != "answer_free_forecast_manifest" or manifest.get("targetValuesAttached") is not False:
        raise CalibrationError("forecast manifest must be answer-free")
    if manifest.get("protocol") != PROTOCOL or manifest.get("expectedPairCount") != EXPECTED_EVAL_PAIRS:
        raise CalibrationError("forecast manifest frozen protocol mismatch")
    if manifest.get("inputs") != {"directReportSha256": direct_report_sha256, "codeSha256": code_sha256}:
        raise CalibrationError("forecast manifest input/code hashes mismatch")
    stated = manifest.get("manifestSha256")
    check = dict(manifest)
    check.pop("manifestSha256", None)
    if stated != sha256(canonical(check)):
        raise CalibrationError("forecast manifest SHA-256 mismatch")
    forecasts = manifest.get("forecasts")
    if not isinstance(forecasts, list) or manifest.get("forecastCount") != len(forecasts):
        raise CalibrationError("forecast list/count mismatch")
    mapped = {}
    for row in forecasts:
        if not isinstance(row, dict) or "actual" in row:
            raise CalibrationError("forecast rows must be objects without actual values")
        key = _key(row)
        if key in mapped:
            raise CalibrationError(f"duplicate forecast key: {key}")
        _, _, _, target = key
        if not EVAL_START <= target <= EVAL_END:
            raise CalibrationError("forecast target outside frozen evaluation period")
        direct = _positive(row.get("direct"), "forecast direct")
        candidate = _positive(row.get("candidate"), "forecast candidate")
        persistence = _positive(row.get("persistence"), "forecast persistence")
        calibration = row.get("calibration")
        if not isinstance(calibration, dict) or isinstance(calibration.get("residualCount"), bool) or not isinstance(calibration.get("residualCount"), int):
            raise CalibrationError("malformed calibration audit")
        n = calibration["residualCount"]
        if n < MIN_RESIDUALS:
            if not row.get("fallback") or row.get("fallbackReason") != "fewer_than_30_eligible_residuals" or candidate != direct:
                raise CalibrationError("insufficient-history fallback must equal direct")
        else:
            if row.get("fallback") or row.get("fallbackReason") is not None:
                raise CalibrationError("eligible calibration cannot claim fallback")
            corr = calibration.get("logCorrection")
            if isinstance(corr, bool) or not isinstance(corr, (int, float)) or not math.isfinite(corr) or abs(corr) > MAX_ABS_LOG_CORRECTION + 1e-12:
                raise CalibrationError("calibration correction outside frozen clamp")
            if not math.isclose(candidate, direct * math.exp(corr), rel_tol=1e-12, abs_tol=1e-12):
                raise CalibrationError("candidate prediction does not match its correction")
        mapped[key] = row
    if expected_pairs is not None and len(mapped) != expected_pairs:
        raise CalibrationError(f"forecast pair count differs from frozen count: {len(mapped)}")
    return mapped


def _holm(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(range(len(rows)), key=lambda i: rows[i]["pValue"])
    running = 0.0
    for rank, index in enumerate(ordered):
        running = max(running, min(1.0, (len(rows) - rank) * rows[index]["pValue"]))
        rows[index]["holmPValue"] = running
        rows[index]["holmSignificant"] = running < 0.05


def _metrics(rows: list[dict[str, Any]], field: str) -> dict[str, float | None]:
    if not rows:
        return {"mae": None, "mape": None}
    return {"mae": sum(abs(r["actual"] - r[field]) for r in rows) / len(rows),
            "mape": 100.0 * sum(abs(r["actual"] - r[field]) / r["actual"] for r in rows) / len(rows)}


def score_manifest(forecast_manifest: dict[str, Any], direct_report: dict[str, Any],
                   blend_manifest: dict[str, Any], blend_report: dict[str, Any], *,
                   direct_report_sha256: str, code_sha256: str,
                   expected_pairs: int | None = EXPECTED_EVAL_PAIRS) -> dict[str, Any]:
    """Score only the frozen common keys; caller must verify all source file hashes."""
    direct_rows = _direct_rows(direct_report)
    candidate_rows = _validate_forecast(forecast_manifest, direct_report_sha256=direct_report_sha256,
                                        code_sha256=code_sha256, expected_pairs=expected_pairs)
    if blend_manifest.get("phase") != "forecast_only" or blend_manifest.get("targetValuesAttached") is not False:
        raise CalibrationError("blend manifest is not a forecast-only artifact")
    if blend_report.get("manifestSha256") != blend_manifest.get("manifestSha256"):
        raise CalibrationError("blend report does not refer to the pinned blend manifest")
    if blend_report.get("inputSha256", {}).get("directReport") != DIRECT_SHA256:
        raise CalibrationError("blend report direct-report lineage mismatch")
    blend_rows = blend_manifest.get("forecasts")
    if not isinstance(blend_rows, list):
        raise CalibrationError("blend forecast manifest lacks forecast rows")
    blend_by_key = {}
    for row in blend_rows:
        key = _key(row)
        if key in blend_by_key:
            raise CalibrationError(f"duplicate blend forecast key: {key}")
        if "actual" in row:
            raise CalibrationError("blend forecast manifest must not attach target actuals")
        blend_by_key[key] = row
    expected_keys = {key for key in direct_rows if EVAL_START <= key[3] <= EVAL_END}
    if expected_pairs is not None and len(expected_keys) != expected_pairs:
        raise CalibrationError(f"direct evaluation key count differs from frozen count: {len(expected_keys)}")
    if set(candidate_rows) != expected_keys or set(blend_by_key) != expected_keys:
        raise CalibrationError("direct, candidate, and blend evaluation keys do not match exactly")
    for key, candidate in candidate_rows.items():
        direct = direct_rows[key]
        prior = [row for row in direct_rows.values()
                 if row["product"] == key[0] and row["horizon"] == key[1] and row["target"] < key[3]]
        expected_audit = _correction(prior, direct["origin"], direct["direct"])
        expected_calibration = {k: v for k, v in expected_audit.items() if k != "prediction"}
        if candidate["fallback"] != expected_audit["fallback"] or candidate.get("fallbackReason") != expected_audit["fallbackReason"]:
            raise CalibrationError(f"forecast fallback differs from recomputed history: {key}")
        if not math.isclose(candidate["candidate"], expected_audit["prediction"], rel_tol=1e-12, abs_tol=1e-12):
            raise CalibrationError(f"candidate differs from recomputed correction: {key}")
        stated_calibration = candidate["calibration"]
        if set(stated_calibration) != set(expected_calibration):
            raise CalibrationError(f"calibration audit fields mismatch: {key}")
        for name, expected_value in expected_calibration.items():
            stated_value = stated_calibration[name]
            if isinstance(expected_value, float):
                if (isinstance(stated_value, bool) or not isinstance(stated_value, (int, float)) or
                        not math.isclose(stated_value, expected_value, rel_tol=1e-12, abs_tol=1e-12)):
                    raise CalibrationError(f"calibration audit value mismatch for {name}: {key}")
            elif stated_value != expected_value:
                raise CalibrationError(f"calibration audit value mismatch for {name}: {key}")
    joined = []
    for key in sorted(expected_keys, key=lambda x: (x[3], x[0], x[1])):
        direct = direct_rows[key]
        candidate = candidate_rows[key]
        blend = blend_by_key[key]
        predictions = blend.get("predictions")
        if not isinstance(predictions, dict):
            raise CalibrationError("blend forecast predictions missing")
        if not math.isclose(_positive(predictions.get("direct_ridge"), "blend direct"), direct["direct"], rel_tol=1e-12, abs_tol=1e-12):
            raise CalibrationError(f"direct prediction mismatch against blend manifest: {key}")
        if not math.isclose(candidate["direct"], direct["direct"], rel_tol=1e-12, abs_tol=1e-12):
            raise CalibrationError(f"candidate direct baseline mismatch: {key}")
        if not math.isclose(candidate["persistence"], direct["persistence"], rel_tol=1e-12, abs_tol=1e-12):
            raise CalibrationError(f"candidate persistence mismatch: {key}")
        joined.append({"product": key[0], "horizon": key[1], "origin": key[2], "target": key[3],
                       "actual": direct["actual"], "candidate": candidate["candidate"],
                       "direct": direct["direct"], "blend": _positive(predictions.get("blend"), "blend prediction"),
                       "persistence": direct["persistence"],
                       "fallback": bool(candidate["fallback"]),
                       "residualCount": candidate["calibration"]["residualCount"]})
    fields = ("candidate", "direct", "blend", "persistence")
    cells: dict[str, Any] = {}
    dm_rows = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            part = [r for r in joined if r["product"] == product and r["horizon"] == horizon]
            cell = {"n": len(part), "metrics": {field: _metrics(part, field) for field in fields},
                    "fallbackCount": sum(r["fallback"] for r in part),
                    "fallbackFraction": sum(r["fallback"] for r in part) / len(part) if part else None}
            cells[f"{product}/{horizon}"] = cell
            for comparator in COMPARATORS:
                for loss in LOSSES:
                    diffs = {}
                    for row in part:
                        if loss == "AE":
                            delta = abs(row["actual"] - row["candidate"]) - abs(row["actual"] - row[comparator])
                        else:
                            delta = 100 * (abs(row["actual"] - row["candidate"]) - abs(row["actual"] - row[comparator])) / row["actual"]
                        diffs[row["target"]] = delta
                    tests = {str(bw): direct_horizon._dm(diffs, bw) for bw in BANDWIDTHS}
                    dm_rows.append({"product": product, "horizon": horizon, "comparator": comparator,
                                    "loss": loss, "n": len(part),
                                    "candidateMetric": cell["metrics"]["candidate"]["mae" if loss == "AE" else "mape"],
                                    "comparatorMetric": cell["metrics"][comparator]["mae" if loss == "AE" else "mape"],
                                    "dm": tests})
    if len(dm_rows) != FAMILY_SIZE:
        raise AssertionError("frozen 300-test family changed")
    empty_cells = [key for key, cell in cells.items() if cell["n"] == 0]
    if empty_cells:
        raise CalibrationError(f"evaluation population has empty product/horizon cells: {empty_cells}")
    for bw in BANDWIDTHS:
        _holm([row["dm"][str(bw)] for row in dm_rows])
    macro = {field: {metric: sum(cells[f"{p}/{h}"]["metrics"][field][metric]
                                 for p in PRODUCTS for h in HORIZONS) / (len(PRODUCTS) * len(HORIZONS))
                     for metric in ("mae", "mape")} for field in fields}
    by_product = {p: {field: _metrics([r for r in joined if r["product"] == p], field) for field in fields}
                  for p in PRODUCTS}
    blend_cells = blend_report.get("cellMetrics")
    if not isinstance(blend_cells, list):
        raise CalibrationError("pinned blend report lacks cell metrics")
    blend_cell_map = {}
    for row in blend_cells:
        key = (row.get("product"), row.get("horizon"))
        if key in blend_cell_map or key[0] not in PRODUCTS or key[1] not in HORIZONS:
            raise CalibrationError("invalid or duplicate cell in pinned blend report")
        blend_cell_map[key] = row
    if len(blend_cell_map) != len(PRODUCTS) * len(HORIZONS):
        raise CalibrationError("pinned blend report does not cover all 50 cells")
    for product in PRODUCTS:
        for horizon in HORIZONS:
            published = blend_cell_map[(product, horizon)]
            current = cells[f"{product}/{horizon}"]
            if published.get("plannedPairs") != current["n"]:
                raise CalibrationError(f"pinned blend report cell count mismatch: {product}/{horizon}")
            for blend_field, current_field in (("blend", "blend"), ("direct_ridge", "direct"),
                                               ("persistence", "persistence")):
                expected_metrics = published.get("metrics", {}).get(blend_field, {})
                for metric in ("mae", "mape"):
                    expected_value = expected_metrics.get(metric)
                    current_value = current["metrics"][current_field][metric]
                    if (not isinstance(expected_value, (int, float)) or
                            not math.isclose(expected_value, current_value, rel_tol=1e-9, abs_tol=1e-9)):
                        raise CalibrationError(f"pinned blend report metric mismatch: {product}/{horizon}/{blend_field}/{metric}")
    coverage_counts = {field: len(joined) for field in fields}
    gates = {
        "candidateMacroMapeAtMost10Pct": macro["candidate"]["mape"] <= 10.0,
        "candidateImprovesBlendByAtLeast0_5PctPoints": macro["candidate"]["mape"] <= BLEND_MACRO_MAPE_PCT - 0.5,
        "candidateCoverageAtLeastAllComparators": all(coverage_counts["candidate"] >= coverage_counts[c] for c in COMPARATORS),
        "eachProductMapeWorseningVsBlendAtMost1PctPoint": all(
            by_product[p]["candidate"]["mape"] - by_product[p]["blend"]["mape"] <= 1.0 for p in PRODUCTS),
        "eachProductMaeWorseningVsBlendAtMost5Pct": all(
            by_product[p]["blend"]["mae"] > 0 and
            100 * (by_product[p]["candidate"]["mae"] - by_product[p]["blend"]["mae"]) /
            by_product[p]["blend"]["mae"] <= 5.0 for p in PRODUCTS),
    }
    conclusions = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            tests = [r for r in dm_rows if r["product"] == product and r["horizon"] == horizon]
            primary = [row["dm"]["9"] for row in tests]
            available = all(test["status"] == "tested" for test in primary)
            all_better = all(row["candidateMetric"] < row["comparatorMetric"] for row in tests)
            adverse = any(test.get("holmSignificant") and test.get("meanLossDifference", 0) > 0 for test in primary)
            better = available and all_better and all(test.get("holmSignificant") and test["meanLossDifference"] < 0 for test in primary)
            conclusions.append({"product": product, "horizon": horizon,
                                "conclusion": "worsening" if adverse else "improvement" if better else "inconclusive",
                                "inferenceAvailable": available, "allPointMetricsLower": all_better})
    return {"schemaVersion": 1, "status": "scored", "protocol": PROTOCOL,
            "inputHashes": {"directReportSha256": direct_report_sha256,
                            "blendManifestSha256": BLEND_MANIFEST_SHA256,
                            "blendReportSha256": BLEND_REPORT_SHA256,
                            "forecastManifestSha256": forecast_manifest["manifestSha256"],
                            "codeSha256": code_sha256},
            "pairedKeyCount": len(joined), "byProductHorizon": cells, "macroMetricsEqual50Cells": macro,
            "byProductMetrics": by_product, "coverage": {"pairedRows": len(joined),
                                                            "fallbackRows": sum(r["fallback"] for r in joined),
                                                            "candidate": len(joined), "direct": len(joined),
                                                            "blend": len(joined), "persistence": len(joined)},
            "dm": {"familySizePerBandwidth": FAMILY_SIZE, "bandwidths": list(BANDWIDTHS), "comparisons": dm_rows},
            "conclusions": conclusions, "gates": gates,
            "adoptCandidate": all(gates.values()),
            "interpretation": "Nonsignificance is inconclusive, not equivalence; retrospective revised data are not prospective or official-score evidence."}


def _write_immutable(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def forecast_phase(direct_report_path: Path = DIRECT_REPORT_PATH,
                   output_path: Path = FORECAST_OUTPUT) -> dict[str, Any]:
    raw = direct_report_path.read_bytes()
    digest = sha256(raw)
    if digest != DIRECT_SHA256:
        raise CalibrationError("pinned direct report SHA-256 mismatch")
    report = json.loads(raw)
    code_digest = sha256(Path(__file__).read_bytes())
    manifest = build_forecast_manifest(report, direct_report_sha256=digest, code_sha256=code_digest)
    encoded = canonical(manifest)
    _write_immutable(output_path, encoded)
    return {"output": str(output_path), "sha256": sha256(encoded), "forecastCount": manifest["forecastCount"]}


def score_phase(forecast_path: Path = FORECAST_OUTPUT, direct_report_path: Path = DIRECT_REPORT_PATH,
                blend_manifest_path: Path = BLEND_MANIFEST_PATH, blend_report_path: Path = BLEND_REPORT_PATH,
                output_path: Path = SCORE_OUTPUT) -> dict[str, Any]:
    forecast_raw, direct_raw = forecast_path.read_bytes(), direct_report_path.read_bytes()
    blend_manifest_raw, blend_report_raw = blend_manifest_path.read_bytes(), blend_report_path.read_bytes()
    input_hashes = (sha256(direct_raw), sha256(blend_manifest_raw), sha256(blend_report_raw))
    if input_hashes != (DIRECT_SHA256, BLEND_MANIFEST_SHA256, BLEND_REPORT_SHA256):
        raise CalibrationError("one or more pinned scoring input SHA-256 values mismatch")
    code_digest = sha256(Path(__file__).read_bytes())
    forecast = json.loads(forecast_raw)
    if forecast.get("inputs", {}).get("codeSha256") != code_digest:
        raise CalibrationError("forecast manifest was created by a different source-code hash")
    result = score_manifest(forecast, json.loads(direct_raw), json.loads(blend_manifest_raw),
                            json.loads(blend_report_raw), direct_report_sha256=DIRECT_SHA256,
                            code_sha256=code_digest)
    encoded = canonical(result)
    _write_immutable(output_path, encoded)
    return {"output": str(output_path), "sha256": sha256(encoded), "pairedRows": result["pairedKeyCount"],
            "adoptCandidate": result["adoptCandidate"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    forecast = sub.add_parser("forecast", help="create answer-free manifest; reads direct report only")
    forecast.add_argument("--direct-report", type=Path, default=DIRECT_REPORT_PATH)
    forecast.add_argument("--output", type=Path, default=FORECAST_OUTPUT)
    score = sub.add_parser("score", help="score a prior immutable manifest against pinned evaluation inputs")
    score.add_argument("--forecast-manifest", type=Path, default=FORECAST_OUTPUT)
    score.add_argument("--direct-report", type=Path, default=DIRECT_REPORT_PATH)
    score.add_argument("--blend-manifest", type=Path, default=BLEND_MANIFEST_PATH)
    score.add_argument("--blend-report", type=Path, default=BLEND_REPORT_PATH)
    score.add_argument("--output", type=Path, default=SCORE_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = forecast_phase(args.direct_report, args.output) if args.phase == "forecast" else score_phase(
            args.forecast_manifest, args.direct_report, args.blend_manifest, args.blend_report, args.output)
    except (CalibrationError, OSError, json.JSONDecodeError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
