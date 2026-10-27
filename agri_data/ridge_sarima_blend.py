"""Frozen 50:50 direct Ridge / calendar SARIMA blend; forecast then score.

Forecast reads only the pinned answer-free ten-strategy manifest. Scoring is a
separate command and is the first phase allowed to open truth-bearing reports.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from typing import Any

from . import dashboard

ROOT = Path(__file__).resolve().parents[1]
TEN_MANIFEST_PATH = ROOT / "outputs/experiments/ten-strategy-benchmark-20261008/forecast-manifest.json"
TEN_REPORT_PATH = ROOT / "outputs/experiments/ten-strategy-benchmark-20261008/report.json"
DIRECT_REPORT_PATH = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
TEN_MANIFEST_SHA = "30cc2c86e23027902667d554f8cddcec630172a1019a47b838b954b528adbeea"
TEN_REPORT_SHA = "a63726a72fa9671556926ee87833b416b60f78bcf02247c0b3a813d3b85c2a41"
DIRECT_REPORT_SHA = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
TEN_PROTOCOL = "ten-strategy-same-pair-2026-10-08"
PROTOCOL = "ridge-sarima-fixed-50-50-2026-10-08"
PRODUCTS = tuple(dashboard.PRODUCTS)
HORIZONS = tuple(range(1, 11))
PLANNED = 12_508
TARGET_START = dt.date(2025, 9, 25)
TARGET_END = dt.date(2026, 9, 24)
BANDWIDTHS = (9, 28)
ALPHA = .05
METHODS = ("blend", "direct_ridge", "calendar_sarima_111_100_7", "persistence")
COMPARATORS = METHODS[1:]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_hash(obj: dict[str, Any]) -> str:
    return _sha(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":"), allow_nan=False).encode())


def _finite_positive(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("prediction_not_positive_finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("prediction_not_positive_finite") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError("prediction_not_positive_finite")
    return result


def _date(value: Any) -> dt.date:
    if not isinstance(value, str):
        raise ValueError("date_invalid")
    return dt.date.fromisoformat(value)


def _key(row: dict[str, Any]) -> tuple[str, dt.date, dt.date, int]:
    product = row.get("product")
    horizon = row.get("horizon")
    if product not in PRODUCTS or isinstance(horizon, bool) or horizon not in HORIZONS:
        raise ValueError("forecast_key_invalid")
    target = _date(row.get("target"))
    origin = _date(row["origin"]) if row.get("origin") else target - dt.timedelta(days=horizon)
    if target - origin != dt.timedelta(days=horizon):
        raise ValueError("origin_horizon_mismatch")
    if not TARGET_START <= target <= TARGET_END:
        raise ValueError("target_outside_frozen_period")
    return product, origin, target, horizon


def _key_string(key: tuple[str, dt.date, dt.date, int]) -> str:
    p, origin, target, horizon = key
    return f"{p}|{origin.isoformat()}|{target.isoformat()}|{horizon}"


def _check_source_manifest(doc: dict[str, Any], raw: bytes) -> dict[tuple[str, dt.date, dt.date, int], dict[str, Any]]:
    if _sha(raw) != TEN_MANIFEST_SHA:
        raise ValueError("ten_strategy_manifest_sha_mismatch")
    if doc.get("protocol") != TEN_PROTOCOL or doc.get("phase") != "forecast_only" or doc.get("targetValuesAttached") is not False:
        raise ValueError("source_manifest_not_answer_free")
    forecasts = doc.get("forecasts")
    if not isinstance(forecasts, list):
        raise ValueError("source_manifest_forecasts_missing")
    keyed = {}
    for row in forecasts:
        if not isinstance(row, dict) or "actual" in row or "targetValue" in row:
            raise ValueError("source_manifest_answer_leakage")
        key = _key(row)
        if key in keyed:
            raise ValueError("source_manifest_duplicate_key")
        predictions = row.get("predictions")
        if not isinstance(predictions, dict):
            raise ValueError("source_predictions_missing")
        for name in ("direct_ridge", "calendar_sarima_111_100_7", "naive"):
            _finite_positive(predictions.get(name))
        fallback_map = row.get("fallback", {})
        status_map = row.get("methodStatus", {})
        status = status_map.get("calendar_sarima_111_100_7", {})
        if "fallback" in status and "calendar_sarima_111_100_7" in fallback_map and bool(status["fallback"]) != bool(fallback_map["calendar_sarima_111_100_7"]):
            raise ValueError("sarima_fallback_lineage_mismatch")
        native = row.get("nativePredictions", {}).get("calendar_sarima_111_100_7")
        if native is not None:
            native_value = _finite_positive(native)
            if not bool(status.get("fallback", fallback_map.get("calendar_sarima_111_100_7", False))) and not math.isclose(native_value, float(predictions["calendar_sarima_111_100_7"]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("sarima_native_prediction_lineage_mismatch")
        keyed[key] = row
    if len(keyed) != PLANNED:
        raise ValueError(f"source_manifest_key_count_mismatch:{len(keyed)}")
    return keyed


def build_forecast(source_raw: bytes) -> dict[str, Any]:
    """Build the answer-free blend manifest; never accepts truth-bearing inputs."""
    source = json.loads(source_raw)
    rows = _check_source_manifest(source, source_raw)
    forecasts = []
    for key in sorted(rows):
        row = rows[key]
        predictions = row["predictions"]
        status = row.get("methodStatus", {}).get("calendar_sarima_111_100_7", {})
        sarima_fallback = bool(status.get("fallback", row.get("fallback", {}).get("calendar_sarima_111_100_7", False)))
        direct = _finite_positive(predictions["direct_ridge"])
        sarima = _finite_positive(predictions["calendar_sarima_111_100_7"])
        persistence = _finite_positive(predictions["naive"])
        blend = direct if sarima_fallback else .5 * direct + .5 * sarima
        forecasts.append({"product": key[0], "origin": key[1].isoformat(),
                          "target": key[2].isoformat(), "horizon": key[3],
                          "predictions": {"blend": blend, "direct_ridge": direct,
                                          "calendar_sarima_111_100_7": sarima,
                                          "persistence": persistence},
                          "lineage": {"direct_ridge": "ten_strategy_manifest.predictions.direct_ridge",
                                      "sarima": "ten_strategy_manifest.predictions.calendar_sarima_111_100_7",
                                      "sarimaFallback": sarima_fallback,
                                      "sarimaFallbackReason": status.get("fallbackReason"),
                                      "sarimaNativePrediction": row.get("nativePredictions", {}).get("calendar_sarima_111_100_7"),
                                      "blendRule": "direct fallback if SARIMA fallback; else arithmetic 0.5/0.5"}})
    manifest = {"schemaVersion": 1, "protocol": PROTOCOL, "phase": "forecast_only",
                "targetValuesAttached": False, "sourceManifestSha256": _sha(source_raw),
                "codeSha256": _sha(Path(__file__).read_bytes()),
                "evaluation": {"targetStart": TARGET_START.isoformat(), "targetEnd": TARGET_END.isoformat(),
                               "plannedPairs": PLANNED, "products": list(PRODUCTS),
                               "horizons": list(HORIZONS)},
                "formula": "0.5*direct_ridge + 0.5*calendar_sarima_111_100_7; direct fallback if SARIMA fallback",
                "forecasts": forecasts}
    manifest["manifestSha256"] = _canonical_hash(manifest)
    return manifest


def _immutable_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(doc, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _loss(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    n = len(rows)
    return {"n": n,
            "mae": sum(abs(row["actual"] - row["predictions"][method]) for row in rows) / n if n else None,
            "mape": 100 * sum(abs(row["actual"] - row["predictions"][method]) / row["actual"] for row in rows) / n if n else None}


def _dm(diffs: list[tuple[dt.date, float]], lag: int) -> dict[str, Any]:
    n = len(diffs)
    mean = sum(value for _, value in diffs) / n if n else None
    base = {"n": n, "meanLossDifference": mean, "calendarHacMaxLagDays": lag}
    if n < 30:
        return {**base, "status": "inference_unavailable_insufficient_pairs", "statistic": None, "pValue": 1.0}
    centered = {day: value - mean for day, value in diffs}
    variance = sum(value * value for value in centered.values())
    for k in range(1, lag + 1):
        variance += 2 * (1 - k / (lag + 1)) * sum(value * centered.get(day - dt.timedelta(days=k), 0.0)
                                                    for day, value in centered.items())
    if not math.isfinite(variance) or variance <= 0:
        return {**base, "status": "inference_unavailable_nonpositive_variance", "statistic": None, "pValue": 1.0}
    z = mean / (math.sqrt(variance) / n)
    return {**base, "status": "tested", "statistic": z,
            "pValue": math.erfc(abs(z) / math.sqrt(2))}


def _holm(tests: list[dict[str, Any]]) -> None:
    ordered = sorted(tests, key=lambda item: (item["pValue"], item["product"], item["horizon"], item["comparator"], item["loss"]))
    adjusted = 0.0
    for i, item in enumerate(ordered):
        adjusted = max(adjusted, min(1.0, (len(ordered) - i) * item["pValue"]))
        item["holmPValue"] = adjusted
        item["holmSignificant"] = item["status"] == "tested" and adjusted < ALPHA


def _verify_forecast(m: dict[str, Any], manifest_raw: bytes) -> dict[tuple[str, dt.date, dt.date, int], dict[str, Any]]:
    if m.get("protocol") != PROTOCOL or m.get("phase") != "forecast_only" or m.get("targetValuesAttached") is not False:
        raise ValueError("blend_manifest_not_answer_free")
    if m.get("sourceManifestSha256") != TEN_MANIFEST_SHA or _sha(manifest_raw) != _sha(json.dumps(m, ensure_ascii=False, sort_keys=True, indent=2).encode() + b"\n"):
        raise ValueError("blend_manifest_source_or_bytes_mismatch")
    clean = dict(m)
    saved = clean.pop("manifestSha256", None)
    if saved != _canonical_hash(clean) or saved != _canonical_hash({k: v for k, v in m.items() if k != "manifestSha256"}):
        raise ValueError("blend_manifest_hash_mismatch")
    if m.get("codeSha256") != _sha(Path(__file__).read_bytes()):
        raise ValueError("blend_manifest_code_hash_mismatch")
    keyed = {}
    for row in m.get("forecasts", []):
        if "actual" in row or "targetValue" in row:
            raise ValueError("blend_manifest_answer_leakage")
        key = _key(row)
        if key in keyed:
            raise ValueError("blend_manifest_duplicate_key")
        if set(row.get("predictions", {})) != set(METHODS):
            raise ValueError("blend_manifest_methods_invalid")
        for method in METHODS:
            _finite_positive(row["predictions"][method])
        keyed[key] = row
    if len(keyed) != PLANNED:
        raise ValueError("blend_manifest_key_count_mismatch")
    return keyed


def _validate_source_lineage(forecasts: dict[tuple[str, dt.date, dt.date, int], dict[str, Any]],
                             source_rows: dict[tuple[str, dt.date, dt.date, int], dict[str, Any]]) -> None:
    """Verify every emitted component, fallback state, and blend rule pre-score."""
    for key, forecast in forecasts.items():
        source = source_rows[key]
        source_predictions = source["predictions"]
        output_predictions = forecast["predictions"]
        status = source.get("methodStatus", {}).get("calendar_sarima_111_100_7", {})
        fallback_map = source.get("fallback", {})
        sarima_fallback = bool(status.get("fallback", fallback_map.get("calendar_sarima_111_100_7", False)))
        expected = {
            "direct_ridge": _finite_positive(source_predictions["direct_ridge"]),
            "calendar_sarima_111_100_7": _finite_positive(source_predictions["calendar_sarima_111_100_7"]),
            "persistence": _finite_positive(source_predictions["naive"]),
        }
        for method, value in expected.items():
            if not math.isclose(_finite_positive(output_predictions.get(method)), value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"manifest_source_component_mismatch:{method}")
        expected_blend = expected["direct_ridge"] if sarima_fallback else .5 * expected["direct_ridge"] + .5 * expected["calendar_sarima_111_100_7"]
        if not math.isclose(_finite_positive(output_predictions.get("blend")), expected_blend, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("manifest_blend_rule_mismatch")
        lineage = forecast.get("lineage", {})
        if lineage.get("sarimaFallback") is not sarima_fallback:
            raise ValueError("manifest_sarima_fallback_state_mismatch")
        if lineage.get("sarimaFallbackReason") != status.get("fallbackReason"):
            raise ValueError("manifest_sarima_fallback_reason_mismatch")
        source_native = source.get("nativePredictions", {}).get("calendar_sarima_111_100_7")
        emitted_native = lineage.get("sarimaNativePrediction")
        if source_native is None:
            if emitted_native is not None:
                raise ValueError("manifest_sarima_native_state_mismatch")
        elif not math.isclose(_finite_positive(emitted_native), _finite_positive(source_native), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("manifest_sarima_native_value_mismatch")
        if not sarima_fallback and source_native is None:
            raise ValueError("source_sarima_native_missing_without_fallback")
        if lineage.get("blendRule") != "direct fallback if SARIMA fallback; else arithmetic 0.5/0.5":
            raise ValueError("manifest_blend_lineage_invalid")


def _report_rows(doc: dict[str, Any], name: str) -> dict[tuple[str, dt.date, dt.date, int], dict[str, Any]]:
    rows = doc.get("pairedRows")
    if not isinstance(rows, list):
        raise ValueError(f"{name}_paired_rows_missing")
    result = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("_summary"):
            continue
        target = _date(row.get("target"))
        if not TARGET_START <= target <= TARGET_END:
            continue
        key = _key(row)
        if key in result:
            raise ValueError(f"{name}_duplicate_key")
        result[key] = row
    return result


def _report_summary_check(report: dict[str, Any], cells: dict[tuple[str, int], dict[str, Any]]) -> None:
    report_cells = {(item.get("product"), item.get("horizon")): item for item in report.get("cellMetrics", [])}
    if len(report_cells) != len(PRODUCTS) * len(HORIZONS):
        raise ValueError("ten_strategy_report_cells_invalid")
    for cell, values in cells.items():
        reference = report_cells.get(cell)
        if reference is None:
            raise ValueError("ten_strategy_report_cell_missing")
        for method, report_method in (("direct_ridge", "direct_ridge"),
                                      ("calendar_sarima_111_100_7", "calendar_sarima_111_100_7"),
                                      ("persistence", "naive")):
            got, expected = values[method], reference["metrics"][report_method]
            if got["n"] != expected["n"]:
                raise ValueError("ten_strategy_report_pair_count_mismatch")
            for field in ("mae", "mape"):
                if not math.isclose(got[field], expected[field], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError("ten_strategy_report_metric_mismatch")


def score(manifest: dict[str, Any], manifest_raw: bytes, source_raw: bytes,
          ten_report_raw: bytes, direct_raw: bytes) -> dict[str, Any]:
    forecasts = _verify_forecast(manifest, manifest_raw)
    source_rows = _check_source_manifest(json.loads(source_raw), source_raw)
    if set(source_rows) != set(forecasts):
        raise ValueError("source_manifest_key_set_mismatch")
    _validate_source_lineage(forecasts, source_rows)
    if _sha(ten_report_raw) != TEN_REPORT_SHA or _sha(direct_raw) != DIRECT_REPORT_SHA:
        raise ValueError("pinned_scoring_report_sha_mismatch")
    ten_report, direct_report = json.loads(ten_report_raw), json.loads(direct_raw)
    if ten_report.get("protocol") != TEN_PROTOCOL:
        raise ValueError("ten_strategy_report_protocol_mismatch")
    direct_map = _report_rows(direct_report, "direct")
    if set(direct_map) != set(forecasts):
        raise ValueError("direct_report_key_set_mismatch")
    scored = []
    for key, forecast in forecasts.items():
        direct = direct_map[key]
        actual = _finite_positive(direct.get("actual"))
        direct_prediction = _finite_positive(direct.get("direct"))
        persistence = _finite_positive(direct.get("persistence"))
        if not math.isclose(direct_prediction, forecast["predictions"]["direct_ridge"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("actual_direct_concordance_failed")
        if not math.isclose(persistence, forecast["predictions"]["persistence"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("persistence_concordance_failed")
        scored.append({**forecast, "actual": actual})
    if len(scored) != PLANNED:
        raise ValueError("score_pair_count_mismatch")
    cells = {}
    tests = {band: [] for band in BANDWIDTHS}
    cell_metrics = []
    macro = {method: [] for method in METHODS}
    native_sarima_rows = [r for r in scored if not r["lineage"]["sarimaFallback"]]
    for product in PRODUCTS:
        for horizon in HORIZONS:
            group = [r for r in scored if r["product"] == product and r["horizon"] == horizon]
            metrics = {method: _loss(group, method) for method in METHODS}
            for method in METHODS:
                macro[method].append(metrics[method])
            cells[(product, horizon)] = metrics
            cell_metrics.append({"product": product, "horizon": horizon, "plannedPairs": len(group), "metrics": metrics,
                                 "sarimaNativePairs": sum(not r["lineage"]["sarimaFallback"] for r in group)})
            for comparator in COMPARATORS:
                for loss_name in ("AE", "APE"):
                    diffs = []
                    for row in group:
                        y = row["actual"]
                        pred = row["predictions"]["blend"]
                        base = row["predictions"][comparator]
                        delta = abs(y - pred) - abs(y - base) if loss_name == "AE" else 100 * (abs(y-pred)/y - abs(y-base)/y)
                        diffs.append((_date(row["target"]), delta))
                    for band in BANDWIDTHS:
                        tests[band].append({"product": product, "horizon": horizon,
                                            "comparator": comparator, "loss": loss_name,
                                            **_dm(diffs, band)})
    for band in BANDWIDTHS:
        _holm(tests[band])
    _report_summary_check(ten_report, cells)
    macro_metrics = {method: {"meanOf50CellMape": sum(x["mape"] for x in values) / len(values),
                              "meanOf50CellMae": sum(x["mae"] for x in values) / len(values)}
                     for method, values in macro.items()}
    native_sarima_metrics = {method: _loss(native_sarima_rows, method) for method in METHODS[1:]} if native_sarima_rows else {}
    conclusions = []
    hac9 = {(t["product"], t["horizon"], t["comparator"], t["loss"]): t for t in tests[9]}
    for product, horizon in ((p, h) for p in PRODUCTS for h in HORIZONS):
        for comparator in COMPARATORS:
            ae = hac9[(product, horizon, comparator, "AE")]
            ape = hac9[(product, horizon, comparator, "APE")]
            if ae["status"] != "tested" or ape["status"] != "tested":
                verdict = "inference_unavailable"
            elif (ae["holmSignificant"] and ape["holmSignificant"] and
                  ae["meanLossDifference"] < 0 and ape["meanLossDifference"] < 0):
                verdict = "improvement"
            elif ((ae["holmSignificant"] and ae["meanLossDifference"] > 0) or
                  (ape["holmSignificant"] and ape["meanLossDifference"] > 0)):
                verdict = "worsening_or_mixed"
            else:
                verdict = "inconclusive"
            conclusions.append({"product": product, "horizon": horizon,
                                "comparator": comparator, "HAC9": verdict,
                                "ae": ae, "ape": ape})
    by_product = {}
    for product in PRODUCTS:
        groups = [r for r in scored if r["product"] == product]
        by_product[product] = {"blend": _loss(groups, "blend"), "direct_ridge": _loss(groups, "direct_ridge")}
    cell_joint = {}
    for comparator in ("direct_ridge", "calendar_sarima_111_100_7"):
        cell_joint[comparator] = any(
            (t := hac9[(p, h, comparator, "AE")])["holmSignificant"] and t["meanLossDifference"] < 0
            and (u := hac9[(p, h, comparator, "APE")])["holmSignificant"] and u["meanLossDifference"] < 0
            and cells[(p, h)]["blend"]["mae"] < cells[(p, h)][comparator]["mae"]
            and cells[(p, h)]["blend"]["mape"] < cells[(p, h)][comparator]["mape"]
            for p in PRODUCTS for h in HORIZONS)
    gates = {
        "blendCoverageAtLeastDirect": len(scored) >= len(scored),
        "macroMapeAtMost10Percent": macro_metrics["blend"]["meanOf50CellMape"] <= 10,
        "macroMapeImprovesDirectByAtLeast0_5pp": macro_metrics["direct_ridge"]["meanOf50CellMape"] - macro_metrics["blend"]["meanOf50CellMape"] >= .5,
        "eachProductDirectMapeWorseningAtMost1pp": all(by_product[p]["blend"]["mape"] - by_product[p]["direct_ridge"]["mape"] <= 1 for p in PRODUCTS),
        "eachProductDirectMaeWorseningAtMost5Percent": all(by_product[p]["blend"]["mae"] <= by_product[p]["direct_ridge"]["mae"] * 1.05 for p in PRODUCTS),
        "significantJointImprovementCellVsDirect": cell_joint["direct_ridge"],
        "significantJointImprovementCellVsSarima": cell_joint["calendar_sarima_111_100_7"],
    }
    return {"schemaVersion": 1, "protocol": PROTOCOL, "manifestSha256": manifest["manifestSha256"],
            "inputSha256": {"tenStrategyManifest": _sha(source_raw),
                            "tenStrategyReport": _sha(ten_report_raw), "directReport": _sha(direct_raw)},
            "coverage": {"plannedPairs": PLANNED, "scoredPairs": len(scored),
                         "byMethodForecasted": {method: PLANNED for method in METHODS},
                         "commonScoredPairs": len(scored),
                         "sarimaFallbackPairs": PLANNED - len(native_sarima_rows),
                         "sarimaNativePairs": len(native_sarima_rows)},
            "cellMetrics": cell_metrics, "macroMetrics": macro_metrics,
            "sarimaNativePairMetrics": native_sarima_metrics, "productMetrics": by_product,
            "dmTests": {**{f"HAC{band}": tests[band] for band in BANDWIDTHS},
                        "familySizePerBandwidth": len(PRODUCTS) * len(HORIZONS) * len(COMPARATORS) * 2,
                        "holmAlpha": ALPHA},
            "comparisonConclusions": conclusions, "gates": gates,
            "expansionEligible": all(gates.values()),
            "interpretation": "HAC9 primary; HAC28 sensitivity. Nonsignificance is not equivalence. Retrospective revised proxy-price results do not establish official or prospective user benefit."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    forecast = sub.add_parser("forecast")
    forecast.add_argument("--source-manifest", type=Path, default=TEN_MANIFEST_PATH)
    forecast.add_argument("--output", type=Path, required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--source-manifest", type=Path, default=TEN_MANIFEST_PATH)
    scoring.add_argument("--manifest", type=Path, required=True)
    scoring.add_argument("--ten-report", type=Path, default=TEN_REPORT_PATH)
    scoring.add_argument("--direct-report", type=Path, default=DIRECT_REPORT_PATH)
    scoring.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    source_raw = args.source_manifest.read_bytes()
    if args.command == "forecast":
        manifest = build_forecast(source_raw)
        _immutable_json(args.output, manifest)
        print(json.dumps({"manifest": str(args.output), "pairs": len(manifest["forecasts"])}))
    else:
        manifest_raw = args.manifest.read_bytes()
        result = score(json.loads(manifest_raw), manifest_raw, source_raw,
                       args.ten_report.read_bytes(), args.direct_report.read_bytes())
        _immutable_json(args.output, result)
        print(json.dumps({"report": str(args.output), "pairs": result["coverage"]["scoredPairs"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
