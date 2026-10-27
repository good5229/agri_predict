"""Frozen offline median ensemble over previously generated national forecasts."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import direct_horizon, national_price_selection as base

PROTOCOL_ID = "national-median-ensemble-2026-10-09"
PRODUCTS = base.PRODUCTS
HORIZONS = base.HORIZONS
COMPONENTS = ("ridge_alpha_100", "calendar_sarima", "persistence")
METHODS = ("median_ensemble", *COMPONENTS)
COMPARATORS = COMPONENTS
BANDWIDTHS = (9, 28)
MIN_PAIRS = 30
WINDOWS = ((dt.date(2026, 1, 1), dt.date(2026, 3, 31)),
           (dt.date(2026, 4, 1), dt.date(2026, 6, 30)),
           (dt.date(2026, 7, 1), dt.date(2026, 9, 30)))
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


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hash(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _positive(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def _date(value: str) -> dt.date:
    if not isinstance(value, str):
        raise ValueError("date must be ISO text")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("date must be ISO text") from exc


def _load_frozen_manifest(kind: str, path: str | Path | None = None) -> dict:
    if kind not in MANIFEST_SHA256:
        raise ValueError("unsupported input manifest kind")
    source_path = Path(path) if path is not None else PROJECT_ROOT / MANIFEST_PATHS[kind]
    raw = source_path.read_bytes()
    if _sha(raw) != MANIFEST_SHA256[kind]:
        raise ValueError(f"{kind} manifest byte SHA-256 mismatch")
    doc = json.loads(raw)
    if doc.get("manifestSha256") != MANIFEST_CONTENT_SHA256[kind]:
        raise ValueError(f"{kind} manifest canonical SHA-256 mismatch")
    body = dict(doc)
    body.pop("manifestSha256", None)
    if _hash(body) != doc["manifestSha256"]:
        raise ValueError(f"{kind} manifest content hash mismatch")
    base._verify_manifest(doc)
    return doc


def _load_sources(raw_bundles: dict) -> tuple[dict, dict]:
    if not isinstance(raw_bundles, dict) or set(raw_bundles) != {"development", "fresh"}:
        raise ValueError("raw_bundles must contain development and fresh bundles")
    dev_bundle, fresh_bundle = raw_bundles["development"], raw_bundles["fresh"]
    if not isinstance(dev_bundle, dict) or set(dev_bundle) != {"archivePath"}:
        raise ValueError("development source bundle must contain only archivePath")
    if not isinstance(fresh_bundle, dict) or set(fresh_bundle) != {"archivePath", "coverageReportPath"}:
        raise ValueError("fresh source bundle must contain archivePath and coverageReportPath")
    dev_source = base._load_source(dev_bundle)
    fresh_source = base._load_source(fresh_bundle)
    if dev_source["sourceDigests"].get("archiveSha256") != fresh_source["sourceDigests"].get("archiveSha256"):
        raise ValueError("development and fresh sources do not share the same archive")
    return dev_source, fresh_source


def _forecast_identity(row: dict) -> tuple:
    key = tuple(row.get(k) for k in ("product", "origin", "target", "horizon"))
    product, origin_text, target_text, horizon = key
    if product not in PRODUCTS or isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
        raise ValueError("forecast product/horizon key invalid")
    origin, target = _date(origin_text), _date(target_text)
    if target - origin != dt.timedelta(days=horizon):
        raise ValueError("forecast target-origin/horizon mismatch")
    if row.get("cutoff") != (origin - dt.timedelta(days=2)).isoformat():
        raise ValueError("forecast cutoff mismatch")
    return key


def _merge_forecasts(development: dict, fresh: dict) -> list[dict]:
    keyed = {}
    for row in [*development["forecasts"], *fresh["forecasts"]]:
        key = _forecast_identity(row)
        if key in keyed:
            raise ValueError("duplicate forecast identity across pinned manifests")
        predictions = row.get("predictions")
        if not isinstance(predictions, dict) or not set(COMPONENTS).issubset(predictions):
            raise ValueError("forecast is missing a required ensemble component")
        keyed[key] = row
    expected = {(p, (target - dt.timedelta(days=h)).isoformat(), target.isoformat(), h)
                for target in (dt.date(2026, 1, 1) + dt.timedelta(days=i) for i in range(273))
                for p in PRODUCTS for h in HORIZONS}
    observed = {key for key in keyed if dt.date(2026, 1, 1) <= _date(key[2]) <= dt.date(2026, 9, 30)}
    if observed != expected:
        raise ValueError(f"planned forecast key grid mismatch: expected {len(expected)}, got {len(observed)}")
    return [keyed[k] for k in sorted(keyed) if dt.date(2026, 1, 1) <= _date(k[2]) <= dt.date(2026, 9, 30)]


def _compose_forecast(row: dict) -> dict:
    predictions = row["predictions"]
    values = {name: _positive(predictions.get(name)) for name in COMPONENTS}
    invalid = [name for name in COMPONENTS if values[name] is None]
    ridge = values["ridge_alpha_100"]
    if invalid:
        candidate = ridge
        audit = {"fallback": True,
                 "fallbackReason": "invalid_component:" + ",".join(invalid),
                 "componentValues": values}
    else:
        candidate = float(statistics.median(values[name] for name in COMPONENTS))
        audit = {"fallback": False, "fallbackReason": None, "componentValues": values}
    return {**{k: row[k] for k in ("product", "origin", "target", "horizon", "cutoff")},
            "predictions": {**predictions, "median_ensemble": candidate},
            "sourceLineage": row.get("lineage", {}), "lineage": audit}


def build_manifest(development: dict, fresh: dict, raw_bundles: dict) -> dict:
    """Create answer-free medians using the already frozen component forecasts."""
    dev_source, fresh_source = _load_sources(raw_bundles)
    for kind, doc, source in (("development", development, dev_source), ("fresh", fresh, fresh_source)):
        # The caller may pass parsed objects, but only the pinned files are accepted.
        frozen = _load_frozen_manifest(kind)
        if doc != frozen:
            raise ValueError(f"{kind} manifest differs from pinned input")
        if doc.get("sourceDigests") != source.get("sourceDigests"):
            raise ValueError(f"{kind} source digest mismatch")
    rows = _merge_forecasts(development, fresh)
    forecasts = [_compose_forecast(row) for row in rows]
    source_digests = {"development": dev_source["sourceDigests"], "fresh": fresh_source["sourceDigests"]}
    doc = {"schemaVersion": 1, "protocolId": PROTOCOL_ID, "phase": "forecast_only",
           "targetValuesAttached": False, "inputManifestSha256": dict(MANIFEST_SHA256),
           "inputManifestContentSha256": dict(MANIFEST_CONTENT_SHA256), "sourceDigests": source_digests,
           "codeSha256": _sha(Path(__file__).read_bytes()), "products": list(PRODUCTS),
           "horizons": list(HORIZONS), "window": ["2026-01-01", "2026-09-30"],
           "methods": list(METHODS), "plannedForecastCount": len(forecasts), "forecasts": forecasts}
    doc["manifestSha256"] = _hash(doc)
    return doc


def verify_manifest(doc: dict) -> None:
    if doc.get("protocolId") != PROTOCOL_ID or doc.get("phase") != "forecast_only" or doc.get("targetValuesAttached") is not False:
        raise ValueError("not an answer-free median manifest")
    if doc.get("codeSha256") != _sha(Path(__file__).read_bytes()):
        raise ValueError("candidate code hash mismatch")
    body = dict(doc)
    saved = body.pop("manifestSha256", None)
    if saved != _hash(body):
        raise ValueError("candidate manifest hash mismatch")
    rows = doc.get("forecasts")
    if not isinstance(rows, list) or len(rows) != doc.get("plannedForecastCount"):
        raise ValueError("candidate planned count mismatch")
    seen = set()
    for row in rows:
        key = _forecast_identity(row)
        if key in seen:
            raise ValueError("duplicate candidate forecast key")
        seen.add(key)
        if "actual" in row or "targetValue" in row:
            raise ValueError("answer-bearing field in candidate manifest")


def _verify_replay(doc: dict, development: dict, fresh: dict, raw_bundles: dict) -> None:
    verify_manifest(doc)
    rebuilt = build_manifest(development, fresh, raw_bundles)
    if doc.get("forecasts") != rebuilt.get("forecasts") or doc.get("manifestSha256") != rebuilt.get("manifestSha256"):
        raise ValueError("median candidate manifest replay mismatch")


def _dm(diffs: dict[dt.date, float], bandwidth: int) -> dict:
    n = len(diffs)
    mean = float(np.mean(list(diffs.values()))) if n else None
    if n < MIN_PAIRS:
        return {"n": n, "status": "inference_unavailable_insufficient_pairs", "meanLossDifference": mean,
                "statistic": None, "pValue": 1.0, "calendarHacMaxLagDays": bandwidth}
    values = list(diffs.values())
    if all(value == values[0] for value in values[1:]):
        return {"n": n, "status": "inference_unavailable_nonpositive_variance", "meanLossDifference": mean,
                "statistic": None, "pValue": 1.0, "calendarHacMaxLagDays": bandwidth}
    centered = {day: value - mean for day, value in diffs.items()}
    variance = sum(value * value for value in centered.values())
    for lag in range(1, bandwidth + 1):
        covariance = sum(value * centered.get(day - dt.timedelta(days=lag), 0.0)
                         for day, value in centered.items())
        variance += 2 * (1 - lag / (bandwidth + 1)) * covariance
    if not math.isfinite(variance) or variance <= 0:
        return {"n": n, "status": "inference_unavailable_nonpositive_variance", "meanLossDifference": mean,
                "statistic": None, "pValue": 1.0, "calendarHacMaxLagDays": bandwidth}
    statistic = mean / (math.sqrt(variance) / n)
    return {"n": n, "status": "available", "meanLossDifference": mean, "statistic": statistic,
            "pValue": math.erfc(abs(statistic) / math.sqrt(2)), "calendarHacMaxLagDays": bandwidth}


def _holm_family(records: list[dict], bandwidth: int) -> None:
    tests = [record["dm"][str(bandwidth)] for record in records]
    running = 0.0
    for rank, test in enumerate(sorted(tests, key=lambda x: x["pValue"])):
        running = max(running, min(1.0, (len(tests) - rank) * test["pValue"]))
        test["holmAdjustedPValue"] = running
        test["holmSignificant"] = test["status"] == "available" and running < 0.05
        test["conclusion"] = ("inference_unavailable" if test["status"] != "available" else
                              "improvement" if test["holmSignificant"] and test["meanLossDifference"] < 0 else
                              "worsening" if test["holmSignificant"] and test["meanLossDifference"] > 0 else "inconclusive")


def _joint_verdicts(tests: list[dict]) -> list[dict]:
    index = {(r["product"], r["horizon"], r["comparator"], r["loss"]): r for r in tests}
    output = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            for comparator in COMPARATORS:
                ae = index[(product, horizon, comparator, "mae")]
                ape = index[(product, horizon, comparator, "mape")]
                row = {"product": product, "horizon": horizon, "comparator": comparator,
                       "nMae": ae["n"], "nMape": ape["n"], "verdictByBandwidth": {}}
                for bw in BANDWIDTHS:
                    a, p = ae["dm"][str(bw)]["conclusion"], ape["dm"][str(bw)]["conclusion"]
                    joint = (a if a == p and a in ("improvement", "worsening") else
                             "inference_unavailable" if "inference_unavailable" in (a, p) else "inconclusive")
                    row["verdictByBandwidth"][str(bw)] = {"mae": a, "mape": p, "joint": joint}
                output.append(row)
    return output


def _period_gate(complete: bool, macro: dict, per_crop: dict, coverage: dict, joint: list[dict]) -> dict:
    candidate, baseline = "median_ensemble", "ridge_alpha_100"
    gain = baseline_mape - candidate_mape if (baseline_mape := macro[baseline]["mape"]) is not None and (candidate_mape := macro[candidate]["mape"]) is not None else None
    baseline_mae, candidate_mae = macro[baseline]["mae"], macro[candidate]["mae"]
    mae_change = 100 * (candidate_mae - baseline_mae) / baseline_mae if candidate_mae is not None and baseline_mae not in (None, 0) else None
    crop_mape = {p: per_crop[p][candidate]["equalHorizonMape"] - per_crop[p][baseline]["equalHorizonMape"]
                 if per_crop[p][candidate]["equalHorizonMape"] is not None and per_crop[p][baseline]["equalHorizonMape"] is not None else None for p in PRODUCTS}
    crop_mae = {p: 100 * (per_crop[p][candidate]["equalHorizonMae"] - per_crop[p][baseline]["equalHorizonMae"])
                / per_crop[p][baseline]["equalHorizonMae"] if per_crop[p][candidate]["equalHorizonMae"] is not None
                and per_crop[p][baseline]["equalHorizonMae"] not in (None, 0) else None for p in PRODUCTS}
    jointly_improved = {(r["product"], r["horizon"]) for r in joint if r["comparator"] == baseline
                        and r["verdictByBandwidth"]["9"]["joint"] == "improvement"
                        and r["verdictByBandwidth"]["28"]["joint"] == "improvement"}
    criteria = {"macroMapeGainAtLeast0_5pp": gain is not None and gain >= 0.5,
                "issuedCoverageNotDecreased": coverage[candidate]["issued"] >= coverage[baseline]["issued"],
                "eachCropMapeWorseningAtMost1pp": all(v is not None and v <= 1.0 for v in crop_mape.values()),
                "eachCropMaeWorseningAtMost5pct": all(v is not None and v <= 5.0 for v in crop_mae.values()),
                "atLeastOneCellJointlyImprovedAtHac9AndHac28": bool(jointly_improved)}
    return {"status": "unsupported" if not complete else "pass" if all(criteria.values()) else "fail",
            "criteria": criteria, "failureReasons": [] if complete and all(criteria.values()) else
                (["incomplete_40_cell_coverage"] if not complete else [k for k, v in criteria.items() if not v]),
            "thresholds": {"macroMapeGainPpMinimum": 0.5, "cropMapeWorseningPpMaximum": 1.0,
                           "cropMaeWorseningPctMaximum": 5.0, "jointImprovementBandwidthDays": [9, 28],
                           "jointImprovementCellsMinimum": 1}, "macroMapeGainPp": gain,
            "productMapeChangePp": crop_mape, "productMaeChangePct": crop_mae,
            "jointImprovedCells": len(jointly_improved)}


def _exclusions(selected: list[dict], source: dict, actuals: dict, status_evidence: dict,
                common_keys: set[tuple]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in selected:
        key = _forecast_identity(row)
        if _date(row["target"]) not in actuals[row["product"]]:
            state, _ = base._status(source, row["product"], _date(row["target"]), status_evidence)
            counts[state] += 1
        elif key not in common_keys:
            counts["incomplete_common_forecast"] += 1
    return dict(counts)


def score_manifest(doc: dict, development: dict, fresh: dict, raw_bundles: dict) -> dict:
    """Replay the fixed median manifest, then attach outcomes from the reparsed source."""
    _verify_replay(doc, development, fresh, raw_bundles)
    _, source = _load_sources(raw_bundles)
    actuals = base._prices(source)
    evidence = base._coverage_status_evidence(raw_bundles["fresh"])
    report = {"protocolId": PROTOCOL_ID, "candidateManifestSha256": doc["manifestSha256"], "periods": {}}
    for start, end in WINDOWS:
        selected = [r for r in doc["forecasts"] if start <= _date(r["target"]) <= end]
        common = []
        for row in selected:
            if _date(row["target"]) not in actuals[row["product"]]:
                continue
            if all(_positive(row["predictions"].get(method)) is not None for method in METHODS):
                common.append({**row, "actual": actuals[row["product"]][_date(row["target"])]})
        common_keys = {_forecast_identity(r) for r in common}
        cells = {}
        for product in PRODUCTS:
            for horizon in HORIZONS:
                cell = [r for r in common if r["product"] == product and r["horizon"] == horizon]
                cells[f"{product}/{horizon}"] = {"n": len(cell), "metrics": {
                    method: direct_horizon._metrics([r["actual"] for r in cell], [r["predictions"][method] for r in cell])
                    for method in METHODS}}
        complete = all(cells[f"{p}/{h}"]["n"] >= MIN_PAIRS for p in PRODUCTS for h in HORIZONS)
        macro = {method: {loss: (sum(cells[f"{p}/{h}"]["metrics"][method][loss]
                                          for p in PRODUCTS for h in HORIZONS) / 40 if complete else None)
                          for loss in ("mae", "mape")} for method in METHODS}
        coverage = {method: {"planned": len(selected),
                             "issued": sum(_positive(r["predictions"].get(method)) is not None for r in selected),
                             "issuedOnValidTargets": sum(_date(r["target"]) in actuals[r["product"]]
                                                         and _positive(r["predictions"].get(method)) is not None for r in selected)}
                    for method in METHODS}
        coverage_by_cell = {}
        for product in PRODUCTS:
            for horizon in HORIZONS:
                key = f"{product}/{horizon}"
                planned_rows = [r for r in selected if r["product"] == product and r["horizon"] == horizon]
                valid_rows = [r for r in planned_rows if _date(r["target"]) in actuals[product]]
                coverage_by_cell[key] = {"planned": len(planned_rows), "validTargetCount": len(valid_rows),
                    "issued": {method: sum(_positive(r["predictions"].get(method)) is not None for r in planned_rows)
                               for method in METHODS}, "commonScored": cells[key]["n"]}
        per_crop = {}
        for product in PRODUCTS:
            per_crop[product] = {}
            for method in METHODS:
                by_horizon = [cells[f"{product}/{h}"]["metrics"][method] for h in HORIZONS]
                per_crop[product][method] = {
                    **direct_horizon._metrics([r["actual"] for r in common if r["product"] == product],
                                              [r["predictions"][method] for r in common if r["product"] == product]),
                    "equalHorizonMae": float(np.mean([m["mae"] for m in by_horizon])) if all(m["mae"] is not None for m in by_horizon) else None,
                    "equalHorizonMape": float(np.mean([m["mape"] for m in by_horizon])) if all(m["mape"] is not None for m in by_horizon) else None}
        tests = []
        paired_losses = []
        for product in PRODUCTS:
            for horizon in HORIZONS:
                cell = [r for r in common if r["product"] == product and r["horizon"] == horizon]
                for comparator in COMPARATORS:
                    for loss in ("mae", "mape"):
                        diffs = {}
                        for r in cell:
                            actual = r["actual"]
                            c, b = r["predictions"]["median_ensemble"], r["predictions"][comparator]
                            diffs[_date(r["target"])] = abs(c-actual)-abs(b-actual) if loss == "mae" else 100*(abs(c-actual)-abs(b-actual))/actual
                        tests.append({"product": product, "horizon": horizon, "comparator": comparator,
                                      "loss": loss, "n": len(diffs), "dm": {str(bw): _dm(diffs, bw) for bw in BANDWIDTHS}})
                    paired_losses.append({"product": product, "horizon": horizon, "comparator": comparator,
                        "n": len(cell),
                        "meanPairedMaeDifference": float(np.mean([abs(r["predictions"]["median_ensemble"]-r["actual"])-abs(r["predictions"][comparator]-r["actual"]) for r in cell])) if cell else None,
                        "meanPairedMapeDifferencePp": float(np.mean([100*(abs(r["predictions"]["median_ensemble"]-r["actual"])-abs(r["predictions"][comparator]-r["actual"]))/r["actual"] for r in cell])) if cell else None})
        for bw in BANDWIDTHS:
            _holm_family(tests, bw)
        joint = _joint_verdicts(tests)
        gate = _period_gate(complete, macro, per_crop, coverage, joint) if start.month in (4, 7) else None
        key = f"{start.isoformat()}_{end.isoformat()}"
        report["periods"][key] = {"planned": len(selected), "issuedCoverage": coverage,
            "coverageByCell": coverage_by_cell,
            "validCommonPairs": len(common), "exclusionsByState": _exclusions(selected, source, actuals, evidence, common_keys),
            "complete40Cells": complete, "macro40EqualCell": macro, "cells": cells, "perCrop": per_crop,
            "pairedCellLosses": paired_losses, "dmComparisons": tests, "jointCellVerdicts": joint,
            "holmFamilySizePerBandwidth": len(tests), "predeclaredGate": gate}
    later_gates = [v["predeclaredGate"] for k, v in report["periods"].items() if k.startswith(("2026-04", "2026-07"))]
    overall = ("pass" if len(later_gates) == 2 and all(g["status"] == "pass" for g in later_gates) else
               "fail" if len(later_gates) == 2 and any(g["status"] == "fail" for g in later_gates) else "unsupported")
    report["twoWindowPredeclaredConclusion"] = overall
    report["resultType"] = "post_hoc_exploratory"
    report["independentConfirmation"] = False
    report["interpretation"] = "Existing Jul-Sep component results were already seen; even a pass is exploratory and not independent confirmation. Nonsignificance is not equivalence."
    report["reportSha256"] = _hash(report)
    return report


def _write_json_create_only(path: str | Path, doc: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen offline national median ensemble")
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    common.add_argument("--coverage-report", default=str(DEFAULT_COVERAGE))
    common.add_argument("--development-manifest", default=str(PROJECT_ROOT / MANIFEST_PATHS["development"]))
    common.add_argument("--fresh-manifest", default=str(PROJECT_ROOT / MANIFEST_PATHS["fresh"]))
    build = sub.add_parser("build", parents=[common], help="write answer-free median manifest")
    build.add_argument("--output", required=True)
    score = sub.add_parser("score", parents=[common], help="replay and score saved median manifest")
    score.add_argument("--manifest", required=True)
    score.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    development = _load_frozen_manifest("development", args.development_manifest)
    fresh = _load_frozen_manifest("fresh", args.fresh_manifest)
    bundles = {"development": {"archivePath": args.archive},
               "fresh": {"archivePath": args.archive, "coverageReportPath": args.coverage_report}}
    if args.command == "build":
        result = build_manifest(development, fresh, bundles)
    else:
        candidate = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        result = score_manifest(candidate, development, fresh, bundles)
    _write_json_create_only(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
