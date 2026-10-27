"""Frozen offline evaluation of the four-crop national price route.

Build the answer-free route manifest first; score it separately against the
validated national target source. This evaluates a retrospective composition,
not a prospective or official five-crop result.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import direct_horizon as dh
from . import national_price_selection as selection

ROOT = Path(__file__).resolve().parents[1]
DEV_PATH = Path("outputs/experiments/national-price-selection-20261009/development-manifest.json")
FRESH_PATH = Path("outputs/experiments/national-price-selection-20261009/fresh-manifest.json")
INPUT_HASHES = {
    str(DEV_PATH): "8209689fbaddd33ffdfe7f30a87b200fb1fa881d5788e38ffebb13de8974b2e8",
    str(FRESH_PATH): "925ac27d569a1ca299451f7f4a3073bcece99b7e0901d9bbc2cf3d1a089f3fa4",
}
ROUTE = {"cabbage": "calendar_sarima", "radish": "ridge_alpha_100",
         "onion": "ridge_alpha_100", "pepper": "ridge_alpha_100"}
PRODUCTS = tuple(selection.PRODUCTS)
HORIZONS = tuple(selection.HORIZONS)
METHODS = ("crop_route", "ridge_alpha_100", "calendar_sarima", "fixed_blend_50_50")
COMPARATORS = ("ridge_alpha_100", "calendar_sarima", "fixed_blend_50_50")
WINDOWS = {"jan_mar": (dt.date(2026, 1, 1), dt.date(2026, 3, 31)),
           "apr_jun": (dt.date(2026, 4, 1), dt.date(2026, 6, 30)),
           "jul_sep": (dt.date(2026, 7, 1), dt.date(2026, 9, 30))}
BANDWIDTHS = (9, 28)
MIN_PAIRS = 30
ALPHA = 0.05
PROTOCOL = {
    "id": "national-crop-route-eval-2026-10-09",
    "route": ROUTE,
    "sourceManifests": INPUT_HASHES,
    "targetWindow": ["2026-01-01", "2026-09-30"],
    "grid": "four national crops x target date x horizons 1-10; cutoff is origin minus two calendar days",
    "scoring": "four methods on identical positive-target pairs; crop/horizon cell equal-weight macro; paired MAE and MAPE",
    "inference": "route vs Ridge100/SARIMA/fixed50, 4 crops x 10 horizons x 3 comparators x 2 losses = 240 tests per window and bandwidth; calendar Bartlett HAC9/HAC28, separate Holm .05; n<30 or invalid variance stays with p=1",
    "gate": "Apr-Jun and Jul-Sep route MAPE at least 0.25 pp below best of three shared methods; coverage nondecreasing; no crop MAPE >1 pp and MAE >5% worse than that crop's best shared method",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def _load_inputs(root: Path = ROOT) -> tuple[dict, dict]:
    docs = {}
    for rel, expected in INPUT_HASHES.items():
        payload = (root / rel).read_bytes()
        if _sha(payload) != expected:
            raise ValueError(f"pinned source manifest SHA-256 mismatch: {rel}")
        doc = json.loads(payload)
        selection._verify_manifest(doc)
        if doc.get("sourceDigests", {}).get("archiveSha256") != selection.national_transfer.ARCHIVE_SHA256:
            raise ValueError("source manifest archive digest differs from pinned national archive")
        docs[rel] = doc
    dev, fresh = docs[str(DEV_PATH)], docs[str(FRESH_PATH)]
    if dev["window"] != ["2025-10-01", "2026-06-30"] or fresh["window"] != ["2026-07-01", "2026-09-30"]:
        raise ValueError("pinned source manifest cutoffs differ from frozen windows")
    if dev["sourceDigests"].get("archiveSha256") != fresh["sourceDigests"].get("archiveSha256"):
        raise ValueError("development and fresh manifests do not share the pinned archive")
    return dev, fresh


def _all_rows(manifests: tuple[dict, dict]) -> list[dict]:
    indexed = {}
    for manifest in manifests:
        for row in manifest["forecasts"]:
            target = dt.date.fromisoformat(row["target"])
            if not (dt.date(2026, 1, 1) <= target <= dt.date(2026, 9, 30)):
                continue
            key = (row["product"], row["origin"], row["target"], row["horizon"])
            if key in indexed:
                raise ValueError("duplicate forecast key across source manifests")
            if row["product"] not in PRODUCTS or row["horizon"] not in HORIZONS:
                raise ValueError("unexpected crop or horizon in source manifest")
            if dt.date.fromisoformat(row["cutoff"]) != dt.date.fromisoformat(row["origin"]) - dt.timedelta(days=2):
                raise ValueError("forecast cutoff does not match frozen origin-minus-two rule")
            for method in selection.METHODS:
                value = row["predictions"].get(method)
                if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
                    raise ValueError("source manifest contains invalid candidate prediction")
            indexed[key] = row
    expected = {(p, (dt.date(2026, 1, 1) + dt.timedelta(days=i) - dt.timedelta(days=h)).isoformat(),
                 (dt.date(2026, 1, 1) + dt.timedelta(days=i)).isoformat(), h)
                for i in range(273) for p in PRODUCTS for h in HORIZONS}
    actual = {(r["product"], r["origin"], r["target"], r["horizon"]) for r in indexed.values()}
    if actual != expected or len(indexed) != 10_920:
        raise ValueError(f"route grid mismatch: expected 10920 exact keys, got {len(indexed)}")
    return [indexed[k] for k in sorted(indexed)]


def build_manifest(*, root: Path = ROOT) -> dict:
    """Create answer-free route forecasts after replaying both pinned inputs."""
    dev, fresh = _load_inputs(root)
    # Rebuild both upstream forecast sets from their validated immutable source.
    for manifest in (dev, fresh):
        bundle = {"archivePath": str(root / selection.national_transfer.INPUT_ARCHIVE.relative_to(ROOT))}
        if "coverageReportSha256" in manifest["sourceDigests"]:
            bundle["coverageReportPath"] = str(root / "data/raw/nongnet_national_fresh_holdout_20260701-20260930/coverage-report.json")
        rebuilt = selection.build_manifest(bundle, start=dt.date.fromisoformat(manifest["window"][0]),
                                            end=dt.date.fromisoformat(manifest["window"][1]))
        if rebuilt["forecasts"] != manifest["forecasts"] or rebuilt["sourceDigests"] != manifest["sourceDigests"]:
            raise ValueError("upstream manifest forecast rows do not replay from fixed source")
    rows = []
    for source in _all_rows((dev, fresh)):
        crop = source["product"]
        predictions = source["predictions"]
        rows.append({**{k: source[k] for k in ("product", "origin", "target", "horizon", "cutoff")},
                     "predictions": {"crop_route": predictions[ROUTE[crop]],
                                     **{m: predictions[m] for m in COMPARATORS}},
                     "lineage": {"crop_route": {"routedMethod": ROUTE[crop],
                                                    "sourceLineage": source.get("lineage", {}).get(ROUTE[crop])}}})
    doc = {"schemaVersion": 1, "protocol": PROTOCOL, "phase": "forecast_only",
           "targetValuesAttached": False,
           "sourceManifestSha256": {str(DEV_PATH): INPUT_HASHES[str(DEV_PATH)], str(FRESH_PATH): INPUT_HASHES[str(FRESH_PATH)]},
           "sourceDigests": {"archiveSha256": dev["sourceDigests"]["archiveSha256"],
                             "developmentObservationsSha256": dev["sourceDigests"]["combinedObservationsSha256"],
                             "freshObservationsSha256": fresh["sourceDigests"]["combinedObservationsSha256"],
                             "freshCoverageReportSha256": fresh["sourceDigests"]["coverageReportSha256"]},
           "window": list(PROTOCOL["targetWindow"]), "plannedForecastCount": len(rows), "forecasts": rows}
    doc["manifestSha256"] = _canonical_hash(doc)
    return doc


def _verify_route_manifest(manifest: dict) -> None:
    if manifest.get("protocol") != PROTOCOL or manifest.get("phase") != "forecast_only" or manifest.get("targetValuesAttached") is not False:
        raise ValueError("not a forecast-only frozen crop route manifest")
    if manifest.get("sourceManifestSha256") != INPUT_HASHES:
        raise ValueError("route manifest source hashes differ from frozen values")
    forecasts = manifest.get("forecasts", [])
    if any("actual" in r or "targetValue" in r for r in forecasts):
        raise ValueError("answer-bearing field in route forecast manifest")
    if len(forecasts) != 10_920 or manifest.get("plannedForecastCount") != 10_920:
        raise ValueError("route forecast count is not the complete fixed grid")
    saved = manifest.get("manifestSha256")
    body = dict(manifest); body.pop("manifestSha256", None)
    if saved != _canonical_hash(body):
        raise ValueError("route manifest hash mismatch")


def _joint_classification(conclusions: list[str]) -> str:
    if "inference_unavailable" in conclusions:
        return "inference_unavailable"
    if all(x == "improvement" for x in conclusions):
        return "joint_improvement"
    if all(x == "worsening" for x in conclusions):
        return "joint_worsening"
    if "improvement" in conclusions and "worsening" in conclusions:
        return "mixed"
    return "inconclusive"


def _window_report(rows: list[dict], planned: list[dict], exclusions: dict) -> dict:
    comparisons, tests = [], {bw: [] for bw in BANDWIDTHS}
    cell_values = {}
    for crop in PRODUCTS:
        for h in HORIZONS:
            cell = [r for r in rows if r["product"] == crop and r["horizon"] == h]
            metrics = {m: dh._metrics([r["actual"] for r in cell], [r["predictions"][m] for r in cell]) for m in METHODS}
            cell_values[(crop, h)] = {"n": len(cell), "metrics": metrics}
            for comparator in COMPARATORS:
                for loss, key in (("ae", "mae"), ("ape", "mape")):
                    diffs = {}
                    for r in cell:
                        y, a, b = r["actual"], r["predictions"]["crop_route"], r["predictions"][comparator]
                        diffs[dt.date.fromisoformat(r["target"])] = abs(y-a)-abs(y-b) if loss == "ae" else 100*(abs(y-a)-abs(y-b))/y
                    item = {"product": crop, "horizon": h, "comparator": comparator, "loss": loss, "n": len(diffs),
                            "routeMetric": metrics["crop_route"][key], "comparatorMetric": metrics[comparator][key]}
                    for bw in BANDWIDTHS:
                        dm = dh._dm(diffs, bw)
                        item[f"hac{bw}"] = dm
                        tests[bw].append({**dm, "pValue": dm.get("pValue", 1.0), "_item": item})
                    comparisons.append(item)
    for bw in BANDWIDTHS:
        family = tests[bw]
        if len(family) != 240:
            raise AssertionError("frozen 240-test DM family shape mismatch")
        dh._holm(family)
        for result in family:
            dm = {k: v for k, v in result.items() if not k.startswith("_")}
            dm["holmAdjustedPValue"] = result["holmPValue"]
            dm["conclusion"] = ("inference_unavailable" if dm["status"] != "tested" else
                                "improvement" if dm["meanLossDifference"] < 0 and dm["holmSignificant"] else
                                "worsening" if dm["meanLossDifference"] > 0 and dm["holmSignificant"] else "inconclusive")
            result["_item"][f"hac{bw}"] = dm
    method_metrics = {}
    for method in METHODS:
        common = [r for r in rows if all(r["predictions"].get(m) is not None for m in METHODS)]
        method_metrics[method] = {"planned": len(planned), "issued": sum(r["predictions"].get(method) is not None for r in planned),
                                  "commonPairs": len(common), "mae": dh._metrics([r["actual"] for r in common], [r["predictions"][method] for r in common])["mae"],
                                  "mape": dh._metrics([r["actual"] for r in common], [r["predictions"][method] for r in common])["mape"]}
    valid40 = all(cell_values[(p,h)]["n"] >= MIN_PAIRS for p in PRODUCTS for h in HORIZONS)
    macro = {m: {loss: (sum(cell_values[(p,h)]["metrics"][m][loss] for p in PRODUCTS for h in HORIZONS)/40 if valid40 else None)
                 for loss in ("mae", "mape")} for m in METHODS}
    crop_macro = {}
    for crop in PRODUCTS:
        cells = [cell_values[(crop, h)] for h in HORIZONS]
        crop_macro[crop] = {m: {loss: (sum(c["metrics"][m][loss] for c in cells)/10
                                       if all(c["metrics"][m][loss] is not None for c in cells) else None)
                                for loss in ("mae", "mape")} for m in METHODS}
    cell_reports = []
    for crop in PRODUCTS:
        for h in HORIZONS:
            cell_rows = [r for r in rows if r["product"] == crop and r["horizon"] == h]
            identical = {m: all(r["predictions"].get("crop_route") == r["predictions"].get(m) for r in cell_rows)
                         for m in COMPARATORS} if cell_rows else {m: False for m in COMPARATORS}
            cell_reports.append({"product": crop, "horizon": h, **cell_values[(crop,h)],
                                 "identicalPredictionsToComparator": identical})
    joint = []
    by_key = {(r["product"], r["horizon"], r["comparator"], r["loss"]): r for r in comparisons}
    for crop in PRODUCTS:
        for h in HORIZONS:
            for comparator in COMPARATORS:
                needed = [by_key[(crop,h,comparator,loss)][f"hac{bw}"]["conclusion"]
                          for loss in ("ae", "ape") for bw in BANDWIDTHS]
                classification = _joint_classification(needed)
                joint.append({"product": crop, "horizon": h, "comparator": comparator,
                              "classifications": {"ae_hac9": needed[0], "ae_hac28": needed[1],
                                                   "ape_hac9": needed[2], "ape_hac28": needed[3]},
                              "jointConclusion": classification})
    return {"planned": len(planned), "issuedCoverage": method_metrics, "commonPairs": len(rows),
            "exclusionCounts": exclusions, "complete40CellsAt30Pairs": valid40, "macro40CellEqualWeight": macro,
            "perCrop10HorizonEqualWeight": crop_macro, "cells": cell_reports,
            "dmComparisons": comparisons, "dmFamilySizePerBandwidth": 240,
            "jointDmConclusions": joint,
            "atLeastOneJointImprovement": {c: any(x["comparator"] == c and x["jointConclusion"] == "joint_improvement" for x in joint) for c in COMPARATORS},
            "anyJointImprovement": any(x["jointConclusion"] == "joint_improvement" for x in joint),
            "statisticalGate": "a joint classification requires AE and APE improvement/worsening at both HAC9 and HAC28 with Holm significance; unavailable, mixed and inconclusive remain distinct"}


def _validate_target_bundle(root: Path, raw_bundle: dict | None, expected: dict) -> dict:
    archive = root / selection.national_transfer.INPUT_ARCHIVE.relative_to(ROOT)
    coverage = root / "data/raw/nongnet_national_fresh_holdout_20260701-20260930/coverage-report.json"
    bundle = raw_bundle
    if not isinstance(bundle, dict) or set(bundle) != {"archivePath", "coverageReportPath"}:
        raise ValueError("scoring requires the pinned archive and fresh coverage report")
    if Path(bundle["archivePath"]).resolve() != archive.resolve() or Path(bundle["coverageReportPath"]).resolve() != coverage.resolve():
        raise ValueError("target source bundle paths differ from the pinned source")
    parsed = selection._load_source(bundle)
    digests = parsed["sourceDigests"]
    if (digests.get("archiveSha256") != expected.get("archiveSha256") or
            digests.get("coverageReportSha256") != expected.get("freshCoverageReportSha256") or
            digests.get("combinedObservationsSha256") != expected.get("freshObservationsSha256")):
        raise ValueError("target source observation or fresh coverage digest differs from frozen manifest")
    dev, fresh = _load_inputs(root)
    if (dev["sourceDigests"].get("combinedObservationsSha256") != expected.get("developmentObservationsSha256") or
            fresh["sourceDigests"].get("combinedObservationsSha256") != expected.get("freshObservationsSha256") or
            fresh["sourceDigests"].get("coverageReportSha256") != expected.get("freshCoverageReportSha256")):
        raise ValueError("route source digests differ from pinned source manifests")
    return bundle


def score_manifest(manifest: dict, *, raw_bundle: dict, root: Path = ROOT) -> dict:
    """Replay route/source, then attach outcomes and score the frozen windows."""
    _verify_route_manifest(manifest)
    bundle = _validate_target_bundle(root, raw_bundle, manifest.get("sourceDigests", {}))
    rebuilt = build_manifest(root=root)
    if rebuilt["forecasts"] != manifest["forecasts"] or rebuilt["manifestSha256"] != manifest["manifestSha256"]:
        raise ValueError("route manifest does not replay from fixed source manifests")
    source = selection._load_source(bundle)
    status_evidence = selection._coverage_status_evidence(bundle)
    output = {"schemaVersion": 1, "protocol": PROTOCOL, "manifestSha256": manifest["manifestSha256"],
              "sourceDigests": source["sourceDigests"], "windows": {}}
    for name, (start, end) in WINDOWS.items():
        planned = [r for r in manifest["forecasts"] if start <= dt.date.fromisoformat(r["target"]) <= end]
        scored, exclusions = [], {}
        for row in planned:
            status, observed = selection._status(source, row["product"], dt.date.fromisoformat(row["target"]), status_evidence)
            actual = selection._positive(observed.get("price")) if observed else None
            if actual is None:
                exclusions[status] = exclusions.get(status, 0) + 1
                continue
            if any(selection._positive(row["predictions"].get(m)) is None for m in METHODS):
                exclusions["incomplete_common_candidate_forecast"] = exclusions.get("incomplete_common_candidate_forecast", 0) + 1
                continue
            scored.append({**row, "actual": actual, "marketStatus": status})
        report = _window_report(scored, planned, exclusions)
        valid_targets = 0
        for row in planned:
            _, observed = selection._status(source, row["product"], dt.date.fromisoformat(row["target"]), status_evidence)
            valid_targets += int(observed is not None and selection._positive(observed.get("price")) is not None)
        report["coverage"] = {"planned": len(planned), "issued": {m: sum(selection._positive(r["predictions"].get(m)) is not None for r in planned) for m in METHODS},
                              "validPositiveTargets": valid_targets, "commonScored": len(scored)}
        output["windows"][name] = report
    output["gate"] = _gate(output["windows"])
    output["reportSha256"] = _canonical_hash(output)
    return output


def _gate(windows: dict) -> dict:
    criteria = {}
    for name in ("apr_jun", "jul_sep"):
        report = windows.get(name, {})
        macro = report.get("macro40CellEqualWeight", {})
        crop_macro = report.get("perCrop10HorizonEqualWeight", {})
        route_mape = macro.get("crop_route", {}).get("mape")
        available = [(m, macro.get(m, {}).get("mape")) for m in COMPARATORS
                     if macro.get(m, {}).get("mape") is not None]
        best_method = min(available, key=lambda pair: (pair[1], COMPARATORS.index(pair[0])))[0] if available else None
        best_mape = macro.get(best_method, {}).get("mape") if best_method else None
        crop_bad = {}
        for crop in PRODUCTS:
            vals = crop_macro.get(crop, {})
            route_vals = vals.get("crop_route", {})
            crop_available = [(m, vals.get(m, {}).get("mape")) for m in COMPARATORS
                              if vals.get(m, {}).get("mape") is not None]
            crop_best = min(crop_available, key=lambda pair: (pair[1], COMPARATORS.index(pair[0])))[0] if crop_available else None
            crop_best_vals = vals.get(crop_best, {}) if crop_best else {}
            route_crop_mape, route_crop_mae = route_vals.get("mape"), route_vals.get("mae")
            base_mape, base_mae = crop_best_vals.get("mape"), crop_best_vals.get("mae")
            crop_bad[crop] = {"chosenComparator": crop_best,
                              "mapeWorsePp": route_crop_mape-base_mape if route_crop_mape is not None and base_mape is not None else None,
                              "maeWorsePct": 100*(route_crop_mae-base_mae)/base_mae if route_crop_mae is not None and base_mae not in (None, 0) else None}
        coverage = report.get("issuedCoverage", {})
        route_issued = coverage.get("crop_route", {}).get("issued")
        comparator_counts = [coverage.get(m, {}).get("issued") for m in COMPARATORS]
        coverage_ok = route_issued is not None and all(x is not None and route_issued >= x for x in comparator_counts)
        full = report.get("complete40CellsAt30Pairs") is True
        criteria[name] = {"complete40Cells": full,
                          "bestCommonMethodByMacroMape": best_method,
                          "mapeAtLeast0_25ppBelowBestCommonMethod": bool(full and route_mape is not None and best_mape is not None and best_mape-route_mape >= .25),
                          "coverageNondecreasing": bool(coverage_ok),
                          "noCropExceedsPlus1ppMapeAndPlus5pctMae": bool(full and all(v["mapeWorsePp"] is not None and v["maeWorsePct"] is not None and v["mapeWorsePp"] <= 1 and v["maeWorsePct"] <= 5 for v in crop_bad.values())),
                          "cropChanges": crop_bad}
    passed = bool(criteria) and all(all(v[k] for k in ("complete40Cells", "mapeAtLeast0_25ppBelowBestCommonMethod", "coverageNondecreasing", "noCropExceedsPlus1ppMapeAndPlus5pctMae")) for v in criteria.values())
    return {"status": "pass" if passed else "fail_or_unsupported", "criteria": criteria,
            "interpretation": "This retrospective 2026 comparison does not establish prospective benefit or an official five-crop score."}


def _exclusive_output(path: Path, inputs: tuple[Path, ...]) -> None:
    destination = path.resolve()
    if any(destination == source.resolve() for source in inputs):
        raise ValueError("output path must not overwrite an input")
    if destination.exists():
        raise FileExistsError(f"output already exists: {destination}")


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="write answer-free routed forecast manifest")
    build.add_argument("output", type=Path)
    score = sub.add_parser("score", help="replay and score an answer-free route manifest")
    score.add_argument("manifest", type=Path)
    score.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    inputs = tuple(ROOT / rel for rel in INPUT_HASHES)
    if args.command == "build":
        _exclusive_output(args.output, inputs)
    else:
        manifest_path = args.manifest.resolve()
        _exclusive_output(args.output, (*inputs, args.manifest))
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
    bundle = {"archivePath": str(ROOT / selection.national_transfer.INPUT_ARCHIVE.relative_to(ROOT)),
              "coverageReportPath": str(ROOT / "data/raw/nongnet_national_fresh_holdout_20260701-20260930/coverage-report.json")}
    result = (build_manifest() if args.command == "build" else
              score_manifest(json.loads(manifest_path.read_text(encoding="utf-8")), raw_bundle=bundle))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
