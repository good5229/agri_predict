"""Frozen offline evaluation of a 50:50 nonlinear price-ratio/SARIMA blend.

The build command reads only the pinned answer-free component manifest. The
separate score command replays that manifest and only then attaches national
positive-price targets from the pinned archive and fresh coverage source.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from . import direct_horizon as dh
from . import national_price_selection as selection

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = Path("outputs/experiments/national-nonlinear-ratio-20261009/forecast-manifest.json")
INPUT_SHA256 = "1a459d61637af66f024af2c144326347769eed4a5978771bfb7c9b1710908f08"
INPUT_PROTOCOL = "national-nonlinear-ratio-2026-10-09"
CANDIDATE = "ratio_sarima_blend_50_50"
RATIO = "hist_gradient_boosting_log_price_ratio"
SARIMA = "calendar_sarima"
BASELINE_BLEND = "fixed_blend_50_50"
METHODS = (CANDIDATE, SARIMA, BASELINE_BLEND, RATIO)
COMPARATORS = (SARIMA, BASELINE_BLEND, RATIO)
PRODUCTS = tuple(selection.PRODUCTS)
HORIZONS = tuple(selection.HORIZONS)
WINDOWS = {
    "jan_mar": (dt.date(2026, 1, 1), dt.date(2026, 3, 31)),
    "apr_jun": (dt.date(2026, 4, 1), dt.date(2026, 6, 30)),
    "jul_sep": (dt.date(2026, 7, 1), dt.date(2026, 9, 30)),
}
BANDWIDTHS = (9, 28)
MIN_PAIRS = 30
ALPHA = 0.05
ARCHIVE_SHA256 = selection.national_transfer.ARCHIVE_SHA256
COVERAGE_PATH = Path("data/raw/nongnet_national_fresh_holdout_20260701-20260930/coverage-report.json")
PROTOCOL = {
    "id": "national-sarima-ratio-blend-2026-10-09",
    "inputManifestPath": str(INPUT_PATH), "inputManifestSha256": INPUT_SHA256,
    "composition": "arithmetic 0.5 * hist_gradient_boosting_log_price_ratio + 0.5 * calendar_sarima",
    "targetWindow": ["2026-01-01", "2026-09-30"],
    "comparators": list(COMPARATORS),
    "inference": "per quarter: 4 crops x 10 horizons x 3 comparators x AE/APE = 240 two-sided DM tests per calendar Bartlett HAC9/HAC28 bandwidth; separate Holm .05; n<30 or invalid variance remains p=1",
    "gate": "at least two quarters gain >=0.25 MAPE pp over lower-MAPE SARIMA/fixed50; remaining quarter loss <=0.25 pp; coverage nondecreasing; no crop >1 pp MAPE or >5% MAE worse than its lower-MAPE SARIMA/fixed50 comparator; at least one crop/horizon joint Holm improvement over that comparator",
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hash(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _identity(row: dict) -> tuple[str, str, str, int]:
    product, origin_text, target_text, horizon = (row.get(k) for k in ("product", "origin", "target", "horizon"))
    if product not in PRODUCTS or isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
        raise ValueError("forecast product/horizon key invalid")
    try:
        origin, target = dt.date.fromisoformat(origin_text), dt.date.fromisoformat(target_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("forecast dates must be ISO dates") from exc
    if target - origin != dt.timedelta(days=horizon):
        raise ValueError("forecast target-origin/horizon mismatch")
    if row.get("cutoff") != (origin - dt.timedelta(days=2)).isoformat():
        raise ValueError("forecast cutoff mismatch")
    return product, origin_text, target_text, horizon


def validate_component_manifest(doc: dict) -> list[dict]:
    """Validate answer-free component content and the exact planned grid."""
    if (doc.get("schemaVersion") != 1 or doc.get("protocolId") != INPUT_PROTOCOL or
            doc.get("phase") != "forecast_only" or doc.get("targetValuesAttached") is not False):
        raise ValueError("input is not the pinned answer-free nonlinear-ratio manifest")
    if doc.get("window") != ["2026-01-01", "2026-09-30"] or doc.get("products") != list(PRODUCTS) or doc.get("horizons") != list(HORIZONS):
        raise ValueError("input window/product/horizon declarations differ from frozen grid")
    if doc.get("methods") != [RATIO, "ridge_alpha_100", SARIMA, BASELINE_BLEND, "persistence"]:
        raise ValueError("input method declaration differs from frozen component set")
    if doc.get("sourceDigests", {}).get("development", {}).get("archiveSha256") != ARCHIVE_SHA256 or doc.get("sourceDigests", {}).get("fresh", {}).get("archiveSha256") != ARCHIVE_SHA256:
        raise ValueError("input source archive digest differs from pinned national archive")
    digests = doc["sourceDigests"]
    if set(digests) != {"development", "fresh"} or set(digests["development"]) != {"archiveSha256", "combinedObservationsSha256"} or set(digests["fresh"]) != {"archiveSha256", "combinedObservationsSha256", "coverageReportSha256"}:
        raise ValueError("input source digest schema differs from pinned component manifest")
    fresh = digests["fresh"]
    if not all(isinstance(digests[k].get(name), str) and len(digests[k][name]) == 64
               for k, names in (("development", ("archiveSha256", "combinedObservationsSha256")),
                                ("fresh", ("archiveSha256", "combinedObservationsSha256", "coverageReportSha256")))
               for name in names):
        raise ValueError("input source digest values are incomplete")
    forecasts = doc.get("forecasts")
    if not isinstance(forecasts, list) or len(forecasts) != 10_920 or doc.get("plannedForecastCount") != 10_920:
        raise ValueError("input manifest must contain exactly 10,920 planned rows")
    body = dict(doc)
    saved_hash = body.pop("manifestSha256", None)
    if saved_hash != _hash(body):
        raise ValueError("input manifest canonical content hash mismatch")
    indexed = {}
    for row in forecasts:
        key = _identity(row)
        if key in indexed:
            raise ValueError("duplicate input forecast key")
        if "actual" in row or "targetValue" in row:
            raise ValueError("answer-bearing field in input forecast manifest")
        predictions = row.get("predictions")
        if not isinstance(predictions, dict):
            raise ValueError("input forecast predictions missing")
        for method in (RATIO, SARIMA, BASELINE_BLEND, "ridge_alpha_100", "persistence"):
            if _positive(predictions.get(method)) is None:
                raise ValueError(f"input forecast prediction is not positive finite: {method}")
        indexed[key] = row
    expected = {(p, (target-dt.timedelta(days=h)).isoformat(), target.isoformat(), h)
                for i in range(273) for target in [dt.date(2026, 1, 1)+dt.timedelta(days=i)]
                for p in PRODUCTS for h in HORIZONS}
    if set(indexed) != expected:
        raise ValueError("input manifest does not have the exact 2026 Jan-Sep crop/target/horizon keys")
    return [indexed[k] for k in sorted(indexed)]


def load_component_manifest(root: Path = ROOT) -> tuple[dict, bytes]:
    path = root / INPUT_PATH
    raw = path.read_bytes()
    if _sha(raw) != INPUT_SHA256:
        raise ValueError("pinned nonlinear-ratio manifest byte SHA-256 mismatch")
    doc = json.loads(raw)
    validate_component_manifest(doc)
    return doc, raw


def _build_from_component(doc: dict, input_sha: str) -> dict:
    rows = validate_component_manifest(doc)
    forecasts = []
    for source in rows:
        prediction = 0.5 * float(source["predictions"][RATIO]) + 0.5 * float(source["predictions"][SARIMA])
        if _positive(prediction) is None:
            raise ValueError("blended prediction is not positive finite")
        preds = {CANDIDATE: prediction, SARIMA: source["predictions"][SARIMA],
                 BASELINE_BLEND: source["predictions"][BASELINE_BLEND], RATIO: source["predictions"][RATIO]}
        forecasts.append({**{k: source[k] for k in ("product", "origin", "target", "horizon", "cutoff")},
                          "predictions": preds,
                          "lineage": {"candidate": {"formula": "0.5*ratio+0.5*calendar_sarima",
                                                       "ratio": source["predictions"][RATIO],
                                                       "calendarSarima": source["predictions"][SARIMA]},
                                      "componentSourceLineage": source.get("lineage", {}).get("componentSources", {})}})
    result = {"schemaVersion": 1, "protocol": PROTOCOL, "phase": "forecast_only", "targetValuesAttached": False,
              "sourceDigests": doc["sourceDigests"], "inputManifestSha256": input_sha,
              "inputManifestContentSha256": doc["manifestSha256"], "window": ["2026-01-01", "2026-09-30"],
              "products": list(PRODUCTS), "horizons": list(HORIZONS), "methods": list(METHODS),
              "plannedForecastCount": len(forecasts), "forecasts": forecasts}
    result["manifestSha256"] = _hash(result)
    return result


def build_manifest(*, root: Path = ROOT) -> dict:
    """Build answer-free blended forecasts without loading target truth."""
    doc, raw = load_component_manifest(root)
    return _build_from_component(doc, _sha(raw))


def verify_manifest(doc: dict) -> None:
    if doc.get("protocol") != PROTOCOL or doc.get("phase") != "forecast_only" or doc.get("targetValuesAttached") is not False:
        raise ValueError("candidate manifest is not answer-free under the frozen protocol")
    if doc.get("inputManifestSha256") != INPUT_SHA256:
        raise ValueError("candidate manifest input SHA differs from frozen value")
    rows = doc.get("forecasts")
    if not isinstance(rows, list) or len(rows) != 10_920 or doc.get("plannedForecastCount") != 10_920:
        raise ValueError("candidate forecast count mismatch")
    body = dict(doc)
    if body.pop("manifestSha256", None) != _hash(body):
        raise ValueError("candidate manifest content hash mismatch")
    if any("actual" in row or "targetValue" in row for row in rows):
        raise ValueError("target leaked into candidate manifest")


def _target_source(raw_bundle: dict, expected: dict) -> tuple[dict, dict]:
    if not isinstance(raw_bundle, dict) or set(raw_bundle) != {"archivePath", "coverageReportPath"}:
        raise ValueError("scoring requires the pinned national archive and fresh coverage source")
    source = selection._load_source(raw_bundle)
    development_source = selection._load_source({"archivePath": raw_bundle["archivePath"]})
    digests = source["sourceDigests"]
    expected_dev, expected_fresh = expected.get("development", {}), expected.get("fresh", {})
    if (digests.get("archiveSha256") != ARCHIVE_SHA256 or
            digests.get("combinedObservationsSha256") != expected_fresh.get("combinedObservationsSha256") or
            digests.get("coverageReportSha256") != expected_fresh.get("coverageReportSha256") or
            development_source["sourceDigests"].get("combinedObservationsSha256") != expected_dev.get("combinedObservationsSha256")):
        raise ValueError("scoring target source digests differ from pinned component manifest")
    return source, selection._coverage_status_evidence(raw_bundle)


def _dm(diffs: dict[dt.date, float], bandwidth: int) -> dict:
    n = len(diffs)
    mean = sum(diffs.values()) / n if n else None
    if n < MIN_PAIRS:
        return {"status": "inference_unavailable_insufficient_pairs", "n": n, "meanLossDifference": mean, "statistic": None, "pValue": 1.0, "bandwidthDays": bandwidth}
    centered = {day: value-mean for day, value in diffs.items()}
    var_sum = sum(v*v for v in centered.values())
    for lag in range(1, bandwidth+1):
        cross = sum(v*centered.get(day+dt.timedelta(days=lag), 0.0) for day, v in centered.items())
        var_sum += 2*(1-lag/(bandwidth+1))*cross
    variance_mean = var_sum/(n*n)
    if not math.isfinite(variance_mean) or variance_mean <= 0:
        return {"status": "inference_unavailable_nonpositive_variance", "n": n, "meanLossDifference": mean, "statistic": None, "pValue": 1.0, "bandwidthDays": bandwidth}
    statistic = mean/math.sqrt(variance_mean)
    return {"status": "available", "n": n, "meanLossDifference": mean, "statistic": statistic,
            "pValue": math.erfc(abs(statistic)/math.sqrt(2)), "bandwidthDays": bandwidth}


def _holm(tests: list[dict], bandwidth: int) -> None:
    ordered = sorted(tests, key=lambda row: row["dm"][str(bandwidth)]["pValue"])
    running = 0.0
    for rank, row in enumerate(ordered):
        dm = row["dm"][str(bandwidth)]
        running = max(running, min(1.0, (len(tests)-rank)*dm["pValue"]))
        dm["holmAdjustedPValue"] = running
        dm["holmSignificant"] = dm["status"] == "available" and running < ALPHA
        dm["conclusion"] = ("inference_unavailable" if dm["status"] != "available" else
                            "improvement" if dm["holmSignificant"] and dm["meanLossDifference"] < 0 else
                            "worsening" if dm["holmSignificant"] and dm["meanLossDifference"] > 0 else "inconclusive")


def _joint_classification(labels: list[str]) -> str:
    if "inference_unavailable" in labels:
        return "inference_unavailable"
    if all(x == "improvement" for x in labels):
        return "joint_improvement"
    if all(x == "worsening" for x in labels):
        return "joint_worsening"
    if "improvement" in labels and "worsening" in labels:
        return "mixed"
    return "inconclusive"


def _period_report(planned: list[dict], scored: list[dict], exclusions: dict) -> dict:
    cell_rows, tests = {}, []
    for crop in PRODUCTS:
        for h in HORIZONS:
            cell = [r for r in scored if r["product"] == crop and r["horizon"] == h]
            cell_rows[(crop, h)] = cell
            cell_metrics = {m: dh._metrics([r["actual"] for r in cell], [r["predictions"][m] for r in cell]) for m in METHODS}
            for comparator in COMPARATORS:
                for loss, metric in (("ae", "mae"), ("ape", "mape")):
                    diffs = {}
                    for row in cell:
                        y, a, b = row["actual"], row["predictions"][CANDIDATE], row["predictions"][comparator]
                        diffs[dt.date.fromisoformat(row["target"])] = abs(y-a)-abs(y-b) if loss == "ae" else 100*(abs(y-a)-abs(y-b))/y
                    tests.append({"product": crop, "horizon": h, "comparator": comparator, "loss": loss,
                                  "n": len(diffs), "candidateMetric": cell_metrics[CANDIDATE][metric],
                                  "comparatorMetric": cell_metrics[comparator][metric],
                                  "dm": {str(bw): _dm(diffs, bw) for bw in BANDWIDTHS}})
    for bw in BANDWIDTHS:
        family = [t for t in tests]
        if len(family) != 240:
            raise AssertionError("fixed 240-test family size mismatch")
        _holm(family, bw)
    cells = []
    complete = True
    for crop in PRODUCTS:
        for h in HORIZONS:
            rows = cell_rows[(crop, h)]
            metrics = {m: dh._metrics([r["actual"] for r in rows], [r["predictions"][m] for r in rows]) for m in METHODS}
            complete &= len(rows) >= MIN_PAIRS
            cells.append({"product": crop, "horizon": h, "n": len(rows), "metrics": metrics})
    macro = {m: {metric: (sum(c["metrics"][m][metric] for c in cells)/40 if complete else None)
                 for metric in ("mae", "mape")} for m in METHODS}
    per_crop = {crop: {m: {metric: (sum(next(c for c in cells if c["product"] == crop and c["horizon"] == h)["metrics"][m][metric] for h in HORIZONS)/10
                                      if all(next(c for c in cells if c["product"] == crop and c["horizon"] == h)["metrics"][m][metric] is not None for h in HORIZONS) else None)
                           for metric in ("mae", "mape")} for m in METHODS} for crop in PRODUCTS}
    indexed = {(t["product"], t["horizon"], t["comparator"], t["loss"]): t for t in tests}
    joint = []
    for crop in PRODUCTS:
        for h in HORIZONS:
            for comp in COMPARATORS:
                vals = [indexed[(crop,h,comp,loss)]["dm"][str(bw)]["conclusion"] for loss in ("ae", "ape") for bw in BANDWIDTHS]
                joint.append({"product": crop, "horizon": h, "comparator": comp,
                              "conclusions": dict(zip(("ae_hac9", "ae_hac28", "ape_hac9", "ape_hac28"), vals)),
                              "jointConclusion": _joint_classification(vals)})
    coverage = {m: {"planned": len(planned), "issued": sum(_positive(r["predictions"].get(m)) is not None for r in planned),
                    "issuedOnValidTargets": sum(r.get("_hasValidTarget", False) and _positive(r["predictions"].get(m)) is not None for r in planned)} for m in METHODS}
    cov_by_cell = {}
    for crop in PRODUCTS:
        for h in HORIZONS:
            rows = [r for r in planned if r["product"] == crop and r["horizon"] == h]
            valid_n = sum(r.get("_hasValidTarget", False) for r in rows)
            cov_by_cell[f"{crop}/{h}"] = {"planned": len(rows), "validPositiveTargets": valid_n,
                "issued": {m: sum(_positive(r["predictions"].get(m)) is not None for r in rows) for m in METHODS},
                "issuedOnValidTargets": {m: sum(r.get("_hasValidTarget", False) and _positive(r["predictions"].get(m)) is not None for r in rows) for m in METHODS},
                "commonScored": len(cell_rows[(crop,h)])}
    exclusions["valid_positive_target_unissued_or_incomplete"] = sum(r.get("_hasValidTarget", False) for r in planned)-len(scored)
    return {"planned": len(planned), "validCommonPairs": len(scored), "exclusionsByState": exclusions,
            "complete40CellsAt30Pairs": bool(complete), "cells": cells, "macro40CellEqualWeight": macro,
            "perCrop10HorizonEqualWeight": per_crop, "issuedCoverage": coverage, "coverageByCell": cov_by_cell,
            "pairedDmComparisons": tests, "jointDmConclusions": joint,
            "anyJointImprovement": any(r["jointConclusion"] == "joint_improvement" for r in joint),
            "holmFamilySizePerBandwidth": 240}


def _gate(periods: dict) -> dict:
    criteria = {}
    for name in WINDOWS:
        if name not in periods:
            criteria[name] = {"complete40CellsAt30Pairs": False, "macroGainAtLeast0_25pp": False,
                              "remainingPeriodLossAtMost0_25pp": False, "coverageNondecreasing": False,
                              "cropSafeguardsPass": False, "jointImprovementEvidence": False,
                              "strongestComparator": None}
            continue
        report = periods[name]
        if report.get("complete40CellsAt30Pairs") is not True:
            criteria[name] = {"complete40CellsAt30Pairs": False, "macroGainPp": None,
                              "macroGainAtLeast0_25pp": False, "coverageNondecreasing": False,
                              "cropSafeguardsPass": False, "cropSafeguards": {},
                              "jointImprovementEvidence": False, "strongestComparator": None}
            continue
        macro = report.get("macro40CellEqualWeight", {})
        available = [(m, macro.get(m, {}).get("mape")) for m in (SARIMA, BASELINE_BLEND) if macro.get(m, {}).get("mape") is not None]
        strongest = min(available, key=lambda x: (x[1], (SARIMA, BASELINE_BLEND).index(x[0]))) if available else (None, None)
        gain = strongest[1]-macro[CANDIDATE]["mape"] if strongest[0] and macro[CANDIDATE]["mape"] is not None and report["complete40CellsAt30Pairs"] else None
        crop_safeguards = {}
        for crop in PRODUCTS:
            options = [(m, report["perCrop10HorizonEqualWeight"][crop][m]["mape"]) for m in (SARIMA, BASELINE_BLEND)
                       if report["perCrop10HorizonEqualWeight"][crop][m]["mape"] is not None]
            chosen = min(options, key=lambda x: (x[1], (SARIMA, BASELINE_BLEND).index(x[0]))) if options else (None, None)
            cand = report["perCrop10HorizonEqualWeight"][crop][CANDIDATE]
            base = report["perCrop10HorizonEqualWeight"][crop].get(chosen[0], {}) if chosen[0] else {}
            dm = {r["comparator"]: r for r in report["jointDmConclusions"] if r["product"] == crop}
            crop_safeguards[crop] = {"chosenComparator": chosen[0],
                "mapeWorsePp": cand["mape"]-chosen[1] if cand["mape"] is not None and chosen[1] is not None else None,
                "maeWorsePct": 100*(cand["mae"]-base.get("mae"))/base["mae"] if cand["mae"] is not None and base.get("mae") not in (None, 0) else None}
        cov = report["issuedCoverage"]
        coverage_ok = strongest[0] is not None and cov[CANDIDATE]["issued"] >= cov[strongest[0]]["issued"]
        joint = any(r["comparator"] == strongest[0] and r["jointConclusion"] == "joint_improvement" for r in report["jointDmConclusions"]) if strongest[0] else False
        criteria[name] = {"complete40CellsAt30Pairs": report["complete40CellsAt30Pairs"],
            "strongestComparator": strongest[0], "macroGainPp": gain,
            "macroGainAtLeast0_25pp": gain is not None and gain >= .25,
            "coverageNondecreasing": coverage_ok,
            "cropSafeguardsPass": report["complete40CellsAt30Pairs"] and all(v["mapeWorsePp"] is not None and v["maeWorsePct"] is not None and v["mapeWorsePp"] <= 1 and v["maeWorsePct"] <= 5 for v in crop_safeguards.values()),
            "cropSafeguards": crop_safeguards, "jointImprovementEvidence": joint}
    supported = all(v["complete40CellsAt30Pairs"] for v in criteria.values())
    gain_count = sum(v["macroGainAtLeast0_25pp"] for v in criteria.values())
    other_loss_ok = len(criteria) == 3 and all(v["macroGainPp"] is not None and v["macroGainPp"] >= -.25 for name, v in criteria.items() if not v["macroGainAtLeast0_25pp"])
    result = {"quartersMeetingGain": gain_count, "criteriaByQuarter": criteria,
              "atLeastTwoQuarterGains": gain_count >= 2,
              "remainingQuarterLossAtMost0_25pp": other_loss_ok,
              "coverageNondecreasing": supported and all(v["coverageNondecreasing"] for v in criteria.values()),
              "cropSafeguardsPass": supported and all(v["cropSafeguardsPass"] for v in criteria.values()),
              "atLeastOneJointImprovement": any(v["jointImprovementEvidence"] for v in criteria.values())}
    result["status"] = "unsupported" if not supported else "pass" if all(result[k] for k in ("atLeastTwoQuarterGains", "remainingQuarterLossAtMost0_25pp", "coverageNondecreasing", "cropSafeguardsPass", "atLeastOneJointImprovement")) else "fail"
    return result


def score_manifest(manifest: dict, *, raw_bundle: dict, root: Path = ROOT) -> dict:
    """Replay the frozen blend and target source before attaching actuals."""
    verify_manifest(manifest)
    component, raw = load_component_manifest(root)
    expected_candidate = _build_from_component(component, _sha(raw))
    if expected_candidate["manifestSha256"] != manifest["manifestSha256"] or expected_candidate["forecasts"] != manifest["forecasts"]:
        raise ValueError("candidate manifest replay mismatch")
    source, status_evidence = _target_source(raw_bundle, manifest["sourceDigests"])
    output = {"schemaVersion": 1, "protocol": PROTOCOL, "manifestSha256": manifest["manifestSha256"],
              "sourceDigests": source["sourceDigests"], "windows": {}}
    for name, (start, end) in WINDOWS.items():
        planned = [r for r in manifest["forecasts"] if start <= dt.date.fromisoformat(r["target"]) <= end]
        scored, exclusions, period_plan = [], {}, []
        for row in planned:
            status, observed = selection._status(source, row["product"], dt.date.fromisoformat(row["target"]), status_evidence)
            actual = selection._positive(observed.get("price")) if observed else None
            period_plan.append({**row, "_hasValidTarget": actual is not None})
            if actual is None:
                exclusions[status] = exclusions.get(status, 0)+1
                continue
            if any(_positive(row["predictions"].get(m)) is None for m in METHODS):
                exclusions["incomplete_common_candidate_forecast"] = exclusions.get("incomplete_common_candidate_forecast", 0)+1
                continue
            scored.append({**row, "actual": actual})
        output["windows"][name] = _period_report(period_plan, scored, exclusions)
    output["gate"] = _gate(output["windows"])
    output["resultType"] = "post_hoc_exploratory"
    output["independentConfirmation"] = False
    output["interpretation"] = "This retrospective 2026 comparison was chosen after component and period results were seen. It is not independent confirmation or an official five-crop result; nonsignificance is not equivalence."
    output["reportSha256"] = _hash(output)
    return output


def _exclusive_output(path: Path, inputs: tuple[Path, ...]) -> None:
    dest = path.resolve()
    if any(dest == source.resolve() for source in inputs):
        raise ValueError("output path must not overwrite an input")
    if dest.exists():
        raise FileExistsError(f"output already exists: {dest}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="create answer-free 50:50 forecasts")
    build.add_argument("output", type=Path)
    score = sub.add_parser("score", help="replay and score saved candidate manifest")
    score.add_argument("manifest", type=Path)
    score.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    component_path = ROOT / INPUT_PATH
    inputs = (component_path, ROOT / selection.national_transfer.INPUT_ARCHIVE.relative_to(ROOT), ROOT / COVERAGE_PATH)
    if args.command == "build":
        _exclusive_output(args.output, inputs)
        result = build_manifest()
    else:
        _exclusive_output(args.output, (*inputs, args.manifest))
        if not args.manifest.is_file():
            raise FileNotFoundError(args.manifest)
        raw_bundle = {"archivePath": str(inputs[1]), "coverageReportPath": str(inputs[2])}
        result = score_manifest(json.loads(args.manifest.read_text(encoding="utf-8")), raw_bundle=raw_bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
