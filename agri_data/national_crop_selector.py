"""Frozen per-crop selection over previously validated national forecast methods."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from . import direct_horizon, national_median_ensemble as frozen, national_price_selection as base

PROTOCOL_ID = "national-crop-selector-2026-10-09"
PRODUCTS = base.PRODUCTS
HORIZONS = base.HORIZONS
SELECTION_METHODS = ("ridge_alpha_100", "ridge_alpha_10", "fixed_blend_50_50", "calendar_sarima")
TIE_ORDER = SELECTION_METHODS
OUTPUT_METHODS = ("crop_selected", "ridge_alpha_100", "calendar_sarima", "persistence")
COMPARATORS = ("ridge_alpha_100", "calendar_sarima", "persistence")
BANDWIDTHS = (9, 28)
MIN_PAIRS = 30
SELECTION_START = dt.date(2026, 4, 1)
SELECTION_END = dt.date(2026, 6, 19)  # earliest Jul 1, horizon 10 has cutoff Jun 19
FRESH_START = dt.date(2026, 7, 1)
FRESH_END = dt.date(2026, 9, 30)
PROJECT_ROOT = frozen.PROJECT_ROOT
DEFAULT_ARCHIVE = frozen.DEFAULT_ARCHIVE
DEFAULT_COVERAGE = frozen.DEFAULT_COVERAGE


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hash(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def _positive(value: Any) -> float | None:
    return frozen._positive(value)


def _day(value: str) -> dt.date:
    return frozen._date(value)


def _load_inputs(raw_bundles: dict, development_path: str | Path | None = None,
                 fresh_path: str | Path | None = None) -> tuple[dict, dict, dict, dict]:
    development = frozen._load_frozen_manifest("development", development_path)
    fresh = frozen._load_frozen_manifest("fresh", fresh_path)
    dev_source, fresh_source = frozen._load_sources(raw_bundles)
    if development.get("sourceDigests") != dev_source.get("sourceDigests"):
        raise ValueError("development source digest mismatch")
    if fresh.get("sourceDigests") != fresh_source.get("sourceDigests"):
        raise ValueError("fresh source digest mismatch")
    rows = frozen._merge_forecasts(development, fresh)
    for row in rows:
        if not isinstance(row.get("predictions"), dict) or not set(SELECTION_METHODS).issubset(row["predictions"]):
            raise ValueError("frozen forecast missing one of four selector methods")
    return development, fresh, dev_source, fresh_source


def _selection(rows: list[dict], source: dict) -> tuple[dict, dict]:
    """Select one method per crop from common, mature Apr 1-Jun 19 labels."""
    selected_rows = [r for r in rows if SELECTION_START <= _day(r["target"]) <= SELECTION_END]
    actuals = base._prices(source)
    common_by_crop_horizon: dict[tuple[str, int], list[dict]] = defaultdict(list)
    exclusion_states: dict[str, int] = defaultdict(int)
    for row in selected_rows:
        product, target = row["product"], _day(row["target"])
        actual = actuals[product].get(target)
        if actual is None or any(_positive(row["predictions"].get(method)) is None for method in SELECTION_METHODS):
            state, _ = base._status(source, product, target, None)
            if actual is not None:
                state = "incomplete_common_selector_forecast"
            exclusion_states[state] += 1
            continue
        common_by_crop_horizon[(product, row["horizon"])].append({**row, "actual": actual})
    choices, diagnostics = {}, {}
    for product in PRODUCTS:
        by_horizon = {h: common_by_crop_horizon[(product, h)] for h in HORIZONS}
        counts = {str(h): len(by_horizon[h]) for h in HORIZONS}
        complete = all(len(by_horizon[h]) >= MIN_PAIRS for h in HORIZONS)
        metrics = {}
        for method in SELECTION_METHODS:
            horizon_metrics = {}
            for h in HORIZONS:
                cell = by_horizon[h]
                horizon_metrics[str(h)] = {"n": len(cell), **direct_horizon._metrics(
                    [r["actual"] for r in cell], [r["predictions"][method] for r in cell])}
            valid = [horizon_metrics[str(h)] for h in HORIZONS]
            metrics[method] = {"byHorizon": horizon_metrics,
                "equalHorizonMae": float(np.mean([v["mae"] for v in valid])) if complete else None,
                "equalHorizonMape": float(np.mean([v["mape"] for v in valid])) if complete else None}
        planned = [r for r in selected_rows if r["product"] == product]
        issued = {method: sum(_positive(r["predictions"].get(method)) is not None for r in planned)
                  for method in SELECTION_METHODS}
        baseline_mae = metrics["ridge_alpha_100"]["equalHorizonMae"]
        eligible = [method for method in SELECTION_METHODS if complete and baseline_mae is not None
                    and issued[method] >= issued["ridge_alpha_100"]
                    and metrics[method]["equalHorizonMae"] <= 1.01 * baseline_mae]
        if not complete:
            chosen, status, reason = "ridge_alpha_100", "fallback", "one_or_more_horizons_below_30_common_pairs"
        elif not eligible:
            chosen, status, reason = "ridge_alpha_100", "fallback", "no_method_met_mae_and_coverage_eligibility"
        else:
            chosen = min(eligible, key=lambda method: (metrics[method]["equalHorizonMape"], TIE_ORDER.index(method)))
            status, reason = "selected", None
        choices[product] = {"method": chosen, "status": status, "fallbackReason": reason}
        diagnostics[product] = {"selectionWindow": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
            "sampleCountsByHorizon": counts, "minimumPairsPerHorizon": MIN_PAIRS, "complete10Horizons": complete,
            "issuedCoverageByMethod": issued, "eligibleMethods": eligible, "selectedMethod": chosen,
            "selectionStatus": status, "fallbackReason": reason, "metricsByMethod": metrics}
    return {"choices": choices, "diagnostics": diagnostics,
            "exclusionCounts": dict(sorted(exclusion_states.items())),
            "plannedPairs": len(selected_rows), "commonPositivePairs": sum(len(v) for v in common_by_crop_horizon.values())}


def _make_candidate_row(row: dict, choice: dict) -> dict:
    method = choice["method"]
    selected = _positive(row["predictions"].get(method))
    ridge = _positive(row["predictions"].get("ridge_alpha_100"))
    if selected is None:
        value = ridge
        lineage = {"fallback": True, "selectedMethod": method,
                   "fallbackReason": f"selected_method_invalid:{method}", "fallbackValueMethod": "ridge_alpha_100"}
    else:
        value = selected
        lineage = {"fallback": False, "selectedMethod": method, "fallbackReason": None}
    return {**{k: row[k] for k in ("product", "origin", "target", "horizon", "cutoff")},
            "selectedMethod": method,
            "predictions": {"crop_selected": value,
                "ridge_alpha_100": row["predictions"].get("ridge_alpha_100"),
                "calendar_sarima": row["predictions"].get("calendar_sarima"),
                "persistence": row["predictions"].get("persistence")},
            "sourceLineage": row.get("lineage", {}), "lineage": lineage}


def build_manifest(development: dict, fresh: dict, raw_bundles: dict) -> dict:
    """Build answer-free Jul-Sep predictions using crop choices fixed through Jun 19."""
    # Re-load and validate pinned source documents and raw-source digests.
    dev_pinned, fresh_pinned, dev_source, fresh_source = _load_inputs(raw_bundles)
    if development != dev_pinned or fresh != fresh_pinned:
        raise ValueError("provided forecast manifest differs from the pinned input")
    rows = frozen._merge_forecasts(development, fresh)
    selection = _selection(rows, dev_source)
    fresh_rows = [r for r in rows if FRESH_START <= _day(r["target"]) <= FRESH_END]
    if len(fresh_rows) != 3680:
        raise ValueError("fresh forecast grid must contain exactly 3,680 planned keys")
    forecasts = [_make_candidate_row(r, selection["choices"][r["product"]]) for r in fresh_rows]
    doc = {"schemaVersion": 1, "protocolId": PROTOCOL_ID, "phase": "forecast_only",
        "targetValuesAttached": False, "inputManifestSha256": dict(frozen.MANIFEST_SHA256),
        "inputManifestContentSha256": dict(frozen.MANIFEST_CONTENT_SHA256),
        "sourceDigests": {"development": dev_source["sourceDigests"], "fresh": fresh_source["sourceDigests"]},
        "codeSha256": _sha(Path(__file__).read_bytes()), "selectionWindow": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
        "selectionMethodPriority": list(TIE_ORDER), "selectionChoices": selection["choices"],
        "selectionProvenance": {product: {k: v for k, v in selection["diagnostics"][product].items()
            if k not in ("metricsByMethod",)} for product in PRODUCTS},
        "selectionExclusionCounts": selection["exclusionCounts"], "plannedSelectionPairs": selection["plannedPairs"],
        "products": list(PRODUCTS), "horizons": list(HORIZONS), "window": [FRESH_START.isoformat(), FRESH_END.isoformat()],
        "methods": list(OUTPUT_METHODS), "plannedForecastCount": len(forecasts), "forecasts": forecasts}
    doc["manifestSha256"] = _hash(doc)
    return doc


def verify_manifest(doc: dict) -> None:
    if doc.get("protocolId") != PROTOCOL_ID or doc.get("phase") != "forecast_only" or doc.get("targetValuesAttached") is not False:
        raise ValueError("not an answer-free crop selector manifest")
    if doc.get("codeSha256") != _sha(Path(__file__).read_bytes()):
        raise ValueError("candidate code hash mismatch")
    body = dict(doc)
    saved = body.pop("manifestSha256", None)
    if saved != _hash(body):
        raise ValueError("candidate manifest hash mismatch")
    forecasts = doc.get("forecasts")
    if not isinstance(forecasts, list) or len(forecasts) != doc.get("plannedForecastCount"):
        raise ValueError("candidate forecast count mismatch")
    seen = set()
    for row in forecasts:
        key = frozen._forecast_identity(row)
        if key in seen:
            raise ValueError("duplicate candidate key")
        seen.add(key)
        if any(k in row for k in ("actual", "targetValue")):
            raise ValueError("answer-bearing field in candidate manifest")
        if row.get("selectedMethod") not in SELECTION_METHODS:
            raise ValueError("candidate selected method invalid")


def _verify_replay(doc: dict, development: dict, fresh: dict, raw_bundles: dict) -> None:
    verify_manifest(doc)
    rebuilt = build_manifest(development, fresh, raw_bundles)
    if (doc.get("forecasts") != rebuilt.get("forecasts") or doc.get("selectionChoices") != rebuilt.get("selectionChoices")
            or doc.get("manifestSha256") != rebuilt.get("manifestSha256")):
        raise ValueError("crop selector candidate manifest replay mismatch")


def _joint(tests: list[dict]) -> list[dict]:
    index = {(r["product"], r["horizon"], r["comparator"], r["loss"]): r for r in tests}
    out = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            for comparator in COMPARATORS:
                ae = index[(product, horizon, comparator, "mae")]
                ape = index[(product, horizon, comparator, "mape")]
                item = {"product": product, "horizon": horizon, "comparator": comparator,
                        "nMae": ae["n"], "nMape": ape["n"], "verdictByBandwidth": {}}
                for bw in BANDWIDTHS:
                    a, p = ae["dm"][str(bw)]["conclusion"], ape["dm"][str(bw)]["conclusion"]
                    verdict = (a if a == p and a in ("improvement", "worsening") else
                               "inference_unavailable" if "inference_unavailable" in (a, p) else "inconclusive")
                    item["verdictByBandwidth"][str(bw)] = {"mae": a, "mape": p, "joint": verdict}
                out.append(item)
    return out


def _score_window(rows: list[dict], source: dict, status_evidence: dict, *, start: dt.date, end: dt.date) -> dict:
    actuals = base._prices(source)
    selected = [r for r in rows if start <= _day(r["target"]) <= end]
    common = []
    for row in selected:
        y = actuals[row["product"]].get(_day(row["target"]))
        if y is not None and all(_positive(row["predictions"].get(m)) is not None for m in OUTPUT_METHODS):
            common.append({**row, "actual": y})
    common_keys = {frozen._forecast_identity(r) for r in common}
    cells = {}
    for product in PRODUCTS:
        for horizon in HORIZONS:
            cell = [r for r in common if r["product"] == product and r["horizon"] == horizon]
            cells[f"{product}/{horizon}"] = {"n": len(cell), "metrics": {method: direct_horizon._metrics(
                [r["actual"] for r in cell], [r["predictions"][method] for r in cell]) for method in OUTPUT_METHODS}}
    complete = all(cells[f"{p}/{h}"]["n"] >= MIN_PAIRS for p in PRODUCTS for h in HORIZONS)
    macro = {method: {loss: sum(cells[f"{p}/{h}"]["metrics"][method][loss] for p in PRODUCTS for h in HORIZONS) / 40 if complete else None
                      for loss in ("mae", "mape")} for method in OUTPUT_METHODS}
    coverage = {method: {"planned": len(selected),
                         "issued": sum(_positive(r["predictions"].get(method)) is not None for r in selected),
                         "issuedOnValidTargets": sum(_day(r["target"]) in actuals[r["product"]]
                                                     and _positive(r["predictions"].get(method)) is not None for r in selected)}
                for method in OUTPUT_METHODS}
    per_crop = {}
    for product in PRODUCTS:
        per_crop[product] = {}
        for method in OUTPUT_METHODS:
            by_h = [cells[f"{product}/{h}"]["metrics"][method] for h in HORIZONS]
            product_rows = [r for r in common if r["product"] == product]
            per_crop[product][method] = {**direct_horizon._metrics([r["actual"] for r in product_rows],
                                                                      [r["predictions"][method] for r in product_rows]),
                "equalHorizonMae": float(np.mean([m["mae"] for m in by_h])) if all(m["mae"] is not None for m in by_h) else None,
                "equalHorizonMape": float(np.mean([m["mape"] for m in by_h])) if all(m["mape"] is not None for m in by_h) else None}
    tests, paired = [], []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            cell = [r for r in common if r["product"] == product and r["horizon"] == horizon]
            for comparator in COMPARATORS:
                for loss in ("mae", "mape"):
                    diffs = {}
                    for row in cell:
                        y, c, b = row["actual"], row["predictions"]["crop_selected"], row["predictions"][comparator]
                        diffs[_day(row["target"])] = abs(c-y)-abs(b-y) if loss == "mae" else 100*(abs(c-y)-abs(b-y))/y
                    tests.append({"product": product, "horizon": horizon, "comparator": comparator,
                                  "loss": loss, "n": len(diffs), "dm": {str(bw): frozen._dm(diffs, bw) for bw in BANDWIDTHS}})
                paired.append({"product": product, "horizon": horizon, "comparator": comparator, "n": len(cell),
                    "meanPairedMaeDifference": float(np.mean([abs(r["predictions"]["crop_selected"]-r["actual"])-abs(r["predictions"][comparator]-r["actual"]) for r in cell])) if cell else None,
                    "meanPairedMapeDifferencePp": float(np.mean([100*(abs(r["predictions"]["crop_selected"]-r["actual"])-abs(r["predictions"][comparator]-r["actual"]))/r["actual"] for r in cell])) if cell else None})
    if len(tests) != len(PRODUCTS) * len(HORIZONS) * len(COMPARATORS) * 2:
        raise AssertionError("frozen DM family size mismatch")
    for bw in BANDWIDTHS:
        frozen._holm_family(tests, bw)
    joint = _joint(tests)
    by_cell = {}
    for product in PRODUCTS:
        for horizon in HORIZONS:
            plan = [r for r in selected if r["product"] == product and r["horizon"] == horizon]
            valid_targets = [r for r in plan if _day(r["target"]) in actuals[product]]
            by_cell[f"{product}/{horizon}"] = {"planned": len(plan), "validTargetCount": len(valid_targets),
                "commonScored": cells[f"{product}/{horizon}"]["n"],
                "issued": {m: sum(_positive(r["predictions"].get(m)) is not None for r in plan) for m in OUTPUT_METHODS}}
    return {"start": start.isoformat(), "end": end.isoformat(), "planned": len(selected),
        "issuedCoverage": coverage, "coverageByCell": by_cell, "validCommonPairs": len(common),
        "exclusionsByState": frozen._exclusions(selected, source, actuals, status_evidence, common_keys),
        "complete40Cells": complete, "macro40EqualCell": macro, "cells": cells, "perCrop": per_crop,
        "pairedCellLosses": paired, "dmComparisons": tests, "jointCellVerdicts": joint,
        "holmFamilySizePerBandwidth": len(tests)}


def _gate(period: dict) -> dict:
    candidate, baseline = "crop_selected", "ridge_alpha_100"
    macro = period["macro40EqualCell"]
    gain = macro[baseline]["mape"] - macro[candidate]["mape"] if macro[baseline]["mape"] is not None and macro[candidate]["mape"] is not None else None
    crop_mape = {p: period["perCrop"][p][candidate]["equalHorizonMape"] - period["perCrop"][p][baseline]["equalHorizonMape"]
                 if period["perCrop"][p][candidate]["equalHorizonMape"] is not None and period["perCrop"][p][baseline]["equalHorizonMape"] is not None else None for p in PRODUCTS}
    crop_mae = {p: 100 * (period["perCrop"][p][candidate]["equalHorizonMae"] - period["perCrop"][p][baseline]["equalHorizonMae"])
                / period["perCrop"][p][baseline]["equalHorizonMae"] if period["perCrop"][p][candidate]["equalHorizonMae"] is not None
                and period["perCrop"][p][baseline]["equalHorizonMae"] not in (None, 0) else None for p in PRODUCTS}
    improved = {(r["product"], r["horizon"]) for r in period["jointCellVerdicts"] if r["comparator"] == baseline
                and r["verdictByBandwidth"]["9"]["joint"] == "improvement"
                and r["verdictByBandwidth"]["28"]["joint"] == "improvement"}
    criteria = {"macroMapeGainAtLeast0_5pp": gain is not None and gain >= 0.5,
                "issuedCoverageNotDecreased": period["issuedCoverage"][candidate]["issued"] >= period["issuedCoverage"][baseline]["issued"],
                "eachCropMapeWorseningAtMost1pp": all(v is not None and v <= 1.0 for v in crop_mape.values()),
                "eachCropMaeWorseningAtMost5pct": all(v is not None and v <= 5.0 for v in crop_mae.values()),
                "atLeastOneCellJointlyImprovedAtHac9AndHac28": bool(improved)}
    complete = period["complete40Cells"]
    return {"status": "unsupported" if not complete else "pass" if all(criteria.values()) else "fail",
        "criteria": criteria, "failureReasons": [] if complete and all(criteria.values()) else
            (["incomplete_40_cell_coverage"] if not complete else [k for k, v in criteria.items() if not v]),
        "thresholds": {"macroMapeGainPpMinimum": 0.5, "cropMapeWorseningPpMaximum": 1.0,
                       "cropMaeWorseningPctMaximum": 5.0, "jointImprovementBandwidthDays": [9, 28],
                       "jointImprovementCellsMinimum": 1}, "macroMapeGainPp": gain,
        "productMapeChangePp": crop_mape, "productMaeChangePct": crop_mae,
        "jointImprovedCells": len(improved)}


def score_manifest(doc: dict, development: dict, fresh: dict, raw_bundles: dict) -> dict:
    _verify_replay(doc, development, fresh, raw_bundles)
    _, _, dev_source, fresh_source = _load_inputs(raw_bundles)
    evidence = base._coverage_status_evidence(raw_bundles["fresh"])
    all_rows = frozen._merge_forecasts(development, fresh)
    selection = _selection(all_rows, dev_source)
    jul_sep = _score_window(doc["forecasts"], fresh_source, evidence, start=FRESH_START, end=FRESH_END)
    jul_sep["predeclaredGate"] = _gate(jul_sep)
    report = {"protocolId": PROTOCOL_ID, "candidateManifestSha256": doc["manifestSha256"],
        "inputManifestSha256": doc["inputManifestSha256"], "sourceDigests": doc["sourceDigests"],
        "selectionPeriodInSampleDiagnostic": {"selectionWindow": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
            "selectionChoices": selection["choices"], "selectionDiagnostics": selection["diagnostics"],
            "plannedPairs": selection["plannedPairs"], "commonPositivePairs": selection["commonPositivePairs"],
            "exclusionCounts": selection["exclusionCounts"],
            "independence": "in_sample_selection_diagnostic_not_confirmation"},
        "freshHoldout": jul_sep, "resultType": "post_hoc_exploratory",
        "independentConfirmation": False,
        "interpretation": f"The per-crop methods were selected using labels through {SELECTION_END.isoformat()}. July-September is a separate later window, but the exploratory selection followed prior inspection of existing forecast results; a pass is not independent confirmation."}
    report["reportSha256"] = _hash(report)
    return report


def _write_json_create_only(path: str | Path, doc: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            destination.unlink()
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Frozen offline per-crop national forecast selector")
    subs = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--archive", default=str(DEFAULT_ARCHIVE))
    common.add_argument("--coverage-report", default=str(DEFAULT_COVERAGE))
    common.add_argument("--development-manifest", default=str(PROJECT_ROOT / frozen.MANIFEST_PATHS["development"]))
    common.add_argument("--fresh-manifest", default=str(PROJECT_ROOT / frozen.MANIFEST_PATHS["fresh"]))
    build = subs.add_parser("build", parents=[common], help="write the answer-free Jul-Sep forecast manifest")
    build.add_argument("--output", required=True)
    score = subs.add_parser("score", parents=[common], help="replay and score a saved Jul-Sep manifest")
    score.add_argument("--manifest", required=True)
    score.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    raw_bundles = {"development": {"archivePath": args.archive},
                   "fresh": {"archivePath": args.archive, "coverageReportPath": args.coverage_report}}
    development, fresh, _, _ = _load_inputs(raw_bundles, args.development_manifest, args.fresh_manifest)
    if args.command == "build":
        output = build_manifest(development, fresh, raw_bundles)
    else:
        candidate = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
        output = score_manifest(candidate, development, fresh, raw_bundles)
    _write_json_create_only(args.output, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
