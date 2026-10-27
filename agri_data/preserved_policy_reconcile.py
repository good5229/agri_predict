"""Reconcile the frozen preserved-feature policy over every original fixed50 key.

This is a descriptive posthoc reconciliation only: it never refits models,
reselects crops, runs significance tests, or edits the original report.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from . import full_history_return_eval as price_helper

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "fixed50Manifest": ROOT / "outputs/experiments/five-crop-specialists-20261009/forecast-manifest.json",
    "evaluationManifest": ROOT / "outputs/experiments/preserved-feature-20261010/evaluation-manifest.json",
    "confirmationDecision": ROOT / "outputs/experiments/preserved-feature-20261010/confirmation-decision.json",
    "evaluationReport": ROOT / "outputs/experiments/preserved-feature-20261010/evaluation-report.json",
    "prices": ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json",
}
EXPECTED_SHA256 = {
    "fixed50Manifest": "1497ee362945fcab45353a7b2403be752af3bf63c5f584cc0aedb2cc111ceaa8",
    "evaluationManifest": "35164a11143768f26e7567e80643ec323c7279083ac5e6ef166a7fe8d185aa20",
    "confirmationDecision": "2a8f026299af4aeca025b8f86852da6302964a12fe5497af9b8c20ccb3211ad6",
    "evaluationReport": "d4fbb06a983e95159aeae5fef095a280f996543d022ecf65c462709ebbcb4708",
    "prices": "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8",
}
PRODUCTS = ("cabbage", "radish", "onion", "garlic", "pepper")
HORIZONS = tuple(range(1, 11))
EVAL_START, EVAL_END = dt.date(2026, 1, 1), dt.date(2026, 9, 27)
CANDIDATE = "preserved106Blend"


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _date(value: Any) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _prediction(record: Any) -> float | None:
    if not isinstance(record, dict) or record.get("status") != "issued":
        return None
    return _positive(record.get("prediction"))


def _baseline_prediction(row: dict[str, Any]) -> float | None:
    predictions = row.get("predictions")
    return _positive(predictions.get("fixed50")) if isinstance(predictions, dict) else None


def _candidate_prediction(row: dict[str, Any] | None) -> float | None:
    models = row.get("models") if isinstance(row, dict) else None
    return _prediction(models.get(CANDIDATE)) if isinstance(models, dict) else None


def _manifest_index(rows: Any, *, key_fields: tuple[str, str, str], label: str) -> dict[tuple[str, dt.date, int], dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} forecast rows must be a list")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"malformed {label} forecast row")
        crop = row.get(key_fields[0])
        day = _date(row.get(key_fields[1]))
        horizon = row.get(key_fields[2])
        if crop not in PRODUCTS or day is None or isinstance(horizon, bool) or horizon not in HORIZONS:
            raise ValueError(f"invalid {label} forecast key")
        key = (crop, day, int(horizon))
        if key in indexed:
            raise ValueError(f"duplicate {label} forecast key: {key}")
        indexed[key] = row
    return indexed


def _metric(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    usable = [row for row in rows if row.get("actual") is not None and row.get(field) is not None]
    if not usable:
        return {"n": 0, "uniqueTargetCount": 0, "firstTarget": None, "lastTarget": None, "mae": None, "mape": None}
    errors = [abs(row[field] - row["actual"]) for row in usable]
    targets = sorted({row["target"] for row in usable})
    return {"n": len(usable), "uniqueTargetCount": len(targets), "firstTarget": targets[0], "lastTarget": targets[-1],
            "mae": math.fsum(errors) / len(errors),
            "mape": 100 * math.fsum(error / row["actual"] for error, row in zip(errors, usable)) / len(usable)}


def _validate_inputs(docs: dict[str, dict[str, Any]]) -> list[str]:
    baseline = docs["fixed50Manifest"]
    evaluation = docs["evaluationManifest"]
    decision = docs["confirmationDecision"]
    report = docs["evaluationReport"]
    prices = docs["prices"]
    if baseline.get("phase") != "forecast_only" or baseline.get("targetValuesAttached") is not False:
        raise ValueError("fixed50 input must be answer-free forecast-only")
    if evaluation.get("phase") != "forecast_only" or evaluation.get("phaseName") != "evaluation" or evaluation.get("targetValuesAttached") is not False:
        raise ValueError("preserved-feature evaluation input must be answer-free")
    if decision.get("schemaVersion") != 1 or decision.get("kind") != "preserved_feature_confirmation_gate":
        raise ValueError("invalid preserved-feature confirmation decision")
    unsigned = dict(decision)
    saved_digest = unsigned.pop("decisionSha256", None)
    if saved_digest != _sha(_canonical(unsigned)):
        raise ValueError("confirmation decision digest mismatch")
    selected = decision.get("selectedCrops")
    if not isinstance(selected, list) or any(crop not in PRODUCTS for crop in selected) or len(selected) != len(set(selected)):
        raise ValueError("confirmation decision has invalid selected crops")
    if evaluation.get("selectedCrops") != selected:
        raise ValueError("evaluation manifest selection differs from immutable confirmation decision")
    if report.get("phase") != "scored" or report.get("scoredPhase") != "evaluation":
        raise ValueError("pinned original evaluation report has wrong phase")
    if report.get("selectedCrops") != selected or report.get("confirmationDecisionSha256") != saved_digest:
        raise ValueError("original evaluation report does not bind to confirmation decision")
    report_unsigned = dict(report)
    report_digest = report_unsigned.pop("reportSha256", None)
    if report_digest != _sha(_canonical(report_unsigned)):
        raise ValueError("pinned original evaluation report digest mismatch")
    if report.get("forecastManifestSha256") != evaluation.get("manifestSha256"):
        raise ValueError("original evaluation report is not bound to the pinned evaluation manifest")
    input_hashes = evaluation.get("inputSha256", {})
    if input_hashes.get("confirmationDecisionFile") != EXPECTED_SHA256["confirmationDecision"]:
        raise ValueError("evaluation manifest is not bound to the pinned confirmation decision file")
    if input_hashes.get("prices") != EXPECTED_SHA256["prices"]:
        raise ValueError("evaluation manifest is not bound to the pinned prices")
    if report.get("priceInputSha256") != EXPECTED_SHA256["prices"]:
        raise ValueError("original report is not bound to the pinned price snapshot")
    if prices.get("schemaVersion") != 1:
        raise ValueError("prices must use dashboard schemaVersion 1")
    _manifest_index(baseline.get("forecasts"), key_fields=("product", "target", "horizon"), label="fixed50")
    _manifest_index(evaluation.get("forecasts"), key_fields=("product", "target", "horizon"), label="preserved evaluation")
    return selected


def _reconcile_documents(docs: dict[str, dict[str, Any]], file_hashes: dict[str, str], file_paths: dict[str, str]) -> dict[str, Any]:
    selected = _validate_inputs(docs)
    baseline = _manifest_index(docs["fixed50Manifest"].get("forecasts"), key_fields=("product", "target", "horizon"), label="fixed50")
    candidate = _manifest_index(docs["evaluationManifest"].get("forecasts"), key_fields=("product", "target", "horizon"), label="preserved evaluation")
    series = price_helper._series(docs["prices"])
    actuals = {(crop, day): price for crop, observations in series.items() for day, price in observations.items()
               if EVAL_START <= day <= EVAL_END}

    keys = []
    for key, base_row in sorted(baseline.items()):
        crop, day, horizon = key
        if not EVAL_START <= day <= EVAL_END:
            continue
        fixed = _baseline_prediction(base_row)
        if fixed is None:
            continue
        eval_row = candidate.get(key)
        origin = _date(base_row.get("origin"))
        if origin != day - dt.timedelta(days=horizon):
            raise ValueError(f"fixed50 target origin mismatch for {key}")
        if eval_row is not None:
            eval_origin = _date(eval_row.get("origin"))
            if eval_origin != day - dt.timedelta(days=horizon):
                raise ValueError(f"preserved candidate target origin mismatch for {key}")
            eligible = eval_row.get("eligible")
            if eligible is not (eval_origin >= EVAL_START):
                raise ValueError(f"preserved candidate eligibility mismatch for {key}")
            if eval_row.get("selectedCrop") is not (crop in selected):
                raise ValueError(f"preserved candidate selected-crop flag mismatch for {key}")
        else:
            eligible = origin >= EVAL_START
        candidate_value = _candidate_prediction(eval_row) if crop in selected else None
        fallback = crop in selected and candidate_value is None
        fallback_reason = None
        candidate_record = eval_row.get("models", {}).get(CANDIDATE) if isinstance(eval_row, dict) and isinstance(eval_row.get("models"), dict) else None
        if fallback:
            if not eligible:
                fallback_reason = "preperiod_origin_candidate_not_issued"
            elif eval_row is None:
                fallback_reason = "candidate_forecast_row_missing"
            else:
                fallback_reason = (candidate_record or {}).get("reason") or (candidate_record or {}).get("status") or "candidate_not_issued"
        policy = candidate_value if crop in selected and candidate_value is not None else fixed
        keys.append({"product": crop, "target": day.isoformat(), "horizon": horizon,
                     "origin": origin.isoformat(), "eligibleOrigin": bool(eligible),
                     "actual": actuals.get((crop, day)), "selectedCrop": crop in selected,
                     "fixed50": fixed, "candidate": candidate_value, "policy": policy,
                     "fallback": fallback, "fallbackReason": fallback_reason,
                     "preperiodOriginFallback": bool(fallback and not eligible)})

    cells = []
    by_cell = {}
    candidate_failures = []
    for row in keys:
        if row["fallback"]:
            candidate_failures.append({"product": row["product"], "target": row["target"], "horizon": row["horizon"],
                "origin": row["origin"], "eligibleOrigin": row["eligibleOrigin"], "reason": row["fallbackReason"],
                "fallback": "fixed50"})
    for crop in PRODUCTS:
        for horizon in HORIZONS:
            cell_rows = [row for row in keys if row["product"] == crop and row["horizon"] == horizon]
            policy_metric = _metric(cell_rows, "policy")
            fixed_metric = _metric(cell_rows, "fixed50")
            candidate_issued = sum(row["candidate"] is not None for row in cell_rows)
            fallback_rows = [row for row in cell_rows if row["fallback"]]
            item = {"product": crop, "horizon": horizon, "selectedCropUsesPreserved106": crop in selected,
                "fixed50IssuedKeys": len(cell_rows), "eligibleOriginFixed50Keys": sum(row["eligibleOrigin"] for row in cell_rows),
                "preperiodOriginFixed50Keys": sum(not row["eligibleOrigin"] for row in cell_rows),
                "candidateIssuedOnFixed50Keys": candidate_issued,
                "candidateFailures": len(fallback_rows),
                "preperiodOriginFallbacks": sum(row["preperiodOriginFallback"] for row in fallback_rows),
                "positiveActualPairs": policy_metric["n"], "uniqueTargets": policy_metric["uniqueTargetCount"],
                "firstTarget": policy_metric["firstTarget"], "lastTarget": policy_metric["lastTarget"],
                "policyMae": policy_metric["mae"], "fixed50Mae": fixed_metric["mae"],
                "policyMape": policy_metric["mape"], "fixed50Mape": fixed_metric["mape"]}
            cells.append(item)
            by_cell[(crop, horizon)] = item

    crop_metrics = []
    for crop in PRODUCTS:
        supported = [by_cell[(crop, horizon)] for horizon in HORIZONS if by_cell[(crop, horizon)]["positiveActualPairs"]]
        crop_keys = [row for row in keys if row["product"] == crop]
        crop_metrics.append({"product": crop, "selectedCropUsesPreserved106": crop in selected,
            "horizonsIncluded": len(supported), "positiveActualPairsAcrossHorizons": sum(row["positiveActualPairs"] for row in supported),
            "uniqueTargetsAcrossHorizons": len({row["target"] for row in crop_keys if row["actual"] is not None}),
            "firstTarget": min((row["firstTarget"] for row in supported if row["firstTarget"]), default=None),
            "lastTarget": max((row["lastTarget"] for row in supported if row["lastTarget"]), default=None),
            "equalHorizonPolicyMae": math.fsum(row["policyMae"] for row in supported) / len(supported) if supported else None,
            "equalHorizonFixed50Mae": math.fsum(row["fixed50Mae"] for row in supported) / len(supported) if supported else None,
            "equalHorizonPolicyMape": math.fsum(row["policyMape"] for row in supported) / len(supported) if supported else None,
            "equalHorizonFixed50Mape": math.fsum(row["fixed50Mape"] for row in supported) / len(supported) if supported else None,
            "fixed50IssuedKeys": len(crop_keys),
            "eligibleOriginFixed50Keys": sum(row["eligibleOrigin"] for row in crop_keys),
            "preperiodOriginFixed50Keys": sum(not row["eligibleOrigin"] for row in crop_keys),
            "preperiodOriginFallbacks": sum(row["preperiodOriginFallback"] for row in crop_keys),
            "candidateFailures": sum(row["fallback"] for row in crop_keys)})

    supported_cells = [row for row in cells if row["positiveActualPairs"] > 0]
    complete = len(supported_cells) == len(PRODUCTS) * len(HORIZONS)
    policy_portfolio = {"cellsPossible": len(PRODUCTS) * len(HORIZONS), "cellsIncluded": len(supported_cells),
        "complete50CellCoverage": complete,
        "equalCellPolicyMae": math.fsum(row["policyMae"] for row in supported_cells) / len(supported_cells) if supported_cells else None,
        "equalCellFixed50Mae": math.fsum(row["fixed50Mae"] for row in supported_cells) / len(supported_cells) if supported_cells else None,
        "equalCellPolicyMape": math.fsum(row["policyMape"] for row in supported_cells) / len(supported_cells) if supported_cells else None,
        "equalCellFixed50Mape": math.fsum(row["fixed50Mape"] for row in supported_cells) / len(supported_cells) if supported_cells else None,
        "all50CellPolicyMae": math.fsum(row["policyMae"] for row in supported_cells) / len(supported_cells) if complete else None,
        "all50CellFixed50Mae": math.fsum(row["fixed50Mae"] for row in supported_cells) / len(supported_cells) if complete else None,
        "all50CellPolicyMape": math.fsum(row["policyMape"] for row in supported_cells) / len(supported_cells) if complete else None,
        "all50CellFixed50Mape": math.fsum(row["fixed50Mape"] for row in supported_cells) / len(supported_cells) if complete else None,
        "supportedCellMeansAreDescriptiveOnly": not complete}
    output = {"schemaVersion": 1, "phase": "preserved_policy_reconciled",
        "policy": "confirmation-selected crop uses preserved106Blend whenever issued; all other crops use original fixed50; unavailable selected-crop forecasts fall back to the same fixed50 key",
        "selection": {"selectedCrops": selected, "source": "immutable confirmation decision; evaluation gate does not reselect or erase candidate performance",
            "evaluationGate": docs["evaluationReport"].get("candidateGates")},
        "fixed50TargetRange": [EVAL_START.isoformat(), EVAL_END.isoformat()],
        "coverage": {"originalFixed50IssuedKeys": len(keys), "keysWithPositiveActual": sum(row["actual"] is not None for row in keys),
            "eligibleOriginKeys": sum(row["eligibleOrigin"] for row in keys), "preperiodOriginKeys": sum(not row["eligibleOrigin"] for row in keys),
            "candidateFailures": len(candidate_failures), "preperiodOriginFallbacks": sum(row["preperiodOriginFallback"] for row in keys),
            "policyIssuedKeys": sum(row["policy"] is not None for row in keys)},
        "cellMetrics": cells, "equalHorizonByCrop": crop_metrics, "portfolioEqualWeightAcross50Cells": policy_portfolio,
        "candidateFailureTargets": candidate_failures,
        "inputs": {name: {"path": file_paths[name], "sha256": file_hashes[name]} for name in sorted(file_hashes)}}
    output["reportSha256"] = _sha(_canonical(output))
    return output


def reconcile_bytes(raw_inputs: dict[str, bytes], *, file_paths: dict[str, str] | None = None) -> dict[str, Any]:
    if set(raw_inputs) != set(INPUTS):
        raise ValueError(f"expected exactly these input artifacts: {sorted(INPUTS)}")
    hashes = {name: _sha(raw) for name, raw in raw_inputs.items()}
    for name, expected in EXPECTED_SHA256.items():
        if hashes[name] != expected:
            raise ValueError(f"pinned input hash mismatch for {name}")
    docs = {name: json.loads(raw_inputs[name]) for name in raw_inputs}
    paths = file_paths or {name: str(path.relative_to(ROOT)) for name, path in INPUTS.items()}
    return _reconcile_documents(docs, hashes, paths)


def write_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    raw_inputs = {name: path.read_bytes() for name, path in INPUTS.items()}
    result = reconcile_bytes(raw_inputs)
    write_exclusive(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
