"""Paired retrospective significance summaries for trained dashboard predictions."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import experiments

PRODUCT_IDS = tuple(experiments.PRODUCT_IDS)
MODEL_IDS = ("ridge", "hist_gb")
CADENCES = tuple(experiments.CADENCES)
HORIZONS = tuple(experiments.HORIZONS)
LOSSES = ("absolute_error", "absolute_percentage_error")
PRIMARY_LAG_DAYS = 9
SENSITIVITY_LAG_DAYS = 28
MIN_PAIRS = 30
ALPHA = 0.05
FAMILY_SIZE = len(PRODUCT_IDS) * len(MODEL_IDS) * len(CADENCES) * len(HORIZONS) * len(LOSSES)


def _day(value: Any, label: str) -> dt.date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc


def _positive_finite(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a positive finite number") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return result


def _prediction_map(rows: Any, label: str, evaluation_start: dt.date,
                    evaluation_end: dt.date, source_dates: set[str], *,
                    require_fit: bool) -> dict[tuple[str, int], dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"{label} predictions must be an array")
    result = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{label} prediction {index} must be an object")
        target = _day(row.get("target"), f"{label} target")
        origin = _day(row.get("origin"), f"{label} origin")
        horizon = row.get("horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
            raise ValueError(f"{label} horizon must be an integer from 1 to 10")
        if target - origin != dt.timedelta(days=horizon):
            raise ValueError(f"{label} target-origin does not match horizon")
        if target < evaluation_start or target > evaluation_end:
            raise ValueError(f"{label} target is outside the evaluation period")
        source_date = _day(row.get("sourceDate"), f"{label} sourceDate")
        if source_date > origin:
            raise ValueError(f"{label} sourceDate is after origin")
        if source_date.isoformat() not in source_dates:
            raise ValueError(f"{label} sourceDate is not an observed price date")
        if (origin - source_date).days > experiments.dashboard.MAX_SOURCE_AGE_DAYS:
            raise ValueError(f"{label} sourceDate exceeds the maximum allowed source age")
        if require_fit:
            fit_cutoff = _day(row.get("fitCutoff"), f"{label} fitCutoff")
            if fit_cutoff > origin:
                raise ValueError(f"{label} fitCutoff is after origin")
        predicted = _positive_finite(row.get("predicted"), f"{label} predicted")
        key = (target.isoformat(), horizon)
        if key in result:
            raise ValueError(f"duplicate {label} target/horizon prediction {key}")
        result[key] = {**row, "target": target.isoformat(), "predicted": predicted}
    return result


def _validate_input(source: Any) -> tuple[dict[str, Any], dict[str, Any], dt.date, dt.date]:
    if not isinstance(source, dict) or source.get("schemaVersion") != 2:
        raise ValueError("input must be a dashboard schemaVersion 2 model export")
    meta = source.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("input has invalid metadata")
    evaluation_start = _day(meta.get("evaluationStart"), "evaluationStart")
    evaluation_end = _day(meta.get("evaluationEnd"), "evaluationEnd")
    if evaluation_start > evaluation_end:
        raise ValueError("evaluationStart must not be after evaluationEnd")
    products = source.get("products")
    if not isinstance(products, list) or len(products) != len(PRODUCT_IDS):
        raise ValueError("input must contain exactly five products")
    product_map: dict[str, Any] = {}
    actual_maps: dict[str, dict[str, float]] = {}
    for product in products:
        if not isinstance(product, dict) or product.get("id") not in PRODUCT_IDS:
            raise ValueError("input contains an unknown product")
        product_id = product["id"]
        if product_id in product_map:
            raise ValueError(f"duplicate product {product_id}")
        product_map[product_id] = product
        observations = product.get("observations")
        if not isinstance(observations, list):
            raise ValueError(f"{product_id} observations must be an array")
        actuals = {}
        for row in observations:
            if not isinstance(row, dict):
                raise ValueError(f"{product_id} observation must be an object")
            day = _day(row.get("date"), f"{product_id} observation date")
            price = _positive_finite(row.get("actual"), f"{product_id} actual")
            if day.isoformat() in actuals:
                raise ValueError(f"duplicate {product_id} observation date {day.isoformat()}")
            actuals[day.isoformat()] = price
        actual_maps[product_id] = actuals
    if set(product_map) != set(PRODUCT_IDS):
        raise ValueError("input must contain every expected product")

    predictions: dict[str, Any] = {}
    expected_runs = {(model, cadence) for model in MODEL_IDS for cadence in CADENCES}
    for product_id, product in product_map.items():
        source_dates = set(actual_maps[product_id])
        baseline = _prediction_map(product.get("predictions"), f"{product_id}/baseline",
                                   evaluation_start, evaluation_end, source_dates, require_fit=False)
        for key, row in baseline.items():
            if key[0] not in actual_maps[product_id]:
                raise ValueError(f"{product_id} baseline prediction has no observed target {key[0]}")
        runs = product.get("runs")
        if not isinstance(runs, list):
            raise ValueError(f"{product_id} runs must be an array")
        run_map = {}
        for run in runs:
            if not isinstance(run, dict):
                raise ValueError(f"{product_id} run must be an object")
            key = (run.get("modelId"), run.get("cadence"))
            if key not in expected_runs:
                raise ValueError(f"{product_id} contains an unsupported model/cadence")
            if key in run_map:
                raise ValueError(f"duplicate {product_id} run {key}")
            run_map[key] = run
        if set(run_map) != expected_runs:
            raise ValueError(f"{product_id} must contain all 8 trained model/cadence runs")
        for run_key, run in run_map.items():
            label = f"{product_id}/{run_key[0]}/{run_key[1]}"
            candidate = _prediction_map(run.get("predictions"), label, evaluation_start,
                                        evaluation_end, source_dates, require_fit=True)
            for key in candidate:
                if key[0] not in actual_maps[product_id]:
                    raise ValueError(f"{label} prediction has no observed target {key[0]}")
            predictions[(product_id, *run_key)] = candidate
        predictions[(product_id, "baseline")] = baseline
    return product_map, predictions, evaluation_start, evaluation_end


def _losses(candidate: float, baseline: float, actual: float) -> tuple[float, float, float, float]:
    candidate_ae = abs(candidate - actual)
    baseline_ae = abs(baseline - actual)
    return candidate_ae, baseline_ae, candidate_ae / actual * 100, baseline_ae / actual * 100


def _dm_stats(candidate_rows: list[dict[str, Any]], baseline_rows: list[dict[str, Any]],
              actuals: dict[tuple[str, int], float], horizon: int, loss: str,
              max_lag: int, n: int, mean_difference: float | None) -> dict[str, Any]:
    if n < MIN_PAIRS:
        return {"n": n, "status": "unavailable_insufficient_pairs", "statistic": None, "pValue": None,
                "meanLossDifference": mean_difference, "calendarHacMaxLagDays": max_lag}
    dm = experiments._dm(candidate_rows, baseline_rows, actuals, horizon, loss, max_lag=max_lag)
    if dm.get("status") != "available":
        return {"n": n, "status": dm.get("status", "unavailable"), "statistic": None, "pValue": None,
                "meanLossDifference": mean_difference, "calendarHacMaxLagDays": max_lag}
    return {"n": n, "status": "available", "statistic": dm["statistic"], "pValue": dm["pValue"],
            "meanLossDifference": mean_difference, "calendarHacMaxLagDays": max_lag}


def _paired_comparison(candidate: dict[tuple[str, int], dict[str, Any]],
                       baseline: dict[tuple[str, int], dict[str, Any]],
                       actuals: dict[str, float], horizon: int, loss: str,
                       max_lag: int) -> dict[str, Any]:
    actual_keys = {(day, horizon) for day in actuals}
    candidate_keys = {key for key in candidate if key[1] == horizon}
    baseline_keys = {key for key in baseline if key[1] == horizon}
    matched = sorted(actual_keys & candidate_keys & baseline_keys)
    candidate_rows = [candidate[key] for key in matched]
    baseline_rows = [baseline[key] for key in matched]
    losses = []
    for key in matched:
        c_ae, b_ae, c_ape, b_ape = _losses(candidate[key]["predicted"], baseline[key]["predicted"], actuals[key[0]])
        losses.append((c_ae, b_ae) if loss == "absolute_error" else (c_ape, b_ape))
    n = len(losses)
    candidate_mean = sum(pair[0] for pair in losses) / n if n else None
    baseline_mean = sum(pair[1] for pair in losses) / n if n else None
    difference = candidate_mean - baseline_mean if n else None
    actual_horizon_count = len(actuals)
    metric = {
        "n": n,
        "eligibleActualCount": actual_horizon_count,
        "candidateMissingCount": len(actual_keys - candidate_keys),
        "baselineMissingCount": len(actual_keys - baseline_keys),
        "unmatchedPairCount": actual_horizon_count - n,
        "candidateMeanLoss": candidate_mean,
        "baselineMeanLoss": baseline_mean,
        "meanLossDifference": difference,
        "matchedDateStart": matched[0][0] if matched else None,
        "matchedDateEnd": matched[-1][0] if matched else None,
    }
    actual_map = {(day, horizon): actuals[day] for day, _ in matched}
    test = _dm_stats(candidate_rows, baseline_rows, actual_map, horizon, loss, max_lag, n, difference)
    metric["test"] = test
    return metric


def _apply_fixed_holm(records: list[dict[str, Any]], key: str) -> None:
    rows = []
    for record in records:
        test = record[key]["test"]
        rows.append({"pValue": test["pValue"] if test["pValue"] is not None else 1.0,
                     "record": record})
    experiments._apply_holm(rows)
    for row in rows:
        record, test = row["record"], row["record"][key]["test"]
        test["holmAdjustedPValue"] = row["holmAdjustedPValue"]
        test["holmSignificantAt005"] = bool(test["status"] == "available" and row["holmAdjustedPValue"] < ALPHA)


def _conclusion(primary_ae: dict[str, Any], primary_ape: dict[str, Any]) -> dict[str, Any]:
    tests = (primary_ae["test"], primary_ape["test"])
    negative = [test["status"] == "available" and test["meanLossDifference"] is not None
                and test["meanLossDifference"] < 0 and test["holmAdjustedPValue"] < ALPHA for test in tests]
    positive = [test["status"] == "available" and test["meanLossDifference"] is not None
                and test["meanLossDifference"] > 0 and test["holmAdjustedPValue"] < ALPHA for test in tests]
    if all(negative):
        status = "improved"
    elif any(negative) and any(positive):
        status = "mixed_evidence"
    elif any(positive):
        status = "worsened"
    elif any(test["status"] != "available" for test in tests):
        status = "unavailable"
    else:
        status = "no_evidence"
    return {"status": status,
            "improvedRequiresBothLosses": True,
            "primaryImproved": status == "improved",
            "sensitivityStatus": None}


def analyze(source: dict[str, Any], source_sha256: str | None = None) -> dict[str, Any]:
    product_map, predictions, evaluation_start, evaluation_end = _validate_input(source)
    if source_sha256 is None:
        source_sha256 = hashlib.sha256(json.dumps(source, ensure_ascii=False, sort_keys=True,
                                                   separators=(",", ":")).encode("utf-8")).hexdigest()
    records = []
    record_by_key = {}
    for product_id in PRODUCT_IDS:
        product = product_map[product_id]
        actuals = {row["date"]: _positive_finite(row["actual"], f"{product_id} actual")
                   for row in product["observations"] if evaluation_start <= _day(row["date"], "observation date") <= evaluation_end}
        baseline = predictions[(product_id, "baseline")]
        for model_id in MODEL_IDS:
            for cadence in CADENCES:
                candidate = predictions[(product_id, model_id, cadence)]
                for horizon in HORIZONS:
                    for loss in LOSSES:
                        primary = _paired_comparison(candidate, baseline, actuals, horizon, loss, PRIMARY_LAG_DAYS)
                        sensitivity = _paired_comparison(candidate, baseline, actuals, horizon, loss, SENSITIVITY_LAG_DAYS)
                        record = {"productId": product_id, "modelId": model_id, "cadence": cadence,
                                  "horizon": horizon, "loss": loss,
                                  "primary": primary, "sensitivity": sensitivity}
                        key = (product_id, model_id, cadence, horizon, loss)
                        records.append(record)
                        record_by_key[key] = record
    _apply_fixed_holm(records, "primary")
    _apply_fixed_holm(records, "sensitivity")

    conclusions = []
    for product_id in PRODUCT_IDS:
        for model_id in MODEL_IDS:
            for cadence in CADENCES:
                for horizon in HORIZONS:
                    ae = record_by_key[(product_id, model_id, cadence, horizon, "absolute_error")]
                    ape = record_by_key[(product_id, model_id, cadence, horizon, "absolute_percentage_error")]
                    conclusion = _conclusion(ae["primary"], ape["primary"])
                    sensitivity_tests = (ae["sensitivity"]["test"], ape["sensitivity"]["test"])
                    if all(test["status"] == "available" for test in sensitivity_tests):
                        if all(test["meanLossDifference"] < 0 and test["holmAdjustedPValue"] < ALPHA
                               for test in sensitivity_tests):
                            sensitivity_status = "improved"
                        elif any(test["meanLossDifference"] > 0 and test["holmAdjustedPValue"] < ALPHA
                                 for test in sensitivity_tests):
                            sensitivity_status = "worsened_or_mixed"
                        else:
                            sensitivity_status = "no_evidence"
                    else:
                        sensitivity_status = "unavailable"
                    conclusion["sensitivityStatus"] = sensitivity_status
                    conclusions.append({"productId": product_id, "modelId": model_id, "cadence": cadence,
                                        "horizon": horizon, **conclusion})
    return {
        "schemaVersion": 1,
        "generatedAt": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "protocol": {
            "name": "retrospective paired trained-model versus persistence inference",
            "evaluationStart": evaluation_start.isoformat(),
            "evaluationEnd": evaluation_end.isoformat(),
            "inputSha256": source_sha256,
            "primary": {"method": "Diebold-Mariano asymptotic normal, two-sided; calendar Bartlett HAC",
                        "maxLagDays": PRIMARY_LAG_DAYS, "alpha": ALPHA, "minimumPairs": MIN_PAIRS,
                        "holmFamilySize": FAMILY_SIZE, "unavailablePValueForHolm": 1.0},
            "sensitivity": {"method": "Diebold-Mariano asymptotic normal, two-sided; calendar Bartlett HAC",
                            "maxLagDays": SENSITIVITY_LAG_DAYS, "alpha": ALPHA,
                            "minimumPairs": MIN_PAIRS, "holmFamilySize": FAMILY_SIZE,
                            "unavailablePValueForHolm": 1.0},
            "losses": {"absolute_error": "|predicted-actual| in 원/kg",
                       "absolute_percentage_error": "100×|predicted-actual|/actual"},
            "family": "5 products × 2 models × 4 cadences × 10 horizons × 2 losses = 800 tests; primary and sensitivity adjusted independently",
            "improvementRule": "both AE and APE mean loss differences <0 and Holm-adjusted primary p<0.05",
            "scope": "retrospective exploratory analysis; not preregistered confirmatory evidence",
            "interpretationLimitations": [
                "유의하지 않음은 동등성이나 차이 없음의 증거가 아닙니다.",
                "30쌍 기준은 실무상 최소 기준이며 추론의 정확성을 보장하지 않습니다.",
                "점근적 정규 DM 검정은 표본이 작거나 분포가 불안정할 때 근사 오차가 큽니다.",
                "결측 날짜가 정보성 결측일 수 있고 구조 변화와 추정 불확실성을 제거하지 않습니다.",
                "9일 및 28일 HAC 민감도는 선택한 대역폭에 대한 민감도이며 모든 의존성을 제거한다고 보장하지 않습니다.",
            ],
            "references": [
                "https://doi.org/10.1080/07350015.1995.10524599",
                "https://stat.ethz.ch/R-manual/R-devel/library/stats/html/p.adjust.html",
                "https://www.federalreserve.gov/pubs/ifdp/2012/1060/ifdp1060.htm",
            ],
            "environment": {"python": platform.python_version(), "implementation": platform.python_implementation()},
        },
        "comparisons": records,
        "conclusions": conclusions,
    }


def markdown_report(report: dict[str, Any]) -> str:
    protocol = report["protocol"]
    conclusions = report["conclusions"]
    counts = {}
    for row in conclusions:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    lines = ["# Paired model significance report", "",
             f"Evaluation: {protocol['evaluationStart']} to {protocol['evaluationEnd']}",
             f"Input SHA-256: `{protocol['inputSha256']}`", "",
             "Retrospective exploratory analysis; not preregistered confirmatory evidence. "
             "Non-rejection is not evidence of equivalence.", "",
             f"Primary: calendar Bartlett HAC, max lag {PRIMARY_LAG_DAYS} days; Holm family size {FAMILY_SIZE}.",
             f"Sensitivity: max lag {SENSITIVITY_LAG_DAYS} days; Holm family size {FAMILY_SIZE}.",
             "", "## Pair conclusions", "",
             "| Status | Count |", "|---|---:|"]
    lines.extend(f"| {status} | {counts.get(status, 0)} |" for status in
                 ("improved", "worsened", "mixed_evidence", "no_evidence", "unavailable"))
    lookup = {(r["productId"], r["modelId"], r["cadence"], r["horizon"], r["loss"]): r
              for r in report["comparisons"]}
    outcome_lookup = {(row["productId"], row["modelId"], row["cadence"], row["horizon"]): row
                      for row in conclusions}
    lines.extend(["", "## Horizon 1 daily overview", "",
                  "| Product | Model | Conclusion | n AE/APE | AE Δ | AE Holm p | APE Δ | APE Holm p | 28d sensitivity |",
                  "|---|---|---|---:|---:|---:|---:|---:|---|"])
    for product_id in PRODUCT_IDS:
        for model_id in MODEL_IDS:
            base = (product_id, model_id, "daily", 1)
            ae = lookup[(*base, "absolute_error")]["primary"]
            ape = lookup[(*base, "absolute_percentage_error")]["primary"]
            outcome = outcome_lookup[base]
            ae_test, ape_test = ae["test"], ape["test"]
            ae_delta = "—" if ae["meanLossDifference"] is None else f"{ae['meanLossDifference']:.6g}"
            ape_delta = "—" if ape["meanLossDifference"] is None else f"{ape['meanLossDifference']:.6g}"
            ae_p = (f"{ae_test['holmAdjustedPValue']:.4g}" if ae_test["status"] == "available" else "—")
            ape_p = (f"{ape_test['holmAdjustedPValue']:.4g}" if ape_test["status"] == "available" else "—")
            lines.append(f"| {product_id} | {model_id} | {outcome['status']} | {ae['n']}/{ape['n']} | "
                         f"{ae_delta} | {ae_p} | {ape_delta} | {ape_p} | {outcome['sensitivityStatus']} |")
    lines.extend(["", "## Improved comparisons", "", "| Product | Model | Cadence | Horizon | AE Δ | AE p(Holm) | APE Δ | APE p(Holm) | 28d sensitivity |",
                  "|---|---|---|---:|---:|---:|---:|---:|---|"])
    for conclusion in conclusions:
        if conclusion["status"] != "improved":
            continue
        base = (conclusion["productId"], conclusion["modelId"], conclusion["cadence"], conclusion["horizon"])
        ae = lookup[(*base, "absolute_error")]["primary"]["test"]
        ape = lookup[(*base, "absolute_percentage_error")]["primary"]["test"]
        lines.append(f"| {base[0]} | {base[1]} | {base[2]} | {base[3]} | {ae['meanLossDifference']:.6g} | "
                     f"{ae['holmAdjustedPValue']:.4g} | {ape['meanLossDifference']:.6g} | "
                     f"{ape['holmAdjustedPValue']:.4g} | {conclusion['sensitivityStatus']} |")
    if counts.get("improved", 0) == 0:
        lines.append("| — | — | — | — | — | — | — | — | — |")
    lines.extend(["", "## Worsened or mixed comparisons", "",
                  "| Product | Model | Cadence | Horizon | Status | AE Δ | APE Δ | 28d sensitivity |",
                  "|---|---|---|---:|---|---:|---:|---|"])
    for conclusion in conclusions:
        if conclusion["status"] not in ("worsened", "mixed_evidence"):
            continue
        base = (conclusion["productId"], conclusion["modelId"], conclusion["cadence"], conclusion["horizon"])
        ae = lookup[(*base, "absolute_error")]["primary"]["test"]["meanLossDifference"]
        ape = lookup[(*base, "absolute_percentage_error")]["primary"]["test"]["meanLossDifference"]
        lines.append(f"| {base[0]} | {base[1]} | {base[2]} | {base[3]} | {conclusion['status']} | "
                     f"{ae:.6g} | {ape:.6g} | {conclusion['sensitivityStatus']} |")
    if not any(row["status"] in ("worsened", "mixed_evidence") for row in conclusions):
        lines.append("| — | — | — | — | — | — | — | — |")
    lines.extend(["", "Full paired metrics and both adjusted 800-test families are in the JSON report.", ""])
    return "\n".join(lines)


def _atomic_write(path: str | Path, payload: bytes, overwrite: bool = False) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent))
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temp_name, output)
        else:
            os.link(temp_name, output)
            os.unlink(temp_name)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Paired significance inference for retrospective trained-model predictions.")
    parser.add_argument("--input", default="outputs/dashboard-expanded-20260929.json")
    parser.add_argument("--output", default="outputs/significance-expanded-20260929.json")
    parser.add_argument("--markdown", default="outputs/significance-expanded-20260929.md")
    parser.add_argument("--overwrite", action="store_true", help="replace existing report files atomically")
    args = parser.parse_args(argv)
    try:
        input_path = Path(args.input).resolve()
        json_path = Path(args.output).resolve()
        markdown_path = Path(args.markdown).resolve() if args.markdown else None
        if json_path == input_path or markdown_path == input_path or (markdown_path and markdown_path == json_path):
            raise ValueError("input, JSON output, and Markdown output paths must all differ")
        raw = input_path.read_bytes()
        source = json.loads(raw.decode("utf-8"))
        report = analyze(source, hashlib.sha256(raw).hexdigest())
        json_bytes = (json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")
        md_bytes = markdown_report(report).encode("utf-8") if markdown_path else None
        if not args.overwrite and (json_path.exists() or (markdown_path and markdown_path.exists())):
            raise FileExistsError("refusing to overwrite an existing report; choose a new path or pass --overwrite")
        _atomic_write(json_path, json_bytes, args.overwrite)
        if markdown_path and md_bytes is not None:
            try:
                _atomic_write(markdown_path, md_bytes, args.overwrite)
            except Exception:
                # Avoid leaving a JSON result that looks like a paired output when its Markdown failed.
                if not args.overwrite:
                    try:
                        json_path.unlink()
                    except OSError:
                        pass
                raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"significance analysis failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
