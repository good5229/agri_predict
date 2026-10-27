"""Frozen retrospective onion/peeled-garlic ensemble and causal bias-correction study."""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any

from . import dashboard, direct_horizon as dh

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/daily/2026-10-06/20261006T090116-f908e0a4/predictions.json"
INPUT_SHA256 = "5b4d7cbb362a6fa257fb6e6d37b23e5611c3487c8ddc540ecbdbf3f2665c4127"
PRODUCTS = ("onion", "garlic")
HORIZONS = tuple(range(1, 11))
SELECTION_START, SELECTION_END = dt.date(2023, 10, 6), dt.date(2025, 9, 12)
TEST_START, TEST_END = dt.date(2025, 10, 6), dt.date(2026, 10, 5)
RESIDUAL_WINDOW_DAYS = 90
RESIDUAL_LAG_DAYS = 14
MIN_RESIDUAL_PAIRS = 15
BIAS_SCALE = 0.5
MAX_LOG_BIAS = 0.08
MIN_SELECTION_IMPROVEMENT_PP = 0.1
MAX_SELECTION_MAE_REGRESSION = 0.01
BANDWIDTHS = (9, 28)
MIN_DM_PAIRS = 30
ALPHA = 0.05
COMPARATORS = ("ridge", "persistence")
LOSSES = ("mae", "mape")
CANDIDATES = ("A", "B", "C", "D")
FAMILY_SIZE = len(PRODUCTS) * len(HORIZONS) * len(COMPARATORS) * len(LOSSES)
FIRST_TEST_ORIGIN = TEST_START-dt.timedelta(days=max(HORIZONS))
if SELECTION_END > FIRST_TEST_ORIGIN-dt.timedelta(days=RESIDUAL_LAG_DAYS):
    raise AssertionError("selection end leaks outcomes unavailable at the earliest test origin")


def _day(value: Any, label: str) -> dt.date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        result = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc
    if result.isoformat() != value:
        raise ValueError(f"{label} must be canonical YYYY-MM-DD")
    return result


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _parse_snapshot(raw: dict[str, Any]):
    if not isinstance(raw, dict) or raw.get("schemaVersion") != 2 or not isinstance(raw.get("products"), list):
        raise ValueError("predictions must use dashboard schemaVersion 2")
    if raw.get("meta", {}).get("contextEnd") != "2026-10-05":
        raise ValueError("snapshot contextEnd differs from frozen protocol")
    indexed = {}
    for item in raw["products"]:
        pid = item.get("id")
        if pid in indexed or pid not in dashboard.PRODUCTS:
            raise ValueError("unknown or duplicate product in snapshot")
        indexed[pid] = item
    if not set(PRODUCTS).issubset(indexed):
        raise ValueError("snapshot lacks onion or peeled garlic")

    data = {}
    for product in PRODUCTS:
        item = indexed[product]
        actuals, excluded_actual = {}, {}
        observations = item.get("observations")
        if not isinstance(observations, list):
            raise ValueError(f"missing observations for {product}")
        for row in observations:
            target = _day(row.get("date"), f"{product} observation date")
            if target > TEST_END:
                raise ValueError("observation exceeds frozen context cutoff")
            if target in actuals or target in excluded_actual:
                raise ValueError(f"duplicate observation {product}/{target}")
            value = _positive(row.get("actual"))
            if value is None:
                excluded_actual[target] = "null_or_nonpositive_or_invalid_actual"
            else:
                actuals[target] = value

        forecasts = {}
        invalid = {}
        runs = item.get("runs")
        if not isinstance(runs, list):
            raise ValueError(f"missing forecast runs for {product}")
        for model in ("ridge", "hist_gb"):
            selected = [r for r in runs if r.get("modelId") == model and r.get("cadence") == "daily"]
            if len(selected) != 1 or not isinstance(selected[0].get("predictions"), list):
                raise ValueError(f"expected exactly one daily {model} run for {product}")
            forecasts[model] = {}
            invalid[model] = {}
            for row in selected[0]["predictions"]:
                target = _day(row.get("target"), f"{product} {model} target")
                if target > TEST_END:
                    raise ValueError(f"{model} forecast exceeds frozen target cutoff")
                origin = _day(row.get("origin"), f"{product} {model} origin")
                horizon = row.get("horizon")
                if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
                    raise ValueError(f"invalid {model} horizon for {product}/{target}")
                if origin != target-dt.timedelta(days=horizon):
                    raise ValueError(f"{model} origin/horizon mismatch for {product}/{target}/h{horizon}")
                key = (target, horizon)
                if key in forecasts[model] or key in invalid[model]:
                    raise ValueError(f"duplicate {model} forecast {product}/{target}/h{horizon}")
                value = _positive(row.get("predicted"))
                if value is None:
                    invalid[model][key] = "null_nonpositive_or_nonfinite_prediction"
                else:
                    forecasts[model][key] = {"value": value, "origin": origin}

        persistence, invalid_persistence = {}, {}
        if not isinstance(item.get("predictions"), list):
            raise ValueError(f"missing persistence predictions for {product}")
        for row in item["predictions"]:
            target = _day(row.get("target"), f"{product} persistence target")
            if target > TEST_END:
                raise ValueError("persistence forecast exceeds frozen target cutoff")
            origin = _day(row.get("origin"), f"{product} persistence origin")
            horizon = row.get("horizon")
            if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
                raise ValueError(f"invalid persistence horizon for {product}/{target}")
            if origin != target-dt.timedelta(days=horizon):
                raise ValueError(f"persistence origin/horizon mismatch for {product}/{target}/h{horizon}")
            key = (target, horizon)
            if key in persistence or key in invalid_persistence:
                raise ValueError(f"duplicate persistence forecast {product}/{target}/h{horizon}")
            value = _positive(row.get("predicted"))
            if value is None:
                invalid_persistence[key] = "null_nonpositive_or_nonfinite_prediction"
            else:
                persistence[key] = {"value": value, "origin": origin}
        data[product] = {"actuals": actuals, "excludedActuals": excluded_actual, "forecasts": forecasts,
                         "actualDays": sorted(actuals), "excludedActualDays": sorted(excluded_actual),
                         "invalidForecasts": invalid, "persistence": persistence,
                         "invalidPersistence": invalid_persistence}
    return data


def _candidate_values(ridge: float, hist: float, residuals: list[float], invalid_history_count: int):
    a = .5*ridge + .5*hist
    b = .75*ridge + .25*hist
    result = {"A": (a, {"applied": False, "reason": "not_applicable", "n": 0, "medianLogResidual": None, "scaledLogCorrection": 0.0}),
              "B": (b, {"applied": False, "reason": "not_applicable", "n": 0, "medianLogResidual": None, "scaledLogCorrection": 0.0})}
    for label, base in (("C", ridge), ("D", a)):
        if invalid_history_count:
            result[label] = (base, {"applied": False, "reason": "invalid_residual_input", "n": len(residuals),
                                    "invalidHistoryCount": invalid_history_count, "medianLogResidual": None, "scaledLogCorrection": 0.0})
        elif len(residuals) < MIN_RESIDUAL_PAIRS:
            result[label] = (base, {"applied": False, "reason": "insufficient_residual_pairs", "n": len(residuals),
                                    "invalidHistoryCount": 0, "medianLogResidual": None, "scaledLogCorrection": 0.0})
        else:
            median = statistics.median(residuals)
            capped_median = max(-MAX_LOG_BIAS, min(MAX_LOG_BIAS, median))
            scaled = BIAS_SCALE*capped_median
            corrected = base*math.exp(scaled)
            if not math.isfinite(corrected) or corrected <= 0:
                result[label] = (base, {"applied": False, "reason": "nonfinite_corrected_prediction", "n": len(residuals),
                    "invalidHistoryCount": 0, "medianLogResidual": median, "scaledLogCorrection": 0.0})
            else:
                result[label] = (corrected, {"applied": True, "reason": None, "n": len(residuals),
                    "invalidHistoryCount": 0, "medianLogResidual": median, "scaledLogCorrection": scaled})
    return result


def _residual_history(source, horizon, window_start, cutoff, model):
    residuals, targets, invalid_targets = [], [], []
    actuals = source["actuals"]
    days = source["actualDays"]
    lo, hi = bisect_left(days, window_start), bisect_right(days, cutoff)
    for target in days[lo:hi]:
        actual = actuals[target]
        key = (target, horizon)
        ridge = source["forecasts"]["ridge"].get(key)
        hist = source["forecasts"]["hist_gb"].get(key) if model == "A" else None
        if ridge is None or (model == "A" and hist is None):
            if (key in source["invalidForecasts"]["ridge"] or
                    (model == "A" and key in source["invalidForecasts"]["hist_gb"])):
                invalid_targets.append(target)
            continue
        if target <= ridge["origin"] or (hist is not None and target <= hist["origin"]):
            invalid_targets.append(target)
            continue
        prediction = ridge["value"] if model == "ridge" else .5*ridge["value"] + .5*hist["value"]
        residuals.append(math.log(actual)-math.log(prediction))
        targets.append(target)
    invalid_days = source["excludedActualDays"]
    lo, hi = bisect_left(invalid_days, window_start), bisect_right(invalid_days, cutoff)
    for target in invalid_days[lo:hi]:
        key = (target, horizon)
        has_forecast = key in source["forecasts"]["ridge"] or key in source["invalidForecasts"]["ridge"]
        if model == "A":
            has_forecast = has_forecast or key in source["forecasts"]["hist_gb"] or key in source["invalidForecasts"]["hist_gb"]
        if has_forecast:
            invalid_targets.append(target)
    return residuals, targets, invalid_targets


def _build_rows(data: dict[str, Any], start: dt.date, end: dt.date):
    rows, excluded, fallback_counts = [], {}, {}
    for product in PRODUCTS:
        source = data[product]
        for target in sorted(d for d in source["actuals"] if start <= d <= end):
            actual = source["actuals"][target]
            for horizon in HORIZONS:
                origin = target-dt.timedelta(days=horizon)
                key = (target, horizon)
                ridge = source["forecasts"]["ridge"].get(key)
                hist = source["forecasts"]["hist_gb"].get(key)
                persist = source["persistence"].get(key)
                if not ridge or not hist or not persist:
                    missing = []
                    if ridge is None: missing.append("ridge_" + ("invalid" if key in source["invalidForecasts"]["ridge"] else "missing"))
                    if hist is None: missing.append("histGb_" + ("invalid" if key in source["invalidForecasts"]["hist_gb"] else "missing"))
                    if persist is None: missing.append("persistence_" + ("invalid" if key in source["invalidPersistence"] else "missing"))
                    bucket = excluded.setdefault(f"{product}/{horizon}", {})
                    for reason in missing:
                        bucket[reason] = bucket.get(reason, 0)+1
                    continue
                if any(x["origin"] != origin for x in (ridge,hist,persist)):
                    raise ValueError(f"forecast origin mismatch for {product}/{target}/h{horizon}")
                values = {"ridge": ridge["value"], "hist_gb": hist["value"], "persistence": persist["value"]}
                base_a = .5*values["ridge"] + .5*values["hist_gb"]
                cutoff = origin-dt.timedelta(days=RESIDUAL_LAG_DAYS)
                window_start = cutoff-dt.timedelta(days=RESIDUAL_WINDOW_DAYS-1)
                correction_meta, corrected = {}, {}
                for label, model in (("C", "ridge"), ("D", "A")):
                    residuals, residual_targets, invalid_targets = _residual_history(source,horizon,window_start,cutoff,model)
                    predicted, detail = _candidate_values(values["ridge"],values["hist_gb"],residuals,len(invalid_targets))[label]
                    detail.update({"residualWindowStart": window_start.isoformat(), "residualCutoff": cutoff.isoformat(),
                        "residualWindowDays": RESIDUAL_WINDOW_DAYS, "lagDays": RESIDUAL_LAG_DAYS,
                        "residualTargetCount": len(residual_targets),
                        "residualTargetDatesSha256": hashlib.sha256("\n".join(d.isoformat() for d in sorted(residual_targets)).encode()).hexdigest(),
                        "residualTargetStart": min(residual_targets).isoformat() if residual_targets else None,
                        "residualTargetEnd": max(residual_targets).isoformat() if residual_targets else None,
                        "invalidResidualTargets": [d.isoformat() for d in sorted(invalid_targets)]})
                    corrected[label], correction_meta[label] = predicted, detail
                    fallback_counts[f"{product}/{label}"] = fallback_counts.get(f"{product}/{label}",0)+int(not detail["applied"])
                row = {"product": product, "target": target.isoformat(), "origin": origin.isoformat(), "horizon": horizon,
                    "actual": actual, "ridge": values["ridge"], "histGb": values["hist_gb"], "persistence": values["persistence"],
                    "candidateA": base_a, "candidateB": .75*values["ridge"]+.25*values["hist_gb"],
                    "candidateC": corrected["C"], "candidateD": corrected["D"], "correction": correction_meta,
                    "sourceForecastOrigins": {k:v["origin"].isoformat() for k,v in (("ridge",ridge),("hist_gb",hist),("persistence",persist))}}
                rows.append(row)
    return rows, excluded, fallback_counts

def _macro(rows: list[dict[str, Any]], method: str, product: str):
    by_h = {}
    for horizon in HORIZONS:
        group = [r for r in rows if r["product"] == product and r["horizon"] == horizon]
        by_h[str(horizon)] = {"n": len(group), **dh._metrics([r["actual"] for r in group], [r[method] for r in group])}
    mape_values = [x["mape"] for x in by_h.values()]
    mae_values = [x["mae"] for x in by_h.values()]
    return {"mapeByHorizon": by_h, "equal10HorizonMape": sum(mape_values)/10 if all(x is not None for x in mape_values) else None,
            "equal10HorizonMae": sum(mae_values)/10 if all(x is not None for x in mae_values) else None,
            "pointEstimateAvailable": all(x is not None for x in mape_values)}


def _select(selection_rows: list[dict[str, Any]]):
    decisions = {}
    for product in PRODUCTS:
        metrics = {name: _macro(selection_rows, name, product) for name in ("ridge", *[f"candidate{x}" for x in CANDIDATES])}
        ridge = metrics["ridge"]
        eligible = []
        for order, label in enumerate(CANDIDATES):
            candidate = metrics[f"candidate{label}"]
            if ridge["equal10HorizonMape"] is None or ridge["equal10HorizonMae"] is None or not candidate["pointEstimateAvailable"]:
                continue
            if (candidate["equal10HorizonMape"] <= ridge["equal10HorizonMape"]-MIN_SELECTION_IMPROVEMENT_PP and
                candidate["equal10HorizonMae"] <= ridge["equal10HorizonMae"]*(1+MAX_SELECTION_MAE_REGRESSION)):
                eligible.append((candidate["equal10HorizonMape"], candidate["equal10HorizonMae"], order, label))
        choice = min(eligible)[3] if eligible else "ridge"
        decisions[product] = {"selected": choice, "selectionRule": {
            "minimumMapeImprovementPercentagePoints": MIN_SELECTION_IMPROVEMENT_PP,
            "maximumMaeRegressionFraction": MAX_SELECTION_MAE_REGRESSION,
            "tieBreak": "MAPE, then MAE, then A/B/C/D"},
            "metrics": metrics, "eligibleCandidates": [v[3] for v in eligible]}
    return decisions


def _conclusion(comparisons: list[dict[str, Any]], coverage: dict[str, Any], product: str, horizon: int):
    rows = [r for r in comparisons if r["product"] == product and r["horizon"] == horizon]
    tests = [r["dm"]["9"] for r in rows]
    enough = all(t["status"] == "tested" and t["n"] >= MIN_DM_PAIRS for t in tests)
    lower = all(r["candidateMetric"] is not None and r["comparatorMetric"] is not None and r["candidateMetric"] < r["comparatorMetric"] for r in rows)
    significant = all(t.get("holmSignificant") and t.get("meanLossDifference") is not None and t["meanLossDifference"] < 0 for t in tests)
    cov_ok = coverage[f"{product}/{horizon}"]["coverageNondeclining"]
    adverse = any(t.get("holmSignificant") and t.get("meanLossDifference") is not None and t["meanLossDifference"] > 0 for t in tests)
    status = "inference_unavailable" if not enough else "improvement" if lower and significant and cov_ok else "worsening_or_mixed" if adverse else "inconclusive"
    return {"product": product, "horizon": horizon, "conclusion": status, "inferenceAvailable": enough,
            "allFourHac9HolmTestsImprove": significant, "allFourPairedMetricsImprove": lower,
            "coverageNondeclining": cov_ok, "significantWorseningEvidence": adverse}


def _coverage_audit(data, rows, start, end):
    result = {}
    for product in PRODUCTS:
        source = data[product]
        actual_dates = {d for d in source["actuals"] if start <= d <= end}
        for horizon in HORIZONS:
            ridge = {d for d,h in source["forecasts"]["ridge"] if h == horizon and d in actual_dates}
            hist = {d for d,h in source["forecasts"]["hist_gb"] if h == horizon and d in actual_dates}
            persistence = {d for d,h in source["persistence"] if h == horizon and d in actual_dates}
            candidate = ridge & hist
            common = candidate & persistence
            paired = {dt.date.fromisoformat(r["target"]) for r in rows if r["product"] == product and r["horizon"] == horizon}
            if paired != common:
                raise AssertionError(f"paired rows differ from common forecast coverage for {product}/h{horizon}")
            losses = {"ridge": sorted(d.isoformat() for d in ridge-candidate),
                      "persistence": sorted(d.isoformat() for d in persistence-candidate)}
            result[f"{product}/{horizon}"] = {"eligiblePositiveActualCount": len(actual_dates),
                "candidateCoverage": len(candidate), "ridgeCoverage": len(ridge), "histGbCoverage": len(hist),
                "persistenceCoverage": len(persistence), "commonPairedCount": len(common),
                "eligibleMinusPaired": len(actual_dates)-len(common),
                "candidateCoverageLossDates": losses, "candidateCoverageLossCounts": {k:len(v) for k,v in losses.items()},
                "coverageNondeclining": all(not v for v in losses.values()),
                "pairedDates": sorted(d.isoformat() for d in common)}
    return result


def _test_assessment(rows: list[dict[str, Any]], selections: dict[str, Any], coverage):
    paired, comparisons = [], []
    for product in PRODUCTS:
        selected_label = selections[product]["selected"]
        candidate_field = "ridge" if selected_label == "ridge" else f"candidate{selected_label}"
        for horizon in HORIZONS:
            group = [r for r in rows if r["product"] == product and r["horizon"] == horizon]
            paired.extend([{**r, "selectedMethod": selected_label, "selectedPrediction": r[candidate_field]} for r in group])
            for comparator in COMPARATORS:
                for loss in LOSSES:
                    actual = [r["actual"] for r in group]
                    cand = [r[candidate_field] for r in group]
                    comp = [r[comparator] for r in group]
                    diffs = {dt.date.fromisoformat(r["target"]): (abs(r["actual"]-r[candidate_field])-abs(r["actual"]-r[comparator])) if loss == "mae" else 100*(abs(r["actual"]-r[candidate_field])/r["actual"]-abs(r["actual"]-r[comparator])/r["actual"]) for r in group}
                    comparisons.append({"product": product, "horizon": horizon, "comparator": comparator, "loss": loss, "n": len(group),
                        "candidateMetric": dh._metrics(actual,cand)[loss], "comparatorMetric": dh._metrics(actual,comp)[loss],
                        "dm": {str(bw): dh._dm(diffs,bw) for bw in BANDWIDTHS}})
    if len(comparisons) != FAMILY_SIZE:
        raise AssertionError("frozen test Holm family must contain 80 tests")
    for bw in BANDWIDTHS:
        dh._holm([r["dm"][str(bw)] for r in comparisons])
    macro = {p: {method: _macro(paired, method, p) for method in ("selectedPrediction", "ridge", "persistence")} for p in PRODUCTS}
    strata = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            group = [r for r in paired if r["product"] == product and r["horizon"] == horizon]
            def season(day):
                if day.month in (12,1,2): return "DJF"
                if day.month in (3,4,5): return "MAM"
                if day.month in (6,7,8): return "JJA"
                return "SON"
            for label, fn in (("year", lambda d: str(d.year)), ("season", season)):
                groups = {}
                for row in group:
                    date = dt.date.fromisoformat(row["target"]); groups.setdefault(fn(date), []).append(row)
                for value, subset in sorted(groups.items()):
                    strata.append({"product": product, "horizon": horizon, "stratum": label, "value": value, "n": len(subset),
                        **{method: dh._metrics([r["actual"] for r in subset],[r[method] for r in subset]) for method in ("selectedPrediction","ridge","persistence")}})
    conclusions = [_conclusion(comparisons,coverage,p,h) for p in PRODUCTS for h in HORIZONS]
    return {"pairedRows": paired, "comparisons": comparisons, "coverage": coverage,
            "macroMetrics": macro, "yearSeasonMetrics": strata, "conclusions": conclusions}


def build(raw: dict[str, Any], *, hashes: dict[str, str] | None = None):
    data = _parse_snapshot(raw)
    data_quality = {p: {"excludedActualCount": len(data[p]["excludedActuals"]),
        "invalidForecastCounts": {model: len(values) for model,values in data[p]["invalidForecasts"].items()},
        "invalidPersistenceCount": len(data[p]["invalidPersistence"])} for p in PRODUCTS}
    selection_rows, selection_exclusions, selection_fallbacks = _build_rows(data, SELECTION_START, SELECTION_END)
    selection = _select(selection_rows)
    selection_coverage = _coverage_audit(data,selection_rows,SELECTION_START,SELECTION_END)
    test_rows, test_exclusions, _ = _build_rows(data, TEST_START, TEST_END)
    test_coverage = _coverage_audit(data,test_rows,TEST_START,TEST_END)
    test = _test_assessment(test_rows, selection, test_coverage)
    return {"schemaVersion": 1, "protocol": {"id": "onion-garlic-7pct-causal-blend-2026-10-07",
        "inputSha256": INPUT_SHA256, "products": list(PRODUCTS), "selectionTargetInterval": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
        "testTargetInterval": [TEST_START.isoformat(), TEST_END.isoformat()], "horizons": list(HORIZONS),
        "earliestTestOrigin": FIRST_TEST_ORIGIN.isoformat(), "selectionCausalEmbargoDays": RESIDUAL_LAG_DAYS,
        "residualWindowDays": RESIDUAL_WINDOW_DAYS, "residualCutoff": "origin-14 days inclusive; 90-date window ending at cutoff",
        "minimumResidualPairs": MIN_RESIDUAL_PAIRS, "biasScale": BIAS_SCALE,
        "maxAbsoluteMedianLogResidual": MAX_LOG_BIAS, "maxAbsoluteAppliedLogCorrection": BIAS_SCALE*MAX_LOG_BIAS,
        "candidates": {"A":"0.5 Ridge + 0.5 HistGB", "B":"0.75 Ridge + 0.25 HistGB",
                       "C":"Ridge * exp(0.5 * clipped median log(actual/Ridge))", "D":"A * exp(0.5 * clipped median log(actual/A))"},
        "selection": {"mapeImprovementPercentagePoints": MIN_SELECTION_IMPROVEMENT_PP,
                      "maximumMaeRegressionFraction": MAX_SELECTION_MAE_REGRESSION, "fallback":"Ridge"},
        "testDM": {"familySize": FAMILY_SIZE, "twoSided":True, "bartlettCalendarHACDays":list(BANDWIDTHS),
                   "holmAlpha":ALPHA, "minimumPairs":MIN_DM_PAIRS, "unavailablePValue":1.0},
        "limitations":"Retrospective revised two-market prices; not prospective or official contest-score evidence."},
        "hashes": hashes or {}, "dataQuality": data_quality, "selection": selection,
        "selectionAudit": {"pairedRows": selection_rows, "coverage": selection_coverage, "exclusionCounts": selection_exclusions,
                           "fallbackCounts": selection_fallbacks},
        "test": {**test, "exclusionCounts": test_exclusions,
                 "fallbackCounts": {f"{p}/{label}": sum(1 for r in test["pairedRows"] if r["product"]==p and not r["correction"][label]["applied"]) for p in PRODUCTS for label in ("C","D")},
                 "targetGoalUnder7Percent": {p: (test["macroMetrics"][p]["selectedPrediction"]["equal10HorizonMape"] < 7.0
                     if test["macroMetrics"][p]["selectedPrediction"]["equal10HorizonMape"] is not None else None) for p in PRODUCTS}}}


def _write_immutable(path: Path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            fd = None
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.link(tmp, path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite immutable output: {path}") from exc
    finally:
        if fd is not None: os.close(fd)
        try: os.unlink(tmp)
        except FileNotFoundError: pass


def run(input_path: Path, output_path: Path):
    if Path(output_path).exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {output_path}")
    data = Path(input_path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != INPUT_SHA256:
        raise ValueError("input SHA-256 differs from frozen protocol")
    raw = json.loads(data)
    result = build(raw, hashes={"predictionSnapshotSha256": digest})
    encoded = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n").encode()
    _write_immutable(Path(output_path), encoded)
    return {"output": str(output_path), "sha256": hashlib.sha256(encoded).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.input,args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
