"""Run the frozen radish low-volume retrospective comparison on a saved dashboard JSON."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
from bisect import bisect_right
from pathlib import Path
from typing import Any

from . import modeling
from . import volume_guard as guard

METHODS = ("baseline", "ridge_daily")
BANDS = ("overall", "event_target", "event_source_followup", "event_input_followup", "from_2024")
LOSSES = ("absoluteError", "absolutePercentageError")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_run(product: dict[str, Any], model_id: str) -> dict[tuple[dt.date, dt.date, int], dict[str, Any]]:
    if model_id == "baseline":
        rows = product.get("predictions", [])
    else:
        runs = [run for run in product.get("runs", [])
                if run.get("modelId") == "ridge" and run.get("cadence") == "daily"]
        if len(runs) != 1:
            raise ValueError("dashboard must contain exactly one daily Ridge run for radish")
        rows = runs[0].get("predictions", [])
    result = {}
    for row in rows:
        origin = modeling._parse_day(row.get("origin"))
        target = modeling._parse_day(row.get("target"))
        horizon = row.get("horizon")
        predicted = modeling._finite_positive(row.get("predicted"))
        source_date = modeling._parse_day(row.get("sourceDate"))
        if origin is None or target is None or not isinstance(horizon, int) or predicted is None:
            continue
        if target - origin != dt.timedelta(days=horizon):
            raise ValueError("saved forecast target/origin/horizon mismatch")
        result[(origin, target, horizon)] = {**row, "sourceDateParsed": source_date}
    return result


def _event_input_exposure(series: dict[dt.date, tuple[float, float]],
                          origin: dt.date, method: str,
                          days: tuple[dt.date, ...] | None = None) -> dict[str, Any]:
    days = days if days is not None else tuple(series)
    hi = bisect_right(days, origin)
    if hi == 0:
        return {"latestSource": False, "lags": [], "windows": [], "any": False}
    latest = days[hi - 1]
    exposed_lags = []
    if method == "ridge_daily":
        for lag in modeling.LAGS:
            cutoff = origin - dt.timedelta(days=lag)
            lag_idx = bisect_right(days, cutoff, 0, hi) - 1
            if lag_idx >= 0 and days[lag_idx] == guard.EVENT_DAY:
                exposed_lags.append(lag)
    exposed_windows = []
    if method == "ridge_daily":
        for window in modeling.WINDOWS:
            first_day = origin - dt.timedelta(days=window - 1)
            if first_day <= guard.EVENT_DAY <= origin and guard.EVENT_DAY in series:
                exposed_windows.append(window)
    latest_uses_event = latest == guard.EVENT_DAY
    return {"latestSource": latest_uses_event, "lags": exposed_lags, "windows": exposed_windows,
            "any": latest_uses_event or bool(exposed_lags) or bool(exposed_windows)}


def _bands(key: tuple[dt.date, dt.date, int], original_source: dt.date | None,
           event_exposure: dict[str, Any]) -> set[str]:
    _, target, _ = key
    result = {"overall"}
    if target == guard.EVENT_DAY:
        result.add("event_target")
    if target > guard.EVENT_DAY and original_source == guard.EVENT_DAY:
        result.add("event_source_followup")
    if target > guard.EVENT_DAY and event_exposure["any"]:
        result.add("event_input_followup")
    if target >= dt.date(2024, 1, 1):
        result.add("from_2024")
    return result


def _paired_rows(original: dict[tuple[dt.date, dt.date, int], dict[str, Any]],
                 candidate: dict[tuple[dt.date, dt.date, int], dict[str, Any]],
                 actuals: dict[dt.date, float], method: str,
                 raw_series: dict[dt.date, tuple[float, float]]) -> list[dict[str, Any]]:
    result = []
    days = tuple(raw_series)
    exposure_cache: dict[tuple[str, dt.date], dict[str, Any]] = {}
    for key in sorted(set(original) & set(candidate)):
        origin, target, horizon = key
        if not guard.EVALUATION_START <= target <= guard.EVALUATION_END:
            continue
        if horizon not in guard.HORIZONS:
            continue
        actual = actuals.get(target)
        if actual is None or actual <= 0:
            continue
        old, new = original[key], candidate[key]
        cache_key = (method, origin)
        if cache_key not in exposure_cache:
            exposure_cache[cache_key] = _event_input_exposure(raw_series, origin, method, days)
        exposure = exposure_cache[cache_key]
        old_loss = guard.loss_metrics(actual, float(old["predicted"]))
        new_loss = guard.loss_metrics(actual, float(new["predicted"]))
        result.append({"method": method, "origin": origin.isoformat(), "target": target.isoformat(),
                       "horizon": horizon, "actual": actual,
                       "originalPredicted": float(old["predicted"]),
                       "candidatePredicted": float(new["predicted"]),
                       "originalSourceDate": old.get("sourceDateParsed").isoformat()
                           if old.get("sourceDateParsed") else old.get("sourceDate"),
                       "candidateSourceDate": candidate[key].get("sourceDate"),
                       "originalEventInputExposure": exposure,
                       "originalLoss": old_loss, "candidateLoss": new_loss,
                       "lossDifference": {name: new_loss[name] - old_loss[name] for name in LOSSES},
                       "bands": sorted(_bands(key, old.get("sourceDateParsed"), exposure))})
    return result


def _eligible_keys(forecasts: dict[tuple[dt.date, dt.date, int], dict[str, Any]],
                   actuals: dict[dt.date, float]) -> set[tuple[dt.date, dt.date, int]]:
    return {key for key in forecasts
            if guard.EVALUATION_START <= key[1] <= guard.EVALUATION_END
            and key[2] in guard.HORIZONS and key[1] in actuals}


def _report_stats(pairs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tables = []
    tests_by_bandwidth: dict[int, list[dict[str, Any]]] = {9: [], 28: []}
    for method in METHODS:
        for horizon in guard.HORIZONS:
            for band in BANDS:
                selected = [row for row in pairs if row["method"] == method and row["horizon"] == horizon
                            and band in row["bands"]]
                old_metrics = guard.aggregate_metrics([row["originalLoss"] for row in selected])
                new_metrics = guard.aggregate_metrics([row["candidateLoss"] for row in selected])
                tables.append({"method": method, "horizon": horizon, "band": band,
                               "original": old_metrics, "candidate": new_metrics,
                               "pairedCoverage": len(selected)})
            overall = [row for row in pairs if row["method"] == method and row["horizon"] == horizon
                       and "overall" in row["bands"]]
            for loss in LOSSES:
                differences = [(dt.date.fromisoformat(row["target"]), row["lossDifference"][loss])
                               for row in overall]
                for bandwidth in (9, 28):
                    test = guard.dm_calendar_hac(differences, bandwidth)
                    test.update({"method": method, "horizon": horizon, "loss": loss})
                    tests_by_bandwidth[bandwidth].append(test)
    tests = []
    for bandwidth, family in tests_by_bandwidth.items():
        guard.holm_adjust(family)
        for test in family:
            if test["status"] != "tested":
                test["holmStatus"] = "inference_unavailable_retained_in_family"
            elif test["holmPValue"] < 0.05:
                test["holmStatus"] = "significant_at_5_percent"
            else:
                test["holmStatus"] = "not_significant"
        tests.extend(family)
    return tables, tests


def _dual_loss_conclusions(tests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conclusions = []
    for bandwidth in (9, 28):
        for method in METHODS:
            for horizon in guard.HORIZONS:
                rows = [row for row in tests if row["bandwidthCalendarDays"] == bandwidth
                        and row["method"] == method and row["horizon"] == horizon]
                rows.sort(key=lambda row: row["loss"])
                unavailable = [row for row in rows if row["status"] != "tested"]
                means = [row.get("meanLossDifference") for row in rows]
                significant = [row.get("holmPValue", 1.0) < 0.05 for row in rows]
                directions = {"lower" if value < 0 else "higher" if value > 0 else "equal"
                              for value in means if value is not None}
                significant_directions = {"lower" if value < 0 else "higher"
                                          for value, sig in zip(means, significant)
                                          if value is not None and sig and value != 0}
                if unavailable:
                    verdict = "inference_unavailable"
                elif len(directions - {"equal"}) > 1 or len(significant_directions) > 1:
                    verdict = "mixed"
                elif all(value is not None and value < 0 for value in means) and all(significant):
                    verdict = "improved"
                elif all(value is not None and value > 0 for value in means) and all(significant):
                    verdict = "worsened"
                else:
                    verdict = "no_evidence"
                conclusions.append({"bandwidthCalendarDays": bandwidth, "method": method,
                                    "horizon": horizon, "verdict": verdict,
                                    "rule": "requires both MAE and MAPE mean differences in the same direction and both Holm p<0.05 for improved/worsened",
                                    "lossTests": rows})
    return conclusions


def run_experiment(input_path: str | Path,
                  price_ratio_threshold: float | None = None) -> dict[str, Any]:
    if price_ratio_threshold is not None and price_ratio_threshold < 1.0:
        raise ValueError("price ratio threshold must be at least 1")
    path = Path(input_path).resolve()
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("schemaVersion") not in (1, 2) or not isinstance(source.get("products"), list):
        raise ValueError("input must be a saved dashboard JSON with products")
    products = [product for product in source["products"] if product.get("id") == "radish"]
    if len(products) != 1:
        raise ValueError("input must contain exactly one radish product")
    product = products[0]
    raw_series = guard._observations(product)
    quarantined = guard.quarantine_dates(raw_series, price_ratio_threshold)
    filtered_series = guard.quality_series(raw_series, quarantined)
    if not raw_series:
        raise ValueError("radish observations are empty")
    actuals = {day: price for day, (price, _) in raw_series.items()}
    original = {method: _read_run(product, "baseline" if method == "baseline" else "ridge")
                for method in METHODS}
    origins = []
    day = guard.EVALUATION_START - dt.timedelta(days=max(modeling.HORIZONS))
    final_origin = guard.EVALUATION_END - dt.timedelta(days=1)
    while day <= final_origin:
        origins.append(day)
        day += dt.timedelta(days=1)
    baseline_candidate_list = guard.persistence_predictions(filtered_series, origins, guard.HORIZONS)
    candidate_baseline = {(dt.date.fromisoformat(row["origin"]), dt.date.fromisoformat(row["target"]),
                           row["horizon"]): row for row in baseline_candidate_list}

    # Refit just daily Ridge on the quality-filtered series. Labels remain raw only for
    # evaluation; the sample pool is built from filtered observations and thus excludes
    # quarantined labels as well as quarantined price/volume features.
    train_samples, _ = modeling._supervised_samples(filtered_series, guard.EVALUATION_END, {})
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("numpy is required for the Ridge experiment") from exc
    matrix = np.asarray([sample[3] for sample in train_samples], dtype=float)
    labels = np.asarray([sample[4] for sample in train_samples], dtype=float)
    ridge_run, _ = modeling._fit_cadence(
        filtered_series, "ridge", "daily", origins[0], final_origin, actuals,
        train_samples, {}, matrix, labels,
        evaluation_start=guard.EVALUATION_START, evaluation_end=guard.EVALUATION_END)
    candidate_ridge = {}
    for row in ridge_run.get("predictions", []):
        key = (dt.date.fromisoformat(row["origin"]), dt.date.fromisoformat(row["target"]), row["horizon"])
        candidate_ridge[key] = row
    candidate = {"baseline": candidate_baseline, "ridge_daily": candidate_ridge}

    paired = []
    coverage = []
    for method in METHODS:
        method_pairs = _paired_rows(original[method], candidate[method], actuals, method, raw_series)
        paired.extend(method_pairs)
        expected = _eligible_keys(original[method], actuals)
        candidate_expected = _eligible_keys(candidate[method], actuals)
        matched = { (dt.date.fromisoformat(row["origin"]), dt.date.fromisoformat(row["target"]), row["horizon"])
                    for row in method_pairs }
        missing = expected - candidate_expected
        candidate_only = candidate_expected - expected
        key_rows = lambda keys: [{"origin": key[0].isoformat(), "target": key[1].isoformat(),
                                  "horizon": key[2]} for key in sorted(keys)]
        coverage.append({"method": method, "eligibleOriginalForecasts": len(expected),
                         "candidateForecasts": len(candidate_expected),
                         "pairedForecasts": len(method_pairs),
                         "originalForecastsWithoutCandidateMatch": len(missing),
                         "originalForecastsWithoutCandidateMatchKeys": key_rows(missing),
                         "originalForecastsWithoutCandidateMatchDates": sorted({key[1].isoformat() for key in missing}),
                         "candidateForecastsWithoutOriginalMatch": len(candidate_only),
                         "candidateForecastsWithoutOriginalMatchKeys": key_rows(candidate_only),
                         "candidateForecastsWithoutOriginalMatchDates": sorted({key[1].isoformat() for key in candidate_only}),
                         "pairedKeyMismatchCount": len((expected & candidate_expected) - matched)})

    tables, tests = _report_stats(paired)
    conclusions = _dual_loss_conclusions(tests)
    event_original = raw_series.get(guard.EVENT_DAY)
    source_counts = {}
    input_exposure_summary = {}
    input_exposure_records = {}
    exposure_cache: dict[tuple[str, dt.date], dict[str, Any]] = {}
    raw_days = tuple(raw_series)
    for method in METHODS:
        source_counts[method] = 0
        summary = {"latestSource": 0, "lags": {str(lag): 0 for lag in modeling.LAGS},
                   "windows": {str(window): 0 for window in modeling.WINDOWS}, "any": 0}
        exposures = []
        for key, row in original[method].items():
            origin, target, horizon = key
            if target <= guard.EVENT_DAY or not guard.EVALUATION_START <= target <= guard.EVALUATION_END \
                    or horizon not in guard.HORIZONS:
                continue
            cache_key = (method, origin)
            if cache_key not in exposure_cache:
                exposure_cache[cache_key] = _event_input_exposure(raw_series, origin, method, raw_days)
            exposure = exposure_cache[cache_key]
            if exposure["latestSource"]:
                source_counts[method] += 1
            if not exposure["any"]:
                continue
            summary["any"] += 1
            summary["latestSource"] += int(exposure["latestSource"])
            for lag in exposure["lags"]:
                summary["lags"][str(lag)] += 1
            for window in exposure["windows"]:
                summary["windows"][str(window)] += 1
            exposures.append({"origin": origin.isoformat(), "target": target.isoformat(),
                              "horizon": horizon, "sourceDate": row.get("sourceDate"),
                              "exposure": exposure})
        input_exposure_summary[method] = summary
        input_exposure_records[method] = exposures
    return {
        "schemaVersion": 1,
        "experiment": "radish_low_volume_guard",
        "status": "retrospective_exploratory",
        "configuration": {"priceRatioThreshold": price_ratio_threshold},
        "protocol": {"evaluationStart": guard.EVALUATION_START.isoformat(),
                     "evaluationEnd": guard.EVALUATION_END.isoformat(),
                     "horizonsCalendarDays": list(guard.HORIZONS),
                     "lookbackObservedDays": guard.LOOKBACK_OBSERVATIONS,
                     "minimumPriorObservedDays": guard.MIN_PRIOR_OBSERVATIONS,
                     "quarantineBelowMedianFraction": guard.VOLUME_FRACTION,
                     "priceRatioThreshold": price_ratio_threshold,
                     "priceQuarantineRule": ("disabled; low volume alone is sufficient" if price_ratio_threshold is None
                                             else f"price >= {price_ratio_threshold:g}x or <= 1/{price_ratio_threshold:g}x prior observed-day median"),
                     "featurePolicy": "source observations are <= origin, matching the saved dashboard model",
                     "eventInputFollowupPolicy": "separate diagnostic stratum: baseline latest source only; Ridge latest source or exact lag 1/7/28/90 or rolling window 7/28/90 input at origin",
                     "pastRidgeFitHistoryInfluence": "not identifiable from saved predictions; not included in direct-input exposure flags",
                     "quarantinedDatesExcludedFrom": ["price features", "volume features", "training labels"],
                     "trainingHorizonDays": list(modeling.HORIZONS),
                     "evaluationActuals": "raw dashboard observations, including quarantined dates",
                     "minimumPairCountForDM": 30,
                     "dm": "two-sided normal approximation; calendar-day Bartlett HAC; Holm 5% correction across 12 tests per bandwidth"},
        "provenance": {"inputPath": str(path), "inputSha256": _sha256(path),
                       "dashboardGeneratedAt": source.get("generatedAt"),
                       "dashboardSourceMeta": source.get("meta", {}),
                       "analysisConfiguration": {"priceRatioThreshold": price_ratio_threshold,
                                                  "volumeFractionThreshold": guard.VOLUME_FRACTION,
                                                  "lookbackObservedDays": guard.LOOKBACK_OBSERVATIONS,
                                                  "minimumPriorObservedDays": guard.MIN_PRIOR_OBSERVATIONS},
                       "pythonVersion": platform.python_version(),
                       "observationFirstDate": min(raw_series).isoformat(),
                       "observationLastDate": max(raw_series).isoformat(),
                       "rawObservationCount": len(raw_series),
                       "qualityObservationCount": len(filtered_series)},
        "event": {"date": guard.EVENT_DAY.isoformat(),
                  "rawPrice": event_original[0] if event_original else None,
                  "rawNetVolumeKg": event_original[1] if event_original else None,
                  "quarantined": any(row["date"] == guard.EVENT_DAY.isoformat() for row in quarantined),
                  "quarantineDiagnostics": next((row for row in quarantined
                                                  if row["date"] == guard.EVENT_DAY.isoformat()), None),
                  "originalForecastsUsingAsSourceAfterEvent": source_counts,
                  "originalDirectInputExposureSummary": input_exposure_summary,
                  "originalDirectInputExposureForecasts": input_exposure_records,
                  "fitTrainingHistoryInfluence": "not identifiable from saved daily forecasts; direct per-origin feature exposure is reported separately"},
        "quarantinedDates": quarantined,
        "coverage": coverage,
        "metrics": tables,
        "statisticalTests": tests,
        "statisticalConclusions": conclusions,
        "pairedForecastDetails": paired,
        "candidateFitDiagnostics": ridge_run.get("diagnostics", {}),
        "candidateFitCount": len(ridge_run.get("fits", [])),
        "limitations": ["The low-volume threshold was selected after seeing the 2023-01-21 event; this is post-hoc exploratory evidence.",
                        "Revised historical observations do not establish what was known at each original forecast time.",
                        "Nonsignificance is not evidence of equivalence; inference may be unsupported when fewer than 30 pairs remain.",
                        "This retrospective comparison does not establish official competition-score improvement or justify operational deployment."],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Radish low-volume guard experiment", "",
             f"Status: **{report['status']}**", "",
             f"Input SHA-256: `{report['provenance']['inputSha256']}`", "",
             "## Frozen setup", "",
             f"The experiment isolates dates with net volume below 1% of the median of the prior 28 observed dates, after at least 14 prior observations. Price ratio threshold: {report['configuration']['priceRatioThreshold'] if report['configuration']['priceRatioThreshold'] is not None else 'disabled'}. When enabled, the date must also have a price at or above the threshold multiple or at or below its reciprocal of the prior-28 observed-day median. It excludes flagged dates from Ridge/persistence inputs and training labels while retaining raw prices as evaluation truth. Feature inputs include observations on the origin date, matching the saved dashboard forecast semantics. All ten horizons remain in candidate training; paired metrics and coverage include only horizons 1, 5, and 10.", "",
             f"Quarantined dates: {len(report['quarantinedDates'])}; quality observations: {report['provenance']['qualityObservationCount']} / {report['provenance']['rawObservationCount']}.", "",
             "## Coverage", "", "| Method | Original eligible | Candidate eligible | Paired | Original-only | Candidate-only |", "|---|---:|---:|---:|---:|---:|"]
    for row in report["coverage"]:
        lines.append(f"| {row['method']} | {row['eligibleOriginalForecasts']} | {row['candidateForecasts']} | {row['pairedForecasts']} | {row['originalForecastsWithoutCandidateMatch']} | {row['candidateForecastsWithoutOriginalMatch']} |")
        lines.append(f"  - Original-only dates: {', '.join(row['originalForecastsWithoutCandidateMatchDates']) or 'none'}")
        lines.append(f"  - Candidate-only dates: {', '.join(row['candidateForecastsWithoutOriginalMatchDates']) or 'none'}")
    lines += ["", "## Paired errors", "", "MAE is in the source price unit. MAPE is percent. Candidate is the filtered-input baseline or daily Ridge refit; original rows come from the frozen dashboard.", "",
              "| Method | Horizon | Band | n | Original MAE | Candidate MAE | Original MAPE % | Candidate MAPE % |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in report["metrics"]:
        old, new = row["original"], row["candidate"]
        def fmt(value): return "—" if value is None else f"{value:.4f}"
        lines.append(f"| {row['method']} | {row['horizon']} | {row['band']} | {row['pairedCoverage']} | {fmt(old['mae'])} | {fmt(new['mae'])} | {fmt(old['mapePercent'])} | {fmt(new['mapePercent'])} |")
    lines += ["", "## Diebold–Mariano tests", "", "Loss difference is candidate minus original. Each bandwidth has a fixed family of 12 tests (2 methods × 3 horizons × 2 losses); unsupported tests remain in the Holm family.", "",
              "| Bandwidth days | Method | Horizon | Loss | n | Mean difference | p | Holm p | Status |", "|---:|---|---:|---|---:|---:|---:|---:|---|"]
    for row in report["statisticalTests"]:
        def fmt(value): return "—" if value is None else f"{value:.6g}"
        lines.append(f"| {row['bandwidthCalendarDays']} | {row['method']} | {row['horizon']} | {row['loss']} | {row['n']} | {fmt(row.get('meanLossDifference'))} | {fmt(row.get('pValue'))} | {fmt(row.get('holmPValue'))} | {row['holmStatus']} |")
    lines += ["", "## Dual-loss conclusions", "", "Both MAE and MAPE must have mean candidate-minus-original loss in the same direction and Holm-adjusted p < 0.05 for an improved/worsened verdict. HAC 28 days is the sensitivity analysis.", "",
              "| HAC days | Method | Horizon | Verdict |", "|---:|---|---:|---|"]
    for row in report["statisticalConclusions"]:
        lines.append(f"| {row['bandwidthCalendarDays']} | {row['method']} | {row['horizon']} | {row['verdict']} |")
    lines += ["", "## Event input exposure", "", "The source-only follow-up stratum is preserved. The additional direct-input stratum includes the persistence latest source and, for Ridge, exact saved-feature lag/window use at origin. Saved forecasts cannot identify any effect through past Ridge fit history.", "",
              "| Method | Directly exposed forecasts | Latest source | Lag 1 | Lag 7 | Lag 28 | Lag 90 | Window 7 | Window 28 | Window 90 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, row in report["event"]["originalDirectInputExposureSummary"].items():
        lines.append(f"| {method} | {row['any']} | {row['latestSource']} | {row['lags']['1']} | {row['lags']['7']} | {row['lags']['28']} | {row['lags']['90']} | {row['windows']['7']} | {row['windows']['28']} | {row['windows']['90']} |")
    lines += ["", "## Quarantined dates", "", "| Date | Price | Prior median price | Price ratio | Net kg | Prior median kg | Fraction |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in report["quarantinedDates"]:
        lines.append(f"| {row['date']} | {row['price']:.4f} | {row['prior28ObservedDayMedianPrice']:.4f} | {row['priceToPriorMedianRatio']:.6f} | {row['volumeKg']:.4f} | {row['prior28ObservedDayMedianKg']:.4f} | {row['fractionOfMedian']:.6%} |")
    lines += ["", "## Limits", ""] + [f"- {item}" for item in report["limitations"]]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="frozen dashboard JSON")
    parser.add_argument("--output-dir", type=Path, required=True, help="new experiment output directory")
    parser.add_argument("--price-ratio-threshold", type=float, default=None,
                        help="also require price >= threshold or <= reciprocal of prior-28 observed-day median")
    args = parser.parse_args(argv)
    report = run_experiment(args.input, args.price_ratio_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "report.json"
    md_path = args.output_dir / "report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": report["status"], "reportJson": str(json_path),
                      "reportMarkdown": str(md_path), "quarantinedDateCount": len(report["quarantinedDates"]),
                      "pairedForecastCount": sum(row["pairedCoverage"] for row in report["metrics"]
                                                  if row["method"] == "baseline" and row["band"] == "overall")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
