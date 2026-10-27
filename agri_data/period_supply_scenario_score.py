"""Paired scoring and uncertainty summaries for the frozen V8 supply scenario."""
from __future__ import annotations

import argparse
import datetime as dt
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import period_supply_scenario as scenario

OUT = scenario.OUT
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20261016
BLOCK_LENGTHS = (6, 12)
DM_LAGS = (3, 6)
SUPPLY_CELLS = {(crop, horizon) for crop in ("cabbage", "radish") for horizon in (2, 3)}
ALL_CELLS = scenario.EXPECTED_CELLS


def _dekad_index(day: str) -> int:
    date = dt.date.fromisoformat(str(day)[:10])
    dekad = (date.month - 1) * 3 + (1 if date.day <= 10 else 2 if date.day <= 20 else 3)
    return date.year * 36 + dekad - 1


def _float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _pythonize(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_pythonize(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _pythonize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pythonize(item) for item in value]
    return value


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__} in scoring output")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, default=_json_default)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def _key(row: dict[str, Any]) -> str:
    return "|".join((str(row["crop"]), str(int(row["horizon"])), str(row["targetStart"]),
                    str(row["targetEnd"]), str(row["origin"])))


def load_scored_rows(out: Path = OUT) -> list[dict[str, Any]]:
    """Join target-free forecasts to the pinned target snapshot by complete key."""
    manifest_path = out / "forecast-manifest.json"
    forecast_path = out / "forecasts.parquet"
    if not manifest_path.is_file() or not forecast_path.is_file():
        raise FileNotFoundError("forecast stage must finish before scoring")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "forecast_complete" or not manifest.get("labelFreeForecastArtifact"):
        raise ValueError("forecast manifest does not describe a complete label-free artifact")
    if scenario.sha256(forecast_path) != manifest.get("forecastSha256"):
        raise ValueError("forecast artifact hash differs from its manifest")
    match_path = out / "matched-facts.csv"
    if not match_path.is_file() or scenario.sha256(match_path) != manifest.get("matchCsvSha256"):
        raise ValueError("matched-facts.csv is missing or differs from the forecast manifest")
    if manifest.get("inputSha256") != scenario._hashes():
        raise ValueError("forecast inputs differ from the frozen source identities")
    raw = [_pythonize(row) for row in pd.read_parquet(forecast_path).to_dict("records")]
    by_key = {_key(row): row for row in raw}
    if len(by_key) != len(raw) or len(raw) != scenario.EXPECTED_PLANNED_ROWS:
        raise ValueError("forecast keys are duplicated or the planned grid is incomplete")
    labels = scenario._read_forecasts()
    label_by_key = {row["rowKey"]: row for row in labels}
    if len(label_by_key) != len(labels) or set(label_by_key) != set(by_key):
        raise ValueError("forecast and frozen target snapshot keys do not match exactly")
    matched_counts = _matched_fact_counts(out)
    scored = []
    for key, base_row in label_by_key.items():
        row = by_key[key]
        joined = {**base_row, **row, "actual": _float(base_row.get("actual")),
                  "base": _float(row.get("base")), "d0": _float(row.get("d0")),
                  "d1": _float(row.get("d1")), "rowKey": key,
                  "targetIndex": _dekad_index(base_row["targetStart"])}
        joined.update(matched_counts.get(key, {"matchedFactCount": 0, "matchedNumericFactCount": 0,
                                                "matchedUpdateGroupIds": []}))
        if joined["base"] != _float(base_row.get("base")):
            raise ValueError(f"forecast BASE differs from the frozen V5 output at {key}")
        scored.append(joined)
    return sorted(scored, key=lambda r: (r["targetIndex"], r["crop"], r["horizon"], r["origin"]))


def _matched_fact_counts(out: Path) -> dict[str, dict[str, Any]]:
    """Summarize the persisted fact-level match file without changing forecast matching."""
    path = out / "matched-facts.csv"
    if not path.is_file():
        raise FileNotFoundError("matched-facts.csv is required for support diagnostics")
    import csv
    from collections import defaultdict
    rows: dict[str, dict[str, Any]] = defaultdict(lambda: {"factIds": set(), "numericFactIds": set(), "updateIds": set()})
    with path.open(newline="", encoding="utf-8-sig") as f:
        for item in csv.DictReader(f):
            if item.get("availability") != "matched" or not item.get("factId"):
                continue
            key = "|".join((str(item["crop"]), str(int(item["horizon"])), str(item["targetStart"]),
                            str(item["targetEnd"]), str(item["origin"])))
            entry = rows[key]
            fid = str(item["factId"])
            entry["factIds"].add(fid)
            if item.get("tieConflict", "false").lower() != "true" and any(
                    _float(item.get(name)) is not None for name in ("value", "valueLower", "valueUpper")):
                entry["numericFactIds"].add(fid)
            if item.get("updateGroupId"):
                entry["updateIds"].add(str(item["updateGroupId"]))
    return {key: {"matchedFactCount": len(value["factIds"]),
                  "matchedNumericFactCount": len(value["numericFactIds"]),
                  "matchedUpdateGroupIds": sorted(value["updateIds"])} for key, value in rows.items()}


def _errors(rows: list[dict[str, Any]], prediction: str) -> list[tuple[dict[str, Any], float, float]]:
    result = []
    for row in rows:
        actual, pred = _float(row.get("actual")), _float(row.get(prediction))
        if actual is None or pred is None or actual == 0:
            continue
        ae = abs(actual - pred)
        result.append((row, ae, 100.0 * ae / abs(actual)))
    return result


def _cell_metric(rows: list[dict[str, Any]], prediction: str) -> dict[str, Any]:
    errors = _errors(rows, prediction)
    return {"prediction": prediction, "n": len(errors),
            "mae": float(np.mean([e[1] for e in errors])) if errors else None,
            "mape_pct": float(np.mean([e[2] for e in errors])) if errors else None}


def _comparison_differences(rows: list[dict[str, Any]], candidate: str,
                            comparator: str, metric: str) -> list[tuple[int, float]]:
    out = []
    for row in rows:
        actual = _float(row.get("actual"))
        p1, p0 = _float(row.get(candidate)), _float(row.get(comparator))
        if actual is None or actual == 0 or p1 is None or p0 is None:
            continue
        e1, e0 = abs(actual - p1), abs(actual - p0)
        diff = e1 - e0 if metric == "mae" else 100.0 * (e1 - e0) / abs(actual)
        out.append((int(row["targetIndex"]), float(diff)))
    return out


def _normal_two_sided_p(z: float) -> float:
    return math.erfc(abs(z) / math.sqrt(2.0))


def _dm_test(differences: list[tuple[int, float]], lag: int) -> dict[str, Any]:
    # Index by the original target dekad so absent dekads remain gaps in HAC.
    grouped: dict[int, list[float]] = {}
    for t, value in differences:
        grouped.setdefault(int(t), []).append(float(value))
    vals = {t: float(np.mean(values)) for t, values in grouped.items()}
    n = len(vals)
    if n < lag + 2:
        return {"n": n, "meanDifference": None, "statistic": None, "pRaw": None,
                "status": "unsupported", "reason": f"n={n} target dekads is insufficient for HAC lag {lag}"}
    mean = float(np.mean(list(vals.values())))
    centered = {t: v - mean for t, v in vals.items()}
    lrv = sum(v * v for v in centered.values()) / n
    for h in range(1, lag + 1):
        cross = sum(centered[t] * centered[t + h] for t in centered if t + h in centered) / n
        lrv += 2.0 * (1.0 - h / (lag + 1.0)) * cross
    if not math.isfinite(lrv) or lrv <= 1e-15:
        return {"n": n, "meanDifference": mean, "statistic": None, "pRaw": None,
                "status": "unsupported", "reason": "HAC long-run variance is zero or non-finite"}
    statistic = mean / math.sqrt(lrv / n)
    return {"n": n, "meanDifference": mean, "statistic": statistic,
            "pRaw": _normal_two_sided_p(statistic), "status": "computed", "reason": None}


def _holm(results: list[dict[str, Any]], family_size: int = 16) -> None:
    """Holm adjustment over the full frozen family; unsupported slots count as p=1."""
    ordered = sorted(results, key=lambda r: (r["pRaw"] if r["pRaw"] is not None else 1.0,
                                              r["hypothesisId"]))
    running = 0.0
    for rank, result in enumerate(ordered):
        raw = result["pRaw"] if result["pRaw"] is not None else 1.0
        adjusted = min(1.0, (family_size - rank) * raw)
        running = max(running, adjusted)
        result["pHolm"] = running
        result["familySize"] = family_size
        result["holmSignificant5pct"] = result["pRaw"] is not None and running < 0.05


def dm_family(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for lag in DM_LAGS:
        family = []
        for crop, horizon in sorted(SUPPLY_CELLS):
            cell = [r for r in scored if r["crop"] == crop and r["horizon"] == horizon]
            for candidate, comparator in (("d1", "d0"), ("d1", "base")):
                for metric in ("mae", "ape"):
                    hypothesis_id = f"{candidate}-{comparator}|{crop}|h{horizon}|{metric}|hac{lag}"
                    test = _dm_test(_comparison_differences(cell, candidate, comparator, metric), lag)
                    family.append({"hypothesisId": hypothesis_id, "crop": crop, "horizon": horizon,
                                   "candidate": candidate, "comparator": comparator,
                                   "metric": metric, "hacLagDekads": lag, **test})
        if len(family) != 16:
            raise AssertionError("the frozen DM family must contain 16 hypotheses")
        _holm(family, 16)
        results.extend(family)
    return results


def _block_resample_indices(calendar: list[int], length: int, rng: np.random.Generator) -> list[int]:
    if len(calendar) < length:
        return []
    min_idx, max_idx = calendar[0], calendar[-1]
    starts = np.arange(min_idx, max_idx - length + 2)
    if not len(starts):
        return []
    sampled: list[int] = []
    while len(sampled) < len(calendar):
        start = int(rng.choice(starts))
        sampled.extend(range(start, start + length))
    return sampled[:len(calendar)]


def _mean_for_boot(rows: list[dict[str, Any]], candidate: str, comparator: str,
                   metric: str, resampled: list[int]) -> float | None:
    by_index: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_index.setdefault(int(row["targetIndex"]), []).append(row)
    diff_values = []
    for idx in resampled:
        for row in by_index.get(idx, []):
            actual = _float(row.get("actual"))
            p1, p0 = _float(row.get(candidate)), _float(row.get(comparator))
            if actual is None or actual == 0 or p1 is None or p0 is None:
                continue
            e1, e0 = abs(actual - p1), abs(actual - p0)
            diff_values.append(e1 - e0 if metric == "mae" else 100.0 * (e1 - e0) / abs(actual))
    return float(np.mean(diff_values)) if diff_values else None


def block_bootstrap(scored: list[dict[str, Any]], replicates: int = BOOTSTRAP_REPLICATES,
                    seed: int = BOOTSTRAP_SEED) -> list[dict[str, Any]]:
    """Shared calendar-block bootstrap for cells and equal-cell 4/8-cell summaries."""
    observed_calendar = sorted({int(r["targetIndex"]) for r in scored})
    calendar = list(range(observed_calendar[0], observed_calendar[-1] + 1)) if observed_calendar else []
    output: list[dict[str, Any]] = []
    for length in BLOCK_LENGTHS:
        rng = np.random.default_rng(seed)
        draws = [_block_resample_indices(calendar, length, rng) for _ in range(replicates)]
        for candidate, comparator in (("d1", "d0"), ("d1", "base")):
            for metric in ("mae", "ape"):
                for aggregate, cells in (("supply_4cell", SUPPLY_CELLS), ("all_8cell", ALL_CELLS)):
                    samples = []
                    for draw in draws:
                        per_cell = []
                        for crop, horizon in sorted(cells):
                            cell = [r for r in scored if r["crop"] == crop and r["horizon"] == horizon]
                            value = _mean_for_boot(cell, candidate, comparator, metric, draw)
                            if value is not None:
                                per_cell.append(value)
                        if per_cell:
                            samples.append(float(np.mean(per_cell)))
                    # Pairs/cell/methods use the same sampled calendar blocks.
                    output.append({"blockLengthDekads": length, "candidate": candidate,
                        "comparator": comparator, "metric": metric, "aggregate": aggregate,
                        "replicatesRequested": replicates, "replicatesUsable": len(samples),
                        "seed": seed, "ci95Lower": float(np.quantile(samples, 0.025)) if samples else None,
                        "ci95Upper": float(np.quantile(samples, 0.975)) if samples else None,
                        "meanDifference": float(np.mean(samples)) if samples else None,
                        "status": "computed" if samples else "unsupported_no_paired_scores"})
                for crop, horizon in sorted(ALL_CELLS):
                    cell = [r for r in scored if r["crop"] == crop and r["horizon"] == horizon]
                    samples = [_mean_for_boot(cell, candidate, comparator, metric, draw) for draw in draws]
                    samples = [v for v in samples if v is not None]
                    output.append({"blockLengthDekads": length, "candidate": candidate,
                        "comparator": comparator, "metric": metric, "aggregate": f"{crop}_h{horizon}",
                        "replicatesRequested": replicates, "replicatesUsable": len(samples), "seed": seed,
                        "ci95Lower": float(np.quantile(samples, 0.025)) if samples else None,
                        "ci95Upper": float(np.quantile(samples, 0.975)) if samples else None,
                        "meanDifference": float(np.mean(samples)) if samples else None,
                        "status": "computed" if samples else "unsupported_no_paired_scores"})
    return output


def _diagnostics(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, int, str, int], list[dict[str, Any]]] = {}
    for row in scored:
        target_year = dt.date.fromisoformat(str(row["targetStart"])[:10]).year
        by_cell.setdefault((row["crop"], row["horizon"], row["split"], target_year), []).append(row)
    diagnostics = []
    for (crop, horizon, split, target_year), rows in sorted(by_cell.items()):
        paired = [r for r in rows if r["actual"] is not None and r["d0"] is not None and r["d1"] is not None and r["base"] is not None]
        d1_base = [r for r in rows if r["d1"] is not None and r["base"] is not None]
        matched = [r for r in rows if r.get("selectedFactIds")]
        numeric = [r for r in rows if r.get("supportedCurrentNumericColumns")]
        fresh_d1 = _errors([r for r in rows if r.get("selectedFactIds")], "d1")
        fresh_d0 = _errors([r for r in rows if r.get("selectedFactIds")], "d0")
        first_new_rows = [r for r in rows if r.get("firstNewUpdateOrigin")]
        first_new_d0 = _errors(first_new_rows, "d0")
        first_new_d1 = _errors(first_new_rows, "d1")
        first_new_base = _errors(first_new_rows, "base")
        changed_d1 = [r for r in rows if r["d1"] is not None and r["base"] is not None and not math.isclose(r["d1"], r["base"], rel_tol=1e-12, abs_tol=1e-12)]
        changed_d1_score_keys = {r["rowKey"] for r in changed_d1}
        update_uses: dict[str, int] = {}
        for r in rows:
            for group in (r.get("matchedUpdateGroupIds") or []):
                update_uses[group] = update_uses.get(group, 0) + 1
        diagnostic = {"crop": crop, "horizon": horizon, "split": split, "targetYear": target_year,
            "plannedRows": len(rows), "labelRows": sum(r["actual"] is not None for r in rows),
            "baseAvailableRows": sum(r["base"] is not None for r in rows),
            "d0AvailableRows": sum(r["d0"] is not None for r in rows),
            "d1AvailableRows": sum(r["d1"] is not None for r in rows),
            "commonPairedScoreRows": len(paired),
            "d0BaseFallbackRows": sum(r.get("d0Status") == "base_fallback" for r in rows),
            "d1BaseFallbackRows": sum(r.get("d1Status") == "base_fallback" for r in rows),
            "d0RidgeRows": sum(r.get("d0Status") == "ridge_prequential" for r in rows),
            "d1RidgeRows": sum(r.get("d1Status") == "ridge_prequential" for r in rows),
            "matchedFactRows": len(matched), "rowsWithSupportedNumericColumns": len(numeric),
            "matchedFactRecordCount": sum(r.get("matchedFactCount", 0) for r in rows),
            "matchedDistinctFactIds": len({fid for r in rows for fid in (r.get("selectedFactIds") or [])}),
            "rowsWithNumericMatchedFacts": sum(r.get("matchedNumericFactCount", 0) > 0 for r in rows),
            "matchedNumericFactRecordCount": sum(r.get("matchedNumericFactCount", 0) for r in rows),
            "matchedDistinctUpdateGroupIds": len({gid for r in rows for gid in (r.get("matchedUpdateGroupIds") or [])}),
            "matchedUpdateGroupOriginAppearances": sum(update_uses.values()),
            "repeatedUpdateGroupOriginAppearances": sum(max(0, count - 1) for count in update_uses.values()),
            "firstNewUpdateRows": len(first_new_rows), "firstNewUpdateGroups": sum(len(r.get("newUpdateGroups", [])) for r in rows),
            "d1ChangedFromBaseRows": len(changed_d1),
            "d1ChangedFromBaseScoreRows": sum(r["rowKey"] in changed_d1_score_keys and r["actual"] is not None for r in rows),
            "freshFactD0ScoreN": len(fresh_d0), "freshFactD0MAE": float(np.mean([x[1] for x in fresh_d0])) if fresh_d0 else None,
            "freshFactD0MAPE_pct": float(np.mean([x[2] for x in fresh_d0])) if fresh_d0 else None,
            "freshFactD1ScoreN": len(fresh_d1), "freshFactD1MAE": float(np.mean([x[1] for x in fresh_d1])) if fresh_d1 else None,
            "freshFactD1MAPE_pct": float(np.mean([x[2] for x in fresh_d1])) if fresh_d1 else None,
            "firstNewUpdateD0ScoreN": len(first_new_d0),
            "firstNewUpdateD0MAE": float(np.mean([x[1] for x in first_new_d0])) if first_new_d0 else None,
            "firstNewUpdateD0MAPE_pct": float(np.mean([x[2] for x in first_new_d0])) if first_new_d0 else None,
            "firstNewUpdateD1ScoreN": len(first_new_d1),
            "firstNewUpdateD1MAE": float(np.mean([x[1] for x in first_new_d1])) if first_new_d1 else None,
            "firstNewUpdateD1MAPE_pct": float(np.mean([x[2] for x in first_new_d1])) if first_new_d1 else None,
            "firstNewUpdateBaseScoreN": len(first_new_base),
            "firstNewUpdateBaseMAE": float(np.mean([x[1] for x in first_new_base])) if first_new_base else None,
            "firstNewUpdateBaseMAPE_pct": float(np.mean([x[2] for x in first_new_base])) if first_new_base else None,
            "trainingOriginCounts": {str(k): sum(r.get("trainingOrigins") == k for r in rows) for k in sorted({r.get("trainingOrigins") for r in rows})},
            "fallbackReasons": dict(pd.Series([r.get("fallbackReason") for r in rows if r.get("fallbackReason")]).value_counts()),
            "targetStatusCounts": dict(pd.Series([r.get("targetStatus") for r in rows]).value_counts(dropna=False))}
        for method in ("base", "d0", "d1"):
            diagnostic[f"{method}Metrics"] = _cell_metric(rows, method)
        diagnostics.append(diagnostic)
    return diagnostics


def _mark_new_updates(scored: list[dict[str, Any]]) -> None:
    for crop, horizon in sorted(ALL_CELLS):
        rows = sorted((r for r in scored if r["crop"] == crop and r["horizon"] == horizon),
                      key=lambda r: (r["origin"], r["targetStart"], r["split"]))
        seen: set[str] = set()
        for row in rows:
            groups = {str(x) for x in (row.get("selectedUpdateGroups") or []) if x}
            new = sorted(groups - seen)
            row["newUpdateGroups"] = new
            row["firstNewUpdateOrigin"] = bool(new)
            seen.update(groups)


def primary_evaluation_rows(scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select the frozen held-out 2024–2026 rows for inference and adoption criterion."""
    return [row for row in scored if row.get("split") == "evaluation"]


def score(out: Path = OUT, replicates: int = BOOTSTRAP_REPLICATES) -> dict[str, Any]:
    if replicates != BOOTSTRAP_REPLICATES:
        raise ValueError(f"the frozen protocol requires exactly {BOOTSTRAP_REPLICATES} block-bootstrap replicates")
    rows = load_scored_rows(out)
    _mark_new_updates(rows)
    evaluation_rows = primary_evaluation_rows(rows)
    development_rows = [row for row in rows if row["split"] == "development"]
    if len(evaluation_rows) != 792 or len({row["targetIndex"] for row in evaluation_rows}) != 99:
        raise ValueError("frozen evaluation grid must contain 792 rows over 99 target dekads")
    dm = dm_family(evaluation_rows)
    bootstrap = block_bootstrap(evaluation_rows, replicates=replicates)
    diagnostics = _diagnostics(rows)
    metrics = []
    for split, split_rows in (("primary_evaluation", evaluation_rows),
                              ("development_descriptive", development_rows),
                              ("all_v5_descriptive", rows)):
        for crop, horizon in sorted(ALL_CELLS):
            cell = [r for r in split_rows if r["crop"] == crop and r["horizon"] == horizon]
            for method in ("base", "d0", "d1"):
                metrics.append({"aggregate": "cell", "crop": crop, "horizon": horizon,
                                "split": split, **_cell_metric(cell, method)})
        for aggregate, cells in (("supply_4cell", SUPPLY_CELLS), ("all_8cell", ALL_CELLS)):
            for method in ("base", "d0", "d1"):
                cell_scores = []
                for crop, horizon in sorted(cells):
                    cell_scores.append(_cell_metric([r for r in split_rows if r["crop"] == crop and r["horizon"] == horizon], method))
                usable = [x for x in cell_scores if x["n"]]
                metrics.append({"aggregate": aggregate, "split": split, "prediction": method,
                    "weighting": "equal_cell_mean", "cellsPlanned": len(cells), "cellsWithScores": len(usable),
                    "nRowsSum": sum(x["n"] for x in usable),
                    "mae": float(np.mean([x["mae"] for x in usable])) if usable else None,
                    "mape_pct": float(np.mean([x["mape_pct"] for x in usable])) if usable else None,
                    "mapeAtMost10pct": (float(np.mean([x["mape_pct"] for x in usable])) <= 10.0) if usable else None})
    improvement = _observed_improvement(metrics)
    labels = {r["rowKey"]: r["actual"] for r in rows}
    scored_frame = pd.DataFrame([{k: v for k, v in r.items() if k not in {"selectedFacts", "featureRows", "featureNames"}} for r in rows])
    scored_frame.to_parquet(out / "scored-rows.parquet", index=False)
    _write_csv(out / "metrics.csv", metrics)
    _write_csv(out / "dm-tests.csv", dm)
    _write_csv(out / "calendar-block-bootstrap.csv", bootstrap)
    _write_csv(out / "matched-support-diagnostics.csv", diagnostics)
    manifest = {"status": "scoring_complete", "forecastManifestSha256": scenario.sha256(out / "forecast-manifest.json"),
        "scoringImplementationSha256": scenario.sha256(Path(__file__)),
        "forecastSha256": scenario.sha256(out / "forecasts.parquet"), "scoredRowsSha256": scenario.sha256(out / "scored-rows.parquet"),
        "metricsSha256": scenario.sha256(out / "metrics.csv"), "dmTestsSha256": scenario.sha256(out / "dm-tests.csv"),
        "bootstrapSha256": scenario.sha256(out / "calendar-block-bootstrap.csv"),
        "diagnosticsSha256": scenario.sha256(out / "matched-support-diagnostics.csv"),
        "plannedKeys": len(rows), "labeledRows": sum(v is not None for v in labels.values()),
        "missingTargetRowsRetained": sum(v is None for v in labels.values()),
        "primaryInferenceSplit": "evaluation 2024-01 through 2026-09 only",
        "developmentRowsExcludedFromPrimaryInference": True,
        "dmFamilyHypothesesPerLag": 16, "dmHacLags": list(DM_LAGS), "holmAlpha": 0.05,
        "bootstrapReplicates": replicates, "bootstrapSeed": BOOTSTRAP_SEED,
        "bootstrapBlockLengthsDekads": list(BLOCK_LENGTHS), "targetGapsPreserved": True,
        "sameCalendarDrawsAcrossCellsAndMethods": True, "bootstrapScope": "evaluation only",
        "observedImprovementAssessment": improvement,
        "interpretation": "Exploratory publication-scenario experiment; not strict historical-vintage evidence or adoption.",
        "scoringLabelSource": str(scenario.PRICE_SOURCE),
        "scoringLabelSourceSha256": scenario.sha256(scenario.PRICE_SOURCE),
        "sourceInputSha256": scenario._hashes()}
    (out / "scoring-manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _observed_improvement(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    by_method = {row["prediction"]: row for row in metrics
                 if row.get("aggregate") == "supply_4cell" and row.get("split") == "primary_evaluation"}
    if set(by_method) != {"base", "d0", "d1"} or any(by_method[x].get("mape_pct") is None for x in by_method):
        return {"status": "not_assessable", "reason": "one or more 4-cell aggregate metrics are unavailable"}
    d1, d0, base = by_method["d1"], by_method["d0"], by_method["base"]
    checks = {
        "d1MapeBelowD0": d1["mape_pct"] < d0["mape_pct"],
        "d1MapeBelowBase": d1["mape_pct"] < base["mape_pct"],
        "d1MaeNotWorseThanD0": d1["mae"] <= d0["mae"],
        "d1MaeNotWorseThanBase": d1["mae"] <= base["mae"],
        "samePairedRowCoverage": d1["nRowsSum"] == d0["nRowsSum"] == base["nRowsSum"],
        "allFourCellsScored": d1["cellsWithScores"] == d0["cellsWithScores"] == base["cellsWithScores"] == 4,
    }
    return {"status": "observed_improvement" if all(checks.values()) else "criterion_not_met",
            "checks": checks,
            "interpretation": "Descriptive protocol criterion only; it does not establish statistical significance, generalization, strict as-of performance, or adoption."}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--replicates", type=int, default=BOOTSTRAP_REPLICATES)
    args = parser.parse_args(argv)
    print(json.dumps(score(args.out, args.replicates), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
