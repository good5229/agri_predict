"""Read-only evaluator for immutable live forecast artifacts and a frozen price snapshot."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from . import dashboard, direct_horizon

KST = ZoneInfo("Asia/Seoul")
PILOT_START = dt.date(2026, 10, 6)
HORIZONS = tuple(range(1, 11))
PRODUCTS = tuple(dashboard.PRODUCTS)
COMPARATORS = ("last_observed_persistence",)
LOSSES = ("mae", "mape")
HAC = (9, 28)
FAMILY_SIZE = len(PRODUCTS) * len(HORIZONS) * len(LOSSES)


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _date(value: Any, label: str = "date") -> dt.date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc


def _date_arg(value: str) -> dt.date:
    return _date(value, "date argument")


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{label} must be finite" + (" and positive" if positive else ""))
    return result


def _snapshot(path: Path) -> tuple[dict[str, dict[dt.date, float]], dt.date, dt.date, str]:
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict) or raw.get("schemaVersion") != 1:
        raise ValueError("price snapshot must use dashboard schemaVersion 1")
    meta = raw.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("price snapshot meta is required")
    data_end = _date(meta.get("dataEnd"), "snapshot meta.dataEnd")
    evaluation_end = _date(meta.get("evaluationEnd"), "snapshot meta.evaluationEnd")
    aggregation = meta.get("aggregation")
    if (not isinstance(aggregation, str) or "부호 있는 총금액(totprc)" not in aggregation
            or "총중량(unit_tot_qty)" not in aggregation or "kg" not in aggregation):
        raise ValueError("snapshot must document the validated signed-price/kg aggregation")
    market_names = meta.get("marketNames")
    if not isinstance(market_names, list) or set(market_names) != set(dashboard.MARKETS.values()):
        raise ValueError("snapshot marketNames do not match the two validated markets")
    products = raw.get("products")
    if not isinstance(products, list):
        raise ValueError("price snapshot products must be a list")
    actuals: dict[str, dict[dt.date, float]] = {}
    for product in products:
        if not isinstance(product, dict):
            raise ValueError("snapshot product must be an object")
        pid = product.get("id")
        if pid not in PRODUCTS or pid in actuals:
            raise ValueError("unknown or duplicate product in price snapshot")
        rows = product.get("observations")
        if not isinstance(rows, list):
            raise ValueError(f"snapshot observations must be a list for {pid}")
        series: dict[dt.date, float] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("snapshot observation must be an object")
            day = _date(row.get("date"), "observation date")
            price = _number(row.get("actual"), "observation actual", positive=True)
            _number(row.get("volumeKg"), "observation volumeKg", positive=True)
            if day in series:
                raise ValueError(f"duplicate snapshot observation {pid}/{day}")
            if day > data_end or day > evaluation_end:
                raise ValueError(f"observation {pid}/{day} exceeds validated snapshot end")
            series[day] = price
        actuals[pid] = series
    if set(actuals) != set(PRODUCTS):
        raise ValueError("snapshot must contain all dashboard products")
    return actuals, data_end, evaluation_end, _sha_bytes(raw_bytes)


def _validate_provenance(rows: Any, *, selected_end: dt.date):
    if not isinstance(rows, list):
        raise ValueError("selected provenance must be the daily_training list export")
    result: dict[tuple[dt.date, str], dict[str, Any]] = {}
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("selected provenance entries must be objects")
        day = _date(row.get("date"), "provenance date")
        market, product = row.get("market"), row.get("product")
        if market not in dashboard.MARKETS or product not in PRODUCTS:
            raise ValueError("selected provenance has an unknown market or product")
        count = row.get("rowCount")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("selected provenance rowCount must be a nonnegative integer")
        job_id = row.get("jobId")
        if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id <= 0:
            raise ValueError("selected provenance jobId must be a positive integer")
        for field in ("epochId", "responseAt"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"selected provenance {field} is required")
        if not isinstance(row.get("selectedPrevious"), bool):
            raise ValueError("selected provenance selectedPrevious must be boolean")
        key = (day, product)
        scope_key = (day, market, product)
        if scope_key in seen:
            raise ValueError(f"duplicate selected provenance scope {day}/{market}/{product}")
        seen.add(scope_key)
        result.setdefault(key, {})[market] = {"rowCount": count, "jobId": job_id,
            "epochId": row["epochId"], "responseAt": row["responseAt"],
            "selectedPrevious": row["selectedPrevious"]}
        if day > selected_end:
            raise ValueError("selected provenance extends beyond snapshot meta.evaluationEnd")
    return result


def _provenance(path: Path, *, selected_end: dt.date):
    payload = path.read_bytes()
    rows = json.loads(payload)
    scopes = _validate_provenance(rows, selected_end=selected_end)
    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return scopes, _sha_bytes(payload), _sha_bytes(canonical.encode("utf-8"))


def _truth_status(target: dt.date, product: str, actuals: dict[str, dict[dt.date, float]],
                  provenance: dict[tuple[dt.date, str], dict[str, Any]], evaluation_end: dt.date):
    if target > evaluation_end:
        return "immature_target", None
    scopes = provenance.get((target, product), {})
    if set(scopes) != set(dashboard.MARKETS):
        return "incomplete_or_unknown_scope", None
    row_count = sum(scopes[market]["rowCount"] for market in dashboard.MARKETS)
    if row_count == 0:
        return "complete_empty_response", None
    actual = actuals[product].get(target)
    if actual is None:
        return "complete_rows_no_eligible_price", None
    return "validated_actual", actual


def _artifact(path: Path) -> dict[str, Any]:
    raw_bytes = path.read_bytes()
    raw = json.loads(raw_bytes)
    if not isinstance(raw, dict) or raw.get("schemaVersion") != 1:
        raise ValueError(f"live forecast must use schemaVersion 1: {path}")
    issue = raw.get("issueAtKst")
    try:
        parsed = dt.datetime.fromisoformat(issue)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid issueAtKst in {path}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"issueAtKst must have a timezone: {path}")
    local = parsed.astimezone(KST)
    issue_day = _date(raw.get("issueDateKst"), "issueDateKst")
    if local.date() != issue_day or path.parent.name != issue_day.isoformat():
        raise ValueError(f"issue date does not match artifact directory: {path}")
    cutoff = _date(raw.get("sourceCutoffDate"), "sourceCutoffDate")
    if cutoff >= issue_day:
        raise ValueError("sourceCutoffDate must precede issue date")
    products = raw.get("products")
    if not isinstance(products, list):
        raise ValueError(f"live forecast products must be a list: {path}")
    checked = {}
    for item in products:
        if not isinstance(item, dict):
            raise ValueError("live forecast product must be an object")
        pid = item.get("productId")
        if pid not in PRODUCTS or pid in checked:
            raise ValueError("unknown or duplicate live forecast product")
        preds = item.get("predictions")
        if not isinstance(preds, list):
            raise ValueError(f"predictions must be a list for {pid}")
        per_horizon = {}
        for pred in preds:
            if not isinstance(pred, dict):
                raise ValueError("prediction must be an object")
            horizon = pred.get("horizonDays")
            if isinstance(horizon, bool) or horizon not in HORIZONS:
                raise ValueError(f"invalid forecast horizon for {pid}")
            target = _date(pred.get("target"), "prediction target")
            if target != issue_day + dt.timedelta(days=horizon):
                raise ValueError(f"target/origin mismatch for {pid}/{horizon}")
            pred_source = _date(pred.get("sourceDate"), "prediction sourceDate")
            if pred_source > cutoff:
                raise ValueError(f"prediction sourceDate exceeds source cutoff for {pid}/h{horizon}")
            if horizon in per_horizon:
                raise ValueError(f"duplicate prediction {pid}/h{horizon}")
            per_horizon[horizon] = {"target": target, "predicted": _number(
                pred.get("predictedPrice"), "predictedPrice", positive=True), "sourceDate": pred_source.isoformat()}
        last = item.get("lastObservedPrice")
        last_value = None if last is None else _number(last, "lastObservedPrice", positive=True)
        source_date = item.get("sourceDate")
        if source_date is not None:
            source_day = _date(source_date, "sourceDate")
            if source_day > cutoff:
                raise ValueError(f"sourceDate exceeds source cutoff for {pid}")
            if any(pred["sourceDate"] != source_date for pred in per_horizon.values()):
                raise ValueError(f"product/prediction sourceDate mismatch for {pid}")
        last_date = item.get("lastObservedPriceDate")
        if last_date is not None:
            last_day = _date(last_date, "lastObservedPriceDate")
            if source_date is not None and last_day.isoformat() != source_date:
                raise ValueError(f"lastObservedPriceDate/sourceDate mismatch for {pid}")
        checked[pid] = {"status": item.get("status"), "reason": item.get("reason"),
            "sourceDate": source_date, "lastObservedPriceDate": last_date, "lastObservedPrice": last_value,
            "qualityGuard": item.get("qualityGuard"), "predictions": per_horizon}
    if set(checked) != set(PRODUCTS):
        raise ValueError("live forecast must contain every dashboard product")
    return {"date": issue_day, "issuedAt": local, "hash": _sha_bytes(raw_bytes),
        "sourceCutoffDate": raw.get("sourceCutoffDate"), "products": checked, "path": str(path)}


def _metrics(actual: list[float], predicted: list[float]) -> dict[str, float | None]:
    return direct_horizon._metrics(actual, predicted)


def _forecast_files(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("????-??-??/*.json")
                  if not path.stem.endswith("-selected-provenance"))


def build(files: list[Path], actuals: dict[str, dict[dt.date, float]], data_end: dt.date,
          evaluation_end: dt.date, snapshot_sha256: str,
          provenance: dict[tuple[dt.date, str], dict[str, Any]], provenance_sha256: str,
          provenance_byte_sha256: str, start: dt.date, end: dt.date) -> dict[str, Any]:
    if end < start:
        raise ValueError("evaluation end precedes start")
    parsed = [_artifact(path) for path in files]
    by_day: dict[dt.date, list[dict[str, Any]]] = defaultdict(list)
    for row in parsed:
        if start <= row["date"] <= end:
            by_day[row["date"]].append(row)
    selected, daily_audit = {}, {}
    for i in range((end - start).days + 1):
        day = start + dt.timedelta(days=i)
        eligible = sorted((r for r in by_day.get(day, []) if dt.time(10) <= r["issuedAt"].time() <= dt.time(12)),
                          key=lambda r: (r["issuedAt"], r["path"]))
        if eligible:
            chosen = eligible[0]
            selected[day] = chosen
            daily_audit[day.isoformat()] = {"status": "issued", "selectedFile": chosen["path"],
                "selectedSha256": chosen["hash"], "issueAtKst": chosen["issuedAt"].isoformat(),
                "sourceCutoffDate": chosen["sourceCutoffDate"], "ignoredInWindowFiles": len(eligible)-1,
                "outOfWindowFiles": len(by_day.get(day, []))-len(eligible)}
        else:
            daily_audit[day.isoformat()] = {"status": "not_issued_in_window", "outOfWindowFiles": len(by_day.get(day, []))}

    paired_rows, counts, exclusions = [], {}, {}
    for product in PRODUCTS:
        for horizon in HORIZONS:
            key = f"{product}/{horizon}"
            c = Counter(); reasons = Counter()
            for i in range((end - start).days + 1):
                issue_day = start + dt.timedelta(days=i)
                c["issuancePossible"] += 1
                artifact = selected.get(issue_day)
                if artifact is None:
                    reasons["no_issue_in_window"] += 1; continue
                c["successfulArtifactIssue"] += 1
                item = artifact["products"][product]
                status = item["status"]
                if status == "available":
                    c["successfulIssueAvailable"] += 1
                else:
                    c["productHeldOrUnavailable"] += 1
                    reasons[f"product_status:{status or 'missing'}"] += 1
                    if item["reason"]:
                        reasons[f"hold_reason:{item['reason']}"] += 1
                target = issue_day + dt.timedelta(days=horizon)
                truth_state, actual = _truth_status(target, product, actuals, provenance, evaluation_end)
                if truth_state == "immature_target":
                    c["immatureTarget"] += 1; reasons[truth_state] += 1; continue
                if truth_state == "incomplete_or_unknown_scope":
                    c[truth_state] += 1
                    reasons[truth_state] += 1
                    continue
                c["matureTarget"] += 1
                if truth_state != "validated_actual":
                    c[truth_state] += 1
                    reasons[truth_state] += 1
                    continue
                c["validatedActual"] += 1
                if status != "available":
                    reasons["issue_product_not_available"] += 1; continue
                forecast = item["predictions"].get(horizon)
                baseline = item["lastObservedPrice"]
                if baseline is None:
                    reasons["baseline_price_missing"] += 1; continue
                c["baselineCommonOpportunity"] += 1
                if forecast is None:
                    reasons["forecast_missing_for_horizon"] += 1; continue
                c["predictionAndBaselineAvailable"] += 1
                if actual is None:
                    continue
                row = {"product": product, "horizon": horizon, "issueDate": issue_day.isoformat(),
                    "issueAtKst": artifact["issuedAt"].isoformat(), "forecastFileSha256": artifact["hash"],
                    "target": target.isoformat(), "actual": actual, "predicted": forecast["predicted"],
                    "persistence": baseline, "sourceDate": item["sourceDate"],
                    "sourceCutoffDate": artifact["sourceCutoffDate"]}
                paired_rows.append(row); c["commonPairs"] += 1
            counts[key] = {name: c.get(name, 0) for name in ("issuancePossible", "successfulArtifactIssue",
                "successfulIssueAvailable", "productHeldOrUnavailable", "matureTarget", "immatureTarget", "validatedActual",
                "incomplete_or_unknown_scope", "complete_empty_response", "complete_rows_no_eligible_price",
                "baselineCommonOpportunity", "predictionAndBaselineAvailable", "commonPairs")}
            exclusions[key] = dict(reasons)
            counts[key]["candidateAndBaselineMaturePredictions"] = counts[key]["predictionAndBaselineAvailable"]
    summaries = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            key = f"{product}/{horizon}"
            rows = [r for r in paired_rows if r["product"] == product and r["horizon"] == horizon]
            for loss in LOSSES:
                actual = [r["actual"] for r in rows]
                preds = [r["predicted"] for r in rows]
                baseline = [r["persistence"] for r in rows]
                diffs = {}
                for row in rows:
                    y, p, b = row["actual"], row["predicted"], row["persistence"]
                    diffs[dt.date.fromisoformat(row["target"])] = abs(y-p)-abs(y-b) if loss == "mae" else 100*(abs(y-p)/y-abs(y-b)/y)
                n = len(rows)
                tests = {str(bw): (direct_horizon._dm(diffs, bw) if n >= 30 else {
                    "status": "insufficient_pairs", "n": n, "meanLossDifference": (sum(diffs.values())/n if n else None),
                    "statistic": None, "pValue": 1.0, "reason": f"{n} common pairs; minimum is 30"}) for bw in HAC}
                for test in tests.values():
                    if test["status"] != "tested" and "reason" not in test:
                        test["reason"] = "HAC variance is zero or invalid; test unavailable"
                summaries.append({"product": product, "horizon": horizon, "loss": loss, "n": n,
                    "forecastMetric": _metrics(actual, preds)[loss], "persistenceMetric": _metrics(actual, baseline)[loss],
                    "meanLossDifference": (sum(diffs.values())/n if n else None), "dm": tests})
    for bw in HAC:
        family = [r["dm"][str(bw)] for r in summaries]
        if len(family) != FAMILY_SIZE: raise AssertionError("fixed Holm family size mismatch")
        direct_horizon._holm(family)
    strata = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            rows = [r for r in paired_rows if r["product"] == product and r["horizon"] == horizon]
            for name, key_fn in (("year", lambda d: d.year), ("season", lambda d: (d.month-1)//3+1)):
                groups = defaultdict(list)
                for row in rows: groups[key_fn(dt.date.fromisoformat(row["target"]))].append(row)
                for value, group in sorted(groups.items()):
                    ys = [r["actual"] for r in group]
                    strata.append({"product": product, "horizon": horizon, "stratum": name, "value": value,
                        "n": len(group), "forecast": _metrics(ys, [r["predicted"] for r in group]),
                        "persistence": _metrics(ys, [r["persistence"] for r in group])})
    conclusions = []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            req = [r for r in summaries if r["product"] == product and r["horizon"] == horizon]
            enough = all(r["n"] >= 30 and r["dm"]["9"]["status"] == "tested" for r in req)
            lower = all(r["forecastMetric"] is not None and r["persistenceMetric"] is not None and r["forecastMetric"] < r["persistenceMetric"] for r in req)
            sig = enough and all(r["dm"]["9"]["holmSignificant"] and r["dm"]["9"]["meanLossDifference"] is not None and r["dm"]["9"]["meanLossDifference"] < 0 for r in req)
            # Huber and its same-file persistence baseline share issuance eligibility by design.
            coverage_ok = all(r["n"] >= counts[f"{product}/{horizon}"].get("baselineCommonOpportunity", 0) for r in req)
            outcome = "improvement" if enough and lower and sig and coverage_ok else "inference_unavailable" if not enough else "worsening_or_mixed" if any(r["forecastMetric"] is not None and r["persistenceMetric"] is not None and r["forecastMetric"] > r["persistenceMetric"] for r in req) else "inconclusive"
            conclusions.append({"product": product, "horizon": horizon, "conclusion": outcome,
                "inferenceAvailable": enough, "bothLossesImprove": lower, "bothHac9HolmTestsImprove": sig,
                "coverageNondeclining": coverage_ok, "commonPairs": req[0]["n"]})
    return {"protocol": {"id": "prospective-live-pilot-2026-10-06", "issueWindowKst": ["10:00", "12:00"],
        "selection": "earliest valid artifact within window per KST date", "evaluationRange": [start.isoformat(), end.isoformat()],
        "maturityRule": "targetDate <= snapshot meta.evaluationEnd AND selected provenance has both validated market scopes",
        "horizons": list(HORIZONS),
        "minimumPairsForInference": 30, "comparators": list(COMPARATORS), "losses": list(LOSSES),
        "holmFamilySizePerBandwidth": FAMILY_SIZE, "bandwidthsCalendarDays": list(HAC), "alpha": 0.05,
        "limitation": "Snapshot prices may include later revisions; first publication time is unavailable, so this evaluates prospective issuance records against retrospective validated actuals."},
        "inputs": {"priceSnapshotSha256": snapshot_sha256, "selectedProvenanceByteSha256": provenance_byte_sha256,
            "selectedProvenanceCanonicalSha256": provenance_sha256, "validatedDataEnd": data_end.isoformat(),
            "validatedEvaluationEnd": evaluation_end.isoformat()},
        "dailyIssuance": daily_audit, "coverage": counts, "exclusionCounts": exclusions,
        "issueRecords": [{"date": day.isoformat(), "sha256": row["hash"], "path": row["path"],
            "issueAtKst": row["issuedAt"].isoformat(), "sourceCutoffDate": row["sourceCutoffDate"],
            "products": {pid: {"status": v["status"], "reason": v["reason"], "sourceDate": v["sourceDate"],
                "lastObservedPriceDate": v["lastObservedPriceDate"], "lastObservedPrice": v["lastObservedPrice"],
                "qualityGuard": v["qualityGuard"], "predictions": [{"horizonDays": h,
                    "target": p["target"].isoformat(), "predictedPrice": p["predicted"], "sourceDate": p["sourceDate"]}
                    for h, p in sorted(v["predictions"].items())]}
                         for pid, v in row["products"].items()}} for day, row in sorted(selected.items())],
        "pairedRows": paired_rows, "comparisons": summaries, "strata": strata, "jointConclusions": conclusions}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecasts", type=Path, default=Path("outputs/live-forecasts"))
    parser.add_argument("--prices-snapshot", type=Path, required=True, help="explicit validated dashboard schema-v1 prices.json")
    parser.add_argument("--selected-provenance", type=Path, required=True,
                        help="selected-query provenance sidecar from the same daily snapshot run")
    parser.add_argument("--start-date", type=_date_arg, default=PILOT_START)
    parser.add_argument("--end-date", type=_date_arg, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    snapshot = args.prices_snapshot.resolve()
    provenance_path = args.selected_provenance.resolve()
    if args.output.resolve() in (snapshot, provenance_path): parser.error("output must not replace an input")
    actuals, data_end, evaluation_end, snapshot_hash = _snapshot(snapshot)
    provenance, provenance_bytes_hash, provenance_canonical_hash = _provenance(
        provenance_path, selected_end=evaluation_end)
    files = _forecast_files(args.forecasts)
    if any(p.resolve() == args.output.resolve() for p in files): parser.error("output must not replace a forecast")
    if any(p.resolve() in (snapshot, provenance_path) for p in files):
        parser.error("an input file was discovered as a forecast")
    result = build(files, actuals, data_end, evaluation_end, snapshot_hash, provenance,
        provenance_canonical_hash, provenance_bytes_hash, args.start_date, args.end_date)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(args.output)
    print(json.dumps({"output": str(args.output), "snapshotSha256": snapshot_hash,
        "selectedIssues": len(result["issueRecords"]), "conclusions": result["jointConclusions"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
