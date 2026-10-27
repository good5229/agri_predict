"""Frozen offline KMA-forecast residual correction candidate evaluation."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import dashboard, direct_horizon as dh
from .kma_historical_forecast import ForecastArchive, parse_kma_forecast_zip

ROOT = Path(__file__).resolve().parents[1]
INPUT_PRICES = dh.INPUT_PRICES
INPUT_REPORT = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
RAW_DIR = ROOT / "data/raw/kma-forecasts"
ARCHIVES = (
    RAW_DIR / "seoul-20240101-20241230-request1.zip",
    RAW_DIR / "seoul-20250101-20251231-request1.zip",
    RAW_DIR / "seoul-20260101-20261006-request1.zip",
)
PRICE_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
REPORT_SHA256 = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
ARCHIVE_SHA256 = (
    "7364b2ff4fd9a6ef68a1e8f29dbb7a52884fb6abccc5e84e85cc3226b80d7a2c",
    "b05c1711f5e4066a991db82be866927dd0374b00e4d1f1b4566d22d924d8b70e",
    "9f5a2fcf8339bf1a385e367c93319b6047944e72fa34d3488f7887ab57f56e4e",
)
PRODUCTS = tuple(dashboard.PRODUCTS)
HORIZONS = tuple(range(1, 11))
FROZEN_HORIZONS = tuple(range(1, 11))
TRAIN_START, TRAIN_END = dt.date(2024, 1, 1), dt.date(2025, 9, 1)
EVAL_START, EVAL_END = dt.date(2025, 9, 25), dt.date(2026, 9, 24)
MIN_TRAIN_DAYS = 90
BANDWIDTHS = (9, 28)
FAMILY_SIZE = len(PRODUCTS) * len(HORIZONS) * 2 * 2
KST = dt.timezone(dt.timedelta(hours=9))
FEATURE_HOURS = (6, 12, 18, 6, 12, 18)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _day(value: Any) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid ISO date: {value!r}") from exc


def _positive(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def parse_archives(archive_inputs: list[tuple[str, bytes]]) -> tuple[list[ForecastArchive], dict[str, str]]:
    if len(archive_inputs) != 3:
        raise ValueError("exactly the three frozen KMA archives are required")
    parsed = []
    hashes = {}
    for (name, data), expected in zip(archive_inputs, ARCHIVE_SHA256):
        archive = parse_kma_forecast_zip(data, name, expected_location="seoul", expected_location_code="60_127", cutoff_hour_kst=10)
        if archive.archive_sha256 != expected:
            raise ValueError(f"KMA archive hash mismatch: {name}")
        if archive.location_code != "60_127":
            raise ValueError(f"unexpected KMA grid in {name}")
        if archive.quarantined_rows:
            # Preserve counts in the report, but these records never enter feature maps.
            pass
        parsed.append(archive)
        hashes[name] = archive.archive_sha256
    return parsed, hashes


def _weather_maps(archives: list[ForecastArchive]):
    values: dict[tuple[dt.datetime, dt.datetime], list[tuple[float, str, str]]] = defaultdict(list)
    for archive in archives:
        for row in archive.records:
            values[(row.issue_kst, row.valid_kst)].append((row.value, archive.archive_sha256, row.value_source))
    return values


def _features(origin: dt.date, weather: dict, quarantined_issue_dates=(), missing_issue_dates=()) -> tuple[list[float] | None, dict[str, Any]]:
    issue = dt.datetime.combine(origin, dt.time(8), KST)
    issue_utc_date = issue.astimezone(dt.timezone.utc).date()
    origin_time = dt.datetime.combine(origin, dt.time(10), KST)
    if issue >= origin_time:
        return None, {"status": "issue_not_before_origin", "expectedIssueKst": issue.isoformat(), "selected": []}
    selected = []
    vals = []
    for offset in (1, 2):
        valid_day = origin + dt.timedelta(days=offset)
        for hour in FEATURE_HOURS[(offset - 1) * 3:offset * 3]:
            valid = dt.datetime.combine(valid_day, dt.time(hour), KST)
            matches = weather.get((issue, valid), [])
            unique = {entry[0] for entry in matches}
            archives = sorted({entry[1] for entry in matches})
            source_values = sorted({entry[2] if len(entry) > 2 else str(entry[0]) for entry in matches})
            selected.append({"validKst": valid.isoformat(), "values": sorted(unique), "sourceValues": source_values,
                             "archiveSha256": archives})
            if not matches:
                status = ("quarantined_rows_present_but_forecast_missing" if issue_utc_date in quarantined_issue_dates
                          else "no_issue_records" if issue_utc_date in missing_issue_dates else "missing_forecast")
                audit = {"status": status, "expectedIssueKst": issue.isoformat(), "selected": selected}
                if any(weather.get((issue, dt.datetime.fromisoformat(item["validKst"])), []) for item in selected):
                    audit["observedIssueKst"] = issue.isoformat()
                return None, audit
            if len(unique) != 1:
                return None, {"status": "conflicting_forecast", "issueKst": issue.isoformat(), "selected": selected}
            vals.append(next(iter(unique)))
    return vals, {"status": "available", "issueKst": issue.isoformat(), "selected": selected}


def _observations(prices: dict[str, Any]):
    if prices.get("schemaVersion") != 1 or not isinstance(prices.get("products"), list):
        raise ValueError("prices must use dashboard schemaVersion 1")
    result = {}
    for product in prices["products"]:
        pid = product.get("id")
        if pid not in PRODUCTS or pid in result:
            raise ValueError("unknown or duplicate price product")
        series = {}
        for row in product.get("observations", []):
            day = _day(row.get("date"))
            actual = _positive(row.get("actual"), "actual price")
            if day in series:
                raise ValueError(f"duplicate price date for {pid}/{day}")
            series[day] = actual
        result[pid] = series
    if set(result) != set(PRODUCTS):
        raise ValueError("prices must contain all five products")
    return result


def _direct_rows(report: dict[str, Any], prices: dict[str, Any]):
    if report.get("protocol", {}).get("id") != "direct-horizon-ridge-2026-10-06":
        raise ValueError("unexpected direct-horizon protocol")
    series = _observations(prices)
    rows = {}
    for row in report.get("pairedRows", []):
        pid, horizon, target = row.get("product"), row.get("horizon"), _day(row.get("target"))
        if pid not in PRODUCTS or isinstance(horizon, bool) or horizon not in HORIZONS:
            raise ValueError("invalid product or horizon in direct report")
        origin = target - dt.timedelta(days=horizon)
        if row.get("origin") is not None and _day(row.get("origin")) != origin:
            raise ValueError(f"origin/target/horizon mismatch: {pid}/{target}/h{horizon}")
        key = (pid, horizon, target)
        if key in rows:
            raise ValueError(f"duplicate direct report row: {key}")
        actual = _positive(row.get("actual"), "report actual")
        if target not in series[pid] or not math.isclose(series[pid][target], actual, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"report actual does not align with frozen prices: {key}")
        direct = _positive(row.get("direct"), "direct prediction")
        persistence = _positive(row.get("persistence"), "persistence prediction")
        source_day = max((day for day in series[pid] if day <= origin), default=None)
        if source_day is None or (origin - source_day).days > dashboard.MAX_SOURCE_AGE_DAYS:
            raise ValueError(f"report row lacks an eligible persistence source: {key}")
        if not math.isclose(persistence, series[pid][source_day], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"report persistence does not align with frozen prices: {key}")
        rows[key] = {"product": pid, "horizon": horizon, "target": target, "origin": origin,
                     "actual": actual, "direct": direct, "persistence": persistence}
    if not rows:
        raise ValueError("direct report has no paired rows")
    return rows


def _dm(diff: dict[dt.date, float], bandwidth: int):
    n = len(diff)
    if n < 30:
        return {"status": "insufficient_pairs", "n": n, "meanLossDifference": None, "statistic": None, "pValue": 1.0}
    mean = sum(diff.values()) / n
    centered = {d: v - mean for d, v in diff.items()}
    variance = sum(v * v for v in centered.values())
    for lag in range(1, bandwidth + 1):
        cross = sum(centered[d] * centered.get(d + dt.timedelta(days=lag), 0.0) for d in centered)
        variance += 2 * (1 - lag / (bandwidth + 1)) * cross
    variance /= n * n
    if not math.isfinite(variance) or variance <= 0:
        return {"status": "zero_or_invalid_hac_variance", "n": n, "meanLossDifference": mean, "statistic": None, "pValue": 1.0}
    stat = mean / math.sqrt(variance)
    return {"status": "tested", "n": n, "meanLossDifference": mean, "statistic": stat,
            "pValue": math.erfc(abs(stat) / math.sqrt(2))}


def _holm(items):
    order = sorted(range(len(items)), key=lambda i: items[i]["pValue"])
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(items) - rank) * items[idx]["pValue"]))
        items[idx]["holmPValue"] = running
        items[idx]["holmSignificant"] = running < .05


def _metrics(rows, pred_field):
    if not rows:
        return {"mae": None, "mape": None}
    return {"mae": sum(abs(r["actual"] - r[pred_field]) for r in rows) / len(rows),
            "mape": 100 * sum(abs(r["actual"] - r[pred_field]) / r["actual"] for r in rows) / len(rows)}


def _point_estimate_goal_gate(grouped, products):
    equal_horizon_mape = {}
    for product in products:
        by_horizon = {h: grouped.get(f"{product}/{h}", []) for h in FROZEN_HORIZONS}
        values = {}
        for model, field in (("candidate", "candidate"), ("direct", "direct"), ("persistence", "persistence")):
            horizon_mapes = [_metrics(by_horizon[h], field)["mape"] for h in FROZEN_HORIZONS]
            values[model] = (sum(horizon_mapes) / len(FROZEN_HORIZONS)
                             if all(v is not None for v in horizon_mapes) else None)
        equal_horizon_mape[product] = values
    under10 = sum(v["candidate"] is not None and v["candidate"] < 10 for v in equal_horizon_mape.values())
    under7 = sum(v["candidate"] is not None and v["candidate"] < 7 for v in equal_horizon_mape.values())
    return {"aggregation": "unweighted mean of ten horizon-specific MAPE values; all ten must be available",
            "under10PercentProductCount": under10, "under7PercentProductCount": under7,
            "goalMet": under10 >= 3 or under7 >= 2,
            "equalHorizonMapeByProduct": equal_horizon_mape}


def build(prices: dict[str, Any], report: dict[str, Any], archives: list[ForecastArchive], *, hashes=None):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    direct = _direct_rows(report, prices)
    weather = _weather_maps(archives)
    quarantined_utc_dates = {dt.date.fromisoformat(day) for a in archives
                         for day in a.coverage_evidence.get("quarantined_marker_dates", [])}
    missing_issue_utc_dates = {dt.date.fromisoformat(day) for a in archives
                           for day in a.coverage_evidence.get("missing_issue_dates", [])}
    feature_cache = {}
    for row in direct.values():
        feature_cache[row["origin"]] = _features(row["origin"], weather, quarantined_utc_dates, missing_issue_utc_dates)
    predictions, fit_audits = [], {}
    fitted = {}
    for product in PRODUCTS:
        for horizon in HORIZONS:
            key = f"{product}/{horizon}"
            candidates = [r for r in direct.values() if r["product"] == product and r["horizon"] == horizon
                          and TRAIN_START <= r["target"] <= TRAIN_END]
            train = [r for r in candidates if feature_cache[r["origin"]][0] is not None]
            train_days = len({r["target"] for r in train})
            audit = {"pairedTrainingRows": len(candidates), "weatherAvailableTrainingRows": len(train),
                     "distinctTrainingTargets": train_days, "status": "insufficient_training"}
            if train_days >= MIN_TRAIN_DAYS:
                X = np.asarray([feature_cache[r["origin"]][0] for r in train], dtype=float)
                y = np.asarray([math.log(r["actual"] / r["direct"]) for r in train], dtype=float)
                scaler = StandardScaler()
                Xs = scaler.fit_transform(X)
                scaler.scale_[scaler.scale_ == 0] = 1.0
                # StandardScaler already emits zero for constant features; retain scale=1 in provenance.
                model = Ridge(alpha=100, fit_intercept=False)
                model.fit(Xs, y)
                fitted[key] = (scaler, model)
                audit["status"] = "fitted"
                audit["trainTargetStart"] = min(r["target"] for r in train).isoformat()
                audit["trainTargetEnd"] = max(r["target"] for r in train).isoformat()
            fit_audits[key] = audit
    eligible = [r for r in direct.values() if EVAL_START <= r["target"] <= EVAL_END]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        key = f"{row['product']}/{row['horizon']}"
        vec, selection = feature_cache[row["origin"]]
        candidate, fallback = row["direct"], "model_unavailable"
        if key in fitted and vec is not None:
            scaler, model = fitted[key]
            pred_residual = float(model.predict(scaler.transform(np.asarray([vec], dtype=float)))[0])
            adjusted = row["direct"] * math.exp(max(-.2, min(.2, pred_residual)))
            if math.isfinite(adjusted) and adjusted > 0:
                candidate, fallback = adjusted, None
            else:
                fallback = "nonfinite_or_nonpositive_candidate"
        elif vec is None:
            fallback = selection["status"]
        out = {**row, "candidate": candidate, "weather": selection, "fallbackReason": fallback,
               "archiveSha256": [a.archive_sha256 for a in archives]}
        predictions.append(out)
        grouped[key].append(out)

    summaries, joint, strata = [], [], []
    for product in PRODUCTS:
        for horizon in HORIZONS:
            key = f"{product}/{horizon}"
            rows = grouped[key]
            summaries_for_key = []
            for comparator in ("direct", "persistence"):
                for loss in ("ae", "ape"):
                    diff = {}
                    for r in rows:
                        fn = (lambda p: abs(r["actual"] - p)) if loss == "ae" else (lambda p: 100 * abs(r["actual"] - p) / r["actual"])
                        diff[r["target"]] = fn(r["candidate"]) - fn(r[comparator])
                    record = {"product": product, "horizon": horizon, "comparator": comparator, "loss": loss,
                              "n": len(rows), "candidateMetric": _metrics(rows, "candidate")["mae" if loss == "ae" else "mape"],
                              "comparatorMetric": _metrics(rows, comparator)["mae" if loss == "ae" else "mape"],
                              "dm": {str(b): _dm(diff, b) for b in BANDWIDTHS}}
                    summaries.append(record); summaries_for_key.append(record)
            for bw in BANDWIDTHS:
                _holm([r["dm"][str(bw)] for r in summaries if r["dm"][str(bw)] is not None])
            # strata by target year and meteorological quarter, descriptive only
            for field, fn in (("year", lambda d: d.year), ("season", lambda d: (d.month - 1) // 3 + 1)):
                groups = defaultdict(list)
                for r in rows: groups[fn(r["target"])].append(r)
                for value, part in sorted(groups.items()):
                    strata.append({"product": product, "horizon": horizon, "stratum": field, "value": value,
                                   "n": len(part), "candidate": _metrics(part, "candidate"),
                                   "direct": _metrics(part, "direct"), "persistence": _metrics(part, "persistence")})
    for bw in BANDWIDTHS:
        family = [item["dm"][str(bw)] for item in summaries]
        if len(family) != FAMILY_SIZE: raise AssertionError("fixed 200-test family changed")
        # Per-bandwidth Holm must span all 200 tests; reapply globally after creating rows.
        _holm(family)
    for product in PRODUCTS:
        for horizon in HORIZONS:
            key = f"{product}/{horizon}"
            rows, block = grouped[key], [r for r in summaries if r["product"] == product and r["horizon"] == horizon]
            tests = [r["dm"]["9"] for r in block]
            lower = all(r["candidateMetric"] is not None and r["candidateMetric"] < r["comparatorMetric"] for r in block)
            sig = all(t["holmSignificant"] and t["meanLossDifference"] < 0 for t in tests)
            enough = all(t["status"] == "tested" for t in tests)
            adverse = any(t["holmSignificant"] and t["meanLossDifference"] > 0 for t in tests)
            coverage_nondeclining = len(rows) == len([r for r in eligible if r["product"] == product and r["horizon"] == horizon])
            joint.append({"product": product, "horizon": horizon,
                          "conclusion": "inference_unavailable" if not enough else "improvement" if lower and sig and coverage_nondeclining else "worsening_or_mixed" if adverse else "inconclusive",
                          "allMetricsLower": lower, "allHac9HolmTestsImprove": sig, "coverageNondeclining": coverage_nondeclining,
                          "inferenceAvailable": enough, "eligibleRows": len(rows),
                          "weatherAvailableRows": sum(r["weather"]["status"] == "available" for r in rows),
                          "fallbackRows": sum(r["fallbackReason"] is not None for r in rows)})
    goal_gate = _point_estimate_goal_gate(grouped, PRODUCTS)
    return {"protocol": {"id": "kma-08kst-residual-ridge-2026-10-07", "trainTargets": [TRAIN_START.isoformat(), TRAIN_END.isoformat()],
             "evaluationTargets": [EVAL_START.isoformat(), EVAL_END.isoformat()], "issueKst": "08:00", "originKst": "10:00",
             "featureValidHours": ["D+1 06:00", "D+1 12:00", "D+1 18:00", "D+2 06:00", "D+2 12:00", "D+2 18:00"],
             "model": {"pipeline": ["StandardScaler", "Ridge"], "alpha": 100, "fit_intercept": False, "clipLogCorrection": [-.2, .2]},
             "minimumDistinctTrainingTargets": MIN_TRAIN_DAYS, "dmFamilySize": FAMILY_SIZE, "bandwidthsCalendarDays": list(BANDWIDTHS),
             "limitations": "Retrospective revised price and forecast files; original publication time is unproven; Seoul grid is not a crop-region proxy."},
            "hashes": hashes or {}, "archiveAudits": [{"archiveName": a.archive_name, "archiveSha256": a.archive_sha256,
              "memberName": a.member_name, "memberSha256": a.member_sha256, "parsedRecords": len(a.records),
              "exactDuplicateCount": a.exact_duplicate_count, "quarantinedRows": len(a.quarantined_rows),
              "coverageEvidence": a.coverage_evidence} for a in archives],
            "fitAudits": fit_audits, "coverage": {k: {"eligibleDirectRows": len(v),
              "weatherAvailableRows": sum(r["weather"]["status"] == "available" for r in v),
              "fallbackRows": sum(r["fallbackReason"] is not None for r in v),
            "fallbackReasons": dict(__import__("collections").Counter(r["fallbackReason"] for r in v if r["fallbackReason"]))}
              for k, v in grouped.items()}, "predictions": [
                {k: (v.isoformat() if isinstance(v, dt.date) else v) for k, v in r.items()} for r in predictions],
            "comparisons": summaries, "jointConclusions": joint, "pointEstimateGoalGate": goal_gate, "strata": strata}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prices", type=Path, default=INPUT_PRICES)
    p.add_argument("--report", type=Path, default=INPUT_REPORT)
    p.add_argument("--archive", type=Path, action="append", help="repeat exactly three times in frozen chronological order")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    archives_paths = [Path(x) for x in a.archive] if a.archive else list(ARCHIVES)
    if len(archives_paths) != 3: p.error("exactly three --archive paths are required")
    price_bytes, report_bytes = a.prices.read_bytes(), a.report.read_bytes()
    input_archives = [(x.name, x.read_bytes()) for x in archives_paths]
    hashes = {"pricesSha256": sha256(price_bytes), "directReportSha256": sha256(report_bytes)}
    if hashes["pricesSha256"] != PRICE_SHA256 or hashes["directReportSha256"] != REPORT_SHA256:
        p.error("price or direct report hash does not match the frozen protocol")
    parsed, archive_hashes = parse_archives(input_archives)
    hashes["kmaArchives"] = archive_hashes
    result = build(json.loads(price_bytes), json.loads(report_bytes), parsed, hashes=hashes)
    if a.output.resolve() in [a.prices.resolve(), a.report.resolve(), *(x.resolve() for x in archives_paths)]:
        p.error("output must not replace an input")
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(a.output), "hashes": hashes, "jointConclusions": result["jointConclusions"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
