"""Frozen national-wholesale archive target-transfer evaluation.

The archive reader is offline and fail-closed. It never fetches data, opens a
database, trains a model, deduplicates source rows, or fills missing dates.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path
import uuid
import zipfile

from . import dashboard, direct_horizon as dh, market_recompose

ROOT = Path(__file__).resolve().parents[1]
INPUT_ARCHIVE = ROOT / "data/raw/national-wholesale-20240708-20260630.zip"
INPUT_RECOMPOSE = ROOT / "outputs/experiments/market-recompose-20261007/report.json"
INPUT_PAGE_REFERENCE = ROOT / "data/raw/nongnet-public-reference-20261007.json"
OUTPUT_DIR = ROOT / "outputs/experiments/national-transfer-20261007"
ARCHIVE_SHA256 = "b9a38d4a62477054758e7576a7320131bc5acc8129a4079a33977f9ed876d7d7"
RECOMPOSE_SHA256 = "e5f23f1012dad2acc774c83f995e2ee2851409af913ee82483aa3d953ee87581"
PAGE_REFERENCE_SHA256 = "c3f2d457d2065b32b666ac7d1a38b9e080caf6269d9030b5f5a10d6583d0e9a4"
EVAL_START = dt.date(2024, 7, 8)
EVAL_END = dt.date(2026, 6, 30)
HORIZONS = tuple(range(1, 11))
PRODUCT_CODES = {"cabbage": "1001", "radish": "1101", "onion": "1201", "pepper": "1207"}
PRODUCT_NAMES = {"cabbage": "배추", "radish": "무", "onion": "양파", "pepper": "건고추"}
EXPECTED_HEADERS = ("경매일자", "도매시장코드", "도매시장명", "품목코드", "품목명", "변경총물량", "경매낙찰총금액")
PERIODS = (
    (dt.date(2024, 7, 8), dt.date(2024, 12, 31)),
    (dt.date(2025, 1, 1), dt.date(2025, 6, 30)),
    (dt.date(2025, 7, 1), dt.date(2025, 12, 31)),
    (dt.date(2026, 1, 1), dt.date(2026, 6, 30)),
)
PAGE_CHECKS = ((dt.date(2025, 7, 15), "cabbage"), (dt.date(2025, 7, 15), "onion"),
               (dt.date(2026, 6, 30), "cabbage"), (dt.date(2026, 6, 30), "onion"))
MIN_PAIRS = 30
HAC = (9, 28)
ALPHA = 0.05
FAMILY_SIZE = 4 * 10 * 2 * 2
PROTOCOL = {
    "id": "national-wholesale-target-transfer-2026-10-07",
    "archiveSha256": ARCHIVE_SHA256,
    "archiveRows": "all markets and rows; signed Decimal amount and quantity; no deduplication or forward fill",
    "targets": {code: product for product, code in PRODUCT_CODES.items()},
    "targetPrice": "sum signed auction amount / sum signed quantity, only when both sums are complete and quantity is positive",
    "evaluation": [EVAL_START.isoformat(), EVAL_END.isoformat()],
    "pairing": "same product, origin, target, and calendar horizon from pinned market-recompose report",
    "comparators": ["directRidge", "persistence"],
    "comparisonFamily": "4 products x 10 horizons x 2 comparators x 2 losses = 160 two-sided DM tests per bandwidth; calendar Bartlett HAC9/HAC28; separate Holm alpha .05; unusable p=1",
    "minimumPairs": MIN_PAIRS,
    "publicPageValidation": [{"date": day.isoformat(), "product": product} for day, product in PAGE_CHECKS],
    "limitations": "retrospective revised archive; first-publication time is unknown; this is not official contest-score evidence",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _parse_day(value: str) -> dt.date:
    raw = value.strip()
    try:
        if len(raw) == 8 and raw.isdigit():
            return dt.datetime.strptime(raw, "%Y%m%d").date()
        return dt.date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"invalid auction date: {value!r}") from exc


def _decimal(value: str, label: str) -> Decimal | None:
    raw = value.strip()
    if not raw:
        return None
    try:
        result = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"non-numeric {label}: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"non-finite {label}: {value!r}")
    return result


def _period_index(day: dt.date) -> int:
    for i, (start, end) in enumerate(PERIODS):
        if start <= day <= end:
            return i
    raise ValueError(f"archive date outside frozen range: {day}")


def parse_archive(data: bytes, *, expected_sha256: str | None = ARCHIVE_SHA256) -> dict:
    """Validate and aggregate synthetic or pinned archive bytes without repairs."""
    digest = _sha(data)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("national archive SHA-256 differs from frozen protocol")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError("national archive is not a valid ZIP") from exc
    with archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        if len(names) != 4 or len(set(names)) != 4:
            raise ValueError("archive must contain exactly four uniquely named CSV files")
        for info in infos:
            path = Path(info.filename)
            if info.is_dir() or path.is_absolute() or ".." in path.parts or "\\" in info.filename or path.suffix.lower() != ".csv":
                raise ValueError(f"unsafe or unexpected archive member: {info.filename!r}")
            mode = (info.external_attr >> 16) & 0o170000
            if mode not in (0, 0o100000):
                raise ValueError(f"non-regular archive member: {info.filename!r}")

        period_sources: dict[int, dict] = {}
        aggregates: dict[tuple[str, dt.date], dict] = {}
        duplicate_counter: dict[tuple[str, ...], int] = {}
        total_rows = 0
        for info in infos:
            raw_bytes = archive.read(info)
            try:
                text = raw_bytes.decode("cp949", errors="strict")
            except UnicodeDecodeError as exc:
                raise ValueError(f"archive CSV is not valid CP949: {info.filename}") from exc
            reader = csv.DictReader(io.StringIO(text, newline=""))
            if tuple(reader.fieldnames or ()) != EXPECTED_HEADERS:
                raise ValueError(f"archive CSV schema changed: {info.filename}")
            rows = list(reader)
            if not rows:
                raise ValueError(f"archive CSV is empty and has no verifiable period: {info.filename}")
            row_days = []
            source_row_count = 0
            for line_no, row in enumerate(rows, start=2):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"malformed CSV row width: {info.filename}:{line_no}")
                day = _parse_day(row["경매일자"])
                pi = _period_index(day)
                row_days.append((day, pi))
                market_code = row["도매시장코드"].strip()
                market_name = row["도매시장명"].strip()
                product_code = row["품목코드"].strip()
                product_name = row["품목명"].strip()
                quantity = _decimal(row["변경총물량"], f"quantity {info.filename}:{line_no}")
                amount = _decimal(row["경매낙찰총금액"], f"amount {info.filename}:{line_no}")
                if not market_code or not market_name or not product_code or not product_name:
                    raise ValueError(f"blank market/product identity: {info.filename}:{line_no}")
                raw_key = tuple(row[h] for h in EXPECTED_HEADERS)
                duplicate_counter[raw_key] = duplicate_counter.get(raw_key, 0) + 1
                source_row_count += 1
                total_rows += 1
                product = next((p for p, code in PRODUCT_CODES.items() if product_code == code), None)
                if product is None:
                    continue
                if product_name != PRODUCT_NAMES[product]:
                    # Preserve the name conflict in the audit, while refusing to
                    # assign a different named commodity to the frozen code.
                    name_conflict = True
                else:
                    name_conflict = False
                key = (product, day)
                agg = aggregates.setdefault(key, {"amount": Decimal(0), "quantity": Decimal(0), "rowCount": 0,
                    "nullAmountCount": 0, "nullQuantityCount": 0, "negativeAmountCount": 0,
                    "negativeQuantityCount": 0, "duplicateRowCount": 0, "markets": set(),
                    "marketNames": set(), "productNames": set(), "nameConflictCount": 0, "complete": True})
                agg["rowCount"] += 1
                if duplicate_counter[raw_key] > 1:
                    agg["duplicateRowCount"] += 1
                agg["markets"].add(market_code)
                agg["marketNames"].add(market_name)
                agg["productNames"].add(product_name)
                agg["nameConflictCount"] += int(name_conflict)
                if name_conflict:
                    agg["complete"] = False
                if amount is None:
                    agg["nullAmountCount"] += 1
                    agg["complete"] = False
                else:
                    agg["amount"] += amount
                    agg["negativeAmountCount"] += int(amount < 0)
                if quantity is None:
                    agg["nullQuantityCount"] += 1
                    agg["complete"] = False
                else:
                    agg["quantity"] += quantity
                    agg["negativeQuantityCount"] += int(quantity < 0)
            periods = {pi for _, pi in row_days}
            if len(periods) != 1:
                raise ValueError(f"CSV spans multiple frozen periods: {info.filename}")
            pi = next(iter(periods))
            if pi in period_sources:
                raise ValueError(f"duplicate archive period: {info.filename}")
            period_sources[pi] = {"name": info.filename, "sha256": _sha(raw_bytes), "byteCount": len(raw_bytes),
                                  "rowCount": source_row_count, "start": min(d for d, _ in row_days).isoformat(),
                                  "end": max(d for d, _ in row_days).isoformat()}
        missing = sorted(set(range(len(PERIODS))) - set(period_sources))
        if missing:
            raise ValueError(f"archive is missing frozen periods: {missing}")

        day_rows: dict[str, dict[str, dict]] = {p: {} for p in PRODUCT_CODES}
        for (product, day), agg in sorted(aggregates.items(), key=lambda item: (item[0][0], item[0][1])):
            positive = agg["complete"] and agg["quantity"] > 0
            price = agg["amount"] / agg["quantity"] if positive else None
            usable_price = price is not None and price > 0
            day_rows[product][day.isoformat()] = {
                "amountSum": str(agg["amount"]), "quantitySum": str(agg["quantity"]),
                "price": str(price) if price is not None else None,
                "priceReason": (None if usable_price else "product_name_conflict" if agg["nameConflictCount"] else
                                "null_components" if not agg["complete"] else
                                "nonpositive_signed_quantity" if price is None else "nonpositive_aggregate_price"),
                "rowCount": agg["rowCount"], "marketCount": len(agg["markets"]),
                "marketCodes": sorted(agg["markets"]), "marketNames": sorted(agg["marketNames"]),
                "productNames": sorted(agg["productNames"]), "nameConflictCount": agg["nameConflictCount"],
                "nullAmountCount": agg["nullAmountCount"], "nullQuantityCount": agg["nullQuantityCount"],
                "negativeAmountCount": agg["negativeAmountCount"], "negativeQuantityCount": agg["negativeQuantityCount"],
                "duplicateRowCount": agg["duplicateRowCount"],
            }
        duplicates = sum(max(0, count - 1) for count in duplicate_counter.values())
        return {"archiveSha256": digest, "archiveByteCount": len(data), "csvCount": len(infos),
                "sourceRowCount": total_rows, "exactDuplicateRowCount": duplicates,
                "periods": [period_sources[i] for i in range(len(PERIODS))], "observations": day_rows}


def _finite_positive(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be positive finite")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be positive finite") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be positive finite")
    return number


def _report_index(rows, *, kind):
    if not isinstance(rows, list):
        raise ValueError(f"market-recompose {kind} must be a list")
    indexed = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"market-recompose {kind} row {i} is invalid")
        product = row.get("product")
        if product in dashboard.PRODUCTS and product not in PRODUCT_CODES:
            continue
        if product not in PRODUCT_CODES:
            raise ValueError(f"market-recompose {kind} has an unknown product")
        horizon = row.get("horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
            raise ValueError(f"market-recompose {kind} horizon is invalid")
        origin = dt.date.fromisoformat(row["origin"])
        target = dt.date.fromisoformat(row["target"])
        if target - origin != dt.timedelta(days=horizon):
            raise ValueError(f"market-recompose {kind} origin/target/horizon mismatch")
        key = (product, origin, target, horizon)
        if key in indexed:
            raise ValueError(f"duplicate market-recompose {kind} key: {key}")
        indexed[key] = row
    return indexed


def _loss_diff(actual, candidate, comparator, loss):
    if loss == "mae":
        return abs(actual - candidate) - abs(actual - comparator)
    return 100.0 * (abs(actual - candidate) - abs(actual - comparator)) / actual


def _season(day):
    if day.month in (12, 1, 2):
        return "winter"
    if day.month in (3, 4, 5):
        return "spring"
    if day.month in (6, 7, 8):
        return "summer"
    return "autumn"


def validate_public_page_values(archive, values):
    """Compare explicitly supplied page integers; never fetch the page."""
    results = []
    for day, product in PAGE_CHECKS:
        raw = values.get(day.isoformat(), {}).get(product) if isinstance(values, dict) else None
        observation = archive.get("observations", {}).get(product, {}).get(day.isoformat())
        national_price = Decimal(observation["price"]) if observation and observation.get("price") is not None else None
        if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0 or national_price is None:
            results.append({"date": day.isoformat(), "product": product, "pageValue": raw,
                            "archivePrice": str(national_price) if national_price is not None else None,
                            "differenceWonPerKg": None, "status": "not_comparable_missing_page_value_or_archive_price"})
            continue
        diff = abs(national_price - Decimal(raw))
        results.append({"date": day.isoformat(), "product": product, "pageValue": raw,
                        "archivePrice": str(national_price), "differenceWonPerKg": str(diff),
                        "status": "within_one_won" if diff <= 1 else "discrepant"})
    all_supplied = all(isinstance(item["pageValue"], int) and not isinstance(item["pageValue"], bool) and item["pageValue"] > 0 for item in results)
    all_comparable = all(item["status"] in ("within_one_won", "discrepant") for item in results)
    aligned = all_supplied and all_comparable and all(item["status"] == "within_one_won" for item in results)
    return {"status": "aligned_with_public_page" if aligned else "unverified_or_discrepant",
            "pageFetched": False, "allFourObservedIntegersSupplied": all_supplied, "checks": results,
            "alignmentClaimAllowed": aligned}


def load_page_reference(data: bytes, *, expected_sha256: str | None = PAGE_REFERENCE_SHA256):
    """Validate the four manually supplied public-page observations and provenance."""
    digest = _sha(data)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("public-page reference SHA-256 differs from pinned capture")
    try:
        doc = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("public-page reference is invalid JSON") from exc
    selection = doc.get("selection") if isinstance(doc, dict) else None
    if (not isinstance(doc, dict) or doc.get("schemaVersion") != 1 or
            selection != {"market": "*전국도매시장", "variety": "전체", "priceType": "평균", "unit": "원/kg"}):
        raise ValueError("public-page reference schema/selection differs from frozen check")
    values = {}
    for row in doc.get("observations", []):
        if not isinstance(row, dict):
            raise ValueError("public-page observation is invalid")
        day, code, price = row.get("date"), row.get("cropCode"), row.get("displayPriceWonPerKg")
        product = next((p for p, c in PRODUCT_CODES.items() if c == code), None)
        if product is None or (dt.date.fromisoformat(day), product) not in PAGE_CHECKS:
            raise ValueError("public-page reference includes an unplanned date/product")
        if row.get("cropName") != PRODUCT_NAMES[product]:
            raise ValueError("public-page crop name/code mismatch")
        if isinstance(price, bool) or not isinstance(price, int) or price <= 0:
            raise ValueError("public-page value must be a positive integer won/kg")
        if day in values and product in values[day]:
            raise ValueError("duplicate public-page date/product observation")
        values.setdefault(day, {})[product] = price
    if {(dt.date.fromisoformat(day), product) for day, products in values.items() for product in products} != set(PAGE_CHECKS):
        raise ValueError("public-page reference must contain exactly the four frozen checks")
    return values, {"sha256": digest, "sourceUrl": doc.get("sourceUrl"), "capturedAtUtc": doc.get("capturedAtUtc"),
                    "selection": selection, "limitations": doc.get("limitations")}


def build_assessment(archive, report, *, archive_sha256=None, report_sha256=None, page_values=None, page_reference_provenance=None):
    """Pair national positive prices to frozen market-recompose origins."""
    if not isinstance(report, dict) or report.get("schemaVersion") != 1 or report.get("status") != "complete":
        raise ValueError("market-recompose report is not a complete schemaVersion 1 report")
    protocol = report.get("protocol")
    if not isinstance(protocol, dict) or protocol.get("id") != market_recompose.PROTOCOL.get("id"):
        raise ValueError("market-recompose report protocol differs from frozen source")
    evidence = _report_index(report.get("originEvidence"), kind="originEvidence")
    report_pairs = _report_index(report.get("pairedRows"), kind="pairedRows")
    observations = archive.get("observations")
    if not isinstance(observations, dict) or set(observations) != set(PRODUCT_CODES):
        raise ValueError("archive observations must contain exactly the four frozen target products")

    eligible_keys = {key for key in evidence if EVAL_START <= key[2] <= EVAL_END}
    actuals = {}
    for product, dates in observations.items():
        for day_text, row in dates.items():
            day = dt.date.fromisoformat(day_text)
            price_raw = row.get("price")
            if price_raw is None:
                continue
            try:
                price = float(price_raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"archive actual {product}/{day} is not numeric") from exc
            if not math.isfinite(price):
                raise ValueError(f"archive actual {product}/{day} is not finite")
            if price > 0:
                actuals[(product, day)] = price

    pair_rows = []
    reasons = {}
    method_issues = {product: {name: set() for name in ("candidate", "directRidge", "persistence")} for product in PRODUCT_CODES}
    eligible_by_group = {(p, h): 0 for p in PRODUCT_CODES for h in HORIZONS}
    for key in sorted(eligible_keys):
        product, origin, target, horizon = key
        eligible_by_group[(product, horizon)] += 1
        evidence_row = evidence[key]
        for name, field in (("candidate", "candidatePrediction"), ("directRidge", "baselinePrediction"), ("persistence", "persistencePrediction")):
            value = evidence_row.get(field)
            if value is not None:
                try:
                    _finite_positive(value, field)
                    method_issues[product][name].add(key)
                except ValueError:
                    pass
        national = actuals.get((product, target))
        if national is None:
            archive_row = observations.get(product, {}).get(target.isoformat())
            reason = ("national_target_unusable:" + str(archive_row.get("priceReason"))
                      if archive_row is not None else "national_target_missing_or_nonpositive")
        else:
            pair = report_pairs.get(key)
            if pair is None:
                reason = "missing_common_market_recompose_pair"
            else:
                try:
                    candidate = _finite_positive(pair.get("candidatePrediction"), "candidatePrediction")
                    ridge = _finite_positive(pair.get("directRidge"), "directRidge")
                    persistence = _finite_positive(pair.get("persistence"), "persistence")
                except ValueError:
                    reason = "candidate_or_comparator_unavailable_or_invalid"
                else:
                    two_market_actual = _finite_positive(pair.get("actual"), "two-market actual")
                    row = {"product": product, "origin": origin.isoformat(), "target": target.isoformat(),
                           "horizon": horizon, "nationalActual": national, "twoMarketActual": two_market_actual,
                           "candidatePrediction": candidate, "directRidge": ridge, "persistence": persistence,
                           "nationalMinusTwoMarket": national - two_market_actual,
                           "nationalAbsolutePercentDifference": 100.0 * abs(national - two_market_actual) / national}
                    pair_rows.append(row)
                    continue
        reasons[reason] = reasons.get(reason, 0) + 1

    comparisons = []
    for product in PRODUCT_CODES:
        for horizon in HORIZONS:
            rows = [r for r in pair_rows if r["product"] == product and r["horizon"] == horizon]
            for comparator, field in (("directRidge", "directRidge"), ("persistence", "persistence")):
                for loss in ("mae", "mape"):
                    diff = {dt.date.fromisoformat(row["target"]): _loss_diff(row["nationalActual"], row["candidatePrediction"], row[field], loss) for row in rows}
                    actual = [row["nationalActual"] for row in rows]
                    candidate = [row["candidatePrediction"] for row in rows]
                    base = [row[field] for row in rows]
                    metric_name = "mae" if loss == "mae" else "mape"
                    comparisons.append({"product": product, "horizon": horizon, "comparator": comparator, "loss": loss,
                        "n": len(rows), "candidateMetric": dh._metrics(actual, candidate)[metric_name],
                        "comparatorMetric": dh._metrics(actual, base)[metric_name],
                        "dm": {str(bw): dh._dm(diff, bw) for bw in HAC}})
    if len(comparisons) != FAMILY_SIZE:
        raise AssertionError(f"frozen national-transfer comparison family must contain {FAMILY_SIZE} tests")
    for bandwidth in HAC:
        dh._holm([row["dm"][str(bandwidth)] for row in comparisons])

    coverage = {}
    joint = []
    for product in PRODUCT_CODES:
        for horizon in HORIZONS:
            group = [r for r in pair_rows if r["product"] == product and r["horizon"] == horizon]
            method_sets = {name: values for name, values in method_issues[product].items()}
            available = {name: {key for key in values if key[3] == horizon and (key[0], key[2]) in actuals}
                         for name, values in method_sets.items()}
            losses = {name: sorted((list(key[:1]) + [key[1].isoformat(), key[2].isoformat(), key[3]])
                                   for key in (available[name] - available["candidate"]))
                      for name in ("directRidge", "persistence")}
            cov_ok = all(not dates for dates in losses.values())
            coverage[f"{product}/{horizon}"] = {"eligibleOriginsInArchiveWindow": eligible_by_group[(product, horizon)],
                "paired": len(group), "candidateIssuedWithPositiveTruth": len(available["candidate"]),
                "directRidgeIssuedWithPositiveTruth": len(available["directRidge"]),
                "persistenceIssuedWithPositiveTruth": len(available["persistence"]),
                "candidateCoverageLossKeys": losses}
            tests = [r for r in comparisons if r["product"] == product and r["horizon"] == horizon]
            enough = all(t["n"] >= MIN_PAIRS and t["dm"]["9"]["status"] == "tested" for t in tests)
            all_point_better = all(t["candidateMetric"] is not None and t["comparatorMetric"] is not None and t["candidateMetric"] < t["comparatorMetric"] for t in tests)
            all_inference_better = all(t["dm"]["9"].get("holmSignificant") and t["dm"]["9"].get("meanLossDifference") is not None and t["dm"]["9"]["meanLossDifference"] < 0 for t in tests)
            adverse = any(t["dm"]["9"].get("holmSignificant") and t["dm"]["9"].get("meanLossDifference", 0) > 0 for t in tests)
            conclusion = "inference_unavailable" if not enough else "improvement" if cov_ok and all_point_better and all_inference_better else "worsening_or_mixed" if adverse else "inconclusive"
            joint.append({"product": product, "horizon": horizon, "conclusion": conclusion,
                "inferenceAvailable": enough, "allFourPointMetricsImprove": all_point_better,
                "allFourHac9HolmTestsImprove": all_inference_better, "coverageNondeclining": cov_ok})

    macro = {}
    for product in PRODUCT_CODES:
        by_h = {h: [r for r in pair_rows if r["product"] == product and r["horizon"] == h] for h in HORIZONS}
        metrics = {str(h): dh._metrics([r["nationalActual"] for r in by_h[h]], [r["candidatePrediction"] for r in by_h[h]])["mape"] for h in HORIZONS}
        macro[product] = {"equalHorizonMape": (sum(metrics.values()) / len(HORIZONS) if all(v is not None for v in metrics.values()) else None),
                          "mapeByHorizon": metrics, "pairedCountsByHorizon": {str(h): len(by_h[h]) for h in HORIZONS},
                          "pointEstimateAvailable": all(v is not None for v in metrics.values()),
                          "interpretation": "descriptive_point_estimate_not_equivalence_or_success_evidence"}

    strata = []
    for product in PRODUCT_CODES:
        for horizon in HORIZONS:
            rows = [r for r in pair_rows if r["product"] == product and r["horizon"] == horizon]
            for kind, label_func in (("year", lambda d: str(d.year)), ("season", lambda d: _season(d))):
                buckets = {}
                for row in rows:
                    target = dt.date.fromisoformat(row["target"])
                    buckets.setdefault(label_func(target), []).append(row)
                for label, group in sorted(buckets.items()):
                    strata.append({"product": product, "horizon": horizon, "stratum": kind, "value": label, "n": len(group),
                        "candidate": dh._metrics([r["nationalActual"] for r in group], [r["candidatePrediction"] for r in group]),
                        "directRidge": dh._metrics([r["nationalActual"] for r in group], [r["directRidge"] for r in group]),
                        "persistence": dh._metrics([r["nationalActual"] for r in group], [r["persistence"] for r in group])})

    discrepancies = {}
    for product in PRODUCT_CODES:
        rows = [r for r in pair_rows if r["product"] == product]
        target_values = {}
        for row in rows:
            target_values[row["target"]] = (row["nationalActual"], row["twoMarketActual"])
        diffs = [national - local for national, local in target_values.values()]
        discrepancies[product] = {"pairedForecastRows": len(rows), "uniqueTargetDates": len(target_values),
            "meanNationalMinusTwoMarket": sum(diffs) / len(diffs) if diffs else None,
            "maePriceDifference": sum(abs(x) for x in diffs) / len(diffs) if diffs else None,
            "meanAbsolutePercentDifferenceVsNational": (sum(100.0 * abs(n - local) / n for n, local in target_values.values()) / len(target_values) if target_values else None)}

    page_report = validate_public_page_values(archive, page_values or {})
    if page_values and page_reference_provenance is None:
        page_reference_provenance = {"sha256": None, "sourceUrl": None, "capturedAtUtc": None,
                                     "selection": None, "limitations": ["explicit page values supplied without a captured source artifact"]}
        page_report["alignmentClaimAllowed"] = False
        page_report["status"] = "unattributed_explicit_values"
    return {"schemaVersion": 1, "status": "complete", "protocol": PROTOCOL,
        "hashes": {"archiveSha256": archive_sha256 or archive.get("archiveSha256"), "marketRecomposeSha256": report_sha256,
                   "publicPageReferenceSha256": page_reference_provenance.get("sha256") if page_reference_provenance else None},
        "archiveAudit": {k: archive[k] for k in ("archiveSha256", "archiveByteCount", "csvCount", "sourceRowCount", "exactDuplicateRowCount", "periods")},
        "targetAudit": archive["observations"],
        "publicPageAlignment": {**page_report, "reference": page_reference_provenance},
        "coverage": coverage, "exclusionReasonCounts": reasons, "pairedRows": pair_rows,
        "comparisons": comparisons, "jointConclusions": joint, "yearSeasonMetrics": strata,
        "equalHorizonMacroMape": macro, "nationalVsTwoMarketDiscrepancy": discrepancies,
        "limitations": PROTOCOL["limitations"]}


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_new(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def run(archive_path=INPUT_ARCHIVE, report_path=INPUT_RECOMPOSE, *, page_reference_path=INPUT_PAGE_REFERENCE, output_dir=OUTPUT_DIR):
    archive_bytes = Path(archive_path).read_bytes()
    report_bytes = Path(report_path).read_bytes()
    archive_sha, report_sha = _sha(archive_bytes), _sha(report_bytes)
    if archive_sha != ARCHIVE_SHA256:
        raise ValueError("national archive SHA-256 does not match frozen plan")
    if report_sha != RECOMPOSE_SHA256:
        raise ValueError("market-recompose report SHA-256 does not match frozen plan")
    archive = parse_archive(archive_bytes)
    try:
        report = json.loads(report_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("market-recompose report is invalid JSON") from exc
    page_values, page_provenance = load_page_reference(Path(page_reference_path).read_bytes()) if page_reference_path is not None else ({}, None)
    result = build_assessment(archive, report, archive_sha256=archive_sha, report_sha256=report_sha,
                              page_values=page_values, page_reference_provenance=page_provenance)
    output_path = Path(output_dir) / "report.json"
    encoded = _json_bytes(result)
    _write_new(output_path, encoded)
    return {"output": str(output_path), "sha256": _sha(encoded), "pairedRows": len(result["pairedRows"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=INPUT_ARCHIVE)
    parser.add_argument("--recompose", type=Path, default=INPUT_RECOMPOSE)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--page-reference-json", type=Path, default=INPUT_PAGE_REFERENCE,
                        help="pinned manually observed public-page values; never fetched by this tool")
    parser.add_argument("--no-page-reference", action="store_true", help="omit public-page validation")
    args = parser.parse_args(argv)
    result = run(args.archive, args.recompose,
                 page_reference_path=None if args.no_page_reference else args.page_reference_json,
                 output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
