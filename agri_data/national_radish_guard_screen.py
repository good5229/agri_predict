"""Offline input screen for the frozen national radish quality-guard proposal.

This screen reports archive observations and flags only the prespecified joint
volume-collapse/price-discontinuity condition. It does not fit or score a model.
"""
from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path

from . import national_transfer

ROOT = Path(__file__).resolve().parents[1]
INPUT_ARCHIVE = ROOT / "data/raw/national-wholesale-20240708-20260630.zip"
OUTPUT_PATH = ROOT / "outputs/experiments/national-radish-guard-screen-20261009/report.json"
PRODUCT = "radish"
PRODUCT_NAME = national_transfer.PRODUCT_NAMES[PRODUCT]
MIN_PRIOR_OBSERVATIONS = 14
PRIOR_OBSERVATION_COUNT = 28
VOLUME_RATIO_LIMIT = Decimal("0.01")
PRICE_RATIO_LIMIT = Decimal("4")
PROTOCOL_ID = "national-radish-quality-guard-input-screen-2026-10-09"


def _decimal(value, label: str) -> Decimal | None:
    if value is None:
        return None
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if not result.is_finite():
        raise ValueError(f"non-finite {label}")
    return result


def _eligible(row: dict) -> tuple[bool, str | None, Decimal | None, Decimal | None]:
    quantity = _decimal(row.get("quantitySum"), "quantitySum")
    price = _decimal(row.get("price"), "price")
    if row.get("productNames") != [PRODUCT_NAME] or row.get("nameConflictCount") != 0:
        return False, "product_name_not_exact", price, quantity
    if row.get("priceReason") is not None or price is None or price <= 0:
        return False, "price_incomplete_or_nonpositive", price, quantity
    if quantity is None or quantity <= 0:
        return False, "net_quantity_not_positive", price, quantity
    return True, None, price, quantity


def screen(parsed: dict) -> dict:
    observations = parsed.get("observations")
    if not isinstance(observations, dict) or set(observations) != set(national_transfer.PRODUCT_CODES):
        raise ValueError("parsed national archive has incomplete product coverage")
    raw_days = observations[PRODUCT]
    if not isinstance(raw_days, dict):
        raise ValueError("parsed radish observations are invalid")
    ordered = []
    for text, row in raw_days.items():
        try:
            day = dt.date.fromisoformat(text)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid radish observation date") from exc
        if not isinstance(row, dict):
            raise ValueError("invalid radish observation row")
        ordered.append((day, text, row))
    ordered.sort(key=lambda item: item[0])
    if len({day for day, _, _ in ordered}) != len(ordered):
        raise ValueError("duplicate radish observation date")

    eligible_history: list[dict] = []
    daily = []
    flagged = []
    anomalous = []
    for day, text, row in ordered:
        ok, reason, price, quantity = _eligible(row)
        prior = eligible_history[-PRIOR_OBSERVATION_COUNT:]
        prior_median_price = prior_median_quantity = None
        volume_ratio = price_ratio = None
        has_window = ok and len(prior) >= MIN_PRIOR_OBSERVATIONS
        is_flag = False
        if has_window:
            sorted_prices = sorted(item["_price"] for item in prior)
            sorted_quantities = sorted(item["_quantity"] for item in prior)
            mid = len(prior) // 2
            prior_median_price = (sorted_prices[mid] if len(prior) % 2 else
                                  (sorted_prices[mid - 1] + sorted_prices[mid]) / 2)
            prior_median_quantity = (sorted_quantities[mid] if len(prior) % 2 else
                                     (sorted_quantities[mid - 1] + sorted_quantities[mid]) / 2)
            # Current eligible quantity and price are positive by construction.
            volume_ratio = quantity / prior_median_quantity
            price_ratio = price / prior_median_price
            is_flag = (volume_ratio < VOLUME_RATIO_LIMIT and
                       (price_ratio >= PRICE_RATIO_LIMIT or price_ratio <= Decimal(1) / PRICE_RATIO_LIMIT))
        anomaly = any(int(row.get(key, 0)) > 0 for key in (
            "negativeAmountCount", "negativeQuantityCount", "nullAmountCount",
            "nullQuantityCount", "duplicateRowCount", "nameConflictCount"))
        entry = {
            "date": text, "eligibleObservation": ok, "exclusionReason": reason,
            "source": {key: row.get(key) for key in (
                "rowCount", "marketCount", "marketCodes", "marketNames", "productNames",
                "nameConflictCount", "nullAmountCount", "nullQuantityCount",
                "negativeAmountCount", "negativeQuantityCount", "duplicateRowCount",
                "amountSum", "quantitySum", "price", "priceReason")},
            "priorEligibleObservationCount": len(prior),
            "priorEligibleDates": [item["date"] for item in prior],
            "prior28ObservationWindowAvailable": has_window,
            "prior28MedianPrice": str(prior_median_price) if prior_median_price is not None else None,
            "prior28MedianNetQuantity": str(prior_median_quantity) if prior_median_quantity is not None else None,
            "netQuantityToPriorMedianRatio": str(volume_ratio) if volume_ratio is not None else None,
            "priceToPriorMedianRatio": str(price_ratio) if price_ratio is not None else None,
            "flagged": is_flag,
            "sourceAnomaly": anomaly,
        }
        daily.append(entry)
        if anomaly:
            anomalous.append(text)
        if is_flag:
            flagged.append(text)
        if ok:
            eligible_history.append({"date": text, "_price": price, "_quantity": quantity})

    report = {
        "schemaVersion": 1,
        "protocolId": PROTOCOL_ID,
        "protocol": {
            "archiveSha256": national_transfer.ARCHIVE_SHA256,
            "reader": "national_transfer.parse_archive; all source rows retained in signed daily aggregates",
            "eligibleObservation": "exact radish product name; complete aggregate; positive aggregate price; positive signed net quantity",
            "priorWindow": "the immediately preceding 28 eligible observed dates, requiring at least 14; current date excluded; no calendar filling or imputation",
            "flag": "current net quantity / prior 28-observation median < 0.01 AND (current price / prior median >= 4 OR <= 0.25)",
        },
        "source": {key: parsed.get(key) for key in (
            "archiveSha256", "archiveByteCount", "sourceRowCount", "exactDuplicateRowCount", "periods")},
        "counts": {"observedDates": len(daily), "eligibleDates": len(eligible_history),
                   "sourceAnomalyDates": len(anomalous), "flaggedDates": len(flagged)},
        "sourceAnomalyDates": anomalous,
        "flaggedDates": flagged,
        "dailyObservations": daily,
        "interpretation": "Input go/no-go screen only. No model fitting or performance scoring; flags do not establish that source variety, grade, or market composition caused the price change. If no date is flagged or composition cannot be assessed, retain Ridge100 and do not fit the national guard.",
    }
    report["reportSha256"] = _hash(report)
    return report


def _hash(document: dict) -> str:
    payload = json.dumps(document, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def build_report(archive_bytes: bytes, *, expected_sha256: str | None = national_transfer.ARCHIVE_SHA256) -> dict:
    return screen(national_transfer.parse_archive(archive_bytes, expected_sha256=expected_sha256))


def write_create_only(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(report, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=INPUT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args(argv)
    report = build_report(args.archive.read_bytes())
    write_create_only(args.output, report)
    print(json.dumps({"output": str(args.output), "reportSha256": report["reportSha256"],
                      "flaggedDates": report["flaggedDates"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
