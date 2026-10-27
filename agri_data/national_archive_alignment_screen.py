"""Offline comparison of pinned national auction aggregates and Nongnet JSON."""
from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
from typing import Any

from agri_data import national_transfer
from agri_data import nongnet_national_history_screen as history_screen

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = ROOT / "data/raw/national-wholesale-20240708-20260630.zip"
HISTORY_DIR = ROOT / "data/raw/nongnet_national_history_screen"
ARCHIVE_SHA256 = "b9a38d4a62477054758e7576a7320131bc5acc8129a4079a33977f9ed876d7d7"
DATES = ("2024-09-11", "2025-09-11")
PRODUCTS = (
    ("cabbage", "1001"), ("radish", "1101"), ("onion", "1201"), ("pepper", "1207"),
)
ENDPOINT = history_screen.ENDPOINT


def _params(date: str, code: str) -> dict[str, str]:
    day = dt.date.fromisoformat(date)
    return {"searchUnitGrade": "normal", "searchDate": f"{day.year}년 {day.month:02d}월 {day.day:02d}일",
            "searchSymbol1": "100000", "searchSymbol2": code, "searchSymbol3": ""}


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return number if number.is_finite() else None


def _effective_decimal_places(value: Decimal) -> int:
    parts = value.as_tuple()
    places = max(0, -parts.exponent)
    trailing = 0
    for digit in reversed(parts.digits):
        if trailing >= places or digit != 0:
            break
        trailing += 1
    return places - trailing


def _screen_source(date: str, product: str, code: str, history_dir: Path) -> dict[str, Any]:
    pattern = f"*_{date}_{product}_*.manifest.json"
    manifests = sorted(history_dir.glob(pattern))
    base: dict[str, Any] = {"manifestCount": len(manifests), "manifestFile": None,
                             "rawFile": None, "screenPrice": None, "screenVolumeT": None,
                             "screenValidationState": None, "screenIssues": []}
    if len(manifests) != 1:
        base["screenStatus"] = "manifest_count_not_one"
        return base
    manifest_path = manifests[0]
    base["manifestFile"] = manifest_path.name
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        base["screenStatus"] = "manifest_unreadable"
        return base
    params = manifest.get("requestParams")
    if (manifest.get("endpoint") != ENDPOINT or manifest.get("requestedDate") != date or
            manifest.get("category") != product or params != _params(date, code)):
        base["screenStatus"] = "request_metadata_mismatch"
        return base
    raw_name = manifest.get("rawFile")
    if not isinstance(raw_name, str) or Path(raw_name).name != raw_name:
        base["screenStatus"] = "raw_file_reference_invalid"
        return base
    raw_path = history_dir / raw_name
    base["rawFile"] = raw_name
    try:
        raw = raw_path.read_bytes()
    except OSError:
        base["screenStatus"] = "raw_file_missing"
        return base
    if len(raw) != manifest.get("responseBytes") or hashlib.sha256(raw).hexdigest() != manifest.get("sha256"):
        base["screenStatus"] = "raw_integrity_mismatch"
        return base
    validation = history_screen.validate_json(raw, date, manifest.get("httpStatus") or 0,
                                              truncated=bool(manifest.get("truncated")),
                                              transport_error=manifest.get("transportError"))
    base["screenValidationState"] = validation.get("state")
    base["screenIssues"] = list(validation.get("issues", []))
    if validation.get("matchingDateCount") != 1 or not isinstance(validation.get("matchingRow"), dict):
        base["screenStatus"] = "requested_date_match_not_unique"
        return base
    if validation.get("state") != "valid" or validation["matchingRow"].get("issues"):
        base["screenStatus"] = "requested_date_row_not_valid"
        return base
    original_validation = manifest.get("validation")
    if (not isinstance(original_validation, dict) or
            original_validation.get("state") != "valid" or
            original_validation.get("matchingDateCount") != 1):
        base["screenStatus"] = "manifest_validation_not_valid"
        return base
    base["screenPrice"] = _decimal(validation["matchingRow"].get("selectDay"))
    base["screenVolumeT"] = _decimal(validation["matchingRow"].get("selectDayWgt"))
    if base["screenPrice"] is None or base["screenVolumeT"] is None:
        base["screenStatus"] = "screen_value_not_finite_numeric"
        return base
    base["screenStatus"] = "valid"
    return base


def build_report(archive_bytes: bytes, history_dir: Path = HISTORY_DIR, *,
                 expected_archive_sha256: str | None = ARCHIVE_SHA256) -> dict[str, Any]:
    """Build the frozen eight-pair report; never fetches or changes source files."""
    archive = national_transfer.parse_archive(archive_bytes, expected_sha256=expected_archive_sha256)
    results = []
    for date in DATES:
        for product, code in PRODUCTS:
            observed = archive.get("observations", {}).get(product, {}).get(date)
            screen = _screen_source(date, product, code, Path(history_dir))
            quantity_sum = Decimal(observed["quantitySum"]) if observed else None
            archive_price = Decimal(observed["price"]) if observed and observed.get("price") is not None else None
            price_difference = abs(archive_price - screen["screenPrice"]) if archive_price is not None and screen["screenPrice"] is not None else None
            archive_volume_t = quantity_sum / Decimal(1000) if observed and observed.get("nullQuantityCount") == 0 and quantity_sum is not None and quantity_sum > 0 else None
            volume_difference = abs(archive_volume_t - screen["screenVolumeT"]) if archive_volume_t is not None and screen["screenVolumeT"] is not None else None
            volume_precision = (_effective_decimal_places(screen["screenVolumeT"])
                                if screen["screenVolumeT"] is not None else None)
            volume_tolerance = (Decimal("0.5") if volume_precision == 0 else
                                Decimal("0.05") if volume_precision == 1 else None)
            screen_valid = screen["screenStatus"] == "valid"
            price_comparable = archive_price is not None and archive_price > 0 and screen_valid
            volume_comparable = (archive_volume_t is not None and screen_valid and
                                 volume_precision in (0, 1) and
                                 observed.get("nameConflictCount", 0) == 0 if observed else False)
            archive_complete = bool(observed and observed.get("nullAmountCount") == 0 and
                                    observed.get("nullQuantityCount") == 0 and archive_price is not None and
                                    observed.get("nameConflictCount", 0) == 0)
            results.append({
                "date": date, "product": product, "productCode": code,
                "archiveStatus": "available" if observed else "no_observation",
                "archiveComplete": archive_complete,
                "archiveAmountSumWon": observed.get("amountSum") if observed else None,
                "archiveQuantitySumKg": observed.get("quantitySum") if observed else None,
                "archivePriceWonPerKg": observed.get("price") if observed else None,
                "archiveMarketCount": observed.get("marketCount") if observed else 0,
                "archiveRowCount": observed.get("rowCount") if observed else 0,
                "archiveNegativeAmountCount": observed.get("negativeAmountCount") if observed else 0,
                "archiveNegativeQuantityCount": observed.get("negativeQuantityCount") if observed else 0,
                "archiveNullAmountCount": observed.get("nullAmountCount") if observed else 0,
                "archiveNullQuantityCount": observed.get("nullQuantityCount") if observed else 0,
                "archiveNameConflictCount": observed.get("nameConflictCount") if observed else 0,
                "archivePriceReason": observed.get("priceReason") if observed else None,
                "screenManifestFile": screen["manifestFile"], "screenRawFile": screen["rawFile"],
                "screenStatus": screen["screenStatus"], "screenValidationState": screen["screenValidationState"],
                "screenIssues": screen["screenIssues"],
                "screenPriceWonPerKg": str(screen["screenPrice"]) if screen["screenPrice"] is not None else None,
                "screenVolumeT": str(screen["screenVolumeT"]) if screen["screenVolumeT"] is not None else None,
                "archiveVolumeT": str(archive_volume_t) if archive_volume_t is not None else None,
                "priceDifferenceWonPerKg": str(price_difference) if price_difference is not None else None,
                "priceToleranceWonPerKg": "1",
                "priceMatch": bool(price_comparable and price_difference <= Decimal(1)),
                "volumeDifferenceT": str(volume_difference) if volume_difference is not None else None,
                "screenVolumeEffectiveDecimalPlaces": volume_precision,
                "volumeToleranceT": str(volume_tolerance) if volume_tolerance is not None else "unsupported_precision",
                "volumeMatch": bool(volume_comparable and volume_difference <= volume_tolerance),
                "priceComparable": price_comparable, "volumeComparable": volume_comparable,
                "pairStatus": ("comparable" if price_comparable and volume_comparable else
                               "partially_comparable" if price_comparable or volume_comparable else "not_comparable"),
            })
    return {"schemaVersion": 1, "archiveSha256": archive["archiveSha256"],
            "archiveByteCount": archive["archiveByteCount"],
            "protocol": {"dates": list(DATES), "products": dict(PRODUCTS),
                         "pairCount": 8, "priceToleranceWonPerKg": "1",
                         "volumeToleranceT": {"integerDisplay": "0.5", "oneDecimalDisplay": "0.05",
                                              "moreThanOneDecimal": "non_comparable"},
                         "noFill": True, "readOnly": True},
            "results": results, "complete": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--history-dir", type=Path, default=HISTORY_DIR)
    args = parser.parse_args(argv)
    report = build_report(args.archive.read_bytes(), args.history_dir)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
