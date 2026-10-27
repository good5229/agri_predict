import csv
import datetime as dt
import hashlib
import io
import json
import zipfile
from decimal import Decimal

from agri_data import national_archive_alignment_screen as alignment
from agri_data import nongnet_national_history_screen as history


HEADERS = ("경매일자", "도매시장코드", "도매시장명", "품목코드", "품목명", "변경총물량", "경매낙찰총금액")
NAMES = {"cabbage": "배추", "radish": "무", "onion": "양파", "pepper": "건고추"}
DATES = ("2024-09-11", "2025-09-11")
PARTS = (("2024-07-08", "2024-12-31"), ("2025-01-01", "2025-06-30"),
         ("2025-07-01", "2025-12-31"), ("2026-01-01", "2026-06-30"))


def _make_archive(extra_rows=()):
    grouped = {0: [], 1: [], 2: [], 3: []}
    dates = ("2024-09-11", "2025-03-01", "2025-09-11", "2026-02-01")
    for d in (DATES[0], DATES[1]):
        period = 0 if d == DATES[0] else 2
        day_digits = d.replace("-", "")
        for product, code in alignment.PRODUCTS:
            price = Decimal("10") + Decimal(code) / Decimal("1000")
            grouped[period].append((day_digits, "M1", "Market One", code, NAMES[product], "1000", str(price * 1000)))
    grouped[1].append((dates[1].replace("-", ""), "M1", "Market One", "1001", "배추", "1", "1"))
    grouped[3].append((dates[3].replace("-", ""), "M1", "Market One", "1001", "배추", "1", "1"))
    for row in extra_rows:
        day = dt.datetime.strptime(row[0], "%Y%m%d").date()
        period = 0 if day.year == 2024 else 2
        grouped[period].append(row)
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for i, rows in grouped.items():
            text = io.StringIO(newline="")
            writer = csv.writer(text, lineterminator="\r\n")
            writer.writerow(HEADERS)
            writer.writerows(rows)
            archive.writestr(f"period_{i}.csv", text.getvalue().encode("cp949"))
    return output.getvalue()


def _make_history(directory, *, bad_metadata=None, screen_volume=1):
    directory.mkdir()
    for date in DATES:
        for product, code in alignment.PRODUCTS:
            parsed = dt.date.fromisoformat(date)
            params = {"searchUnitGrade": "normal", "searchDate": f"{parsed.year}년 {parsed.month:02d}월 {parsed.day:02d}일",
                      "searchSymbol1": "100000", "searchSymbol2": code, "searchSymbol3": ""}
            if bad_metadata == (date, product):
                params["searchSymbol2"] = "wrong"
            price = 10 + Decimal(code) / Decimal("1000")
            body = json.dumps({"datalist": [{"saledate": date.replace("-", ""),
                                             "selectDay": str(price), "selectDayWgt": screen_volume}]}).encode()
            stem = f"nongnet_national_history_{date}_{product}_a1_fixture"
            raw_name, manifest_name = stem + ".json", stem + ".manifest.json"
            (directory / raw_name).write_bytes(body)
            validation = history.validate_json(body, date)
            manifest = {"endpoint": alignment.ENDPOINT, "requestedDate": date, "category": product,
                        "requestParams": params, "rawFile": raw_name, "responseBytes": len(body),
                        "sha256": hashlib.sha256(body).hexdigest(), "httpStatus": 200,
                        "truncated": False, "transportError": None, "validation": validation}
            (directory / manifest_name).write_text(json.dumps(manifest), encoding="utf-8")


def test_eight_fixed_pairs_use_pinned_parser_and_compare_price_and_tolerance(tmp_path):
    source = _make_archive()
    history_dir = tmp_path / "history"
    _make_history(history_dir)
    report = alignment.build_report(source, history_dir, expected_archive_sha256=None)
    assert len(report["results"]) == 8
    assert all(row["pairStatus"] == "comparable" for row in report["results"])
    assert all(row["priceMatch"] and row["volumeMatch"] for row in report["results"])
    assert all(row["archiveQuantitySumKg"] == "1000" and row["archiveVolumeT"] == "1"
               for row in report["results"])


def test_signed_negative_and_null_audit_values_are_preserved(tmp_path):
    source = _make_archive(extra_rows=[
        ("20240911", "M2", "Market Two", "1001", "배추", "-100", "-1000"),
        ("20240911", "M3", "Market Three", "1001", "배추", "", "500"),
    ])
    history_dir = tmp_path / "history"
    _make_history(history_dir)
    row = next(x for x in alignment.build_report(source, history_dir, expected_archive_sha256=None)["results"]
               if x["date"] == "2024-09-11" and x["product"] == "cabbage")
    assert row["archiveQuantitySumKg"] == "900"
    assert row["archiveAmountSumWon"] == "10501.000"
    assert row["archiveNegativeAmountCount"] == 1
    assert row["archiveNegativeQuantityCount"] == 1
    assert row["archiveNullQuantityCount"] == 1
    assert not row["archiveComplete"] and not row["priceComparable"]
    assert not row["volumeComparable"]  # unknown quantity is not imputed into the sum


def test_metadata_mismatch_is_reported_without_comparison(tmp_path):
    source = _make_archive()
    history_dir = tmp_path / "history"
    _make_history(history_dir, bad_metadata=("2024-09-11", "cabbage"))
    row = next(x for x in alignment.build_report(source, history_dir, expected_archive_sha256=None)["results"]
               if x["date"] == "2024-09-11" and x["product"] == "cabbage")
    assert row["screenStatus"] == "request_metadata_mismatch"
    assert row["pairStatus"] == "not_comparable"
    assert not row["priceMatch"] and not row["volumeMatch"]


def test_one_decimal_volume_display_uses_tighter_tolerance(tmp_path):
    history_dir = tmp_path / "history"
    _make_history(history_dir, screen_volume=1.4)
    row = alignment.build_report(_make_archive(), history_dir,
                                 expected_archive_sha256=None)["results"][0]
    assert row["volumeToleranceT"] == "0.05"
    assert row["volumeDifferenceT"] == "0.4"
    assert not row["volumeMatch"]


def test_trailing_zero_volume_display_normalizes_but_excess_precision_is_unsupported(tmp_path):
    history_dir = tmp_path / "history"
    _make_history(history_dir, screen_volume="1.40")
    row = alignment.build_report(_make_archive(), history_dir,
                                 expected_archive_sha256=None)["results"][0]
    assert row["screenVolumeEffectiveDecimalPlaces"] == 1
    assert row["volumeToleranceT"] == "0.05"
    assert not row["volumeMatch"]

    history_dir = tmp_path / "history_more_precision"
    _make_history(history_dir, screen_volume="1.42")
    row = alignment.build_report(_make_archive(), history_dir,
                                 expected_archive_sha256=None)["results"][0]
    assert row["screenVolumeEffectiveDecimalPlaces"] == 2
    assert row["volumeToleranceT"] == "unsupported_precision"
    assert not row["volumeComparable"] and not row["volumeMatch"]


def test_product_name_conflicts_are_preserved_and_excluded_from_volume_match(tmp_path):
    source = _make_archive(extra_rows=[
        ("20240911", "M2", "Market Two", "1001", "Not Cabbage", "10", "110"),
    ])
    history_dir = tmp_path / "history"
    _make_history(history_dir)
    row = next(x for x in alignment.build_report(source, history_dir,
                                                 expected_archive_sha256=None)["results"]
               if x["date"] == "2024-09-11" and x["product"] == "cabbage")
    assert row["archiveNameConflictCount"] == 1
    assert row["archivePriceReason"] == "product_name_conflict"
    assert not row["volumeComparable"] and not row["volumeMatch"]
