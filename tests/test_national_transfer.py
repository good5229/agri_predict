import csv
import datetime as dt
from decimal import Decimal
import io
import json
import zipfile

import pytest

from agri_data import market_recompose
from agri_data.national_transfer import (
    EXPECTED_HEADERS, FAMILY_SIZE, PAGE_CHECKS, PERIODS, PRODUCT_CODES,
    build_assessment, load_page_reference, parse_archive, validate_public_page_values,
)


def _csv_bytes(rows, headers=EXPECTED_HEADERS):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=headers, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("cp949")


def _row(day, code="1001", name="배추", qty="10", amount="1000", market="01"):
    return {"경매일자": day.strftime("%Y%m%d"), "도매시장코드": market,
            "도매시장명": "시장" + market, "품목코드": code, "품목명": name,
            "변경총물량": qty, "경매낙찰총금액": amount}


def _zip(period_rows=None, *, names=None, headers=EXPECTED_HEADERS):
    period_rows = period_rows or [[_row(start)] for start, _ in PERIODS]
    names = names or [f"period-{i}.csv" for i in range(4)]
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, rows in zip(names, period_rows):
            archive.writestr(name, _csv_bytes(rows, headers=headers))
    return output.getvalue()


def _archive_for_pairs(targets):
    observations = {p: {} for p in PRODUCT_CODES}
    for product, day in targets:
        observations[product][day.isoformat()] = {"price": "1000", "amountSum": "10000", "quantitySum": "10",
            "rowCount": 1, "marketCount": 1, "marketCodes": ["01"], "marketNames": ["시장01"],
            "productNames": ["배추"], "nameConflictCount": 0, "nullAmountCount": 0, "nullQuantityCount": 0,
            "negativeAmountCount": 0, "negativeQuantityCount": 0, "duplicateRowCount": 0, "priceReason": None}
    return {"archiveSha256": "a" * 64, "archiveByteCount": 1, "csvCount": 4, "sourceRowCount": 1,
            "exactDuplicateRowCount": 0, "periods": [{"name": f"p{i}", "sha256": "b" * 64, "byteCount": 1,
                "rowCount": 1, "start": start.isoformat(), "end": end.isoformat()} for i, (start, end) in enumerate(PERIODS)],
            "observations": observations}


def _report(*, count=30, missing_pair_at=None):
    evidence, paired = [], []
    start = dt.date(2025, 7, 15)
    for i in range(count):
        target = start + dt.timedelta(days=i)
        origin = target - dt.timedelta(days=1)
        key = {"product": "cabbage", "origin": origin.isoformat(), "target": target.isoformat(), "horizon": 1}
        evidence.append({**key, "candidatePrediction": 990.0 + i % 3, "baselinePrediction": 1010.0,
                         "persistencePrediction": 1020.0})
        if i != missing_pair_at:
            paired.append({**key, "actual": 1005.0, "candidatePrediction": 990.0 + i % 3,
                           "directRidge": 1010.0, "persistence": 1020.0})
    return {"schemaVersion": 1, "status": "complete", "protocol": {"id": market_recompose.PROTOCOL["id"]},
            "originEvidence": evidence, "pairedRows": paired}


def test_archive_parses_all_periods_and_preserves_signed_null_duplicate_audit():
    day = dt.date(2024, 7, 8)
    rows = [[_row(day, qty="10", amount="1000"), _row(day, qty="-2", amount="-100"),
             _row(day, qty="10", amount="1000"), _row(day, qty="", amount="50"),
             _row(day, code="1209", name="마늘", qty="4", amount="400")]]
    rows += [[_row(start)] for start, _ in PERIODS[1:]]
    parsed = parse_archive(_zip(rows), expected_sha256=None)
    observed = parsed["observations"]["cabbage"][day.isoformat()]
    assert observed["amountSum"] == "1950"
    assert observed["quantitySum"] == "18"
    assert observed["price"] is None
    assert observed["priceReason"] == "null_components"
    assert observed["negativeAmountCount"] == 1
    assert observed["negativeQuantityCount"] == 1
    assert observed["nullQuantityCount"] == 1
    assert observed["duplicateRowCount"] == 1
    assert observed["marketCount"] == 1
    assert "1209" not in parsed["observations"]
    assert parsed["sourceRowCount"] == 8
    assert parsed["exactDuplicateRowCount"] == 1
    assert len(parsed["periods"]) == 4


def test_archive_uses_all_markets_and_signed_decimal_ratio_without_fill():
    day = dt.date(2024, 7, 9)
    rows = [[_row(day, qty="1.25", amount="12.50", market="01"),
             _row(day, qty="0.75", amount="-1.50", market="02")]]
    rows += [[_row(start)] for start, _ in PERIODS[1:]]
    parsed = parse_archive(_zip(rows), expected_sha256=None)
    observed = parsed["observations"]["cabbage"][day.isoformat()]
    assert Decimal(observed["amountSum"]) == Decimal("11.00")
    assert Decimal(observed["quantitySum"]) == Decimal("2.00")
    assert Decimal(observed["price"]) == Decimal("5.5")
    assert observed["marketCount"] == 2
    assert observed["marketCodes"] == ["01", "02"]
    assert parsed["observations"]["cabbage"].get("2024-07-10") is None


def test_negative_signed_price_is_preserved_but_not_admitted_as_positive_truth():
    day = dt.date(2024, 7, 8)
    rows = [[_row(day, qty="2", amount="-10")]]
    rows += [[_row(start)] for start, _ in PERIODS[1:]]
    parsed = parse_archive(_zip(rows), expected_sha256=None)
    observed = parsed["observations"]["cabbage"][day.isoformat()]
    assert observed["price"] == "-5"
    assert observed["priceReason"] == "nonpositive_aggregate_price"


def test_archive_fails_closed_on_hash_member_schema_period_and_nonnumeric_values():
    valid = _zip()
    with pytest.raises(ValueError, match="SHA-256"):
        parse_archive(valid)
    rows = [[_row(start)] for start, _ in PERIODS]
    rows[1] = [_row(PERIODS[0][0])]
    with pytest.raises(ValueError, match="duplicate archive period|missing frozen periods"):
        parse_archive(_zip(rows), expected_sha256=None)
    with pytest.raises(ValueError, match="schema changed"):
        parse_archive(_zip(headers=EXPECTED_HEADERS[:-1]), expected_sha256=None)
    rows = [[_row(start)] for start, _ in PERIODS]
    rows[0][0]["경매낙찰총금액"] = "oops"
    with pytest.raises(ValueError, match="non-numeric"):
        parse_archive(_zip(rows), expected_sha256=None)
    bad_path = _zip(names=["../escape.csv", "p1.csv", "p2.csv", "p3.csv"])
    with pytest.raises(ValueError, match="unsafe"):
        parse_archive(bad_path, expected_sha256=None)


def test_page_check_requires_explicit_four_values_and_never_fetches():
    page_values = {"2025-07-15": {"cabbage": 1000, "onion": 1000},
                   "2026-06-30": {"cabbage": 1000, "onion": 1000}}
    archive = _archive_for_pairs((product, day) for day, product in PAGE_CHECKS)
    result = validate_public_page_values(archive, page_values)
    assert result["pageFetched"] is False
    assert result["allFourObservedIntegersSupplied"] is True
    assert result["alignmentClaimAllowed"] is True
    assert result["status"] == "aligned_with_public_page"
    partial = validate_public_page_values(archive, {"2025-07-15": {"cabbage": 1000}})
    assert partial["alignmentClaimAllowed"] is False
    assert partial["allFourObservedIntegersSupplied"] is False
    discrepant = validate_public_page_values(archive, {"2025-07-15": {"cabbage": 1004, "onion": 1000},
        "2026-06-30": {"cabbage": 1000, "onion": 1000}})
    assert discrepant["status"] == "unverified_or_discrepant"
    assert discrepant["alignmentClaimAllowed"] is False


def test_manual_page_reference_validates_four_fixed_values_and_is_hashed():
    doc = {"schemaVersion": 1, "capturedAtUtc": "2026-10-07T00:00:00Z", "sourceUrl": "https://example.invalid",
        "selection": {"market": "*전국도매시장", "variety": "전체", "priceType": "평균", "unit": "원/kg"},
        "observations": [{"date": day.isoformat(), "cropCode": PRODUCT_CODES[product], "cropName": {"cabbage": "배추", "onion": "양파"}[product],
            "displayPriceWonPerKg": 1000} for day, product in PAGE_CHECKS]}
    values, provenance = load_page_reference(json.dumps(doc, ensure_ascii=False).encode("utf-8"), expected_sha256=None)
    assert values["2025-07-15"] == {"cabbage": 1000, "onion": 1000}
    assert len(provenance["sha256"]) == 64
    doc["observations"].pop()
    with pytest.raises(ValueError, match="exactly the four"):
        load_page_reference(json.dumps(doc, ensure_ascii=False).encode("utf-8"), expected_sha256=None)


def test_fixed_pairing_family_metrics_coverage_macro_and_unusable_dm_p1():
    start = dt.date(2025, 7, 15)
    archive = _archive_for_pairs(("cabbage", start + dt.timedelta(days=i)) for i in range(30))
    result = build_assessment(archive, _report(), archive_sha256="a" * 64, report_sha256="c" * 64)
    assert len(result["comparisons"]) == FAMILY_SIZE == 160
    assert all(r["dm"]["9"]["pValue"] >= 0 for r in result["comparisons"])
    assert all("holmPValue" in r["dm"]["9"] and "holmPValue" in r["dm"]["28"] for r in result["comparisons"])
    assert all(r["dm"]["9"]["pValue"] == 1 for r in result["comparisons"] if r["n"] < 30)
    assert len(result["pairedRows"]) == 30
    assert result["coverage"]["cabbage/1"]["paired"] == 30
    assert result["coverage"]["cabbage/1"]["candidateCoverageLossKeys"] == {"directRidge": [], "persistence": []}
    assert result["equalHorizonMacroMape"]["cabbage"]["pointEstimateAvailable"] is False
    assert result["equalHorizonMacroMape"]["cabbage"]["mapeByHorizon"]["1"] is not None
    assert result["equalHorizonMacroMape"]["radish"]["equalHorizonMape"] is None
    assert result["nationalVsTwoMarketDiscrepancy"]["cabbage"]["uniqueTargetDates"] == 30
    assert result["publicPageAlignment"]["alignmentClaimAllowed"] is False
    assert result["yearSeasonMetrics"][0]["stratum"] in ("season", "year")
    json.dumps(result, allow_nan=False)


def test_join_uses_full_identity_and_reports_missing_pairs_and_national_targets():
    target = dt.date(2025, 7, 15)
    archive = _archive_for_pairs([("cabbage", target)])
    report = _report(count=1, missing_pair_at=0)
    result = build_assessment(archive, report, archive_sha256="a" * 64, report_sha256="c" * 64)
    assert result["pairedRows"] == []
    assert result["exclusionReasonCounts"] == {"missing_common_market_recompose_pair": 1}
    report["pairedRows"] = [{"product": "cabbage", "origin": (target - dt.timedelta(days=2)).isoformat(),
        "target": target.isoformat(), "horizon": 1, "actual": 1000, "candidatePrediction": 900,
        "directRidge": 1000, "persistence": 1000}]
    with pytest.raises(ValueError, match="origin/target/horizon mismatch"):
        build_assessment(archive, report)
    report = _report(count=1)
    report["pairedRows"][0]["origin"] = (target - dt.timedelta(days=2)).isoformat()
    report["pairedRows"][0]["horizon"] = 2
    result = build_assessment(archive, report)
    assert result["pairedRows"] == []
    assert result["exclusionReasonCounts"] == {"missing_common_market_recompose_pair": 1}
    archive["observations"]["cabbage"][target.isoformat()]["price"] = None
    archive["observations"]["cabbage"][target.isoformat()]["priceReason"] = "nonpositive_signed_quantity"
    report["pairedRows"] = _report(count=1)["pairedRows"]
    result = build_assessment(archive, report)
    assert result["exclusionReasonCounts"] == {"national_target_unusable:nonpositive_signed_quantity": 1}


def test_duplicate_paired_keys_rejected_and_exact_report_protocol_required():
    target = dt.date(2025, 7, 15)
    archive = _archive_for_pairs([("cabbage", target)])
    report = _report(count=1)
    report["pairedRows"].append(dict(report["pairedRows"][0]))
    with pytest.raises(ValueError, match="duplicate market-recompose pairedRows key"):
        build_assessment(archive, report)
    report = _report(count=1)
    report["protocol"]["id"] = "other"
    with pytest.raises(ValueError, match="protocol differs"):
        build_assessment(archive, report)
