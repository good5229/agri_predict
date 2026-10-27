import datetime as dt
import json

import pytest

from agri_data.national_level_anchor import (
    FAMILY_SIZE, NATIONAL_ARCHIVE_SHA256, NATIONAL_TRANSFER_PROTOCOL,
    MARKET_RECOMPOSE_SHA256, PRODUCTS, REPORT_SHA256, build_assessment,
    evaluate_bytes,
)


def _pair(product, target, horizon=1, *, national=120.0, local=100.0,
          candidate=100.0, ridge=110.0, persistence=130.0):
    origin = target - dt.timedelta(days=horizon)
    return {"product": product, "origin": origin.isoformat(), "target": target.isoformat(), "horizon": horizon,
            "nationalActual": national, "twoMarketActual": local, "candidatePrediction": candidate,
            "directRidge": ridge, "persistence": persistence}


def _report(rows):
    target_audit = {product: {} for product in PRODUCTS}
    for row in rows:
        product, day = row["product"], row["target"]
        current = target_audit[product].get(day)
        price = str(row["nationalActual"])
        if current is not None and current["price"] != price:
            raise AssertionError("test fixture has inconsistent national actuals")
        target_audit[product][day] = {"amountSum": "1200", "quantitySum": "10", "price": price,
            "priceReason": None, "rowCount": 1, "marketCount": 2, "marketCodes": ["01", "02"],
            "marketNames": ["Market 1", "Market 2"], "productNames": [product], "nameConflictCount": 0,
            "nullAmountCount": 0, "nullQuantityCount": 0, "negativeAmountCount": 0,
            "negativeQuantityCount": 0, "duplicateRowCount": 0}
    return {"schemaVersion": 1, "status": "complete",
        "protocol": {"id": NATIONAL_TRANSFER_PROTOCOL, "archiveSha256": NATIONAL_ARCHIVE_SHA256},
        "hashes": {"archiveSha256": NATIONAL_ARCHIVE_SHA256, "marketRecomposeSha256": MARKET_RECOMPOSE_SHA256},
        "targetAudit": target_audit, "pairedRows": rows}


def _history_row(day, ratio, *, product="cabbage"):
    return _pair(product, day, 1, national=100.0 * ratio, local=100.0,
                 candidate=100.0, ridge=100.0, persistence=100.0)


def test_origin_anchor_uses_median_between_minus_15_and_minus_2_without_leakage():
    origin = dt.date(2026, 1, 20)
    rows = [
        _history_row(origin - dt.timedelta(days=16), 99.0),  # outside window
        _history_row(origin - dt.timedelta(days=15), 1.1),
        _history_row(origin - dt.timedelta(days=10), 1.2),
        _history_row(origin - dt.timedelta(days=2), 1.3),
        _history_row(origin - dt.timedelta(days=1), 88.0),  # forbidden near-origin value
        _pair("cabbage", origin + dt.timedelta(days=1), national=150.0, local=100.0,
              candidate=100.0, ridge=110.0, persistence=120.0),
    ]
    report = _report(rows)
    result = build_assessment(report, report_sha256="a" * 64)
    anchor = next(x for x in result["originAnchors"] if x["origin"] == origin.isoformat())
    assert anchor["anchorDates"] == [
        (origin - dt.timedelta(days=15)).isoformat(),
        (origin - dt.timedelta(days=10)).isoformat(),
        (origin - dt.timedelta(days=2)).isoformat(),
    ]
    assert anchor["eligibleDateCount"] == 3
    assert anchor["ratio"] == pytest.approx(1.2)
    assert anchor["fallback"] is False
    row = next(x for x in result["pairedRows"] if x["origin"] == origin.isoformat())
    assert row["candidatePrediction"] == pytest.approx(120.0)
    assert row["scaledDirectRidge"] == pytest.approx(132.0)
    assert row["scaledPersistence"] == pytest.approx(144.0)


def test_fewer_than_three_anchor_days_uses_ratio_one_and_records_reason():
    origin = dt.date(2026, 2, 1)
    rows = [_history_row(origin - dt.timedelta(days=5), 2.0, product="onion"),
            _history_row(origin - dt.timedelta(days=3), 3.0, product="onion"),
            _pair("onion", origin + dt.timedelta(days=1), national=200.0, local=100.0)]
    result = build_assessment(_report(rows), report_sha256="b" * 64)
    anchor = next(x for x in result["originAnchors"] if x["origin"] == origin.isoformat())
    assert anchor["ratio"] == 1.0
    assert anchor["eligibleDateCount"] == 2
    assert anchor["fallback"] is True
    assert anchor["fallbackReason"] == "fewer_than_3_eligible_anchor_dates"
    assert result["pairedRows"][0]["candidatePrediction"] == 100.0
    assert result["fallback"]["rowCount"] == 3


def test_all_240_tests_are_emitted_and_insufficient_inference_is_p1():
    target = dt.date(2026, 1, 10)
    result = build_assessment(_report([_pair("cabbage", target)]), report_sha256="c" * 64)
    assert len(result["comparisons"]) == FAMILY_SIZE == 240
    assert len(result["jointConclusions"]) == 40
    assert all(row["n"] == 0 or row["n"] == 1 for row in result["comparisons"])
    assert all(row["dm"]["9"]["pValue"] == 1 and row["dm"]["28"]["pValue"] == 1
               for row in result["comparisons"])
    assert len(result["pairedRows"]) == 1
    assert result["macro10EqualHorizonMape"]["cabbage"]["equalHorizonMape"] is None
    assert result["fallback"]["reasonCounts"]
    json.dumps(result, allow_nan=False)


def test_thirty_pairs_remain_common_and_macro_is_equal_horizon_point_estimate():
    start, end = dt.date(2025, 12, 16), dt.date(2026, 1, 30)
    rows = [_pair("cabbage", start + dt.timedelta(days=i), candidate=98.0 + (i % 3),
                  ridge=110.0, persistence=130.0)
            for i in range((end - start).days + 1)]
    result = build_assessment(_report(rows), report_sha256="d" * 64)
    h1 = next(r for r in result["comparisons"] if r["product"] == "cabbage" and r["horizon"] == 1
              and r["comparator"] == "originalCandidate" and r["loss"] == "mae")
    assert h1["n"] == 30
    assert len(result["pairedRows"]) == 30
    assert result["coverage"]["cabbage/1"]["eligibleCommonPairs"] == 30
    assert result["coverage"]["cabbage/1"]["candidateCoverageLoss"] == {
        "originalCandidate": 0, "scaledDirectRidge": 0, "scaledPersistence": 0}
    assert result["macro10EqualHorizonMape"]["cabbage"]["pairedCountsByHorizon"]["1"] == 30
    assert result["macro10EqualHorizonMape"]["cabbage"]["equalHorizonMape"] is None


def test_repeated_target_date_actuals_must_match_national_audit_and_each_other():
    target = dt.date(2026, 1, 15)
    rows = [_pair("radish", target, 1), _pair("radish", target, 2)]
    report = _report(rows)
    rows[1]["twoMarketActual"] += 1
    with pytest.raises(ValueError, match="inconsistent repeated target-date actual.*two-market"):
        build_assessment(report)
    rows = [_pair("radish", target, 1), _pair("radish", target, 2)]
    report = _report(rows)
    rows[1]["nationalActual"] += 1
    with pytest.raises(ValueError, match="inconsistent repeated target-date actual.*national"):
        build_assessment(report)


def test_strict_report_hash_schema_duplicate_keys_and_input_protocol():
    report = _report([_pair("pepper", dt.date(2026, 1, 5))])
    raw = json.dumps(report, sort_keys=True).encode()
    with pytest.raises(ValueError, match="SHA-256"):
        evaluate_bytes(raw)
    assert evaluate_bytes(raw, expected_sha256=None)["hashes"]["nationalTransferReportSha256"] == __import__("hashlib").sha256(raw).hexdigest()
    report["protocol"]["id"] = "changed"
    with pytest.raises(ValueError, match="protocol"):
        build_assessment(report)
    duplicate_json = b'{"schemaVersion":1,"schemaVersion":1}'
    with pytest.raises(ValueError, match="duplicate JSON key"):
        evaluate_bytes(duplicate_json, expected_sha256=None)


def test_duplicate_pair_key_bad_horizon_and_missing_audit_actual_fail_closed():
    target = dt.date(2026, 2, 5)
    rows = [_pair("onion", target)]
    report = _report(rows + [dict(rows[0])])
    with pytest.raises(ValueError, match="duplicate national-transfer paired key"):
        build_assessment(report)
    row = _pair("onion", target)
    row["horizon"] = 2
    report = _report([row])
    with pytest.raises(ValueError, match="origin/target/horizon mismatch"):
        build_assessment(report)
    report = _report([_pair("onion", target)])
    report["targetAudit"]["onion"].pop(target.isoformat())
    with pytest.raises(ValueError, match="no positive targetAudit"):
        build_assessment(report)
