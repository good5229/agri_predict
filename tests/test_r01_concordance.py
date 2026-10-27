import pytest
import json

from agri_data import dashboard, r01_concordance as r01


def payload(trade="경매", unit="kg", subclass="01"):
    return {"gds_lclsf_cd": "10", "gds_sclsf_cd": subclass, "grd_nm": "특", "orgplc_nm": "국산",
            "gds_lclsf_nm": "채소류", "gds_mclsf_cd": "01", "gds_mclsf_nm": "엽경채류",
            "gds_sclsf_nm": "배추", "grd_cd": "01", "plor_cd": "1101", "plor_nm": "국산",
            "gds_mclsf_cd": "01", "trd_clcln_ymd": "2026-09-11", "whsl_mrkt_cd": next(iter(dashboard.MARKETS)),
            "trd_se": trade, "unit_nm": unit, "totprc": "100", "unit_tot_qty": "2"}


def query(market=None):
    market = market or next(iter(dashboard.MARKETS))
    return {"cond[gds_lclsf_cd::EQ]": "10", "cond[gds_mclsf_cd::EQ]": "01",
            "cond[trd_clcln_ymd::EQ]": "2026-09-11", "cond[whsl_mrkt_cd::EQ]": market,
            "cond[gds_sclsf_cd::EQ]": ""}


def test_summarize_selected_payload_dimensions_and_categories():
    rows = [
        {"date": "2026-09-11", "market": next(iter(dashboard.MARKETS)), "payload": payload(),
         "queryProduct": "cabbage", "exclusionReason": None},
        {"date": "2026-09-11", "market": next(iter(dashboard.MARKETS)), "payload": payload("정가"),
         "queryProduct": "cabbage", "exclusionReason": "trade_mode_not_auction"},
    ]
    states = {(d, m): "complete" for d in r01.DATES for m in dashboard.MARKETS}
    totals = {(d, m): {"signedNetAmount": 100, "signedNetWeight": 2,
        "eligibleAuctionKgRows": 1 if d == r01.DATES[0] and m == next(iter(dashboard.MARKETS)) else 0}
        for d in r01.DATES for m in dashboard.MARKETS}
    hashes = {"12": "abc", "13": "empty-response-hash"}
    result = r01.summarize_rows(rows, states, totals, hashes)
    first_market = next(iter(dashboard.MARKETS))
    cell = result["dates"]["2026-09-11"]["markets"][first_market]
    assert cell["payloadRows"] == 2
    assert cell["includedAuctionKgRows"] == 1
    assert cell["excludedCategories"] == {"trade_mode_not_auction": 1}
    dims = json.loads(next(iter(cell["dimensions"])))
    assert dims["originCode"] == "1101" and dims["originName"] == "국산"
    assert dims["productLargeClassCode"] == "10" and dims["productMediumClassCode"] == "01"
    assert dims["subclassCode"] == "01" and dims["subclassName"] == "배추"
    assert dims["gradeCode"] == "01" and dims["gradeName"] == "특"
    assert cell["signedNetAmount"] == "100"
    assert result["selectedResponseSha256"] == hashes
    assert result["limitations"] == ["inbound t vs settled netkg not comparable", "R01 row inclusion/aggregation/vintage unknown"]


def test_missing_or_incomplete_broad_scope_fails_closed():
    states = {(d, m): "complete" for d in r01.DATES for m in dashboard.MARKETS}
    states[(r01.DATES[0], next(iter(dashboard.MARKETS)))] = "failed"
    with pytest.raises(ValueError, match="incomplete broad query"):
        r01.summarize_rows([], states, {})


def test_official_comparison_values_are_frozen():
    assert [r01.OFFICIAL[d]["priceWonPerKg"] for d in r01.DATES] == [1164, 1164, 1277]
    assert [r01.OFFICIAL[d]["inboundTonnes"] for d in r01.DATES] == [1743, 1349, 1758]


def test_classification_matches_product_scope_and_decimal_gates():
    market = next(iter(dashboard.MARKETS))
    q, row = query(market), payload()
    assert r01._classify_payload(q, row, market, set()) is None
    wrong_product = dict(q, **{"cond[gds_mclsf_cd::EQ]": "99"})
    assert r01._classify_payload(wrong_product, row, market, set()) == "query_product_mismatch"
    wrong_scope = dict(row, **{"whsl_mrkt_cd": "other"})
    assert r01._classify_payload(q, wrong_scope, market, set()) == "market_mismatch"
    bad_amount = dict(row, **{"totprc": "bad"})
    assert r01._classify_payload(q, bad_amount, market, set()) == "invalid_signed_amount"
    bad_weight = dict(row, **{"unit_tot_qty": "bad"})
    assert r01._classify_payload(q, bad_weight, market, set()) == "invalid_signed_net_kg"


def test_non_cabbage_selected_query_is_excluded_before_payload_reporting():
    assert r01._is_cabbage_query(query())
    assert not r01._is_cabbage_query(dict(query(), **{"cond[gds_mclsf_cd::EQ]": "07"}))


def test_nullable_selector_states_are_json_serializable_with_sorted_keys():
    provenance = {"scopeCount": 3, "scopesSha256": "digest", "sourceSelectionPolicy": "validated",
                  "selectedStates": {"complete": 2, "empty": 0, None: 1}}
    safe = r01._json_safe_broad_provenance(provenance)
    encoded = json.dumps({"broadSelection": safe}, sort_keys=True, allow_nan=False)
    assert '"ambiguous_or_unavailable": 1' in encoded


def test_missing_dimension_codes_and_names_remain_explicit_nulls():
    market = next(iter(dashboard.MARKETS))
    row = payload()
    for field in ("plor_cd", "plor_nm", "grd_cd", "gds_sclsf_nm"):
        row.pop(field)
    states = {(d, m): "complete" for d in r01.DATES for m in dashboard.MARKETS}
    totals = {(d, m): {"signedNetAmount": 0, "signedNetWeight": 0, "eligibleAuctionKgRows": 1
        if d == r01.DATES[0] and m == market else 0} for d in r01.DATES for m in dashboard.MARKETS}
    result = r01.summarize_rows([{"date": r01.DATES[0], "market": market, "payload": row,
        "queryProduct": "cabbage", "exclusionReason": None}], states, totals)
    dims = json.loads(next(iter(result["dates"][r01.DATES[0]]["markets"][market]["dimensions"])))
    assert dims["originCode"] is None and dims["originName"] is None
    assert dims["gradeCode"] is None and dims["subclassName"] is None
