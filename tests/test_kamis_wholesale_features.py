import datetime as dt
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path

import pytest

from agri_data import robust_price_features as baseline
from agri_data.kamis_wholesale_export import parse_export
from agri_data.kamis_wholesale_features import (
    CROPS, KG_DIVISORS, _json_bytes, _mean_per_kg, _write_exclusive, features,
    validate_query_checkpoint, validate_archive, _entry_gate,
)
from agri_data.kamis_wholesale_history import build_query_plan


def _auction_history(end=dt.date(2024, 12, 31), days=500):
    records, direct = {}, {}
    for i in range(days):
        day = end - dt.timedelta(days=days - 1 - i)
        price = 100.0 + i * 0.05
        records[day] = {"price": price, "volume": 1000.0 + i % 11,
                        "rowCount": 10 + i % 5, "markets": frozenset({"110001", "110008"})}
        direct[day] = (price, 1000.0 + i % 11)
    return records, direct


def _survey_record(crop="cabbage", raw="1,000", day="2024-12-31", *, unit=None,
                   status="observed", flags=(), eligible=True, decimal="1000"):
    return {"date": day, "crop": crop, "meanStatus": status, "meanFlags": list(flags),
            "meanRaw": raw, "meanDecimal": decimal,
            "originalUnit": unit or CROPS[crop]["unit"], "kgDivisor": KG_DIVISORS[crop],
            "eligiblePositiveMean": eligible, "queryId": f"{crop}-query",
            "sourceSha256": "a" * 64}


def test_seven_survey_features_preserve_106_prefix_and_original_unit_provenance():
    records, direct = _auction_history()
    target, horizon = dt.date(2025, 1, 1), 1
    cutoff = target - dt.timedelta(days=2)
    survey = {
        cutoff: _survey_record(day=cutoff.isoformat(), raw="1,000"),
        cutoff - dt.timedelta(days=7): _survey_record(day=(cutoff-dt.timedelta(days=7)).isoformat(), raw="900", decimal="900"),
        cutoff - dt.timedelta(days=28): _survey_record(day=(cutoff-dt.timedelta(days=28)).isoformat(), raw="800", decimal="800"),
    }
    base, p0, _ = baseline.features(records, direct, target, horizon)
    vector, returned_p0, audit = features("cabbage", records, direct, survey, target, horizon)
    assert len(vector) == 113 and vector[:106] == base[:106] and returned_p0 == p0
    assert vector[106] == 1 and vector[108] == 0
    assert vector[109] == 1 and vector[111] == 1
    assert vector[110] == pytest.approx(math.log(100 / 90))
    assert vector[112] == pytest.approx(math.log(100 / 80))
    assert audit["anchors"]["0"]["meanRaw"] == "1,000"
    assert audit["anchors"]["0"]["originalUnit"] == "10kg(그물망 3포기)"
    assert audit["anchors"]["0"]["kgDivisor"] == 10
    radish = _survey_record("radish", raw="1,000", unit="20kg")
    assert _mean_per_kg(radish, "radish") == 50


def test_missing_and_stale_survey_anchors_are_masked_without_forward_fill():
    records, direct = _auction_history()
    target = dt.date(2025, 1, 1)
    no_survey, _, audit = features("cabbage", records, direct, {}, target, 1)
    assert no_survey[106:] == [0.0] * 7
    assert not audit["anchors"]["0"]["available"]
    old_day = dt.date(2024, 10, 1)
    stale = {old_day: _survey_record(day=old_day.isoformat())}
    vector, _, stale_audit = features("cabbage", records, direct, stale, target, 1)
    assert vector[106:] == [0.0] * 7
    assert stale_audit["anchors"]["0"]["ageDays"] > 14


def test_future_survey_auction_poison_does_not_change_causal_features():
    records, direct = _auction_history()
    target = dt.date(2025, 1, 1)
    cutoff = target - dt.timedelta(days=2)
    prior = {cutoff: _survey_record(day=cutoff.isoformat())}
    expected, _, _ = features("cabbage", records, direct, prior, target, 1)
    class Poison(dict):
        def get(self, *args, **kwargs): raise AssertionError("future survey value was inspected")
        def __getitem__(self, key): raise AssertionError("future survey value was inspected")
    poison_day = cutoff + dt.timedelta(days=1)
    with_poison = dict(prior); with_poison[poison_day] = Poison()
    poisoned_records = dict(records); poisoned_records[poison_day] = Poison()
    poisoned_direct = dict(direct); poisoned_direct[poison_day] = Poison()
    actual, _, _ = features("cabbage", poisoned_records, poisoned_direct, with_poison, target, 1)
    assert actual == expected


def test_base_unavailable_makes_whole_feature_vector_unavailable():
    vector, p0, audit = features("cabbage", {}, {},
                                 {dt.date(2024, 12, 31): _survey_record()},
                                 dt.date(2025, 1, 1), 1)
    assert vector is None and p0 is None and audit["status"] != "available"


def test_diagnostic_entry_gate_requires_joint_support_and_nonconstant_signal():
    assert not _entry_gate([0.1] * 29)["eligibleForLaterReview"]
    enough_but_constant = _entry_gate([0.1] * 30)
    assert not enough_but_constant["eligibleForLaterReview"]
    eligible = _entry_gate([0.1] * 29 + [0.2])
    assert eligible["eligibleForLaterReview"] and eligible["jointBaseAndQ0Origins"] == 30


def test_mean_flags_nonpositive_and_profile_mismatches_are_never_features():
    flagged = _survey_record(flags=["negative_price"])
    assert _mean_per_kg(flagged, "cabbage") is None
    zero = _survey_record(raw="0", decimal="0")
    assert _mean_per_kg(zero, "cabbage") is None
    wrong_unit = _survey_record(unit="20kg")
    with pytest.raises(ValueError, match="unit/divisor"):
        _mean_per_kg(wrong_unit, "cabbage")
    changed_decimal = _survey_record(raw="1,000", decimal="999")
    with pytest.raises(ValueError, match="raw mean/decimal"):
        _mean_per_kg(changed_decimal, "cabbage")


def _write_one_checkpoint(directory: Path, *, duplicate=False):
    query = build_query_plan(dt.date(2024, 1, 1), dt.date(2024, 1, 2), ["cabbage"], korean_language=True)[0]
    dates = ["24-01-02", "24-01-01"] if not duplicate else ["24-01-02", "24-01-02"]
    body = ("<table id='itemTable_1'><caption>중도매인 판매가격 채소류, 배추, 전체, 상품, 10kg(그물망 3포기) 기간별/일간 가격정보</caption>"
            "<thead><tr><th>구분</th><th>평균</th><th>등락율</th><th>서울</th><th>부산</th><th>대구</th><th>광주</th><th>대전</th></tr></thead><tbody>" +
            "".join(f"<tr><th>{day}</th><td>1,000</td><td>0.0</td><td>1,000</td><td>1,000</td><td>1,000</td><td>1,000</td><td>1,000</td></tr>" for day in dates) +
            "</tbody></table>").encode()
    digest = hashlib.sha256(body).hexdigest()
    body_name = f"{query['id']}-attempt-1.body"
    (directory / body_name).write_bytes(body)
    parse_body = body
    if duplicate:
        valid_body = body.replace(b"24-01-02</th><td>1,000", b"24-01-01</th><td>1,000", 1)
        parse_body = valid_body
    parsed = parse_export(parse_body, start=dt.date(2024, 1, 1), end=dt.date(2024, 1, 2),
                          item="배추", variety="전체", grade="상품", unit="10kg(그물망 3포기)", table_id="itemTable_1")
    parsed["requested"]["crop"] = "cabbage"
    parsed["qualityStatus"] = "clear"
    rows_bytes = _json_bytes(parsed)
    rows_name = f"query-{query['id']}.rows.json"
    (directory / rows_name).write_bytes(rows_bytes)
    response = {"bodyFile": body_name, "bodyBytesCaptured": len(body), "bodySha256": digest,
                "httpStatus": 200, "outcome": "http_success", "framingComplete": True, "attempt": 1,
                "requestUrl": query["url"], "responseUrl": query["url"]}
    meta = {"id": query["id"], "crop": "cabbage", "start": query["start"], "end": query["end"],
            "method": query["method"], "url": query["url"], "requestBody": query["requestBody"],
            "requestSha256": query["requestSha256"], "fields": query["formFields"],
            "requestHeaders": query["requestHeaders"], "state": "completed", "httpStatus": 200,
            "responses": [response], "sourceBodyFile": body_name, "sourceSha256": digest,
            "rowsFile": rows_name, "rowsSha256": hashlib.sha256(rows_bytes).hexdigest()}
    (directory / f"query-{query['id']}.json").write_bytes(_json_bytes(meta))
    return query, body, rows_bytes


def test_query_checkpoint_validates_source_hash_and_reparsed_rows(tmp_path):
    query, _body, _rows = _write_one_checkpoint(tmp_path)
    parsed, manifest = validate_query_checkpoint(tmp_path, query)
    assert len(parsed["rawRows"]) == 2
    assert manifest["sourceSha256"] == hashlib.sha256(_body).hexdigest()
    path = tmp_path / f"query-{query['id']}.rows.json"
    saved = json.loads(path.read_text()); saved["rawRows"][0]["priceRaw"] = "999"
    tampered = _json_bytes(saved); path.write_bytes(tampered)
    meta_path = tmp_path / f"query-{query['id']}.json"
    meta = json.loads(meta_path.read_text()); meta["rowsSha256"] = hashlib.sha256(tampered).hexdigest()
    meta_path.write_bytes(_json_bytes(meta))
    with pytest.raises(ValueError, match="differ from reparsed source"):
        validate_query_checkpoint(tmp_path, query)


def test_query_checkpoint_rejects_duplicate_source_dates_and_profile_mismatch(tmp_path):
    query, _body, _rows = _write_one_checkpoint(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="duplicate date"):
        validate_query_checkpoint(tmp_path, query)
    query["item"] = "무"
    with pytest.raises(ValueError, match="item mismatch"):
        validate_query_checkpoint(tmp_path, query)


def test_validate_archive_builds_records_and_eligible_means_from_korean_average_column(tmp_path):
    queries = build_query_plan(dt.date(2014, 1, 1), dt.date(2026, 9, 30),
                               list(CROPS), korean_language=True)
    (tmp_path / "plan.json").write_bytes(_json_bytes({"queries": queries}))
    headers = ("<th>구분</th><th>평균</th><th>등락율</th><th>서울</th><th>부산</th>"
               "<th>대구</th><th>광주</th><th>대전</th>")
    for query in queries:
        profile = CROPS[query["crop"]]
        body = (f"<table id='itemTable_1'><caption>중도매인 판매가격 채소류, {profile['item']}, "
                f"{profile['variety']}, 상품, {profile['unit']} 기간별/일간 가격정보</caption>"
                f"<thead><tr>{headers}</tr></thead><tbody><tr><th>{query['start']}</th>"
                "<td>1,000</td><td>0</td><td>1,000</td><td>1,000</td><td>1,000</td>"
                "<td>1,000</td><td>1,000</td></tr></tbody></table>").encode("utf-8")
        digest = hashlib.sha256(body).hexdigest()
        body_name = f"query-{query['id']}-attempt-1.body"
        (tmp_path / body_name).write_bytes(body)
        parsed = parse_export(body, start=dt.date.fromisoformat(query["start"]),
                              end=dt.date.fromisoformat(query["end"]), item=profile["item"],
                              variety=profile["variety"], grade="상품", unit=profile["unit"],
                              table_id="itemTable_1")
        parsed["requested"]["crop"] = query["crop"]
        parsed["qualityStatus"] = "clear"
        rows_bytes = _json_bytes(parsed)
        rows_name = f"query-{query['id']}.rows.json"
        (tmp_path / rows_name).write_bytes(rows_bytes)
        response = {"attempt": 1, "requestUrl": query["url"], "responseUrl": query["url"],
                    "httpStatus": 200, "outcome": "http_success", "bodyFile": body_name,
                    "bodyBytesCaptured": len(body), "bodySha256": digest, "framingComplete": True}
        metadata = {"id": query["id"], "crop": query["crop"], "start": query["start"],
                    "end": query["end"], "method": query["method"], "url": query["url"],
                    "requestBody": query["requestBody"], "requestSha256": query["requestSha256"],
                    "fields": query["formFields"], "requestHeaders": query["requestHeaders"],
                    "state": "completed", "httpStatus": 200, "responses": [response],
                    "sourceBodyFile": body_name, "sourceSha256": digest,
                    "rowsFile": rows_name, "rowsSha256": hashlib.sha256(rows_bytes).hexdigest()}
        (tmp_path / f"query-{query['id']}.json").write_bytes(_json_bytes(metadata))
    archive = validate_archive(tmp_path, expected_plan_sha256=None, expected_query_count=255)
    assert archive["planQueryCount"] == 255
    assert len(archive["recordsByCrop"]["cabbage"]) == 51
    assert archive["auditByCrop"]["cabbage"]["eligiblePositiveMean"] == 51
    first = next(iter(archive["recordsByCrop"]["cabbage"].values()))
    assert first["meanRaw"] == "1,000" and first["eligiblePositiveMean"]


def test_diagnostic_artifact_write_is_create_only(tmp_path):
    output = tmp_path / "nested" / "diagnostic.json"
    _write_exclusive(output, {"status": "ok"})
    assert json.loads(output.read_text()) == {"status": "ok"}
    with pytest.raises(FileExistsError):
        _write_exclusive(output, {"status": "changed"})
    assert json.loads(output.read_text()) == {"status": "ok"}
