import datetime as dt
import hashlib
import json

from agri_data import nongnet_national_volume_pilot as pilot


class Response:
    def __init__(self, status, body, headers=None):
        self.status_code = status
        self.body = body
        self.headers = headers or {"Content-Type": "application/json", "X-Test": "mock"}
        self.closed = False

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), 11):
            yield self.body[i:i+11]

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.headers = {}
        self.verify = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


def _json(rows):
    return json.dumps({"datalist": rows}, ensure_ascii=False).encode()


def _row(date="20260911", price="1164", weight="1743", **more):
    return {"saledate": date, "selectDay": price, "selectDayWgt": weight,
            "selectDayMax": "349", "selectDayMin": "100", **more}


def _now():
    return dt.datetime(2026, 10, 8, tzinfo=dt.timezone.utc)


def test_fixed_three_query_matrix_and_dry_run(capsys):
    matrix = pilot.request_matrix()
    assert len(matrix) == 3
    assert [p["searchDate"] for p in matrix] == ["2026년 09월 11일", "2026년 09월 12일", "2026년 09월 14일"]
    assert all(p == {"searchUnitGrade": "normal", "searchDate": p["searchDate"],
                     "searchSymbol1": "100000", "searchSymbol2": "1001", "searchSymbol3": ""}
               for p in matrix)
    assert pilot.main([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["requestSent"] is False and output["requestCount"] == 3


def test_valid_unique_date_preserves_and_separates_average_price_and_weight():
    result = pilot.validate_json(_json([_row()]), "2026-09-11")
    assert result["state"] == "valid"
    assert result["matchingDateCount"] == 1
    assert result["matchingRow"]["selectDay"] == "1164"
    assert result["matchingRow"]["selectDayWgt"] == "1743"
    assert result["rows"][0]["sourceValues"]["selectDayMax"] == "349"
    assert result["matchingRow"]["selectDayWgt"] != result["rows"][0]["sourceValues"]["selectDayMax"]
    with_other_issue = pilot.validate_json(_json([_row(), _row("20260912", weight=None)]), "2026-09-11")
    assert with_other_issue["state"] == "questionable"
    assert with_other_issue["rows"][1]["sourceValues"]["selectDayWgt"] is None
    assert "selectDayWgt_null_value" in with_other_issue["rows"][1]["issues"]


def test_no_date_duplicate_date_and_negative_fields_are_classified():
    no_date = pilot.validate_json(_json([_row("20260912")]), "2026-09-11")
    assert no_date["state"] == "requested_date_missing"
    assert no_date["rows"][0]["sourceValues"]["saledate"] == "20260912"
    duplicate = pilot.validate_json(_json([_row(), _row(price="1170")]), "2026-09-11")
    assert duplicate["state"] == "duplicate_requested_date"
    negative = pilot.validate_json(_json([_row(price="-1", weight="-1743")]), "2026-09-11")
    assert negative["state"] == "questionable"
    assert negative["matchingRow"]["selectDay"] == "-1"
    assert "selectDay_negative_value" in negative["matchingRow"]["issues"]
    assert "selectDayWgt_negative_value" in negative["matchingRow"]["issues"]


def test_non_string_saledate_is_flagged_without_crashing():
    for saledate, expected_issue in ((20260911, "saledate_not_string"),
                                     (None, "saledate_null_value")):
        result = pilot.validate_json(_json([_row(date=saledate)]), "2026-09-11")
        assert result["state"] == "questionable"
        assert result["matchingRow"] is None
        assert result["rows"][0]["dateMatch"] is False
        assert expected_issue in result["rows"][0]["issues"]


def test_non_object_row_makes_response_questionable_even_with_match():
    result = pilot.validate_json(_json([_row(), "unexpected"]), "2026-09-11")
    assert result["state"] == "questionable"
    assert result["matchingDateCount"] == 1
    assert result["matchingRow"]["saledate"] == "20260911"
    assert "row_1_questionable" in result["issues"]


def test_malformed_error_and_empty_responses_are_distinct():
    assert pilot.validate_json(b"", "2026-09-11")["state"] == "empty_response"
    assert pilot.validate_json(b"<html>error</html>", "2026-09-11")["state"] == "non_json_response"
    assert pilot.validate_json(b'{"error":"denied","datalist":[]}', "2026-09-11")["state"] == "error_payload"
    assert pilot.validate_json(b'{"data":[]}', "2026-09-11")["state"] == "invalid_json_structure"
    assert pilot.validate_json(_json([]), "2026-09-11")["state"] == "normal_empty"


def test_collect_persists_each_raw_attempt_and_manifest(tmp_path):
    payloads = [_json([_row(date=d.replace("-", ""))]) for d in pilot.DATES]
    session = Session([Response(200, body) for body in payloads])
    results = pilot.collect(tmp_path, session=session, now=_now, sleep=lambda _: None)
    assert len(results) == 3 and len(session.calls) == 3
    assert session.verify is True and session.headers["User-Agent"] == "Mozilla/5.0"
    for (url, kwargs), target, raw_expected in zip(session.calls, pilot.request_matrix(), payloads):
        assert url == pilot.ENDPOINT
        assert kwargs["verify"] is True and kwargs["timeout"] == pilot.TIMEOUT_SECONDS
        assert kwargs["params"] == target
    paths = list(tmp_path.glob("*.manifest.json"))
    assert len(paths) == 3
    for path in paths:
        manifest = json.loads(path.read_text())
        raw = (tmp_path / manifest["rawFile"]).read_bytes()
        assert len(raw) == manifest["responseBytes"]
        assert hashlib.sha256(raw).hexdigest() == manifest["sha256"]
        assert manifest["complete"] is False
        assert dt.datetime.fromisoformat(manifest["retrievedAtUtc"]).tzinfo is not None


def test_bounded_reader_keeps_prefix_and_marks_truncated():
    response = Response(200, b"x" * (pilot.MAX_RESPONSE_BYTES + 1))
    body, truncated, error = pilot._read_bounded(response)
    assert len(body) == pilot.MAX_RESPONSE_BYTES
    assert truncated and "3mb" in error
    assert response.closed
