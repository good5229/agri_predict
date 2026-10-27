import datetime as dt
import hashlib
import json

import pytest

from agri_data import garak_reservation_pilot as pilot


class Response:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code = status
        self.body = body
        self.headers = headers or {"Content-Type": "application/json", "X-Test": "ok"}
        self.closed = False

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), 5):
            yield self.body[i:i+5]

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.headers = {}
        self.verify = False

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


def _json(data):
    return json.dumps({"data": data}, ensure_ascii=False).encode()


def _now():
    return dt.datetime(2026, 10, 8, 1, 2, 3, tzinfo=dt.timezone.utc)


def test_fixed_request_matrix_and_dry_run(capsys):
    assert len(pilot.request_matrix()) == 8
    assert {(x["marketKind"], x["dueDate"]) for x in pilot.request_matrix()} == {
        (m, day) for m in (1, 2) for day in pilot.DUE_DATES
    }
    assert pilot.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["requestSent"] is False
    assert plan["requestCount"] == 8


def test_validation_distinguishes_empty_from_errors_and_keeps_source_values():
    empty = pilot.validate_response(_json([]), 200)
    assert empty["state"] == "normal_empty"
    assert empty["itemCount"] == 0
    assert pilot.validate_response(json.dumps({"message": "no warning", "data": []}).encode(), 200)["state"] == "normal_empty"
    error_with_data = pilot.validate_response(json.dumps({"error": "denied", "data": []}).encode(), 200)
    assert error_with_data["state"] == "error_payload"
    assert pilot.validate_response(b"", 200)["state"] == "empty_response"
    assert pilot.validate_response(b"<html>denied</html>", 200)["state"] == "error_or_non_json_response"
    assert pilot.validate_response(b'{"error":"blocked"}', 403)["state"] == "http_error"
    body = _json([
        {"pumjongNm": "배추", "pumjongCd": "10", "totqty": "12.50", "alltotqty": "90"},
        {"pumjongNm": "무", "pumjongCd": "11", "totqty": "-2", "alltotqty": None},
        {"pumjongNm": "기타", "pumjongCd": "99", "totqty": "NaN", "alltotqty": "100"},
    ])
    result = pilot.validate_response(body, 200)
    assert result["state"] == "data_present"
    assert result["names"] == ["기타", "무", "배추"]
    assert result["cabbagePresent"] and result["radishPresent"]
    assert result["items"][1]["sourceValues"]["totqty"] == "-2"
    assert "totqty_negative_value" in result["items"][1]["issues"]
    assert result["items"][2]["sourceValues"]["totqty"] == "NaN"
    assert "totqty_nonfinite_value" in result["items"][2]["issues"]
    lexical = pilot.validate_response(_json([{"pumjongNm": "열무", "pumjongCd": "11",
                                               "totqty": "1", "alltotqty": "1"}]), 200)
    assert lexical["radishPresent"] is True
    assert "literal substring scan" in lexical["namePresenceSemantics"]


def test_null_name_and_code_remain_preserved_and_count_as_questionable():
    result = pilot.validate_response(_json([{"pumjongNm": None, "pumjongCd": None,
                                             "totqty": "600", "alltotqty": "600"}]), 200)
    record = result["items"][0]
    assert record["sourceValues"]["pumjongNm"] is None
    assert record["sourceValues"]["pumjongCd"] is None
    assert "pumjongNm_null_value" in record["issues"]
    assert "pumjongCd_null_value" in record["issues"]
    assert result["questionableItemCount"] == 1
    assert result["itemIssueCount"] == 2
    assert result["names"] == []
    assert result["cabbagePresent"] is False and result["radishPresent"] is False


def test_null_and_empty_quantities_receive_one_numeric_issue_each():
    result = pilot.validate_response(_json([{"pumjongNm": "배추", "pumjongCd": "10",
                                             "totqty": None, "alltotqty": ""}]), 200)
    record = result["items"][0]
    assert record["sourceValues"]["totqty"] is None
    assert record["sourceValues"]["alltotqty"] == ""
    assert record["issues"].count("totqty_null_value") == 1
    assert record["issues"].count("alltotqty_null_value") == 1
    assert len(record["issues"]) == 2
    assert result["itemIssueCount"] == 2


def test_collect_records_exact_eight_requests_and_raw_provenance(tmp_path):
    payload = _json([{"pumjongNm": "배추", "pumjongCd": "10", "totqty": "23.0", "alltotqty": "90.0"}])
    responses = [Response(200, payload) for _ in range(8)]
    session = Session(responses)
    result = pilot.collect(tmp_path, session=session, now=_now,
                           today_kst=lambda: pilot.EXECUTION_DATE, sleep=lambda _: None)
    assert len(result) == 8
    assert len(session.calls) == 8
    assert session.verify is True
    assert session.headers["User-Agent"] == "Mozilla/5.0"
    for index, (url, kwargs) in enumerate(session.calls):
        assert url == pilot.ENDPOINT
        assert kwargs["verify"] is True
        assert kwargs["timeout"] == pilot.TIMEOUT_SECONDS
        assert kwargs["data"]["dueDate"] == pilot.DUE_DATES[index % 4]
        assert kwargs["data"]["marketKind"] == (1 if index < 4 else 2)
        assert "goodCd" not in kwargs["data"]
    manifests = list(tmp_path.glob("*.manifest.json"))
    raws = list(tmp_path.glob("*.raw"))
    assert len(manifests) == 8 and len(raws) == 8
    for meta_path in manifests:
        meta = json.loads(meta_path.read_text())
        assert set(meta["requestParams"]) == {"dueDate", "marketKind"}
        raw = (tmp_path / meta["rawFile"]).read_bytes()
        assert raw == payload
        assert meta["responseBytes"] == len(raw)
        assert meta["sha256"] == hashlib.sha256(raw).hexdigest()
        assert dt.datetime.fromisoformat(meta["retrievedAtUtc"]).tzinfo is not None
        assert meta["result"]["cabbagePresent"] is True


def test_retries_keep_every_attempt_and_distinguish_states(tmp_path):
    payload = _json([])
    session = Session([Response(503, b"busy"), Response(200, payload)] +
                      [Response(200, payload) for _ in range(7)])
    result = pilot.collect(tmp_path, session=session, now=_now,
                           today_kst=lambda: pilot.EXECUTION_DATE, sleep=lambda _: None)
    assert len(session.calls) == 9
    assert len(result[0]["attempts"]) == 2
    first = result[0]["attempts"][0]
    assert first["httpStatus"] == 503
    assert first["result"]["state"] == "http_error"
    assert (tmp_path / first["rawFile"]).read_bytes() == b"busy"
    assert result[0]["state"] == "normal_empty"
    assert len(list(tmp_path.glob("*.manifest.json"))) == 9


def test_execution_date_guard_prevents_any_post(tmp_path):
    session = Session([])
    with pytest.raises(pilot.CollectionError, match="only_on_2026-10-08_KST"):
        pilot.collect(tmp_path, session=session, today_kst=lambda: dt.date(2026, 10, 9),
                      sleep=lambda _: None)
    assert session.calls == []


def test_execution_date_rollover_stops_before_next_post_and_keeps_attempt(tmp_path):
    session = Session([Response(200, _json([])) for _ in range(8)])
    dates = iter([pilot.EXECUTION_DATE, pilot.EXECUTION_DATE, pilot.EXECUTION_DATE,
                  dt.date(2026, 10, 9)])
    with pytest.raises(pilot.CollectionError, match="only_on_2026-10-08_KST"):
        pilot.collect(tmp_path, session=session, now=_now, today_kst=lambda: next(dates),
                      sleep=lambda _: None)
    assert len(session.calls) == 1
    assert len(list(tmp_path.glob("*.manifest.json"))) == 1
    assert len(list(tmp_path.glob("*.raw"))) == 1


def test_truncated_and_stream_interrupted_states_are_flagged():
    oversized = Response(200, b"x" * (pilot.MAX_RESPONSE_BYTES + 1))
    body, truncated, error = pilot._read_bounded(oversized)
    assert len(body) == pilot.MAX_RESPONSE_BYTES
    assert truncated and "2mb" in error
    validation = pilot.validate_response(body, 200, truncated=truncated, transport_error=error)
    assert validation["state"] == "truncated"

    class Interrupted(Response):
        def iter_content(self, chunk_size):
            yield b"prefix"
            import requests
            raise requests.ConnectionError("stream ended")

    body, truncated, error = pilot._read_bounded(Interrupted())
    assert body == b"prefix"
    assert truncated and "stream_error" in error


def test_concise_output_does_not_expose_raw_items():
    item = {"market": "garak", "dueDate": pilot.DUE_DATES[0], "marketKind": 1,
            "state": "data_present", "complete": False,
            "attempts": [{"result": {"itemCount": 1, "positiveQuantityCount": 1,
                                      "questionableItemCount": 0, "itemIssueCount": 0,
                                      "names": ["배추"], "cabbagePresent": True,
                                      "radishPresent": False,
                                      "namePresenceSemantics": "literal substring scan"}}]}
    output = json.dumps(pilot._concise([item]), ensure_ascii=False)
    assert "rawFile" not in output and "sourceValues" not in output
    assert "cabbagePresent" in output and "radishPresent" in output
