import datetime as dt
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import requests

from agri_data import ecmwf_operational_probe as probe


def grib_message():
    trailer = b"7777"
    payload = b"test-field"
    length = 16 + len(payload) + len(trailer)
    return b"GRIB" + b"\x00\x00\x00\x02" + length.to_bytes(8, "big") + payload + trailer


def index_body(offset=100, length=None, *, extra_records=0):
    field = grib_message()
    record = {**probe.EXPECTED, "_offset": offset, "_length": len(field) if length is None else length}
    rows = [json.dumps(record)] * (extra_records + 1)
    return ("\n".join(rows) + "\n").encode(), field, record


class FakeResponse:
    def __init__(self, body=b"", status=200, headers=None, chunks=None):
        self.body = body
        self.status_code = status
        self.headers = headers or {}
        self.chunks = chunks
        self.closed = False

    def iter_content(self, chunk_size):
        if self.chunks is not None:
            yield from self.chunks
        elif self.body:
            yield self.body

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def response_set(index=None, field=None, *, index_status=200, head_status=200,
                 range_status=206, content_range=None, object_size=None):
    index_bytes, default_field, record = index or index_body()
    field_bytes = default_field if field is None else field
    object_size = object_size or (record.get("_offset", 0) + record.get("_length", len(field_bytes)) + 10)
    if content_range is None:
        content_range = f"bytes {record.get('_offset')}-{record.get('_offset', 0) + record.get('_length', 0) - 1}/{object_size}"
    return [
        FakeResponse(index_bytes, index_status, {"Content-Type": "application/x-ndjson"}),
        FakeResponse(b"", head_status, {"Content-Length": str(object_size),
                                        "Last-Modified": "Sun, 31 Aug 2025 12:00:00 GMT",
                                        "ETag": '"sample"'}),
        FakeResponse(field_bytes, range_status, {"Content-Range": content_range,
                                                 "Content-Length": str(len(field_bytes))}),
    ], record, field_bytes, object_size


def run_probe(tmp_path, responses, *, now=None, source="aws"):
    session = FakeSession(responses)
    result = probe.probe(raw_dir=tmp_path, session=session,
                         now=now or dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc),
                         sleeper=lambda _: None, source=source)
    return result, session


def test_dry_run_sends_no_request(capsys):
    with mock.patch.object(probe, "probe", side_effect=AssertionError("network requested")):
        assert probe.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["request_sent"] is False
    assert result["expected_index_fields"]["param"] == "2t"


def test_dry_run_cloudfront_selects_fixed_mirror_without_request(capsys):
    with mock.patch.object(probe, "probe", side_effect=AssertionError("network requested")):
        assert probe.main(["--cloudfront"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["source"] == "cloudfront"
    assert result["index_url"].startswith("https://d2zvc0wgha4k2l.cloudfront.net/")
    assert result["request_sent"] is False


def test_dry_run_google_selects_fixed_mirror_without_request(capsys):
    with mock.patch.object(probe, "probe", side_effect=AssertionError("network requested")):
        assert probe.main(["--google"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["source"] == "google"
    assert result["index_url"] == (
        "https://storage.googleapis.com/ecmwf-open-data/" + probe.OBJECT_KEY + ".index")
    assert result["request_sent"] is False


def test_success_captures_index_and_exact_grib_range_with_provenance(tmp_path):
    responses, record, field, size = response_set()
    result, session = run_probe(tmp_path, responses)
    assert result["complete"] is True
    assert result["issues"] == []
    assert result["selected_record"] == record
    assert result["head"]["last_modified"] == "Sun, 31 Aug 2025 12:00:00 GMT"
    assert result["range"]["requested_start"] == record["_offset"]
    assert result["range"]["requested_end"] == record["_offset"] + record["_length"] - 1
    raw_index = (tmp_path / result["index"]["raw_file"]).read_bytes()
    raw_field = (tmp_path / result["range"]["raw_file"]).read_bytes()
    assert raw_index == responses[0].body
    assert raw_field == field
    assert result["range"]["sha256"] == hashlib.sha256(field).hexdigest()
    assert [call[0] for call in session.calls] == ["GET", "HEAD", "GET"]
    assert session.calls[-1][2]["headers"]["Range"] == f"bytes={record['_offset']}-{record['_offset'] + record['_length'] - 1}"
    assert all(call[2]["verify"] is True and call[2]["allow_redirects"] is False
               for call in session.calls)
    saved = json.loads((tmp_path / result["metadata_file"]).read_text())
    assert saved["complete"] is True


def test_explicit_cloudfront_execution_uses_same_key_and_records_source(tmp_path):
    responses, _, _, _ = response_set()
    result, session = run_probe(tmp_path, responses, source="cloudfront")
    assert result["complete"] is True
    assert result["source"] == "cloudfront"
    assert result["index_url"].startswith("https://d2zvc0wgha4k2l.cloudfront.net/")
    assert result["grib_url"].startswith("https://d2zvc0wgha4k2l.cloudfront.net/")
    assert all("20250831/12z/ifs/0p25/oper/20250831120000-24h-oper-fc" in call[1]
               for call in session.calls)


def test_explicit_google_execution_retrieves_and_records_source(tmp_path):
    responses, _, field, _ = response_set()
    result, session = run_probe(tmp_path, responses, source="google")
    expected_base = "https://storage.googleapis.com/ecmwf-open-data/" + probe.OBJECT_KEY
    assert result["complete"] is True
    assert result["source"] == "google"
    assert result["index_url"] == expected_base + ".index"
    assert result["grib_url"] == expected_base + ".grib2"
    assert [call[1] for call in session.calls] == [expected_base + ".index",
                                                   expected_base + ".grib2",
                                                   expected_base + ".grib2"]
    assert (tmp_path / result["range"]["raw_file"]).read_bytes() == field
    assert json.loads((tmp_path / result["metadata_file"]).read_text())["source"] == "google"


def test_index_http_error_is_preserved_and_never_complete(tmp_path):
    responses, _, _, _ = response_set(index_status=503)
    responses = [FakeResponse(b"service unavailable", 503, {"Content-Type": "text/plain"})]
    result, session = run_probe(tmp_path, responses)
    assert result["complete"] is False
    assert "index_http_status_not_success" in result["issues"]
    assert (tmp_path / result["index"]["raw_file"]).read_bytes() == b"service unavailable"
    assert len(session.calls) == 1


def test_only_transport_failures_retry_and_retries_are_bounded(tmp_path):
    session = FakeSession([requests.Timeout("temporary"), requests.ConnectionError("temporary"),
                           requests.Timeout("final"),])
    result = probe.probe(raw_dir=tmp_path, session=session,
                         now=dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc), sleeper=lambda _: None)
    assert result["complete"] is False
    assert result["attempts"]["index_get"] == probe.MAX_ATTEMPTS
    assert len(session.calls) == probe.MAX_ATTEMPTS
    assert "transport_failed_after_bounded_retries" in result["issues"]
    assert "raw_file" not in result["index"]


@pytest.mark.parametrize("status,content_range,expected_issue", [
    (200, "bytes 100-125/200", "range_http_status_not_206"),
    (206, "bytes 101-126/200", "content_range_mismatch"),
])
def test_range_status_or_content_range_mismatch_fails(tmp_path, status, content_range, expected_issue):
    responses, _, _, _ = response_set(range_status=status, content_range=content_range)
    result, _ = run_probe(tmp_path, responses)
    assert result["complete"] is False
    assert expected_issue in result["issues"]


def test_matching_index_field_must_be_unique(tmp_path):
    body, _, _ = index_body(extra_records=1)
    responses, _, _, _ = response_set(index=(body, grib_message(), json.loads(body.splitlines()[0])))
    result, session = run_probe(tmp_path, responses[:1])
    assert result["complete"] is False
    assert result["issues"] == ["matching_index_record_not_unique"]
    assert len(session.calls) == 1


@pytest.mark.parametrize("body,issue", [
    (b"{not json}\n", "index_invalid_json_line_1"),
    (json.dumps({**probe.EXPECTED, "_offset": -1, "_length": 20}).encode() + b"\n", "index_offset_invalid"),
    (json.dumps({**probe.EXPECTED, "_offset": 5, "_length": 0}).encode() + b"\n", "index_length_invalid"),
])
def test_malformed_index_and_offsets_fail_before_head(tmp_path, body, issue):
    result, session = run_probe(tmp_path, [FakeResponse(body)])
    assert result["complete"] is False
    assert issue in result["issues"]
    assert len(session.calls) == 1


def test_partial_field_stream_is_saved_and_incomplete(tmp_path):
    responses, _, field, _ = response_set()
    responses[-1] = FakeResponse(status=206,
                                 headers=responses[-1].headers,
                                 chunks=[field[:11], requests.ConnectionError("interrupted")])
    # FakeResponse yields the exception as a chunk; use a response that raises while iterating.
    class PartialResponse(FakeResponse):
        def iter_content(self, chunk_size):
            yield field[:11]
            raise requests.ConnectionError("interrupted")
    responses[-1] = PartialResponse(status=206, headers=responses[-1].headers)
    result, _ = run_probe(tmp_path, responses)
    assert result["complete"] is False
    assert "range_response_stream_failed" in result["issues"]
    assert (tmp_path / result["range"]["raw_file"]).read_bytes() == field[:11]


def test_bad_grib_magic_or_declared_length_never_completes(tmp_path):
    responses, _, field, _ = response_set()
    corrupted = b"NOPE" + field[4:]
    responses[-1] = FakeResponse(corrupted, 206, responses[-1].headers)
    result, _ = run_probe(tmp_path, responses)
    assert "grib_magic_missing" in result["issues"]
    assert result["complete"] is False
