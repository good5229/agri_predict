import datetime as dt
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import requests

from agri_data import ecmwf_archive_coverage_probe as probe


def index_body(date, *, offset=125, length=987, extra=0):
    record = {"domain": "g", "date": date, "time": "1200", "type": "fc",
              "stream": "oper", "step": "24", "levtype": "sfc", "param": "2t",
              "_offset": offset, "_length": length}
    return (("\n".join(json.dumps(record) for _ in range(extra + 1))) + "\n").encode()


class FakeResponse:
    def __init__(self, body=b"", status=200, headers=None, chunks=None):
        self.body, self.status_code, self.headers = body, status, headers or {}
        self.chunks = chunks
        self.closed = False

    def iter_content(self, chunk_size):
        if self.chunks is not None:
            for item in self.chunks:
                if isinstance(item, Exception):
                    raise item
                yield item
        elif self.body:
            yield self.body

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def _next(self, method, url, kwargs):
        self.calls.append((method, url, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, **kwargs):
        return self._next("GET", url, kwargs)

    def head(self, url, **kwargs):
        return self._next("HEAD", url, kwargs)


def success_pair(date, last_modified="Sun, 14 Sep 2025 20:00:00 GMT"):
    index = index_body(date)
    head = FakeResponse(status=200, headers={"Content-Length": "126472991",
                                             "Last-Modified": last_modified,
                                             "ETag": '"fixture"'})
    return [FakeResponse(index, 200, {"Content-Length": str(len(index)),
                                      "Content-Type": "application/x-ndjson"}), head]


def fixed_time():
    return dt.datetime(2026, 10, 7, 12, tzinfo=dt.timezone.utc)


def test_dry_run_lists_fixed_dates_six_requests_and_no_network(capsys):
    with mock.patch.object(probe, "probe", side_effect=AssertionError("network path invoked")):
        assert probe.main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dates"] == list(probe.DATES)
    assert payload["logical_requests_max"] == 6
    assert payload["max_http_attempts_including_transport_retries"] == 18
    assert payload["request_sent"] is False


def test_success_preserves_raw_index_hash_last_modified_and_deadline_flag(tmp_path):
    session = FakeSession(success_pair(probe.DATES[0]))
    result = probe.probe_date(probe.DATES[0], raw_dir=tmp_path, session=session,
                              retrieved_at=fixed_time(), sleeper=lambda _: None)
    raw = (tmp_path / result["index"]["raw_file"]).read_bytes()
    metadata = json.loads((tmp_path / result["metadata_file"]).read_text())
    assert result["state"] == "complete" and result["complete"] is True
    assert raw == index_body(probe.DATES[0])
    assert result["index"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["selected_record"]["param"] == "2t"
    assert result["object_head"]["object_size_bytes"] == 126472991
    assert result["object_head"]["last_modified_utc"] == "2025-09-14T20:00:00+00:00"
    assert result["object_head"]["next_day_10k_kst_utc"] == "2025-09-15T01:00:00+00:00"
    assert result["object_head"]["last_modified_before_origin"] is True
    assert result["object_head"]["last_modified_proves_first_publication"] is False
    assert metadata["complete"] is True
    assert [call[0] for call in session.calls] == ["GET", "HEAD"]
    assert all(call[2]["verify"] is True and call[2]["allow_redirects"] is False
               for call in session.calls)


@pytest.mark.parametrize("status,state,issue", [
    (404, "unavailable", "index_unavailable_404"),
    (503, "error", "index_http_error"),
])
def test_http_error_bodies_are_preserved_and_not_retried(tmp_path, status, state, issue):
    body = b"provider error body"
    session = FakeSession([FakeResponse(body, status, {"Content-Length": str(len(body))})])
    result = probe.probe_date(probe.DATES[0], raw_dir=tmp_path, session=session,
                              retrieved_at=fixed_time(), sleeper=lambda _: None)
    assert result["state"] == state and result["complete"] is False
    assert issue in result["issues"]
    assert (tmp_path / result["index"]["raw_file"]).read_bytes() == body
    assert len(session.calls) == 1


@pytest.mark.parametrize("body,expected_issue", [
    (b"{broken json}\n", "invalid_json_line_1"),
    (index_body(probe.DATES[0], extra=1), "matching_2t_record_not_unique"),
    (index_body(probe.DATES[0], offset=-1), "index_offset_invalid"),
    (index_body(probe.DATES[0], length=0), "index_length_invalid"),
])
def test_bad_index_is_preserved_incomplete_and_skips_head(tmp_path, body, expected_issue):
    session = FakeSession([FakeResponse(body, 200)])
    result = probe.probe_date(probe.DATES[0], raw_dir=tmp_path, session=session,
                              retrieved_at=fixed_time(), sleeper=lambda _: None)
    assert result["state"] == "incomplete" and result["complete"] is False
    assert expected_issue in result["issues"]
    assert (tmp_path / result["index"]["raw_file"]).read_bytes() == body
    assert len(session.calls) == 1


def test_index_partial_stream_saved_with_partial_state(tmp_path):
    response = FakeResponse(status=200, headers={"Content-Length": "100"},
                            chunks=[b'{"date":"20250914"}', requests.ConnectionError("interrupted")])
    session = FakeSession([response])
    result = probe.probe_date(probe.DATES[0], raw_dir=tmp_path, session=session,
                              retrieved_at=fixed_time(), sleeper=lambda _: None)
    assert result["state"] == "partial" and result["complete"] is False
    assert result["index"]["transfer_error"] == "response_stream_failed"
    assert (tmp_path / result["index"]["raw_file"]).read_bytes() == b'{"date":"20250914"}'
    assert response.closed


def test_head_error_and_bad_last_modified_are_not_claimed_as_complete(tmp_path):
    index = index_body(probe.DATES[1])
    session = FakeSession([FakeResponse(index, 200), FakeResponse(b"", 503, {})])
    result = probe.probe_date(probe.DATES[1], raw_dir=tmp_path, session=session,
                              retrieved_at=fixed_time(), sleeper=lambda _: None)
    assert result["state"] == "error" and result["complete"] is False
    assert result["issues"] == ["object_head_http_error"]

    session = FakeSession([FakeResponse(index, 200), FakeResponse(b"", 200,
                              {"Content-Length": "100", "Last-Modified": "not a date"})])
    result = probe.probe_date(probe.DATES[1], raw_dir=tmp_path / "invalid-date", session=session,
                              retrieved_at=fixed_time(), sleeper=lambda _: None)
    assert result["state"] == "incomplete" and result["complete"] is False
    assert "last_modified_missing_or_invalid" in result["issues"]


def test_selected_index_range_must_fit_inside_head_object_size(tmp_path):
    body = index_body(probe.DATES[2], offset=500, length=100)
    responses = [FakeResponse(body, 200), FakeResponse(b"", 200, {
        "Content-Length": "550", "Last-Modified": "Mon, 21 Sep 2026 20:00:00 GMT"})]
    session = FakeSession(responses)
    result = probe.probe_date(probe.DATES[2], raw_dir=tmp_path, session=session,
                              retrieved_at=fixed_time(), sleeper=lambda _: None)
    assert result["state"] == "incomplete" and result["complete"] is False
    assert result["object_head"]["selected_index_range_within_object"] is False
    assert "selected_index_range_outside_object" in result["issues"]
    assert (tmp_path / result["index"]["raw_file"]).read_bytes() == body
    assert len(session.calls) == 2


def test_fixed_three_date_sample_uses_at_most_six_requests(tmp_path):
    responses = []
    for date in probe.DATES:
        responses.extend(success_pair(date, "Sun, 14 Sep 2025 20:00:00 GMT"))
    session = FakeSession(responses)
    results = probe.probe(raw_dir=tmp_path, session=session, retrieved_at=fixed_time(), sleeper=lambda _: None)
    assert [item["run_date"] for item in results] == list(probe.DATES)
    assert all(item["complete"] for item in results)
    assert len(session.calls) == 6
    assert all(call[0] != "GET" or call[2].get("stream") is True for call in session.calls)
    assert not any("Range" in call[2] for call in session.calls)


def test_transport_retries_are_bounded_and_metadata_records_failure(tmp_path):
    session = FakeSession([requests.Timeout("x"), requests.ConnectionError("y"), requests.Timeout("z")])
    result = probe.probe_date(probe.DATES[2], raw_dir=tmp_path, session=session,
                              retrieved_at=fixed_time(), sleeper=lambda _: None)
    assert result["state"] == "error" and result["complete"] is False
    assert result["attempts"]["index_get"] == probe.MAX_ATTEMPTS
    assert len(session.calls) == probe.MAX_ATTEMPTS
    assert result["issues"] == ["transport_failed_after_bounded_retries"]
    assert "raw_file" not in result["index"]
