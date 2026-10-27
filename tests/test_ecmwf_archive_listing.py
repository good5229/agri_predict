import datetime as dt
import hashlib
import json
from unittest import mock

import pytest
import requests

from agri_data import ecmwf_archive_listing as listing


class FakeResponse:
    def __init__(self, body=b"", status=200, headers=None, chunks=None):
        self.body, self.status_code, self.headers = body, status, headers or {}
        self.chunks = chunks
        self.closed = False

    def iter_content(self, chunk_size):
        if self.chunks is not None:
            for chunk in self.chunks:
                if isinstance(chunk, Exception):
                    raise chunk
                yield chunk
        elif self.body:
            yield self.body

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def page(prefixes=(), next_token=None):
    payload = {"kind": "storage#objects", "prefixes": list(prefixes)}
    if next_token is not None:
        payload["nextPageToken"] = next_token
    body = json.dumps(payload, separators=(",", ":")).encode()
    return FakeResponse(body, 200, {"Content-Type": "application/json", "Content-Length": str(len(body))})


def now():
    return dt.datetime(2026, 10, 7, 15, tzinfo=dt.timezone.utc)


def test_dry_run_does_not_send_requests(capsys):
    with mock.patch.object(listing, "collect", side_effect=AssertionError("network path invoked")):
        assert listing.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["request_sent"] is False
    assert result["years"] == ["2024", "2025", "2026"]
    assert result["max_results_per_page"] == 1000


def test_complete_listing_paginates_preserves_each_page_and_reports_missing_dates(tmp_path):
    responses = [page(["20240101/", "20240930/"]),
                 page(["20250101/"], "page-2"), page(["20250901/"]),
                 page(["20260922/"])]
    session = FakeSession(responses)
    report = listing.collect(raw_dir=tmp_path, session=session, retrieved_at=now(), sleeper=lambda _: None)
    assert report["complete"] is True
    assert len(session.calls) == 4
    assert all(url == listing.API_URL for url, _ in session.calls)
    assert [kwargs["params"]["prefix"] for _, kwargs in session.calls] == ["2024", "2025", "2025", "2026"]
    assert all(kwargs["params"]["delimiter"] == "/" and kwargs["params"]["maxResults"] == 1000
               for _, kwargs in session.calls)
    assert "pageToken" not in session.calls[1][1]["params"]
    assert session.calls[2][1]["params"]["pageToken"] == "page-2"
    training = report["date_ranges"][0]
    assert training["listing_complete"] is True
    assert "2024-01-01" in training["observed_present_dates"]
    assert "2025-09-01" in training["observed_present_dates"]
    assert "2024-01-02" in training["missing_dates"]
    test_origins = report["date_ranges"][1]
    assert "2026-09-22" in test_origins["observed_present_dates"]
    assert "2025-09-14" in test_origins["missing_dates"]
    assert "no run" in test_origins["coverage_claim"]
    first_page = report["years"]["2024"]["pages"][0]
    raw = (tmp_path / first_page["raw_file"]).read_bytes()
    metadata = json.loads((tmp_path / first_page["metadata_file"]).read_text())
    assert first_page["sha256"] == hashlib.sha256(raw).hexdigest()
    assert metadata["request"]["prefix"] == "2024"
    assert report["summary_sha256"] == hashlib.sha256(
        (tmp_path / report["summary_report"]).read_bytes()).hexdigest()


@pytest.mark.parametrize("status,state", [(404, "unavailable"), (503, "error")])
def test_http_error_body_is_preserved_and_not_retried(tmp_path, status, state):
    error_body = b"provider error body"
    session = FakeSession([FakeResponse(error_body, status,
                                        {"Content-Length": str(len(error_body))})])
    result = listing._year_listing("2024", raw_dir=tmp_path, session=session,
                                   retrieved=now(), sleeper=lambda _: None)
    assert result["complete"] is False
    assert result["pages"][0]["state"] == state
    assert result["state"] == "incomplete"
    assert len(session.calls) == 1
    page_meta = result["pages"][0]
    assert (tmp_path / page_meta["raw_file"]).read_bytes() == error_body
    assert page_meta["http_status"] == status


@pytest.mark.parametrize("prefixes", [
    ["20240230/"], ["2025notdate/"], ["20230101/"], ["20240101", 123],
])
def test_invalid_date_prefixes_mark_listing_incomplete(tmp_path, prefixes):
    session = FakeSession([page(prefixes)])
    result = listing._year_listing("2024", raw_dir=tmp_path, session=session,
                                   retrieved=now(), sleeper=lambda _: None)
    assert result["complete"] is False
    assert "page_1_invalid_date_prefixes" in result["issues"]
    assert result["prefixes"] == []
    assert result["pages"][0]["invalid_prefixes"]


def test_malformed_json_is_saved_and_date_range_missing_is_not_asserted(tmp_path):
    session = FakeSession([FakeResponse(b"{broken", 200)])
    year = listing._year_listing("2024", raw_dir=tmp_path, session=session,
                                 retrieved=now(), sleeper=lambda _: None)
    report = listing._date_range_report({
        "2024": year,
        "2025": {"complete": True, "prefixes": []},
        "2026": {"complete": True, "prefixes": []},
    })
    assert year["state"] == "incomplete"
    assert report[0]["missing_dates"] is None
    assert (tmp_path / year["pages"][0]["raw_file"]).read_bytes() == b"{broken"


def test_partial_stream_and_content_length_mismatch_are_retained(tmp_path):
    partial = FakeResponse(status=200, headers={"Content-Length": "100"},
                           chunks=[b'{"prefixes":["20240101/"]}', requests.ConnectionError("cut")])
    result = listing._year_listing("2024", raw_dir=tmp_path, session=FakeSession([partial]),
                                   retrieved=now(), sleeper=lambda _: None)
    assert result["complete"] is False
    assert result["pages"][0]["state"] == "partial"
    assert (tmp_path / result["pages"][0]["raw_file"]).read_bytes() == b'{"prefixes":["20240101/"]}'

    body = json.dumps({"prefixes": []}).encode()
    mismatch = FakeResponse(body, 200, {"Content-Length": "1"})
    result = listing._year_listing("2024", raw_dir=tmp_path / "mismatch",
                                   session=FakeSession([mismatch]), retrieved=now(), sleeper=lambda _: None)
    assert result["pages"][0]["state"] == "partial"
    assert "content_length_mismatch" in result["pages"][0]["issues"]


def test_page_token_loop_and_page_limit_leave_listing_incomplete(tmp_path, monkeypatch):
    session = FakeSession([page([], "repeat"), page([], "repeat")])
    result = listing._year_listing("2024", raw_dir=tmp_path / "loop", session=session,
                                   retrieved=now(), sleeper=lambda _: None)
    assert result["complete"] is False
    assert "page_2_pagination_token_loop" in result["issues"]
    persisted = json.loads((tmp_path / "loop" / result["pages"][1]["metadata_file"]).read_text())
    assert "pagination_token_loop" in persisted["issues"]

    monkeypatch.setattr(listing, "MAX_PAGES_PER_YEAR", 1)
    result = listing._year_listing("2024", raw_dir=tmp_path / "limit",
                                   session=FakeSession([page([], "more")]),
                                   retrieved=now(), sleeper=lambda _: None)
    assert result["complete"] is False
    assert "page_limit_reached_with_next_token" in result["issues"]


@pytest.mark.parametrize("bad_token", [["unexpected"], {"unexpected": "object"}])
def test_non_string_page_token_is_saved_incomplete_without_crashing(tmp_path, bad_token):
    body = json.dumps({"prefixes": ["20240101/"], "nextPageToken": bad_token}).encode()
    response = FakeResponse(body, 200, {"Content-Length": str(len(body))})
    result = listing._year_listing("2024", raw_dir=tmp_path, session=FakeSession([response]),
                                   retrieved=now(), sleeper=lambda _: None)
    assert result["complete"] is False
    assert "page_1_next_page_token_invalid" in result["issues"]
    page_meta = result["pages"][0]
    assert page_meta["state"] == "incomplete"
    assert "next_page_token_invalid" in page_meta["issues"]
    assert (tmp_path / page_meta["raw_file"]).read_bytes() == body
    persisted = json.loads((tmp_path / page_meta["metadata_file"]).read_text())
    assert "next_page_token_invalid" in persisted["issues"]


def test_transport_retries_bounded_and_missing_page_does_not_become_empty_listing(tmp_path):
    session = FakeSession([requests.Timeout("one"), requests.ConnectionError("two"),
                           requests.Timeout("three")])
    result = listing._year_listing("2024", raw_dir=tmp_path, session=session,
                                   retrieved=now(), sleeper=lambda _: None)
    assert result["complete"] is False
    assert len(session.calls) == listing.MAX_ATTEMPTS
    assert result["issues"] == ["page_1_transport_failed_after_bounded_retries"]
    assert result["prefixes"] == []
    assert result["pages"][0]["state"] == "error"
    assert "raw_file" not in result["pages"][0]
