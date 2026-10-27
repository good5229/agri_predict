import hashlib
import http.client
import io
import json
from pathlib import Path
import ssl

import pytest

from agri_data.kamis_wholesale_history import (DEFAULT_URL, MAX_BYTES, _opener, probe,
    build_query_plan, collect_history, compare_five_crop_pilot, POST_URL, CROPS, _json_bytes)


class FakeResponse:
    def __init__(self, body=b"page", status=200, url=DEFAULT_URL, headers=None):
        self.stream = io.BytesIO(body)
        self.status = status
        self.headers = headers or {"Content-Type": "text/html", "Set-Cookie": "sid=do-not-copy"}
        self.url = url
        self.closed = False

    def read(self, size=-1): return self.stream.read(size)
    def geturl(self): return self.url
    def close(self): self.closed = True


class PartialErrorResponse(FakeResponse):
    def __init__(self, partial, status=200, advertised=None):
        headers = {"Content-Type": "text/html"}
        if advertised is not None: headers["Content-Length"] = str(advertised)
        super().__init__(partial, status=status, headers=headers)
        self.first = True

    def read(self, size=-1):
        if self.first:
            self.first = False
            return self.stream.read(min(size, 5))
        exc = OSError("stream broke")
        exc.partial = b""
        raise exc


class IncompleteReadResponse(PartialErrorResponse):
    def read(self, size=-1):
        if self.first:
            self.first = False
            return self.stream.read(min(size, 5))
        raise http.client.IncompleteRead(b"tail", expected=10)


class FakeOpener:
    def __init__(self, results): self.results, self.calls = list(results), []
    def open(self, request, timeout):
        self.calls.append((request, timeout))
        result = self.results.pop(0)
        if isinstance(result, Exception): raise result
        return result


def test_success_saves_verifiable_raw_response_and_checkpoint(tmp_path):
    opener = FakeOpener([FakeResponse(b"<html>ok</html>")])
    result = probe(output_dir=tmp_path, opener=opener, now=lambda: "2026-01-01T00:00:00Z")
    assert result["state"] == "complete" and result["complete"]
    response = result["responses"][0]
    body_path = tmp_path / response["bodyFile"]
    assert body_path.read_bytes() == b"<html>ok</html>"
    assert response["bodySha256"] == hashlib.sha256(body_path.read_bytes()).hexdigest()
    assert "Set-Cookie" not in response["headers"]
    request, timeout = opener.calls[0]
    assert request.get_method() == "GET" and timeout == 20
    assert request.get_header("Cookie") is None
    resumed = probe(output_dir=tmp_path, opener=FakeOpener([]))
    assert resumed["checkpoint"] == "already_recorded"
    body_path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint failed"):
        probe(output_dir=tmp_path, opener=FakeOpener([]))


@pytest.mark.parametrize("status", [401, 403, 429, 302, 404])
def test_auth_rate_redirect_and_client_statuses_do_not_retry(tmp_path, status):
    opener = FakeOpener([FakeResponse(b"failure", status=status)])
    sleeps = []
    result = probe(output_dir=tmp_path, opener=opener, sleep=sleeps.append)
    assert result["state"] == "http_failure" and not result["complete"]
    assert len(opener.calls) == 1 and sleeps == []


def test_transient_server_failure_retries_and_archives_each_response(tmp_path):
    opener = FakeOpener([FakeResponse(b"temporary", 503), FakeResponse(b"source", 200)])
    sleeps = []
    result = probe(output_dir=tmp_path, opener=opener, sleep=sleeps.append)
    assert result["state"] == "complete" and len(result["responses"]) == 2
    assert sleeps == [1] and len(opener.calls) == 2
    assert [(tmp_path / r["bodyFile"]).read_bytes() for r in result["responses"]] == [b"temporary", b"source"]


def test_final_transport_failure_has_null_final_response_fields_and_keeps_prior_http_body(tmp_path):
    opener = FakeOpener([FakeResponse(b"first", status=503), OSError("network down"), OSError("network down")])
    result = probe(output_dir=tmp_path, opener=opener, sleep=lambda _: None)
    assert result["state"] == "transport_failure" and result["httpStatus"] is None
    assert result["bodyFile"] is None and result["bodySha256"] is None
    assert result["bodyBytesCaptured"] == 0 and result["responseHeaders"] == {}
    assert len(result["responses"]) == 1 and result["responses"][0]["httpStatus"] == 503
    assert (tmp_path / result["responses"][0]["bodyFile"]).read_bytes() == b"first"


def test_premature_eof_against_content_length_is_incomplete_then_retried(tmp_path):
    short = FakeResponse(b"short", status=200, headers={"Content-Length": "10"})
    good = FakeResponse(b"done", status=200, headers={"Content-Length": "4"})
    opener = FakeOpener([short, good]); sleeps = []
    result = probe(output_dir=tmp_path, opener=opener, sleep=sleeps.append)
    assert result["state"] == "complete" and len(result["responses"]) == 2
    assert result["responses"][0]["outcome"] == "incomplete_response"
    assert result["responses"][0]["framingComplete"] is False
    assert (tmp_path / result["responses"][0]["bodyFile"]).read_bytes() == b"short"
    assert sleeps == [1]


def test_partial_read_error_on_429_is_preserved_without_retry(tmp_path):
    response = PartialErrorResponse(b"partial", status=429, advertised=20)
    opener = FakeOpener([response]); sleeps = []
    result = probe(output_dir=tmp_path, opener=opener, sleep=sleeps.append)
    assert result["state"] == "http_failure" and len(opener.calls) == 1 and not sleeps
    record = result["responses"][0]
    assert record["httpStatus"] == 429 and record["readErrorClass"] == "OSError"
    assert record["headers"]["Content-Length"] == "20"
    assert (tmp_path / record["bodyFile"]).read_bytes() == b"parti"


def test_http_incomplete_read_partial_bytes_are_preserved(tmp_path):
    opener = FakeOpener([IncompleteReadResponse(b"prefix", status=401)])
    result = probe(output_dir=tmp_path, opener=opener)
    record = result["responses"][0]
    assert result["state"] == "http_failure" and len(opener.calls) == 1
    assert record["readErrorClass"] == "IncompleteRead"
    assert (tmp_path / record["bodyFile"]).read_bytes() == b"prefitail"


def test_partial_read_error_on_200_preserves_each_attempt_and_stops_after_two_retries(tmp_path):
    opener = FakeOpener([PartialErrorResponse(b"one"), PartialErrorResponse(b"two"), PartialErrorResponse(b"three")])
    sleeps = []
    result = probe(output_dir=tmp_path, opener=opener, sleep=sleeps.append)
    assert result["state"] == "incomplete_response" and len(opener.calls) == 3
    assert len(result["responses"]) == 3 and sleeps == [1, 1]
    assert [r["readErrorClass"] for r in result["responses"]] == ["OSError"] * 3
    assert [(tmp_path / r["bodyFile"]).read_bytes() for r in result["responses"]] == [b"one", b"two", b"three"]


def test_transport_failures_are_bounded_and_checkpointed(tmp_path):
    opener = FakeOpener([OSError("network down"), OSError("network down"), OSError("network down")])
    sleeps = []
    result = probe(output_dir=tmp_path, opener=opener, sleep=sleeps.append)
    assert result["state"] == "transport_failure" and not result["complete"]
    assert len(result["attempts"]) == 3 and len(opener.calls) == 3 and sleeps == [1, 1]
    assert result["httpStatus"] is None


def test_oversize_response_is_capped_and_marked_incomplete(tmp_path):
    opener = FakeOpener([FakeResponse(b"x" * (MAX_BYTES + 37))])
    result = probe(output_dir=tmp_path, opener=opener)
    response = result["responses"][0]
    assert result["state"] == "oversize_response" and not result["complete"]
    assert response["bodyBytesCaptured"] == MAX_BYTES and response["bodyTruncated"]
    assert len((tmp_path / response["bodyFile"]).read_bytes()) == MAX_BYTES


def test_rejects_cross_origin_response_url(tmp_path):
    opener = FakeOpener([FakeResponse(b"x", url="https://example.com/redirected")])
    result = probe(output_dir=tmp_path, opener=opener)
    assert result["state"] == "redirect_rejected" and not result["complete"]
    assert len(opener.calls) == 1


def test_default_opener_uses_verified_tls():
    opener = _opener()
    handler = next(h for h in opener.handlers if hasattr(h, "_context"))
    assert handler._context.verify_mode == ssl.CERT_REQUIRED and handler._context.check_hostname


@pytest.mark.parametrize("url", ["http://www.kamis.or.kr/a", "https://example.com/a", "https://user@www.kamis.or.kr/a", "https://www.kamis.or.kr/a#fragment"])
def test_rejects_malformed_or_cross_origin_urls(tmp_path, url):
    opener = FakeOpener([])
    with pytest.raises(ValueError): probe(url, output_dir=tmp_path, opener=opener)
    assert opener.calls == []


def test_rejects_incomplete_body_checkpoint_without_overwrite(tmp_path):
    key = hashlib.sha256(DEFAULT_URL.encode()).hexdigest()
    (tmp_path / f"probe-{key}-attempt-1.body").write_bytes(b"partial")
    opener = FakeOpener([])
    with pytest.raises(FileExistsError): probe(output_dir=tmp_path, opener=opener)
    assert opener.calls == []


def test_query_plan_is_quarter_bounded_and_preserves_exact_public_form_fields():
    import datetime as dt
    rows = build_query_plan(dt.date(2024, 3, 1), dt.date(2024, 4, 2), ["pepper"])
    assert [(r["start"], r["end"]) for r in rows] == [("2024-03-01", "2024-03-31"), ("2024-04-01", "2024-04-02")]
    q = rows[0]
    assert q["method"] == "POST" and q["url"] == POST_URL
    assert q["formFields"]["itemcode"] == "241" and q["formFields"]["kindcode"] == "AA"
    assert q["formFields"]["period"] == "-10" and q["formFields"]["convert_kg_yn"] == "N"
    assert q["requestSha256"]


def test_opt_in_korean_language_changes_only_header_variant_and_request_hash(tmp_path):
    import datetime as dt
    ordinary = build_query_plan(dt.date(2024, 1, 1), dt.date(2024, 1, 2), ["cabbage"])[0]
    korean = build_query_plan(dt.date(2024, 1, 1), dt.date(2024, 1, 2), ["cabbage"], korean_language=True)[0]
    assert "requestHeaders" not in ordinary
    assert korean["requestHeaders"]["Accept-Language"] == "ko-KR,ko;q=0.9,en;q=0.8"
    assert ordinary["requestBody"] == korean["requestBody"]
    assert ordinary["requestSha256"] != korean["requestSha256"]
    opener = FakeOpener([FakeResponse(b"denied", 403)])
    collect_history(dt.date(2024, 1, 1), dt.date(2024, 1, 2), crops=["cabbage"], output_dir=tmp_path,
                    max_requests=1, korean_language=True, opener=opener, sleep=lambda _: None)
    assert opener.calls[0][0].get_header("Accept-language") == "ko-KR,ko;q=0.9,en;q=0.8"
    record = json.loads(next(tmp_path.glob("query-*.json")).read_text())
    assert record["requestHeaders"]["Accept-Language"] == "ko-KR,ko;q=0.9,en;q=0.8"


def test_legacy_default_plan_and_completed_checkpoint_resume_without_requests(tmp_path):
    import datetime as dt
    import hashlib
    start, end = dt.date(2024, 1, 1), dt.date(2024, 1, 2)
    query = build_query_plan(start, end, ["cabbage"])[0]
    assert "requestHeaders" not in query
    plan = _json_bytes({"schemaVersion": 1, "method": "official public UI form POST", "queries": [query]})
    (tmp_path / "plan.json").write_bytes(plan)
    rows = {"rawRows": [], "support": {"requestedDates": 2, "observedDates": 0}, "qualityStatus": "clear"}
    rows_bytes = _json_bytes(rows)
    rows_path = tmp_path / f"query-{query['id']}.rows.json"
    rows_path.write_bytes(rows_bytes)
    meta = {"id": query["id"], "state": "completed", "requestSha256": query["requestSha256"],
            "method": "POST", "url": query["url"], "requestBody": query["requestBody"],
            "rowsFile": rows_path.name, "rowsSha256": hashlib.sha256(rows_bytes).hexdigest(), "responses": []}
    (tmp_path / f"query-{query['id']}.json").write_bytes(_json_bytes(meta))
    opener = FakeOpener([])
    result = collect_history(start, end, crops=["cabbage"], output_dir=tmp_path, max_requests=1, opener=opener)
    assert opener.calls == [] and result["requestsUsed"] == 0
    assert result["queries"][0]["state"] == "completed"
    with pytest.raises(ValueError, match="frozen query plan differs"):
        collect_history(start, end, crops=["cabbage"], output_dir=tmp_path, max_requests=1,
                        korean_language=True, opener=FakeOpener([]))


def test_request_budget_counts_retry_and_resume_keeps_unrequested_queries_pending(tmp_path):
    import datetime as dt
    opener = FakeOpener([FakeResponse(b"busy", 503)])
    sleeps = []
    summary = collect_history(dt.date(2024, 1, 1), dt.date(2024, 6, 30), crops=["radish"],
                              output_dir=tmp_path, max_requests=1, opener=opener,
                              sleep=sleeps.append, clock=lambda: "2026-01-01T00:00:00Z")
    assert len(opener.calls) == 1 and sleeps == [] and summary["requestsUsed"] == 1
    assert [q["state"] for q in summary["queries"]] == ["http_failure", "pending"]
    again = collect_history(dt.date(2024, 1, 1), dt.date(2024, 6, 30), crops=["radish"],
                            output_dir=tmp_path, max_requests=0, opener=FakeOpener([]),
                            clock=lambda: "2026-01-01T00:00:00Z")
    assert [q["state"] for q in again["queries"]] == ["http_failure", "pending"]
    assert len(list(tmp_path.glob("summary-*.json"))) == 2


def test_collect_parses_web_table_and_preserves_request_and_response_provenance(tmp_path):
    import datetime as dt
    body = ("<table id='itemTable_1'><caption>중도매인 판매가격 채소류, 배추, 전체, 상품, "
            "10kg(그물망 3포기) 기간별/일간 가격정보</caption><thead><tr>"
            "<th>구분</th><th>평균</th><th>등락율</th><th>서울</th><th>부산</th><th>대구</th><th>광주</th><th>대전</th>"
            "</tr></thead><tbody><tr><th>24-01-01</th><td>1,000</td><td>0</td><td>900</td><td>910</td>"
            "<td>920</td><td>930</td><td>940</td></tr></tbody></table>").encode("utf-8")
    opener = FakeOpener([FakeResponse(body)])
    result = collect_history(dt.date(2024, 1, 1), dt.date(2024, 1, 2), crops=["cabbage"],
                             output_dir=tmp_path, max_requests=1, opener=opener, sleep=lambda _: None,
                             clock=lambda: "2026-01-01T00:00:00Z")
    assert result["completed"] == 1 and result["empty"] == 0 and result["error"] == 0
    query = result["queries"][0]
    assert query["state"] == "completed" and query["support"]["missingDates"] == ["2024-01-02"]
    meta = json.loads((tmp_path / f"query-{query['id']}.json").read_text())
    assert meta["method"] == "POST" and meta["httpStatus"] == 200
    assert meta["sourceSha256"] == hashlib.sha256(body).hexdigest()
    request, timeout = opener.calls[0]
    assert request.get_method() == "POST" and timeout == 20
    assert request.data.decode("ascii") == meta["requestBody"]
    assert meta["requestSha256"] == hashlib.sha256(
        b"POST\n" + POST_URL.encode("ascii") + b"\n" + request.data).hexdigest()
    raw_path = tmp_path / meta["sourceBodyFile"]
    assert raw_path.read_bytes() == body
    parsed = json.loads((tmp_path / meta["rowsFile"]).read_text())
    assert parsed["rawRows"][0]["prices"]["평균"]["decimal"] == "1000"


@pytest.mark.parametrize("status", [401, 403, 429])
def test_post_auth_and_rate_errors_stop_without_retry(tmp_path, status):
    import datetime as dt
    opener = FakeOpener([FakeResponse(b"denied", status)])
    result = collect_history(dt.date(2024, 1, 1), dt.date(2024, 1, 2), crops=["cabbage"],
                            output_dir=tmp_path, max_requests=3, opener=opener, sleep=lambda _: None)
    assert len(opener.calls) == 1 and result["requestsUsed"] == 1
    assert result["queries"][0]["state"] == "http_failure"


def test_auth_failure_stops_later_queries_in_the_frozen_plan(tmp_path):
    import datetime as dt
    opener = FakeOpener([FakeResponse(b"denied", 403)])
    result = collect_history(dt.date(2024, 1, 1), dt.date(2024, 4, 2), crops=["cabbage"],
                            output_dir=tmp_path, max_requests=4, opener=opener, sleep=lambda _: None)
    assert len(opener.calls) == 1
    assert [q["state"] for q in result["queries"]] == ["http_failure", "pending"]


def test_transient_failure_requires_explicit_bounded_retry_generation(tmp_path):
    import datetime as dt
    start, end = dt.date(2024, 1, 1), dt.date(2024, 1, 2)
    first = collect_history(start, end, crops=["cabbage"], output_dir=tmp_path, max_requests=1,
                            opener=FakeOpener([FakeResponse(b"busy", 503)]), sleep=lambda _: None)
    assert first["queries"][0]["state"] == "http_failure"
    ordinary = collect_history(start, end, crops=["cabbage"], output_dir=tmp_path, max_requests=1,
                               opener=FakeOpener([]), sleep=lambda _: None)
    assert ordinary["requestsUsed"] == 0
    retry = collect_history(start, end, crops=["cabbage"], output_dir=tmp_path, max_requests=1,
                            retry_transient=True, opener=FakeOpener([FakeResponse(b"busy again", 503)]),
                            sleep=lambda _: None)
    assert retry["requestsUsed"] == 1 and retry["queries"][0]["state"] == "http_failure"
    assert len(list(tmp_path.glob("query-*.body"))) == 2
    assert len(list(tmp_path.glob("query-*-retry-1-attempt-*.body"))) == 1
    assert len(list(tmp_path.glob("query-*-retry-1.json"))) == 1


def test_saved_later_rate_limit_preflights_before_retrying_earlier_transient(tmp_path):
    import datetime as dt
    start, end = dt.date(2024, 1, 1), dt.date(2024, 4, 2)
    collect_history(start, end, crops=["cabbage"], output_dir=tmp_path, max_requests=4,
                    opener=FakeOpener([FakeResponse(b"temporary-1", 503), FakeResponse(b"temporary-2", 503),
                                       FakeResponse(b"temporary-3", 503), FakeResponse(b"rate", 429)]),
                    sleep=lambda _: None)
    opener = FakeOpener([])
    result = collect_history(start, end, crops=["cabbage"], output_dir=tmp_path, max_requests=5,
                              retry_transient=True, opener=opener, sleep=lambda _: None)
    assert opener.calls == [] and result["requestsUsed"] == 0
    assert [q["state"] for q in result["queries"]] == ["http_failure", "http_failure"]


def _pilot_fixture(tmp_path, *, mismatched_ui=False):
    import datetime as dt
    from agri_data.kamis_wholesale_export import parse_export
    web_dir, ui_dir = tmp_path / "web", tmp_path / "ui"
    web_dir.mkdir(); ui_dir.mkdir()
    start, end = dt.date(2024, 1, 1), dt.date(2024, 3, 31)
    dates = []
    day = start
    while day <= end:
        if day.weekday() < 5 and day not in (dt.date(2024, 2, 9), dt.date(2024, 2, 12)):
            dates.append(day); 
        day += dt.timedelta(days=1)
    dates = dates[:61]
    queries = build_query_plan(start, end)
    (web_dir / "plan.json").write_bytes(_json_bytes({"queries": queries}))
    entries = []
    headers = "<th>구분</th><th>평균</th><th>등락율</th><th>서울</th><th>부산</th><th>대구</th><th>광주</th><th>대전</th>"
    for query in queries:
        profile = CROPS[query["crop"]]
        caption = f"중도매인 판매가격 채소류, {profile['item']}, {profile['variety']}, 상품, {profile['unit']} 기간별/일간 가격정보"
        rows = []
        for d in dates:
            values = "<td>100</td><td>-</td><td>102</td><td>103</td><td>104</td><td>105</td><td>106</td>"
            rows.append(f"<tr><th>{d:%y-%m-%d}</th>{values}</tr>")
        html = (f"<table id='itemTable_1'><caption>{caption}</caption><thead><tr>{headers}</tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table>").encode()
        parsed = parse_export(html, start=start, end=end, item=profile["item"], variety=profile["variety"],
                              grade="상품", unit=profile["unit"], table_id="itemTable_1")
        parsed["requested"]["crop"] = query["crop"]
        parsed["qualityStatus"] = "clear"
        raw_name, rows_name = f"{query['id']}.body", f"{query['id']}.rows.json"
        (web_dir / raw_name).write_bytes(html)
        rows_bytes = _json_bytes(parsed); (web_dir / rows_name).write_bytes(rows_bytes)
        metadata = {"id": query["id"], "state": "completed", "requestSha256": query["requestSha256"],
                    "method": query["method"], "url": query["url"], "requestBody": query["requestBody"],
                    "sourceBodyFile": raw_name, "sourceSha256": __import__("hashlib").sha256(html).hexdigest(),
                    "rowsFile": rows_name, "rowsSha256": __import__("hashlib").sha256(rows_bytes).hexdigest(), "responses": []}
        if "requestHeaders" in query: metadata["requestHeaders"] = query["requestHeaders"]
        (web_dir / f"query-{query['id']}.json").write_bytes(_json_bytes(metadata))
        ui_name = f"{query['crop']}.xls"
        ui_raw = html.replace(b"itemTable_1", b"data")
        if mismatched_ui and query["crop"] == "radish":
            ui_raw = ui_raw.replace(f"<tr><th>{dates[-1]:%y-%m-%d}</th>".encode(), b"<tr><th>24-03-31</th>")
        (ui_dir / ui_name).write_bytes(ui_raw)
        entries.append({"path": ui_name, "item": profile["item"], "variety": profile["variety"], "grade": "상품",
                        "unit": profile["unit"], "start": start.isoformat(), "end": end.isoformat(),
                        "bytes": len(ui_raw), "sha256": __import__("hashlib").sha256(ui_raw).hexdigest(), "uiDataRows": 61})
    manifest_path = tmp_path / "provenance.json"
    manifest_path.write_bytes(_json_bytes({"files": entries}))
    return web_dir, manifest_path, ui_dir


def test_compare_pilot_accepts_matching_observed_dates_without_calendar_filling(tmp_path):
    web_dir, manifest, ui_dir = _pilot_fixture(tmp_path)
    report = compare_five_crop_pilot(web_dir, manifest, ui_dir, tmp_path / "report.json")
    assert report["allExactMatch"] and report["networkRequests"] == 0
    assert all(c["comparedDates"] == 61 and c["comparedPriceCells"] == 366 for c in report["crops"])


def test_compare_pilot_rejects_mismatched_observed_date_support(tmp_path):
    web_dir, manifest, ui_dir = _pilot_fixture(tmp_path, mismatched_ui=True)
    with pytest.raises(ValueError, match="date support mismatch"):
        compare_five_crop_pilot(web_dir, manifest, ui_dir, tmp_path / "report.json")
