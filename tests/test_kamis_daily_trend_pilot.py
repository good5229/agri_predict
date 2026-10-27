import hashlib
import json
from pathlib import Path
import requests
from urllib.parse import urlencode

from agri_data import kamis_daily_trend_pilot as pilot


def _page(records):
    rows = []
    for kind, category, item, variety, day, region, reporter, text in records:
        rows.append(f'''<div class="flex_tit"><div class="rows"><span class="state blue">{kind}</span>
        <h3 class="lg_tit">{category}/ {item}/ {variety}, {day}, {region}</h3></div><span>{reporter} 조사원</span></div>
        <pre class="txt_box"><p>{text}</p></pre>''')
    return ('<html><div id="area_list">' + ''.join(rows) + '</div></html>').encode()


def test_first_third_tuesday_pilot_is_reproducible():
    days = pilot.first_third_tuesdays()
    assert len(days) == 42
    assert days[0] == "2025-01-07" and days[-1] == "2026-09-15"
    assert all(__import__('datetime').date.fromisoformat(x).weekday() == 1 for x in days)


def test_parser_preserves_records_and_classifies_exact_target_crop():
    body = _page([
        ("중도매", "채소류", "배추", "여름(고랭지)", "2026-10-08", "대전지역", "김수연", "● 공급 증가<br>● 가격 하락"),
        ("소매", "채소류", "마늘", "마늘", "2026-10-08", "서울지역", "이은정", "일반 마늘 기록"),
        ("소매", "채소류", "깐마늘", "깐마늘", "2026-10-08", "서울지역", "이은정", "깐마늘 기록"),
        ("소매", "채소류", "건고추", "화건", "2026-10-08", "광주지역", "진창선", "건고추 기록"),
    ])
    result = pilot.parse_html(body, "2026-10-08")
    assert result["state"] == "valid" and result["recordCount"] == 4
    cabbage, garlic, peeled, dried = result["records"]
    assert (cabbage["crop"], cabbage["variety"], cabbage["region"], cabbage["reporter"]) == ("배추", "여름(고랭지)", "대전지역", "김수연 조사원")
    assert cabbage["text"] == "● 공급 증가\n● 가격 하락"
    assert not garlic["isTargetCrop"] and "generic_garlic_ambiguous" in garlic["issues"]
    assert peeled["isTargetCrop"] and dried["isTargetCrop"]


def test_parser_keeps_nested_inline_markup_inside_record_text():
    body = '''<div id="area_list"><div class="flex_tit"><span class="state">중도매</span>
    <h3 class="lg_tit">채소류/ 배추/ 여름, 2026-10-08, 서울지역</h3><span>홍길동 조사원</span></div>
    <pre class="txt_box"><p>앞<span class="state">본문 상태</span><h3 class="lg_tit">본문 제목</h3>뒤</p></pre></div>'''.encode()
    parsed = pilot.parse_html(body, "2026-10-08")
    assert parsed["state"] == "valid"
    assert parsed["records"][0]["text"] == "앞본문 상태본문 제목뒤"


def test_parser_accepts_action_list_fragment_without_area_wrapper():
    fragment = _page([("중도매", "채소류", "양파", "양파", "2026-10-08", "서울지역", "이은정", "반입 동향")])
    fragment = fragment.replace(b'<html><div id="area_list">', b"").replace(b"</div></html>", b"")
    result = pilot.parse_html(fragment, "2026-10-08")
    assert result["state"] == "valid" and result["recordCount"] == 1
    assert result["records"][0]["crop"] == "양파"


def test_parser_distinguishes_empty_transport_http_and_bad_html():
    assert pilot.parse_html(b"", "2026-10-08")["state"] == "empty_response"
    assert pilot.parse_html(b"", "2026-10-08", http_status=503)["state"] == "http_error"
    assert pilot.parse_html(b"", "2026-10-08", transport_error="timeout")["state"] == "transport_error"
    assert pilot.parse_html(b"<html>challenge</html>", "2026-10-08")["state"] == "invalid_html"
    empty = pilot.parse_html(b"<html><div id=\"area_list\"></div></html>", "2026-10-08")
    assert empty["state"] == "questionable"
    assert "no_parseable_records; empty result not verified" in empty["issues"]
    malformed = pilot.parse_html('<div id="area_list"><h3 class="lg_tit">카드 표식</h3></div>'.encode(), "2026-10-08")
    assert malformed["state"] == "questionable"
    bad_heading = _page([("중도매", "채소류", "배추", "여름", "not-a-date", "서울지역", "홍길동", "본문")])
    assert pilot.parse_html(bad_heading, "2026-10-08")["state"] == "questionable"


def test_only_exact_visible_no_data_heading_counts_as_normal_empty():
    visible = '<h3 class="s_tit fl">검색된 데이터가 없습니다.</h3>'
    comment = '<!-- <h3 class="s_tit fl">검색된 데이터가 없습니다.</h3> -->'
    assert pilot.parse_html((visible + comment).encode(), "2026-10-08")["state"] == "normal_empty"
    assert pilot.parse_html(comment.encode(), "2026-10-08")["state"] == "invalid_html"
    malformed = '<h3 class="s_tit fl">검색된 데이터가 없습니다.'
    assert pilot.parse_html(malformed.encode(), "2026-10-08")["state"] == "invalid_html"
    wrong_close = '<div><h3 class="s_tit fl">검색된 데이터가 없습니다.</div>'
    assert pilot.parse_html(wrong_close.encode(), "2026-10-08")["state"] == "invalid_html"
    outside_area = ('<div id="area_list"></div>' + visible).encode()
    assert pilot.parse_html(outside_area, "2026-10-08")["state"] == "questionable"
    card = _page([("중도매", "채소류", "무", "고랭지", "2026-10-08", "서울지역", "홍길동", "기록")])
    card = card.replace(b'<html><div id="area_list">', b"").replace(b"</div></html>", b"")
    conflicting = (visible + card.decode()).encode()
    parsed = pilot.parse_html(conflicting, "2026-10-08")
    assert parsed["state"] == "questionable"
    assert "conflicting_no_data_and_record_markers" in parsed["issues"]


def test_bounded_reader_retains_partial_bytes_on_stream_failure():
    class BrokenStream:
        def iter_content(self, chunk_size):
            yield b"partial response"
            raise requests.ConnectionError("offline synthetic failure")
    body, truncated, error = pilot._read_bounded(BrokenStream())
    assert body == b"partial response" and not truncated
    assert error == "response_stream_error:ConnectionError"


class _Response:
    def __init__(self, body, status=200, url=None, headers=None):
        self.body = body; self.status_code = status; self.url = url
        self.headers = {"content-type": "text/html; charset=utf-8", **(headers or {})}; self.closed = False
    def iter_content(self, chunk_size): yield self.body
    def close(self): self.closed = True


class _Session:
    verify = True
    def __init__(self, body, status=200, response_url=None, response_headers=None):
        self.body = body; self.status = status; self.response_url = response_url
        self.response_headers = response_headers or {}; self.calls = []; self.headers = {}; self.responses = []
    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        response_url = self.response_url or (url + "&" + urlencode(kwargs.get("params", {})))
        response = _Response(self.body, self.status, response_url, self.response_headers)
        self.responses.append(response); return response


def test_collection_keeps_raw_hash_request_and_retry_state(tmp_path):
    day = "2026-10-08"
    body = _page([("중도매", "채소류", "무", "고랭지", day, "광주지역", "최지혜", "공급 전망")])
    session = _Session(body)
    ckpt = pilot.collect([day], tmp_path, session=session, sleep=lambda _: None)
    assert len(session.calls) == 1
    url, kwargs = session.calls[0]
    assert url == pilot.LIST_URL and kwargs["verify"] is True and kwargs["allow_redirects"] is False
    assert kwargs["params"] == {"regday": "2026.10.08", "itemcategorycode": "", "itemcode": "",
                                 "kindcode": "", "period": "1", "productclscode": "", "countycode": ""}
    assert session.headers["Referer"] == pilot.URL
    entry = ckpt["requests"][day]
    manifest = json.loads((tmp_path / entry["attempts"][0]["manifestFile"]).read_text())
    raw = (tmp_path / manifest["rawFile"]).read_bytes()
    assert raw == body and manifest["responseSha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["method"] == "GET" and manifest["requestUrl"].startswith(pilot.LIST_URL)
    assert "regday=2026.10.08" in manifest["requestUrl"]
    assert manifest["records"][0]["crop"] == "무" and manifest["retrievedAtUtc"].endswith("+00:00")
    pilot.collect([day], tmp_path, session=session, sleep=lambda _: None)
    assert len(session.calls) == 1


def test_rate_limit_is_checkpointed_without_retry_or_later_date_requests(tmp_path):
    session = _Session(b"slow down", status=429)
    checkpoint = pilot.collect(["2026-10-08", "2026-10-09"], tmp_path, session=session, sleep=lambda _: None)
    assert len(session.calls) == 1
    assert checkpoint["requests"]["2026-10-08"]["state"] == "rate_limited"
    assert "2026-10-09" not in checkpoint["requests"]
    again = pilot.collect(["2026-10-08", "2026-10-09"], tmp_path, session=session, sleep=lambda _: None)
    assert len(session.calls) == 1 and "2026-10-09" not in again["requests"]


def test_retry_after_on_service_error_checkpoints_and_pauses_sweep(tmp_path):
    session = _Session(b"temporary unavailable", status=503, response_headers={"Retry-After": "120"})
    checkpoint = pilot.collect(["2026-10-08", "2026-10-09"], tmp_path, session=session, sleep=lambda _: None)
    assert len(session.calls) == 1
    assert checkpoint["requests"]["2026-10-08"]["state"] == "retry_after_pause"
    manifest = json.loads((tmp_path / checkpoint["requests"]["2026-10-08"]["attempts"][0]["manifestFile"]).read_text())
    assert manifest["responseHeaders"]["Retry-After"] == "120"
    assert session.responses[0].closed
    assert "2026-10-09" not in checkpoint["requests"]


def test_cross_origin_response_is_blocked_and_not_retried(tmp_path):
    session = _Session(b"redirect", status=302, response_url="http://elsewhere.example/path")
    checkpoint = pilot.collect(["2026-10-08", "2026-10-09"], tmp_path, session=session, sleep=lambda _: None)
    assert len(session.calls) == 1
    assert session.calls[0][1]["allow_redirects"] is False
    assert checkpoint["requests"]["2026-10-08"]["state"] == "redirect_blocked"
    assert "2026-10-09" not in checkpoint["requests"]


def test_revalidation_writes_separate_report_and_leaves_source_artifacts_unchanged(tmp_path):
    day = "2025-05-06"
    body = '<h3 class="s_tit fl">검색된 데이터가 없습니다.</h3>'.encode()
    ckpt = pilot.collect([day], tmp_path, session=_Session(body), sleep=lambda _: None)
    source_names = ["checkpoint.json", *(x for a in ckpt["requests"][day]["attempts"]
                                           for x in (a["manifestFile"], a["rawFile"]))]
    before = {name: (tmp_path / name).read_bytes() for name in source_names}
    result = pilot.revalidate_output(tmp_path, now=lambda: __import__('datetime').datetime(2026, 10, 9, tzinfo=__import__('datetime').timezone.utc))
    assert result["dateCount"] == 1 and result["attemptCount"] == 1
    assert result["dates"] == [{"date": day, "attemptCount": 1, "latestState": "normal_empty",
                                "allAttemptsEvidenceVerified": True}]
    assert result["attempts"][0]["newState"] == "normal_empty"
    assert result["attempts"][0]["evidenceVerified"] is True
    assert (tmp_path / result["reportFile"]).exists()
    assert before == {name: (tmp_path / name).read_bytes() for name in source_names}


def test_revalidation_fails_closed_after_raw_tampering(tmp_path):
    day = "2025-05-06"
    body = '<h3 class="s_tit fl">검색된 데이터가 없습니다.</h3>'.encode()
    checkpoint = pilot.collect([day], tmp_path, session=_Session(body), sleep=lambda _: None)
    raw_name = checkpoint["requests"][day]["attempts"][0]["rawFile"]
    (tmp_path / raw_name).write_bytes(b"x" * len(body))
    result = pilot.revalidate_output(tmp_path)
    assert result["attempts"][0]["evidenceVerified"] is False
    assert result["attempts"][0]["parsedState"] == "invalid_html"
    assert result["attempts"][0]["newState"] == "questionable"
    assert result["dates"][0]["latestState"] == "questionable"
