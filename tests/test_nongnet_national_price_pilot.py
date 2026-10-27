import datetime as dt
import hashlib
import json

from agri_data import nongnet_national_price_pilot as pilot


class Response:
    def __init__(self, status, body, headers=None):
        self.status_code = status
        self.body = body
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8", "X-Reply": "mock"}
        self.closed = False

    def iter_content(self, chunk_size):
        for i in range(0, len(self.body), 13):
            yield self.body[i:i+13]

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


def _fields(target=None):
    target = target or pilot.TARGETS[0]
    return pilot._params(target)


def _html(requested=None, date=None, item_name=None, heading_item=None, average="1,164", unit="1kg"):
    requested = dict(requested or _fields())
    if date is not None:
        requested["searchDate"] = date
    if item_name is not None:
        requested["searchName2"] = item_name
    heading_item = requested["searchName2"] if heading_item is None else heading_item
    fields = "".join(f'<input name="{key}" value="{value}">' for key, value in requested.items())
    return f'''<!doctype html><html><body><form>{fields}</form>
      <div class="tradeWrap"><div class="dataTit"><h4><span class="date">[{requested['searchDate']}]</span>
      <em>{requested['searchName1']}</em><em>{heading_item}</em><em>{requested['searchName3']}</em></h4></div>
      <div class="priceChartAll"><div class="itemList">
      <div class="item"><p class="grade">고</p><p class="price"><strong>1,500</strong><span>1kg</span></p></div>
      <div class="item"><p class="grade">중</p><p class="price"><strong>1,200</strong><span>1kg</span></p></div>
      <div class="item"><p class="grade">저</p><p class="price"><strong>900</strong><span>1kg</span></p></div>
      <div class="item"><p class="grade">평균</p><p class="price"><strong>{average}</strong><span>{unit}</span></p></div>
      </div></div><div class="periodSummary"><p class="grade">평균</p><strong>9,999</strong></div>
      </div></body></html>'''.encode()


def _now():
    return dt.datetime(2026, 10, 8, tzinfo=dt.timezone.utc)


def test_frozen_eight_request_matrix_and_dry_run(capsys):
    matrix = pilot.request_matrix()
    assert len(matrix) == 8
    assert [(x["targetDate"], x["itemCode"], x["varietyCode"]) for x in matrix] == [
        ("2026-09-11", "1001", ""), ("2026-09-12", "1001", ""),
        ("2026-09-14", "1001", ""), ("2026-09-11", "1101", ""),
        ("2026-09-11", "1201", ""), ("2026-09-11", "1207", ""),
        ("2026-09-11", "1209", ""), ("2026-09-11", "1209", "120906")]
    assert matrix[0]["searchDate"] == "2026년 09월 11일"
    assert matrix[0]["searchSymbol1"] == "100000"
    assert pilot.main([]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["requestSent"] is False and plan["requestCount"] == 8


def test_parser_extracts_mean_1kg_separately_from_grades_and_period_summary():
    result = pilot.parse_html(_html(), _fields())
    assert result["state"] == "valid"
    assert result["average"] == {"priceSource": "1,164", "unit": "1kg"}
    assert result["grades"]["고"]["priceSource"] == "1,500"
    assert result["grades"]["중"]["priceSource"] == "1,200"
    assert result["grades"]["저"]["priceSource"] == "900"
    assert result["average"]["priceSource"] != "9,999"


def test_parser_rejects_echoed_date_item_or_variety_mismatch():
    expected = _fields(pilot.TARGETS[-1])
    mismatch_date = pilot.parse_html(_html(expected, date="2026년 09월 10일"), expected)
    assert mismatch_date["state"] == "selection_mismatch"
    assert "searchDate" in mismatch_date["mismatches"]
    mismatch_item = pilot.parse_html(_html(expected, item_name="양파"), expected)
    assert mismatch_item["state"] == "selection_mismatch"
    assert "searchName2" in mismatch_item["mismatches"]
    visible_mismatch = pilot.parse_html(_html(expected, heading_item="양파"), expected)
    assert visible_mismatch["state"] == "selection_mismatch"
    assert "visibleSelection" in visible_mismatch["mismatches"]
    mismatch_variety = dict(expected, searchName3="", searchSymbol3="")
    parsed = pilot.parse_html(_html(mismatch_variety), expected)
    assert parsed["state"] == "selection_mismatch"
    assert "searchSymbol3" in parsed["mismatches"]


def test_empty_error_missing_average_and_bad_unit_states():
    expected = _fields()
    assert pilot.parse_html(b"", expected)["state"] == "empty_response"
    assert pilot.parse_html("<!doctype html><html><body>오류가 발생했습니다</body></html>".encode(), expected)["state"] == "error_html"
    bare_empty = pilot.parse_html("<!doctype html><html><body>검색된 데이터가 없습니다</body></html>".encode(), expected)
    assert bare_empty["state"] == "selection_mismatch"
    assert "visibleDate" in bare_empty["mismatches"]
    matching_empty = _html(expected).decode()
    panel_start = matching_empty.index('<div class="priceChartAll">')
    panel_end = matching_empty.index('</div></div><div class="periodSummary">', panel_start) + len('</div></div>')
    matching_empty = (matching_empty[:panel_start] + '<p>검색된 데이터가 없습니다</p>' + matching_empty[panel_end:]).encode()
    assert pilot.parse_html(matching_empty, expected)["state"] == "normal_empty"
    wrong_filter_empty = _html(expected, item_name="양파").decode()
    panel_start = wrong_filter_empty.index('<div class="priceChartAll">')
    panel_end = wrong_filter_empty.index('</div></div><div class="periodSummary">', panel_start) + len('</div></div>')
    wrong_filter_empty = (wrong_filter_empty[:panel_start] + '<p>검색된 데이터가 없습니다</p>' + wrong_filter_empty[panel_end:]).encode()
    assert pilot.parse_html(wrong_filter_empty, expected)["state"] == "selection_mismatch"
    without_mean = _html(expected).decode().replace(
        '<div class="item"><p class="grade">평균</p><p class="price"><strong>1,164</strong><span>1kg</span></p></div>',
        '').encode()
    assert pilot.parse_html(without_mean, expected)["state"] == "missing_average"
    bad_unit = pilot.parse_html(_html(unit="1상자"), expected)
    assert bad_unit["state"] == "questionable"
    assert "average_unit_not_1kg" in bad_unit["issues"]


def test_collect_preserves_raw_response_and_exact_params(tmp_path):
    responses = [Response(200, _html(_fields(target))) for target in pilot.TARGETS]
    session = Session(responses)
    results = pilot.collect(tmp_path, session=session, now=_now, sleep=lambda _: None)
    assert len(results) == len(session.calls) == 8
    assert session.verify is True
    assert session.headers["User-Agent"] == "Mozilla/5.0"
    for index, (url, kwargs) in enumerate(session.calls):
        assert url == pilot.URL
        assert kwargs["verify"] is True and kwargs["timeout"] == pilot.TIMEOUT_SECONDS
        assert kwargs["params"] == _fields(pilot.TARGETS[index])
    metas = list(tmp_path.glob("*.manifest.json"))
    assert len(metas) == 8
    for path in metas:
        meta = json.loads(path.read_text())
        raw = (tmp_path / meta["rawFile"]).read_bytes()
        assert len(raw) == meta["responseBytes"]
        assert hashlib.sha256(raw).hexdigest() == meta["sha256"]
        assert dt.datetime.fromisoformat(meta["retrievedAtUtc"]).tzinfo is not None
        assert meta["complete"] is False


def test_bounded_body_preserves_prefix_and_marks_truncation():
    response = Response(200, b"x" * (pilot.MAX_RESPONSE_BYTES + 1))
    body, truncated, error = pilot._read_bounded(response)
    assert len(body) == pilot.MAX_RESPONSE_BYTES
    assert truncated and "5mb" in error
    assert response.closed
