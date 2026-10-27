import datetime as dt
import hashlib
import json
import requests

from agri_data import seoan_pepper_collection as collection


class Response:
    def __init__(self, status, body, headers=None):
        self.status_code = status
        self.body = body
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.closed = False

    def iter_content(self, chunk_size):
        for pos in range(0, len(self.body), 7):
            yield self.body[pos:pos + 7]

    def close(self):
        self.closed = True


class Session:
    def __init__(self, post_bodies):
        self.post_bodies = iter(post_bodies)
        self.calls = []
        self.headers = {}
        self.verify = False

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        return Response(200, b"<html>form</html>")

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        return next(self.post_bodies)


HTML = """<table><tr><th>경락일</th><th>품종</th><th>중량단위</th><th>물량</th><th>가격단위</th><th>경락가</th></tr>
<tr><td>2024-08-01</td><td>화건</td><td>kg</td><td>12.50</td><td>원/kg</td><td>1,234</td></tr>
<tr><td>2024-08-02</td><td>화건(손꼭무)</td><td>근</td><td>-3</td><td>원/근</td><td>900</td></tr>
<tr><td>2024-08-03</td><td>기타</td><td>kg</td><td></td><td>원/kg</td><td>abc</td></tr></table>""".encode()


def test_dry_run_is_fixed_and_sends_nothing(capsys):
    assert collection.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "dry_run"
    assert report["request_sent"] is False
    assert [(x["start"], x["end"]) for x in report["requests"]] == list(collection.WEEKS)
    assert all(x["params"]["searchSpcs"] == "" for x in report["requests"])


def test_validator_preserves_questionable_and_other_product_rows():
    result = collection.validate_body(HTML, "2024-08-01", "2024-08-07", "text/html")
    assert result["format"] == "html_table"
    assert len(result["rows"]) == 3
    assert result["rows"][1]["weight"] == "-3"
    assert "weight_negative_value" in result["rows"][1]["issues"]
    assert result["rows"][2]["product"] == "기타"
    assert "weight_null_value" in result["rows"][2]["issues"]
    assert "price_non_numeric_value" in result["rows"][2]["issues"]
    assert result["completeness"] == "unverified_subset_or_full_response"


def test_validator_marks_error_html_and_partial_output():
    error = collection.validate_body("<html>오류가 발생했습니다</html>".encode(), *collection.WEEKS[0])
    assert "error_html_or_text" in error["issues"]
    partial = collection.validate_body(b"<table><tr><th>hello</th></tr></table>", *collection.WEEKS[0])
    assert "no_data_rows_or_partial_output" in partial["issues"]
    schema = collection.validate_body(
        b"<table><tr><th>date</th><th>price</th><th>weight</th></tr><tr><td>2024-08-01</td><td>5</td><td>2</td></tr></table>",
        *collection.WEEKS[0])
    assert "unrecognized_table_schema" in schema["issues"]


def test_collect_saves_raw_bytes_hash_and_request_provenance(tmp_path):
    bodies = [Response(200, HTML), Response(200, HTML)]
    session = Session(bodies)
    manifests = collection.collect(tmp_path, session=session, sleep=lambda _: None,
                                   now=lambda: dt.datetime(2026, 10, 8, tzinfo=dt.timezone.utc))
    assert session.verify is True
    assert session.headers["User-Agent"] == "Mozilla/5.0"
    assert len([call for call in session.calls if call[0] == "POST"]) == 2
    assert all(call[2]["timeout"] == collection.TIMEOUT_SECONDS for call in session.calls)
    assert session.calls[1][2]["data"] == {"searchStartDt": "20240801", "searchEndDt": "20240807", "searchSpcs": ""}
    for manifest in manifests:
        captured = dt.datetime.fromisoformat(manifest["attempts"][-1]["retrieved_at_utc"])
        assert captured.tzinfo is not None
        landing = json.loads((tmp_path / manifest["landing_page_manifest_file"]).read_text())
        assert dt.datetime.fromisoformat(landing["retrieved_at_utc"]).tzinfo is not None
        raw = (tmp_path / manifest["raw_file"]).read_bytes()
        assert manifest["sha256"] == hashlib.sha256(raw).hexdigest()
        assert (tmp_path / manifest["landing_page_raw_file"]).read_bytes() == b"<html>form</html>"
        assert (tmp_path / manifest["landing_page_manifest_file"]).exists()
        assert manifest["complete"] is False
        assert manifest["first_publication_time"] == "unknown"
        saved = json.loads((tmp_path / manifest["attempts"][-1]["manifest_file"]).read_text())
        assert saved["request"]["searchSpcs"] == ""


def test_bounded_reader_closes_response_on_overflow():
    response = Response(200, b"x" * (collection.MAX_RESPONSE_BYTES + 1))
    body, truncated, error = collection._read_bounded(response)
    assert len(body) == collection.MAX_RESPONSE_BYTES
    assert truncated and "20mb" in error
    assert response.closed


def test_ui_paired_units_are_kept_as_separate_fields():
    body = ("<table><tr><th>날짜</th><th>품목</th><th>총수량(근)</th><th>총수량(kg)</th>"
            "<th>단가(근)</th><th>단가(kg)</th><th>비고</th></tr>"
            "<tr><td>2024-08-01</td><td>화건</td><td>10</td><td>6</td><td>300</td><td>500</td><td></td></tr>"
            "</table>").encode()
    result = collection.validate_body(body, "2024-08-01", "2024-08-07")
    row = result["rows"][0]
    assert row["quantity_geun"] == "10"
    assert row["quantity_kg"] == "6"
    assert row["price_geun"] == "300"
    assert row["price_kg"] == "500"
    assert "weight" not in row and "price" not in row


def test_nonfinite_numeric_values_are_flagged_and_preserved():
    body = ("<table><tr><th>날짜</th><th>품목</th><th>총수량(근)</th><th>총수량(kg)</th>"
            "<th>단가(근)</th><th>단가(kg)</th></tr>"
            "<tr><td>2024-08-01</td><td>화건</td><td>NaN</td><td>inf</td><td>-inf</td><td>2</td></tr></table>").encode()
    result = collection.validate_body(body, "2024-08-01", "2024-08-07")
    row = result["rows"][0]
    assert row["quantity_geun"] == "NaN"
    assert "quantity_geun_nonfinite_value" in row["issues"]
    assert "quantity_kg_nonfinite_value" in row["issues"]
    assert "price_geun_nonfinite_value" in row["issues"]


def test_retry_archives_5xx_response_before_success(tmp_path):
    session = Session([Response(503, b"busy"), Response(200, HTML), Response(200, HTML)])
    manifests = collection.collect(tmp_path, session=session, sleep=lambda _: None,
                                  now=lambda: dt.datetime(2026, 10, 8, tzinfo=dt.timezone.utc))
    first = manifests[0]["attempts"][0]
    assert first["http_status"] == 503
    assert (tmp_path / first["raw_file"]).read_bytes() == b"busy"
    assert (tmp_path / first["manifest_file"]).exists()
    assert len(manifests[0]["attempts"]) == 2


def test_stream_failure_keeps_bounded_prefix():
    class Interrupted(Response):
        def iter_content(self, chunk_size):
            yield b"prefix"
            raise requests.ConnectionError("cut off")

    response = Interrupted(200, b"")
    body, truncated, error = collection._read_bounded(response)
    assert body == b"prefix"
    assert truncated and "stream_error" in error
