import datetime as dt
import json

from agri_data import nongnet_national_history_screen as screen


def _row(date="20260911", price="599", weight="12"):
    return {"saledate": date, "selectDay": price, "selectDayWgt": weight}


def _body(rows):
    return json.dumps({"datalist": rows}).encode()


def test_exact_23_query_matrix_and_dry_run_has_no_network(monkeypatch, capsys):
    matrix = screen.request_matrix()
    assert len(matrix) == 23
    assert sum(r["requestedDate"] == "2026-09-11" for r in matrix) == 5
    assert all(r["searchUnitGrade"] == "normal" and r["searchSymbol1"] == "100000" for r in matrix)
    assert matrix[0]["searchDate"] == "2023년 01월 03일"
    garlic = [r for r in matrix if r["category"].startswith("garlic") and r["requestedDate"] == "2023-01-03"]
    assert [(r["searchSymbol2"], r["searchSymbol3"]) for r in garlic] == [("1209", ""), ("1209", "120906")]
    monkeypatch.setattr(screen.requests, "Session", lambda: (_ for _ in ()).throw(AssertionError("network session created")))
    assert screen.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["requestSent"] is False and report["requestCount"] == 23


def test_reviewed_exact_date_validation_states_and_zero_quality():
    assert screen.validate_json(b"", "2026-09-11")["state"] == "empty_response"
    missing = screen.validate_json(_body([_row("20260912")]), "2026-09-11")
    assert missing["state"] == "requested_date_missing"
    duplicate = screen.validate_json(_body([_row(), _row()]), "2026-09-11")
    assert duplicate["state"] == "duplicate_requested_date"
    zero = screen.validate_json(_body([_row(price="0", weight="0")]), "2026-09-11")
    assert zero["state"] == "questionable"
    assert zero["matchingRow"]["selectDay"] == "0"
    assert zero["matchingRow"]["selectDayWgt"] == "0"
    assert "selectDay_zero_value" in zero["matchingRow"]["issues"]
    assert "selectDayWgt_zero_value" in zero["matchingRow"]["issues"]


def test_malformed_nonmatching_date_is_flagged_and_matching_row_issues_stay_synced():
    result = screen.validate_json(
        _body([_row(price="0", weight="4"), _row("20261399", "10", "2")]),
        "2026-09-11",
    )
    assert result["state"] == "questionable"
    assert result["rows"][1]["dateMatch"] is False
    assert "saledate_invalid_format" in result["rows"][1]["issues"]
    assert result["matchingRow"]["issues"] == result["rows"][0]["issues"]
    assert "selectDay_zero_value" in result["matchingRow"]["issues"]


def test_reuse_requires_pinned_json_and_original_query_provenance(tmp_path):
    existing = screen.reuse_existing_cabbage()
    assert existing["state"] == "reused_existing_json"
    assert existing["requestParams"] == screen.EXISTING_CABBAGE_PARAMS

    arbitrary = tmp_path / "some-other-2026-09-11.json"
    arbitrary.write_bytes(_body([_row()]))
    rejected = screen.reuse_existing_cabbage(arbitrary)
    assert rejected["state"] == "existing_json_hash_mismatch"


class Response:
    status_code = 200
    headers = {"Content-Type": "application/json", "X-Mock": "1"}

    def __init__(self, body):
        self.body = body
        self.closed = False

    def iter_content(self, chunk_size):
        yield self.body

    def close(self):
        self.closed = True


class Session:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.headers = {}
        self.calls = []
        self.verify = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


def test_collector_preserves_attempt_and_manifest_with_limits(tmp_path, monkeypatch):
    matrix = screen.request_matrix()
    session = Session([Response(_body([_row(date=r["requestedDate"].replace("-", ""))])) for r in matrix])
    monkeypatch.setattr(screen.time, "monotonic", iter(range(100)).__next__)
    results = screen.collect(tmp_path, session=session,
                             now=lambda: dt.datetime(2026, 10, 8, tzinfo=dt.timezone.utc),
                             sleep=lambda _: None)
    assert len(results) == len(session.calls) == 23
    assert session.verify and session.headers["User-Agent"] == "Mozilla/5.0"
    assert all(call[1]["verify"] is True and call[1]["timeout"] == screen.TIMEOUT_SECONDS
               for call in session.calls)
    manifests = list(tmp_path.glob("*.manifest.json"))
    assert len(manifests) == 23
    for path in manifests:
        manifest = json.loads(path.read_text())
        assert manifest["complete"] is False
        assert manifest["responseBytes"] == len((tmp_path / manifest["rawFile"]).read_bytes())
        assert manifest["sha256"]
