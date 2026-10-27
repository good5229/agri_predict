import datetime as dt
import hashlib
import json

import pytest

from agri_data import nongnet_national_vintage_capture as capture


TODAY = dt.date(2026, 10, 8)


def _body(date="20261007", price="873", weight="123"):
    return json.dumps({"datalist": [{"saledate": date, "selectDay": price,
                                      "selectDayWgt": weight}]}).encode()


def test_six_frozen_queries_and_default_dry_run(monkeypatch, capsys):
    matrix = capture.request_matrix("2026-10-07", today=TODAY)
    assert len(matrix) == 6
    assert [r["category"] for r in matrix] == ["cabbage", "radish", "onion", "pepper", "garlic_all", "garlic_peeled"]
    assert [(r["searchSymbol2"], r["searchSymbol3"]) for r in matrix] == [
        ("1001", ""), ("1101", ""), ("1201", ""), ("1207", ""), ("1209", ""), ("1209", "120906")]
    assert all(r["searchUnitGrade"] == "normal" and r["searchSymbol1"] == "100000"
               and r["searchDate"] == "2026년 10월 07일" for r in matrix)
    monkeypatch.setattr(capture.requests, "Session", lambda: (_ for _ in ()).throw(AssertionError("network")))
    assert capture.main(["--date", "2026-10-07"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["requestSent"] is False and report["requestCount"] == 6


def test_invalid_or_future_observation_dates_are_rejected():
    with pytest.raises(ValueError):
        capture.request_matrix("not-a-date", today=TODAY)
    with pytest.raises(ValueError):
        capture.request_matrix("2026-10-09", today=TODAY)


class Response:
    status_code = 200
    headers = {"Content-Type": "application/json", "Set-Cookie": "secret", "X-Mock": "yes"}

    def __init__(self, body):
        self.body = body

    def iter_content(self, chunk_size):
        yield self.body

    def close(self):
        pass


class Session:
    def __init__(self, bodies):
        self.bodies = iter(bodies)
        self.calls = []
        self.headers = {}
        self.verify = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return Response(next(self.bodies))


def test_capture_preserves_hashes_provenance_zero_and_new_run_versions(tmp_path, monkeypatch):
    monkeypatch.setattr(capture.time, "monotonic", iter(range(100)).__next__)
    stamp = iter([dt.datetime(2026, 10, 8, 1, 2, 3, i, tzinfo=dt.timezone.utc) for i in range(20)])
    now = lambda: next(stamp)
    bodies = [_body(price="0" if i == 4 else "873", weight="0" if i == 4 else "123") for i in range(6)]
    session = Session(bodies)
    first = capture.collect("2026-10-07", tmp_path, session=session, now=now,
                            sleep=lambda _: None, today=TODAY)
    assert len(first["results"]) == len(session.calls) == 6
    assert session.verify is True and session.headers["User-Agent"] == "Mozilla/5.0"
    assert all(call[1]["verify"] is True and call[1]["timeout"] == capture.TIMEOUT_SECONDS
               for call in session.calls)
    assert first["results"][4]["state"] == "questionable"
    manifests = sorted(tmp_path.glob("*.manifest.json"))
    assert len(manifests) == 6
    for path in manifests:
        manifest = json.loads(path.read_text())
        assert manifest["complete"] is False
        assert manifest["runId"] == first["runId"]
        assert manifest["requestedDate"] == "2026-10-07"
        assert manifest["responseHeaders"]["Set-Cookie"] == "[redacted]"
        raw = (tmp_path / manifest["rawFile"]).read_bytes()
        assert len(raw) == manifest["responseBytes"]
        assert hashlib.sha256(raw).hexdigest() == manifest["sha256"]

    second = capture.collect("2026-10-07", tmp_path,
                             session=Session([_body() for _ in range(6)]), now=now,
                             sleep=lambda _: None, today=TODAY)
    assert second["runId"] != first["runId"]
    assert len(list(tmp_path.glob("*.manifest.json"))) == 12


def test_empty_response_and_transport_failure_are_distinct():
    assert capture.history.validate_json(b"", "2026-10-07")["state"] == "empty_response"
    failure = capture.history.validate_json(b"", "2026-10-07", transport_error="request_error:Timeout")
    assert failure["state"] == "transport_error"


@pytest.mark.parametrize("number,issue", [
    (b"NaN", "nonstandard_json_constant_nan"),
    (b"Infinity", "nonstandard_json_constant_infinity"),
    (b"-Infinity", "nonstandard_json_constant_-infinity"),
    (b"1e999", "json_number_overflow_nonfinite"),
])
def test_nonfinite_and_overflow_json_is_classified_without_nonstandard_manifest_values(number, issue):
    body = b'{"datalist":[{"saledate":"20261007","selectDay":' + number + b',"selectDayWgt":"1"}]}'
    validation = capture.validate_json(body, "2026-10-07")
    assert validation["state"] == "nonfinite_json_value"
    assert validation["issues"] == [issue]
    assert "NaN" not in json.dumps(validation, allow_nan=False)
    assert "Infinity" not in json.dumps(validation, allow_nan=False)
