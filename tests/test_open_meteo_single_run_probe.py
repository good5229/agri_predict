import datetime as dt
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import requests

from agri_data import open_meteo_single_run_probe as probe


def valid_body():
    start = dt.datetime(2025, 8, 31, 12, tzinfo=dt.timezone.utc)
    times = [(start + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(240)]
    data = {"latitude": 34.5, "longitude": 126.625, "elevation": 20.0,
            "utc_offset_seconds": 0, "timezone": "UTC", "timezone_abbreviation": "GMT",
            "model": "ecmwf_ifs", "hourly_units": {"temperature_2m": "°C", "precipitation": "mm"},
            "hourly": {"time": times, "temperature_2m": [12.5] * 240,
                       "precipitation": [0.0] * 240}}
    return json.dumps(data, separators=(",", ":")).encode()


def observed_capture_body():
    data = json.loads(valid_body())
    data.update(latitude=34.622143, longitude=126.59091, elevation=36.0,
                timezone="GMT", utc_offset_seconds=0, timezone_abbreviation="GMT")
    data.pop("model", None)
    data["hourly"]["precipitation"][0] = None
    return json.dumps(data, separators=(",", ":")).encode()


class Response:
    def __init__(self, body, status=200, headers=None):
        self.body, self.status_code = body, status
        self.headers = headers or {"Content-Type": "application/json", "Date": "Wed, 01 Oct 2025 00:00:00 GMT"}
        self.closed = False

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.body), 97):
            yield self.body[offset:offset + 97]

    def close(self):
        self.closed = True


def fixed_now():
    return dt.datetime(2025, 9, 1, tzinfo=dt.timezone.utc)


def test_dry_run_never_sends_request(capsys):
    with mock.patch.object(probe, "probe", side_effect=AssertionError("network path invoked")):
        assert probe.main([]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["request_sent"] is False
    assert result["request"]["models"] == "ecmwf_ifs"
    assert result["request"]["forecast_days"] == 10
    assert "api_key" not in json.dumps(result).lower()


def test_valid_capture_is_exact_immutable_and_provenanced(tmp_path):
    body = valid_body()
    response = Response(body)
    session = SimpleNamespace(get=mock.Mock(return_value=response))
    result = probe.probe(raw_dir=tmp_path, session=session, now=fixed_now(), sleeper=lambda _: None)
    raw_path = tmp_path / result["raw_file"]
    meta_path = tmp_path / result["metadata_file"]
    assert raw_path.read_bytes() == body
    assert result["sha256"] == hashlib.sha256(body).hexdigest()
    assert result["complete"] is True
    assert result["validation"]["null_counts"]["temperature_2m"] == 0
    assert result["run_initialization_utc"] == "2025-08-31T12:00"
    assert result["public_release_time"] is None
    assert response.closed
    assert session.get.call_count == 1
    kwargs = session.get.call_args.kwargs
    assert kwargs["verify"] is True and kwargs["allow_redirects"] is False
    assert kwargs["params"]["timezone"] == "UTC"
    with pytest.raises(FileExistsError):
        probe._atomic_immutable(raw_path, b"replacement")
    assert json.loads(meta_path.read_text())["sha256"] == result["sha256"]


def test_gmt_zero_offset_capture_preserves_partial_precipitation_coverage(tmp_path):
    body = observed_capture_body()
    response = Response(body)
    result = probe.probe(raw_dir=tmp_path, session=SimpleNamespace(get=lambda *a, **kw: response),
                         now=fixed_now(), sleeper=lambda _: None)
    coverage = result["validation"]["variable_coverage"]
    assert (tmp_path / result["raw_file"]).read_bytes() == body
    assert result["validation"]["valid"] is True
    assert result["complete"] is False
    assert result["validation"]["response_timezone"] == {
        "timezone": "GMT", "utc_offset_seconds": 0, "timezone_abbreviation": "GMT"}
    assert coverage["temperature_2m"] == {
        "array_present": True, "expected_count": 240, "returned_count": 240,
        "numeric_count": 240, "null_count": 0, "invalid_count": 0,
        "coverage_fraction": 1.0, "complete": True}
    assert coverage["precipitation"]["numeric_count"] == 239
    assert coverage["precipitation"]["null_count"] == 1
    assert coverage["precipitation"]["coverage_fraction"] == pytest.approx(239 / 240)
    assert coverage["precipitation"]["complete"] is False
    assert result["validation"]["incomplete_variables"] == ["precipitation"]
    assert result["validation"]["null_counts"]["precipitation"] == 1
    assert result["validation"]["response_model"] is None
    assert result["request"]["models"] == "ecmwf_ifs"


@pytest.mark.parametrize("status,body,issue", [
    (429, b'{"reason":"rate limited"}', "http_status_not_success"),
    (200, b"", "response_not_json"),
    (200, b"{broken", "response_not_json"),
])
def test_http_errors_and_invalid_json_are_saved_incomplete(tmp_path, status, body, issue):
    response = Response(body, status)
    result = probe.probe(raw_dir=tmp_path, session=SimpleNamespace(get=lambda *a, **kw: response),
                         now=fixed_now(), sleeper=lambda _: None)
    assert (tmp_path / result["raw_file"]).read_bytes() == body
    assert result["complete"] is False
    assert issue in result["validation"]["issues"]
    assert result["http_status"] == status


def test_incomplete_horizon_mismatch_and_null_values_are_not_filled(tmp_path):
    data = json.loads(valid_body())
    data["hourly"]["precipitation"] = data["hourly"]["precipitation"][:-1]
    data["hourly"]["temperature_2m"][0] = None
    body = json.dumps(data).encode()
    result = probe.probe(raw_dir=tmp_path, session=SimpleNamespace(get=lambda *a, **kw: Response(body)),
                         now=fixed_now(), sleeper=lambda _: None)
    assert result["complete"] is False
    assert "precipitation_length_mismatch" in result["validation"]["issues"]
    assert result["validation"]["variable_coverage"]["temperature_2m"]["complete"] is False
    assert result["validation"]["variable_coverage"]["temperature_2m"]["null_count"] == 1
    assert json.loads((tmp_path / result["raw_file"]).read_bytes())["hourly"]["temperature_2m"][0] is None


def test_only_transport_failures_retry_and_429_is_not_retried(tmp_path):
    rate_limited = Response(b'{"error":true}', 429)
    session = SimpleNamespace(get=mock.Mock(side_effect=[requests.Timeout("temporary"), rate_limited]))
    result = probe.probe(raw_dir=tmp_path, session=session, now=fixed_now(), sleeper=lambda _: None)
    assert session.get.call_count == 2
    assert result["attempts"] == 2
    assert result["http_status"] == 429
    assert (tmp_path / result["raw_file"]).read_bytes() == b'{"error":true}'


def test_oversize_response_saves_bounded_partial_capture(tmp_path):
    response = Response(b"x" * (probe.MAX_RESPONSE_BYTES + 1))
    result = probe.probe(raw_dir=tmp_path, session=SimpleNamespace(get=lambda *a, **kw: response),
                         now=fixed_now(), sleeper=lambda _: None)
    assert len((tmp_path / result["raw_file"]).read_bytes()) == probe.MAX_RESPONSE_BYTES
    assert result["complete"] is False
    assert result["transfer_state"] == "incomplete"
    assert result["transfer_error"] == "response_exceeded_maximum_size"


def test_stream_failure_persists_partial_body_status_and_metadata(tmp_path):
    class BrokenResponse(Response):
        def iter_content(self, chunk_size):
            yield b'{"partial":'
            raise requests.ConnectionError("interrupted")

    response = BrokenResponse(b"unused", 503)
    result = probe.probe(raw_dir=tmp_path, session=SimpleNamespace(get=lambda *a, **kw: response),
                         now=fixed_now(), sleeper=lambda _: None)
    assert (tmp_path / result["raw_file"]).read_bytes() == b'{"partial":'
    assert result["http_status"] == 503
    assert result["complete"] is False
    assert result["transfer_error"] == "response_stream_failed"
    saved = json.loads((tmp_path / result["metadata_file"]).read_text())
    assert saved["transfer_state"] == "incomplete"


def test_conflicting_echoed_model_makes_response_invalid():
    data = json.loads(valid_body())
    data["model"] = "some_other_model"
    validation = probe.validate_response(json.dumps(data).encode(), 200)
    assert validation["valid"] is False
    assert "response_model_conflicts_with_requested_model" in validation["issues"]


def test_hourly_series_starts_at_initialization_and_spans_240_hours():
    data = json.loads(valid_body())
    start = dt.datetime(2025, 8, 31, 13, tzinfo=dt.timezone.utc)
    data["hourly"]["time"] = [(start + dt.timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M")
                               for i in range(240)]
    validation = probe.validate_response(json.dumps(data).encode(), 200)
    assert "hourly_time_start_does_not_match_run" in validation["issues"]
    data = json.loads(valid_body())
    data["hourly"]["time"][0] = "2025-08-31T11:00"
    validation = probe.validate_response(json.dumps(data).encode(), 200)
    assert "hourly_time_start_does_not_match_run" in validation["issues"]
    data = json.loads(valid_body())
    assert probe.validate_response(json.dumps(data).encode(), 200)["valid"] is True


def test_non_numeric_value_and_bad_time_make_response_incomplete():
    data = json.loads(valid_body())
    data["hourly"]["temperature_2m"][1] = float("nan")
    data["hourly"]["time"][1] = "bad"
    validation = probe.validate_response(json.dumps(data).encode(), 200)
    assert validation["valid"] is False
    assert "hourly_time_not_iso8601_hour" in validation["issues"]
    assert "temperature_2m_contains_non_finite_or_non_numeric_value" in validation["issues"]
