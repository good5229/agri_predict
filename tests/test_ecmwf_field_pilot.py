import datetime as dt
import json
from pathlib import Path

from agri_data import ecmwf_field_pilot as pilot


class FakeResponse:
    def __init__(self, status, headers, body):
        self.status_code = status
        self.headers = headers
        self.body = body
        self.closed = False

    def iter_content(self, chunk_size):
        for start in range(0, len(self.body), 7):
            yield self.body[start : start + 7]

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, fail_index=None, fail_status=403):
        self.calls = []
        self.fail_index = fail_index
        self.fail_status = fail_status

    @staticmethod
    def _grib():
        body = bytearray(20)
        body[:4] = b"GRIB"
        body[7] = 2
        body[8:16] = len(body).to_bytes(8, "big")
        body[-4:] = b"7777"
        return bytes(body)

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url.endswith(".index"):
            # Infer the date and step from either historical or current path layout.
            file_date = url.rsplit("/", 1)[-1][:8]
            step = int(url.rsplit("/", 1)[-1].split("-")[1].removesuffix("h"))
            if self.fail_index in ((file_date, step), file_date):
                body = b"not found" if self.fail_status == 404 else b"denied!!"
                return FakeResponse(self.fail_status, {"Content-Length": str(len(body))}, body)
            rows = []
            for param, offset in (("2t", 0), ("tp", 64)):
                rows.append({"date": file_date, "time": "1200", "step": step,
                             "type": "fc", "stream": "oper", "levtype": "sfc",
                             "param": param, "_offset": offset, "_length": 20})
            body = "\n".join(json.dumps(row) for row in rows).encode()
            return FakeResponse(200, {"Content-Length": str(len(body))}, body)
        offset, end = map(int, kwargs["headers"]["Range"][6:].split("-"))
        body = self._grib()
        run_date = dt.datetime.strptime(url.rsplit("/", 1)[-1][:8], "%Y%m%d").date()
        modified = dt.datetime.combine(run_date + dt.timedelta(days=1), dt.time(0, 30), tzinfo=dt.timezone.utc)
        return FakeResponse(206, {
            "Content-Range": f"bytes {offset}-{end}/100",
            "Content-Length": str(len(body)),
            "Last-Modified": modified.strftime("%a, %d %b %Y %H:%M:%S GMT"),
        }, body)


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


def test_frozen_dates_are_run_dates_in_initialization_and_urls():
    run_date, run_init, index_url, grib_url = pilot._urls("20250914", 72)
    assert run_date == "20250914"
    assert run_init == "2025-09-14T12:00Z"
    assert "/20250914/12z/ifs/0p25/oper/20250914120000-72h-oper-fc.index" == index_url.removeprefix(pilot.BASE_URL)
    assert grib_url.endswith("20250914120000-72h-oper-fc.grib2")
    assert pilot._deadline_utc("20250914") == dt.datetime(2025, 9, 15, 1, tzinfo=dt.timezone.utc)


def test_ecmwf_pre_cutoff_path_omits_ifs_directory():
    run_date, run_init, index_url, grib_url = pilot._urls("20240115", 24)
    assert run_date == "20240115"
    assert run_init == "2024-01-15T12:00Z"
    assert index_url.removeprefix(pilot.BASE_URL) == "/20240115/12z/0p25/oper/20240115120000-24h-oper-fc.index"
    assert grib_url.removeprefix(pilot.BASE_URL) == "/20240115/12z/0p25/oper/20240115120000-24h-oper-fc.grib2"
    assert "/20240227/12z/0p25/oper/" in pilot._urls("20240227", 24)[2]
    assert "/20240228/12z/ifs/0p25/oper/" in pilot._urls("20240228", 24)[2]


def test_index_requires_one_exact_valid_record_per_parameter():
    run_date = "20250913"
    rows = [
        {"date": run_date, "time": "1200", "step": 24, "type": "fc", "stream": "oper",
         "levtype": "sfc", "param": "2t", "_offset": 0, "_length": 10},
        {"date": run_date, "time": "1200", "step": 24, "type": "fc", "stream": "oper",
         "levtype": "sfc", "param": "tp", "_offset": 10, "_length": 10},
    ]
    body = ("\n".join(json.dumps(row) for row in rows)).encode()
    selected, errors = pilot._parse_index(body, run_date, 24)
    assert set(selected) == {"2t", "tp"}
    assert errors == {"2t": [], "tp": []}

    duplicate = body + b"\n" + json.dumps(rows[0]).encode()
    selected, errors = pilot._parse_index(duplicate, run_date, 24)
    assert "2t" not in selected
    assert errors["2t"] == ["matching_index_record_not_unique"]

    malformed = dict(rows[1], _offset=-1)
    selected, errors = pilot._parse_index(
        (json.dumps(rows[0]) + "\n" + json.dumps(malformed)).encode(), run_date, 24)
    assert "tp" not in selected
    assert errors["tp"] == ["index_offset_invalid"]


def test_full_pilot_uses_27_requests_and_resumes_complete_checkpoint(tmp_path):
    session = FakeSession()
    fake_time = FakeTime()
    at = dt.datetime(2026, 10, 7, 0, 0, tzinfo=dt.timezone.utc)
    report = pilot.collect(raw_dir=tmp_path, session=session, retrieved_at=at,
                           sleeper=fake_time.sleep, clock=fake_time.clock)
    assert report["complete"] is True
    assert report["indices_complete"] is True
    assert len(session.calls) == 27
    assert len(fake_time.sleeps) == 26
    assert all(delay >= 1.0 for delay in fake_time.sleeps)
    assert all(call[1]["verify"] is True and call[1]["allow_redirects"] is False
               and call[1]["timeout"] == pilot.TIMEOUT_SECONDS for call in session.calls)
    ranges = [call[1]["headers"]["Range"] for call in session.calls
              if "Range" in (call[1].get("headers") or {})]
    assert ranges.count("bytes=0-19") == 9
    assert ranges.count("bytes=64-83") == 9
    assert all(field["field_latest"]["last_modified_before_origin"] is True
               for field in report["fields"] if field["parameter"] == "2t")
    assert all((tmp_path / attempt["raw_file"]).exists()
               for group in ("indices", "fields")
               for capture in json.loads((tmp_path / report["checkpoint_file"]).read_text())["captures"][group].values()
               for attempt in capture["attempts"] if attempt["raw_file"])

    resumed_session = FakeSession()
    resumed = pilot.collect(raw_dir=tmp_path, session=resumed_session, retrieved_at=at,
                            sleeper=fake_time.sleep, clock=fake_time.clock)
    assert resumed["complete"] is True
    assert resumed_session.calls == []


def test_dry_run_never_sends_requests(capsys):
    assert pilot.main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "dry_run"
    assert payload["request_sent"] is False
    assert payload["logical_requests_max"] == 27


def test_march_diagnostic_dry_run_isolated_and_bounded(capsys):
    assert pilot.main(["--march-2024-diagnostic"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["diagnostic"] is True
    assert payload["run_dates"] == ["20240315"]
    assert payload["logical_requests_max"] == 9
    assert payload["max_http_attempts_including_transport_retries"] == 27
    assert payload["raw_dir"] == "data/raw/ecmwf_march_2024_diagnostic"
    assert payload["request_sent"] is False


def test_march_diagnostic_collects_only_frozen_run_and_separate_directory(tmp_path):
    session = FakeSession()
    fake_time = FakeTime()
    raw_dir = tmp_path / "march-only"
    report = pilot.collect(raw_dir=raw_dir, session=session,
                           retrieved_at=dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc),
                           sleeper=fake_time.sleep, clock=fake_time.clock,
                           run_dates=pilot.MARCH_DIAGNOSTIC_RUN_DATES)
    assert report["complete"] is True
    assert report["status"] == "march_2024_diagnostic"
    assert report["run_dates"] == ["20240315"]
    assert len(report["fields"]) == 6
    assert report["logical_requests_max"] == 9
    assert len(session.calls) == 9
    assert all("/20240315/12z/ifs/0p25/oper/" in url for url, _ in session.calls)
    assert all(not any(f"/{other}/" in url for other in ("20240115", "20250914", "20260922"))
               for url, _ in session.calls)
    checkpoint = json.loads((raw_dir / report["checkpoint_file"]).read_text())
    assert checkpoint["run_dates"] == ["20240315"]


def test_checkpoint_scope_prevents_overwriting_frozen_run_with_diagnostic(tmp_path):
    fake_time = FakeTime()
    pilot.collect(raw_dir=tmp_path, session=FakeSession(),
                  retrieved_at=dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc),
                  sleeper=fake_time.sleep, clock=fake_time.clock)
    diagnostic_session = FakeSession()
    try:
        pilot.collect(raw_dir=tmp_path, session=diagnostic_session,
                      retrieved_at=dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc),
                      sleeper=FakeTime().sleep, clock=FakeTime().clock,
                      run_dates=pilot.MARCH_DIAGNOSTIC_RUN_DATES)
    except pilot.PilotError as exc:
        assert "different frozen run-date scope" in str(exc)
    else:
        raise AssertionError("diagnostic must not share the frozen pilot checkpoint")
    assert diagnostic_session.calls == []


def test_http_status_is_terminal_and_error_body_is_preserved(tmp_path):
    session = FakeSession(fail_index=("20240115", 24))
    fake_time = FakeTime()
    report = pilot.collect(raw_dir=tmp_path, session=session,
                           retrieved_at=dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc),
                           sleeper=fake_time.sleep, clock=fake_time.clock)
    assert report["complete"] is False
    assert report["indices_complete"] is False
    failed = json.loads((tmp_path / report["checkpoint_file"]).read_text())["captures"]["indices"]["20240115/24"]
    assert len(failed["attempts"]) == 1
    saved_body = (tmp_path / failed["latest"]["raw_file"]).read_bytes()
    assert saved_body == b"denied!!"
    assert failed["latest"]["state"] == "error"
    assert not any("20240115120000-24h-oper-fc.grib2" in url
                   for url, _ in session.calls)


def test_2024_retry_uses_legacy_layout_and_reuses_completed_later_runs(tmp_path):
    at = dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc)
    initial_session = FakeSession(fail_index="20240115", fail_status=404)
    initial_time = FakeTime()
    first = pilot.collect(raw_dir=tmp_path, session=initial_session, retrieved_at=at,
                          sleeper=initial_time.sleep, clock=initial_time.clock)
    assert first["complete"] is False
    first_checkpoint = json.loads((tmp_path / first["checkpoint_file"]).read_text())
    failed_index = first_checkpoint["captures"]["indices"]["20240115/24"]
    assert failed_index["latest"]["state"] == "unavailable"
    assert (tmp_path / failed_index["latest"]["raw_file"]).read_bytes() == b"not found"
    assert sum("/20240115/12z/ifs/" in url and url.endswith(".index")
               for url, _ in initial_session.calls) == 0
    assert sum("/20240115/12z/0p25/oper/" in url and url.endswith(".index")
               for url, _ in initial_session.calls) == 3

    retry_session = FakeSession()
    retry_time = FakeTime()
    resumed = pilot.collect(raw_dir=tmp_path, session=retry_session, retrieved_at=at,
                            sleeper=retry_time.sleep, clock=retry_time.clock)
    assert resumed["complete"] is True
    assert len(retry_session.calls) == 9  # only three failed indexes and six fields
    assert all("/20240115/12z/0p25/oper/" in url for url, _ in retry_session.calls)
    assert not any("/20250914/" in url or "/20260922/" in url
                   for url, _ in retry_session.calls)

    final_checkpoint = json.loads((tmp_path / resumed["checkpoint_file"]).read_text())
    retried_index = final_checkpoint["captures"]["indices"]["20240115/24"]
    assert len(retried_index["attempts"]) == 2
    assert retried_index["attempts"][0]["state"] == "unavailable"
    assert (tmp_path / retried_index["attempts"][0]["raw_file"]).read_bytes() == b"not found"
    assert retried_index["latest"]["complete"] is True


def test_resume_refetches_field_when_raw_or_metadata_artifact_is_damaged(tmp_path):
    at = dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc)
    first_session = FakeSession()
    first_time = FakeTime()
    report = pilot.collect(raw_dir=tmp_path, session=first_session, retrieved_at=at,
                           sleeper=first_time.sleep, clock=first_time.clock)
    assert report["complete"] is True
    checkpoint_path = tmp_path / report["checkpoint_file"]
    checkpoint = json.loads(checkpoint_path.read_text())
    field = checkpoint["captures"]["fields"]["20240115/24/2t"]["latest"]
    (tmp_path / field["raw_file"]).write_bytes(b"tampered")

    resumed_session = FakeSession()
    resumed_time = FakeTime()
    resumed = pilot.collect(raw_dir=tmp_path, session=resumed_session, retrieved_at=at,
                             sleeper=resumed_time.sleep, clock=resumed_time.clock)
    assert resumed["complete"] is True
    field_calls = [(url, kwargs) for url, kwargs in resumed_session.calls
                   if "Range" in (kwargs.get("headers") or {})
                   and "20240115120000-24h-oper-fc.grib2" in url]
    assert len(field_calls) == 1
    assert field_calls[0][1]["headers"]["Range"] == "bytes=0-19"
    assert resumed["checkpoint_file"] != report["checkpoint_file"]

    current_checkpoint = json.loads((tmp_path / resumed["checkpoint_file"]).read_text())
    current_field = current_checkpoint["captures"]["fields"]["20240115/24/2t"]["latest"]
    (tmp_path / current_field["metadata_file"]).write_bytes(b"tampered metadata")
    metadata_resume_session = FakeSession()
    metadata_resume_time = FakeTime()
    metadata_resumed = pilot.collect(raw_dir=tmp_path, session=metadata_resume_session,
                                     retrieved_at=at, sleeper=metadata_resume_time.sleep,
                                     clock=metadata_resume_time.clock)
    assert metadata_resumed["complete"] is True
    assert sum("20240115120000-24h-oper-fc.grib2" in url
               and "Range" in (kwargs.get("headers") or {})
               for url, kwargs in metadata_resume_session.calls) == 1


def test_checkpoint_resume_sets_actual_checkpoint_filename_and_rejects_hash_tampering(tmp_path):
    at = dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc)
    timebox = FakeTime()
    report = pilot.collect(raw_dir=tmp_path, session=FakeSession(), retrieved_at=at,
                           sleeper=timebox.sleep, clock=timebox.clock)
    state = pilot._load_checkpoint(tmp_path, at)
    assert state["latest_checkpoint"] == report["checkpoint_file"]
    checkpoint_path = tmp_path / report["checkpoint_file"]
    checkpoint_path.write_bytes(checkpoint_path.read_bytes() + b" ")
    try:
        pilot._load_checkpoint(tmp_path, at)
    except pilot.PilotError as exc:
        assert "hash mismatch" in str(exc)
    else:
        raise AssertionError("tampered checkpoint should fail closed")
