import datetime as dt
import hashlib
import json

import pytest

from agri_data import ecmwf_field_collection as collection


class FakeResponse:
    def __init__(self, status, headers, body):
        self.status_code = status
        self.headers = headers
        self.body = body

    def iter_content(self, chunk_size):
        for start in range(0, len(self.body), 11):
            yield self.body[start : start + 11]

    def close(self):
        pass


class FakeSession:
    def __init__(self, fail_index=None, offset_2t=0, last_modified_modes=None):
        self.calls = []
        self.fail_index = fail_index
        self.offset_2t = offset_2t
        self.last_modified_modes = last_modified_modes or {}

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
        filename = url.rsplit("/", 1)[-1]
        run_date = filename[:8]
        step = int(filename.split("-")[1].removesuffix("h"))
        if url.endswith(".index"):
            if self.fail_index == (run_date, step):
                return FakeResponse(404, {"Content-Length": "9"}, b"not found")
            rows = []
            for param, offset in (("2t", self.offset_2t), ("tp", 64)):
                rows.append({"date": run_date, "time": "1200", "step": step,
                             "type": "fc", "stream": "oper", "levtype": "sfc",
                             "param": param, "_offset": offset, "_length": 20})
            body = "\n".join(json.dumps(row) for row in rows).encode()
            return FakeResponse(200, {"Content-Length": str(len(body))}, body)
        offset, end = map(int, kwargs["headers"]["Range"][6:].split("-"))
        body = self._grib()
        parsed = dt.datetime.strptime(run_date, "%Y%m%d").date()
        headers = {"Content-Range": f"bytes {offset}-{end}/{max(100, end + 1)}",
                   "Content-Length": str(len(body))}
        mode = self.last_modified_modes.get(offset, "before")
        if mode != "missing":
            modified_time = dt.time(2, 0) if mode == "late" else dt.time(0, 30)
            modified = dt.datetime.combine(parsed + dt.timedelta(days=1), modified_time,
                                           tzinfo=dt.timezone.utc)
            headers["Last-Modified"] = modified.strftime("%a, %d %b %Y %H:%M:%S GMT")
        return FakeResponse(206, headers, body)


class FakeTime:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class SimulatedInterruption(Exception):
    pass


def test_period_is_exact_and_dry_run_sends_nothing(capsys):
    dates = collection.expected_run_dates()
    assert len(dates) == 922
    assert dates[0] == "20240315"
    assert dates[-1] == "20260922"
    assert collection.main(["--max-dates", "1"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["mode"] == "dry_run"
    assert report["expected_days"] == 922
    assert report["selected_days_this_invocation"] == 1
    assert report["logical_requests_max_this_invocation"] == 9
    assert report["request_sent"] is False


def test_response_reader_caps_and_closes_oversized_body():
    class OversizeResponse:
        closed = False

        def iter_content(self, chunk_size):
            yield b"x" * (collection.MAX_RESPONSE_BYTES + 1)

        def close(self):
            self.closed = True

    response = OversizeResponse()
    body, error = collection.pilot._read_bounded(response)
    assert len(body) == collection.MAX_RESPONSE_BYTES
    assert error == "response_exceeded_10mb_limit"
    assert response.closed is True


def test_one_day_capture_uses_only_indexes_and_single_ranges_and_resumes(tmp_path):
    session = FakeSession()
    timing = FakeTime()
    at = dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc)
    report = collection.collect(raw_dir=tmp_path, session=session, retrieved_at=at,
                                sleeper=timing.sleep, clock=timing.clock, max_dates=1)
    assert report["selected_complete"] is True
    assert report["complete"] is False  # full frozen period still has uncollected dates
    assert report["selected_run_dates"] == ["20240315"]
    assert report["complete_fields"] == 6
    assert report["expected_fields"] == 922 * 6
    assert report["complete_days"] == 1
    assert report["coverage_fraction_fields"] == 6 / (922 * 6)
    assert len(session.calls) == 9
    assert len(timing.sleeps) == 8
    assert all(delay >= 1 for delay in timing.sleeps)
    assert all(call[1]["verify"] is True and call[1]["allow_redirects"] is False
               and call[1]["timeout"] == collection.TIMEOUT_SECONDS for call in session.calls)
    assert all("Range" not in (kwargs.get("headers") or {}) for url, kwargs in session.calls if url.endswith(".index"))
    ranges = [kwargs["headers"]["Range"] for url, kwargs in session.calls if "Range" in (kwargs.get("headers") or {})]
    assert sorted(ranges) == ["bytes=0-19"] * 3 + ["bytes=64-83"] * 3
    assert all("/20240315/12z/ifs/0p25/oper/" in url for url, _ in session.calls)

    second = FakeSession()
    resumed = collection.collect(raw_dir=tmp_path, session=second, retrieved_at=at,
                                 sleeper=FakeTime().sleep, clock=FakeTime().clock, max_dates=1)
    assert resumed["selected_complete"] is True
    assert second.calls == []


def test_incomplete_day_is_not_retried_without_explicit_flag(tmp_path):
    at = dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc)
    first_session = FakeSession(fail_index=("20240315", 24))
    first_time = FakeTime()
    first = collection.collect(raw_dir=tmp_path, session=first_session, retrieved_at=at,
                               sleeper=first_time.sleep, clock=first_time.clock, max_dates=1)
    assert first["selected_complete"] is False
    assert first["failures"]["incomplete_days"] == 1
    assert len(first_session.calls) == 7  # failed index, two valid indexes and four ranges

    no_retry = FakeSession()
    skipped = collection.collect(raw_dir=tmp_path, session=no_retry, retrieved_at=at,
                                 sleeper=FakeTime().sleep, clock=FakeTime().clock, max_dates=1)
    assert no_retry.calls == []
    assert skipped["skipped_incomplete_run_dates"] == ["20240315"]

    retry = FakeSession()
    completed = collection.collect(raw_dir=tmp_path, session=retry, retrieved_at=at,
                                   sleeper=FakeTime().sleep, clock=FakeTime().clock,
                                   max_dates=1, retry_incomplete=True)
    assert completed["selected_complete"] is True
    assert len(retry.calls) == 3  # only the failed step's index and two fields
    assert all("/20240315/12z/ifs/0p25/oper/" in url for url, _ in retry.calls)
    index_files = list(tmp_path.glob("ecmwf_indices_20240315_24_*.index"))
    assert len(index_files) == 2
    assert b"not found" in [path.read_bytes() for path in index_files]
    final_path = sorted(tmp_path.glob("day_20240315_checkpoint_*.json"))[-1]
    final_checkpoint = json.loads(final_path.read_text())
    failed_attempts = final_checkpoint["captures"]["indices"]["20240315/24"]["attempts"]
    assert [attempt["state"] for attempt in failed_attempts] == ["unavailable", "complete"]
    assert (tmp_path / failed_attempts[0]["raw_file"]).read_bytes() == b"not found"


def test_interrupted_day_is_left_untouched_until_explicit_retry(tmp_path):
    class InterruptAfterFirstRequest(FakeSession):
        def get(self, url, **kwargs):
            if self.calls:
                raise SimulatedInterruption("simulated process interruption")
            return super().get(url, **kwargs)

    at = dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc)
    interrupted = InterruptAfterFirstRequest()
    try:
        collection.collect(raw_dir=tmp_path, session=interrupted, retrieved_at=at,
                           sleeper=FakeTime().sleep, clock=FakeTime().clock, max_dates=1)
    except SimulatedInterruption:
        pass
    else:
        raise AssertionError("interruption fixture should stop after one checkpointed index")
    assert len(interrupted.calls) == 1
    checkpoint = json.loads(sorted(tmp_path.glob("day_20240315_checkpoint_*.json"))[-1].read_text())
    assert checkpoint["captures"]["indices"]["20240315/24"]["complete"] is True
    assert checkpoint["day_complete"] is False

    skipped_session = FakeSession()
    skipped = collection.collect(raw_dir=tmp_path, session=skipped_session, retrieved_at=at,
                                 sleeper=FakeTime().sleep, clock=FakeTime().clock, max_dates=1)
    assert skipped_session.calls == []
    assert skipped["skipped_incomplete_run_dates"] == ["20240315"]

    retry_session = FakeSession()
    resumed = collection.collect(raw_dir=tmp_path, session=retry_session, retrieved_at=at,
                                 sleeper=FakeTime().sleep, clock=FakeTime().clock,
                                 max_dates=1, retry_incomplete=True)
    assert resumed["selected_complete"] is True
    assert len(retry_session.calls) == 8  # reuse the first valid index; fetch all remaining records


def test_three_day_tranche_resume_skips_completed_days_and_advances(tmp_path):
    at = dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc)
    first_session = FakeSession()
    first = collection.collect(raw_dir=tmp_path, session=first_session, retrieved_at=at,
                               sleeper=FakeTime().sleep, clock=FakeTime().clock, max_dates=3)
    assert first["selected_complete"] is True
    assert first["complete_days"] == 3
    assert len(first_session.calls) == 27

    class StopOnFirstNewDay(FakeSession):
        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            assert "/20240318/" in url
            raise SimulatedInterruption("confirmed resume advanced to first uncaptured date")

    resumed_session = StopOnFirstNewDay()
    try:
        collection.collect(raw_dir=tmp_path, session=resumed_session, retrieved_at=at,
                           sleeper=FakeTime().sleep, clock=FakeTime().clock)
    except SimulatedInterruption:
        pass
    else:
        raise AssertionError("resume should proceed to the first date beyond the completed tranche")
    assert len(resumed_session.calls) == 1
    assert "/20240318/" in resumed_session.calls[0][0]


@pytest.mark.parametrize("artifact_key", ["metadata_file", "raw_file"])
def test_corrupted_completed_artifact_marks_day_incomplete_then_explicitly_retries(tmp_path, artifact_key):
    at = dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc)
    initial = collection.collect(raw_dir=tmp_path, session=FakeSession(), retrieved_at=at,
                                 sleeper=FakeTime().sleep, clock=FakeTime().clock, max_dates=1)
    checkpoint_path = sorted(tmp_path.glob("day_20240315_checkpoint_*.json"))[-1]
    checkpoint = json.loads(checkpoint_path.read_text())
    field = checkpoint["captures"]["fields"]["20240315/24/2t"]["latest"]
    (tmp_path / field[artifact_key]).write_bytes(b"corrupt")

    no_retry_session = FakeSession()
    no_retry = collection.collect(raw_dir=tmp_path, session=no_retry_session, retrieved_at=at,
                                 sleeper=FakeTime().sleep, clock=FakeTime().clock, max_dates=1)
    assert no_retry_session.calls == []
    assert no_retry["complete_fields"] == 5
    assert no_retry["selected_complete"] is False
    repaired_checkpoint = json.loads(
        (tmp_path / no_retry["day_status"][0]["latest_checkpoint"]).read_text()
    )
    assert repaired_checkpoint["day_complete"] is False
    assert repaired_checkpoint["captures"]["fields"]["20240315/24/2t"]["complete"] is False

    retry_session = FakeSession()
    repaired = collection.collect(raw_dir=tmp_path, session=retry_session, retrieved_at=at,
                                  sleeper=FakeTime().sleep, clock=FakeTime().clock,
                                  max_dates=1, retry_incomplete=True)
    assert repaired["selected_complete"] is True
    assert len(retry_session.calls) == 1
    assert "Range" in retry_session.calls[0][1]["headers"]


def test_changed_index_invalidates_fields_tied_to_old_index_hash(tmp_path):
    at = dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc)
    collection.collect(raw_dir=tmp_path, session=FakeSession(), retrieved_at=at,
                       sleeper=FakeTime().sleep, clock=FakeTime().clock, max_dates=1)
    checkpoint = json.loads(sorted(tmp_path.glob("day_20240315_checkpoint_*.json"))[-1].read_text())
    index = checkpoint["captures"]["indices"]["20240315/24"]["latest"]
    (tmp_path / index["raw_file"]).write_bytes(b"damaged index")

    changed_session = FakeSession(offset_2t=128)
    report = collection.collect(raw_dir=tmp_path, session=changed_session, retrieved_at=at,
                                sleeper=FakeTime().sleep, clock=FakeTime().clock,
                                max_dates=1, retry_incomplete=True)
    assert report["selected_complete"] is True
    step_calls = [(url, kwargs) for url, kwargs in changed_session.calls
                  if "/20240315/12z/ifs/0p25/oper/" in url]
    assert len(step_calls) == 3
    ranges = [kwargs["headers"]["Range"] for _, kwargs in step_calls
              if "Range" in (kwargs.get("headers") or {})]
    assert sorted(ranges) == ["bytes=128-147", "bytes=64-83"]


def test_resume_revalidates_field_index_hash_after_interruption_between_checkpoints(tmp_path):
    at = dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc)
    collection.collect(raw_dir=tmp_path, session=FakeSession(), retrieved_at=at,
                       sleeper=FakeTime().sleep, clock=FakeTime().clock, max_dates=1)
    old_checkpoint = json.loads(sorted(tmp_path.glob("day_20240315_checkpoint_*.json"))[-1].read_text())
    old_index = old_checkpoint["captures"]["indices"]["20240315/24"]["latest"]
    (tmp_path / old_index["raw_file"]).write_bytes(b"damaged index")

    class InterruptAfterUpdatedIndex(FakeSession):
        def get(self, url, **kwargs):
            if url.endswith(".grib2"):
                raise SimulatedInterruption("stop after updated index checkpoint")
            return super().get(url, **kwargs)

    try:
        collection.collect(raw_dir=tmp_path, session=InterruptAfterUpdatedIndex(offset_2t=128),
                           retrieved_at=at, sleeper=FakeTime().sleep, clock=FakeTime().clock,
                           max_dates=1, retry_incomplete=True)
    except SimulatedInterruption:
        pass
    else:
        raise AssertionError("fixture must interrupt between index and field checkpoints")
    interrupted_checkpoint = json.loads(
        sorted(tmp_path.glob("day_20240315_checkpoint_*.json"))[-1].read_text()
    )
    assert interrupted_checkpoint["captures"]["indices"]["20240315/24"]["complete"] is True
    assert interrupted_checkpoint["captures"]["fields"]["20240315/24/2t"]["complete"] is False
    assert interrupted_checkpoint["day_complete"] is False

    no_retry_session = FakeSession()
    incomplete = collection.collect(raw_dir=tmp_path, session=no_retry_session, retrieved_at=at,
                                    sleeper=FakeTime().sleep, clock=FakeTime().clock, max_dates=1)
    assert no_retry_session.calls == []
    assert incomplete["selected_complete"] is False
    assert incomplete["complete_fields"] == 4
    assert incomplete["skipped_incomplete_run_dates"] == ["20240315"]

    repair_session = FakeSession(offset_2t=128)
    repaired = collection.collect(raw_dir=tmp_path, session=repair_session, retrieved_at=at,
                                  sleeper=FakeTime().sleep, clock=FakeTime().clock,
                                  max_dates=1, retry_incomplete=True)
    assert repaired["selected_complete"] is True
    assert len(repair_session.calls) == 2
    assert sorted(kwargs["headers"]["Range"] for _, kwargs in repair_session.calls) == [
        "bytes=128-147", "bytes=64-83"]


def test_last_modified_timing_aggregates_do_not_change_structural_completeness(tmp_path):
    session = FakeSession(last_modified_modes={0: "late", 64: "missing"})
    report = collection.collect(raw_dir=tmp_path, session=session,
                                retrieved_at=dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc),
                                sleeper=FakeTime().sleep, clock=FakeTime().clock, max_dates=1)
    assert report["selected_complete"] is True
    assert report["structurally_complete_fields"] == 6
    assert report["last_modified_before_origin_fields"] == 0
    assert report["last_modified_after_origin_fields"] == 3
    assert report["last_modified_missing_or_invalid_fields"] == 3
    assert report["last_modified_assessed_fields"] == 3
    assert report["last_modified_before_origin_ratio"] == 0


def test_source_scope_mismatch_fails_closed(tmp_path):
    at = dt.datetime(2026, 10, 7, tzinfo=dt.timezone.utc)
    collection.collect(raw_dir=tmp_path, session=FakeSession(), retrieved_at=at,
                       sleeper=FakeTime().sleep, clock=FakeTime().clock, max_dates=1)
    path = sorted(tmp_path.glob("day_20240315_checkpoint_*.json"))[-1]
    state = json.loads(path.read_text())
    state["source_scope"]["model_path"] = "other"
    state["checkpoint_sequence"] += 1
    payload = (json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()[:12]
    path = tmp_path / f"day_20240315_checkpoint_{state['checkpoint_sequence']:04d}_{digest}.json"
    path.write_bytes(payload)
    try:
        collection._load_day(tmp_path, "20240315", at)
    except collection.CollectionError as exc:
        assert "source scope mismatch" in str(exc)
    else:
        raise AssertionError("edited checkpoint should fail closed")
