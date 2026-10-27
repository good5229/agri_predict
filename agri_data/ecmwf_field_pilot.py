"""Bounded ECMWF Google-mirror field pilot for three frozen origin dates.

Dry-run is the default. Execution retrieves only the fixed index pages and
separate single byte ranges for 2t/tp at steps 24/72/168; it never collects
full-period data or trains a model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://storage.googleapis.com/ecmwf-open-data"
RUN_DATES = ("20240115", "20250914", "20260922")
MARCH_DIAGNOSTIC_RUN_DATE = "20240315"
MARCH_DIAGNOSTIC_RUN_DATES = (MARCH_DIAGNOSTIC_RUN_DATE,)
RUN_HOUR = "1200"
STEPS = (24, 72, 168)
PARAMS = ("2t", "tp")
MAX_RESPONSE_BYTES = 10_000_000
TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.0
MIN_REQUEST_INTERVAL_SECONDS = 1.0
CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")


class PilotError(RuntimeError):
    pass


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise
        finally:
            if tmp.exists():
                tmp.unlink()
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def _safe_headers(headers: Any) -> dict[str, str]:
    allowed = {"content-type", "content-length", "content-range", "last-modified", "etag", "date"}
    return {str(k).lower(): str(v) for k, v in headers.items() if str(k).lower() in allowed}


def _run_date_value(run_date: str) -> dt.date:
    return dt.datetime.strptime(run_date, "%Y%m%d").date()


def _run_initialization(run_date: str) -> tuple[str, str]:
    date = _run_date_value(run_date)
    return date.strftime("%Y%m%d"), f"{date.isoformat()}T12:00Z"


def _deadline_utc(run_date: str) -> dt.datetime:
    origin_next_day = _run_date_value(run_date) + dt.timedelta(days=1)
    return dt.datetime.combine(origin_next_day, dt.time(1), tzinfo=dt.timezone.utc)


def _urls(run_date: str, step: int) -> tuple[str, str, str, str]:
    run_date, run_init = _run_initialization(run_date)
    stem = f"{run_date}{RUN_HOUR}00-{step}h-oper-fc"
    initialization = dt.datetime.combine(
        _run_date_value(run_date), dt.time(12), tzinfo=dt.timezone.utc
    )
    # ECMWF's Google mirror added the /ifs/ path component on 2024-02-28 06Z.
    model_directory = "ifs/" if initialization >= dt.datetime(2024, 2, 28, 6, tzinfo=dt.timezone.utc) else ""
    key = f"{run_date}/12z/{model_directory}0p25/oper/{stem}"
    return run_date, run_init, f"{BASE_URL}/{key}.index", f"{BASE_URL}/{key}.grib2"


class RequestGate:
    """Enforce a one-second gap between starts of all HTTP attempts."""

    def __init__(self, sleeper: Any = time.sleep, clock: Any = time.monotonic):
        self.sleeper = sleeper
        self.clock = clock
        self.last_start: float | None = None

    def wait(self) -> None:
        now = self.clock()
        if self.last_start is not None:
            delay = MIN_REQUEST_INTERVAL_SECONDS - (now - self.last_start)
            if delay > 0:
                self.sleeper(delay)
                now = self.clock()
        self.last_start = now


def _request(session: Any, url: str, *, params: dict[str, str] | None,
             headers: dict[str, str] | None, gate: RequestGate,
             sleeper: Any) -> tuple[Any | None, int, str | None]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        gate.wait()
        try:
            response = session.get(url, params=params, headers=headers, timeout=TIMEOUT_SECONDS,
                                   verify=True, allow_redirects=False, stream=True)
            return response, attempt, None
        except requests.RequestException:
            if attempt == MAX_ATTEMPTS:
                return None, attempt, "transport_failed_after_bounded_retries"
            sleeper(RETRY_DELAY_SECONDS * attempt)
    return None, MAX_ATTEMPTS, "transport_failed_after_bounded_retries"


def _read_bounded(response: Any) -> tuple[bytes, str | None]:
    body = bytearray()
    error = None
    try:
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            remaining = MAX_RESPONSE_BYTES - len(body)
            body.extend(chunk[:remaining])
            if len(chunk) > remaining:
                error = "response_exceeded_10mb_limit"
                break
    except (requests.RequestException, OSError):
        error = "response_stream_failed"
    finally:
        response.close()
    return bytes(body), error


def _parse_index(body: bytes, run_date: str, step: int) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    parsed: dict[str, list[dict[str, Any]]] = {param: [] for param in PARAMS}
    errors: dict[str, list[str]] = {param: [] for param in PARAMS}
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return {}, {param: ["index_not_utf8"] for param in PARAMS}
    expected = {"date": run_date, "time": RUN_HOUR, "step": str(step),
                "type": "fc", "stream": "oper", "levtype": "sfc"}
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            for param in PARAMS:
                errors[param].append(f"invalid_json_line_{line_no}")
            continue
        if not isinstance(record, dict):
            for param in PARAMS:
                errors[param].append(f"non_object_line_{line_no}")
            continue
        param = record.get("param")
        if param in PARAMS and all(str(record.get(key)) == value for key, value in expected.items()):
            parsed[param].append(record)
    selected = {}
    for param in PARAMS:
        if errors[param]:
            continue
        if len(parsed[param]) != 1:
            errors[param].append("matching_index_record_not_unique")
            continue
        record = parsed[param][0]
        offset, length = record.get("_offset"), record.get("_length")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            errors[param].append("index_offset_invalid")
        elif isinstance(length, bool) or not isinstance(length, int) or length <= 0:
            errors[param].append("index_length_invalid")
        elif offset + length > 2**63 - 1:
            errors[param].append("index_range_overflow")
        else:
            selected[param] = record
    return selected, errors


def _parse_last_modified(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        parsed = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _validate_grib(body: bytes, expected_length: int) -> list[str]:
    issues = []
    if len(body) != expected_length:
        issues.append("field_length_mismatch")
    if len(body) < 20 or body[:4] != b"GRIB":
        issues.append("grib_magic_missing")
        return issues
    if body[7] != 2:
        issues.append("grib_edition_not_2")
    if int.from_bytes(body[8:16], "big") != len(body):
        issues.append("grib_declared_length_mismatch")
    if not body.endswith(b"7777"):
        issues.append("grib_trailer_missing")
    return issues


def _checkpoint_path(root: Path, sequence: int, payload: bytes) -> Path:
    return root / f"checkpoint_{sequence:06d}_{_sha256(payload)[:12]}.json"


def _write_checkpoint(state: dict[str, Any], root: Path) -> None:
    state["checkpoint_sequence"] = int(state.get("checkpoint_sequence", 0)) + 1
    payload = (json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    path = _checkpoint_path(root, state["checkpoint_sequence"], payload)
    _atomic_immutable(path, payload)
    state["latest_checkpoint"] = path.name


def _load_checkpoint(root: Path, retrieved: dt.datetime,
                     run_dates: tuple[str, ...] = RUN_DATES) -> dict[str, Any]:
    paths = sorted(root.glob("checkpoint_*.json"))
    if paths:
        try:
            payload = paths[-1].read_bytes()
            state = json.loads(payload)
        except (OSError, json.JSONDecodeError) as exc:
            raise PilotError("existing checkpoint is unreadable; refusing to overwrite or restart") from exc
        if not paths[-1].name.endswith(f"_{_sha256(payload)[:12]}.json"):
            raise PilotError("checkpoint hash mismatch; refusing to resume")
        if state.get("schema_version") != 1 or not isinstance(state.get("captures"), dict):
            raise PilotError("existing checkpoint has an unsupported format")
        stored_run_dates = tuple(state.get("run_dates", RUN_DATES))
        if stored_run_dates != run_dates:
            raise PilotError("checkpoint belongs to a different frozen run-date scope")
        state["latest_checkpoint"] = paths[-1].name
        for collection in ("indices", "fields"):
            group = state["captures"].get(collection)
            if not isinstance(group, dict):
                raise PilotError("checkpoint capture groups are malformed")
            for entry in group.values():
                if not isinstance(entry, dict):
                    raise PilotError("checkpoint capture entry is malformed")
                latest = entry.get("latest")
                if entry.get("complete") or (isinstance(latest, dict) and latest.get("complete")):
                    if not _valid_completed_attempt(root, latest):
                        if isinstance(latest, dict):
                            latest["complete"] = False
                            latest["state"] = "incomplete"
                            latest.setdefault("issues", []).append("completed_artifact_missing_or_hash_mismatch")
                        entry["complete"] = False
        return state
    return {"schema_version": 1, "run_dates": list(run_dates),
            "created_at_utc": retrieved.isoformat(),
            "updated_at_utc": retrieved.isoformat(), "checkpoint_sequence": 0,
            "captures": {"indices": {}, "fields": {}}}


def _valid_completed_attempt(root: Path, attempt: Any) -> bool:
    if not isinstance(attempt, dict) or not attempt.get("complete"):
        return False
    metadata_name, raw_name = attempt.get("metadata_file"), attempt.get("raw_file")
    metadata_hash, raw_hash = attempt.get("metadata_sha256"), attempt.get("sha256")
    if not all(isinstance(value, str) and Path(value).name == value
               for value in (metadata_name, raw_name, metadata_hash, raw_hash)):
        return False
    try:
        metadata_bytes = (root / metadata_name).read_bytes()
        raw_bytes = (root / raw_name).read_bytes()
        metadata = json.loads(metadata_bytes)
    except (OSError, json.JSONDecodeError):
        return False
    return (_sha256(metadata_bytes) == metadata_hash
            and _sha256(raw_bytes) == raw_hash
            and metadata.get("complete") is True
            and metadata.get("sha256") == raw_hash)


def _store_attempt(root: Path, state: dict[str, Any], collection: str, key: str,
                   metadata: dict[str, Any], body: bytes | None, extension: str) -> dict[str, Any]:
    group = state["captures"].setdefault(collection, {})
    entry = group.setdefault(key, {"complete": False, "attempts": []})
    attempt_no = len(entry["attempts"]) + 1
    stamp = metadata["captured_at_utc"].replace(":", "").replace("-", "").replace("+00:00", "Z")
    digest = _sha256(body) if body is not None else None
    suffix = digest[:12] if digest else "noresponse"
    unique = uuid.uuid4().hex[:8]
    stem = f"ecmwf_pilot_{key.replace('/', '_')}_a{attempt_no:02d}_{stamp}_{suffix}_{unique}"
    if body is not None:
        raw_path = root / f"{stem}.{extension}"
        _atomic_immutable(raw_path, body)
        metadata["raw_file"] = raw_path.name
        metadata["sha256"] = digest
    metadata_path = root / f"{stem}.meta.json"
    encoded = (json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    _atomic_immutable(metadata_path, encoded)
    attempt = {"metadata_file": metadata_path.name,
               "metadata_sha256": _sha256(encoded),
               "raw_file": metadata.get("raw_file"), "sha256": digest,
               "state": metadata["state"], "complete": bool(metadata.get("complete")),
               "issues": list(metadata.get("issues", [])),
               "selected_records": metadata.get("selected_records"),
               "last_modified_utc": metadata.get("last_modified_utc"),
               "last_modified_before_origin": metadata.get("last_modified_before_origin"),
               "offset": metadata.get("offset"), "length": metadata.get("length"),
               "content_range_total": metadata.get("content_range_total")}
    entry["attempts"].append(attempt)
    entry["latest"] = attempt
    entry["complete"] = attempt["complete"]
    state["updated_at_utc"] = metadata["captured_at_utc"]
    _write_checkpoint(state, root)
    return attempt


def _capture_index(run_date: str, step: int, *, root: Path, state: dict[str, Any],
                   session: Any, gate: RequestGate, sleeper: Any, captured_at: dt.datetime) -> dict[str, Any]:
    run_date, run_init, index_url, grib_url = _urls(run_date, step)
    key = f"{run_date}/{step}"
    existing = state["captures"]["indices"].get(key)
    if existing and existing.get("complete"):
        return existing["latest"]
    response, attempts, transport_error = _request(session, index_url, params=None, headers=None,
                                                   gate=gate, sleeper=sleeper)
    base = {"kind": "index", "run_date": run_date, "run_initialization_utc": run_init,
            "step_hours": step, "source": BASE_URL, "request_url": index_url,
            "grib_url": grib_url, "captured_at_utc": captured_at.isoformat(),
            "attempt_count": attempts, "http_status": None, "headers": {},
            "response_bytes": 0, "transfer_error": None, "state": "error",
            "complete": False, "issues": [], "selected_records": {}}
    body = None
    if response is None:
        base["issues"].append(transport_error)
        return _store_attempt(root, state, "indices", key, base, body, "index")
    body, transfer_error = _read_bounded(response)
    status = int(response.status_code)
    headers = _safe_headers(response.headers)
    base.update(http_status=status, headers=headers, response_bytes=len(body),
                transfer_error=transfer_error)
    if transfer_error:
        base["state"] = "partial"
        base["issues"].append(transfer_error)
        return _store_attempt(root, state, "indices", key, base, body, "index")
    if status == 404:
        base["state"] = "unavailable"
        base["issues"].append("index_http_404")
        return _store_attempt(root, state, "indices", key, base, body, "index")
    if not 200 <= status < 300:
        base["state"] = "error"
        base["issues"].append("index_http_status_not_success")
        return _store_attempt(root, state, "indices", key, base, body, "index")
    content_length = headers.get("content-length")
    if content_length is not None:
        try:
            content_ok = int(content_length) == len(body)
        except ValueError:
            content_ok = False
        if not content_ok:
            base["state"] = "partial"
            base["issues"].append("index_content_length_mismatch")
            return _store_attempt(root, state, "indices", key, base, body, "index")
    records, errors = _parse_index(body, run_date, step)
    base["selected_records"] = records
    base["field_index_issues"] = errors
    base["complete"] = all(param in records and not errors[param] for param in PARAMS)
    base["state"] = "complete" if base["complete"] else "incomplete"
    base["issues"].extend(issue for param in PARAMS for issue in errors[param])
    return _store_attempt(root, state, "indices", key, base, body, "index")


def _capture_field(run_date: str, step: int, param: str, record: dict[str, Any], *,
                   index_meta: dict[str, Any], root: Path, state: dict[str, Any],
                   session: Any, gate: RequestGate, sleeper: Any,
                   captured_at: dt.datetime) -> dict[str, Any]:
    run_date, run_init, _, grib_url = _urls(run_date, step)
    key = f"{run_date}/{step}/{param}"
    existing = state["captures"]["fields"].get(key)
    if existing and existing.get("complete"):
        return existing["latest"]
    offset, length = record["_offset"], record["_length"]
    end = offset + length - 1
    response, attempts, transport_error = _request(
        session, grib_url, params=None, headers={"Range": f"bytes={offset}-{end}"},
        gate=gate, sleeper=sleeper)
    base = {"kind": "field", "run_date": run_date, "run_initialization_utc": run_init,
            "step_hours": step, "parameter": param, "source": BASE_URL,
            "index_sha256": index_meta.get("sha256"), "request_url": grib_url,
            "requested_offset": offset, "requested_length": length,
            "requested_end": end, "captured_at_utc": captured_at.isoformat(),
            "attempt_count": attempts, "http_status": None, "headers": {},
            "response_bytes": 0, "transfer_error": None, "state": "error",
            "complete": False, "issues": [], "last_modified_raw": None,
            "last_modified_utc": None, "last_modified_before_origin": None,
            "last_modified_proves_first_publication": False}
    body = None
    if response is None:
        base["issues"].append(transport_error)
        return _store_attempt(root, state, "fields", key, base, body, "grib2")
    body, transfer_error = _read_bounded(response)
    status = int(response.status_code)
    headers = _safe_headers(response.headers)
    base.update(http_status=status, headers=headers, response_bytes=len(body),
                transfer_error=transfer_error)
    if transfer_error:
        base["state"] = "partial"
        base["issues"].append(transfer_error)
    elif status != 206:
        base["state"] = "unavailable" if status == 404 else "error"
        base["issues"].append("range_http_404" if status == 404 else "range_http_status_not_206")
    else:
        content_range = headers.get("content-range", "")
        match = CONTENT_RANGE_RE.fullmatch(content_range)
        if not match:
            base["issues"].append("content_range_missing_or_invalid")
        else:
            range_start, range_end, total = map(int, match.groups())
            base["content_range"] = content_range
            base["content_range_total"] = total
            if range_start != offset or range_end != end:
                base["issues"].append("content_range_bounds_mismatch")
            if end >= total:
                base["issues"].append("requested_range_outside_object")
        if len(body) != length:
            base["issues"].append("field_length_mismatch")
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) != len(body):
                    base["issues"].append("range_content_length_mismatch")
            except ValueError:
                base["issues"].append("range_content_length_invalid")
        base["issues"].extend(_validate_grib(body, length))
        raw_last_modified = headers.get("last-modified")
        parsed_last_modified = _parse_last_modified(raw_last_modified)
        base["last_modified_raw"] = raw_last_modified
        base["last_modified_utc"] = parsed_last_modified.isoformat() if parsed_last_modified else None
        deadline = _deadline_utc(run_date)
        base["origin_deadline_utc"] = deadline.isoformat()
        if parsed_last_modified is None:
            base["issues"].append("last_modified_missing_or_invalid")
        else:
            base["last_modified_before_origin"] = parsed_last_modified < deadline
        base["complete"] = not base["issues"]
        base["state"] = "complete" if base["complete"] else "incomplete"
    return _store_attempt(root, state, "fields", key, base, body, "grib2")


def _report(state: dict[str, Any], run_dates: tuple[str, ...] = RUN_DATES) -> dict[str, Any]:
    rows = []
    for run_date in run_dates:
        _, run_init = _run_initialization(run_date)
        for step in STEPS:
            index_key = f"{run_date}/{step}"
            index_entry = state["captures"]["indices"].get(index_key)
            index_latest = index_entry.get("latest") if index_entry else None
            for param in PARAMS:
                field_key = f"{run_date}/{step}/{param}"
                field_entry = state["captures"]["fields"].get(field_key)
                field_latest = field_entry.get("latest") if field_entry else None
                index_param_issue = None
                if index_latest and not index_latest.get("complete"):
                    metadata_path = state.get("root", "")
                    # The detailed per-parameter reason remains in the immutable index metadata.
                    index_param_issue = "index_incomplete_or_field_record_invalid"
                if not index_latest:
                    field_state = "incomplete"
                    field_complete = False
                    field_issue = "index_not_captured"
                elif param not in (index_latest.get("selected_records") or {}):
                    field_state = index_latest.get("state", "incomplete")
                    field_complete = False
                    field_issue = index_param_issue or "matching_index_record_unavailable"
                elif field_latest:
                    field_state = field_latest["state"]
                    field_complete = bool(field_latest["complete"])
                    field_issue = field_latest["issues"]
                else:
                    field_state = "incomplete"
                    field_complete = False
                    field_issue = "field_not_captured"
                rows.append({"run_date": run_date,
                             "run_initialization_utc": run_init, "step_hours": step,
                             "parameter": param,
                             "index_complete": bool(index_entry and index_entry.get("complete")),
                             "index_latest": index_latest,
                             "field_state": field_state, "field_complete": field_complete,
                             "field_latest": field_latest, "field_issue": field_issue})
    logical_requests = len(run_dates) * len(STEPS) * (1 + len(PARAMS))
    return {"schema_version": 1, "source": BASE_URL,
            "status": "march_2024_diagnostic" if run_dates == MARCH_DIAGNOSTIC_RUN_DATES else "pilot_only",
            "complete": len(rows) == len(run_dates) * len(STEPS) * len(PARAMS)
            and all(row["field_complete"] for row in rows)
            and all(state["captures"]["indices"].get(f"{run_date}/{step}", {}).get("complete", False)
                    for run_date in run_dates for step in STEPS),
            "run_dates": list(run_dates), "steps_hours": list(STEPS),
            "parameters": list(PARAMS), "logical_requests_max": logical_requests,
            "max_http_attempts_including_transport_retries": logical_requests * MAX_ATTEMPTS,
            "min_seconds_between_http_attempt_starts": MIN_REQUEST_INTERVAL_SECONDS,
            "claim_limit": ("one frozen March 2024 run only; no continuous coverage, first-publication, or model-performance claim"
                            if run_dates == MARCH_DIAGNOSTIC_RUN_DATES else
                            "three frozen run dates only; no continuous coverage, first-publication, or model-performance claim"),
            "fields": rows,
            "indices_complete": all(
                state["captures"]["indices"].get(f"{run_date}/{step}", {}).get("complete", False)
                for run_date in run_dates for step in STEPS
            ),
            "checkpoint_sequence": state["checkpoint_sequence"]}


def collect(*, raw_dir: str | Path | None = None, session: Any = requests,
            retrieved_at: dt.datetime | None = None, sleeper: Any = time.sleep,
            clock: Any = time.monotonic,
            run_dates: tuple[str, ...] = RUN_DATES) -> dict[str, Any]:
    if run_dates not in (RUN_DATES, MARCH_DIAGNOSTIC_RUN_DATES):
        raise ValueError("run_dates must select the frozen pilot or March 2024 diagnostic")
    if raw_dir is None:
        raw_dir = ("data/raw/ecmwf_march_2024_diagnostic"
                   if run_dates == MARCH_DIAGNOSTIC_RUN_DATES
                   else "data/raw/ecmwf_field_pilot")
    root = Path(raw_dir)
    root.mkdir(parents=True, exist_ok=True)
    captured = (retrieved_at or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    state = _load_checkpoint(root, captured, run_dates)
    state["root"] = str(root)
    state["run_dates"] = list(run_dates)
    gate = RequestGate(sleeper=sleeper, clock=clock)
    for run_date in run_dates:
        for step in STEPS:
            index_key = f"{run_date}/{step}"
            index_entry = state["captures"]["indices"].get(index_key)
            if index_entry and index_entry.get("complete"):
                index_latest = index_entry["latest"]
            else:
                index_latest = _capture_index(run_date, step, root=root, state=state,
                                              session=session, gate=gate, sleeper=sleeper,
                                              captured_at=captured)
            records = index_latest.get("selected_records") or {}
            for param in PARAMS:
                record = records.get(param)
                if record is None:
                    continue
                field_key = f"{run_date}/{step}/{param}"
                field_entry = state["captures"]["fields"].get(field_key)
                if field_entry and field_entry.get("complete"):
                    continue
                _capture_field(run_date, step, param, record, index_meta=index_latest,
                               root=root, state=state, session=session, gate=gate,
                               sleeper=sleeper, captured_at=captured)
    report = _report(state, run_dates)
    report["retrieved_at_utc"] = captured.isoformat()
    encoded = (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    summary_path = root / f"ecmwf_field_pilot_summary_{captured.strftime('%Y%m%dT%H%M%S%fZ')}_{_sha256(encoded)[:12]}.json"
    _atomic_immutable(summary_path, encoded)
    report["summary_report"] = summary_path.name
    report["summary_sha256"] = _sha256(encoded)
    report["checkpoint_file"] = state.get("latest_checkpoint")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=None,
                        help="raw/checkpoint directory (defaults to a mode-specific directory)")
    parser.add_argument("--march-2024-diagnostic", action="store_true",
                        help="select only the frozen 2024-03-15 12Z diagnostic")
    parser.add_argument("--execute", action="store_true", help="execute the selected bounded field pilot")
    args = parser.parse_args(argv)
    run_dates = MARCH_DIAGNOSTIC_RUN_DATES if args.march_2024_diagnostic else RUN_DATES
    logical_requests = len(run_dates) * len(STEPS) * (1 + len(PARAMS))
    raw_dir = args.raw_dir or ("data/raw/ecmwf_march_2024_diagnostic"
                               if args.march_2024_diagnostic else "data/raw/ecmwf_field_pilot")
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "diagnostic": args.march_2024_diagnostic,
                          "source": BASE_URL, "run_dates": list(run_dates), "steps_hours": list(STEPS),
                          "parameters": list(PARAMS), "logical_requests_max": logical_requests,
                          "max_http_attempts_including_transport_retries": logical_requests * MAX_ATTEMPTS,
                          "min_seconds_between_http_attempt_starts": MIN_REQUEST_INTERVAL_SECONDS,
                          "raw_dir": raw_dir,
                          "request_sent": False}, ensure_ascii=False))
        return 0
    try:
        report = collect(raw_dir=raw_dir, run_dates=run_dates)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["complete"] else 1
    except (OSError, ValueError, PilotError) as exc:
        print(f"field pilot failed: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
