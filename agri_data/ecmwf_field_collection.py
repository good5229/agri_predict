"""Restartable, bounded collector for the frozen 12Z ECMWF field period.

Dry-run is the default. Execution downloads only one index page per forecast
step and single byte ranges for the selected 2t/tp fields; it never downloads
full GRIB files or decodes fields.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Any

import requests

from . import ecmwf_field_pilot as pilot

BASE_URL = pilot.BASE_URL
START_DATE = dt.date(2024, 3, 15)
END_DATE = dt.date(2026, 9, 22)
RUN_HOUR = "1200"
STEPS = pilot.STEPS
PARAMS = pilot.PARAMS
MAX_RESPONSE_BYTES = pilot.MAX_RESPONSE_BYTES
TIMEOUT_SECONDS = pilot.TIMEOUT_SECONDS
MAX_ATTEMPTS = pilot.MAX_ATTEMPTS
RETRY_DELAY_SECONDS = pilot.RETRY_DELAY_SECONDS
MIN_REQUEST_INTERVAL_SECONDS = pilot.MIN_REQUEST_INTERVAL_SECONDS

SOURCE_SCOPE = {
    "source": BASE_URL,
    "model_path": "ifs/0p25/oper",
    "run_hour_utc": RUN_HOUR,
    "start_date_inclusive": START_DATE.isoformat(),
    "end_date_inclusive": END_DATE.isoformat(),
    "steps_hours": list(STEPS),
    "parameters": list(PARAMS),
    "layout_rule": "12z/ifs/0p25/oper",
    "min_seconds_between_http_attempt_starts": MIN_REQUEST_INTERVAL_SECONDS,
    "max_transport_attempts": MAX_ATTEMPTS,
    "timeout_seconds": TIMEOUT_SECONDS,
    "max_response_bytes": MAX_RESPONSE_BYTES,
    "protocol_version": 1,
}
SCOPE_HASH = hashlib.sha256(
    json.dumps(SOURCE_SCOPE, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()


class CollectionError(RuntimeError):
    pass


def expected_run_dates() -> list[str]:
    count = (END_DATE - START_DATE).days
    return [(START_DATE + dt.timedelta(days=offset)).strftime("%Y%m%d")
            for offset in range(count + 1)]


def _urls(run_date: str, step: int) -> tuple[str, str, str]:
    date, run_init, index_url, grib_url = pilot._urls(run_date, step)
    return date, run_init, index_url, grib_url


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _default_day_state(run_date: str) -> dict[str, Any]:
    return {"schema_version": 1, "source_scope": SOURCE_SCOPE, "source_scope_hash": SCOPE_HASH,
            "run_date": run_date, "created_at_utc": None, "updated_at_utc": None,
            "checkpoint_sequence": 0, "day_started": False, "day_complete": False,
            "captures": {"indices": {}, "fields": {}}}


def _checkpoint_name(run_date: str, sequence: int, payload: bytes) -> str:
    return f"day_{run_date}_checkpoint_{sequence:04d}_{hashlib.sha256(payload).hexdigest()[:12]}.json"


def _write_checkpoint(root: Path, state: dict[str, Any]) -> None:
    state["checkpoint_sequence"] = int(state.get("checkpoint_sequence", 0)) + 1
    payload = (json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    name = _checkpoint_name(state["run_date"], state["checkpoint_sequence"], payload)
    pilot._atomic_immutable(root / name, payload)
    state["latest_checkpoint"] = name


def _valid_completed_attempt(root: Path, attempt: Any) -> bool:
    if not isinstance(attempt, dict) or attempt.get("complete") is not True:
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
    return (hashlib.sha256(metadata_bytes).hexdigest() == metadata_hash
            and hashlib.sha256(raw_bytes).hexdigest() == raw_hash
            and metadata.get("complete") is True
            and metadata.get("sha256") == raw_hash
            and metadata.get("source_scope_hash") == SCOPE_HASH)


def _load_day(root: Path, run_date: str, retrieved_at: dt.datetime) -> tuple[dict[str, Any], bool]:
    paths = sorted(root.glob(f"day_{run_date}_checkpoint_*.json"))
    if not paths:
        state = _default_day_state(run_date)
        state["created_at_utc"] = retrieved_at.isoformat()
        state["updated_at_utc"] = retrieved_at.isoformat()
        return state, False
    path = paths[-1]
    try:
        payload = path.read_bytes()
        state = json.loads(payload)
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectionError(f"unreadable checkpoint for {run_date}; refusing to restart") from exc
    if not path.name.endswith(f"_{hashlib.sha256(payload).hexdigest()[:12]}.json"):
        raise CollectionError(f"checkpoint hash mismatch for {run_date}; refusing to resume")
    if (state.get("schema_version") != 1 or state.get("source_scope") != SOURCE_SCOPE
            or state.get("source_scope_hash") != SCOPE_HASH or state.get("run_date") != run_date
            or not isinstance(state.get("captures"), dict)):
        raise CollectionError(f"checkpoint source scope mismatch for {run_date}; refusing to resume")
    state["latest_checkpoint"] = path.name
    invalidated = False
    for collection in ("indices", "fields"):
        group = state["captures"].get(collection)
        if not isinstance(group, dict):
            raise CollectionError(f"malformed {collection} checkpoint group for {run_date}")
        for entry in group.values():
            if not isinstance(entry, dict):
                raise CollectionError(f"malformed checkpoint entry for {run_date}")
            latest = entry.get("latest")
            if entry.get("complete") or (isinstance(latest, dict) and latest.get("complete")):
                if not _valid_completed_attempt(root, latest):
                    if isinstance(latest, dict):
                        latest["complete"] = False
                        latest["state"] = "incomplete"
                        latest.setdefault("issues", []).append("completed_artifact_missing_or_hash_mismatch")
                    entry["complete"] = False
                    invalidated = True
    invalidated = _invalidate_fields_with_stale_index_provenance(state) or invalidated
    previously_complete = bool(state.get("day_complete"))
    state["day_complete"] = _day_complete(state)
    if invalidated or previously_complete != state["day_complete"]:
        state["updated_at_utc"] = retrieved_at.isoformat()
        _write_checkpoint(root, state)
    return state, bool(state.get("day_started") or any(
        entry.get("attempts") for group in state["captures"].values() for entry in group.values()
    ))


def _day_complete(state: dict[str, Any]) -> bool:
    indices = state["captures"].get("indices", {})
    fields = state["captures"].get("fields", {})
    for step in STEPS:
        index_entry = indices.get(f"{state['run_date']}/{step}", {})
        index_attempt = index_entry.get("latest", {})
        if (not index_entry.get("complete") or not isinstance(index_attempt, dict)
                or not index_attempt.get("complete")):
            return False
        index_hash = index_attempt.get("sha256")
        for param in PARAMS:
            field_entry = fields.get(f"{state['run_date']}/{step}/{param}", {})
            field_attempt = field_entry.get("latest", {})
            if (not field_entry.get("complete") or not isinstance(field_attempt, dict)
                    or not field_attempt.get("complete")
                    or not index_hash or field_attempt.get("index_sha256") != index_hash):
                return False
    return True


def _invalidate_fields_with_stale_index_provenance(state: dict[str, Any]) -> bool:
    run_date = state["run_date"]
    indices = state["captures"].get("indices", {})
    fields = state["captures"].get("fields", {})
    invalidated = False
    for step in STEPS:
        index_entry = indices.get(f"{run_date}/{step}")
        index_attempt = index_entry.get("latest") if index_entry else None
        expected_hash = index_attempt.get("sha256") if index_attempt and index_attempt.get("complete") else None
        for param in PARAMS:
            field_entry = fields.get(f"{run_date}/{step}/{param}")
            if not field_entry or not field_entry.get("complete"):
                continue
            field_attempt = field_entry.get("latest")
            if (not expected_hash or not isinstance(field_attempt, dict)
                    or field_attempt.get("index_sha256") != expected_hash):
                field_entry["complete"] = False
                invalidated = True
                if isinstance(field_attempt, dict):
                    field_attempt["complete"] = False
                    field_attempt["state"] = "incomplete"
                    field_attempt.setdefault("issues", []).append("index_provenance_hash_mismatch")
    return invalidated


def _store_attempt(root: Path, state: dict[str, Any], collection: str, key: str,
                   metadata: dict[str, Any], body: bytes | None, extension: str) -> dict[str, Any]:
    entry = state["captures"][collection].setdefault(key, {"complete": False, "attempts": []})
    attempt_no = len(entry["attempts"]) + 1
    stamp = metadata["captured_at_utc"].replace(":", "").replace("-", "").replace("+00:00", "Z")
    raw_hash = hashlib.sha256(body).hexdigest() if body is not None else None
    suffix = raw_hash[:12] if raw_hash else "noresponse"
    unique = uuid.uuid4().hex[:8]
    stem = f"ecmwf_{collection}_{key.replace('/', '_')}_a{attempt_no:02d}_{stamp}_{suffix}_{unique}"
    if body is not None:
        raw_path = root / f"{stem}.{extension}"
        pilot._atomic_immutable(raw_path, body)
        metadata["raw_file"] = raw_path.name
        metadata["sha256"] = raw_hash
    metadata_path = root / f"{stem}.meta.json"
    encoded = (json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    pilot._atomic_immutable(metadata_path, encoded)
    attempt = {"metadata_file": metadata_path.name, "metadata_sha256": hashlib.sha256(encoded).hexdigest(),
               "raw_file": metadata.get("raw_file"), "sha256": raw_hash,
               "state": metadata["state"], "complete": bool(metadata.get("complete")),
               "issues": list(metadata.get("issues", [])),
               "selected_records": metadata.get("selected_records"),
               "index_sha256": metadata.get("index_sha256"),
               "last_modified_utc": metadata.get("last_modified_utc"),
               "last_modified_before_origin": metadata.get("last_modified_before_origin"),
               "last_modified_issue": metadata.get("last_modified_issue"),
               "response_bytes": metadata.get("response_bytes", 0)}
    entry["attempts"].append(attempt)
    entry["latest"] = attempt
    entry["complete"] = attempt["complete"]
    state["updated_at_utc"] = metadata["captured_at_utc"]
    _invalidate_fields_with_stale_index_provenance(state)
    state["day_complete"] = _day_complete(state)
    _write_checkpoint(root, state)
    return attempt


def _request_metadata(run_date: str, step: int, kind: str, url: str,
                      captured_at: dt.datetime) -> dict[str, Any]:
    _, run_init, _, _ = _urls(run_date, step)
    return {"kind": kind, "run_date": run_date, "run_initialization_utc": run_init,
            "step_hours": step, "source": BASE_URL, "source_scope_hash": SCOPE_HASH,
            "request_url": url, "captured_at_utc": captured_at.isoformat(),
            "attempt_count": 0, "http_status": None, "headers": {},
            "response_bytes": 0, "transfer_error": None, "state": "error",
            "complete": False, "issues": []}


def _capture_index(run_date: str, step: int, *, root: Path, state: dict[str, Any],
                   session: Any, gate: pilot.RequestGate, sleeper: Any) -> dict[str, Any]:
    _, _, index_url, grib_url = _urls(run_date, step)
    key = f"{run_date}/{step}"
    response, attempts, transport_error = pilot._request(
        session, index_url, params=None, headers=None, gate=gate, sleeper=sleeper)
    body = None
    if response is None:
        metadata = _request_metadata(run_date, step, "index", index_url, _now_utc())
        metadata.update(grib_url=grib_url, attempt_count=attempts)
        metadata["issues"].append(transport_error)
        return _store_attempt(root, state, "indices", key, metadata, body, "index")
    body, transfer_error = pilot._read_bounded(response)
    status = int(response.status_code)
    headers = pilot._safe_headers(response.headers)
    metadata = _request_metadata(run_date, step, "index", index_url, _now_utc())
    metadata.update(grib_url=grib_url, attempt_count=attempts)
    metadata.update(http_status=status, headers=headers, response_bytes=len(body),
                   transfer_error=transfer_error)
    if transfer_error:
        metadata["state"] = "partial"
        metadata["issues"].append(transfer_error)
    elif status == 404:
        metadata["state"] = "unavailable"
        metadata["issues"].append("index_http_404")
    elif status != 200:
        metadata["issues"].append("index_http_status_not_200")
    else:
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) != len(body):
                    metadata["issues"].append("index_content_length_mismatch")
            except ValueError:
                metadata["issues"].append("index_content_length_invalid")
        if not metadata["issues"]:
            records, errors = pilot._parse_index(body, run_date, step)
            metadata["selected_records"] = records
            metadata["field_index_issues"] = errors
            metadata["complete"] = all(param in records and not errors[param] for param in PARAMS)
            metadata["state"] = "complete" if metadata["complete"] else "incomplete"
            metadata["issues"].extend(issue for param in PARAMS for issue in errors[param])
    return _store_attempt(root, state, "indices", key, metadata, body, "index")


def _capture_field(run_date: str, step: int, param: str, record: dict[str, Any], *,
                   index_attempt: dict[str, Any], root: Path, state: dict[str, Any],
                   session: Any, gate: pilot.RequestGate, sleeper: Any) -> dict[str, Any]:
    _, _, _, grib_url = _urls(run_date, step)
    key = f"{run_date}/{step}/{param}"
    offset, length = record["_offset"], record["_length"]
    end = offset + length - 1
    provenance = {"parameter": param, "index_sha256": index_attempt.get("sha256"),
                  "requested_offset": offset, "requested_length": length, "requested_end": end,
                  "last_modified_raw": None, "last_modified_utc": None,
                  "last_modified_before_origin": None,
                  "last_modified_proves_first_publication": False}
    if length > MAX_RESPONSE_BYTES:
        metadata = _request_metadata(run_date, step, "field", grib_url, _now_utc())
        metadata.update(provenance)
        metadata["issues"].append("field_length_exceeds_10mb_response_limit")
        return _store_attempt(root, state, "fields", key, metadata, None, "grib2")
    response, attempts, transport_error = pilot._request(
        session, grib_url, params=None, headers={"Range": f"bytes={offset}-{end}"},
        gate=gate, sleeper=sleeper)
    if response is None:
        metadata = _request_metadata(run_date, step, "field", grib_url, _now_utc())
        metadata.update(provenance, attempt_count=attempts)
        metadata["issues"].append(transport_error)
        return _store_attempt(root, state, "fields", key, metadata, None, "grib2")
    body, transfer_error = pilot._read_bounded(response)
    status = int(response.status_code)
    headers = pilot._safe_headers(response.headers)
    metadata = _request_metadata(run_date, step, "field", grib_url, _now_utc())
    metadata.update(provenance, attempt_count=attempts)
    metadata.update(http_status=status, headers=headers, response_bytes=len(body),
                    transfer_error=transfer_error)
    if transfer_error:
        metadata["state"] = "partial"
        metadata["issues"].append(transfer_error)
    elif status == 404:
        metadata["state"] = "unavailable"
        metadata["issues"].append("range_http_404")
    elif status != 206:
        metadata["issues"].append("range_http_status_not_206")
    else:
        content_range = headers.get("content-range", "")
        match = pilot.CONTENT_RANGE_RE.fullmatch(content_range)
        if not match:
            metadata["issues"].append("content_range_missing_or_invalid")
        else:
            start, stop, total = map(int, match.groups())
            metadata["content_range"] = content_range
            metadata["content_range_total"] = total
            if (start, stop) != (offset, end):
                metadata["issues"].append("content_range_bounds_mismatch")
            if end >= total:
                metadata["issues"].append("requested_range_outside_object")
        if len(body) != length:
            metadata["issues"].append("field_length_mismatch")
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                if int(content_length) != len(body):
                    metadata["issues"].append("range_content_length_mismatch")
            except ValueError:
                metadata["issues"].append("range_content_length_invalid")
        metadata["issues"].extend(pilot._validate_grib(body, length))
        last_modified = pilot._parse_last_modified(headers.get("last-modified"))
        metadata["last_modified_raw"] = headers.get("last-modified")
        metadata["last_modified_utc"] = last_modified.isoformat() if last_modified else None
        metadata["origin_deadline_utc"] = pilot._deadline_utc(run_date).isoformat()
        if last_modified is None:
            metadata["last_modified_issue"] = "missing_or_invalid"
            metadata["last_modified_status"] = "missing_or_invalid"
        else:
            metadata["last_modified_before_origin"] = last_modified < pilot._deadline_utc(run_date)
            metadata["last_modified_status"] = (
                "before_origin" if metadata["last_modified_before_origin"] else "after_origin"
            )
        metadata["complete"] = not metadata["issues"]
        metadata["state"] = "complete" if metadata["complete"] else "incomplete"
    return _store_attempt(root, state, "fields", key, metadata, body, "grib2")


def _day_metrics(state: dict[str, Any] | None, run_date: str) -> dict[str, Any]:
    if state is None:
        return {"run_date": run_date, "started": False, "day_complete": False,
                "captured": False,
                "captured_indices": 0, "complete_indices": 0,
                "captured_fields": 0, "complete_fields": 0,
                "last_modified_before_origin_fields": 0,
                "last_modified_after_origin_fields": 0,
                "last_modified_missing_or_invalid_fields": 0,
                "failed_indices": 0, "failed_fields": 0, "bytes_captured": 0}
    indices, fields = state["captures"]["indices"], state["captures"]["fields"]
    entries = list(indices.values()) + list(fields.values())
    bytes_captured = sum(attempt.get("response_bytes", 0)
                         for entry in entries for attempt in entry.get("attempts", []))
    complete_indices = sum(bool(indices.get(f"{run_date}/{step}", {}).get("complete")) for step in STEPS)
    complete_fields = sum(bool(fields.get(f"{run_date}/{step}/{param}", {}).get("complete"))
                          for step in STEPS for param in PARAMS)
    complete_field_attempts = [fields[f"{run_date}/{step}/{param}"].get("latest", {})
                               for step in STEPS for param in PARAMS
                               if fields.get(f"{run_date}/{step}/{param}", {}).get("complete")]
    before_origin = sum(attempt.get("last_modified_before_origin") is True
                        for attempt in complete_field_attempts)
    after_origin = sum(attempt.get("last_modified_before_origin") is False
                       for attempt in complete_field_attempts)
    missing_last_modified = sum(attempt.get("last_modified_before_origin") is None
                                for attempt in complete_field_attempts)
    expected_index_keys = [f"{run_date}/{step}" for step in STEPS]
    expected_field_keys = [f"{run_date}/{step}/{param}" for step in STEPS for param in PARAMS]
    captured_indices = sum(bool(indices.get(key, {}).get("attempts")) for key in expected_index_keys)
    captured_fields = sum(bool(fields.get(key, {}).get("attempts")) for key in expected_field_keys)
    failed_indices = sum(key in indices and not indices[key].get("complete") for key in expected_index_keys)
    failed_fields = sum(key in fields and not fields[key].get("complete") for key in expected_field_keys)
    started = bool(state.get("day_started") or entries)
    return {"run_date": run_date, "started": started, "captured": bool(entries),
            "day_complete": bool(state.get("day_complete")),
            "captured_indices": captured_indices, "complete_indices": complete_indices,
            "captured_fields": captured_fields, "complete_fields": complete_fields,
            "last_modified_before_origin_fields": before_origin,
            "last_modified_after_origin_fields": after_origin,
            "last_modified_missing_or_invalid_fields": missing_last_modified,
            "failed_indices": failed_indices, "failed_fields": failed_fields,
            "bytes_captured": bytes_captured,
            "latest_checkpoint": state.get("latest_checkpoint")}


def _summary(all_dates: list[str], selected_dates: list[str],
             states: dict[str, dict[str, Any]], skipped_incomplete: list[str]) -> dict[str, Any]:
    metrics = [_day_metrics(states.get(run_date), run_date) for run_date in all_dates]
    expected_fields = len(all_dates) * len(STEPS) * len(PARAMS)
    complete_fields = sum(day["complete_fields"] for day in metrics)
    before_origin_fields = sum(day["last_modified_before_origin_fields"] for day in metrics)
    after_origin_fields = sum(day["last_modified_after_origin_fields"] for day in metrics)
    missing_last_modified_fields = sum(day["last_modified_missing_or_invalid_fields"] for day in metrics)
    complete_days = sum(day["day_complete"] for day in metrics)
    started_days = sum(day["started"] for day in metrics)
    captured_days = sum(day["captured"] for day in metrics)
    bytes_captured = sum(day["bytes_captured"] for day in metrics)
    failures = {"incomplete_days": sum(day["started"] and not day["day_complete"] for day in metrics),
                "not_started_days": len(all_dates) - started_days,
                "incomplete_indices": sum(day["failed_indices"] for day in metrics),
                "incomplete_fields": sum(day["failed_fields"] for day in metrics),
                "missing_indices": len(all_dates) * len(STEPS) - sum(day["captured_indices"] for day in metrics),
                "missing_fields": expected_fields - sum(day["captured_fields"] for day in metrics)}
    selected_set = set(selected_dates)
    selected_complete = all(states[date].get("day_complete", False) for date in selected_dates)
    return {"schema_version": 1, "source": BASE_URL, "status": "full_period_collection",
            "source_scope": SOURCE_SCOPE, "source_scope_hash": SCOPE_HASH,
            "run_date_start": all_dates[0], "run_date_end": all_dates[-1],
            "expected_days": len(all_dates), "selected_days_this_invocation": len(selected_dates),
            "selected_run_dates": selected_dates,
            "expected_indices": len(all_dates) * len(STEPS),
            "captured_indices": sum(day["captured_indices"] for day in metrics),
            "complete_indices": sum(day["complete_indices"] for day in metrics),
            "expected_fields": expected_fields,
            "captured_fields": sum(day["captured_fields"] for day in metrics),
            "complete_fields": complete_fields, "structurally_complete_fields": complete_fields,
            "last_modified_before_origin_fields": before_origin_fields,
            "last_modified_after_origin_fields": after_origin_fields,
            "last_modified_missing_or_invalid_fields": missing_last_modified_fields,
            "last_modified_assessed_fields": before_origin_fields + after_origin_fields,
            "last_modified_before_origin_ratio": (
                before_origin_fields / complete_fields if complete_fields else None
            ),
            "complete_days": complete_days, "captured_days": captured_days,
            "started_days": started_days,
            "coverage_fraction_fields": complete_fields / expected_fields,
            "complete": complete_days == len(all_dates), "selected_complete": selected_complete,
            "failures": failures, "bytes_captured_total": bytes_captured,
            "skipped_incomplete_run_dates": skipped_incomplete,
            "day_status": [day for day in metrics if day["run_date"] in selected_set or day["started"]]}


def collect(*, raw_dir: str | Path = "data/raw/ecmwf_field_collection", session: Any = requests,
            retrieved_at: dt.datetime | None = None, sleeper: Any = time.sleep,
            clock: Any = time.monotonic, max_dates: int | None = None,
            retry_incomplete: bool = False) -> dict[str, Any]:
    all_dates = expected_run_dates()
    if max_dates is not None and (isinstance(max_dates, bool) or not 1 <= max_dates <= len(all_dates)):
        raise ValueError(f"max_dates must be between 1 and {len(all_dates)}")
    selected_dates = all_dates[:max_dates] if max_dates is not None else all_dates
    root = Path(raw_dir)
    root.mkdir(parents=True, exist_ok=True)
    captured = (retrieved_at or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    states: dict[str, dict[str, Any]] = {}
    started: dict[str, bool] = {}
    # Revalidate every completed artifact before using it for coverage or reuse.
    for run_date in all_dates:
        states[run_date], started[run_date] = _load_day(root, run_date, captured)
    gate = pilot.RequestGate(sleeper=sleeper, clock=clock)
    skipped_incomplete = []
    for run_date in selected_dates:
        state = states[run_date]
        if state.get("day_complete"):
            continue
        if started[run_date] and not retry_incomplete:
            skipped_incomplete.append(run_date)
            continue
        if not started[run_date]:
            state["day_started"] = True
            state["updated_at_utc"] = captured.isoformat()
            _write_checkpoint(root, state)
            started[run_date] = True
        for step in STEPS:
            index_key = f"{run_date}/{step}"
            index_entry = state["captures"]["indices"].get(index_key)
            if index_entry and index_entry.get("complete"):
                index_attempt = index_entry["latest"]
            else:
                index_attempt = _capture_index(run_date, step, root=root, state=state,
                                               session=session, gate=gate, sleeper=sleeper,
                                               )
                started[run_date] = True
            records = index_attempt.get("selected_records") or {}
            for param in PARAMS:
                record = records.get(param)
                if record is None:
                    continue
                field_key = f"{run_date}/{step}/{param}"
                field_entry = state["captures"]["fields"].get(field_key)
                if (field_entry and field_entry.get("complete")
                        and field_entry["latest"].get("index_sha256") != index_attempt.get("sha256")):
                    field_entry["complete"] = False
                    field_entry["latest"]["complete"] = False
                    field_entry["latest"]["state"] = "incomplete"
                    field_entry["latest"].setdefault("issues", []).append("index_provenance_hash_mismatch")
                if field_entry and field_entry.get("complete"):
                    continue
                _capture_field(run_date, step, param, record, index_attempt=index_attempt,
                               root=root, state=state, session=session, gate=gate,
                               sleeper=sleeper)
                started[run_date] = True
        state["day_complete"] = _day_complete(state)
        if not started[run_date]:
            state["updated_at_utc"] = captured.isoformat()
            _write_checkpoint(root, state)
            started[run_date] = True
    report = _summary(all_dates, selected_dates, states, skipped_incomplete)
    finished_at = _now_utc()
    report["collection_started_at_utc"] = captured.isoformat()
    report["retrieved_at_utc"] = finished_at.isoformat()
    encoded = (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    summary_path = root / f"ecmwf_field_collection_summary_{finished_at.strftime('%Y%m%dT%H%M%S%fZ')}_{hashlib.sha256(encoded).hexdigest()[:12]}.json"
    pilot._atomic_immutable(summary_path, encoded)
    report["summary_report"] = summary_path.name
    report["summary_sha256"] = hashlib.sha256(encoded).hexdigest()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/ecmwf_field_collection")
    parser.add_argument("--max-dates", type=int, help="process the first N dates of the frozen range")
    parser.add_argument("--retry-incomplete", action="store_true",
                        help="retry already-started but incomplete days; default leaves them untouched")
    parser.add_argument("--execute", action="store_true", help="send bounded Google mirror requests")
    args = parser.parse_args(argv)
    dates = expected_run_dates()
    if args.max_dates is not None and not 1 <= args.max_dates <= len(dates):
        parser.error(f"--max-dates must be between 1 and {len(dates)}")
    selected_count = args.max_dates if args.max_dates is not None else len(dates)
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "source_scope": SOURCE_SCOPE,
                          "expected_days": len(dates), "selected_days_this_invocation": selected_count,
                          "run_date_start": dates[0], "run_date_end": dates[-1],
                          "steps_hours": list(STEPS), "parameters": list(PARAMS),
                          "requests_per_day_max": len(STEPS) * (1 + len(PARAMS)),
                          "logical_requests_max_this_invocation": selected_count * len(STEPS) * (1 + len(PARAMS)),
                          "max_http_attempts_per_logical_request": MAX_ATTEMPTS,
                          "timeout_seconds": TIMEOUT_SECONDS,
                          "transport_retry_delay_seconds": RETRY_DELAY_SECONDS,
                          "response_body_limit_bytes": MAX_RESPONSE_BYTES,
                          "minimum_seconds_between_http_attempt_starts": MIN_REQUEST_INTERVAL_SECONDS,
                          "verify_tls": True, "allow_redirects": False,
                          "full_grib_downloads": False, "retry_incomplete": args.retry_incomplete,
                          "request_sent": False}, ensure_ascii=False))
        return 0
    try:
        report = collect(raw_dir=args.raw_dir, max_dates=args.max_dates,
                         retry_incomplete=args.retry_incomplete)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["selected_complete"] else 1
    except (OSError, ValueError, CollectionError, pilot.PilotError) as exc:
        print(f"ECMWF collection failed: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
