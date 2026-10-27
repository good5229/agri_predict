"""Sample Google GCS availability metadata for three fixed ECMWF runs.

Dry-run is the default. Execution makes at most one index GET and one object
HEAD per fixed date; it does not request a GRIB byte range or model field.
"""
from __future__ import annotations

import argparse
import datetime as dt
import email.utils
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

BASE_URL = "https://storage.googleapis.com/ecmwf-open-data"
DATES = ("20250914", "20260314", "20260922")
RUN_TIME = "1200"
STEP = "24"
MAX_INDEX_BYTES = 10_000_000
TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.0
EXPECTED_FIELDS = {"time": RUN_TIME, "step": STEP, "type": "fc", "stream": "oper", "levtype": "sfc"}


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
        finally:
            if tmp.exists():
                tmp.unlink()
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def _headers(headers: Any) -> dict[str, str]:
    allowed = {"content-type", "content-length", "last-modified", "etag", "date"}
    return {str(key).lower(): str(value) for key, value in headers.items()
            if str(key).lower() in allowed}


def _request(session: Any, method: str, url: str, *, sleeper: Any,
             **kwargs: Any) -> tuple[Any | None, int, str | None]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = getattr(session, method.lower())(
                url, timeout=TIMEOUT_SECONDS, verify=True, allow_redirects=False, **kwargs)
            return response, attempt, None
        except requests.RequestException:
            if attempt == MAX_ATTEMPTS:
                return None, attempt, "transport_failed_after_bounded_retries"
            sleeper(RETRY_DELAY_SECONDS * attempt)
    return None, MAX_ATTEMPTS, "transport_failed_after_bounded_retries"


def _read_bounded(response: Any) -> tuple[bytes, str | None]:
    body = bytearray()
    transfer_error = None
    try:
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            remaining = MAX_INDEX_BYTES - len(body)
            body.extend(chunk[:remaining])
            if len(chunk) > remaining:
                transfer_error = "response_exceeded_10mb_limit"
                break
    except (requests.RequestException, OSError):
        transfer_error = "response_stream_failed"
    finally:
        response.close()
    return bytes(body), transfer_error


def _index_match(body: bytes, date: str) -> tuple[dict[str, Any] | None, list[str]]:
    issues = []
    matches = []
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return None, ["index_not_utf8"]
    expected = {**EXPECTED_FIELDS, "date": date, "param": "2t"}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            issues.append(f"invalid_json_line_{number}")
            continue
        if not isinstance(record, dict):
            issues.append(f"non_object_line_{number}")
            continue
        if all(str(record.get(key)) == value for key, value in expected.items()):
            matches.append(record)
    if issues:
        return None, issues
    if len(matches) != 1:
        return None, ["matching_2t_record_not_unique"]
    record = matches[0]
    offset, length = record.get("_offset"), record.get("_length")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        return None, ["index_offset_invalid"]
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        return None, ["index_length_invalid"]
    if offset + length > 2**63 - 1:
        return None, ["index_range_overflow"]
    return record, []


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


def _origin_deadline_utc(date: str) -> dt.datetime:
    next_day = dt.datetime.strptime(date, "%Y%m%d").date() + dt.timedelta(days=1)
    # 10:00 KST is 01:00 UTC.
    return dt.datetime.combine(next_day, dt.time(1), tzinfo=dt.timezone.utc)


def _urls(date: str) -> tuple[str, str]:
    key = f"{date}/12z/ifs/0p25/oper/{date}120000-24h-oper-fc"
    return BASE_URL + "/" + key + ".index", BASE_URL + "/" + key + ".grib2"


def _save(result: dict[str, Any], body: bytes | None, raw_dir: Path, retrieved: dt.datetime) -> dict[str, Any]:
    date = result["run_date"]
    stamp = retrieved.strftime("%Y%m%dT%H%M%S%fZ")
    digest = _sha256(body) if body is not None else None
    suffix = digest[:12] if digest else "noresponse"
    stem = f"ecmwf_archive_coverage_{date}_{stamp}_{suffix}"
    if body is not None:
        raw_path = raw_dir / f"{stem}.index"
        _atomic_immutable(raw_path, body)
        result["index"]["raw_file"] = raw_path.name
        result["index"]["sha256"] = digest
    metadata_path = raw_dir / f"{stem}.meta.json"
    encoded = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _atomic_immutable(metadata_path, encoded)
    result["metadata_file"] = metadata_path.name
    return result


def probe_date(date: str, *, raw_dir: str | Path = "data/raw/ecmwf_archive_coverage",
               session: Any = requests, retrieved_at: dt.datetime | None = None,
               sleeper: Any = time.sleep) -> dict[str, Any]:
    if date not in DATES:
        raise ValueError("date is outside the frozen three-date sample")
    retrieved = (retrieved_at or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    index_url, grib_url = _urls(date)
    raw_root = Path(raw_dir)
    result: dict[str, Any] = {
        "schema_version": 1, "source": "google_gcs", "run_date": date,
        "run_time_utc": RUN_TIME, "forecast_step_hours": 24,
        "index_url": index_url, "object_url": grib_url,
        "retrieved_at_utc": retrieved.isoformat(), "state": "error", "complete": False,
        "issues": [], "attempts": {}, "index": {}, "object_head": {},
        "expected_index_fields": {**EXPECTED_FIELDS, "date": date, "param": "2t"},
        "selected_record": None,
    }
    response, attempts, transport_error = _request(
        session, "GET", index_url, sleeper=sleeper, stream=True)
    result["attempts"]["index_get"] = attempts
    if response is None:
        result["issues"].append(transport_error)
        result["state"] = "error"
        return _save(result, None, raw_root, retrieved)
    body, transfer_error = _read_bounded(response)
    status = int(response.status_code)
    index_headers = _headers(response.headers)
    result["index"] = {"http_status": status, "headers": index_headers,
                       "response_bytes": len(body), "transfer_error": transfer_error}
    if status == 404:
        result["issues"].append("index_unavailable_404")
        result["state"] = "unavailable"
        return _save(result, body, raw_root, retrieved)
    if not 200 <= status < 300:
        result["issues"].append("index_http_error")
        result["state"] = "error"
        return _save(result, body, raw_root, retrieved)
    if transfer_error:
        result["issues"].append(transfer_error)
        result["state"] = "partial"
        return _save(result, body, raw_root, retrieved)
    content_length = index_headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) != len(body):
                result["issues"].append("index_content_length_mismatch")
                result["state"] = "partial"
                return _save(result, body, raw_root, retrieved)
        except ValueError:
            result["issues"].append("index_content_length_invalid")
            result["state"] = "partial"
            return _save(result, body, raw_root, retrieved)

    record, issues = _index_match(body, date)
    result["issues"].extend(issues)
    if record is None:
        result["state"] = "incomplete"
        return _save(result, body, raw_root, retrieved)
    result["selected_record"] = record

    head, attempts, transport_error = _request(session, "HEAD", grib_url, sleeper=sleeper)
    result["attempts"]["object_head"] = attempts
    if head is None:
        result["issues"].append(transport_error)
        result["state"] = "error"
        return _save(result, body, raw_root, retrieved)
    head_status = int(head.status_code)
    head_headers = _headers(head.headers)
    head.close()
    size = None
    try:
        size = int(head_headers.get("content-length", ""))
    except ValueError:
        pass
    last_modified_raw = head_headers.get("last-modified")
    last_modified = _parse_last_modified(last_modified_raw)
    deadline = _origin_deadline_utc(date)
    result["object_head"] = {
        "http_status": head_status, "headers": head_headers,
        "object_size_bytes": size, "last_modified_raw": last_modified_raw,
        "last_modified_utc": last_modified.isoformat() if last_modified else None,
        "next_day_10k_kst_utc": deadline.isoformat(),
        "last_modified_before_origin": last_modified < deadline if last_modified else None,
        "last_modified_proves_first_publication": False,
        "selected_index_range_within_object": (
            result["selected_record"]["_offset"] + result["selected_record"]["_length"] <= size
            if size is not None and result["selected_record"] is not None else None),
    }
    if head_status == 404:
        result["issues"].append("object_unavailable_404")
        result["state"] = "unavailable"
    elif head_status != 200:
        result["issues"].append("object_head_http_error")
        result["state"] = "error"
    elif size is None or size <= 0:
        result["issues"].append("object_content_length_missing_or_invalid")
        result["state"] = "incomplete"
    elif last_modified is None:
        result["issues"].append("last_modified_missing_or_invalid")
        result["state"] = "incomplete"
    elif result["selected_record"]["_offset"] + result["selected_record"]["_length"] > size:
        result["issues"].append("selected_index_range_outside_object")
        result["state"] = "incomplete"
    else:
        result["state"] = "complete"
        result["complete"] = True
    return _save(result, body, raw_root, retrieved)


def probe(*, raw_dir: str | Path = "data/raw/ecmwf_archive_coverage",
          session: Any = requests, retrieved_at: dt.datetime | None = None,
          sleeper: Any = time.sleep) -> list[dict[str, Any]]:
    retrieved = retrieved_at or dt.datetime.now(dt.timezone.utc)
    return [probe_date(date, raw_dir=raw_dir, session=session, retrieved_at=retrieved, sleeper=sleeper)
            for date in DATES]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/ecmwf_archive_coverage")
    parser.add_argument("--execute", action="store_true", help="perform six bounded metadata requests")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "source": BASE_URL, "dates": list(DATES),
                          "logical_requests_max": 6, "max_http_attempts_including_transport_retries": 18,
                          "request_sent": False}, ensure_ascii=False))
        return 0
    try:
        results = probe(raw_dir=args.raw_dir)
        print(json.dumps(results, ensure_ascii=False))
        return 0 if all(item["complete"] for item in results) else 1
    except (OSError, ValueError) as exc:
        print(f"probe failed: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
