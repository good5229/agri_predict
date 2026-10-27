"""One-shot, immutable capture of one ECMWF IFS operational GRIB field.

Dry-run is the default. The explicit execution mode retrieves one index, HEAD
metadata, and one bounded byte range; it does not decode or interpret values.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

SOURCE_BASES = {
    "aws": "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com",
    "cloudfront": "https://d2zvc0wgha4k2l.cloudfront.net",
    "google": "https://storage.googleapis.com/ecmwf-open-data",
}
OBJECT_KEY = "20250831/12z/ifs/0p25/oper/20250831120000-24h-oper-fc"
BASE_URL = SOURCE_BASES["aws"] + "/" + OBJECT_KEY
INDEX_URL = BASE_URL + ".index"
GRIB_URL = BASE_URL + ".grib2"
EXPECTED = {"param": "2t", "date": "20250831", "time": "1200", "step": "24",
            "type": "fc", "stream": "oper", "levtype": "sfc"}
MAX_INDEX_BYTES = 10_000_000
MAX_FIELD_BYTES = 10_000_000
TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.0


class ProbeError(RuntimeError):
    pass


def _atomic_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _safe_headers(headers: Any) -> dict[str, str]:
    allowed = {"content-type", "content-length", "content-range", "last-modified", "etag", "date"}
    return {str(k).lower(): str(v) for k, v in headers.items() if str(k).lower() in allowed}


def _declared_length_matches(headers: dict[str, str], body: bytes) -> bool:
    declared = headers.get("content-length")
    if declared is None:
        return True
    try:
        return int(declared) == len(body)
    except ValueError:
        return False


def _read_response(response: Any, maximum: int) -> tuple[bytes, str | None]:
    body = bytearray()
    error = None
    try:
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            remaining = maximum - len(body)
            body.extend(chunk[:remaining])
            if len(chunk) > remaining:
                error = "response_exceeded_configured_limit"
                break
    except (requests.RequestException, OSError):
        error = "response_stream_failed"
    finally:
        response.close()
    return bytes(body), error


def _request(session: Any, method: str, url: str, *, sleeper: Any,
             **kwargs: Any) -> tuple[Any | None, int, str | None]:
    attempts = 0
    for attempt in range(MAX_ATTEMPTS):
        attempts = attempt + 1
        try:
            response = session.request(method, url, timeout=TIMEOUT_SECONDS, verify=True,
                                       allow_redirects=False, **kwargs)
            return response, attempts, None
        except requests.RequestException:
            if attempt + 1 == MAX_ATTEMPTS:
                return None, attempts, "transport_failed_after_bounded_retries"
            sleeper(RETRY_DELAY_SECONDS * (attempt + 1))
    return None, attempts, "transport_failed_after_bounded_retries"


def _index_record(body: bytes) -> tuple[dict[str, Any] | None, list[str]]:
    issues: list[str] = []
    records: list[dict[str, Any]] = []
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        return None, ["index_not_utf8"]
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            issues.append(f"index_invalid_json_line_{line_number}")
            continue
        if not isinstance(record, dict):
            issues.append(f"index_non_object_line_{line_number}")
            continue
        if all(str(record.get(key)) == expected for key, expected in EXPECTED.items()):
            records.append(record)
    if issues:
        return None, issues
    if len(records) != 1:
        return None, ["matching_index_record_not_unique"]
    record = records[0]
    offset, length = record.get("_offset"), record.get("_length")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        return None, ["index_offset_invalid"]
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        return None, ["index_length_invalid"]
    if offset + length > 2**63 - 1:
        return None, ["index_range_overflow"]
    return record, []


def _validate_grib(body: bytes, expected_length: int) -> list[str]:
    issues = []
    if len(body) != expected_length:
        issues.append("field_length_mismatch")
    if len(body) < 20 or body[:4] != b"GRIB":
        issues.append("grib_magic_missing")
        return issues
    if body[7] != 2:
        issues.append("grib_edition_not_2")
    declared_length = int.from_bytes(body[8:16], "big")
    if declared_length != len(body):
        issues.append("grib_declared_length_mismatch")
    if not body.endswith(b"7777"):
        issues.append("grib_trailer_missing")
    return issues


def probe(*, raw_dir: str | Path = "data/raw/ecmwf_operational_probe",
          session: Any = requests, now: dt.datetime | None = None,
          sleeper: Any = time.sleep, source: str = "aws") -> dict[str, Any]:
    if source not in SOURCE_BASES:
        raise ValueError("source must be 'aws' or 'cloudfront'")
    base_url = SOURCE_BASES[source] + "/" + OBJECT_KEY
    index_url = base_url + ".index"
    grib_url = base_url + ".grib2"
    retrieved = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    stamp = retrieved.strftime("%Y%m%dT%H%M%S%fZ")
    root = Path(raw_dir)
    result: dict[str, Any] = {"schema_version": 1, "source": source,
                              "index_url": index_url, "grib_url": grib_url,
                              "expected_index_fields": EXPECTED, "retrieved_at_utc": retrieved.isoformat(),
                              "complete": False, "issues": [], "attempts": {}, "index": {},
                              "head": {}, "range": {}, "selected_record": None}

    index_response, attempts, transport_error = _request(
        session, "GET", index_url, sleeper=sleeper, stream=True)
    result["attempts"]["index_get"] = attempts
    if index_response is None:
        result["issues"].append(transport_error)
        return _save_result(result, root, stamp, index_body=None, field_body=None)
    index_body, stream_error = _read_response(index_response, MAX_INDEX_BYTES)
    status = int(index_response.status_code)
    index_headers = _safe_headers(index_response.headers)
    result["index"] = {"http_status": status, "headers": index_headers,
                       "response_bytes": len(index_body), "sha256": hashlib.sha256(index_body).hexdigest(),
                       "transfer_error": stream_error}
    if not 200 <= status < 300:
        result["issues"].append("index_http_status_not_success")
    if stream_error:
        result["issues"].append(f"index_{stream_error}")
    if not _declared_length_matches(index_headers, index_body):
        result["issues"].append("index_content_length_mismatch")
    if result["issues"]:
        return _save_result(result, root, stamp, index_body=index_body, field_body=None)

    record, record_issues = _index_record(index_body)
    result["issues"].extend(record_issues)
    if record is None:
        return _save_result(result, root, stamp, index_body=index_body, field_body=None)
    result["selected_record"] = record
    offset, length = record["_offset"], record["_length"]
    if length > MAX_FIELD_BYTES:
        result["issues"].append("selected_field_exceeds_configured_limit")
        return _save_result(result, root, stamp, index_body=index_body, field_body=None)

    head, attempts, transport_error = _request(session, "HEAD", grib_url, sleeper=sleeper)
    result["attempts"]["object_head"] = attempts
    if head is None:
        result["issues"].append(transport_error)
        return _save_result(result, root, stamp, index_body=index_body, field_body=None)
    head_status = int(head.status_code)
    head_headers = _safe_headers(head.headers)
    head.close()
    object_size = None
    try:
        object_size = int(head_headers.get("content-length", ""))
    except ValueError:
        pass
    result["head"] = {"http_status": head_status, "headers": head_headers,
                       "content_length": object_size, "last_modified": head_headers.get("last-modified")}
    if head_status != 200:
        result["issues"].append("object_head_http_status_not_200")
    if object_size is None or offset + length > object_size:
        result["issues"].append("selected_range_outside_object_or_size_unknown")
    if result["issues"]:
        return _save_result(result, root, stamp, index_body=index_body, field_body=None)

    end = offset + length - 1
    ranged, attempts, transport_error = _request(
        session, "GET", grib_url, sleeper=sleeper, stream=True,
        headers={"Range": f"bytes={offset}-{end}"})
    result["attempts"]["field_range_get"] = attempts
    if ranged is None:
        result["issues"].append(transport_error)
        return _save_result(result, root, stamp, index_body=index_body, field_body=None)
    field_body, range_stream_error = _read_response(ranged, MAX_FIELD_BYTES)
    range_status = int(ranged.status_code)
    range_headers = _safe_headers(ranged.headers)
    result["range"] = {"http_status": range_status, "headers": range_headers,
                       "requested_start": offset, "requested_end": end,
                       "response_bytes": len(field_body), "sha256": hashlib.sha256(field_body).hexdigest(),
                       "transfer_error": range_stream_error}
    if range_status != 206:
        result["issues"].append("range_http_status_not_206")
    expected_content_range = f"bytes {offset}-{end}/{object_size}"
    if range_headers.get("content-range") != expected_content_range:
        result["issues"].append("content_range_mismatch")
    if range_stream_error:
        result["issues"].append(f"range_{range_stream_error}")
    if not _declared_length_matches(range_headers, field_body):
        result["issues"].append("range_content_length_mismatch")
    result["issues"].extend(_validate_grib(field_body, length))
    result["complete"] = not result["issues"]
    return _save_result(result, root, stamp, index_body=index_body, field_body=field_body)


def _save_result(result: dict[str, Any], root: Path, stamp: str, *,
                 index_body: bytes | None, field_body: bytes | None) -> dict[str, Any]:
    index_sha = hashlib.sha256(index_body).hexdigest() if index_body is not None else None
    stem = f"ecmwf_ifs_20250831T1200_step24_{stamp}_{index_sha[:12] if index_sha else 'noindex'}"
    if index_body is not None:
        raw_index = root / f"{stem}.index"
        _atomic_immutable(raw_index, index_body)
        result["index"]["raw_file"] = raw_index.name
    if field_body is not None:
        field_sha = hashlib.sha256(field_body).hexdigest()
        raw_field = root / f"{stem}_2t.grib2"
        _atomic_immutable(raw_field, field_body)
        result["range"]["raw_file"] = raw_field.name
        result["range"]["sha256"] = field_sha
    metadata_path = root / f"{stem}.meta.json"
    metadata = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _atomic_immutable(metadata_path, metadata)
    result["metadata_file"] = metadata_path.name
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/ecmwf_operational_probe")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--cloudfront", action="store_true",
                              help="use the fixed CloudFront mirror instead of the AWS S3 hostname")
    source_group.add_argument("--google", action="store_true",
                              help="use the fixed Google Cloud Storage mirror instead of AWS S3")
    parser.add_argument("--execute", action="store_true", help="retrieve one index and one GRIB field range")
    args = parser.parse_args(argv)
    source = "cloudfront" if args.cloudfront else "google" if args.google else "aws"
    if not args.execute:
        base = SOURCE_BASES[source] + "/" + OBJECT_KEY
        print(json.dumps({"mode": "dry_run", "source": source,
                          "index_url": base + ".index", "grib_url": base + ".grib2",
                          "expected_index_fields": EXPECTED, "request_sent": False}, ensure_ascii=False))
        return 0
    try:
        result = probe(raw_dir=args.raw_dir, source=source)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["complete"] else 1
    except (OSError, ValueError) as exc:
        print(f"probe failed: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
