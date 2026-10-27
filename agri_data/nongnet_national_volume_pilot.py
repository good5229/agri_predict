"""Three-query raw JSON pilot for Nongnet nationwide average-grade volume.

Dry-run is the default. This validates access and source fields only; it does
not train or score a model and never equates price, weight, or grade fields.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
from typing import Any

import requests

ENDPOINT = "https://nongnet.or.kr/front/M000000259/marketInfo/getNationWideUnitList.do"
DATES = ("2026-09-11", "2026-09-12", "2026-09-14")
TIMEOUT_SECONDS = (5, 25)
MAX_ATTEMPTS = 3
MAX_RESPONSE_BYTES = 3_000_000
RETRY_DELAY_SECONDS = 1.0
MIN_REQUEST_INTERVAL_SECONDS = 1.0
DEFAULT_OUTPUT = Path("data/raw/nongnet_national_volume_pilot")
EXPECTED_FIELDS = ("saledate", "selectDay", "selectDayWgt")


class CollectionError(RuntimeError):
    pass


def request_matrix() -> list[dict[str, str]]:
    params = []
    for date in DATES:
        parsed = dt.date.fromisoformat(date)
        params.append({"searchUnitGrade": "normal",
                       "searchDate": f"{parsed.year}년 {parsed.month:02d}월 {parsed.day:02d}일",
                       "searchSymbol1": "100000", "searchSymbol2": "1001",
                       "searchSymbol3": ""})
    return params


def _numeric_issue(value: Any) -> str | None:
    if value is None or value == "" or (isinstance(value, str) and value.strip().lower() in {"null", "none", "-"}):
        return "null_value"
    if isinstance(value, bool):
        return "non_numeric_value"
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return "non_numeric_value"
    if not math.isfinite(number):
        return "nonfinite_value"
    if number < 0:
        return "negative_value"
    return None


def validate_json(body: bytes, requested_date: str, http_status: int = 200, *,
                  truncated: bool = False, transport_error: str | None = None) -> dict[str, Any]:
    """Preserve all datalist rows and identify exactly one requested-day row."""
    expected_saledate = requested_date.replace("-", "")
    result: dict[str, Any] = {"state": None, "issues": [], "rowCount": 0,
                              "matchingDateCount": 0, "matchingRow": None, "rows": []}
    if truncated:
        result["state"] = "truncated"
        result["issues"].append(transport_error or "response_truncated")
        return result
    if transport_error:
        result["state"] = "transport_error"
        result["issues"].append(transport_error)
        return result
    if http_status < 200 or http_status >= 300:
        result["state"] = "http_error"
        result["issues"].append(f"http_status_{http_status}")
        return result
    if not body:
        result["state"] = "empty_response"
        return result
    try:
        doc = json.loads(body.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        result["state"] = "non_json_response"
        result["issues"].append("response_not_valid_json")
        return result
    if isinstance(doc, dict) and any(key in doc and doc[key] not in (None, "", False, [], {})
                                     for key in ("error", "errors")):
        result["state"] = "error_payload"
        result["issues"].append("json_error_payload")
        return result
    if not isinstance(doc, dict) or not isinstance(doc.get("datalist"), list):
        result["state"] = "invalid_json_structure"
        result["issues"].append("expected_object_with_datalist_array")
        return result
    datalist = doc["datalist"]
    result["rowCount"] = len(datalist)
    if not datalist:
        result["state"] = "normal_empty"
        return result
    matches: list[dict[str, Any]] = []
    for index, source_row in enumerate(datalist):
        row = {"sourceIndex": index, "sourceValues": source_row, "issues": [], "dateMatch": False}
        if not isinstance(source_row, dict):
            row["issues"].append("row_not_object")
            result["rows"].append(row)
            result["issues"].append(f"row_{index}_questionable")
            continue
        for field in EXPECTED_FIELDS:
            if field not in source_row:
                row["issues"].append(f"missing_{field}")
        saledate = source_row.get("saledate")
        if saledate is not None and not isinstance(saledate, str):
            row["issues"].append("saledate_not_string")
        elif saledate is None:
            row["issues"].append("saledate_null_value")
        elif saledate is not None and saledate != expected_saledate:
            row["dateMatch"] = False
        else:
            row["dateMatch"] = saledate == expected_saledate
        if row["dateMatch"]:
            matches.append(row)
        for field in ("selectDay", "selectDayWgt"):
            if field in source_row:
                issue = _numeric_issue(source_row[field])
                if issue:
                    row["issues"].append(f"{field}_{issue}")
        if row["issues"]:
            result["issues"].append(f"row_{index}_questionable")
        result["rows"].append(row)
    result["matchingDateCount"] = len(matches)
    if not matches:
        if result["issues"]:
            result["state"] = "questionable"
        else:
            result["state"] = "requested_date_missing"
            result["issues"].append("no_row_for_requested_saledate")
    elif len(matches) > 1:
        result["state"] = "duplicate_requested_date"
        result["issues"].append("multiple_rows_for_requested_saledate")
    else:
        row = matches[0]
        source = row.get("sourceValues")
        result["matchingRow"] = {"saledate": source.get("saledate"),
                                 "selectDay": source.get("selectDay"),
                                 "selectDayWgt": source.get("selectDayWgt"),
                                 "issues": list(row["issues"])}
        result["state"] = "questionable" if result["issues"] or row["issues"] else "valid"
    return result


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp, path)
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def _read_bounded(response: Any) -> tuple[bytes, bool, str | None]:
    chunks, total, truncated, error = [], 0, False, None
    try:
        for chunk in response.iter_content(64 * 1024):
            if not chunk:
                continue
            remaining = MAX_RESPONSE_BYTES - total
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                truncated, error = True, "response_exceeded_3mb_limit"
                break
            chunks.append(chunk)
            total += len(chunk)
    except requests.RequestException as exc:
        truncated, error = True, f"stream_error:{type(exc).__name__}:{exc}"
    finally:
        response.close()
    return b"".join(chunks), truncated, error


def collect(output: Path = DEFAULT_OUTPUT, *, session: Any = None,
            now=lambda: dt.datetime.now(dt.timezone.utc), sleep=time.sleep) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    session.verify = True
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    results = []
    previous_start = None
    for date, params in zip(DATES, request_matrix()):
        attempts = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if previous_start is not None:
                elapsed = time.monotonic() - previous_start
                if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
                    sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
            previous_start = time.monotonic()
            response, body, truncated, error = None, b"", False, None
            try:
                response = session.get(ENDPOINT, params=params, timeout=TIMEOUT_SECONDS,
                                       verify=True, stream=True)
                body, truncated, error = _read_bounded(response)
            except requests.RequestException as exc:
                error = f"request_error:{type(exc).__name__}:{exc}"
            captured = now().astimezone(dt.timezone.utc).isoformat()
            token = captured.replace(":", "").replace("+00:00", "Z") + "_" + uuid.uuid4().hex[:8]
            stem = f"nongnet_national_volume_{date}_a{attempt}_{token}"
            headers = dict(response.headers) if response is not None else {}
            headers = {key: ("[redacted]" if key.lower() in {"set-cookie", "authorization"} else value)
                       for key, value in headers.items()}
            status = response.status_code if response is not None else 0
            raw_file = stem + ".json" if body else None
            if body:
                _immutable(output / raw_file, body)
            validation = validate_json(body, date, status, truncated=truncated, transport_error=error)
            manifest = {"schemaVersion": 1, "endpoint": ENDPOINT, "requestParams": params,
                        "requestedDate": date, "attempt": attempt, "retrievedAtUtc": captured,
                        "httpStatus": status or None, "responseHeaders": headers,
                        "rawFile": raw_file, "responseBytes": len(body),
                        "sha256": _sha(body) if body else None, "truncated": truncated,
                        "transportError": error, "validation": validation,
                        "complete": False}
            manifest_file = stem + ".manifest.json"
            _immutable(output / manifest_file,
                       (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
            attempts.append({"manifestFile": manifest_file, **manifest})
            if 200 <= status < 500 and status != 429 and not truncated and not error:
                break
            if attempt < MAX_ATTEMPTS:
                sleep(RETRY_DELAY_SECONDS)
        results.append({"requestedDate": date, "attempts": attempts,
                        "state": attempts[-1]["validation"]["state"], "complete": False})
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="send the three frozen GET requests")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "endpoint": ENDPOINT,
                          "requestSent": False, "requestCount": len(DATES),
                          "requests": request_matrix()}, ensure_ascii=False))
        return 0
    results = collect(args.output)
    summary = [{"date": row["requestedDate"], "state": row["state"],
                "attemptCount": len(row["attempts"]),
                "matchingRow": row["attempts"][-1]["validation"]["matchingRow"]}
               for row in results]
    print(json.dumps({"mode": "execute", "requestCount": len(results), "results": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
