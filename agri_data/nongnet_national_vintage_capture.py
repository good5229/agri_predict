"""Capture one immutable six-item Nongnet nationwide response vintage.

Dry-run is the default. Each explicit execution creates a distinct run ID and
preserves every HTTP attempt; this is an availability/version capture only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import time
import uuid
from typing import Any
from zoneinfo import ZoneInfo

import requests

from agri_data import nongnet_national_history_screen as history
from agri_data import nongnet_national_volume_pilot as volume

ENDPOINT = volume.ENDPOINT
DEFAULT_OUTPUT = Path("data/raw/nongnet_national_vintage_capture")
TIMEOUT_SECONDS = volume.TIMEOUT_SECONDS
MAX_RESPONSE_BYTES = volume.MAX_RESPONSE_BYTES
MAX_ATTEMPTS = volume.MAX_ATTEMPTS
RETRY_DELAY_SECONDS = volume.RETRY_DELAY_SECONDS
MIN_REQUEST_INTERVAL_SECONDS = volume.MIN_REQUEST_INTERVAL_SECONDS
CATEGORIES = history.CATEGORIES


class _NonfiniteJSONNumber(ValueError):
    pass


def _strict_json_precheck(body: bytes) -> str | None:
    """Reject Python's permissive NaN tokens and exponent overflow before validation."""
    def parse_constant(token: str) -> Any:
        raise _NonfiniteJSONNumber(f"nonstandard_json_constant_{token.lower()}")

    def parse_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise _NonfiniteJSONNumber("json_number_overflow_nonfinite")
        return value

    try:
        json.loads(body.decode("utf-8-sig"), parse_constant=parse_constant, parse_float=parse_float)
    except _NonfiniteJSONNumber as exc:
        return str(exc)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None  # shared validator provides its established malformed-JSON classification
    return None


def validate_json(body: bytes, requested_date: str, http_status: int = 200, *,
                  truncated: bool = False, transport_error: str | None = None) -> dict[str, Any]:
    """Use the reviewed source validator after rejecting nonfinite JSON values."""
    if truncated or transport_error or http_status < 200 or http_status >= 300 or not body:
        return history.validate_json(body, requested_date, http_status, truncated=truncated,
                                     transport_error=transport_error)
    nonfinite = _strict_json_precheck(body)
    if nonfinite:
        return {"state": "nonfinite_json_value", "issues": [nonfinite], "rowCount": 0,
                "matchingDateCount": 0, "matchingRow": None, "rows": []}
    return history.validate_json(body, requested_date, http_status, truncated=truncated,
                                 transport_error=transport_error)


def validate_observation_date(value: str, *, today: dt.date | None = None) -> str:
    try:
        observed = dt.date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("observation date must be YYYY-MM-DD") from exc
    if observed.isoformat() != value:
        raise ValueError("observation date must be YYYY-MM-DD")
    today = today or dt.datetime.now(ZoneInfo("Asia/Seoul")).date()
    if observed > today:
        raise ValueError("observation date cannot be in the future (KST)")
    return value


def request_matrix(observation_date: str, *, today: dt.date | None = None) -> list[dict[str, str]]:
    observation_date = validate_observation_date(observation_date, today=today)
    parsed = dt.date.fromisoformat(observation_date)
    korean_date = f"{parsed.year}년 {parsed.month:02d}월 {parsed.day:02d}일"
    return [{
        "category": label,
        "searchUnitGrade": "normal",
        "searchDate": korean_date,
        "searchSymbol1": "100000",
        "searchSymbol2": item_code,
        "searchSymbol3": variety_code,
    } for label, item_code, variety_code in CATEGORIES]


def _response_headers(response: Any) -> dict[str, Any]:
    return {key: ("[redacted]" if key.lower() in {"set-cookie", "authorization"} else value)
            for key, value in dict(response.headers).items()}


def collect(observation_date: str, output: Path = DEFAULT_OUTPUT, *, session: Any = None,
            now=lambda: dt.datetime.now(dt.timezone.utc), sleep=time.sleep,
            today: dt.date | None = None) -> dict[str, Any]:
    observation_date = validate_observation_date(observation_date, today=today)
    output.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    session.verify = True
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    run_started = now().astimezone(dt.timezone.utc)
    run_id = run_started.strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:8]
    results = []
    previous_start = None
    for request in request_matrix(observation_date, today=today):
        category = request["category"]
        params = {key: value for key, value in request.items() if key != "category"}
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
                body, truncated, error = volume._read_bounded(response)
            except requests.RequestException as exc:
                error = f"request_error:{type(exc).__name__}:{exc}"
            captured = now().astimezone(dt.timezone.utc).isoformat()
            token = captured.replace(":", "").replace("+00:00", "Z") + "_" + uuid.uuid4().hex[:8]
            stem = f"nongnet_national_vintage_{observation_date}_{category}_{run_id}_a{attempt}_{token}"
            status = response.status_code if response is not None else 0
            raw_file = stem + ".json" if body else None
            if body:
                volume._immutable(output / raw_file, body)
            validation = validate_json(body, observation_date, status,
                                       truncated=truncated, transport_error=error)
            manifest = {
                "schemaVersion": 1, "runId": run_id, "endpoint": ENDPOINT,
                "requestedDate": observation_date, "observedDate": observation_date,
                "category": category, "requestParams": params, "attempt": attempt,
                "retrievedAtUtc": captured, "httpStatus": status or None,
                "responseHeaders": _response_headers(response) if response is not None else {},
                "rawFile": raw_file, "responseBytes": len(body),
                "sha256": volume._sha(body) if body else None, "truncated": truncated,
                "transportError": error, "validation": validation, "complete": False,
            }
            manifest_file = stem + ".manifest.json"
            volume._immutable(output / manifest_file,
                              (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2,
                                          allow_nan=False) + "\n").encode())
            attempts.append({"manifestFile": manifest_file, **manifest})
            if 200 <= status < 500 and status != 429 and not truncated and not error:
                break
            if attempt < MAX_ATTEMPTS:
                sleep(RETRY_DELAY_SECONDS)
        results.append({"category": category, "attempts": attempts,
                        "state": attempts[-1]["validation"]["state"], "complete": False})
    return {"runId": run_id, "requestedDate": observation_date,
            "startedAtUtc": run_started.isoformat(), "results": results, "complete": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="observation date YYYY-MM-DD (not future in KST)")
    parser.add_argument("--execute", action="store_true", help="send exactly six GET requests")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        date = validate_observation_date(args.date)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "requestSent": False,
                          "requestCount": 6, "requestedDate": date,
                          "requests": request_matrix(date)}, ensure_ascii=False))
        return 0
    result = collect(date, args.output)
    print(json.dumps({"mode": "execute", "requestCount": len(result["results"]),
                      "runId": result["runId"],
                      "results": [{"category": row["category"], "state": row["state"],
                                   "attemptCount": len(row["attempts"])} for row in result["results"]]},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
