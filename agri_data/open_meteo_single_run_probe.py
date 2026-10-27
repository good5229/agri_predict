"""One-shot, immutable probe for an archived Open-Meteo IFS run.

The command is dry-run by default. A successful capture describes the source
response only; it does not establish the run's original public release time.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

URL = "https://single-runs-api.open-meteo.com/v1/forecast"
LATITUDE = 34.57
LONGITUDE = 126.60
RUN = "2025-08-31T12:00"
MODEL = "ecmwf_ifs"
HOURLY = "temperature_2m,precipitation"
FORECAST_DAYS = 10
MAX_RESPONSE_BYTES = 2_000_000
TIMEOUT_SECONDS = 20
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.0


class ProbeError(RuntimeError):
    pass


def request_params() -> dict[str, str | int | float]:
    return {"latitude": LATITUDE, "longitude": LONGITUDE, "run": RUN,
            "models": MODEL, "hourly": HOURLY, "forecast_days": FORECAST_DAYS,
            "timezone": "UTC", "timeformat": "iso8601"}


def _atomic_immutable(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL on the final path prevents clobbering an earlier capture. Rename
    # is atomic; the temporary file is in the same directory/filesystem.
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
            raise
        finally:
            if tmp.exists():
                tmp.unlink()
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def validate_response(body: bytes, status: int) -> dict[str, Any]:
    """Return validation evidence without changing or coercing source values."""
    result: dict[str, Any] = {"valid": False, "complete": False, "issues": [], "response_model": None,
                              "grid": None, "hourly_count": None, "time_start": None,
                              "time_end": None, "response_timezone": None, "null_counts": {},
                              "incomplete_variables": [],
                              "variable_coverage": {}}
    issues: list[str] = result["issues"]
    if not 200 <= status < 300:
        issues.append("http_status_not_success")
        return result
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        issues.append("response_not_json")
        return result
    if not isinstance(payload, dict):
        issues.append("response_root_not_object")
        return result
    result["response_model"] = payload.get("model") or payload.get("models")
    reported_model = result["response_model"]
    if isinstance(reported_model, str) and reported_model and reported_model != MODEL:
        issues.append("response_model_conflicts_with_requested_model")
    elif isinstance(reported_model, list) and reported_model and MODEL not in reported_model:
        issues.append("response_model_conflicts_with_requested_model")
    coord = {key: payload.get(key) for key in ("latitude", "longitude", "elevation")}
    result["grid"] = coord
    for field, expected in (("latitude", LATITUDE), ("longitude", LONGITUDE)):
        value = payload.get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            issues.append(f"missing_or_invalid_{field}")
        elif abs(value - expected) > 0.5:
            issues.append(f"{field}_outside_expected_grid_tolerance")
    result["response_timezone"] = {"timezone": payload.get("timezone"),
                                   "utc_offset_seconds": payload.get("utc_offset_seconds"),
                                   "timezone_abbreviation": payload.get("timezone_abbreviation")}
    response_tz = payload.get("timezone")
    utc_offset = payload.get("utc_offset_seconds")
    if utc_offset != 0 or response_tz not in {"UTC", "GMT"}:
        issues.append("response_timezone_not_utc")
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        issues.append("hourly_not_object")
        return result
    times = hourly.get("time")
    if not isinstance(times, list) or not times:
        issues.append("hourly_time_missing_or_empty")
        return result
    parsed: list[dt.datetime] = []
    for value in times:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", value):
            issues.append("hourly_time_not_iso8601_hour")
            break
        try:
            parsed.append(dt.datetime.strptime(value, "%Y-%m-%dT%H:%M").replace(tzinfo=dt.timezone.utc))
        except ValueError:
            issues.append("hourly_time_invalid_calendar_value")
            break
    result["hourly_count"] = len(times)
    result["time_start"] = times[0] if times else None
    result["time_end"] = times[-1] if times else None
    if len(parsed) == len(times):
        if any((b - a) != dt.timedelta(hours=1) for a, b in zip(parsed, parsed[1:])):
            issues.append("hourly_time_not_contiguous")
        if parsed[0] != dt.datetime.fromisoformat(RUN).replace(tzinfo=dt.timezone.utc):
            issues.append("hourly_time_start_does_not_match_run")
        if len(parsed) != FORECAST_DAYS * 24:
            issues.append("hourly_count_does_not_match_10_day_horizon")
    for name in ("temperature_2m", "precipitation"):
        values = hourly.get(name)
        if not isinstance(values, list):
            issues.append(f"{name}_missing_or_not_array")
            result["variable_coverage"][name] = {"array_present": False, "expected_count": len(times),
                                                   "returned_count": None, "numeric_count": 0,
                                                   "null_count": 0, "invalid_count": 0,
                                                   "coverage_fraction": 0.0, "complete": False}
            result["null_counts"][name] = None
            result["incomplete_variables"].append(name)
            continue
        if len(values) != len(times):
            issues.append(f"{name}_length_mismatch")
        null_count = 0
        numeric_count = 0
        invalid_count = 0
        for value in values:
            if value is None:
                null_count += 1
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                invalid_count += 1
            else:
                numeric_count += 1
        if invalid_count:
            issues.append(f"{name}_contains_non_finite_or_non_numeric_value")
        expected = len(times)
        variable_complete = (len(values) == expected and numeric_count == expected and
                             null_count == 0 and invalid_count == 0)
        result["variable_coverage"][name] = {
            "array_present": True, "expected_count": expected, "returned_count": len(values),
            "numeric_count": numeric_count, "null_count": null_count,
            "invalid_count": invalid_count,
            "coverage_fraction": numeric_count / expected if expected else 0.0,
            "complete": variable_complete,
        }
        result["null_counts"][name] = null_count
        if not variable_complete:
            result["incomplete_variables"].append(name)
    result["valid"] = not issues
    result["complete"] = result["valid"] and all(
        item["complete"] for item in result["variable_coverage"].values()
    ) and len(result["variable_coverage"]) == 2
    return result


def probe(*, raw_dir: str | Path = "data/raw/open_meteo_single_runs",
          session: Any = requests, now: dt.datetime | None = None,
          sleeper: Any = time.sleep) -> dict[str, Any]:
    attempts = 0
    response = None
    for attempt in range(MAX_ATTEMPTS):
        attempts = attempt + 1
        try:
            response = session.get(URL, params=request_params(), timeout=TIMEOUT_SECONDS,
                                   verify=True, allow_redirects=False, stream=True)
            break
        except requests.RequestException:
            if attempt + 1 == MAX_ATTEMPTS:
                raise ProbeError("Open-Meteo request failed after bounded transport retries") from None
            sleeper(RETRY_DELAY_SECONDS * (attempt + 1))
    assert response is not None
    body = bytearray()
    transfer_error: str | None = None
    try:
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            remaining = MAX_RESPONSE_BYTES - len(body)
            body.extend(chunk[:remaining])
            if len(chunk) > remaining:
                transfer_error = "response_exceeded_maximum_size"
                break
    except (requests.RequestException, OSError):
        transfer_error = "response_stream_failed"
    finally:
        response.close()
    raw = bytes(body)
    status = int(response.status_code)
    retrieved = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    stamp = retrieved.strftime("%Y%m%dT%H%M%S%fZ")
    digest = hashlib.sha256(raw).hexdigest()
    stem = f"single_run_{RUN.replace('-', '').replace(':', '')}_{stamp}_{digest[:12]}"
    raw_path = Path(raw_dir) / f"{stem}.json"
    metadata_path = Path(raw_dir) / f"{stem}.meta.json"
    _atomic_immutable(raw_path, raw)
    headers = {k.lower(): v for k, v in response.headers.items()
               if k.lower() in {"content-type", "content-length", "date", "server"}}
    validation = validate_response(raw, status)
    if transfer_error:
        validation["valid"] = False
        validation["complete"] = False
        validation["issues"].append(transfer_error)
    metadata = {
        "schema_version": 1, "source": URL, "request": request_params(),
        "retrieved_at_utc": retrieved.isoformat(), "http_status": status,
        "response_headers": headers, "response_bytes": len(raw), "sha256": digest,
        "attempts": attempts, "redirect_refused": status in {301, 302, 303, 307, 308},
        "transfer_state": "complete" if transfer_error is None else "incomplete",
        "transfer_error": transfer_error,
        "validation": validation, "complete": bool(validation["complete"]),
        "run_initialization_utc": RUN, "public_release_time": None,
        "raw_file": raw_path.name,
    }
    encoded = (json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    try:
        _atomic_immutable(metadata_path, encoded)
    except BaseException:
        # Keep the raw evidence; it is content-addressed and immutable.
        raise
    metadata["metadata_file"] = metadata_path.name
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/open_meteo_single_runs")
    parser.add_argument("--execute", action="store_true", help="send exactly one bounded diagnostic request")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "source": URL, "request": request_params(),
                          "request_sent": False}, ensure_ascii=False))
        return 0
    try:
        result = probe(raw_dir=args.raw_dir)
        print(json.dumps(result, ensure_ascii=False))
        return 0 if result["complete"] else 1
    except (ProbeError, OSError) as exc:
        print(f"probe failed: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
