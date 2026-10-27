"""Frozen 23-query Nongnet historical availability screen; dry-run by default."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
from typing import Any
import uuid

import requests

from agri_data import nongnet_national_volume_pilot as volume

ENDPOINT = volume.ENDPOINT
DATES = ("2023-01-03", "2024-09-11", "2025-09-11")
SCREEN_DATE = "2026-09-11"
CATEGORIES = (
    ("cabbage", "1001", ""),
    ("radish", "1101", ""),
    ("onion", "1201", ""),
    ("pepper", "1207", ""),
    ("garlic_all", "1209", ""),
    ("garlic_peeled", "1209", "120906"),
)
DEFAULT_OUTPUT = Path("data/raw/nongnet_national_history_screen")
EXISTING_CABBAGE_JSON = Path(
    "data/raw/nongnet_national_volume_pilot/"
    "nongnet_national_volume_2026-09-11_a1_2026-10-08T121544.278359+0000_bc0d0b11.json"
)
EXISTING_CABBAGE_SHA256 = "1d0a1e01f12bd7427cbe811b5c41ce09bbfc3f6ac309d33346c35b69fb148714"
EXISTING_CABBAGE_PARAMS = {
    "searchUnitGrade": "normal", "searchDate": "2026년 09월 11일",
    "searchSymbol1": "100000", "searchSymbol2": "1001", "searchSymbol3": "",
}
TIMEOUT_SECONDS = volume.TIMEOUT_SECONDS
MAX_ATTEMPTS = volume.MAX_ATTEMPTS
RETRY_DELAY_SECONDS = volume.RETRY_DELAY_SECONDS
MIN_REQUEST_INTERVAL_SECONDS = volume.MIN_REQUEST_INTERVAL_SECONDS


def request_matrix() -> list[dict[str, str]]:
    """Return the exact ordered GET matrix (18 historical plus five screen GETs)."""
    requests_out = []
    for date in (*DATES, SCREEN_DATE):
        for label, item_code, variety_code in CATEGORIES:
            if date == SCREEN_DATE and label == "cabbage":
                continue  # use already-collected JSON; do not issue a second GET
            parsed = dt.date.fromisoformat(date)
            requests_out.append({
                "requestedDate": date,
                "category": label,
                "searchUnitGrade": "normal",
                "searchDate": f"{parsed.year}년 {parsed.month:02d}월 {parsed.day:02d}일",
                "searchSymbol1": "100000",
                "searchSymbol2": item_code,
                "searchSymbol3": variety_code,
            })
    return requests_out


def _positive_quality(validation: dict[str, Any]) -> dict[str, Any]:
    """Add explicit zero/nonpositive quality flags without changing source values."""
    added_quality_issue = False
    for row in validation.get("rows", []):
        source = row.get("sourceValues")
        if not isinstance(source, dict):
            continue
        for field in ("selectDay", "selectDayWgt"):
            if field not in source:
                continue
            value = source[field]
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError):
                continue
            issue = f"{field}_zero_value" if number == 0 else f"{field}_nonpositive_value"
            if number <= 0 and issue not in row["issues"]:
                row["issues"].append(issue)
                added_quality_issue = True
                marker = f"row_{row['sourceIndex']}_questionable"
                if marker not in validation["issues"]:
                    validation["issues"].append(marker)
    if added_quality_issue and validation.get("state") in {"valid", "requested_date_missing"}:
        validation["state"] = "questionable"
    _sync_matching_issues(validation)
    return validation


def _sync_matching_issues(validation: dict[str, Any]) -> None:
    matched_row = validation.get("matchingRow")
    if not isinstance(matched_row, dict):
        return
    match = next((row for row in validation.get("rows", [])
                  if row.get("dateMatch") and
                  isinstance(row.get("sourceValues"), dict) and
                  row["sourceValues"].get("saledate") == matched_row.get("saledate")), None)
    if match is not None:
        matched_row["issues"] = list(match.get("issues", []))


def validate_json(body: bytes, requested_date: str, http_status: int = 200, *,
                  truncated: bool = False, transport_error: str | None = None) -> dict[str, Any]:
    """Reuse the reviewed exact-date/row validator and add positivity checks."""
    validation = _positive_quality(volume.validate_json(
        body, requested_date, http_status, truncated=truncated, transport_error=transport_error
    ))
    format_issue_added = False
    for row in validation.get("rows", []):
        source = row.get("sourceValues")
        if not isinstance(source, dict):
            continue
        saledate = source.get("saledate")
        if not isinstance(saledate, str):
            continue  # null and non-string values are already classified by the shared validator
        try:
            parsed = dt.datetime.strptime(saledate, "%Y%m%d")
            valid_date = parsed.strftime("%Y%m%d") == saledate
        except ValueError:
            valid_date = False
        if not valid_date and "saledate_invalid_format" not in row["issues"]:
            row["issues"].append("saledate_invalid_format")
            marker = f"row_{row['sourceIndex']}_questionable"
            if marker not in validation["issues"]:
                validation["issues"].append(marker)
            format_issue_added = True
    if format_issue_added and validation.get("state") in {"valid", "requested_date_missing"}:
        validation["state"] = "questionable"
    _sync_matching_issues(validation)
    return validation


def collect(output: Path = DEFAULT_OUTPUT, *, session: Any = None,
            now=lambda: dt.datetime.now(dt.timezone.utc), sleep=time.sleep) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    session.verify = True
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    results = []
    previous_start = None
    for request in request_matrix():
        date = request["requestedDate"]
        params = {key: value for key, value in request.items() if key not in {"requestedDate", "category"}}
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
            stem = (f"nongnet_national_history_{date}_{request['category']}_a{attempt}_" +
                    captured.replace(":", "").replace("+00:00", "Z") + "_" + uuid.uuid4().hex[:8])
            headers = dict(response.headers) if response is not None else {}
            headers = {key: ("[redacted]" if key.lower() in {"set-cookie", "authorization"} else value)
                       for key, value in headers.items()}
            status = response.status_code if response is not None else 0
            raw_file = stem + ".json" if body else None
            if body:
                volume._immutable(output / raw_file, body)
            validation = validate_json(body, date, status, truncated=truncated, transport_error=error)
            manifest = {"schemaVersion": 1, "endpoint": ENDPOINT, "requestParams": params,
                        "requestedDate": date, "category": request["category"], "attempt": attempt,
                        "retrievedAtUtc": captured, "httpStatus": status or None,
                        "responseHeaders": headers, "rawFile": raw_file,
                        "responseBytes": len(body), "sha256": volume._sha(body) if body else None,
                        "truncated": truncated, "transportError": error,
                        "validation": validation, "complete": False}
            manifest_file = stem + ".manifest.json"
            volume._immutable(output / manifest_file,
                              (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
            attempts.append({"manifestFile": manifest_file, **manifest})
            if 200 <= status < 500 and status != 429 and not truncated and not error:
                break
            if attempt < MAX_ATTEMPTS:
                sleep(RETRY_DELAY_SECONDS)
        results.append({"requestedDate": date, "category": request["category"],
                        "attempts": attempts, "state": attempts[-1]["validation"]["state"],
                        "complete": False})
    return results


def reuse_existing_cabbage(path: Path = EXISTING_CABBAGE_JSON) -> dict[str, Any]:
    """Validate and identify the immutable earlier 2026-09-11 cabbage JSON artifact."""
    if not path.exists():
        return {"state": "existing_json_unavailable", "path": str(path), "complete": False}
    body = path.read_bytes()
    digest = hashlib.sha256(body).hexdigest()
    if digest != EXISTING_CABBAGE_SHA256:
        return {"state": "existing_json_hash_mismatch", "path": str(path),
                "sha256": digest, "complete": False}
    manifest_path = path.with_name(path.name.removesuffix(".json") + ".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"state": "existing_json_manifest_unavailable", "path": str(path),
                "sha256": digest, "complete": False}
    if manifest.get("sha256") != digest or manifest.get("requestedDate") != SCREEN_DATE or manifest.get("requestParams") != EXISTING_CABBAGE_PARAMS:
        return {"state": "existing_json_provenance_mismatch", "path": str(path),
                "sha256": digest, "complete": False}
    return {"state": "reused_existing_json", "path": str(path),
            "sha256": digest, "responseBytes": len(body),
            "requestParams": EXISTING_CABBAGE_PARAMS,
            "validation": validate_json(body, SCREEN_DATE), "complete": False}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="send the frozen 23 GET requests")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--existing-cabbage-json", type=Path, default=EXISTING_CABBAGE_JSON)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "requestSent": False,
                          "requestCount": len(request_matrix()), "requests": request_matrix(),
                          "reused2026Cabbage": str(args.existing_cabbage_json)}, ensure_ascii=False))
        return 0
    results = collect(args.output)
    summary = [{"date": row["requestedDate"], "category": row["category"],
                "state": row["state"], "attemptCount": len(row["attempts"]),
                "matchingRow": row["attempts"][-1]["validation"]["matchingRow"]}
               for row in results]
    print(json.dumps({"mode": "execute", "requestCount": len(results),
                      "reused2026Cabbage": reuse_existing_cabbage(args.existing_cabbage_json),
                      "results": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
