"""List fixed ECMWF Google Cloud Storage date prefixes with bounded pagination.

Dry-run is the default. Execution records listing pages only; a date prefix
does not prove any particular run, forecast field, or publication time exists.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import requests

API_URL = "https://storage.googleapis.com/storage/v1/b/ecmwf-open-data/o"
YEARS = ("2024", "2025", "2026")
MAX_RESULTS = 1000
MAX_PAGES_PER_YEAR = 10
MAX_RESPONSE_BYTES = 10_000_000
TIMEOUT_SECONDS = 30
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.0
DATE_PREFIX_RE = re.compile(r"^(\d{4})(\d{2})(\d{2})/$")
DATE_RANGES = (
    {"name": "training_targets", "start": "2024-01-01", "end": "2025-09-01"},
    {"name": "test_origin_previous_days", "start": "2025-09-14", "end": "2026-09-22"},
)


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
    allowed = {"content-type", "content-length", "date", "etag", "last-modified"}
    return {str(key).lower(): str(value) for key, value in headers.items()
            if str(key).lower() in allowed}


def _request(session: Any, url: str, *, params: dict[str, Any], sleeper: Any):
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(url, params=params, timeout=TIMEOUT_SECONDS,
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


def _parse_date_prefixes(values: Any, year: str) -> tuple[list[str], list[str]]:
    if not isinstance(values, list):
        return [], ["prefixes_not_array"]
    accepted: set[str] = set()
    invalid: list[str] = []
    for value in values:
        if not isinstance(value, str):
            invalid.append(str(value))
            continue
        match = DATE_PREFIX_RE.fullmatch(value)
        if not match or match.group(1) != year:
            invalid.append(value)
            continue
        try:
            dt.date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            invalid.append(value)
            continue
        accepted.add(value[:-1])
    return sorted(accepted), invalid


def _save_page(metadata: dict[str, Any], body: bytes | None, raw_dir: Path,
               stamp: str) -> dict[str, Any]:
    sha = _sha256(body) if body is not None else None
    token = metadata["request"].get("pageToken")
    token_key = hashlib.sha256(token.encode()).hexdigest()[:8] if token else "first"
    stem = f"ecmwf_listing_{metadata['year']}_p{metadata['page_number']:03d}_{token_key}_{stamp}_{sha[:12] if sha else 'noresponse'}"
    if body is not None:
        raw_path = raw_dir / f"{stem}.body"
        _atomic_immutable(raw_path, body)
        metadata["raw_file"] = raw_path.name
        metadata["sha256"] = sha
    meta_path = raw_dir / f"{stem}.meta.json"
    encoded = (json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _atomic_immutable(meta_path, encoded)
    metadata["metadata_file"] = meta_path.name
    return metadata


def _year_listing(year: str, *, raw_dir: Path, session: Any,
                  retrieved: dt.datetime, sleeper: Any) -> dict[str, Any]:
    prefixes: set[str] = set()
    pages: list[dict[str, Any]] = []
    issues: list[str] = []
    page_token: str | None = None
    seen_tokens: set[str] = set()
    state = "incomplete"
    for page_number in range(1, MAX_PAGES_PER_YEAR + 1):
        params: dict[str, Any] = {"prefix": year, "delimiter": "/", "maxResults": MAX_RESULTS}
        if page_token is not None:
            params["pageToken"] = page_token
        response, attempts, transport_error = _request(session, API_URL, params=params, sleeper=sleeper)
        page_meta: dict[str, Any] = {
            "year": year, "page_number": page_number, "request": params,
            "attempts": attempts, "retrieved_at_utc": retrieved.isoformat(),
            "http_status": None, "response_headers": {}, "response_bytes": 0,
            "transfer_error": None, "state": "error", "issues": [],
            "returned_prefixes": [], "next_page_token": None,
        }
        if response is None:
            page_meta["issues"].append(transport_error)
            saved = _save_page(page_meta, None, raw_dir, retrieved.strftime("%Y%m%dT%H%M%S%fZ"))
            pages.append(saved)
            issues.append(f"page_{page_number}_{transport_error}")
            state = "error"
            break
        body, transfer_error = _read_bounded(response)
        status = int(response.status_code)
        headers = _headers(response.headers)
        page_meta.update(http_status=status, response_headers=headers,
                         response_bytes=len(body), transfer_error=transfer_error)
        if transfer_error:
            page_meta["issues"].append(transfer_error)
            page_meta["state"] = "partial"
            saved = _save_page(page_meta, body, raw_dir, retrieved.strftime("%Y%m%dT%H%M%S%fZ"))
            pages.append(saved)
            issues.append(f"page_{page_number}_{transfer_error}")
            state = "incomplete"
            break
        if not 200 <= status < 300:
            page_meta["state"] = "unavailable" if status == 404 else "error"
            page_meta["issues"].append("http_404" if status == 404 else "http_status_not_success")
            saved = _save_page(page_meta, body, raw_dir, retrieved.strftime("%Y%m%dT%H%M%S%fZ"))
            pages.append(saved)
            issues.append(f"page_{page_number}_http_{status}")
            state = "incomplete"
            break
        content_length = headers.get("content-length")
        if content_length is not None:
            try:
                length_ok = int(content_length) == len(body)
            except ValueError:
                length_ok = False
            if not length_ok:
                page_meta["state"] = "partial"
                page_meta["issues"].append("content_length_mismatch")
                saved = _save_page(page_meta, body, raw_dir, retrieved.strftime("%Y%m%dT%H%M%S%fZ"))
                pages.append(saved)
                issues.append(f"page_{page_number}_content_length_mismatch")
                state = "incomplete"
                break
        try:
            payload = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError):
            page_meta["state"] = "incomplete"
            page_meta["issues"].append("response_not_json")
            saved = _save_page(page_meta, body, raw_dir, retrieved.strftime("%Y%m%dT%H%M%S%fZ"))
            pages.append(saved)
            issues.append(f"page_{page_number}_response_not_json")
            state = "incomplete"
            break
        if not isinstance(payload, dict):
            page_meta["state"] = "incomplete"
            page_meta["issues"].append("response_root_not_object")
            saved = _save_page(page_meta, body, raw_dir, retrieved.strftime("%Y%m%dT%H%M%S%fZ"))
            pages.append(saved)
            issues.append(f"page_{page_number}_response_root_not_object")
            state = "incomplete"
            break
        page_prefixes, invalid = _parse_date_prefixes(payload.get("prefixes", []), year)
        prefixes.update(page_prefixes)
        page_meta["returned_prefixes"] = page_prefixes
        page_meta["invalid_prefixes"] = invalid
        if invalid:
            page_meta["issues"].append("invalid_date_prefixes")
        next_token = payload.get("nextPageToken")
        if next_token is not None and (not isinstance(next_token, str) or not next_token):
            page_meta["issues"].append("next_page_token_invalid")
            token_valid = False
        elif isinstance(next_token, str) and next_token:
            page_meta["next_page_token"] = next_token
            token_valid = True
        else:
            token_valid = False
        token_loop = token_valid and (next_token in seen_tokens or next_token == page_token)
        page_limit = token_valid and page_number == MAX_PAGES_PER_YEAR
        if token_loop:
            page_meta["issues"].append("pagination_token_loop")
            issues.append(f"page_{page_number}_pagination_token_loop")
        if page_limit:
            page_meta["issues"].append("page_limit_reached_with_next_token")
            issues.append("page_limit_reached_with_next_token")
        page_meta["state"] = "complete" if not page_meta["issues"] else "incomplete"
        saved = _save_page(page_meta, body, raw_dir, retrieved.strftime("%Y%m%dT%H%M%S%fZ"))
        pages.append(saved)
        if invalid:
            issues.append(f"page_{page_number}_invalid_date_prefixes")
        if page_meta["issues"] and "next_page_token_invalid" in page_meta["issues"]:
            issues.append(f"page_{page_number}_next_page_token_invalid")
            state = "incomplete"
            break
        if token_loop or page_limit:
            state = "incomplete"
            break
        if next_token is None:
            state = "complete" if not issues else "incomplete"
            break
        seen_tokens.add(next_token)
        page_token = next_token
    else:
        state = "incomplete"
    return {"year": year, "state": state, "complete": state == "complete",
            "prefixes": sorted(prefixes), "prefix_count": len(prefixes),
            "pages": pages, "issues": issues}


def _date_range_report(listings: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    present_all = {date for item in listings.values() for date in item["prefixes"]}
    output = []
    for spec in DATE_RANGES:
        start = dt.date.fromisoformat(spec["start"])
        end = dt.date.fromisoformat(spec["end"])
        dates = []
        day = start
        while day <= end:
            dates.append(day.isoformat())
            day += dt.timedelta(days=1)
        years = sorted({date[:4] for date in dates})
        listing_complete = all(listings[year]["complete"] for year in years)
        present = [date for date in dates if date.replace("-", "") in present_all]
        output.append({"name": spec["name"], "start": spec["start"], "end": spec["end"],
                       "expected_date_count": len(dates), "listing_complete": listing_complete,
                       "observed_present_dates": present,
                       "missing_dates": [date for date in dates if date not in present] if listing_complete else None,
                       "coverage_claim": "date_prefixes_only; no run, field, or publication-time guarantee"})
    return output


def collect(*, raw_dir: str | Path = "data/raw/ecmwf_archive_listing",
            session: Any = requests, retrieved_at: dt.datetime | None = None,
            sleeper: Any = time.sleep) -> dict[str, Any]:
    retrieved = (retrieved_at or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    root = Path(raw_dir)
    listings = {}
    for year in YEARS:
        listings[year] = _year_listing(year, raw_dir=root, session=session,
                                       retrieved=retrieved, sleeper=sleeper)
    report = {
        "schema_version": 1, "source": API_URL, "bucket": "ecmwf-open-data",
        "retrieved_at_utc": retrieved.isoformat(), "years": listings,
        "date_ranges": _date_range_report(listings),
        "complete": all(item["complete"] for item in listings.values()),
        "limitations": ["date prefix existence does not establish a specific run or field",
                        "date prefix existence does not prove origin-time availability",
                        "no GRIB field or model data was retrieved"],
    }
    report["state"] = "complete" if report["complete"] else "incomplete"
    stamp = retrieved.strftime("%Y%m%dT%H%M%S%fZ")
    encoded = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    report_path = root / f"ecmwf_archive_listing_summary_{stamp}_{_sha256(encoded)[:12]}.json"
    _atomic_immutable(report_path, encoded)
    report["summary_report"] = report_path.name
    report["summary_sha256"] = _sha256(encoded)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/ecmwf_archive_listing")
    parser.add_argument("--execute", action="store_true", help="list bounded pages for the three fixed years")
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "source": API_URL, "bucket": "ecmwf-open-data",
                          "years": list(YEARS), "max_pages_per_year": MAX_PAGES_PER_YEAR,
                          "max_results_per_page": MAX_RESULTS, "request_sent": False}, ensure_ascii=False))
        return 0
    try:
        report = collect(raw_dir=args.raw_dir)
        print(json.dumps(report, ensure_ascii=False))
        return 0 if report["complete"] else 1
    except (OSError, ValueError) as exc:
        print(f"listing failed: {exc}", file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
