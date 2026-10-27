"""Bounded, provenance-preserving pilot capture for Nongnet Seoan pepper data.

The command is dry-run by default. A successful HTTP response is not evidence
that the requested week is complete; every manifest records that limitation.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import html
import io
import json
import math
import re
import time
import uuid
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

PAGE_URL = "https://nongnet.or.kr/front/M000000204/marketInfo/seoan.do"
POST_URL = "https://nongnet.or.kr/front/M000000204/marketInfo/seoanExcelDown.do"
WEEKS = (("2024-08-01", "2024-08-07"), ("2026-08-01", "2026-08-07"))
TIMEOUT_SECONDS = (5, 30)
MAX_ATTEMPTS = 3
MAX_RESPONSE_BYTES = 20_000_000
RETRY_DELAY_SECONDS = 1.0
MIN_REQUEST_INTERVAL_SECONDS = 1.0
NULLS = {"", "-", "--", "null", "NULL", "None", "없음"}


class CollectionError(RuntimeError):
    pass


def _read_bounded(response: Any) -> tuple[bytes, bool, str | None]:
    chunks: list[bytes] = []
    total = 0
    error = None
    truncated = False
    try:
        for chunk in response.iter_content(64 * 1024):
            if not chunk:
                continue
            remaining = MAX_RESPONSE_BYTES - total
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                truncated = True
                error = "response_exceeded_20mb_limit"
                break
            chunks.append(chunk)
            total += len(chunk)
    except requests.RequestException as exc:
        truncated = True
        error = f"stream_error:{type(exc).__name__}:{exc}"
    finally:
        response.close()
    return b"".join(chunks), truncated, error


def _norm(value: str) -> str:
    return re.sub(r"\s+", "", value).strip().lower()


def _table_rows(text: str) -> list[list[str]]:
    """Extract simple HTML tables without requiring a third-party parser."""
    tables = re.findall(r"<table\b[^>]*>(.*?)</table\s*>", text, flags=re.I | re.S)
    for table in tables:
        rows = []
        for row in re.findall(r"<tr\b[^>]*>(.*?)</tr\s*>", table, flags=re.I | re.S):
            cells = re.findall(r"<(?:td|th)\b[^>]*>(.*?)</(?:td|th)\s*>", row, flags=re.I | re.S)
            if cells:
                rows.append([html.unescape(re.sub(r"<[^>]+>", "", cell)).strip() for cell in cells])
        if rows:
            return rows
    return []


def _xlsx_rows(body: bytes) -> list[list[str]]:
    """Read the first worksheet from a standard XLSX package using the stdlib."""
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            shared = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                shared = ["".join(node.itertext()) for node in root.findall("{*}si")]
            sheet_name = next((name for name in archive.namelist()
                               if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)), None)
            if not sheet_name:
                return []
            root = ET.fromstring(archive.read(sheet_name))
            rows = []
            for row in root.findall(".//{*}sheetData/{*}row"):
                cells: dict[int, str] = {}
                for cell in row.findall("{*}c"):
                    ref = cell.attrib.get("r", "")
                    letters = re.match(r"[A-Z]+", ref)
                    if not letters:
                        continue
                    col = 0
                    for char in letters.group():
                        col = col * 26 + ord(char) - 64
                    value = cell.find("{*}v")
                    text = "" if value is None else value.text or ""
                    if cell.attrib.get("t") == "s" and text:
                        text = shared[int(text)]
                    elif cell.attrib.get("t") == "inlineStr":
                        text = "".join(cell.itertext())
                    cells[col - 1] = text
                if cells:
                    rows.append([cells.get(i, "") for i in range(max(cells) + 1)])
            return rows
    except (zipfile.BadZipFile, ET.ParseError, ValueError, IndexError, KeyError):
        return []


def _columns(header: list[str]) -> dict[str, int]:
    aliases = {
        "date": ("경락일", "거래일", "일자", "날짜", "auctiondate", "date"),
        "product": ("품목", "품종", "상품", "product", "spcs"),
        "weight_unit": ("단위", "중량단위", "수량단위", "weightunit"),
        "weight": ("물량", "수량", "중량", "거래량", "weight", "qty"),
        "price_unit": ("가격단위", "단가단위", "단위가격", "priceunit"),
        "price": ("경락가", "경락가격", "단가", "가격", "price"),
        "quantity_geun": ("총수량(근)", "수량(근)", "물량(근)"),
        "quantity_kg": ("총수량(kg)", "총수량(㎏)", "수량(kg)", "수량(㎏)", "물량(kg)", "물량(㎏)"),
        "price_geun": ("단가(근)", "경락가(근)", "가격(근)"),
        "price_kg": ("단가(kg)", "단가(㎏)", "경락가(kg)", "경락가(㎏)", "가격(kg)", "가격(㎏)"),
    }
    norm = [_norm(x) for x in header]
    found = {}
    # Prefer exact labels so "중량단위" cannot be mistaken for the weight value.
    for key, choices in aliases.items():
        exact = {_norm(alias) for alias in choices}
        found_match = next((i for i, cell in enumerate(norm) if cell in exact), None)
        if found_match is not None:
            found[key] = found_match
            continue
        for i, cell in enumerate(norm):
            if any(_norm(alias) in cell and cell != _norm(alias) + "단위"
                   for alias in choices):
                found[key] = i
                break
    if {"quantity_geun", "quantity_kg", "price_geun", "price_kg"}.issubset(found):
        for key in ("weight", "price", "weight_unit", "price_unit"):
            found.pop(key, None)
    return found


def _numeric_issue(value: str) -> str | None:
    if value.strip() in NULLS:
        return "null_value"
    cleaned = value.strip().replace(",", "").replace("원", "").replace("kg", "").replace("㎏", "").replace("근", "").strip()
    try:
        number = float(cleaned)
    except ValueError:
        return "non_numeric_value"
    if not math.isfinite(number):
        return "nonfinite_value"
    if number < 0:
        return "negative_value"
    return None


def validate_body(body: bytes, start: str, end: str, content_type: str = "") -> dict[str, Any]:
    """Parse recognizable HTML/CSV output and annotate, never discard, bad rows."""
    decoded = body.decode("utf-8-sig", errors="replace")
    lower = decoded.lower()
    result: dict[str, Any] = {"format": None, "rows": [], "issues": [], "table_header_found": False}
    if not body:
        result["issues"].append("empty_response")
        return result
    if body.startswith(b"PK\x03\x04"):
        rows = _xlsx_rows(body)
        result["format"] = "xlsx"
        if not rows:
            result["issues"].append("unrecognized_or_partial_xlsx")
            return result
    else:
        rows = []
    if any(term in lower for term in ("오류가 발생", "잘못된 요청", "error", "access denied", "exception")) and "<table" not in lower:
        result["issues"].append("error_html_or_text")
        return result
    if not rows:
        rows = _table_rows(decoded)
        if rows:
            result["format"] = "html_table"
        else:
            try:
                dialect = csv.Sniffer().sniff(decoded[:8192], delimiters=",\t;")
                rows = list(csv.reader(io.StringIO(decoded), dialect))
                result["format"] = "delimited_text"
            except csv.Error:
                result["issues"].append("unrecognized_or_binary_format")
                return result
    if len(rows) < 2:
        result["issues"].append("no_data_rows_or_partial_output")
        return result
    columns = _columns(rows[0])
    paired = {"quantity_geun", "quantity_kg", "price_geun", "price_kg"}
    required = {"date", "product"}
    if not required.issubset(columns) or not (paired.issubset(columns) or {"weight", "price"}.issubset(columns)):
        result["issues"].append("unrecognized_table_schema")
        return result
    result["table_header_found"] = True
    lo, hi = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    for index, cells in enumerate(rows[1:], 1):
        if not any(cell.strip() for cell in cells):
            continue
        record = {"source_row": index + 1, "source_values": cells, "issues": []}
        for field, col in columns.items():
            record[field] = cells[col].strip() if col < len(cells) else None
            if col >= len(cells):
                record["issues"].append(f"missing_{field}_cell")
        date_text = record.get("date") or ""
        parsed = None
        for fmt in ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%Y%m%d", "%Y년 %m월 %d일"):
            try:
                parsed = dt.datetime.strptime(date_text.strip(), fmt).date()
                break
            except ValueError:
                continue
        if parsed is None:
            record["issues"].append("invalid_date")
        elif not lo <= parsed <= hi:
            record["issues"].append("date_outside_requested_week")
        numeric_fields = ("quantity_geun", "quantity_kg", "price_geun", "price_kg") if paired.issubset(columns) else ("weight", "price")
        for field in numeric_fields:
            if record.get(field) is not None:
                issue = _numeric_issue(record[field])
                if issue:
                    record["issues"].append(f"{field}_{issue}")
        if record.get("product") is None:
            record["issues"].append("product_column_unavailable")
        if not paired.issubset(columns) and (record.get("weight_unit") is None or record.get("price_unit") is None):
            record["issues"].append("unit_column_unavailable")
        result["rows"].append(record)
    if not result["rows"]:
        result["issues"].append("no_data_rows_or_partial_output")
    if result["issues"] or any(row["issues"] for row in result["rows"]):
        result["issues"].append("questionable_or_incomplete_rows_present")
    # Partiality/completeness cannot be established by a subset of returned rows.
    result["completeness"] = "unverified_subset_or_full_response"
    result["first_publication_time"] = "unknown"
    return result


def _atomic_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CollectionError(f"immutable artifact already exists: {path.name}")
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
        temp.rename(path)
    finally:
        if temp.exists():
            temp.unlink()


def _safe_headers(headers: Any) -> dict[str, Any]:
    return {key: ("[redacted]" if key.lower() in {"set-cookie", "authorization"} else value)
            for key, value in dict(headers or {}).items()}


def _capture_stamp(now: Any) -> tuple[str, str]:
    captured_at = now().astimezone(dt.timezone.utc).isoformat()
    filename_token = captured_at.replace(":", "").replace("+00:00", "Z") + "_" + uuid.uuid4().hex[:8]
    return captured_at, filename_token


def _save_attempt(output: Path, stem: str, *, request: dict[str, str], captured: str,
                  response: Any, body: bytes, truncated: bool, error: str | None,
                  kind: str) -> dict[str, Any]:
    headers = _safe_headers(response.headers if response is not None else {})
    status = response.status_code if response is not None else None
    raw_name = stem + ".raw" if body else None
    digest = hashlib.sha256(body).hexdigest() if body else None
    if body:
        _atomic_immutable(output / raw_name, body)
    metadata = {"schema_version": 1, "kind": kind, "request": request,
                "retrieved_at_utc": captured, "http_status": status,
                "response_headers": headers, "raw_file": raw_name,
                "response_bytes": len(body), "sha256": digest,
                "truncated": truncated, "transport_error": error}
    encoded = (json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    _atomic_immutable(output / (stem + ".manifest.json"), encoded)
    return metadata


def collect(output: Path, *, session: Any = None, sleep=time.sleep,
            now=lambda: dt.datetime.now(dt.timezone.utc)) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    session.verify = True
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    page = None
    page_body = b""
    page_truncated = False
    page_error = None
    try:
        page = session.get(PAGE_URL, timeout=TIMEOUT_SECONDS, stream=True)
        page_body, page_truncated, page_error = _read_bounded(page)
    except requests.RequestException as exc:
        page_error = f"request_error:{type(exc).__name__}:{exc}"
    page_time, page_token = _capture_stamp(now)
    page_meta = _save_attempt(output, "seoan_landing_" + page_token,
                              request={"method": "GET", "url": PAGE_URL}, captured=page_time,
                              response=page, body=page_body, truncated=page_truncated,
                              error=page_error, kind="landing_page")
    page_hash = page_meta["sha256"]
    if page is None or page.status_code != 200 or page_error or page_truncated:
        raise CollectionError(f"landing page capture unusable; see seoan_landing_{page_token}.manifest.json")
    results = []
    last_start = 0.0
    for start, end in WEEKS:
        params = {"searchStartDt": start.replace("-", ""),
                  "searchEndDt": end.replace("-", ""), "searchSpcs": ""}
        attempts = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            delay = MIN_REQUEST_INTERVAL_SECONDS - (time.monotonic() - last_start)
            if delay > 0:
                sleep(delay)
            last_start = time.monotonic()
            response = None
            body = b""
            truncated = False
            error = None
            try:
                response = session.post(urljoin(PAGE_URL, POST_URL), data=params,
                                        timeout=TIMEOUT_SECONDS, stream=True)
                body, truncated, error = _read_bounded(response)
            except (requests.RequestException, CollectionError) as exc:
                error = f"request_error:{type(exc).__name__}:{exc}"
            captured, token = _capture_stamp(now)
            stem = f"seoan_{start}_{end}_attempt{attempt:02d}_{token}"
            attempt_meta = _save_attempt(output, stem, request=params, captured=captured,
                                         response=response, body=body, truncated=truncated,
                                         error=error, kind="export_attempt")
            attempt_meta["manifest_file"] = stem + ".manifest.json"
            attempts.append(attempt_meta)
            if response is not None and response.status_code < 500 and not error and not truncated:
                break
            if attempt < MAX_ATTEMPTS:
                sleep(RETRY_DELAY_SECONDS)
        latest = attempts[-1]
        validation = validate_body((output / latest["raw_file"]).read_bytes(), start, end,
                                   latest["response_headers"].get("Content-Type", "")) if latest.get("raw_file") else {"format": None, "rows": [], "issues": ["empty_response"]}
        manifest = {"schema_version": 1, "source_page": PAGE_URL, "post_url": POST_URL,
                    "request_params": params, "start_date_inclusive": start,
                    "end_date_inclusive": end, "retrieved_at_utc": latest["retrieved_at_utc"],
                    "landing_page_http_status": page.status_code,
                    "landing_page_raw_file": page_meta["raw_file"],
                    "landing_page_manifest_file": "seoan_landing_" + page_token + ".manifest.json",
                    "landing_page_sha256": page_hash, "attempts": attempts,
                    "http_status": latest["http_status"], "response_headers": latest["response_headers"],
                    "raw_file": latest["raw_file"], "response_bytes": latest["response_bytes"],
                    "sha256": latest["sha256"], "transport_error": latest["transport_error"],
                    "truncated": latest["truncated"], "validation": validation,
                    "complete": False,
                    "completeness_note": "A returned file or subset does not establish full-week completeness.",
                    "first_publication_time": "unknown"}
        _, summary_token = _capture_stamp(now)
        summary_name = f"seoan_{start}_{end}_summary_{summary_token}.manifest.json"
        _atomic_immutable(output / summary_name,
                          (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
        results.append(manifest)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/raw/seoan_pepper_pilot"))
    parser.add_argument("--execute", action="store_true", help="send the two fixed HTTP requests")
    args = parser.parse_args(argv)
    plan = [{"start": start, "end": end, "params": {"searchStartDt": start.replace("-", ""),
             "searchEndDt": end.replace("-", ""), "searchSpcs": ""}} for start, end in WEEKS]
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "requests": plan, "request_sent": False}, ensure_ascii=False))
        return 0
    results = collect(args.output)
    print(json.dumps({"mode": "execute", "results": results}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
