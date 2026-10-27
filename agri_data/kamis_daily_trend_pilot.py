"""Resumable collector for public KAMIS daily trend pages.

The public KAMIS page loads daily.do?action=list with a date and filter query.
This pilot preserves each response and records retrieval time;
it does not infer when the information was first published.
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta, timezone, datetime
from html.parser import HTMLParser
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib.parse import urlsplit
import uuid

import requests

URL = "https://www.kamis.or.kr/customer/trend/trade/daily.do"
LIST_URL = URL + "?action=list"
TARGET_CROPS = {"배추", "무", "양파", "깐마늘", "건고추"}
MAX_ATTEMPTS = 2
TIMEOUT = (5, 20)
MAX_BYTES = 5_000_000
MIN_INTERVAL = 1.0
RETRY_DELAY = 1.0
DEFAULT_OUTPUT = Path("data/raw/kamis_daily_trend_pilot")


def first_third_tuesdays(start: str = "2025-01", end: str = "2026-09") -> list[str]:
    """Generate first and third Tuesdays for inclusive YYYY-MM bounds."""
    sy, sm = map(int, start.split("-")); ey, em = map(int, end.split("-"))
    lo, hi = sy * 12 + sm - 1, ey * 12 + em - 1
    if sm not in range(1, 13) or em not in range(1, 13) or lo > hi:
        raise ValueError("expected an ordered YYYY-MM range")
    out = []
    for serial in range(lo, hi + 1):
        y, m = divmod(serial, 12); m += 1
        first = date(y, m, 1)
        first_tuesday = first + timedelta(days=(1 - first.weekday()) % 7)
        out.extend([(first_tuesday).isoformat(), (first_tuesday + timedelta(days=14)).isoformat()])
    return out


class _PageParser(HTMLParser):
    """Extract paired KAMIS .flex_tit / .txt_box visible content."""
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.records: list[dict[str, Any]] = []
        self.current: dict[str, Any] | None = None
        self.capture: str | None = None
        self.capture_depth = 0
        self.parts: list[str] = []
        # action=list returns the contents loaded into #area_list, often without
        # the containing div. Start in fragment mode and restrict parsing only
        # when a full-page area_list wrapper is present.
        self.in_area = True
        self.restrict_to_area = False
        self.area_depth = 0
        self.card_markers = 0
        self.text_markers = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "div" and attrs.get("id") == "area_list":
            self.in_area = True; self.restrict_to_area = True; self.area_depth = 1; return
        if not self.in_area: return
        classes = attrs.get("class", "").split()
        if tag == "div" and "flex_tit" in classes:
            self.card_markers += 1
            if self.capture is None:
                self.current = {"recordType": "", "reporter": "", "heading": "", "text": ""}
        if tag == "pre" and "txt_box" in classes: self.text_markers += 1
        # Nested inline markup inside captured text must stay part of the text.
        if self.capture is None and tag == "span" and self.current is not None and "state" in classes:
            self.capture = "recordType"; self.capture_depth = 1; self.parts = []
        elif self.capture is None and tag == "h3" and self.current is not None and "lg_tit" in classes:
            self.capture = "heading"; self.capture_depth = 1; self.parts = []
        elif self.capture is None and tag == "span" and self.current is not None and self.current["heading"]:
            # Reporter is the trailing span in the same flex_tit container.
            self.capture = "reporter"; self.capture_depth = 1; self.parts = []
        elif self.capture is None and tag == "pre" and self.current is not None and "txt_box" in classes:
            self.capture = "text"; self.capture_depth = 1; self.parts = []
        elif self.capture == "text" and tag == "br": self.parts.append("\n")
        if self.restrict_to_area and tag in {"div", "pre"} and not (tag == "pre" and self.capture == "text"):
            self.area_depth += 1

    def handle_endtag(self, tag):
        if not self.in_area: return
        if self.capture and ((self.capture in {"recordType", "reporter"} and tag == "span") or
                             (self.capture == "heading" and tag == "h3") or
                             (self.capture == "text" and tag == "pre")):
            value = "".join(self.parts).replace("\xa0", " ").strip()
            self.current[self.capture] = value
            self.capture = None; self.parts = []
            if tag == "pre" and self.current:
                self.records.append(self.current); self.current = None
        if self.restrict_to_area and tag == "div":
            self.area_depth -= 1
            if self.area_depth <= 0: self.in_area = False

    def handle_data(self, data):
        if self.capture: self.parts.append(data)


class _NoDataMarkerParser(HTMLParser):
    """Recognize the exact visible KAMIS no-results heading, excluding comments."""
    VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    def __init__(self, require_area: bool = False):
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.tags: list[str] = []
        self.parts: list[str] = []
        self.found = False
        self.malformed = False
        self.require_area = require_area
        self.in_area = not require_area
        self.area_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if self.require_area and not self.in_area:
            if tag == "div" and attrs.get("id") == "area_list":
                self.in_area = True; self.area_depth = 1
            return
        if self.require_area and tag == "div": self.area_depth += 1
        if not self.in_area: return
        if self.tags:
            if tag not in self.VOID_TAGS: self.tags.append(tag)
            self.depth = len(self.tags)
        elif tag == "h3" and set(attrs.get("class", "").split()) == {"s_tit", "fl"}:
            self.tags = ["h3"]
            self.depth = 1
            self.parts = []

    def handle_endtag(self, tag):
        if self.in_area and self.tags:
            if self.tags[-1] == tag:
                self.tags.pop()
                self.depth = len(self.tags)
                if not self.tags and not self.malformed and "".join(self.parts).strip() == "검색된 데이터가 없습니다.":
                    self.found = True
            else:
                self.malformed = True
        if self.require_area and self.in_area and tag == "div":
            self.area_depth -= 1
            if self.area_depth <= 0: self.in_area = False

    def handle_data(self, data):
        if self.in_area and self.depth: self.parts.append(data)


def parse_html(body: bytes, requested_date: str, *, http_status: int = 200,
               truncated: bool = False, transport_error: str | None = None) -> dict[str, Any]:
    out = {"state": "", "requestedDate": requested_date, "recordCount": 0, "records": [], "issues": []}
    if transport_error:
        out.update(state="transport_error"); out["issues"].append(transport_error); return out
    if truncated:
        out.update(state="truncated"); out["issues"].append("response_size_limit"); return out
    if not 200 <= http_status < 300:
        out.update(state="http_error"); out["issues"].append(f"http_status_{http_status}"); return out
    if not body:
        out["state"] = "empty_response"; return out
    try: text = body.decode("utf-8-sig")
    except UnicodeDecodeError:
        out.update(state="invalid_encoding"); out["issues"].append("response_not_utf8"); return out
    has_area = "id=\"area_list\"" in text or "id='area_list'" in text
    no_data_parser = _NoDataMarkerParser(require_area=has_area); no_data_parser.feed(text)
    has_card_markers = "flex_tit" in text or "txt_box" in text
    if not has_area and not has_card_markers and not no_data_parser.found:
        out.update(state="invalid_html"); out["issues"].append("area_list_missing"); return out
    parser = _PageParser(); parser.feed(text)
    wanted = date.fromisoformat(requested_date)
    records = []
    for record in parser.records:
        # Heading schema is category / item / variety, date, region.
        before, sep, after = record["heading"].partition(",")
        components = [x.strip() for x in before.split("/")]
        date_text, comma, region = after.partition(",")
        try: observed = date.fromisoformat(date_text.strip())
        except ValueError: observed = None
        crop = components[1] if len(components) > 1 else ""
        variety = components[2] if len(components) > 2 else ""
        issues = []
        if not sep or not comma or len(components) != 3: issues.append("heading_structure_questionable")
        if observed is None: issues.append("heading_date_invalid")
        elif observed != wanted: issues.append("heading_date_mismatch")
        if crop == "마늘": issues.append("generic_garlic_ambiguous")
        record.update({"category": components[0] if components else "", "crop": crop,
                       "variety": variety, "date": date_text.strip(), "region": region.strip(),
                       "isTargetCrop": crop in TARGET_CROPS, "issues": issues})
        records.append(record)
    out["recordCount"] = len(records); out["records"] = records
    if not records:
        if no_data_parser.found and parser.card_markers == 0 and parser.text_markers == 0:
            out["state"] = "normal_empty"
            out["issues"].append("explicit_visible_no_data_marker")
        else:
            out["state"] = "questionable"
            out["issues"].append("no_parseable_records; empty result not verified")
    elif parser.card_markers != len(records) or parser.text_markers != len(records):
        out["state"] = "questionable"
        out["issues"].append("unparsed_record_markers")
    elif any(issue in {"heading_structure_questionable", "heading_date_invalid", "heading_date_mismatch"}
             for row in records for issue in row["issues"]):
        out["state"] = "questionable"
        out["issues"].append("record_heading_questionable")
    else:
        out["state"] = "valid"
    if no_data_parser.found and (parser.card_markers or parser.text_markers):
        out["state"] = "questionable"
        out["issues"].append("conflicting_no_data_and_record_markers")
    if any("heading_date_mismatch" in r["issues"] for r in records):
        out["state"] = "questionable"; out["issues"].append("record_date_mismatch")
    return out


def _sha(body: bytes) -> str: return hashlib.sha256(body).hexdigest()


def revalidate_output(output: Path, *, now=lambda: datetime.now(timezone.utc)) -> dict[str, Any]:
    """Reparse saved raw responses into a new report without altering source artifacts."""
    output = Path(output)
    checkpoint = json.loads((output / "checkpoint.json").read_text("utf-8"))
    attempts_out = []
    by_date: dict[str, list[dict[str, Any]]] = {}
    for day, entry in sorted(checkpoint.get("requests", {}).items()):
        for ordinal, attempt in enumerate(entry.get("attempts", []), start=1):
            manifest_name = attempt.get("manifestFile", "")
            raw_name = attempt.get("rawFile", "")
            if (not manifest_name or Path(manifest_name).name != manifest_name or
                    not raw_name or Path(raw_name).name != raw_name):
                raise ValueError(f"unsafe/missing artifact name for {day}")
            manifest_path = output / manifest_name
            raw_path = output / raw_name
            if manifest_path.resolve().parent != output.resolve() or raw_path.resolve().parent != output.resolve():
                raise ValueError(f"artifact path escapes output directory for {day}")
            manifest = json.loads(manifest_path.read_text("utf-8"))
            raw = raw_path.read_bytes()
            digest = _sha(raw)
            expected_sha = manifest.get("responseSha256")
            expected_size = manifest.get("responseBytes")
            verified = (digest == expected_sha == attempt.get("sha256") and
                        isinstance(expected_size, int) and len(raw) == expected_size and
                        manifest.get("rawFile") == raw_name and manifest.get("date") == day and
                        manifest.get("attempt") == ordinal)
            parsed = parse_html(raw, day, http_status=manifest.get("httpStatus") or 0,
                                truncated=manifest.get("truncated", False),
                                transport_error=manifest.get("transportError"))
            issues = list(parsed["issues"])
            if not verified: issues.append("raw_provenance_hash_length_or_manifest_mismatch")
            item = {"date": day, "attempt": ordinal, "manifestFile": manifest_name,
                    "rawFile": raw_name, "oldState": manifest.get("state", attempt.get("state")),
                    "parsedState": parsed["state"],
                    "newState": parsed["state"] if verified else "questionable",
                    "responseBytes": len(raw), "responseSha256": digest,
                    "evidenceVerified": verified, "recordCount": parsed["recordCount"], "issues": issues}
            attempts_out.append(item)
            by_date.setdefault(day, []).append(item)
    date_summaries = []
    for day, attempts in sorted(by_date.items()):
        date_summaries.append({"date": day, "attemptCount": len(attempts),
                               "latestState": attempts[-1]["newState"],
                               "allAttemptsEvidenceVerified": all(x["evidenceVerified"] for x in attempts)})
    generated = now().astimezone(timezone.utc)
    report = {"schemaVersion": 1, "generatedAtUtc": generated.isoformat(),
              "sourceDirectory": str(output), "dateCount": len(date_summaries),
              "attemptCount": len(attempts_out), "dates": date_summaries,
              "attempts": attempts_out, "sourceArtifactsModified": False}
    report_name = "kamis_revalidation_" + generated.strftime("%Y%m%dT%H%M%S%fZ") + ".json"
    _save_immutable(output / report_name,
                    (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
    report["reportFile"] = report_name
    return report

def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2); f.write("\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name): os.unlink(name)

def _save_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f: f.write(data)

def _read_bounded(response) -> tuple[bytes, bool, str | None]:
    chunks = []; size = 0
    try:
        for chunk in response.iter_content(65536):
            if not chunk: continue
            if size + len(chunk) > MAX_BYTES:
                chunks.append(chunk[:MAX_BYTES-size]); return b"".join(chunks), True, None
            chunks.append(chunk); size += len(chunk)
    except requests.RequestException as exc:
        return b"".join(chunks), False, f"response_stream_error:{type(exc).__name__}"
    return b"".join(chunks), False, None

def collect(dates: list[str], output: Path, *, session=None, now=lambda: datetime.now(timezone.utc),
            sleep=time.sleep) -> dict[str, Any]:
    for day in dates: date.fromisoformat(day)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text("utf-8")) if checkpoint_path.exists() else {"schemaVersion": 1, "requests": {}}
    session = session or requests.Session(); session.verify = True
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; agri-predict research collector)",
                            "Referer": URL})
    last_start = None
    for day in dates:
        entry = checkpoint["requests"].get(day)
        if entry and entry.get("state") in {"rate_limited", "retry_after_pause", "redirect_blocked"}: break
        if entry and entry.get("state") in {"valid", "questionable", "normal_empty"}: continue
        attempts = list(entry.get("attempts", [])) if entry else []
        for attempt in range(len(attempts) + 1, MAX_ATTEMPTS + 1):
            if last_start is not None:
                wait = MIN_INTERVAL - (time.monotonic() - last_start)
                if wait > 0: sleep(wait)
            last_start = time.monotonic(); response = None; body = b""; truncated = False; error = None
            payload = {"regday": date.fromisoformat(day).strftime("%Y.%m.%d"),
                       "itemcategorycode": "", "itemcode": "", "kindcode": "", "period": "1",
                       "productclscode": "", "countycode": ""}
            request_url = requests.Request("GET", LIST_URL, params=payload).prepare().url
            try:
                response = session.get(LIST_URL, params=payload, timeout=TIMEOUT, verify=True,
                                       stream=True, allow_redirects=False)
                try:
                    body, truncated, stream_error = _read_bounded(response)
                    if stream_error: error = stream_error
                finally:
                    response.close()
            except requests.RequestException as exc: error = f"request_error:{type(exc).__name__}"
            status = response.status_code if response is not None else 0
            final_url = getattr(response, "url", request_url) if response is not None else request_url
            final_parts = urlsplit(final_url)
            request_parts = urlsplit(URL)
            unsafe_url = (final_parts.scheme != "https" or final_parts.hostname != request_parts.hostname)
            headers = ({k: ("[redacted]" if k.lower() in {"set-cookie", "authorization", "proxy-authorization"} else v)
                        for k, v in dict(response.headers).items()} if response is not None else {})
            has_retry_after = any(k.lower() == "retry-after" for k in headers)
            fetched = now().astimezone(timezone.utc).isoformat()
            parsed = parse_html(body, day, http_status=status, truncated=truncated, transport_error=error)
            if status == 429:
                parsed.update(state="rate_limited")
                parsed["issues"].append("http_429; collection stopped")
            elif has_retry_after:
                parsed.update(state="retry_after_pause")
                parsed["issues"].append("retry_after_header; collection paused")
            elif unsafe_url:
                parsed.update(state="redirect_blocked")
                parsed["issues"].append("response_url_not_same_origin_https")
            elif 300 <= status < 400:
                parsed.update(state="redirect_blocked")
                parsed["issues"].append("redirect_response_not_followed")
            stem = f"{day}_{attempt}_{uuid.uuid4().hex[:8]}"
            raw_name = stem + ".html"; _save_immutable(output / raw_name, body)
            digest = _sha(body)
            manifest = {"schemaVersion": 1, "date": day, "endpoint": LIST_URL, "method": "GET",
                        "requestUrl": final_url, "requestParams": payload,
                        "retrievedAtUtc": fetched, "attempt": attempt, "httpStatus": status or None,
                        "responseHeaders": headers, "rawFile": raw_name, "responseBytes": len(body),
                        "responseSha256": digest, "state": parsed["state"], "truncated": truncated,
                        "transportError": error, "issues": parsed["issues"], "records": parsed["records"]}
            manifest_name = stem + ".json"; _save_immutable(output / manifest_name, (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2)+"\n").encode())
            attempts.append({"manifestFile": manifest_name, "rawFile": raw_name, "sha256": digest,
                             "httpStatus": status or None, "state": parsed["state"], "retrievedAtUtc": fetched})
            checkpoint["requests"][day] = {"state": parsed["state"], "attempts": attempts}
            _atomic_json(checkpoint_path, checkpoint)
            if parsed["state"] in {"valid", "questionable", "normal_empty", "rate_limited", "retry_after_pause", "redirect_blocked"}: break
            if parsed["state"] in {"rate_limited", "retry_after_pause", "redirect_blocked"}: break
            if attempt < MAX_ATTEMPTS: sleep(RETRY_DELAY)
        if checkpoint["requests"][day]["state"] in {"rate_limited", "retry_after_pause", "redirect_blocked"}: break
    return checkpoint

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", action="append", help="explicit YYYY-MM-DD date; may be repeated")
    parser.add_argument("--first-third-tuesday", action="store_true", help="use the 2025-01 through 2026-09 pilot dates")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true", help="perform public page requests (default is dry-run)")
    parser.add_argument("--revalidate-output", type=Path,
                        help="reparse saved raw responses in this directory without network access")
    args = parser.parse_args(argv)
    if args.revalidate_output is not None:
        if args.execute or args.date or args.first_third_tuesday:
            parser.error("--revalidate-output cannot be combined with collection options")
        result = revalidate_output(args.revalidate_output)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)); return 0
    if args.date and args.first_third_tuesday: parser.error("choose --date or --first-third-tuesday")
    dates = args.date or (first_third_tuesdays() if args.first_third_tuesday else [])
    report = {"endpoint": LIST_URL, "dates": dates, "requestCount": len(dates), "requestSent": False,
              "publicationTime": "unknown; retrievedAtUtc is not first publication time"}
    if args.execute:
        if not dates: parser.error("supply --date or --first-third-tuesday")
        checkpoint = collect(dates, args.output)
        report["requestSent"] = True; report["states"] = {k: v["state"] for k, v in checkpoint["requests"].items() if k in dates}
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)); return 0

if __name__ == "__main__": raise SystemExit(main())
