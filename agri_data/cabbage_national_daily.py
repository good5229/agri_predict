"""Capture and issue the prospective daily cabbage national-level challenger.

The public-page GET is bounded, TLS-verified, and injectable for offline tests.
Captured response bytes and a manifest are written before deciding whether the
day has sufficient inputs to issue a challenger.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import email.utils
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable

from . import cabbage_national_prospective as base

KST = base.KST
PAGE_BASE_URL = "https://www.nongnet.or.kr/front/M000000259/marketInfo/nationWide.do"
TIMEOUT_SECONDS = 15
MAX_RESPONSE_BYTES = 10 * 1024 * 1024
MAX_ATTEMPTS = 3
RETRY_DELAYS_SECONDS = (0.25, 0.5)
MAX_RETRY_AFTER_SECONDS = 5.0
SELECTION = {
    "market": "*전국도매시장", "product": "배추", "productCode": "1001",
    "marketCode": "100000", "variety": "전체", "priceType": "평균",
    "unit": "원/kg",
}


class DailyCabbageError(ValueError):
    """Raised when prospective daily inputs cannot support issuance."""


class ResponseTooLarge(DailyCabbageError):
    def __init__(self, prefix: bytes, status: int | None = None, headers: dict[str, str] | None = None):
        super().__init__("HTTP response exceeded the configured byte limit")
        self.prefix = prefix
        self.status = status
        self.headers = headers or {}


def _clock_now(clock: Callable[[], dt.datetime] | None) -> dt.datetime:
    return (clock or (lambda: dt.datetime.now(KST)))()


def _date(value: str) -> dt.date:
    return base._date(value, "issue date")


def issue_date_now(clock: Callable[[], dt.datetime] | None = None) -> dt.date:
    stamp = _clock_now(clock)
    return stamp.astimezone(KST).date()


def build_page_url(selected_date: dt.date) -> str:
    date_label = f"{selected_date.year}년 {selected_date.month:02d}월 {selected_date.day:02d}일"
    query = urllib.parse.urlencode({
        "searchDate": date_label,
        "searchName1": SELECTION["market"],
        "searchName2": SELECTION["product"],
        "searchSymbol1": SELECTION["marketCode"],
        "searchSymbol2": SELECTION["productCode"],
    })
    return f"{PAGE_BASE_URL}?{query}"


class _SelectionParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.values: dict[str, list[str]] = {}
        self.current_form: str | None = None
        self.forms_seen: set[str] = set()
        self.select_name: str | None = None
        self.option_selected = False
        self.option_value = ""
        self.option_text: list[str] = []
        self.visible_class: str | None = None
        self.visible_text: list[str] = []
        self.class_text: dict[str, list[str]] = {}
        self.all_text: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag.lower() == "form":
            self.current_form = attrs.get("name")
            if self.current_form:
                self.forms_seen.add(str(self.current_form))
        elif tag.lower() == "select":
            self.select_name = attrs.get("name") if self.current_form == "actForm" else None
        elif tag.lower() == "option" and self.select_name:
            self.option_selected = "selected" in attrs
            self.option_value = str(attrs.get("value", ""))
            self.option_text = []
        elif tag.lower() == "input" and attrs.get("name") and self.current_form == "actForm":
            self.values.setdefault(str(attrs["name"]), []).append(str(attrs.get("value", "")))
        if attrs.get("class") and tag.lower() in {"p", "span", "strong"}:
            self.visible_class = str(attrs["class"])
            self.visible_text = []

    def handle_data(self, data):
        self.all_text.append(data)
        if self.select_name and self.option_text is not None:
            self.option_text.append(data)
        if self.visible_class:
            self.visible_text.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "option" and self.select_name:
            if self.option_selected:
                self.values.setdefault(self.select_name, []).extend(
                    (self.option_value, " ".join(self.option_text).strip()))
            self.option_selected = False
            self.option_value = ""
            self.option_text = []
        elif tag.lower() == "select":
            self.select_name = None
            self.option_text = []
        elif tag.lower() == "form":
            self.current_form = None
        if self.visible_class and tag.lower() in {"p", "span", "strong"}:
            self.class_text.setdefault(self.visible_class, []).append(" ".join(self.visible_text).strip())
            self.visible_class = None
            self.visible_text = []


def _response_selection(html: str, selected_date: dt.date) -> dict[str, str]:
    parser = _SelectionParser()
    parser.feed(html)
    missing = []
    if "actForm" not in parser.forms_seen:
        missing.append("actForm")
    def has_exact(control: str, expected: str) -> bool:
        return expected in ["".join(value.split()) for value in parser.values.get(control, [])]
    if not has_exact("searchSymbol1", "100000"):
        missing.append("marketCode")
    if not has_exact("searchName1", "*전국도매시장"):
        missing.append("marketName")
    if not has_exact("searchSymbol2", "1001"):
        missing.append("productCode")
    if not has_exact("searchName2", "배추"):
        missing.append("productName")
    if "searchName3" not in parser.values or any(value.strip() for value in parser.values["searchName3"]):
        missing.append("variety_all_unselected")
    date_forms = {
        selected_date.isoformat(), selected_date.strftime("%Y%m%d"),
        f"{selected_date.year}년{selected_date.month:02d}월{selected_date.day:02d}일",
        f"{selected_date.year}년{selected_date.month}월{selected_date.day:02d}일",
    }
    # The Nongnet form control is named `searchDate`; `pickerDate` is its
    # browser-facing id, not its submitted name.
    date_values = parser.values.get("searchDate", [])
    if not any("".join(value.split()) in date_forms for value in date_values):
        missing.append("selectedDate")
    if missing:
        raise DailyCabbageError("response selection is absent or mismatched: " + ",".join(missing))
    return {"market": SELECTION["market"], "marketCode": SELECTION["marketCode"],
            "product": SELECTION["product"], "productCode": SELECTION["productCode"],
            "variety": SELECTION["variety"], "priceType": SELECTION["priceType"],
            "unit": SELECTION["unit"], "unitEvidence": "positive mean price card + 1kg",
            "varietyEvidence": "searchName3 is empty (page default: all varieties)",
            "selectedDate": selected_date.isoformat()}


def _validate_mean_price_card(html: str) -> None:
    parser = _SelectionParser()
    parser.feed(html)
    # Nongnet shows several cards (고/중/저/평균); only the 평균 card is relevant.
    grade_cards = [" ".join(value.split()) for value in parser.class_text.get("grade", [])]
    if "평균" not in grade_cards:
        raise DailyCabbageError("response has no mean (평균) price card")
    all_text = "".join(" ".join(parser.all_text).split()).lower()
    if "1kg" not in all_text and "1㎏" not in all_text:
        raise DailyCabbageError("mean price card has no 1kg unit evidence")


def _safe_response_headers(headers: dict[str, Any]) -> dict[str, str]:
    allowed = {"content-type", "content-length", "last-modified", "etag", "date", "cache-control", "retry-after"}
    return {str(k): str(v) for k, v in headers.items() if str(k).lower() in allowed}


def _retry_after_seconds(headers: dict[str, Any]) -> float | None:
    value = next((str(v).strip() for k, v in headers.items()
                  if str(k).lower() == "retry-after"), None)
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            parsed = email.utils.parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return max(0.0, (parsed - dt.datetime.now(dt.timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _default_transport(url: str, timeout: float, limit: int) -> tuple[int, bytes, dict[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": "agri-predict-prospective/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=ssl.create_default_context()) as response:
            body = response.read(limit + 1)
            if len(body) > limit:
                raise ResponseTooLarge(body[:limit], int(response.status), dict(response.headers.items()))
            return int(response.status), body, dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        body = exc.read(limit + 1)
        if len(body) > limit:
            raise ResponseTooLarge(body[:limit], int(exc.code), dict(exc.headers.items()) if exc.headers else {}) from exc
        return int(exc.code), body, dict(exc.headers.items()) if exc.headers else {}


def _request_page(url: str, *, transport: Callable[[str, float, int], tuple[int, bytes, dict]],
                  sleeper: Callable[[float], None]) -> tuple[int, bytes, dict[str, str], int, str | None, list[dict[str, Any]]]:
    last_error = None
    attempt_records: list[dict[str, Any]] = []
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            status, body, headers = transport(url, TIMEOUT_SECONDS, MAX_RESPONSE_BYTES)
            if not isinstance(body, bytes):
                raise DailyCabbageError("HTTP transport returned non-byte body")
            if len(body) > MAX_RESPONSE_BYTES:
                body = body[:MAX_RESPONSE_BYTES]
                attempt_records.append({"attempt": attempt, "status": status,
                    "attemptedAtKst": dt.datetime.now(KST).isoformat(timespec="microseconds"),
                    "headers": _safe_response_headers(headers), "body": body, "bodyComplete": False})
                return 599, body, headers, attempt, "HTTP response exceeded the configured byte limit", attempt_records
            attempt_records.append({"attempt": attempt, "status": status,
                "attemptedAtKst": dt.datetime.now(KST).isoformat(timespec="microseconds"),
                "headers": _safe_response_headers(headers), "body": body, "bodyComplete": True})
            if (status == 429 or 500 <= status < 600) and attempt < MAX_ATTEMPTS:
                retry_after = _retry_after_seconds(headers)
                if retry_after is not None and retry_after > MAX_RETRY_AFTER_SECONDS:
                    return status, body, headers, attempt, f"retry_deferred_after_seconds:{retry_after:.3f}", attempt_records
                sleeper(retry_after if retry_after is not None else RETRY_DELAYS_SECONDS[attempt - 1])
                continue
            return status, body, headers, attempt, None, attempt_records
        except ResponseTooLarge as exc:
            attempt_records.append({"attempt": attempt, "status": exc.status or 599,
                "attemptedAtKst": dt.datetime.now(KST).isoformat(timespec="microseconds"),
                "headers": _safe_response_headers(exc.headers), "body": exc.prefix, "bodyComplete": False})
            return exc.status or 599, exc.prefix, exc.headers, attempt, str(exc), attempt_records
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = type(exc).__name__
            attempt_records.append({"attempt": attempt, "status": None,
                "attemptedAtKst": dt.datetime.now(KST).isoformat(timespec="microseconds"),
                "headers": {}, "body": b"", "bodyComplete": True,
                "transportErrorType": last_error})
            if attempt < MAX_ATTEMPTS:
                sleeper(RETRY_DELAYS_SECONDS[attempt - 1])
    return (0, b"", {}, MAX_ATTEMPTS,
            f"transport_failed_after_{MAX_ATTEMPTS}_attempts:{last_error or 'unknown'}", attempt_records)


def _write_capture(path: Path, data: bytes) -> None:
    try:
        base._write_immutable(path, data)
    except base.ProspectiveCabbageError as exc:
        raise DailyCabbageError(str(exc)) from exc


def select_earliest_valid_capture(root: str | Path, issue_date: dt.date) -> dict[str, Any] | None:
    """Reuse the first validated complete capture on a same-day rerun."""
    project = Path(root).resolve()
    directory = project / "outputs" / "prospective-cabbage" / "daily" / issue_date.isoformat() / "captures"
    if not directory.is_dir():
        return None
    selected_date = issue_date - dt.timedelta(days=1)
    valid = []
    for manifest_path in directory.glob("national-*.json"):
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            raw_ref = manifest.get("rawResponse")
            if manifest.get("captureState") != "validated_price_observations" or not isinstance(raw_ref, dict):
                continue
            raw_rel = Path(raw_ref.get("path", ""))
            capture_dir = directory.resolve()
            if (raw_rel.is_absolute() or ".." in raw_rel.parts or raw_rel.suffix != ".html"):
                continue
            raw_path = (project / raw_rel).resolve()
            if raw_path.parent != capture_dir:
                continue
            raw_bytes = raw_path.read_bytes()
            if base._sha256(raw_bytes) != raw_ref.get("sha256"):
                continue
            if manifest.get("issueDateKst") != issue_date.isoformat() or manifest.get("selectedDate") != selected_date.isoformat():
                continue
            if manifest.get("requestUrl") != build_page_url(selected_date) or manifest.get("httpStatus") != 200:
                continue
            html = raw_bytes.decode("utf-8", errors="strict")
            selection = _response_selection(html, selected_date)
            _validate_mean_price_card(html)
            prices = base.parse_nongnet_mean_series(html)
            if (selected_date.isoformat() not in prices or
                    any(dt.date.fromisoformat(day) > selected_date for day in prices)):
                continue
            captured = base._parse_kst(manifest.get("capturedAtKst"), "capturedAtKst")
            if (captured.date() != issue_date
                    or not dt.time(10, 0) <= captured.time().replace(tzinfo=None) <= dt.time(12, 0)):
                continue
            valid.append((captured, manifest_path, manifest, prices, raw_ref, selection))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError,
                base.ProspectiveCabbageError, DailyCabbageError, TypeError):
            continue
    if not valid:
        return None
    _, manifest_path, manifest, prices, raw_ref, selection = min(valid, key=lambda row: (row[0], str(row[1])))
    return {"state": "validated_price_observations", "reason": None,
            "manifestPath": str(manifest_path.relative_to(project)),
            "manifestSha256": base._sha256(manifest_path.read_bytes()),
            "rawPath": raw_ref["path"], "rawSha256": raw_ref["sha256"],
            "requestUrl": manifest["requestUrl"], "httpStatus": manifest["httpStatus"],
            "capturedAtKst": manifest["capturedAtKst"], "selection": selection,
            "nationalPrices": prices, "reused": True}


def capture_national_page(*, root: str | Path, issue_date: dt.date,
                          transport: Callable[[str, float, int], tuple[int, bytes, dict]] | None = None,
                          clock: Callable[[], dt.datetime] | None = None,
                          sleeper: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Fetch and durably capture one selected prior-day page, including invalid responses."""
    project = Path(root).resolve()
    reused = select_earliest_valid_capture(project, issue_date)
    if reused is not None:
        return reused
    selected_date = issue_date - dt.timedelta(days=1)
    url = build_page_url(selected_date)
    start_at = _clock_now(clock).astimezone(KST)
    status, body, headers, attempts, transport_error, attempt_records = _request_page(
        url, transport=transport or _default_transport, sleeper=sleeper)
    captured_at = _clock_now(clock).astimezone(KST)
    stamp = captured_at.strftime("%Y%m%dT%H%M%S.%f%z") + "-" + uuid.uuid4().hex[:8]
    rel_dir = Path("outputs/prospective-cabbage/daily") / issue_date.isoformat() / "captures"
    raw_rel = rel_dir / f"national-{stamp}.html" if status != 0 else None
    manifest_rel = rel_dir / f"national-{stamp}.json"
    raw_sha = None
    persisted_attempts = []
    for record in attempt_records:
        attempt_body = record.pop("body")
        attempt_rel = rel_dir / f"attempt-{record['attempt']:02d}-{stamp}.bin"
        if record["status"] is not None:
            _write_capture(project / attempt_rel, attempt_body)
            record["bodyPath"] = str(attempt_rel)
            record["bodySha256"] = base._sha256(attempt_body)
            record["bodyByteCount"] = len(attempt_body)
        persisted_attempts.append(record)
    state = "transport_failure"
    reason = transport_error
    response_selection = None
    mean_prices: dict[str, float] | None = None
    if status != 0:
        raw_sha = base._sha256(body)
        _write_capture(project / raw_rel, body)
        if transport_error and transport_error.startswith("retry_deferred_after_seconds:"):
            state, reason = "retry_deferred", transport_error
        elif transport_error:
            state, reason = "response_limit_exceeded", transport_error
        elif status != 200:
            state, reason = "http_failure", f"http_status_{status}"
        elif not body:
            state, reason = "empty_http_body", "successful_http_response_has_zero_bytes"
        else:
            try:
                html = body.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                state, reason = "invalid_response_encoding", "response_is_not_utf8"
            else:
                try:
                    response_selection = _response_selection(html, selected_date)
                except DailyCabbageError as exc:
                    state, reason = "selection_mismatch", str(exc)
                else:
                    try:
                        mean_prices = base.parse_nongnet_mean_series(html)
                    except base.ProspectiveCabbageError as exc:
                        message = str(exc)
                        if ("no mean chart date/value rows" in message or
                                "no makeGradeChart function" in message):
                            state, reason = "http_success_empty_chart", "no_chart_price_rows; not evidence of closure or no-trade"
                        else:
                            state, reason = "chart_validation_failed", message
                    else:
                        if selected_date.isoformat() not in mean_prices:
                            state, reason = "chart_validation_failed", "selected prior-day date missing from chart values"
                        elif any(dt.date.fromisoformat(day) > selected_date for day in mean_prices):
                            state, reason = "chart_validation_failed", "chart includes dates after selected prior day"
                        else:
                            try:
                                _validate_mean_price_card(html)
                            except DailyCabbageError as exc:
                                state, reason = "selection_mismatch", str(exc)
                            else:
                                state, reason = "validated_price_observations", None
    manifest = {
        "schemaVersion": 1, "captureState": state, "reason": reason,
        "issueDateKst": issue_date.isoformat(), "selectedDate": selected_date.isoformat(),
        "requestUrl": url, "httpStatus": status or None, "attemptCount": attempts,
        "requestStartedAtKst": start_at.isoformat(timespec="seconds"),
        "capturedAtKst": captured_at.isoformat(timespec="microseconds"),
        "sameDayIssueWindowEligibleAtCapture": (
            captured_at.date() == issue_date
            and dt.time(10, 0) <= captured_at.time().replace(tzinfo=None) <= dt.time(12, 0)),
        "responseHeaders": _safe_response_headers(headers),
        "httpAttempts": persisted_attempts,
        "bodyByteCount": len(body),
        "bodyComplete": (transport_error is None or
                         transport_error.startswith("retry_deferred_after_seconds:")),
        "rawResponse": ({"path": str(raw_rel), "sha256": raw_sha} if raw_rel else None),
        "requestedSelection": {**SELECTION, "selectedDate": selected_date.isoformat()},
        "selection": response_selection if state == "validated_price_observations" else None,
        "validatedSelection": response_selection if state == "validated_price_observations" else None,
        "meanPricesWonPerKg": mean_prices,
        "interpretation": ("validated chart observations only; first public posting time unknown"
                           if state == "validated_price_observations" else
                           "capture state does not establish market closure, no trade, or any price"),
    }
    manifest_path = project / manifest_rel
    _write_capture(manifest_path, (json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                                               indent=2, allow_nan=False) + "\n").encode("utf-8"))
    manifest_bytes = manifest_path.read_bytes()
    return {"state": state, "reason": reason, "manifestPath": str(manifest_rel),
            "manifestSha256": base._sha256(manifest_bytes),
            "rawPath": str(raw_rel) if raw_rel else None, "rawSha256": raw_sha,
            "requestUrl": url, "httpStatus": status or None,
            "capturedAtKst": captured_at.isoformat(timespec="microseconds"),
            "selection": response_selection, "nationalPrices": mean_prices}


def select_newest_local_snapshot(root: str | Path, issue_date: dt.date, *,
                                 not_after: dt.datetime) -> tuple[Path, bytes, dict[str, Any], dict[str, Any]]:
    """Select newest completed o-1 run whose price context ends exactly at o-2."""
    project = Path(root).resolve()
    if not_after.tzinfo is None or not_after.utcoffset() is None:
        raise DailyCabbageError("snapshot selection cutoff must be timezone-aware")
    as_of = not_after.astimezone(KST)
    snapshot_day = issue_date - dt.timedelta(days=1)
    expected_end = issue_date - dt.timedelta(days=2)
    directory = project / "outputs" / "daily" / snapshot_day.isoformat()
    if not directory.is_dir():
        raise DailyCabbageError(f"no prior-day completed prices snapshot directory: {snapshot_day}")
    candidates = []
    for run_dir in directory.iterdir():
        if not run_dir.is_dir():
            continue
        status_path, prices_path = run_dir / "run-status.json", run_dir / "prices.json"
        try:
            status_bytes = status_path.read_bytes()
            status = json.loads(status_bytes.decode("utf-8"))
            payload = prices_path.read_bytes()
            snapshot = json.loads(payload.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not isinstance(status, dict) or status.get("status") != "complete":
            continue
        if status.get("cutoff") != expected_end.isoformat():
            continue
        meta = snapshot.get("meta") if isinstance(snapshot, dict) else None
        if not isinstance(meta, dict) or meta.get("contextEnd") != expected_end.isoformat():
            continue
        finished = status.get("finishedAtKst")
        try:
            finished_at = base._parse_kst(finished, "run finishedAtKst")
        except base.ProspectiveCabbageError:
            continue
        generated = snapshot.get("generatedAt")
        try:
            generated_at = base._parse_kst(generated, "prices generatedAt")
        except base.ProspectiveCabbageError:
            continue
        if (finished_at.date() != snapshot_day or generated_at.date() != snapshot_day
                or finished_at >= as_of or generated_at >= as_of):
            continue
        safe_status = {"path": str(status_path.relative_to(project)),
                       "sha256": base._sha256(status_bytes), "status": "complete",
                       "cutoff": status["cutoff"], "finishedAtKst": finished,
                       "snapshotSha256": status.get("snapshotSha256")}
        candidates.append((finished_at, run_dir, payload, snapshot, safe_status))
    if not candidates:
        raise DailyCabbageError(f"no completed prior-day snapshot with contextEnd={expected_end.isoformat()}")
    _, run_dir, payload, snapshot, status = max(candidates, key=lambda row: (row[0], str(row[1])))
    try:
        base.local_cabbage_prices(snapshot)
    except base.ProspectiveCabbageError as exc:
        raise DailyCabbageError(f"selected local snapshot is invalid: {exc}") from exc
    return run_dir / "prices.json", payload, snapshot, status


def _skip(reason: str, capture: dict[str, Any] | None, *, root: Path | None = None,
          issue_date: dt.date | None = None, **extra) -> dict[str, Any]:
    result = {"status": "not_issued", "reason": reason, "capture": capture, **extra}
    if root is not None and issue_date is not None:
        record = {"schemaVersion": 1, "status": "not_issued", "reason": reason,
                  "issueDateKst": issue_date.isoformat(),
                  "details": extra,
                  "captureManifestPath": capture.get("manifestPath") if capture else None,
                  "captureManifestSha256": capture.get("manifestSha256") if capture else None}
        path = root / "outputs" / "prospective-cabbage" / "daily" / issue_date.isoformat() / "skips" / f"skip-{uuid.uuid4().hex}.json"
        payload = (json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
        base._write_immutable(path, payload)
        result["skipRecordPath"] = str(path.relative_to(root))
        result["skipRecordSha256"] = base._sha256(payload)
    return result


def issue_daily(*, root: str | Path = ".", issue_date: dt.date | None = None,
                transport: Callable[[str, float, int], tuple[int, bytes, dict]] | None = None,
                clock: Callable[[], dt.datetime] | None = None,
                sleeper: Callable[[float], None] = time.sleep) -> dict[str, Any]:
    """Capture the prior-day truth page, then issue only when all inputs validate."""
    project = Path(root).resolve()
    preflight_at = _clock_now(clock).astimezone(KST)
    day = issue_date or preflight_at.date()
    output = project / "outputs" / "prospective-cabbage" / "daily" / day.isoformat() / "challenger.json"
    if day <= dt.date(2026, 10, 7):
        return _skip("daily_issuance_available_only_after_first_issue_date_2026-10-07",
                     None, root=project, issue_date=day)
    if output.exists():
        existing = output.read_bytes()
        try:
            json.loads(existing.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DailyCabbageError("existing daily challenger is invalid; refusing to replace it") from exc
        return {"status": "already_issued", "artifact": str(output.relative_to(project)),
                "artifactSha256": base._sha256(existing)}
    try:
        base.validate_current_issue_time(day, preflight_at, preflight_at, max_skew_seconds=0)
    except base.ProspectiveCabbageError as exc:
        return _skip(f"issue_window_unavailable:{exc}", None, root=project, issue_date=day)

    # Capture independently of issuance readiness: it is needed later as
    # truth for earlier immutable forecasts.
    capture = capture_national_page(root=project, issue_date=day, transport=transport,
                                    clock=clock, sleeper=sleeper)
    if capture["state"] != "validated_price_observations":
        return _skip(f"national_page_{capture['state']}:{capture.get('reason')}", capture,
                     root=project, issue_date=day)

    try:
        local_path, local_bytes, local_snapshot, local_status = select_newest_local_snapshot(
            project, day, not_after=preflight_at)
    except DailyCabbageError as exc:
        return _skip(f"local_snapshot_unavailable:{exc}", capture, root=project, issue_date=day)
    try:
        baseline_path, baseline_bytes, baseline = base.select_earliest_baseline(project, day)
    except base.ProspectiveCabbageError as exc:
        return _skip(f"baseline_unavailable:{exc}", capture,
                     root=project, issue_date=day,
                     localSnapshotPath=str(local_path.relative_to(project)),
                     localSnapshotSha256=base._sha256(local_bytes))

    national = {"nationalPrices": capture["nationalPrices"], "htmlSha256": capture["rawSha256"],
                "manifestSha256": capture["manifestSha256"], "pageUrl": capture["requestUrl"],
                "selection": capture["selection"], "pageMetadata": {
                    "captureManifestPath": capture["manifestPath"],
                    "captureManifestSha256": capture["manifestSha256"],
                    "capturedAtKst": capture["capturedAtKst"],
                    "httpStatus": capture["httpStatus"],
                }}
    now = _clock_now(clock).astimezone(KST)
    try:
        issued_at = base.validate_current_issue_time(day, now, now, max_skew_seconds=0)
    except base.ProspectiveCabbageError as exc:
        return _skip(f"issue_window_closed_before_publication:{exc}", capture,
                     root=project, issue_date=day,
                     localSnapshotPath=str(local_path.relative_to(project)),
                     localSnapshotSha256=base._sha256(local_bytes),
                     baselinePath=str(baseline_path.relative_to(project)),
                     baselineSha256=base._sha256(baseline_bytes))
    try:
        artifact = base.build_challenger(issue_date=day, issued_at=issued_at,
                                         national_inputs=national, local_snapshot_bytes=local_bytes,
                                         local_snapshot=local_snapshot, baseline=baseline,
                                         baseline_path=str(baseline_path.relative_to(project)),
                                         baseline_sha256=base._sha256(baseline_bytes),
                                         local_snapshot_path=str(local_path.relative_to(project)),
                                         local_expected_sha256=base._sha256(local_bytes))
    except base.ProspectiveCabbageError as exc:
        return _skip(f"challenger_validation_failed:{exc}", capture, root=project, issue_date=day)
    artifact["artifactType"] = "cabbage_national_level_daily_challenger"
    artifact["inputs"]["nationalHtmlPath"] = capture["rawPath"]
    artifact["inputs"]["nationalManifestPath"] = capture["manifestPath"]
    artifact["inputs"]["captureManifestPath"] = capture["manifestPath"]
    artifact["inputs"]["captureManifestSha256"] = capture["manifestSha256"]
    artifact["inputs"]["httpStatus"] = capture["httpStatus"]
    artifact["inputs"]["requestUrl"] = capture["requestUrl"]
    artifact["inputs"]["capturedAtKst"] = capture["capturedAtKst"]
    artifact["inputs"]["localSnapshotRunStatus"] = local_status
    json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    final_at = _clock_now(clock).astimezone(KST)
    try:
        final_at = base.validate_current_issue_time(day, final_at, final_at, max_skew_seconds=0)
    except base.ProspectiveCabbageError as exc:
        return _skip(f"issue_window_closed_before_publication:{exc}", capture,
                     root=project, issue_date=day,
                     localSnapshotPath=str(local_path.relative_to(project)),
                     localSnapshotSha256=base._sha256(local_bytes),
                     baselinePath=str(baseline_path.relative_to(project)),
                     baselineSha256=base._sha256(baseline_bytes))
    baseline_at = base._parse_kst(baseline["issueAtKst"], "baseline issueAtKst")
    if baseline_at > final_at:
        return _skip("baseline_issue_time_follows_challenger_publication", capture,
                     root=project, issue_date=day)
    captured_at = base._parse_kst(capture["capturedAtKst"], "national page capturedAtKst")
    if (captured_at.date() != day or not dt.time(10, 0) <= captured_at.time().replace(tzinfo=None) <= dt.time(12, 0)
            or captured_at > final_at):
        return _skip("national_capture_time_outside_issue_window_or_after_publication", capture,
                     root=project, issue_date=day)
    artifact["issueAtKst"] = final_at.isoformat(timespec="microseconds")
    payload = (json.dumps(artifact, ensure_ascii=False, sort_keys=True,
                          indent=2, allow_nan=False) + "\n").encode("utf-8")
    base._write_immutable(output, payload)
    return {"status": "complete", "artifact": str(output.relative_to(project)),
            "artifactSha256": base._sha256(payload), "capture": capture,
            "levelRatio": artifact["levelRatio"]["ratio"],
            "eligibleAnchorDays": artifact["levelRatio"]["eligiblePairCount"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--issue-date", help="KST date (defaults to the current KST date)")
    args = parser.parse_args(argv)
    try:
        current = dt.datetime.now(KST)
        day = _date(args.issue_date) if args.issue_date else current.date()
        result = issue_daily(root=args.root, issue_date=day,
                             clock=lambda: dt.datetime.now(KST))
    except (DailyCabbageError, base.ProspectiveCabbageError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
