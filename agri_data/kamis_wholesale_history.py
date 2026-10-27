"""Bounded, provenance-preserving probe of the public KAMIS website."""
from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal
import hashlib
import http.client
import json
from pathlib import Path
import ssl
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, HTTPSHandler, Request, build_opener
from agri_data.kamis_wholesale_export import parse_export

DEFAULT_URL = "https://www.kamis.or.kr/customer/price/wholesale/period.do?action=daily"
MAX_BYTES = 5 * 1024 * 1024
TIMEOUT_SECONDS = 20
MAX_RETRIES = 2
MIN_INTERVAL_SECONDS = 1
ROOT = Path(__file__).resolve().parents[1]
POST_URL = "https://www.kamis.or.kr/customer/price/wholesale/period.do"
POST_PATH = "/customer/price/wholesale/period.do"
CROPS = {
    "cabbage": {"item": "배추", "variety": "전체", "unit": "10kg(그물망 3포기)", "itemcode": "211", "kindcode": ""},
    "radish": {"item": "무", "variety": "전체", "unit": "20kg", "itemcode": "231", "kindcode": ""},
    "onion": {"item": "양파", "variety": "양파", "unit": "15kg", "itemcode": "245", "kindcode": ""},
    "garlic": {"item": "깐마늘(국산)", "variety": "전체", "unit": "20kg", "itemcode": "258", "kindcode": ""},
    "pepper": {"item": "건고추", "variety": "전체(화건)", "unit": "30kg", "itemcode": "241", "kindcode": "AA"},
}
PILOT_HEADERS = ["구분", "평균", "등락율", "서울", "부산", "대구", "광주", "대전"]
KOREAN_ACCEPT_LANGUAGE = "ko-KR,ko;q=0.9,en;q=0.8"


def _post_headers(korean_language=False):
    headers = {"Accept": "text/html,application/xhtml+xml,application/javascript,*/*;q=0.8",
               "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
               "Origin": "https://www.kamis.or.kr",
               "Referer": "https://www.kamis.or.kr/customer/price/wholesale/period.do?action=daily",
               "User-Agent": "AgriPredict public-source-collection/1.0"}
    if korean_language: headers["Accept-Language"] = KOREAN_ACCEPT_LANGUAGE
    return headers


def _check_url(url: str) -> None:
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError as exc:
        raise ValueError("invalid URL") from exc
    if (parts.scheme != "https" or parts.hostname != "www.kamis.or.kr" or
            port not in (None, 443) or parts.username or parts.password or parts.fragment):
        raise ValueError("URL must use the exact https://www.kamis.or.kr origin without credentials or fragment")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _opener():
    return build_opener(HTTPSHandler(context=ssl.create_default_context()), _NoRedirect())


def _utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


class _ResponseReadFailure(Exception):
    def __init__(self, body: bytes, oversize: bool, error: Exception):
        super().__init__(type(error).__name__)
        self.body, self.oversize, self.error_class = body, oversize, type(error).__name__


def _read_bounded(response, expected_length=None):
    chunks = bytearray()
    while len(chunks) <= MAX_BYTES:
        try:
            block = response.read(min(65536, MAX_BYTES + 1 - len(chunks)))
        except (URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            partial = getattr(exc, "partial", b"")
            if partial: chunks.extend(partial)
            oversize = len(chunks) > MAX_BYTES
            raise _ResponseReadFailure(bytes(chunks[:MAX_BYTES]), oversize, exc) from exc
        if not block: break
        chunks.extend(block)
    oversize = len(chunks) > MAX_BYTES
    captured = bytes(chunks[:MAX_BYTES])
    premature = expected_length is not None and not oversize and len(captured) != expected_length
    return captured, oversize, premature


def probe(url: str = DEFAULT_URL, *, output_dir: Path | str = ROOT / "data/raw/kamis_web_probe",
          opener=None, now=None, sleep=time.sleep) -> dict:
    """Fetch one same-origin public URL, with bounded retry and create-only checkpoint."""
    _check_url(url)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()
    body_path = output_dir / f"probe-{key}.body"
    meta_path = output_dir / f"probe-{key}.json"
    if meta_path.exists():
        checkpoint = json.loads(meta_path.read_text(encoding="utf-8"))
        for response in checkpoint.get("responses", []):
            saved = (output_dir / response["bodyFile"]).read_bytes()
            if len(saved) != response["bodyBytesCaptured"] or hashlib.sha256(saved).hexdigest() != response["bodySha256"]:
                raise ValueError("saved response checkpoint failed byte-count or SHA-256 verification")
        return {**checkpoint, "checkpoint": "already_recorded"}
    if body_path.exists() or list(output_dir.glob(f"probe-{key}-attempt-*.body")):
        raise FileExistsError(f"incomplete create-only checkpoint exists for {url}")
    client = opener or _opener()
    now = now or _utc_now
    attempts, responses = [], []
    final_state = "transport_failure"
    for attempt in range(MAX_RETRIES + 1):
        if attempt: sleep(MIN_INTERVAL_SECONDS)
        started = now()
        body, headers, status, response_url = b"", {}, None, None
        final_has_response = False
        try:
            response = client.open(Request(url, headers={
                "Accept": "text/html,application/xhtml+xml,application/javascript,*/*;q=0.8",
                "User-Agent": "AgriPredict offline-source-inspection/1.0",
            }, method="GET"), timeout=TIMEOUT_SECONDS)
        except HTTPError as exc:
            response = exc
        except (URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            ended = now()
            attempts.append({"attempt": attempt + 1, "startedAtUtc": started, "endedAtUtc": ended,
                             "outcome": "transport_failure", "errorClass": type(exc).__name__, "httpStatus": None})
            if attempt < MAX_RETRIES: continue
            final_state = "transport_failure"
            break
        final_has_response = True
        status = getattr(response, "status", None) or getattr(response, "code", None)
        response_url = response.geturl() if hasattr(response, "geturl") else url
        headers_obj = getattr(response, "headers", {})
        headers = {str(k): str(v) for k, v in headers_obj.items()
                   if str(k).lower() not in {"set-cookie", "authorization", "proxy-authorization", "www-authenticate"}}
        content_length = headers_obj.get("Content-Length") if hasattr(headers_obj, "get") else None
        expected_length, invalid_length = None, False
        if content_length is not None:
            try:
                expected_length = int(content_length)
                if expected_length < 0: raise ValueError
            except (TypeError, ValueError):
                expected_length, invalid_length = None, True
        read_error_class = None
        try:
            body, oversize, premature = _read_bounded(response, expected_length)
        except _ResponseReadFailure as exc:
            body, oversize, premature = exc.body, exc.oversize, True
            read_error_class = exc.error_class
        finally:
            close = getattr(response, "close", None)
            if close: close()
        if _origin(response_url) != ("https", "www.kamis.or.kr", 443):
            final_state, outcome = "redirect_rejected", "redirect_rejected"
        elif oversize:
            final_state, outcome = "oversize_response", "oversize_response"
        elif status is not None and 200 <= status < 300:
            incomplete = premature or invalid_length or read_error_class is not None
            final_state, outcome = (("incomplete_response", "incomplete_response") if incomplete else ("complete", "http_success"))
        else:
            final_state, outcome = "http_failure", "http_failure"
        ended = now()
        attempt_path = output_dir / f"probe-{key}-attempt-{attempt + 1}.body"
        with attempt_path.open("xb") as f: f.write(body)
        response_record = {"attempt": attempt + 1, "requestUrl": url, "responseUrl": response_url,
                           "startedAtUtc": started, "endedAtUtc": ended, "httpStatus": status,
                           "outcome": outcome, "headers": headers, "bodyFile": attempt_path.name,
                           "bodyBytesCaptured": len(body),
                           "bodyTruncated": oversize or premature or invalid_length or read_error_class is not None,
                           "bodySha256": hashlib.sha256(body).hexdigest(),
                           "advertisedContentLength": content_length,
                           "framingComplete": not (oversize or premature or invalid_length or read_error_class is not None),
                           "readErrorClass": read_error_class}
        responses.append(response_record)
        attempts.append({"attempt": attempt + 1, "startedAtUtc": started, "endedAtUtc": ended,
                         "outcome": outcome, "httpStatus": status, "readErrorClass": read_error_class})
        retryable = (status is None or (status is not None and 500 <= status <= 599) or
                     (status is not None and 200 <= status < 300 and final_state == "incomplete_response"))
        if final_state == "redirect_rejected": retryable = False
        if retryable and attempt < MAX_RETRIES:
            continue
        break
    body_hash = hashlib.sha256(body).hexdigest() if final_has_response else None
    meta = {"requestedUrl": url, "responseUrl": response_url, "httpStatus": status,
            "state": final_state, "complete": final_state == "complete",
            "attempts": attempts, "responses": responses, "responseHeaders": headers,
            "bodyFile": responses[-1]["bodyFile"] if final_has_response and responses else None,
            "bodyBytesCaptured": len(body),
            "bodyTruncated": responses[-1]["bodyTruncated"] if final_has_response and responses else False,
            "bodySha256": body_hash,
            "firstAttemptAtUtc": attempts[0]["startedAtUtc"] if attempts else None,
            "lastAttemptAtUtc": attempts[-1]["endedAtUtc"] if attempts else None,
            "cookiesOrCredentialsSent": False}
    with meta_path.open("x", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
    return meta


def _origin(url):
    p = urlsplit(url)
    return p.scheme, p.hostname, p.port or (443 if p.scheme == "https" else 80)


def _json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def _quarter_ranges(start: dt.date, end: dt.date):
    if start > end: raise ValueError("start must be on or before end")
    day = start
    while day <= end:
        qend_month = ((day.month - 1) // 3 + 1) * 3
        qend = dt.date(day.year, 12, 31) if qend_month == 12 else dt.date(day.year, qend_month + 1, 1) - dt.timedelta(days=1)
        chunk_end = min(qend, end)
        yield day, chunk_end
        day = chunk_end + dt.timedelta(days=1)


def build_query_plan(start: dt.date, end: dt.date, crops=None, *, korean_language=False):
    selected = list(CROPS if crops is None else crops)
    if not selected or len(set(selected)) != len(selected) or any(c not in CROPS for c in selected):
        raise ValueError("crops must be unique names from: " + ", ".join(CROPS))
    queries = []
    for crop in selected:
        profile = CROPS[crop]
        for first, last in _quarter_ranges(start, end):
            fields = {"action": "daily", "actionUrl": POST_PATH, "itemcategorycode": "200",
                      "yyyy": "", "start_mmdd": "", "period": "-10", "mm": "", "week": "", "regday": "",
                      "startday": first.isoformat(), "endday": last.isoformat(), "itemcode": profile["itemcode"],
                      "kindcode": profile["kindcode"], "productrankcode": "", "marketclscode": "", "marketcode": "",
                      "countycode": "", "convert_kg_yn": "N"}
            form = urlencode(fields)
            headers = _post_headers(korean_language)
            if korean_language:
                request_hash = hashlib.sha256(_json_bytes({"method": "POST", "url": POST_URL,
                    "requestBody": form, "requestHeaders": headers})).hexdigest()
            else:
                # Preserve the original default plan identity so completed legacy
                # captures can be resumed and verified without rewriting them.
                request_hash = hashlib.sha256(b"POST\n" + POST_URL.encode() + b"\n" + form.encode()).hexdigest()
            qid = f"{crop}_{first.isoformat()}_{last.isoformat()}_{request_hash[:12]}"
            query = {"id": qid, "crop": crop, "item": profile["item"], "variety": profile["variety"],
                            "grade": "상품", "unit": profile["unit"], "start": first.isoformat(), "end": last.isoformat(),
                            "method": "POST", "url": POST_URL, "formFields": fields, "requestBody": form,
                            "requestSha256": request_hash}
            if korean_language: query["requestHeaders"] = headers
            queries.append(query)
    return queries


def _write_create(path: Path, data: bytes):
    with path.open("xb") as f: f.write(data)


def _verify_query_checkpoint(directory: Path, query: dict):
    meta_path = directory / f"query-{query['id']}.json"
    return _verify_query_checkpoint_file(directory, query, meta_path)


def _verify_query_checkpoint_file(directory: Path, query: dict, meta_path: Path):
    if not meta_path.exists():
        if list(directory.glob(f"query-{query['id']}*-attempt-*.body")):
            return {"state": "error", "error": "orphaned response body without query metadata"}
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    keys = ["requestSha256", "method", "url", "requestBody"]
    if "requestHeaders" in query: keys.append("requestHeaders")
    for key in keys:
        if meta.get(key) != query[key]: raise ValueError(f"saved query checkpoint plan mismatch for {query['id']}")
    for response in meta.get("responses", []):
        saved = (directory / response["bodyFile"]).read_bytes()
        if len(saved) != response["bodyBytesCaptured"] or hashlib.sha256(saved).hexdigest() != response["bodySha256"]:
            raise ValueError(f"saved response checkpoint failed SHA/length verification for {query['id']}")
    if meta.get("rowsFile"):
        row_path = directory / meta["rowsFile"]
        row_bytes = row_path.read_bytes()
        if hashlib.sha256(row_bytes).hexdigest() != meta["rowsSha256"]:
            raise ValueError(f"saved parsed rows SHA mismatch for {query['id']}")
        rows = json.loads(row_bytes)
        return {"state": meta.get("state", "completed"), "rows": rows, "meta": meta, "resumed": True}
    return {"state": meta.get("state", "error"), "error": meta.get("error", meta.get("state", "saved failure")), "meta": meta, "resumed": True}


def _price_counts(rows):
    prices = [p for row in rows for p in row.get("prices", {}).values()]
    return {"priceCells": len(prices), "missing": sum(p["status"] == "missing" for p in prices),
            "invalid": sum(p["status"] == "invalid" for p in prices),
            "negative": sum("negative_price" in p["flags"] for p in prices),
            "nonfinite": sum("nonfinite_price" in p["flags"] for p in prices)}


def _capture_post(query: dict, directory: Path, *, opener, budget: int, used: int, sleep, clock, generation=0):
    attempts, responses = [], []
    final_state, final_error = "transport_failure", None
    last = {"status": None, "headers": {}, "responseUrl": None, "body": b"", "hasResponse": False}
    while len(attempts) < MAX_RETRIES + 1 and used < budget:
        if used: sleep(MIN_INTERVAL_SECONDS)
        attempt = len(attempts) + 1
        started = clock(); used += 1
        last = {"status": None, "headers": {}, "responseUrl": None, "body": b"", "hasResponse": False}
        request = Request(query["url"], data=query["requestBody"].encode("ascii"), method="POST",
                          headers=query.get("requestHeaders", _post_headers()))
        try:
            response = opener.open(request, timeout=TIMEOUT_SECONDS)
        except HTTPError as exc:
            response = exc
        except (URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            ended = clock()
            attempts.append({"attempt": attempt, "startedAtUtc": started, "endedAtUtc": ended,
                             "outcome": "transport_failure", "httpStatus": None, "errorClass": type(exc).__name__})
            final_state, final_error = "transport_failure", type(exc).__name__
            if len(attempts) < MAX_RETRIES + 1 and used < budget: continue
            break
        status = getattr(response, "status", None) or getattr(response, "code", None)
        response_url = response.geturl() if hasattr(response, "geturl") else query["url"]
        hobj = getattr(response, "headers", {})
        headers = {str(k): str(v) for k, v in hobj.items()
                   if str(k).lower() not in {"set-cookie", "authorization", "proxy-authorization", "www-authenticate"}}
        content_length = hobj.get("Content-Length") if hasattr(hobj, "get") else None
        expected, invalid_length = None, False
        if content_length is not None:
            try:
                expected = int(content_length)
                if expected < 0: raise ValueError
            except (TypeError, ValueError): expected, invalid_length = None, True
        read_error = None
        try:
            body, oversize, premature = _read_bounded(response, expected)
        except _ResponseReadFailure as exc:
            body, oversize, premature, read_error = exc.body, exc.oversize, True, exc.error_class
        finally:
            close = getattr(response, "close", None)
            if close: close()
        ended = clock()
        same_origin = _origin(response_url) == ("https", "www.kamis.or.kr", 443)
        incomplete = premature or invalid_length or read_error is not None
        if not same_origin or (status is not None and 300 <= status < 400): final_state, outcome = "redirect_rejected", "redirect_rejected"
        elif status in (401, 403, 429): final_state, outcome = "http_failure", "http_failure"
        elif oversize: final_state, outcome = "oversize_response", "oversize_response"
        elif status is not None and 200 <= status < 300 and incomplete: final_state, outcome = "incomplete_response", "incomplete_response"
        elif status is not None and 200 <= status < 300: final_state, outcome = "http_success", "http_success"
        else: final_state, outcome = "http_failure", "http_failure"
        suffix = f"-retry-{generation}" if generation else ""
        body_path = directory / f"query-{query['id']}{suffix}-attempt-{attempt}.body"
        _write_create(body_path, body)
        rec = {"attempt": attempt, "requestUrl": query["url"], "responseUrl": response_url,
               "startedAtUtc": started, "endedAtUtc": ended, "httpStatus": status, "outcome": outcome,
               "headers": headers, "bodyFile": body_path.name, "bodyBytesCaptured": len(body),
               "bodySha256": hashlib.sha256(body).hexdigest(), "advertisedContentLength": content_length,
               "framingComplete": not (oversize or incomplete), "readErrorClass": read_error}
        responses.append(rec); attempts.append({k: rec[k] for k in ("attempt", "startedAtUtc", "endedAtUtc", "outcome", "httpStatus", "readErrorClass")})
        last = {"status": status, "headers": headers, "responseUrl": response_url, "body": body, "hasResponse": True}
        last["bodyFile"] = body_path.name
        last["bodySha256"] = rec["bodySha256"]
        retryable = same_origin and ((status is not None and 500 <= status <= 599) or
                                     (status is not None and 200 <= status < 300 and incomplete))
        if retryable and len(attempts) < MAX_RETRIES + 1 and used < budget: continue
        if outcome == "http_success": final_state = "http_success"
        elif outcome == "http_failure": final_state = "http_failure"
        final_error = None if final_state == "http_success" else outcome
        break
    return {"attempts": attempts, "responses": responses, "state": final_state,
            "error": final_error, "final": last, "requestsUsed": used}


def compare_ui_export(web_rows: dict, ui_rows: dict):
    """Compare a single POST snapshot to one UI export without joining source epochs."""
    fields = ("item", "variety", "grade", "unit", "start", "end")
    mismatches = [field for field in fields if web_rows["requested"].get(field) != ui_rows["requested"].get(field)]
    if web_rows.get("captionRaw") != ui_rows.get("captionRaw"): mismatches.append("captionRaw")
    columns = [h for h in PILOT_HEADERS if h not in ("구분", "등락율")]
    web = {r["date"]: r for r in web_rows["rawRows"]}; ui = {r["date"]: r for r in ui_rows["rawRows"]}
    differences = []
    for day in sorted(set(web) | set(ui)):
        if day not in web or day not in ui:
            differences.append({"date": day, "reason": "date_support_mismatch"}); continue
        for column in columns:
            a, b = web[day]["prices"][column], ui[day]["prices"][column]
            if a["status"] != b["status"] or a["flags"] or b["flags"]:
                differences.append({"date": day, "column": column, "reason": "status_or_quality_mismatch",
                                    "webRaw": a["raw"], "uiRaw": b["raw"]})
            elif a["status"] == "observed" and Decimal(a["decimal"]) != Decimal(b["decimal"]):
                differences.append({"date": day, "column": column, "reason": "price_mismatch",
                                    "webRaw": a["raw"], "uiRaw": b["raw"]})
    return {"webSourceSha256": web_rows["sourceSha256"], "uiSourceSha256": ui_rows["sourceSha256"],
            "sameCaptionAndRequest": not mismatches, "metadataMismatches": mismatches,
            "priceColumns": columns, "differences": differences,
            "exactMatch": not mismatches and not differences, "epochsMerged": False}


def compare_five_crop_pilot(collection_dir: Path | str, manifest_path: Path | str,
                            input_dir: Path | str, output_path: Path | str):
    """Verify and compare the five frozen Q1 2024 web checkpoints to their UI exports."""
    collection_dir, manifest_path = Path(collection_dir), Path(manifest_path)
    input_dir, output_path = Path(input_dir).resolve(), Path(output_path)
    start, end = dt.date(2024, 1, 1), dt.date(2024, 3, 31)
    quarter_days = {(start + dt.timedelta(days=n)).isoformat() for n in range((end-start).days + 1)}
    plan_path = collection_dir / "plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    queries = [q for q in plan.get("queries", []) if q.get("start") == start.isoformat() and q.get("end") == end.isoformat()]
    if len(queries) != 5 or {q.get("crop") for q in queries} != set(CROPS):
        raise ValueError("pilot collection must contain exactly one Q1 2024 query for each of the five crops")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    entries = manifest.get("files")
    if not isinstance(entries, list): raise ValueError("UI provenance manifest files must be a list")
    report_items = []
    for query in queries:
        qmeta_path = collection_dir / f"query-{query['id']}.json"
        for generation in range(1, 3):
            candidate = collection_dir / f"query-{query['id']}-retry-{generation}.json"
            if candidate.exists(): qmeta_path = candidate
        meta = json.loads(qmeta_path.read_text(encoding="utf-8"))
        if meta.get("state") != "completed": raise ValueError(f"web pilot {query['crop']} is not complete")
        _verify_query_checkpoint_file(collection_dir, query, qmeta_path)
        raw_path = collection_dir / meta["sourceBodyFile"]
        raw = raw_path.read_bytes()
        source_hash = hashlib.sha256(raw).hexdigest()
        if source_hash != meta.get("sourceSha256"): raise ValueError(f"web source hash mismatch for {query['crop']}")
        rows_path = collection_dir / meta["rowsFile"]
        rows_bytes = rows_path.read_bytes()
        if hashlib.sha256(rows_bytes).hexdigest() != meta.get("rowsSha256"): raise ValueError(f"web rows hash mismatch for {query['crop']}")
        web_rows = json.loads(rows_bytes)
        web_dates = {r["date"] for r in web_rows.get("rawRows", [])}
        if not web_dates or not web_dates <= quarter_days: raise ValueError(f"web pilot {query['crop']} has invalid Q1 date support")
        if web_rows.get("qualityStatus") != "clear": raise ValueError(f"web pilot {query['crop']} has unresolved price quality")
        profile = CROPS[query["crop"]]
        candidates = [e for e in entries if e.get("start") == start.isoformat() and e.get("end") == end.isoformat()
                      and e.get("item") == profile["item"] and e.get("variety") == profile["variety"]
                      and e.get("unit") == profile["unit"] and e.get("grade") == "상품"]
        if len(candidates) != 1: raise ValueError(f"UI manifest must contain exactly one Q1 file for {query['crop']}")
        entry = candidates[0]
        rel = entry["path"]
        ui_path = (input_dir / rel).resolve()
        if ui_path.parent != input_dir: raise ValueError(f"UI manifest path must be a direct child: {rel}")
        ui_raw = ui_path.read_bytes()
        ui_sha = hashlib.sha256(ui_raw).hexdigest()
        if len(ui_raw) != entry["bytes"] or ui_sha != entry["sha256"]: raise ValueError(f"UI source provenance mismatch for {query['crop']}")
        ui_rows = parse_export(ui_raw, start=start, end=end, item=entry["item"], variety=entry["variety"],
                               grade=entry["grade"], unit=entry["unit"])
        if len(ui_rows["rawRows"]) != entry["uiDataRows"]: raise ValueError(f"UI row-count provenance mismatch for {query['crop']}")
        ui_dates = {r["date"] for r in ui_rows["rawRows"]}
        if not ui_dates or not ui_dates <= quarter_days: raise ValueError(f"UI pilot {query['crop']} has invalid Q1 date support")
        if len(ui_rows["rawRows"]) != entry["uiDataRows"]: raise ValueError(f"UI row-count provenance mismatch for {query['crop']}")
        if web_dates != ui_dates: raise ValueError(f"web/UI observed-date support mismatch for {query['crop']}")
        ui_prices = [p for row in ui_rows["rawRows"] for p in row["prices"].values()]
        if any(p["status"] not in ("observed", "missing") or p["flags"] for p in ui_prices): raise ValueError(f"UI pilot {query['crop']} has unresolved price quality")
        comparison = compare_ui_export(web_rows, ui_rows)
        if not comparison["exactMatch"]: raise ValueError(f"UI and web pilot differ for {query['crop']}: {comparison['differences'][:3]}")
        report_items.append({"crop": query["crop"], "start": start.isoformat(), "end": end.isoformat(),
                             "webSourceSha256": source_hash, "webRowsSha256": meta["rowsSha256"],
                             "uiPath": rel, "uiSourceSha256": ui_sha, "uiBytes": len(ui_raw),
                             "uiDataRows": len(ui_rows["rawRows"]), "comparedDates": len(web_dates),
                             "comparedPriceCells": len(web_dates) * len(comparison["priceColumns"]),
                             "comparison": comparison})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {"source": "offline KAMIS Q1 2024 five-crop UI comparison", "collectionPlanSha256": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
              "uiManifestSha256": hashlib.sha256(manifest_bytes).hexdigest(), "cropCount": len(report_items),
              "allExactMatch": all(x["comparison"]["exactMatch"] for x in report_items), "crops": report_items,
              "networkRequests": 0, "epochsMerged": False}
    _write_create(output_path, _json_bytes(report))
    return report


def collect_history(start: dt.date, end: dt.date, *, crops=None, output_dir: Path | str,
                    max_requests: int = 5, retry_transient: bool = False, korean_language=False,
                    opener=None, sleep=time.sleep, clock=_utc_now):
    """Collect quarter-bounded public KAMIS daily wholesale POST results with immutable checkpoints."""
    if max_requests < 0: raise ValueError("max_requests must be nonnegative")
    queries = build_query_plan(start, end, crops, korean_language=korean_language)
    directory = Path(output_dir); directory.mkdir(parents=True, exist_ok=True)
    plan_bytes = _json_bytes({"schemaVersion": 1, "method": "official public UI form POST", "queries": queries})
    plan_hash = hashlib.sha256(plan_bytes).hexdigest()
    plan_path = directory / "plan.json"
    if plan_path.exists() and plan_path.read_bytes() != plan_bytes: raise ValueError("frozen query plan differs from saved plan")
    if not plan_path.exists(): _write_create(plan_path, plan_bytes)
    # Verify all saved generations before any request. A saved auth/rate/redirect
    # response anywhere in the plan suppresses retries for this invocation.
    saved_by_id, existing_blocker = {}, False
    for query in queries:
        saved = _verify_query_checkpoint(directory, query)
        generation = 0
        for n in range(1, 3):
            retry_path = directory / f"query-{query['id']}-retry-{n}.json"
            if retry_path.exists():
                saved = _verify_query_checkpoint_file(directory, query, retry_path)
                generation = n
            elif list(directory.glob(f"query-{query['id']}-retry-{n}-attempt-*.body")):
                raise ValueError(f"orphaned retry response body without query metadata for {query['id']} generation {n}")
        saved_by_id[query["id"]] = (saved, generation)
        meta = saved.get("meta", {}) if saved else {}
        existing_blocker |= meta.get("state") == "redirect_rejected" or meta.get("httpStatus") in (401, 403, 429)
    client = opener or _opener(); used = 0; result_rows = []; stopped = False
    for query in queries:
        if stopped:
            result_rows.append({"id": query["id"], "crop": query["crop"], "start": query["start"], "end": query["end"],
                                "state": "pending", "resumed": False, "error": "collection stopped after security/rate-limit response"})
            continue
        saved, generation = saved_by_id[query["id"]]
        if saved:
            meta = saved.get("meta", {})
            blocked = meta.get("state") == "redirect_rejected" or meta.get("httpStatus") in (401, 403, 429)
            retryable_saved = (meta.get("state") in ("transport_failure", "incomplete_response", "oversize_response") or
                               (meta.get("httpStatus") is not None and 500 <= meta["httpStatus"] <= 599))
            if retry_transient and not existing_blocker and retryable_saved and generation < 2:
                saved = None
                generation += 1
            else:
                stopped = blocked
        if saved:
            saved_rows = saved.get("rows", {}).get("rawRows", [])
            result_rows.append({"id": query["id"], "crop": query["crop"], "start": query["start"], "end": query["end"],
                                "state": saved["state"], "resumed": True, "error": saved.get("error"),
                                "support": saved.get("rows", {}).get("support"),
                                "qualityStatus": saved.get("rows", {}).get("qualityStatus"),
                                "priceCounts": _price_counts(saved_rows) if saved_rows else None})
            continue
        if existing_blocker:
            result_rows.append({"id": query["id"], "crop": query["crop"], "start": query["start"], "end": query["end"],
                                "state": "pending", "resumed": False, "error": "saved security/rate-limit response in frozen plan; no requests sent"})
            continue
        if used >= max_requests:
            result_rows.append({"id": query["id"], "crop": query["crop"], "start": query["start"], "end": query["end"],
                                "state": "pending", "resumed": False})
            continue
        captured = _capture_post(query, directory, opener=client, budget=max_requests, used=used, sleep=sleep, clock=clock,
                                 generation=generation)
        used = captured["requestsUsed"]
        state, parsed, error, rows_path, rows_hash = captured["state"], None, captured["error"], None, None
        if state == "http_success":
            try:
                parsed = parse_export(captured["final"]["body"], start=dt.date.fromisoformat(query["start"]),
                                      end=dt.date.fromisoformat(query["end"]), item=query["item"], variety=query["variety"],
                                      grade=query["grade"], unit=query["unit"], table_id="itemTable_1")
                parsed["requested"].update({"crop": query["crop"]})
                price_cells = [p for row in parsed["rawRows"] for p in row["prices"].values()]
                parsed["qualityStatus"] = "review_needed" if any(p["status"] == "invalid" or p["flags"] for p in price_cells) else "clear"
                if parsed["rawRows"]:
                    state = "completed"
                else:
                    state, error = "unclassified_unavailable", "table present without data rows; no normal-empty marker verified"
                rows_suffix = f"-retry-{generation}" if generation else ""
                rows_path = directory / f"query-{query['id']}{rows_suffix}.rows.json"
                rows_bytes = _json_bytes(parsed); _write_create(rows_path, rows_bytes)
                rows_hash = hashlib.sha256(rows_bytes).hexdigest()
            except (ValueError, UnicodeError) as exc:
                error = str(exc)
                if "missing caption or header" in error or "exactly one complete table" in error:
                    state = "unclassified_unavailable"
                else: state = "malformed"
        final_response = captured["final"] if captured["final"]["hasResponse"] else None
        record = {"id": query["id"], "crop": query["crop"], "start": query["start"], "end": query["end"],
                  "method": query["method"], "url": query["url"], "requestBody": query["requestBody"],
                  **({"requestHeaders": query["requestHeaders"]} if "requestHeaders" in query else {}),
                  "requestSha256": query["requestSha256"],
                  "fields": query["formFields"], "state": state,
                  "error": error, "attempts": captured["attempts"], "responses": captured["responses"],
                  "httpStatus": final_response["status"] if final_response else None,
                  "responseUrl": final_response["responseUrl"] if final_response else None,
                  "responseHeaders": final_response["headers"] if final_response else {},
                  "sourceSha256": final_response["bodySha256"] if final_response else None,
                  "sourceBodyFile": final_response["bodyFile"] if final_response else None,
                  "rowsFile": rows_path.name if rows_path else None, "rowsSha256": rows_hash,
                  "firstPublicationTime": None}
        meta_path = directory / (f"query-{query['id']}-retry-{generation}.json" if generation else f"query-{query['id']}.json")
        _write_create(meta_path, _json_bytes(record))
        result_rows.append({"id": query["id"], "crop": query["crop"], "start": query["start"], "end": query["end"],
                            "state": state, "resumed": False, "error": error,
                            "support": parsed.get("support") if parsed else None,
                            "qualityStatus": parsed.get("qualityStatus") if parsed else None,
                            "priceCounts": _price_counts(parsed["rawRows"]) if parsed else None})
        if state == "redirect_rejected" or (final_response and final_response["status"] in (401, 403, 429)):
            stopped = True
    counts = {"completed": sum(r["state"] == "completed" for r in result_rows),
              "empty": sum(r["state"] == "empty" for r in result_rows),
              "pending": sum(r["state"] == "pending" for r in result_rows)}
    counts["error"] = len(result_rows) - counts["completed"] - counts["empty"] - counts["pending"]
    summary = {"planSha256": plan_hash, "plannedQueries": len(queries), "requestsUsed": used,
               **counts, "planComplete": counts["completed"] + counts["empty"] == len(queries),
               "missingDatesNotClassifiedAsClosure": True,
               "limitations": ["First publication time unknown.", "Survey prices are not auction/kg target equivalents.",
                               "No normal-empty marker has been verified; unavailable tables are not classified as empty."],
               "queries": result_rows}
    stamp = clock().replace(":", "").replace("-", "")
    summary_path = directory / f"summary-{stamp}.json"
    suffix = 1
    while summary_path.exists():
        summary_path = directory / f"summary-{stamp}-{suffix}.json"
        suffix += 1
    _write_create(summary_path, _json_bytes(summary))
    summary["summaryFile"] = summary_path.name
    return summary


def main(argv=None):
    argv = list(__import__("sys").argv[1:] if argv is None else argv)
    if argv and argv[0] == "collect":
        collect_parser = argparse.ArgumentParser(description="Collect bounded quarter-wise official KAMIS wholesale history")
        collect_parser.add_argument("mode", choices=["collect"])
        collect_parser.add_argument("--start", type=dt.date.fromisoformat, default=dt.date(2014, 1, 1))
        collect_parser.add_argument("--end", type=dt.date.fromisoformat, default=dt.date(2026, 9, 30))
        collect_parser.add_argument("--crops", nargs="+", choices=list(CROPS), default=list(CROPS))
        collect_parser.add_argument("--korean-language", action="store_true", help="add Accept-Language: ko-KR,ko;q=0.9,en;q=0.8 for one controlled public-request diagnostic")
        collect_parser.add_argument("--max-requests", type=int, default=5)
        collect_parser.add_argument("--retry-transient", action="store_true", help="start a bounded new generation for prior transient failures (at most two retries per query)")
        collect_parser.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/kamis_wholesale_history")
        args = collect_parser.parse_args(argv)
        raw_root = (ROOT / "data/raw").resolve(); destination = args.output_dir.resolve()
        if destination != raw_root and raw_root not in destination.parents:
            collect_parser.error("--output-dir must be under this repository's Git-ignored data/raw directory")
        try:
            summary = collect_history(args.start, args.end, crops=args.crops,
                                      output_dir=destination, max_requests=args.max_requests,
                                      retry_transient=args.retry_transient, korean_language=args.korean_language)
        except (OSError, ValueError) as exc:
            collect_parser.error(str(exc))
        print(json.dumps({k: summary[k] for k in ("summaryFile", "planSha256", "plannedQueries", "requestsUsed",
                                                   "completed", "empty", "error", "pending", "planComplete")}, ensure_ascii=False))
        return 0 if summary["planComplete"] else 1
    if argv and argv[0] == "compare-pilot":
        compare_parser = argparse.ArgumentParser(description="Compare verified Q1 five-crop KAMIS web captures with UI exports")
        compare_parser.add_argument("mode", choices=["compare-pilot"])
        compare_parser.add_argument("--collection-dir", required=True, type=Path)
        compare_parser.add_argument("--manifest", required=True, type=Path)
        compare_parser.add_argument("--input-dir", required=True, type=Path)
        compare_parser.add_argument("--output", required=True, type=Path)
        args = compare_parser.parse_args(argv)
        try: compare_five_crop_pilot(args.collection_dir, args.manifest, args.input_dir, args.output)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as exc:
            print(f"pilot comparison failed: {exc}", file=__import__("sys").stderr); return 2
        return 0
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("url", nargs="?", default=DEFAULT_URL)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data/raw/kamis_web_probe")
    args = ap.parse_args(argv)
    raw_root = (ROOT / "data/raw").resolve()
    destination = args.output_dir.resolve()
    if destination != raw_root and raw_root not in destination.parents:
        ap.error("--output-dir must be under this repository's Git-ignored data/raw directory")
    try:
        result = probe(args.url, output_dir=destination)
    except (OSError, ValueError) as exc:
        ap.error(str(exc))
    print(json.dumps({"state": result["state"], "bodySha256": result["bodySha256"],
                      "bodyBytesCaptured": result["bodyBytesCaptured"], "bodyFile": result["bodyFile"]}, ensure_ascii=False))
    return 0 if result["complete"] else 1


if __name__ == "__main__": raise SystemExit(main())
