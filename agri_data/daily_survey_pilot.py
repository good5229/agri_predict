"""Small, resumable aT daily survey price API pilot; defaults to a dry run."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import ssl
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener
from email.utils import parsedate_to_datetime

ENDPOINT = "https://apis.data.go.kr/B552845/perDay/price"
WINDOWS = (("2025-09-01", "2025-09-07"), ("2026-07-01", "2026-07-07"))
CATEGORY = ("200", "채소류")
ITEMS = (("211", "배추"), ("231", "무"), ("241", "건고추"))
FIELDS = ("exmn_ymd", "se_cd", "se_nm", "ctgry_cd", "ctgry_nm", "item_cd", "item_nm",
          "vrty_cd", "vrty_nm", "grd_cd", "grd_nm", "sgg_cd", "sgg_nm", "unit",
          "unit_sz", "mrkt_cd", "mrkt_nm", "exmn_dd_prc", "exmn_dd_cnvs_prc", "orgnl_reg_dt")
MAX_PAGE_SIZE = 1000
MAX_BODY = 2 * 1024 * 1024
MAX_RETRIES = 3
KST = dt.timezone(dt.timedelta(hours=9))


class PilotError(RuntimeError):
    pass


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _load_key(env_file: Path) -> str:
    """Read only the named key, without exposing any other env-file values."""
    key = os.environ.get("DATA_GO_KR_SERVICE_KEY_DECODED", "")
    if not key:
        try:
            lines = env_file.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise PilotError("cannot read the specified env file") from exc
        for line in lines:
            name, sep, value = line.partition("=")
            if sep and name.strip() == "DATA_GO_KR_SERVICE_KEY_DECODED":
                key = value.strip().strip("\"'")
                break
    if not key:
        raise PilotError("missing DATA_GO_KR_SERVICE_KEY_DECODED")
    return key


def _requests_for(page_size: int) -> list[dict[str, str]]:
    requests = []
    for start, end in WINDOWS:
        for item, _ in ITEMS:
            requests.append({"start": start, "end": end, "item": item,
                             "page": "1", "size": str(page_size)})
    return requests


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _check_row(row: Any, req: dict[str, str]) -> list[str]:
    issues = []
    if not isinstance(row, dict):
        return ["item_not_object"]
    for name in FIELDS:
        if name not in row:
            issues.append("missing:" + name)
    if str(row.get("ctgry_cd", "")) != CATEGORY[0]:
        issues.append("category_code_mismatch")
    if str(row.get("item_cd", "")) != req["item"]:
        issues.append("item_code_mismatch")
    try:
        day = dt.date.fromisoformat(str(row.get("exmn_ymd", ""))[:10])
        if not dt.date.fromisoformat(req["start"]) <= day <= dt.date.fromisoformat(req["end"]):
            issues.append("survey_date_outside_request")
    except ValueError:
        issues.append("invalid_survey_date")
    for name in ("exmn_dd_prc", "exmn_dd_cnvs_prc"):
        val = row.get(name)
        if val is not None and (isinstance(val, bool) or not isinstance(val, (str, int, float))):
            issues.append("invalid_price:" + name)
    registered = row.get("orgnl_reg_dt")
    if registered in (None, ""):
        issues.append("missing_or_null_orgnl_reg_dt")
    else:
        try:
            stamp = dt.datetime.fromisoformat(str(registered).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            issues.append("unparsed_orgnl_reg_dt")
        else:
            stamp = stamp.replace(tzinfo=KST) if stamp.tzinfo is None else stamp.astimezone(KST)
            try:
                survey = dt.date.fromisoformat(str(row.get("exmn_ymd", ""))[:10])
            except ValueError:
                survey = None
            if survey is not None:
                if stamp >= dt.datetime.combine(survey + dt.timedelta(days=1), dt.time.min, KST):
                    issues.append("registered_after_survey_day")
                if stamp > dt.datetime.combine(survey + dt.timedelta(days=1), dt.time(10), KST):
                    issues.append("registered_after_next_day_10_kst")
    return issues


def _decode_items(document: dict[str, Any]) -> tuple[list[Any], int, int, int]:
    if not isinstance(document, dict) or not isinstance(document.get("header"), dict):
        raise PilotError("invalid_envelope")
    header = document["header"]
    if str(header.get("resultCode", "")) not in ("00", "0", "200"):
        raise PilotError("api_error:" + str(header.get("resultCode", "unknown")))
    body = document.get("body")
    if not isinstance(body, dict):
        raise PilotError("invalid_body")
    try:
        page = int(body["pageNo"])
        size = int(body["numOfRows"])
        total = int(body["totalCount"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PilotError("invalid_pagination_fields") from exc
    container = body.get("items")
    if not isinstance(container, dict) or "item" not in container:
        raise PilotError("invalid_items_container")
    raw = container["item"]
    if raw is None:
        items = []
    elif isinstance(raw, list):
        items = raw
    elif isinstance(raw, dict):
        items = [raw]
    else:
        raise PilotError("invalid_item_value")
    return items, page, size, total


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _retry_after(headers: dict[str, Any]) -> float | None:
    value = next((str(v) for k, v in headers.items() if k.lower() == "retry-after"), "")
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            when = parsedate_to_datetime(value)
            if when.tzinfo is None:
                when = when.replace(tzinfo=dt.timezone.utc)
            return max(0.0, (when - dt.datetime.now(dt.timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _safe_headers(headers: Any) -> dict[str, str]:
    allowed = {"content-type", "content-length", "retry-after", "date", "server", "x-ratelimit-limit",
               "x-ratelimit-remaining", "x-ratelimit-reset"}
    return {str(k): str(v)[:1000] for k, v in headers.items() if str(k).lower() in allowed}


def _valid_saved_page(root: Path, page: dict[str, Any]) -> bool:
    name, digest = page.get("raw_file"), page.get("sha256")
    if not isinstance(name, str) or Path(name).name != name or not isinstance(digest, str):
        return False
    try:
        content = (root / name).read_bytes()
    except OSError:
        return False
    return hashlib.sha256(content).hexdigest() == digest


def _write_raw(path: Path, payload: bytes) -> None:
    if not path.exists() or hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(payload).digest():
        path.write_bytes(payload)
    path.chmod(0o600)


def _persist_attempt(root: Path, reqid: str, attempt_no: int, *, status: int | None,
                     headers: Any, payload: bytes, error_type: str | None,
                     query: dict[str, str], retrieved_at: str) -> dict[str, Any]:
    digest = hashlib.sha256(payload).hexdigest()
    raw_name = f"{reqid}_attempt{attempt_no}_{digest}.bin"
    raw_path = root / raw_name
    _write_raw(raw_path, payload)
    record = {"retrieved_at": retrieved_at, "endpoint": ENDPOINT,
              "query": query, "http_status": status, "headers": _safe_headers(headers),
              "raw_file": raw_name, "sha256": digest, "bytes": len(payload),
              "transport_error_type": error_type}
    with (root / "attempts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def run(*, env_file: Path, data_root: Path, execute: bool = False, page_size: int = MAX_PAGE_SIZE,
        max_pages: int = 30, opener: Any | None = None, sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic) -> dict[str, Any]:
    if not 1 <= page_size <= MAX_PAGE_SIZE or max_pages < 1:
        raise PilotError("page size and page budget are out of range")
    if not execute:
        return {"mode": "dry_run", "requests": _requests_for(page_size), "page_size": page_size,
                "max_pages": max_pages, "live_requests": 0}
    key = _load_key(env_file)
    root = data_root / "raw" / "daily-survey"
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = root / "checkpoint.json"
    scope = {"windows": WINDOWS, "category": CATEGORY[0], "items": ITEMS, "page_size": page_size}
    if checkpoint_path.exists():
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if state.get("scope") != json.loads(json.dumps(scope)):
            raise PilotError("checkpoint scope mismatch")
    else:
        state = {"scope": scope, "pages": {}, "attempts": 0, "failures": []}
    state["pages"] = {key: value for key, value in state.get("pages", {}).items()
                       if _valid_saved_page(root, value)}
    opener = opener or build_opener(NoRedirect(), __import__("urllib.request").request.HTTPSHandler(context=ssl.create_default_context()))
    last_call: float | None = None
    outcomes = []
    quota_stop = False
    for base in _requests_for(page_size):
        if quota_stop:
            break
        prefix = f"{base['start']}_{base['end']}_{base['item']}_"
        saved_first = state["pages"].get(prefix + "1")
        total_pages = (max(1, (saved_first["total_count"] + page_size - 1) // page_size)
                       if saved_first is not None else None)
        page = 1
        while total_pages is None or page <= total_pages:
            if page > max_pages:
                outcomes.append({"request": base, "state": "incomplete_page_budget", "next_page": page})
                break
            req = dict(base, page=str(page))
            reqid = f"{base['start']}_{base['end']}_{base['item']}_{page}"
            if reqid in state["pages"]:
                page += 1
                continue
            params = {"serviceKey": key, "pageNo": page, "numOfRows": page_size,
                      "cond[exmn_ymd::GTE]": base["start"], "cond[exmn_ymd::LTE]": base["end"],
                      "cond[ctgry_cd::EQ]": CATEGORY[0], "cond[item_cd::EQ]": base["item"], "returnType": "JSON"}
            url = ENDPOINT + "?" + urlencode(params)
            status, headers, payload = None, {}, b""
            sanitized_query = {"pageNo": str(page), "numOfRows": str(page_size),
                               "cond[exmn_ymd::GTE]": base["start"],
                               "cond[exmn_ymd::LTE]": base["end"],
                               "cond[ctgry_cd::EQ]": CATEGORY[0],
                               "cond[item_cd::EQ]": base["item"], "returnType": "JSON"}
            transport_error = None
            for attempt in range(MAX_RETRIES + 1):
                if last_call is not None:
                    sleep(max(0.0, 1.0 - (clock() - last_call)))
                last_call = clock()
                state["attempts"] += 1
                try:
                    response = opener.open(Request(url, headers={"Accept": "application/json"}), timeout=30)
                    status, headers = response.status, dict(response.headers.items())
                    payload = response.read(MAX_BODY + 1)
                    response.close()
                except HTTPError as exc:
                    status, headers = exc.code, dict(exc.headers.items())
                    payload = exc.read(MAX_BODY + 1)
                except (URLError, TimeoutError, OSError) as exc:
                    transport_error = type(exc).__name__
                    _persist_attempt(root, reqid, attempt + 1, status=None, headers={}, payload=b"",
                                     error_type=transport_error, query=sanitized_query,
                                     retrieved_at=_utc_now())
                    if attempt < MAX_RETRIES:
                        sleep(min(2 ** attempt, 8))
                        continue
                    failure = {"request": req, "state": "transport_failure", "error_type": transport_error}
                    outcomes.append(failure); state["failures"].append(failure)
                    break
                _persist_attempt(root, reqid, attempt + 1, status=status, headers=headers, payload=payload,
                                 error_type=None, query=sanitized_query, retrieved_at=_utc_now())
                if status in (401, 403):
                    failure = {"request": req, "state": "auth_rejected", "http_status": status}
                    outcomes.append(failure); state["failures"].append(failure)
                    break
                if status == 429:
                    if attempt < MAX_RETRIES:
                        delay = _retry_after(headers)
                        if delay is None or delay > 60:
                            failure = {"request": req, "state": "quota_wait_required", "http_status": status,
                                       "retry_after_seconds": delay}
                            outcomes.append(failure); state["failures"].append(failure)
                            quota_stop = True
                            break
                        sleep(delay)
                        continue
                    failure = {"request": req, "state": "quota_exhausted", "http_status": status}
                    outcomes.append(failure); state["failures"].append(failure)
                    quota_stop = True
                    break
                if status >= 500:
                    if attempt < MAX_RETRIES:
                        sleep(min(2 ** attempt, 8))
                        continue
                    failure = {"request": req, "state": "server_failure", "http_status": status}
                    outcomes.append(failure); state["failures"].append(failure)
                    break
                if status < 200 or status >= 300:
                    failure = {"request": req, "state": "http_failure", "http_status": status}
                    outcomes.append(failure); state["failures"].append(failure)
                    break
                if len(payload) > MAX_BODY:
                    failure = {"request": req, "state": "response_too_large", "http_status": status}
                    outcomes.append(failure); state["failures"].append(failure)
                    break
                try:
                    document = json.loads(payload)
                    items, response_page, response_size, total_count = _decode_items(document)
                except (ValueError, TypeError, PilotError) as exc:
                    failure = {"request": req, "state": "invalid_response", "reason": str(exc)[:100]}
                    outcomes.append(failure); state["failures"].append(failure)
                    break
                if response_page != page:
                    failure = {"request": req, "state": "page_mismatch", "reported_page": response_page}
                    outcomes.append(failure); state["failures"].append(failure)
                    break
                if total_count < 0 or response_size != page_size or len(items) != min(page_size, max(0, total_count - (page - 1) * page_size)):
                    failure = {"request": req, "state": "pagination_count_mismatch",
                               "reported_size": response_size, "items": len(items), "total_count": total_count}
                    outcomes.append(failure); state["failures"].append(failure)
                    break
                prior = next((v for k, v in state["pages"].items()
                              if k.startswith(prefix) and v.get("state") in ("ok", "ok_empty")), None)
                if prior is not None and prior["total_count"] != total_count:
                    failure = {"request": req, "state": "total_count_changed"}
                    outcomes.append(failure); state["failures"].append(failure)
                    break
                digest = hashlib.sha256(payload).hexdigest()
                raw_path = root / f"{reqid}_{digest}.json"
                _write_raw(raw_path, payload)
                row_quality = [{"index": i, "issues": _check_row(item, base)} for i, item in enumerate(items)]
                state["pages"][reqid] = {"state": "ok_empty" if not items and total_count == 0 else "ok",
                    "http_status": status, "headers": _safe_headers(headers),
                    "raw_file": raw_path.name, "sha256": digest, "page": page, "num_of_rows": response_size,
                    "total_count": total_count, "items_count": len(items), "quality": row_quality,
                    "retrieved_at": _utc_now(), "endpoint": ENDPOINT}
                _atomic_json(checkpoint_path, state)
                total_pages = max(1, (total_count + page_size - 1) // page_size)
                outcomes.append({"request": req, "state": state["pages"][reqid]["state"], "items": len(items), "total_count": total_count})
                break
            else:
                outcomes.append({"request": req, "state": "retry_exhausted"})
            if reqid not in state["pages"]:
                break
            page += 1
    complete = True
    for base in _requests_for(page_size):
        prefix = f"{base['start']}_{base['end']}_{base['item']}_"
        first = state["pages"].get(prefix + "1")
        if first is None:
            complete = False
            continue
        needed = max(1, (first["total_count"] + page_size - 1) // page_size)
        entries = [state["pages"].get(prefix + str(n)) for n in range(1, needed + 1)]
        if any(e is None or e.get("total_count") != first["total_count"] for e in entries) or \
                sum(e["items_count"] for e in entries if e) != first["total_count"]:
            complete = False
    complete = complete and all(x["state"] in ("ok", "ok_empty") for x in outcomes)
    _atomic_json(checkpoint_path, state)
    quality = [q for saved in state["pages"].values() for q in saved.get("quality", [])]
    row_count = len(quality)
    counts = {name: sum(name in q["issues"] for q in quality) for name in
              ("missing_or_null_orgnl_reg_dt", "unparsed_orgnl_reg_dt", "registered_after_survey_day",
               "registered_after_next_day_10_kst")}
    evaluable = sum(not any(issue in q["issues"] for issue in
                            ("missing_or_null_orgnl_reg_dt", "unparsed_orgnl_reg_dt",
                             "invalid_survey_date", "item_not_object")) for q in quality)
    return {"mode": "execute", "scope": scope, "pages_saved": len(state["pages"]),
            "attempts": state["attempts"], "outcomes": outcomes, "complete": complete,
            "orgnl_reg_dt_diagnostics": {"rows": row_count, "evaluable_rows": evaluable,
                "denominators": {"all_rows": row_count, "evaluable_timestamps": evaluable}, "counts": counts,
                "missing_fraction_over_rows": counts["missing_or_null_orgnl_reg_dt"] / row_count if row_count else None,
                "unparseable_fraction_over_rows": counts["unparsed_orgnl_reg_dt"] / row_count if row_count else None,
                "late_fractions_over_evaluable": {
                    "after_survey_day": counts["registered_after_survey_day"] / evaluable if evaluable else None,
                    "after_next_day_10_kst": counts["registered_after_next_day_10_kst"] / evaluable if evaluable else None},
                "interpretation": "registration timestamp is not proof of public availability"}}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True, help="explicit file containing DATA_GO_KR_SERVICE_KEY_DECODED")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--execute", action="store_true", help="perform live HTTPS API requests; default only prints the bounded plan")
    parser.add_argument("--page-size", type=int, default=MAX_PAGE_SIZE)
    parser.add_argument("--max-pages", type=int, default=30)
    args = parser.parse_args(argv)
    try:
        report = run(env_file=args.env_file, data_root=args.data_root, execute=args.execute,
                     page_size=args.page_size, max_pages=args.max_pages)
    except PilotError as exc:
        print(json.dumps({"state": "error", "message": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report.get("complete", not args.execute) else 1


if __name__ == "__main__":
    raise SystemExit(main())
