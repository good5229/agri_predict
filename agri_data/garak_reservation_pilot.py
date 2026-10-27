"""One-day public aggregate reservation pilot for Garak and Gangseo.

Dry-run is the default. The only executable scope is the frozen eight public
aggregate requests for 2026-10-08 KST; no detail endpoints are used.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

ENDPOINT = "https://einvoice.garak.co.kr/web/distrInfo/shipPurQtyTotal/shipQtyTotalTreeMap.json"
MARKETS = ((1, "garak"), (2, "gangseo"))
DUE_DATES = ("2026-10-08", "2026-10-09", "2026-10-13", "2026-10-16")
EXECUTION_DATE = dt.date(2026, 10, 8)
KST = ZoneInfo("Asia/Seoul")
TIMEOUT_SECONDS = (5, 20)
MAX_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 1.0
MIN_REQUEST_INTERVAL_SECONDS = 1.0
MAX_RESPONSE_BYTES = 2_000_000
DEFAULT_OUTPUT = Path("data/raw/garak_reservation_pilot")
EXPECTED_FIELDS = ("pumjongNm", "pumjongCd", "totqty", "alltotqty")


class CollectionError(RuntimeError):
    pass


def request_matrix() -> list[dict[str, Any]]:
    return [{"dueDate": day, "marketKind": market, "market": name}
            for market, name in MARKETS for day in DUE_DATES]


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stamp(now: Any) -> tuple[str, str]:
    captured = now().astimezone(dt.timezone.utc).isoformat()
    token = captured.replace(":", "").replace("+00:00", "Z") + "_" + uuid.uuid4().hex[:8]
    return captured, token


def _atomic_immutable(path: Path, payload: bytes) -> None:
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
    chunks = []
    total = 0
    truncated = False
    error = None
    try:
        for chunk in response.iter_content(64 * 1024):
            if not chunk:
                continue
            remaining = MAX_RESPONSE_BYTES - total
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                truncated = True
                error = "response_exceeded_2mb_limit"
                break
            chunks.append(chunk)
            total += len(chunk)
    except requests.RequestException as exc:
        truncated = True
        error = f"stream_error:{type(exc).__name__}:{exc}"
    finally:
        response.close()
    return b"".join(chunks), truncated, error


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


def validate_response(body: bytes, http_status: int | None, *, truncated: bool = False,
                      transport_error: str | None = None) -> dict[str, Any]:
    """Classify outcome and annotate items, preserving every source value."""
    result: dict[str, Any] = {"state": None, "issues": [], "items": [],
                              "itemCount": 0, "positiveQuantityCount": 0,
                              "questionableItemCount": 0, "itemIssueCount": 0,
                              "names": [], "cabbagePresent": False, "radishPresent": False,
                              "namePresenceSemantics": "literal substring scan of pumjongNm; not a product-category mapping"}
    if truncated:
        result["state"] = "truncated"
        result["issues"].append(transport_error or "response_truncated")
        return result
    if transport_error:
        result["state"] = "transport_error"
        result["issues"].append(transport_error)
        return result
    if http_status is None:
        result["state"] = "no_response"
        return result
    if not 200 <= http_status < 300:
        result["state"] = "http_error"
        result["issues"].append(f"http_status_{http_status}")
        return result
    if not body:
        result["state"] = "empty_response"
        return result
    try:
        doc = json.loads(body.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        result["state"] = "error_or_non_json_response"
        result["issues"].append("response_not_valid_json")
        return result
    if isinstance(doc, dict) and any(
            key in doc and doc[key] not in (None, "", False, [], {})
            for key in ("error", "errors")):
        result["state"] = "error_payload"
        result["issues"].append("json_error_payload")
        return result
    if not isinstance(doc, dict) or "data" not in doc or not isinstance(doc["data"], list):
        result["state"] = "invalid_json_structure"
        result["issues"].append("expected_object_with_data_array")
        return result
    data = doc["data"]
    if not data:
        result["state"] = "normal_empty"
        return result
    result["state"] = "data_present"
    names = []
    for index, item in enumerate(data):
        record = {"sourceIndex": index, "sourceValues": {}, "issues": []}
        if not isinstance(item, dict):
            record["sourceValues"] = item
            record["issues"].append("item_not_object")
            result["items"].append(record)
            continue
        for field in EXPECTED_FIELDS:
            record["sourceValues"][field] = item.get(field)
            if field not in item:
                record["issues"].append(f"missing_{field}")
            elif field in ("pumjongNm", "pumjongCd") and (item[field] is None or item[field] == ""):
                record["issues"].append(f"{field}_null_value")
        name = item.get("pumjongNm")
        if name is not None and not isinstance(name, str):
            record["issues"].append("pumjongNm_not_string")
        if isinstance(name, str):
            names.append(name)
        code = item.get("pumjongCd")
        if code is not None and not isinstance(code, (str, int)):
            record["issues"].append("pumjongCd_invalid_type")
        for field in ("totqty", "alltotqty"):
            if field in item:
                issue = _numeric_issue(item[field])
                if issue:
                    record["issues"].append(f"{field}_{issue}")
        quantity_issue = _numeric_issue(item.get("totqty"))
        if quantity_issue is None:
            result["positiveQuantityCount"] += float(item["totqty"]) > 0
        if record["issues"]:
            result["issues"].append(f"item_{index}_questionable")
        result["items"].append(record)
    result["itemCount"] = len(data)
    result["names"] = sorted(set(names))
    result["questionableItemCount"] = sum(bool(record["issues"]) for record in result["items"])
    result["itemIssueCount"] = sum(len(record["issues"]) for record in result["items"])
    result["cabbagePresent"] = any("배추" in name for name in names)
    result["radishPresent"] = any("무" in name for name in names)
    return result


def _save_attempt(output: Path, stem: str, request_params: dict[str, Any], captured: str,
                  response: Any, body: bytes, truncated: bool, error: str | None,
                  attempt: int) -> dict[str, Any]:
    headers = dict(response.headers) if response is not None else {}
    headers = {key: ("[redacted]" if key.lower() in {"set-cookie", "authorization"} else value)
               for key, value in headers.items()}
    status = response.status_code if response is not None else None
    raw_name = stem + ".raw" if body else None
    if body:
        _atomic_immutable(output / raw_name, body)
    validation = validate_response(body, status, truncated=truncated, transport_error=error)
    manifest = {"schemaVersion": 1, "endpoint": ENDPOINT,
                "requestParams": {"dueDate": request_params["dueDate"], "marketKind": request_params["marketKind"]},
                "market": request_params["marketName"], "attempt": attempt,
                "retrievedAtUtc": captured, "httpStatus": status,
                "responseHeaders": headers, "rawFile": raw_name,
                "responseBytes": len(body), "sha256": _sha(body) if body else None,
                "truncated": truncated, "transportError": error, "result": validation,
                "complete": False}
    manifest_name = stem + ".manifest.json"
    _atomic_immutable(output / manifest_name,
                      (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
    return manifest | {"manifestFile": manifest_name}


def collect(output: Path = DEFAULT_OUTPUT, *, session: Any = None,
            now=lambda: dt.datetime.now(dt.timezone.utc), today_kst=lambda: dt.datetime.now(KST).date(),
            sleep=time.sleep) -> list[dict[str, Any]]:
    if today_kst() != EXECUTION_DATE:
        raise CollectionError("execution_allowed_only_on_2026-10-08_KST")
    output.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    session.verify = True
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    manifests = []
    previous_start = None
    for plan in request_matrix():
        params = {"dueDate": plan["dueDate"], "marketKind": plan["marketKind"]}
        full_params = params | {"marketName": plan["market"]}
        attempts = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if today_kst() != EXECUTION_DATE:
                raise CollectionError("execution_allowed_only_on_2026-10-08_KST")
            if previous_start is not None:
                elapsed = (time.monotonic() - previous_start)
                if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
                    sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
            if today_kst() != EXECUTION_DATE:
                raise CollectionError("execution_allowed_only_on_2026-10-08_KST")
            previous_start = time.monotonic()
            response = None
            body = b""
            truncated = False
            error = None
            try:
                response = session.post(ENDPOINT, data=params, timeout=TIMEOUT_SECONDS,
                                        verify=True, stream=True)
                body, truncated, error = _read_bounded(response)
            except requests.RequestException as exc:
                error = f"request_error:{type(exc).__name__}:{exc}"
            captured, token = _stamp(now)
            stem = f"garak_reservation_{plan['market']}_{plan['dueDate']}_a{attempt:02d}_{token}"
            attempt_manifest = _save_attempt(output, stem, full_params, captured,
                                             response, body, truncated, error, attempt)
            attempts.append(attempt_manifest)
            status = attempt_manifest["httpStatus"]
            if status is not None and 200 <= status < 500 and status != 429 and not truncated and not error:
                break
            if attempt < MAX_ATTEMPTS:
                sleep(RETRY_DELAY_SECONDS)
        latest = attempts[-1]
        manifests.append({"market": plan["market"], "dueDate": plan["dueDate"],
                          "marketKind": plan["marketKind"], "attempts": attempts,
                          "state": latest["result"]["state"], "complete": False})
    return manifests


def _concise(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    return {"mode": "execute", "requestCount": len(manifests),
            "results": [{"market": item["market"], "dueDate": item["dueDate"],
                         "state": item["state"],
                         "attemptCount": len(item["attempts"]),
                         "itemCount": item["attempts"][-1]["result"]["itemCount"],
                         "positiveQuantityCount": item["attempts"][-1]["result"]["positiveQuantityCount"],
                         "questionableItemCount": item["attempts"][-1]["result"]["questionableItemCount"],
                         "itemIssueCount": item["attempts"][-1]["result"]["itemIssueCount"],
                         "names": item["attempts"][-1]["result"]["names"],
                         "cabbagePresent": item["attempts"][-1]["result"]["cabbagePresent"],
                         "radishPresent": item["attempts"][-1]["result"]["radishPresent"],
                         "namePresenceSemantics": item["attempts"][-1]["result"]["namePresenceSemantics"]}
                        for item in manifests]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="send the eight frozen aggregate requests")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps({"mode": "dry_run", "requestCount": 8,
                          "requests": request_matrix(), "requestSent": False}, ensure_ascii=False))
        return 0
    manifests = collect(args.output)
    print(json.dumps(_concise(manifests), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
