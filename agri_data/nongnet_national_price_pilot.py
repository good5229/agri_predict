"""Fixed eight-query raw HTML pilot for Nongnet nationwide average prices.

Dry-run is the default. The pilot preserves source responses and does not train
or score a model; a result manifest always keeps complete=false.
"""
from __future__ import annotations

import argparse
import datetime as dt
from html.parser import HTMLParser
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import uuid
from typing import Any

import requests

URL = "https://nongnet.or.kr/front/M000000259/marketInfo/nationWide.do"
MARKET_NAME = "*전국도매시장"
MARKET_CODE = "100000"
TARGETS = (
    ("2026-09-11", "배추", "1001", "", ""),
    ("2026-09-12", "배추", "1001", "", ""),
    ("2026-09-14", "배추", "1001", "", ""),
    ("2026-09-11", "무", "1101", "", ""),
    ("2026-09-11", "양파", "1201", "", ""),
    ("2026-09-11", "건고추", "1207", "", ""),
    ("2026-09-11", "마늘", "1209", "", ""),
    ("2026-09-11", "마늘", "1209", "깐마늘", "120906"),
)
TIMEOUT_SECONDS = (5, 25)
MAX_ATTEMPTS = 3
MAX_RESPONSE_BYTES = 5_000_000
RETRY_DELAY_SECONDS = 1.0
MIN_REQUEST_INTERVAL_SECONDS = 1.0
DEFAULT_OUTPUT = Path("data/raw/nongnet_national_price_pilot")
NULLS = {"", "-", "--", "null", "없음"}


class CollectionError(RuntimeError):
    pass


def _params(target: tuple[str, str, str, str, str]) -> dict[str, str]:
    date, item_name, item_code, variety_name, variety_code = target
    parsed = dt.date.fromisoformat(date)
    return {"searchDate": f"{parsed.year}년 {parsed.month:02d}월 {parsed.day:02d}일",
            "searchName1": MARKET_NAME, "searchSymbol1": MARKET_CODE,
            "searchName2": item_name, "searchSymbol2": item_code,
            "searchName3": variety_name, "searchSymbol3": variety_code}


def request_matrix() -> list[dict[str, str]]:
    return [{"targetDate": target[0], "itemName": target[1], "itemCode": target[2],
             "varietyName": target[3], "varietyCode": target[4], **_params(target)}
            for target in TARGETS]


class _Node:
    __slots__ = ("tag", "attrs", "children", "parts", "parent")

    def __init__(self, tag: str, attrs: dict[str, str], parent: "_Node | None"):
        self.tag, self.attrs, self.children, self.parts, self.parent = tag, attrs, [], [], parent

    @property
    def text(self) -> str:
        return "".join(self.parts) + "".join(child.text for child in self.children)


class _TreeParser(HTMLParser):
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = _Node("root", {}, None)
        self.stack = [self.root]
        self.inputs: dict[str, list[str]] = {}

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if tag == "input" and values.get("name"):
            self.inputs.setdefault(values["name"], []).append(values.get("value", ""))
        node = _Node(tag, values, self.stack[-1])
        self.stack[-1].children.append(node)
        if tag not in self.VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, 0, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data):
        self.stack[-1].parts.append(data)


def _descendants(node: _Node, *, tag: str | None = None, cls: str | None = None) -> list[_Node]:
    result = []
    for child in node.children:
        classes = child.attrs.get("class", "").split()
        if (tag is None or child.tag == tag) and (cls is None or cls in classes):
            result.append(child)
        result.extend(_descendants(child, tag=tag, cls=cls))
    return result


def _has_ancestor(node: _Node, tag: str, cls: str) -> bool:
    current = node.parent
    while current is not None:
        if current.tag == tag and cls in current.attrs.get("class", "").split():
            return True
        current = current.parent
    return False


def _text(node: _Node) -> str:
    return re.sub(r"\s+", " ", node.text).strip()


def _format_price(value: str) -> tuple[str, str | None]:
    source = value.strip()
    if source in NULLS:
        return source, "null_value"
    try:
        number = float(source.replace(",", "").replace("원", "").strip())
    except (TypeError, ValueError, OverflowError):
        return source, "non_numeric_value"
    if not math.isfinite(number):
        return source, "nonfinite_value"
    if number < 0:
        return source, "negative_value"
    return source, None


def parse_html(body: bytes, requested: dict[str, str], *, http_status: int = 200) -> dict[str, Any]:
    """Verify selected filters and isolate the top average 1kg card."""
    result: dict[str, Any] = {"state": None, "issues": [], "selected": None,
                              "average": None, "grades": {}}
    if not 200 <= http_status < 300:
        result.update(state="http_error")
        result["issues"].append(f"http_status_{http_status}")
        return result
    if not body:
        result["state"] = "empty_response"
        return result
    text = body.decode("utf-8-sig", errors="replace")
    if "<!doctype html" not in text.lower() and "<html" not in text.lower():
        result["state"] = "non_html_response"
        result["issues"].append("expected_html_document")
        return result
    lower = text.lower()
    if any(marker in lower for marker in ("오류가 발생했습니다", "접근이 거부되었습니다", "access denied", "internal server error")):
        result["state"] = "error_html"
        result["issues"].append("error_page_marker")
        return result
    parser = _TreeParser()
    try:
        parser.feed(text)
        parser.close()
    except Exception:
        result["state"] = "malformed_html"
        result["issues"].append("html_parse_error")
        return result
    visible = " ".join(_text(node) for node in _descendants(parser.root, cls="tradeWrap"))
    expected = {key: requested[key] for key in ("searchDate", "searchName1", "searchSymbol1",
                                                 "searchName2", "searchSymbol2", "searchName3", "searchSymbol3")}
    mismatches = {}
    for key, value in expected.items():
        observed = parser.inputs.get(key, [])
        if not observed or any(item != value for item in observed):
            mismatches[key] = {"expected": value, "observed": observed}
    title_nodes = _descendants(parser.root, cls="dataTit")
    heading = next(iter(_descendants(title_nodes[0], tag="h4")), None) if title_nodes else None
    visible_date = None
    visible_names: list[str] = []
    if heading is not None:
        date_nodes = _descendants(heading, cls="date")
        if date_nodes:
            visible_date = _text(date_nodes[0]).strip("[] ")
        visible_names = [_text(node) for node in _descendants(heading, tag="em") if _text(node)]
    if visible_date != expected["searchDate"]:
        mismatches["visibleDate"] = {"expected": expected["searchDate"], "observed": visible_date}
    expected_names = [expected["searchName1"], expected["searchName2"]]
    if expected["searchName3"]:
        expected_names.append(expected["searchName3"])
    if visible_names != expected_names:
        mismatches["visibleSelection"] = {"expected": expected_names, "observed": visible_names}
    result["selected"] = expected | {"visibleDate": visible_date, "visibleNames": visible_names,
                                    "visibleHeading": visible[:1200]}
    if mismatches:
        result["state"] = "selection_mismatch"
        result["issues"].append("selected_form_values_do_not_match_request")
        result["mismatches"] = mismatches
        return result
    if any(term in text for term in ("조회된 데이터가 없습니다", "검색결과가 없습니다", "검색된 데이터가 없습니다")) and "priceChartAll" not in text:
        result["state"] = "normal_empty"
        return result
    panels = _descendants(parser.root, cls="priceChartAll")
    if not panels:
        result["state"] = "missing_price_panel"
        result["issues"].append("priceChartAll_missing")
        return result
    card_nodes = [node for node in _descendants(parser.root, cls="item")
                  if _has_ancestor(node, "div", "priceChartAll")]
    for card in card_nodes:
        grade_nodes = _descendants(card, cls="grade")
        price_nodes = _descendants(card, cls="price")
        if not grade_nodes or not price_nodes:
            continue
        grade = _text(grade_nodes[0])
        strong = _descendants(price_nodes[0], tag="strong")
        spans = _descendants(price_nodes[0], tag="span")
        price = _text(strong[0]) if strong else ""
        unit = _text(spans[0]) if spans else ""
        result["grades"][grade] = {"priceSource": price, "unit": unit}
        if grade == "평균":
            preserved, issue = _format_price(price)
            result["average"] = {"priceSource": preserved, "unit": unit}
            if issue:
                result["issues"].append(f"average_price_{issue}")
            if unit != "1kg":
                result["issues"].append("average_unit_not_1kg")
    if result["average"] is None:
        result["state"] = "missing_average"
        result["issues"].append("average_grade_not_found_in_price_panel")
        return result
    result["state"] = "valid" if not result["issues"] else "questionable"
    return result


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _immutable(path: Path, payload: bytes) -> None:
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
    chunks, total, truncated, error = [], 0, False, None
    try:
        for chunk in response.iter_content(64 * 1024):
            if not chunk:
                continue
            remaining = MAX_RESPONSE_BYTES - total
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                truncated, error = True, "response_exceeded_5mb_limit"
                break
            chunks.append(chunk)
            total += len(chunk)
    except requests.RequestException as exc:
        truncated, error = True, f"stream_error:{type(exc).__name__}:{exc}"
    finally:
        response.close()
    return b"".join(chunks), truncated, error


def collect(output: Path = DEFAULT_OUTPUT, *, session: Any = None,
            now=lambda: dt.datetime.now(dt.timezone.utc), sleep=time.sleep) -> list[dict[str, Any]]:
    output.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    session.verify = True
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    results = []
    previous_start = None
    for target in TARGETS:
        requested = _params(target)
        attempts = []
        for attempt in range(1, MAX_ATTEMPTS + 1):
            if previous_start is not None:
                elapsed = time.monotonic() - previous_start
                if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
                    sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)
            previous_start = time.monotonic()
            response, body, truncated, error = None, b"", False, None
            try:
                response = session.get(URL, params=requested, timeout=TIMEOUT_SECONDS,
                                       verify=True, stream=True)
                body, truncated, error = _read_bounded(response)
            except requests.RequestException as exc:
                error = f"request_error:{type(exc).__name__}:{exc}"
            captured = now().astimezone(dt.timezone.utc).isoformat()
            token = captured.replace(":", "").replace("+00:00", "Z") + "_" + uuid.uuid4().hex[:8]
            stem = f"nongnet_national_{target[0]}_{target[2]}_{target[4] or 'all'}_a{attempt}_{token}"
            headers = dict(response.headers) if response is not None else {}
            headers = {k: ("[redacted]" if k.lower() in {"set-cookie", "authorization"} else v)
                       for k, v in headers.items()}
            status = response.status_code if response is not None else None
            raw_file = stem + ".html" if body else None
            if body:
                _immutable(output / raw_file, body)
            validation = (parse_html(body, requested, http_status=status or 0)
                          if not error and not truncated else
                          {"state": "truncated" if truncated else "transport_error",
                           "issues": [error], "selected": None, "average": None, "grades": {}})
            metadata = {"schemaVersion": 1, "endpoint": URL, "requestParams": requested,
                        "requestedTarget": {"date": target[0], "itemName": target[1],
                                            "itemCode": target[2], "varietyName": target[3],
                                            "varietyCode": target[4]},
                        "attempt": attempt, "retrievedAtUtc": captured,
                        "httpStatus": status, "responseHeaders": headers,
                        "rawFile": raw_file, "responseBytes": len(body),
                        "sha256": _sha(body) if body else None, "truncated": truncated,
                        "transportError": error, "validation": validation,
                        "complete": False}
            manifest_file = stem + ".manifest.json"
            _immutable(output / manifest_file,
                       (json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode())
            attempts.append({"manifestFile": manifest_file, **metadata})
            if status is not None and status < 500 and status != 429 and not truncated and not error:
                break
            if attempt < MAX_ATTEMPTS:
                sleep(RETRY_DELAY_SECONDS)
        results.append({"requestedTarget": attempts[-1]["requestedTarget"],
                        "attempts": attempts, "state": attempts[-1]["validation"]["state"],
                        "complete": False})
    return results


def _plan() -> dict[str, Any]:
    return {"mode": "dry_run", "requestSent": False, "requestCount": len(TARGETS),
            "endpoint": URL, "requests": request_matrix()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="send the eight frozen GET requests")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(_plan(), ensure_ascii=False))
        return 0
    results = collect(args.output)
    concise = [{"date": r["requestedTarget"]["date"], "item": r["requestedTarget"]["itemName"],
                "variety": r["requestedTarget"]["varietyName"], "state": r["state"],
                "attempts": len(r["attempts"]), "average": r["attempts"][-1]["validation"]["average"]}
               for r in results]
    print(json.dumps({"mode": "execute", "requestCount": len(results), "results": concise}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
