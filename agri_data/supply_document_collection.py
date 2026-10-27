"""Bounded collector for public supply announcements and crop survey originals.

The collector accepts an explicitly enumerated URL manifest. It never crawls
or discovers more URLs on its own. Original response bodies are stored
immutably; extracted text is a derived convenience copy.
"""
from __future__ import annotations

import argparse
import hashlib
import html
from html.parser import HTMLParser
import io
import importlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

PROTOCOL = "public-supply-document-capture-v1"
ALLOWED_HOSTS = frozenset({
    "mafra.go.kr", "www.mafra.go.kr", "jasdc.or.kr", "www.jasdc.or.kr",
    "agri.jeju.go.kr", "www.agri.jeju.go.kr",
})
DEFAULT_TIMEOUT_SECONDS = 25
DEFAULT_MAX_BYTES = 40 * 1024 * 1024
DEFAULT_RETRIES = 2
DEFAULT_DELAY_SECONDS = 1.0
MAX_DELAY_SECONDS = 10.0
MAX_RETRIES = 3
MAX_BUDGET_SECONDS = 1200
MAX_PDF_PAGES = 500
MAX_EXTRACTED_TEXT_BYTES = 10 * 1024 * 1024


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_immutable(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_EXCL prevents a resume or rerun from silently replacing captured evidence.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def write_immutable_idempotent(path: Path, data: bytes) -> None:
    """Atomically create immutable content, accepting identical prior captures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(temp_name, 0o444)
        try:
            os.link(temp_name, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError("immutable_capture_path_collision")
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.hostname.lower() not in ALLOWED_HOSTS:
        raise ValueError("url_outside_https_allowlist")
    if parsed.username or parsed.password or parsed.port not in (None, 443):
        raise ValueError("url_authority_not_allowed")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Reject every redirect, including same-origin redirects: the manifest
        # must name each exact endpoint so redirects cannot widen collection.
        raise HTTPError(req.full_url, code, "redirect_rejected", headers, fp)


def _read_manifest(path: Path) -> dict:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("protocol") != PROTOCOL:
        raise ValueError("manifest_protocol_mismatch")
    reqs = doc.get("requests")
    if not isinstance(reqs, list) or not reqs:
        raise ValueError("manifest_requests_missing")
    seen = set()
    for row in reqs:
        if not isinstance(row, dict) or not row.get("id") or not row.get("url"):
            raise ValueError("invalid_request_entry")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}", row["id"]):
            raise ValueError("request_id_not_path_safe")
        if row["id"] in seen:
            raise ValueError("duplicate_request_id")
        seen.add(row["id"])
        validate_url(row["url"])
        if row.get("kind") not in {"listing", "document", "reference"}:
            raise ValueError("invalid_request_kind")
        if row.get("method", "GET") != "GET":
            raise ValueError("only_exact_get_requests_are_allowed")
    # Pagination is represented by explicit page chains. The collector does
    # not infer that a range is complete from the pages that happened to load.
    for chain in doc.get("page_chains", []):
        pages = chain.get("pages", [])
        if not pages or [p.get("page_index") for p in pages] != list(range(1, len(pages) + 1)):
            raise ValueError("page_chain_indices_not_contiguous")
        for i, page in enumerate(pages):
            if page.get("request_id") not in seen:
                raise ValueError("page_chain_request_missing")
            req_by_id = next(r for r in reqs if r["id"] == page["request_id"])
            if req_by_id.get("kind") != "listing" or req_by_id["url"] != page.get("url"):
                raise ValueError("page_chain_request_url_mismatch")
            final = i == len(pages) - 1
            if final and page.get("has_next") is not False:
                raise ValueError("page_chain_end_not_proven")
            if not final:
                if page.get("has_next") is not True:
                    raise ValueError("page_chain_gap")
                if page.get("next_url") != pages[i + 1].get("url"):
                    raise ValueError("page_chain_next_url_mismatch")
    listing_ids = {r["id"] for r in reqs if r.get("kind") == "listing"}
    chained_ids = {p["request_id"] for c in doc.get("page_chains", []) for p in c.get("pages", [])}
    if listing_ids != chained_ids:
        raise ValueError("listing_requests_not_fully_chained")
    return doc


def _download(url: str, *, timeout: float, max_bytes: int, retries: int, delay: float,
              deadline: float | None = None) -> dict:
    validate_url(url)
    opener = build_opener(_NoRedirect())
    attempts = []
    for attempt in range(1, retries + 2):
        remaining = deadline - time.monotonic() if deadline is not None else timeout
        if remaining <= 0:
            return {"ok": False, "error": "time_budget_exhausted", "attempts": attempts, "retrieved_at": _now(), "headers": {}}
        started = _now()
        headers = {"User-Agent": "agri-predict-public-document-collector/1.0", "Accept": "text/html,application/pdf,*/*;q=0.5"}
        req = Request(url, headers=headers, method="GET")
        try:
            with opener.open(req, timeout=min(timeout, remaining)) as response:
                final_url = response.geturl()
                if final_url != url:
                    raise ValueError("response_url_changed")
                body_parts = []
                size = 0
                while True:
                    remaining = deadline - time.monotonic() if deadline is not None else timeout
                    if remaining <= 0:
                        raise ValueError("time_budget_exhausted_during_body")
                    sock = getattr(getattr(getattr(response, "fp", None), "raw", None), "_sock", None)
                    if sock is not None:
                        sock.settimeout(min(timeout, remaining))
                    chunk_reader = getattr(response, "read1", None) or response.read
                    chunk = chunk_reader(min(64 * 1024, max_bytes + 1 - size))
                    if not chunk:
                        break
                    body_parts.append(chunk)
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError("response_exceeds_max_bytes")
                body = b"".join(body_parts)
                result = {"ok": True, "status": response.status, "body": body,
                          "headers": dict(response.headers.items()), "final_url": final_url,
                          "retrieved_at": _now(), "attempts": attempts + [{"attempt": attempt, "started_at": started, "outcome": "http_200"}]}
                return result
        except HTTPError as exc:
            outcome = "redirect_rejected" if 300 <= exc.code < 400 else f"http_{exc.code}"
            retry_after = next((value for key, value in exc.headers.items() if key.lower() == "retry-after"), None) if exc.headers else None
            response_headers = dict(exc.headers.items()) if exc.headers else {}
            exc.close()
            attempts.append({"attempt": attempt, "started_at": started, "finished_at": _now(), "outcome": outcome})
            if exc.code in (429, 503) and retry_after:
                return {"ok": False, "status": exc.code, "error": "retry_deferred", "retry_after": retry_after,
                        "attempts": attempts, "retrieved_at": _now(), "headers": response_headers}
            # Only retry transient HTTP errors; redirects and ordinary client errors are final.
            if exc.code not in (408, 425, 429, 500, 502, 503, 504) or attempt > retries:
                return {"ok": False, "status": exc.code, "error": outcome, "attempts": attempts, "retrieved_at": _now(), "headers": response_headers}
        except (URLError, TimeoutError, OSError) as exc:
            attempts.append({"attempt": attempt, "started_at": started, "finished_at": _now(), "outcome": type(exc).__name__})
            if attempt > retries:
                return {"ok": False, "error": type(exc).__name__, "attempts": attempts, "retrieved_at": _now(), "headers": {}}
        except ValueError as exc:
            attempts.append({"attempt": attempt, "started_at": started, "finished_at": _now(), "outcome": str(exc)})
            return {"ok": False, "error": str(exc), "attempts": attempts, "retrieved_at": _now(), "headers": {}}
        except socket.timeout as exc:
            attempts.append({"attempt": attempt, "started_at": started, "finished_at": _now(), "outcome": "socket_timeout"})
            if attempt > retries:
                return {"ok": False, "error": "socket_timeout", "attempts": attempts, "retrieved_at": _now(), "headers": {}}
        pause = min(delay * attempt, MAX_DELAY_SECONDS)
        if deadline is not None:
            pause = min(pause, max(0.0, deadline - time.monotonic()))
        time.sleep(pause)
    raise AssertionError("retry loop exhausted")


def collect(manifest_path: Path, output_dir: Path, *, timeout: float = DEFAULT_TIMEOUT_SECONDS,
            max_bytes: int = DEFAULT_MAX_BYTES, retries: int = DEFAULT_RETRIES,
            delay: float = DEFAULT_DELAY_SECONDS, budget_seconds: float = 900) -> dict:
    numeric = (timeout, delay, budget_seconds, float(max_bytes), float(retries))
    if not all(math.isfinite(v) for v in numeric) or timeout <= 0 or max_bytes < 1 or retries < 0 or retries > MAX_RETRIES or delay < 0 or budget_seconds <= 0 or budget_seconds > MAX_BUDGET_SECONDS:
        raise ValueError("invalid_runtime_limits")
    manifest = _read_manifest(manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "checkpoint.json"
    scope_hash = sha256(manifest_path.read_bytes())
    checkpoint = {"protocol": PROTOCOL, "manifestSha256": scope_hash, "requests": {}}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("manifestSha256") != scope_hash:
            raise ValueError("checkpoint_manifest_mismatch")
    deadline = time.monotonic() + budget_seconds
    for row in manifest["requests"]:
        rid = row["id"]
        prior = checkpoint["requests"].get(rid)
        if prior and prior.get("status") == "captured":
            raw = output_dir / prior["raw_file"]
            if raw.exists() and sha256(raw.read_bytes()) == prior.get("sha256"):
                continue
            raise ValueError("checkpoint_raw_evidence_missing_or_changed")
        if time.monotonic() >= deadline:
            checkpoint["status"] = "interrupted_time_budget"
            break
        method = "GET"
        result = _download(row["url"], timeout=timeout, max_bytes=max_bytes, retries=retries, delay=delay, deadline=deadline)
        record = {"request": {k: v for k, v in row.items() if k != "url"}, "url": row["url"],
                  "request_method": method,
                  "request_headers": {"User-Agent": "agri-predict-public-document-collector/1.0", "Accept": "text/html,application/pdf,*/*;q=0.5"},
                  "retrieved_at": result["retrieved_at"], "attempts": result["attempts"],
                  "status": "captured" if result["ok"] else "failed", "http_status": result.get("status"),
                  "response_headers": result.get("headers", {}), "final_url": result.get("final_url"), "error": result.get("error")}
        if result["ok"]:
            hint = str(row.get("filename_extension", "")).lower().lstrip(".")
            if hint in {"hwp", "hwpx"}:
                ext = "." + hint
            else:
                ext = ".pdf" if "pdf" in result["headers"].get("Content-Type", "").lower() or result["body"].startswith(b"%PDF-") else ".bin"
            digest = sha256(result["body"])
            raw_name = f"originals/{rid}-{digest}{ext}"
            write_immutable_idempotent(output_dir / raw_name, result["body"])
            record.update({"raw_file": raw_name, "sha256": digest, "byte_length": len(result["body"])})
        checkpoint["requests"][rid] = record
        checkpoint["updated_at"] = _now()
        checkpoint["status"] = "retry_deferred" if result.get("error") == "retry_deferred" else "in_progress"
        write_checkpoint(checkpoint_path, checkpoint)
        if result.get("error") == "retry_deferred":
            break
        pause = min(delay, MAX_DELAY_SECONDS)
        time.sleep(min(pause, max(0.0, deadline - time.monotonic())))
    expected = {r["id"] for r in manifest["requests"]}
    captured = {rid for rid, r in checkpoint["requests"].items() if r.get("status") == "captured"}
    page_chains_complete = all(chain.get("enumeration_complete") is True for chain in manifest.get("page_chains", []))
    all_requests_complete = expected <= captured
    if all_requests_complete:
        checkpoint["status"] = "complete"
    elif checkpoint.get("status") not in {"interrupted_time_budget", "retry_deferred", "http_budget_exhausted"}:
        checkpoint["status"] = "incomplete"
    checkpoint["retrievalComplete"] = all_requests_complete
    checkpoint["enumeratedScopeComplete"] = bool(manifest.get("scope_enumeration_complete") is True and page_chains_complete and all_requests_complete)
    checkpoint["scopeLimitations"] = manifest.get("scope_limitations", [])
    checkpoint["updated_at"] = _now()
    write_checkpoint(checkpoint_path, checkpoint)
    return checkpoint


def write_checkpoint(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(_json_bytes(value))
    os.replace(tmp, path)


def _extract_pdf(body: bytes, raw_path: Path) -> tuple[str, str, dict]:
    try:
        pypdf = importlib.import_module("pypdf")
    except ImportError:
        pypdf = None
    if pypdf is not None:
        reader = pypdf.PdfReader(io.BytesIO(body), strict=False)
        total_pages = len(reader.pages)
        parts = []
        byte_count = 0
        pages_read = 0
        for page_index, page in enumerate(reader.pages):
            if page_index >= MAX_PDF_PAGES:
                break
            pages_read += 1
            text = page.extract_text() or ""
            encoded_size = len(text.encode("utf-8", errors="replace"))
            if byte_count + encoded_size > MAX_EXTRACTED_TEXT_BYTES:
                room = MAX_EXTRACTED_TEXT_BYTES - byte_count
                text = text.encode("utf-8", errors="replace")[:room].decode("utf-8", errors="ignore")
                parts.append(text)
                byte_count = MAX_EXTRACTED_TEXT_BYTES
                break
            parts.append(text)
            byte_count += encoded_size
        truncated = total_pages > pages_read or byte_count >= MAX_EXTRACTED_TEXT_BYTES
        return "\n\n".join(parts), "pypdf", {"pageCount": total_pages, "pagesExtracted": pages_read, "truncated": truncated}
    executable = shutil.which("pdftotext")
    if not executable:
        raise FileNotFoundError("pdf_text_tool_unavailable")
    proc = subprocess.run([executable, "-enc", "UTF-8", str(raw_path), "-"], capture_output=True, timeout=45, check=False)
    if proc.returncode:
        raise RuntimeError(f"pdftotext_failed_{proc.returncode}")
    if len(proc.stdout) > MAX_EXTRACTED_TEXT_BYTES:
        return proc.stdout[:MAX_EXTRACTED_TEXT_BYTES].decode("utf-8", errors="replace"), "pdftotext", {"truncated": True}
    return proc.stdout.decode("utf-8", errors="replace"), "pdftotext", {"truncated": False}


class _TextHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}:
            self.hidden += 1
        if tag in {"p", "br", "div", "li", "tr", "h1", "h2", "h3", "article"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.hidden:
            self.hidden -= 1
        if tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "article"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def extract(output_dir: Path) -> dict:
    checkpoint_path = output_dir / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    outcomes = []
    for rid, record in checkpoint.get("requests", {}).items():
        if record.get("status") != "captured":
            continue
        raw_path = output_dir / record["raw_file"]
        body = raw_path.read_bytes()
        if sha256(body) != record["sha256"]:
            raise ValueError("raw_hash_mismatch")
        content_type = record.get("response_headers", {}).get("Content-Type", "").lower()
        if body.startswith(b"%PDF-") or "application/pdf" in content_type:
            try:
                text, method, pdf_info = _extract_pdf(body, raw_path)
            except FileNotFoundError as exc:
                outcomes.append({"id": rid, "status": str(exc), "sourceSha256": record["sha256"]})
                continue
            except Exception as exc:
                outcomes.append({"id": rid, "status": "pdf_text_extraction_failed", "sourceSha256": record["sha256"], "error": type(exc).__name__})
                continue
        elif record.get("request", {}).get("filename_extension", "").lower().lstrip(".") in {"hwp", "hwpx"}:
            outcomes.append({"id": rid, "status": "hwp_text_extraction_unsupported", "sourceSha256": record["sha256"]})
            continue
        else:
            parser = _TextHTML()
            parser.feed(body.decode("utf-8", errors="replace"))
            text = html.unescape("".join(parser.parts))
            method = "stdlib-html-parser"
        text = re.sub(r"[\t\f\v ]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text).strip() + "\n"
        text_bytes = text.encode("utf-8")
        text_digest = sha256(text_bytes)
        text_path = output_dir / "derived_text" / f"{rid}-{text_digest}.txt"
        write_immutable_idempotent(text_path, text_bytes)
        outcomes.append({"id": rid, "status": "extracted", "sourceSha256": record["sha256"],
                         "textFile": str(text_path.relative_to(output_dir)), "textSha256": text_digest, "method": method,
                         **(pdf_info if body.startswith(b"%PDF-") or "application/pdf" in content_type else {})})
    result = {"protocol": PROTOCOL, "sourceManifestSha256": checkpoint.get("manifestSha256"), "outcomes": outcomes}
    result_bytes = _json_bytes(result)
    result_path = output_dir / f"extraction-{sha256(result_bytes)}.json"
    write_immutable_idempotent(result_path, result_bytes)
    result["extractionFile"] = result_path.name
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--manifest", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    c.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    c.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    c.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS)
    c.add_argument("--budget-seconds", type=float, default=900)
    e = sub.add_parser("extract")
    e.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = collect(args.manifest, args.output, timeout=args.timeout, max_bytes=args.max_bytes,
                         retries=args.retries, delay=args.delay, budget_seconds=args.budget_seconds) if args.command == "collect" else extract(args.output)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
