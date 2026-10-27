import json
import io
from pathlib import Path

import pytest

from agri_data import supply_document_collection as collector


def manifest(tmp_path, **extra):
    value = {
        "protocol": collector.PROTOCOL,
        "scope_id": "fixture-2022-2023",
        "scope_enumeration_complete": False,
        "scope_limitations": ["historical first-publication time unproven"],
        "requests": [{"id": "notice-1", "url": "https://www.mafra.go.kr/bbs/home/792/example/download.do",
                      "kind": "document", "crop": "cabbage", "document_date": "2022-04-01"}],
    }
    value.update(extra)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_url_validator_rejects_http_and_outside_redirect_hosts():
    for url in ("http://www.mafra.go.kr/file.pdf", "https://example.com/file.pdf",
                "https://www.mafra.go.kr.evil.test/file.pdf", "https://user@www.mafra.go.kr/file.pdf"):
        with pytest.raises(ValueError):
            collector.validate_url(url)


def test_manifest_requires_contiguous_explicit_chain_and_terminal_page(tmp_path):
    pages = [{"request_id": "p1", "page_index": 1, "url": "https://www.mafra.go.kr/list?page=1",
              "has_next": True, "next_url": "https://www.mafra.go.kr/list?page=2"},
             {"request_id": "p2", "page_index": 2, "url": "https://www.mafra.go.kr/list?page=2", "has_next": False}]
    rows = [{"id": "p1", "url": pages[0]["url"], "kind": "listing"},
            {"id": "p2", "url": pages[1]["url"], "kind": "listing"}]
    path = manifest(tmp_path, requests=rows, page_chains=[{"enumeration_complete": True, "pages": pages}])
    assert collector._read_manifest(path)["page_chains"][0]["pages"] == pages
    pages[-1]["has_next"] = True
    path.write_text(json.dumps({"protocol": collector.PROTOCOL, "requests": rows,
                                "page_chains": [{"enumeration_complete": True, "pages": pages}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="page_chain_end_not_proven"):
        collector._read_manifest(path)
    pages[-1]["has_next"] = False
    pages[0]["url"] = "https://www.mafra.go.kr/list?page=99"
    path.write_text(json.dumps({"protocol": collector.PROTOCOL, "requests": rows,
                                "page_chains": [{"enumeration_complete": True, "pages": pages}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="page_chain_request_url_mismatch"):
        collector._read_manifest(path)


def test_redirects_are_rejected_without_following(monkeypatch):
    class RedirectOpener:
        def open(self, req, timeout):
            raise collector.HTTPError(req.full_url, 302, "Found", {"Location": "https://example.com/"}, None)

    monkeypatch.setattr(collector, "build_opener", lambda *_: RedirectOpener())
    result = collector._download("https://www.mafra.go.kr/notice", timeout=1, max_bytes=100,
                                 retries=0, delay=0)
    assert result["ok"] is False
    assert result["error"] == "redirect_rejected"
    assert len(result["attempts"]) == 1


def test_collect_captures_original_bytes_hash_and_keeps_scope_incomplete(tmp_path, monkeypatch):
    source = manifest(tmp_path)
    body = b"%PDF-fixture original bytes"
    monkeypatch.setattr(collector, "_download", lambda *a, **k: {
        "ok": True, "status": 200, "body": body, "headers": {"Content-Type": "application/pdf"},
        "final_url": "https://www.mafra.go.kr/bbs/home/792/example/download.do",
        "retrieved_at": "2026-10-16T00:00:00Z", "attempts": [{"attempt": 1, "outcome": "http_200"}],
    })
    out = tmp_path / "raw"
    result = collector.collect(source, out, delay=0)
    record = result["requests"]["notice-1"]
    assert (out / record["raw_file"]).read_bytes() == body
    assert record["sha256"] == collector.sha256(body)
    assert result["retrievalComplete"] is True
    assert result["enumeratedScopeComplete"] is False
    with pytest.raises(FileExistsError):
        collector.write_immutable(out / record["raw_file"], b"replacement")


def test_resume_refuses_changed_manifest_and_changed_original(tmp_path, monkeypatch):
    source = manifest(tmp_path)
    body = b"original"
    monkeypatch.setattr(collector, "_download", lambda *a, **k: {
        "ok": True, "status": 200, "body": body, "headers": {}, "final_url": "https://www.mafra.go.kr/bbs/home/792/example/download.do",
        "retrieved_at": "2026-10-16T00:00:00Z", "attempts": [],
    })
    out = tmp_path / "raw"
    result = collector.collect(source, out, delay=0)
    raw_path = out / result["requests"]["notice-1"]["raw_file"]
    raw_path.chmod(0o644)
    raw_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checkpoint_raw_evidence_missing_or_changed"):
        collector.collect(source, out, delay=0)


def test_offline_html_extraction_is_derived_and_bound_to_source_hash(tmp_path):
    out = tmp_path / "raw"
    raw = b"<html><script>secret()</script><p>Notice &amp; release time</p></html>"
    raw_path = out / "originals" / "page.bin"
    collector.write_immutable(raw_path, raw)
    cp = {"protocol": collector.PROTOCOL, "manifestSha256": "a" * 64,
          "requests": {"listing": {"status": "captured", "raw_file": "originals/page.bin",
                       "sha256": collector.sha256(raw), "response_headers": {"Content-Type": "text/html; charset=utf-8"}}}}
    (out / "checkpoint.json").write_text(json.dumps(cp), encoding="utf-8")
    result = collector.extract(out)
    assert result["outcomes"][0]["sourceSha256"] == collector.sha256(raw)
    extracted = (out / result["outcomes"][0]["textFile"]).read_text(encoding="utf-8")
    assert "Notice & release time" in extracted
    assert "secret" not in extracted


def test_pdf_extraction_reports_missing_tool_without_modifying_original(tmp_path, monkeypatch):
    out = tmp_path / "raw"
    raw = b"%PDF-fixture"
    collector.write_immutable(out / "originals/doc.pdf", raw)
    (out / "checkpoint.json").write_text(json.dumps({"protocol": collector.PROTOCOL, "manifestSha256": "x",
        "requests": {"doc": {"status": "captured", "raw_file": "originals/doc.pdf", "sha256": collector.sha256(raw),
                              "response_headers": {"Content-Type": "application/pdf"}}}}), encoding="utf-8")
    def unavailable(*_): raise FileNotFoundError("pdf_text_tool_unavailable")
    monkeypatch.setattr(collector, "_extract_pdf", unavailable)
    result = collector.extract(out)
    assert result["outcomes"][0]["status"] == "pdf_text_tool_unavailable"
    assert (out / "originals/doc.pdf").read_bytes() == raw


def test_download_enforces_elapsed_budget_during_trickling_body(monkeypatch):
    clock = {"t": 0.0}

    class Response:
        status = 200
        headers = {"Content-Type": "text/html"}

        def __enter__(self): return self
        def __exit__(self, *_): pass
        def geturl(self): return "https://www.mafra.go.kr/page"
        def read1(self, _size):
            clock["t"] += 2
            return b"x"

    class Opener:
        def open(self, _request, timeout): return Response()

    monkeypatch.setattr(collector, "build_opener", lambda *_: Opener())
    monkeypatch.setattr(collector.time, "monotonic", lambda: clock["t"])
    result = collector._download("https://www.mafra.go.kr/page", timeout=10, max_bytes=100,
                                 retries=0, delay=0, deadline=1)
    assert result["ok"] is False
    assert result["error"] == "time_budget_exhausted_during_body"


def test_retry_after_is_deferred_and_http_error_closed(monkeypatch):
    class ClosedError(collector.HTTPError):
        closed = False
        def close(self):
            self.closed = True
            super().close()

    err = ClosedError("https://www.mafra.go.kr/page", 429, "Too Many", {"retry-after": "3600"}, io.BytesIO(b"body"))

    class Opener:
        def open(self, *_args, **_kwargs): raise err

    monkeypatch.setattr(collector, "build_opener", lambda *_: Opener())
    result = collector._download("https://www.mafra.go.kr/page", timeout=10, max_bytes=100,
                                 retries=2, delay=0)
    assert result["error"] == "retry_deferred"
    assert result["retry_after"] == "3600"
    assert err.closed is True


def test_id_validation_and_nonfinite_limits_are_rejected(tmp_path):
    source = manifest(tmp_path, requests=[{"id": "../escape", "url": "https://www.mafra.go.kr/x", "kind": "document"}])
    with pytest.raises(ValueError, match="request_id_not_path_safe"):
        collector._read_manifest(source)
    source = manifest(tmp_path)
    with pytest.raises(ValueError, match="invalid_runtime_limits"):
        collector.collect(source, tmp_path / "raw", timeout=float("nan"))
    with pytest.raises(ValueError, match="invalid_runtime_limits"):
        collector.collect(source, tmp_path / "raw", budget_seconds=1201)


def test_post_manifest_is_rejected(tmp_path):
    source = manifest(tmp_path, requests=[{"id": "search-page", "url": "https://www.mafra.go.kr/bbs/home/792/artclList.do",
        "kind": "listing", "method": "POST", "form_data": {"srchWrd": "배추"}}])
    with pytest.raises(ValueError, match="only_exact_get_requests_are_allowed"):
        collector._read_manifest(source)


def test_extraction_is_idempotent_and_content_versioned(tmp_path):
    out = tmp_path / "raw"
    raw = b"<p>First version</p>"
    collector.write_immutable(out / "originals/page.bin", raw)
    cp = {"protocol": collector.PROTOCOL, "manifestSha256": "b" * 64,
          "requests": {"listing": {"status": "captured", "raw_file": "originals/page.bin",
                       "sha256": collector.sha256(raw), "response_headers": {"Content-Type": "text/html"}}}}
    (out / "checkpoint.json").write_text(json.dumps(cp), encoding="utf-8")
    first = collector.extract(out)
    second = collector.extract(out)
    assert first["extractionFile"] == second["extractionFile"]


def test_pdf_extraction_prefers_pypdf_and_records_page_coverage(monkeypatch, tmp_path):
    class FakePage:
        def __init__(self, text): self.text = text
        def extract_text(self): return self.text

    class FakeReader:
        def __init__(self, *_args, **_kwargs): self.pages = [FakePage("release"), FakePage("cabbage 8-10%")]

    class FakeModule:
        PdfReader = FakeReader

    monkeypatch.setattr(collector.importlib, "import_module", lambda name: FakeModule() if name == "pypdf" else None)
    text, method, info = collector._extract_pdf(b"%PDF-fixture", tmp_path / "doc.pdf")
    assert method == "pypdf"
    assert "cabbage 8-10%" in text
    assert info == {"pageCount": 2, "pagesExtracted": 2, "truncated": False}


def test_retry_deferred_stops_following_requests(tmp_path, monkeypatch):
    source = manifest(tmp_path, requests=[
        {"id": "first", "url": "https://www.mafra.go.kr/one", "kind": "document"},
        {"id": "second", "url": "https://www.mafra.go.kr/two", "kind": "document"}],
        page_chains=[])
    calls = []
    monkeypatch.setattr(collector, "_download", lambda url, **_kwargs: calls.append(url) or {
        "ok": False, "status": 429, "error": "retry_deferred", "retry_after": "120",
        "retrieved_at": "now", "attempts": [{"attempt": 1, "outcome": "http_429"}], "headers": {"retry-after": "120"}})
    result = collector.collect(source, tmp_path / "raw", retries=2, delay=0)
    assert calls == ["https://www.mafra.go.kr/one"]
    assert result["status"] == "retry_deferred"
    assert result["retrievalComplete"] is False
