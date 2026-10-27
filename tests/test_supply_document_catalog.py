import json
from pathlib import Path

import pytest

from agri_data import supply_document_catalog as catalog
from agri_data import supply_document_collection as collector


def test_catalog_preserves_dates_origin_flags_and_excludes_yangbaechu_keyword_match(tmp_path):
    root = tmp_path / "capture"
    raw = b"%PDF placeholder"
    collector.write_immutable(root / "originals/doc.pdf", raw)
    text = ("조사일: 2023.12.01 월동무 수확 진행률 8~10%, 390ha. "
            "양배추 가격과 배추 공급계획, 무) 동향. 농업관측정보를 참고함.").encode()
    collector.write_immutable(root / "derived_text/doc.txt", text)
    cp = {"manifestSha256": "a" * 64, "requests": {"doc": {
        "status": "captured", "request": {"kind": "document", "source": "jeju-agri-extension",
            "link_text": "weekly bulletin", "row_context": "2023년 주간동향 2023-12-22 17:00"},
        "url": "https://agri.jeju.go.kr/files/board/doc.pdf", "raw_file": "originals/doc.pdf",
        "sha256": collector.sha256(raw), "byte_length": len(raw), "retrieved_at": "2026-10-16T00:00:00Z"}}}
    extraction = {"sourceManifestSha256": "a" * 64, "outcomes": [{"id": "doc", "status": "extracted",
        "sourceSha256": collector.sha256(raw), "textFile": "derived_text/doc.txt",
        "textSha256": collector.sha256(text), "method": "pypdf", "truncated": False}]}
    cp_path = root / "checkpoint.json"
    ex_path = root / "extraction.json"
    cp_path.write_text(json.dumps(cp), encoding="utf-8")
    ex_path.write_text(json.dumps(extraction), encoding="utf-8")
    result = catalog.build_catalog(cp_path, ex_path, tmp_path / "catalog.json")
    entry = result["entries"][0]
    assert entry["report_date"] == "2023.12.01"
    assert entry["listing_date"] == "2023-12-22"
    assert entry["crop_keyword_hits"] == {"cabbage": True, "radish": True, "winter_radish": True}
    assert entry["quantity_type_hits"] == ["harvest_progress", "stock_release_or_contract"]
    assert entry["krei_flag"] is True
    assert entry["source_origin_review_required"] is True
    assert entry["first_publication_time"] is None
    assert entry["historical_availability"] == "unknown_current_retrieval_only"
    assert entry["model_input_status"] == "discovery_only_not_approved"
    assert catalog._excerpts("양배추 생산", catalog._KEYWORD_PATTERNS["cabbage"]) == []


def test_catalog_rejects_extraction_bound_to_another_raw_source(tmp_path):
    root = tmp_path / "capture"
    raw = b"original bytes"
    collector.write_immutable(root / "originals/doc.bin", raw)
    cp = {"manifestSha256": "b" * 64, "requests": {"doc": {"status": "captured", "request": {},
        "url": "https://www.mafra.go.kr/doc", "raw_file": "originals/doc.bin", "sha256": collector.sha256(raw)}}}
    extraction = {"sourceManifestSha256": "b" * 64, "outcomes": [{"id": "doc", "status": "extracted",
        "sourceSha256": "0" * 64}]}
    cp_path, ex_path = root / "checkpoint.json", root / "extraction.json"
    cp_path.write_text(json.dumps(cp), encoding="utf-8")
    ex_path.write_text(json.dumps(extraction), encoding="utf-8")
    with pytest.raises(ValueError, match="extracted_source_hash_mismatch"):
        catalog.build_catalog(cp_path, ex_path, tmp_path / "catalog.json")


def test_attachment_catalog_uses_source_listing_context_for_title_and_date(tmp_path):
    root = tmp_path / "capture"
    raw = b"%PDF placeholder"
    collector.write_immutable(root / "originals/attachment.pdf", raw)
    cp = {"manifestSha256": "d" * 64, "requests": {"attachment": {
        "status": "captured", "request": {"kind": "document", "source": "mafra",
            "filename_hint": "supply.pdf", "source_list_row_context": "2024.01.15 겨울무 수급 대책"},
        "url": "https://www.mafra.go.kr/bbs/home/792/1/download.do",
        "raw_file": "originals/attachment.pdf", "sha256": collector.sha256(raw)}}}
    extraction = {"sourceManifestSha256": "d" * 64, "outcomes": []}
    cp_path, ex_path = root / "checkpoint.json", root / "extraction.json"
    cp_path.write_text(json.dumps(cp), encoding="utf-8")
    ex_path.write_text(json.dumps(extraction), encoding="utf-8")
    entry = catalog.build_catalog(cp_path, ex_path, tmp_path / "catalog.json")["entries"][0]
    assert entry["source_title"] == "2024.01.15 겨울무 수급 대책"
    assert entry["source_listing_row"] == "2024.01.15 겨울무 수급 대책"
    assert entry["listing_date"] == "2024.01.15"
    assert entry["first_publication_time"] is None


def test_attachment_builder_prefers_pdf_and_keeps_hwp_fallback(tmp_path):
    root = tmp_path / "details"
    html = b"<p>detail</p>"
    collector.write_immutable(root / "originals/detail1.bin", html)
    cp = {"manifestSha256": "b" * 64, "requests": {
        "mafra-detail-pdf": {"status": "captured", "url": "https://www.mafra.go.kr/bbs/home/792/1/artclView.do",
            "raw_file": "originals/detail1.bin", "sha256": collector.sha256(html), "request": {"source": "mafra", "candidate_kind": "detail_page", "topic": "cabbage"}},
        "mafra-detail-hwp": {"status": "captured", "url": "https://www.mafra.go.kr/bbs/home/792/2/artclView.do",
            "raw_file": "originals/detail1.bin", "sha256": collector.sha256(html), "request": {"source": "mafra", "candidate_kind": "detail_page", "topic": "radish"}}}}
    cp_path = root / "checkpoint.json"
    cp_path.write_text(json.dumps(cp), encoding="utf-8")
    candidates = {"protocol": "offline-supply-link-discovery-v1", "source_checkpoint": str(cp_path), "candidates": [
        {"candidate_id": "pdf-option", "source_request_id": "mafra-detail-pdf", "url": "https://www.mafra.go.kr/bbs/home/792/10/download.do", "link_text": "release.pdf", "row_context": "release.pdf", "source_page_sha256": collector.sha256(html)},
        {"candidate_id": "hwp-option", "source_request_id": "mafra-detail-pdf", "url": "https://www.mafra.go.kr/bbs/home/792/11/download.do", "link_text": "release.hwp", "row_context": "release.hwp", "source_page_sha256": collector.sha256(html)},
        {"candidate_id": "hwp-fallback", "source_request_id": "mafra-detail-hwp", "url": "https://www.mafra.go.kr/bbs/home/792/12/download.do", "link_text": "release.hwpx", "row_context": "release.hwpx", "source_page_sha256": collector.sha256(html)}]}
    candidate_path = tmp_path / "candidates.json"
    candidate_path.write_text(json.dumps(candidates), encoding="utf-8")
    result = catalog.build_attachment_manifest(candidate_path, tmp_path / "attachments.json")
    assert {r["id"] for r in result["requests"]} == {"pdf-option", "hwp-fallback"}
    assert result["detail_page_count"] == 2
    assert result["expected_request_count"] == 2
    assert result["parent_decisions"][0]["selection"] in {"pdf", "hwp_fallback"}


def test_hwp_original_is_preserved_and_extraction_is_marked_unsupported(tmp_path):
    out = tmp_path / "raw"
    raw = b"HWP original bytes"
    collector.write_immutable(out / "originals/file.hwp", raw)
    (out / "checkpoint.json").write_text(json.dumps({"protocol": collector.PROTOCOL, "manifestSha256": "c",
        "requests": {"hwp": {"status": "captured", "raw_file": "originals/file.hwp", "sha256": collector.sha256(raw),
            "response_headers": {"Content-Type": "application/octet-stream"}, "request": {"filename_extension": "hwp"}}}}), encoding="utf-8")
    result = collector.extract(out)
    assert result["outcomes"][0]["status"] == "hwp_text_extraction_unsupported"
    assert (out / "originals/file.hwp").read_bytes() == raw
