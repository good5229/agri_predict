import json

import pytest

from agri_data import supply_document_collection as collector
from agri_data import supply_document_discovery as discovery


def test_offline_discovery_emits_jeju_synap_pdf_candidate_and_rejects_krei(tmp_path):
    out = tmp_path / "listings"
    html = b'''<html><table><tr><td>2023-12-22 17:00</td><td>2023 weekly crop survey</td>
    <td><a href="#none" onclick="fn_callSynapViewer('/files/board/', 'abc-123.pdf', 'weekly.pdf')">Download</a></td></tr>
    <tr><td><a href="https://www.krei.re.kr/nope">KREI</a></td></tr></table></html>'''
    raw_path = out / "originals" / "jeju-page.bin"
    collector.write_immutable(raw_path, html)
    checkpoint = {"manifestSha256": "f" * 64, "requests": {"jeju-weekly-2023-p1": {
        "status": "captured", "request": {"kind": "listing", "expected_result_count": 1}, "url": "https://agri.jeju.go.kr/jadx/web/agriBbs/agriInfo/trendList.do",
        "raw_file": "originals/jeju-page.bin", "sha256": collector.sha256(html)}}}
    cp = out / "checkpoint.json"
    cp.write_text(json.dumps(checkpoint), encoding="utf-8")
    result = discovery.discover(cp, tmp_path / "candidates.json")
    assert result["candidate_count"] == 1
    item = result["candidates"][0]
    assert item["url"] == "https://agri.jeju.go.kr/files/board/abc-123.pdf"
    assert item["disposition"] == "candidate_requires_review"
    assert "2023-12-22 17:00" in item["row_context"]
    assert "2023 weekly crop survey" in item["row_context"]
    assert result["chain_checks"] == [{"chain_id": "jeju-weekly-2023", "expected_result_count": 1,
                                       "candidate_link_count": 1, "count_matches_expected": True}]


def test_offline_discovery_rejects_modified_listing_original(tmp_path):
    out = tmp_path / "listings"
    raw_path = out / "originals" / "page.bin"
    collector.write_immutable(raw_path, b'<a href="https://www.mafra.go.kr/x">doc</a>')
    cp = out / "checkpoint.json"
    cp.write_text(json.dumps({"requests": {"p1": {"status": "captured", "request": {"kind": "listing"},
        "url": "https://www.mafra.go.kr/list", "raw_file": "originals/page.bin", "sha256": "0" * 64}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="raw_hash_mismatch"):
        discovery.discover(cp, tmp_path / "candidates.json")


def test_discovery_handles_nested_mafra_download_and_preview_anchors(tmp_path):
    out = tmp_path / "details"
    raw = b'''<div class="file_list"><ul><li><a href="/bbs/home/792/562535/download.do">
    announcement.pdf&nbsp;&nbsp;(244 KB)<a target="_blank" href="/sn3hcv_v2023/view.jsp?filepath=/home/atchmnfl/bbs/792/file.pdf">preview</a></a></li></ul></div>'''
    collector.write_immutable(out / "originals/detail.bin", raw)
    cp = out / "checkpoint.json"
    cp.write_text(json.dumps({"manifestSha256": "f" * 64, "requests": {"detail": {
        "status": "captured", "request": {"kind": "reference", "source": "mafra"},
        "url": "https://www.mafra.go.kr/bbs/home/792/564189/artclView.do",
        "raw_file": "originals/detail.bin", "sha256": collector.sha256(raw)}}}), encoding="utf-8")
    result = discovery.discover(cp, tmp_path / "candidates.json")
    assert result["candidate_count"] == 1
    assert result["candidates"][0]["url"] == "https://www.mafra.go.kr/bbs/home/792/562535/download.do"
    assert "announcement.pdf" in result["candidates"][0]["row_context"]


def test_document_manifest_binds_candidates_to_listing_hash_and_context(tmp_path):
    raw = b'''<table><tr><td>2023-12-22 17:00</td><td>2023 weekly crop survey</td>
      <td><a href="#none" onclick="fn_callSynapViewer('/files/board/', 'weekly.pdf', 'weekly.pdf')">PDF</a></td></tr></table>'''
    out = tmp_path / "listings"
    collector.write_immutable(out / "originals/p1.bin", raw)
    cp = out / "checkpoint.json"
    cp.write_text(json.dumps({"manifestSha256": "a" * 64, "requests": {"jeju-weekly-2023-p1": {
        "status": "captured", "url": "https://agri.jeju.go.kr/jadx/web/agriBbs/agriInfo/trendList.do?pageIndex=1",
        "request": {"kind": "listing", "source": "jeju-agri-extension", "topic": "weekly", "year": 2023,
                    "page_index": 1, "expected_result_count": 1},
        "raw_file": "originals/p1.bin", "sha256": collector.sha256(raw)}}}), encoding="utf-8")
    candidate_file = tmp_path / "candidates.json"
    candidate_doc = discovery.discover(cp, candidate_file)
    manifest = discovery.build_document_manifest(candidate_file, tmp_path / "documents.json")
    assert manifest["expected_request_count"] == 1
    request = manifest["requests"][0]
    assert request["kind"] == "document"
    assert request["source_page_sha256"] == collector.sha256(raw)
    assert "2023-12-22 17:00" in request["row_context"]
    assert manifest["candidate_list_sha256"] == collector.sha256(candidate_file.read_bytes())
