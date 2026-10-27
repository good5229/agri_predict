import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from agri_data import supply_document_eval_audit as a


def _cp(root, observations, *, status="complete"):
    root.mkdir(parents=True, exist_ok=True)
    scope = {"scope_mode": "fixed_window", "window_start": "2024-01-01", "window_end": "2026-09-30",
             "jeju_years": [2024, 2025, 2026], "queries": ["배추", "무", "수급"]}
    (root / "checkpoint.json").write_text(json.dumps({"run_id": a.RUN_ID, "status": status, "scope": scope,
        "observations": observations}), encoding="utf-8")
    (root / "run.json").write_text(json.dumps({"run_id": a.RUN_ID, "status": status}), encoding="utf-8")


def _obs(root, url, kind, body, metadata, *, status=200):
    sha = hashlib.sha256(body).hexdigest()
    object_root = root.parent.parent if root.parent.name == "runs" else root
    path = object_root / "objects" / sha[:2] / sha
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return {"url": url, "kind": kind, "metadata": metadata, "status": status, "ok": True,
        "headers": {"content-type": metadata.get("content_type", "text/html")},
        "started_at": "2026-10-16T01:00:00Z", "retrieved_at": "2026-10-16T01:00:01Z",
        "sha256": sha, "byte_length": len(body), "object_path": str(path.relative_to(object_root))}


def test_integrity_audit_checks_every_captured_body_and_keeps_html_and_pdf(tmp_path):
    root = tmp_path / "run"
    pdf = _obs(root, "https://agri.jeju.go.kr/files/board/a.pdf", "document", b"%PDF-fixture",
               {"source": "jeju-weekly", "kind": "pdf", "content_type": "application/pdf"})
    detail = _obs(root, "https://www.mafra.go.kr/bbs/home/792/1/artclView.do", "detail", b"<p>page</p>",
                  {"source": "mafra-title"})
    _cp(root, [pdf, detail])
    result = a.audit_integrity(root, expected_observations=None)
    assert result["integrity_pass"] is True
    assert result["observation_count"] == 2
    assert result["pdf_observation_count"] == 1
    assert result["verified_capture_count"] == 2
    assert result["observations"][1]["doc_id"].startswith("mafra-eval-")


def test_production_layout_resolves_corpus_root_objects_and_detail_sha_link(tmp_path, monkeypatch):
    root = tmp_path / "collection" / "runs" / a.RUN_ID
    detail = _obs(root, "https://www.mafra.go.kr/bbs/home/792/579316/artclView.do", "detail",
                  "<p>등록일 2025. 4. 17.</p>".encode(), {"source": "mafra-title"})
    pdf = _obs(root, "https://www.mafra.go.kr/bbs/home/792/600799/download.do", "document", b"%PDF-fixture",
               {"source": "mafra-title", "kind": "pdf", "content_type": "application/pdf",
                "detail_sha256": detail["sha256"]})
    _cp(root, [detail, pdf])
    integrity = a.audit_integrity(root, expected_observations=None)
    docs, details, links = a._observation_docs(integrity)
    assert len(details) == 1 and docs[0]["detail_document_id"] == next(iter(details))
    assert docs[0]["parent_detail_link_status"] == "matched_by_detail_sha256"
    assert links["matched_unique"] == 1
    assert a._object_path(root, pdf).read_bytes() == b"%PDF-fixture"
    monkeypatch.setattr(a, "_pdf_info_page_count", lambda *args, **kwargs: 1)
    monkeypatch.setattr(a, "_extract_pdf_pages", lambda *args, **kwargs: (1, [""], None))
    out = tmp_path / "audit"
    a.build_review_queue(root, out, render=False, expected_observations=None)
    cues = json.loads((out / "detail-timestamp-candidates.json").read_text())
    assert cues["details"][0]["html_timestamp_candidates"][0]["source_hash"] == detail["sha256"]
    assert cues["pdf_parent_detail_links"]["matched_unique"] == 1


def test_html_timestamp_candidates_use_integrity_hash_aliases():
    body = "<p>게시일 2025. 4. 17.</p>".encode()
    row = {"doc_id": "mafra-eval-html", "url": "https://www.mafra.go.kr/a", "sha256_actual": hashlib.sha256(body).hexdigest()}
    cue = a._html_timestamp_candidates(body, row)[0]
    assert cue["source_hash"] == row["sha256_actual"]
    assert "2025. 4. 17." in cue["raw_quote"]


def test_pdf_extraction_preserves_page_boundaries_and_errors(tmp_path, monkeypatch):
    pdf = tmp_path / "fixture.pdf"
    pdf.write_bytes(b"%PDF-fixture")
    calls = []
    def runner(argv, **kwargs):
        calls.append(argv)
        if "pdfinfo" in argv[0]:
            return subprocess.CompletedProcess(argv, 0, b"Pages:          2\n", b"")
        return subprocess.CompletedProcess(argv, 0, "1페이지\f배추 생산량 50톤".encode(), b"")
    monkeypatch.setattr(a, "_tool", lambda name: name)
    pages, texts, error = a._extract_pdf_pages(pdf, runner=runner)
    assert pages == 2
    assert texts == ["1페이지", "배추 생산량 50톤"]
    assert error is None
    assert [Path(x[0]).name for x in calls] == ["pdfinfo", "pdftotext"]


def test_review_queue_writes_page_level_error_and_candidate_without_render(tmp_path, monkeypatch):
    root = tmp_path / "run"
    pdf = _obs(root, "https://www.mafra.go.kr/bbs/home/792/7/download.do", "document", b"%PDF-fixture",
               {"source": "mafra-title", "kind": "pdf", "content_type": "application/pdf"})
    _cp(root, [pdf])
    monkeypatch.setattr(a, "_pdf_info_page_count", lambda *args, **kwargs: 2)
    monkeypatch.setattr(a, "_extract_pdf_pages", lambda *args, **kwargs: (2, ["보도시점 2025.4.17\n배추 수급 20톤"], "page_count_mismatch:pdfinfo=2:text=1"))
    out = tmp_path / "audit"
    result = a.build_review_queue(root, out, render=False, expected_observations=None)
    assert result["pdf_count"] == 1 and result["pdf_pages_indexed"] == 2
    pages = json.loads((out / "pdf-page-index.json").read_text())
    doc = pages["documents"][0]
    assert doc["page_checks"] == [
        {"page": 1, "status": "extracted", "text_sha256": doc["page_hashes"][0]["text_sha256"]},
        {"page": 2, "status": "extraction_error", "error": "page_count_mismatch:pdfinfo=2:text=1"}]
    queue = json.loads((out / "page-candidates.json").read_text())
    assert queue["page_candidates"][0]["page_candidate_status"] == "unreviewed_candidate_not_fact"
    assert queue["page_candidates"][0]["page_image"] is None
    assert "page_text" not in queue["page_candidates"][0]


def test_review_queue_marks_blank_extracted_page_unsearchable(tmp_path, monkeypatch):
    root = tmp_path / "run"
    pdf = _obs(root, "https://www.mafra.go.kr/bbs/home/792/7/download.do", "document", b"%PDF-fixture",
               {"source": "mafra-title", "kind": "pdf", "content_type": "application/pdf"})
    _cp(root, [pdf])
    monkeypatch.setattr(a, "_pdf_info_page_count", lambda *args, **kwargs: 1)
    monkeypatch.setattr(a, "_extract_pdf_pages", lambda *args, **kwargs: (1, [""], None))
    result = a.build_review_queue(root, tmp_path / "audit", render=False, expected_observations=None)
    pages = json.loads((tmp_path / "audit" / "pdf-page-index.json").read_text())
    assert pages["documents"][0]["page_checks"][0]["status"] == "empty_text_unsearchable"
    assert result["empty_text_unsearchable_page_count"] == 1


def test_pdf_page_candidates_are_unreviewed_and_never_parse_numeric_values():
    page = "가을배추 수급 동향\n정부 비축량\n방출계획 2,000~3,000톤\n가격 10,000원/kg"
    doc = {"doc_id": "mafra-eval-123", "source": "mafra-title", "url": "https://www.mafra.go.kr/a",
           "sha256": "abc"}
    candidate = a._page_candidate(page, 1, doc)
    assert candidate["page_candidate_status"] == "unreviewed_candidate_not_fact"
    assert candidate["numeric_span_candidates"]
    assert all(x["value_candidate"] is None and x["unit_candidate"] is None for x in candidate["numeric_span_candidates"])
    assert candidate["crop_candidates"] == ["cabbage"]


def test_government_release_page_is_candidate_without_automatic_fact():
    doc = {"doc_id": "mafra-eval-123", "source": "mafra-title", "url": "https://www.mafra.go.kr/a",
           "sha256": "abc"}
    candidate = a._page_candidate("보도시점 2025. 4. 17.(목) 배포 시\n농림축산식품부", 1, doc)
    assert candidate is not None
    assert candidate["candidate_reasons"] == ["government_release_or_embargo_first_page"]
    assert candidate["timestamp_candidates"][0]["review_status"] == "unverified_source_timestamp_candidate"


def test_timestamp_quote_requires_source_date_and_clock_when_present():
    ev = {"timestamp_text_candidate": "2025. 4. 17.(목) 14:00", "raw_quote": "보도시점 2025. 4. 17.(목) 14:00 배포 시"}
    assert a._timestamp_quote_matches("2025-04-17T14:00:00+09:00", ev)
    assert not a._timestamp_quote_matches("2025-04-17T16:00:00+09:00", ev)
    assert a._timestamp_quote_matches("2025-04-17", {"timestamp_text_candidate": "2025. 4. 17.", "raw_quote": "작성일 2025. 4. 17."})


def test_krei_is_excluded_at_fact_level_not_entire_government_pdf():
    assert a._explicit_krei_fact({"source_org": "KREI"})
    assert a._explicit_krei_fact({"source_org": "농림축산식품부", "origin_review_status": "source_attribution_krei"})
    assert not a._explicit_krei_fact({"source_org": "농림축산식품부", "provenance_origin": "Separate KREI statistic excluded; this fact is a government reserve release"})


def test_origin_support_uses_frozen_origin_and_preserves_unknown_applicability():
    fact = {"fact_id": "f1", "update_group_id": "u1", "crop": "radish", "document_id": "d1",
        "publish_eligible_at_scenario": "2024-01-02T00:00:00+09:00", "asof_eligibility": "retrospective_publication_scenario",
        "effective_start": None, "effective_end": None, "series_admissibility": None,
        "horizon_relevance": "unknown_no_documented_effective_interval"}
    targets = [{"crop": "radish", "horizon": 2, "targetStart": "2024-01-10", "targetEnd": "2024-01-20",
        "origin": "2024-01-02", "inputCutoff": "2024-01-01", "evaluation_period": "evaluation_2024_2026-09"}]
    result = a.compute_origin_support([fact], targets)
    row = result["support_rows"][0]
    assert row["scenario_known_past_fact"] is True
    assert row["strict_historical_asof"] is False
    assert row["target_overlap_status"] == "overlap_unknown"
    assert row["is_new_update_at_scenario_origin"] is True


def test_gates_fail_closed_for_missing_crop_horizon_year_cells():
    gates = a.evaluate_gates([], {"release_origin_support": [], "support_rows": []}, integrity_pass=True,
        review_scope_complete=True, evaluation_documents_same_scope=True, target_mapping={"cabbage": "unknown", "radish": "unknown"})
    assert gates["G1_integrity_and_review"]["pass"] is False
    assert gates["G2_historical_asof"]["pass"] is False
    assert gates["G3_development_events_and_G6_applicability"]["pass"] is False
    assert gates["G4_evaluation_support"]["pass"] is False
    assert gates["all_pre_fit_gates_pass"] is False


def test_g4_target_count_is_unique_frozen_target_pairs():
    support_rows = []
    for period in ["evaluation_2024_2026-09"]:
        for year in range(2024, 2027):
            for ix in range(12):
                support_rows.append({"evaluation_period": period, "crop": "cabbage", "horizon": 2,
                    "target_year": str(year), "target_start": f"{year}-{ix+1:02d}-01", "target_end": f"{year}-{ix+1:02d}-14"})
    rel = [{"evaluation_period": "evaluation_2024_2026-09", "crop": "cabbage", "horizon": 2,
            "target_year": str(y), "first_new_update_origins": [f"{y}-{m:02d}-01" for m in range(1, 9)], "first_new_update_origin_count": 8}
           for y in range(2024, 2027)]
    gates = a.evaluate_gates([], {"release_origin_support": rel, "support_rows": support_rows}, integrity_pass=True,
        review_scope_complete=True, evaluation_documents_same_scope=True, target_mapping={"cabbage": "unknown", "radish": "unknown"})
    cabbage_h2 = next(x for x in gates["G4_evaluation_support"]["cells"] if x["crop"] == "cabbage" and x["horizon"] == 2)
    assert cabbage_h2["comparison_target_count"] == 36
    assert cabbage_h2["g4_pass"] is True
    assert gates["G4_evaluation_support"]["pass"] is False
    assert gates["G4_evaluation_support"]["partial_cohort_counts_are_lower_bound_only"] is True


def test_phase_a_support_is_carried_forward_with_failed_applicability_intact():
    from agri_data import period_supply_document_v8 as prior
    rows, provenance = a._load_phase_a_support()
    prior._mark_first_update_origins(rows)
    support = a._summarize_origin_support(rows)
    cell = next(r for r in support["release_origin_support"] if r["evaluation_period"] == "development_2022_2023"
        and r["target_year"] == "2022" and r["crop"] == "cabbage" and r["horizon"] == 2)
    assert provenance["frozen_prior_support_rows"] > 0
    assert cell["first_new_update_count"] == 7 and cell["first_new_update_origin_count"] == 6
    assert cell["documented_effective_interval_fact_origin_count"] == 6
    assert cell["known_effective_overlap_non_event_origin_count"] == 5


def test_normalize_review_requires_exact_hash_and_raw_time_quote(tmp_path, monkeypatch):
    body = b"%PDF-fixture"
    digest = hashlib.sha256(body).hexdigest()
    doc = {"doc_id": "mafra-eval-one", "url": "https://www.mafra.go.kr/bbs/home/792/7/download.do",
           "sha256_actual": digest, "sha256_expected": digest, "retrieved_at": "2026-10-16T01:00:00Z", "is_pdf": True}
    integrity = {"integrity_pass": True, "observations": [doc]}
    integ_path = tmp_path / "integrity.json"; integ_path.write_text(json.dumps(integrity))
    cand_path = tmp_path / "candidates.json"; cand_path.write_text(json.dumps({"page_candidates": []}))
    page_path = tmp_path / "pages.json"; page_path.write_text(json.dumps({"documents": [{"doc_id": "mafra-eval-one", "page_count": 1}]}))
    evidence = {"field": "release_at", "source_document_id": "mafra-eval-one", "source_hash": digest, "source_page": 1,
                "timestamp_text_candidate": "2025. 4. 17.(목) 14:00", "raw_quote": "보도시점 2025. 4. 17.(목) 14:00 배포 시"}
    common = {"document_id": "mafra-eval-one", "source_hash": digest, "page": 1, "evidence_span": "정부 비축물량 방출 90톤",
        "numeric_token": "90톤", "value": 90, "unit": "tonnes", "quantity_kind": "actual_release", "crop": "cabbage",
        "evidence_status": "agent_visual_verified", "posted_at": None, "release_at": "2025-04-17T14:00:00+09:00",
        "distribution_at": None, "availability_evidence": [evidence], "version_status": "current_capture_historical_bytes_unproven",
        "origin_review_status": "government_own_action_explicit", "region": "대한민국", "source_org": "농림축산식품부",
        "provenance_origin": "government_policy_announcement", "update_group_id": "u1", "target_mapping_status": "unknown"}
    review = {"cohort_status": "complete_selected_cohort", "reviewer_kind": "agent_visual", "facts": [
        {"fact_id": "gov-fact", **common},
        {"fact_id": "krei-fact", **{**common, "source_org": "KREI", "origin_review_status": "source_attribution_krei", "update_group_id": "uk"}},
        {"fact_id": "krei-archived-fact", **{**common, "source_org": "KREI", "origin_review_status": "source_attribution_krei",
          "update_group_id": "uk-archived", "version_status": "strict_historical_asof_verified",
          "historical_version_evidence": "missing-archive.pdf", "historical_version_evidence_sha256": "bad",
          "asof_eligibility": "strict_historical_asof"}}]}
    review_path = tmp_path / "review.json"; review_path.write_text(json.dumps(review, ensure_ascii=False))
    normalized = a.normalize_manual_facts(review_path, integ_path, cand_path, page_path)
    facts = normalized["reviewed_facts"]
    assert facts[0]["asof_eligibility"] == "retrospective_publication_scenario"
    assert facts[0]["publish_eligible_at_scenario"].startswith("2025-04-17T14:00:00")
    assert facts[0]["first_seen_at"] == doc["retrieved_at"]
    assert facts[0]["availability_evidence"][0]["raw_quote"] == evidence["raw_quote"]
    assert facts[1]["asof_eligibility"] == "ineligible"
    assert facts[2]["asof_eligibility"] == "ineligible"
    assert normalized["fact_counts"]["explicit_krei_fact_count"] == 2
    assert normalized["automatic_candidates_promoted"] == 0


def test_manual_review_rejects_date_without_hash_associated_raw_quote(tmp_path):
    body = b"%PDF-fixture"; digest = hashlib.sha256(body).hexdigest()
    integ = tmp_path / "i.json"; integ.write_text(json.dumps({"observations": [{"doc_id": "d", "url": "https://x.test/a.pdf", "sha256_actual": digest, "retrieved_at": "now", "is_pdf": True}]}))
    cand = tmp_path / "c.json"; cand.write_text(json.dumps({"page_candidates": []}))
    page = tmp_path / "p.json"; page.write_text(json.dumps({"documents": [{"doc_id": "d", "page_count": 1}]}))
    fact = {"fact_id": "f", "document_id": "d", "source_hash": digest, "page": 1, "evidence_span": "배추 1톤", "crop": "cabbage",
        "source_org": "MAFRA", "origin_review_status": "reviewed", "evidence_status": "agent_visual_verified",
        "release_at": "2024-01-01", "availability_evidence": []}
    review = tmp_path / "r.json"; review.write_text(json.dumps({"cohort_status": "complete_selected_cohort", "facts": [fact]}))
    with pytest.raises(ValueError, match="timestamp_missing_raw_quote"):
        a.normalize_manual_facts(review, integ, cand, page)


def test_manual_review_requires_nonempty_stable_fact_id(tmp_path):
    body = b"%PDF-fixture"; digest = hashlib.sha256(body).hexdigest()
    integ = tmp_path / "i.json"; integ.write_text(json.dumps({"observations": [{"doc_id": "d", "url": "https://x.test/a.pdf", "sha256_actual": digest, "retrieved_at": "now", "is_pdf": True}]}))
    cand = tmp_path / "c.json"; cand.write_text(json.dumps({"page_candidates": []}))
    page = tmp_path / "p.json"; page.write_text(json.dumps({"documents": [{"doc_id": "d", "page_count": 1}]}))
    fact = {"document_id": "d", "source_hash": digest, "page": 1, "evidence_span": "배추 1톤", "crop": "cabbage",
        "source_org": "MAFRA", "origin_review_status": "reviewed", "evidence_status": "agent_visual_verified"}
    review = tmp_path / "r.json"; review.write_text(json.dumps({"cohort_status": "complete_selected_cohort", "facts": [fact]}))
    with pytest.raises(ValueError, match="manual_review_fact_id_missing"):
        a.normalize_manual_facts(review, integ, cand, page)


def test_raw_strict_eligibility_requires_verified_version_branch(tmp_path):
    digest = hashlib.sha256(b"%PDF-fixture").hexdigest()
    integ = tmp_path / "i.json"; integ.write_text(json.dumps({"observations": [{"doc_id": "d", "url": "https://x.test/a.pdf", "sha256_actual": digest, "retrieved_at": "now", "is_pdf": True}]}))
    cand = tmp_path / "c.json"; cand.write_text(json.dumps({"page_candidates": []}))
    page = tmp_path / "p.json"; page.write_text(json.dumps({"documents": [{"doc_id": "d", "page_count": 1}]}))
    fact = {"fact_id": "f", "document_id": "d", "source_hash": digest, "page": 1, "evidence_span": "배추 1톤", "crop": "cabbage",
        "source_org": "MAFRA", "origin_review_status": "reviewed", "evidence_status": "agent_visual_verified",
        "asof_eligibility": "strict_historical_asof", "version_status": "current_capture_historical_bytes_unproven"}
    review = tmp_path / "r.json"; review.write_text(json.dumps({"cohort_status": "complete_selected_cohort", "facts": [fact]}))
    with pytest.raises(ValueError, match="strict_asof_without_verified_version_status"):
        a.normalize_manual_facts(review, integ, cand, page)


def test_manual_draft_adapter_binds_pdf_and_parent_time_quotes(tmp_path):
    root = tmp_path / "collection" / "runs" / a.RUN_ID
    detail = _obs(root, "https://www.mafra.go.kr/bbs/home/792/579316/artclView.do", "detail",
                  "<dd class='date'>2025. 4. 17. 14:00:00</dd>".encode(), {"source": "mafra-title"})
    pdf = _obs(root, "https://www.mafra.go.kr/bbs/home/792/600799/download.do", "document", b"%PDF-fixture",
               {"source": "mafra-title", "kind": "pdf", "content_type": "application/pdf", "detail_sha256": detail["sha256"]})
    detail2 = _obs(root, "https://www.mafra.go.kr/bbs/home/792/598513/artclView.do", "detail",
                   "<dd class='date'>2026. 7. 2. 17:00:00</dd>".encode(), {"source": "mafra-title"})
    pdf2 = _obs(root, "https://www.mafra.go.kr/bbs/home/792/600802/download.do", "document", b"%PDF-fixture-2",
               {"source": "mafra-title", "kind": "pdf", "content_type": "application/pdf", "detail_sha256": detail2["sha256"]})
    _cp(root, [detail, pdf, detail2, pdf2])
    integrity = a.audit_integrity(root, expected_observations=None)
    ip = tmp_path / "integrity.json"; ip.write_text(json.dumps(integrity))
    draft = {"reviewer_kind": "agent_visual", "scope": "selected", "documents": [{
        "date": "2025-04-17", "url": pdf["url"], "release_at": "2025-04-17T16:00:00+09:00",
        "release_quote": "2025. 4. 17.(목) 16:00 보도시점", "distribution_at": "2025-04-17T14:00:00+09:00",
        "distribution_quote": "2025. 4. 17.(목) 14:00", "facts": [["fact-1", 1, "cabbage", 90, "tonnes", "actual_release",
        "정부 비축물량 90톤 방출", "retrospective_publication_scenario", "government action", "MAFRA"]]}, {
        "date": "2026-07-02", "url": pdf2["url"], "release_at": "2026-07-02T17:00:00+09:00",
        "release_quote": "2026. 7. 2.(목) 17:00", "distribution_at": "2025-07-02T09:00:00+09:00",
        "distribution_quote": "2025. 7. 2.(목) 09:00", "facts": [["fact-2", 1, "cabbage", 3000, "tonnes", "planned_release_subset",
        "필요 시기(7~9월)에 정부 비축분 중 일부(3천톤 수준)를 공급", "retrospective_publication_scenario", "planned period", "MAFRA"]]}]}
    dp = tmp_path / "draft.json"; dp.write_text(json.dumps(draft, ensure_ascii=False))
    out = tmp_path / "review.json"
    result = a.adapt_manual_draft(dp, out, integrity_path=ip, run_root=root)
    adapted = json.loads(out.read_text())
    fact = adapted["facts"][0]
    assert result["adapted_fact_count"] == 2 and not adapted["whole_corpus_review_complete"]
    assert fact["posted_at"] == "2025-04-17T14:00:00+09:00"
    assert fact["release_at"] == "2025-04-17T16:00:00+09:00"
    assert fact["distribution_at"] == "2025-04-17T14:00:00+09:00"
    assert any(e["field"] == "posted_at" and e["source_document_id"] == "mafra-eval-" + a.sha256(detail["url"].encode())[:16]
        for e in fact["availability_evidence"])
    assert any(e["field"] == "release_condition" and "16:00" in e["raw_quote"] for e in fact["availability_evidence"])
    assert any(e["field"] == "release_at" and e["timestamp_text_candidate"].endswith("16:00") for e in fact["availability_evidence"])
    fact2 = adapted["facts"][1]
    assert fact2["posted_at"] == "2026-07-02T17:00:00+09:00"
    assert fact2["release_at"] == "2026-07-02T17:00:00+09:00"
    assert fact2["distribution_at"] == "2025-07-02T09:00:00+09:00"
    assert fact2["effective_start"] is None and fact2["effective_end"] is None
    candidates = tmp_path / "candidates.json"; candidates.write_text(json.dumps({"page_candidates": []}))
    pages = tmp_path / "pages.json"; pages.write_text(json.dumps({"documents": [
        {"doc_id": fact["document_id"], "page_count": 1}, {"doc_id": fact2["document_id"], "page_count": 1}]}))
    normalized = a.normalize_manual_facts(out, ip, candidates, pages)["reviewed_facts"]
    assert normalized[0]["posted_at"] == "2025-04-17T14:00:00+09:00"
    assert normalized[0]["publish_eligible_at_scenario"] == "2025-04-17T16:00:00+09:00"
    assert normalized[1]["posted_at"] == "2026-07-02T17:00:00+09:00"
    assert normalized[1]["distribution_at"] == "2025-07-02T09:00:00+09:00"
