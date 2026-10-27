from agri_data.period_supply_document_v8 import (
    _available_at_origin, _candidate_numeric, _candidate_spans,
    _normalize_review_fact, _origin_timestamp, _parse_dt,
)


def test_candidate_spans_capture_jeju_winter_radish_harvest_and_area():
    text = """❍ 무가격동향) 전주 가격 상승
재배동향) 수확, 출하 시작(7% 내외 수확, 390ha), 지상부 노출 언피해 및 열근 발생 우려"""
    rows = _candidate_spans(text, {}, "jeju-agri-extension", 1)
    rows = [r for r in rows if "390ha" in r["line_text"]]
    assert [r["numeric_token"] for r in rows] == ["7%", "390ha"]
    row = rows[0]
    assert row["crop_candidate"] == "radish"
    assert "harvest_progress" in row["quantity_kind_candidates"]


def test_candidate_spans_keep_range_and_reject_unrelated_crop_prices_and_phone():
    text = """m 월동무: 수확 및 출하작업 진행중(8~10%수확), 월동 후 수확예정 포장은 습해관리 철저
❍ 양배추가격동향) 제주시공판장 평균가격 3,600원/8kg
※감귤품평회 개최 알림 문의처:762-3091"""
    rows = _candidate_spans(text, {}, "jeju-agri-extension", 4)
    assert len(rows) == 1
    assert rows[0]["numeric_token"] == "8~10%"
    assert rows[0]["crop_candidate"] == "radish"


def test_candidate_spans_separate_government_cabbage_radish_and_mixed_total():
    text = """수급불안 대비 배추·무·마늘·양파 3만 4천톤 비축 추진
▲ (배추) 여름철 공급안정을 위해 봄배추 6천톤 비축 및 가격상승 시 시장 상장
▲ (무) 여름철 수급불안 대비 봄무 2천톤 비축 및 가격상승 시 시장 상장
▲ (마늘) 대비하기 위해 6천톤 비축을 추진"""
    rows = _candidate_spans(text, {}, "mafra", 1)
    by_line = {r["line_text"]: r for r in rows}
    mixed = next(r for r in rows if "3만 4천톤" in r["line_text"])
    cabbage = next(r for r in rows if "봄배추" in r["line_text"])
    radish = next(r for r in rows if "봄무" in r["line_text"])
    assert mixed["crop_candidate"] == "mixed_crop_unallocated"
    assert mixed["numeric_token"] == "3만 4천톤"
    assert cabbage["crop_candidate"] == "cabbage"
    assert cabbage["numeric_token"] == "6천톤"
    assert radish["crop_candidate"] == "radish"
    assert radish["numeric_token"] == "2천톤"
    assert not any("6천톤" == r["numeric_token"] for r in rows if "마늘" in r["line_text"])


def test_candidates_are_not_elevated_and_preserve_krei_and_release_flags():
    text = "월동무 수확 진행률 8~10%입니다. 농업관측정보 인용. 보도 시점 2022.05.29 11:00"
    rows = _candidate_spans(text, {}, "mafra", 1)
    assert rows[0]["review_status"] == "unreviewed_candidate_not_fact"
    assert "2022.05.29" in rows[0]["date_tokens_in_span"]
    assert "krei_or_agricultural_outlook_mention_review_required" in rows[0]["origin_flags"]
    assert "embargo_or_release_time_text_review_required" in rows[0]["origin_flags"]


def test_candidate_spans_do_not_merge_yangbaechu_with_target_baechu():
    text = "❍ 양배추 가격동향 재배면적 1,200ha"
    rows = _candidate_spans(text, {}, "jeju-agri-extension", 1)
    assert rows == []


def test_review_normalizer_keeps_dates_distinct_and_requires_historical_artifact_for_strict():
    raw = {"fact_id": "f1", "update_id": "u1", "document_id": "doc", "page": 1,
        "evidence_span": "span", "crop": "radish", "quantity_kind": "harvest_progress",
        "posted_at": "2023-12-22T17:00:00+09:00", "document_date": "2023-12-22",
        "release_at": None, "first_seen_at": "incorrect-local-value",
        "version_status": "strict_historical_asof_verified", "evidence_status": "agent_visual_verified"}
    row = _normalize_review_fact(raw, "agent_visual", {"sha256": "a" * 64,
        "url": "https://example.invalid/doc.pdf", "retrieved_at": "2026-10-15T23:00:00Z"})
    assert row["update_group_id"] == "u1"
    assert row["source_hash"] == "a" * 64
    assert row["first_seen_at"] == "2026-10-15T23:00:00Z"
    assert row["posted_at"] == "2023-12-22T17:00:00+09:00"
    assert row["document_date"] == "2023-12-22"
    assert row["asof_eligibility"] == "retrospective_publication_scenario"


def test_date_only_publication_and_issue_origin_use_separate_conservative_boundaries():
    publication = _parse_dt("2022-05-27")
    origin = _origin_timestamp("2022-05-27")
    assert publication.isoformat() == "2022-05-28T00:00:00+09:00"
    assert origin.isoformat() == "2022-05-27T10:00:00+09:00"


def test_hold_series_override_known_publication_date_without_discarding_fact():
    row = _normalize_review_fact({
        "fact_id": "held", "update_id": "same-issue", "document_id": "doc", "page": 1,
        "crop": "radish", "quantity_kind": "harvest_progress", "value": 25,
        "unit": "percent", "posted_at": "2023-12-22T17:00:00+09:00",
        "series_admissibility": "hold_population_or_basis_inconsistency",
    }, "agent_visual", {"sha256": "a" * 64, "retrieved_at": "2026-10-15T23:00:00Z"})
    assert row["value"] == 25
    assert row["update_group_id"] == "same-issue"
    assert row["asof_eligibility"] == "ineligible"
    assert row["publish_eligible_at_scenario"] == "2023-12-22T17:00:00+09:00"
    assert row["exclusion_reason"] == "series_admissibility_hold"


def test_publication_boundary_uses_late_release_even_if_scenario_boundary_is_earlier():
    row = _normalize_review_fact({
        "fact_id": "late-release", "update_id": "u1", "document_id": "doc", "page": 1,
        "crop": "radish", "quantity_kind": "planned_release", "value": 100,
        "unit": "tonnes", "publication_scenario_not_before": "2023-12-22T10:00:00+09:00",
        "posted_at": "2023-12-22T11:00:00+09:00", "release_at": "2023-12-22T17:00:00+09:00",
    }, "agent_visual", {"sha256": "a" * 64, "retrieved_at": "2026-10-15T23:00:00Z"})
    assert row["publish_eligible_at_scenario"] == "2023-12-22T17:00:00+09:00"
    assert not _available_at_origin(row, "2023-12-22")
    assert _available_at_origin(row, "2023-12-23")


def test_strict_historical_fact_still_requires_publication_boundary_for_origin_support(tmp_path):
    evidence = tmp_path / "historical.pdf"
    evidence.write_bytes(b"archived version")
    import hashlib
    row = _normalize_review_fact({
        "fact_id": "strict", "update_id": "u1", "document_id": "doc", "page": 1,
        "crop": "radish", "quantity_kind": "planned_release", "value": 100,
        "unit": "tonnes", "version_status": "strict_historical_asof_verified",
        "historical_version_evidence": str(evidence),
        "historical_version_evidence_sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        "posted_at": "2023-12-22T11:00:00+09:00", "release_at": "2023-12-22T17:00:00+09:00",
    }, "agent_visual", {"sha256": "a" * 64, "retrieved_at": "2026-10-15T23:00:00Z"})
    assert row["asof_eligibility"] == "strict_historical_asof"
    assert row["publish_eligible_at_scenario"] == "2023-12-22T17:00:00+09:00"
    assert not _available_at_origin(row, "2023-12-22")
    assert _available_at_origin(row, "2023-12-23")


def test_candidate_numeric_distinguishes_seedlings_from_tonnes():
    values, unit = _candidate_numeric("5만주")
    assert values == {50000.0}
    assert unit == "seedlings"
    values, unit = _candidate_numeric("5천톤")
    assert values == {5000.0}
    assert unit == "tonnes"


def test_new_update_first_origin_is_global_across_evaluation_boundaries():
    from agri_data.period_supply_document_v8 import _mark_first_update_origins
    support = [
        {"fact_id": "f1", "update_group_id": "u1", "crop": "radish", "horizon": 2,
         "origin": "2023-12-20", "scenario_known_past_fact": True,
         "series_admissibility": None, "horizon_relevance": "known_supply_update"},
        {"fact_id": "f1", "update_group_id": "u1", "crop": "radish", "horizon": 2,
         "origin": "2024-01-01", "scenario_known_past_fact": True,
         "series_admissibility": None, "horizon_relevance": "known_supply_update"},
        {"fact_id": "f2", "update_group_id": "u1", "crop": "radish", "horizon": 2,
         "origin": "2023-12-20", "scenario_known_past_fact": True,
         "series_admissibility": None, "horizon_relevance": "known_supply_update"},
        {"fact_id": "f3", "update_group_id": "u2", "crop": "radish", "horizon": 2,
         "origin": "2023-12-20", "scenario_known_past_fact": True,
         "series_admissibility": None, "horizon_relevance": "historical_comparator_not_current_supply"},
        {"fact_id": "f4", "update_group_id": "u3", "crop": "radish", "horizon": 2,
         "origin": "2023-12-20", "scenario_known_past_fact": True,
         "series_admissibility": "hold_basis", "horizon_relevance": "known_supply_update"},
    ]
    _mark_first_update_origins(support)
    assert [r["is_new_update_at_scenario_origin"] for r in support] == [True, False, True, False, False]
