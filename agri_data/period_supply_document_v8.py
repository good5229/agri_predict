"""Offline Phase A evidence inventory for V8 supply-document PDFs.

All emitted numeric spans are review candidates, never verified facts. The
module intentionally has no network, forecasting, or model-fitting code.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import io
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
COLLECTION = ROOT / "outputs/collection/supply-documents-20261016"
DATA = ROOT / "data/raw/supply_documents_20261016"
PHASE_A = ROOT / "outputs/experiments/period-supply-document-signal-v8/phase-A"
DOC_CATALOG = COLLECTION / "document-catalog.json"
ATT_CATALOG = COLLECTION / "attachment-catalog.json"
DOC_CHECKPOINT = DATA / "documents/checkpoint.json"
ATT_CHECKPOINT = DATA / "attachments/checkpoint.json"
LIST_CHECKPOINT = DATA / "listings/checkpoint.json"
LIST_EXTRACTION = COLLECTION / "listings-extraction.json"
DOC_EXTRACTION = COLLECTION / "documents-extraction.json"
ATT_EXTRACTION = COLLECTION / "attachments-extraction.json"

_CROP_RE = re.compile(r"배추|월동무|봄무|가을무|겨울무|양배추|(?<![가-힣])무(?![가-힣])")
_VALUE_RE = re.compile(
    r"(?<![\d])(?:약\s*)?\d[\d,]*(?:\.\d+)?\s*(?:~|∼|～|에서|부터|[-–—])\s*\d[\d,]*(?:\.\d+)?\s*(?:천톤|만톤|톤|ha|㏊|%|퍼센트|kg|㎏|원|만주)?"
    r"|(?<![\d])\d+\s*만\s*\d+\s*천톤"
    r"|(?<![\d])(?:약\s*)?\d[\d,]*(?:\.\d+)?\s*(?:천톤|만톤|톤|ha|㏊|%|퍼센트|kg|㎏|원|만주)"
)
_TYPE_PATTERNS = {
    "planned_reserve": re.compile(r"비축\s*(?:계획|예정|추진)|비축계획"),
    "actual_reserve": re.compile(r"(?:비축|재고량|보유량)\s*(?:실적|현황|확보)"),
    "planned_release": re.compile(r"(?:방출|공급|출하)\s*(?:계획|예정)|시장공급"),
    "actual_release": re.compile(r"(?:방출|출고|출하)\s*(?:실적|완료|실시)"),
    "area": re.compile(r"재배면적|재배 면적|수확면적|면적"),
    "harvest_progress": re.compile(r"수확\s*(?:진행률|률|진도)|수확률|수확[^\n]{0,20}(?:%|내외)"),
    "crop_condition": re.compile(r"생육|작황|결구|비대|동해|습해"),
}
_DATE_RE = re.compile(r"(?<!\d)(?:20\d{2})[.년/-]\s*\d{1,2}[.월/-]\s*\d{1,2}(?:일)?")
_POSTED_RE = re.compile(r"(20\d{2})[-.]\s*(\d{1,2})[-.]\s*(\d{1,2})\s+(\d{1,2}):(\d{2})")
_ALL_CROP_RE = re.compile(r"월동무|봄무|가을무|겨울무|양배추|배추|무|브로콜리|콜라비|비트|당근|양파|마늘|감자")
_TARGET_CROP_NAMES = {"월동무": "radish", "봄무": "radish", "가을무": "radish", "겨울무": "radish",
                      "배추": "cabbage", "무": "radish"}
_TARGET_CABBAGE_RE = re.compile(r"봄배추|가을배추|월동배추|(?<![가-힣])배추")
_SUPPLY_UNIT_RE = re.compile(r"(?:%|퍼센트|천톤|만톤|톤|ha|㏊|만주|주)(?:$|\s|[),.])", re.I)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_all_captures(pdf_checks: list[dict]) -> dict:
    stages = [
        ("listings", LIST_CHECKPOINT, LIST_EXTRACTION, DATA / "listings", None),
        ("documents", DOC_CHECKPOINT, DOC_EXTRACTION, DATA / "documents", DOC_CATALOG),
        ("attachments", ATT_CHECKPOINT, ATT_EXTRACTION, DATA / "attachments", ATT_CATALOG),
    ]
    results = []
    for stage, cp_path, extraction_path, raw_root, catalog_path in stages:
        cp, extraction = _load_json(cp_path), _load_json(extraction_path)
        extraction_by_id = {r["id"]: r for r in extraction.get("outcomes", [])}
        catalog_by_id = ({r["request_id"]: r for r in _load_json(catalog_path).get("entries", [])}
                         if catalog_path else {})
        for request_id, record in sorted(cp.get("requests", {}).items()):
            raw_path = raw_root / record.get("raw_file", "")
            if record.get("status") != "captured" or not raw_path.is_file():
                raise ValueError(f"capture_missing:{stage}:{request_id}")
            raw = raw_path.read_bytes()
            raw_hash = sha256(raw)
            if raw_hash != record.get("sha256") or len(raw) != record.get("byte_length"):
                raise ValueError(f"capture_hash_or_length_mismatch:{stage}:{request_id}")
            outcome = extraction_by_id.get(request_id)
            if not outcome or outcome.get("sourceSha256") != raw_hash:
                raise ValueError(f"extraction_source_association_mismatch:{stage}:{request_id}")
            text_path = raw_root / outcome["textFile"] if outcome.get("textFile") else None
            if not text_path or not text_path.is_file() or sha256(text_path.read_bytes()) != outcome.get("textSha256"):
                raise ValueError(f"extraction_text_hash_mismatch:{stage}:{request_id}")
            catalog = catalog_by_id.get(request_id)
            if catalog_path and (not catalog or catalog.get("raw_sha256") != raw_hash):
                raise ValueError(f"catalog_raw_association_mismatch:{stage}:{request_id}")
            results.append({"capture_stage": stage, "request_id": request_id,
                "source": record.get("request", {}).get("source"), "request_status": record.get("status"),
                "http_status": record.get("http_status"), "raw_file": str(raw_path.relative_to(ROOT)),
                "raw_sha256": raw_hash, "raw_byte_length": len(raw),
                "extraction_status": outcome.get("status"), "text_file": str(text_path.relative_to(ROOT)),
                "text_sha256": outcome.get("textSha256"),
                "catalog_status": "hash_associated" if catalog_path else "listing_extraction_associated",
                "catalog_source_sha256": catalog.get("raw_sha256") if catalog else None})
    if len(results) != 150:
        raise ValueError(f"unexpected_capture_count:{len(results)}")
    duplicates = defaultdict(list)
    for row in results:
        duplicates[row["raw_sha256"]].append(row["request_id"])
    duplicate_groups = [ids for ids in duplicates.values() if len(ids) > 1]
    return {"scope_status": "verified_existing_captures_only", "capture_count": len(results),
            "raw_hashes_verified": len(results), "text_extractions_hash_associated": len(results),
            "document_catalog_hash_associations": sum(r["catalog_status"] == "hash_associated" for r in results),
            "listing_extraction_associations": sum(r["catalog_status"] == "listing_extraction_associated" for r in results),
            "duplicate_raw_hash_groups": duplicate_groups,
            "pdf_page_checks": pdf_checks, "captures": results}


_FACT_FIELDS = ["fact_id", "update_group_id", "document_id", "source_url", "source_hash", "page",
    "evidence_span", "source_org", "provenance_origin", "crop", "variety_or_crop_type", "region",
    "quantity_kind", "value", "value_lower", "value_upper", "unit", "denominator_definition",
    "survey_at", "document_date", "posted_at", "release_at", "distribution_at", "first_seen_at",
    "revision_at", "version_status", "effective_start", "effective_end", "horizon_relevance",
    "evidence_status", "reviewer_kind", "asof_eligibility", "publish_eligible_at_scenario",
    "historical_version_evidence", "exclusion_reason", "series_admissibility", "approximate",
    "origin_review_status", "origin_source_excerpt", "duplicate_group_id", "duplicate_status",
    "visual_evidence", "numeric_token", "posted_at_evidence", "release_date", "release_condition",
    "distribution_date", "publication_scenario_not_before", "publication_scenario_basis",
    "effective_period_text", "evidence_note", "document_date_note", "posted_at_source_url",
    "posted_at_source_hash", "historical_version_evidence_sha256", "candidate_link_status"]


def _parse_dt(value: object) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y.%m.%d %H:%M", "%Y.%m.%d"):
        try:
            dt = datetime.strptime(text, fmt)
            if dt.tzinfo is None:
                from datetime import timezone, timedelta
                dt = dt.replace(tzinfo=timezone(timedelta(hours=9)))
            if fmt in {"%Y-%m-%d", "%Y.%m.%d"}:
                from datetime import timedelta
                # Date-only sources are conservatively usable only from the
                # following KST midnight (publication may be late that day).
                dt += timedelta(days=1)
            return dt
        except ValueError:
            continue
    m = re.search(r"(20\d{2})[.년/-]\s*(\d{1,2})[.월/-]\s*(\d{1,2})(?:일)?\s*(\d{1,2}):(\d{2})", text)
    if m:
        from datetime import timezone, timedelta
        return datetime(*(int(m.group(i)) for i in range(1, 4)), int(m.group(4)), int(m.group(5)),
                        tzinfo=timezone(timedelta(hours=9)))
    return None


def _origin_timestamp(value: object) -> datetime | None:
    """Frozen V5 issue date interpreted at 10:00 KST per existing contract."""
    if not value:
        return None
    text = str(value)
    try:
        from datetime import timezone, timedelta
        return datetime.strptime(text[:10], "%Y-%m-%d").replace(
            hour=10, minute=0, second=0, tzinfo=timezone(timedelta(hours=9)))
    except ValueError:
        return _parse_dt(value)


def _publication_boundary(raw: dict) -> datetime | None:
    """Latest available publication/release/distribution boundary, in KST."""
    candidates = [_parse_dt(raw.get(k)) for k in (
        "publication_scenario_not_before", "posted_at", "release_at", "distribution_at")]
    return max((value for value in candidates if value is not None), default=None)


def _available_at_origin(fact: dict, origin: object) -> bool:
    if fact.get("asof_eligibility") not in {"strict_historical_asof", "retrospective_publication_scenario"}:
        return False
    boundary = _parse_dt(fact.get("publish_eligible_at_scenario"))
    at = _origin_timestamp(origin)
    return boundary is not None and at is not None and boundary <= at


def _normalize_review_fact(raw: dict, reviewer_kind: str, source_record: dict) -> dict:
    row = {name: None for name in _FACT_FIELDS}
    row.update(raw)
    row["update_group_id"] = raw.get("update_group_id") or raw.get("update_id")
    row["document_id"] = raw["document_id"]
    row["source_url"] = source_record.get("final_url") or source_record.get("url")
    row["source_hash"] = source_record.get("sha256")
    row["first_seen_at"] = source_record.get("retrieved_at")
    row["reviewer_kind"] = raw.get("reviewer_kind") or reviewer_kind
    row["historical_version_evidence"] = raw.get("historical_version_evidence")
    row["origin_review_status"] = raw.get("origin_review_status") or raw.get("origin_review")
    # A citation to a release/publication time is not proof that the captured
    # bytes are the historical version. Strict eligibility needs a separate,
    # verifiable archival artifact and is therefore false by default.
    strict_proof = False
    evidence_path = raw.get("historical_version_evidence")
    evidence_sha = raw.get("historical_version_evidence_sha256")
    if raw.get("version_status") == "strict_historical_asof_verified" and evidence_path and evidence_sha:
        path = Path(evidence_path)
        if not path.is_absolute(): path = ROOT / path
        strict_proof = path.is_file() and sha256(path.read_bytes()) == evidence_sha
    publication_boundary = _publication_boundary(raw)
    if publication_boundary is not None:
        row["publish_eligible_at_scenario"] = publication_boundary.isoformat()
    held_series = str(raw.get("series_admissibility") or "").startswith("hold_")
    if held_series:
        # A source-authored hold (for example, unresolved population/method
        # changes) is a hard eligibility boundary, even when a publication
        # date is known. Preserve the record but never count it as support.
        row["asof_eligibility"] = "ineligible"
        row["exclusion_reason"] = row.get("exclusion_reason") or "series_admissibility_hold"
    elif strict_proof:
        row["asof_eligibility"] = "strict_historical_asof"
    else:
        reviewed_eligibility = raw.get("asof_eligibility")
        if reviewed_eligibility in {"ineligible", "pending_source_review"}:
            row["asof_eligibility"] = reviewed_eligibility
        elif publication_boundary is not None:
            row["asof_eligibility"] = "retrospective_publication_scenario"
        else:
            row["asof_eligibility"] = reviewed_eligibility or "prospective_only"
    if row.get("effective_start") and row.get("effective_end"):
        if not row.get("horizon_relevance"):
            row["horizon_relevance"] = "dated_effective_interval_needs_target_intersection"
    else:
        row["horizon_relevance"] = row.get("horizon_relevance") or "unknown_no_documented_effective_interval"
    for field in _FACT_FIELDS:
        if field not in row:
            row[field] = None
    return {field: row.get(field) for field in _FACT_FIELDS}


def _target_rows() -> list[dict]:
    import pandas as pd
    files = [ROOT / "outputs/experiments/period-pooled-direct-lgbm-v5-20261016/development-forecasts.parquet",
             ROOT / "outputs/experiments/period-pooled-direct-lgbm-v5-20261016/forecasts.parquet"]
    rows = []
    for file in files:
        frame = pd.read_parquet(file, columns=["crop", "horizon", "targetStart", "targetEnd", "origin", "inputCutoff"])
        for row in frame.to_dict("records"):
            row["evaluation_period"] = "development_2022_2023" if "development" in file.name else "evaluation_2024_2026-09"
            rows.append(row)
    # V5 has one row for every crop/horizon/target and some origin duplication
    # is possible; retain unique frozen keys only.
    seen, unique = set(), []
    for row in rows:
        key = tuple(row[k] for k in ("crop", "horizon", "targetStart", "targetEnd", "origin", "inputCutoff"))
        if key not in seen:
            seen.add(key); unique.append(row)
    return unique


def _mark_first_update_origins(support: list[dict]) -> None:
    """Mark each eligible update once at its first origin across all periods."""
    first_update_origin = {}
    for row in support:
        if (row.get("scenario_known_past_fact") and row.get("update_group_id")
                and not str(row.get("series_admissibility") or "").startswith("hold_")
                and row.get("horizon_relevance") != "historical_comparator_not_current_supply"):
            key = (row.get("crop"), row.get("horizon"), row["update_group_id"])
            origin_dt = _origin_timestamp(row.get("origin"))
            if origin_dt is not None and (key not in first_update_origin or origin_dt < first_update_origin[key][0]):
                first_update_origin[key] = (origin_dt, row["origin"])
    for row in support:
        key = (row.get("crop"), row.get("horizon"), row.get("update_group_id"))
        row["is_new_update_at_scenario_origin"] = bool(
            row.get("scenario_known_past_fact") and key in first_update_origin
            and row.get("origin") == first_update_origin[key][1]
            and not str(row.get("series_admissibility") or "").startswith("hold_")
            and row.get("horizon_relevance") != "historical_comparator_not_current_supply")


def _phase_a_summary(reviewed: list[dict], support: list[dict], release_support: list[dict],
                     source_checks: dict) -> dict:
    by_crop = defaultdict(list)
    for row in reviewed:
        by_crop[row.get("crop") or "unknown"].append(row)
    crop_summaries = []
    for crop, rows in sorted(by_crop.items()):
        known = [r for r in rows if r.get("effective_start") and r.get("effective_end")]
        unknown = [r for r in rows if not (r.get("effective_start") and r.get("effective_end"))]
        krei_excluded = [r for r in rows if r["asof_eligibility"] == "ineligible" and re.search(
            r"KREI|한국농촌경제연구원|농촌경제연구원", " ".join(str(r.get(k) or "") for k in (
                "source_org", "provenance_origin", "exclusion_reason", "evidence_note")), re.I)]
        interval_map = defaultdict(list)
        for row in known:
            interval_map[(row["effective_start"], row["effective_end"])].append(row["fact_id"])
        crop_summaries.append({
            "crop": crop, "reviewed_fact_count": len(rows),
            "unique_update_count_including_ineligible": len({r["update_group_id"] for r in rows if r.get("update_group_id")}),
            "scenario_eligible_fact_count": sum(r["asof_eligibility"] == "retrospective_publication_scenario" for r in rows),
            "scenario_eligible_unique_update_count": len({r["update_group_id"] for r in rows
                if r.get("update_group_id") and r["asof_eligibility"] == "retrospective_publication_scenario"}),
            "strict_historical_fact_count": sum(r["asof_eligibility"] == "strict_historical_asof" for r in rows),
            "ineligible_fact_count": sum(r["asof_eligibility"] == "ineligible" for r in rows),
            "pending_source_review_fact_count": sum(r["asof_eligibility"] == "pending_source_review" for r in rows),
            "explicit_krei_excluded_count": len(krei_excluded),
            "effective_interval_known_fact_count": len(known),
            "effective_interval_unknown_fact_count": len(unknown),
            "effective_interval_partial_bound_fact_count": sum(bool(r.get("effective_start")) != bool(r.get("effective_end")) for r in rows),
            "known_exact_effective_intervals": [{"effective_start": start, "effective_end": end,
                "fact_count": len(ids), "fact_ids": sorted(ids)} for (start, end), ids in sorted(interval_map.items())],
            "unknown_effective_interval_fact_ids": sorted(r["fact_id"] for r in unknown),
        })
    g3 = []
    for row in release_support:
        if row["evaluation_period"] != "development_2022_2023":
            continue
        g3.append({"target_year": row["target_year"], "crop": row["crop"], "horizon": row["horizon"],
            "first_new_update_count": row["first_new_update_count"],
            "first_new_update_origin_count": row["first_new_update_origin_count"],
            "first_new_update_month_count": row["first_new_update_month_count"],
            "minimum_events_pass": row["first_new_update_count"] >= 12,
            "minimum_distinct_origins_pass": row["first_new_update_origin_count"] >= 12,
            "minimum_months_pass": row["first_new_update_month_count"] >= 6,
            "g3_support_pass": row["first_new_update_count"] >= 12 and row["first_new_update_origin_count"] >= 12
                and row["first_new_update_month_count"] >= 6})
    g4 = []
    for crop in sorted({r["crop"] for r in support}):
        for horizon in (2, 3):
            rows = [r for r in support if r["evaluation_period"] == "evaluation_2024_2026-09"
                    and r["crop"] == crop and r["horizon"] == horizon]
            events = [r for r in rows if r["is_new_update_at_scenario_origin"]]
            groups = {r["update_group_id"] for r in events if r["update_group_id"]}
            origins = {r["origin"] for r in events}
            years = {r["target_year"] for r in events}
            targets = {(r["target_start"], r["target_end"]) for r in rows}
            g4.append({"crop": crop, "horizon": horizon, "evaluation_first_new_update_group_count": len(groups),
                "evaluation_first_new_update_origin_count": len(origins),
                "evaluation_first_new_update_year_count": len(years), "evaluation_target_count": len(targets),
                "minimum_events_pass": len(origins) >= 24, "minimum_years_pass": len(years) >= 2,
                "minimum_targets_pass": len(targets) >= 36,
                "g4_support_pass": len(origins) >= 24 and len(years) >= 2 and len(targets) >= 36})
    return {
        "scope_status": "offline_phase_a_support_audit_no_training_no_price_scoring_no_new_network_requests",
        "staleness_and_missingness": "Scenario-known means the document publication boundary precedes the forecast origin; it does not establish current target-period applicability. No fact is forward-filled. Missing effective intervals remain unknown and are not treated as future supply.",
        "source_funnel": {"captured_response_count": source_checks.get("capture_count"),
            "raw_hashes_verified": source_checks.get("raw_hashes_verified"),
            "text_extractions_hash_associated": source_checks.get("text_extractions_hash_associated"),
            "document_catalog_hash_associations": source_checks.get("document_catalog_hash_associations"),
            "listing_extraction_associations": source_checks.get("listing_extraction_associations")},
        "crop_effective_bounds_and_eligibility": crop_summaries,
        "reviewed_cohort_counts": {"reviewed_fact_count": len(reviewed),
            "unique_fact_count": len({r["fact_id"] for r in reviewed}),
            "unique_update_count_including_ineligible": len({r["update_group_id"] for r in reviewed if r.get("update_group_id")}),
            "scenario_eligible_fact_count": sum(r["asof_eligibility"] == "retrospective_publication_scenario" for r in reviewed),
            "scenario_eligible_unique_update_count": len({r["update_group_id"] for r in reviewed
                if r.get("update_group_id") and r["asof_eligibility"] == "retrospective_publication_scenario"}),
            "strict_historical_fact_count": sum(r["asof_eligibility"] == "strict_historical_asof" for r in reviewed),
            "ineligible_fact_count": sum(r["asof_eligibility"] == "ineligible" for r in reviewed),
            "pending_source_review_fact_count": sum(r["asof_eligibility"] == "pending_source_review" for r in reviewed)},
        "development_annual_support_and_g3": g3,
        "evaluation_g4_support": g4,
        "annual_origin_support": [{k: row[k] for k in ("evaluation_period", "target_year", "crop", "horizon",
            "planned_origin_count", "scenario_known_fact_origin_count", "scenario_known_unique_update_count",
            "first_new_update_count", "first_new_update_origin_count", "first_new_update_month_count",
            "first_new_update_months", "first_new_update_origins", "first_new_update_publication_boundaries",
            "first_new_update_document_ids", "no_observed_new_update_origin_count",
            "strict_asof_fact_origin_count", "actual_target_overlap_fact_origin_count",
            "overlap_unknown_fact_origin_count", "documented_effective_interval_fact_origin_count",
            "known_effective_overlap_non_event_origin_count")} for row in release_support],
        "interpretation": "Scenario-known counts include old known records repeated at later planned origins. New-update metrics count each update group only at its first available frozen origin. no_observed_new_update_origin_count is the full planned-origin calendar minus origins with a newly available reviewed update; it is not proof that no supply event occurred.",
    }


def _csv_bytes(rows: list[dict], fieldnames: list[str] | None = None) -> bytes:
    buffer = io.StringIO(newline="")
    columns = fieldnames or (list(rows[0]) if rows else [])
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def _candidate_numeric(token: str) -> tuple[set[float], str | None]:
    unit = "percent" if "%" in token or "퍼센트" in token else (
        "seedlings" if "주" in token else (
        "area_ha" if re.search(r"ha|㏊", token, re.I) else (
        "tonnes" if "톤" in token else None)))
    nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", token)]
    if "만" in token and "천" in token and len(nums) >= 2:
        nums = [nums[0] * 10000 + nums[1] * 1000]
    elif "만" in token and len(nums) == 1:
        nums = [nums[0] * 10000]
    elif "천" in token and len(nums) == 1:
        nums = [nums[0] * 1000]
    return set(nums), unit


def _fact_numeric_values(fact: dict) -> tuple[set[float], str | None]:
    values = {float(fact[k]) for k in ("value", "value_lower", "value_upper") if fact.get(k) is not None}
    unit = str(fact.get("unit") or "").lower()
    if "percent" in unit or "%" in unit: unit = "percent"
    elif "ha" in unit: unit = "area_ha"
    elif "주" in unit: unit = "seedlings"
    elif "ton" in unit: unit = "tonnes"
    else: unit = None
    return values, unit


def build_reviewed_phase_a(jeju_facts_path: Path, mafra_facts_path: Path,
                           output_dir: Path = PHASE_A) -> dict:
    """Normalize visual-review annotations and report conservative support.

    The candidate queue is not auto-promoted; only manually annotated records
    in the two supplied review files become fact rows.
    """
    import pandas as pd
    docs = _load_json(DOC_CHECKPOINT)["requests"]
    atts = _load_json(ATT_CHECKPOINT)["requests"]
    source_records = {**docs, **atts}
    reviewed = []
    for path, default_reviewer in ((jeju_facts_path, "agent_visual"), (mafra_facts_path, "agent_visual")):
        doc = _load_json(path)
        rows = doc if isinstance(doc, list) else doc.get("facts", [])
        default_reviewer = "agent_visual" if isinstance(doc, list) else doc.get("reviewer_kind", "agent_visual")
        for raw in rows:
            if raw.get("document_id") not in source_records:
                raise ValueError(f"review_fact_unknown_document:{raw.get('fact_id')}")
            record = source_records[raw["document_id"]]
            if raw.get("source_hash") and raw["source_hash"] != record.get("sha256"):
                raise ValueError(f"review_fact_source_hash_mismatch:{raw.get('fact_id')}")
            page = raw.get("page")
            if not isinstance(page, int) or page < 1:
                raise ValueError(f"review_fact_invalid_page:{raw.get('fact_id')}")
            reviewed.append(_normalize_review_fact(raw, default_reviewer, record))
    ids = [r["fact_id"] for r in reviewed]
    if None in ids or len(ids) != len(set(ids)):
        raise ValueError("review_fact_ids_missing_or_duplicated")

    # Mark possible duplicates, preserving every row. Hash-equal files are an
    # objective duplicate signal; text-equal fact spans across distinct files
    # remain a candidate relation for review.
    by_hash = defaultdict(list)
    for row in reviewed:
        by_hash[row["source_hash"]].append(row)
    duplicate_links = []
    for digest, rows in by_hash.items():
        documents = sorted({r["document_id"] for r in rows})
        if len(documents) > 1:
            group = "duplicate-raw-sha256-" + digest
            for row in rows:
                row["duplicate_group_id"] = group
                row["duplicate_status"] = "same_raw_bytes_preserved_as_separate_source_records"
            duplicate_links.append({"duplicate_group_id": group, "duplicate_method": "identical_raw_sha256",
                                    "document_ids": " | ".join(documents), "fact_ids": " | ".join(r["fact_id"] for r in rows)})
    for i, left in enumerate(reviewed):
        for right in reviewed[i + 1:]:
            if (left["document_id"] != right["document_id"] and left["crop"] == right["crop"]
                    and left["quantity_kind"] == right["quantity_kind"] and left["unit"] == right["unit"]
                    and left["value"] == right["value"] and left["value_lower"] == right["value_lower"]
                    and left["value_upper"] == right["value_upper"]
                    and re.sub(r"\s+", "", left["evidence_span"] or "") == re.sub(r"\s+", "", right["evidence_span"] or "")):
                group = "possible-content-repost-" + sha256((left["fact_id"] + "|" + right["fact_id"]).encode())[:16]
                duplicate_links.append({"duplicate_group_id": group, "duplicate_method": "same_crop_type_values_and_span_candidate",
                                        "document_ids": f"{left['document_id']} | {right['document_id']}",
                                        "fact_ids": f"{left['fact_id']} | {right['fact_id']}"})
                for row in (left, right):
                    if not row.get("duplicate_group_id"): row["duplicate_group_id"] = group
                    row["duplicate_status"] = "possible_repost_preserved_pending_review"

    # Resolve page candidate links to preserve one-to-many exact span relations.
    candidate_file = PHASE_A / "page-review-v4/page-candidates.json"
    candidates_by_doc_page = defaultdict(list)
    if candidate_file.is_file():
        for cand in _load_json(candidate_file).get("page_candidates", []):
            candidates_by_doc_page[(cand["document_id"], cand["page"])].append(cand)
    candidate_links = []
    for row in reviewed:
        fact_values, fact_unit = _fact_numeric_values(row)
        linked = []
        for cand in candidates_by_doc_page[(row["document_id"], row["page"])]:
            if cand.get("crop_candidate") not in {row["crop"], "mixed_crop_unallocated"}:
                continue
            cand_values, cand_unit = _candidate_numeric(cand.get("numeric_token", ""))
            same_unit = not fact_unit or not cand_unit or fact_unit == cand_unit
            if same_unit and fact_values & cand_values:
                linked.append(cand)
        row["candidate_ids"] = " | ".join(c["fact_candidate_id"] for c in linked)
        row["candidate_link_status"] = "candidate_numeric_crop_link" if linked else "manual_fact_not_found_by_candidate_scan"
        candidate_links.extend({"fact_id": row["fact_id"], "candidate_id": c["fact_candidate_id"],
            "match_basis": "same_document_page_crop_and_numeric_value_unit_class"} for c in linked)

    # Build fact x frozen-origin support; distinguish publication scenario,
    # actual target-period overlap, and unknown overlap.
    targets = _target_rows()
    support = []
    uses = Counter()
    for fact in reviewed:
        effective_start = _parse_dt(fact.get("effective_start"))
        effective_end = _parse_dt(fact.get("effective_end"))
        for target in targets:
            if target["crop"] != fact["crop"] or target["horizon"] not in {2, 3}:
                continue
            scenario_known = _available_at_origin(fact, target["origin"])
            if scenario_known:
                uses[fact["fact_id"]] += 1
            if effective_start and effective_end:
                t0, t1 = _parse_dt(target["targetStart"]), _parse_dt(target["targetEnd"])
                overlap = bool(t0 and t1 and effective_start <= t1 and effective_end >= t0)
                overlap_status = "supported" if overlap else "not_overlapping"
            else:
                overlap = None
                overlap_status = "overlap_unknown"
            support.append({"fact_id": fact["fact_id"], "update_group_id": fact["update_group_id"],
                "crop": fact["crop"], "horizon": target["horizon"], "target_start": target["targetStart"],
                "target_end": target["targetEnd"], "origin": target["origin"], "input_cutoff": target["inputCutoff"],
                "evaluation_period": target["evaluation_period"],
                "target_year": str(target["targetStart"])[:4],
                "publish_eligible_at_scenario": fact.get("publish_eligible_at_scenario"),
                "document_id": fact.get("document_id"),
                "strict_historical_asof": fact["asof_eligibility"] == "strict_historical_asof" and scenario_known,
                "scenario_known_past_fact": scenario_known,
                "actual_target_overlap_supported": overlap,
                "target_overlap_status": overlap_status,
                "series_admissibility": fact.get("series_admissibility"),
                "horizon_relevance": fact.get("horizon_relevance"),
                "is_new_update_at_scenario_origin": False})
    # An update is new only at its first scenario-available origin over the
    # complete frozen calendar. This is grouped by issue/update, not numeric
    # fact row, and does not reset at the development/evaluation boundary.
    _mark_first_update_origins(support)
    for row in reviewed:
        row["scenario_available_origin_count"] = uses[row["fact_id"]]
        row["repeat_reuse_count"] = max(0, uses[row["fact_id"]] - 1)

    # Coverage uses only visual-reviewed fact candidates, with fact/update
    # counts shown separately. Scenario facts are never strict-as-of support.
    crop_groups = defaultdict(list)
    for row in reviewed:
        year = (row.get("document_date") or row.get("posted_at") or "")[:4] or "unknown"
        crop_groups[(row["crop"] or "unknown", row["variety_or_crop_type"] or "unknown",
                     row["region"] or "unknown", year, row["quantity_kind"] or "unknown",
                     row["source_org"] or "unknown")].append(row)
    crop_coverage = []
    for key, rows in sorted(crop_groups.items()):
        crop_coverage.append({"crop": key[0], "variety_or_crop_type": key[1], "region": key[2],
            "document_year": key[3], "quantity_kind": key[4], "source_org": key[5],
            "visual_reviewed_fact_count": len(rows),
            "unique_update_count": len({r["update_group_id"] for r in rows if r["update_group_id"]}),
            "scenario_eligible_unique_update_count": len({r["update_group_id"] for r in rows
                if r["update_group_id"] and r["asof_eligibility"] == "retrospective_publication_scenario"}),
            "strict_historical_fact_count": sum(r["asof_eligibility"] == "strict_historical_asof" for r in rows),
            "publication_scenario_fact_count": sum(r["asof_eligibility"] == "retrospective_publication_scenario" for r in rows),
            "prospective_only_fact_count": sum(r["asof_eligibility"] == "prospective_only" for r in rows),
            "ineligible_fact_count": sum(r["asof_eligibility"] == "ineligible" for r in rows),
            "pending_source_count": sum(r["asof_eligibility"] == "pending_source_review" for r in rows),
            "explicit_krei_excluded_count": sum(r["asof_eligibility"] == "ineligible" and bool(re.search(
                r"KREI|한국농촌경제연구원|농촌경제연구원", " ".join(str(r.get(k) or "") for k in (
                    "source_org", "provenance_origin", "exclusion_reason", "evidence_note")), re.I)) for r in rows),
            "effective_interval_known_fact_count": sum(bool(r["effective_start"] and r["effective_end"]) for r in rows),
            "effective_interval_unknown_fact_count": sum(not (r["effective_start"] and r["effective_end"]) for r in rows),
            "effective_interval_partial_bound_fact_count": sum(bool(r["effective_start"]) != bool(r["effective_end"]) for r in rows),
            "exact_effective_bounds": " | ".join(sorted({f"{r['effective_start'] or '?'}..{r['effective_end'] or '?'}"
                for r in rows if r["effective_start"] or r["effective_end"]}))})
    vintage_groups = defaultdict(list)
    for row in reviewed:
        date_text = row.get("document_date") or row.get("posted_at") or ""
        source_year = date_text[:4] or "unknown"
        source_month = date_text[5:7] if len(date_text) >= 7 else "unknown"
        vintage_groups[(row["crop"] or "unknown", row["variety_or_crop_type"] or "unknown",
            source_year, source_month, row["asof_eligibility"], row["version_status"] or "unknown")].append(row)
    vintage_coverage = []
    for key, rows in sorted(vintage_groups.items()):
        vintage_coverage.append({"crop": key[0], "variety_or_crop_type": key[1], "source_year": key[2],
            "source_month": key[3], "asof_eligibility": key[4], "version_status": key[5],
            "fact_count": len(rows), "unique_fact_count": len({r["fact_id"] for r in rows}),
            "unique_update_count": len({r["update_group_id"] for r in rows if r["update_group_id"]}),
            "unique_documents": len({r["document_id"] for r in rows}),
            "strict_historical_fact_count": sum(r["asof_eligibility"] == "strict_historical_asof" for r in rows),
            "scenario_eligible_fact_count": sum(r["asof_eligibility"] == "retrospective_publication_scenario" for r in rows),
            "ineligible_fact_count": sum(r["asof_eligibility"] == "ineligible" for r in rows),
            "pending_fact_count": sum(r["asof_eligibility"] == "pending_source_review" for r in rows),
            "first_seen_dates": " | ".join(sorted({str(r["first_seen_at"])[:10] for r in rows if r.get("first_seen_at")})),
            "interpretation": "current 2026 capture; publication time does not prove historical content version"})
    support_groups = defaultdict(list)
    for r in support:
        support_groups[(r["evaluation_period"], r["target_year"], r["crop"], r["horizon"])].append(r)
    release_support = []
    for key, rows in sorted(support_groups.items()):
        known = [r for r in rows if r["scenario_known_past_fact"]]
        overlap = [r for r in known if r["actual_target_overlap_supported"] is True]
        unknown = [r for r in known if r["target_overlap_status"] == "overlap_unknown"]
        new_events = [r for r in rows if r["is_new_update_at_scenario_origin"]]
        event_groups = {r["update_group_id"] for r in new_events if r["update_group_id"]}
        event_origins = {r["origin"] for r in new_events}
        event_months = {str(r["origin"])[:7] for r in new_events}
        event_publication_boundaries = {r["publish_eligible_at_scenario"] for r in new_events
                                        if r.get("publish_eligible_at_scenario")}
        event_documents = {r["document_id"] for r in new_events if r.get("document_id")}
        documented_overlap = [r for r in known if r["target_overlap_status"] == "supported"]
        documented_origins = {r["origin"] for r in documented_overlap}
        documented_event_origins = {r["origin"] for r in new_events
                                    if r["target_overlap_status"] == "supported"}
        release_support.append({"evaluation_period": key[0], "target_year": key[1], "crop": key[2], "horizon": key[3],
            "planned_origin_count": len({r["origin"] for r in rows}),
            "strict_asof_fact_origin_count": len({r["origin"] for r in rows if r["strict_historical_asof"]}),
            "scenario_known_fact_origin_count": len({r["origin"] for r in known}),
            "scenario_known_unique_update_count": len({r["update_group_id"] for r in known if r["update_group_id"]}),
            "first_new_update_count": len(event_groups),
            "first_new_update_origin_count": len(event_origins),
            "first_new_update_month_count": len(event_months),
            "first_new_update_months": " | ".join(sorted(event_months)),
            "first_new_update_origins": " | ".join(sorted(event_origins)),
            "first_new_update_publication_boundaries": " | ".join(sorted(event_publication_boundaries)),
            "first_new_update_document_ids": " | ".join(sorted(event_documents)),
            "no_observed_new_update_origin_count": max(0, len({r["origin"] for r in rows}) - len(event_origins)),
            "actual_target_overlap_fact_origin_count": len({r["origin"] for r in overlap}),
            "overlap_unknown_fact_origin_count": len({r["origin"] for r in unknown}),
            "documented_effective_interval_fact_origin_count": len(documented_origins),
            "known_effective_overlap_non_event_origin_count": len(documented_origins - documented_event_origins),
            "comparison_target_count": len({(r["target_start"], r["target_end"]) for r in rows}),
            "interpretation": "scenario timing is not historical version proof; unknown intervals are not eligible future supply support"})

    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_create(output_dir / "facts.csv", _csv_bytes(reviewed, _FACT_FIELDS + ["candidate_ids", "scenario_available_origin_count", "repeat_reuse_count"]))
    parquet_data = io.BytesIO()
    pd.DataFrame(reviewed).to_parquet(parquet_data, index=False)
    _atomic_create(output_dir / "facts.parquet", parquet_data.getvalue())
    _atomic_create(output_dir / "fact-duplicate-links.csv", _csv_bytes(duplicate_links))
    _atomic_create(output_dir / "fact-candidate-links.csv", _csv_bytes(candidate_links))
    _atomic_create(output_dir / "vintage-evidence.csv", _csv_bytes([{k: r.get(k) for k in (
        "fact_id", "document_id", "source_hash", "document_date", "posted_at", "release_at", "distribution_at",
        "first_seen_at", "revision_at", "version_status", "asof_eligibility", "publish_eligible_at_scenario",
        "historical_version_evidence", "evidence_status", "exclusion_reason")} for r in reviewed]))
    _atomic_create(output_dir / "vintage-coverage.csv", _csv_bytes(vintage_coverage))
    _atomic_create(output_dir / "crop-season-coverage.csv", _csv_bytes(crop_coverage))
    _atomic_create(output_dir / "fact-origin-support.csv", _csv_bytes(support))
    _atomic_create(output_dir / "release-origin-support.csv", _csv_bytes(release_support))
    counts = {"reviewed_fact_count": len(reviewed),
        "unique_fact_count": len({r["fact_id"] for r in reviewed}),
        "unique_update_count": len({r["update_group_id"] for r in reviewed if r["update_group_id"]}),
        "scenario_eligible_fact_count": sum(r["asof_eligibility"] == "retrospective_publication_scenario" for r in reviewed),
        "scenario_eligible_unique_update_count": len({r["update_group_id"] for r in reviewed
            if r["update_group_id"] and r["asof_eligibility"] == "retrospective_publication_scenario"}),
        "ineligible_fact_count": sum(r["asof_eligibility"] == "ineligible" for r in reviewed),
        "pending_source_review_fact_count": sum(r["asof_eligibility"] == "pending_source_review" for r in reviewed),
        "strict_historical_fact_count": sum(r["asof_eligibility"] == "strict_historical_asof" for r in reviewed),
        "scenario_fact_count": sum(r["asof_eligibility"] == "retrospective_publication_scenario" for r in reviewed),
        "prospective_only_fact_count": sum(r["asof_eligibility"] == "prospective_only" for r in reviewed),
        "fact_rows_preserved_after_duplicate_check": len(reviewed),
        "candidate_link_count": len(candidate_links),
        "reviewed_fact_not_found_by_candidate_scan_count": sum(r["candidate_link_status"] != "candidate_numeric_crop_link" for r in reviewed),
        "candidate_queue_count": sum(1 for rows in candidates_by_doc_page.values() for _ in rows),
        "candidate_unlinked_count": max(0, sum(1 for rows in candidates_by_doc_page.values() for _ in rows)
            - len({r["candidate_id"] for r in candidate_links}))}
    _atomic_create(output_dir / "reviewed-fact-counts.json", (json.dumps(counts, indent=2) + "\n").encode())
    source_checks = _load_json(PHASE_A / "source-checks.json")
    phase_summary = _phase_a_summary(reviewed, support, release_support, source_checks)
    _atomic_create(output_dir / "phase-a-support-summary.json",
        (json.dumps(phase_summary, ensure_ascii=False, indent=2) + "\n").encode())
    # Read-back check for schema/row count protects against failed parquet write.
    if len(pd.read_parquet(output_dir / "facts.parquet")) != len(reviewed):
        raise ValueError("facts_parquet_readback_mismatch")
    return counts


def _atomic_create(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(data)


def _guess_crop(text: str, request: dict) -> tuple[str | None, str | None]:
    # Crop mention must be local to the evidence span; broad MAFRA search scope
    # is not itself evidence that an attachment contains cabbage facts.
    target_mentions = set()
    if _TARGET_CABBAGE_RE.search(text): target_mentions.add("cabbage")
    if re.search(r"월동무|봄무|가을무|겨울무|무가격동향|무\s*(?:수급|비축|출하|수확)|(?<![가-힣])무(?=[·\s:：의가를은로에서]|$)", text):
        target_mentions.add("radish")
    if len(target_mentions) > 1:
        return "mixed_crop_unallocated", None
    if re.search(r"월동무", text):
        return "radish", "winter_radish"
    if re.search(r"봄무", text):
        return "radish", "spring_radish"
    if re.search(r"가을무", text):
        return "radish", "fall_radish"
    if _TARGET_CABBAGE_RE.search(text):
        return "cabbage", None
    if re.search(r"월동무|봄무|가을무|겨울무|무가격동향|무\s*(?:수급|비축|출하|수확)|(?<![가-힣])무(?=[·\s:：의가를은로에서]|$)", text):
        return "radish", None
    return None, None


def _origin_flags(text: str, request: dict, source: str) -> list[str]:
    flags = ["current_capture_version_only", "first_publication_unproven"]
    if source == "mafra":
        flags.append("government_source_candidate_manual_origin_review")
    else:
        flags.append("jeju_extension_bulletin_manual_origin_review")
    if re.search(r"KREI|한국농촌경제연구원|농촌경제연구원|농업관측", text, re.I):
        flags.append("krei_or_agricultural_outlook_mention_review_required")
    if re.search(r"보도\s*시점", text):
        flags.append("embargo_or_release_time_text_review_required")
    return flags


def _section_crop(line: str, current_crop: str | None, source: str) -> str | None:
    # Explicit category labels delimit regions; local crop labels may then
    # update the active crop before following wrapped/value lines.
    if re.search(r"^\s*[▲△]\s*\(?(마늘|양파|깐마늘|당근|감자)", line) or re.search(r"❍\s*(양파|마늘|당근|감자)", line):
        return None
    if re.search(r"양배추", line):
        return None
    if re.search(r"(?:^|m\s*)(?:월동무|봄무|가을무)\s*[:：]", line):
        return "radish"
    if re.search(r"^\s*[▲△]\s*\(?(배추)", line):
        return "cabbage"
    if re.search(r"^\s*[▲△]\s*\(?(무)", line):
        return "radish"
    if _TARGET_CABBAGE_RE.search(line):
        if re.search(r"(?:가격동향|생육동향|재배동향|수확|출하|비축|수급|봄배추|가을배추|월동배추)", line):
            return "cabbage"
    if re.search(r"(월동무|봄무|가을무|겨울무|무가격동향|무 수급|무 비축|무 출하|무 수확)", line):
        return "radish"
    if re.search(r"월동채소", line):
        return None
    # A heading like '재배동향' inherits the immediately active crop; do not
    # infer crop from arbitrary occurrences of the Hangul syllable 무.
    if re.search(r"(당근|양파|감자|마늘)\s*(?:가격|생육|재배|수확|동향)", line):
        return None
    return current_crop


def _candidate_spans(page_text: str, request: dict, source: str, page_number: int) -> list[dict]:
    lines = [re.sub(r"\s+", " ", line).strip() for line in page_text.splitlines()]
    out = []
    active_crop = None
    for i, line in enumerate(lines):
        active_crop = _section_crop(line, active_crop, source)
        explicit_crop, variety = _guess_crop(line, request)
        line_crop = explicit_crop or active_crop
        # Reject incidental '무' (e.g. 사무국, 무름병); it must be a crop
        # heading or explicit crop label, or the line must inherit a known
        # crop-specific section.
        if line_crop is None or not line or not _VALUE_RE.search(line):
            continue
        # Do not enqueue price labels/amounts as supply evidence. Values are
        # candidates only when attached to a supply quantity, unit, or status.
        has_supply_term = bool(re.search(r"비축|방출|수확|재배면적|출하|계획|확보|공급|진행", line))
        if re.search(r"가격|공판장|도매가|경락|원/|평균가격", line) and not has_supply_term:
            continue
        matches = [m for m in _VALUE_RE.finditer(line) if _SUPPLY_UNIT_RE.search(m.group(0))]
        if not matches:
            continue
        # Keep exact extracted line and neighbors. This is evidence context,
        # not a parsed table row or fact claim.
        start, end = max(0, i - 1), min(len(lines), i + 2)
        span = " | ".join(x for x in lines[start:end] if x)
        if re.search(r"사무국|무름병|무더위|무단|무관|무조건", line):
            continue
        quantity_types = [name for name, rx in _TYPE_PATTERNS.items() if rx.search(span)]
        crop_mentions = [(m.start(), m.end(), m.group(0)) for m in _ALL_CROP_RE.finditer(line)]
        mixed_bundle = bool(re.search(r"배추[·ㆍ,、/\s]+무.*마늘.*양파|배추·무·마늘·양파", line))
        for match_index, match in enumerate(matches, 1):
            nearest = [r for r in crop_mentions if r[0] < match.start()]
            token_crop = _TARGET_CROP_NAMES.get(nearest[-1][2]) if nearest else line_crop
            if mixed_bundle and len(matches) == 1:
                token_crop, token_variety = "mixed_crop_unallocated", None
            else:
                token_variety = variety
            if token_crop not in {"cabbage", "radish", "mixed_crop_unallocated"}:
                continue
            out.append({
                "page": page_number, "evidence_span": span[:1400],
                "line_text": line[:800], "numeric_token": match.group(0),
                "value_candidate": None, "value_lower_candidate": None,
                "value_upper_candidate": None, "unit_candidate": None,
                "value_parse_status": "manual_review_required_no_midpoint_or_unit_inference",
                "candidate_group_id": f"p{page_number:04d}-l{i+1:04d}",
                "candidate_value_index": match_index,
                "crop_candidate": token_crop, "variety_or_crop_type_candidate": token_variety,
                "quantity_kind_candidates": quantity_types,
                "date_tokens_in_span": _DATE_RE.findall(span),
                "origin_flags": _origin_flags(span, request, source),
                "review_status": "unreviewed_candidate_not_fact",
            })
    return out


def build_page_candidates(output_dir: Path = PHASE_A / "page-review-v1") -> dict:
    import pdfplumber
    docs_cp, atts_cp = _load_json(DOC_CHECKPOINT), _load_json(ATT_CHECKPOINT)
    catalogs = {"jeju-agri-extension": _load_json(DOC_CATALOG), "mafra": _load_json(ATT_CATALOG)}
    cat_by_source_id = {source: {e["request_id"]: e for e in cat["entries"]}
                        for source, cat in catalogs.items()}
    checkpoint_dirs = {"jeju-agri-extension": (docs_cp, DATA / "documents"),
                       "mafra": (atts_cp, DATA / "attachments")}
    expected = {"jeju-agri-extension": 101, "mafra": 19}
    sources = {s: (cp, root) for s, (cp, root) in checkpoint_dirs.items()}
    inventory, candidates, source_checks = [], [], []
    page_text_rows = []
    raw_sha_owner: dict[str, list[str]] = defaultdict(list)
    for source in ("jeju-agri-extension", "mafra"):
        checkpoint, raw_root = sources[source]
        count = 0
        for request_id, record in sorted(checkpoint["requests"].items()):
            request = record.get("request", {})
            if record.get("status") != "captured" or not str(record.get("raw_file", "")).lower().endswith(".pdf"):
                continue
            raw_path = raw_root / record["raw_file"]
            raw = raw_path.read_bytes()
            actual_sha = sha256(raw)
            if actual_sha != record.get("sha256") or len(raw) != record.get("byte_length"):
                raise ValueError(f"source_integrity_mismatch:{request_id}")
            catalog_entry = cat_by_source_id[source].get(request_id)
            if not catalog_entry or catalog_entry.get("raw_sha256") != actual_sha:
                raise ValueError(f"catalog_association_mismatch:{request_id}")
            pdf = pdfplumber.open(io.BytesIO(raw))
            if len(pdf.pages) > 1000:
                raise ValueError(f"page_limit_exceeded:{request_id}")
            count += 1
            raw_sha_owner[actual_sha].append(request_id)
            for page_index, page in enumerate(pdf.pages, 1):
                text = page.extract_text(layout=False) or ""
                page_cands = _candidate_spans(text, request, source, page_index)
                page_text_rows.append({"document_id": request_id, "source": source,
                    "page": page_index, "source_hash": actual_sha, "page_text": text})
                listing_context = (request.get("row_context") or request.get("source_list_row_context") or "")
                post_match = _POSTED_RE.search(listing_context)
                posted_candidate = (f"{post_match.group(1)}-{int(post_match.group(2)):02d}-{int(post_match.group(3)):02d}T"
                    f"{int(post_match.group(4)):02d}:{post_match.group(5)}:00+09:00" if post_match else catalog_entry.get("listing_date"))
                candidates.extend({
                    "fact_candidate_id": f"{request_id}:p{page_index:04d}:{c['candidate_group_id']}:v{c['candidate_value_index']:02d}",
                    "candidate_group_id": f"{request_id}:{c['candidate_group_id']}",
                    "document_id": request_id,
                    "source": source,
                    "source_org_candidate": "MAFRA" if source == "mafra" else "Jeju Special Self-Governing Province Agricultural Research and Extension Services",
                    "source_url": record.get("final_url") or record.get("url"),
                    "source_hash": actual_sha,
                    "page": page_index,
                    "document_date_candidate": None,
                    "source_listing_row": listing_context,
                    "posted_at_candidate": posted_candidate,
                    "listing_date_catalog_value": catalog_entry.get("listing_date"),
                    "first_seen_at": record.get("retrieved_at"),
                    **c,
                } for n, c in enumerate(page_cands, 1))
                inventory.append({
                    "document_id": request_id, "source": source,
                    "source_title": catalog_entry.get("source_title"),
                    "listing_date_candidate": catalog_entry.get("listing_date"),
                    "source_url": record.get("final_url") or record.get("url"),
                    "source_hash": actual_sha, "raw_file": str(raw_path.relative_to(ROOT)),
                    "page": page_index, "page_text_chars": len(text),
                    "numeric_crop_candidate_count": len(page_cands),
                    "krei_or_outlook_mention": bool(re.search(r"KREI|한국농촌경제연구원|농촌경제연구원|농업관측", text, re.I)),
                    "origin_flags": ";".join(_origin_flags(text, request, source)),
                    "first_seen_at": record.get("retrieved_at"),
                    "render_suggestion": f"outputs/experiments/period-supply-document-signal-v8/phase-A/page-review-v1/render/{source}/{request_id}-p{page_index:04d}.png",
                })
            source_checks.append({"document_id": request_id, "source": source,
                                 "raw_sha256": actual_sha, "raw_byte_length": len(raw),
                                 "catalog_raw_sha256": catalog_entry.get("raw_sha256"),
                                 "catalog_source_title": catalog_entry.get("source_title"),
                                 "retrieved_at": record.get("retrieved_at"),
                                 "http_status": record.get("http_status"), "page_count": len(pdf.pages),
                                 "integrity_status": "verified_sha256_catalog_and_checkpoint"})
        if count != expected[source]:
            raise ValueError(f"expected_pdf_count_mismatch:{source}:{count}")
    duplicate_groups = [ids for ids in raw_sha_owner.values() if len(ids) > 1]
    for group in duplicate_groups:
        for row in source_checks:
            if row["document_id"] in group:
                row["duplicate_raw_sha_group"] = group
    payload = {"protocol": "period-supply-document-signal-v8-page-review-v1",
               "status": "unreviewed_candidate_evidence_only",
               "network_requests": 0, "model_fits": 0,
               "scope": {"jeju_pdf_count": 101, "mafra_pdf_count": 19,
                         "original_pdf_count": len(source_checks), "pdf_page_count": len(inventory),
                         "candidate_span_count": len(candidates)},
               "limitations": ["PDF text extraction is a candidate-finding aid, not visual review.",
                               "No fact is human-verified; crop, type, source origin, units, and values need page review.",
                               "listing_date_candidate is not proven first publication time or historical vintage evidence.",
                               "No as-of-0 feature use or price modeling is performed."],
               "source_checks": source_checks, "page_candidates": candidates}
    _atomic_create(output_dir / "page-candidates.json",
                   (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode())
    with (output_dir / "page-text.jsonl").open("x", encoding="utf-8") as f:
        for row in page_text_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    inv_path = output_dir / "page-inventory.csv"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    with inv_path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(inventory[0]))
        writer.writeheader(); writer.writerows(inventory)
    # A human-review queue with explicit blank verdict fields. The candidates
    # remain unchanged; reviewers can annotate a copy without implying that
    # automated extraction confirmed the fact.
    review_fields = ["fact_candidate_id", "candidate_group_id", "document_id", "source", "source_org_candidate",
                     "source_url", "source_hash", "page", "source_listing_row", "evidence_span", "numeric_token",
                     "value_candidate", "value_lower_candidate", "value_upper_candidate", "unit_candidate", "value_parse_status",
                     "crop_candidate", "variety_or_crop_type_candidate", "quantity_kind_candidates",
                     "date_tokens_in_span", "posted_at_candidate", "listing_date_catalog_value", "first_seen_at", "origin_flags",
                     "render_suggestion", "human_review_status", "verified_fact_id",
                     "crop_verified", "variety_verified", "region_verified", "quantity_kind_verified",
                     "value_lower_verified", "value_upper_verified", "unit_verified",
                     "denominator_verified", "origin_verified", "date_fields_verified", "review_notes"]
    review_path = output_dir / "candidate-review-queue.csv"
    with review_path.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=review_fields, extrasaction="ignore")
        writer.writeheader()
        for row in candidates:
            writer.writerow({**row,
                             "quantity_kind_candidates": " | ".join(row["quantity_kind_candidates"]),
                             "date_tokens_in_span": " | ".join(row["date_tokens_in_span"]),
                             "origin_flags": " | ".join(row["origin_flags"]),
                             "human_review_status": "pending_visual_review"})
    checks_path = output_dir / "source-checks.json"
    _atomic_create(checks_path, (json.dumps(_validate_all_captures(source_checks),
        ensure_ascii=False, indent=2) + "\n").encode())
    return payload["scope"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=PHASE_A / "page-review-v1")
    args = parser.parse_args()
    print(json.dumps(build_page_candidates(args.output_dir), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
