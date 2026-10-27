"""Offline attachment selection and usefulness index for captured supply documents."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from urllib.parse import unquote

from agri_data.supply_document_collection import sha256, write_immutable

PROTOCOL = "supply-document-catalog-v1"
_DATE_PATTERNS = (
    re.compile(r"(?<!\d)(20\d{2})\s*[./-]\s*(\d{1,2})\s*[./-]\s*(\d{1,2})(?!\d)"),
    re.compile(r"(20\d{2})년\s*(\d{1,2})월\s*(\d{1,2})일?"),
)
_LABEL_RE = re.compile(r"(?:보도시점|배포|작성일|게시일|등록일|조사일|기준일|관측일)")
_KREI_RE = re.compile(r"KREI|한국농촌경제연구원|농촌경제연구원|농업관측(?:정보)?", re.IGNORECASE)
_KEYWORD_PATTERNS = {
    "winter_radish": re.compile(r"월동무"),
    "radish": re.compile(r"(?<![가-힣])무(?:[)）]|\s*(?:생산|재배|수확|출하|작황|면적|가격|농가))"),
    "cabbage": re.compile(r"(?<!양)배추"),
    "harvest_progress": re.compile(r"수확.{0,24}(?:%|％|퍼센트|완료|진행|률)|(?:%|％|퍼센트).{0,24}수확|수확률|수확 완료율"),
    "stock_release_or_contract": re.compile(r"비축|방출량?|출하약정|수매약정|공급량|공급계획|가용물량"),
}


def _date_strings(text: str) -> list[str]:
    found = []
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0)
            if value not in found:
                found.append(value)
    return found


def _excerpts(text: str, pattern: re.Pattern, *, limit: int = 3, radius: int = 100) -> list[str]:
    results = []
    for match in pattern.finditer(text):
        start, end = max(0, match.start() - radius), min(len(text), match.end() + radius)
        excerpt = re.sub(r"\s+", " ", text[start:end]).strip()
        if excerpt and excerpt not in results:
            results.append(excerpt)
        if len(results) >= limit:
            break
    return results


def _labeled_dates(text: str) -> list[dict]:
    rows = []
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            before = text[max(0, match.start() - 50):match.start()]
            label_match = _LABEL_RE.search(before)
            row = {"date_text": match.group(0), "label": label_match.group(0) if label_match else None,
                   "context": re.sub(r"\s+", " ", text[max(0, match.start() - 70):min(len(text), match.end() + 30)]).strip()}
            if row not in rows:
                rows.append(row)
    return rows


def build_attachment_manifest(candidate_path: Path, output_path: Path) -> dict:
    candidates_doc = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidates_doc.get("protocol") != "offline-supply-link-discovery-v1":
        raise ValueError("candidate_protocol_mismatch")
    checkpoint_path = Path(candidates_doc["source_checkpoint"])
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    by_parent = {}
    rejected = []
    for candidate in candidates_doc.get("candidates", []):
        parent_id = candidate.get("source_request_id")
        parent = checkpoint.get("requests", {}).get(parent_id)
        if not parent or parent.get("status") != "captured":
            continue
        if parent.get("sha256") != candidate.get("source_page_sha256"):
            raise ValueError("candidate_source_hash_mismatch")
        source = parent.get("request", {}).get("source")
        if source != "mafra":
            continue
        filename_text = " ".join((candidate.get("url", ""), candidate.get("link_text", ""), candidate.get("row_context", "")))
        extension_match = re.search(r"(?i)([^/\\\s\"'<>?&=]+\.(?:pdf|hwpx?|hwp))(?=$|[\s\"'<>?&=])", unquote(filename_text))
        ext = extension_match.group(1).rsplit(".", 1)[-1].lower() if extension_match else None
        if ext not in {"pdf", "hwp", "hwpx"}:
            rejected.append({"candidate_id": candidate.get("candidate_id"), "reason": "filename_extension_not_visible"})
            continue
        by_parent.setdefault(parent_id, []).append((candidate, parent, ext, extension_match.group(1)))
    selected = []
    parent_decisions = []
    detail_ids = {rid for rid, record in checkpoint.get("requests", {}).items()
                  if record.get("status") == "captured" and record.get("request", {}).get("source") == "mafra"
                  and record.get("request", {}).get("candidate_kind") == "detail_page"}
    for parent_id in sorted(detail_ids | set(by_parent)):
        options = by_parent.get(parent_id, [])
        pdfs = [item for item in options if item[2] == "pdf"]
        chosen = pdfs if pdfs else [item for item in options if item[2] in {"hwp", "hwpx"}]
        parent_decisions.append({"source_detail_request_id": parent_id,
                                 "pdf_candidates": len(pdfs), "fallback_candidates": len(options) - len(pdfs),
                                 "selected_count": len(chosen), "selection": "pdf" if pdfs else ("hwp_fallback" if chosen else "no_supported_attachment")})
        for candidate, parent, ext, filename in chosen:
            request = {"id": candidate["candidate_id"], "url": candidate["url"], "kind": "document",
                       "source": "mafra", "topic": parent.get("request", {}).get("topic"),
                       "filename_hint": filename, "filename_extension": ext,
                       "source_detail_request_id": parent_id,
                       "source_detail_sha256": parent.get("sha256"),
                       "source_detail_url": parent.get("url"),
                       "source_list_request_id": parent.get("request", {}).get("source_request_id"),
                       "source_list_page_sha256": parent.get("request", {}).get("source_page_sha256"),
                       "source_list_row_context": parent.get("request", {}).get("row_context")}
            selected.append(request)
    result = {"protocol": "public-supply-document-capture-v1",
              "scope_id": "mafra-supply-document-attachments-from-captured-details",
              "scope_description": "Attachment URLs selected from captured MAFRA detail HTML; PDF takes precedence when visible by filename extension, HWP/HWPX are retained only when no PDF candidate exists for that detail page.",
              "scope_enumeration_complete": True,
              "scope_limitations": ["Only the MAFRA details present in the captured request checkpoint were examined.",
                  "No attachment outside the fixed HTTPS allowlist is included.",
                  "HTML detail pages remain preserved even when a PDF attachment is selected.",
                  "HWP/HWPX fallback originals are preserved but this extractor does not convert them to text.",
                  "No selected document is promoted to a model input."],
              "source_candidates_file_sha256": sha256(candidate_path.read_bytes()),
              "source_checkpoint_manifest_sha256": checkpoint.get("manifestSha256"),
              "detail_page_count": len(detail_ids), "parent_decisions": parent_decisions, "rejected_candidates": rejected,
              "expected_request_count": len(selected), "requests": selected}
    write_immutable(output_path, (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return result


def build_catalog(checkpoint_path: Path, extraction_path: Path, output_path: Path) -> dict:
    checkpoint_path = checkpoint_path.resolve()
    root = checkpoint_path.parent
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    if extraction.get("sourceManifestSha256") != checkpoint.get("manifestSha256"):
        raise ValueError("extraction_source_manifest_mismatch")
    extraction_by_id = {row["id"]: row for row in extraction.get("outcomes", [])}
    entries = []
    for request_id, record in sorted(checkpoint.get("requests", {}).items()):
        raw_path = root / record["raw_file"] if record.get("raw_file") else None
        raw_sha = record.get("sha256")
        if record.get("status") != "captured" or raw_path is None or not raw_path.is_file():
            entries.append({"request_id": request_id, "capture_status": record.get("status"), "catalog_status": "source_missing"})
            continue
        raw = raw_path.read_bytes()
        if sha256(raw) != raw_sha:
            raise ValueError("raw_hash_mismatch")
        extracted = extraction_by_id.get(request_id, {})
        if extracted and extracted.get("sourceSha256") != raw_sha:
            raise ValueError("extracted_source_hash_mismatch")
        text_path = root / extracted["textFile"] if extracted.get("textFile") else None
        text = ""
        text_sha = extracted.get("textSha256")
        if text_path and text_path.is_file():
            text_bytes = text_path.read_bytes()
            if sha256(text_bytes) != text_sha:
                raise ValueError("extracted_text_hash_mismatch")
            text = text_bytes.decode("utf-8", errors="replace")
        request = record.get("request", {})
        row_context = (request.get("row_context") or request.get("source_list_row_context") or "")
        searchable = text
        keyword_hits = {name: _excerpts(searchable, pattern) for name, pattern in _KEYWORD_PATTERNS.items()}
        krei_hits = _excerpts(searchable, _KREI_RE)
        label_contexts = _excerpts(searchable, _LABEL_RE, limit=10, radius=70)
        date_candidates = _labeled_dates(searchable)
        list_date_candidates = _date_strings(row_context)
        all_hits = [value for values in keyword_hits.values() for value in values]
        body_excerpt = (all_hits[0] if all_hits else re.sub(r"\s+", " ", searchable[:240]).strip())
        crop = {"cabbage": bool(keyword_hits["cabbage"]),
                "radish": bool(keyword_hits["radish"] or keyword_hits["winter_radish"]),
                "winter_radish": bool(keyword_hits["winter_radish"])}
        report_dates = [d["date_text"] for d in date_candidates if d["label"] in {"작성일", "조사일", "기준일"}]
        quantity_types = [name for name in ("harvest_progress", "stock_release_or_contract") if keyword_hits[name]]
        mentions_krei = bool(krei_hits)
        entries.append({"request_id": request_id, "request_kind": request.get("kind"),
                        "source": request.get("source"), "source_url": record.get("url"),
                        "source_title": (request.get("row_context") or request.get("link_text")
                                         or request.get("source_list_row_context") or request.get("filename_hint")), "source_excerpt": body_excerpt,
                        "source_listing_row": row_context,
                        "document_url": record.get("url"), "raw_file": record.get("raw_file"),
                        "raw_sha256": raw_sha, "raw_byte_length": record.get("byte_length"),
                        "retrieved_at": record.get("retrieved_at"),
                        "report_date": report_dates[0] if report_dates else None,
                        "report_date_candidates": date_candidates,
                        "listing_date": list_date_candidates[0] if list_date_candidates else None,
                        "listing_date_candidates": list_date_candidates,
                        "date_label_excerpts": label_contexts,
                        "first_publication_time": None,
                        "historical_availability": "unknown_current_retrieval_only",
                        "crop_keyword_hits": crop, "keyword_excerpts": keyword_hits,
                        "quantity_type_hits": quantity_types,
                        "krei_flag": mentions_krei, "krei_mentions": mentions_krei, "krei_mention_excerpts": krei_hits,
                        "source_origin_review_required": mentions_krei,
                        "source_origin_signals": sorted({term for term in ("KREI", "한국농촌경제연구원", "농촌경제연구원", "농업관측", "농업관측정보") if re.search(term, searchable, re.IGNORECASE)}),
                        "krei_origin_review": "manual_review_required" if mentions_krei else "not_established_by_keyword_scan",
                        "model_input_status": "discovery_only_not_approved",
                        "text_status": extracted.get("status", "not_extracted"),
                        "text_file": extracted.get("textFile"), "text_sha256": text_sha,
                        "text_extraction_method": extracted.get("method"),
                        "text_extraction_truncated": extracted.get("truncated", False)})
    result = {"protocol": PROTOCOL, "source_manifest_sha256": checkpoint.get("manifestSha256"),
              "source_checkpoint": str(checkpoint_path), "extraction_manifest": str(extraction_path),
              "entry_count": len(entries), "entries": entries,
              "limitations": ["Keyword excerpts support manual source screening; they are not validated structured quantities.",
                  "A KREI mention triggers review but does not establish data provenance; absence of a match does not prove no KREI-derived content.",
                  "Report/listing dates are date candidates only. First public availability remains unknown unless independently proven.",
                  "No content is an approved model input or evidence of forecast benefit."]}
    write_immutable(output_path, (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("attachments")
    a.add_argument("--candidates", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    c = sub.add_parser("catalog")
    c.add_argument("--checkpoint", type=Path, required=True)
    c.add_argument("--extraction", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "attachments":
        result = build_attachment_manifest(args.candidates, args.output)
        print(json.dumps({"manifest": str(args.output), "requests": result["expected_request_count"],
                          "details": len(result["parent_decisions"])}, ensure_ascii=False))
    else:
        result = build_catalog(args.checkpoint, args.extraction, args.output)
        print(json.dumps({"catalog": str(args.output), "entries": result["entry_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
