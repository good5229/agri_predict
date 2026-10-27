"""Offline integrity and human-review workflow for the fixed-window supply corpus.

PDF/text matches are review candidates, never accepted facts. This module has no
network, price-scoring, forecasting, or model-fitting code.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import shutil
import subprocess
import csv
from collections import Counter, defaultdict
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20261016T115223+0900-26c5b2"
RUN_ROOT = ROOT / "data/raw/supply_documents_evaluation_20261016/runs" / RUN_ID
OUTPUT_ROOT = ROOT / "outputs/experiments/period-supply-document-signal-v8/eval-audit-20261016"
PRIOR_PHASE_A = ROOT / "outputs/experiments/period-supply-document-signal-v8/phase-A"
PRIOR_MODULE = ROOT / "agri_data/period_supply_document_v8.py"
PHASE_A_PROTOCOL = PRIOR_PHASE_A / "protocol.md"
PROTOCOL = ROOT / "outputs/experiments/period-supply-document-signal-v8/eval-audit-20261016/protocol.md"
EXPECTED_OBSERVATIONS = 829

POPPLER = Path("/Users/bellhundred/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin")

# Broad enough to queue price/supply tables with wrapped row labels, but every
# span remains unparsed and unapproved until source review.
_CROP_PATTERNS = {
    "cabbage": re.compile(r"(?<!양)(?:봄|여름|가을|겨울|월동|고랭지)?배추"),
    "radish": re.compile(r"월동무|봄무|가을무|겨울무|무가격동향|(?<![가-힣])무(?![가-힣])"),
}
_NUMERIC_RE = re.compile(
    r"(?<![\d])(?:약\s*)?\d[\d,]*(?:\.\d+)?\s*(?:~|∼|～|에서|부터|[-–—])\s*\d[\d,]*(?:\.\d+)?\s*(?:천톤|만톤|톤|ha|㏊|%|퍼센트|kg|㎏|원|만주|주)?"
    r"|(?<![\d])\d+\s*만\s*\d+\s*천톤"
    r"|(?<![\d])(?:약\s*)?\d[\d,]*(?:\.\d+)?\s*(?:천톤|만톤|톤|ha|㏊|%|퍼센트|kg|㎏|원|만주|주)"
)
_SUPPLY_RE = re.compile(r"비축|재고|방출|출하|공급|수급|생산|재배|면적|수확|작황|반입|출고|생육|수매|수급조절|가격|전망|계약재배|입고")
_RELEASE_RE = re.compile(r"보도\s*시점|보도\s*자료|배포\s*시|엠바고|공개\s*시점|발표\s*시점|보도일시", re.I)
_DATE_RE = re.compile(r"20\d{2}[.년/-]\s*\d{1,2}[.월/-]\s*\d{1,2}(?:일)?\.?(?:\s*\([^)]*\))?(?:\s*\d{1,2}:\d{2}(?::\d{2})?)?")
_KREI_RE = re.compile(r"KREI|한국농촌경제연구원|농촌경제연구원|농업관측", re.I)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _tool(name: str) -> str:
    found = shutil.which(name)
    fallback = POPPLER / name
    if found:
        return found
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError(f"{name}_unavailable")


def _object_path(run_root: Path, obs: dict) -> Path:
    rel = Path(obs.get("object_path") or "")
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError("invalid_object_path")
    # Collector checkpoints store object paths relative to the collection
    # corpus root (the parent of ``runs/``), not the individual run folder.
    object_root = run_root.parent.parent if run_root.parent.name == "runs" else run_root
    path = object_root / rel
    if not path.is_file():
        raise FileNotFoundError("captured_object_missing")
    return path


def _source(obs: dict) -> str:
    return str(obs.get("metadata", {}).get("source") or "unknown")


def _is_pdf(obs: dict, body: bytes | None = None) -> bool:
    meta = obs.get("metadata", {})
    headers = {str(k).lower(): str(v).lower() for k, v in (obs.get("headers") or {}).items()}
    return (meta.get("kind") == "pdf" or "application/pdf" in str(meta.get("content_type", "")).lower()
            or "application/pdf" in headers.get("content-type", "") or bool(body and body.startswith(b"%PDF-")))


def _doc_id(obs: dict) -> str:
    src = _source(obs)
    prefix = "jeju" if src == "jeju-weekly" else "mafra" if src == "mafra-title" else re.sub(r"[^a-z0-9]+", "-", src.lower()).strip("-") or "source"
    return f"{prefix}-eval-{sha256(str(obs.get('url', '')).encode())[:16]}"


def _run_inputs(run_root: Path) -> tuple[dict, dict]:
    checkpoint = _json(run_root / "checkpoint.json")
    run = _json(run_root / "run.json")
    if checkpoint.get("run_id") != RUN_ID or run.get("run_id") != RUN_ID:
        raise ValueError("fixed_run_id_mismatch")
    if checkpoint.get("status") != "complete" or run.get("status") != "complete":
        raise ValueError("fixed_run_not_complete")
    scope = checkpoint.get("scope", {})
    if (scope.get("scope_mode"), scope.get("window_start"), scope.get("window_end"), scope.get("jeju_years"), scope.get("queries")) != (
            "fixed_window", "2024-01-01", "2026-09-30", [2024, 2025, 2026], ["배추", "무", "수급"]):
        raise ValueError("fixed_run_scope_mismatch")
    return checkpoint, run


def audit_integrity(run_root: Path = RUN_ROOT, expected_observations: int | None = EXPECTED_OBSERVATIONS) -> dict:
    """Verify each captured observation and retain failures as explicit rows."""
    cp, run = _run_inputs(run_root)
    observations = cp.get("observations", [])
    if expected_observations is not None and len(observations) != expected_observations:
        raise ValueError(f"unexpected_observation_count:{len(observations)}")
    rows, pdf_hashes, url_ids = [], defaultdict(list), {}
    for ordinal, obs in enumerate(observations, 1):
        url = str(obs.get("url") or "")
        if not url or urlparse(url).scheme != "https":
            raise ValueError(f"invalid_observation_url:{ordinal}")
        src = _source(obs)
        pdf = _is_pdf(obs)
        row = {"observation_ordinal": ordinal, "url": url, "kind": obs.get("kind"), "source": src,
               "metadata": obs.get("metadata", {}), "ok": bool(obs.get("ok")), "http_status": obs.get("status"),
               "started_at": obs.get("started_at"), "retrieved_at": obs.get("retrieved_at"),
               "capture_origin": obs.get("capture_origin"), "reused_from": obs.get("reused_from"),
               "original_run_id": obs.get("original_run_id"), "sha256_expected": obs.get("sha256"),
               "byte_length_expected": obs.get("byte_length"), "object_path": obs.get("object_path"),
               "is_pdf": pdf, "integrity_status": "not_captured",
               "doc_id": _doc_id(obs) if pdf or obs.get("kind") in {"detail", "listing"} else None}
        if obs.get("ok"):
            try:
                path = _object_path(run_root, obs)
                body = path.read_bytes()
                digest = sha256(body)
                row["byte_length_actual"] = len(body)
                row["sha256_actual"] = digest
                if digest != obs.get("sha256") or len(body) != obs.get("byte_length"):
                    row["integrity_status"] = "hash_or_length_mismatch"
                elif pdf and not body.startswith(b"%PDF-"):
                    row["integrity_status"] = "pdf_signature_mismatch"
                else:
                    row["integrity_status"] = "verified"
                if pdf:
                    pdf_hashes[digest].append(row["doc_id"])
                row["object_local_path"] = str(path)
            except (OSError, ValueError) as exc:
                row["integrity_status"] = f"object_read_failed:{type(exc).__name__}"
        rows.append(row)
        if pdf:
            url_ids[url] = row["doc_id"]
    counts = Counter((r["kind"], r["integrity_status"]) for r in rows)
    expected_hash_rows = [r for r in rows if r["ok"]]
    return {"protocol": "supply-document-eval-audit-v1", "scope_status": "offline_fixed_window_capture_integrity_audit",
            "run_id": RUN_ID, "run_status": cp.get("status"), "run_scope": cp.get("scope"),
            "protocol_sha256": sha256(PROTOCOL.read_bytes()) if PROTOCOL.is_file() else None,
            "phase_a_protocol_sha256": sha256(PHASE_A_PROTOCOL.read_bytes()) if PHASE_A_PROTOCOL.is_file() else None,
            "observation_count": len(rows), "observation_kind_counts": dict(Counter(r["kind"] for r in rows)),
            "verified_capture_count": sum(r["integrity_status"] == "verified" for r in rows),
            "pdf_observation_count": sum(r["is_pdf"] for r in rows),
            "pdf_distinct_hash_count": len(pdf_hashes),
            "pdf_duplicate_hash_groups": [{"sha256": h, "doc_ids": ids} for h, ids in pdf_hashes.items() if len(ids) > 1],
            "status_counts": {f"{k[0]}:{k[1]}": v for k, v in counts.items()},
            "integrity_pass": bool(rows) and all(r["integrity_status"] == "verified" for r in rows),
            "observations": rows}


def _pdf_info_page_count(path: Path, runner=subprocess.run) -> int:
    p = runner([_tool("pdfinfo"), str(path)], capture_output=True, timeout=45, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"pdfinfo_failed:{p.returncode}")
    m = re.search(rb"(?m)^Pages:\s+(\d+)", p.stdout)
    if not m:
        raise RuntimeError("pdfinfo_page_count_missing")
    return int(m.group(1))


def _extract_pdf_pages(path: Path, runner=subprocess.run) -> tuple[int, list[str], str | None]:
    """Extract the full PDF once; preserve page boundaries and per-page hashes later."""
    pages = _pdf_info_page_count(path, runner)
    p = runner([_tool("pdftotext"), "-enc", "UTF-8", "-layout", str(path), "-"],
               capture_output=True, timeout=90, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"pdftotext_failed:{p.returncode}")
    chunks = p.stdout.decode("utf-8", errors="replace").split("\f")
    if chunks and not chunks[-1].strip():
        chunks.pop()
    error = f"page_count_mismatch:pdfinfo={pages}:text={len(chunks)}" if len(chunks) != pages else None
    # The page count from pdfinfo stays authoritative. The caller records
    # missing page text explicitly and never joins or silently truncates it.
    return pages, chunks[:pages], error


def _crop_mentions(text: str) -> list[str]:
    return [crop for crop, pattern in _CROP_PATTERNS.items() if pattern.search(text)]


def _page_candidate(page_text: str, page_number: int, doc: dict) -> dict | None:
    crops = _crop_mentions(page_text)
    numbers = list(_NUMERIC_RE.finditer(page_text))
    lines = page_text.splitlines()
    supply = bool(_SUPPLY_RE.search(page_text))
    spans = []
    for i, line in enumerate(lines, 1):
        if not _NUMERIC_RE.search(line):
            continue
        # Keep nearby section-label context for wrapped source tables. No value
        # parsing or midpoint/unit conversion occurs here.
        context = " | ".join(x.strip() for x in lines[max(0, i-2):min(len(lines), i+1)] if x.strip())
        if _crop_mentions(context) or (crops and supply and _SUPPLY_RE.search(context)):
            for m in _NUMERIC_RE.finditer(line):
                spans.append({"line_number": i, "numeric_token": m.group(0), "evidence_span": context[:1200],
                              "crop_candidates": _crop_mentions(context) or crops,
                              "value_candidate": None, "unit_candidate": None,
                              "review_status": "unreviewed_candidate_not_fact"})
    release = doc.get("source") == "mafra-title" and page_number == 1 and bool(_RELEASE_RE.search(page_text))
    # Numeric supply candidates require a crop label plus a supply context at
    # page level. Release/embargo first pages remain reviewable independently.
    relevant = bool(spans and crops and supply) or release
    if not relevant:
        return None
    dates = []
    for m in _DATE_RE.finditer(page_text):
        a, b = max(0, m.start()-80), min(len(page_text), m.end()+80)
        dates.append({"date_token_candidate": m.group(0), "raw_quote": page_text[a:b].strip(),
                      "position": m.start(), "review_status": "unverified_source_timestamp_candidate"})
    return {"candidate_id": f"{doc['doc_id']}:p{page_number:04d}", "document_id": doc["doc_id"],
            "source": doc["source"], "source_url": doc["url"], "source_hash": doc["sha256"],
            "page": page_number, "page_text_sha256": sha256(page_text.encode("utf-8")),
            "crop_candidates": crops, "numeric_span_candidates": spans,
            "page_candidate_status": "unreviewed_candidate_not_fact",
            "candidate_reasons": [*(["crop_supply_numeric_page"] if spans and crops and supply else []),
                                  *(["government_release_or_embargo_first_page"] if release else [])],
            "release_or_embargo_terms_present": release, "timestamp_candidates": dates,
            "page_image": None}


def _observation_docs(integrity: dict) -> tuple[list[dict], dict[str, dict], dict]:
    docs = [dict(r) for r in integrity["observations"] if r["is_pdf"] and r["integrity_status"] == "verified"]
    details_by_hash = defaultdict(list)
    details = {}
    for r in integrity["observations"]:
        if r["kind"] == "detail" and r["integrity_status"] == "verified":
            digest = r.get("sha256_actual") or r.get("sha256_expected")
            if digest:
                details_by_hash[digest].append(r)
                details[r["doc_id"]] = r
    links = {"matched_unique": 0, "unmatched": [], "ambiguous": []}
    for d in docs:
        d["detail_document_id"] = None
        if d["source"] == "mafra-title":
            digest = d.get("metadata", {}).get("detail_sha256")
            matches = details_by_hash.get(digest, [])
            if len(matches) == 1:
                d["detail_document_id"] = matches[0]["doc_id"]
                d["parent_detail_link_status"] = "matched_by_detail_sha256"
                links["matched_unique"] += 1
            elif len(matches) > 1:
                d["parent_detail_link_status"] = "ambiguous_detail_sha256"
                d["parent_detail_candidate_ids"] = [r["doc_id"] for r in matches]
                links["ambiguous"].append(d["doc_id"])
            else:
                d["parent_detail_link_status"] = "unmatched_detail_sha256"
                links["unmatched"].append(d["doc_id"])
        else:
            d["parent_detail_link_status"] = "not_applicable"
    return docs, details, links


class _HTMLVisible(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts=[]; self.hidden=0
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}: self.hidden += 1
        if tag in {"p", "br", "div", "li", "tr", "dd", "dt", "h1", "h2", "h3"}: self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.hidden: self.hidden -= 1
    def handle_data(self, data):
        if not self.hidden: self.parts.append(data)


def _html_timestamp_candidates(body: bytes, record: dict) -> list[dict]:
    """Capture raw visible HTML quote around timestamp; never label it vintage proof."""
    raw = body.decode("utf-8", errors="replace")
    parser = _HTMLVisible(); parser.feed(raw); text = html.unescape("".join(parser.parts))
    found = []
    for m in _DATE_RE.finditer(text):
        quote = text[max(0, m.start()-100):min(len(text), m.end()+100)].strip()
        found.append({"source_document_id": record["doc_id"], "source_url": record["url"],
                      "source_hash": record.get("sha256_actual") or record.get("sha256_expected"), "source_page": None,
                      "timestamp_text_candidate": m.group(0), "raw_quote": quote,
                      "candidate_status": "current_displayed_html_date_not_vintage_proof"})
    return found


def build_review_queue(run_root: Path = RUN_ROOT, output_dir: Path = OUTPUT_ROOT, *, render: bool = True,
                       expected_observations: int | None = EXPECTED_OBSERVATIONS) -> dict:
    """Full fixed-corpus offline audit; call only after frozen review scope approval."""
    cp, _run = _run_inputs(run_root)
    integrity = audit_integrity(run_root, expected_observations)
    if not integrity["integrity_pass"]:
        raise ValueError("capture_integrity_not_complete")
    doc_rows, detail_rows, parent_links = _observation_docs(integrity)
    # Add HTML timestamp quotes to the evidence index, kept separate from PDF
    # issue/release cues; capture time and posted display time remain distinct.
    for detail in detail_rows.values():
        detail["html_timestamp_candidates"] = _html_timestamp_candidates(_object_path(run_root, detail).read_bytes(), detail)
    pdf_rows, candidates, errors = [], [], []
    for doc in doc_rows:
        path = Path(doc["object_local_path"])
        entry = {k: doc.get(k) for k in ("doc_id", "source", "url", "sha256_expected", "byte_length_expected", "retrieved_at", "started_at", "capture_origin", "reused_from", "original_run_id", "metadata", "detail_document_id")}
        entry.update({"sha256": doc["sha256_actual"], "raw_path": str(path), "page_count": None,
                      "extraction_status": "pending", "page_hashes": [], "page_checks": [], "candidate_page_count": 0,
                      "extraction_error": None})
        try:
            pages = _pdf_info_page_count(path)
            entry["page_count"] = pages
            _pages, texts, page_error = _extract_pdf_pages(path)
            for page in range(1, pages + 1):
                if page > len(texts):
                    entry["page_checks"].append({"page": page, "status": "extraction_error", "error": page_error or "page_text_missing"})
                    continue
                text = texts[page-1]
                digest = sha256(text.encode("utf-8"))
                entry["page_hashes"].append({"page": page, "text_sha256": digest, "text_characters": len(text)})
                page_status = "extracted" if text.strip() else "empty_text_unsearchable"
                entry["page_checks"].append({"page": page, "status": page_status, "text_sha256": digest})
                candidate = _page_candidate(text, page, entry)
                if candidate:
                    candidate["page_image"] = f"evidence-pages/{entry['doc_id']}-p{page:04d}.png" if render else None
                    candidates.append(candidate)
            entry["candidate_page_count"] = len([c for c in candidates if c["document_id"] == entry["doc_id"]])
            entry["extraction_status"] = "partial_error" if page_error else "extracted"
            if page_error:
                entry["extraction_error"] = page_error
                errors.append({"document_id": entry["doc_id"], "source_hash": entry["sha256"], "error": page_error})
        except Exception as exc:
            entry["extraction_status"] = "error"; entry["extraction_error"] = f"{type(exc).__name__}:{str(exc)[:240]}"
            if entry["page_count"]:
                entry["page_checks"] = [{"page": p, "status": "extraction_error", "error": entry["extraction_error"]}
                                         for p in range(1, entry["page_count"] + 1)]
            errors.append({"document_id": entry["doc_id"], "source_hash": entry["sha256"], "error": entry["extraction_error"]})
        pdf_rows.append(entry)
    if render:
        renderer = _tool("pdftoppm")
        by_id = {d["doc_id"]: d for d in pdf_rows}
        for cand in candidates:
            doc = by_id[cand["document_id"]]
            image_path = output_dir / cand["page_image"]
            image_path.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run([renderer, "-f", str(cand["page"]), "-l", str(cand["page"]), "-singlefile", "-png", "-r", "120", doc["raw_path"], str(image_path.with_suffix(""))], capture_output=True, timeout=90, check=False)
            if proc.returncode != 0 or not image_path.is_file():
                cand["page_image"] = None; cand["render_error"] = f"pdftoppm_failed:{proc.returncode}"
            else:
                cand["page_image_sha256"] = sha256(image_path.read_bytes())
    # Candidate JSON intentionally contains text evidence but no accepted facts.
    _write_json(output_dir / "capture-integrity.json", integrity)
    _write_json(output_dir / "pdf-page-index.json", {"run_id": RUN_ID, "pdf_count": len(pdf_rows),
        "pdf_pages_indexed": sum(d["page_count"] or 0 for d in pdf_rows), "pdf_page_extraction_errors": errors,
        "documents": pdf_rows})
    listings = [r for r in integrity["observations"] if r["kind"] == "listing" and r["integrity_status"] == "verified"]
    for listing in listings:
        listing["html_timestamp_candidates"] = _html_timestamp_candidates(_object_path(run_root, listing).read_bytes(), listing)
    _write_json(output_dir / "detail-timestamp-candidates.json", {"candidate_status": "unverified_timestamp_cues_not_vintage_proof",
        "details": list(detail_rows.values()), "listings": listings, "pdf_parent_detail_links": parent_links})
    _write_json(output_dir / "page-candidates.json", {"candidate_status": "unreviewed_candidate_not_fact",
        "page_candidate_count": len(candidates), "page_candidates": candidates})
    _write_json(output_dir / "coverage-comparison.json", _coverage_comparison(cp, integrity))
    _write_review_md(output_dir, pdf_rows, candidates, errors)
    return {"observation_count": integrity["observation_count"], "pdf_count": len(pdf_rows),
            "pdf_pages_indexed": sum(d["page_count"] or 0 for d in pdf_rows),
            "empty_text_unsearchable_page_count": sum(sum(p.get("status") == "empty_text_unsearchable" for p in d["page_checks"]) for d in pdf_rows),
            "pdf_page_extraction_errors": len(errors), "page_candidate_count": len(candidates),
            "rendered_page_count": sum(bool(c.get("page_image")) for c in candidates) if render else 0,
            "status": "complete_offline_candidates_unreviewed" if not errors else "complete_with_pdf_errors"}


def _coverage_comparison(cp: dict, integrity: dict) -> dict:
    query_counts = Counter()
    for obs in cp.get("observations", []):
        if obs.get("kind") == "listing" and obs.get("metadata", {}).get("source") == "mafra-title":
            query_counts[obs.get("metadata", {}).get("query") or "unknown"] += 1
    old = _json(PRIOR_PHASE_A / "phase-a-source-composition.json")
    return {"fixed_evaluation_search_queries": ["배추", "무", "수급"],
            "fixed_evaluation_mafra_listing_pages_by_query": dict(query_counts),
            "prior_development_source_capture_count": old.get("capture_and_derived_link_checks", {}).get("captured_responses"),
            "prior_development_scope_note": "Earlier development corpus followed the narrower cabbage-title search rule. This fixed evaluation corpus uses cabbage, radish, and supply title searches; source coverage is broader and the difference must be reported when comparing periods.",
            "comparison_interpretation": "Search-rule coverage is not identical across development and evaluation; do not treat document counts or missing documents as comparable prevalence without adjusting scope."}


def _write_review_md(output_dir: Path, docs: list[dict], candidates: list[dict], errors: list[dict]) -> None:
    lines = ["# Evaluation supply-document visual review queue", "",
             "All spans below are machine-detected review candidates, not facts. Review the full page image and surrounding source material; numbers, crop, source attribution, release/posting times, effective period, and duplicate status require human verification. PDF metadata and displayed dates do not prove historical byte availability.", "",
             f"PDF files: {len(docs)} · indexed pages: {sum(d.get('page_count') or 0 for d in docs)} · empty-text pages requiring visual attention: {sum(sum(p.get('status') == 'empty_text_unsearchable' for p in d.get('page_checks', [])) for d in docs)} · relevant review pages: {len(candidates)} · extraction errors: {len(errors)}", "",
             "Use the exact source hash and physical page number in any annotation. Government attachment hosting does not establish government authorship for every fact; exclude KREI facts individually while preserving other government-attributed facts from the same document.", "",
             "| Document | Source | Page | Candidate reason | Crop cues | Raw number cues | Image |", "|---|---|---:|---|---|---|---|"]
    by_id = {d['doc_id']: d for d in docs}
    for c in candidates:
        d=by_id[c['document_id']]
        nums="; ".join(x['numeric_token'] for x in c['numeric_span_candidates']) or "—"
        image=f"[{c['page_image']}]({c['page_image']})" if c.get('page_image') else "not rendered"
        lines.append(f"| `{c['document_id']}` (`{d['sha256'][:12]}`) | {c['source']} | {c['page']} | {', '.join(c['candidate_reasons'])} | {', '.join(c['crop_candidates']) or '—'} | {nums} | {image} |")
    if errors:
        lines += ["", "## Extraction errors", ""]
        lines += [f"- `{e['document_id']}`: {e['error']}" for e in errors]
    lines += ["", "## Rendering command", "", "Candidate pages are rendered with bundled Poppler `pdftoppm` at 120 dpi. For any additional source page, run:", "", "```sh", f"{_tool('pdftoppm') if shutil.which('pdftoppm') or (POPPLER / 'pdftoppm').is_file() else 'pdftoppm'} -f PAGE -l PAGE -singlefile -png -r 120 /path/to/captured.pdf /path/to/output/page", "```", ""]
    (output_dir / "visual-review.md").write_text("\n".join(lines), encoding="utf-8")


def _manual_quote_valid(evidence: dict, source_records: dict[str, dict]) -> bool:
    sid = evidence.get("source_document_id")
    rec = source_records.get(sid)
    quote = evidence.get("raw_quote")
    if not rec or not quote or evidence.get("source_hash") != rec.get("sha256"):
        return False
    if rec.get("is_pdf") and (not isinstance(evidence.get("source_page"), int) or evidence["source_page"] < 1):
        return False
    return True


def _timestamp_quote_matches(value: str, evidence: dict) -> bool:
    """Compare normalized times to their unmodified source quote conservatively."""
    candidate = str(evidence.get("timestamp_text_candidate") or "")
    quote = str(evidence.get("raw_quote") or "")
    def parts(text: str):
        m = re.search(r"(20\d{2})\D+(\d{1,2})\D+(\d{1,2})(?:\D+(\d{1,2}):(\d{2})(?::(\d{2}))?)?", text)
        return tuple(int(x) if x is not None else None for x in m.groups()) if m else ()
    wanted = parts(value)
    displayed = parts(candidate)
    quoted = parts(quote)
    if len(wanted) < 3 or displayed[:3] != wanted[:3] or quoted[:3] != displayed[:3]:
        return False
    # A supplied clock time must itself be visible in the supporting quote.
    if wanted[3] is not None and (displayed[3:5] != wanted[3:5] or quoted[3:5] != displayed[3:5]):
        return False
    if wanted[5] is not None and any((parts_[5] or 0) != wanted[5] for parts_ in (displayed, quoted)):
        return False
    return True


def _normalized_html_timestamp(candidate: str) -> str | None:
    """Normalize a captured Korean date/time cue to KST, preserving precision."""
    m = re.search(r"(20\d{2})\D+(\d{1,2})\D+(\d{1,2})(?:일)?(?:\D+(\d{1,2}):(\d{2})(?::(\d{2}))?)?", candidate)
    if not m:
        return None
    year, month, day = (int(m.group(i)) for i in (1, 2, 3))
    if m.group(4) is None:
        return f"{year:04d}-{month:02d}-{day:02d}"
    from datetime import datetime, timedelta, timezone
    dt = datetime(year, month, day, int(m.group(4)), int(m.group(5)), int(m.group(6) or 0),
                  tzinfo=timezone(timedelta(hours=9)))
    return dt.isoformat()


def _explicit_krei_fact(raw: dict) -> bool:
    org = str(raw.get("source_org") or "").strip()
    status = str(raw.get("origin_review_status") or "").strip().lower()
    exact_org = bool(re.fullmatch(r"(?:KREI|한국농촌경제연구원|농촌경제연구원|농업관측센터)", org, re.I))
    explicit_status = status in {"source_attribution_krei", "krei_fact_origin", "krei_origin_verified"}
    return exact_org or explicit_status


def normalize_manual_facts(review_path: Path, integrity_path: Path = OUTPUT_ROOT / "capture-integrity.json",
                           candidate_path: Path = OUTPUT_ROOT / "page-candidates.json",
                           page_index_path: Path = OUTPUT_ROOT / "pdf-page-index.json") -> dict:
    """Validate visual-review facts and normalize with the frozen Phase-A rules."""
    from agri_data import period_supply_document_v8 as prior
    integrity = _json(integrity_path)
    candidates = _json(candidate_path).get("page_candidates", [])
    page_index = _json(page_index_path) if page_index_path.is_file() else {"documents": []}
    page_counts = {r["doc_id"]: r.get("page_count") for r in page_index.get("documents", [])}
    docs = {r["doc_id"]: r for r in integrity.get("observations", []) if r.get("is_pdf")}
    # Include captured parent HTML so a posted timestamp quote can cite the
    # actual detail response independently from the current PDF capture.
    for r in integrity.get("observations", []):
        if r.get("kind") in {"detail", "listing"}: docs[r["doc_id"]] = r
    source_records = {}
    for did, row in docs.items():
        source_records[did] = {**row, "final_url": row.get("url"), "sha256": row.get("sha256_actual") or row.get("sha256_expected"),
                               "retrieved_at": row.get("retrieved_at"), "is_pdf": row.get("is_pdf", False)}
    candidate_pages = {(c["document_id"], c["page"]): c for c in candidates}
    raw_doc = _json(review_path)
    if raw_doc.get("cohort_status") not in {"complete_selected_cohort", "complete_reviewed_scope"}:
        raise ValueError("manual_review_scope_not_declared_complete")
    raw_facts = raw_doc.get("facts", [])
    facts, links, krei_rows = [], [], []
    for raw in raw_facts:
        fact_id = raw.get("fact_id")
        if not isinstance(fact_id, str) or not fact_id.strip():
            raise ValueError("manual_review_fact_id_missing")
        did = raw.get("document_id"); source = source_records.get(did)
        if source is None or not source.get("is_pdf"):
            raise ValueError(f"review_fact_unknown_or_nonpdf_document:{raw.get('fact_id')}")
        if raw.get("source_hash") != source["sha256"]:
            raise ValueError(f"review_fact_hash_mismatch:{raw.get('fact_id')}")
        page = raw.get("page")
        if not isinstance(page, int) or page < 1:
            raise ValueError(f"review_fact_invalid_page:{raw.get('fact_id')}")
        if page_counts.get(did) is not None and page > page_counts[did]:
            raise ValueError(f"review_fact_page_out_of_range:{raw.get('fact_id')}")
        if raw.get("evidence_status") != "agent_visual_verified" or not raw.get("evidence_span"):
            raise ValueError(f"review_fact_not_visually_verified:{raw.get('fact_id')}")
        evidence = raw.get("availability_evidence", [])
        for ev in evidence:
            if not _manual_quote_valid(ev, source_records):
                raise ValueError(f"availability_quote_not_hash_associated:{raw.get('fact_id')}")
        page_candidate = candidate_pages.get((did, page))
        raw["candidate_link_status"] = "page_candidate_link" if page_candidate else "manual_fact_not_found_by_candidate_scan"
        raw["candidate_ids"] = page_candidate.get("candidate_id") if page_candidate else None
        # Strict as-of eligibility requires a human attestation plus an extant,
        # hash-matching local archival artifact. A published date is not proof.
        fact_is_krei = _explicit_krei_fact(raw)
        strict = raw.get("version_status") == "strict_historical_asof_verified"
        raw_strict_eligibility = raw.get("asof_eligibility") == "strict_historical_asof"
        if raw_strict_eligibility and not strict and not fact_is_krei:
            raise ValueError(f"strict_asof_without_verified_version_status:{raw.get('fact_id')}")
        if strict and not fact_is_krei:
            ev_path = raw.get("historical_version_evidence")
            ev_sha = raw.get("historical_version_evidence_sha256")
            path = Path(ev_path) if ev_path else None
            if path and not path.is_absolute(): path = ROOT / path
            archive_quote = raw.get("historical_version_evidence_capture_quote") or ""
            archive_time = raw.get("historical_version_evidence_capture_at")
            archive_candidate = raw.get("historical_version_evidence_capture_time_candidate")
            proof_bytes_match = bool(path and path.is_file() and sha256(path.read_bytes()) == ev_sha == source["sha256"])
            proof_timestamp_matches = bool(archive_time and archive_quote and archive_candidate and
                _timestamp_quote_matches(str(archive_time), {"timestamp_text_candidate": archive_candidate, "raw_quote": archive_quote}))
            proof_source_matches = raw.get("historical_version_evidence_source_url") == source["url"]
            if not (raw.get("origin_review_status") == "archival_vintage_verified" and proof_bytes_match
                    and proof_timestamp_matches and proof_source_matches and raw.get("historical_version_evidence_archive_url")):
                raise ValueError(f"strict_vintage_evidence_not_verified:{raw.get('fact_id')}")
        # Only exact attribution of the fact to KREI is excluded. A KREI name
        # elsewhere in a government PDF does not exclude another fact.
        if fact_is_krei:
            raw["asof_eligibility"] = "ineligible"
            raw["exclusion_reason"] = raw.get("exclusion_reason") or "fact_level_krei_origin_excluded"
            krei_rows.append(raw.get("fact_id"))
        # The scenario boundary is derived from separately evidenced dates.
        dated = []
        for field in ("posted_at", "release_at", "distribution_at"):
            value = raw.get(field)
            if not value: continue
            matching = [ev for ev in evidence if ev.get("field") == field and ev.get("timestamp_text_candidate")]
            if not matching:
                raise ValueError(f"timestamp_missing_raw_quote:{raw.get('fact_id')}:{field}")
            if not any(_timestamp_quote_matches(value, ev) for ev in matching):
                raise ValueError(f"timestamp_quote_value_mismatch:{raw.get('fact_id')}:{field}")
            dt = prior._parse_dt(value)
            if dt: dated.append(dt)
        if dated:
            raw["publication_scenario_not_before"] = max(dated).isoformat()
            raw["publication_scenario_basis"] = "latest independently quoted displayed-posting, release, or distribution boundary; not proof of historical byte version"
        else:
            raw["publication_scenario_not_before"] = None
        # Normalize against exact captured bytes. Historical first_seen_at is
        # the actual response/reuse timestamp and remains separate from dates.
        normalized = prior._normalize_review_fact(raw, raw_doc.get("reviewer_kind", "agent_visual"), source)
        if fact_is_krei:
            # A verified archive hash in the frozen normalizer cannot override
            # the fact-level exclusion of KREI-attributed evidence.
            normalized["asof_eligibility"] = "ineligible"
            normalized["exclusion_reason"] = raw.get("exclusion_reason") or "fact_level_krei_origin_excluded"
        normalized["availability_evidence"] = evidence
        normalized["target_mapping_status"] = raw.get("target_mapping_status")
        normalized["manual_value_status"] = raw.get("manual_value_status")
        normalized["candidate_ids"] = raw.get("candidate_ids")
        normalized["candidate_link_status"] = raw.get("candidate_link_status")
        for key in ("source_parent_detail_document_id", "source_parent_detail_hash", "source_parent_posted_quote",
                    "draft_review_notes", "historical_version_evidence_capture_at", "historical_version_evidence_capture_quote",
                    "historical_version_evidence_capture_time_candidate", "historical_version_evidence_archive_url",
                    "historical_version_evidence_source_url"):
            normalized[key] = raw.get(key)
        facts.append(normalized)
        links.append({"fact_id": raw.get("fact_id"), "candidate_id": raw.get("candidate_ids"),
                      "candidate_link_status": raw.get("candidate_link_status"),
                      "source_hash": source["sha256"], "document_id": did, "page": page})
    ids = [x.get("fact_id") for x in facts]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("manual_review_fact_ids_missing_or_duplicated")
    return {"reviewed_facts": facts, "fact_candidate_links": links,
            "fact_counts": {"reviewed_fact_count": len(facts), "strict_historical_fact_count": sum(x["asof_eligibility"] == "strict_historical_asof" for x in facts),
                            "scenario_fact_count": sum(x["asof_eligibility"] == "retrospective_publication_scenario" for x in facts),
                            "ineligible_fact_count": sum(x["asof_eligibility"] == "ineligible" for x in facts),
                            "pending_source_review_fact_count": sum(x["asof_eligibility"] == "pending_source_review" for x in facts),
                            "manual_fact_not_found_by_candidate_scan_count": sum(x["candidate_link_status"] != "page_candidate_link" for x in facts),
                            "explicit_krei_fact_count": len(krei_rows)},
            "explicit_krei_fact_ids": krei_rows, "automatic_candidates_promoted": 0}


def compute_origin_support(reviewed_facts: list[dict], targets: list[dict] | None = None) -> dict:
    """Apply frozen origin map, event grouping, availability and applicability thresholds."""
    from agri_data import period_supply_document_v8 as prior
    if targets is None:
        targets = prior._target_rows()
    support, uses = [], Counter()
    for fact in reviewed_facts:
        t0, t1 = prior._parse_dt(fact.get("effective_start")), prior._parse_dt(fact.get("effective_end"))
        for target in targets:
            if target.get("crop") != fact.get("crop") or target.get("horizon") not in {2, 3}:
                continue
            known = prior._available_at_origin(fact, target.get("origin"))
            if known: uses[fact["fact_id"]] += 1
            if t0 and t1:
                a, b = prior._parse_dt(target.get("targetStart")), prior._parse_dt(target.get("targetEnd"))
                overlap = bool(a and b and t0 <= b and t1 >= a)
                status = "supported" if overlap else "not_overlapping"
            else:
                overlap, status = None, "overlap_unknown"
            archive_at = prior._parse_dt(fact.get("historical_version_evidence_capture_at"))
            origin_at = prior._origin_timestamp(target.get("origin"))
            strict_available = fact.get("asof_eligibility") == "strict_historical_asof" and known and bool(
                archive_at and origin_at and archive_at <= origin_at)
            support.append({"fact_id": fact["fact_id"], "update_group_id": fact.get("update_group_id"),
                "crop": fact.get("crop"), "horizon": target.get("horizon"),
                "target_start": target.get("targetStart"), "target_end": target.get("targetEnd"),
                "origin": target.get("origin"), "input_cutoff": target.get("inputCutoff"),
                "evaluation_period": target.get("evaluation_period"), "target_year": str(target.get("targetStart"))[:4],
                "publish_eligible_at_scenario": fact.get("publish_eligible_at_scenario"),
                "document_id": fact.get("document_id"), "strict_historical_asof": strict_available,
                "scenario_known_past_fact": known, "actual_target_overlap_supported": overlap,
                "target_overlap_status": status, "series_admissibility": fact.get("series_admissibility"),
                "horizon_relevance": fact.get("horizon_relevance"), "is_new_update_at_scenario_origin": False})
    prior._mark_first_update_origins(support)
    return _summarize_origin_support(support, uses)


def _summarize_origin_support(support: list[dict], uses: Counter | None = None) -> dict:
    """Rebuild cell summaries from the frozen Phase-A rows plus eval facts."""
    by_group = defaultdict(list)
    for row in support:
        by_group[(row["evaluation_period"], row["target_year"], row["crop"], row["horizon"])].append(row)
    releases = []
    for key, rows in sorted(by_group.items()):
        known = [r for r in rows if r["scenario_known_past_fact"]]
        events = [r for r in rows if r["is_new_update_at_scenario_origin"]]
        event_origins = {r["origin"] for r in events}
        event_groups = {r["update_group_id"] for r in events if r["update_group_id"]}
        documented = {r["origin"] for r in known if r["target_overlap_status"] == "supported"}
        documented_events = {r["origin"] for r in events if r["target_overlap_status"] == "supported"}
        releases.append({"evaluation_period": key[0], "target_year": key[1], "crop": key[2], "horizon": key[3],
            "planned_origin_count": len({r["origin"] for r in rows}),
            "scenario_known_fact_origin_count": len({r["origin"] for r in known}),
            "strict_asof_fact_origin_count": len({r["origin"] for r in rows if r["strict_historical_asof"]}),
            "first_new_update_count": len(event_groups), "first_new_update_origin_count": len(event_origins),
            "first_new_update_month_count": len({str(o)[:7] for o in event_origins}),
            "first_new_update_origins": sorted(event_origins),
            "documented_effective_interval_fact_origin_count": len(documented),
            "known_effective_overlap_non_event_origin_count": len(documented - documented_events),
            "actual_target_overlap_fact_origin_count": len({r["origin"] for r in known if r["actual_target_overlap_supported"] is True}),
            "overlap_unknown_fact_origin_count": len({r["origin"] for r in known if r["target_overlap_status"] == "overlap_unknown"}),
            "comparison_target_count": len({(r["target_start"], r["target_end"]) for r in rows})})
    return {"support_rows": support, "release_origin_support": releases,
            "scenario_available_origin_count_by_fact": dict(uses or {})}


def evaluate_gates(reviewed_facts: list[dict], support: dict, *, integrity_pass: bool,
                   review_scope_complete: bool, evaluation_documents_same_scope: bool,
                   target_mapping: dict | None = None, evaluation_review_scope_complete: bool = False) -> dict:
    """Frozen protocol gates G1–G6; no threshold is relaxed based on observed data."""
    release = support["release_origin_support"]
    crops = ("cabbage", "radish")
    development = []
    for crop in crops:
        for year in ("2022", "2023"):
            for horizon in (2, 3):
                rows = [r for r in release if r["evaluation_period"] == "development_2022_2023" and r["target_year"] == year and r["crop"] == crop and r["horizon"] == horizon]
                # There can be a missing crop/horizon/year cell; missing is zero.
                x = rows[0] if rows else {}
                events = x.get("first_new_update_count", 0)
                event_origins = x.get("first_new_update_origin_count", 0)
                months = x.get("first_new_update_month_count", 0)
                applicable = x.get("documented_effective_interval_fact_origin_count", 0)
                non_events = x.get("known_effective_overlap_non_event_origin_count", 0)
                development.append({"crop": crop, "target_year": year, "horizon": horizon,
                    "unique_new_updates": events, "distinct_event_origins": event_origins, "event_months": months,
                    "applicable_known_origins": applicable, "applicable_non_event_origins": non_events,
                    "g3_pass": events >= 12 and event_origins >= 12 and months >= 6,
                    "g6_pass": applicable >= 12 and non_events >= 12})
    evaluation = []
    for crop in crops:
        for horizon in (2, 3):
            rows = [r for r in release if r["evaluation_period"] == "evaluation_2024_2026-09" and r["crop"] == crop and r["horizon"] == horizon]
            event_origins = {o for r in rows for o in r.get("first_new_update_origins", [])}
            years = {r["target_year"] for r in rows if r.get("first_new_update_origin_count", 0)}
            targets = len({(r.get("target_start"), r.get("target_end")) for r in support["support_rows"]
                           if r.get("evaluation_period") == "evaluation_2024_2026-09" and r.get("crop") == crop and r.get("horizon") == horizon})
            evaluation.append({"crop": crop, "horizon": horizon, "first_new_update_origin_count": len(event_origins),
                "support_year_count": len(years), "comparison_target_count": targets,
                "g4_pass": len(event_origins) >= 24 and len(years) >= 2 and targets >= 36})
    fact_link_ok = all(f.get("evidence_status") == "agent_visual_verified" and f.get("source_hash") and f.get("page") and f.get("evidence_span") and f.get("crop") in crops and f.get("source_org") and f.get("origin_review_status") for f in reviewed_facts)
    g1 = integrity_pass and review_scope_complete and bool(reviewed_facts) and fact_link_ok
    selected = [f for f in reviewed_facts if f.get("asof_eligibility") in {"strict_historical_asof", "retrospective_publication_scenario"}]
    strict_support_ids = {r.get("fact_id") for r in support.get("support_rows", []) if r.get("strict_historical_asof")}
    g2 = bool(selected) and all(f.get("asof_eligibility") == "strict_historical_asof" and f.get("fact_id") in strict_support_ids for f in selected)
    g3 = bool(development) and all(x["g3_pass"] for x in development) and all(x["g6_pass"] for x in development)
    g4 = evaluation_documents_same_scope and evaluation_review_scope_complete and bool(evaluation) and all(x["g4_pass"] for x in evaluation)
    mapping = target_mapping or {}
    def mapping_is_documented(value):
        values = value if isinstance(value, list) else [value]
        return bool(values) and all(v in {"possible", "unknown", "mismatch"} for v in values)
    g5 = all(mapping_is_documented(mapping.get(crop)) for crop in crops)
    return {"protocol": "phase-A/protocol.md", "frozen_thresholds": {"g3_events_per_crop_year": 12, "g3_distinct_origins": 12, "g3_months": 6, "g6_applicable_origins_per_crop_horizon_year": 12, "g6_non_event_origins": 12, "g4_eval_new_origins": 24, "g4_years": 2, "g4_comparison_targets": 36},
            "G1_integrity_and_review": {"pass": g1, "integrity_pass": integrity_pass,
                "selected_review_scope_complete": review_scope_complete,
                "whole_corpus_manual_review_complete": False, "facts_complete": fact_link_ok},
            "G2_historical_asof": {"pass": g2, "strict_fact_count": sum(f.get("asof_eligibility") == "strict_historical_asof" for f in selected), "selected_fact_count": len(selected), "reviewed_fact_count": len(reviewed_facts)},
            "G3_development_events_and_G6_applicability": {"pass": g3, "cells": development},
            "G4_evaluation_support": {"pass": g4, "same_search_scope": evaluation_documents_same_scope,
                "selected_evaluation_scope_complete": evaluation_review_scope_complete,
                "partial_cohort_counts_are_lower_bound_only": not evaluation_review_scope_complete, "cells": evaluation},
            "G5_target_mapping": {"pass": g5, "crop_mapping": mapping},
            "all_pre_fit_gates_pass": all((g1, g2, g3, g4, g5)), "fit_authorized_by_gates": all((g1, g2, g3, g4, g5))}


def _fact_coverage(reviewed: list[dict]) -> dict:
    groups = defaultdict(list)
    for fact in reviewed:
        date_text = fact.get("document_date") or fact.get("posted_at") or ""
        key = (fact.get("crop") or "unknown", fact.get("variety_or_crop_type") or "unknown",
               fact.get("region") or "unknown", str(date_text)[:4] or "unknown",
               fact.get("quantity_kind") or "unknown", fact.get("source_org") or "unknown")
        groups[key].append(fact)
    rows = []
    for key, facts in sorted(groups.items()):
        rows.append({"crop": key[0], "variety_or_crop_type": key[1], "region": key[2], "document_year": key[3],
            "quantity_kind": key[4], "source_org": key[5], "reviewed_fact_count": len(facts),
            "unique_update_count": len({f.get("update_group_id") for f in facts if f.get("update_group_id")}),
            "unique_document_count": len({f.get("document_id") for f in facts}),
            "strict_historical_fact_count": sum(f.get("asof_eligibility") == "strict_historical_asof" for f in facts),
            "publication_scenario_fact_count": sum(f.get("asof_eligibility") == "retrospective_publication_scenario" for f in facts),
            "prospective_only_fact_count": sum(f.get("asof_eligibility") == "prospective_only" for f in facts),
            "ineligible_fact_count": sum(f.get("asof_eligibility") == "ineligible" for f in facts),
            "pending_source_review_fact_count": sum(f.get("asof_eligibility") == "pending_source_review" for f in facts),
            "effective_interval_known_fact_count": sum(bool(f.get("effective_start") and f.get("effective_end")) for f in facts),
            "effective_interval_unknown_fact_count": sum(not (f.get("effective_start") and f.get("effective_end")) for f in facts),
            "target_mapping_status_counts": dict(Counter(f.get("target_mapping_status") or "not_reviewed" for f in facts))})
    return {"reviewed_facts_are_not_independent_events": True,
            "coverage_by_crop_type_region_year_quantity_source": rows}


def adapt_manual_draft(draft_path: Path, output_path: Path, *,
                       integrity_path: Path = OUTPUT_ROOT / "capture-integrity.json",
                       run_root: Path = RUN_ROOT) -> dict:
    """Mechanically bind a human-authored MAFRA draft to captured IDs and quotes.

    This adapter copies no fact not present in the draft. It only supplies the
    exact captured PDF/detail identity and quote-hash references required by
    the validator, while retaining contradictory date/time strings verbatim.
    """
    draft = _json(draft_path)
    integrity = _json(integrity_path)
    verified_pdfs = {row["url"]: row for row in integrity.get("observations", [])
                     if row.get("is_pdf") and row.get("integrity_status") == "verified"}
    details_by_hash = defaultdict(list)
    for row in integrity.get("observations", []):
        if row.get("kind") == "detail" and row.get("integrity_status") == "verified":
            digest = row.get("sha256_actual") or row.get("sha256_expected")
            if digest: details_by_hash[digest].append(row)
    candidate_doc = {row.get("doc_id"): row for row in integrity.get("observations", [])
                     if row.get("doc_id")}
    facts = []
    for doc in draft.get("documents", []):
        pdf = verified_pdfs.get(doc.get("url"))
        if not pdf:
            raise ValueError(f"manual_draft_pdf_not_in_verified_capture:{doc.get('url')}")
        digest = pdf.get("sha256_actual") or pdf.get("sha256_expected")
        matches = details_by_hash.get(pdf.get("metadata", {}).get("detail_sha256"), [])
        if len(matches) != 1:
            raise ValueError(f"manual_draft_parent_detail_not_unique:{pdf.get('doc_id')}:{len(matches)}")
        detail = matches[0]
        detail_id = detail["doc_id"]
        if not detail_id or detail_id not in candidate_doc:
            raise ValueError(f"manual_draft_parent_detail_id_missing:{pdf.get('doc_id')}")
        html_dates = _html_timestamp_candidates(_object_path(run_root, detail).read_bytes(), detail)
        posted_ev = next((ev for ev in html_dates if _timestamp_quote_matches(
            doc.get("date", ""), {"timestamp_text_candidate": ev["timestamp_text_candidate"], "raw_quote": ev["raw_quote"]})), None)
        posted_at = _normalized_html_timestamp(posted_ev["timestamp_text_candidate"]) if posted_ev else None
        release_at = doc.get("release_at")
        release_quote = doc.get("release_quote") or ""
        distribution_at = doc.get("distribution_at")
        distribution_quote = doc.get("distribution_quote") or ""
        # If a date-only distribution note accompanies a documented release
        # hour, retain the quote as a conflict cue but do not synthesize a time.
        if distribution_at and not re.search(r"\d{1,2}:\d{2}", distribution_quote):
            distribution_at = None
        # Release and distribution are separate event times. Reuse an explicit
        # distribution clock for release only where the source says `배포시`.
        release_uses_distribution_cue = bool(re.search(r"배포\s*시", release_quote) and
            re.search(r"\d{1,2}:\d{2}", distribution_quote))
        time_quote = distribution_quote if release_uses_distribution_cue else release_quote
        time_candidate = _DATE_RE.search(time_quote)
        time_candidate_text = time_candidate.group(0) if time_candidate else ""
        if time_candidate_text and not re.search(r"\d{1,2}:\d{2}", time_candidate_text):
            clock = re.search(r"\d{1,2}:\d{2}", time_quote)
            if clock: time_candidate_text = time_candidate_text.rstrip() + " " + clock.group(0)
        availability = []
        if posted_ev:
            availability.append({"field": "posted_at", "source_document_id": detail_id,
                "source_hash": detail.get("sha256_actual") or detail.get("sha256_expected"), "source_page": None,
                "timestamp_text_candidate": posted_ev["timestamp_text_candidate"], "raw_quote": posted_ev["raw_quote"]})
        if release_at and time_candidate_text:
            availability.append({"field": "release_at", "source_document_id": pdf["doc_id"],
                "source_hash": digest, "source_page": 1,
                "timestamp_text_candidate": time_candidate_text, "raw_quote": time_quote})
        if release_quote:
            availability.append({"field": "release_condition", "source_document_id": pdf["doc_id"],
                "source_hash": digest, "source_page": 1,
                "timestamp_text_candidate": release_quote, "raw_quote": release_quote})
        if distribution_at:
            availability.append({"field": "distribution_at", "source_document_id": pdf["doc_id"],
                "source_hash": digest, "source_page": 1,
                "timestamp_text_candidate": distribution_quote, "raw_quote": distribution_quote})
        for item in doc.get("facts", []):
            if len(item) not in {10, 12}:
                raise ValueError(f"manual_draft_fact_shape_unrecognized:{item[0] if item else 'unknown'}")
            fact_id, page, crop, value, unit, quantity_kind, span, eligibility, note, source_org = item[:10]
            lower, upper = (item[10], item[11]) if len(item) == 12 else (None, None)
            fact = {"fact_id": str(fact_id), "update_group_id": f"mafra-{pdf['doc_id']}",
                "document_id": pdf["doc_id"], "source_hash": digest, "page": page, "evidence_span": span,
                "numeric_token": str(value) if value is not None else None, "crop": crop, "value": value,
                "value_lower": lower, "value_upper": upper, "unit": unit, "quantity_kind": quantity_kind,
                "evidence_status": "agent_visual_verified", "reviewer_kind": "agent_visual",
                "document_date": doc.get("date"), "posted_at": posted_at,
                "release_at": release_at, "distribution_at": distribution_at,
                "availability_evidence": availability, "version_status": "current_capture_historical_bytes_unproven",
                "asof_eligibility": eligibility, "origin_review_status":
                    "source_attribution_krei" if str(source_org or "").strip().upper() == "KREI" else
                    "source_attribution_pending" if not source_org else "source_attribution_reviewed",
                "source_org": source_org, "provenance_origin": note, "exclusion_reason": note if eligibility == "ineligible" else None,
                "region": None, "target_mapping_status": "unknown", "manual_value_status": "verified_from_visual_review",
                "effective_start": None, "effective_end": None, "effective_period_text": None,
                "source_parent_detail_document_id": detail_id,
                "source_parent_detail_hash": detail.get("sha256_actual") or detail.get("sha256_expected"),
                "source_parent_posted_quote": posted_ev["raw_quote"] if posted_ev else None,
                "draft_review_notes": doc.get("notes")}
            # Keep both the condition and explicit hour when the source says
            # `배포시`; the adapter does not manufacture an hour from the cue.
            facts.append(fact)
    output = {"reviewer_kind": draft.get("reviewer_kind", "agent_visual"),
        "cohort_status": "complete_selected_cohort", "scope": draft.get("scope"),
        "whole_corpus_review_complete": False, "facts": facts}
    _write_json(output_path, output)
    return {"adapted_fact_count": len(facts), "verified_pdf_count": len({f["document_id"] for f in facts}),
            "whole_corpus_review_complete": False, "output": str(output_path)}


def _load_phase_a_support() -> tuple[list[dict], dict]:
    """Read the frozen 93-fact Phase-A origin rows without rewriting them."""
    support_path = PRIOR_PHASE_A / "fact-origin-support.csv"
    rows = list(csv.DictReader(support_path.open(encoding="utf-8-sig", newline="")))
    bool_fields = {"strict_historical_asof", "scenario_known_past_fact", "actual_target_overlap_supported",
                   "is_new_update_at_scenario_origin"}
    int_fields = {"horizon"}
    for row in rows:
        for field in bool_fields:
            value = row.get(field)
            row[field] = True if value == "True" else False if value == "False" else None
        for field in int_fields:
            row[field] = int(row[field]) if row.get(field) else None
    # The published support file is the frozen source of the previous result;
    # include its hash in the new audit for traceability.
    return rows, {"path": str(support_path.relative_to(ROOT)), "sha256": sha256(support_path.read_bytes()),
                  "frozen_prior_support_rows": len(rows)}


def integrate_manual_review(review_path: Path, *, output_dir: Path = OUTPUT_ROOT,
                            integrity_path: Path | None = None, candidate_path: Path | None = None,
                            page_index_path: Path | None = None, integrity_pass: bool | None = None,
                            evaluation_documents_same_scope: bool = False) -> dict:
    """Join reviewed facts to captured hashes and the frozen Phase-A calendar."""
    ip = integrity_path or output_dir / "capture-integrity.json"
    cp = candidate_path or output_dir / "page-candidates.json"
    pp = page_index_path or output_dir / "pdf-page-index.json"
    review = _json(review_path)
    normalized = normalize_manual_facts(review_path, ip, cp, pp)
    new_support = compute_origin_support(normalized["reviewed_facts"])
    from agri_data import period_supply_document_v8 as prior
    prior_support_rows, prior_support_provenance = _load_phase_a_support()
    combined_support_rows = [*prior_support_rows, *new_support["support_rows"]]
    prior._mark_first_update_origins(combined_support_rows)
    support = _summarize_origin_support(combined_support_rows)
    integrity = _json(ip)
    review_complete = review.get("cohort_status") in {"complete_selected_cohort", "complete_reviewed_scope"}
    mapping = {}
    for crop in ("cabbage", "radish"):
        statuses = {f.get("target_mapping_status") for f in normalized["reviewed_facts"] if f.get("crop") == crop}
        statuses.discard(None)
        mapping[crop] = sorted(statuses) if statuses else []
    gates = evaluate_gates(normalized["reviewed_facts"], support,
        integrity_pass=integrity.get("integrity_pass", False) if integrity_pass is None else integrity_pass,
        review_scope_complete=review_complete, evaluation_documents_same_scope=evaluation_documents_same_scope,
        evaluation_review_scope_complete=review.get("whole_corpus_review_complete") is True,
        target_mapping=mapping)
    coverage = _fact_coverage(normalized["reviewed_facts"])
    source_scope = _coverage_comparison(_json(RUN_ROOT / "checkpoint.json"), integrity)
    result = {"reviewed_facts": normalized["reviewed_facts"], "fact_candidate_links": normalized["fact_candidate_links"],
        "fact_counts": normalized["fact_counts"], "coverage": coverage, "support": support, "gates": gates,
        "source_scope_comparison": source_scope, "frozen_phase_a_support": prior_support_provenance,
        "manual_review_scope": review.get("scope") or review.get("review_scope"),
        "whole_corpus_manual_review_complete": review.get("whole_corpus_review_complete") is True,
        "automatic_candidates_promoted": 0}
    _write_json(output_dir / "reviewed-facts.json", result["reviewed_facts"])
    _write_json(output_dir / "fact-origin-support.json", support["support_rows"])
    _write_json(output_dir / "release-origin-support.json", support["release_origin_support"])
    _write_json(output_dir / "fact-candidate-links.json", result["fact_candidate_links"])
    _write_json(output_dir / "fact-coverage.json", coverage)
    _write_json(output_dir / "gate-results.json", gates)
    _write_json(output_dir / "manual-review-summary.json", {"fact_counts": result["fact_counts"],
        "source_scope_comparison": source_scope, "automatic_candidates_promoted": 0})
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("audit", help="audit the fixed capture and build PDF page visual-review queue")
    p.add_argument("--run-root", type=Path, default=RUN_ROOT)
    p.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    p.add_argument("--no-render", action="store_true")
    r = sub.add_parser("normalize-review", help="validate human review JSON; does not fit a model")
    r.add_argument("review_json", type=Path)
    r.add_argument("--output", type=Path, default=OUTPUT_ROOT / "reviewed-facts.json")
    r.add_argument("--integrate", action="store_true", help="also calculate frozen origin support and G1-G6")
    a = sub.add_parser("adapt-review", help="bind manually reviewed records to captured source IDs and quotes")
    a.add_argument("draft", type=Path)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--integrity", type=Path, default=OUTPUT_ROOT / "capture-integrity.json")
    a.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args(argv)
    if args.command == "audit":
        result = build_review_queue(args.run_root, args.output, render=not args.no_render)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    elif args.command == "adapt-review":
        result = adapt_manual_draft(args.draft, args.output, integrity_path=args.integrity, run_root=args.run_root)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    else:
        if args.integrate:
            result = integrate_manual_review(args.review_json, output_dir=args.output.parent)
            print(json.dumps({"status": "review_json_integrated", "all_pre_fit_gates_pass": result["gates"]["all_pre_fit_gates_pass"], "output": str(args.output.parent)}, ensure_ascii=False))
        else:
            _write_json(args.output, normalize_manual_facts(args.review_json))
            print(json.dumps({"status": "review_json_validated", "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
