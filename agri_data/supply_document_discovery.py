"""Offline link discovery from already captured public listing-page HTML."""
from __future__ import annotations

import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import urlparse
from urllib.parse import urljoin

from agri_data.supply_document_collection import sha256, validate_url, write_immutable


class _Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.link_stack = []
        self.row_links = []
        self.row_text = []
        self.in_row = False

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in {"tr", "li"}:
            self.in_row = True
            self.row_links = []
            self.row_text = []
        elif tag == "a":
            link = {"attrs": dict(attrs), "text": []}
            self.link_stack.append(link)
            if self.in_row:
                self.row_links.append(link)

    def handle_data(self, data):
        for link in self.link_stack:
            link["text"].append(data)
        if self.in_row:
            self.row_text.append(data)

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag == "a" and self.link_stack:
            link = self.link_stack.pop()
            link["text"] = " ".join("".join(link["text"]).split())
            if not self.in_row:
                link["row_context"] = link["text"]
                self.links.append(link)
        elif tag in {"tr", "li"}:
            row_context = " ".join("".join(self.row_text).split())
            for link in self.row_links:
                link["row_context"] = row_context
                self.links.append(link)
            self.row_links = []
            self.row_text = []
            self.in_row = False


def discover(checkpoint_path: Path, output_path: Path) -> dict:
    checkpoint_path = checkpoint_path.resolve()
    root = checkpoint_path.parent
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    candidates = []
    chain_candidates = {}
    chain_expected = {}
    for request_id, record in checkpoint.get("requests", {}).items():
        if record.get("status") != "captured" or record.get("request", {}).get("kind") not in {"listing", "reference"}:
            continue
        raw = (root / record["raw_file"]).read_bytes()
        source_digest = sha256(raw)
        if source_digest != record.get("sha256"):
            raise ValueError("raw_hash_mismatch")
        parser = _Links()
        parser.feed(raw.decode("utf-8", errors="replace"))
        for link in parser.links:
            attrs = link["attrs"]
            onclick = attrs.get("onclick", "")
            # Jeju Synap links store the PDF directory and filename in onclick.
            raw_target = None
            if "fn_callSynapViewer" in onclick:
                args = re.findall(r"['\"]([^'\"]*)['\"]", onclick)
                if len(args) >= 2:
                    raw_target = args[0].rstrip("/") + "/" + args[1]
            if not raw_target:
                href = attrs.get("href")
                raw_target = href if href and href.strip() not in {"#", "javascript:;"} else None
            if not raw_target:
                continue
            absolute = urljoin(record["url"], raw_target)
            try:
                validate_url(absolute)
            except ValueError:
                continue
            path = urlparse(absolute).path.lower()
            if "fn_callSynapViewer" in onclick or path.endswith(".pdf"):
                candidate_kind = "document_original"
            elif "/artclview.do" in path:
                candidate_kind = "detail_page"
            elif path.endswith("/download.do"):
                candidate_kind = "document_original"
            else:
                continue
            text = link["text"]
            candidate_id = f"{request_id}-link-{len(candidates)+1:04d}"
            chain_id = re.sub(r"-p\d+$", "", request_id)
            chain_candidates[chain_id] = chain_candidates.get(chain_id, 0) + 1
            expected = record.get("request", {}).get("expected_result_count")
            if expected is not None:
                chain_expected[chain_id] = expected
            candidates.append({"candidate_id": candidate_id, "source_request_id": request_id,
                               "source_page_sha256": source_digest, "url": absolute,
                               "link_text": text, "row_context": link.get("row_context") or text,
                               "candidate_kind": candidate_kind,
                               "onclick": onclick or None,
                               "disposition": "candidate_requires_review"})
    chain_checks = [{"chain_id": chain_id, "expected_result_count": chain_expected.get(chain_id),
                     "candidate_link_count": count,
                     "count_matches_expected": count == chain_expected.get(chain_id)}
                    for chain_id, count in sorted(chain_candidates.items())]
    result = {"protocol": "offline-supply-link-discovery-v1", "source_checkpoint": str(checkpoint_path),
              "source_manifest_sha256": checkpoint.get("manifestSha256"), "candidate_count": len(candidates),
              "chain_checks": chain_checks, "candidates": candidates,
              "limitations": ["derived only from captured HTML; no network requests", "link targets are candidates and are not asserted to be relevant originals"]}
    write_immutable(output_path, (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
    return result


def build_document_manifest(candidate_path: Path, manifest_path: Path) -> dict:
    candidates_doc = json.loads(candidate_path.read_text(encoding="utf-8"))
    if candidates_doc.get("protocol") != "offline-supply-link-discovery-v1":
        raise ValueError("candidate_protocol_mismatch")
    checks = candidates_doc.get("chain_checks", [])
    if not checks or any(not check.get("count_matches_expected") for check in checks):
        raise ValueError("listing_candidate_counts_do_not_match")
    rows = candidates_doc.get("candidates", [])
    if len(rows) != candidates_doc.get("candidate_count"):
        raise ValueError("candidate_count_mismatch")
    checkpoint_path = Path(candidates_doc["source_checkpoint"])
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    source_by_id = checkpoint.get("requests", {})
    requests = []
    for row in rows:
        source_id = row.get("source_request_id")
        source_record = source_by_id.get(source_id)
        if not source_record or source_record.get("status") != "captured":
            raise ValueError("candidate_source_not_captured")
        if source_record.get("sha256") != row.get("source_page_sha256"):
            raise ValueError("candidate_source_hash_mismatch")
        kind = "document" if row.get("candidate_kind") == "document_original" else "reference"
        requests.append({"id": row["candidate_id"], "url": row["url"], "kind": kind,
                         "source": source_record.get("request", {}).get("source"),
                         "topic": source_record.get("request", {}).get("topic"),
                         "source_request_id": source_id,
                         "source_page_sha256": row["source_page_sha256"],
                         "candidate_kind": row["candidate_kind"],
                         "link_text": row.get("link_text"),
                         "row_context": row.get("row_context"),
                         "source_page_url": source_record.get("url"),
                         "source_list_page_index": source_record.get("request", {}).get("page_index"),
                         "document_year": source_record.get("request", {}).get("year")})
    result = {"protocol": "public-supply-document-capture-v1",
              "scope_id": "supply-documents-from-2022-2023-query-results",
              "scope_description": "Document/detail targets discovered offline from the 119 rows in the three reviewed, UI-enumerated source query chains.",
              "scope_enumeration_complete": True,
              "scope_limitations": [
                  "Coverage is limited to the exact MAFRA cabbage-title and Jeju weekly bulletin queries in the source listing manifest.",
                  "All candidate URLs are derived from captured listing HTML; row title/date context and listing-page SHA-256 are retained per request.",
                  "The 18 MAFRA targets are detail pages; attachment URLs require a later offline pass over those captured detail pages.",
                  "Content has not yet been checked for KREI-derived values, crop specificity, or first historical publication time.",
                  "No record in this manifest is a model input or evidence of forecast benefit."],
              "candidate_source_checkpoint": str(checkpoint_path),
              "candidate_source_manifest_sha256": checkpoint.get("manifestSha256"),
              "candidate_list_sha256": sha256(candidate_path.read_bytes()),
              "listing_chain_checks": checks,
              "expected_request_count": len(requests), "requests": requests}
    encoded = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    write_immutable(manifest_path, encoded)
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest-output", type=Path)
    args = p.parse_args(argv)
    result = discover(args.checkpoint, args.output)
    manifest_result = build_document_manifest(args.output, args.manifest_output) if args.manifest_output else None
    print(json.dumps({"output": str(args.output), "candidates": result["candidate_count"],
                      "manifest_output": str(args.manifest_output) if manifest_result else None,
                      "manifest_requests": manifest_result["expected_request_count"] if manifest_result else None}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
