"""Offline parser and validator for KAMIS public wholesale Excel exports (HTML)."""
from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata


class _TableParser(HTMLParser):
    def __init__(self, table_id="data"):
        super().__init__(convert_charrefs=True)
        self.target_id = table_id
        self.in_data = False
        self.depth = 0
        self.capture = None
        self.cell = None
        self.row = None
        self.caption = ""
        self.headers = []
        self.rows = []
        self.data_tables = 0
        self.closed_data_tables = 0
        self.structural_errors = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "table":
            if attrs.get("id") == self.target_id: self.data_tables += 1
            if not self.in_data and attrs.get("id") == self.target_id:
                self.in_data, self.depth = True, 1
            elif self.in_data:
                self.structural_errors.append("nested table inside table id=data")
                self.depth += 1
        elif self.in_data and self.depth == 1:
            if tag == "caption": self.capture = "caption"
            elif tag == "tr":
                if self.row is not None or self.cell is not None:
                    self.structural_errors.append("overlapping or unclosed table row")
                else:
                    self.row = []
            elif tag in ("th", "td"):
                if self.row is None or self.cell is not None:
                    self.structural_errors.append("cell outside row or overlapping unclosed cell")
                else:
                    self.cell = {"tag": tag, "text": ""}
        elif self.in_data and self.cell is not None and tag == "br":
            self.cell["text"] += " "

    def handle_endtag(self, tag):
        if not self.in_data: return
        if self.depth == 1:
            if tag in ("th", "td"):
                if self.cell is None or self.row is None or self.cell["tag"] != tag:
                    self.structural_errors.append("closing cell without open cell")
                else:
                    self.cell["text"] = self.cell["text"].strip()
                    self.row.append(self.cell)
                    self.cell = None
            elif tag == "tr":
                if self.row is None or self.cell is not None:
                    self.structural_errors.append("closing row with no open row or unclosed cell")
                else:
                    if self.row:
                        if all(c["tag"] == "th" for c in self.row): self.headers.append([c["text"] for c in self.row])
                        else: self.rows.append([c["text"] for c in self.row])
                    self.row = None
            elif tag == "caption": self.capture = None
        if tag == "table":
            self.depth -= 1
            if self.depth == 0:
                self.closed_data_tables += 1
                self.in_data = False

    def handle_data(self, data):
        if self.capture == "caption": self.caption += data
        if self.cell is not None: self.cell["text"] += data


def _clean(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def _norm(s):
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s or ""))


def _price(raw):
    raw = raw.strip()
    flags = []
    if raw in ("", "-"):
        return {"raw": raw, "decimal": None, "status": "missing", "flags": flags}
    try:
        value = Decimal(raw.replace(",", ""))
    except InvalidOperation:
        return {"raw": raw, "decimal": None, "status": "invalid", "flags": ["invalid_price"]}
    if not value.is_finite(): flags.append("nonfinite_price")
    elif value < 0: flags.append("negative_price")
    return {"raw": raw, "decimal": str(value), "status": "observed", "flags": flags}


def _date(raw, requested):
    s = re.sub(r"\s+", "", raw)
    full = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", s)
    short = re.fullmatch(r"(\d{2})-(\d{2})-(\d{2})", s)
    if full:
        try: return dt.date(*map(int, full.groups()))
        except ValueError as e: raise ValueError(f"invalid date cell {raw!r}") from e
    if short:
        yy, mm, dd = map(int, short.groups())
        candidates = [d for d in requested if d.year % 100 == yy and d.month == mm and d.day == dd]
        if len(candidates) != 1: raise ValueError(f"YY-MM-DD date {raw!r} is ambiguous or outside requested dates")
        return candidates[0]
    raise ValueError(f"unrecognized date cell {raw!r}")


def parse_export(path: Path | bytes, *, start: dt.date, end: dt.date, item: str,
                 variety: str, grade: str, unit: str, table_id: str = "data") -> dict:
    """Parse and validate an export against explicit expected caption and dates."""
    if start > end: raise ValueError("start must be on or before end")
    raw_bytes = path if isinstance(path, bytes) else path.read_bytes()
    source = raw_bytes.decode("utf-8-sig")
    parser = _TableParser(table_id=table_id); parser.feed(source); parser.close()
    if parser.data_tables != 1 or parser.closed_data_tables != 1 or parser.in_data or parser.row is not None or parser.cell is not None or parser.structural_errors:
        raise ValueError(f"expected exactly one complete table id={table_id} with closed rows")
    if not parser.headers or not parser.caption: raise ValueError(f"missing caption or header in table id={table_id}")
    parts = [_norm(part) for part in parser.caption.split(",")]
    if len(parts) != 5 or "중도매인판매가격" not in parts[0] or "채소류" not in parts[0] or "기간별/일간가격정보" not in parts[4]:
        raise ValueError("caption is not a KAMIS daily wholesale product-family caption")
    expected_parts = (("item", item, parts[1]), ("variety", variety, parts[2]), ("grade", grade, parts[3]))
    for label, expected, actual in expected_parts:
        if _norm(expected) != actual: raise ValueError(f"caption {label} mismatch: expected {expected!r}, got {actual!r}")
    final = parts[4]
    expected_unit = _norm(unit)
    if not final.startswith(expected_unit) or not final[len(expected_unit):].startswith("기간별/일간가격정보"):
        raise ValueError(f"caption unit mismatch: expected {unit!r}")
    headers = parser.headers[-1]
    if len({_norm(h) for h in headers}) != len(headers): raise ValueError("duplicate header labels")
    expected_headers = ["구분", "평균", "등락율", "서울", "부산", "대구", "광주", "대전"]
    if [_norm(h) for h in headers] != expected_headers:
        raise ValueError(f"KAMIS national pilot header schema mismatch: {headers!r}")
    def one(label):
        matches = [i for i, h in enumerate(headers) if h == label]
        if len(matches) != 1: raise ValueError(f"expected exactly one {label!r} column; found {len(matches)}")
        return matches[0]
    date_i, price_i = one("구분"), one("평균")
    price_indices = [i for i, h in enumerate(headers) if i != date_i and _norm(h) != "등락율"]
    requested = [start + dt.timedelta(days=n) for n in range((end-start).days+1)]
    dates = set(requested); seen = set(); out = []
    for idx, cells in enumerate(parser.rows, 1):
        if len(cells) != len(headers): raise ValueError(f"row {idx} has {len(cells)} cells; expected {len(headers)}")
        day = _date(cells[date_i], dates)
        if day not in dates: raise ValueError(f"date {day.isoformat()} outside requested range")
        if day in seen: raise ValueError(f"duplicate date {day.isoformat()}")
        seen.add(day)
        parsed_prices = {headers[i]: _price(cells[i]) for i in price_indices}
        mean = parsed_prices[headers[price_i]]
        out.append({"date": day.isoformat(), "rawCells": cells, "prices": parsed_prices,
                    "priceRaw": mean["raw"], "priceDecimal": mean["decimal"],
                    "priceStatus": mean["status"], "flags": mean["flags"]})
    missing = [d.isoformat() for d in requested if d not in seen]
    return {"source": "KAMIS public wholesale price export", "sourceSha256": hashlib.sha256(raw_bytes).hexdigest(),
            "format": "UTF-8 HTML table delivered with .xls extension", "captionRaw": _clean(parser.caption),
            "requested": {"item": item, "variety": variety, "grade": grade, "unit": unit,
                          "start": start.isoformat(), "end": end.isoformat()},
            "headersRaw": headers, "rawRows": out,
            "support": {"requestedDates": len(requested), "observedDates": len(seen), "missingDates": missing,
                        "firstObserved": min(seen).isoformat() if seen else None,
                        "lastObserved": max(seen).isoformat() if seen else None},
            "unitConversion": None, "publicationTime": None,
            "limitations": ["No inference of first publication time.", "Wholesale auction target equivalence is not established."]}


def _normalized_price(value):
    parsed = _price(value)
    if parsed["status"] == "observed":
        number = Decimal(parsed["decimal"])
        if not number.is_finite(): return ("nonfinite", re.sub(r"\s+", "", value)), False
        return ("observed", str(number.normalize())), True
    if parsed["status"] == "missing": return ("missing", "-"), True
    return (parsed["status"], re.sub(r"\s+", "", value)), False


def validate_manifest(manifest_path: Path, input_dir: Path, output_path: Path) -> dict:
    """Validate an offline UI-export manifest and create a compact JSON report once."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries: raise ValueError("manifest files must be a nonempty list")
    summaries, parsed_by_path = [], {}
    seen_paths = set()
    for entry in entries:
        rel = entry["path"]
        if rel in seen_paths: raise ValueError(f"duplicate manifest path: {rel}")
        seen_paths.add(rel)
        path = (input_dir / rel).resolve()
        if path.parent != input_dir.resolve(): raise ValueError(f"manifest path must be a direct child: {rel}")
        raw = path.read_bytes()
        sha = hashlib.sha256(raw).hexdigest()
        if len(raw) != entry["bytes"]: raise ValueError(f"byte-count mismatch for {rel}")
        if sha != entry["sha256"]: raise ValueError(f"SHA-256 mismatch for {rel}")
        result = parse_export(path, start=dt.date.fromisoformat(entry["start"]), end=dt.date.fromisoformat(entry["end"]),
                              item=entry["item"], variety=entry["variety"], grade=entry["grade"], unit=entry["unit"])
        if len(result["rawRows"]) != entry["uiDataRows"]: raise ValueError(f"UI row-count mismatch for {rel}")
        parsed_by_path[rel] = (entry, result)
        prices = [price for row in result["rawRows"] for price in row["prices"].values()]
        summaries.append({"path": rel, "sha256": sha, "bytes": len(raw), "requested": result["requested"],
                          "uiDataRows": entry["uiDataRows"], "support": result["support"],
                          "priceCellCount": len(prices),
                          "missingPriceCells": sum(p["status"] == "missing" for p in prices),
                          "invalidPriceCells": sum(p["status"] == "invalid" for p in prices),
                          "negativePriceCells": sum("negative_price" in p["flags"] for p in prices),
                          "nonfinitePriceCells": sum("nonfinite_price" in p["flags"] for p in prices),
                          "qualityStatus": "review_needed" if any(p["status"] == "invalid" or p["flags"] for p in prices) else "clear",
                          "missingDatesNotClassifiedAsClosure": True,
                          "firstPublicationTime": None})
    comparisons = []
    for small_path, (small, small_result) in parsed_by_path.items():
        small_days = (dt.date.fromisoformat(small["end"]) - dt.date.fromisoformat(small["start"])).days + 1
        if small_days > 10: continue
        for large_path, (large, large_result) in parsed_by_path.items():
            large_days = (dt.date.fromisoformat(large["end"]) - dt.date.fromisoformat(large["start"])).days + 1
            if small_path == large_path or large_days < 60 or any(small[k] != large[k] for k in ("item", "variety", "grade", "unit")):
                continue
            large_rows = {r["date"]: r for r in large_result["rawRows"]}
            overlap = sorted({r["date"] for r in small_result["rawRows"]} & set(large_rows))
            if not overlap: continue
            a_rows = {r["date"]: r for r in small_result["rawRows"]}
            columns = [h for h in small_result["headersRaw"] if h not in ("구분", "등락율")]
            differences = []
            for day in overlap:
                for col in columns:
                    a = a_rows[day]["prices"][col]["raw"]
                    b = large_rows[day]["prices"][col]["raw"]
                    norm_a, valid_a = _normalized_price(a)
                    norm_b, valid_b = _normalized_price(b)
                    if not valid_a or not valid_b or norm_a != norm_b:
                        differences.append({"date": day, "column": col, "shortRaw": a, "quarterRaw": b,
                                            "numericComparison": "unavailable" if not valid_a or not valid_b else "different"})
            comparisons.append({"shortPath": small_path, "quarterPath": large_path,
                                "overlapDates": overlap, "dateNormalization": "ISO YYYY-MM-DD",
                                "priceColumns": columns, "comparison": "Decimal numeric equality; '-' and empty both missing; nonfinite/invalid values are not comparable",
                                "cellComparisons": len(overlap) * len(columns), "differences": differences,
                                "exactMatch": not differences, "qualityStatus": "review_needed" if differences else "clear"})
    report = {"source": "offline KAMIS public UI export validation", "manifestSha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              "fileCount": len(summaries), "files": summaries,
              "cabbageWeekQuarterComparisons": comparisons,
              "qualityStatus": "review_needed" if any(s["qualityStatus"] == "review_needed" for s in summaries) or any(c["qualityStatus"] == "review_needed" for c in comparisons) else "clear",
              "missingCellsAreSeparateFromPriceErrors": True,
              "missingDatesNotClassifiedAsClosure": True,
              "limitations": manifest.get("limitations", []),
              "firstPublicationTime": None}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
    return report


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "validate-manifest":
        ap = argparse.ArgumentParser(description="Validate downloaded KAMIS files against an offline provenance manifest")
        ap.add_argument("mode", choices=["validate-manifest"])
        ap.add_argument("--manifest", required=True, type=Path)
        ap.add_argument("--input-dir", required=True, type=Path)
        ap.add_argument("--output", required=True, type=Path)
        args = ap.parse_args(argv)
        try: validate_manifest(args.manifest, args.input_dir, args.output)
        except (OSError, UnicodeError, ValueError, KeyError, TypeError) as e:
            print(f"manifest validation failed: {e}", file=sys.stderr); return 2
        return 0
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", type=Path)
    ap.add_argument("--start", required=True, type=dt.date.fromisoformat)
    ap.add_argument("--end", required=True, type=dt.date.fromisoformat)
    ap.add_argument("--item", required=True); ap.add_argument("--variety", required=True)
    ap.add_argument("--grade", required=True); ap.add_argument("--unit", required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args(argv)
    try:
        result = parse_export(args.file, start=args.start, end=args.end, item=args.item,
                              variety=args.variety, grade=args.grade, unit=args.unit)
        with args.output.open("x", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write("\n")
    except (OSError, UnicodeError, ValueError) as e:
        print(f"validation failed: {e}", file=sys.stderr); return 2
    return 0


if __name__ == "__main__": raise SystemExit(main())
