"""Offline parser and bounded-interval coverage summary for KMA warning CSV files."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

HEADER = ("발표시간", "지역", "발효시각", "해당지역", "내용")


class WarningCsvError(ValueError):
    """The CSV cannot be interpreted without losing source meaning."""


@dataclass(frozen=True)
class WarningRow:
    issue_text: str
    region: str
    effective_text: str
    affected_area: str
    content: str
    issue_time: dt.datetime


def _parse_issue_time(value: str, line: int) -> dt.datetime:
    # This is a local Korea timestamp, not a claim about first publication time.
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y%m%d%H%M%S", "%Y-%m-%d %H:%M"):
        try:
            return dt.datetime.strptime(value, fmt)
        except ValueError:
            pass
    raise WarningCsvError(f"line {line}: invalid 발표시간")


def parse_csv(path: str | Path) -> tuple[list[WarningRow], str]:
    """Read CP949 (UTF-8 BOM tolerated) and preserve all five source strings."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            text = raw.decode("cp949")
        except UnicodeDecodeError as exc:
            raise WarningCsvError("input is neither UTF-8 nor CP949") from exc
    rows: list[WarningRow] = []
    seen: dict[tuple[str, str, str, str], str] = {}
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    try:
        header = next(reader)
    except (StopIteration, csv.Error) as exc:
        raise WarningCsvError("missing or malformed CSV header") from exc
    if tuple(header) != HEADER:
        raise WarningCsvError("unexpected CSV header")
    for record in reader:
        line = reader.line_num
        if len(record) != 5:
            raise WarningCsvError(f"line {line}: expected exactly 5 fields")
        issue_text, region, effective_text, affected_area, content = record
        issue = _parse_issue_time(issue_text, line)
        if not region:
            raise WarningCsvError(f"line {line}: blank 지역")
        key = (issue_text, region, effective_text, affected_area)
        if key in seen and seen[key] != content:
            raise WarningCsvError(f"line {line}: conflicting duplicate warning row")
        seen[key] = content
        rows.append(WarningRow(issue_text, region, effective_text, affected_area, content, issue))
    return rows, digest


def build_manifest(path: str | Path, query_start: dt.date, query_end: dt.date,
                   allowed_regions: Iterable[str] | None = None) -> dict:
    if query_end < query_start:
        raise ValueError("query end must not precede query start")
    allowed = sorted(set(allowed_regions or ()))
    rows, digest = parse_csv(path)
    start = dt.datetime.combine(query_start, dt.time.min)
    end_exclusive = dt.datetime.combine(query_end + dt.timedelta(days=1), dt.time.min)
    outside = [r for r in rows if not start <= r.issue_time < end_exclusive]
    if outside:
        raise WarningCsvError("one or more 발표시간 values fall outside the declared query range")
    if allowed:
        unexpected = Counter(r.region for r in rows if r.region not in set(allowed))
        if unexpected:
            details = ", ".join(f"{region} ({count})" for region, count in sorted(unexpected.items()))
            raise WarningCsvError(f"source contains regions outside --allowed-region: {details}")
    date_counts = Counter(r.issue_time.date().isoformat() for r in rows)
    region_counts = Counter(r.region for r in rows)
    origins = []
    date = query_start
    while date <= query_end:
        origin = dt.datetime.combine(date, dt.time(10, 0))
        lower = origin - dt.timedelta(hours=24)
        if lower >= start and origin <= end_exclusive:
            by_region = Counter(r.region for r in rows if lower <= r.issue_time < origin)
            if allowed:
                by_region = Counter({region: by_region[region] for region in allowed})
            origins.append({"origin_kst": origin.isoformat(timespec="minutes"),
                            "prior_24h_event_counts_by_region": dict(sorted(by_region.items()))})
        date += dt.timedelta(days=1)
    return {
        "source_file": Path(path).name,
        "source_sha256": digest,
        "encoding": "UTF-8" if _is_utf8(Path(path).read_bytes()) else "CP949",
        "timestamp_interpretation": "naive Korea local datetime; not asserted to be first publication time",
        "query_interval": {"start_date_inclusive": query_start.isoformat(),
                           "end_date_inclusive": query_end.isoformat()},
        "allowed_regions": allowed,
        "row_count": len(rows),
        "counts_by_issue_date": dict(sorted(date_counts.items())),
        "counts_by_region": dict(sorted(region_counts.items())),
        "before_10_kst_issue_count": sum(r.issue_time.hour < 10 for r in rows),
        "ten_kst_origins": origins,
        "coverage_note": "Counts describe rows in this declared query interval. Event counts do not establish active-warning state; completeness beyond the supplied file cannot be proven.",
    }


def _is_utf8(raw: bytes) -> bool:
    try:
        raw.decode("utf-8-sig")
        return True
    except UnicodeDecodeError:
        return False


def atomic_json(path: str | Path, value: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, target)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--query-start", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--query-end", required=True, type=dt.date.fromisoformat)
    parser.add_argument("--allowed-region", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.input.resolve() == args.output.resolve():
            raise WarningCsvError("output path resolves to the input CSV; refusing to overwrite source data")
        manifest = build_manifest(args.input, args.query_start, args.query_end, args.allowed_region)
        atomic_json(args.output, manifest)
    except (OSError, WarningCsvError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
