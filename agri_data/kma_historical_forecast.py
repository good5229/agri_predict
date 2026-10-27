"""Strict offline parsing for downloaded KMA historical short-term forecast ZIPs.

This module never performs network or browser access. It keeps the input archive
hash and raw member bytes available to callers so parsed values remain traceable.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import math
import re
import zipfile
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath

UTC = dt.timezone.utc
KST = dt.timezone(dt.timedelta(hours=9))
MAX_ARCHIVE_BYTES = 10 * 1024 * 1024
MAX_MEMBER_BYTES = 20 * 1024 * 1024
_OBSERVED_INVALID_DAY_SENTINEL = "-971917344"
_ARCHIVE_NAME = re.compile(r"(?P<location>[a-z0-9-]+)-(?P<start>\d{8})-(?P<end>\d{8})-request\d+\.zip\Z")
_HEADER = re.compile(
    r"^format: day\(UTC\),hour\(UTC\),forecast,value,day\(KST\),hour\(KST\)\s+"
    r"location:(?P<location>\d+_\d+)\s+Start\s*:\s*(?P<start>\d{8})\s*$"
)
_START_MARKER = re.compile(r"^Start\s*:\s*(?P<start>\d{8})\s*$")
_COMBINED_START_HEADER = re.compile(
    r"^Start\s*:\s*(?P<marker>\d{8})\s+(?P<header>format:.*)$"
)
_KST_COLUMNS_SWITCH_DATE = dt.date(2024, 11, 28)


@dataclass(frozen=True)
class ForecastRecord:
    issue_utc: dt.datetime
    issue_kst: dt.datetime
    lead_hours: int
    valid_utc: dt.datetime
    valid_kst: dt.datetime
    value: float
    value_source: str
    source_field_count: int
    kst_timestamp_source: str
    eligible_at_10_kst: bool


@dataclass(frozen=True)
class QuarantinedForecastRow:
    line_number: int
    marker_date: dt.date
    raw_line: str
    reason: str


@dataclass(frozen=True)
class ForecastArchive:
    archive_name: str
    archive_sha256: str
    member_name: str
    member_sha256: str
    raw_member: bytes
    location_code: str
    start_date: dt.date
    end_date: dt.date
    records: tuple[ForecastRecord, ...]
    quarantined_rows: tuple[QuarantinedForecastRow, ...]
    exact_duplicate_count: int
    # These are observations only. The source provides no row-count contract.
    coverage_evidence: dict[str, object]


def _date(text: str) -> dt.date:
    return dt.datetime.strptime(text, "%Y%m%d").date()


def _clock(text: str) -> dt.time:
    if not re.fullmatch(r"\d{4}", text):
        raise ValueError(f"invalid HHMM clock: {text!r}")
    hour, minute = int(text[:2]), int(text[2:])
    if hour > 23 or minute > 59:
        raise ValueError(f"invalid HHMM clock: {text!r}")
    return dt.time(hour, minute)


def parse_kma_forecast_zip(
    archive: bytes,
    archive_name: str,
    *,
    expected_location: str = "seoul",
    expected_location_code: str = "60_127",
    cutoff_hour_kst: int = 10,
) -> ForecastArchive:
    """Parse and validate one locally downloaded ZIP without external access.

    Exact duplicate rows are retained once and counted. Conflicting values for
    the same issue/lead pair are rejected. The input bytes and member bytes are
    returned unchanged for provenance.
    """
    if not isinstance(archive, bytes) or not archive:
        raise ValueError("archive must be non-empty bytes")
    if len(archive) > MAX_ARCHIVE_BYTES:
        raise ValueError("archive exceeds size cap")
    if not 0 <= cutoff_hour_kst <= 23:
        raise ValueError("cutoff_hour_kst must be between 0 and 23")
    match = _ARCHIVE_NAME.fullmatch(PurePosixPath(archive_name).name)
    if not match:
        raise ValueError("archive filename must encode location, date range, and request number")
    if match.group("location") != expected_location:
        raise ValueError("archive filename location does not match expected location")
    start, end = _date(match.group("start")), _date(match.group("end"))
    if end < start:
        raise ValueError("archive date range is reversed")

    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as zf:
            entries = zf.infolist()
            if len(entries) != 1:
                raise ValueError("archive must contain exactly one member")
            entry = entries[0]
            if entry.is_dir() or entry.file_size <= 0 or entry.file_size > MAX_MEMBER_BYTES:
                raise ValueError("archive member is empty, a directory, or exceeds size cap")
            if entry.flag_bits & 0x1:
                raise ValueError("encrypted archive members are unsupported")
            member = zf.read(entry)  # zipfile also verifies the CRC
    except (zipfile.BadZipFile, RuntimeError, OSError) as exc:
        raise ValueError(f"invalid or truncated ZIP archive: {exc}") from exc
    if len(member) != entry.file_size:
        raise ValueError("archive member length does not match ZIP metadata")
    try:
        text = member.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("CSV must be UTF-8 or UTF-8 with BOM") from exc
    lines = text.splitlines()
    if len(lines) < 2:
        raise ValueError("CSV has no data records")
    last_content_line = len(lines) - 1
    while last_content_line >= 0 and not lines[last_content_line].strip():
        last_content_line -= 1
    if last_content_line < 0:
        raise ValueError("CSV has no content")
    parsed: dict[tuple[dt.datetime, int], ForecastRecord] = {}
    quarantined: list[QuarantinedForecastRow] = []
    duplicates = 0
    active_date: dt.date | None = None
    last_marker: dt.date | None = None
    marker_dates_seen: set[dt.date] = set()
    marker_cycles = 0
    location_header_seen = False
    combined_line_count = 0
    logical_lines = []
    for line_number, physical_line in enumerate(lines[:last_content_line + 1], start=1):
        combined = _COMBINED_START_HEADER.fullmatch(physical_line.strip())
        if combined:
            combined_line_count += 1
            logical_lines.append((line_number, f"Start : {combined.group('marker')}"))
            logical_lines.append((line_number, combined.group("header")))
        else:
            logical_lines.append((line_number, physical_line))
    for line_number, raw_line in logical_lines:
        if not raw_line.strip():
            raise ValueError(f"blank line inside CSV at line {line_number}")
        header = _HEADER.fullmatch(raw_line.strip())
        if header:
            if header.group("location") != expected_location_code:
                raise ValueError(f"CSV location code does not match expected location at line {line_number}")
            marker = _date(header.group("start"))
            location_header_seen = True
        else:
            start_marker = _START_MARKER.fullmatch(raw_line.strip())
            if not start_marker:
                marker = None
            else:
                if not location_header_seen:
                    raise ValueError(f"Start marker appears before the location header at line {line_number}")
                marker = _date(start_marker.group("start"))
        if marker is not None:
            if not start <= marker <= end:
                raise ValueError(f"Start marker outside archive date range at line {line_number}")
            if last_marker is None:
                if marker != start:
                    raise ValueError("first Start marker does not match archive filename start")
                marker_cycles = 1
            elif marker == start and last_marker == end:
                marker_cycles += 1
                pass  # A second complete provider cycle may repeat records.
            elif marker > last_marker:
                pass
            else:
                raise ValueError(f"inconsistent or out-of-order Start marker at line {line_number}")
            active_date = marker
            last_marker = marker
            marker_dates_seen.add(marker)
            continue
        try:
            row = next(csv.reader([raw_line], strict=True))
        except (csv.Error, StopIteration) as exc:
            raise ValueError(f"malformed or truncated CSV row at line {line_number}") from exc
        row = [cell.strip() for cell in row]
        if len(row) not in (4, 6) or any(cell == "" for cell in row):
            raise ValueError(f"malformed CSV row at line {line_number}")
        if active_date is None:
            raise ValueError(f"data row appears before a Start marker at line {line_number}")
        utc_day, utc_clock, lead_text, value_text = row[:4]
        if utc_day == _OBSERVED_INVALID_DAY_SENTINEL:
            if len(row) != 4:
                raise ValueError(f"observed invalid-day sentinel has unsupported field count at line {line_number}")
            _clock(utc_clock)
            if not re.fullmatch(r"\+\d+", lead_text):
                raise ValueError(f"invalid forecast lead at line {line_number}")
            try:
                sentinel_value = Decimal(value_text)
                sentinel_float = float(sentinel_value)
            except (InvalidOperation, ValueError, OverflowError) as exc:
                raise ValueError(f"invalid numeric value at line {line_number}") from exc
            if not sentinel_value.is_finite() or not math.isfinite(sentinel_float):
                raise ValueError(f"non-finite numeric value at line {line_number}")
            quarantined.append(QuarantinedForecastRow(
                line_number=line_number,
                marker_date=active_date,
                raw_line=raw_line,
                reason="source UTC day equals observed invalid-day sentinel; issue date unresolved",
            ))
            continue
        if not re.fullmatch(r"0?[1-9]|[12]\d|3[01]", utc_day):
            raise ValueError(f"invalid UTC calendar day at line {line_number}")
        if not re.fullmatch(r"\+\d+", lead_text):
            raise ValueError(f"invalid forecast lead at line {line_number}")
        try:
            value_decimal = Decimal(value_text)
            value = float(value_decimal)
        except (InvalidOperation, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid numeric value at line {line_number}") from exc
        if not value_decimal.is_finite() or not math.isfinite(value):
            raise ValueError(f"non-finite numeric value at line {line_number}")
        utc_time = _clock(utc_clock)
        if int(utc_day) != active_date.day:
            if len(row) == 4:
                if active_date >= _KST_COLUMNS_SWITCH_DATE:
                    raise ValueError(f"four-field row is on or after KMA KST-column switch date at line {line_number}")
            else:
                if active_date < _KST_COLUMNS_SWITCH_DATE:
                    raise ValueError(f"six-field row precedes KMA KST-column switch date at line {line_number}")
                kst_day, kst_clock = row[4:]
                issue_clock = utc_time
                expected_kst_day = int(utc_day) + int(issue_clock.hour >= 15)
                if not re.fullmatch(r"0?[1-9]|[12]\d|3[01]|32", kst_day) or int(kst_day) != expected_kst_day:
                    raise ValueError(f"UTC and KST issue dates disagree at line {line_number}")
                if _clock(kst_clock) != (dt.datetime.combine(active_date, issue_clock, tzinfo=UTC)
                                          .astimezone(KST).timetz().replace(tzinfo=None)):
                    raise ValueError(f"UTC and KST issue timestamps disagree at line {line_number}")
            quarantined.append(QuarantinedForecastRow(
                line_number=line_number,
                marker_date=active_date,
                raw_line=raw_line,
                reason=f"source UTC day {int(utc_day)} disagrees with active Start marker {active_date.isoformat()}; issue date unresolved",
            ))
            continue
        issue_utc = dt.datetime.combine(active_date, utc_time, tzinfo=UTC)
        issue_kst = issue_utc.astimezone(KST)
        if len(row) == 4:
            if issue_utc.date() >= _KST_COLUMNS_SWITCH_DATE:
                raise ValueError(f"four-field row is on or after KMA KST-column switch date at line {line_number}")
            kst_timestamp_source = "derived_from_utc_plus_9"
        else:
            if issue_utc.date() < _KST_COLUMNS_SWITCH_DATE:
                raise ValueError(f"six-field row precedes KMA KST-column switch date at line {line_number}")
            kst_day, kst_clock = row[4:]
            expected_kst_day = active_date.day + int(issue_kst.date() != active_date)
            if not re.fullmatch(r"0?[1-9]|[12]\d|3[01]|32", kst_day) or int(kst_day) != expected_kst_day:
                raise ValueError(f"UTC and KST issue dates disagree at line {line_number}")
            if _clock(kst_clock) != issue_kst.timetz().replace(tzinfo=None):
                raise ValueError(f"UTC and KST issue timestamps disagree at line {line_number}")
            kst_timestamp_source = "source_columns"
        if issue_utc.date() < start or issue_utc.date() > end:
            raise ValueError(f"issue date outside archive date range at line {line_number}")
        lead = int(lead_text[1:])
        valid_utc = issue_utc + dt.timedelta(hours=lead)
        record = ForecastRecord(
            issue_utc=issue_utc,
            issue_kst=issue_kst,
            lead_hours=lead,
            valid_utc=valid_utc,
            valid_kst=valid_utc.astimezone(KST),
            value=value,
            value_source=value_text,
            source_field_count=len(row),
            kst_timestamp_source=kst_timestamp_source,
            eligible_at_10_kst=issue_kst.timetz().replace(tzinfo=None) < dt.time(cutoff_hour_kst),
        )
        key = (issue_utc, lead)
        previous = parsed.get(key)
        if previous:
            if previous.value_source != value_text:
                raise ValueError(f"conflicting duplicate forecast at line {line_number}")
            duplicates += 1
        else:
            parsed[key] = record
    if not parsed and not quarantined:
        raise ValueError("CSV has no valid data records")
    records = tuple(sorted(parsed.values(), key=lambda r: (r.issue_utc, r.lead_hours)))
    expected_dates = {start + dt.timedelta(days=i) for i in range((end - start).days + 1)}
    issue_dates = {r.issue_utc.date() for r in records}
    return ForecastArchive(
        archive_name=PurePosixPath(archive_name).name,
        archive_sha256=hashlib.sha256(archive).hexdigest(),
        member_name=entry.filename,
        member_sha256=hashlib.sha256(member).hexdigest(),
        raw_member=member,
        location_code=expected_location_code,
        start_date=start,
        end_date=end,
        records=records,
        quarantined_rows=tuple(quarantined),
        exact_duplicate_count=duplicates,
        coverage_evidence={
            "record_count_after_exact_deduplication": len(records),
            "exact_duplicate_rows": duplicates,
            "issue_dates_observed": sorted({r.issue_utc.date().isoformat() for r in records}),
            "start_marker_dates_observed": sorted(day.isoformat() for day in marker_dates_seen),
            "start_marker_cycles_observed": marker_cycles,
            "combined_marker_header_lines": combined_line_count,
            "missing_marker_dates": sorted(day.isoformat() for day in expected_dates - marker_dates_seen),
            "missing_issue_dates": sorted(day.isoformat() for day in expected_dates - issue_dates),
            "quarantined_row_count": len(quarantined),
            "quarantined_marker_dates": sorted({row.marker_date.isoformat() for row in quarantined}),
            "quarantined_reasons": sorted({row.reason for row in quarantined}),
            "lead_hours_observed": sorted({r.lead_hours for r in records}),
            "first_issue_utc": min((r.issue_utc for r in records), default=None).isoformat()
            if records else None,
            "last_issue_utc": max((r.issue_utc for r in records), default=None).isoformat()
            if records else None,
            "completeness_claim": "not_asserted; source has no expected-row-count contract",
        },
    )
