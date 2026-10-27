"""Issue an immutable, prospective cabbage national-level anchor challenger.

This module deliberately does not fetch the national page or call the live
forecast issuer. It validates already captured, hash-pinned inputs and builds
an auditable challenger artifact from them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from statistics import median
from typing import Any

KST = dt.timezone(dt.timedelta(hours=9), "KST")
RAW_HTML_PATH = Path("data/raw/nongnet-cabbage-national-page-20261007T0926KST.html")
MANIFEST_PATH = Path("data/raw/nongnet-cabbage-national-page-20261007T0926KST.json")
LOCAL_PRICES_PATH = Path("outputs/daily/2026-10-06/20261006T090116-f908e0a4/prices.json")
RAW_HTML_SHA256 = "02f3eeacbf2a6b3f80d7551cd34dc654bb536f6423f2ac4e6762ee7adb096e4c"
MANIFEST_SHA256 = "c1807e87cff98bf8ad3035162d759f2d1ed48354c8990b6beaf66f858cdc8bcf"
LOCAL_PRICES_SHA256 = "47cdd787b66a6b0c6e42b750d0b3f7e845dbde4d7ab4fdeed3c8c40472da5605"
NATIONAL_PAGE_URL = ("https://www.nongnet.or.kr/front/M000000259/marketInfo/nationWide.do"
                     "?searchDate=2026%EB%85%84%2010%EC%9B%94%2006%EC%9D%BC"
                     "&searchName1=%2A%EC%A0%84%EA%B5%AD%EB%8F%84%EB%A7%A4%EC%8B%9C%EC%9E%A5"
                     "&searchName2=%EB%B0%B0%EC%B6%94&searchSymbol1=100000&searchSymbol2=1001")
EXPECTED_SELECTION = {
    "market": "*전국도매시장", "product": "배추", "productCode": "1001",
    "variety": "전체", "priceType": "평균", "unit": "원/kg",
    "selectedDate": "2026-10-06",
}
MARKET_CODES = {"110001", "110008"}
EXPECTED_NATIONAL_VALUES = {
    "2026-10-01": 928, "2026-10-02": 1007, "2026-10-03": 1008,
    "2026-10-05": 1026, "2026-10-06": 934,
}


class ProspectiveCabbageError(ValueError):
    """Raised when an input fails the frozen publication contract."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(data: bytes, label: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(
            ProspectiveCabbageError(f"{label} contains non-finite JSON number {x}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProspectiveCabbageError(f"{label} is not valid UTF-8 JSON") from exc


def _date(value: Any, label: str) -> dt.date:
    if not isinstance(value, str):
        raise ProspectiveCabbageError(f"{label} must be an ISO date")
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ProspectiveCabbageError(f"{label} must be an ISO date") from exc
    if parsed.isoformat() != value:
        raise ProspectiveCabbageError(f"{label} must use YYYY-MM-DD")
    return parsed


def _positive(value: Any, label: str) -> float:
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ProspectiveCabbageError(f"{label} must be numeric") from exc
    if not number.is_finite() or number <= 0:
        raise ProspectiveCabbageError(f"{label} must be finite and positive")
    result = float(number)
    if not math.isfinite(result) or result <= 0:
        raise ProspectiveCabbageError(f"{label} is outside finite JSON numeric range")
    return result


def _normalize_selection(manifest: dict[str, Any]) -> dict[str, Any]:
    selection = manifest.get("selection")
    if not isinstance(selection, dict):
        raise ProspectiveCabbageError("manifest selection metadata is missing")
    # These aliases only accommodate the two common transcription labels; the
    # actual selected values remain exact and no wildcard is inferred.
    aliases = {"market": ("market",), "product": ("product",),
               "productCode": ("productCode", "productId"), "variety": ("variety",),
               "priceType": ("priceType",), "unit": ("unit",),
               "selectedDate": ("selectedDate",)}
    normalized: dict[str, Any] = {}
    for key, candidates in aliases.items():
        for candidate in candidates:
            if candidate in selection:
                normalized[key] = str(selection[candidate])
                break
    if normalized != EXPECTED_SELECTION:
        raise ProspectiveCabbageError("manifest selection does not match the frozen nationwide cabbage mean series")
    return normalized


def _normalize_manifest_values(manifest: dict[str, Any]) -> dict[str, float]:
    raw = manifest.get("visibleMeanPricesWonPerKg")
    if isinstance(raw, list):
        values: dict[str, float] = {}
        for row in raw:
            if not isinstance(row, dict):
                raise ProspectiveCabbageError("manifest observations must be objects")
            date_value = row.get("date")
            value = row.get("value", row.get("price"))
            day = _date(date_value, "manifest observation date").isoformat()
            if day in values:
                raise ProspectiveCabbageError("manifest contains duplicate national dates")
            values[day] = _positive(value, f"manifest value for {day}")
        return values
    if isinstance(raw, dict):
        values = {}
        for key, value in raw.items():
            day = _date(key, "manifest observation date").isoformat()
            if day in values:
                raise ProspectiveCabbageError("manifest contains duplicate national dates")
            values[day] = _positive(value, f"manifest value for {day}")
        return values
    raise ProspectiveCabbageError("manifest values are missing")


def parse_nongnet_mean_series(html: str) -> dict[str, float]:
    """Read only date/value pairs assigned in the chart's `else` mean branch."""
    marker = re.search(r"function\s+makeGradeChart\s*\([^)]*\)\s*\{", html)
    if not marker:
        raise ProspectiveCabbageError("raw page has no makeGradeChart function")
    # Locate the function's balanced closing brace while respecting JS strings.
    start = marker.end()
    depth, quote, escaped = 1, None, False
    end = start
    while end < len(html) and depth:
        ch = html[end]
        if quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
        elif ch in "\"'`":
            quote = ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        end += 1
    if depth:
        raise ProspectiveCabbageError("makeGradeChart function is incomplete")
    function = html[start:end - 1]
    pairs: list[tuple[str, str]] = []
    for branch in re.finditer(r"\belse\s*\{", function):
        bstart = branch.end()
        bdepth, quote, escaped, bend = 1, None, False, bstart
        while bend < len(function) and bdepth:
            ch = function[bend]
            if quote:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = None
            elif ch in "\"'`":
                quote = ch
            elif ch == "{":
                bdepth += 1
            elif ch == "}":
                bdepth -= 1
            bend += 1
        block = function[bstart:bend - 1]
        date_match = re.search(r'tmpObj\s*\[\s*["\']date["\']\s*\]\s*=\s*["\'](\d{8})["\']', block)
        value_match = re.search(r'tmpObj\s*\[\s*["\']value["\']\s*\]\s*=\s*["\']([0-9,.]+)["\']', block)
        if date_match or value_match:
            if not (date_match and value_match):
                raise ProspectiveCabbageError("mean chart branch has an incomplete date/value pair")
            pairs.append((date_match.group(1), value_match.group(1).replace(",", "")))
    if not pairs:
        raise ProspectiveCabbageError("raw page has no mean chart date/value rows")
    result: dict[str, float] = {}
    for compact, raw_value in pairs:
        try:
            day = dt.datetime.strptime(compact, "%Y%m%d").date().isoformat()
        except ValueError as exc:
            raise ProspectiveCabbageError("raw page contains an invalid chart date") from exc
        if day in result:
            raise ProspectiveCabbageError("raw page contains duplicate mean chart dates")
        result[day] = _positive(raw_value, f"raw national value for {day}")
    return result


def validate_national_inputs(html_bytes: bytes, manifest_bytes: bytes, *,
                             expected_html_sha256: str = RAW_HTML_SHA256,
                             expected_manifest_sha256: str = MANIFEST_SHA256) -> dict[str, Any]:
    if _sha256(html_bytes) != expected_html_sha256:
        raise ProspectiveCabbageError("raw national HTML SHA-256 mismatch")
    if _sha256(manifest_bytes) != expected_manifest_sha256:
        raise ProspectiveCabbageError("national manifest SHA-256 mismatch")
    manifest = _json_bytes(manifest_bytes, "national manifest")
    if not isinstance(manifest, dict):
        raise ProspectiveCabbageError("national manifest must be an object")
    selection = _normalize_selection(manifest)
    transcribed = _normalize_manifest_values(manifest)
    expected = {k: float(v) for k, v in EXPECTED_NATIONAL_VALUES.items()}
    if transcribed != expected:
        raise ProspectiveCabbageError("manifest series differs from frozen transcribed national values")
    html = html_bytes.decode("utf-8", errors="strict")
    displayed = parse_nongnet_mean_series(html)
    if not set(transcribed).issubset(displayed):
        raise ProspectiveCabbageError("manifest national dates are missing from the raw HTML mean series")
    for day, value in transcribed.items():
        if displayed[day] != value:
            raise ProspectiveCabbageError(f"raw HTML mean value differs from manifest for {day}")
    raw_response = manifest.get("rawResponse")
    if not isinstance(raw_response, dict) or raw_response.get("path") != str(RAW_HTML_PATH):
        raise ProspectiveCabbageError("manifest rawResponse path does not match pinned raw HTML")
    if raw_response.get("sha256") != expected_html_sha256:
        raise ProspectiveCabbageError("manifest rawResponse SHA-256 differs from pinned raw HTML")
    if manifest.get("sourceUrl") != NATIONAL_PAGE_URL:
        raise ProspectiveCabbageError("manifest sourceUrl does not match the frozen national page")
    captured = manifest.get("capturedAtUtcMinute")
    if not isinstance(captured, str):
        raise ProspectiveCabbageError("manifest capturedAtUtcMinute is missing")
    try:
        captured_at = dt.datetime.fromisoformat(captured.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProspectiveCabbageError("manifest capturedAtUtcMinute is invalid") from exc
    if captured_at.tzinfo is None or captured_at.astimezone(KST) != dt.datetime(2026, 10, 7, 9, 26, tzinfo=KST):
        raise ProspectiveCabbageError("manifest capture time does not match pinned 09:26 KST page")
    metadata = {"capturedAtUtcMinute": captured, "rawResponse": raw_response}
    return {"selection": selection, "nationalPrices": displayed,
            "pageMetadata": metadata, "htmlSha256": expected_html_sha256,
            "manifestSha256": expected_manifest_sha256, "pageUrl": NATIONAL_PAGE_URL}


def _parse_kst(value: Any, label: str) -> dt.datetime:
    if not isinstance(value, str):
        raise ProspectiveCabbageError(f"{label} must be a timezone-aware timestamp")
    try:
        stamp = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProspectiveCabbageError(f"{label} must be an ISO timestamp") from exc
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ProspectiveCabbageError(f"{label} must include a timezone")
    return stamp.astimezone(KST)


def validate_current_issue_time(issue_date: dt.date, requested_at: dt.datetime,
                                current_at: dt.datetime, *, max_skew_seconds: int = 60) -> dt.datetime:
    """Reject backdated/future timestamps and issuance outside the live window."""
    requested = _parse_kst(requested_at.isoformat(), "requested issue timestamp")
    current = _parse_kst(current_at.isoformat(), "current clock timestamp")
    for label, stamp in (("requested", requested), ("current", current)):
        if stamp.date() != issue_date or not dt.time(10, 0) <= stamp.time().replace(tzinfo=None) <= dt.time(12, 0):
            raise ProspectiveCabbageError(f"{label} issue time must be on issue date from 10:00 through 12:00 KST")
    if abs((requested - current).total_seconds()) > max_skew_seconds:
        raise ProspectiveCabbageError("requested issue timestamp is inconsistent with current KST clock")
    return requested


def validate_baseline_artifact(data: Any, *, issue_date: dt.date) -> dict[str, Any]:
    if not isinstance(data, dict) or data.get("schemaVersion") != 1:
        raise ProspectiveCabbageError("baseline artifact schema is unsupported")
    artifact_date = _date(data.get("issueDateKst"), "baseline issueDateKst")
    if artifact_date != issue_date:
        raise ProspectiveCabbageError("baseline issue date differs from the requested issue date")
    issue_at = _parse_kst(data.get("issueAtKst"), "baseline issueAtKst")
    if issue_at.date() != issue_date or not dt.time(10, 0) <= issue_at.time().replace(tzinfo=None) <= dt.time(12, 0):
        raise ProspectiveCabbageError("baseline issue timestamp must be on issue date from 10:00 through 12:00 KST")
    if _date(data.get("sourceCutoffDate"), "baseline sourceCutoffDate") != issue_date - dt.timedelta(days=1):
        raise ProspectiveCabbageError("baseline source cutoff must be the day before issue date")
    products = data.get("products")
    if not isinstance(products, list):
        raise ProspectiveCabbageError("baseline products must be a list")
    cabbage = [p for p in products if isinstance(p, dict) and p.get("productId") == "cabbage"]
    if len(cabbage) != 1:
        raise ProspectiveCabbageError("baseline must contain exactly one cabbage product")
    product = cabbage[0]
    if product.get("status") != "available" or product.get("method") != "ridge":
        raise ProspectiveCabbageError("baseline cabbage must be available direct Ridge")
    predictions = product.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != 10:
        raise ProspectiveCabbageError("baseline cabbage must contain exactly 10 predictions")
    normalized = {}
    for row in predictions:
        if not isinstance(row, dict):
            raise ProspectiveCabbageError("baseline prediction must be an object")
        horizon = row.get("horizonDays")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in range(1, 11) or horizon in normalized:
            raise ProspectiveCabbageError("baseline horizons must be unique integers 1 through 10")
        target = _date(row.get("target"), "baseline target")
        if target != issue_date + dt.timedelta(days=horizon):
            raise ProspectiveCabbageError("baseline target does not match its calendar horizon")
        normalized[horizon] = {"target": target.isoformat(), "predictedPrice": _positive(
            row.get("predictedPrice"), f"baseline h{horizon} prediction"),
            "sourceDate": row.get("sourceDate")}
    if set(normalized) != set(range(1, 11)):
        raise ProspectiveCabbageError("baseline horizons must be exactly 1 through 10")
    fingerprint = _sha256(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), allow_nan=False).encode("utf-8"))
    return {"issueAtKst": issue_at.isoformat(timespec="seconds"),
            "sourceCutoffDate": (issue_date - dt.timedelta(days=1)).isoformat(),
            "artifactSha256": fingerprint, "predictions": normalized}


def select_earliest_baseline(project: Path, issue_date: dt.date) -> tuple[Path, bytes, dict[str, Any]]:
    """Select the earliest valid same-day cabbage Ridge artifact by KST issue time."""
    directory = project / "outputs" / "live-forecasts" / issue_date.isoformat()
    if not directory.is_dir():
        raise ProspectiveCabbageError("same-day live forecast artifact directory is missing")
    candidates = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "latest.json" or "provenance" in path.name:
            continue
        try:
            payload = path.read_bytes()
            data = _json_bytes(payload, "baseline forecast artifact")
            validated = validate_baseline_artifact(data, issue_date=issue_date)
        except (OSError, ProspectiveCabbageError):
            continue
        candidates.append((validated["issueAtKst"], path, payload, validated))
    if not candidates:
        raise ProspectiveCabbageError("no validated same-day available cabbage Ridge artifact")
    _, path, payload, validated = min(candidates, key=lambda row: (row[0], str(row[1])))
    return path, payload, validated


def local_cabbage_prices(snapshot: Any) -> dict[dt.date, float]:
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("products"), list):
        raise ProspectiveCabbageError("local prices snapshot is invalid")
    products = [p for p in snapshot["products"] if isinstance(p, dict) and p.get("id") == "cabbage"]
    if len(products) != 1:
        raise ProspectiveCabbageError("local prices snapshot must contain exactly one cabbage product")
    result: dict[dt.date, float] = {}
    observations = products[0].get("observations")
    if not isinstance(observations, list):
        raise ProspectiveCabbageError("local cabbage observations must be a list")
    for row in observations:
        if not isinstance(row, dict):
            raise ProspectiveCabbageError("local observation must be an object")
        day = _date(row.get("date"), "local observation date")
        market_codes = {str(v) for v in row.get("marketCodes", [])}
        if market_codes != MARKET_CODES:
            continue
        try:
            actual_decimal = Decimal(str(row.get("actual")))
        except (InvalidOperation, ValueError, TypeError) as exc:
            raise ProspectiveCabbageError(f"local cabbage price for {day} must be numeric") from exc
        if not actual_decimal.is_finite():
            raise ProspectiveCabbageError(f"local cabbage price for {day} must be finite")
        # A zero or negative aggregate is not a usable positive market price;
        # retain it in the source snapshot but exclude it from the anchor.
        if actual_decimal <= 0:
            continue
        actual = _positive(actual_decimal, f"local cabbage price for {day}")
        if day in result:
            raise ProspectiveCabbageError("local snapshot has duplicate cabbage dates")
        result[day] = actual
    return result


def build_challenger(*, issue_date: dt.date, issued_at: dt.datetime,
                     national_inputs: dict[str, Any], local_snapshot_bytes: bytes,
                     local_snapshot: Any, baseline: dict[str, Any],
                     baseline_path: str, baseline_sha256: str,
                     local_snapshot_path: str = str(LOCAL_PRICES_PATH),
                     local_expected_sha256: str = LOCAL_PRICES_SHA256) -> dict[str, Any]:
    issue_kst = _parse_kst(issued_at.isoformat(), "challenger issue timestamp")
    if issue_kst.date() != issue_date or not dt.time(10, 0) <= issue_kst.time().replace(tzinfo=None) <= dt.time(12, 0):
        raise ProspectiveCabbageError("challenger issue timestamp must be issue date from 10:00 through 12:00 KST")
    baseline_time = _parse_kst(baseline["issueAtKst"], "baseline issueAtKst")
    if baseline_time > issue_kst:
        raise ProspectiveCabbageError("baseline issue time must not follow challenger issue time")
    if _sha256(local_snapshot_bytes) != local_expected_sha256:
        raise ProspectiveCabbageError("local prices snapshot SHA-256 mismatch")
    national_prices: dict[str, float] = national_inputs.get("nationalPrices", {})
    if not national_prices:
        raise ProspectiveCabbageError("validated national price series is empty")
    local_prices = local_cabbage_prices(local_snapshot)
    national_dates = {dt.date.fromisoformat(day): _positive(value, "national price")
                      for day, value in national_prices.items()}
    eligible = []
    start, stop = issue_date - dt.timedelta(days=15), issue_date - dt.timedelta(days=2)
    for day in sorted(set(national_dates) & set(local_prices)):
        if start <= day <= stop:
            eligible.append({"date": day.isoformat(), "nationalPrice": national_dates[day],
                             "localPrice": local_prices[day],
                             "ratio": national_dates[day] / local_prices[day]})
    if len(eligible) >= 3:
        ratio = median(row["ratio"] for row in eligible)
        fallback_reason = None
    else:
        ratio = 1.0
        fallback_reason = f"fewer than 3 eligible same-day positive national/local pairs ({len(eligible)})"
    if not math.isfinite(ratio) or ratio <= 0:
        raise ProspectiveCabbageError("computed national/local ratio must be finite and positive")
    forecasts = []
    for horizon in range(1, 11):
        base = baseline["predictions"][horizon]
        forecasts.append({"horizonDays": horizon, "target": base["target"],
                          "baselineRidgePrediction": base["predictedPrice"],
                          "nationalLevelRatio": ratio,
                          "challengerPrediction": base["predictedPrice"] * ratio})
    return {
        "schemaVersion": 1, "artifactType": "cabbage_national_level_prospective_challenger",
        "issueDateKst": issue_date.isoformat(), "issueAtKst": issue_kst.isoformat(timespec="seconds"),
        "originDate": issue_date.isoformat(), "targetRule": "origin + calendar horizons 1 through 10",
        "inputs": {"nationalHtmlPath": str(RAW_HTML_PATH), "nationalHtmlSha256": national_inputs["htmlSha256"],
                   "nationalManifestPath": str(MANIFEST_PATH), "nationalManifestSha256": national_inputs["manifestSha256"],
                   "nationalPageUrl": national_inputs["pageUrl"], "nationalSelection": national_inputs["selection"],
                   "nationalPageMetadata": national_inputs.get("pageMetadata", {}),
                   "localPricesPath": local_snapshot_path, "localPricesSha256": local_expected_sha256,
                   "baselineArtifactPath": baseline_path, "baselineArtifactSha256": baseline_sha256,
                   "baselineIssueAtKst": baseline["issueAtKst"],
                   "localObservationDates": [d.isoformat() for d in sorted(local_prices)]},
        "levelRatio": {"windowStart": start.isoformat(), "windowEnd": stop.isoformat(),
                       "eligiblePairCount": len(eligible), "eligiblePairs": eligible,
                       "ratio": ratio, "fallbackReason": fallback_reason,
                       "sourcePublicationTimeKnown": False},
        "forecasts": forecasts,
        "evaluationNote": "Prospective challenger artifact only; no official outcome or performance claim is implied.",
    }


def _read_verified(path: Path, expected: str, label: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise ProspectiveCabbageError(f"cannot read {label}: {path}") from exc
    if _sha256(data) != expected:
        raise ProspectiveCabbageError(f"{label} SHA-256 mismatch")
    return data


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # link is create-only and fails if an artifact already exists.
        os.link(tmp_name, path)
    except FileExistsError as exc:
        raise ProspectiveCabbageError(f"refusing to overwrite existing artifact: {path}") from exc
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def issue_from_files(*, root: str | Path, baseline_path: str | Path | None,
                     issue_date: dt.date, issued_at: dt.datetime,
                     output_path: str | Path | None = None,
                     clock: Any | None = None) -> dict[str, Any]:
    """Build from frozen local inputs; caller must only use after code review."""
    project = Path(root).resolve()
    current_at = (clock or (lambda: dt.datetime.now(KST)))()
    issued_at = validate_current_issue_time(issue_date, issued_at, current_at)
    html_path, manifest_path, prices_path = (project / RAW_HTML_PATH,
                                              project / MANIFEST_PATH,
                                              project / LOCAL_PRICES_PATH)
    html_bytes = _read_verified(html_path, RAW_HTML_SHA256, "raw national HTML")
    manifest_bytes = _read_verified(manifest_path, MANIFEST_SHA256, "national manifest")
    prices_bytes = _read_verified(prices_path, LOCAL_PRICES_SHA256, "local prices snapshot")
    national = validate_national_inputs(html_bytes, manifest_bytes)
    prices = _json_bytes(prices_bytes, "local prices snapshot")
    selected_path, base_bytes, baseline = select_earliest_baseline(project, issue_date)
    if baseline_path is not None:
        requested_path = Path(baseline_path)
        if not requested_path.is_absolute():
            requested_path = project / requested_path
        if requested_path.resolve() != selected_path.resolve():
            raise ProspectiveCabbageError("provided baseline is not the earliest validated same-day artifact")
    # Input validation can take long enough to cross noon. Re-read the live
    # clock immediately before building/writing and record this publication
    # timestamp, rather than the caller's earlier requested timestamp.
    publication_at = (clock or (lambda: dt.datetime.now(KST)))()
    publication_at = validate_current_issue_time(issue_date, publication_at, publication_at,
                                                 max_skew_seconds=0)
    baseline_at = _parse_kst(baseline["issueAtKst"], "baseline issueAtKst")
    if baseline_at > publication_at:
        raise ProspectiveCabbageError("baseline issue time must not follow challenger publication time")
    artifact = build_challenger(issue_date=issue_date, issued_at=publication_at,
                                national_inputs=national, local_snapshot_bytes=prices_bytes,
                                local_snapshot=prices, baseline=baseline,
                                baseline_path=str(selected_path.relative_to(project)),
                                baseline_sha256=_sha256(base_bytes))
    output = Path(output_path) if output_path else project / "outputs" / "prospective-cabbage" / issue_date.isoformat() / "national-level-challenger.json"
    if not output.is_absolute():
        output = project / output
    # Serialize once before the final clock read so the read is after all
    # potentially slow validation/build work and immediately precedes publish.
    json.dumps(artifact, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    final_publication_at = (clock or (lambda: dt.datetime.now(KST)))()
    final_publication_at = validate_current_issue_time(
        issue_date, final_publication_at, final_publication_at, max_skew_seconds=0)
    if baseline_at > final_publication_at:
        raise ProspectiveCabbageError("baseline issue time must not follow challenger publication time")
    artifact["issueAtKst"] = final_publication_at.isoformat(timespec="seconds")
    payload = (json.dumps(artifact, ensure_ascii=False, sort_keys=True,
                          indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_immutable(output, payload)
    return {"status": "complete", "artifact": str(output.relative_to(project)),
            "artifactSha256": _sha256(output.read_bytes()), "eligibleAnchorDays": artifact["levelRatio"]["eligiblePairCount"],
            "levelRatio": artifact["levelRatio"]["ratio"], "fallbackReason": artifact["levelRatio"]["fallbackReason"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--baseline", help="optional expected earliest same-day live_forecast JSON")
    parser.add_argument("--issue-date", default="2026-10-07")
    parser.add_argument("--issued-at", required=True, help="timezone-aware issue timestamp")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        result = issue_from_files(root=args.root, baseline_path=args.baseline,
                                  issue_date=_date(args.issue_date, "issue date"),
                                  issued_at=_parse_kst(args.issued_at, "issue timestamp"),
                                  output_path=args.output)
    except ProspectiveCabbageError as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
