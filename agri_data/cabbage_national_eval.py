"""Evaluate immutable prospective cabbage national-level challenger issues."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from statistics import median
from typing import Any

from . import cabbage_national_daily as daily
from . import cabbage_national_prospective as prospective
from . import direct_horizon as dh

ROOT = Path(__file__).resolve().parents[1]
KST = prospective.KST
FIRST_ISSUE = dt.date(2026, 10, 7)
STATIC_ARTIFACT = Path("outputs/prospective-cabbage/2026-10-07/national-level-challenger.json")
DAILY_ARTIFACT_ROOT = Path("outputs/prospective-cabbage/daily")
CAPTURE_ROOT = DAILY_ARTIFACT_ROOT
HORIZONS = tuple(range(1, 11))
LOSSES = ("ae", "ape")
HAC = (9, 28)
MIN_PAIRS = 30
ALPHA = 0.05


class CabbageEvalError(ValueError):
    """Raised when an immutable forecast or source fails validation."""


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(data: bytes, label: str) -> Any:
    try:
        return json.loads(data.decode("utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(
            CabbageEvalError(f"{label} contains non-finite JSON number {value}")))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CabbageEvalError(f"{label} is not valid UTF-8 JSON") from exc


def _safe_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CabbageEvalError(f"{label} path is missing")
    rel = Path(value)
    if rel.is_absolute() or ".." in rel.parts:
        raise CabbageEvalError(f"{label} path must be project-relative")
    resolved = (root / rel).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise CabbageEvalError(f"{label} path escapes project root") from exc
    return resolved


def _read_sha(root: Path, path_value: Any, expected: Any, label: str) -> tuple[Path, bytes]:
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise CabbageEvalError(f"{label} SHA-256 is invalid")
    path = _safe_path(root, path_value, label)
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise CabbageEvalError(f"cannot read {label}: {path_value}") from exc
    if _sha(payload) != expected:
        raise CabbageEvalError(f"{label} SHA-256 mismatch")
    return path, payload


def _kst(value: Any, label: str) -> dt.datetime:
    try:
        result = prospective._parse_kst(value, label)
    except prospective.ProspectiveCabbageError as exc:
        raise CabbageEvalError(str(exc)) from exc
    return result


def _date(value: Any, label: str) -> dt.date:
    try:
        return prospective._date(value, label)
    except prospective.ProspectiveCabbageError as exc:
        raise CabbageEvalError(str(exc)) from exc


def _validate_static_source(root: Path, inputs: dict[str, Any]) -> dict[str, Any]:
    if inputs.get("nationalHtmlPath") != str(prospective.RAW_HTML_PATH):
        raise CabbageEvalError("first-issue national HTML path differs from pinned source")
    if inputs.get("nationalManifestPath") != str(prospective.MANIFEST_PATH):
        raise CabbageEvalError("first-issue national manifest path differs from pinned source")
    html_path, html = _read_sha(root, inputs.get("nationalHtmlPath"),
                                inputs.get("nationalHtmlSha256"), "first-issue national HTML")
    manifest_path, manifest = _read_sha(root, inputs.get("nationalManifestPath"),
                                        inputs.get("nationalManifestSha256"), "first-issue national manifest")
    if inputs.get("nationalHtmlSha256") != prospective.RAW_HTML_SHA256 or inputs.get("nationalManifestSha256") != prospective.MANIFEST_SHA256:
        raise CabbageEvalError("first-issue source hashes differ from frozen values")
    if inputs.get("nationalPageUrl") != prospective.NATIONAL_PAGE_URL or inputs.get("nationalSelection") != prospective.EXPECTED_SELECTION:
        raise CabbageEvalError("first-issue selected national source metadata differs from frozen values")
    if (inputs.get("localPricesPath") != str(prospective.LOCAL_PRICES_PATH) or
            inputs.get("localPricesSha256") != prospective.LOCAL_PRICES_SHA256):
        raise CabbageEvalError("first-issue local prices path/hash differs from frozen input")
    try:
        validated = prospective.validate_national_inputs(html, manifest)
    except prospective.ProspectiveCabbageError as exc:
        raise CabbageEvalError(f"first-issue national source validation failed: {exc}") from exc
    return {"nationalHtmlPath": str(html_path.relative_to(root)), "nationalHtmlSha256": _sha(html),
            "nationalManifestPath": str(manifest_path.relative_to(root)), "nationalManifestSha256": _sha(manifest),
            "nationalPrices": validated["nationalPrices"]}


def _validate_daily_source(root: Path, issue: dt.date, issued_at: dt.datetime,
                           inputs: dict[str, Any]) -> dict[str, Any]:
    html_path, html_bytes = _read_sha(root, inputs.get("nationalHtmlPath"),
                                      inputs.get("nationalHtmlSha256"), "daily national HTML")
    manifest_path, manifest_bytes = _read_sha(root, inputs.get("nationalManifestPath"),
                                              inputs.get("captureManifestSha256"), "daily capture manifest")
    if inputs.get("nationalManifestSha256") != _sha(manifest_bytes):
        raise CabbageEvalError("daily national manifest SHA-256 mismatch")
    if inputs.get("captureManifestPath") != inputs.get("nationalManifestPath"):
        raise CabbageEvalError("daily capture manifest path disagrees with national manifest path")
    manifest = _json_bytes(manifest_bytes, "daily capture manifest")
    if not isinstance(manifest, dict):
        raise CabbageEvalError("daily capture manifest must be an object")
    if (manifest.get("captureState") != "validated_price_observations" or
            manifest.get("httpStatus") != 200 or manifest.get("bodyComplete") is not True):
        raise CabbageEvalError("daily national capture is not a complete validated HTTP 200 response")
    selected = issue - dt.timedelta(days=1)
    if manifest.get("issueDateKst") != issue.isoformat() or manifest.get("selectedDate") != selected.isoformat():
        raise CabbageEvalError("daily capture issue/selected date mismatch")
    if inputs.get("requestUrl") != daily.build_page_url(selected) or manifest.get("requestUrl") != daily.build_page_url(selected):
        raise CabbageEvalError("daily capture request URL differs from frozen selection")
    if inputs.get("httpStatus") != 200 or inputs.get("nationalSelection") != manifest.get("validatedSelection"):
        raise CabbageEvalError("daily artifact selected-page metadata differs from capture manifest")
    captured = _kst(manifest.get("capturedAtKst"), "daily capture time")
    if captured.date() != issue or not dt.time(10, 0) <= captured.time().replace(tzinfo=None) <= dt.time(12, 0):
        raise CabbageEvalError("daily capture was outside same-day 10:00-12:00 KST window")
    if captured > issued_at:
        raise CabbageEvalError("daily capture follows challenger issue time")
    if inputs.get("capturedAtKst") != manifest.get("capturedAtKst"):
        raise CabbageEvalError("daily artifact capture timestamp differs from manifest")
    raw_ref = manifest.get("rawResponse")
    if not isinstance(raw_ref, dict) or raw_ref.get("path") != inputs.get("nationalHtmlPath") or raw_ref.get("sha256") != _sha(html_bytes):
        raise CabbageEvalError("daily manifest raw response does not match challenger source")
    try:
        html = html_bytes.decode("utf-8", errors="strict")
        selected_data = daily._response_selection(html, selected)
        daily._validate_mean_price_card(html)
        chart = prospective.parse_nongnet_mean_series(html)
    except (UnicodeDecodeError, daily.DailyCabbageError, prospective.ProspectiveCabbageError) as exc:
        raise CabbageEvalError(f"daily raw national source cannot be revalidated: {exc}") from exc
    if selected.isoformat() not in chart or any(dt.date.fromisoformat(day) > selected for day in chart):
        raise CabbageEvalError("daily chart has no selected date or includes future values")
    manifest_chart = manifest.get("meanPricesWonPerKg")
    if not isinstance(manifest_chart, dict) or {str(k): float(v) for k, v in manifest_chart.items()} != chart:
        raise CabbageEvalError("daily manifest chart differs from raw HTML")
    if manifest.get("selection") != selected_data or manifest.get("validatedSelection") != selected_data:
        raise CabbageEvalError("daily manifest validated selection differs from response")
    return {"nationalHtmlPath": str(html_path.relative_to(root)), "nationalHtmlSha256": _sha(html_bytes),
            "nationalManifestPath": str(manifest_path.relative_to(root)), "nationalManifestSha256": _sha(manifest_bytes),
            "capturedAtKst": captured.isoformat(), "selectedDate": selected.isoformat(),
            "nationalPrices": chart}


def _validate_artifact(root: Path, artifact_path: Path, payload: bytes) -> dict[str, Any]:
    try:
        if artifact_path.read_bytes() != payload:
            raise CabbageEvalError("forecast artifact bytes differ from immutable file on disk")
    except OSError as exc:
        raise CabbageEvalError(f"cannot read forecast artifact: {artifact_path}") from exc
    artifact = _json_bytes(payload, f"forecast artifact {artifact_path}")
    if not isinstance(artifact, dict) or artifact.get("schemaVersion") != 1:
        raise CabbageEvalError("forecast artifact schema is unsupported")
    kind = artifact.get("artifactType")
    static = kind == "cabbage_national_level_prospective_challenger"
    if not static and kind != "cabbage_national_level_daily_challenger":
        raise CabbageEvalError("unsupported cabbage challenger artifact type")
    issue = _date(artifact.get("issueDateKst"), "issueDateKst")
    if (static and issue != FIRST_ISSUE) or (not static and issue <= FIRST_ISSUE):
        raise CabbageEvalError("forecast artifact issue date does not match its frozen path family")
    origin = _date(artifact.get("originDate"), "originDate")
    if origin != issue:
        raise CabbageEvalError("forecast origin must equal issue date")
    issued_at = _kst(artifact.get("issueAtKst"), "issueAtKst")
    if issued_at.date() != issue or not dt.time(10, 0) <= issued_at.time().replace(tzinfo=None) <= dt.time(12, 0):
        raise CabbageEvalError("forecast issue timestamp is outside 10:00-12:00 KST")
    expected_rel = STATIC_ARTIFACT if static else DAILY_ARTIFACT_ROOT / issue.isoformat() / "challenger.json"
    try:
        relative = artifact_path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise CabbageEvalError("forecast artifact is outside project root") from exc
    if relative != expected_rel:
        raise CabbageEvalError("forecast artifact is not at its canonical immutable path")
    inputs = artifact.get("inputs")
    if not isinstance(inputs, dict):
        raise CabbageEvalError("forecast input provenance is missing")
    source = _validate_static_source(root, inputs) if static else _validate_daily_source(root, issue, issued_at, inputs)
    local_path, local_bytes = _read_sha(root, inputs.get("localPricesPath"), inputs.get("localPricesSha256"), "local prices snapshot")
    local_payload = _json_bytes(local_bytes, "local prices snapshot")
    try:
        local_prices = prospective.local_cabbage_prices(local_payload)
    except prospective.ProspectiveCabbageError as exc:
        raise CabbageEvalError(f"local prices source cannot be revalidated: {exc}") from exc
    baseline_path, baseline_bytes = _read_sha(root, inputs.get("baselineArtifactPath"),
                                              inputs.get("baselineArtifactSha256"), "baseline artifact")
    baseline_doc = _json_bytes(baseline_bytes, "baseline artifact")
    try:
        baseline = prospective.validate_baseline_artifact(baseline_doc, issue_date=issue)
    except prospective.ProspectiveCabbageError as exc:
        raise CabbageEvalError(f"baseline artifact cannot be revalidated: {exc}") from exc
    baseline_at = _kst(baseline_doc.get("issueAtKst"), "baseline issueAtKst")
    if baseline_at > issued_at or inputs.get("baselineIssueAtKst") != baseline_doc.get("issueAtKst"):
        raise CabbageEvalError("baseline chronology or recorded issue time mismatch")
    if inputs.get("localObservationDates") is not None:
        local_dates = sorted(day.isoformat() for day in local_prices)
        if sorted(inputs["localObservationDates"]) != local_dates:
            raise CabbageEvalError("recorded local observation dates differ from source snapshot")
    try:
        earliest_path, earliest_bytes, _ = prospective.select_earliest_baseline(root, issue)
    except prospective.ProspectiveCabbageError as exc:
        raise CabbageEvalError(f"same-day earliest Ridge baseline cannot be selected: {exc}") from exc
    if (earliest_path.resolve() != baseline_path.resolve() or
            _sha(earliest_bytes) != _sha(baseline_bytes)):
        raise CabbageEvalError("referenced baseline is not the earliest valid same-day Ridge artifact")
    national_prices = source.get("nationalPrices")
    if not isinstance(national_prices, dict):
        raise CabbageEvalError("validated national price series is missing")
    ratio_info = artifact.get("levelRatio")
    if not isinstance(ratio_info, dict):
        raise CabbageEvalError("challenger level-ratio provenance is missing")
    window_start, window_end = issue - dt.timedelta(days=15), issue - dt.timedelta(days=2)
    if (ratio_info.get("windowStart") != window_start.isoformat() or
            ratio_info.get("windowEnd") != window_end.isoformat() or
            ratio_info.get("sourcePublicationTimeKnown") is not False):
        raise CabbageEvalError("challenger level-ratio window or publication-time caveat differs from protocol")
    eligible = []
    for day in sorted(set(local_prices) & {dt.date.fromisoformat(key) for key in national_prices}):
        if window_start <= day <= window_end:
            national_value = prospective._positive(national_prices[day.isoformat()], f"national price {day}")
            local_value = local_prices[day]
            eligible.append({"date": day.isoformat(), "nationalPrice": national_value,
                             "localPrice": local_value, "ratio": national_value / local_value})
    expected_ratio = median(row["ratio"] for row in eligible) if len(eligible) >= 3 else 1.0
    if ratio_info.get("eligiblePairCount") != len(eligible):
        raise CabbageEvalError("recorded level-ratio eligible-pair count differs from source inputs")
    recorded_pairs = ratio_info.get("eligiblePairs")
    if not isinstance(recorded_pairs, list) or len(recorded_pairs) != len(eligible):
        raise CabbageEvalError("recorded level-ratio eligible pairs differ from source inputs")
    for recorded, expected in zip(recorded_pairs, eligible):
        if not isinstance(recorded, dict) or any(recorded.get(key) != expected[key] for key in ("date",)):
            raise CabbageEvalError("recorded level-ratio eligible date differs from source inputs")
        for key in ("nationalPrice", "localPrice", "ratio"):
            try:
                actual_num = float(recorded.get(key))
            except (TypeError, ValueError, OverflowError) as exc:
                raise CabbageEvalError("recorded level-ratio pair contains an invalid number") from exc
            if not math.isclose(actual_num, expected[key], rel_tol=1e-12, abs_tol=1e-12):
                raise CabbageEvalError("recorded level-ratio pair differs from source inputs")
    if not math.isclose(prospective._positive(ratio_info.get("ratio"), "recorded national/local ratio"),
                        expected_ratio, rel_tol=1e-12, abs_tol=1e-12):
        raise CabbageEvalError("recorded national/local ratio differs from validated source inputs")
    expected_fallback = (None if len(eligible) >= 3 else
                         f"fewer than 3 eligible same-day positive national/local pairs ({len(eligible)})")
    if ratio_info.get("fallbackReason") != expected_fallback:
        raise CabbageEvalError("recorded level-ratio fallback reason differs from frozen rule")
    rows = artifact.get("forecasts")
    if not isinstance(rows, list) or len(rows) != 10:
        raise CabbageEvalError("challenger must contain exactly ten forecast rows")
    normalized = {}
    for row in rows:
        if not isinstance(row, dict):
            raise CabbageEvalError("challenger forecast row must be an object")
        h = row.get("horizonDays")
        if isinstance(h, bool) or not isinstance(h, int) or h not in HORIZONS or h in normalized:
            raise CabbageEvalError("challenger horizons must be unique integers 1 through 10")
        target = _date(row.get("target"), f"h{h} target")
        expected_target = issue + dt.timedelta(days=h)
        if target != expected_target:
            raise CabbageEvalError(f"challenger h{h} target does not equal issue date plus horizon")
        baseline_row = baseline["predictions"][h]
        if static:
            baseline_value = prospective._positive(row.get("baselineRidgePrediction"), f"baseline h{h}")
            candidate = prospective._positive(row.get("challengerPrediction"), f"candidate h{h}")
        else:
            baseline_value = prospective._positive(row.get("baselineRidgePrediction"), f"baseline h{h}")
            candidate = prospective._positive(row.get("challengerPrediction"), f"candidate h{h}")
        if baseline_value != baseline_row["predictedPrice"]:
            raise CabbageEvalError(f"recorded baseline forecast differs from baseline artifact at h{h}")
        if not math.isclose(candidate, baseline_value * expected_ratio, rel_tol=1e-12, abs_tol=1e-10):
            raise CabbageEvalError(f"challenger h{h} prediction differs from verified baseline × level ratio")
        normalized[h] = {"horizon": h, "origin": issue, "target": target,
                         "baseline": baseline_value, "candidate": candidate}
    if set(normalized) != set(HORIZONS):
        raise CabbageEvalError("challenger horizons are incomplete")
    return {"issueDate": issue, "issuedAt": issued_at, "path": str(relative),
            "sha256": _sha(payload), "source": source,
            "baselinePath": str(baseline_path.relative_to(root)),
            "baselineSha256": _sha(baseline_bytes), "localPricesPath": str(local_path.relative_to(root)),
            "localPricesSha256": _sha(local_bytes), "rows": normalized}


def _capture_manifests(root: Path, issue_day: dt.date) -> list[tuple[dt.datetime, Path, bytes, dict[str, Any]]]:
    root = root.resolve()
    directory = root / CAPTURE_ROOT / issue_day.isoformat() / "captures"
    if not directory.is_dir():
        return []
    records = []
    for path in directory.glob("national-*.json"):
        try:
            payload = path.read_bytes()
            manifest = _json_bytes(payload, f"capture manifest {path}")
            timestamp = _kst(manifest.get("capturedAtKst"), "capturedAtKst")
            records.append((timestamp, path, payload, manifest))
        except (OSError, CabbageEvalError):
            # A malformed earliest manifest still matters; use its filename ordering
            # only when it cannot be assigned a capture timestamp.
            fallback = dt.datetime.combine(issue_day, dt.time.min, KST)
            records.append((fallback, path, b"", {}))
    return sorted(records, key=lambda item: (item[0], item[1].name))


def _raw_nonpositive_target(html: str, target: dt.date) -> bool:
    day = target.strftime("%Y%m%d")
    pattern = re.compile(r"else\s*\{[^}]*?tmpObj\s*\[\s*['\"]date['\"]\s*\]\s*=\s*['\"]" + day +
                         r"['\"][^}]*?tmpObj\s*\[\s*['\"]value['\"]\s*\]\s*=\s*['\"](-?[0-9,.]+)['\"]", re.S)
    match = pattern.search(html)
    if not match:
        return False
    try:
        value = float(match.group(1).replace(",", ""))
    except ValueError:
        return False
    return math.isfinite(value) and value <= 0


def _raw_target_date_present(html: str, target: dt.date) -> bool:
    day = target.strftime("%Y%m%d")
    return re.search(r"else\s*\{[^}]*?tmpObj\s*\[\s*['\"]date['\"]\s*\]\s*=\s*['\"]" + day + r"['\"]", html, re.S) is not None


def _truth_for_target(root: Path, target: dt.date, as_of: dt.datetime) -> dict[str, Any]:
    root = root.resolve()
    available_at = dt.datetime.combine(target + dt.timedelta(days=1), dt.time(12, 0), KST)
    records = _capture_manifests(root, target + dt.timedelta(days=1))
    if not records:
        if as_of < available_at:
            return {"status": "immature", "reason": "next_day_capture_window_not_closed"}
        return {"status": "missed_next_day_capture", "reason": "no_capture_manifest_on_target_plus_one"}
    failures = []
    prior_attempts = []
    for captured_at, manifest_path, manifest_bytes, manifest in records:
        if captured_at > as_of:
            continue
        raw_metadata = manifest.get("rawResponse") if isinstance(manifest, dict) else None
        audit = {"captureManifestPath": str(manifest_path.relative_to(root)),
                 "captureManifestSha256": _sha(manifest_bytes) if manifest_bytes else None,
                 "capturedAtKst": captured_at.isoformat(), "captureState": manifest.get("captureState"),
                 "httpStatus": manifest.get("httpStatus"), "bodyComplete": manifest.get("bodyComplete"),
                 "rawHtmlPath": raw_metadata.get("path") if isinstance(raw_metadata, dict) else None,
                 "rawHtmlSha256": raw_metadata.get("sha256") if isinstance(raw_metadata, dict) else None}
        if not manifest:
            failures.append(("invalid_capture", "capture_manifest_invalid"))
            prior_attempts.append({**audit, "reason": "capture_manifest_invalid"})
            continue
        state = manifest.get("captureState")
        if state in ("empty_http_body", "http_success_empty_chart"):
            failures.append(("empty_capture", str(manifest.get("reason") or state)))
            prior_attempts.append({**audit, "reason": str(manifest.get("reason") or state)})
            continue
        if (manifest.get("issueDateKst") != (target + dt.timedelta(days=1)).isoformat() or
                manifest.get("selectedDate") != target.isoformat() or manifest.get("httpStatus") != 200 or
                manifest.get("bodyComplete") is not True):
            failures.append(("invalid_capture", "capture_not_complete_selected_http_200"))
            prior_attempts.append({**audit, "reason": "capture_not_complete_selected_http_200"})
            continue
        if (captured_at.date() != target + dt.timedelta(days=1) or
                not dt.time(10, 0) <= captured_at.time().replace(tzinfo=None) <= dt.time(12, 0) or captured_at > as_of):
            failures.append(("invalid_capture", "capture_time_outside_next_day_window_or_after_as_of"))
            prior_attempts.append({**audit, "reason": "capture_time_outside_next_day_window_or_after_as_of"})
            continue
        base_record = {**audit, "priorCaptureAttempts": list(prior_attempts)}
        try:
            raw_ref = manifest.get("rawResponse")
            if not isinstance(raw_ref, dict):
                raise CabbageEvalError("raw response reference missing")
            raw_path, raw_bytes = _read_sha(root, raw_ref.get("path"), raw_ref.get("sha256"), "truth raw HTML")
            if raw_path.parent != manifest_path.parent or raw_path.suffix != ".html":
                raise CabbageEvalError("truth raw HTML path is outside capture directory")
            if manifest.get("requestUrl") != daily.build_page_url(target):
                raise CabbageEvalError("truth request URL differs from target selection")
            html = raw_bytes.decode("utf-8", errors="strict")
            response_selection = daily._response_selection(html, target)
            if (manifest.get("selection") != response_selection or
                    manifest.get("validatedSelection") != response_selection):
                raise CabbageEvalError("capture manifest selection differs from selected page")
            daily._validate_mean_price_card(html)
            if state == "chart_validation_failed" and _raw_nonpositive_target(html, target):
                failures.append(("nonpositive_truth", "first_selected_capture_target_price_not_positive"))
                prior_attempts.append({**audit, "reason": "first_selected_capture_target_price_not_positive"})
                continue
            if state == "chart_validation_failed" and _raw_target_date_present(html, target):
                failures.append(("invalid_capture", "target_value_null_or_invalid_in_first_selected_chart"))
                prior_attempts.append({**audit, "reason": "target_value_null_or_invalid_in_first_selected_chart"})
                continue
            if state != "validated_price_observations":
                raise CabbageEvalError("capture response did not pass full validation")
            try:
                prices = prospective.parse_nongnet_mean_series(html)
            except prospective.ProspectiveCabbageError as exc:
                if _raw_nonpositive_target(html, target):
                    failures.append(("nonpositive_truth", "first_selected_capture_target_price_not_positive"))
                    prior_attempts.append({**audit, "reason": "first_selected_capture_target_price_not_positive"})
                    continue
                if _raw_target_date_present(html, target):
                    failures.append(("invalid_capture", "target_value_null_or_invalid_in_first_selected_chart"))
                    prior_attempts.append({**audit, "reason": "target_value_null_or_invalid_in_first_selected_chart"})
                    continue
                raise exc
            if target.isoformat() not in prices:
                failures.append(("invalid_capture", "target_date_absent_from_first_valid_chart"))
                prior_attempts.append({**audit, "reason": "target_date_absent_from_first_valid_chart"})
                continue
            recorded_prices = manifest.get("meanPricesWonPerKg")
            if not isinstance(recorded_prices, dict) or {str(k): float(v) for k, v in recorded_prices.items()} != prices:
                raise CabbageEvalError("capture manifest prices differ from raw chart")
            return {"status": "paired", "reason": None, "actual": prices[target.isoformat()],
                    "rawHtmlPath": str(raw_path.relative_to(root)), "rawHtmlSha256": _sha(raw_bytes), **base_record}
        except (OSError, UnicodeDecodeError, CabbageEvalError, daily.DailyCabbageError,
                prospective.ProspectiveCabbageError, TypeError, ValueError) as exc:
            failures.append(("invalid_capture", f"capture_validation_failed:{exc}"))
            prior_attempts.append({**audit, "reason": str(exc)})
    # Failed/empty attempts remain visible, but only the first fully validated
    # response can define truth. No subsequent-day page is considered.
    if as_of < available_at:
        return {"status": "immature", "reason": "next_day_capture_window_not_closed",
                "priorCaptureAttempts": prior_attempts}
    status, reason = failures[0] if failures else ("missed_next_day_capture", "no_eligible_capture_before_as_of")
    return {"status": status, "reason": reason, "priorCaptureAttempts": prior_attempts}


def _baseline_days(root: Path, as_of: dt.datetime) -> dict[dt.date, dict[str, Any]]:
    """Index same-day validated Ridge baselines, including days with no challenger."""
    directory = root / "outputs" / "live-forecasts"
    result: dict[dt.date, dict[str, Any]] = {}
    if not directory.is_dir():
        return result
    for day_dir in directory.iterdir():
        try:
            issue = dt.date.fromisoformat(day_dir.name)
        except ValueError:
            continue
        if not day_dir.is_dir() or issue < FIRST_ISSUE or issue > as_of.date():
            continue
        options = []
        for path in day_dir.glob("*.json"):
            if path.name == "latest.json" or "provenance" in path.name:
                continue
            try:
                payload = path.read_bytes()
                document = _json_bytes(payload, f"baseline {path}")
                validated = prospective.validate_baseline_artifact(document, issue_date=issue)
                issued = _kst(document.get("issueAtKst"), "baseline issueAtKst")
                if issued <= as_of:
                    options.append((issued, path, payload, validated))
            except (OSError, CabbageEvalError, prospective.ProspectiveCabbageError):
                continue
        if options:
            issued, path, payload, validated = min(options, key=lambda item: (item[0], str(item[1])))
            result[issue] = {"issuedAt": issued, "path": str(path.relative_to(root)),
                             "sha256": _sha(payload), "predictions": validated["predictions"]}
    return result


def _check_as_of(as_of: dt.datetime, now: dt.datetime) -> dt.datetime:
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise CabbageEvalError("as-of timestamp must be timezone-aware")
    if now.tzinfo is None or now.utcoffset() is None:
        raise CabbageEvalError("current clock timestamp must be timezone-aware")
    current = now.astimezone(KST)
    requested = as_of.astimezone(KST)
    if requested > current:
        raise CabbageEvalError("as-of timestamp cannot be later than the current KST time")
    return requested


def _metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    actual = [r["actual"] for r in rows]
    return {"n": len(rows), "candidate": dh._metrics(actual, [r["candidate"] for r in rows]),
            "baseline": dh._metrics(actual, [r["baseline"] for r in rows])}


def _loss_diff(row: dict[str, Any], loss: str) -> float:
    actual, candidate, baseline = row["actual"], row["candidate"], row["baseline"]
    return abs(actual - candidate) - abs(actual - baseline) if loss == "ae" else 100 * (abs(actual - candidate) - abs(actual - baseline)) / actual


def build_assessment(root: str | Path, forecast_inputs: list[tuple[str | Path, bytes]], as_of: dt.datetime) -> dict[str, Any]:
    project = Path(root).resolve()
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise CabbageEvalError("as-of timestamp must be timezone-aware")
    as_of = as_of.astimezone(KST)
    artifacts = []
    seen = set()
    for path_value, payload in forecast_inputs:
        path = _safe_path(project, str(path_value), "forecast artifact")
        header = _json_bytes(payload, f"forecast artifact {path}")
        if isinstance(header, dict):
            header_time = _kst(header.get("issueAtKst"), "issueAtKst")
            if header_time > as_of.astimezone(KST):
                continue
        artifact = _validate_artifact(project, path, payload)
        issue = artifact["issueDate"]
        if issue in seen:
            raise CabbageEvalError(f"duplicate challenger issue date: {issue}")
        seen.add(issue)
        artifacts.append(artifact)
    artifacts.sort(key=lambda item: (item["issueDate"], item["issuedAt"]))
    baselines = _baseline_days(project, as_of)
    coverage, paired_rows, reasons = [], [], {}
    truth_cache: dict[dt.date, dict[str, Any]] = {}

    def get_truth(target: dt.date) -> dict[str, Any]:
        if target not in truth_cache:
            truth_cache[target] = _truth_for_target(project, target, as_of)
        return truth_cache[target]

    for artifact in artifacts:
        for horizon in HORIZONS:
            forecast = artifact["rows"][horizon]
            target = forecast["target"]
            truth = get_truth(target)
            row = {"origin": artifact["issueDate"].isoformat(), "target": target.isoformat(),
                   "horizon": horizon, "forecastPath": artifact["path"],
                   "forecastSha256": artifact["sha256"], "baseline": forecast["baseline"],
                   "candidate": forecast["candidate"], "candidateStatus": "issued",
                   "truthStatus": truth["status"],
                   "truthReason": truth.get("reason"), "truthCaptureManifestPath": truth.get("captureManifestPath"),
                   "truthRawHtmlPath": truth.get("rawHtmlPath"), "truthRawHtmlSha256": truth.get("rawHtmlSha256"),
                   "priorCaptureAttempts": truth.get("priorCaptureAttempts", [])}
            if truth["status"] == "paired":
                row["actual"] = truth["actual"]
                paired_rows.append(row)
            reasons[truth["status"]] = reasons.get(truth["status"], 0) + 1
            coverage.append(row)
    for issue, baseline in sorted(baselines.items()):
        if issue in seen:
            continue
        for horizon in HORIZONS:
            target = issue + dt.timedelta(days=horizon)
            truth = get_truth(target)
            row = {"origin": issue.isoformat(), "target": target.isoformat(), "horizon": horizon,
                   "forecastPath": None, "forecastSha256": None, "baselinePath": baseline["path"],
                   "baselineSha256": baseline["sha256"], "baseline": baseline["predictions"][horizon]["predictedPrice"],
                   "candidate": None, "candidateStatus": "not_issued", "truthStatus": truth["status"],
                   "truthReason": truth.get("reason"), "unpairedReason": "challenger_not_issued",
                   "truthCaptureManifestPath": truth.get("captureManifestPath"),
                   "truthRawHtmlPath": truth.get("rawHtmlPath"), "truthRawHtmlSha256": truth.get("rawHtmlSha256"),
                   "priorCaptureAttempts": truth.get("priorCaptureAttempts", [])}
            coverage.append(row)
            reasons["challenger_not_issued"] = reasons.get("challenger_not_issued", 0) + 1
            if truth["status"] != "paired":
                reasons[truth["status"]] = reasons.get(truth["status"], 0) + 1
    comparisons = []
    for horizon in HORIZONS:
        rows = [r for r in paired_rows if r["horizon"] == horizon]
        tests = []
        for loss in LOSSES:
            diffs = {_date(r["target"], "DM target date"): _loss_diff(r, loss) for r in rows}
            tests.append({"horizon": horizon, "loss": loss, "n": len(rows),
                          "candidateMetric": dh._metrics([r["actual"] for r in rows], [r["candidate"] for r in rows])["mae" if loss == "ae" else "mape"],
                          "baselineMetric": dh._metrics([r["actual"] for r in rows], [r["baseline"] for r in rows])["mae" if loss == "ae" else "mape"],
                          "dm": {str(bw): dh._dm(diffs, bw) for bw in HAC}})
        comparisons.extend(tests)
    for bandwidth in HAC:
        dh._holm([test["dm"][str(bandwidth)] for test in comparisons])
    conclusions = []
    for horizon in HORIZONS:
        rows = [r for r in paired_rows if r["horizon"] == horizon]
        tests = [r for r in comparisons if r["horizon"] == horizon]
        enough = len(rows) >= MIN_PAIRS and all(test["dm"]["9"].get("status") == "tested" for test in tests)
        metrics_better = all(test["candidateMetric"] is not None and test["baselineMetric"] is not None and test["candidateMetric"] < test["baselineMetric"] for test in tests)
        tests_improve = all(test["dm"]["9"].get("holmSignificant") and test["dm"]["9"].get("meanLossDifference", 0) < 0 for test in tests)
        baseline_coverage = sum(1 for issue, baseline in baselines.items()
                                if get_truth(issue + dt.timedelta(days=horizon))["status"] == "paired")
        coverage_nondeclining = len(rows) >= baseline_coverage
        worsening = any(test["dm"]["9"].get("holmSignificant") and test["dm"]["9"].get("meanLossDifference", 0) > 0 for test in tests)
        status = ("inference_unavailable" if not enough else "improvement" if metrics_better and tests_improve and coverage_nondeclining
                  else "worsening" if worsening else "inconclusive")
        conclusions.append({"horizon": horizon, "conclusion": status, "pairedMetricsBothLower": metrics_better,
                            "bothHac9HolmTestsImprove": tests_improve, "coverageNondeclining": coverage_nondeclining,
                            "candidatePairedCoverage": len(rows), "baselineEligibleCoverage": baseline_coverage,
                            "minimumPairsMet": len(rows) >= MIN_PAIRS,
                            "interpretation": "non-significance is not evidence of equivalence"})
    macro = {}
    for method in ("candidate", "baseline"):
        per_h = []
        counts = {}
        for horizon in HORIZONS:
            rows = [r for r in paired_rows if r["horizon"] == horizon]
            metric = dh._metrics([r["actual"] for r in rows], [r[method] for r in rows])["mape"]
            per_h.append(metric)
            counts[str(horizon)] = len(rows)
        macro[method] = {"equalHorizonMape": sum(per_h) / len(per_h) if all(v is not None for v in per_h) else None,
                         "pairedCountsByHorizon": counts, "interpretation": "descriptive_only"}
    seasonal = []
    for horizon in HORIZONS:
        horizon_rows = [r for r in paired_rows if r["horizon"] == horizon]
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in horizon_rows:
            day = _date(row["target"], "target")
            season = "winter" if day.month in (12, 1, 2) else "spring" if day.month in (3, 4, 5) else "summer" if day.month in (6, 7, 8) else "autumn"
            groups.setdefault(("year", str(day.year)), []).append(row)
            groups.setdefault(("season", season), []).append(row)
        for (kind, label), group in sorted(groups.items()):
            seasonal.append({"horizon": horizon, "stratum": kind, "value": label, **_metric_rows(group)})
    return {"schemaVersion": 1, "status": "complete", "asOfKst": as_of.isoformat(),
            "protocol": {"horizons": list(HORIZONS), "minimumPairs": MIN_PAIRS,
                         "hacCalendarDays": list(HAC), "alpha": ALPHA,
                         "familySizePerBandwidth": len(HORIZONS) * len(LOSSES),
                         "truthRule": "first next-day capture only; no later revisions; no imputation"},
            "forecastArtifacts": [{"path": a["path"], "sha256": a["sha256"],
                                   "issueDate": a["issueDate"].isoformat(), "issuedAtKst": a["issuedAt"].isoformat(),
                                   "baselinePath": a["baselinePath"], "baselineSha256": a["baselineSha256"],
                                   "source": a["source"], "localPricesPath": a["localPricesPath"],
                                   "localPricesSha256": a["localPricesSha256"]} for a in artifacts],
            "baselineIssueDays": [{"issueDate": day.isoformat(), "issuedAtKst": entry["issuedAt"].isoformat(),
                                   "path": entry["path"], "sha256": entry["sha256"]}
                                  for day, entry in sorted(baselines.items())],
            "coverage": coverage, "exclusionCounts": reasons, "pairedRows": paired_rows,
            "pairedCoverageByHorizon": {str(h): {"challengerIssued": sum(r["horizon"] == h and r["candidateStatus"] == "issued" for r in coverage),
                                                   "baselineEligible": sum(1 for issue in baselines if get_truth(issue+dt.timedelta(days=h))["status"] == "paired"),
                                                   "paired": sum(r["horizon"] == h for r in paired_rows),
                                                   "coverageNondeclining": sum(r["horizon"] == h for r in paired_rows) >= sum(1 for issue in baselines if get_truth(issue+dt.timedelta(days=h))["status"] == "paired"),
                                                   "exclusions": {reason: sum(r["horizon"] == h and (r.get("unpairedReason") == reason or (r.get("unpairedReason") is None and r["truthStatus"] == reason)) for r in coverage) for reason in sorted(reasons)}} for h in HORIZONS},
            "comparisons": comparisons, "conclusions": conclusions,
            "macroMetrics": macro, "yearSeasonMetrics": seasonal,
            "limitations": "National-page first public posting time is unknown; results describe first validated next-day captures only, not official contemporaneous availability. Small or incomplete samples do not establish equivalence."}


def _write_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(tmp, path)
    except FileExistsError as exc:
        raise CabbageEvalError(f"refusing to overwrite existing assessment: {path}") from exc
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _forecast_inputs(root: Path) -> list[Path]:
    paths = []
    static = root / STATIC_ARTIFACT
    if static.is_file():
        paths.append(static)
    daily_dir = root / DAILY_ARTIFACT_ROOT
    if daily_dir.is_dir():
        for day_dir in sorted(daily_dir.iterdir()):
            path = day_dir / "challenger.json"
            if path.is_file():
                paths.append(path)
    return paths


def run(root: str | Path, output: str | Path, as_of: dt.datetime, *,
        clock=None) -> dict[str, Any]:
    project = Path(root).resolve()
    as_of = _check_as_of(as_of, (clock or (lambda: dt.datetime.now(KST)))())
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = project / output_path
    inputs = _forecast_inputs(project)
    if not inputs:
        raise CabbageEvalError("no immutable cabbage challenger artifacts found")
    for path in inputs:
        if path.resolve() == output_path.resolve():
            raise CabbageEvalError("assessment output must not replace a forecast input")
    files = [(path.relative_to(project), path.read_bytes()) for path in inputs]
    result = build_assessment(project, files, as_of)
    payload = (json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    _write_immutable(output_path, payload)
    return {"output": str(output_path), "sha256": _sha(payload),
            "forecastArtifactCount": len(result["forecastArtifacts"]), "pairedCount": len(result["pairedRows"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--as-of", required=True, help="timezone-aware ISO timestamp")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        as_of = _kst(args.as_of, "as-of timestamp")
        result = run(args.root, args.output, as_of)
    except (CabbageEvalError, prospective.ProspectiveCabbageError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
