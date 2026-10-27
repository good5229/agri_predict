"""Read-only, immutable-cohort evaluator for the prospective E106 comparison."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from . import daily_training, dashboard, prospective_eval, equal_info_online_score as score
from . import e106_mean_intervals as intervals

ROOT = Path(__file__).resolve().parents[1]
KST = ZoneInfo("Asia/Seoul")
PRODUCTS = ("cabbage", "radish", "onion", "garlic")
HORIZONS = tuple(range(1, 11))
METHODS = ("causal50", "E", "E106")
COMPARISONS = {
    "E106_minus_E": ("E106", "E"),
    "E_minus_causal50": ("E", "causal50"),
    "E106_minus_causal50": ("E106", "causal50"),
}
LOSSES = ("ae", "ape")
BANDWIDTHS = (9, 28)
MIN_GARLIC_DATES = 30
REPLICATES = 2000
SEED = 20261011
BLOCKS = (14, 28)
EARLIEST_REVIEW = dt.date(2026, 11, 30)
ISSUE_ROOT = Path("outputs/prospective-e106/2026-10-11/validated-runtime/forecast.json")
INITIAL_RAW_SHA256 = "bd2e8566d15aed3e8a4765cd43d7def35a9b7ee04a573d18ff9a426f6364cbdc"
INITIAL_CANONICAL_SHA256 = "dbc79c9624bb5cacc4de8c3477c3b253f0bec0ebe0c3e81a2bba816102f9873f"
INITIAL_ISSUE_AT = "2026-10-11T15:19:46+09:00"
INITIAL_RUN_ID = "20261011T120011-0014dce9"
INITIAL_RUN_STATUS_SHA256 = "ce9cbf87db731a52e1233b07ef1aede6f750747477ac0564a7ca7fac6c718683"
PROTOCOL_ID = "prospective-e106-v1"
DEFAULT_REGISTRY = ROOT / "outputs/prospective-e106/registry"
DEFAULT_OUTPUT = ROOT / "outputs/prospective-e106/evaluations"


def _protocol_payload(initial: dict[str, Any]) -> dict[str, Any]:
    return {"schemaVersion": 1, "protocolId": PROTOCOL_ID,
        "cohortStartDateKst": "2026-10-11", "dailySelection": {
            "firstAttemptAtOrAfterKst": "18:00:00", "failureConsumesCohort": True,
            "rerunMayReplaceClaim": False, "plannedForecastKeyCount": 40,
            "plannedForecastGridSha256Rule": "sha256(canonical JSON sorted product, origin, target, horizon keys)"},
        "inference": {"firstMonthEndAfterSupport": EARLIEST_REVIEW.isoformat(),
            "minimumDistinctPositiveGarlicTargetsPerHorizon": MIN_GARLIC_DATES,
            "primary": "E106_minus_E garlic h1-h10 AE/APE; DM HAC9/28; Holm per bandwidth, family 20",
            "bootstrap": {"comparisons": ["E_minus_causal50/cabbage10", "E106_minus_causal50/fourCrop40",
                                            "E106_minus_E/garlic10"],
                "metrics": ["ae", "ape"], "blocks": list(BLOCKS), "replicates": REPLICATES, "seed": SEED}},
        "initialSelectedIssue": initial}


def canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def pretty(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + "\n").encode("utf-8")


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _date(value: Any, label: str) -> dt.date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc


def _timestamp(value: Any, label: str) -> dt.datetime:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a timezone-aware timestamp")
    try:
        result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be a timezone-aware timestamp") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{label} must be a timezone-aware timestamp")
    return result.astimezone(KST)


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _grid(issue_date: dt.date) -> list[dict[str, Any]]:
    return [{"product": p, "origin": issue_date.isoformat(), "target": (issue_date + dt.timedelta(days=h)).isoformat(),
             "horizon": h} for p in PRODUCTS for h in HORIZONS]


def grid_sha256(issue_date: dt.date) -> str:
    return sha(canonical(_grid(issue_date)))


def _write_create_only(path: Path, value: dict[str, Any]) -> bool:
    """Atomically create an immutable JSON file; same-byte repeats are idempotent."""
    payload = pretty(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() == payload:
            return False
        raise FileExistsError(f"immutable file already exists with different contents: {path}")
    temp = path.with_name(path.name + f".{os.getpid()}.{next(tempfile._get_candidate_names())}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temp, path)
            return True
        except FileExistsError:
            if path.read_bytes() == payload:
                return False
            raise FileExistsError(f"immutable file already exists with different contents: {path}")
    finally:
        temp.unlink(missing_ok=True)


def _forecast_doc(path: Path, *, expected_raw: str | None = None,
                  expected_canonical: str | None = None) -> tuple[dict[str, Any], str, str]:
    raw_bytes = path.read_bytes()
    raw_sha = sha(raw_bytes)
    doc = json.loads(raw_bytes)
    if not isinstance(doc, dict) or doc.get("schemaVersion") != 1:
        raise ValueError("forecast artifact must use schemaVersion 1")
    unsigned = dict(doc)
    embedded = unsigned.pop("forecastSha256", None)
    canonical_sha = sha(canonical(unsigned))
    if embedded != canonical_sha:
        raise ValueError("forecast embedded canonical digest mismatch")
    if expected_raw is not None and raw_sha != expected_raw:
        raise ValueError("forecast raw artifact hash mismatch")
    if expected_canonical is not None and canonical_sha != expected_canonical:
        raise ValueError("forecast canonical artifact hash mismatch")
    issue_date = _date(doc.get("issueDateKst"), "forecast issueDateKst")
    issued_at = _timestamp(doc.get("issueAtKst"), "forecast issueAtKst")
    if issued_at.date() != issue_date:
        raise ValueError("forecast issue date/time disagree")
    if (doc.get("phase") != "prospective_forecast_only" or doc.get("targetValuesAttached") is not False
            or doc.get("methods") != list(METHODS) or doc.get("products") != list(PRODUCTS)
            or doc.get("horizons") != list(HORIZONS) or doc.get("plannedCount") != 40):
        raise ValueError("forecast artifact is not the frozen four-crop, three-method 40-key format")
    sources = doc.get("sources")
    if not isinstance(sources, dict) or sources.get("noTargetOutcomesRead") is not True:
        raise ValueError("forecast source lineage or answer-free marker is missing")
    cutoff = _date(sources.get("sourceCutoff"), "forecast sourceCutoff")
    if cutoff >= issue_date:
        raise ValueError("forecast source cutoff must precede issue date")
    rows = doc.get("forecasts")
    expected = {(p, issue_date, issue_date + dt.timedelta(days=h), h) for p in PRODUCTS for h in HORIZONS}
    seen = set()
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("forecast artifact must contain exactly 40 records")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("forecast record must be an object")
        product = row.get("product")
        horizon = row.get("horizon")
        origin = _date(row.get("origin"), "forecast origin")
        target = _date(row.get("target"), "forecast target")
        key = (product, origin, target, horizon)
        if (type(horizon) is not int or key not in expected or key in seen
                or horizon not in HORIZONS):
            raise ValueError("forecast record is duplicate or outside its exact 40-key grid")
        if row.get("sourceCutoff") != cutoff.isoformat():
            raise ValueError("forecast keys do not share the artifact source cutoff")
        for method in METHODS:
            model = row.get(method)
            if not isinstance(model, dict) or model.get("status") not in ("issued", "not_issued"):
                raise ValueError(f"{method} forecast status is missing or invalid")
            prediction = _positive(model.get("prediction"))
            if (model["status"] == "issued") != (prediction is not None):
                raise ValueError(f"{method} forecast status/prediction mismatch")
        seen.add(key)
    if seen != expected:
        raise ValueError("forecast artifact has an incomplete 40-key grid")
    return doc, raw_sha, canonical_sha


def _initial_pin(project_root: Path) -> dict[str, Any]:
    forecast_path = project_root / ISSUE_ROOT
    doc, raw_sha, canonical_sha = _forecast_doc(
        forecast_path, expected_raw=INITIAL_RAW_SHA256, expected_canonical=INITIAL_CANONICAL_SHA256)
    sources = doc["sources"]
    source_run_id = sources.get("dailyTrainingRunId")
    run_dir = project_root / "outputs/daily/2026-10-11" / str(source_run_id)
    status_path = run_dir / "run-status.json"
    status_raw = status_path.read_bytes()
    if sha(status_raw) != INITIAL_RUN_STATUS_SHA256:
        raise ValueError("initial immutable daily-training run-status hash mismatch")
    status = json.loads(status_raw)
    if (status.get("status") != "complete" or status.get("runId") != INITIAL_RUN_ID
            or status.get("snapshotSha256") != sources.get("dailyTrainingSnapshotRecordSha256")
            or status.get("cutoff") != sources.get("sourceCutoff")):
        raise ValueError("initial forecast is not linked to the pinned complete source run")
    if _timestamp(status.get("finishedAtKst"), "initial source run finishedAtKst") > _timestamp(INITIAL_ISSUE_AT, "initial issue timestamp"):
        raise ValueError("initial source run finished after the pinned forecast issue")
    if doc.get("issueAtKst") != INITIAL_ISSUE_AT or source_run_id != INITIAL_RUN_ID:
        raise ValueError("initial forecast does not match the explicit first-success pin")
    return {"issueDateKst": doc["issueDateKst"], "attemptId": "initial-e106-success",
            "startedAtKst": doc["issueAtKst"], "artifactTarget": str(ISSUE_ROOT),
            "artifactRawSha256": raw_sha, "artifactCanonicalSha256": canonical_sha,
            "source": {"runId": source_run_id,
                       "runStatusFileSha256": INITIAL_RUN_STATUS_SHA256,
                       "snapshotSha256": sources["dailyTrainingSnapshotRecordSha256"],
                       "cutoff": sources["sourceCutoff"],
                       "pricesFileSha256": sources["pricesFileSha256"],
                       "provenanceFileSha256": sources["provenanceFileSha256"]}}


def bootstrap_registry(registry_root: Path = DEFAULT_REGISTRY,
                       project_root: Path = ROOT) -> dict[str, Any]:
    """Pin the known first normal Oct 11 issue without fitting or rescoring it."""
    initial = _initial_pin(project_root)
    protocol = _protocol_payload(initial)
    protocol["protocolSha256"] = sha(canonical(protocol))
    _write_create_only(registry_root / "protocol.json", protocol)
    issue_date = dt.date.fromisoformat(initial["issueDateKst"])
    claim = {"schemaVersion": 1, "issueDateKst": issue_date.isoformat(),
        "attemptId": initial["attemptId"], "startedAtKst": initial["startedAtKst"],
        "attemptOrdinal": 1, "plannedForecastKeyCount": 40,
        "plannedForecastGridSha256": grid_sha256(issue_date), "source": initial["source"],
        "artifactTarget": initial["artifactTarget"], "bootstrapPin": True}
    terminal = {"schemaVersion": 1, "attemptId": initial["attemptId"], "status": "complete",
        "artifactPath": initial["artifactTarget"], "artifactRawSha256": initial["artifactRawSha256"],
        "artifactCanonicalSha256": initial["artifactCanonicalSha256"], "bootstrapPin": True}
    day_dir = registry_root / "days" / issue_date.isoformat()
    _write_create_only(day_dir / "claim.json", claim)
    _write_create_only(day_dir / "terminal.json", terminal)
    return {"protocol": registry_root / "protocol.json", "claim": day_dir / "claim.json",
            "terminal": day_dir / "terminal.json", "protocolSha256": protocol["protocolSha256"]}


def _load_protocol(registry_root: Path) -> dict[str, Any]:
    raw = (registry_root / "protocol.json").read_bytes()
    doc = json.loads(raw)
    if not isinstance(doc, dict) or doc.get("schemaVersion") != 1 or doc.get("protocolId") != PROTOCOL_ID:
        raise ValueError("unsupported or malformed E106 evaluation registry protocol")
    unsigned = dict(doc)
    digest = unsigned.pop("protocolSha256", None)
    if digest != sha(canonical(unsigned)):
        raise ValueError("registry protocol digest mismatch")
    initial = doc.get("initialSelectedIssue")
    if not isinstance(initial, dict) or initial.get("artifactRawSha256") != INITIAL_RAW_SHA256 \
            or initial.get("artifactCanonicalSha256") != INITIAL_CANONICAL_SHA256:
        raise ValueError("registry does not explicitly pin the first normal Oct 11 issue")
    if (initial.get("issueDateKst") != "2026-10-11" or initial.get("attemptId") != "initial-e106-success"
            or initial.get("startedAtKst") != INITIAL_ISSUE_AT
            or initial.get("source", {}).get("runId") != INITIAL_RUN_ID
            or initial.get("source", {}).get("runStatusFileSha256") != INITIAL_RUN_STATUS_SHA256):
        raise ValueError("registry initial issue lineage differs from the frozen normal forecast")
    expected = _protocol_payload(initial)
    if {k: v for k, v in doc.items() if k != "protocolSha256"} != expected:
        raise ValueError("registry protocol fields differ from the frozen E106 review protocol")
    return doc


def _scheduled_dates(as_of: dt.datetime) -> list[dt.date]:
    end = as_of.date()
    if as_of.time().replace(tzinfo=None) < dt.time(18, 0):
        end -= dt.timedelta(days=1)
    start = dt.date(2026, 10, 11)
    if end < start:
        dates = []
    else:
        dates = [start + dt.timedelta(days=i) for i in range((end - start).days + 1)]
    initial_at = _timestamp(INITIAL_ISSUE_AT, "initial issue timestamp")
    if start not in dates and as_of >= initial_at:
        dates.insert(0, start)
    return dates


def _load_claim(registry_root: Path, issue_date: dt.date, initial: dict[str, Any],
                protocol_sha256: str) -> dict[str, Any] | None:
    path = registry_root / "days" / issue_date.isoformat() / "claim.json"
    if not path.is_file():
        return None
    raw = path.read_bytes()
    claim = json.loads(raw)
    if (not isinstance(claim, dict) or claim.get("schemaVersion") != 1
            or claim.get("issueDateKst") != issue_date.isoformat()):
        raise ValueError(f"claim date mismatch: {path}")
    if (claim.get("attemptOrdinal", 1) != 1 or claim.get("plannedForecastKeyCount") != 40
            or claim.get("plannedForecastGridSha256") != grid_sha256(issue_date)):
        raise ValueError(f"claim is not the immutable first 40-key attempt: {path}")
    if not isinstance(claim.get("attemptId"), str) or not claim["attemptId"]:
        raise ValueError(f"claim attemptId is missing: {path}")
    if issue_date.isoformat() == initial.get("issueDateKst"):
        for key in ("attemptId", "startedAtKst", "artifactTarget"):
            if claim.get(key) != initial.get(key):
                raise ValueError(f"initial claim differs from explicit first-success pin: {key}")
        if claim.get("source") != initial.get("source") or claim.get("bootstrapPin") is not True:
            raise ValueError("initial claim source/bootstrap marker differs from explicit pin")
    else:
        if claim.get("protocolSha256") != protocol_sha256:
            raise ValueError(f"claim is bound to a different registry protocol: {path}")
        unsigned = dict(claim)
        digest = unsigned.pop("claimSha256", None)
        if digest != sha(canonical(unsigned)):
            raise ValueError(f"claim content digest mismatch: {path}")
    started = _timestamp(claim.get("startedAtKst"), "claim startedAtKst")
    if started.date() != issue_date:
        raise ValueError(f"claim start date mismatch: {path}")
    if issue_date.isoformat() != "2026-10-11" and started.time().replace(tzinfo=None) < dt.time(18, 0):
        raise ValueError(f"selected first claim started before the frozen 18:00 KST selection time: {path}")
    return claim


def _terminal(registry_root: Path, issue_date: dt.date, claim: dict[str, Any]) -> dict[str, Any] | None:
    path = registry_root / "days" / issue_date.isoformat() / "terminal.json"
    if not path.is_file():
        return None
    raw = path.read_bytes()
    terminal = json.loads(raw)
    if (not isinstance(terminal, dict) or terminal.get("schemaVersion") != 1
            or terminal.get("attemptId") != claim.get("attemptId")):
        raise ValueError(f"terminal does not immutably link to first claim: {path}")
    if terminal.get("status") not in ("complete", "failed", "failure"):
        raise ValueError(f"invalid terminal status: {path}")
    if terminal.get("status") == "complete":
        for field in ("artifactPath", "artifactRawSha256", "artifactCanonicalSha256"):
            if not isinstance(terminal.get(field), str) or not terminal[field]:
                raise ValueError(f"complete terminal is missing {field}: {path}")
        if not claim.get("bootstrapPin"):
            source = terminal.get("source")
            required = ("runId", "snapshotSha256", "cutoff", "pricesFileSha256", "provenanceFileSha256")
            if not isinstance(source, dict) or any(not isinstance(source.get(key), str) or not source[key]
                                                   for key in required):
                raise ValueError(f"complete terminal source binding is missing or incomplete: {path}")
    if not claim.get("bootstrapPin"):
        if terminal.get("protocolSha256") != claim.get("protocolSha256"):
            raise ValueError(f"terminal is bound to a different registry protocol: {path}")
        if terminal.get("claimSha256") != claim.get("claimSha256"):
            raise ValueError(f"terminal is bound to different claim contents: {path}")
        unsigned = dict(terminal)
        digest = unsigned.pop("terminalSha256", None)
        if digest != sha(canonical(unsigned)):
            raise ValueError(f"terminal content digest mismatch: {path}")
        _timestamp(terminal.get("finishedAtKst"), "terminal finishedAtKst")
    elif (terminal.get("bootstrapPin") is not True
          or terminal.get("artifactRawSha256") != INITIAL_RAW_SHA256
          or terminal.get("artifactCanonicalSha256") != INITIAL_CANONICAL_SHA256):
        raise ValueError(f"initial terminal differs from explicit pinned artifact: {path}")
    return terminal


def _validate_truth_run(run_dir: Path, as_of: dt.datetime) -> dict[str, Any]:
    """Read a completed daily artifact and immutable run-status; never query its DB."""
    status_path = run_dir / "run-status.json"
    status_raw = status_path.read_bytes()
    status_sha = sha(status_raw)
    status = json.loads(status_raw)
    if not isinstance(status, dict) or status.get("status") != "complete":
        raise ValueError("truth source must be a complete immutable daily run-status")
    if (status.get("runId") != run_dir.name or status.get("runDirectory") not in (None, str(run_dir),
            str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else None)):
        raise ValueError("truth run directory does not match immutable run-status identity")
    finished = _timestamp(status.get("finishedAtKst"), "truth run finishedAtKst")
    if finished > as_of:
        raise ValueError("truth source run completed after evaluatedAsOf")
    prices_path, provenance_path = run_dir / "prices.json", run_dir / "selected-provenance.json"
    prices_raw, provenance_raw = prices_path.read_bytes(), provenance_path.read_bytes()
    prices, provenance = json.loads(prices_raw), json.loads(provenance_raw)
    fingerprint = daily_training._pipeline_fingerprint(prices, provenance)
    if fingerprint != status.get("snapshotSha256"):
        raise ValueError("truth daily-training fingerprint does not match immutable run-status")
    actuals, data_end, evaluation_end, snapshot_sha = prospective_eval._snapshot(prices_path)
    provenance_map = prospective_eval._validate_provenance(provenance, selected_end=evaluation_end)
    if (status.get("cutoff") != evaluation_end.isoformat()
            or prices.get("meta", {}).get("evaluationEnd") != evaluation_end.isoformat()):
        raise ValueError("truth cutoff does not agree with immutable daily run-status")
    for row in provenance:
        response_at = _timestamp(row.get("responseAt"), "truth source responseAt")
        if response_at > as_of:
            raise ValueError("truth source response timestamp is after evaluatedAsOf")
    scope_states = {}
    for row in provenance:
        day = _date(row["date"], "truth provenance date")
        product, market = row["product"], row["market"]
        scope_states[(product, day, market)] = {"rowCount": row["rowCount"], "epochId": row["epochId"],
            "responseAt": row["responseAt"], "jobId": row["jobId"],
            "selectedPrevious": row["selectedPrevious"], "state": row.get("state")}
    return {"runId": status["runId"], "runStatusSha256": status_sha,
            "snapshotSha256": fingerprint, "pricesSha256": sha(prices_raw),
            "provenanceSha256": sha(provenance_raw),
            "provenanceCanonicalSha256": sha(canonical(provenance)),
            "dataEnd": data_end.isoformat(), "evaluationEnd": evaluation_end.isoformat(),
            "finishedAtKst": status["finishedAtKst"], "actuals": actuals,
            "scopeStates": scope_states, "prices": prices,
            "provenanceScopeCount": len(provenance_map)}


def _validate_forecast_sources(doc: dict[str, Any], claim: dict[str, Any], terminal_source: dict[str, Any]) -> None:
    source = claim.get("source")
    if not isinstance(source, dict):
        raise ValueError("successful claim source binding must be an object")
    if not isinstance(terminal_source, dict):
        raise ValueError("successful terminal source binding must be an object")
    required = ("runId", "snapshotSha256", "cutoff", "pricesFileSha256", "provenanceFileSha256")
    if any(not isinstance(terminal_source.get(key), str) or not terminal_source[key] for key in required):
        raise ValueError("successful terminal source binding is incomplete")
    forecast_source = doc.get("sources", {})
    comparisons = {"runId": "dailyTrainingRunId", "snapshotSha256": "dailyTrainingSnapshotRecordSha256",
                   "cutoff": "sourceCutoff", "pricesFileSha256": "pricesFileSha256",
                   "provenanceFileSha256": "provenanceFileSha256"}
    for claim_key, forecast_key in comparisons.items():
        if terminal_source.get(claim_key) != forecast_source.get(forecast_key):
            raise ValueError(f"forecast source does not match successful terminal field {claim_key}")
        if source.get(claim_key) != terminal_source.get(claim_key):
            raise ValueError(f"terminal source differs from immutable first-claim field {claim_key}")


def _load_selected_issues(registry_root: Path, project_root: Path, protocol: dict[str, Any],
                          dates: list[dt.date], as_of: dt.datetime) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected, day_coverage, seen_paths, seen_targets = [], [], set(), set()
    initial = protocol["initialSelectedIssue"]
    for issue_date in dates:
        claim_path = registry_root / "days" / issue_date.isoformat() / "claim.json"
        claim = _load_claim(registry_root, issue_date, initial, protocol["protocolSha256"])
        if claim is None:
            day_coverage.append({"issueDateKst": issue_date.isoformat(), "attemptStatus": "not_attempted",
                "plannedForecastKeyCount": 40, "plannedGridSha256": grid_sha256(issue_date)})
            continue
        claim_started = _timestamp(claim["startedAtKst"], "claim startedAtKst")
        if claim_started > as_of:
            day_coverage.append({"issueDateKst": issue_date.isoformat(), "attemptStatus": "not_yet_attempted_as_of",
                "plannedForecastKeyCount": 40, "plannedGridSha256": grid_sha256(issue_date)})
            continue
        claim_raw_sha = sha(claim_path.read_bytes())
        claim_unsigned = dict(claim); claim_digest = claim_unsigned.pop("claimSha256", None)
        if claim.get("bootstrapPin"):
            claim_digest = sha(canonical(claim))
        terminal = _terminal(registry_root, issue_date, claim)
        terminal_path = registry_root / "days" / issue_date.isoformat() / "terminal.json"
        terminal_raw_sha = sha(terminal_path.read_bytes()) if terminal_path.is_file() else None
        if terminal is not None and not claim.get("bootstrapPin"):
            finished = _timestamp(terminal["finishedAtKst"], "terminal finishedAtKst")
            if finished < claim_started or finished > as_of:
                terminal = None
                terminal_raw_sha = None
        entry = {"issueDateKst": issue_date.isoformat(), "attemptId": claim["attemptId"],
                 "attemptStartedAtKst": claim["startedAtKst"], "plannedForecastKeyCount": 40,
                 "plannedGridSha256": grid_sha256(issue_date), "source": claim.get("source"),
                 "artifactTarget": claim.get("artifactTarget"),
                 "protocolSha256": protocol["protocolSha256"], "claimFileSha256": claim_raw_sha,
                 "claimSha256": claim_digest, "terminalFileSha256": terminal_raw_sha,
                 "terminalStatus": terminal.get("status") if terminal else "pending_or_abandoned"}
        if terminal is None or terminal.get("status") in ("failed", "failure"):
            day_coverage.append({**entry, "forecastKeyCount": 40,
                "executionState": entry["terminalStatus"], "forecastRows": []})
            continue
        artifact_rel = terminal.get("artifactPath", claim.get("artifactTarget"))
        if artifact_rel != claim.get("artifactTarget"):
            raise ValueError("terminal artifact path differs from immutable first claim")
        if issue_date.isoformat() == initial.get("issueDateKst"):
            if (artifact_rel != initial.get("artifactTarget")
                    or terminal.get("artifactRawSha256") != initial.get("artifactRawSha256")
                    or terminal.get("artifactCanonicalSha256") != initial.get("artifactCanonicalSha256")):
                raise ValueError("initial terminal does not match explicit pinned artifact hashes/path")
        artifact_path = (project_root / artifact_rel).resolve()
        if project_root.resolve() not in artifact_path.parents:
            raise ValueError("forecast artifact path escapes project root")
        if artifact_path in seen_paths:
            raise ValueError("one forecast artifact cannot satisfy multiple day claims")
        seen_paths.add(artifact_path)
        doc, raw_sha, canonical_sha = _forecast_doc(artifact_path,
            expected_raw=terminal.get("artifactRawSha256"),
            expected_canonical=terminal.get("artifactCanonicalSha256"))
        if doc["issueDateKst"] != issue_date.isoformat():
            raise ValueError("selected forecast issue date differs from its immutable claim")
        if doc["issueAtKst"] != claim.get("startedAtKst"):
            raise ValueError("selected forecast issue time differs from its first-claim start")
        _validate_forecast_sources(doc, claim, terminal.get("source") or initial.get("source"))
        if issue_date.isoformat() == initial.get("issueDateKst"):
            if (raw_sha != initial.get("artifactRawSha256")
                    or canonical_sha != initial.get("artifactCanonicalSha256")
                    or doc["issueAtKst"] != initial.get("startedAtKst")
                    or doc["sources"].get("dailyTrainingRunId") != initial.get("source", {}).get("runId")
                    or doc["sources"].get("dailyTrainingSnapshotRecordSha256") != initial.get("source", {}).get("snapshotSha256")):
                raise ValueError("initial forecast artifact differs from explicit first-success pin")
        finish = terminal.get("finishedAtKst")
        if not claim.get("bootstrapPin"):
            finished = _timestamp(finish, "terminal finishedAtKst")
            if finished < claim_started or finished > as_of:
                raise ValueError("terminal completion time is outside claim/evaluation bounds")
        rows = doc["forecasts"]
        for row in rows:
            key = (row["product"], row["horizon"], row["target"])
            if key in seen_targets:
                raise ValueError("duplicate product/horizon/target across selected issue days")
            seen_targets.add(key)
        selected.append({**entry, "forecast": doc, "artifactRawSha256": raw_sha,
                         "artifactCanonicalSha256": canonical_sha})
        day_coverage.append({**entry, "forecastKeyCount": 40, "executionState": "complete",
                             "artifactRawSha256": raw_sha, "artifactCanonicalSha256": canonical_sha,
                             "forecastRows": rows})
    return selected, day_coverage


def _target_state(product: str, target: dt.date, truth: dict[str, Any], as_of: dt.datetime) -> tuple[str, float | None, dict[str, Any]]:
    if target > as_of.date():
        return "target_not_yet_arrived", None, {}
    cutoff = dt.date.fromisoformat(truth["evaluationEnd"])
    if target > cutoff:
        return "truth_source_extent_not_reached", None, {}
    actual = truth["actuals"].get(product, {}).get(target)
    sources = {market: truth["scopeStates"].get((product, target, market)) for market in dashboard.MARKETS}
    if any(item is None for item in sources.values()):
        return "source_error_or_incomplete", None, {m: x for m, x in sources.items() if x is not None}
    if any(item.get("state") in ("failed", "error", "pending", "incomplete") for item in sources.values()):
        return "source_error_or_incomplete", None, sources
    if actual is not None:
        if all(item["rowCount"] == 0 for item in sources.values()):
            return "source_error_or_incomplete", None, sources
        return "positive_actual_observed", float(actual), sources
    counts = [item["rowCount"] for item in sources.values()]
    if all(count == 0 for count in counts):
        return "normal_complete_no_eligible_price", None, sources
    # The pinned price export carries aggregate, not target-level signed-net
    # diagnostics. Do not infer that a missing positive target is nonpositive.
    known_nonpositive = False
    product_doc = next((p for p in truth["prices"].get("products", []) if p.get("id") == product), {})
    daily_states = (product_doc.get("diagnostics", {}).get("targetStatesByDate", {})
                    if isinstance(product_doc.get("diagnostics"), dict) else {})
    if isinstance(daily_states, dict):
        known_nonpositive = daily_states.get(target.isoformat()) == "nonpositive_signed_target"
    if known_nonpositive:
        return "nonpositive_signed_target", None, sources
    return "complete_rows_no_eligible_price_unclassified", None, {
        "scopes": sources,
        "reason": "complete source rows exist but positive-price and signed-target evidence is unavailable"}


def _method_outcome(row: dict[str, Any], method: str) -> dict[str, Any]:
    item = row[method]
    prediction = _positive(item.get("prediction"))
    reason = item.get("reason")
    fallback_reason = item.get("fallbackReason") or row.get(method + "FallbackReason")
    fallback = bool(item.get("fallback") or fallback_reason)
    underlying = None
    if method in ("E", "E106") and not fallback:
        # E and unchanged E106 keys preserve the causal50 policy; expose its
        # nested direct-only/SARIMA fallback rather than reporting a false zero.
        underlying = row.get("causal50")
    if method == "E106" and not fallback and isinstance(row.get("E106Blend"), dict):
        underlying = row["E106Blend"]
    if isinstance(underlying, dict):
        provenance = underlying.get("provenance") if isinstance(underlying.get("provenance"), dict) else {}
        has_nested_fallback = bool(underlying.get("fallback") or provenance.get("fallback"))
        if has_nested_fallback:
            fallback = True
            fallback_reason = (underlying.get("fallbackReason") or provenance.get("fallbackReason")
                               or underlying.get("reason") or "underlying_causal50_policy_fallback")
    return {"status": item["status"], "prediction": prediction,
            "fallback": fallback, "fallbackReason": fallback_reason,
            "reason": reason}


def _pair_row(issue: dict[str, Any], forecast_row: dict[str, Any], truth: dict[str, Any],
              as_of: dt.datetime) -> dict[str, Any]:
    issue_date = dt.date.fromisoformat(issue["issueDateKst"])
    target = dt.date.fromisoformat(forecast_row["target"])
    product, horizon = forecast_row["product"], forecast_row["horizon"]
    state, actual, source = _target_state(product, target, truth, as_of)
    methods = {method: _method_outcome(forecast_row, method) for method in METHODS}
    cohort_mature = (dt.date.fromisoformat(truth["evaluationEnd"]) >= issue_date + dt.timedelta(days=10)
                     and as_of.date() >= issue_date + dt.timedelta(days=10))
    pairs = {}
    for comparison, (a, b) in COMPARISONS.items():
        if actual is None:
            pair_state = state
        elif methods[a]["prediction"] is None or methods[b]["prediction"] is None:
            pair_state = "prediction_missing"
        else:
            pair_state = "paired_positive_actual"
        pairs[comparison] = {"status": pair_state,
            "aPrediction": methods[a]["prediction"], "bPrediction": methods[b]["prediction"]}
    return {"issueDateKst": issue_date.isoformat(), "origin": issue_date.isoformat(),
            "product": product, "horizon": horizon, "target": target.isoformat(),
            "cohortMature10d": cohort_mature, "truthState": state, "actual": actual,
            "sourceScopes": source, "methods": methods, "pairs": pairs}


def _metrics(rows: list[dict[str, Any]], method_a: str, method_b: str) -> dict[str, Any]:
    if not rows:
        return {"nPairs": 0, "uniqueTargetDates": 0, "issueCount": 0,
                "aMae": None, "bMae": None, "maeDifference": None,
                "aMape": None, "bMape": None, "mapeDifference": None}
    ae_a = [abs(r[method_a] - r["actual"]) for r in rows]
    ae_b = [abs(r[method_b] - r["actual"]) for r in rows]
    ape_a = [100 * x / r["actual"] for x, r in zip(ae_a, rows)]
    ape_b = [100 * x / r["actual"] for x, r in zip(ae_b, rows)]
    return {"nPairs": len(rows), "uniqueTargetDates": len({r["target"] for r in rows}),
            "issueCount": len({r["issueDateKst"] for r in rows}),
            "aMae": math.fsum(ae_a) / len(rows), "bMae": math.fsum(ae_b) / len(rows),
            "maeDifference": math.fsum(a - b for a, b in zip(ae_a, ae_b)) / len(rows),
            "aMape": math.fsum(ape_a) / len(rows), "bMape": math.fsum(ape_b) / len(rows),
            "mapeDifference": math.fsum(a - b for a, b in zip(ape_a, ape_b)) / len(rows)}


def _comparison_rows(pair_rows: list[dict[str, Any]], comparison: str) -> list[dict[str, Any]]:
    a, b = COMPARISONS[comparison]
    result = []
    for row in pair_rows:
        pair = row["pairs"][comparison]
        if pair["status"] != "paired_positive_actual":
            continue
        result.append({"issueDateKst": row["issueDateKst"], "product": row["product"],
                       "horizon": row["horizon"], "target": row["target"], "actual": row["actual"],
                       a: pair["aPrediction"], b: pair["bPrediction"]})
    return result


def _fixed_macro(pair_rows: list[dict[str, Any]], comparison: str,
                 cells: set[tuple[str, int]]) -> dict[str, Any]:
    a, b = COMPARISONS[comparison]
    by_cell: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in _comparison_rows(pair_rows, comparison):
        by_cell[(row["product"], row["horizon"])].append(row)
    metrics_by_cell = {f"{p}/{h}": _metrics(by_cell.get((p, h), []), a, b)
                       for p, h in sorted(cells)}
    absent = sorted(f"{p}/{h}" for p, h in cells if not by_cell.get((p, h)))
    if absent:
        return {"status": "incomplete_cell_support", "expectedCells": len(cells),
                "cellsIncluded": len(cells) - len(absent), "missingCells": absent,
                "aMae": None, "bMae": None, "maeDifference": None,
                "aMape": None, "bMape": None, "mapeDifference": None,
                "byCell": metrics_by_cell}
    cell_measures = [metrics_by_cell[f"{p}/{h}"] for p, h in sorted(cells)]
    return {"status": "complete_fixed_cell_macro", "expectedCells": len(cells),
            "cellsIncluded": len(cells), "missingCells": [],
            **{key: math.fsum(float(x[key]) for x in cell_measures) / len(cells)
               for key in ("aMae", "bMae", "maeDifference", "aMape", "bMape", "mapeDifference")},
            "byCell": metrics_by_cell}


def _thresholds(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_h = {}
    for h in HORIZONS:
        dates = {r["target"] for r in pair_rows if r["product"] == "garlic" and r["horizon"] == h
                 and r["cohortMature10d"] and r["pairs"]["E106_minus_E"]["status"] == "paired_positive_actual"}
        by_h[str(h)] = {"uniquePositivePairedTargetDates": len(dates), "required": MIN_GARLIC_DATES,
                        "ready": len(dates) >= MIN_GARLIC_DATES}
    return {"byGarlicHorizon": by_h,
            "allHorizonsReady": all(item["ready"] for item in by_h.values()),
            "criterion": "each of ten garlic horizons has at least 30 unique positive paired target dates among 10-day-mature cohorts"}


def _execution_summary(day_coverage: list[dict[str, Any]]) -> dict[str, Any]:
    states = Counter(row.get("executionState", row.get("attemptStatus", "not_attempted")) for row in day_coverage)
    issue_attempts = len(day_coverage)
    planned = issue_attempts * 40
    method_fallbacks = Counter()
    fallback_reasons = Counter()
    if any(row.get("forecastRows") for row in day_coverage):
        for day in day_coverage:
            for row in day.get("forecastRows", []):
                for method in METHODS:
                    info = _method_outcome(row, method)
                    if info["fallback"]:
                        method_fallbacks[method] += 1
                        fallback_reasons[f"{method}:{info['fallbackReason'] or info['reason'] or 'unspecified'}"] += 1
    return {"scheduledIssueDays": issue_attempts, "plannedForecastKeys": planned,
            "attemptStates": dict(sorted(states.items())),
            "fallbackKeyCountsByMethod": dict(sorted(method_fallbacks.items())),
            "fallbackReasons": dict(sorted(fallback_reasons.items()))}


def _complete_case(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_issue_product: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        by_issue_product[(row["issueDateKst"], row["product"])].append(row)
    complete = {(issue, product) for (issue, product), rows in by_issue_product.items()
                if len(rows) == 10 and all(r["truthState"] == "positive_actual_observed"
                                           and all(m["prediction"] is not None for m in r["methods"].values())
                                           for r in rows)}
    pairs = [r for r in pair_rows if (r["issueDateKst"], r["product"]) in complete]
    return {"issueProductCount": len(complete), "positiveAllTenTargetCaseRows": len(pairs),
            "definition": "separate complete-case summary; does not remove partial observed pairs from primary tables"}


def _review_inference(pair_rows: list[dict[str, Any]], truth: dict[str, Any]) -> dict[str, Any]:
    # Formal inference is restricted to the frozen 10-day-mature cohort.
    formal_rows = [row for row in pair_rows if row["cohortMature10d"]]
    tests = []
    for h in HORIZONS:
        pairs = [r for r in formal_rows if r["product"] == "garlic" and r["horizon"] == h
                 and r["cohortMature10d"]
                 and r["pairs"]["E106_minus_E"]["status"] == "paired_positive_actual"]
        for loss in LOSSES:
            diff = {}
            for row in pairs:
                day = _date(row["target"], "paired target")
                pair = row["pairs"]["E106_minus_E"]
                ae = abs(pair["aPrediction"] - row["actual"]) - abs(pair["bPrediction"] - row["actual"])
                diff[day] = ae if loss == "ae" else 100 * ae / row["actual"]
            for bandwidth in BANDWIDTHS:
                test = score._dm(diff, bandwidth)
                tests.append({"comparison": "E106_minus_E", "product": "garlic", "horizon": h,
                              "loss": loss, "bandwidth": bandwidth, **test})
    for bandwidth in BANDWIDTHS:
        family = [r for r in tests if r["bandwidth"] == bandwidth]
        if len(family) != 20:
            raise ValueError("primary DM/Holm family must remain exactly 20 tests per bandwidth")
        score._holm(family)
    intervals_by_comparison = {}
    calendar_start = dt.date(2026, 10, 12)
    primary_cells = {("garlic", h) for h in HORIZONS}
    primary_macro = _fixed_macro(formal_rows, "E106_minus_E", primary_cells)
    intervals_by_comparison["E106_minus_E"] = {"scope": "garlic 10 horizons", "macro": primary_macro,
                                                "blockIntervals": {}}
    comparisons = (("E106_minus_E", primary_cells),
                   ("E_minus_causal50", {("cabbage", h) for h in HORIZONS}),
                   ("E106_minus_causal50", {(p, h) for p in PRODUCTS for h in HORIZONS}))
    for comparison, cells in comparisons:
        rows = [r for r in _comparison_rows(formal_rows, comparison)
                if (r["product"], r["horizon"]) in cells]
        rows = [dict(r, a=r[COMPARISONS[comparison][0]], b=r[COMPARISONS[comparison][1]]) for r in rows]
        macro = _fixed_macro(formal_rows, comparison, cells)
        if comparison != "E106_minus_E":
            intervals_by_comparison[comparison] = {"scope": "cabbage 10 horizons" if comparison == "E_minus_causal50"
                else "four crops x 10 cells", "macro": macro, "blockIntervals": {}}
        block_intervals = intervals_by_comparison[comparison]["blockIntervals"]
        if macro["status"] != "complete_fixed_cell_macro":
            for metric in ("ae", "ape"):
                block_intervals[metric] = {}
                for block in BLOCKS:
                    block_intervals[metric][str(block)] = {
                        "status": "unavailable_incomplete_fixed_cell_support", "interval95": None}
            continue
        calendar_end = max((_date(r["target"], "paired target") for r in rows), default=calendar_start)
        for metric in ("ae", "ape"):
            block_intervals[metric] = {}
            for block in BLOCKS:
                result = intervals.bootstrap_estimate(rows, metric=metric, block=block, cells=frozenset(cells),
                    replicates=REPLICATES, seed=SEED, calendar_start=calendar_start, calendar_end=calendar_end)
                block_intervals[metric][str(block)] = result
    assessments = {}
    for loss, key in (("ae", "maeDifference"), ("ape", "mapeDifference")):
        difference = primary_macro[key]
        direction = "inconclusive" if difference is None else ("improvement" if difference < 0 else
                    "worsening" if difference > 0 else "no_mean_difference")
        assessments[loss] = {"meanDirection": direction, "meanDifference": difference,
                             "significantByBandwidth": {}}
        for bandwidth in BANDWIDTHS:
            family = [r for r in tests if r["loss"] == loss and r["bandwidth"] == bandwidth]
            improvements = sum(r.get("holmSignificant") is True and r.get("meanLossDifference") is not None
                              and r["meanLossDifference"] < 0 for r in family)
            worsenings = sum(r.get("holmSignificant") is True and r.get("meanLossDifference") is not None
                             and r["meanLossDifference"] > 0 for r in family)
            label = ("mixed_significant_directions" if improvements and worsenings else
                     "significant_improvements_only" if improvements else
                     "significant_worsenings_only" if worsenings else "inconclusive_no_significant_tests")
            assessments[loss]["significantByBandwidth"][str(bandwidth)] = {
                "improvementTests": improvements, "worseningTests": worsenings,
                "inconclusiveTests": len(family) - improvements - worsenings, "conclusion": label}
    return {"status": "inference_complete", "primary": {
                "comparison": "E106_minus_E", "product": "garlic", "familySizePerBandwidth": 20,
                "alpha": 0.05, "testCount": len(tests), "tests": tests,
                "meanEffectAssessment": assessments},
            "supplementary": intervals_by_comparison}


def build_report(registry_root: Path, truth: dict[str, Any], as_of: dt.datetime,
                 *, output_root: Path, mode: str = "daily", project_root: Path = ROOT) -> dict[str, Any]:
    protocol = _load_protocol(registry_root)
    dates = _scheduled_dates(as_of)
    selected, day_coverage = _load_selected_issues(registry_root, project_root, protocol, dates, as_of)
    pair_rows = []
    for issue in selected:
        for row in issue["forecast"]["forecasts"]:
            pair_rows.append(_pair_row(issue, row, truth, as_of))
    by_comparison = {}
    for comparison, (a, b) in COMPARISONS.items():
        relevant = [r for r in pair_rows if r["product"] == "garlic" and comparison == "E106_minus_E"
                    or r["product"] == "cabbage" and comparison == "E_minus_causal50"
                    or comparison == "E106_minus_causal50"]
        cells = ({("garlic", h) for h in HORIZONS} if comparison == "E106_minus_E" else
                 {("cabbage", h) for h in HORIZONS} if comparison == "E_minus_causal50" else
                 {(p, h) for p in PRODUCTS for h in HORIZONS})
        cell_summaries = {}
        for product, horizon in sorted(cells):
            pairs = []
            for row in relevant:
                pair = row["pairs"][comparison]
                if row["product"] == product and row["horizon"] == horizon and pair["status"] == "paired_positive_actual":
                    a_value, b_value = pair["aPrediction"], pair["bPrediction"]
                    pairs.append({"issueDateKst": row["issueDateKst"], "target": row["target"],
                                  "actual": row["actual"], a: a_value, b: b_value})
            cell_summaries[f"{product}/{horizon}"] = _metrics(pairs, a, b)
        macro = _fixed_macro(pair_rows, comparison, cells)
        by_comparison[comparison] = {"cells": cell_summaries, "fixedMacro": macro,
                                     "scope": "all observed positive paired dates; descriptive until frozen mature-cohort review"}
    thresholds = _thresholds(pair_rows)
    monthly_gate = as_of.date() >= EARLIEST_REVIEW and _is_month_end(as_of.date())
    reasons = []
    if not thresholds["allHorizonsReady"]:
        lacking = [h for h, x in thresholds["byGarlicHorizon"].items() if not x["ready"]]
        reasons.append({"code": "garlic_horizon_support_below_30", "horizons": lacking})
    if as_of.date() < EARLIEST_REVIEW:
        reasons.append({"code": "before_earliest_possible_review", "earliest": EARLIEST_REVIEW.isoformat()})
    elif not _is_month_end(as_of.date()):
        reasons.append({"code": "not_kst_month_end"})
    if mode not in ("daily", "review"):
        raise ValueError("mode must be daily or review")
    if mode == "daily":
        status = "not_ready"
        inference = {"status": "not_run_before_explicit_review", "reason": "daily coverage only; DM/Holm/CI are not computed"}
    elif not thresholds["allHorizonsReady"] or not monthly_gate:
        status = "not_ready"
        inference = {"status": "not_run_before_explicit_review", "reason": "review date or garlic support threshold is not satisfied"}
    else:
        status = "review_eligible"
        inference = {"status": "deferred_until_atomic_review_claim", "reason": "review claim must be reserved before inference"}
    report = {"schemaVersion": 1, "protocolId": PROTOCOL_ID,
        "protocolSha256": protocol["protocolSha256"], "status": status,
        "mode": mode, "evaluatedAsOfKst": as_of.isoformat(), "truthSource": {
            k: truth[k] for k in ("runId", "runStatusSha256", "snapshotSha256", "pricesSha256",
                                  "provenanceSha256", "provenanceCanonicalSha256", "dataEnd",
                                  "evaluationEnd", "finishedAtKst")},
        "scheduledDays": day_coverage, "executionSummary": _execution_summary(day_coverage),
        "outcomes": pair_rows, "targetStateCounts": dict(sorted(Counter(r["truthState"] for r in pair_rows).items())),
        "comparisonSummaries": by_comparison, "completeCase": _complete_case(pair_rows),
        "thresholds": thresholds, "reviewReasons": reasons, "inference": inference}
    report["reportSha256"] = sha(canonical(report))
    return report


def _is_month_end(day: dt.date) -> bool:
    return (day + dt.timedelta(days=1)).month != day.month


def _eligible_review_date(first_ready: dt.date) -> dt.date:
    """First month end on or after immutable first detection of support readiness."""
    year, month = first_ready.year, first_ready.month
    while True:
        next_day = dt.date(year + (month == 12), 1 if month == 12 else month + 1, 1)
        month_end = next_day - dt.timedelta(days=1)
        if month_end >= first_ready:
            return max(month_end, EARLIEST_REVIEW)
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1


def _load_first_ready(registry_root: Path) -> tuple[dict[str, Any], str]:
    path = registry_root / "readiness" / "first-ready.json"
    raw = path.read_bytes()
    doc = json.loads(raw)
    unsigned = dict(doc)
    digest = unsigned.pop("readinessSha256", None)
    if (not isinstance(doc, dict) or doc.get("schemaVersion") != 1
            or doc.get("protocolId") != PROTOCOL_ID
            or digest != sha(canonical(unsigned))):
        raise ValueError("immutable first-readiness marker is invalid")
    protocol = _load_protocol(registry_root)
    if doc.get("protocolSha256") != protocol["protocolSha256"]:
        raise ValueError("first-readiness marker protocol binding mismatch")
    if doc.get("thresholds", {}).get("allHorizonsReady") is not True:
        raise ValueError("first-readiness marker does not record the frozen support threshold")
    return doc, sha(raw)


def _first_review_window(registry_root: Path, as_of: dt.datetime,
                         current_support_ready: bool) -> dict[str, Any]:
    path = registry_root / "readiness" / "first-ready.json"
    if not path.is_file():
        return {"eligible": False, "reason": "first_readiness_not_recorded"}
    marker, marker_sha = _load_first_ready(registry_root)
    ready_at = _timestamp(marker["firstReadyDetectedAtKst"], "first readiness timestamp")
    if ready_at > as_of:
        return {"eligible": False, "reason": "first_readiness_not_yet_observed_as_of",
                "readinessFileSha256": marker_sha}
    eligible_date = _eligible_review_date(ready_at.date())
    if as_of.date() > eligible_date:
        return {"eligible": False, "reason": "first_eligible_month_end_was_missed",
                "eligibleDateKst": eligible_date.isoformat(), "readinessFileSha256": marker_sha}
    if as_of.date() < eligible_date:
        return {"eligible": False, "reason": "not_first_month_end_after_readiness",
                "eligibleDateKst": eligible_date.isoformat(), "readinessFileSha256": marker_sha}
    if not current_support_ready:
        return {"eligible": False, "reason": "support_threshold_lost_on_first_eligible_month_end",
                "eligibleDateKst": eligible_date.isoformat(), "readinessFileSha256": marker_sha}
    return {"eligible": True, "eligibleDateKst": eligible_date.isoformat(),
            "readinessFileSha256": marker_sha}


def _reserve_review(registry_root: Path, protocol_sha256: str, readiness_sha256: str,
                    truth_source: dict[str, Any], as_of: dt.datetime) -> tuple[Path, dict[str, Any], bool]:
    value = {"schemaVersion": 1, "protocolId": PROTOCOL_ID, "protocolSha256": protocol_sha256,
        "readinessSha256": readiness_sha256, "reservedAtKst": as_of.isoformat(),
        "truthSource": truth_source}
    value["reviewClaimSha256"] = sha(canonical(value))
    path = registry_root / "reviews" / "first-review-claim.json"
    try:
        created = _write_create_only(path, value)
    except FileExistsError:
        raw = path.read_bytes()
        existing = json.loads(raw)
        unsigned = dict(existing)
        digest = unsigned.pop("reviewClaimSha256", None)
        if digest != sha(canonical(unsigned)):
            raise ValueError("existing first-review claim is corrupt")
        return path, existing, False
    if not created:
        existing = json.loads(path.read_bytes())
        return path, existing, False
    return path, value, True


def run_evaluation(registry_root: Path, truth_run_dir: Path, output_root: Path = DEFAULT_OUTPUT,
                   *, mode: str = "daily", as_of: dt.datetime | None = None) -> tuple[Path, dict[str, Any]]:
    if as_of is None:
        as_of = dt.datetime.now(KST)
    if as_of.tzinfo is None or as_of.utcoffset() is None:
        raise ValueError("evaluatedAsOf must be timezone-aware")
    as_of = as_of.astimezone(KST)
    actual_now = dt.datetime.now(KST)
    if as_of > actual_now:
        raise ValueError("evaluatedAsOf cannot be in the future")
    if as_of.date() != actual_now.date():
        raise ValueError("retrospective evaluation timestamps are not allowed")
    if mode == "review" and (actual_now - as_of).total_seconds() > 120:
        raise ValueError("first review must use the current KST timestamp, not a backdated as-of")
    truth = _validate_truth_run(truth_run_dir.resolve(), as_of)
    report = build_report(registry_root, truth, as_of, output_root=output_root, mode=mode)
    # Source-version-specific filenames preserve later truth corrections.
    stamp = as_of.strftime("%H%M%S%f%z")
    daily_path = output_root / "daily" / as_of.date().isoformat() / f"{stamp}-{truth['snapshotSha256']}.json"
    if report["thresholds"]["allHorizonsReady"]:
        readiness = {"schemaVersion": 1, "protocolId": PROTOCOL_ID,
            "protocolSha256": _load_protocol(registry_root)["protocolSha256"],
            "firstReadyDetectedAtKst": as_of.isoformat(), "truthSource": report["truthSource"],
            "thresholds": report["thresholds"]}
        readiness["readinessSha256"] = sha(canonical(readiness))
        readiness_path = registry_root / "readiness" / "first-ready.json"
        if not readiness_path.exists():
            try:
                _write_create_only(readiness_path, readiness)
            except FileExistsError:
                # A concurrent first-ready report wins; it remains immutable.
                existing = json.loads(readiness_path.read_bytes())
                unsigned = dict(existing)
                digest = unsigned.pop("readinessSha256", None)
                if (digest != sha(canonical(unsigned)) or existing.get("protocolId") != PROTOCOL_ID):
                    raise ValueError("existing first-readiness marker is corrupt")
    if mode == "daily":
        _write_create_only(daily_path, report)
        return daily_path, report
    review_path = registry_root / "reviews" / "first-review.json"
    if review_path.exists():
        saved = json.loads(review_path.read_bytes())
        unsigned = dict(saved); digest = unsigned.pop("reportSha256", None)
        if digest != sha(canonical(unsigned)) or saved.get("protocolId") != PROTOCOL_ID:
            raise ValueError("existing first-review report is corrupt")
        return review_path, saved
    protocol = _load_protocol(registry_root)
    window = _first_review_window(registry_root, as_of, report["thresholds"]["allHorizonsReady"])
    report["firstReviewWindow"] = window
    if not window["eligible"]:
        report["reviewReasons"].append({"code": window["reason"],
            "eligibleDateKst": window.get("eligibleDateKst")})
    else:
        review_claim_path, review_claim, matching_claim = _reserve_review(
            registry_root, protocol["protocolSha256"], window["readinessFileSha256"],
            report["truthSource"], as_of)
        if not matching_claim:
            report["reviewReasons"].append({"code": "first_review_already_reserved",
                "reservedAtKst": review_claim.get("reservedAtKst"),
                "truthSource": review_claim.get("truthSource")})
        else:
            report["status"] = "review_complete"
            report["reviewClaimSha256"] = review_claim.get("reviewClaimSha256")
            report["reviewClaimFileSha256"] = sha(review_claim_path.read_bytes())
            report["readinessFileSha256"] = window["readinessFileSha256"]
            report["inference"] = _review_inference(report["outcomes"], truth)
            report.pop("reportSha256", None)
            report["reportSha256"] = sha(canonical(report))
            _write_create_only(daily_path, report)
            _write_create_only(review_path, report)
            return review_path, report
    report["status"] = "not_ready"
    report["inference"] = {"status": "not_run", "reason": "review eligibility or immutable reservation failed"}
    report.pop("reportSha256", None)
    report["reportSha256"] = sha(canonical(report))
    _write_create_only(daily_path, report)
    return daily_path, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    bootstrap = subs.add_parser("bootstrap", help="pin the first normal Oct 11 issue (no fit)")
    bootstrap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    evaluate = subs.add_parser("evaluate", help="record daily coverage or the one explicit month-end review")
    evaluate.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    evaluate.add_argument("--truth-run-dir", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    evaluate.add_argument("--mode", choices=("daily", "review"), default="daily")
    args = parser.parse_args(argv)
    if args.command == "bootstrap":
        result = bootstrap_registry(args.registry)
        print(json.dumps({k: str(v) for k, v in result.items()}, ensure_ascii=False))
        return 0
    path, report = run_evaluation(args.registry, args.truth_run_dir, args.output_dir, mode=args.mode)
    print(json.dumps({"result": str(path), "status": report["status"],
                      "evaluatedAsOfKst": report["evaluatedAsOfKst"],
                      "garlicSupport": report["thresholds"]["byGarlicHorizon"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
