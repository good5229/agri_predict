"""Publish an already completed daily model/significance pair atomically enough for readers."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid
from typing import Any
from zoneinfo import ZoneInfo

from . import dashboard, modeling, significance

KST = ZoneInfo("Asia/Seoul")
PRODUCT_IDS = tuple(significance.PRODUCT_IDS)
EXPECTED_COMPARISON_KEYS = {
    (product, model, cadence, horizon, loss)
    for product in significance.PRODUCT_IDS
    for model in significance.MODEL_IDS
    for cadence in significance.CADENCES
    for horizon in significance.HORIZONS
    for loss in significance.LOSSES
}
EXPECTED_CONCLUSION_KEYS = {
    (product, model, cadence, horizon)
    for product in significance.PRODUCT_IDS
    for model in significance.MODEL_IDS
    for cadence in significance.CADENCES
    for horizon in significance.HORIZONS
}


class PublishError(RuntimeError):
    pass


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublishError(f"invalid JSON: {path.name}") from exc
    if not isinstance(value, dict):
        raise PublishError(f"expected JSON object: {path.name}")
    return value, raw


def _validate_significance(report: dict[str, Any], source: dict[str, Any], source_bytes: bytes,
                           significance_bytes: bytes) -> tuple[dt.date, dt.date]:
    try:
        _, _, eval_start, eval_end = significance._validate_input(source)
    except (TypeError, ValueError, KeyError) as exc:
        raise PublishError(f"invalid prediction export: {exc}") from exc
    protocol = report.get("protocol")
    if report.get("schemaVersion") != 1 or not isinstance(protocol, dict):
        raise PublishError("unsupported significance report schema")
    expected_sha = _sha256(source_bytes)
    if protocol.get("inputSha256") != expected_sha:
        raise PublishError("significance report does not match exact prediction bytes")
    if protocol.get("evaluationStart") != eval_start.isoformat() or protocol.get("evaluationEnd") != eval_end.isoformat():
        raise PublishError("significance period does not match prediction period")
    primary, sensitivity = protocol.get("primary"), protocol.get("sensitivity")
    if not isinstance(primary, dict) or not isinstance(sensitivity, dict):
        raise PublishError("significance report is missing primary/sensitivity protocol")
    for item, expected_lag in ((primary, significance.PRIMARY_LAG_DAYS),
                               (sensitivity, significance.SENSITIVITY_LAG_DAYS)):
        if (item.get("holmFamilySize") != significance.FAMILY_SIZE
                or item.get("alpha") != significance.ALPHA
                or item.get("maxLagDays") != expected_lag
                or item.get("minimumPairs") != significance.MIN_PAIRS):
            raise PublishError("significance protocol does not match the fixed inference design")
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, list) or len(comparisons) != significance.FAMILY_SIZE:
        raise PublishError("significance report must contain exactly 800 comparisons")
    comparison_keys = []
    for row in comparisons:
        if not isinstance(row, dict):
            raise PublishError("invalid significance comparison row")
        for family in ("primary", "sensitivity"):
            summary = row.get(family)
            if (not isinstance(summary, dict) or not isinstance(summary.get("test"), dict)
                    or not isinstance(summary.get("n"), int)):
                raise PublishError("significance comparison has an invalid paired-test shape")
        comparison_keys.append((row.get("productId"), row.get("modelId"), row.get("cadence"),
                                row.get("horizon"), row.get("loss")))
    if len(set(comparison_keys)) != len(comparison_keys) or set(comparison_keys) != EXPECTED_COMPARISON_KEYS:
        raise PublishError("significance comparison keys are incomplete or duplicated")
    conclusions = report.get("conclusions")
    if not isinstance(conclusions, list) or len(conclusions) != len(EXPECTED_CONCLUSION_KEYS):
        raise PublishError("significance report must contain exactly 400 conclusions")
    conclusion_keys = []
    for row in conclusions:
        if not isinstance(row, dict):
            raise PublishError("invalid significance conclusion row")
        if row.get("status") not in {"improved", "worsened", "mixed_evidence", "no_evidence", "unavailable"}:
            raise PublishError("invalid significance conclusion status")
        conclusion_keys.append((row.get("productId"), row.get("modelId"), row.get("cadence"), row.get("horizon")))
    if len(set(conclusion_keys)) != len(conclusion_keys) or set(conclusion_keys) != EXPECTED_CONCLUSION_KEYS:
        raise PublishError("significance conclusion keys are incomplete or duplicated")
    return eval_start, eval_end


def _safe_run_dir(project: Path, run_directory: str | Path) -> Path:
    candidate = Path(run_directory)
    if not candidate.is_absolute():
        candidate = project / candidate
    resolved = candidate.resolve()
    daily_root = (project / "outputs" / "daily").resolve()
    if daily_root not in resolved.parents:
        raise PublishError("run directory must be inside outputs/daily")
    return resolved


def _resolve_run(project: Path, run_directory: str | Path | None) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if run_directory is None:
        state, _ = _read_json_bytes(project / "runs" / "daily-training-status.json")
        latest = state.get("latestSuccess")
        if not isinstance(latest, dict) or not isinstance(latest.get("runDirectory"), str):
            raise PublishError("there is no successful daily run to publish")
        run_dir = _safe_run_dir(project, latest["runDirectory"])
    else:
        run_dir = _safe_run_dir(project, run_directory)
        latest = {}
    status, _ = _read_json_bytes(run_dir / "run-status.json")
    if status.get("status") != "complete" or not status.get("runId") or not status.get("cutoff"):
        raise PublishError("selected archive is not a completed daily run")
    if latest and (status.get("runId") != latest.get("runId")
                   or status.get("cutoff") != latest.get("cutoff")
                   or status.get("snapshotSha256") != latest.get("snapshotSha256")):
        raise PublishError("latest-success pointer does not match archived run status")
    if status.get("runDirectory") != str(run_dir.relative_to(project)):
        raise PublishError("run-status directory does not match selected archive")
    return run_dir, status, latest


def _atomic_replace(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise


def _backup_pair(project: Path, existing: dict[Path, bytes | None], run_id: str) -> Path | None:
    if not any(payload is not None for payload in existing.values()):
        return None
    stamp = dt.datetime.now(KST).strftime("%Y%m%dT%H%M%S")
    backup = project / "outputs" / "dashboard-publication-backups" / f"{stamp}-{run_id}-{uuid.uuid4().hex[:6]}"
    backup.mkdir(parents=True, exist_ok=False)
    manifest = {"runId": run_id, "createdAtKst": dt.datetime.now(KST).isoformat(), "files": {}}
    for target, payload in existing.items():
        name = target.name
        if payload is None:
            manifest["files"][name] = None
            continue
        backup_file = backup / name
        with backup_file.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        manifest["files"][name] = {"sha256": _sha256(payload), "backup": name}
    with (backup / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return backup


def _restore_pair(existing: dict[Path, bytes | None]) -> list[str]:
    failures = []
    for target, payload in existing.items():
        try:
            if payload is None:
                with contextlib.suppress(FileNotFoundError):
                    target.unlink()
            else:
                _atomic_replace(target, payload)
        except OSError as exc:
            failures.append(f"{target.name}:{type(exc).__name__}")
    return failures


def _write_status(project: Path, status: dict[str, Any]) -> None:
    path = project / "runs" / "dashboard-publication-status.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(status, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise


def publish(*, root: str | Path = ".", run_directory: str | Path | None = None,
            dry_run: bool = False, require_today: bool = False,
            today: dt.date | None = None) -> dict[str, Any]:
    project = Path(root).resolve()
    lock_path = project / "runs" / "daily-training.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PublishError("daily training is active; publication deferred") from exc
        attempt: dict[str, Any] | None = None
        try:
            run_dir, run_status, _ = _resolve_run(project, run_directory)
            expected_day = today or dt.datetime.now(KST).date()
            archive_day = dt.date.fromisoformat(str(run_status.get("finishedAtKst", run_status.get("cutoff")))[:10])
            if require_today and archive_day != expected_day:
                raise PublishError(f"archive date {archive_day.isoformat()} is not today's {expected_day.isoformat()}")
            archive_relative = run_dir.relative_to(project / "outputs" / "daily")
            if require_today and (not archive_relative.parts or archive_relative.parts[0] != expected_day.isoformat()):
                raise PublishError("selected archive directory is not under today's archive folder")
            predictions_path = run_dir / "predictions.json"
            significance_path = run_dir / "significance-full.json"
            source, source_bytes = _read_json_bytes(predictions_path)
            report, report_bytes = _read_json_bytes(significance_path)
            eval_start, eval_end = _validate_significance(report, source, source_bytes, report_bytes)
            if eval_end.isoformat() != run_status.get("cutoff"):
                raise PublishError("prediction evaluation end does not match completed run cutoff")
            if run_status.get("predictions") != str(predictions_path.relative_to(project)) or \
                    run_status.get("fullSignificance") != str(significance_path.relative_to(project)):
                raise PublishError("run-status artifact paths do not match archived files")
            targets = {project / "outputs" / "dashboard.json": source_bytes,
                       project / "outputs" / "significance.json": report_bytes}
            hashes = {path.name: _sha256(payload) for path, payload in targets.items()}
            attempt = {"runId": run_status["runId"], "runDirectory": str(run_dir.relative_to(project)),
                       "status": "validated", "evaluationStart": eval_start.isoformat(),
                       "evaluationEnd": eval_end.isoformat(), "files": hashes,
                       "validatedAtKst": dt.datetime.now(KST).isoformat()}
            if dry_run:
                return {**attempt, "status": "dry_run"}
            existing = {path: path.read_bytes() if path.exists() else None for path in targets}
            if all(existing[path] == payload for path, payload in targets.items()):
                attempt.update({"status": "no_change", "publishedAtKst": dt.datetime.now(KST).isoformat(),
                                "backupDirectory": None})
                _write_status(project, attempt)
                return attempt
            attempt["backupDirectory"] = None
            changed = False
            try:
                backup = _backup_pair(project, existing, run_status["runId"])
                attempt["backupDirectory"] = str(backup.relative_to(project)) if backup else None
                # Significance is replaced first. The serving layer validates the byte hash before loading either file.
                _atomic_replace(project / "outputs" / "significance.json", report_bytes)
                changed = True
                _atomic_replace(project / "outputs" / "dashboard.json", source_bytes)
                attempt.update({"status": "published", "publishedAtKst": dt.datetime.now(KST).isoformat()})
                _write_status(project, attempt)
                return attempt
            except Exception as exc:
                rollback_failures = _restore_pair(existing) if changed else []
                attempt.update({"status": "failed", "error": type(exc).__name__,
                                "rollbackFailures": rollback_failures,
                                "failedAtKst": dt.datetime.now(KST).isoformat()})
                with contextlib.suppress(OSError):
                    _write_status(project, attempt)
                raise PublishError(f"dashboard publication failed; rollback failures={rollback_failures}") from exc
        except Exception as exc:
            if attempt is None and not dry_run:
                failure = {"status": "failed", "error": str(exc), "failedAtKst": dt.datetime.now(KST).isoformat()}
                _write_status(project, failure)
            raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--run-directory")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-today", action="store_true",
                        help="reject an archived run whose finishedAtKst is not today")
    args = parser.parse_args(argv)
    try:
        result = publish(root=args.root, run_directory=args.run_directory, dry_run=args.dry_run,
                         require_today=args.require_today)
    except (PublishError, OSError, ValueError) as exc:
        print(f"dashboard publication failed: {exc}", file=__import__("sys").stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
