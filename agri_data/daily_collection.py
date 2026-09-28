"""Bounded, resumable daily wrapper around the public-price collector."""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import io
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import shutil
import sqlite3
import sys
from typing import Any, Callable
from zoneinfo import ZoneInfo

from . import collector

KST = ZoneInfo("Asia/Seoul")
UTC = dt.timezone.utc
DEFAULT_START = dt.date(2016, 1, 1)
DEFAULT_CUTOFF = dt.date(2026, 9, 27)
DAILY_HTTP_LIMIT = 9000
EXECUTION_CHUNK = 250
DISK_FLOOR_BYTES = 5 * 1024**3
MARKETS = ("110001", "110008")


class DailyRunError(RuntimeError):
    pass


def year_ranges(start: dt.date = DEFAULT_START, cutoff: dt.date = DEFAULT_CUTOFF) -> list[tuple[dt.date, dt.date]]:
    if start > cutoff:
        return []
    result: list[tuple[dt.date, dt.date]] = []
    cursor = start
    while cursor <= cutoff:
        stop = min(cutoff, dt.date(cursor.year, 12, 31))
        result.append((cursor, stop))
        cursor = stop + dt.timedelta(days=1)
    return result


def _arguments(db_path: str, env_file: str, start: dt.date, end: dt.date,
               max_requests: int | None = None) -> argparse.Namespace:
    argv = ["prices", "--db", db_path, "--env-file", env_file,
            "--start", start.isoformat(), "--end", end.isoformat(),
            "--rate", "2.0"]
    for market in MARKETS:
        argv.extend(("--market", market))
    if max_requests is not None:
        argv.extend(("--max-requests", str(max_requests)))
    return collector.make_parser().parse_args(argv)


def _canonical_job_config(args: argparse.Namespace) -> str:
    config = {key: value.isoformat() if isinstance(value, dt.date) else value
              for key, value in vars(args).items() if key != "env_file" and not key.startswith("_")}
    config = {key: value for key, value in config.items() if key not in ("max_requests", "refresh")}
    return collector.json_text(config)


def _job_state(db: sqlite3.Connection, config: str) -> str:
    rows = db.execute("SELECT state FROM jobs WHERE command='prices' AND config_json=? ORDER BY job_id DESC", (config,)).fetchall()
    if any(row[0] == "complete" for row in rows):
        return "complete"
    if any(row[0] in ("incomplete", "interrupted", "running") for row in rows):
        return "resume"
    if any(row[0] == "failed" for row in rows):
        return "manual_review"
    return "pending"


def _utc_bounds(day: dt.date) -> tuple[str, str]:
    start = dt.datetime.combine(day, dt.time.min, KST).astimezone(UTC).isoformat()
    end = dt.datetime.combine(day + dt.timedelta(days=1), dt.time.min, KST).astimezone(UTC).isoformat()
    return start, end


def _today_attempts(db: sqlite3.Connection, day: dt.date) -> tuple[int, int, int, int]:
    start, end = _utc_bounds(day)
    count, limited, api_errors, other_errors = db.execute(
        """SELECT COUNT(*),
                  SUM(CASE WHEN http_status=429 THEN 1 ELSE 0 END),
                  SUM(CASE WHEN error_type='api' OR http_status IN (401,403) THEN 1 ELSE 0 END),
                  SUM(CASE WHEN error_type IS NOT NULL THEN 1 ELSE 0 END)
           FROM request_attempts WHERE started_at>=? AND started_at<?""", (start, end)
    ).fetchone()
    return int(count or 0), int(limited or 0), int(api_errors or 0), int(other_errors or 0)


def _attempts_for_path(path: Path, day: dt.date) -> tuple[int, int, int, int]:
    db = collector.open_db(str(path), readonly=True)
    try:
        return _today_attempts(db, day)
    finally:
        db.close()


def _load_status(path: Path) -> dict[str, Any]:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        raise DailyRunError(f"status file cannot be read safely: {path.name}") from exc
    if not isinstance(result, dict):
        raise DailyRunError(f"status file has invalid shape: {path.name}")
    return result


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temp.unlink()


def _logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"agri.daily.{log_path.resolve()}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.addHandler(RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024,
                                          backupCount=1, encoding="utf-8"))
    return logger


def _disk_free(path: Path) -> int:
    target = path if path.exists() else path.parent
    return shutil.disk_usage(target).free


def _plan(db: sqlite3.Connection, db_path: str, env_file: str,
          ranges: list[tuple[dt.date, dt.date]]) -> list[dict[str, Any]]:
    plan = []
    for start, end in ranges:
        args = _arguments(db_path, env_file, start, end)
        state = _job_state(db, _canonical_job_config(args))
        plan.append({"start": start.isoformat(), "end": end.isoformat(), "state": state})
    return plan


def _failure_gate(db: sqlite3.Connection, day: dt.date, status: dict[str, Any]) -> tuple[str, str] | None:
    if status.get("date") == day.isoformat() and status.get("stopKind") in {
        "collector_error", "rate_limit", "auth_failure", "repeated_errors", "disk_low", "interrupted", "day_rollover"
    }:
        return status["stopKind"], "today is already stopped after an earlier error; resume tomorrow"
    if status.get("date") == day.isoformat() and status.get("inflightReservation", 0):
        return "interrupted", "previous process stopped during a reserved HTTP chunk; resume tomorrow"
    count, rate_limits, api_errors, other_errors = _today_attempts(db, day)
    if rate_limits:
        return "rate_limit", "HTTP 429 was recorded today; collection is paused until tomorrow"
    if api_errors:
        return "auth_failure", "API or authentication error was recorded today; collection is paused until tomorrow"
    if other_errors >= 3:
        return "repeated_errors", "repeated HTTP/transport errors were recorded today; collection is paused until tomorrow"
    return None


def run_daily(*, db_path: str = "data/agri.sqlite", env_file: str,
              root: str | Path = ".", dry_run: bool = False,
              today: dt.date | None = None, daily_limit: int = DAILY_HTTP_LIMIT,
              chunk_size: int = EXECUTION_CHUNK, disk_floor: int = DISK_FLOOR_BYTES,
              execute: Callable[[argparse.Namespace], int] | None = None,
              disk_free: Callable[[Path], int] = _disk_free,
              kst_today: Callable[[], dt.date] | None = None) -> dict[str, Any]:
    """Run in calendar-year order. All HTTP work is delegated to collector.execute."""
    project = Path(root).resolve()
    db_file = Path(db_path)
    collector_db_path = os.fspath(db_path) if not db_file.is_absolute() else str(db_file)
    if not db_file.is_absolute():
        db_file = project / db_file
    db_file = db_file.resolve()
    env_path = str(Path(env_file).expanduser().resolve())
    run_dir = project / "runs"
    status_path = run_dir / "daily-collection-status.json"
    actual_kst_today = kst_today or (lambda: dt.datetime.now(KST).date())
    day = today or actual_kst_today()
    ranges = [(dt.date(2016, 1, 1), dt.date(2016, 3, 31)),
              (dt.date(2016, 4, 1), dt.date(2016, 12, 31)),
              *year_ranges(dt.date(2017, 1, 1))]
    if not db_file.is_file():
        raise DailyRunError(f"database does not exist: {db_file}")
    db = collector.open_db(str(db_file), readonly=True)
    try:
        jobs = _plan(db, collector_db_path, env_path, ranges)
        attempt_count, rate_limits, api_errors, error_attempts = _today_attempts(db, day)
        remaining = max(0, daily_limit - attempt_count)
        status = _load_status(status_path)
        result: dict[str, Any] = {
            "date": day.isoformat(), "cutoff": DEFAULT_CUTOFF.isoformat(),
            "dailyHttpLimit": daily_limit, "httpAttemptsToday": attempt_count,
            "remainingHttpBudget": remaining, "ranges": jobs,
            "diskFreeBytes": disk_free(db_file.parent), "status": "planned",
        }
        if dry_run:
            result["alreadyProcessedYear"] = status.get("yearWorked") if status.get("date") == day.isoformat() else None
            result["status"] = "dry_run"
            return result
        if chunk_size <= 0 or daily_limit <= 0:
            raise DailyRunError("daily limit and execution chunk must be positive")
    finally:
        db.close()

    lock_path = run_dir / "daily-collection.lock"
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_stream = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise DailyRunError("another daily collection runner holds the lock") from exc
        # Re-read the same-day gate after taking the lock to close the planning race.
        db = collector.open_db(str(db_file), readonly=True)
        try:
            latest_status = _load_status(status_path)
            if disk_free(db_file.parent) < disk_floor:
                result.update(status="stopped", stopKind="disk_low")
                _atomic_json(status_path, result)
                return result
            blocked = _failure_gate(db, day, latest_status)
            if blocked:
                result.update(status="stopped", stopKind=blocked[0], message=blocked[1])
                _atomic_json(status_path, result)
                return result
            if latest_status.get("date") == day.isoformat() and latest_status.get("stopKind") == "budget_exhausted":
                result.update(status="budget_exhausted", stopKind="budget_exhausted")
                return result
            if latest_status.get("date") == day.isoformat() and latest_status.get("yearWorked") is not None:
                result.update(status="already_processed_today", yearWorked=latest_status["yearWorked"])
                return result
            count, _, _, _ = _today_attempts(db, day)
            if count >= daily_limit or (latest_status.get("date") == day.isoformat()
                                        and latest_status.get("stopKind") == "budget_exhausted"):
                result.update(status="budget_exhausted", stopKind="budget_exhausted",
                              httpAttemptsToday=count, remainingHttpBudget=max(0, daily_limit-count))
                _atomic_json(status_path, result)
                return result
        finally:
            db.close()

        logger = _logger(run_dir / "daily-collection.log")
        execute_fn = execute or collector.execute
        result["status"] = "running"
        result["startedAtKst"] = dt.datetime.now(KST).isoformat()
        _atomic_json(status_path, result)
        unfinished = [item for item in jobs if item["state"] != "complete"]
        if not unfinished:
            result.update(status="complete")
        else:
            year = min(int(item["start"][:4]) for item in unfinished)
            result["yearWorked"] = year
            selected = [item for item in unfinished if int(item["start"][:4]) == year]
            for item in selected:
                if item["state"] == "manual_review":
                    result.update(status="stopped", stopKind="collector_error",
                                  message="an earlier failed year job requires inspection")
                    break
                start, end = dt.date.fromisoformat(item["start"]), dt.date.fromisoformat(item["end"])
                logger.info("year-range start=%s end=%s state=%s", start, end, item["state"])
                while True:
                    db = collector.open_db(str(db_file), readonly=True)
                    try:
                        used, rl, auth, errs = _today_attempts(db, day)
                        if rl or auth or errs >= 3:
                            result.update(status="stopped", stopKind="rate_limit" if rl else
                                          "auth_failure" if auth else "repeated_errors",
                                          httpAttemptsToday=used)
                            break
                        job_state = _job_state(db, _canonical_job_config(_arguments(
                            collector_db_path, env_path, start, end)))
                    finally:
                        db.close()
                    if job_state == "complete":
                        item["state"] = "complete"
                        break
                    if job_state == "manual_review":
                        result.update(status="stopped", stopKind="collector_error",
                                      message="year range has failed collector history")
                        break
                    used, _, _, _ = _attempts_for_path(db_file, day)
                    remaining = max(0, daily_limit - used)
                    if remaining <= 0:
                        result.update(status="budget_exhausted", stopKind="budget_exhausted",
                                      httpAttemptsToday=used, remainingHttpBudget=0)
                        break
                    free = disk_free(db_file.parent)
                    if free < disk_floor:
                        result.update(status="stopped", stopKind="disk_low", diskFreeBytes=free,
                                      httpAttemptsToday=used)
                        break
                    allowance = min(chunk_size, remaining)
                    # Reserve before entering the collector: a killed process cannot spend this chunk twice.
                    reservation = {"date": day.isoformat(), "inflightReservation": allowance,
                                   "yearWorked": year, "status": "running",
                                   "activeRange": [start.isoformat(), end.isoformat()]}
                    _atomic_json(status_path, {**result, **reservation})
                    args = _arguments(collector_db_path, env_path, start, end, allowance)
                    output = io.StringIO()
                    guard_failure: list[str] = []
                    original_budget = collector.RequestBudget

                    class GuardedRequestBudget(original_budget):
                        def reserve(self):
                            if actual_kst_today() != day:
                                guard_failure.append("day_rollover")
                                raise collector.BudgetExhausted("KST day boundary reached; resume next scheduled run")
                            if disk_free(db_file.parent) < disk_floor:
                                guard_failure.append("disk_low")
                                raise collector.BudgetExhausted("disk safety floor reached; resume next scheduled run")
                            return super().reserve()

                    try:
                        collector.RequestBudget = GuardedRequestBudget
                        try:
                            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                                code = execute_fn(args)
                        except Exception as exc:
                            code = 2
                            output.write(f"collector exception: {type(exc).__name__}")
                    finally:
                        collector.RequestBudget = original_budget
                    secrets = [v for v in collector.load_env(env_path).values() if v]
                    message = collector.redact(output.getvalue(), secrets)
                    db = collector.open_db(str(db_file), readonly=True)
                    try:
                        after, rl, auth, errs = _today_attempts(db, day)
                        current_state = _job_state(db, _canonical_job_config(args))
                    finally:
                        db.close()
                    spent = after - used
                    result.update(httpAttemptsToday=after, remainingHttpBudget=max(0, daily_limit-after),
                                  diskFreeBytes=disk_free(db_file.parent), inflightReservation=0,
                                  yearWorked=year)
                    _atomic_json(status_path, result)
                    logger.info("chunk done range=%s..%s attempts=%s exit=%s state=%s",
                                start, end, spent, code, current_state)
                    if rl:
                        result.update(status="stopped", stopKind="rate_limit")
                        break
                    if auth:
                        result.update(status="stopped", stopKind="auth_failure")
                        break
                    if guard_failure:
                        result.update(status="stopped", stopKind=guard_failure[0])
                        break
                    if errs >= 3:
                        result.update(status="stopped", stopKind="repeated_errors")
                        break
                    if code == 0 and current_state == "complete":
                        item["state"] = "complete"
                        break
                    expected_budget = "request budget exhausted" in message.lower()
                    if code != 0 and not expected_budget:
                        result.update(status="stopped", stopKind="collector_error")
                        logger.error("collector stopped range=%s..%s output=%s", start, end,
                                     message[-500:].replace("\n", " "))
                        break
                    if result["remainingHttpBudget"] == 0:
                        result.update(status="budget_exhausted", stopKind="budget_exhausted")
                        break
                    if spent <= 0:
                        result.update(status="stopped", stopKind="collector_error")
                        logger.error("collector made no recorded HTTP progress range=%s..%s", start, end)
                        break
                if result["status"] in ("stopped", "budget_exhausted"):
                    break
            if result["status"] == "running":
                result.update(status="year_complete")
        result["finishedAtKst"] = dt.datetime.now(KST).isoformat()
        _atomic_json(status_path, result)
        logger.info("run finished status=%s attempts=%s", result["status"], result.get("httpAttemptsToday"))
        return result
    except KeyboardInterrupt:
        result.update(status="stopped", stopKind="interrupted", inflightReservation=chunk_size)
        _atomic_json(status_path, result)
        raise
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(lock_stream.fileno(), fcntl.LOCK_UN)
        lock_stream.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Safely resume yearly public aT price collection.")
    parser.add_argument("--db", default="data/agri.sqlite")
    parser.add_argument("--env-file", required=True, help="explicit approved .env file; key values are never logged")
    parser.add_argument("--dry-run", action="store_true", help="read-only plan; no API calls or local state changes")
    parser.add_argument("--daily-limit", type=int, default=DAILY_HTTP_LIMIT)
    parser.add_argument("--chunk-size", type=int, default=EXECUTION_CHUNK)
    parser.add_argument("--disk-floor-bytes", type=int, default=DISK_FLOOR_BYTES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_daily(db_path=args.db, env_file=args.env_file, root=Path.cwd(),
                           dry_run=args.dry_run, daily_limit=args.daily_limit,
                           chunk_size=args.chunk_size, disk_floor=args.disk_floor_bytes)
    except DailyRunError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] in ("dry_run", "complete", "year_complete", "already_processed_today",
                                      "budget_exhausted", "pending") else 2


if __name__ == "__main__":
    raise SystemExit(main())
