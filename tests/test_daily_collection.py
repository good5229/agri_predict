import contextlib
import datetime as dt
import fcntl
import io
import io
import json
import tempfile
import unittest
from pathlib import Path

from agri_data import collector
from agri_data import daily_collection as daily


TEST_DAY = dt.date(2026, 9, 28)


class DailyCollectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "data").mkdir()
        (self.root / "test.env").write_text("DATA_GO_KR_SERVICE_KEY_DECODED=TEST_SECRET\n", encoding="utf-8")
        self.db_path = self.root / "data" / "agri.sqlite"
        db = collector.open_db(str(self.db_path))
        db.close()

    def tearDown(self):
        self.tmp.cleanup()

    def create_job(self, start=dt.date(2016, 1, 1), end=dt.date(2016, 3, 31), state="incomplete"):
        args = daily._arguments("data/agri.sqlite", str(self.root / "test.env"), start, end)
        db = collector.open_db(str(self.db_path))
        config = json.loads(daily._canonical_job_config(args))
        job = collector.begin_job(db, "prices", config)
        collector.finish_job(db, job, state)
        db.close()
        return job, args

    def add_attempt(self, job, when="2026-09-28T00:10:00+00:00", status=200, error=None):
        db = collector.open_db(str(self.db_path))
        db.execute("""INSERT INTO request_attempts(job_id,request_key,epoch_id,attempt_no,started_at,finished_at,
                    http_status,error_type,error_message) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (job, "test", "e", 1, when, when, status, error, "redacted" if error else None))
        db.commit()
        db.close()

    def invoke(self, **kwargs):
        options = {"db_path": "data/agri.sqlite", "env_file": str(self.root / "test.env"),
                   "root": self.root, "today": TEST_DAY, "disk_floor": 5,
                   "disk_free": lambda _: 100}
        options.update(kwargs)
        options.setdefault("kst_today", lambda: options["today"])
        return daily.run_daily(**options)

    def test_schedule_keeps_partial_2016_and_fixed_cutoff(self):
        ranges = [(dt.date(2016, 1, 1), dt.date(2016, 3, 31)),
                  (dt.date(2016, 4, 1), dt.date(2016, 12, 31)),
                  *daily.year_ranges(dt.date(2017, 1, 1))]
        self.assertEqual(ranges[0], (dt.date(2016, 1, 1), dt.date(2016, 3, 31)))
        self.assertEqual(ranges[1], (dt.date(2016, 4, 1), dt.date(2016, 12, 31)))
        self.assertEqual(ranges[-1], (dt.date(2026, 1, 1), dt.date(2026, 9, 27)))
        self.assertEqual(daily.year_ranges(dt.date(2025, 12, 30), dt.date(2026, 1, 2)), [
            (dt.date(2025, 12, 30), dt.date(2025, 12, 31)),
            (dt.date(2026, 1, 1), dt.date(2026, 1, 2))])

    def test_dry_run_is_readonly_and_shows_resume_budget(self):
        job, _ = self.create_job()
        self.add_attempt(job, status=200)
        result = self.invoke(dry_run=True, execute=lambda _: self.fail("must not execute"), daily_limit=4)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["httpAttemptsToday"], 1)
        self.assertEqual(result["remainingHttpBudget"], 3)
        self.assertEqual(result["ranges"][0]["state"], "resume")
        self.assertFalse((self.root / "runs").exists())

    def test_one_calendar_year_per_day_and_same_day_noop(self):
        calls = []

        def complete(args):
            calls.append((args.start, args.end))
            # collector.execute attaches its runtime RequestBudget to this Namespace.
            args._budget = collector.RequestBudget(args.max_requests)
            db = collector.open_db(str(self.db_path))
            config = json.loads(daily._canonical_job_config(args))
            job = collector.begin_job(db, "prices", config)
            collector.finish_job(db, job, "complete")
            db.close()
            print("done TEST_SECRET")
            return 0

        first = self.invoke(execute=complete, daily_limit=100, chunk_size=2)
        self.assertEqual(first["status"], "year_complete")
        self.assertEqual(first["yearWorked"], 2016)
        self.assertEqual(len(calls), 2)  # 2016 Q1 and its tail; no 2017 request.
        again = self.invoke(execute=complete, daily_limit=100, chunk_size=2)
        self.assertEqual(again["status"], "already_processed_today")
        self.assertEqual(len(calls), 2)
        next_day = self.invoke(execute=complete, daily_limit=100, chunk_size=2,
                            today=TEST_DAY + dt.timedelta(days=1))
        self.assertEqual(next_day["status"], "year_complete")
        self.assertEqual(next_day["yearWorked"], 2017)
        self.assertEqual(len(calls), 3)
        status_text = (self.root / "runs" / "daily-collection-status.json").read_text(encoding="utf-8")
        log_text = (self.root / "runs" / "daily-collection.log").read_text(encoding="utf-8")
        self.assertNotIn("TEST_SECRET", status_text + log_text)

    def test_remaining_budget_is_reserved_exactly_and_chunk_stop_is_resumable(self):
        job, _ = self.create_job()
        calls = []

        def stop_at_budget(args):
            calls.append(args.max_requests)
            for _ in range(args.max_requests):
                self.add_attempt(job, status=200)
            print("job incomplete: request budget exhausted; resume to continue")
            return 2

        result = self.invoke(execute=stop_at_budget, daily_limit=2, chunk_size=250)
        self.assertEqual(calls, [2])
        self.assertEqual(result["status"], "budget_exhausted")
        second = self.invoke(execute=lambda _: self.fail("daily cap already spent"), daily_limit=2)
        self.assertEqual(second["status"], "budget_exhausted")

    def test_http_429_blocks_retries_for_same_day(self):
        job, _ = self.create_job()
        self.add_attempt(job, status=429, error="http")
        calls = []
        result = self.invoke(execute=lambda args: calls.append(args) or 0)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["stopKind"], "rate_limit")
        self.assertFalse(calls)
        retry = self.invoke(execute=lambda args: calls.append(args) or 0)
        self.assertEqual(retry["stopKind"], "rate_limit")
        self.assertFalse(calls)

    def test_disk_floor_and_existing_runner_lock_stop_before_collector(self):
        calls = []
        low_disk = daily.run_daily(db_path="data/agri.sqlite", env_file=str(self.root / "test.env"),
                                   root=self.root, today=TEST_DAY, disk_floor=5,
                                   disk_free=lambda _: 4, execute=lambda args: calls.append(args))
        self.assertEqual(low_disk["stopKind"], "disk_low")
        self.assertFalse(calls)
        lock = self.root / "runs" / "daily-collection.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        with lock.open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            old_status = (self.root / "runs" / "daily-collection-status.json").read_text(encoding="utf-8")
            with self.assertRaisesRegex(daily.DailyRunError, "holds the lock"):
                self.invoke(execute=lambda args: calls.append(args))
            self.assertEqual((self.root / "runs" / "daily-collection-status.json").read_text(encoding="utf-8"), old_status)
        self.assertFalse(calls)

    def test_nonbudget_collector_error_is_not_retried_same_day(self):
        self.create_job()
        calls = []

        def fail(args):
            calls.append(args)
            print("collector failed with redacted error")
            return 2

        result = self.invoke(execute=fail)
        self.assertEqual(result["stopKind"], "collector_error")
        again = self.invoke(execute=lambda args: calls.append(args) or 0)
        self.assertEqual(again["stopKind"], "collector_error")
        self.assertEqual(len(calls), 1)

    def test_corrupt_status_fails_closed_without_overwrite(self):
        run_dir = self.root / "runs"
        run_dir.mkdir()
        status = run_dir / "daily-collection-status.json"
        status.write_text("{broken", encoding="utf-8")
        with self.assertRaisesRegex(daily.DailyRunError, "status file cannot be read safely"):
            self.invoke(dry_run=True)
        self.assertEqual(status.read_text(encoding="utf-8"), "{broken")

    def test_budget_guard_checks_disk_and_kst_at_each_reservation(self):
        job, _ = self.create_job()
        budget_clock = [TEST_DAY]
        free = [100]

        def execute_disk(args):
            budget = collector.RequestBudget(args.max_requests)
            budget.reserve()
            self.add_attempt(job)
            free[0] = 4
            with self.assertRaisesRegex(collector.BudgetExhausted, "disk safety floor"):
                budget.reserve()
            print("disk safety floor reached; resume next scheduled run")
            return 2

        disk_result = daily.run_daily(db_path="data/agri.sqlite", env_file=str(self.root / "test.env"),
                                      root=self.root, today=TEST_DAY, disk_floor=5,
                                      disk_free=lambda _: free[0], execute=execute_disk, chunk_size=2,
                                      kst_today=lambda: TEST_DAY)
        self.assertEqual(disk_result["stopKind"], "disk_low")

        # A new KST day removes the old status gate; the per-reserve guard still stops within the call.
        next_day = TEST_DAY + dt.timedelta(days=1)
        free[0] = 100
        current = [next_day]

        def execute_rollover(args):
            budget = collector.RequestBudget(args.max_requests)
            budget.reserve()
            self.add_attempt(job, when="2026-09-29T00:10:00+00:00")
            current[0] = next_day + dt.timedelta(days=1)
            with self.assertRaisesRegex(collector.BudgetExhausted, "KST day boundary"):
                budget.reserve()
            print("KST day boundary reached; resume next scheduled run")
            return 2

        rollover = daily.run_daily(db_path="data/agri.sqlite", env_file=str(self.root / "test.env"),
                                   root=self.root, today=next_day, disk_floor=5,
                                   disk_free=lambda _: free[0], execute=execute_rollover,
                                   chunk_size=2, kst_today=lambda: current[0])
        self.assertEqual(rollover["stopKind"], "day_rollover")


if __name__ == "__main__":
    unittest.main()
