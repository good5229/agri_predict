import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agri_data import collector as c
from agri_data import coverage as cov


URL = "https://example.test/prices"


class CoverageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "coverage.sqlite")
        self.db = c.open_db(self.path)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def job(self, state="complete"):
        job = c.begin_job(self.db, "prices", {"test": True})
        self.db.execute("UPDATE jobs SET state=? WHERE job_id=?", (state, job))
        self.db.commit()
        return job

    @staticmethod
    def query(day="2014-06-03", market="G1", subclass=None):
        q = {
            "cond[trd_clcln_ymd::EQ]": day,
            "cond[whsl_mrkt_cd::EQ]": market,
            "cond[gds_lclsf_cd::EQ]": "10",
            "cond[gds_mclsf_cd::EQ]": "01",
        }
        if subclass is not None:
            q["cond[gds_sclsf_cd::EQ]"] = subclass
        return q

    def plan(self, job, query, state="complete", epoch="e1", rows=1):
        key = c.task_key("prices_origin", URL, query)
        self.db.execute(
            "INSERT INTO planned_queries(job_id,task_key,dataset,source_url,query_json,state,row_count,epoch_id) VALUES(?,?,?,?,?,?,?,?)",
            (job, key, "prices_origin", URL, c.json_text(query), state, rows if state in ("complete", "empty") else None, epoch),
        )
        self.db.commit()
        return key

    def request(self, job, query, epoch, *, row=None, state="complete", total=1, page=1,
                retrieved_at="2026-01-01T00:00:00+00:00"):
        params = dict(query, pageNo=str(page), numOfRows="1000", returnType="json")
        request_key = c.stable_request_key("prices_origin", URL, params, epoch)
        payload = c.json_text(row) if row is not None else None
        cursor = self.db.execute(
            """INSERT INTO source_requests(job_id,request_key,epoch_id,dataset,source_url,query_json,retrieved_at,
                 response_sha256,http_status,api_status,expected_count,returned_count,page_no,state,response_text,attempt_count)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (job, request_key, epoch, "prices_origin", URL, c.json_text(params), retrieved_at,
             "abc", 200, "0", total, int(row is not None), page, state, "{}", 1),
        )
        req = cursor.lastrowid
        if row is not None:
            self.db.execute("INSERT INTO raw_records VALUES(?,?,?)", (req, 0, payload))
            self.db.execute("INSERT INTO normalized_records VALUES(?,?,?,?,?,?)",
                            (req, 0, "prices_origin", c.canonical_date(row.get("trd_clcln_ymd")), payload, c.json_text(c.quality_flags("prices_origin", row))))
        self.db.commit()
        return req

    @staticmethod
    def record(day="2014-06-03", subclass="001", **extra):
        return {"trd_clcln_ymd": day, "trd_se": "01", "gds_sclsf_cd": subclass,
                "unit": "10kg", "unit_qty": "10", "unit_tot_qty": "20", "mdfcn_dt": "2014-07-01", **extra}

    def report(self):
        db = cov.open_readonly(self.path)
        try:
            return cov.build_report(db, self.path)
        finally:
            db.close()

    def test_duplicate_jobs_and_latest_incomplete_use_only_latest_completed_epoch(self):
        query = self.query()
        old_job = self.job()
        self.plan(old_job, query, epoch="old", rows=1)
        self.request(old_job, query, "old", row=self.record(day="2014-06-04", subclass="old", qty="-1"))
        new_job = self.job("incomplete")
        self.plan(new_job, query, state="incomplete", epoch="partial")
        # A successful page from an unfinished query epoch must not be mixed into coverage.
        self.request(new_job, query, "partial", row=self.record(subclass="partial"), state="complete")

        with contextlib.redirect_stdout(io.StringIO()):
            result = self.report()
        group = result["groups"][0]
        self.assertEqual(group["planned_days"], 1)
        self.assertEqual(group["normalized_rows"], 1)
        self.assertEqual(group["subclass_code_counts"], {"old": 1})
        self.assertEqual(group["latest_attempt_state_days"], {"incomplete": 1})
        self.assertEqual(group["selected_previous_completion_days"], 1)
        self.assertEqual(group["incomplete_days"], 1)
        self.assertEqual(group["source_date_mismatch_rows"], 1)
        self.assertEqual(group["first_observed_date"], "2014-06-04")
        self.assertEqual(group["mdfcn_dt_distribution"], {"2014-07-01": 1})
        self.assertEqual(group["quality_flag_counts"]["negative:qty"], 1)
        self.assertIn("publication/version time is unknown", group["publication_time_interpretation"])

    def test_latest_confirmed_empty_supersedes_prior_nonempty_epoch(self):
        query = self.query()
        old_job = self.job()
        self.plan(old_job, query, epoch="old", rows=1)
        self.request(old_job, query, "old", row=self.record())
        new_job = self.job()
        self.plan(new_job, query, state="empty", epoch="new", rows=0)
        self.request(new_job, query, "new", total=0)

        group = self.report()["groups"][0]
        self.assertEqual(group["normalized_rows"], 0)
        self.assertEqual(group["observed_days"], 0)
        self.assertEqual(group["empty_days"], 1)
        self.assertEqual(group["first_observed_date"], None)
        self.assertIn("not a closure day", group["empty_interpretation"])

    def test_epoch_order_uses_response_time_not_resumed_job_started_at(self):
        query = self.query()
        older = self.job()
        self.plan(older, query, epoch="old", rows=1)
        self.request(older, query, "old", row=self.record(subclass="old"),
                     retrieved_at="2020-01-01T00:00:00+00:00")
        newer = self.job()
        self.plan(newer, query, epoch="new", rows=1)
        self.request(newer, query, "new", row=self.record(subclass="new"),
                     retrieved_at="2025-01-01T00:00:00+00:00")
        # Resuming an older job changes started_at; that must not make its vintage newer.
        self.db.execute("UPDATE jobs SET started_at='2099-01-01T00:00:00+00:00' WHERE job_id=?", (older,))
        self.db.execute("UPDATE jobs SET started_at='2000-01-01T00:00:00+00:00' WHERE job_id=?", (newer,))
        self.db.commit()

        group = self.report()["groups"][0]
        self.assertEqual(group["subclass_code_counts"], {"new": 1})
        self.assertEqual(group["selected_response_last_retrieved_at"], "2025-01-01T00:00:00+00:00")

    def test_same_job_old_epoch_is_recovered_after_manifest_pointer_changes(self):
        query = self.query()
        job = self.job("incomplete")
        self.plan(job, query, state="incomplete", epoch="partial")
        self.request(job, query, "old-complete", row=self.record(subclass="old"), total=1, page=1,
                     retrieved_at="2024-01-01T00:00:00+00:00")
        group = self.report()["groups"][0]
        self.assertEqual(group["normalized_rows"], 1)
        self.assertEqual(group["subclass_code_counts"], {"old": 1})
        self.assertEqual(group["selected_previous_completion_days"], 1)

    def test_truncated_page_is_excluded_even_if_manifest_says_complete(self):
        query = self.query()
        old_job = self.job()
        self.plan(old_job, query, state="complete", epoch="old", rows=1)
        self.request(old_job, query, "old", row=self.record(subclass="old"), total=1, page=1,
                     retrieved_at="2024-01-01T00:00:00+00:00")
        newer = self.job()
        self.plan(newer, query, state="complete", epoch="truncated", rows=1500)
        # Page one cannot complete a 1,500-row result; page two is absent.
        self.request(newer, query, "truncated", row=self.record(subclass="partial"), total=1500, page=1,
                     retrieved_at="2025-01-01T00:00:00+00:00")

        group = self.report()["groups"][0]
        self.assertEqual(group["normalized_rows"], 1)
        self.assertEqual(group["subclass_code_counts"], {"old": 1})
        self.assertEqual(group["selected_previous_completion_days"], 1)
        self.assertGreaterEqual(group["excluded_epoch_candidates"], 1)

    def test_scopes_and_partial_month_denominator_are_separate(self):
        job = self.job()
        broad = self.query(day="2014-06-03")
        filtered = self.query(day="2014-06-03", subclass="006")
        for query, epoch, subclass in ((broad, "broad", "001"), (filtered, "filtered", "006")):
            self.plan(job, query, epoch=epoch, rows=1)
            self.request(job, query, epoch, row=self.record(subclass=subclass))
        pending = self.query(day="2014-06-04")
        self.plan(job, pending, state="pending", epoch="pending")
        failed = self.query(day="2014-06-05")
        self.plan(job, failed, state="failed", epoch="failed")

        report = self.report()
        self.assertEqual(len(report["groups"]), 2)
        for group in report["groups"]:
            self.assertEqual(group["calendar_days"], 30)
            self.assertEqual(group["observed_days"], 1)
            if group["scope"] == "all_subclasses":
                self.assertEqual(group["planned_days"], 3)
                self.assertEqual(group["unplanned_calendar_days"], 27)
                self.assertEqual(group["pending_days"], 1)
                self.assertEqual(group["failed_days"], 1)
            else:
                self.assertEqual(group["planned_days"], 1)
                self.assertEqual(group["unplanned_calendar_days"], 29)
                self.assertEqual(group["pending_days"], 0)
                self.assertEqual(group["failed_days"], 0)
        self.assertEqual({g["scope"] for g in report["groups"]}, {"all_subclasses", "subclass:006"})

    def test_readonly_cli_emits_one_json_document_and_cannot_write(self):
        query = self.query()
        job = self.job()
        self.plan(job, query, epoch="one", rows=1)
        self.request(job, query, "one", row=self.record())
        self.db.close()
        db = cov.open_readonly(self.path)
        with self.assertRaises(sqlite3.OperationalError):
            db.execute("UPDATE jobs SET state='broken'")
        db.close()
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            self.assertEqual(cov.main(["--db", self.path]), 0)
        decoded = json.loads(stdout.getvalue())
        self.assertTrue(decoded["read_only"])
        self.assertEqual(len(decoded["groups"]), 1)

    def test_direct_request_pages_join_matches_previous_source_request_join(self):
        query = self.query()
        job = self.job()
        self.plan(job, query, epoch="one", rows=1)
        self.request(job, query, "one", row=self.record(), retrieved_at="2025-04-05T06:07:08+00:00")
        self.db.create_function("coverage_task_key", 3, cov._request_task_key, deterministic=True)

        previous_join = cov.SELECTION_CTE + """
SELECT p.*,
       r.request_id,r.page_no,r.expected_count,r.returned_count,r.api_status,r.retrieved_at,r.published_at,r.version_published_at,
       COALESCE(rp.raw_n,0) AS raw_n,COALESCE(rp.norm_n,0) AS norm_n,
       n.reference_date,n.payload_json,n.quality_flags
FROM picked p
LEFT JOIN source_requests r
  ON r.job_id=p.selected_job_id AND r.dataset=p.dataset AND r.source_url=p.source_url
 AND r.epoch_id=p.selected_epoch_id AND r.state='complete'
LEFT JOIN request_pages rp ON rp.request_id=r.request_id
LEFT JOIN normalized_records n ON n.request_id=r.request_id
ORDER BY p.dataset,p.task_key,r.request_id,n.row_no
"""
        before = [tuple(row) for row in self.db.execute(previous_join)]
        after = [tuple(row) for row in self.db.execute(cov._selection_sql())]
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()
