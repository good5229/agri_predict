import contextlib
import io
import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

from agri_data import collector as c
from agri_data import coverage
from agri_data import dashboard


URL = "https://example.test/B552845/katOrigin/trades"


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "dashboard.sqlite")
        self.db = c.open_db(self.db_path)
        self.job = c.begin_job(self.db, "prices", {"fixture": True})

    def tearDown(self):
        try:
            self.db.close()
        except sqlite3.ProgrammingError:
            pass
        self.tmp.cleanup()

    def add_query(self, day, market, large, medium, rows, subclass=None):
        params = {
            "cond[trd_clcln_ymd::EQ]": day,
            "cond[whsl_mrkt_cd::EQ]": market,
            "cond[gds_lclsf_cd::EQ]": large,
            "cond[gds_mclsf_cd::EQ]": medium,
        }
        if subclass is not None:
            params["cond[gds_sclsf_cd::EQ]"] = subclass
        key = c.task_key("prices_origin", URL, params)
        epoch = "e1"
        plan_state = "complete" if rows else "empty"
        self.db.execute(
            "INSERT INTO planned_queries(job_id,task_key,dataset,source_url,query_json,state,row_count,epoch_id) VALUES(?,?,?,?,?,?,?,?)",
            (self.job, key, "prices_origin", URL, c.json_text(params), plan_state, len(rows), epoch),
        )
        request_params = dict(params, pageNo="1", numOfRows="1000", returnType="json")
        request_key = c.stable_request_key("prices_origin", URL, request_params, epoch)
        cursor = self.db.execute(
            """INSERT INTO source_requests(job_id,request_key,epoch_id,dataset,source_url,query_json,retrieved_at,
                 response_sha256,http_status,api_status,expected_count,returned_count,page_no,state,response_text,attempt_count)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (self.job, request_key, epoch, "prices_origin", URL, c.json_text(request_params),
             "2026-09-28T00:00:00+00:00", "fixture", 200, "0", len(rows), len(rows), 1,
             "complete", "{}", 1),
        )
        request_id = cursor.lastrowid
        for row_no, row in enumerate(rows):
            payload = c.json_text(row)
            self.db.execute("INSERT INTO raw_records VALUES(?,?,?)", (request_id, row_no, payload))
            self.db.execute(
                "INSERT INTO normalized_records VALUES(?,?,?,?,?,?)",
                (request_id, row_no, "prices_origin", c.canonical_date(row.get("trd_clcln_ymd")),
                 payload, c.json_text(c.quality_flags("prices_origin", row))),
            )
        self.db.commit()

    @staticmethod
    def row(day, market, large="10", medium="01", *, amount="100", volume="10",
            trade="경매", unit="kg", subclass="04", subclass_name="봄배추", **extra):
        return {
            "trd_clcln_ymd": day,
            "whsl_mrkt_cd": market,
            "whsl_mrkt_nm": {"110001": "서울가락", "110008": "서울강서"}.get(market),
            "gds_lclsf_cd": large,
            "gds_mclsf_cd": medium,
            "gds_sclsf_cd": subclass,
            "gds_sclsf_nm": subclass_name,
            "trd_se": trade,
            "unit_nm": unit,
            "totprc": amount,
            "unit_tot_qty": volume,
            "unit_qty": "10",
            "qty": "1",
            "scsbd_prc": amount,
            **extra,
        }

    def export(self):
        ro = coverage.open_readonly(self.db_path)
        try:
            return dashboard.build_dashboard(ro, self.db_path, generated_at="2026-09-28T00:00:00+00:00")
        finally:
            ro.close()

    def test_auction_kg_signed_weighted_price_and_causal_horizons(self):
        # The target day's combined signed amount/kg is (100 - 20 + 300) / (10 - 2 + 30) = 10.
        self.add_query("2014-12-20", "110001", "10", "01", [
            self.row("2014-12-20", "110001", amount="50", volume="10")
        ])
        self.add_query("2014-12-20", "110008", "10", "01", [])
        self.add_query("2015-01-01", "110001", "10", "01", [
            self.row("2015-01-01", "110001", amount="100", volume="10"),
            self.row("2015-01-01", "110001", amount="-20", volume="-2"),
            self.row("2015-01-01", "110001", amount="9999", volume="999", trade="정가수의"),
            self.row("2015-01-01", "110001", amount="9999", volume="999", unit="ton"),
            self.row("2015-01-01", "110008", amount="9999", volume="999"),
            self.row("2015-01-02", "110001", amount="9999", volume="999"),
        ])
        self.add_query("2015-01-01", "110008", "10", "01", [
            self.row("2015-01-01", "110008", amount="300", volume="30")
        ])
        self.add_query("2015-01-05", "110001", "10", "01", [
            self.row("2015-01-05", "110001", amount="500", volume="10"),
            self.row("2015-01-05", "110001", amount="-20", volume="0"),
        ])
        self.add_query("2015-01-05", "110008", "10", "01", [])
        # The target is observed, but its last available source would be older than the 14-day cap.
        self.add_query("2015-02-01", "110001", "10", "01", [
            self.row("2015-02-01", "110001", amount="700", volume="10")
        ])
        self.add_query("2015-02-01", "110008", "10", "01", [])
        self.add_query("2015-02-02", "110001", "10", "01", [
            self.row("2015-02-02", "110001", amount="-10", volume="10")
        ])
        self.add_query("2015-02-02", "110008", "10", "01", [])

        report = self.export()
        cabbage = next(p for p in report["products"] if p["id"] == "cabbage")
        observed = {item["date"]: item for item in cabbage["observations"]}
        self.assertEqual(observed["2015-01-01"]["actual"], 10)
        self.assertEqual(observed["2015-01-01"]["volumeKg"], 38)
        self.assertEqual(observed["2015-01-01"]["rowCount"], 3)
        self.assertEqual(observed["2015-01-01"]["marketCodes"], ["110001", "110008"])
        self.assertEqual(cabbage["diagnostics"]["negativeAmountRowsRetained"], 3)
        self.assertEqual(cabbage["diagnostics"]["negativeVolumeRowsRetained"], 1)
        self.assertEqual(cabbage["diagnostics"]["zeroVolumeRowsRetained"], 1)
        self.assertEqual(cabbage["diagnostics"]["excludedRows"]["not_auction"], 1)
        self.assertEqual(cabbage["diagnostics"]["excludedRows"]["unit_not_kg"], 1)
        self.assertEqual(cabbage["diagnostics"]["excludedRows"]["market_mismatch"], 1)
        self.assertEqual(cabbage["diagnostics"]["sourceDateMismatchRows"], 1)
        self.assertNotIn("2015-01-03", observed)

        predictions = [p for p in cabbage["predictions"] if p["target"] == "2015-01-05"]
        p1 = next(p for p in predictions if p["horizon"] == 1)
        p10 = next(p for p in predictions if p["horizon"] == 10)
        self.assertEqual(p1["origin"], "2015-01-04")
        self.assertEqual(p1["sourceDate"], "2015-01-01")
        self.assertEqual(p1["predicted"], 10)
        self.assertEqual(observed["2015-01-05"]["actual"], 48)
        self.assertEqual(p10["origin"], "2014-12-26")
        self.assertEqual(p10["sourceDate"], "2014-12-20")
        self.assertEqual(p10["predicted"], 5)
        self.assertFalse([p for p in cabbage["predictions"] if p["target"] == "2015-02-01"])
        self.assertEqual(cabbage["diagnostics"]["stalePriorObservationByHorizon"]["1"], 1)
        self.assertNotIn("2015-02-02", observed)
        self.assertEqual(cabbage["diagnostics"]["nonpositiveNetPriceDays"], 1)
        self.assertIn("최초 공개 시점은 확인할 수 없으며", report["meta"]["limitations"][0])

    def test_garlic_requires_verified_peeled_subclass_and_pepper_is_labeled_sparse(self):
        garlic_rows = [
            self.row("2015-03-01", "110001", "12", "09", amount="50", volume="5",
                     subclass="06", subclass_name=None),
            self.row("2015-03-01", "110001", "12", "09", amount="100", volume="10",
                     subclass="16", subclass_name="깐마늘(수입)"),
            self.row("2015-03-01", "110001", "12", "09", amount="150", volume="15",
                     subclass="42", subclass_name="깐마늘 대서"),
            self.row("2015-03-01", "110001", "12", "09", amount="90", volume="9",
                     subclass="43", subclass_name="깐마늘 남도"),
            self.row("2015-03-01", "110001", "12", "09", amount="999", volume="1",
                     subclass="01", subclass_name="저장형 한지"),
        ]
        self.add_query("2015-03-01", "110001", "12", "09", garlic_rows)
        self.add_query("2015-03-01", "110008", "12", "09", [])
        # A specific subclass query overlaps the broad garlic result; the specific
        # response takes precedence for subclass 42 without deduplicating trade rows.
        self.add_query("2015-03-01", "110001", "12", "09", [
            self.row("2015-03-01", "110001", "12", "09", amount="150", volume="15",
                     subclass="42", subclass_name="깐마늘 대서")
        ], subclass="42")
        self.add_query("2015-03-01", "110008", "12", "07", [
            self.row("2015-03-01", "110008", "12", "07", amount="25", volume="5",
                     subclass="01", subclass_name="건고추")
        ])
        self.add_query("2015-03-01", "110001", "12", "07", [])
        report = self.export()
        garlic = next(p for p in report["products"] if p["id"] == "garlic")
        pepper = next(p for p in report["products"] if p["id"] == "pepper")
        self.assertEqual(garlic["observations"][0]["actual"], 10)
        self.assertEqual(garlic["observations"][0]["rowCount"], 4)
        self.assertEqual(garlic["observations"][0]["volumeKg"], 39)
        self.assertEqual(garlic["diagnostics"]["excludedRows"]["not_verified_peeled_garlic_class"], 1)
        self.assertEqual(garlic["diagnostics"]["excludedRows"]["overlap_preferred_specific_query"], 1)
        self.assertEqual(pepper["status"], "sparse")
        self.assertIn("거래가 드문", " ".join(pepper["notes"]))
        self.assertEqual([p["id"] for p in report["products"]], ["cabbage", "radish", "onion", "garlic", "pepper"])

    def test_cli_writes_finite_atomic_json_and_database_is_read_only(self):
        self.add_query("2015-06-01", "110001", "11", "01", [
            self.row("2015-06-01", "110001", "11", "01", amount="10.5", volume="2")
        ])
        self.add_query("2015-06-01", "110008", "11", "01", [])
        output = Path(self.tmp.name) / "nested" / "dashboard.json"
        self.assertEqual(dashboard.main(["--db", self.db_path, "--output", str(output)]), 0)
        decoded = json.loads(output.read_text(encoding="utf-8"), parse_constant=lambda x: self.fail(f"nonfinite {x}"))
        self.assertEqual(decoded["schemaVersion"], 1)
        self.assertEqual(len(decoded["products"]), 5)
        self.assertEqual(decoded["meta"]["sourceUrls"], [URL])
        self.assertNotIn("serviceKey", output.read_text(encoding="utf-8"))
        self.assertTrue(all(math.isfinite(item["actual"]) for product in decoded["products"] for item in product["observations"]))
        ro = coverage.open_readonly(self.db_path)
        with self.assertRaises(sqlite3.OperationalError):
            ro.execute("UPDATE jobs SET state='broken'")
        ro.close()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            self.assertEqual(dashboard.main(["--db", self.db_path, "--output", self.db_path]), 2)
        self.assertIn("must differ", stderr.getvalue())
        verify = coverage.open_readonly(self.db_path)
        self.assertEqual(verify.execute("SELECT COUNT(*) FROM jobs").fetchone()[0], 1)
        verify.close()

    def test_observation_requires_both_market_broad_queries_to_be_complete_or_empty(self):
        self.add_query("2015-06-02", "110001", "12", "01", [
            self.row("2015-06-02", "110001", "12", "01", amount="120", volume="10")
        ])
        report = self.export()
        onion = next(p for p in report["products"] if p["id"] == "onion")
        self.assertFalse([o for o in onion["observations"] if o["date"] == "2015-06-02"])
        self.assertGreater(onion["diagnostics"]["missingOrIncompleteMarketQueries"], 0)
        self.assertGreater(onion["diagnostics"]["incompleteMarketDays"], 0)


if __name__ == "__main__":
    unittest.main()
