import datetime as dt
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agri_data import collector, diagnostics, experiments


def _weather_db(current_state="incomplete"):
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript("""
    CREATE TABLE jobs(job_id INTEGER PRIMARY KEY, started_at TEXT, state TEXT);
    CREATE TABLE planned_queries(job_id INTEGER,task_key TEXT,dataset TEXT,source_url TEXT,query_json TEXT,state TEXT,row_count INTEGER,epoch_id TEXT);
    CREATE TABLE source_requests(request_id INTEGER PRIMARY KEY,job_id INTEGER,epoch_id TEXT,dataset TEXT,source_url TEXT,query_json TEXT,retrieved_at TEXT,response_sha256 TEXT,http_status INTEGER,api_status TEXT,expected_count INTEGER,returned_count INTEGER,page_no INTEGER,state TEXT,published_at TEXT,version_published_at TEXT);
    CREATE TABLE raw_records(request_id INTEGER,row_no INTEGER,payload_json TEXT);
    CREATE TABLE normalized_records(request_id INTEGER,row_no INTEGER,dataset TEXT,reference_date TEXT,payload_json TEXT,quality_flags TEXT);
    """)
    params = {"dataCd": "ASOS", "dateCd": "DAY", "dataType": "JSON", "startDt": "20140101", "endDt": "20141231", "stnIds": "100"}
    task_key = collector.task_key("weather", collector.ASOS, {k: str(v) for k, v in params.items()})
    db.execute("INSERT INTO jobs VALUES(1,'2026-01-01','running')")
    db.execute("INSERT INTO planned_queries VALUES(1,?,'weather',?,?,?,NULL,'e2')",
               (task_key, collector.ASOS, json.dumps(params), current_state))
    for request_id, epoch, when, day in ((1, "e1", "2025-01-01T00:00:00Z", "2014-01-01"),
                                         (2, "e2", "2026-01-01T00:00:00Z", "2014-01-02")):
        page_params = {**params, "pageNo": "1", "numOfRows": "999"}
        db.execute("""INSERT INTO source_requests VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                   (request_id, 1, epoch, "weather", collector.ASOS, json.dumps(page_params), when,
                    f"hash{request_id}", 200, "00", 1, 1, 1, "complete", None, None))
        payload = {"stnId": "100", "tm": day, "avgTa": "-2.5", "maxTa": "1", "minTa": "-5", "sumRn": "0"}
        db.execute("INSERT INTO raw_records VALUES(?,?,?)", (request_id, 1, json.dumps(payload)))
        db.execute("INSERT INTO normalized_records VALUES(?,?,?,?,?,?)",
                   (request_id, 1, "weather", day, json.dumps(payload), "[]"))
    db.commit()
    return db


class DiagnosticsTests(unittest.TestCase):
    def test_design_matrix_has_observed_weekday_fixed_effects(self):
        import numpy as np
        samples = [
            {"date": dt.date(2017, 1, 2), "x": 1.0, "y": 1.0},
            {"date": dt.date(2017, 1, 3), "x": 2.0, "y": 2.0},
            {"date": dt.date(2017, 2, 7), "x": 3.0, "y": 3.0},
        ]
        matrix, names = diagnostics._design_matrix(samples, np)
        self.assertIn("weekday_01", names)
        self.assertEqual(matrix[:, names.index("weekday_01")].tolist(), [0.0, 1.0, 1.0])

    def test_composition_pairs_cover_gap_buckets_and_reconciliation_gate(self):
        def obs(day, price, volume=10.0, category_share=0.8):
            return {"date": day.isoformat(), "actual": price, "volumeKg": volume, "rowCount": 2,
                    "marketData": {"110001": {"price": price, "volumeKg": volume,
                        "compositionFieldShares": {"gds_sclsf_cd": {"A": category_share},
                                                   "grd_cd": {"1": category_share},
                                                   "plor_cd": {"KR": category_share}}}}}
        days = [dt.date(2017, 1, 1), dt.date(2017, 1, 2), dt.date(2017, 1, 4), dt.date(2017, 1, 15)]
        rows = [obs(days[0], 10), obs(days[1], 11), obs(days[2], 12), obs(days[3], 14)]
        rows[2]["marketData"]["110001"]["volumeKg"] = 5
        series = diagnostics._price_series({"id": "pepper", "observations": rows})
        result = diagnostics._composition_diagnostics(series, days[0], days[-1])
        self.assertEqual(result["priceObservationTransitionsByCalendarGap"], {"1": 1, "2-7": 1, ">7": 1})
        gap1 = result["jointPairDiagnosticsByCalendarGap"]["1"]
        self.assertEqual(gap1["pairs"][0]["compositionTvd"]["market"], 0.0)
        self.assertFalse(result["jointPairDiagnosticsByCalendarGap"]["2-7"]["pairs"][0]["reconciliationValid"])
        self.assertIsNone(result["jointPairDiagnosticsByCalendarGap"][">7"]["pairs"][0]["compositionTvd"].get("market"))
        self.assertEqual(result["compositionShareChanges"]["market"]["transitionCount"], 1)

    def test_weather_epoch_fallback_rejects_current_incomplete_pointer(self):
        db = _weather_db()
        selected, report = diagnostics._weather_plans(db, dt.date(2014, 1, 1), dt.date(2014, 12, 31))
        self.assertEqual(report["selectedQueryCount"], 1)
        item = next(iter(selected.values()))
        self.assertEqual(item["plan"]["epochId"], "e1")
        self.assertTrue(item["fallback"])
        self.assertTrue(any(row["epochId"] == "e2" and row["reason"] == "current_manifest_epoch_not_complete"
                            for row in report["queryScopes"][0]["excludedEpochs"]))
        weather, _ = diagnostics._weather_records(db, selected, dt.date(2014, 1, 1), dt.date(2014, 12, 31))
        self.assertEqual(weather["100"][dt.date(2014, 1, 1)]["avgTa"], -2.5)
        self.assertNotIn(dt.date(2014, 1, 2), weather["100"])

    def test_current_complete_epoch_wins_and_context_filters_query_tail(self):
        db = _weather_db(current_state="complete")
        db.execute("UPDATE planned_queries SET row_count=1")
        db.commit()
        selected, _ = diagnostics._weather_plans(db, dt.date(2014, 1, 1), dt.date(2014, 1, 31))
        self.assertEqual(next(iter(selected.values()))["plan"]["epochId"], "e2")
        db.execute("UPDATE normalized_records SET payload_json=replace(payload_json,'2014-01-02','2014-01-01'), reference_date='2014-01-01' WHERE request_id=2")
        db.commit()
        weather, _ = diagnostics._weather_records(db, selected, dt.date(2014, 1, 1), dt.date(2014, 1, 1))
        self.assertEqual(set(weather["100"]), {dt.date(2014, 1, 1)})

    def test_onion_error_identity_and_strict_overshoot_boundary(self):
        source = {"schemaVersion": 2, "meta": {"evaluationStart": "2017-01-01", "evaluationEnd": "2017-01-03"}, "products": []}
        date = "2017-01-01"
        product = {"id": "onion", "observations": [{"date": date, "actual": 12}],
                   "predictions": [{"target": date, "horizon": 1, "predicted": 10}],
                   "runs": [{"modelId": "ridge", "cadence": "daily",
                             "predictions": [{"target": date, "horizon": 1, "predicted": 14}]}]}
        original = diagnostics.significance._validate_input
        try:
            diagnostics.significance._validate_input = lambda _: ({"onion": product}, {},
                dt.date(2017, 1, 1), dt.date(2017, 1, 3))
            result = diagnostics._onion_error_diagnostics(source, "abc", None)["primaryDailyHorizon1"][0]
        finally:
            diagnostics.significance._validate_input = original
        self.assertEqual(result["classifications"], {"same_direction_nonharmful": 1})
        self.assertEqual(result["meanMseDifferenceContribution"], result["observedMseDifference"])

    def test_market_share_keeps_unaccounted_mass_without_renormalizing(self):
        observation = {"volumeKg": 10, "marketData": {"110001": {"volumeKg": 10,
            "compositionFieldShares": {"gds_sclsf_cd": {"A": 0.7}}}}}
        shares, invalid, missing = diagnostics._shares(observation, "subclass")
        self.assertFalse(invalid)
        self.assertEqual(shares["A"], 0.7)
        self.assertAlmostEqual(shares[diagnostics.TVD_MISSING], 0.3)
        self.assertAlmostEqual(missing, 0.3)

    def test_module_cli_runs_against_temporary_readonly_database(self):
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            db_path = tmp_path / "sample.sqlite"
            connection = collector.open_db(str(db_path))
            connection.close()
            products = [{"id": product, "name": product, "observations": []}
                        for product in experiments.PRODUCT_IDS]
            price_path = tmp_path / "prices.json"
            price_path.write_text(json.dumps({"schemaVersion": 1, "products": products}), encoding="utf-8")
            model_products = []
            for product in experiments.PRODUCT_IDS:
                model_products.append({"id": product, "observations": [], "predictions": [],
                    "runs": [{"modelId": model, "cadence": cadence, "predictions": []}
                             for model in ("ridge", "hist_gb")
                             for cadence in ("daily", "weekly", "monthly", "seasonal")]})
            predictions_path = tmp_path / "predictions.json"
            predictions_path.write_text(json.dumps({"schemaVersion": 2,
                "meta": {"evaluationStart": "2017-01-01", "evaluationEnd": "2018-12-20"},
                "products": model_products}), encoding="utf-8")
            out_json, out_md = tmp_path / "report.json", tmp_path / "report.md"
            result = subprocess.run([sys.executable, "-m", "agri_data.diagnostics", "--db", str(db_path),
                "--prices", str(price_path), "--predictions", str(predictions_path),
                "--output", str(out_json), "--markdown", str(out_md)],
                cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertEqual(len(report["associations"]), diagnostics.FAMILY_SIZE)
            self.assertTrue(report["protocol"]["predictionsSha256"])
            self.assertTrue(out_md.exists())


if __name__ == "__main__":
    unittest.main()
