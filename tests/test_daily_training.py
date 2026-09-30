import contextlib
import datetime as dt
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from agri_data import collector, coverage, daily_training, dashboard, modeling


def _valid_schema2(end="2018-12-20"):
    products = []
    for product in dashboard.PRODUCTS:
        products.append({"id": product, "observations": [], "predictions": [],
                         "runs": [{"modelId": model, "cadence": cadence, "predictions": []}
                                  for model in modeling.MODEL_IDS for cadence in modeling.CADENCES]})
    return {"schemaVersion": 2,
            "meta": {"contextStart": "2014-01-01", "contextEnd": end,
                     "evaluationStart": "2017-01-01", "evaluationEnd": end},
            "products": products}


def _complete_prefix(end: dt.date):
    scopes = set(daily_training.EXPECTED_SCOPES)
    result = {}
    day = daily_training.CONTEXT_START
    while day <= end:
        result[day] = scopes.copy()
        day += dt.timedelta(days=1)
    return result


class DailyTrainingTests(unittest.TestCase):
    def test_contiguous_cutoff_stops_before_first_missing_scope_or_day(self):
        day1 = dt.date(2014, 1, 1)
        day2 = day1 + dt.timedelta(days=1)
        rows = {day1: set(daily_training.EXPECTED_SCOPES), day2: set(daily_training.EXPECTED_SCOPES)}
        missing_scope = next(iter(daily_training.EXPECTED_SCOPES))
        rows[day2].remove(missing_scope)
        cutoff, first_missing = daily_training.contiguous_cutoff(rows, day1, day2)
        self.assertEqual(cutoff, day1)
        self.assertEqual(first_missing, day2.isoformat())

    def test_coverage_prefix_requires_ten_valid_broad_scopes_and_accepts_empty(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.sqlite"
            db = collector.open_db(str(path))
            day = "2014-01-01"
            endpoint = f"{collector.PORTAL}/B552845/katOrigin/trades"
            db.execute("INSERT INTO jobs(command,config_json,started_at,state) VALUES('prices','{}','2026-01-01','complete')")
            job_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
            for market in dashboard.MARKETS:
                for product, spec in dashboard.PRODUCTS.items():
                    params = {"cond[trd_clcln_ymd::EQ]": day, "cond[whsl_mrkt_cd::EQ]": market,
                              "cond[gds_lclsf_cd::EQ]": spec["codes"][0],
                              "cond[gds_mclsf_cd::EQ]": spec["codes"][1]}
                    key = collector.task_key("prices_origin", endpoint, params)
                    db.execute("INSERT INTO planned_queries(job_id,task_key,dataset,source_url,query_json,state,row_count,epoch_id) VALUES(?,?,?,?,?,'empty',0,'e1')",
                               (job_id, key, "prices_origin", endpoint, collector.json_text(params)))
                    page = {**params, "pageNo": "1", "numOfRows": "100"}
                    db.execute("""INSERT INTO source_requests(job_id,request_key,epoch_id,dataset,source_url,query_json,retrieved_at,
                               http_status,api_status,expected_count,returned_count,page_no,state)
                               VALUES(?,?,?,?,?,?,?,200,'00',0,0,1,'complete')""",
                               (job_id, collector.stable_request_key("prices_origin", endpoint, page, "e1"), "e1",
                                "prices_origin", endpoint, collector.json_text(page), "2026-01-01T00:00:00Z"))
            db.commit()
            db.close()
            readonly = coverage.open_readonly(str(path))
            try:
                selected, provenance = daily_training._coverage_days(readonly, dt.date(2014, 1, 1), dt.date(2014, 1, 2))
            finally:
                readonly.close()
            self.assertEqual(selected[dt.date(2014, 1, 1)], daily_training.EXPECTED_SCOPES)
            cutoff, gap = daily_training.contiguous_cutoff(selected, dt.date(2014, 1, 1), dt.date(2014, 1, 2))
            self.assertEqual(cutoff, dt.date(2014, 1, 1))
            self.assertEqual(gap, "2014-01-02")
            self.assertEqual(len(provenance), 10)
            self.assertTrue(all(row["rowCount"] == 0 for row in provenance))

    def test_snapshot_hash_ignores_generated_at_but_tracks_provenance_and_code(self):
        source = {"schemaVersion": 1, "generatedAt": "one", "meta": {"provenance": {"ids": [1]}}, "products": []}
        later = {**source, "generatedAt": "two"}
        self.assertEqual(daily_training._pipeline_fingerprint(source, []),
                         daily_training._pipeline_fingerprint(later, []))
        self.assertNotEqual(daily_training._pipeline_fingerprint(source, []),
                            daily_training._pipeline_fingerprint(source, [{"epochId": "new-vintage"}]))

    def test_incremental_source_filters_predictions_but_preserves_context(self):
        source = _valid_schema2()
        product = source["products"][0]
        product["observations"] = [{"date": "2014-01-01", "actual": 10},
                                   {"date": "2018-12-21", "actual": 11}]
        product["predictions"] = [{"origin": "2018-12-19", "target": "2018-12-20", "horizon": 1,
                                   "predicted": 10, "sourceDate": "2018-12-19"},
                                  {"origin": "2018-12-20", "target": "2018-12-21", "horizon": 1,
                                   "predicted": 11, "sourceDate": "2018-12-20"}]
        product["runs"][0]["predictions"] = list(product["predictions"])
        sliced = daily_training._incremental_source(source, dt.date(2018, 12, 21), dt.date(2018, 12, 21))
        self.assertEqual(len(sliced["products"][0]["observations"]), 2)
        self.assertEqual([row["target"] for row in sliced["products"][0]["predictions"]], ["2018-12-21"])
        self.assertEqual(sliced["meta"]["evaluationStart"], "2018-12-21")
        self.assertEqual(sliced["meta"]["statisticsScope"], "incremental_targets_only")
        self.assertNotIn("diagnostics", sliced["products"][0])
        self.assertNotIn("fits", sliced["products"][0]["runs"][0])
        self.assertEqual(sliced["meta"]["parentEvaluationEnd"], "2018-12-20")

    def test_no_change_skips_models_and_archives_provenance(self):
        cutoff = dt.date(2018, 12, 20)
        snapshot = {"schemaVersion": 1, "generatedAt": "transient", "meta": {"contextStart": "2014-01-01"},
                    "products": [{"id": product, "observations": []} for product in dashboard.PRODUCTS]}
        fingerprint = daily_training._pipeline_fingerprint(snapshot, [])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_path = root / "data" / "agri.sqlite"
            db_path.parent.mkdir()
            db = collector.open_db(str(db_path))
            db.close()
            (root / "runs").mkdir()
            (root / "runs" / "daily-collection.lock").touch()
            status = {"schemaVersion": 1, "latestSuccess": {"cutoff": cutoff.isoformat(),
                      "snapshotSha256": fingerprint, "runDirectory": "outputs/daily/old/run"}}
            (root / "runs" / "daily-training-status.json").write_text(json.dumps(status), encoding="utf-8")
            mocked_days = _complete_prefix(cutoff)
            post_gap_provenance = [{"date": (cutoff + dt.timedelta(days=10)).isoformat(),
                                    "market": "post-gap", "product": "post-gap", "epochId": "later"}]
            with mock.patch.object(daily_training, "_coverage_days",
                                   return_value=(mocked_days, post_gap_provenance)), \
                 mock.patch.object(daily_training.shutil, "disk_usage", return_value=SimpleNamespace(free=10 * 1024**3)):
                result = daily_training.run_daily_training(root=root, db_path=db_path,
                    today=dt.date(2026, 9, 30), exporter=lambda *args, **kwargs: snapshot,
                    model_main=lambda argv: self.fail("no-change run must not retrain"))
            self.assertEqual(result["status"], "no_change")
            run_dir = root / result["runDirectory"]
            self.assertTrue((run_dir / "selected-provenance.json").is_file())
            self.assertEqual(json.loads((run_dir / "selected-provenance.json").read_text(encoding="utf-8")), [])
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("outputs/daily/old/run", report)
            self.assertNotIn("모델 artifact: `models/`", report)
            saved = json.loads((root / "runs" / "daily-training-status.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["latestSuccess"], status["latestSuccess"])

    def test_success_path_runs_fixed_models_full_and_incremental_reports(self):
        cutoff = dt.date(2018, 12, 20)
        baseline_cutoff = dt.date(2018, 12, 19)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_path = root / "data" / "agri.sqlite"
            db_path.parent.mkdir()
            db = collector.open_db(str(db_path))
            db.close()
            (root / "runs").mkdir()
            (root / "runs" / "daily-collection.lock").touch()
            baseline = root / "outputs" / "dashboard-expanded-20260929.json"
            baseline.parent.mkdir(parents=True)
            baseline.write_text(json.dumps(_valid_schema2(baseline_cutoff.isoformat())), encoding="utf-8")
            snapshot = {"schemaVersion": 1, "generatedAt": "snapshot", "meta": {"contextStart": "2014-01-01"},
                        "products": [{"id": p, "observations": []} for p in dashboard.PRODUCTS]}
            expected_prediction = {"origin": "2018-12-19", "target": "2018-12-20", "horizon": 1,
                                   "predicted": 10.0, "sourceDate": "2018-12-19", "fitCutoff": "2018-12-19"}
            def fake_model(argv):
                output_path = Path(argv[argv.index("--output") + 1])
                products = []
                for product in dashboard.PRODUCTS:
                    products.append({"id": product, "observations": [{"date": "2018-12-20", "actual": 11.0}],
                                     "predictions": [expected_prediction],
                                     "runs": [{"modelId": model, "cadence": cadence,
                                               "predictions": [expected_prediction]}
                                              for model in modeling.MODEL_IDS for cadence in modeling.CADENCES]})
                output_path.write_text(json.dumps({"schemaVersion": 2,
                    "meta": {"contextStart": "2014-01-01", "contextEnd": cutoff.isoformat(),
                             "evaluationStart": "2017-01-01", "evaluationEnd": cutoff.isoformat()},
                    "products": products}), encoding="utf-8")
                return 0
            sig_calls = []
            def fake_significance(argv):
                source_path, output_path = Path(argv[argv.index("--input") + 1]), Path(argv[argv.index("--output") + 1])
                source = json.loads(source_path.read_text(encoding="utf-8"))
                sig_calls.append(source["meta"]["evaluationStart"])
                records, conclusions = [], []
                for product in dashboard.PRODUCTS:
                    for model in modeling.MODEL_IDS:
                        for loss in ("absolute_error", "absolute_percentage_error"):
                            records.append({"productId": product, "modelId": model, "cadence": "daily", "horizon": 1,
                                "loss": loss, "primary": {"n": 1, "candidateMeanLoss": 2.0, "baselineMeanLoss": 3.0,
                                    "test": {"status": "unavailable_insufficient_pairs", "holmAdjustedPValue": 1.0}}})
                        conclusions.append({"productId": product, "modelId": model, "cadence": "daily", "horizon": 1,
                                            "status": "unavailable"})
                output_path.write_text(json.dumps({"comparisons": records, "conclusions": conclusions}), encoding="utf-8")
                Path(argv[argv.index("--markdown") + 1]).write_text("synthetic", encoding="utf-8")
                return 0
            mocked_days = _complete_prefix(cutoff)
            with mock.patch.object(daily_training, "_coverage_days", return_value=(mocked_days, [])), \
                 mock.patch.object(daily_training.shutil, "disk_usage", return_value=SimpleNamespace(free=10 * 1024**3)):
                result = daily_training.run_daily_training(root=root, db_path=db_path, baseline_dashboard=baseline,
                    today=dt.date(2026, 9, 30), exporter=lambda *args, **kwargs: snapshot,
                    model_main=fake_model, significance_main=fake_significance)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(sig_calls, ["2017-01-01", "2018-12-20"])
            run_dir = root / result["runDirectory"]
            self.assertTrue((run_dir / "predictions.json").exists())
            self.assertTrue((run_dir / "predictions-incremental.json").exists())
            self.assertTrue((run_dir / "significance-full.json").exists())
            self.assertTrue((run_dir / "significance-incremental.json").exists())
            report = (run_dir / "report.md").read_text(encoding="utf-8")
            self.assertIn("기준 MAE", report)
            self.assertIn("기준 MAPE%", report)
            self.assertIn("신규 기간 — 일별 1일 예측", report)
            incremental = json.loads((run_dir / "predictions-incremental.json").read_text(encoding="utf-8"))
            self.assertEqual(incremental["meta"]["statisticsScope"], "incremental_targets_only")
            self.assertNotIn("diagnostics", incremental["products"][0])

    def test_failure_pointer_preserves_previous_success(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            old = {"schemaVersion": 1, "latestSuccess": {"runId": "good", "cutoff": "2020-01-01"}}
            (root / "runs").mkdir()
            (root / "runs" / "daily-training-status.json").write_text(json.dumps(old), encoding="utf-8")
            failed = daily_training._failure_report(root, "bad-run", dt.date(2026, 9, 30),
                                                     "synthetic model failure", old)
            state = json.loads((root / "runs" / "daily-training-status.json").read_text(encoding="utf-8"))
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(state["latestSuccess"], old["latestSuccess"])
            self.assertEqual(state["latestAttempt"]["runId"], "bad-run")
            self.assertTrue((root / failed["runDirectory"] / "report.md").exists())

    def test_busy_training_lock_never_rewrites_shared_status(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db_path = root / "data" / "agri.sqlite"
            db_path.parent.mkdir()
            db = collector.open_db(str(db_path))
            db.close()
            (root / "runs").mkdir()
            lock_path = root / "runs" / "daily-training.lock"
            lock_path.touch()
            status_path = root / "runs" / "daily-training-status.json"
            status_path.write_text('{"latestSuccess":{"runId":"stable"}}', encoding="utf-8")
            before = status_path.read_bytes()
            with lock_path.open("a+") as lock:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    code = daily_training.main(["--root", str(root), "--db", str(db_path)])
                self.assertEqual(code, 2)
                self.assertEqual(status_path.read_bytes(), before)
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    unittest.main()
