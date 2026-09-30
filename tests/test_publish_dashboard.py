import datetime as dt
import fcntl
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agri_data import dashboard, modeling, publish_dashboard, significance


def _model_export():
    products = []
    for product in dashboard.PRODUCTS:
        products.append({"id": product, "observations": [], "predictions": [],
                         "runs": [{"modelId": model, "cadence": cadence, "predictions": []}
                                  for model in modeling.MODEL_IDS for cadence in modeling.CADENCES]})
    return {"schemaVersion": 2, "meta": {"evaluationStart": "2017-01-01", "evaluationEnd": "2018-12-20",
                                           "contextStart": "2014-01-01", "contextEnd": "2018-12-20"},
            "products": products}


class PublishDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.today = dt.date(2026, 9, 30)
        self.run_id = "daily-run"
        self.run_dir = self.root / "outputs" / "daily" / self.today.isoformat() / self.run_id
        self.run_dir.mkdir(parents=True)
        source = _model_export()
        source_bytes = (json.dumps(source, ensure_ascii=False, indent=2) + "\n").encode()
        self.source_bytes = source_bytes
        (self.run_dir / "predictions.json").write_bytes(source_bytes)
        report = significance.analyze(source, hashlib.sha256(source_bytes).hexdigest())
        self.report_bytes = (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode()
        (self.run_dir / "significance-full.json").write_bytes(self.report_bytes)
        status = {"runId": self.run_id, "runDirectory": str(self.run_dir.relative_to(self.root)),
                 "status": "complete", "cutoff": "2018-12-20", "snapshotSha256": "snapshot",
                 "finishedAtKst": "2026-09-30T12:23:42+09:00",
                 "predictions": str((self.run_dir / "predictions.json").relative_to(self.root)),
                 "fullSignificance": str((self.run_dir / "significance-full.json").relative_to(self.root))}
        (self.run_dir / "run-status.json").write_text(json.dumps(status), encoding="utf-8")
        runs = self.root / "runs"
        runs.mkdir()
        pointer = {"latestSuccess": {"runId": self.run_id, "runDirectory": status["runDirectory"],
                                      "cutoff": status["cutoff"], "snapshotSha256": "snapshot"}}
        (runs / "daily-training-status.json").write_text(json.dumps(pointer), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_publish_valid_latest_pair_and_backup_existing_files(self):
        outputs = self.root / "outputs"
        dashboard_path, significance_path = outputs / "dashboard.json", outputs / "significance.json"
        dashboard_path.write_bytes(b"old dashboard")
        significance_path.write_bytes(b"old significance")
        result = publish_dashboard.publish(root=self.root, today=self.today)
        self.assertEqual(result["status"], "published")
        self.assertEqual(dashboard_path.read_bytes(), self.source_bytes)
        self.assertEqual(significance_path.read_bytes(), self.report_bytes)
        backup = self.root / result["backupDirectory"]
        self.assertEqual((backup / "dashboard.json").read_bytes(), b"old dashboard")
        self.assertEqual((backup / "significance.json").read_bytes(), b"old significance")
        published = json.loads((self.root / "runs" / "dashboard-publication-status.json").read_text())
        self.assertEqual(published["status"], "published")

    def test_second_replace_failure_rolls_back_existing_pair_and_absent_file(self):
        outputs = self.root / "outputs"
        outputs.mkdir(exist_ok=True)
        dashboard_path, significance_path = outputs / "dashboard.json", outputs / "significance.json"
        significance_path.write_bytes(b"old significance")
        real_replace = publish_dashboard._atomic_replace
        calls = 0

        def fail_dashboard(path, payload):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic second replacement failure")
            return real_replace(path, payload)

        with mock.patch.object(publish_dashboard, "_atomic_replace", side_effect=fail_dashboard):
            with self.assertRaisesRegex(publish_dashboard.PublishError, r"rollback failures=\[\]"):
                publish_dashboard.publish(root=self.root, today=self.today)
        self.assertFalse(dashboard_path.exists())
        self.assertEqual(significance_path.read_bytes(), b"old significance")
        status = json.loads((self.root / "runs" / "dashboard-publication-status.json").read_text())
        self.assertEqual(status["status"], "failed")
        self.assertTrue((self.root / status["backupDirectory"] / "significance.json").exists())

    def test_hash_mismatch_and_dry_run_never_change_served_pair(self):
        outputs = self.root / "outputs"
        outputs.mkdir(exist_ok=True)
        dashboard_path, significance_path = outputs / "dashboard.json", outputs / "significance.json"
        dashboard_path.write_bytes(b"keep dashboard")
        significance_path.write_bytes(b"keep significance")
        before = (dashboard_path.read_bytes(), significance_path.read_bytes())
        bad = json.loads(self.report_bytes)
        bad["protocol"]["inputSha256"] = "wrong"
        (self.run_dir / "significance-full.json").write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(publish_dashboard.PublishError, "exact prediction bytes"):
            publish_dashboard.publish(root=self.root, today=self.today)
        self.assertEqual((dashboard_path.read_bytes(), significance_path.read_bytes()), before)
        failed_status = json.loads((self.root / "runs" / "dashboard-publication-status.json").read_text())
        self.assertEqual(failed_status["status"], "failed")
        status_before_dry = (self.root / "runs" / "dashboard-publication-status.json").read_bytes()

        with self.assertRaises(publish_dashboard.PublishError):
            publish_dashboard.publish(root=self.root, today=self.today, dry_run=True)
        self.assertEqual((self.root / "runs" / "dashboard-publication-status.json").read_bytes(), status_before_dry)
        self.assertEqual((dashboard_path.read_bytes(), significance_path.read_bytes()), before)

        (self.run_dir / "significance-full.json").write_bytes(self.report_bytes)
        dry = publish_dashboard.publish(root=self.root, today=self.today, dry_run=True)
        self.assertEqual(dry["status"], "dry_run")
        self.assertEqual((dashboard_path.read_bytes(), significance_path.read_bytes()), before)
        self.assertEqual((self.root / "runs" / "dashboard-publication-status.json").read_bytes(), status_before_dry)

    def test_explicit_run_directory_and_idempotent_publish(self):
        explicit = self.run_dir.relative_to(self.root)
        first = publish_dashboard.publish(root=self.root, run_directory=explicit, today=self.today)
        self.assertEqual(first["status"], "published")
        backups = list((self.root / "outputs" / "dashboard-publication-backups").glob("*"))
        # The scheduled no-change path on the following day republishes the prior successful archive safely.
        second = publish_dashboard.publish(root=self.root, today=dt.date(2026, 10, 1))
        self.assertEqual(second["status"], "no_change")
        self.assertEqual(len(list((self.root / "outputs" / "dashboard-publication-backups").glob("*"))), len(backups))

    def test_rejects_training_lock_and_noncurrent_archive(self):
        lock_path = self.root / "runs" / "daily-training.lock"
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(publish_dashboard.PublishError, "training is active"):
                publish_dashboard.publish(root=self.root, today=self.today)
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        with self.assertRaisesRegex(publish_dashboard.PublishError, "not today's"):
            publish_dashboard.publish(root=self.root, today=dt.date(2026, 10, 1), require_today=True)

    def test_backup_failure_does_not_touch_served_files(self):
        outputs = self.root / "outputs"
        outputs.mkdir(exist_ok=True)
        dashboard_path, significance_path = outputs / "dashboard.json", outputs / "significance.json"
        dashboard_path.write_bytes(b"existing dashboard")
        significance_path.write_bytes(b"existing stats")
        before = (dashboard_path.read_bytes(), significance_path.read_bytes())
        with mock.patch.object(publish_dashboard, "_backup_pair", side_effect=OSError("backup failed")):
            with self.assertRaisesRegex(publish_dashboard.PublishError, "rollback failures"):
                publish_dashboard.publish(root=self.root, today=self.today)
        self.assertEqual((dashboard_path.read_bytes(), significance_path.read_bytes()), before)
        status = json.loads((self.root / "runs" / "dashboard-publication-status.json").read_text())
        self.assertEqual(status["status"], "failed")

    def test_status_write_failure_after_replace_rolls_back_served_pair(self):
        outputs = self.root / "outputs"
        outputs.mkdir(exist_ok=True)
        dashboard_path, significance_path = outputs / "dashboard.json", outputs / "significance.json"
        dashboard_path.write_bytes(b"old dashboard")
        significance_path.write_bytes(b"old significance")
        before = (dashboard_path.read_bytes(), significance_path.read_bytes())
        with mock.patch.object(publish_dashboard, "_write_status", side_effect=OSError("status write failed")):
            with self.assertRaisesRegex(publish_dashboard.PublishError, "rollback failures"):
                publish_dashboard.publish(root=self.root, today=self.today)
        self.assertEqual((dashboard_path.read_bytes(), significance_path.read_bytes()), before)


if __name__ == "__main__":
    unittest.main()
