import contextlib
import datetime as dt
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agri_data import modeling


PRODUCT_IDS = ("cabbage", "radish", "onion", "garlic", "pepper")


def base_dashboard(series_by_product):
    products = []
    for product_id in PRODUCT_IDS:
        observations = []
        for day, price, volume in series_by_product.get(product_id, []):
            observations.append({"date": day.isoformat(), "actual": price, "volumeKg": volume, "rowCount": 1})
        products.append({
            "id": product_id,
            "name": product_id,
            "unit": "원/kg",
            "status": "available" if observations else "no_data",
            "notes": ["학습하지 않은 지속성 기준선이며, 미래 운영용 예측은 아닙니다."],
            "observations": observations,
            "predictions": [{"origin": "2014-12-31", "target": "2015-01-01", "horizon": 1,
                             "predicted": 100.0, "sourceDate": "2014-12-31"}],
            "diagnostics": {},
        })
    return {
        "schemaVersion": 1,
        "generatedAt": "2026-09-28T00:00:00+00:00",
        "meta": {
            "title": "테스트",
            "modelKind": "baseline",
            "limitations": ["학습하지 않은 지속성 기준선입니다."],
            "provenance": {"selectedJobIds": [1]},
        },
        "products": products,
    }


def daily_series(end_date, future_scale=1.0):
    start = dt.date(2014, 1, 1)
    rows = []
    day = start
    index = 0
    while day <= end_date:
        base = 100.0 + (index % 19) * 1.25 + math.sin(index / 5.0) * 2.0
        price = base * (future_scale if day >= dt.date(2015, 1, 1) else 1.0)
        rows.append((day, price, 1000.0 + (index % 11) * 10.0))
        day += dt.timedelta(days=1)
        index += 1
    return rows


class ModelingTests(unittest.TestCase):
    def test_cadence_boundaries_are_calendar_only(self):
        self.assertEqual(modeling._cadence_key("daily", dt.date(2015, 3, 2)), dt.date(2015, 3, 2))
        self.assertEqual(modeling._cadence_key("weekly", dt.date(2015, 3, 4)), dt.date(2015, 3, 2))
        self.assertEqual(modeling._cadence_key("monthly", dt.date(2015, 3, 31)), dt.date(2015, 3, 1))
        self.assertEqual(modeling._cadence_key("seasonal", dt.date(2015, 1, 2)), dt.date(2014, 12, 1))
        self.assertEqual(modeling._cadence_key("seasonal", dt.date(2015, 3, 1)), dt.date(2015, 3, 1))

    def test_training_samples_respect_cutoff_and_source_age(self):
        series = {day: (price, 100.0) for day, price, _ in daily_series(dt.date(2015, 2, 1))}
        cutoff = dt.date(2014, 12, 31)
        samples, target_days = modeling._supervised_samples(series, cutoff)
        self.assertTrue(samples)
        self.assertTrue(all(target <= cutoff and origin <= cutoff and origin < target
                            for target, origin, _, _, _ in samples))
        self.assertTrue(all(len(features) == len(modeling.FEATURE_NAMES) for _, _, _, features, _ in samples))
        self.assertTrue(all(target <= cutoff for target in target_days))

    def test_source_age_14_days_is_allowed_and_15_days_is_rejected(self):
        origin = dt.date(2014, 12, 31)
        target = dt.date(2015, 1, 1)
        day_14 = origin - dt.timedelta(days=14)
        day_15 = origin - dt.timedelta(days=15)
        fresh = {day_14: (10.0, 5.0)}
        stale = {day_15: (10.0, 5.0)}
        self.assertEqual(modeling._feature_row(fresh, origin, target, 1, {})[0], day_14)
        self.assertIsNone(modeling._feature_row(stale, origin, target, 1, {}))

    def test_future_actual_values_do_not_change_prior_origin_model_predictions(self):
        end = dt.date(2015, 1, 10)
        original_rows = daily_series(end)
        changed_rows = daily_series(end, future_scale=3.0)
        source_a = base_dashboard({product_id: original_rows for product_id in PRODUCT_IDS})
        source_b = base_dashboard({product_id: changed_rows for product_id in PRODUCT_IDS})
        cutoff = dt.date(2014, 12, 31)
        a = modeling.build_model_dashboard(source_a, origin_start=cutoff, origin_end=cutoff)
        b = modeling.build_model_dashboard(source_b, origin_start=cutoff, origin_end=cutoff)
        self.assertEqual(a["schemaVersion"], 2)
        self.assertEqual(a["products"][0]["predictions"], source_a["products"][0]["predictions"])
        for product_a, product_b in zip(a["products"], b["products"]):
            for run_a, run_b in zip(product_a["runs"], product_b["runs"]):
                self.assertEqual(run_a["modelId"], run_b["modelId"])
                self.assertEqual(run_a["cadence"], run_b["cadence"])
                self.assertEqual(run_a["predictions"], run_b["predictions"])
                for prediction in run_a["predictions"]:
                    origin = dt.date.fromisoformat(prediction["origin"])
                    source_day = dt.date.fromisoformat(prediction["sourceDate"])
                    target = dt.date.fromisoformat(prediction["target"])
                    self.assertLessEqual(source_day, origin)
                    self.assertLessEqual(dt.date.fromisoformat(prediction["fitCutoff"]), origin)
                    self.assertEqual((target - origin).days, prediction["horizon"])
                    self.assertTrue(math.isfinite(prediction["predicted"]) and prediction["predicted"] > 0)

    def test_future_target_availability_changes_only_match_count_not_issued_predictions(self):
        all_rows = daily_series(dt.date(2015, 1, 10))
        missing_target_rows = [row for row in all_rows if row[0] != dt.date(2015, 1, 4)]
        source_a = base_dashboard({product_id: all_rows for product_id in PRODUCT_IDS})
        source_b = base_dashboard({product_id: missing_target_rows for product_id in PRODUCT_IDS})
        cutoff = dt.date(2014, 12, 31)
        a = modeling.build_model_dashboard(source_a, origin_start=cutoff, origin_end=cutoff)
        b = modeling.build_model_dashboard(source_b, origin_start=cutoff, origin_end=cutoff)
        for product_a, product_b in zip(a["products"], b["products"]):
            for run_a, run_b in zip(product_a["runs"], product_b["runs"]):
                self.assertEqual(run_a["diagnostics"]["issuedPredictionCount"], run_b["diagnostics"]["issuedPredictionCount"])
                self.assertEqual(run_a["diagnostics"]["matchedPredictionCount"],
                                 run_b["diagnostics"]["matchedPredictionCount"] + 1)
                shared = {(p["target"], p["horizon"]): p["predicted"] for p in run_a["predictions"]}
                shared_b = {(p["target"], p["horizon"]): p["predicted"] for p in run_b["predictions"]}
                self.assertEqual({k: v for k, v in shared.items() if k in shared_b}, shared_b)

    def test_sparse_product_is_insufficient_and_artifact_manifest_is_reproducible(self):
        long_rows = daily_series(dt.date(2014, 12, 31))
        short_rows = [(dt.date(2014, 12, 31), 100.0, 25.0)]
        source = base_dashboard({"cabbage": long_rows, **{key: short_rows for key in PRODUCT_IDS[1:]}})
        start = dt.date(2014, 12, 31)
        with tempfile.TemporaryDirectory() as tmp:
            artifacts_dir = Path(tmp) / "models"
            output = modeling.build_model_dashboard(source, artifacts_dir,
                                                    origin_start=start, origin_end=start)
            garlic = next(p for p in output["products"] if p["id"] == "garlic")
            self.assertTrue(all(run["status"] == "insufficient_data" for run in garlic["runs"]))
            self.assertTrue(all("최소 90일" in run["reason"] for run in garlic["runs"]))
            run_dir = Path(output["meta"]["training"]["artifacts"]["directory"])
            manifest_path = run_dir / "manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(manifest["models"]), 8)
            self.assertTrue(all((run_dir / item["file"]).is_file() for item in manifest["models"]))
            self.assertEqual(output["meta"]["modelKind"], "comparison")
            self.assertIn("hyperparametersWereTuned", output["meta"]["training"])
            self.assertFalse(output["meta"]["training"]["performanceImprovementClaimed"])
            self.assertTrue(all("학습하지 않은 지속성 기준선" not in note
                                for p in output["products"] for note in p["notes"]))

    def test_model_or_json_failure_preserves_previous_json_and_artifact_run(self):
        long_source = base_dashboard({key: daily_series(dt.date(2014, 12, 31)) for key in PRODUCT_IDS})
        short_source = base_dashboard({key: [(dt.date(2014, 12, 31), 100.0, 10.0)] for key in PRODUCT_IDS})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "prices.json"
            output_path = root / "dashboard.json"
            artifacts = root / "models"
            input_path.write_text(json.dumps(long_source), encoding="utf-8")
            output_path.write_text('{"old":"dashboard"}\n', encoding="utf-8")
            prior_run = artifacts / "run-old"
            prior_run.mkdir(parents=True)
            (prior_run / "manifest.json").write_text('{"old":"manifest"}\n', encoding="utf-8")
            original_json = output_path.read_text(encoding="utf-8")
            original_manifest = (prior_run / "manifest.json").read_text(encoding="utf-8")

            with mock.patch.object(modeling, "_new_estimator", side_effect=RuntimeError("synthetic fit failure")):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(modeling.main(["--input", str(input_path), "--output", str(output_path),
                                                    "--artifacts-dir", str(artifacts)]), 2)
            self.assertEqual(output_path.read_text(encoding="utf-8"), original_json)
            self.assertEqual((prior_run / "manifest.json").read_text(encoding="utf-8"), original_manifest)

            real_write = modeling.write_json_atomic
            input_path.write_text(json.dumps(short_source), encoding="utf-8")
            def fail_output_only(path, value):
                if Path(path).resolve() == output_path.resolve():
                    raise OSError("synthetic output failure")
                return real_write(path, value)

            with mock.patch.object(modeling, "write_json_atomic", side_effect=fail_output_only):
                with contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(modeling.main(["--input", str(input_path), "--output", str(output_path),
                                                    "--artifacts-dir", str(artifacts)]), 2)
            self.assertEqual(output_path.read_text(encoding="utf-8"), original_json)
            self.assertEqual((prior_run / "manifest.json").read_text(encoding="utf-8"), original_manifest)
            self.assertEqual({p.name for p in artifacts.iterdir()}, {"run-old"})


if __name__ == "__main__":
    unittest.main()
