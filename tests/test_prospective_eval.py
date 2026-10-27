import datetime as dt
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from agri_data import dashboard, prospective_eval


def make_snapshot(end, actual_days):
    products = []
    for pid in dashboard.PRODUCTS:
        observations = [{"date": day.isoformat(), "actual": 100.0 + day.toordinal() % 19,
                         "volumeKg": 1000.0} for day in actual_days]
        products.append({"id": pid, "observations": observations})
    return {"schemaVersion": 1, "meta": {"dataEnd": end.isoformat(), "evaluationEnd": end.isoformat(),
        "marketNames": list(dashboard.MARKETS.values()),
        "aggregation": "각 시장별 부호 있는 총금액(totprc) 합계 ÷ 총중량(unit_tot_qty) 합계. kg"}, "products": products}


def make_provenance(days, *, omit=None, empty=None, rows_without_price=None):
    omit = omit or set(); empty = empty or set(); rows_without_price = rows_without_price or set()
    records = []
    for day in days:
        for pid in dashboard.PRODUCTS:
            for market in dashboard.MARKETS:
                key = (day, pid, market)
                if key in omit: continue
                count = 0 if key in empty else 4 if key in rows_without_price else 1
                records.append({"date": day.isoformat(), "market": market, "product": pid,
                    "jobId": 1000 + len(records), "epochId": f"epoch-{day}-{pid}-{market}",
                    "responseAt": f"{day.isoformat()}T12:00:00+09:00", "rowCount": count,
                    "selectedPrevious": False})
    return records


def write_issue(root, issue, clock, *, status="available", reason=None, horizons=range(1, 11), offset=0):
    path = root / issue.isoformat() / f"{clock.replace(':', '')}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    products = []
    for pid in dashboard.PRODUCTS:
        predictions = [{"target": (issue + dt.timedelta(days=h)).isoformat(), "horizonDays": h,
                        "predictedPrice": 90.0 + h + offset, "sourceDate": (issue-dt.timedelta(days=1)).isoformat(),
                        "clippedLogPrice": False} for h in horizons]
        products.append({"productId": pid, "status": status, "reason": reason,
            "sourceDate": (issue-dt.timedelta(days=1)).isoformat(), "lastObservedPriceDate": (issue-dt.timedelta(days=1)).isoformat(),
            "lastObservedPrice": 95.0, "qualityGuard": {"state": "clear"}, "predictions": predictions})
    payload = {"schemaVersion": 1, "issueAtKst": f"{issue.isoformat()}T{clock}:00+09:00",
        "issueDateKst": issue.isoformat(), "sourceCutoffDate": (issue-dt.timedelta(days=1)).isoformat(),
        "products": products}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class ProspectiveEvaluationTests(unittest.TestCase):
    def test_earliest_window_issue_hash_maturity_hold_and_missing_issue(self):
        start = dt.date(2026, 10, 6)
        end = start + dt.timedelta(days=2)
        data_end = start + dt.timedelta(days=1)
        days = [start, start+dt.timedelta(days=1)]
        snapshot = make_snapshot(data_end, days)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            early = write_issue(root, start, "10:15")
            write_issue(root, start, "10:30")
            write_issue(root, start, "09:59")
            (root / start.isoformat() / "aux-selected-provenance.json").write_text("[]")
            write_issue(root, start+dt.timedelta(days=1), "10:05", status="unavailable_no_fresh_price", reason="stale source")
            self.assertNotIn(root / start.isoformat() / "aux-selected-provenance.json",
                             prospective_eval._forecast_files(root))
            actuals = {p["id"]: {dt.date.fromisoformat(r["date"]): r["actual"] for r in p["observations"]}
                       for p in snapshot["products"]}
            provenance_rows = make_provenance([start+dt.timedelta(days=1)])
            provenance = prospective_eval._validate_provenance(provenance_rows, selected_end=data_end)
            canonical = json.dumps(provenance_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            result = prospective_eval.build(prospective_eval._forecast_files(root), actuals, data_end, data_end,
                "a"*64, provenance, prospective_eval._sha_bytes(canonical.encode()), "b"*64, start, end)
            chosen = result["dailyIssuance"][start.isoformat()]
            self.assertEqual(chosen["selectedFile"], str(early))
            self.assertEqual(chosen["selectedSha256"], prospective_eval._sha_bytes(early.read_bytes()))
            selected_product = result["issueRecords"][0]["products"]["cabbage"]
            self.assertEqual(len(selected_product["predictions"]), 10)
            self.assertEqual(selected_product["lastObservedPriceDate"], (start-dt.timedelta(days=1)).isoformat())
            cabbage1 = result["coverage"]["cabbage/1"]
            self.assertEqual(cabbage1["issuancePossible"], 3)
            self.assertEqual(cabbage1["commonPairs"], 1)
            self.assertIn("immature_target", result["exclusionCounts"]["cabbage/2"])
            self.assertIn("no_issue_in_window", result["exclusionCounts"]["cabbage/1"])
            self.assertIn("hold_reason:stale source", result["exclusionCounts"]["cabbage/1"])
            self.assertEqual(result["jointConclusions"][0]["conclusion"], "inference_unavailable")
            self.assertEqual(len(result["comparisons"]), prospective_eval.FAMILY_SIZE)

    def test_artifact_rejects_duplicate_and_misaligned_target(self):
        issue = dt.date(2026, 10, 6)
        with tempfile.TemporaryDirectory() as tmp:
            path = write_issue(Path(tmp), issue, "10:01", horizons=(1, 1))
            with self.assertRaisesRegex(ValueError, "duplicate prediction"):
                prospective_eval._artifact(path)
            path = write_issue(Path(tmp), issue, "10:01", horizons=(1,))
            raw = json.loads(path.read_text())
            raw["products"][0]["predictions"][0]["target"] = "2026-10-10"
            path.write_text(json.dumps(raw))
            with self.assertRaisesRegex(ValueError, "target/origin mismatch"):
                prospective_eval._artifact(path)

    def test_missing_forecast_horizon_reduces_coverage_against_same_file_baseline(self):
        issue = dt.date(2026, 10, 6)
        data_end = issue + dt.timedelta(days=1)
        snapshot = make_snapshot(data_end, [issue+dt.timedelta(days=1)])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = write_issue(root, issue, "10:00", horizons=range(2, 11))
            actuals = {p["id"]: {dt.date.fromisoformat(r["date"]): r["actual"] for r in p["observations"]}
                       for p in snapshot["products"]}
            prov_rows = make_provenance([issue+dt.timedelta(days=1)])
            prov = prospective_eval._validate_provenance(prov_rows, selected_end=data_end)
            canon = json.dumps(prov_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            result = prospective_eval.build([path], actuals, data_end, data_end, "c"*64, prov,
                prospective_eval._sha_bytes(canon.encode()), "d"*64, issue, issue)
            coverage = result["coverage"]["cabbage/1"]
            self.assertEqual(coverage["baselineCommonOpportunity"], 1)
            self.assertEqual(coverage["commonPairs"], 0)
            conclusion = result["jointConclusions"][0]
            self.assertFalse(conclusion["coverageNondeclining"])
            self.assertEqual(conclusion["conclusion"], "inference_unavailable")

    def test_truth_status_distinguishes_incomplete_empty_rows_and_immature(self):
        issue = dt.date(2026, 10, 6)
        evaluation_end = issue + dt.timedelta(days=4)
        target_days = [issue+dt.timedelta(days=i) for i in range(1, 5)]
        actual_days = [target_days[-1]]
        snapshot = make_snapshot(evaluation_end, actual_days)
        omit = {(target_days[0], "cabbage", list(dashboard.MARKETS)[-1])}
        empty = {(target_days[1], "cabbage", market) for market in dashboard.MARKETS}
        no_price = {(target_days[2], "cabbage", market) for market in dashboard.MARKETS}
        rows = make_provenance(target_days, omit=omit, empty=empty, rows_without_price=no_price)
        prov = prospective_eval._validate_provenance(rows, selected_end=evaluation_end)
        actuals = {p["id"]: {dt.date.fromisoformat(r["date"]): r["actual"] for r in p["observations"]}
                   for p in snapshot["products"]}
        self.assertEqual(prospective_eval._truth_status(target_days[0], "cabbage", actuals, prov, evaluation_end)[0],
                         "incomplete_or_unknown_scope")
        self.assertEqual(prospective_eval._truth_status(target_days[1], "cabbage", actuals, prov, evaluation_end)[0],
                         "complete_empty_response")
        self.assertEqual(prospective_eval._truth_status(target_days[2], "cabbage", actuals, prov, evaluation_end)[0],
                         "complete_rows_no_eligible_price")
        self.assertEqual(prospective_eval._truth_status(evaluation_end+dt.timedelta(days=1), "cabbage", actuals, prov, evaluation_end)[0],
                         "immature_target")
        with tempfile.TemporaryDirectory() as tmp:
            path = write_issue(Path(tmp), issue, "10:00")
            canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            result = prospective_eval.build([path], actuals, evaluation_end, evaluation_end, "f"*64,
                prov, prospective_eval._sha_bytes(canonical.encode()), "g"*64, issue, issue)
            exclusions = result["exclusionCounts"]
            self.assertEqual(exclusions["cabbage/1"].get("incomplete_or_unknown_scope"), 1)
            self.assertEqual(result["coverage"]["cabbage/1"]["matureTarget"], 0)
            self.assertEqual(result["coverage"]["cabbage/1"]["incomplete_or_unknown_scope"],
                             exclusions["cabbage/1"]["incomplete_or_unknown_scope"])
            self.assertEqual(exclusions["cabbage/2"].get("complete_empty_response"), 1)
            self.assertEqual(result["coverage"]["cabbage/2"]["complete_empty_response"],
                             exclusions["cabbage/2"]["complete_empty_response"])
            self.assertEqual(exclusions["cabbage/3"].get("complete_rows_no_eligible_price"), 1)
            self.assertEqual(result["coverage"]["cabbage/3"]["complete_rows_no_eligible_price"],
                             exclusions["cabbage/3"]["complete_rows_no_eligible_price"])
            self.assertEqual(result["coverage"]["cabbage/4"]["commonPairs"], 1)
            self.assertEqual(exclusions["cabbage/5"].get("immature_target"), 1)
            self.assertEqual(result["coverage"]["cabbage/5"]["immatureTarget"],
                             exclusions["cabbage/5"]["immature_target"])

    def test_snapshot_validation_and_inference_starts_at_thirty_common_pairs(self):
        start = dt.date(2026, 10, 6)
        end = start + dt.timedelta(days=29)
        data_end = end + dt.timedelta(days=10)
        actual_days = [start+dt.timedelta(days=i) for i in range((data_end-start).days+1)]
        snapshot = make_snapshot(data_end, actual_days)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = [write_issue(root, start+dt.timedelta(days=i), "10:00", offset=i % 7)
                     for i in range(30)]
            actuals = {p["id"]: {dt.date.fromisoformat(r["date"]): r["actual"] for r in p["observations"]}
                       for p in snapshot["products"]}
            # One fewer issue means 29 common pairs, which cannot enter inference.
            prov_rows = make_provenance(actual_days)
            prov = prospective_eval._validate_provenance(prov_rows, selected_end=data_end)
            canon = json.dumps(prov_rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            short = prospective_eval.build(files[:-1], actuals, data_end, data_end, "b"*64, prov,
                prospective_eval._sha_bytes(canon.encode()), "e"*64, start, end)
            self.assertEqual(short["comparisons"][0]["n"], 29)
            self.assertEqual(short["comparisons"][0]["dm"]["9"]["status"], "insufficient_pairs")
            complete = prospective_eval.build(files, actuals, data_end, data_end, "b"*64, prov,
                prospective_eval._sha_bytes(canon.encode()), "e"*64, start, end)
            self.assertEqual(complete["comparisons"][0]["n"], 30)
            self.assertIn(complete["comparisons"][0]["dm"]["9"]["status"],
                          ("tested", "zero_or_invalid_hac_variance"))
            self.assertEqual(len(complete["comparisons"]), 100)
            bad = make_snapshot(data_end, actual_days)
            bad["meta"]["dataEnd"] = "not-a-date"
            path = root / "bad-prices.json"
            path.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError, "dataEnd"):
                prospective_eval._snapshot(path)

    def test_cli_accepts_date_arguments_and_hashes_explicit_sidecar(self):
        day = dt.date(2026, 10, 6)
        snapshot = make_snapshot(day, [day])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prices_path = root / "prices.json"
            provenance_path = root / "selected-provenance.json"
            output_path = root / "result.json"
            forecast_dir = root / "live-forecasts"
            prices_path.write_text(json.dumps(snapshot), encoding="utf-8")
            provenance_rows = make_provenance([day])
            provenance_path.write_text(json.dumps(provenance_rows), encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()):
                status = prospective_eval.main(["--forecasts", str(forecast_dir),
                    "--prices-snapshot", str(prices_path), "--selected-provenance", str(provenance_path),
                    "--start-date", day.isoformat(), "--end-date", day.isoformat(), "--output", str(output_path)])
            self.assertEqual(status, 0)
            result = json.loads(output_path.read_text())
            self.assertEqual(result["inputs"]["selectedProvenanceByteSha256"],
                             prospective_eval._sha_bytes(provenance_path.read_bytes()))
            self.assertEqual(result["inputs"]["validatedEvaluationEnd"], day.isoformat())

    def test_provenance_job_id_must_match_positive_integer_schema(self):
        day = dt.date(2026, 10, 6)
        rows = make_provenance([day])
        self.assertEqual(len(prospective_eval._validate_provenance(rows, selected_end=day)),
                         len(dashboard.PRODUCTS))
        rows[0]["jobId"] = "123"
        with self.assertRaisesRegex(ValueError, "positive integer"):
            prospective_eval._validate_provenance(rows, selected_end=day)
        rows[0]["jobId"] = True
        with self.assertRaisesRegex(ValueError, "positive integer"):
            prospective_eval._validate_provenance(rows, selected_end=day)


if __name__ == "__main__":
    unittest.main()
