import datetime as dt
import math
import unittest
from unittest import mock

from agri_data import radish_horizon_sharing_score as score


def fixture(start, n=45, *, eligible_from=None, omit_d_at=None, products=("radish",)):
    forecasts = []
    baseline = {}
    actuals = {}
    for i in range(n):
        target = start + dt.timedelta(days=i)
        actual = 100.0 + i * .7
        for product in products:
            actuals[(product, target)] = actual
        for product in products:
          for h in score.HORIZONS:
            origin = target - dt.timedelta(days=h)
            key = (product, origin, target, h)
            conds = {}
            for c, offset in (("B", 10.0 + (i % 3)), ("D", 7.0 + (i % 4)),
                              ("C", 2.0 + (i % 5)), ("P", 8.0 + (i % 2))):
                status = "not_issued" if c == "D" and omit_d_at == target else "issued"
                direct = actual + offset
                conds[c] = {"status": status, "directLogPrediction": math.log(direct/100.0) if status == "issued" else None,
                            "directPrediction": direct if status == "issued" else None,
                            "blendPrediction": direct if status == "issued" else None,
                            "fallbackReason": None, "fitId": f"{c}-{h}", "cutoff": origin.isoformat(),
                            "p0": 100.0, "p0Date": origin.isoformat(), "sarimaPrediction": 100.0}
            row = {"key": {"product": product, "originDate": origin.isoformat(),
                           "targetDate": target.isoformat(), "horizon": h},
                   "eligible": origin >= (eligible_from or start), "conditions": conds}
            forecasts.append(row)
            if product == "radish":
                baseline[key] = {"status": conds["B"]["status"], "prediction": conds["B"]["blendPrediction"],
                                 "reason": None, "directStatus": "issued", "directPrediction": conds["B"]["directPrediction"],
                                 "sarimaStatus": "issued", "sarimaPrediction": 100.0, "fallbackReason": None}
    train = [{"targetDate": (start-dt.timedelta(days=i+1)).isoformat(), "actualLogReturn": .1+i*.01,
              "B": .09+i*.01, "D": .095+i*.01, "C": .1+i*.01, "P": .08+i*.01}
             for i in range(3)]
    diagnostics = [{"fitId": "fit-1", "horizon": 1, "fitOrigin": start.isoformat(),
                    "trainingCount": len(train), "trainingRows": train}]
    return {"forecasts": forecasts}, baseline, actuals, diagnostics


class RadishHorizonSharingScoreTests(unittest.TestCase):
    def test_primary_pairs_ignore_control_missingness_and_family_stays_fixed(self):
        start = dt.date(2026, 4, 1)
        end = start + dt.timedelta(days=44)
        doc, baseline, actuals, diagnostics = fixture(start, n=45, omit_d_at=start+dt.timedelta(days=10))
        with mock.patch.dict(score.TARGET_WINDOWS, {"V": (start, end)}), mock.patch.object(score, "PRODUCTS", ("radish",)), mock.patch.object(score, "BOOTSTRAP_REPS", 12):
            report = score.score_manifest(doc, baseline_rows=baseline, actuals=actuals,
                                          fit_diagnostics=diagnostics, phase="V")
        dm = report["dm"]
        self.assertEqual(len([r for r in dm if r["bandwidth"] == 9]), 60)
        cb = next(r for r in dm if r["comparison"] == "C_vs_B" and r["horizon"] == 1 and r["loss"] == "ae" and r["bandwidth"] == 9)
        cd = next(r for r in dm if r["comparison"] == "C_vs_D" and r["horizon"] == 1 and r["loss"] == "ae" and r["bandwidth"] == 9)
        self.assertEqual(cb["n"], 44)
        self.assertEqual(cd["n"], 43)
        self.assertEqual(report["radishPrimary"]["perHorizon"][9]["commonPairs"], 35)
        self.assertEqual(len(report["radishMetricsByHorizonCondition"]), 40)
        self.assertEqual(len(report["allFourCommonMetrics"]), 40)
        self.assertTrue(all({"blendMae", "blendMape"} <= set(r) for r in report["allFourCommonMetrics"]))
        self.assertEqual(len(report["monthlyCoverage"]), 2 * 10)
        self.assertEqual(report["fitDiagnostics"]["fitDiagnostics"][0]["logMseDecomposition"]["C_vs_B"]["identityResidual"], 0.0)
        self.assertGreater(len(report["fitDiagnostics"]["fitDiagnostics"][0]["logMseDecompositionByTargetMonth"]), 0)
        expected_macro = sum(r["CBlendMape"] for r in report["radishPrimary"]["perHorizon"])/10
        self.assertAlmostEqual(report["radishPrimary"]["equalHorizonMetrics"]["blendMape"]["candidate"], expected_macro)
        self.assertTrue(any(r["comparison"] == "C_vs_B" and r["sampleScope"] == "all_four_common" and r["n"] > 0 for r in report["technicalPairMetrics"]))
        self.assertTrue(any(r["aggregation"] == "equal_weight_10_horizons" for r in report["logMseDecomposition"]))
        self.assertTrue(any(r["fitOrigin"] == start.isoformat() and r["targetMonth"] == "all" for r in report["fitDiagnostics"]["jointFitOriginEqualHorizonDecomposition"]))
        self.assertEqual(len(report["bootstrap"][0]["results"]), 6)
        self.assertEqual(report["bootstrap"][0]["replicatesRequested"], 12)

    def test_manifest_key_grid_duplicate_and_baseline_binding(self):
        start = dt.date(2026, 4, 1)
        end = start + dt.timedelta(days=10)
        doc, baseline, actuals, diagnostics = fixture(start, n=11)
        with mock.patch.dict(score.TARGET_WINDOWS, {"V": (start, end)}), mock.patch.object(score, "PRODUCTS", ("radish",)), mock.patch.object(score, "BOOTSTRAP_REPS", 2):
            bad = {**doc, "forecasts": [*doc["forecasts"], doc["forecasts"][0]]}
            with self.assertRaisesRegex(ValueError, "duplicate"):
                score.score_manifest(bad, baseline_rows=baseline, actuals=actuals, fit_diagnostics=diagnostics, phase="V")
            baseline[next(iter(baseline))]["prediction"] += 1
            with self.assertRaisesRegex(ValueError, "reproduce"):
                score.score_manifest(doc, baseline_rows=baseline, actuals=actuals, fit_diagnostics=diagnostics, phase="V")

    def test_fallback_status_is_not_silently_converted_to_issue(self):
        start = dt.date(2026, 4, 1)
        end = start + dt.timedelta(days=10)
        doc, baseline, actuals, diagnostics = fixture(start, n=11)
        row = doc["forecasts"][10]
        row["conditions"]["C"]["status"] = "fallback"
        with mock.patch.dict(score.TARGET_WINDOWS, {"V": (start, end)}), mock.patch.object(score, "PRODUCTS", ("radish",)), mock.patch.object(score, "BOOTSTRAP_REPS", 2):
            report = score.score_manifest(doc, baseline_rows=baseline, actuals=actuals, fit_diagnostics=diagnostics, phase="V")
        ccoverage = report["coverage"]["fallbackKeys"].get("C", 0)
        self.assertEqual(ccoverage, 1)

    def test_t_uses_july_first_origin_cutoff_and_full_target_grid(self):
        target_start = dt.date(2026, 7, 11)
        origin_cutoff = dt.date(2026, 7, 1)
        doc, baseline, actuals, diagnostics = fixture(target_start, n=79, eligible_from=origin_cutoff)
        with mock.patch.object(score, "PRODUCTS", ("radish",)), mock.patch.object(score, "BOOTSTRAP_REPS", 2):
            report = score.score_manifest(doc, baseline_rows=baseline, actuals=actuals,
                                          fit_diagnostics=diagnostics, phase="T")
        self.assertEqual(report["coverage"]["plannedEligibleKeys"], 790)
        self.assertEqual(report["coverage"]["C_vs_BCommonPairsByProductHorizon"]["radish/10"], 79)

    def test_four_crop_auxiliary_cells_use_all_four_condition_common_rows(self):
        start = dt.date(2026, 4, 1)
        end = start + dt.timedelta(days=10)
        doc, baseline, actuals, diagnostics = fixture(start, n=11, products=score.PRODUCTS)
        with mock.patch.dict(score.TARGET_WINDOWS, {"V": (start, end)}), mock.patch.object(score, "BOOTSTRAP_REPS", 2):
            report = score.score_manifest(doc, baseline_rows=baseline, actuals=actuals,
                                          fit_diagnostics=diagnostics, phase="V")
        cells = report["allFourCommonMetrics"]
        self.assertEqual(len(cells), 4*10*4)
        self.assertEqual({r["condition"] for r in cells}, set(score.CONDITIONS))
        self.assertTrue(all(r["n"] > 0 and r["blendMae"] is not None and r["blendMape"] is not None for r in cells))
        self.assertEqual(len(report["allFourMonthlyMetrics"]), 4*10*4)

    def test_nonzero_prediction_changes_and_baseline_replay_tolerance(self):
        start = dt.date(2026, 4, 1)
        end = start + dt.timedelta(days=10)
        doc, baseline, actuals, diagnostics = fixture(start, n=11)
        row = doc["forecasts"][10]
        c = row["conditions"]["C"]
        c["directLogPrediction"] += .01
        c["directPrediction"] = c["p0"] * math.exp(c["directLogPrediction"])
        c["blendPrediction"] = c["directPrediction"]
        b = row["conditions"]["B"]
        b["blendPrediction"] += 5e-7
        with mock.patch.dict(score.TARGET_WINDOWS, {"V": (start, end)}), mock.patch.object(score, "PRODUCTS", ("radish",)), mock.patch.object(score, "BOOTSTRAP_REPS", 2):
            report = score.score_manifest(doc, baseline_rows=baseline, actuals=actuals,
                                          fit_diagnostics=diagnostics, phase="V")
        change = next(r for r in report["predictionChangesVsB"] if r["horizon"] == 1 and r["condition"] == "C" and r["targetMonth"] == "2026-04")
        self.assertGreater(change["nChangedKeys"], 0)
        self.assertGreater(change["rmsLogPredictionChange"], 0)

    def test_same_predictions_are_separate_from_constant_loss(self):
        start = dt.date(2026, 4, 1)
        end = start + dt.timedelta(days=44)
        doc, baseline, actuals, diagnostics = fixture(start, n=45)
        for row in doc["forecasts"]:
            if not row["eligible"]: continue
            b, c = row["conditions"]["B"], row["conditions"]["C"]
            c["directLogPrediction"] = b["directLogPrediction"]
            c["directPrediction"] = b["directPrediction"]
            c["blendPrediction"] = b["blendPrediction"]
        with mock.patch.dict(score.TARGET_WINDOWS, {"V": (start, end)}), mock.patch.object(score, "PRODUCTS", ("radish",)), mock.patch.object(score, "BOOTSTRAP_REPS", 2):
            report = score.score_manifest(doc, baseline_rows=baseline, actuals=actuals,
                                          fit_diagnostics=diagnostics, phase="V")
        test = next(r for r in report["dm"] if r["comparison"] == "C_vs_B" and r["horizon"] == 1 and r["loss"] == "ae" and r["bandwidth"] == 9)
        self.assertEqual(test["status"], "identical_predictions")
        self.assertEqual(test["pValue"], 1.0)

    def test_unavailable_control_does_not_shrink_primary_and_keeps_fixed_p1_cells(self):
        start = dt.date(2026, 4, 1)
        end = start + dt.timedelta(days=10)
        doc, baseline, actuals, diagnostics = fixture(start, n=11)
        for row in doc["forecasts"]:
            row["conditions"]["D"].update(status="not_issued", directLogPrediction=None,
                                           directPrediction=None, blendPrediction=None)
        with mock.patch.dict(score.TARGET_WINDOWS, {"V": (start, end)}), mock.patch.object(score, "PRODUCTS", ("radish",)), mock.patch.object(score, "BOOTSTRAP_REPS", 2):
            report = score.score_manifest(doc, baseline_rows=baseline, actuals=actuals,
                                          fit_diagnostics=diagnostics, phase="V")
        cb = next(r for r in report["dm"] if r["comparison"] == "C_vs_B" and r["horizon"] == 10 and r["loss"] == "ape" and r["bandwidth"] == 9)
        cd = next(r for r in report["dm"] if r["comparison"] == "C_vs_D" and r["horizon"] == 10 and r["loss"] == "ape" and r["bandwidth"] == 9)
        self.assertEqual(cb["n"], 1)
        self.assertEqual(cd["n"], 0)
        self.assertEqual(cd["pValue"], 1.0)
        self.assertEqual(report["dmSummaryByBandwidth"][0]["familySize"], 60)


if __name__ == "__main__":
    unittest.main()
