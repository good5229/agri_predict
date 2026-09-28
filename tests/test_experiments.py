import datetime as dt
import contextlib
import io
import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from agri_data import collector, coverage, experiments, modeling


def make_product(rows, product_id="cabbage"):
    return {"id": product_id, "observations": rows, "diagnostics": {}}


def daily_rows(end=dt.date(2015, 1, 20), change_after=None, scale=1.0):
    day = dt.date(2014, 1, 1)
    rows = []
    i = 0
    while day <= end:
        price = 100 + (i % 13) * 1.7 + math.sin(i / 6) * 3
        if change_after and day >= change_after:
            price *= scale
        rows.append({
            "date": day.isoformat(), "actual": price, "volumeKg": 1000 + i % 17,
            "rowCount": 2,
            "marketData": {
                "110001": {"price": price * 0.98, "volumeKg": 600, "rowCount": 1,
                           "qualityFlagCount": 0, "composition": {
                               "subclass": {"distinctCategories": 2, "largestSignedVolumeShare": .7,
                                            "concentrationHhi": .58},
                               "grade": {"distinctCategories": 1, "largestSignedVolumeShare": 1,
                                         "concentrationHhi": 1},
                               "origin": {"distinctCategories": 2, "largestSignedVolumeShare": .8,
                                          "concentrationHhi": .68}}},
                "110008": {"price": price * 1.02, "volumeKg": 400, "rowCount": 1,
                           "qualityFlagCount": 1, "composition": {}},
            }})
        day += dt.timedelta(days=1)
        i += 1
    return rows


def source(rows):
    product_ids = ("cabbage", "radish", "onion", "garlic", "pepper")
    return {"products": [make_product(rows if pid == "cabbage" else [], pid) for pid in product_ids]}


class ExperimentTests(unittest.TestCase):
    def test_feature_panel_is_causal_and_market_composition_uses_known_rows(self):
        origin = dt.date(2015, 1, 10)
        target = dt.date(2015, 1, 13)
        rows_a = daily_rows(dt.date(2015, 1, 20))
        rows_b = daily_rows(dt.date(2015, 1, 20), change_after=origin + dt.timedelta(days=1), scale=8)
        p_a, p_b = make_product(rows_a), make_product(rows_b)
        series_a, series_b = experiments._series(p_a), experiments._series(p_b)
        for feature_set in ("core", "relative", "same_weekday", "trading_observation", "combined",
                            "core+market_composition+publication"):
            left = experiments._feature(p_a, series_a, origin, target, 3, feature_set, {},
                                        [{"measure": "production", "value": 123, "referenceYear": 2014,
                                          "availableDate": "2015-01-11", "publishedAt": "2015-01-10T12:00:00+09:00"}])
            right = experiments._feature(p_b, series_b, origin, target, 3, feature_set, {},
                                         [{"measure": "production", "value": 123, "referenceYear": 2014,
                                           "availableDate": "2015-01-11", "publishedAt": "2015-01-10T12:00:00+09:00"}])
            self.assertEqual(left, right)
        composed = experiments._feature(p_a, series_a, origin, target, 3, "core+market_composition", {})[1]
        self.assertGreater(len(composed), len(modeling.FEATURE_NAMES))
        self.assertTrue(all(math.isfinite(v) for v in composed))

    def test_combined_adds_all_three_feature_families_and_market_width_is_stable(self):
        rows = daily_rows(dt.date(2015, 1, 20))
        origin, target = dt.date(2015, 1, 10), dt.date(2015, 1, 13)
        product = make_product(rows)
        series = experiments._series(product)
        base = experiments._feature(product, series, origin, target, 3, "core", {})[1]
        combined = experiments._feature(product, series, origin, target, 3, "combined", {})[1]
        self.assertEqual(len(combined)-len(base), 3+9+12)
        no_markets = make_product([{**r, "marketData": {}} for r in rows])
        none_vector = experiments._feature(no_markets, experiments._series(no_markets), origin, target, 3,
                                           "core+market_composition", {})[1]
        self.assertEqual(len(none_vector)-len(base), 2*14)
        stale_rows = []
        stale_date = (origin-dt.timedelta(days=15)).isoformat()
        for row in rows:
            market_data = {}
            if row["date"] == stale_date:
                market_data["110001"] = row["marketData"]["110001"]
            if row["date"] == origin.isoformat():
                market_data["110008"] = row["marketData"]["110008"]
            stale_rows.append({**row, "marketData": market_data})
        stale_product = make_product(stale_rows)
        stale_vector = experiments._feature(stale_product, experiments._series(stale_product), origin, target, 3,
                                             "core+market_composition", {})[1]
        self.assertEqual(len(stale_vector)-len(base), 2*14)

    def test_e0_matches_existing_modeling_monthly_predictions(self):
        rows = daily_rows(dt.date(2015, 1, 8))
        product = make_product(rows)
        series = experiments._series(product)
        cache = {}
        sample_pool = modeling._supervised_samples(series, max(series), cache)[0]
        matrix = np.asarray([s[3] for s in sample_pool], dtype=np.float64)
        labels = np.asarray([s[4] for s in sample_pool], dtype=np.float64)
        start, end = dt.date(2015, 1, 1), dt.date(2015, 1, 8)
        actuals = {d: p for d, (p, _) in series.items() if start <= d <= end}
        legacy, _ = modeling._fit_cadence(
            series, "hist_gb", "monthly", start - dt.timedelta(days=10),
            end - dt.timedelta(days=1), actuals, sample_pool, cache, matrix, labels)
        current = experiments._fit(product, {"model": "hist_gb", "features": "core", "target": "log_ratio",
                                              "grouping": "pooled", "cadence": "monthly"},
                                   start, end, 90)
        legacy_by_key = {(p["target"], p["horizon"]): p["predicted"] for p in legacy["predictions"]}
        current_by_key = {(p["target"], p["horizon"]): p["predicted"] for p in current["predictions"]}
        self.assertEqual(legacy_by_key.keys(), current_by_key.keys())
        for key in legacy_by_key:
            self.assertAlmostEqual(legacy_by_key[key], current_by_key[key], places=10)
        by_origin_month = {}
        for pred in current["predictions"]:
            origin = dt.date.fromisoformat(pred["origin"])
            by_origin_month.setdefault(origin.strftime("%Y-%m"), set()).add(pred["fitCutoff"])
        self.assertTrue(all(len(cutoffs) == 1 for cutoffs in by_origin_month.values()))

    def test_horizon_groups_and_transform_inverse_are_fixed(self):
        self.assertEqual(experiments._group(1, "grouped"), "1-3")
        self.assertEqual(experiments._group(3, "grouped"), "1-3")
        self.assertEqual(experiments._group(4, "grouped"), "4-7")
        self.assertEqual(experiments._group(8, "grouped"), "8-10")
        self.assertEqual(experiments._group(10, "pooled"), "pooled")
        self.assertAlmostEqual(experiments._inverse(experiments._label(110, 100, "log_ratio"), 100, "log_ratio"), 110)
        self.assertEqual(experiments._inverse(experiments._label(110, 100, "price_delta"), 100, "price_delta"), 110)
        self.assertEqual(experiments._inverse(110, 100, "price_level"), 110)

    def test_parent_gate_requires_same_pair_both_metrics_and_no_coverage_loss(self):
        actuals = {(f"2015-01-{d:02}", 1): 100.0 for d in range(1, 13)}
        parent = [{"target": key[0], "horizon": 1, "predicted": 120.0} for key in actuals]
        candidate = [{"target": key[0], "horizon": 1, "predicted": 101.0} for key in actuals]
        changed, comparison = experiments._improves(candidate, parent, actuals)
        self.assertTrue(changed)
        self.assertEqual(comparison["candidate"]["n"], 12)
        lost_coverage = candidate[:-1]
        changed, comparison = experiments._improves(lost_coverage, parent, actuals)
        self.assertFalse(changed)
        self.assertFalse(comparison["coverageNondecrease"])

    def test_dm_hac_uses_observed_pairs_and_actual_calendar_gaps(self):
        actuals, candidate, parent = {}, [], []
        for index, day in enumerate((1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16)):
            key = (f"2015-01-{day:02}", 1)
            actuals[key] = 100.0
            parent.append({"target": key[0], "horizon": 1, "predicted": 110 + index % 3})
            candidate.append({"target": key[0], "horizon": 1, "predicted": 100 + index % 2})
        result = experiments._dm(candidate, parent, actuals, 1, "absolute_error")
        self.assertEqual(result["n"], 11)
        self.assertEqual(result["calendarHacMaxLagDays"], 9)
        self.assertIn(result["status"], ("available", "unavailable_nonpositive_variance"))

    def test_publication_feature_begins_only_after_confirmed_release_day(self):
        rows, metadata = experiments._publication_evidence("docs/publication-evidence.json")
        self.assertGreater(metadata["includedObservations"], 0)
        first = min(experiments.parse_day(row["availableDate"]) for row in rows)
        before = experiments._publish_features(first - dt.timedelta(days=1), rows)
        at = experiments._publish_features(first, rows)
        self.assertTrue(all(value == 0 for value in before))
        self.assertTrue(any(value != 0 for value in at))
        self.assertTrue(all(row["publishedAt"].endswith("+09:00") for row in rows))

    def test_holdout_forwards_frozen_publication_values_only_to_s4_winner(self):
        panel = {"products": [make_product([], pid) for pid in experiments.PRODUCT_IDS],
                 "panelStart": "2014-01-01", "panelEnd": "2016-03-31",
                 "marketCodes": list(experiments.MARKETS), "marketNames": list(experiments.MARKETS.values()),
                 "sourceUrls": [], "selectedJobIds": [], "selectedRequestCount": 0, "publicationTimeKnown": False}
        evidence = [{"measure": "production", "value": 100, "referenceYear": 2014,
                     "availableDate": "2015-01-01", "publishedAt": "2014-12-31T12:00:00+09:00"}]
        evidence_hash = experiments._json_hash(evidence)
        frozen = {}
        for pid in experiments.PRODUCT_IDS:
            winner = {"model": "hist_gb", "features": "core+publication" if pid == "onion" else "core",
                      "target": "log_ratio", "grouping": "pooled", "cadence": "monthly"}
            frozen[pid] = {"winner": winner, "e0": {**winner, "features": "core"}}
        development = {"mode": "development", "selectionFrozen": True,
                       "protocolSha256": experiments._json_hash(experiments.PROTOCOL),
                       "panel": {k: panel[k] for k in ("panelStart", "panelEnd", "marketCodes", "marketNames", "sourceUrls")},
                       "pricePanelSha256": experiments._panel_hash(panel),
                       "publicationEvidenceSha256": evidence_hash, "frozenSelection": frozen}
        seen = []
        def fake_run(product, config, start, end, min_days, label, callback, publication_values=None):
            seen.append((product["id"], config["features"], publication_values))
            return {"status": "insufficient_data", "predictions": [], "fits": [],
                    "diagnostics": {"issued": 0, "matched": 0}}
        with mock.patch.object(experiments, "_fit_logged", side_effect=fake_run):
            experiments.run_holdout(panel, development, min_days=10, cadences=("monthly",),
                                    publication_values=evidence, publication_sha256=evidence_hash)
        onion_s4 = [entry for entry in seen if entry[0] == "onion" and "publication" in entry[1]]
        self.assertEqual(len(onion_s4), 1)
        self.assertEqual(onion_s4[0][2], evidence)

    def test_panel_requires_complete_two_market_broad_query_and_signed_aggregation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "panel.sqlite")
            db = collector.open_db(path)
            job = collector.begin_job(db, "prices", {"fixture": True})
            url = "https://example.test/B552845/katOrigin/trades"
            day = "2015-06-01"
            def add(market, rows, query_day=day):
                params = {"cond[trd_clcln_ymd::EQ]": query_day, "cond[whsl_mrkt_cd::EQ]": market,
                          "cond[gds_lclsf_cd::EQ]": "10", "cond[gds_mclsf_cd::EQ]": "01"}
                key = collector.task_key("prices_origin", url, params)
                state = "complete" if rows else "empty"
                db.execute("INSERT INTO planned_queries(job_id,task_key,dataset,source_url,query_json,state,row_count,epoch_id) VALUES(?,?,?,?,?,?,?,?)",
                           (job, key, "prices_origin", url, collector.json_text(params), state, len(rows), "e1"))
                request = dict(params, pageNo="1", numOfRows="1000", returnType="json")
                reqkey = collector.stable_request_key("prices_origin", url, request, "e1")
                cur = db.execute("""INSERT INTO source_requests(job_id,request_key,epoch_id,dataset,source_url,query_json,
                    retrieved_at,response_sha256,http_status,api_status,expected_count,returned_count,page_no,state,response_text,attempt_count)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (job, reqkey, "e1", "prices_origin", url, collector.json_text(request),
                     "2016-01-01T00:00:00+00:00", "hash", 200, "0", len(rows), len(rows), 1, "complete", "{}", 1))
                rid = cur.lastrowid
                for i, row in enumerate(rows):
                    payload = collector.json_text(row)
                    db.execute("INSERT INTO raw_records VALUES(?,?,?)", (rid, i, payload))
                    db.execute("INSERT INTO normalized_records VALUES(?,?,?,?,?,?)",
                               (rid, i, "prices_origin", query_day, payload, collector.json_text([])))
            def row(market, amount, volume, row_day=day):
                return {"trd_clcln_ymd": row_day, "whsl_mrkt_cd": market, "gds_lclsf_cd": "10",
                        "gds_mclsf_cd": "01", "gds_sclsf_cd": "04", "gds_sclsf_nm": "배추",
                        "trd_se": "경매", "unit_nm": "kg", "totprc": amount, "unit_tot_qty": volume,
                        "grd_cd": "01", "grd_nm": "특", "plor_cd": "11", "plor_nm": "국내"}
            add("110001", [row("110001", "100", "10")])
            add("110008", [row("110008", "-20", "0")])
            add("110001", [row("110001", "500", "50", "2015-06-02")], "2015-06-02")
            db.commit()
            ro = coverage.open_readonly(path)
            try:
                panel = experiments.build_price_panel(ro, dt.date(2015, 6, 1), dt.date(2015, 6, 2))
                json.dumps(panel, allow_nan=False)
                self.assertEqual(len(experiments._panel_hash(panel)), 64)
                product = next(x for x in panel["products"] if x["id"] == "cabbage")
                self.assertEqual(product["observations"][0]["actual"], 8.0)
                self.assertEqual(product["observations"][0]["volumeKg"], 10.0)
                self.assertEqual([row["date"] for row in product["observations"]], ["2015-06-01"])
                self.assertIn("grade", product["diagnostics"]["compositionFieldsSeen"])
                self.assertIn("origin", product["diagnostics"]["compositionFieldsSeen"])
                shares = product["observations"][0]["marketData"]["110001"]["compositionFieldShares"]
                self.assertEqual(shares["grd_cd"], {"01": 1.0})
                self.assertEqual(shares["plor_cd"], {"11": 1.0})
                with self.assertRaises(sqlite3.OperationalError):
                    ro.execute("UPDATE jobs SET state='bad'")
            finally:
                ro.close()
                db.close()

    def test_target_guard_checks_dates_price_and_volume_with_declared_tolerance(self):
        rows = [{"date": "2015-06-01", "actual": 12.0, "volumeKg": 30.0}]
        panel = {"panelStart": "2014-01-01", "panelEnd": "2016-03-31",
                 "products": [make_product(rows, pid) for pid in experiments.PRODUCT_IDS]}
        reference = {"schemaVersion": 1,
                     "products": [{"id": pid, "observations": [dict(row) for row in rows]}
                                  for pid in experiments.PRODUCT_IDS]}
        report = experiments.verify_target_equivalence(panel, reference)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["observationsByProduct"]["onion"], 1)
        reference["products"][0]["observations"][0]["actual"] += 1e-8
        experiments.verify_target_equivalence(panel, reference)
        reference["products"][0]["observations"][0]["volumeKg"] += 1
        with self.assertRaisesRegex(ValueError, "volumeKg mismatch"):
            experiments.verify_target_equivalence(panel, reference)
        reference["products"][0]["observations"][0]["volumeKg"] -= 1
        reference["products"][0]["observations"].clear()
        with self.assertRaisesRegex(ValueError, "date mismatch"):
            experiments.verify_target_equivalence(panel, reference)

    def test_holdout_claim_is_exclusive_and_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            development_file = Path(tmp) / "development.json"
            output_file = Path(tmp) / "holdout.json"
            development = {"protocolSha256": "protocol", "selectionFrozen": True}
            development_file.write_text(json.dumps(development), encoding="utf-8")
            claim = experiments._claim_holdout(development_file, output_file, development)
            with self.assertRaises(FileExistsError):
                experiments._claim_holdout(development_file, output_file, development)
            experiments._update_claim(claim, "failed", "SyntheticFailure")
            state = json.loads(claim.read_text(encoding="utf-8"))
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["errorType"], "SyntheticFailure")

            output_file.write_text('{"old":true}\n', encoding="utf-8")
            original = output_file.read_text(encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(experiments.main(["--db", str(Path(tmp)/"missing.sqlite"),
                                                   "--output", str(output_file)]), 2)
            self.assertEqual(output_file.read_text(encoding="utf-8"), original)

    def test_fixed_splits_and_holdout_manifest_cannot_reopen_selection(self):
        panel = {"products": [make_product([], pid) for pid in experiments.PRODUCT_IDS],
                 "panelStart": "2014-01-01", "panelEnd": "2016-03-31",
                 "marketCodes": list(experiments.MARKETS), "marketNames": list(experiments.MARKETS.values()),
                 "sourceUrls": [], "selectedJobIds": [], "selectedRequestCount": 0, "publicationTimeKnown": False}
        with self.assertRaises(ValueError):
            experiments.run_development(panel, dt.date(2016, 1, 1), dt.date(2016, 3, 31), 10, None)
        with self.assertRaises(ValueError):
            experiments.run_holdout(panel, {}, dt.date(2016, 1, 2), dt.date(2016, 3, 31), 10)
        development = experiments.run_development(panel, min_days=10, publication_path=None)
        self.assertTrue(development["selectionFrozen"])
        self.assertTrue(all(p["frozenWinner"]["model"] == "persistence" for p in development["products"]))
        holdout = experiments.run_holdout(panel, development, min_days=10, cadences=("monthly",))
        self.assertFalse(holdout["selectionReopened"])
        self.assertEqual(holdout["developmentManifestSha256"], experiments._json_hash(development))


if __name__ == "__main__":
    unittest.main()
