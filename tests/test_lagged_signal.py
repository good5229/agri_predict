import datetime as dt
import json
import math
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

from agri_data import lagged_signal as lag


KST = ZoneInfo("Asia/Seoul")


def _observation(day, price, volume=100.0, market_price=None, origin_shares=None):
    market_price = price if market_price is None else market_price
    origin_shares = origin_shares or {"250000": 1.0}
    return {"date": day.isoformat(), "actual": price, "volumeKg": volume, "rowCount": 2,
        "marketData": {"110001": {"price": market_price, "volumeKg": volume*0.6, "rowCount": 1,
            "qualityFlagCount": 0,
            "composition": {"subclass": {"distinctCategories": 2, "largestSignedVolumeShare": 0.7,
                            "concentrationHhi": 0.58, "signedVolumeShares": {"01": 0.7, "02": 0.3}},
                            "grade": {"distinctCategories": 1, "largestSignedVolumeShare": 1.0,
                            "concentrationHhi": 1.0, "signedVolumeShares": {"1": 1.0}}},
            "compositionFieldShares": {"plor_cd": origin_shares}}}}


def _archive(issue_at="2019-07-11T08:00:00+09:00"):
    published_at, available_at = "2019-07-11T08:05:00+09:00", "2019-07-11T08:10:00+09:00"
    return {"schemaVersion": 1,
        "sourceDocuments": [{"documentId": "doc1", "sourceUrl": "https://archive.test/forecast.json",
            "sourceSha256": "a"*64, "issuedAt": issue_at, "publishedAt": published_at,
            "availableAt": available_at, "retrievedAt": "2026-09-30T10:00:00+09:00",
            "issueTimeEvidence": {"verified": True, "verifiedIssuedAt": issue_at,
                "sourceUrl": "https://archive.test/issue-proof.pdf", "sha256": "b"*64},
            "publicationTimeEvidence": {"verified": True, "verifiedPublishedAt": published_at,
                "sourceUrl": "https://archive.test/publication-proof.pdf", "sha256": "e"*64},
            "availabilityTimeEvidence": {"verified": True, "verifiedAvailableAt": available_at,
                "sourceUrl": "https://archive.test/availability-proof.pdf", "sha256": "f"*64}}],
        "originLocationCrosswalk": [{"originCode": "250000", "locationKey": "hongcheon-grid",
            "verified": True, "codeValidityVerified": True,
            "evidence": {"sourceUrl": "https://archive.test/crosswalk.pdf", "sha256": "c"*64}}],
        "forecasts": [{"documentId": "doc1", "locationKey": "hongcheon-grid", "validDate": "2019-07-12",
            "element": "temperature_c", "value": 24.0, "leadHours": 24},
            {"documentId": "doc1", "locationKey": "hongcheon-grid", "validDate": "2019-07-12",
             "element": "rain_mm", "value": 12.0, "leadHours": 24}]}


class LaggedSignalTests(unittest.TestCase):
    def test_trade_features_use_latest_prior_selected_observation_and_age_gate(self):
        start = dt.date(2019, 1, 1)
        product = {"id": "cabbage", "observations": [_observation(start, 100),
            _observation(start+dt.timedelta(days=14), 110), _observation(start+dt.timedelta(days=16), 999)]}
        feature = lag.lagged_trade_features(product, start+dt.timedelta(days=15))
        self.assertEqual(feature["sourceDate"], start+dt.timedelta(days=14))
        self.assertEqual(feature["sourceAgeDays"], 1)
        self.assertEqual(len(feature["features"]), len(lag.lagged_trade_feature_names()))
        stale = lag.lagged_trade_features(product, start+dt.timedelta(days=31))
        self.assertIsNone(stale)
        same_day = {"id": "cabbage", "observations": [_observation(start, 100),
            _observation(start+dt.timedelta(days=1), 999)]}
        self.assertEqual(lag.lagged_trade_features(same_day, start+dt.timedelta(days=1))["sourceDate"], start)

    def test_volume_change_and_volatility_windows_are_asof_and_mark_sparse_windows(self):
        start = dt.date(2019, 1, 1)
        product = {"id": "cabbage", "observations": [
            _observation(start, 100, volume=10),
            _observation(start+dt.timedelta(days=2), 110, volume=30),
            _observation(start+dt.timedelta(days=3), 120, volume=9000)]}
        result = lag.lagged_trade_features(product, start+dt.timedelta(days=3))
        names = result["featureNames"]
        features = dict(zip(names, result["features"]))
        self.assertEqual(features["volume_window_3d_observed_days"], 2)
        self.assertEqual(features["volume_window_3d_has_pair"], 1)
        self.assertAlmostEqual(features["volume_window_3d_log_change"], math.log1p(30)-math.log1p(10))
        self.assertEqual(features["volume_window_28d_observed_days"], 2)
        sparse = lag.lagged_trade_features({"id": "cabbage", "observations": [_observation(start, 100, 10)]},
                                           start+dt.timedelta(days=1))
        sparse_features = dict(zip(sparse["featureNames"], sparse["features"]))
        self.assertEqual(sparse_features["volume_window_3d_has_pair"], 0)
        self.assertEqual(sparse_features["volume_window_3d_log_change"], 0)

    def test_future_transaction_values_do_not_change_origin_features(self):
        start = dt.date(2019, 1, 1)
        product = {"id": "radish", "observations": [_observation(start, 100),
            _observation(start+dt.timedelta(days=1), 110), _observation(start+dt.timedelta(days=2), 120)]}
        before = lag.lagged_trade_features(product, start+dt.timedelta(days=1))["features"]
        product["observations"][2] = _observation(start+dt.timedelta(days=2), 90000, volume=900000)
        after = lag.lagged_trade_features(product, start+dt.timedelta(days=1))["features"]
        self.assertEqual(before, after)

    def test_strict_price_control_excludes_volume_while_transaction_run_uses_it(self):
        from agri_data import experiments
        start = dt.date(2018, 1, 1)
        first = {"id": "cabbage", "observations": [_observation(start+dt.timedelta(days=i), 100+i, 100+i)
                                                       for i in range(100)]}
        scaled = {"id": "cabbage", "observations": [_observation(start+dt.timedelta(days=i), 100+i, (100+i)*10)
                                                        for i in range(100)]}
        origin = start+dt.timedelta(days=90)
        target = origin+dt.timedelta(days=1)
        core_a = lag._feature_row(first, experiments._series(first), origin, target, 1,
                                  "strict_asof_ridge", None, {})
        core_b = lag._feature_row(scaled, experiments._series(scaled), origin, target, 1,
                                  "strict_asof_ridge", None, {})
        self.assertEqual(core_a[1], core_b[1])
        tx_a = lag._feature_row(first, experiments._series(first), origin, target, 1,
                                "trade_only", None, {})
        tx_b = lag._feature_row(scaled, experiments._series(scaled), origin, target, 1,
                                "trade_only", None, {})
        self.assertNotEqual(tx_a[1], tx_b[1])

    def test_asof_model_and_trade_features_share_exact_14_day_source_age_gate(self):
        from agri_data import experiments
        origin = dt.date(2019, 2, 20)
        for age, expected in ((14, True), (15, False)):
            product = {"id": "radish", "observations": [
                _observation(origin-dt.timedelta(days=age), 100)]}
            row = lag._feature_row(product, experiments._series(product), origin,
                origin+dt.timedelta(days=1), 1, "strict_asof_ridge", None, {})
            self.assertEqual(row is not None, expected)

    def test_forecast_archive_requires_aware_verified_issue_timestamp_and_hashes(self):
        raw = _archive()
        data = json.dumps(raw).encode()
        archive = lag.ForecastArchive(raw, data)
        self.assertEqual(archive.provenance["evidencedDocumentCount"], 1)
        raw = _archive()
        raw["sourceDocuments"][0]["issuedAt"] = "2019-07-11T08:00:00"
        with self.assertRaises(ValueError):
            lag.ForecastArchive(raw, json.dumps(raw).encode())
        raw = _archive()
        raw["sourceDocuments"][0]["issueTimeEvidence"]["verified"] = False
        with self.assertRaises(ValueError):
            lag.ForecastArchive(raw, json.dumps(raw).encode())
        raw = _archive()
        raw["sourceDocuments"][0]["availabilityTimeEvidence"]["verified"] = False
        with self.assertRaises(ValueError):
            lag.ForecastArchive(raw, json.dumps(raw).encode())

    def test_issue_time_is_strict_and_features_use_only_verified_location_crosswalk(self):
        data = _archive()
        # A second issue exactly at the fixed 09:00 origin cutoff must be excluded.
        data["sourceDocuments"].append({"documentId": "doc2", "sourceUrl": "https://archive.test/later.json",
            "sourceSha256": "d"*64, "issuedAt": "2019-07-11T09:00:00+09:00",
            "publishedAt": "2019-07-11T09:00:00+09:00", "availableAt": "2019-07-11T09:00:00+09:00",
            "retrievedAt": "2026-09-30T10:01:00+09:00",
            "issueTimeEvidence": {"verified": True, "verifiedIssuedAt": "2019-07-11T09:00:00+09:00",
                "sourceUrl": "https://archive.test/later-proof.pdf", "sha256": "1"*64},
            "publicationTimeEvidence": {"verified": True, "verifiedPublishedAt": "2019-07-11T09:00:00+09:00",
                "sourceUrl": "https://archive.test/later-publication.pdf", "sha256": "2"*64},
            "availabilityTimeEvidence": {"verified": True, "verifiedAvailableAt": "2019-07-11T09:00:00+09:00",
                "sourceUrl": "https://archive.test/later-availability.pdf", "sha256": "3"*64}})
        data["forecasts"].append({"documentId": "doc2", "locationKey": "hongcheon-grid",
            "validDate": "2019-07-12", "element": "temperature_c", "value": -5.0, "leadHours": 0})
        archive = lag.ForecastArchive(data, json.dumps(data).encode())
        product = {"id": "radish", "observations": [_observation(dt.date(2019, 7, 10), 100,
            origin_shares={"250000": 0.75, "579000": 0.25})]}
        got = lag.forecast_feature_values(product, dt.date(2019, 7, 11), dt.date(2019, 7, 12), archive)
        self.assertEqual(got["status"], "available")
        self.assertEqual(got["features"][0], 24.0)
        self.assertEqual(got["features"][1], 0.75)
        self.assertEqual(got["features"][4], 12.0)
        self.assertEqual(got["provenance"][0]["documentId"], "doc1")
        self.assertEqual(got["originCutoffAt"], "2019-07-11T09:00:00+09:00")
        # Mapping the county to an unevidenced grid does not generate a weather feature.
        product["observations"][0]["marketData"]["110001"]["compositionFieldShares"]["plor_cd"] = {"579000": 1.0}
        unavailable = lag.forecast_feature_values(product, dt.date(2019, 7, 11), dt.date(2019, 7, 12), archive)
        self.assertEqual(unavailable["status"], "unavailable_no_asof_forecast_for_origin_regions")

    def test_issue_timestamp_alone_does_not_prove_availability_by_nine(self):
        raw = _archive(issue_at="2019-07-11T08:00:00+09:00")
        doc = raw["sourceDocuments"][0]
        doc["publishedAt"] = "2019-07-11T12:30:00+09:00"
        doc["availableAt"] = "2019-07-11T12:35:00+09:00"
        doc["publicationTimeEvidence"]["verifiedPublishedAt"] = doc["publishedAt"]
        doc["availabilityTimeEvidence"]["verifiedAvailableAt"] = doc["availableAt"]
        archive = lag.ForecastArchive(raw, json.dumps(raw).encode())
        self.assertIsNone(archive.issued_before(lag._origin_cutoff(dt.date(2019, 7, 11)),
            dt.date(2019, 7, 12), "hongcheon-grid", "temperature_c"))

    def test_holm_keeps_full_family_and_dm_honors_30_pair_gate(self):
        records = [{"test": {"pValue": 0.01 if i == 0 else 1.0}} for i in range(lag.FAMILY_SIZE)]
        lag._holm(records)
        self.assertEqual(len(records), 16)
        self.assertEqual(records[0]["test"]["holmAdjustedPValue"], 0.16)
        actuals = {( (dt.date(2020, 1, 1)+dt.timedelta(days=i)).isoformat(), 1): 100.0 for i in range(29)}
        candidate = [{"target": day, "horizon": 1, "predicted": 101.0} for day, _ in actuals]
        parent = [{"target": day, "horizon": 1, "predicted": 102.0} for day, _ in actuals]
        dm = lag._dm_checked(candidate, parent, actuals, "absolute_error", 9)
        self.assertEqual(dm["status"], "unavailable_insufficient_pairs")
        self.assertEqual(dm["pValue"], 1.0)

    def test_fixed_comparison_family_has_four_asof_contrasts_per_product(self):
        day = dt.date(2020, 1, 1)
        actuals, model_runs = {}, {}
        for product in lag.PRODUCTS:
            actuals[product] = {(day+dt.timedelta(days=i)).isoformat(): 100.0+i for i in range(35)}
            predictions = [{"origin": (day+dt.timedelta(days=i-1)).isoformat(),
                "target": (day+dt.timedelta(days=i)).isoformat(), "horizon": 1,
                "predicted": 101.0+i} for i in range(35)]
            model_runs[product] = {name: {"status": "available", "predictions": predictions}
                                   for name in (*lag.MODEL_RUNS, "asof_persistence")}
            model_runs[product][lag.CANDIDATES[1]] = {"status": "unavailable_forecast_archive", "predictions": []}
        records, conclusions = lag._comparison_records(model_runs, actuals, (day, day+dt.timedelta(days=34)))
        self.assertEqual(lag.FAMILY_SIZE, 16)
        self.assertEqual(len(records), 32)  # two separate HAC families
        self.assertEqual(len(conclusions), 16)
        self.assertEqual(set((r["candidateId"], r["comparatorId"]) for r in records), set(lag.PAIRINGS))
        withheld = [r for r in records if r["candidateId"] == lag.CANDIDATES[1]]
        self.assertEqual(len(withheld), 8)
        self.assertTrue(all(r["test"]["pValue"] == 1.0 and
                            r["test"]["status"] == "unavailable_forecast_archive" for r in withheld))

    def test_asof_persistence_never_uses_origin_date_market_close(self):
        start = dt.date(2019, 1, 1)
        product = {"observations": [_observation(start-dt.timedelta(days=1), 100), _observation(start, 900)]}
        run = lag._asof_persistence(product, start+dt.timedelta(days=1), start+dt.timedelta(days=1))
        self.assertEqual(run["predictions"][0]["sourceDate"], (start-dt.timedelta(days=1)).isoformat())
        self.assertEqual(run["predictions"][0]["predicted"], 100)

    def test_build_report_uses_frozen_observations_for_actuals_and_serializes(self):
        from agri_data import dashboard, experiments
        obs = {"date": "2014-01-03", "actual": 100.0, "volumeKg": 20.0, "rowCount": 1}
        origin_obs = {"date": "2019-01-01", "actual": 105.0, "volumeKg": 25.0, "rowCount": 1}
        eval_obs = {"date": "2019-01-02", "actual": 110.0, "volumeKg": 30.0, "rowCount": 1}
        historical_prediction = {"origin": "2019-01-01", "target": "2019-01-02", "horizon": 1,
            "predicted": 100.0, "sourceDate": "2019-01-01", "fitCutoff": "2019-01-01"}
        panel_products, frozen_products = [], []
        for product_id in dashboard.PRODUCTS:
            panel_products.append({"id": product_id, "observations": [dict(obs), dict(origin_obs), dict(eval_obs)], "diagnostics": {}})
            frozen_products.append({"id": product_id, "observations": [dict(obs), dict(origin_obs), dict(eval_obs)],
                "predictions": [dict(historical_prediction)],
                "runs": [{"modelId": model, "cadence": cadence,
                          "predictions": [dict(historical_prediction)] if model == "ridge" and cadence == "monthly" else []}
                    for model in lag.significance.MODEL_IDS for cadence in lag.significance.CADENCES]})
        panel = {"panelStart": "2014-01-01", "panelEnd": "2020-02-18", "products": panel_products}
        frozen = {"schemaVersion": 2,
            "meta": {"evaluationStart": "2017-01-01", "evaluationEnd": "2020-02-18"},
            "products": frozen_products}
        frozen_bytes = json.dumps(frozen).encode()
        def fake_fit(_product, _start, _end, mode, _archive):
            status = "unavailable_forecast_archive" if mode == "trade_forecast" else "available"
            return {"status": status, "predictions": [], "fits": [], "diagnostics": {}}
        with patch.object(lag, "_fit_candidate", side_effect=fake_fit):
            report = lag.build_report(panel, frozen, frozen_bytes)
        encoded = json.dumps(report, allow_nan=False)
        self.assertIn('"historicalRidgeReference"', encoded)
        self.assertEqual(report["splits"]["exploratoryEvaluation"]["fixedFamilySizePerHac"], 16)
        self.assertEqual(len(report["splits"]["exploratoryEvaluation"]["comparisons"]), 32)
        self.assertIn("excludedFromInference", report["historicalRidgeReference"]["cabbage"])
        self.assertEqual(report["historicalRidgeReference"]["cabbage"]["allSourceDateAtOrAfterOriginCount"], 1)
        self.assertEqual(report["historicalBaselineReference"]["cabbage"]["allSourceDateAtOrAfterOriginCount"], 1)
        rendered = lag.markdown_report(report)
        table_rows = [line for line in rendered.splitlines()
                      if line.startswith("| cabbage |") or line.startswith("| radish |")]
        self.assertEqual(len(table_rows), 16)  # 8 contrasts per split; each contrast displayed once

    def test_walk_forward_cutoff_excludes_same_and_future_target_labels(self):
        first = dt.date(2015, 1, 1)
        observations = [_observation(first+dt.timedelta(days=i), 100+i*.1, 100+i,
                       market_price=100+i*.1) for i in range(180)]
        product = {"id": "cabbage", "observations": observations}
        start, end = first+dt.timedelta(days=130), first+dt.timedelta(days=136)
        run = lag._fit_candidate(product, start, end, "trade_only", None, min_train_days=30)
        self.assertEqual(run["status"], "available")
        for fit in run["fits"]:
            if fit["status"] == "fitted":
                self.assertLess(fit["trainEnd"], fit["cutoff"])
                self.assertTrue(fit["labelsStrictlyBeforeCutoff"])
        self.assertTrue(all(row["tradeSourceDate"] < row["origin"] for row in run["predictions"]))
        self.assertTrue(all(row["sourceDate"] < row["origin"] for row in run["predictions"]))
        self.assertTrue(all(row["fitCutoff"] <= row["origin"] for row in run["predictions"]))


if __name__ == "__main__":
    unittest.main()
