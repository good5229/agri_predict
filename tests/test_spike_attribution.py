import datetime as dt
import math
import json
import sqlite3
import unittest
from collections import defaultdict
from decimal import Decimal

from agri_data import collector, coverage, dashboard, diagnostics, spike_attribution as spike


def _raw_day(entries):
    rows = {"110001": [], "110008": []}
    for market, subclass, grade, amount, volume in entries:
        rows[market].append({"amount": Decimal(str(amount)), "volume": Decimal(str(volume)),
            "subclass": subclass, "grade": grade, "key": (market, subclass, grade)})
    return spike._daily_composition(rows)


class SpikeAttributionTests(unittest.TestCase):
    def test_real_price_sql_projection_extracts_validated_rows_and_request_provenance(self):
        db = collector.open_db(":memory:")
        db.row_factory = sqlite3.Row
        db.create_function("coverage_task_key", 3, coverage._request_task_key, deterministic=True)
        db.execute("INSERT INTO jobs(job_id,command,config_json,started_at,state) VALUES(1,'prices','{}','2020-01-01','complete')")
        url = "https://apis.data.go.kr/B552845/katOrigin/trades"
        day = "2018-12-31"
        for request_id, market in enumerate(dashboard.MARKETS, 1):
            base = {"cond[trd_clcln_ymd::EQ]": day, "cond[whsl_mrkt_cd::EQ]": market,
                    "cond[gds_lclsf_cd::EQ]": "10", "cond[gds_mclsf_cd::EQ]": "01"}
            task_key = collector.task_key("prices_origin", url, base)
            db.execute("INSERT INTO planned_queries(job_id,task_key,dataset,source_url,query_json,state,row_count,epoch_id) VALUES(1,?,'prices_origin',?,?, 'complete',1,'e1')",
                       (task_key, url, json.dumps(base)))
            params = {**base, "pageNo": "1", "numOfRows": "1000"}
            payload = {"trd_clcln_ymd": day, "whsl_mrkt_cd": market, "gds_lclsf_cd": "10",
                "gds_mclsf_cd": "01", "gds_sclsf_cd": "01", "gds_sclsf_nm": "결구배추",
                "grd_cd": "1", "grd_nm": "특", "trd_se": "경매", "unit_nm": "kg",
                "totprc": "10000", "unit_tot_qty": "100"}
            request_key = collector.stable_request_key("prices_origin", url, params, "e1")
            db.execute("""INSERT INTO source_requests(request_id,job_id,request_key,epoch_id,dataset,source_url,
                query_json,retrieved_at,response_sha256,http_status,api_status,expected_count,returned_count,
                page_no,state,published_at,version_published_at) VALUES(?,1,?,'e1','prices_origin',?,?,
                '2026-01-01T00:00:00Z',?,200,'0',1,1,1,'complete',NULL,NULL)""",
                (request_id, request_key, url, json.dumps(params), f"sha{request_id}"))
            db.execute("INSERT INTO raw_records(request_id,row_no,payload_json) VALUES(?,1,?)",
                       (request_id, json.dumps(payload)))
            db.execute("INSERT INTO normalized_records(request_id,row_no,dataset,reference_date,payload_json,quality_flags) VALUES(?,1,'prices_origin',?,?, '[]')",
                       (request_id, day, json.dumps(payload)))
        db.commit()
        result = spike.extract_prices(db, dt.date.fromisoformat(day), dt.date.fromisoformat(day))
        obs = result["series"]["cabbage"][dt.date.fromisoformat(day)]
        self.assertEqual(obs["actual"], 100.0)
        self.assertEqual(obs["volumeKg"], 200.0)
        self.assertEqual(obs["rowCount"], 2)
        self.assertEqual(len(result["provenance"]["cabbage"]["requestMetadata"]), 2)
        self.assertEqual(result["provenance"]["cabbage"]["publishedUnknownRequestCount"], 2)
        db.close()

    def test_midpoint_decomposition_with_entry_and_exit_reconciles_exactly(self):
        previous = {"invalidCells": [], "cells": {
            ("A",): {"share": 0.6, "price": 100.0},
            ("B",): {"share": 0.4, "price": 200.0}}}
        current = {"invalidCells": [], "cells": {
            ("A",): {"share": 0.3, "price": 120.0},
            ("C",): {"share": 0.7, "price": 300.0}}}
        result = spike.decompose_pair(previous, current, 140.0, 246.0)
        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["within"], 9.0)
        self.assertAlmostEqual(result["mix"], -33.0)
        self.assertAlmostEqual(result["entry"], 210.0)
        self.assertAlmostEqual(result["exit"], -80.0)
        self.assertAlmostEqual(result["priceDelta"], 106.0)
        self.assertAlmostEqual(result["within"]+result["mix"]+result["turnover"], 106.0)
        self.assertAlmostEqual(result["commonMassCoverage"], 0.3)

    def test_raw_cell_decomposition_preserves_missing_category_and_blocks_negative_cell(self):
        old = _raw_day([("110001", "A", "1", 6000, 60), ("110008", spike.MISSING, "2", 8000, 40)])
        new = _raw_day([("110001", "A", "1", 3600, 30), ("110008", "C", "3", 21000, 70)])
        result = spike._decompose_raw_pair(old, new, 140, 246)
        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["reconciliationResidual"], 0)
        self.assertTrue(any(spike.MISSING in contribution["cell"] for contribution in result["topCellContributions"]))
        bad = _raw_day([("110001", "A", "1", 100, 10), ("110008", "B", "2", -5, -1)])
        blocked = spike._decompose_raw_pair(old, bad, 140, 10)
        self.assertEqual(blocked["status"], "unavailable_nonpositive_or_invalid_cell")
        self.assertFalse(blocked["renormalized"])

    def test_market_level_decomposition_aggregates_grade_cells_separately(self):
        old = _raw_day([("110001", "A", "1", 5000, 50), ("110001", "A", "2", 5000, 50)])
        new = _raw_day([("110001", "A", "1", 6000, 50), ("110001", "A", "2", 6000, 50)])
        result = spike._decompose_raw_pair(old, new, 100, 120, market_only=True)
        self.assertEqual(result["status"], "available")
        self.assertAlmostEqual(result["within"], 20)
        self.assertAlmostEqual(result["mix"], 0)
        self.assertAlmostEqual(result["priceDelta"], 20)

    def test_event_selection_uses_previous_observation_gap_and_calendar_weather_windows(self):
        days = [dt.date(2018, 12, 31), dt.date(2019, 1, 1), dt.date(2019, 1, 2), dt.date(2019, 1, 5)]
        prices = [100.0, 130.0, 125.0, 180.0]
        series, market_rows, baseline, ridge, actuals = {}, {}, {}, {}, {}
        for day, price in zip(days, prices):
            entries = [("110001", "A", "1", price * 60, 60),
                       ("110008", "B", "2", price * 40, 40)]
            market_rows[day] = {"110001": [], "110008": []}
            for market, subclass, grade, amount, volume in entries:
                market_rows[day][market].append({"amount": Decimal(str(amount)), "volume": Decimal(str(volume)),
                    "subclass": subclass, "grade": grade})
            composition = spike._daily_composition(market_rows[day])
            series[day] = {"actual": price, "volumeKg": 100, "rowCount": 2, "marketData": composition["marketData"]}
            if day >= dt.date(2019, 1, 1):
                key = (day.isoformat(), 1)
                baseline[key] = {"predicted": 100.0}
                ridge[key] = {"predicted": 110.0}
                actuals[day] = price
        weather = {station: {} for station in diagnostics.STATIONS}
        events, selected, cutoff = spike._events(series, market_rows, days[1:], baseline, ridge, actuals, weather)
        self.assertEqual(len(events), 3)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["date"], "2019-01-05")
        jan1 = next(row for row in events if row["date"] == "2019-01-01")
        jan5 = next(row for row in events if row["date"] == "2019-01-05")
        self.assertEqual(jan1["gapDays"], 1)
        self.assertEqual(jan1["nextObservation"]["date"], "2019-01-02")
        self.assertEqual(jan5["gapDays"], 3)
        self.assertEqual(jan5["weatherLag7"]["100"]["sumRn_7day_sum"]["status"], "missing_calendar_day")
        self.assertTrue(math.isclose(cutoff, abs(math.log(180/125))))

    def test_consecutive_association_pairs_include_eval_start_context_and_no_gap_fill(self):
        series = {dt.date(2018, 12, 31): {"actual": 10, "volumeKg": 3},
                  dt.date(2019, 1, 1): {"actual": 11, "volumeKg": 4},
                  dt.date(2019, 1, 3): {"actual": 12, "volumeKg": 5}}
        composition = {
            dt.date(2018, 12, 31): {"marketShares": {"m": 1.0}, "gradeShares": {"g": 1.0}, "marketInvalid": False, "gradeInvalid": False},
            dt.date(2019, 1, 1): {"marketShares": {"m": 1.0}, "gradeShares": {"g": 1.0}, "marketInvalid": False, "gradeInvalid": False},
            dt.date(2019, 1, 3): {"marketShares": {"m": 1.0}, "gradeShares": {"g": 1.0}, "marketInvalid": False, "gradeInvalid": False}}
        rows, _ = spike._consecutive_pairs(series, composition, [dt.date(2019, 1, 1), dt.date(2019, 1, 3)])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["date"], dt.date(2019, 1, 1))

    def test_fixed_associations_contain_exactly_twelve_tests_and_two_holm_families(self):
        dates = [dt.date(2019, 1, 1) + dt.timedelta(days=i) for i in range(110)]
        prices, composition, weather, target_dates = {}, {}, {}, {}
        for product_i, product in enumerate(spike.PRODUCTS):
            prices[product] = {}
            composition[product] = {}
            target_dates[product] = dates[1:]
            for i, day in enumerate(dates):
                prices[product][day] = {"actual": 100 + i + product_i, "volumeKg": 100 + (i % 7)}
                composition[product][day] = {
                    "marketShares": {"110001": 0.5 + (i % 3)*0.01, "110008": 0.5 - (i % 3)*0.01},
                    "gradeShares": {"1": 0.6 + (i % 5)*0.01, "2": 0.4 - (i % 5)*0.01},
                    "marketInvalid": False, "gradeInvalid": False}
            for station_i, station in enumerate(diagnostics.STATIONS):
                weather[station] = {}
                for i, day in enumerate(dates):
                    v = float((i*i + 3*i + station_i*7) % 31)
                    weather[station][day] = {"avgTa": v, "maxTa": v+4, "minTa": v-3, "sumRn": float(i % 4)}
        results = spike._associations(prices, composition, weather, target_dates)
        self.assertEqual(len(results), 12)
        for item in results:
            self.assertIn("primary", item)
            self.assertIn("sensitivity", item)
            self.assertIn("holmAdjustedPValue", item["primary"])
            self.assertIn("holmAdjustedPValue", item["sensitivity"])
            self.assertEqual(item["eligibleConsecutivePairs"], 109)


if __name__ == "__main__":
    unittest.main()
