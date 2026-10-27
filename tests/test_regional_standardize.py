import datetime as dt
import json
import math
import sqlite3
import tempfile
import unittest
from collections import defaultdict
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from agri_data import regional_standardize as std


def row(**updates):
    value = {"gds_sclsf_cd": "01", "gds_sclsf_nm": "배추", "grd_cd": "1", "grd_nm": "특",
        "sz_cd": "100", "sz_nm": "대", "pkg_cd": "11", "pkg_nm": "망",
        "corp_gds_item_nm": "쌈배추", "corp_gds_vrty_nm": "쌈배추", "unit_qty": "8.000",
        "unit": "8kg", "unit_nm": "kg", "trd_se": "경매"}
    value.update(updates)
    return value


def fixture():
    strata = defaultdict(std._empty_stratum)
    eligible = defaultdict(Decimal)
    excluded = defaultdict(lambda: defaultdict(lambda: {"rows": 0, "weightKg": Decimal(0)}))
    return strata, eligible, excluded


def add(strata, eligible, excluded, market, day, amount, weight, **payload):
    p = row(**payload)
    std._add_row(strata, eligible, excluded, "cabbage", day, market, p, Decimal(str(amount)), Decimal(str(weight)))


class RegionalStandardizeTests(unittest.TestCase):
    def test_exact_unit_strata_and_shared_weight_daily_then_equal_date_weighting(self):
        strata, eligible, excluded = fixture()
        # First day has two strata; the higher-weight one dominates that day's gap.
        # Mirrors the read-only 2018-04-02 sample: exact 8kg unit amount and
        # normalized 쌈배추 name match; size label '.' remains unknown.
        add(strata, eligible, excluded, std.GWANGJU, "2018-04-02", 100, 1, unit_qty="8.000", corp_gds_vrty_nm="쌈 배추", sz_nm=".")
        add(strata, eligible, excluded, std.GARAK, "2018-04-02", 300, 3, unit_qty="8", corp_gds_vrty_nm="쌈   배추", sz_nm=".")
        add(strata, eligible, excluded, std.GWANGJU, "2018-04-02", 1800, 9, unit_qty="10", corp_gds_vrty_nm="알배기")
        add(strata, eligible, excluded, std.GARAK, "2018-04-02", 1000, 10, unit_qty="10", corp_gds_vrty_nm="알배기")
        add(strata, eligible, excluded, std.GWANGJU, "2018-04-03", 2200, 20, unit_qty="8", corp_gds_vrty_nm="쌈 배추")
        add(strata, eligible, excluded, std.GARAK, "2018-04-03", 1000, 10, unit_qty="8", corp_gds_vrty_nm="쌈 배추")
        report = std.analyze_strata(strata, eligible_weights=eligible, exclusions=excluded)
        product = next(p for p in report["products"] if p["productId"] == "cabbage")
        self.assertEqual(product["matchedStrata"], 3)
        self.assertEqual(product["matchedDates"], 2)
        self.assertEqual(product["sizeUnknownPairedKg"], 1)
        first_day = math.log(2) * 9 / 10
        self.assertAlmostEqual(product["dailyLogGaps"]["2018-04-02"], first_day)
        self.assertAlmostEqual(product["dailyLogGaps"]["2018-04-03"], math.log(1.1))
        self.assertEqual(product["inference"]["status"], "inference_unavailable")
        self.assertAlmostEqual(product["descriptiveGeometricPriceRatio"], math.exp((first_day + math.log(1.1)) / 2))
        self.assertEqual(product["descriptiveResult"]["ciStatus"], "unavailable_inference_gate")
        self.assertEqual(product["descriptiveResult"]["pValueStatus"], "unavailable_inference_gate")
        self.assertAlmostEqual(product["inference"]["reasons"].count("fewer_than_90_paired_dates"), 1)
        # 8.000 and 8 are Decimal-equal; whitespace normalization is confined to corp names.
        self.assertEqual(len(strata), 3)

    def test_unknown_physical_size_is_counted_and_name_conflict_or_missing_excluded(self):
        strata, eligible, excluded = fixture()
        for market in (std.GWANGJU, std.GARAK):
            add(strata, eligible, excluded, market, "2018-04-02", 1000, 10, sz_nm=".")
        # Same code/quantity but a different corporate variety label conflicts across markets.
        add(strata, eligible, excluded, std.GWANGJU, "2018-04-03", 1000, 10, corp_gds_vrty_nm="청배추")
        add(strata, eligible, excluded, std.GARAK, "2018-04-03", 1000, 10, corp_gds_vrty_nm="쌈배추")
        # A missing required grade name is separately rejected.
        add(strata, eligible, excluded, std.GWANGJU, "2018-04-04", 1000, 10, grd_nm="")
        add(strata, eligible, excluded, std.GARAK, "2018-04-04", 1000, 10, grd_nm="")
        report = std.analyze_strata(strata, eligible_weights=eligible, exclusions=excluded)
        product = next(p for p in report["products"] if p["productId"] == "cabbage")
        self.assertEqual(product["matchedStrata"], 1)
        self.assertAlmostEqual(product["sizeUnknownPairedKg"], 10)
        self.assertAlmostEqual(product["sizeUnknownShareOfMatchedKg"], 1)
        self.assertEqual(product["excluded"]["missing_market_stratum"]["strata"], 2)
        self.assertEqual(product["excluded"]["missing_or_placeholder_meaningful_name"]["strata"], 1)

    def test_distinct_exact_varieties_with_shared_codes_are_separate_pairs(self):
        strata, eligible, excluded = fixture()
        for market in (std.GWANGJU, std.GARAK):
            add(strata, eligible, excluded, market, "2018-04-02", 1000, 10,
                unit_qty="8", corp_gds_item_nm="쌈배추", corp_gds_vrty_nm="쌈배추")
            add(strata, eligible, excluded, market, "2018-04-02", 2000, 5,
                unit_qty="8", corp_gds_item_nm="쌈배추", corp_gds_vrty_nm="적쌈배추")
        add(strata, eligible, excluded, std.GWANGJU, "2018-04-02", 3000, 5,
            unit_qty="8", corp_gds_item_nm="쌈배추", corp_gds_vrty_nm="외톨이 품종")
        report = std.analyze_strata(strata, eligible_weights=eligible)
        product = next(p for p in report["products"] if p["productId"] == "cabbage")
        self.assertEqual(product["matchedStrata"], 2)
        self.assertEqual(product["excluded"]["missing_market_stratum"]["strata"], 1)
        coverage = product["dailyCoverage"]["2018-04-02"]
        self.assertAlmostEqual(coverage["shareOfGwangjuEligibleKg"], .75)
        self.assertAlmostEqual(coverage["shareOfSeoulEligibleKg"], 1)

    def test_read_only_cabbage_sample_is_synthetic_exact_unit_unknown_size_regression(self):
        strata, eligible, excluded = fixture()
        # Synthetic fixture based only on the reported sample aggregates:
        # Gwangju 8kg 쌈배추 / 4,392kg vs Garak 8kg 쌈배추 / 22,824kg.
        add(strata, eligible, excluded, std.GWANGJU, "2018-04-02", 439200, 4392,
            unit_qty="8.000", corp_gds_vrty_nm="쌈배추", sz_nm=".")
        add(strata, eligible, excluded, std.GARAK, "2018-04-02", 2282400, 22824,
            unit_qty="8.000", corp_gds_vrty_nm="쌈배추", sz_nm=".")
        report = std.analyze_strata(strata, eligible_weights=eligible, exclusions=excluded)
        product = next(p for p in report["products"] if p["productId"] == "cabbage")
        self.assertEqual(product["matchedStrata"], 1)
        self.assertEqual(product["matchedKg"], 4392)
        self.assertEqual(product["sizeUnknownPairedKg"], 4392)
        daily = product["dailyCoverage"]["2018-04-02"]
        self.assertAlmostEqual(daily["shareOfGwangjuEligibleKg"], 1)
        self.assertAlmostEqual(daily["shareOfSeoulEligibleKg"], 4392 / 22824)

    def test_incomplete_scope_and_nonpositive_signed_cell_do_not_pair(self):
        strata, eligible, excluded = fixture()
        for market in (std.GWANGJU, std.GARAK):
            add(strata, eligible, excluded, market, "2018-04-02", 1000, 10)
        add(strata, eligible, excluded, std.GWANGJU, "2018-04-03", -1, 10)
        add(strata, eligible, excluded, std.GARAK, "2018-04-03", 1000, 10)
        states = {"scopeStates": {f"cabbage|2018-04-02|{m}": ["complete"] for m in std.MARKETS}}
        states["scopeStates"][f"cabbage|2018-04-02|{std.GANGSEO}"] = ["source_incomplete:failed"]
        states["scopeStates"].update({f"cabbage|2018-04-03|{m}": ["complete"] for m in std.MARKETS})
        report = std.analyze_strata(strata, states, eligible)
        product = next(p for p in report["products"] if p["productId"] == "cabbage")
        self.assertEqual(product["matchedStrata"], 0)
        self.assertEqual(product["excluded"]["source_incomplete"]["strata"], 1)
        self.assertEqual(product["excluded"]["nonpositive_net_stratum"]["strata"], 1)

    def test_positive_and_negative_distinct_strata_keep_coverage_bounded(self):
        strata, eligible, excluded = fixture()
        for market in (std.GWANGJU, std.GARAK):
            add(strata, eligible, excluded, market, "2018-04-02", 1000, 10, unit_qty="8")
            add(strata, eligible, excluded, market, "2018-04-02", -900, -9, unit_qty="10")
        report = std.analyze_strata(strata, eligible_weights=eligible)
        product = next(p for p in report["products"] if p["productId"] == "cabbage")
        coverage = product["dailyCoverage"]["2018-04-02"]
        self.assertEqual(product["matchedKg"], 10)
        self.assertEqual(product["excluded"]["nonpositive_net_eligible_source_group"]["strata"], 2)
        self.assertEqual(product["excluded"]["nonpositive_net_eligible_source_group"]["weightKg"], 18)
        self.assertEqual(product["excluded"]["nonpositive_net_eligible_source_group"]["signedNetWeightKg"], -18)
        self.assertEqual(coverage["gwangjuEligibleKg"], 10)
        self.assertEqual(coverage["seoulEligibleKg"], 10)
        self.assertLessEqual(coverage["shareOfGwangjuEligibleKg"], 1)
        self.assertLessEqual(coverage["shareOfSeoulEligibleKg"], 1)

    def test_negative_adjustments_net_within_same_source_attribute_group(self):
        strata, eligible, excluded = fixture()
        for market in (std.GWANGJU, std.GARAK):
            add(strata, eligible, excluded, market, "2018-04-02", 1000, 10, unit_qty="8")
            add(strata, eligible, excluded, market, "2018-04-02", 1000, 10, unit_qty="9")
            add(strata, eligible, excluded, market, "2018-04-02", -900, -9, unit_qty="9")
        report = std.analyze_strata(strata, eligible_weights=eligible)
        product = next(p for p in report["products"] if p["productId"] == "cabbage")
        coverage = product["dailyCoverage"]["2018-04-02"]
        self.assertEqual(product["matchedStrata"], 2)
        self.assertEqual(product["matchedKg"], 11)
        self.assertEqual(coverage["gwangjuEligibleKg"], 11)
        self.assertEqual(coverage["seoulEligibleKg"], 11)

    def test_eligible_denominator_includes_missing_grade_name_and_sums_all_market_weights(self):
        strata, eligible, excluded = fixture()
        add(strata, eligible, excluded, std.GWANGJU, "2018-04-02", 1000, 10)
        add(strata, eligible, excluded, std.GARAK, "2018-04-02", 1000, 10)
        # Insert the invalid-name market first to guard against early-exit loss.
        add(strata, eligible, excluded, std.GWANGJU, "2018-04-02", 10000, 100, grd_nm="")
        add(strata, eligible, excluded, std.GARAK, "2018-04-02", 20000, 200, grd_nm="")
        report = std.analyze_strata(strata, eligible_weights=eligible, exclusions=excluded)
        product = next(p for p in report["products"] if p["productId"] == "cabbage")
        coverage = product["dailyCoverage"]["2018-04-02"]
        self.assertEqual(coverage["matchedKg"], 10)
        self.assertEqual(coverage["gwangjuEligibleKg"], 110)
        self.assertEqual(coverage["seoulEligibleKg"], 210)
        self.assertAlmostEqual(coverage["shareOfGwangjuEligibleKg"], 10 / 110)
        self.assertAlmostEqual(coverage["shareOfSeoulEligibleKg"], 10 / 210)
        self.assertEqual(product["excluded"]["missing_or_placeholder_meaningful_name"]["weightKg"], 300)

    def test_missing_unit_quantity_or_dimension_code_remains_in_eligible_denominator(self):
        for invalid_field, expected_reason in (("unit_qty", "missing_or_invalid_unit_qty"), ("grd_cd", "missing_or_placeholder_dimension_code")):
            strata, eligible, excluded = fixture()
            for market in (std.GWANGJU, std.GARAK):
                add(strata, eligible, excluded, market, "2018-04-02", 1000, 10)
            add(strata, eligible, excluded, std.GWANGJU, "2018-04-02", 10000, 100, **{invalid_field: None})
            add(strata, eligible, excluded, std.GARAK, "2018-04-02", 20000, 200, **{invalid_field: None})
            report = std.analyze_strata(strata, eligible_weights=eligible, exclusions=excluded)
            product = next(p for p in report["products"] if p["productId"] == "cabbage")
            coverage = product["dailyCoverage"]["2018-04-02"]
            self.assertEqual(coverage["gwangjuEligibleKg"], 110)
            self.assertEqual(coverage["seoulEligibleKg"], 210)
            self.assertAlmostEqual(coverage["shareOfGwangjuEligibleKg"], 10 / 110)
            self.assertAlmostEqual(coverage["shareOfSeoulEligibleKg"], 10 / 210)
            self.assertEqual(product["sourceExclusions"][expected_reason]["rows"], 2)

    def test_cli_output_metadata_markdown_and_database_protection(self):
        report = std.analyze_strata({}, provenance={"selectedScopeCount": 0, "requestCount": 0})
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); db = root / "db.sqlite"; db.touch(); out = root / "out.json"
            with patch.object(std, "run_analysis", return_value=report):
                self.assertEqual(std.main(["--db", str(db), "--output", str(out)]), 0)
            parsed = json.loads(out.read_text())
            self.assertTrue(parsed["exploratoryPostHoc"])
            self.assertIn("exact Decimal unit_qty", parsed["protocol"]["stratum"])
            md = out.with_suffix(".md").read_text()
            self.assertIn("물리 크기 미확인", md)
            self.assertIn("HAC28", md)
            self.assertIn("Holm15 adjusted p", md)
            table_rows = [line for line in md.splitlines() if line.startswith("| cabbage |") or line.startswith("| radish |") or line.startswith("| onion |") or line.startswith("| garlic |") or line.startswith("| pepper |")]
            self.assertEqual(len(table_rows), 5)
            first_detail = next(i for i, line in enumerate(md.splitlines()) if line.startswith("- cabbage 제외"))
            last_row = max(i for i, line in enumerate(md.splitlines()) if line in table_rows)
            self.assertGreater(first_detail, last_row)
            with self.assertRaisesRegex(ValueError, "must not overwrite"):
                std.main(["--db", str(db), "--output", str(db)])

    def test_descriptive_ratio_and_date_count_render_when_gate_fails(self):
        strata, eligible, excluded = fixture()
        for market in (std.GWANGJU, std.GARAK):
            add(strata, eligible, excluded, market, "2018-04-02", 1100 if market == std.GWANGJU else 1000, 10)
        report = std.analyze_strata(strata, eligible_weights=eligible)
        product = next(p for p in report["products"] if p["productId"] == "cabbage")
        self.assertEqual(product["matchedDates"], 1)
        self.assertEqual(product["inference"]["status"], "inference_unavailable")
        self.assertAlmostEqual(product["descriptiveGeometricPriceRatio"], 1.1)
        self.assertEqual(product["descriptiveResult"]["ciStatus"], "unavailable_inference_gate")
        self.assertIsNone(product["descriptiveResult"]["priceRatioCI90"])
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report.json"; md = Path(td) / "report.md"
            std.write_report(report, out, md)
            cabbage_row = next(line for line in md.read_text().splitlines() if line.startswith("| cabbage |"))
            self.assertIn("1.100 (90% CI 추론 불가)", cabbage_row)

    def test_inference_gate_and_fixed_holm15_are_unchanged(self):
        strata, eligible, excluded = fixture()
        scope = {}
        for i in range(365 * 2):
            day = (dt.date(2018, 1, 1) + dt.timedelta(days=i)).isoformat()
            for market in std.MARKETS:
                scope[f"cabbage|{day}|{market}"] = ["complete"]
                if market == std.GWANGJU:
                    add(strata, eligible, excluded, market, day, 1000, 10)
                else:
                    amount = 990 if i % 2 == 0 else 800
                    add(strata, eligible, excluded, market, day, amount, 10)
        report = std.analyze_strata(strata, {"scopeStates": scope}, eligible, excluded)
        product = next(p for p in report["products"] if p["productId"] == "cabbage")
        self.assertEqual(product["inference"]["status"], "available")
        self.assertEqual(product["inference"]["meanDifferenceHAC28"]["lagDays"], 28)
        self.assertEqual(product["inference"]["equivalenceTOSTHAC9"]["hacDays"], 9)
        self.assertEqual(product["inference"]["smallGapDays"]["blockDays"], 28)
        self.assertEqual(len(report["multipleTesting"]), 15)

    def test_source_extractor_keeps_selected_hashes_and_scoped_states(self):
        db = sqlite3.connect(":memory:"); db.row_factory = sqlite3.Row
        db.executescript("""
          CREATE TABLE scope(latest_job_id,selected_job_id,selected_epoch_id,source_url,task_key,query_json,selected_state,latest_state,selected_previous,dataset);
          CREATE TABLE source_requests(request_id,job_id,dataset,source_url,task_key,epoch_id,state,retrieved_at,response_sha256);
          CREATE TABLE normalized_records(request_id,row_no,payload_json);
        """)
        d = "2018-04-02"
        for i, market in enumerate(std.MARKETS):
            q = {"cond[trd_clcln_ymd::EQ]": d, "cond[whsl_mrkt_cd::EQ]": market,
                 "cond[gds_lclsf_cd::EQ]": "10", "cond[gds_mclsf_cd::EQ]": "01"}
            db.execute("INSERT INTO scope VALUES(?,?,?,?,?,?,?,?,?,'prices_origin')", (1, 1, "e1", "url", f"t{i}", json.dumps(q), "complete", "complete", 0))
            db.execute("INSERT INTO source_requests VALUES(?,1,'prices_origin','url',?,'e1','complete','2026-01-01',?)", (i + 1, f"t{i}", f"hash{i}"))
            p = {**row(), **q, "trd_clcln_ymd": d, "whsl_mrkt_cd": market, "gds_lclsf_cd": "10", "gds_mclsf_cd": "01", "totprc": str(1000 + i), "unit_tot_qty": "10"}
            db.execute("INSERT INTO normalized_records VALUES(?,?,?)", (i + 1, 1, json.dumps(p, ensure_ascii=False)))
        db.commit()
        cte = """WITH picked AS (SELECT latest_job_id,selected_job_id,selected_epoch_id,source_url,task_key,query_json,selected_state,latest_state,selected_previous,dataset FROM scope),
        request_pages AS (SELECT request_id,job_id,dataset,source_url,task_key,epoch_id,state,retrieved_at FROM source_requests)"""
        with patch.object(std.market_extension, "_scoped_selection_cte", return_value=cte):
            strata, states, eligible, exclusions, provenance = std._source_strata(db)
        self.assertEqual(provenance["requestCount"], 3)
        self.assertEqual(set(provenance["responseHashes"].values()), {"hash0", "hash1", "hash2"})
        report = std.analyze_strata(strata, states, eligible, exclusions, provenance)
        product = next(p for p in report["products"] if p["productId"] == "cabbage")
        self.assertEqual(product["matchedStrata"], 1)
        db.close()


if __name__ == "__main__":
    unittest.main()
