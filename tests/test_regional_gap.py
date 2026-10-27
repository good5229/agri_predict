import datetime as dt
import json
import sqlite3
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from agri_data import regional_gap as gap


def payload(sub="06", size_name="대", grade_name="특", pkg_name="망"):
    return {"gds_sclsf_cd": sub, "gds_sclsf_nm": "깐마늘",
            "grd_cd": "1", "grd_nm": grade_name, "sz_cd": "10", "sz_nm": size_name,
            "pkg_cd": "2", "pkg_nm": pkg_name}


def record(market, day, p, amount="1000", weight="10"):
    return {"amount": Decimal(amount), "weight": Decimal(weight), "rows": 1,
            "names": gap._names_for(p)}


class RegionalGapTests(unittest.TestCase):
    def test_dimension_names_must_be_present_compatible_and_nonplaceholder(self):
        p = payload()
        self.assertEqual(gap._name_status([p, dict(p)]), "compatible")
        self.assertEqual(gap._name_status([p, payload(size_name=".")]), "placeholder_dimension_name")
        self.assertEqual(gap._name_status([p, payload(size_name="중")]), "dimension_name_conflict")
        missing = payload(); missing["pkg_nm"] = None
        self.assertEqual(gap._name_status([p, missing]), "missing_dimension_name")

    def test_same_code_with_market_name_conflict_does_not_pair_primary_or_sensitivity(self):
        d="2018-04-02";codes=("06","1","10","2")
        cells={
            ("garlic",d,gap.GWANGJU,"06",codes):record(gap.GWANGJU,d,payload(size_name="대")),
            ("garlic",d,gap.GARAK,"06",codes):record(gap.GARAK,d,payload(size_name="중")),
        }
        report=gap.analyze_cells(cells)
        p=next(x for x in report["products"] if x["productId"]=="garlic")
        self.assertEqual(p["matchedDates"],0)
        self.assertEqual(p["excludedCellCounts"]["dimension_name_conflict"],1)
        self.assertEqual(p["marketSensitivities"]["gwangjuVsGarak"]["excludedCellCounts"]["dimension_name_conflict"],1)

    def test_pairing_combines_seoul_by_signed_sums_and_weights_daily_log_gap(self):
        d = "2018-04-02"
        cells = {
            ("garlic", d, gap.GWANGJU, "06", ("06", "1", "10", "2")): record(gap.GWANGJU, d, payload(), "2000", "20"),
            ("garlic", d, gap.GARAK, "06", ("06", "1", "10", "2")): record(gap.GARAK, d, payload(), "900", "9"),
            ("garlic", d, gap.GANGSEO, "06", ("06", "1", "10", "2")): record(gap.GANGSEO, d, payload(), "1100", "11"),
        }
        report = gap.analyze_cells(cells)
        product = next(x for x in report["products"] if x["productId"] == "garlic")
        self.assertEqual(product["matchedDates"], 1)
        self.assertAlmostEqual(product["pairedDailyLogGaps"][d], 0.0)
        self.assertEqual(product["matchedCells"], 1)

    def test_inference_gate_prevents_claim_from_small_sample(self):
        result = gap._infer({dt.date(2019, 1, 1): .01})
        self.assertEqual(result["status"], "inference_unavailable")
        self.assertIn("fewer_than_90_paired_dates", result["reasons"])

    def test_placeholder_cell_is_excluded_from_matched_estimate(self):
        d = "2018-04-02"
        cells = {
            ("garlic", d, gap.GWANGJU, "06", ("06", "1", "10", "2")): record(gap.GWANGJU, d, payload(size_name=".")),
            ("garlic", d, gap.GARAK, "06", ("06", "1", "10", "2")): record(gap.GARAK, d, payload(size_name=".")),
        }
        report = gap.analyze_cells(cells)
        product = next(x for x in report["products"] if x["productId"] == "garlic")
        self.assertEqual(product["matchedDates"], 0)
        self.assertEqual(product["excludedCellCounts"], {"placeholder_dimension_name": 1})
        self.assertEqual(product["excludedCellWeightKg"]["placeholder_dimension_name"], 10.0)

    def test_recorded_code_policy_keeps_missing_and_placeholder_labels_exploratory(self):
        d="2018-04-02";codes=("06","1","10","2")
        cells={}
        for market in (gap.GWANGJU,gap.GARAK,gap.GANGSEO):
            p=payload()
            if market==gap.GWANGJU:p["sz_nm"]=""
            elif market==gap.GARAK:p["sz_nm"]="."
            cells[("garlic",d,market,"06",codes)]=record(market,d,p,"1000","10")
        strict=gap.analyze_cells(cells)
        default=gap.analyze_cells(cells,name_policy="strict")
        exploratory=gap.analyze_cells(cells,name_policy="recorded_code")
        strict_product=next(p for p in strict["products"] if p["productId"]=="garlic")
        default_product=next(p for p in default["products"] if p["productId"]=="garlic")
        exp_product=next(p for p in exploratory["products"] if p["productId"]=="garlic")
        self.assertEqual(strict_product["matchedCells"],0)
        self.assertEqual(default_product["matchedCells"],strict_product["matchedCells"])
        self.assertNotIn("namePolicy",strict)
        self.assertNotIn("namePolicy",strict_product)
        self.assertEqual(exp_product["matchedCells"],1)
        self.assertEqual(exp_product["unknownLabelMatchedCells"],1)
        self.assertAlmostEqual(exp_product["unknownLabelMatchedCellShare"],1)
        self.assertAlmostEqual(exp_product["unknownLabelMatchedWeightKg"],10)
        self.assertAlmostEqual(exp_product["unknownLabelMatchedWeightShare"],1)
        self.assertEqual(exploratory["exploratoryPostHoc"],True)
        self.assertNotEqual(exploratory["protocolSha256"],strict["protocolSha256"])
        self.assertEqual(exp_product["inference"]["status"],"inference_unavailable")

    def test_recorded_code_policy_still_rejects_present_name_conflicts(self):
        d="2018-04-02";codes=("06","1","10","2")
        cells={
            ("garlic",d,gap.GWANGJU,"06",codes):record(gap.GWANGJU,d,payload(size_name="대")),
            ("garlic",d,gap.GARAK,"06",codes):record(gap.GARAK,d,payload(size_name="중")),
        }
        report=gap.analyze_cells(cells,name_policy="recorded_code")
        product=next(p for p in report["products"] if p["productId"]=="garlic")
        self.assertEqual(product["matchedCells"],0)
        self.assertEqual(product["excludedCellCounts"]["dimension_name_conflict"],1)
        self.assertEqual(product["unknownLabelMatchedCells"],0)
        self.assertEqual(product["unknownLabelMatchedWeightShare"],None)

    def test_recorded_code_allows_only_missing_and_exact_dot_placeholder_labels(self):
        d="2018-04-02";codes=("06","1","10","2")
        for label in ("", "."):
            cells={}
            for market in (gap.GWANGJU,gap.GARAK):
                p=payload()
                p["sz_nm"]=label if market==gap.GWANGJU else "대"
                cells[("garlic",d,market,"06",codes)]=record(market,d,p)
            report=gap.analyze_cells(cells,name_policy="recorded_code")
            product=next(p for p in report["products"] if p["productId"]=="garlic")
            self.assertEqual(product["matchedCells"],1,label)
            self.assertEqual(product["unknownLabelMatchedCells"],1,label)
        for label in ("-", "--", "?", "미상", "기타"):
            cells={}
            for market in (gap.GWANGJU,gap.GARAK):
                p=payload()
                p["sz_nm"]=label if market==gap.GWANGJU else "대"
                cells[("garlic",d,market,"06",codes)]=record(market,d,p)
            report=gap.analyze_cells(cells,name_policy="recorded_code")
            product=next(p for p in report["products"] if p["productId"]=="garlic")
            self.assertEqual(product["matchedCells"],0,label)
            self.assertEqual(product["excludedCellCounts"]["placeholder_dimension_name"],1,label)

    def test_recorded_code_policy_keeps_frozen_inference_and_holm_family(self):
        cells={};codes=("06","1","10","2")
        for i in range(365*2):
            day=(dt.date(2018,1,1)+dt.timedelta(days=i)).isoformat()
            p=payload()
            if i%2==0:p["sz_nm"]="."
            else:p["sz_nm"]=""
            cells[("garlic",day,gap.GWANGJU,"06",codes)]=record(gap.GWANGJU,day,p,"1000","10")
            seoul_amount="990" if i%2==0 else "800"
            cells[("garlic",day,gap.GARAK,"06",codes)]=record(gap.GARAK,day,p,seoul_amount,"10")
            cells[("garlic",day,gap.GANGSEO,"06",codes)]=record(gap.GANGSEO,day,p,seoul_amount,"10")
        strict=gap.analyze_cells(cells)
        recorded=gap.analyze_cells(cells,name_policy="recorded_code")
        product=next(p for p in recorded["products"] if p["productId"]=="garlic")
        strict_product=next(p for p in strict["products"] if p["productId"]=="garlic")
        self.assertEqual(strict_product["matchedDates"],0)
        self.assertEqual(product["matchedDates"],365*2)
        self.assertEqual(product["unknownLabelMatchedCells"],365*2)
        self.assertEqual(product["inference"]["status"],"available")
        self.assertEqual(product["inference"]["meanDifferenceHAC28"]["lagDays"],28)
        self.assertEqual(product["inference"]["equivalenceTOSTHAC9"]["hacDays"],9)
        self.assertEqual(product["inference"]["smallGapDays"]["blockDays"],28)
        self.assertEqual(len(recorded["multipleTesting"]),15)
        self.assertEqual(len(product["holmTests"]),3)
        self.assertIn("5 products x 3 tests",recorded["protocol"]["primary"]["holmFamily"])

    def test_invalid_dimension_code_cannot_pair_and_is_reported_in_pairwise_sensitivity(self):
        d="2018-04-02";bad=("06","1",None,"2")
        cells={
            ("garlic",d,gap.GWANGJU,"06",bad):record(gap.GWANGJU,d,payload()),
            ("garlic",d,gap.GARAK,"06",bad):record(gap.GARAK,d,payload()),
        }
        report=gap.analyze_cells(cells)
        product=next(x for x in report["products"] if x["productId"]=="garlic")
        self.assertEqual(product["matchedDates"],0)
        self.assertEqual(product["excludedCellCounts"]["missing_dimension_code"],1)
        self.assertEqual(product["marketSensitivities"]["gwangjuVsGarak"]["excludedCellCounts"]["missing_dimension_code"],1)

    def test_incomplete_market_scope_blocks_primary_comparison(self):
        d="2018-04-02";codes=("06","1","10","2")
        cells={
            ("garlic",d,gap.GWANGJU,"06",codes):record(gap.GWANGJU,d,payload()),
            ("garlic",d,gap.GARAK,"06",codes):record(gap.GARAK,d,payload()),
            ("garlic",d,gap.GANGSEO,"06",codes):record(gap.GANGSEO,d,payload()),
        }
        scope={f"garlic|{d}|{m}":["complete"] for m in gap.MARKETS}
        scope[f"garlic|{d}|{gap.GARAK}"]=["source_incomplete:failed"]
        report=gap.analyze_cells(cells,{"counts":{"source_incomplete:failed":1},"scopeStates":scope})
        product=next(x for x in report["products"] if x["productId"]=="garlic")
        self.assertEqual(product["matchedDates"],0)
        self.assertEqual(product["sourceIncompleteDates"],1)
        self.assertEqual(product["excludedCellCounts"]["source_incomplete"],1)

    def test_calendar_hac_and_bootstrap_gates_fail_closed(self):
        values={}
        for i in range(365*2):
            day=dt.date(2018,1,1)+dt.timedelta(days=i)
            values[day]=.01 if i%2==0 else .2
        inference=gap._infer(values)
        self.assertEqual(inference["status"],"available")
        self.assertEqual(inference["equivalenceTOSTHAC9"]["hacDays"],9)
        self.assertIn("circular",inference["smallGapDays"]["method"])
        constant_indicator={d:(.01 if i%2==0 else .02) for i,d in enumerate(values)}
        failed=gap._infer(constant_indicator)
        self.assertEqual(failed["status"],"inference_unavailable")
        self.assertIn("degenerate_indicator_bootstrap",failed["reasons"][0])

    def test_seasons_use_djf_mam_jja_son(self):
        season=gap._season_number
        self.assertEqual([season(dt.date(2020,m,1)) for m in (12,1,2,3,6,9)],[0,0,0,1,2,3])

    def test_scoped_extractor_keeps_failed_and_completed_request_provenance(self):
        db=sqlite3.connect(":memory:");db.row_factory=sqlite3.Row
        db.executescript("""
          CREATE TABLE scope(latest_job_id,selected_job_id,selected_epoch_id,source_url,task_key,query_json,selected_state,latest_state,selected_previous,dataset);
          CREATE TABLE source_requests(request_id,job_id,dataset,source_url,task_key,epoch_id,state,retrieved_at,published_at,version_published_at,response_sha256);
          CREATE TABLE normalized_records(request_id,row_no,payload_json);
        """)
        date="2018-04-02"
        for i,m in enumerate(gap.MARKETS):
            q={"cond[trd_clcln_ymd::EQ]":date,"cond[whsl_mrkt_cd::EQ]":m,
               "cond[gds_lclsf_cd::EQ]":"12","cond[gds_mclsf_cd::EQ]":"09","cond[gds_sclsf_cd::EQ]":"06"}
            state="failed" if m==gap.GARAK else "complete"
            selected=None if m==gap.GARAK else 1
            latest=state
            db.execute("INSERT INTO scope VALUES(?,?,?,?,?,?,?,?,?,'prices_origin')",(1,selected,"e1","url",f"task{m}",json.dumps(q),state if selected else None,latest,0))
            if selected:
                rid=i+1
                db.execute("INSERT INTO source_requests VALUES(?,1,'prices_origin','url',?,'e1','complete','2026-01-01',NULL,NULL,?)",(rid,f"task{m}",f"hash{rid}"))
                p={**q,"trd_clcln_ymd":date,"whsl_mrkt_cd":m,"gds_lclsf_cd":"12","gds_mclsf_cd":"09","gds_sclsf_cd":"06","gds_sclsf_nm":"깐마늘","grd_cd":"1","grd_nm":"특","sz_cd":"10","sz_nm":"대","pkg_cd":"2","pkg_nm":"망","trd_se":"경매","unit_nm":"kg","totprc":"1000","unit_tot_qty":"10"}
                if m==gap.GWANGJU:
                    p["sz_cd"]=None;p["sz_nm"]=""
                db.execute("INSERT INTO normalized_records VALUES(?,?,?)",(rid,1,json.dumps(p,ensure_ascii=False)))
        db.commit()
        cte="""WITH picked AS (SELECT latest_job_id,selected_job_id,selected_epoch_id,source_url,task_key,query_json,selected_state,latest_state,selected_previous,dataset FROM scope),
        request_pages AS (SELECT request_id,job_id,dataset,source_url,task_key,epoch_id,state,retrieved_at,published_at,version_published_at FROM source_requests)"""
        with patch.object(gap.market_extension,"_scoped_selection_cte",return_value=cte):
            cells,states,provenance,excluded,raw,eligible=gap._source_cells(db)
        self.assertEqual(states["counts"]["source_incomplete:failed"],1)
        self.assertEqual(provenance["requestCount"],2)
        self.assertEqual(set(provenance["responseHashes"]),{"1","3"})
        report=gap.analyze_cells(cells,states,provenance,excluded,raw,eligible)
        garlic=next(p for p in report["products"] if p["productId"]=="garlic")
        self.assertEqual(garlic["matchedDates"],0)
        self.assertEqual(garlic["sourceIncompleteDates"],1)
        self.assertIn(date,garlic["dailyCoverage"])
        self.assertEqual(garlic["dailyCoverage"][date]["matchedCells"],0)
        self.assertEqual(report["exclusions"]["garlic"]["missing_dimension_code"]["weightKg"],10.0)
        self.assertEqual(garlic["rawAggregateDailyLogGaps"][date],0.0)

    def test_matched_weight_share_uses_positive_code_cell_denominators(self):
        d="2018-04-02";codes=("06","1","10","2");negative=("06","1","20","2")
        cells={}
        weights={gap.GWANGJU:("10","-9"),gap.GARAK:("8","-9"),gap.GANGSEO:("4","-9")}
        for market,(positive,negative_weight) in weights.items():
            cells[("garlic",d,market,"06",codes)]=record(market,d,payload(),"1000",positive)
            cells[("garlic",d,market,"06",negative)]=record(market,d,payload(),"0",negative_weight)
        eligible={( "garlic",d,m,codes):{"weight":Decimal(pos)} for m,(pos,_) in weights.items()}
        eligible.update({("garlic",d,m,negative):{"weight":Decimal(neg)} for m,(_,neg) in weights.items()})
        report=gap.analyze_cells(cells,eligible_cells=eligible)
        p=next(x for x in report["products"] if x["productId"]=="garlic")
        coverage=p["dailyCoverage"][d]
        self.assertAlmostEqual(coverage["matchedShareOfGwangju"],1.0)
        self.assertAlmostEqual(coverage["matchedShareOfSeoul"],10/12)
        self.assertLessEqual(coverage["matchedShareOfGwangju"],1.0)
        self.assertLessEqual(coverage["matchedShareOfSeoul"],1.0)

    def test_cli_writes_atomic_json_and_markdown_and_protects_database(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);db=root/"db.sqlite";db.touch();output=root/"result.json"
            report=gap.analyze_cells({})
            with patch.object(gap,"run_analysis",return_value=report):
                self.assertEqual(gap.main(["--db",str(db),"--output",str(output)]),0)
            self.assertEqual(json.loads(output.read_text())["status"],"complete")
            self.assertTrue((root/"report.md").is_file())
            with self.assertRaisesRegex(ValueError,"must not overwrite"):
                gap.main(["--db",str(db),"--output",str(db)])

    def test_cli_selects_recorded_code_and_markdown_labels_it_exploratory(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);db=root/"db.sqlite";db.touch();output=root/"result.json"
            report=gap.analyze_cells({},name_policy="recorded_code")
            with patch.object(gap,"run_analysis",return_value=report) as run:
                self.assertEqual(gap.main(["--db",str(db),"--output",str(output),"--name-policy","recorded_code"]),0)
                run.assert_called_once_with(db.resolve(),"recorded_code")
            saved=json.loads(output.read_text())
            self.assertTrue(saved["exploratoryPostHoc"])
            self.assertIn("only missing labels or the exact '.' label",saved["protocol"]["pairing"])
            markdown=(root/"report.md").read_text()
            self.assertIn("post-hoc",markdown)
            self.assertIn("코드가 실제 크기·품질",markdown)
            self.assertIn("명칭 누락과 정확히 `.`인 경우만 허용",markdown)
            self.assertIn("평균 0차이 검정",markdown)
            self.assertIn("실무상 작은 차이 판정",markdown)


if __name__ == "__main__":
    unittest.main()
