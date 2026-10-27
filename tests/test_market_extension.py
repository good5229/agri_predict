import datetime as dt
import contextlib
import io
import json
import math
import sqlite3
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from agri_data import market_extension as ext


class MarketExtensionTests(unittest.TestCase):
    def _db(self, root):
        path = root / "fixture.sqlite"
        db = sqlite3.connect(path)
        db.executescript("""
        CREATE TABLE jobs(job_id INTEGER,command TEXT,config_json TEXT,state TEXT,started_at TEXT);
        CREATE TABLE source_requests(request_id INTEGER,job_id INTEGER,dataset TEXT,source_url TEXT,epoch_id TEXT,
          state TEXT,retrieved_at TEXT,expected_count INTEGER,returned_count INTEGER,page_no INTEGER,
          api_status TEXT,published_at TEXT,version_published_at TEXT,query_json TEXT,response_sha256 TEXT);
        CREATE TABLE planned_queries(job_id INTEGER,dataset TEXT,source_url TEXT,task_key TEXT,query_json TEXT,
          state TEXT,epoch_id TEXT,row_count INTEGER);
        CREATE TABLE request_attempts(job_id INTEGER,epoch_id TEXT,started_at TEXT);
        CREATE TABLE raw_records(request_id INTEGER,row_no INTEGER,payload_json TEXT);
        CREATE TABLE normalized_records(request_id INTEGER,row_no INTEGER,dataset TEXT,reference_date TEXT,payload_json TEXT,quality_flags TEXT);
        """)
        for jid, start, end in ((30,"2017-01-01","2018-12-31"),(31,"2019-01-01","2020-12-31")):
            config={"market":[ext.MARKET],"start":start,"end":end}
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,?)",(jid,"prices",json.dumps(config),"complete",start+"T00:00:00"))
        db.execute("INSERT INTO jobs VALUES(?,?,?,?,?)",(32,"prices",json.dumps({"market":["110001"],"start":"2017-01-01","end":"2018-12-31"}),"complete","2017-01-01T00:00:00"))
        query={"cond[trd_clcln_ymd::EQ]":"2017-01-02","cond[whsl_mrkt_cd::EQ]":ext.MARKET,
               "cond[gds_lclsf_cd::EQ]":"10","cond[gds_mclsf_cd::EQ]":"01",
               "pageNo":"1","numOfRows":"1000","returnType":"JSON"}
        query_text=json.dumps(query,separators=(",",":"))
        url="https://fixture.test/prices"
        task=ext.coverage._request_task_key("prices_origin",url,query_text)
        db.execute("INSERT INTO planned_queries VALUES(30,'prices_origin',?,?,?,?,?,?)",(url,task,query_text,"complete","",5))
        db.execute("INSERT INTO source_requests VALUES(1,30,'prices_origin',?,'','complete','2019-01-01T00:00:00',5,5,1,'OK',NULL,NULL,?,?)",
                   (url,query_text,"a"*64))
        # Signed correction is retained in the daily numerator and denominator.
        rows=[{"trd_se":"경매","unit_nm":"kg","gds_lclsf_cd":"10","gds_mclsf_cd":"01",
               "gds_sclsf_cd":"01","trd_clcln_ymd":"2017-01-02","whsl_mrkt_cd":ext.MARKET,"totprc":"200","unit_tot_qty":"2"},
              {"trd_se":"경매","unit_nm":"kg","gds_lclsf_cd":"10","gds_mclsf_cd":"01",
               "gds_sclsf_cd":"01","trd_clcln_ymd":"2017-01-02","whsl_mrkt_cd":ext.MARKET,"totprc":"-20","unit_tot_qty":"-1"},
              {"trd_se":"경매","unit_nm":"kg","gds_lclsf_cd":"10","gds_mclsf_cd":"01","gds_sclclsf_cd":"01",
               "gds_sclsf_cd":"01","trd_clcln_ymd":"2017-01-03","whsl_mrkt_cd":ext.MARKET,"totprc":"999","unit_tot_qty":"1"},
              {"trd_se":"경매","unit_nm":"kg","gds_lclsf_cd":"10","gds_mclsf_cd":"01",
               "gds_sclsf_cd":"01","trd_clcln_ymd":"2017-01-02","whsl_mrkt_cd":"110001","totprc":"999","unit_tot_qty":"1"},
              {"trd_se":"경매","unit_nm":"kg","gds_lclsf_cd":"11","gds_mclsf_cd":"01",
               "gds_sclsf_cd":"01","trd_clcln_ymd":"2017-01-02","whsl_mrkt_cd":ext.MARKET,"totprc":"999","unit_tot_qty":"1"}]
        for i,row in enumerate(rows):
            payload=json.dumps(row)
            db.execute("INSERT INTO normalized_records VALUES(?,?,'prices_origin','2017-01-02',?,?)",(1,i,payload,"[]"))
            db.execute("INSERT INTO raw_records VALUES(?,?,?)",(1,i,payload))
        core_query={"cond[trd_clcln_ymd::EQ]":"2017-01-02","cond[whsl_mrkt_cd::EQ]":"110001",
                    "cond[gds_lclsf_cd::EQ]":"10","cond[gds_mclsf_cd::EQ]":"01",
                    "pageNo":"1","numOfRows":"1000","returnType":"JSON"}
        core_text=json.dumps(core_query,separators=(",",":")); core_url="https://fixture.test/core"
        core_task=ext.coverage._request_task_key("prices_origin",core_url,core_text)
        db.execute("INSERT INTO planned_queries VALUES(32,'prices_origin',?,?,?,?,?,?)",(core_url,core_task,core_text,"complete","",1))
        db.execute("INSERT INTO source_requests VALUES(2,32,'prices_origin',?,'','complete','2019-01-01T00:00:00',1,1,1,'OK',NULL,NULL,?,?)",
                   (core_url,core_text,"b"*64))
        core_row={"trd_se":"경매","unit_nm":"kg","gds_lclsf_cd":"10","gds_mclsf_cd":"01",
                  "gds_sclsf_cd":"01","trd_clcln_ymd":"2017-01-02","whsl_mrkt_cd":"110001",
                  "totprc":"100","unit_tot_qty":"10"}
        db.execute("INSERT INTO normalized_records VALUES(2,0,'prices_origin','2017-01-02',?,?)",(json.dumps(core_row),"[]"))
        db.execute("INSERT INTO raw_records VALUES(2,0,?)",(json.dumps(core_row),))
        db.commit(); db.close()
        return path

    def test_gwangju_signed_rows_are_aggregated_and_job_scope_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            path=self._db(Path(temp)); db=sqlite3.connect(path); db.row_factory=sqlite3.Row
            prices, provenance=ext.extract_gwangju(db)
            self.assertEqual(prices["cabbage"][dt.date(2017,1,2)]["price"],180)
            self.assertEqual(prices["cabbage"][dt.date(2017,1,2)]["weight"],1)
            self.assertEqual(provenance["jobIds"],[30,31])
            later,later_prov=ext.extract_gwangju_period(db,dt.date(2017,1,1),dt.date(2017,1,3))
            self.assertEqual(later["cabbage"][dt.date(2017,1,2)]["price"],180)
            self.assertEqual(later_prov["jobIds"],[30])
            core,core_prov=ext._extract_core_market_prices(db)
            self.assertEqual(core["cabbage"]["110001"][dt.date(2017,1,2)],10)
            self.assertEqual(core_prov["selectedRequestCount"],1)
            self.assertEqual(sum(core_prov["selectedQueryStates"].values()),core_prov["selectedScopeCount"])
            core_new,new_prov,diagnostics=ext._extract_core_market_prices(db,dt.date(2017,1,2),dt.date(2017,1,2),True)
            self.assertEqual(core_new["cabbage"]["110001"][dt.date(2017,1,2)],10)
            self.assertEqual(diagnostics[("cabbage",dt.date(2017,1,2),"110001")]["eligibleAuctionKgRows"],1)
            broad_states,broad_prov=ext._core_broad_query_states(db,dt.date(2017,1,1))
            self.assertEqual(broad_states[("cabbage",dt.date(2017,1,2),ext.MARKET)]["selectedState"],"complete")
            self.assertEqual(broad_states[("cabbage",dt.date(2017,1,2),"110001")]["selectedState"],"complete")
            self.assertEqual(broad_prov["scopeCount"],2)
            db.close()

    def test_latest_prices_selector_matches_completed_run_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            def make_run(day,run_id,finished,state):
                directory=root/"outputs"/"daily"/day/run_id
                directory.mkdir(parents=True)
                (directory/"prices.json").write_text("{}")
                (directory/"run-status.json").write_text(json.dumps({"status":state,"runDirectory":str(directory.relative_to(root)),"finishedAtKst":finished}))
                return directory/"prices.json"
            make_run("2026-10-04","old","2026-10-04T13:00:00+09:00","complete")
            latest=make_run("2026-10-05","latest","2026-10-05T13:00:00+09:00","complete")
            make_run("2026-10-06","failed","2026-10-06T13:00:00+09:00","failed")
            self.assertEqual(ext.latest_prices_file(root).resolve(),latest.resolve())

    def test_model_features_source_age_gate_and_gwangju_market_differences(self):
        start=dt.date(2020,1,1)
        series={start:(100,20)}
        self.assertIsNone(ext.modeling._feature_row(series,start+dt.timedelta(days=15),start+dt.timedelta(days=16),1,{}))
        observed={start:(100,20),start+dt.timedelta(days=2):(999,50)}
        source,features=ext.modeling._feature_row(observed,start+dt.timedelta(days=1),start+dt.timedelta(days=2),1,{})
        self.assertEqual(source,start)
        self.assertAlmostEqual(features[0],math.log(100))
        self.assertAlmostEqual(ext.modeling._inverse_log_ratio(100,math.log(1.1))[0],110)
        features=ext._market_differences({"110001":{start:110},"110008":{start:90}},start,100,start)
        self.assertAlmostEqual(features[0],math.log(1.1))
        self.assertAlmostEqual(features[2],math.log(.9))

    def test_payload_must_match_query_and_specific_empty_scope_shadows_broad(self):
        query={"cond[trd_clcln_ymd::EQ]":"2017-01-02","cond[whsl_mrkt_cd::EQ]":ext.MARKET,
               "cond[gds_lclsf_cd::EQ]":"10","cond[gds_mclsf_cd::EQ]":"01"}
        payload={"gds_lclsf_cd":"10","gds_mclsf_cd":"01","trd_clcln_ymd":"2017-01-02",
                 "whsl_mrkt_cd":ext.MARKET,"gds_sclsf_cd":"01"}
        self.assertTrue(ext._valid_payload_scope(payload,query,"cabbage",ext.MARKET)[0])
        for key,value in (("trd_clcln_ymd","2017-01-03"),("whsl_mrkt_cd","110001"),("gds_lclsf_cd","11")):
            mismatch=dict(payload);mismatch[key]=value
            self.assertFalse(ext._valid_payload_scope(mismatch,query,"cabbage",ext.MARKET)[0])
        # `specific_scopes` is built from completed or empty selected queries, so
        # an empty specific query still shadows the matching broad response.
        specific={"cond[trd_clcln_ymd::EQ]":"2017-01-02","cond[whsl_mrkt_cd::EQ]":ext.MARKET,
                  "cond[gds_lclsf_cd::EQ]":"12","cond[gds_mclsf_cd::EQ]":"09",
                  "cond[gds_sclsf_cd::EQ]":"06"}
        scope=ext._selected_specific_scope(specific,"empty")
        self.assertEqual(scope,("2017-01-02",ext.MARKET,"06"))
        self.assertTrue(ext._overlap_shadowed("garlic","2017-01-02",ext.MARKET,"","06",{scope}))

    def test_holm_is_separate_for_each_hac_family_and_keeps_unavailable_as_one(self):
        records=[]
        for i in range(ext.FAMILY_SIZE):
            pvalue=0.001 if i==0 else None
            records.append({"tests":{"hac9":{"status":"available" if pvalue else "unavailable", "pValue":pvalue},
                            "hac28":{"status":"unavailable","pValue":None}}})
        ext._holm_fixed(records,"hac9"); ext._holm_fixed(records,"hac28")
        self.assertEqual(records[0]["tests"]["hac9"]["holmAdjustedPValue"],0.02)
        self.assertEqual(records[0]["tests"]["hac28"]["holmAdjustedPValue"],1.0)

    def test_unavailable_sensitivity_blocks_inference_and_significant_one_loss_worsening_is_visible(self):
        pair=[]
        for loss in ("MAE","MAPE"):
            pair.append({"n":40,"loss":loss,"meanLossDifference":1.0 if loss=="MAE" else 0.0,
                "tests":{"hac9":{"status":"available","holmSignificantAt005":loss=="MAE"},
                         "hac28":{"status":"available" if loss=="MAE" else "unavailable"}}})
        result=ext._pair_conclusion(pair)
        self.assertEqual(result["status"],"inference_unavailable")
        pair[1]["tests"]["hac28"]["status"]="available"
        result=ext._pair_conclusion(pair)
        self.assertEqual(result["status"],"mixed_evidence")
        self.assertEqual(result["significantlyWorsenedLosses"],["MAE"])

    def test_report_files_are_written_with_expected_ui_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            product={"id":"cabbage","name":"배추","coverage":{"commonTargets":40,"eligibleTargetDays":731,
                "excludedByReason":{"common_targets":40,"gwangju_missing_or_stale":10},"evaluationStart":"2019-01-01","evaluationEnd":"2020-12-31"},
                "metrics":{"candidate":{"MAE":1.2,"MAPE":2.3},"control":{"MAE":1.4,"MAPE":2.5},
                           "persistence":{"MAE":1.5,"MAPE":2.8}},
                "conclusions":{"control":{"status":"mixed_evidence","significantlyWorsenedLosses":["MAE"],"significantlyImprovedLosses":[]},
                               "persistence":{"status":"no_evidence","significantlyWorsenedLosses":[],"significantlyImprovedLosses":[]},"overall":"mixed_evidence"},"predictions":[]}
            comparison={"productId":"cabbage","comparator":"control","loss":"MAE","n":40,"candidateMean":1.2,
                "comparatorMean":1.4,"meanLossDifference":-0.2,"tests":{"hac9":{"status":"available","statistic":-2.1,"pValue":0.035,
                "holmAdjustedPValue":0.04,"holmSignificantAt005":True},"hac28":{"status":"unavailable_nonpositive_variance","statistic":None,
                "pValue":None,"holmAdjustedPValue":1.0,"holmSignificantAt005":False}}}
            report={"schemaVersion":1,"status":"complete","protocolSha256":"p-hash","inputSha256":"i-hash",
                    "protocol":ext.PROTOCOL,"products":[product],"comparisons":[comparison]}
            ext.write_report(report,root)
            self.assertEqual(json.loads((root/"report.json").read_text())["products"][0]["id"],"cabbage")
            markdown=(root/"report.md").read_text()
            self.assertIn("Evaluation targets: 2019-01-01 through 2020-12-31",markdown)
            self.assertNotIn("common_targets",markdown)
            for evidence in ("paired loss tests","MAPE","HAC9 primary","HAC28 sensitivity","Holm=0.0400",
                             "gwangju_missing_or_stale","significantly worsened losses: MAE","first publication time is unknown"):
                self.assertIn(evidence.lower(),markdown.lower())

    def test_revalidation_common_cutoff_stops_at_first_gap(self):
        start=dt.date(2023,1,1)
        states={}
        markets=(ext.MARKET,*ext.dashboard.MARKETS)
        for offset in range(5):
            day=start+dt.timedelta(days=offset)
            for product in ext.PRODUCTS:
                for market in markets:
                    state="failed" if offset==3 and product=="cabbage" and market==ext.MARKET else ("empty" if offset%2 else "complete")
                    states[(product,day,market)]={"selectedState":state,"latestState":state}
        cutoff,detail=ext._verified_common_cutoff(states,start,start+dt.timedelta(days=2))
        self.assertEqual(cutoff,start+dt.timedelta(days=2))
        self.assertEqual(detail["firstGapDate"],(start+dt.timedelta(days=3)).isoformat())
        self.assertEqual(detail["requiredBroadScopesPerDate"],15)

    def test_revalidation_cutoff_rejects_unverified_minimum(self):
        states={(p,dt.date(2023,1,1),m):{"selectedState":"empty"}
            for p in ext.PRODUCTS for m in (ext.MARKET,*ext.dashboard.MARKETS)}
        with self.assertRaisesRegex(ValueError,"earlier than required minimum"):
            ext._verified_common_cutoff(states,dt.date(2023,1,1),dt.date(2023,1,2))

    def test_new_inference_gate_needs_dates_years_and_three_seasons(self):
        dates=[]
        for year in (2023,2024):
            for month in (1,4,7,10):
                dates.extend(dt.date(year,month,day) for day in range(1,13))
        self.assertTrue(ext._coverage_gate(dates)["passed"])
        self.assertFalse(ext._coverage_gate(dates[:80])["passed"])
        self.assertFalse(ext._coverage_gate([dt.date(2023,1,1)+dt.timedelta(days=i) for i in range(90)])["passed"])

    def test_no_target_classification_keeps_empty_failure_and_nonpositive_distinct(self):
        day=dt.date(2024,1,3);product="cabbage"
        states={(product,day,m):{"selectedState":"empty","latestState":"complete"} for m in ext.dashboard.MARKETS}
        diagnostics={}
        self.assertEqual(ext._unobserved_target_reason(product,day,states,diagnostics),"verified_empty_response_closure_or_no_trade_unknown")
        states[(product,day,"110001")]={"selectedState":None,"latestState":"failed"}
        self.assertEqual(ext._unobserved_target_reason(product,day,states,diagnostics),"api_failure")
        states={(product,day,m):{"selectedState":"complete","latestState":"complete"} for m in ext.dashboard.MARKETS}
        self.assertEqual(ext._unobserved_target_reason(product,day,states,diagnostics),"no_eligible_auction_kg_rows")
        diagnostics[(product,day,"110001")]={"eligibleAuctionKgRows":1,"signedNetWeight":Decimal(-1),"signedNetAmount":Decimal(-5)}
        self.assertEqual(ext._unobserved_target_reason(product,day,states,diagnostics),"nonpositive_net_weight")

    def test_input_reconciliation_checks_common_target_snapshot_scope(self):
        day=dt.date(2023,1,1);states={};diagnostics={};core={p:{} for p in ext.PRODUCTS}
        for product in ext.PRODUCTS:
            for market in ext.dashboard.MARKETS:
                states[(product,day,market)]={"selectedState":"empty"}
        states[("cabbage",day,"110001")]={"selectedState":"complete"}
        diagnostics[("cabbage",day,"110001")]={"signedNetAmount":Decimal(200),"signedNetWeight":Decimal(2)}
        core["cabbage"][day]={"price":100.0,"volume":2.0}
        report=ext._validate_revalidation_input({"meta":{"contextStart":"2017-01-01","contextEnd":"2023-01-01"}},
            core,states,diagnostics,day)
        self.assertEqual(report["evaluationTargetsChecked"],len(ext.PRODUCTS))
        self.assertEqual(report["positiveSourceTargetsReconciled"],1)
        with self.assertRaisesRegex(ValueError,"precedes verified cutoff"):
            ext._validate_revalidation_input({"meta":{"contextStart":"2017-01-01","contextEnd":"2022-12-31"}},core,states,diagnostics,day)

    def test_revalidation_conclusion_uses_hac9_primary_and_reports_hac28_separately(self):
        pair=[{"n":100,"loss":loss,"meanLossDifference":-1.0,
            "tests":{"hac9":{"status":"available","holmSignificantAt005":True},
                     "hac28":{"status":"unavailable_nonpositive_variance","holmSignificantAt005":False}}}
            for loss in ("MAE","MAPE")]
        result=ext._revalidation_pair_conclusion(pair)
        self.assertEqual(result["status"],"improved")
        self.assertEqual(result["sensitivityBandwidth"],"HAC28")

    def test_revalidation_conclusions_use_holm20_and_constant_noise_is_unavailable(self):
        self.assertTrue(ext._constant_difference([1.0, 1.0 + 1e-14, 1.0 - 1e-14]))
        self.assertFalse(ext._constant_difference([1.0, 1.01]))
        records=[]
        for i in range(ext.REVALIDATION_FAMILY_SIZE):
            records.append({"productId":ext.PRODUCTS[i//4],"comparator":"control" if i%4<2 else "persistence",
                "loss":"MAE" if i%2==0 else "MAPE","n":100,"meanLossDifference":-1.0,
                "tests":{"hac9":{"status":"available","pValue":.003,"holmSignificantAt005":False},
                         "hac28":{"status":"available","pValue":.003,"holmSignificantAt005":False}}})
        ext._holm_fixed(records,"hac9")
        ext._holm_fixed(records,"hac28")
        pair=[x for x in records if x["productId"]==ext.PRODUCTS[0] and x["comparator"]=="control"]
        self.assertEqual(ext._revalidation_pair_conclusion(pair)["status"],"no_evidence")
        self.assertGreater(pair[0]["tests"]["hac9"]["holmAdjustedPValue"],.05)

    def test_revalidation_manifest_is_persisted_before_first_model_fit_and_counts_exclusions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);core_path=root/"prices.json";manifest=root/"analysis-manifest.json"
            start=dt.date(2022,9,1);cutoff=dt.date(2023,1,3)
            gwangju={p:{} for p in ext.PRODUCTS};product_rows=[]
            for product_index,product in enumerate(ext.PRODUCTS):
                product_start=start+dt.timedelta(days=20 if product_index else 0)
                observations=[]
                for offset in range((cutoff-product_start).days+1):
                    day=product_start+dt.timedelta(days=offset)
                    observations.append({"date":day.isoformat(),"actual":100+offset*.1,"volumeKg":10})
                    gwangju[product][day]={"price":100+offset*.1,"weight":10}
                product_rows.append({"id":product,"observations":observations})
            payload={"schemaVersion":1,"meta":{"contextStart":"2017-01-01","contextEnd":cutoff.isoformat()},
                "products":product_rows}
            core_path.write_text(json.dumps(payload))
            gwprov={"responseHashes":{},"selectedScopesSha256":"gw"}
            market_prices={p:{m:{} for m in ext.dashboard.MARKETS} for p in ext.PRODUCTS}
            market_prov={"selectedRequestHashes":{},"selectedScopesSha256":"market"}
            snapshot={"contextStart":"2017-01-01","contextEnd":cutoff.isoformat()}
            cutoff_detail={"coverageStart":"2023-01-01","coverageEnd":cutoff.isoformat()}
            db_file=root/"fixture.sqlite";db_file.touch()
            real_connect=sqlite3.connect
            def fit(*args):
                saved=json.loads(manifest.read_text())
                self.assertTrue(saved["frozenBeforeModelFit"])
                self.assertIn("cutoff",saved["inputs"])
                return 0.0
            with patch.object(ext.sqlite3,"connect",side_effect=lambda *a,**kw:real_connect(":memory:")), \
                 patch.object(ext,"_core_broad_query_states",return_value=({}, {"scopesSha256":"broad"})), \
                 patch.object(ext,"_common_revalidation_cutoff",return_value=(cutoff,cutoff_detail)), \
                 patch.object(ext,"extract_gwangju_period",return_value=(gwangju,gwprov)), \
                 patch.object(ext,"_extract_core_market_prices",return_value=(market_prices,market_prov,{})), \
                 patch.object(ext,"_validate_revalidation_input",return_value=snapshot), \
                 patch.object(ext,"_ridge_fit",side_effect=fit):
                report=ext.run_revalidation_analysis(core_path,db_file,manifest)
            saved=json.loads(manifest.read_text())
            self.assertEqual(report["analysisManifest"]["persistedBeforeModelFit"],True)
            self.assertEqual(report["analysisManifest"]["sha256"],ext._sha(manifest.read_bytes()))
            self.assertEqual(saved["inputs"]["cutoff"],cutoff_detail)
            cabbage=next(x for x in report["products"] if x["id"]==ext.PRODUCTS[0])
            later=next(x for x in report["products"] if x["id"]==ext.PRODUCTS[1])
            self.assertEqual(cabbage["coverage"]["initialTrainingDatesWithBothSources"],121)
            self.assertEqual(later["coverage"]["initialTrainingDatesWithBothSources"],101)
            self.assertEqual(cabbage["coverage"]["exclusionsByYearSeason"]["2023-winter"]["common_targets"],3)

    def test_revalidation_cli_refuses_legacy_output_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            old=root/"outputs/market-extension-20261005";old.mkdir(parents=True)
            marker=old/"report.json";marker.write_text("preserve")
            with self.assertRaises(SystemExit),contextlib.redirect_stderr(io.StringIO()):
                ext.main(["--root",str(root),"--prices",str(root/"prices.json"),"--period","revalidation-2023",
                          "--output-dir",str(old)])
            self.assertEqual(marker.read_text(),"preserve")

    def test_revalidation_cli_default_output_is_separate_from_legacy(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);capture={}
            with patch.object(ext,"run_revalidation_analysis",return_value={"synthetic":True}), \
                 patch.object(ext,"write_report",side_effect=lambda report,path:capture.update(path=path,report=report)):
                self.assertEqual(ext.main(["--root",str(root),"--prices",str(root/"prices.json"),
                    "--period","revalidation-2023"]),0)
            self.assertEqual(Path(capture["path"]).resolve(),(root/"outputs/market-extension-revalidation-2023").resolve())
            self.assertNotEqual(Path(capture["path"]),(root/"outputs/market-extension-20261005"))


if __name__ == "__main__": unittest.main()
