import json
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from agri_data import collector as c


def portal(rows, total=None, code="0", msg="정상"):
    return json.dumps({"response":{"header":{"resultCode":code,"resultMsg":msg},
                       "body":{"items":{"item":rows},"totalCount":len(rows) if total is None else total}}},ensure_ascii=False)


class CollectorTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.db=c.open_db(str(Path(self.tmp.name)/"db.sqlite"))
        self.job=c.begin_job(self.db,"prices",{"start":"2018-01-01"})
        c._LAST_REQUEST_AT=0.0
        self.sleep_patch=patch.object(c.time,"sleep"); self.sleep_patch.start()
        self.addCleanup(self.sleep_patch.stop)

    def tearDown(self):
        self.db.close(); self.tmp.cleanup()

    def test_zero_rows_is_complete_and_cached(self):
        with patch.object(c,"curl_fetch",return_value=(200,portal([]),"")) as fetch:
            rows,total,_=c.request_page(self.db,self.job,"prices_origin","https://example.test/api",{"pageNo":"1"},"secret")
            again,_,_=c.request_page(self.db,self.job,"prices_origin","https://example.test/api",{"pageNo":"1"},"secret")
        self.assertEqual((rows,total,again),([],0,[])); fetch.assert_called_once()
        self.assertEqual(self.db.execute("select state,returned_count from source_requests").fetchone(),("complete",0))

    def test_api_error_never_becomes_empty_and_stops_retry(self):
        body=portal([],0,"30","SERVICE_KEY_IS_NOT_REGISTERED_ERROR")
        with patch.object(c,"curl_fetch",return_value=(200,body,"")) as fetch, patch.object(c.time,"sleep"):
            with self.assertRaisesRegex(c.CollectionError,"API failure 30"):
                c.request_page(self.db,self.job,"prices_origin","https://example.test",{},"mysecret")
        fetch.assert_called_once()
        self.assertEqual(self.db.execute("select state from source_requests").fetchone()[0],"failed")
        self.assertNotIn("mysecret",str(self.db.execute("select * from source_requests").fetchall()))

    def test_append_only_failure_does_not_change_previous_response(self):
        params={"pageNo":"1"}
        with patch.object(c,"curl_fetch",return_value=(200,portal([{"x":"saved"}]),"")):
            c.request_page(self.db,self.job,"x","https://example.test",params,"secret",epoch_id="same")
        old_text,old_state=self.db.execute("select response_text,state from source_requests").fetchone()
        auth=portal([],0,"30","not registered")
        with patch.object(c,"curl_fetch",return_value=(200,auth,"")):
            with self.assertRaises(c.CollectionError): c.request_page(self.db,self.job,"x","https://example.test",params,"secret",epoch_id="same",refresh=True)
        self.assertEqual(self.db.execute("select response_text,state from source_requests order by request_id limit 1").fetchone(),(old_text,old_state))
        self.assertEqual(self.db.execute("select count(*) from source_requests").fetchone()[0],2)

    def test_html_and_corrupt_json_are_not_success(self):
        for payload in ("<html>proxy error</html>","{broken"):
            with self.subTest(payload=payload), patch.object(c,"curl_fetch",return_value=(200,payload,"")), patch.object(c.time,"sleep"):
                with self.assertRaises(c.CollectionError):
                    c.request_page(self.db,self.job,"prices_origin","https://example.test",{"q":payload},"s",retries=1)
        states=[r[0] for r in self.db.execute("select state from source_requests")]
        self.assertEqual(states,["failed","failed"])

    def test_timeout_retries_are_bounded(self):
        with patch.object(c,"curl_fetch",return_value=(0,"","timed out")) as fetch, patch.object(c.time,"sleep"):
            with self.assertRaises(c.CollectionError):
                c.request_page(self.db,self.job,"x","https://example.test",{},"s",retries=3)
        self.assertEqual(fetch.call_count,3)

    def test_xml_import_and_kosis_json_shapes(self):
        xml="""<response><header><resultCode>00</resultCode><resultMsg>정상</resultMsg></header><body><items><item><hsCode>0703101000</hsCode><impWgt>0</impWgt></item></items></body></response>"""
        rows,count,status=c.decode_api("imports",xml)
        self.assertEqual((rows,count,status),([{"hsCode":"0703101000","impWgt":"0"}],1,"00"))
        kosis='[{"ITM_ID":"T06","TBL_ID":"DT_1ET0028","PRD_DE":"2016","DT":"-","C1":"00"}]'
        self.assertEqual(c.decode_api("production",kosis)[0][0]["DT"],"-")
        with self.assertRaisesRegex(c.CollectionError,"lacks body"):
            c.decode_api("prices_origin",'{"response":{"header":{"resultCode":"00"}}}')
        nodata='{"response":{"header":{"resultCode":"03","resultMsg":"NODATA_ERROR"}}}'
        self.assertEqual(c.decode_api("weather",nodata),([],0,"03:NODATA_ERROR"))
        no_data='{"response":{"header":{"resultCode":"03","resultMsg":"NO_DATA"}}}'
        self.assertEqual(c.decode_api("weather",no_data),([],0,"03:NO_DATA"))
        with self.assertRaises(c.CollectionError): c.decode_api("prices_origin",nodata)
        with self.assertRaises(c.CollectionError): c.decode_api("prices_origin",no_data)
        for code,msg in (("99","NO_DATA"),("01","NODATA_ERROR"),("03","NO_DATA ")):
            failed=json.dumps({"response":{"header":{"resultCode":code,"resultMsg":msg}}})
            with self.subTest(code=code,msg=msg), self.assertRaises(c.CollectionError):
                c.decode_api("weather",failed)
        mismatch="""<response><header><resultCode>00</resultCode><resultMsg>정상</resultMsg></header><body><items><item><hsCode>0703</hsCode></item></items><totalCount>2</totalCount></body></response>"""
        self.assertEqual(c.decode_api("imports",mismatch)[1],2)
        with self.assertRaisesRegex(c.CollectionError,"totalCount"):
            c.decode_api("prices_origin",'{"response":{"header":{"resultCode":"0"},"body":{"items":{"item":[]},"totalCount":"NaN"}}}')

    def test_curl_headers_do_not_trim_blank_lines_from_body(self):
        body="{\n\n  \"value\": 1\n}\n"
        stdout="HTTP/2 200 OK\r\nContent-Type: application/json\r\n\r\n"+body
        with patch.object(c.subprocess,"run",return_value=SimpleNamespace(returncode=0,stdout=stdout,stderr="")):
            status,received,_=c.curl_fetch("https://example.test")
        self.assertEqual(status,200); self.assertEqual(received,body)

    def test_pagination_complete_and_truncated(self):
        def fetch(url,timeout=25):
            page=int(__import__("urllib.parse").parse.parse_qs(__import__("urllib.parse").parse.urlsplit(url).query)["pageNo"][0])
            values=[{"n":"1"},{"n":"2"}] if page==1 else [{"n":"3"}]
            return 200,portal(values,3),""
        with patch.object(c,"curl_fetch",side_effect=fetch),patch.object(c.time,"sleep"):
            pages,rows=c.paged_collect(self.db,self.job,"x","https://example.test",{},"s",page_size=2)
        self.assertEqual((pages,rows),(2,3))
        job2=c.begin_job(self.db,"prices",{"start":"2020"})
        def truncated(url,timeout=25):
            page=int(__import__("urllib.parse").parse.parse_qs(__import__("urllib.parse").parse.urlsplit(url).query)["pageNo"][0])
            return 200,portal([{"n":"1"},{"n":"2"}] if page==1 else [],5),""
        with patch.object(c,"curl_fetch",side_effect=truncated),patch.object(c.time,"sleep"):
            with self.assertRaisesRegex(c.CollectionError,"truncated pagination"):
                c.paged_collect(self.db,job2,"x","https://example.test",{},"s",page_size=2)

    def test_page_totals_must_be_stable_within_epoch(self):
        def changed(url,timeout=25):
            query=__import__("urllib.parse").parse.parse_qs(__import__("urllib.parse").parse.urlsplit(url).query)
            page=int(query["pageNo"][0]); total=4 if page==1 else 5
            rows=[{"n":str(page*2-1)},{"n":str(page*2)}]
            return 200,portal(rows,total),""
        with patch.object(c,"curl_fetch",side_effect=changed):
            with self.assertRaisesRegex(c.CollectionError,"page count changed"):
                c.paged_collect(self.db,self.job,"x","https://example.test",{},"s",page_size=2,epoch_id="e1")

    def test_weather_request_page_size_does_not_exceed_asos_limit(self):
        args=SimpleNamespace(command="weather",start=c.dt.date(2020,1,1),end=c.dt.date(2020,1,1),
                              station=["108"],rate=100.0,refresh=False,_budget=c.RequestBudget(1))
        c.build_manifest(self.db,self.job,args)
        with patch.object(c,"curl_fetch",return_value=(200,portal([],0),"")) as fetch:
            c.collect_weather(self.db,self.job,args,"secret")
        request_url=fetch.call_args.args[0]
        query=__import__("urllib.parse").parse.parse_qs(__import__("urllib.parse").parse.urlsplit(request_url).query)
        self.assertLessEqual(int(query["numOfRows"][0]),999)

    def test_incomplete_logical_query_gets_new_epoch_on_resume(self):
        url=c.ASOS; params={"dataCd":"ASOS","dateCd":"DAY","dataType":"JSON","startDt":"20200101","endDt":"20200101","stnIds":"108"}
        self.db.execute("insert into planned_queries(job_id,task_key,dataset,source_url,query_json,state,epoch_id) values(?,?,?,?,?,'incomplete','old')",
                        (self.job,c.task_key("weather",url,params),"weather",url,c.json_text(params))); self.db.commit()
        seen=[]
        c.task_call(self.db,self.job,"weather",url,params,lambda epoch: seen.append(epoch) or 1)
        self.assertNotEqual(seen[0],"old")

    def test_retry_attempts_all_consume_shared_http_budget(self):
        with patch.object(c,"curl_fetch",side_effect=[(503,"busy",""),(200,portal([{"x":"ok"}]),"")]) as fetch:
            with self.assertRaises(c.BudgetExhausted):
                c.request_page(self.db,self.job,"x","https://example.test",{},"s",budget=c.RequestBudget(1),epoch_id="budget-test")
        fetch.assert_called_once()
        self.assertEqual(self.db.execute("select actual_http_attempts from jobs where job_id=?",(self.job,)).fetchone()[0],1)
        self.assertEqual(self.db.execute("select count(*) from request_attempts where job_id=?",(self.job,)).fetchone()[0],1)
        self.assertEqual(self.db.execute("select state from source_requests").fetchone()[0],"incomplete")

    def test_retry_success_records_each_http_attempt(self):
        with patch.object(c,"curl_fetch",side_effect=[(503,"busy",""),(200,portal([{"x":"ok"}]),"")]):
            c.request_page(self.db,self.job,"x","https://example.test",{},"s",budget=c.RequestBudget(2))
        self.assertEqual(self.db.execute("select actual_http_attempts from jobs where job_id=?",(self.job,)).fetchone()[0],2)
        self.assertEqual(self.db.execute("select count(*) from request_attempts").fetchone()[0],2)

    def test_resume_cache_and_refresh_create_versions(self):
        params={"pageNo":"1"}
        with patch.object(c,"curl_fetch",return_value=(200,portal([{"x":"one"}]),"")) as fetch:
            c.request_page(self.db,self.job,"x","https://example.test",params,"s")
            c.request_page(self.db,self.job,"x","https://example.test",params,"s")
        self.assertEqual(fetch.call_count,1)
        with patch.object(c,"curl_fetch",return_value=(200,portal([{"x":"two"}]),"")):
            c.request_page(self.db,self.job,"x","https://example.test",params,"s",refresh=True)
        self.assertEqual(self.db.execute("select count(*) from source_requests where state='complete'").fetchone()[0],2)
        self.assertEqual(self.db.execute("select count(*) from raw_records").fetchone()[0],2)

    def test_secrets_removed_from_url_and_response(self):
        with patch.object(c,"curl_fetch",return_value=(200,portal([{"echo":"secret"}]),"")):
            c.request_page(self.db,self.job,"x","https://example.test/path",{"pageNo":"1"},"secret")
        request=self.db.execute("select source_url,query_json,response_text from source_requests").fetchone()
        self.assertNotIn("secret",str(request))
        self.assertNotIn("secret",str(self.db.execute("select payload_json from raw_records").fetchone()))

    def test_decimal_quality_flags_preserve_original_values(self):
        row={"scsbd_prc":"7400","qty":"-2","totprc":"-14200","unit_qty":"1.250","unit_tot_qty":"2.50","scsbd_dt":None}
        flags=c.quality_flags("prices_origin",row)
        self.assertIn("negative:qty",flags)
        self.assertIn("amount_mismatch:scsbd_prc*qty!=totprc",flags)
        self.assertIn("missing:scsbd_dt",flags)
        self.assertEqual(row["qty"],"-2")
        self.assertEqual(flags.count("negative:unit_tot_qty"),0)
        imp=c.quality_flags("imports",{"year":"총계","hsCode":"-","impWgt":"-0.12345678901234567890"})
        self.assertIn("is_total_row",imp); self.assertIn("negative:impWgt",imp)
        decimal_body='{"response":{"header":{"resultCode":"0","resultMsg":"정상"},"body":{"items":{"item":[{"v":0.12345678901234567890}]},"totalCount":1}}}'
        exact=c.decode_api("prices_origin",decimal_body)[0][0]["v"]
        self.assertEqual(exact,"0.12345678901234567890")

    def test_manifest_keeps_unstarted_queries_pending(self):
        args=type("Args",(),{"command":"weather","start":c.dt.date(2020,1,1),"end":c.dt.date(2020,1,2),"station":["108"]})()
        c.build_manifest(self.db,self.job,args)
        self.assertEqual(self.db.execute("select count(*) from planned_queries").fetchone()[0],1)
        self.assertEqual(self.db.execute("select count(*) from planned_queries where state='pending'").fetchone()[0],1)

    def test_import_month_chunks_cover_partial_boundary_months_in_twelve_month_batches(self):
        start,end=c.dt.date(2020,1,31),c.dt.date(2021,1,1)
        chunks=list(c.month_chunks(start,end))
        self.assertEqual(chunks,[("202001","202012"),("202101","202101")])
        ordinals=[]
        for first,last in chunks:
            first_n=int(first[:4])*12+int(first[4:])-1
            last_n=int(last[:4])*12+int(last[4:])-1
            self.assertLessEqual(last_n-first_n+1,12)
            ordinals.extend(range(first_n,last_n+1))
        expected_start=start.year*12+start.month-1
        expected_end=end.year*12+end.month-1
        self.assertEqual(ordinals,list(range(expected_start,expected_end+1)))

        args=SimpleNamespace(command="imports",start=start,end=end,hs_code=["070310"])
        c.build_manifest(self.db,self.job,args)
        manifest=[tuple(json.loads(row[0])[key] for key in ("strtYymm","endYymm"))
                  for row in self.db.execute("select query_json from planned_queries order by task_key")]
        self.assertCountEqual(manifest,chunks)

    def test_import_chunk_policy_prevents_resuming_old_monthly_job(self):
        old=c.begin_job(self.db,"imports",{"start":"2020-01-01","end":"2020-12-31"})
        config={"start":"2020-01-01","end":"2020-12-31","month_chunk_months":12}
        self.assertNotEqual(c.resume_or_begin_job(self.db,"imports",config),old)

    def test_future_date_rejected(self):
        with self.assertRaises(Exception): c.iso_date("2999-01-01")

    def test_cli_date_config_serializes_before_any_http_call(self):
        dbpath=str(Path(self.tmp.name)/"cli.sqlite")
        args=c.make_parser().parse_args(["imports","--db",dbpath,"--start","2020-01-01","--end","2020-01-02","--max-requests","0"])
        with patch.dict("os.environ",{"DATA_GO_KR_SERVICE_KEY_DECODED":"test-api-key"}),patch.object(c,"curl_fetch") as fetch:
            result=c.execute(args)
        self.assertEqual(result,2); fetch.assert_not_called()
        db=c.open_db(dbpath,readonly=True)
        cfg=json.loads(db.execute("select config_json from jobs").fetchone()[0])
        self.assertEqual(cfg["start"],"2020-01-01")
        self.assertEqual(db.execute("select state from jobs").fetchone()[0],"incomplete")
        self.assertEqual(db.execute("select count(*) from planned_queries where state='pending'").fetchone()[0],5)
        self.assertEqual(db.execute("select count(*) from planned_queries where state='incomplete'").fetchone()[0],1)
        db.close()

    def test_cli_single_response_imports_and_production_validate_cleanly(self):
        from contextlib import redirect_stdout
        import io
        cases=(
            ("imports",["imports","--start","2020-01-01","--end","2020-01-31","--hs-code","070310"],
             "<response><header><resultCode>00</resultCode><resultMsg>정상</resultMsg></header><body><items><item><year>2020.01</year><hsCode>070310</hsCode><impWgt>1</impWgt></item></items><totalCount>1</totalCount></body></response>"),
            ("production",["production","--start","2020-01-01","--end","2020-12-31","--table","DT_1ET0028"],
             '[{"ITM_ID":"T06","TBL_ID":"DT_1ET0028","PRD_DE":"2020","DT":"12.5","C1":"00"}]'),
        )
        for name,argv,body in cases:
            with self.subTest(command=name):
                dbpath=str(Path(self.tmp.name)/(name+".sqlite"))
                args=c.make_parser().parse_args(argv+["--db",dbpath,"--max-requests","1"])
                url=c.CUSTOMS if name=="imports" else c.KOSIS
                with patch.dict("os.environ",{"DATA_GO_KR_SERVICE_KEY_DECODED":"test-portal-key","KOSIS_API_KEY":"test-kosis-key"}), \
                     patch.object(c,"curl_fetch",return_value=(200,body,"")) as fetch, \
                     redirect_stdout(io.StringIO()):
                    self.assertEqual(c.execute(args),0)
                self.assertEqual(fetch.call_count,1)
                with redirect_stdout(io.StringIO()):
                    validation=c.execute(c.make_parser().parse_args(["validate","--db",dbpath]))
                self.assertEqual(validation,0)
                db=c.open_db(dbpath,readonly=True)
                self.assertEqual(db.execute("select count(*) from source_requests where source_url=? and state='complete'",(url,)).fetchone()[0],1)
                self.assertEqual(db.execute("select state from jobs").fetchone()[0],"complete")
                db.close()

    def test_validate_checks_completed_epoch_and_row_links(self):
        args=SimpleNamespace(command="weather",start=c.dt.date(2020,1,1),end=c.dt.date(2020,1,1),station=["108"])
        c.build_manifest(self.db,self.job,args)
        url=c.ASOS; params={"dataCd":"ASOS","dateCd":"DAY","dataType":"JSON","startDt":"20200101","endDt":"20200101","stnIds":"108"}
        with patch.object(c,"curl_fetch",return_value=(200,portal([{"tm":"2020-01-01","stnId":"108"}]),"")):
            c.task_call(self.db,self.job,"weather",url,params,lambda epoch:c.paged_collect(self.db,self.job,"weather",url,params,"secret",epoch_id=epoch,budget=c.RequestBudget(1)))
        self.db.execute("update jobs set state='complete',completed_requests=1 where job_id=?",(self.job,)); self.db.commit()
        from contextlib import redirect_stdout
        import io
        with redirect_stdout(io.StringIO()): result=c.validate(self.db,self.job)
        self.assertEqual(result,0)
        self.db.execute("delete from normalized_records"); self.db.commit()
        with redirect_stdout(io.StringIO()): result=c.validate(self.db,self.job)
        self.assertEqual(result,1)

    def test_reversed_range_and_invalid_limits_rejected(self):
        args=SimpleNamespace(start=c.dt.date(2020,2,1),end=c.dt.date(2020,1,1),max_requests=1,rate=1)
        with self.assertRaisesRegex(c.CollectionError,"precedes"): c.validate_args(args)
        args.end=args.start; args.max_requests=-1
        with self.assertRaisesRegex(c.CollectionError,"zero or greater"): c.validate_args(args)
        with self.assertRaises(Exception): c.positive_float("0")

    def test_readonly_validate_rejects_missing_database_and_job(self):
        with self.assertRaises(c.CollectionError): c.open_db(str(Path(self.tmp.name)/"missing.sqlite"),readonly=True)
        with self.assertRaisesRegex(c.CollectionError,"does not exist"):
            c.validate(self.db,999)


if __name__=="__main__": unittest.main()
