import datetime as dt
import json
import sqlite3
import unittest
from decimal import Decimal

from agri_data import collector, coverage, dashboard, origin_linkage as audit


class OriginLinkageTests(unittest.TestCase):
    def test_station_mapping_requires_supported_code_and_raw_name(self):
        self.assertEqual(audit._station_match("250000", "홍천군" )["stationId"], "212")
        self.assertEqual(audit._station_match("579000", "부안군" )["stationId"], "243")
        self.assertEqual(audit._station_match("585000", "고창군" )["stationId"], "251")
        self.assertEqual(audit._station_match("232000", "평창군 대관령면" )["stationId"], "100")
        self.assertIsNone(audit._station_match("232000", "평창군"))
        self.assertIsNone(audit._station_match("250000", "시장 주소"))
        self.assertIsNone(audit._station_match("999999", "홍천군"))

    def test_signed_origin_weight_changes_retain_negative_adjustments(self):
        before = [
            {"originCode": "250000", "originNameRaw": "홍천군", "originLabelKind": "administrative_name_text_unverified",
             "amount": Decimal("28638000"), "weightKg": Decimal("286380"), "marketCode": "110001",
             "subclassCode": "01", "gradeCode": "1"},
            {"originCode": "579000", "originNameRaw": "부안군", "originLabelKind": "administrative_name_text_unverified",
             "amount": Decimal("-100"), "weightKg": Decimal("-10"), "marketCode": "110008",
             "subclassCode": "01", "gradeCode": "1"},
        ]
        after = [{**before[0], "amount": Decimal("14304000"), "weightKg": Decimal("143040")},
                 {**before[1], "amount": Decimal("500"), "weightKg": Decimal("5")}]
        result = audit._weight_contributions(before, after)
        by_code = {row["originCode"]: row for row in result["origins"]}
        self.assertEqual(by_code["250000"]["weightDeltaKg"], -143340.0)
        self.assertEqual(by_code["579000"]["weightDeltaKg"], 15.0)
        self.assertEqual(result["netWeightDeltaKg"], -143325.0)

    def test_weight_change_handles_entering_and_exiting_origins(self):
        old = [{"originCode": "old", "originNameRaw": "주소 미상", "originLabelKind": "unknown",
                "amount": Decimal("100"), "weightKg": Decimal("10"), "marketCode": "110001",
                "subclassCode": "01", "gradeCode": "1"}]
        new = [{"originCode": "new", "originNameRaw": "제주도", "originLabelKind": "administrative_name_text_unverified",
                "amount": Decimal("300"), "weightKg": Decimal("30"), "marketCode": "110008",
                "subclassCode": "01", "gradeCode": "2"}]
        result = audit._weight_contributions(old, new)
        rows = {row["originCode"]: row for row in result["origins"]}
        self.assertEqual(rows["old"]["afterWeightKg"], 0)
        self.assertEqual(rows["new"]["beforeWeightKg"], 0)

    def test_equal_one_day_origin_market_decomposition_reconciles_and_retains_signed_adjustment(self):
        def row(code, name, market, amount, kg):
            return {"originCode": code, "originNameRaw": name,
                "originLabelKind": "administrative_name_text_unverified", "marketCode": market,
                "amount": Decimal(str(amount)), "weightKg": Decimal(str(kg))}
        previous = [row("250000", "홍천군", "110001", 6000, 60),
                    row("232000", "평창군", "110008", 8000, 40)]
        event = [row("250000", "홍천군", "110001", 3600, 30),
                 row("250000", "홍천군", "110001", -500, -5),
                 row("245000", "삼척시", "110008", 21000, 70)]
        result = audit._one_day_origin_change(previous, event, dt.date(2019, 7, 26), dt.date(2019, 7, 27))
        self.assertEqual(result["comparisonBasis"], "one prior completed trading date vs one event date; equal one-day duration")
        self.assertEqual(result["wholeDayWeightDeltaKg"], -5.0)
        self.assertTrue(result["periods"]["previousTradingDay"]["reconciliation"]["originToMarket"])
        self.assertTrue(result["periods"]["eventDay"]["reconciliation"]["marketToWholeDay"])
        by_code = {item["originCode"]: item for item in result["originChanges"]}
        self.assertEqual(by_code["250000"]["weightDeltaKg"], -35.0)
        self.assertEqual(by_code["250000"]["markets"]["110001"]["afterRowCount"], 2)
        self.assertEqual(by_code["250000"]["markets"]["110001"]["afterWeightKg"], 25.0)
        self.assertEqual(by_code["232000"]["weightDeltaKg"], -40.0)
        self.assertEqual(by_code["245000"]["weightDeltaKg"], 70.0)
        self.assertEqual(sum(row["weightDeltaKg"] for row in result["originChanges"]), -5.0)

    def test_markdown_reports_equal_length_prior_day_and_event_day_comparison(self):
        previous = [{"originCode": "250000", "originNameRaw": "홍천군", "originLabelKind": "administrative_name_text_unverified",
                     "amount": Decimal("6000"), "weightKg": Decimal("60"), "marketCode": "110001",
                     "subclassCode": "01", "gradeCode": "1"}]
        current = [{**previous[0], "amount": Decimal("3000"), "weightKg": Decimal("30")}]
        one_day = audit._one_day_origin_change(previous, current, dt.date(2019, 10, 2), dt.date(2019, 10, 3))
        empty = audit._period([])
        event = {"productId": "radish", "eventDate": "2019-10-03", "periods": {
            "before7": empty, "eventDay": audit._period(current), "after7": empty},
            "coverage": {name: {"completeOrEmptyBothMarkets": 1, "calendarDays": 1}
                         for name in ("before7", "eventDay", "after7")},
            "topOriginWeightChanges": [], "oneDayPriorTradeVsEvent": one_day,
            "weather": {"limitedExactPlaceMatches": []}}
        text = audit.to_markdown({"protocol": {"frozenPredictionSha256": "a"*64}, "events": [event]})
        self.assertIn("직전 완료 거래일 대 사건일: 1일 대 1일", text)
        self.assertIn("2019-10-02 → 사건일 2019-10-03", text)
        self.assertIn("홍천군", text)
        self.assertIn("60.00", text)

    def test_administrative_labels_are_not_counted_as_unknown_or_address_rows(self):
        rows = []
        for code, name, weight in (("250000", "홍천군", "10"), ("(missing)", "(missing)", "2"),
                                   ("110001", "가락시장 내", "3")):
            code, name, kind = audit._origin(code, name)
            rows.append({"originCode": code, "originNameRaw": name, "originLabelKind": kind,
                "amount": Decimal("100"), "weightKg": Decimal(weight), "marketCode": "110001",
                "subclassCode": "01", "gradeCode": "1"})
        period = audit._period(rows)
        self.assertEqual(period["unknownOrAddressLikeRows"], 2)
        self.assertEqual(period["originLabelClassTotals"]["administrative_name_text_unverified"]["rowCount"], 1)
        self.assertEqual(period["originLabelClassTotals"]["unknown"]["rowCount"], 1)

    def test_weather_match_list_excludes_unmapped_origins(self):
        names = [("250000", "홍천군", "administrative_name_text_unverified"),
                 ("232000", "평창군", "administrative_name_text_unverified"),
                 ("999999", "시장 주소", "address_or_facility_text_unverified")]
        matches = audit._matched_origins(names, {"212": {}}, {"212": {}}, dt.date(2019, 7, 12))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["stationId"], "212")

    def test_weather_lags_exclude_event_day_and_do_not_zero_fill_missing_rain(self):
        event = dt.date(2019, 7, 12)
        records = {event-dt.timedelta(days=i): {"avgTa": 20.0+i, "sumRn": 1.0} for i in range(1, 8)}
        records[event] = {"avgTa": -99.0, "sumRn": 999.0}
        records[event-dt.timedelta(days=2)]["sumRn"] = None
        result = audit._weather_lags({"243": records}, event)["243"]["lagBeforeEvent"]
        self.assertEqual(result["1"]["meanAvgTaC"], 21.0)
        self.assertEqual(result["7"]["temperatureValidN"], 7)
        self.assertEqual(result["7"]["sumRainMm"], None)
        self.assertEqual(result["7"]["rainValidN"], 6)

    def test_price_sql_extraction_uses_selected_signed_auction_kg_rows_and_request_metadata(self):
        db = collector.open_db(":memory:")
        db.row_factory = sqlite3.Row
        db.create_function("coverage_task_key", 3, coverage._request_task_key, deterministic=True)
        db.execute("INSERT INTO jobs(job_id,command,config_json,started_at,state) VALUES(1,'prices','{}','2020-01-01','complete')")
        url = f"{collector.PORTAL}/B552845/katOrigin/trades"
        day = "2019-07-12"
        for request_id, market in enumerate(dashboard.MARKETS, 1):
            base = {"cond[trd_clcln_ymd::EQ]": day, "cond[whsl_mrkt_cd::EQ]": market,
                    "cond[gds_lclsf_cd::EQ]": "11", "cond[gds_mclsf_cd::EQ]": "01"}
            task_key = collector.task_key("prices_origin", url, base)
            db.execute("""INSERT INTO planned_queries(job_id,task_key,dataset,source_url,query_json,state,row_count,epoch_id)
                           VALUES(1,?,'prices_origin',?,?,'complete',1,'e1')""",
                       (task_key, url, json.dumps(base)))
            payload = {"trd_clcln_ymd": day, "whsl_mrkt_cd": market,
                "whsl_mrkt_nm": dashboard.MARKETS[market], "gds_lclsf_cd": "11", "gds_mclsf_cd": "01",
                "gds_sclsf_cd": "06", "gds_sclsf_nm": "무", "grd_cd": "1", "grd_nm": "특",
                "trd_se": "경매", "unit_nm": "kg", "totprc": "-100", "unit_tot_qty": "-2",
                "plor_cd": "585000", "plor_nm": "고창군"}
            params = {**base, "pageNo": "1", "numOfRows": "1000"}
            request_key = collector.stable_request_key("prices_origin", url, params, "e1")
            db.execute("""INSERT INTO source_requests(request_id,job_id,request_key,epoch_id,dataset,source_url,
                query_json,retrieved_at,response_sha256,http_status,api_status,expected_count,returned_count,
                page_no,state,published_at,version_published_at) VALUES(?,1,?,'e1','prices_origin',?,?,'2026-01-01Z',?,200,'0',1,1,1,'complete',NULL,NULL)""",
                (request_id, request_key, url, json.dumps(params), f"hash-{request_id}"))
            raw = json.dumps(payload, ensure_ascii=False)
            db.execute("INSERT INTO raw_records(request_id,row_no,payload_json) VALUES(?,1,?)", (request_id, raw))
            db.execute("""INSERT INTO normalized_records(request_id,row_no,dataset,reference_date,payload_json,quality_flags)
                           VALUES(?,1,'prices_origin',?,?,'[]')""", (request_id, day, raw))
        db.commit()
        rows, states, requests, excluded, query_requests = audit._extract(
            db, dt.date.fromisoformat(day), dt.date.fromisoformat(day))
        self.assertEqual(sum(len(v) for v in rows["radish"].values()), 2)
        self.assertEqual(sum((r["weightKg"] for r in rows["radish"][dt.date.fromisoformat(day)]), Decimal(0)), Decimal("-4"))
        self.assertTrue(all(r["originCode"] == "585000" and r["originNameRaw"] == "고창군" for r in rows["radish"][dt.date.fromisoformat(day)]))
        self.assertEqual(len(requests["radish"]), 2)
        self.assertTrue(all(meta["publicationTimeUnknown"] for meta in requests["radish"].values()))
        self.assertEqual(len(query_requests["radish"][dt.date.fromisoformat(day)]), 2)
        self.assertEqual(states["radish"][(day, "110001")], "complete")
        self.assertEqual(dict(excluded["radish"]), {})
        db.close()

    def test_asos_epoch_selection_preserves_hash_and_uses_only_valid_complete_epoch(self):
        db = collector.open_db(":memory:")
        db.row_factory = sqlite3.Row
        db.execute("INSERT INTO jobs(job_id,command,config_json,started_at,state) VALUES(1,'weather','{}','2020-01-01','complete')")
        params = {"dataCd": "ASOS", "dateCd": "DAY", "dataType": "JSON", "startDt": "20190701",
                  "endDt": "20191004", "stnIds": "212"}
        url = collector.ASOS
        task = collector.task_key("weather", url, params)
        db.execute("""INSERT INTO planned_queries(job_id,task_key,dataset,source_url,query_json,state,row_count,epoch_id)
                       VALUES(1,?,'weather',?,?,'complete',1,'w1')""", (task, url, json.dumps(params),))
        page_params = {**params, "pageNo": "1", "numOfRows": "999"}
        request_key = collector.stable_request_key("weather", url, page_params, "w1")
        db.execute("""INSERT INTO source_requests(request_id,job_id,request_key,epoch_id,dataset,source_url,query_json,
            retrieved_at,response_sha256,http_status,api_status,expected_count,returned_count,page_no,state,
            published_at,version_published_at) VALUES(1,1,?,'w1','weather',?,?,'2026-09-30T00:00:00Z','weather-hash',200,'NORMAL_SERVICE',1,1,1,'complete',NULL,NULL)""",
            (request_key, url, json.dumps(page_params)))
        payload = {"stnId": "212", "tm": "2019-07-11", "avgTa": "18.5", "maxTa": "22.0", "minTa": "15.0", "sumRn": ""}
        raw = json.dumps(payload)
        db.execute("INSERT INTO raw_records(request_id,row_no,payload_json) VALUES(1,1,?)", (raw,))
        db.execute("""INSERT INTO normalized_records(request_id,row_no,dataset,reference_date,payload_json,quality_flags)
                       VALUES(1,1,'weather','2019-07-11',?,'[]')""", (raw,))
        db.commit()
        weather, provenance, report = audit._select_weather(db, dt.date(2019, 7, 5), dt.date(2019, 7, 20))
        self.assertEqual(weather["212"][dt.date(2019, 7, 11)]["avgTa"], 18.5)
        self.assertIsNone(weather["212"][dt.date(2019, 7, 11)]["sumRn"])
        self.assertEqual(provenance["212"][dt.date(2019, 7, 11)]["responseSha256"], "weather-hash")
        self.assertEqual(report["selectedRequestIds"], [1])
        self.assertEqual(report["publicationTimeUnknownRequestCount"], 1)
        self.assertEqual(report["selectedRequests"][0]["retrievedAt"], "2026-09-30T00:00:00Z")
        self.assertTrue(report["selectedRequests"][0]["publicationTimeUnknown"])
        db.close()


if __name__ == "__main__":
    unittest.main()
