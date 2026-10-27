import datetime as dt
import hashlib
import json

import pytest

from agri_data import transaction_dispersion_pilot as pilot
from agri_data import collector, coverage


def test_weighted_quantiles_use_lot_weight_and_positive_rows_only():
    rows=[("10","1"),("200","10"),("0","2"),("-4","1"),("3","-1")]
    assert pilot.weighted_quantile(rows,.25)==pytest.approx(20)
    assert pilot.weighted_quantile(rows,.5)==pytest.approx(20)
    assert pilot.weighted_quantile(rows,.75)==pytest.approx(20)


def test_negative_adjustment_kept_but_excluded_from_quantile_feature():
    day=dt.date(2024,8,1); rows=[]
    for i in range(5): rows.append({"product":"onion","tradeDate":day.isoformat(),"amountRaw":str(100+i*10),"weightRaw":"1"})
    rows.append({"product":"onion","tradeDate":day.isoformat(),"amountRaw":"-50.00","weightRaw":"-0.5"})
    daily={("onion",day.isoformat()):200.0}
    vec,audit=pilot.dispersion_features(rows,daily,"onion",dt.date(2024,8,2))
    assert vec is not None
    assert audit["positiveLotCount"]==5
    assert audit["rejected"][day.isoformat()]["nonpositive_amount"]==1
    assert rows[-1]["amountRaw"]=="-50.00" and rows[-1]["weightRaw"]=="-0.5"


def test_origin_is_strict_and_stale_and_missing_are_unavailable():
    origin=dt.date(2024,8,2)
    rows=[{"product":"onion","tradeDate":origin.isoformat(),"amountRaw":"100","weightRaw":"1"} for _ in range(5)]
    vec,audit=pilot.dispersion_features(rows,{("onion",origin.isoformat()):100.0},"onion",origin)
    assert vec is None and audit["reason"]=="no_prior_trading_day"
    rows=[dict(r,tradeDate="2024-07-01") for r in rows]
    vec,audit=pilot.dispersion_features(rows,{("onion","2024-07-01"):100.},"onion",origin)
    assert vec is None and audit["reason"]=="source_older_than_14_days"
    vec,audit=pilot.dispersion_features([],{},"onion",origin)
    assert vec is None and audit["status"]=="unavailable"


def test_latest_trading_day_with_four_lots_does_not_fall_back():
    rows=[{"product":"onion","tradeDate":"2024-08-01","amountRaw":"100","weightRaw":"1"} for _ in range(5)]
    rows += [{"product":"onion","tradeDate":"2024-08-02","amountRaw":"100","weightRaw":"1"} for _ in range(4)]
    vec,audit=pilot.dispersion_features(rows,{("onion","2024-08-01"):100.,("onion","2024-08-02"):100.},"onion",dt.date(2024,8,3))
    assert vec is None and audit["sourceDate"]=="2024-08-02" and audit["reason"]=="fewer_than_5_positive_lots"


def test_reconcile_signed_proxy_fails_closed():
    assert pilot._reconcile_signed_prices({("onion","2024-08-01"):100.},{("onion","2024-08-01"):100.})
    with pytest.raises(ValueError,match="signed_price_mismatch"):
        pilot._reconcile_signed_prices({("onion","2024-08-01"):99.},{("onion","2024-08-01"):100.})
    with pytest.raises(ValueError,match="signed_numerator_denominator_mismatch"):
        pilot._reconcile_signed_prices({("onion","2024-08-01"):100.},{("onion","2024-08-01"): {"actual":100.,"volumeKg":5.}}, {("onion","2024-08-01"):(500.,4.)})


def test_source_date_without_frozen_price_never_becomes_feature_even_at_zero_overlap():
    day="2024-08-01"; rows=[{"product":"onion","tradeDate":day,"amountRaw":"100","weightRaw":"1"} for _ in range(5)]
    signed={("onion",day):100.}
    reconciled=pilot._reconcile_signed_prices(signed,{})
    assert reconciled==[]
    vec,audit=pilot.dispersion_features(rows,{},"onion",dt.date(2024,8,2),{("onion",day)})
    assert vec is None and audit["reason"]=="not_in_frozen_prices_snapshot"


def test_payload_product_and_verified_complete_garlic_query_preference():
    def rec(code,state):
        q={"cond[gds_lclsf_cd::EQ]":"12","cond[gds_mclsf_cd::EQ]":"09","cond[gds_sclsf_cd::EQ]":code,"cond[trd_clcln_ymd::EQ]":"2024-08-01","cond[whsl_mrkt_cd::EQ]":"110001"}
        return {"query_json":json.dumps(q),"selected_state":state}
    scopes=pilot._preferred_garlic_scopes([rec("06","complete"),rec("43","failed"),rec("999","complete")])
    assert scopes=={("2024-08-01","110001","06")}
    q={"cond[gds_lclsf_cd::EQ]":"12","cond[gds_mclsf_cd::EQ]":"09","cond[trd_clcln_ymd::EQ]":"2024-08-01","cond[whsl_mrkt_cd::EQ]":"110001"}
    payload={"gds_lclsf_cd":"12","gds_mclsf_cd":"09","gds_sclsf_cd":"06","gds_sclsf_nm":None}
    assert pilot._payload_category_exclusion("garlic",q,payload,scopes)=="overlap_preferred_specific_query"
    assert pilot._payload_category_exclusion("garlic",q,payload,set()) is None
    assert pilot._payload_category_exclusion("onion",q,payload,set())=="payload_category_mismatch"


def test_dm_denominator_and_holm_family_conclusions():
    assert pilot._dm({},9)["conclusion"]=="inference_unavailable"
    rows=[]
    for i in range(35):
        d=dt.date(2024,1,1)+dt.timedelta(days=i)
        rows.append({"product":"onion","horizon":1,"target":d.isoformat(),"actual":100.,"candidate":90.+i%3,"ablation":110.,"direct":110.,"persistence":110.})
    got=pilot._summarize(rows)
    item=next(x for x in got["dmTestsByHacCalendarDays"]["9"] if x["product"]=="onion" and x["horizon"]==1 and x["comparator"]=="direct" and x["loss"]=="AE")
    assert item["n"]==35 and item["conclusion"]=="improvement"
    assert item["status"]=="tested" and item["holmAdjustedPValue"]>=item["pValue"]
    assert got["coverage"]["planned"]==1275


def test_holm_known_adjusted_p_values_and_decisions():
    family=[{"status":"tested","pValue":p,"meanLossDifference":-1.} for p in (.001,.01,.02,.04)]
    pilot._holm(family)
    assert [r["holmAdjustedPValue"] for r in family]==pytest.approx([.004,.03,.04,.04])
    assert [r["conclusion"] for r in family]==["improvement"]*4


def test_immutable_writer_never_replaces_existing_output(tmp_path):
    path=tmp_path/"manifest.json"; pilot._write_immutable(path,{"x":1})
    before=path.read_bytes()
    with pytest.raises(FileExistsError): pilot._write_immutable(path,{"x":2})
    assert path.read_bytes()==before and json.loads(before)=={"x":1}


def test_collector_schema_extract_preserves_provenance_and_manifest_is_validated(tmp_path,monkeypatch):
    dbpath=tmp_path/"synthetic.sqlite"; db=collector.open_db(str(dbpath))
    url="https://unit.test/prices"; epoch="e1"
    db.execute("INSERT INTO jobs(job_id,command,config_json,started_at,finished_at,state) VALUES(1,'fixture','{}','2024-08-08','2024-08-08','complete')")
    for ix,market in enumerate(("110001","110008"),1):
        query={"cond[trd_clcln_ymd::EQ]":"2024-08-07","cond[whsl_mrkt_cd::EQ]":market,"cond[gds_lclsf_cd::EQ]":"12","cond[gds_mclsf_cd::EQ]":"01","pageNo":"1","numOfRows":"100"}
        qtext=collector.json_text(query); key=coverage._request_task_key("prices_origin",url,qtext)
        db.execute("INSERT INTO planned_queries(job_id,task_key,dataset,source_url,query_json,state,row_count,epoch_id) VALUES(1,?,'prices_origin',?,?,'complete',5,?)",(key,url,qtext,epoch))
        db.execute("INSERT INTO source_requests(request_id,job_id,request_key,epoch_id,dataset,source_url,query_json,retrieved_at,response_sha256,http_status,api_status,expected_count,returned_count,page_no,state,response_text) VALUES(?,1,?,?,'prices_origin',?,?,?, ?,200,'OK',5,5,1,'complete','fixture')",(ix,str(ix),epoch,url,qtext,"2024-08-08T01:00:00Z",f"response-hash-{ix}"))
        for j in range(5):
            payload={"gds_lclsf_cd":"12","gds_mclsf_cd":"01","gds_sclsf_cd":"","gds_sclsf_nm":None,"trd_se":"경매","unit_nm":"kg","whsl_mrkt_cd":market,"trd_clcln_ymd":"2024-08-07","totprc":"100.00","unit_tot_qty":"1.000"}
            if j==4: payload.update(totprc="-20.00",unit_tot_qty="-0.200")
            text=json.dumps(payload,ensure_ascii=False)
            db.execute("INSERT INTO raw_records(request_id,row_no,payload_json) VALUES(?,?,?)",(ix,j,text))
            db.execute("INSERT INTO normalized_records(request_id,row_no,dataset,reference_date,payload_json,quality_flags) VALUES(?,?,'prices_origin','2024-08-07',?,'[]')",(ix,j,text))
    empty_query={"cond[trd_clcln_ymd::EQ]":"2024-08-07","cond[whsl_mrkt_cd::EQ]":"110001","cond[gds_lclsf_cd::EQ]":"12","cond[gds_mclsf_cd::EQ]":"09","cond[gds_sclsf_cd::EQ]":"06","pageNo":"1","numOfRows":"100"}
    eq=collector.json_text(empty_query); ek=coverage._request_task_key("prices_origin",url,eq)
    db.execute("INSERT INTO planned_queries(job_id,task_key,dataset,source_url,query_json,state,row_count,epoch_id) VALUES(1,?,'prices_origin',?,?,'empty',0,?)",(ek,url,eq,epoch))
    db.execute("INSERT INTO source_requests(request_id,job_id,request_key,epoch_id,dataset,source_url,query_json,retrieved_at,response_sha256,http_status,api_status,expected_count,returned_count,page_no,state,response_text) VALUES(3,1,'3',?,'prices_origin',?,?,?,'empty-response-sha',200,'OK',0,0,1,'complete','[]')",(epoch,url,eq,"2024-08-08T01:00:00Z"))
    broad_query={"cond[trd_clcln_ymd::EQ]":"2024-08-07","cond[whsl_mrkt_cd::EQ]":"110001","cond[gds_lclsf_cd::EQ]":"12","cond[gds_mclsf_cd::EQ]":"09","pageNo":"1","numOfRows":"100"}
    bq=collector.json_text(broad_query); bk=coverage._request_task_key("prices_origin",url,bq)
    db.execute("INSERT INTO planned_queries(job_id,task_key,dataset,source_url,query_json,state,row_count,epoch_id) VALUES(1,?,'prices_origin',?,?,'complete',1,?)",(bk,url,bq,epoch))
    db.execute("INSERT INTO source_requests(request_id,job_id,request_key,epoch_id,dataset,source_url,query_json,retrieved_at,response_sha256,http_status,api_status,expected_count,returned_count,page_no,state,response_text) VALUES(5,1,'5',?,'prices_origin',?,?,?,'broad-response-sha',200,'OK',1,1,1,'complete','fixture')",(epoch,url,bq,"2024-08-08T01:00:00Z"))
    garlic={"gds_lclsf_cd":"12","gds_mclsf_cd":"09","gds_sclsf_cd":"41","gds_sclsf_nm":"깐마늘 한지","trd_se":"경매","unit_nm":"kg","whsl_mrkt_cd":"110001","trd_clcln_ymd":"2024-08-07","totprc":"100.00","unit_tot_qty":"1.000"}
    garlic_text=json.dumps(garlic,ensure_ascii=False)
    db.execute("INSERT INTO raw_records(request_id,row_no,payload_json) VALUES(5,0,?)",(garlic_text,))
    db.execute("INSERT INTO normalized_records(request_id,row_no,dataset,reference_date,payload_json,quality_flags) VALUES(5,0,'prices_origin','2024-08-07',?,'[]')",(garlic_text,))
    failed_query={"cond[trd_clcln_ymd::EQ]":"2024-08-06","cond[whsl_mrkt_cd::EQ]":"110001","cond[gds_lclsf_cd::EQ]":"12","cond[gds_mclsf_cd::EQ]":"01","pageNo":"1","numOfRows":"100"}
    fq=collector.json_text(failed_query); fk=coverage._request_task_key("prices_origin",url,fq)
    db.execute("INSERT INTO planned_queries(job_id,task_key,dataset,source_url,query_json,state,row_count,epoch_id) VALUES(1,?,'prices_origin',?,?,'failed',NULL,?)",(fk,url,fq,epoch))
    db.execute("INSERT INTO source_requests(request_id,job_id,request_key,epoch_id,dataset,source_url,query_json,retrieved_at,response_sha256,http_status,api_status,state,response_text) VALUES(4,1,'4',?,'prices_origin',?,?,?,'failed-response-sha',500,'ERROR','failed','failure')",(epoch,url,fq,"2024-08-08T01:01:00Z"))
    db.commit();db.close()
    price_obj={"schemaVersion":1,"products":[{"id":p,"observations":[{"date":"2024-08-07","actual":100.0,"volumeKg":7.6}] if p=="onion" else []} for p in pilot.PRODUCTS]}
    price_bytes=(json.dumps(price_obj,separators=(",",":"))+"\n").encode(); (tmp_path/"prices.json").write_bytes(price_bytes)
    monkeypatch.setattr(pilot,"PRICE_SHA256",hashlib.sha256(price_bytes).hexdigest())
    output=tmp_path/"manifest.json"; manifest=pilot.extract(dbpath,output)
    src_rows=json.loads((tmp_path/manifest["sourceRowsArtifact"]).read_text())
    assert len(src_rows)==11 and any(r["amountRaw"]=="-20.00" for r in src_rows)
    summary=next(x for x in manifest["sourceQuerySummary"] if x["product"]=="onion" and x["tradeDate"]=="2024-08-07" and x["market"]=="110001")
    assert summary["requests"]["1"]["responseSha256"]=="response-hash-1"
    garlic_queries=[x for x in manifest["sourceQuerySummary"] if x["product"]=="garlic"]
    assert len(garlic_queries)==2 and len({x["taskKey"] for x in garlic_queries})==2
    empty=next(x for x in garlic_queries if x["conditions"].get("cond[gds_sclsf_cd::EQ]")=="06")
    assert empty["selectedState"]=="empty" and empty["requests"]["3"]["responseSha256"]=="empty-response-sha"
    broad=next(x for x in garlic_queries if "cond[gds_sclsf_cd::EQ]" not in x["conditions"])
    assert broad["selectedState"]=="complete" and broad["requests"]["5"]["responseSha256"]=="broad-response-sha"
    gf=next(x for x in manifest["features"] if x["product"]=="garlic" and x["origin"]=="2024-08-08")
    assert gf["features"] is None and gf["audit"]["reason"]=="not_in_frozen_prices_snapshot"
    assert manifest["signedPriceReconciliation"]["status"]=="partial_unreconciled_source_dates"
    unavailable=next(x for x in manifest["sourceQuerySummary"] if x["tradeDate"]=="2024-08-06")
    assert unavailable["selectedState"] is None and unavailable["requests"]["4"]["state"]=="failed"
    f=next(x for x in manifest["features"] if x["product"]=="onion" and x["origin"]=="2024-08-08")
    assert f["audit"]["sourceDate"]=="2024-08-07" and f["audit"]["positiveLotCount"]==8
    assert pilot._validate_frozen_manifest(manifest,output)
    bad=dict(manifest); bad["features"]=manifest["features"][:-1]
    with pytest.raises(ValueError,match="feature_keys_mismatch"):pilot._validate_frozen_manifest(bad,output)
    bad=json.loads(json.dumps(manifest)); f=next(x for x in bad["features"] if x["product"]=="onion" and x["origin"]=="2024-08-08"); f["audit"]["sourceDate"]="2024-08-08"
    with pytest.raises(ValueError,match="invalid_feature_source_date"):pilot._validate_frozen_manifest(bad,output)
    no_overlap_obj={"schemaVersion":1,"products":[{"id":p,"observations":[]} for p in pilot.PRODUCTS]}
    no_overlap_bytes=(json.dumps(no_overlap_obj,separators=(",",":"))+"\n").encode();(tmp_path/"prices.json").write_bytes(no_overlap_bytes)
    monkeypatch.setattr(pilot,"PRICE_SHA256",hashlib.sha256(no_overlap_bytes).hexdigest())
    no_overlap=pilot.extract(dbpath,tmp_path/"no-overlap.json")
    assert no_overlap["signedPriceReconciliation"]["overlappingDatesChecked"]==0
    assert no_overlap["signedPriceReconciliation"]["status"]=="no_overlapping_dates"
    nf=next(x for x in no_overlap["features"] if x["product"]=="onion" and x["origin"]=="2024-08-08")
    assert nf["features"] is None and nf["audit"]["reason"]=="not_in_frozen_prices_snapshot"
