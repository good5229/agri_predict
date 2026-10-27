import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from agri_data import supply_document_prospective as p


def test_frozen_config_and_date_window():
    cfg = p._load_cfg(p.DEFAULT_CONFIG)
    assert cfg["limits"] == {"max_requests": 300, "budget_seconds": 600, "min_interval_seconds": 1.0,
                             "retries": 1, "timeout_seconds": 25, "max_bytes": 41943040}
    first, last = p._date_window(p.datetime(2026, 10, 16, 1, tzinfo=p.timezone.utc), 90)
    assert (first, last) == (date(2026, 7, 19), date(2026, 10, 16))
    fixed=json.loads((p.ROOT/"configs/collection/supply_documents_evaluation_20261016.json").read_text())
    assert fixed["scope"]["fixed_window"]=={"start":"2024-01-01","end":"2026-09-30"}
    assert fixed["scope"]["jeju_years"]==[2024,2025,2026]
    assert fixed["reuse_roots"]==["data/raw/supply_documents_prospective"]


def test_fixed_window_validation_rejects_mixed_or_incomplete_scope(tmp_path):
    base=json.loads((p.ROOT/"configs/collection/supply_documents_evaluation_20261016.json").read_text())
    cfg=tmp_path/"bad.json"
    for update,match in [({"window_days":90},"mutually_exclusive"),({"jeju_years":[2024,2025]},"jeju_years_must_match"),({"fixed_window":{"start":"2026-09-30","end":"2024-01-01"}},"fixed_window_order")]:
        bad=json.loads(json.dumps(base)); bad["scope"].update(update); cfg.write_text(json.dumps(bad))
        with pytest.raises(ValueError,match=match): p._load_cfg(cfg)


def test_mafra_and_jeju_pagination_parse():
    mafra = b"<div>21 posts</div><a href=\"#\" onclick=\"page_link('3')\">3</a><div class=\"_paging\"><a class=\"_last\" href=\"javascript:page_link('18')\">last</a></div><table><tr><td>x</td></tr></table>"
    rows, total, pages = p._pagination(mafra, {"id": "mafra-title"})
    assert total is None and pages == 18 and rows[0]["text"] == "x"
    assert p._mafra_last_page(mafra)==18
    jeju = "<div class=\"result-count\">총 게시글 : <strong>36</strong>건</div><a onclick=\"fn_dataList(3); return false;\">3</a><tr><td>제주 주간 2026.10.01</td><a onclick=\"fn_callSynapViewer('/files/board/','abc.pdf','x.pdf')\">PDF</a></tr>".encode()
    rows, total, pages = p._pagination(jeju, {"id": "jeju-weekly"})
    assert total == 36 and pages == 3
    items = p._row_items(rows, "https://agri.jeju.go.kr/list", "jeju-weekly", None, 2026, 1)
    assert items[0]["url"] == "https://agri.jeju.go.kr/files/board/abc.pdf"
    assert items[0]["date_candidate"] == "2026.10.01"


def test_jeju_page_filters_global_nav_and_window_rows():
    html = """<ul><li><a href='/jadx/home'>농업기술원</a></li>
      <li><a href='#'>50 2026년 영농동향 2026.10.01 PDF보기</a>
       <a href='#' onclick=\"fn_callSynapViewer('/files/board/','new.pdf','new.pdf')\">PDF</a></li>
      <li><a href='#'>29 2026년 영농동향 2026.07.16 PDF보기</a>
       <a href='#' onclick=\"fn_callSynapViewer('/files/board/','old.pdf','old.pdf')\">PDF</a></li></ul>""".encode()
    rows,total,pages=p._pagination(html,{"id":"jeju-weekly"})
    items=p._row_items(rows,"https://agri.jeju.go.kr/list","jeju-weekly",None,2026,1)
    assert len(items)==2
    assert [p._within_window(x,date(2026,7,19),date(2026,10,16)) for x in items] == ["in_window","outside_rolling_window_not_fetched"]


def test_repeated_listing_page_is_rejected():
    seen=[]
    p._append_page_hash(seen,"abc")
    with pytest.raises(RuntimeError,match="repeated_listing_page_hash"):
        p._append_page_hash(seen,"abc")


def test_detail_parser_keeps_mixed_quote_onclick_attachment():
    body=b'''<html><ul><li><a href="/bbs/home/792/600799/download.do">bulletin.pdf</a><a target="_blank" href="/sn3hcv_v2023/view.jsp?filepath=/home/atchmnfl/bbs/792/600799.pdf">view</a></li><li><a href="/bbs/home/792/600800/download.do">bulletin.hwp</a><a href="/sn3hcv_v2023/view.jsp?filepath=/home/atchmnfl/bbs/792/600800.hwp">view</a></li></ul></html>'''
    docs=p._details(body,"https://www.mafra.go.kr/bbs/home/792/1/artclView.do",{"source":"mafra-title"})
    assert docs==[{"source":"mafra-title","url":"https://www.mafra.go.kr/bbs/home/792/600799/download.do","kind":"pdf"}]


def test_resume_reuses_immutable_response_without_spending_new_request(tmp_path):
    body=b"<html>saved</html>"; sha=hashlib.sha256(body).hexdigest()
    root=tmp_path/"data"; run_dir=root/"runs"/"run1"; run_dir.mkdir(parents=True)
    obj=root/"objects"/sha[:2]/sha; obj.parent.mkdir(parents=True); obj.write_bytes(body)
    cp={"request_count":300,"observations":[{"url":"https://www.mafra.go.kr/a","kind":"listing","ok":True,"sha256":sha,"object_path":str(obj.relative_to(root)),"retrieved_at":"2026-10-16T00:00:00Z"}]}
    cfg={"limits":{"max_requests":300,"budget_seconds":600,"min_interval_seconds":1,"retries":1,"timeout_seconds":25,"max_bytes":1000}}
    f=p.Fetcher(cfg,{"www.mafra.go.kr"},run_dir,cp)
    got,obs=f.get("https://www.mafra.go.kr/a","listing",{})
    assert got==body and obs["retrieved_at"]=="2026-10-16T00:00:00Z" and f.count==0


def test_fixed_window_can_reuse_exact_url_from_prior_run(tmp_path,monkeypatch):
    monkeypatch.setattr(p,"ROOT",tmp_path)
    source=tmp_path/"rolling"; body=b"<html>previously captured</html>"; sha=hashlib.sha256(body).hexdigest()
    obj=source/"objects"/sha[:2]/sha; obj.parent.mkdir(parents=True); obj.write_bytes(body)
    run_id="rolling-1"; cp_dir=source/"runs"/run_id; cp_dir.mkdir(parents=True)
    original={"url":"https://www.mafra.go.kr/bbs/home/792/1/artclView.do","kind":"detail","sha256":sha,"ok":True,"status":206,"headers":{"content-type":"text/html","x-source":"verified"},"started_at":"2026-10-16T01:59:59Z","retrieved_at":"2026-10-16T02:00:00Z","byte_length":len(body),"object_path":str(obj.relative_to(source))}
    (cp_dir/"checkpoint.json").write_text(json.dumps({"status":"complete","observations":[original]}))
    (source/"latest.json").write_text(json.dumps({"run_id":run_id,"status":"complete"}))
    (source/"url_hash_index.json").write_text(json.dumps({"urls":{original["url"]:{"latest_sha256":sha,"latest_seen_at":original["retrieved_at"],"versions":[{"sha256":sha,"first_seen_at":original["retrieved_at"],"last_seen_at":original["retrieved_at"],"run_id":run_id}]}}}))
    target=tmp_path/"evaluation"/"runs"/"eval-1"; target.mkdir(parents=True)
    cp={"request_count":0,"observations":[]}
    cfg={"limits":{"max_requests":300,"budget_seconds":600,"min_interval_seconds":1,"retries":1,"timeout_seconds":25,"max_bytes":1000},"reuse_roots":["rolling"]}
    fetch=p.Fetcher(cfg,{"www.mafra.go.kr","agri.jeju.go.kr"},target,cp)
    got,obs=fetch.get(original["url"],"detail",{"source":"mafra-title"})
    assert got==body and obs["retrieved_at"]==original["retrieved_at"]
    assert obs["started_at"]==original["started_at"] and obs["status"]==206 and obs["headers"]==original["headers"]
    assert obs["capture_origin"]=="reused_prior_capture" and fetch.count==0


def test_fixed_window_refuses_to_synthesize_prior_capture_metadata(tmp_path,monkeypatch):
    from urllib.error import URLError
    monkeypatch.setattr(p,"ROOT",tmp_path)
    source=tmp_path/"rolling"; body=b"old"; sha=hashlib.sha256(body).hexdigest()
    obj=source/"objects"/sha[:2]/sha; obj.parent.mkdir(parents=True); obj.write_bytes(body)
    (source/"url_hash_index.json").write_text(json.dumps({"urls":{"https://www.mafra.go.kr/a":{"latest_sha256":sha,"versions":[{"sha256":sha,"run_id":"old"}]}}}))
    (source/"latest.json").write_text(json.dumps({"run_id":"old","status":"complete"}))
    old=source/"runs"/"old"; old.mkdir(parents=True); (old/"checkpoint.json").write_text(json.dumps({"status":"complete","observations":[]}))
    target=tmp_path/"eval"/"runs"/"1"; target.mkdir(parents=True); cp={"request_count":0,"observations":[]}
    cfg={"limits":{"max_requests":300,"budget_seconds":600,"min_interval_seconds":0,"retries":0,"timeout_seconds":1,"max_bytes":100},"reuse_roots":["rolling"]}
    fetch=p.Fetcher(cfg,{"www.mafra.go.kr"},target,cp)
    class FailingOpener:
        calls=0
        def open(self,*a,**k): self.calls+=1; raise URLError("offline fixture")
    opener=FailingOpener(); fetch.opener=opener
    with pytest.raises(RuntimeError,match="request_failed"): fetch.get("https://www.mafra.go.kr/a","document",{})
    assert opener.calls==1 and fetch.count==1


def test_run_completion_and_partial_checkpoint_preserve_cumulative_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(p,"ROOT",tmp_path)
    cfg={"protocol":p.PROTOCOL,"scope":{"window_days":90,"timezone":"Asia/Seoul","sources":[
        {"id":"mafra-title","base_url":"https://www.mafra.go.kr/bbs/home/792/artclList.do","queries":["배추","무","수급"],"page_size":10},
        {"id":"jeju-weekly","base_url":"https://agri.jeju.go.kr/jadx/web/agriBbs/agriInfo/trendList.do","search_type":"baTitle","page_size":12}]},
        "limits":{"max_requests":300,"budget_seconds":600,"min_interval_seconds":0,"retries":1,"timeout_seconds":2,"max_bytes":100000},
        "hosts":["www.mafra.go.kr","agri.jeju.go.kr"],"data_root":"prospective"}
    config=tmp_path/"config.json"; config.write_text(json.dumps(cfg))
    jeju_list='''<div class="result-count">총 게시글 : <strong>1</strong>건</div><ul><li><a href="#">1 report 2026.10.01</a><a href="#" onclick="fn_callSynapViewer('/files/board/','j.pdf','j.pdf')">PDF</a></li></ul>'''.encode()
    detail=b'''<ul><li><a href="/bbs/home/792/9/download.do">x.pdf</a><a href="/sn3hcv_v2023/view.jsp?filepath=/home/x.pdf">view</a></li></ul>'''
    pdf=b"%PDF-1.4\nfixture"

    class FakeFetcher:
        fail_after=None
        def __init__(self,config,hosts,run_dir,checkpoint):
            self.cfg=config["limits"]; self.hosts=hosts; self.run_dir=run_dir; self.cp=checkpoint; self.count=0; self.total=checkpoint.get("request_count",0)
        def get(self,url,kind,meta):
            for old in reversed(self.cp.get("observations",[])):
                if old.get("url")==url and old.get("ok"):
                    path=self.run_dir.parent.parent/old["object_path"]
                    return path.read_bytes(),old
            self.count+=1; self.total+=1; self.cp["request_count"]=self.total
            if self.fail_after and self.count>self.fail_after: raise RuntimeError("fake_failure")
            if kind=="listing":
                if "agri.jeju" in url: body=jeju_list
                else:
                    from urllib.parse import parse_qs,urlparse
                    term=parse_qs(urlparse(url).query).get("srchWrd",["배추"])[0]
                    rid={"배추":1,"무":2,"수급":3}[term]
                    body=f'<table><tr><td><a href="/bbs/home/792/{rid}/artclView.do">{rid} title 2026.10.01</a></td></tr></table><div class="_paging"><a class="_last" href="javascript:page_link(\'1\')">last</a></div>'.encode()
            elif kind=="detail":
                from urllib.parse import urlparse
                rid=urlparse(url).path.split("/")[-2]
                body=detail.replace(b"/9/",f"/{rid}/".encode()).replace(b"/600799/",f"/6007{rid}/".encode()).replace(b"600799.pdf",f"6007{rid}.pdf".encode())
            else: body=pdf
            sha=hashlib.sha256(body).hexdigest(); obj=self.run_dir.parent.parent/"objects"/sha[:2]/sha; obj.parent.mkdir(parents=True,exist_ok=True); obj.write_bytes(body)
            obs={"url":url,"kind":kind,"metadata":meta,"attempt":1,"started_at":"t","retrieved_at":"2026-10-16T02:00:00Z","status":200,"headers":{"content-type":"application/pdf" if body.startswith(b"%PDF-") else "text/html"},"ok":True,"error":None,"sha256":sha,"byte_length":len(body),"object_path":str(obj.relative_to(self.run_dir.parent.parent))}
            self.cp.setdefault("observations",[]).append(obs)
            return body,obs

    monkeypatch.setattr(p,"Fetcher",FakeFetcher)
    complete=p.run(config)
    assert complete["status"]=="complete" and complete["listing_item_count"]==4 and complete["document_count"]==4
    FakeFetcher.fail_after=2
    with pytest.raises(RuntimeError,match="fake_failure"):
        p.run(config)
    latest=json.loads((tmp_path/"prospective/latest.json").read_text())
    checkpoint=json.loads((tmp_path/"prospective"/latest["checkpoint"]).read_text())
    assert latest["status"]=="incomplete"
    assert checkpoint["request_count"]==3 and checkpoint["request_count_this_invocation"]==3


def test_urls_are_confined_to_exact_hosts():
    assert p._safe_url("https://www.mafra.go.kr/bbs/home/792/artclList.do", {"www.mafra.go.kr"})
    with pytest.raises(ValueError):
        p._safe_url("https://evil.example/a", {"www.mafra.go.kr"})
    with pytest.raises(ValueError):
        p._safe_url("http://www.mafra.go.kr/a", {"www.mafra.go.kr"})


def test_config_scope_has_only_approved_public_queries():
    cfg=json.loads(p.DEFAULT_CONFIG.read_text())
    assert [x for x in cfg["scope"]["sources"][0]["queries"]] == ["배추", "무", "수급"]
    assert all("krei" not in x["base_url"].lower() for x in cfg["scope"]["sources"])
