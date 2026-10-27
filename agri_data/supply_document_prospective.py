"""Prospective, versioned capture of scoped MAFRA and Jeju public documents.

Only the explicitly configured queries and their enumerated pagination are
visited. Each invocation is a fresh observation; an incomplete run resumes its
frozen plan. Raw response bodies are content-addressed and never overwritten.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
import time
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin, urlparse, parse_qs
from urllib.request import Request, HTTPRedirectHandler, build_opener
from zoneinfo import ZoneInfo

from agri_data.supply_document_collection import write_immutable_idempotent

PROTOCOL = "supply-documents-prospective-v1"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/collection/supply_documents_prospective.json"
KST = ZoneInfo("Asia/Seoul")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_write(path: Path, obj: object) -> None:
    data = (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        try: os.unlink(temp)
        except FileNotFoundError: pass


class _Lock:
    def __init__(self, path: Path): self.path, self.fd = path, None
    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try: fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            os.close(self.fd); self.fd=None
            raise RuntimeError("collector_locked") from e
        os.ftruncate(self.fd, 0); os.write(self.fd, f"pid={os.getpid()} started={_now()}\n".encode()); os.fsync(self.fd)
        return self
    def __exit__(self, *_):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN); os.close(self.fd)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HTTPError(req.full_url, code, "redirect_rejected", headers, fp)


class _HTML(HTMLParser):
    """Keep row text, anchors, onclicks and visible pagination evidence."""
    def __init__(self):
        super().__init__(); self.rows=[]; self.row=None; self.anchor_stack=[]; self.links=[]; self.text=[]
    def handle_starttag(self, tag, attrs):
        a=dict(attrs); tag=tag.lower()
        if tag in ("tr", "li"):
            self.row={"text":[],"links":[]}; self.rows.append(self.row)
        if tag == "a":
            anchor={"attrs":a,"text":[]}
            self.anchor_stack.append(anchor); self.links.append(anchor)
            if self.row is not None: self.row["links"].append(anchor)
        if tag in ("strong", "b", "span", "em"):
            cls=a.get("class","")
            if any(x in cls.lower() for x in ("total", "count", "paging")): self.text.append(" ")
    def handle_data(self, d):
        self.text.append(d)
        if self.row is not None: self.row["text"].append(d)
        for anchor in self.anchor_stack: anchor["text"].append(d)
    def handle_endtag(self, tag):
        if tag.lower()=="a" and self.anchor_stack:
            anchor=self.anchor_stack.pop(); anchor["text"]=" ".join("".join(anchor["text"]).split())
        if tag.lower() in ("tr", "li"):
            self.row=None


def _parse_html(body: bytes):
    p=_HTML(); p.feed(body.decode("utf-8",errors="replace")); p.close()
    for row in p.rows: row["text"]=" ".join("".join(row["text"]).split())
    return p.rows, " ".join("".join(p.text).split())


def _safe_url(url: str, hosts: set[str]) -> str:
    from agri_data.supply_document_collection import validate_url
    validate_url(url)
    u=urlparse(url)
    if u.hostname not in hosts: raise ValueError("host_outside_task_allowlist")
    return url


class Fetcher:
    def __init__(self, config, hosts, run_dir, checkpoint):
        self.cfg=config["limits"]; self.hosts=hosts; self.run_dir=run_dir; self.cp=checkpoint
        self.start=time.monotonic(); self.last=None; self.count=0; self.total=checkpoint.get("request_count",0)
        self.reuse_roots=[ROOT/x for x in config.get("reuse_roots",[])]
        self.opener=build_opener(_NoRedirect())
    def get(self,url,kind,meta):
        _safe_url(url,self.hosts)
        # A resumed incomplete run reuses its exact prior successful response;
        # a later invocation creates a new run and always fetches afresh.
        for prior in reversed(self.cp.get("observations", [])):
            if prior.get("url") == url and prior.get("ok") and prior.get("object_path"):
                path=self.run_dir.parent.parent/prior["object_path"]
                body=path.read_bytes()
                if _digest(body)!=prior.get("sha256"): raise RuntimeError("checkpoint_object_hash_mismatch")
                return body,prior
        # Exact-URL captures from the completed rolling run are reused without
        # another HTTP request; their original capture time remains the
        # availability boundary. Each source index is read-only.
        for source_root in self.reuse_roots if kind in {"detail", "document"} else []:
            index_path=source_root/"url_hash_index.json"
            if not index_path.exists(): continue
            index=json.loads(index_path.read_text(encoding="utf-8"))
            rec=index.get("urls",{}).get(url)
            if not rec: continue
            sha=rec.get("latest_sha256")
            if not sha: continue
            latest_path=source_root/"latest.json"
            if not latest_path.exists(): continue
            latest=json.loads(latest_path.read_text(encoding="utf-8"))
            if latest.get("status")!="complete": continue
            source_run=latest.get("run_id")
            if not source_run: continue
            source_cp=source_root/"runs"/source_run/"checkpoint.json"
            if not source_cp.exists(): continue
            source_doc=json.loads(source_cp.read_text(encoding="utf-8"))
            if source_doc.get("status")!="complete": continue
            original_obs=next((o for o in reversed(source_doc.get("observations",[])) if o.get("url")==url and o.get("sha256")==sha and o.get("ok") and o.get("kind")==kind),None)
            if not original_obs: continue
            object_path=source_root/original_obs.get("object_path","")
            if not object_path.is_file(): continue
            body=object_path.read_bytes()
            if _digest(body)!=sha: continue
            reused_path=self.run_dir.parent.parent/"objects"/sha[:2]/sha
            write_immutable_idempotent(reused_path,body)
            obs={"url":url,"kind":kind,"metadata":meta,"attempt":0,"started_at":original_obs["started_at"],"retrieved_at":original_obs["retrieved_at"],"status":original_obs["status"],"headers":original_obs["headers"],"ok":True,"error":None,"sha256":sha,"byte_length":original_obs["byte_length"],"object_path":str(reused_path.relative_to(self.run_dir.parent.parent)),"capture_origin":"reused_prior_capture","reused_from":str(source_root.relative_to(ROOT)),"original_run_id":source_run}
            self.cp.setdefault("observations",[]).append(obs); _json_write(self.run_dir/"checkpoint.json",self.cp)
            return body,obs
        for attempt in range(self.cfg["retries"]+1):
            remaining=self.cfg["budget_seconds"]-(time.monotonic()-self.start)
            if self.count >= self.cfg["max_requests"] or remaining<=0:
                raise RuntimeError("request_or_time_budget_exhausted")
            if self.last is not None:
                pause=max(self.cfg["min_interval_seconds"]-(time.monotonic()-self.last),0)
                if pause>remaining: raise RuntimeError("time_budget_exhausted_before_next_request")
                time.sleep(pause)
            remaining=self.cfg["budget_seconds"]-(time.monotonic()-self.start)
            if remaining<=0: raise RuntimeError("time_budget_exhausted_before_next_request")
            self.last=time.monotonic(); self.count+=1; self.total+=1; self.cp["request_count"]=self.total
            started=_now(); status=None; headers={}; body=b""; error=None
            attempt_row={"url":url,"kind":kind,"metadata":meta,"attempt":attempt+1,"started_at":started,"retrieved_at":None,"status":None,"headers":{},"ok":False,"error":"request_in_progress","sha256":None,"byte_length":None}
            self.cp.setdefault("observations",[]).append(attempt_row); _json_write(self.run_dir/"checkpoint.json",self.cp)
            try:
                req=Request(url,headers={"User-Agent":"agri-predict-prospective/1.0","Accept":"text/html,application/pdf,*/*;q=0.5"})
                with self.opener.open(req,timeout=min(self.cfg["timeout_seconds"],remaining)) as res:
                    status=res.status; headers=dict(res.headers.items()); final=res.geturl()
                    if final != url: raise ValueError("response_url_changed")
                    chunks=[]; size=0
                    while True:
                        remaining=self.cfg["budget_seconds"]-(time.monotonic()-self.start)
                        if remaining<=0: raise ValueError("time_budget_exhausted_during_body")
                        sock=getattr(getattr(getattr(res,"fp",None),"raw",None),"_sock",None)
                        if sock is not None: sock.settimeout(min(self.cfg["timeout_seconds"],remaining))
                        chunk_reader=getattr(res,"read1",None) or res.read
                        chunk=chunk_reader(min(65536,self.cfg["max_bytes"]+1-size))
                        if not chunk: break
                        chunks.append(chunk); size+=len(chunk)
                        if size>self.cfg["max_bytes"]: raise ValueError("response_exceeds_max_bytes")
                    body=b"".join(chunks)
            except HTTPError as e:
                status=e.code; headers=dict(e.headers.items()) if e.headers else {}; error=f"http_{e.code}"; e.close()
            except (URLError, TimeoutError, OSError, ValueError) as e: error=str(e) or type(e).__name__
            retry_after=None
            if error and status in (429,503) and attempt<self.cfg["retries"]:
                retry_after=next((v for k,v in headers.items() if k.lower()=="retry-after"),None)
            if retry_after:
                try: delay=max(0,float(retry_after))
                except ValueError:
                    try: delay=max(0,(parsedate_to_datetime(retry_after)-datetime.now(timezone.utc)).total_seconds())
                    except Exception: delay=0
                if time.monotonic()-self.start+delay+1>=self.cfg["budget_seconds"]: error="retry_after_exceeds_run_budget"
                else: time.sleep(delay)
            obs=attempt_row; obs.update({"retrieved_at":_now(),"status":status,"headers":headers,"ok":not bool(error),"error":error,"sha256":_digest(body) if not error else None,"byte_length":len(body) if not error else None})
            if not error:
                sha=_digest(body); path=self.run_dir.parent.parent/"objects"/sha[:2]/sha
                write_immutable_idempotent(path,body)
                obs["object_path"]=str(path.relative_to(self.run_dir.parent.parent))
                _json_write(self.run_dir/"checkpoint.json",self.cp)
                return body,obs
            _json_write(self.run_dir/"checkpoint.json",self.cp)
            if attempt>=self.cfg["retries"] or error=="retry_after_exceeds_run_budget":
                raise RuntimeError(f"request_failed:{url}:{error}")
        raise AssertionError


def _date_window(now: datetime, days: int):
    today=now.astimezone(KST).date(); return today-timedelta(days=days-1), today


def _pagination(html: bytes, source: dict, year=None, query=None):
    rows,visible=_parse_html(html)
    raw=html.decode("utf-8",errors="replace")
    if source["id"]=="mafra-title":
        pages=[int(x) for x in re.findall(r"page_link\s*\(\s*['\"]?(\d+)",raw)]
        total=None
        if re.search(r"검색\s*결과가\s*없습니다|검색된\s*게시물이\s*없습니다|조회된\s*게시물이\s*없습니다|등록된\s*게시물이\s*없습니다",raw): total=0
    else:
        pages=[int(x) for x in re.findall(r"fn_dataList\s*\(\s*(\d+)",raw)]
        m=re.search(r'class=["\'][^"\']*result-count[^"\']*["\'][^>]*>\s*총\s*게시글\s*:\s*<strong>\s*([\d,]+)\s*</strong>\s*건',raw,re.I)
        total=int(m.group(1).replace(",","")) if m else None
    count_pages=max(pages,default=1)
    return rows,total,count_pages


def _mafra_last_page(html: bytes) -> int | None:
    raw=html.decode("utf-8",errors="replace")
    m=re.search(r'<a\b[^>]*class=["\'][^"\']*_last[^"\']*["\'][^>]*href=["\']javascript:page_link\(["\']?(\d+)',raw,re.I)
    if not m:
        m=re.search(r'<a\b[^>]*href=["\']javascript:page_link\(["\']?(\d+)["\']?\)["\'][^>]*class=["\'][^"\']*_last',raw,re.I)
    return int(m.group(1)) if m else None


def _listing_url(source, page, query=None, start=None, end=None, year=None):
    if source["id"]=="mafra-title":
        q={"rgsBgndeStr":start.strftime("%Y.%m.%d"),"rgsEnddeStr":end.strftime("%Y.%m.%d"),"srchColumn":"sj","srchWrd":query,"page":str(page)}
    else: q={"searchType":source["search_type"],"pageIndex":str(page),"searchKeyword":str(year)}
    return source["base_url"]+"?"+urlencode(q)


def _row_items(rows,base,source_id,query,year,page):
    out=[]
    for r in rows:
        title=r["text"]
        if not title: continue
        date_match=re.search(r"20\d\d[.\-/]\s?\d{1,2}[.\-/]\s?\d{1,2}",title)
        pubdate=date_match.group(0) if date_match else None
        record_links=[]
        for a in r["links"]:
            attrs=a["attrs"]; href=attrs.get("href",""); click=attrs.get("onclick","")
            if not href and not click: continue
            target=None; kind=None
            if "fn_callSynapViewer" in click:
                args=re.findall(r"['\"]([^'\"]*)['\"]",click)
                if len(args)>=2: target=args[0].rstrip("/")+"/"+args[1]; kind="pdf"
            if not target and href and href.strip() not in ("#","javascript:;"):
                target=href; path=urlparse(urljoin(base,href)).path.lower()
                if path.endswith(".pdf") or "download" in path: kind="attachment"
                elif "artclview.do" in path: kind="detail"
            if not target: continue
            absurl=urljoin(base,target)
            if kind in ("pdf","attachment","detail"):
                record_links.append({"url":absurl,"kind":"pdf" if kind in ("pdf","attachment") else "detail"})
        # One record is counted once even where the row offers several formats.
        # Restrict Jeju to report rows with a parsed publication date and a direct PDF.
        if source_id=="jeju-weekly":
            if not pubdate: continue
            record_links=[x for x in record_links if x["kind"]=="pdf"][:1]
            kind=record_links[0]["kind"] if record_links else "listing_record"
            target=record_links[0]["url"] if record_links else None
            key=f"{source_id}:{year}:{pubdate}:{title}"
            out.append({"url":target,"kind":kind,"title":title,"date_candidate":pubdate,"source":source_id,"query":query,"year":year,"page":page,"row_context":title,"record_key":key})
        else:
            record_links=[x for x in record_links if x["kind"] in ("detail","pdf")]
            if not record_links: continue
            # Prefer the official article view as the record identity. Attachment
            # links are discovered from that page so multiple formats stay linked.
            link=next((x for x in record_links if x["kind"]=="detail"),record_links[0])
            out.append({"url":link["url"],"kind":link["kind"],"title":title,"date_candidate":pubdate,"source":source_id,"query":query,"year":year,"page":page,"row_context":title,"record_key":link["url"]})
    return out


def _details(body,base,meta):
    parser=_HTML(); parser.feed(body.decode("utf-8",errors="replace")); out=[]
    for row in parser.rows:
        download=None; pdf=False
        for link in row["links"]:
            attrs=link["attrs"]; href=attrs.get("href",""); click=attrs.get("onclick","")
            if urlparse(href).path.lower().endswith(".pdf"): download=href; pdf=True
            elif href.lower().endswith("download.do"): download=href
            if "fn_callSynapViewer" in click:
                args=re.findall(r"['\"]([^'\"]*)['\"]",click)
                if len(args)>=2 and args[1].lower().endswith(".pdf"):
                    download=args[0].rstrip("/")+"/"+args[1]; pdf=True
            if "/view.jsp" in href:
                filepath=parse_qs(urlparse(href).query).get("filepath",[""])[0]
                if filepath.lower().endswith(".pdf"): pdf=True
        # MAFRA uses paired download.do and viewer links per file row. Only
        # fetch a download when the corresponding original is explicitly PDF.
        if download and pdf: out.append({**meta,"url":urljoin(base,download),"kind":"pdf"})
    return out


def _as_date(value):
    if not value: return None
    m=re.search(r"(20\d{2})\s*[.\-/년]\s*(\d{1,2})\s*[.\-/월]\s*(\d{1,2})",value)
    if not m: return None
    try: return date(*(int(x) for x in m.groups()))
    except ValueError: return None


def _append_page_hash(seen: list[str], value: str) -> None:
    if value in seen: raise RuntimeError("repeated_listing_page_hash")
    seen.append(value)


def _within_window(item: dict, start: date, end: date) -> str:
    d=_as_date(item.get("date_candidate"))
    if d is None: return "date_unknown_not_fetched"
    if not start<=d<=end: return "outside_rolling_window_not_fetched"
    return "in_window"


def _load_cfg(path):
    c=json.loads(path.read_text(encoding="utf-8"))
    if c.get("protocol")!=PROTOCOL: raise ValueError("config_protocol_mismatch")
    scope=c.get("scope",{}); fixed=scope.get("fixed_window")
    if fixed is not None:
        if "window_days" in scope: raise ValueError("fixed_and_rolling_windows_are_mutually_exclusive")
        try: first=date.fromisoformat(fixed["start"]); last=date.fromisoformat(fixed["end"])
        except (KeyError,TypeError,ValueError) as e: raise ValueError("fixed_window_dates_invalid") from e
        if last<first: raise ValueError("fixed_window_order_invalid")
        expected=list(range(first.year,last.year+1))
        if scope.get("jeju_years")!=expected: raise ValueError("jeju_years_must_match_fixed_window")
    elif not isinstance(scope.get("window_days"),int) or scope["window_days"]<1:
        raise ValueError("rolling_window_days_invalid")
    return c


def run(config_path: Path, resume: bool=True):
    config=_load_cfg(config_path); data_root=ROOT/config["data_root"]
    data_root.mkdir(parents=True,exist_ok=True)
    with _Lock(data_root/"collector.lock"):
        latest_path=data_root/"latest.json"; latest=json.loads(latest_path.read_text()) if latest_path.exists() else None
        if resume and latest and latest.get("status")!="complete":
            run_dir=data_root/"runs"/latest["run_id"]; cp_path=run_dir/"checkpoint.json"
            cp=json.loads(cp_path.read_text())
            if cp.get("config_sha256")!=_digest(config_path.read_bytes()): raise ValueError("resume_config_changed")
            frozen=cp["scope"]
        else:
            now=datetime.now(KST)
            fixed=config["scope"].get("fixed_window")
            if fixed:
                first=date.fromisoformat(fixed["start"]); last=date.fromisoformat(fixed["end"])
                if last<first: raise ValueError("fixed_window_invalid")
            else: first,last=_date_window(now,config["scope"]["window_days"])
            run_id=now.strftime("%Y%m%dT%H%M%S%z")+"-"+os.urandom(3).hex()
            run_dir=data_root/"runs"/run_id; run_dir.mkdir(parents=True,exist_ok=False)
            years=list(config["scope"].get("jeju_years",range(first.year,last.year+1)))
            frozen={"scope_mode":"fixed_window" if fixed else "rolling_window","started_at":_now(),"window_start":first.isoformat(),"window_end":last.isoformat(),"queries":["배추","무","수급"],"jeju_years":years}
            cp={"protocol":PROTOCOL,"run_id":run_id,"status":"running","config_sha256":_digest(config_path.read_bytes()),"scope":frozen,"request_count":0,"observations":[],"items":[]}
            _json_write(run_dir/"checkpoint.json",cp)
        run_id=cp["run_id"]; frozen=cp["scope"]
        _json_write(latest_path,{"run_id":run_id,"status":"running","started_at":_now(),"checkpoint":"runs/"+run_id+"/checkpoint.json"})
        try:
            fetch=Fetcher(config,set(config["hosts"]),run_dir,cp); sources=config["scope"]["sources"]
            start=date.fromisoformat(frozen["window_start"]); end=date.fromisoformat(frozen["window_end"])
            listings=[]; page_validations=[]
            chains=[]
            for src in sources:
                searches=[("query",q) for q in frozen["queries"]] if src["id"]=="mafra-title" else [("year",y) for y in frozen["jeju_years"]]
                for typ,val in searches:
                    q=val if typ=="query" else None; y=val if typ=="year" else None
                    u=_listing_url(src,1,q,start,end,y); b,o=fetch.get(u,"listing",{"source":src["id"],"query":q,"year":y,"page":1})
                    rows,total,pages=_pagination(b,src,y,q); chain_items=[]; page_hashes=[]
                    if src["id"]=="mafra-title" and _mafra_last_page(b) is None:
                        raise RuntimeError("mafra_last_page_not_proven")
                    if src["id"]=="mafra-title": pages=_mafra_last_page(b)
                    page_size=src["page_size"]
                    expected_pages=max(1,(total+page_size-1)//page_size) if total is not None else pages
                    if total is not None and pages<expected_pages: raise RuntimeError("pagination_links_below_expected_pages")
                    if expected_pages!=pages: pages=expected_pages
                    for page in range(1,pages+1):
                        if page==1: pb=b; prow=rows; po=o
                        else:
                            pu=_listing_url(src,page,q,start,end,y); pb,po=fetch.get(pu,"listing",{"source":src["id"],"query":q,"year":y,"page":page}); prow,pt,pp=_pagination(pb,src,y,q)
                            if src["id"]=="mafra-title" and _mafra_last_page(pb)!=pages: raise RuntimeError("mafra_last_page_changed_mid_run")
                            # On the last page a source may omit the current
                            # page from clickable links; the requested page,
                            # exact reported total, and computed page count
                            # jointly prove the terminal page was reached.
                            if pp<pages and page<pages: raise RuntimeError("pagination_links_below_expected_pages")
                            if pp>pages: raise RuntimeError("pagination_grew_mid_run")
                            if pt is not None and total is not None and pt!=total: raise RuntimeError("listing_total_changed_mid_run")
                        _append_page_hash(page_hashes,po["sha256"])
                        if not prow: raise RuntimeError("listing_page_has_no_rows")
                        record_items=_row_items(prow,po["url"],src["id"],q,y,page)
                        if len(record_items)>page_size: raise RuntimeError("listing_records_exceed_page_size")
                        if src["id"]=="mafra-title" and total is None and len(record_items)==0:
                            raise RuntimeError("mafra_page_without_records")
                        if src["id"]=="mafra-title" and page<pages and len(record_items)!=page_size:
                            raise RuntimeError("mafra_nonterminal_page_count_mismatch")
                        chain_items.extend(record_items)
                        page_validations.append({"source":src["id"],"query":q,"year":y,"page":page,"record_count":len(record_items),"ui_total":total,"page_count":pages,"sha256":po["sha256"]})
                    keys=[x["record_key"] for x in chain_items]
                    if len(set(keys))!=len(keys): raise RuntimeError("duplicate_listing_record_or_repeated_page")
                    if total is not None and len(chain_items)!=total: raise RuntimeError("listing_record_count_mismatch")
                    if total is None and (len(chain_items)==0 or any(v["record_count"]==0 for v in page_validations if v["source"]==src["id"] and v["query"]==q and v["year"]==y)): raise RuntimeError("mafra_page_without_records")
                    listings.extend(chain_items); chains.append({"source":src["id"],"query":q,"year":y,"ui_total":total,"derived_record_count":len(chain_items),"page_count":pages,"count_valid":True})
            # Keep every detail and attachment candidate; never infer crop eligibility from title substring.
            # Distinguish discovery scope from the requested 90-day capture set.
            out_of_window=0; unknown_date=0; in_window=[]
            for item in listings:
                item["capture_disposition"]=_within_window(item,start,end)
                if item["capture_disposition"]=="date_unknown_not_fetched": unknown_date+=1
                elif item["capture_disposition"]=="outside_rolling_window_not_fetched": out_of_window+=1
                else: in_window.append(item)
            unresolved=[x for x in listings if x["capture_disposition"]=="date_unknown_not_fetched"]
            missing_in_scope=[x for x in in_window if not x.get("url")]
            if unresolved or missing_in_scope:
                raise RuntimeError(f"listing_records_unresolved:unknown_date={len(unresolved)};missing_in_scope_url={len(missing_in_scope)}")
            unique={}
            for item in in_window:
                if not item.get("url"): continue
                unique.setdefault(item["url"],item)
            details=[x for x in unique.values() if x["kind"]=="detail"]
            documents=[x for x in unique.values() if x["kind"]=="pdf"]
            for item in details:
                body,obs=fetch.get(item["url"],"detail",item)
                item["detail_sha256"]=obs["sha256"]; item["detail_retrieved_at"]=obs["retrieved_at"]
                documents.extend(_details(body,item["url"],item))
            seen_doc={}
            for x in documents: seen_doc.setdefault(x["url"],x)
            for item in seen_doc.values():
                body,obs=fetch.get(item["url"],"document",item)
                if not body.startswith(b"%PDF-"): raise RuntimeError(f"pdf_signature_invalid:{item['url']}")
                item.update({"sha256":obs["sha256"],"byte_length":obs["byte_length"],"first_seen_at":obs["retrieved_at"],"last_seen_at":obs["retrieved_at"],"object_path":obs.get("object_path"),"availability_status":"current_capture_only","earliest_usable_at":obs["retrieved_at"],"content_type":next((v for k,v in obs["headers"].items() if k.lower()=="content-type"),None)})
                if "krei" in (item.get("title","")+" "+item["url"]).lower(): item["classification_status"]="queued_source_review_krei_possible"
                else: item["classification_status"]="discovery_candidate_unreviewed"
            # Update cross-run URL+hash history only after all current captures are complete.
            index_path=data_root/"url_hash_index.json"; index=json.loads(index_path.read_text()) if index_path.exists() else {"protocol":PROTOCOL,"urls":{}}
            change_counts={"new_url":0,"changed_hash":0,"unchanged_hash":0}
            latest_observations={}
            for obs in reversed(cp.get("observations", [])):
                if obs.get("ok") and obs.get("sha256"): latest_observations.setdefault(obs["url"],obs)
            for obs in latest_observations.values():
                rec=index["urls"].setdefault(obs["url"],{"versions":[]})
                previous=rec.get("latest_sha256") or (rec["versions"][-1]["sha256"] if rec["versions"] else None)
                if previous is None: obs["change_status"]="new_url"; change_counts["new_url"]+=1
                elif previous==obs["sha256"]: obs["change_status"]="unchanged_hash"; change_counts["unchanged_hash"]+=1
                else: obs["change_status"]="changed_hash"; change_counts["changed_hash"]+=1
                version=next((v for v in rec["versions"] if v["sha256"]==obs["sha256"]),None)
                if version is None: rec["versions"].append({"sha256":obs["sha256"],"first_seen_at":obs["retrieved_at"],"last_seen_at":obs["retrieved_at"],"run_id":run_id,"kind":obs["kind"]})
                else: version["last_seen_at"]=obs["retrieved_at"]
                rec["latest_sha256"]=obs["sha256"]; rec["latest_seen_at"]=obs["retrieved_at"]
            for item in seen_doc.values():
                prior=index["urls"].get(item["url"],{}).get("versions",[])
                match=next((v for v in prior if v["sha256"]==item["sha256"]),None)
                if match: item["first_seen_at"]=match["first_seen_at"]
            _json_write(index_path,index)
            report={"protocol":PROTOCOL,"run_id":run_id,"scope":frozen,"status":"complete","request_count":fetch.total,"request_count_this_invocation":fetch.count,"request_limit_per_invocation":config["limits"]["max_requests"],"response_change_counts":change_counts,"page_chains":chains,"page_validations":page_validations,"empty_queries":[{"source":x["source"],"query":x["query"],"year":x["year"],"source_reported_zero":x["ui_total"]==0} for x in chains if x["ui_total"]==0],"page_chains_count_valid":all(x["count_valid"] for x in chains),"listing_item_count":len(listings),"out_of_window_discovery_count":out_of_window,"unknown_date_discovery_count":unknown_date,"detail_count":len(details),"document_count":len(seen_doc),"documents":list(seen_doc.values()),"empty_query_is_no_signal":False,"attachment_selection":"PDF originals only; non-PDF formats are explicitly excluded from this collection path","limitations":["displayed dates do not prove first publication or historical availability","source/crop/quantity eligibility requires item-level review; no automatic supply classification","KREI-origin facts are not accepted; mixed-source items remain queued"]}
            _json_write(run_dir/"run.json",report); _json_write(run_dir/"checkpoint.json",{**cp,"status":"complete","completed_at":_now(),"request_count":fetch.total,"request_count_this_invocation":fetch.count,"scope":frozen,"page_chains":chains,"page_validations":page_validations,"items":list(seen_doc.values())})
            _json_write(latest_path,{"run_id":run_id,"status":"complete","completed_at":_now(),"report":"runs/"+run_id+"/run.json"})
            return report
        except Exception as e:
            cp["status"]="incomplete"; cp["last_error"]=str(e); cp["request_count"]=getattr(locals().get("fetch",None),"total",cp.get("request_count",0)); cp["request_count_this_invocation"]=getattr(locals().get("fetch",None),"count",0); cp["updated_at"]=_now(); _json_write(run_dir/"checkpoint.json",cp)
            _json_write(latest_path,{"run_id":run_id,"status":"incomplete","updated_at":_now(),"checkpoint":"runs/"+run_id+"/checkpoint.json"})
            raise


def status(config_path: Path):
    c=_load_cfg(config_path); root=ROOT/c["data_root"]; lp=root/"latest.json"
    if not lp.exists(): return {"status":"not_started","data_root":str(root)}
    latest=json.loads(lp.read_text()); cp=json.loads((root/"runs"/latest["run_id"]/("run.json" if latest["status"]=="complete" else "checkpoint.json")).read_text())
    return {"run_id":latest["run_id"],"status":latest["status"],"request_count":cp.get("request_count"),"last_error":cp.get("last_error"),"report":latest.get("report"),"checkpoint":latest.get("checkpoint")}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("command",choices=["run","status"]); p.add_argument("--config",type=Path,default=DEFAULT_CONFIG); p.add_argument("--new-run",action="store_true",help="start a new snapshot instead of resuming latest incomplete")
    a=p.parse_args(argv)
    try:
        result=run(a.config,resume=not a.new_run) if a.command=="run" else status(a.config)
        print(json.dumps(result if a.command=="status" else {k:v for k,v in result.items() if k!="documents"},ensure_ascii=False,sort_keys=True)); return 0
    except Exception as e:
        print(json.dumps({"status":"incomplete","error":str(e)},ensure_ascii=False),file=sys.stderr); return 2

if __name__=="__main__": raise SystemExit(main())
