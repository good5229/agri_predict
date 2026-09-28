from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import sqlite3
import subprocess
import sys
import time
import urllib.parse
import uuid
import xml.etree.ElementTree as ET
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

UTC = dt.timezone.utc
PORTAL = "https://apis.data.go.kr"
CUSTOMS = "https://apis.data.go.kr/1220000/Itemtrade/getItemtradeList"
KOSIS = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
ASOS = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
PRICE_GROUPS = {"cabbage": ("10", "01"), "radish": ("11", "01"), "onion": ("12", "01"), "garlic": ("12", "09"), "pepper": ("12", "07")}
GARLIC_SUBCODES = ("06", "16", "41", "42", "43")
KOSIS_TABLES = {
    "DT_1ET0028": ("T06", "T10"), "DT_1ET0029": ("T06", "T10"),
    "DT_1ET0291": ("T10", "T14", "T76", "T80", "T88", "T92"),
}
CUSTOMS_CODES = ("070310", "070320", "070490", "070690", "090420", "090421")
SECRET_KEYS = ("DATA_GO_KR_SERVICE_KEY_DECODED", "KOSIS_API_KEY")
_LAST_REQUEST_AT = 0.0


class CollectionError(RuntimeError):
    pass


class BudgetExhausted(CollectionError):
    pass


class RequestBudget:
    def __init__(self, limit: int | None):
        self.limit=limit
        self.used=0

    def reserve(self) -> int:
        if self.limit is not None and self.used>=self.limit:
            raise BudgetExhausted("request budget exhausted; resume to continue")
        self.used+=1
        return self.used


def redact(value: str, secrets: Iterable[str] = ()) -> str:
    for secret in secrets:
        if secret and len(secret) >= 4:
            value = value.replace(secret, "[REDACTED]")
            value = value.replace(urllib.parse.quote(secret, safe=""), "[REDACTED]")
    return value


def sanitize(value: Any, secrets: Iterable[str]) -> Any:
    if isinstance(value,str): return redact(value,secrets)
    if isinstance(value,list): return [sanitize(x,secrets) for x in value]
    if isinstance(value,dict): return {sanitize(k,secrets):sanitize(v,secrets) for k,v in value.items()}
    return value


def canonical_date(value: Any) -> str | None:
    if value in (None,""): return None
    text=str(value).strip()
    if re.fullmatch(r"\d{4}",text): return text+"-01-01"
    if re.fullmatch(r"\d{8}",text):
        try: return dt.datetime.strptime(text,"%Y%m%d").date().isoformat()
        except ValueError: return None
    try: return dt.date.fromisoformat(text[:10]).isoformat()
    except ValueError: return None


def api_result_code(body: str) -> str | None:
    try:
        if body.lstrip().startswith("<"):
            root=ET.fromstring(body)
            node=root.find("header") or root.find("cmmMsgHeader")
            if node is None:return None
            return node.findtext("resultCode") or node.findtext("returnReasonCode")
        obj=json.loads(body)
        if not isinstance(obj,dict): return None
        if "response" in obj: return str(obj.get("response",{}).get("header",{}).get("resultCode"))
        return str(obj.get("OpenAPI_ServiceResponse",{}).get("cmmMsgHeader",{}).get("returnReasonCode"))
    except (ET.ParseError,json.JSONDecodeError,AttributeError): return None


def load_env(path: str | None = None) -> dict[str, str]:
    env = dict(os.environ)
    if path:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip("\"'")
            if key in SECRET_KEYS:
                env[key] = value
    return env


def iso_date(value: str) -> dt.date:
    try:
        result = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid date {value!r}; use YYYY-MM-DD") from exc
    if result > dt.datetime.now(ZoneInfo("Asia/Seoul")).date():
        raise argparse.ArgumentTypeError("future dates are not allowed")
    return result


def date_range(start: dt.date, end: dt.date):
    if end < start:
        raise CollectionError("end date precedes start date")
    day = start
    while day <= end:
        yield day
        day += dt.timedelta(days=1)


def bounded_date_chunks(start: dt.date,end: dt.date,max_days: int=365):
    if end<start: raise CollectionError("end date precedes start date")
    cursor=start
    while cursor<=end:
        stop=min(end,cursor+dt.timedelta(days=max_days-1))
        yield cursor,stop
        cursor=stop+dt.timedelta(days=1)


def open_db(path: str, *, readonly: bool = False) -> sqlite3.Connection:
    if readonly:
        if not Path(path).is_file(): raise CollectionError(f"database does not exist: {path}")
        uri="file:"+urllib.parse.quote(str(Path(path).resolve()),safe="/")+"?mode=ro"
        db=sqlite3.connect(uri,uri=True,timeout=30)
        db.execute("PRAGMA foreign_keys=ON")
        return db
    db = sqlite3.connect(path, timeout=30)
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS jobs (
      job_id INTEGER PRIMARY KEY, command TEXT NOT NULL, config_json TEXT NOT NULL,
      started_at TEXT NOT NULL, finished_at TEXT, state TEXT NOT NULL DEFAULT 'running',
      expected_requests INTEGER, completed_requests INTEGER NOT NULL DEFAULT 0,
      error TEXT, actual_http_attempts INTEGER NOT NULL DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS source_requests (
      request_id INTEGER PRIMARY KEY, job_id INTEGER NOT NULL REFERENCES jobs(job_id),
      request_key TEXT NOT NULL, epoch_id TEXT NOT NULL DEFAULT '', dataset TEXT NOT NULL, source_url TEXT NOT NULL,
      query_json TEXT NOT NULL, retrieved_at TEXT NOT NULL, response_sha256 TEXT,
      http_status INTEGER, api_status TEXT, expected_count INTEGER, returned_count INTEGER,
      page_no INTEGER, state TEXT NOT NULL, error_type TEXT, error_message TEXT,
      response_text TEXT, attempt_count INTEGER NOT NULL DEFAULT 0,
      published_at TEXT, version_published_at TEXT,
      vintage_status TEXT NOT NULL DEFAULT 'current_retrieved_vintage',
      availability_basis TEXT NOT NULL DEFAULT 'retrieved_at; source publication/version time unknown'
    );
    CREATE TABLE IF NOT EXISTS raw_records (
      request_id INTEGER NOT NULL REFERENCES source_requests(request_id), row_no INTEGER NOT NULL,
      payload_json TEXT NOT NULL, PRIMARY KEY(request_id, row_no)
    );
    CREATE TABLE IF NOT EXISTS normalized_records (
      request_id INTEGER NOT NULL REFERENCES source_requests(request_id), row_no INTEGER NOT NULL,
      dataset TEXT NOT NULL, reference_date TEXT, payload_json TEXT NOT NULL,
      quality_flags TEXT NOT NULL DEFAULT '[]', PRIMARY KEY(request_id, row_no)
    );
    CREATE TABLE IF NOT EXISTS planned_queries (
      job_id INTEGER NOT NULL REFERENCES jobs(job_id), task_key TEXT NOT NULL,
      dataset TEXT NOT NULL, source_url TEXT NOT NULL, query_json TEXT NOT NULL,
      state TEXT NOT NULL DEFAULT 'pending', row_count INTEGER, error TEXT, epoch_id TEXT,
      PRIMARY KEY(job_id, task_key)
    );
    CREATE TABLE IF NOT EXISTS request_attempts (
      attempt_id INTEGER PRIMARY KEY, job_id INTEGER NOT NULL REFERENCES jobs(job_id),
      request_key TEXT NOT NULL, epoch_id TEXT NOT NULL, attempt_no INTEGER NOT NULL,
      started_at TEXT NOT NULL, finished_at TEXT NOT NULL, http_status INTEGER,
      response_sha256 TEXT, error_type TEXT, error_message TEXT, response_text TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_norm_dataset_date ON normalized_records(dataset, reference_date);
    CREATE INDEX IF NOT EXISTS idx_req_state ON source_requests(state);
    """)
    migrations={
      "jobs":{"actual_http_attempts":"INTEGER NOT NULL DEFAULT 0"},
      "source_requests":{"epoch_id":"TEXT NOT NULL DEFAULT ''","attempt_count":"INTEGER NOT NULL DEFAULT 0","published_at":"TEXT","version_published_at":"TEXT","vintage_status":"TEXT NOT NULL DEFAULT 'current_retrieved_vintage'","availability_basis":"TEXT NOT NULL DEFAULT 'retrieved_at; source publication/version time unknown'"},
      "planned_queries":{"epoch_id":"TEXT","source_url":"TEXT NOT NULL DEFAULT ''"},
    }
    for table,columns in migrations.items():
        existing={row[1] for row in db.execute(f"PRAGMA table_info({table})")}
        for name,decl in columns.items():
            if name not in existing: db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
    db.commit()
    return db


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def stable_request_key(dataset: str, url: str, params: dict[str, str], epoch_id: str = "") -> str:
    return hashlib.sha256(json_text([dataset, url, sorted(params.items()),epoch_id]).encode()).hexdigest()


def page_is_complete(db: sqlite3.Connection, job: int, dataset: str, url: str,
                     params: dict[str, str], refresh: bool = False, epoch_id: str = "") -> bool:
    if refresh: return False
    key=stable_request_key(dataset,url,params,epoch_id)
    row=db.execute("SELECT state FROM source_requests WHERE job_id=? AND request_key=? ORDER BY request_id DESC LIMIT 1",(job,key)).fetchone()
    return bool(row and row[0]=="complete")


def task_key(dataset: str, url: str, params: dict[str,str]) -> str:
    return stable_request_key(dataset,url,params)


def mark_task(db: sqlite3.Connection, job: int, dataset: str, url: str,
              params: dict[str,str], state: str, row_count: int | None = None, error: str | None = None,
              epoch_id: str | None = None):
    db.execute("UPDATE planned_queries SET state=?,row_count=?,error=?,epoch_id=COALESCE(?,epoch_id) WHERE job_id=? AND task_key=?",
               (state,row_count,error,epoch_id,job,task_key(dataset,url,params)))
    db.commit()


def build_manifest(db: sqlite3.Connection, job: int, args) -> None:
    tasks=[]
    if args.command=="prices":
        tasks.extend((("codes_markets",f"{PORTAL}/B552845/katCode/wholesaleMarkets",{}),
                      ("codes_goods",f"{PORTAL}/B552845/katCode/goods",{})))
        crops=args.crop or list(PRICE_GROUPS)
        src="origin" if args.source=="origin" else "sale"
        endpoint=f"{PORTAL}/B552845/kat{'Origin' if src=='origin' else 'Sale'}/trades"
        for day in date_range(args.start,args.end):
            for market in args.market or []:
                for crop in crops:
                    large,medium=PRICE_GROUPS[crop]
                    subcodes=GARLIC_SUBCODES if crop=="garlic" and args.garlic_subcodes else (None,)
                    for sub in subcodes:
                        q={"cond[trd_clcln_ymd::EQ]":day.isoformat(),"cond[whsl_mrkt_cd::EQ]":market,"cond[gds_lclsf_cd::EQ]":large,"cond[gds_mclsf_cd::EQ]":medium}
                        if sub:q["cond[gds_sclsf_cd::EQ]"]=sub
                        tasks.append(("prices_"+src,endpoint,q))
    elif args.command=="imports":
        for a,b in month_chunks(args.start,args.end):
            for hs in args.hs_code or CUSTOMS_CODES:
                tasks.append(("imports",CUSTOMS,{"strtYymm":a,"endYymm":b,"hsSgn":hs}))
    elif args.command=="production":
        for table in args.table or list(KOSIS_TABLES):
            q={"method":"getList","orgId":"101","tblId":table,"objL1":"00","itmId":"ALL","prdSe":"Y","format":"json","jsonVD":"Y","startPrdDe":str(args.start.year),"endPrdDe":str(args.end.year)}
            tasks.append(("production",KOSIS,q))
    elif args.command=="weather":
        for start,end in bounded_date_chunks(args.start,args.end):
            for station in args.station:
                tasks.append(("weather",ASOS,{"dataCd":"ASOS","dateCd":"DAY","dataType":"JSON","startDt":start.strftime("%Y%m%d"),"endDt":end.strftime("%Y%m%d"),"stnIds":station}))
    with db:
        for dataset,url,q in tasks:
            db.execute("INSERT OR IGNORE INTO planned_queries(job_id,task_key,dataset,source_url,query_json) VALUES(?,?,?,?,?)",
                       (job,task_key(dataset,url,q),dataset,url,json_text(q)))
        db.execute("UPDATE jobs SET expected_requests=? WHERE job_id=?",(len(tasks),job))


def task_call(db,job,dataset,url,params,func,refresh=False):
    old=db.execute("SELECT state,epoch_id FROM planned_queries WHERE job_id=? AND task_key=?",
                   (job,task_key(dataset,url,params))).fetchone()
    if old and old[0] in ("complete","empty") and not refresh:
        epoch_id=old[1] or uuid.uuid4().hex
    else:
        epoch_id=uuid.uuid4().hex
    mark_task(db,job,dataset,url,params,"incomplete",epoch_id=epoch_id)
    try:
        rows=func(epoch_id)
    except Exception as exc:
        state="incomplete" if "budget exhausted" in str(exc) else "failed"
        mark_task(db,job,dataset,url,params,state,error=redact(str(exc)),epoch_id=epoch_id)
        raise
    count=rows[1] if isinstance(rows,tuple) and len(rows)==2 else rows
    mark_task(db,job,dataset,url,params,"empty" if count==0 else "complete",row_count=count,epoch_id=epoch_id)
    return rows


def begin_job(db: sqlite3.Connection, command: str, config: dict[str, Any], total: int | None = None) -> int:
    cur = db.execute("INSERT INTO jobs(command,config_json,started_at,expected_requests) VALUES(?,?,?,?)",
                     (command, json_text(config), dt.datetime.now(UTC).isoformat(), total))
    db.commit()
    return int(cur.lastrowid)


def resume_or_begin_job(db: sqlite3.Connection, command: str, config: dict[str, Any]) -> int:
    comparable = {k:v for k,v in config.items() if k not in ("max_requests", "refresh", "env_file")}
    encoded = json_text(comparable)
    row = db.execute("SELECT job_id FROM jobs WHERE command=? AND config_json=? AND state IN ('incomplete','interrupted','running') ORDER BY job_id DESC LIMIT 1",
                     (command, encoded)).fetchone()
    if row:
        db.execute("UPDATE jobs SET started_at=?,finished_at=NULL,state='running',error=NULL WHERE job_id=?",
                   (dt.datetime.now(UTC).isoformat(), row[0]))
        db.commit()
        return int(row[0])
    return begin_job(db,command,comparable)


def finish_job(db: sqlite3.Connection, job_id: int, state: str, error: str | None = None) -> None:
    db.execute("UPDATE jobs SET finished_at=?,state=?,error=? WHERE job_id=?",
               (dt.datetime.now(UTC).isoformat(), state, error, job_id))
    db.commit()


def encode_url(url: str, params: dict[str, str]) -> str:
    return url + "?" + urllib.parse.urlencode(params, doseq=True)


def curl_fetch(url: str, timeout: int = 25) -> tuple[int, str, str]:
    # The URL is supplied through curl's stdin config, avoiding secrets in argv.
    cfg = "url = " + json.dumps(url) + "\nmax-time = " + str(timeout) + "\nfail = false\ninclude = true\n"
    try:
        p = subprocess.run(["/usr/bin/curl", "--silent", "--show-error", "--config", "-"],
                           input=cfg, capture_output=True, text=True, timeout=timeout + 5, check=False)
    except subprocess.TimeoutExpired as exc:
        return 0, "", str(exc)
    except OSError as exc:
        return 0,"",str(exc)
    if p.returncode:
        return 0, p.stdout, p.stderr or f"curl exit {p.returncode}"
    # Consume only leading HTTP header blocks; blank lines inside JSON/XML bodies remain intact.
    remainder=p.stdout
    status=0
    while remainder.startswith("HTTP/"):
        header,sep,tail=remainder.partition("\r\n\r\n")
        if not sep: header,sep,tail=remainder.partition("\n\n")
        if not sep: break
        match=re.match(r"HTTP/\S+\s+(\d{3})",header)
        if match: status=int(match.group(1))
        remainder=tail
    return status, remainder, p.stderr


def decode_api(dataset: str, body: str) -> tuple[list[dict[str, Any]], int, str]:
    if dataset=="imports" and body.lstrip().startswith("<"):
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            raise CollectionError("corrupt XML API response") from exc
        cmm = root.find("cmmMsgHeader")
        if cmm is not None:
            code = cmm.findtext("returnReasonCode") or "unknown"
            message = cmm.findtext("errMsg") or cmm.findtext("returnAuthMsg") or "API error"
            raise CollectionError(f"API failure {code}: {message[:160]}")
        header = root.find("header")
        code = (header.findtext("resultCode") if header is not None else "") or ""
        msg = (header.findtext("resultMsg") if header is not None else "") or ""
        if code!="00":
            raise CollectionError(f"API failure {code}: {msg[:160]}")
        body_node = root.find("body")
        if body_node is None: raise CollectionError("XML API response has no body")
        items_node = body_node.find("items")
        if items_node is None: raise CollectionError("XML API response has no items element")
        rows = [
            {child.tag: child.text for child in item} for item in items_node.findall("item")]
        total_text=body_node.findtext("totalCount")
        try: count=int(total_text) if total_text is not None else len(rows)
        except ValueError as exc: raise CollectionError("invalid totalCount") from exc
        return rows,count,code
    try:
        obj = json.loads(body,parse_float=str)
    except (json.JSONDecodeError, TypeError) as exc:
        raise CollectionError("corrupt or non-JSON API response") from exc
    if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
        if dataset=="production" and any(not {"ITM_ID","TBL_ID","PRD_DE","DT"}.issubset(row) for row in obj):
            raise CollectionError("KOSIS row lacks required fields")
        return obj, len(obj), "ok"
    if isinstance(obj,dict) and isinstance(obj.get("OpenAPI_ServiceResponse"),dict):
        cmm=obj["OpenAPI_ServiceResponse"].get("cmmMsgHeader",{})
        raise CollectionError(f"API failure {cmm.get('returnReasonCode','unknown')}: {cmm.get('errMsg','API error')}")
    if isinstance(obj, dict) and "response" in obj:
        resp = obj["response"]
        if not isinstance(resp,dict) or not isinstance(resp.get("header"),dict): raise CollectionError("API response lacks header")
        header = resp["header"]
        code = str(header.get("resultCode", ""))
        msg = str(header.get("resultMsg", ""))
        if dataset=="weather" and code=="03" and msg in ("NODATA_ERROR","NO_DATA"):
            return [],0,"03:"+msg
        if code not in ("0", "00"):
            raise CollectionError(f"API failure {code}: {msg[:160]}")
        body_obj=resp.get("body")
        if not isinstance(body_obj,dict): raise CollectionError("API response lacks body")
        items_container=body_obj.get("items")
        if items_container is None: raise CollectionError("API response lacks items")
        if "totalCount" not in body_obj: raise CollectionError("API response lacks totalCount")
        try: count=int(body_obj["totalCount"])
        except (TypeError,ValueError) as exc: raise CollectionError("invalid totalCount") from exc
        if count<0: raise CollectionError("negative totalCount")
        items = items_container
        if isinstance(items, dict):
            if "item" not in items: raise CollectionError("API items lack item field")
            items = items["item"]
        if items is None: raise CollectionError("API items is null")
        if isinstance(items, dict): items = [items]
        if not isinstance(items, list) or not all(isinstance(row,dict) for row in items): raise CollectionError("unexpected API items structure")
        if count<len(items): raise CollectionError("totalCount smaller than returned page")
        return items,count,code
    if isinstance(obj, dict) and "result" in obj and isinstance(obj["result"], dict):
        result = obj["result"]
        if str(result.get("code", "")).lower() not in ("", "success", "0"):
            raise CollectionError("API failure: " + str(result.get("msg", "unknown"))[:160])
    if isinstance(obj, dict) and isinstance(obj.get("row"), list):
        return obj["row"], len(obj["row"]), "ok"
    raise CollectionError("unrecognized API response structure")


def request_page(db: sqlite3.Connection, job_id: int, dataset: str, url: str,
                 params: dict[str, str], api_key: str, *, refresh: bool = False,
                 retries: int = 3, timeout: int = 25, key_name: str = "serviceKey", rate: float = 2.0,
                 epoch_id: str = "", budget=None) -> tuple[list[dict[str, Any]], int, int]:
    request_params = dict(params)
    request_params[key_name] = api_key
    full_url = encode_url(url, request_params)
    key = stable_request_key(dataset, url, params,epoch_id)
    old = db.execute("SELECT request_id,state,response_text,expected_count,http_status FROM source_requests WHERE job_id=? AND request_key=? ORDER BY request_id DESC LIMIT 1",
                     (job_id, key)).fetchone()
    if old and old[1] == "complete" and not refresh:
        rows, count, _ = decode_api(dataset, old[2] or "")
        return rows, count, int(old[0])
    last_error = "request failed"
    last_body = ""
    last_status = 0
    secrets = [api_key]
    attempts=0
    for attempt in range(retries):
        global _LAST_REQUEST_AT
        delay=1.0/max(rate,0.01)-(time.monotonic()-_LAST_REQUEST_AT)
        if _LAST_REQUEST_AT and delay>0: time.sleep(delay)
        _LAST_REQUEST_AT=time.monotonic()
        if budget is not None:
            try: attempt_no=budget.reserve()
            except BudgetExhausted:
                if attempts:
                    insert_request_result(db,job_id,key,epoch_id,dataset,url,params,
                        state="incomplete",error_type="budget",error_message="request budget exhausted during retries",
                        body=last_body,status=last_status,attempt_count=attempts,secrets=secrets)
                raise
        else: attempt_no=attempt+1
        attempts+=1
        attempt_started=dt.datetime.now(UTC).isoformat()
        status, body, stderr = curl_fetch(full_url, timeout)
        attempt_finished=dt.datetime.now(UTC).isoformat()
        last_status,last_body=status,body
        attempt_error=(f"HTTP {status}" if status and not 200<=status<300 else (stderr if not status else None))
        with db:
            db.execute("INSERT INTO request_attempts(job_id,request_key,epoch_id,attempt_no,started_at,finished_at,http_status,response_sha256,error_type,error_message,response_text) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (job_id,key,epoch_id,attempt_no,attempt_started,attempt_finished,status or None,
                 hashlib.sha256(body.encode()).hexdigest() if body else None,
                 "http" if status and not 200<=status<300 else ("transport" if not status else None),
                 redact(attempt_error,secrets) if attempt_error else None,redact(body,secrets) if body else None))
            db.execute("UPDATE jobs SET actual_http_attempts=actual_http_attempts+1 WHERE job_id=?",(job_id,))
        if status and 200 <= status < 300:
            try:
                rows, count, api_status = decode_api(dataset, body)
                if "pageNo" in params or "numOfRows" in params:
                    scope=api_page_scope(body)
                    for key_name in ("pageNo","numOfRows"):
                        if key_name in params and key_name in scope and str(scope[key_name])!=str(params[key_name]):
                            raise CollectionError(f"API returned unexpected {key_name}")
                if dataset=="imports" and count!=len(rows):
                    raise CollectionError(f"response row count mismatch: declared {count}, received {len(rows)}")
                rows = sanitize(rows,[api_key])
                safe_body = redact(body, [api_key])
                digest = hashlib.sha256(body.encode()).hexdigest()
                with db:
                    cur = db.execute("""INSERT INTO source_requests(job_id,request_key,epoch_id,dataset,source_url,query_json,retrieved_at,response_sha256,http_status,api_status,expected_count,returned_count,page_no,state,response_text,attempt_count)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (job_id,key,epoch_id,dataset,url,json_text(params),dt.datetime.now(UTC).isoformat(),digest,status,api_status,count,len(rows),params.get("pageNo"),"complete",safe_body,attempts))
                    reqid = int(cur.lastrowid)
                    for i, row in enumerate(rows):
                        payload = json_text(row)
                        db.execute("INSERT INTO raw_records VALUES(?,?,?)", (reqid,i,payload))
                        refdate = next((canonical_date(row[k]) for k in ("trd_clcln_ymd","basDt","reference_date","PRD_DE","prdDe","date","tm") if row.get(k) and canonical_date(row[k])), None)
                        if not refdate and row.get("year") and re.match(r"^\d{4}\.\d{2}$",str(row["year"])):
                            refdate=str(row["year"]).replace(".","-")+"-01"
                        flags = quality_flags(dataset,row)
                        db.execute("INSERT INTO normalized_records VALUES(?,?,?,?,?,?)", (reqid,i,dataset,refdate,payload,json_text(flags)))
                    db.execute("UPDATE jobs SET completed_requests=completed_requests+1 WHERE job_id=?",(job_id,))
                return rows, count, reqid
            except (CollectionError,ValueError,TypeError,KeyError) as exc:
                last_error = str(exc)
                # A declared success with a malformed/incomplete body is a response failure.
                with db:
                    db.execute("UPDATE request_attempts SET error_type='api',error_message=? WHERE job_id=? AND request_key=? AND epoch_id=? AND attempt_no=?",
                               (redact(last_error,secrets),job_id,key,epoch_id,attempt_no))
                if "API failure" in last_error or not isinstance(exc,CollectionError):
                    break
                if "response row count mismatch" in last_error or "corrupt" in last_error or "unrecognized" in last_error or "no body" in last_error:
                    break
        else:
            last_error = f"HTTP {status}" if status else "transport error: " + (stderr or "unknown")
            if status in (400,401,403,404) or (status and 400 <= status < 500 and status != 429): break
        if attempt + 1 < retries:
            time.sleep(min(8, 0.5 * (2 ** attempt)))
    msg = redact(last_error, secrets)
    final_error_type="api" if (last_status and 200<=last_status<300) else ("http" if last_status else "transport")
    reqid=insert_request_result(db,job_id,key,epoch_id,dataset,url,params,state="failed",
        error_type=final_error_type,
        error_message=msg,body=last_body,status=last_status,attempt_count=attempts,secrets=secrets)
    raise CollectionError(msg)


def insert_request_result(db,job_id,key,epoch_id,dataset,url,params,*,state,error_type,error_message,body,status,attempt_count,secrets):
    safe_body=redact(body,secrets) if body else None
    digest=hashlib.sha256(body.encode()).hexdigest() if body else None
    api_status=api_result_code(body) if body else None
    with db:
        cur=db.execute("""INSERT INTO source_requests(job_id,request_key,epoch_id,dataset,source_url,query_json,retrieved_at,response_sha256,http_status,api_status,state,error_type,error_message,response_text,attempt_count)
          VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",(job_id,key,epoch_id,dataset,url,json_text(params),dt.datetime.now(UTC).isoformat(),digest,status or None,api_status,state,error_type,error_message,safe_body,attempt_count))
    return int(cur.lastrowid)


def api_page_scope(body: str) -> dict[str,Any]:
    try:
        obj=json.loads(body,parse_float=str)
        body_obj=obj.get("response",{}).get("body",{}) if isinstance(obj,dict) else {}
        return {name:body_obj[name] for name in ("pageNo","numOfRows") if name in body_obj}
    except (json.JSONDecodeError,AttributeError): return {}


def quality_flags(dataset: str, row: dict[str, Any]) -> list[str]:
    flags = []
    for key in ("scsbd_prc", "qty", "unit_qty", "unit_tot_qty", "totprc", "avgprc", "unit_tot_prc",
                "impDlr","impWgt","expDlr","expWgt","balPayments","DT"):
        if key in row and row[key] not in (None, ""):
            try:
                if Decimal(str(row[key])) < 0: flags.append("negative:" + key)
            except InvalidOperation: flags.append("not_numeric:" + key)
    try:
        price, qty, total = (Decimal(str(row[k])) for k in ("scsbd_prc","qty","totprc"))
        if price * qty != total: flags.append("amount_mismatch:scsbd_prc*qty!=totprc")
    except (KeyError, InvalidOperation, TypeError): pass
    try:
        unit, qty, total = (Decimal(str(row[k])) for k in ("unit_qty","qty","unit_tot_qty"))
        if unit * qty != total: flags.append("amount_mismatch:unit_qty*qty!=unit_tot_qty")
    except (KeyError, InvalidOperation, TypeError): pass
    if dataset == "prices_origin" and not row.get("scsbd_dt"): flags.append("missing:scsbd_dt")
    if dataset == "production" and row.get("DT") in (None,"","-"): flags.append("missing_numeric:DT")
    if dataset=="imports" and (str(row.get("year",""))=="총계" or str(row.get("hsCode",row.get("hsCd",""))) in ("","-")):
        flags.append("is_total_row")
    return list(dict.fromkeys(flags))


def paged_collect(db: sqlite3.Connection, job: int, dataset: str, url: str, base: dict[str,str], key: str,
                  *, page_size: int = 1000, max_requests: int | None = None, refresh=False,
                  rate: float = 2.0, epoch_id: str = "", budget=None) -> tuple[int,int]:
    pages = 0
    page = 1
    total = None
    received = 0
    while True:
        params = dict(base, pageNo=str(page), numOfRows=str(page_size), returnType="json")
        fresh=not page_is_complete(db,job,dataset,url,params,refresh,epoch_id)
        if max_requests is not None and pages >= max_requests and fresh: raise CollectionError("request budget exhausted; resume to continue")
        rows, count, _ = request_page(db,job,dataset,url,params,key,refresh=refresh,rate=rate,epoch_id=epoch_id,budget=budget)
        if fresh: pages += 1
        if total is not None and count!=total: raise CollectionError(f"page count changed in query epoch: {total} -> {count}")
        total = count if total is None else total
        received += len(rows)
        if total is not None and received > total:
            raise CollectionError(f"pagination count exceeded: expected {total}, received {received}")
        # The API total count is the expected full query count; stop only when all expected rows arrived.
        if not rows:
            if total and received < total: raise CollectionError(f"truncated pagination: expected {total}, received {received}")
            break
        if total is not None and received >= total: break
        if len(rows) < page_size:
            if total is not None and received < total: raise CollectionError(f"truncated pagination: expected {total}, received {received}")
            break
        page += 1
    return pages,received


def parse_env_arg(parser: argparse.ArgumentParser):
    parser.add_argument("--env-file", help="explicit .env file containing approved API keys")
    parser.add_argument("--db", default="data/agri.sqlite", help="SQLite database path (default: data/agri.sqlite)")


def add_date_args(parser: argparse.ArgumentParser):
    parser.add_argument("--start", required=True, type=iso_date)
    parser.add_argument("--end", required=True, type=iso_date)


def positive_float(value: str) -> float:
    parsed=float(value)
    if not math.isfinite(parsed) or parsed<=0: raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed=int(value)
    if parsed<0: raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def validate_args(args):
    if args.end<args.start: raise CollectionError("end date precedes start date")
    if args.max_requests is not None and args.max_requests<0: raise CollectionError("max-requests must be zero or greater")
    if args.rate<=0: raise CollectionError("rate must be greater than zero")


def make_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(prog="agri-data", description="Collect and validate public agricultural data; no model training.")
    sub=p.add_subparsers(dest="command",required=True)
    for name in ("prices","imports","production","weather"):
        q=sub.add_parser(name); parse_env_arg(q); add_date_args(q)
        q.add_argument("--max-requests",type=nonnegative_int,help="maximum HTTP attempts for this run (zero creates a pending plan without network calls)")
        q.add_argument("--rate",type=positive_float,default=2.0,help="maximum requests per second")
        q.add_argument("--refresh",action="store_true",help="collect completed queries again as a new response version")
        if name=="prices":
            q.add_argument("--source",choices=("origin","sale"),default="origin")
            q.add_argument("--market",action="append",required=True,help="market code (repeatable)")
            q.add_argument("--crop",action="append",choices=tuple(PRICE_GROUPS),help="crop (repeatable; default all five)")
            q.add_argument("--garlic-subcodes",action="store_true",help="also request the five garlic subgroups")
        elif name=="imports": q.add_argument("--hs-code",action="append",help="HS prefix/code; default documented candidate groups")
        elif name=="production": q.add_argument("--table",action="append",choices=tuple(KOSIS_TABLES),help="KOSIS table (default all three)")
        else: q.add_argument("--station",action="append",required=True,help="ASOS station id (repeatable)")
    v=sub.add_parser("validate"); parse_env_arg(v); v.add_argument("--job-id",type=int)
    s=sub.add_parser("status"); parse_env_arg(s)
    e=sub.add_parser("export",help="optional Parquet export when pyarrow is already available")
    e.add_argument("--db",default="data/agri.sqlite"); e.add_argument("--output",required=True); e.add_argument("--job-id",required=True,type=int)
    e.add_argument("--dataset",action="append",help="dataset filter (repeatable)")
    return p


def execute(args) -> int:
    if args.command in ("prices","imports","production","weather"):
        validate_args(args)
        Path(args.db).parent.mkdir(parents=True,exist_ok=True)
    db=open_db(args.db,readonly=args.command in ("status","validate","export"))
    if args.command=="status":
        if db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]==0: raise CollectionError("database contains no collection jobs")
        for row in db.execute("SELECT job_id,command,state,started_at,finished_at,error FROM jobs ORDER BY job_id DESC LIMIT 30"):
            jid,command,state,started,finished,error=row
            counts={s:0 for s in ("complete","empty","incomplete","pending","failed")}
            counts.update({s:n for s,n in db.execute("SELECT state,COUNT(*) FROM planned_queries WHERE job_id=? GROUP BY state",(jid,))})
            reqs=db.execute("SELECT SUM(CASE WHEN state='complete' AND returned_count=0 THEN 1 ELSE 0 END),SUM(CASE WHEN state='failed' THEN 1 ELSE 0 END),SUM(CASE WHEN state='complete' THEN 1 ELSE 0 END) FROM source_requests WHERE job_id=?",(jid,)).fetchone()
            attempts=db.execute("SELECT actual_http_attempts FROM jobs WHERE job_id=?",(jid,)).fetchone()[0]
            print(json.dumps({"job_id":jid,"command":command,"state":state,"started_at":started,"finished_at":finished,"actual_http_attempts":attempts,
                "planned":sum(counts.values()),"queries":counts,"request_pages":{"complete":(reqs[2] or 0)-(reqs[0] or 0),"empty":reqs[0] or 0,"failed":reqs[1] or 0},"error":error},ensure_ascii=False))
        return 0
    if args.command=="validate": return validate(db,args.job_id)
    if args.command=="export": return export_parquet(db,args.output,args.dataset,args.job_id)
    env=load_env(args.env_file)
    portal_key=env.get("DATA_GO_KR_SERVICE_KEY_DECODED","")
    kosis_key=env.get("KOSIS_API_KEY","")
    if args.command in ("prices","imports","weather") and not portal_key: raise CollectionError("missing DATA_GO_KR_SERVICE_KEY_DECODED")
    if args.command=="production" and not kosis_key: raise CollectionError("missing KOSIS_API_KEY")
    config={k:(v.isoformat() if isinstance(v,dt.date) else v) for k,v in vars(args).items() if k not in ("env_file",)}
    config={k:v for k,v in config.items() if k not in ("max_requests", "refresh")}
    if args.command=="imports": config["month_chunk_months"]=12
    job=resume_or_begin_job(db,args.command,config)
    args._budget=RequestBudget(args.max_requests)
    try:
        build_manifest(db,job,args)
        if args.command=="prices": collect_prices(db,job,args,portal_key)
        elif args.command=="imports": collect_imports(db,job,args,portal_key)
        elif args.command=="production": collect_production(db,job,args,kosis_key)
        elif args.command=="weather": collect_weather(db,job,args,portal_key)
        pending=db.execute("SELECT COUNT(*) FROM planned_queries WHERE job_id=? AND state NOT IN ('complete','empty')",(job,)).fetchone()[0]
        finish_job(db,job,"complete" if pending==0 else "incomplete",None if pending==0 else f"{pending} planned queries remain")
        print(f"job {job} {'complete' if pending==0 else 'incomplete'}; pending queries: {pending}")
        return 0 if pending==0 else 2
    except KeyboardInterrupt:
        finish_job(db,job,"interrupted","interrupted by user")
        print(f"job {job} interrupted; resume with the same command",file=sys.stderr); return 130
    except Exception as exc:
        msg=redact(str(exc),[portal_key,kosis_key])
        finish_job(db,job,"incomplete",msg)
        print(f"job {job} incomplete: {msg}",file=sys.stderr); return 2


def collect_prices(db,job,args,key):
    crops=args.crop or list(PRICE_GROUPS)
    if not args.market: raise CollectionError("prices requires at least one --market code")
    endpoint=f"{PORTAL}/B552845/kat{ 'Origin' if args.source=='origin' else 'Sale'}/trades"
    dataset="prices_"+args.source
    for code_kind,code_path in (("markets","wholesaleMarkets"),("goods","goods")):
        code_url=f"{PORTAL}/B552845/katCode/{code_path}"
        task_call(db,job,"codes_"+code_kind,code_url,{},lambda epoch:paged_collect(
            db,job,"codes_"+code_kind,code_url,{},key,page_size=10000,rate=args.rate,refresh=args.refresh,epoch_id=epoch,budget=args._budget),refresh=args.refresh)
    for day in date_range(args.start,args.end):
        for market in args.market:
            for crop in crops:
                large,medium=PRICE_GROUPS[crop]
                subcodes=GARLIC_SUBCODES if crop=="garlic" and args.garlic_subcodes else (None,)
                for subcode in subcodes:
                    query={"cond[trd_clcln_ymd::EQ]":day.isoformat(),"cond[whsl_mrkt_cd::EQ]":market,"cond[gds_lclsf_cd::EQ]":large,"cond[gds_mclsf_cd::EQ]":medium}
                    if subcode: query["cond[gds_sclsf_cd::EQ]"]=subcode
                    task_call(db,job,dataset,endpoint,query,lambda epoch:paged_collect(db,job,dataset,endpoint,query,key,rate=args.rate,refresh=args.refresh,epoch_id=epoch,budget=args._budget),refresh=args.refresh)


def month_chunks(start: dt.date,end: dt.date):
    if end<start: raise CollectionError("end date precedes start date")
    first=start.year*12+start.month-1
    last=end.year*12+end.month-1
    cursor=first
    while cursor<=last:
        stop=min(last,cursor+11)
        start_year,start_month=divmod(cursor,12)
        end_year,end_month=divmod(stop,12)
        yield f"{start_year:04d}{start_month+1:02d}",f"{end_year:04d}{end_month+1:02d}"
        cursor=stop+1


def collect_imports(db,job,args,key):
    for a,b in month_chunks(args.start,args.end):
        for hs in args.hs_code or CUSTOMS_CODES:
            params={"strtYymm":a,"endYymm":b,"hsSgn":hs}
            # no page pagination is documented for this monthly endpoint; response row count is retained.
            task_call(db,job,"imports",CUSTOMS,params,lambda epoch:request_page(db,job,"imports",CUSTOMS,params,key,refresh=args.refresh,rate=args.rate,epoch_id=epoch,budget=args._budget)[:2],refresh=args.refresh)


def collect_production(db,job,args,key):
    for table in args.table or list(KOSIS_TABLES):
        params={"method":"getList","orgId":"101","tblId":table,"objL1":"00","itmId":"ALL","prdSe":"Y","format":"json","jsonVD":"Y","startPrdDe":str(args.start.year),"endPrdDe":str(args.end.year)}
        task_call(db,job,"production",KOSIS,params,lambda epoch:request_page(db,job,"production",KOSIS,params,key,key_name="apiKey",refresh=args.refresh,rate=args.rate,epoch_id=epoch,budget=args._budget)[:2],refresh=args.refresh)


def collect_weather(db,job,args,key):
    for start,end in bounded_date_chunks(args.start,args.end):
        for station in args.station:
            params={"dataCd":"ASOS","dateCd":"DAY","dataType":"JSON","startDt":start.strftime("%Y%m%d"),"endDt":end.strftime("%Y%m%d"),"stnIds":station}
            task_call(db,job,"weather",ASOS,params,lambda epoch:paged_collect(db,job,"weather",ASOS,params,key,page_size=999,rate=args.rate,refresh=args.refresh,epoch_id=epoch,budget=args._budget),refresh=args.refresh)


def validate(db: sqlite3.Connection,job: int|None=None) -> int:
    jobs=[r[0] for r in db.execute("SELECT job_id FROM jobs"+(" WHERE job_id=?" if job is not None else ""),((job,) if job is not None else ()))]
    if job is not None and not jobs: raise CollectionError(f"job does not exist: {job}")
    if not jobs: raise CollectionError("database contains no collection jobs")
    clauses=" WHERE r.job_id=?" if job is not None else ""
    params=(job,) if job is not None else ()
    active="""SELECT r.* FROM source_requests r JOIN planned_queries p
      ON p.job_id=r.job_id AND p.dataset=r.dataset AND p.source_url=r.source_url AND p.epoch_id=r.epoch_id
      WHERE p.state IN ('complete','empty')"""
    if job is not None: active+=" AND r.job_id=?"
    active_rows=db.execute(active,params).fetchall()
    failures=sum(1 for r in active_rows if r[14]!="complete")
    mismatches=0; raw_errors=0; normalized_errors=0; page_gaps=0; manifest_errors=0
    check_db=db.execute("PRAGMA foreign_key_check").fetchall()
    for request in active_rows:
        rid=request[0]; req_count=request[12] or 0
        raw_n=db.execute("SELECT COUNT(*) FROM raw_records WHERE request_id=?",(rid,)).fetchone()[0]
        norm_n=db.execute("SELECT COUNT(*) FROM normalized_records WHERE request_id=?",(rid,)).fetchone()[0]
        if raw_n!=req_count: raw_errors+=1
        if norm_n!=req_count: normalized_errors+=1
        if request[11] is not None and req_count>request[11]: mismatches+=1
    for jid in jobs:
        jobrow=db.execute("SELECT state,expected_requests,actual_http_attempts,completed_requests FROM jobs WHERE job_id=?",(jid,)).fetchone()
        totals=db.execute("SELECT COUNT(*),SUM(CASE WHEN state IN ('complete','empty') THEN 1 ELSE 0 END),SUM(CASE WHEN state='pending' THEN 1 ELSE 0 END),SUM(CASE WHEN state IN ('incomplete','failed') THEN 1 ELSE 0 END) FROM planned_queries WHERE job_id=?",(jid,)).fetchone()
        total_planned=totals[0] or 0
        if jobrow[1] != total_planned or jobrow[0]!="complete" or totals[2] or totals[3] or (totals[1] or 0)!=total_planned: manifest_errors+=1
        attempt_rows=db.execute("SELECT COUNT(*) FROM request_attempts WHERE job_id=?",(jid,)).fetchone()[0]
        if attempt_rows!=jobrow[2]: manifest_errors+=1
        completed_pages=db.execute("SELECT COUNT(*) FROM source_requests WHERE job_id=? AND state='complete' AND attempt_count>0",(jid,)).fetchone()[0]
        if completed_pages!=jobrow[3]: manifest_errors+=1
        attempt_mismatch=db.execute("""SELECT COUNT(*) FROM source_requests r WHERE r.job_id=? AND r.attempt_count>0 AND
          r.attempt_count!=(SELECT COUNT(*) FROM request_attempts a WHERE a.job_id=r.job_id AND a.request_key=r.request_key AND a.epoch_id=r.epoch_id)""",(jid,)).fetchone()[0]
        if attempt_mismatch: manifest_errors+=attempt_mismatch
        for ds,url,qraw,state,rowcount,epoch in db.execute("SELECT dataset,source_url,query_json,state,row_count,epoch_id FROM planned_queries WHERE job_id=? AND state IN ('complete','empty')",(jid,)):
            expected_query=json.loads(qraw)
            pages=[]
            for req in active_rows:
                if req[1]!=jid or req[4]!=ds or req[5]!=url or req[3]!=epoch: continue
                page_query=json.loads(req[6]); base={k:v for k,v in page_query.items() if k not in ("pageNo","numOfRows","returnType")}
                if base==expected_query: pages.append((int(page_query.get("pageNo","0")),req))
            pages.sort(key=lambda x:x[0])
            if ds in ("imports", "production"):
                if len(pages)!=1:
                    page_gaps+=1
                    continue
                page_query=json.loads(pages[0][1][6])
                request=pages[0][1]
                declared=request[11]
                returned=request[12] or 0
                if "pageNo" in page_query or "numOfRows" in page_query or declared is None or declared!=returned or returned!=(rowcount or 0):
                    mismatches+=1
                if state=="empty" and (rowcount or 0)!=0: mismatches+=1
                continue
            if not pages or [p for p,_ in pages]!=list(range(1,len(pages)+1)): page_gaps+=1; continue
            totals_page={r[11] for _,r in pages}
            page_sizes={json.loads(r[6]).get("numOfRows") for _,r in pages}
            if len(page_sizes)!=1 or None in page_sizes: mismatches+=1
            if len(totals_page)!=1 or None in totals_page: mismatches+=1
            else:
                expected_total=next(iter(totals_page)); received=sum(r[12] or 0 for _,r in pages)
                if received!=expected_total or received!=(rowcount or 0): mismatches+=1
            if state=="empty" and (rowcount or 0)!=0: mismatches+=1
    flags_query="SELECT n.dataset,n.quality_flags FROM normalized_records n JOIN source_requests r USING(request_id)"+clauses
    flags={}
    for ds,raw in db.execute(flags_query,params):
        for flag in json.loads(raw): flags[flag]=flags.get(flag,0)+1
    task_sql="SELECT state,COUNT(*) FROM planned_queries"+(" WHERE job_id=?" if job is not None else "")+" GROUP BY state"
    task_states={s:0 for s in ("complete","empty","incomplete","pending","failed")}
    task_states.update({s:n for s,n in db.execute(task_sql,params)})
    date_mismatches=0
    date_sql="SELECT r.dataset,r.query_json,n.payload_json FROM source_requests r JOIN raw_records n USING(request_id) WHERE r.state='complete'"
    if job is not None: date_sql+=" AND r.job_id=?"
    for dataset,qraw,rowraw in db.execute(date_sql,params):
        q=json.loads(qraw); row=json.loads(rowraw)
        expected=q.get("cond[trd_clcln_ymd::EQ]") if dataset.startswith("prices_") else None
        source=row.get("trd_clcln_ymd") if expected else None
        if expected and (not source or canonical_date(source)!=canonical_date(expected)): date_mismatches+=1
        if dataset=="imports":
            if str(row.get("year",""))=="총계" or str(row.get("hsCode",row.get("hsCd",""))) in ("","-"):
                continue
            try:
                source_month=str(row["year"]).replace(".","")
                if not (q["strtYymm"]<=source_month<=q["endYymm"]): date_mismatches+=1
            except (KeyError,TypeError): date_mismatches+=1
        if dataset=="weather":
            try:
                expected_start=dt.datetime.strptime(q["startDt"],"%Y%m%d").date()
                expected_end=dt.datetime.strptime(q["endDt"],"%Y%m%d").date()
                source=dt.date.fromisoformat(str(row["tm"])[:10])
                if not expected_start<=source<=expected_end: date_mismatches+=1
            except (KeyError,ValueError,TypeError): date_mismatches+=1
        if dataset=="production":
            expected_year=int(q.get("startPrdDe",0)); end_year=int(q.get("endPrdDe",9999))
            try: source_year=int(str(row.get("PRD_DE",""))[:4])
            except ValueError: source_year=None
            if source_year is None or not expected_year<=source_year<=end_year: date_mismatches+=1
    empty_sql="SELECT COUNT(*) FROM source_requests WHERE state='complete' AND returned_count=0"
    if job is not None: empty_sql+=" AND job_id=?"
    output={"job_ids":jobs,"job_states":{j:db.execute("SELECT state FROM jobs WHERE job_id=?",(j,)).fetchone()[0] for j in jobs},
            "failed_active_requests":failures,"count_mismatches":mismatches,"raw_record_count_errors":raw_errors,
            "normalized_record_count_errors":normalized_errors,"page_gaps":page_gaps,"manifest_errors":manifest_errors,
            "foreign_key_errors":len(check_db),"quality_flags":flags,
            "planned_queries":task_states,"source_date_mismatches":date_mismatches,
            "complete_empty_requests":db.execute(empty_sql,params).fetchone()[0]}
    print(json.dumps(output,ensure_ascii=False,indent=2))
    return 1 if failures or mismatches or raw_errors or normalized_errors or page_gaps or manifest_errors or check_db or date_mismatches or task_states["pending"] or task_states["incomplete"] or task_states["failed"] else 0


def export_parquet(db: sqlite3.Connection, output: str, datasets: list[str] | None, job_id: int) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise CollectionError("Parquet export requires optional pyarrow; install it separately when desired") from exc
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    status=db.execute("SELECT state FROM jobs WHERE job_id=?",(job_id,)).fetchone()
    if status is None: raise CollectionError(f"job does not exist: {job_id}")
    if status[0]!="complete": raise CollectionError("Parquet export requires a completed collection job")
    sql="""SELECT r.job_id,r.epoch_id,r.request_id,n.row_no,n.dataset,n.reference_date,r.retrieved_at,
      r.published_at,r.version_published_at,r.vintage_status,r.availability_basis,n.payload_json,n.quality_flags
      FROM normalized_records n JOIN source_requests r USING(request_id)
      JOIN planned_queries p ON p.job_id=r.job_id AND p.dataset=r.dataset AND p.source_url=r.source_url AND p.epoch_id=r.epoch_id
      WHERE r.state='complete' AND p.state IN ('complete','empty') AND r.job_id=?"""
    params=[job_id]
    if datasets:
        sql+=" AND n.dataset IN ("+",".join("?" for _ in datasets)+")"; params.extend(datasets)
    rows=[{"job_id":r[0],"epoch_id":r[1],"request_id":r[2],"row_no":r[3],"dataset":r[4],"reference_date":r[5],"retrieved_at":r[6],"published_at":r[7],"version_published_at":r[8],"vintage_status":r[9],"availability_basis":r[10],"payload_json":r[11],"quality_flags":r[12]} for r in db.execute(sql,params)]
    pq.write_table(pa.Table.from_pylist(rows),output,compression="zstd")
    print(f"exported {len(rows)} normalized records to {output}")
    return 0


def main(argv=None):
    parser=make_parser(); args=parser.parse_args(argv)
    try: return execute(args)
    except (CollectionError,OSError,sqlite3.Error) as exc:
        print("error: "+redact(str(exc)),file=sys.stderr); return 2


if __name__=="__main__": raise SystemExit(main())
