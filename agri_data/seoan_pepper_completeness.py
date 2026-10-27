"""Resumable internal completeness check for Seoan Hwageon August 2026 prices.

Dry-run is the default. Even a successful month/week equality check remains an
internal cross-check only and is always recorded with ``complete: false``.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

try:
    from . import seoan_pepper_collection as base
except ImportError:  # supports direct script invocation as well as python -m
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from agri_data import seoan_pepper_collection as base

MONTH = ("2026-08-01", "2026-08-31")
WEEKS = (("2026-08-01", "2026-08-07"), ("2026-08-08", "2026-08-14"),
         ("2026-08-15", "2026-08-21"), ("2026-08-22", "2026-08-28"),
         ("2026-08-29", "2026-08-31"))
FROZEN_NAMES = ("화건", "화건(꼭무)", "화건(손꼭무)")
MAX_ATTEMPTS = 3
MIN_REQUEST_INTERVAL_SECONDS = 1.0
RETRY_DELAY_SECONDS = 1.0
TIMEOUT_SECONDS = base.TIMEOUT_SECONDS
MAX_RESPONSE_BYTES = base.MAX_RESPONSE_BYTES
OUTPUT_DEFAULT = Path("data/raw/seoan_pepper_completeness_20261009")
CHECKPOINT_NAME = "checkpoint.json"
SUMMARY_NAME = "comparison.json"
WEIGHT_UNIT_TOKENS={"kg","㎏","킬로그램","근","관"}
PRICE_UNIT_TOKENS={"원/kg","원/㎏","원/근","원/관","원/1kg","원/500g"}


class CompletenessError(RuntimeError):
    pass


def _request_plan() -> list[dict[str, Any]]:
    requests_plan=[]
    for request_id,(start,end) in [("month",MONTH)]+[(f"week_{i:02d}",w) for i,w in enumerate(WEEKS,1)]:
        params={"searchStartDt":start.replace("-",""),"searchEndDt":end.replace("-",""),"searchSpcs":""}
        requests_plan.append({"request_id":request_id,"start":start,"end":end,"params":params})
    return requests_plan


def _plan_hash() -> str:
    data=json.dumps(_request_plan(),sort_keys=True,ensure_ascii=False,separators=(",",":"))
    return hashlib.sha256(data.encode()).hexdigest()


def _atomic_checkpoint(path: Path, payload: dict) -> None:
    temp=path.with_name(path.name+".tmp")
    encoded=(json.dumps(payload,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
    with temp.open("wb") as stream:
        stream.write(encoded);stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)


def _checkpoint(output: Path) -> dict:
    path=output/CHECKPOINT_NAME
    if path.exists():
        data=json.loads(path.read_text(encoding="utf-8"))
        if data.get("planSha256")!=_plan_hash():
            raise CompletenessError("checkpoint belongs to a different frozen request plan")
        return data
    data={"schemaVersion":1,"planSha256":_plan_hash(),"complete":False,
          "requestIds":["landing"]+[r["request_id"] for r in _request_plan()],
          "states":{key:"pending" for key in ["landing"]+[r["request_id"] for r in _request_plan()]},
          "inFlightAttempts":{}}
    _atomic_checkpoint(path,data)
    return data


def _request_spec(request_id: str) -> dict[str,str]:
    if request_id=="landing":
        return {"request_id":"landing","method":"GET","url":base.PAGE_URL}
    item=next((x for x in _request_plan() if x["request_id"]==request_id),None)
    if item is None: raise ValueError("unknown frozen request id")
    return {"request_id":request_id,"method":"POST","url":base.POST_URL,**item["params"]}


def _discover_attempts(output: Path, request_id: str) -> list[dict]:
    expected=_request_spec(request_id); found=[]
    for path in output.glob("*.manifest.json"):
        try: meta=json.loads(path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError): continue
        if meta.get("kind") not in {"seoan_recheck_landing","seoan_recheck_export"} or meta.get("request")!=expected:
            continue
        record={**meta,"manifest_file":path.name}
        if meta.get("raw_file"):
            raw_path=output/meta["raw_file"]
            try: body=raw_path.read_bytes()
            except OSError: body=b""
            if hashlib.sha256(body).hexdigest()!=meta.get("sha256"):
                record["integrity_error"]="saved_raw_sha256_mismatch"
            else: record["_body"]=body
        found.append(record)
    def seq(record:dict)->int:
        name=record["manifest_file"].split("_attempt",1)[-1]
        try:return int(name.split("_",1)[0])
        except (ValueError,IndexError):return 0
    return sorted(found,key=lambda x:(seq(x),x.get("retrieved_at_utc", ""),x["manifest_file"]))


def _orphaned_attempts(output:Path,request_id:str,attempts:list[dict])->list[str]:
    known={a.get("raw_file") for a in attempts if a.get("raw_file")}
    known_manifests={a.get("manifest_file") for a in attempts}
    orphans=[]
    for path in output.glob(f"seoan_recheck_{request_id}_attempt*.raw"):
        manifest=path.with_suffix(".manifest.json")
        if path.name not in known or not manifest.is_file():orphans.append(path.name)
    for path in output.glob(f"seoan_recheck_{request_id}_attempt*.raw.tmp"):
        orphans.append(path.name)
    for path in output.glob(f"seoan_recheck_{request_id}_attempt*.manifest.json.tmp"):
        orphans.append(path.name)
    for path in output.glob(f"seoan_recheck_{request_id}_attempt*.manifest.json"):
        if path.name not in known_manifests:
            # A malformed or mismatched final manifest may be the only trace of
            # an already-issued request; do not silently issue it again.
            orphans.append(path.name)
    return sorted(set(orphans))


def _extract_rows(body: bytes) -> list[list[str]]:
    if body.startswith(b"PK\x03\x04"):
        return base._xlsx_rows(body)
    text=body.decode("utf-8-sig",errors="replace")
    rows=base._table_rows(text)
    if rows:return rows
    try:
        dialect=csv.Sniffer().sniff(text[:8192],delimiters=",\t;")
        return list(csv.reader(io.StringIO(text),dialect))
    except csv.Error:
        return []


def _columns(header:list[str])->dict[str,int]:
    # Accept ordinary separator variants without broad substring matching.
    compact=[cell.replace("_","").replace("-","") for cell in header]
    return base._columns(compact)


def _header_only(body: bytes) -> tuple[list[str],dict[str,int]] | None:
    rows=_extract_rows(body)
    if len(rows)!=1:return None
    cols=_columns(rows[0])
    required={"date","product"}
    paired={"quantity_geun","quantity_kg","price_geun","price_kg"}
    if required.issubset(cols) and (paired.issubset(cols) or
                                     {"weight","price","weight_unit","price_unit"}.issubset(cols)):
        return rows[0],cols
    return None


def _validation(body: bytes,start:str,end:str,content_type:str="") -> dict:
    if body.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(body)) as archive:
                sheets=[n for n in archive.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")]
        except zipfile.BadZipFile:
            sheets=[]
        if len(sheets)>1:
            return {"state":"parse_incomplete","usable":False,"format":"xlsx",
                    "header":None,"schema":None,"rows":[],"qualityIssueCount":0,
                    "issues":["multiple_worksheets_not_supported"]}
    parsed=base.validate_body(body,start,end,content_type)
    if "no_data_rows_or_partial_output" in parsed.get("issues",[]):
        empty=_header_only(body)
        if empty is not None:
            header,cols=empty
            return {"state":"normal_empty","usable":True,"format":parsed.get("format"),
                    "header":header,"schema":sorted(cols),"rows":[],"qualityIssueCount":0,
                    "issues":[]}
    if not parsed.get("table_header_found"):
        return {"state":"parse_incomplete","usable":False,"format":parsed.get("format"),
                "header":None,"schema":None,"rows":[],"qualityIssueCount":0,
                "issues":parsed.get("issues",[])}
    header_rows=_extract_rows(body)
    header=header_rows[0] if header_rows else None
    columns=_columns(header or [])
    paired={"quantity_geun","quantity_kg","price_geun","price_kg"}
    structure_issues=[]
    for issue in parsed.get("issues",[]):
        if issue not in {"questionable_or_incomplete_rows_present"}:
            structure_issues.append(issue)
    quality=0
    for row in parsed.get("rows",[]):
        source_values=row.get("source_values",[])
        for field,column in columns.items():
            row[field]=source_values[column].strip() if column<len(source_values) else None
        for issue in row.get("issues",[]):
            if issue.startswith("date_outside") or issue=="invalid_date" or issue.startswith("missing_"):
                structure_issues.append(issue)
            elif issue.endswith(("_null_value","_negative_value","_nonfinite_value","_non_numeric_value")):
                quality+=1
        if not paired.issubset(columns):
            wu=base._norm(row.get("weight_unit") or "")
            pu=base._norm(row.get("price_unit") or "")
            if not wu or not pu:
                structure_issues.append("unit_value_unavailable")
            elif wu not in WEIGHT_UNIT_TOKENS or pu not in PRICE_UNIT_TOKENS:
                structure_issues.append("unit_value_unrecognized")
        if not row.get("product"):
            structure_issues.append("product_value_unavailable")
    usable=not structure_issues
    state="parsed" if usable and parsed.get("rows") else "normal_empty" if usable else "parse_incomplete"
    return {"state":state,"usable":usable,"format":parsed.get("format"),"header":header,
            "schema":sorted(columns),"rows":parsed.get("rows",[]),"qualityIssueCount":quality,
            "issues":sorted(set(structure_issues))}


def _attempt_result(output: Path, request_id: str, request: dict) -> tuple[list[dict],dict|None]:
    attempts=_discover_attempts(output,request_id)
    for attempt in attempts:
        if attempt.get("integrity_error"):
            attempt["usable"]=False;attempt["state"]="raw_integrity_failure";continue
        status=attempt.get("http_status")
        body=attempt.get("_body",b"")
        if attempt.get("truncated"):
            attempt["usable"]=False;attempt["state"]="response_truncated";continue
        if attempt.get("transport_error"):
            attempt["usable"]=False;attempt["state"]="transport_failure";continue
        if status!=200:
            attempt["usable"]=False;attempt["state"]="http_failure";continue
        if not body:
            attempt["usable"]=False;attempt["state"]="empty_body_response";continue
        if request_id=="landing":
            attempt["usable"]=True;attempt["state"]="captured";continue
        parsed=_validation(body,request["start"],request["end"],
                           attempt.get("response_headers",{}).get("Content-Type",""))
        attempt.update(parsed)
    usable=next((a for a in reversed(attempts) if a.get("usable")),None)
    return attempts,usable


def _attempt_outcomes(attempts:list[dict]) -> list[dict]:
    return [{"state":a.get("state"),"httpStatus":a.get("http_status"),
             "responseBytes":a.get("response_bytes"),"truncated":bool(a.get("truncated")),
             "transportError":a.get("transport_error"),"issues":a.get("issues",[]),
             "sha256":a.get("sha256"),"manifestFile":a.get("manifest_file")}
            for a in attempts]


def _is_nonretryable_4xx(attempt:dict)->bool:
    status=attempt.get("http_status")
    return isinstance(status,int) and 400<=status<500


def _request_once(output: Path, session: Any, *, request_id: str, request: dict,
                  attempt_number: int, sleep=time.sleep, now=lambda:dt.datetime.now(dt.timezone.utc)) -> dict:
    sleep(MIN_REQUEST_INTERVAL_SECONDS)
    response=None;body=b"";truncated=False;error=None
    try:
        if request_id=="landing":
            response=session.get(base.PAGE_URL,timeout=TIMEOUT_SECONDS,stream=True)
        else:
            response=session.post(urljoin(base.PAGE_URL,base.POST_URL),
                                  data=request["params"],timeout=TIMEOUT_SECONDS,stream=True)
        body,truncated,error=base._read_bounded(response)
    except requests.RequestException as exc:
        error=f"request_error:{type(exc).__name__}:{exc}"
    captured,token=base._capture_stamp(now)
    stem=f"seoan_recheck_{request_id}_attempt{attempt_number:02d}_{token}"
    request_meta=_request_spec(request_id)
    saved=base._save_attempt(output,stem,request=request_meta,captured=captured,
                             response=response,body=body,truncated=truncated,error=error,
                             kind="seoan_recheck_landing" if request_id=="landing" else "seoan_recheck_export")
    saved["manifest_file"]=stem+".manifest.json"
    return saved


def _process_request(output: Path,session:Any,request_id:str,request:dict,
                     checkpoint:dict,sleep,now) -> dict:
    attempts,usable=_attempt_result(output,request_id,request)
    orphaned=_orphaned_attempts(output,request_id,attempts)
    if orphaned:
        checkpoint["states"][request_id]="orphaned_outcome_unknown"
        _atomic_checkpoint(output/CHECKPOINT_NAME,checkpoint)
        return {"requestId":request_id,"state":"orphaned_outcome_unknown","attempts":len(attempts),
                "usable":False,"orphanedArtifacts":orphaned,"attemptOutcomes":_attempt_outcomes(attempts)}
    pending=checkpoint.setdefault("inFlightAttempts",{}).get(request_id)
    if pending is not None:
        archived=any(_attempt_number(a.get("manifest_file",""))==pending for a in attempts)
        if not archived:
            checkpoint["states"][request_id]="interrupted_outcome_unknown"
            _atomic_checkpoint(output/CHECKPOINT_NAME,checkpoint)
            return {"requestId":request_id,"state":"interrupted_outcome_unknown",
                    "attempts":len(attempts),"usable":False,"lastStatus":None,
                    "attemptOutcomes":_attempt_outcomes(attempts)}
        checkpoint["inFlightAttempts"].pop(request_id,None)
        _atomic_checkpoint(output/CHECKPOINT_NAME,checkpoint)
        attempts,usable=_attempt_result(output,request_id,request)
    if usable is None and attempts and _is_nonretryable_4xx(attempts[-1]):
        checkpoint["states"][request_id]="http_nonretryable_4xx"
        _atomic_checkpoint(output/CHECKPOINT_NAME,checkpoint)
        return {"requestId":request_id,"state":"http_nonretryable_4xx","attempts":len(attempts),
                "usable":False,"lastStatus":attempts[-1].get("http_status"),
                "attemptOutcomes":_attempt_outcomes(attempts)}
    if usable is None and len(attempts)<MAX_ATTEMPTS:
        for number in range(len(attempts)+1,MAX_ATTEMPTS+1):
            if number>1:sleep(RETRY_DELAY_SECONDS)
            checkpoint.setdefault("inFlightAttempts",{})[request_id]=number
            checkpoint["states"][request_id]="in_flight"
            _atomic_checkpoint(output/CHECKPOINT_NAME,checkpoint)
            _request_once(output,session,request_id=request_id,request=request,
                          attempt_number=number,sleep=sleep,now=now)
            checkpoint["inFlightAttempts"].pop(request_id,None)
            attempts,usable=_attempt_result(output,request_id,request)
            checkpoint["states"][request_id]="complete" if usable else "incomplete"
            _atomic_checkpoint(output/CHECKPOINT_NAME,checkpoint)
            if usable:break
            if attempts and _is_nonretryable_4xx(attempts[-1]):
                checkpoint["states"][request_id]="http_nonretryable_4xx"
                _atomic_checkpoint(output/CHECKPOINT_NAME,checkpoint)
                return {"requestId":request_id,"state":"http_nonretryable_4xx","attempts":len(attempts),
                        "usable":False,"lastStatus":attempts[-1].get("http_status"),
                        "attemptOutcomes":_attempt_outcomes(attempts)}
    state="complete" if usable else "attempts_exhausted"
    checkpoint["states"][request_id]=state
    _atomic_checkpoint(output/CHECKPOINT_NAME,checkpoint)
    if usable is None:
        return {"requestId":request_id,"state":state,"attempts":len(attempts),
                "usable":False,"lastStatus":attempts[-1].get("http_status") if attempts else None,
                "attemptOutcomes":_attempt_outcomes(attempts)}
    return {"requestId":request_id,"state":usable.get("state"),"attempts":len(attempts),"usable":True,
            "httpStatus":usable.get("http_status"),"responseBytes":usable.get("response_bytes"),
            "sha256":usable.get("sha256"),"rawFile":usable.get("raw_file"),
            "manifestFile":usable.get("manifest_file"),"rowCount":len(usable.get("rows",[])),
            "qualityIssueCount":usable.get("qualityIssueCount",0),"issues":usable.get("issues",[]),
            "header":usable.get("header"),"schema":usable.get("schema"),"rows":usable.get("rows",[]),
            "attemptOutcomes":_attempt_outcomes(attempts)}


def _attempt_number(filename:str)->int|None:
    try:return int(filename.split("_attempt",1)[1].split("_",1)[0])
    except (IndexError,ValueError):return None


def _counts(rows:list[dict]) -> dict:
    daily=collections.Counter();frozen=collections.Counter();products=collections.Counter()
    for row in rows:
        day=row.get("date");product=row.get("product")
        if day:
            parsed=None
            for fmt in ("%Y-%m-%d","%Y.%m.%d","%Y/%m/%d","%Y%m%d","%Y년 %m월 %d일"):
                try:parsed=dt.datetime.strptime(day.strip(),fmt).date().isoformat();break
                except ValueError:continue
            day=parsed
        if day and product is not None:
            daily[(day,product)]+=1;products[product]+=1
        if product in FROZEN_NAMES:frozen[product]+=1
    return {"dailyProductCounts":{f"{day}|{product}":count for (day,product),count in sorted(daily.items())},
            "productCounts":dict(sorted(products.items())),
            "frozenHwageonNameCounts":{name:frozen.get(name,0) for name in FROZEN_NAMES}}


def _multiset(rows:list[dict]) -> collections.Counter:
    return collections.Counter(tuple(row.get("source_values",[])) for row in rows)


def _comparison(results:dict[str,dict]) -> dict:
    plan_ids=[x["request_id"] for x in _request_plan()]
    all_usable=all(results.get(key,{}).get("usable") for key in plan_ids)
    landing_ok=bool(results.get("landing",{}).get("usable"))
    month=results.get("month",{});weekly=[results.get(f"week_{i:02d}",{}) for i in range(1,6)]
    if all_usable:
        month_rows=month["rows"]
        week_rows=[row for result in weekly for row in result["rows"]]
        month_counter=_multiset(month_rows);week_counter=_multiset(week_rows)
        header_match=all(r.get("header")==month.get("header") for r in weekly)
        schema_match=all(r.get("schema")==month.get("schema") for r in weekly)
        m_counts=_counts(month_rows);w_counts=_counts(week_rows)
        exact_match=month_counter==week_counter
        internal_consistency=bool(header_match and schema_match and exact_match)
        frozen_count=sum(m_counts["frozenHwageonNameCounts"].values())
        diff_count=sum((month_counter-week_counter).values())+sum((week_counter-month_counter).values())
    else:
        month_rows=[];week_rows=[];header_match=schema_match=exact_match=internal_consistency=None
        m_counts=w_counts={"dailyProductCounts":{},"productCounts":{},"frozenHwageonNameCounts":{n:0 for n in FROZEN_NAMES}}
        diff_count=None;frozen_count=0
    reasons=[]
    if not landing_ok:reasons.append("landing_request_incomplete")
    if not all_usable:reasons.append("one_or_more_export_requests_incomplete")
    if all_usable and not header_match:reasons.append("header_mismatch")
    if all_usable and not schema_match:reasons.append("schema_mismatch")
    if all_usable and not exact_match:reasons.append("source_values_multiset_mismatch")
    no_signal=all_usable and frozen_count==0
    if no_signal:reasons.append("no_usable_pepper_signal")
    stop=not (landing_ok and all_usable and internal_consistency and not no_signal)
    return {"complete":False,"requestSetUsable":bool(landing_ok and all_usable),
            "internalConsistency":internal_consistency,"headersMatch":header_match,"schemasMatch":schema_match,
            "exactFullSourceValuesMultisetMatch":exact_match,"multisetDifferenceRowCount":diff_count,
            "monthRowCount":len(month_rows),"fiveWeekUnionRowCount":len(week_rows),
            "monthCounts":m_counts,"weekCounts":w_counts,"usablePepperSignalRowCount":frozen_count,
            "noUsablePepperSignal":no_signal,"stopEnlargement":stop,"incompleteReasons":reasons,
            "limitations":["same-endpoint internal cross-check only","does not establish first publication time, post hoc revisions, product-target/unit correspondence, or service-independent completeness"]}


def run(output:Path,*,session:Any=None,sleep=time.sleep,
        now=lambda:dt.datetime.now(dt.timezone.utc)) -> dict:
    output.mkdir(parents=True,exist_ok=True)
    checkpoint=_checkpoint(output)
    session=session or requests.Session()
    session.verify=True
    session.headers.update({"User-Agent":"Mozilla/5.0"})
    request_results={}
    # Reuse a validated landing capture; export requests do not begin before it.
    landing_attempts,landing_valid=_attempt_result(output,"landing",{"request_id":"landing"})
    landing_result=_process_request(output,session,"landing",{"request_id":"landing"},checkpoint,sleep,now)
    request_results["landing"]=landing_result
    if not landing_result.get("usable"):
        result={"schemaVersion":1,"planSha256":_plan_hash(),"landing":landing_result,
                "exports":[],"complete":False,"requestSetUsable":False,
                "internalConsistency":None,"incompleteReasons":["landing_request_incomplete"]}
        checkpoint["complete"]=False;_atomic_checkpoint(output/CHECKPOINT_NAME,checkpoint)
        return result
    for item in _request_plan():
        if request_results and any(v.get("state")=="http_nonretryable_4xx" for v in request_results.values()):
            skipped={"requestId":item["request_id"],"state":"not_attempted_after_nonretryable_4xx",
                     "attempts":0,"usable":False,"attemptOutcomes":[]}
            checkpoint["states"][item["request_id"]]=skipped["state"]
            _atomic_checkpoint(output/CHECKPOINT_NAME,checkpoint)
            request_results[item["request_id"]]=skipped
            continue
        request_results[item["request_id"]]=_process_request(output,session,item["request_id"],item,checkpoint,sleep,now)
    combined={**request_results}
    comparison=_comparison(combined)
    summary={"schemaVersion":1,"planSha256":_plan_hash(),"sourcePage":base.PAGE_URL,
             "exportUrl":base.POST_URL,"month":{"start":MONTH[0],"end":MONTH[1]},
             "weeks":[{"start":a,"end":b} for a,b in WEEKS],"frozenHwageonNames":list(FROZEN_NAMES),
             "landing":request_results["landing"],
             "exports":[request_results[x["request_id"]] for x in _request_plan()],**comparison}
    # Parsed raw rows are needed only for the comparison and are not copied to summary.
    for item in summary["exports"]:
        item.pop("rows",None)
    checkpoint["complete"]=False
    checkpoint["internalConsistency"]=summary["internalConsistency"]
    _atomic_checkpoint(output/CHECKPOINT_NAME,checkpoint)
    summary_path=output/SUMMARY_NAME
    if summary_path.exists():
        prior=json.loads(summary_path.read_text(encoding="utf-8"))
        if prior!=summary:raise CompletenessError("immutable comparison summary already exists with different contents")
    else:
        base._atomic_immutable(summary_path,(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode())
    return summary


def dry_run() -> dict:
    return {"mode":"dry_run","requestSent":False,"landing":{"method":"GET","url":base.PAGE_URL},
            "exports":[{"requestId":item["request_id"],"method":"POST","url":base.POST_URL,
                        "start":item["start"],"end":item["end"],"params":item["params"]}
                       for item in _request_plan()],
            "maxAttemptsPerRequest":MAX_ATTEMPTS,"minimumRequestSpacingSeconds":MIN_REQUEST_INTERVAL_SECONDS,
            "timeoutSeconds":TIMEOUT_SECONDS,"maxResponseBytes":MAX_RESPONSE_BYTES,
            "complete":False,"namesToCount":list(FROZEN_NAMES)}


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT_DEFAULT)
    parser.add_argument("--execute",action="store_true",help="send the frozen landing/export requests")
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(dry_run(),ensure_ascii=False,sort_keys=True))
        return 0
    result=run(args.output)
    concise={k:result.get(k) for k in ("complete","requestSetUsable","internalConsistency","headersMatch",
                                           "schemasMatch","exactFullSourceValuesMultisetMatch","monthRowCount",
                                           "fiveWeekUnionRowCount","noUsablePepperSignal","stopEnlargement",
                                           "incompleteReasons")}
    concise["requestStates"]={"landing":result.get("landing",{}).get("state"),
        **{x["requestId"]:x["state"] for x in result.get("exports",[])}}
    print(json.dumps(concise,ensure_ascii=False,sort_keys=True))
    return 0


if __name__=="__main__":raise SystemExit(main())
