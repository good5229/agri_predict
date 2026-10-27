"""Bounded, resumable Seoan monthly history capture (dry-run by default).

Frozen plan: capture 2024-07 through 2026-06 inclusive as 24 calendar-month
exports with a blank cultivar filter, after one landing GET. Preserve every
attempt's raw response and manifest. Never claim complete source history.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urljoin

import requests

from . import seoan_pepper_collection as base
from . import seoan_pepper_completeness as validation

START=dt.date(2024,7,1)
END=dt.date(2026,6,30)
FROZEN_NAMES=("화건","화건(꼭무)","화건(손꼭무)")
MAX_ATTEMPTS=3
MIN_INTERVAL=1.0
OUTPUT_DEFAULT=Path("data/raw/seoan_pepper_history_202407_202606")
CHECKPOINT="checkpoint.json"
SUMMARY="summary.json"


class HistoryError(RuntimeError): pass


def month_plan()->list[dict[str,Any]]:
    out=[];year,month=START.year,START.month
    while (year,month)<=(END.year,END.month):
        start=dt.date(year,month,1)
        stop=(dt.date(year+1,1,1)-dt.timedelta(days=1) if month==12 else dt.date(year,month+1,1)-dt.timedelta(days=1))
        rid=f"{year:04d}-{month:02d}"
        out.append({"requestId":rid,"start":start.isoformat(),"end":stop.isoformat(),
                    "params":{"searchStartDt":start.strftime("%Y%m%d"),"searchEndDt":stop.strftime("%Y%m%d"),"searchSpcs":""}})
        if month==12:year,month=year+1,1
        else:month+=1
    return out


def _plan_hash()->str:
    return hashlib.sha256(json.dumps(month_plan(),sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def _atomic_json(path:Path,obj:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+".tmp")
    data=(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
    with tmp.open("wb") as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def _checkpoint(out:Path)->dict:
    path=out/CHECKPOINT
    if path.exists():
        data=json.loads(path.read_text(encoding="utf-8"))
        if data.get("planSha256")!=_plan_hash():raise HistoryError("checkpoint does not match frozen monthly plan")
        return data
    ids=["landing"]+[m["requestId"] for m in month_plan()]
    data={"schemaVersion":1,"planSha256":_plan_hash(),"complete":False,
          "requestIds":ids,"states":{x:"pending" for x in ids},"inFlight":None}
    _atomic_json(path,data);return data


def _request(request_id:str)->dict:
    if request_id=="landing":return {"method":"GET","url":base.PAGE_URL}
    month=next((m for m in month_plan() if m["requestId"]==request_id),None)
    if month is None:raise ValueError("unknown month request")
    return {"method":"POST","url":urljoin(base.PAGE_URL,base.POST_URL),"params":month["params"]}


def _stem(request_id:str,n:int,token:str)->str:return f"seoan_history_{request_id}_attempt{n:02d}_{token}"


def _attempts(out:Path,request_id:str)->tuple[list[dict],list[str]]:
    req=_request(request_id);found=[];known_raw=set();known_manifest=set();orphans=[]
    for path in out.glob(f"seoan_history_{request_id}_attempt*.manifest.json"):
        try:meta=json.loads(path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError):orphans.append(path.name);continue
        if meta.get("kind") not in {"seoan_history_landing","seoan_history_export"} or meta.get("request")!=req:
            orphans.append(path.name);continue
        meta["manifestFile"]=path.name;meta["body"]=b"";known_manifest.add(path.name)
        raw=meta.get("raw_file")
        byte_count=meta.get("response_bytes")
        digest=meta.get("sha256")
        if (not isinstance(raw,str) or not raw or not isinstance(byte_count,int) or byte_count<0 or
                not isinstance(digest,str) or len(digest)!=64 or any(c not in "0123456789abcdef" for c in digest.lower())):
            meta["integrityError"]="capture_metadata_missing_or_invalid"
            paired_raw=path.name.removesuffix(".manifest.json")+".raw"
            if (out/paired_raw).is_file():known_raw.add(paired_raw)
            found.append(meta)
            continue
        if Path(raw).name!=raw:
            meta["integrityError"]="raw_filename_invalid"
            found.append(meta)
            continue
        known_raw.add(raw)
        raw_path=out/raw
        try:body=raw_path.read_bytes()
        except OSError:meta["integrityError"]="raw_file_missing";body=b""
        if len(body)!=byte_count or hashlib.sha256(body).hexdigest()!=digest:
            meta["integrityError"]="raw_integrity_failure"
        else:meta["body"]=body
        found.append(meta)
    for p in out.glob(f"seoan_history_{request_id}_attempt*.raw"):
        if p.name not in known_raw:orphans.append(p.name)
    for pattern in (f"seoan_history_{request_id}_attempt*.raw.tmp",f"seoan_history_{request_id}_attempt*.manifest.json.tmp"):
        orphans.extend(p.name for p in out.glob(pattern))
    found.sort(key=lambda x:(x.get("request",{}).get("params",{}).get("searchStartDt", ""), x["manifestFile"]))
    return found,sorted(set(orphans))


def _validated_attempts(out:Path,request_id:str)->tuple[list[dict],list[str]]:
    found,orphans=_attempts(out,request_id)
    if request_id=="landing":
        for a in found:
            a["state"]="captured" if a.get("http_status")==200 and a.get("body") and not a.get("truncated") and not a.get("transport_error") and not a.get("integrityError") else "incomplete"
            a["usable"]=a["state"]=="captured"
        return found,orphans
    month=next(m for m in month_plan() if m["requestId"]==request_id)
    for a in found:
        if a.get("integrityError"):
            a.update(state="raw_integrity_failure",usable=False,issues=[a["integrityError"]]);continue
        if a.get("truncated") or a.get("transport_error") or a.get("http_status")!=200 or not a.get("body"):
            a.update(state="http_or_transport_failure",usable=False,issues=["response_not_usable"]);continue
        parsed=validation._validation(a["body"],month["start"],month["end"],a.get("response_headers",{}).get("Content-Type",""))
        a.update(state=parsed["state"],usable=parsed["usable"],header=parsed["header"],schema=parsed["schema"],
                 rows=parsed["rows"],qualityIssueCount=parsed["qualityIssueCount"],issues=parsed["issues"])
    return found,orphans


def _capture(out:Path,session:Any,request_id:str,number:int,*,sleep,now)->dict:
    req=_request(request_id);response=None;body=b"";truncated=False;error=None
    try:
        if request_id=="landing":response=session.get(req["url"],timeout=base.TIMEOUT_SECONDS,stream=True)
        else:response=session.post(req["url"],data=req["params"],timeout=base.TIMEOUT_SECONDS,stream=True)
        body,truncated,error=base._read_bounded(response)
    except requests.RequestException as exc:
        error=f"request_error:{type(exc).__name__}:{exc}"
    captured,token=base._capture_stamp(now);stem=_stem(request_id,number,token)
    raw_name=stem+".raw"
    # Archive even a zero-byte HTTP response so its exact bytes have provenance.
    base._atomic_immutable(out/raw_name,body)
    digest=hashlib.sha256(body).hexdigest()
    meta={"schema_version":1,"kind":"seoan_history_landing" if request_id=="landing" else "seoan_history_export",
          "request":req,"retrieved_at_utc":captured,"http_status":response.status_code if response is not None else None,
          "response_headers":base._safe_headers(response.headers if response is not None else {}),
          "raw_file":raw_name,"response_bytes":len(body),"sha256":digest,"truncated":truncated,"transport_error":error}
    encoded=(json.dumps(meta,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
    base._atomic_immutable(out/(stem+".manifest.json"),encoded)
    return meta


def _name_counts(rows:list[dict])->dict:
    by_name={n:{"rowCount":0,"distinctDateCount":0,"dates":{}} for n in FROZEN_NAMES}
    for row in rows:
        name=row.get("product");day=row.get("date")
        if name not in by_name:continue
        if day:
            try:day=dt.datetime.strptime(day.strip(),"%Y-%m-%d").date().isoformat()
            except ValueError:pass
        target=by_name[name];target["rowCount"]+=1
        if day:target["dates"][day]=target["dates"].get(day,0)+1
    for item in by_name.values():item["distinctDateCount"]=len(item["dates"])
    return by_name


def _month_summary(out:Path,request_id:str)->dict:
    plan=next(m for m in month_plan() if m["requestId"]==request_id)
    attempts,orphans=_validated_attempts(out,request_id)
    if orphans:return {"requestId":request_id,"start":plan["start"],"end":plan["end"],"state":"orphaned_outcome_unknown","usable":False,"attempts":len(attempts),"orphanedArtifacts":orphans}
    if not attempts:return {"requestId":request_id,"start":plan["start"],"end":plan["end"],"state":"pending","usable":False,"attempts":0}
    if any(a.get("integrityError") for a in attempts):return {"requestId":request_id,"start":plan["start"],"end":plan["end"],"state":"raw_integrity_failure","usable":False,"attempts":len(attempts)}
    usable=next((a for a in reversed(attempts) if a.get("usable")),None)
    latest=attempts[-1]
    if usable is None:
        state=("http_nonretryable_4xx" if isinstance(latest.get("http_status"),int) and 400<=latest["http_status"]<500
               else "response_outcome_unknown" if latest.get("transport_error") or latest.get("truncated")
               else "attempts_exhausted")
        return {"requestId":request_id,"start":plan["start"],"end":plan["end"],"state":state,"usable":False,"attempts":len(attempts),
                "httpStatus":latest.get("http_status"),"format":latest.get("format"),"header":latest.get("header"),
                "schema":latest.get("schema"),"issues":latest.get("issues",[]),
                "attemptOutcomes":[{"httpStatus":a.get("http_status"),"state":a.get("state"),"sha256":a.get("sha256"),
                                    "manifestFile":a.get("manifestFile")} for a in attempts]}
    return {"requestId":request_id,"state":usable["state"],"usable":True,"attempts":len(attempts),
            "start":plan["start"],"end":plan["end"],
            "httpStatus":usable.get("http_status"),"responseBytes":usable.get("response_bytes"),
            "rawFile":usable.get("raw_file"),"manifestFile":usable.get("manifestFile"),"sha256":usable.get("sha256"),
            "format":usable.get("format"),"header":usable.get("header"),"schema":usable.get("schema"),
            "rowCount":len(usable.get("rows",[])),"qualityIssueCount":usable.get("qualityIssueCount",0),
            "issues":usable.get("issues",[]),"frozenNameCounts":_name_counts(usable.get("rows",[])),
            "attemptOutcomes":[{"httpStatus":a.get("http_status"),"state":a.get("state"),"sha256":a.get("sha256"),
                                "manifestFile":a.get("manifestFile")} for a in attempts]}


def _summary(out:Path,landing:dict,stop_reason:str|None,checkpoint:dict)->dict:
    months=[_month_summary(out,m["requestId"]) for m in month_plan()]
    for item in months:
        if item["state"]=="pending" and checkpoint["states"].get(item["requestId"]) not in {None,"pending"}:
            item["state"]=checkpoint["states"][item["requestId"]]
    finished=all(m["usable"] for m in months)
    structurally_usable=all(m["usable"] and m["state"] in {"parsed","normal_empty"} for m in months)
    return {"schemaVersion":1,"planSha256":_plan_hash(),"period":{"start":START.isoformat(),"end":END.isoformat()},
            "landing":landing,"months":months,"complete":False,"allMonthsUsable":finished,
            "allMonthsStructurallyReadable":structurally_usable,"stopReason":stop_reason,
            "limitations":["monthly captures do not establish within-month completeness","historical first-publication versions are unavailable","complete is always false"]}


def run(out:Path,*,session:Any=None,sleep=time.sleep,now=lambda:dt.datetime.now(dt.timezone.utc))->dict:
    out.mkdir(parents=True,exist_ok=True);checkpoint=_checkpoint(out)
    session=session or requests.Session();session.verify=True;session.headers.update({"User-Agent":"Mozilla/5.0"})
    ids=["landing"]+[m["requestId"] for m in month_plan()]
    terminal_states={"interrupted_outcome_unknown","orphaned_outcome_unknown","raw_integrity_failure",
                     "http_nonretryable_4xx","response_outcome_unknown","attempts_exhausted",
                     "month_structure_or_response_incomplete","landing_incomplete","landing_attempts_exhausted"}
    stop_reason=next((checkpoint["states"].get(x) for x in ids if checkpoint["states"].get(x) in terminal_states),None)
    if checkpoint.get("inFlight"):
        rid=checkpoint["inFlight"]
        attempts,orphans=_attempts(out,rid)
        if orphans or not any(_attempt_number(a.get("manifestFile",""))==checkpoint.get("inFlightAttempt") for a in attempts):
            stop_reason="interrupted_outcome_unknown"
            checkpoint["states"][rid]=stop_reason;checkpoint["inFlight"]=None;checkpoint.pop("inFlightAttempt",None);_atomic_json(out/CHECKPOINT,checkpoint)
        else:
            # An immutable manifest exists for the pending attempt. Reconcile it
            # below, then keep the checkpoint's in-flight marker cleared.
            checkpoint["inFlight"]=None;checkpoint.pop("inFlightAttempt",None);_atomic_json(out/CHECKPOINT,checkpoint)
    last_request=None
    def wait_interval():
        nonlocal last_request
        if last_request is not None:sleep(max(0,MIN_INTERVAL-(time.monotonic()-last_request)))
        last_request=time.monotonic()
    # Landing request must be captured before monthly exports; valid archived landing is reused.
    if not stop_reason:
        landing_attempts,orphans=_validated_attempts(out,"landing")
        landing=next((a for a in reversed(landing_attempts) if a.get("usable")),None)
        if orphans:stop_reason="orphaned_outcome_unknown"
        if not stop_reason and any(a.get("integrityError") for a in landing_attempts):
            stop_reason="raw_integrity_failure"
        if not stop_reason and landing_attempts:
            latest=landing_attempts[-1]
            status=latest.get("http_status")
            if isinstance(status,int) and 400<=status<500:
                stop_reason="http_nonretryable_4xx"
            elif latest.get("transport_error") or latest.get("truncated"):
                stop_reason="response_outcome_unknown"
        if landing is None and not stop_reason:
            prior=len(landing_attempts)
            if prior>=MAX_ATTEMPTS:stop_reason="landing_attempts_exhausted"
            else:
                for number in range(prior+1,MAX_ATTEMPTS+1):
                    checkpoint["inFlight"]="landing";checkpoint["inFlightAttempt"]=number;_atomic_json(out/CHECKPOINT,checkpoint)
                    wait_interval();_capture(out,session,"landing",number,sleep=sleep,now=now)
                    checkpoint["inFlight"]=None;checkpoint.pop("inFlightAttempt",None);_atomic_json(out/CHECKPOINT,checkpoint)
                    landing_attempts,orphans=_validated_attempts(out,"landing")
                    if orphans:stop_reason="orphaned_outcome_unknown";break
                    landing=next((a for a in reversed(landing_attempts) if a.get("usable")),None)
                    if landing:break
                    if landing_attempts and isinstance(landing_attempts[-1].get("http_status"),int) and 400<=landing_attempts[-1]["http_status"]<500:
                        stop_reason="http_nonretryable_4xx";break
                    if landing_attempts and (landing_attempts[-1].get("transport_error") or landing_attempts[-1].get("truncated")):
                        stop_reason="response_outcome_unknown";break
                    sleep(base.RETRY_DELAY_SECONDS)
        landing_summary={"state":"captured" if landing else (stop_reason or "landing_incomplete"),
                        "usable":bool(landing),"httpStatus":landing.get("http_status") if landing else None,
                        "manifestFile":landing.get("manifestFile") if landing else None}
        checkpoint["states"]["landing"]="complete" if landing else (stop_reason or "incomplete")
    else:
        landing_attempts,_=_validated_attempts(out,"landing")
        landing=next((a for a in reversed(landing_attempts) if a.get("usable")),None)
        landing_summary={"state":"captured" if landing else stop_reason,"usable":bool(landing),
                         "httpStatus":landing.get("http_status") if landing else None,
                         "manifestFile":landing.get("manifestFile") if landing else None}
    if not landing_summary["usable"] and not stop_reason:stop_reason="landing_incomplete"
    if landing_summary["usable"] and not stop_reason:
        for month in month_plan():
            rid=month["requestId"];res=_month_summary(out,rid)
            if res["usable"]:checkpoint["states"][rid]="complete";continue
            if res["state"] not in {"pending","attempts_exhausted"}:
                stop_reason=res["state"];break
            attempts,_=_validated_attempts(out,rid)
            if len(attempts)>=MAX_ATTEMPTS:stop_reason="attempts_exhausted";break
            for number in range(len(attempts)+1,MAX_ATTEMPTS+1):
                checkpoint["inFlight"]=rid;checkpoint["inFlightAttempt"]=number;checkpoint["states"][rid]="in_flight";_atomic_json(out/CHECKPOINT,checkpoint)
                wait_interval();_capture(out,session,rid,number,sleep=sleep,now=now)
                checkpoint["inFlight"]=None;checkpoint.pop("inFlightAttempt",None);_atomic_json(out/CHECKPOINT,checkpoint)
                attempts,orphans=_validated_attempts(out,rid)
                if orphans:stop_reason="orphaned_outcome_unknown";break
                if attempts and attempts[-1].get("integrityError"):stop_reason="raw_integrity_failure";break
                if attempts and isinstance(attempts[-1].get("http_status"),int) and 400<=attempts[-1]["http_status"]<500:
                    stop_reason="http_nonretryable_4xx";break
                if attempts and (attempts[-1].get("transport_error") or attempts[-1].get("truncated")):
                    stop_reason="response_outcome_unknown";break
                if attempts and attempts[-1].get("usable"):
                    checkpoint["states"][rid]="complete";break
                if number<MAX_ATTEMPTS:sleep(base.RETRY_DELAY_SECONDS)
            if stop_reason:checkpoint["states"][rid]=stop_reason;break
            if not _month_summary(out,rid)["usable"]:
                stop_reason="month_structure_or_response_incomplete";checkpoint["states"][rid]=stop_reason;break
    if stop_reason:
        after=ids.index(next((i for i in ids if checkpoint["states"].get(i)==stop_reason),"landing")) if stop_reason in checkpoint["states"].values() else -1
        for rid in ids:
            if checkpoint["states"].get(rid)=="pending" and (rid=="landing" or not landing_summary["usable"] or stop_reason):
                checkpoint["states"][rid]="not_attempted_after_stop"
    checkpoint["complete"]=False;_atomic_json(out/CHECKPOINT,checkpoint)
    result=_summary(out,landing_summary,stop_reason,checkpoint)
    _atomic_json(out/SUMMARY,result)
    return result


def _attempt_number(filename:str)->int|None:
    try:return int(filename.split("_attempt",1)[1].split("_",1)[0])
    except (IndexError,ValueError):return None


def dry_run()->dict:
    return {"mode":"dry_run","requestSent":False,"landing":{"method":"GET","url":base.PAGE_URL},
            "months":month_plan(),"monthlyRequestCount":len(month_plan()),"maxAttemptsPerMonth":MAX_ATTEMPTS,
            "minimumIntervalSeconds":MIN_INTERVAL,"maxResponseBytes":base.MAX_RESPONSE_BYTES,
            "timeoutSeconds":base.TIMEOUT_SECONDS,"complete":False}


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",type=Path,default=OUTPUT_DEFAULT)
    parser.add_argument("--execute",action="store_true",help="perform the frozen landing and monthly requests")
    args=parser.parse_args(argv)
    result=run(args.output) if args.execute else dry_run()
    print(json.dumps(result if not args.execute else {"complete":False,"stopReason":result.get("stopReason"),
        "landing":result["landing"],"months":[{"requestId":m["requestId"],"state":m["state"],"rowCount":m.get("rowCount"),
        "frozenNameCounts":m.get("frozenNameCounts")} for m in result["months"]]},ensure_ascii=False,sort_keys=True))
    return 0


if __name__=="__main__":raise SystemExit(main())
