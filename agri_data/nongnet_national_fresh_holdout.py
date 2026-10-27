"""Raw-only Nongnet nationwide daily target collection for a fresh holdout.

Dry-run is the default. Collection preserves every response and never trains,
fills dates, or turns an empty response into a confirmed no-trade observation.
"""
from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import uuid
from typing import Any

import requests

from agri_data import nongnet_national_volume_pilot as volume
from agri_data import national_transfer as nt

ENDPOINT=volume.ENDPOINT
OVERLAP_START=dt.date(2026,6,20); OVERLAP_END=dt.date(2026,6,30)
HOLDOUT_START=dt.date(2026,7,1); HOLDOUT_END=dt.date(2026,9,30)
DEFAULT_START=OVERLAP_START; DEFAULT_END=HOLDOUT_END
REQUEST_START=DEFAULT_START; REQUEST_END=DEFAULT_END
ARCHIVE_START=dt.date(2024,7,8); ARCHIVE_END=dt.date(2026,6,30)
PERIODS=(("overlap",OVERLAP_START,OVERLAP_END),("holdout",HOLDOUT_START,HOLDOUT_END))
CATEGORIES=(
    ("cabbage","1001",""),("radish","1101",""),("onion","1201",""),
    ("pepper","1207",""),("garlic_peeled","1209","120906"),
)
ANCHOR_SPACING_DAYS=7
TARGETED_MISSING_PASSES=2
MAX_ATTEMPTS=volume.MAX_ATTEMPTS
TIMEOUT_SECONDS=volume.TIMEOUT_SECONDS
MAX_RESPONSE_BYTES=volume.MAX_RESPONSE_BYTES
RETRY_DELAY_SECONDS=volume.RETRY_DELAY_SECONDS
MIN_REQUEST_INTERVAL_SECONDS=volume.MIN_REQUEST_INTERVAL_SECONDS
DEFAULT_OUTPUT=Path("data/raw/nongnet_national_fresh_holdout_20260701-20260930")
ARCHIVE_PATH=Path("data/raw/national-wholesale-20240708-20260630.zip")
EXPECTED_FIELDS=("saledate","selectDay","selectDayWgt")
PROTOCOL={"id":"nongnet-national-fresh-holdout-2026-07-01-2026-09-30-v1",
 "overlap":[OVERLAP_START.isoformat(),OVERLAP_END.isoformat()],"holdout":[HOLDOUT_START.isoformat(),HOLDOUT_END.isoformat()],
 "categories":[{"label":n,"itemCode":c,"varietyCode":v} for n,c,v in CATEGORIES],
 "overlapSourceMatchCrops":["cabbage","radish","onion","pepper"],
 "overlapSourceMatchGate":{"dates":[OVERLAP_START.isoformat(),OVERLAP_END.isoformat()],"minimumComparableDaysPerCrop":5,
    "maximumMedianAbsoluteRelativeDifferencePct":1,"maximumAbsoluteRelativeDifferencePct":5},
 "selection":{"searchUnitGrade":"normal","searchSymbol1":"100000","garlicPeeledItem":"1209","garlicPeeledVariety":"120906"},
 "collection":"weekly spaced anchors consume recent response rows; two bounded targeted passes for dates still missing; immutable raw per attempt; no forward fill",
 "emptySemantics":"normal_empty is a valid empty API response; no_trade_unconfirmed means two successful nonempty targeted responses omitted the date and is not an official no-trade assertion",
 "limitations":"retrieval-time revised public screen; historical publication vintages are unavailable; no model training or scoring"}

def configure_window(start=None,end=None):
    """Set an explicit, bounded date window; archive-overlap dates use four crops."""
    global REQUEST_START,REQUEST_END,PERIODS,PROTOCOL
    if (start is None)!=(end is None): raise ValueError('--start and --end must be supplied together')
    lo=DEFAULT_START if start is None else dt.date.fromisoformat(start)
    hi=DEFAULT_END if end is None else dt.date.fromisoformat(end)
    if lo>hi: raise ValueError('start date must not follow end date')
    if (hi-lo).days+1>120: raise ValueError('requested window exceeds 120 calendar days')
    REQUEST_START,REQUEST_END=lo,hi
    overlap_lo=max(lo,ARCHIVE_START); overlap_hi=min(hi,ARCHIVE_END)
    segments=[]
    if overlap_lo<=overlap_hi:
        if lo<overlap_lo: segments.append(('holdout',lo,overlap_lo-dt.timedelta(days=1)))
        segments.append(('overlap',overlap_lo,overlap_hi))
        if overlap_hi<hi: segments.append(('holdout',overlap_hi+dt.timedelta(days=1),hi))
    else:
        segments=[('holdout',lo,hi)]
    PERIODS=tuple(segments)
    PROTOCOL={**PROTOCOL,'id':f"nongnet-national-fresh-window-{lo.isoformat()}-{hi.isoformat()}-v1",
        'requestedDateWindow':[lo.isoformat(),hi.isoformat()],
        'periods':[{'label':p,'start':a.isoformat(),'end':b.isoformat()} for p,a,b in PERIODS]}
    return PERIODS

def _sha(raw:bytes)->str: return hashlib.sha256(raw).hexdigest()
def _daterange(start:dt.date,end:dt.date):
    day=start
    while day<=end:
        yield day
        day+=dt.timedelta(days=1)

def anchor_dates(start=OVERLAP_START,end=HOLDOUT_END):
    anchors=[]
    for _,lo,hi in PERIODS:
        day=lo
        while day<=hi:
            anchors.append(day)
            day+=dt.timedelta(days=ANCHOR_SPACING_DAYS)
        if anchors[-1]!=hi: anchors.append(hi)
    return sorted(set(anchors))

def _period_categories(period):
    return CATEGORIES if period=="holdout" else CATEGORIES[:4]

def _period_anchor_dates(start,end):
    dates=[]; day=start
    while day<=end:
        dates.append(day); day+=dt.timedelta(days=ANCHOR_SPACING_DAYS)
    if dates[-1]!=end: dates.append(end)
    return dates

def _params(day:dt.date,item_code:str,variety_code:str):
    return {"searchUnitGrade":"normal","searchDate":f"{day.year}년 {day.month:02d}월 {day.day:02d}일",
            "searchSymbol1":"100000","searchSymbol2":item_code,"searchSymbol3":variety_code}

def request_matrix():
    requests_out=[]
    for period,start,end in PERIODS:
        for day in _period_anchor_dates(start,end):
            for name,code,variety in _period_categories(period):
                requests_out.append({"kind":"anchor","period":period,"requestedDate":day.isoformat(),"category":name,
                    "requestParams":_params(day,code,variety)})
    return requests_out

def _valid_day(raw):
    if not isinstance(raw,str) or len(raw)!=8 or not raw.isdigit(): return None
    try:
        day=dt.datetime.strptime(raw,"%Y%m%d").date()
        return day if day.strftime("%Y%m%d")==raw else None
    except ValueError: return None

def validate_response(body:bytes,requested_date:str,http_status:int=200,*,truncated=False,transport_error=None):
    result={"state":None,"requestedDate":requested_date,"requestedDateState":None,"issues":[],"rowCount":0,"observedDates":[],"matchingRows":[],"rows":[]}
    if truncated:
        result["state"]="truncated"; result["issues"].append(transport_error or "response_truncated"); return result
    if transport_error:
        result["state"]="transport_error"; result["issues"].append(transport_error); return result
    if http_status<200 or http_status>=300:
        result["state"]="http_error"; result["issues"].append(f"http_status_{http_status}"); return result
    if not body:
        result["state"]="empty_response"; return result
    try: doc=json.loads(body.decode("utf-8-sig"))
    except (UnicodeDecodeError,json.JSONDecodeError):
        result["state"]="non_json_response"; result["issues"].append("response_not_valid_json"); return result
    if isinstance(doc,dict) and any(k in doc and doc[k] not in (None,"",False,[],{}) for k in ("error","errors")):
        result["state"]="error_payload"; result["issues"].append("json_error_payload"); return result
    if not isinstance(doc,dict) or not isinstance(doc.get("datalist"),list):
        result["state"]="invalid_json_structure"; result["issues"].append("expected_object_with_datalist_array"); return result
    entries=doc["datalist"]; result["rowCount"]=len(entries)
    if not entries:
        result["state"]="normal_empty"; return result
    expected=dt.date.fromisoformat(requested_date); observed=set()
    for i,source in enumerate(entries):
        row={"sourceIndex":i,"sourceValues":source,"issues":[]}
        if not isinstance(source,dict):
            row["issues"].append("row_not_object"); result["rows"].append(row); result["issues"].append(f"row_{i}_questionable"); continue
        for field in EXPECTED_FIELDS:
            if field not in source: row["issues"].append(f"missing_{field}")
        parsed=_valid_day(source.get("saledate"))
        if parsed is None: row["issues"].append("saledate_invalid_or_missing")
        else:
            row["parsedDate"]=parsed.isoformat(); observed.add(parsed.isoformat())
            if parsed==expected: result["matchingRows"].append(row)
        for field in ("selectDay","selectDayWgt"):
            if field not in source: continue
            issue=volume._numeric_issue(source[field])
            if issue: row["issues"].append(f"{field}_{issue}")
            elif source[field] not in (None,"") and float(source[field])==0:
                row["issues"].append(f"{field}_zero_value")
        if row["issues"]: result["issues"].append(f"row_{i}_questionable")
        result["rows"].append(row)
    result["observedDates"]=sorted(observed)
    result["state"]="valid" if not result["issues"] else "questionable"
    if len(result["matchingRows"])>1:
        result["issues"].append("duplicate_requested_date"); result["requestedDateState"]="duplicate_requested_date"
    elif not result["matchingRows"]: result["requestedDateState"]="requested_date_missing"
    else: result["requestedDateState"]="observed" if not result["matchingRows"][0]['issues'] else "questionable_observed"
    return result

def _checkpoint_write(path:Path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:
            json.dump(obj,f,ensure_ascii=False,sort_keys=True,indent=2); f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)

def _request_id(request):
    payload=json.dumps(request,ensure_ascii=False,sort_keys=True,separators=(",",":" )).encode()
    return _sha(payload)[:20]

def _read_existing(output,checkpoint,request_id):
    entry=checkpoint.get("requests",{}).get(request_id)
    if not entry or not entry.get("attempts"): return None
    last=entry["attempts"][-1]
    if not last.get("rawFile"): return None
    raw=(output/last["rawFile"]).read_bytes()
    if _sha(raw)!=last.get("sha256"): raise ValueError(f"checkpoint raw hash mismatch for {request_id}")
    valid=validate_response(raw,last["request"]["requestedDate"],last.get("httpStatus") or 0,
        truncated=last.get("truncated",False),transport_error=last.get("transportError"))
    if valid["state"] in {"transport_error","http_error","truncated","non_json_response","error_payload","invalid_json_structure"}: return None
    return entry

def _execute_request(request,output,checkpoint,*,session,now,sleep,last_start):
    rid=_request_id(request); existing=checkpoint.get('requests',{}).get(rid); old=_read_existing(output,checkpoint,rid)
    if old is not None: return old,last_start
    attempts=list(existing.get('attempts',[])) if existing else []
    if len(attempts)>=MAX_ATTEMPTS:
        return existing,last_start
    for attempt in range(len(attempts)+1,MAX_ATTEMPTS+1):
        if last_start[0] is not None:
            elapsed=time.monotonic()-last_start[0]
            if elapsed<MIN_REQUEST_INTERVAL_SECONDS: sleep(MIN_REQUEST_INTERVAL_SECONDS-elapsed)
        last_start[0]=time.monotonic(); response=None; body=b""; truncated=False; error=None
        try:
            response=session.get(ENDPOINT,params=request["requestParams"],timeout=TIMEOUT_SECONDS,verify=True,stream=True)
            body,truncated,error=volume._read_bounded(response)
        except requests.RequestException as exc: error=f"request_error:{type(exc).__name__}:{exc}"
        captured=now().astimezone(dt.timezone.utc).isoformat()
        status=response.status_code if response is not None else 0
        headers=dict(response.headers) if response is not None else {}
        headers={k:("[redacted]" if k.lower() in {"set-cookie","authorization"} else v) for k,v in headers.items()}
        stem=f"nongnet_fresh_{request['category']}_{request['requestedDate']}_{request['kind']}_{rid}_a{attempt}_{uuid.uuid4().hex[:8]}"
        raw_name=stem+".json"; volume._immutable(output/raw_name,body)
        validation=validate_response(body,request['requestedDate'],status,truncated=truncated,transport_error=error)
        manifest={"schemaVersion":1,"requestId":rid,"request":request,"endpoint":ENDPOINT,"retrievedAtUtc":captured,
            "attempt":attempt,"httpStatus":status or None,"responseHeaders":headers,"rawFile":raw_name,"responseBytes":len(body),
            "sha256":_sha(body),"truncated":truncated,"transportError":error,"validation":validation,"complete":False}
        manifest_name=stem+".manifest.json"
        volume._immutable(output/manifest_name,(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode())
        attempts.append({"manifestFile":manifest_name,**manifest})
        checkpoint.setdefault("requests",{})[rid]={"request":request,"attempts":attempts,"state":validation['state'],
            "requestedDateState":validation.get('requestedDateState')}
        _checkpoint_write(output/"checkpoint.json",checkpoint)
        if 200<=status<300 and not truncated and not error and validation['state'] in {'valid','questionable','normal_empty'}: break
        if attempt<MAX_ATTEMPTS: sleep(RETRY_DELAY_SECONDS)
    entry={"request":request,"attempts":attempts,"state":attempts[-1]['validation']['state'],
           "requestedDateState":attempts[-1]['validation'].get('requestedDateState')}
    checkpoint.setdefault("requests",{})[rid]=entry
    _checkpoint_write(output/"checkpoint.json",checkpoint)
    return entry,last_start

def _all_rows(output,checkpoint):
    found=[]
    for rid,entry in checkpoint.get("requests",{}).items():
        for attempt in entry.get("attempts",[]):
            raw=(output/attempt['rawFile']).read_bytes()
            if _sha(raw)!=attempt['sha256']: raise ValueError(f"raw response hash mismatch: {attempt['rawFile']}")
            parsed=validate_response(raw,attempt['request']['requestedDate'],attempt.get('httpStatus') or 0,
                truncated=attempt.get('truncated',False),transport_error=attempt.get('transportError'))
            if parsed['state'] not in {'valid','questionable'}: continue
            request=attempt['request']
            for row in parsed['rows']:
                day=row.get('parsedDate')
                if day is None: continue
                d=dt.date.fromisoformat(day)
                if REQUEST_START<=d<=REQUEST_END:
                    found.append({"date":day,"category":request['category'],"requestId":rid,"requestKind":request['kind'],
                        "requestDate":request['requestedDate'],"attempt":attempt['attempt'],"responseSha256":attempt['sha256'],
                        "sourceIndex":row['sourceIndex'],"sourceValues":row['sourceValues'],"issues":row['issues']})
    return found

def _coverage(found,checkpoint):
    by={(r['category'],r['date']):[] for r in []}
    grouped={}
    for row in found: grouped.setdefault((row['category'],row['date']),[]).append(row)
    coverage=[]; unresolved=[]
    for period,start,end in PERIODS:
      for name,_,_ in _period_categories(period):
       for day in _daterange(start,end):
        records=grouped.get((name,day.isoformat()),[])
        signatures={json.dumps(r['sourceValues'],ensure_ascii=False,sort_keys=True) for r in records}
        targeted=[entry for entry in checkpoint.get('requests',{}).values()
                  if entry['request']['category']==name and entry['request']['requestedDate']==day.isoformat()
                  and entry['request']['kind'].startswith('targeted')]
        final_state=(targeted[-1].get('requestedDateState') or targeted[-1]['state']) if targeted else None
        if records and len(signatures)>1: state='conflicting_responses'
        elif records: state='observed' if not any(r['issues'] for r in records) else 'questionable_observed'
        elif final_state=='normal_empty': state='normal_empty'
        elif final_state=='empty_response': state='empty_response'
        elif final_state in {'http_error','transport_error','truncated','non_json_response','error_payload','invalid_json_structure'}: state='api_error'
        elif len(targeted)>=TARGETED_MISSING_PASSES and all((x.get('requestedDateState') or x['state'])=='requested_date_missing' for x in targeted[-TARGETED_MISSING_PASSES:]): state='no_trade_unconfirmed'
        elif targeted: state='requested_date_missing'
        else: state='not_queried'
        row={"period":period,"category":name,"date":day.isoformat(),"state":state,"responseCopyCount":len(records),
             "exactRepeatedResponseRows":max(0,len(records)-len(signatures)),"sourceValues":records[0]['sourceValues'] if records and len(signatures)==1 else None,
             "provenance":[{"requestId":r['requestId'],"requestKind":r['requestKind'],"requestDate":r['requestDate'],"responseSha256":r['responseSha256']} for r in records]}
        coverage.append(row)
        if state not in {'observed','questionable_observed','normal_empty','no_trade_unconfirmed'}: unresolved.append(row)
    return coverage,unresolved

def _overlap_audit(coverage):
    rows=[r for r in coverage if r['period']=='overlap']
    bycat={}
    for name,_,_ in _period_categories('overlap'):
        group=[r for r in rows if r['category']==name]
        bycat[name]={"plannedDates":len(group),"observed":sum(r['state']=='observed' for r in group),
            "questionableObserved":sum(r['state']=='questionable_observed' for r in group),
            "normalEmpty":sum(r['state']=='normal_empty' for r in group),
            "noTradeUnconfirmed":sum(r['state']=='no_trade_unconfirmed' for r in group),
            "unresolved":sum(r['state'] not in {'observed','questionable_observed','normal_empty','no_trade_unconfirmed'} for r in group),
            "dailyRows":[r for r in group]}
    dates=[dt.date.fromisoformat(r['date']) for r in rows]
    date_range=[min(dates).isoformat(),max(dates).isoformat()] if dates else None
    return {"dateRange":date_range,"byCategory":bycat}

def _verified_row_sources(row,checkpoint,raw_dir):
    """Independently validate a coverage row against hash-checked immutable attempts."""
    if not isinstance(checkpoint,dict) or raw_dir is None: return []
    raw_dir=Path(raw_dir); category=row.get('category'); day=row.get('date')
    spec=next((x for x in CATEGORIES if x[0]==category),None)
    if spec is None: return []
    verified=[]
    for ref in row.get('provenance',[]):
        rid=ref.get('requestId'); entry=checkpoint.get('requests',{}).get(rid)
        if not entry: continue
        request=entry.get('request',{}); params=request.get('requestParams',{})
        if (request.get('category')!=category or params.get('searchUnitGrade')!='normal' or
            params.get('searchSymbol1')!='100000' or params.get('searchSymbol2')!=spec[1] or
            params.get('searchSymbol3')!=spec[2]): continue
        for attempt in entry.get('attempts',[]):
            if attempt.get('sha256')!=ref.get('responseSha256') or attempt.get('request')!=request: continue
            manifest_name=attempt.get('manifestFile')
            if not isinstance(manifest_name,str) or Path(manifest_name).name!=manifest_name: continue
            try: manifest=json.loads((raw_dir/manifest_name).read_text(encoding='utf-8'))
            except (OSError,ValueError,json.JSONDecodeError): continue
            expected_manifest={'endpoint':ENDPOINT,'requestId':rid,'request':request,
                'rawFile':attempt.get('rawFile'),'sha256':attempt.get('sha256'),
                'httpStatus':attempt.get('httpStatus'),'truncated':bool(attempt.get('truncated',False)),
                'transportError':attempt.get('transportError')}
            if any(manifest.get(key)!=value for key,value in expected_manifest.items()): continue
            if not isinstance(attempt.get('rawFile'),str) or Path(attempt['rawFile']).name!=attempt['rawFile']: continue
            raw_path=raw_dir/attempt['rawFile']
            try: raw=raw_path.read_bytes()
            except (OSError,ValueError): continue
            if _sha(raw)!=ref.get('responseSha256'): continue
            parsed=validate_response(raw,request.get('requestedDate',''),attempt.get('httpStatus') or 0,
                truncated=attempt.get('truncated',False),transport_error=attempt.get('transportError'))
            matches=[x for x in parsed['rows'] if x.get('parsedDate')==day and not x['issues']
                and x['sourceValues']==row.get('sourceValues')]
            if matches:
                verified.append({'requestId':rid,'responseSha256':ref['responseSha256'],
                    'rawFile':attempt['rawFile'],'sourceIndex':matches[0]['sourceIndex']})
    return verified

def _source_match_audit(coverage,archive_path=ARCHIVE_PATH,*,checkpoint=None,raw_dir=None):
    path=Path(archive_path)
    if not path.exists():
        return {"status":"archive_unavailable","archivePath":str(path),"comparedCount":0,"pairs":[]}
    data=path.read_bytes(); digest=_sha(data)
    if digest!=nt.ARCHIVE_SHA256:
        return {"status":"archive_hash_mismatch","archivePath":str(path),"archiveSha256":digest,"comparedCount":0,"pairs":[]}
    archive=nt.parse_archive(data)
    pairs=[]
    june_rows=[r for r in coverage if r['period']=='overlap' and OVERLAP_START<=dt.date.fromisoformat(r['date'])<=OVERLAP_END]
    by={(r['category'],r['date']):r for r in june_rows}
    overlap_rows=june_rows
    days=sorted({dt.date.fromisoformat(r['date']) for r in overlap_rows})
    by_crop={}
    for category,crop in zip(_period_categories('overlap'),('cabbage','radish','onion','pepper')):
        name=category[0]
        crop_pairs=[]
        for day in days:
            public=by.get((name,day.isoformat()))
            if public is None: continue
            raw=public.get('sourceValues'); archive_row=archive['observations'][crop].get(day.isoformat())
            item={"date":day.isoformat(),"crop":crop,"publicState":public['state'],
                "publicPriceRaw":raw.get('selectDay') if isinstance(raw,dict) else None,
                "archivePrice":archive_row.get('price') if archive_row else None,
                "rawEvidenceVerified":bool(_verified_row_sources(public,checkpoint,raw_dir))}
            try:
                public_price=Decimal(str(item['publicPriceRaw']).replace(',',''))
                archive_price=Decimal(item['archivePrice']) if item['archivePrice'] is not None else None
                valid=(item['rawEvidenceVerified'] and public['state']=='observed' and public_price.is_finite() and public_price>0 and
                    archive_price is not None and archive_price.is_finite() and archive_price>0)
            except (InvalidOperation,TypeError,ValueError): valid=False; public_price=archive_price=None
            if valid:
                difference=abs(public_price-archive_price)
                item.update({"status":"compared","absoluteDifferenceWon":str(difference),
                    "relativeDifferencePct":str(difference/archive_price*Decimal(100)),
                    "exact":difference==0,"withinOneWon":difference<=1})
                crop_pairs.append(item)
            else: item['status']='not_comparable'
            pairs.append(item)
        diffs=sorted(Decimal(x['relativeDifferencePct']) for x in crop_pairs)
        median=(diffs[len(diffs)//2] if len(diffs)%2 else (diffs[len(diffs)//2-1]+diffs[len(diffs)//2])/2) if diffs else None
        by_crop[crop]={"comparableDays":len(diffs),"medianAbsoluteRelativeDifferencePct":str(median) if median is not None else None,
            "maximumAbsoluteRelativeDifferencePct":str(max(diffs)) if diffs else None,
            "minimumComparableDaysMet":len(diffs)>=5,
            "medianThresholdMet":median is not None and median<=Decimal('1'),
            "maximumThresholdMet":bool(diffs) and max(diffs)<=Decimal('5'),
            "gatePass":len(diffs)>=5 and median is not None and median<=Decimal('1') and max(diffs)<=Decimal('5')}
    compared=[x for x in pairs if x['status']=='compared']
    exact=sum(x['exact'] for x in compared)
    planned=len(pairs)
    gate_applicable=planned>0
    gate_pass=gate_applicable and len(by_crop)==4 and all(x['gatePass'] for x in by_crop.values())
    status='not_applicable' if not gate_applicable else 'source_match_gate_pass' if gate_pass else 'source_match_gate_fail'
    return {"status":status,"archivePath":str(path),"archiveSha256":digest,"comparison":"public integer selectDay vs signed amount/quantity archive price; no harmonization",
        "gateProtocol":{"minimumComparableDaysPerCrop":5,"maximumMedianAbsoluteRelativeDifferencePct":1,
            "maximumAbsoluteRelativeDifferencePct":5},"plannedPairs":planned,"comparedCount":len(compared),"exactCount":exact,
        "notComparableCount":planned-len(compared),"byCrop":by_crop,"pairs":pairs}

def build_combined_target_source(coverage_report_path,archive_path=ARCHIVE_PATH):
    """Build an offline four-crop target source with independently checked provenance.

    The returned target observations are separate from model feature manifests.
    Only clean, observed four-crop rows after the pinned archive are appended.
    """
    coverage_path=Path(coverage_report_path); report_bytes=coverage_path.read_bytes()
    report=json.loads(report_bytes); checkpoint_path=coverage_path.parent/'checkpoint.json'
    checkpoint=json.loads(checkpoint_path.read_text(encoding='utf-8'))
    archive_bytes=Path(archive_path).read_bytes(); archive_sha=_sha(archive_bytes)
    if archive_sha!=nt.ARCHIVE_SHA256: raise ValueError('national archive SHA-256 differs from frozen protocol')
    observations=nt.parse_archive(archive_bytes)['observations']
    provenance={'archive':{'path':str(archive_path),'sha256':archive_sha,'dateRange':[nt.EVAL_START.isoformat(),nt.EVAL_END.isoformat()]},
        'coverageReport':{'path':str(coverage_path),'sha256':_sha(report_bytes)},'appended':[]}
    crop_names=[x[0] for x in CATEGORIES[:4]]
    coverage=report.get('coverage')
    if not isinstance(coverage,list): raise ValueError('coverage report has no coverage rows')
    source_match=_source_match_audit(coverage,archive_path,checkpoint=checkpoint,raw_dir=coverage_path.parent)
    if source_match.get('status')!='source_match_gate_pass':
        raise ValueError(f"June source-match gate must pass before appending targets: {source_match.get('status')}")
    rows=[r for r in coverage if r.get('category') in crop_names and r.get('period')=='holdout'
          and dt.date.fromisoformat(r['date'])>nt.EVAL_END and r.get('state')=='observed']
    for row in rows:
        category=row['category']; day=row['date']; values=row.get('sourceValues')
        try:
            price=Decimal(str(values['selectDay']).replace(',',''))
            weight=Decimal(str(values['selectDayWgt']).replace(',',''))
            if not price.is_finite() or price<=0 or not weight.is_finite() or weight<=0: raise ValueError
        except (KeyError,InvalidOperation,TypeError,ValueError):
            raise ValueError(f'nonpositive or invalid source values for {category} {day}')
        verified=_verified_row_sources(row,checkpoint,coverage_path.parent)
        if not verified: raise ValueError(f'no hash-verified clean raw response supports {category} {day}')
        crop={'cabbage':'cabbage','radish':'radish','onion':'onion','pepper':'pepper'}[category]
        prior=observations[crop].get(day)
        if prior is not None: raise ValueError(f'overlapping archive and fresh target date: {crop} {day}')
        observations[crop][day]={'price':str(price)}
        provenance['appended'].append({'crop':crop,'date':day,'state':row['state'],'selectDayRaw':values['selectDay'],
            'selectDayWgtRaw':values['selectDayWgt'],'sources':verified})
    canonical=json.dumps(observations,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
    return {'schemaVersion':1,'observations':observations,'provenance':provenance,
        'sourceDigests':{'archiveSha256':archive_sha,'coverageReportSha256':_sha(report_bytes),
            'combinedObservationsSha256':_sha(canonical)}}

def collect(output=DEFAULT_OUTPUT,*,session=None,now=lambda:dt.datetime.now(dt.timezone.utc),sleep=time.sleep):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); checkpoint_path=output/"checkpoint.json"
    if checkpoint_path.exists(): checkpoint=json.loads(checkpoint_path.read_text())
    else:
        checkpoint={"schemaVersion":1,"protocol":PROTOCOL,"requests":{},"collectionStartedAtUtc":now().astimezone(dt.timezone.utc).isoformat()}
        _checkpoint_write(checkpoint_path,checkpoint)
    if checkpoint.get('protocol') not in (None,PROTOCOL): raise ValueError('checkpoint protocol mismatch')
    checkpoint['protocol']=PROTOCOL; session=session or requests.Session(); session.verify=True; session.headers.update({"User-Agent":"Mozilla/5.0"})
    last_start=[None]
    for request in request_matrix():
        _,last_start=_execute_request(request,output,checkpoint,session=session,now=now,sleep=sleep,last_start=last_start)
    found=_all_rows(output,checkpoint); observed={(r['category'],r['date']) for r in found}
    for pass_no in range(1,TARGETED_MISSING_PASSES+1):
        missing=[]
        for period,start,end in PERIODS:
            days=list(_daterange(start,end))
            missing.extend((name,day,period) for name,_,_ in _period_categories(period) for day in days if (name,day.isoformat()) not in observed)
        if not missing: break
        for name,day,period in missing:
            _,code,variety=next(x for x in CATEGORIES if x[0]==name)
            request={"kind":f"targeted_missing_pass{pass_no}","period":period,"requestedDate":day.isoformat(),"category":name,
                     "requestParams":_params(day,code,variety)}
            _execute_request(request,output,checkpoint,session=session,now=now,sleep=sleep,last_start=last_start)
        found=_all_rows(output,checkpoint); observed={(r['category'],r['date']) for r in found}
    coverage,unresolved=_coverage(found,checkpoint)
    overlap=_overlap_audit(coverage); overlap['sourceMatch']=_source_match_audit(coverage,ARCHIVE_PATH,checkpoint=checkpoint,raw_dir=output)
    four_crop_states=[x for x in coverage if x['category'] in {c[0] for c in CATEGORIES[:4]}]
    garlic_states=[x for x in coverage if x['category']=='garlic_peeled']
    classified={'observed','questionable_observed','normal_empty','no_trade_unconfirmed'}
    audit={"schemaVersion":1,"protocol":PROTOCOL,"source":"Nongnet public nationwide normal-grade endpoint",
        "createdAtUtc":checkpoint['collectionStartedAtUtc'],"rawAttemptCount":sum(len(x['attempts']) for x in checkpoint['requests'].values()),
        "requestCount":len(checkpoint['requests']),"sourceRowCount":len(found),
        "sourceAudit":{"nullPriceRows":sum(volume._numeric_issue(r['sourceValues'].get('selectDay'))=='null_value' for r in found),
            "nullWeightRows":sum(volume._numeric_issue(r['sourceValues'].get('selectDayWgt'))=='null_value' for r in found),
            "negativePriceRows":sum((volume._numeric_issue(r['sourceValues'].get('selectDay'))=='negative_value') for r in found),
            "negativeWeightRows":sum((volume._numeric_issue(r['sourceValues'].get('selectDayWgt'))=='negative_value') for r in found),
            "nonfinitePriceRows":sum(volume._numeric_issue(r['sourceValues'].get('selectDay'))=='nonfinite_value' for r in found),
            "nonfiniteWeightRows":sum(volume._numeric_issue(r['sourceValues'].get('selectDayWgt'))=='nonfinite_value' for r in found),
            "zeroPriceRows":sum(r['sourceValues'].get('selectDay')==0 or r['sourceValues'].get('selectDay')=='0' for r in found),
            "zeroWeightRows":sum(r['sourceValues'].get('selectDayWgt')==0 or r['sourceValues'].get('selectDayWgt')=='0' for r in found),
            "duplicateDateRows":sum(r['exactRepeatedResponseRows'] for r in coverage),
            "conflictingDateCount":sum(r['state']=='conflicting_responses' for r in coverage)},
        "coverageCompleteAsClassified":not unresolved,
        "fourCropCoverageCompleteAsClassified":all(r['state'] in classified for r in four_crop_states),
        "peeledGarlicCoverageCompleteAsClassified":all(r['state'] in classified for r in garlic_states),
        "unresolvedDateCount":len(unresolved),"coverage":coverage,
        "sourceMatchGateStatus":overlap['sourceMatch']['status'],
        "holdoutScoringGateOpen":overlap['sourceMatch']['status']=='source_match_gate_pass',
        "overlapAudit":overlap,"unresolved":unresolved}
    payload=(json.dumps(audit,ensure_ascii=False,sort_keys=True,indent=2)+"\n").encode()
    report=output/"coverage-report.json"
    if report.exists():
        if report.read_bytes()!=payload:
            report=output/f"coverage-report-{_sha(payload)[:16]}.json"
            if report.exists() and report.read_bytes()!=payload: raise ValueError('coverage report hash collision')
            if not report.exists(): volume._immutable(report,payload)
    else: volume._immutable(report,payload)
    audit['coverageReportPath']=str(report)
    return audit

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--execute',action='store_true'); parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--start',help='inclusive date YYYY-MM-DD (must be paired with --end)')
    parser.add_argument('--end',help='inclusive date YYYY-MM-DD (must be paired with --start)')
    args=parser.parse_args(argv)
    try: configure_window(args.start,args.end)
    except ValueError as exc: parser.error(str(exc))
    if not args.execute:
        matrix=request_matrix(); print(json.dumps({"mode":"dry_run","requestSent":False,"endpoint":ENDPOINT,
            "anchorRequestCount":len(matrix),"weeklySpacingDays":ANCHOR_SPACING_DAYS,"targetedMissingPasses":TARGETED_MISSING_PASSES,
            "requestedDateWindow":[REQUEST_START.isoformat(),REQUEST_END.isoformat()],
            "dateCounts":{p:sum((end-start).days+1 for label,start,end in PERIODS if label==p) for p in ('overlap','holdout')},
            "categories":[x[0] for x in CATEGORIES],"anchorRequests":matrix},ensure_ascii=False)); return 0
    result=collect(args.output); print(json.dumps({"mode":"execute","requestSent":True,"requestCount":result['requestCount'],
        "coverageCompleteAsClassified":result['coverageCompleteAsClassified'],"unresolvedDateCount":result['unresolvedDateCount']},ensure_ascii=False)); return 0

if __name__=='__main__': raise SystemExit(main())
