"""Frozen offline product/unit plausibility check (protocol, fixed before reading prices).

Protocol: read only the immutable Seoan 2024-08-01..07 XLSX pilot whose raw
SHA-256 is 2a52a3df84adedc0384e4025d031cf50b8e1c7e7c9f3f91251ed917403681bcb
and the existing national wholesale ZIP pinned by national_transfer. Validate
the Seoan request manifest's dates, empty all-species filter, raw-file name,
byte count, and digest; require one worksheet and preserve the parser's raw
decimal strings, duplicate rows, nulls, negatives, and questionable rows. For
each exact source product name 화건, 화건(꼭무), 화건(손꼭무), compare same-date
price_kg distributions against national item code 1207, using signed national
auction amount and changed volume summed over all markets and amount/volume
only when every matching row is numeric and the signed daily volume is
positive. Report all seven dates, raw headers/units, valid/missing/invalid
counts, row-level raw numeric strings, same-date overlap, and daily price-ratio
min/median/max. Do not trim outliers, deduplicate, convert units, or silently
filter questionable rows; retain them with issues and use them only to block a
ratio where relevant. Separately test whether 2026-08 is covered by the ZIP's
frozen 2024-07-08..2026-06-30 range. This is product/unit plausibility only,
not commodity identity, first-publication evidence, or forecasting benefit.
No live API, model fitting, or training is part of this check.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import zipfile
from typing import Any

from . import national_transfer
from . import seoan_pepper_collection as seoan

ROOT=Path(__file__).resolve().parents[1]
OUTPUT_DIR=ROOT/"outputs/experiments/seoan-pepper-target-mapping-20261009"
OUTPUT_PATH=OUTPUT_DIR/"report-v2.json"
SEOAN_START="2024-08-01"
SEOAN_END="2024-08-07"
SEOAN_RAW=ROOT/"data/raw/seoan_pepper_pilot/seoan_2024-08-01_2024-08-07_attempt01_2026-10-08T112918.572501+0000_b599fff2.raw"
SEOAN_MANIFEST=ROOT/"data/raw/seoan_pepper_pilot/seoan_2024-08-01_2024-08-07_attempt01_2026-10-08T112918.572501+0000_b599fff2.manifest.json"
SEOAN_SHA256="2a52a3df84adedc0384e4025d031cf50b8e1c7e7c9f3f91251ed917403681bcb"
NATIONAL_ZIP=national_transfer.INPUT_ARCHIVE
NATIONAL_SHA256=national_transfer.ARCHIVE_SHA256
NATIONAL_START=dt.date(2024,7,8)
NATIONAL_END=dt.date(2026,6,30)
PEPPER_CODE="1207"
PEPPER_NAME="건고추"
SEOAN_NAMES=("화건","화건(꼭무)","화건(손꼭무)")
EXPECTED_SEOAN_PARAMS={"searchStartDt":"20240801","searchEndDt":"20240807","searchSpcs":""}
NATIONAL_HEADERS=national_transfer.EXPECTED_HEADERS


def _sha(data:bytes)->str:
    return hashlib.sha256(data).hexdigest()


def _decimal(raw:Any)->Decimal|None:
    if raw is None:return None
    text=str(raw).strip()
    if not text:return None
    try:value=Decimal(text)
    except InvalidOperation:return None
    return value if value.is_finite() else None


def _decimal_text(value:Decimal|None)->str|None:
    return None if value is None else format(value,"f")


def _seoan_input(root:Path=ROOT)->tuple[bytes,dict]:
    raw=(root/SEOAN_RAW.relative_to(ROOT)).read_bytes()
    manifest=json.loads((root/SEOAN_MANIFEST.relative_to(ROOT)).read_text(encoding="utf-8"))
    if _sha(raw)!=SEOAN_SHA256 or manifest.get("sha256")!=SEOAN_SHA256:
        raise ValueError("pinned Seoan raw digest mismatch")
    if manifest.get("raw_file")!=SEOAN_RAW.name or manifest.get("response_bytes")!=len(raw):
        raise ValueError("Seoan raw-file metadata mismatch")
    if manifest.get("request")!={"searchStartDt":"20240801","searchEndDt":"20240807","searchSpcs":""}:
        raise ValueError("Seoan request parameters differ from frozen week")
    if manifest.get("http_status")!=200 or manifest.get("truncated") is not False or manifest.get("transport_error"):
        raise ValueError("Seoan capture HTTP/transport state is not usable")
    if manifest.get("complete") is True:
        raise ValueError("Seoan source unexpectedly claims completeness")
    return raw,manifest


def _parse_seoan(raw:bytes,start:str=SEOAN_START,end:str=SEOAN_END)->dict:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            sheets=[name for name in archive.namelist() if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")]
    except zipfile.BadZipFile as exc:
        raise ValueError("Seoan pilot is not a valid XLSX ZIP") from exc
    if len(sheets)!=1:raise ValueError("Seoan XLSX must have exactly one worksheet")
    parsed=seoan.validate_body(raw,start,end,"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    if not parsed.get("table_header_found") or not parsed.get("rows"):
        raise ValueError("Seoan XLSX header/data rows are not parseable")
    rows=seoan._xlsx_rows(raw)
    parsed["header"]=rows[0]
    parsed["columns"]=seoan._columns(rows[0])
    return parsed


def _parse_national(raw:bytes,expected_sha:str=NATIONAL_SHA256)->dict:
    # Use the project's pinned archive reader to verify ZIP members, CP949 CSV
    # schema, dates, identities, and archive-wide numeric validity.
    validated=national_transfer.parse_archive(raw,expected_sha256=expected_sha)
    source=[]
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for info in archive.infolist():
            text=archive.read(info).decode("cp949",errors="strict")
            reader=csv.DictReader(io.StringIO(text,newline=""))
            if tuple(reader.fieldnames or ())!=NATIONAL_HEADERS:
                raise ValueError("national archive header differs from frozen schema")
            for line_no,row in enumerate(reader,start=2):
                if row["품목코드"].strip()!=PEPPER_CODE:continue
                day=national_transfer._parse_day(row["경매일자"])
                source.append({"date":day.isoformat(),"sourceMember":info.filename,"sourceLine":line_no,
                               "productCode":row["품목코드"],"productName":row["품목명"],
                               "quantityRaw":row["변경총물량"],"amountRaw":row["경매낙찰총금액"],
                               "marketCode":row["도매시장코드"],"marketName":row["도매시장명"]})
    return {"validation":validated,"rows":source}


def _stats(values:list[Decimal])->dict:
    if not values:return {"n":0,"min":None,"median":None,"max":None,"mean":None}
    ordered=sorted(values);n=len(ordered)
    med=ordered[n//2] if n%2 else (ordered[n//2-1]+ordered[n//2])/Decimal(2)
    return {"n":n,"min":_decimal_text(ordered[0]),"median":_decimal_text(med),
            "max":_decimal_text(ordered[-1]),"mean":_decimal_text(sum(ordered,Decimal(0))/Decimal(n))}


def analyze_rows(seoan_rows:list[dict],national_rows:list[dict],*,
                 national_coverage:tuple[dt.date,dt.date]=(NATIONAL_START,NATIONAL_END),
                 header:list[str]|None=None,schema:dict|None=None)->dict:
    """Summarize same-day distributions while retaining each selected raw row."""
    days=[(dt.date.fromisoformat(SEOAN_START)+dt.timedelta(days=i)).isoformat() for i in range(7)]
    national_by_day:dict[str,list[dict]]=collections.defaultdict(list)
    for row in national_rows:
        if row.get("productCode")==PEPPER_CODE and row.get("date") in days:
            national_by_day[row["date"]].append(row)
    selected=[]
    for row in seoan_rows:
        name=row.get("product")
        if name not in SEOAN_NAMES:continue
        raw_price=row.get("price_kg")
        parsed=_decimal(raw_price)
        issues=list(row.get("issues",[]))
        if raw_price is None or str(raw_price).strip()=="":issues.append("missing_price_kg")
        elif parsed is None:issues.append("invalid_price_kg")
        elif parsed <= 0:issues.append("nonpositive_price_kg")
        selected.append({"date":row.get("date"),"product":name,"sourceRow":row.get("source_row"),
                         "weightUnitRaw":row.get("weight_unit"),"priceUnitRaw":row.get("price_unit"),
                         "quantityKgRaw":row.get("quantity_kg",row.get("weight")),
                         "quantityGeunRaw":row.get("quantity_geun"),"priceKgRaw":raw_price,
                         "issues":sorted(set(issues))})
    national_daily={}
    for day in days:
        rows=national_by_day.get(day,[])
        quantities=[_decimal(r.get("quantityRaw")) for r in rows]
        amounts=[_decimal(r.get("amountRaw")) for r in rows]
        valid=bool(rows) and all(v is not None for v in quantities+amounts)
        names_match=all(r.get("productName")==PEPPER_NAME for r in rows)
        q=sum(quantities,Decimal(0)) if valid else None
        a=sum(amounts,Decimal(0)) if valid else None
        price=a/q if valid and names_match and q and q>0 else None
        national_daily[day]={"rowCount":len(rows),"missingOrInvalidNumericCount":sum(v is None for v in quantities+amounts),
                             "productNameMismatchCount":sum(r.get("productName")!=PEPPER_NAME for r in rows),
                             "signedQuantityTotalRaw":_decimal_text(q),"signedAuctionAmountTotalRaw":_decimal_text(a),
                             "signedPricePerKg":_decimal_text(price),"validPrice":price is not None,
                             "sourceNumericStrings":[{"quantity":r.get("quantityRaw"),"amount":r.get("amountRaw"),
                                                      "sourceMember":r.get("sourceMember"),"sourceLine":r.get("sourceLine")}
                                                     for r in rows]}
    per_day={};ratio_values={name:[] for name in SEOAN_NAMES}
    selected_by_day:dict[str,list[dict]]=collections.defaultdict(list)
    for row in selected:selected_by_day[row.get("date")].append(row)
    for day in days:
        per_day[day]={}
        nprice=_decimal(national_daily[day]["signedPricePerKg"])
        for name in SEOAN_NAMES:
            rows=[r for r in selected_by_day.get(day,[]) if r["product"]==name]
            prices=[_decimal(r["priceKgRaw"]) for r in rows]
            valid=[p for p in prices if p is not None and p > 0]
            questionable=any(r["issues"] for r in rows)
            ratios=[p/nprice for p in valid] if nprice is not None and nprice>0 and not questionable else []
            ratio_values[name].extend(ratios)
            per_day[day][name]={"rowCount":len(rows),"validPriceCount":len(valid),
                                "missingPriceCount":sum(r["priceKgRaw"] is None or str(r["priceKgRaw"]).strip()=="" for r in rows),
                                "invalidPriceCount":sum((p is None or p <= 0) and r["priceKgRaw"] is not None and str(r["priceKgRaw"]).strip()!="" for p,r in zip(prices,rows)),
                                "questionableRowsBlockDailyRatio":questionable,
                                "priceKgDistribution":_stats(valid),"nationalPricePerKg":national_daily[day]["signedPricePerKg"],
                                "ratioMin":_decimal_text(min(ratios)) if ratios else None,
                                "ratioMedian":_decimal_text(sorted(ratios)[len(ratios)//2] if len(ratios)%2 else
                                    (sorted(ratios)[len(ratios)//2-1]+sorted(ratios)[len(ratios)//2])/Decimal(2)) if ratios else None,
                                "ratioMax":_decimal_text(max(ratios)) if ratios else None,
                                "pairedRawPrices":[r["priceKgRaw"] for r in rows]}
    covered_2026=(national_coverage[0]<=dt.date(2026,8,1)<=national_coverage[1])
    overlap=sum(bool(national_daily[d]["validPrice"]) and any(per_day[d][name]["validPriceCount"] for name in SEOAN_NAMES) for d in days)
    return {"schemaVersion":1,"protocol":"seoan-pepper-target-mapping-2026-10-09",
            "interpretation":"Product/unit plausibility only; not identity, publication-time evidence, or forecasting benefit.",
            "seoan":{"period":[SEOAN_START,SEOAN_END],"header":header,"schema":schema,
                     "selectedNames":list(SEOAN_NAMES),"selectedRowCount":len(selected),"rows":selected},
            "national":{"productCode":PEPPER_CODE,"productName":PEPPER_NAME,
                        "coverage":[national_coverage[0].isoformat(),national_coverage[1].isoformat()],
                        "periodOverlapsSeoan":national_coverage[0]<=dt.date(2024,8,1)<=national_coverage[1],
                        "periodCovers2026August":covered_2026,"daily":national_daily},
            "sameDate":{"overlapDays":overlap,"expectedDays":len(days),"byDateAndSeoanName":per_day,
                         "ratioRangeByName":{name:{"n":len(values),"min":_decimal_text(min(values)) if values else None,
                            "median":_decimal_text(sorted(values)[len(values)//2] if len(values)%2 else
                                (sorted(values)[len(values)//2-1]+sorted(values)[len(values)//2])/Decimal(2)) if values else None,
                            "max":_decimal_text(max(values)) if values else None} for name,values in ratio_values.items()}}}


def build_report(root:Path=ROOT)->dict:
    seoan_raw,manifest=_seoan_input(root)
    archive_raw=(root/NATIONAL_ZIP.relative_to(ROOT)).read_bytes()
    if hashlib.sha256(archive_raw).hexdigest()!=NATIONAL_SHA256:
        raise ValueError("pinned national ZIP digest mismatch")
    parsed_seoan=_parse_seoan(seoan_raw)
    parsed_national=_parse_national(archive_raw)
    result=analyze_rows(parsed_seoan["rows"],parsed_national["rows"],header=parsed_seoan["header"],
                        schema=parsed_seoan.get("columns"))
    # Add source provenance only after deterministic parsing and aggregation.
    result["sourceDigests"]={"seoanXlsx":SEOAN_SHA256,"nationalZip":NATIONAL_SHA256}
    result["sourceManifests"]={"seoanAttemptManifest":str(SEOAN_MANIFEST.relative_to(ROOT)),
                               "seoanCaptureStatus":manifest.get("http_status"),
                               "seoanCapturedAtUtc":manifest.get("retrieved_at_utc"),
                               "nationalArchive":str(NATIONAL_ZIP.relative_to(ROOT))}
    result["reportSha256"]=_sha(json.dumps(result,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode())
    return result


def write_report(path:Path=OUTPUT_PATH,root:Path=ROOT)->dict:
    report=build_report(root)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return report


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUTPUT_PATH)
    args=parser.parse_args(argv)
    result=write_report(args.output)
    print(json.dumps({"output":str(args.output),"seoanRows":result["seoan"]["selectedRowCount"],
                      "nationalCoverage":result["national"]["coverage"],
                      "overlapDays":result["sameDate"]["overlapDays"],
                      "periodCovers2026August":result["national"]["periodCovers2026August"],
                      "reportSha256":result["reportSha256"]},ensure_ascii=False,sort_keys=True))
    return 0


if __name__=="__main__":raise SystemExit(main())
