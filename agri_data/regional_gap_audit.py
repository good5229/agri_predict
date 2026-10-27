"""Audit why strict cross-market regional price cells do or do not match."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sqlite3
import tempfile
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from . import coverage, market_extension, regional_gap


def _market_cells(cells):
    result=defaultdict(dict)
    for (product,day,market,subclass,codes),value in cells.items():
        key=(product,day,subclass,codes)
        cell=result[market].setdefault(key,{"amount":Decimal(0),"weight":Decimal(0),"rows":0,"names":[]})
        cell["amount"]+=value["amount"];cell["weight"]+=value["weight"];cell["rows"]+=value["rows"]
        cell["names"]=regional_gap._merge_names(cell["names"],value["names"])
    return result


def _name_status(left,right):
    return regional_gap._name_status(regional_gap._combine_market_names(left,right))


def audit_cells(cells, states=None, provenance=None, source_exclusions=None, request_scopes=None):
    """Create progressive code/name compatibility counts from selected normalized cells."""
    markets=_market_cells(cells);products=[]
    scope_states=(states or {}).get("scopeStates",{})
    def ready(product,day,market):
        if not scope_states:return True
        items=scope_states.get(f"{product}|{day}|{market}",[])
        return bool(items) and all(s in ("complete","empty") for s in items)
    request_scopes=request_scopes or {}
    for product in regional_gap.PRODUCTS:
        by_market={m:{} for m in regional_gap.MARKETS}
        for m in regional_gap.MARKETS:
            by_market[m]={k:v for k,v in markets.get(m,{}).items() if k[0]==product}
        market_dates={m:{k[1] for k in by_market[m]} for m in regional_gap.MARKETS}
        common_dates=sorted(d for d in (market_dates[regional_gap.GWANGJU] &
            (market_dates[regional_gap.GARAK]|market_dates[regional_gap.GANGSEO]))
            if all(ready(product,d,m) for m in regional_gap.MARKETS))
        seoul={}
        for m in (regional_gap.GARAK,regional_gap.GANGSEO):
            for key,value in by_market[m].items():
                item=seoul.setdefault(key,{"amount":Decimal(0),"weight":Decimal(0),"rows":0,"names":[]})
                item["amount"]+=value["amount"];item["weight"]+=value["weight"];item["rows"]+=value["rows"]
                item["names"]=regional_gap._merge_names(item["names"],value["names"])
        left=by_market[regional_gap.GWANGJU]
        stages=[]
        for length,label in enumerate(("subclass_code","grade_code","size_code","package_code"),start=1):
            def prefixes(mapping):
                out=defaultdict(set)
                for key in mapping:
                    p,day,sub,codes=key
                    if day in common_dates: out[(day,)+tuple(codes[:length])].add(key)
                return out
            gprefix=prefixes(left);sprefix=prefixes(seoul)
            matched=set(gprefix)&set(sprefix)
            stages.append({"stage":label,"matchedKeyCount":len(matched),"matchedDateCount":len({x[0] for x in matched})})
        exact_keys={k for k in left if k in seoul and k[1] in common_dates}
        positive=[];nonpositive=0;name_counts=defaultdict(int);name_weights=defaultdict(Decimal);matched_weight=Decimal(0)
        for key in exact_keys:
            g,s=left[key],seoul[key]
            if g["amount"]<=0 or g["weight"]<=0 or s["amount"]<=0 or s["weight"]<=0:
                nonpositive+=1;continue
            label_status=_name_status(g["names"],s["names"])
            name_counts[label_status]+=1;name_weights[label_status]+=min(g["weight"],s["weight"])
            if label_status=="compatible":
                positive.append(key);matched_weight+=min(g["weight"],s["weight"])
        stage_rows=defaultdict(int)
        for (p,day,m,sub,codes),v in cells.items():
            if p==product:stage_rows[m]+=v["rows"]
        combinations={}
        code_sets={}
        for m in regional_gap.MARKETS:
            summary=defaultdict(lambda:{"rows":0,"netWeightKg":Decimal(0),"names":[set() for _ in regional_gap.DIMENSIONS]})
            for key,v in by_market[m].items():
                _,day,sub,codes=key
                item=summary[tuple(codes)];item["rows"]+=v["rows"];item["netWeightKg"]+=v["weight"]
                for i,values in enumerate(v["names"]):item["names"][i].update(values)
            ranked=sorted(summary.items(),key=lambda item:(-item[1]["rows"],item[0]))
            combinations[m]=[{ "codes":list(codes),"names":[sorted(x) for x in item["names"]],
                "rows":item["rows"],"netWeightKg":float(item["netWeightKg"])} for codes,item in ranked[:25]]
            code_sets[m]=set(summary)
        usage_differences={
            "gwangjuOnlyCount":len(code_sets[regional_gap.GWANGJU]-code_sets[regional_gap.GARAK]-code_sets[regional_gap.GANGSEO]),
            "garakOnlyCount":len(code_sets[regional_gap.GARAK]-code_sets[regional_gap.GWANGJU]-code_sets[regional_gap.GANGSEO]),
            "gangseoOnlyCount":len(code_sets[regional_gap.GANGSEO]-code_sets[regional_gap.GWANGJU]-code_sets[regional_gap.GARAK]),
            "gwangjuNotInEitherSeoul": [list(x) for x in sorted(code_sets[regional_gap.GWANGJU]-(code_sets[regional_gap.GARAK]|code_sets[regional_gap.GANGSEO]))[:50]],
            "garakNotInOtherMarkets": [list(x) for x in sorted(code_sets[regional_gap.GARAK]-(code_sets[regional_gap.GWANGJU]|code_sets[regional_gap.GANGSEO]))[:50]],
            "gangseoNotInOtherMarkets": [list(x) for x in sorted(code_sets[regional_gap.GANGSEO]-(code_sets[regional_gap.GWANGJU]|code_sets[regional_gap.GARAK]))[:50]]}
        examples=[]
        if common_dates:
            chosen=sorted(set((common_dates[0],common_dates[-1])))[:2]
            for day in chosen:
                examples.append({"date":day,"markets":{m:request_scopes.get(f"{product}|{day}|{m}",[])[:3] for m in regional_gap.MARKETS}})
        product_exclusions=(source_exclusions or {}).get(product,{})
        products.append({"productId":product,"commonDates":len(common_dates),"commonDateExamples":common_dates[:2],
            "stages":stages,"positiveNetExactCodeCells":len(exact_keys)-nonpositive,"nonpositiveNetExactCodeCells":nonpositive,
            "nameCompatibility":{reason:{"cells":count,"sharedNetWeightKg":float(name_weights[reason])} for reason,count in name_counts.items()},
            "matchedCells":len(positive),"matchedSharedNetWeightKg":float(matched_weight),
            "sourceExcludedRowsAndWeight":product_exclusions,"rowsByMarket":dict(stage_rows),
            "codeCombinationsByMarket":combinations,"codeUsageDifferences":usage_differences,
            "requestExamples":examples})
    source_prov=provenance or {}
    safe_prov={k:source_prov[k] for k in ("period","markets","selectedScopeCount","selectedScopesSha256","requestCount") if k in source_prov}
    return {"schemaVersion":1,"status":"complete","period":[regional_gap.START.isoformat(),regional_gap.END.isoformat()],
        "markets":dict(gwangju=regional_gap.GWANGJU,garak=regional_gap.GARAK,gangseo=regional_gap.GANGSEO),
        "selectedQueryStates":(states or {}).get("counts",{}),"provenance":safe_prov,
        "products":products,"auditDefinition":{"matchingStages":["date and subclass code","plus grade code","plus size code","plus package code","positive signed net amount and weight","nonmissing nonplaceholder labels with no cross-market conflict"],
        "stageCountMeaning":"matchedKeyCount is the number of distinct matched date×prefix-code groups at that stage; it is not a count of cells lost and can rise as more code dimensions split groups. matchedDateCount is distinct dates represented by those groups.",
        "auditedPopulation":"regional_gap._source_cells first excludes source rows with missing or placeholder values in any of the four dimension codes. Common dates and all prefix-stage counts use the remaining complete four-code cells. Those source exclusions are reported separately under sourceExcludedRowsAndWeight.",
        "commonDateDefinition":"date with an eligible four-code cell in Gwangju and at least one Seoul market, and all scoped product/date queries for Gwangju, Garak and Gangseo are complete or empty.",
        "cellInterpretation":"Code and name agreement is a conservative screen, not proof of physical-quality equivalence.",
        "outputExcludes":["prices","raw payloads","query keys","credentials"]}}


def _request_scopes(db):
    db.create_function("coverage_task_key",3,coverage._request_task_key,deterministic=True)
    cte=market_extension._scoped_selection_cte(regional_gap.START,regional_gap.END,regional_gap.MARKETS)
    rows=db.execute(cte+"""SELECT r.request_id,s.response_sha256,p.query_json,p.selected_state,r.state
      FROM picked p JOIN request_pages r ON r.job_id=p.selected_job_id AND r.dataset=p.dataset
       AND r.source_url=p.source_url AND r.task_key=p.task_key AND r.epoch_id=p.selected_epoch_id
       AND r.state IN ('complete','empty')
      JOIN source_requests s ON s.request_id=r.request_id
      WHERE p.dataset='prices_origin' AND p.selected_state IN ('complete','empty')
      ORDER BY p.query_json,r.request_id""")
    result=defaultdict(list)
    for rid,digest,qtext,state,page_state in rows:
        query=json.loads(qtext);dims=coverage.price_dimensions(query,"prices_origin")
        if not dims or dims[2] not in regional_gap.PRODUCTS:continue
        product=dims[2];day=coverage.canonical_day(query.get("cond[trd_clcln_ymd::EQ]"));market=str(query.get("cond[whsl_mrkt_cd::EQ]",""))
        key=f"{product}|{day}|{market}"
        item={"requestId":rid,"responseSha256":digest,"queryState":state,"pageState":page_state}
        if item not in result[key]:result[key].append(item)
    return dict(result)


def run_audit(db_path:Path):
    db=sqlite3.connect("file:"+str(db_path.resolve())+"?mode=ro",uri=True);db.row_factory=sqlite3.Row;db.execute("PRAGMA query_only=ON")
    try:
        db.execute("BEGIN")
        cells,states,provenance,exclusions,_,_ = regional_gap._source_cells(db)
        request_scopes=_request_scopes(db)
        report=audit_cells(cells,states,provenance,exclusions,request_scopes)
        db.rollback();return report
    finally:db.close()


def _atomic(path:Path,text:str):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def write_report(report,json_path:Path,markdown_path:Path):
    lines=["# 도매시장 코드 매칭 감사","",
        f"기간 {report['period'][0]}–{report['period'][1]}, 광주 {regional_gap.GWANGJU} 대 가락 {regional_gap.GARAK}·강서 {regional_gap.GANGSEO}.",
        "가격과 원천 payload는 출력하지 않습니다. 명칭 일치는 코드의 물리적 규격 동등성을 입증하지 않습니다.","",
        "감사 모집단은 앞 단계에서 네 규격 코드가 모두 유효한 행으로 제한됩니다. 아래 단계 수는 셀 탈락 개수가 아니라 해당 단계에서 일치한 고유 date×prefix-code 그룹 수이며, 세분될수록 증가할 수 있습니다. 괄호는 해당 그룹이 나타난 거래일 수입니다. 누락/placeholder 코드는 sourceExcludedRowsAndWeight에 별도 집계됩니다.","",
        "| 품목 | 공통일 | 소분류 그룹(일) | +등급 그룹(일) | +크기 그룹(일) | +포장 그룹(일) | 양수 순합계 셀 | 최종 셀 | 공유중량 kg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for p in report["products"]:
        stage=[x["matchedKeyCount"] for x in p["stages"]]
        stage_dates=[x["matchedDateCount"] for x in p["stages"]]
        lines.append(f"| {p['productId']} | {p['commonDates']} | {stage[0]} ({stage_dates[0]}) | {stage[1]} ({stage_dates[1]}) | {stage[2]} ({stage_dates[2]}) | {stage[3]} ({stage_dates[3]}) | {p['positiveNetExactCodeCells']} | {p['matchedCells']} | {p['matchedSharedNetWeightKg']:.1f} |")
    lines.extend(["","## 제외 사유 및 코드 사용 차이",""])
    for p in report["products"]:
        names=", ".join(f"{k}: {v['cells']}" for k,v in p["nameCompatibility"].items()) or "명칭 제외 없음"
        lines.append(f"- {p['productId']}: 비양수 셀 {p['nonpositiveNetExactCodeCells']}; 명칭 검사 {names}; 코드 조합 고유 수—광주 {p['codeUsageDifferences']['gwangjuOnlyCount']}, 가락 {p['codeUsageDifferences']['garakOnlyCount']}, 강서 {p['codeUsageDifferences']['gangseoOnlyCount']}.")
        top={m:[{"codes":x["codes"],"names":x["names"],"rows":x["rows"]} for x in p["codeCombinationsByMarket"][m][:3]] for m in regional_gap.MARKETS}
        lines.append("  상위 코드·명칭 조합: "+json.dumps(top,ensure_ascii=False,separators=(",",":"))+".")
        if p["sourceExcludedRowsAndWeight"]:
            lines.append("  코드/행 제외: "+json.dumps(p["sourceExcludedRowsAndWeight"],ensure_ascii=False,separators=(",",":"))+".")
        for example in p["requestExamples"]:
            lines.append(f"  예시 날짜 {example['date']} 요청 ID/응답 해시: "+json.dumps(example["markets"],ensure_ascii=False,separators=(",",":"))+".")
    lines.extend(["","공통일은 광주에 해당 품목의 유효 4코드 셀이 있고, 가락 또는 강서 중 적어도 한 곳에도 셀이 있으며, 해당 품목·날짜의 세 시장 scoped query가 모두 complete/empty인 날입니다. 계획 query 중 실패·미완료·stale fallback이 하나라도 있으면 공통 매칭에서 제외합니다. 단계별 그룹 수는 소분류→등급→크기→포장 코드 순서이며, 최종 가중치는 광주와 서울 합산 순중량 중 작은 값입니다.",""])
    _atomic(json_path,json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    _atomic(markdown_path,"\n".join(lines))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--db",default="data/agri.sqlite");parser.add_argument("--output",required=True);parser.add_argument("--report")
    args=parser.parse_args(argv);db=Path(args.db).resolve();out=Path(args.output).resolve();md=Path(args.report).resolve() if args.report else out.parent/"report.md"
    if out==db or md==db:raise ValueError("outputs must not overwrite database")
    if out==md:raise ValueError("JSON and Markdown outputs must differ")
    if out==md:raise ValueError("JSON and Markdown outputs must differ")
    write_report(run_audit(db),out,md);return 0


if __name__=="__main__":raise SystemExit(main())
