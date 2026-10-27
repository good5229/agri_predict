"""Post-hoc date-tolerance sensitivity for exact standardized market strata."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path

from . import regional_gap, regional_standardize

GWANGJU, GARAK, GANGSEO = regional_standardize.GWANGJU, regional_standardize.GARAK, regional_standardize.GANGSEO
MARKETS = regional_standardize.MARKETS
PRODUCTS = regional_standardize.PRODUCTS
PROTOCOL = {
    "id": "regional-settlement-date-tolerance-exploratory-2026-10-05",
    "period": [regional_standardize.START.isoformat(), regional_standardize.END.isoformat()],
    "matching": "all standardized non-date stratum fields exact; Seoul markets pooled within each settlement date",
    "assignment": "one-to-one per exact stratum; prioritize absolute lag 0, then 1, then 2; ties choose earlier Seoul date",
    "lag": "Seoul settlement date minus Gwangju anchor date, calendar days",
    "pairing": "signed net amount and weight must be positive; common kg=min(Gwangju kg, Seoul kg); normalize within Gwangju anchor date then equally weight dates",
    "coverage": "matched common kg divided by full-period positive net source-eligible kg by market",
    "exclusions": "candidatePairs counts candidate anchor-date x Seoul-date edges rejected before one-to-one assignment; potentialCommonKg sums min(max(Gwangju net kg,0), max(Seoul net kg,0)) per rejected edge and may count a date/stratum more than once",
    "inference": "tolerance 2 only; regional_gap._infer; Holm15 across five products and three tests",
    "limitations": "post-hoc exploratory; nonzero-date pairs have temporal mismatch and do not establish same-day prices or physical-size equivalence",
}


def _missing_label(rec):
    """Missing/dot names are unknown, while conflicting known labels exclude."""
    size_unknown = False
    for market, cell in rec["markets"].items():
        for field, values in cell["names"].items():
            values = set(values)
            if len(values) > 1:
                return "meaningful_name_conflict", False
            value = next(iter(values), "")
            if field == "sz_nm" and value == ".":
                size_unknown = True
            elif value in regional_standardize.MISSING:
                return "missing_or_placeholder_meaningful_name", False
    return None, size_unknown


def _stratum_parts(key):
    return key[0], dt.date.fromisoformat(key[1]), key[2:]


def _positive_seoul(rec):
    amount = sum((rec["markets"].get(m, {}).get("amount", 0) for m in (GARAK, GANGSEO)), start=0)
    weight = sum((rec["markets"].get(m, {}).get("weight", 0) for m in (GARAK, GANGSEO)), start=0)
    return amount, weight


def _scope_ready(states, product, day, market):
    return regional_standardize._scope_ready(states, product, day.isoformat(), market)


def _match_tolerance(strata, states, eligible_weights, tolerance):
    groups = defaultdict(lambda: {"gw": {}, "seoul": {}})
    excluded = defaultdict(lambda: defaultdict(lambda: {"candidatePairs": 0, "potentialCommonKg": 0.0}))

    def exclude(product, reason, gw_weight, seoul_weight):
        item = excluded[product][reason]
        item["candidatePairs"] += 1
        item["potentialCommonKg"] += float(min(max(gw_weight, 0), max(seoul_weight, 0)))

    for key, rec in strata.items():
        product, day, identity = _stratum_parts(key)
        if product not in PRODUCTS:
            continue
        gw = rec["markets"].get(GWANGJU)
        sa, sw = _positive_seoul(rec)
        if gw:
            groups[(product, identity)]["gw"][day] = (gw["amount"], gw["weight"], rec)
        if any(market in rec["markets"] for market in (GARAK, GANGSEO)):
            # Preserve observed Seoul date presence even when signed amount and
            # weight cancel to zero; the candidate then receives an explicit
            # nonpositive-net exclusion instead of disappearing.
            groups[(product, identity)]["seoul"][day] = (sa, sw, rec)
    pairs = defaultdict(list)
    for (product, identity), sides in groups.items():
        candidates = []
        for gd, (ga, gw, grec) in sides["gw"].items():
            for sd, (sa, sw, srec) in sides["seoul"].items():
                lag = (sd-gd).days
                if abs(lag) > tolerance:
                    continue
                if not all(_scope_ready(states, product, d, m) for d in {gd, sd} for m in MARKETS):
                    exclude(product, "source_incomplete", gw, sw)
                    continue
                reason, unk = _missing_label(grec)
                reason2, unk2 = _missing_label(srec)
                if reason or reason2:
                    exclude(product, reason or reason2, gw, sw)
                    continue
                if min(ga, gw, sa, sw) <= 0:
                    exclude(product, "nonpositive_net_pair", gw, sw)
                    continue
                candidates.append((abs(lag), sd, gd, lag, ga, gw, sa, sw, unk or unk2))
        used_g, used_s = set(), set()
        for _, sd, gd, lag, ga, gw, sa, sw, unknown in sorted(candidates, key=lambda x:(x[0], x[1], x[2])):
            if gd in used_g or sd in used_s:
                continue
            used_g.add(gd); used_s.add(sd)
            common = min(gw, sw)
            gap = math.log(float(ga/gw) / float(sa/sw))
            pairs[product].append({"gwangjuDate": gd, "seoulDate": sd, "lagDays": lag,
                "commonKg": float(common), "gap": gap, "sizeUnknown": unknown})
    reports = {}
    for product in PRODUCTS:
        by_day = defaultdict(lambda: {"sum":0.0,"weight":0.0,"cells":0,"sizeUnknownKg":0.0,"sizeUnknownCells":0})
        for pair in pairs[product]:
            d = by_day[pair["gwangjuDate"]]
            d["sum"] += pair["commonKg"]*pair["gap"]; d["weight"] += pair["commonKg"]; d["cells"] += 1
            if pair["sizeUnknown"]:
                d["sizeUnknownKg"] += pair["commonKg"]; d["sizeUnknownCells"] += 1
        gaps = {d.isoformat(): v["sum"]/v["weight"] for d,v in by_day.items() if v["weight"] > 0}
        matched = sum(v["weight"] for v in by_day.values())
        gwden=seolden=0.0
        for (p, day, market, _), net in eligible_weights.items():
            if p==product and net>0:
                if market==GWANGJU: gwden += float(net)
                elif market in (GARAK,GANGSEO): seolden += float(net)
        lags=defaultdict(int)
        for x in pairs[product]: lags[str(x["lagDays"])] += 1
        unknown_cells=sum(v["sizeUnknownCells"] for v in by_day.values()); unknown_kg=sum(v["sizeUnknownKg"] for v in by_day.values())
        mean=sum(gaps.values())/len(gaps) if gaps else None
        reports[product]={"matchedDates":len(gaps),"matchedCells":sum(v["cells"] for v in by_day.values()),
            "matchedWeightKg":matched,"gwangjuEligibleKg":gwden,"seoulEligibleKg":seolden,
            "shareOfGwangjuEligibleKg":matched/gwden if gwden else None,"shareOfSeoulEligibleKg":matched/seolden if seolden else None,
            "sizeUnknownCells":unknown_cells,"sizeUnknownWeightKg":unknown_kg,
            "sizeUnknownShareOfMatchedCells":unknown_cells/sum(v["cells"] for v in by_day.values()) if sum(v["cells"] for v in by_day.values()) else None,
            "sizeUnknownShareOfMatchedWeightKg":unknown_kg/matched if matched else None,
            "signedLagCounts":dict(sorted(lags.items(),key=lambda x:int(x[0]))),"meanDailyLogGap":mean,
            "geometricPriceRatio":math.exp(mean) if mean is not None else None,"dailyLogGaps":gaps,
            "excluded":{k:dict(v) for k,v in excluded[product].items()}}
    return reports


def analyze_strata(strata, states=None, eligible_weights=None, exclusions=None, provenance=None):
    eligible_weights=eligible_weights or {}; states=states or {}
    scenarios={str(k):_match_tolerance(strata,states,eligible_weights,k) for k in (0,1,2)}
    baseline=regional_standardize.analyze_strata(strata,states,eligible_weights,exclusions,provenance)
    # Enforce same-day agreement with the established exact-stratum estimator.
    for p, base in zip(PRODUCTS, baseline["products"]):
        zero = scenarios["0"][p]
        if (zero["matchedDates"] != base["matchedDates"] or
                zero["matchedCells"] != base["matchedStrata"] or
                not math.isclose(zero["matchedWeightKg"], base["matchedKg"], rel_tol=1e-12, abs_tol=1e-9) or
                set(zero["dailyLogGaps"]) != set(base["dailyLogGaps"]) or
                any(not math.isclose(zero["dailyLogGaps"][d], base["dailyLogGaps"][d], rel_tol=1e-12, abs_tol=1e-12)
                    for d in zero["dailyLogGaps"])):
            raise ValueError(f"0-day match does not reproduce standardized baseline for {p}")
    tests=[]
    for p in PRODUCTS:
        values={dt.date.fromisoformat(d):v for d,v in scenarios["2"][p]["dailyLogGaps"].items()}
        inf=regional_gap._infer(values)
        scenarios["2"][p]["inference"]=inf
        for test,key in (("meanZeroTest","meanZeroTest"),("equivalenceTOST","equivalenceTOST"),("smallGapDays","smallGapDays")):
            item=inf.get(key,{}) if inf.get("status")=="available" else {}
            tests.append({"productId":p,"test":test,"pValue":item.get("pValue"),"available":inf.get("status")=="available"})
    ordered=sorted(tests,key=lambda x:(x["pValue"] if x["pValue"] is not None else 1,p:=x["productId"],x["test"]))
    running=0.0
    for i,item in enumerate(ordered):
        running=max(running,min(1.0,(len(ordered)-i)*(item["pValue"] if item["pValue"] is not None else 1.0)))
        item["holmAdjustedPValue"]=running; item["holmSignificantAt005"]=running<=.05
    lookup={(x["productId"],x["test"]):x for x in tests}
    for p in PRODUCTS: scenarios["2"][p]["holmTests"]={t:lookup[(p,t)] for t in ("meanZeroTest","equivalenceTOST","smallGapDays")}
    return {"schemaVersion":1,"status":"complete","exploratoryPostHoc":True,
        "temporalMismatchWarning":"1- and 2-day matches compare different settlement dates; descriptive exploratory sensitivity only",
        "protocol":{**PROTOCOL,"sha256":regional_standardize._sha(PROTOCOL)},"provenance":provenance or {},
        "sourceStates":states,"scenarios":scenarios,"multipleTestingTolerance2":tests,
        "sameDayBaseline":{"protocolId":baseline["protocol"]["id"],"products":baseline["products"]}}


def run_analysis(db_path):
    db=sqlite3.connect("file:"+str(Path(db_path).resolve())+"?mode=ro",uri=True); db.row_factory=sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    try:
        db.execute("BEGIN")
        strata,states,eligible,exclusions,provenance=regional_standardize._source_strata(db)
        result=analyze_strata(strata,states,eligible,exclusions,provenance); db.rollback(); return result
    finally: db.close()


def _atomic(path,text):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f: f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def write_report(report,json_path,markdown_path):
    _atomic(json_path,json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    lines=["# 정산일 ±2일 민감도 분석","","사후 탐색 분석입니다. 정산일이 다른 거래는 시간적 불일치가 있으며, 기록 코드·명칭 일치도 물리적 크기나 품질의 동등성을 뜻하지 않습니다.","",
        "| 허용차 | 품목 | 짝 날짜 | 매칭 셀 | 공통 kg | 광주 적격량 비율 | 서울 적격량 비율 | 가격비 | 추론 |","|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for tol, products in report["scenarios"].items():
        for p,x in products.items():
            ratio=x["geometricPriceRatio"]; fmt=f"{ratio:.3f}" if ratio is not None else "—"
            inference=x.get("inference",{}).get("status","설명통계만")
            lines.append(f"| ±{tol}일 | {p} | {x['matchedDates']} | {x['matchedCells']} | {x['matchedWeightKg']:.1f} | {x['shareOfGwangjuEligibleKg']:.1%}" if x["shareOfGwangjuEligibleKg"] is not None else f"| ±{tol}일 | {p} | {x['matchedDates']} | {x['matchedCells']} | {x['matchedWeightKg']:.1f} | —")
            lines[-1]+=f" | {x['shareOfSeoulEligibleKg']:.1%}" if x["shareOfSeoulEligibleKg"] is not None else " | —"
            lines[-1]+=f" | {fmt} | {inference} |"
    lines += ["", "Exclusions are counted as candidate date-pair edges (one exact stratum × one Gwangju date × one Seoul date), before one-to-one assignment. Potential common kg sums min(positive net Gwangju kg, positive net Seoul kg) over these edges, so the same stratum/date may contribute to multiple rejected candidate edges."]
    for tol, products in report["scenarios"].items():
        for p, x in products.items():
            detail = ", ".join(f"{reason}: {item['candidatePairs']} candidate pairs/{item['potentialCommonKg']:.1f} potential kg" for reason, item in x["excluded"].items())
            if detail:
                lines.append(f"- ±{tol}일 {p} 제외: {detail}")
    lines += ["", "Only the ±2-day scenario receives the frozen inference and Holm15 tests. The ±1-day scenario is descriptive. The ±0-day scenario is checked against the existing exact-date standardized baseline."]
    _atomic(markdown_path,"\n".join(lines)+"\n")


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--db",default="data/agri.sqlite"); ap.add_argument("--output",required=True); ap.add_argument("--report")
    args=ap.parse_args(argv); db=Path(args.db).resolve(); out=Path(args.output).resolve(); md=Path(args.report).resolve() if args.report else out.with_suffix(".md")
    if out in (db,md) or md==db or out==md: raise ValueError("output paths must be distinct from each other and the database")
    write_report(run_analysis(db),out,md); return 0

if __name__=="__main__": raise SystemExit(main())
