"""Selective post-hoc date tolerance for garlic and pepper recorded-code cells."""
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

from . import regional_gap

GARLIC, PEPPER = "garlic", "pepper"
SELECTED_PRODUCTS = (GARLIC, PEPPER)
ALL_PRODUCTS = regional_gap.PRODUCTS
GWANGJU, GARAK, GANGSEO = regional_gap.GWANGJU, regional_gap.GARAK, regional_gap.GANGSEO
MARKETS = regional_gap.MARKETS
ROOT = Path(__file__).resolve().parents[1]
RECORDED_CODE_REPORT = ROOT / "outputs/diagnostics/regional-gap-recorded-code-20261005/report.json"
STANDARDIZED_REPORT = ROOT / "outputs/diagnostics/regional-standardized-20261005/report.json"
BASELINE_EXPECTED = {
    GARLIC: {"matchedDates": 209, "matchedCells": 323, "matchedWeightKg": 129334.5},
    PEPPER: {"matchedDates": 1, "matchedCells": 1, "matchedWeightKg": 18.0},
}
PROTOCOL = {
    "id": "selective-near-date-recorded-code-exploratory-2026-10-06",
    "period": [regional_gap.START.isoformat(), regional_gap.END.isoformat()],
    "selectedProducts": list(SELECTED_PRODUCTS),
    "unchangedProducts": [p for p in ALL_PRODUCTS if p not in SELECTED_PRODUCTS],
    "matching": "exact four recorded codes; exact date first, then absolute 1 day, then absolute 2 days; tie earlier Seoul date; one-to-one without reuse",
    "seoul": "pool Garak and Gangseo signed net amount and kg by exact code cell and same settlement date",
    "pairing": "positive signed net amount and kg on both sides; common kg=min side kg; normalize by common kg within Gwangju anchor date, then equal weight dates",
    "lag": "Seoul settlement date minus Gwangju anchor date, calendar days",
    "names": "allow missing labels and exact dot marker; reject other placeholders and conflicts among known labels",
    "coverage": "matched common kg divided by full-period positive source-eligible kg by market",
    "inference": "only ±2 days; regional_gap._infer; Holm6 across two products x three tests",
    "interpretation": "post-hoc exploratory selection; temporal mismatch for nonzero lag; recorded codes do not prove physical-size or quality equality",
}


def _load_json(path):
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


def _scope_ready(states, product, day, market):
    return regional_gap._scope_ready(states, product, day, market)


def _pool_seoul(market):
    out = defaultdict(dict)
    for m in (GARAK, GANGSEO):
        for key, cell in market.get(m, {}).items():
            if key[0] not in SELECTED_PRODUCTS:
                continue
            target = out[key[0]].setdefault((key[1], key[2], key[3]),
                {"amount": 0, "weight": 0, "names": [], "rows": 0})
            target["amount"] += cell["amount"]
            target["weight"] += cell["weight"]
            target["names"] = regional_gap._merge_names(target["names"], cell["names"])
            target["rows"] += cell["rows"]
    return out


def _eligible_totals(eligible_cells):
    totals = {p: {m: 0.0 for m in MARKETS} for p in SELECTED_PRODUCTS}
    for (product, _day, market, _codes), rec in (eligible_cells or {}).items():
        if product in totals and market in MARKETS:
            totals[product][market] += float(max(rec["weight"], 0))
    return totals


def _match_scenario(cells, states, eligible_cells, tolerance):
    market = regional_gap._market_cells(cells)
    seoul = _pool_seoul(market)
    reports = {}
    for product in SELECTED_PRODUCTS:
        # Index each exact four-code stratum by date, preserving observed zero-net cells.
        gw_by = defaultdict(dict)
        seoul_by = defaultdict(dict)
        identities = set()
        for key, rec in market.get(GWANGJU, {}).items():
            if key[0] == product:
                day, subclass, dims = dt.date.fromisoformat(key[1]), key[2], tuple(key[3])
                ident = (subclass, dims); identities.add(ident); gw_by[ident][day] = rec
        for (day_text, subclass, dims), rec in seoul[product].items():
            day = dt.date.fromisoformat(day_text); ident = (subclass, tuple(dims))
            identities.add(ident); seoul_by[ident][day] = rec

        by_anchor = defaultdict(lambda: {"weighted": 0.0, "weight": 0.0, "cells": 0,
                                         "unknownCells": 0, "unknownKg": 0.0})
        lags = defaultdict(int)
        excluded = defaultdict(lambda: {"candidatePairs": 0, "potentialCommonKg": 0.0})
        matched_pairs = []

        def reject(reason, gw_weight, seoul_weight):
            item = excluded[reason]; item["candidatePairs"] += 1
            item["potentialCommonKg"] += float(min(max(gw_weight, 0), max(seoul_weight, 0)))

        for ident in identities:
            candidates = []
            for gd, g in gw_by[ident].items():
                for sd, s in seoul_by[ident].items():
                    lag = (sd-gd).days
                    if abs(lag) > tolerance:
                        continue
                    if regional_gap._dimension_code_status(ident[1]):
                        reject("missing_or_placeholder_dimension_code", g["weight"], s["weight"])
                        continue
                    if any(not _scope_ready(states, product, d.isoformat(), m)
                           for d in {gd, sd} for m in MARKETS):
                        reject("source_incomplete", g["weight"], s["weight"])
                        continue
                    status = regional_gap._name_status(
                        regional_gap._combine_market_names(g["names"], s["names"]), "recorded_code")
                    if status not in ("compatible", "unknown_label"):
                        reject(status, g["weight"], s["weight"])
                        continue
                    if min(g["amount"], g["weight"], s["amount"], s["weight"]) <= 0:
                        reject("nonpositive_net_pair", g["weight"], s["weight"])
                        continue
                    candidates.append((abs(lag), sd, gd, lag, g, s, status == "unknown_label"))
            used_gw, used_seoul = set(), set()
            for _, sd, gd, lag, g, s, unknown in sorted(candidates, key=lambda x:(x[0], x[1], x[2])):
                if gd in used_gw or sd in used_seoul:
                    continue
                used_gw.add(gd); used_seoul.add(sd)
                common = min(g["weight"], s["weight"])
                gap = math.log(float(g["amount"]/g["weight"]) / float(s["amount"]/s["weight"]))
                day = by_anchor[gd]
                day["weighted"] += float(common)*gap; day["weight"] += float(common); day["cells"] += 1
                if unknown:
                    day["unknownCells"] += 1; day["unknownKg"] += float(common)
                lags[str(lag)] += 1
                matched_pairs.append({"gwangjuDate": gd.isoformat(), "seoulDate": sd.isoformat(),
                    "lagDays": lag, "codes": list(ident[1]), "sharedWeightKg": float(common),
                    "unknownLabel": unknown})
        daily = {day.isoformat(): rec["weighted"]/rec["weight"]
                 for day, rec in by_anchor.items() if rec["weight"] > 0}
        common_kg = sum(rec["weight"] for rec in by_anchor.values())
        cells_n = sum(rec["cells"] for rec in by_anchor.values())
        unknown_n = sum(rec["unknownCells"] for rec in by_anchor.values())
        unknown_kg = sum(rec["unknownKg"] for rec in by_anchor.values())
        eligible = _eligible_totals(eligible_cells)[product]
        mean = sum(daily.values())/len(daily) if daily else None
        reports[product] = {
            "matchedDates": len(daily), "matchedCells": cells_n, "matchedWeightKg": common_kg,
            "gwangjuEligibleKg": eligible[GWANGJU],
            "seoulEligibleKg": eligible[GARAK]+eligible[GANGSEO],
            "shareOfGwangjuEligibleKg": common_kg/eligible[GWANGJU] if eligible[GWANGJU] else None,
            "shareOfSeoulEligibleKg": common_kg/(eligible[GARAK]+eligible[GANGSEO]) if eligible[GARAK]+eligible[GANGSEO] else None,
            "signedLagCounts": dict(sorted(lags.items(), key=lambda x:int(x[0]))),
            "unknownLabelMatchedCells": unknown_n, "unknownLabelMatchedWeightKg": unknown_kg,
            "unknownLabelMatchedCellShare": unknown_n/cells_n if cells_n else None,
            "unknownLabelMatchedWeightShare": unknown_kg/common_kg if common_kg else None,
            "dailyLogGaps": daily, "meanDailyLogGap": mean,
            "geometricPriceRatio": math.exp(mean) if mean is not None else None,
            "excluded": {k: dict(v) for k,v in excluded.items()},
            "matchedPairs": matched_pairs,
        }
    return reports


def _assert_baseline(actual, baseline_report):
    """Require exact 0-day matched dates/cells/weights/daily gaps vs recorded-code report."""
    expected = {p["productId"]: p for p in baseline_report["products"]}
    for product in SELECTED_PRODUCTS:
        got, want = actual[product], expected[product]
        if (got["matchedDates"] != want["matchedDates"] or
                got["matchedCells"] != want["matchedCells"] or
                not math.isclose(got["matchedWeightKg"], want["matchedWeightKg"], rel_tol=1e-12, abs_tol=1e-9) or
                set(got["dailyLogGaps"]) != set(want["pairedDailyLogGaps"]) or
                any(not math.isclose(got["dailyLogGaps"][d], want["pairedDailyLogGaps"][d], rel_tol=1e-12, abs_tol=1e-12)
                    for d in got["dailyLogGaps"])):
            raise ValueError(f"zero-day selective matcher does not reproduce recorded-code baseline for {product}")
        if BASELINE_EXPECTED[product] != {k: got[k] for k in BASELINE_EXPECTED[product]}:
            # Exact fixed count/weight guard against an accidentally changed stored baseline.
            raise ValueError(f"recorded-code baseline no longer matches frozen reference for {product}")


def analyze_cells(cells, states=None, provenance=None, exclusions=None, raw_totals=None,
                  eligible_cells=None, recorded_code_baseline=None, standardized_baseline=None):
    states = states or {}
    if recorded_code_baseline is None:
        recorded_code_baseline = _load_json(RECORDED_CODE_REPORT)
    if standardized_baseline is None:
        standardized_baseline = _load_json(STANDARDIZED_REPORT)
    scenarios = {str(k): _match_scenario(cells, states, eligible_cells, k) for k in (0,1,2)}
    _assert_baseline(scenarios["0"], recorded_code_baseline)
    for p in SELECTED_PRODUCTS:
        scenarios["2"][p]["inference"] = regional_gap._infer(
            {dt.date.fromisoformat(d):v for d,v in scenarios["2"][p]["dailyLogGaps"].items()})
    tests=[]
    for product in SELECTED_PRODUCTS:
        inf=scenarios["2"][product]["inference"]
        for name in ("meanZeroTest", "equivalenceTOST", "smallGapDays"):
            test=inf.get(name,{}) if inf.get("status")=="available" else {}
            tests.append({"productId":product,"test":name,"pValue":test.get("pValue"),"available":inf.get("status")=="available"})
    ordered=sorted(tests,key=lambda r:(r["pValue"] if r["pValue"] is not None else 1.0,r["productId"],r["test"]))
    running=0.0
    for i,test in enumerate(ordered):
        running=max(running,min(1.0,(len(ordered)-i)*(test["pValue"] if test["pValue"] is not None else 1.0)))
        test["holmAdjustedPValue"]=running;test["holmSignificantAt005"]=running<=.05
    test_lookup={(x["productId"],x["test"]):x for x in tests}
    for product in SELECTED_PRODUCTS:
        scenarios["2"][product]["holmTests"]={n:test_lookup[(product,n)] for n in ("meanZeroTest","equivalenceTOST","smallGapDays")}
    standardized={p["productId"]:p for p in standardized_baseline["products"]}
    unchanged={p:standardized[p] for p in ALL_PRODUCTS if p not in SELECTED_PRODUCTS}
    recorded={p["productId"]:p for p in recorded_code_baseline["products"]}
    return {"schemaVersion":1,"status":"complete","exploratoryPostHoc":True,
        "interpretation":PROTOCOL["interpretation"],"protocol":{**PROTOCOL,"sha256":regional_gap._sha(PROTOCOL)},
        "provenance":provenance or {},"sourceStates":states,
        "unchangedExactDayProducts":unchanged,
        "scenarios":scenarios,
        "recordedCodeExactDayBaseline":{p:recorded[p] for p in SELECTED_PRODUCTS},
        "multipleTestingTolerance2":tests}


def run_analysis(db_path, recorded_code_baseline=None, standardized_baseline=None):
    db=sqlite3.connect("file:"+str(Path(db_path).resolve())+"?mode=ro",uri=True);db.row_factory=sqlite3.Row
    db.execute("PRAGMA query_only=ON")
    try:
        db.execute("BEGIN")
        cells,states,provenance,exclusions,raw_totals,eligible=regional_gap._source_cells(db)
        result=analyze_cells(cells,states,provenance,exclusions,raw_totals,eligible,
            recorded_code_baseline,standardized_baseline)
        db.rollback();return result
    finally: db.close()


def _atomic(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",suffix=".tmp",dir=path.parent)
    try:
        with os.fdopen(fd,"w",encoding="utf-8") as f:f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def write_report(report,json_path,markdown_path):
    _atomic(json_path,json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    lines=["# 마늘·고추 한정 인접일 기록 코드 매칭","",
        "배추·무·양파의 앞선 동일일 표준화 결과는 그대로 유지합니다. 이 사후 탐색 비교는 깐마늘·건고추만 다루며, 서로 다른 정산일의 가격에는 시간 변동이 섞입니다. 기록 코드나 명칭이 같아도 물리 크기·품질이 같음을 뜻하지 않습니다.","",
        "| 허용차 | 품목 | 매칭 날짜 | 코드 셀 | 공통 kg | 광주 적격량 비율 | 서울 적격량 비율 | 가격비 | 추론 |","|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for tol,products in report["scenarios"].items():
        for product,x in products.items():
            ratio=x["geometricPriceRatio"];ratio_text=f"{ratio:.3f}" if ratio is not None else "—"
            inf=x.get("inference",{}).get("status","기술통계만")
            gw=x["shareOfGwangjuEligibleKg"];se=x["shareOfSeoulEligibleKg"]
            lines.append(f"| ±{tol}일 | {product} | {x['matchedDates']} | {x['matchedCells']} | {x['matchedWeightKg']:.1f} | {gw:.1%}" if gw is not None else f"| ±{tol}일 | {product} | {x['matchedDates']} | {x['matchedCells']} | {x['matchedWeightKg']:.1f} | —")
            lines[-1] += f" | {se:.1%}" if se is not None else " | —"
            lines[-1] += f" | {ratio_text} | {inf} |"
    lines += ["","±1일은 기술통계만 제시하고, ±2일에만 기존 표본 게이트·HAC28/9·블록 부트스트랩 및 Holm6 추론을 적용합니다. ±0일은 저장된 기존 기록 코드 결과와 날짜별 격차·셀·중량을 일치 검증합니다."]
    _atomic(markdown_path,"\n".join(lines)+"\n")


def _validate_output_paths(db_path, output_path, report_path):
    db, out, md = (Path(x).resolve() for x in (db_path, output_path, report_path))
    frozen_inputs = {RECORDED_CODE_REPORT.resolve(), STANDARDIZED_REPORT.resolve()}
    if out == md or out == db or md == db:
        raise ValueError("JSON/report outputs must be distinct from each other and the database")
    if out in frozen_inputs or md in frozen_inputs:
        raise ValueError("output paths must not overwrite frozen baseline reports")
    return db, out, md


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--db",default="data/agri.sqlite");ap.add_argument("--output",required=True);ap.add_argument("--report")
    args=ap.parse_args(argv);out=Path(args.output).resolve();md=Path(args.report).resolve() if args.report else out.with_suffix(".md")
    db,out,md=_validate_output_paths(args.db,out,md)
    write_report(run_analysis(db),out,md);return 0

if __name__=="__main__":raise SystemExit(main())
