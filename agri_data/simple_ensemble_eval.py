"""Frozen 50:50 Ridge/HistGB retrospective evaluation over pinned reports."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

DIRECT_SHA256 = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
HISTGB_SHA256 = "1032caa80628cbab54c38d9efe408339348ca9e6341b0df4e3306df64a606a1e"
START = dt.date(2025, 9, 25)
END = dt.date(2026, 9, 24)
EXPECTED_SHARED = 45_880
EXPECTED_EVALUATION = 12_508
PRODUCTS = ("cabbage", "radish", "onion", "garlic", "pepper")
HORIZONS = tuple(range(1, 11))
COMPARATORS = ("direct_ridge", "full_history_histgb", "persistence")
LOSSES = ("absolute_error", "absolute_percentage_error")
LAGS = (9, 28)
FROZEN_PROTOCOL = {
    "source": "docs/PLAN.md, 2026-10-07 section '거리별 Ridge·HistGB 단순 결합'",
    "evaluationTargets": [START.isoformat(), END.isoformat()],
    "expectedEvaluationPairs": EXPECTED_EVALUATION,
    "sharedInputRows": EXPECTED_SHARED,
    "candidate": "50:50 arithmetic mean of same-origin direct Ridge and full-history HistGB predictions; use direct Ridge if HistGB is unavailable, nonfinite, or nonpositive",
    "comparators": {"direct_ridge": "direct Ridge prediction", "full_history_histgb": "full-history HistGB prediction; compare only where available", "persistence": "persistence prediction"},
    "losses": {"absolute_error": "absolute error in price/kg", "absolute_percentage_error": "absolute absolute error / actual * 100"},
    "dm": {"twoSided": True, "primaryCalendarBartlettHacLagDays": 9, "sensitivityCalendarBartlettHacLagDays": 28,
           "holmAlpha": 0.05, "holmFamilySizePerBandwidth": 300,
           "unavailableRule": "n < 30 or nonpositive/nonfinite HAC variance remains in the family with p=1"},
    "macro": "For each candidate/comparator, average the available horizon-specific MAPE values equally within each product, then average available product values equally; report denominators.",
    "jointCellImprovement": "All six HAC9 comparator-by-loss tests must have negative mean loss difference and Holm-adjusted p < 0.05; cell must have at least 30 candidate pairs and coverage cannot be below any comparator.",
    "fallbackCoverage": "Candidate fallback is direct Ridge, so retains direct Ridge coverage; record count and reason.",
}


def _number(v: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(v, bool): raise ValueError(f"{label} must be finite")
    try: x = float(v)
    except (TypeError, ValueError, OverflowError) as e: raise ValueError(f"{label} must be finite") from e
    if not math.isfinite(x) or (positive and x <= 0): raise ValueError(f"{label} must be {'positive ' if positive else ''}finite")
    return x


def _key(row: dict[str, Any], label: str) -> tuple[str, int, str]:
    try: day = dt.date.fromisoformat(row["target"])
    except (KeyError, TypeError, ValueError) as e: raise ValueError(f"{label} target must be ISO date") from e
    h = row.get("horizon")
    if isinstance(h, bool) or not isinstance(h, int) or h not in HORIZONS: raise ValueError(f"{label} horizon must be 1..10")
    product = row.get("product")
    if product not in PRODUCTS: raise ValueError(f"{label} has unknown product")
    return product, h, day.isoformat()


def validate_reports(direct: dict[str, Any], hist: dict[str, Any]) -> tuple[dict, dict]:
    """Validate unique full report keys and frozen cross-report identity before scoring."""
    maps = []
    for src, label in ((direct, "direct"), (hist, "histGB")):
        rows = src.get("pairedRows") if isinstance(src, dict) else None
        if not isinstance(rows, list) or len(rows) != EXPECTED_SHARED: raise ValueError(f"{label} pairedRows count mismatch")
        out = {}
        for i, r in enumerate(rows):
            if not isinstance(r, dict): raise ValueError(f"{label} row {i} invalid")
            k = _key(r, label)
            if k in out: raise ValueError(f"duplicate {label} key {k}")
            out[k] = r
        maps.append(out)
    a, b = maps
    if set(a) != set(b): raise ValueError("pinned report keys do not match")
    for k in a:
        x, y = a[k], b[k]
        for field in ("actual", "direct", "persistence"):
            av = _number(x.get(field), f"direct {field}")
            bv = _number(y.get("directRidge" if field == "direct" else field), f"histGB {field}")
            if av != bv: raise ValueError(f"report mismatch for {k} field {field}")
        origin = y.get("origin")
        if origin is not None:
            try: od = dt.date.fromisoformat(origin)
            except (TypeError, ValueError) as e: raise ValueError(f"invalid origin for {k}") from e
            target = dt.date.fromisoformat(k[2])
            if (target - od).days != k[1]: raise ValueError(f"origin/horizon mismatch for {k}")
    return a, b


def _season(day: dt.date) -> str:
    return "winter" if day.month in (12, 1, 2) else "spring" if day.month in (3, 4, 5) else "summer" if day.month in (6, 7, 8) else "autumn"


def _metrics(rows: list[dict]) -> dict:
    n = len(rows)
    return {"n": n, "mae": sum(abs(r["pred"]-r["actual"]) for r in rows)/n if n else None,
            "mape": sum(abs(r["pred"]-r["actual"])/r["actual"]*100 for r in rows)/n if n else None}


def _dm(rows: list[dict], comparator: str, loss: str, lag: int) -> dict:
    n = len(rows)
    diffs = []
    for r in rows:
        a, b = abs(r["candidate"]-r["actual"]), abs(r[comparator]-r["actual"])
        if loss == "absolute_percentage_error": a, b = a/r["actual"]*100, b/r["actual"]*100
        diffs.append((dt.date.fromisoformat(r["target"]), a-b))
    mean = sum(v for _, v in diffs)/n if n else None
    base = {"n": n, "meanLossDifference": mean, "calendarHacMaxLagDays": lag}
    if n < 30: return {**base, "status": "unavailable_insufficient_pairs", "statistic": None, "pValue": 1.0}
    centered = [(d, v-mean) for d, v in diffs]; by_day = dict(centered)
    variance = sum(v*v for _, v in centered)
    for k in range(1, lag+1):
        cov = sum(v*by_day.get(day-dt.timedelta(days=k), 0.0) for day, v in centered)
        variance += 2*(1-k/(lag+1))*cov
    if variance <= 0 or not math.isfinite(variance): return {**base, "status": "unavailable_nonpositive_variance", "statistic": None, "pValue": 1.0}
    stat = mean/(math.sqrt(variance)/n)
    return {**base, "status": "available", "statistic": stat, "pValue": math.erfc(abs(stat)/math.sqrt(2))}


def _holm(tests: list[dict]) -> None:
    ordered = sorted(tests, key=lambda t: (t["pValue"], t["product"], t["horizon"], t["comparator"], t["loss"]))
    running = 0.0; m = len(ordered)
    for i, t in enumerate(ordered):
        running = max(running, min(1.0, (m-i)*t["pValue"]))
        t["holmAdjustedPValue"] = running
        t["holmSignificantAt005"] = t["status"] == "available" and running < .05


def _direction(test: dict) -> str:
    difference = test["meanLossDifference"]
    if test["status"] == "available" and test["holmSignificantAt005"]:
        if difference < 0: return "improved"
        if difference > 0: return "worsened"
    return "inconclusive"


def _macro_metrics(cells: list[dict]) -> dict:
    models = ("candidate", *COMPARATORS)
    per_product = {}
    for product in PRODUCTS:
        per_product[product] = {}
        product_cells = [c for c in cells if c["product"] == product]
        for model in models:
            available = [c["metrics"][model]["mape"] for c in product_cells if c["metrics"][model]["mape"] is not None]
            per_product[product][model] = {"mape": sum(available)/len(available) if available else None,
                                           "availableHorizons": len(available), "totalHorizons": len(HORIZONS)}
    result = {}
    for model in models:
        values = [per_product[p][model]["mape"] for p in PRODUCTS if per_product[p][model]["mape"] is not None]
        result[model] = {"productEqualWeightMape": sum(values)/len(values) if values else None,
                         "availableProducts": len(values), "totalProducts": len(PRODUCTS)}
    return {"byProduct": per_product, "fiveProductEqualWeight": result}


def analyze(direct: dict, hist: dict, hashes: dict[str, str], *, now: str | None = None) -> dict:
    a, b = validate_reports(direct, hist)
    rows = []
    for k in sorted(a):
        d = dt.date.fromisoformat(k[2])
        if not START <= d <= END: continue
        x, y = a[k], b[k]
        actual = _number(x["actual"], "actual", positive=True)
        ridge = _number(x["direct"], "direct Ridge", positive=True)
        persistence = _number(x["persistence"], "persistence", positive=True)
        raw_hist = y.get("fullHistoryHistGb")
        try: hg = _number(raw_hist, "full-history HistGB", positive=True)
        except ValueError: hg = None
        pred, fallback = ((ridge + hg)/2, False) if hg is not None else (ridge, True)
        rows.append({"product": k[0], "horizon": k[1], "target": k[2], "origin": y.get("origin"), "actual": actual,
                     "candidate": pred, "direct_ridge": ridge, "full_history_histgb": hg, "persistence": persistence,
                     "fallback": fallback, "fallbackReason": None if not fallback else "unavailable_or_nonpositive_histgb"})
    if len(rows) != EXPECTED_EVALUATION: raise ValueError(f"evaluation pair count mismatch: expected {EXPECTED_EVALUATION}, found {len(rows)}")
    grouped: dict[tuple[str,int], list[dict]] = defaultdict(list)
    for r in rows: grouped[(r["product"],r["horizon"])].append(r)
    tests_by_lag = {lag: [] for lag in LAGS}; cells=[]
    for product in PRODUCTS:
        for h in HORIZONS:
            cell = grouped[(product,h)]
            metrics = {name: _metrics([{"pred": r["candidate" if name=="candidate" else name], "actual": r["actual"]} for r in cell if name != "full_history_histgb" or r[name] is not None]) for name in ("candidate",*COMPARATORS)}
            # Paired comparisons use only eligible candidate+comparator rows.
            comparisons = {}
            for comp in COMPARATORS:
                paired = [r for r in cell if comp != "full_history_histgb" or r[comp] is not None]
                comparisons[comp] = _metrics([{"pred":r["candidate"],"actual":r["actual"]} for r in paired])
                comparisons[comp]["comparator"] = _metrics([{"pred":r[comp],"actual":r["actual"]} for r in paired])
                comparisons[comp]["meanLossDifference"] = {m: (comparisons[comp][m]-comparisons[comp]["comparator"][m] if comparisons[comp][m] is not None else None) for m in ("mae", "mape")}
                for loss in LOSSES:
                    for lag in LAGS:
                        t = {"product":product,"horizon":h,"comparator":comp,"loss":loss,**_dm(paired,comp,loss,lag)}
                        tests_by_lag[lag].append(t)
            by_year={}; by_season={}
            for key, selector in ((by_year,lambda r:str(r["target"][:4])),(by_season,lambda r:_season(dt.date.fromisoformat(r["target"])) )):
                buckets=defaultdict(list)
                for r in cell: buckets[selector(r)].append(r)
                for name, bucket in buckets.items(): key[name]={"n":len(bucket),"fallbackCount":sum(r["fallback"] for r in bucket),"metrics":{m:_metrics([{"pred":r["candidate"] if m=="candidate" else r[m],"actual":r["actual"]} for r in bucket if m!="full_history_histgb" or r[m] is not None]) for m in ("candidate",*COMPARATORS)}}
            cells.append({"product":product,"horizon":h,"n":len(cell),"fallbackCount":sum(r["fallback"] for r in cell),"metrics":metrics,"comparisons":comparisons,"byYear":by_year,"bySeason":by_season})
    for tests in tests_by_lag.values(): _holm(tests)
    for test in tests_by_lag[9]: test["directionalConclusion"] = _direction(test)
    # Strict joint improvement requires all six directions; any significant worsening is retained as adverse evidence.
    for cell in cells:
        relevant=[t for t in tests_by_lag[9] if t["product"]==cell["product"] and t["horizon"]==cell["horizon"]]
        improved=all(t["directionalConclusion"]=="improved" for t in relevant)
        worsened=all(t["directionalConclusion"]=="worsened" for t in relevant)
        adverse=any(t["directionalConclusion"]=="worsened" for t in relevant)
        coverage_ok=all(cell["n"] >= cell["comparisons"][comp]["comparator"]["n"] for comp in COMPARATORS)
        cell["conclusionHAC9"]="improved" if improved and cell["n"]>=30 and coverage_ok else "worsened" if worsened and cell["n"]>=30 else "mixed_or_adverse_evidence" if adverse else "inconclusive"
        cell["coverageNondecrease"] = coverage_ok
        cell["testConclusionsHAC9"] = {f'{t["comparator"]}/{t["loss"]}':t["directionalConclusion"] for t in relevant}
    macro = _macro_metrics(cells)
    return {"schemaVersion":1,"status":"complete","createdAt":now or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "protocol":{**FROZEN_PROTOCOL,"directReportSha256":hashes["direct"],"histGbReportSha256":hashes["histgb"],"fallbackCount":sum(r["fallback"] for r in rows)},
            "coverage":{"n":len(rows),"byYear":{str(y):sum(r["target"].startswith(str(y)) for r in rows) for y in range(2025,2027)},"fallbackCount":sum(r["fallback"] for r in rows)},
            "macro":macro,"cells":cells,"pairedRows":rows,
            "multipleTesting":{"HAC9":tests_by_lag[9],"HAC28":tests_by_lag[28]}}


def _immutable_json(path: Path, payload: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw=(json.dumps(payload,ensure_ascii=False,allow_nan=False,sort_keys=True,separators=(",",":"))+"\n").encode()
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,"wb") as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    return hashlib.sha256(raw).hexdigest()


def run(direct_path: Path, hist_path: Path, manifest_path: Path, output_path: Path, *, execute: bool=False) -> dict:
    raw_a=direct_path.read_bytes(); raw_b=hist_path.read_bytes(); ha=hashlib.sha256(raw_a).hexdigest(); hb=hashlib.sha256(raw_b).hexdigest()
    if ha!=DIRECT_SHA256 or hb!=HISTGB_SHA256: raise ValueError("pinned report SHA-256 mismatch")
    direct=json.loads(raw_a); hist=json.loads(raw_b); validate_reports(direct,hist)
    manifest={"schemaVersion":1,"status":"frozen_before_scoring","createdAt":dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),"inputSha256":{"direct":ha,"histgb":hb},"protocol":{**FROZEN_PROTOCOL,"directReportSha256":ha,"histGbReportSha256":hb}}
    manifest_hash=_immutable_json(manifest_path,manifest)
    if not execute: return {"status":"dryrun","manifest":str(manifest_path),"manifestSha256":manifest_hash}
    report=analyze(direct,hist,{"direct":ha,"histgb":hb})
    report["manifestSha256"]=manifest_hash
    report_hash=_immutable_json(output_path,report)
    return {"status":"complete","manifest":str(manifest_path),"manifestSha256":manifest_hash,"report":str(output_path),"reportSha256":report_hash}


def main(argv=None) -> int:
    p=argparse.ArgumentParser(); root=Path(__file__).resolve().parents[1]
    p.add_argument("--direct",type=Path,default=root/"outputs/experiments/direct-horizon-20261006/report.json")
    p.add_argument("--histgb",type=Path,default=root/"outputs/experiments/recent-window-20261006/report.json")
    p.add_argument("--manifest",type=Path,default=root/"outputs/experiments/simple-ensemble-20261007/manifest.json")
    p.add_argument("--output",type=Path,default=root/"outputs/experiments/simple-ensemble-20261007/report.json")
    p.add_argument("--execute",action="store_true",help="write immutable scored report; default only freezes manifest")
    args=p.parse_args(argv)
    print(json.dumps(run(args.direct,args.histgb,args.manifest,args.output,execute=args.execute),ensure_ascii=False,indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
