"""Frozen retrospective median log-bias correction for three crops."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import dashboard, direct_horizon as dh

ROOT = Path(__file__).resolve().parents[1]
INPUT_REPORT = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
REPORT_SHA256 = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
START, END = dt.date(2025, 9, 25), dt.date(2026, 9, 24)
HORIZONS = tuple(range(1, 11))
CORRECTED = {"cabbage", "radish", "pepper"}
MIN_HISTORY, LAG, WINDOW, CAP, SHRINK = 15, 14, 90, .08, .5
COMPARATORS, LOSSES, BANDWIDTHS = ("directRidge", "persistence"), ("mae", "mape"), (9, 28)
FAMILY_SIZE = len(dashboard.PRODUCTS) * 10 * len(COMPARATORS) * len(LOSSES)
EXPECTED_PAIRS = 12_508


def _pos(x: Any, label: str) -> float:
    if isinstance(x, bool): raise ValueError(f"{label} must be positive and finite")
    try: value = float(x)
    except (TypeError, ValueError, OverflowError) as e: raise ValueError(f"{label} must be positive and finite") from e
    if not math.isfinite(value) or value <= 0: raise ValueError(f"{label} must be positive and finite")
    return value


def rows(report: dict[str, Any]) -> dict[tuple[str, dt.date, dt.date, int], dict[str, float]]:
    if report.get("schemaVersion", 1) != 1 or report.get("protocol", {}).get("id") != "direct-horizon-ridge-2026-10-06":
        raise ValueError("unexpected direct-horizon report schema or protocol")
    out = {}
    for i, r in enumerate(report.get("pairedRows", [])):
        if not isinstance(r, dict) or r.get("_summary"): continue
        p, h = r.get("product"), r.get("horizon")
        if p not in dashboard.PRODUCTS or isinstance(h, bool) or h not in HORIZONS: raise ValueError(f"invalid product/horizon at pairedRows[{i}]")
        try:
            target = dt.date.fromisoformat(r["target"])
            origin = dt.date.fromisoformat(r["origin"]) if r.get("origin") is not None else target-dt.timedelta(days=h)
        except (KeyError, TypeError, ValueError) as e: raise ValueError(f"invalid date at pairedRows[{i}]") from e
        if target - origin != dt.timedelta(days=h): raise ValueError(f"origin/horizon mismatch at pairedRows[{i}]")
        key = (p, origin, target, h)
        if key in out: raise ValueError(f"duplicate direct-report identity {key}")
        out[key] = {"actual": _pos(r.get("actual"), "actual"), "directRidge": _pos(r.get("direct"), "direct Ridge"),
                    "persistence": _pos(r.get("persistence"), "persistence")}
    if not out: raise ValueError("direct report contains no pairedRows")
    return out


def _correct(hist: list[tuple[dt.date, float, float]], origin: dt.date, direct: float):
    eligible = [(t, a, p) for t, a, p in hist if origin-dt.timedelta(days=WINDOW+LAG-1) <= t <= origin-dt.timedelta(days=LAG)]
    targets = {t for t, _, _ in eligible}
    if len(targets) != len(eligible): raise ValueError("duplicate historical target")
    logs = [math.log(a/p) for _, a, p in eligible if a > 0 and p > 0 and math.isfinite(a) and math.isfinite(p)]
    if len(logs) < MIN_HISTORY: return direct, {"applied": False, "reason": "insufficient_history", "distinctTargets": len(logs), "historyStart": None, "historyEnd": None, "rawMedian": None, "clampedMedian": None}
    med = sorted(logs)[len(logs)//2] if len(logs)%2 else (sorted(logs)[len(logs)//2-1]+sorted(logs)[len(logs)//2])/2
    c = max(-CAP, min(CAP, med)); candidate = direct * math.exp(SHRINK*c)
    if not math.isfinite(candidate) or candidate <= 0:
        return direct, {"applied": False, "reason": "invalid_candidate", "distinctTargets": len(logs), "historyStart": min(t for t,_,_ in eligible).isoformat(), "historyEnd": max(t for t,_,_ in eligible).isoformat(), "rawMedian": med, "clampedMedian": c}
    return candidate, {"applied": True, "reason": None, "distinctTargets": len(logs), "historyStart": min(t for t,_,_ in eligible).isoformat(), "historyEnd": max(t for t,_,_ in eligible).isoformat(), "rawMedian": med, "clampedMedian": c}


def build(report: dict[str, Any], *, expected_pairs: int | None = EXPECTED_PAIRS) -> dict[str, Any]:
    direct = rows(report)
    history: dict[tuple[str,int], list[tuple[dt.date,float,float]]] = defaultdict(list)
    for (p, o, t, h), v in direct.items(): history[(p,h)].append((t,v["actual"],v["directRidge"]))
    pairs, correction_audit = [], []
    for (p,o,t,h),v in sorted(direct.items(), key=lambda x:(x[0][2],x[0][0],x[0][3])):
        if not START <= t <= END: continue
        if p in CORRECTED:
            pred, audit = _correct(history[(p,h)], o, v["directRidge"])
        else:
            pred, audit = v["directRidge"], {"applied":False,"reason":"unchanged_crop","distinctTargets":0,"historyStart":None,"historyEnd":None,"rawMedian":None,"clampedMedian":None}
        entry={"product":p,"origin":o.isoformat(),"target":t.isoformat(),"horizon":h,"actual":v["actual"],"candidate":pred,"directRidge":v["directRidge"],"persistence":v["persistence"],"correction":audit}
        pairs.append(entry); correction_audit.append({k:entry[k] for k in ("product","origin","target","horizon","correction")})
    comparisons=[]
    for p in dashboard.PRODUCTS:
      for h in HORIZONS:
       subset=[r for r in pairs if r["product"]==p and r["horizon"]==h]
       for comp in COMPARATORS:
        for loss in LOSSES:
         def value(r, pred):
          d=abs(r["actual"]-r[pred]); return d if loss=="mae" else 100*d/r["actual"]
         a=[r["actual"] for r in subset]
         cm=dh._metrics(a,[r["candidate"] for r in subset])[loss]
         bm=dh._metrics(a,[r[comp] for r in subset])[loss]
         diff={dt.date.fromisoformat(r["target"]):value(r,"candidate")-value(r,comp) for r in subset}
         tests={str(bw):dh._dm(diff,bw) for bw in BANDWIDTHS}
         comparisons.append({"product":p,"horizon":h,"comparator":comp,"loss":loss,"n":len(subset),"candidateMetric":cm,"comparatorMetric":bm,"dm":tests})
    for bw in BANDWIDTHS:
        family=[r["dm"][str(bw)] for r in comparisons]; dh._holm(family)
    conclusions=[]
    by_test={(r["product"],r["horizon"],r["comparator"],r["loss"]):r for r in comparisons}
    for p in dashboard.PRODUCTS:
      for h in HORIZONS:
       req=[by_test[(p,h,c,l)] for c in COMPARATORS for l in LOSSES]
       tests=[r["dm"]["9"] for r in req]
       inference=all(t["status"]=="tested" and t["n"]>=30 for t in tests)
       lower=all(r["candidateMetric"] is not None and r["comparatorMetric"] is not None and r["candidateMetric"]<r["comparatorMetric"] for r in req)
       sig=all(t.get("holmSignificant",False) and t.get("meanLossDifference") is not None and t["meanLossDifference"]<0 for t in tests)
       coverage_pass=True  # Same frozen pairedRows are used for each candidate/control.
       adverse=any((r["candidateMetric"] is not None and r["comparatorMetric"] is not None and r["candidateMetric"]>r["comparatorMetric"]) or (t["status"]=="tested" and t.get("holmSignificant",False) and t.get("meanLossDifference") is not None and t["meanLossDifference"]>0) for r,t in zip(req,tests))
       conclusion="inference_unavailable" if not inference else "improvement" if lower and sig and coverage_pass else "worsening_or_mixed" if adverse else "inconclusive"
       conclusions.append({"product":p,"horizon":h,"conclusion":conclusion,"inferenceAvailable":inference,"allFourHac9HolmTestsImprove":sig,"allFourPairedMetricsImprove":lower,"coverageNondeclining":coverage_pass,"requiredTests":[{"comparator":r["comparator"],"loss":r["loss"],"hac9":r["dm"]["9"]} for r in req]})
    coverage=[]
    for p in dashboard.PRODUCTS:
      for h in HORIZONS:
       s=[r for r in pairs if r["product"]==p and r["horizon"]==h]
       coverage.append({"product":p,"horizon":h,"pairs":len(s),"corrected":sum(r["correction"]["applied"] for r in s),"fallback":sum(not r["correction"]["applied"] for r in s),"candidateDates":[r["target"] for r in s],"directDates":[r["target"] for r in s]})
    macro=[]
    for p in dashboard.PRODUCTS:
        gs=[r for r in comparisons if r["product"]==p and r["comparator"]=="directRidge" and r["loss"]=="mape"]
        vals=[r["candidateMetric"] for r in gs]
        macro.append({"product":p,"equalHorizonCandidateMAPE":sum(vals)/10 if len(vals)==10 and all(v is not None for v in vals) else None})
    strata=[]
    for grouping, keyfn in (("year",lambda d:str(d.year)),("season",lambda d: ("DJF","MAM","JJA","SON")[(d.month%12)//3]),("month",lambda d:d.strftime("%Y-%m"))):
        for label in sorted({keyfn(dt.date.fromisoformat(r["target"])) for r in pairs}):
            subset=[r for r in pairs if keyfn(dt.date.fromisoformat(r["target"]))==label]
            for p in dashboard.PRODUCTS:
                for h in HORIZONS:
                    group=[r for r in subset if r["product"]==p and r["horizon"]==h]
                    if group:
                        actual=[r["actual"] for r in group]
                        strata.append({"grouping":grouping,"period":label,"product":p,"horizon":h,"n":len(group),"candidate":dh._metrics(actual,[r["candidate"] for r in group]),"directRidge":dh._metrics(actual,[r["directRidge"] for r in group]),"persistence":dh._metrics(actual,[r["persistence"] for r in group])})
    if expected_pairs is not None and len(pairs)!=expected_pairs: raise ValueError(f"evaluation pairedRows count {len(pairs)} != frozen {expected_pairs}")
    return {"schemaVersion":1,"status":"complete","protocol":{"id":"leaf-root-median-log-bias-2026-10-07","evaluation":[START.isoformat(),END.isoformat()],"expectedPairedRows":expected_pairs,"history":"same product+horizon targets in [origin-103d, origin-14d] inclusive","minimumDistinctTargets":MIN_HISTORY,"windowDays":WINDOW,"lagDays":LAG,"cap":CAP,"shrink":SHRINK,"correctedProducts":sorted(CORRECTED),"unchangedProducts":["onion","garlic"],"dm":{"familySize":FAMILY_SIZE,"bandwidthDays":list(BANDWIDTHS),"holmAlpha":.05}},"hashes":{"directReportSha256":REPORT_SHA256},"pairedRows":pairs,"correctionAudit":correction_audit,"coverage":coverage,"comparisons":comparisons,"jointConclusions":conclusions,"strata":strata,"limitations":"Retrospective evaluation uses revised prices and cannot establish historical first publication time. Previous inspection of the evaluation period informed protocol selection; this is not independent validation or prospective evidence. DM inference uses paired actual target dates with calendar Bartlett HAC9/HAC28; inference is unavailable for groups with fewer than 30 pairs or invalid HAC variance. Statistical non-significance does not establish equivalence.","macroMetrics":{"products":macro,"equalProductMAPE":sum(x["equalHorizonCandidateMAPE"] for x in macro)/5 if all(x["equalHorizonCandidateMAPE"] is not None for x in macro) else None}}


def run(input_path: Path = INPUT_REPORT, output_path: Path | None = None):
    digest=hashlib.sha256(input_path.read_bytes()).hexdigest()
    if digest != REPORT_SHA256: raise ValueError(f"direct report SHA-256 mismatch: {digest}")
    report=json.loads(input_path.read_text(encoding="utf-8"))
    result=build(report); payload=(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    if output_path:
        output_path.parent.mkdir(parents=True,exist_ok=True); output_path.write_bytes(payload)
    return {"output":str(output_path) if output_path else None,"sha256":hashlib.sha256(payload).hexdigest(),"pairedRows":len(result["pairedRows"])}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,default=INPUT_REPORT); ap.add_argument("--output",type=Path,default=ROOT/"outputs/experiments/leaf-root-bias-20261007/report.json"); a=ap.parse_args()
    print(json.dumps(run(a.input,a.output),ensure_ascii=False))

if __name__=="__main__": main()
