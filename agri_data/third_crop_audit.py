"""Frozen audit of existing sequential forecast methods for a third sub-10% crop."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from . import dashboard, direct_horizon as dh

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIRECT = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
INPUT_RECENT = ROOT / "outputs/experiments/recent-window-20261006/report.json"
INPUT_RECOMPOSE = ROOT / "outputs/experiments/market-recompose-20261007/report.json"
HASHES = {"directHorizonSha256": "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67",
          "recentWindowSha256": "1032caa80628cbab54c38d9efe408339348ca9e6341b0df4e3306df64a606a1e",
          "marketRecomposeSha256": "e5f23f1012dad2acc774c83f995e2ee2851409af913ee82483aa3d953ee87581"}
PRODUCTS = tuple(dashboard.PRODUCTS)
HORIZONS = tuple(range(1, 11))
METHODS = ("monthlyPooledRidge", "dailyPooledRidge", "recentWindowHistGb", "fullHistoryHistGb", "monthlyPooledHistGb", "marketRecompose")
METHOD_FIELDS = {"monthlyPooledRidge": "monthlyPooledRidge", "dailyPooledRidge": "dailyPooledRidge",
                 "recentWindowHistGb": "recentWindowHistGb", "fullHistoryHistGb": "fullHistoryHistGb",
                 "monthlyPooledHistGb": "monthlyPooledHistGb", "marketRecompose": "candidatePrediction"}
COMPARATORS = ("directRidge", "persistence")
LOSSES = ("mae", "mape")
BANDWIDTHS = (9, 28)
MIN_PAIRS = 30
ALPHA = .05
FAMILY_SIZE = len(PRODUCTS)*len(HORIZONS)*len(COMPARATORS)*len(LOSSES)
EXPECTED_PAIRED_ROWS = 45_880
SELECTION_START, SELECTION_END = dt.date(2024, 9, 25), dt.date(2025, 9, 1)
TEST_START, TEST_END = dt.date(2025, 9, 25), dt.date(2026, 9, 24)
SOURCE_START = dt.date(2023, 1, 1)
FIRST_TEST_ORIGIN = TEST_START-dt.timedelta(days=max(HORIZONS))
SELECTION_EMBARGO = 14
if SELECTION_END > FIRST_TEST_ORIGIN-dt.timedelta(days=SELECTION_EMBARGO):
    raise AssertionError("selection window leaks outcomes unavailable at earliest test origin")


def _day(value, label):
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        result = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc
    if result.isoformat() != value:
        raise ValueError(f"{label} must be canonical YYYY-MM-DD")
    return result


def _positive(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be positive and finite")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be positive and finite") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def _optional_positive(value):
    if value is None or isinstance(value, bool):
        return None
    try: result = float(value)
    except (TypeError, ValueError, OverflowError): return None
    return result if math.isfinite(result) and result > 0 else None


def _index_report(report: dict[str, Any], source: str):
    if not isinstance(report, dict) or report.get("schemaVersion", 1) != 1:
        raise ValueError(f"{source} must be schemaVersion 1")
    protocol_id = {"direct": "direct-horizon-ridge-2026-10-06",
                   "recent": "recent-730-day-direct-hist-gb-2026-10-06",
                   "market": "market-wise-direct-ridge-exante-share-2026-10-07"}[source]
    if report.get("protocol", {}).get("id") != protocol_id:
        raise ValueError(f"unexpected {source} protocol")
    if source == "market" and report.get("status") != "complete":
        raise ValueError("market recomposition report is not complete")
    rows = report.get("pairedRows")
    if not isinstance(rows, list):
        raise ValueError(f"{source} report lacks pairedRows")
    index = {}
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("_summary"):
            raise ValueError(f"unexpected summary or malformed row in {source} pairedRows[{i}]")
        product = row.get("product")
        if product not in PRODUCTS:
            raise ValueError(f"unknown product in {source}: {product!r}")
        target = _day(row.get("target"), f"{source} target")
        horizon = row.get("horizon")
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon not in HORIZONS:
            raise ValueError(f"invalid {source} horizon")
        origin = _day(row.get("origin"), f"{source} origin") if "origin" in row else target-dt.timedelta(days=horizon)
        if origin != target-dt.timedelta(days=horizon):
            raise ValueError(f"{source} origin/horizon mismatch for {product}/{target}/h{horizon}")
        if target > TEST_END:
            raise ValueError(f"{source} target exceeds frozen test end")
        if target < SOURCE_START:
            raise ValueError(f"{source} target precedes frozen source start")
        key = (product, origin, target, horizon)
        if key in index:
            raise ValueError(f"duplicate {source} identity {key}")
        actual = _positive(row.get("actual"), f"{source} actual {key}")
        ridge_field = "direct" if source == "direct" else "directRidge"
        ridge = _positive(row.get(ridge_field), f"{source} direct Ridge {key}")
        persistence = _positive(row.get("persistence"), f"{source} persistence {key}")
        values = {"actual": actual, "directRidge": ridge, "persistence": persistence}
        if source == "direct":
            for method in ("monthlyPooledRidge", "dailyPooledRidge"):
                values[method] = _optional_positive(row.get(method))
        elif source == "recent":
            for method in ("recentWindowHistGb", "fullHistoryHistGb", "monthlyPooledHistGb"):
                values[method] = _optional_positive(row.get(method))
        else:
            values["marketRecompose"] = _optional_positive(row.get("candidatePrediction"))
        index[key] = values
    if {(key[0], key[3]) for key in index} != {(p, h) for p in PRODUCTS for h in HORIZONS}:
        raise ValueError(f"{source} pairedRows must cover all five products and ten horizons")
    return index


def _align(direct, recent, market, expected_rows):
    if expected_rows is not None:
        for label, index in (("direct", direct), ("recent", recent), ("market", market)):
            if len(index) != expected_rows:
                raise ValueError(f"{label} pairedRows count {len(index)} differs from frozen {expected_rows}")
    key_sets = {"direct": set(direct), "recent": set(recent), "market": set(market)}
    if not (key_sets["direct"] == key_sets["recent"] == key_sets["market"]):
        differences = {label: len(key_sets[label] ^ key_sets["direct"]) for label in ("recent", "market")}
        raise ValueError(f"paired-row identity mismatch across reports; symmetric differences={differences}")
    rows, common, invalid = [], [], []
    for key in sorted(key_sets["direct"]):
        source_rows = (direct[key], recent[key], market[key])
        core = source_rows[0]
        for field in ("actual", "directRidge", "persistence"):
            value = core[field]
            for label, other in zip(("recent", "market"), source_rows[1:]):
                if value != other[field]:
                    raise ValueError(f"{field} mismatch for {key} between direct and {label}")
        row = {"product": key[0], "origin": key[1].isoformat(), "target": key[2].isoformat(), "horizon": key[3], **core}
        row.update({method: source_rows[0].get(method) for method in ("monthlyPooledRidge", "dailyPooledRidge")})
        row.update({method: source_rows[1][method] for method in ("recentWindowHistGb", "fullHistoryHistGb", "monthlyPooledHistGb")})
        row["marketRecompose"] = source_rows[2]["marketRecompose"]
        missing = [name for name in METHODS if row[name] is None]
        if missing:
            invalid.append({"product":key[0],"origin":key[1].isoformat(),"target":key[2].isoformat(),"horizon":key[3],"missingOrInvalidMethods":missing})
        rows.append(row)
        if not missing:
            common.append(row)
    return rows, common, invalid


def _coverage(rows, product, horizon, start, end):
    subset = [r for r in rows if r["product"] == product and r["horizon"] == horizon and start <= dt.date.fromisoformat(r["target"]) <= end]
    controls = {c: {r["target"] for r in subset if r[c] is not None} for c in COMPARATORS}
    method_dates = {m: {r["target"] for r in subset if r[m] is not None} for m in METHODS}
    common_dates = set.intersection(*(method_dates[m] for m in METHODS), controls["directRidge"], controls["persistence"]) if subset else set()
    losses = {m: {c: sorted(controls[c]-method_dates[m]) for c in COMPARATORS} for m in METHODS}
    return {"eligibleAlignedRowCount": len(subset), "commonPairedCount": len(common_dates), "pairedDates": sorted(common_dates),
            "methodCoverage": {m: len(method_dates[m]) for m in METHODS},
            "directRidgeCoverage": len(controls["directRidge"]), "persistenceCoverage": len(controls["persistence"]),
            "coverageLossDates": losses,
            "coverageNondeclining": {m: all(not v for v in losses[m].values()) for m in METHODS},
            "coverageDenominator": "aligned source pairedRows within frozen reports; not all possible target origins"}


def _macro(rows, product, method):
    per_h = {}
    for horizon in HORIZONS:
        group = [r for r in rows if r["product"] == product and r["horizon"] == horizon]
        per_h[str(horizon)] = {"n":len(group), **dh._metrics([r["actual"] for r in group],[r[method] for r in group])}
    mapes=[v["mape"] for v in per_h.values()]; maes=[v["mae"] for v in per_h.values()]
    return {"mapeByHorizon":per_h,"equal10HorizonMape":sum(mapes)/10 if all(v is not None for v in mapes) else None,
            "equal10HorizonMae":sum(maes)/10 if all(v is not None for v in maes) else None,
            "pairedCountByHorizon":{h:v["n"] for h,v in per_h.items()},"pointEstimateAvailable":all(v is not None for v in mapes)}


def _select(rows, full_rows, start, end):
    selection_rows=[r for r in rows if start <= dt.date.fromisoformat(r["target"]) <= end]
    decisions={}
    for product in PRODUCTS:
        metrics={m:_macro(selection_rows,product,m) for m in ("directRidge",*METHODS)}
        ridge=metrics["directRidge"]
        coverage={m:{h:_coverage(full_rows,product,h,start,end) for h in HORIZONS} for m in METHODS}
        eligible=[]
        for order,method in enumerate(METHODS):
            m=metrics[method]
            cov_ok=all(coverage[method][h]["coverageNondeclining"][method] for h in HORIZONS)
            if (m["equal10HorizonMape"] is not None and ridge["equal10HorizonMape"] is not None and
                m["equal10HorizonMae"] is not None and ridge["equal10HorizonMae"] is not None and
                cov_ok and m["equal10HorizonMape"] <= ridge["equal10HorizonMape"]-.25 and
                m["equal10HorizonMae"] <= ridge["equal10HorizonMae"]*1.01):
                eligible.append((m["equal10HorizonMape"],m["equal10HorizonMae"],order,method))
        chosen=min(eligible)[3] if eligible else "directRidge"
        decisions[product]={"selected":chosen,"eligibleCandidates":[x[3] for x in eligible],"metrics":metrics,
            "selectionCoverage":{m:{str(h):coverage[m][h]["coverageNondeclining"][m] for h in HORIZONS} for m in METHODS},
            "rule":{"minimumMapeImprovementPercentagePoints":.25,"maximumMaeRegressionFraction":.01,
                    "tieBreak":"MAPE, MAE, then frozen candidate order"}}
    return decisions,selection_rows


def _season(day):
    if day.month in (12,1,2): return "DJF"
    if day.month in (3,4,5): return "MAM"
    if day.month in (6,7,8): return "JJA"
    return "SON"


def _test(rows, full_rows, selection):
    test_rows=[r for r in rows if TEST_START <= dt.date.fromisoformat(r["target"]) <= TEST_END]
    paired=[]; comparisons=[]; cover={}; strata=[]
    for p in PRODUCTS:
        method=selection[p]["selected"]
        for h in HORIZONS:
            group=[r for r in test_rows if r["product"]==p and r["horizon"]==h]
            paired.extend([{**r,"selectedMethod":method,"selectedPrediction":r[method]} for r in group])
            cov=_coverage(full_rows,p,h,TEST_START,TEST_END)
            cover[f"{p}/{h}"]={**cov,"selectedMethod":method,"selectedCoverageNondeclining":cov["coverageNondeclining"][method] if method in cov["coverageNondeclining"] else True}
            for comparator in COMPARATORS:
                for loss in LOSSES:
                    diff={dt.date.fromisoformat(r["target"]):(abs(r["actual"]-r[method])-abs(r["actual"]-r[comparator])) if loss=="mae" else 100*(abs(r["actual"]-r[method])/r["actual"]-abs(r["actual"]-r[comparator])/r["actual"]) for r in group}
                    comparisons.append({"product":p,"horizon":h,"comparator":comparator,"loss":loss,"n":len(group),
                        "candidateMetric":dh._metrics([r["actual"] for r in group],[r[method] for r in group])[loss],
                        "comparatorMetric":dh._metrics([r["actual"] for r in group],[r[comparator] for r in group])[loss],
                        "dm":{str(bw):dh._dm(diff,bw) for bw in BANDWIDTHS}})
            for label,fn in (("year",lambda d:str(d.year)),("season",_season)):
                buckets={}
                for row in group:
                    day=dt.date.fromisoformat(row["target"]); buckets.setdefault(fn(day),[]).append(row)
                all_buckets={}
                for row in full_rows:
                    day=dt.date.fromisoformat(row["target"])
                    if row["product"]==p and row["horizon"]==h and TEST_START<=day<=TEST_END:
                        all_buckets.setdefault(fn(day),[]).append(row)
                for value,eligible in sorted(all_buckets.items()):
                    subset=buckets.get(value,[])
                    comparator_dates={c:{r["target"] for r in eligible if r[c] is not None} for c in COMPARATORS}
                    selected_dates={r["target"] for r in eligible if r[method] is not None}
                    strata.append({"product":p,"horizon":h,"stratum":label,"value":value,"n":len(subset),
                        "eligibleAlignedRows":len(eligible),"selectedCoverage":len(selected_dates),
                        "selectedCoverageLossDates":{c:sorted(comparator_dates[c]-selected_dates) for c in COMPARATORS},
                        "coverageNondeclining":all(not (comparator_dates[c]-selected_dates) for c in COMPARATORS),
                        "selectedMethod":method,"selected":dh._metrics([r["actual"] for r in subset],[r[method] for r in subset]),
                        "directRidge":dh._metrics([r["actual"] for r in subset],[r["directRidge"] for r in subset]),
                        "persistence":dh._metrics([r["actual"] for r in subset],[r["persistence"] for r in subset])})
    if len(comparisons)!=FAMILY_SIZE: raise AssertionError("frozen third-crop family must contain 200 tests")
    for bw in BANDWIDTHS: dh._holm([r["dm"][str(bw)] for r in comparisons])
    conclusions=[]
    indexed={(r["product"],r["horizon"],r["comparator"],r["loss"]):r for r in comparisons}
    for p in PRODUCTS:
        for h in HORIZONS:
            required=[indexed[(p,h,c,l)] for c in COMPARATORS for l in LOSSES]
            tests=[r["dm"]["9"] for r in required]
            enough=all(t["status"]=="tested" and t["n"]>=MIN_PAIRS for t in tests)
            lower=all(r["candidateMetric"] is not None and r["comparatorMetric"] is not None and r["candidateMetric"]<r["comparatorMetric"] for r in required)
            significant=all(t.get("holmSignificant") and t.get("meanLossDifference") is not None and t["meanLossDifference"]<0 for t in tests)
            cov_ok=cover[f"{p}/{h}"]["selectedCoverageNondeclining"]
            adverse=any(t.get("holmSignificant") and t.get("meanLossDifference") is not None and t["meanLossDifference"]>0 for t in tests)
            status="inference_unavailable" if not enough else "improvement" if lower and significant and cov_ok else "worsening_or_mixed" if adverse else "inconclusive"
            conclusions.append({"product":p,"horizon":h,"conclusion":status,"inferenceAvailable":enough,
                "allFourPairedMetricsImprove":lower,"allFourHac9HolmTestsImprove":significant,"coverageNondeclining":cov_ok})
    macro={p:{name:_macro(paired,p,name) for name in ("selectedPrediction","directRidge","persistence")} for p in PRODUCTS}
    return {"pairedRows":paired,"coverage":cover,"comparisons":comparisons,"conclusions":conclusions,
            "macroMetrics":macro,"yearSeasonMetrics":strata,
            "goal":{"thresholdPercentLessThan10":10.0,"cropsBelow10Percent":sum(v["selectedPrediction"]["equal10HorizonMape"] is not None and v["selectedPrediction"]["equal10HorizonMape"]<10.0 for v in macro.values()),
                    "requiredCrops":3}}


def build(direct_report, recent_report, market_report, *, hashes=None, expected_paired_rows=EXPECTED_PAIRED_ROWS):
    direct=_index_report(direct_report,"direct"); recent=_index_report(recent_report,"recent"); market=_index_report(market_report,"market")
    all_rows, common_rows, invalid=_align(direct,recent,market,expected_paired_rows)
    complete, selection_rows=_select(common_rows,all_rows,SELECTION_START,SELECTION_END)
    test=_test(common_rows,all_rows,complete)
    return {"schemaVersion":1,"protocol":{"id":"existing-sequential-third-crop-audit-2026-10-07",
        "inputHashes":HASHES,"expectedSourcePairedRows":EXPECTED_PAIRED_ROWS,"products":list(PRODUCTS),"horizons":list(HORIZONS),
        "selectionTargetInterval":[SELECTION_START.isoformat(),SELECTION_END.isoformat()],
        "testTargetInterval":[TEST_START.isoformat(),TEST_END.isoformat()],"earliestTestOrigin":FIRST_TEST_ORIGIN.isoformat(),
        "selectionEmbargoDays":SELECTION_EMBARGO,"methods":list(METHODS),"comparators":list(COMPARATORS),
        "selectionRule":{"mapeImprovementPercentagePoints":.25,"maxMaeRegressionFraction":.01,"fallback":"direct Ridge"},
        "DM":{"familySize":FAMILY_SIZE,"twoSided":True,"calendarBartlettHACDays":list(BANDWIDTHS),"holmAlpha":ALPHA,
              "minimumPairs":MIN_PAIRS,"unavailablePValue":1.0},
        "limitations":"Retrospective revised-data analysis of previously evaluated sequential forecasts; not prospective or official nationwide performance."},
        "hashes":hashes or {},"alignment":{"sourceRowCounts":{"direct":len(direct),"recent":len(recent),"market":len(market)},
        "alignedRows":len(all_rows),"invalidRows":invalid,"allSourceIdentitiesMatched":True},
        "selection":complete,"selectionAudit":{"commonPairedCount":len(selection_rows),
        "coverageByProductHorizon":{f"{p}/{h}":_coverage(all_rows,p,h,SELECTION_START,SELECTION_END) for p in PRODUCTS for h in HORIZONS},
        "pairedRows":[r for r in selection_rows if SELECTION_START<=dt.date.fromisoformat(r["target"])<=SELECTION_END]},"test":test}


def _write_immutable(path:Path,payload:bytes):
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as stream:
            fd=None; stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.link(tmp,path)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite immutable output: {path}") from exc
    finally:
        if fd is not None: os.close(fd)
        try: os.unlink(tmp)
        except FileNotFoundError: pass


def _read(path):
    b=Path(path).read_bytes()
    try: value=json.loads(b)
    except (UnicodeDecodeError,json.JSONDecodeError) as exc: raise ValueError(f"invalid JSON input {path}") from exc
    if not isinstance(value,dict): raise ValueError(f"input must be a JSON object: {path}")
    return value,b


def run(direct_path=INPUT_DIRECT,recent_path=INPUT_RECENT,market_path=INPUT_RECOMPOSE,output_path=None):
    if output_path is None: raise ValueError("output_path is required")
    if Path(output_path).exists(): raise FileExistsError(f"refusing to overwrite immutable output: {output_path}")
    direct,db=_read(direct_path); recent,rb=_read(recent_path); market,mb=_read(market_path)
    hashes={"directHorizonSha256":hashlib.sha256(db).hexdigest(),"recentWindowSha256":hashlib.sha256(rb).hexdigest(),"marketRecomposeSha256":hashlib.sha256(mb).hexdigest()}
    if hashes!=HASHES: raise ValueError("source report SHA-256 differs from frozen protocol")
    result=build(direct,recent,market,hashes=hashes)
    payload=(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    _write_immutable(Path(output_path),payload)
    return {"output":str(output_path),"sha256":hashlib.sha256(payload).hexdigest()}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct",type=Path,default=INPUT_DIRECT); parser.add_argument("--recent",type=Path,default=INPUT_RECENT)
    parser.add_argument("--market",type=Path,default=INPUT_RECOMPOSE); parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv); run(args.direct,args.recent,args.market,args.output); return 0


if __name__=="__main__": raise SystemExit(main())
