"""Frozen relative-return Ridge evaluation using 2014-2025 training only.

Frozen before loading price values: target 2014-01-01..2025-12-31, five crops
and horizons 1..10, one training-only StandardScaler + Ridge(alpha=10) per
crop/horizon, at least 90 valid targets; fit log(actual/p0), predict p0 times
exp(predicted log return). At target t/horizon h, origin is t-h and features
use observations through origin-1 only. p0 is the latest positive observed
price no older than 14 days; p1 is its prior distinct observation within 7
days; p7/p30 are the latest observations at or before p0_date-7/-30, within
7/14 days of those anchors. Features are log(p0/p1,p7,p30), separate presence
flags and actual gaps, p0 age, target weekday and annual sin/cos. Missing return
features are zero plus missing flags; no calendar-price filling. The 2026 grid
is Jan 1-Sep 27; origins before Jan 1 are planned but not issued because the
fixed model's labels were not yet available. The answer-free manifest precedes
scoring. Scoring uses same-row positive actuals, persistence and pinned answer-
free fixed50, paired MAE/MAPE, 200 two-sided calendar-Bartlett DM tests per
HAC9/HAC28 family with Holm correction, p=1 and reasons if n<30 or HAC variance
is invalid. A crop advances only if its fixed50 10-horizon equal-weight MAPE
improves by >=0.5 percentage points, MAE improves, candidate coverage is not
lower, no horizon MAPE worsens by >1 point, and at least one horizon has joint
AE/APE Holm improvement at both bandwidths. No API or database access occurs.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import bisect
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PRICES_PATH = ROOT / "outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
FIXED50_PATH = ROOT / "outputs/experiments/five-crop-specialists-20261009/forecast-manifest.json"
PRICES_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
FIXED50_SHA256 = "1497ee362945fcab45353a7b2403be752af3bf63c5f584cc0aedb2cc111ceaa8"
PRODUCTS = ("cabbage", "radish", "onion", "garlic", "pepper")
HORIZONS = tuple(range(1, 11))
TRAIN_START, TRAIN_END = dt.date(2014, 1, 1), dt.date(2025, 12, 31)
EVAL_START, EVAL_END = dt.date(2026, 1, 1), dt.date(2026, 9, 27)
MIN_TRAIN = 90
BANDWIDTHS = (9, 28)
ALPHA = 0.05
COMPARATORS = ("persistence", "fixed50")
LOSSES = ("ae", "ape")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _date(value: Any) -> dt.date | None:
    try:
        return dt.date.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        return None


def _positive(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        x = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) and x > 0 else None


def _source_audit(raw: dict[str, Any]) -> dict[str, Any]:
    audit={}
    for product in raw.get("products",[]):
        if not isinstance(product,dict) or product.get("id") not in PRODUCTS: continue
        counts={"rows":0,"invalidDateRows":0,"nonpositiveOrInvalidActualRows":0,"duplicateUsableDates":0}
        seen=set()
        for row in product.get("observations",[]) if isinstance(product.get("observations"),list) else []:
            counts["rows"]+=1
            if not isinstance(row,dict):
                counts["invalidDateRows"]+=1; counts["nonpositiveOrInvalidActualRows"]+=1; continue
            day=_date(row.get("date")); value=_positive(row.get("actual"))
            if day is None: counts["invalidDateRows"]+=1
            if value is None: counts["nonpositiveOrInvalidActualRows"]+=1
            if day is not None and value is not None:
                if day in seen: counts["duplicateUsableDates"]+=1
                seen.add(day)
        audit[product["id"]]=counts
    return audit


def _series(raw: dict[str, Any]) -> dict[str, dict[dt.date, float]]:
    if raw.get("schemaVersion") != 1 or not isinstance(raw.get("products"), list):
        raise ValueError("prices must use dashboard schemaVersion 1")
    out: dict[str, dict[dt.date, float]] = {}
    for p in raw["products"]:
        pid = p.get("id") if isinstance(p, dict) else None
        if pid not in PRODUCTS or pid in out or not isinstance(p.get("observations"), list):
            raise ValueError("unknown, duplicate, or malformed price product")
        rows: dict[dt.date, float] = {}
        for row in p["observations"]:
            if not isinstance(row, dict):
                continue
            day, value = _date(row.get("date")), _positive(row.get("actual"))
            if day is None or value is None:
                continue
            if day in rows:
                raise ValueError(f"duplicate usable observation {pid}/{day}")
            rows[day] = value
        out[pid] = rows
    if set(out) != set(PRODUCTS):
        raise ValueError("prices must contain all five products")
    return out


def _features(series: dict[dt.date, float], target: dt.date, horizon: int, ordered: list[dt.date] | None = None) -> tuple[list[float] | None, float | None, dict[str, Any]]:
    days = ordered if ordered is not None else sorted(series)
    origin = target - dt.timedelta(days=horizon)
    end = bisect.bisect_right(days, origin - dt.timedelta(days=1))
    if not end:
        return None, None, {"reason": "no_pre_origin_observation"}
    d0 = days[end-1]; p0 = series[d0]; age = (origin-d0).days
    if age > 14:
        return None, None, {"reason":"p0_older_than_14_days","p0Date":d0.isoformat(),"p0AgeDays":age}
    def anchor(cutoff: dt.date, tolerance: int):
        idx=bisect.bisect_right(days,cutoff,0,end)-1
        if idx<0 or days[idx]>=d0 or (cutoff-days[idx]).days>tolerance: return None,None
        return days[idx],series[days[idx]]
    if end >= 2 and (d0-days[end-2]).days <= 7: d1,p1=days[end-2],series[days[end-2]]
    else: d1,p1=None,None
    d7,p7=anchor(d0-dt.timedelta(days=7),7)
    d30,p30=anchor(d0-dt.timedelta(days=30),14)
    vals=[]
    for dd,pp in ((d1,p1),(d7,p7),(d30,p30)):
        vals.extend([math.log(p0/pp) if pp else 0.0,float(pp is not None),float((d0-dd).days if dd else 0)])
    vals.extend([float(age)])
    aw=2*math.pi*target.weekday()/7; ay=2*math.pi*(target.timetuple().tm_yday-1)/(366 if _leap(target.year) else 365)
    vals.extend([math.sin(aw),math.cos(aw),math.sin(ay),math.cos(ay)])
    return vals,p0,{"p0Date":d0.isoformat(),"p0AgeDays":age}


def _leap(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _fixed50(raw: dict[str, Any]) -> dict[tuple[str, dt.date, int], float]:
    if raw.get("phase") != "forecast_only" or raw.get("targetValuesAttached") is not False or not isinstance(raw.get("forecasts"), list):
        raise ValueError("fixed50 manifest must be answer-free forecast_only")
    out = {}
    for row in raw["forecasts"]:
        if not isinstance(row, dict) or row.get("product") not in PRODUCTS:
            continue
        day, h = _date(row.get("target")), row.get("horizon")
        pred = _positive((row.get("predictions") or {}).get("fixed50"))
        if day and h in HORIZONS and pred:
            key = (row["product"], day, h)
            if key in out: raise ValueError(f"duplicate fixed50 row {key}")
            out[key] = pred
    return out


def _grid() -> list[tuple[str, dt.date, int]]:
    days = (EVAL_END - EVAL_START).days + 1
    return [(p, EVAL_START + dt.timedelta(days=i), h) for p in PRODUCTS for i in range(days) for h in HORIZONS]


def _fit_predict(X: list[list[float]], y: list[float], x: list[float]) -> float:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0, fit_intercept=True))
    model.fit(X, y)
    return float(model.predict([x])[0])


def build(prices: dict[str, Any], fixed50: dict[str, Any], *, price_sha: str | None = None, fixed50_sha: str | None = None) -> dict[str, Any]:
    series = _series(prices); benchmark = _fixed50(fixed50)
    rows = []
    grid = _grid()
    for product in PRODUCTS:
      s = series[product]; ordered = sorted(s)
      for h in HORIZONS:
        X, y = [], []; train_dates=[]
        for day in sorted(s):
            if not TRAIN_START <= day <= TRAIN_END: continue
            feat, p0, _ = _features(s, day, h, ordered)
            actual = s[day]
            if feat is None or p0 is None or not actual > 0: continue
            X.append(feat); y.append(math.log(actual/p0)); train_dates.append(day)
        estimator = None; fit_error = None
        if len(X) >= MIN_TRAIN:
            try:
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import Ridge
                estimator = make_pipeline(StandardScaler(), Ridge(alpha=10.0, fit_intercept=True)).fit(X, y)
            except Exception as exc:
                fit_error = f"fit_failed:{type(exc).__name__}"
        pending = []
        for prod, day, horizon in (k for k in grid if k[0] == product and k[2] == h):
            feat, p0, audit = _features(s, day, h, ordered)
            origin = day-dt.timedelta(days=h)
            reason = None
            if origin < EVAL_START: reason = "model_not_yet_fitted_at_origin"
            elif len(X) < MIN_TRAIN: reason = "fewer_than_90_valid_training_targets"
            elif fit_error: reason = fit_error
            elif feat is None or p0 is None: reason = audit.get("reason")
            pending.append({"product":product,"target":day.isoformat(),"origin":origin.isoformat(),"horizon":h,
                "status":"not_issued" if reason else "issued","reason":reason,"prediction":None,
                "persistence":p0 if audit.get("reason") is None else None,
                "fixed50":benchmark.get((product,day,h)),"featureAudit":audit,
                "training":{"count":len(X),"firstTarget":min(train_dates).isoformat() if train_dates else None,"lastTarget":max(train_dates).isoformat() if train_dates else None},
                "_feature":feat if reason is None else None})
        ready=[r for r in pending if r["status"]=="issued"]
        if ready:
            try:
                predictions=estimator.predict([r["_feature"] for r in ready])
                for row, log_ratio in zip(ready,predictions):
                    pred=row["persistence"]*math.exp(float(log_ratio))
                    if not math.isfinite(pred) or pred<=0:
                        row["status"]="not_issued";row["reason"]="nonpositive_or_nonfinite_prediction"
                    else: row["prediction"]=pred
            except Exception as exc:
                for row in ready: row["status"]="not_issued";row["prediction"]=None;row["reason"]=f"prediction_failed:{type(exc).__name__}"
        for row in pending:
            row.pop("_feature",None); rows.append(row)

    result={"schemaVersion":1,"phase":"forecast_only","targetValuesAttached":False,
            "protocol":{"id":"full-history-relative-return-ridge-v1","products":list(PRODUCTS),"horizons":list(HORIZONS),
              "trainTargetDates":[TRAIN_START.isoformat(),TRAIN_END.isoformat()],"evaluationTargets":[EVAL_START.isoformat(),EVAL_END.isoformat()],
              "estimator":"training-only StandardScaler + Ridge(alpha=10, fit_intercept=True)","target":"log(actual/p0)",
              "minimumTrainingTargets":MIN_TRAIN,"originCutoff":"origin minus one calendar day",
              "preEvaluationOrigins":"not issued until training labels through 2025-12-31 are available",
              "featureDefinitions":{"p0":"latest positive observation age <=14d","p1":"prior distinct observation within 7d of p0","p7":"last observation <=p0_date-7d, within 7d of anchor","p30":"last observation <=p0_date-30d, within 14d of anchor","features":["log(p0/p1),log(p0/p7),log(p0/p30)","presence flags and actual gaps","p0 age","target weekday and annual sin/cos"]}},
            "inputSha256":{"prices":price_sha,"fixed50Manifest":fixed50_sha},"sourceDataAudit":_source_audit(prices),"plannedForecastCount":len(rows),"forecasts":rows}
    result["manifestSha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


def _canonical(value: Any) -> bytes:
    return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def build_from_paths(prices_path: Path = PRICES_PATH, fixed_path: Path = FIXED50_PATH) -> dict[str, Any]:
    if _sha(prices_path) != PRICES_SHA256: raise ValueError("pinned prices hash mismatch")
    if _sha(fixed_path) != FIXED50_SHA256: raise ValueError("pinned fixed50 manifest hash mismatch")
    return build(json.loads(prices_path.read_text()),json.loads(fixed_path.read_text()),price_sha=PRICES_SHA256,fixed50_sha=FIXED50_SHA256)


def verify_manifest(path: Path) -> bool:
    raw=json.loads(path.read_text()); expected=raw.get("manifestSha256"); copy=dict(raw); copy.pop("manifestSha256",None)
    if expected != hashlib.sha256(_canonical(copy)).hexdigest(): return False
    rebuilt=build_from_paths()
    return _canonical(raw)==_canonical(rebuilt)


def _dm(diff: dict[dt.date,float], bandwidth: int) -> dict[str,Any]:
    n=len(diff)
    if any(not math.isfinite(x) for x in diff.values()):
        return {"status":"nonfinite_loss_difference","reason":"nonfinite_loss_difference","n":n,"statistic":None,"meanLossDifference":None,"pValue":1.0}
    mean=math.fsum(diff.values())/n if n else None
    if n<30: return {"status":"insufficient_pairs","reason":"fewer_than_30_pairs","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.0}
    values=list(diff.values()); spread=max(values)-min(values); magnitude=max(abs(min(values)),abs(max(values)))
    spread_tolerance=8*math.ulp(magnitude) if magnitude else 8*math.ulp(0.0)
    if spread<=spread_tolerance:
        return {"status":"zero_spread","reason":"constant_loss_difference_within_float_tolerance","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.0}
    centered={d:v-mean for d,v in diff.items()}; gamma0=math.fsum(v*v for v in centered.values()); terms=[gamma0]
    for lag in range(1,bandwidth+1):
        cross=math.fsum(centered[d]*centered.get(d+dt.timedelta(days=lag),0.0) for d in centered)
        terms.append(2*(1-lag/(bandwidth+1))*cross)
    total=math.fsum(terms); variance=total/(n*n)
    tolerance=8*math.ulp(math.fsum(abs(x) for x in terms)) if any(terms) else 0.0
    if not math.isfinite(variance) or total<=tolerance:
        return {"status":"invalid_hac_variance","reason":"nonpositive_or_cancellation_dominated_hac_variance","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.0}
    stat=mean/math.sqrt(variance)
    return {"status":"tested","reason":None,"n":n,"statistic":stat,"meanLossDifference":mean,"pValue":math.erfc(abs(stat)/math.sqrt(2))}


def _holm(rows: list[dict[str,Any]]) -> None:
    indices=sorted(range(len(rows)),key=lambda i:rows[i]["pValue"]); run=0.0; n=len(rows)
    for rank,i in enumerate(indices):
        run=max(run,min(1.0,(n-rank)*rows[i]["pValue"])); rows[i]["holmPValue"]=run; rows[i]["holmSignificant"]=run<ALPHA


def score(manifest: dict[str,Any], prices: dict[str,Any]) -> dict[str,Any]:
    if manifest.get("phase")!="forecast_only" or manifest.get("targetValuesAttached") is not False: raise ValueError("expected answer-free manifest")
    declared=manifest.get("manifestSha256"); unsigned=dict(manifest); unsigned.pop("manifestSha256",None)
    if not isinstance(declared,str) or declared!=hashlib.sha256(_canonical(unsigned)).hexdigest(): raise ValueError("manifest hash mismatch")
    expected=set(_grid()); seen=set()
    for row in manifest.get("forecasts",[]):
        key=(row.get("product"),_date(row.get("target")),row.get("horizon"))
        if key not in expected or key in seen or row.get("status") not in ("issued","not_issued"): raise ValueError("manifest forecast grid is invalid")
        seen.add(key)
    if seen!=expected: raise ValueError("manifest forecast grid is incomplete")
    series=_series(prices); actuals={(p,d):v for p,rows in series.items() for d,v in rows.items() if EVAL_START<=d<=EVAL_END}
    scored=[]; predmap={(r["product"],_date(r["target"]),r["horizon"]):r for r in manifest["forecasts"]}
    coverage={}; metrics=[]; dmrows=[]; temporal_coverage={}
    for p in PRODUCTS:
      for h in HORIZONS:
        pp=[r for r in manifest["forecasts"] if r["product"]==p and r["horizon"]==h]
        pairrows=[]
        for r in pp:
            day=_date(r["target"]); actual=actuals.get((p,day))
            if actual is None or actual<=0: continue
            z={"product":p,"target":day.isoformat(),"horizon":h,"actual":actual,"status":r["status"],"candidate":r["prediction"] if r["status"]=="issued" else None,"persistence":r["persistence"],"fixed50":r["fixed50"]}
            scored.append(z); pairrows.append(z)
        for comp in COMPARATORS:
            common=[r for r in pairrows if r["status"]=="issued" and _positive(r["candidate"]) and _positive(r[comp])]
            ys=[r["actual"] for r in common]; a=[r["candidate"] for r in common]; b=[r[comp] for r in common]
            def mets(pred):
                if not ys:return {"n":0,"mae":None,"mape":None}
                return {"n":len(ys),"mae":sum(abs(y-v) for y,v in zip(ys,pred))/len(ys),"mape":100*sum(abs(y-v)/y for y,v in zip(ys,pred))/len(ys)}
            metrics.append({"product":p,"horizon":h,"comparator":comp,"candidate":mets(a),"comparatorMetrics":mets(b)})
            for loss in LOSSES:
                diffs={_date(r["target"]): (abs(r["candidate"]-r["actual"]) - abs(r[comp]-r["actual"]) if loss=="ae" else abs(r["candidate"]-r["actual"])/r["actual"]-abs(r[comp]-r["actual"])/r["actual"]) for r in common}
                for bw in BANDWIDTHS: dmrows.append({"product":p,"horizon":h,"comparator":comp,"loss":loss,"bandwidth":bw,**_dm(diffs,bw)})
        fixed_keys=[r for r in pp if _positive(r.get("fixed50"))]
        lost=[r["target"] for r in fixed_keys if r["status"]!="issued" or not _positive(r.get("prediction"))]
        coverage[f"{p}/{h}"]={"planned":len(pp),"issued":sum(r["status"]=="issued" for r in pp),"actualPositive":len(pairrows),
          "persistenceCommon":sum(bool(_positive(r["candidate"]) and _positive(r["persistence"])) for r in pairrows),
          "fixed50Common":sum(bool(_positive(r["candidate"]) and _positive(r["fixed50"])) for r in pairrows),
          "fixed50AvailableOnPlannedGrid":len(fixed_keys),"candidateIssuedOnFixed50Keys":len(fixed_keys)-len(lost),"candidateCoverageLossTargets":lost}
    forecast_by_key={(r["product"],_date(r["target"]),r["horizon"]):r for r in manifest["forecasts"]}
    for product in PRODUCTS:
      planned_days=[EVAL_START+dt.timedelta(days=i) for i in range((EVAL_END-EVAL_START).days+1)]
      for period in ("month","quarter"):
        groups={}
        for day in planned_days:
            label=f"{day.year}-{day.month:02d}" if period=="month" else f"{day.year}-Q{(day.month-1)//3+1}"
            groups.setdefault(label,[]).append(day)
        for label,days in groups.items():
            cells=[forecast_by_key[(product,day,h)] for day in days for h in HORIZONS]
            actual_count=sum((product,day) in actuals for day in days)*len(HORIZONS)
            temporal_coverage[f"{product}/{period}/{label}"]={"planned":len(cells),"issued":sum(bool(r["status"]=="issued" and _positive(r.get("prediction"))) for r in cells),
                "positiveActualPairs":actual_count,"issuedWithPositiveActual":sum(bool(r["status"]=="issued" and _positive(r.get("prediction")) and (product,_date(r["target"])) in actuals) for r in cells),
                "fixed50AvailableOnPlannedGrid":sum(bool(_positive(r.get("fixed50"))) for r in cells),
                "candidateCoverageLossCount":sum(bool(_positive(r.get("fixed50"))) and (r["status"]!="issued" or not _positive(r.get("prediction"))) for r in cells)}
    for bw in BANDWIDTHS: _holm([r for r in dmrows if r["bandwidth"]==bw])
    product_macro=[]; crop_gates=[]
    for p in PRODUCTS:
      for comp in COMPARATORS:
        cells=[r for r in metrics if r["product"]==p and r["comparator"]==comp and r["candidate"]["mape"] is not None and r["comparatorMetrics"]["mape"] is not None]
        product_macro.append({"product":p,"comparator":comp,"horizonsIncluded":len(cells),
          "candidateEqualHorizonMape":sum(r["candidate"]["mape"] for r in cells)/len(cells) if cells else None,
          "comparatorEqualHorizonMape":sum(r["comparatorMetrics"]["mape"] for r in cells)/len(cells) if cells else None,
          "candidateEqualHorizonMae":sum(r["candidate"]["mae"] for r in cells)/len(cells) if cells else None,
          "comparatorEqualHorizonMae":sum(r["comparatorMetrics"]["mae"] for r in cells)/len(cells) if cells else None})
      if comp=="fixed50":
        fixedcells=[r for r in metrics if r["product"]==p and r["comparator"]=="fixed50" and r["candidate"]["mape"] is not None]
        cm=sum(r["candidate"]["mape"] for r in fixedcells)/len(fixedcells) if fixedcells else None
        bm=sum(r["comparatorMetrics"]["mape"] for r in fixedcells)/len(fixedcells) if fixedcells else None
        ca=sum(r["candidate"]["mae"] for r in fixedcells)/len(fixedcells) if fixedcells else None
        ba=sum(r["comparatorMetrics"]["mae"] for r in fixedcells)/len(fixedcells) if fixedcells else None
        per_h={r["horizon"]:r for r in fixedcells}
        per_h_coverage=[]
        for h in HORIZONS:
            cell=coverage[f"{p}/{h}"]
            per_h_coverage.append(cell["candidateCoverageLossTargets"]==[])
        no_coverage_loss=all(per_h_coverage)
        complete_horizon_comparison=len(per_h)==len(HORIZONS)
        no_material_harm=complete_horizon_comparison and all((r["candidate"]["mape"]-r["comparatorMetrics"]["mape"]<=1.0) for r in fixedcells)
        joint=False
        for h in HORIZONS:
            tests={(r["loss"],r["bandwidth"]):r for r in dmrows if r["product"]==p and r["horizon"]==h and r["comparator"]=="fixed50"}
            if all((tests.get((loss,bw),{}).get("holmSignificant") is True and tests[(loss,bw)].get("meanLossDifference",0)<0) for loss in LOSSES for bw in BANDWIDTHS): joint=True
        passes=bool(fixedcells and cm is not None and bm is not None and cm<=bm-0.5 and ca<ba and no_coverage_loss and no_material_harm and joint)
        crop_gates.append({"product":p,"benchmark":"fixed50","candidateEqualHorizonMape":cm,"fixed50EqualHorizonMape":bm,"mapeImprovementPctPoints":bm-cm if cm is not None and bm is not None else None,
          "candidateEqualHorizonMae":ca,"fixed50EqualHorizonMae":ba,"coverageNondeclining":no_coverage_loss,"all10HorizonsHaveCommonPairs":complete_horizon_comparison,"noHorizonMapeWorseByMoreThan1PctPoint":no_material_harm,"jointAeApeHolmImprovementAtSomeHorizonBothHac":joint,"decision":"eligible_for_prospective_confirmation" if passes else "retain_existing_method"})
    return {"schemaVersion":1,"phase":"scored","manifestSha256":manifest.get("manifestSha256"),"metrics":metrics,"coverage":coverage,"temporalCoverage":temporal_coverage,"productMacro":product_macro,"cropGates":crop_gates,"dm":dmrows,"scoredPairs":scored}


def write_exclusive(path: Path, value: dict[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as f: f.write(_canonical(value))


def main(argv: list[str] | None = None) -> int:
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    b=sub.add_parser("build"); b.add_argument("output",type=Path)
    v=sub.add_parser("verify"); v.add_argument("manifest",type=Path)
    s=sub.add_parser("score"); s.add_argument("manifest",type=Path); s.add_argument("output",type=Path)
    args=ap.parse_args(argv)
    if args.cmd=="build": write_exclusive(args.output,build_from_paths())
    elif args.cmd=="verify":
        if not verify_manifest(args.manifest): raise SystemExit("manifest replay mismatch")
    else:
        if not verify_manifest(args.manifest): raise SystemExit("manifest replay mismatch")
        if _sha(PRICES_PATH)!=PRICES_SHA256: raise SystemExit("pinned prices hash mismatch")
        write_exclusive(args.output,score(json.loads(args.manifest.read_text()),json.loads(PRICES_PATH.read_text())))
    return 0

if __name__=="__main__": raise SystemExit(main())
