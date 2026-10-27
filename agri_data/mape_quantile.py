"""Frozen pooled-horizon quantile-HistGB experiment for relative-price error."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import tempfile
from typing import Any, Callable

from . import dashboard, direct_horizon as dh, modeling

ROOT = Path(__file__).resolve().parents[1]
INPUT_PRICES = dh.INPUT_PRICES
INPUT_REPORT = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
PRICES_SHA256 = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
REPORT_SHA256 = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
START, END = dt.date(2025, 9, 25), dt.date(2026, 9, 24)
HORIZONS = tuple(range(1, 11))
MIN_TRAIN_TARGETS = 90
BANDWIDTHS = (9, 28)
COMPARATORS = ("directRidge", "persistence")
LOSSES = ("mae", "mape")
FAMILY_SIZE = len(dashboard.PRODUCTS) * 10 * 2 * 2
MODEL_PARAMS = {"loss": "quantile", "quantile": .5, "max_iter": 40, "max_leaf_nodes": 15,
                "min_samples_leaf": 10, "learning_rate": .08, "l2_regularization": 1.0,
                "random_state": 42, "early_stopping": False}


def _pos(value: Any, label: str) -> float:
    if isinstance(value, bool): raise ValueError(f"{label} must be positive and finite")
    try: value = float(value)
    except (TypeError, ValueError, OverflowError) as exc: raise ValueError(f"{label} must be positive and finite") from exc
    if not math.isfinite(value) or value <= 0: raise ValueError(f"{label} must be positive and finite")
    return value


def _rows(report: dict[str, Any]) -> dict[tuple[str, dt.date, dt.date, int], dict[str, float]]:
    if report.get("schemaVersion", 1) != 1 or report.get("protocol", {}).get("id") != "direct-horizon-ridge-2026-10-06":
        raise ValueError("unexpected direct-horizon report schema or protocol")
    out = {}
    for row in report.get("pairedRows", []):
        if not isinstance(row, dict) or row.get("_summary"): continue
        p, h = row.get("product"), row.get("horizon")
        if p not in dashboard.PRODUCTS or isinstance(h, bool) or h not in HORIZONS: raise ValueError("invalid product or horizon")
        target = dt.date.fromisoformat(row["target"])
        origin = dt.date.fromisoformat(row["origin"]) if row.get("origin") else target-dt.timedelta(days=h)
        if target-origin != dt.timedelta(days=h): raise ValueError("origin/horizon mismatch")
        key = (p, origin, target, h)
        if key in out: raise ValueError(f"duplicate direct-report identity {key}")
        out[key] = {"actual": _pos(row["actual"], "actual"), "directRidge": _pos(row["direct"], "direct Ridge"),
                    "persistence": _pos(row["persistence"], "persistence")}
    if not out: raise ValueError("direct report contains no pairedRows")
    return out


def _make_estimator():
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
    except ImportError as exc: raise RuntimeError("scikit-learn is required") from exc
    return HistGradientBoostingRegressor(**MODEL_PARAMS)


def _season(day: dt.date) -> str:
    return ("DJF", "MAM", "JJA", "SON")[(day.month % 12) // 3]


def _score(actual: list[float], pred: list[float]) -> dict[str, float | None]:
    return dh._metrics(actual, pred)


def _build(prices: dict[str, Any], report: dict[str, Any], *, estimator_factory: Callable[[], Any] = _make_estimator,
           hashes: dict[str, str] | None = None) -> dict[str, Any]:
    series_map = dh._observations(prices)
    direct = _rows(report)
    rows = [dict(product=p, origin=o, target=t, horizon=h, **values) for (p,o,t,h),values in sorted(direct.items())
            if START <= t <= END]
    # Only report identities with target outcomes still present in the pinned price snapshot are eligible.
    for row in rows:
        observed = series_map[row["product"]].get(row["target"])
        if observed is None or not math.isclose(observed[0],row["actual"],rel_tol=1e-12,abs_tol=0):
            raise ValueError(f"pinned price/report actual mismatch at {row['product']}/{row['target']}")
    candidate: dict[tuple[str, dt.date, dt.date, int], dict[str, Any]] = {}
    fit_audit: dict[str, list[dict[str, Any]]] = {}
    for product, series in series_map.items():
        cache: dict[Any, Any] = {}
        samples = []
        for target, (actual, _) in series.items():
            for horizon in HORIZONS:
                origin = target - dt.timedelta(days=horizon)
                feat = modeling._feature_row(series, origin, target, horizon, cache)
                if feat is None: continue
                source_day, vector = feat
                source_price = series[source_day][0]
                weight = source_price / actual
                label = math.log(actual / source_price)
                if not math.isfinite(weight) or weight <= 0 or not math.isfinite(label):
                    raise ValueError(f"invalid training weight/label at {product}/{target}/h{horizon}")
                samples.append({"target": target, "origin": origin, "horizon": horizon, "sourceDate": source_day,
                                "sourcePrice": source_price, "actual": actual, "weight": weight,
                                "label": label, "features": vector})
        months: dict[tuple[int,int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["product"] == product: months[(row["origin"].year,row["origin"].month)].append(row)
        product_audit=[]
        for month in sorted(months):
            eval_rows = months[month]
            fit_origin = min(r["origin"] for r in eval_rows)
            training = [s for s in samples if s["target"] < fit_origin]
            distinct = len({s["target"] for s in training})
            entry={"fitMonth":f"{month[0]:04d}-{month[1]:02d}","fitOrigin":fit_origin.isoformat(),
                   "fitTrainTargetEnd":max((s["target"] for s in training),default=None).isoformat() if training else None,
                   "fitDistinctTargets":distinct,"fitSamples":len(training),"status":None}
            if distinct < MIN_TRAIN_TARGETS:
                entry["status"]="insufficient_training"
                for r in eval_rows: r["candidateMissingReason"]="insufficient_training"
                product_audit.append(entry); continue
            try:
                import numpy as np
                model=estimator_factory()
                model.fit(np.asarray([s["features"] for s in training],dtype=float),
                          np.asarray([s["label"] for s in training],dtype=float),
                          sample_weight=np.asarray([s["weight"] for s in training],dtype=float))
            except Exception as exc:
                entry["status"]="fit_failed"; entry["errorType"]=type(exc).__name__; entry["error"] = str(exc)
                for r in eval_rows: r["candidateMissingReason"]="fit_failed"
                product_audit.append(entry); continue
            month_predictions={}
            for r in eval_rows:
                source_day, vector = modeling._feature_row(series,r["origin"],r["target"],r["horizon"],cache) or (None,None)
                if source_day is None:
                    r["candidateMissingReason"]="feature_unavailable_or_source_too_old"; continue
                source_price=series[source_day][0]
                try:
                    log_ratio=float(model.predict(np.asarray([vector],dtype=float))[0])
                    if not math.isfinite(log_ratio):
                        r["candidateMissingReason"]="prediction_failed:nonfinite_prediction"; continue
                    predicted, clipped=modeling._inverse_log_ratio(source_price,log_ratio)
                    if not math.isfinite(predicted) or predicted<=0:
                        r["candidateMissingReason"]="prediction_failed:invalid_price"; continue
                    month_predictions[(product,r["origin"],r["target"],r["horizon"]) ]={"prediction":predicted,
                        "sourceDate":source_day,"sourcePrice":source_price,"fitMonth":entry["fitMonth"],
                        "fitOrigin":fit_origin,"fitTrainTargetEnd":entry["fitTrainTargetEnd"],
                        "fitDistinctTargets":distinct,"fitSamples":len(training),"predictedLogRatio":log_ratio,"clipped":clipped}
                except Exception as exc:
                    r["candidateMissingReason"]=f"prediction_failed:{type(exc).__name__}"
            candidate.update(month_predictions)
            entry["status"]="fitted" if len(month_predictions)==len(eval_rows) else "fitted_with_prediction_gaps"
            product_audit.append(entry)
        fit_audit[product]=product_audit

    paired=[]; comparisons=[]; coverage={}; strata=[]
    for product in dashboard.PRODUCTS:
        for h in HORIZONS:
            group=[r for r in rows if r["product"]==product and r["horizon"]==h]
            comp=[]
            for r in group:
                pred=candidate.get((product,r["origin"],r["target"],h))
                if pred:
                    out={**r,**pred,"candidatePrediction":pred["prediction"]}; comp.append(out)
                    paired.append({**out,"origin":out["origin"].isoformat(),"target":out["target"].isoformat(),
                                   "sourceDate":out["sourceDate"].isoformat(),"fitOrigin":out["fitOrigin"].isoformat()})
            valid_dates={r["target"] for r in comp}
            coverage[f"{product}/{h}"]={"eligibleDirectReportRows":len(group),"candidatePairedCount":len(comp),
                "candidateCoverageLossDates":[r["target"].isoformat() for r in group if r["target"] not in valid_dates],
                "candidateCoverageNondeclining":len(comp)==len(group),"missingReasons":sorted({r["candidateMissingReason"] for r in group if "candidateMissingReason" in r}),
                "missingOriginsTargets":[{"origin":r["origin"].isoformat(),"target":r["target"].isoformat(),
                    "reason":r.get("candidateMissingReason","candidate_prediction_missing")} for r in group if r["target"] not in valid_dates]}
            for comparator in COMPARATORS:
                for loss in LOSSES:
                    diffs={r["target"]: (abs(r["actual"]-r["candidatePrediction"])-abs(r["actual"]-r[comparator])) if loss=="mae" else 100*(abs(r["actual"]-r["candidatePrediction"])/r["actual"]-abs(r["actual"]-r[comparator])/r["actual"]) for r in comp}
                    comparisons.append({"product":product,"horizon":h,"comparator":comparator,"loss":loss,"n":len(comp),
                        "candidateMetric":_score([r["actual"] for r in comp],[r["candidatePrediction"] for r in comp])[loss],
                        "comparatorMetric":_score([r["actual"] for r in comp],[r[comparator] for r in comp])[loss],
                        "dm":{str(b):dh._dm(diffs,b) for b in BANDWIDTHS}})
            for field,fn in (("year",lambda d:str(d.year)),("season",_season)):
                buckets=defaultdict(list)
                for r in comp: buckets[fn(r["target"])].append(r)
                for label,bucket in sorted(buckets.items()):
                    strata.append({"product":product,"horizon":h,"stratum":field,"value":label,"n":len(bucket),
                        "candidate":_score([r["actual"] for r in bucket],[r["candidatePrediction"] for r in bucket]),
                        "directRidge":_score([r["actual"] for r in bucket],[r["directRidge"] for r in bucket]),
                        "persistence":_score([r["actual"] for r in bucket],[r["persistence"] for r in bucket])})
    for bw in BANDWIDTHS: dh._holm([x["dm"][str(bw)] for x in comparisons])
    macro={}
    for product in dashboard.PRODUCTS:
        per_method={}
        for method,field in (("candidate","candidatePrediction"),("directRidge","directRidge"),("persistence","persistence")):
            per_h={str(h):_score([r["actual"] for r in paired if r["product"]==product and r["horizon"]==h],
                                [r[field] for r in paired if r["product"]==product and r["horizon"]==h]) for h in HORIZONS}
            per_method[method]={"byHorizon":per_h,
                "equal10HorizonMape":sum(v["mape"] for v in per_h.values())/10 if all(v["mape"] is not None for v in per_h.values()) else None,
                "equal10HorizonMae":sum(v["mae"] for v in per_h.values())/10 if all(v["mae"] is not None for v in per_h.values()) else None}
        macro[product]=per_method
    goals={"below10PercentCount":sum(v["candidate"]["equal10HorizonMape"] is not None and v["candidate"]["equal10HorizonMape"]<10 for v in macro.values()),
           "below7PercentCount":sum(v["candidate"]["equal10HorizonMape"] is not None and v["candidate"]["equal10HorizonMape"]<7 for v in macro.values()),
           "goalMet":False}
    goals["goalMet"]=goals["below10PercentCount"]>=3 or goals["below7PercentCount"]>=2
    indexed={(r["product"],r["horizon"],r["comparator"],r["loss"]):r for r in comparisons}
    conclusions=[]
    for product in dashboard.PRODUCTS:
        for h in HORIZONS:
            family=[indexed[(product,h,c,l)] for c in COMPARATORS for l in LOSSES]
            dm9=[item["dm"]["9"] for item in family]
            all_lower=all(item["candidateMetric"] is not None and item["comparatorMetric"] is not None and item["candidateMetric"]<item["comparatorMetric"] for item in family)
            all_sig=all(x.get("holmSignificant") and x.get("meanLossDifference") is not None and x["meanLossDifference"]<0 for x in dm9)
            available=all(x["status"]=="tested" and x["n"]>=30 for x in dm9)
            nondeclining=coverage[f"{product}/{h}"]["candidateCoverageNondeclining"]
            adverse=any(x.get("holmSignificant") and x.get("meanLossDifference") is not None and x["meanLossDifference"]>0 for x in dm9)
            conclusion="inference_unavailable" if not available else "improvement" if all_lower and all_sig and nondeclining else "worsening_or_mixed" if adverse else "inconclusive"
            conclusions.append({"product":product,"horizon":h,"conclusion":conclusion,"inferenceAvailable":available,
                "allFourPairedMetricsImprove":all_lower,"allFourHac9HolmTestsImprove":all_sig,"coverageNondeclining":nondeclining})
    return {"schemaVersion":1,"protocol":{"id":"relative-error-pooled-histgb-2026-10-07","inputHashes":{"prices":PRICES_SHA256,"directReport":REPORT_SHA256},
        "testTargetInterval":[START.isoformat(),END.isoformat()],"model":"HistGradientBoostingRegressor","parameters":MODEL_PARAMS,
        "pooling":"one fit per product and origin month; horizon 1..10 pooled","target":"log(actual/sourcePrice)","sampleWeight":"sourcePrice/actual",
        "minimumDistinctTrainingTargets":MIN_TRAIN_TARGETS,"trainingCutoff":"target date strictly before first evaluation origin of month",
        "dmFamilySize":FAMILY_SIZE,"comparators":list(COMPARATORS),"losses":list(LOSSES),"hacCalendarDays":list(BANDWIDTHS),"holmAlpha":.05,"minimumDmPairs":30},
        "hashes":hashes or {},"fitAudit":fit_audit,"coverage":coverage,"pairedRows":paired,"comparisons":comparisons,
        "conclusions":conclusions,"macroMetrics":macro,"yearSeasonMetrics":strata,"goals":goals,
        "limitation":"Retrospective evaluation of revised historical inputs; not independent confirmation or official nationwide performance."}


def build(prices: dict[str, Any], report: dict[str, Any], *, hashes=None, estimator_factory=_make_estimator):
    return _build(prices,report,hashes=hashes,estimator_factory=estimator_factory)


def _read(path):
    raw=Path(path).read_bytes()
    try: data=json.loads(raw)
    except (UnicodeDecodeError,json.JSONDecodeError) as exc: raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(data,dict): raise ValueError("input must be a JSON object")
    return data,raw


def _write_immutable(path:Path,payload:bytes):
    path.parent.mkdir(parents=True,exist_ok=True); fd,tmp=tempfile.mkstemp(prefix=f".{path.name}.",dir=path.parent)
    try:
        with os.fdopen(fd,"wb") as f: fd=None; f.write(payload); f.flush(); os.fsync(f.fileno())
        os.link(tmp,path)
    except FileExistsError as exc: raise FileExistsError(f"refusing to overwrite immutable output: {path}") from exc
    finally:
        if fd is not None: os.close(fd)
        try: os.unlink(tmp)
        except FileNotFoundError: pass


def run(prices_path=INPUT_PRICES, report_path=INPUT_REPORT, output_path=None):
    if output_path is None: raise ValueError("output_path is required")
    p,pb=_read(prices_path); r,rb=_read(report_path)
    hashes={"prices":hashlib.sha256(pb).hexdigest(),"directReport":hashlib.sha256(rb).hexdigest()}
    if hashes != {"prices":PRICES_SHA256,"directReport":REPORT_SHA256}: raise ValueError("input SHA-256 differs from frozen protocol")
    result=build(p,r,hashes=hashes)
    payload=(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode()
    _write_immutable(Path(output_path),payload)
    return {"output":str(output_path),"sha256":hashlib.sha256(payload).hexdigest()}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices",type=Path,default=INPUT_PRICES); parser.add_argument("--report",type=Path,default=INPUT_REPORT)
    parser.add_argument("--output",type=Path,required=True); a=parser.parse_args(argv)
    run(a.prices,a.report,a.output); return 0


if __name__=="__main__": raise SystemExit(main())
