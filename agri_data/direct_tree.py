"""Frozen offline direct-per-horizon HistGradientBoosting experiment."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import dashboard, direct_horizon as dh, modeling

ROOT = Path(__file__).resolve().parents[1]
INPUT_PRICES = dh.INPUT_PRICES
INPUT_PREDICTIONS = dh.INPUT_PREDICTIONS
INPUT_RIDGE_REPORT = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
PRICES_SHA256 = dh.PRICES_SHA256
PREDICTIONS_SHA256 = dh.PREDICTIONS_SHA256
RIDGE_REPORT_SHA256 = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
PRODUCTS = tuple(dashboard.PRODUCTS)
HORIZONS = (1, 5, 10)
EVAL_START, EVAL_END = dh.EVAL_START, dh.EVAL_END
MIN_TRAIN_DAYS = 90
COMPARATORS = ("monthly_pooled_hist_gb", "direct_ridge", "persistence")
LOSSES = ("mae", "mape")
BANDWIDTHS = (9, 28)
FAMILY_SIZE = len(PRODUCTS) * len(HORIZONS) * len(COMPARATORS) * len(LOSSES)


def _ridge_reference(report: dict[str, Any]):
    if report.get("protocol", {}).get("id") != "direct-horizon-ridge-2026-10-06":
        raise ValueError("unexpected direct-horizon Ridge report")
    values = {}
    coverage = report.get("coverage")
    if not isinstance(coverage, dict):
        raise ValueError("Ridge report lacks coverage")
    for row in report.get("pairedRows", []):
        key = (row["product"], int(row["horizon"]), dt.date.fromisoformat(row["target"]))
        values[key] = (float(row["actual"]), float(row["direct"]))
    available = {}
    for key, item in coverage.items():
        available[key] = {dt.date.fromisoformat(x) for x in item.get("candidateAvailableDates", [])}
    return values, available


def _model():
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=40, max_leaf_nodes=15, learning_rate=.08,
        l2_regularization=1.0, random_state=42, early_stopping=False)


def _thread_limit_context(threadpool_limits):
    try:
        return threadpool_limits(limits=1)
    except Exception:
        return nullcontext()


def _conclusion(rows, coverage, controls=COMPARATORS):
    tests=[r['dm']['9'] for r in rows]
    enough=all(t['status']=='tested' and t['n']>=30 for t in tests)
    low=all(r['candidateMetric'] is not None and r['candidateMetric']<r['comparatorMetric'] for r in rows)
    sig=all(t.get('holmSignificant') and t['meanLossDifference']<0 for t in tests)
    cov=all(coverage['candidateCoverageLossDates'][c]==0 for c in controls)
    significant_worsening=any(t['status']=='tested' and t.get('holmSignificant') and t['meanLossDifference']>0 for t in tests)
    direction=('lower' if low else 'higher' if any(r['candidateMetric'] is not None and r['candidateMetric']>r['comparatorMetric'] for r in rows) else 'mixed_or_equal')
    status='inference_unavailable' if not enough else 'improvement' if low and sig and cov else 'worsening_or_mixed' if significant_worsening else 'inconclusive'
    return {"conclusion":status,"inferenceAvailable":enough,"descriptiveMetricDirection":direction,
        "significantWorseningEvidence":significant_worsening,"allSixHac9HolmTestsImprove":sig,
        "allSixPairedMetricsImprove":low,"coverageNondeclining":cov}


def _monthly_hist_gb(raw):
    if raw.get("schemaVersion") != 2 or not isinstance(raw.get("products"), list):
        raise ValueError("predictions must use dashboard schemaVersion 2")
    result = {}
    for product in raw["products"]:
        pid = product.get("id")
        if pid not in PRODUCTS or pid in result: raise ValueError("unknown or duplicate prediction product")
        selected = [r for r in product.get("runs", []) if r.get("modelId") == "hist_gb" and r.get("cadence") == "monthly"]
        if len(selected) != 1: raise ValueError(f"expected one monthly pooled HistGB run for {pid}")
        values = {}
        for row in selected[0].get("predictions", []):
            day=modeling._parse_day(row.get("target")); origin=modeling._parse_day(row.get("origin")); h=row.get("horizon")
            if day is None or h not in HORIZONS: continue
            if origin != day-dt.timedelta(days=h): raise ValueError("monthly HistGB origin mismatch")
            pred=float(row.get("predicted"))
            if not math.isfinite(pred) or pred<=0: continue
            key=(day,h)
            if key in values: raise ValueError("duplicate monthly HistGB prediction")
            values[key]=pred
        result[pid]=values
    if set(result)!=set(PRODUCTS): raise ValueError("monthly HistGB predictions must cover every product")
    return result


def build(prices: dict[str, Any], forecasts: dict[str, Any], ridge_report: dict[str, Any], *, hashes=None):
    series_by_product = dh._observations(prices)
    _, _, persistence = dh._forecast_maps(forecasts)
    pooled = _monthly_hist_gb(forecasts)
    ridge, ridge_available = _ridge_reference(ridge_report)
    predictions, fits, statuses = [], {}, {}
    for product in PRODUCTS:
        series = series_by_product[product]
        cache = {}
        for horizon in HORIZONS:
            samples = []
            for target, (actual, _) in series.items():
                origin = target - dt.timedelta(days=horizon)
                feature = modeling._feature_row(series, origin, target, horizon, cache)
                if feature is None: continue
                source, vector = feature
                samples.append((target, origin, source, actual, vector,
                                math.log(actual) - math.log(series[source][0])))
            evals = [(d, d-dt.timedelta(days=horizon)) for d in series
                     if EVAL_START <= d <= EVAL_END and d-dt.timedelta(days=horizon) >= EVAL_START]
            months = defaultdict(list)
            for target, origin in evals: months[(origin.year, origin.month)].append((target, origin))
            audit = {"fitAttempts": 0, "fitSuccesses": 0, "months": []}
            for (year, month), targets in sorted(months.items()):
                origin0 = min(o for _, o in targets)
                train = [r for r in samples if r[0] < origin0]
                distinct = len({r[0] for r in train})
                entry = {"originMonth": f"{year:04}-{month:02}", "fitOrigin": origin0.isoformat(),
                         "fitTrainTargetEnd": max((r[0] for r in train), default=None).isoformat() if train else None,
                         "fitDistinctTargets": distinct, "fitSamples": len(train), "status": None}
                if distinct < MIN_TRAIN_DAYS:
                    entry["status"] = "insufficient_training"
                    for target, _ in targets: statuses[(product,horizon,target)] = "insufficient_training"
                    audit["months"].append(entry); continue
                audit["fitAttempts"] += 1
                try:
                    import numpy as np
                    from threadpoolctl import threadpool_limits
                    estimator = _model()
                    month_predictions = []
                    with _thread_limit_context(threadpool_limits):
                        estimator.fit(np.asarray([r[4] for r in train]), np.asarray([r[5] for r in train]))
                        for target, origin in targets:
                            sample = next((r for r in samples if r[0] == target and r[1] == origin), None)
                            if not sample: continue
                            pred, clipped = modeling._inverse_log_ratio(series[sample[2]][0], float(estimator.predict(np.asarray([sample[4]]))[0]))
                            month_predictions.append({"product":product,"horizon":horizon,"target":target.isoformat(),"origin":origin.isoformat(),
                                "sourceDate":sample[2].isoformat(),"actual":sample[3],"predicted":pred,"clipped":clipped,
                                "fitOrigin":origin0.isoformat(),"fitTrainTargetEnd":entry["fitTrainTargetEnd"],
                                "fitDistinctTargets":distinct,"fitSamples":len(train)})
                    predictions.extend(month_predictions)
                    for row in month_predictions:
                        statuses[(product,horizon,dt.date.fromisoformat(row["target"]))] = "candidate_available"
                    audit["fitSuccesses"] += 1; entry["status"]="fitted"
                except Exception as exc:
                    entry["status"]="fit_failed"; entry["errorType"]=type(exc).__name__; entry["error"]=str(exc)[:200]
                    for target,_ in targets: statuses[(product,horizon,target)]="fit_failed"
                audit["months"].append(entry)
            fits[f"{product}/{horizon}"] = audit

    predmap={(r['product'],r['horizon'],dt.date.fromisoformat(r['target'])):r for r in predictions}
    paired, coverage, summaries, strata = [], {}, [], []
    controls = ("monthly_pooled_hist_gb", "direct_ridge", "persistence")
    for product in PRODUCTS:
        series=series_by_product[product]
        for horizon in HORIZONS:
            eligible=sorted(d for d in series if EVAL_START<=d<=EVAL_END and d-dt.timedelta(days=horizon)>=EVAL_START)
            comp_dates={c:set() for c in controls}; candidates=[]
            for day in eligible:
                key=(day,horizon); origin=day-dt.timedelta(days=horizon)
                cand=predmap.get((product,horizon,day)); month=pooled[product].get(key); pers=persistence[product].get(key)
                rr=ridge.get((product,horizon,day)); actual=series[day][0]
                if month is not None: comp_dates[controls[0]].add(day)
                if day in ridge_available.get(f"{product}/{horizon}",set()): comp_dates[controls[1]].add(day)
                if pers is not None: comp_dates[controls[2]].add(day)
                source=max((d for d in series if d<=origin),default=None)
                if rr is not None and not math.isclose(rr[0], actual, rel_tol=1e-12, abs_tol=0.0):
                    raise ValueError(f"direct Ridge actual does not match frozen prices for {product}/{day}/h{horizon}")
                if cand and month and pers and rr and source and (origin-source).days<=dashboard.MAX_SOURCE_AGE_DAYS:
                    candidates.append({"product":product,"horizon":horizon,"target":day.isoformat(),"origin":origin.isoformat(),"actual":actual,
                        "directTree":cand["predicted"], controls[0]:month, controls[1]:rr[1], controls[2]:pers})
            paired.extend(candidates)
            candidate_dates={k[2] for k in predmap if k[0]==product and k[1]==horizon and k[2] in set(eligible)}
            covkey=f"{product}/{horizon}"
            coverage[covkey]={"eligibleCount":len(eligible),"candidateAvailableCount":sum(1 for k in predmap if k[0]==product and k[1]==horizon),
                "commonPairedCount":len(candidates),"pairedDates":[r['target'] for r in candidates],
                "candidateCoverageLossDates":{c:len(ds-candidate_dates) for c,ds in comp_dates.items()},
                "controlAvailableCounts":{c:len(ds) for c,ds in comp_dates.items()},
                "exclusionCounts":{"unpaired":len(eligible)-len(candidates)}}
            for comp in controls:
                for loss in LOSSES:
                    diff={}; ys=[]; ps=[]; qs=[]
                    for r in candidates:
                        y,a,b=r['actual'],r['directTree'],r[comp]
                        diff[dt.date.fromisoformat(r['target'])]=abs(y-a)-abs(y-b) if loss=='mae' else 100*(abs(y-a)/y-abs(y-b)/y)
                        ys.append(y); ps.append(a); qs.append(b)
                    dmh={str(bw):dh._dm(diff,bw) for bw in BANDWIDTHS}
                    summaries.append({"product":product,"horizon":horizon,"comparator":comp,"loss":loss,"n":len(diff),
                        "candidateMetric":dh._metrics(ys,ps)[loss],"comparatorMetric":dh._metrics(ys,qs)[loss],"dm":dmh})
            for field,fn in (("year",lambda d:d.year),("season",lambda d:(d.month-1)//3+1)):
                groups=defaultdict(list)
                for r in candidates: groups[fn(dt.date.fromisoformat(r['target']))].append(r)
                for val,group in sorted(groups.items()):
                    ys=[r['actual'] for r in group]
                    strata.append({"product":product,"horizon":horizon,"stratum":field,"value":val,"n":len(group),
                        "directTree":dh._metrics(ys,[r['directTree'] for r in group]),**{c:dh._metrics(ys,[r[c] for r in group]) for c in controls}})
    for bw in BANDWIDTHS:
        family=[r['dm'][str(bw)] for r in summaries]
        if len(family)!=FAMILY_SIZE: raise AssertionError("fixed Holm family size mismatch")
        dh._holm(family)
    conclusions=[]
    for p in PRODUCTS:
        for h in HORIZONS:
            rows=[r for r in summaries if r['product']==p and r['horizon']==h]
            conclusions.append({"product":p,"horizon":h,**_conclusion(rows,coverage[f'{p}/{h}'])})
    return {"protocol":{"id":"direct-per-horizon-hist-gb-2026-10-06","evaluation":[EVAL_START.isoformat(),EVAL_END.isoformat()],
        "products":list(PRODUCTS),"horizons":list(HORIZONS),"minimumDistinctTrainingTargets":MIN_TRAIN_DAYS,
        "trainTargetsStrictlyBeforeFitOrigin":True,"maxSourceAgeDays":dashboard.MAX_SOURCE_AGE_DAYS,
        "model":{"name":"HistGradientBoostingRegressor","max_iter":40,"max_leaf_nodes":15,"learning_rate":0.08,
                 "l2_regularization":1.0,"random_state":42,"early_stopping":False},
        "comparators":list(COMPARATORS),"losses":list(LOSSES),"holmFamilySizePerBandwidth":FAMILY_SIZE,
        "bandwidthsCalendarDays":list(BANDWIDTHS),"alpha":0.05,
        "limitations":"Retrospective revised prices; historical publication times are unknown. Direct Ridge uses its frozen report cutoff <= origin; tree uses strict < origin."},
        "hashes":hashes or {},"fitAudits":fits,"fitTotals":{"fitAttempts":sum(v['fitAttempts'] for v in fits.values()),"fitSuccesses":sum(v['fitSuccesses'] for v in fits.values())},
        "coverage":coverage,"predictions":predictions,"pairedRows":paired,"comparisons":summaries,"strata":strata,"jointConclusions":conclusions}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices',type=Path,default=INPUT_PRICES); parser.add_argument('--predictions',type=Path,default=INPUT_PREDICTIONS)
    parser.add_argument('--ridge-report',type=Path,default=INPUT_RIDGE_REPORT); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-nonfrozen-inputs',action='store_true'); args=parser.parse_args(argv)
    if args.output.resolve() in (args.prices.resolve(),args.predictions.resolve(),args.ridge_report.resolve()): parser.error('output must not replace input')
    hashes={"pricesSha256":dh._sha(args.prices),"predictionsSha256":dh._sha(args.predictions),"directRidgeReportSha256":dh._sha(args.ridge_report)}
    if not args.allow_nonfrozen_inputs and (hashes['pricesSha256']!=PRICES_SHA256 or hashes['predictionsSha256']!=PREDICTIONS_SHA256 or hashes['directRidgeReportSha256']!=RIDGE_REPORT_SHA256):
        parser.error('input SHA-256 does not match frozen protocol')
    result=build(dh._read(args.prices),dh._read(args.predictions),dh._read(args.ridge_report),hashes=hashes)
    args.output.parent.mkdir(parents=True,exist_ok=True); tmp=args.output.with_suffix(args.output.suffix+'.tmp')
    tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8'); tmp.replace(args.output)
    print(json.dumps({"output":str(args.output),"hashes":hashes,"fitTotals":result['fitTotals']},ensure_ascii=False)); return 0


if __name__ == '__main__': raise SystemExit(main())
