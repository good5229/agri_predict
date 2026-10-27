"""Frozen offline same-weekday history feature experiment (no API access)."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any
from . import dashboard, direct_horizon as dh, direct_tree, modeling

ROOT=Path(__file__).resolve().parents[1]
INPUT_PRICES=dh.INPUT_PRICES
INPUT_PREDICTIONS=dh.INPUT_PREDICTIONS
INPUT_RIDGE_REPORT=ROOT/"outputs/experiments/direct-horizon-20261006/report.json"
PRICES_SHA256=dh.PRICES_SHA256
PREDICTIONS_SHA256=dh.PREDICTIONS_SHA256
RIDGE_REPORT_SHA256=direct_tree.RIDGE_REPORT_SHA256
PRODUCTS=tuple(dashboard.PRODUCTS); HORIZONS=tuple(range(1,11))
EVAL_START,EVAL_END=dh.EVAL_START,dh.EVAL_END
MIN_TRAIN_DAYS=90; COMPARATORS=("monthly_pooled_hist_gb","direct_ridge","persistence")
LOSSES=("mae","mape"); BANDWIDTHS=(9,28); FAMILY_SIZE=len(PRODUCTS)*len(HORIZONS)*len(COMPARATORS)*len(LOSSES)


def weekday_features(series: dict[dt.date, tuple[float,float]], origin: dt.date, target: dt.date, days=None):
    """Return median log price, count, age of last matching weekday and latest-minus-median."""
    days = tuple(series) if days is None else days
    lo=bisect_left(days,origin-dt.timedelta(days=56)); hi=bisect_left(days,origin)
    selected=[d for d in days[lo:hi] if d.weekday()==target.weekday()]
    latest_idx=bisect_right(days,origin)-1
    if latest_idx<0: raise ValueError("origin has no observed source price")
    latest=math.log(series[days[latest_idx]][0])
    if selected:
        vals=sorted(math.log(series[d][0]) for d in selected); n=len(vals)
        median=(vals[n//2] if n%2 else (vals[n//2-1]+vals[n//2])/2)
        age=(origin-selected[-1]).days
        delta=latest-median
    else: median=delta=0.0; n=0; age=57
    return (median,float(n),float(age),delta)


def _feature(series, origin, target, horizon, cache):
    base=modeling._feature_row(series,origin,target,horizon,cache)
    if base is None:return None
    source,vector=base
    weekday_days=cache.get("weekday_days")
    if weekday_days is None:
        weekday_days=tuple(series)
        cache["weekday_days"]=weekday_days
    return source,vector+weekday_features(series,origin,target,weekday_days)


def _direct_ridge_reference(report):
    if report.get("protocol",{}).get("id")!="direct-horizon-ridge-2026-10-06":raise ValueError("unexpected direct-horizon Ridge report")
    values={}
    for row in report.get("pairedRows",[]): values[(row["product"],int(row["horizon"]),dt.date.fromisoformat(row["target"]))]=float(row["direct"])
    return values


def _monthly_hist_gb_all_horizons(raw):
    if raw.get("schemaVersion") != 2 or not isinstance(raw.get("products"), list):
        raise ValueError("predictions must use dashboard schemaVersion 2")
    result = {}
    for product in raw["products"]:
        pid = product.get("id")
        if pid not in PRODUCTS or pid in result:
            raise ValueError("unknown or duplicate prediction product")
        runs = [r for r in product.get("runs", []) if r.get("modelId") == "hist_gb" and r.get("cadence") == "monthly"]
        if len(runs) != 1:
            raise ValueError(f"expected one monthly pooled HistGB run for {pid}")
        values = {}
        for row in runs[0].get("predictions", []):
            day = modeling._parse_day(row.get("target")); origin = modeling._parse_day(row.get("origin")); h = row.get("horizon")
            if day is None or isinstance(h, bool) or h not in HORIZONS:
                continue
            if origin != day - dt.timedelta(days=h):
                raise ValueError(f"monthly HistGB origin mismatch for {pid}/{day}/h{h}")
            try: pred = float(row.get("predicted"))
            except (ValueError, TypeError, OverflowError): continue
            if not math.isfinite(pred) or pred <= 0: continue
            key = (day, h)
            if key in values: raise ValueError(f"duplicate monthly HistGB prediction for {pid}/{day}/h{h}")
            values[key] = pred
        result[pid] = values
    if set(result) != set(PRODUCTS): raise ValueError("monthly HistGB predictions must cover every product")
    return result


def _model(): return modeling._new_estimator("hist_gb")
def _metrics(actual,pred): return dh._metrics(actual,pred)
def _dm(diff,bw): return dh._dm(diff,bw)
def _holm(rows): return dh._holm(rows)


def _macro_metrics(paired_rows):
    fields=("candidate","monthlyPooledHistGb","directRidge","persistence")
    output={}; counts={}
    for field in fields:
        group_values=[]
        for p in PRODUCTS:
            for h in HORIZONS:
                rows=[r for r in paired_rows if r["product"]==p and r["horizon"]==h]
                group_values.append(_metrics([r["actual"] for r in rows],[r[field] for r in rows])["mape"])
        counts[field]=sum(x is not None for x in group_values)
        output[field]=sum(group_values)/50 if len(group_values)==50 and all(x is not None for x in group_values) else None
    return output,counts


def _coverage_loss(control_dates, candidate_dates):
    return sorted(set(control_dates) - set(candidate_dates))


def _joint_conclusions(comparisons, coverage):
    indexed={(r["product"],r["horizon"],r["comparator"],r["loss"]):r for r in comparisons}
    conclusions=[]
    for p in PRODUCTS:
      for h in HORIZONS:
        cov=coverage[f"{p}/{h}"]
        required=[indexed[(p,h,c,l)] for c in COMPARATORS for l in LOSSES]
        tests=[r["dm"]["9"] for r in required]
        enough=all(t["status"]=="tested" and t["n"]>=30 for t in tests)
        lower=all(r["candidateMetric"] is not None and r["comparatorMetric"] is not None and r["candidateMetric"]<r["comparatorMetric"] for r in required)
        sig=all(t.get("holmSignificant") and t.get("meanLossDifference") is not None and t["meanLossDifference"]<0 for t in tests)
        cov_ok=all(not cov["candidateCoverageLossDates"][c] for c in COMPARATORS)
        adverse=any(t["status"]=="tested" and t.get("holmSignificant") and t.get("meanLossDifference") is not None and t["meanLossDifference"]>0 for t in tests)
        status="inference_unavailable" if not enough else "improvement" if lower and sig and cov_ok else "worsening_or_mixed" if adverse else "inconclusive"
        conclusions.append({"product":p,"horizon":h,"conclusion":status,"inferenceAvailable":enough,"allSixHac9HolmTestsImprove":sig,"allSixPairedMetricsImprove":lower,"coverageNondeclining":cov_ok,"requiredTests":[{"comparator":r["comparator"],"loss":r["loss"],"hac9":r["dm"]["9"]} for r in required]})
    return conclusions


def build(prices, forecasts, ridge_report, *, hashes=None):
    series_by=dh._observations(prices); pooled,_,persistence=dh._forecast_maps(forecasts)
    hist=_monthly_hist_gb_all_horizons(forecasts); ridge=_direct_ridge_reference(ridge_report)
    prediction_rows=[]; fits={}; statuses={}
    for product in PRODUCTS:
        series=series_by[product]; cache={}
        days=[]; by_origin=defaultdict(list); observation_days=tuple(series)
        for target,(actual,_) in series.items():
            for h in HORIZONS:
                origin=target-dt.timedelta(days=h)
                if origin<dt.date(2014,1,1) or origin>EVAL_END:continue
                feature=_feature(series,origin,target,h,cache)
                if feature is None:continue
                source,vec=feature
                days.append((origin,target,source,actual,vec,math.log(actual)-math.log(series[source][0]),h))
        days.sort(key=lambda r:(r[0],r[1],r[6]))
        # Horizon-local arrays and monthly evaluation partitions avoid repeated scans.
        for h in HORIZONS:
            rows=sorted((r for r in days if r[6]==h),key=lambda r:r[1])
            target_dates=[r[1] for r in rows]
            evals=[(t,t-dt.timedelta(days=h)) for t in series if EVAL_START<=t<=EVAL_END and t-dt.timedelta(days=h)>=EVAL_START]
            months=defaultdict(list)
            for t,o in evals:months[(o.year,o.month)].append((t,o))
            audit={"fitAttempts":0,"fitSuccesses":0,"months":[]}
            sample_by_key={(r[1],r[0]):r for r in rows}
            for (year,month), targets in sorted(months.items()):
                fit_origin=min(o for _,o in targets); end=bisect_left(target_dates,fit_origin)
                train=rows[:end]; distinct=len({r[1] for r in train})
                train_end=max((r[1] for r in train),default=None)
                entry={"originMonth":f"{year:04d}-{month:02d}","fitOrigin":fit_origin.isoformat(),"fitTrainTargetEnd":train_end.isoformat() if train_end else None,"fitDistinctTargets":distinct,"fitSamples":len(train),"status":None}
                if distinct<MIN_TRAIN_DAYS:
                    entry["status"]="insufficient_training"
                    for t,_ in targets:statuses[(product,h,t)]="insufficient_training"
                    audit["months"].append(entry);continue
                audit["fitAttempts"]+=1
                try:
                    import numpy as np
                    from threadpoolctl import threadpool_limits
                    X=np.asarray([r[4] for r in train]); y=np.asarray([r[5] for r in train]); valid=[]
                    for t,o in targets:
                        r=sample_by_key.get((t,o))
                        if r is not None:valid.append(r)
                    with direct_tree._thread_limit_context(threadpool_limits):
                        model=_model(); model.fit(X,y)
                        predicted=model.predict(np.asarray([r[4] for r in valid])) if valid else []
                    built=[]
                    for r,delta in zip(valid,predicted):
                        price,clipped=modeling._inverse_log_ratio(series[r[2]][0],float(delta))
                        built.append({"product":product,"target":r[1].isoformat(),"origin":r[0].isoformat(),"horizon":h,"sourceDate":r[2].isoformat(),"actual":r[3],"predicted":price,"fitOrigin":fit_origin.isoformat(),"fitTrainTargetEnd":train_end.isoformat(),"fitDistinctTargets":distinct,"fitSamples":len(train),"clipped":clipped})
                    prediction_rows.extend(built); audit["fitSuccesses"]+=1;entry["status"]="fitted"
                    for t,o in targets:statuses[(product,h,t)]="candidate_available" if (t,o) in sample_by_key else "feature_unavailable_or_source_too_old"
                except Exception as exc:
                    entry["status"]="fit_failed";entry["errorType"]=type(exc).__name__
                    for t,_ in targets:statuses[(product,h,t)]="fit_failed"
                audit["months"].append(entry)
            fits[f"{product}/{h}"]=audit

    predmap={(r['product'],dt.date.fromisoformat(r['target']),r['horizon']):r for r in prediction_rows}
    paired=[]; summaries=[]; coverage={}; exclusions={}
    for product in PRODUCTS:
      series=series_by[product]; observation_days=tuple(series)
      for h in HORIZONS:
        eligible=sorted(t for t in series if EVAL_START<=t<=EVAL_END and t-dt.timedelta(days=h)>=EVAL_START)
        rows=[]; why={}
        for t in eligible:
            key=(t,h); cand=predmap.get((product,t,h)); histp=hist[product].get(key); ridgep=ridge.get((product,h,t)); persist=persistence[product].get(key)
            source_idx=bisect_right(observation_days,t-dt.timedelta(days=h))-1
            source=observation_days[source_idx] if source_idx>=0 else None
            reasons=[]
            for x,name in ((cand,"candidate"),(histp,"monthly_control"),(ridgep,"direct_ridge"),(persist,"persistence")):
                if x is None:reasons.append(name+"_unavailable")
            if source is None or (t-dt.timedelta(days=h)-source).days>dashboard.MAX_SOURCE_AGE_DAYS:reasons.append("source_older_than_max_age")
            if reasons:why[t.isoformat()]=reasons;continue
            rows.append({"product":product,"target":t.isoformat(),"horizon":h,"actual":series[t][0],"candidate":cand['predicted'],"monthlyPooledHistGb":histp,"directRidge":ridgep,"persistence":persist})
        paired.extend(rows); avail={"candidate":sorted(t.isoformat() for t in eligible if (product,t,h) in predmap)}
        compdates={c:sorted(t.isoformat() for t in eligible if ((hist[product].get((t,h)) if c==COMPARATORS[0] else ridge.get((product,h,t)) if c==COMPARATORS[1] else persistence[product].get((t,h))) is not None)) for c in COMPARATORS}
        coverage[f"{product}/{h}"]={"eligibleEvaluationDates":[t.isoformat() for t in eligible],"candidateAvailableDates":sorted(r['target'] for r in prediction_rows if r['product']==product and r['horizon']==h),"comparatorAvailableDates":compdates,"commonPairedDates":sorted(r['target'] for r in rows),"candidateCoverageLossDates":{c:_coverage_loss(ds,avail['candidate']) for c,ds in compdates.items()},"eligibleDateCount":len(eligible),"commonPairedCount":len(rows),"exclusionReasons":why}
        exclusions[f"{product}/{h}"]=why
        for c in COMPARATORS:
          for loss in LOSSES:
            diffs={}; a=[]; cp=[]; op=[]
            field={COMPARATORS[0]:"monthlyPooledHistGb",COMPARATORS[1]:"directRidge",COMPARATORS[2]:"persistence"}[c]
            for r in rows:
                y=r['actual'];x=r['candidate'];z=r[field];d=abs(y-x)-abs(y-z) if loss=='mae' else 100*(abs(y-x)/y-abs(y-z)/y)
                diffs[dt.date.fromisoformat(r['target'])]=d;a.append(y);cp.append(x);op.append(z)
            summaries.append({"product":product,"horizon":h,"comparator":c,"loss":loss,"n":len(rows),"candidateMetric":_metrics(a,cp)[loss],"comparatorMetric":_metrics(a,op)[loss],"dm":{str(b):_dm(diffs,b) for b in BANDWIDTHS}})
    for b in BANDWIDTHS:
        family=[x['dm'][str(b)] for x in summaries]
        if len(family)!=FAMILY_SIZE:raise AssertionError("Holm family size mismatch")
        _holm(family)
    joint=_joint_conclusions(summaries,coverage)
    # Macro equal-weight mean across exactly 50 product/horizon MAPE groups.
    macro,macro_group_counts=_macro_metrics(paired)
    strata=[]
    for p in PRODUCTS:
      for h in HORIZONS:
       subset=[r for r in paired if r['product']==p and r['horizon']==h]
       for kind,keyfn in (("year",lambda d:d.year),("season",lambda d:(d.month-1)//3+1)):
        groups=defaultdict(list)
        for r in subset:groups[keyfn(dt.date.fromisoformat(r['target']))].append(r)
        for val,g in sorted(groups.items()):strata.append({"product":p,"horizon":h,"stratum":kind,"value":val,"n":len(g),**{field:_metrics([r['actual'] for r in g],[r[field] for r in g]) for field in ("candidate","monthlyPooledHistGb","directRidge","persistence")}})
    return {"schemaVersion":1,"protocol":{"id":"weekday-features-direct-hist-gb-2026-10-06","evaluation":[EVAL_START.isoformat(),EVAL_END.isoformat()],"horizons":list(HORIZONS),"featureAdditions":["same-weekday 56-calendar-day median log price","matching observation count 0..8","days since most recent matching weekday (57 when absent)","latest log price minus same-weekday median"],"historyInterval":"origin-56 days <= observation < origin","monthlyRefit":"first eligible evaluation origin per origin month","trainTargetStrictlyBeforeFitOrigin":True,"minimumDistinctTrainingTargets":MIN_TRAIN_DAYS,"estimator":"modeling._new_estimator('hist_gb')","comparators":list(COMPARATORS),"losses":list(LOSSES),"dm":{"calendarBartlettHACDays":list(BANDWIDTHS),"twoSided":True,"holmAlpha":.05,"familySizePerBandwidth":FAMILY_SIZE},"macroMAPE":"unweighted mean of exactly 50 product/horizon MAPEs; null unless all groups are available"},"hashes":hashes or {},"fitAudits":fits,"fitTotals":{"fitAttempts":sum(x['fitAttempts'] for x in fits.values()),"fitSuccesses":sum(x['fitSuccesses'] for x in fits.values())},"coverage":coverage,"exclusions":exclusions,"macroMetrics":macro,"macroMetricGroupCounts":macro_group_counts,"jointConclusions":joint,"limitations":"Retrospective use of revised prices; first publication times are unknown. This is not prospective or official-target evidence.","predictions":prediction_rows,"pairedRows":paired,"comparisons":summaries,"strata":strata}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()
def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--prices',type=Path,default=INPUT_PRICES);ap.add_argument('--predictions',type=Path,default=INPUT_PREDICTIONS);ap.add_argument('--ridge-report',type=Path,default=INPUT_RIDGE_REPORT);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--allow-nonfrozen-inputs',action='store_true');a=ap.parse_args(argv)
    if a.output.resolve() in (a.prices.resolve(),a.predictions.resolve(),a.ridge_report.resolve()):ap.error('output must not replace an input')
    hashes={'pricesSha256':sha(a.prices),'predictionsSha256':sha(a.predictions),'directRidgeReportSha256':sha(a.ridge_report)}
    expected={'pricesSha256':PRICES_SHA256,'predictionsSha256':PREDICTIONS_SHA256,'directRidgeReportSha256':RIDGE_REPORT_SHA256}
    if not a.allow_nonfrozen_inputs and hashes!=expected:ap.error('input SHA-256 does not match frozen protocol')
    result=build(json.loads(a.prices.read_text()),json.loads(a.predictions.read_text()),json.loads(a.ridge_report.read_text()),hashes=hashes)
    a.output.parent.mkdir(parents=True,exist_ok=True);tmp=a.output.with_suffix(a.output.suffix+'.tmp');tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(a.output);return 0
if __name__=='__main__':raise SystemExit(main())
