"""Frozen forward search over causal radish price and volume features."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import tempfile
import time
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any

from . import modeling, volume_guard

# Fixed single-thread linear algebra keeps daily refits bounded and reproducible.
for _thread_var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_var] = "1"

HORIZONS = tuple(range(1, 11))
SCORED_HORIZONS = (1, 5, 10)
BANDWIDTHS = ((9, "hac9"), (28, "hac28"))
ALPHA = 10.0
MIN_TRAIN_DAYS = 90
VAL_START, VAL_END = dt.date(2018, 1, 1), dt.date(2022, 12, 31)
TEST_START, TEST_END = dt.date(2023, 1, 11), dt.date(2026, 10, 5)
EVENT_DAY = dt.date(2023, 1, 21)
FEATURES = [
    ("latest_log_price", "Latest eligible log price"),
    ("sma7_log_price", "7 calendar-day simple mean of observed log prices"),
    ("sma28_log_price", "28 calendar-day simple mean of observed log prices"),
    ("ewma7_log_price", "Observed log prices with 7-day half-life"),
    ("ewma28_log_price", "Observed log prices with 28-day half-life"),
    ("median28_log_price", "28 calendar-day median of observed log prices"),
    ("sma7_minus_sma28", "7-day minus 28-day log-price means"),
    ("slope28_log_price", "OLS slope of observed log price against calendar day in 28 days"),
    ("change7_log_price", "Latest log price minus latest eligible log price at least 7 days old"),
    ("change28_log_price", "Latest log price minus latest eligible log price at least 28 days old"),
    ("std28_log_returns", "Population SD of consecutive observed log returns within 28 days"),
    ("range28_log_price", "28-day log-price range"),
    ("sma7_log_volume", "7 calendar-day mean log of positive observed net kg"),
    ("sma7_minus_sma28_log_volume", "7-day minus 28-day observed mean log net kg"),
    ("season1_log_price", "Prior-year target-date +/-14-day log-price mean"),
    ("season2_log_price", "Mean of prior- and two-years-prior target-date +/-14-day log-price means"),
]
IDX = {name: i for i, (name, _) in enumerate(FEATURES)}
_EWMA_PREFIXES = {}


def _series(product: dict[str, Any]):
    raw = modeling._observed_series(product)
    flags = volume_guard.quarantine_dates(raw, price_ratio_threshold=4.0)
    return raw, volume_guard.quality_series(raw, flags), flags


def _features(series, origin: dt.date, target: dt.date, ordered_days=None):
    """All values use observations strictly before origin."""
    ordered_days = tuple(series) if ordered_days is None else ordered_days
    end = bisect_left(ordered_days, origin)
    if end == 0:
        return None
    source_day=ordered_days[end-1]
    if (origin-source_day).days > modeling.MAX_SOURCE_AGE_DAYS:
        return None
    src = source_day
    recent_start=bisect_left(ordered_days,origin-dt.timedelta(days=28),0,end)
    recent_days=ordered_days[recent_start:end]
    logs = [math.log(series[d][0]) for d in recent_days]
    vals: list[float | None] = [logs[-1]]
    for window in (7, 28):
        start=bisect_left(ordered_days,origin-dt.timedelta(days=window),0,end)
        xs = [math.log(series[d][0]) for d in ordered_days[start:end]]
        vals.append(sum(xs)/len(xs) if xs else None)
        if window == 7: x7 = xs
        else: x28 = xs
    vals.append(_ewma(series, ordered_days, end, origin, 7))
    vals.append(_ewma(series, ordered_days, end, origin, 28))
    vals.append(_median(x28))
    vals.append((sum(x7)/len(x7)-sum(x28)/len(x28)) if x7 and x28 else None)
    recent = [(d, math.log(series[d][0])) for d in recent_days]
    vals.append(_slope(recent))
    for lag in (7, 28):
        prev_idx=bisect_right(ordered_days,origin-dt.timedelta(days=lag),0,end)-1
        vals.append(logs[-1]-math.log(series[ordered_days[prev_idx]][0]) if prev_idx>=0 else None)
    returns = [math.log(series[b][0]/series[a][0]) for a,b in zip(recent_days, recent_days[1:])]
    vals.append(_sd(returns) if returns else None)
    vals.append(max(x28)-min(x28) if x28 else None)
    for window in (7, 28):
        start=bisect_left(ordered_days,origin-dt.timedelta(days=window),0,end)
        vs = [math.log(series[d][1]) for d in ordered_days[start:end]]
        if window == 7: v7 = vs
        else: v28 = vs
    vals.append(sum(v7)/len(v7) if v7 else None)
    vals.append(sum(v7)/len(v7)-sum(v28)/len(v28) if v7 and v28 else None)
    seasonal = []
    for year_back in (1, 2):
        try: center = target.replace(year=target.year-year_back)
        except ValueError: center = target.replace(year=target.year-year_back, day=28)
        lo=bisect_left(ordered_days,center-dt.timedelta(days=14),0,end)
        hi=bisect_right(ordered_days,center+dt.timedelta(days=14),0,end)
        windowdays = ordered_days[lo:hi]
        seasonal.append(sum(math.log(series[d][0]) for d in windowdays)/len(windowdays)
                        if len(windowdays) >= 5 else None)
    vals.extend((seasonal[0], (sum(seasonal)/2 if all(v is not None for v in seasonal) else None)))
    if len(vals) != len(FEATURES):
        raise AssertionError(f"feature count mismatch {len(vals)}")
    return tuple(vals), src


def _ewma(series, ordered_days, end, origin, half_life):
    # Half-life defines decay; absent a separate window limit, use the full causal history.
    entry=_EWMA_PREFIXES.get(id(series))
    if entry is None or entry[0] is not series or entry[1] is not ordered_days:
        anchor=ordered_days[-1] if ordered_days else origin
        prefixes={}
        for hl in (7,28):
            sw=[0.0];sy=[0.0]
            for day in ordered_days:
                weight=2 ** ((day-anchor).days/hl)
                sw.append(sw[-1]+weight);sy.append(sy[-1]+weight*math.log(series[day][0]))
            prefixes[hl]=(sw,sy)
        entry=(series,ordered_days,prefixes);_EWMA_PREFIXES[id(series)]=entry
    sw,sy=entry[2][half_life]
    return sy[end]/sw[end] if end and sw[end] else None


def _median(xs):
    if not xs: return None
    ys=sorted(xs); n=len(ys)
    return ys[n//2] if n%2 else (ys[n//2-1]+ys[n//2])/2


def _sd(xs):
    m=sum(xs)/len(xs); return math.sqrt(sum((x-m)**2 for x in xs)/len(xs))


def _slope(points):
    if len(points)<2:return None
    x=[(d-points[0][0]).days for d,_ in points]; y=[v for _,v in points]
    mx=sum(x)/len(x);my=sum(y)/len(y);den=sum((a-mx)**2 for a in x)
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/den if den else None


def _build_samples(series):
    samples=[]; cache={};days=tuple(series)
    for target,(price,_) in series.items():
        for h in HORIZONS:
            origin=target-dt.timedelta(days=h)
            key=(origin,target)
            if key not in cache: cache[key]=_features(series,origin,target,days)
            row=cache[key]
            if row is None: continue
            vals,src=row
            samples.append({"target":target,"origin":origin,"h":h,"x":vals,
                            "y":math.log(price/series[src][0]),"src":src,
                            "srcPrice":series[src][0],"actual":price})
    return samples


def _fit(X,y):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    m=make_pipeline(StandardScaler(),Ridge(alpha=ALPHA));m.fit(X,y);return m


def _validation_fold(month, samples):
    end=(dt.date(month.year+1,1,1)-dt.timedelta(days=1) if month.month==12 else dt.date(month.year,month.month+1,1)-dt.timedelta(days=1))
    valid=[s for s in samples if month<=s["target"]<=end and s["h"] in SCORED_HORIZONS
           and all(v is not None for v in s["x"])]
    earliest_origin=min((s['origin'] for s in valid),default=None)
    train=[s for s in samples if earliest_origin is not None and dt.date(2016,1,1)<=s["target"]<earliest_origin
           and all(v is not None for v in s["x"])]
    return train,valid,earliest_origin


def _select_validation(samples):
    import numpy as np
    folds=[]
    months=[]
    month=dt.date(VAL_START.year,VAL_START.month,1)
    while month<=VAL_END:
        months.append(month)
        month=(dt.date(month.year+1,1,1) if month.month==12 else dt.date(month.year,month.month+1,1))
    for month in months:
        train,valid,_=_validation_fold(month,samples)
        if len({s['target'] for s in train})>=MIN_TRAIN_DAYS and valid:
            folds.append((np.asarray([s['x']+(float(s['h']),) for s in train],dtype=float),np.asarray([s['y'] for s in train],dtype=float),
                          np.asarray([s['x']+(float(s['h']),) for s in valid],dtype=float),valid))
    selected=[]; steps=[]
    if not folds:
        return [], [{"status":"selection_unavailable_no_monthly_common_validation_folds"}], 0
    for step in range(5):
        scored=[]
        for j,(name,_) in enumerate(FEATURES):
            if j in selected:continue
            inds=selected+[j]+[len(FEATURES)]; errors={h:[] for h in SCORED_HORIZONS}
            for xt,yt,xv,valid in folds:
                model=_fit(xt[:,inds],yt)
                preds=model.predict(xv[:,inds])
                for s,p in zip(valid,preds):
                    if s['h'] in errors:
                        errors[s['h']].append((float(p), s))
            missing_horizons=[h for h,rows in errors.items() if not rows]
            if missing_horizons:
                return [],[{"status":"selection_unavailable_missing_common_scored_horizon",
                            "horizonsWithoutValidationPairs":missing_horizons}],len(folds)
            maps={}; maes={}
            for h,rows in errors.items():
                abses=[abs(s['srcPrice']*math.exp(p)-s['actual'])/s['actual']*100 for p,s in rows]
                maps[h]=sum(abses)/len(abses) if abses else math.inf
                maes[h]=sum(abs(s['srcPrice']*math.exp(p)-s['actual']) for p,s in rows)/len(rows) if rows else math.inf
            score=sum(maps.values())/len(maps)
            scored.append((score,sum(maes.values())/len(maes),j,inds,maps,maes))
        scored.sort(key=lambda z:(z[0],z[1],z[2]))
        winner=scored[0];selected.append(winner[2])
        steps.append({"step":step+1,"selectedFeature":FEATURES[winner[2]][0],"scoreMeanMAPEPercent":winner[0],
                      "maeByHorizon":{str(h):winner[5][h] for h in SCORED_HORIZONS},
                      "candidateScores":[{"feature":FEATURES[z[2]][0],"meanMAPEPercent":z[0],
                                          "mapeByHorizon":{str(h):z[4][h] for h in SCORED_HORIZONS},
                                          "maeByHorizon":{str(h):z[5][h] for h in SCORED_HORIZONS}} for z in scored]})
    # Decide whether the fifth feature meets the fixed MAPE and no MAE >1% worsening gate.
    if len(steps)>=5:
        prev=steps[3]; fifth=steps[4]
        mape_gain=prev['scoreMeanMAPEPercent']-fifth['scoreMeanMAPEPercent']
        mae_ok=all(fifth['maeByHorizon'][str(h)] <= prev['maeByHorizon'][str(h)]*1.01 for h in SCORED_HORIZONS)
        steps[4]['fifthFeatureGate']={"mapeImprovementPercentagePoints":mape_gain,"maeNoHorizonOver1PercentWorse":mae_ok,
                                      "included":mape_gain>=0.1 and mae_ok}
        if not (mape_gain>=0.1 and mae_ok):selected=selected[:4]
    return selected,steps,len(folds)


def _candidate_samples(series):
    samples=[]; cache={};days=tuple(series)
    for target,(price,_) in series.items():
        for h in HORIZONS:
            origin=target-dt.timedelta(days=h);key=(origin,target)
            if key not in cache:cache[key]=_features(series,origin,target,days)
            item=cache[key]
            if item is None:continue
            x,src=item
            samples.append({"target":target,"origin":origin,"h":h,"x":x,
                            "y":math.log(price/series[src][0]),"src":src,
                            "srcPrice":series[src][0],"actual":price})
    return samples


def _strict_full_row(series, origin, target, horizon, cache):
    days=cache.setdefault("series_days",tuple(series))
    logs=cache.setdefault("series_log_prices",tuple(math.log(series[d][0]) for d in days))
    hi=bisect_left(days,origin);source_idx=hi-1
    if source_idx<0:return None
    source=days[source_idx];age=(origin-source).days
    if age>modeling.MAX_SOURCE_AGE_DAYS:return None
    values=[logs[source_idx],float(horizon),float(age)]
    for lag in modeling.LAGS:
        cutoff=origin-dt.timedelta(days=lag);idx=bisect_right(days,cutoff,0,hi)-1
        if idx<0:values.extend((0.0,float(lag+1),0.0))
        else:values.extend((logs[idx],float((cutoff-days[idx]).days),1.0))
    for window in modeling.WINDOWS:
        lo=bisect_left(days,origin-dt.timedelta(days=window),0,hi)
        selected=logs[lo:hi];volumes=[series[d][1] for d in days[lo:hi]];count=len(selected)
        if count:
            mean=sum(selected)/count;variance=sum((v-mean)**2 for v in selected)/count
            std=math.sqrt(max(0.0,variance));minimum=min(selected);maximum=max(selected)
            log_volume=math.log1p(max(0.0,sum(volumes)))
        else:mean=std=minimum=maximum=log_volume=0.0
        values.extend((mean,std,minimum,maximum,float(count),count/window,log_volume))
    weekday=2*math.pi*target.weekday()/7
    year_days=366 if modeling._is_leap(target.year) else 365
    year=2*math.pi*(target.timetuple().tm_yday-1)/year_days
    values.extend((math.sin(weekday),math.cos(weekday),math.sin(year),math.cos(year)))
    if len(values)!=len(modeling.FEATURE_NAMES) or not all(math.isfinite(v) for v in values):
        raise ValueError("strict full-model feature generation failed")
    return source,tuple(values)


def _price_from_log_ratio(source_price, predicted_log_ratio):
    return source_price*math.exp(max(-700.0,min(700.0,float(predicted_log_ratio))))


def _event_slices(day, input_exposed=False):
    labels=[]
    if day==EVENT_DAY:labels.append("event_target")
    if input_exposed:labels.append("event_input_followup")
    if day>=dt.date(2024,1,1):labels.append("from_2024")
    return labels


def _strict_source(series, origin):
    days=tuple(series);idx=bisect_left(days,origin)-1
    if idx<0:return None
    day=days[idx]
    return day if (origin-day).days<=modeling.MAX_SOURCE_AGE_DAYS else None


def _event_input_exposed(raw, without_event, origin, target, horizon, selected_indices, comparator):
    """Whether the unguarded event could change selected/comparator inputs at this origin."""
    if EVENT_DAY not in raw or EVENT_DAY>=origin:return False
    selected_changed=False
    if selected_indices:
        if any(i in selected_indices for i in (IDX['ewma7_log_price'],IDX['ewma28_log_price'])):
            selected_changed=True
        else:
            before=_features(raw,origin,target)
            after=_features(without_event,origin,target)
            if before is not None and after is not None:
                selected_changed=before[1]!=after[1] or any(before[0][i]!=after[0][i] for i in selected_indices)
            elif before != after:selected_changed=True
    if comparator=="persistence":
        comparator_changed=_strict_source(raw,origin)!=_strict_source(without_event,origin)
    elif comparator=="full_ridge":
        full_before=_strict_full_row(raw,origin,target,horizon,{})
        full_after=_strict_full_row(without_event,origin,target,horizon,{})
        comparator_changed=full_before!=full_after
    else:return selected_changed
    return selected_changed or comparator_changed


def _full_samples(series):
    out=[];cache={}
    for target,(price,_) in series.items():
        for h in HORIZONS:
            origin=target-dt.timedelta(days=h);key=(origin,target,h)
            if key not in cache:cache[key]=_strict_full_row(series,origin,target,h,cache)
            row=cache[key]
            if row is None:continue
            src,x=row
            out.append({"target":target,"origin":origin,"h":h,"x":x,
                        "y":math.log(price/series[src][0]),"src":src,
                        "srcPrice":series[src][0],"actual":price})
    return out


def _dm(rows, bandwidth):
    # Keep identical semantics to the established radish guard experiment.
    return volume_guard.dm_calendar_hac(rows, bandwidth)


def _availability_report(samples, start, end):
    rows=[s for s in samples if start<=s['target']<=end]
    complete=[s for s in rows if all(v is not None for v in s['x'])]
    missing={FEATURES[j][0]:sum(s['x'][j] is None for s in rows) for j in range(len(FEATURES))}
    by_year={};by_season={}
    for s in rows:
        if all(v is not None for v in s['x']):continue
        y=str(s['target'].year);by_year[y]=by_year.get(y,0)+1
        season=_season(s['target']);by_season[season]=by_season.get(season,0)+1
    return {"candidateRows":len(rows),"commonCompleteRows":len(complete),"incompleteRows":len(rows)-len(complete),
            "incompleteByYear":by_year,"incompleteBySeason":by_season,"missingFeatureValueCount":missing,
            "volumeFeatureMissingRows":max(missing['sma7_log_volume'],missing['sma7_minus_sma28_log_volume'])}


def _canonical_sha256(value):
    blob=json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _feature_correlations(samples, start, end):
    import numpy as np
    rows=[s['x'] for s in samples if start<=s['target']<=end and all(v is not None for v in s['x'])]
    if len(rows)<2:return {"n":len(rows),"matrix":[]}
    matrix=np.corrcoef(np.asarray(rows,dtype=float),rowvar=False)
    return {"n":len(rows),"matrix":[[None if not math.isfinite(v) else float(v) for v in row] for row in matrix]}


def analyze(source: dict[str,Any], input_sha: str):
    product=next((p for p in source.get("products",[]) if p.get("id")=="radish"),None)
    if product is None:raise ValueError("dashboard has no radish product")
    raw,eligible,flags=_series(product)
    if not raw:raise ValueError("radish observations are empty")
    all_cand=_candidate_samples(eligible)
    full_rows=_full_samples(eligible)
    selected,steps,nfolds=_select_validation(all_cand)
    event_quarantined=any(row.get("date")==EVENT_DAY.isoformat() for row in flags)
    without_event=dict(raw)
    if event_quarantined:without_event.pop(EVENT_DAY,None)
    test_targets=[d for d in raw if TEST_START<=d<=TEST_END]
    horizons=SCORED_HORIZONS
    methods={"selected":{},"full_ridge":{},"persistence":{}}
    candidate_cache={}; cand_days=tuple(eligible); full_feature_cache={}
    # Refit at each daily origin; every supervised target date is strictly earlier.
    for d in test_targets:
        for h in horizons:
            origin=d-dt.timedelta(days=h)
            if d<TEST_START:continue
            key=(origin, d, h)
            ci=_features(eligible,origin,d,cand_days)
            fi=_strict_full_row(eligible,origin,d,h,full_feature_cache)
            if ci is None or fi is None:continue
            selected_src=ci[1]; full_src=fi[0]
            rec={"target":d,"origin":origin,"h":h,"actual":raw[d][0],
                 "selectedSource":selected_src,"selectedSourcePrice":eligible[selected_src][0],
                 "fullSource":full_src,"fullSourcePrice":eligible[full_src][0]}
            methods["persistence"][key]={**rec,"predicted":eligible[full_src][0]}
            for method,inds in (("selected",selected),("full_ridge",list(range(len(modeling.FEATURE_NAMES))))):
                if method=="selected" and not selected:continue
                rows=all_cand if method=="selected" else full_rows
                idxs=inds if method=="selected" else None
                fitkey=(method,origin)
                if fitkey not in candidate_cache:
                    train=[s for s in rows if s['target']<origin and
                           (method!="selected" or (s['target']>=dt.date(2016,1,1) and all(v is not None for v in s['x'])))]
                    if len({s['target'] for s in train})<MIN_TRAIN_DAYS:
                        candidate_cache[fitkey]=None
                        continue
                    xs=[[s['x'][j] for j in idxs]+[float(s['h'])] for s in train] if idxs is not None else [s['x'] for s in train]
                    candidate_cache[fitkey]=_fit(xs,[s['y'] for s in train])
                model=candidate_cache[fitkey]
                if model is None:continue
                x=ci[0] if method=="selected" else fi[1]
                if method=="selected" and any(x[j] is None for j in idxs):continue
                x=([x[j] for j in idxs]+[float(h)] if idxs is not None else x)
                source_price=rec['selectedSourcePrice'] if method=="selected" else rec['fullSourcePrice']
                pred=_price_from_log_ratio(source_price,model.predict([x])[0])
                methods[method][key]={**rec,"predicted":pred}
    common={h:set.intersection(*(set(k for k in methods[m] if k[2]==h) for m in methods)) for h in horizons}
    comparisons=[]; tests_by_band={bw:[] for bw,_ in BANDWIDTHS}; prediction_rows=[]
    for h in horizons:
        keys=sorted(common[h],key=lambda k:k[1])
        for key in keys:
            rec=methods['selected'][key]
            prediction_rows.append({"target":rec['target'].isoformat(),"origin":rec['origin'].isoformat(),"horizon":h,
                "actual":rec['actual'],"selectedSourceDate":rec['selectedSource'].isoformat(),
                "selectedSourcePrice":rec['selectedSourcePrice'],"fullSourceDate":rec['fullSource'].isoformat(),
                "fullSourcePrice":rec['fullSourcePrice'],
                "selected":methods['selected'][key]['predicted'],"fullRidge":methods['full_ridge'][key]['predicted'],
                "persistence":methods['persistence'][key]['predicted']})
        for comparator in ("full_ridge","persistence"):
            loss_rows={"absolute_error":[],"absolute_percentage_error":[]}
            metrics={name:{"mae":0.0,"mapePercent":0.0,"n":len(keys)} for name in ("selected",comparator)}
            for key in keys:
                y=methods['selected'][key]['actual']; p=methods['selected'][key]['predicted']; q=methods[comparator][key]['predicted']
                ae=abs(p-y);ape=ae/y*100
                qe=abs(q-y);qpe=qe/y*100
                metrics['selected']['mae']+=ae;metrics['selected']['mapePercent']+=ape
                metrics[comparator]['mae']+=qe;metrics[comparator]['mapePercent']+=qpe
                loss_rows['absolute_error'].append((key[1],ae-qe))
                loss_rows['absolute_percentage_error'].append((key[1],ape-qpe))
            for v in metrics.values():
                if v['n']:
                    v['mae']/=v['n'];v['mapePercent']/=v['n']
                else:v['mae']=v['mapePercent']=None
            yr=[];season=[]
            byyear={};byseason={};special={"event_target":[],"event_input_followup":[],"from_2024":[]}
            for key in keys:
                r=methods['selected'][key];y=r['actual'];p=r['predicted'];q=methods[comparator][key]['predicted']
                errors=(abs(p-y),abs(q-y),abs(p-y)/y*100,abs(q-y)/y*100)
                for dest,k in ((byyear,str(r['target'].year)),(byseason,_season(r['target']))):
                    dest.setdefault(k,[]).append(errors)
                exposed=(event_quarantined and _event_input_exposed(
                    raw,without_event,key[0],key[1],h,selected,
                    "full_ridge" if comparator=="full_ridge" else "persistence"))
                for special_slice in _event_slices(r['target'],exposed):special[special_slice].append(errors)
            slice_stats={"year":_slice_report(byyear),"season":_slice_report(byseason),
                         **{name:_slice_report({"all":rows}) for name,rows in special.items()}}
            comparison={"horizon":h,"candidate":"selected","comparator":comparator,"n":len(keys),"metrics":metrics,
                        "pairedPredictions":len(keys),"slices":slice_stats,"tests":{}}
            for loss,rows in loss_rows.items():
                comp={"selected_full_ridge" if comparator=="full_ridge" else "selected_persistence":comparison}
                comparison['tests'][loss]={}
                for bw,label in BANDWIDTHS:
                    test=_dm(rows,bw);test.update({"comparison":comparison["candidate"]+"_vs_"+comparator,"horizon":h,"loss":loss})
                    tests_by_band[bw].append(test);comparison['tests'][loss][label]=test
            comparisons.append(comparison)
    for bw,items in tests_by_band.items():volume_guard.holm_adjust(items)
    tests_lookup={(x['comparison'],x['horizon'],x['loss'],x['bandwidthCalendarDays']):x for xs in tests_by_band.values() for x in xs}
    for c in comparisons:
        for loss,groups in c['tests'].items():
            for bw,label in BANDWIDTHS:
                t=tests_lookup.get(("selected_vs_"+c['comparator'],c['horizon'],loss,bw))
                if t:groups[label]=t
        c['conclusion']=_conclusion(c)
    full_comparisons=[c for c in comparisons if c['comparator']=='full_ridge']
    all_sig=bool(full_comparisons) and all(
        c['metrics']['selected']['mae'] is not None and c['metrics']['full_ridge']['mae'] is not None and
        c['metrics']['selected']['mapePercent'] is not None and c['metrics']['full_ridge']['mapePercent'] is not None and
        c['metrics']['selected']['mae']<c['metrics']['full_ridge']['mae'] and
        c['metrics']['selected']['mapePercent']<c['metrics']['full_ridge']['mapePercent'] and
        all(c['tests'][loss][label].get('holmPValue') is not None and c['tests'][loss][label]['holmPValue']<.05
            for loss in ("absolute_error","absolute_percentage_error") for _,label in BANDWIDTHS) for c in full_comparisons)
    # Explicit severe-slice guard for any year/season with at least 90 paired targets.
    severe=[]
    for c in full_comparisons:
        for kind in ('year','season'):
            for label,s in c['slices'][kind].items():
                mae_increase=s['maeDelta']/s['comparatorMAE']*100 if s['comparatorMAE']>0 else math.inf
                if s['n']>=90 and (s['mapeDeltaPercentagePoints']>1 or mae_increase>5):
                    severe.append({"horizon":c['horizon'],"slice":kind,"label":label,"n":s['n'],
                                   "mapeDeltaPercentagePoints":s['mapeDeltaPercentagePoints'],"maeDelta":s['maeDelta'],
                                   "maeIncreasePercent":mae_increase if math.isfinite(mae_increase) else None})
    coverage_ok=all(sum(k[2]==h for k in methods['selected'])>=sum(k[2]==h for k in methods['full_ridge']) for h in horizons)
    promotion_pass=all_sig and coverage_ok and not severe
    final_origin=max((d-dt.timedelta(days=1) for d in test_targets),default=TEST_START)
    final_selected_train=[s for s in all_cand if selected and dt.date(2016,1,1)<=s['target']<final_origin and all(v is not None for v in s['x'])]
    final_full_train=[s for s in full_rows if s['target']<final_origin]
    selected_train_identity=[{"target":s['target'].isoformat(),"horizon":s['h'],"label":s['y'],"features":[s['x'][j] for j in selected]} for s in final_selected_train]
    full_train_identity=[{"target":s['target'].isoformat(),"horizon":s['h'],"label":s['y']} for s in final_full_train]
    eligible_identity=[{"date":d.isoformat(),"actual":v[0],"volumeKg":v[1]} for d,v in eligible.items()]
    input_dates=[d for d in raw if TEST_START<=d<=TEST_END]
    return {"protocolId":"radish-derived-feature-search-2026-10-06","inputSha256":input_sha,
        "periods":{"validationTarget":[VAL_START.isoformat(),VAL_END.isoformat()],"testTarget":[TEST_START.isoformat(),TEST_END.isoformat()],
                   "trainingFeatureHistoryStart":"2016-01-01","horizons":list(HORIZONS)},
        "guard":{"ruleVersion":"radish-low-volume-extreme-price-v1","flaggedDates":flags,
                 "rawObservationCount":len(raw),"eligibleObservationCount":len(eligible),
                 "eventTargetQuarantined":event_quarantined,
                 "guardedModelsNeverUseQuarantinedEvent":event_quarantined and EVENT_DAY not in eligible},
        "features":[{"id":n,"definition":d} for n,d in FEATURES],
        "selection":{"selectedFeatureIds":[FEATURES[i][0] for i in selected],"validationMonthlyFolds":nfolds,"steps":steps,
                     "validationAvailability":_availability_report(all_cand,VAL_START,VAL_END),
                     "candidateCorrelationPearson":_feature_correlations(all_cand,VAL_START,VAL_END),
                     "rule":"greedy 4 features; add fifth only if mean MAPE improves >=0.1 percentage points and no scored horizon MAE worsens >1%; fit includes fixed horizonDays context"},
        "test":{"commonTargetCountByHorizon":{str(h):len(common[h]) for h in horizons},"comparisons":comparisons,
                "pairedPredictions":prediction_rows,"holmFamilySizePerBandwidth":12,
                "inference":"two-sided DM normal approximation; Bartlett calendar HAC 9 primary and 28 sensitivity; Holm 12 each; under 30 pairs unsupported",
                "coverageByHorizon":{"selected":{str(h):sum(k[2]==h for k in methods['selected']) for h in horizons},"fullRidge":{str(h):sum(k[2]==h for k in methods['full_ridge']) for h in horizons},"persistence":{str(h):sum(k[2]==h for k in methods['persistence']) for h in horizons}},
                "availability":_availability_report(all_cand,TEST_START,TEST_END),
                "trainingHistoryDifference":{"selected":{"start":"2016-01-01","finalFitOrigin":final_origin.isoformat(),
                    "distinctTargetDays":len({s['target'] for s in final_selected_train}),"supervisedRows":len(final_selected_train)},
                    "guardedFullRidge":{"start":min((s['target'] for s in final_full_train),default=None).isoformat() if final_full_train else None,
                    "finalFitOrigin":final_origin.isoformat(),"distinctTargetDays":len({s['target'] for s in final_full_train}),"supervisedRows":len(final_full_train)},
                    "reason":"selected training follows the frozen 2016 start; guarded full Ridge uses all eligible history"},
                "promotionAssessment":{"eligibleForFutureReview":promotion_pass,"allHorizonsImproveBothLossesWithHAC9And28":all_sig,
                    "coverageNotReducedVsFullRidge":coverage_ok,"severeYearSeasonWorsening":severe,
                    "rule":"requires all 1/5/10 paired MAE and MAPE improvements vs guarded full Ridge significant under both HAC bandwidths, no coverage loss, and no >=90-pair year/season with MAPE >+1pp or MAE >+5%"}},
        "provenance":{"source":"saved outputs/dashboard.json radish observations only; no database/API access","inputSha256":input_sha,
                      "eligibleSeriesSha256":_canonical_sha256(eligible_identity),
                      "selectedFeatureIdsSha256":_canonical_sha256([FEATURES[i][0] for i in selected]),
                      "selectedTrainingSha256":_canonical_sha256(selected_train_identity),"fullTrainingSha256":_canonical_sha256(full_train_identity),
                      "rawTestTargetCount":len(input_dates),
                      "actualTargets":"raw actual including guard date; only guarded series used in model features and training labels",
                      "trainingRule":"sample target date strictly before refit origin; all candidate and full-Ridge feature observations strictly before each sample origin; validation labels start 2016 and precede the earliest monthly forecast origin",
                      "descriptiveSliceRanges":{"event_target":EVENT_DAY.isoformat(),
                         "event_input_followup":"Per origin/target/horizon; marked only where unguarded event changes a selected or comparator input/source",
                         "from_2024":["2024-01-01",TEST_END.isoformat()]},
                      "timingSeconds":None}}


def _season(day):
    return {12:"winter",1:"winter",2:"winter",3:"spring",4:"spring",5:"spring",6:"summer",7:"summer",8:"summer"}.get(day.month,"autumn")


def _slice_report(groups):
    out={}
    for key,rows in sorted(groups.items()):
        n=len(rows)
        if not n:
            out[key]={"n":0,"selectedMAE":None,"comparatorMAE":None,"selectedMAPEPercent":None,
                      "comparatorMAPEPercent":None,"mapeDeltaPercentagePoints":None,"maeDelta":None}
            continue
        out[key]={"n":n,"selectedMAE":sum(x[0] for x in rows)/n,"comparatorMAE":sum(x[1] for x in rows)/n,
            "selectedMAPEPercent":sum(x[2] for x in rows)/n,"comparatorMAPEPercent":sum(x[3] for x in rows)/n,
            "mapeDeltaPercentagePoints":sum(x[2]-x[3] for x in rows)/n,"maeDelta":sum(x[0]-x[1] for x in rows)/n}
    return out


def _conclusion(c):
    primary=[c['tests'][loss]['hac9'] for loss in ('absolute_error','absolute_percentage_error')]
    unavailable=[test.get('status','missing_primary_dm_status') for test in primary if test.get('status')!='tested']
    if unavailable:return 'inference_unavailable: '+', '.join(sorted(set(unavailable)))
    a=primary[0].get('holmPValue');b=primary[1].get('holmPValue')
    m=c['metrics']; candidate=m['selected'];base=m[c['comparator']]
    if candidate['mae'] is not None and base['mae'] is not None and candidate['mae']<base['mae'] and candidate['mapePercent']<base['mapePercent'] and a is not None and b is not None and a<.05 and b<.05:return 'improvement_evidence'
    if candidate['mae'] is None:return 'inference_not_supported'
    if candidate['mae']>base['mae'] and candidate['mapePercent']>base['mapePercent']:return 'worse_point_estimates_no_improvement_evidence'
    return 'insufficient_or_mixed_evidence'


def _atomic(path:Path,text:str):
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f'.{path.name}.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(text);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def write_report(report,json_path,md_path):
    _atomic(Path(json_path),json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 무 시계열 파생변수 선택 실험','',
        '후향적 탐색이며 최초 공개 시각을 재현하거나 미래 운영 성능을 보장하지 않습니다. 원래 actual은 평가 정답으로 유지하고 2023-01-21 품질 가드는 입력과 훈련 정답에서만 적용합니다.','',
        '선택 변수: '+', '.join(report['selection']['selectedFeatureIds']),'',
        '| 거리 | 비교 | 공통일 | MAE 차이 | MAPE 차이(pp) | HAC9 Holm p (AE / APE) | 판정 |','|---:|---|---:|---:|---:|---|---|']
    for c in report['test']['comparisons']:
        x=c['metrics']['selected'];b=c['metrics'][c['comparator']]
        fmt=lambda v:'—' if v is None else f'{v:.4g}'
        ae=c['tests']['absolute_error']['hac9'].get('holmPValue');ape=c['tests']['absolute_percentage_error']['hac9'].get('holmPValue')
        d_mae=x['mae']-b['mae'] if x['mae'] is not None and b['mae'] is not None else None
        d_mape=x['mapePercent']-b['mapePercent'] if x['mapePercent'] is not None and b['mapePercent'] is not None else None
        lines.append(f"| {c['horizon']} | selected vs {c['comparator']} | {c['n']} | {fmt(d_mae)} | {fmt(d_mape)} | {fmt(ae)} / {fmt(ape)} | {c['conclusion']} |")
    lines += ['',f"원천 SHA-256: `{report['inputSha256']}`. 모델·평가 원점과 날짜별 짝지은 예측, 연도·계절 slice, 미보정/보정 검정은 JSON에 있습니다.",
        '선택 구간은 2018~2022 월별 확장 검증이며, 2016-01-01 이후 관측 표본을 사용합니다. Ridge alpha=10, 목표는 최신 가용가격 대비 목표가격의 로그비입니다. 전체 Ridge 및 지속성 기준선과 동일 공통 날짜를 비교합니다. HAC9 주 분석·HAC28 민감도이며 대역폭별 Holm12를 적용합니다. 비유의는 동등성 입증이 아닙니다.']
    _atomic(Path(md_path),'\n'.join(lines)+'\n')


def run(input_path):
    path=Path(input_path); raw=path.read_bytes(); sha=hashlib.sha256(raw).hexdigest(); source=json.loads(raw)
    start=time.monotonic(); result=analyze(source,sha); result['provenance']['timingSeconds']=time.monotonic()-start
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',default='outputs/dashboard.json');ap.add_argument('--output-dir',default='outputs/experiments/radish-feature-search-20261006')
    a=ap.parse_args(argv);out=Path(a.output_dir);report=run(a.input);write_report(report,out/'report.json',out/'report.md');return 0


if __name__=='__main__':raise SystemExit(main())
