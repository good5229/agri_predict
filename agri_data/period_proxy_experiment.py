"""Frozen exploratory dekad proxy experiment; no network access.

The source is revised retrospective data, so outputs are pseudo-as-of and never
represent official national period prices or real-time availability.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, bisect_right
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/period_proxy_protocol_20261015.json"
SOURCE = ROOT / "outputs/daily/2026-10-10/20261010T120014-01ac7655/prices.json"
PROTOCOL_SHA = "d4a8b94fe6a056e22fefc3f44576018e2fd049397165ea6f856b7b10e2b4a3d4"
SOURCE_SHA = "6ca1366d1dda618c96f48e62fea578193696b24dc893e68320bdeaae8e43e659"
CROPS = ("cabbage", "radish", "onion", "garlic")
HORIZONS = (2, 3)
ALPHAS = (1.0, 10.0, 100.0)
HAC_LAGS = (3, 6)
TARGET_START = dt.date(2022, 1, 1)
TARGET_END = dt.date(2026, 9, 30)
POLICY_AVAILABLE_FROM = dt.date(2024, 1, 2)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_inputs(protocol_path: Path = PROTOCOL, source_path: Path = SOURCE) -> tuple[dict, dict]:
    if sha256(protocol_path) != PROTOCOL_SHA:
        raise ValueError("protocol hash differs from frozen protocol")
    if sha256(source_path) != SOURCE_SHA:
        raise ValueError("prices source hash differs from pinned snapshot")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen_before_model_results" or protocol.get("sourceSha256") != SOURCE_SHA:
        raise ValueError("protocol identity or source digest mismatch")
    validate_snapshot_meta(raw)
    return protocol, raw


def validate_snapshot_meta(raw):
    if raw.get("meta", {}).get("contextEnd") != "2026-10-09":
        raise ValueError("pinned snapshot context end mismatch")


def dekad_bounds(day: dt.date) -> tuple[dt.date, dt.date]:
    if day.day <= 10:
        return day.replace(day=1), day.replace(day=10)
    if day.day <= 20:
        return day.replace(day=11), day.replace(day=20)
    last = (day.replace(day=28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
    return day.replace(day=21), last


def dekads(start: dt.date, end: dt.date) -> list[tuple[dt.date, dt.date]]:
    out = []
    cur = start.replace(day=1)
    while cur <= end:
        month_end = (cur.replace(day=28)+dt.timedelta(days=4)).replace(day=1)-dt.timedelta(days=1)
        for a, b in ((1, 10), (11, 20), (21, month_end.day)):
            lo, hi = cur.replace(day=a), cur.replace(day=b)
            if start <= hi and lo <= end:
                out.append((lo, hi))
        cur = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
    return out


def aggregate_targets(series: dict[dt.date, tuple[float, float]]) -> dict[tuple[dt.date, dt.date], dict[str, Any]]:
    result = {}
    for lo, hi in dekads(min(series), max(series)) if series else []:
        rows = [(d, p, v) for d, (p, v) in series.items() if lo <= d <= hi and p > 0]
        enough = len(rows) >= 3
        result[(lo, hi)] = {"actual": mean(x[1] for x in rows) if enough else None,
                            "observedDays": len(rows),
                            "targetStatus": "valid" if enough else "insufficient_target_observations",
                            "volumeKg": sum(x[2] for x in rows), "dates": [x[0].isoformat() for x in rows]}
    return result


def series_from_raw(raw: dict) -> dict[str, dict[dt.date, tuple[float, float]]]:
    products = raw.get("products")
    if not isinstance(products, list):
        raise ValueError("prices snapshot has no products list")
    result = {}
    for p in products:
        pid = p.get("id")
        if pid not in CROPS or pid in result:
            if pid == "pepper":
                continue
            raise ValueError(f"unexpected or duplicate crop {pid!r}")
        values = {}
        for row in p.get("observations", []):
            day = dt.date.fromisoformat(row["date"])
            price, volume = float(row["actual"]), float(row["volumeKg"])
            if not math.isfinite(price) or price <= 0 or not math.isfinite(volume):
                raise ValueError("invalid observation in frozen snapshot")
            if day in values:
                raise ValueError("duplicate price date")
            values[day] = (price, volume)
        result[pid] = values
    if set(result) != set(CROPS):
        raise ValueError("snapshot must contain exactly the four in-scope crops")
    return result


def _period_index(period: tuple[dt.date, dt.date]) -> int:
    lo, _ = period
    return lo.year * 36 + (lo.month - 1) * 3 + (1 if lo.day == 1 else 2 if lo.day == 11 else 3)


def _next_period(period: tuple[dt.date, dt.date], n: int) -> tuple[dt.date, dt.date]:
    # _period_index uses 1-based slot labels for readable/HAC indexing.
    index = _period_index(period) - 1 + n
    year, within_year = divmod(index, 36)
    month_index, slot = divmod(within_year, 3)
    month = month_index + 1
    first = (1, 11, 21)[slot]
    month_end = (dt.date(year, month, 28) + dt.timedelta(days=4)).replace(day=1) - dt.timedelta(days=1)
    last = (10, 20, month_end.day)[slot]
    return dt.date(year, month, first), dt.date(year, month, last)


def origin_for(period: tuple[dt.date, dt.date]) -> dt.date:
    return period[1] + dt.timedelta(days=2)


def input_cutoff(origin: dt.date) -> dt.date:
    return origin - dt.timedelta(days=2)


def label_matured(period: tuple[dt.date, dt.date], fit_origin: dt.date) -> bool:
    return period[1] + dt.timedelta(days=2) <= fit_origin


class FeatureBuilder:
    """Lazily cache as-of rows and use prefix sums for fixed daily windows."""
    def __init__(self, series, targets, market=None):
        self.days=tuple(sorted(series))
        self.prices=tuple(series[d][0] for d in self.days)
        self.volumes=tuple(series[d][1] for d in self.days)
        self.price_prefix=[0.]
        self.volume_prefix=[0.]
        for price,volume in zip(self.prices,self.volumes):
            self.price_prefix.append(self.price_prefix[-1]+price)
            self.volume_prefix.append(self.volume_prefix[-1]+volume)
        valid=sorted((p[1],v["actual"]) for p,v in targets.items() if v.get("actual") is not None)
        self.target_ends=tuple(x[0] for x in valid)
        self.target_prices=tuple(x[1] for x in valid)
        self.market=market
        self.market_cache={}
        self.cache={}

    def row(self, origin, target):
        key=(origin,target)
        if key in self.cache:
            return self.cache[key]
        cutoff=input_cutoff(origin)
        end_idx=bisect_right(self.days,cutoff)
        if not end_idx:
            self.cache[key]=None
            return None
        source_idx=end_idx-1
        latest=self.prices[source_idx]
        vals=[math.log(latest),float((origin-self.days[source_idx]).days)]
        for width in (7,14,28):
            lo=bisect_left(self.days,cutoff-dt.timedelta(days=width-1),0,end_idx)
            count=end_idx-lo
            if count:
                vals.extend([math.log((self.price_prefix[end_idx]-self.price_prefix[lo])/count),
                             math.log1p(self.volume_prefix[end_idx]-self.volume_prefix[lo]),float(count)])
            else:
                # Preserve an actually empty lookback; never fill dates or divide by zero.
                vals.extend([0.,0.,0.])
        target_idx=bisect_right(self.target_ends,cutoff)-1
        past=self.target_prices[target_idx] if target_idx>=0 else None
        td=target[0]
        vals.extend([math.log(past) if past is not None else 0.,float(past is not None),
                     float((td-origin).days),float(td.month),float(td.day),float((target[1]-target[0]).days+1)])
        if self.market is not None:
            vals.extend(self.market_at(cutoff))
        self.cache[key]=(vals,latest)
        return self.cache[key]

    def market_at(self, cutoff):
        if self.market is None:
            return None
        if cutoff not in self.market_cache:
            self.market_cache[cutoff]=market_features(self.market,cutoff)
        return self.market_cache[cutoff]


def feature_row(series: dict[dt.date, tuple[float, float]], targets: dict, origin: dt.date,
                target: tuple[dt.date, dt.date], market: dict | None = None,
                label: dict | None = None) -> tuple[list[float], float | None, float] | None:
    row=FeatureBuilder(series,targets,market).row(origin,target)
    if row is None:
        return None
    vals,latest=row
    return vals, math.log(label["actual"] / latest) if label is not None else None, latest


def market_features(market: dict, origin: dt.date) -> list[float]:
    """Signed shift using adjacent 14-day windows ending by the supplied as-of date."""
    end_recent = origin
    recent_start, prior_end = origin-dt.timedelta(days=13), origin-dt.timedelta(days=14)
    prior_start = origin-dt.timedelta(days=27)
    def aggregate(lo, hi):
        rows = {m: [r for d, r in entries.items() if lo <= d <= hi and r.get("price", 0)>0 and r.get("weight", 0)>0]
                for m, entries in market.items()}
        supported = [m for m, rs in rows.items() if rs]
        if not supported: return {}, set()
        prices = {m: sum(r["price"]*r["weight"] for r in rs)/sum(r["weight"] for r in rs) for m,rs in rows.items() if rs}
        weights = {m: sum(r["weight"] for r in rs) for m,rs in rows.items() if rs}
        total = sum(weights.values())
        return ({m: (prices[m], weights[m]/total) for m in supported}, set(supported))
    prior, ps = aggregate(prior_start, prior_end)
    recent, rs = aggregate(recent_start, end_recent)
    common = ps & rs
    complete = bool(common) and len(common) == len(market)
    if not common:
        return [0., 0., 0., float(len(rs)), float(len(ps))]
    base = sum(prior[m][0]*prior[m][1] for m in common)
    shift = sum((recent[m][1]-prior[m][1])*(prior[m][0]/base-1) for m in common)
    return [shift, float(len(common)), float(len(market)), float(complete), float(len(rs))]


def _previous_period_target(target_period, n):
    return _next_period(target_period, -n)


def _target_periods(series):
    return aggregate_targets(series)


def _quarter_key(day): return (day.year, (day.month-1)//3)


def planned_origins(start, end, horizon):
    rows=[]
    for target in dekads(start,end):
        origin_period=_previous_period_target(target,horizon)
        origin=origin_for(origin_period)
        rows.append({"target":target,"originPeriod":origin_period,"origin":origin,"inputCutoff":input_cutoff(origin)})
    return rows


def _inverse_log_ratio(reference_price, predicted_log_ratio):
    return reference_price * math.exp(predicted_log_ratio)


def _with_b1_fallback(candidate, b1, planned):
    out={}
    for target in planned:
        if target in candidate:
            out[target]={**candidate[target],"fallback":False}
        elif target in b1:
            out[target]={**b1[target],"fallback":True}
    return out


def _policy_available_predictions(predictions, cutoff=POLICY_AVAILABLE_FROM):
    return {target:row for target,row in predictions.items()
            if dt.date.fromisoformat(row["origin"]) >= cutoff}


def _c1_inference_status(test, pairs, native_count, gate_open):
    if not gate_open:
        return {"status":"gate_not_open","n":len(pairs)}
    if native_count == 0 and pairs:
        return {"status":"fallback_only","n":len(pairs)}
    if not pairs:
        return {"status":"not_issued","n":0}
    return test


def rolling_predictions(series, targets, horizon, alpha, *, start, end, cadence="origin", market=None,
                        policy_available_from=None, feature_builder=None):
    """Forecast target dekads; no label enters a fit until target end + 2 <= fit origin."""
    predictions, audits = {}, []
    periods = sorted(tp for tp,value in targets.items() if value.get("actual") is not None)
    features=feature_builder or FeatureBuilder(series,targets,market)
    by_origin = {}
    for schedule in planned_origins(start,end,horizon):
        target=schedule["target"]; origin_period=schedule["originPeriod"]; origin=schedule["origin"]
        by_origin.setdefault(_quarter_key(origin), []).append((origin, target))
    groups = list(by_origin.values()) if cadence == "quarter" else [[r] for group in by_origin.values() for r in group]
    for group in groups:
        fit_origin = group[0][0]
        if policy_available_from is not None and fit_origin < policy_available_from:
            audits.append({"fitOrigin":fit_origin.isoformat(),"plannedForecasts":len(group),"trainLabels":0,
                           "status":"policy_unavailable_before_development_labels_mature"})
            continue
        trainX, trainY, train_periods = [], [], []
        matured_calendar=[p for p in dekads(min(series),input_cutoff(fit_origin)) if label_matured(p,fit_origin)] if series else []
        available_matured=[p for p in periods if label_matured(p,fit_origin)]
        for tp in periods:
            if not label_matured(tp, fit_origin): continue
            origin_period = _previous_period_target(tp, horizon)
            train_origin = origin_for(origin_period)
            if train_origin >= fit_origin: continue
            row=features.row(train_origin,tp)
            if row is None or (market is not None and row[0][18] < 1):
                continue
            trainX.append(row[0]); trainY.append(math.log(targets[tp]["actual"]/row[1])); train_periods.append(tp)
        valid = []
        for origin, tp in group:
            row=features.row(origin,tp)
            if row is None or (market is not None and row[0][18] < 1):
                continue
            valid.append((origin,tp,row[0],row[1]))
        if not trainX:
            audits.append({"fitOrigin": fit_origin.isoformat(), "trainLabels": len(trainX),
                           "maturedCalendarPeriods":len(matured_calendar),"availableMaturedLabels":len(available_matured),
                           "maturedLabelCoverage":len(available_matured)/len(matured_calendar) if matured_calendar else None,
                           "latestMaturedTargetEnd":max((p[1].isoformat() for p in train_periods),default=None),
                           "status":"insufficient_training"}); continue
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(np.asarray(trainX), np.asarray(trainY))
        for origin,tp,x,reference_price in valid:
            pred = _inverse_log_ratio(reference_price,float(model.predict(np.asarray([x]))[0]))
            predictions[tp] = {"origin": origin.isoformat(), "targetStart":tp[0].isoformat(),"targetEnd":tp[1].isoformat(),"predicted":pred,
                               "fallback": False, "alpha":alpha}
        audits.append({"fitOrigin":fit_origin.isoformat(),"trainLabels":len(trainX),
                       "maturedCalendarPeriods":len(matured_calendar),"availableMaturedLabels":len(available_matured),
                       "maturedLabelCoverage":len(available_matured)/len(matured_calendar) if matured_calendar else None,
                       "latestMaturedTargetEnd":max((p[1].isoformat() for p in train_periods),default=None),
                       "forecastCount":len(valid),"status":"fitted"})
    return predictions, audits


def baseline_predictions(series, targets, horizon, start, end):
    out = {"B0a":{},"B0b":{},"B0c":{}}
    days=tuple(sorted(series))
    valid_targets=sorted((period[1],period,value["actual"]) for period,value in targets.items() if value.get("actual") is not None)
    valid_ends=tuple(row[0] for row in valid_targets)
    for schedule in planned_origins(start,end,horizon):
        target=schedule["target"]; origin_period=schedule["originPeriod"]; origin=schedule["origin"]; cutoff=schedule["inputCutoff"]
        day_idx=bisect_right(days,cutoff)-1
        if day_idx>=0:
            d=days[day_idx]; p=series[d][0]
            out["B0a"][target]={"predicted":p,"origin":origin.isoformat(),"sourceDate":d.isoformat()}
        target_idx=bisect_right(valid_ends,cutoff)-1
        if target_idx>=0:
            _,tp,p=valid_targets[target_idx]
            out["B0b"][target]={"predicted":p,"origin":origin.isoformat(),"sourceTargetEnd":tp[1].isoformat()}
        prev=_previous_year_period(target)
        if prev in targets and targets[prev].get("actual") is not None and label_matured(prev,origin):
            out["B0c"][target]={"predicted":targets[prev]["actual"],"origin":origin.isoformat(),"sourceTargetEnd":prev[1].isoformat()}
    return out


def _previous_year_period(period):
    return _next_period(period, -36)


def _metrics(rows):
    if not rows:return {"n":0,"mae":None,"mape":None}
    return {"n":len(rows),"mae":mean(abs(a-p) for a,p in rows),"mape":mean(abs(a-p)/a for a,p in rows)*100}


def _hac_dm(pairs, loss, lag):
    if len(pairs)<30:return {"status":"insufficient_pairs","n":len(pairs)}
    vals={_period_index(tp): ((abs(p-a) if loss=="ae" else abs(p-a)/a) -
                              (abs(q-a) if loss=="ae" else abs(q-a)/a)) for tp,a,p,q in pairs}
    ordered=sorted(vals.items()); n=len(ordered); d=float(np.mean([v for _,v in ordered]))
    z={idx:value-d for idx,value in ordered}
    var=float(sum(value*value for value in z.values())/n)
    for k in range(1,lag+1):
        cov=float(sum(z[idx]*z.get(idx-k,0.) for idx in z if idx-k in z)/n)
        var += 2.*(1.-k/(lag+1))*cov
    if not math.isfinite(var) or var<=0:return {"status":"nonpositive_hac_variance","n":len(pairs),"meanLossDifference":d}
    stat=d/math.sqrt(var/n)
    return {"status":"tested","n":len(pairs),"meanLossDifference":d,"hacVariance":var,"statistic":stat,"p":math.erfc(abs(stat)/math.sqrt(2))}


def holm(items):
    valid=sorted((x["p"],i) for i,x in enumerate(items) if x.get("status")=="tested")
    adj=[None]*len(items); running=0.
    for rank,(p,i) in enumerate(valid):
        running=max(running,min(1.,(len(items)-rank)*p)); adj[i]=running
    return [{**x,"holmP":adj[i],"holmSignificant":adj[i] is not None and adj[i]<.05} for i,x in enumerate(items)]


def _choose_baseline(actuals, maps):
    anchor=set(actuals)
    common_anchor={t for t in anchor if t in maps["B0a"]}
    if not common_anchor:
        return "B0a"
    scores=[]
    for i,name in enumerate(("B0a","B0b","B0c")):
        keys={t for t in common_anchor if t in maps[name]}
        if len(keys)<30 or len(keys)/len(common_anchor)<.90:
            continue
        rows=[(actuals[t],maps[name][t]["predicted"]) for t in keys]
        scores.append((_metrics(rows)["mape"] if rows else math.inf,i,name))
    return min(scores)[2]


def run(raw, protocol, market_data=None):
    allseries=series_from_raw(raw); results={}; family=[]
    for crop in CROPS:
        series=allseries[crop]; targets=_target_periods(series)
        b1_feature_builder=FeatureBuilder(series,targets)
        c1_feature_builder=FeatureBuilder(series,targets,market_data[crop]) if market_data is not None else None
        results[crop]={}
        for h in HORIZONS:
            base_dev=baseline_predictions(series,targets,h,dt.date(2022,1,1),dt.date(2023,12,31))
            dev_targets={tp:v for tp,v in targets.items() if v.get("actual") is not None and dt.date(2022,1,1)<=tp[0]<=dt.date(2023,12,31)}
            selected=_choose_baseline({tp:v["actual"] for tp,v in dev_targets.items()},base_dev)
            alpha_scores=[]; dev_ridge={}
            for alpha in ALPHAS:
                pred,audit=rolling_predictions(series,targets,h,alpha,start=dt.date(2022,1,1),end=dt.date(2023,12,31),feature_builder=b1_feature_builder)
                dev_ridge[alpha]=pred
                pairs=[(dev_targets[t]["actual"],r["predicted"]) for t,r in pred.items() if t in dev_targets]
                alpha_scores.append({"alpha":alpha,**_metrics(pairs)})
            selected_alpha=min(alpha_scores,key=lambda x:(math.inf if x["mape"] is None else x["mape"],x["alpha"]))["alpha"]
            rdev=dev_ridge[selected_alpha]
            # C1 gate is fitted from unique development targets only.
            gate={"attempted":False,"status":"market_source_unavailable" if market_data is None else "diagnostic_unavailable"}
            c1_alpha_scores=[]; c1_selected_alpha=None
            if market_data is not None:
                signed=[]; residual=[]; years={2022:[],2023:[]}
                for tp,v in dev_targets.items():
                    if tp not in rdev: continue
                    org=origin_for(_previous_period_target(tp,h)); f=c1_feature_builder.market_at(input_cutoff(org))
                    if f[1] < 1: continue
                    signed.append(f[0]); residual.append(v["actual"]-rdev[tp]["predicted"]); years[tp[0].year].append((f[0],v["actual"]-rdev[tp]["predicted"]))
                rho=float(spearmanr(signed,residual).statistic) if len(signed)>=2 else float("nan")
                ycorr={str(y):float(np.corrcoef([a for a,b in pairs],[b for a,b in pairs])[0,1]) if len(pairs)>=2 and np.std([a for a,b in pairs]) and np.std([b for a,b in pairs]) else None for y,pairs in years.items()}
                enabled=len(set(tp for tp in dev_targets if tp in rdev))>=20 and len(signed)>=20 and math.isfinite(rho) and rho>=.1 and all(ycorr[str(y)] is not None and ycorr[str(y)]>0 for y in (2022,2023))
                gate={"attempted":enabled,"status":"enabled" if enabled else "diagnostic_gate_failed","n":len(signed),"spearman":rho if math.isfinite(rho) else None,"yearlyCorrelations":ycorr}
                if enabled:
                    for alpha in ALPHAS:
                        cp,_=rolling_predictions(series,targets,h,alpha,start=dt.date(2022,1,1),end=dt.date(2023,12,31),market=market_data[crop],feature_builder=c1_feature_builder)
                        # Missing C1 features always use the already selected B1 procedure.
                        bp=dev_ridge[selected_alpha]
                        c1dev=_with_b1_fallback(cp,bp,dev_targets)
                        rows=[(dev_targets[t]["actual"],c1dev[t]["predicted"]) for t in dev_targets if t in c1dev]
                        c1_alpha_scores.append({"alpha":alpha,**_metrics(rows)})
                    c1_selected_alpha=min(c1_alpha_scores,key=lambda x:(math.inf if x["mape"] is None else x["mape"],x["alpha"]))["alpha"]
            # Test fixed alpha, quarterly fits. C1 forecast falls back to B1 where feature absent.
            test_base=baseline_predictions(series,targets,h,dt.date(2024,1,1),dt.date(2026,9,30))
            # The comparator was selected on development only; retain this choice for test.
            selected_test=selected
            b1,fit_audit=rolling_predictions(series,targets,h,selected_alpha,start=dt.date(2024,1,1),end=dt.date(2026,9,30),cadence="quarter",policy_available_from=POLICY_AVAILABLE_FROM,feature_builder=b1_feature_builder)
            selected_b0=_policy_available_predictions(test_base[selected_test])
            c1={}; c1_fit_audit=[]
            if gate.get("attempted") and market_data is not None:
                c1raw,c1_fit_audit=rolling_predictions(series,targets,h,c1_selected_alpha,start=dt.date(2024,1,1),end=dt.date(2026,9,30),cadence="quarter",market=market_data[crop],policy_available_from=POLICY_AVAILABLE_FROM,feature_builder=c1_feature_builder)
                expected=set(dekads(dt.date(2024,1,1),dt.date(2026,9,30)))
                c1=_with_b1_fallback(c1raw,b1,expected)
            planned_test=dekads(dt.date(2024,1,1),dt.date(2026,9,30))
            test_actual={tp:v["actual"] for tp,v in targets.items() if v.get("actual") is not None and dt.date(2024,1,1)<=tp[0]<=dt.date(2026,9,30)}
            methods={"B0a":test_base["B0a"],"B0b":test_base["B0b"],"B0c":test_base["B0c"],"selectedB0":selected_b0,"B1":b1,"C1":c1}
            pairs={"B1_vs_selectedB0":[],"C1_vs_B1":[],"C1_vs_selectedB0":[]}
            predrows=[]
            for tp in planned_test:
                actual=test_actual.get(tp)
                forecasts={n:m[tp] for n,m in methods.items() if tp in m}
                op=_previous_period_target(tp,h); issue=origin_for(op)
                target_record=targets.get(tp,{"actual":None,"observedDays":0,"targetStatus":"insufficient_target_observations"})
                policy_available=issue>=POLICY_AVAILABLE_FROM
                predrows.append({"targetStart":tp[0].isoformat(),"targetEnd":tp[1].isoformat(),"originPeriodStart":op[0].isoformat(),
                                 "originPeriodEnd":op[1].isoformat(),"origin":issue.isoformat(),"inputCutoff":input_cutoff(issue).isoformat(),
                                 "learnedPolicyAvailable":policy_available,
                                 "learnedPolicyUnavailableReason":None if policy_available else "development_target_labels_not_matured_by_origin",
                                 "targetObserved":actual is not None,"actual":actual,"observedDays":target_record["observedDays"],
                                 "targetStatus":target_record["targetStatus"],"forecasts":forecasts})
            comp_pairs={name:[] for name in pairs}
            for row in predrows:
                tp=(dt.date.fromisoformat(row["targetStart"]),dt.date.fromisoformat(row["targetEnd"])); f=row["forecasts"]
                for comp,left,right in (("B1_vs_selectedB0","B1","selectedB0"),("C1_vs_B1","C1","B1"),("C1_vs_selectedB0","C1","selectedB0")):
                    if row["actual"] is not None and left in f and right in f and f[left].get("predicted") is not None and f[right].get("predicted") is not None:
                        comp_pairs[comp].append((tp,row["actual"],f[left]["predicted"],f[right]["predicted"]))
            stats_raw=[]
            for comp in ("B1_vs_selectedB0","C1_vs_B1","C1_vs_selectedB0"):
                ps=comp_pairs[comp]
                native_c1=sum(1 for row in predrows if row["actual"] is not None and "C1" in row["forecasts"] and
                              not row["forecasts"]["C1"].get("fallback",False) and
                              ("selectedB0" in row["forecasts"] if comp.endswith("selectedB0") else "B1" in row["forecasts"])) if comp.startswith("C1_") else None
                for loss in ("ae","ape"):
                    for lag in HAC_LAGS:
                        test=_hac_dm(ps,loss,lag)
                        if comp.startswith("C1_"):
                            test=_c1_inference_status(test,ps,native_c1,gate.get("attempted",False))
                        stats_raw.append({"crop":crop,"horizon":h,"comparison":comp,"loss":loss,"lag":lag,**test})
            stats=stats_raw
            for item in stats:
                family.append((crop,str(h),item))
            metrics={}
            for name in methods:
                rows=[(r["actual"],r["forecasts"][name]["predicted"]) for r in predrows if r["actual"] is not None and name in r["forecasts"] and r["forecasts"][name].get("predicted") is not None]
                metrics[name]=_metrics(rows)
            paired_metrics={}
            for comp,left,right in (("B1_vs_selectedB0","B1","selectedB0"),("C1_vs_B1","C1","B1"),("C1_vs_selectedB0","C1","selectedB0")):
                lp=[(r["actual"],r["forecasts"][left]["predicted"]) for r in predrows if r["actual"] is not None and left in r["forecasts"] and right in r["forecasts"] and r["forecasts"][left].get("predicted") is not None and r["forecasts"][right].get("predicted") is not None]
                rp=[(r["actual"],r["forecasts"][right]["predicted"]) for r in predrows if r["actual"] is not None and left in r["forecasts"] and right in r["forecasts"] and r["forecasts"][left].get("predicted") is not None and r["forecasts"][right].get("predicted") is not None]
                paired_metrics[comp]={"n":len(lp),"left":_metrics(lp),"right":_metrics(rp)}
            results[crop][str(h)]={"selectedAlpha":selected_alpha,"alphaDevelopment":alpha_scores,"selectedSimpleBaseline":selected_test,"baselineDevelopmentSelection":selected,
                "c1SelectedAlpha":c1_selected_alpha,"c1AlphaDevelopment":c1_alpha_scores,
                "c1DiagnosticGate":gate,"fitAudit":{"B1":fit_audit,"C1":c1_fit_audit},"plannedOrigins":len(planned_test),
                "policyUnavailableOrigins":sum(row["origin"]<POLICY_AVAILABLE_FROM.isoformat() for row in predrows),
                "policyUnavailableReason":"development target labels ending 2023-12-31 mature on 2024-01-02",
                "coverage":{n:{"plannedOrigins":len(planned_test),"targetForecasts":len(m),"targetsWithObservedActual":len(test_actual),"scoredForecasts":sum(tp in m for tp in test_actual),"targetRate":sum(tp in m for tp in test_actual)/len(test_actual) if test_actual else None} for n,m in methods.items()},
                "c1ForecastCounts":{"native":sum(not x.get("fallback",False) for x in c1.values()),"fallback":sum(x.get("fallback",False) for x in c1.values()),"issued":len(c1)},
                "metrics":metrics,"pairedMetrics":paired_metrics,"comparisons":stats,"predictions":predrows,"annual":_annual(predrows)}
    # Apply Holm to the full frozen family (48 tests) separately for each HAC lag.
    for lag in HAC_LAGS:
        group=[x[2] for x in family if x[2]["lag"]==lag]
        adjusted=holm(group)
        ix=0
        for crop,h,item in family:
            if item["lag"]!=lag: continue
            result=adjusted[ix]; ix+=1
            dest=results[crop][h]["comparisons"]
            dest[dest.index(item)]=result
    for crop in CROPS:
        for h in HORIZONS:
            cell=results[crop][str(h)]
            cell["conclusions"]={name:_conclusion(cell,name) for name in ("B1_vs_selectedB0","C1_vs_B1","C1_vs_selectedB0")}
    return {"schemaVersion":1,"protocolId":protocol["id"],"protocolSha256":PROTOCOL_SHA,"sourceSha256":SOURCE_SHA,
            "analysisCodeSha256":sha256(Path(__file__).resolve()),
            "statisticalProtocol":{"familySizePerBandwidth":48,"hacBartlettLags":list(HAC_LAGS),"holmAlpha":.05,
                "minimumDistinctPairedTargets":30,"losses":["AE","APE"],"twoSided":True},
            "sourceMeaning":"revised pseudo-as-of Garak+Gangseo signed aggregate; historical publication times unknown",
            "evaluation":"exploratory; proxy dekads are not official national targets or prospective results",
            "targetWindow":{"development":["2022-01-01","2023-12-31"],"test":["2024-01-01","2026-09-30"]},"crops":results}


def _annual(rows):
    out={}
    for y in range(2024,2027):
        sel=[r for r in rows if dt.date.fromisoformat(r["targetStart"]).year==y]
        out[str(y)]={name:_metrics([(r["actual"],f[name]["predicted"]) for r in sel if r["actual"] is not None and name in (f:=r["forecasts"]) and f[name].get("predicted") is not None]) for name in ("B0a","B0b","B0c","B1","C1")}
    return out


def _conclusion(cell, comparison):
    metrics=cell["pairedMetrics"][comparison]
    if metrics["n"]==0:
        return {"observed":"no_common_predictions","statistical":"inference_unavailable","operationalAdoption":False}
    left,right=metrics["left"],metrics["right"]
    better=left["mae"]<right["mae"] and left["mape"]<right["mape"]
    worse=left["mae"]>right["mae"] and left["mape"]>right["mape"]
    observed="improved" if better else "worsened" if worse else "mixed"
    primary=[x for x in cell["comparisons"] if x["comparison"]==comparison and x["lag"]==3]
    inferable=len(primary)==2 and all(x["status"]=="tested" for x in primary)
    signs={-1 if x.get("meanLossDifference",0)<0 else 1 if x.get("meanLossDifference",0)>0 else 0 for x in primary}
    both_sig=inferable and all(x.get("holmSignificant",False) for x in primary)
    if not inferable:
        stat="inference_unavailable"
    elif len(signs)>1:
        stat="mixed"
    elif signs=={1} and both_sig:
        stat="significant_worsening"
    elif signs=={-1} and both_sig and better:
        stat="superiority"
    else:
        stat="inconclusive"
    return {"observed":observed,"statistical":stat,
            "operationalAdoption":False,"adoptionReason":"retrospective revised proxy; protocol prohibits operational adoption"}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/experiments/period-proxy-20261015/report.json")
    parser.add_argument("--with-market",action="store_true",help="include reconciled read-only Garak/Gangseo SQLite features for C1")
    args=parser.parse_args(argv)
    protocol,raw=validate_inputs()
    market_data=None
    market_provenance=None
    if args.with_market:
        from . import direct_horizon, market_mix
        if not market_mix.DB_PATH.is_file():
            raise ValueError("C1 requested but the reconciled market database is unavailable")
        core=direct_horizon._observations(raw)
        market_data, provenance=market_mix.extract_market_data(market_mix.DB_PATH,raw,core)
        dbfiles=[market_mix.DB_PATH]+([market_mix.DB_PATH.with_name(market_mix.DB_PATH.name+"-wal")] if market_mix.DB_PATH.with_name(market_mix.DB_PATH.name+"-wal").exists() else [])
        market_provenance={"databaseFiles":[{"path":str(p),"sha256":sha256(p)} for p in dbfiles],
                           "snapshotReconciliation":provenance["reconciliation"],
                           "marketSource":"read-only SQLite; signed amount/net-weight recomposed by market_mix"}
    report=run(raw,protocol,market_data)
    report["marketProvenance"]=market_provenance
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(args.output),"crops":len(report["crops"]),"sourceSha256":SOURCE_SHA}))


if __name__=="__main__": main()
