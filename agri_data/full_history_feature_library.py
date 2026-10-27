"""Named causal scale-free feature library for the full-history Ridge study.

The first 40 values preserve the reviewed rich-Ridge feature order. The final
31 are the preregistered expansion. Every value is derived from observations
dated no later than origin-1; no missing dates are imputed.
"""
from __future__ import annotations

import datetime as dt
import bisect
import math
import statistics
from typing import Any

from . import full_history_xgb_rf_eval as prior

MARKET_CODES=prior.MARKET_CODES

BASE_FEATURE_NAMES=(
    "price_logratio_p1","price_present_p1","price_gap_p1",
    "price_logratio_p7","price_present_p7","price_gap_p7",
    "price_logratio_p30","price_present_p30","price_gap_p30","p0_age_days",
    "target_weekday_sin","target_weekday_cos","target_annual_sin","target_annual_cos",
    *(x for w in (7,30,90) for x in (f"price_logratio_mean_{w}d",f"price_log_std_{w}d",f"price_log_range_{w}d",f"price_count_{w}d")),
    *(x for w in (7,30,90) for x in (f"volume_logratio_{w}d",f"volume_count_{w}d",f"volume_present_{w}d")),
    "rowcount_logratio_30d","rowcount_count_30d","rowcount_present_30d",
    "p0_market_count","dual_market_share_30d",
)
EXPANDED_FEATURE_NAMES=(
    *(x for w in (3,14,60) for x in (f"price_logratio_anchor_{w}d",f"price_anchor_present_{w}d",f"price_anchor_gap_{w}d")),
    *(x for w in (14,60) for x in (f"price_log_slope_{w}d",f"price_slope_count_{w}d",f"price_slope_present_{w}d")),
    *(x for w in (14,60) for x in (f"volume_logratio_{w}d",f"volume_count_{w}d",f"volume_present_{w}d")),
    *(x for w in (7,90) for x in (f"rowcount_logratio_{w}d",f"rowcount_count_{w}d",f"rowcount_present_{w}d")),
    *(x for w in (7,90) for x in (f"dual_market_share_{w}d",f"market_observation_count_{w}d")),
)
FEATURE_NAMES=BASE_FEATURE_NAMES+EXPANDED_FEATURE_NAMES
assert len(BASE_FEATURE_NAMES)==40 and len(EXPANDED_FEATURE_NAMES)==31

def features(records:dict[dt.date,dict[str,Any]],target:dt.date,horizon:int,
                   ordered:list[dt.date]|None=None,prices:dict[dt.date,float]|None=None
                   )->tuple[list[float]|None,float|None,dict[str,Any]]:
    """Return (71-value vector, p0, audit); missing/stale p0 returns None."""
    days=ordered if ordered is not None else sorted(records)
    price_values=prices if prices is not None else {d:r["price"] for d,r in records.items()}
    base,p0,audit=prior._rich_features(records,target,horizon,days,price_values)
    if base is None or p0 is None:return None,None,audit
    p0_day=dt.date.fromisoformat(audit["p0Date"])
    end=bisect.bisect_right(days,p0_day)
    def bounds(lower:dt.date,upper:dt.date)->tuple[int,int]:
        lo=bisect.bisect_left(days,lower,0,end)
        hi=bisect.bisect_right(days,upper,lo,end)
        return lo,hi
    extra=[]
    for window in (3,14,60):
        cutoff=p0_day-dt.timedelta(days=window)
        lo,hi=bounds(cutoff-dt.timedelta(days=window),cutoff)
        candidates=[days[i] for i in range(lo,hi) if price_values.get(days[i],0)>0]
        anchor=max(candidates) if candidates else None
        value=price_values.get(anchor) if anchor else None
        extra.extend([math.log(p0/value) if value else 0.,float(value is not None),float((p0_day-anchor).days if anchor else 0)])
    for window in (14,60):
        lo,hi=bounds(p0_day-dt.timedelta(days=window-1),p0_day)
        ds=[days[i] for i in range(lo,hi) if price_values.get(days[i],0)>0]
        logs=[math.log(price_values[d]) for d in ds]
        has=len(ds)>=3
        slope=(logs[-1]-logs[0])/(ds[-1]-ds[0]).days if has and ds[-1]!=ds[0] else 0.
        extra.extend([slope,float(len(ds)),float(has)])
    def strict_prior(window:int,field:str)->list[float]:
        low=p0_day-dt.timedelta(days=window)
        lo,hi=bounds(low,p0_day-dt.timedelta(days=1))
        return [float(records[days[i]][field]) for i in range(lo,hi) if records[days[i]].get(field) is not None and float(records[days[i]][field])>0]
    for window in (14,60):
        values=strict_prior(window,"volume");median=statistics.median(values) if values else None
        current=records[p0_day].get("volume");present=current is not None and median is not None
        extra.extend([math.log(float(current)/median) if present else 0.,float(len(values)),float(present)])
    for window in (7,90):
        values=strict_prior(window,"rowCount");median=statistics.median(values) if values else None
        current=records[p0_day].get("rowCount");present=current is not None and median is not None
        extra.extend([math.log(float(current)/median) if present else 0.,float(len(values)),float(present)])
    for window in (7,90):
        low=p0_day-dt.timedelta(days=window)
        lo,hi=bounds(low,p0_day-dt.timedelta(days=1))
        obs=[records[days[i]] for i in range(lo,hi)]
        dual=sum(MARKET_CODES.issubset(r.get("markets",frozenset())) for r in obs)
        extra.extend([dual/len(obs) if obs else 0.,float(len(obs))])
    vector=list(base)+extra
    if len(base)!=40 or len(vector)!=len(FEATURE_NAMES):raise ValueError("feature schema mismatch")
    if not all(math.isfinite(x) for x in vector):raise ValueError("nonfinite feature")
    return vector,p0,{**audit,"baseFeatureCount":len(base),"featureCount":len(vector),"featureNames":list(FEATURE_NAMES)}

build_features=features
