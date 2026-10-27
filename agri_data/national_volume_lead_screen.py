"""Offline descriptive screen of two-market volume pressure vs future national price changes."""
from __future__ import annotations

import argparse
import datetime as dt
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
from pathlib import Path
from collections import Counter
from typing import Any

import numpy as np
from scipy.stats import rankdata

from . import dashboard, direct_horizon, national_price_selection as price_selection, national_transfer

PROJECT_ROOT=Path(__file__).resolve().parent.parent
INPUT_PRICES=PROJECT_ROOT/"outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
INPUT_ARCHIVE=PROJECT_ROOT/"data/raw/national-wholesale-20240708-20260630.zip"
PRICES_SHA256="f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
ARCHIVE_SHA256=national_transfer.ARCHIVE_SHA256
PRODUCTS=tuple(national_transfer.PRODUCT_CODES)
HORIZONS=(5,10)
BLOCKS=((dt.date(2024,9,1),dt.date(2025,3,31)),
        (dt.date(2025,4,1),dt.date(2025,9,30)),
        (dt.date(2025,10,1),dt.date(2026,3,31)),
        (dt.date(2026,4,1),dt.date(2026,6,30)))
MIN_COVERAGE=0.70
MIN_SIGN_MATCH_BLOCKS=3
MIN_MAGNITUDE_BLOCKS=2
MIN_ABS_RHO=0.10


def _sha(data: bytes)->str:return hashlib.sha256(data).hexdigest()


def _positive(value: Any)->float|None:
    try:x=float(value)
    except (TypeError,ValueError,OverflowError):return None
    return x if math.isfinite(x) and x>0 else None


def _finite_signed(value: Any)->float|None:
    if value is None or isinstance(value,bool):return None
    try:x=float(value)
    except (TypeError,ValueError,OverflowError):return None
    return x if math.isfinite(x) else None


def _date(value: str)->dt.date:
    return dt.date.fromisoformat(value)


def _last_price(series: dict[dt.date,float], cutoff: dt.date, max_age: int=14)->tuple[dt.date,float]|None:
    day=max((d for d,v in series.items() if d<=cutoff and _positive(v) is not None),default=None)
    if day is None or (cutoff-day).days>max_age:return None
    return day,float(series[day])


def _volume_pressure(series: dict[dt.date,float|None], cutoff: dt.date)->tuple[float|None,dict]:
    seven_start=cutoff-dt.timedelta(days=6)
    prior_start=cutoff-dt.timedelta(days=27)
    seven=[v for d,v in series.items() if seven_start<=d<=cutoff and v is not None and math.isfinite(v)]
    prior=[v for d,v in series.items() if prior_start<=d<=cutoff-dt.timedelta(days=7) and v is not None and math.isfinite(v)]
    signed={"sevenDayNegativeObserved":sum(v<0 for v in seven),"sevenDayZeroObserved":sum(v==0 for v in seven),
            "prior21DayNegativeObserved":sum(v<0 for v in prior),"prior21DayZeroObserved":sum(v==0 for v in prior)}
    if not seven:return None,{"reason":"no_observed_volume_in_7d",**signed}
    if not prior:return None,{"reason":"no_observed_volume_in_prior_21d",**signed}
    mean7=float(np.mean(seven));mean21=float(np.mean(prior))
    if not math.isfinite(mean7) or not math.isfinite(mean21):return None,{"reason":"nonfinite_signed_volume_mean",**signed}
    if mean7<=0 or mean21<=0:return None,{"reason":"nonpositive_signed_volume_mean",**signed}
    pressure=math.log(mean7/mean21)
    if not math.isfinite(pressure):return None,{"reason":"nonfinite_log_ratio",**signed}
    return pressure,{"reason":None,"sevenDayObservedDays":len(seven),"prior21DayObservedDays":len(prior),
                     "sevenDayMeanSignedVolumeKg":mean7,"prior21DayMeanSignedVolumeKg":mean21,**signed}


def _spearman(xs:list[float],ys:list[float])->float|None:
    if len(xs)<2:return None
    rx=rankdata(np.asarray(xs),method="average");ry=rankdata(np.asarray(ys),method="average")
    if np.std(rx)==0 or np.std(ry)==0:return None
    rho=float(np.corrcoef(rx,ry)[0,1])
    return rho if math.isfinite(rho) else None


def build_screen(national_prices:dict[str,dict[dt.date,float]], volume_series:dict[str,dict[dt.date,float|None]], source_digests:dict|None=None)->dict:
    if set(national_prices)!=set(PRODUCTS) or set(volume_series)!=set(PRODUCTS):raise ValueError("both sources must contain the four frozen products")
    blocks={}; all_cells=[]
    for start,end in BLOCKS:
        block_key=f"{start.isoformat()}_{end.isoformat()}";block={}
        for product in PRODUCTS:
            for horizon in HORIZONS:
                planned=target_observed=eligible=0; xs=[];ys=[];missing=Counter()
                for offset in range((end-start).days+1):
                    target=start+dt.timedelta(days=offset);origin=target-dt.timedelta(days=horizon);cutoff=origin-dt.timedelta(days=2)
                    planned+=1
                    base=_last_price(national_prices[product],cutoff)
                    pressure,detail=_volume_pressure(volume_series[product],cutoff)
                    actual=_positive(national_prices[product].get(target))
                    if actual is not None:target_observed+=1
                    if base is None:missing["missing_asof_national_price"]+=1
                    if pressure is None:missing[detail["reason"]]+=1
                    if base is not None and pressure is not None:eligible+=1
                    if base is not None and pressure is not None and actual is not None:
                        xs.append(pressure);ys.append(math.log(actual/base[1]))
                rho=_spearman(xs,ys)
                coverage=(len(xs)/target_observed) if target_observed else None
                cell={"product":product,"horizon":horizon,"block":[start.isoformat(),end.isoformat()],
                      "plannedOrigins":planned,"targetObservedOrigins":target_observed,"featureEligibleOrigins":eligible,
                      "pairedFeatureTargetCount":len(xs),"pairCoverageAmongTargetObserved":coverage,
                      "spearmanRho":rho,"spearmanSign":("positive" if rho>0 else "negative" if rho<0 else "zero_or_undefined") if rho is not None else None,
                      "featureMissingness":dict(sorted(missing.items()))}
                block[f"{product}/{horizon}"]=cell;all_cells.append(cell)
        blocks[block_key]=block
    eligibility={}
    for product in PRODUCTS:
        for horizon in HORIZONS:
            cells=[block[f"{product}/{horizon}"] for block in blocks.values()]
            signs=[1 if c["spearmanRho"] is not None and c["spearmanRho"]>0 else -1 if c["spearmanRho"] is not None and c["spearmanRho"]<0 else 0 for c in cells]
            positives=sum(s==1 for s in signs);negatives=sum(s==-1 for s in signs)
            dominant=1 if positives>negatives else -1 if negatives>positives else 0
            sign_matches=sum(s==dominant for s in signs) if dominant else 0
            magnitude_blocks=sum(c["spearmanRho"] is not None and abs(c["spearmanRho"])>=MIN_ABS_RHO for c in cells)
            coverage_each=all(c["pairCoverageAmongTargetObserved"] is not None and c["pairCoverageAmongTargetObserved"]>=MIN_COVERAGE for c in cells)
            sign_ok=dominant!=0 and sign_matches>=MIN_SIGN_MATCH_BLOCKS
            magnitude_ok=magnitude_blocks>=MIN_MAGNITUDE_BLOCKS
            eligibility[f"{product}/{horizon}"]={"status":"eligible_for_follow_on_model" if coverage_each and sign_ok and magnitude_ok else "screen_not_met",
                "thresholds":{"minimumPairCoverageEachBlock":MIN_COVERAGE,"minimumSignConsistentBlocks":MIN_SIGN_MATCH_BLOCKS,
                              "minimumMagnitudeBlocks":MIN_MAGNITUDE_BLOCKS,"minimumAbsoluteRho":MIN_ABS_RHO},
                "coverageCriterionMet":coverage_each,"dominantNonzeroSign":("positive" if dominant>0 else "negative" if dominant<0 else None),
                "signMatchesDominant":sign_matches,"signConsistencyCriterionMet":sign_ok,
                "blocksWithAbsRhoAtLeastThreshold":magnitude_blocks,"magnitudeCriterionMet":magnitude_ok,
                "heuristicOnly":True}
    report={"schemaVersion":1,"protocolId":"national-volume-lead-descriptive-screen-2026-10-09",
        "resultType":"descriptive_feasibility_screen","sourceDigests":source_digests or {},
        "featureDefinition":"log(mean signed daily two-market volumeKg over observed days in [o-8,o-2] / mean over observed days in [o-29,o-9]); cutoff=o-2; no date filling; future target log(price/as-of national price)",
        "targetBlocksByTargetDate":[[a.isoformat(),b.isoformat()] for a,b in BLOCKS],"horizonsDays":list(HORIZONS),
        "blocks":blocks,"followOnEligibility":eligibility,
        "interpretation":"Spearman sign and magnitude are descriptive only; no forecasting model, significance test, or MAPE claim is made.",
        "reportSha256":None}
    report["reportSha256"]=_sha(json.dumps({k:v for k,v in report.items() if k!="reportSha256"},sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode())
    return report


def _volume_source(raw:dict)->tuple[dict[str,dict[dt.date,float|None]],dict]:
    if raw.get("schemaVersion")!=1 or not isinstance(raw.get("products"),list):raise ValueError("two-market prices must use schemaVersion 1")
    product_docs={}
    for product in raw["products"]:
        pid=product.get("id")
        if pid not in dashboard.PRODUCTS or pid in product_docs:raise ValueError("unknown or duplicate two-market product")
        product_docs[pid]=product
    if set(product_docs)!=set(dashboard.PRODUCTS):raise ValueError("two-market prices must contain all dashboard products")
    result={};coverage={}
    for product in PRODUCTS:
        dates={};states=Counter();doc=product_docs[product]
        observations=doc.get("observations")
        if not isinstance(observations,list):raise ValueError("product observations must be a list")
        for row in observations:
            if not isinstance(row,dict):raise ValueError("observation row must be an object")
            day=dt.date.fromisoformat(row["date"])
            if day in dates:raise ValueError("duplicate two-market volume date")
            raw_volume=row.get("volumeKg")
            volume=_finite_signed(raw_volume)
            if raw_volume is None:state="null"
            elif volume is None:state="invalid_or_nonfinite"
            elif volume<0:state="negative_signed"
            elif volume==0:state="zero"
            else:state="positive"
            states[state]+=1;dates[day]=volume
        result[product]=dates
        coverage[product]={"observationRows":len(observations),"dateCount":len(dates),"dateStart":min(dates).isoformat() if dates else None,
                           "dateEnd":max(dates).isoformat() if dates else None,"volumeValueStates":dict(sorted(states.items()))}
    return result,coverage


def load_inputs(prices_path:Path=INPUT_PRICES,archive_path:Path=INPUT_ARCHIVE)->tuple[dict,dict,dict,dict]:
    prices_bytes=prices_path.read_bytes();archive_bytes=archive_path.read_bytes()
    if _sha(prices_bytes)!=PRICES_SHA256:raise ValueError("two-market prices file SHA-256 differs from frozen input")
    if _sha(archive_bytes)!=ARCHIVE_SHA256:raise ValueError("national archive SHA-256 differs from frozen input")
    raw_prices=direct_horizon._read(prices_path)
    volumes,volume_coverage=_volume_source(raw_prices)
    national_source=national_transfer.parse_archive(archive_bytes,expected_sha256=ARCHIVE_SHA256)
    national_prices=price_selection._prices(national_source)
    national_coverage={p:{"observationRows":len(national_source["observations"][p]),
        "positivePriceDays":len(national_prices[p]),"dateStart":min(national_source["observations"][p]) if national_source["observations"][p] else None,
        "dateEnd":max(national_source["observations"][p]) if national_source["observations"][p] else None,
        "nonpositiveOrIncompletePriceDays":sum(v.get("price") is None for v in national_source["observations"][p].values())} for p in PRODUCTS}
    digests={"twoMarketPricesSha256":_sha(prices_bytes),"nationalArchiveSha256":_sha(archive_bytes),
             "twoMarketSourceCoverage":volume_coverage,"nationalArchiveCoverage":national_coverage}
    return national_prices,volumes,digests,national_source


def _write_json_create_only(path:Path,doc:dict)->None:
    path.parent.mkdir(parents=True,exist_ok=True)
    body=dict(doc);body["reportSha256"]=_sha(json.dumps({k:v for k,v in body.items() if k!="reportSha256"},sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False).encode())
    payload=(json.dumps(body,sort_keys=True,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n").encode()
    fd=os.open(path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,"wb") as f:f.write(payload);f.flush();os.fsync(f.fileno())


def main(argv:list[str]|None=None)->int:
    parser=argparse.ArgumentParser(description="Offline descriptive two-market volume lead screen")
    parser.add_argument("--prices",default=str(INPUT_PRICES));parser.add_argument("--archive",default=str(INPUT_ARCHIVE));parser.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    national,volumes,digests,_=load_inputs(Path(args.prices),Path(args.archive))
    report=build_screen(national,volumes,digests);_write_json_create_only(Path(args.output),report)
    return 0


if __name__=="__main__":raise SystemExit(main())
