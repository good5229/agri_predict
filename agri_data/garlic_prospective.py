"""Issue the frozen daily Ridge/static-blend garlic forecast prospectively."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import uuid
from zoneinfo import ZoneInfo

from . import dashboard, direct_horizon as dh, market_mix, modeling

ROOT = Path(__file__).resolve().parents[1]
ISSUE_DATE = dt.date(2026, 10, 7)
KST = ZoneInfo("Asia/Seoul")
HORIZONS = tuple(range(1, 11))
MIN_TRAIN_DAYS = 90
MAX_SOURCE_AGE_DAYS = 14
STATIC_REPORT = ROOT / "outputs/experiments/static-blend-20261006/report.json"
PRICES = ROOT / "outputs/daily/2026-10-06/20261006T090116-f908e0a4/prices.json"
DB_PATH = ROOT / "data/agri.sqlite"
OUTPUT = ROOT / "outputs/forecasts/garlic/2026-10-07.json"
STATIC_SHA256 = "64df30484cce60ddbadca52f42ddebfc9469c1265efc19ede110715a92ca9aa3"
PRICES_SHA256 = "47cdd787b66a6b0c6e42b750d0b3f7e845dbde4d7ab4fdeed3c8c40472da5605"
WEIGHTS = {h: (1.0 if h in (3, 4) else 0.75) for h in HORIZONS}
PROTOCOL = {
    "id": "garlic-fixed-static-blend-prospective-2026-10-07",
    "target": "garlic signed auction/kg daily price snapshot; issue-date input cutoff is strictly before issue date",
    "fit": "modeling._supervised_samples pooled across horizons 1..10; target < issue date; StandardScaler + Ridge(alpha=10); minimum 90 distinct target dates",
    "weights": {str(h): WEIGHTS[h] for h in HORIZONS},
    "forecast": "blend = persistence + frozen weight * (pooled Ridge - persistence)",
    "sourceAgeDays": MAX_SOURCE_AGE_DAYS,
    "retrospectiveReference": "static-blend-20261006 frozen selection; retrospective metric is not prospective evidence",
}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _valid_sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _daily_output_path(issue_date):
    return ROOT / "outputs/forecasts/garlic" / f"{issue_date.isoformat()}.json"


def _validate_weights(report, report_sha256):
    if report_sha256 != STATIC_SHA256:
        raise ValueError("static-blend report SHA-256 does not match frozen selection")
    selected = {}
    for row in report.get("results", []):
        if row.get("productId") != "garlic":
            continue
        h = row.get("horizon")
        if isinstance(h, bool) or h not in HORIZONS or h in selected:
            raise ValueError("static-blend garlic horizon selection is missing, invalid, or duplicated")
        selection = row.get("selection") or {}
        model = selection.get("selectedModel"); weight = selection.get("weight")
        if model != "ridge" or weight != WEIGHTS[h]:
            raise ValueError(f"static-blend frozen garlic selection mismatch for horizon {h}")
        selected[h] = {"model": model, "weight": weight}
    if set(selected) != set(HORIZONS):
        raise ValueError("static-blend report must contain all ten frozen garlic selections")
    return selected


def _finite_positive(value):
    try: x=float(value)
    except (TypeError, ValueError, OverflowError): return False
    return math.isfinite(x) and x > 0


def build(raw, static_report, issue_date: dt.date, issued_at: dt.datetime, prices_sha256: str,
          static_sha256: str, selected_provenance: dict, *, prices_path=PRICES):
    """Pure issue builder; raw observations and provenance are injected for offline tests."""
    if issue_date < ISSUE_DATE:
        raise ValueError("retrospective garlic issue dates are not allowed")
    if issued_at.tzinfo is None or issued_at.astimezone(KST).date() != issue_date:
        raise ValueError("issuedAtKst must fall on issueDateKst")
    if not _valid_sha256(prices_sha256):
        raise ValueError("prices snapshot SHA-256 is invalid")
    if issue_date == ISSUE_DATE and prices_sha256 != PRICES_SHA256:
        raise ValueError("first issue prices snapshot SHA-256 does not match the pinned verified input")
    selected = _validate_weights(static_report, static_sha256)
    raw_cutoff = dt.date.fromisoformat(raw["meta"]["contextEnd"])
    if raw_cutoff >= issue_date:
        raise ValueError("snapshot contextEnd must be strictly before issue date")
    core = dh._observations(raw)
    series = core.get("garlic")
    if not series:
        raise ValueError("frozen snapshot has no garlic target series")
    if any(day > raw_cutoff for product_series in core.values() for day in product_series):
        raise ValueError("snapshot observations extend beyond the declared contextEnd cutoff")
    asof = {day: row for day, row in series.items() if day < issue_date}
    if not asof:
        raise ValueError("no strictly pre-issue garlic observations")
    latest_source = max(asof)
    source_age = (issue_date - latest_source).days
    samples, target_days = modeling._supervised_samples(asof, issue_date - dt.timedelta(days=1), {})
    # Explicitly enforce the target cutoff even if the shared feature builder changes later.
    samples = [row for row in samples if row[0] < issue_date and row[1] < issue_date]
    target_days = {row[0] for row in samples}
    fit_state = {"trainingTargetDays": len(target_days), "trainingSampleCount": len(samples),
                 "trainingTargetEnd": max(target_days).isoformat() if target_days else None,
                 "issueCutoffExclusive": issue_date.isoformat(), "status": "held", "reason": None}
    model_state = None
    model = None
    if len(target_days) < MIN_TRAIN_DAYS:
        fit_state["reason"] = "fewer_than_90_distinct_training_target_days"
    else:
        try:
            import numpy as np
            from threadpoolctl import threadpool_limits
            try:
                thread_limits = threadpool_limits(limits=1)
            except AttributeError:
                # Some bundled BLAS builds expose no configuration string, which
                # makes threadpoolctl fail before it can enter its context manager.
                # The estimator can still fit; retain that fallback for issuance.
                from contextlib import nullcontext
                thread_limits = nullcontext()
            with thread_limits:
                model = modeling._new_estimator("ridge")
                model.fit(np.asarray([r[3] for r in samples], dtype=float), np.asarray([r[4] for r in samples], dtype=float))
            scaler, ridge = model.named_steps["standardscaler"], model.named_steps["ridge"]
            model_state = {"features": list(modeling.FEATURE_NAMES), "scalerMean": scaler.mean_.tolist(),
                           "scalerScale": scaler.scale_.tolist(), "ridgeCoef": ridge.coef_.tolist(),
                           "ridgeIntercept": float(ridge.intercept_), "ridgeAlpha": float(ridge.alpha)}
            params=[*model_state["scalerMean"],*model_state["scalerScale"],*model_state["ridgeCoef"],model_state["ridgeIntercept"],model_state["ridgeAlpha"]]
            if not all(math.isfinite(float(x)) for x in params) or any(float(x)<=0 for x in model_state["scalerScale"]):
                raise ValueError("non_finite_or_invalid_model_parameters")
            fit_state["status"] = "fitted"
        except Exception as exc:
            model=None; model_state=None
            fit_state["reason"] = "ridge_fit_failed:" + type(exc).__name__
    fit_state["sourceDate"] = latest_source.isoformat()
    fit_state["sourceAgeDays"] = source_age
    if source_age > MAX_SOURCE_AGE_DAYS:
        fit_state["sourceHoldReason"] = "latest_source_older_than_14_days"
    cache={}; forecasts=[]
    for h in HORIZONS:
        row={"horizon":h,"targetDate":(issue_date+dt.timedelta(days=h)).isoformat(),"weight":selected[h]["weight"],
             "ridgePrediction":None,"persistencePrediction":asof[latest_source][0],"blendPrediction":None,
             "sourceDate":latest_source.isoformat(),"sourceAgeDays":source_age,
             "status":"held","holdReason":None}
        feature=modeling._feature_row(asof,issue_date,issue_date+dt.timedelta(days=h),h,cache)
        if feature is None:
            row["holdReason"]="latest_source_older_than_14_days" if source_age > MAX_SOURCE_AGE_DAYS else "no_valid_asof_feature_source"
        else:
            source_day,vector=feature; age=(issue_date-source_day).days
            if source_day > issue_date - dt.timedelta(days=1): raise ValueError("forecast feature source is not strictly before issue date")
            row["sourceDate"]=source_day.isoformat(); row["sourceAgeDays"]=age
            persistence=asof[source_day][0]; row["persistencePrediction"]=persistence
            if fit_state["status"] != "fitted": row["holdReason"]=fit_state["reason"]
            elif age > MAX_SOURCE_AGE_DAYS: row["holdReason"]="latest_source_older_than_14_days"
            else:
                try:
                    import numpy as np
                    delta=float(model.predict(np.asarray([vector],dtype=float))[0])
                    if not math.isfinite(delta): raise ValueError("non_finite_log_ratio_prediction")
                    ridge_value,_=modeling._inverse_log_ratio(persistence,delta)
                    blend=persistence+selected[h]["weight"]*(ridge_value-persistence)
                    if not _finite_positive(ridge_value) or not _finite_positive(blend): raise ValueError("non_finite_or_nonpositive_forecast")
                    row.update(ridgePrediction=float(ridge_value),blendPrediction=float(blend),status="available")
                except Exception as exc: row["holdReason"]="prediction_failed:"+str(exc)[:120]
        forecasts.append(row)
    available=sum(r["status"]=="available" for r in forecasts)
    return {"schemaVersion":1,"protocol":PROTOCOL,"status":"available" if available==len(HORIZONS) else "held_or_partial",
            "issueDateKst":issue_date.isoformat(),"issuedAtKst":issued_at.astimezone(KST).isoformat(timespec="seconds"),
            "input":{"pricesPath":str(prices_path),"pricesSha256":prices_sha256,"snapshotCutoff":raw_cutoff.isoformat(),
                     "staticBlendReportPath":str(STATIC_REPORT),"staticBlendReportSha256":static_sha256,
                     "selectedResponseHashes":selected_provenance.get("selectedRequestHashes",{}),
                     "selectedResponseCount":selected_provenance.get("selectedRequestCount"),
                     "selectedQueryStates":selected_provenance.get("selectedQueryStates"),
                     "selectedScopesSha256":selected_provenance.get("selectedScopesSha256"),
                     "broadScopesSha256":selected_provenance.get("broadScopesSha256"),
                     "reconciliation":selected_provenance.get("reconciliation")},
            "fit":fit_state,"model":model_state,"forecastCount":available,"forecasts":forecasts,
            "interpretation":"prospective validation issue only; no target actuals exist at issuance; historical selection metrics do not establish forecast improvement"}


def _immutable(path: Path, payload: bytes):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temp.open("xb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
        os.link(temp,path)
    finally:
        try: temp.unlink()
        except FileNotFoundError: pass


def run(prices_path=None, db_path=DB_PATH, static_path=STATIC_REPORT, output_path=None, *, now=None):
    now=(now or dt.datetime.now(KST)).astimezone(KST)
    if now.date()<ISSUE_DATE: raise ValueError("real issuance is restricted to 2026-10-07 KST or later")
    if prices_path is None:
        if now.date()!=ISSUE_DATE: raise ValueError("a current-day previously verified prices snapshot must be supplied for follow-on issues")
        prices_path=PRICES
    expected_output=_daily_output_path(now.date())
    if output_path is None: output_path=expected_output
    if Path(output_path).resolve()!=expected_output.resolve(): raise ValueError("output must be the immutable file for the actual KST issue date")
    for output, source in ((output_path,prices_path),(output_path,db_path),(output_path,static_path)):
        if output.resolve()==source.resolve(): raise ValueError("output must not replace an input or database")
    raw_bytes=prices_path.read_bytes(); prices_hash=_sha(raw_bytes)
    if now.date()==ISSUE_DATE and prices_hash!=PRICES_SHA256: raise ValueError("first issue prices snapshot SHA-256 does not match frozen verified input")
    static_bytes=static_path.read_bytes(); static_hash=_sha(static_bytes)
    if static_hash!=STATIC_SHA256: raise ValueError("static-blend report SHA-256 does not match frozen selection")
    raw=json.loads(raw_bytes); static_report=json.loads(static_bytes); core=dh._observations(raw)
    _,prov=market_mix.extract_market_data(db_path,raw,core)
    selected={**prov["selected"],"broadScopesSha256":prov["broad"].get("scopesSha256"),"reconciliation":prov["reconciliation"]}
    payload=build(raw,static_report,now.date(),now,prices_hash,static_hash,selected,prices_path=prices_path)
    _immutable(output_path,_json_bytes(payload))
    return {"output":str(output_path),"sha256":_sha(_json_bytes(payload)),"status":payload["status"],"forecastCount":payload["forecastCount"]}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prices",type=Path); parser.add_argument("--db",type=Path,default=DB_PATH)
    parser.add_argument("--static-report",type=Path,default=STATIC_REPORT); parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    print(json.dumps(run(args.prices,args.db,args.static_report,args.output),ensure_ascii=False)); return 0


if __name__=="__main__": raise SystemExit(main())
