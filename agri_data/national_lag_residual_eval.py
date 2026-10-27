"""Frozen offline evaluation of a national-price lag residual correction."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import uuid
from collections import Counter, defaultdict

from . import dashboard, direct_horizon as dh, kma_residual_eval as common, national_transfer

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_PATH = ROOT / "data/raw/national-wholesale-20240708-20260630.zip"
PRICE_PATH = dh.INPUT_PRICES
REPORT_PATH = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
OUTPUT_DIR = ROOT / "outputs/experiments/national-lag-residual-20261007"
ARCHIVE_SHA256 = national_transfer.ARCHIVE_SHA256
PRICE_SHA256 = common.PRICE_SHA256
REPORT_SHA256 = common.REPORT_SHA256
PRODUCTS = tuple(dashboard.PRODUCTS)
NATIONAL_PRODUCTS = tuple(national_transfer.PRODUCT_CODES)
HORIZONS = tuple(range(1, 11))
TRAIN_START, TRAIN_END = dt.date(2024, 7, 8), dt.date(2025, 9, 1)
EVAL_START, EVAL_END = dt.date(2025, 9, 25), dt.date(2026, 9, 24)
MIN_TRAIN_DAYS, MAX_AGE = 90, 14
BANDWIDTHS = (9, 28)
FAMILY_SIZE = 5 * 10 * 2 * 2
EXPECTED_EVAL_PAIRS = 12_508
FEATURES = ("log_national_to_two_market", "log_signed_quantity", "national_age_days", "national_log_change_9d", "national_log_change_9d_available")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _feature(origin: dt.date, obs: dict[str, dict], core: dict[str, str]):
    series = obs
    def positive(day, row):
        try:
            return row.get("price") is not None and float(row["price"]) > 0 and math.isfinite(float(row["price"])) and float(row["quantitySum"]) > 0 and math.isfinite(float(row["quantitySum"]))
        except (TypeError, ValueError, OverflowError):
            return False
    eligible = [(dt.date.fromisoformat(d), r) for d, r in series.items() if dt.date.fromisoformat(d) <= origin - dt.timedelta(days=2) and positive(dt.date.fromisoformat(d), r)]
    day, row = max(eligible, key=lambda x: x[0]) if eligible else (None, None)
    if day is None or (origin - dt.timedelta(days=2) - day).days > MAX_AGE:
        return None, {"status": "national_missing_or_stale", "nationalDate": day.isoformat() if day else None}
    # Core series is supplied in the per-product observation metadata by caller.
    core_day = max((dt.date.fromisoformat(d) for d in core if dt.date.fromisoformat(d) <= origin), default=None)
    if core_day is None or (origin-core_day).days > MAX_AGE:
        return None, {"status": "two_market_missing_or_stale", "nationalDate": day.isoformat(), "twoMarketDate": core_day.isoformat() if core_day else None}
    core_value = core[core_day.isoformat()]
    # dashboard observation parsing returns (price, weight), not a scalar.
    core_price = float(core_value[0] if isinstance(core_value, (tuple, list)) else core_value)
    national_price = float(row["price"])
    if national_price <= 0 or core_price <= 0:
        return None, {"status": "nonpositive_feature_source", "nationalDate": day.isoformat(), "twoMarketDate": core_day.isoformat()}
    lag_cut = origin - dt.timedelta(days=9)
    lag_candidates = [(dt.date.fromisoformat(d), r) for d, r in series.items() if dt.date.fromisoformat(d) <= lag_cut and positive(dt.date.fromisoformat(d), r)]
    lag_day, lag_row = max(lag_candidates, key=lambda x: x[0]) if lag_candidates else (None, None)
    lag_ok = lag_day is not None and (lag_cut-lag_day).days <= MAX_AGE
    delta = math.log(float(row["price"])/float(lag_row["price"])) if lag_ok else 0.0
    features = [math.log(national_price/core_price), math.log(float(row["quantitySum"])), float((origin-day).days), delta, float(lag_ok)]
    return features, {"status": "available", "nationalDate": day.isoformat(), "twoMarketDate": core_day.isoformat(), "lagDate": lag_day.isoformat() if lag_ok else None, "nationalPrice": national_price, "twoMarketPrice": core_price, "signedQuantity": float(row["quantitySum"])}


def _adjusted_prediction(direct: float, residual: float) -> float | None:
    if not math.isfinite(residual):
        return None
    adjusted = direct * math.exp(max(-.2, min(.2, residual)))
    return adjusted if math.isfinite(adjusted) and adjusted > 0 else None


def _holm(items):
    order = sorted(range(len(items)), key=lambda i: items[i]["pValue"])
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (len(items)-rank)*items[idx]["pValue"]))
        items[idx]["holmPValue"] = running
        items[idx]["holmSignificant"] = running < .05


def build(prices, report, archive, *, hashes=None):
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    direct = common._direct_rows(report, prices)
    national = archive["observations"]
    core = dh._observations(prices)
    feature_cache = {}
    for key, row in direct.items():
        product = row["product"]
        if product not in NATIONAL_PRODUCTS:
            feature_cache[key] = (None, {"status": "garlic_direct_fallback"})
            continue
        core_series = {d.isoformat(): v for d, v in core[product].items()}
        feature_cache[key] = _feature(row["origin"], national[product], core_series)
    models, audits = {}, {}
    for product in PRODUCTS:
        for h in HORIZONS:
            key = f"{product}/{h}"
            paired = [r for r in direct.values() if r["product"] == product and r["horizon"] == h and TRAIN_START <= r["target"] <= TRAIN_END]
            train = [r for r in paired if feature_cache[(product,h,r["target"])][0] is not None]
            ndays = len({r["target"] for r in train})
            audit = {"pairedTrainingRows": len(paired), "featureAvailableTrainingRows": len(train), "distinctTrainingTargets": ndays, "status": "garlic_direct_fallback" if product == "garlic" else "insufficient_training"}
            if product != "garlic" and ndays >= MIN_TRAIN_DAYS:
                X = np.asarray([feature_cache[(product,h,r["target"])][0] for r in train], dtype=float)
                y = np.asarray([math.log(r["actual"]/r["direct"]) for r in train], dtype=float)
                scaler = StandardScaler().fit(X)
                scaler.scale_[scaler.scale_ == 0] = 1.0
                model = Ridge(alpha=100, fit_intercept=False).fit(scaler.transform(X), y)
                models[key] = scaler, model
                audit.update(status="fitted", trainTargetStart=min(r["target"] for r in train).isoformat(), trainTargetEnd=max(r["target"] for r in train).isoformat())
            audits[key] = audit
    eligible = [r for r in direct.values() if EVAL_START <= r["target"] <= EVAL_END]
    preds, grouped = [], defaultdict(list)
    for row in eligible:
        key = (row["product"],row["horizon"],row["target"])
        vec, provenance = feature_cache[key]
        candidate, fallback = row["direct"], "model_unavailable"
        if row["product"] == "garlic": fallback = "garlic_direct_fallback"
        elif vec is None: fallback = provenance["status"]
        elif f"{row['product']}/{row['horizon']}" in models:
            scaler, model = models[f"{row['product']}/{row['horizon']}"]
            residual = float(model.predict(scaler.transform(np.asarray([vec])))[0])
            adjusted = _adjusted_prediction(row["direct"], residual)
            if adjusted is not None: candidate, fallback = adjusted, None
            else: fallback = "nonfinite_or_nonpositive_candidate"
        out = {**row, "candidate": candidate, "national": provenance, "fallbackReason": fallback}
        preds.append(out); grouped[f"{row['product']}/{row['horizon']}"].append(out)
    comparisons=[]
    for product in PRODUCTS:
        for h in HORIZONS:
            rows=grouped[f"{product}/{h}"]
            for comparator in ("direct","persistence"):
                for loss in ("ae","ape"):
                    diff={r["target"]: ((abs(r["actual"]-r["candidate"])-abs(r["actual"]-r[comparator])) if loss=="ae" else 100*(abs(r["actual"]-r["candidate"])-abs(r["actual"]-r[comparator]))/r["actual"]) for r in rows}
                    m=lambda field: common._metrics(rows,field)
                    record={"product":product,"horizon":h,"comparator":comparator,"loss":loss,"n":len(rows),"candidateMetric":m("candidate")["mae" if loss=="ae" else "mape"],"comparatorMetric":m(comparator)["mae" if loss=="ae" else "mape"],"dm":{str(b):dh._dm(diff,b) for b in BANDWIDTHS}}
                    comparisons.append(record)
    for b in BANDWIDTHS:
        family=[x["dm"][str(b)] for x in comparisons]
        if len(family)!=FAMILY_SIZE: raise AssertionError("fixed 200-test family changed")
        _holm(family)
    horizons={}; crop_mape={}; crop_metrics={}
    for product in PRODUCTS:
        vals=[]
        for h in HORIZONS:
            rows=grouped[f"{product}/{h}"]
            values={field:common._metrics(rows,field) for field in ("candidate","direct","persistence")}
            vals.append(values["candidate"]["mape"])
            horizons[f"{product}/{h}"]={"n":len(rows),"metrics":values,"nationalAvailable":sum(r["national"]["status"]=="available" for r in rows),"fallbackRows":sum(r["fallbackReason"] is not None for r in rows),"fallbackReasons":dict(Counter(r["fallbackReason"] for r in rows if r["fallbackReason"]))}
        pooled=[r for h in HORIZONS for r in grouped[f"{product}/{h}"]]
        crop_metrics[product]={field:common._metrics(pooled,field) for field in ("candidate","direct","persistence")}
        crop_mape[product]=sum(vals)/len(vals) if all(v is not None for v in vals) else None
    macro=sum(crop_mape.values())/len(PRODUCTS) if all(v is not None for v in crop_mape.values()) else None
    conclusions=[]
    for product in PRODUCTS:
        for h in HORIZONS:
            group=[x for x in comparisons if x["product"]==product and x["horizon"]==h]
            tests=[x["dm"]["9"] for x in group]
            better=all(x["candidateMetric"] is not None and x["candidateMetric"]<x["comparatorMetric"] for x in group)
            enough=all(t["status"]=="tested" for t in tests)
            improve=better and enough and all(t["holmSignificant"] and t["meanLossDifference"]<0 for t in tests)
            worse=any(t["status"]=="tested" and t["holmSignificant"] and t["meanLossDifference"]>0 for t in tests)
            conclusions.append({"product":product,"horizon":h,"conclusion":"improvement" if improve else "worsening" if worse else "inconclusive","inferenceAvailable":enough,"allPointMetricsLower":better,"allHac9HolmTestsImprove":improve,"coverageNondeclining":True})
    strata=[]
    for product in PRODUCTS:
        for h in HORIZONS:
            rows=grouped[f"{product}/{h}"]
            for name,fn in (("year",lambda d:str(d.year)),("season",lambda d:(d.month-1)//3+1)):
                buckets=defaultdict(list)
                for r in rows: buckets[fn(r["target"])].append(r)
                for label,part in sorted(buckets.items()):
                    strata.append({"product":product,"horizon":h,"stratum":name,"value":label,"n":len(part),"candidate":common._metrics(part,"candidate"),"direct":common._metrics(part,"direct"),"persistence":common._metrics(part,"persistence")})
    return {"schemaVersion":1,"protocol":{"id":"national-lag-residual-2026-10-07","trainingTargets":[TRAIN_START.isoformat(),TRAIN_END.isoformat()],"evaluationTargets":[EVAL_START.isoformat(),EVAL_END.isoformat()],"nationalAvailabilityCutoff":"origin-2 inclusive","nationalMaxAgeDays":MAX_AGE,"lagCutoff":"origin-9 inclusive","lagMaxAgeDays":MAX_AGE,"features":list(FEATURES),"featureProducts":{p:national_transfer.PRODUCT_CODES[p] for p in NATIONAL_PRODUCTS},"garlic":"full direct Ridge fallback; archive 1209 not split for peeled garlic","model":{"scaler":"training-only StandardScaler; constant scale=1","estimator":"Ridge(alpha=100,fit_intercept=False)","clipLogCorrection":[-.2,.2]},"minimumDistinctTrainingTargets":MIN_TRAIN_DAYS,"comparisonFamilySizePerBandwidth":FAMILY_SIZE,"bandwidthsCalendarDays":list(BANDWIDTHS),"holmAlpha":.05},"hashes":hashes or {},"archiveAudit":{"archiveSha256":archive["archiveSha256"],"sourceRowCount":archive["sourceRowCount"],"exactDuplicateRowCount":archive["exactDuplicateRowCount"],"periods":archive["periods"]},"fitAudits":audits,"coverage":{"evaluationPairedRows":len(preds),"byProductHorizon":horizons},"cropMetrics":crop_metrics,"cropEqualHorizonMape":crop_mape,"fiveCropEqualWeightMape":macro,"predictions":[{**r,"origin":r["origin"].isoformat(),"target":r["target"].isoformat()} for r in preds],"comparisons":comparisons,"jointConclusions":conclusions,"yearSeasonMetrics":strata,"limitations":"Retrospective revised national archive; first publication time and then-current version are unknown; not prospective or official-target evidence."}


def _write_new(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("xb") as stream: stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.link(tmp,path)
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass


def run(archive_path=ARCHIVE_PATH, prices_path=PRICE_PATH, report_path=REPORT_PATH, output_dir=OUTPUT_DIR):
    ab,pb,rb=Path(archive_path).read_bytes(),Path(prices_path).read_bytes(),Path(report_path).read_bytes()
    hashes={"archiveSha256":sha256(ab),"pricesSha256":sha256(pb),"directReportSha256":sha256(rb)}
    if hashes != {"archiveSha256":ARCHIVE_SHA256,"pricesSha256":PRICE_SHA256,"directReportSha256":REPORT_SHA256}: raise ValueError("input SHA-256 differs from frozen protocol")
    archive=national_transfer.parse_archive(ab)
    prices=json.loads(pb); report=json.loads(rb)
    direct=common._direct_rows(report,prices)
    eval_count=sum(EVAL_START <= row["target"] <= EVAL_END for row in direct.values())
    if eval_count != EXPECTED_EVAL_PAIRS:
        raise ValueError(f"frozen evaluation pair count changed: expected {EXPECTED_EVAL_PAIRS}, found {eval_count}")
    manifest={"schemaVersion":1,"protocolId":"national-lag-residual-2026-10-07","hashes":hashes,"status":"inputs_validated_fit_not_started"}
    out=Path(output_dir); _write_new(out/"manifest.json",(json.dumps(manifest,sort_keys=True,indent=2)+"\n").encode())
    result=build(prices,report,archive,hashes=hashes)
    encoded=(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    _write_new(out/"report.json",encoded)
    return {"output":str(out/"report.json"),"sha256":sha256(encoded),"pairedRows":len(result["predictions"])}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--archive",type=Path,default=ARCHIVE_PATH); p.add_argument("--prices",type=Path,default=PRICE_PATH); p.add_argument("--report",type=Path,default=REPORT_PATH); p.add_argument("--output-dir",type=Path,default=OUTPUT_DIR)
    a=p.parse_args(argv); print(json.dumps(run(a.archive,a.prices,a.report,a.output_dir),ensure_ascii=False)); return 0

if __name__=="__main__": raise SystemExit(main())
