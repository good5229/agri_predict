"""Read-only descriptive diagnostics for the frozen five-crop specialist run.

All diagnostics are descriptive, not causal or inferential. They are fixed in
advance: quarter/horizon paired AE and APE against the frozen comparator;
candidate log-correction sign, magnitude, calibration error, and overshoot;
candidate/ablation/extra MAPE; and Pearson/Spearman association between the
extra-signal log ratio and the residual target. No result-dependent tuning is
performed. The output contains aggregates only, never row-level actuals.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from . import five_crop_specialists as spec

ROOT = spec.ROOT
RUN_DIR = Path("outputs/experiments/five-crop-specialists-20261009")
MANIFEST_PATH = RUN_DIR / "forecast-manifest.json"
REPORT_PATH = RUN_DIR / "report.json"
MANIFEST_FILE_SHA256 = "1497ee362945fcab45353a7b2403be752af3bf63c5f584cc0aedb2cc111ceaa8"
REPORT_FILE_SHA256 = "0839147b64716c92773e191b55098e0065b10e92d29913a27d55813fc63ddc51"
FROZEN_BASELINE = {"cabbage": spec.DIRECT, "radish": spec.FIXED,
                   "onion": spec.EXTRA, "garlic": spec.FIXED, "pepper": spec.FIXED}
METHODS_TO_COMPARE = (spec.CANDIDATE, spec.ABLATION, spec.FIXED, spec.EXTRA)
NEAR_ZERO_LOG_CORRECTION = 0.01


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _positive(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered=sorted(values)
    position=(len(ordered)-1)*probability
    lower=math.floor(position); upper=math.ceil(position)
    return ordered[lower]+(position-lower)*(ordered[upper]-ordered[lower])


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.
        for k in order[i:j]:
            ranks[k] = rank
        i = j
    return ranks


def _corr(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = _mean(x), _mean(y)
    dx, dy = [v - mx for v in x], [v - my for v in y]
    den = math.sqrt(sum(v*v for v in dx) * sum(v*v for v in dy))
    return sum(a*b for a, b in zip(dx, dy)) / den if den > 0 else None


def _association(rows: list[tuple[float, float]]) -> dict:
    x, y = [r[0] for r in rows], [r[1] for r in rows]
    return {"n": len(rows), "pearson": _corr(x, y),
            "spearman": _corr(_rank(x), _rank(y)) if len(rows) >= 2 else None}


def _quarter(target: str) -> str:
    day = dt.date.fromisoformat(target)
    return f"{day.year}-Q{(day.month - 1)//3 + 1}"


def _paired(rows: list[dict], baseline: str) -> dict:
    usable = []
    for row in rows:
        y = _positive(row.get("actual"))
        c = _positive(row["predictions"].get(spec.CANDIDATE))
        b = _positive(row["predictions"].get(baseline))
        if y is not None and c is not None and b is not None:
            usable.append((y, c, b))
    if not usable:
        return {"n": 0, "candidateMae": None, "baselineMae": None,
                "candidateMape": None, "baselineMape": None,
                "candidateMinusBaselineMae": None, "candidateMinusBaselineMapePp": None,
                "candidateAeWinShare": None, "candidateApeWinShare": None}
    ae_c = [abs(y-c) for y,c,_ in usable]
    ae_b = [abs(y-b) for y,_,b in usable]
    ape_c = [100*e/y for (y,_,_),e in zip(usable,ae_c)]
    ape_b = [100*e/y for (y,_,_),e in zip(usable,ae_b)]
    return {"n": len(usable), "candidateMae": _mean(ae_c), "baselineMae": _mean(ae_b),
            "candidateMape": _mean(ape_c), "baselineMape": _mean(ape_b),
            "candidateMinusBaselineMae": _mean([a-b for a,b in zip(ae_c,ae_b)]),
            "candidateMinusBaselineMapePp": _mean([a-b for a,b in zip(ape_c,ape_b)]),
            "candidateAeWinShare": sum(a < b for a,b in zip(ae_c,ae_b))/len(usable),
            "candidateApeWinShare": sum(a < b for a,b in zip(ape_c,ape_b))/len(usable)}


def _correction(rows: list[dict]) -> dict:
    usable = []
    for row in rows:
        y = _positive(row.get("actual"))
        fixed = _positive(row["predictions"].get(spec.FIXED))
        candidate = _positive(row["predictions"].get(spec.CANDIDATE))
        if y is not None and fixed is not None and candidate is not None:
            predicted = math.log(candidate / fixed)
            truth = math.log(y / fixed)
            usable.append((predicted, truth))
    if not usable:
        return {"n": 0, "trueNonzeroN": 0, "meanPredictedLogCorrection": None,
                "meanTrueLogCorrection": None, "meanAbsolutePredictedCorrection": None,
                "meanAbsoluteTrueCorrection": None, "correctionMae": None,
                "wrongSignShareAmongTrueNonzero": None, "zeroPredictionShareAmongTrueNonzero": None,
                "magnitudeOvershootShare": None,
                "meanOvershootMagnitude": None}
    p, t = [a for a,_ in usable], [b for _,b in usable]
    nonzero = [(a,b) for a,b in usable if b != 0]
    wrong = [a*b < 0 for a,b in nonzero]
    zero_pred = [a == 0 for a,_ in nonzero]
    overshoot = [max(0., abs(a)-abs(b)) for a,b in usable]
    return {"n":len(usable), "trueNonzeroN":len(nonzero),
            "meanPredictedLogCorrection":_mean(p), "meanTrueLogCorrection":_mean(t),
            "meanAbsolutePredictedCorrection":_mean([abs(v) for v in p]),
            "meanAbsoluteTrueCorrection":_mean([abs(v) for v in t]),
            "correctionMae":_mean([abs(a-b) for a,b in usable]),
            "wrongSignShareAmongTrueNonzero":sum(wrong)/len(wrong) if wrong else None,
            "zeroPredictionShareAmongTrueNonzero":sum(zero_pred)/len(zero_pred) if zero_pred else None,
            "magnitudeOvershootShare":sum(v>0 for v in overshoot)/len(overshoot),
            "meanOvershootMagnitude":_mean(overshoot)}


def _mape_by_method(rows: list[dict]) -> dict:
    result = {}
    for method in METHODS_TO_COMPARE:
        vals = []
        for row in rows:
            y, p = _positive(row.get("actual")), _positive(row["predictions"].get(method))
            if y is not None and p is not None:
                vals.append(100*abs(y-p)/y)
        result[method] = {"n":len(vals), "mape":_mean(vals)}
    return result


def _source_fallback(rows: list[dict]) -> dict:
    status = {"known":0,"unknown":0}
    value = {"fallback":0,"notFallback":0,"unknown":0}
    reasons: dict[str,int] = {}
    for row in rows:
        if row.get("sourceFallbackStatus") == "known":
            status["known"] += 1
            if row.get("sourceFallback") is True:
                value["fallback"] += 1
            elif row.get("sourceFallback") is False:
                value["notFallback"] += 1
            else:
                value["unknown"] += 1
        else:
            status["unknown"] += 1
            value["unknown"] += 1
        reason = row.get("sourceFallbackReason")
        if reason:
            label = json.dumps(reason, sort_keys=True, ensure_ascii=False) if isinstance(reason,(list,dict)) else str(reason)
            reasons[label] = reasons.get(label,0)+1
    return {"statusCounts":status,"fallbackCounts":value,"reasonCounts":reasons}


def _fallback_groups(rows: list[dict], baseline: str) -> dict:
    groups={"knownFallback":[],"knownNonFallback":[],"unknown":[]}
    for row in rows:
        if row.get("sourceFallbackStatus")!="known":
            groups["unknown"].append(row)
        elif row.get("sourceFallback") is True:
            groups["knownFallback"].append(row)
        elif row.get("sourceFallback") is False:
            groups["knownNonFallback"].append(row)
        else:
            groups["unknown"].append(row)
    return {name:{"plannedRows":len(items),"pairedVsStrongest":_paired(items,baseline)}
            for name,items in groups.items()}


def _mean_signal_target(rows: list[dict], *, forecasts: bool) -> dict:
    signal=[]; target=[]
    for row in rows:
        if forecasts:
            actual=_positive(row.get("actual")); fixed=_positive(row["predictions"].get(spec.FIXED))
            extra=_positive(row["predictions"].get(spec.EXTRA))
            if actual is None or fixed is None or extra is None: continue
            x=math.log(extra/fixed); y=math.log(actual/fixed)
        else:
            x= row.get("candidateFeatures",[])
            if len(x)<5: continue
            x=float(x[4]); y=row.get("targetLogRatio")
            if not isinstance(y,(int,float)) or not math.isfinite(y): continue
            y=float(y)
        if math.isfinite(x) and math.isfinite(y): signal.append(x);target.append(y)
    return {"n":len(signal),"meanSignalLogRatio":_mean(signal),"meanTargetLogRatio":_mean(target)}


def diagnose_rows(forecasts: list[dict], actuals: dict[tuple,float], training_audit: dict,
                  coverage_by_cell: dict | None = None) -> dict:
    """Produce fixed descriptive aggregates from synthetic or verified rows."""
    by_crop: dict[str,dict] = {}
    for crop in spec.CROPS:
        rows=[]
        for row in forecasts:
            if row.get("product") != crop:
                continue
            key=spec._key(row)
            actual=_positive(actuals.get(key))
            rows.append({**row,"actual":actual})
        baseline=FROZEN_BASELINE[crop]
        quarter_groups={}
        horizon_groups={}
        joint_groups={}
        for row in rows:
            q=_quarter(row["target"]); h=str(row["horizon"])
            quarter_groups.setdefault(q,[]).append(row)
            horizon_groups.setdefault(h,[]).append(row)
            joint_groups.setdefault(f"{q}/h{h}",[]).append(row)
        training_rows=training_audit.get(crop,{}).get("trainingRowProvenance",[])
        train_assoc=[]
        for r in training_rows:
            features=r.get("candidateFeatures",[])
            signal=_positive(r.get("extraSignal")); target=r.get("targetLogRatio")
            if signal is not None and len(features)>=5 and isinstance(target,(int,float)) and math.isfinite(target):
                train_assoc.append((float(features[4]),float(target)))
        test_assoc=[]
        for row in rows:
            y,fixed,extra=(_positive(row.get("actual")),_positive(row["predictions"].get(spec.FIXED)),
                           _positive(row["predictions"].get(spec.EXTRA)))
            if y is not None and fixed is not None and extra is not None:
                test_assoc.append((math.log(extra/fixed),math.log(y/fixed)))
        item={"frozenStrongestBaseline":baseline,"plannedRows":len(rows),
              "validActualRows":sum(_positive(r.get("actual")) is not None for r in rows),
              "positiveExtraSignalRows":sum(_positive(r["predictions"].get(spec.EXTRA)) is not None for r in rows),
              "pairedVsStrongest":_paired(rows,baseline),
              "byQuarter":{k:_paired(v,baseline) for k,v in sorted(quarter_groups.items())},
              "byHorizon":{k:_paired(v,baseline) for k,v in sorted(horizon_groups.items(),key=lambda kv:int(kv[0]))},
              "byQuarterAndHorizon":{k:_paired(v,baseline) for k,v in sorted(joint_groups.items())},
              "mapeByMethod":_mape_by_method(rows),
              "pairedVsStrongestBySourceFallback":_fallback_groups(rows,baseline),
              "correctionCalibration":_correction(rows),
              "meanSignalAndTargetLogRatios":{"trainingA":_mean_signal_target(training_rows,forecasts=False),
                                                "testB":_mean_signal_target(rows,forecasts=True)},
              "signalTargetAssociation":{"trainingA":_association(train_assoc),"testB_descriptive":_association(test_assoc)},
              "training":{"eligibleRows":training_audit.get(crop,{}).get("eligibleTrainingRows"),
                          "uniqueTargets":training_audit.get(crop,{}).get("uniqueTrainingTargets"),
                          "status":training_audit.get(crop,{}).get("status")},
              "sourceFallback":_source_fallback(rows),
              "candidateFallbackRows":sum(bool(r.get("fallback")) for r in rows)}
        if crop=="onion":
            distances=[];closer_fixed=[];ratios=[];near_zero=0
            candidate_ape=[];extra_ape=[];candidate_ae=[];extra_ae=[]
            for row in rows:
                y,fixed,extra,cand=(_positive(row.get("actual")),_positive(row["predictions"].get(spec.FIXED)),
                                     _positive(row["predictions"].get(spec.EXTRA)),_positive(row["predictions"].get(spec.CANDIDATE)))
                if None in (y,fixed,extra,cand): continue
                distances.append(abs(cand-extra))
                candidate_ae.append(abs(cand-y));extra_ae.append(abs(extra-y))
                candidate_ape.append(100*abs(cand-y)/y);extra_ape.append(100*abs(extra-y)/y)
                closer_fixed.append(abs(cand-fixed)<abs(cand-extra))
                denominator=math.log(extra/fixed)
                if abs(denominator)<NEAR_ZERO_LOG_CORRECTION:
                    near_zero+=1
                else:
                    ratios.append(math.log(cand/fixed)/denominator)
            item["marketSignalDistance"]={"n":len(distances),"meanAbsoluteCandidateMinusMarketExtra":_mean(distances),
                "candidateCloserToFixed50Share":sum(closer_fixed)/len(closer_fixed) if closer_fixed else None,
                "correctionRatioDefinition":"log(candidate/fixed50)/log(marketExtra/fixed50); report median and P10/P90 only when absolute denominator >= 0.01",
                "correctionRatioMedian":_quantile(ratios,.5),"correctionRatioP10":_quantile(ratios,.1),
                "correctionRatioP90":_quantile(ratios,.9),"correctionRatioEligibleN":len(ratios),
                "nearZeroDenominatorN":near_zero,
                "candidateMae":_mean(candidate_ae),"marketExtraMae":_mean(extra_ae),
                "candidateMinusMarketExtraMae":_mean([a-b for a,b in zip(candidate_ae,extra_ae)]),
                "candidateMape":_mean(candidate_ape),"marketExtraMape":_mean(extra_ape),
                "candidateMinusMarketExtraMapePp":_mean([a-b for a,b in zip(candidate_ape,extra_ape)])}
        if coverage_by_cell is not None:
            item["issuedAndTargetCoverage"]={str(h):coverage_by_cell.get(f"{crop}/{h}") for h in spec.HORIZONS}
        by_crop[crop]=item
    return {"schemaVersion":1,"analysis":"descriptive_failure_diagnosis",
            "period":{"start":spec.B[0].isoformat(),"end":spec.B[1].isoformat()},
            "interpretation":"Associations and error patterns are descriptive; they do not establish causes or independent confirmation.",
            "baselineRule":dict(FROZEN_BASELINE),"byCrop":by_crop}


def _verified_actual_map(docs: dict, forecasts: list[dict]) -> dict[tuple,float]:
    """Read B actuals only after the caller has validated complete forecast replay."""
    direct=spec._direct_rows(docs,"B",include_b_actual=True)
    signals=spec._signal_rows(docs)["B"]
    source_rows={}
    for crop in spec.CROPS:
        source_name,_=spec.SIGNAL_SOURCE[crop]
        rows=docs[source_name].get("originEvidence",[]) if crop in ("cabbage","onion") else docs[source_name].get("pairedRows",[])
        mapped={}
        for raw in rows:
            product=raw.get("product","radish" if crop=="radish" else None)
            if product!=crop or spec._window(spec._date(raw.get("target")))!="B": continue
            key=spec._key({**raw,"product":crop})
            if key in mapped: raise ValueError(f"duplicate B actual source key: {crop}")
            mapped[key]=raw
        source_rows[crop]=mapped
    forecast_map={spec._key(r):r for r in forecasts}
    if len(forecast_map)!=len(forecasts): raise ValueError("duplicate forecast key before diagnostic join")
    expected=set(forecast_map)
    if set(direct)!=expected or any(set(source_rows[c])!={k for k in expected if k[0]==c} for c in spec.CROPS):
        raise ValueError("exact B direct/source/forecast key sets differ")
    actuals={}
    for key,row in forecast_map.items():
        crop=key[0]; raw=source_rows[crop][key]; d=direct[key]
        field=spec.SIGNAL_SOURCE[crop][1]
        actual=_positive(raw.get("actual")); direct_actual=_positive(d.get("actualForScoring"))
        if actual is None or direct_actual is None or not math.isclose(actual,direct_actual,rel_tol=1e-10,abs_tol=1e-8):
            raise ValueError(f"B source/direct actual parity failure at {key}")
        if _positive(raw.get(field)) != _positive(row["predictions"].get(spec.EXTRA)):
            raise ValueError(f"B feature signal replay mismatch at {key}")
        source_direct=signals[crop][key].get("sourceDirect")
        if source_direct is None or not math.isclose(source_direct,d["direct"],rel_tol=1e-10,abs_tol=1e-8):
            raise ValueError(f"B source direct parity failure at {key}")
        actuals[key]=direct_actual
    return actuals


def run_diagnostic(*, root: Path=ROOT, manifest_path: Path|None=None,
                   report_path: Path|None=None) -> dict:
    manifest_path=manifest_path or root/MANIFEST_PATH
    report_path=report_path or root/REPORT_PATH
    manifest_raw=manifest_path.read_bytes(); report_raw=report_path.read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest()!=MANIFEST_FILE_SHA256:
        raise ValueError("pinned specialist manifest file SHA-256 mismatch")
    if hashlib.sha256(report_raw).hexdigest()!=REPORT_FILE_SHA256:
        raise ValueError("pinned specialist report file SHA-256 mismatch")
    manifest=json.loads(manifest_raw); saved_report=json.loads(report_raw)
    spec.verify_manifest(manifest)
    body=dict(saved_report); expected_report_hash=body.pop("reportSha256",None)
    if expected_report_hash!=_canonical_hash(body):
        raise ValueError("saved specialist report canonical hash mismatch")
    if saved_report.get("manifestSha256")!=manifest.get("manifestSha256"):
        raise ValueError("saved report does not identify pinned manifest")
    report_baselines={crop:saved_report.get("gate",{}).get("perCrop",{}).get(crop,{}).get("strongestBaseline")
                      for crop in spec.CROPS}
    if report_baselines!=FROZEN_BASELINE:
        raise ValueError(f"frozen diagnostic baselines differ from scored report gate: {report_baselines}")
    docs=spec._load_inputs(root)
    replay,audits=spec._compose(docs)
    if replay!=manifest["forecasts"] or audits["training"]!=manifest["trainingAudit"] or audits["testFallbacks"]!=manifest["testFallbackAudit"]:
        raise ValueError("complete answer-free B forecast replay differs from pinned manifest")
    # score_manifest independently repeats the replay gate; only then can it join targets.
    reproduced=spec.score_manifest(manifest,root=root)
    if reproduced!=saved_report:
        raise ValueError("pinned report is not reproduced from verified sources and manifest")
    actuals=_verified_actual_map(docs,replay)
    result=diagnose_rows(replay,actuals,audits["training"],reproduced.get("coverageByCell"))
    result.update({"manifestFileSha256":MANIFEST_FILE_SHA256,"reportFileSha256":REPORT_FILE_SHA256,
                   "manifestSha256":manifest["manifestSha256"],"reportSha256":saved_report["reportSha256"],
                   "verifiedBForecastRows":len(replay),"verifiedBActualRows":len(actuals)})
    result["diagnosticSha256"]=_canonical_hash(result)
    return result


def main(argv: list[str]|None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args(argv)
    destination=args.output.resolve()
    if destination.exists(): raise FileExistsError(f"output already exists: {destination}")
    result=run_diagnostic()
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("x",encoding="utf-8") as stream:
        stream.write(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return 0


if __name__=="__main__": raise SystemExit(main())
