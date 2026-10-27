"""Frozen offline multi-horizon nonlinear model of changes from last price."""
from __future__ import annotations

import datetime as dt
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from . import direct_horizon, national_median_ensemble as inputs, national_price_selection as base

PROTOCOL_ID = "national-nonlinear-ratio-2026-10-09"
PRODUCTS, HORIZONS = base.PRODUCTS, base.HORIZONS
METHOD = "hist_gradient_boosting_log_price_ratio"
COMPARATORS = ("ridge_alpha_100", "calendar_sarima", "fixed_blend_50_50", "persistence")
WINDOWS = inputs.WINDOWS
BANDWIDTHS, MIN_PAIRS = (9, 28), 30
MIN_UNIQUE_TARGETS, MIN_TRAINING_ROWS = 120, 30
MODEL_SPEC = {"loss":"absolute_error", "max_iter":120, "max_leaf_nodes":15,
              "min_samples_leaf":25, "learning_rate":0.05, "l2_regularization":10.0,
              "random_state":42, "early_stopping":False}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE, DEFAULT_COVERAGE = inputs.DEFAULT_ARCHIVE, inputs.DEFAULT_COVERAGE


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _hash(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode())


def _positive(value: Any) -> float | None:
    try: x = float(value)
    except (TypeError, ValueError, OverflowError): return None
    return x if math.isfinite(x) and x > 0 else None


def _date(value: str) -> dt.date:
    try: return dt.date.fromisoformat(value)
    except (TypeError, ValueError) as exc: raise ValueError("invalid ISO date") from exc


def _feature(series: dict[dt.date,float], target: dt.date, horizon: int) -> list[float] | None:
    cutoff = target - dt.timedelta(days=horizon + 2)
    history = sorted((d,v) for d,v in series.items() if d <= cutoff and _positive(v) is not None)
    if not history or (cutoff-history[-1][0]).days > 14: return None
    last_day,last = history[-1]; loglast=math.log(last)
    logs=[(d,math.log(v)) for d,v in history if (cutoff-d).days < 56]
    out=[loglast]
    for w in (7,14,28,56):
        vals=[x for d,x in logs if (cutoff-d).days < w]
        out.extend((float(np.mean(vals))-loglast if vals else 0., float(np.std(vals)) if len(vals)>1 else 0., float(len(vals))))
    out.extend(((cutoff-last_day).days,float(horizon),math.sin(2*math.pi*target.weekday()/7),math.cos(2*math.pi*target.weekday()/7),
                math.sin(2*math.pi*target.timetuple().tm_yday/365.25),math.cos(2*math.pi*target.timetuple().tm_yday/365.25)))
    return out


def _last_asof(series: dict[dt.date,float], target: dt.date, horizon: int) -> tuple[dt.date,float] | None:
    cutoff=target-dt.timedelta(days=horizon+2)
    latest=max((d for d,v in series.items() if d<=cutoff and _positive(v) is not None),default=None)
    if latest is None or (cutoff-latest).days>14:return None
    return latest,float(series[latest])


def _fit_pooled(series: dict[dt.date,float], fit_cutoff: dt.date) -> tuple[Any, dict]:
    xs=[]; ys=[]; labels=[]; horizon_counts={str(h):0 for h in HORIZONS}; horizon_targets={str(h):[] for h in HORIZONS}; training_row_provenance=[]
    for target, actual in sorted(series.items()):
        if target > fit_cutoff: continue
        for horizon in HORIZONS:
            feat=_feature(series,target,horizon)
            last=_last_asof(series,target,horizon)
            if feat is not None and last is not None:
                last_day,last_price=last
                xs.append(feat); ys.append(math.log(actual/last_price)); labels.append(target)
                horizon_counts[str(horizon)]+=1; horizon_targets[str(horizon)].append(target)
                training_row_provenance.append({"target":target.isoformat(),"horizon":horizon,
                    "featureCutoff":(target-dt.timedelta(days=horizon+2)).isoformat(),
                    "lastObservedDate":last_day.isoformat(),"lastObservedPrice":last_price})
    unique=len(set(labels))
    deficient={h:n for h,n in horizon_counts.items() if n<MIN_TRAINING_ROWS}
    if unique < MIN_UNIQUE_TARGETS or deficient:
        raise ValueError(f"insufficient_training_data:unique_targets={unique};horizon_rows={horizon_counts}")
    model=HistGradientBoostingRegressor(**MODEL_SPEC)
    model.fit(np.asarray(xs),np.asarray(ys))
    return model,{"fitCutoff":fit_cutoff.isoformat(),"trainingRows":len(xs),"uniqueTrainingTargets":unique,
                  "maxTrainingTargetDate":max(labels).isoformat(),
                  "horizonTrainingRows":horizon_counts,
                  "horizonMaxTrainingTargetDate":{h:max(days).isoformat() for h,days in horizon_targets.items()},
                  "trainingRowProvenance":training_row_provenance}


def _forecast(series: dict[dt.date,float], target: dt.date, horizon: int, month_fit_cutoff: dt.date,
              state: dict) -> tuple[float|None,dict]:
    cutoff=target-dt.timedelta(days=horizon+2)
    asof=_last_asof(series,target,horizon)
    latest,last_price=asof if asof is not None else (None,None)
    fallback=last_price
    key=(target-dt.timedelta(days=horizon)).strftime("%Y-%m")
    try:
        if key not in state:
            try: state[key]={"fit":_fit_pooled(series,month_fit_cutoff),"failure":None}
            except Exception as exc: state[key]={"fit":None,"failure":f"{type(exc).__name__}:{exc}"}
        entry=state[key]
        if entry["failure"]: raise RuntimeError(entry["failure"])
        model,prov=entry["fit"]
        if _date(prov["fitCutoff"])>cutoff: raise ValueError("monthly_fit_cutoff_after_feature_cutoff")
        feat=_feature(series,target,horizon)
        if feat is None: raise ValueError("no_recent_asof_price_feature")
        logpred=float(model.predict(np.asarray([feat]))[0]); pred=last_price*math.exp(logpred)
        if not math.isfinite(pred) or pred<=0: raise ValueError("nonpositive_or_nonfinite_prediction")
        public_prov={k:v for k,v in prov.items() if k!="trainingRowProvenance"}
        return pred,{**public_prov,"fallback":False,"fallbackReason":None,"featureCutoff":cutoff.isoformat(),
                     "lastObservedDate":latest.isoformat(),"lastObservedPrice":last_price,"fallbackPriceDate":None}
    except Exception as exc:
        reason=f"{type(exc).__name__}:{exc}"
        entry=state.get(key,{})
        prov=(entry.get("fit") or (None,{}))[1]
        prov={k:v for k,v in prov.items() if k!="trainingRowProvenance"}
        if entry.get("failure"): reason=entry["failure"]
        if fallback is None: return None,{**prov,"fallback":False,"fallbackReason":reason,"featureCutoff":cutoff.isoformat(),"lastObservedDate":latest.isoformat() if latest else None,"lastObservedPrice":last_price,"fallbackPriceDate":latest.isoformat() if latest else None}
        return fallback,{**prov,"fallback":True,"fallbackReason":reason,"featureCutoff":cutoff.isoformat(),"lastObservedDate":latest.isoformat(),"lastObservedPrice":last_price,"fallbackPriceDate":latest.isoformat(),"fallbackValue":fallback}


def _source_prices(source: dict) -> dict[str,dict[dt.date,float]]:
    return base._prices(source)


def _build(prices: dict, source_digests: dict, forecast_rows: list[dict]) -> dict:
    expected={(p,(t-dt.timedelta(days=h)).isoformat(),t.isoformat(),h) for t in (dt.date(2026,1,1)+dt.timedelta(days=i) for i in range(273)) for p in PRODUCTS for h in HORIZONS}
    keyed={}
    for row in forecast_rows:
        key=inputs._forecast_identity(row)
        if key in keyed: raise ValueError("duplicate input forecast key")
        keyed[key]=row
    if set(keyed)!=expected: raise ValueError("pinned component forecasts do not cover exact Jan-Sep grid")
    states={}; forecasts=[]
    first_origin_by_month={}
    for product,origin_text,_,_ in expected:
        origin=_date(origin_text); month=origin.strftime("%Y-%m"); k=(product,month)
        first_origin_by_month[k]=min(first_origin_by_month.get(k,origin),origin)
    for product,origin_text,target_text,h in sorted(expected,key=lambda x:(x[2],x[0],x[3])):
        row=keyed[(product,origin_text,target_text,h)]; target=_date(target_text); origin=_date(origin_text)
        month=origin.strftime("%Y-%m"); fit_cutoff=first_origin_by_month[(product,month)]-dt.timedelta(days=2)
        value,lineage=_forecast(prices[product],target,h,fit_cutoff,states.setdefault((product,month),{}))
        preds={k:row.get("predictions",{}).get(k) for k in COMPARATORS}
        preds[METHOD]=value
        forecasts.append({"product":product,"origin":origin_text,"target":target_text,"horizon":h,"cutoff":(origin-dt.timedelta(days=2)).isoformat(),
                          "predictions":preds,"lineage":{"candidate":lineage,"componentSources":row.get("lineage",{})}})
    monthly_fit_provenance={}
    for (product,month),month_state in sorted(states.items()):
        entry=month_state.get(month)
        if entry and entry.get("fit"):
            model,prov=entry["fit"]
            monthly_fit_provenance[f"{product}/{month}"]={k:v for k,v in prov.items() if k!="trainingRowProvenance"}
            monthly_fit_provenance[f"{product}/{month}"]["trainingRowProvenance"]=prov["trainingRowProvenance"]
        elif entry:
            monthly_fit_provenance[f"{product}/{month}"]={"fitCutoff":(first_origin_by_month[(product,month)]-dt.timedelta(days=2)).isoformat(),"fitFailure":entry["failure"]}
    doc={"schemaVersion":1,"protocolId":PROTOCOL_ID,"phase":"forecast_only","targetValuesAttached":False,
         "sourceDigests":source_digests,"inputManifestSha256":dict(inputs.MANIFEST_SHA256),"inputManifestContentSha256":dict(inputs.MANIFEST_CONTENT_SHA256),
         "codeSha256":_sha(Path(__file__).read_bytes()),"modelSpec":MODEL_SPEC,"products":list(PRODUCTS),"horizons":list(HORIZONS),
         "window":["2026-01-01","2026-09-30"],"methods":[METHOD,*COMPARATORS],"monthlyFitProvenance":monthly_fit_provenance,"plannedForecastCount":len(forecasts),"forecasts":forecasts}
    doc["manifestSha256"]=_hash(doc); return doc


def build_manifest(development: dict, fresh: dict, raw_bundles: dict) -> dict:
    dev_source,fresh_source=inputs._load_sources(raw_bundles)
    for kind,doc,source in (("development",development,dev_source),("fresh",fresh,fresh_source)):
        if doc != inputs._load_frozen_manifest(kind): raise ValueError(f"{kind} forecast manifest differs from pinned file")
        if doc.get("sourceDigests") != source.get("sourceDigests"): raise ValueError(f"{kind} manifest/source digest mismatch")
    rows=inputs._merge_forecasts(development,fresh)
    return _build(_source_prices(fresh_source),{"development":dev_source["sourceDigests"],"fresh":fresh_source["sourceDigests"]},rows)


def verify_manifest(doc: dict) -> None:
    if doc.get("protocolId")!=PROTOCOL_ID or doc.get("phase")!="forecast_only" or doc.get("targetValuesAttached") is not False: raise ValueError("candidate manifest must be answer-free")
    if doc.get("codeSha256")!=_sha(Path(__file__).read_bytes()): raise ValueError("candidate code hash mismatch")
    body=dict(doc); saved=body.pop("manifestSha256",None)
    if saved!=_hash(body): raise ValueError("candidate manifest content hash mismatch")
    rows=doc.get("forecasts")
    if not isinstance(rows,list) or len(rows)!=10920 or doc.get("plannedForecastCount")!=10920: raise ValueError("planned count mismatch")
    seen=set()
    for row in rows:
        key=inputs._forecast_identity(row)
        if key in seen: raise ValueError("duplicate candidate key")
        seen.add(key)
        if "actual" in row or "targetValue" in row: raise ValueError("target leaked into forecast manifest")
        if row.get("lineage",{}).get("candidate",{}).get("featureCutoff") != row["cutoff"]: raise ValueError("feature cutoff mismatch")


def _verify_replay(doc: dict, development: dict, fresh: dict, raw_bundles: dict) -> None:
    verify_manifest(doc)
    rebuilt=build_manifest(development,fresh,raw_bundles)
    if doc.get("manifestSha256")!=rebuilt.get("manifestSha256") or doc.get("forecasts")!=rebuilt.get("forecasts") or doc.get("monthlyFitProvenance")!=rebuilt.get("monthlyFitProvenance"):
        raise ValueError("candidate replay mismatch")


def _dm(diffs: dict[dt.date,float], bandwidth: int) -> dict:
    n=len(diffs); mean=float(np.mean(list(diffs.values()))) if n else None
    def unavailable(status): return {"n":n,"status":status,"meanLossDifference":mean,"statistic":None,"pValue":1.,"calendarHacMaxLagDays":bandwidth}
    if n<MIN_PAIRS:return unavailable("inference_unavailable_insufficient_pairs")
    if all(value == next(iter(diffs.values())) for value in diffs.values()):
        return unavailable("inference_unavailable_nonpositive_variance")
    centered={d:v-mean for d,v in diffs.items()}; variance=sum(v*v for v in centered.values())
    for lag in range(1,bandwidth+1):
        cov=sum(v*centered.get(day-dt.timedelta(days=lag),0.) for day,v in centered.items())
        variance+=2*(1-lag/(bandwidth+1))*cov
    if not math.isfinite(variance) or variance<=0:return unavailable("inference_unavailable_nonpositive_variance")
    stat=mean/(math.sqrt(variance)/n)
    return {"n":n,"status":"available","meanLossDifference":mean,"statistic":stat,"pValue":math.erfc(abs(stat)/math.sqrt(2)),"calendarHacMaxLagDays":bandwidth}


def _holm(records: list[dict], bw: int) -> None:
    tests=[r["dm"][str(bw)] for r in records]; run=0.
    for rank,t in enumerate(sorted(tests,key=lambda x:x["pValue"])):
        run=max(run,min(1.,(len(tests)-rank)*t["pValue"])); t["holmAdjustedPValue"]=run
        t["holmSignificant"]=t["status"]=="available" and run<.05
        t["conclusion"]="inference_unavailable" if t["status"]!="available" else "improvement" if t["holmSignificant"] and t["meanLossDifference"]<0 else "worsening" if t["holmSignificant"] and t["meanLossDifference"]>0 else "inconclusive"


def score_manifest(doc: dict, development: dict, fresh: dict, raw_bundles: dict) -> dict:
    _verify_replay(doc,development,fresh,raw_bundles)
    source=base._load_source({"archivePath":raw_bundles["fresh"]["archivePath"],"coverageReportPath":raw_bundles["fresh"]["coverageReportPath"]})
    actuals=_source_prices(source); status_evidence=base._coverage_status_evidence(raw_bundles["fresh"])
    report={"protocolId":PROTOCOL_ID,"candidateManifestSha256":doc["manifestSha256"],"periods":{}}
    for start,end in WINDOWS:
        selected=[r for r in doc["forecasts"] if start<=_date(r["target"])<=end]; common=[]
        for r in selected:
            actual=actuals[r["product"]].get(_date(r["target"]))
            if actual and all(_positive(r["predictions"].get(m)) for m in (METHOD,*COMPARATORS)): common.append({**r,"actual":actual})
        cells={}
        for p in PRODUCTS:
            for h in HORIZONS:
                c=[r for r in common if r["product"]==p and r["horizon"]==h]
                cells[f"{p}/{h}"]={"n":len(c),"metrics":{m:direct_horizon._metrics([r["actual"] for r in c],[r["predictions"][m] for r in c]) for m in (METHOD,*COMPARATORS)}}
        complete=all(cells[f"{p}/{h}"]["n"]>=MIN_PAIRS for p in PRODUCTS for h in HORIZONS)
        macro={m:{loss:sum(cells[f"{p}/{h}"]["metrics"][m][loss] for p in PRODUCTS for h in HORIZONS)/40 if complete else None for loss in ("mae","mape")} for m in (METHOD,*COMPARATORS)}
        coverage={m:{"planned":len(selected),"issued":sum(_positive(r["predictions"].get(m)) is not None for r in selected),"issuedOnValidTargets":sum(_date(r["target"]) in actuals[r["product"]] and _positive(r["predictions"].get(m)) is not None for r in selected)} for m in (METHOD,*COMPARATORS)}
        coverage_by_cell={}
        for p in PRODUCTS:
            for h in HORIZONS:
                rows=[r for r in selected if r["product"]==p and r["horizon"]==h]
                valid=[r for r in rows if _date(r["target"]) in actuals[p]]
                coverage_by_cell[f"{p}/{h}"]={"planned":len(rows),"validTargetCount":len(valid),
                    "issued":{m:sum(_positive(r["predictions"].get(m)) is not None for r in rows) for m in (METHOD,*COMPARATORS)},
                    "issuedOnValidTargets":{m:sum(_date(r["target"]) in actuals[p] and _positive(r["predictions"].get(m)) is not None for r in rows) for m in (METHOD,*COMPARATORS)},
                    "commonScored":cells[f"{p}/{h}"]["n"]}
        per_crop={p:{m:direct_horizon._metrics([r["actual"] for r in common if r["product"]==p],[r["predictions"][m] for r in common if r["product"]==p]) for m in (METHOD,*COMPARATORS)} for p in PRODUCTS}
        tests=[]
        for p in PRODUCTS:
            for h in HORIZONS:
                cell=[r for r in common if r["product"]==p and r["horizon"]==h]
                for comp in COMPARATORS:
                    for loss in ("mae","mape"):
                        diffs={_date(r["target"]):(abs(r["predictions"][METHOD]-r["actual"])-abs(r["predictions"][comp]-r["actual"])) if loss=="mae" else 100*(abs(r["predictions"][METHOD]-r["actual"])-abs(r["predictions"][comp]-r["actual"]))/r["actual"] for r in cell}
                        tests.append({"product":p,"horizon":h,"comparator":comp,"loss":loss,"n":len(diffs),"dm":{str(bw):_dm(diffs,bw) for bw in BANDWIDTHS}})
        for bw in BANDWIDTHS:_holm(tests,bw)
        common_keys={inputs._forecast_identity(r) for r in common}
        exclusions=inputs._exclusions(selected,source,actuals,status_evidence,common_keys)
        paired=[]
        for p in PRODUCTS:
            for h in HORIZONS:
                c=[r for r in common if r["product"]==p and r["horizon"]==h]
                for comp in COMPARATORS:
                    paired.append({"product":p,"horizon":h,"comparator":comp,"n":len(c),
                        "meanPairedMaeDifference":float(np.mean([abs(r["predictions"][METHOD]-r["actual"])-abs(r["predictions"][comp]-r["actual"]) for r in c])) if c else None,
                        "meanPairedMapeDifferencePp":float(np.mean([100*(abs(r["predictions"][METHOD]-r["actual"])-abs(r["predictions"][comp]-r["actual"]))/r["actual"] for r in c])) if c else None})
        gate=None
        if start.month in (4,7):
            strongest=min((macro[m]["mape"],m) for m in COMPARATORS[:3] if macro[m]["mape"] is not None) if complete else (None,None)
            crop_mape={p:per_crop[p][METHOD]["mape"]-per_crop[p]["ridge_alpha_100"]["mape"] if per_crop[p][METHOD]["mape"] is not None and per_crop[p]["ridge_alpha_100"]["mape"] is not None else None for p in PRODUCTS}
            crop_mae={p:100*(per_crop[p][METHOD]["mae"]-per_crop[p]["ridge_alpha_100"]["mae"])/per_crop[p]["ridge_alpha_100"]["mae"] if per_crop[p][METHOD]["mae"] is not None and per_crop[p]["ridge_alpha_100"]["mae"] not in (None,0) else None for p in PRODUCTS}
            joint_improved={(t["product"],t["horizon"]) for t in tests if t["comparator"]=="ridge_alpha_100" and t["loss"]=="mae"}
            # Require both losses and both bandwidths to have Holm-significant gains in a shared cell.
            idx={(t["product"],t["horizon"],t["loss"]):t for t in tests if t["comparator"]=="ridge_alpha_100"}
            joint_count=sum(all(idx[(p,h,l)]["dm"][str(bw)]["conclusion"]=="improvement" for l in ("mae","mape") for bw in BANDWIDTHS) for p,h in joint_improved)
            gain=strongest[0]-macro[METHOD]["mape"] if complete else None
            criteria={"macroMapeGainAtLeast0_5pp":gain is not None and gain>=0.5,
                      "issuedCoverageNotDecreasedVsBestCandidate":strongest[1] is not None and coverage[METHOD]["issued"]>=coverage[strongest[1]]["issued"],
                      "eachCropMapeWorseningVsRidgeAtMost1pp":all(v is not None and v<=1 for v in crop_mape.values()),
                      "eachCropMaeWorseningVsRidgeAtMost5pct":all(v is not None and v<=5 for v in crop_mae.values()),
                      "atLeastOneCellJointImprovementVsRidge":joint_count>=1}
            gate={"status":"unsupported" if not complete else "pass" if all(criteria.values()) else "fail","criteria":criteria,
                  "macroMapeBestComparator":strongest[1],"macroMapeGainPp":gain,"productMapeChangePp":crop_mape,
                  "productMaeChangePct":crop_mae,"jointImprovedCells":joint_count,
                  "failureReasons":[] if complete and all(criteria.values()) else ["incomplete_40_cell_coverage"] if not complete else [k for k,v in criteria.items() if not v]}
        report["periods"][f"{start}_{end}"]={"planned":len(selected),"validCommonPairs":len(common),"exclusionsByState":exclusions,"complete40Cells":complete,"macro40EqualCell":macro,"cells":cells,"perCrop":per_crop,"issuedCoverage":coverage,"coverageByCell":coverage_by_cell,"pairedCellLosses":paired,"dmComparisons":tests,"holmFamilySizePerBandwidth":len(tests),"predeclaredGate":gate}
    gates=[r["predeclaredGate"] for k,r in report["periods"].items() if k.startswith(("2026-04","2026-07"))]
    report["twoWindowPredeclaredConclusion"]="pass" if len(gates)==2 and all(g["status"]=="pass" for g in gates) else "fail" if len(gates)==2 and any(g["status"]=="fail" for g in gates) else "unsupported"
    report["resultType"]="post_hoc_exploratory"
    report["independentConfirmation"]=False
    report["interpretation"]="Existing 2026 baseline results were already observed before this design; even a pass is exploratory, not independent confirmation or official contest score. Nonsignificance is not equivalence."
    report["reportSha256"]=_hash(report); return report


def _write_json_create_only(path: str | Path, doc: dict) -> None:
    destination=Path(path)
    destination.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(doc,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
    fd=os.open(destination,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    try:
        with os.fdopen(fd,"wb") as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
    except Exception:
        try: destination.unlink()
        except OSError: pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(description="Frozen offline national nonlinear price-ratio model")
    sub=parser.add_subparsers(dest="command",required=True)
    common=argparse.ArgumentParser(add_help=False)
    common.add_argument("--archive",default=str(DEFAULT_ARCHIVE))
    common.add_argument("--coverage-report",default=str(DEFAULT_COVERAGE))
    common.add_argument("--development-manifest",default=str(PROJECT_ROOT/inputs.MANIFEST_PATHS["development"]))
    common.add_argument("--fresh-manifest",default=str(PROJECT_ROOT/inputs.MANIFEST_PATHS["fresh"]))
    build=sub.add_parser("build",parents=[common],help="create answer-free forecast manifest")
    build.add_argument("--output",required=True)
    score=sub.add_parser("score",parents=[common],help="replay and score a saved forecast manifest")
    score.add_argument("--manifest",required=True);score.add_argument("--output",required=True)
    args=parser.parse_args(argv)
    development=inputs._load_frozen_manifest("development",args.development_manifest)
    fresh=inputs._load_frozen_manifest("fresh",args.fresh_manifest)
    bundles={"development":{"archivePath":args.archive},"fresh":{"archivePath":args.archive,"coverageReportPath":args.coverage_report}}
    if args.command=="build": result=build_manifest(development,fresh,bundles)
    else: result=score_manifest(json.loads(Path(args.manifest).read_text(encoding="utf-8")),development,fresh,bundles)
    _write_json_create_only(args.output,result)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
