"""Frozen small-sample pepper area feature pilot; read-only and deterministic."""
from __future__ import annotations

import argparse
from bisect import bisect_left
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import uuid

from . import dashboard, live_forecast, supply_pressure_prospective as prospective
from . import supply_pressure_reentry_screen as reentry
from .kma_residual_eval import _holm

START, END = dt.date(2023, 9, 1), dt.date(2023, 10, 31)
DIRECT_TRAIN_START = dt.date(2017, 1, 1)
TRAIN_YEARS = tuple(range(2018, 2023))
HORIZONS = (1, 5, 10)
ORIGINS = tuple(START + dt.timedelta(days=i) for i in range((END - START).days + 1))
AREA = {
    2017: (28329, "2017-08-30"), 2018: (28824, "2018-08-30"),
    2019: (31644, "2019-08-30"), 2020: (31146, "2020-08-28"),
    2021: (33373, "2021-08-30"), 2022: (29770, "2022-08-30"),
    2023: (27129, "2023-08-29"), 2024: (26436, "2024-08-27"),
}
AREA_URLS = {
    2017: "https://mods.go.kr/board.es?act=view&bid=229&list_no=362580&mid=a10301010000",
    2018: "https://mods.go.kr/board.es?act=view&bid=229&list_no=370378&mid=a10301010000",
    2019: "https://mods.go.kr/board.es?act=view&bid=229&list_no=377093&mid=a10301010000",
    2020: "https://mods.go.kr/board.es?act=view&bid=229&list_no=384692&mid=a10301010000",
    2021: "https://mods.go.kr/board.es?act=view&bid=229&list_no=391619&mid=a10301010000",
    2022: "https://mods.go.kr/board.es?act=view&bid=229&list_no=420146&mid=a10301010000",
    2023: "https://mods.go.kr/board.es?act=view&bid=229&list_no=426787&mid=a10301010000",
    2024: "https://www.mods.go.kr/board.es?act=view&bid=229&list_no=432376&mid=a10301010000",
}
MODELS = ("candidatePrice", "ablationPrice", "directPrice", "persistencePrice")
COMPARATORS = (("ablation", "ablationPrice"), ("direct", "directPrice"), ("persistence", "persistencePrice"))
LOSSES = ("AE", "APE")
BANDWIDTHS = (9, 28)
MIN_TRAIN = 90
MIN_PAIRS = 30
PINNED_PRICE_SHA256 = "eeb014d9c7487aae047eb1a0df153f18dc0d496a457a52c0125430401353926c"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def area_feature(year: int, origin: dt.date, area=AREA):
    """Return log current/prior preliminary area only after both posting dates."""
    if year not in area or year - 1 not in area:
        return None, None, "area_year_unavailable"
    current, current_post = area[year]
    previous, previous_post = area[year - 1]
    available = max(dt.date.fromisoformat(current_post), dt.date.fromisoformat(previous_post)) + dt.timedelta(days=1)
    if origin < available:
        return None, None, "area_not_yet_published"
    value = math.log(current / previous)
    return value, {"year": year, "currentHa": current, "priorHa": previous,
                   "currentPostDate": current_post, "priorPostDate": previous_post,
                   "currentSourceUrl": AREA_URLS[year], "priorSourceUrl": AREA_URLS[year-1],
                   "availableFrom": available.isoformat()}, None


def _series(product):
    result = {}
    for row in product.get("observations", []):
        day = dt.date.fromisoformat(row["date"])
        price, volume = float(row["actual"]), float(row["volumeKg"])
        if day in result or not math.isfinite(price) or price <= 0 or not math.isfinite(volume) or volume <= 0:
            raise ValueError(f"invalid or duplicate pepper observation: {day}")
        result[day] = (price, volume)
    return dict(sorted(result.items()))


def _feature(series, origin, target, horizon):
    base = reentry.issue_feature(series, origin, target, horizon)
    if base is None:
        return None
    vals, source, _branch = base
    area, provenance, reason = area_feature(origin.year, origin)
    if reason:
        return None
    return tuple(vals), source, area, provenance


def _train_rows(series, cutoff):
    days = tuple(sorted(series)); logs = tuple(math.log(series[d][0]) for d in days)
    pooled, direct = [], {h: [] for h in HORIZONS}
    for target, (price, _) in series.items():
        if target < DIRECT_TRAIN_START or target >= cutoff:
            continue
        for h in HORIZONS:
            origin = target - dt.timedelta(days=h)
            strict = live_forecast._strict_feature_row(series, origin, target, h, days, logs)
            if strict is not None:
                src, vals = strict
                y = math.log(price) - math.log(series[src][0])
                if math.isfinite(y): direct[h].append((target, vals, y))
            # Pooled area models use only the frozen 2018-2022 September/October
            # origin population; their targets may extend into November.
            if origin.year in TRAIN_YEARS and origin.month in (9, 10):
                candidate = _feature(series, origin, target, h)
                if candidate is not None:
                    vals, src, area_val, prov = candidate
                    y = math.log(price) - math.log(series[src][0])
                    if math.isfinite(y): pooled.append((target, vals, area_val, y, prov))
    return pooled, direct


def _fit(rows, n_features):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len({r[0] for r in rows}) < MIN_TRAIN:
        raise ValueError("fewer_than_90_distinct_training_target_days")
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    def vector(row):
        if len(row) == 5:  # pooled (target, 15 base values, area value, label, provenance)
            values=list(row[1]) + ([row[2]] if n_features == 16 else [])
            label=row[3]
        else:  # direct (target, full strict vector, label)
            values=list(row[1]); label=row[2]
        if n_features is not None and len(values) != n_features:
            raise ValueError(f"expected_{n_features}_features_got_{len(values)}")
        return values, label
    prepared=[vector(row) for row in rows]
    model.fit([x for x,_ in prepared], [y for _,y in prepared])
    return model


def _predict(model, feature, source_price):
    if model is None or feature is None: return None
    try: pred = source_price * math.exp(float(model.predict([feature])[0]))
    except (ValueError, OverflowError): return None
    return pred if math.isfinite(pred) and pred > 0 else None


def _metrics(rows, field):
    if not rows: return {"mae": None, "mape": None}
    return {"mae": sum(abs(r["actual"]-r[field]) for r in rows)/len(rows),
            "mape": 100*sum(abs(r["actual"]-r[field])/r["actual"] for r in rows)/len(rows)}


def _growth_gate(metrics, coverage):
    c, a, d = metrics["candidatePrice"], metrics["ablationPrice"], metrics["directPrice"]
    result={"allHorizonPairsAtLeast30":all(g["allComparatorsAtLeast30"] for g in coverage),
        "candidateMacroMapeAtMost10Percent":c["macroMape"] is not None and c["macroMape"]<=10,
        "candidateMapeAtLeast1ppBetterThanAblation":c["macroMape"] is not None and a["macroMape"] is not None and c["macroMape"]<=a["macroMape"]-1,
        "candidateMapeAtLeast1ppBetterThanDirect":c["macroMape"] is not None and d["macroMape"] is not None and c["macroMape"]<=d["macroMape"]-1,
        "candidateMaeNoMoreThan5pctWorse":all(c["macroMae"] is not None and metrics[f]["macroMae"] is not None and c["macroMae"]<=metrics[f]["macroMae"]*1.05 for f in ("ablationPrice","directPrice")),
        "candidateCoverageNoDecline":all(g["candidateCoverageNoDecline"] for g in coverage)}
    result["passed"]=all(result.values())
    return result


def _coverage_summary(rows, horizon):
    planned=[r for r in rows if r["horizon"]==horizon]
    observed=[r for r in planned if r["targetStatus"]=="present"]
    common_pairs=sum(r["pairedEvaluable"] for r in observed)
    pair_counts={c:common_pairs for c,_f in COMPARATORS}
    coverage_counts={model:sum(r["statuses"][model]=="available" for r in observed) for model in MODELS}
    return {"horizon":horizon,"planned":len(planned),"targetObserved":len(observed),
        "targetMissing":len(planned)-len(observed),"candidatePairsByComparator":pair_counts,
        "coveragePredictions":coverage_counts,"allComparatorsAtLeast30":all(n>=MIN_PAIRS for n in pair_counts.values()),
        "candidateCoverageNoDecline":all(coverage_counts["candidatePrice"]>=coverage_counts[m] for m in ("ablationPrice","directPrice"))}


def _dm(diff, bandwidth):
    n = len(diff)
    if n < MIN_PAIRS:
        return {"status":"insufficient_pairs", "n":n, "meanLossDifference":None, "statistic":None, "pValue":1.0}
    mean = sum(diff.values())/n
    c = {d:v-mean for d,v in diff.items()}
    variance = sum(v*v for v in c.values())
    for lag in range(1, bandwidth+1):
        cross = sum(c[d]*c.get(d+dt.timedelta(days=lag), 0.0) for d in c)
        variance += 2*(1-lag/(bandwidth+1))*cross
    variance /= n*n
    if not math.isfinite(variance) or variance <= 0:
        return {"status":"invalid_hac_variance", "n":n, "meanLossDifference":mean, "statistic":None, "pValue":1.0}
    stat=mean/math.sqrt(variance)
    return {"status":"tested", "n":n, "meanLossDifference":mean,"statistic":stat,
            "pValue":math.erfc(abs(stat)/math.sqrt(2))}


def _dm_family(rows, bandwidth):
    tests=[]
    for h in HORIZONS:
        for comp, field in COMPARATORS:
            pairs=[r for r in rows if r["horizon"]==h and r["pairedEvaluable"]]
            for loss in LOSSES:
                diff={dt.date.fromisoformat(r["target"]): (abs(r["actual"]-r["candidatePrice"])
                    -abs(r["actual"]-r[field])) if loss=="AE" else 100*((abs(r["actual"]-r["candidatePrice"])
                    -abs(r["actual"]-r[field]))/r["actual"]) for r in pairs}
                item=_dm(diff,bandwidth); item.update({"horizon":h,"comparator":comp,"loss":loss})
                item["conclusion"]="inference_unavailable" if item["status"]!="tested" else "inconclusive"
                tests.append(item)
    # Holm correction covers the frozen family of 18 tests for this bandwidth.
    _holm(tests)
    for t in tests:
        if t["status"]=="tested":
            if t.get("holmSignificant") and t["meanLossDifference"]<0: t["conclusion"]="improvement"
            elif t.get("holmSignificant") and t["meanLossDifference"]>0: t["conclusion"]="worsening"
    return tests


def _write_immutable(path, payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x",encoding="utf-8") as f:
            f.write(json.dumps(payload,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n")
            f.flush(); os.fsync(f.fileno())
        os.link(tmp,path)
    finally:
        try: tmp.unlink()
        except FileNotFoundError: pass


def build(prices_path: Path, output: Path):
    raw=prices_path.read_bytes(); prices=json.loads(raw)
    if _sha(raw) != PINNED_PRICE_SHA256: raise ValueError("prices_snapshot_hash_does_not_match_frozen_pin")
    if prices.get("schemaVersion")!=1: raise ValueError("unsupported_prices_schema")
    products={p.get("id"):p for p in prices.get("products",[]) if isinstance(p,dict)}
    if "pepper" not in products: raise ValueError("pepper_missing")
    series=_series(products["pepper"])
    rows=[]; audits=[]
    for origin in ORIGINS:
        train, direct_train=_train_rows(series,origin)
        cand=abl=direct={}
        reasons={}
        for name, subset, nf in (("candidate",train,16),("ablation",train,15)):
            try: model=_fit(subset,nf); reasons[name]=None
            except Exception as e: model=None; reasons[name]=str(e)
            if name=="candidate": cand=model
            else: abl=model
        dm={}
        for h, subset in direct_train.items():
            try: dm[h]=_fit(subset,None); reasons[f"direct{h}"]=None
            except Exception as e: dm[h]=None; reasons[f"direct{h}"]=str(e)
        audits.append({"origin":origin.isoformat(),"exclusiveTrainingTargetCutoff":origin.isoformat(),
            "candidateDistinctTargetDays":len({x[0] for x in train}),"ablationDistinctTargetDays":len({x[0] for x in train}),
            "directDistinctTargetDays":{str(h):len({x[0] for x in direct_train[h]}) for h in HORIZONS},"fitReasons":reasons})
        src_i=bisect_left(tuple(series),origin)-1; src=tuple(series)[src_i] if src_i>=0 else None
        persistence=series[src][0] if src and (origin-src).days<=dashboard.MAX_SOURCE_AGE_DAYS else None
        for h in HORIZONS:
            target=origin+dt.timedelta(days=h); actual=series.get(target,(None,None))[0]
            f=_feature(series,origin,target,h); areaf,prov,area_reason=(None,None,"feature_unavailable")
            if f is not None: areaf,prov=f[2],f[3]
            # All base features (15) plus area feature for candidate.
            source_price=series[f[1]][0] if f else None
            base=list(f[0]) if f else None
            cp=_predict(cand,base+[areaf] if base is not None and areaf is not None else None,source_price)
            ap=_predict(abl,base,source_price)
            days=tuple(series); logs=tuple(math.log(series[d][0]) for d in days)
            strict=live_forecast._strict_feature_row(series,origin,target,h,days,logs)
            dp=None
            if strict and dm[h] is not None:
                dsrc, df=strict; dp=_predict(dm[h],list(df),series[dsrc][0])
            present=actual is not None and math.isfinite(actual) and actual>0
            vals={"candidatePrice":cp,"ablationPrice":ap,"directPrice":dp,"persistencePrice":persistence}
            statuses={k:("available" if v is not None and math.isfinite(v) and v>0 else "unavailable") for k,v in vals.items()}
            paired=bool(present and all(statuses[k]=="available" for k in MODELS))
            rows.append({"origin":origin.isoformat(),"target":target.isoformat(),"horizon":h,"targetStatus":"present" if present else "missing_observation",
                "actual":actual if present else None,**vals,"statuses":statuses,"featureStatus":"available" if f else "unavailable",
                "featureFailureReason":None if f else area_reason,"areaFeature":areaf,"areaProvenance":prov,
                "modelFailureReasons":{"candidate":reasons["candidate"],"ablation":reasons["ablation"],
                    "direct":reasons[f"direct{h}"],"persistence":None if persistence is not None else "no_fresh_pre_origin_price"},
                "pairedEvaluable":paired,"pairFailureReasons":[] if paired else (["target_missing"] if not present else [])+[k for k in MODELS if statuses[k]!="available"]})
    # Metrics use each comparator's identical pair with candidate.
    paired_metrics={}
    for field in MODELS:
        byh={str(h):_metrics([r for r in rows if r["horizon"]==h and r["pairedEvaluable"]],field) for h in HORIZONS}
        paired_metrics[field]={"byHorizon":byh,"macroMape":sum(byh[str(h)]["mape"] for h in HORIZONS)/3 if all(byh[str(h)]["mape"] is not None for h in HORIZONS) else None,
            "macroMae":sum(byh[str(h)]["mae"] for h in HORIZONS)/3 if all(byh[str(h)]["mae"] is not None for h in HORIZONS) else None}
    gates=[]
    for h in HORIZONS:
        gates.append(_coverage_summary(rows,h))
    m=paired_metrics
    gate_result=_growth_gate(m,gates)
    collection_doc=Path(__file__).resolve().parents[1]/"docs"/"COLLECTION.md"
    plan_doc=collection_doc.with_name("PLAN.md")
    payload={"schemaVersion":1,"experiment":"pepper_preliminary_area_small_pilot","originWindow":{"start":START.isoformat(),"end":END.isoformat()},
        "originCount":len(ORIGINS),"horizons":HORIZONS,"trainingYears":TRAIN_YEARS,"minimumTrainingTargetDays":MIN_TRAIN,
        "minimumPairsPerHorizon":MIN_PAIRS,"areaInputs":{str(y):{"hectares":v,"postDate":d,"sourceUrl":AREA_URLS[y]} for y,(v,d) in AREA.items()},
        "featureNames":list(prospective.FEATURES)+["logMeanVolume7","missing28","logCurrentToPriorPreliminaryArea"],
        "pairMetrics":m,"coverageByHorizon":gates,"growthGate":gate_result,"dmTestsByHacCalendarDays":{str(b):_dm_family(rows,b) for b in BANDWIDTHS},
        "rows":rows,"fitAudits":audits,"input":{"pricesPath":str(prices_path.resolve()),"pricesSha256":_sha(raw),
            "frozenPriceSha256":PINNED_PRICE_SHA256,"collectionDocSha256":_sha(collection_doc.read_bytes()),"planDocSha256":_sha(plan_doc.read_bytes())},
        "codeSha256":_sha(Path(__file__).read_bytes()),"limitations":["small exploratory sample; area values are preliminary publication summaries","historical prices may be revised"]}
    _write_immutable(output,payload); return payload


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--prices",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(argv); build(a.prices,a.output); return 0

if __name__=="__main__": raise SystemExit(main())
