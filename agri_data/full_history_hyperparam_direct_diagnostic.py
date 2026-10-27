"""Read-only comparison of the saved tuned Ridge with fixed50 and its direct Ridge.

No model is fitted. The diagnostic pairs identical crop/target/horizon keys
using the pinned 2026 answer snapshot, and audits the direct component's fit and
feature cutoffs against the fixed-through-2025 tuned Ridge.
"""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math
from pathlib import Path
from typing import Any

from . import full_history_hyperparam_forecasts as hp
from . import full_history_return_eval as narrow

ROOT=Path(__file__).resolve().parents[1]
HP_MANIFEST=ROOT/"outputs/experiments/full-history-hyperparam-20261010/forecast-manifest.json"
FIXED50_MANIFEST=ROOT/"outputs/experiments/five-crop-specialists-20261009/forecast-manifest.json"
DIRECT_REPORT=ROOT/"outputs/experiments/direct-horizon-20261006/report.json"
PRICES=ROOT/"outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
PINNED={
 "hyperparamForecast":"9d08211d7ad3b207d9ae182360864ca06d98da17ad85e2b9bdffed3384eff0cc",
 "fixed50Manifest":"1497ee362945fcab45353a7b2403be752af3bf63c5f584cc0aedb2cc111ceaa8",
 "directHorizonReport":"a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67",
 "prices":"f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8",
 "hyperparamEvaluation":"a11822bb5e887906c6858718951e61408a417a6d1409d49d012b2337e1588e8c",
}
EVAL_START=dt.date(2026,1,1)

def _sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def _key(row:dict[str,Any]):return (row.get("product"),row.get("target"),row.get("horizon"))
def _positive(x):return narrow._positive(x)

def _metrics(rows:list[dict[str,Any]],field:str)->dict[str,Any]:
    if not rows:return {"n":0,"mape":None,"mae":None}
    errs=[abs(r[field]-r["actual"]) for r in rows]
    return {"n":len(rows),"mape":100*math.fsum(e/r["actual"] for e,r in zip(errs,rows))/len(rows),"mae":math.fsum(errs)/len(errs)}

def _by_horizon(rows:list[dict[str,Any]],fields:tuple[str,...])->dict[str,Any]:
    cells={}
    for h in range(1,11):
        subset=[r for r in rows if r["horizon"]==h]
        cells[str(h)]={field:_metrics(subset,field) for field in fields}
    macros={}
    for field in fields:
        valid=[cells[str(h)][field] for h in range(1,11) if cells[str(h)][field]["n"]]
        macros[field]={"horizonsIncluded":len(valid),"equalHorizonMape":math.fsum(x["mape"] for x in valid)/len(valid) if valid else None,
                       "equalHorizonMae":math.fsum(x["mae"] for x in valid)/len(valid) if valid else None}
    return {"horizons":cells,"equalHorizon":macros}

def diagnose(hp_manifest:dict[str,Any],fixed_manifest:dict[str,Any],direct_report:dict[str,Any],prices:dict[str,Any])->dict[str,Any]:
    hp._validate_manifest(hp_manifest)
    if hp_manifest.get("inputSha256",{}).get("fixed50Manifest")!=PINNED["fixed50Manifest"]:raise ValueError("hyperparameter manifest fixed50 source mismatch")
    if hp_manifest.get("inputSha256",{}).get("prices")!=PINNED["prices"]:raise ValueError("hyperparameter manifest price source mismatch")
    if fixed_manifest.get("phase")!="forecast_only" or fixed_manifest.get("targetValuesAttached") is not False:raise ValueError("fixed50 source must be answer-free")
    if fixed_manifest.get("protocol",{}).get("inputSha256",{}).get("direct")!=PINNED["directHorizonReport"]:raise ValueError("fixed50 direct-component source hash mismatch")
    if direct_report.get("protocol",{}).get("id")!="direct-horizon-ridge-2026-10-06":raise ValueError("unexpected direct Ridge source report")
    hp_rows={_key(r):r for r in hp_manifest["evaluationForecasts"]}
    fixed_rows={_key(r):r for r in fixed_manifest["forecasts"]}
    direct_rows={_key(r):r for r in direct_report["predictions"]}
    series=narrow._series(prices)
    actuals={(p,d):v for p,vals in series.items() for d,v in vals.items()}
    joined=[];direct_disagreements=[];planned={p:[] for p in hp.PRODUCTS}
    cutoff={p:{"directFitEndLaterThanTuned":0,"directSourceOnOriginDay":0,"directFitEndDaysLaterMax":None,"examples":[]} for p in hp.PRODUCTS}
    coverage={p:{"eligiblePlanned":0,"ridgeIssued":0,"directAvailable":0,"fixed50Available":0,"tripleCommonPositiveActual":0,"ridgeMissedDirectTargets":0,"ridgeMissedBlendTargets":0} for p in hp.PRODUCTS}
    for key,row in hp_rows.items():
        p,target,h=key;day=dt.date.fromisoformat(target);origin=dt.date.fromisoformat(row["origin"])
        if day<EVAL_START or not row["eligible"]:continue
        count=coverage[p];count["eligiblePlanned"]+=1
        model=(row.get("models") or {}).get("ridgeTuned",{});ridge=_positive(model.get("prediction")) if model.get("status")=="issued" else None
        base=fixed_rows.get(key,{}).get("predictions",{})
        direct=_positive(base.get("direct"));blend=_positive(base.get("fixed50"))
        count["ridgeIssued"]+=ridge is not None;count["directAvailable"]+=direct is not None;count["fixed50Available"]+=blend is not None
        count["ridgeMissedDirectTargets"]+=direct is not None and ridge is None
        count["ridgeMissedBlendTargets"]+=blend is not None and ridge is None
        src=direct_rows.get(key)
        if direct is not None and src and _positive(src.get("predicted")) is not None and not math.isclose(direct,src["predicted"],rel_tol=1e-12,abs_tol=1e-9):direct_disagreements.append({"key":key,"fixed50Direct":direct,"sourceDirect":src["predicted"]})
        actual=actuals.get((p,day))
        if ridge is None or direct is None or blend is None or actual is None:continue
        count["tripleCommonPositiveActual"]+=1
        joined.append({"product":p,"target":target,"origin":row["origin"],"horizon":h,"actual":actual,"ridgeTuned":ridge,"directRidge":direct,"fixed50":blend})
        fit_end=dt.date.fromisoformat(src["fitTrainTargetEnd"]) if src and src.get("fitTrainTargetEnd") else None
        source_date=dt.date.fromisoformat(src["sourceDate"]) if src and src.get("sourceDate") else None
        tuned_end=dt.date.fromisoformat(model.get("training",{}).get("lastTarget")) if model.get("training",{}).get("lastTarget") else None
        if fit_end and tuned_end and fit_end>tuned_end:
            audit=cutoff[p];audit["directFitEndLaterThanTuned"]+=1;delta=(fit_end-tuned_end).days
            audit["directFitEndDaysLaterMax"]=max(audit["directFitEndDaysLaterMax"] or 0,delta)
            if len(audit["examples"])<3:audit["examples"].append({"target":target,"origin":row["origin"],"directFitOrigin":src.get("fitOrigin"),"directFitTargetEnd":src.get("fitTrainTargetEnd"),"tunedTrainingEnd":tuned_end.isoformat()})
        if source_date and source_date>origin-dt.timedelta(days=1):cutoff[p]["directSourceOnOriginDay"]+=1
    per_crop={}
    for p in hp.PRODUCTS:
        same=[r for r in joined if r["product"]==p]
        per_crop[p]={"coverage":coverage[p],"sameTripleCommonPairs":_by_horizon(same,("ridgeTuned","directRidge","fixed50")),"directCutoffAudit":cutoff[p]}
    return {"schemaVersion":1,"diagnostic":"saved_2026_tuned_ridge_vs_fixed50_direct_component","noTraining":True,
      "sourceHashes":dict(PINNED),"sourceManifestContentHashes":{"hyperparam":hp_manifest.get("manifestSha256"),"fixed50":fixed_manifest.get("manifestSha256")},
      "eligibleDateRange":["2026-01-01","2026-09-27"],"fixed50CoverageDateRange":["2025-09-25","2026-09-24"],
      "directEqualsPinnedDirectHorizonSource":{"matchedPairs":len(fixed_rows.keys()&direct_rows.keys()),"disagreements":direct_disagreements[:20],"disagreementCount":len(direct_disagreements)},
      "commonPairDefinition":"positive target in pinned prices; hp ridgeTuned issued; five-crop direct and fixed50 positive on exact product/target/horizon key",
      "commonPairCount":len(joined),"perCrop":per_crop,
      "cutoffSummary":{"candidate":"full-history tuned Ridge fit once through 2025-12-31; feature source <= origin-1","direct":"direct-horizon Ridge refit monthly at first eligible origin; fit targets <= fitOrigin and price features <= origin","note":"Direct Ridge can use later 2026 training labels and same-origin-day source observations unavailable to the tuned Ridge; these are not information-matched model comparisons."}}

def run_paths()->dict[str,Any]:
    paths={"hyperparamForecast":HP_MANIFEST,"hyperparamEvaluation":ROOT/"outputs/experiments/full-history-hyperparam-20261010/evaluation.json","fixed50Manifest":FIXED50_MANIFEST,"directHorizonReport":DIRECT_REPORT,"prices":PRICES}
    for key,path in paths.items():
        if _sha(path)!=PINNED[key]:raise ValueError(f"pinned {key} hash mismatch")
    hp_manifest=json.loads(HP_MANIFEST.read_text());evaluation=json.loads(paths["hyperparamEvaluation"].read_text())
    if evaluation.get("manifestSha256")!=hp_manifest.get("manifestSha256"):raise ValueError("hyperparameter score references another manifest")
    return diagnose(hp_manifest,json.loads(FIXED50_MANIFEST.read_text()),json.loads(DIRECT_REPORT.read_text()),json.loads(PRICES.read_text()))

def main(argv=None)->int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(run_paths(),ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))
    return 0

if __name__=="__main__":raise SystemExit(main())
