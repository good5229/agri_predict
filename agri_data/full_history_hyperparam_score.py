"""Replay-gated scoring for the frozen time-ordered hyperparameter experiment."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math
from pathlib import Path
from typing import Any
from . import full_history_return_eval as price_helper
from . import full_history_hyperparam_forecasts as forecasts

PRODUCTS=price_helper.PRODUCTS
HORIZONS=price_helper.HORIZONS
CONFIRM_START,CONFIRM_END=dt.date(2025,1,1),dt.date(2025,12,31)
EVAL_START,EVAL_END=dt.date(2026,1,1),dt.date(2026,9,27)
ORIGIN_START_2025,ORIGIN_START_2026=CONFIRM_START,EVAL_START
BANDWIDTHS=(9,28); ALPHA=.05
ARMS=("ridgeTuned","ridgeDefault","xgbTuned","xgbDefault","rfTuned","rfDefault")
FAMILIES=(("ridge","ridgeTuned","ridgeDefault"),("xgb","xgbTuned","xgbDefault"),("rf","rfTuned","rfDefault"))
CONFIRM_COMPARISONS=tuple((f"{name}_tuned_vs_default",tuned,default) for name,tuned,default in FAMILIES)
EVAL_COMPARISONS=CONFIRM_COMPARISONS+tuple((f"{name}_tuned_vs_fixed50",tuned,"fixed50") for name,tuned,_ in FAMILIES)+tuple((f"{name}_default_vs_fixed50",default,"fixed50") for name,_,default in FAMILIES)
LOSSES=("ae","ape")

def _canonical(value:Any)->bytes:
 return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def _check_manifest(manifest:dict[str,Any])->None:
 if manifest.get("phase")!="forecast_only" or manifest.get("targetValuesAttached") is not False:raise ValueError("expected answer-free forecast-only manifest")
 unsigned=dict(manifest);digest=unsigned.pop("manifestSha256",None)
 if digest!=hashlib.sha256(_canonical(unsigned)).hexdigest():raise ValueError("manifest hash mismatch")
 def validate(rows,start,end,eval_year):
  expected={(p,start+dt.timedelta(days=i),h) for p in PRODUCTS for i in range((end-start).days+1) for h in HORIZONS};seen=set()
  for row in rows:
   key=(row.get("product"),price_helper._date(row.get("target")),row.get("horizon"))
   if key not in expected or key in seen:raise ValueError("invalid or duplicate forecast key")
   product,day,h=key;origin=day-dt.timedelta(days=h)
   if price_helper._date(row.get("origin"))!=origin:raise ValueError("forecast origin mismatch")
   eligible=origin >= (ORIGIN_START_2025 if eval_year==2025 else ORIGIN_START_2026)
   if row.get("eligible") is not eligible:raise ValueError("forecast eligibility mismatch")
   models=row.get("models")
   if not isinstance(models,dict):raise ValueError("forecast row lacks models mapping")
   for arm in ARMS:
    pred=models.get(arm)
    if not isinstance(pred,dict) or pred.get("status") not in ("issued","not_issued"):raise ValueError("invalid arm forecast status")
    positive=price_helper._positive(pred.get("prediction")) is not None
    if (pred["status"]=="issued") != positive:raise ValueError("arm status/prediction mismatch")
   if eval_year==2026:
    pred=row.get("fixed50")
    if not isinstance(pred,dict) or pred.get("status") not in ("issued","not_issued"):raise ValueError("invalid fixed50 status")
    if (pred["status"]=="issued") != (price_helper._positive(pred.get("prediction")) is not None):raise ValueError("fixed50 status/prediction mismatch")
   seen.add(key)
  if seen!=expected:raise ValueError("incomplete forecast calendar grid")
 validate(manifest.get("confirmationForecasts"),CONFIRM_START,CONFIRM_END,2025)
 validate(manifest.get("evaluationForecasts"),EVAL_START,EVAL_END,2026)
 if not isinstance(manifest.get("selection"),dict):raise ValueError("manifest missing frozen selections")

def _dm(diff:dict[dt.date,float],bandwidth:int)->dict[str,Any]:
 n=len(diff)
 if any(not math.isfinite(v) for v in diff.values()):return {"status":"nonfinite_loss_difference","reason":"nonfinite_loss_difference","n":n,"statistic":None,"meanLossDifference":None,"pValue":1.}
 mean=math.fsum(diff.values())/n if n else None
 if n<30:return {"status":"insufficient_pairs","reason":"fewer_than_30_pairs","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.}
 vals=list(diff.values());spread=max(vals)-min(vals);magnitude=max(abs(min(vals)),abs(max(vals)))
 if spread<=8*math.ulp(magnitude):return {"status":"zero_spread","reason":"constant_loss_difference_within_float_tolerance","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.}
 centered={d:v-mean for d,v in diff.items()};terms=[math.fsum(v*v for v in centered.values())]
 for lag in range(1,bandwidth+1):terms.append(2*(1-lag/(bandwidth+1))*math.fsum(centered[d]*centered.get(d+dt.timedelta(days=lag),0.) for d in centered))
 total=math.fsum(terms);tol=8*math.ulp(math.fsum(abs(v) for v in terms)) if any(terms) else 0.
 if not math.isfinite(total) or total<=tol:return {"status":"invalid_hac_variance","reason":"nonpositive_or_cancellation_dominated_hac_variance","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.}
 stat=mean/math.sqrt(total/(n*n));return {"status":"tested","reason":None,"n":n,"statistic":stat,"meanLossDifference":mean,"pValue":math.erfc(abs(stat)/math.sqrt(2))}
def _holm(rows:list[dict[str,Any]])->None:
 running=0.;m=len(rows)
 for rank,i in enumerate(sorted(range(m),key=lambda j:rows[j]["pValue"])):
  running=max(running,min(1.,(m-rank)*rows[i]["pValue"]));rows[i]["holmPValue"]=running;rows[i]["holmSignificant"]=running<ALPHA

def _metrics(pairs:list[dict[str,Any]],arm:str)->dict[str,Any]:
 if not pairs:return {"n":0,"mae":None,"mape":None}
 errors=[abs(r[arm]-r["actual"]) for r in pairs]
 return {"n":len(pairs),"mae":math.fsum(errors)/len(errors),"mape":100*math.fsum(e/r["actual"] for e,r in zip(errors,pairs))/len(errors)}
def _common(rows,actuals,arms):
 pairs=[]
 for row in rows:
  day=price_helper._date(row["target"]);actual=actuals.get((row["product"],day))
  vals={a:price_helper._positive(row["models"][a].get("prediction")) for a in ARMS}
  if "fixed50" in arms:vals["fixed50"]=price_helper._positive(row["fixed50"].get("prediction"))
  if actual is not None and all(vals.get(a) is not None for a in arms):pairs.append({"product":row["product"],"target":day.isoformat(),"horizon":row["horizon"],"actual":actual,**vals})
 return pairs
def _evaluate_period(rows,actuals,arms,comparisons,*,period_name,start,end,expected_dm_family):
 eligible=[r for r in rows if r["eligible"]];common=_common(eligible,actuals,arms);by_cell={(p,h):[r for r in common if r["product"]==p and r["horizon"]==h] for p in PRODUCTS for h in HORIZONS}
 metrics=[];dm=[];coverage={};temporal={};training_audit={}
 for product in PRODUCTS:
  for h in HORIZONS:
   cellrows=[r for r in eligible if r["product"]==product and r["horizon"]==h];pairs=by_cell[(product,h)]
   for comp,a,b in comparisons:
    metrics.append({"product":product,"horizon":h,"comparison":comp,"a":_metrics(pairs,a),"b":_metrics(pairs,b)})
    for loss in LOSSES:
     diffs={price_helper._date(r["target"]):(abs(r[a]-r["actual"])-abs(r[b]-r["actual"]) if loss=="ae" else (abs(r[a]-r["actual"])-abs(r[b]-r["actual"]))/r["actual"]) for r in pairs}
     for bw in BANDWIDTHS:dm.append({"product":product,"horizon":h,"comparison":comp,"loss":loss,"bandwidth":bw,**_dm(diffs,bw)})
   coverage[f"{product}/{h}"]={"planned":sum(r["product"]==product and r["horizon"]==h for r in rows),"eligiblePlanned":len(cellrows),"positiveActualPairs":sum((product,price_helper._date(r["target"])) in actuals for r in cellrows),"commonPairs":len(pairs)}
   representative=next((r for r in cellrows if r.get("models")),None)
   training_audit[f"{product}/{h}"]={arm:representative["models"][arm].get("training") if representative else None for arm in ARMS}
   for arm in arms:
    available=lambda r: price_helper._positive(r["fixed50"].get("prediction")) if arm=="fixed50" else price_helper._positive(r["models"][arm].get("prediction"))
    coverage[f"{product}/{h}"][arm+"Issued"]=sum(available(r) is not None for r in cellrows)
   for arm in arms:
    if arm=="fixed50":continue
    lost=[r["target"] for r in cellrows if price_helper._positive(r["fixed50"].get("prediction")) is not None and price_helper._positive(r["models"][arm].get("prediction")) is None] if any("fixed50" in r for r in cellrows) else []
    coverage[f"{product}/{h}"][arm+"CoverageLossTargets"]=lost
   for family,tuned,default in FAMILIES:
    if all(t in r.get("models",{}) and d in r.get("models",{}) for r in cellrows for t,d in ((tuned,default),)):
     coverage[f"{product}/{h}"][tuned+"Vs"+default+"CoverageLossTargets"]=[r["target"] for r in cellrows if price_helper._positive(r["models"][default].get("prediction")) is not None and price_helper._positive(r["models"][tuned].get("prediction")) is None]
 for product in PRODUCTS:
  for grain in ("month","quarter"):
   groups={}
   for i in range((end-start).days+1):
    day=start+dt.timedelta(days=i);label=f"{day.year}-{day.month:02d}" if grain=="month" else f"{day.year}-Q{(day.month-1)//3+1}"
    groups.setdefault(label,set()).add(day)
   for label,days in groups.items():
    planned=[r for r in rows if r["product"]==product and price_helper._date(r["target"]) in days]; elig=[r for r in planned if r["eligible"]];valid=[r for r in common if r["product"]==product and price_helper._date(r["target"]) in days]
    info={"planned":len(planned),"eligiblePlanned":len(elig),"positiveActualTargets":sum((product,d) in actuals for d in days),"commonPairs":len(valid)}
    for arm in arms:
     info[arm+"Issued"]=sum((price_helper._positive(r["fixed50"].get("prediction")) if arm=="fixed50" else price_helper._positive(r["models"][arm].get("prediction"))) is not None for r in elig)
    temporal[f"{product}/{grain}/{label}"]=info
 for bw in BANDWIDTHS:
  family=[r for r in dm if r["bandwidth"]==bw]
  if len(family)!=expected_dm_family:raise ValueError(f"{period_name} Holm family expected {expected_dm_family}, got {len(family)}")
  _holm(family)
 macro=[]
 for product in PRODUCTS:
  for comp,a,b in comparisons:
   cell=[r for r in metrics if r["product"]==product and r["comparison"]==comp and r["a"]["n"]]
   macro.append({"product":product,"comparison":comp,"horizonsIncluded":len(cell),"aEqualHorizonMae":math.fsum(r["a"]["mae"] for r in cell)/len(cell) if cell else None,"bEqualHorizonMae":math.fsum(r["b"]["mae"] for r in cell)/len(cell) if cell else None,"aEqualHorizonMape":math.fsum(r["a"]["mape"] for r in cell)/len(cell) if cell else None,"bEqualHorizonMape":math.fsum(r["b"]["mape"] for r in cell)/len(cell) if cell else None})
 return {"period":period_name,"metrics":metrics,"equalHorizonMetrics":macro,"coverage":coverage,"temporalCoverage":temporal,"trainingAudit":training_audit,"dm":dm,"dmSummaryByBandwidth":[{"bandwidth":bw,"familySize":sum(r["bandwidth"]==bw for r in dm),"significantImprovement":sum(r["bandwidth"]==bw and r["status"]=="tested" and r["holmSignificant"] and r["meanLossDifference"]<0 for r in dm),"significantWorsening":sum(r["bandwidth"]==bw and r["status"]=="tested" and r["holmSignificant"] and r["meanLossDifference"]>0 for r in dm),"inferenceUnavailable":sum(r["bandwidth"]==bw and r["status"]!="tested" for r in dm)} for bw in BANDWIDTHS],"commonPairs":common}
def _gate(eval_result,confirm_result):
 decisions=[];confirm_macro={(r["product"],r["comparison"]):r for r in confirm_result["equalHorizonMetrics"]}; eval_macro={(r["product"],r["comparison"]):r for r in eval_result["equalHorizonMetrics"]}
 for product in PRODUCTS:
  for family,tuned,default in FAMILIES:
   c=confirm_macro.get((product,f"{family}_tuned_vs_default"),{}); e=eval_macro.get((product,f"{family}_tuned_vs_fixed50"),{})
   cmape=c.get("aEqualHorizonMape"); cdef=c.get("bEqualHorizonMape"); cmae=c.get("aEqualHorizonMae"); cdefmae=c.get("bEqualHorizonMae")
   mape=e.get("aEqualHorizonMape"); fmape=e.get("bEqualHorizonMape"); mae=e.get("aEqualHorizonMae"); fmae=e.get("bEqualHorizonMae")
   no_harm=True; common_horizons=0
   for r in eval_result["metrics"]:
    if r["product"]==product and r["comparison"]==f"{family}_tuned_vs_fixed50" and r["a"]["n"]:
     common_horizons+=1
     if r["a"]["mape"]-r["b"]["mape"]>1.:no_harm=False
   coverage_loss=any(eval_result["coverage"].get(f"{product}/{h}",{}).get(tuned+"CoverageLossTargets",[]) for h in HORIZONS)
   joint=False
   for h in HORIZONS:
    q={(r["loss"],r["bandwidth"]):r for r in eval_result["dm"] if r["product"]==product and r["horizon"]==h and r["comparison"]==f"{family}_tuned_vs_fixed50"}
    if all(q.get((loss,bw),{}).get("holmSignificant") and q[(loss,bw)]["meanLossDifference"]<0 for loss in LOSSES for bw in BANDWIDTHS):joint=True
   conf_cells=[r for r in confirm_result["metrics"] if r["product"]==product and r["comparison"]==f"{family}_tuned_vs_default"]
   conf_all10=len(conf_cells)==len(HORIZONS) and all(r["a"]["n"]>=30 for r in conf_cells)
   conf_coverage_loss=any(confirm_result["coverage"].get(f"{product}/{h}",{}).get(tuned+"Vs"+default+"CoverageLossTargets",[]) for h in HORIZONS)
   confirmation=bool(conf_all10 and not conf_coverage_loss and cmape is not None and cdef is not None and cmae is not None and cdefmae is not None and cmape<cdef and cmae<cdefmae)
   passes=bool(confirmation and common_horizons==len(HORIZONS) and mape is not None and fmape is not None and mape<=fmape-.5 and mae<fmae and not coverage_loss and no_harm and joint)
   decisions.append({"product":product,"family":family,"confirmationImprovedBoth":confirmation,"confirmationAll10DistancesAtLeast30Pairs":conf_all10,"confirmationCoverageNondeclining":not conf_coverage_loss,"confirmationTunedMape":cmape,"confirmationDefaultMape":cdef,"confirmationTunedMae":cmae,"confirmationDefaultMae":cdefmae,"evaluationTunedMape":mape,"fixed50Mape":fmape,"mapeImprovementPctPoints":fmape-mape if fmape is not None and mape is not None else None,"evaluationTunedMae":mae,"fixed50Mae":fmae,"coverageNondeclining":not coverage_loss,"all10HorizonsHaveCommonPairs":common_horizons==len(HORIZONS),"noHorizonMapeWorseByMoreThan1PctPoint":no_harm,"jointAeApeHolmImprovementAtSomeHorizonBothHac":joint,"decision":"eligible_for_prospective_confirmation" if passes else "retain_existing_method"})
 return decisions
def score(manifest:dict[str,Any],prices:dict[str,Any],*,replay_verified:bool=False)->dict[str,Any]:
 if replay_verified is not True:raise ValueError("full deterministic forecast replay must be verified before scoring")
 _check_manifest(manifest);series=price_helper._series(prices);actuals={(p,d):v for p,rows in series.items() for d,v in rows.items() if CONFIRM_START<=d<=EVAL_END}
 confirm=_evaluate_period(manifest["confirmationForecasts"],actuals,ARMS,CONFIRM_COMPARISONS,period_name="2025_confirmation",start=CONFIRM_START,end=CONFIRM_END,expected_dm_family=len(PRODUCTS)*len(HORIZONS)*len(CONFIRM_COMPARISONS)*len(LOSSES))
 eval_arms=ARMS+("fixed50",)
 evaluation=_evaluate_period(manifest["evaluationForecasts"],actuals,eval_arms,EVAL_COMPARISONS,period_name="2026_evaluation",start=EVAL_START,end=EVAL_END,expected_dm_family=len(PRODUCTS)*len(HORIZONS)*len(EVAL_COMPARISONS)*len(LOSSES))
 return {"schemaVersion":1,"phase":"scored","manifestSha256":manifest["manifestSha256"],"confirmation2025":confirm,"evaluation2026":evaluation,"selection":manifest.get("selection"),"candidateGates":_gate(evaluation,confirm),"interpretation":{"nonsignificantIsNotEquivalence":True,"2025AlreadyViewedNotIndependentHoldout":True,"2026RetrospectiveRepeatedlyViewed":True}}
def write_exclusive(path:Path,value:dict[str,Any])->None:
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("xb") as f:f.write(_canonical(value))
def main(argv=None)->int:
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="cmd",required=True);v=sub.add_parser("verify");v.add_argument("manifest",type=Path);s=sub.add_parser("score");s.add_argument("manifest",type=Path);s.add_argument("prices",type=Path);s.add_argument("output",type=Path);a=ap.parse_args(argv)
 if a.cmd=="verify":
  if not forecasts.verify_manifest(a.manifest):raise SystemExit("manifest replay mismatch")
  return 0
 if not forecasts.verify_manifest(a.manifest):raise SystemExit("manifest replay mismatch")
 if forecasts._sha(a.prices)!=forecasts.PRICES_SHA256:raise SystemExit("pinned price hash mismatch")
 raw=json.loads(a.manifest.read_text());price_doc=json.loads(a.prices.read_text())
 write_exclusive(a.output,score(raw,price_doc,replay_verified=True))
 return 0
if __name__=="__main__":raise SystemExit(main())
