"""Replay-gated scoring for the frozen equal-information online experiment."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math
from pathlib import Path
from typing import Any
from . import full_history_return_eval as price_helper
from . import equal_info_online_forecasts as forecasts

PRODUCTS=price_helper.PRODUCTS; HORIZONS=price_helper.HORIZONS
CONF_START,CONF_END=dt.date(2025,10,1),dt.date(2025,12,31)
EVAL_START,EVAL_END=dt.date(2026,1,1),dt.date(2026,9,27)
BANDWIDTHS=(9,28); ALPHA=.05
METHODS=("online71","causalDirect","causalSarima","causal50","online71Blend","persistence","fixed50")
COMPARISONS=(("online71Blend_vs_causal50","online71Blend","causal50"),("online71_vs_causalDirect","online71","causalDirect"),("online71Blend_vs_causalDirect","online71Blend","causalDirect"),("online71Blend_vs_causalSarima","online71Blend","causalSarima"))
LOSSES=("ae","ape")
def _canonical(v):return (json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def _positive(v):
 try:x=float(v)
 except (TypeError,ValueError,OverflowError):return None
 return x if math.isfinite(x) and x>0 else None
def _dm(diff:dict[dt.date,float],bandwidth:int)->dict[str,Any]:
 n=len(diff)
 if any(not math.isfinite(v) for v in diff.values()):return {"status":"nonfinite_loss_difference","reason":"nonfinite_loss_difference","n":n,"statistic":None,"meanLossDifference":None,"pValue":1.}
 mean=math.fsum(diff.values())/n if n else None
 if n<30:return {"status":"insufficient_pairs","reason":"fewer_than_30_pairs","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.}
 vals=list(diff.values());spread=max(vals)-min(vals);mag=max(abs(min(vals)),abs(max(vals)))
 if spread<=8*math.ulp(mag):return {"status":"zero_spread","reason":"constant_loss_difference_within_float_tolerance","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.}
 centered={d:v-mean for d,v in diff.items()};terms=[math.fsum(v*v for v in centered.values())]
 for lag in range(1,bandwidth+1):terms.append(2*(1-lag/(bandwidth+1))*math.fsum(v*centered.get(d+dt.timedelta(days=lag),0.) for d,v in centered.items()))
 total=math.fsum(terms);tol=8*math.ulp(math.fsum(abs(v) for v in terms)) if any(terms) else 0.
 if not math.isfinite(total) or total<=tol:return {"status":"invalid_hac_variance","reason":"nonpositive_or_cancellation_dominated_hac_variance","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.}
 stat=mean/math.sqrt(total/(n*n));return {"status":"tested","reason":None,"n":n,"statistic":stat,"meanLossDifference":mean,"pValue":math.erfc(abs(stat)/math.sqrt(2))}
def _holm(rows):
 m=len(rows);running=0.
 for rank,i in enumerate(sorted(range(m),key=lambda j:rows[j]["pValue"])):
  running=max(running,min(1.,(m-rank)*rows[i]["pValue"]));rows[i]["holmPValue"]=running;rows[i]["holmSignificant"]=running<ALPHA
def _metrics(pairs,a,b):
 if not pairs:return {"n":0,"aMae":None,"bMae":None,"aMape":None,"bMape":None}
 ea=[abs(r[a]-r["actual"]) for r in pairs];eb=[abs(r[b]-r["actual"]) for r in pairs]
 return {"n":len(pairs),"aMae":math.fsum(ea)/len(ea),"bMae":math.fsum(eb)/len(eb),"aMape":100*math.fsum(e/r["actual"] for e,r in zip(ea,pairs))/len(ea),"bMape":100*math.fsum(e/r["actual"] for e,r in zip(eb,pairs))/len(eb)}
def _validate_manifest(doc):
 if doc.get("phase")!="forecast_only" or doc.get("targetValuesAttached") is not False:raise ValueError("expected answer-free forecast-only manifest")
 unsigned=dict(doc);digest=unsigned.pop("manifestSha256",None)
 if digest!=hashlib.sha256(_canonical(unsigned)).hexdigest():raise ValueError("manifest digest mismatch")
 for key,start,end in (("confirmationForecasts",CONF_START,CONF_END),("evaluationForecasts",EVAL_START,EVAL_END)):
  rows=doc.get(key);expected={(p,start+dt.timedelta(days=i),h) for p in PRODUCTS for i in range((end-start).days+1) for h in HORIZONS};seen=set()
  if not isinstance(rows,list) or len(rows)!=len(expected):raise ValueError("forecast grid length mismatch")
  for r in rows:
   day=price_helper._date(r.get("target"));origin=price_helper._date(r.get("origin"));h=r.get("horizon");ident=(r.get("product"),day,h)
   if ident not in expected or ident in seen or origin!=day-dt.timedelta(days=h):raise ValueError("invalid forecast identity")
   if r.get("eligible") is not (origin>=start):raise ValueError("forecast eligibility mismatch")
   models=r.get("models")
   if not isinstance(models,dict) or set(models)!=set(METHODS):raise ValueError("method grid mismatch")
   for name,m in models.items():
    if not isinstance(m,dict) or m.get("status") not in ("issued","not_issued"):raise ValueError("invalid prediction status")
    pred=_positive(m.get("prediction"))
    if (m["status"]=="issued")!=(pred is not None):raise ValueError("status/prediction mismatch")
   seen.add(ident)
  if seen!=expected:raise ValueError("incomplete forecast grid")
def _score_period(rows,actuals,name,start,end):
 eligible=[r for r in rows if r["eligible"]]; metrics=[];dm=[];coverage={};temporal={};macro=[]
 for product in PRODUCTS:
  for h in HORIZONS:
   planned=[r for r in rows if r["product"]==product and r["horizon"]==h];cell=[r for r in eligible if r["product"]==product and r["horizon"]==h]
   for cname,a,b in COMPARISONS:
    pairs=[]
    for r in cell:
     day=price_helper._date(r["target"]);actual=actuals.get((product,day));pa=_positive(r["models"][a].get("prediction"));pb=_positive(r["models"][b].get("prediction"))
     if actual and pa and pb:pairs.append({"target":day.isoformat(),"actual":actual,a:pa,b:pb})
    met=_metrics(pairs,a,b);metrics.append({"product":product,"horizon":h,"comparison":cname,**met})
    for loss in LOSSES:
     diffs={price_helper._date(r["target"]):(abs(r[a]-r["actual"])-abs(r[b]-r["actual"]) if loss=="ae" else (abs(r[a]-r["actual"])-abs(r[b]-r["actual"]))/r["actual"]) for r in pairs}
     for bw in BANDWIDTHS:dm.append({"product":product,"horizon":h,"comparison":cname,"loss":loss,"bandwidth":bw,**_dm(diffs,bw)})
   coverage[f"{product}/{h}"]={"planned":len(planned),"eligiblePlanned":len(cell),"positiveActualTargets":sum((product,price_helper._date(r["target"])) in actuals for r in cell)}
   for method in METHODS:coverage[f"{product}/{h}"][method+"Issued"]=sum(_positive(r["models"][method].get("prediction")) is not None for r in cell)
   # Main contrast common-pair coverage and directional issue losses.
   for cname,a,b in COMPARISONS:
    coverage[f"{product}/{h}"][cname+"CommonPairs"]=next(m["n"] for m in metrics if m["product"]==product and m["horizon"]==h and m["comparison"]==cname)
    coverage[f"{product}/{h}"][cname+"CoverageLossTargets"]=[r["target"] for r in cell if _positive(r["models"][b].get("prediction")) and not _positive(r["models"][a].get("prediction"))]
 for product in PRODUCTS:
  for cname,a,b in COMPARISONS:
   cells=[r for r in metrics if r["product"]==product and r["comparison"]==cname and r["n"]]
   macro.append({"product":product,"comparison":cname,"horizonsIncluded":len(cells),"aEqualHorizonMae":math.fsum(r["aMae"] for r in cells)/len(cells) if cells else None,"bEqualHorizonMae":math.fsum(r["bMae"] for r in cells)/len(cells) if cells else None,"aEqualHorizonMape":math.fsum(r["aMape"] for r in cells)/len(cells) if cells else None,"bEqualHorizonMape":math.fsum(r["bMape"] for r in cells)/len(cells) if cells else None})
  for grain in ("month","quarter"):
   labels={}
   for i in range((end-start).days+1):
    d=start+dt.timedelta(days=i);label=f"{d.year}-{d.month:02d}" if grain=="month" else f"{d.year}-Q{(d.month-1)//3+1}";labels.setdefault(label,set()).add(d)
   for label,days in labels.items():
    rs=[r for r in rows if r["product"]==product and price_helper._date(r["target"]) in days];es=[r for r in rs if r["eligible"]]
    temporal_info={"planned":len(rs),"eligiblePlanned":len(es),"positiveActualTargets":sum((product,d) in actuals for d in days),**{m+"Issued":sum(_positive(r["models"][m].get("prediction")) is not None for r in es) for m in METHODS}}
    for cname,a,b in COMPARISONS:
     temporal_info[cname+"CommonPairs"]=sum(1 for r in es if (product,price_helper._date(r["target"])) in actuals and _positive(r["models"][a].get("prediction")) is not None and _positive(r["models"][b].get("prediction")) is not None)
    temporal[f"{product}/{grain}/{label}"]=temporal_info
 for bw in BANDWIDTHS:
  family=[r for r in dm if r["bandwidth"]==bw]
  if len(family)!=len(PRODUCTS)*len(HORIZONS)*len(COMPARISONS)*len(LOSSES):raise ValueError("unexpected Holm family size")
  _holm(family)
 return {"period":name,"metrics":metrics,"equalHorizonMetrics":macro,"coverage":coverage,"temporalCoverage":temporal,"dm":dm,"dmSummaryByBandwidth":[{"bandwidth":bw,"familySize":sum(r["bandwidth"]==bw for r in dm),"significantImprovement":sum(r["bandwidth"]==bw and r["status"]=="tested" and r["holmSignificant"] and r["meanLossDifference"]<0 for r in dm),"significantWorsening":sum(r["bandwidth"]==bw and r["status"]=="tested" and r["holmSignificant"] and r["meanLossDifference"]>0 for r in dm),"inferenceUnavailable":sum(r["bandwidth"]==bw and r["status"]!="tested" for r in dm)} for bw in BANDWIDTHS]}
def _gates(ev,conf):
 out=[]; em={(r["product"],r["comparison"]):r for r in ev["equalHorizonMetrics"]};cm={(r["product"],r["comparison"]):r for r in conf["equalHorizonMetrics"]};comp=COMPARISONS[0][0]
 for p in PRODUCTS:
  e=em.get((p,comp),{});c=cm.get((p,comp),{}); no_harm=True; joint=False;all_horizons=0;loss_cov=False
  for h in HORIZONS:
   m=next((r for r in ev["metrics"] if r["product"]==p and r["horizon"]==h and r["comparison"]==comp),None)
   if m and m["n"]:all_horizons+=1
   if m and m["n"] and m["aMape"]-m["bMape"]>1.:no_harm=False
   if ev["coverage"].get(f"{p}/{h}",{}).get(comp+"CoverageLossTargets"):loss_cov=True
   qs={(r["loss"],r["bandwidth"]):r for r in ev["dm"] if r["product"]==p and r["horizon"]==h and r["comparison"]==comp}
   if all(qs.get((loss,bw),{}).get("holmSignificant") and qs[(loss,bw)]["meanLossDifference"]<0 for loss in LOSSES for bw in BANDWIDTHS):joint=True
  conf_support=c.get("horizonsIncluded")==len(HORIZONS)
  conf_ok=conf_support and c.get("aEqualHorizonMape") is not None and c.get("bEqualHorizonMape") is not None and c["aEqualHorizonMape"]<c["bEqualHorizonMape"] and c.get("aEqualHorizonMae")<c.get("bEqualHorizonMae")
  improve=e.get("aEqualHorizonMape") is not None and e.get("bEqualHorizonMape") is not None and e["aEqualHorizonMape"]<=e["bEqualHorizonMape"]-.5 and e["aEqualHorizonMae"]<e["bEqualHorizonMae"]
  passes=conf_ok and improve and not loss_cov and no_harm and all_horizons==len(HORIZONS) and joint
  out.append({"product":p,"confirmationImprovedBoth":conf_ok,"confirmationAllHorizonsSupported":conf_support,"confirmationSupportReason":None if conf_support else "confirmation_macro_does_not_include_all_horizons","confirmationMapeCandidate":c.get("aEqualHorizonMape"),"confirmationMapeBaseline":c.get("bEqualHorizonMape"),"confirmationMaeCandidate":c.get("aEqualHorizonMae"),"confirmationMaeBaseline":c.get("bEqualHorizonMae"),"evaluationMapeCandidate":e.get("aEqualHorizonMape"),"evaluationMapeBaseline":e.get("bEqualHorizonMape"),"mapeImprovementPctPoints":(e["bEqualHorizonMape"]-e["aEqualHorizonMape"]) if e.get("aEqualHorizonMape") is not None and e.get("bEqualHorizonMape") is not None else None,"evaluationMaeCandidate":e.get("aEqualHorizonMae"),"evaluationMaeBaseline":e.get("bEqualHorizonMae"),"allHorizonsCommon":all_horizons==len(HORIZONS),"coverageNondeclining":not loss_cov,"noHorizonMapeWorseByMoreThan1pp":no_harm,"jointAeApeHolmAtSomeHorizonBothBandwidths":joint,"decision":"eligible_for_prospective_confirmation" if passes else "retain_existing_method"})
 return out
def score(manifest,prices,*,replay_verified=False):
 if replay_verified is not True:raise ValueError("full deterministic forecast replay must be verified before scoring")
 _validate_manifest(manifest);series=price_helper._series(prices);actuals={(p,d):v for p,rows in series.items() for d,v in rows.items() if CONF_START<=d<=EVAL_END}
 c=_score_period(manifest["confirmationForecasts"],actuals,"2025_confirmation",CONF_START,CONF_END);e=_score_period(manifest["evaluationForecasts"],actuals,"2026_evaluation",EVAL_START,EVAL_END)
 return {"schemaVersion":1,"phase":"scored","manifestSha256":manifest["manifestSha256"],"confirmation2025":c,"evaluation2026":e,"candidateGates":_gates(e,c),"interpretation":{"nonsignificantIsNotEquivalence":True,"2025ConfirmationIsRetrospective":True,"2026EvaluationIsRetrospective":True}}
def write_exclusive(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("xb") as f:f.write(_canonical(value))
def main(argv=None):
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="cmd",required=True);v=sub.add_parser("verify");v.add_argument("manifest",type=Path);s=sub.add_parser("score");s.add_argument("manifest",type=Path);s.add_argument("prices",type=Path);s.add_argument("output",type=Path);a=ap.parse_args(argv)
 if a.cmd=="verify":
  if not forecasts.verify_manifest(a.manifest):raise SystemExit("manifest replay mismatch")
  return 0
 if not forecasts.verify_manifest(a.manifest):raise SystemExit("manifest replay mismatch")
 if forecasts._sha(a.prices)!=forecasts.PRICES_SHA256:raise SystemExit("pinned price hash mismatch")
 write_exclusive(a.output,score(json.loads(a.manifest.read_text()),json.loads(a.prices.read_text()),replay_verified=True));return 0
if __name__=="__main__":raise SystemExit(main())
