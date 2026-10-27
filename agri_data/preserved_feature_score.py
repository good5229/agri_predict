"""Replay-gated scoring and confirmation policy for preserved-feature forecasts."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math
from pathlib import Path
from typing import Any
from . import full_history_return_eval as price_helper
from . import equal_info_online_forecasts as baseline_builder
from . import equal_info_online_score as stats

ROOT=Path(__file__).resolve().parents[1]
PRICES_PATH=baseline_builder.PRICES_PATH
BASELINE_PATH=ROOT/"outputs/experiments/equal-info-online-20261010/forecast-manifest.json"
PRICES_SHA256=baseline_builder.PRICES_SHA256
BASELINE_SHA256="d7d74823754c2a48d33cc4ddaeab674c686601fb927d399d84c5dbcd235184df"
PRODUCTS=tuple(price_helper.PRODUCTS); HORIZONS=tuple(price_helper.HORIZONS)
CONF_START,CONF_END=dt.date(2025,10,1),dt.date(2025,12,31)
EVAL_START,EVAL_END=dt.date(2026,1,1),dt.date(2026,9,27)
BANDWIDTHS=(9,28); ALPHA=.05
COMPARISONS=(("preserved106Blend_vs_causal50","preserved106Blend","causal50"),("preserved106Blend_vs_fixed50","preserved106Blend","fixed50"),("preserved106_vs_causalDirect","preserved106","causalDirect"))
MAIN_COMPARISONS=COMPARISONS[:2]
CANDIDATE="preserved106Blend"


def _canonical(v:Any)->bytes:return (json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode("utf-8")
def _unsigned_digest(v:dict[str,Any],field:str)->str:
 x=dict(v);x.pop(field,None)
 return hashlib.sha256(_canonical(x).rstrip(b"\n")).hexdigest()
def _file_sha(path:Path)->str:return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _positive(v:Any):return stats._positive(v)
def _series(prices):return price_helper._series(prices)
def _grid(start,end):return {(p,start+dt.timedelta(days=i),h) for p in PRODUCTS for i in range((end-start).days+1) for h in HORIZONS}
def _prediction(row,method):
 m=row.get("models",{}).get(method,{})
 if m.get("status")!="issued":return None
 return _positive(m.get("prediction"))
def _baseline_prediction(row,method):
 m=row.get("models",{}).get(method,{})
 if m.get("status")!="issued":return None
 return _positive(m.get("prediction"))

def _validate_phase_manifest(doc,phase):
 start,end=(CONF_START,CONF_END) if phase=="confirmation" else (EVAL_START,EVAL_END)
 if doc.get("phase")!="forecast_only" or doc.get("phaseName")!=phase or doc.get("targetValuesAttached") is not False:raise ValueError("expected answer-free phase manifest")
 if doc.get("protocol",{}).get("phase")!=phase:raise ValueError("phase protocol mismatch")
 unsigned=dict(doc);digest=unsigned.pop("manifestSha256",None)
 if digest!=_unsigned_digest(doc,"manifestSha256"):raise ValueError("forecast manifest digest mismatch")
 rows=doc.get("forecasts");expected=_grid(start,end)
 if not isinstance(rows,list) or len(rows)!=len(expected):raise ValueError("phase forecast grid length mismatch")
 seen=set()
 for r in rows:
  day=price_helper._date(r.get("target"));origin=price_helper._date(r.get("origin"));h=r.get("horizon");key=(r.get("product"),day,h)
  if key not in expected or key in seen or origin!=day-dt.timedelta(days=h):raise ValueError("invalid phase forecast identity")
  if r.get("eligible") is not (origin>=start):raise ValueError("phase eligibility mismatch")
  models=r.get("models")
  if not isinstance(models,dict):raise ValueError("missing phase model map")
  for method in ("preserved106","preserved106Blend","causal50","causalDirect","causalSarima","fixed50"):
   m=models.get(method)
   if not isinstance(m,dict):raise ValueError(f"missing model {method}")
   status=m.get("status")
   if method in ("preserved106",CANDIDATE) and phase=="evaluation" and r.get("selectedCrop") is False:
    if status not in ("not_selected","not_issued"):raise ValueError("unselected crop contains issued candidate")
   elif status not in ("issued","not_issued"):raise ValueError(f"invalid {method} status")
   if status=="issued" and _positive(m.get("prediction")) is None:raise ValueError(f"invalid issued {method} prediction")
   if status in ("not_issued","not_selected") and m.get("prediction") is not None:raise ValueError(f"unissued {method} has a prediction")
  seen.add(key)
 if seen!=expected:raise ValueError("phase forecast grid incomplete")
 if phase=="evaluation":
  selected=doc.get("selectedCrops")
  if not isinstance(selected,list) or any(p not in PRODUCTS for p in selected) or len(selected)!=len(set(selected)):raise ValueError("invalid selected crop policy")
  if doc.get("protocol",{}).get("evaluationSelection")!=selected:raise ValueError("selected-crop protocol mismatch")
  if any(r.get("selectedCrop") is not (r.get("product") in selected) for r in rows):raise ValueError("row selection flag mismatch")

def _baseline_index(doc,phase):
 key="confirmationForecasts" if phase=="confirmation" else "evaluationForecasts"
 rows=doc.get(key)
 if not isinstance(rows,list):raise ValueError("equal-info baseline lacks phase grid")
 return {(r.get("product"),price_helper._date(r.get("target")),r.get("horizon")):r for r in rows}
def _common_sets(rows,baseline,actuals,selected,phase):
 sets={name:[] for name,_,_ in COMPARISONS};eligible=[r for r in rows if r.get("eligible")]
 for r in eligible:
  p=r["product"];day=price_helper._date(r["target"]);h=r["horizon"];key=(p,day,h)
  if phase=="evaluation" and p not in selected:continue
  actual=actuals.get((p,day));base=baseline.get(key)
  if actual is None or base is None:continue
  blend=_prediction(r,CANDIDATE);base50=_baseline_prediction(base,"causal50");fixed=_baseline_prediction(base,"fixed50")
  # Both adoption contrasts use one strict triple-common population.
  if blend is None or base50 is None or fixed is None:continue
  values={CANDIDATE:blend,"causal50":base50,"fixed50":fixed}
  for name,a,b in MAIN_COMPARISONS:
   sets[name].append({"product":p,"target":day.isoformat(),"horizon":h,"actual":actual,a:values[a],b:values[b]})
  direct=_prediction(r,"preserved106");base_direct=_baseline_prediction(base,"causalDirect")
  if direct is not None and base_direct is not None:
   name,a,b=COMPARISONS[2]
   sets[name].append({"product":p,"target":day.isoformat(),"horizon":h,"actual":actual,a:direct,b:base_direct})
 return sets,eligible

def _pair_metrics(pairs,a,b):
 if not pairs:return {"n":0,"uniqueTargetCount":0,"firstTarget":None,"lastTarget":None,"maeA":None,"maeB":None,"mapeA":None,"mapeB":None}
 errorsA=[abs(r[a]-r["actual"]) for r in pairs];errorsB=[abs(r[b]-r["actual"]) for r in pairs];days=sorted({r["target"] for r in pairs})
 return {"n":len(pairs),"uniqueTargetCount":len(days),"firstTarget":days[0],"lastTarget":days[-1],"maeA":math.fsum(errorsA)/len(pairs),"maeB":math.fsum(errorsB)/len(pairs),"mapeA":100*math.fsum(e/r["actual"] for e,r in zip(errorsA,pairs))/len(pairs),"mapeB":100*math.fsum(e/r["actual"] for e,r in zip(errorsB,pairs))/len(pairs)}

def _coverage(rows,baseline,actuals,selected,phase,start,end):
 out={};temporal={};eligible=[r for r in rows if r.get("eligible")]
 for p in PRODUCTS:
  for h in HORIZONS:
   planned=[r for r in rows if r["product"]==p and r["horizon"]==h];es=[r for r in eligible if r["product"]==p and r["horizon"]==h]
   key=f"{p}/{h}";item={"plannedForecastRows":len(planned),"eligibleForecastRows":len(es),"positiveActualEligibleForecastPairs":sum((p,price_helper._date(r["target"])) in actuals for r in es)}
   for method in ("preserved106","preserved106Blend","causal50","causalDirect","causalSarima","fixed50"):
    item[method+"Issued"]=sum(_prediction(r,method) is not None for r in es)
   for base in ("causal50","fixed50"):
    shared=[r for r in es if (p,price_helper._date(r["target"]),h) in baseline and _baseline_prediction(baseline[(p,price_helper._date(r["target"]),h)],base) is not None]
    item[f"{CANDIDATE}Vs{base}CommonGrid"]=len(shared)
    item[f"{CANDIDATE}Vs{base}CoverageLossTargets"]=[r["target"] for r in shared if _prediction(r,CANDIDATE) is None]
   out[key]=item
  # period monthly/quarterly planned, eligible, issue, actual and paired coverage
  for grain in ("month","quarter"):
   groups={}
   for i in range((end-start).days+1):
    day=start+dt.timedelta(days=i);label=f"{day.year}-{day.month:02d}" if grain=="month" else f"{day.year}-Q{(day.month-1)//3+1}"
    groups.setdefault(label,set()).add(day)
   for label,days in groups.items():
    rs=[r for r in rows if r["product"]==p and price_helper._date(r["target"]) in days];es=[r for r in rs if r.get("eligible")]
    temporal[f"{p}/{grain}/{label}"]={"plannedForecastRows":len(rs),"eligibleForecastRows":len(es),"positiveActualUniqueTargets":sum((p,d) in actuals for d in days),**{m+"Issued":sum(_prediction(r,m) is not None for r in es) for m in ("preserved106","preserved106Blend","causal50","causalDirect","causalSarima","fixed50")}}
 return out,temporal

def _evaluate(rows,baseline,actuals,selected,phase,start,end):
 pairsets,eligible=_common_sets(rows,baseline,actuals,selected,phase);metrics=[];dm=[]
 for p in PRODUCTS:
  for h in HORIZONS:
   for name,a,b in COMPARISONS:
    cell=[r for r in pairsets[name] if r["product"]==p and r["horizon"]==h]
    metrics.append({"product":p,"horizon":h,"comparison":name,**_pair_metrics(cell,a,b)})
    for loss in ("ae","ape"):
     if phase=="evaluation" and p not in selected:
      why="crop_not_selected_by_confirmation_gate";test={"status":"not_selected","reason":why,"n":0,"statistic":None,"meanLossDifference":None,"pValue":1.}
     else:
      diffs={price_helper._date(r["target"]):(abs(r[a]-r["actual"])-abs(r[b]-r["actual"]) if loss=="ae" else (abs(r[a]-r["actual"])-abs(r[b]-r["actual"]))/r["actual"]) for r in cell}
      test=None
     for bw in BANDWIDTHS:
      dm.append({"product":p,"horizon":h,"comparison":name,"loss":loss,"bandwidth":bw,**(test if test is not None else stats._dm(diffs,bw))})
 macro=[]
 for p in PRODUCTS:
  for name,a,b in COMPARISONS:
   cells=[m for m in metrics if m["product"]==p and m["comparison"]==name and m["n"]]
   targets={r["target"] for r in pairsets[name] if r["product"]==p}
   macro.append({"product":p,"comparison":name,"horizonsIncluded":len(cells),"uniqueTargetsAcrossHorizons":len(targets),"aEqualHorizonMae":math.fsum(m["maeA"] for m in cells)/len(cells) if cells else None,"bEqualHorizonMae":math.fsum(m["maeB"] for m in cells)/len(cells) if cells else None,"aEqualHorizonMape":math.fsum(m["mapeA"] for m in cells)/len(cells) if cells else None,"bEqualHorizonMape":math.fsum(m["mapeB"] for m in cells)/len(cells) if cells else None})
 coverage,temporal=_coverage(rows,baseline,actuals,selected,phase,start,end)
 for p in PRODUCTS:
  for grain in ("month","quarter"):
   for key,info in temporal.items():
    if not key.startswith(f"{p}/{grain}/"):continue
    label=key.split("/")[-1]
    targets={r["target"] for name in pairsets for r in pairsets[name] if r["product"]==p and ((r["target"][:7]==label) if grain=="month" else (f"{r['target'][:4]}-Q{(int(r['target'][5:7])-1)//3+1}"==label))}
    info["commonUniqueTargetsAcrossComparisons"]=len(targets)
    for name,_,_ in COMPARISONS:
     info[name+"CommonPairs"]=sum(1 for r in pairsets[name] if r["product"]==p and ((r["target"][:7]==label) if grain=="month" else (f"{r['target'][:4]}-Q{(int(r['target'][5:7])-1)//3+1}"==label)))
 expected=len(PRODUCTS)*len(HORIZONS)*len(COMPARISONS)*2
 for bw in BANDWIDTHS:
  family=[r for r in dm if r["bandwidth"]==bw]
  if len(family)!=expected:raise ValueError(f"expected {expected} tests in each Holm family; got {len(family)}")
  stats._holm(family)
 return {"phase":phase,"metrics":metrics,"equalHorizonMetrics":macro,"coverage":coverage,"temporalCoverage":temporal,"dm":dm,"dmSummaryByBandwidth":[{"bandwidth":bw,"familySize":sum(r["bandwidth"]==bw for r in dm),"significantImprovement":sum(r["bandwidth"]==bw and r["status"]=="tested" and r.get("holmSignificant") and r.get("meanLossDifference",0)<0 for r in dm),"significantWorsening":sum(r["bandwidth"]==bw and r["status"]=="tested" and r.get("holmSignificant") and r.get("meanLossDifference",0)>0 for r in dm),"inferenceUnavailable":sum(r["bandwidth"]==bw and r["status"]!="tested" for r in dm),"notSelected":sum(r["bandwidth"]==bw and r["status"]=="not_selected" for r in dm)} for bw in BANDWIDTHS]}

def _policy_portfolio(rows,baseline,actuals,selected):
 """Equal-weight 50-cell portfolio: selected uses 106 blend, others inherit fixed50."""
 cells=[];by_cell={};selected=set(selected)
 for p in PRODUCTS:
  for h in HORIZONS:
   planned=[r for r in rows if r["product"]==p and r["horizon"]==h and r.get("eligible")]
   fixed_keys=[];pairs=[];policy_issued=0;candidate_issued=0;policy_fallback=[];policy_lost=[]
   for r in planned:
    day=price_helper._date(r["target"]);key=(p,day,h);base=baseline.get(key)
    fixed=_baseline_prediction(base,"fixed50") if base else None
    if fixed is None:continue
    fixed_keys.append(r["target"])
    candidate=_prediction(r,CANDIDATE) if p in selected else None
    if candidate is not None:candidate_issued+=1
    if p in selected and candidate is None:policy_fallback.append(r["target"])
    policy=candidate if candidate is not None else fixed
    if policy is None:
     policy_lost.append(r["target"]);continue
    policy_issued+=1
    actual=actuals.get((p,day))
    if actual is not None:pairs.append({"actual":actual,"policy":policy,"fixed50":fixed,"target":r["target"]})
   met=_pair_metrics(pairs,"policy","fixed50")
   cell={"product":p,"horizon":h,"selectedCropUses106Blend":p in selected,"plannedEligible":len(planned),"fixed50AvailableKeys":len(fixed_keys),"candidateIssuedOnFixed50Keys":candidate_issued,"policyIssuedOnFixed50Keys":policy_issued,"selectedCandidateFixed50FallbackTargets":policy_fallback,"policyCoverageLossTargets":policy_lost,"positiveActualCommonPairs":len(pairs),"uniqueTargets":met["uniqueTargetCount"],"firstTarget":met["firstTarget"],"lastTarget":met["lastTarget"],"policyMae":met["maeA"],"fixed50Mae":met["maeB"],"policyMape":met["mapeA"],"fixed50Mape":met["mapeB"]}
   cells.append(cell);by_cell[(p,h)]=cell
 crop=[]
 for p in PRODUCTS:
  cs=[by_cell[(p,h)] for h in HORIZONS if by_cell[(p,h)]["positiveActualCommonPairs"]]
  crop.append({"product":p,"selectedCropUses106Blend":p in selected,"horizonsIncluded":len(cs),"aEqualHorizonMae":math.fsum(c["policyMae"] for c in cs)/len(cs) if cs else None,"fixed50EqualHorizonMae":math.fsum(c["fixed50Mae"] for c in cs)/len(cs) if cs else None,"aEqualHorizonMape":math.fsum(c["policyMape"] for c in cs)/len(cs) if cs else None,"fixed50EqualHorizonMape":math.fsum(c["fixed50Mape"] for c in cs)/len(cs) if cs else None,"commonPairs":sum(c["positiveActualCommonPairs"] for c in cs),"fixed50AvailableKeys":sum(c["fixed50AvailableKeys"] for c in cs),"policyIssuedOnFixed50Keys":sum(c["policyIssuedOnFixed50Keys"] for c in cs),"coverageLossCount":sum(len(c["policyCoverageLossTargets"]) for c in cs)})
 portfolio_cells=[r for r in cells if r["positiveActualCommonPairs"]]
 temporal={}
 start,end=EVAL_START,EVAL_END
 for p in PRODUCTS:
  for grain in ("month","quarter"):
   labels={}
   for i in range((end-start).days+1):
    d=start+dt.timedelta(days=i);label=f"{d.year}-{d.month:02d}" if grain=="month" else f"{d.year}-Q{(d.month-1)//3+1}";labels.setdefault(label,set()).add(d)
   for label,days in labels.items():
    eligible=[r for r in rows if r["product"]==p and r.get("eligible") and price_helper._date(r["target"]) in days];fixed=[];issued=[];common=[];lost=[];candidate_count=0;fallback_count=0
    for r in eligible:
     day=price_helper._date(r["target"]);base=baseline.get((p,day,r["horizon"]));fv=_baseline_prediction(base,"fixed50") if base else None
     if fv is None:continue
     fixed.append(r)
     candidate=_prediction(r,CANDIDATE) if p in selected else None
     if candidate is not None:candidate_count+=1
     elif p in selected:fallback_count+=1
     pv=candidate if candidate is not None else fv
     if pv is None:lost.append(r);continue
     issued.append(r)
     if (p,day) in actuals:common.append(r)
    temporal[f"{p}/{grain}/{label}"]={"plannedEligible":len(eligible),"fixed50AvailableKeys":len(fixed),"candidateIssuedOnFixed50Keys":candidate_count,"selectedCandidateFixed50FallbackKeys":fallback_count,"policyIssuedOnFixed50Keys":len(issued),"positiveActualCommonPairs":len(common),"coverageLossCount":len(lost)}
 cells_possible=len(PRODUCTS)*len(HORIZONS);complete_cells=len(portfolio_cells)==cells_possible
 supported={"policyMae":math.fsum(c["policyMae"] for c in portfolio_cells)/len(portfolio_cells) if portfolio_cells else None,"fixed50Mae":math.fsum(c["fixed50Mae"] for c in portfolio_cells)/len(portfolio_cells) if portfolio_cells else None,"policyMape":math.fsum(c["policyMape"] for c in portfolio_cells)/len(portfolio_cells) if portfolio_cells else None,"fixed50Mape":math.fsum(c["fixed50Mape"] for c in portfolio_cells)/len(portfolio_cells) if portfolio_cells else None}
 summary={"cellsPossible":cells_possible,"cellsIncluded":len(portfolio_cells),"complete50CellCoverage":complete_cells,"policyMae":supported["policyMae"] if complete_cells else None,"fixed50Mae":supported["fixed50Mae"] if complete_cells else None,"policyMape":supported["policyMape"] if complete_cells else None,"fixed50Mape":supported["fixed50Mape"] if complete_cells else None,"supportedCellDescriptiveMeans":supported if not complete_cells else None,"commonPairs":sum(c["positiveActualCommonPairs"] for c in portfolio_cells),"fixed50AvailableKeys":sum(c["fixed50AvailableKeys"] for c in portfolio_cells),"policyIssuedOnFixed50Keys":sum(c["policyIssuedOnFixed50Keys"] for c in portfolio_cells),"coverageLossCount":sum(len(c["policyCoverageLossTargets"]) for c in portfolio_cells)}
 return {"policy":"selected crops=preserved106Blend with fixed50 fallback when the candidate is not issued; nonselected crops=the inherited fixed50 forecast; no refitting","comparison":"same original fixed50-available keys only; no missing dates imputed","cellMetrics":cells,"equalHorizonByCrop":crop,"temporalCoverage":temporal,"portfolioEqualWeightAcrossProductHorizonCells":summary}

def _crop_gate(report):
 decisions=[];metrics=report["metrics"];coverage=report["coverage"];macro={(r["product"],r["comparison"]):r for r in report["equalHorizonMetrics"]}
 for p in PRODUCTS:
  cells=[r for r in metrics if r["product"]==p and r["comparison"] in {MAIN_COMPARISONS[0][0],MAIN_COMPARISONS[1][0]}]
  each={name:[r for r in cells if r["comparison"]==name] for name,_,_ in MAIN_COMPARISONS}
  all10=all(len(each[name])==len(HORIZONS) and all(r["n"]>=30 for r in each[name]) for name,_,_ in MAIN_COMPARISONS)
  lower=all10 and all((macro.get((p,name),{}).get("horizonsIncluded")==len(HORIZONS) and macro[(p,name)]["aEqualHorizonMae"]<macro[(p,name)]["bEqualHorizonMae"] and macro[(p,name)]["aEqualHorizonMape"]<macro[(p,name)]["bEqualHorizonMape"]) for name,_,_ in MAIN_COMPARISONS)
  no_cov=True;no_harm=True
  for _,_,base in MAIN_COMPARISONS:
   for h in HORIZONS:
    c=coverage.get(f"{p}/{h}",{});no_cov &= not bool(c.get(f"{CANDIDATE}Vs{base}CoverageLossTargets"))
    m=next((r for r in metrics if r["product"]==p and r["horizon"]==h and r["comparison"]==f"{CANDIDATE}_vs_{base}"),None)
    if m and m["n"] and m["mapeA"]-m["mapeB"]>1.:no_harm=False
  eligible=all10 and lower and no_cov and no_harm
  reasons=[]
  if not all10:reasons.append("requires_at_least_30_common_pairs_at_all_10_horizons_against_both_baselines")
  if all10 and not lower:reasons.append("confirmation_not_strictly_better_on_equal_horizon_mae_and_mape_against_both_baselines")
  if not no_cov:reasons.append("candidate_coverage_loss_against_a_baseline")
  if not no_harm:reasons.append("horizon_mape_harm_over_1_percentage_point")
  decisions.append({"product":p,"eligible":eligible,"reasons":reasons,"all10HorizonsAtLeast30Pairs":all10,"strictlyLowerMaeAndMapeAgainstBoth":lower,"coverageNondecliningAgainstBoth":no_cov,"noHorizonOver1ppMapeHarmAgainstBoth":no_harm})
 return decisions

def _report_digest(report):return hashlib.sha256(_canonical(report)).hexdigest()
def _select(decisions):
 selected={d["product"] for d in decisions if d["eligible"]}
 return [p for p in PRODUCTS if p in selected]

def score_confirmation(manifest,baseline,prices,*,replay_verified=False):
 if replay_verified is not True:raise ValueError("complete forecast replay must be verified before scoring")
 _validate_phase_manifest(manifest,"confirmation");actuals={(p,d):v for p,rs in _series(prices).items() for d,v in rs.items() if CONF_START<=d<=CONF_END}
 base=_baseline_index(baseline,"confirmation");result=_evaluate(manifest["forecasts"],base,actuals,PRODUCTS,"confirmation",CONF_START,CONF_END);result["cropDecisions"]=_crop_gate(result)
 return _finalize_report(result,manifest,baseline,"confirmation")
def score_evaluation(manifest,baseline,prices,confirmation_manifest,confirmation_report,decision,*,replay_verified=False,confirmation_replay_verified=False,confirmation_file_sha=None):
 if replay_verified is not True or confirmation_replay_verified is not True:raise ValueError("both confirmation and evaluation forecast replays must be verified before scoring")
 _validate_phase_manifest(confirmation_manifest,"confirmation");_validate_phase_manifest(manifest,"evaluation")
 expected=_finalize_report(_confirmation_recomputed(confirmation_manifest,baseline,prices),confirmation_manifest,baseline,"confirmation")
 if _canonical(expected)!=_canonical(confirmation_report):raise ValueError("confirmation score does not match replayed forecast and pinned actuals")
 _validate_decision(decision,confirmation_manifest,confirmation_report,expected,confirmation_file_sha)
 selected=_select(expected["cropDecisions"])
 if not selected:raise ValueError("no crop passed confirmation; phase-2 fitting/scoring is prohibited")
 declared=manifest.get("selectedCrops")
 if declared!=selected:raise ValueError("evaluation manifest selected-crop policy mismatch")
 actuals={(p,d):v for p,rs in _series(prices).items() for d,v in rs.items() if EVAL_START<=d<=EVAL_END};base=_baseline_index(baseline,"evaluation")
 result=_evaluate(manifest["forecasts"],base,actuals,selected,"evaluation",EVAL_START,EVAL_END)
 result["selectedCrops"]=selected;result["notSelectedCrops"]=[p for p in PRODUCTS if p not in selected]
 result["candidateGates"]=_eval_gates(result,selected)
 result["portfolioPolicy"]=_policy_portfolio(manifest["forecasts"],_baseline_index(baseline,"evaluation"),{(p,d):v for p,rs in _series(prices).items() for d,v in rs.items() if EVAL_START<=d<=EVAL_END},selected)
 result["confirmationDecisionSha256"]=decision["decisionSha256"]
 return _finalize_report(result,manifest,baseline,"evaluation")
def _confirmation_recomputed(manifest,baseline,prices):
 actuals={(p,d):v for p,rs in _series(prices).items() for d,v in rs.items() if CONF_START<=d<=CONF_END};base=_baseline_index(baseline,"confirmation")
 result=_evaluate(manifest["forecasts"],base,actuals,PRODUCTS,"confirmation",CONF_START,CONF_END);result["cropDecisions"]=_crop_gate(result);return result
def _finalize_report(result,manifest,baseline,phase):
 result.update({"schemaVersion":1,"phase":"scored","scoredPhase":phase,"forecastManifestSha256":manifest.get("manifestSha256"),"forecastInputSha256":manifest.get("inputSha256"),"baselineManifestSha256":baseline.get("manifestSha256"),"baselineFileSha256":BASELINE_SHA256,"priceInputSha256":PRICES_SHA256,"scorerSha256":_file_sha(Path(__file__)),"fixed50Availability":"use exact inherited baseline keys only; absent dates, including 2026-09-25..27, are not imputed","nonsignificantIsNotEquivalence":True})
 result["reportSha256"]=_unsigned_digest(result,"reportSha256");return result
def _validate_decision(decision,confirmation_manifest,confirmation_report,expected_report,confirmation_file_sha):
 if decision.get("schemaVersion")!=1 or decision.get("kind")!="preserved_feature_confirmation_gate":raise ValueError("invalid confirmation decision document")
 if decision.get("decisionSha256")!=_unsigned_digest(decision,"decisionSha256"):raise ValueError("decision digest mismatch")
 want=_select(expected_report["cropDecisions"])
 if decision.get("selectedCrops")!=want:raise ValueError("decision crop list differs from replayed confirmation gate")
 if decision.get("confirmationForecastManifestSha256")!=confirmation_file_sha:raise ValueError("decision confirmation file hash mismatch")
 if decision.get("confirmationInternalManifestSha256")!=confirmation_manifest.get("manifestSha256"):raise ValueError("decision confirmation manifest binding mismatch")
 if decision.get("confirmationEvaluationSha256")!=expected_report.get("reportSha256"):raise ValueError("decision confirmation evaluation binding mismatch")
 if decision.get("scorerSha256")!=_file_sha(Path(__file__)):raise ValueError("decision scorer hash mismatch")
 if decision.get("cropDecisions")!=expected_report["cropDecisions"]:raise ValueError("decision rationale does not match replayed gate")
def create_decision(confirmation_manifest_path,confirmation_manifest,confirmation_report):
 decisions=confirmation_report["cropDecisions"];doc={"schemaVersion":1,"kind":"preserved_feature_confirmation_gate","confirmationForecastManifestSha256":_file_sha(Path(confirmation_manifest_path)),"confirmationInternalManifestSha256":confirmation_manifest["manifestSha256"],"confirmationEvaluationSha256":confirmation_report["reportSha256"],"scorerSha256":_file_sha(Path(__file__)),"selectedCrops":_select(decisions),"cropDecisions":decisions}
 doc["decisionSha256"]=_unsigned_digest(doc,"decisionSha256");return doc
def _eval_gates(report,selected):
 out=[]
 for p in selected:
  macros={r["comparison"]:r for r in report["equalHorizonMetrics"] if r["product"]==p};checks=[]
  for name,_,_ in MAIN_COMPARISONS:
   m=macros.get(name,{});checks.append(m.get("horizonsIncluded")==len(HORIZONS) and m.get("aEqualHorizonMape") is not None and m["bEqualHorizonMape"]-m["aEqualHorizonMape"]>=.5 and m["aEqualHorizonMae"]<m["bEqualHorizonMae"])
  no_cov=all(not report["coverage"].get(f"{p}/{h}",{}).get(f"{CANDIDATE}Vs{base}CoverageLossTargets") for _,_,base in MAIN_COMPARISONS for h in HORIZONS)
  no_harm=True
  for _,_,base in MAIN_COMPARISONS:
   comp=f"{CANDIDATE}_vs_{base}"
   for h in HORIZONS:
    m=next((r for r in report["metrics"] if r["product"]==p and r["horizon"]==h and r["comparison"]==comp),None)
    if not m or not m["n"]:no_harm=False
    elif m["mapeA"]-m["mapeB"]>1:no_harm=False
  joint=False
  for h in HORIZONS:
   tests={(r["comparison"],r["loss"],r["bandwidth"]):r for r in report["dm"] if r["product"]==p and r["horizon"]==h}
   same_horizon=all((lambda r:r is not None and r.get("status")=="tested" and r.get("holmSignificant") and r.get("meanLossDifference",0)<0)(tests.get((f"{CANDIDATE}_vs_{base}",loss,bw))) for _,_,base in MAIN_COMPARISONS for loss in ("ae","ape") for bw in BANDWIDTHS)
   joint=joint or same_horizon
  passed=all(checks) and no_cov and no_harm and joint
  out.append({"product":p,"mapeMaeGainAtLeast0_5ppAndStrictMaeAgainstBoth":all(checks),"coverageNondecliningAgainstBoth":no_cov,"noHorizonOver1ppMapeHarmAgainstBoth":no_harm,"sameHorizonJointAeApeHolmImprovementAgainstBothAtHac9And28":joint,"decision":"continue_candidate" if passed else "retain_fixed50_policy"})
 return out

def score_confirmation_paths(forecast_path,baseline_path,prices_path):
 from . import preserved_feature_forecasts as gen
 if not gen.verify_manifest(forecast_path,"confirmation"):raise ValueError("confirmation forecast replay mismatch")
 if _file_sha(Path(baseline_path))!=BASELINE_SHA256 or _file_sha(Path(prices_path))!=PRICES_SHA256:raise ValueError("pinned baseline or price input hash mismatch")
 return score_confirmation(json.loads(Path(forecast_path).read_text()),json.loads(Path(baseline_path).read_text()),json.loads(Path(prices_path).read_text()),replay_verified=True)
def validate_confirmation_gate_paths(forecast_path,report_path,decision_path,baseline_path=BASELINE_PATH,prices_path=PRICES_PATH):
 """Replay and recompute confirmation before the generator starts phase-2 fitting."""
 report=score_confirmation_paths(forecast_path,baseline_path,prices_path)
 saved_report=json.loads(Path(report_path).read_text());manifest=json.loads(Path(forecast_path).read_text())
 if _canonical(saved_report)!=_canonical(report):raise ValueError("saved confirmation report differs from replayed score")
 decision=json.loads(Path(decision_path).read_text());expected=create_decision(forecast_path,manifest,report)
 if _canonical(decision)!=_canonical(expected):raise ValueError("confirmation decision differs from replayed score and source hashes")
 return _select(report["cropDecisions"]),{"confirmationForecastManifest":decision["confirmationForecastManifestSha256"],"confirmationInternalManifest":decision["confirmationInternalManifestSha256"],"confirmationEvaluation":decision["confirmationEvaluationSha256"],"confirmationDecision":decision["decisionSha256"],"confirmationScorer":decision["scorerSha256"]}
def score_evaluation_paths(forecast_path,baseline_path,prices_path,confirmation_path,report_path,decision_path):
 from . import preserved_feature_forecasts as gen
 if _file_sha(Path(baseline_path))!=BASELINE_SHA256 or _file_sha(Path(prices_path))!=PRICES_SHA256:raise ValueError("pinned baseline or price input hash mismatch")
 if not gen.verify_manifest(forecast_path,"evaluation",decision_path=Path(decision_path),confirmation_path=Path(confirmation_path),confirmation_report_path=Path(report_path)):raise ValueError("evaluation forecast replay mismatch")
 return score_evaluation(json.loads(Path(forecast_path).read_text()),json.loads(Path(baseline_path).read_text()),json.loads(Path(prices_path).read_text()),json.loads(Path(confirmation_path).read_text()),json.loads(Path(report_path).read_text()),json.loads(Path(decision_path).read_text()),replay_verified=True,confirmation_replay_verified=True,confirmation_file_sha=_file_sha(Path(confirmation_path)))
def write_exclusive(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("xb") as f:f.write(_canonical(value))
def main(argv=None):
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest="cmd",required=True)
 c=sub.add_parser("score-confirmation");c.add_argument("forecast",type=Path);c.add_argument("report",type=Path);c.add_argument("decision",type=Path);c.add_argument("--baseline",type=Path,default=BASELINE_PATH);c.add_argument("--prices",type=Path,default=PRICES_PATH)
 e=sub.add_parser("score-evaluation");e.add_argument("forecast",type=Path);e.add_argument("confirmation",type=Path);e.add_argument("confirmation_report",type=Path);e.add_argument("decision",type=Path);e.add_argument("report",type=Path);e.add_argument("--baseline",type=Path,default=BASELINE_PATH);e.add_argument("--prices",type=Path,default=PRICES_PATH)
 a=ap.parse_args(argv)
 if a.cmd=="score-confirmation":
  result=score_confirmation_paths(a.forecast,a.baseline,a.prices);decision=create_decision(a.forecast,json.loads(a.forecast.read_text()),result);write_exclusive(a.report,result);write_exclusive(a.decision,decision);return 0
 result=score_evaluation_paths(a.forecast,a.baseline,a.prices,a.confirmation,a.confirmation_report,a.decision);write_exclusive(a.report,result);return 0
if __name__=="__main__":raise SystemExit(main())
