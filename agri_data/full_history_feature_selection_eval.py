"""Frozen training-only feature selection for full-history Ridge forecasts."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math, random
from pathlib import Path
from typing import Any
from . import full_history_return_eval as narrow
from . import full_history_feature_library as library
ROOT=Path(__file__).resolve().parents[1]
PRICES_PATH=ROOT/"outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
NARROW_PATH=ROOT/"outputs/experiments/full-history-return-20261009/forecast-manifest.json"
RICH_PATH=ROOT/"outputs/experiments/full-history-xgb-rf-20261009/forecast-manifest.json"
FIXED50_PATH=ROOT/"outputs/experiments/five-crop-specialists-20261009/forecast-manifest.json"
PRICES_SHA256="f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
NARROW_SHA256="86da40f5bcfb6669c5f0a4b34e9ad33c9c69d7f7492b85d58dd43bd90b119d45"
RICH_SHA256="fde1dc24c11d0ba6cf2a7b74714cc79050837789c5f283e1be25c5fd81a5c14e"
FIXED50_SHA256="1497ee362945fcab45353a7b2403be752af3bf63c5f584cc0aedb2cc111ceaa8"
FEATURE_LIBRARY_SHA256="58bcde98d11239494ae13cc6169fb4c7f84a49a15ad11cf5b3cf5c49b4f39389"
RICH_HELPER_SHA256="1951b34bc019ccd9fe680776f4277105e1edb470d3d7d5a37802c781088d9245"
NARROW_HELPER_SHA256="276f389e7153e07ef02836183d55e9823a6eb7c9bf24acc882d663946c55c5c5"
PRODUCTS=narrow.PRODUCTS; HORIZONS=narrow.HORIZONS
TRAIN_START=dt.date(2014,1,1); RANK_START=dt.date(2023,1,1); RANK_END=dt.date(2024,12,31); N_START=dt.date(2025,1,1); N_END=dt.date(2025,12,31); EVAL_START=narrow.EVAL_START; EVAL_END=narrow.EVAL_END; ORIGIN_START=dt.date(2026,1,1)
MIN_TRAIN=narrow.MIN_TRAIN; BANDWIDTHS=narrow.BANDWIDTHS; ALPHA=narrow.ALPHA
CANDIDATE_N=(8,16,24,32); COMPARISONS=(("selected_vs_fixed50","selected","fixed50"),("selected_vs_all","selected","allFeature"),("selected_vs_rich","selected","richRidge"),("all_vs_rich","allFeature","richRidge")); LOSSES=("ae","ape")
def _sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def _check_sources():
 for path,expected,label in ((Path(library.__file__),FEATURE_LIBRARY_SHA256,"feature library"),(Path(__import__("agri_data.full_history_xgb_rf_eval",fromlist=["__file__"]).__file__),RICH_HELPER_SHA256,"rich feature helper"),(Path(narrow.__file__),NARROW_HELPER_SHA256,"narrow Ridge helper")):
  if _sha(path)!=expected:raise ValueError(f"pinned {label} source hash mismatch")
def _canonical(v): return (json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()
def _ridge(X,y):
 from sklearn.pipeline import make_pipeline
 from sklearn.preprocessing import StandardScaler
 from sklearn.linear_model import Ridge
 return make_pipeline(StandardScaler(),Ridge(alpha=10.,fit_intercept=True)).fit(X,y)
def _raw_maps(raw):
 if raw.get("schemaVersion")!=1 or not isinstance(raw.get("products"),list): raise ValueError("prices must use dashboard schemaVersion 1")
 maps={}; audit={}
 for p in raw["products"]:
  pid=p.get("id") if isinstance(p,dict) else None
  if pid not in PRODUCTS or pid in maps or not isinstance(p.get("observations"),list): raise ValueError("unknown, duplicate, or malformed price product")
  rows={}; a={"rows":0,"invalidDate":0,"invalidOrNonpositivePrice":0,"invalidOrNonpositiveVolume":0,"invalidOrNonpositiveRowCount":0,"invalidMarketCodes":0}
  for r in p["observations"]:
   a["rows"]+=1
   if not isinstance(r,dict): a["invalidDate"]+=1;a["invalidOrNonpositivePrice"]+=1;continue
   d=narrow._date(r.get("date"));v=narrow._positive(r.get("actual"));vol=narrow._positive(r.get("volumeKg"));cnt=narrow._positive(r.get("rowCount")); markets=r.get("marketCodes")
   if d is None:a["invalidDate"]+=1
   if v is None:a["invalidOrNonpositivePrice"]+=1
   if vol is None:a["invalidOrNonpositiveVolume"]+=1
   if cnt is None:a["invalidOrNonpositiveRowCount"]+=1
   if not isinstance(markets,list) or any(not isinstance(x,str) for x in markets):a["invalidMarketCodes"]+=1;markets=[]
   if d is None or v is None:continue
   if d in rows:raise ValueError(f"duplicate usable price date {pid}/{d}")
   rows[d]={"price":v,"volume":vol,"rowCount":cnt,"markets":frozenset(markets)}
  maps[pid]=rows;audit[pid]=a
 if set(maps)!=set(PRODUCTS):raise ValueError("prices must contain all five products")
 return maps,audit
def _feature(records,target,h,ordered=None,prices=None): return library.features(records,target,h,ordered=ordered,prices=prices)
def _grid(): return [(p,EVAL_START+dt.timedelta(days=i),h) for p in PRODUCTS for i in range((EVAL_END-EVAL_START).days+1) for h in HORIZONS]
def _forecast_rows(raw):
 if raw.get("phase")!="forecast_only" or raw.get("targetValuesAttached") is not False or not isinstance(raw.get("forecasts"),list):raise ValueError("comparator manifest must be answer-free")
 out={}
 for r in raw["forecasts"]:
  k=(r.get("product"),narrow._date(r.get("target")),r.get("horizon"))
  if k[0] not in PRODUCTS or k[1] is None or k[2] not in HORIZONS or k in out: raise ValueError("invalid/duplicate comparator key")
  out[k]=r
 return out
def _mape(yhat,y): return 100*math.fsum(abs(math.exp(float(a))-math.exp(b))/math.exp(b) for a,b in zip(yhat,y))/len(y) if y else None
def _mae(yhat,y,p0s=None): return math.fsum((p0 if p0s is not None else 1.)*abs(math.exp(float(a))-math.exp(b)) for a,b,p0 in zip(yhat,y,p0s or [1.]*len(y)))/len(y) if y else None
def _importance(model, rows, ys):
 base=list(model.predict([r for r in rows])); base_score=_mape(base,ys); out=[]
 for j in range(len(rows[0])):
  col=[r[j] for r in rows]; random.Random(0).shuffle(col); altered=[r.copy() for r in rows]
  for i,v in enumerate(col):altered[i][j]=v
  out.append(_mape(list(model.predict(altered)),ys)-base_score)
 return base_score,out
def build(prices, rich_manifest, fixed50, *, price_sha=None, rich_sha=None, fixed50_sha=None):
 _check_sources()
 records,audit=_raw_maps(prices); rich=_forecast_rows(rich_manifest); fixed=narrow._fixed50(fixed50)
 for k,r in rich.items():
  if k in fixed and narrow._positive(r.get("fixed50"))!=fixed[k]:raise ValueError("rich/fixed50 manifests disagree")
 names=list(library.FEATURE_NAMES); F=len(names)
 if len(set(names))!=F or F<=32:raise ValueError("feature library names invalid")
 forecasts=[]; selections={}
 for product in PRODUCTS:
  rec=records[product]; days=sorted(rec); prices_map={d:r["price"] for d,r in rec.items()}; datasets={}
  for h in HORIZONS:
   data=[]
   for day in days:
    if day>dt.date(2025,12,31):continue
    feat,p0,fa=_feature(rec,day,h,days,prices_map)
    if feat is not None and p0 is not None and len(feat)==F and all(math.isfinite(float(x)) for x in feat): data.append({"day":day,"x":feat,"y":math.log(rec[day]["price"]/p0),"p0":p0,"audit":fa})
   datasets[h]=data
  per_h={}; eligible=[]
  for h,data in datasets.items():
   train=[r for r in data if TRAIN_START<=r["day"]<=dt.date(2022,12,31)]
   val=[r for r in data if RANK_START<=r["day"]<=RANK_END and r["day"]-dt.timedelta(days=h)>=RANK_START]
   imp=[None]*F;base=None;reason=None
   if len(train)<MIN_TRAIN:reason="fewer_than_90_2014_2022_training_targets"
   elif len(val)<30:reason="fewer_than_30_2023_2024_validation_pairs"
   else:
    m=_ridge([r["x"] for r in train],[r["y"] for r in train]);base,imp=_importance(m,[r["x"] for r in val],[r["y"] for r in val]);eligible.append(h)
   per_h[str(h)]={"trainingCount":len(train),"trainingFirst":train[0]["day"].isoformat() if train else None,"trainingLast":train[-1]["day"].isoformat() if train else None,"validationCount":len(val),"validationOriginEligibleCount":len(val),"baselineMape":base,"importance":imp,"rankEligible":reason is None,"reason":reason}
  means=[math.fsum(per_h[str(h)]["importance"][j] for h in eligible)/len(eligible) for j in range(F)] if eligible else []
  ranking=sorted(range(F),key=lambda j:(-means[j],names[j])) if means else []
  nreports={}; chosen=None
  for n in CANDIDATE_N:
   if n>F:continue
   byh=[]
   if not ranking:
    nreports[str(n)]={"horizons":[],"horizonsIncluded":0,"equalHorizonMae":None,"equalHorizonMape":None,"validationPairs":0,"reason":"no_rank_eligible_horizons"}
    continue
   for h,data in datasets.items():
    train=[r for r in data if TRAIN_START<=r["day"]<=dt.date(2024,12,31)]
    val=[r for r in data if N_START<=r["day"]<=N_END and r["day"]-dt.timedelta(days=h)>=N_START]
    if len(train)<MIN_TRAIN or not val:continue
    ix=ranking[:n];m=_ridge([[r["x"][j] for j in ix] for r in train],[r["y"] for r in train]);pred=list(m.predict([[r["x"][j] for j in ix] for r in val]));ys=[r["y"] for r in val]
    byh.append({"horizon":h,"trainingCount":len(train),"trainingFirst":train[0]["day"].isoformat() if train else None,"trainingLast":train[-1]["day"].isoformat() if train else None,"n":len(val),"mae":_mae(pred,ys,[r["p0"] for r in val]),"mape":_mape(pred,ys),"firstTarget":val[0]["day"].isoformat(),"lastTarget":val[-1]["day"].isoformat()})
   nreports[str(n)]={"horizons":byh,"horizonsIncluded":len(byh),"equalHorizonMae":math.fsum(x["mae"] for x in byh)/len(byh) if byh else None,"equalHorizonMape":math.fsum(x["mape"] for x in byh)/len(byh) if byh else None,"validationPairs":sum(x["n"] for x in byh)}
  if eligible and any(v["equalHorizonMape"] is not None for v in nreports.values()):
   chosen=min((int(n) for n,v in nreports.items() if v["equalHorizonMape"] is not None),key=lambda n:(nreports[str(n)]["equalHorizonMape"],nreports[str(n)]["equalHorizonMae"],n))
  selections[product]={"featureNames":names,"horizonRanking":per_h,"eligibleHorizons":eligible,"meanPermutationMapeIncrease":means,"rankingIndices":ranking,"rankingNames":[names[i] for i in ranking],"nValidation":nreports,"selectedN":chosen,"selectionReason":"lowest_2025_equal_horizon_mape_then_mae_then_smaller_n" if chosen else "no_eligible_ranking_or_2025_validation"}
  ix=ranking[:chosen] if chosen else []
  for h,data in datasets.items():
   final=[r for r in data if TRAIN_START<=r["day"]<=dt.date(2025,12,31)]
   selected_model=all_model=None; fit_errors={}
   if chosen and len(final)>=MIN_TRAIN:
    try:selected_model=_ridge([[r["x"][j] for j in ix] for r in final],[r["y"] for r in final])
    except Exception as e:fit_errors["selected"]=type(e).__name__
   if len(final)>=MIN_TRAIN:
    try:all_model=_ridge([r["x"] for r in final],[r["y"] for r in final])
    except Exception as e:fit_errors["allFeature"]=type(e).__name__
   for _,day,hh in (k for k in _grid() if k[0]==product and k[2]==h):
    origin=day-dt.timedelta(days=h); xforecast,p0forecast,faudit=_feature(rec,day,h,days,prices_map); source={"x":xforecast,"p0":p0forecast,"audit":faudit} if xforecast is not None and p0forecast is not None and len(xforecast)==F and all(math.isfinite(float(x)) for x in xforecast) else None; reason=None
    if origin<ORIGIN_START:reason="origin_before_2026"
    elif source is None:reason=faudit.get("reason","missing_or_stale_pre_origin_price")
    row={"product":product,"target":day.isoformat(),"origin":origin.isoformat(),"horizon":h,"eligible":origin>=ORIGIN_START,"selectedStatus":"not_issued","selectedReason":reason,"selected":None,"allFeatureStatus":"not_issued","allFeatureReason":reason,"allFeature":None,"richRidge":narrow._positive((rich.get((product,day,h)) or {}).get("richRidge")),"fixed50":fixed.get((product,day,h)),"selectedFeatureIndices":ix,"selectedN":chosen,"trainingCount":len(final),"trainingFirst":final[0]["day"].isoformat() if final else None,"trainingLast":final[-1]["day"].isoformat() if final else None,"featureAudit":{k:v for k,v in source["audit"].items() if k!="featureNames"} if source else faudit}
    for arm,model,cols in (("selected",selected_model,ix),("allFeature",all_model,list(range(F)))):
     if reason is None and model is None:row[arm+"Reason"]="selection_unavailable" if arm=="selected" and not chosen else fit_errors.get(arm,"fewer_than_90_final_training_targets")
     elif reason is None:
      try:
       lp=float(model.predict([[source["x"][j] for j in cols]])[0]);v=source["p0"]*math.exp(lp)
       if not math.isfinite(v) or v<=0:raise ValueError("nonpositive_or_nonfinite_prediction")
       row[arm]=v;row[arm+"Status"]="issued";row[arm+"Reason"]=None
      except Exception as e:row[arm+"Reason"]=str(e) if isinstance(e,ValueError) else "prediction_failed:"+type(e).__name__
    forecasts.append(row)
 doc={"schemaVersion":1,"phase":"forecast_only","targetValuesAttached":False,"protocol":{"id":"full-history-training-only-feature-selection-v1","products":list(PRODUCTS),"horizons":list(HORIZONS),"trainTargets":[TRAIN_START.isoformat(),"2025-12-31"],"rankTargets":[RANK_START.isoformat(),RANK_END.isoformat()],"nSelectionTargets":[N_START.isoformat(),N_END.isoformat()],"evaluationTargets":[EVAL_START.isoformat(),EVAL_END.isoformat()],"minimumTrainingTargets":MIN_TRAIN,"featureNames":names,"models":{"selected":"StandardScaler+Ridge(alpha=10)","allFeature":"StandardScaler+Ridge(alpha=10)"},"target":"log(actual/p0)","featureCutoff":"origin-1","nCandidates":list(CANDIDATE_N),"comparisons":[x[0] for x in COMPARISONS]},"inputSha256":{"prices":price_sha,"richRidgeManifest":rich_sha,"fixed50Manifest":fixed50_sha,"featureLibrary":FEATURE_LIBRARY_SHA256,"richFeatureHelper":RICH_HELPER_SHA256,"narrowRidgeHelper":NARROW_HELPER_SHA256},"sourceAudit":audit,"selection":selections,"plannedForecastCount":len(forecasts),"forecasts":forecasts}
 doc["manifestSha256"]=hashlib.sha256(_canonical(doc)).hexdigest();return doc
def build_from_paths():
 _check_sources()
 for path,expected,label in ((PRICES_PATH,PRICES_SHA256,"prices"),(RICH_PATH,RICH_SHA256,"rich Ridge"),(FIXED50_PATH,FIXED50_SHA256,"fixed50")):
  if _sha(path)!=expected:raise ValueError(f"pinned {label} hash mismatch")
 return build(json.loads(PRICES_PATH.read_text()),json.loads(RICH_PATH.read_text()),json.loads(FIXED50_PATH.read_text()),price_sha=PRICES_SHA256,rich_sha=RICH_SHA256,fixed50_sha=FIXED50_SHA256)
def _check_manifest(m):
 if m.get("phase")!="forecast_only" or m.get("targetValuesAttached") is not False:raise ValueError("expected answer-free manifest")
 u=dict(m);digest=u.pop("manifestSha256",None)
 if digest!=hashlib.sha256(_canonical(u)).hexdigest():raise ValueError("manifest hash mismatch")
 expected=set(_grid()); seen=set()
 for r in m.get("forecasts",[]):
  k=(r.get("product"),narrow._date(r.get("target")),r.get("horizon"))
  if k not in expected or k in seen:raise ValueError("invalid or duplicate manifest row")
  d,h=k[1],k[2]
  if narrow._date(r.get("origin"))!=d-dt.timedelta(days=h) or r.get("eligible") is not (d-dt.timedelta(days=h)>=ORIGIN_START):raise ValueError("origin/eligibility mismatch")
  for arm in ("selected","allFeature"):
   if r.get(arm+"Status") not in ("issued","not_issued") or (r[arm+"Status"]=="issued") != bool(narrow._positive(r.get(arm))):raise ValueError("forecast status/value mismatch")
  seen.add(k)
 if seen!=expected:raise ValueError("incomplete 13,500 row grid")
def verify_manifest(path):
 m=json.loads(Path(path).read_text());return _canonical(m)==_canonical(build_from_paths())
# Statistical helpers follow the reviewed evaluator's calendar HAC and Holm protocol.
def _dm(diff,bw):
 n=len(diff);mean=math.fsum(diff.values())/n if n else None
 if any(not math.isfinite(v) for v in diff.values()):return {"status":"nonfinite_loss_difference","reason":"nonfinite_loss_difference","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.}
 if n<30:return {"status":"insufficient_pairs","reason":"fewer_than_30_pairs","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.}
 vals=list(diff.values());spread=max(vals)-min(vals);mag=max(abs(min(vals)),abs(max(vals)))
 if spread<=8*math.ulp(mag):return {"status":"zero_spread","reason":"constant_loss_difference_within_float_tolerance","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.}
 c={d:v-mean for d,v in diff.items()};terms=[math.fsum(v*v for v in c.values())]
 for lag in range(1,bw+1):terms.append(2*(1-lag/(bw+1))*math.fsum(c[d]*c.get(d+dt.timedelta(days=lag),0.) for d in c))
 total=math.fsum(terms);tol=8*math.ulp(math.fsum(abs(x) for x in terms)) if any(terms) else 0.
 if not math.isfinite(total) or total<=tol:return {"status":"invalid_hac_variance","reason":"nonpositive_or_cancellation_dominated_hac_variance","n":n,"statistic":None,"meanLossDifference":mean,"pValue":1.}
 stat=mean/math.sqrt(total/(n*n));return {"status":"tested","reason":None,"n":n,"statistic":stat,"meanLossDifference":mean,"pValue":math.erfc(abs(stat)/math.sqrt(2))}
def _holm(rows):
 run=0.;m=len(rows)
 for rank,i in enumerate(sorted(range(m),key=lambda i:rows[i]["pValue"])):run=max(run,min(1.,(m-rank)*rows[i]["pValue"]));rows[i]["holmPValue"]=run;rows[i]["holmSignificant"]=run<ALPHA
def score(manifest,prices):
 _check_manifest(manifest); actuals=narrow._series(prices);lookup={(r["product"],narrow._date(r["target"]),r["horizon"]):r for r in manifest["forecasts"]};metrics=[];dm=[];coverage={};temporal={};common=[];days=[EVAL_START+dt.timedelta(days=i) for i in range((EVAL_END-EVAL_START).days+1)]
 for p in PRODUCTS:
  for h in HORIZONS:
   cells=[lookup[(p,d,h)] for d in days];eligible=[r for r in cells if r["eligible"]]; pairs=[]
   for r in eligible:
    d=narrow._date(r["target"]);actual=actuals[p].get(d);vals={a:(r.get(a) if r[a+"Status"]=="issued" else None) for a in ("selected","allFeature")};vals.update(richRidge=r.get("richRidge"),fixed50=r.get("fixed50"))
    if actual and all(narrow._positive(vals[k]) for k in ("selected","allFeature","richRidge","fixed50")):
     q={"product":p,"target":d.isoformat(),"horizon":h,"actual":actual,**vals};pairs.append(q);common.append(q)
   for name,a,b in COMPARISONS:
    def metric(k):
     es=[abs(r[k]-r["actual"]) for r in pairs];return {"n":len(es),"mae":math.fsum(es)/len(es) if es else None,"mape":100*math.fsum(e/r["actual"] for e,r in zip(es,pairs))/len(es) if es else None}
    metrics.append({"product":p,"horizon":h,"comparison":name,"a":metric(a),"b":metric(b)})
    for loss in LOSSES:
     diff={narrow._date(r["target"]):(abs(r[a]-r["actual"])-abs(r[b]-r["actual"]) if loss=="ae" else (abs(r[a]-r["actual"])-abs(r[b]-r["actual"]))/r["actual"]) for r in pairs}
     for bw in BANDWIDTHS:dm.append({"product":p,"horizon":h,"comparison":name,"loss":loss,"bandwidth":bw,**_dm(diff,bw)})
   coverage[f"{p}/{h}"]={"planned":len(cells),"excludedPrefit":sum(not r["eligible"] for r in cells),"eligiblePlanned":len(eligible),"positiveActuals":sum(narrow._date(r["target"]) in actuals[p] for r in eligible),"allFourCommonPairs":len(pairs)}
   for arm in ("selected","allFeature","richRidge","fixed50"):
    coverage[f"{p}/{h}"][arm+"Issued"]=sum(narrow._positive(r.get(arm)) is not None for r in eligible)
    coverage[f"{p}/{h}"][arm+"CoverageLossTargets"]=[r["target"] for r in eligible if narrow._positive(r.get("fixed50")) and narrow._positive(r.get(arm)) is None] if arm!="fixed50" else []
 for bw in BANDWIDTHS:
  family=[r for r in dm if r["bandwidth"]==bw]
  if len(family)!=400:raise ValueError("frozen 400-test Holm family mismatch")
  _holm(family)
 macros=[];gates=[]
 for p in PRODUCTS:
  for name,a,b in COMPARISONS:
   rows=[r for r in metrics if r["product"]==p and r["comparison"]==name and r["a"]["n"]]
   macros.append({"product":p,"comparison":name,"horizonsIncluded":len(rows),"aEqualHorizonMae":math.fsum(r["a"]["mae"] for r in rows)/len(rows) if rows else None,"bEqualHorizonMae":math.fsum(r["b"]["mae"] for r in rows)/len(rows) if rows else None,"aEqualHorizonMape":math.fsum(r["a"]["mape"] for r in rows)/len(rows) if rows else None,"bEqualHorizonMape":math.fsum(r["b"]["mape"] for r in rows)/len(rows) if rows else None})
  rows=[r for r in metrics if r["product"]==p and r["comparison"]=="selected_vs_fixed50" and r["a"]["n"]]; ma=math.fsum(r["a"]["mape"] for r in rows)/len(rows) if rows else None;mb=math.fsum(r["b"]["mape"] for r in rows)/len(rows) if rows else None;ea=math.fsum(r["a"]["mae"] for r in rows)/len(rows) if rows else None;eb=math.fsum(r["b"]["mae"] for r in rows)/len(rows) if rows else None
  covloss=any(coverage[f"{p}/{h}"]["selectedCoverageLossTargets"] for h in HORIZONS);no_harm=len(rows)==10 and all(r["a"]["mape"]-r["b"]["mape"]<=1 for r in rows);joint=False
  for h in HORIZONS:
   q={(r["loss"],r["bandwidth"]):r for r in dm if r["product"]==p and r["horizon"]==h and r["comparison"]=="selected_vs_fixed50"}
   if all(q[(l,bw)]["holmSignificant"] and q[(l,bw)]["meanLossDifference"]<0 for l in LOSSES for bw in BANDWIDTHS):joint=True
  passed=bool(len(rows)==10 and ma is not None and mb is not None and ma<=mb-.5 and ea<eb and not covloss and no_harm and joint)
  gates.append({"product":p,"candidateEqualHorizonMape":ma,"fixed50EqualHorizonMape":mb,"mapeImprovementPctPoints":mb-ma if ma is not None and mb is not None else None,"candidateEqualHorizonMae":ea,"fixed50EqualHorizonMae":eb,"coverageNondeclining":not covloss,"noHorizonMapeWorseByMoreThan1PctPoint":no_harm,"jointAeApeHolmImprovementAtSomeHorizonBothHac":joint,"decision":"eligible_for_prospective_confirmation" if passed else "retain_existing_method"})
 for p in PRODUCTS:
  for period in ("month","quarter"):
   groups={}
   for d in days:groups.setdefault(f"{d.year}-{d.month:02d}" if period=="month" else f"{d.year}-Q{(d.month-1)//3+1}",[]).append(d)
   for label,ds in groups.items():
    rs=[lookup[(p,d,h)] for d in ds for h in HORIZONS];eligible=[r for r in rs if r["eligible"]];temporal[f"{p}/{period}/{label}"]={"planned":len(rs),"excludedPrefit":sum(not r["eligible"] for r in rs),"eligiblePlanned":len(eligible),"selectedIssued":sum(r["selectedStatus"]=="issued" for r in eligible),"allFeatureIssued":sum(r["allFeatureStatus"]=="issued" for r in eligible),"richRidgeIssued":sum(narrow._positive(r.get("richRidge")) is not None for r in eligible),"fixed50Issued":sum(narrow._positive(r.get("fixed50")) is not None for r in eligible),"positiveActualPairs":sum(narrow._date(r["target"]) in actuals[p] for r in eligible),"allFourCommonPairs":sum(1 for r in eligible if narrow._date(r["target"]) in actuals[p] and all(narrow._positive(r.get(k)) is not None for k in ("selected","allFeature","richRidge","fixed50")))}
 return {"schemaVersion":1,"phase":"scored","manifestSha256":manifest["manifestSha256"],"metrics":metrics,"macros":macros,"coverage":coverage,"temporalCoverage":temporal,"candidateGates":gates,"dmSummaryByBandwidth":[{"bandwidth":bw,"familySize":sum(r["bandwidth"]==bw for r in dm),"significantImprovement":sum(r["bandwidth"]==bw and r["status"]=="tested" and r["holmSignificant"] and r["meanLossDifference"]<0 for r in dm),"significantWorsening":sum(r["bandwidth"]==bw and r["status"]=="tested" and r["holmSignificant"] and r["meanLossDifference"]>0 for r in dm),"inferenceUnavailable":sum(r["bandwidth"]==bw and r["status"]!="tested" for r in dm)} for bw in BANDWIDTHS],"dm":dm,"commonPairs":common}
def write_exclusive(path,value):
 Path(path).parent.mkdir(parents=True,exist_ok=True)
 with Path(path).open("xb") as f:f.write(_canonical(value))
def main(argv=None):
 ap=argparse.ArgumentParser();s=ap.add_subparsers(dest="cmd",required=True);b=s.add_parser("build");b.add_argument("output",type=Path);v=s.add_parser("verify");v.add_argument("manifest",type=Path);q=s.add_parser("score");q.add_argument("manifest",type=Path);q.add_argument("output",type=Path);a=ap.parse_args(argv)
 if a.cmd=="build":write_exclusive(a.output,build_from_paths())
 elif a.cmd=="verify":
  if not verify_manifest(a.manifest):raise SystemExit("manifest replay mismatch")
 else:
  if not verify_manifest(a.manifest):raise SystemExit("manifest replay mismatch")
  if _sha(PRICES_PATH)!=PRICES_SHA256:raise SystemExit("pinned price hash mismatch")
  write_exclusive(a.output,score(json.loads(a.manifest.read_text()),json.loads(PRICES_PATH.read_text())))
 return 0
if __name__=="__main__":raise SystemExit(main())
