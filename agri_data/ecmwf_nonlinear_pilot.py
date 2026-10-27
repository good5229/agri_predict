"""Offline-only nonlinear ECMWF feature pilot with separated forecast and score stages."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math
from pathlib import Path
from typing import Any

from . import dashboard, ecmwf_price_eval as base, direct_horizon as dh

ROOT = Path(__file__).resolve().parents[1]
PRICE_PATH = dh.INPUT_PRICES
DIRECT_PATH = ROOT / "outputs/experiments/direct-horizon-20261006/report.json"
FEATURE_PATH = ROOT / "outputs/experiments/ecmwf-point-features-20261007/features_fe353b1d99a785fee5f23fa3f9acfad71445c231aaa01f649e49258b44717fe0.json"
FEATURE_MANIFEST_PATH = ROOT / "outputs/experiments/ecmwf-point-features-20261007/manifest_737b36aa5f0f938d45eaf18e5ef9e97e928857960ed46f3c2cb812edb86a6686.json"
PRICE_SHA = "f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
DIRECT_SHA = "a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
FEATURE_SHA = "fe353b1d99a785fee5f23fa3f9acfad71445c231aaa01f649e49258b44717fe0"
FEATURE_MANIFEST_SHA = "737b36aa5f0f938d45eaf18e5ef9e97e928857960ed46f3c2cb812edb86a6686"
PRODUCTS = tuple(dashboard.PRODUCTS)
HORIZONS = (1, 5, 10)
TRAIN_START, TRAIN_END = dt.date(2024,3,16), dt.date(2026,5,7)
EVAL_START, EVAL_END = dt.date(2026,6,1), dt.date(2026,7,31)
BANDWIDTHS = (9,28); COMPARATORS = ("linear", "direct", "persistence"); LOSSES=("mae","mape")
FAMILY=5*3*3*2
PROTOCOL={"id":"ecmwf-nonlinear-pilot-2026-10-08","trainTargetDates":[TRAIN_START.isoformat(),TRAIN_END.isoformat()],"evalOrigins":[EVAL_START.isoformat(),EVAL_END.isoformat()],"products":list(PRODUCTS),"horizons":list(HORIZONS),"requiredGridRows":915,"minimumDistinctTrainingTargetDates":90,"target":"log(actual/direct)","nonlinear":{"estimator":"HistGradientBoostingRegressor","max_iter":60,"max_leaf_nodes":3,"min_samples_leaf":25,"l2_regularization":25,"learning_rate":0.05,"random_state":0},"linear":{"estimator":"StandardScaler(training rows only)+Ridge","alpha":100,"fit_intercept":True},"clipLogCorrection":[-0.08,0.08],"comparisons":list(COMPARATORS),"dm":{"twoSided":True,"calendarBartlettHacDays":list(BANDWIDTHS),"familySizePerBandwidth":FAMILY,"holmAlpha":0.05,"minimumPairs":30,"unavailablePValue":1.0},"expansionGates":{"commonPairsPerCell":30,"nonlinearCoverageAtLeastDirect":True,"nonlinearFiveProductMacroMapeAtMostPct":10,"nonlinearMapeImprovementVsLinearAndDirectPctPoints":0.5,"perProductMapeWorseningAtMostPctPoints":1,"perProductGateBenchmark":"direct","equalThreeHorizonMaeWorseningAtMostPct":5}}
class PilotError(ValueError): pass

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def canon(x:Any)->bytes:return (json.dumps(x,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
def load(p:Path):
 try:
  b=p.read_bytes(); x=json.loads(b)
 except (OSError,json.JSONDecodeError) as e: raise PilotError(f"cannot read JSON: {p}") from e
 if not isinstance(x,dict):raise PilotError(f"JSON root must be object: {p}")
 return x,b
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def product_series(prices):
 out={}
 if prices.get('schemaVersion')!=1:raise PilotError('unexpected prices schema')
 for p in prices.get('products',[]):
  pid=p.get('id')
  if pid not in PRODUCTS or pid in out:raise PilotError('unknown/duplicate product')
  vals={}
  for r in p.get('observations',[]):
   try:d=dt.date.fromisoformat(r['date']); v=float(r['actual'])
   except (ValueError,TypeError,KeyError):continue
   if math.isfinite(v) and v>0: vals[d]=v
  out[pid]=vals
 if set(out)!=set(PRODUCTS):raise PilotError('prices must contain all five products')
 return out

def _report_map(report):
 if report.get('protocol',{}).get('id')!='direct-horizon-ridge-2026-10-06':raise PilotError('unexpected direct report protocol')
 result={}
 for r in report.get('pairedRows',[]):
  try:
   pid=r['product']; target=dt.date.fromisoformat(r['target']); h=r['horizon']; direct=float(r['direct'])
  except (KeyError,ValueError,TypeError):continue
  if pid not in PRODUCTS or isinstance(h,bool) or h not in range(1,11) or not math.isfinite(direct) or direct<=0:continue
  origin=target-dt.timedelta(days=h)
  if r.get('origin') and dt.date.fromisoformat(r['origin'])!=origin:raise PilotError('direct row origin mismatch')
  key=(pid,h,target)
  if key in result:raise PilotError('duplicate direct row')
  # Deliberately do not access r['actual'] here: forecast construction cannot see evaluation answers.
  result[key]={'product':pid,'horizon':h,'target':target,'origin':origin,'direct':direct}
 if not result:raise PilotError('direct report contains no predictions')
 return result

def build_forecast(prices, report, feature_doc, feature_manifest, hashes=None, feature_bytes=None):
 vectors=base.validate_feature_inputs(feature_doc,feature_manifest,feature_bytes=feature_bytes)
 series=product_series(prices); direct=_report_map(report); grid=[]; models={}
 for pid in PRODUCTS:
  for h in HORIZONS:
   train=[r for r in direct.values() if r['product']==pid and r['horizon']==h and TRAIN_START<=r['target']<=TRAIN_END and r['target'] in series[pid] and vectors.get(r['origin'].isoformat()) is not None]
   train.sort(key=lambda r:(r['target'],r['origin'])); days=sorted({r['target'] for r in train})
   hg=scaler=ridge=None
   if len(days)<90: hg_status=ridge_status='insufficient_training'
   else:
    try:
     import numpy as np
     from sklearn.ensemble import HistGradientBoostingRegressor
     X=np.asarray([vectors[r['origin'].isoformat()] for r in train],float); y=np.asarray([math.log(series[pid][r['target']]/r['direct']) for r in train],float)
     if not np.isfinite(X).all() or not np.isfinite(y).all(): raise ValueError('nonfinite training data')
     hg=HistGradientBoostingRegressor(max_iter=60,max_leaf_nodes=3,min_samples_leaf=25,l2_regularization=25,learning_rate=.05,random_state=0).fit(X,y); hg_status='fitted'
    except Exception as e: hg_status='fit_failed:'+type(e).__name__
    try:
     import numpy as np
     from sklearn.linear_model import Ridge
     from sklearn.preprocessing import StandardScaler
     X=np.asarray([vectors[r['origin'].isoformat()] for r in train],float); y=np.asarray([math.log(series[pid][r['target']]/r['direct']) for r in train],float)
     if not np.isfinite(X).all() or not np.isfinite(y).all(): raise ValueError('nonfinite training data')
     scaler=StandardScaler().fit(X); scaler.scale_[scaler.scale_==0]=1; ridge=Ridge(alpha=100,fit_intercept=True).fit(scaler.transform(X),y); ridge_status='fitted'
    except Exception as e: ridge_status='fit_failed:'+type(e).__name__
   audit={'trainingRows':len(train),'distinctTrainingTargetDates':len(days),'nonlinearStatus':hg_status,'linearStatus':ridge_status,'trainingTargetStart':days[0].isoformat() if days else None,'trainingTargetEnd':days[-1].isoformat() if days else None}
   models[(pid,h)]=(hg,scaler,ridge,audit)
 for i in range((EVAL_END-EVAL_START).days+1):
  origin=EVAL_START+dt.timedelta(days=i); vec=vectors.get(origin.isoformat())
  for pid in PRODUCTS:
   src=max((d for d in series[pid] if d<origin),default=None)
   persistence=series[pid][src] if src and (origin-src).days<=dashboard.MAX_SOURCE_AGE_DAYS else None
   for h in HORIZONS:
    target=origin+dt.timedelta(days=h); r=direct.get((pid,h,target)); direct_value=r['direct'] if r else None
    hg,scaler,ridge,audit=models[(pid,h)]; values={'nonlinear':None,'linear':None}; reasons={}
    if direct_value is None: reasons['directMissingReason']='direct_prediction_missing'
    for name,model in (('nonlinear',hg),('linear',ridge)):
     reason='direct_prediction_missing' if direct_value is None else None
     if reason is None and model is None: reason=audit['nonlinearStatus' if name=='nonlinear' else 'linearStatus']
     elif reason is None and vec is None: reason='feature_missing_or_nonfinite'
     elif reason is None:
      try:
       import numpy as np
       x=np.asarray([vec],float); correction=float(model.predict(x)[0] if name=='nonlinear' else model.predict(scaler.transform(x))[0])
       if not math.isfinite(correction): raise ValueError('nonfinite correction')
       pred=direct_value*math.exp(max(-.08,min(.08,correction)))
       if not math.isfinite(pred) or pred<=0: raise ValueError('invalid corrected forecast')
       values[name]=pred
      except Exception as e: reason='prediction_failed:'+type(e).__name__
     if reason: reasons[name+'FallbackReason']=reason
     if reason and direct_value is not None: values[name]=direct_value
    if persistence is None: reasons['persistenceMissingReason']='no_prior_source_or_source_too_old'
    grid.append({'product':pid,'origin':origin.isoformat(),'target':target.isoformat(),'horizon':h,'nonlinear':values['nonlinear'],'linear':values['linear'],'direct':direct_value,'persistence':persistence,'persistenceSourceDate':src.isoformat() if persistence is not None else None,'featureOrigin':origin.isoformat(),'featureAvailable':vec is not None,**reasons})
 if len(grid)!=915 or len({(r['product'],r['origin'],r['horizon']) for r in grid})!=915: raise PilotError('forecast grid is not exactly 915 unique rows')
 fallback_counts={name:{} for name in ('nonlinear','linear','direct','persistence')}
 for row in grid:
  for name,reason_key in (('nonlinear','nonlinearFallbackReason'),('linear','linearFallbackReason'),('direct','directMissingReason'),('persistence','persistenceMissingReason')):
   reason=row.get(reason_key)
   if reason:fallback_counts[name][reason]=fallback_counts[name].get(reason,0)+1
 return {'schemaVersion':1,'status':'answer_free_forecast_manifest','protocol':PROTOCOL,'inputs':hashes or {},'featureManifest':{'featureSha256':feature_manifest.get('feature_sha256'),'sourceScopeHash':feature_manifest.get('source_scope_hash'),'featureOrder':feature_manifest.get('feature_order')},'trainingAudit':{f'{p}/{h}':models[(p,h)][3] for p in PRODUCTS for h in HORIZONS},'fallbackCounts':fallback_counts,'forecasts':grid}

def holm(rows):
 ix=sorted(range(len(rows)),key=lambda i:rows[i]['pValue']); run=0.
 for rank,i in enumerate(ix):run=max(run,min(1.,(len(rows)-rank)*rows[i]['pValue']));rows[i]['holmPValue']=run;rows[i]['holmSignificant']=run<.05
def metric(rows,name):
 rows=[r for r in rows if finite(r.get(name)) and r[name]>0]
 if not rows:return {'mae':None,'mape':None}
 return {'mae':sum(abs(r['actual']-r[name]) for r in rows)/len(rows),'mape':100*sum(abs(r['actual']-r[name])/r['actual'] for r in rows)/len(rows)}
def dm(rows,comp,loss,bw):
 diff={dt.date.fromisoformat(r['target']):(abs(r['actual']-r['nonlinear'])-abs(r['actual']-r[comp]) if loss=='mae' else 100*(abs(r['actual']-r['nonlinear'])-abs(r['actual']-r[comp]))/r['actual']) for r in rows}
 return dh._dm(diff,bw)
def expected_grid():
 return {(p,(EVAL_START+dt.timedelta(days=i)).isoformat(),h, (EVAL_START+dt.timedelta(days=i)+dt.timedelta(days=h)).isoformat()) for i in range(61) for p in PRODUCTS for h in HORIZONS}

def validate_forecasts(forecasts, series):
 if not isinstance(forecasts,list) or len(forecasts)!=915: raise PilotError('forecast grid must contain exactly 915 rows')
 seen=set()
 for f in forecasts:
  if not isinstance(f,dict): raise PilotError('forecast row must be an object')
  if 'actual' in f: raise PilotError('answer-bearing actual key is forbidden in forecast rows')
  try: p=f['product']; o=dt.date.fromisoformat(f['origin']); t=dt.date.fromisoformat(f['target']); h=f['horizon']
  except (KeyError,ValueError,TypeError): raise PilotError('malformed forecast identity')
  key=(p,o.isoformat(),h,t.isoformat())
  if key in seen: raise PilotError('duplicate forecast grid row')
  seen.add(key)
  if p not in PRODUCTS or h not in HORIZONS or t!=o+dt.timedelta(days=h): raise PilotError('forecast identity outside fixed grid')
  if key not in expected_grid(): raise PilotError('forecast row is outside fixed 915-row grid')
  for name in ('nonlinear','linear','direct','persistence'):
   value=f.get(name); reason=f.get(name+'FallbackReason') if name in ('nonlinear','linear') else f.get('directMissingReason') if name=='direct' else f.get('persistenceMissingReason')
   if value is not None and (not finite(value) or value<=0): raise PilotError(f'invalid {name} forecast value')
   if value is None and not reason: raise PilotError(f'null {name} forecast lacks reason')
   if value is not None and reason and name in ('direct','persistence'): raise PilotError(f'non-null {name} forecast contradicts missing reason')
   if name=='persistence' and value is not None:
    try: src=dt.date.fromisoformat(f['persistenceSourceDate'])
    except (KeyError,ValueError,TypeError): raise PilotError('persistence source date missing')
    if src>=o or src not in series[p] or not math.isclose(value,series[p][src],rel_tol=1e-12,abs_tol=1e-12): raise PilotError('persistence must match an observed price strictly before origin')
   if name=='persistence' and value is None and f.get('persistenceSourceDate') is not None: raise PilotError('null persistence has a source date')
  for name in ('nonlinear','linear'):
   if f[name] is None and f.get('direct') is not None: raise PilotError(f'{name} may be null only when direct is missing')
   if f.get('direct') is None and (f[name] is not None or f.get('directMissingReason')!='direct_prediction_missing'): raise PilotError(f'{name} must be null with explicit missing-direct reason')
   if f.get(name+'FallbackReason') and f.get('direct') is not None and f[name]!=f['direct']: raise PilotError(f'{name} fallback must equal direct forecast')
   if f.get(name+'FallbackReason') and f.get('direct') is None and f.get(name+'FallbackReason')!='direct_prediction_missing': raise PilotError(f'{name} fallback reason contradicts missing direct prediction')
  if f.get('direct') is not None and f.get('directMissingReason') is not None: raise PilotError('direct missing reason contradicts available direct forecast')
 if seen!=expected_grid(): raise PilotError('forecast rows do not exactly cover the fixed grid')

def score(forecast, prices, forecast_bytes=None, prices_bytes=None):
 if forecast.get('status')!='answer_free_forecast_manifest' or forecast.get('protocol')!=PROTOCOL: raise PilotError('forecast manifest protocol/status mismatch')
 if forecast_bytes is not None:
  check=dict(forecast); stated=check.pop('manifestSha256',None)
  if sha(canon(check))!=stated: raise PilotError('forecast manifest SHA mismatch')
 if prices_bytes is not None and forecast.get('inputs',{}).get('prices')!=sha(prices_bytes): raise PilotError('price input hash differs from forecast manifest')
 series=product_series(prices); forecasts=forecast.get('forecasts'); validate_forecasts(forecasts,series)
 actual_rows=[]; drops={'missing_actual':0,'missing_nonlinear':0,'missing_linear':0,'missing_direct':0,'missing_persistence':0}
 forecast_counts={m:0 for m in ('nonlinear','linear','direct','persistence')}; scoreable_counts={m:0 for m in forecast_counts}
 for f in forecasts:
  for m in forecast_counts:
   if finite(f.get(m)) and f[m]>0: forecast_counts[m]+=1
  target=dt.date.fromisoformat(f['target']); y=series[f['product']].get(target)
  if y is None: drops['missing_actual']+=1; continue
  if not finite(y) or y<=0: drops['missing_actual']+=1; continue
  row={**f,'actual':y}; actual_rows.append(row)
  for m in scoreable_counts:
   if finite(f.get(m)) and f[m]>0: scoreable_counts[m]+=1
   else:drops['missing_'+m]+=1
 common=[r for r in actual_rows if all(finite(r.get(m)) and r[m]>0 for m in forecast_counts)]
 bycell={}; comparisons=[]
 for p in PRODUCTS:
  for h in HORIZONS:
   cell=[r for r in common if r['product']==p and r['horizon']==h]; bycell[f'{p}/{h}']={'n':len(cell),'metrics':{m:metric(cell,m) for m in ('nonlinear','linear','direct','persistence')}}
   for c in COMPARATORS:
    for loss in LOSSES:
     tests={str(b):dm(cell,c,loss,b) for b in BANDWIDTHS}; comparisons.append({'product':p,'horizon':h,'comparator':c,'loss':loss,'n':len(cell),'candidateMetric':bycell[f'{p}/{h}']['metrics']['nonlinear'][loss],'comparatorMetric':bycell[f'{p}/{h}']['metrics'][c][loss],'dm':tests})
 for b in BANDWIDTHS: holm([x['dm'][str(b)] for x in comparisons])
 macro={}; macro_mae={}; product_metrics={}
 for m in ('nonlinear','linear','direct','persistence'):
  product_metrics[m]={p:{'mape':sum(bycell[f'{p}/{h}']['metrics'][m]['mape'] for h in HORIZONS)/3,'mae':sum(bycell[f'{p}/{h}']['metrics'][m]['mae'] for h in HORIZONS)/3} if all(bycell[f'{p}/{h}']['metrics'][m]['mape'] is not None for h in HORIZONS) else {'mape':None,'mae':None} for p in PRODUCTS}
  vals=[product_metrics[m][p]['mape'] for p in PRODUCTS]
  macro[m]=sum(vals)/5 if all(v is not None for v in vals) else None
  maes=[product_metrics[m][p]['mae'] for p in PRODUCTS]
  macro_mae[m]=sum(maes)/5 if all(v is not None for v in maes) else None
 product_deltas={}; per_product_pass=[]
 for p in PRODUCTS:
  nm=product_metrics['nonlinear'][p]['mape']; dmape=product_metrics['direct'][p]['mape']; na=product_metrics['nonlinear'][p]['mae']; da=product_metrics['direct'][p]['mae']
  delta=nm-dmape if nm is not None and dmape is not None else None
  mae_pct=100*(na-da)/da if na is not None and da is not None and da>0 else None
  product_deltas[p]={'mapeWorseningPctPointsVsDirect':delta,'equalThreeHorizonMaeWorseningPctVsDirect':mae_pct}
  per_product_pass.append(delta is not None and delta<=1 and mae_pct is not None and mae_pct<=5)
 macro_improve={c:(macro['nonlinear'] is not None and macro[c] is not None and macro['nonlinear']<=macro[c]-.5) for c in ('linear','direct')}
 candidate_coverage=forecast_counts['nonlinear']/915; direct_coverage=forecast_counts['direct']/915
 gates={'all15CellsCommonPairsAtLeast30':all(bycell[f'{p}/{h}']['n']>=30 for p in PRODUCTS for h in HORIZONS),'nonlinearCoverageAtLeastDirect':candidate_coverage>=direct_coverage,'nonlinearFiveProductMacroMapeAtMost10Pct':macro['nonlinear'] is not None and macro['nonlinear']<=10,'nonlinearMapeImprovesVsLinearByAtLeast0_5PctPoints':macro_improve['linear'],'nonlinearMapeImprovesVsDirectByAtLeast0_5PctPoints':macro_improve['direct'],'everyProductMapeWorseningAtMost1PctPoint':all(x['mapeWorseningPctPointsVsDirect'] is not None and x['mapeWorseningPctPointsVsDirect']<=1 for x in product_deltas.values()),'everyProductEqualThreeHorizonMaeWorseningAtMost5Pct':all(x['equalThreeHorizonMaeWorseningPctVsDirect'] is not None and x['equalThreeHorizonMaeWorseningPctVsDirect']<=5 for x in product_deltas.values())}
 gates['expand']=all(gates.values())
 conclusions=[]
 for p in PRODUCTS:
  for h in HORIZONS:
   c=[x for x in comparisons if x['product']==p and x['horizon']==h]; tests=[x['dm']['9'] for x in c]
   lower=all(x['candidateMetric'] is not None and x['comparatorMetric'] is not None and x['candidateMetric']<x['comparatorMetric'] for x in c)
   tested=all(t['status']=='tested' for t in tests); improve=all(t.get('holmSignificant') and t.get('meanLossDifference') is not None and t['meanLossDifference']<0 for t in tests)
   adverse=any(t.get('holmSignificant') and t.get('meanLossDifference') is not None and t['meanLossDifference']>0 for t in tests)
   conclusions.append({'product':p,'horizon':h,'conclusion':'improvement' if lower and tested and improve else ('worsening_or_mixed' if adverse else ('inference_unavailable' if not tested else 'inconclusive')),'allPairedLossesLower':lower,'allHac9TestsImprove':improve,'inferenceAvailable':tested})
 return {'schemaVersion':1,'status':'scored','forecastManifestSha256':forecast.get('manifestSha256'),'pairedRows':len(common),'fourWayCommonRows':len(common),'byProductHorizon':bycell,'fiveProductEqualWeightThreeHorizonMape':macro,'fiveProductEqualWeightThreeHorizonMae':macro_mae,'byProductEqualHorizonMetrics':product_metrics,'productGateDeltas':product_deltas,'gates':gates,'cellConclusions':conclusions,'coverage':{'forecastRows':len(forecasts),'rowsWithActual':len(actual_rows),'validForecastRows':forecast_counts,'fractionOfPlannedGrid':{k:v/915 for k,v in forecast_counts.items()},'scoreableValidForecastRows':scoreable_counts,'fractionOfRowsWithActual':{k:(v/len(actual_rows) if actual_rows else 0.0) for k,v in scoreable_counts.items()},'fourWayCommonRows':len(common),'dropReasons':drops},'dm':{'familySize':FAMILY,'comparisons':comparisons,'bandwidths':list(BANDWIDTHS)},'sampleLimits':{'requiredPlanRows':915,'observedForecastRows':len(forecasts),'scoredRows':len(common),'minimumPerCellForInference':30,'cellsBelowThirty':[k for k,v in bycell.items() if v['n']<30]}}

def write_immutable(path,raw):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(raw)
def main(argv=None):
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
 b=sp.add_parser('forecast');b.add_argument('--prices',type=Path,default=PRICE_PATH);b.add_argument('--direct-report',type=Path,default=DIRECT_PATH);b.add_argument('--features',type=Path,default=FEATURE_PATH);b.add_argument('--feature-manifest',type=Path,default=FEATURE_MANIFEST_PATH);b.add_argument('--output',type=Path,required=True)
 s=sp.add_parser('score');s.add_argument('--forecast-manifest',type=Path,required=True);s.add_argument('--prices',type=Path,default=PRICE_PATH);s.add_argument('--output',type=Path,required=True)
 a=ap.parse_args(argv)
 try:
  if a.cmd=='forecast':
   docs=[]
   for p in (a.prices,a.direct_report,a.features,a.feature_manifest):docs.append(load(p))
   (prices,pb),(report,rb),(features,fb),(fm,fmb)=docs
   for name,raw,expected in [('prices',pb,PRICE_SHA),('direct',rb,DIRECT_SHA),('features',fb,FEATURE_SHA),('feature manifest',fmb,FEATURE_MANIFEST_SHA)]:
    if sha(raw)!=expected:raise PilotError(f'pinned {name} SHA mismatch')
   hashes={'prices':sha(pb),'directReport':sha(rb),'features':sha(fb),'featureManifest':sha(fmb),'code':sha(Path(__file__).read_bytes())}
   result=build_forecast(prices,report,features,fm,hashes,feature_bytes=fb);result['manifestSha256']=sha(canon(result));write_immutable(a.output,canon(result))
  else:
   f,fb=load(a.forecast_manifest);p,pb=load(a.prices)
   if sha(pb)!=PRICE_SHA:raise PilotError('pinned prices SHA mismatch')
   for key,path,expected in [('directReport',DIRECT_PATH,DIRECT_SHA),('features',FEATURE_PATH,FEATURE_SHA),('featureManifest',FEATURE_MANIFEST_PATH,FEATURE_MANIFEST_SHA),('code',Path(__file__),None)]:
    observed=sha(path.read_bytes())
    if observed!=f.get('inputs',{}).get(key) or (expected is not None and observed!=expected):raise PilotError(f'{key} input/code hash differs from forecast manifest')
   result=score(f,p,fb,pb);result['pricesSha256']=sha(pb);result['reportSha256']=f.get('inputs',{}).get('directReport');result['codeSha256']=f.get('inputs',{}).get('code');write_immutable(a.output,canon(result))
 except (PilotError,OSError,ValueError,KeyError) as e:ap.error(str(e))
 return 0
if __name__=='__main__':raise SystemExit(main())
