"""Offline analog-residual pilot with an answer-free forecast manifest and separate scorer."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math
from pathlib import Path
from typing import Any

from . import dashboard, direct_horizon as dh, modeling

ROOT=Path(__file__).resolve().parents[1]
PRICE_PATH=dh.INPUT_PRICES
DIRECT_PATH=ROOT/"outputs/experiments/direct-horizon-20261006/report.json"
PRICE_SHA="f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
DIRECT_SHA="a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
PRODUCTS=tuple(dashboard.PRODUCTS); HORIZONS=(1,5,10)
TRAIN_END=dt.date(2024,8,7); EVAL_START=dt.date(2024,8,8); EVAL_END=dt.date(2024,10,31)
LOOKBACK=90; SOURCE_MAX_AGE=14; MIN_TRAIN_DAYS=90; NEIGHBORS=20; CLIP=.08
BANDWIDTHS=(9,28); COMPARATORS=("meanAblation","direct","persistence"); LOSSES=("mae","mape"); FAMILY=5*3*3*2
PROTOCOL={"id":"analog-residual-pilot-2026-10-08","trainingTargetEnd":TRAIN_END.isoformat(),"evaluationOrigins":[EVAL_START.isoformat(),EVAL_END.isoformat()],"products":list(PRODUCTS),"horizons":list(HORIZONS),"gridRows":1275,"features":["log(last price / third-last price)","log(last price / tenth-last price)","log(mean last three volumes / mean preceding seven volumes)","annual sine","annual cosine"],"sourceRule":"90 calendar days strictly before origin; last 10 positive price and volume observations; latest observation at most 14 days old","minimumDistinctTrainingTargetDates":MIN_TRAIN_DAYS,"neighborCount":NEIGHBORS,"distance":"Euclidean after training-only per-feature standardization; ties by target date ascending","neighborWeights":"1/(distance+0.1)","target":"log(actual/direct)","clipLogCorrection":[-CLIP,CLIP],"comparators":list(COMPARATORS),"dm":{"twoSided":True,"calendarBartlettHacDays":list(BANDWIDTHS),"familySizePerBandwidth":FAMILY,"holmAlpha":.05,"minimumPairs":30,"unavailablePValue":1.0},"gates":{"commonPairsPerCell":30,"candidateCoverageAtLeastDirect":True,"candidateFiveProductMacroMapeAtMostPct":10,"candidateMapeImprovementVsMeanAndDirectPctPoints":.5,"perProductMapeWorseningAtMostPctPoints":1,"perProductGateBenchmark":"direct","equalThreeHorizonMaeWorseningAtMostPct":5},"limitation":"Pinned direct Ridge report was generated with origin-day information; this pilot does not claim strict historical issuance-time reconstruction."}
class PilotError(ValueError):pass

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def canon(v:Any)->bytes:return (json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def load(path):
 try: raw=Path(path).read_bytes(); obj=json.loads(raw)
 except (OSError,json.JSONDecodeError) as e:raise PilotError(f"cannot read JSON: {path}") from e
 if not isinstance(obj,dict):raise PilotError("JSON root must be object")
 return obj,raw

def _series(prices):
 try: out=dh._observations(prices)
 except (ValueError,TypeError,KeyError) as e:raise PilotError(f"invalid frozen prices: {e}") from e
 return out

def _direct_map(report):
 if report.get('schemaVersion',1)!=1 or report.get('protocol',{}).get('id')!='direct-horizon-ridge-2026-10-06':raise PilotError('unexpected direct report schema/protocol')
 out={}
 for row in report.get('pairedRows',[]):
  if not isinstance(row,dict):raise PilotError('direct report rows must be objects')
  p=row.get('product'); h=row.get('horizon')
  if p not in PRODUCTS or isinstance(h,bool) or not isinstance(h,int) or h not in range(1,11):raise PilotError('invalid direct report product/horizon')
  try:t=dt.date.fromisoformat(row['target']); o=dt.date.fromisoformat(row.get('origin') or (t-dt.timedelta(days=h)).isoformat()); pred=float(row['direct'])
  except (KeyError,TypeError,ValueError) as e:raise PilotError('malformed direct report row') from e
  if t-o!=dt.timedelta(days=h):raise PilotError('direct report origin/horizon mismatch')
  if not finite(pred) or pred<=0:continue
  key=(p,o,t,h)
  if key in out:raise PilotError('duplicate direct report row')
  # Deliberately do not inspect report actuals in the forecast stage.
  out[key]={'product':p,'origin':o,'target':t,'horizon':h,'direct':pred}
 return out

def state_features(series,origin):
 cutoff=origin-dt.timedelta(days=1); start=origin-dt.timedelta(days=LOOKBACK)
 days=sorted(d for d,(price,vol) in series.items() if start<=d<=cutoff and finite(price) and price>0 and finite(vol) and vol>0)
 if len(days)<10:return None,{'status':'fewer_than_ten_valid_observations','validObservations':len(days)}
 recent=days[-10:]; last=recent[-1]
 if (origin-last).days>SOURCE_MAX_AGE:return None,{'status':'latest_source_stale','validObservations':len(days),'latestSourceDate':last.isoformat()}
 ps=[series[d][0] for d in recent]; vs=[series[d][1] for d in recent]
 vals=[math.log(ps[-1]/ps[-3]),math.log(ps[-1]/ps[0]),math.log((sum(vs[-3:])/3)/(sum(vs[:7])/7))]
 angle=2*math.pi*(origin.timetuple().tm_yday-1)/365.25
 vals.extend((math.sin(angle),math.cos(angle)))
 if not all(math.isfinite(v) for v in vals):return None,{'status':'nonfinite_feature','validObservations':len(days),'latestSourceDate':last.isoformat()}
 return vals,{'status':'available','validObservations':len(days),'featureDates':[d.isoformat() for d in recent],'latestSourceDate':last.isoformat()}

def _fit_models(direct,series_map):
 fitted={}
 for p in PRODUCTS:
  for h in HORIZONS:
   train=[]
   for row in direct.values():
    if row['product']!=p or row['horizon']!=h or row['target']>TRAIN_END:continue
    actual=series_map[p].get(row['target']); x,audit=state_features(series_map[p],row['origin'])
    if actual is not None and finite(actual[0]) and actual[0]>0 and x is not None:train.append({**row,'actual':actual[0],'x':x})
   train.sort(key=lambda r:(r['target'],r['origin'])); days=sorted({r['target'] for r in train})
   audit={'trainingRows':len(train),'distinctTrainingTargetDates':len(days),'status':'insufficient_training'}
   fitted[p,h]={'train':train,'audit':audit,'meanCorrection':None,'neighborFit':None}
   if len(days)<MIN_TRAIN_DAYS:continue
   residuals=[math.log(r['actual']/r['direct']) for r in train]
   if not all(math.isfinite(y) for y in residuals):audit['status']='invalid_training_residual';continue
   import numpy as np
   X=np.asarray([r['x'] for r in train],float)
   means=X.mean(axis=0); scales=X.std(axis=0); scales[scales==0]=1.
   Z=(X-means)/scales
   if not np.isfinite(Z).all():audit['status']='invalid_training_features';continue
   fitted[p,h].update(meanCorrection=sum(residuals)/len(residuals),neighborFit={'means':means.tolist(),'scales':scales.tolist(),'Z':Z,'residuals':residuals,'rows':train})
   audit.update(status='fitted',trainingTargetStart=days[0].isoformat(),trainingTargetEnd=days[-1].isoformat(),standardizationMean=means.tolist(),standardizationScale=scales.tolist())
 return fitted

def _predict(row,vec,fit):
 direct=row['direct']; out={'candidate':None,'meanAblation':None,'candidateFallbackReason':None,'meanFallbackReason':None,'neighbors':[]}
 audit=fit['audit']; train=fit['train']
 if direct is None:
  out.update(candidateFallbackReason='direct_prediction_missing',meanFallbackReason='direct_prediction_missing');return out
 if fit['meanCorrection'] is None:
  reason=audit['status'];out.update(candidate=direct,meanAblation=direct,candidateFallbackReason=reason,meanFallbackReason=reason);return out
 mean_corr=fit['meanCorrection']
 try:
  mean_pred=direct*math.exp(max(-CLIP,min(CLIP,mean_corr)))
  if not math.isfinite(mean_pred) or mean_pred<=0:raise ValueError('invalid mean correction')
  out['meanAblation']=mean_pred
 except (ValueError,OverflowError):
  out['meanAblation']=direct;out['meanFallbackReason']='invalid_mean_correction'
 if vec is None:
  out.update(candidate=direct,candidateFallbackReason='feature_unavailable');return out
 try:
  import numpy as np
  nf=fit['neighborFit']; q=(np.asarray(vec,float)-np.asarray(nf['means']))/np.asarray(nf['scales'])
  distances=np.sqrt(((nf['Z']-q)**2).sum(axis=1))
  if not np.isfinite(distances).all():raise ValueError('nonfinite distance')
  order=sorted(range(len(distances)),key=lambda i:(float(distances[i]),nf['rows'][i]['target']))[:NEIGHBORS]
  out['neighbors']=[{'target':nf['rows'][i]['target'].isoformat(),'origin':nf['rows'][i]['origin'].isoformat(),'distance':float(distances[i])} for i in order]
  if len(order)<NEIGHBORS:raise ValueError('fewer_than_20_neighbors')
  weights=[1/(float(distances[i])+.1) for i in order]; corrections=[nf['residuals'][i] for i in order]
  correction=sum(w*y for w,y in zip(weights,corrections))/sum(weights)
  if not math.isfinite(correction):raise ValueError('nonfinite_neighbor_correction')
  out['candidate']=direct*math.exp(max(-CLIP,min(CLIP,correction)))
 except Exception as e:
  out['candidate']=direct;out['candidateFallbackReason']=str(e) if str(e)=='fewer_than_20_neighbors' else 'neighbor_prediction_failed:'+type(e).__name__
 if out['meanAblation'] is None or not finite(out['meanAblation']) or out['meanAblation']<=0:
  out['meanAblation']=direct;out['meanFallbackReason']='invalid_mean_correction'
 for name in ('candidate','meanAblation'):
  if not finite(out[name]) or out[name]<=0:out[name]=direct;out[name+'FallbackReason']='invalid_candidate'
 return out

def _origin_dates():return [EVAL_START+dt.timedelta(days=i) for i in range(85)]
def build_forecast(prices,report,*,hashes=None):
 series_map=_series(prices); direct=_direct_map(report); fitted=_fit_models(direct,series_map); grid=[]
 for origin in _origin_dates():
  for p in PRODUCTS:
   vec,feat=state_features(series_map[p],origin)
   latest=max((d for d in series_map[p] if d<origin),default=None)
   persistence=series_map[p][latest][0] if latest and (origin-latest).days<=SOURCE_MAX_AGE else None
   persist_reason=None if persistence is not None else 'no_prior_price_or_stale_source'
   for h in HORIZONS:
    target=origin+dt.timedelta(days=h); row=direct.get((p,origin,target,h)); direct_value=row['direct'] if row else None
    base_row={'product':p,'origin':origin.isoformat(),'target':target.isoformat(),'horizon':h,'direct':direct_value,'persistence':persistence,'persistenceSourceDate':latest.isoformat() if persistence is not None else None,'persistenceMissingReason':persist_reason,'featureAvailable':vec is not None,'featureAudit':feat}
    pred=_predict({'direct':direct_value},vec,fitted[p,h]);base_row.update(pred)
    if row is None:base_row['directMissingReason']='direct_prediction_missing'
    grid.append(base_row)
 if len(grid)!=1275:raise AssertionError('frozen plan grid size mismatch')
 counts={m:{} for m in ('candidate','meanAblation','direct','persistence')}
 for r in grid:
  for method,reason in [('candidate','candidateFallbackReason'),('meanAblation','meanFallbackReason'),('direct','directMissingReason'),('persistence','persistenceMissingReason')]:
   if r.get(reason):counts[method][r[reason]]=counts[method].get(r[reason],0)+1
 return {'schemaVersion':1,'status':'answer_free_forecast_manifest','protocol':PROTOCOL,'inputs':hashes or {},'trainingAudit':{f'{p}/{h}':fitted[p,h]['audit'] for p in PRODUCTS for h in HORIZONS},'fallbackCounts':counts,'forecasts':grid}

def expected_grid():return {(p,o.isoformat(),h,(o+dt.timedelta(days=h)).isoformat()) for o in _origin_dates() for p in PRODUCTS for h in HORIZONS}
def _validate_grid(rows,series_map):
 if not isinstance(rows,list) or len(rows)!=1275:raise PilotError('forecast grid must contain 1275 rows')
 expected=expected_grid();seen=set()
 for r in rows:
  if not isinstance(r,dict):raise PilotError('forecast row must be object')
  if 'actual' in r:raise PilotError('forecast rows must be answer-free')
  try:o=dt.date.fromisoformat(r['origin']);t=dt.date.fromisoformat(r['target']);p=r['product'];h=r['horizon']
  except (KeyError,ValueError,TypeError):raise PilotError('malformed forecast identity')
  if not isinstance(p,str) or isinstance(h,bool) or not isinstance(h,int) or h not in HORIZONS:raise PilotError('invalid product/horizon identity')
  key=(p,o.isoformat(),h,t.isoformat())
  if key in seen:raise PilotError('duplicate grid identity')
  seen.add(key)
  if key not in expected or t!=o+dt.timedelta(days=h):raise PilotError('grid identity outside fixed plan')
  if r.get('direct') is None and r.get('directMissingReason')!='direct_prediction_missing':raise PilotError('null direct lacks explicit missing reason')
  if r.get('direct') is not None and r.get('directMissingReason') is not None:raise PilotError('direct missing reason contradicts numeric direct')
  for m,reason in [('candidate','candidateFallbackReason'),('meanAblation','meanFallbackReason'),('direct','directMissingReason'),('persistence','persistenceMissingReason')]:
   v=r.get(m); why=r.get(reason)
   if v is not None and (not finite(v) or v<=0):raise PilotError(f'invalid {m} value')
   if v is None and not why:raise PilotError(f'null {m} lacks reason')
   if m in ('candidate','meanAblation') and why and r.get('direct') is not None and v!=r['direct']:raise PilotError(f'{m} fallback must equal direct')
   if m in ('direct','persistence') and v is not None and why:raise PilotError(f'{m} value contradicts missing reason')
  if r.get('direct') is None and (r.get('candidate') is not None or r.get('meanAblation') is not None):raise PilotError('candidate must be null when direct is missing')
  if r.get('persistence') is not None:
   try:src=dt.date.fromisoformat(r['persistenceSourceDate'])
   except (KeyError,ValueError,TypeError):raise PilotError('persistence source date missing')
   if src>=o or src not in series_map[p] or not math.isclose(r['persistence'],series_map[p][src][0],rel_tol=1e-12,abs_tol=1e-12):raise PilotError('persistence source must be observed strictly before origin')
  elif r.get('persistenceSourceDate') is not None:raise PilotError('null persistence has source date')
 if seen!=expected:raise PilotError('forecast grid does not exactly cover plan')

def _metric(rows,field):
 if not rows:return {'mae':None,'mape':None}
 return {'mae':sum(abs(r['actual']-r[field]) for r in rows)/len(rows),'mape':100*sum(abs(r['actual']-r[field])/r['actual'] for r in rows)/len(rows)}
def _metrics_on_common(rows):return {m:_metric(rows,m) for m in ('candidate','meanAblation','direct','persistence')}
def _dm(rows,comparator,loss,bw):
 diff={dt.date.fromisoformat(r['target']):(abs(r['actual']-r['candidate'])-abs(r['actual']-r[comparator]) if loss=='mae' else 100*(abs(r['actual']-r['candidate'])-abs(r['actual']-r[comparator]))/r['actual']) for r in rows}
 return dh._dm(diff,bw)
def _holm(items):dh._holm(items)

def score(forecast,prices,forecast_bytes=None,prices_bytes=None):
 if forecast.get('status')!='answer_free_forecast_manifest' or forecast.get('protocol')!=PROTOCOL:raise PilotError('forecast protocol/status mismatch')
 if forecast_bytes is not None:
  check=dict(forecast); stated=check.pop('manifestSha256',None)
  if sha(canon(check))!=stated:raise PilotError('forecast manifest SHA mismatch')
 if prices_bytes is not None and forecast.get('inputs',{}).get('prices')!=sha(prices_bytes):raise PilotError('prices hash mismatch')
 series_map=_series(prices);grid=forecast.get('forecasts');_validate_grid(grid,series_map)
 actual_by={p:series_map[p] for p in PRODUCTS}; joined=[]; drop={'missingActual':0,'missingCandidate':0,'missingMeanAblation':0,'missingDirect':0,'missingPersistence':0}
 forecast_counts={m:0 for m in ('candidate','meanAblation','direct','persistence')}; scoreable_counts=forecast_counts.copy()
 for r in grid:
  for m in forecast_counts:
   if finite(r.get(m)) and r[m]>0:forecast_counts[m]+=1
  target=dt.date.fromisoformat(r['target']); actual=actual_by[r['product']].get(target)
  if actual is None or not finite(actual[0]) or actual[0]<=0:drop['missingActual']+=1;continue
  x={**r,'actual':actual[0]}
  for m in scoreable_counts:
   if finite(r.get(m)) and r[m]>0:scoreable_counts[m]+=1
   else:drop['missing'+m[0].upper()+m[1:]]+=1
  joined.append(x)
 common=[r for r in joined if all(finite(r.get(m)) and r[m]>0 for m in forecast_counts)]
 cells={};comp=[]
 for p in PRODUCTS:
  for h in HORIZONS:
   rows=[r for r in common if r['product']==p and r['horizon']==h]; cells[f'{p}/{h}']={'n':len(rows),'metrics':_metrics_on_common(rows)}
   for c in COMPARATORS:
    for loss in LOSSES:
     tests={str(b):_dm(rows,c,loss,b) for b in BANDWIDTHS}
     comp.append({'product':p,'horizon':h,'comparator':c,'loss':loss,'n':len(rows),'candidateMetric':cells[f'{p}/{h}']['metrics']['candidate'][loss],'comparatorMetric':cells[f'{p}/{h}']['metrics'][c][loss],'dm':tests})
 for b in BANDWIDTHS:_holm([x['dm'][str(b)] for x in comp])
 macro={};product_metrics={}
 for method in ('candidate','meanAblation','direct','persistence'):
  product_metrics[method]={}
  for p in PRODUCTS:
   groups=[cells[f'{p}/{h}']['metrics'][method] for h in HORIZONS]
   product_metrics[method][p]={key:(sum(g[key] for g in groups)/3 if all(g[key] is not None for g in groups) else None) for key in ('mae','mape')}
  macro[method]={key:(sum(product_metrics[method][p][key] for p in PRODUCTS)/5 if all(product_metrics[method][p][key] is not None for p in PRODUCTS) else None) for key in ('mae','mape')}
 deltas={p:{'mapeWorseningPctPointsVsDirect':(product_metrics['candidate'][p]['mape']-product_metrics['direct'][p]['mape'] if product_metrics['candidate'][p]['mape'] is not None and product_metrics['direct'][p]['mape'] is not None else None),'equalThreeHorizonMaeWorseningPctVsDirect':(100*(product_metrics['candidate'][p]['mae']-product_metrics['direct'][p]['mae'])/product_metrics['direct'][p]['mae'] if product_metrics['candidate'][p]['mae'] is not None and product_metrics['direct'][p]['mae'] not in (None,0) else None)} for p in PRODUCTS}
 gates={'all15CellsCommonPairsAtLeast30':all(cells[f'{p}/{h}']['n']>=30 for p in PRODUCTS for h in HORIZONS),'candidateCoverageAtLeastDirect':forecast_counts['candidate']>=forecast_counts['direct'],'candidateFiveProductMacroMapeAtMost10Pct':macro['candidate']['mape'] is not None and macro['candidate']['mape']<=10,'candidateMapeImprovesVsMeanByAtLeast0_5PctPoints':macro['candidate']['mape'] is not None and macro['meanAblation']['mape'] is not None and macro['candidate']['mape']<=macro['meanAblation']['mape']-.5,'candidateMapeImprovesVsDirectByAtLeast0_5PctPoints':macro['candidate']['mape'] is not None and macro['direct']['mape'] is not None and macro['candidate']['mape']<=macro['direct']['mape']-.5,'eachProductMapeWorseningAtMost1PctPoint':all(deltas[p]['mapeWorseningPctPointsVsDirect'] is not None and deltas[p]['mapeWorseningPctPointsVsDirect']<=1 for p in PRODUCTS),'eachProductEqualThreeHorizonMaeWorseningAtMost5Pct':all(deltas[p]['equalThreeHorizonMaeWorseningPctVsDirect'] is not None and deltas[p]['equalThreeHorizonMaeWorseningPctVsDirect']<=5 for p in PRODUCTS)};gates['expand']=all(gates.values())
 conclusions=[]
 for p in PRODUCTS:
  for h in HORIZONS:
   family=[x for x in comp if x['product']==p and x['horizon']==h]; tests=[x['dm']['9'] for x in family]
   enough=all(t['status']=='tested' for t in tests);lower=all(x['candidateMetric'] is not None and x['comparatorMetric'] is not None and x['candidateMetric']<x['comparatorMetric'] for x in family)
   improve=all(t.get('holmSignificant') and t.get('meanLossDifference') is not None and t['meanLossDifference']<0 for t in tests); adverse=any(t.get('holmSignificant') and t.get('meanLossDifference') is not None and t['meanLossDifference']>0 for t in tests)
   conclusion='worsening_or_mixed' if adverse else 'improvement' if enough and lower and improve else 'inference_unavailable' if not enough else 'inconclusive'
   conclusions.append({'product':p,'horizon':h,'conclusion':conclusion,'allPairedLossesLower':lower,'allHac9HolmTestsImprove':improve,'inferenceAvailable':enough,'inferenceIncomplete':not enough,'significantAdverseHac9Tests':sum(bool(t.get('holmSignificant') and t.get('meanLossDifference') is not None and t['meanLossDifference']>0) for t in tests)})
 return {'schemaVersion':1,'status':'scored','forecastManifestSha256':forecast.get('manifestSha256'),'fourWayCommonRows':len(common),'byProductHorizon':cells,'macroMetrics':macro,'byProductEqualThreeHorizonMetrics':product_metrics,'productGateDeltas':deltas,'gates':gates,'conclusions':conclusions,'coverage':{'plannedRows':1275,'forecastValidRows':forecast_counts,'forecastFractionOfPlan':{m:n/1275 for m,n in forecast_counts.items()},'rowsWithActual':len(joined),'scoreableValidRows':scoreable_counts,'scoreableFractionOfRowsWithActual':{m:(n/len(joined) if joined else 0) for m,n in scoreable_counts.items()},'fourWayCommonRows':len(common),'dropReasons':drop},'dm':{'familySize':FAMILY,'comparisons':comp,'bandwidths':list(BANDWIDTHS)},'sampleLimits':{'cellsBelowThirty':[k for k,v in cells.items() if v['n']<30],'minimumPairs':30}}

def _write(path,raw):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(raw)
def main(argv=None):
 ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
 f=sp.add_parser('forecast');f.add_argument('--prices',type=Path,default=PRICE_PATH);f.add_argument('--direct-report',type=Path,default=DIRECT_PATH);f.add_argument('--output',type=Path,required=True)
 s=sp.add_parser('score');s.add_argument('--forecast-manifest',type=Path,required=True);s.add_argument('--prices',type=Path,default=PRICE_PATH);s.add_argument('--output',type=Path,required=True)
 a=ap.parse_args(argv)
 try:
  if a.cmd=='forecast':
   prices,pb=load(a.prices);report,rb=load(a.direct_report)
   if sha(pb)!=PRICE_SHA or sha(rb)!=DIRECT_SHA:raise PilotError('pinned input hash mismatch')
   hashes={'prices':sha(pb),'directReport':sha(rb),'code':sha(Path(__file__).read_bytes())}
   result=build_forecast(prices,report,hashes=hashes);result['manifestSha256']=sha(canon(result));_write(a.output,canon(result))
  else:
   forecast,fb=load(a.forecast_manifest);prices,pb=load(a.prices)
   if sha(pb)!=PRICE_SHA:raise PilotError('pinned prices hash mismatch')
   report_raw=DIRECT_PATH.read_bytes()
   if sha(report_raw)!=DIRECT_SHA or forecast.get('inputs',{}).get('directReport')!=DIRECT_SHA:raise PilotError('pinned direct report hash mismatch')
   if forecast.get('inputs',{}).get('code')!=sha(Path(__file__).read_bytes()):raise PilotError('forecast code hash mismatch')
   result=score(forecast,prices,fb,pb);result['inputHashes']={'prices':sha(pb),'directReport':sha(report_raw),'code':forecast['inputs']['code']};_write(a.output,canon(result))
 except (PilotError,ValueError,OSError,KeyError) as e:ap.error(str(e))
 return 0
if __name__=='__main__':raise SystemExit(main())
