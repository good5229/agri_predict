"""Frozen paired evaluation for PERIOD_POOLED_DIRECT_LGBM_V5."""
from __future__ import annotations
import datetime as dt, json, math
from pathlib import Path
from typing import Any
import numpy as np
from . import period_proxy_experiment as proxy
from . import period_pooled_lgbm as m
from . import period_mape10_score as audited

def _bholm(pvalues):
 n=len(pvalues);order=sorted(range(n),key=lambda i:pvalues[i]);adj=[1.]*n;running=0.
 for rank,i in enumerate(order):
  running=max(running,min(1.,(n-rank)*float(pvalues[i])));adj[i]=running
 return adj
def _metric(actual,pred):
 a=np.asarray(actual,float);p=np.asarray(pred,float)
 if len(a)==0 or len(a)!=len(p):raise ValueError('empty or unpaired metrics')
 return {'n':int(len(a)),'mae':float(np.mean(np.abs(p-a))),'mape':float(100*np.mean(np.abs((p-a)/a)))}
def _dm_pairs(pairs, loss, lag):
    result=audited.hac(pairs,loss,lag)
    if result.get('status')!='tested':
        return {'n':result.get('n',0),'distinctTargets':result.get('distinctTargets'),
                'mean':result.get('meanLossDifference'),'se':None,'p':None,
                'status':result.get('status')}
    return {'n':result['n'],'distinctTargets':result['distinctTargets'],
            'mean':result['meanLossDifference'],
            'se':math.sqrt(result['hacVariance']/result['n']),
            'p':result['p'],'status':'tested'}

def _holm_slots(items):
    """V3 Holm: family size includes unsupported hypotheses, whose p stays null."""
    ordered=sorted(((float(x['p']),i) for i,x in enumerate(items) if x.get('p') is not None),key=lambda q:q[0])
    adjusted={};running=0.0
    for rank,(p,i) in enumerate(ordered):
        running=max(running,min(1.0,(len(items)-rank)*p));adjusted[i]=running
    return {x['name']:adjusted.get(i) for i,x in enumerate(items)}
def _macro(rows,method):
 cell={}
 for c in m.CROPS:
  for h in m.HORIZONS:
   rr=[r for r in rows if r['crop']==c and int(r['horizon'])==h and r.get('actual') is not None and r.get(method) is not None]
   cell[f'{c}:{h}']=_metric([r['actual'] for r in rr],[r[method] for r in rr])
 return {'cells':cell,'mae8':float(np.mean([cell[f'{c}:{h}']['mae'] for c in m.CROPS for h in m.HORIZONS])),
         'mape8':float(np.mean([cell[f'{c}:{h}']['mape'] for c in m.CROPS for h in m.HORIZONS]))}
def _join(split,out):
 forecast_meta=m.load_stage('develop' if split=='develop' else 'forecast',out)['forecastArtifact'];pred=m.read_rows(forecast_meta)
 series,catalog,targets=m._sources();v3={}
 raw=m._v3_rows('forecasts.parquet' if split=='develop' else 'forecasts_test.parquet');sel=json.loads((m.V3_OUT/'selection.json').read_text())
 old=json.loads(m.OLD_REPORT.read_text(encoding='utf8')) if split=='develop' else None
 old_b0={}
 if old is not None:
  # Recreate the old frozen per-cell development-selected baseline on each
  # development origin from the source series; the old report only stores test forecasts.
  for crop in m.CROPS:
   for h in m.HORIZONS:
    cell=old['crops'][crop][str(h)]
    method=cell.get('baselineDevelopmentSelection') or cell.get('selectedSimpleBaseline')
    maps=proxy.baseline_predictions(series[crop],targets[crop],h,dt.date(2022,1,1),dt.date(2023,12,31))
    for (start,end),item in maps[method].items():
     key=(crop,h,start.isoformat(),end.isoformat(),item['origin'])
     old_b0[key]=float(item['predicted'])
 for r in raw:
  key=(r['crop'],int(r['horizon']),r['targetStart'],r['targetEnd'],r['origin']);f=r.get('forecasts',{})
  if split=='develop':
   p=(f.get(sel['P'][r['crop']]['method']) or {}).get('predicted')
  else:p=(f.get('P') or {}).get('predicted')
  key=(r['crop'],int(r['horizon']),r['targetStart'],r['targetEnd'],r['origin'])
  b=old_b0.get(key) if split=='develop' else (f.get('B0') or {}).get('predicted')
  v3[key]={'P':p,'B0':b,'unavailable':bool(r.get('policyUnavailable',False))}
 joined=[]
 for r in pred:
  key=(r['crop'],int(r['horizon']),r['targetStart'],r['targetEnd'],r['origin'])
  target=(dt.date.fromisoformat(r['targetStart']),dt.date.fromisoformat(r['targetEnd']))
  actual=targets[r['crop']].get(target,{}).get('actual')
  base=v3.get(key,{})
  predictions=r.get('predictions',{})
  outrow={**r,'actual':actual,'P':base.get('P'),'B0':base.get('B0'),'targetPeriodKey':m._periodkey(target),'year':target[0].year}
  for name,item in predictions.items():
   if item is None:continue
   outrow[name]=item.get('predicted');outrow[f'{name}_pred_z']=item.get('pred_z');outrow[f'{name}_pred_z_raw']=item.get('pred_z_raw');outrow[f'{name}_pred_z_nonfinite']=bool(item.get('pred_z_raw_nonfinite',False));outrow[f'{name}_status']=item.get('status');outrow[f'{name}_fallbackReason']=item.get('fallbackReason')
  joined.append(outrow)
 return joined

def _verify_fit_ledger(out,selection):
 ledger_path=Path(out)/'fit-ledger.json';ledger=json.loads(ledger_path.read_text(encoding='utf8'))
 if len(ledger)>600:raise ValueError('fit ledger exceeds the frozen 600-attempt cap')
 if any(x.get('status')=='started' for x in ledger):raise ValueError('fit ledger contains an unfinished model attempt')
 for x in ledger:
  if x.get('status')=='complete':
   p=Path(x['modelPath'])
   if not p.is_file() or m.digest(p)!=x.get('modelSha256'):raise ValueError(f"saved model digest mismatch: {x['modelPath']}")
   fitid=x.get('fitIdentity',{})
   if m.sha_json(fitid)!=x.get('fitIdentitySha256'):raise ValueError('fit ledger identity digest mismatch')
 # Retries can leave failed attempts, but each required origin/config must have one verified completed fit.
 rows_dev=m._v3_rows('forecasts.parquet');rows_test=m._v3_rows('forecasts_test.parquet')
 dev_origins={r['origin'] for r in rows_dev};test_origins={r['origin'] for r in rows_test if not r.get('policyUnavailable') and (r.get('forecasts',{}).get('P') or {}).get('predicted') is not None and (r.get('forecasts',{}).get('B0') or {}).get('predicted') is not None}
 chosen=selection['chosenConfig']
 expected={('develop',o,k,c) for o in dev_origins for k,c in [('G','A'),('G','B'),('G','C'),('L','L')]}
 expected|={('test',o,k,c) for o in test_origins for k,c in [('G',chosen),('L','L')]}
 complete={(x['split'],x['origin'],x['kind'],x['config']) for x in ledger if x.get('status')=='complete'}
 if not expected.issubset(complete):raise ValueError('fit ledger lacks a successful fit for a scheduled origin/config')
 return ledger

def score_development(out=m.OUT):
 if m._stage_path(out,'development-score').exists():
  cached=m.load_stage('development-score',out);p=Path(out)/'development-selection.json'
  if not p.is_file() or m.digest(p)!=cached.get('selectionSha256'):raise ValueError('completed development selection/checkpoint differs')
  return cached
 m.load_stage('develop',out);rows=_join('develop',out)
 allkeys={(r['crop'],int(r['horizon']),r['targetStart'],r['targetEnd']) for r in rows}
 if len(allkeys)!=len(rows):raise ValueError('duplicate development target row')
 names=['P','B0','G-A','G-B','G-C','L-L']
 if len(rows)!=576:raise ValueError(f'development artifact must preserve 576 planned rows, found {len(rows)}')
 observed=sum(r.get('actual') is not None for r in rows)
 if observed!=556:raise ValueError(f'development scoring support must be 556/576, found {observed}')
 for c in m.CROPS:
  for h in m.HORIZONS:
   cell=[r for r in rows if r['crop']==c and int(r['horizon'])==h and r.get('actual') is not None]
   if not cell:raise ValueError(f'no observed development labels for {c}/{h}')
   keys={(r['targetStart'],r['targetEnd'],r['origin']) for r in cell}
   for method in names:
    if any(r.get(method) is None or not math.isfinite(float(r[method])) for r in cell):raise ValueError(f'incomplete paired development coverage: {method}/{c}/{h}')
    if {(r['targetStart'],r['targetEnd'],r['origin']) for r in cell if r.get(method) is not None}!=keys:raise ValueError('development paired key mismatch')
 stats={k:_macro(rows,k) for k in names}
 complete=[k for k in ('G-A','G-B','G-C') if all(stats[k]['cells'][f'{c}:{h}']['n']>0 for c in m.CROPS for h in m.HORIZONS)]
 if len(complete)!=3:raise ValueError('incomplete development coverage blocks config selection')
 guard=stats['P']['mae8']*1.03;protected=[k for k in complete if stats[k]['mae8']<=guard]
 pool=protected or complete;order=['G-A','G-B','G-C'];chosen=min(pool,key=lambda k:(stats[k]['mape8'],order.index(k)))
 selection={'runId':m.RUN_ID,'runIdentity':m.run_identity(),'chosenConfig':chosen[-1],'chosenMethod':chosen,
  'developmentMetrics':stats,'maeGuard':guard,'guardPassed':bool(protected),'candidateEligibility':complete,
  'baselineCaveat':'Development P replays V3 saved crop-specific R/W selections; B0 replays the prior report selectedSimpleBaseline, both retrospective development selections.',
  'selectionRule':'equal 8-cell MAPE among candidates with MAE8 <= Pdev*1.03; otherwise lowest MAPE8; A/B/C tie order',
  'selectionHashInputs':{'protocolSha256':m.PROTOCOL_SHA,'codeSha256':m.code_hash(),'developmentForecastSha256':m.load_stage('develop',out)['forecastArtifact']['sha256']}}
 selection['developmentNativeFallback']={name:{
  'native':sum(r.get(f'{name}_status')=='ok' for r in rows),
  'fallbackP':sum(r.get(f'{name}_status')=='technical_fallback_P' for r in rows),
  'fallbackReasons':{reason:sum(r.get(f'{name}_fallbackReason')==reason for r in rows)
                     for reason in sorted({r.get(f'{name}_fallbackReason') for r in rows if r.get(f'{name}_fallbackReason')})}}
  for name in ('G-A','G-B','G-C','L-L')}
 p=Path(out)/'development-selection.json';p.write_text(json.dumps(selection,indent=2,allow_nan=False)+'\n',encoding='utf8')
 doc=m.save_stage('development-score',{'stage':'development-score','selectionPath':str(p),'selectionSha256':m.digest(p),'metrics':stats,'chosenConfig':chosen},out)
 # The test-stage gate expects identity and selected policy in this durable object.
 selection['runIdentity']=m.run_identity();p.write_text(json.dumps(selection,indent=2,allow_nan=False)+'\n',encoding='utf8')
 return doc

def score(out=m.OUT):
 if m._stage_path(out,'score').exists():
  cached=m.load_stage('score',out)
  for name,key in (('score-report.json','scoreReportSha256'),('decision.md','decisionSha256')):
   p=Path(out)/name
   if not p.is_file() or m.digest(p)!=cached.get(key):raise ValueError(f'completed score artifact differs: {name}')
  return cached
 m.load_stage('forecast',out);rows=_join('test',out);all_rows=rows
 selection_path=Path(out)/'development-selection.json';selection=json.loads(selection_path.read_text(encoding='utf8'))
 forecast_stage=m.load_stage('forecast',out)
 if forecast_stage.get('selectionSha256')!=m.digest(selection_path):raise ValueError('test forecast used a different frozen development selection')
 planned_rows=len(rows);unavailable=[r for r in rows if r.get('policyUnavailable')]
 base_rows=m._v3_rows('forecasts_test.parquet')
 expected={(r['crop'],int(r['horizon']),r['targetStart'],r['targetEnd'],r['origin']) for r in base_rows
           if not r.get('policyUnavailable') and (r.get('forecasts',{}).get('P') or {}).get('predicted') is not None
           and (r.get('forecasts',{}).get('B0') or {}).get('predicted') is not None}
 actual_keys={(r['crop'],int(r['horizon']),r['targetStart'],r['targetEnd'],r['origin']) for r in rows if not r.get('policyUnavailable')}
 if expected!=actual_keys:raise ValueError('forecast key set differs from exact frozen P/B0 paired support')
 rows=[r for r in rows if not r.get('policyUnavailable')]
 # Assert all shared P/B0 keys and full planned row universe are retained.
 if len(rows)!=780:raise ValueError(f'expected 780 policy-available forecast keys, found {len(rows)}')
 if planned_rows!=792 or len(unavailable)!=12:raise ValueError('planned/available/unavailable forecast rows differ from 792/780/12')
 if any(r.get('P') is None or r.get('B0') is None for r in rows):raise ValueError('P/B0 coverage loss blocks paired score')
 if any(r.get('actual') is None for r in rows):raise ValueError('invalid or absent actuals within frozen paired set')
 gname='G-'+selection['chosenConfig']
 if any(r.get(gname) is None or r.get('L-L') is None for r in rows):raise ValueError('G/L test coverage loss blocks scoring')
 methods=['P','B0',gname,'L-L'];metrics={x:_macro(rows,x) for x in methods}
 # Comparisons are same-key paired cell loss differences; HAC indexed by dekad order.
 comparisons={};families={'main':[],'structure':[]}
 for candidate,baseline,family in ((gname,'P','main'),(gname,'B0','main'),(gname,'L-L','structure')):
  for c in m.CROPS:
   for h in m.HORIZONS:
    rr=sorted([r for r in rows if r['crop']==c and int(r['horizon'])==h],key=lambda r:r['targetStart'])
    for loss in ('AE','APE'):
     a=np.array([abs(r[candidate]-r['actual']) if loss=='AE' else abs((r[candidate]-r['actual'])/r['actual']) for r in rr]);b=np.array([abs(r[baseline]-r['actual']) if loss=='AE' else abs((r[baseline]-r['actual'])/r['actual']) for r in rr]);d=a-b
     pairs=[((dt.date.fromisoformat(r['targetStart']),dt.date.fromisoformat(r['targetEnd'])),
             float(r['actual']),float(r[candidate]),float(r[baseline])) for r in rr]
     entry={'n':len(d),'meanCandidateMinusBaseline':float(d.mean()),'hac':{str(l):_dm_pairs(pairs,loss.lower(),l) for l in (3,6)}}
     comparisons[f'{candidate}-{baseline}:{c}:{h}:{loss}']=entry;families[family].append((f'{candidate}-{baseline}:{c}:{h}:{loss}',entry))
 if len(families['main'])!=32 or len(families['structure'])!=16:raise ValueError('frozen HAC/Holm family sizes must be 32 and 16')
 holm={}
 for family,items in families.items():
  for lag in (3,6):
   slots=[{'name':name,'p':e['hac'][str(lag)]['p']} for name,e in items]
   holm[f'{family}:HAC{lag}']=_holm_slots(slots)
 years={}
 for y in (2024,2025,2026):
  subset=[r for r in rows if r['year']==y]
  if subset:years[str(y)]={x:_macro(subset,x) for x in methods}
 # Reuse the V3 audited circular calendar-block procedure, including all 99
 # dekads and the 12 policy-unavailable rows (which carry no candidate forecast).
 bootstrap_rows=[]
 for r in all_rows:
  fc={}
  for name in ('P','B0',gname,'L-L'):
   value=r.get(name)
   if value is not None:fc[name]={'predicted':value}
  bootstrap_rows.append({**r,'forecasts':fc})
 blocks={}
 for length in (6,12):
  paired_ci={}
  for base in ('P','B0','L-L'):
   result=audited.bootstrap(bootstrap_rows,(gname,base),length,reps=2000)
   paired_ci[f'{gname}-{base}']={
    'mape8':result['mape8']['ci95'] if result.get('status')=='tested' else None,
    'mae8':result['mae8']['ci95'] if result.get('status')=='tested' else None,
    'status':result.get('status')}
  blocks[str(length)]={'blockLengthDekads':length,'seed':0,'replicates':2000,
   'sameIndicesAcrossCells':True,'calendarGapsRetained':True,
   'pairedDifferenceCI95':paired_ci}
 # Mechanisms and supports
 diagnostics={'rows':len(rows),'uniqueCropTarget':len({(r['crop'],r['targetStart'],r['targetEnd']) for r in rows}),'duplicateRows':len(rows)-len({(r['crop'],r['targetStart'],r['targetEnd']) for r in rows}), 'cells':{},'response':{},'jump':{}}
 for c in m.CROPS:
  for h in m.HORIZONS:
   rr=[r for r in rows if r['crop']==c and int(r['horizon'])==h];cell=f'{c}:{h}';diagnostics['cells'][cell]={'n':len(rr),'G':metrics[gname]['cells'][cell],'P':metrics['P']['cells'][cell],'B0':metrics['B0']['cells'][cell],'L':metrics['L-L']['cells'][cell]}
   for name in methods:
    z=np.array([math.log(r[name]/r['p_ref']) for r in rr if r[name] and r.get('p_ref')]);actualz=np.array([math.log(r['actual']/r['p_ref']) for r in rr if r.get('actual') and r.get('p_ref')])
    actz=np.array([math.log(r['actual']/r['p_ref']) for r in rr if r.get('actual') and r.get('p_ref')])
    q20,q80=np.quantile(actz,[.2,.8],method='linear') if len(actz) else (None,None)
    predz=np.array([r.get(f'{name}_pred_z') for r in rr if isinstance(r.get(f'{name}_pred_z'),(int,float)) and math.isfinite(float(r[f'{name}_pred_z']))]) if name not in ('P','B0') else np.array([math.log(r[name]/r['p_ref']) for r in rr if r.get(name) and r.get('p_ref')])
    low=[r for r in rr if math.log(r['actual']/r['p_ref'])<=q20] if q20 is not None else []
    high=[r for r in rr if math.log(r['actual']/r['p_ref'])>=q80] if q80 is not None else []
    def bias(sub):
     if not sub:return {'n':0,'logBias':None,'priceBias':None}
     return {'n':len(sub),'logBias':float(np.mean([math.log(r[name]/r['p_ref'])-math.log(r['actual']/r['p_ref']) for r in sub])),'priceBias':float(np.mean([r[name]-r['actual'] for r in sub]))}
    diagnostics['response'].setdefault(cell,{})[name]={'n':len(z),'emittedLogRatioMean':float(z.mean()),'emittedLogRatioStd':float(z.std(ddof=0)),'emittedLogRatioQuantiles':{str(q):float(np.quantile(z,q,method='linear')) for q in (0,.1,.25,.5,.75,.9,1)},'emittedPositiveShare':float(np.mean(z>0)),'emittedNegativeShare':float(np.mean(z<0)),'rawPredZN':len(predz),'rawPredZNonfiniteCount':sum(bool(r.get(f'{name}_pred_z_nonfinite')) for r in rr),'rawPredZMean':float(predz.mean()) if len(predz) else None,'rawPredZStd':float(predz.std(ddof=0)) if len(predz) else None,'rawPredZQuantiles':{str(q):float(np.quantile(predz,q,method='linear')) for q in (0,.1,.25,.5,.75,.9,1)} if len(predz) else None,'actualLogRatioN':len(actualz),'actualLogRatioMean':float(actualz.mean()) if len(actualz) else None,'actualLogRatioStd':float(actualz.std(ddof=0)) if len(actualz) else None,'actualLogRatioQuantiles':{str(q):float(np.quantile(actualz,q,method='linear')) for q in (0,.1,.2,.25,.5,.75,.8,.9,1)} if len(actualz) else None,'actualLogRatioQ20':float(q20) if q20 is not None else None,'actualLogRatioQ80':float(q80) if q80 is not None else None,'lowActualTailBias':bias(low),'highActualTailBias':bias(high)}
   jump=[r for r in rr if abs(r['actual']/r['p_ref']-1)>=.2];normal=[r for r in rr if r not in jump]
   def jump_metrics(ss):
    result={'n':len(ss)}
    for method in methods:
     if ss:
      ae=np.array([abs(x[method]-x['actual']) for x in ss]);ape=np.array([abs((x[method]-x['actual'])/x['actual']) for x in ss])
      all_ae=np.array([abs(x[method]-x['actual']) for x in rr]);all_ape=np.array([abs((x[method]-x['actual'])/x['actual']) for x in rr])
      result[method]={'mae':float(ae.mean()),'mape':float(ape.mean()*100),'aeTotal':float(ae.sum()),'apeTotal':float(ape.sum()),'aeShareCell':float(ae.sum()/all_ae.sum()) if all_ae.sum() else None,'apeShareCell':float(ape.sum()/all_ape.sum()) if all_ape.sum() else None}
     else:result[method]=None
    return result
   diagnostics['jump'][cell]={'threshold':'abs(actual/p_ref-1)>=0.20','jump':jump_metrics(jump),'nonjump':jump_metrics(normal)}
 crop_contributions={}
 for method in methods:
  crop_contributions[method]={c:{'maeContribution':float(sum(metrics[method]['cells'][f'{c}:{h}']['mae'] for h in m.HORIZONS)/8),
   'maeDifferenceContributionVsP':float(sum(metrics[method]['cells'][f'{c}:{h}']['mae']-metrics['P']['cells'][f'{c}:{h}']['mae'] for h in m.HORIZONS)/8)} for c in m.CROPS}
 fit_ledger=_verify_fit_ledger(out,selection)
 fallback={}
 for name in ('G-'+selection['chosenConfig'],'L-L'):
  statuses=[r.get(f'{name}_status') for r in rows]
  fallback[name]={'native':sum(s=='ok' for s in statuses),'fallbackP':sum(s=='technical_fallback_P' for s in statuses),'nativeFraction':float(sum(s=='ok' for s in statuses)/len(rows))}
 support={'byHorizon':{str(h):len({(r['crop'],r['targetStart'],r['targetEnd']) for r in rows if int(r['horizon'])==h}) for h in m.HORIZONS},'uniqueTargetPeriodsByHorizon':{str(h):len({r['targetPeriodKey'] for r in rows if int(r['horizon'])==h}) for h in m.HORIZONS},'uniqueDekads':len({r['targetPeriodKey'] for r in rows})}
 if support['uniqueTargetPeriodsByHorizon']!={'2':98,'3':97}:raise ValueError(f'valid scored period counts differ from 98/97: {support}')
 bootstrap_conf={}
 for length,result in blocks.items():
  bootstrap_conf[length]={}
  for base in ('P','B0','L-L'):
   ci=result['pairedDifferenceCI95'][f'{gname}-{base}']
   bootstrap_conf[length][base]={'mapeDifferenceCI95':ci['mape8'],'maeDifferenceCI95':ci['mae8'],'status':ci['status']}
 inference={'pairedBlockBootstrap':bootstrap_conf,
  'interpretation':'Cellwise HAC/Holm and equal-cell paired block inference answer different questions; non-significance is not equivalence.'}
 observed=(metrics[gname]['mape8']<metrics['P']['mape8'] and metrics[gname]['mae8']<=metrics['P']['mae8'] and len(rows)==780)
 classifications={'observedImprovementVsP':observed,'additionalB0MaeProtection':metrics[gname]['mae8']<=221.2328,
  'tenPercentChallenge':metrics[gname]['mape8']<=10.,'cellsAtOrBelowTenPercent':sum(metrics[gname]['cells'][f'{c}:{h}']['mape']<=10 for c in m.CROPS for h in m.HORIZONS),
  'statisticalEvidence':'See signed per-cell HAC/Holm tests and paired 6/12-dekad block-bootstrap intervals; unresolved or non-significant evidence is inconclusive, not equivalence.',
  'operationalAdoption':'not evaluated; this experiment does not replace production policy'}
 report={'runId':m.RUN_ID,'metrics':metrics,'comparisons':comparisons,'holmAdjustedP':holm,'bootstrap':blocks,'inference':inference,'classifications':classifications,'yearly':years,'diagnostics':diagnostics,'cropMaeContributions':crop_contributions,'fitCount':len([x for x in fit_ledger if x.get('status')=='complete']),'fitAttempts':len(fit_ledger),'nativeFallback':fallback,'scoredSupport':support,'selection':selection,'coverage':{'pairedKeys':len(rows),'plannedRows':planned_rows,'unavailableRows':len(unavailable),'allGAndLIssued':True}}
 path=Path(out)/'score-report.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf8')
 p=metrics[gname];delta=metrics[gname]['mape8']-metrics['P']['mape8'];maechange=metrics[gname]['mae8']-metrics['P']['mae8']
 korean=f"""# PERIOD_POOLED_DIRECT_LGBM_V5 결과

P MAPE8 {metrics['P']['mape8']:.4f}% → G-{selection['chosenConfig']} MAPE8 {p['mape8']:.4f}% ({delta:+.4f}%p). 10% 목표까지 {p['mape8']-10:.4f}%p입니다. G MAE8 {p['mae8']:.4f}원/kg (P 대비 {maechange:+.4f}), B0 MAE8 {metrics['B0']['mae8']:.4f}원/kg입니다. 개발 선택 MAE 보호 가드는 {'통과' if selection['guardPassed'] else '미통과'}했습니다. 개발 P와 B0는 이전 기간에 선택된 규칙을 재생했으므로 회고 선택 편향이 있습니다.

공통 정책 예측 키는 {len(rows)}/780이고 미가용 12행을 포함한 계획표는 792행입니다. 평가 지원은 2순 98개, 3순 97개 고유 목표기간입니다. 실제 목표순은 겹치므로 780행을 독립 표본 780개로 해석하지 않습니다. 8셀 중 MAPE 10% 이하는 {classifications['cellsAtOrBelowTenPercent']}개입니다.

판정은 각각 따로 봅니다. P 대비 관측 개선은 {classifications['observedImprovementVsP']}이며, B0 MAE 보호는 {classifications['additionalB0MaeProtection']}, 10% 도전 목표는 {classifications['tenPercentChallenge']}, 통계적 근거는 “{classifications['statisticalEvidence']}”입니다. 통계 가족은 32개 주가설 및 별도 16개 기전 가설에 HAC3/HAC6와 Holm을 적용했습니다. 동일 99개 달력 순을 유지한 6/12순 블록 구간도 함께 보고합니다. 비유의 결과를 동등성 증명으로 해석하지 않습니다.

선택 G 설정은 개발구간의 8셀 MAPE와 P 개발 MAE×1.03 보호 가드에 따라 {selection['chosenMethod']}로 동결했습니다. 실제 적합 {report['fitCount']}회, 총 시도 {report['fitAttempts']}회였고 하드 상한은 600회입니다. 시험에서 선택 G의 native 발행 비율은 {fallback[gname]['nativeFraction']:.4f}입니다. 연도×셀 손실, 품목별 MAE 기여, 급변·비급변 손실, 셀별 원시 로그 예측과 발행가격 반응은 `score-report.json`을 보십시오.

가격은 고정 일별 스냅숏에서 만든 가상 순 평균이며 공식 거래량가중 순가격이 아닙니다. 과거 최초 공개판을 복원하지 못했고, 2024–2026은 반복 검토된 후향 구간입니다. G−L은 손실·규제 차이가 섞여 순수 비선형 효과가 아닙니다. 문헌의 전이 한계는 [reference-comparison.md](reference-comparison.md), 고정 절차는 [protocol.md](protocol.md)에 있습니다. 실행 결과만으로 운영정책을 교체하지 않습니다.
"""
 (Path(out)/'decision.md').write_text(korean,encoding='utf8')
 return m.save_stage('score',{'stage':'score','scoreReportSha256':m.digest(path),'decisionSha256':m.digest(Path(out)/'decision.md'),'metrics':metrics,'pairedKeys':len(rows)},out)
