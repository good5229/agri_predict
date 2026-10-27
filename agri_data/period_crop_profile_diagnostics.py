"""Read-only descriptive diagnostics for the completed pooled V5 run."""
from __future__ import annotations
import csv, datetime as dt, hashlib, json, math
from pathlib import Path
from itertools import combinations
import numpy as np
from scipy.stats import spearmanr

from . import period_proxy_experiment as proxy
from . import period_pooled_lgbm as v5

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/experiments/period-crop-profile-diagnostics-v2-post-v5'
REQUEST=OUT/'request.md'; PROTOCOL=OUT/'diagnostic_protocol.md'
PINS={'forecasts.parquet':'1a9847d08121aee8d3d808961d116bf5abf0151f74957a1b49639e1d3c9ac0c7',
      'score-report.json':'54dbad25c91fe5463716ea8c2cbb53cc55d196adc72f0495f0facaa50bfef8ab',
      'prices.json':proxy.SOURCE_SHA}
V3=v5.V3_OUT; V5=v5.OUT
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def period_index(day): return (day.year*36+(day.month-1)*3+(0 if day.day<=10 else 1 if day.day<=20 else 2))
def period_key(row): return (row['crop'],int(row['horizon']),row['origin'],row['targetStart'],row['targetEnd'])
def load_source(path=proxy.SOURCE):
 raw=json.loads(Path(path).read_text(encoding='utf8')); return raw,proxy.series_from_raw(raw)
def stats(vals):
 x=np.asarray([float(v) for v in vals if v is not None and math.isfinite(float(v))],dtype=float)
 if not len(x): return {'n':0}
 q=np.quantile(x,[.05,.1,.25,.5,.75,.9,.95]); return {'n':len(x),'mean':float(x.mean()),'sd':float(x.std()),'cv':float(x.std()/abs(x.mean())) if x.mean() else None,'min':float(x.min()),'max':float(x.max()),'p05':q[0],'p10':q[1],'p25':q[2],'median':q[3],'p75':q[4],'p90':q[5],'p95':q[6],'iqr':float(q[4]-q[2])}
def adjacent_returns(series):
 days=sorted(series); return [(d,(series[d][0]/series[p][0]-1),math.log(series[d][0]/series[p][0])) for p,d in zip(days,days[1:]) if (d-p).days==1 and series[d][0]>0 and series[p][0]>0]
def write_csv(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def read_frame(path):
 import pandas as pd
 return pd.read_parquet(path)
def verify_inputs():
 v5._check_pins()
 paths={'forecasts.parquet':V5/'forecasts.parquet','score-report.json':V5/'score-report.json','prices.json':proxy.SOURCE}
 for n,p in paths.items():
  if not p.is_file() or sha(p)!=PINS[n]: raise ValueError('blocked_integrity: pinned input mismatch '+str(p))
 for name,digest in v5.V3_FILES.items():
  if sha(V3/name)!=digest: raise ValueError('blocked_integrity: V3 input mismatch '+name)
 manifest={n:{'path':str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),'sha256':sha(p)} for n,p in paths.items()}
 manifest.update({'V3:'+n:{'path':str((V3/n).relative_to(ROOT)),'sha256':sha(V3/n)} for n in v5.V3_FILES})
 for name in ('development-forecasts.parquet','development-selection.json','fit-ledger.json','development-score.json','reference-comparison.md','resolved-config.json','forecast.json','develop.json','preflight.json','score.json'):
  p=V5/name
  if not p.is_file(): raise ValueError('blocked_integrity: required V5 artifact missing '+name)
  manifest['V5:'+name]={'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
 for p in (REQUEST,PROTOCOL,v5.PROTOCOL):
  if not p.is_file(): raise ValueError('required protocol/request missing '+str(p))
  manifest[str(p.relative_to(ROOT))]={'path':str(p.relative_to(ROOT)),'sha256':sha(p)}
 for stage in ('preflight','develop','development-score','forecast','score'):
  doc=v5.load_stage(stage,V5);checkpoint=v5._stage_path(V5,stage)
  manifest['V5-checkpoint:'+stage]={'path':str(checkpoint.relative_to(ROOT)),'sha256':sha(checkpoint)}
  if stage in ('develop','forecast'):
   meta=doc.get('forecastArtifact') or {}
   expected=V5/('development-forecasts.parquet' if stage=='develop' else 'forecasts.parquet')
   artifact=Path(meta.get('path',''))
   if artifact.resolve()!=expected.resolve() or not artifact.is_file() or sha(artifact)!=meta.get('sha256'):
    raise ValueError('blocked_integrity: V5 forecast stage artifact is not the exact consumed forecast file')
   split='develop' if stage=='develop' else 'test';snapshot=V5/f'fit-ledger-{split}.json'
   if not snapshot.is_file() or sha(snapshot)!=doc.get('fitLedgerSha256'):
    raise ValueError('blocked_integrity: V5 stage ledger snapshot does not match checkpoint')
   manifest['V5-fit-ledger-snapshot:'+split]={'path':str(snapshot.relative_to(ROOT)),'sha256':sha(snapshot)}
   if stage=='forecast':
    selection=V5/'development-selection.json'
    if not selection.is_file() or sha(selection)!=doc.get('selectionSha256'):
     raise ValueError('blocked_integrity: test forecast does not bind to the consumed frozen selection')
  if stage=='development-score':
   selection=V5/'development-selection.json'
   if not selection.is_file() or Path(doc.get('selectionPath','')).resolve()!=selection.resolve() or sha(selection)!=doc.get('selectionSha256'):
    raise ValueError('blocked_integrity: frozen development selection does not match saved development-score checkpoint')
   manifest['V5:development-selection.json']={'path':str(selection.relative_to(ROOT)),'sha256':sha(selection)}
 # Tie the mutable combined ledger used for reconstruction to the committed
 # split snapshots whose digests the forecast checkpoints froze.
 all_ledger=json.loads((V5/'fit-ledger.json').read_text(encoding='utf8'))
 for split in ('develop','test'):
  snapshot=json.loads((V5/f'fit-ledger-{split}.json').read_text(encoding='utf8'))
  if [x for x in all_ledger if x.get('split')==split]!=snapshot:
   raise ValueError(f'blocked_integrity: combined V5 ledger differs from immutable {split} snapshot')
 score_stage=v5.load_stage('score',V5)
 if sha(V5/'score-report.json')!=score_stage.get('scoreReportSha256'):
  raise ValueError('blocked_integrity: saved V5 score report does not match score checkpoint')
 return manifest
def load_fit_support():
 path=V5/'fit-ledger.json'; raw=json.loads(path.read_text(encoding='utf8'))
 entries=raw if isinstance(raw,list) else raw.get('entries',raw.get('ledger',[]))
 if not isinstance(entries,list): raise ValueError('fit ledger schema unsupported')
 complete=[r for r in entries if r.get('status')=='complete']
 return {'ledger_path':str(path.relative_to(ROOT)),'ledger_sha256':sha(path),'fit_attempts':len(entries),'completed_fits':len(complete),
         'train_rows_total_over_repeated_fits':sum(int(r.get('trainRows',0)) for r in complete),
         'unique_crop_target_total_over_repeated_fits':sum(int(r.get('uniqueCropTarget',0)) for r in complete),
         'last_mature_target_end_min':min((r['lastMatureTargetEnd'] for r in complete if r.get('lastMatureTargetEnd')),default=None),
         'last_mature_target_end_max':max((r['lastMatureTargetEnd'] for r in complete if r.get('lastMatureTargetEnd')),default=None)}
def audit_fit_contract(series, ledger=None):
 """Rebuild only input/label/weight contracts from the pinned source; never fits or predicts."""
 if ledger is None:
  raw=json.loads((V5/'fit-ledger.json').read_text(encoding='utf8'));ledger=raw if isinstance(raw,list) else raw.get('entries',[])
 complete=[r for r in ledger if r.get('status')=='complete']
 catalog,_=v5.build_catalog(series)
 by_origin={}; details=[]; code_id=v5.run_identity()
 for entry in complete:
  origin=entry.get('origin')
  if origin not in by_origin:
   train=v5.eligible_train(catalog,dt.date.fromisoformat(origin));contract=v5._fit_contract(train);weights=contract['weights'];n=len(train)
   # Independently confirm 1/m row balancing followed by equal cell weighting.
   counts={}
   for r in train:
    key=(r['crop'],r['target']);counts[key]=counts.get(key,0)+1
   base=np.asarray([1/counts[(r['crop'],r['target'])] for r in train],float)
   expected=base.copy();cell_sizes=contract['cellSupport']
   for c in v5.CROPS:
    for h in v5.HORIZONS:
     idx=np.array([i for i,r in enumerate(train) if r['crop']==c and r['horizon']==h]);expected[idx]*=(n/8)/expected[idx].sum()
   expected*=n/expected.sum()
   if not np.allclose(expected,weights,rtol=0,atol=1e-14):raise ValueError('blocked_integrity: reconstructed duplicate/cell weights differ')
   cell_sums={f'{c}:{h}':float(sum(weights[i] for i,r in enumerate(train) if r['crop']==c and r['horizon']==h)) for c in v5.CROPS for h in v5.HORIZONS}
   if any(abs(x-n/8)>1e-8 for x in cell_sums.values()):raise ValueError('blocked_integrity: training weighted cell totals not equal')
   detail=[]
   for c in v5.CROPS:
    for h in v5.HORIZONS:
     ix=[r for r in train if r['crop']==c and r['horizon']==h]
     uniq=len({(r['crop'],r['target']) for r in ix})
     detail.append({'fit_origin':origin,'crop':c,'horizon':h,'training_rows':len(ix),'unique_crop_target':uniq,'duplicate_rows':len(ix)-uniq,'weighted_cell_sum':cell_sums[f'{c}:{h}'],'target_cell_sum':n/8,'last_mature_target_end':max((r['targetEnd'].isoformat() for r in ix),default=None)})
   by_origin[origin]={'trainRows':n,'uniqueCropTarget':len(counts),'cellSupport':cell_sizes,'contract':contract,'details':detail}
  snap=by_origin[origin];identity=entry.get('fitIdentity') or {}
  if v5.sha_json(identity)!=entry.get('fitIdentitySha256'):
   raise ValueError(f'blocked_integrity: ledger fit identity digest mismatch {origin}')
  expected_config=v5.CONFIGS.get(entry.get('config')) if entry.get('kind')=='G' else None
  if identity.get('split')!=entry.get('split') or identity.get('origin')!=origin or identity.get('config')!=expected_config:
   raise ValueError(f'blocked_integrity: ledger fit metadata/configuration mismatch {origin}')
  for k in ('trainingKeySha256','numericFeatureMatrixSha256','labelSha256','weightSha256','featureNames'):
   if identity.get(k)!=snap['contract'].get(k):raise ValueError(f'blocked_integrity: reconstructed fit contract mismatch {origin} {k}')
  if entry.get('trainRows')!=snap['trainRows'] or entry.get('uniqueCropTarget')!=snap['uniqueCropTarget'] or entry.get('trainCells')!=snap['cellSupport']:
   raise ValueError(f'blocked_integrity: reconstructed fit ledger support mismatch {origin}')
  ri=identity.get('runIdentity') or {}
  if ri.get('analysisCodeSha256')!=code_id['analysisCodeSha256'] or ri.get('protocolSha256')!=v5.PROTOCOL_SHA:
   raise ValueError(f'blocked_integrity: fit ledger run identity mismatch {origin}')
  if identity.get('kind')!=entry.get('kind'):
   raise ValueError(f'blocked_integrity: fit ledger model/config mismatch {origin}')
 origins=sorted(by_origin)
 details=[row for o in origins for row in by_origin[o]['details']]
 crop_origin=[]
 for o in origins:
  train=v5.eligible_train(catalog,dt.date.fromisoformat(o))
  for c in v5.CROPS:
   part=[r for r in train if r['crop']==c];unique=len({r['target'] for r in part})
   crop_origin.append({'fit_origin':o,'crop':c,'training_rows':len(part),'unique_crop_target':unique,'duplicate_rows':len(part)-unique,'last_mature_target_end':max((r['targetEnd'].isoformat() for r in part),default=None)})
 return {'ledgerEntries':len(ledger),'completeFits':len(complete),'uniqueOrigins':len(origins),'originSupportRows':details,'cropOriginSupportRows':crop_origin,'contractChecked':len(complete),'weightsChecked':len(origins),'trainingRowsByOrigin':{o:by_origin[o]['trainRows'] for o in origins}}
def mape8(rows,model):
 cells=[]
 for crop in proxy.CROPS:
  for h in proxy.HORIZONS:
   q=[r for r in rows if r['crop']==crop and r['horizon']==h]
   if not q: raise ValueError('missing cell in equal-cell MAPE')
   cells.append(float(np.mean([r[model+'_ape'] for r in q]))*100)
 return float(np.mean(cells)),cells
def direction_summary(actual, predicted):
 """Three-way sign agreement; exact zero is kept as neutral in numerator/denominator."""
 if len(actual)!=len(predicted): raise ValueError('direction arrays differ in length')
 signs=lambda x: 1 if x>0 else -1 if x<0 else 0
 n=len(actual)
 return {'n':n,'zero_actual':sum(signs(x)==0 for x in actual),'zero_predicted':sum(signs(x)==0 for x in predicted),
         'agreement_including_zero':sum(signs(a)==signs(p) for a,p in zip(actual,predicted))/n if n else None}
def build_metric_rows(g_frame=None,p_frame=None,series=None,require_planned_grid=False):
 # The frozen score artifact is the source for selected development aggregate;
 # only saved test predictions are joined to source actuals here.
 g=g_frame if g_frame is not None else read_frame(V5/'forecasts.parquet'); p=p_frame if p_frame is not None else read_frame(V3/'forecasts_test.parquet')
 series=series if series is not None else proxy.series_from_raw(json.loads(proxy.SOURCE.read_text(encoding='utf8')))
 # Exact grid/key/source consistency precedes any metric construction.
 def indexed(frame,label):
  records=frame.to_dict('records');keys=[period_key(r) for r in records]
  if len(set(keys))!=len(keys): raise ValueError(f'blocked_integrity: duplicate {label} forecast key')
  return dict(zip(keys,records))
 gm=indexed(g,'V5');pm=indexed(p,'V3')
 if set(gm)!=set(pm): raise ValueError('blocked_integrity: V3/V5 planned grid keys differ')
 if require_planned_grid and (len(gm)!=792 or len(pm)!=792): raise ValueError('blocked_integrity: V3/V5 planned grids are not 792 rows')
 unavail_g={k for k,r in gm.items() if r.get('policyUnavailable') is True}
 unavail_p={k for k,r in pm.items() if r.get('policyUnavailable') is True}
 # Pandas may deserialize booleans as numpy.bool_; use the same explicit null-safe check.
 def is_unavail(r):
  x=r.get('policyUnavailable');return x is not None and bool(x) and str(x).lower()!='nan'
 unavail_g={k for k,r in gm.items() if is_unavail(r)};unavail_p={k for k,r in pm.items() if is_unavail(r)}
 if require_planned_grid and len(unavail_g)!=12: raise ValueError('blocked_integrity: planned unavailable forecast count is not 12')
 if unavail_g!=unavail_p: raise ValueError('blocked_integrity: unavailable forecast identities differ')
 # keyed saved rows; reconstruct actual from the pinned daily observations.
 by={}
 for r in gm.values():
  key=period_key(r); pred=r.get('predictions') or {}
  by.setdefault(key,{}).update({'G-B':(pred.get('G-B') or {}).get('predicted'),'L-L':(pred.get('L-L') or {}).get('predicted'),'G-B_status':(pred.get('G-B') or {}).get('status'),'L-L_status':(pred.get('L-L') or {}).get('status'),'p_ref':r.get('p_ref'),'features':r.get('asOfFeatures') or {}})
 for r in pm.values():
  key=period_key(r); pred=r.get('forecasts') or {}
  by.setdefault(key,{}).update({'P':(pred.get('P') or {}).get('predicted'),'B0':(pred.get('B0') or {}).get('predicted'),'P_status':(pred.get('P') or {}).get('fallbackReason') or (pred.get('P') or {}).get('status'),'p_ref':(pred.get('P') or {}).get('referencePrice') or by.get(key,{}).get('p_ref')})
 rows=[]
 for key,m in by.items():
  crop,h,origin,start,end=key; lo=dt.date.fromisoformat(start);hi=dt.date.fromisoformat(end)
  # target actual = arithmetic mean of >=3 positive daily prices.
  obs=[x[0] for d,x in series[crop].items() if lo<=d<=hi and x[0]>0]
  if len(obs)<3: continue
  actual=float(np.mean(obs));row={'crop':crop,'horizon':h,'origin':origin,'targetStart':start,'targetEnd':end,'actual':actual,'p_ref':m.get('p_ref'),'features':m.get('features',{})}
  for model in ('P','G-B','L-L'):row[model+'_status']=m.get(model+'_status')
  for model in ('P','B0','G-B','L-L'):
   pred=m.get(model)
   if pred is not None and math.isfinite(float(pred)) and float(pred)>0:
    row[model+'_pred']=float(pred);row[model+'_ae']=abs(float(pred)-actual);row[model+'_ape']=abs(float(pred)-actual)/actual
    row[model+'_z']=math.log(float(pred)/float(row['p_ref'])) if row['p_ref'] else None
  row['actual_z']=math.log(actual/float(row['p_ref'])) if row['p_ref'] else None
  row['year']=lo.year
  row['actual_change']=actual/float(row['p_ref'])-1 if row['p_ref'] else None
  rows.append(row)
 # A row is retained only for models that exist; exact paired rows are checked below.
 paired=[r for r in rows if all(r.get(k+'_pred') for k in ('P','B0','G-B','L-L'))]
 if len(paired)!=780: raise ValueError(f'blocked_integrity: paired forecast support {len(paired)} != 780')
 if len({(r['crop'],r['horizon'],r['origin'],r['targetStart'],r['targetEnd']) for r in paired})!=780:
  raise ValueError('blocked_integrity: duplicate paired forecast keys')
 expected={(c,h):(98 if h==2 else 97) for c in proxy.CROPS for h in proxy.HORIZONS}
 observed={(c,h):sum(r['crop']==c and r['horizon']==h for r in paired) for c in proxy.CROPS for h in proxy.HORIZONS}
 if observed!=expected: raise ValueError(f'blocked_integrity: paired per-cell support mismatch {observed}')
 if len({(r['targetStart'],r['targetEnd']) for r in paired})!=98 or len({(r['crop'],r['targetStart'],r['targetEnd']) for r in paired})!=392:
  raise ValueError('blocked_integrity: unique calendar/crop-target support differs from frozen 98/392')
 for r in paired:
  v5row=gm[(r['crop'],r['horizon'],r['origin'],r['targetStart'],r['targetEnd'])]
  v3row=pm[(r['crop'],r['horizon'],r['origin'],r['targetStart'],r['targetEnd'])]
  pref=v3row['forecasts']['P']['referencePrice'];
  if not math.isclose(float(pref),float(v5row['p_ref']),rel_tol=1e-12,abs_tol=1e-12):raise ValueError('blocked_integrity: V3/V5 reference-price mismatch')
 import pandas as pd
 for key,m in by.items():
  pref_value=m.get('p_ref')
  missing_pref=pref_value is None or bool(pd.isna(pref_value))
  if key in unavail_g:
   # Keep the twelve explicitly unavailable early-origin keys in the plan;
   # their null p_ref is expected and must not be treated as a usable forecast.
   if not missing_pref:raise ValueError('blocked_integrity: policy-unavailable row unexpectedly has a reference price')
   continue
  if missing_pref:raise ValueError('blocked_integrity: available forecast key has a missing reference price')
  pref=float(pref_value)
  if not math.isfinite(pref) or pref<=0:raise ValueError('blocked_integrity: available forecast key has an invalid reference price')
 return paired
def build_development_rows(g_frame=None,p_frame=None,series=None,selection=None):
 """Pair only saved development P/G-B/L-L rows; never synthesize B0 forecasts."""
 g=g_frame if g_frame is not None else read_frame(V5/'development-forecasts.parquet')
 p=p_frame if p_frame is not None else read_frame(V3/'forecasts.parquet')
 series=series if series is not None else proxy.series_from_raw(json.loads(proxy.SOURCE.read_text(encoding='utf8')))
 if selection is None:
  selection_path=V3/'selection.json'
  expected=v5.V3_FILES.get('selection.json')
  if expected is None or not selection_path.is_file() or sha(selection_path)!=expected:
   raise ValueError('blocked_integrity: pinned V3 development baseline selection mismatch')
  selection=json.loads(selection_path.read_text(encoding='utf8'))
 p_method={c:(selection.get('P',{}).get(c) or {}).get('method') for c in proxy.CROPS}
 if any(m not in ('R','W') for m in p_method.values()):
  raise ValueError(f'blocked_integrity: unexpected saved V3 development P method map {p_method}')
 grecords=g.to_dict('records');precords=p.to_dict('records');gkeys=[period_key(r) for r in grecords];pkeys=[period_key(r) for r in precords]
 if len(set(gkeys))!=len(gkeys) or len(set(pkeys))!=len(pkeys):raise ValueError('blocked_integrity: duplicate development forecast key')
 gp=dict(zip(gkeys,grecords));pp=dict(zip(pkeys,precords))
 if len(gp)!=576 or len(pp)!=576: raise ValueError(f'blocked_integrity: development planned grids {len(gp)}/{len(pp)} != 576')
 if set(gp)!=set(pp): raise ValueError('blocked_integrity: development saved forecast keys differ')
 common=sorted(gp); out=[]
 for key in common:
  gr=gp[key];pr=pp[key];crop,h,origin,start,end=key
  preds=gr.get('predictions') or {}; old=pr.get('forecasts') or {}; method=p_method[crop]
  selected=old.get(method)
  if not isinstance(selected,dict) or selected.get('predicted') is None or selected.get('referencePrice') is None:
   raise ValueError(f'blocked_integrity: saved V3 development selected P method {method} missing for {crop}/{h}/{origin}')
  ppred=selected.get('predicted');oldpref=selected.get('referencePrice')
  if not math.isfinite(float(ppred)) or float(ppred)<=0 or not math.isfinite(float(oldpref)) or float(oldpref)<=0:
   raise ValueError(f'blocked_integrity: invalid saved V3 development selected P prediction/reference {crop}/{h}/{origin}')
  gpred=(preds.get('G-B') or {}).get('predicted'); lpred=(preds.get('L-L') or {}).get('predicted')
  pref=gr.get('p_ref')
  if pref is None:pref=oldpref
  if oldpref is not None and gr.get('p_ref') is not None and not math.isclose(float(oldpref),float(gr['p_ref']),rel_tol=1e-12,abs_tol=1e-12):raise ValueError('blocked_integrity: V3/V5 development reference-price mismatch')
  if pref is None or not math.isfinite(float(pref)) or float(pref)<=0:raise ValueError(f'blocked_integrity: invalid saved development reference price {crop}/{h}/{origin}')
  lo=dt.date.fromisoformat(start);hi=dt.date.fromisoformat(end)
  obs=[x[0] for day,x in series[crop].items() if lo<=day<=hi and x[0]>0]
  if len(obs)<3: continue
  actual=float(np.mean(obs));r={'crop':crop,'horizon':h,'origin':origin,'targetStart':start,'targetEnd':end,'year':lo.year,'actual':actual,'p_ref':pref,'actual_z':math.log(actual/pref),'actual_change':actual/pref-1,'features':gr.get('asOfFeatures') or {},'P_selected_method':method,'P_saved_status':selected.get('status'),'P_clippedAtFloor':selected.get('clippedAtFloor'),'P_ratioRaw':selected.get('ratioRaw'),'P_ratioUsed':selected.get('ratioUsed')}
  for m,pred in (('P',ppred),('G-B',gpred),('L-L',lpred)):
   if pred is None or not math.isfinite(float(pred)) or float(pred)<=0: continue
   r[m+'_pred']=float(pred);r[m+'_ae']=abs(float(pred)-actual);r[m+'_ape']=r[m+'_ae']/actual;r[m+'_z']=math.log(float(pred)/pref)
  if all(r.get(m+'_pred') is not None for m in ('P','G-B','L-L')): out.append(r)
 if len(out)!=556: raise ValueError(f'blocked_integrity: saved development eligible paired support {len(out)} != 556')
 cell_counts={(c,h):sum(r['crop']==c and r['horizon']==h for r in out) for c in proxy.CROPS for h in proxy.HORIZONS}
 expected={(c,h):(62 if c=='garlic' else 72) for c in proxy.CROPS for h in proxy.HORIZONS}
 if cell_counts!=expected: raise ValueError(f'blocked_integrity: development per-cell support differs: {cell_counts}')
 return out
def verify_forecast_grids(g_frame,p_frame):
 def count(frame):
  if len(frame)!=792: raise ValueError(f'blocked_integrity: forecast grid rows {len(frame)} != 792')
  import pandas as pd
  unavailable=sum(1 for x in frame.get('policyUnavailable',[]) if x is not None and not pd.isna(x) and bool(x))
  if unavailable!=12: raise ValueError(f'blocked_integrity: policy-unavailable rows {unavailable} != 12')
  return {'rows':len(frame),'policy_unavailable':unavailable,'available':len(frame)-unavailable}
 g=count(g_frame);p=count(p_frame)
 return {'V5_GB_LL':g,'V3_P_B0':p}
def metric_table(rows):
 out=[]
 for crop in proxy.CROPS:
  for h in proxy.HORIZONS:
   q=[r for r in rows if r['crop']==crop and r['horizon']==h]
   rec={'crop':crop,'horizon':h,'n':len(q),'unique_targets':len({r['targetStart'] for r in q})}
   for m in ('P','B0','G-B','L-L'):
    if not all(r.get(m+'_ape') is not None for r in q): continue
    rec[m+'_MAE']=float(np.mean([r[m+'_ae'] for r in q]));rec[m+'_MAPE_pct']=100*float(np.mean([r[m+'_ape'] for r in q]))
   if 'G-B_MAPE_pct' in rec:
    rec['G-B_MAPE8_contribution_pp']=rec['G-B_MAPE_pct']/8;rec['G-B_gap_to_cell_10pp']=rec['G-B_MAPE_pct']-10
   out.append(rec)
 return out
def annual_table(rows):
 out=[]
 for year in sorted({r['year'] for r in rows}):
  for crop in proxy.CROPS:
   for h in proxy.HORIZONS:
    q=[r for r in rows if r['year']==year and r['crop']==crop and r['horizon']==h]
    if not q: continue
    z={'year':year,'crop':crop,'horizon':h,'n_forecasts':len(q),'unique_targets':len({r['targetStart'] for r in q})}
    for m in ('P','B0','G-B','L-L'):
     z[m+'_MAE']=float(np.mean([r[m+'_ae'] for r in q]));z[m+'_MAPE_pct']=100*float(np.mean([r[m+'_ape'] for r in q]))
    z['G_minus_P_MAPE_pp']=z['G-B_MAPE_pct']-z['P_MAPE_pct'];z['G_minus_P_MAE']=z['G-B_MAE']-z['P_MAE'];out.append(z)
 return out
def actual_response_table(rows,split='test'):
 """Calendar-distance and realized ratio distributions by crop/horizon/year."""
 out=[]
 for c in proxy.CROPS:
  for h in proxy.HORIZONS:
   for year in sorted({r['year'] for r in rows if r['crop']==c and r['horizon']==h}):
    q=[r for r in rows if r['crop']==c and r['horizon']==h and r['year']==year]
    changes=[r['actual_change'] for r in q]
    logchanges=[math.log1p(x) for x in changes if x>-1]
    distances=[(dt.date.fromisoformat(r['targetStart'])-dt.date.fromisoformat(r['origin'])).days for r in q]
    d={'diagnostic_type':'actual_response_calendar','split':split,'crop':c,'horizon':h,'year':year,'n_forecasts':len(q),'unique_targets':len({r['targetStart'] for r in q}),
       'up_count':sum(x>0 for x in changes),'flat_count':sum(x==0 for x in changes),'down_count':sum(x<0 for x in changes),
       'abs_change_ge10_n':sum(abs(x)>=.1 for x in changes),'abs_change_ge20_n':sum(abs(x)>=.2 for x in changes),'abs_change_ge30_n':sum(abs(x)>=.3 for x in changes)}
    d.update({'simple_change_'+k:v for k,v in stats(changes).items()});d.update({'absolute_change_'+k:v for k,v in stats([abs(x) for x in changes]).items()});d.update({'log_change_'+k:v for k,v in stats(logchanges).items()});d.update({'target_start_distance_days_'+k:v for k,v in stats(distances).items()})
    out.append(d)
 return out
def model_reconciliation(rows,models=('B0','P','G-B','L-L')):
 rec=[]
 for m in models:
  mae_cells=[];mape_cells=[]
  for c in proxy.CROPS:
   for h in proxy.HORIZONS:
    q=[r for r in rows if r['crop']==c and r['horizon']==h]
    mae_cells.append(float(np.mean([r[m+'_ae'] for r in q])));mape_cells.append(100*float(np.mean([r[m+'_ape'] for r in q])))
  rec.append({'model':m,'paired_rows':len(rows),'cells':len(mae_cells),'MAE8':float(np.mean(mae_cells)),'MAPE8_pct':float(np.mean(mape_cells))})
 return rec
def validate_saved_score(rows,score,models=('P','B0','G-B','L-L'),years=None):
 """Compare row-reconstructed full-precision cell/year scores with frozen score JSON."""
 cells=metric_table(rows); by={(x['crop'],x['horizon']):x for x in cells}
 split=score.get('metrics',{}) if years is None else years
 for model in models:
  report=split.get(model)
  if not report: raise ValueError(f'blocked_integrity: saved score missing {model}')
  actual=[r['actual'] for r in rows]
  # Check every crop-distance cell including support counts and full precision metrics.
  for crop in proxy.CROPS:
   for h in proxy.HORIZONS:
    key=f'{crop}:{h}'; expected=report.get('cells',{}).get(key); observed=by[(crop,h)]
    if not expected or int(expected['n'])!=observed['n']:raise ValueError(f'blocked_integrity: saved score cell support differs {model} {key}')
    for source,target in (('mae',model+'_MAE'),('mape',model+'_MAPE_pct')):
     if abs(float(expected[source])-observed[target])>1e-9:
      raise ValueError(f'blocked_integrity: saved score full precision cell {source} differs {model} {key}')
  rec=model_reconciliation(rows,(model,))[0]
  for key,observed in (('mae8',rec['MAE8']),('mape8',rec['MAPE8_pct'])):
   if abs(float(report[key])-observed)>1e-9:raise ValueError(f'blocked_integrity: saved score full precision aggregate {key} differs {model}')
 return {'modelsChecked':list(models),'cellsChecked':8*len(models),'tolerance':1e-9}
def validate_yearly_score(rows,report):
 checked=0
 for year,models in report.get('yearly',{}).items():
  yr=int(year); yr_rows=[r for r in rows if r['year']==yr]
  if not yr_rows: raise ValueError(f'blocked_integrity: saved score year absent in reconstructed rows {year}')
  for model,metric in models.items():
   by={(r['crop'],r['horizon']):r for r in metric_table(yr_rows)}
   for crop in proxy.CROPS:
    for h in proxy.HORIZONS:
     key=f'{crop}:{h}';expected=metric['cells'][key];observed=by[(crop,h)]
     if int(expected['n'])!=observed['n']:raise ValueError(f'blocked_integrity: yearly cell support mismatch {year} {model} {key}')
     if abs(float(expected['mae'])-observed[model+'_MAE'])>1e-9 or abs(float(expected['mape'])-observed[model+'_MAPE_pct'])>1e-9:
      raise ValueError(f'blocked_integrity: full precision yearly cell score mismatch {year} {model} {key}')
   checked+=8
   macro=model_reconciliation(yr_rows,(model,))[0]
   for key,value in (('mae8',macro['MAE8']),('mape8',macro['MAPE8_pct'])):
    if key not in metric or abs(float(metric[key])-value)>1e-9:
     raise ValueError(f'blocked_integrity: yearly equal-cell {key} mismatch {year} {model}')
 return {'yearCellsChecked':checked,'yearAggregatesChecked':2*sum(len(x) for x in report.get('yearly',{}).values()),'years':list(report.get('yearly',{})),'tolerance':1e-9}
def direction_table(rows,split='test'):
 out=[]; sign=lambda x:1 if x>0 else -1 if x<0 else 0; names={-1:'down',0:'flat',1:'up'}
 for c in proxy.CROPS:
  for h in proxy.HORIZONS:
   for year in sorted({r['year'] for r in rows if r['crop']==c and r['horizon']==h}):
    q=[r for r in rows if r['crop']==c and r['horizon']==h and r['year']==year]
    for m in ('B0','P','G-B','L-L'):
     available=[r for r in q if r.get(m+'_z') is not None and r.get(m+'_pred') is not None]
     if not available:continue
     for axis in ('actual','predicted'):
      for value in (-1,0,1):
       part=[r for r in available if sign(r['actual_z'] if axis=='actual' else r[m+'_z'])==value]
       actual_signs=[sign(r['actual_z']) for r in part];pred_signs=[sign(r[m+'_z']) for r in part]
       out.append({'diagnostic_type':'conditional_direction','split':split,'crop':c,'horizon':h,'year':year,'model':m,'condition_axis':axis,'condition_direction':names[value],'n_forecasts':len(part),'n_unique_targets':len({r['targetStart'] for r in part}),
        'actual_up':sum(x>0 for x in actual_signs),'actual_flat':sum(x==0 for x in actual_signs),'actual_down':sum(x<0 for x in actual_signs),
        'pred_up':sum(x>0 for x in pred_signs),'pred_flat':sum(x==0 for x in pred_signs),'pred_down':sum(x<0 for x in pred_signs),
        'direction_agreement_including_zero':sum(a==p for a,p in zip(actual_signs,pred_signs))/len(part) if part else None,
        'underprediction':sum(r[m+'_pred']<r['actual'] for r in part),'overprediction':sum(r[m+'_pred']>r['actual'] for r in part),
        'MAE':float(np.mean([r[m+'_ae'] for r in part])) if part else None,'MAPE_pct':100*float(np.mean([r[m+'_ape'] for r in part])) if part else None})
 return out
def tail_table(rows,split='test'):
 out=[]
 for c in proxy.CROPS:
  for h in proxy.HORIZONS:
   for year in sorted({r['year'] for r in rows if r['crop']==c and r['horizon']==h}):
    q=[r for r in rows if r['crop']==c and r['horizon']==h and r['year']==year and r.get('actual_z') is not None]
    if not q: continue
    lo,hi=np.quantile([r['actual_z'] for r in q],[.2,.8])
    for label,test in [('actual_z_bottom20',lambda r:r['actual_z']<=lo),('actual_z_top20',lambda r:r['actual_z']>=hi)]:
     g=[r for r in q if test(r)];rec={'diagnostic_type':'realized_z_tail','split':split,'crop':c,'horizon':h,'year':year,'group':label,'actual_z_cut':float(lo if label.endswith('bottom20') else hi),'n_forecasts':len(g),'unique_targets':len({r['targetStart'] for r in g})}
     for m in ('P','B0','G-B','L-L'):
      if not all(r.get(m+'_ae') is not None for r in q):continue
      rec[m+'_mean_price_bias']=float(np.mean([r[m+'_pred']-r['actual'] for r in g])) if g else None;rec[m+'_MAE']=float(np.mean([r[m+'_ae'] for r in g])) if g else None;rec[m+'_MAPE_pct']=100*float(np.mean([r[m+'_ape'] for r in g])) if g else None
     out.append(rec)
 return out
def compression_table(rows,by_year=True,split='test'):
 out=[]
 for c in proxy.CROPS:
  for h in proxy.HORIZONS:
   years=sorted({r['year'] for r in rows if r['crop']==c and r['horizon']==h}) if by_year else ['ALL']
   for year in years:
    q=[r for r in rows if r['crop']==c and r['horizon']==h and (not by_year or r['year']==year) and r.get('actual_z') is not None]
    if not q: continue
    actual=np.asarray([r['actual_z'] for r in q]);rec={'diagnostic_type':'prediction_compression','split':split,'crop':c,'horizon':h,'year':year,'n_forecasts':len(q),'actual_z_sd':float(actual.std()),'actual_z_mean':float(actual.mean()),'actual_z_positive_share':float(np.mean(actual>0)),'actual_z_negative_share':float(np.mean(actual<0)),'actual_z_zero_share':float(np.mean(actual==0))}
    rec.update({'actual_z_'+k:v for k,v in stats(actual).items() if k!='n'})
    for m in ('P','B0','G-B','L-L'):
     z=np.asarray([r[m+'_z'] for r in q if r.get(m+'_z') is not None]);rec[m+'_z_sd']=float(z.std()) if len(z) else None;rec[m+'_std_ratio_to_actual']=float(z.std()/actual.std()) if len(z) and actual.std() else None;rec[m+'_z_mean']=float(z.mean()) if len(z) else None
     rec[m+'_z_positive_share']=float(np.mean(z>0)) if len(z) else None;rec[m+'_z_negative_share']=float(np.mean(z<0)) if len(z) else None;rec[m+'_z_zero_share']=float(np.mean(z==0)) if len(z) else None
     rec.update({m+'_z_'+k:v for k,v in stats(z).items() if k!='n'})
    out.append(rec)
 return out
def error_table(rows,split='test'):
 out=[]
 for crop in proxy.CROPS:
  for h in proxy.HORIZONS:
   for year in sorted({r['year'] for r in rows if r['crop']==crop and r['horizon']==h}):
    allq=[r for r in rows if r['crop']==crop and r['horizon']==h and r['year']==year]
    prices=[r['actual'] for r in allq];q20,q80=np.quantile(prices,[.2,.8]) if prices else (None,None)
    groups=[('realized_abs_change_lt20',[r for r in allq if abs(r['actual_change'])<.2]),('realized_abs_change_ge20',[r for r in allq if abs(r['actual_change'])>=.2]),('actual_price_bottom20',[r for r in allq if r['actual']<=q20]),('actual_price_top20',[r for r in allq if r['actual']>=q80])]
    for label,q in groups:
     z={'diagnostic_type':'realized_regime','split':split,'crop':crop,'horizon':h,'year':year,'group':label,'n_forecasts':len(q),'unique_targets':len({r['targetStart'] for r in q}),'actual_price_mean':float(np.mean([r['actual'] for r in q])) if q else None,'actual_price_median':float(np.median([r['actual'] for r in q])) if q else None,'actual_p_ref_mean':float(np.mean([r['p_ref'] for r in q])) if q else None,'actual_change_mean':float(np.mean([r['actual_change'] for r in q])) if q else None}
     for m in ('P','B0','G-B','L-L'):
      if not all(r.get(m+'_ae') is not None for r in allq):continue
      ae=np.asarray([r[m+'_ae'] for r in q]);ape=np.asarray([r[m+'_ape'] for r in q]);full_ae=np.asarray([r[m+'_ae'] for r in allq]);full_ape=np.asarray([r[m+'_ape'] for r in allq])
      z[m+'_MAE']=float(ae.mean()) if len(ae) else None;z[m+'_MAPE_pct']=100*float(ape.mean()) if len(ape) else None;z[m+'_AE_total']=float(ae.sum()) if len(ae) else 0.;z[m+'_APE_total_pct_points']=100*float(ape.sum()) if len(ape) else 0.;z[m+'_AE_share_of_cell']=float(ae.sum()/full_ae.sum()) if len(ae) and full_ae.sum() else None;z[m+'_APE_share_of_cell']=float(ape.sum()/full_ape.sum()) if len(ape) and full_ape.sum() else None
     out.append(z)
 return out
def asof_groups(rows,dev_path=V5/'development-forecasts.parquet',development_rows=None):
 """Apply development-only unique crop/origin tertiles to test feature rows."""
 dev=read_frame(dev_path).to_dict('records')
 availability={'reference_age_days':'reference_available','price_sd_7':'price_available_7','price_sd_28':'price_available_28','price_sd_90':'price_available_90','price_rel_28':'price_available_28','volume_ratio_10_20':'volume_ratio_available','volume_coverage_28':'volume_available_28'}
 features=tuple(availability)
 cut={}
 for f in features:
  vals={}
  avail=availability[f]
  for r in dev:
   if r.get('origin') is None: continue
   v=(r.get('asOfFeatures') or {}).get(f)
   mask=(r.get('asOfFeatures') or {}).get(avail,0)
   if bool(mask) and v is not None and math.isfinite(float(v)): vals[(r['crop'],r['origin'])]=float(v)
  cut[f]=np.quantile(list(vals.values()),[1/3,2/3]).tolist() if vals else []
 out=[]; splits=[('test',rows)]
 if development_rows is not None:splits.insert(0,('development',development_rows))
 split_models={name:tuple(m for m in ('P','B0','G-B','L-L') if any(r.get(m+'_ae') is not None for r in part)) for name,part in splits}
 for f in features:
  if len(cut[f])!=2: continue
  for split,split_rows in splits:
   for c in proxy.CROPS:
    q=[r for r in split_rows if r['crop']==c]
    for h in proxy.HORIZONS:
     for year in sorted({r['year'] for r in q if r['horizon']==h}):
      cell=[r for r in q if r['horizon']==h and r['year']==year]
      parts=[]
      for label,selector in [('low',lambda x:x<=cut[f][0]),('middle',lambda x:cut[f][0]<x<=cut[f][1]),('high',lambda x:x>cut[f][1])]:
       parts.append((label,[r for r in cell if bool((r.get('features') or {}).get(availability[f],0)) and (r.get('features') or {}).get(f) is not None and selector(float(r['features'][f]))]))
      parts.append(('missing_or_unavailable',[r for r in cell if not bool((r.get('features') or {}).get(availability[f],0))]))
      for group,part in parts:
       rec={'diagnostic_type':'asof_feature_tercile','split':split,'crop':c,'horizon':h,'year':year,'feature':f,'availability_feature':availability[f],'group':group,'dev_q33':cut[f][0],'dev_q67':cut[f][1],'n_forecasts':len(part),'unique_targets':len({r['targetStart'] for r in part}),'unique_crop_targets':len({(r['crop'],r['targetStart'],r['targetEnd']) for r in part}),'feature_available_n':sum(bool((r.get('features') or {}).get(availability[f],0)) for r in part),'feature_missing_n':sum(not bool((r.get('features') or {}).get(availability[f],0)) for r in part),'actual_price_mean':float(np.mean([r['actual'] for r in part])) if part else None,'actual_price_median':float(np.median([r['actual'] for r in part])) if part else None,'feature_mean':float(np.mean([(r.get('features') or {}).get(f) for r in part if (r.get('features') or {}).get(f) is not None])) if any((r.get('features') or {}).get(f) is not None for r in part) else None}
       for m in split_models[split]:
        base_ae=sum(r[m+'_ae'] for r in cell if r.get(m+'_ae') is not None);base_ape=sum(r[m+'_ape'] for r in cell if r.get(m+'_ape') is not None)
        group_ae=sum(r[m+'_ae'] for r in part if r.get(m+'_ae') is not None);group_ape=sum(r[m+'_ape'] for r in part if r.get(m+'_ape') is not None)
        rec[m+'_MAE']=float(np.mean([r[m+'_ae'] for r in part])) if part else None;rec[m+'_MAPE_pct']=100*float(np.mean([r[m+'_ape'] for r in part])) if part else None
        rec[m+'_AE_total']=group_ae;rec[m+'_APE_total_pct_points']=100*group_ape;rec[m+'_AE_share_of_cell_year']=group_ae/base_ae if base_ae else None;rec[m+'_APE_share_of_cell_year']=group_ape/base_ape if base_ape else None
       out.append(rec)
 # Fixed, interpretable target calendar and horizon-distance groups use no test-fitted cutpoints.
 for split,split_rows in splits:
  for c in proxy.CROPS:
   for h in proxy.HORIZONS:
    for year in sorted({r['year'] for r in split_rows if r['crop']==c and r['horizon']==h}):
     cell=[r for r in split_rows if r['crop']==c and r['horizon']==h and r['year']==year]
     feature_groups=[]
     feature_groups.append(('days_to_target_start',[("0-3 days",[r for r in cell if int((r.get('features') or {}).get('days_to_target_start',(dt.date.fromisoformat(r['targetStart'])-dt.date.fromisoformat(r['origin'])).days))<=3]),("4-7 days",[r for r in cell if 3<int((r.get('features') or {}).get('days_to_target_start',(dt.date.fromisoformat(r['targetStart'])-dt.date.fromisoformat(r['origin'])).days))<=7]),(">7 days",[r for r in cell if int((r.get('features') or {}).get('days_to_target_start',(dt.date.fromisoformat(r['targetStart'])-dt.date.fromisoformat(r['origin'])).days))>7])]))
     feature_groups.append(('target_season',[(name,[r for r in cell if name==('winter' if dt.date.fromisoformat(r['targetStart']).month in (12,1,2) else 'spring' if dt.date.fromisoformat(r['targetStart']).month in (3,4,5) else 'summer' if dt.date.fromisoformat(r['targetStart']).month in (6,7,8) else 'autumn')]) for name in ('winter','spring','summer','autumn')]))
     feature_groups.append(('target_dekad_phase',[(name,[r for r in cell if name==('early' if dt.date.fromisoformat(r['targetStart']).day<=10 else 'middle' if dt.date.fromisoformat(r['targetStart']).day<=20 else 'late')]) for name in ('early','middle','late')]))
     for feature,parts in feature_groups:
      for group,part in parts:
       rec={'diagnostic_type':'asof_fixed_calendar_group','split':split,'crop':c,'horizon':h,'year':year,'feature':feature,'availability_feature':'calendar_known_at_origin','group':group,'n_forecasts':len(part),'unique_targets':len({r['targetStart'] for r in part}),'unique_crop_targets':len({(r['crop'],r['targetStart'],r['targetEnd']) for r in part}),'actual_price_mean':float(np.mean([r['actual'] for r in part])) if part else None,'actual_price_median':float(np.median([r['actual'] for r in part])) if part else None}
       for m in split_models[split]:
        base_ae=sum(r[m+'_ae'] for r in cell if r.get(m+'_ae') is not None);base_ape=sum(r[m+'_ape'] for r in cell if r.get(m+'_ape') is not None);ga=sum(r[m+'_ae'] for r in part if r.get(m+'_ae') is not None);gp=sum(r[m+'_ape'] for r in part if r.get(m+'_ape') is not None)
        rec[m+'_MAE']=float(np.mean([r[m+'_ae'] for r in part])) if part else None;rec[m+'_MAPE_pct']=100*float(np.mean([r[m+'_ape'] for r in part])) if part else None;rec[m+'_AE_total']=ga;rec[m+'_APE_total_pct_points']=100*gp;rec[m+'_AE_share_of_cell_year']=ga/base_ae if base_ae else None;rec[m+'_APE_share_of_cell_year']=gp/base_ape if base_ape else None
       out.append(rec)
 return out,cut
def crop_table(series):
 out=[]
 for c,s in series.items():
  prices=[v[0] for v in s.values() if v[0]>0]; vols=[v[1] for v in s.values() if v[1]>0]
  rec={'crop':c,'first_date':min(s).isoformat() if s else None,'last_date':max(s).isoformat() if s else None,'observed_days':len(s),'positive_volume_days':len(vols)}
  rec.update({'price_'+k:v for k,v in stats(prices).items()});rec.update({'volume_'+k:v for k,v in stats(vols).items()})
  rr=adjacent_returns(s);rec['adjacent_day_pairs']=len(rr);rec['day_return_abs20_count']=sum(abs(x[1])>=.2 for x in rr)
  targets=proxy.aggregate_targets(s); valid={period_index(hi):float(v['actual']) for (lo,hi),v in targets.items() if v['actual'] is not None}
  rec['valid_dekads']=len(valid);rec.update({'dekad_price_'+k:v for k,v in stats(valid.values()).items()})
  dret=[valid[i]/valid[i-1]-1 for i in valid if i-1 in valid];lret=[math.log(valid[i]/valid[i-1]) for i in valid if i-1 in valid];rec['dekad_adjacent_pairs']=len(dret);rec.update({'dekad_return_'+k:v for k,v in stats(dret).items()});rec.update({'dekad_log_return_'+k:v for k,v in stats(lret).items()})
  for threshold in (.1,.2,.3):rec[f'dekad_abs_change_ge{int(threshold*100)}_count']=sum(abs(x)>=threshold for x in dret)
  for lag in (1,2,3,6,12,36):
   pairs=[(math.log(valid[i]),math.log(valid[i+lag])) for i in sorted(valid) if i+lag in valid]
   rec['acf_price_lag_'+str(lag)]=float(np.corrcoef([p[0] for p in pairs],[p[1] for p in pairs])[0,1]) if len(pairs)>2 else None;rec['acf_price_n_'+str(lag)]=len(pairs)
   diffs={i:math.log(valid[i]/valid[i-1]) for i in valid if i-1 in valid}
   dp=[(diffs[i],diffs[i+lag]) for i in sorted(diffs) if i+lag in diffs]
   rec['acf_logdiff_lag_'+str(lag)]=float(np.corrcoef([p[0] for p in dp],[p[1] for p in dp])[0,1]) if len(dp)>2 else None;rec['acf_logdiff_n_'+str(lag)]=len(dp)
  events=[(i,(valid[i]/valid[i-1]-1)) for i in valid if i-1 in valid and abs(valid[i]/valid[i-1]-1)>=.2]
  for ahead in (1,2,3):
   fut=[(valid[i+ahead]/valid[i]-1) for i,change in events if i+ahead in valid]
   rec['event_ge20_next_'+str(ahead)+'_n']=len(fut);rec['event_ge20_next_'+str(ahead)+'_mean_change']=float(np.mean(fut)) if fut else None
   rec['event_ge20_next_'+str(ahead)+'_reverse_share']=sum((change>0 and f<0) or (change<0 and f>0) for (i,change) in events if i+ahead in valid for f in [valid[i+ahead]/valid[i]-1])/len(fut) if fut else None
   for direction,label in ((1,'up'),(-1,'down')):
    selected=[(i,change) for i,change in events if np.sign(change)==direction and i+ahead in valid]
    changes=[valid[i+ahead]/valid[i]-1 for i,change in selected]
    rec[f'event_{label}_next_{ahead}_n']=len(changes);rec[f'event_{label}_next_{ahead}_mean_change']=float(np.mean(changes)) if changes else None
    rec[f'event_{label}_next_{ahead}_reverse_share']=sum((direction==1 and x<0) or (direction==-1 and x>0) for x in changes)/len(changes) if changes else None
  out.append(rec)
 return out
def support_profiles(series):
 """Return exact-common support and precomputed daily/dekad/season/volume plot tables."""
 import pandas as pd
 daysets=[set(s) for s in series.values()];common_days=set.intersection(*daysets) if daysets else set()
 targets={c:proxy.aggregate_targets(s) for c,s in series.items()}
 valid_sets=[{t for t,x in targets[c].items() if x['actual'] is not None} for c in series]
 common_periods=set.intersection(*valid_sets) if valid_sets else set()
 common_starts=sorted({lo for lo,hi in common_periods})
 daily_returns=[];dekad_returns=[];common_dekad_returns=[];seasonal=[];annual_volume=[];dekad_support=[]
 for crop,s in series.items():
  for day,simple,logret in adjacent_returns(s):
   prev=day-dt.timedelta(days=1);daily_returns.append({'crop':crop,'date':day.isoformat(),'previous_date':prev.isoformat(),'simple_return':simple,'log_return':logret})
  valid={period_index(hi):(lo,hi,float(x['actual'])) for (lo,hi),x in targets[crop].items() if x['actual'] is not None}
  for idx in sorted(valid):
   lo,hi,p=valid[idx]
   if idx-1 in valid:
    plo,phi,pp=valid[idx-1];dekad_returns.append({'crop':crop,'period_start':lo.isoformat(),'period_end':hi.isoformat(),'previous_period_end':phi.isoformat(),'simple_return':p/pp-1,'log_return':math.log(p/pp)})
  common_idx={period_index(hi) for lo,hi in common_periods}
  for idx in sorted(common_idx):
   if idx-1 in common_idx and idx in valid and idx-1 in valid:
    lo,hi,p=valid[idx];plo,phi,pp=valid[idx-1]
    common_dekad_returns.append({'crop':crop,'period_start':lo.isoformat(),'period_end':hi.isoformat(),'previous_period_end':phi.isoformat(),'simple_return':p/pp-1,'log_return':math.log(p/pp),'support':'both return endpoints are in exact four-crop valid-dekad intersection'})
  # Within-calendar-year centering uses only that year and is retrospective description.
  years=sorted({lo.year for lo,hi in targets[crop]}); slots_by_year={}
  for year in years:
   slots_by_year[year]={}
   for (lo,hi),x in targets[crop].items():
    if lo.year==year and x['actual'] is not None:
     slot=(lo.month-1)*3+(0 if lo.day==1 else 1 if lo.day==11 else 2)+1
     slots_by_year[year][slot]=(float(x['actual']),int(x['observedDays']))
   vals=[math.log(p) for p,_ in slots_by_year[year].values()];med=float(np.median(vals)) if vals else None
  for year in years:
   vals=[math.log(p) for p,_ in slots_by_year[year].values()];med=float(np.median(vals)) if vals else None
   for slot in range(1,37):
    pair=slots_by_year[year].get(slot);seasonal.append({'crop':crop,'year':year,'dekad':slot,'centered_log_price':math.log(pair[0])-med if pair and med is not None else None,'n_days':pair[1] if pair else 0,'n_years_by_dekad':sum(slot in d for d in slots_by_year.values())})
  for year in sorted({d.year for d in s}):
   vols=[float(v) for d,(p,v) in s.items() if d.year==year and math.isfinite(v) and v>0]
   prices=[float(p) for d,(p,v) in s.items() if d.year==year and math.isfinite(p) and p>0]
   price_days=sum(1 for d,(p,v) in s.items() if d.year==year and p>0)
   dekads_year=sum(lo.year==year and x['actual'] is not None for (lo,hi),x in targets[crop].items())
   rec={'crop':crop,'year':year,'n_valid_days':price_days,'n_valid_dekads':dekads_year,'common_price_days':sum(d.year==year for d in common_days),'common_price_dekads':sum(lo.year==year for lo,hi in common_periods)}
   rec.update({'volume_'+k:v for k,v in stats(vols).items()});rec['volume_median']=stats(vols).get('median');rec.update({'price_'+k:v for k,v in stats(prices).items()});annual_volume.append(rec)
  for (lo,hi),x in targets[crop].items():
   dekad_support.append({'diagnostic_type':'dekad_support','crop':crop,'target_start':lo.isoformat(),'target_end':hi.isoformat(),'year':lo.year,'observed_price_days':x['observedDays'],'valid_target':x['actual'] is not None,'status':x['targetStatus'],'common_valid_target':(lo,hi) in common_periods})
 # Add global exact intersection and per-crop distribution in the common overlap.
 # Annual distributions retain calendar gaps and separately isolate returns
 # whose two endpoints are both valid for all four crops.
 return_distribution=[]
 for scope,source in (('crop_valid_adjacent',dekad_returns),('four_crop_common_adjacent',common_dekad_returns)):
  for crop in series:
   for year in sorted({dt.date.fromisoformat(r['period_end']).year for r in source if r['crop']==crop}):
    q=[r for r in source if r['crop']==crop and dt.date.fromisoformat(r['period_end']).year==year]
    z={'scope':scope,'crop':crop,'year':year,'n_adjacent_period_returns':len(q),'first_previous_period_end':min((r['previous_period_end'] for r in q),default=None),'last_period_end':max((r['period_end'] for r in q),default=None)}
    z.update({'simple_return_'+k:v for k,v in stats([r['simple_return'] for r in q]).items()});z.update({'log_return_'+k:v for k,v in stats([r['log_return'] for r in q]).items()});return_distribution.append(z)
 cropdist=crop_table(series)
 for row,c in zip(cropdist,series):
  vals=[series[c][d][0] for d in common_days if series[c][d][0]>0]
  row['common_first_date']=min(common_days).isoformat() if common_days else None;row['common_last_date']=max(common_days).isoformat() if common_days else None
  row['common_price_days']=len(vals);row['common_valid_dekads']=sum(t in common_periods for t in targets[c])
  row.update({'common_price_'+k:v for k,v in stats(vals).items()})
  common_period_values=[float(targets[c][t]['actual']) for t in common_periods if t in targets[c] and targets[c][t]['actual'] is not None]
  row.update({'common_dekad_price_'+k:v for k,v in stats(common_period_values).items()})
 return {'daily_price_returns':daily_returns,'dekad_price_returns':dekad_returns,'common_dekad_returns':common_dekad_returns,'annual_return_distribution':return_distribution,'seasonal_pattern':seasonal,'annual_volume_support':annual_volume,'dekad_support':dekad_support,'crop_distribution':cropdist,'common_date_count':len(common_days),'common_period_count':len(common_periods),'common_period_first':min(common_starts).isoformat() if common_starts else None,'common_period_last':max(common_starts).isoformat() if common_starts else None}
def annual_data_support(crop_origin_support, development_rows, test_rows):
 """Annualize repeated training support by fit-origin year and eval support by target year."""
 out={}
 def rec(c,y):
  return out.setdefault((c,int(y)),{'crop':c,'year':int(y),'train_rows':0,'unique_crop_target':0,'duplicate_rows':0,'fit_origins':0,'eval_forecasts':0,'eval_unique_targets':0,'development_forecasts':0,'development_unique_targets':0,'_eval_keys':set(),'_dev_keys':set()})
 for r in crop_origin_support:
  z=rec(r['crop'],str(r['fit_origin'])[:4]);z['train_rows']+=int(r['training_rows']);z['unique_crop_target']+=int(r['unique_crop_target']);z['duplicate_rows']+=int(r['duplicate_rows']);z['fit_origins']+=1
 for split, rows in (('development',development_rows),('test',test_rows)):
  for r in rows:
   z=rec(r['crop'],r['year']);key=(r['crop'],r['targetStart'],r['targetEnd'])
   if split=='test':z['eval_forecasts']+=1;z['_eval_keys'].add(key)
   else:z['development_forecasts']+=1;z['_dev_keys'].add(key)
 for z in out.values():
  z['eval_unique_targets']=len(z.pop('_eval_keys'));z['development_unique_targets']=len(z.pop('_dev_keys'))
 return [out[k] for k in sorted(out)]
def cross_table(series):
 out=[]; rmap={}; dates_by_index={}
 for crop,daily in series.items():
  valid={period_index(hi):float(row['actual']) for (lo,hi),row in proxy.aggregate_targets(daily).items() if row['actual'] is not None and row['actual']>0}
  dates_by_index.update({period_index(hi):(lo,hi) for lo,hi in proxy.aggregate_targets(daily)})
  rmap[crop]={i:math.log(valid[i]/valid[i-1]) for i in valid if i-1 in valid}
 eras=[('all',None,None),('development',dt.date(2022,1,1),dt.date(2023,12,31)),('test',dt.date(2024,1,1),dt.date(2026,9,30))]
 def in_era(idx,lo,hi):
  if lo is None:return True
  bounds=dates_by_index.get(idx)
  return bool(bounds and lo<=bounds[1]<=hi)
 def emit(a,b,era,lag,ix,iy,lo,hi,lead_desc):
  keep=[(i,j) for i,j in zip(ix,iy) if in_era(i,lo,hi) and in_era(j,lo,hi)]
  x=[rmap[a][i] for i,j in keep];y=[rmap[b][j] for i,j in keep]
  ends=[(dates_by_index[i][1].isoformat(),dates_by_index[j][1].isoformat()) for i,j in keep if i in dates_by_index and j in dates_by_index]
  out.append({'crop_a':a,'crop_b':b,'era':era,'lag_periods':lag,'lead_direction':lead_desc,'n':len(x),
   'pearson':float(np.corrcoef(x,y)[0,1]) if len(x)>2 else None,
   'spearman':float(spearmanr(x,y).statistic) if len(x)>2 else None,
   'first_pair_ends':ends[0] if ends else None,'last_pair_ends':ends[-1] if ends else None,
   'support':'valid >=3-price-day target dekads; consecutive calendar index only; revised retrospective snapshot, no known first-publication time'})
 for a,b in combinations(series,2):
  amap=rmap[a];bmap=rmap[b]
  for era,lo,hi in eras:
   same=sorted(set(amap)&set(bmap));emit(a,b,era,0,same,same,lo,hi,'contemporaneous')
   for lag in (1,2,3):
    # A(t) against B(t+k), then independently B(t) against A(t+k).
    ix=sorted(i for i in amap if i+lag in bmap);emit(a,b,era,lag,ix,[i+lag for i in ix],lo,hi,f'{a} leads {b}')
    ix=sorted(i for i in bmap if i+lag in amap);emit(b,a,era,lag,ix,[i+lag for i in ix],lo,hi,f'{b} leads {a}')
 return out

def run(out=OUT):
 """Generate read-only static diagnostics. Call only after reviewer clearance."""
 import pandas as pd
 manifest=verify_inputs(); raw,series=load_source(); score_report=json.loads((V5/'score-report.json').read_text(encoding='utf8'));dev_report=json.loads((V5/'development-score.json').read_text(encoding='utf8'));g_frame=read_frame(V5/'forecasts.parquet');p_frame=read_frame(V3/'forecasts_test.parquet');grid_support=verify_forecast_grids(g_frame,p_frame);rows=build_metric_rows(g_frame,p_frame,series,require_planned_grid=True); dev_g=read_frame(V5/'development-forecasts.parquet');dev_p=read_frame(V3/'forecasts.parquet');devrows=build_development_rows(dev_g,dev_p,series);devmetrics=metric_table(devrows);devrecon=model_reconciliation(devrows,('P','G-B','L-L'));test_check=validate_saved_score(rows,score_report);year_check=validate_yearly_score(rows,score_report);dev_check=validate_saved_score(devrows,dev_report,('P','G-B','L-L'));mrows=metric_table(rows); xrows=cross_table(series); annual=annual_table(rows); errors=error_table(rows);dev_errors=error_table(devrows,split='development');groups,cuts=asof_groups(rows,development_rows=devrows); fit_audit=audit_fit_contract(series);fit_support=load_fit_support();profile=support_profiles(series);crows=profile['crop_distribution']
 out=Path(out); (out/'tables').mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(parents=True,exist_ok=True)
 tails=tail_table(rows);dev_tails=tail_table(devrows,split='development');compression=compression_table(rows);dev_compression=compression_table(devrows,split='development');compression_plot=compression_table(rows,by_year=False);responses=actual_response_table(rows);dev_responses=actual_response_table(devrows,split='development');dirs=direction_table(rows);dev_dirs=direction_table(devrows,split='development');reconciliation=model_reconciliation(rows)
 published={'B0':(221.2328,15.7323),'P':(225.4959,14.3279),'G-B':(212.8862,13.4780),'L-L':(233.7344,14.4567)}
 for r in reconciliation:
  mae,mape=published[r['model']]
  if abs(r['MAE8']-mae)>.00011 or abs(r['MAPE8_pct']-mape)>.00011:
   raise ValueError(f"blocked_integrity: {r['model']} reconstructed MAE8/MAPE8 {r['MAE8']:.6f}/{r['MAPE8_pct']:.6f} differs from published rounded {mae}/{mape}")
 dev_p=next(x for x in devrecon if x['model']=='P');dev_g=next(x for x in devrecon if x['model']=='G-B')
 if abs(dev_p['MAPE8_pct']-17.3745)>.00011 or abs(dev_g['MAPE8_pct']-18.2400)>.00011:
  raise ValueError(f"blocked_integrity: saved development P/G-B MAPE8 {dev_p['MAPE8_pct']:.6f}/{dev_g['MAPE8_pct']:.6f} differs from published 17.3745/18.2400")
 write_csv(out/'tables/metric_reconciliation.csv',mrows+[{'split':'test','model':r['model'],'crop':'ALL','horizon':'8-cell equal weight','n':len(rows),'MAE8':r['MAE8'],'MAPE8_pct':r['MAPE8_pct']} for r in reconciliation]+[{'split':'development','model':r['model'],'crop':'ALL','horizon':'8-cell equal weight','n':len(devrows),'MAE8':r['MAE8'],'MAPE8_pct':r['MAPE8_pct']} for r in devrecon]);write_csv(out/'tables/error_decomposition.csv',errors+dev_errors+groups+dirs+dev_dirs+tails+dev_tails+compression+dev_compression+responses+dev_responses);write_csv(out/'tables/annual_crop_comparison.csv',annual);write_csv(out/'tables/crop_distribution.csv',crows);write_csv(out/'tables/cross_crop_relationships.csv',xrows)
 support_rows=[]
 for c in series:
  product=next((p for p in raw.get('products',[]) if p.get('id')==c),{});diag=product.get('diagnostics') or {}
  support_rows.append({'diagnostic_type':'price_support','crop':c,'observed_days':len(series[c]),'first':min(series[c]).isoformat(),'last':max(series[c]).isoformat(),
   'market_query_status':json.dumps(diag.get('marketQueryStatusCounts'),ensure_ascii=False),'selected_response_states':json.dumps(diag.get('selectedResponseStates'),ensure_ascii=False),
   'latest_attempt_states':json.dumps(diag.get('latestAttemptStates'),ensure_ascii=False),'incompleteMarketDays':json.dumps(diag.get('incompleteMarketDays'),ensure_ascii=False),
   'missing_or_incomplete_queries':diag.get('missingOrIncompleteQueries'),'empty_response_count':diag.get('emptyResponseCount'),
   'negative_rows_retained':diag.get('negativeRowsRetained'),'nonpositive_price_days':diag.get('nonpositivePriceDays'),
   'source_market_scope':json.dumps(product.get('markets'),ensure_ascii=False),'publication_time_known':False})
 for crop in proxy.CROPS:
  training=[r for r in fit_audit['cropOriginSupportRows'] if r['crop']==crop]
  test_crop=[r for r in rows if r['crop']==crop];dev_crop=[r for r in devrows if r['crop']==crop]
  support_rows.append({'diagnostic_type':'train_eval_crop_support','crop':crop,'train_rows':sum(r['training_rows'] for r in training),'unique_crop_target_sum_across_origins':sum(r['unique_crop_target'] for r in training),'duplicate_rows_sum_across_origins':sum(r['duplicate_rows'] for r in training),'fit_origin_cell_rows':len(training),'eval_forecasts':len(test_crop),'eval_unique_crop_target_periods':len({(r['targetStart'],r['targetEnd']) for r in test_crop}),'development_forecasts':len(dev_crop),'development_unique_crop_target_periods':len({(r['targetStart'],r['targetEnd']) for r in dev_crop}),'independence_note':'training sums repeat history at each fit origin; forecast rows repeat targets across horizons'})
 support_rows += [{'diagnostic_type':'fit_ledger_summary','crop':'ALL','observed_days':fit_support['completed_fits'],'market_query_status':json.dumps(fit_support,ensure_ascii=False)}]+[{'diagnostic_type':'training_origin_support',**r} for r in fit_audit['originSupportRows']]
 annual_support=annual_data_support(fit_audit['cropOriginSupportRows'],devrows,rows)
 support_rows += [{'diagnostic_type':'annual_train_eval_crop_support',**r} for r in annual_support]
 write_csv(out/'tables/data_support.csv',support_rows)
 write_csv(out/'tables/annual_data_support.csv',annual_support)
 write_csv(out/'tables/cross_crop_relationships.csv',xrows)
 for key,filename in (('daily_price_returns','daily_price_returns.csv'),('dekad_price_returns','dekad_price_returns.csv'),('common_dekad_returns','common_dekad_returns.csv'),('annual_return_distribution','annual_return_distribution.csv'),('seasonal_pattern','seasonal_pattern.csv'),('annual_volume_support','annual_volume_support.csv'),('dekad_support','dekad_support.csv')):
  write_csv(out/'tables'/filename,profile[key])
 # Figures are rendered only from the explicitly computed, auditable tables.
 from .period_crop_profile_figures import create_period_crop_profile_figures
 metric_plot=mrows
 # Renderer expects one pooled row per crop/horizon/feature/group; retain
 # year-specific groups in the CSV/analysis summary and collapse only for plot.
 ag={}
 for r in groups:
  if r['split']!='test':continue
  k=(r['crop'],r['horizon'],r['feature'],r['group'])
  z=ag.setdefault(k,{'diagnostic_type':r['diagnostic_type'],'crop':r['crop'],'horizon':r['horizon'],'feature':r['feature'],'group':r['group'],'n_forecasts':0,'unique_targets':0,'_parts':[]})
  z['n_forecasts']+=r['n_forecasts'];z['unique_targets']+=r['unique_targets'];z['_parts'].append(r)
 asof_plot=[]
 for z in ag.values():
  n=z['n_forecasts']
  for m in ('P','B0','G-B','L-L'):
   z[m+'_MAE']=sum((x[m+'_MAE'] or 0)*x['n_forecasts'] for x in z['_parts'])/n if n else None
   z[m+'_MAPE_pct']=sum((x[m+'_MAPE_pct'] or 0)*x['n_forecasts'] for x in z['_parts'])/n if n else None
  z.pop('_parts');asof_plot.append(z)
 realized_plot=[]
 for c in proxy.CROPS:
  for h in proxy.HORIZONS:
   for group in ('realized_abs_change_lt20','realized_abs_change_ge20'):
    parts=[r for r in errors if r['crop']==c and r['horizon']==h and r['group']==group]
    n=sum(r['n_forecasts'] for r in parts)
    rec={'diagnostic_type':'realized_regime','crop':c,'horizon':h,'group':group,'n_forecasts':n,'unique_targets':sum(r['unique_targets'] for r in parts),'feature':None}
    for m in ('P','G-B'):
     rec[m+'_MAPE_pct']=sum((r[m+'_MAPE_pct'] or 0)*r['n_forecasts'] for r in parts)/n if n else None
    realized_plot.append(rec)
 regime_plot=realized_plot+asof_plot
 comp_rows=compression
 plot_tables={
  'crop_distribution':pd.DataFrame(profile['crop_distribution']),
  'daily_price_returns':pd.DataFrame(profile['daily_price_returns']),
  'dekad_price_returns':pd.DataFrame(profile['dekad_price_returns']),
  'seasonal_pattern':pd.DataFrame(profile['seasonal_pattern']),
  'annual_volume_support':pd.DataFrame(profile['annual_volume_support']),
  'metric_reconciliation':pd.DataFrame(metric_plot),
  'annual_crop_comparison':pd.DataFrame(annual),
  'prediction_z':pd.DataFrame([{k:r.get(k) for k in ('crop','horizon','year','actual_z','P_z','G-B_z')} for r in rows]),
  'error_decomposition':pd.DataFrame(compression_plot),
  'regime_plot_support':pd.DataFrame(regime_plot),
  'cross_crop_relationships':pd.DataFrame(xrows),
  'data_support':pd.DataFrame([r for r in support_rows if r.get('diagnostic_type')=='train_eval_crop_support']),
  'annual_data_support':pd.DataFrame(annual_support),
 }
 # Adapt explicit aggregate columns while preserving repeated-fit semantics.
 plot_tables['data_support']['unique_crop_target']=plot_tables['data_support']['unique_crop_target_sum_across_origins']
 plot_tables['data_support']['duplicate_rows']=plot_tables['data_support']['duplicate_rows_sum_across_origins']
 plot_tables['data_support']['eval_unique_targets']=plot_tables['data_support']['eval_unique_crop_target_periods']
 plot_tables['data_support']=pd.DataFrame([({**r,'unique_crop_target':r.get('unique_crop_target',r.get('unique_crop_target_sum_across_origins',0)), 'eval_unique_targets':r.get('eval_unique_targets',r.get('eval_unique_crop_target_periods',0))}) for r in plot_tables['data_support'].to_dict('records')])
 figures=create_period_crop_profile_figures(plot_tables,out/'figures')
 reconciliation=model_reconciliation(rows)
 if fit_support['completed_fits']!=488: raise ValueError(f"blocked_integrity: V5 completed fit count {fit_support['completed_fits']} != 488")
 # Detect mutation during analysis as well as stale inputs before it starts.
 if verify_inputs()!=manifest: raise ValueError('blocked_integrity: an input changed during diagnostics')
 fallbacks={m:sum(1 for r in rows if r.get(m+'_status') not in (None,'ok','selected','saved_prior_selected_B0','saved_prior_quarterly_F')) for m in ('P','G-B','L-L')}
 checks={'status':'verified_reuse','model_fits':0,'new_predictions':0,'external_requests':0,'forecastGridSupport':grid_support,'pairedRows':len(rows),'developmentPlannedRows':576,'developmentPairedRows':len(devrows),'developmentReconciliation':devrecon,'developmentScoreCheck':dev_check,'testScoreCheck':test_check,'yearlyScoreCheck':year_check,'sameKeyRows':len(rows),'uniqueTargetPeriods':len({(r['targetStart'],r['targetEnd']) for r in rows}),'uniqueCropTargetPeriods':len({(r['crop'],r['targetStart'],r['targetEnd']) for r in rows}),'cells':8,'annualCellRows':len(annual),'asOfDiagnosticGroups':len(groups),'tercileCutpoints':cuts,'reconciliation':reconciliation,'publishedRoundedMetricsMatch':True,'directionRows':len(dirs),'fitSupport':fit_support,'fitAudit':{'ledgerEntries':fit_audit['ledgerEntries'],'completeFits':fit_audit['completeFits'],'uniqueOrigins':fit_audit['uniqueOrigins'],'contractChecked':fit_audit['contractChecked'],'weightsChecked':fit_audit['weightsChecked']},'savedFallbackStatusCounts':fallbacks,'savedDevelopmentB0Aggregate':dev_report.get('metrics',{}).get('B0'),'protocolSha256':sha(PROTOCOL),'requestSha256':sha(REQUEST)}
 (out/'preflight.json').write_text(json.dumps({'checks':checks,'inputs':manifest},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 (out/'source_manifest.json').write_text(json.dumps({'inputs':manifest,'requestSha256':sha(REQUEST),'protocolSha256':sha(PROTOCOL),'snapshotMeta':raw.get('meta',{})},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 (out/'checks.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 # Compact, data-derived evidence for the nine decision questions in request.md.
 by_model={r['model']:r for r in reconciliation}
 by_year={(r['year'],r['crop'],r['horizon']):r for r in annual}
 test_2026=[r for r in rows if r['year']==2026]
 y26_recon=model_reconciliation(test_2026)
 by26={r['model']:r for r in y26_recon}
 year_recon=[]
 for year in sorted({r['year'] for r in rows}):
  yr=[r for r in rows if r['year']==year]
  year_recon.append({'year':year,**{k:v for m in model_reconciliation(yr) for k,v in ((m['model']+'_MAE8',m['MAE8']),(m['model']+'_MAPE8_pct',m['MAPE8_pct']))}})
 cell_gap=[]
 for r in mrows:
  signed=(float(r['G-B_MAPE_pct'])-10.)/8
  cell_gap.append({'crop':r['crop'],'horizon':r['horizon'],'G_B_MAPE_pct':r['G-B_MAPE_pct'],'G_B_MAE':r['G-B_MAE'],'signed_contribution_to_macro_minus_10_pp':signed,'positive_excess_above_10_pp':max(0.,r['G-B_MAPE_pct']-10.)/8,'cell_MAE_contribution_to_MAE8':r['G-B_MAE']/8})
 crop_scenarios=[]
 for c in proxy.CROPS:
  cr=[r for r in mrows if r['crop']==c]
  other=[r for r in mrows if r['crop']!=c]
  base=by_model['G-B']['MAPE8_pct']
  perfect=(sum(r['G-B_MAPE_pct'] for r in other)+0.)/8
  half=(sum(r['G-B_MAPE_pct'] for r in other)+sum(r['G-B_MAPE_pct']/2 for r in cr))/8
  to10=(sum(r['G-B_MAPE_pct'] for r in other)+sum(min(r['G-B_MAPE_pct'],10.) for r in cr))/8
  crop_scenarios.append({'crop':c,'current_crop_cell_mape_sum':sum(r['G-B_MAPE_pct'] for r in cr),'current_crop_macro_contribution_pp':sum(r['G-B_MAPE_pct'] for r in cr)/8,'to_zero_cells_hypothetical_MAPE8':perfect,'to_half_cells_hypothetical_MAPE8':half,'cap_cells_at_10_hypothetical_MAPE8':to10,'could_perfect_one_crop_alone_reach_10':perfect<=10.,'could_cap_one_crop_cells_at_10_alone_reach_10':to10<=10.,'MAE8_if_MAE_unchanged':by_model['G-B']['MAE8'],'MAE_protection_note':'These MAPE-only arithmetic scenarios do not imply an MAE change; any candidate must report MAE protection separately.'})
 if abs(sum(r['signed_contribution_to_macro_minus_10_pp'] for r in cell_gap)-(by_model['G-B']['MAPE8_pct']-10.))>1e-10:raise ValueError('internal arithmetic: signed cell contributions do not sum to macro gap')
 profile_comparison=[]
 for c in proxy.CROPS:
  pr=next(x for x in crows if x['crop']==c)
  profile_comparison.append({'crop':c,'observed_price_days':pr['observed_days'],'valid_dekads':pr['valid_dekads'],'price_median':pr.get('price_median'),'price_sd':pr.get('price_sd'),'volume_positive_days':pr['positive_volume_days'],'common_price_days':pr.get('common_price_days'),'common_valid_dekads':pr.get('common_valid_dekads'),'acf_price_lag_1':pr.get('acf_price_lag_1'),'acf_logdiff_lag_1':pr.get('acf_logdiff_lag_1')})
 analysis_summary={'status':'verified_reuse','scope':'retrospective descriptive diagnostics; no fit, new forecast, API/DB call or inference',
  'integrity':{'test_paired_rows':len(rows),'same_keys':True,'test_score':test_check,'yearly_score':year_check,'development_planned_rows':576,'development_actual_supported_rows':len(devrows),'development_cell_counts':{f"{c}:{h}":sum(r['crop']==c and r['horizon']==h for r in devrows) for c in proxy.CROPS for h in proxy.HORIZONS},'source_manifest':manifest},
  'q1_price_volume_season_support':{'crop_profiles':profile_comparison,'exact_common_days':profile['common_date_count'],'exact_common_periods':profile['common_period_count'],'common_period_first':profile['common_period_first'],'common_period_last':profile['common_period_last']},
  'q2_temporal_structure':{'within_year_season_rows':len(profile['seasonal_pattern']),'dekad_return_rows':len(profile['dekad_price_returns']),'common_dekad_return_rows':len(profile['common_dekad_returns']),'annual_return_distribution':profile['annual_return_distribution'],'daily_return_rows':len(profile['daily_price_returns']),'horizon_actual_response_by_year':responses,'development_actual_response_by_year':dev_responses,'acf_lags':[1,2,3,6,12,36]},
  'q3_relative_model_performance':{'test_8cell':reconciliation,'development_8cell':devrecon,'test_years':year_recon,'cell_deltas':annual},
  'q4_crop_horizon_heterogeneity':{'cells':mrows,'annual_cell_deltas':annual,'development_error_groups':dev_errors,'test_error_groups':errors,'development_feature_groups':[r for r in groups if r['split']=='development'],'test_feature_groups':[r for r in groups if r['split']=='test']},
  'q5_2026_recent_change':{'8cell':by26,'annual_crop_horizon': [r for r in annual if r['year']==2026]},
  'q6_response_compression_and_tails':{'compression_by_year':compression,'compression_all_test':compression_plot,'development_compression':dev_compression,'realized_z_tail_by_year':tails,'development_realized_z_tail':dev_tails,'direction_conditional_by_year':dirs,'development_direction_conditional':dev_dirs,'realized_regime_error_budget_by_year':errors},
  'q7_pooling_support_and_cross_crop':{'fit_summary':fit_support,'fit_contract_audit':{'complete_fits':fit_audit['completeFits'],'unique_origins':fit_audit['uniqueOrigins'],'contract_checked':fit_audit['contractChecked'],'weights_checked':fit_audit['weightsChecked']},'per_crop_unique_history_by_fit_origin':fit_audit['cropOriginSupportRows'],'per_crop_horizon_origin_support':fit_audit['originSupportRows'],'cross_crop_relationships':xrows},
  'q8_remaining_10pct_gap':{'G_B_MAPE8_pct':by_model['G-B']['MAPE8_pct'],'remaining_pp':by_model['G-B']['MAPE8_pct']-10.,'cells_at_or_below_10':sum(r['G-B_MAPE_pct']<=10 for r in mrows),'signed_cell_gap_contributions_pp':cell_gap,'single_crop_arithmetic_scenarios':crop_scenarios,'one_crop_can_alone_hit_10':any(r['could_perfect_one_crop_alone_reach_10'] for r in crop_scenarios),'MAE8_current':by_model['G-B']['MAE8'],'MAPE_scenarios_do_not_imply_MAE':True},
  'q9_followup_question':'개발구간에서 발행 전 관측지원/변동성 신호가 반복적으로 급변 오차를 구분하는가? 후속 고정 프로토콜에서 해당 정보의 추가 가치를 반증 가능하게 검정할 것; 이번 진단에서는 실행하지 않음.',
  'inference_reused_without_new_tests':{'V5_inference':score_report.get('inference'),'V5_holm_adjusted_p':score_report.get('holmAdjustedP'),'V5_block_bootstrap':score_report.get('bootstrap')},
  'limits':['retrospective revised snapshot has no proven first-publication time','common-pool correlation does not establish causality or explain G-B gains','no new significance inference; prior V5 inferential results remain as reported','forecast rows share repeated dekad targets and are not independent observations']}
 (out/'analysis_summary.json').write_text(json.dumps(analysis_summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 text='''# 00 요약\n\n가락·강서 대리 순별 목표이며 공식 순별 가격이 아닙니다. 2024~2026 반복 검토 후향자료입니다. 새 적합·추론·API 호출은 0회입니다.\n\nG-B 13.4780%, P 14.3279%로 보고되었으며 본 산출물은 저장 예측 공통 780행을 재검산합니다. 전체 10% 목표까지 3.4780%p이며, 현재 3/8 셀만 10% 이하입니다. 2026년 P 11.3762%, G-B 11.6110%로 G-B가 악화했습니다.\n\n해석은 설명적 기술통계입니다. 작물간 상관은 인과나 G-B의 원인이 아니며 G와 L의 차이는 손실·규제·전처리 차이를 포함합니다.\n\n후속 질문 하나: 개발구간에서 발행시점에 이용 가능한 변동성/관측지원 지표가 급변 오차를 반복적으로 구분하는지, 고정 프로토콜로 별도 검증할 가치가 있는가? 이번에는 실행하지 않습니다.\n'''
 (out/'00_executive_summary.md').write_text(text,encoding='utf8')
 def mdtable(headers,body):
  return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(str(v) for v in row)+' |' for row in body)+'\n'
 a_body=[[r['crop'],r['first_date'],r['last_date'],r['observed_days'],round(r.get('price_median',0),2),round(r.get('price_sd',0),2),r['adjacent_day_pairs']] for r in crows]
 (out/'01_crop_price_distribution.md').write_text('# A. 가격·거래량 분포와 시간 특성\n\n일별 양수 가격을 사용했습니다. 관측되지 않은 날짜/순은 채우지 않았고 일 변화는 실제 인접 날짜에서만 계산했습니다. 거래량은 양수 중량 관측의 기술통계이며 시장별 비율을 의미하지 않습니다.\n\n'+mdtable(['품목','최초','최종','관측일','가격 중앙값 (원/kg)','가격 SD','인접일 변화쌍'],a_body)+'\n연중 상대가격은 연도별 로그가격 중앙값을 제거한 사후 기술통계입니다. ACF는 별도 확증검정이 아니며 자료 공백을 제거한 lag 추정은 하지 않습니다. 표와 그림의 지원을 참고하세요. 시장 조회상태와 거래기여는 구분되며, 공개시각은 확인되지 않았습니다.\n',encoding='utf8')
 (out/'02_post_v5_error_diagnostics.md').write_text('# B. 저장 예측 오차·반응 진단\n\n같은 목표 정답을 원래 일별 양수 가격 3일 이상으로 재구성했습니다. 지표는 품목×거리 셀별로 먼저 계산한 후 8셀 동일 가중합니다. 그룹은 사후 설명용이며 예측 규칙이 아닙니다.\n\n'+mdtable(['모형','동일쌍','MAE8 (원/kg)','MAPE8 (%)'],[[x['model'],x['paired_rows'],round(x['MAE8'],4),round(x['MAPE8_pct'],4)] for x in reconciliation])+'\n2026년의 방향은 품목별로 다르며 전체 악화를 모든 품목에 일반화하지 않습니다. 사후 급변은 |actual/p_ref−1|≥20%로 분류했습니다. 발행 전 연속 특징의 삼분위 경계는 개발 원점 중복 제거 자료에서만 얻었습니다. 반응폭 그림은 실제와 저장 예측의 log(price/p_ref) 분포를 나타냅니다.\n\n'+mdtable(['품목','순','연도','G−P MAPE pp','G−P MAE'],[[x['crop'],x['horizon'],x['year'],round(x['G_minus_P_MAPE_pp'],3),round(x['G_minus_P_MAE'],2)] for x in annual if x['year']>=2024])+'\n급변 및 사전 가용 특징별 전체 결과는 tables/error_decomposition.csv에 있습니다. 낮은 표본 그룹과 반복 목표순은 독립 표본으로 해석하지 않습니다. MAPE 10% 격차는 관측된 오차 기여의 산술 분해이며 실현 보장이 아닙니다.\n',encoding='utf8')
 (out/'03_cross_crop_and_pooling_readiness.md').write_text('# C. 작물간 동행·선행과 공동학습 지원\n\n상관은 같은 실제 관측 달력순에서 계산했습니다. lag는 달력 순 인덱스로 정렬하고 빈 순을 당겨 붙이지 않았습니다. 상관은 인과나 예측력 증거가 아닙니다. V5는 작물간 가격 시계열을 직접 입력으로 쓰지 않고 학습행을 함께 사용했습니다.\n\n'+mdtable(['품목 A','품목 B','lag 순','n','Pearson r'],[[x['crop_a'],x['crop_b'],x['lag_periods'],x['n'],round(x['pearson'],3) if x['pearson'] is not None else 'NA'] for x in xrows if x['lag_periods']<=1])+'\n가격 규모·이력·변동폭의 차이는 하나의 모델에서 지원 가중치와 예측 압축 위험을 만들 수 있다는 가설입니다. 이번 기술통계만으로 원인을 분리하거나 G−L 차이를 순수 비선형성 효과로 볼 수 없습니다. 다음 시험은 개발구간에서 발행 전 변동/지원 신호가 급변 오차를 반복 식별하는지 확인하는 정보가치 평가를 우선하되 별도 사전 프로토콜과 승인이 필요합니다.\n',encoding='utf8')
 return checks

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT));p.add_argument('--run',action='store_true');a=p.parse_args()
 if not a.run: raise SystemExit('Use --run only after reviewer clearance')
 print(json.dumps(run(a.out),ensure_ascii=False))
