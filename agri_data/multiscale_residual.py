"""Causal multiscale residual experiment; forecast and scoring are separate phases."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math, os, uuid
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from . import dashboard, direct_horizon as dh

ROOT=Path(__file__).resolve().parents[1]
PRICES=ROOT/"outputs/daily/2026-10-05/20261005T091737-a9f53eec/prices.json"
DIRECT=ROOT/"outputs/experiments/direct-horizon-20261006/report.json"
BLEND=ROOT/"outputs/experiments/ridge-sarima-blend-20261008/forecast-manifest.json"
BLEND_REPORT=ROOT/"outputs/experiments/ridge-sarima-blend-20261008/report.json"
PRICES_SHA="f525c10d79534896ff99784ffc1725fc7141ab8428781aba65181b73f4112cc8"
DIRECT_SHA="a1369ad7968f99e17ce4f3c0b94221871eee665c6216dbfe6522a9c6e96c4e67"
BLEND_SHA="e73cd6b3c47b163000d020dbf60373af67d4edb79309af3a91b349167d945d40"
BLEND_REPORT_SHA="423c70ce3c34dc4eaf80c3b3fb3367fa6eedc73745d2098bf66c8f62c1a5a822"
TRAIN_END=dt.date(2025,8,31); EVAL_START=dt.date(2025,9,25); EVAL_END=dt.date(2026,9,24)
HORIZONS=tuple(range(1,11)); PRODUCTS=tuple(dashboard.PRODUCTS); BANDWIDTHS=(9,28)
METHODS=("candidate","ablation","direct","blend","persistence")

def sha(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def canonical(o):return sha(json.dumps(o,sort_keys=True,separators=(",",":"),allow_nan=False).encode())
def key(row):
 p=row.get('product'); h=row.get('horizon'); t=dt.date.fromisoformat(row['target']); origin=dt.date.fromisoformat(row['origin'])
 if p not in PRODUCTS or isinstance(h,bool) or h not in HORIZONS or t-origin!=dt.timedelta(days=h):raise ValueError('invalid forecast identity')
 return p,origin,t,h

EVAL_PAIR_COUNT=12_508

def _features(series,origin,target,h,season=True):
 hist=sorted((d,v[0]) for d,v in series.items() if d<=origin and v[0]>0 and math.isfinite(v[0]))
 if not hist:return None
 d0,p0=hist[-1]; lp=math.log(p0); ages=[(origin-d).days for d,_ in hist]
 vals=[]
 for window in (7,28,56):
  vs=[math.log(p) for d,p in hist if 0<=(origin-d).days<window]
  vals += [float(np.mean(vs)-lp) if vs else 0.,float(len(vs)),float(np.std(vs)) if len(vs)>1 else 0.]
 recent=[math.log(p) for d,p in hist if (origin-d).days<=7]
 long=[math.log(p) for d,p in hist if (origin-d).days<=56]
 vals += [float(recent[0]-recent[-1]) if len(recent)>1 else 0.,float(long[0]-long[-1]) if len(long)>1 else 0.,float((origin-d0).days),float(h)]
 wd=target.weekday()*2*math.pi/7; vals += [math.sin(wd),math.cos(wd)]
 if season:
  yd=target.timetuple().tm_yday*2*math.pi/(366 if __import__('calendar').isleap(target.year) else 365)
  vals += [math.sin(yd),math.cos(yd)]
 return vals

def _pairs(report, include_all_actuals=True):
 rows=report.get('pairedRows'); out={}
 if not isinstance(rows,list):raise ValueError('direct pairedRows missing')
 for r in rows:
  k=(r.get('product'),dt.date.fromisoformat(r['target'])-dt.timedelta(days=int(r['horizon'])),dt.date.fromisoformat(r['target']),int(r['horizon']))
  if k in out:raise ValueError('duplicate direct identity')
  if r.get('product') not in PRODUCTS or isinstance(r.get('horizon'), bool) or int(r.get('horizon', 0)) not in HORIZONS:raise ValueError('invalid direct identity')
  try:
   direct=float(r['direct']); persistence=float(r['persistence'])
   actual=float(r['actual']) if include_all_actuals or dt.date.fromisoformat(r['target'])<=TRAIN_END else None
  except (KeyError,ValueError,TypeError):continue
  if math.isfinite(direct) and direct>0 and math.isfinite(persistence) and persistence>0 and (actual is None or math.isfinite(actual) and actual>0):out[k]={**r,'actual':actual,'direct':direct,'persistence':persistence}
 return out

def _source_rows(doc):
 rows={}
 for r in doc.get('forecasts',[]):
  if 'actual' in r or 'targetValue' in r:raise ValueError('source manifest has answers')
  k=key(r)
  if k in rows:raise ValueError('duplicate blend identity')
  pred=r.get('predictions',{}); b=float(pred['blend'])
  if not math.isfinite(b) or b<=0:raise ValueError('invalid blend')
  rows[k]=b
 return rows

def build_forecast(prices_raw:bytes,direct_raw:bytes,blend_raw:bytes)->dict[str,Any]:
 if sha(prices_raw)!=PRICES_SHA or sha(direct_raw)!=DIRECT_SHA or sha(blend_raw)!=BLEND_SHA:raise ValueError('pinned input SHA mismatch')
 raw=json.loads(prices_raw); report=json.loads(direct_raw); blend=json.loads(blend_raw)
 series=dh._observations(raw); pairs=_pairs(report, include_all_actuals=False); blends=_source_rows(blend)
 train={p:[] for p in PRODUCTS}; evaluation=[]
 for k,r in pairs.items():
  p,o,t,h=k
  if p not in series or t-o!=dt.timedelta(days=h):continue
  f=_features(series[p],o,t,h,True)
  item=(k,r,f)
  if t<=TRAIN_END and o<t and f is not None and r['actual']>0 and r['direct']>0:train[p].append(item)
  elif EVAL_START<=t<=EVAL_END and k in blends:evaluation.append(item)
 if len(evaluation)!=EVAL_PAIR_COUNT:raise ValueError(f'evaluation pair count mismatch:{len(evaluation)}')
 predictions=[]
 for p in PRODUCTS:
  tr=train[p]
  if len(tr)<30:raise ValueError(f'insufficient training rows:{p}')
  # Stable order is key order and both variants use identical rows and estimator settings.
  X=np.asarray([x[2] for x in tr]); y=np.log([x[1]['actual']/x[1]['direct'] for x in tr])
  for season,name in ((True,'candidate'),(False,'ablation')):
   cols=X if season else X[:,:-2]
   estimator=make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(24,12),alpha=10.,random_state=1729,max_iter=300,early_stopping=False))
   estimator.fit(cols,y)
   for k,r,feat in evaluation:
    if k[0]!=p:continue
    if feat is None:pred=r['direct']
    else:
     ff=np.asarray(feat if season else feat[:-2]).reshape(1,-1)
     correction=float(np.clip(estimator.predict(ff)[0],-.10,.10))
     pred=r['direct']*math.exp(correction)
    predictions.append((k,name,pred))
 by={}; fallback_keys={k for k,_,f in evaluation if f is None}
 for k,n,v in predictions:by.setdefault(k,{})[n]=v
 output=[]
 for k in sorted(by):
  directrow=pairs[k]; output.append({'product':k[0],'origin':k[1].isoformat(),'target':k[2].isoformat(),'horizon':k[3],
   'predictions':{'candidate':by[k]['candidate'],'ablation':by[k]['ablation'],'direct':directrow['direct'],'blend':blends[k],'persistence':directrow['persistence']},'lineage':{'fallback':k in fallback_keys,'fallbackReason':'no_valid_asof_price_history' if k in fallback_keys else None}})
 doc={'schemaVersion':1,'protocol':'multiscale-residual-mlp-2026-10-08','phase':'forecast_only','targetValuesAttached':False,
  'inputSha256':{'prices':PRICES_SHA,'direct':DIRECT_SHA,'blend':BLEND_SHA},'codeSha256':sha(Path(__file__).read_bytes()),'trainingTargetEnd':TRAIN_END.isoformat(),
  'features':'causal 7/28/56-day log-price deviations, changes/count/volatility/source age, horizon, target weekday; candidate adds annual sine/cosine',
  'model':'per-crop StandardScaler + MLPRegressor(24,12; alpha=10; seed=1729; max_iter=300); log(actual/direct), clipped +/-0.10',
  'trainingRowsByProduct':{p:len(train[p]) for p in PRODUCTS},'fallbackCount':sum(r['lineage']['fallback'] for r in output),'forecasts':output}
 doc['manifestSha256']=canonical(doc);return doc

def _verify_manifest(m,raw):
 clean=dict(m); saved=clean.pop('manifestSha256',None)
 if m.get('phase')!='forecast_only' or m.get('targetValuesAttached') is not False:raise ValueError('manifest phase invalid')
 if m.get('codeSha256')!=sha(Path(__file__).read_bytes()):raise ValueError('manifest code hash mismatch')
 if saved!=canonical(clean) or sha(raw)!=sha((json.dumps(m,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()):raise ValueError('manifest tampered')
 rows={}
 for r in m.get('forecasts',[]):
  if 'actual' in r:raise ValueError('answer leakage')
  k=key(r)
  if k in rows:raise ValueError('duplicate manifest identity')
  if set(r.get('predictions',{}))!=set(METHODS):raise ValueError('manifest methods invalid')
  for value in r['predictions'].values():
   if not math.isfinite(float(value)) or float(value)<=0:raise ValueError('invalid manifest prediction')
  rows[k]=r
 return rows

def _lossdiff(rows,a,b,kind):
 out=[]
 for k,r in rows.items():
  actual=r['actual']; pa=r['predictions'][a]; pb=r['predictions'][b]
  va=abs(actual-pa); vb=abs(actual-pb)
  out.append((k[2],(va-vb) if kind=='ae' else (va-vb)/actual))
 return out

def _dm(diffs,lag):
 n=len(diffs); mean=sum(v for _,v in diffs)/n if n else None
 if n<30:return {'n':n,'status':'inference_unavailable_insufficient_pairs','statistic':None,'pValue':1.}
 c={d:v-mean for d,v in diffs}; var=sum(v*v for v in c.values())
 for lagday in range(1,lag+1):var+=2*(1-lagday/(lag+1))*sum(v*c.get(d-dt.timedelta(days=lagday),0.) for d,v in c.items())
 if not math.isfinite(var) or var<=0:return {'n':n,'status':'inference_unavailable_nonpositive_variance','statistic':None,'pValue':1.}
 z=mean/(math.sqrt(var)/n);return {'n':n,'status':'tested','statistic':z,'pValue':math.erfc(abs(z)/math.sqrt(2))}

def _holm(items):
 s=sorted(items,key=lambda x:x['pValue']); adj=0.
 for i,x in enumerate(s):adj=max(adj,min(1.,(len(s)-i)*x['pValue']));x['holmPValue']=adj;x['conclusion']=('improvement' if x['meanLossDifference']<0 else 'worsening') if x['status']=='tested' and adj<.05 else ('inconclusive' if x['status']=='tested' else 'inference_unavailable')

def score(manifest_raw:bytes,prices_raw:bytes,direct_raw:bytes,blend_raw:bytes,blend_report_raw:bytes)->dict[str,Any]:
 if sha(prices_raw)!=PRICES_SHA or sha(direct_raw)!=DIRECT_SHA or sha(blend_raw)!=BLEND_SHA or sha(blend_report_raw)!=BLEND_REPORT_SHA:raise ValueError('pinned input SHA mismatch')
 blend_report=json.loads(blend_report_raw)
 if blend_report.get('manifestSha256')!=json.loads(blend_raw).get('manifestSha256'):raise ValueError('blend report manifest lineage mismatch')
 m=json.loads(manifest_raw); forecasts=_verify_manifest(m,manifest_raw)
 expected=build_forecast(prices_raw,direct_raw,blend_raw)
 if m!=expected:raise ValueError('forecast manifest differs from deterministic reconstruction')
 pairs=_pairs(json.loads(direct_raw)); blends=_source_rows(json.loads(blend_raw)); obs=dh._observations(json.loads(prices_raw))
 actuals={}
 for p,series in obs.items():actuals.update({(p,d):v[0] for d,v in series.items() if v[0]>0 and math.isfinite(v[0])})
 common={}
 for k,r in forecasts.items():
  if k not in pairs or k not in blends:continue
  actual=pairs[k]['actual']; snap=actuals.get((k[0],k[2]))
  if snap is None or not math.isclose(snap,actual,rel_tol=1e-12,abs_tol=1e-12):raise ValueError('actual mismatch against frozen prices')
  pred=r['predictions']
  if not math.isclose(float(pred['direct']),pairs[k]['direct'],rel_tol=1e-12,abs_tol=1e-12) or not math.isclose(float(pred['blend']),blends[k],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('baseline prediction mismatch')
  common[k]={'actual':actual,'predictions':{**pred,'persistence':pairs[k]['persistence']}}
 if len(common)!=EVAL_PAIR_COUNT:raise ValueError(f'common pair count mismatch:{len(common)}')
 metrics={}
 for method in METHODS:
  cells={}
  for p in PRODUCTS:
   for h in HORIZONS:
    rows=[r for k,r in common.items() if k[0]==p and k[3]==h];n=len(rows)
    cells[f'{p}|{h}']={'n':n,'mae':sum(abs(r['actual']-r['predictions'][method]) for r in rows)/n if n else None,'mape':100*sum(abs(r['actual']-r['predictions'][method])/r['actual'] for r in rows)/n if n else None}
  crop_metrics={}
  for p in PRODUCTS:
   rows=[r for k,r in common.items() if k[0]==p]; n=len(rows)
   crop_metrics[p]={'n':n,'mae':sum(abs(r['actual']-r['predictions'][method]) for r in rows)/n if n else None,'mape':100*sum(abs(r['actual']-r['predictions'][method])/r['actual'] for r in rows)/n if n else None}
  metrics[method]={'cells':cells,'byProduct':crop_metrics,'macro50CellMae':sum(c['mae'] for c in cells.values())/50 if len(cells)==50 and all(c['mae'] is not None for c in cells.values()) else None,'macro50CellMape':sum(c['mape'] for c in cells.values())/50 if len(cells)==50 and all(c['mape'] is not None for c in cells.values()) else None,'coverage':len(common)}
 tests=[]
 for candidate in ('candidate',):
  for comparator in ('ablation','direct','blend','persistence'):
   for p in PRODUCTS:
    for h in HORIZONS:
     rows={k:r for k,r in common.items() if k[0]==p and k[3]==h}
     for loss in ('ae','ape'):
      diffs=_lossdiff(rows,candidate,comparator,loss)
      for bw in BANDWIDTHS:
       result=_dm(diffs,bw);tests.append({'candidate':candidate,'comparator':comparator,'product':p,'horizon':h,'loss':loss,'bandwidth':bw,'meanLossDifference':sum(v for _,v in diffs)/len(diffs) if diffs else None,**result})
 for bw in BANDWIDTHS:_holm([x for x in tests if x['bandwidth']==bw])
 grouped={}
 for item in tests:grouped.setdefault((item['candidate'],item['comparator'],item['product'],item['horizon']),[]).append(item)
 conclusions=[]
 for (candidate,comparator,product,horizon),items in grouped.items():
  states=[x['conclusion'] for x in items]
  direction='improvement' if len(states)==4 and all(x=='improvement' for x in states) else 'worsening' if len(states)==4 and all(x=='worsening' for x in states) else 'inference_unavailable' if 'inference_unavailable' in states else 'inconclusive'
  conclusions.append({'candidate':candidate,'comparator':comparator,'product':product,'horizon':horizon,'conclusion':direction})
 return {'protocol':m['protocol'],'manifestSha256':m['manifestSha256'],'inputSha256':{'prices':PRICES_SHA,'direct':DIRECT_SHA,'blendManifest':BLEND_SHA,'blendReport':BLEND_REPORT_SHA},'coverage':{'planned':EVAL_PAIR_COUNT,'manifestForecasts':len(forecasts),'commonScoredPairs':len(common),'fallbackForecasts':m.get('fallbackCount')},'metrics':metrics,'dmTests':tests,'cellConclusions':conclusions,
 'interpretation':'Retrospective, repeatedly inspected revised-price study. Nonsignificance is not equivalence and significance is not proof of independent or operational user benefit.'}

def immutable(path,doc):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex+'.tmp')
 try:
  with tmp.open('x') as f:f.write(json.dumps(doc,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
  os.link(tmp,path)
 finally:
  try:tmp.unlink()
  except FileNotFoundError:pass

def main(argv=None):
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=('forecast','score'));ap.add_argument('--prices',type=Path,default=PRICES);ap.add_argument('--direct',type=Path,default=DIRECT);ap.add_argument('--blend',type=Path,default=BLEND);ap.add_argument('--blend-report',type=Path,default=BLEND_REPORT);ap.add_argument('--manifest',type=Path,required=True);ap.add_argument('--report',type=Path)
 a=ap.parse_args(argv);p=a.prices.read_bytes();d=a.direct.read_bytes();b=a.blend.read_bytes()
 if a.phase=='forecast':immutable(a.manifest,build_forecast(p,d,b))
 else:
  if not a.report:ap.error('--report required for score phase')
  immutable(a.report,score(a.manifest.read_bytes(),p,d,b,a.blend_report.read_bytes()))
 return 0
if __name__=='__main__':raise SystemExit(main())
