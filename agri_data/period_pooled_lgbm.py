"""Pooled direct LightGBM V5; real stages require frozen protocol and reviewer clearance."""
from __future__ import annotations
import argparse, datetime as dt, hashlib, json, math, os, time
from pathlib import Path
from typing import Any
# The frozen run is single-threaded across LightGBM and BLAS. Set these before
# importing NumPy/scikit-learn so native runtimes see the limits at load time.
for _thread_var in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[_thread_var]='1'
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from . import period_proxy_experiment as proxy
from . import period_mape10 as v3
from . import period_mape10_score as v3score

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/experiments/period-pooled-direct-lgbm-v5-20261016'
RUN_ID='PERIOD_POOLED_DIRECT_LGBM_V5'
PROTOCOL=ROOT/'docs/period_pooled_lgbm_protocol.md'
PROTOCOL_SHA='8590883c606e77e4669cce2972623f6e3e97915621a7111a6e95872c314a2e69'
SOURCE_SHA='6ca1366d1dda618c96f48e62fea578193696b24dc893e68320bdeaae8e43e659'
PROXY_CODE_SHA='b6896eac8c43f1f324673e1a1251bd080469c517a80e99d051ceb50c7617fc1c'
V3_CODE_SHA='cabce6ad19104c6c71148b6c8ce1bd7c8b2efe941941deb2051ebb94405fb6f5'
V3_SCORE_CODE_SHA='ab2cc43784f4dc10509715c9c545ccc0e43a8706080a6dc6a5e021d19898c051'
V3_PROTOCOL_SHA='778de0205cea77591ff0e5b292fc427373e8b44d7d8ff2e9412755d074abae01'
PROXY_PROTOCOL_SHA='d4a8b94fe6a056e22fefc3f44576018e2fd049397165ea6f856b7b10e2b4a3d4'
V3_OUT=ROOT/'outputs/experiments/period-direct-mape-rolling-v3-20261015'
V4_OUT=ROOT/'outputs/experiments/period-seasonal-relative-analog-v4-20261016'
V3_FILES={'forecasts.parquet':'9bf20704883f396d13fcd5583eb1c9248a8f720ad51013723e76ad83b42f42a2','forecasts_test.parquet':'697839423df8b9145290f5c2c3866f21c5bab05c2307829124614d98b744b712','selection.json':'fb030e723e3b46674c6555bf6356308246966f12af1f0662d5f541d764af4cae','score_report.json':'4d8ded8817557239549d3a8f2331a375a1262e68e1c75fd3f1edddcde69be1f1'}
V4_FILES={'score_report.json':'cc795c345599a1d8b5a5f2f75489973e8bc20deed860099133027f36ae7159d0','forecasts_test.parquet':'9abfedb0cfdf30a57d2f8a08f1b5cd7ba2826ff9796382d1c2088f34ac320495'}
V4_CODE_SHA='34aaf6460db24ea4de78c8b794d3adccfe84fe033819bf7db6d0db2046f1641d'
V4_SCORE_CODE_SHA='3f1ef913b42f4ad9865072523a2851ea720c2c938014d0be93c5b4e5ea00b30f'
OLD_REPORT=ROOT/'outputs/experiments/period-proxy-20261015/report.json'
OLD_SHA='be77991bc4b2b48a1aaf02d286eaedba1633b92d8b8c19c9b337f0184d72d112'
CROPS=proxy.CROPS; HORIZONS=proxy.HORIZONS
CONFIGS={
 'A':dict(max_depth=2,num_leaves=4,min_data_in_leaf=60,learning_rate=.03,n_estimators=200,reg_lambda=30),
 'B':dict(max_depth=3,num_leaves=7,min_data_in_leaf=40,learning_rate=.03,n_estimators=300,reg_lambda=10),
 'C':dict(max_depth=3,num_leaves=7,min_data_in_leaf=25,learning_rate=.03,n_estimators=400,reg_lambda=5)}
PRICE_WINDOWS=(7,28,90); LAGS=(1,2,3,6,12)
STARTS={'develop':(dt.date(2022,1,1),dt.date(2023,12,31)), 'test':(dt.date(2024,1,1),dt.date(2026,9,30))}
STAGES=('preflight','develop','forecast','score')

def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def code_hash():
 h=hashlib.sha256()
 for p in (Path(__file__).resolve(),Path(__file__).with_name('period_pooled_lgbm_score.py')):
  h.update(p.name.encode());h.update(bytes.fromhex(digest(p)))
 return h.hexdigest()
def _check_pins():
 if digest(PROTOCOL)!=PROTOCOL_SHA:raise ValueError('V5 protocol digest mismatch')
 if digest(proxy.SOURCE)!=SOURCE_SHA:raise ValueError('daily source digest mismatch')
 if digest(OLD_REPORT)!=OLD_SHA:raise ValueError('prior report digest mismatch')
 for folder,files in ((V3_OUT,V3_FILES),(V4_OUT,V4_FILES)):
  for name,sha in files.items():
   p=folder/name
   if not p.is_file() or digest(p)!=sha:raise ValueError(f'pinned dependency mismatch: {p}')
 checks=((Path(proxy.__file__),PROXY_CODE_SHA),(Path(v3.__file__),V3_CODE_SHA),
  (Path(v3score.__file__),V3_SCORE_CODE_SHA),(Path(__file__).with_name('period_seasonal_analog.py'),V4_CODE_SHA),
  (Path(__file__).with_name('period_seasonal_analog_score.py'),V4_SCORE_CODE_SHA),
  (proxy.PROTOCOL,PROXY_PROTOCOL_SHA),(v3.PROTOCOL,V3_PROTOCOL_SHA),
  (ROOT/'docs/period_seasonal_analog_protocol.md','1f4742b36200a75a338d9e1766004915946df7d6718b2a339fca027ce8f4e42a'))
 for path,expected in checks:
  if not path.is_file() or digest(path)!=expected:raise ValueError(f'pinned code/protocol dependency mismatch: {path}')
def _lightgbm_identity():
 import lightgbm
 import pandas,scipy,sklearn,sys,pyarrow
 versions={'lightgbm':lightgbm.__version__,'numpy':np.__version__,'pandas':pandas.__version__,
  'scipy':scipy.__version__,'scikitLearn':sklearn.__version__,'pyarrow':pyarrow.__version__}
 expected={'lightgbm':'4.6.0','numpy':'1.26.4','pandas':'2.2.3','scipy':'1.10.1','scikitLearn':'1.3.0','pyarrow':'11.0.0'}
 if versions!=expected:raise ValueError(f'forecast runtime differs from frozen dependencies: {versions}')
 return {**versions,'python':sys.version.split()[0]}
def run_identity():
 _check_pins()
 deps={'sourceSha256':SOURCE_SHA,'proxyCodeSha256':PROXY_CODE_SHA,'proxyProtocolSha256':PROXY_PROTOCOL_SHA,
  'v3CodeSha256':V3_CODE_SHA,'v3ScoreCodeSha256':V3_SCORE_CODE_SHA,'v3ProtocolSha256':V3_PROTOCOL_SHA,
  'v3Artifacts':V3_FILES,'v4CodeSha256':V4_CODE_SHA,'v4ScoreCodeSha256':V4_SCORE_CODE_SHA,
  'v4ProtocolSha256':'1f4742b36200a75a338d9e1766004915946df7d6718b2a339fca027ce8f4e42a',
  'v4Artifacts':V4_FILES,'oldReportSha256':OLD_SHA}
 return {'runId':RUN_ID,'protocolSha256':PROTOCOL_SHA,'analysisCodeSha256':code_hash(),'dependencies':{**deps,**_lightgbm_identity()}}
def _stage_path(out,stage):return Path(out)/(stage+'.json')
def _upstream(stage,out):
 need={'preflight':(),'develop':('preflight',),'development-score':('preflight','develop'),
       'forecast':('preflight','development-score'),'score':('forecast','development-score')}
 if stage not in need:raise ValueError(f'unknown stage {stage}')
 return {s:digest(_stage_path(out,s)) for s in need[stage]}
def save_stage(stage,payload,out=OUT):
 Path(out).mkdir(parents=True,exist_ok=True); ident=run_identity()
 doc={**payload,'runIdentity':ident,'upstreamCheckpointSha256':_upstream(stage,out),'complete':True}
 p=_stage_path(out,stage)
 if p.exists():
  existing=json.loads(p.read_text(encoding='utf8'))
  if existing.get('runIdentity')!=ident or existing.get('upstreamCheckpointSha256')!=_upstream(stage,out):
   raise ValueError(f'existing {stage} checkpoint is stale; refusing to overwrite')
  return existing
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(doc,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf8');os.replace(tmp,p)
 return doc
def load_stage(stage,out=OUT):
 p=_stage_path(out,stage);doc=json.loads(p.read_text());
 if doc.get('runIdentity')!=run_identity() or not doc.get('complete') or doc.get('upstreamCheckpointSha256')!=_upstream(stage,out):raise ValueError(f'stale/incomplete {stage} checkpoint')
 if stage=='preflight':
  cfg=Path(out)/'resolved-config.json'
  if not cfg.is_file() or digest(cfg)!=doc.get('resolvedConfigSha256'):raise ValueError('resolved config/checkpoint digest mismatch')
 return doc
def sha_json(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def feature_order():
 names=['log_p_ref','reference_age_days','reference_available']
 for w in PRICE_WINDOWS:names += [f'price_meanlog_{w}',f'price_rel_{w}',f'price_range_{w}',f'price_sd_{w}',f'price_n_{w}',f'price_coverage_{w}',f'price_available_{w}',f'volume_logsum_{w}',f'volume_n_{w}',f'volume_coverage_{w}',f'volume_available_{w}']
 names += ['volume_ratio_10_20','volume_ratio_available','volume_ratio_recent_n','volume_ratio_prior_n']
 for n in LAGS:names += [f'lag_{n}_logratio',f'lag_{n}_available',f'lag_{n}_elapsed']
 names += ['days_to_target_start','days_to_target_end','target_length_days','month_sin','month_cos','dekad_sin','dekad_cos']
 return sorted(names)
def write_rows(path,rows):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 if any('actual' in r or 'targetObserved' in r for r in rows):raise ValueError('forecast rows must be label-free')
 import pandas as pd
 try:
  import pyarrow # noqa
  p=path.with_suffix('.parquet');pd.DataFrame(rows).to_parquet(p,index=False);fmt='parquet'
 except ImportError:
  p=path.with_suffix('.jsonl');p.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows));fmt='jsonl_fallback_no_parquet_engine'
 return {'path':str(p),'sha256':digest(p),'rows':len(rows),'format':fmt}
def read_rows(meta):
 p=Path(meta['path'])
 if not p.is_file() or digest(p)!=meta['sha256']:raise ValueError('artifact path/hash mismatch')
 if meta['format']=='parquet':
  import pandas as pd
  return pd.read_parquet(p).to_dict('records')
 if meta['format']=='jsonl_fallback_no_parquet_engine':return [json.loads(x) for x in p.read_text().splitlines()]
 raise ValueError('unknown artifact format')

def _periodkey(t):return f'{t[0].isoformat()}|{t[1].isoformat()}'
def _latest(series,cutoff):
 days=[d for d,(p,v) in series.items() if d<=cutoff and math.isfinite(p) and p>0]
 if not days:return None,None
 d=max(days);return float(series[d][0]),d

def feature_row(series,targets,crop,h,target,origin):
 """As-of features for one issue row; future source mutations cannot affect it."""
 cutoff=proxy.input_cutoff(origin);pref,pday=_latest(series,cutoff)
 f={'log_p_ref':math.log(pref) if pref is not None else 0.,'reference_age_days':(cutoff-pday).days if pday else 0.,'reference_available':int(pref is not None)}
 meta={'referencePrice':pref,'referenceDate':pday.isoformat() if pday else None,'inputCutoff':cutoff.isoformat(),'origin':origin.isoformat()}
 for w in PRICE_WINDOWS:
  obs=[float(p) for d,(p,v) in series.items() if cutoff-dt.timedelta(days=w-1)<=d<=cutoff and math.isfinite(p) and p>0]
  logs=np.log(obs) if obs else np.array([],dtype=float)
  f.update({f'price_meanlog_{w}':float(logs.mean()) if len(logs) else 0.,f'price_rel_{w}':float(math.log(pref)-logs.mean()) if pref is not None and len(logs) else 0.,f'price_range_{w}':float(logs.max()-logs.min()) if len(logs) else 0.,f'price_sd_{w}':float(logs.std(ddof=0)) if len(logs) else 0.,f'price_n_{w}':len(logs),f'price_coverage_{w}':len(logs)/w,f'price_available_{w}':int(len(logs)>0 and pref is not None)})
  vols=[float(v) for d,(p,v) in series.items() if cutoff-dt.timedelta(days=w-1)<=d<=cutoff and math.isfinite(v) and v>0]
  f.update({f'volume_logsum_{w}':math.log1p(sum(vols)),f'volume_n_{w}':len(vols),f'volume_coverage_{w}':len(vols)/w,f'volume_available_{w}':int(bool(vols))})
 # ratio of means per observed day with fixed support thresholds
 for name,lo,hi,minn in [('10v20',9,0,3)]: pass
 end10=cutoff;start10=cutoff-dt.timedelta(days=9);start20=cutoff-dt.timedelta(days=29);end20=cutoff-dt.timedelta(days=10)
 a=[v for d,(p,v) in series.items() if start10<=d<=end10 and math.isfinite(v) and v>0];b=[v for d,(p,v) in series.items() if start20<=d<=end20 and math.isfinite(v) and v>0]
 f['volume_ratio_10_20']=math.log(float(np.mean(a))/float(np.mean(b))) if len(a)>=3 and len(b)>=6 else 0.;f['volume_ratio_available']=int(len(a)>=3 and len(b)>=6);f['volume_ratio_recent_n']=len(a);f['volume_ratio_prior_n']=len(b)
 # latest mature, valid target periods as-of original origin
 eligible=sorted(((t,v) for t,v in targets.items() if v.get('actual') is not None and proxy.label_matured(t,origin)),key=lambda z:z[0][1],reverse=True)
 recent_calendar=proxy._previous_period_target(proxy.dekad_bounds(cutoff),0) if False else None
 last_complete=proxy._next_period(proxy.dekad_bounds(cutoff),-1) if proxy.dekad_bounds(cutoff)[1]>cutoff else proxy.dekad_bounds(cutoff)
 latest_idx=proxy._period_index(last_complete)
 for n in LAGS:
  if len(eligible)>=n:
   t,v=eligible[n-1];f[f'lag_{n}_logratio']=math.log(float(v['actual'])/pref) if pref else 0.;f[f'lag_{n}_available']=int(pref is not None);f[f'lag_{n}_elapsed']=max(0,latest_idx-proxy._period_index(t))
  else:f[f'lag_{n}_logratio']=0.;f[f'lag_{n}_available']=0;f[f'lag_{n}_elapsed']=0
 lo,hi=target
 f.update({'days_to_target_start':(lo-origin).days,'days_to_target_end':(hi-origin).days,'target_length_days':(hi-lo).days+1,'month_sin':math.sin(2*math.pi*(lo.month-1)/12),'month_cos':math.cos(2*math.pi*(lo.month-1)/12),'dekad_sin':math.sin(2*math.pi*((proxy._period_index(target)-1)%36)/36),'dekad_cos':math.cos(2*math.pi*((proxy._period_index(target)-1)%36)/36)})
 for k in f:
  if not math.isfinite(float(f[k])):raise ValueError(f'nonfinite feature {k}')
 meta['features']=f;meta['crop']=crop;meta['horizon']=h;meta['targetStart']=lo.isoformat();meta['targetEnd']=hi.isoformat();meta['targetKey']=_periodkey(target)
 meta['nonpositiveVolumeExcluded']=sum(1 for d,(p,v) in series.items() if cutoff-dt.timedelta(days=89)<=d<=cutoff and (not math.isfinite(v) or v<=0))
 return meta

def build_catalog(series_by_crop):
 targets={c:proxy._target_periods(series_by_crop[c]) for c in CROPS}; rows=[]
 for crop in CROPS:
  for h in HORIZONS:
   for s in proxy.planned_origins(dt.date(2013,12,1),dt.date(2026,9,30),h):
    t=s['target']; y=targets[crop].get(t,{}).get('actual')
    if y is None or s['origin']<dt.date(2014,1,1):continue
    fr=feature_row(series_by_crop[crop],targets[crop],crop,h,t,s['origin'])
    if fr['referencePrice'] is None:continue
    rows.append({'crop':crop,'horizon':h,'target':t,'origin':s['origin'],'targetEnd':t[1],'actual':float(y),'p_ref':fr['referencePrice'],'features':fr['features'],'featureMeta':fr})
 return rows,targets

def eligible_train(catalog,origin):
 return [r for r in catalog if r['targetEnd']+dt.timedelta(days=2)<=origin]
def row_weights(rows):
 if not rows:raise ValueError('empty training rows')
 keycounts={}
 for r in rows:key=(r['crop'],r['target']);keycounts[key]=keycounts.get(key,0)+1
 base=np.array([1/keycounts[(r['crop'],r['target'])] for r in rows],float)
 n=len(rows);cell_sizes={f'{c}:{h}':sum(r['crop']==c and r['horizon']==h for r in rows) for c in CROPS for h in HORIZONS}
 if any(v==0 for v in cell_sizes.values()):raise ValueError(f'training cell empty: {cell_sizes}')
 # Per-cell sum equal; cell average scaling, then global mean one.
 out=base.copy()
 for c in CROPS:
  for h in HORIZONS:
   idx=np.array([i for i,r in enumerate(rows) if r['crop']==c and r['horizon']==h]);out[idx]*=(n/8)/out[idx].sum()
 out*=n/out.sum()
 return out,cell_sizes

def _numeric_features(row):return row['features']
FEATURE_NAMES=None
def _matrix(rows, *,linear=False):
 names=sorted(rows[0]['features']);x=np.array([[r['features'][k] for k in names] for r in rows],float)
 crops={c:i for i,c in enumerate(CROPS)};hvals={2:0,3:1}
 if linear:x=np.column_stack([x,*[np.array([r['crop']==c for r in rows],float) for c in CROPS[1:]],np.array([r['horizon']==3 for r in rows],float)])
 else:x=np.column_stack([x,np.array([crops[r['crop']] for r in rows],dtype=np.int32),np.array([r['horizon'] for r in rows],float)])
 return x,names

def _fit_contract(rows):
 w,cells=row_weights(rows);y=np.asarray([math.log(r['actual']/r['p_ref']) for r in rows],dtype='<f8')
 x,names=_matrix(rows,linear=False);numeric=np.asarray(x[:,:len(names)],dtype='<f8')
 keys=[(r['crop'],r['horizon'],r['target'][0].isoformat(),r['origin'].isoformat()) for r in rows]
 return {'trainingKeySha256':sha_json(keys),'numericFeatureMatrixSha256':hashlib.sha256(numeric.tobytes()).hexdigest(),
  'labelSha256':hashlib.sha256(y.tobytes()).hexdigest(),'weightSha256':sha_json(w.tolist()),'featureNames':names,
  'cellSupport':cells,'weights':w,'labels':y,'numeric':numeric}

def fit_model(rows,kind,config=None):
 import lightgbm as lgb
 contract=_fit_contract(rows);w=contract['weights'];y=contract['labels'];cells=contract['cellSupport'];x,names=_matrix(rows,linear=kind=='L')
 model_meta={'kind':kind,'rows':len(rows),'cellSupport':cells,'trainingKeySha256':contract['trainingKeySha256'],
  'numericFeatureMatrixSha256':contract['numericFeatureMatrixSha256'],'labelSha256':contract['labelSha256'],
  'weightSha256':contract['weightSha256'],'featureNames':names}
 if kind=='L':
  scaler=StandardScaler().fit(x[:,:len(names)]);xs=np.column_stack([scaler.transform(x[:,:len(names)]),x[:,len(names):]])
  model=Ridge(alpha=10.,fit_intercept=True).fit(xs,y,sample_weight=w);pred=model.predict(xs)
  model_meta.update({'scalerMean':scaler.mean_.tolist(),'scalerScale':scaler.scale_.tolist(),'model':model,'scaler':scaler})
 else:
  # category coded as pandas category, crop is the only categorical feature
  import pandas as pd
  cat=pd.Categorical([r['crop'] for r in rows],categories=CROPS,ordered=False)
  xdf=pd.DataFrame(x[:,:-2],columns=names);xdf['crop_id']=cat;xdf['horizon']=x[:,-1]
  params={**(config or CONFIGS['A']),'objective':'regression_l1','random_state':42,'verbosity':-1,'feature_fraction':1.,'bagging_fraction':1.,'bagging_freq':0,'max_bin':63,'force_col_wise':True,'deterministic':True,'num_threads':1}
  model=lgb.LGBMRegressor(**params)
  model.fit(xdf,y,sample_weight=w,categorical_feature=['crop_id']);pred=model.predict(xdf,num_threads=1)
  model_meta.update({'model':model,'categoricalLevels':list(CROPS),'lightgbmParameters':params,'featureNames':names})
 replay=model.predict(np.column_stack([scaler.transform(x[:,:len(names)]),x[:,len(names):]])) if kind=='L' else model.predict(xdf,num_threads=1)
 model_meta['trainPredReplayMaxAbs']=float(np.max(np.abs(np.asarray(pred)-np.asarray(replay))))
 return model_meta

def predict_rows(model_meta,rows):
 kind=model_meta['kind'];x,names=_matrix(rows,linear=kind=='L')
 if names!=model_meta['featureNames']:raise ValueError('feature order mismatch')
 if kind=='L':
  numeric=(x[:,:len(names)]-np.asarray(model_meta['scalerMean']))/np.asarray(model_meta['scalerScale'])
  x=np.column_stack([numeric,x[:,len(names):]])
  z=np.asarray(model_meta['model'].predict(x))
 else:
  import pandas as pd
  xdf=pd.DataFrame(x[:,:-2],columns=names);xdf['crop_id']=pd.Categorical([r['crop'] for r in rows],categories=CROPS);xdf['horizon']=x[:,-1]
  z=np.asarray(model_meta['model'].predict(xdf,num_threads=1))
 return z

def _save_fit(model_meta,path,identity):
 import joblib
 path.parent.mkdir(parents=True,exist_ok=True);joblib.dump({'identity':identity,'meta':model_meta},path);return digest(path)
def _load_fit(path,expected_sha,identity):
 import joblib
 if digest(path)!=expected_sha:raise ValueError('saved model hash mismatch')
 obj=joblib.load(path)
 if obj['identity']!=identity:raise ValueError('saved model identity mismatch')
 return obj['meta']

def _sources():
 _,raw=proxy.validate_inputs();series=proxy.series_from_raw(raw);catalog,targets=build_catalog(series);return series,catalog,targets
def _schedules(split):
 a,b=STARTS[split];return [s for h in HORIZONS for s in proxy.planned_origins(a,b,h)]
def _v3_rows(name):
 import pandas as pd
 p=V3_OUT/name
 if digest(p)!=V3_FILES[name]:raise ValueError('pinned V3 forecast mismatch')
 return pd.read_parquet(p).to_dict('records')
def _key(r):return (r['crop'],int(r['horizon']),r['targetStart'],r['targetEnd'],r['origin'])
def _v3_predictions():
 dev=_v3_rows('forecasts.parquet');test=_v3_rows('forecasts_test.parquet');selection=json.loads((V3_OUT/'selection.json').read_text())
 def val(row,method):
  obj=row.get('forecasts',{}).get(method) or {}
  p=obj.get('predicted')
  return float(p) if p is not None and math.isfinite(float(p)) and float(p)>0 else None
 out={}
 for split,rows in [('develop',dev),('test',test)]:
  for r in rows:
   if split=='develop':
    crop=r['crop'];method=selection['P'][crop]['method'];pp=val(r,method)
   else:pp=val(r,'P')
   out[(split,*_key(r))]={'P':pp,'B0':val(r,'B0'),'policyUnavailable':bool(r.get('policyUnavailable',False))}
 return out,selection

def preflight(out=OUT):
 ident=run_identity();Path(out).mkdir(parents=True,exist_ok=True)
 if _stage_path(out,'preflight').exists():
  old=load_stage('preflight',out);cfg=Path(out)/'resolved-config.json'
  if not cfg.is_file() or digest(cfg)!=old.get('resolvedConfigSha256'):raise ValueError('resolved config/checkpoint digest mismatch')
  return old
 _check_pins();protocol,raw=proxy.validate_inputs();series=proxy.series_from_raw(raw)
 # Confirm exact stored grid keys and no forecast/label integrity mismatch.
 v3rows=_v3_rows('forecasts_test.parquet');devrows=_v3_rows('forecasts.parquet')
 if len(v3rows)!=792 or len(devrows)!=576:raise ValueError('V3 fixed grid row count mismatch')
 for split,rows in (('develop',devrows),('test',v3rows)):
  a,b=STARTS[split];expected_grid={}
  for crop in CROPS:
   for h in HORIZONS:
    for sched in proxy.planned_origins(a,b,h):
     t=sched['target'];key=(crop,h,t[0].isoformat(),t[1].isoformat())
     expected_grid[key]=(sched['origin'].isoformat(),sched['inputCutoff'].isoformat())
  observed_grid={}
  for r in rows:
   key=(r['crop'],int(r['horizon']),r['targetStart'],r['targetEnd'])
   if key in observed_grid:raise ValueError(f'duplicate saved V3 forecast key: {key}')
   observed_grid[key]=(r['origin'],r['inputCutoff'])
  if observed_grid!=expected_grid:raise ValueError(f'V3 {split} target/origin/cutoff grid differs from the frozen schedule')
 origins={s['origin'] for s in _schedules('develop')};test_s=_schedules('test')
 if len(origins)!=73:raise ValueError(f'development origins differ from frozen 73: {len(origins)}')
 test_origins={s['origin'] for s in test_s if s['origin']>=proxy.POLICY_AVAILABLE_FROM and any(
   not bool(r.get('policyUnavailable',False)) for r in v3rows if r['origin']==s['origin'].isoformat())}
 if len(test_origins)!=98:raise ValueError(f'test supported origins differ from frozen 98: {len(test_origins)}')
 fits=4*len(origins)+2*len(test_origins)
 if fits>600:raise ValueError(f'fit cap exceeded: {fits}')
 # Reaggregate exact V3 P/B0 global metrics with proxy score routine.
 targets={c:proxy._target_periods(s) for c,s in series.items()};pairs={'P':[],'B0':[]};cell={}
 unavailable_count=sum(bool(r.get('policyUnavailable',False)) for r in v3rows)
 if unavailable_count!=12:raise ValueError(f'expected 12 unavailable baseline rows, found {unavailable_count}')
 expected_p=set();expected_b=set()
 for r in v3rows:
  f=r.get('forecasts',{})
  if (f.get('P') or {}).get('predicted') is not None:expected_p.add(_key(r))
  if (f.get('B0') or {}).get('predicted') is not None:expected_b.add(_key(r))
 if expected_p!=expected_b or len(expected_p)!=780:raise ValueError('frozen V3 P/B0 key sets are not exactly the expected 780 paired keys')
 dev_actual_count=sum(targets[r['crop']].get((dt.date.fromisoformat(r['targetStart']),dt.date.fromisoformat(r['targetEnd'])),{}).get('actual') is not None for r in devrows)
 if dev_actual_count!=556:raise ValueError(f'expected 556 observed development labels among 576 rows, found {dev_actual_count}')
 for crop in CROPS:
  for h in HORIZONS:
   rr=[r for r in v3rows if r['crop']==crop and int(r['horizon'])==h];pp={'P':[],'B0':[]}
   for r in rr:
    for method in ('P','B0'):
     p=(r.get('forecasts',{}).get(method) or {}).get('predicted')
     if p is None:continue
     target=(dt.date.fromisoformat(r['targetStart']),dt.date.fromisoformat(r['targetEnd']));y=targets[crop].get(target,{}).get('actual')
     if y is not None:pp[method].append((y,float(p)));pairs[method].append((y,float(p)))
   cell[f'{crop}:{h}']={m:proxy._metrics(pp[m]) for m in ('P','B0')}
   expected=(98 if h==2 else 97)
   if len(pp['P'])!=expected or len(pp['B0'])!=expected:raise ValueError('V3 test paired support differs')
 equal={m:{'mae':float(np.mean([cell[f'{c}:{h}'][m]['mae'] for c in CROPS for h in HORIZONS])),
            'mape':float(np.mean([cell[f'{c}:{h}'][m]['mape'] for c in CROPS for h in HORIZONS]))} for m in ('P','B0')}
 # V3 standard saved score is the authority for exact support/macro values.
 score=json.loads((V3_OUT/'score_report.json').read_text())
 for m in ('P','B0'):
  for metric in ('mae','mape'):
   if not math.isclose(equal[m][metric],score['metrics8EqualCell'][m][metric],rel_tol=0,abs_tol=1e-8):raise ValueError(f'V3 macro reaggregation mismatch {m}/{metric}')
 doc={'stage':'preflight','status':'verified','runIdentity':ident,'grid':{'devRows':len(devrows),'testRows':len(v3rows),'testPolicyAvailableKeys':780,'testUnavailableRows':12,'devOrigins':len(origins),'testSupportedOrigins':len(test_origins),'projectedFits':fits,'fitLimit':600},'v3Metrics8EqualCell':equal,'v3Cells':cell,'artifactHashes':{'source':SOURCE_SHA,'v3':V3_FILES,'v4':V4_FILES,'oldReport':OLD_SHA}}
 cfg={'runId':RUN_ID,'protocolSha256':PROTOCOL_SHA,'sourceSha256':SOURCE_SHA,'analysisCodeSha256':code_hash(),'runIdentity':ident,'crops':CROPS,'horizons':HORIZONS,'featureWindows':PRICE_WINDOWS,'lags':LAGS,'featureOrder':feature_order(),'featureTypes':{**{n:'numeric' for n in feature_order()},'crop_id':'categorical','horizon':'numeric'},'configs':CONFIGS,'maxFits':600,'testRows':792,'comparisonKeys':780,'forecastTarget':'log(actual/p_ref)','sampleWeightContract':'1/m(crop,target), equal crop×h cell sums, global mean one'}
 config_path=Path(out)/'resolved-config.json';config_path.write_text(json.dumps(cfg,indent=2,allow_nan=False)+'\n')
 doc['resolvedConfigSha256']=digest(config_path)
 return save_stage('preflight',doc,out)

def _fit_identity(split,origin,kind,config,rows):
 common={'runIdentity':run_identity(),'split':split,'origin':origin.isoformat(),'config':config,**{k:v for k,v in _fit_contract(rows).items() if k not in ('weights','labels','numeric','cellSupport')}}
 return {**common,'kind':kind}

def _write_fit_ledger(out,ledger):
 p=Path(out)/'fit-ledger.json';tmp=p.with_suffix('.json.tmp')
 tmp.write_text(json.dumps(ledger,indent=2,default=str,allow_nan=False)+'\n',encoding='utf8');os.replace(tmp,p)

def _forecast_origin(split,origin,schedules,catalog,series,targets,configs,out,ledger,base_predictions):
 train=eligible_train(catalog,origin);current=[]
 for s in schedules:
  crop=s['crop'];h=s['horizon'];t=s['target'];fr=feature_row(series[crop],targets[crop],crop,h,t,origin)
  current.append({'crop':crop,'horizon':h,'target':t,'origin':origin,'targetEnd':t[1],'actual':None,'p_ref':fr['referencePrice'],'features':fr['features'],'featureMeta':fr})
 if any(r['p_ref'] is None for r in current):
  for r in current:r['p_ref']=r['p_ref']
 output=[]
 common_contract=None
 for kind,config_name,config in configs:
  if len(ledger)>=600:raise ValueError('hard fit cap of 600 reached; refusing another model fit')
  identity=_fit_identity(split,origin,kind,config,train);identhash=sha_json(identity)
  attempt_no=len(ledger)+1
  model_path=Path(out)/'models'/f'{split}-{origin.isoformat()}-{kind}-{config_name}-attempt{attempt_no:03d}.joblib'
  entry={'split':split,'origin':origin.isoformat(),'kind':kind,'config':config_name,'fitIdentitySha256':identhash,
   'trainRows':len(train),'trainCells':row_weights(train)[1],'uniqueCropTarget':len({(r['crop'],r['target']) for r in train}),
   'lastMatureTargetEnd':max((r['targetEnd'] for r in train),default=None).isoformat() if train else None,
   'modelPath':str(model_path),'fitIdentity':identity,'status':'started','attempt':attempt_no,'startedAtUtc':dt.datetime.now(dt.timezone.utc).isoformat()}
  ledger.append(entry);_write_fit_ledger(out,ledger)
  try:model_meta=fit_model(train,kind,config)
  except Exception as exc:
   entry.update({'status':'failed','failureType':type(exc).__name__,'failureMessage':str(exc)[:500],
    'finishedAtUtc':dt.datetime.now(dt.timezone.utc).isoformat()});_write_fit_ledger(out,ledger);raise
  for contract_key in ('trainingKeySha256','numericFeatureMatrixSha256','labelSha256','weightSha256','featureNames'):
   if model_meta.get(contract_key)!=identity.get(contract_key):raise ValueError(f'model training contract mismatch: {contract_key}')
  contract={k:model_meta[k] for k in ('trainingKeySha256','numericFeatureMatrixSha256','labelSha256','weightSha256')}
  if common_contract is None:common_contract=contract
  elif contract!=common_contract:raise ValueError('G/L candidate training rows/features/labels/weights differ')
  sha=_save_fit(model_meta,model_path,identity);loaded=_load_fit(model_path,sha,identity)
  original_z=predict_rows(model_meta,current);reloaded_z=predict_rows(loaded,current)
  if not np.allclose(original_z,reloaded_z,rtol=1e-12,atol=1e-9,equal_nan=True):raise ValueError('saved model raw-log replay mismatch')
  def replay_price(pref,z):
   try:
    value=float(pref)*math.exp(float(z)) if pref is not None else float('nan')
    return value if math.isfinite(value) and value>0 else float('nan')
   except OverflowError:return float('nan')
  original_price=np.array([replay_price(r['p_ref'],z) for r,z in zip(current,original_z)])
  reloaded_price=np.array([replay_price(r['p_ref'],z) for r,z in zip(current,reloaded_z)])
  valid=np.isfinite(original_price)&np.isfinite(reloaded_price)
  if not np.allclose(original_price[valid],reloaded_price[valid],rtol=1e-12,atol=1e-9):raise ValueError('saved model price replay mismatch')
  prices=[]
  for r,rawz in zip(current,reloaded_z):
   p=None;status='ok';fallback=None
   try:
    if r['p_ref'] is None:raise ValueError('missing_p_ref')
    p=float(r['p_ref'])*math.exp(float(rawz))
    if not math.isfinite(p) or p<=0:raise ValueError('invalid_inverse_output')
   except (OverflowError,ValueError) as exc:
    status='technical_fallback_P';fallback=str(exc)
    key=(r['crop'],r['horizon'],r['target'][0].isoformat(),r['target'][1].isoformat(),origin.isoformat())
    p=base_predictions.get(key,{}).get('P')
   finite_raw=math.isfinite(float(rawz))
   rawz_value=format(float(rawz),'.17g') if finite_raw else ('NaN' if math.isnan(float(rawz)) else 'Infinity' if float(rawz)>0 else '-Infinity')
   prices.append({'crop':r['crop'],'horizon':r['horizon'],'targetStart':r['target'][0].isoformat(),'targetEnd':r['target'][1].isoformat(),'targetKey':r['target'][0].isoformat(),'origin':origin.isoformat(),'inputCutoff':r['featureMeta']['inputCutoff'],'p_ref':r['p_ref'],'pred_z':float(rawz) if finite_raw else None,'pred_z_raw':rawz_value,'pred_z_raw_nonfinite':not finite_raw,'predicted':p,'status':status,'fallbackReason':fallback,'featureMeta':r['featureMeta']})
  entry.update({'status':'complete','finishedAtUtc':dt.datetime.now(dt.timezone.utc).isoformat(),'modelSha256':sha,
   'trainKeySha256':model_meta['trainingKeySha256'],'numericFeatureMatrixSha256':model_meta['numericFeatureMatrixSha256'],
   'labelSha256':model_meta['labelSha256'],'weightSha256':model_meta['weightSha256'],
   'replayRawZMaxAbs':float(np.max(np.abs(original_z[np.isfinite(original_z)]-reloaded_z[np.isfinite(reloaded_z)]))) if np.isfinite(original_z).any() else None,
   'replayPriceMaxAbs':float(np.max(np.abs(original_price[valid]-reloaded_price[valid]))) if valid.any() else None})
  _write_fit_ledger(out,ledger)
  output.append((kind,config_name,prices))
 # Pivot outputs by target key, never carry labels into artifact.
 by={}
 for kind,cfg,rows in output:
  for r in rows:
   key=(r['crop'],r['horizon'],r['targetStart'],r['targetEnd'],r['origin'])
   by.setdefault(key,{**{k:v for k,v in r.items() if k not in ('predicted','pred_z','pred_z_raw','pred_z_raw_nonfinite','status','fallbackReason','featureMeta')},
      'asOfFeatures':r['featureMeta']['features'],'featureMetadata':{k:v for k,v in r['featureMeta'].items() if k!='features'}})
   by[key].setdefault('predictions',{})[f'{kind}-{cfg}']={k:r[k] for k in ('predicted','pred_z','pred_z_raw','pred_z_raw_nonfinite','status','fallbackReason')}
 return list(by.values())

def _run_forecasts(split,out=OUT):
 stage='develop' if split=='develop' else 'forecast'
 if _stage_path(out,stage).exists():
  cached=load_stage(stage,out);read_rows(cached['forecastArtifact'])
  ledger_path=Path(out)/f'fit-ledger-{split}.json'
  if not ledger_path.is_file() or digest(ledger_path)!=cached.get('fitLedgerSha256'):raise ValueError('completed forecast stage fit ledger snapshot differs')
  if split=='test' and (Path(out)/'development-selection.json').is_file() and digest(Path(out)/'development-selection.json')!=cached.get('selectionSha256'):
   raise ValueError('frozen development selection differs from completed test forecast')
  return cached
 load_stage('preflight',out);series,catalog,targets=_sources();selection_path=Path(out)/'development-selection.json'
 selection_sha=None
 if split=='test':
  if not selection_path.is_file():raise FileNotFoundError('development selection must be frozen before test forecasts')
  selection=json.loads(selection_path.read_text())
  if selection.get('runIdentity')!=run_identity():raise ValueError('development selection identity mismatch')
  devscore=load_stage('development-score',out)
  if devscore.get('selectionSha256')!=digest(selection_path):raise ValueError('selection differs from frozen development-score checkpoint')
  selection_sha=digest(selection_path)
  chosen=selection['chosenConfig']
  configs=[('G',chosen,CONFIGS[chosen]),('L','L',None)]
 else:configs=[('G',k,c) for k,c in CONFIGS.items()]+[('L','L',None)]
 a,b=STARTS[split];grid={}
 v3name='forecasts.parquet' if split=='develop' else 'forecasts_test.parquet'
 v3rows=_v3_rows(v3name)
 for r in v3rows:
  origin=dt.date.fromisoformat(r['origin'])
  if split=='test' and (r.get('policyUnavailable') or origin<proxy.POLICY_AVAILABLE_FROM):continue
  target_start=dt.date.fromisoformat(r['targetStart'])
  if not a<=target_start<=b:continue
  grid.setdefault(origin,[]).append({'crop':r['crop'],'horizon':int(r['horizon']),'target':(dt.date.fromisoformat(r['targetStart']),dt.date.fromisoformat(r['targetEnd']))})
 rows=[];ledger=[]
 progress=Path(out)/f'{split}-progress.json'
 progress_rows=Path(out)/f'{split}-progress-rows.jsonl'
 ledger_path=Path(out)/'fit-ledger.json'
 done=set()
 if ledger_path.exists():ledger=json.loads(ledger_path.read_text(encoding='utf8'))
 if not progress.exists():
  changed=False
  for attempt in ledger:
   if attempt.get('split')==split and attempt.get('status')=='started':
    attempt.update({'status':'interrupted','finishedAtUtc':dt.datetime.now(dt.timezone.utc).isoformat(),
     'interruptionNote':'retry began without a committed origin progress checkpoint'});changed=True
  if changed:_write_fit_ledger(out,ledger)
 if progress.exists():
  state=json.loads(progress.read_text(encoding='utf8'))
  upstream_stage='develop' if split=='develop' else 'forecast'
  if state.get('runIdentity')!=run_identity() or state.get('split')!=split or state.get('upstream')!=_upstream(upstream_stage,out) or state.get('selectionSha256')!=selection_sha:raise ValueError('forecast progress identity/protocol/selection mismatch')
  if not progress_rows.is_file() or digest(progress_rows)!=state.get('rowsSha256') or not ledger_path.is_file():
   raise ValueError('forecast progress artifact digest mismatch')
  rows=[json.loads(x) for x in progress_rows.read_text().splitlines() if x]
  ledger=json.loads(ledger_path.read_text());done=set(state.get('completedOrigins',[]))
  committed=int(state.get('committedAttemptCount',-1))
  if committed<0 or len(ledger)<committed or sha_json(ledger[:committed])!=state.get('committedLedgerPrefixSha256'):
   raise ValueError('forecast progress committed ledger prefix mismatch')
  extras=ledger[committed:]
  if any(x.get('split')!=split or x.get('origin') in done for x in extras):
   raise ValueError('uncommitted fit attempts do not belong to an incomplete origin')
  interrupted=False
  for attempt in extras:
   if attempt.get('status')=='started':
    attempt.update({'status':'interrupted','finishedAtUtc':dt.datetime.now(dt.timezone.utc).isoformat(),
     'interruptionNote':'process restarted before the fit outcome was checkpointed'});interrupted=True
  if interrupted:_write_fit_ledger(out,ledger)
 _,v3sel=_v3_predictions()
 base_rows=_v3_rows('forecasts.parquet' if split=='develop' else 'forecasts_test.parquet')
 base_predictions={}
 for br in base_rows:
  k=(br['crop'],int(br['horizon']),br['targetStart'],br['targetEnd'],br['origin'])
  forecasts=br.get('forecasts',{})
  def getpred(method):
   x=(forecasts.get(method) or {}).get('predicted');return float(x) if x is not None and math.isfinite(float(x)) and float(x)>0 else None
  pm=v3sel['P'][br['crop']]['method'] if split=='develop' else 'P'
  base_predictions[k]={'P':getpred(pm),'B0':getpred('B0')}
 for origin,sch in sorted(grid.items()):
  if origin.isoformat() in done:continue
  rows.extend(_forecast_origin(split,origin,sch,catalog,series,targets,configs,out,ledger,base_predictions))
  done.add(origin.isoformat())
  progress_rows.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows),encoding='utf8')
  progress.write_text(json.dumps({'runIdentity':run_identity(),'split':split,'upstream':_upstream('develop' if split=='develop' else 'forecast',out),'selectionSha256':selection_sha,'completedOrigins':sorted(done),
   'rowsSha256':digest(progress_rows),'committedAttemptCount':len(ledger),'committedLedgerPrefixSha256':sha_json(ledger)},indent=2)+'\n',encoding='utf8')
 if split=='test':
  # Retain all 792 planned keys, including the 12 policy-unavailable early rows.
  present={(r['crop'],int(r['horizon']),r['targetStart'],r['targetEnd'],r['origin']) for r in rows}
  for br in base_rows:
   if not br.get('policyUnavailable'):continue
   key=(br['crop'],int(br['horizon']),br['targetStart'],br['targetEnd'],br['origin'])
   if key not in present:
    rows.append({'crop':key[0],'horizon':key[1],'targetStart':key[2],'targetEnd':key[3],'targetKey':key[2],
     'origin':key[4],'inputCutoff':br['inputCutoff'],'policyUnavailable':True,'predictions':{},'status':'policy_unavailable'})
 if split=='develop':stage='develop';fname='development-forecasts.parquet'
 else:stage='forecast';fname='forecasts.parquet'
 forecast_meta=write_rows(Path(out)/fname,rows)
 ledger_path.write_text(json.dumps(ledger,indent=2,default=str,allow_nan=False)+'\n')
 snapshot=Path(out)/f'fit-ledger-{split}.json'
 split_ledger=[x for x in ledger if x.get('split')==split]
 snapshot.write_text(json.dumps(split_ledger,indent=2,default=str,allow_nan=False)+'\n')
 doc={'stage':stage,'forecastArtifact':forecast_meta,'fitCount':len([x for x in split_ledger if x.get('status')=='complete']),'fitAttempts':len(split_ledger),'originCount':len(grid),'fitLedgerSha256':digest(snapshot),'selectionSha256':selection_sha}
 progress.unlink(missing_ok=True);progress_rows.unlink(missing_ok=True)
 return save_stage(stage,doc,out)
def develop(out=OUT):return _run_forecasts('develop',out)
def forecast(out=OUT):return _run_forecasts('test',out)

def main(argv=None):
 p=argparse.ArgumentParser(description=RUN_ID);p.add_argument('stage',choices=('preflight','develop','development-score','forecast','score','run'));p.add_argument('--output',type=Path,default=OUT);a=p.parse_args(argv)
 from . import period_pooled_lgbm_score as scoremod
 if a.stage=='preflight':result=preflight(a.output)
 elif a.stage=='develop':result=develop(a.output)
 elif a.stage=='development-score':result=scoremod.score_development(a.output)
 elif a.stage=='forecast':result=forecast(a.output)
 elif a.stage=='score':result=scoremod.score(a.output)
 else:
  result=preflight(a.output);result=develop(a.output);result=scoremod.score_development(a.output);result=forecast(a.output);result=scoremod.score(a.output)
 print(json.dumps({'stage':a.stage,'complete':result.get('complete',True),'status':result.get('status'),'fitCount':result.get('fitCount')},ensure_ascii=False))
if __name__=='__main__':main()
