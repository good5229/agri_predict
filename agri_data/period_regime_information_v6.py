"""V6 risk-probability experiment. No price forecasts are fitted here."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import warnings
from pathlib import Path

for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import numpy as np
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss

from . import period_proxy_experiment as proxy
from . import period_pooled_lgbm as v5

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/experiments/period-regime-information-v6-20261016'
CROPS = ('cabbage', 'radish')
HORIZON = 3
FEATURES_RB = ('month_sin','month_cos','dekad_sin','dekad_cos','days_to_target_start','log_p_ref')
FEATURES_RV = FEATURES_RB + ('price_sd_28','price_available_28')
NUMERIC_RB = FEATURES_RB
NUMERIC_RV = FEATURES_RB + ('price_sd_28',)
SUPPORT = {'per_crop_rows': 24, 'pooled_each_class': 10, 'gate_each_crop_each_year': 24,
           'gate_each_class_each_crop_year': 5, 'log_loss_delta_max': .02}
MAX_ATTEMPTS = 450
PLANNED_ATTEMPTS = 342
MODEL = dict(penalty='l2', C=1.0, solver='lbfgs', class_weight=None,
             fit_intercept=True, max_iter=2000, tol=1e-8, random_state=42)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def as_date(value):
    if isinstance(value, dt.datetime): return value.date()
    if isinstance(value, dt.date): return value
    return dt.date.fromisoformat(str(value)[:10])


def optional_float(value):
    if value is None: return None
    try:
        v=float(value)
    except (TypeError,ValueError): return None
    return v if math.isfinite(v) else None


def optional_probability(value):
    if value is None: return None
    try: p=float(value)
    except (TypeError,ValueError) as exc: raise ValueError('invalid saved probability') from exc
    if math.isnan(p): return None
    if not math.isfinite(p) or not 0 <= p <= 1: raise ValueError('saved probability outside [0,1]')
    return p


def run_identity(out=OUT):
    protocol=Path(out)/'protocol.md'
    if not protocol.is_file() or sha256(protocol)!='ef68429d97be7615279b3d96e5bdc9a551117ee9a1d07ab4d805d9beb1752de7':
        raise ValueError('blocked_integrity: frozen V6 protocol digest mismatch')
    request=Path(out)/'request.md'
    if not request.is_file(): raise ValueError('blocked_integrity: original V6 request missing')
    return {'protocolSha256':sha256(protocol),'coreCodeSha256':sha256(__file__),
            'sourceSha256':sha256(proxy.SOURCE),'v5CodeSha256':sha256(v5.__file__),
            'proxyCodeSha256':sha256(proxy.__file__),'requestSha256':sha256(request)}


def verify_source_manifest(out=OUT):
    path=Path(out)/'source-manifest.json'
    manifest=json.loads(path.read_text(encoding='utf-8'))
    checked=0
    for key,item in manifest.items():
        if not isinstance(item,dict) or 'path' not in item or 'sha256' not in item: continue
        source=Path(item['path'])
        if not source.is_absolute(): source=ROOT/source
        if not source.is_file() or sha256(source)!=item['sha256']:
            raise ValueError(f'blocked_integrity: source manifest mismatch: {key}')
        checked+=1
    if checked<10: raise ValueError('blocked_integrity: source manifest has too few verified entries')
    return {'sha256':sha256(path),'checkedEntries':checked}


def label_event(y, p_ref):
    if y is None or p_ref is None or not (math.isfinite(float(y)) and math.isfinite(float(p_ref))) or y <= 0 or p_ref <= 0:
        return None
    return int(abs(float(y) - float(p_ref)) >= .20 * float(p_ref))


def row_features(row, method):
    """Return contract features; RV missing SD is represented by mask + train mean imputation."""
    features = row['features']
    names = FEATURES_RB if method == 'RB' else FEATURES_RV
    out = [float(features[n]) for n in FEATURES_RB]
    if method == 'RV':
        available = int(features.get('price_available_28', 0))
        sd = float(features.get('price_sd_28', 0.0)) if available else float('nan')
        out.extend((sd, float(available)))
    return np.asarray(out, dtype=float), names


def _matrix(rows, method, stats=None):
    names = FEATURES_RB if method == 'RB' else FEATURES_RV
    numerical = NUMERIC_RB if method == 'RB' else NUMERIC_RV
    x = np.vstack([row_features(r, method)[0] for r in rows])
    col = {name: i for i, name in enumerate(names)}
    if stats is None:
        stats = {}
        for name in numerical:
            vals = x[:, col[name]]
            if name == 'price_sd_28':
                vals = vals[np.isfinite(vals)]
            if not len(vals):
                raise ValueError('insufficient_feature_support:' + name)
            mean = float(vals.mean()); scale = float(vals.std(ddof=0))
            stats[name] = {'mean': mean, 'scale': scale if scale > 0 else 1.0}
    for name in numerical:
        i = col[name]; mean = stats[name]['mean']; scale = stats[name]['scale']
        vals = x[:, i]
        vals = np.where(np.isfinite(vals), vals, mean)
        x[:, i] = (vals - mean) / scale
    # Fixed one-hot cabbage/radish columns; masks and category columns stay unscaled.
    crops = np.asarray([r['crop'] for r in rows])
    x = np.column_stack((x, (crops == 'cabbage').astype(float), (crops == 'radish').astype(float)))
    if not np.isfinite(x).all():
        raise ValueError('nonfinite_design_matrix')
    return x, stats


def assert_training_maturity(rows, origin):
    for row in rows:
        end = as_date(row['targetEnd'])
        if end + dt.timedelta(days=2) > origin:
            raise ValueError('future_or_unmatured_training_label:' + str(row.get('targetStart')))


def training_rows(catalog, origin):
    eligible = []
    for r in catalog:
        end = as_date(r['targetEnd'])
        if end + dt.timedelta(days=2) <= origin and r.get('actual') is not None and r.get('p_ref') and r['p_ref'] > 0 and r.get('features'):
            y = label_event(r['actual'], r['p_ref'])
            if y is not None:
                eligible.append({**r, 'event': y})
    return eligible


def fit_probabilities(train, tests, method):
    """One pooled chronological fit per origin/method predicts both crop rows."""
    classes = [r['event'] for r in train]
    crop_n = {c: sum(r['crop'] == c for r in train) for c in CROPS}
    if any(crop_n[c] < SUPPORT['per_crop_rows'] for c in CROPS) or min(classes.count(0), classes.count(1)) < SUPPORT['pooled_each_class']:
        return [None for _ in tests], {'status':'insufficient_class_support','trainingRows':len(train),'trainingByCrop':crop_n,
                      'trainingPositive':classes.count(1),'trainingNegative':classes.count(0)}
    assert_training_maturity(train, tests[0]['origin'] if isinstance(tests[0]['origin'], dt.date) else dt.date.fromisoformat(tests[0]['origin']))
    x_train, stats = _matrix(train, method)
    x_test, _ = _matrix(tests, method, stats)
    model = LogisticRegression(**MODEL)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(x_train, np.asarray(classes, dtype=int))
    if caught:
        raise RuntimeError('logistic_convergence_warning:' + '; '.join(str(w.message) for w in caught))
    probs = [float(v) for v in model.predict_proba(x_test)[:, list(model.classes_).index(1)]]
    if any(not math.isfinite(prob) or not 0 <= prob <= 1 for prob in probs):
        raise RuntimeError('invalid_probability')
    return probs, {'status':'ok','trainingRows':len(train),'trainingByCrop':crop_n,
                  'trainingPositive':classes.count(1),'trainingNegative':classes.count(0),
                  'preprocessing':stats,'model':MODEL.copy()}


def fit_probability(train, test, method):
    """Scalar convenience wrapper used by focused synthetic checks."""
    probabilities, info = fit_probabilities(train, [test], method)
    return probabilities[0], info


def score_rows(rows):
    """Paired RB/RV scoring after forecasts have been persisted."""
    out = {}
    for year in sorted({int(r['year']) for r in rows}):
        for crop in CROPS:
            group = [r for r in rows if int(r['year']) == year and r['crop'] == crop and r.get('event') is not None
                     and r.get('prob_RB') is not None and r.get('prob_RV') is not None]
            out[f'{year}:{crop}'] = score_group(group)
    return out


def score_group(rows):
    rows = [r for r in rows if r.get('event') is not None and r.get('prob_RB') is not None and r.get('prob_RV') is not None]
    n = len(rows); ys = np.asarray([int(r['event']) for r in rows], dtype=int)
    result = {'n':n,'positive':int(ys.sum()),'negative':int(n-ys.sum()),'prevalence':float(ys.mean()) if n else None}
    for method in ('RB','RV'):
        p = np.asarray([float(r['prob_'+method]) for r in rows], dtype=float)
        if n:
            result[method] = {'brier':float(np.mean((p-ys)**2)),
                              'logLoss':float(log_loss(ys, np.clip(p,1e-15,1-1e-15), labels=[0,1])),
                              'prAuc':float(average_precision_score(ys,p)) if len(set(ys)) == 2 else None,
                              'brierPositiveContribution':float(np.sum((p[ys==1]-1)**2)),
                              'brierNegativeContribution':float(np.sum(p[ys==0]**2))}
        else: result[method] = {'brier':None,'logLoss':None,'prAuc':None}
    return result


def development_gate(scores):
    """Frozen protocol gate: year-level equal-crop Brier improvements plus support/LL guard."""
    details = {}
    for year in (2022, 2023):
        cells = [scores.get(f'{year}:{c}', {}) for c in CROPS]
        supported = all(x.get('n',0) >= SUPPORT['gate_each_crop_each_year'] and
                        x.get('positive',0) >= SUPPORT['gate_each_class_each_crop_year'] and
                        x.get('negative',0) >= SUPPORT['gate_each_class_each_crop_year'] for x in cells)
        if supported:
            db = float(np.mean([x['RV']['brier']-x['RB']['brier'] for x in cells]))
            dll = float(np.mean([x['RV']['logLoss']-x['RB']['logLoss'] for x in cells]))
        else: db = dll = None
        details[str(year)] = {'support':supported,'brierDeltaEqualCrop':db,'logLossDeltaEqualCrop':dll,
                              'cropScores':{c:scores.get(f'{year}:{c}') for c in CROPS}}
    if not all(x['support'] for x in details.values()): status='inconclusive_support'
    elif all(x['brierDeltaEqualCrop'] < 0 for x in details.values()) and all(x['logLossDeltaEqualCrop'] <= SUPPORT['log_loss_delta_max'] for x in details.values()): status='pass'
    elif any(x['brierDeltaEqualCrop'] >= 0 for x in details.values()):
        status='year_inconsistent' if any(x['brierDeltaEqualCrop'] < 0 for x in details.values()) else 'risk_signal_inconsistent'
    else: status='risk_signal_inconsistent'
    return {'status':status,'testFitsAllowed':status=='pass','years':details,'rules':SUPPORT.copy()}


def calibration(rows):
    bins = [(0,.2),( .2,.4),(.4,.6),(.6,.8),(.8,1.0000000001)]
    result={}
    for method in ('RB','RV'):
        items=[]
        for lo,hi in bins:
            selected=[r for r in rows if r.get('event') is not None and r.get('prob_'+method) is not None and lo <= float(r['prob_'+method]) < hi]
            items.append({'lower':lo,'upper':hi,'n':len(selected),'meanProbability':float(np.mean([r['prob_'+method] for r in selected])) if selected else None,
                          'eventRate':float(np.mean([r['event'] for r in selected])) if selected else None,
                          'support':'insufficient_n_lt_10' if len(selected)<10 else 'supported_descriptively'})
        result[method]=items
    return result


def period_index(day):
    if isinstance(day,str): day=dt.date.fromisoformat(day)
    slot=0 if day.day<=10 else 1 if day.day<=20 else 2
    return day.year*36+(day.month-1)*3+slot


def dm_hac(rows, metric, lag):
    """Paired RV-RB loss DM statistic with calendar-distance Bartlett HAC."""
    vals=[]
    for r in rows:
        if '_lossDelta' in r:
            vals.append((period_index(r['targetStart']),float(r['_lossDelta'])))
            continue
        if r.get('event') is None or r.get('prob_RB') is None or r.get('prob_RV') is None: continue
        y=int(r['event']); rb=float(r['prob_RB']); rv=float(r['prob_RV'])
        if metric=='brier': a,b=(rb-y)**2,(rv-y)**2
        else:
            a,b=-math.log(min(1-1e-15,max(1e-15,rb))) if y else -math.log(1-min(1-1e-15,max(1e-15,rb))),\
                -math.log(min(1-1e-15,max(1e-15,rv))) if y else -math.log(1-min(1-1e-15,max(1e-15,rv)))
        vals.append((period_index(r['targetStart']), (b-a)))
    if len(vals)<24: return {'status':'unavailable','n':len(vals),'reason':'n_lt_24'}
    if '_lossDelta' not in rows[0]:
        ys=[int(r['event']) for r in rows if r.get('event') is not None and r.get('prob_RB') is not None and r.get('prob_RV') is not None]
        if len(ys)<24 or ys.count(1)<5 or ys.count(0)<5:
            return {'status':'unavailable','n':len(ys),'positive':ys.count(1),'negative':ys.count(0),'reason':'n_or_class_support_below_frozen_minimum'}
    d=np.asarray([x[1] for x in vals],float); ix=np.asarray([x[0] for x in vals],int); mean=float(d.mean()); centered=d-mean
    lrv=float(np.dot(centered,centered)/len(d))
    for h in range(1,lag+1):
        left={int(t):float(x) for t,x in zip(ix,centered)}
        cov=[left[t]*left[t+h] for t in left if t+h in left]
        if cov: lrv += 2*(1-h/(lag+1))*float(np.sum(cov)/len(d))
    if not math.isfinite(lrv) or lrv<=0: return {'status':'unavailable','n':len(vals),'meanDelta':mean,'reason':'nonpositive_hac_variance'}
    se=math.sqrt(lrv/len(d)); z=mean/se
    return {'status':'tested','n':len(vals),'meanDelta':mean,'hacVariance':lrv,'se':se,'z':z,'p':float(2*norm.sf(abs(z)))}


def dm_families(rows, split):
    years=sorted({int(r['year']) for r in rows})
    groups=[(str(y),[r for r in rows if int(r['year'])==y]) for y in years]
    groups.append(('all',rows)); out={}
    expected=18 if split=='develop' else 24
    for lag in (3,6):
        tests=[]
        for group_name,group in groups:
            crop_support={}
            for crop in CROPS:
                ys=[int(r['event']) for r in group if r['crop']==crop and r.get('event') is not None and r.get('prob_RB') is not None and r.get('prob_RV') is not None]
                crop_support[crop]={'n':len(ys),'positive':ys.count(1),'negative':ys.count(0)}
            pooled_support_ok=all(v['n']>=24 and v['positive']>=5 and v['negative']>=5 for v in crop_support.values())
            for metric in ('brier','logLoss'):
                # Pool crop-correlated observations by averaging the paired loss
                # difference within each target period before HAC.
                pooled=[]; by_target={}
                for r in group: by_target.setdefault(r['targetStart'],[]).append(r)
                for t,pair in by_target.items():
                    valid=[r for r in pair if r.get('event') is not None and r.get('prob_RB') is not None and r.get('prob_RV') is not None]
                    if valid:
                        delta=[]
                        for r in valid:
                            y=int(r['event']);rb=float(r['prob_RB']);rv=float(r['prob_RV'])
                            if metric=='brier': delta.append((rv-y)**2-(rb-y)**2)
                            else:
                                rb=min(1-1e-15,max(1e-15,rb));rv=min(1-1e-15,max(1e-15,rv))
                                delta.append((-math.log(rv) if y else -math.log(1-rv))-(-math.log(rb) if y else -math.log(1-rb)))
                        pooled.append({'targetStart':t,'_lossDelta':float(np.mean(delta))})
                for crop in ('pooled',)+CROPS:
                    subset=pooled if crop=='pooled' else [r for r in group if r['crop']==crop]
                    support=crop_support if crop=='pooled' else {crop:crop_support[crop]}
                    ok=pooled_support_ok if crop=='pooled' else all(v['n']>=24 and v['positive']>=5 and v['negative']>=5 for v in support.values())
                    if not ok:
                        result={'status':'unavailable','n':len(subset),'reason':'frozen_group_support_minimum_not_met','support':support}
                    else: result=dm_hac(subset,metric,lag)
                    tests.append({'name':f'{group_name}:{crop}:{metric}','result':result})
        ps=[x['result'].get('p') for x in tests]
        valid=sorted((float(p),i) for i,p in enumerate(ps) if p is not None)
        adj=[None]*len(tests); running=0.
        for rank,(p,i) in enumerate(valid):
            running=max(running,min(1.,(expected-rank)*p));adj[i]=running
        out[str(lag)]={'familySize':expected,'tests':[{'name':x['name'],**x['result'],'holmAdjustedP':adj[i]} for i,x in enumerate(tests)],
                       'interpretation':'Exploratory normal-approximation DM; temporal HAC and multiplicity applied. Non-significance is not equivalence.'}
    return out


def paired_block_bootstrap(rows, *, block, metric='brier', reps=2000, seed=42):
    """Joint crop-paired target-dekad block interval for equal-crop loss difference."""
    targets=sorted({as_date(r['targetStart']).isoformat() for r in rows})
    scored=[r for r in rows if r.get('event') is not None and r.get('prob_RB') is not None and r.get('prob_RV') is not None]
    per_crop={c:[r for r in scored if r['crop']==c] for c in CROPS}
    if len(targets)<24:return {'status':'inference_unavailable','reason':f'valid_calendar_target_periods_lt_24:{len(targets)}'}
    for c in CROPS:
        ys=[int(r['event']) for r in per_crop[c]]
        if len(ys)<24 or ys.count(1)<5 or ys.count(0)<5:
            return {'status':'inference_unavailable','reason':f'{c}_support_n_pos_neg={len(ys)}_{ys.count(1)}_{ys.count(0)}'}
    by={t:{r['crop']:r for r in scored if as_date(r['targetStart']).isoformat()==t} for t in targets}
    rng=np.random.default_rng(seed); diffs=[]
    for _ in range(reps):
        chosen=[]
        while len(chosen)<len(targets):
            starts=range(max(1,len(targets)-block+1))
            ix=int(rng.choice(list(starts)))
            chosen.extend(range(ix,min(ix+block,len(targets))))
        sampled=[targets[i] for i in chosen[:len(targets)]]
        crop_delta=[]
        for c in CROPS:
            vals=[]
            for t in sampled:
                r=by.get(t,{}).get(c)
                if not r: continue
                y=int(r['event']);rb=float(r['prob_RB']);rv=float(r['prob_RV'])
                if metric=='brier': delta=(rv-y)**2-(rb-y)**2
                else:
                    rb=min(1-1e-15,max(1e-15,rb));rv=min(1-1e-15,max(1e-15,rv))
                    delta=(-math.log(rv) if y else -math.log(1-rv))-(-math.log(rb) if y else -math.log(1-rb))
                vals.append(delta)
            if vals: crop_delta.append(float(np.mean(vals)))
        if len(crop_delta)==2: diffs.append(float(np.mean(crop_delta)))
    if len(diffs)<reps*.9:return {'status':'inference_unavailable','reason':'insufficient_joint_valid_resamples','validReplicates':len(diffs)}
    return {'status':'exploratory_percentile_interval','metric':metric,'estimand':'equal-crop mean RV-RB paired loss difference',
            'calendarTargetPeriods':len(targets),'blockLength':block,'replicates':len(diffs),'seed':seed,
            'meanDelta':float(np.mean(diffs)),'ci95':[float(np.quantile(diffs,.025)),float(np.quantile(diffs,.975))],
            'warning':'Blocks are few and periods overlap; interval is descriptive, not equivalence evidence.'}


def build_schedule(split):
    if split == 'develop': start,end=dt.date(2022,1,1),dt.date(2023,12,31)
    elif split == 'test': start,end=dt.date(2024,1,1),dt.date(2026,9,30)
    else: raise ValueError('unknown split')
    return [s for s in proxy.planned_origins(start,end,HORIZON)]


def _load_catalog(out=OUT):
    """Build once, then reuse a source/code-bound cache across every stage."""
    cache = Path(out) / 'v5-feature-catalog.parquet'
    meta_path = Path(out) / 'v5-feature-catalog.json'
    protocol=Path(out)/'protocol.md'
    if not protocol.is_file() or sha256(protocol)!='ef68429d97be7615279b3d96e5bdc9a551117ee9a1d07ab4d805d9beb1752de7':
        raise ValueError('blocked_integrity: frozen V6 protocol digest mismatch')
    request=Path(out)/'request.md'
    if not request.is_file(): raise ValueError('blocked_integrity: original V6 request missing')
    protocol_sha=sha256(protocol)
    identity = {'sourceSha256':sha256(proxy.SOURCE), 'v5CodeSha256':sha256(v5.__file__),
                'proxyCodeSha256':sha256(proxy.__file__),'coreCodeSha256':sha256(__file__),
                'protocolSha256':protocol_sha,'requestSha256':sha256(request)}
    if cache.is_file() and meta_path.is_file():
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        if meta.get('identity') != identity or meta.get('sha256') != sha256(cache):
            raise ValueError('blocked_integrity: V5 feature catalog cache identity/hash mismatch')
        import pandas as pd
        rows=pd.read_parquet(cache).to_dict('records')
        targets={c:{} for c in CROPS}
        for r in rows:
            target=(as_date(r['targetStart']),as_date(r['targetEnd']))
            targets[r['crop']][target]={'actual':optional_float(r.get('actual'))}
            r['targetStart']=as_date(r['targetStart']).isoformat();r['targetEnd']=as_date(r['targetEnd'])
            r['origin']=as_date(r['origin'])
            r['actual']=optional_float(r.get('actual'));r['p_ref']=optional_float(r.get('p_ref'))
        return rows,targets,meta
    import json as _json
    raw=_json.loads(proxy.SOURCE.read_text(encoding='utf-8')); series=proxy.series_from_raw(raw)
    targets={c:proxy._target_periods(series[c]) for c in CROPS}
    rows=make_catalog(series,targets)
    import pandas as pd
    cache.parent.mkdir(parents=True,exist_ok=True)
    tmp=cache.with_suffix('.tmp.parquet');pd.DataFrame(rows).to_parquet(tmp,index=False);os.replace(tmp,cache)
    meta={'identity':identity,'sha256':sha256(cache),'rows':len(rows),
          'features':'V5 feature_row frozen subset, as-of cutoff; full feature/meta payload retained'}
    _write_json(meta_path,meta)
    return rows,targets,meta


def make_catalog(series, targets):
    catalog=[]
    for c in CROPS:
        for sched in proxy.planned_origins(dt.date(2013,12,1),dt.date(2026,9,30),HORIZON):
            target=sched['target']; actual=targets[c].get(target,{}).get('actual')
            if sched['origin'] < dt.date(2014,1,1): continue
            meta=v5.feature_row(series[c],targets[c],c,HORIZON,target,sched['origin'])
            meta=meta or {'features':None,'referencePrice':None,'referenceDate':None,'inputCutoff':proxy.input_cutoff(sched['origin']).isoformat(),
                          'origin':sched['origin'].isoformat(),'targetStart':target[0].isoformat(),'targetEnd':target[1].isoformat()}
            catalog.append({'crop':c,'horizon':HORIZON,'targetStart':target[0].isoformat(),
                            'targetEnd':target[1],'origin':sched['origin'],'actual':float(actual) if actual is not None else None,
                            'p_ref':float(meta['referencePrice']) if meta.get('referencePrice') is not None else None,'features':meta.get('features'),
                            'featureMeta':{k:meta.get(k) for k in ('referencePrice','referenceDate','inputCutoff','targetStart','targetEnd')}})
    return catalog


def _write_json(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8');os.replace(tmp,path)
    return sha256(path)


def _write_parquet(path, rows):
    import pandas as pd
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.parquet');pd.DataFrame(rows).to_parquet(tmp,index=False);os.replace(tmp,path)
    try: shown=str(path.relative_to(ROOT))
    except ValueError: shown=str(path.resolve())
    return {'path':shown,'sha256':sha256(path),'rows':len(rows)}


def run_preflight(out=OUT):
    """Read-only integrity and exact V5 audit; blocks any risk fit on mismatch."""
    diagnostic=__import__('agri_data.period_crop_profile_diagnostics',fromlist=['verify_inputs'])
    manifest=diagnostic.verify_inputs()
    # Rebuild the frozen V5/V3 paired score at full precision before any risk fitting.
    score_report=json.loads((v5.OUT/'score-report.json').read_text(encoding='utf-8'))
    test_rows=diagnostic.build_metric_rows(require_planned_grid=True)
    score_check=diagnostic.validate_saved_score(test_rows,score_report,models=('P','B0','G-B','L-L'))
    year_check=diagnostic.validate_yearly_score(test_rows,score_report)
    test_recon=diagnostic.model_reconciliation(test_rows,('P','B0','G-B','L-L'))
    dev_report=json.loads((v5.OUT/'development-score.json').read_text(encoding='utf-8'))
    dev_g=diagnostic.read_frame(v5.OUT/'development-forecasts.parquet')
    dev_p=diagnostic.read_frame(v5.V3_OUT/'forecasts.parquet')
    dev_series=proxy.series_from_raw(json.loads(proxy.SOURCE.read_text(encoding='utf-8')))
    dev_rows=diagnostic.build_development_rows(dev_g,dev_p,dev_series)
    dev_score_check=diagnostic.validate_saved_score(dev_rows,dev_report,models=('P','G-B','L-L'))
    dev_recon=diagnostic.model_reconciliation(dev_rows,('P','G-B','L-L'))
    catalog,targets,cache_meta=_load_catalog(out)
    plans={}
    for split in ('develop','test'):
        sched=build_schedule(split);plans[split]=len(sched)*len(CROPS)
    # Verify actual V5 feature definitions and 2026 cutoff target cap.
    for r in catalog:
        if r['horizon'] != 3 or r['crop'] not in CROPS: raise ValueError('unexpected_catalog_scope')
        if r['featureMeta']['inputCutoff'] != proxy.input_cutoff(r['origin']).isoformat(): raise ValueError('blocked_integrity: cutoff mismatch')
        if r['features'] is not None and int(r['features']['price_available_28']) != int(r['features']['price_n_28'] > 0 and r['features']['reference_available']): raise ValueError('blocked_integrity: 28-day mask mismatch')
    # Compare all overlapping saved V5 H3 issue rows against the reconstructed V5 feature_row.
    bykey={(r['crop'],as_date(r['origin']).isoformat(),r['targetStart'],as_date(r['targetEnd']).isoformat()):r for r in catalog}
    comparisons=[];unsupported_saved=0
    for stage in ('develop','forecast'):
        checkpoint=v5.load_stage(stage, v5.OUT)
        for old in v5.read_rows(checkpoint['forecastArtifact']):
            if old['crop'] not in CROPS or int(old['horizon']) != HORIZON: continue
            if old.get('policyUnavailable') or old.get('p_ref') is None or old.get('asOfFeatures') is None:
                unsupported_saved += 1
                continue
            k=(old['crop'],as_date(old['origin']).isoformat(),as_date(old['targetStart']).isoformat(),as_date(old['targetEnd']).isoformat())
            current=bykey.get(k)
            if current is None:
                raise ValueError('blocked_integrity: V5 H3 row absent from reconstructed feature catalog '+repr(k))
            if current['featureMeta']['inputCutoff'] != old['inputCutoff']:
                raise ValueError('blocked_integrity: V5 feature cutoff mismatch '+repr(k))
            if current.get('p_ref') is None or current.get('features') is None: raise ValueError('blocked_integrity: saved feature support differs '+repr(k))
            if not math.isclose(current['p_ref'],float(old['p_ref']),rel_tol=0,abs_tol=1e-9):
                raise ValueError('blocked_integrity: V5 p_ref mismatch '+repr(k))
            prior=old.get('asOfFeatures') or {}
            max_abs=0.0
            for name,value in current['features'].items():
                if name not in prior: raise ValueError('blocked_integrity: saved V5 feature absent '+name)
                err=abs(float(value)-float(prior[name]));max_abs=max(max_abs,err)
                if err>1e-12: raise ValueError(f'blocked_integrity: V5 feature mismatch {name} {k} delta={err}')
            comparisons.append({'split':stage,'key':list(k),'featureCount':len(current['features']),'maxAbsDelta':max_abs})
    # Include relevant immutable source identities; source files are hashed, never copied.
    source_manifest={**manifest}
    for key,path in {
        'V5:analysisCode':Path(v5.__file__), 'V5:scoreCode':Path(v5.__file__).with_name('period_pooled_lgbm_score.py'),
        'proxy:analysisCode':Path(proxy.__file__), 'V6:request':Path(out)/'request.md', 'V6:protocol':Path(out)/'protocol.md',
    }.items():
        if path.is_file(): source_manifest[key]={'path':str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),'sha256':sha256(path)}
    source_manifest['V6:protocolExpectedSha256']='ef68429d97be7615279b3d96e5bdc9a551117ee9a1d07ab4d805d9beb1752de7'
    if (Path(out)/'protocol.md').is_file() and sha256(Path(out)/'protocol.md') != source_manifest['V6:protocolExpectedSha256']:
        raise ValueError('blocked_integrity: frozen V6 protocol digest mismatch')
    _write_json(Path(out)/'source-manifest.json',source_manifest)
    _write_json(Path(out)/'input-and-feature-audit.json',{'savedV5FeatureComparisons':len(comparisons),'savedV5H3UnsupportedRows':unsupported_saved,'comparisons':comparisons,
                'definitions':payload_feature_definition()})
    payload={'status':'pass','plans':plans,'maxAttempts':MAX_ATTEMPTS,'plannedAttempts':sum(plans.values()),
             'catalogRows':len(catalog),'catalogCache':cache_meta,'targetEndMax':'2026-09-30','testPeriod':'exploratory_reused_period',
             'v5InputManifest':manifest,'sourceSha256':sha256(proxy.SOURCE),
             'savedV5FeatureComparisons':len(comparisons),'savedV5H3UnsupportedRows':unsupported_saved,
             'priceScoreReconciliation':{'test':{'check':score_check,'yearlyCheck':year_check,'metrics':test_recon},
                                         'development':{'check':dev_score_check,'metrics':dev_recon}},
             'featureDefinition':payload_feature_definition(),
             'note':'Read-only V5/V3 price-score and feature reconciliation completed; preflight performed no risk-model fitting.'}
    cache_meta=json.loads((Path(out)/'v5-feature-catalog.json').read_text(encoding='utf-8'))
    payload['runIdentity']={**run_identity(out),'catalogSha256':cache_meta['sha256'],
                            'sourceManifestSha256':sha256(Path(out)/'source-manifest.json'),
                            'featureAuditSha256':sha256(Path(out)/'input-and-feature-audit.json')}
    _write_json(Path(out)/'preflight.json',payload)
    return payload


def payload_feature_definition():
    return {'price_sd_28':'ddof=0 SD of positive observed daily log price levels during [cutoff-27d, cutoff], V5 price_available_28 mask',
            'price_n_28':'positive observed daily prices in [cutoff-27d, cutoff]',
            'p_ref':'latest positive observed daily price on/before input_cutoff; no age cap',
            'input_cutoff':'issue origin minus 2 calendar days',
            'risk_label':'abs(Y/p_ref-1)>=0.20; for scoring only, never an input'}


def verify_preflight(out=OUT):
    out=Path(out);pre=json.loads((out/'preflight.json').read_text(encoding='utf-8'))
    cache_meta=json.loads((out/'v5-feature-catalog.json').read_text(encoding='utf-8'))
    manifest_check=verify_source_manifest(out)
    expected={**run_identity(out),'catalogSha256':cache_meta['sha256'],
              'sourceManifestSha256':manifest_check['sha256'],
              'featureAuditSha256':sha256(out/'input-and-feature-audit.json')}
    if pre.get('status')!='pass' or pre.get('runIdentity')!=expected:
        raise ValueError('blocked_integrity: stale or mismatched V6 preflight identity')
    return pre


def run_forecast(split, out=OUT):
    """Write label-free forecasts first; only call test after development gate pass."""
    out=Path(out);pre=verify_preflight(out)
    if split=='test':
        gate=json.loads((out/'development-gate.json').read_text())
        dev_ledger=json.loads((out/'development-fit-ledger.json').read_text(encoding='utf-8'))
        dev_forecast=out/'development-risk-forecasts.parquet';dev_score=out/'development-risk-scores.json'
        if (gate.get('status')!='pass' or gate.get('testFitsAllowed') is not True or
            gate.get('runIdentity')!=pre['runIdentity'] or gate.get('forecastSha256')!=sha256(dev_forecast) or
            gate.get('scoreSha256')!=sha256(dev_score) or gate.get('ledgerSha256')!=sha256(out/'development-fit-ledger.json') or
            gate.get('preflightSha256')!=sha256(out/'preflight.json') or not dev_ledger.get('complete')):
            raise ValueError('test blocked by development gate/checkpoint identity')
    label='development' if split=='develop' else 'test'
    artifact_path=out/f'{label}-risk-forecasts.parquet'
    ledger_path=out/f'{label}-fit-ledger.json'
    if artifact_path.is_file() or ledger_path.is_file():
        if not artifact_path.is_file() or not ledger_path.is_file(): raise ValueError('incomplete stage checkpoint; refusing duplicate fits')
        meta=json.loads(ledger_path.read_text(encoding='utf-8'))
        if (meta.get('forecastSha256') != sha256(artifact_path) or not meta.get('complete') or
            meta.get('runIdentity')!=pre['runIdentity'] or meta.get('preflightSha256')!=sha256(out/'preflight.json')):
            raise ValueError('stage checkpoint identity/hash mismatch')
        return {'rows':meta['rows'],'attempts':meta['attempts'],'artifact':{'path':str(artifact_path.relative_to(ROOT)),'sha256':sha256(artifact_path),'rows':meta['rows']},'checkpoint':'reused'}
    catalog,targets,_=_load_catalog(out);forecasts=[]; ledger=[]
    attempts=0
    if split=='test':
        prior=json.loads((out/'development-fit-ledger.json').read_text(encoding='utf-8'))
        if not prior.get('complete') or prior.get('runIdentity')!=pre['runIdentity']:
            raise ValueError('blocked_integrity: development ledger incomplete or stale')
        attempts=int(prior.get('cumulativeFitAttempts',prior.get('attempts',0)))
    stage_start_attempts=attempts
    def save_ledger(*,complete=False,rows_count=None,forecast_sha=None):
        data={'attempts':attempts-stage_start_attempts,'cumulativeFitAttempts':attempts,
              'planned':len(build_schedule(split))*2,'entries':ledger,'complete':complete,
              'runIdentity':pre['runIdentity'],'preflightSha256':sha256(out/'preflight.json')}
        if complete: data.update({'rows':rows_count,'forecastSha256':forecast_sha})
        return _write_json(ledger_path,data)
    for sched in build_schedule(split):
        origin=sched['origin'];target=sched['target']
        test_rows=[]
        for c in CROPS:
            match=next((r for r in catalog if r['crop']==c and r['targetStart']==target[0].isoformat() and r['targetEnd']==target[1]),None)
            # The schedule row is retained even when the frozen feature table has no usable reference.
            test_rows.append({'crop':c,'horizon':HORIZON,'origin':origin,'targetStart':target[0].isoformat(),
                              'targetEnd':target[1].isoformat(),'p_ref':match.get('p_ref') if match else None,
                              'features':match.get('features') if match else None,
                              'featureMeta':match.get('featureMeta') if match else None,
                              'rowStatus':'ready' if match and match.get('p_ref') is not None and match.get('features') is not None else 'reference_unavailable'})
        train=[r for r in training_rows(catalog,origin) if r['targetEnd'] <= target[1]]
        # Do not expose target Y to predictions. It is joined only in score stage.
        ready=[test for test in test_rows if test['rowStatus']=='ready']
        for test in test_rows:
            row={k:v for k,v in test.items() if k not in ('features','actual')}; row['year']=target[0].year;row['split']=split
            if test['rowStatus'] != 'ready':
                row.update({'prob_RB':None,'prob_RV':None,'status_RB':'reference_unavailable','status_RV':'reference_unavailable'})
            else: row['_ready']=True
            forecasts.append(row)
        if not ready: continue
        if not any(int(r['features'].get('price_available_28',0)) == 1 for r in train):
            for row in forecasts[-len(test_rows):]:
                if row.get('_ready'):
                    row.update({'prob_RB':None,'prob_RV':None,'status_RB':'insufficient_feature_support','status_RV':'insufficient_feature_support'})
                    row.pop('_ready',None)
            ledger.append({'attempt':attempts,'split':split,'origin':origin.isoformat(),
                           'method':'RB+RV','status':'insufficient_feature_support','detail':'no mature training row has an available price_sd_28'})
            continue
        for method in ('RB','RV'):
            attempts += 1
            if attempts > MAX_ATTEMPTS: raise ValueError('blocked_budget')
            ledger.append({'attempt':attempts,'split':split,'origin':origin.isoformat(),'method':method,'status':'started'})
            save_ledger()
            try:
                fit_tests=[dict(t,origin=origin) for t in ready]
                probs, info=fit_probabilities(train,fit_tests,method)
                for test,prob in zip(ready,probs):
                    row=next(z for z in forecasts[-len(test_rows):] if z['crop']==test['crop'])
                    row['prob_'+method]=prob;row['status_'+method]=info['status']
                ledger[-1]={'attempt':attempts,'split':split,'origin':origin.isoformat(),'method':method,**info}
            except Exception as exc:
                ledger[-1]={'attempt':attempts,'split':split,'origin':origin.isoformat(),'method':method,'status':'technical_error','error':type(exc).__name__+': '+str(exc)}
                save_ledger()
                raise
            save_ledger()
        for row in forecasts[-len(test_rows):]: row.pop('_ready',None)
    artifact=_write_parquet(out/f'{label}-risk-forecasts.parquet',forecasts)
    ledger_sha=save_ledger(complete=True,rows_count=len(forecasts),forecast_sha=artifact['sha256'])
    return {'rows':len(forecasts),'attemptsThisStage':attempts-stage_start_attempts,'cumulativeFitAttempts':attempts,'artifact':artifact,'ledgerSha256':ledger_sha}


def run_score(split, out=OUT):
    out=Path(out);label='development' if split=='develop' else 'test'
    pre=verify_preflight(out)
    p=out/f'{label}-risk-forecasts.parquet'
    if not p.is_file():raise ValueError('missing label-free forecast artifact')
    ledger_path=out/f'{label}-fit-ledger.json'
    if not ledger_path.is_file(): raise ValueError('missing fit ledger')
    ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
    if (not ledger.get('complete') or ledger.get('forecastSha256')!=sha256(p) or ledger.get('runIdentity')!=pre['runIdentity'] or
        ledger.get('preflightSha256')!=sha256(out/'preflight.json') or any(x.get('status')=='started' for x in ledger.get('entries',[]))):
        raise ValueError('blocked_integrity: forecast/ledger/preflight checkpoint mismatch')
    if split=='test':
        gate=json.loads((out/'development-gate.json').read_text(encoding='utf-8'))
        dev_forecast=out/'development-risk-forecasts.parquet';dev_score=out/'development-risk-scores.json';dev_ledger=out/'development-fit-ledger.json'
        if (gate.get('testFitsAllowed') is not True or gate.get('runIdentity')!=pre['runIdentity'] or
            gate.get('forecastSha256')!=sha256(dev_forecast) or gate.get('scoreSha256')!=sha256(dev_score) or
            gate.get('ledgerSha256')!=sha256(dev_ledger) or gate.get('preflightSha256')!=sha256(out/'preflight.json')):
            raise ValueError('test scoring blocked by development gate identity')
    import pandas as pd
    rows=pd.read_parquet(p).to_dict('records')
    if any('actual' in r or 'event' in r for r in rows): raise ValueError('forecast artifact contains scoring labels')
    for r in rows:
        r['prob_RB']=optional_probability(r.get('prob_RB'))
        r['prob_RV']=optional_probability(r.get('prob_RV'))
    expected={(c,s['origin'].isoformat(),s['target'][0].isoformat(),s['target'][1].isoformat()) for c in CROPS for s in build_schedule(split)}
    keys={(r['crop'],as_date(r['origin']).isoformat(),as_date(r['targetStart']).isoformat(),as_date(r['targetEnd']).isoformat()) for r in rows}
    if len(rows)!=len(expected) or keys!=expected: raise ValueError('blocked_integrity: risk forecast schedule key contract mismatch')
    for r in rows:
        if (r.get('prob_RB') is None) != (r.get('prob_RV') is None): raise ValueError('blocked_integrity: RB/RV paired coverage differs')
        if r.get('prob_RB') is None and r.get('status_RB') != r.get('status_RV'): raise ValueError('blocked_integrity: unavailable method status mismatch')
    _,targets,_=_load_catalog(out)
    joined=[]
    for r in rows:
        target=(as_date(r['targetStart']),as_date(r['targetEnd']))
        actual=optional_float(targets[r['crop']].get(target,{}).get('actual'))
        event=label_event(actual,r.get('p_ref'))
        joined.append({**r,'event':event,'actual':actual})
    scores=score_rows(joined)
    years=sorted({int(r['year']) for r in joined})
    scopes={str(y):[r for r in joined if int(r['year'])==y] for y in years};scopes['all']=joined
    block_intervals={}
    for scope,subset in scopes.items():
        block_intervals[scope]={str(k):{metric:paired_block_bootstrap(subset,block=k,metric=metric,seed=42)
                                               for metric in ('brier','logLoss')} for k in (6,12)}
    obj={'split':split,'period': 'exploratory_reused_period' if split=='test' else 'development',
         'forecastSha256':sha256(p),'plannedRows':len(build_schedule(split))*len(CROPS),'issuedRows':len(rows),
         'scoredRows':sum(r['event'] is not None and r['prob_RB'] is not None and r['prob_RV'] is not None for r in joined),
         'byYearCrop':scores,'calibration':calibration(joined),
         'blockBootstrap':block_intervals,
         'pairedDmHac':dm_families(joined,split),'runIdentity':pre['runIdentity'],
         'ledgerSha256':sha256(ledger_path),'preflightSha256':sha256(out/'preflight.json')}
    score_path=out/f'{label}-risk-scores.json'
    if score_path.exists():
        existing=json.loads(score_path.read_text(encoding='utf-8'))
        if existing!=obj: raise ValueError('blocked_integrity: scored checkpoint differs from recomputed deterministic score')
        return existing
    _write_json(score_path,obj)
    return obj


def run_gate(out=OUT):
    out=Path(out);p=out/'development-risk-scores.json';scores=json.loads(p.read_text())['byYearCrop']
    pre=verify_preflight(out)
    score=json.loads(p.read_text(encoding='utf-8'))
    ledger_path=out/'development-fit-ledger.json';forecast_path=out/'development-risk-forecasts.parquet'
    ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
    if (score.get('runIdentity')!=pre['runIdentity'] or score.get('forecastSha256')!=sha256(forecast_path) or
        score.get('ledgerSha256')!=sha256(ledger_path) or score.get('preflightSha256')!=sha256(out/'preflight.json') or
        not ledger.get('complete') or ledger.get('forecastSha256')!=sha256(forecast_path) or
        ledger.get('runIdentity')!=pre['runIdentity'] or ledger.get('preflightSha256')!=sha256(out/'preflight.json') or
        any(x.get('status')=='started' for x in ledger.get('entries',[]))):
        raise ValueError('blocked_integrity: development score/forecast/ledger mismatch')
    gate_path=out/'development-gate.json'
    gate=development_gate(scores);gate['scoreSha256']=sha256(p);gate['testFitsAllowed']=gate['status']=='pass'
    gate.update({'runIdentity':pre['runIdentity'],'forecastSha256':sha256(forecast_path),
                 'ledgerSha256':sha256(ledger_path),'preflightSha256':sha256(out/'preflight.json')})
    if gate_path.is_file():
        existing=json.loads(gate_path.read_text(encoding='utf-8'))
        if existing!=gate: raise ValueError('blocked_integrity: frozen development gate differs from recomputed decision')
        return existing
    _write_json(gate_path,gate)
    return gate


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('preflight','develop','development-score','development-gate','test','test-score'))
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args(argv)
    if args.stage=='preflight': result=run_preflight(args.out)
    elif args.stage=='develop': result=run_forecast('develop',args.out)
    elif args.stage=='development-score': result=run_score('develop',args.out)
    elif args.stage=='development-gate': result=run_gate(args.out)
    elif args.stage=='test': result=run_forecast('test',args.out)
    else: result=run_score('test',args.out)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))


if __name__=='__main__': main()
