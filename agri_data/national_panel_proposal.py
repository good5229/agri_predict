"""Frozen, offline national market-panel proposal experiment.

Commands create-manifest and score are deliberately separate. The manifest has
no outcomes; scoring requires byte-identical inputs and regenerates forecasts.
"""
from __future__ import annotations

import argparse, csv, datetime as dt, hashlib, io, json, math, zipfile
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

from . import national_transfer as nt

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/raw/national-wholesale-20240708-20260630.zip"
MANIFEST = ROOT / "outputs/experiments/national-panel-proposal/forecast-manifest.json"
REPORT = ROOT / "outputs/experiments/national-panel-proposal/report.json"
TRAIN_END = dt.date(2025, 9, 30); DIAG_START = dt.date(2025, 10, 1); DIAG_END = dt.date(2025, 12, 31)
TEST_START = dt.date(2026, 1, 1); TEST_END = dt.date(2026, 6, 30)
MIN_TRAIN_DAYS = 90; HORIZONS = range(1, 11); HAC_DAYS = (9, 28); FAMILY = 80
PROTOCOL = {"id":"national-panel-proposal-2026-10-09-v1", "archiveSha256":nt.ARCHIVE_SHA256,
 "crops":nt.PRODUCT_CODES, "trainThrough":TRAIN_END.isoformat(), "diagnostic":[DIAG_START.isoformat(),DIAG_END.isoformat()],
 "test":[TEST_START.isoformat(),TEST_END.isoformat()], "horizons":list(HORIZONS), "featureCutoff":"origin minus 2 calendar days",
 "direct":"per crop/horizon StandardScaler+Ridge(alpha=100), log target/recent national price; 14d last value, 7/28d changes/levels, month/weekday",
 "panel":"per crop/horizon shared Ridge(alpha=100), log market target/recent national price; market one-hot, market/national ratio and 7d changes with missing indicators, asinh signed recent volume, signed 28d share, age, month/weekday",
 "diagnosticEligibility":"Oct-Dec 2025 rows with origin-2 before 2025-09-30 remain planned and are marked training_labels_not_matured; diagnostics do not retune or select",
 "minimumTrainingTargets":MIN_TRAIN_DAYS,"share":"same-weekday recent 8-week median of signed market net quantity / signed national net quantity, >=4 dates per market; else that market's 28-calendar-day average; clip only final market weights at zero, then normalize; audit negative/null/duplicate inputs and clipped shares",
 "fallback":"stale market >14 days uses direct prediction; no usable market shares uses direct; unavailable market forecast receives direct price; model failures or nonpositive values use direct then last valid national price",
 "inference":"4 crops x 10 horizons x AE/APE=80 two-sided DM tests; calendar Bartlett HAC9 primary/HAC28 sensitivity; separate Holm .05; unavailable tests retained p=1; min 30; target-date calendar gaps",
 "limitations":"revised retrospective archive; first publication timestamp unknown; not prospective or official contest-score evidence"}

def sha(b): return hashlib.sha256(b).hexdigest()
def _date(s): return s if isinstance(s,dt.date) else dt.date.fromisoformat(s)
def _features(series, origin, crop, horizon):
    # series: date -> positive price; cutoff enforces two-day latency.
    cutoff=origin-dt.timedelta(days=2); hist=[(d,p) for d,p in series.items() if d<=cutoff]
    hist.sort(); recent=[x for x in hist if x[0]>=cutoff-dt.timedelta(days=14)]
    if not recent: return None
    last=recent[-1][1]
    def avg(days):
        x=[p for d,p in hist if cutoff-dt.timedelta(days=days)<d<=cutoff]
        return sum(x)/len(x) if x else None
    a7,a28=avg(7),avg(28)
    f=[math.log(last), math.log(a7) if a7 else 0., math.log(a28) if a28 else 0.,
       math.log(last/a7) if a7 else 0., math.log(last/a28) if a28 else 0.,
       float(a7 is None),float(a28 is None),math.sin(2*math.pi*origin.month/12),math.cos(2*math.pi*origin.month/12),
       math.sin(2*math.pi*origin.weekday()/7),math.cos(2*math.pi*origin.weekday()/7)]
    return f,last

def parse_market_archive(data, *, expected_sha256=nt.ARCHIVE_SHA256):
    """Read every source row; preserve signed aggregates and row-level audit."""
    national=defaultdict(lambda:[Decimal(0),Decimal(0),0,True]); market=defaultdict(lambda:[Decimal(0),Decimal(0),0,True]); audit=defaultdict(lambda:defaultdict(int)); names=defaultdict(set); dups=defaultdict(int); seen=defaultdict(int); rows=0
    # The shared reader validates the four frozen periods, safe member names,
    # and the pinned archive digest; its national totals are our reconciliation.
    frozen=nt.parse_archive(data,expected_sha256=expected_sha256)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            raw=z.read(info); text=raw.decode('cp949'); reader=csv.DictReader(io.StringIO(text,newline=''))
            if tuple(reader.fieldnames or ())!=nt.EXPECTED_HEADERS: raise ValueError('archive CSV schema changed')
            for n,r in enumerate(reader,2):
                if None in r or any(v is None for v in r.values()): raise ValueError(f'malformed row {info.filename}:{n}')
                code=r['품목코드'].strip(); crop=next((p for p,c in nt.PRODUCT_CODES.items() if code==c),None)
                if not crop: continue
                day=nt._parse_day(r['경매일자']); m=r['도매시장코드'].strip(); name=r['품목명'].strip()
                if not m: raise ValueError('blank market code')
                try: q=nt._decimal(r['변경총물량'], 'quantity'); a=nt._decimal(r['경매낙찰총금액'],'amount')
                except ValueError: raise
                rows+=1; names[crop].add(name); sk=(crop,day); mk=(crop,m,day); audit[sk]['rows']+=1; audit[mk]['rows']+=1
                rawkey=tuple(r[h] for h in nt.EXPECTED_HEADERS); seen[rawkey]+=1
                if seen[rawkey]>1: dups[sk]+=1; dups[mk]+=1
                if name!=nt.PRODUCT_NAMES[crop]: audit[sk]['nameConflict']+=1; audit[mk]['nameConflict']+=1
                for key,obj in ((sk,national),(mk,market)):
                    x=obj[key]; x[2]+=1
                    if a is None: audit[key]['nullAmount']+=1; x[3]=False
                    else: x[0]+=a; audit[key]['negativeAmount']+=int(a<0)
                    if q is None: audit[key]['nullQuantity']+=1; x[3]=False
                    else: x[1]+=q; audit[key]['negativeQuantity']+=int(q<0)
                    if name!=nt.PRODUCT_NAMES[crop]: x[3]=False
    if not rows: raise ValueError('no frozen crop rows')
    def pack(obj,keyfn):
        out={}
        for k,(a,q,n,complete) in obj.items():
            auditrow=audit[k]; reason=None
            price=a/q if complete and q>0 else None
            if auditrow['nameConflict']: reason='product_name_conflict'
            elif not complete: reason='null_components'
            elif q<=0: reason='nonpositive_signed_quantity'
            elif price<=0: reason='nonpositive_aggregate_price'
            out[keyfn(k)]={'amount':str(a),'quantity':str(q),'price':str(price) if price is not None else None,'usable':price is not None and price>0 and complete,
                'quantityUsable':not bool(auditrow['nullQuantity']) and not bool(auditrow['nameConflict']),
                'rowCount':n,'nullAmountCount':auditrow['nullAmount'],'nullQuantityCount':auditrow['nullQuantity'],'negativeAmountCount':auditrow['negativeAmount'],'negativeQuantityCount':auditrow['negativeQuantity'],'duplicateRowCount':dups[k],'nameConflictCount':auditrow['nameConflict'],'reason':reason}
        return out
    national_rows=pack(national,lambda k:(k[0],k[1].isoformat()))
    # Reconcile every crop/day aggregate and its price usability with the
    # independently validated national-only parser.
    for (crop,day),v in national_rows.items():
        expected=frozen['observations'][crop].get(day)
        if expected is None or v['amount']!=expected['amountSum'] or v['quantity']!=expected['quantitySum']:
            raise ValueError(f'national aggregate reconciliation failed: {crop} {day}')
        if v['usable'] != (expected['price'] is not None and Decimal(expected['price'])>0 and expected['priceReason'] is None):
            raise ValueError(f'national price validity reconciliation failed: {crop} {day}')
    return {'national':national_rows, 'market':pack(market,lambda k:(k[0],k[1],k[2].isoformat())), 'sourceCropRowCount':rows,
            'archiveAudit':{'sourceRowCount':frozen['sourceRowCount'],'exactDuplicateRowCount':frozen['exactDuplicateRowCount'],'periods':frozen['periods'],
                'selectedCropRows':rows,'selectedMarketDayAggregates':len(market),
                'nullAmountRows':sum(x['nullAmountCount'] for x in pack(market,lambda k:k).values()),
                'nullQuantityRows':sum(x['nullQuantityCount'] for x in pack(market,lambda k:k).values()),
                'negativeAmountRows':sum(x['negativeAmountCount'] for x in pack(market,lambda k:k).values()),
                'negativeQuantityRows':sum(x['negativeQuantityCount'] for x in pack(market,lambda k:k).values()),
                'exactDuplicateSelectedRows':sum(x['duplicateRowCount'] for x in pack(market,lambda k:k).values()),
                'nameConflictRows':sum(x['nameConflictCount'] for x in pack(market,lambda k:k).values())}}

def _model(x,y):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    m=make_pipeline(StandardScaler(),Ridge(alpha=100.0)); m.fit(x,y); return m

def _forecast(parsed, dates, *, include_answers=False):
    national={c:{} for c in nt.PRODUCT_CODES}; markets={c:defaultdict(dict) for c in nt.PRODUCT_CODES}
    for (c,d),v in parsed['national'].items():
        if v['usable']: national[c][_date(d)]=float(v['price'])
    for (c,m,d),v in parsed['market'].items():
        day=_date(d)
        if v['usable']: markets[c][m][day]=float(v['price'])
    quantity_index=_quantity_index(parsed)
    rows=[]; market_ids={c:sorted(markets[c]) for c in nt.PRODUCT_CODES}
    for crop in nt.PRODUCT_CODES:
      for h in HORIZONS:
        tx=[]; ty=[]
        for td,tp in national[crop].items():
            if td<=TRAIN_END:
                z=_features(national[crop],td-dt.timedelta(days=h),crop,h)
                if z: tx.append(z[0]); ty.append(math.log(tp/z[1]))
        direct_model=None; direct_fit='insufficient_training_targets'
        if len(ty)>=MIN_TRAIN_DAYS:
            try: direct_model=_model(tx,ty); direct_fit='native'
            except Exception as exc: direct_fit='fit_failure:'+type(exc).__name__
        # Fit pooled panel once per crop/horizon from each training target's own origin features.
        X=[]; Y=[]; panel_days=set(); ids=market_ids[crop]
        for m in ids:
            series=markets[crop][m]
            for td,tp in series.items():
                if td>TRAIN_END: continue
                origin=td-dt.timedelta(days=h); cutoff=origin-dt.timedelta(days=2)
                mf=_features(series,origin,crop,h); nf=_features(national[crop],origin,crop,h)
                if not mf or not nf: continue
                last=max((d for d in series if d<=cutoff),default=None)
                if last is None: continue
                age=(cutoff-last).days
                volume=_recent_market_volume(quantity_index,crop,m,cutoff)
                share=_historical_market_share(quantity_index,crop,m,cutoff)
                X.append([*_panel_core(mf,nf,age,volume,share,origin),*[float(m==mid) for mid in ids]])
                Y.append(math.log(tp/nf[1])); panel_days.add(td)
        panel_model=None; panel_fit='insufficient_training_targets'
        if len(panel_days)>=MIN_TRAIN_DAYS:
            try: panel_model=_model(X,Y); panel_fit='native'
            except Exception as exc: panel_fit='fit_failure:'+type(exc).__name__
        for target in dates:
            origin=target-dt.timedelta(days=h); cutoff=origin-dt.timedelta(days=2)
            phase='diagnostic' if DIAG_START<=target<=DIAG_END else 'test'
            if phase=='diagnostic' and cutoff<TRAIN_END:
                rows.append({'crop':crop,'origin':origin.isoformat(),'target':target.isoformat(),'horizon':h,'phase':phase,
                    'forecastEligibility':'training_labels_not_matured','directPrediction':None,'panelPrediction':None,
                    'marketPredictions':{},'marketWeights':{},'rawMarketShareEstimates':{},'marketShareAudit':{},
                    'directFitStatus':direct_fit,'directFallbackRootCause':'training_labels_not_matured','directFallbackAction':None,
                    'panelStatus':panel_fit,'panelPredictFailures':{},'panelFallback':'ineligible',
                    'unavailableMarketWeight':None,'trainingTargets':len(ty),'trainingMarketTargetDays':len(panel_days),
                    'trainingMarketTargetRows':len(Y),'trainingEnd':TRAIN_END.isoformat()})
                continue
            nf=_features(national[crop],origin,crop,h); direct=None; direct_root=None; direct_action=None
            if nf:
                if direct_model is not None:
                    try:
                        candidate=nf[1]*math.exp(float(direct_model.predict([nf[0]])[0]))
                        if candidate>0 and math.isfinite(candidate): direct=candidate
                        else: direct_root='nonpositive_prediction'
                    except Exception as exc: direct_root='predict_failure:'+type(exc).__name__
                else: direct_root=direct_fit
                if direct is None:
                    direct=nf[1]; direct_action='last_national'; direct_root=direct_root or 'native_model_unavailable'
            else:
                # Retain a valid persistence fallback even when no national
                # observation falls inside the 14-day feature window.
                previous=max((d for d in national[crop] if d<=cutoff),default=None)
                if previous is not None:
                    direct=national[crop][previous]; direct_action='last_national'; direct_root='no_recent_national_feature'
                else:
                    direct_root='no_prior_national_value'
            share_audit=_forecast_share_detail(quantity_index,crop,target,origin); shares=share_audit['weights']; market_predictions={}; failures={}
            if panel_model is not None and nf:
                for m in ids:
                    series=markets[crop][m]; mf=_features(series,origin,crop,h)
                    if not mf: continue
                    last=max((d for d in series if d<=cutoff),default=None)
                    if last is None or (cutoff-last).days>14: continue
                    age=(cutoff-last).days
                    volume=_recent_market_volume(quantity_index,crop,m,cutoff)
                    share=_historical_market_share(quantity_index,crop,m,cutoff)
                    vector=[*_panel_core(mf,nf,age,volume,share,origin),*[float(m==mid) for mid in ids]]
                    try:
                        candidate=_anchored_market_price(nf[1],float(panel_model.predict([vector])[0]))
                        if candidate>0 and math.isfinite(candidate): market_predictions[m]=candidate
                        else: failures[m]='nonpositive_prediction'
                    except Exception as exc: failures[m]='predict_failure:'+type(exc).__name__
            panel=None; missing_weight=sum(shares.values()) if shares else 0.; fallback='no_direct_prediction'
            if direct is not None:
                if shares:
                    panel=sum(w*market_predictions.get(m,direct) for m,w in shares.items())
                    missing_weight=sum(w for m,w in shares.items() if m not in market_predictions)
                    fallback='partial_market_fallback' if missing_weight else None
                else: panel=direct; fallback='direct_no_shares'
            row={'crop':crop,'origin':origin.isoformat(),'target':target.isoformat(),'horizon':h,'phase':phase,
                 'forecastEligibility':'eligible',
                 'directPrediction':direct,'panelPrediction':panel,'marketPredictions':market_predictions,'marketWeights':shares,
                 'rawMarketShareEstimates':share_audit['signedEstimates'],'marketShareAudit':{k:v for k,v in share_audit.items() if k not in ('weights','signedEstimates')},
                 'directFitStatus':direct_fit,'directFallbackRootCause':direct_root,'directFallbackAction':direct_action,
                 'panelStatus':panel_fit,'panelFitStatus':panel_fit,'panelPredictFailures':failures,
                 'panelFallback':fallback,'unavailableMarketWeight':missing_weight,'trainingTargets':len(ty),
                 'trainingMarketTargetDays':len(panel_days),'trainingMarketTargetRows':len(Y),'trainingEnd':TRAIN_END.isoformat()}
            if include_answers:
                v=parsed['national'].get((crop,target.isoformat())); row['actual']=float(v['price']) if v and v['usable'] else None
            rows.append(row)
    return rows

def _panel_core(market_feature,national_feature,age,volume,share,origin):
    return [math.log(market_feature[1]/national_feature[1]),float(age),math.asinh(volume),float(share),
            *market_feature[0][1:5],*national_feature[0][1:5],
            market_feature[0][5],market_feature[0][6],national_feature[0][5],national_feature[0][6],
            math.sin(2*math.pi*origin.month/12),math.cos(2*math.pi*origin.month/12),
            math.sin(2*math.pi*origin.weekday()/7),math.cos(2*math.pi*origin.weekday()/7)]

def _anchored_market_price(recent_national_price, predicted_log_market_ratio):
    return recent_national_price*math.exp(predicted_log_market_ratio)

def _quantity_index(parsed):
    daily=defaultdict(lambda:defaultdict(dict)); first_seen=defaultdict(dict)
    for (crop,market,ds),v in parsed['market'].items():
        if v.get('quantityUsable',False):
            day=_date(ds); daily[crop][day][market]=float(Decimal(v['quantity']))
            first_seen[crop][market]=min(day,first_seen[crop].get(market,day))
    national={}
    for (crop,ds),v in parsed['national'].items():
        if v.get('quantityUsable',False): national[(crop,_date(ds))]=float(Decimal(v['quantity']))
    return {'daily':daily,'national':national,'firstSeen':first_seen}

def _recent_market_volume(index,crop,market,cutoff):
    daily=index['daily'][crop]
    return sum(daily.get(cutoff-dt.timedelta(days=offset),{}).get(market,0.) for offset in range(28))

def _historical_market_share(index,crop,market,cutoff):
    daily=index['daily'][crop]
    ratios=[]
    for offset in range(1,29):
        day=cutoff-dt.timedelta(days=offset-1); total=index['national'].get((crop,day))
        if total is not None and total>0: ratios.append(daily.get(day,{}).get(market,0.)/total)
    return sum(ratios)/len(ratios) if ratios else 0.

def _forecast_shares(index,crop,target,origin):
    return _forecast_share_detail(index,crop,target,origin)['weights']

def _forecast_share_detail(index,crop,target,origin):
    cutoff=origin-dt.timedelta(days=2); daily=index['daily'][crop]
    ids=sorted(m for m,first in index['firstSeen'][crop].items() if first<=cutoff)
    totals={cutoff-dt.timedelta(days=offset):index['national'].get((crop,cutoff-dt.timedelta(days=offset))) for offset in range(56)}
    dates=[d for d,total in totals.items() if total is not None and total>0]
    weights={}
    for m in ids:
        # Missing market rows on a nationally trading date are zero share.
        weekday=[daily.get(d,{}).get(m,0.)/totals[d] for d in dates if d.weekday()==target.weekday()]
        if len(weekday)>=4:
            s=sorted(weekday); n=len(s); weights[m]=s[n//2] if n%2 else (s[n//2-1]+s[n//2])/2
        else:
            fallback=[daily.get(d,{}).get(m,0.)/totals[d] for d in dates if cutoff-dt.timedelta(days=27)<=d<=cutoff]
            if fallback: weights[m]=sum(fallback)/len(fallback)
    signed_estimates={m:float(w) for m,w in weights.items() if math.isfinite(float(w))}
    negative_count=sum(w<0 for w in signed_estimates.values())
    nonfinite_count=sum(not math.isfinite(float(w)) for w in weights.values())
    weights={m:max(0.,float(w)) for m,w in weights.items() if math.isfinite(float(w))}
    total=sum(weights.values())
    normalized={m:w/total for m,w in sorted(weights.items())} if total>0 else {}
    return {'weights':normalized,'signedEstimates':signed_estimates,'negativeRawShareCount':negative_count,'nonfiniteRawShareCount':nonfinite_count,
            'zeroAfterClip':bool(weights) and total==0.,'noPositiveShareMass':total<=0.,'marketCount':len(ids)}

def _hac_p(loss_diff, days, bandwidth):
    """Two-sided normal DM approximation with calendar Bartlett HAC variance."""
    n=len(loss_diff)
    if n<30: return {'status':'insufficient_pairs','p':1.0,'n':n}
    mean=sum(loss_diff)/n; centered=[x-mean for x in loss_diff]
    # Calendar-day lag covariance, so missing targets retain actual spacing.
    byday=dict(zip(days,centered)); gamma0=sum(x*x for x in centered)/n
    if not math.isfinite(gamma0) or gamma0<=1e-18: return {'status':'nonpositive_variance','p':1.0,'n':n}
    long=gamma0
    for lag in range(1,bandwidth+1):
        cov=sum(byday[d]*byday.get(d-dt.timedelta(days=lag),0.) for d in days)/n
        long+=2*(1-lag/(bandwidth+1))*cov
    var=long/n
    if not math.isfinite(var) or var<=1e-18: return {'status':'nonpositive_variance','p':1.0,'n':n}
    z=abs(mean/math.sqrt(var)); p=math.erfc(z/math.sqrt(2))
    return {'status':'available','p':p,'n':n,'meanLossDifference':mean,'statistic':mean/math.sqrt(var)}

def _holm(tests, alpha=.05):
    order=sorted(range(len(tests)),key=lambda i:tests[i]['p'])
    adjusted=[1.]*len(tests); running=0.
    for rank,i in enumerate(order):
        running=max(running,min(1.,(len(tests)-rank)*tests[i]['p'])); adjusted[i]=running
    for i,t in enumerate(tests): t['holmP']=adjusted[i]; t['holmSignificantAt005']=adjusted[i]<alpha
    return tests

def _score_rows(rows):
    answer=[r for r in rows if r['actual'] is not None and math.isfinite(r['actual']) and r['actual']>0]
    common=[r for r in answer if r['directPrediction'] is not None and r['panelPrediction'] is not None and r['directPrediction']>0 and r['panelPrediction']>0]
    paired=common
    cells=[]; tests={lag:[] for lag in HAC_DAYS}
    for crop in nt.PRODUCT_CODES:
      for h in HORIZONS:
        cell=[r for r in paired if r['crop']==crop and r['horizon']==h]
        if cell:
            da=[abs(r['directPrediction']-r['actual']) for r in cell]; pa=[abs(r['panelPrediction']-r['actual']) for r in cell]
            de=[x/r['actual'] for x,r in zip(da,cell)]; pe=[abs(r['panelPrediction']-r['actual'])/r['actual'] for r in cell]
        else: da=pa=de=pe=[]
        c={'crop':crop,'horizon':h,'n':len(cell),'directMAE':sum(da)/len(da) if cell else None,'panelMAE':sum(pa)/len(pa) if cell else None,
           'directMAPE':100*sum(de)/len(de) if cell else None,'panelMAPE':100*sum(pe)/len(pe) if cell else None}
        for loss,diffs in [('AE',[abs(r['panelPrediction']-r['actual'])-abs(r['directPrediction']-r['actual']) for r in cell]),
                           ('APE',[abs(r['panelPrediction']-r['actual'])/r['actual']-abs(r['directPrediction']-r['actual'])/r['actual'] for r in cell])]:
            for lag in HAC_DAYS:
                t=_hac_p(diffs,[ _date(r['target']) for r in cell],lag); tests[lag].append({'crop':crop,'horizon':h,'loss':loss,'bandwidth':lag,**t})
        cells.append(c)
    for lag in HAC_DAYS:
        if len(tests[lag])!=FAMILY: raise AssertionError('frozen DM family size mismatch')
        _holm(tests[lag])
    test_lookup={(t['crop'],t['horizon'],t['loss'],int(lag)):t for lag,items in tests.items() for t in items}
    for cell in cells:
        crop,h=cell['crop'],cell['horizon']
        paired_cell=[r for r in paired if r['crop']==crop and r['horizon']==h]
        ae=[abs(r['panelPrediction']-r['actual'])-abs(r['directPrediction']-r['actual']) for r in paired_cell]
        ape=[(abs(r['panelPrediction']-r['actual'])-abs(r['directPrediction']-r['actual']))/r['actual'] for r in paired_cell]
        d_ae=sum(ae)/len(ae) if ae else None; d_ape=sum(ape)/len(ape) if ape else None
        required=[test_lookup[(crop,h,loss,band)] for loss in ('AE','APE') for band in HAC_DAYS]
        conclusion=_cell_conclusion(d_ae,d_ape,required,len(paired_cell))
        cell.update({'panelMinusDirectMAE':d_ae,'panelMinusDirectMAPE':100*d_ape if d_ape is not None else None,
            **conclusion,'conclusionRule':'both paired loss directions and all four Holm-adjusted HAC9/HAC28 AE/APE tests must agree; nonsignificance is not equivalence'})
    def macro(metric,crop=None):
        vals=[x[metric] for x in cells if x[metric] is not None and (crop is None or x['crop']==crop)]
        return sum(vals)/len(vals) if vals else None
    methods={}
    for method,predkey,statuskey in [('direct','directPrediction','directFallbackAction'),('panel','panelPrediction','panelFallback')]:
        emitted=[r for r in rows if r[predkey] is not None and r[predkey]>0]
        methods[method]={'plannedCount':len(rows),'issuedCount':len(emitted),'answerHeldCount':len([r for r in answer if r[predkey] is not None and r[predkey]>0]),
            'commonIssuedCount':len(common),'plannedCoverage':len(emitted)/len(rows) if rows else 0.,
            'answerCoverage':sum(r[predkey] is not None and r[predkey]>0 for r in answer)/len(answer) if answer else 0.,
            'nativeCount':sum(r[predkey] is not None and r[predkey]>0 and r[statuskey] in ('native',None) for r in rows),
            'fallbackCount':sum(r[predkey] is not None and r[predkey]>0 and r[statuskey] not in ('native',None) for r in rows),
            'fallbackCauseCounts':dict((cause,sum(r.get('directFallbackRootCause')==cause for r in rows))
                for cause in sorted({r.get('directFallbackRootCause') for r in rows if method=='direct' and r.get('directFallbackRootCause')})) if method=='direct' else
                dict((cause,sum(r.get('panelFallback')==cause for r in rows)) for cause in sorted({r.get('panelFallback') for r in rows if r.get('panelFallback')}))}
    return {'plannedCount':len(rows),'answerHeldCount':len(answer),'commonIssuedCount':len(common),
       'answerCoverageOfPlan':len(answer)/len(rows) if rows else 0.,'commonCoverageOfPlan':len(common)/len(rows) if rows else 0.,
       'methodCoverage':methods,'cells':cells,
       'directMacro40CellEqualWeightMAE':macro('directMAE'),'panelMacro40CellEqualWeightMAE':macro('panelMAE'),
       'directMacro40CellEqualWeightMAPE':macro('directMAPE'),'panelMacro40CellEqualWeightMAPE':macro('panelMAPE'),
       'perCropMacro10Horizon':{crop:{'directMAE':macro('directMAE',crop),'panelMAE':macro('panelMAE',crop),
           'directMAPE':macro('directMAPE',crop),'panelMAPE':macro('panelMAPE',crop)} for crop in nt.PRODUCT_CODES},
       'dm':{str(k):v for k,v in tests.items()},'limitations':PROTOCOL['limitations']}

def _cell_conclusion(mean_ae,mean_ape,tests,n):
    available=n>=30 and len(tests)==4 and all(t['status']=='available' for t in tests)
    improved=available and mean_ae is not None and mean_ape is not None and mean_ae<0 and mean_ape<0 and all(t['holmSignificantAt005'] and t['meanLossDifference']<0 for t in tests)
    worsened=available and mean_ae is not None and mean_ape is not None and mean_ae>0 and mean_ape>0 and all(t['holmSignificantAt005'] and t['meanLossDifference']>0 for t in tests)
    return {'conclusion':'improved' if improved else 'worsened' if worsened else 'inference_unavailable' if not available else 'inconclusive',
        'inferenceAvailable':available,'improvementIsEquivalence':False}

def _dates(start,end):
    out=[]; d=start
    while d<=end: out.append(d); d+=dt.timedelta(days=1)
    return out

def _attach_actuals(parsed,forecasts):
    scored=[]
    for r in forecasts:
        v=parsed['national'].get((r['crop'],r['target']))
        scored.append({**r,'actual':float(v['price']) if v and v['usable'] else None})
    return scored

def _score_diagnostic(rows):
    eligible=[r for r in rows if r.get('forecastEligibility')=='eligible']
    actual=[r for r in eligible if r.get('actual') is not None and r['actual']>0]
    assessment=_score_rows(rows)
    return {'status':'intermediate_diagnostic_no_selection_or_retuning','selectionAllowed':False,'retuningAllowed':False,'plannedCount':len(rows),
        'eligibleCount':len(eligible),'trainingLabelsNotMaturedCount':sum(r.get('forecastEligibility')=='training_labels_not_matured' for r in rows),
        'answerHeldEligibleCount':len(actual),'manifestRowsRetainedInDenominator':True,'assessment':assessment}

def create_manifest(archive_path=ARCHIVE, manifest_path=MANIFEST):
    data=Path(archive_path).read_bytes(); digest=sha(data)
    if digest!=nt.ARCHIVE_SHA256: raise ValueError('national archive SHA-256 differs from frozen protocol')
    parsed=parse_market_archive(data)
    dates=_dates(TEST_START,TEST_END)
    diagnostic=_forecast(parsed,_dates(DIAG_START,DIAG_END))
    forecasts=_forecast(parsed,dates)
    code=Path(__file__).read_bytes()
    doc={'schemaVersion':1,'protocol':PROTOCOL,'archiveSha256':digest,'codeSha256':sha(code),'sourceCropRowCount':parsed['sourceCropRowCount'],
         'archiveAudit':parsed['archiveAudit'],'diagnosticPlannedForecastCount':len(diagnostic),'diagnosticForecastRows':diagnostic,
         'plannedForecastCount':len(forecasts),'forecastRows':forecasts}
    dest=Path(manifest_path); dest.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(doc,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode()
    if dest.exists():
        if dest.read_bytes()!=payload: raise ValueError('existing manifest differs; immutable output refused')
    else: dest.write_bytes(payload)
    return {'path':str(dest),'sha256':sha(payload),'count':len(forecasts)}

def score_manifest(archive_path=ARCHIVE, manifest_path=MANIFEST, report_path=REPORT):
    data=Path(archive_path).read_bytes(); digest=sha(data)
    if digest!=nt.ARCHIVE_SHA256: raise ValueError('national archive SHA-256 differs from frozen protocol')
    parsed=parse_market_archive(data); doc=json.loads(Path(manifest_path).read_text())
    if doc.get('protocol')!=PROTOCOL or doc.get('archiveSha256')!=digest or doc.get('codeSha256')!=sha(Path(__file__).read_bytes()): raise ValueError('manifest protocol/input/code hash mismatch')
    diagnostic_replay=_forecast(parsed,_dates(DIAG_START,DIAG_END))
    replay=_forecast(parsed,_dates(TEST_START,TEST_END))
    if diagnostic_replay!=doc.get('diagnosticForecastRows') or replay!=doc.get('forecastRows'):
        raise ValueError('manifest forecast replay mismatch')
    # Only now connect target values, after all forecasts match immutable manifest.
    scored=_attach_actuals(parsed,replay); diagnostic_scored=_attach_actuals(parsed,diagnostic_replay)
    result=_score_rows(scored); diagnostic_result=_score_diagnostic(diagnostic_scored)
    out={'schemaVersion':1,'manifestSha256':sha(Path(manifest_path).read_bytes()),'archiveSha256':digest,
         'archiveAudit':parsed['archiveAudit'],'diagnostic':diagnostic_result,'result':result}
    p=Path(report_path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
    return result

def main(argv=None):
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='command',required=True)
    for name in ('create-manifest','score'):
        sp=sub.add_parser(name); sp.add_argument('--archive',type=Path,default=ARCHIVE); sp.add_argument('--manifest',type=Path,default=MANIFEST)
        if name=='score': sp.add_argument('--report',type=Path,default=REPORT)
    a=ap.parse_args(argv)
    result=create_manifest(a.archive,a.manifest) if a.command=='create-manifest' else score_manifest(a.archive,a.manifest,a.report)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))
if __name__=='__main__': main()
