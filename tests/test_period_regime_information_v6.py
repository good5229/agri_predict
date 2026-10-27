import datetime as dt

import numpy as np
import pytest

from agri_data import period_regime_information_v6 as v6


def _row(crop, year, event, *, sd, available=1, extra=None):
    target = dt.date(year, 1, 1)
    features = {k: 0.25 for k in v6.FEATURES_RB}
    features.update({'price_sd_28': sd if available else 0.0, 'price_available_28': available})
    if extra:
        features.update(extra)
    return {'crop': crop, 'origin': dt.date(year, 1, 12), 'targetEnd': dt.date(year, 1, 10),
            'targetStart': target.isoformat(), 'features': features, 'event': event}


def test_event_is_strictly_price_ratio_label_and_ignores_unavailable_values():
    assert v6.label_event(120, 100) == 1
    assert v6.label_event(80, 100) == 1
    assert v6.label_event(119.999, 100) == 0
    assert v6.label_event(100, 0) is None
    assert v6.label_event(None, 100) is None


def test_rv_uses_feature_availability_mask_and_train_only_scaling():
    train = [_row('cabbage', 2021, i % 2, sd=float(i), available=1) for i in range(24)]
    train += [_row('radish', 2021, (i + 1) % 2, sd=float(i + 4), available=1) for i in range(24)]
    # An unavailable raw zero is imputed from the training mean and remains mask=0.
    test = _row('cabbage', 2022, 1, sd=0.0, available=0)
    xtrain, stats = v6._matrix(train, 'RV')
    xtest, _ = v6._matrix([test], 'RV', stats)
    sd_index = v6.FEATURES_RV.index('price_sd_28')
    mask_index = v6.FEATURES_RV.index('price_available_28')
    assert xtrain[:, sd_index].mean() == pytest.approx(0, abs=1e-12)
    assert xtest[0, sd_index] == pytest.approx(0)
    assert xtest[0, mask_index] == 0
    assert stats['price_sd_28']['mean'] == pytest.approx(np.mean([r['features']['price_sd_28'] for r in train]))


def test_future_matured_label_is_rejected():
    bad = {'targetEnd': dt.date(2022, 1, 10), 'targetStart':'2022-01-01'}
    with pytest.raises(ValueError, match='unmatured'):
        v6.assert_training_maturity([bad], dt.date(2022, 1, 11))
    v6.assert_training_maturity([bad], dt.date(2022, 1, 12))


def test_development_gate_requires_each_development_year_improve_and_support():
    scores = {}
    for year, rb, rv in ((2022,.20,.19),(2023,.20,.18)):
        for crop in v6.CROPS:
            scores[f'{year}:{crop}'] = {
                'n':36,'positive':8,'negative':28,
                'RB':{'brier':rb,'logLoss':.60},'RV':{'brier':rv,'logLoss':.61}}
    assert v6.development_gate(scores)['status'] == 'pass'
    scores['2023:cabbage']['RV']['brier'] = .21
    scores['2023:radish']['RV']['brier'] = .21
    assert v6.development_gate(scores)['status'] == 'year_inconsistent'
    scores['2023:radish']['n'] = 23
    assert v6.development_gate(scores)['status'] == 'inconclusive_support'


def test_scoring_and_calibration_ignore_rows_without_label_or_pair():
    rows = [
        {'event':1,'prob_RB':.7,'prob_RV':.6},
        {'event':0,'prob_RB':.2,'prob_RV':.3},
        {'event':None,'prob_RB':.99,'prob_RV':.99},
    ]
    score = v6.score_group(rows)
    assert score['n'] == 2 and score['positive'] == 1
    assert score['RB']['brier'] == pytest.approx(.065)
    assert sum(x['n'] for x in v6.calibration(rows)['RV']) == 2


def test_single_pooled_logistic_fit_returns_both_crop_probabilities():
    train = []
    for i in range(36):
        train.append(_row('cabbage', 2021, i % 2, sd=float(i % 8), available=1))
        train.append(_row('radish', 2021, (i // 2) % 2, sd=float(i % 6), available=1))
    tests = [_row('cabbage', 2022, 0, sd=1., available=1), _row('radish', 2022, 0, sd=2., available=1)]
    probs, info = v6.fit_probabilities(train, tests, 'RV')
    assert len(probs) == 2 and all(0 <= p <= 1 for p in probs)
    assert info['status'] == 'ok' and info['trainingByCrop'] == {'cabbage':36,'radish':36}


def test_paired_hac_respects_calendar_lags_and_emits_two_sided_p():
    rows=[]
    for i in range(40):
        # Build valid start dates on a dekad-like monthly schedule below.
        month=(i//3)+1
        year=2022+(month-1)//12
        month=(month-1)%12+1
        start=dt.date(year,month,(1,11,21)[i%3]).isoformat()
        rows.append({'targetStart':start,'event':i%2,'prob_RB':.5,'prob_RV':(.25+i*.01) if i%2 else (.75-i*.01)})
    result=v6.dm_hac(rows,'brier',3)
    assert result['status'] == 'tested' and result['n'] == 40
    assert 0 <= result['p'] <= 1


def test_catalog_schedule_and_issue_features_do_not_depend_on_future_target_y(monkeypatch):
    target=(dt.date(2022,1,1),dt.date(2022,1,10))
    schedule=[{'target':target,'origin':dt.date(2022,1,12)}]
    monkeypatch.setattr(v6.proxy,'planned_origins',lambda *args,**kwargs:schedule)
    def feature_row(series, targets, crop, horizon, period, origin):
        f={k:.3 for k in v6.FEATURES_RB}
        f.update({'price_sd_28':.12,'price_available_28':1,'price_n_28':12,'reference_available':1})
        return {'features':f,'referencePrice':100.,'referenceDate':'2022-01-09','inputCutoff':'2022-01-10'}
    monkeypatch.setattr(v6.v5,'feature_row',feature_row)
    series={'cabbage':{},'radish':{}}
    targets_a={c:{target:{'actual':123.}} for c in v6.CROPS}
    targets_b={c:{target:{'actual':None}} for c in v6.CROPS}
    a=v6.make_catalog(series,targets_a);b=v6.make_catalog(series,targets_b)
    assert len(a)==len(b)==2
    assert [(r['crop'],r['targetStart'],r['targetEnd'],r['features']) for r in a] == [(r['crop'],r['targetStart'],r['targetEnd'],r['features']) for r in b]
    # The changed future label cannot enter training before target end + 2 days.
    assert v6.training_rows(b,dt.date(2022,1,11)) == []
    assert len(v6.training_rows(a,dt.date(2022,1,12))) == 2


def test_parquet_date_and_nested_feature_values_are_normalized(tmp_path):
    import pandas as pd
    path=tmp_path/'cache.parquet'
    row={'crop':'cabbage','origin':dt.date(2022,1,12),'targetStart':'2022-01-01','targetEnd':dt.date(2022,1,10),
         'actual':None,'p_ref':100.,'features':{'price_sd_28':.1,'price_available_28':1},
         'featureMeta':{'inputCutoff':'2022-01-10'}}
    pd.DataFrame([row]).to_parquet(path,index=False)
    result=pd.read_parquet(path).to_dict('records')[0]
    assert v6.as_date(result['origin']) == dt.date(2022,1,12)
    assert v6.as_date(result['targetEnd']) == dt.date(2022,1,10)
    assert v6.optional_float(result['actual']) is None
    assert result['features']['price_sd_28'] == pytest.approx(.1)


def test_year_scoped_block_interval_supports_brier_and_logloss():
    rows=[]
    for c in v6.CROPS:
        for i in range(36):
            month=i//3+1;day=(1,11,21)[i%3]
            import calendar
            end=(10,20,calendar.monthrange(2022,month)[1])[i%3]
            start=dt.date(2022,month,day).isoformat();event=i%2
            rows.append({'crop':c,'targetStart':start,'event':event,'prob_RB':.5,
                         'prob_RV':(.2+.005*i) if event else (.8-.005*i)})
    for metric in ('brier','logLoss'):
        result=v6.paired_block_bootstrap(rows,block=6,metric=metric,reps=100)
        assert result['status']=='exploratory_percentile_interval'
        assert result['estimand']=='equal-crop mean RV-RB paired loss difference'
        assert len(result['ci95'])==2


def test_development_stages_bind_label_free_forecast_score_and_frozen_gate(tmp_path, monkeypatch):
    out=tmp_path/'v6';out.mkdir()
    (out/'protocol.md').write_text('frozen protocol fixture')
    fixed_identity={'protocolSha256':'protocol-fixture','coreCodeSha256':'core-fixture','sourceSha256':'source-fixture',
                    'v5CodeSha256':'v5-fixture','proxyCodeSha256':'proxy-fixture'}
    monkeypatch.setattr(v6,'run_identity',lambda _out:fixed_identity)
    cache=out/'v5-feature-catalog.parquet';cache.write_bytes(b'fixture-cache')
    cache_meta={'sha256':v6.sha256(cache)}
    (out/'v5-feature-catalog.json').write_text(__import__('json').dumps(cache_meta))
    import json
    source_file=out/'source-fixture.txt';source_file.write_text('source fixture')
    (out/'source-manifest.json').write_text(json.dumps({f'fixture-{i}':{'path':str(source_file),'sha256':v6.sha256(source_file)} for i in range(10)}))
    (out/'input-and-feature-audit.json').write_text('{}')
    identity={**fixed_identity,'catalogSha256':cache_meta['sha256'],
              'sourceManifestSha256':v6.sha256(out/'source-manifest.json'),
              'featureAuditSha256':v6.sha256(out/'input-and-feature-audit.json')}
    (out/'preflight.json').write_text(json.dumps({'status':'pass','runIdentity':identity}))

    schedules=[];catalog=[];targets={c:{} for c in v6.CROPS}
    # 48 matured, balanced labels per crop satisfy the frozen training support.
    for c in v6.CROPS:
        for i in range(48):
            start=dt.date(2018,1,1)+dt.timedelta(days=10*i);end=start+dt.timedelta(days=9)
            feat={name:.2 for name in v6.FEATURES_RB};feat.update({'price_sd_28':.1,'price_available_28':1,'price_n_28':28,'reference_available':1})
            catalog.append({'crop':c,'horizon':3,'targetStart':start.isoformat(),'targetEnd':end,'origin':end+dt.timedelta(days=2),
                            'actual':120. if i%2 else 100.,'p_ref':100.,'features':feat,
                            'featureMeta':{'inputCutoff':'2018-01-01'}})
        for year in (2022,2023):
            for slot in range(36):
                month=slot//3+1;day=(1,11,21)[slot%3]
                import calendar
                target=(dt.date(year,month,day),dt.date(year,month,(10,20,calendar.monthrange(year,month)[1])[slot%3]))
                origin=target[0]-dt.timedelta(days=10)
                if c == 'cabbage': schedules.append({'origin':origin,'target':target})
                event=(slot+year)%2
                feat={name:.2 for name in v6.FEATURES_RB};feat.update({'price_sd_28':.1,'price_available_28':1,'price_n_28':28,'reference_available':1})
                unavailable=(c=='cabbage' and year==2022 and slot==0)
                catalog.append({'crop':c,'horizon':3,'targetStart':target[0].isoformat(),'targetEnd':target[1],
                                'origin':origin,'actual':120. if event else 100.,'p_ref':None if unavailable else 100.,'features':None if unavailable else feat,
                                'featureMeta':{'referencePrice':100.,'referenceDate':'2021-12-30','inputCutoff':(origin-dt.timedelta(days=2)).isoformat()}})
                targets[c][target]={'actual':120. if event else 100.}
    # Only available historical labels are training data; eval labels never enter the training rows.
    monkeypatch.setattr(v6,'_load_catalog',lambda _out:(catalog,targets,cache_meta))
    monkeypatch.setattr(v6,'build_schedule',lambda split:schedules if split=='develop' else [])
    fit_calls=[]
    def synthetic_fit(train,tests,method):
        fit_calls.append((tests[0]['origin'],method,len(train)))
        probs=[]
        for test in tests:
            d=dt.date.fromisoformat(test['targetStart'])
            slot=(0 if d.day==1 else 1 if d.day==11 else 2)
            event=((d.month-1)*3+slot+d.year)%2
            probs.append((.9 if event else .1) if method=='RV' else .5)
        return probs,{'status':'ok','trainingRows':len(train),'trainingByCrop':{'cabbage':48,'radish':48},'trainingPositive':48,'trainingNegative':48}
    monkeypatch.setattr(v6,'fit_probabilities',synthetic_fit)
    monkeypatch.setattr(v6,'paired_block_bootstrap',lambda rows,**kw:{'status':'synthetic','metric':kw['metric']})

    result=v6.run_forecast('develop',out)
    assert result['rows']==144 and result['attemptsThisStage']==144
    assert len(fit_calls)==144 and all(n>=96 for _,_,n in fit_calls)
    forecast_path=out/'development-risk-forecasts.parquet'
    forecast_rows=__import__('pandas').read_parquet(forecast_path).to_dict('records')
    assert len(forecast_rows)==144 and not any('actual' in r or 'event' in r for r in forecast_rows)
    score=v6.run_score('develop',out)
    assert score['scoredRows']==143
    assert set(score['blockBootstrap'])=={'2022','2023','all'}
    assert set(score['blockBootstrap']['2022']['6'])=={'brier','logLoss'}
    gate=v6.run_gate(out)
    assert gate['status']=='pass' and gate['testFitsAllowed'] is True
    frozen=(out/'development-gate.json').read_bytes()
    assert v6.run_gate(out)==gate and (out/'development-gate.json').read_bytes()==frozen
    (out/'development-gate.json').write_text(json.dumps({**gate,'status':'risk_signal_inconsistent','testFitsAllowed':False}))
    with pytest.raises(ValueError,match='development gate'):
        v6.run_forecast('test',out)
    assert not (out/'test-risk-forecasts.parquet').exists() and len(fit_calls)==144
    # Editing the forecast breaks its ledger hash; scoring cannot silently consume it.
    (out/'development-risk-forecasts.parquet').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint mismatch'):
        v6.run_score('develop',out)
