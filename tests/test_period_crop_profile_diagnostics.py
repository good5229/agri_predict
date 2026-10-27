import datetime as dt
import math
import json
import pandas as pd
import numpy as np

from agri_data import period_crop_profile_diagnostics as d


def test_adjacent_returns_do_not_bridge_missing_days():
    s={dt.date(2024,1,1):(100.,10.),dt.date(2024,1,2):(110.,12.),dt.date(2024,1,5):(88.,9.)}
    r=d.adjacent_returns(s)
    assert len(r)==1
    assert r[0][0]==dt.date(2024,1,2)
    assert math.isclose(r[0][1],.1)


def test_period_index_preserves_calendar_gaps():
    assert d.period_index(dt.date(2024,1,1))+1 == d.period_index(dt.date(2024,1,11))
    assert d.period_index(dt.date(2024,1,11))+1 == d.period_index(dt.date(2024,1,21))
    assert d.period_index(dt.date(2024,1,21))+1 == d.period_index(dt.date(2024,2,1))


def test_summary_statistics_keep_empty_explicit():
    assert d.stats([])=={'n':0}
    x=d.stats([1,2,3,4])
    assert x['n']==4 and x['median']==2.5 and x['iqr']==1.5


def test_crop_distribution_uses_observed_pairs_only():
    s={'onion':{dt.date(2024,1,1):(10.,2.),dt.date(2024,1,2):(20.,0.),dt.date(2024,1,4):(30.,4.)}}
    x=d.crop_table(s)[0]
    assert x['observed_days']==3 and x['positive_volume_days']==2
    assert x['adjacent_day_pairs']==1 and x['day_return_abs20_count']==1


def test_cross_table_does_not_fill_calendar_gaps():
    periods=[(dt.date(2024,1,1),dt.date(2024,1,10)),(dt.date(2024,1,11),dt.date(2024,1,20)),(dt.date(2024,2,1),dt.date(2024,2,10))]
    series={'a':{},'b':{}}
    for i,(lo,hi) in enumerate(periods):
        for day in (lo,lo+dt.timedelta(days=1),lo+dt.timedelta(days=2)):
            series['a'][day]=(float(i+2),1.)
            if i<2: series['b'][day]=(float(i+4),1.)
    rows=d.cross_table(series)
    same=next(r for r in rows if r['crop_a']=='a' and r['crop_b']=='b' and r['lag_periods']==0)
    assert same['n']==1
    lead=next(r for r in rows if r['crop_a']=='a' and r['crop_b']=='b' and r['lag_periods']==1 and r['lead_direction']=='a leads b')
    assert lead['n']==0  # only two dekads, so there is no future dekad pair


def test_cross_table_keeps_lead_directions_asymmetric():
    periods=d.proxy.dekads(dt.date(2024,1,1),dt.date(2024,3,1))[:6]
    series={'a':{},'b':{}}
    avals=[10.,20.,18.,54.,27.,81.]
    bvals=[15.,12.,36.,18.,72.,54.]
    for (lo,hi),a,b in zip(periods,avals,bvals):
        for off in range(3):
            day=lo+dt.timedelta(days=off)
            series['a'][day]=(a,1.);series['b'][day]=(b,1.)
    rows=d.cross_table(series)
    ab=next(r for r in rows if r['era']=='all' and r['crop_a']=='a' and r['crop_b']=='b' and r['lag_periods']==1 and r['lead_direction']=='a leads b')
    ba=next(r for r in rows if r['era']=='all' and r['crop_a']=='b' and r['crop_b']=='a' and r['lag_periods']==1 and r['lead_direction']=='b leads a')
    assert ab['n']==4 and ba['n']==4
    assert ab['pearson'] != ba['pearson']


def test_mape8_is_equal_cell_mean_not_forecast_row_mean():
    rows=[]
    for crop,h,n,ape in [('cabbage',2,9,.1),('cabbage',3,1,.5)]:
        for i in range(n): rows.append({'crop':crop,'horizon':h,'P_ape':ape})
    for crop in ('radish','onion','garlic'):
        for h in (2,3): rows.append({'crop':crop,'horizon':h,'P_ape':.2})
    value,cells=d.mape8(rows,'P')
    assert abs(value-22.5)<1e-9
    assert d.direction_summary([1,-1,0],[1,1,0])['agreement_including_zero']==2/3


def test_response_distribution_has_quantiles_and_explicit_sign_shares_by_year():
    rows=[]
    for c in d.proxy.CROPS:
        for h in d.proxy.HORIZONS:
            for i,z in enumerate((-0.2,0.0,0.3)):
                r={'crop':c,'horizon':h,'year':2025,'actual_z':z}
                for model,offset in (('P',.1),('B0',.2),('G-B',-.1),('L-L',0.)):
                    r[model+'_z']=z+offset
                rows.append(r)
    summary=d.compression_table(rows,by_year=True,split='test')
    row=next(x for x in summary if x['crop']=='cabbage' and x['horizon']==2)
    assert row['actual_z_min']==-.2 and row['actual_z_median']==0.0 and row['actual_z_max']==.3
    assert row['actual_z_positive_share']==1/3 and row['actual_z_negative_share']==1/3 and row['actual_z_zero_share']==1/3
    assert row['G-B_z_p10'] is not None and row['G-B_z_positive_share']==1/3


def test_annual_data_support_separates_fit_origin_and_evaluation_years():
    training=[{'fit_origin':'2024-01-02','crop':'onion','training_rows':12,'unique_crop_target':8,'duplicate_rows':4},
              {'fit_origin':'2024-01-12','crop':'onion','training_rows':14,'unique_crop_target':9,'duplicate_rows':5}]
    dev=[{'crop':'onion','year':2023,'targetStart':'2023-01-01','targetEnd':'2023-01-10'}]
    test=[{'crop':'onion','year':2024,'targetStart':'2024-01-01','targetEnd':'2024-01-10'},
          {'crop':'onion','year':2024,'targetStart':'2024-01-01','targetEnd':'2024-01-10'}]
    out=d.annual_data_support(training,dev,test)
    fit=next(r for r in out if r['year']==2024)
    assert (fit['train_rows'],fit['unique_crop_target'],fit['duplicate_rows'],fit['fit_origins'])==(26,17,9,2)
    assert fit['eval_forecasts']==2 and fit['eval_unique_targets']==1
    development=next(r for r in out if r['year']==2023)
    assert development['development_forecasts']==1 and development['development_unique_targets']==1


def test_common_calendar_dekad_log_returns_require_both_common_endpoints():
    series={c:{} for c in d.proxy.CROPS}
    periods=d.proxy.dekads(dt.date(2024,1,1),dt.date(2024,1,31))[:3]
    for ci,c in enumerate(d.proxy.CROPS):
        for pi,(lo,hi) in enumerate(periods):
            for offset in range(3):
                day=lo+dt.timedelta(days=offset)
                series[c][day]=(float(100+ci*10+pi*10),2.)
    profile=d.support_profiles(series)
    assert len(profile['common_dekad_returns'])==8
    assert len(profile['annual_return_distribution'])==8 # overall and common, one year per crop
    common=next(r for r in profile['annual_return_distribution'] if r['crop']=='cabbage' and r['scope']=='four_crop_common_adjacent')
    assert common['n_adjacent_period_returns']==2 and common['log_return_sd'] is not None
    crop=next(r for r in d.crop_table(series) if r['crop']=='cabbage')
    assert crop['dekad_log_return_n']==2 and crop['dekad_log_return_p90'] is not None


def test_direction_table_conditions_on_actual_and_predicted_signs():
    rows=[]
    for c in d.proxy.CROPS:
        for h in d.proxy.HORIZONS:
            for i,z in enumerate((-.2,0.,.3)):
                r={'crop':c,'horizon':h,'year':2024,'targetStart':f'2024-01-0{i+1}','actual':100.,'actual_z':z,'actual_change':math.expm1(z)}
                for m in ('P','B0','G-B','L-L'):
                    r.update({m+'_z':z*.5,m+'_pred':99. if z<0 else 101.,m+'_ae':1.,m+'_ape':.01})
                rows.append(r)
    out=d.direction_table(rows)
    assert any(r['condition_axis']=='actual' and r['condition_direction']=='flat' and r['n_forecasts']==1 for r in out)
    assert any(r['condition_axis']=='predicted' and r['condition_direction']=='up' and r['n_forecasts']==1 for r in out)
    assert all('underprediction' in r and 'direction_agreement_including_zero' in r for r in out)


def test_feature_terciles_are_frozen_from_development_unique_origins(tmp_path):
    dev=[]
    for crop in d.proxy.CROPS:
        for i,v in enumerate((1.,2.,3.,100.)):
            dev.append({'crop':crop,'origin':f'o{i}','asOfFeatures':{'reference_age_days':v,'reference_available':1}})
        dev.append({'crop':crop,'origin':'unavailable','asOfFeatures':{'reference_age_days':0.,'reference_available':0}})
    path=tmp_path/'dev.parquet';pd.DataFrame(dev).to_parquet(path,index=False)
    test=[{'crop':'cabbage','horizon':2,'year':2024,'origin':'2023-12-30','targetStart':'2024-01-01','targetEnd':'2024-01-10','actual':10.,'features':{'reference_age_days':2.5,'reference_available':1},'P_ae':1.,'P_ape':.1,'B0_ae':1.,'B0_ape':.1,'G-B_ae':1.,'G-B_ape':.1,'L-L_ae':1.,'L-L_ape':.1},
          {'crop':'cabbage','horizon':2,'year':2024,'origin':'2024-01-09','targetStart':'2024-01-11','targetEnd':'2024-01-20','actual':11.,'features':{'reference_age_days':0.,'reference_available':0},'P_ae':1.,'P_ape':.1,'B0_ae':1.,'B0_ape':.1,'G-B_ae':1.,'G-B_ape':.1,'L-L_ae':1.,'L-L_ape':.1}]
    development=[{**test[0],'features':{'reference_age_days':1.5,'reference_available':1},'year':2022}]
    groups,cuts=d.asof_groups(test,path,development)
    assert cuts['reference_age_days']==[2.,3.]
    low=next(g for g in groups if g['split']=='test' and g['feature']=='reference_age_days' and g['group']=='low')
    missing=next(g for g in groups if g['split']=='test' and g['feature']=='reference_age_days' and g['group']=='missing_or_unavailable')
    assert low['n_forecasts']==0 and missing['n_forecasts']==1
    assert any(g['split']=='development' and g['n_forecasts']==1 for g in groups)


def test_score_reconciliation_detects_a_full_precision_cell_mismatch():
    rows=[]
    report={'metrics':{}}
    for crop in d.proxy.CROPS:
        for h in d.proxy.HORIZONS:
            rows.append({'crop':crop,'horizon':h,'targetStart':'2024-01-01','actual':10.,'P_pred':11.,'P_ae':1.,'P_ape':.1})
    report['metrics']['P']={'cells':{f'{c}:{h}':{'n':1,'mae':1.,'mape':10.} for c in d.proxy.CROPS for h in d.proxy.HORIZONS},'mae8':1.,'mape8':10.}
    assert d.validate_saved_score(rows,report,('P',))['cellsChecked']==8
    report['metrics']['P']['cells']['cabbage:2']['mape']=10.01
    with __import__('pytest').raises(ValueError,match='full precision cell mape'):
        d.validate_saved_score(rows,report,('P',))


def test_yearly_score_checks_equal_cell_aggregate_and_support():
    rows=[];cells={}
    for crop in d.proxy.CROPS:
        for h in d.proxy.HORIZONS:
            rows.append({'crop':crop,'horizon':h,'year':2024,'targetStart':'2024-01-01','actual':10.,'P_pred':11.,'P_ae':1.,'P_ape':.1})
            cells[f'{crop}:{h}']={'n':1,'mae':1.,'mape':10.}
    report={'yearly':{'2024':{'P':{'cells':cells,'mae8':1.,'mape8':10.}}}}
    assert d.validate_yearly_score(rows,report)['yearAggregatesChecked']==2
    report['yearly']['2024']['P']['mae8']=1.1
    with __import__('pytest').raises(ValueError,match='yearly equal-cell mae8'):
        d.validate_yearly_score(rows,report)


def test_common_price_and_dekad_support_and_season_year_counts():
    series={c:{} for c in d.proxy.CROPS}
    for year,price in ((2024,100.),(2025,120.)):
        for c in d.proxy.CROPS:
            for day in (dt.date(year,1,1),dt.date(year,1,2),dt.date(year,1,3)):
                series[c][day]=(price,2.)
    profile=d.support_profiles(series)
    assert profile['common_date_count']==6 and profile['common_period_count']==2
    crop=profile['crop_distribution'][0]
    assert crop['common_price_days']==6 and crop['common_valid_dekads']==2
    assert crop['common_dekad_price_n']==2 and crop['common_dekad_price_median']==110.
    seasonal=[r for r in profile['seasonal_pattern'] if r['crop']=='cabbage' and r['dekad']==1]
    assert len(seasonal)==2 and all(r['n_years_by_dekad']==2 for r in seasonal)
    annual=profile['annual_volume_support']
    assert all('price_median' in r and 'volume_median' in r for r in annual)


def test_input_hash_mismatch_blocks_preflight(tmp_path,monkeypatch):
    bad=tmp_path/'bad';bad.write_text('changed')
    monkeypatch.setattr(d,'V5',tmp_path);monkeypatch.setattr(d,'V3',tmp_path)
    monkeypatch.setattr(d,'PINS',{'forecasts.parquet':'x','score-report.json':'y','prices.json':'z'})
    with __import__('pytest').raises(ValueError,match='blocked_integrity'):
        d.verify_inputs()


def test_saved_prediction_pairing_reconstructs_780_exact_keys_and_actuals(tmp_path):
    periods=d.proxy.dekads(dt.date(2024,1,1),dt.date(2026,9,30))
    g=[];p=[];series={c:{} for c in d.proxy.CROPS}
    for crop in d.proxy.CROPS:
        for h,n in ((2,98),(3,97)):
            for i,(lo,hi) in enumerate(periods[:n]):
                origin=(lo-dt.timedelta(days=2)).isoformat()
                for day in (lo,lo+dt.timedelta(days=1),lo+dt.timedelta(days=2)):
                    series[crop][day]=(100.+i,1.)
                base={'crop':crop,'horizon':h,'targetStart':lo.isoformat(),'targetEnd':hi.isoformat(),'origin':origin,'p_ref':100.,'targetKey':lo.isoformat(),'policyUnavailable':False}
                g.append({**base,'asOfFeatures':{'reference_age_days':1},'predictions':{'G-B':{'predicted':101.+i,'status':'ok'},'L-L':{'predicted':102.+i,'status':'ok'}}})
                p.append({**base,'forecasts':{'P':{'predicted':100.+i,'referencePrice':100.},'B0':{'predicted':99.+i}}})
        early=(('2023-12-22',2,dt.date(2024,1,1),dt.date(2024,1,10)),('2023-12-12',3,dt.date(2024,1,1),dt.date(2024,1,10)),('2023-12-22',3,dt.date(2024,1,11),dt.date(2024,1,20)))
        for origin,h,lo,hi in early:
            base={'crop':crop,'horizon':h,'targetStart':lo.isoformat(),'targetEnd':hi.isoformat(),'origin':origin,'p_ref':float('nan'),'targetKey':lo.isoformat(),'policyUnavailable':True}
            g.append({**base,'asOfFeatures':{},'predictions':{'G-B':{'predicted':None,'status':'policy_unavailable'},'L-L':{'predicted':None,'status':'policy_unavailable'}}})
            p.append({**base,'forecasts':{'P':{'predicted':None,'referencePrice':float('nan')},'B0':{'predicted':None}}})
    # Match the production Parquet round-trip: null floating reference prices
    # return from pandas/Arrow as NaN, while all 12 unavailable rows stay visible.
    gp=tmp_path/'V5.parquet';pp=tmp_path/'V3.parquet'
    pd.DataFrame(g).to_parquet(gp,index=False);pd.DataFrame(p).to_parquet(pp,index=False)
    gf=pd.read_parquet(gp);pf=pd.read_parquet(pp)
    assert len(gf)==len(pf)==792 and int(gf.policyUnavailable.fillna(False).astype(bool).sum())==12
    rows=d.build_metric_rows(gf,pf,series,require_planned_grid=True)
    assert len(rows)==780
    assert len({d.period_key(r) for r in rows})==780
    assert all(r['actual']==r['P_pred'] for r in rows)
    assert all(r['G-B_pred'] is not None and r['L-L_pred'] is not None for r in rows)


def test_development_reuses_selected_v3_rw_method_from_saved_parquet(tmp_path):
    periods=d.proxy.dekads(dt.date(2022,1,1),dt.date(2023,12,31))
    methods={'cabbage':'R','radish':'R','onion':'R','garlic':'W'}
    selection={'P':{c:{'method':m} for c,m in methods.items()}}
    g=[];p=[];series={c:{} for c in d.proxy.CROPS}
    for crop in d.proxy.CROPS:
        actual_count=62 if crop=='garlic' else 72
        for h in d.proxy.HORIZONS:
            for i,(lo,hi) in enumerate(periods):
                key={'crop':crop,'horizon':h,'targetStart':lo.isoformat(),'targetEnd':hi.isoformat(),'origin':(lo-dt.timedelta(days=2)).isoformat(),'inputCutoff':(lo-dt.timedelta(days=2)).isoformat(),'targetKey':lo.isoformat()}
                if i<actual_count:
                    for day in (lo,lo+dt.timedelta(days=1),lo+dt.timedelta(days=2)):
                        series[crop][day]=(100.+i,2.)
                g.append({**key,'p_ref':100.,'asOfFeatures':{'reference_age_days':1.},'featureMetadata':{'reference_available':True},'predictions':{'G-B':{'predicted':102.+i,'status':'ok'},'L-L':{'predicted':103.+i,'status':'ok'}}})
                p.append({**key,'forecasts':{'R':{'predicted':101.+i,'referencePrice':100.,'clippedAtFloor':False,'ratioRaw':1.01,'ratioUsed':1.01},'W':{'predicted':1001.+i,'referencePrice':100.,'clippedAtFloor':False,'ratioRaw':10.01,'ratioUsed':10.01}}})
    gp=tmp_path/'v5-development.parquet';pp=tmp_path/'v3-development.parquet'
    pd.DataFrame(g).to_parquet(gp,index=False);pd.DataFrame(p).to_parquet(pp,index=False)
    rows=d.build_development_rows(pd.read_parquet(gp),pd.read_parquet(pp),series,selection)
    assert len(g)==len(p)==576 and len(rows)==556
    counts={(c,h):sum(r['crop']==c and r['horizon']==h for r in rows) for c in d.proxy.CROPS for h in d.proxy.HORIZONS}
    assert counts=={(c,h):(62 if c=='garlic' else 72) for c in d.proxy.CROPS for h in d.proxy.HORIZONS}
    assert all(r['P_pred']>1000 for r in rows if r['crop']=='garlic')
    assert all(r['P_pred']<200 for r in rows if r['crop']!='garlic')
    assert all(r['P_selected_method']==methods[r['crop']] and r['P_clippedAtFloor'] is False for r in rows)
    assert all(r['P_ratioUsed']==(10.01 if r['crop']=='garlic' else 1.01) for r in rows)
    broken=pd.read_parquet(pp);broken.at[0,'forecasts']={'W':{'predicted':1.,'referencePrice':100.}}
    with __import__('pytest').raises(ValueError,match='selected P method R missing'):
        d.build_development_rows(pd.read_parquet(gp),broken,series,selection)


def test_forecast_grid_retains_twelve_early_unavailable_rows():
    frame=pd.DataFrame({'policyUnavailable':[False]*780+[True]*12})
    result=d.verify_forecast_grids(frame,frame)
    assert result['V5_GB_LL']=={'rows':792,'policy_unavailable':12,'available':780}


def test_no_fit_or_predict_entrypoints_exist():
    assert not hasattr(d,'fit_model')
    assert not hasattr(d,'predict_model')


def test_synthetic_run_writes_all_tables_summary_and_nine_figures(tmp_path,monkeypatch):
    """Exercise run() plumbing with synthetic rows; all source/model inputs are stubbed."""
    v5=tmp_path/'V5';v3=tmp_path/'V3';v5.mkdir();v3.mkdir()
    (v5/'score-report.json').write_text(json.dumps({'inference':{'copied':'frozen'},'holmAdjustedP':{},'bootstrap':{},'metrics':{},'yearly':{}}))
    (v5/'development-score.json').write_text(json.dumps({'metrics':{'B0':{'saved':'aggregate'}}}))
    monkeypatch.setattr(d,'V5',v5);monkeypatch.setattr(d,'V3',v3)
    monkeypatch.setattr(d,'verify_inputs',lambda:{'synthetic':'pinned'})
    monkeypatch.setattr(d,'verify_forecast_grids',lambda *a:{'synthetic':True})
    monkeypatch.setattr(d,'validate_saved_score',lambda *a,**k:{'synthetic':True})
    monkeypatch.setattr(d,'validate_yearly_score',lambda *a,**k:{'synthetic':True})
    monkeypatch.setattr(d,'audit_fit_contract',lambda *a:{'ledgerEntries':488,'completeFits':488,'uniqueOrigins':171,'contractChecked':488,'weightsChecked':171,
        'originSupportRows':[{'fit_origin':'2026-01-01','crop':c,'horizon':h,'training_rows':100,'unique_crop_target':50,'duplicate_rows':50} for c in d.proxy.CROPS for h in d.proxy.HORIZONS],
        'cropOriginSupportRows':[{'fit_origin':'2026-01-01','crop':c,'training_rows':200,'unique_crop_target':120,'duplicate_rows':80} for c in d.proxy.CROPS]})
    monkeypatch.setattr(d,'load_fit_support',lambda:{'completed_fits':488,'fit_attempts':488})
    # Distinct synthetic targets across the 2024-26 test and 2022-23 development splits.
    test=[];development=[]
    periods=d.proxy.dekads(dt.date(2024,1,1),dt.date(2026,9,30))
    for ci,c in enumerate(d.proxy.CROPS):
        for h,n in ((2,98),(3,97)):
            for i,(lo,hi) in enumerate(periods[:n]):
                actual=1579.*(.7,1.,.9,1.1,1.3)[i%5];row={'crop':c,'horizon':h,'year':lo.year,'origin':(lo-dt.timedelta(days=2)).isoformat(),'targetStart':lo.isoformat(),'targetEnd':hi.isoformat(),'actual':actual,'p_ref':1579.,'actual_z':math.log(actual/1579.),'actual_change':actual/1579.-1,'features':{'reference_age_days':float(i+1),'reference_available':1,'price_sd_7':.1,'price_available_7':1,'price_sd_28':.2,'price_available_28':1,'price_sd_90':.3,'price_available_90':1,'price_rel_28':.2,'volume_ratio_10_20':.1,'volume_ratio_available':1,'volume_coverage_28':.5,'volume_available_28':1}}
                for mi,m in enumerate(('P','B0','G-B','L-L')):
                    pred=actual*(1+(.143279,.157323,.13478,.144567)[mi])
                    row[m+'_pred']=pred;row[m+'_ae']=abs(pred-actual);row[m+'_ape']=abs(pred-actual)/actual;row[m+'_z']=math.log(pred/1000.);row[m+'_status']='ok'
                test.append(row)
    devperiods=d.proxy.dekads(dt.date(2022,1,1),dt.date(2023,12,31))
    for ci,c in enumerate(d.proxy.CROPS):
        for h in (2,3):
            n=62 if c=='garlic' else 72
            for i,(lo,hi) in enumerate(devperiods[:n]):
                actual=1200.*(.7,1.,.9,1.1,1.3)[i%5];row={'crop':c,'horizon':h,'year':lo.year,'origin':(lo-dt.timedelta(days=2)).isoformat(),'targetStart':lo.isoformat(),'targetEnd':hi.isoformat(),'actual':actual,'p_ref':1200.,'actual_z':math.log(actual/1200.),'actual_change':actual/1200.-1,'features':{'reference_age_days':float(i+1),'reference_available':1,'price_sd_7':.1,'price_available_7':1,'price_sd_28':.2,'price_available_28':1,'price_sd_90':.3,'price_available_90':1,'price_rel_28':.2,'volume_ratio_10_20':.1,'volume_ratio_available':1,'volume_coverage_28':.5,'volume_available_28':1}}
                for mi,m in enumerate(('P','G-B','L-L')):
                    pred=actual*(1+(.15,.17,.16)[mi]);row[m+'_pred']=pred;row[m+'_ae']=abs(pred-actual);row[m+'_ape']=abs(pred-actual)/actual;row[m+'_z']=math.log(pred/1200.);row[m+'_status']='ok'
                development.append(row)
    assert len(test)==780 and len(development)==556
    monkeypatch.setattr(d,'build_metric_rows',lambda *a,**k:test)
    monkeypatch.setattr(d,'build_development_rows',lambda *a,**k:development)
    def fake_reconciliation(rows,models=('B0','P','G-B','L-L')):
        numbers={'B0':(221.2328,15.7323),'P':(225.4959,14.3279),'G-B':(212.8862,13.4780),'L-L':(233.7344,14.4567)}
        if len(rows)==556:numbers={'P':(225.,17.3745),'G-B':(230.,18.2400),'L-L':(240.,19.)}
        return [{'model':m,'paired_rows':len(rows),'cells':8,'MAE8':numbers[m][0],'MAPE8_pct':numbers[m][1]} for m in models]
    monkeypatch.setattr(d,'model_reconciliation',fake_reconciliation)
    series={c:{} for c in d.proxy.CROPS}
    start=dt.date(2022,1,1)
    for i in range(5*36*3):
        day=start+dt.timedelta(days=i)
        for ci,c in enumerate(d.proxy.CROPS):series[c][day]=(500.+ci*70+(i%51),float(1+i%7))
    raw={'meta':{'synthetic':True},'products':[{'id':c,'diagnostics':{'marketQueryStatusCounts':{'complete':1},'incompleteMarketDays':0}} for c in d.proxy.CROPS]}
    monkeypatch.setattr(d,'load_source',lambda:(raw,series))
    dev_frame=pd.DataFrame([{'crop':c,'origin':f'{i:04d}','asOfFeatures':{'reference_age_days':i+1,'reference_available':1,'price_sd_7':.1,'price_available_7':1,'price_sd_28':.2,'price_available_28':1,'price_sd_90':.3,'price_available_90':1,'price_rel_28':.2,'volume_ratio_10_20':.1,'volume_ratio_available':1,'volume_coverage_28':.5,'volume_available_28':1}} for c in d.proxy.CROPS for i in range(12)])
    monkeypatch.setattr(d,'read_frame',lambda p:dev_frame)
    out=tmp_path/'out';checks=d.run(out)
    assert checks['pairedRows']==780 and checks['developmentPairedRows']==556
    for name in ('analysis_summary.json','checks.json','preflight.json','source_manifest.json','00_executive_summary.md','01_crop_price_distribution.md','02_post_v5_error_diagnostics.md','03_cross_crop_and_pooling_readiness.md'):
        assert (out/name).is_file(),name
    assert len(list((out/'figures').glob('*.png')))==9
    assert {'daily_price_returns.csv','dekad_price_returns.csv','seasonal_pattern.csv','annual_volume_support.csv','dekad_support.csv'} <= {p.name for p in (out/'tables').glob('*.csv')}
    summary=json.loads((out/'analysis_summary.json').read_text())
    assert abs(sum(r['signed_contribution_to_macro_minus_10_pp'] for r in summary['q8_remaining_10pct_gap']['signed_cell_gap_contributions_pp'])-3.478)<1e-9
    assert summary['inference_reused_without_new_tests']['V5_inference']=={'copied':'frozen'}
