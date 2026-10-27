import csv
import datetime as dt
import io
import math
import zipfile
from decimal import Decimal

import pytest

from agri_data import national_panel_proposal as np
from agri_data.national_transfer import EXPECTED_HEADERS, PERIODS


def _row(day, market='01', qty='10', amount='1000', code='1001', name='배추'):
    return {'경매일자':day.strftime('%Y%m%d'),'도매시장코드':market,'도매시장명':'시장'+market,
            '품목코드':code,'품목명':name,'변경총물량':qty,'경매낙찰총금액':amount}

def _zip(period_rows):
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w') as z:
        for i,rows in enumerate(period_rows):
            f=io.StringIO(newline=''); w=csv.DictWriter(f,fieldnames=EXPECTED_HEADERS,lineterminator='\n'); w.writeheader(); w.writerows(rows)
            z.writestr(f'p{i}.csv',f.getvalue().encode('cp949'))
    return out.getvalue()

def test_market_parser_preserves_signed_null_duplicate_and_name_conflicts():
    day=dt.date(2024,7,8)
    groups=[[_row(day),_row(day,market='02',qty='-2',amount='-100'),_row(day),
             _row(day,market='03',qty='',amount='5'),_row(day,market='04',name='잘못된 이름')]]
    groups += [[_row(start)] for start,_ in PERIODS[1:]]
    parsed=np.parse_market_archive(_zip(groups),expected_sha256=None)
    n=parsed['national'][('cabbage',day.isoformat())]
    assert Decimal(n['amount'])==Decimal('2905')
    assert Decimal(n['quantity'])==Decimal('28')
    assert n['negativeQuantityCount']==1 and n['negativeAmountCount']==1
    assert n['nullQuantityCount']==1 and n['duplicateRowCount']==1
    assert n['rowCount']==5 and parsed['archiveAudit']['sourceRowCount']>=parsed['sourceCropRowCount']
    assert n['nameConflictCount']==1 and not n['usable']
    assert parsed['archiveAudit']['nullQuantityRows']==1
    assert parsed['archiveAudit']['negativeQuantityRows']==1
    assert parsed['archiveAudit']['exactDuplicateSelectedRows']==1
    assert parsed['market'][('cabbage','02',day.isoformat())]['quantity']=='-2'

def test_calendar_hac_uses_calendar_gaps_and_minimum_pair_rule():
    days=[dt.date(2026,1,1)+dt.timedelta(days=2*i) for i in range(30)]
    result=np._hac_p([(-1.)**i for i in range(30)],days,9)
    assert result['n']==30 and result['status'] in {'available','nonpositive_variance'}
    assert np._hac_p([1.]*29,days[:29],9)['status']=='insufficient_pairs'

def test_holm_is_fixed_family_monotone_and_keeps_unavailable_p_one():
    tests=[{'p':.001},{'p':.02},{'p':1.0}]
    np._holm(tests)
    assert tests[0]['holmP'] <= tests[1]['holmP'] <= tests[2]['holmP']==1.0
    assert tests[2]['holmSignificantAt005'] is False

def test_market_shares_are_normalized_and_use_pre_cutoff_rows():
    parsed={'market':{},'national':{}}
    target=dt.date(2025,1,20)
    cutoff=target-dt.timedelta(days=2)
    for i in range(1,20):
        d=cutoff-dt.timedelta(days=i)
        parsed['market'][('cabbage','01',d.isoformat())]={'usable':True,'quantity':'10','quantityUsable':True}
        parsed['market'][('cabbage','02',d.isoformat())]={'usable':True,'quantity':'30','quantityUsable':True}
        parsed['national'][('cabbage',d.isoformat())]={'quantity':'40','quantityUsable':True}
    # Future row must not enter the estimate.
    parsed['market'][('cabbage','03',(cutoff+dt.timedelta(days=1)).isoformat())]={'usable':True,'quantity':'999','quantityUsable':True}
    weights=np._forecast_shares(np._quantity_index(parsed),'cabbage',target,target-dt.timedelta(days=1))
    assert set(weights)=={'01','02'}
    assert sum(weights.values())==pytest.approx(1.0)
    assert weights['02']>weights['01']

def test_market_projection_is_anchored_to_national_reference_price():
    assert np._anchored_market_price(1000.,0.)==1000.
    assert np._anchored_market_price(1000.,0.1)==pytest.approx(1000.*math.exp(.1))

def test_panel_features_preserve_signed_volume_share_and_missing_change_flags():
    origin=dt.date(2025,1,20)
    f=[0.,0.,0.,0.,0.,1.,1.]
    core=np._panel_core((f,100.),(f,100.),0,-5.,-0.25,origin)
    assert core[2]==pytest.approx(math.asinh(-5.))
    assert core[3]==-.25
    assert core[-8:-4]==[1.,1.,1.,1.]

def test_signed_share_estimate_is_reported_before_final_clipping():
    target=dt.date(2025,2,2); origin=target-dt.timedelta(days=1); cutoff=origin-dt.timedelta(days=2)
    parsed={'market':{},'national':{}}
    for k in range(4):
        day=target-dt.timedelta(days=7*(k+1))
        parsed['market'][('cabbage','A',day.isoformat())]={'quantity':'20','quantityUsable':True}
        parsed['market'][('cabbage','B',day.isoformat())]={'quantity':'-10','quantityUsable':True}
        parsed['national'][('cabbage',day.isoformat())]={'quantity':'10','quantityUsable':True}
    detail=np._forecast_share_detail(np._quantity_index(parsed),'cabbage',target,origin)
    assert detail['signedEstimates']['B']==pytest.approx(-1.)
    assert detail['negativeRawShareCount']==1
    assert detail['weights']['B']==0.
    assert detail['weights']['A']==1.

def test_training_features_obey_each_origin_two_day_cutoff():
    origin=dt.date(2025,1,20)
    series={origin-dt.timedelta(days=3):100.,origin-dt.timedelta(days=1):10000.}
    features,last=np._features(series,origin,'cabbage',1)
    assert last==100.
    later=series.copy(); later[origin-dt.timedelta(days=3)]=200.
    assert np._features(later,origin,'cabbage',1)[1]==200.
    # A separately shifted training origin sees only its own allowable past.
    assert np._features(series,origin+dt.timedelta(days=1),'cabbage',1)[1]==10000.

def test_share_estimation_uses_target_weekday_and_zeroes_nontrading_markets():
    target=dt.date(2025,2,2)  # Sunday
    cutoff=target-dt.timedelta(days=2)
    parsed={'market':{},'national':{}}
    for k in range(4):
        day=target-dt.timedelta(days=7*(k+1))
        parsed['market'][('cabbage','A',day.isoformat())]={'quantity':'100','quantityUsable':True,'usable':True}
        parsed['national'][('cabbage',day.isoformat())]={'quantity':'100','quantityUsable':True}
    # B trades once on a Sunday and once on another weekday. Its three absent
    # Sunday rows are zero shares; median Sunday share is therefore zero.
    bday=cutoff-dt.timedelta(days=14)
    parsed['market'][('cabbage','B',bday.isoformat())]={'quantity':'100','quantityUsable':True,'usable':True}
    parsed['national'][('cabbage',bday.isoformat())]={'quantity':'200','quantityUsable':True}
    parsed['market'][('cabbage','B',(cutoff-dt.timedelta(days=1)).isoformat())]={'quantity':'100','quantityUsable':True,'usable':True}
    weights=np._forecast_shares(np._quantity_index(parsed),'cabbage',target,target-dt.timedelta(days=1))
    assert weights['B']==0.
    assert weights['A']==pytest.approx(1.)

def test_score_reports_both_method_macros_and_coverage():
    rows=[]
    for crop in np.nt.PRODUCT_CODES:
        for h in np.HORIZONS:
            for i in range(30):
                day=dt.date(2026,1,1)+dt.timedelta(days=i)
                rows.append({'crop':crop,'horizon':h,'target':day.isoformat(),'actual':100.,
                    'directPrediction':101.,'panelPrediction':99.,'directFallbackAction':None,
                    'directFallbackRootCause':None,'panelFallback':None})
    report=np._score_rows(rows)
    assert report['plannedCount']==1200 and report['answerHeldCount']==1200 and report['commonIssuedCount']==1200
    assert report['directMacro40CellEqualWeightMAE']==pytest.approx(1.)
    assert report['panelMacro40CellEqualWeightMAE']==pytest.approx(1.)
    assert report['perCropMacro10Horizon']['cabbage']['directMAPE']==pytest.approx(1.)
    assert report['methodCoverage']['direct']['nativeCount']==1200
    assert report['methodCoverage']['panel']['nativeCount']==1200

def test_diagnostic_runs_separate_dm_families_and_retains_ineligible_plan_rows():
    rows=[]
    for crop in np.nt.PRODUCT_CODES:
        for h in np.HORIZONS:
            for i in range(30):
                day=dt.date(2025,10,15)+dt.timedelta(days=i)
                rows.append({'crop':crop,'horizon':h,'target':day.isoformat(),'actual':100.,
                    'directPrediction':110.+i%3,'panelPrediction':100.+i%2,
                    'directFallbackAction':None,'directFallbackRootCause':None,'panelFallback':None,
                    'forecastEligibility':'eligible'})
    rows.append({'crop':'cabbage','horizon':10,'target':'2025-10-01','actual':100.,
        'directPrediction':None,'panelPrediction':None,'forecastEligibility':'training_labels_not_matured'})
    report=np._score_diagnostic(rows)
    assert report['status']=='intermediate_diagnostic_no_selection_or_retuning'
    assert report['selectionAllowed'] is False and report['retuningAllowed'] is False
    assert report['plannedCount']==1201 and report['eligibleCount']==1200
    assert report['trainingLabelsNotMaturedCount']==1
    assessment=report['assessment']
    assert assessment['plannedCount']==1201
    assert assessment['commonCoverageOfPlan']==pytest.approx(1200/1201)
    assert len(assessment['dm']['9'])==80 and len(assessment['dm']['28'])==80
    assert len(assessment['cells'])==40
    assert all('conclusion' in cell for cell in assessment['cells'])

def test_cell_conclusion_requires_both_losses_and_both_bandwidths():
    tests=[{'status':'available','meanLossDifference':-1.,'holmSignificantAt005':True} for _ in range(4)]
    assert np._cell_conclusion(-1.,-.01,tests,30)['conclusion']=='improved'
    assert np._cell_conclusion(1.,.01,[{**t,'meanLossDifference':1.} for t in tests],30)['conclusion']=='worsened'
    assert np._cell_conclusion(-1.,-.01,[{**tests[0],'holmSignificantAt005':False},*tests[1:]],30)['conclusion']=='inconclusive'
    assert np._cell_conclusion(-1.,-.01,[{**tests[0],'status':'nonpositive_variance'},*tests[1:]],30)['conclusion']=='inference_unavailable'

def test_diagnostic_early_origins_stay_in_plan_but_are_ineligible(monkeypatch):
    parsed={'national':{},'market':{}}
    for i in range(130):
        d=dt.date(2024,1,1)+dt.timedelta(days=i)
        parsed['national'][('cabbage',d.isoformat())]={'usable':True,'price':'100','quantity':'10','quantityUsable':True}
        parsed['market'][('cabbage','A',d.isoformat())]={'usable':True,'price':'100','quantity':'10','quantityUsable':True}
    parsed['national'][('cabbage','2025-09-30')]={'usable':True,'price':'100','quantity':'10','quantityUsable':True}
    class FakeModel:
        def predict(self, X): return [0.]*len(X)
    monkeypatch.setattr(np,'_model',lambda x,y:FakeModel())
    rows=np._forecast(parsed,[dt.date(2025,10,10),dt.date(2025,10,12)])
    early=next(r for r in rows if r['crop']=='cabbage' and r['horizon']==10 and r['target']=='2025-10-10')
    mature=next(r for r in rows if r['crop']=='cabbage' and r['horizon']==10 and r['target']=='2025-10-12')
    assert early['forecastEligibility']=='training_labels_not_matured'
    assert early['directPrediction'] is None
    assert mature['forecastEligibility']=='eligible'

def test_direct_fallback_keeps_cause_separate_from_action(monkeypatch):
    parsed={'national':{},'market':{}}
    for d,price in [(dt.date(2025,9,30),123),(dt.date(2025,10,10),125)]:
        parsed['national'][('cabbage',d.isoformat())]={'usable':True,'price':str(price),'quantity':'10','quantityUsable':True}
    rows=np._forecast(parsed,[dt.date(2025,10,20)])
    row=next(r for r in rows if r['crop']=='cabbage' and r['horizon']==1)
    assert row['directPrediction']==125.
    assert row['directFallbackAction']=='last_national'
    assert row['directFallbackRootCause']=='insufficient_training_targets'

def test_model_fits_are_cached_once_per_crop_and_horizon(monkeypatch):
    start=dt.date(2024,1,1); parsed={'national':{},'market':{}}
    for crop,code in np.nt.PRODUCT_CODES.items():
        for i in range(130):
            d=start+dt.timedelta(days=i); ds=d.isoformat()
            price=100.+i%11
            parsed['national'][(crop,ds)]={'usable':True,'price':str(price),'quantity':str(10)}
            parsed['market'][(crop,'M',ds)]={'usable':True,'quantityUsable':True,'price':str(price),'quantity':str(10)}
    class FakeModel:
        def predict(self, x): return [0.]*len(x)
    calls=[]
    monkeypatch.setattr(np,'_model',lambda x,y:(calls.append((len(x),len(y))) or FakeModel()))
    result=np._forecast(parsed,[dt.date(2026,1,1)])
    assert len(result)==40
    assert len(calls)==80  # one direct and one panel fit for each crop × horizon

def test_h10_share_and_prediction_ignore_everything_after_origin_cutoff(monkeypatch):
    # Twenty 2024 training weeks are enough for the synthetic model cache; the
    # fake model turns each market's own recent ratio into its prediction.
    start=dt.date(2024,1,1); target=dt.date(2026,1,20); origin=target-dt.timedelta(days=10)
    parsed={'national':{},'market':{}}
    def add_national(day,price,quantity):
        parsed['national'][('cabbage',day.isoformat())]={'usable':True,'price':str(price),'quantity':str(quantity),'quantityUsable':True}
    def add_market(market,day,price,quantity):
        parsed['market'][('cabbage',market,day.isoformat())]={'usable':True,'price':str(price),'quantity':str(quantity),'quantityUsable':True}
    for i in range(130):
        day=start+dt.timedelta(days=i); add_national(day,1500,20)
        add_market('A',day,1000,10); add_market('B',day,2000,10)
    cutoff=origin-dt.timedelta(days=2)
    day=start+dt.timedelta(days=130)
    while day<=cutoff:
        add_national(day,1500,10); add_market('A',day,1000,10)
        day+=dt.timedelta(days=1)
    # Four preceding target weekdays establish A's stable seasonal share.
    for k in range(1,5):
        d=target-dt.timedelta(days=7*k); add_national(d,1500,10); add_market('A',d,1000,10)
    class RatioModel:
        def predict(self, X): return [x[0] for x in X]
    monkeypatch.setattr(np,'_model',lambda x,y:RatioModel())
    base=np._forecast(parsed,[target])
    contaminated={'national':dict(parsed['national']),'market':dict(parsed['market'])}
    future=origin-dt.timedelta(days=1)  # one day after the allowed feature cutoff
    contaminated['national'][('cabbage',future.isoformat())]={'usable':True,'price':'5000','quantity':'200','quantityUsable':True}
    contaminated['market'][('cabbage','B',future.isoformat())]={'usable':True,'price':'3000','quantity':'200','quantityUsable':True}
    changed=np._forecast(contaminated,[target])
    base_row=next(r for r in base if r['crop']=='cabbage' and r['horizon']==10)
    changed_row=next(r for r in changed if r['crop']=='cabbage' and r['horizon']==10)
    assert base_row['marketWeights']==changed_row['marketWeights']
    assert base_row['directPrediction']==changed_row['directPrediction']
    assert base_row['panelPrediction']==pytest.approx(changed_row['panelPrediction'])
