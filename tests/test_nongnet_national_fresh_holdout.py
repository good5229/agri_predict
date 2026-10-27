import datetime as dt
import json
from pathlib import Path

import pytest

from agri_data import nongnet_national_fresh_holdout as holdout


def _body(rows):
    return json.dumps({'datalist':rows},ensure_ascii=False).encode()

def _row(day,price='100',weight='12'):
    return {'saledate':day.strftime('%Y%m%d'),'selectDay':price,'selectDayWgt':weight}

def _attach_raw_provenance(rows,tmp_path,prefix='raw'):
    requests={}
    for i,row in enumerate(rows):
        name=row['category']; day=dt.date.fromisoformat(row['date'])
        spec=next(x for x in holdout.CATEGORIES if x[0]==name)
        request={'kind':'anchor','period':row['period'],'requestedDate':day.isoformat(),'category':name,
            'requestParams':holdout._params(day,spec[1],spec[2])}
        raw=_body([row['sourceValues']]); raw_name=f'{prefix}-{i}.json'; (tmp_path/raw_name).write_bytes(raw)
        rid=f'{prefix}-{i}'; digest=holdout._sha(raw)
        manifest_name=f'{rid}.manifest.json'
        manifest={'endpoint':holdout.ENDPOINT,'requestId':rid,'request':request,'rawFile':raw_name,'sha256':digest,
            'httpStatus':200,'truncated':False,'transportError':None}
        (tmp_path/manifest_name).write_text(json.dumps(manifest))
        attempt={'request':request,'httpStatus':200,'rawFile':raw_name,'sha256':digest,'truncated':False,
            'transportError':None,'manifestFile':manifest_name}
        requests[rid]={'request':request,'attempts':[attempt]}
        row['provenance']=[{'requestId':rid,'responseSha256':digest}]
    return {'requests':requests}

def test_anchor_matrix_has_weekly_spacing_five_holdout_categories_and_four_overlap_categories():
    matrix=holdout.request_matrix()
    assert matrix
    assert all(r['requestParams']['searchUnitGrade']=='normal' and r['requestParams']['searchSymbol1']=='100000' for r in matrix)
    overlap=[r for r in matrix if r['period']=='overlap']
    fresh=[r for r in matrix if r['period']=='holdout']
    assert {r['category'] for r in overlap}=={'cabbage','radish','onion','pepper'}
    assert {r['category'] for r in fresh}=={'cabbage','radish','onion','pepper','garlic_peeled'}
    garlic=next(r for r in fresh if r['category']=='garlic_peeled')
    assert garlic['requestParams']['searchSymbol2']=='1209' and garlic['requestParams']['searchSymbol3']=='120906'
    for period in ('overlap','holdout'):
        dates=sorted({dt.date.fromisoformat(r['requestedDate']) for r in matrix if r['period']==period})
        assert dates[0] == (holdout.OVERLAP_START if period=='overlap' else holdout.HOLDOUT_START)
        assert dates[-1] == (holdout.OVERLAP_END if period=='overlap' else holdout.HOLDOUT_END)
        assert all((b-a).days<=holdout.ANCHOR_SPACING_DAYS for a,b in zip(dates,dates[1:]))

def test_response_validator_separates_valid_empty_missing_and_api_errors():
    day=dt.date(2026,7,1)
    assert holdout.validate_response(_body([_row(day)]),day.isoformat())['requestedDateState']=='observed'
    missing=holdout.validate_response(_body([_row(day-dt.timedelta(days=1))]),day.isoformat())
    assert missing['state']=='valid' and missing['requestedDateState']=='requested_date_missing'
    assert holdout.validate_response(_body([]),day.isoformat())['state']=='normal_empty'
    assert holdout.validate_response(b'{"error":"denied"}',day.isoformat())['state']=='error_payload'
    assert holdout.validate_response(b'',day.isoformat(),503)['state']=='http_error'

def test_coverage_never_calls_normal_empty_confirmed_no_trade():
    day1=holdout.OVERLAP_START.isoformat(); day2=holdout.HOLDOUT_START.isoformat()
    found=[]
    checkpoint={'requests':{}}
    for period,day,category,state in [('overlap',day1,'cabbage','normal_empty'),
                                      ('holdout',day2,'cabbage','requested_date_missing')]:
        for pass_no in range(1,holdout.TARGETED_MISSING_PASSES+1):
            req={'kind':f'targeted_missing_pass{pass_no}','requestedDate':day,'category':category}
            checkpoint['requests'][f'{period}{pass_no}']={'request':req,'state':'valid','requestedDateState':state}
    old_periods=holdout.PERIODS; old_categories=holdout.CATEGORIES
    try:
        holdout.PERIODS=(('overlap',holdout.OVERLAP_START,holdout.OVERLAP_START),
                         ('holdout',holdout.HOLDOUT_START,holdout.HOLDOUT_START))
        holdout.CATEGORIES=(('cabbage','1001',''),)
        rows,unresolved=holdout._coverage(found,checkpoint)
    finally:
        holdout.PERIODS=old_periods; holdout.CATEGORIES=old_categories
    assert [r['state'] for r in rows]==['normal_empty','no_trade_unconfirmed']
    assert len(unresolved)==0

def test_overlap_source_match_reports_integer_tolerance_and_distribution(tmp_path):
    coverage=[]
    for name,_,_ in holdout.CATEGORIES[:4]:
        for day in holdout._daterange(holdout.OVERLAP_START,holdout.OVERLAP_END):
            coverage.append({'period':'overlap','category':name,'date':day.isoformat(),'state':'observed',
                'sourceValues':{'selectDay':'100','selectDayWgt':'20'}})
    result=holdout._source_match_audit(coverage,tmp_path/'missing.zip')
    assert result['status']=='archive_unavailable'
    assert result['comparedCount']==0

def test_overlap_source_match_computes_exact_within_one_and_relative_differences(tmp_path,monkeypatch):
    archive_path=tmp_path/'synthetic.zip'; archive_path.write_bytes(b'synthetic-only')
    monkeypatch.setattr(holdout.nt,'ARCHIVE_SHA256',holdout._sha(archive_path.read_bytes()))
    observations={crop:{day.isoformat():{'price':'99.5'} for day in holdout._daterange(holdout.OVERLAP_START,holdout.OVERLAP_END)}
        for crop in ('cabbage','radish','onion','pepper')}
    monkeypatch.setattr(holdout.nt,'parse_archive',lambda data:{'observations':observations})
    coverage=[]
    for category,_,_ in holdout.CATEGORIES[:4]:
        for day in holdout._daterange(holdout.OVERLAP_START,holdout.OVERLAP_END):
            coverage.append({'period':'overlap','category':category,'date':day.isoformat(),'state':'observed',
                'sourceValues':_row(day,'100','20')} )
    checkpoint=_attach_raw_provenance(coverage,tmp_path,'source-match')
    result=holdout._source_match_audit(coverage,archive_path,checkpoint=checkpoint,raw_dir=tmp_path)
    assert result['status']=='source_match_gate_pass'
    assert result['comparedCount']==44 and result['exactCount']==0
    assert result['byCrop']['cabbage']['comparableDays']==11
    assert float(result['byCrop']['cabbage']['medianAbsoluteRelativeDifferencePct'])==pytest.approx(0.5/99.5*100)

def test_source_match_gate_allows_nontrade_dates_but_requires_five_comparable_days(tmp_path,monkeypatch):
    archive_path=tmp_path/'synthetic.zip'; archive_path.write_bytes(b'synthetic-only')
    monkeypatch.setattr(holdout.nt,'ARCHIVE_SHA256',holdout._sha(archive_path.read_bytes()))
    days=list(holdout._daterange(holdout.OVERLAP_START,holdout.OVERLAP_END))
    observations={crop:{day.isoformat():{'price':'100'} for day in days[:5]} for crop in ('cabbage','radish','onion','pepper')}
    monkeypatch.setattr(holdout.nt,'parse_archive',lambda data:{'observations':observations})
    coverage=[{'period':'overlap','category':name,'date':day.isoformat(),'state':'observed','sourceValues':_row(day,'100','20')}
        for name,_,_ in holdout.CATEGORIES[:4] for day in days[:5]]
    checkpoint=_attach_raw_provenance(coverage,tmp_path,'nontrade')
    assert holdout._source_match_audit(coverage,archive_path,checkpoint=checkpoint,raw_dir=tmp_path)['status']=='source_match_gate_pass'
    coverage=[r for r in coverage if not (r['category']=='cabbage' and r['date']==days[4].isoformat())]
    result=holdout._source_match_audit(coverage,archive_path,checkpoint=checkpoint,raw_dir=tmp_path)
    assert result['status']=='source_match_gate_fail'
    assert result['byCrop']['cabbage']['comparableDays']==4

def test_source_match_gate_rejects_unverified_coverage_json(tmp_path,monkeypatch):
    archive_path=tmp_path/'synthetic.zip'; archive_path.write_bytes(b'synthetic-only')
    monkeypatch.setattr(holdout.nt,'ARCHIVE_SHA256',holdout._sha(archive_path.read_bytes()))
    days=list(holdout._daterange(holdout.OVERLAP_START,holdout.OVERLAP_END))[:5]
    observations={crop:{day.isoformat():{'price':'100'} for day in days} for crop in ('cabbage','radish','onion','pepper')}
    monkeypatch.setattr(holdout.nt,'parse_archive',lambda data:{'observations':observations})
    coverage=[{'period':'overlap','category':name,'date':day.isoformat(),'state':'observed',
        'sourceValues':_row(day,'100','20'),'provenance':[]} for name,_,_ in holdout.CATEGORIES[:4] for day in days]
    result=holdout._source_match_audit(coverage,archive_path,checkpoint={'requests':{}},raw_dir=tmp_path)
    assert result['status']=='source_match_gate_fail'
    assert result['comparedCount']==0
    assert all(not pair['rawEvidenceVerified'] for pair in result['pairs'])

def test_source_match_gate_rejects_attempt_manifest_mismatch(tmp_path,monkeypatch):
    archive_path=tmp_path/'synthetic.zip'; archive_path.write_bytes(b'synthetic-only')
    monkeypatch.setattr(holdout.nt,'ARCHIVE_SHA256',holdout._sha(archive_path.read_bytes()))
    days=list(holdout._daterange(holdout.OVERLAP_START,holdout.OVERLAP_END))[:5]
    observations={crop:{day.isoformat():{'price':'100'} for day in days} for crop in ('cabbage','radish','onion','pepper')}
    monkeypatch.setattr(holdout.nt,'parse_archive',lambda data:{'observations':observations})
    coverage=[{'period':'overlap','category':name,'date':day.isoformat(),'state':'observed',
        'sourceValues':_row(day,'100','20'),'provenance':[]} for name,_,_ in holdout.CATEGORIES[:4] for day in days]
    checkpoint=_attach_raw_provenance(coverage,tmp_path,'manifest-check')
    row=coverage[0]; ref=row['provenance'][0]
    manifest_path=tmp_path/checkpoint['requests'][ref['requestId']]['attempts'][0]['manifestFile']
    manifest=json.loads((tmp_path/manifest_path).read_text()); manifest['endpoint']='https://wrong.example/'
    (tmp_path/manifest_path).write_text(json.dumps(manifest))
    result=holdout._source_match_audit(coverage,archive_path,checkpoint=checkpoint,raw_dir=tmp_path)
    assert result['status']=='source_match_gate_fail'
    assert result['byCrop']['cabbage']['comparableDays']==4

def test_dry_run_never_creates_session(capsys,monkeypatch,tmp_path):
    monkeypatch.setattr(holdout.requests,'Session',lambda: (_ for _ in ()).throw(AssertionError('network used')))
    assert holdout.main(['--output',str(tmp_path)])==0
    report=json.loads(capsys.readouterr().out)
    assert report['requestSent'] is False
    assert report['targetedMissingPasses']==2

def test_explicit_window_is_bounded_and_archive_overlap_is_dynamic(capsys,monkeypatch,tmp_path):
    old=(holdout.REQUEST_START,holdout.REQUEST_END,holdout.PERIODS,holdout.PROTOCOL)
    try:
        holdout.configure_window('2026-06-01','2026-06-10')
        assert holdout.PERIODS==(('overlap',dt.date(2026,6,1),dt.date(2026,6,10)),)
        assert len(holdout.request_matrix())==12  # weekly anchors plus the inclusive end point
        with pytest.raises(ValueError): holdout.configure_window('2026-01-01','2026-05-01')
        with pytest.raises(ValueError): holdout.configure_window('2026-01-01',None)
        holdout.configure_window('2024-06-01','2024-06-10')
        assert holdout.PERIODS==(('holdout',dt.date(2024,6,1),dt.date(2024,6,10)),)
        assert holdout.main(['--start','2026-09-01','--end','2026-09-03','--output',str(tmp_path)])==0
        report=json.loads(capsys.readouterr().out)
        assert report['requestedDateWindow']==['2026-09-01','2026-09-03']
        assert report['dateCounts']=={'overlap':0,'holdout':3}
        assert len(report['anchorRequests'])==10  # two spaced anchors times five crops
    finally:
        holdout.REQUEST_START,holdout.REQUEST_END,holdout.PERIODS,holdout.PROTOCOL=old

def test_combined_source_appends_only_hash_verified_four_crop_rows(tmp_path,monkeypatch):
    archive=tmp_path/'archive.zip'; archive.write_bytes(b'pinned synthetic archive')
    monkeypatch.setattr(holdout.nt,'ARCHIVE_SHA256',holdout._sha(archive.read_bytes()))
    june_days=list(holdout._daterange(holdout.OVERLAP_START,holdout.OVERLAP_END))[:5]
    base={crop:{**{'2026-06-30':{'price':'50'}},**{day.isoformat():{'price':'100'} for day in june_days}}
          for crop in ('cabbage','radish','onion','pepper')}
    monkeypatch.setattr(holdout.nt,'parse_archive',lambda data:{'observations':{k:dict(v) for k,v in base.items()}})
    requests={}; rows=[]
    for index,(name,crop) in enumerate(zip([x[0] for x in holdout.CATEGORIES[:4]],('cabbage','radish','onion','pepper'))):
        day=dt.date(2026,7,1)+dt.timedelta(days=index)
        values={'saledate':day.strftime('%Y%m%d'),'selectDay':str(100+index),'selectDayWgt':'20'}
        raw=_body([values]); raw_name=f'{name}.json'; (tmp_path/raw_name).write_bytes(raw)
        rid=f'req{index}'; digest=holdout._sha(raw)
        request={'requestedDate':(day+dt.timedelta(days=1)).isoformat(),'category':name,
            'requestParams':holdout._params(day+dt.timedelta(days=1),holdout.CATEGORIES[index][1],holdout.CATEGORIES[index][2])}
        manifest_name=f'{rid}.manifest.json'
        manifest={'endpoint':holdout.ENDPOINT,'requestId':rid,'request':request,'rawFile':raw_name,'sha256':digest,
            'httpStatus':200,'truncated':False,'transportError':None}
        (tmp_path/manifest_name).write_text(json.dumps(manifest))
        requests[rid]={'request':request,'attempts':[{'request':request,'sha256':digest,'rawFile':raw_name,'httpStatus':200,
            'truncated':False,'transportError':None,'manifestFile':manifest_name}]}
        rows.append({'category':name,'period':'holdout','date':day.isoformat(),'state':'observed',
            'sourceValues':values,
            'provenance':[{'requestId':rid,'responseSha256':digest}]})
    june_rows=[]
    for name,_,_ in holdout.CATEGORIES[:4]:
        for day in june_days:
            june_rows.append({'category':name,'period':'overlap','date':day.isoformat(),'state':'observed',
                'sourceValues':_row(day,'100','20'),'provenance':[]})
    june_checkpoint=_attach_raw_provenance(june_rows,tmp_path,'june')
    requests.update(june_checkpoint['requests']); rows.extend(june_rows)
    (tmp_path/'checkpoint.json').write_text(json.dumps({'requests':requests}))
    report=tmp_path/'coverage-report.json'; report.write_text(json.dumps({'coverage':rows}))
    combined=holdout.build_combined_target_source(report,archive)
    assert combined['observations']['cabbage']['2026-07-01']=={'price':'100'}
    assert len(combined['provenance']['appended'])==4
    assert combined['sourceDigests']['archiveSha256']==holdout._sha(archive.read_bytes())
    requests['req0']['request']['category']='radish'
    (tmp_path/'checkpoint.json').write_text(json.dumps({'requests':requests}))
    with pytest.raises(ValueError,match='raw response supports'):
        holdout.build_combined_target_source(report,archive)
    requests['req0']['request']['category']='cabbage'
    (tmp_path/'checkpoint.json').write_text(json.dumps({'requests':requests}))
    report.write_text(json.dumps({'coverage':[r for r in rows if r['period']=='holdout']}))
    with pytest.raises(ValueError,match='June source-match gate'):
        holdout.build_combined_target_source(report,archive)

class Response:
    def __init__(self,body,status=200): self.body=body; self.status_code=status; self.headers={'Content-Type':'application/json'}; self.closed=False
    def iter_content(self,chunk_size): yield self.body
    def close(self): self.closed=True

class Session:
    def __init__(self,responses): self.responses=iter(responses); self.calls=[]; self.headers={}; self.verify=False
    def get(self,url,**kwargs): self.calls.append((url,kwargs)); return next(self.responses)

def test_collect_preserves_raw_provenance_quota_tls_and_resume(tmp_path,monkeypatch):
    old_periods=holdout.PERIODS; old_categories=holdout.CATEGORIES
    day1=dt.date(2026,6,20); day2=dt.date(2026,7,1)
    try:
        holdout.PERIODS=(('overlap',day1,day1),('holdout',day2,day2))
        holdout.CATEGORIES=(('cabbage','1001',''),)
        matrix=holdout.request_matrix()
        session=Session([Response(_body([_row(dt.date.fromisoformat(r['requestedDate']))])) for r in matrix])
        ticks=iter(range(10000)); monkeypatch.setattr(holdout.time,'monotonic',lambda:next(ticks))
        monkeypatch.setattr(holdout,'_source_match_audit',lambda coverage,archive_path,**kwargs:{'status':'archive_unavailable','comparedCount':0})
        now=lambda:dt.datetime(2026,10,9,tzinfo=dt.timezone.utc)
        report=holdout.collect(tmp_path,session=session,now=now,sleep=lambda _:None)
        assert len(session.calls)==len(matrix)
        assert session.verify is True and session.headers['User-Agent']=='Mozilla/5.0'
        assert all(call[1]['verify'] is True and call[1]['timeout']==holdout.TIMEOUT_SECONDS for call in session.calls)
        assert all((tmp_path/attempt['rawFile']).exists() for entry in json.loads((tmp_path/'checkpoint.json').read_text())['requests'].values() for attempt in entry['attempts'])
        assert report['coverageCompleteAsClassified'] is True
        again=holdout.collect(tmp_path,session=Session([]),now=now,sleep=lambda _:None)
        assert again==report
    finally:
        holdout.PERIODS=old_periods; holdout.CATEGORIES=old_categories

def test_collection_targets_each_missing_date_twice_and_never_calls_it_confirmed_no_trade(tmp_path,monkeypatch):
    old_periods=holdout.PERIODS; old_categories=holdout.CATEGORIES
    day1=dt.date(2026,6,20); day2=dt.date(2026,7,1)
    try:
        holdout.PERIODS=(('overlap',day1,day1),('holdout',day2,day2))
        holdout.CATEGORIES=(('cabbage','1001',''),)
        before=[_body([_row(day-dt.timedelta(days=1))]) for day in (day1,day2)]
        session=Session([Response(b) for b in before+before+before+before])
        ticks=iter(range(10000)); monkeypatch.setattr(holdout.time,'monotonic',lambda:next(ticks))
        monkeypatch.setattr(holdout,'_source_match_audit',lambda coverage,archive_path,**kwargs:{'status':'archive_unavailable','comparedCount':0})
        report=holdout.collect(tmp_path,session=session,now=lambda:dt.datetime(2026,10,9,tzinfo=dt.timezone.utc),sleep=lambda _:None)
        assert len(session.calls)==6  # two anchors plus two targeted passes per date
        assert {r['state'] for r in report['coverage']}=={'no_trade_unconfirmed'}
        assert report['coverageCompleteAsClassified'] is True
    finally:
        holdout.PERIODS=old_periods; holdout.CATEGORIES=old_categories
