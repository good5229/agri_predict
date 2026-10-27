import datetime as dt
import json
import math

import pytest

from agri_data import period_pooled_lgbm as m
from agri_data import period_pooled_lgbm_score as score
import numpy as np


def _series(start=dt.date(2020, 1, 1), n=160):
    return {start + dt.timedelta(days=i): (100 + i * .2, 50 + i) for i in range(n)}


def _train_rows(per_cell=5):
    rows = []
    for ci, crop in enumerate(m.CROPS):
        for hi, h in enumerate(m.HORIZONS):
            for j in range(per_cell):
                target = (dt.date(2020, 1, 1) + dt.timedelta(days=10*j),
                          dt.date(2020, 1, 10) + dt.timedelta(days=10*j))
                origin = target[1] + dt.timedelta(days=3)
                features = {f"x{k}": float(ci + hi + j*k/10) for k in range(1, 4)}
                rows.append({"crop": crop, "horizon": h, "target": target,
                             "targetEnd": target[1], "origin": origin,
                             "actual": 100 + ci*5 + hi*3 + j,
                             "p_ref": 100., "features": features})
    return rows


def test_features_are_strictly_asof_and_keep_empty_window_flags():
    series = _series()
    origin = dt.date(2020, 6, 20)
    target = (dt.date(2020, 7, 1), dt.date(2020, 7, 10))
    targets = m.proxy._target_periods(series)
    before = m.feature_row(series, targets, "cabbage", 2, target, origin)
    # Add extreme future observations and a future label; the historical row must not move.
    future = dict(series)
    for i in range(10):
        future[dt.date(2020, 8, 1) + dt.timedelta(days=i)] = (1e12, 1e12)
    after = m.feature_row(future, m.proxy._target_periods(future), "cabbage", 2, target, origin)
    assert before == after
    assert before["referenceDate"] <= before["inputCutoff"]
    empty = m.feature_row({}, {}, "cabbage", 2, target, origin)
    assert empty["referencePrice"] is None
    assert empty["features"]["price_available_7"] == 0
    assert empty["features"]["price_meanlog_7"] == 0


def test_weights_deduplicate_targets_and_balance_all_eight_cells():
    rows = _train_rows(4)
    # Duplicate one crop-target row in its cell; inverse multiplicity keeps its total mass stable.
    duplicate = dict(rows[0]); rows.append(duplicate)
    weights, cells = m.row_weights(rows)
    assert all(cells[f"{c}:{h}"] > 0 for c in m.CROPS for h in m.HORIZONS)
    assert weights.mean() == pytest.approx(1)
    sums = [sum(weights[i] for i, row in enumerate(rows)
                 if row["crop"] == c and row["horizon"] == h)
            for c in m.CROPS for h in m.HORIZONS]
    assert max(sums) - min(sums) < 1e-10
    counts = {}
    for row, weight in zip(rows, weights):
        key = (row["crop"], row["target"])
        counts.setdefault(key, []).append(weight)
    group = counts[(rows[0]["crop"], rows[0]["target"])]
    assert len(group) == 3  # two h=2 rows plus the same target's h=3 row
    assert group[0] == pytest.approx(group[-1])


def test_empty_training_cell_blocks_fit():
    rows = [r for r in _train_rows(1) if not (r["crop"] == "garlic" and r["horizon"] == 3)]
    with pytest.raises(ValueError, match="training cell empty"):
        m.row_weights(rows)


def test_horizon_is_numeric_and_only_crop_is_categorical_for_lgbm():
    rows = _train_rows(2)
    x, names = m._matrix(rows, linear=False)
    assert "horizon" not in names
    assert x[:, -1].tolist() == [r["horizon"] for r in rows]
    xl, names_l = m._matrix(rows, linear=True)
    assert names_l == names
    assert xl.shape[1] == len(names) + 4  # three crop indicators plus numeric horizon


def test_ridge_scales_only_numeric_features_and_model_reloads(tmp_path):
    rows = _train_rows(4)
    fitted = m.fit_model(rows, "L")
    assert len(fitted["scalerMean"]) == len(fitted["featureNames"])
    identity = {"test": "synthetic"}
    path = tmp_path / "ridge.joblib"
    sha = m._save_fit(fitted, path, identity)
    loaded = m._load_fit(path, sha, identity)
    z1 = m.predict_rows(fitted, rows)
    z2 = m.predict_rows(loaded, rows)
    np.testing.assert_allclose(z1, z2, rtol=1e-12, atol=1e-9)


def test_lightgbm_synthetic_fit_uses_shared_sample_weights_and_reloads(tmp_path):
    pytest.importorskip("lightgbm")
    rows = _train_rows(8)
    cfg = {**m.CONFIGS["A"], "n_estimators": 6, "min_data_in_leaf": 2}
    fitted = m.fit_model(rows, "G", cfg)
    assert fitted["weightSha256"] == m.sha_json(m.row_weights(rows)[0].tolist())
    assert fitted["lightgbmParameters"]["objective"] == "regression_l1"
    assert fitted["lightgbmParameters"]["num_threads"] == 1
    identity = {"test": "synthetic"}
    path = tmp_path / "gbm.joblib"
    sha = m._save_fit(fitted, path, identity)
    loaded = m._load_fit(path, sha, identity)
    np.testing.assert_allclose(m.predict_rows(fitted, rows), m.predict_rows(loaded, rows),
                               rtol=1e-12, atol=1e-9)


def test_metrics_and_holm_match_fixed_loss_direction():
    assert score._metric([100, 200], [90, 220]) == {"n": 2, "mae": 15., "mape": 10.}
    slots=[{'name':str(i),'p':p} for i,p in enumerate((.01,.04,.03,None))]
    assert score._holm_slots(slots)=={'0':pytest.approx(.04),'1':pytest.approx(.09),'2':pytest.approx(.09),'3':None}
    pairs=[]
    for i in range(40):
        start=dt.date(2020,1,1)+dt.timedelta(days=10*i)
        end=start+dt.timedelta(days=9)
        pairs.append(((start,end),100.,90.,100.))
    dm=score._dm_pairs(pairs,'ae',3)
    assert dm['mean']==pytest.approx(10)
    assert dm['status']=='nonpositive_hac_variance' and dm['p'] is None


def test_synthetic_pooled_origin_fits_g_and_l_once_and_persists_predictions(tmp_path, monkeypatch):
    train = _train_rows(5)
    origin = dt.date(2021, 12, 25)
    current = []
    schedules = []
    for ci, crop in enumerate(m.CROPS):
        for hi, h in enumerate(m.HORIZONS):
            target = (dt.date(2022, 1, 1) + dt.timedelta(days=10*hi),
                      dt.date(2022, 1, 10) + dt.timedelta(days=10*hi))
            schedules.append({"crop": crop, "horizon": h, "target": target})
    features = {f"x{k}": float(k) for k in range(1, 4)}
    monkeypatch.setattr(m, "feature_row", lambda series, targets, crop, h, target, issue:
                        {"referencePrice": 100., "features": dict(features),
                         "inputCutoff": (issue-dt.timedelta(days=2)).isoformat(),
                         "crop": crop, "horizon": h, "targetStart": target[0].isoformat(),
                         "targetEnd": target[1].isoformat(), "targetKey": m._periodkey(target)})
    base = {}
    for s in schedules:
        key=(s["crop"],s["horizon"],s["target"][0].isoformat(),s["target"][1].isoformat(),origin.isoformat())
        base[key]={"P":110.}
    ledger=[]
    cfg={**m.CONFIGS["A"],"n_estimators":4,"min_data_in_leaf":2}
    outputs=m._forecast_origin("develop",origin,schedules,train,
                               {crop:{} for crop in m.CROPS},{crop:{} for crop in m.CROPS},
                               [("L","L",None),("G","A",cfg)],tmp_path,ledger,base)
    assert len(outputs)==8
    assert len(ledger)==2 and all(row["status"]=="complete" for row in ledger)
    assert ledger[0]["weightSha256"]==ledger[1]["weightSha256"]
    assert ledger[0]["numericFeatureMatrixSha256"]==ledger[1]["numericFeatureMatrixSha256"]
    assert all(set(row["predictions"])=={"L-L","G-A"} for row in outputs)
    assert all(row["predictions"]["G-A"]["predicted"]>0 for row in outputs)
    future=[{**r,'target':(dt.date(2022,2,1),dt.date(2022,2,10)),
             'targetEnd':dt.date(2022,2,10),'origin':dt.date(2022,2,13),
             'actual':1e9,'features':{k:1e9 for k in r['features']}} for r in train[:8]]
    ledger_future=[]
    (tmp_path/'future').mkdir()
    outputs_future=m._forecast_origin('develop',origin,schedules,train+future,
        {crop:{} for crop in m.CROPS},{crop:{} for crop in m.CROPS},
        [('L','L',None),('G','A',cfg)],tmp_path/'future',ledger_future,base)
    assert [x['predictions'] for x in outputs]==[x['predictions'] for x in outputs_future]
    assert [x['weightSha256'] for x in ledger]==[x['weightSha256'] for x in ledger_future]
    assert [x['numericFeatureMatrixSha256'] for x in ledger]==[x['numericFeatureMatrixSha256'] for x in ledger_future]


def test_synthetic_development_selection_and_test_score_preserve_protocol(monkeypatch,tmp_path):
    # Exercise the scorer transition using synthetic forecasts only; no source/API
    # validation or real-data stage is called.
    dev=[]
    periods=m.proxy.dekads(dt.date(2022,1,1),dt.date(2023,12,31))[:72]
    for ci,crop in enumerate(m.CROPS):
      for hi,h in enumerate(m.HORIZONS):
       for j,(start,end) in enumerate(periods):
        actual=None if ci==0 and j<10 else 100.+ci*4+j*.1
        row={'crop':crop,'horizon':h,'targetStart':start.isoformat(),'targetEnd':end.isoformat(),
             'origin':(end+dt.timedelta(days=2)).isoformat(),'actual':actual,
             'P':100.,'B0':101.,'G-A':100.5,'G-B':101.,'G-C':99.5,'L-L':102.,
             'G-A_status':'ok','G-B_status':'ok','G-C_status':'ok','L-L_status':'ok'}
        dev.append(row)
    test=[];raw=[];all_periods=m.proxy.dekads(dt.date(2024,1,1),dt.date(2026,9,30))
    for ci,crop in enumerate(m.CROPS):
      for hi,h in enumerate(m.HORIZONS):
       for j,(start,end) in enumerate(all_periods):
        key=(crop,h,start.isoformat(),end.isoformat(),(end+dt.timedelta(days=2)).isoformat())
        unavailable=j < (1 if h==2 else 2)
        actual=None if unavailable else 100.+ci*4+hi+j*.15
        row={'crop':crop,'horizon':h,'targetStart':start.isoformat(),'targetEnd':end.isoformat(),
             'targetKey':start.isoformat(),'targetPeriodKey':m._periodkey((start,end)),
             'origin':key[-1],'inputCutoff':(end+dt.timedelta(days=0)).isoformat(),
             'actual':actual,'year':start.year,'p_ref':100.,'P':100.,'B0':101.,
             'G-A':100.5,'G-B':101.,'G-C':99.5,'L-L':102.,'G-A_pred_z':.005,'G-B_pred_z':.01,'G-C_pred_z':-.005,'L-L_pred_z':.02,
             'G-A_status':None if unavailable else 'ok','L-L_status':None if unavailable else 'ok',
             'policyUnavailable':unavailable}
        test.append(row)
        raw.append({'crop':crop,'horizon':h,'targetStart':start.isoformat(),'targetEnd':end.isoformat(),
                    'origin':key[-1],'policyUnavailable':unavailable,
                    'forecasts':{'P':{'predicted':None if unavailable else 100.},
                                 'B0':{'predicted':None if unavailable else 101.}}})
    # One issued G-A row records an inverse-transform overflow and explicit P fallback.
    test[1].update({'G-B':100.,'G-B_pred_z':None,'G-B_pred_z_raw':'Infinity',
                    'G-B_pred_z_nonfinite':True,'G-B_status':'technical_fallback_P',
                    'G-B_fallbackReason':'overflow'})
    selection_path=tmp_path/'development-selection.json'
    v3dir=tmp_path/'v3';v3dir.mkdir()
    (v3dir/'selection.json').write_text(json.dumps({'P':{c:{'method':'R'} for c in m.CROPS}}))
    old_path=tmp_path/'old.json'
    old_path.write_text(json.dumps({'crops':{c:{str(h):{'baselineDevelopmentSelection':'B0b'} for h in m.HORIZONS} for c in m.CROPS}}))
    dev_art=[]
    for row in dev:
        dev_art.append({k:v for k,v in row.items() if k not in ('actual','P','B0','G-A','G-B','G-C','L-L','G-A_status','G-B_status','G-C_status','L-L_status')} |
                       {'predictions':{name:{'predicted':row[name],'status':row[f'{name}_status']} for name in ('G-A','G-B','G-C','L-L')}})
    test_art=[]
    for row in test:
        test_art.append({k:v for k,v in row.items() if k not in ('actual','P','B0','G-A','G-B','G-C','L-L','G-A_pred_z','G-B_pred_z','G-C_pred_z','L-L_pred_z','G-A_status','L-L_status')} |
                        {'predictions':({} if row['policyUnavailable'] else {name:{'predicted':row[name],'pred_z':row[f'{name}_pred_z'],'pred_z_raw':row.get(f'{name}_pred_z_raw') or format(row[f'{name}_pred_z'],'.17g'),'pred_z_raw_nonfinite':bool(row.get(f'{name}_pred_z_nonfinite',False)),'status':row.get(f'{name}_status') or 'ok','fallbackReason':row.get(f'{name}_fallbackReason')} for name in ('G-A','G-B','G-C','L-L')})})
    dev_meta=m.write_rows(tmp_path/'development-forecasts.parquet',dev_art)
    test_meta=m.write_rows(tmp_path/'forecasts.parquet',test_art)
    actual_targets={c:{} for c in m.CROPS}
    for row in dev+test:
        if row.get('actual') is not None:
            target=(dt.date.fromisoformat(row['targetStart']),dt.date.fromisoformat(row['targetEnd']))
            actual_targets[row['crop']][target]={'actual':row['actual']}
    dev_raw=[]
    for row in dev:
        dev_raw.append({'crop':row['crop'],'horizon':row['horizon'],'targetStart':row['targetStart'],'targetEnd':row['targetEnd'],'origin':row['origin'],'forecasts':{'R':{'predicted':100.}}})
    monkeypatch.setattr(m,'V3_OUT',v3dir)
    monkeypatch.setattr(m,'OLD_REPORT',old_path)
    monkeypatch.setattr(m.proxy,'baseline_predictions',lambda series,targets,h,start,end:{'B0b':{(dt.date.fromisoformat(r['targetStart']),dt.date.fromisoformat(r['targetEnd'])):{'predicted':101.,'origin':r['origin']} for r in dev if r['horizon']==h}})
    monkeypatch.setattr(m,'_sources',lambda:({c:{} for c in m.CROPS},[],actual_targets))
    monkeypatch.setattr(m,'_v3_rows',lambda name:dev_raw if name=='forecasts.parquet' else raw)
    monkeypatch.setattr(m,'run_identity',lambda:'synthetic-run')
    monkeypatch.setattr(m,'code_hash',lambda:'synthetic-code')
    monkeypatch.setattr(m,'_stage_path',lambda out,stage:tmp_path/f'{stage}.json')
    monkeypatch.setattr(m,'load_stage',lambda stage,out:{'forecastArtifact':dev_meta if stage=='develop' else test_meta,
                    'selectionSha256':m.digest(selection_path) if stage=='forecast' and selection_path.exists() else None})
    monkeypatch.setattr(m,'save_stage',lambda stage,doc,out:doc)
    monkeypatch.setattr(score,'_verify_fit_ledger',lambda out,selection:[])
    d=score.score_development(tmp_path)
    assert json.loads(selection_path.read_text())['chosenConfig'] in ('A','B','C')
    selection=json.loads(selection_path.read_text())
    assert len(selection['developmentNativeFallback'])==4
    # The score path rereads the frozen selection digest from the forecast checkpoint.
    s=score.score(tmp_path)
    assert s['pairedKeys']==780
    report=json.loads((tmp_path/'score-report.json').read_text())
    chosen='G-'+selection['chosenConfig']
    assert report['diagnostics']['response']['cabbage:2'][chosen]['rawPredZNonfiniteCount']==1
