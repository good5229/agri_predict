"""Post-gate descriptive price-error and probability diagnostics for V6."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import period_crop_profile_diagnostics as price_diag
from . import period_pooled_lgbm as v5
from . import period_proxy_experiment as proxy
from . import period_regime_information_v6 as v6

ROOT = Path(__file__).resolve().parents[1]
OUT = v6.OUT
FIXED_CUTS = (0.1087015664, 0.1837007705)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)
    return _sha(path)


def _read_frame(path):
    import pandas as pd
    return pd.read_parquet(path)


def _key(row):
    return (row['crop'], int(row['horizon']), str(row['origin']),
            str(row['targetStart']), str(row['targetEnd']))


def _price_rows():
    """Use the established V3/V5 joins and their frozen development policy."""
    series = proxy.series_from_raw(json.loads(proxy.SOURCE.read_text(encoding='utf-8')))
    development = price_diag.build_development_rows(
        _read_frame(v5.OUT / 'development-forecasts.parquet'),
        _read_frame(v5.V3_OUT / 'forecasts.parquet'), series)
    test = price_diag.build_metric_rows(
        _read_frame(v5.OUT / 'forecasts.parquet'),
        _read_frame(v5.V3_OUT / 'forecasts_test.parquet'), series,
        require_planned_grid=True)
    return {'develop': development, 'test': test}


def _risk_rows(out, split):
    name = 'development' if split == 'develop' else 'test'
    path = Path(out) / f'{name}-risk-forecasts.parquet'
    if not path.is_file():
        return None
    df = _read_frame(path)
    rows = df.to_dict('records')
    schedule = v6.build_schedule(split)
    expected = len(schedule) * len(v6.CROPS)
    keys = [_key(r) for r in rows]
    if len(rows) != expected or len(set(keys)) != expected:
        raise ValueError(f'blocked_integrity: {name} risk forecast grid is {len(rows)}/{expected} or duplicated')
    expected_keys = set()
    for sched in schedule:
        origin, (start, end) = sched['origin'], sched['target']
        for crop in v6.CROPS:
            expected_keys.add((crop, v6.HORIZON, origin.isoformat(), start.isoformat(), end.isoformat()))
    if set(keys) != expected_keys:
        raise ValueError(f'blocked_integrity: {name} risk forecast keys differ from frozen schedule')
    if any(int(r.get('horizon', -1)) != v6.HORIZON or r.get('split') != split for r in rows):
        raise ValueError(f'blocked_integrity: {name} risk forecast split/horizon mismatch')
    if any('actual' in r or 'event' in r for r in rows):
        raise ValueError('blocked_integrity: risk forecast artifact contains joined labels')
    return rows


def _valid(x):
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _risk_price_join(risk, price_rows):
    by = {_key(r): r for r in price_rows}
    joined, missing = [], []
    emitted = {'RB': 0, 'RV': 0, 'paired': 0}
    for r in risk:
        for method in ('RB', 'RV'):
            if _valid(r.get('prob_' + method)):
                emitted[method] += 1
        if all(_valid(r.get('prob_' + method)) for method in ('RB', 'RV')):
            emitted['paired'] += 1
        p = by.get(_key(r))
        if not p or not all(_valid(p.get(k)) for k in ('actual', 'G-B_pred', 'P_pred')):
            missing.append({'crop': r['crop'], 'targetStart': r['targetStart'], 'reason': 'price_key_or_G_P_unavailable'})
            continue
        if _valid(r.get('p_ref')) and not math.isclose(float(r['p_ref']), float(p['p_ref']), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('blocked_integrity: risk and saved price p_ref disagree')
        joined.append({**r, **{k: p.get(k) for k in ('actual','p_ref','G-B_ae','G-B_ape','P_ae','P_ape')},
                       'year': int(r['year'])})
    return joined, {'riskPlannedRows': len(risk), 'riskProbabilityRowsByMethod': emitted,
                    'priceJoinedRows': len(joined),
                    'missingPriceJoinRows': len(missing), 'missingReasons': missing}


def _fixed_feature_descriptive(out):
    """Gate-failure allowance: existing fixed 28-day feature terciles only."""
    catalog_path = Path(out) / 'v5-feature-catalog.parquet'
    if not catalog_path.is_file():
        return {'status': 'unavailable', 'reason': 'V6 feature catalog is absent'}
    rows = _read_frame(catalog_path).to_dict('records')
    result = []
    source_by_year_crop = {(y, c): [r for r in rows if dt.date.fromisoformat(str(r['targetStart'])).year == y and r['crop'] == c]
                           for y in (2024, 2025, 2026) for c in v6.CROPS}
    for year in (2024, 2025, 2026):
        for crop in v6.CROPS:
            cell = source_by_year_crop[(year, crop)]
            valid_feature = [r for r in cell if int((r.get('features') or {}).get('price_available_28', 0) or 0) == 1
                             and _valid((r.get('features') or {}).get('price_sd_28'))]
            for group, pred in (
                ('low', lambda x: x <= FIXED_CUTS[0]),
                ('middle', lambda x: FIXED_CUTS[0] < x <= FIXED_CUTS[1]),
                ('high', lambda x: x > FIXED_CUTS[1])):
                sel = []
                for r in rows:
                    day = dt.date.fromisoformat(str(r['targetStart']))
                    sd = (r.get('features') or {}).get('price_sd_28')
                    y, pref = r.get('actual'), r.get('p_ref')
                    available = int((r.get('features') or {}).get('price_available_28', 0) or 0) == 1
                    if day.year == year and r['crop'] == crop and available and _valid(sd) and _valid(y) and _valid(pref) and float(pref) > 0:
                        if pred(float(sd)):
                            sel.append(v6.label_event(y, pref) == 1)
                result.append({'year': year, 'crop': crop, 'group': group, 'n': len(sel),
                               'uniqueTargets': len(sel), 'riskRate': float(np.mean(sel)) if sel else None,
                               'catalogRowsWithTarget': len(cell), 'featureAvailableRows': len(valid_feature),
                               'featureCoverageWithinCatalog': len(valid_feature) / len(cell) if cell else None,
                               'plannedTargets': 27 if year == 2026 else 36,
                               'featureCoverageOfPlanned': len(valid_feature) / (27 if year == 2026 else 36),
                               'featureCuts': list(FIXED_CUTS), 'feature': 'V5 price_sd_28'})
    return {'status': 'descriptive_only', 'groups': result,
            'warning': 'Fixed legacy feature cuts; actual future event labels are descriptive and were not inputs.'}


def _price_error_groups(joined, q33, q67, split):
    result = []
    for year in sorted({int(r['year']) for r in joined}):
        for crop in v6.CROPS:
            for group, include in (
                ('low', lambda p: p <= q33),
                ('middle', lambda p: q33 < p <= q67),
                ('high', lambda p: p > q67)):
                items = [r for r in joined if int(r['year']) == year and r['crop'] == crop
                         and _valid(r.get('prob_RV')) and include(float(r['prob_RV']))]
                row = {'split': split, 'year': year, 'crop': crop, 'riskGroup': group,
                       'q33': q33, 'q67': q67, 'n': len(items),
                       'uniqueTargets': len({r['targetStart'] for r in items}),
                       'riskRate': float(np.mean([v6.label_event(r['actual'], r['p_ref']) for r in items])) if items else None}
                for model in ('G-B','P'):
                    for metric in ('ae','ape'):
                        vals = [float(r[f'{model}_{metric}']) for r in items if _valid(r.get(f'{model}_{metric}'))]
                        row[f'{model}_{metric.upper()}_mean'] = float(np.mean(vals)) if vals else None
                        row[f'{model}_{metric.upper()}_n'] = len(vals)
                result.append(row)
    return result


def _probability_diagnostics(rows):
    output = {}
    for method in ('RB', 'RV'):
        strata = []
        years = sorted({int(r['year']) for r in rows})
        for year in years:
            for crop in v6.CROPS:
                for event in (0, 1):
                    q = [r for r in rows if int(r['year']) == year and r['crop'] == crop
                         and r.get('event') is not None and int(r['event']) == event
                         and _valid(r.get('prob_' + method))]
                    ps = np.asarray([float(r['prob_' + method]) for r in q])
                    es = np.asarray([int(r['event']) for r in q])
                    losses = (ps-es)**2
                    strata.append({'year': year, 'crop': crop, 'event': event, 'n': len(q),
                                   'meanProbability': float(ps.mean()) if len(q) else None,
                                   'observedEventRate': float(es.mean()) if len(q) else None,
                                   'probabilityMinusEvent': float((ps-es).mean()) if len(q) else None,
                                   'overpredictionN': int(np.sum(ps > es)) if len(q) else 0,
                                   'underpredictionN': int(np.sum(ps < es)) if len(q) else 0,
                                   'brierContributionSum': float(losses.sum()) if len(q) else None,
                                   'brierContributionMean': float(losses.mean()) if len(q) else None})
        output[method] = strata
    return output


def risk_probability_cuts(rows):
    probs = [float(r['prob_RV']) for r in rows if int(r.get('year', 0)) == 2022 and _valid(r.get('prob_RV'))]
    if not probs:
        return {'status': 'unavailable', 'reason': 'no available 2022 development RV probabilities', 'n': 0}
    q33, q67 = map(float, np.quantile(probs, [1/3, 2/3]))
    return {'status': 'available', 'source': 'available 2022 development RV probabilities',
            'q33': q33, 'q67': q67, 'n': len(probs)}


def _overall_test_deltas(test_score):
    by = test_score.get('byYearCrop', {})
    years = (2024, 2025, 2026)
    cells = [by.get(f'{y}:{c}', {}) for y in years for c in v6.CROPS]
    supported = all(x.get('n', 0) >= 24 and x.get('positive', 0) >= 5 and x.get('negative', 0) >= 5 for x in cells)
    if not supported:
        return {'status': 'inconclusive_support', 'reason': 'test year/crop support below frozen minimum'}
    brier = []; logloss = []
    for crop in v6.CROPS:
        n = sum(by[f'{y}:{crop}']['n'] for y in years)
        brier.append(sum(by[f'{y}:{crop}']['n'] * (by[f'{y}:{crop}']['RV']['brier']-by[f'{y}:{crop}']['RB']['brier']) for y in years) / n)
        logloss.append(sum(by[f'{y}:{crop}']['n'] * (by[f'{y}:{crop}']['RV']['logLoss']-by[f'{y}:{crop}']['RB']['logLoss']) for y in years) / n)
    return {'status': 'available', 'equalCropBrierDelta': float(np.mean(brier)),
            'equalCropLogLossDelta': float(np.mean(logloss)), 'cropBrierDeltas': dict(zip(v6.CROPS, map(float,brier))),
            'cropLogLossDeltas': dict(zip(v6.CROPS, map(float,logloss)))}


def _final_status(gate, test_score):
    if gate.get('status') != 'pass':
        return gate.get('status')
    if not test_score:
        raise ValueError('development gate passed but test score artifact is absent')
    by = test_score.get('byYearCrop', {})
    years = (2024, 2025, 2026)
    cells = [by.get(f'{y}:{c}', {}) for y in years for c in v6.CROPS]
    overall = _overall_test_deltas(test_score)
    if overall['status'] != 'available':
        return 'inconclusive_support'
    deltas = {}
    for y in years:
        xs = [by[f'{y}:{c}'] for c in v6.CROPS]
        deltas[y] = (float(np.mean([x['RV']['brier']-x['RB']['brier'] for x in xs])),
                     float(np.mean([x['RV']['logLoss']-x['RB']['logLoss'] for x in xs])))
    if overall['equalCropBrierDelta'] < 0 and all(b < 0 and ll <= .02 for b, ll in deltas.values()):
        return 'risk_signal_supported_exploratory'
    return 'risk_signal_inconsistent'


def finalize(out=OUT):
    out = Path(out)
    gate_path = out / 'development-gate.json'
    if not gate_path.is_file():
        raise ValueError('missing frozen development gate')
    gate = _json(gate_path)
    dev_score_path = out / 'development-risk-scores.json'
    if not dev_score_path.is_file():
        raise ValueError('missing development risk scores')
    dev_score = _json(dev_score_path)
    risk = {'develop': _risk_rows(out, 'develop'), 'test': _risk_rows(out, 'test')}
    test_score_path = out / 'test-risk-scores.json'
    test_score = _json(test_score_path) if test_score_path.is_file() else None
    test_ledger_path = out / 'test-fit-ledger.json'
    if gate.get('status') != 'pass' and (risk['test'] is not None or test_score is not None or test_ledger_path.exists()):
        raise ValueError('blocked_integrity: test fit/score artifacts exist after failed development gate')
    if gate.get('status') == 'pass' and (risk['test'] is None or test_score is None):
        raise ValueError('development passed but complete test forecast/score artifacts are missing')
    if gate.get('testFitsAllowed') is not (gate.get('status') == 'pass'):
        raise ValueError('blocked_integrity: development gate test permission flag is inconsistent')
    preflight = _json(out/'preflight.json')
    for label, score in (('development', dev_score), ('test', test_score)):
        if score is None:
            continue
        forecast_path = out / f'{label}-risk-forecasts.parquet'
        ledger_path = out / f'{label}-fit-ledger.json'
        if not forecast_path.is_file() or not ledger_path.is_file():
            raise ValueError(f'blocked_integrity: {label} forecast/fit ledger is incomplete')
        ledger = _json(ledger_path)
        forecast_digest = _sha(forecast_path)
        if (score.get('forecastSha256') != forecast_digest or ledger.get('forecastSha256') != forecast_digest
                or not ledger.get('complete') or score.get('runIdentity') != preflight.get('runIdentity')
                or ledger.get('runIdentity') != preflight.get('runIdentity')
                or score.get('preflightSha256') != _sha(out/'preflight.json')
                or ledger.get('preflightSha256') != _sha(out/'preflight.json')
                or score.get('ledgerSha256') != _sha(ledger_path)):
            raise ValueError(f'blocked_integrity: {label} forecast/score/ledger hash or completion mismatch')
        current_rows = risk['develop' if label == 'development' else 'test']
        if int(score.get('issuedRows', -1)) != len(current_rows):
            raise ValueError(f'blocked_integrity: {label} issued row count mismatch')
    if gate.get('scoreSha256') != _sha(dev_score_path):
        raise ValueError('blocked_integrity: development gate is not bound to current score file')
    if gate.get('runIdentity') != preflight.get('runIdentity') or gate.get('preflightSha256') != _sha(out/'preflight.json'):
        raise ValueError('blocked_integrity: development gate/preflight identity mismatch')
    if gate.get('status') == 'pass':
        dev_ledger_path = out/'development-fit-ledger.json'
        if (gate.get('forecastSha256') != _sha(out/'development-risk-forecasts.parquet')
                or gate.get('ledgerSha256') != _sha(dev_ledger_path)):
            raise ValueError('blocked_integrity: test permission is not bound to development forecast/ledger')

    # Source identities are sampled before the read-only joins; originals must remain byte-identical.
    originals = price_diag.verify_inputs()
    original_hashes = {k: v['sha256'] for k, v in originals.items()}
    source_manifest_path = out / 'source-manifest.json'
    source_manifest_hashes = {}
    if not source_manifest_path.is_file():
        raise ValueError('blocked_integrity: required source-manifest.json is absent')
    manifest = _json(source_manifest_path)
    for key, item in manifest.items():
        if key == 'V6:protocolExpectedSha256' and isinstance(item, str) and len(item) == 64:
            continue
        if not isinstance(item, dict) or not item.get('path') or not item.get('sha256'):
            raise ValueError(f'blocked_integrity: source manifest entry lacks path/hash: {key}')
        p = Path(item['path'])
        if not p.is_absolute(): p = ROOT / p
        if not p.is_file():
            raise ValueError(f'blocked_integrity: source manifest path missing: {key} {p}')
        digest = _sha(p)
        if digest != item['sha256']:
            raise ValueError(f'blocked_integrity: source manifest mismatch before finalization: {key}')
        source_manifest_hashes[key] = digest
    preflight_path = out/'preflight.json'
    audit_path = out/'input-and-feature-audit.json'
    if not preflight_path.is_file() or not audit_path.is_file():
        raise ValueError('blocked_integrity: required preflight/feature audit missing')
    preflight = _json(preflight_path)
    audit = _json(audit_path)
    if preflight.get('status') != 'pass':
        raise ValueError('blocked_integrity: preflight did not pass')
    preflight = v6.verify_preflight(out)
    price_rows = _price_rows()
    joined_by_split, coverage, groups = {}, {}, []
    if risk['develop'] is not None:
        joined_by_split['develop'], coverage['develop'] = _risk_price_join(risk['develop'], price_rows['develop'])
    if risk['test'] is not None:
        joined_by_split['test'], coverage['test'] = _risk_price_join(risk['test'], price_rows['test'])

    quantiles = risk_probability_cuts(risk['develop'])
    if quantiles['status'] == 'available':
        q33, q67 = quantiles['q33'], quantiles['q67']
        for split, joined in joined_by_split.items():
            groups.extend(_price_error_groups(joined, q33, q67, split))
    else:
        q33 = q67 = None

    cal = {}
    # score JSON intentionally keeps aggregate summaries; reconstruct labels from the already frozen catalog.
    # run_score output does not persist its joined rows, so rebuild only deterministic labels for diagnostics.
    catalog_path = out / 'v5-feature-catalog.parquet'
    catalog = _read_frame(catalog_path).to_dict('records')
    catalog_by = {(r['crop'], str(r['targetStart']), str(r['targetEnd'])): r for r in catalog}
    for split in ('develop','test'):
        if risk[split] is None: continue
        labelled=[]
        for r in risk[split]:
            c = catalog_by.get((r['crop'],str(r['targetStart']),str(r['targetEnd'])))
            actual = c.get('actual') if c else None
            event = v6.label_event(actual,r.get('p_ref'))
            labelled.append({**r,'event':event})
        cal[split] = _probability_diagnostics(labelled)
        score = dev_score if split == 'develop' else test_score
        expected_scored = sum(r.get('event') is not None and _valid(r.get('prob_RB')) and _valid(r.get('prob_RV')) for r in labelled)
        if int(score.get('scoredRows', -1)) != expected_scored:
            raise ValueError(f'blocked_integrity: {split} score paired probability support differs from saved rows')
        if any(_valid(r.get('prob_RB')) != _valid(r.get('prob_RV')) for r in labelled):
            raise ValueError(f'blocked_integrity: {split} RB/RV probability coverage is not paired')

    fixed = _fixed_feature_descriptive(out) if gate.get('status') != 'pass' else None
    test_fits = 0
    ledger_checks = {}
    for split in ('develop','test'):
        stage_name = 'development' if split == 'develop' else 'test'
        lp = out / f'{stage_name}-fit-ledger.json'
        if lp.is_file():
            ledger = _json(lp); n = int(ledger.get('attempts', -1)); ledger_checks[split] = {'attempts': n, 'cumulativeFitAttempts': ledger.get('cumulativeFitAttempts', n), 'planned': ledger.get('planned'), 'complete': ledger.get('complete')}
            if split == 'test': test_fits = n
            catalog_rows, _, _ = v6._load_catalog(out)
            audited = 0
            for entry in ledger.get('entries', []):
                if entry.get('status') not in ('ok', 'insufficient_class_support'):
                    continue
                origin = dt.date.fromisoformat(entry['origin'])
                train = v6.training_rows(catalog_rows, origin)
                v6.assert_training_maturity(train, origin)
                expected_n = entry.get('trainingRows')
                if expected_n is not None and int(expected_n) != len(train):
                    raise ValueError('blocked_integrity: fit ledger training-row count differs from mature catalog')
                audited += 1
            ledger_checks[split]['maturityAuditedFits'] = audited
            if not ledger.get('complete'):
                raise ValueError(f'blocked_integrity: {split} fit ledger is incomplete')
    if 'develop' not in ledger_checks:
        raise ValueError('blocked_integrity: development fit ledger is missing')
    if gate.get('status') == 'pass' and 'test' not in ledger_checks:
        raise ValueError('blocked_integrity: test fit ledger is missing after gate pass')
    attempts = sum(v['attempts'] for v in ledger_checks.values())
    cumulative = ledger_checks.get('test', ledger_checks['develop'])['cumulativeFitAttempts']
    if attempts != cumulative:
        raise ValueError('blocked_integrity: cumulative fit-attempt ledger does not equal stage totals')
    if attempts > v6.MAX_ATTEMPTS: raise ValueError('blocked_budget: V6 attempt ledger exceeds 450')

    final_status = _final_status(gate, test_score)
    facts = {'finalStatus': final_status, 'developmentGate': gate.get('status'),
             'testPeriod': 'exploratory_reused_period' if risk['test'] is not None else None,
             'developmentRiskScores': dev_score, 'testRiskScores': test_score,
             'priceJoinCoverage': coverage, 'riskProbabilityQuantiles': quantiles,
             'testOverallEqualCropLossDelta': _overall_test_deltas(test_score) if test_score else None,
             'priceErrorByRiskGroup': groups, 'probabilityLossAndCalibration': cal,
             'legacyFixedFeatureDescriptive': fixed, 'fitLedger': ledger_checks,
             'fitAttemptCount': attempts, 'fitAttemptLimit': v6.MAX_ATTEMPTS,
             'testFitAttemptCount': test_fits,
             'testArtifactsPresent': {'forecasts': risk['test'] is not None, 'scores': test_score is not None},
             'statisticalAssessment': {'development': dev_score.get('pairedDmHac'), 'test': test_score.get('pairedDmHac') if test_score else None,
                                       'bootstrapDevelopment': dev_score.get('blockBootstrap'), 'bootstrapTest': test_score.get('blockBootstrap') if test_score else None},
             'limitations': ['retrospective revised market proxy; exploratory reused test period',
                             'price-error association is descriptive and does not establish MAPE improvement caused by risk scores']}
    checks = {'status': 'pass', 'finalStatus': final_status, 'fitAttemptCount': attempts,
              'fitAttemptLimit': v6.MAX_ATTEMPTS, 'attemptBudgetPass': attempts <= v6.MAX_ATTEMPTS,
              'gateTestArtifactRulePass': (gate.get('status') == 'pass' or (risk['test'] is None and test_score is None and not test_ledger_path.exists())),
              'riskPriceJoin': coverage, 'inheritedPreflight': preflight,
              'featureAudit': audit,
              'fitMaturityAudit': {k: v.get('maturityAuditedFits', 0) for k, v in ledger_checks.items()},
              'sourceHashesBefore': original_hashes}
    _write_json(out/'final-facts.json', facts)
    _write_json(out/'checks.json', checks)
    lines = ['# V6 품목·가격오차 및 확률 진단 초안', '', f"최종 판정: `{final_status}`.", '',
             '위험확률의 Brier·log loss·PR-AUC, 연도·품목 지원, 보정 구간, paired DM/Holm 및 블록 부트스트랩은 final-facts.json에 보존했다.',
             (f"2022 개발 RV 확률의 고정 삼분위 경계는 q33={q33:.10g}, q67={q67:.10g}이며 이후 연도에도 그대로 적용했다."
              if q33 is not None else '2022 개발 RV 확률이 없어 확률기반 위험집단은 산출하지 않았다.'),
             '기존 저장 G-B/P의 AE·APE는 동일 목표순 키의 교집합에서만 요약했다. 위험 신호와 가격오차의 관계는 사후 기술통계이며 인과 효과나 가격예측 개선을 뜻하지 않는다.',
             '', '기존 V5 가격 성적은 변경하지 않았다. V6는 공식 가격·실시간 발행 가능성의 검증이 아니라 수정된 과거 시장 대리자료의 후향 진단이다.', '']
    (out/'loss-and-calibration-diagnostics.md').write_text('\n'.join(lines), encoding='utf-8')
    # Recheck every preexisting source after writes; our new artifacts are outside these inputs.
    after = price_diag.verify_inputs()
    after_hashes = {k: v['sha256'] for k, v in after.items()}
    if original_hashes != after_hashes:
        raise ValueError('blocked_integrity: original V3/V5/source hashes changed during finalization')
    source_manifest_after = {}
    for key, digest in source_manifest_hashes.items():
        item = _json(source_manifest_path)[key]
        p = Path(item['path'])
        if not p.is_absolute(): p = ROOT / p
        source_manifest_after[key] = _sha(p)
    if source_manifest_hashes != source_manifest_after:
        raise ValueError('blocked_integrity: source-manifest inputs changed during finalization')
    checks['sourceHashesAfter'] = after_hashes
    checks['sourceHashesUnchanged'] = True
    checks['sourceManifestHashesBefore'] = source_manifest_hashes
    checks['sourceManifestHashesAfter'] = source_manifest_after
    checks['sourceManifestInputsUnchanged'] = True
    _write_json(out/'checks.json', checks)
    return {'status': final_status, 'files': ['final-facts.json','checks.json','loss-and-calibration-diagnostics.md'],
            'q33': q33, 'q67': q67, 'fitAttemptCount': attempts}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('finalize',))
    parser.add_argument('--out', type=Path, default=OUT)
    args = parser.parse_args(argv)
    print(json.dumps(finalize(args.out), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
