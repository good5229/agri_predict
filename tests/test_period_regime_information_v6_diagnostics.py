import datetime as dt
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from agri_data import period_regime_information_v6_diagnostics as diag
from agri_data import period_proxy_experiment as proxy


def _row(crop='cabbage', year=2022, prob=.3, event=1, ape=.2, ae=20):
    return {'crop': crop, 'year': year, 'targetStart': f'{year}-01-01', 'targetEnd': f'{year}-01-10',
            'prob_RB': .2, 'prob_RV': prob, 'event': event, 'actual': 120., 'p_ref': 100.,
            'G-B_ape': ape, 'P_ape': ape * 1.2, 'G-B_ae': ae, 'P_ae': ae * 1.2}


def test_price_error_group_uses_fixed_probability_boundaries_and_reports_support():
    rows = [_row(prob=.1), _row(prob=.5), _row(prob=.8)]
    grouped = diag._price_error_groups(rows, .2, .6, 'develop')
    by = {r['riskGroup']: r for r in grouped if r['crop'] == 'cabbage'}
    assert by['low']['n'] == 1
    assert by['middle']['n'] == 1
    assert by['high']['n'] == 1
    assert by['high']['riskRate'] == 1.0
    assert by['high']['G-B_APE_mean'] == .2
    assert by['high']['uniqueTargets'] == 1


def test_probability_diagnostics_reports_direction_and_brier_contribution():
    result = diag._probability_diagnostics([_row(prob=.7, event=1), _row(prob=.2, event=0)])
    positive = next(x for x in result['RV'] if x['event'] == 1)
    negative = next(x for x in result['RV'] if x['event'] == 0)
    assert positive['probabilityMinusEvent'] == pytest.approx(-.3)
    assert positive['underpredictionN'] == 1
    assert positive['brierContributionSum'] == pytest.approx(.09)
    assert negative['overpredictionN'] == 1


def test_final_status_requires_each_test_year_to_improve_and_support():
    cells = {}
    for year in (2024, 2025, 2026):
        for crop in ('cabbage', 'radish'):
            cells[f'{year}:{crop}'] = {'n': 30, 'positive': 8, 'negative': 22,
                'RB': {'brier': .2, 'logLoss': .6}, 'RV': {'brier': .19, 'logLoss': .61}}
    result = diag._final_status({'status': 'pass'}, {'byYearCrop': cells})
    assert result == 'risk_signal_supported_exploratory'
    cells['2025:radish']['RV']['brier'] = .2
    cells['2025:cabbage']['RV']['brier'] = .2
    assert diag._final_status({'status': 'pass'}, {'byYearCrop': cells}) == 'risk_signal_inconsistent'
    cells['2024:cabbage']['positive'] = 4
    assert diag._final_status({'status': 'pass'}, {'byYearCrop': cells}) == 'inconclusive_support'


def test_final_status_also_checks_whole_test_equal_crop_delta_with_unequal_support():
    cells = {}
    specifications = {
        2024: {'cabbage': (100, .20), 'radish': (24, -.21)},
        2025: {'cabbage': (24, -.21), 'radish': (100, .20)},
        2026: {'cabbage': (24, -.21), 'radish': (100, .20)},
    }
    for year, crop_spec in specifications.items():
        for crop, (n, delta) in crop_spec.items():
            cells[f'{year}:{crop}'] = {'n': n, 'positive': 8, 'negative': n-8,
                'RB': {'brier': .3, 'logLoss': .6}, 'RV': {'brier': .3+delta, 'logLoss': .6}}
    assert diag._final_status({'status': 'pass'}, {'byYearCrop': cells}) == 'risk_signal_inconsistent'


def test_risk_probability_cuts_use_2022_only_and_handle_missing_support():
    rows = [{'year': 2022, 'prob_RV': .1}, {'year': 2022, 'prob_RV': .2},
            {'year': 2022, 'prob_RV': .3}, {'year': 2023, 'prob_RV': .99}]
    cuts = diag.risk_probability_cuts(rows)
    assert cuts['n'] == 3
    assert cuts['q33'] == pytest.approx(.1666666667)
    assert cuts['q67'] == pytest.approx(.2333333333)
    missing = diag.risk_probability_cuts([{'year': 2022, 'prob_RV': None}])
    assert missing['status'] == 'unavailable'
    assert '2022' in missing['reason']


def test_failed_gate_keeps_final_status_and_has_no_need_for_test_scores():
    assert diag._final_status({'status': 'year_inconsistent'}, None) == 'year_inconsistent'


@pytest.mark.parametrize('stale_file', ['test-risk-scores.json', 'test-fit-ledger.json'])
def test_finalizer_rejects_any_test_stage_artifact_after_failed_gate(monkeypatch, tmp_path, stale_file):
    out = tmp_path/'v6'; out.mkdir()
    (out/'development-gate.json').write_text(json.dumps({'status': 'year_inconsistent', 'testFitsAllowed': False}))
    (out/'development-risk-scores.json').write_text('{}')
    (out/stale_file).write_text('{}')
    monkeypatch.setattr(diag, '_risk_rows', lambda *_: None)
    with pytest.raises(ValueError, match='test fit/score artifacts exist after failed'):
        diag.finalize(out)


def test_synthetic_stage_artifacts_finalize_with_bound_keys_and_gate(monkeypatch, tmp_path):
    out = tmp_path / 'out'
    out.mkdir()
    identity = {'run': 'synthetic'}
    pre = {'status': 'pass', 'runIdentity': identity}
    (out/'preflight.json').write_text(json.dumps(pre))
    (out/'input-and-feature-audit.json').write_text(json.dumps({'savedV5FeatureComparisons': 1}))
    marker = tmp_path/'source.bin'
    marker.write_bytes(b'original source')
    digest = hashlib.sha256(marker.read_bytes()).hexdigest()
    (out/'source-manifest.json').write_text(json.dumps({'synthetic': {'path': str(marker), 'sha256': digest}}))

    def schedule(year, periods):
        out_rows = []
        for start, end in periods:
            out_rows.append({'origin': end + dt.timedelta(days=2), 'target': (start, end)})
        return out_rows

    dev_periods = [(dt.date(2022, 1, 1), dt.date(2022, 1, 10)),
                   (dt.date(2022, 1, 11), dt.date(2022, 1, 20)),
                   (dt.date(2022, 1, 21), dt.date(2022, 1, 31))]
    test_periods = []
    for year in (2024, 2025, 2026):
        month = 1
        for start, end in __import__('agri_data.period_proxy_experiment', fromlist=['dekads']).dekads(
                dt.date(year, 1, 1), dt.date(year, 12, 31) if year < 2026 else dt.date(year, 9, 30))[:24]:
            test_periods.append((start, end))
    schedules = {'develop': schedule(2022, dev_periods), 'test': schedule(2024, test_periods)}
    monkeypatch.setattr(diag.v6, 'build_schedule', lambda split: schedules[split])
    monkeypatch.setattr(diag.v6, 'verify_preflight', lambda _: pre)
    monkeypatch.setattr(diag.price_diag, 'verify_inputs', lambda: {})

    all_catalog, price_by_split, risk_by_split = [], {'develop': [], 'test': []}, {'develop': [], 'test': []}
    for split, sched in schedules.items():
        for period_ix, item in enumerate(sched):
            start, end = item['target']
            for crop_ix, crop in enumerate(('cabbage', 'radish')):
                event = (period_ix + crop_ix) % 2 if split == 'test' else period_ix % 2
                prob = (.1 + .1 * period_ix) if split == 'develop' else (.65 if event else .1)
                row = {'crop': crop, 'horizon': 3, 'origin': item['origin'].isoformat(),
                       'targetStart': start.isoformat(), 'targetEnd': end.isoformat(), 'p_ref': 100.,
                       'featureMeta': {'source': 'synthetic'}, 'year': start.year, 'split': split, 'rowStatus': 'ready',
                       'prob_RB': .4, 'prob_RV': prob, 'status_RB': 'ok', 'status_RV': 'ok'}
                risk_by_split[split].append(row)
                actual = 120. if event else 100.
                all_catalog.append({'crop': crop, 'targetStart': start.isoformat(), 'targetEnd': end.isoformat(),
                                    'actual': actual, 'p_ref': 100.,
                                    'features': {'price_sd_28': .15, 'price_available_28': 1}})
                price_by_split[split].append({**row, 'actual': actual, 'G-B_pred': 110., 'P_pred': 115.,
                    'G-B_ae': abs(110.-actual), 'G-B_ape': abs(110.-actual)/actual,
                    'P_ae': abs(115.-actual), 'P_ape': abs(115.-actual)/actual})
    catalog_path = out/'v5-feature-catalog.parquet'
    pd.DataFrame(all_catalog).to_parquet(catalog_path, index=False)
    monkeypatch.setattr(diag.v6, '_load_catalog', lambda _: (all_catalog, {}, {}))
    monkeypatch.setattr(diag, '_price_rows', lambda: price_by_split)

    stage_digests, ledger_digests = {}, {}
    for split, label in (('develop', 'development'), ('test', 'test')):
        forecast_path = out/f'{label}-risk-forecasts.parquet'
        pd.DataFrame(risk_by_split[split]).to_parquet(forecast_path, index=False)
        forecast_digest = diag._sha(forecast_path)
        attempts = 2 if split == 'develop' else 4
        cumulative = 2 if split == 'develop' else 6
        ledger = {'attempts': attempts, 'cumulativeFitAttempts': cumulative, 'planned': attempts,
                  'complete': True, 'runIdentity': identity, 'preflightSha256': diag._sha(out/'preflight.json'),
                  'forecastSha256': forecast_digest, 'entries': []}
        ledger_path = out/f'{label}-fit-ledger.json'
        ledger_path.write_text(json.dumps(ledger))
        ledger_digest = diag._sha(ledger_path)
        ledger_digests[label] = ledger_digest
        scored = sum(r['prob_RB'] is not None and r['prob_RV'] is not None for r in risk_by_split[split])
        by_year = {}
        if split == 'test':
            for year in (2024, 2025, 2026):
                for crop in ('cabbage','radish'):
                    subset = [r for r in risk_by_split[split] if r['year'] == year and r['crop'] == crop]
                    events = [int((period_ix + 0) % 2 == 0) for period_ix, _ in enumerate(subset)]
                    positives = sum(events); n = len(subset)
                    by_year[f'{year}:{crop}'] = {'n': n, 'positive': 12, 'negative': n-12,
                        'RB': {'brier': .25, 'logLoss': .69}, 'RV': {'brier': .20, 'logLoss': .68}}
        score = {'split': split, 'forecastSha256': forecast_digest, 'issuedRows': len(risk_by_split[split]),
                 'scoredRows': scored, 'byYearCrop': by_year, 'runIdentity': identity,
                 'preflightSha256': diag._sha(out/'preflight.json'), 'ledgerSha256': ledger_digest,
                 'pairedDmHac': {}, 'blockBootstrap': {}}
        score_path = out/f'{label}-risk-scores.json'
        score_path.write_text(json.dumps(score))
        stage_digests[label] = diag._sha(score_path)
    (out/'development-gate.json').write_text(json.dumps({'status': 'pass', 'testFitsAllowed': True,
        'scoreSha256': stage_digests['development'], 'runIdentity': identity,
        'preflightSha256': diag._sha(out/'preflight.json'),
        'forecastSha256': diag._sha(out/'development-risk-forecasts.parquet'),
        'ledgerSha256': ledger_digests['development']}))

    result = diag.finalize(out)
    assert result['status'] == 'risk_signal_supported_exploratory'
    facts = json.loads((out/'final-facts.json').read_text())
    assert facts['riskProbabilityQuantiles']['n'] == 6
    assert facts['priceJoinCoverage']['test']['riskProbabilityRowsByMethod']['paired'] == 144
    assert json.loads((out/'checks.json').read_text())['sourceManifestInputsUnchanged'] is True


def test_full_synthetic_develop_gate_test_finalize_pipeline(monkeypatch, tmp_path):
    """Run both real stage orchestrators over synthetic data and finalize their artifacts."""
    out = tmp_path/'period-regime-information-v6-20261016'
    out.mkdir()
    identity = {'runId': 'synthetic-v6'}
    preflight = {'status': 'pass', 'runIdentity': identity}
    (out/'preflight.json').write_text(json.dumps(preflight))
    (out/'input-and-feature-audit.json').write_text(json.dumps({'savedV5FeatureComparisons': 1}))
    marker = tmp_path/'source.bin'; marker.write_bytes(b'synthetic immutable source')
    marker_sha = hashlib.sha256(marker.read_bytes()).hexdigest()
    (out/'source-manifest.json').write_text(json.dumps({'testSource': {'path': str(marker), 'sha256': marker_sha}}))
    monkeypatch.setattr(diag.v6, 'verify_preflight', lambda _: preflight)
    monkeypatch.setattr(diag.price_diag, 'verify_inputs', lambda: {})

    def sched(start, end):
        return proxy.planned_origins(start, end, 3)
    schedules = {
        'develop': sched(dt.date(2022, 1, 1), dt.date(2023, 12, 31)),
        'test': sched(dt.date(2024, 1, 1), dt.date(2026, 9, 30)),
    }
    monkeypatch.setattr(diag.v6, 'build_schedule', lambda split: schedules[split])

    catalog = []
    targets = {crop: {} for crop in diag.v6.CROPS}
    all_periods = proxy.dekads(dt.date(2014, 1, 1), dt.date(2026, 9, 30))
    for crop in diag.v6.CROPS:
        for start, end in all_periods:
            event = int(diag.v6.period_index(start) % 2 == 0)
            actual = 120. if event else 100.
            targets[crop][(start, end)] = {'actual': actual}
            feature = {name: .1 for name in diag.v6.FEATURES_RB}
            feature['log_p_ref'] = float(np.log(100.))
            feature.update({'price_sd_28': .15, 'price_available_28': 1, 'price_n_28': 8,
                            'reference_available': 1})
            catalog.append({'crop': crop, 'horizon': 3, 'target': (start, end),
                            'targetStart': start.isoformat(), 'targetEnd': end,
                            'origin': end + dt.timedelta(days=2), 'actual': actual, 'p_ref': 100.,
                            'features': feature,
                            'featureMeta': {'inputCutoff': (end).isoformat(), 'referencePrice': 100.}})
    monkeypatch.setattr(diag.v6, '_load_catalog', lambda _: (catalog, targets, {'sha256': 'synthetic'}))
    pd.DataFrame(catalog).to_parquet(out/'v5-feature-catalog.parquet', index=False)

    def fake_fit(train, tests, method):
        probs = []
        for row in tests:
            event = diag.v6.period_index(dt.date.fromisoformat(row['targetStart'])) % 2 == 0
            probs.append((.8 if event else .1) if method == 'RV' else .4)
        return probs, {'status': 'ok', 'trainingRows': len(train),
                       'trainingByCrop': {c: sum(x['crop'] == c for x in train) for c in diag.v6.CROPS},
                       'trainingPositive': sum(x['event'] for x in train),
                       'trainingNegative': sum(1-x['event'] for x in train)}
    monkeypatch.setattr(diag.v6, 'fit_probabilities', fake_fit)
    monkeypatch.setattr(diag.v6, 'paired_block_bootstrap', lambda *a, **k: {'status': 'synthetic'})
    monkeypatch.setattr(diag.v6, 'dm_families', lambda *a, **k: {'synthetic': True})
    monkeypatch.setattr(diag, '_price_rows', lambda: {
        split: [{**r, 'actual': targets[r['crop']][(dt.date.fromisoformat(r['targetStart']), dt.date.fromisoformat(r['targetEnd']))]['actual'],
                 'p_ref': 100., 'G-B_pred': 110., 'P_pred': 115.,
                 'G-B_ae': abs(110.-targets[r['crop']][(dt.date.fromisoformat(r['targetStart']), dt.date.fromisoformat(r['targetEnd']))]['actual']),
                 'P_ae': abs(115.-targets[r['crop']][(dt.date.fromisoformat(r['targetStart']), dt.date.fromisoformat(r['targetEnd']))]['actual']),
                 'G-B_ape': abs(110.-targets[r['crop']][(dt.date.fromisoformat(r['targetStart']), dt.date.fromisoformat(r['targetEnd']))]['actual']) / targets[r['crop']][(dt.date.fromisoformat(r['targetStart']), dt.date.fromisoformat(r['targetEnd']))]['actual'],
                 'P_ape': abs(115.-targets[r['crop']][(dt.date.fromisoformat(r['targetStart']), dt.date.fromisoformat(r['targetEnd']))]['actual']) / targets[r['crop']][(dt.date.fromisoformat(r['targetStart']), dt.date.fromisoformat(r['targetEnd']))]['actual']}
                for r in pd.read_parquet(out/f"{'development' if split == 'develop' else 'test'}-risk-forecasts.parquet").to_dict('records')]
        for split in ('develop', 'test')})

    assert diag.v6.run_forecast('develop', out)['attemptsThisStage'] == len(schedules['develop']) * 2
    diag.v6.run_score('develop', out)
    assert diag.v6.run_gate(out)['status'] == 'pass'
    assert diag.v6.run_forecast('test', out)['attemptsThisStage'] == len(schedules['test']) * 2
    diag.v6.run_score('test', out)
    result = diag.finalize(out)
    assert result['fitAttemptCount'] == 342
    assert result['status'] == 'risk_signal_supported_exploratory'
    facts = json.loads((out/'final-facts.json').read_text())
    assert facts['riskProbabilityQuantiles']['source'].startswith('available 2022')
    assert facts['priceJoinCoverage']['test']['riskProbabilityRowsByMethod']['paired'] == 198
