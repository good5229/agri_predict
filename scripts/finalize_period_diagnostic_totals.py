#!/usr/bin/env python3
"""Create supplemental all-period diagnostics from already saved predictions.

This script only reuses the reviewed, read-only diagnostic join and summary
functions. It does not fit/load models, predict, or conduct inferential tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agri_data import period_crop_profile_diagnostics as diagnostics
from agri_data import period_proxy_experiment as proxy


def manifest_digest(manifest: dict) -> str:
    raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_supplemental_rows(rows: list[dict]) -> list[dict]:
    """Return period-wide cell diagnostics; group thresholds are per crop/cell."""
    if len(rows) != 780:
        raise ValueError(f"blocked_integrity: paired test support {len(rows)} != 780")
    expected = {(c, h): (98 if h == 2 else 97) for c in proxy.CROPS for h in proxy.HORIZONS}
    observed = {(c, h): sum(r['crop'] == c and r['horizon'] == h for r in rows)
                for c in proxy.CROPS for h in proxy.HORIZONS}
    if observed != expected:
        raise ValueError(f"blocked_integrity: paired test cell support differs: {observed}")
    if len({diagnostics.period_key(r) for r in rows}) != 780:
        raise ValueError("blocked_integrity: duplicate paired forecast keys")
    if len({(r['targetStart'], r['targetEnd']) for r in rows}) != 98:
        raise ValueError("blocked_integrity: unique target-dekad support differs from 98")
    if len({(r['crop'], r['targetStart'], r['targetEnd']) for r in rows}) != 392:
        raise ValueError("blocked_integrity: unique crop-target support differs from 392")

    # These copied rows change only the grouping label. error_table and
    # tail_table still derive thresholds separately within each crop × horizon.
    all_period = [{**r, 'year': 'ALL'} for r in rows]
    regime = diagnostics.error_table(all_period, split='test')
    regimes = [r for r in regime if r['group'] in {
        'realized_abs_change_lt20', 'realized_abs_change_ge20',
        'actual_price_bottom20', 'actual_price_top20',
    }]
    ztails = diagnostics.tail_table(all_period, split='test')
    compression = diagnostics.compression_table(all_period, by_year=False, split='test')

    expected_counts = (('realized_abs_change_lt20', 8), ('realized_abs_change_ge20', 8),
                       ('actual_price_bottom20', 8), ('actual_price_top20', 8))
    for label, count in expected_counts:
        if sum(r['group'] == label for r in regimes) != count:
            raise ValueError(f"blocked_integrity: all-period group support missing {label}")
    if len(ztails) != 16 or len(compression) != 8:
        raise ValueError(f"blocked_integrity: all-period z-tail/compression rows {len(ztails)}/{len(compression)} != 16/8")

    out = []
    for kind, part in (('realized_jump_or_nonjump', regimes),
                       ('actual_price_quantile', regimes),
                       ('actual_z_quantile_tail', ztails),
                       ('prediction_compression', compression)):
        if kind == 'realized_jump_or_nonjump':
            part = [r for r in part if r['group'].startswith('realized_abs_change_')]
        elif kind == 'actual_price_quantile':
            part = [r for r in part if r['group'].startswith('actual_price_')]
        for row in part:
            out.append({**row, 'source_diagnostic_type': row['diagnostic_type'],
                        'diagnostic_type': kind, 'scope': 'all_test_period'})
    if (sum(r['diagnostic_type'] == 'realized_jump_or_nonjump' for r in out) != 16
            or sum(r['diagnostic_type'] == 'actual_price_quantile' for r in out) != 16
            or sum(r['diagnostic_type'] == 'actual_z_quantile_tail' for r in out) != 16
            or sum(r['diagnostic_type'] == 'prediction_compression' for r in out) != 8):
        raise ValueError("blocked_integrity: supplemental all-period row totals differ from 16/16/16/8")
    return out


def run(out_dir: Path | str = diagnostics.OUT) -> dict:
    out = Path(out_dir)
    csv_path = out / 'tables' / 'full_period_error_diagnostics.csv'
    json_path = out / 'full_period_error_diagnostics.json'
    if csv_path.exists() or json_path.exists():
        raise FileExistsError("supplemental output exists; refusing to overwrite")

    before = diagnostics.verify_inputs()
    rows = diagnostics.build_metric_rows(require_planned_grid=True)
    supplemental = build_supplemental_rows(rows)
    after = diagnostics.verify_inputs()
    if before != after:
        raise ValueError("blocked_integrity: pinned inputs changed during supplemental diagnostics")

    source_names = ('prices.json', 'forecasts.parquet', 'score-report.json',
                    'V3:forecasts.parquet', 'V3:forecasts_test.parquet', 'V3:selection.json')
    source_hashes = {name: before[name]['sha256'] for name in source_names}
    source_hashes_after = {name: after[name]['sha256'] for name in source_names}
    cell_support = {f'{c}:{h}': sum(r['crop'] == c and r['horizon'] == h for r in rows)
                    for c in proxy.CROPS for h in proxy.HORIZONS}
    check = {
        'status': 'verified_reuse',
        'scope': 'saved test forecasts only; all-period descriptive groups within each crop × horizon cell',
        'paired_forecasts': len(rows),
        'per_cell_support': cell_support,
        'unique_forecast_keys': len({diagnostics.period_key(r) for r in rows}),
        'unique_target_dekads': len({(r['targetStart'], r['targetEnd']) for r in rows}),
        'unique_crop_target_dekads': len({(r['crop'], r['targetStart'], r['targetEnd']) for r in rows}),
        'row_counts': {kind: sum(r['diagnostic_type'] == kind for r in supplemental) for kind in sorted({r['diagnostic_type'] for r in supplemental})},
        'model_fits': 0,
        'model_loads': 0,
        'new_predictions': 0,
        'new_statistical_tests': 0,
        'external_requests': 0,
        'input_manifest_sha256_before': manifest_digest(before),
        'input_manifest_sha256_after': manifest_digest(after),
        'source_sha256_before': source_hashes,
        'source_sha256_after': source_hashes_after,
        'thresholds': {
            'jump': 'within each crop × horizon: abs(actual / p_ref - 1) < 20% vs >= 20%',
            'actual_price': 'within each crop × horizon: bottom/top empirical quintile of actual price',
            'actual_z': 'within each crop × horizon: bottom/top empirical quintile of log(actual / p_ref)',
            'compression': 'all paired test rows within each crop × horizon; population standard deviations',
        },
    }
    diagnostics.write_csv(csv_path, supplemental)
    json_path.write_text(json.dumps(check, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return check


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=str(diagnostics.OUT))
    args = parser.parse_args()
    print(json.dumps(run(args.out), ensure_ascii=False, indent=2))
