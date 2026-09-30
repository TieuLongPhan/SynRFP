#!/usr/bin/env python3
"""Create one MLP MCC table spanning Track A datasets and EHReact activity."""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synrfp_exp.config import expand_fingerprints, iter_json, load_config, results_dir


def main():
    cfg = load_config(None)
    fps = expand_fingerprints(cfg, cfg['benchmark']['fingerprints'])
    # This supervised TPL-1K encoder is intentionally an extra, TPL-only
    # transfer reference.  It receives the identical 5x5 outer-CV protocol
    # here, but is not a label-free representation baseline.
    tpl_transfer_fp = 'rxnfp_tpl'
    tpl_datasets = ['tpl_ub', 'tpl_b']
    results = results_dir('03_classification')
    runs = pd.DataFrame(iter_json(results / 'benchmark'))
    mlp = runs[(runs['model'] == 'mlp') & (runs['scheme'] == 'cv') & runs['fingerprint'].isin(fps)].copy()
    counts = mlp.groupby(['dataset', 'fingerprint']).size()
    expected = len(cfg['benchmark']['schemes']['cv'])
    missing = counts[counts != expected]
    if not missing.empty:
        raise RuntimeError(f'incomplete MLP CV cells (expected {expected}): {missing.to_dict()}')
    standard = (mlp.groupby(['dataset', 'fingerprint'])['mcc'].agg(['mean', 'std', 'count'])
                .reset_index().assign(protocol='Track A: 25 scaffold-grouped outer folds'))
    transfer = runs[(runs['model'] == 'mlp') & (runs['scheme'] == 'cv') &
                    (runs['fingerprint'] == tpl_transfer_fp) &
                    (runs['dataset'].isin(tpl_datasets))].copy()
    transfer_counts = transfer.groupby(['dataset', 'fingerprint']).size()
    missing_transfer = {dataset: int(transfer_counts.get((dataset, tpl_transfer_fp), 0))
                        for dataset in tpl_datasets
                        if transfer_counts.get((dataset, tpl_transfer_fp), 0) != expected}
    if missing_transfer:
        raise RuntimeError(f'incomplete TPL-1K transfer MLP CV cells (expected {expected}): {missing_transfer}')
    transfer = (transfer.groupby(['dataset', 'fingerprint'])['mcc'].agg(['mean', 'std', 'count'])
                .reset_index().assign(protocol='Track A outer splits; TPL-1K supervised transfer reference'))
    eh_path = ROOT / '10_enzyme_activity' / 'results' / 'runs.csv'
    eh = pd.read_csv(eh_path)
    eh = eh[(eh['model'] == 'mlp') & eh['fingerprint'].isin(fps)].copy()
    expected_eh = 15  # three eligible assays x five repeats
    ecounts = eh.groupby('fingerprint').size()
    absent = [fp for fp in fps if ecounts.get(fp, 0) != expected_eh]
    if absent:
        raise RuntimeError(f'incomplete EHReact MLP cells: {absent}')
    activity = (eh.groupby('fingerprint')['mcc'].agg(['mean', 'std', 'count']).reset_index()
                .assign(dataset='ehreact_activity', protocol='EHReact: mean over 3 assays x 5 grouped-CV repeats'))
    long = pd.concat([standard, transfer,
                      activity[['dataset','fingerprint','mean','std','count','protocol']]], ignore_index=True)
    order = list(cfg['datasets']) + ['ehreact_activity']
    long['dataset'] = pd.Categorical(long['dataset'], order, ordered=True)
    long = long.sort_values(['dataset','fingerprint']).reset_index(drop=True)
    out = results / 'mlp_comparison_all'
    out.mkdir(exist_ok=True)
    long.to_csv(out / 'mlp_mcc_long.csv', index=False)
    display = long.assign(cell=long.apply(lambda r: f"{r['mean']:.3f} +/- {r['std']:.3f}", axis=1))
    columns = fps + [tpl_transfer_fp]
    wide = display.pivot(index='dataset', columns='fingerprint', values='cell').reindex(columns=columns)
    wide.to_csv(out / 'mlp_mcc_table.csv')
    lines = ['# MLP MCC comparison — all datasets', '',
             'The 14 reaction-classification rows are means over 25 scaffold-grouped outer CV folds. '
             'EHReact is the equal-weight mean over its three eligible assays and five grouped-CV repeats (15 assay-repeat scores). '
             'It is a separate binary-activity case study, so it is not combined with the multiclass benchmark average. '
             '`rxnfp_tpl` is the TPL-1K supervised checkpoint; it is reported only for TPL and is not a fair label-free baseline.', '',
             '| Dataset | ' + ' | '.join(columns) + ' |',
             '|---|' + '|'.join(['---:'] * len(columns)) + '|']
    for dataset, row in wide.iterrows():
        lines.append('| ' + str(dataset) + ' | ' + ' | '.join(row.fillna('--')) + ' |')
    lines += ['', 'Mech31k is represented by `mech_1` and `mech_2`.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n')
    print(wide.to_string())

if __name__ == '__main__': main()
