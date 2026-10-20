"""Independent task reports from completed factorials; no fitting or selection.

Preserve the sealed combined analysis. Publish each task after its own audited
completion, with all crossed interaction orders and descriptive paired SDs.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import analysis as A

PRIMARY = {'classification': 'mcc', 'yield': 'r2', 'clustering': 'ami'}
CANDIDATE = 'synrfp-wl-r3-b2048-parity-roles'
SCHEMA = 'synrfp-task-factorial-analysis-v1'


def source_seal(root):
    value = {'schema': SCHEMA, 'source_sha256': A.sha(__file__),
             'borrowed_analysis_sha256': A.sha(Path(__file__).with_name('analysis.py')),
             'plan_sha256': A.sha(root / 'plan.json'), 'orders': [1, 2, 3, 4],
             'configuration_selection': None,
             'uncertainty': 'Descriptive SD of correlated repeats; no p-values or confidence intervals'}
    path = root / 'per_task_analysis_execution_seal.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError('Changed task-specific analysis source or plan')
    if not path.exists():
        A.atomic_json(path, value)
    return value


def full_effects(frame, primary, metrics):
    """Complete-grid marginal reference contrasts, including orders3 and4."""
    rows, refusals = [], []
    wanted = set(itertools.product(*(A.LEVELS[f] for f in A.FACTORS)))
    for (dataset, seed), g in frame[frame.fingerprint.isin(primary)].groupby(['dataset', 'seed']):
        if len(g) != 72 or g.fingerprint.nunique() != 72 or set(map(tuple, g[A.FACTORS].to_numpy())) != wanted:
            raise ValueError('Incomplete or duplicate factorial crossing')
        for metric in metrics:
            if not np.isfinite(g[metric].to_numpy(dtype=float)).all():
                refusals.append({'dataset': dataset, 'seed': int(seed), 'metric': metric,
                                 'reason': 'Undefined metric prevents complete-grid contrast'})
                continue
            for order in range(1, 5):
                for axes in itertools.combinations(A.FACTORS, order):
                    alternatives = [[v for v in A.LEVELS[axis] if v != A.REFERENCES[axis]] for axis in axes]
                    for levels in itertools.product(*alternatives):
                        value = 0.0
                        for bits in itertools.product((0, 1), repeat=order):
                            mask = np.ones(len(g), bool)
                            for axis, level, bit in zip(axes, levels, bits):
                                mask &= g[axis].to_numpy() == (level if bit else A.REFERENCES[axis])
                            selected = g.loc[mask, metric]
                            if len(selected) != 72 // int(np.prod([len(A.LEVELS[x]) for x in axes])):
                                raise ValueError('Unbalanced contrast')
                            value += (-1) ** (order - sum(bits)) * float(selected.mean())
                        rows.append({'dataset': dataset, 'seed': int(seed), 'metric': metric, 'order': order,
                                     'effect': ':'.join(axes), 'level': ':'.join(map(str, levels)),
                                     'reference': ':'.join(str(A.REFERENCES[x]) for x in axes), 'contrast': value})
    return pd.DataFrame(rows), refusals


def definition(root, plan, task):
    if task == 'classification':
        return root / 'classification_COMPLETE.json', Path(plan['export']), 'classification.csv'
    if task == 'yield':
        return root / 'yield_canonical_REPORT_COMPLETE.json', Path(plan['yield']['export']), 'yield.csv'
    return root / 'clustering/COMPLETE.json', Path(plan['clustering']['export']), 'cells.csv'


def load_frame(root, plan, task):
    marker, directory, filename = definition(root, plan, task)
    if not marker.exists():
        return None
    receipt = json.loads(marker.read_text())
    if receipt.get('plan_sha256') != A.sha(root / 'plan.json'):
        raise ValueError('Completion has a different plan')
    if task == 'clustering':
        flags = ['assignment_and_center_hashes_verified', 'prediction_replay_exact', 'metric_replay_verified']
        if receipt.get('cells') != A.COUNTS[task] or any(receipt.get(k) is not True for k in flags):
            raise ValueError('Incomplete clustering audit')
    else:
        flag = 'metrics_recomputed_from_canonical_predictions' if task == 'classification' else 'metrics_recomputed'
        if receipt.get('phase') != 'complete' or receipt.get('completed') != A.COUNTS[task] or \
                not receipt.get('artifacts_verified') or receipt.get(flag) is not True:
            raise ValueError('Incomplete canonical model audit')
    hashes = {str(marker): A.sha(marker), str(root / 'plan.json'): A.sha(root / 'plan.json')}
    for name, digest in receipt.get('reports_sha256', {}).items():
        if A.sha(directory / name) != digest:
            raise ValueError('Changed completed report')
        hashes[str(directory / name)] = digest
    path = directory / filename
    hashes[str(path)] = A.sha(path)
    frame = pd.read_csv(path)
    rank, counters = {}, {}
    for split in plan['yield']['splits']:
        d = split['dataset']; rank[(d, split['split'])] = counters.get(d, 0); counters[d] = counters.get(d, 0) + 1
    if task == 'classification':
        frame = frame.rename(columns={'config': 'fingerprint', 'fold': 'seed'})
        expected = {(t['dataset'], t['config'], int(t['fold'])) for t in plan['tasks'][task]}
    elif task == 'yield':
        frame = frame.rename(columns={'config': 'fingerprint'})
        frame['seed'] = [rank[(row.dataset, row.split)] for row in frame.itertuples()]
        expected = {(t['dataset'], t['config'], rank[(t['dataset'], t['split'])]) for t in plan['tasks'][task]}
    else:
        expected = {(t['dataset'], t['fingerprint'], int(t['seed'])) for t in plan['clustering']['tasks']}
    identities = list(zip(frame.dataset, frame.fingerprint, frame.seed))
    if len(frame) != A.COUNTS[task] or len(set(identities)) != len(identities) or set(identities) != expected:
        raise ValueError('Wrong canonical task inventory')
    for factor in A.FACTORS:
        frame[factor] = [plan['configs'][fp].get(factor) for fp in frame.fingerprint]
    for row in frame.itertuples():
        if task == 'classification':
            p = root / 'results/classification' / row.dataset / ('f%d' % row.seed) / (row.fingerprint + '.json')
        elif task == 'yield':
            p = root / 'results/yield' / row.dataset / row.split / (row.fingerprint + '.json')
        else:
            p = root / 'clustering/cells' / row.dataset / row.fingerprint / (str(row.seed) + '.json')
        record = json.loads(p.read_text())
        if record.get('plan_sha256', record.get('identity', {}).get('plan_sha256')) != A.sha(root / 'plan.json'):
            raise ValueError('Changed model/cluster record identity')
        for metric in A.METRICS[task]:
            observed, saved = getattr(row, metric), record[metric]
            if not ((pd.isna(observed) and saved is None) or np.isclose(observed, saved, atol=1e-12, rtol=1e-12, equal_nan=True)):
                raise ValueError('Changed canonical metric: ' + metric)
        for relative, digest in record.get('artifacts_sha256', {}).items():
            artifact = root / relative
            if A.sha(artifact) != digest:
                raise ValueError('Changed retained model artifact')
            hashes[str(artifact)] = digest
        if task == 'clustering':
            for suffix, key in [('.npz', 'assignment_sha256'), ('.model.npz', 'model_sha256')]:
                artifact = p.with_suffix(suffix)
                if A.sha(artifact) != record[key]:
                    raise ValueError('Changed retained clustering artifact')
                hashes[str(artifact)] = record[key]
        hashes[str(p)] = A.sha(p)
    frame['task'] = task
    return frame, hashes


def descriptive_tables(frame, primary, task):
    summaries = []
    for metric in A.METRICS[task]:
        d = frame.groupby(['dataset', 'fingerprint']).agg(
            repeats=(metric, 'size'), mean=(metric, 'mean'), sample_std=(metric, 'std')).reset_index()
        d['metric'] = metric; d['factorial_cell'] = d.fingerprint.isin(primary)
        summaries.append(d)
    summary = pd.concat(summaries, ignore_index=True)
    metric = PRIMARY[task]
    scores = summary[summary.metric.eq(metric) & summary.factorial_cell].copy()
    scores['observed_rank'] = scores.groupby('dataset')['mean'].rank(ascending=False, method='average')
    stability = scores.groupby('fingerprint').agg(equal_panel_mean=('mean', 'mean'),
        mean_observed_rank=('observed_rank', 'mean'), worst_observed_rank=('observed_rank', 'max'),
        minimum_panel_score=('mean', 'min'), maximum_panel_score=('mean', 'max')).reset_index()
    stability = stability.sort_values('mean_observed_rank')
    stability['diagnostic_only_not_config_selection'] = True
    comparisons = []
    selected = frame[frame.fingerprint.eq(CANDIDATE)].set_index(['dataset', 'seed'])
    for fp in sorted(set(frame.fingerprint)):
        other = frame[frame.fingerprint.eq(fp)].set_index(['dataset', 'seed']).reindex(selected.index)
        if other[metric].isna().any():
            continue
        for dataset in sorted(frame.dataset.unique()):
            delta = selected.loc[dataset, metric].to_numpy() - other.loc[dataset, metric].to_numpy()
            comparisons.append({'dataset': dataset, 'candidate': CANDIDATE, 'comparator': fp,
                'metric': metric, 'mean_paired_difference': float(delta.mean()),
                'sample_std': float(delta.std(ddof=1)), 'repeats': len(delta),
                'positive_repeats': int((delta > 0).sum()), 'negative_repeats': int((delta < 0).sum())})
    return summary, scores, stability, pd.DataFrame(comparisons)


def figures(export, frame, summary, panels, task):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metric = PRIMARY[task]
    datasets = sorted(frame.dataset.unique())
    combos = list(itertools.product(A.LEVELS['tokenizer'], A.LEVELS['sketch'], A.LEVELS['mode']))
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 12), squeeze=False, layout='constrained')
    means = summary[summary.metric.eq(metric)].set_index(['dataset', 'fingerprint'])['mean']
    matrices = []
    for dataset in datasets:
        matrices.append(np.array([[means.loc[(dataset, f'synrfp-{t}-r{r}-b2048-{s}-{m}')]
                                  for r in [2, 3]] for t, s, m in combos]))
    vmin, vmax = min(np.nanmin(x) for x in matrices), max(np.nanmax(x) for x in matrices)
    for i, (dataset, matrix) in enumerate(zip(datasets, matrices)):
        ax = axes[0, i]; im = ax.imshow(matrix, aspect='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax.set_xticks([0, 1], ['r2', 'r3']); ax.set_yticks(range(len(combos)), [' / '.join(x) for x in combos], fontsize=7)
        ax.set_title(dataset + ': ' + metric + ' mean')
        for y in range(len(combos)):
            for x in range(2):
                ax.text(x, y, f'{matrix[y,x]:.3f}', ha='center', va='center', fontsize=7,
                        color='white' if matrix[y,x] < (vmin+vmax)/2 else 'black')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.5, label='Mean of five correlated repeats')
    fig.savefig(export / 'configuration_heatmap.png', dpi=150); fig.savefig(export / 'configuration_heatmap.pdf'); plt.close(fig)
    fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), squeeze=False, layout='constrained')
    for i, dataset in enumerate(datasets):
        d = panels[panels.dataset.eq(dataset) & panels.metric.eq(metric) & panels.order.eq(1)].copy()
        labels = (d.effect + ': ' + d.level + ' vs ' + d.reference).to_list()
        ax = axes[0, i]; ax.barh(labels, d['mean'], xerr=d.sample_std, color='#286d8f')
        ax.axvline(0, color='black', lw=.8); ax.set_title(dataset); ax.set_xlabel('Paired main effect; error bars = descriptive SD')
    fig.savefig(export / 'main_effects.png', dpi=150); fig.savefig(export / 'main_effects.pdf'); plt.close(fig)


def markdown(export, plan, task, summary, scores, stability, panels):
    metric = PRIMARY[task]
    text = [f'# {task.capitalize()} factorial analysis', '',
            f'Completed {A.COUNTS[task]} canonical cells:72 configurations plus DRFP, five repeats per panel.',
            'All configurations have2048 total bits. The panels are capped at5000 rows; BH retains3955.', '',
            'These previously inspected evaluation populations support exploratory analysis. Rankings and effects do not select a configuration or establish significance.', '',
            '## Main factor effects', '',
            f'Changes in {metric} relative to WL/r2/parity/delta, averaged over the other axes. SD measures five correlated paired contrasts.', '',
            '| Panel | Factor | Level versus reference | Change | Descriptive SD |', '|---|---|---|---:|---:|']
    for row in panels[panels.metric.eq(metric) & panels.order.eq(1)].itertuples():
        text.append(f'| {row.dataset} | {row.effect} | {row.level} vs {row.reference} | {row.mean:+.5f} | {row.sample_std:.5f} |')
    text += ['', '## Shared roles r3 candidate', '', '| Panel | Configuration | Mean | SD | Observed rank among72 |', '|---|---|---:|---:|---:|']
    candidates = [CANDIDATE, 'synrfp-wl-r2-b2048-parity-roles',
                  'synrfp-wl-r2-b2048-parity-delta_union', 'synrfp-wl-r3-b2048-parity-delta_union']
    for row in scores[scores.fingerprint.isin(candidates)].itertuples():
        text.append(f'| {row.dataset} | {row.fingerprint} | {row.mean:.5f} | {row.sample_std:.5f} | {row.observed_rank:g} |')
    text += ['', '## Most consistent observed configurations', '',
             'This rank table is diagnostic; it does not designate a winner for final benchmarking.', '',
             '| Configuration | Mean panel rank | Worst panel rank | Equal-panel score mean |', '|---|---:|---:|---:|']
    for row in stability.head(10).itertuples():
        text.append(f'| {row.fingerprint} | {row.mean_observed_rank:g} | {row.worst_observed_rank:g} | {row.equal_panel_mean:.5f} |')
    text += ['', '## Interpretation boundaries', '',
             '- Each task has its own metrics and model/cluster protocol; task scores are not pooled.',
             '- Full order1–4 interactions are in the CSVs; higher-order effects are descriptive.',
             '- DRFP is a contextual control and is excluded from all factor contrasts.',
             '- Primary CW uses unit token support after full multiset arithmetic; native count-weighted CW belongs to the historical study.',
             '- Delta and union use2048-bit sketches. Roles and delta+union use1024+1024-bit channels, so mode effects include this width allocation.',
             '- Neither test-score ranks nor test-label AMI/ARI authorize final configuration selection.',
             '- Clustering fits use no labels or known class count. AMI/ARI ranks are external diagnostics; silhouette and seed stability serve label-free assessment.',
             '- Stereo feature extensions are absent; new stereo work remains confined to barrier studies.', '']
    (export / 'report.md').write_text('\n'.join(text))


def report_task(root, task):
    root = Path(root).resolve(); plan, primary = A.verify(root); binding = source_seal(root)
    export = Path(plan['export']).parent / 'per_task_analysis' / task
    done = export / 'completion.json'
    if done.exists():
        receipt = json.loads(done.read_text())
        if receipt['source_binding'] != binding:
            raise ValueError('Different completed analysis source')
        for name, digest in receipt['outputs_sha256'].items():
            if A.sha(export / name) != digest:
                raise ValueError('Changed completed task analysis')
        return receipt
    loaded = load_frame(root, plan, task)
    if loaded is None:
        return {'phase': 'waiting_task_completion', 'task': task, 'expected': A.COUNTS[task]}
    frame, hashes = loaded
    effects, refused = full_effects(frame, primary, A.METRICS[task]); effects['task'] = task
    panels, family = A.summarize_effects(effects)
    summary, scores, stability, comparisons = descriptive_tables(frame, primary, task)
    export.mkdir(parents=True, exist_ok=True)
    outputs = {'configuration_summary.csv': summary, 'observed_panel_ranks.csv': scores,
               'cross_panel_consistency.csv': stability, 'roles_r3_paired_comparisons.csv': comparisons,
               'paired_full_interactions.csv': effects, 'panel_full_interactions.csv': panels,
               'family_full_interactions.csv': family}
    for name, value in outputs.items():
        A.atomic_csv(export / name, value)
    A.atomic_json(export / 'metric_refusals.json', refused)
    A.atomic_json(export / 'inputs.json', hashes)
    A.atomic_json(export / 'protocol.json', {**binding, 'task': task, 'factors': A.LEVELS,
                  'reference': A.REFERENCES, 'metrics': A.METRICS[task], 'cap': plan['cap'],
                  'scope': plan['scope'], 'ranking': 'diagnostic only; no test-guided configuration selection'})
    markdown(export, plan, task, summary, scores, stability, panels)
    figures(export, frame, summary, panels, task)
    if source_seal(root) != binding or any(A.sha(p) != digest for p, digest in hashes.items()):
        raise ValueError('Analysis input changed during rendering')
    names = list(outputs) + ['metric_refusals.json', 'inputs.json', 'protocol.json', 'report.md',
                            'configuration_heatmap.png', 'configuration_heatmap.pdf', 'main_effects.png', 'main_effects.pdf']
    receipt = {'schema': SCHEMA, 'phase': 'complete', 'task': task, 'finished_utc': A.now(),
               'canonical_cells': len(frame), 'source_binding': binding,
               'all_retained_artifact_hashes_verified': True, 'orders': [1,2,3,4],
               'configuration_selected': None, 'outputs_sha256': {name: A.sha(export / name) for name in names}}
    A.atomic_json(done, receipt)
    return receipt


def self_test():
    rows, primary = [], set()
    for levels in itertools.product(*(A.LEVELS[f] for f in A.FACTORS)):
        name = '|'.join(map(str, levels)); primary.add(name)
        active = [levels[i] == ['morgan',3,'minhash','union'][i] for i in range(4)]
        rows.append({'dataset': 'fixture', 'seed': 0, 'fingerprint': name,
                     **dict(zip(A.FACTORS, levels)), 'score': 5 * int(all(active))})
    rows.append({'dataset':'fixture','seed':0,'fingerprint':'drfp','score':1e9})
    frame = pd.DataFrame(rows); full, refused = full_effects(frame, primary, ['score'])
    assert not refused and len(full) == 71
    fourth = full[(full.order == 4) & (full.level == 'morgan:3:minhash:union')].contrast.iloc[0]
    np.testing.assert_allclose(fourth, 5)
    third = full[(full.effect == 'tokenizer:radius:sketch') & (full.level == 'morgan:3:minhash')].contrast.iloc[0]
    np.testing.assert_allclose(third, 5 / 4)
    old, _ = A.paired_effects(frame, primary, ['score'])
    np.testing.assert_allclose(full[full.order <= 2].contrast.to_numpy(), old.contrast.to_numpy())
    try:
        full_effects(frame.iloc[1:], primary, ['score'])
    except ValueError:
        pass
    else:
        raise AssertionError('Incomplete grid accepted')
    print(json.dumps({'self_test':'passed','all71contrasts':True,'knownhigherorders':True,
                      'lowerordersmatchsealedanalysis':True,'missinggridrefused':True,'DRFPexcluded':True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=A.DEFAULT_ROOT)
    parser.add_argument('--task', choices=list(PRIMARY))
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        while True:
            tasks = [args.task] if args.task else list(PRIMARY)
            statuses = [report_task(args.root, task) for task in tasks]
            print(json.dumps([{k:v for k,v in s.items() if k != 'outputs_sha256'} for s in statuses]), flush=True)
            if all(s['phase'] == 'complete' for s in statuses) or not args.watch:
                break
            time.sleep(30)
