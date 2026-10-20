"""Keep the shared-preset table and add the matched Path challenger separately."""
from __future__ import annotations

import argparse
import csv
import itertools
import math
from pathlib import Path
import statistics

import common as U

ROOT = U.ROOT
MAIN = ROOT.parent / 'final_comparison_v1'
MAIN_EXPORT = U.HERE.parent / '34_final_comparison/publication_metadata/final_comparison_v1'
EXPORT = U.HERE / 'publication_metadata/task_preset_comparison_v1'
BASELINES = ('drfp', 'rdkit_struct', 'rdkit_diff', 'rxnfp_mlm', 'crxnfp')
SHARED_CONFIG = 'synrfp-wl-r2-b2048-parity-delta'
POLICIES = ('all_rows', 'shared_valid_all_six')
ENDPOINTS = ('schneider_ub', 'schneider_b', 'uspto50k_ub', 'uspto50k_b',
             'syntemp_r0', 'mech_1', 'mech_2', 'tpl_ub', 'tpl_b',
             'ecreact_1', 'ecreact_2', 'ecreact_3')
METRICS = ('ami', 'ari', 'nmi', 'mcc_hungarian')


def sources():
    return {str(p.resolve()): U.sha(p) for p in (U.HERE / 'report.py', U.HERE / 'test_report.py')}


def code_ready(root=ROOT, tests=0):
    U.need(type(tests) is int and tests > 0, 'Meaningful report controls required')
    result = dict(phase='complete', schema='task-preset44-report-code-ready-v1',
                  tests_passed=tests, sources_sha256=sources(), task='clustering')
    return U.immutable(Path(root) / 'report_CODE_READY.json', result)


def csv_rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row})
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def validate_grid(rows, methods, endpoints=ENDPOINTS):
    actual = [(r['dataset'], r['population'], int(r['seed']), r['method']) for r in rows]
    expected = set(itertools.product(endpoints, POLICIES, range(10), methods))
    U.need(len(actual) == len(set(actual)) and set(actual) == expected,
           'Missing, duplicate or foreign matched ten-seed cell')
    for row in rows:
        for metric in METRICS:
            U.need(metric in row and math.isfinite(float(row[metric])), 'Missing/nonfinite clustering metric')
    populations = {}
    for row in rows:
        key = row['dataset'], row['population'], int(row['seed'])
        populations.setdefault(key, set()).add((int(row['n_rows']), int(row['n_scored']), int(row['k'])))
    U.need(all(len(values) == 1 for values in populations.values()), 'Methods do not share the same population/K')
    return rows


def combine(original, candidate, config, endpoints=ENDPOINTS):
    """A seventh labelled arm never overwrites the original shared-preset arm."""
    controls, shared, selected = [], [], []
    for raw in original:
        row = dict(raw)
        if row['method'] in BASELINES:
            row.update(result_origin='verified-study34-baseline')
            controls.append(row)
        elif row['method'] == 'synrfp_selected':
            U.need(row['configuration'] == SHARED_CONFIG, 'Original shared preset changed')
            row.update(method='synrfp_shared', result_origin='verified-study34-shared-control')
            shared.append(row)
        else:
            raise ValueError('Unexpected original comparison arm')
    for raw in candidate:
        row = dict(raw)
        U.need(row.get('configuration', row.get('config')) == config, 'Mixed candidate constructions')
        row.update(method='synrfp_task_candidate', configuration=config, result_origin='fresh-study44-path-challenger')
        selected.append(row)
    validate_grid(controls + selected, BASELINES + ('synrfp_task_candidate',), endpoints)
    validate_grid(controls + shared, BASELINES + ('synrfp_shared',), endpoints)
    both = controls + shared + selected
    validate_grid(both, BASELINES + ('synrfp_shared', 'synrfp_task_candidate'), endpoints)
    return controls + selected, both


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row['dataset'], row['population'], row['method']), []).append(row)
    result = []
    for (dataset, population, method), values in sorted(groups.items()):
        U.need({int(r['seed']) for r in values} == set(range(10)) and len(values) == 10,
               'Summary requires the ten paired initialization seeds')
        out = dict(dataset=dataset, population=population, method=method,
                   configuration=values[0]['configuration'], n_cells=10,
                   sd_scope='initialization_seed_variation_at_fixed_known_k',
                   independent_test_replicates=False)
        for metric in METRICS:
            numbers = [float(r[metric]) for r in values]
            out[metric + '_mean'] = statistics.mean(numbers)
            out[metric + '_sd'] = statistics.stdev(numbers)
        result.append(out)
    return result


def original_rows(cache, hashes, main=MAIN, export=MAIN_EXPORT):
    receipt = Path(main) / 'report/COMPLETE.json'
    complete = U.read(receipt)
    U.need(complete['phase'] == 'complete' and complete['tasks']['clustering']['complete'] is True
           and complete['tasks']['clustering']['completed_cells'] == 1440,
           'Original main clustering comparison is incomplete')
    plan = Path(main) / 'report/plan.json'
    cache.verify(plan, complete['plan_sha256'])
    path = Path(export) / 'clustering_cells.csv'
    cache.verify(path, complete['outputs_sha256'][path.name])
    for p in (receipt, plan, path):
        hashes[str(p.resolve())] = U.sha(p)
    rows = csv_rows(path)
    for row in rows:
        native = Path(row.get('source_result_path') or row.get('result_path', ''))
        digest = row.get('source_result_sha256') or row.get('result_sha256')
        U.need(native.is_file() and bool(digest), 'Original row lacks an authoritative native result')
        cache.verify(native, digest)
        hashes[str(native.resolve())] = digest
        saved = U.read(native)
        for name, artifact_hash in saved.get('artifacts_sha256', {}).items():
            artifact = Path(name) if Path(name).is_absolute() else native.parent / name
            cache.verify(artifact, artifact_hash)
            hashes[str(artifact.resolve())] = artifact_hash
        if row.get('source_assignment_path'):
            assignments = Path(row['source_assignment_path'])
            cache.verify(assignments, row['source_assignment_sha256'])
            hashes[str(assignments.resolve())] = row['source_assignment_sha256']
    return rows


def shard_closed(root, plan, hashes, cache, producer):
    first = Path(root) / 'clustering_worker_0_COMPLETE.json'
    if not first.exists():
        return False
    workers = U.read(first)['workers']
    U.need(workers in (1, 2), 'Unexpected clustering worker cohort')
    expected = {str(producer.result_path(root, t).resolve()) for t in producer.expected_tasks(plan)}
    catalog = {}
    for index in range(workers):
        path = Path(root) / ('clustering_worker_' + str(index) + '_COMPLETE.json')
        if not path.exists():
            return False
        r = U.read(path)
        U.need(r['phase'] == 'complete' and r['stage'] == 'clustering'
               and r['worker'] == index and r['workers'] == workers
               and r['plan_sha256'] == U.sha(producer.plan_path(root))
               and r['all_shard_items_verified'] is True,
               'Incomplete or foreign clustering shard')
        outputs = r['outputs_sha256']
        U.need(r['sources_sha256'] == plan['sources_sha256']
               and r['expected_items'] == r['audited_items'] == len(outputs)
               and not (set(catalog) & set(outputs)), 'Wrong shard source/count/catalog')
        catalog.update(outputs)
        hashes[str(path.resolve())] = U.sha(path)
    U.need(set(catalog) == expected, 'Wrong keyed complete candidate population')
    for p, h in catalog.items():
        cache.verify(p, h)
    hashes.update(catalog)
    return True


def once(root=ROOT):
    root = Path(root)
    ready = U.read(root / 'report_CODE_READY.json')
    U.need(ready['phase'] == 'complete' and ready['sources_sha256'] == sources(), 'Report source readiness changed')
    producer = U.module('task44_report_clustering', U.HERE / 'clustering.py')
    if not producer.plan_path(root).exists():
        result = dict(phase='pending', task='clustering', reason='waiting_reviewed_plan')
        U.write(root / 'report/status.json', result)
        return result
    plan = producer.verify(root)
    hashes, cache = {}, U.HashCache()
    if not shard_closed(root, plan, hashes, cache, producer):
        result = dict(phase='pending', task='clustering', reason='waiting_all_240_candidate_cells')
        U.write(root / 'report/status.json', result)
        return result
    selected = producer.canonical_rows(root, plan)
    original = original_rows(cache, hashes)
    primary, both = combine(original, selected, plan['config'])
    for row in selected:
        for path, digest in ((row['source_result_path'], row['source_result_sha256']),
                             (row['source_assignment_path'], row['source_assignment_sha256']),
                             (row['centers_path'], row['centers_sha256'])):
            cache.verify(path, digest)
            hashes[str(Path(path).resolve())] = digest
    for catalog in (plan['pins'], plan['sources_sha256'], sources()):
        U.verify_catalog(catalog, cache)
        hashes.update(catalog)
    decision = U.read(plan['decision_path'])
    out = EXPORT / 'clustering'
    for name, rows in (('cells.csv', primary), ('with_shared_control.csv', both),
                       ('summary.csv', summarize(primary)), ('summary_with_shared_control.csv', summarize(both))):
        write_csv(out / name, rows)
    protocol = dict(task='clustering', protocol='known_k_10_seeds', seeds=list(range(10)),
                    selected_config=plan['config'], selected_status=decision.get('candidate_status'),
                    unchanged_shared_preset=SHARED_CONFIG, primary_methods=6, methods_with_shared_control=7,
                    baseline_cells_reused=1200, candidate_cells=240, actual_candidate_fits=200,
                    identical_population_copies=40, cv_folds=False, exploratory=True,
                    declaration='Explicit Path challenger; corrected Morgan may change the eventual task winner.',
                    class_count_blind_comparison='Separate study41; no results are replaced or pooled.',
                    classification='Deferred by the user; no classification main or GPU work is completed here.')
    U.write(out / 'protocol.json', protocol)
    hashes[str(producer.plan_path(root).resolve())] = U.sha(producer.plan_path(root))
    hashes[str(Path(plan['decision_path']).resolve())] = U.sha(plan['decision_path'])
    U.write(out / 'inputs.json', hashes)
    names = ('cells.csv', 'with_shared_control.csv', 'summary.csv', 'summary_with_shared_control.csv', 'protocol.json', 'inputs.json')
    exported = {name: U.sha(out / name) for name in names}
    receipt = dict(schema='synrfp-task-preset-comparison-complete-v1', phase='complete', task='clustering',
                   candidate=plan['config'], decision_sha256=plan['decision_sha256'],
                   scientific_plan_sha256=U.sha(producer.plan_path(root)), candidate_cells=240, baseline_cells=1200,
                   primary_cells=1440, cells_with_shared_control=1680, all_keyed_cells_verified=True,
                   all_retained_artifact_hashes_verified=True,
                   initialization_seeds=10, classification_deferred=True, exploratory=True,
                   outputs_sha256=exported, completed_utc=U.now())
    U.write(out / 'COMPLETE.json', receipt)
    closed = {**receipt, 'outputs_sha256': {**hashes, **{str((out / n).resolve()): h for n, h in exported.items()}}}
    U.write(root / 'report/clustering/COMPLETE.json', closed)
    U.write(root / 'report/COMPLETE.json', closed)
    U.write(root / 'COMPLETE.json', closed)
    U.write(root / 'report/status.json', closed)
    return closed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--task', choices=('clustering',), default='clustering')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--code-ready', action='store_true')
    parser.add_argument('--tests-passed', type=int, default=0)
    args = parser.parse_args()
    print(code_ready(args.root, args.tests_passed) if args.code_ready else once(args.root))
