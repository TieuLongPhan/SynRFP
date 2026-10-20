"""Matched ten-seed, known-K comparison after the corrected construction grid.

This producer never selects a preset or changes a historical result.  The
task-level decision is a separately frozen, exploratory input.  All five
baselines, rows, populations and numerical estimator recipes come from study34.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import shutil
import signal
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import pairwise_distances_argmin
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = Path.home() / '.local/share/synrfp-runs/task_preset_comparison_v1'
ROOT43 = ROOT.parent / 'morgan_ecfp_correction_v1'
ROOT22 = ROOT.parent / 'all_endpoint_factorial_full_v1'
ROOT34 = ROOT.parent / 'final_comparison_v1/clustering'
REPORT43 = HERE.parent / '43_morgan_ecfp_correction/publication_metadata/morgan_ecfp_correction_v1/clustering'
POLICIES = ['all_rows', 'shared_valid_all_six']
BASELINES = ['drfp', 'rdkit_struct', 'rdkit_diff', 'rxnfp_mlm', 'crxnfp']
SMALL = ['schneider_ub', 'schneider_b', 'uspto50k_ub', 'uspto50k_b', 'syntemp_r0', 'mech_1', 'mech_2']
LARGE = ['tpl_ub', 'tpl_b', 'ecreact_1', 'ecreact_2', 'ecreact_3']
ENDPOINTS = SMALL + LARGE
SEEDS = list(range(10))
SCHEMA = 'synrfp-task-preset-known-k-clustering-v1'
PROFILE = 'synrfp-ecfp-count-v1'
YIELD = False
_NATIVE = None


def need(value, message):
    if not value:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def immutable(path, value):
    path = Path(path)
    need(not path.exists(), 'Invocation receipt already retained: ' + str(path))
    write(path, value)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


class HashCache:
    def __init__(self):
        self.values = {}

    def verify(self, path, expected):
        path = Path(path).resolve()
        st = path.stat()
        stamp = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        if str(path) not in self.values or self.values[str(path)][0] != stamp:
            value = sha(path)
            after = path.stat()
            need(stamp == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                 'Input changed during verification')
            self.values[str(path)] = (stamp, value)
        need(self.values[str(path)][1] == expected, 'Changed pinned input: ' + str(path))
        return expected


def native():
    global _NATIVE
    if _NATIVE is None:
        path = HERE.parent / '34_final_comparison/clustering.py'
        spec = importlib.util.spec_from_file_location('task44_native_clustering34', path)
        _NATIVE = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _NATIVE
        spec.loader.exec_module(_NATIVE)
    return _NATIVE


def sources():
    paths = [HERE / 'clustering.py', HERE / 'test_clustering.py',
             HERE.parent / '34_final_comparison/clustering.py',
             HERE.parent / '34_final_comparison/clustering_recovery_v3.py',
             HERE.parent / '12_validation/07_clustering.py']
    paths += [HERE.parent / 'synrfp_exp' / name for name in ('features.py', 'data.py', 'config.py', 'metrics.py')]
    return {str(p.resolve()): sha(p) for p in paths}


source_pins = sources


def code_ready(root=ROOT, tests=0):
    need(isinstance(tests, int) and tests > 0, 'Executed passing tests are required')
    receipt = dict(schema='task-preset44-clustering-code-ready-v1', phase='complete', stage='clustering',
                   tests_passed=tests, sources_sha256=sources(), finished_utc=now())
    path = Path(root) / 'clustering_CODE_READY.json'
    if path.exists():
        previous = read(path)
        need(previous['sources_sha256'] == receipt['sources_sha256'], 'Existing clustering readiness is sealed')
        return previous
    immutable(path, receipt)
    return receipt


def environment():
    return {'numpy': np.__version__, 'pandas': pd.__version__,
            'scikit-learn': sklearn.__version__, 'scipy': scipy.__version__}


def check_environment(expected):
    need(environment() == expected, 'Use the original synrfp CPU environment')


def stage_root(root):
    return Path(root) / 'clustering'


def plan_path(root):
    return stage_root(root) / 'plan.json'


def task_key(task):
    return task['dataset'] + '__' + task['population'] + '__s' + str(task['seed'])


def result_path(root, task):
    return stage_root(root) / 'cells' / task['dataset'] / task['population'] / ('s' + str(task['seed']) + '.json')


def expected_tasks(panels):
    if 'panels' in panels:
        panels = panels['panels']
    return [dict(dataset=ds, population=population, seed=seed,
                 origin='population_identical_copy' if population != 'all_rows' and panels[ds]['shared_is_all'] else 'fresh')
            for ds in sorted(panels, key=lambda d: (panels[d]['n_rows'], d))
            for seed in SEEDS for population in POLICIES]


def stage_items(plan, worker, workers=2):
    need(workers in (1, 2) and worker in range(workers), 'Use one or two reviewed clustering shards')
    tasks = plan['tasks']
    need(len(tasks) % 2 == 0, 'Broken population pairs')
    groups = [tasks[i:i+2] for i in range(0, len(tasks), 2)]
    for group in groups:
        need([t['population'] for t in group] == POLICIES and
             len({(t['dataset'], t['seed']) for t in group}) == 1, 'Population pair was split')
    return [task for i, group in enumerate(groups) if i % workers == worker for task in group]


def validate_grid(plan):
    need(plan['schema'] == SCHEMA and plan['task'] == 'clustering', 'Wrong clustering schema')
    need(set(plan['panels']) == set(ENDPOINTS) and len(plan['sources']) == 9, 'Wrong full endpoint/source population')
    need(plan['tasks'] == expected_tasks(plan['panels']) and len(plan['tasks']) == 240, 'Changed keyed ten-seed grid')
    need(sum(t['origin'] == 'fresh' for t in plan['tasks']) == 200, 'Changed identical-population copy count')
    need(plan['baseline_cells'] == 1200 and plan['known_k'] is True and plan['normalization'] == 'none'
         and plan['labels_used_for_k'] is True and plan['row_cap'] is None and plan['gpu'] is False,
         'Changed matched main protocol')
    need(plan['configuration']['n_bits'] == 2048 and plan['configuration']['seed'] == 1, 'Wrong selected construction')
    if plan['configuration']['tokenizer'] == 'morgan':
        need(plan['tokenizer_profile'] == PROFILE, 'Historical Morgan is forbidden')
    else:
        need(plan['configuration']['tokenizer'] in ('wl', 'path') and
             plan['tokenizer_profile'] == 'study22-original-wl-path', 'Unknown WL/Path profile')
    for dataset, panel in plan['panels'].items():
        small = dataset in SMALL
        need(panel['method'] == ('kmeans' if small else 'minibatch_kmeans')
             and panel['params'] == (native().NATIVE.SMALL_PARAMS if small else native().NATIVE.LARGE_PARAMS)
             and isinstance(panel['k'], int) and 2 <= panel['k'] <= panel['n_rows'],
             'Changed native known-K estimator recipe')
    return plan


def verify_decision(path, cache=None):
    cache = cache or HashCache()
    d = read(path)
    need(d['phase'] == 'frozen' and d['task'] == 'clustering', 'Clustering decision is not frozen')
    for p, h in d['sources_sha256'].items():
        cache.verify(p, h)
    if d.get('policy') == 'user_declared_path_challenger_20261020':
        need(d['config'] == 'synrfp-path-r2-b2048-parity-delta' and d['configuration']['tokenizer'] == 'path'
             and d['tokenizer_profile'] == 'study22-original-wl-path'
             and d['candidate_status'] == 'provisional_user_requested_challenger' and d['auto_winner'] is False
             and d['plan43_sha256'] is None and d['report43_complete_sha256'] is None,
             'Only the explicitly requested unchanged Path challenger may precede study43 closure')
        cache.verify(ROOT22 / 'plan.json', d['source_plan_sha256'])
        old = read(ROOT22 / 'plan.json')
        need(d['configuration'] == old['configs'][d['config']], 'Changed requested Path construction')
        seal = Path(path).with_name(Path(path).stem + '_seal.json')
        s = read(seal)
        cache.verify(path, s.get('sha256', s.get('decision_sha256')))
        return d, None, old
    cache.verify(ROOT43 / 'plan.json', d['plan43_sha256'])
    p43 = read(ROOT43 / 'plan.json')
    old = read(ROOT22 / 'plan.json')
    need(d['config'] in old['configs'] and d['configuration'] == old['configs'][d['config']],
         'Decision construction differs from the actual factorial grid')
    receipt = REPORT43 / 'COMPLETE.json'
    cache.verify(receipt, d['report43_complete_sha256'])
    done = read(receipt)
    need(done['phase'] == 'complete' and done['task'] == 'clustering' and done['cells'] == 4320
         and done['all_keyed_cells_verified'] is True and done['all_retained_artifact_hashes_verified'] is True
         and done['source_and_feature_closure_verified'] is True and done['plan_sha256'] == d['plan43_sha256'],
         'Corrected complete 72-grid prerequisite is not closed')
    for name, h in done['outputs_sha256'].items():
        cache.verify(REPORT43 / name, h)
    expected = PROFILE if d['configuration']['tokenizer'] == 'morgan' else 'study22-original-wl-path'
    need(d['tokenizer_profile'] == expected, 'Decision tokenizer profile mismatch')
    return d, p43, old


def source_arrays(path, n):
    with np.load(path, allow_pickle=False) as z:
        need(set(z.files) == {'X', 'ok', 'n_bits', 'source_positions', 'row_ids'}, 'Missing selected feature identities')
        value = {name: z[name].copy() for name in z.files}
    need(value['X'].dtype == np.uint8 and value['X'].shape == (n, 256) and int(value['n_bits']) == 2048
         and value['ok'].dtype == np.bool_ and value['ok'].shape == (n,), 'Malformed selected feature cache')
    need(not value['X'][~value['ok']].any(), 'Invalid cached rows must remain zero')
    return value


def compare_source(table, rt, features, anchor):
    need(len(table) == len(rt) and np.array_equal(table.source_position.to_numpy(), np.arange(len(rt)))
         and np.array_equal(table.row_id.astype(str).to_numpy(), rt.r_id.astype(str).to_numpy())
         and np.array_equal(table.rxn.astype(str).to_numpy(), rt.rxn.astype(str).to_numpy()),
         'Factorial/main full-source IDs, text or positions differ; explicit refusal')
    need(np.array_equal(features['source_positions'], table.source_position.to_numpy())
         and np.array_equal(features['row_ids'].astype(str), table.row_id.astype(str).to_numpy()),
         'Feature/full-source identity differs')
    need(np.array_equal(features['ok'], anchor.ok), 'Factorial/main source validity differs; explicit refusal')


def compare_panel(table, ds):
    need(len(table) == len(ds) and np.array_equal(table.source_position.to_numpy(), ds.row.to_numpy())
         and np.array_equal(table.row_id.astype(str).to_numpy(), ds.r_id.astype(str).to_numpy())
         and np.array_equal(table.y.to_numpy(), ds.y.to_numpy()), 'Factorial/main endpoint row identity or labels differ')


def verify_baselines(native_plan, cfg, cache, pins):
    """Revalidate all 1,200 native bindings, IDs, masks and scalar identities."""
    C = native()
    catalog = read(native_plan['baseline_bindings'])
    need(catalog['expected'] == len(catalog['rows']) == 1200, 'Incomplete five-baseline catalog')
    rows = catalog['rows']
    actual = {(r['dataset'], r['population'], r['fingerprint'], int(r['seed'])) for r in rows}
    expected = {(d, p, f, s) for d in ENDPOINTS for p in POLICIES for f in BASELINES for s in SEEDS}
    need(len(actual) == len(rows) and actual == expected, 'Foreign or missing baseline cells')
    grouped = {}
    for row in rows:
        grouped.setdefault(row['dataset'], []).append(row)
    for dataset, records in grouped.items():
        panel = native_plan['panels'][dataset]
        ds = C.dataset_table(cfg, dataset)
        masks = {}
        for fp in BASELINES:
            path = C.OLD / 'encoder/features' / (panel['source'] + '.npz') if fp == 'crxnfp' else cfg['paths']['work'] / 'features' / panel['source'] / (fp + '.npz')
            cache.verify(path, panel['baseline_feature_sha256'][fp])
            masks[fp] = C.feature(path).ok[ds.row.to_numpy(np.int64)]
        for row in records:
            rp = Path(row['source_result_path']); ap = Path(row['source_assignment_path'])
            for p, h in ((rp, row['source_result_sha256']), (ap, row['source_assignment_sha256'])):
                cache.verify(p, h); pins[str(p.resolve())] = h
            rec = read(rp)
            C.validate_native_record(rec, dataset, row['fingerprint'], row['population'], int(row['seed']), panel)
            need(rec['n_scored'] == row['n_scored'], 'Baseline scored population drift')
            C.validate_assignment(ap, ds, C.used_mask(panel, row['population']), masks[row['fingerprint']])
            for key, value in rec['scores'][panel['label']].items():
                need(np.isclose(value, row[key], atol=1e-12, rtol=1e-12), 'Baseline canonical score differs')
    return catalog


def ensure_plan_ready(root, plan):
    """Close an already verified plan, including recovery before this marker.

    A crash after the immutable plan/seal but before PLAN_READY must not cause
    every later preparation invocation to return without the required closure.
    Existing closure evidence is checked and retained byte for byte.
    """
    path = stage_root(root) / 'PLAN_READY.json'
    expected = dict(phase='complete', stage='clustering', plan_sha256=sha(plan_path(root)),
        decision_sha256=plan['decision_sha256'], sources_sha256=plan['sources_sha256'], expected_items=240,
        fresh_fits=200, population_copies=40, baseline_cells=1200)
    if path.exists():
        receipt = read(path)
        need(all(receipt.get(k) == v for k, v in expected.items()), 'Existing prepared-plan closure differs')
        return receipt
    receipt = {**expected, 'finished_utc': now()}
    immutable(path, receipt)
    return receipt


def prepare(root=ROOT, decision_path=None):
    root = Path(root).resolve()
    decision_path = Path(decision_path or root / 'decisions/clustering.json').resolve()
    if plan_path(root).exists():
        p = verify(root)
        need(p['decision_path'] == str(decision_path), 'Existing stage uses another decision')
        ensure_plan_ready(root, p)
        return p
    C = native(); cache = HashCache()
    d, p43, old = verify_decision(decision_path, cache)
    main = C.verify(fresh=True)
    check_environment(main['versions'])
    complete = read(ROOT34 / 'COMPLETE.json')
    need(complete['phase'] == 'complete' and complete['completed'] == complete['expected'] == 240
         and complete['all_artifact_hashes_verified'] is True and complete['full_metric_replay'] is True
         and complete['plan_sha256'] == sha(ROOT34 / 'plan.json'), 'Original main clustering is not closed')
    cfg = C.load_config(C.OLD / 'config.yaml')
    pins = dict(main['pins'])
    for p, h in pins.items():
        cache.verify(p, h)
    pins.update(d['sources_sha256'])
    prerequisites = [decision_path, decision_path.with_name(decision_path.stem + '_seal.json'),
                     ROOT22 / 'plan.json', ROOT22 / 'plan_seal.json', ROOT34 / 'plan.json',
                     ROOT34 / 'plan_seal.json', ROOT34 / 'COMPLETE.json']
    if p43 is not None:
        prerequisites += [ROOT43 / 'plan.json', ROOT43 / 'plan_seal.json', REPORT43 / 'COMPLETE.json']
    for path in prerequisites:
        pins[str(path.resolve())] = sha(path)
    if p43 is not None:
        for name, h in read(REPORT43 / 'COMPLETE.json')['outputs_sha256'].items():
            pins[str((REPORT43 / name).resolve())] = h
    source_bindings = {}
    selected = d['configuration']; config = d['config']
    for source in sorted(main['sources']):
        owner = p43 if selected['tokenizer'] == 'morgan' else old
        src = owner['sources'][source]
        receipt_path = Path(src['feature_receipt']); receipt = read(receipt_path)
        owner_root = ROOT43 if selected['tokenizer'] == 'morgan' else ROOT22
        need(receipt['phase'] == 'complete' and receipt['plan_sha256'] == sha(owner_root / 'plan.json'), 'Selected feature producer is not closed')
        if selected['tokenizer'] == 'morgan':
            need(receipt['tokenizer_profile'] == PROFILE, 'Historical Morgan cache cannot be reused')
        feature = Path(src['features'][config]); metadata = feature.with_suffix('.json'); meta = read(metadata)
        for path, h in ((feature, receipt['files'][feature.name]), (metadata, receipt['files'][metadata.name])):
            cache.verify(path, h); pins[str(path.resolve())] = h
        need(meta['spec'] == selected and meta['feature_sha256'] == receipt['files'][feature.name]
             and meta['plan_sha256'] == receipt['plan_sha256'] and meta['source_table_sha256'] == sha(src['table']),
             'Selected construction metadata differs')
        if selected['tokenizer'] == 'morgan':
            need(meta['tokenizer_profile'] == PROFILE, 'Wrong corrected feature profile')
        for path in (receipt_path, Path(src['table'])):
            pins[str(path.resolve())] = sha(path)
        table = pd.read_csv(src['table'], dtype={'row_id': str})
        rt = C.reaction_table(cfg, source)
        features = source_arrays(feature, len(rt))
        anchor = C.feature(main['sources'][source]['path'])
        compare_source(table, rt, features, anchor)
        need(meta['rowcount'] == len(rt), 'Selected source row count differs')
        source_bindings[source] = dict(path=str(feature.resolve()), sha256=receipt['files'][feature.name],
            metadata_path=str(metadata.resolve()), metadata_sha256=receipt['files'][metadata.name],
            table=str(Path(src['table']).resolve()), rows=len(rt), owner_plan_sha256=receipt['plan_sha256'],
            tokenizer_profile=d['tokenizer_profile'], exact_native_source_equivalence=True)
    panels = {}
    for dataset in ENDPOINTS:
        panel = dict(main['panels'][dataset]); ds = C.dataset_table(cfg, dataset)
        original = old['panels'][dataset]
        pins[str(Path(original['table']).resolve())] = sha(original['table'])
        compare_panel(pd.read_csv(original['table'], dtype={'row_id': str}), ds)
        features = source_arrays(source_bindings[panel['source']]['path'], main['sources'][panel['source']]['rows'])
        with np.load(panel['population_file'], allow_pickle=False) as z:
            need(np.array_equal(z['source_row'], ds.row.to_numpy()) and np.array_equal(z['valid'], features['ok'][ds.row.to_numpy()])
                 and np.array_equal(z['all_rows'], np.ones(len(ds), bool)), 'Original main population/validity drift')
            shared = features['ok'][ds.row.to_numpy()].copy()
            for fp in BASELINES:
                path = C.OLD / 'encoder/features' / (panel['source'] + '.npz') if fp == 'crxnfp' else cfg['paths']['work'] / 'features' / panel['source'] / (fp + '.npz')
                shared &= C.feature(path).ok[ds.row.to_numpy()]
            need(np.array_equal(shared, z['shared_valid']) and bool(shared.all()) == panel['shared_is_all'],
                 'All-six shared population differs; explicit refusal')
        panel['table'] = str((cfg['paths']['work'] / 'tables' / ('ds_' + dataset + '.pkl')).resolve())
        panel['spec'] = {name: cfg['datasets'][dataset][name] for name in ('label', 'hierarchy') if name in cfg['datasets'][dataset]}
        panels[dataset] = panel
    catalog = verify_baselines(main, cfg, cache, pins)
    out = stage_root(root); out.mkdir(parents=True, exist_ok=True)
    write(out / 'baseline_bindings.json', catalog)
    pins[str((out / 'baseline_bindings.json').resolve())] = sha(out / 'baseline_bindings.json')
    plan = dict(schema=SCHEMA, task='clustering', created_utc=now(), root=str(root), config=config,
        configuration=selected, tokenizer_profile=d['tokenizer_profile'], decision_path=str(decision_path),
        decision_sha256=sha(decision_path), plan43_sha256=d['plan43_sha256'], report43_complete_sha256=d['report43_complete_sha256'],
        decision_policy=d.get('policy', 'six-source-family-balanced-v1'),
        candidate_status=d.get('candidate_status', 'exploratory_task_selected_preset'),
        native34_plan_sha256=sha(ROOT34 / 'plan.json'), baseline_cells=1200,
        baseline_bindings=str((out / 'baseline_bindings.json').resolve()), panels=panels, sources=source_bindings,
        tasks=expected_tasks(panels), expected_cells=240, expected_fresh=200, expected_population_copies=40,
        versions=main['versions'], known_k=True, labels_used_for_k=True, normalization='none', row_cap=None, gpu=False,
        seeds=SEEDS, protocol='known_k_10_seeds', interpretation='exploratory_task_selection_after_viewed_factorial',
        thread_policy='Declared worker thread count; unchanged native estimator parameters',
        independent_oracle='study34_float32_forward_error_v3', pins=pins, sources_sha256=sources())
    validate_grid(plan)
    write(plan_path(root), plan)
    write(out / 'plan_seal.json', {'sha256': sha(plan_path(root))})
    verified = verify(root)
    ensure_plan_ready(root, verified)
    return verified


def verify(root=ROOT, cache=None):
    root = Path(root); cache = cache or HashCache()
    cache.verify(plan_path(root), read(stage_root(root) / 'plan_seal.json')['sha256'])
    plan = read(plan_path(root)); validate_grid(plan)
    need(Path(plan['root']).resolve() == root.resolve(), 'Stage root changed')
    check_environment(plan['versions'])
    for p, h in {**plan['pins'], **plan['sources_sha256']}.items():
        cache.verify(p, h)
    return plan


def review(root, plan):
    r = read(Path(root) / 'ROOT_REVIEW.json')
    need(r['schema'] == 'task-preset44-root-reviewed-v1' and r['phase'] == 'reviewed'
         and 'clustering' in r['stages_allowed'] and r['cluster_seeds'] == SEEDS
         and r['study43_priority_preserved'] is True
         and r['GPU0_reserved'] is True, 'Missing reviewed clustering execution')
    if plan.get('decision_policy') == 'user_declared_path_challenger_20261020':
        need(plan['config'] == 'synrfp-path-r2-b2048-parity-delta'
             and r.get('declared_path_challenger') is True,
             'Provisional Path exception was not reviewed')
    else:
        need(r['task43_report_prerequisite'] is True, 'Corrected-grid prerequisite is required')
    for p, h in plan['sources_sha256'].items():
        need(r['sources_sha256'].get(p) == h, 'Review omits clustering source')
    for p, h in r['sources_sha256'].items():
        need(sha(p) == h, 'Reviewed source changed')
    return r


def disk_guard(root):
    for path in (Path.home(), stage_root(root).resolve()):
        need(shutil.disk_usage(path).free >= 40 * 1024**3, 'Below40GiB storage reserve')


def data(plan, task):
    panel = plan['panels'][task['dataset']]
    ds = pd.read_pickle(panel['table'])
    with np.load(panel['population_file'], allow_pickle=False) as z:
        used = z['all_rows' if task['population'] == 'all_rows' else 'shared_valid'].copy()
        valid = z['valid'].copy()
        need(np.array_equal(z['source_row'], ds.row.to_numpy()), 'Source row population changed')
    features = source_arrays(plan['sources'][panel['source']]['path'], plan['sources'][panel['source']]['rows'])
    need(np.array_equal(features['ok'][ds.row.to_numpy()], valid), 'Cached validity differs')
    X = np.unpackbits(features['X'][ds.row.to_numpy()[used]], axis=1, count=2048).astype(np.float32)
    return ds, X, valid, used


def identity(root, plan, task):
    panel = plan['panels'][task['dataset']]
    return {**task, 'task': 'clustering', 'config': plan['config'], 'fingerprint': plan['config'],
            'configuration': plan['configuration'], 'tokenizer_profile': plan['tokenizer_profile'],
            'plan_sha256': sha(plan_path(root)), 'decision_sha256': plan['decision_sha256'],
            'feature_sha256': plan['sources'][panel['source']]['sha256'],
            'dataset_sha256': panel['dataset_sha256'], 'population_sha256': plan['pins'][str(Path(panel['population_file']).resolve())],
            'method': panel['method'], 'k': panel['k'], 'params': panel['params'], 'n_rows': panel['n_rows'],
            'protocol': plan['protocol'], 'known_k': True, 'labels_used_for_k': True, 'normalization': 'none'}


def estimator(panel, seed, centers=None, threads=4):
    cls = KMeans if panel['method'] == 'kmeans' else MiniBatchKMeans
    need(panel['method'] in ('kmeans', 'minibatch_kmeans'), 'Unknown estimator')
    model = cls(n_clusters=panel['k'], random_state=seed, **panel['params'])
    if centers is not None:
        need(centers.dtype == np.float32 and centers.shape == (panel['k'], 2048) and np.isfinite(centers).all(), 'Bad saved centers')
        model.cluster_centers_ = centers
        model.n_features_in_ = 2048
        model._n_threads = threads
    return model


def independent_center_check(X, centers, native_labels):
    """Same validated forward-error boundary as study34 recovery_v3.

    Native predict is authoritative and must replay exactly for all rows.  This
    separate float32 implementation may disagree only within its distance error
    bound, checked independently using explicit float64 squared distances.
    """
    alternative = pairwise_distances_argmin(X, centers, metric='euclidean')
    mismatches = np.flatnonzero(alternative != native_labels)
    u = np.finfo(np.float32).eps / 2
    operations = 2 * X.shape[1] + 8
    gamma = operations*u/(1-operations*u)
    records = []
    c64 = centers.astype(np.float64)
    for i in mismatches:
        x = X[i].astype(np.float64)
        distances = np.sum((c64-x)**2, axis=1, dtype=np.float64)
        best, chosen = int(distances.argmin()), int(native_labels[i])
        a, b = c64[chosen], c64[best]
        scale = float(np.sum(x*x+a*a+2*np.abs(x*a)) + np.sum(x*x+b*b+2*np.abs(x*b)))
        bound = gamma*scale + 16*np.finfo(np.float64).eps*max(1., scale)
        regret = float(distances[chosen]-distances[best])
        need(regret <= bound, 'Independent center distance exceeds float32 forward bound')
        records.append(dict(sample_index=int(i), native_label=chosen, alternative_label=int(alternative[i]),
                            best_float64_label=best, regret=regret, forward_error_bound=bound))
    return dict(schema='float32-distance-forward-error-v1', rows=len(X), mismatches=len(records),
                feature_dimensions=X.shape[1], disagreements=records, native_full_predict_equality_unchanged=True)


def verify_result(root, plan, task, record=None, deep=False, cache=None):
    path = result_path(root, task)
    if record is None:
        if not path.exists():
            return None
        record = read(path)
    need(record['phase'] == 'complete' and record['identity'] == identity(root, plan, task), 'Clustering result identity differs')
    cache = cache or HashCache()
    for p, h in record['artifacts_sha256'].items():
        cache.verify(p, h)
    ds = pd.read_pickle(plan['panels'][task['dataset']]['table'])
    panel = plan['panels'][task['dataset']]
    used = native().used_mask(panel, task['population'])
    with np.load(panel['population_file'], allow_pickle=False) as z:
        valid = z['valid'].copy()
    labels = native().validate_assignment(record['assignment_path'], ds, used, valid)
    need(record['n_scored'] == int(used.sum()) and record['n_valid'] == int(valid.sum()), 'Changed retained coverage')
    need(record['prediction_replay'] is True and record['full_metric_replay'] is True, 'Missing replay evidence')
    if task['origin'] == 'population_identical_copy':
        original_task = {**task, 'population': 'all_rows', 'origin': 'fresh'}
        source = result_path(root, original_task)
        need(used.all() and record['copied_result_sha256'] == sha(source), 'Invalid identical-population copy')
        original = verify_result(root, plan, original_task, deep=False, cache=cache)
        need(record['assignment_path'] == original['assignment_path'] and record['centers_path'] == original['centers_path']
             and record['scores'] == original['scores'], 'Population copy artifacts/scores differ')
    if deep:
        metrics = native().NATIVE.scores(ds, panel['spec'], labels, used)
        need(native().score_equal(metrics, record['scores']), 'Saved assignment metric replay differs')
        ds, X, _, _ = data(plan, task)
        centers = np.load(record['centers_path'], allow_pickle=False)
        with threadpool_limits(limits=record['threads']):
            model = estimator(panel, task['seed'], centers, record['threads'])
            need(np.array_equal(model.predict(X), labels[used]), 'Saved center full prediction replay differs')
            ix = np.unique(np.linspace(0, len(X)-1, min(len(X), 2048), dtype=np.int64))
            independent_center_check(X[ix], centers, labels[used][ix])
    return record


def save_npz(path, **values):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temp.open('wb') as f:
        np.savez_compressed(f, **values)
    temp.replace(path)


def fit(root, plan, task, threads=4):
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Clustering must hide CUDA')
    review(root, plan)
    disk_guard(root)
    existing = verify_result(root, plan, task)
    if existing is not None:
        return existing
    need(task in plan['tasks'] and threads >= 1, 'Unknown clustering task/thread count')
    path = result_path(root, task); path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    ds, X, valid, used = data(plan, task)
    panel = plan['panels'][task['dataset']]
    if task['origin'] == 'population_identical_copy':
        source_task = {**task, 'population': 'all_rows', 'origin': 'fresh'}
        original = verify_result(root, plan, source_task)
        need(original is not None and used.all(), 'Identical-population source must close first')
        record = {**original, 'identity': identity(root, plan, task), 'origin': task['origin'],
                  'copied_result_path': str(result_path(root, source_task)),
                  'copied_result_sha256': sha(result_path(root, source_task)),
                  'fit_seconds': 0., 'elapsed_seconds': time.monotonic()-started, 'completed_utc': now()}
        write(path, record)
        del X
        return verify_result(root, plan, task, deep=True)
    need(len(X) >= panel['k'] and np.isfinite(X).all() and np.isin(X, [0, 1]).all(), 'Invalid unscaled native bit features')
    with threadpool_limits(limits=threads):
        model = estimator(panel, task['seed'])
        t = time.monotonic(); fitted = model.fit_predict(X); fit_seconds = time.monotonic()-t
        need(np.array_equal(model.predict(X), fitted), 'Native full predict must equal fit assignments')
        ix = np.unique(np.linspace(0, len(X)-1, min(len(X), 2048), dtype=np.int64))
        oracle = independent_center_check(X[ix], model.cluster_centers_, fitted[ix])
    labels = np.full(len(ds), -1, np.int32); labels[used] = fitted
    assignment = path.with_suffix('.npz'); centers_path = path.with_name(path.stem + '_centers.npy')
    save_npz(assignment, labels=labels, valid=valid, used=used, source_row=ds.row.to_numpy(np.int64),
             r_id=ds.r_id.astype(str).to_numpy(dtype=str), y=ds.y.to_numpy())
    temporary = centers_path.with_name(centers_path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('wb') as f:
        np.save(f, model.cluster_centers_, allow_pickle=False)
    temporary.replace(centers_path)
    need(np.array_equal(np.load(centers_path, allow_pickle=False), model.cluster_centers_), 'Saved center array differs')
    metrics = native().NATIVE.scores(ds, panel['spec'], labels, used)
    record = dict(phase='complete', identity=identity(root, plan, task), **task,
        config=plan['config'], tokenizer_profile=plan['tokenizer_profile'], n_scored=int(used.sum()), n_valid=int(valid.sum()),
        scores=metrics, threads=threads, fit_seconds=fit_seconds, elapsed_seconds=time.monotonic()-started,
        inertia=float(model.inertia_), n_iter=int(model.n_iter_), assignment_path=str(assignment.resolve()),
        centers_path=str(centers_path.resolve()), prediction_replay=True, full_metric_replay=True,
        independent_center_oracle=oracle, peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2),
        completed_utc=now(), artifacts_sha256={str(p.resolve()): sha(p) for p in (assignment, centers_path)})
    del X, model
    # Verify every retained center prediction and metric before exposing result.
    verify_result(root, plan, task, record=record, deep=True)
    write(path, record)
    return record


def request_yield(signum, frame):
    global YIELD
    YIELD = True


def canonical_rows(root=ROOT, plan=None):
    plan = plan or verify(root)
    rows = []
    cache = HashCache()
    for task in expected_tasks(plan):
        result = verify_result(root, plan, task, cache=cache)
        need(result is not None, 'Canonical clustering export requires every logical cell')
        panel = plan['panels'][task['dataset']]
        rows.append(dict(task='clustering', dataset=task['dataset'], endpoint=task['dataset'],
            population=task['population'], seed=task['seed'], protocol=plan['protocol'],
            method=plan['config'], fingerprint=plan['config'], configuration=plan['config'], config=plan['config'],
            tokenizer_profile=plan['tokenizer_profile'], algorithm=panel['method'], origin=task['origin'],
            n_rows=panel['n_rows'], n_scored=result['n_scored'], k=panel['k'], fit_seconds=result['fit_seconds'],
            result_path=str(result_path(root, task).resolve()), result_sha256=sha(result_path(root, task)),
            source_result_path=str(result_path(root, task).resolve()), source_result_sha256=sha(result_path(root, task)),
            source_assignment_path=result['assignment_path'], source_assignment_sha256=result['artifacts_sha256'][result['assignment_path']],
            centers_path=result['centers_path'], centers_sha256=result['artifacts_sha256'][result['centers_path']],
            plan_sha256=sha(plan_path(root)), **result['scores'][panel['label']]))
    return rows


def run(root=ROOT, worker=0, workers=2, threads=4, run_id=None):
    global YIELD
    YIELD = False
    root = Path(root).resolve(); cache = HashCache(); plan = verify(root, cache)
    review(root, plan)
    need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and threads >= 1, 'CPU-only clustering admission required')
    items = stage_items(plan, worker, workers)
    run_id = run_id or 'clustering-w' + str(worker) + '-' + str(os.getpid())
    need(run_id not in ('', '.', '..') and Path(run_id).name == run_id, 'Unsafe invocation ID')
    invocation = root / 'workers' / run_id
    need(not invocation.exists(), 'Invocation ID already retained')
    invocation.mkdir(parents=True)
    signal.signal(signal.SIGUSR1, request_yield)
    base = dict(stage='clustering', worker=worker, workers=workers, run_id=run_id,
                plan_sha256=sha(plan_path(root)), decision_sha256=plan['decision_sha256'], sources_sha256=plan['sources_sha256'])
    lock_path = stage_root(root) / ('worker' + str(worker) + '.lock')
    with lock_path.open('a+') as worker_lock:
        fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        immutable(invocation / 'STARTED.json', {**base, 'phase': 'ready', 'pid': os.getpid(), 'started_utc': now()})
        verified = {}
        for task in items:
            if YIELD or (root / 'HOLD_NEW_FITS').exists():
                break
            verify(root, cache); review(root, plan)
            output = result_path(root, task)
            if verify_result(root, plan, task, cache=cache) is None:
                claim = stage_root(root) / 'claims' / (task_key(task) + '.lock')
                claim.parent.mkdir(parents=True, exist_ok=True)
                with claim.open('a+') as handle:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    if verify_result(root, plan, task, cache=cache) is None:
                        write(root / ('clustering_worker_' + str(worker) + '_status.json'),
                              {**base, 'phase': 'working', 'item': task, 'updated_utc': now()})
                        fit(root, plan, task, threads)
            need(verify_result(root, plan, task, cache=cache) is not None, 'Unclosed clustering cell')
            verified[str(output.resolve())] = sha(output)
        all_done = len(verified) == len(items)
        receipt = {**base, 'phase': 'complete' if all_done else 'yielded', 'expected_items': len(items),
                   'audited_items': len(verified), 'all_shard_items_verified': all_done,
                   'outputs_sha256': verified, 'finished_utc': now()}
        immutable(invocation / ('COMPLETE.json' if all_done else 'YIELDED.json'), receipt)
        if all_done:
            write(root / ('clustering_worker_' + str(worker) + '_COMPLETE.json'), receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true'); action.add_argument('--run', action='store_true')
    action.add_argument('--code-ready', action='store_true'); parser.add_argument('--tests-passed', type=int, default=0)
    parser.add_argument('--root', type=Path, default=ROOT); parser.add_argument('--decision', type=Path)
    parser.add_argument('--worker', type=int, choices=(0, 1), default=0)
    parser.add_argument('--workers', type=int, default=2); parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--run-id')
    a = parser.parse_args()
    value = code_ready(a.root, a.tests_passed) if a.code_ready else prepare(a.root, a.decision) if a.prepare else run(a.root, a.worker, a.workers, a.threads, a.run_id)
    print(json.dumps({k: value[k] for k in ('phase', 'expected_cells', 'audited_items') if k in value}), flush=True)


if __name__ == '__main__':
    main()
