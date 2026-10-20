"""Five-shard provenance adapter for the unchanged matched clustering report.

Original study44 report arithmetic, populations, baseline reuse and output
schemas remain unchanged.  This module verifies the versioned five-worker
partition and appends its complete operational provenance to the final closure.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


U = load('report44_v3_common', HERE / 'common.py')
previous_common = sys.modules.get('common')
sys.modules['common'] = U
try:
    R = load('report44_v3_original_report', HERE / 'report.py')
finally:
    if previous_common is None:
        sys.modules.pop('common', None)
    else:
        sys.modules['common'] = previous_common

ROOT = U.ROOT
_PARALLEL = None
ORIGINAL_ROWS = R.original_rows


def parallel():
    global _PARALLEL
    if _PARALLEL is None:
        _PARALLEL = load('report44_v3_partition_policy', HERE / 'parallel_v3.py')
    return _PARALLEL


def source_pins():
    return {str((HERE / name).resolve()): U.sha(HERE / name)
            for name in ('report_parallel_v3.py', 'test_report_parallel_v3.py')}


def bind(path, digest, hashes, cache):
    path = Path(path).resolve()
    cache.verify(path, digest)
    key = str(path)
    U.need(key not in hashes or hashes[key] == digest, 'Conflicting five-shard provenance binding')
    hashes[key] = digest


def operational(root, scientific_plan, hashes, cache):
    root = Path(root)
    op = parallel().verify_partition(root)
    directory = root / 'parallel_v3'
    pp, seal = directory / 'plan.json', directory / 'plan_seal.json'
    bind(pp, U.read(seal)['sha256'], hashes, cache)
    bind(seal, U.sha(seal), hashes, cache)
    U.need(op.get('schema') == 'task-preset44-five-shard-plan-v3'
           and op.get('scientific_plan_sha256') == U.sha(root / 'clustering/plan.json')
           and op.get('workers') == 5 and op.get('threads') == 1 and op.get('memory_gib') == 7,
           'Wrong five-shard scientific plan or worker profile')
    assignments = op['assignments']
    U.need(set(assignments) == {str(i) for i in range(5)}, 'Incomplete five-worker partition')
    actual = []
    for worker in range(5):
        tasks = assignments[str(worker)]
        U.need(len(tasks) % 2 == 0, 'A population pair was split')
        for start in range(0, len(tasks), 2):
            pair = tasks[start:start+2]
            U.need([t['population'] for t in pair] == ['all_rows', 'shared_valid_all_six']
                   and pair[0]['dataset'] == pair[1]['dataset'] and pair[0]['seed'] == pair[1]['seed'],
                   'Five-worker population copy dependencies changed')
        actual.extend(tasks)
    def key(task):
        return task['dataset'], task['population'], int(task['seed']), task['origin']
    expected = scientific_plan['tasks']
    U.need(len(actual) == len(expected) == 240 and len({key(t) for t in actual}) == 240
           and {key(t) for t in actual} == {key(t) for t in expected}
           and sum(t['origin'] == 'fresh' for t in actual) == 200
           and sum(t['origin'] == 'population_identical_copy' for t in actual) == 40,
           'Five-worker partition changed the exact native task population')
    review_path, ready_path = directory / 'ROOT_REVIEW.json', directory / 'CODE_READY.json'
    review, ready = U.read(review_path), U.read(ready_path)
    U.need(review.get('schema') == 'task-preset44-five-shard-root-reviewed' and review.get('phase') == 'reviewed'
           and review.get('plan_sha256') == U.sha(pp) and review.get('code_ready_sha256') == U.sha(ready_path)
           and review.get('workers') == 5 and review.get('worker_cpu') == 1
           and review.get('worker_threads', review.get('threads')) == 1
           and review.get('worker_memory_gib', review.get('memory_gib')) == 7
           and review.get('gpu_jobs') == 0, 'Five-shard report requires the exact reviewed CPU cohort')
    U.need(ready.get('phase') == 'complete' and type(ready.get('tests_passed')) is int
           and ready['tests_passed'] > 0 and ready.get('sources_sha256') == op['sources_sha256'],
           'Five-shard code readiness changed')
    for catalog in (op['sources_sha256'], op['inputs_sha256'], review['sources_sha256'], source_pins()):
        U.verify_catalog(catalog, cache)
        for path, digest in catalog.items():
            bind(path, digest, hashes, cache)
    U.need(all(review['sources_sha256'].get(p) == h for p, h in op['sources_sha256'].items()),
           'Five-shard root review omits operational sources')
    U.need(all(op['sources_sha256'].get(p) == h for p, h in source_pins().items()),
           'Five-shard plan omits the report adapter')
    for path in (review_path, ready_path):
        bind(path, U.sha(path), hashes, cache)
    return op


def shard_closed(root, plan, hashes, cache, producer):
    root = Path(root)
    op = operational(root, plan, hashes, cache)
    combined = {}
    for worker in range(5):
        path = root / ('clustering_worker_' + str(worker) + '_COMPLETE.json')
        if not path.exists():
            return False
        receipt = U.read(path)
        expected = {str(producer.result_path(root, task).resolve()) for task in op['assignments'][str(worker)]}
        outputs = receipt.get('outputs_sha256', {})
        U.need(receipt.get('phase') == 'complete' and receipt.get('stage') == 'clustering'
               and receipt.get('worker') == worker and receipt.get('workers') == 5
               and receipt.get('plan_sha256') == U.sha(producer.plan_path(root))
               and receipt.get('sources_sha256') == plan['sources_sha256']
               and receipt.get('all_shard_items_verified') is True
               and receipt.get('expected_items') == receipt.get('audited_items') == len(expected)
               and set(outputs) == expected and not (set(combined) & set(outputs)),
               'Wrong keyed five-shard scientific closure')
        run_id = receipt.get('run_id')
        U.need(isinstance(run_id, str) and run_id not in ('', '.', '..') and Path(run_id).name == run_id,
               'Unsafe or missing five-shard invocation identity')
        invocation = root / 'workers' / run_id
        started_path, terminal_path, op_path = (invocation / name for name in ('STARTED.json', 'COMPLETE.json', 'OPERATIONAL.json'))
        if not op_path.exists() or not terminal_path.exists():
            return False
        started, terminal, proof = U.read(started_path), U.read(terminal_path), U.read(op_path)
        U.need(terminal == receipt and started.get('run_id') == run_id and started.get('worker') == worker
               and started.get('workers') == 5 and started.get('stage') == 'clustering'
               and started.get('plan_sha256') == receipt['plan_sha256']
               and started.get('sources_sha256') == plan['sources_sha256'], 'Worker run and aggregate shard receipts disagree')
        U.need(proof.get('schema') == 'task-preset44-five-shard-worker-v3' and proof.get('phase') == 'ready'
               and proof.get('run_id') == run_id and proof.get('worker') == worker and proof.get('workers') == 5
               and proof.get('threads') == 1 and proof.get('memory_gib') == 7
               and proof.get('pid') == started.get('pid')
               and proof.get('partition_plan_sha256') == U.sha(root / 'parallel_v3/plan.json')
               and proof.get('scientific_plan_sha256') == receipt['plan_sha256']
               and proof.get('sources_sha256') == op['sources_sha256'], 'Five-shard operational witness differs')
        expected_ops = {str((root / 'parallel_v3' / name).resolve()): U.sha(root / 'parallel_v3' / name)
                        for name in ('plan.json', 'plan_seal.json', 'ROOT_REVIEW.json', 'CODE_READY.json')}
        U.need(proof.get('outputs_sha256') == expected_ops, 'Worker operational input catalogue differs')
        for p, digest in {**outputs, **proof['outputs_sha256']}.items():
            bind(p, digest, hashes, cache)
        for p in (path, started_path, terminal_path, op_path):
            bind(p, U.sha(p), hashes, cache)
        combined.update(outputs)
    expected_all = {str(producer.result_path(root, task).resolve()) for task in producer.expected_tasks(plan)}
    U.need(len(combined) == 240 and set(combined) == expected_all, 'Missing or duplicate full five-shard union')
    return True


def once(root=ROOT):
    root = Path(root)
    science = U.module('report44_v3_scientific_producer', HERE / 'clustering.py')
    scientific_plan = science.verify(root)
    provenance, cache = {}, U.HashCache()
    op = operational(root, scientific_plan, provenance, cache)
    def original_rows(cache, hashes, *args, **kwargs):
        rows = ORIGINAL_ROWS(cache, hashes, *args, **kwargs)
        operational(root, scientific_plan, hashes, cache)
        return rows
    previous_shards, previous_rows = R.shard_closed, R.original_rows
    R.shard_closed, R.original_rows = shard_closed, original_rows
    try:
        result = R.once(root)
    finally:
        R.shard_closed, R.original_rows = previous_shards, previous_rows
    if result.get('phase') == 'complete':
        target = root / 'report/clustering/COMPLETE.json'
        receipt = dict(schema='task-preset44-five-shard-report-complete-v3', phase='complete', workers=5,
            scientific_plan_sha256=U.sha(science.plan_path(root)), partition_plan_sha256=U.sha(root / 'parallel_v3/plan.json'),
            candidate_cells=240, baseline_cells=1200, original_shared_control_cells=240,
            scientific_report_math_unchanged=True, old_results_and_thread_counts_preserved=True,
            sources_sha256=op['sources_sha256'], outputs_sha256={str(target.resolve()): U.sha(target)}, finished_utc=U.now())
        U.write(root / 'parallel_v3/REPORT_COMPLETE.json', receipt)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--task', choices=('clustering',), default='clustering')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    print(once(args.root))
