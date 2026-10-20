"""Queue one CPU Path challenger without competing with Morgan or using CUDA.

Classification follow-ups and automatic winner promotion remain deferred.
This controller owns only study44 services. It never stops another governor,
signals a Morgan worker, touches a device lease, or dispatches a GPU job.
"""
from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ROOT = Path.home() / '.local/share/synrfp-runs/task_preset_comparison_v1'
ROOT43 = Path.home() / '.local/share/synrfp-runs/morgan_ecfp_correction_v1'
CONTROL = ROOT / 'control'
PYTHON = '/home/labhhc4/anaconda3/envs/synrfp/bin/python'
ENTRY = HERE / 'clustering.py'
OWN = ('common.py', 'decision.py', 'test_decision.py', 'controller.py', 'test_controller.py',
       'EXECUTION.md', 'synrfp-task-preset44-v1.service')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


U = module('preset44_controller_common', HERE / 'common.py')
P = module('preset44_controller_decision', HERE / 'decision.py')
Q = module('preset44_reviewed_persistence', REPO / 'Experiments/39_ncs_execution/controller.py')
R = module('preset44_resource85', REPO / 'Experiments/42_knn_factorial/resource85.py')
P43 = module('preset44_reviewed_41_pause', REPO / 'Experiments/43_morgan_ecfp_correction/priority_policy_v2.py')
C, B = Q.C, Q.B
Q.ROOT = CONTROL
C.ROOT = CONTROL / 'managed'
B.ROOT = C.ROOT
D = C.Digests()
DECISION_CACHE = U.HashCache()
RAW_SNAPSHOT = Q.resource_snapshot
GIB = 1024 ** 3
LANES = {'c0': ('clustering', 0, 1, 1, 12, '18')}
MORGAN_LANES = {'e0': ('encode', 0, 6, 16), 'c0': ('clustering', 0, 4, 12),
                'c1': ('clustering', 1, 4, 12), 'm0': ('mlp', 0, 2, 16)}


def need(value, message):
    if not value:
        raise C.IntegrityError(message)


read, write = C.read, B.write


def pins(catalog):
    need(isinstance(catalog, dict) and bool(catalog), 'Missing study44 sealed source/input catalogue')
    for path, digest in catalog.items():
        need(Path(path).is_absolute(), 'Study44 hash path must be absolute')
        D.verify(path, digest)


def immutable(path, value):
    if Path(path).exists():
        need(read(path) == value, 'Immutable study44 controller receipt changed: ' + str(path))
    else:
        write(path, value)
    return value


def sources():
    paths = [HERE / name for name in OWN]
    paths += [REPO / 'Experiments' / name for name in (
        '39_ncs_execution/controller.py', '26_ncs_continuous/controller.py',
        '25_followup_scheduler/budget.py', '42_knn_factorial/resource85.py',
        '43_morgan_ecfp_correction/priority_policy_v2.py')]
    return {str(path.resolve()): B.sha(path) for path in paths}


def snapshot():
    # The reviewed helper audits finite quotas for all synrfp research cgroups.
    # R also takes the minimum free space of HOME and resolved study44 storage.
    return R.snapshot(raw_snapshot=RAW_SNAPSHOT, artifact_root=ROOT / 'clustering')


def priority_context(value):
    """Conservatively retain41's stricter pending-stage and RAM headroom.

    Release requires a verified final receipt and no active41 science units.
    Absent that exact proof, temporary inactivity never releases the reserve.
    """
    if P43.paused41(value):
        return dict(active=False, pending_cpu=0, pending_memory_bytes=0,
                    reason='reviewed41_future_dispatch_paused')
    root = R.PRIORITY_ROOT
    path = root / 'control_priority_v2/COMPLETE.json'
    if path.exists():
        receipt = read(path)
        need(receipt.get('phase') == 'complete' and receipt.get('counts') == dict(mapped=9, native=10, comparison=10)
             and receipt.get('endpoints') == 12 and receipt.get('all_keyed_stage_closures_verified') is True,
             'Invalid41 terminal clustering scope')
        need(receipt.get('execution_sha256') == B.sha(root / 'control_priority_v2/execution.json'),
             'Changed41 terminal execution binding')
        pins(receipt.get('outputs_sha256'))
        exempt = {'synrfp-syncluster41-input-repair-v2', 'synrfp-syncluster41-priority-v2',
                  'synrfp-syncluster41-cpu-priority-v2'}
        live = [u for u in value.get('units', []) if u['unit'].startswith('synrfp-syncluster41-')
                and u['unit'].removesuffix('.service') not in exempt]
        props = C.props('synrfp-syncluster41-priority-v2.service')
        if not live and props.get('ActiveState') == 'inactive' and props.get('MainPID', '0') == '0':
            return dict(active=False, pending_cpu=0, pending_memory_bytes=0)
    context = dict(active=True, pending_cpu=2, pending_memory_bytes=24 * GIB)
    active = {u['unit'].removesuffix('.service') for u in value.get('units', [])}
    pending = {}
    for path in (root / 'control/dispatch').glob('*.json'):
        job = read(path)
        stage = job.get('operation', {}).get('stage')
        if stage not in ('native', 'comparison') or job.get('unit', '').removesuffix('.service') in active:
            continue
        marker = Path(job.get('work_done', {}).get('path', '/nonexistent'))
        if marker.is_file() or (root / 'control/managed/latches' / (job.get('id', '') + '.json')).exists():
            continue
        lane = job.get('lane', stage)
        pending[lane] = max(pending.get(lane, 0), int(float(job.get('memory_gib', 8)) * GIB))
    if pending:
        context['pending_memory_bytes'] = max(context['pending_memory_bytes'], sum(pending.values()))
    return context


def morgan_headroom(value):
    """Reserve every unfinished43 lane that is not already in the live snapshot."""
    plan_path = ROOT43 / 'plan.json'
    if not plan_path.exists():
        return 16, 56 * GIB
    plan_hash = B.sha(plan_path)
    plan = read(plan_path)
    active = {u['unit'].removesuffix('.service') for u in value.get('units', [])}
    cpu, memory = 0, 0
    for lane, (stage, index, lane_cpu, lane_memory) in MORGAN_LANES.items():
        marker = ROOT43 / (stage + '_worker_' + str(index) + '_COMPLETE.json')
        if marker.exists():
            receipt = read(marker)
            need(receipt.get('phase') == 'complete' and receipt.get('plan_sha256') == plan_hash
                 and receipt.get('all_shard_items_verified') is True,
                 'Invalid completed43 lane used to release priority headroom')
            verify_morgan_lane(receipt, plan, stage, index)
            continue
        if not any(unit.startswith('synrfp-morgan43-' + lane + '-g') for unit in active):
            cpu += lane_cpu
            memory += lane_memory * GIB
    return cpu, memory


def verify_morgan_lane(receipt, plan, stage, index):
    """A lane is finished only after its exact native artifact catalogue closes."""
    workers = 2 if stage == 'clustering' else 1
    if stage == 'encode':
        expected = {str(Path(v['feature_receipt'])) for v in plan['sources'].values()}
    else:
        task = 'classification' if stage == 'mlp' else 'clustering'
        def tid(t):
            return t['task'] + '__' + t['dataset'] + '__' + ('f' + str(t['fold']) if task == 'classification' else 's' + str(t['seed'])) + '__' + t['config']
        ordered = sorted(plan['tasks'][task], key=lambda t: (plan['panels'][t['dataset']]['n'], t['config'], tid(t)))
        expected = {str(ROOT43 / 'models' / tid(t) / 'result.json') if stage == 'mlp' else
                    str(ROOT43 / 'artifacts/clustering' / t['dataset'] / t['config'] / (str(t['seed']) + '.json'))
                    for t in ordered[index::workers]}
    need(receipt.get('stage') == stage and receipt.get('worker') == index and receipt.get('workers') == workers
         and receipt.get('sources_sha256') == plan['sources_sha256']
         and receipt.get('expected_items') == receipt.get('audited_items') == len(expected)
         and set(receipt.get('outputs_sha256', {})) == expected, 'Wrong native43 lane closure catalogue')
    pins(receipt['sources_sha256'])
    pins(receipt['outputs_sha256'])
    for path in expected:
        tree_audit(path)


def admission(value, cpu, memory_bytes):
    context = priority_context(value)
    reasons = R.admission(value, cpu, memory_bytes, context)
    reserved_cpu, reserved_memory = morgan_headroom(value)
    projected = dict(value, cpu_cap_sum=value['cpu_cap_sum'] + reserved_cpu,
                     memory_cap_sum_bytes=value['memory_cap_sum_bytes'] + reserved_memory)
    # Missing Morgan lanes must still be able to return through the stricter
    # legacy41 admission boundary, not merely through our absolute85% ceiling.
    for reason in R.admission(projected, cpu, memory_bytes, context):
        if reason not in reasons:
            reasons.append('morgan43_priority_' + reason)
    return reasons


B.snapshot, B.admission = snapshot, admission


def manifest():
    files = dict(controller=CONTROL / 'CONTROLLER_CODE_READY.json', science=ROOT / 'clustering_CODE_READY.json',
                 report=ROOT / 'report_CODE_READY.json', review=ROOT / 'ROOT_REVIEW.json', intent=ROOT / 'INTENT.json')
    if not all(path.exists() for path in files.values()):
        return None
    values = {name: read(path) for name, path in files.items()}
    need(values['controller'].get('sources_sha256') == sources(), 'Study44 controller source readiness changed')
    tested = {}
    for name in ('controller', 'science', 'report'):
        receipt = values[name]
        need(receipt.get('phase') == 'complete' and type(receipt.get('tests_passed')) is int
             and receipt['tests_passed'] > 0, 'Study44 stage has no executed passing tests: ' + name)
        pins(receipt['sources_sha256'])
        tested.update(receipt['sources_sha256'])
    need(str(ENTRY) in values['science']['sources_sha256'] and str(HERE / 'report.py') in values['report']['sources_sha256'],
         'Study44 scientific/report entrypoints are unbound')
    review, intent = values['review'], values['intent']
    need(review.get('schema') == 'task-preset44-root-reviewed-v1' and review.get('phase') == 'reviewed'
         and set(review.get('stages_allowed', [])) == {'prepare', 'clustering'}
         and review.get('cluster_seeds') == list(range(10)) and review.get('declared_path_challenger') is True
         and review.get('study43_priority_preserved') is True and review.get('GPU0_reserved') is True,
         'Root44 review is not limited to the requested CPU Path challenger')
    need(review.get('restore41_after_both_reports') is True, 'Root44 review omits the deferred41 governor restore')
    need(review.get('intent_sha256') == B.sha(files['intent']), 'Root44 review does not bind exact user steering')
    need(intent.get('phase') == 'approved' and intent.get('policy') == P.DECLARED_POLICY
         and intent.get('config') == P.DECLARED_CONFIG and intent.get('cluster_seeds') == list(range(10))
         and intent.get('gpu_jobs') == 0 and intent.get('classification_deferred') is True,
         'Study44 intent authorizes a different experiment or GPU work')
    need(all(review.get('sources_sha256', {}).get(path) == digest for path, digest in tested.items()),
         'Root44 review omits a tested source')
    pins(review['sources_sha256'])
    for name, field in (('controller', 'controller_code_ready_sha256'), ('science', 'clustering_code_ready_sha256'),
                        ('report', 'report_code_ready_sha256')):
        need(review.get(field) == B.sha(files[name]), 'Root44 review readiness binding differs: ' + name)
    catalog = {**tested, **review['sources_sha256'], **{str(path): B.sha(path) for path in files.values()}}
    value = dict(schema='task-preset44-controller-execution-v1', sources_sha256=catalog, gpu_jobs=0,
                 gpu0_reserved=True, stages_allowed=['prepare', 'clustering'], expected_cells=240,
                 baseline_cells=1200, fresh_fits=200, population_copies=40, workers=1,
                 worker_cpu=1, worker_memory_gib=12, policy_fraction=.85,
                 classification_dispatch=False, automatic_preset_selection=False,
                 no_35_governor_mutation=True, no_42_automatic_resume=True)
    return immutable(CONTROL / 'execution.json', value)


def decision(execution):
    result = P.declared_path(ROOT, cache=DECISION_CACHE)
    path = ROOT / 'decisions/clustering_path_challenger.json'
    need(result['policy'] == P.DECLARED_POLICY and result['config'] == P.DECLARED_CONFIG,
         'Study44 execution may only use the fixed provisional Path challenger')
    U.verify_catalog(result['sources_sha256'], DECISION_CACHE)
    need(all(execution['sources_sha256'].get(p) == h for p, h in result['sources_sha256'].items()),
         'Frozen declaration sources are not root reviewed')
    return path


def prepared(execution, path):
    marker, pp, seal = ROOT / 'clustering/PLAN_READY.json', ROOT / 'clustering/plan.json', ROOT / 'clustering/plan_seal.json'
    if not marker.exists():
        return None
    need(pp.exists() and seal.exists(), 'Study44 preparation marker precedes its sealed plan')
    D.verify(pp, read(seal)['sha256'])
    plan, receipt = read(pp), read(marker)
    need(receipt.get('phase') == 'complete' and receipt.get('stage') == 'clustering'
         and receipt.get('plan_sha256') == B.sha(pp) and receipt.get('expected_items') == 240
         and receipt.get('fresh_fits') == 200 and receipt.get('population_copies') == 40
         and receipt.get('baseline_cells') == 1200, 'Wrong prepared matched clustering scope')
    need(plan.get('task') == 'clustering' and plan.get('config') == P.DECLARED_CONFIG
         and plan.get('decision_path') == str(path.resolve()) and plan.get('decision_sha256') == B.sha(path)
         and len(plan.get('tasks', [])) == 240 and plan.get('known_k') is True
         and plan.get('normalization') == 'none' and plan.get('row_cap') is None and plan.get('gpu') is False,
         'Prepared plan changed the declared main-comparison protocol')
    need(receipt.get('sources_sha256') == plan.get('sources_sha256'), 'Preparation/scientific source sets differ')
    pins(plan['sources_sha256'])
    pins(plan['pins'])
    need(all(execution['sources_sha256'].get(p) == h for p, h in plan['sources_sha256'].items()),
         'Prepared science source was not reviewed')
    expected = {(d, pop, seed) for d in P.DATASETS['clustering'] for pop in ('all_rows', 'shared_valid_all_six') for seed in range(10)}
    actual = [(t['dataset'], t['population'], t['seed']) for t in plan['tasks']]
    need(len(actual) == len(set(actual)) and set(actual) == expected, 'Wrong keyed240-cell clustering crossing')
    frozen = dict(schema='task-preset44-stage-execution-v1', phase='ready', stage='clustering',
                  execution_sha256=B.sha(CONTROL / 'execution.json'), plan_sha256=B.sha(pp),
                  decision_sha256=B.sha(path), sources_sha256={**execution['sources_sha256'], str(pp): B.sha(pp),
                  str(seal): B.sha(seal), str(marker): B.sha(marker), str(path): B.sha(path),
                  str(path.with_name(path.stem + '_seal.json')): B.sha(path.with_name(path.stem + '_seal.json'))})
    immutable(CONTROL / 'clustering_execution.json', frozen)
    return plan, frozen


def expected_paths(plan):
    return {str(ROOT / 'clustering/cells' / task['dataset'] / task['population'] / ('s' + str(task['seed']) + '.json'))
            for task in plan['tasks']}


def worker_spec(generation, execution):
    run = 'c0-g' + str(generation).zfill(4)
    return dict(id='preset44-' + run, unit='synrfp-preset44-' + run + '-' + execution['plan_sha256'][:8],
                python=PYTHON, arguments=[str(ENTRY), '--run', '--root', str(ROOT), '--worker', '0', '--workers', '1',
                '--threads', '1', '--run-id', run], cpu_equivalents=1, memory_gib=12, cores='18',
                sources_sha256=execution['sources_sha256'], stage='clustering', lane='c0', generation=generation,
                run_id=run, worker=0, workers=1,
                environment=dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                 OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1'),
                work_done=dict(kind='receipt', path=str(CONTROL / 'closed_runs' / (run + '.json')),
                               equals=dict(phase='complete', run_id=run)))


def publish(job):
    path = CONTROL / 'dispatch' / (job['run_id'] + '.json')
    immutable(path, job)
    immutable(CONTROL / 'dispatch_seals' / path.name, dict(sha256=B.sha(path)))


def dispatched():
    jobs = []
    for path in sorted((CONTROL / 'dispatch').glob('*.json')):
        D.verify(path, read(CONTROL / 'dispatch_seals' / path.name)['sha256'])
        job = read(path)
        need(job.get('stage') in ('prepare', 'clustering', 'report')
             and re.fullmatch(r'[pcr]0-g[0-9]{4,}', job.get('run_id', '')) is not None
             and job.get('gpu') is None and job.get('environment', {}).get('CUDA_VISIBLE_DEVICES') == '',
             'Foreign or GPU dispatch in CPU study44')
        pins(job['sources_sha256'])
        C.attest_service(job, C.props(job['unit']))
        jobs.append(job)
    return jobs


def tree_audit(path, seen=None):
    """Audit saved results and their retained artifacts; history JSON is data."""
    path, seen = Path(path), set() if seen is None else seen
    key = str(path.resolve())
    if key in seen or path.suffix != '.json':
        return
    seen.add(key)
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        return
    for field in ('outputs_sha256', 'artifacts_sha256', 'files'):
        for name, digest in value.get(field, {}).items():
            p = Path(name)
            if not p.is_absolute():
                p = Path(value.get('attempt_directory', path.parent)) / p
            D.verify(p, digest)
            if p.suffix == '.json':
                tree_audit(p, seen)
    for field, suffix in (('assignments_sha256', '.npz'), ('centers_sha256', '.model.npz')):
        if field in value:
            D.verify(path.with_suffix(suffix), value[field])


def validate_terminal(path, job, execution, plan):
    receipt = read(path)
    kind = receipt.get('phase')
    need(kind in ('complete', 'yielded') and receipt.get('stage') == 'clustering'
         and receipt.get('run_id') == job['run_id'] and receipt.get('worker') == 0 and receipt.get('workers') == 1
         and receipt.get('plan_sha256') == execution['plan_sha256']
         and receipt.get('decision_sha256') == execution['decision_sha256'], 'Wrong study44 worker closure identity')
    need(receipt.get('sources_sha256') == plan['sources_sha256'], 'Worker44 scientific sources differ')
    pins(receipt['sources_sha256'])
    expected, catalog = expected_paths(plan), receipt.get('outputs_sha256', {})
    need(isinstance(catalog, dict) and set(catalog) <= expected
         and receipt.get('expected_items') == len(expected) and receipt.get('audited_items') == len(catalog),
         'Worker44 keyed closure differs')
    if kind == 'complete':
        need(set(catalog) == expected and receipt.get('all_shard_items_verified') is True, 'Incomplete44 shard')
    else:
        need(set(catalog) != expected and receipt.get('all_shard_items_verified') is False, 'Invalid44 cooperative yield')
    for p, digest in catalog.items():
        D.verify(p, digest)
        tree_audit(p)
    return kind


def terminal(job, execution, plan):
    base = ROOT / 'workers' / job['run_id']
    paths = [p for p in (base / 'COMPLETE.json', base / 'YIELDED.json') if p.exists()]
    need(len(paths) < 2, 'Ambiguous44 terminal receipts')
    if not paths:
        return None
    path = paths[0]
    kind = validate_terminal(path, job, execution, plan)
    need(kind == ('complete' if path.name == 'COMPLETE.json' else 'yielded'), 'Worker44 terminal filename differs')
    immutable(CONTROL / 'closed_runs' / (job['run_id'] + '.json'),
              dict(phase='complete', run_id=job['run_id'], kind=kind, outputs_sha256={str(path): B.sha(path)}))
    return kind


def request_yield(job):
    need(job.get('id') == 'preset44-' + job.get('run_id', '') and job.get('stage') == 'clustering',
         'May signal only this controller\'s clustering invocation')
    props = C.props(job['unit'])
    C.attest_service(job, props)
    if props.get('ActiveState') not in C.ACTIVE:
        return False
    path = ROOT / 'workers' / job['run_id'] / 'STARTED.json'
    if not path.exists():
        return False
    receipt = read(path)
    if str(receipt.get('pid')) != props.get('MainPID'):
        return False
    need(receipt.get('phase') == 'ready' and receipt.get('run_id') == job['run_id']
         and receipt.get('stage') == 'clustering'
         and receipt.get('plan_sha256') == job['sources_sha256'].get(str(ROOT / 'clustering/plan.json')),
         'Unattested44 SIGUSR1 target')
    need(all(job['sources_sha256'].get(p) == h for p, h in receipt.get('sources_sha256', {}).items()),
         'Worker44 yield source identity changed')
    subprocess.run(['systemctl', '--user', 'kill', '--kill-whom=main', '--signal=SIGUSR1', job['unit']], check=True)
    return True


def pressure_state(value, priority_active, previous=False):
    fraction = value['memory_available_bytes'] / value['memory_total_bytes']
    low, recover = (.30, .35) if priority_active else (.15, .20)
    held = fraction < low or value['disk_free_bytes'] < 42 * GIB
    held |= previous and (fraction < recover or value['disk_free_bytes'] < 43 * GIB)
    return dict(held=bool(held), available_memory_fraction=fraction, low_fraction=low, recovery_fraction=recover,
                disk_low_gib=42, disk_recovery_gib=43)


def reservation_visible(job, execution=None, plan=None):
    for _ in range(5):
        value = snapshot()
        units = {u['unit'].removesuffix('.service'): u for u in value['units']}
        item = units.get(job['unit'].removesuffix('.service'))
        if item is not None:
            need(abs(item['cpu_equivalents'] - job['cpu_equivalents']) < 1e-8
                 and item['memory_max_bytes'] == job['memory_gib'] * GIB, 'New44 cgroup reservation differs')
            C.attest_service(job, C.props(job['unit']))
            return True
        if C.props(job['unit']).get('ActiveState') not in C.ACTIVE:
            if job['stage'] == 'clustering' and terminal(job, execution, plan):
                return True
            if job['stage'] == 'prepare' and (ROOT / 'clustering/PLAN_READY.json').exists():
                return True
            if job['stage'] == 'report' and report_complete():
                return True
        time.sleep(.1)
    raise C.IntegrityError('New44 reservation not visible before another admission')


def latest_job(stage, jobs, execution=None, plan=None):
    previous = [job for job in jobs if job['stage'] == stage]
    if not previous:
        return None, 0, None
    job = previous[-1]
    props = C.props(job['unit'])
    kind = terminal(job, execution, plan) if stage == 'clustering' else None
    if props.get('ActiveState') in C.ACTIVE:
        return job, job['generation'], None
    if kind:
        return None, job['generation'], None
    invoked = bool(props.get('InvocationID')) or (ROOT / 'workers' / job['run_id']).exists()
    if not invoked:
        return job, job['generation'], None
    failure_path = CONTROL / 'abandoned_runs' / (job['run_id'] + '.json')
    if not failure_path.exists():
        logs = C.failure_logs(job, props, {})
        immutable(failure_path, dict(phase='abandoned', stage=stage, run_id=job['run_id'],
                  reason=C.fatal_reason(props.get('Result'), logs), logs=logs,
                  retry_after_epoch=time.time() + 30, artifacts_preserved=True))
    failed = read(failure_path)
    failures = [read(path) for path in (CONTROL / 'abandoned_runs').glob('*.json')]
    count = sum(item['stage'] == stage for item in failures)
    if failed.get('reason') or count >= 3:
        return None, job['generation'], dict(phase='latched', reason=failed.get('reason') or 'three_failed_generations')
    if time.time() < failed['retry_after_epoch']:
        return None, job['generation'], dict(phase='retry_backoff')
    return None, job['generation'], None


def auxiliary_spec(stage, generation, execution, path=None):
    need(stage in ('prepare', 'report'), 'Unreviewed auxiliary stage')
    run = ('p0' if stage == 'prepare' else 'r0') + '-g' + str(generation).zfill(4)
    args = [str(ENTRY), '--prepare', '--root', str(ROOT), '--decision', str(path)] if stage == 'prepare' else [
        str(HERE / 'report.py'), '--root', str(ROOT), '--task', 'clustering', '--once']
    marker = ROOT / 'clustering/PLAN_READY.json' if stage == 'prepare' else ROOT / 'report/clustering/COMPLETE.json'
    return dict(id='preset44-' + run, unit='synrfp-preset44-' + run + '-' + B.sha(CONTROL / 'execution.json')[:8],
                python=PYTHON, arguments=args, cpu_equivalents=1, memory_gib=12 if stage == 'prepare' else 8,
                cores='18', sources_sha256=execution['sources_sha256'], stage=stage, lane=run[:2],
                run_id=run, generation=generation,
                environment=dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                                 OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1'),
                work_done=dict(kind='receipt', path=str(marker), equals=dict(phase='complete')))


def report_complete():
    path = ROOT / 'report/clustering/COMPLETE.json'
    if not path.exists():
        return False
    receipt = read(path)
    need(receipt.get('phase') == 'complete' and receipt.get('task') == 'clustering'
         and receipt.get('candidate_cells') == 240 and receipt.get('baseline_cells') == 1200
         and receipt.get('all_keyed_cells_verified') is True
         and receipt.get('all_retained_artifact_hashes_verified') is True,
         'Wrong44 full clustering report closure')
    need(receipt.get('scientific_plan_sha256') == B.sha(ROOT / 'clustering/plan.json')
         and receipt.get('decision_sha256') == read(ROOT / 'clustering/plan.json')['decision_sha256'],
         'Study44 report belongs to another scientific plan or declaration')
    tree_audit(path)
    return True


def morgan_report_complete():
    path = ROOT43 / 'COMPLETE.json'
    if not path.exists():
        return False
    receipt = read(path)
    need(receipt.get('schema') == 'morgan43-full-correction-complete-v1' and receipt.get('phase') == 'complete'
         and receipt.get('corrected_mlp_cells') == 1680 and receipt.get('corrected_clustering_cells') == 1440
         and receipt.get('combined_classification_cells') == 5040 and receipt.get('combined_clustering_cells') == 4320
         and receipt.get('all_keyed_closures_verified') is True
         and receipt.get('plan_sha256') == B.sha(ROOT43 / 'plan.json')
         and receipt.get('tokenizer_profile') == 'synrfp-ecfp-count-v1', 'Wrong43 completion before41 restore')
    expected = {str(ROOT43 / 'report' / name / 'COMPLETE.json') for name in ('classification', 'clustering')}
    need(set(receipt.get('outputs_sha256', {})) == expected, 'Wrong43 report closure catalogue before41 restore')
    tree_audit(path)
    return True


def morgan_services_idle():
    # Aggregate reports are published before their services exit. The priority
    # governor must consume closure and release the original controller lock
    # before41 can resume and invalidate the paused-governor proof.
    for unit in ('synrfp-morgan43-priority-v2.service', 'synrfp-morgan43-v1.service'):
        props = C.props(unit)
        if props.get('ActiveState') in C.ACTIVE or props.get('MainPID', '0') != '0':
            return False
    for kind in ('dispatch', 'report_dispatch'):
        for path in (ROOT43 / 'control' / kind).glob('*.json'):
            D.verify(path, read(ROOT43 / 'control' / (kind + '_seals') / path.name)['sha256'])
            job = read(path)
            need(job['unit'].startswith('synrfp-morgan43-'), 'Wrong43 unit in sealed dispatch catalogue')
            pins(job['sources_sha256'])
            props = C.props(job['unit'])
            C.attest_service(job, props)
            if props.get('ActiveState') in C.ACTIVE or props.get('MainPID', '0') != '0':
                return False
    return not any(unit['unit'].startswith('synrfp-morgan43-') for unit in snapshot().get('units', []))


def restore41(jobs):
    """Restore only the paused governor after both exact report closures."""
    if not morgan_report_complete():
        return dict(phase='waiting_morgan43_complete')
    if not report_complete():
        return dict(phase='waiting_path44_complete')
    live = [job for job in jobs if job['stage'] in ('prepare', 'clustering', 'report') and
            (C.props(job['unit']).get('ActiveState') in C.ACTIVE or C.props(job['unit']).get('MainPID', '0') != '0')]
    if live:
        return dict(phase='waiting_44_service_exit')
    if not morgan_services_idle():
        return dict(phase='waiting_43_services_and_priority_governor_exit')
    marker = CONTROL / 'RESTORED41.json'
    if marker.exists():
        receipt = read(marker)
        need(receipt.get('phase') == 'complete' and receipt.get('unit') == P43.UNIT, 'Changed41 restore closure')
        pins(receipt['outputs_sha256'])
        return dict(phase='complete')
    path = CONTROL / 'RESTORE41_INTENT.json'
    durable = read(path) if path.exists() else None
    props = C.props(P43.UNIT)
    if durable is None:
        proof = P43.verify_pause(snapshot())
        if proof is None:
            return dict(phase='waiting_verified_41_pause')
        ready = dict(phase='ready', unit=P43.UNIT,
                     outputs_sha256={str(ROOT43 / 'COMPLETE.json'): B.sha(ROOT43 / 'COMPLETE.json'),
                                     str(ROOT / 'report/clustering/COMPLETE.json'): B.sha(ROOT / 'report/clustering/COMPLETE.json'),
                                     **proof['outputs_sha256']})
        immutable(path, ready)
        durable = ready
    pins(durable['outputs_sha256'])
    original = read(P43.PRIORITY / 'INTENT.json')
    P43.attest_governor(original, props)
    if props.get('ActiveState') not in C.ACTIVE:
        reasons = admission(snapshot(), .1, 256 * 1024**2)
        if reasons:
            return dict(phase='waiting_resources_for41_restore', reasons=reasons)
        subprocess.run(['systemctl', '--user', 'enable', '--now', P43.UNIT], check=True)
    for _ in range(5):
        props = C.props(P43.UNIT)
        P43.attest_governor(original, props)
        if props.get('ActiveState') in C.ACTIVE and props.get('MainPID', '0') != '0':
            immutable(marker, dict(phase='complete', unit=P43.UNIT,
                      outputs_sha256={str(path): B.sha(path), **durable['outputs_sha256']}))
            return dict(phase='complete')
        time.sleep(.1)
    raise C.IntegrityError('Restored41 governor did not become active')


def launch_auxiliary(stage, execution, jobs, path=None):
    job, generation, blocked = latest_job(stage, jobs)
    if blocked:
        return blocked
    if job is None:
        job = auxiliary_spec(stage, generation + 1, execution, path)
        reasons = admission(snapshot(), job['cpu_equivalents'], job['memory_gib'] * GIB)
        if reasons:
            return dict(phase='waiting_resources', reasons=reasons)
        publish(job)
    result = C.ensure(job, D, held=(ROOT / 'HOLD_NEW_FITS').exists())
    if result['phase'] == 'resumed_or_launched':
        reservation_visible(job)
    return result


def cycle():
    execution = manifest()
    if execution is None:
        return dict(phase='waiting_source_bound_root_review', gpu_jobs=0)
    pins(execution['sources_sha256'])
    path = decision(execution)
    jobs = dispatched()
    ready = prepared(execution, path)
    if ready is None:
        return dict(phase='waiting_prepared_matched_plan', prepare=launch_auxiliary('prepare', execution, jobs, path), gpu_jobs=0)
    plan, frozen = ready
    # Auxiliary preparation must leave its service before the science lane starts.
    if any(j['stage'] == 'prepare' and (C.props(j['unit']).get('ActiveState') in C.ACTIVE
           or C.props(j['unit']).get('MainPID', '0') != '0') for j in jobs):
        return dict(phase='waiting_prepare_service_exit', gpu_jobs=0)
    marker = ROOT / 'clustering_worker_0_COMPLETE.json'
    if marker.exists():
        receipt = read(marker)
        matches = [job for job in jobs if job['stage'] == 'clustering' and job['run_id'] == receipt.get('run_id')]
        need(len(matches) == 1, 'Full44 closure has unknown invocation')
        need(validate_terminal(marker, matches[0], frozen, plan) == 'complete', 'Wrong final44 shard phase')
        if any(j['stage'] == 'clustering' and (C.props(j['unit']).get('ActiveState') in C.ACTIVE
               or C.props(j['unit']).get('MainPID', '0') != '0') for j in jobs):
            return dict(phase='waiting_scientific_service_exit', gpu_jobs=0)
        if report_complete():
            restored = restore41(jobs)
            return dict(phase='complete' if restored['phase'] == 'complete' else 'path_complete_waiting_41_restore',
                        candidate_cells=240, baseline_cells=1200, governor41_restore=restored, gpu_jobs=0)
        return dict(phase='waiting_clustering_report', report=launch_auxiliary('report', frozen, jobs), gpu_jobs=0)
    job, generation, blocked = latest_job('clustering', jobs, frozen, plan)
    if blocked:
        return dict(phase='scientific_lane_blocked', lane=blocked, gpu_jobs=0)
    value = snapshot()
    previous = CONTROL / 'pressure.json'
    state = pressure_state(value, priority_context(value)['active'], read(previous).get('held', False) if previous.exists() else False)
    reasons = admission(value, 0, 0)
    pressure = dict(**state, reasons=reasons, updated_utc=B.now())
    write(previous, pressure)
    if job is not None and C.props(job['unit']).get('ActiveState') in C.ACTIVE:
        if state['held'] or reasons:
            request_yield(job)
        return dict(phase='running_or_yielding', pressure=pressure, gpu_jobs=0)
    if state['held'] or (ROOT / 'HOLD_NEW_FITS').exists():
        return dict(phase='waiting_resource_recovery_or_hold', pressure=pressure, gpu_jobs=0)
    if job is None:
        job = worker_spec(generation + 1, frozen)
        reasons = admission(value, 1, 12 * GIB)
        if reasons:
            return dict(phase='waiting_resources_preserving_morgan_priority', reasons=reasons, gpu_jobs=0)
        publish(job)
    need(not (ROOT / 'workers' / job['run_id']).exists(), 'Existing44 invocation must use a new generation')
    result = C.ensure(job, D)
    if result['phase'] == 'resumed_or_launched':
        reservation_visible(job, frozen, plan)
    return dict(phase='running_or_waiting', worker=result, expected_cells=240, gpu_jobs=0)


def preflight():
    completed = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(HERE), '-p', 'test_controller.py', '-v'],
                               capture_output=True, text=True)
    need(completed.returncode == 0, 'Study44 controller controls failed: ' + completed.stdout + completed.stderr)
    match = re.search(r'Ran (\d+) tests', completed.stderr)
    need(match is not None, 'No executed44 controller test count')
    value = dict(schema='task-preset44-controller-code-ready-v1', phase='complete', tests_passed=int(match.group(1)),
                 sources_sha256=sources(), gpu_jobs=0, worker_cpu=1, worker_memory_gib=12,
                 production_launched=False, ready_utc=B.now())
    path = CONTROL / 'CONTROLLER_CODE_READY.json'
    if path.exists():
        prior = read(path)
        need(prior['sources_sha256'] == value['sources_sha256'], 'Readiness44 source seal changed')
        return prior
    return immutable(path, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    CONTROL.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        print(json.dumps(preflight()))
        return
    with (CONTROL / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (CONTROL / 'FAILED.json').exists():
            return
        try:
            while True:
                status = dict(cycle(), updated_utc=B.now())
                write(CONTROL / 'status.json', status)
                if args.once or status['phase'] == 'complete':
                    return
                time.sleep(30)
        except Exception as error:
            requested, cleanup_error = [], None
            try:
                for job in dispatched():
                    if job['stage'] == 'clustering' and request_yield(job):
                        requested.append(job['run_id'])
            except Exception as cleanup:
                cleanup_error = repr(cleanup)
            write(CONTROL / 'FAILED.json', dict(phase='failed', reason=repr(error), yield_requested=requested,
                  cleanup_error=cleanup_error, artifacts_preserved=True, gpu_jobs=0, other_worker_mutations=0,
                  updated_utc=B.now()))
            raise


if __name__ == '__main__':
    main()
