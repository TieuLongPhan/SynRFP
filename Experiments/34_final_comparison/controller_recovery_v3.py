"""CPU-only continuation with immediate visible scientific-failure latches.

All sealed v1/v2 producers, plans and historical latches remain unchanged.
No GPU job, lease acquisition or GPU-admission helper is invoked by this queue.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
from pathlib import Path
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('sealed_controller_v2_for_cpu_recovery', HERE / 'controller_parallel_v2.py')
N = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N)
C, B = N.C, N.B
ROOT, EXPORT = N.ROOT, N.EXPORT
V3 = ROOT / 'recovery_v3'
N.V2 = V3  # Only the scoped transient-recovery event destination is rebound.
D = C.Digests()
OWN = ('controller_recovery_v3.py', 'test_controller_recovery_v3.py', 'report_refresh_v3.py')
PRODUCER = ('clustering_recovery_v3.py', 'test_clustering_recovery_v3.py')


def checked_ready(path, names):
    if not path.exists():
        return None
    r = C.read(path)
    if r.get('phase') != 'ready':
        return None
    expected = {str((HERE / n).resolve()) for n in names}
    if (type(r.get('tests_passed')) is not int or r['tests_passed'] < 1 or
            set(r.get('sources_sha256', {})) != expected):
        raise ValueError('Recovery readiness must bind exact tested sources')
    for p, h in r['sources_sha256'].items():
        D.verify(p, h)
    return r


def seal():
    own = V3 / 'CODE_READY.json'
    producer = ROOT / 'clustering/recovery_v3/CODE_READY.json'
    a, b = checked_ready(own, OWN), checked_ready(producer, PRODUCER)
    if a is None or b is None:
        return None
    old = ROOT / 'parallel_v2/execution.json'
    oldseal = ROOT / 'parallel_v2/execution_seal.json'
    if B.sha(old) != C.read(oldseal)['sha256']:
        raise ValueError('Previous sealed execution changed')
    original = N.N.seal()
    if original is None:
        raise ValueError('Original scientific execution lost')
    prior = C.read(old)
    sources = dict(prior['sources_sha256'])
    sources.update(a['sources_sha256']); sources.update(b['sources_sha256'])
    for p in (old, oldseal, own, producer, ROOT / 'GPUS_RELEASED.json'):
        sources[str(p)] = B.sha(p)
    for p, h in sources.items():
        D.verify(p, h)
    path = V3 / 'execution.json'
    if path.exists():
        if B.sha(path) != C.read(V3 / 'execution_seal.json')['sha256']:
            raise ValueError('Recovery execution changed')
        r = C.read(path)
        for p, h in r['sources_sha256'].items():
            D.verify(p, h)
        return r
    r = dict(schema='synrfp-final-cpu-clustering-recovery-v3', created_utc=B.now(),
             sources_sha256=sources, previous_execution_sha256=B.sha(old),
             gpu_jobs=0, original_sources_unchanged=True, historical_latches_preserved=True,
             workers=8, cpu_per_worker=2, memory_gib_per_worker=10,
             repair_scope='Secondary independent center replay uses rigorous float64 regret bounds; native fit and exact native predict unchanged')
    B.write(path, r); B.write(V3 / 'execution_seal.json', dict(sha256=B.sha(path)))
    return r


def jobs(manifest):
    pins = manifest['sources_sha256']
    def make(name, script, args, cpu, ram, cores, done):
        return dict(id=name, unit='synrfp-final-comparison-' + name + '-v3', stage=name,
                    python=B.PYTHON, arguments=[str(HERE / script), *args], cpu_equivalents=cpu,
                    memory_gib=ram, cores=cores, work_done=done, sources_sha256=pins,
                    environment=dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                    OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false'))
    rec = ROOT / 'clustering/recovery_v3'
    receipt = N.N.receipt
    result = {
        'prepare': make('recovery-prepare', 'clustering_recovery_v3.py', ['--prepare'], 2, 10, '6-7', receipt(rec/'PREPARED.json', phase='complete')),
        'preflight': make('recovery-preflight', 'clustering_recovery_v3.py', ['--preflight'], 2, 10, '6-7', receipt(rec/'PREFLIGHT_READY.json', phase='complete')),
        'pilot': make('recovery-pilot', 'clustering_recovery_v3.py', ['--pilot'], 2, 10, '8-9', receipt(rec/'PILOT_READY.json', phase='ready')),
        'refresh': make('recovery-refresh', 'report_refresh_v3.py', ['--watch'], 2, 8, '24-25', receipt(ROOT/'report_refresh_v3/WATCH_COMPLETE.json', phase='complete')),
        'finalize': make('recovery-finalize', 'clustering_recovery_v3.py', ['--finalize'], 2, 10, '8-9', receipt(EXPORT/'clustering/final_component.json', phase='complete', completed=240, expected=240)),
        'report': make('recovery-report', 'report.py', ['--once'], 2, 8, '20-21', receipt(ROOT/'COMPLETE.json', phase='complete', logical_cells=8238)),
    }
    for worker in range(8):
        result['worker'+str(worker)] = make('recovery-worker'+str(worker), 'clustering_recovery_v3.py', ['--run','--worker',str(worker)],
            2, 10, str(8+2*worker)+'-'+str(9+2*worker), receipt(rec/('WORKER_'+str(worker)+'_COMPLETE.json'),phase='complete'))
    return result


def scientific_failure(log):
    return bool(re.search(r'(?:ValueError|IntegrityError|AssertionError):|(?:RuntimeError):[^\n]*(?:replay|validation|mismatch|differs|changed|digest|hash|identity|sealed)|Saved center prediction replay differs', log, re.I))


def ensure(job):
    state = C.props(job['unit'])
    C.attest_service(job, state)
    if state.get('ActiveState') == 'failed' and not C.gate(job['work_done'], D):
        path = C.ROOT / 'attempts' / (job['id'] + '.json')
        prior = C.read(path) if path.exists() else {}
        log = C.failure_logs(job, state, prior)
        if scientific_failure(log):
            C.latch(job, 'scientific_validation_failure_no_retry', dict(logs=log, unit=job['unit'], source_guard_unchanged=True))
    return N.safe_ensure(job, D)


def blocked(decisions):
    return {k: v for k, v in decisions.items() if v.get('phase') in ('latched','retry_budget_exhausted','integrity_latched')}


def old_workers_released():
    units = ['synrfp-final-comparison-clustering'+str(i)+'-v1' for i in range(4)]
    units += ['synrfp-final-comparison-clustering-parallel'+str(i)+'-v2' for i in range(8)]
    return not any(B.state(u) in C.ACTIVE for u in units)


def blocked_status(decisions, specs):
    bad = blocked(decisions)
    if not bad:
        return None
    frozen = []
    for key, job in specs.items():
        if key.startswith('worker') and B.state(job['unit']) in C.ACTIVE:
            C.attest_service(job, C.props(job['unit']))
            C.freeze_or_thaw('freeze', job['unit']); frozen.append(job['unit'])
    return dict(phase='blocked_by_scientific_failure', failed_or_latched_jobs=bad,
                decisions=decisions, frozen_recovery_workers=frozen, automatic_bypass=False)


def cycle():
    manifest = seal()
    if manifest is None:
        return dict(phase='waiting_tested_recovery_sources')
    specs = jobs(manifest)
    for job in specs.values():
        C.attest_service(job, C.props(job['unit']))
    pressure = C.pressure(B.snapshot(), list(specs.values()))
    if pressure['units'] or pressure['low_memory'] or pressure['low_disk']:
        return dict(phase='waiting_resource_recovery', pressure=pressure)
    decisions = {}
    # Publish the completed six tasks immediately; the watcher remains CPU-only.
    decisions['refresh'] = ensure(specs['refresh'])
    failure = blocked_status(decisions, specs)
    if failure:
        return failure
    if not old_workers_released():
        return dict(phase='waiting_previous_cpu_workers_to_release', decisions=decisions)
    for stage in ('prepare','preflight','pilot'):
        if not N.N.finished(specs[stage]):
            decisions[stage] = ensure(specs[stage])
            return blocked_status(decisions, specs) or dict(phase='recovery_'+stage, decisions=decisions)
    for worker in range(8):
        key = 'worker'+str(worker)
        decisions[key] = ensure(specs[key])
        failure = blocked_status(decisions, specs)
        if failure:
            return failure
    if all(N.N.finished(specs['worker'+str(i)]) for i in range(8)):
        decisions['finalize'] = ensure(specs['finalize'])
        failure = blocked_status(decisions, specs)
        if failure:
            return failure
        if N.N.finished(specs['finalize']):
            decisions['report'] = ensure(specs['report'])
            failure = blocked_status(decisions, specs)
            if failure:
                return failure
            if N.N.finished(specs['report']):
                value = dict(phase='complete', logical_cells=8238, completed_utc=B.now(),
                             final_comparison_sha256=B.sha(ROOT/'COMPLETE.json'), gpu_jobs=0,
                             recovery_execution_sha256=B.sha(V3/'execution.json'))
                B.write(V3/'COMPLETE.json',value); B.write(ROOT/'BENCHMARK_QUEUES_COMPLETE.json',value)
                return value
    return dict(phase='running_cpu_recovery', decisions=decisions, gpu_jobs=0,
                complete_native_cell_inventory=len(list((ROOT/'clustering/cells').rglob('*.json'))))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); parser.add_argument('--once',action='store_true')
    args = parser.parse_args(); V3.mkdir(parents=True,exist_ok=True)
    if args.prepare:
        if seal() is None:
            raise RuntimeError('Recovery tested sources pending')
        print('CPU recovery extension sealed; GPUs untouched'); return
    with (V3/'controller.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (V3/'CONTROLLER_FAILED.json').exists():
            return
        try:
            while True:
                value=dict(cycle(),updated_utc=B.now(),resources=B.snapshot())
                B.write(V3/'controller_status.json',value); B.write(EXPORT/'controller_recovery_v3_status.json',value)
                if value['phase']=='blocked_by_scientific_failure':
                    B.write(V3/'BLOCKED.json',value)
                    return
                if args.once or value['phase']=='complete':
                    return
                time.sleep(30)
        except Exception as e:
            error=dict(phase='failed',error=repr(e),updated_utc=B.now(),gpu_jobs=0)
            B.write(V3/'CONTROLLER_FAILED.json',error); B.write(V3/'controller_status.json',error)
            raise


if __name__=='__main__':
    main()
