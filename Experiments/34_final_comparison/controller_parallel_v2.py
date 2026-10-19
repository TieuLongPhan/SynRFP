"""Scoped continuation of sealed experiment34 with eight bounded native workers.

The original controller, producers, execution plan and numerical implementations
remain unchanged. This extension fixes only transient-service recreation and CPU
scheduling. Existing GPU regression and pilot services keep their ownership.
"""
from __future__ import annotations
import argparse
import fcntl
import importlib.util
import math
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('final_comparison_original_controller', HERE / 'controller.py')
N = importlib.util.module_from_spec(spec)
spec.loader.exec_module(N)
C, B = N.C, N.B
ROOT, EXPORT = N.ROOT, N.EXPORT
V2 = ROOT / 'parallel_v2'
D = C.Digests()


def parallel_ready():
    path = ROOT / 'clustering/parallel_v2/CODE_READY.json'
    if not path.exists():
        return None
    value = C.read(path)
    expected = {str((HERE / name).resolve()) for name in ('clustering_parallel_v2.py', 'test_clustering_parallel_v2.py')}
    if value.get('phase') != 'ready':
        return None
    if type(value.get('tests_passed')) is not int or value['tests_passed'] < 1:
        raise ValueError('Parallel producer requires meaningful passing tests')
    if set(value.get('sources_sha256', {})) != expected:
        raise ValueError('Parallel readiness must name exactly its two producer files')
    for path, digest in value['sources_sha256'].items():
        D.verify(path, digest)
    return value


def seal_extension():
    native = N.seal()
    ready = parallel_ready()
    if native is None or ready is None:
        return None
    own = V2 / 'CODE_READY.json'
    if not own.exists():
        return None
    tested = C.read(own)
    expected = {str((HERE / name).resolve()) for name in ('controller_parallel_v2.py', 'test_controller_parallel_v2.py')}
    if (tested.get('phase') != 'ready' or type(tested.get('tests_passed')) is not int or
            tested['tests_passed'] < 1 or set(tested.get('sources_sha256', {})) != expected):
        raise ValueError('Parallel controller lacks exact tested source readiness')
    original = ROOT / 'execution.json'
    prepared = ROOT / 'PREPARED.json'
    if not prepared.exists():
        raise ValueError('Parallel continuation requires original audited preparation')
    value = C.read(prepared)
    if value.get('phase') != 'complete' or value['execution_sha256'] != B.sha(original):
        raise ValueError('Native preparation does not bind unchanged original execution')
    for path, digest in value['plan_sha256'].items():
        D.verify(path, digest)
    path = V2 / 'execution.json'
    if path.exists():
        if B.sha(path) != C.read(V2 / 'execution_seal.json')['sha256']:
            raise ValueError('Parallel execution extension seal changed')
        result = C.read(path)
        for source, digest in result['sources_sha256'].items():
            D.verify(source, digest)
        return result
    sources = dict(native['sources_sha256'])
    sources.update(ready['sources_sha256'])
    sources.update(tested['sources_sha256'])
    for name in (original, prepared, own, ROOT / 'clustering/parallel_v2/CODE_READY.json'):
        sources[str(name)] = B.sha(name)
    for source, digest in sources.items():
        D.verify(source, digest)
    result = dict(schema='synrfp-final-comparison-parallel-runtime-v2', created_utc=B.now(),
                  original_execution_sha256=B.sha(original), original_execution_unchanged=True,
                  sources_sha256=sources, clustering_workers=8, cpu_per_worker=2,
                  memory_gib_per_worker=10, logical_clustering_cells=240,
                  numerical_algorithm_changed=False, weights_local_only=True,
                  retry_scope='systemctl start exit5 after transient unit disappeared; no scientific latch bypass')
    B.write(path, result)
    B.write(V2 / 'execution_seal.json', dict(sha256=B.sha(path)))
    return result


def safe_ensure(job, digests):
    """Recreate only an attested vanished transient unit after its failed restart."""
    try:
        return C.ensure(job, digests)
    except subprocess.CalledProcessError as error:
        wanted = ['systemctl', '--user', 'start', job['unit']]
        if error.returncode != 5 or list(error.cmd) != wanted:
            raise
        state = C.props(job['unit'])
        if state.get('LoadState') != 'not-found' or state.get('ActiveState') in C.ACTIVE:
            raise
        latch = C.ROOT / 'latches' / (job['id'] + '.json')
        if latch.exists() or (C.ROOT / 'latches/global_integrity.json').exists():
            raise RuntimeError('Vanished unit has a scientific latch; explicit repair required') from error
        for source, digest in job['sources_sha256'].items():
            digests.verify(source, digest)
        C.attest_service(job, state)
        if C.gate(job['work_done'], digests):
            return C.ensure(job, digests)
        if 'gpu' in job and not B.lease_available(job['gpu']):
            return dict(job=job['id'], phase='waiting_physical_gpu_lease', recovery='vanished_transient')
        reasons = B.admission(B.snapshot(), job['cpu_equivalents'], job['memory_gib'] * B.GIB)
        if reasons:
            return dict(job=job['id'], phase='waiting_resources', reasons=reasons, recovery='vanished_transient')
        launched = B.launch(job)
        if launched.get('phase') not in ('launched', 'already_active'):
            return dict(job=job['id'], phase=launched.get('phase'), recovery='vanished_transient')
        B.write(V2 / 'last_transient_recovery.json', dict(job=job['id'], unit=job['unit'],
                old_error=str(error), sources_reverified=True, scientific_latches_absent=True,
                artifacts_retained=True, recovered_utc=B.now()))
        return C.ensure(job, digests)


def jobs(manifest):
    native = C.read(ROOT / 'execution.json')
    result = {name: job for name, job in N.jobs(native).items()
              if name in ('prepare', 'regression_pilot', 'clustering_pilot', 'regression', 'report')}
    pins = manifest['sources_sha256']
    def scoped(name, arguments, cpu, ram, cores, done):
        return dict(id=name, unit='synrfp-final-comparison-' + name + '-v2', stage=name,
                    python=B.PYTHON, arguments=[str(HERE / 'clustering_parallel_v2.py'), *arguments],
                    cpu_equivalents=cpu, memory_gib=ram, cores=cores, work_done=done,
                    sources_sha256=pins, environment=dict(CUDA_VISIBLE_DEVICES='',
                    OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
                    NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false'))
    base = ROOT / 'clustering/parallel_v2'
    result['parallel_prepare'] = scoped('clustering-parallel-prepare', ['--prepare'], 2, 8, '6-7',
        N.receipt(base / 'PREPARED.json', phase='complete'))
    result['parallel_preflight'] = scoped('clustering-parallel-preflight', ['--preflight'], 2, 8, '6-7',
        N.receipt(base / 'PREFLIGHT_READY.json', phase='complete'))
    for worker in range(8):
        result['parallel' + str(worker)] = scoped('clustering-parallel' + str(worker), ['--run', '--worker', str(worker)],
            2, 10, str(8 + 2 * worker) + '-' + str(9 + 2 * worker),
            N.receipt(base / ('WORKER_' + str(worker) + '_COMPLETE.json'), phase='complete'))
    result['finalize'] = scoped('clustering-parallel-finalize', ['--finalize'], 2, 10, '8-9',
        N.receipt(EXPORT / 'clustering/final_component.json', phase='complete', completed=240, expected=240))
    return result


def native_workers_inactive():
    return not any(B.state('synrfp-final-comparison-clustering' + str(i) + '-v1') in C.ACTIVE for i in range(4))


def pilot_memory_ok(receipt):
    peak = receipt.get('peak_rss_gib')
    return (isinstance(peak, (int, float)) and not isinstance(peak, bool) and
            math.isfinite(peak) and 0 < peak < 9)


def cycle():
    manifest = seal_extension()
    if manifest is None:
        return dict(phase='waiting_tested_parallel_sources')
    specs = jobs(manifest)
    for job in specs.values():
        C.attest_service(job, C.props(job['unit']))
    pressure = C.pressure(B.snapshot(), list(specs.values()))
    if pressure['units'] or pressure['low_memory'] or pressure['low_disk']:
        return dict(phase='waiting_resource_recovery', pressure=pressure)
    decisions = {}
    if not N.finished(specs['prepare']):
        raise RuntimeError('Native preparation lost after parallel extension sealing')
    for component in ('regression', 'clustering'):
        key = component + '_pilot'
        if not N.finished(specs[key]):
            decisions[key] = safe_ensure(specs[key], D)
    if N.finished(specs['regression_pilot']):
        decisions['regression'] = safe_ensure(specs['regression'], D)
    if not native_workers_inactive():
        return dict(phase='waiting_native_clustering_workers_to_release', decisions=decisions)
    if not N.finished(specs['parallel_prepare']):
        decisions['parallel_prepare'] = safe_ensure(specs['parallel_prepare'], D)
        return dict(phase='preparing_parallel_inventory', decisions=decisions)
    if not N.finished(specs['parallel_preflight']):
        decisions['parallel_preflight'] = safe_ensure(specs['parallel_preflight'], D)
        return dict(phase='checking_parallel_inventory', decisions=decisions)
    if not N.finished(specs['clustering_pilot']):
        return dict(phase='waiting_native_large_dataset_pilot', decisions=decisions)
    pilot = C.read(ROOT / 'clustering/PILOT_READY.json')
    if not pilot_memory_ok(pilot):
        return dict(phase='waiting_pilot_memory_evidence_below_9GiB', decisions=decisions, observed_peak_gib=pilot.get('peak_rss_gib'))
    for worker in range(8):
        key = 'parallel' + str(worker)
        decisions[key] = safe_ensure(specs[key], D)
    if all(N.finished(specs['parallel' + str(i)]) for i in range(8)):
        decisions['finalize'] = safe_ensure(specs['finalize'], D)
        if N.finished(specs['finalize']) and N.finished(specs['regression']):
            decisions['report'] = safe_ensure(specs['report'], D)
            if N.finished(specs['report']):
                value = dict(phase='complete', completed_utc=B.now(), execution_sha256=B.sha(V2 / 'execution.json'),
                             final_comparison_sha256=B.sha(ROOT / 'COMPLETE.json'), logical_cells=8238)
                B.write(V2 / 'COMPLETE.json', value)
                B.write(ROOT / 'BENCHMARK_QUEUES_COMPLETE.json', value)
                return value
    return dict(phase='running_eight_native_clustering_workers_and_regression', decisions=decisions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    V2.mkdir(parents=True, exist_ok=True)
    if args.prepare:
        value = seal_extension()
        if value is None:
            raise RuntimeError('Parallel code readiness is pending')
        print('Parallel extension sealed without changing original execution')
        return
    with (V2 / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (V2 / 'CONTROLLER_FAILED.json').exists():
            return
        try:
            while True:
                value = dict(cycle(), updated_utc=B.now(), resources=B.snapshot())
                B.write(V2 / 'controller_status.json', value)
                B.write(EXPORT / 'controller_parallel_v2_status.json', value)
                if args.once or value['phase'] == 'complete':
                    return
                time.sleep(30)
        except Exception as error:
            B.write(V2 / 'CONTROLLER_FAILED.json', dict(phase='failed', error=repr(error), updated_utc=B.now(),
                    existing_fits_preserved=True, automatic_retry=False))
            raise


if __name__ == '__main__':
    main()
