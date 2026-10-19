"""Complete the missed selected-preset benchmark handoff under resource caps.

Only this new study's services are controlled. Native study sources, existing
weights and physical GPU leases remain owned by their original protocols.
"""
from __future__ import annotations

import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = Path.home() / '.local/share/synrfp-runs/final_comparison_v1'
EXPORT = HERE / 'publication_metadata/final_comparison_v1'
HELPER = HERE.parent / '26_ncs_continuous/controller.py'
spec = importlib.util.spec_from_file_location('final_comparison_attested_helpers', HELPER)
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)
B = C.B
C.ROOT = ROOT / 'controller_helpers'
C.EXPORT = EXPORT / 'controller_helpers'
B.ROOT = C.ROOT
B.EXPORT = C.EXPORT
COMPONENT_FILES = {
    'report': ('report.py', 'test_report.py', 'PROTOCOL.md'),
    'regression': ('regression.py', 'test_regression.py'),
    'clustering': ('clustering.py', 'test_clustering.py'),
}
D = C.Digests()


def ready(component):
    path = ROOT / component / 'CODE_READY.json'
    if not path.exists():
        return None
    value = C.read(path)
    if value.get('phase') != 'ready':
        return None
    if type(value.get('tests_passed')) is not int or value['tests_passed'] < 1:
        raise ValueError('Component lacks tested readiness: ' + component)
    hashes = value.get('sources_sha256', {})
    expected = {str((HERE / name).resolve()) for name in COMPONENT_FILES[component]}
    if set(hashes) != expected:
        raise ValueError('Component readiness has unexpected source paths: ' + component)
    for source, digest in hashes.items():
        D.verify(source, digest)
    return value


def seal():
    initial = C.read(ROOT / 'controller_seal.json')
    for path, digest in initial['sources_sha256'].items():
        D.verify(path, digest)
    path = ROOT / 'execution.json'
    if path.exists():
        value = C.read(path)
        for source, digest in value['sources_sha256'].items():
            D.verify(source, digest)
        return value
    proofs = {component: ready(component) for component in COMPONENT_FILES}
    if not all(proofs.values()):
        return None
    sources = dict(initial['sources_sha256'])
    for component, proof in proofs.items():
        sources.update(proof['sources_sha256'])
        p = ROOT / component / 'CODE_READY.json'
        sources[str(p)] = B.sha(p)
    frozen = HERE.parent / '33_shared_preset_decision/publication_metadata/shared_preset_decision_v1/FROZEN_PRESET.json'
    value = C.read(frozen)
    if value.get('configuration_selected') != 'synrfp-wl-r2-b2048-parity-delta':
        raise ValueError('Unexpected frozen primary preset')
    sources[str(frozen)] = B.sha(frozen)
    manifest = dict(schema='synrfp-final-comparison-execution-v1', created_utc=B.now(),
        configuration_selected=value['configuration_selected'], sources_sha256=sources,
        new_regression_fits=174, new_clustering_fits=200, clustering_logical_cells=240,
        expected_label_free_comparison_cells=8238, automatic_clock_stop=False,
        no_xgboost=True, weights_local_only=True, component_readiness=proofs)
    B.write(path, manifest)
    return manifest


def receipt(path, **equals):
    return dict(kind='receipt', path=str(path), equals=equals)


def jobs(manifest):
    pins = manifest['sources_sha256']
    def job(name, script, args, cpu, ram, cores, done, gpu=None):
        result = dict(id=name, unit='synrfp-final-comparison-' + name + '-v1',
            stage=name, python=B.PYTHON, arguments=[str(HERE / script), *args],
            cpu_equivalents=cpu, memory_gib=ram, cores=cores, work_done=done,
            sources_sha256=pins, environment=dict(CUDA_VISIBLE_DEVICES='1' if gpu == 1 else '',
            OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false', CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        if gpu is not None:
            result['gpu'] = gpu
        return result
    result = {
        'prepare': job('prepare', 'prepare_all.py', [], 2, 8, '6-7', receipt(ROOT / 'PREPARED.json', phase='complete')),
        'regression_pilot': job('regression-pilot', 'regression.py', ['--pilot'], 2, 12, '26-29',
            receipt(ROOT / 'regression/PILOT_READY.json', phase='ready', cuda=True), 1),
        'clustering_pilot': job('clustering-pilot', 'clustering.py', ['--pilot'], 2, 16, '8-9',
            receipt(ROOT / 'clustering/PILOT_READY.json', phase='ready')),
        'regression': job('regression', 'regression.py', ['--run'], 2, 12, '26-29',
            receipt(ROOT / 'regression/COMPLETE.json', phase='complete', completed=174, expected_models=174), 1),
        'report': job('report', 'report.py', ['--once'], 2, 8, '20-21', receipt(ROOT / 'COMPLETE.json', phase='complete')),
    }
    for worker in range(4):
        result['clustering' + str(worker)] = job('clustering' + str(worker), 'clustering.py',
            ['--run', '--worker', str(worker)], 2, 16, str(8+2*worker) + '-' + str(9+2*worker),
            receipt(ROOT / ('clustering/WORKER_' + str(worker) + '_COMPLETE.json'), phase='complete'))
    return result


def finished(job):
    return C.gate(job['work_done'], D) and B.state(job['unit']) not in C.ACTIVE


def cycle():
    manifest = seal()
    if manifest is None:
        return dict(phase='waiting_tested_component_sources', ready_components={c: ready(c) is not None for c in COMPONENT_FILES})
    specs = jobs(manifest)
    for job in specs.values():
        C.attest_service(job, C.props(job['unit']))
    resources = B.snapshot()
    pressure = C.pressure(resources, list(specs.values()))
    if pressure['units'] or pressure['low_memory'] or pressure['low_disk']:
        return dict(phase='waiting_resource_recovery', pressure=pressure)
    decisions = {}
    if not finished(specs['prepare']):
        decisions['prepare'] = C.ensure(specs['prepare'], D)
        return dict(phase='preparing_exact_reuse_and_missing_cells', decisions=decisions)
    for component in ('regression', 'clustering'):
        pilot = specs[component + '_pilot']
        if not finished(pilot):
            decisions[component + '_pilot'] = C.ensure(pilot, D)
    if finished(specs['regression_pilot']):
        decisions['regression'] = C.ensure(specs['regression'], D)
    if finished(specs['clustering_pilot']):
        for worker in range(4):
            key = 'clustering' + str(worker)
            decisions[key] = C.ensure(specs[key], D)
    model_keys = ['regression', *('clustering' + str(i) for i in range(4))]
    if all(finished(specs[key]) for key in model_keys):
        decisions['report'] = C.ensure(specs['report'], D)
        if finished(specs['report']):
            result = dict(phase='complete', finished_utc=B.now(), final_comparison=str(ROOT / 'COMPLETE.json'),
                final_comparison_sha256=B.sha(ROOT / 'COMPLETE.json'), execution_sha256=B.sha(ROOT / 'execution.json'))
            B.write(ROOT / 'BENCHMARK_QUEUES_COMPLETE.json', result)
            return result
    return dict(phase='running_missing_matched_comparisons', decisions=decisions)


def stop_owned(manifest):
    if manifest is None:
        return
    for job in jobs(manifest).values():
        p = C.props(job['unit'])
        if p.get('ActiveState') in C.ACTIVE:
            C.attest_service(job, p)
            subprocess.run(['systemctl', '--user', 'stop', job['unit']], check=True)


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / 'CONTROLLER_FAILED.json').exists():
            stop_owned(C.read(ROOT / 'execution.json') if (ROOT / 'execution.json').exists() else None)
            return
        try:
            while True:
                value = dict(**cycle(), updated_utc=B.now(), resources=B.snapshot())
                B.write(ROOT / 'controller_status.json', value)
                if value['phase'] == 'complete':
                    return
                time.sleep(30)
        except Exception as error:
            B.write(ROOT / 'CONTROLLER_FAILED.json', dict(phase='failed', error=repr(error), updated_utc=B.now()))
            stop_owned(C.read(ROOT / 'execution.json') if (ROOT / 'execution.json').exists() else None)
            raise


if __name__ == '__main__':
    main()
