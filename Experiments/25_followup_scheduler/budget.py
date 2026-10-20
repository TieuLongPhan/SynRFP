"""Admit follow-up work under the combined, user-authorized resource budget.

Current factorial services remain unchanged. Quotas apply to each service and
its children; aggregate admission additionally reserves CPU/RAM for the user.
This module uses only the standard library and never initializes CUDA.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ROOT = Path.home() / '.local/share/synrfp-runs/followup_scheduler_v1'
EXPORT = HERE / 'publication_metadata/followup_scheduler_v1'
PYTHON = '/home/labhhc4/anaconda3/envs/synrfp/bin/python'
GPU_PYTHON = '/home/labhhc4/anaconda3/envs/synprop/bin/python'
GIB = 1024 ** 3
PRIMARY_ROOT = Path.home() / '.local/share/synrfp-runs/all_endpoint_factorial_full_v1'
PRIMARY_EXPORT = HERE.parent / '22_all_endpoint_factorial/publication_metadata/all_endpoint_factorial_full_v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def memory():
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        values[key] = int(value.split()[0]) * 1024
    return values


def physical_cores():
    cores = set()
    for cpu in Path('/sys/devices/system/cpu').glob('cpu[0-9]*'):
        if (cpu / 'online').exists() and (cpu / 'online').read_text().strip() == '0':
            continue
        topology = cpu / 'topology'
        cores.add(((topology / 'physical_package_id').read_text().strip(),
                   (topology / 'core_id').read_text().strip()))
    return len(cores)


def state(unit):
    result = subprocess.run(['systemctl', '--user', 'show', unit,
                             '--property=ActiveState', '--value'],
                            capture_output=True, text=True)
    return result.stdout.strip() or 'unknown'


def receipt_complete(path):
    path = Path(path)
    if not path.exists():
        return False
    value = json.loads(path.read_text())
    return value.get('phase') in ('complete', 'passed', 'ready')


def snapshot():
    command = ['systemctl', '--user', 'list-units', '--type=service',
               '--state=active,activating,reloading', '--plain', '--no-pager',
               '--no-legend', 'synrfp*']
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    units = [line.split()[0] for line in result.stdout.splitlines() if line.strip()]
    records = []
    for unit in units:
        result = subprocess.run(['systemctl', '--user', 'show', unit,
                                 '--property=ControlGroup,ActiveState,MainPID'],
                                capture_output=True, text=True, check=True)
        props = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        cgroup = Path('/sys/fs/cgroup') / props['ControlGroup'].lstrip('/')
        if not (cgroup / 'cpu.max').exists():
            if state(unit) not in ('active', 'activating', 'reloading'):
                continue
            raise RuntimeError('Cannot audit active research service: ' + unit)
        try:
            quota, period = (cgroup / 'cpu.max').read_text().split()
            maximum = (cgroup / 'memory.max').read_text().strip()
            events = dict(line.split() for line in (cgroup / 'memory.events').read_text().splitlines())
            current_memory = int((cgroup / 'memory.current').read_text())
        except FileNotFoundError:
            if state(unit) not in ('active', 'activating', 'reloading'):
                continue
            raise
        if quota == 'max' or maximum == 'max':
            raise RuntimeError('Active research service lacks a finite quota: ' + unit)
        records.append({'unit': unit, 'state': props['ActiveState'],
                        'cpu_equivalents': int(quota) / int(period),
                        'memory_max_bytes': int(maximum),
                        'memory_current_bytes': current_memory,
                        'oom_kill_events': int(events.get('oom_kill', 0)),
                        'main_pid': int(props['MainPID'])})
    mem = memory()
    count = physical_cores()
    return {'updated_utc': now(), 'physical_cores': count,
            'cpu_budget_equivalents': count * .75,
            'memory_total_bytes': mem['MemTotal'],
            'memory_budget_bytes': int(mem['MemTotal'] * .75),
            'memory_available_bytes': mem['MemAvailable'],
            'cpu_cap_sum': sum(r['cpu_equivalents'] for r in records),
            'memory_cap_sum_bytes': sum(r['memory_max_bytes'] for r in records),
            'disk_free_bytes': shutil.disk_usage(Path.home()).free,
            'units': records}


def admission(value, cpu, memory_bytes):
    reasons = []
    if value['cpu_cap_sum'] + cpu > value['cpu_budget_equivalents'] + 1e-8:
        reasons.append('aggregate_cpu_quota')
    if value['memory_cap_sum_bytes'] + memory_bytes > value['memory_budget_bytes']:
        reasons.append('aggregate_memory_caps')
    if value['memory_available_bytes'] < value['memory_total_bytes'] * .25:
        reasons.append('global_available_memory_reserve')
    if value['disk_free_bytes'] < 40 * GIB:
        reasons.append('40GiB_disk_reserve')
    return reasons


def pressure_guard(value):
    """Pause only the additional follow-ups if other apps consume the reserve.

    Freezing retains weights and CUDA contexts. The original factorial remains
    active; its models can finish and release memory. A hysteresis prevents
    repeated pause/resume cycles near the reserve boundary.
    """
    path = ROOT / 'pressure_paused.json'
    previous = json.loads(path.read_text()) if path.exists() else {'units': []}
    paused = set(previous['units'])
    low = value['memory_available_bytes'] < value['memory_total_bytes'] * .25
    recovered = value['memory_available_bytes'] >= value['memory_total_bytes'] * .30
    if low:
        for record in value['units']:
            unit = record['unit']
            if unit.startswith(('synrfp-followup-reagent-', 'synrfp-followup-frea-')) and unit not in paused:
                subprocess.run(['systemctl', '--user', 'freeze', unit], check=True)
                paused.add(unit)
    elif recovered:
        for unit in sorted(paused):
            if state(unit) in ('active', 'activating', 'reloading'):
                subprocess.run(['systemctl', '--user', 'thaw', unit], check=True)
        paused.clear()
    record = {'units': sorted(paused), 'phase': 'paused_for_memory' if paused else 'normal',
              'available_memory_bytes': value['memory_available_bytes'], 'updated_utc': now()}
    if paused != set(previous['units']):
        event = ROOT / 'pressure_events' / (str(time.time_ns()) + '.json')
        write(event, record)
    write(path, record)
    return record


def launch(spec):
    unit = spec['unit']
    current = state(unit)
    if current in ('active', 'activating', 'reloading'):
        return {'phase': 'already_active', 'unit': unit}
    if current == 'failed':
        return {'phase': 'failed_requires_inspection', 'unit': unit}
    value = snapshot()
    reasons = admission(value, spec['cpu_equivalents'], spec['memory_gib'] * GIB)
    if reasons:
        return {'phase': 'waiting_resources', 'unit': unit, 'reasons': reasons}
    env = {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2',
           'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '1',
           'TOKENIZERS_PARALLELISM': 'false', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
    env.update(spec.get('environment', {}))
    command = ['systemd-run', '--user', '--unit=' + unit,
               '--property=WorkingDirectory=' + str(REPO),
               '--property=CPUQuota=' + str(spec['cpu_equivalents'] * 100) + '%',
               '--property=MemoryMax=' + str(spec['memory_gib']) + 'G',
               '--property=MemoryHigh=' + str(int(spec['memory_gib'] * GIB * .875)),
               '--property=MemorySwapMax=0', '--property=Nice=10']
    command += ['--setenv=' + key + '=' + value for key, value in env.items()]
    command += ['/usr/bin/taskset', '-c', spec['cores'], spec.get('python', PYTHON),
                '-u', *spec['arguments']]
    subprocess.run(command, check=True)
    receipt = {'phase': 'launched', 'unit': unit, 'launched_utc': now(),
               'spec': spec, 'source_sha256': {p: sha(p) for p in spec.get('sources', [])}}
    write(ROOT / 'launches' / (unit + '.json'), receipt)
    return receipt


def primary_gpu_complete():
    """Require audited model completion, rather than an idle/failed GPU queue."""
    if (PRIMARY_ROOT / 'HOLD_NEW_FITS').exists():
        return False
    for unit in ('synrfp-full-fa-gpu0-v2.service', 'synrfp-full-fa-gpu1-v2.service'):
        if state(unit) not in ('inactive', 'unknown'):
            return False
    plan_path = PRIMARY_ROOT / 'plan.json'
    seal_path = PRIMARY_ROOT / 'plan_seal.json'
    if not plan_path.exists() or not seal_path.exists():
        return False
    plan_hash = sha(plan_path)
    if plan_hash != json.loads(seal_path.read_text())['sha256']:
        raise RuntimeError('Primary factorial plan changed')
    expected = {'classification': 5040, 'activity': 1080, 'yield': 4608}
    for task, cells in expected.items():
        receipt = PRIMARY_EXPORT / task / 'COMPLETE.json'
        if not receipt.exists():
            return False
        value = json.loads(receipt.read_text())
        if not (value.get('phase') == 'complete' and value.get('task') == task
                and value.get('cells') == cells and value.get('plan_sha256') == plan_hash
                and value.get('canonical_metric_and_artifact_checks_passed') is True):
            return False
    return True


def lease_available(gpu):
    path = HERE.parent / '12_validation/runs' / ('gpu_' + str(gpu) + '.lock')
    if not path.exists():
        raise RuntimeError('Missing existing global GPU lease')
    with path.open('a+') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(lease, fcntl.LOCK_UN)
    return True


def watch(manifest):
    ROOT.mkdir(parents=True, exist_ok=True)
    coordinator_lock = (ROOT / 'coordinator.lock').open('a+')
    fcntl.flock(coordinator_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    spec = json.loads(Path(manifest).read_text())
    sealed = sha(manifest)
    for path, digest in spec['sources_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('Changed validated queue implementation: ' + path)
    write(ROOT / 'manifest_seal.json', {'sha256': sealed, 'manifest': str(Path(manifest).resolve()),
                                      'scheduler_source_sha256': sha(__file__), 'created_utc': now()})
    while True:
        if sha(manifest) != sealed:
            raise RuntimeError('Changed follow-up queue manifest')
        value = snapshot()
        pressure = pressure_guard(value)
        decisions = []
        for job in spec['preparation']:
            if receipt_complete(job['done']):
                decisions.append({'unit': job['unit'], 'phase': 'preparation_complete'})
                continue
            if pressure['units']:
                decisions.append({'unit': job['unit'], 'phase': 'waiting_memory_recovery'})
                continue
            decisions.append(launch(job))
        primary_done = primary_gpu_complete()
        for job in spec['gpu']:
            if receipt_complete(job['done']):
                decisions.append({'unit': job['unit'], 'phase': 'benchmark_complete'})
            elif pressure['units']:
                decisions.append({'unit': job['unit'], 'phase': 'waiting_memory_recovery'})
            elif not primary_done:
                decisions.append({'unit': job['unit'], 'phase': 'waiting_primary_model_audit'})
            elif not receipt_complete(job['ready']):
                decisions.append({'unit': job['unit'], 'phase': 'waiting_preparation'})
            elif not all(receipt_complete(path) for path in job.get('dependencies', [])):
                decisions.append({'unit': job['unit'], 'phase': 'waiting_previous_followup_shard'})
            elif state(job['unit']) not in ('active', 'activating') and not lease_available(job['gpu']):
                decisions.append({'unit': job['unit'], 'phase': 'waiting_global_gpu_lease'})
            else:
                decisions.append(launch(job))
        value = snapshot()
        value.update({'phase': 'running_or_queued', 'decisions': decisions,
                      'primary_model_audit_complete': primary_done, 'manifest_sha256': sealed,
                      'memory_pressure_guard': pressure,
                      'xgboost_jobs': 0})
        if value['cpu_cap_sum'] > value['cpu_budget_equivalents'] + 1e-8 or value['memory_cap_sum_bytes'] > value['memory_budget_bytes']:
            value['phase'] = 'resource_violation_no_new_admission'
        write(ROOT / 'status.json', value)
        write(EXPORT / 'status.json', value)
        if all(receipt_complete(job['done']) for job in spec['gpu']):
            write(ROOT / 'COMPLETE.json', {'phase': 'complete', 'finished_utc': now(),
                                          'manifest_sha256': sealed})
            break
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--snapshot', action='store_true')
    args = parser.parse_args()
    if args.snapshot:
        print(json.dumps(snapshot(), indent=2))
    elif args.manifest:
        watch(args.manifest)
    else:
        parser.error('Specify --snapshot or --manifest')
