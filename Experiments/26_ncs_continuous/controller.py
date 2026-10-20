"""Persistent, resource-capped continuation of already authorized NCS studies.

This controller never imports a training framework, changes a study plan,
deletes an artifact, or owns a CUDA lease. Existing producers retain claims
and physical-device locks. Only this versioned scheduler admits new services.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ROOT = Path.home() / '.local/share/synrfp-runs/ncs_continuous_v1'
EXPORT = HERE / 'publication_metadata/ncs_continuous_v1'
spec = importlib.util.spec_from_file_location('continuous_frozen_budget',
    REPO / 'Experiments/25_followup_scheduler/budget.py')
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)
B.ROOT = ROOT
B.EXPORT = EXPORT
ACTIVE = {'active', 'activating', 'reloading', 'deactivating'}
BACKOFF = (30, 120, 600)


class IntegrityError(RuntimeError):
    pass


def read(path):
    try:
        value = json.loads(Path(path).read_text())
        if not isinstance(value, dict):
            raise ValueError('Expected a JSON object')
        return value
    except (OSError, ValueError) as e:
        raise IntegrityError('Malformed or unreadable receipt: ' + str(path)) from e


class Digests:
    """Full hashes remain valid only while all filesystem stat fields agree."""
    def __init__(self):
        self.cache = {}

    @staticmethod
    def signature(path):
        s = Path(path).stat()
        return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    def verify(self, path, expected):
        try:
            before = self.signature(path)
            prior = self.cache.get(str(path))
            if prior is None or prior[0] != before:
                digest = B.sha(path)
                if self.signature(path) != before:
                    raise IntegrityError('Input changed while hashing: ' + str(path))
                self.cache[str(path)] = (before, digest)
            if self.cache[str(path)][1] != expected:
                raise IntegrityError('Sealed input/source digest mismatch: ' + str(path))
        except (OSError, ValueError) as e:
            raise IntegrityError('Missing or invalid sealed input: ' + str(path)) from e


def task_receipt(path, task, cells, plan_hash, digests):
    if not Path(path).exists():
        return False
    v = read(path)
    if not (v.get('phase') == 'complete' and v.get('task') == task and
            v.get('cells') == cells and v.get('plan_sha256') == plan_hash and
            v.get('canonical_metric_and_artifact_checks_passed') is True):
        raise IntegrityError('Invalid canonical task completion: ' + str(path))
    if not isinstance(v.get('outputs_sha256'), dict) or not v['outputs_sha256']:
        raise IntegrityError('Missing audited report output hashes: ' + str(path))
    for name, digest in v['outputs_sha256'].items():
        digests.verify(Path(path).parent / name, digest)
    return True


def gate(spec, digests):
    """Completion is a receipt contract, never a service state or idle GPU."""
    kind = spec['kind']
    if kind == 'all':
        return all(gate(s, digests) for s in spec['items'])
    if kind == 'exists_all':
        # Work inventory only. It cannot authorize the next GPU phase.
        return all(Path(p).is_file() for p in spec['paths'])
    if kind == 'task':
        return task_receipt(spec['path'], spec['task'], spec['cells'], spec['plan_sha256'], digests)
    path = Path(spec['path'])
    if not path.exists():
        return False
    value = read(path)
    for key, expected in spec.get('equals', {}).items():
        if value.get(key) != expected:
            raise IntegrityError('Invalid completion/readiness field ' + key + ': ' + str(path))
    if spec.get('passed_counts') and value.get('passed') != value.get('total'):
        raise IntegrityError('Failed readiness checks: ' + str(path))
    if spec.get('passed_counts') and not value.get('total'):
        raise IntegrityError('Missing readiness check count: ' + str(path))
    mappings = [('outputs_sha256', path.parent, False)]
    mappings += [(h['field'], Path(h.get('base', path.parent)), True) for h in spec.get('hash_maps', [])]
    for field, base, required in mappings:
        hashes = value.get(field, {} if not required else None)
        if not isinstance(hashes, dict) or (required and not hashes):
            raise IntegrityError('Malformed receipt hash mapping ' + field + ': ' + str(path))
        for name, digest in hashes.items():
            digests.verify(base / name, digest)
    for h in spec.get('hash_values', []):
        if not isinstance(value.get(h['field']), str):
            raise IntegrityError('Missing receipt hash value: ' + str(path))
        digests.verify(h['path'], value[h['field']])
    return True


def props(unit):
    keys = ['ActiveState', 'LoadState', 'Result', 'ExecMainStatus', 'InvocationID',
            'MainPID', 'ControlGroup', 'CPUUsageNSec', 'MemoryCurrent', 'FreezerState',
            'ExecStart', 'Environment', 'CPUQuotaPerSecUSec', 'MemoryMax', 'KillMode', 'WorkingDirectory']
    r = subprocess.run(['systemctl', '--user', 'show', unit,
                        '--property=' + ','.join(keys)], capture_output=True, text=True, check=True)
    return dict(line.split('=', 1) for line in r.stdout.splitlines() if '=' in line)


def attest_service(job, properties):
    """Check command and hard caps before adopting or restarting loaded units."""
    if properties.get('LoadState') != 'loaded':
        return
    match = re.search(r'argv\[\]=(.*?) ; ignore_errors=', properties.get('ExecStart', ''))
    actual = shlex.split(match.group(1)) if match else []
    expected = ['/usr/bin/taskset', '-c', job['cores'], job.get('python', B.PYTHON), '-u', *job['arguments']]
    env = dict(item.split('=', 1) for item in shlex.split(properties.get('Environment', '')))
    quota = properties.get('CPUQuotaPerSecUSec', '')
    unit = next((u for u in ('us', 'ms', 's') if quota.endswith(u)), None)
    try:
        cpu = float(quota[:-len(unit)]) * {'us': 1e-6, 'ms': 1e-3, 's': 1}[unit] if unit else -1
        memory = int(properties.get('MemoryMax', '-1'))
    except (ValueError, KeyError):
        raise IntegrityError('Unparseable resource caps: ' + job['unit'])
    required_env = job.get('attest_environment', job.get('environment', {}))
    if (actual != expected or abs(cpu - job['cpu_equivalents']) > 1e-8 or
            memory != job['memory_gib'] * B.GIB or properties.get('KillMode') != 'control-group' or
            properties.get('WorkingDirectory') != str(REPO) or
            any(env.get(k) != v for k, v in required_env.items())):
        raise IntegrityError('Loaded service differs from sealed command/caps: ' + job['unit'])


def fatal_reason(result, log):
    if result == 'oom-kill':
        return 'cgroup_oom'
    patterns = {
        'integrity_failure': r'(?:ValueError|IntegrityError|RuntimeError):[^\n]*(?:changed|mismatch|digest|hash|identity|provenance|sealed|gate changed)',
        'cuda_failure': r'(?:CUDA error|CUDA out of memory|invalid device argument|device-side assert|CUDA driver|no CUDA|CUDA unavailable)',
        'environment_failure': r'(?:ModuleNotFoundError|ImportError|undefined symbol)',
        'bounded_worker_failure': r'(?:Three consecutive worker failures|exhausted bounded retry)',
    }
    for reason, pattern in patterns.items():
        if re.search(pattern, log, re.IGNORECASE):
            return reason
    return None


def next_action(done, state, held, latched, failures, failed_at, now):
    """Pure decision boundary used by the independent scheduler tests."""
    if done:
        return 'work_complete'
    if latched:
        return 'latched'
    if held:
        return 'user_hold'
    if state in ACTIVE:
        return 'running'
    if failures >= len(BACKOFF):
        return 'retry_budget_exhausted'
    if failures and now < failed_at + BACKOFF[failures - 1]:
        return 'retry_backoff'
    return 'admit'


def event(value):
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'events.jsonl').open('a') as out:
        out.write(json.dumps({**value, 'updated_utc': B.now()}, allow_nan=False) + '\n')


def latch(job, reason, details):
    path = ROOT / 'latches' / (job['id'] + '.json')
    if not path.exists():
        value = dict(phase='latched', job=job['id'], reason=reason, details=details,
                     updated_utc=B.now(), all_artifacts_preserved=True)
        B.write(path, value)
        event(value)


def log_positions(job):
    out = {}
    for path in job.get('logs', []):
        p = Path(path)
        if p.exists():
            s = p.stat()
            out[str(p)] = [s.st_dev, s.st_ino, s.st_size]
    return out


def failure_logs(job, properties, prior):
    invocation = properties.get('InvocationID')
    cmd = ['journalctl', '--user', '--no-pager', '-n', '100', '-o', 'cat']
    if invocation:
        cmd += ['_SYSTEMD_INVOCATION_ID=' + invocation]
    else:
        cmd += ['-u', job['unit']]
    logs = subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    # Producers may redirect worker stderr; retain a bounded tail in the event.
    for path in job.get('logs', []):
        p = Path(path)
        if p.exists():
            old = prior.get('log_positions', {}).get(str(p))
            current = p.stat()
            if old is None:
                continue  # No proven current invocation boundary: preserve, don't classify.
            offset = old[2] if [current.st_dev, current.st_ino] == old[:2] else 0
            with p.open('rb') as stream:
                stream.seek(max(offset, current.st_size - 16000))
                logs += '\n' + stream.read().decode(errors='replace')
    return logs[-40000:]


def ensure(job, digests, held=False):
    for p, digest in job['sources_sha256'].items():
        digests.verify(p, digest)
    done = gate(job['work_done'], digests)
    state = props(job['unit'])
    attest_service(job, state)
    path = ROOT / 'attempts' / (job['id'] + '.json')
    prior = read(path) if path.exists() else {'failures': 0, 'last_failure_epoch': 0, 'last_invocation': None}
    current = state.get('ActiveState', 'unknown')
    if not done and current not in ACTIVE and not held:
        invocation = state.get('InvocationID') or prior.get('last_invocation')
        crashed = current == 'failed' or (prior.get('was_active') and current != 'active')
        if crashed and invocation != prior.get('counted_failure'):
            logs = failure_logs(job, state, prior)
            reason = fatal_reason(state.get('Result'), logs)
            prior.update(failures=prior['failures'] + 1, last_failure_epoch=time.time(),
                         counted_failure=invocation, last_failure_logs=logs,
                         last_failure_result=state.get('Result'))
            event(dict(job=job['id'], phase='unexpected_exit', failure_count=prior['failures'], reason=reason))
            if reason:
                latch(job, reason, {'unit': job['unit'], 'logs': logs, 'result': state.get('Result')})
    latched = (ROOT / 'latches' / (job['id'] + '.json')).exists()
    action = next_action(done, current, held, latched, prior['failures'], prior['last_failure_epoch'], time.time())
    if action == 'retry_budget_exhausted':
        latch(job, action, prior)
    if action == 'admit':
        if any(B.state(u) in ACTIVE for u in job.get('exclusive_units', [])):
            action = 'waiting_current_gpu_owner'
        elif 'gpu' in job and not B.lease_available(job['gpu']):
            action = 'waiting_physical_gpu_lease'
        else:
            value = B.snapshot()
            reasons = B.admission(value, job['cpu_equivalents'], job['memory_gib'] * B.GIB)
            if reasons:
                action = 'waiting_resources'
            else:
                prior['log_positions'] = log_positions(job)
                # Stop the complete cgroup before restarting a failed parent;
                # a surviving child must not lose its parent's GPU lease.
                if state.get('LoadState') == 'loaded':
                    subprocess.run(['systemctl', '--user', 'stop', job['unit']], check=True)
                    subprocess.run(['systemctl', '--user', 'reset-failed', job['unit']], check=False,
                                   capture_output=True)
                    subprocess.run(['systemctl', '--user', 'start', job['unit']], check=True)
                else:
                    launched = B.launch(job)
                    if launched['phase'] not in ('launched', 'already_active'):
                        action = launched['phase']
                if action == 'admit':
                    action = 'resumed_or_launched'
                    event(dict(job=job['id'], unit=job['unit'], phase=action,
                               failure_count=prior['failures'], retained_artifacts=True))
                state = props(job['unit'])
                current = state.get('ActiveState', 'unknown')
                attest_service(job, state)
    if current in ACTIVE and 'log_positions' not in prior:
        prior['log_positions'] = log_positions(job)
    prior.update(was_active=current in ACTIVE, last_invocation=state.get('InvocationID') or prior.get('last_invocation'),
                 updated_utc=B.now(), last_action=action)
    B.write(path, prior)
    return dict(job=job['id'], unit=job['unit'], phase=action, failures=prior['failures'], service=state)


def pressure(value, jobs):
    """Freeze managed research cgroups at the reserve; never discard state."""
    path = ROOT / 'pressure.json'
    prior = read(path) if path.exists() else {'units': []}
    units = set(prior['units'])
    low_memory = value['memory_available_bytes'] < .25 * value['memory_total_bytes']
    low_disk = value['disk_free_bytes'] < 40 * B.GIB
    recovered = (value['memory_available_bytes'] >= .30 * value['memory_total_bytes'] and
                 value['disk_free_bytes'] >= 50 * B.GIB)
    if low_memory or low_disk:
        for job in jobs:
            if B.state(job['unit']) in ACTIVE and job['unit'] not in units:
                if freeze_or_thaw('freeze', job['unit']):
                    units.add(job['unit'])
    elif recovered:
        for unit in units:
            if B.state(unit) in ACTIVE:
                freeze_or_thaw('thaw', unit)
        units.clear()
    result = dict(phase='paused_at_resource_reserve' if units else 'normal', units=sorted(units),
                  low_memory=low_memory, low_disk=low_disk, updated_utc=B.now())
    if units != set(prior['units']):
        event(result)
    B.write(path, result)
    return result


def freeze_or_thaw(action, unit):
    result = subprocess.run(['systemctl', '--user', action, unit], capture_output=True, text=True)
    if result.returncode and B.state(unit) in ACTIVE:
        raise RuntimeError('Could not ' + action + ' live research cgroup: ' + unit)
    return result.returncode == 0


def stages(manifest, digests):
    gates = {name: gate(rule, digests) for name, rule in manifest['gates'].items()}
    # Current producer release matters even after receipt publication.
    primary_released = not any(B.state(u) in ACTIVE for u in manifest['primary_gpu_units'])
    reagent_released = not any(B.state(u) in ACTIVE for u in manifest['reagent_gpu_units'])
    frea_released = not any(B.state(u) in ACTIVE for u in manifest['frea_gpu_units'])
    syncat_released = not any(B.state(u) in ACTIVE for u in manifest['syncat_gpu_units'])
    return dict(primary=True, cluster_assist=gates['features'] and B.state(manifest['encoder_unit']) not in ACTIVE,
                reagent=gates['primary_gpu'] and primary_released,
                frea=gates['reagent'] and reagent_released,
                syncat=gates['frea'] and frea_released,
                yield_refs0=gates['frea'] and frea_released,
                yield_refs1=gates['frea'] and frea_released and gates['syncat'] and syncat_released), gates


def cycle(manifest, digests):
    value = B.snapshot()
    reserve = pressure(value, manifest['jobs'])
    enabled, gates = stages(manifest, digests)
    decisions = []
    for job in manifest['jobs']:
        held = any(Path(p).exists() for p in job.get('holds', []))
        if reserve['units']:
            decisions.append(dict(job=job['id'], phase='waiting_resource_recovery'))
        elif not enabled[job['stage']]:
            decisions.append(dict(job=job['id'], phase='waiting_prior_audited_stage'))
        else:
            if any(not gate(rule, digests) for rule in job.get('ready', [])):
                decisions.append(dict(job=job['id'], phase='waiting_readiness_receipt'))
                continue
            decisions.append(ensure(job, digests, held))
    done = all(gates[n] for n in ('primary_all', 'reagent', 'frea', 'syncat', 'yield_refs'))
    return dict(phase='complete' if done else 'running', updated_utc=B.now(),
                resources=B.snapshot(), pressure=reserve, gates=gates, stages_enabled=enabled,
                jobs=decisions, latches=[str(p) for p in sorted((ROOT / 'latches').glob('*.json'))],
                handoff_poll_seconds=30, stop_at_monday=False, checkpoints_retained_locally=True)


def watch(manifest_path, once=False):
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'coordinator.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = read(manifest_path)
        seal = read(ROOT / 'manifest_seal.json')['sha256']
        digests = Digests()
        while True:
            try:
                if (ROOT / 'latches/global_integrity.json').exists():
                    if once:
                        return
                    time.sleep(30)
                    continue
                digests.verify(manifest_path, seal)
                for p, digest in manifest['sources_sha256'].items():
                    digests.verify(p, digest)
                status = cycle(manifest, digests)
            except IntegrityError as e:
                latch({'id': 'global_integrity'}, 'integrity_failure', str(e))
                for job in manifest['jobs']:
                    if B.state(job['unit']) in ACTIVE:
                        freeze_or_thaw('freeze', job['unit'])
                status = dict(phase='integrity_latched', updated_utc=B.now(), error=str(e),
                              all_artifacts_preserved=True)
            except (RuntimeError, subprocess.CalledProcessError) as e:
                status = dict(phase='waiting_resource_or_service_audit', updated_utc=B.now(),
                              error=str(e), all_artifacts_preserved=True)
                event(status)
            B.write(ROOT / 'status.json', status)
            B.write(EXPORT / 'status.json', status)
            if status['phase'] == 'complete':
                B.write(ROOT / 'COMPLETE.json', status)
                B.write(EXPORT / 'COMPLETE.json', status)
                return
            if once:
                return
            # An integrity latch requires explicit repair in a new version.
            # Keep the supervisor alive to expose status without restart loops.
            if (ROOT / 'latches/global_integrity.json').exists():
                while True:
                    time.sleep(30)
            time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=ROOT / 'queue.json')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    watch(args.manifest, args.once)
