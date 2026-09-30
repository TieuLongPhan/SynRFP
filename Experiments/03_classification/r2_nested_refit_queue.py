#!/usr/bin/env python3
"""Run selected r2 nested outer refits concurrently on two GPUs."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CUDA_PY = Path('/home/labhhc4/anaconda3/envs/rxnfp/bin/python')
REFIT = Path(__file__).with_name('r2_nested_refit.py')


def save_status(path, jobs, active, started):
    done = sum(j['out'].exists() for j in jobs)
    failed = sum(j['rc'] not in (None, 0) for j in jobs)
    rec = {'study': 'r2_nested_grouped_mlp_v1', 'stage': 'outer_refit',
           'total': len(jobs), 'done': done, 'failed': failed,
           'active': len(active), 'pending': len(jobs) - done - len(active),
           'elapsed_seconds': time.time() - started,
           'jobs': [{k: v for k, v in j.items() if k not in {'out', 'proc', 'log'}} | {'out': str(j['out'])} for j in jobs]}
    path.write_text(json.dumps(rec, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--poll-seconds', type=float, default=2)
    args = ap.parse_args()
    selection = json.loads((args.out / 'selection.json').read_text())['selected']
    refit_dir = args.out / 'outer_refit'
    jobs = []
    for entry in selection:
        s = entry['selected']; fp = entry['fingerprint']; outer = entry['outer_seed']
        short = 'synrfp' if fp.startswith('synrfp') else 'drfp'
        out = refit_dir / f'outer{outer:02d}_{short}.json'
        jobs.append({'outer': outer, 'fingerprint': fp, 'out': out, 'params': s,
                     'proc': None, 'device': None, 'rc': None})
    refit_dir.mkdir(parents=True, exist_ok=True)
    status_path = args.out / 'refit_status.json'; started = time.time(); active = {}
    while True:
        for gpu, job in list(active.items()):
            rc = job['proc'].poll()
            if rc is not None:
                job['rc'] = rc
                del active[gpu]
                if rc:
                    print(f'FAILED gpu={gpu} outer={job["outer"]} fp={job["fingerprint"]} rc={rc}', flush=True)
        for gpu in range(2):
            if gpu in active:
                continue
            job = next((j for j in jobs if not j['out'].exists() and j['proc'] is None), None)
            if job is None:
                continue
            s = job['params']
            cmd = [str(CUDA_PY), '-u', str(REFIT), '--fingerprint', job['fingerprint'],
                   '--outer-seed', str(job['outer']), '--family', s['family'],
                   '--aux-weight', str(s['aux_weight']), '--lr', str(s['lr']),
                   '--epochs', str(s['refit_epochs']), '--device', f'cuda:{gpu}', '--out', str(job['out'])]
            log = job['out'].with_suffix('.log').open('w')
            job['proc'] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
            job['device'] = gpu; job['log'] = log; active[gpu] = job
            print(f'START gpu={gpu} outer={job["outer"]} fp={job["fingerprint"]}', flush=True)
        save_status(status_path, jobs, active, started)
        if not active and all(j['out'].exists() or j['proc'] is not None for j in jobs):
            break
        time.sleep(args.poll_seconds)
    for j in jobs:
        if 'log' in j: j['log'].close()
    save_status(status_path, jobs, active, started)
    if any(j['rc'] not in (None, 0) for j in jobs):
        sys.exit(1)
    print(f'COMPLETE {sum(j["out"].exists() for j in jobs)}/{len(jobs)}', flush=True)

if __name__ == '__main__':
    main()
