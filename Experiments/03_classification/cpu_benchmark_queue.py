#!/usr/bin/env python
"""Run benchmark kNN to completion, then benchmark XGBoost, with a result gate.

Missing datasets/features are listed separately and excluded from the available
workload. A failed or incomplete kNN stage never starts XGBoost. Existing fits
are resumed by classify.py. No factorial jobs or MLP fits are launched here.
"""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synrfp_exp.config import load_config, expand_fingerprints, results_dir
from synrfp_exp.data import rxn_key


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def expected_results(cfg, model):
    expected, unavailable = [], []
    root = results_dir('03_classification') / 'benchmark'
    fps = expand_fingerprints(cfg, cfg['benchmark']['fingerprints'])
    for dataset in cfg['datasets']:
        work = cfg['paths']['work']
        if not (work / 'tables' / ('ds_%s.pkl' % dataset)).exists():
            unavailable.append(dict(dataset=dataset, reason='missing dataset table'))
            continue
        scope = rxn_key(cfg, dataset)
        for scheme, seeds in cfg['benchmark']['schemes'].items():
            extra = expand_fingerprints(cfg, cfg.get('rxnfp_dataset_fingerprints', {}).get(dataset, [])) if scheme == 'fixed' else []
            for seed in seeds:
                if model == 'knn' and scheme == 'fixed' and seed != min(seeds):
                    continue
                for fp in fps + extra:
                    output = root / dataset / ('%s_s%d' % (scheme, seed)) / (fp + '__%s.json' % model)
                    if not (work / 'features' / scope / (fp + '.npz')).exists():
                        unavailable.append(dict(dataset=dataset, scheme=scheme, seed=seed,
                                                fingerprint=fp, reason='missing features'))
                        continue
                    expected.append(output)
    return expected, unavailable


def missing_results(paths, model):
    missing = []
    for path in paths:
        try:
            result = json.loads(path.read_text())
            valid = result.get('model') == model and 'mcc' in result
        except (FileNotFoundError, json.JSONDecodeError):
            valid = False
        if not valid:
            missing.append(str(path))
    return missing


def partition_datasets(cfg, missing, workers):
    """Disjoint dataset assignments, balancing approximate quadratic kNN cost."""
    counts = Counter(Path(p).parent.parent.name for p in missing)
    weighted = []
    for dataset, count in counts.items():
        meta = json.loads((cfg['paths']['work'] / 'tables' / ('ds_%s.json' % dataset)).read_text())
        weighted.append((count * meta['records_kept'] ** 2, dataset))
    buckets = [[] for _ in range(min(workers, len(weighted)))]
    costs = [0] * len(buckets)
    for cost, dataset in sorted(weighted, reverse=True):
        index = min(range(len(costs)), key=costs.__getitem__)
        buckets[index].append(dataset)
        costs[index] += cost
    return buckets


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config')
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--knn-workers', type=int, default=1)
    ap.add_argument('--control', type=Path, default=ROOT / '03_classification/results/cpu_benchmark_queue')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    if args.threads < 1 or args.knn_workers < 1:
        ap.error('threads and worker count must be positive')
    args.control = args.control.resolve()
    args.control.mkdir(parents=True, exist_ok=True)
    lock = (args.control / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cfg = load_config(args.config)
    expected = {}
    inventory = {}
    for model in ['knn', 'xgb']:
        expected[model], unavailable = expected_results(cfg, model)
        missing = missing_results(expected[model], model)
        inventory[model] = dict(available=len(expected[model]), completed=len(expected[model])-len(missing),
                                remaining=len(missing), unavailable=unavailable)
    write_json(args.control / 'inventory.json', inventory)
    print(json.dumps(inventory, indent=2), flush=True)
    if args.dry_run:
        return
    # Pin absolute inputs and CPU resources without changing the live GPU queue.
    with Path(cfg['_config_path']).open() as f:
        snapshot = yaml.safe_load(f)
    snapshot['paths'] = {key: str(value) for key, value in cfg['paths'].items()}
    snapshot['synrxn']['local_dir'] = str(cfg['paths']['synrxn_local'])
    snapshot['n_jobs'] = args.threads
    snapshot['datasets'] = {d: spec for d, spec in snapshot['datasets'].items()
                            if (cfg['paths']['work'] / 'tables' / ('ds_%s.pkl' % d)).exists()}
    snapshot_path = args.control / 'config.yaml'
    snapshot_path.write_text(yaml.safe_dump(snapshot, sort_keys=False))
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OPENBLAS_NUM_THREADS=str(args.threads),
               MKL_NUM_THREADS=str(args.threads), NUMEXPR_NUM_THREADS=str(args.threads))
    state = dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                 threads_per_worker=args.threads, knn_workers=args.knn_workers,
                 stage='knn', status='running', stages=inventory)
    for model in ['knn', 'xgb']:
        missing = missing_results(expected[model], model)
        if not missing:
            continue
        state.update(stage=model, status='running')
        groups = (partition_datasets(cfg, missing, args.knn_workers) if model == 'knn'
                  else [[d for d in snapshot['datasets'] if any(Path(p).parent.parent.name == d for p in missing)]])
        with ExitStack() as stack:
            processes = []
            state['workers'] = []
            for index, datasets in enumerate(groups):
                cmd = [sys.executable, '-u', str(Path(__file__).with_name('classify.py')),
                       '--config', str(snapshot_path), '--study', 'benchmark', '--models', model,
                       '--datasets', *datasets]
                logfile = args.control / ('%s_worker%d.log' % (model, index))
                log = stack.enter_context(logfile.open('a'))
                process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                                           stderr=subprocess.STDOUT, env=env)
                processes.append(process)
                state['workers'].append(dict(pid=process.pid, datasets=datasets, log=str(logfile)))
                print('START', model, 'PID', process.pid, 'datasets', datasets, flush=True)
            while True:
                remaining = missing_results(expected[model], model)
                state['stages'][model].update(completed=len(expected[model])-len(remaining), remaining=len(remaining))
                state['updated_utc'] = datetime.now(timezone.utc).isoformat()
                write_json(args.control / 'status.json', state)
                codes = [process.poll() for process in processes]
                if all(code is not None for code in codes):
                    break
                time.sleep(15)
        missing = missing_results(expected[model], model)
        state['stages'][model].update(completed=len(expected[model])-len(missing), remaining=len(missing))
        state.pop('workers', None)
        if any(code != 0 for code in codes) or missing:
            state.update(status='blocked', exit_codes=codes, missing_results=missing)
            write_json(args.control / 'status.json', state)
            print('STOP: stage failed or results incomplete; next stage not started', flush=True)
            return
        print('COMPLETE', model, flush=True)
    state.update(status='complete', stage='complete', completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(args.control / 'status.json', state)


if __name__ == '__main__':
    main()
