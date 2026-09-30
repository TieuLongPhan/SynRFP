#!/usr/bin/env python
"""Resume missing benchmark/factorial MLP fits across independent CUDA workers.

Run with the synrfp Python environment; --gpu-python selects the CUDA runtime.
Existing classification results are retained. New results record backend and
training provenance. Touch CONTROL/STOP to stop scheduling after current fits.
"""
import argparse
import fcntl
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synrfp_exp.config import load_config, expand_fingerprints, SynRFPConfig, results_dir
from synrfp_exp.data import dataset_table, rxn_key
from synrfp_exp.features import load_for_run
from synrfp_exp.metrics import classification
from synrfp_exp.scopes import full_scope, factorial_scope
from synrfp_exp.splits import make_split, random_codes, TRAIN, VAL, TEST


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, default=float))
    temp.replace(path)


def prepare_arrays(directory, Xtr, ytr, Xte, params, seed, identifier):
    """Export exact sklearn initialization and internal validation/RNG state."""
    assert len(params['hidden']) == 1, 'Only single-hidden-layer MLP supported'
    lb = LabelBinarizer().fit(ytr)
    target = lb.transform(ytr).astype(bool)
    if len(lb.classes_) < 2:
        raise ValueError('MLP needs at least two training classes')
    clf = MLPClassifier(early_stopping=True, random_state=seed)
    clf._label_binarizer = lb
    clf._random_state = np.random.RandomState(seed)
    clf._initialize(target, [Xtr.shape[1], params['hidden'][0], target.shape[1]], Xtr.dtype)
    train, val = train_test_split(np.arange(len(Xtr)), test_size=0.1,
                                  random_state=clf._random_state,
                                  stratify=target if target.shape[1] == 1 else None)
    state = clf._random_state.get_state()
    np.save(directory / 'train.npy', Xtr)
    np.save(directory / 'test.npy', Xte)
    np.savez(directory / 'state.npz', train=train, val=val,
             y=np.searchsorted(lb.classes_, ytr),
             w0=clf.coefs_[0], w1=clf.coefs_[1], b0=clf.intercepts_[0], b1=clf.intercepts_[1],
             rng_keys=state[1], rng_pos=state[2], rng_has_gauss=state[3], rng_gauss=state[4])
    write_json(directory / 'job.json', dict(id=identifier, params=params))
    return lb.classes_


def inventory(cfg, studies=('benchmark', 'factorial')):
    jobs, blocked, completed = [], [], 0
    root = results_dir('03_classification')
    for study in studies:
        spec = cfg[study]
        datasets = list(cfg['datasets']) if study == 'benchmark' else spec['datasets']
        for dataset in datasets:
            # The existing prepared tables define currently available datasets.
            if not (cfg['paths']['work'] / 'tables' / ('ds_%s.pkl' % dataset)).exists():
                blocked.append(dict(study=study, dataset=dataset, reason='missing dataset table'))
                continue
            ds = dataset_table(cfg, dataset)
            if study == 'benchmark':
                scope = rxn_key(cfg, dataset)
                fps = expand_fingerprints(cfg, spec['fingerprints'])
                runs = [(scheme, seed, fp) for scheme, seeds in spec['schemes'].items()
                        for seed in seeds for fp in fps + (
                            expand_fingerprints(cfg, cfg.get('rxnfp_dataset_fingerprints', {}).get(dataset, []))
                            if scheme == 'fixed' else [])]
            else:
                cap = spec['max_records']
                scope = '%s__sub%d' % (dataset, cap) if len(ds) > cap else rxn_key(cfg, dataset)
                seeds = spec.get('seeds_per_model', {}).get('mlp', spec['seeds'])
                runs = [('random', seed, fp) for seed in seeds
                        for fp in expand_fingerprints(cfg, ['synrfp_grid'])]
            for scheme, seed, fp in runs:
                output = root / study / dataset / ('%s_s%d' % (scheme, seed)) / (fp + '__mlp.json')
                if output.exists():
                    data = json.loads(output.read_text())
                    if data.get('model') != 'mlp' or 'mcc' not in data:
                        raise ValueError('Existing result is invalid: %s' % output)
                    completed += 1
                    continue
                feature = cfg['paths']['work'] / 'features' / scope / (fp + '.npz')
                if not feature.exists():
                    blocked.append(dict(study=study, dataset=dataset, fingerprint=fp,
                                        scheme=scheme, seed=seed, reason='missing features'))
                    continue
                jobs.append(dict(study=study, dataset=dataset, scheme=scheme, seed=seed,
                                 fingerprint=fp, output=str(output), scope=scope,
                                 records=min(len(ds), spec['max_records']) if study == 'factorial' else len(ds)))
    # Long datasets first to avoid leaving a single large job as the final tail.
    jobs.sort(key=lambda j: (-j['records'], j['study'], j['dataset'], j['fingerprint'], j['seed']))
    return jobs, blocked, completed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config')
    ap.add_argument('--gpu-python', type=Path, required=True)
    ap.add_argument('--devices', type=int, nargs='+', default=[0, 1])
    ap.add_argument('--studies', choices=['benchmark', 'factorial'], nargs='+',
                    default=['benchmark', 'factorial'])
    ap.add_argument('--control', type=Path, default=ROOT / '03_classification/results/mlp_gpu_queue')
    ap.add_argument('--scratch', type=Path, default=Path('/tmp'))
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--limit', type=int)
    args = ap.parse_args()
    if len(set(args.devices)) != len(args.devices):
        ap.error('GPU devices must be unique')
    cfg = load_config(args.config)
    args.control.mkdir(parents=True, exist_ok=True)
    lock_file = (args.control / 'queue.lock').open('w')
    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs, blocked, completed = inventory(cfg, args.studies)
    if args.limit:
        jobs = jobs[:args.limit]
    write_json(args.control / 'inventory.json', dict(jobs=jobs, blocked=blocked,
                                                    already_completed=completed))
    print('Pending=%d completed=%d blocked_entries=%d' % (len(jobs), completed, len(blocked)), flush=True)
    if args.dry_run:
        return
    if (args.control / 'STOP').exists():
        raise RuntimeError('Remove the STOP file to resume scheduling')
    run_started = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    state = dict(pid=os.getpid(), started_utc=run_started, initial_pending=len(jobs),
                 already_completed=completed, finished=0, failed=0, active={}, status='running')
    state_lock, split_lock = threading.Lock(), threading.Lock()
    work = queue.Queue()
    for job in jobs:
        work.put(job)
    source_hash = hashlib.sha256(Path(__file__).read_bytes() +
                                 Path(__file__).with_name('mlp_cuda_worker.py').read_bytes()).hexdigest()
    write_json(args.control / 'run_config.json', dict(params=cfg['models']['mlp'], devices=args.devices,
                studies=args.studies,
                source_sha256=source_hash, config_sha256=hashlib.sha256(Path(cfg['_config_path']).read_bytes()).hexdigest(),
                gpu_python=str(args.gpu_python), started_utc=run_started))
    shutil.copyfile(__file__, args.control / 'mlp_gpu_queue.snapshot.py')
    shutil.copyfile(Path(__file__).with_name('mlp_cuda_worker.py'), args.control / 'mlp_cuda_worker.snapshot.py')

    def update():
        state['elapsed_seconds'] = time.perf_counter() - started
        state['remaining'] = state['initial_pending'] - state['finished'] - state['failed']
        write_json(args.control / 'status.json', state)

    def worker(device):
        proc = None
        scratch = Path(tempfile.mkdtemp(prefix='synrfp-mlp-gpu%d-' % device, dir=args.scratch))
        cache_key = None
        try:
            env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=':4096:8',
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
            with (args.control / ('gpu%d.log' % device)).open('a', buffering=1) as log:
                proc = subprocess.Popen([str(args.gpu_python), '-u',
                    str(Path(__file__).with_name('mlp_cuda_worker.py')), '--device', str(device)],
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, env=env)
                ready = json.loads(proc.stdout.readline())
                if not ready.get('ready'):
                    raise RuntimeError('GPU worker failed to initialize')
                while not (args.control / 'STOP').exists():
                    try:
                        job = work.get_nowait()
                    except queue.Empty:
                        break
                    job_start = time.perf_counter()
                    identifier = '%s/%s/%s_s%d/%s' % (job['study'], job['dataset'],
                                      job['scheme'], job['seed'], job['fingerprint'])
                    with state_lock:
                        state['active'][str(device)] = dict(id=identifier, started_utc=datetime.now(timezone.utc).isoformat())
                        update()
                    try:
                        key = (job['study'], job['dataset'])
                        if key != cache_key:
                            ds = dataset_table(cfg, job['dataset'])
                            scope = (full_scope if job['study'] == 'benchmark' else factorial_scope)(cfg, job['dataset'])
                            y = ds['y'].to_numpy()[scope.ds_pos]
                            cache_key = key
                        with split_lock:
                            codes = (make_split(cfg, job['dataset'], job['scheme'], job['seed'])
                                if job['study'] == 'benchmark' else random_codes(cfg, y, job['seed']))
                        outer = np.concatenate([np.flatnonzero(codes == TRAIN), np.flatnonzero(codes == VAL)])
                        test = np.flatnonzero(codes == TEST)
                        feats = load_for_run(cfg, scope.name, job['fingerprint'], job['dataset'], job['scheme'], job['seed'])
                        Xtr, Xte = feats.dense(scope.feat_rows[outer]), feats.dense(scope.feat_rows[test])
                        if not feats.is_bits:
                            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
                            Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
                        classes = prepare_arrays(scratch, Xtr, y[outer], Xte,
                                                  cfg['models']['mlp'], job['seed'], identifier)
                        n_bad = int((~feats.ok[scope.feat_rows[test]]).sum())
                        del Xtr, Xte, feats
                        preparation_seconds = time.perf_counter() - job_start
                        proc.stdin.write(json.dumps(dict(directory=str(scratch))) + '\n')
                        proc.stdin.flush()
                        response = json.loads(proc.stdout.readline())
                        if not response.get('ok'):
                            raise RuntimeError(response.get('error', 'GPU worker failed'))
                        pred = classes[np.load(scratch / 'predictions.npy')]
                        rec = {k: job[k] for k in ['study', 'dataset', 'scheme', 'seed', 'fingerprint']}
                        rec.update(model='mlp', backend='torch_cuda', protocol='sklearn_adam_fp32_v1',
                                   n_train=len(outer), n_val=int((codes == VAL).sum()), n_test=len(test),
                                   n_unfeaturised_test=n_bad, **classification(y[test], pred),
                                   **response['result'], preparation_seconds=preparation_seconds,
                                   elapsed_seconds=time.perf_counter() - job_start,
                                   training_params=cfg['models']['mlp'], source_sha256=source_hash,
                                   completed_utc=datetime.now(timezone.utc).isoformat())
                        if job['fingerprint'].startswith('synrfp-'):
                            c = SynRFPConfig.from_name(job['fingerprint'])
                            rec.update(tokenizer=c.tokenizer, radius=c.radius, bits=c.bits, sketch=c.sketch, mode=c.mode)
                        output = Path(job['output'])
                        # Publish without overwriting a result another runner may have produced.
                        output.parent.mkdir(parents=True, exist_ok=True)
                        tmp = output.with_suffix('.gpu%d.tmp' % device)
                        write_json(tmp, rec)
                        try:
                            os.link(tmp, output)
                        except FileExistsError:
                            pass
                        finally:
                            tmp.unlink()
                        with state_lock:
                            state['finished'] += 1
                            with (args.control / 'completed.jsonl').open('a') as f:
                                f.write(json.dumps(dict(id=identifier, output=str(output), device=device,
                                    seconds=rec['elapsed_seconds'], gpu_seconds=rec['gpu_seconds'], epochs=rec['epochs'])) + '\n')
                            print('DONE %d/%d gpu%d %.1fs %s' % (state['finished'], state['initial_pending'],
                                                               device, rec['elapsed_seconds'], identifier), flush=True)
                    except Exception:
                        error = traceback.format_exc()
                        with state_lock:
                            state['failed'] += 1
                            with (args.control / 'errors.jsonl').open('a') as f:
                                f.write(json.dumps(dict(id=identifier, error=error)) + '\n')
                        print(error, flush=True)
                        if proc.poll() is not None:
                            raise RuntimeError('CUDA worker exited; inspect GPU log')
                    finally:
                        for name in ['train.npy', 'test.npy', 'state.npz', 'predictions.npy', 'job.json']:
                            (scratch / name).unlink(missing_ok=True)
                        with state_lock:
                            state['active'].pop(str(device), None)
                            update()
                        work.task_done()
        except Exception:
            with state_lock:
                state.setdefault('worker_errors', {})[str(device)] = traceback.format_exc()
                update()
            print(traceback.format_exc(), flush=True)
        finally:
            if proc is not None:
                if proc.stdin:
                    proc.stdin.close()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait()
            shutil.rmtree(scratch)

    update()
    threads = [threading.Thread(target=worker, args=(device,)) for device in args.devices]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    state['status'] = ('paused' if (args.control / 'STOP').exists() else
                       'incomplete' if state['failed'] or state['remaining'] or state.get('worker_errors') else 'complete')
    update()
    print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
