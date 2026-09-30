#!/usr/bin/env python
"""Isolated sklearn CPU / torch CUDA MLP pilot; prepare with synrfp Python.

CUDA worker is compatible with the existing Python 3.8 rxnfp environment.
Results never enter the main benchmark. Uses cached Track A 5x5 splits.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def prepare(out):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelBinarizer
    from sklearn.model_selection import train_test_split
    from synrfp_exp.config import load_config
    from synrfp_exp.data import dataset_table
    from synrfp_exp.features import load_for_run
    from synrfp_exp.scopes import full_scope
    from synrfp_exp.splits import make_split, TRAIN, VAL, TEST

    cfg = load_config()
    dataset, fp = 'schneider_ub', 'synrfp-wl-r2-b1024-parity-delta'
    ds, scope = dataset_table(cfg, dataset), full_scope(cfg, dataset)
    feats = load_for_run(cfg, scope.name, fp, dataset, 'cv', 0)
    assert feats.is_bits, 'This pilot is scoped to default SynRFP bits'
    X = feats.dense(scope.feat_rows)
    classes, y = np.unique(ds['y'].to_numpy(), return_inverse=True)
    np.save(out / 'X.npy', X)
    np.save(out / 'y.npy', y)
    p = cfg['models']['mlp']
    assert p['hidden'] == [512]
    save_json(out / 'manifest.json', dict(dataset=dataset, fingerprint=fp,
              records=len(y), classes=len(classes), params=p,
              protocol='5 repeats x 5 product-scaffold-grouped folds',
              cpu_threads=4, gpu='cuda:1', precision='float32; TF32 disabled'))
    for seed in range(25):
        code = make_split(cfg, dataset, 'cv', seed)
        outer = np.concatenate([np.flatnonzero(code == TRAIN), np.flatnonzero(code == VAL)])
        test = np.flatnonzero(code == TEST)
        lb = LabelBinarizer().fit(y[outer])
        target = lb.transform(y[outer]).astype(bool)
        assert target.shape[1] > 2, 'CUDA pilot implements multiclass softmax only'
        clf = MLPClassifier(early_stopping=True, random_state=seed)
        clf._label_binarizer = lb
        clf._random_state = np.random.RandomState(seed)
        clf._initialize(target, [X.shape[1], 512, target.shape[1]], X.dtype)
        train, val = train_test_split(np.arange(len(outer)), test_size=0.1,
                                     random_state=clf._random_state,
                                     stratify=target if target.shape[1] == 1 else None)
        state = clf._random_state.get_state()
        np.savez(out / ('split_%02d.npz' % seed), outer=outer, test=test,
                 train=train, val=val, classes=lb.classes_,
                 w0=clf.coefs_[0], w1=clf.coefs_[1],
                 b0=clf.intercepts_[0], b1=clf.intercepts_[1],
                 rng_keys=state[1], rng_pos=state[2], rng_has_gauss=state[3], rng_gauss=state[4])
    print('Prepared 25 matched splits:', out, flush=True)


def run(out, backend, seed, max_epochs=None):
    result_path = out / ('%s_%02d.json' % (backend, seed))
    p = json.loads((out / 'manifest.json').read_text())['params']
    max_epochs = max_epochs or p['max_iter']
    if result_path.exists():
        if json.loads(result_path.read_text())['max_epochs'] != max_epochs:
            raise ValueError('Use a separate output directory for different epoch limits')
        print('Already completed', result_path, flush=True)
        return
    X, y = np.load(out / 'X.npy', mmap_mode='r'), np.load(out / 'y.npy')
    z = np.load(out / ('split_%02d.npz' % seed))
    outer, test = z['outer'], z['test']
    if backend == 'cpu':
        from sklearn.neural_network import MLPClassifier
        from threadpoolctl import threadpool_limits
        import sklearn
        clf = MLPClassifier(hidden_layer_sizes=tuple(p['hidden']), alpha=p['alpha'],
                            max_iter=max_epochs, batch_size=p['batch_size'],
                            early_stopping=True, validation_fraction=0.1,
                            n_iter_no_change=10, random_state=seed, verbose=True)
        start = time.perf_counter()
        with threadpool_limits(limits=4):
            clf.fit(X[outer], y[outer])
            fit_seconds = time.perf_counter() - start
            pred = clf.predict(X[test])
        epochs, val_scores = clf.n_iter_, clf.validation_scores_
        version = sklearn.__version__
    else:
        import torch
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        device = 'cuda:1'
        # Initialize CUDA before timed transfer + training + inference.
        torch.zeros(1, device=device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        xt = torch.tensor(X[outer[z['train']]], device=device)
        xv = torch.tensor(X[outer[z['val']]], device=device)
        xe = torch.tensor(X[test], device=device)
        classes = z['classes']
        yt = torch.tensor(np.searchsorted(classes, y[outer[z['train']]]), device=device)
        yv = torch.tensor(y[outer[z['val']]], device=device)
        ct = torch.tensor(classes, device=device)
        params = [torch.tensor(z[k], device=device) for k in ['w0', 'w1', 'b0', 'b1']]
        ms, vs = [torch.zeros_like(t) for t in params], [torch.zeros_like(t) for t in params]
        rng = np.random.RandomState()
        rng.set_state(('MT19937', z['rng_keys'], int(z['rng_pos']),
                       int(z['rng_has_gauss']), float(z['rng_gauss'])))
        order = np.arange(len(xt))
        best_score, stale, step = -np.inf, 0, 0
        val_scores = []
        with torch.no_grad():
            for epoch in range(max_epochs):
                order = rng.permutation(order)
                idx = torch.tensor(order, device=device)
                w0, w1, b0, b1 = params
                for offset in range(0, len(order), p['batch_size']):
                    batch = idx[offset:offset + p['batch_size']]
                    xb, yb = xt[batch], yt[batch]
                    hidden = (xb @ w0 + b0).relu()
                    delta = (hidden @ w1 + b1).softmax(1)
                    delta[torch.arange(len(batch), device=device), yb] -= 1
                    dh = (delta @ w1.T) * (hidden > 0)
                    n = len(batch)
                    grads = [(xb.T @ dh + p['alpha'] * w0) / n,
                             (hidden.T @ delta + p['alpha'] * w1) / n,
                             dh.mean(0), delta.mean(0)]
                    step += 1
                    lr = 0.001 * np.sqrt(1 - 0.999 ** step) / (1 - 0.9 ** step)
                    for param, m, v, grad in zip(params, ms, vs, grads):
                        m.mul_(0.9).add_(grad, alpha=0.1)
                        v.mul_(0.999).addcmul_(grad, grad, value=0.001)
                        param.addcdiv_(m, v.sqrt().add_(1e-8), value=-lr)
                prediction = ct[((xv @ w0 + b0).relu() @ w1 + b1).argmax(1)]
                score = (prediction == yv).float().mean().item()
                val_scores.append(score)
                stale = stale + 1 if score < best_score + 1e-4 else 0
                if score > best_score:
                    best_score = score
                    best = [t.clone() for t in params]
                print('gpu seed=%d epoch=%d validation=%.6f elapsed=%.1fs' %
                      (seed, epoch + 1, score, time.perf_counter() - start), flush=True)
                if stale > 10:
                    break
            torch.cuda.synchronize()
            fit_seconds = time.perf_counter() - start
            w0, w1, b0, b1 = best
            pred = ct[((xe @ w0 + b0).relu() @ w1 + b1).argmax(1)].cpu().numpy()
        epochs, version = epoch + 1, torch.__version__
    total = time.perf_counter() - start
    np.save(out / ('%s_%02d_predictions.npy' % (backend, seed)), pred)
    save_json(result_path, dict(backend=backend, seed=seed, repeat=seed // 5,
              fold=seed % 5, epochs=epochs, fit_seconds=fit_seconds,
              total_seconds=total, validation_scores=val_scores, version=version,
              max_epochs=max_epochs, test_accuracy=float(np.mean(pred == y[test]))))
    print('COMPLETE', result_path, 'seconds', total, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['prepare', 'cpu', 'gpu'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=list(range(25)))
    ap.add_argument('--max-epochs', type=int)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.action == 'prepare':
        prepare(args.out)
    else:
        for seed in args.seeds:
            run(args.out, args.action, seed, args.max_epochs)
