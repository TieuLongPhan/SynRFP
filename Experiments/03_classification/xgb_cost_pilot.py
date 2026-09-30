#!/usr/bin/env python
"""Bounded CPU XGBoost timing probes; partial models never enter benchmark results."""
import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synrfp_exp.config import load_config, write_json, default_config
from synrfp_exp.data import dataset_table
from synrfp_exp.features import load_for_run
from synrfp_exp.scopes import full_scope
from synrfp_exp.splits import make_split, TRAIN, VAL, TEST
from synrfp_exp.metrics import classification


class Timer(xgb.callback.TrainingCallback):
    def __init__(self, max_seconds):
        self.max_seconds = max_seconds
        self.round_seconds = []
        self.stopped_by_time = False

    def before_training(self, model):
        self.start = self.last = time.perf_counter()
        return model

    def after_iteration(self, model, epoch, evals_log):
        now = time.perf_counter()
        self.round_seconds.append(now - self.last)
        self.last = now
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print('round=%d elapsed=%.1fs validation=%.6f' %
                  (epoch + 1, now - self.start, evals_log['validation_0']['mlogloss'][-1]), flush=True)
        self.stopped_by_time = now - self.start >= self.max_seconds
        return self.stopped_by_time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--fingerprint', default='synrfp_default')
    ap.add_argument('--rounds', type=int, default=20)
    ap.add_argument('--threads', type=int, default=16)
    ap.add_argument('--seconds', type=float, default=120)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cfg = load_config()
    fp = default_config(cfg).name if args.fingerprint == 'synrfp_default' else args.fingerprint
    ds, sc = dataset_table(cfg, args.dataset), full_scope(cfg, args.dataset)
    code = make_split(cfg, args.dataset, 'cv', 0)
    y = ds['y'].to_numpy()
    feats = load_for_run(cfg, sc.name, fp, args.dataset, 'cv', 0)
    tr, va, te = (np.flatnonzero(code == c) for c in [TRAIN, VAL, TEST])
    classes, yt = np.unique(y[tr], return_inverse=True)
    va = va[np.isin(y[va], classes)]
    yv = np.searchsorted(classes, y[va])
    Xtr, Xva, Xte = (feats.dense(sc.feat_rows[i]) for i in [tr, va, te])
    if not feats.is_bits:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xva, Xte = [(a - mu) / sd for a in [Xtr, Xva, Xte]]
    params = cfg['models']['xgb']
    timer = Timer(args.seconds)
    clf = xgb.XGBClassifier(objective='multi:softprob', tree_method='hist', device='cpu',
                           n_estimators=args.rounds, learning_rate=params['learning_rate'],
                           max_depth=params['max_depth'], early_stopping_rounds=params['early_stopping_rounds'],
                           eval_metric='mlogloss', random_state=0, n_jobs=args.threads, callbacks=[timer])
    print('START', args.dataset, fp, 'train', len(tr), 'classes', len(classes),
          'round_cap', args.rounds, 'threads', args.threads, flush=True)
    started = time.perf_counter()
    clf.fit(Xtr, yt, eval_set=[(Xva, yv)], verbose=False)
    fit_seconds = time.perf_counter() - started
    pred = classes[clf.predict(Xte).astype(int)]
    total_seconds = time.perf_counter() - started
    rounds = clf.get_booster().num_boosted_rounds()
    extrapolated = float(np.median(timer.round_seconds[1:] or timer.round_seconds))
    result = dict(dataset=args.dataset, fingerprint=fp, scheme='cv', seed=0,
                  n_train=len(tr), n_val=len(va), n_test=len(te), classes=len(classes),
                  features=Xtr.shape[1], threads=args.threads, rounds=rounds,
                  requested_rounds=args.rounds, stopped_by_time=timer.stopped_by_time,
                  full_protocol_fit=args.rounds == params['n_estimators'] and not timer.stopped_by_time,
                  fit_seconds=fit_seconds, total_seconds=total_seconds,
                  round_seconds=timer.round_seconds, median_later_round_seconds=extrapolated,
                  projected_500_round_seconds=fit_seconds + max(0, 500-rounds)*extrapolated,
                  best_iteration=int(clf.best_iteration), max_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                  version=xgb.__version__, training_params=params, **classification(y[te],pred))
    write_json(args.out, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
