#!/usr/bin/env python
"""Summarize isolated MLP pilot with the current benchmark metrics environment."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synrfp_exp.metrics import classification


def summarize(out):
    y = np.load(out / 'y.npy')
    rows, repeats = [], []
    for backend in ['cpu', 'gpu']:
        pooled = {}
        for seed in range(25):
            path = out / ('%s_%02d.json' % (backend, seed))
            if not path.exists():
                continue
            rec = json.loads(path.read_text())
            rec.pop('validation_scores')
            test = np.load(out / ('split_%02d.npz' % seed))['test']
            pred = np.load(out / ('%s_%02d_predictions.npy' % (backend, seed)))
            rec.update(classification(y[test], pred))
            rows.append(rec)
            pooled.setdefault(seed // 5, []).append((test, pred))
        for repeat, folds in pooled.items():
            if len(folds) == 5:
                idx = np.concatenate([a for a, _ in folds])
                pred = np.concatenate([b for _, b in folds])
                assert len(idx) == len(y) and len(np.unique(idx)) == len(y)
                repeats.append(dict(backend=backend, repeat=repeat,
                                    **classification(y[idx], pred)))
    for name, data in [('folds.csv', rows), ('repeats.csv', repeats)]:
        if data:
            with (out / name).open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(data[0]))
                writer.writeheader()
                writer.writerows(data)
    report = {}
    for backend in ['cpu', 'gpu']:
        r = [x for x in rows if x['backend'] == backend]
        reps = [x for x in repeats if x['backend'] == backend]
        if not r:
            continue
        report[backend] = dict(completed_folds=len(r), total_seconds=sum(x['total_seconds'] for x in r),
                               total_epochs=sum(x['epochs'] for x in r))
        if reps:
            report[backend]['pooled_repeat_metrics'] = {
                k: dict(mean=float(np.mean([x[k] for x in reps])),
                        std=float(np.std([x[k] for x in reps], ddof=1)) if len(reps) > 1 else None)
                for k in ['mcc', 'accuracy', 'macro_f1', 'weighted_f1', 'balanced_accuracy']}
    paired = []
    for seed in range(25):
        match = {x['backend']: x for x in rows if x['seed'] == seed}
        if len(match) == 2:
            a = np.load(out / ('cpu_%02d_predictions.npy' % seed))
            b = np.load(out / ('gpu_%02d_predictions.npy' % seed))
            paired.append(dict(seed=seed, speedup=match['cpu']['total_seconds'] / match['gpu']['total_seconds'],
                               prediction_agreement=float(np.mean(a == b)),
                               cpu_seconds=match['cpu']['total_seconds'], gpu_seconds=match['gpu']['total_seconds']))
    if paired:
        report['paired'] = dict(folds=len(paired),
            aggregate_speedup=sum(x['cpu_seconds'] for x in paired) / sum(x['gpu_seconds'] for x in paired),
            mean_prediction_agreement=float(np.mean([x['prediction_agreement'] for x in paired])))
    (out / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    summarize(parser.parse_args().out)
