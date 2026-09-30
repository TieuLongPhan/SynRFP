#!/usr/bin/env python
"""Validation-only shared-head MLP pilot for the three SynTemp hierarchies.

``prepare`` runs in the synrfp Python 3.11 environment. ``run`` runs in the
CUDA Python 3.8 environment. Test labels are never used for model selection.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FP = "synrfp-wl-r2-b1024-parity-delta"


def prepare(out, seed):
    sys.path.insert(0, str(ROOT))
    from synrfp_exp.config import load_config
    from synrfp_exp.data import dataset_table
    from synrfp_exp.features import load
    from synrfp_exp.splits import make_split

    cfg = load_config()
    features = load(cfg, "syntemp", FP)
    y_tasks = []
    code = None
    for level in range(3):
        dataset = "syntemp_r%d" % level
        ds = dataset_table(cfg, dataset)
        y = np.full(len(features.X), -1, dtype=np.int64)
        y[ds["row"].to_numpy()] = ds["y"].to_numpy()
        y_tasks.append(y)
        if level == 0:
            code = make_split(cfg, dataset, "cv", seed)
            aligned = np.full(len(features.X), -1, dtype=np.int8)
            aligned[ds["row"].to_numpy()] = code
            code = aligned
    X = features.dense()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, y=np.stack(y_tasks), code=code)
    print("Prepared", out, "X", X.shape, "valid all levels", int((np.stack(y_tasks) >= 0).all(0).sum()))


def run(data, out, device, seed, epochs, patience, batch_size):
    import torch
    from sklearn.metrics import matthews_corrcoef
    from torch import nn

    z = np.load(data)
    X, y, code = z["X"], z["y"], z["code"]
    valid = (y >= 0).all(0)
    tr = np.flatnonzero((code == 0) & valid)
    va = np.flatnonzero((code == 1) & valid)
    dims = [int(y[i].max()) + 1 for i in range(3)]
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(device)
    Xtr, Xva = torch.tensor(X[tr], device=dev), torch.tensor(X[va], device=dev)
    ytr = [torch.tensor(y[i, tr], device=dev) for i in range(3)]

    class HierarchyHead(nn.Module):
        def __init__(self):
            super(HierarchyHead, self).__init__()
            self.shared = nn.Sequential(nn.Linear(X.shape[1], 1024), nn.GELU(), nn.Dropout(0.2))
            self.heads = nn.ModuleList([nn.Linear(1024, dim) for dim in dims])
        def forward(self, value):
            hidden = self.shared(value)
            return [head(hidden) for head in self.heads]

    model = HierarchyHead().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=3e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)
    order = np.arange(len(tr))
    best, best_epoch, stale, state = -np.inf, 0, 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        np.random.shuffle(order)
        for start in range(0, len(order), batch_size):
            idx = torch.tensor(order[start:start + batch_size], device=dev)
            opt.zero_grad()
            outputs = model(Xtr[idx])
            loss = sum(loss_fn(outputs[i], ytr[i][idx]) for i in range(3)) / 3
            loss.backward()
            opt.step()
        model.eval()
        preds = [[] for _ in range(3)]
        with torch.no_grad():
            for start in range(0, len(va), batch_size):
                outputs = model(Xva[start:start + batch_size])
                for i in range(3):
                    preds[i].append(outputs[i].argmax(1).cpu())
        mccs = [float(matthews_corrcoef(y[i, va], torch.cat(preds[i]).numpy())) for i in range(3)]
        mean_mcc = float(np.mean(mccs))
        history.append(dict(mean=mean_mcc, r0=mccs[0], r1=mccs[1], r2=mccs[2]))
        if mean_mcc > best + 1e-4:
            best, best_epoch, stale = mean_mcc, epoch + 1, 0
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        print("epoch=%d val_mcc r0=%.6f r1=%.6f r2=%.6f mean=%.6f" %
              (epoch + 1, mccs[0], mccs[1], mccs[2], mean_mcc), flush=True)
        if stale > patience:
            break
    result = dict(study="multitask_hierarchy_pilot", dataset="syntemp", cv_seed=seed,
                  feature=FP, architecture="shared 1024 GELU dropout 0.2 + three task heads",
                  selection_partition="outer validation only", test_evaluated=False,
                  best_epoch=best_epoch, epochs=epoch + 1, validation_mean_mcc=best,
                  validation_mcc=dict(r0=history[best_epoch - 1]["r0"], r1=history[best_epoch - 1]["r1"],
                                      r2=history[best_epoch - 1]["r2"]), history=history,
                  seconds=time.perf_counter() - started, n_train=len(tr), n_validation=len(va),
                  task_dimensions=dims, device=device)
    temp = out.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.data, args.seed)
    else:
        if args.out is None:
            parser.error("--out is required for run")
        run(args.data, args.out, args.device, args.seed, args.epochs, args.patience, args.batch_size)
