#!/usr/bin/env python
"""Validation-only r2-primary MLP with an optional r1 auxiliary head.

The representation is unchanged (the default 1,024-bit SynRFP).  r2 is the
scored task; r1 merely supplies a small auxiliary cross-entropy signal.  This
is intentionally separate from the equal-weight all-level pilot.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
FP = "synrfp-wl-r2-b1024-parity-delta"


def load(seed):
    labels = []
    rows = None
    for dataset in ["syntemp_r1", "syntemp_r2"]:
        data = pd.read_csv(WORK / "export" / ("ds_%s.csv.gz" % dataset), usecols=["row", "y"])
        if rows is None:
            rows = data["row"].to_numpy()
        elif not np.array_equal(rows, data["row"].to_numpy()):
            raise RuntimeError("SynTemp hierarchy rows are not aligned")
        labels.append(data["y"].to_numpy(dtype=np.int64))
    f = np.load(WORK / "features" / "syntemp" / (FP + ".npz"))
    X = np.unpackbits(f["X"][rows], axis=1, count=int(f["n_bits"])).astype(np.float32)
    code = np.load(WORK / "splits" / "syntemp_r2" / ("cv_s%d.npy" % seed))
    return X, labels[0], labels[1], code


class Net(nn.Module):
    def __init__(self, n1, n2):
        super(Net, self).__init__()
        self.shared = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(0.1))
        self.r1, self.r2 = nn.Linear(1024, n1), nn.Linear(1024, n2)
    def forward(self, x):
        x = self.shared(x)
        return self.r1(x), self.r2(x)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--aux-weight", type=float, required=True)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    args = p.parse_args()
    if args.out.exists():
        print("Already exists:", args.out)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    X, y1, y2, code = load(args.seed)
    tr, va = (np.flatnonzero(code == v) for v in [0, 1])
    dev = torch.device(args.device)
    Xtr, Xva = torch.tensor(X[tr], device=dev), torch.tensor(X[va], device=dev)
    a1, a2 = torch.tensor(y1[tr], device=dev), torch.tensor(y2[tr], device=dev)
    net = Net(int(y1.max()) + 1, int(y2.max()) + 1).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=7e-4, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    order, history, best, stale, snapshot = np.arange(len(tr)), [], -np.inf, 0, None
    started = time.perf_counter()
    for epoch in range(args.epochs):
        net.train(); np.random.shuffle(order)
        for start in range(0, len(order), 1024):
            ii = torch.tensor(order[start:start + 1024], device=dev)
            opt.zero_grad(); o1, o2 = net(Xtr[ii])
            loss = ce(o2, a2[ii]) + args.aux_weight * ce(o1, a1[ii])
            loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred = torch.cat([net(Xva[start:start + 4096])[1].argmax(1).cpu()
                              for start in range(0, len(Xva), 4096)]).numpy()
        score = float(matthews_corrcoef(y2[va], pred)); history.append(score)
        if score > best + 1e-4:
            best, best_epoch, stale = score, epoch + 1, 0
            snapshot = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else: stale += 1
        if stale > args.patience: break
    result = dict(study="r2_aux_hierarchy_pilot", feature=FP, cv_seed=args.seed,
                  aux_weight=args.aux_weight, selection_partition="outer validation only",
                  test_evaluated=False, validation_mcc=best, best_epoch=best_epoch,
                  epochs=epoch + 1, seconds=time.perf_counter()-started, history=history)
    tmp = args.out.with_suffix(".tmp"); tmp.write_text(json.dumps(result, indent=2)); tmp.replace(args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
