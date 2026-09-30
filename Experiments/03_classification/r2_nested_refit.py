#!/usr/bin/env python
"""Train a preselected r2 MLP on an outer development pool and score its test fold."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2_grouped_dev_trial import AuxiliaryHead, PlainHead, predict

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprint", required=True)
    ap.add_argument("--outer-seed", type=int, required=True)
    ap.add_argument("--family", choices=["plain", "auxiliary"], required=True)
    ap.add_argument("--aux-weight", type=float, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--device", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        return
    data = pd.read_csv(WORK / "export" / "ds_syntemp_r2.csv.gz", usecols=["row", "y"])
    rows, y = data["row"].to_numpy(), data["y"].to_numpy(dtype=np.int64)
    code = np.load(WORK / "splits" / "syntemp_r2" / ("cv_s%d.npy" % args.outer_seed))
    train, test = np.flatnonzero(code != 2), np.flatnonzero(code == 2)
    z = np.load(WORK / "features" / "syntemp" / (args.fingerprint + ".npz"))
    X = np.unpackbits(z["X"][rows], axis=1, count=int(z["n_bits"])).astype(np.float32)
    y1 = None
    if args.family == "auxiliary":
        coarse = pd.read_csv(WORK / "export" / "ds_syntemp_r1.csv.gz", usecols=["row", "y"])
        if not np.array_equal(rows, coarse["row"].to_numpy()):
            raise ValueError("r1/r2 export row mismatch")
        y1 = coarse["y"].to_numpy(dtype=np.int64)
    np.random.seed(args.outer_seed); torch.manual_seed(args.outer_seed)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)
    Xtr, Xte = torch.tensor(X[train], device=dev), torch.tensor(X[test], device=dev)
    ytr = torch.tensor(y[train], device=dev)
    if args.family == "plain":
        model = PlainHead(X.shape[1], int(y.max()) + 1, 1024, 0.1)
    else:
        model = AuxiliaryHead(X.shape[1], int(y1.max()) + 1, int(y.max()) + 1, 1024, 0.1)
        y1tr = torch.tensor(y1[train], device=dev)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(); rng = np.random.RandomState(args.outer_seed); order = np.arange(len(train)); started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if epoch in (81, 141):
            for group in opt.param_groups: group["lr"] *= 0.5
        model.train(); rng.shuffle(order)
        for start in range(0, len(order), 256):
            pos = torch.tensor(order[start:start + 256], device=dev)
            opt.zero_grad(); output = model(Xtr[pos])
            if args.family == "auxiliary":
                r2, r1 = output; loss = ce(r2, ytr[pos]) + args.aux_weight * ce(r1, y1tr[pos])
            else: loss = ce(output, ytr[pos])
            loss.backward(); opt.step()
    model.eval(); pred = predict(model, Xte)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out.with_suffix(".predictions.npz"), row=rows[test], y_true=y[test], y_pred=pred)
    rec = {"study": "r2_nested_grouped_mlp_v1", "stage": "outer_refit", "dataset": "syntemp_r2",
           "fingerprint": args.fingerprint, "outer_seed": args.outer_seed, "family": args.family,
           "aux_weight": args.aux_weight, "lr": args.lr, "refit_epochs": args.epochs, "n_train": len(train),
           "n_test": len(test), "mcc": float(matthews_corrcoef(y[test], pred)),
           "accuracy": float(accuracy_score(y[test], pred)), "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
           "macro_f1": float(f1_score(y[test], pred, average="macro", zero_division=0)),
           "seconds": time.perf_counter()-started, "device": args.device}
    tmp=args.out.with_suffix(".tmp"); tmp.write_text(json.dumps(rec,indent=2)); tmp.replace(args.out)
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
