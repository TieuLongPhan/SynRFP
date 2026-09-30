#!/usr/bin/env python
"""GPU-only MLP head selection on fixed SynRFP features.

This is a separate study: candidates use the outer training partition and the
outer validation partition only. Test predictions are disabled by default so
test folds cannot select hyperparameters. Run under the rxnfp Python 3.8
environment, which provides CUDA PyTorch but need not import SynRFP.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import train_test_split
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
FP = "synrfp-wl-r2-b1024-parity-delta"

CANDIDATES = {
    # Reference-sized head, trained against the outer validation partition.
    "baseline": {"hidden": 512, "dropout": 0.0, "lr": 1e-3, "weight_decay": 1e-4,
                 "label_smoothing": 0.0, "activation": "relu"},
    "wide_relu": {"hidden": 1024, "dropout": 0.1, "lr": 1e-3, "weight_decay": 1e-4,
                  "label_smoothing": 0.0, "activation": "relu"},
    "wide_gelu": {"hidden": 1024, "dropout": 0.1, "lr": 7e-4, "weight_decay": 1e-4,
                  "label_smoothing": 0.02, "activation": "gelu"},
    "regularized": {"hidden": 1024, "dropout": 0.2, "lr": 7e-4, "weight_decay": 3e-4,
                    "label_smoothing": 0.05, "activation": "gelu"},
    # r2-specific candidates.  The deepest hierarchy has 680 severely
    # imbalanced labels, so selection is by MCC and these vary only the head
    # and loss while holding the SynRFP representation fixed.
    "deep_relu": {"hidden": [1024, 512], "dropout": 0.1, "lr": 7e-4, "weight_decay": 1e-4,
                  "label_smoothing": 0.0, "activation": "relu", "loss": "ce"},
    "sqrt_balanced": {"hidden": 1024, "dropout": 0.1, "lr": 7e-4, "weight_decay": 1e-4,
                      "label_smoothing": 0.0, "activation": "relu", "loss": "sqrt_balanced"},
    "focal": {"hidden": 1024, "dropout": 0.1, "lr": 7e-4, "weight_decay": 1e-4,
              "label_smoothing": 0.0, "activation": "relu", "loss": "focal"},
}


def scope(dataset):
    return {"schneider_ub": "schneider_u", "schneider_b": "schneider_b",
            "tpl_ub": "tpl_u", "tpl_b": "tpl_b", "uspto50k_ub": "uspto_50k_u",
            "uspto50k_b": "uspto_50k_b", "syntemp_r0": "syntemp",
            "syntemp_r1": "syntemp", "syntemp_r2": "syntemp",
            "ecreact_1": "ecreact", "ecreact_2": "ecreact", "ecreact_3": "ecreact"}[dataset]


def load_arrays(dataset, seed):
    # CSV export is deliberately used here: CUDA PyTorch remains on Python 3.8,
    # while the main experiment tables are Python-3.11 pickles.
    ds = pd.read_csv(WORK / "export" / ("ds_%s.csv.gz" % dataset), usecols=["row", "y"])
    features = np.load(WORK / "features" / scope(dataset) / (FP + ".npz"))
    rows = ds["row"].to_numpy()
    X = np.unpackbits(features["X"][rows], axis=1, count=int(features["n_bits"])).astype(np.float32)
    y = ds["y"].to_numpy(dtype=np.int64)
    code = np.load(WORK / "splits" / dataset / ("cv_s%d.npy" % seed))
    return X, y, code


class Head(nn.Module):
    def __init__(self, inputs, outputs, params):
        super(Head, self).__init__()
        act = nn.ReLU() if params["activation"] == "relu" else nn.GELU()
        widths = params["hidden"] if isinstance(params["hidden"], list) else [params["hidden"]]
        layers, prior = [], inputs
        for width in widths:
            layers.extend([nn.Linear(prior, width), act, nn.Dropout(params["dropout"])])
            prior = width
        layers.append(nn.Linear(prior, outputs))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def predict(model, X, batch):
    parts = []
    with torch.no_grad():
        for start in range(0, len(X), batch):
            parts.append(model(X[start:start + batch]).argmax(1).cpu())
    return torch.cat(parts).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="tpl_ub")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--final-train-outer", action="store_true",
                        help="After selection, refit on outer train+validation; reserve a new 10%% inner stop set.")
    args = parser.parse_args()
    params = CANDIDATES[args.candidate]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        print("Already exists:", args.out, flush=True)
        return
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    X, y, code = load_arrays(args.dataset, args.seed)
    tr, va, te = (np.flatnonzero(code == value) for value in [0, 1, 2])
    if args.final_train_outer:
        outer = np.concatenate([tr, va])
        tr, va = train_test_split(outer, test_size=0.1, random_state=args.seed)
    Xtr, Xva = torch.tensor(X[tr], device=device), torch.tensor(X[va], device=device)
    ytr = torch.tensor(y[tr], device=device)
    yva = y[va]
    model = Head(X.shape[1], int(y.max()) + 1, params).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    loss_kind = params.get("loss", "ce")
    if loss_kind == "sqrt_balanced":
        counts = np.bincount(y[tr], minlength=int(y.max()) + 1).astype(np.float32)
        # Square-root inverse weighting moderates the 1,800-fold raw imbalance;
        # capping prevents the rarest retained class from dominating a batch.
        weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        weights = np.minimum(weights / weights.mean(), 5.0)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=params["label_smoothing"])
    order = np.arange(len(tr))
    best_mcc, best_epoch, stale, best = -np.inf, 0, 0, None
    history = []
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        np.random.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            idx = torch.tensor(order[start:start + args.batch_size], device=device)
            opt.zero_grad()
            logits = model(Xtr[idx])
            if loss_kind == "focal":
                ce = nn.functional.cross_entropy(logits, ytr[idx], reduction="none")
                loss = ((1.0 - torch.exp(-ce)) ** 1.0 * ce).mean()
            else:
                loss = loss_fn(logits, ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        pred = predict(model, Xva, args.batch_size)
        mcc = float(matthews_corrcoef(yva, pred))
        history.append(mcc)
        if mcc > best_mcc + 1e-4:
            best_mcc, best_epoch, stale = mcc, epoch + 1, 0
            best = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        print("%s %s epoch=%d val_mcc=%.6f elapsed=%.1fs" %
              (args.dataset, args.candidate, epoch + 1, mcc, time.perf_counter() - started), flush=True)
        if stale > args.patience:
            break
    model.load_state_dict(best)
    result = {"study": "mlp_model_selection", "dataset": args.dataset, "cv_seed": args.seed,
              "candidate": args.candidate, "params": params, "selection_partition": "outer validation only",
              "final_train_outer": args.final_train_outer, "test_evaluated": args.evaluate_test,
              "epochs": epoch + 1, "best_epoch": best_epoch,
              "validation_mcc": best_mcc, "history": history,
              "seconds": time.perf_counter() - started, "device": str(device),
              "n_train": len(tr), "n_validation": len(va), "n_test": len(te)}
    if args.evaluate_test:
        Xte = torch.tensor(X[te], device=device)
        result["test_mcc"] = float(matthews_corrcoef(y[te], predict(model, Xte, args.batch_size)))
    temp = args.out.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
