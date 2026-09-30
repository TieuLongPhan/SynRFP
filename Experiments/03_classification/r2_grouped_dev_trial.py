#!/usr/bin/env python
"""Grouped development-only MLP trial for a fixed SynTemp-r2 outer fold.

This script never reads the outer-test labels for model selection.  It selects
epochs by MCC on one of three scaffold-grouped partitions of the outer
development rows.  A separate launcher schedules the same trial for SynRFP
and DRFP so their trainer, rows, and training budget remain matched.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "work"
DATASET = "syntemp_r2"


def load(fingerprint, outer_seed, inner_split, inner_fold, aggregate_y, with_auxiliary):
    data = pd.read_csv(WORK / "export" / ("ds_%s.csv.gz" % DATASET), usecols=["row", "y"])
    rows, y = data["row"].to_numpy(), data["y"].to_numpy(dtype=np.int64)
    if aggregate_y >= 0 and aggregate_y > int(y.max()):
        raise ValueError("aggregate encoded label is outside the r2 label range")
    outer_codes = np.load(WORK / "splits" / DATASET / ("cv_s%d.npy" % outer_seed))
    dev = np.flatnonzero(outer_codes != 2)
    inner_codes = np.load(inner_split)
    if inner_codes.ndim != 2 or inner_codes.shape != (3, len(y)):
        raise ValueError("inner split is not aligned with SynTemp r2")
    inner_codes = inner_codes[inner_fold]
    train = np.flatnonzero(inner_codes == 10)
    val = np.flatnonzero(inner_codes == 0)
    if not len(train) or not len(val) or not np.all(outer_codes[np.r_[train, val]] != 2):
        raise ValueError("inner split does not define this outer development fold")
    z = np.load(WORK / "features" / "syntemp" / (fingerprint + ".npz"))
    X = np.unpackbits(z["X"][rows], axis=1, count=int(z["n_bits"])).astype(np.float32)
    y1 = None
    if with_auxiliary:
        coarse = pd.read_csv(WORK / "export" / "ds_syntemp_r1.csv.gz", usecols=["row", "y"])
        if not np.array_equal(rows, coarse["row"].to_numpy()):
            raise ValueError("r1 and r2 exports are not row-aligned")
        y1 = coarse["y"].to_numpy(dtype=np.int64)
    return X, y, y1, rows, train, val, outer_codes, aggregate_y


class PlainHead(nn.Module):
    def __init__(self, n_in, n_out, width, dropout):
        super(PlainHead, self).__init__()
        self.net = nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(width, n_out))

    def forward(self, x):
        return self.net(x)


class AggregateHead(nn.Module):
    """One aggregate-class logit plus a conditional distribution for all others."""
    def __init__(self, n_in, n_out, aggregate_y, width, dropout):
        super(AggregateHead, self).__init__()
        self.trunk = nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Dropout(dropout))
        self.aggregate = nn.Linear(width, 1)
        self.known = nn.Linear(width, n_out - 1)
        self.aggregate_y = aggregate_y
        self.n_out = n_out

    def forward(self, x):
        h = self.trunk(x)
        aggregate_logp = nn.functional.logsigmoid(self.aggregate(h))
        known_logp = nn.functional.logsigmoid(-self.aggregate(h)) + nn.functional.log_softmax(self.known(h), dim=1)
        out = x.new_empty((len(x), self.n_out))
        out[:, self.aggregate_y] = aggregate_logp[:, 0]
        out[:, torch.arange(self.n_out, device=x.device) != self.aggregate_y] = known_logp
        return out


class DeepHead(nn.Module):
    def __init__(self, n_in, n_out, width, dropout):
        super(DeepHead, self).__init__()
        self.net = nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(width, width // 2), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(width // 2, n_out))

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout),
                                    nn.Linear(width, width))

    def forward(self, x):
        return nn.functional.relu(x + self.layers(x))


class ResidualHead(nn.Module):
    def __init__(self, n_in, n_out, width, dropout):
        super(ResidualHead, self).__init__()
        self.input = nn.Sequential(nn.Linear(n_in, width), nn.ReLU())
        self.blocks = nn.Sequential(ResidualBlock(width, dropout), ResidualBlock(width, dropout))
        self.output = nn.Linear(width, n_out)

    def forward(self, x):
        return self.output(self.blocks(self.input(x)))


class AuxiliaryHead(nn.Module):
    def __init__(self, n_in, n_r1, n_r2, width, dropout):
        super(AuxiliaryHead, self).__init__()
        self.trunk = nn.Sequential(nn.Linear(n_in, width), nn.ReLU(), nn.Dropout(dropout))
        self.r2 = nn.Linear(width, n_r2)
        self.r1 = nn.Linear(width, n_r1)

    def forward(self, x):
        h = self.trunk(x)
        return self.r2(h), self.r1(h)


def predict(model, X, batch=8192):
    out = []
    with torch.no_grad():
        for start in range(0, len(X), batch):
            logits = model(X[start:start + batch])
            if isinstance(logits, tuple):
                logits = logits[0]
            out.append(logits.argmax(1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fingerprint", required=True,
                    choices=["synrfp-wl-r2-b1024-parity-delta", "drfp"])
    ap.add_argument("--outer-seed", type=int, default=0)
    ap.add_argument("--inner-split", type=Path, required=True)
    ap.add_argument("--inner-fold", type=int, choices=range(3), required=True)
    ap.add_argument("--family", choices=["plain", "aggregate", "deep", "residual", "auxiliary"], default="plain")
    ap.add_argument("--aux-weight", type=float, default=0.0)
    ap.add_argument("--aggregate-label", default="679", help="Source label recorded for provenance.")
    ap.add_argument("--aggregate-y", type=int, default=None,
                    help="Encoded aggregate class from the adjacent split manifest.")
    ap.add_argument("--width", type=int, choices=[512, 1024], required=True)
    ap.add_argument("--dropout", type=float, required=True)
    ap.add_argument("--batch-size", type=int, choices=[256, 1024], required=True)
    ap.add_argument("--lr", type=float, choices=[0.0003, 0.001], required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--init-seed", type=int, default=0)
    ap.add_argument("--device", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        print("Already exists:", args.out, flush=True)
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.init_seed)
    torch.manual_seed(args.init_seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.family == "aggregate" and args.aggregate_y is None:
        ap.error("--aggregate-y is required for the aggregate head")
    aggregate_y_arg = args.aggregate_y if args.aggregate_y is not None else -1
    X, y, y1, rows, train, val, outer_codes, aggregate_y = load(
        args.fingerprint, args.outer_seed, args.inner_split, args.inner_fold, aggregate_y_arg,
        args.family == "auxiliary"
    )
    dev = torch.device(args.device)
    Xtr, Xva = torch.tensor(X[train], device=dev), torch.tensor(X[val], device=dev)
    ytr, yva = torch.tensor(y[train], device=dev), y[val]
    if args.family == "plain":
        model = PlainHead(X.shape[1], int(y.max()) + 1, args.width, args.dropout)
    elif args.family == "aggregate":
        model = AggregateHead(X.shape[1], int(y.max()) + 1, aggregate_y, args.width, args.dropout)
    elif args.family == "deep":
        model = DeepHead(X.shape[1], int(y.max()) + 1, args.width, args.dropout)
    elif args.family == "auxiliary":
        model = AuxiliaryHead(X.shape[1], int(y1.max()) + 1, int(y.max()) + 1, args.width, args.dropout)
    else:
        model = ResidualHead(X.shape[1], int(y.max()) + 1, args.width, args.dropout)
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    y1tr = torch.tensor(y1[train], device=dev) if args.family == "auxiliary" else None
    rng = np.random.RandomState(args.init_seed)
    order = np.arange(len(train))
    best, best_epoch, stale, state, history = -np.inf, 0, 0, None, []
    steps = 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        if epoch in (81, 141):
            for group in opt.param_groups:
                group["lr"] *= 0.5
        model.train()
        rng.shuffle(order)
        for start in range(0, len(order), args.batch_size):
            pos = torch.tensor(order[start:start + args.batch_size], device=dev)
            opt.zero_grad()
            output = model(Xtr[pos])
            if args.family == "aggregate":
                loss = nn.functional.nll_loss(output, ytr[pos])
            elif args.family == "auxiliary":
                r2_logits, r1_logits = output
                loss = loss_fn(r2_logits, ytr[pos]) + args.aux_weight * loss_fn(r1_logits, y1tr[pos])
            else:
                loss = loss_fn(output, ytr[pos])
            loss.backward()
            opt.step()
            steps += 1
        model.eval()
        pred = predict(model, Xva)
        mcc = float(matthews_corrcoef(yva, pred))
        history.append(mcc)
        if mcc > best + 1e-4:
            best, best_epoch, stale = mcc, epoch, 0
            state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale > args.patience:
            break
    model.load_state_dict(state)
    pred = predict(model, Xva)
    # The companion predictions use only rows within the outer development set.
    np.savez_compressed(args.out.with_suffix(".predictions.npz"), row=rows[val], y_true=yva, y_pred=pred)
    result = {
        "study": "r2_grouped_development_screen_v1",
        "selection_partition": "scaffold-grouped inner validation inside outer development pool",
        "outer_test_read": False,
        "dataset": DATASET,
        "fingerprint": args.fingerprint,
        "outer_seed": args.outer_seed,
        "inner_split": str(args.inner_split),
        "inner_fold": args.inner_fold,
        "params": {"family": args.family, "width": args.width, "dropout": args.dropout,
                   "batch_size": args.batch_size, "lr": args.lr, "weight_decay": 1e-4,
                   "max_epochs": args.epochs, "patience": args.patience,
                   "lr_epoch_multipliers": {80: 0.5, 140: 0.5}, "aux_weight": args.aux_weight},
        "init_seed": args.init_seed,
        "n_train": len(train), "n_validation": len(val),
        "n_outer_test_hidden": int((outer_codes == 2).sum()),
        "n_input_bits": int(X.shape[1]),
        "aggregate_source_label": args.aggregate_label if args.family == "aggregate" else None,
        "aggregate_encoded_label": aggregate_y if args.family == "aggregate" else None,
        "inner_split_metadata": str(args.inner_split.with_suffix(".json")),
        "scaffold_overlap_train_validation": 0,
        "validation_mcc": float(matthews_corrcoef(yva, pred)),
        "validation_accuracy": float(accuracy_score(yva, pred)),
        "validation_balanced_accuracy": float(balanced_accuracy_score(yva, pred)),
        "validation_macro_f1": float(f1_score(yva, pred, average="macro", zero_division=0)),
        "best_epoch": best_epoch, "epochs_ran": epoch, "optimizer_updates": steps,
        "stopping_reason": "patience" if stale > args.patience else "epoch_cap",
        "history_mcc": history,
        "seconds": time.perf_counter() - started,
        "device": args.device,
    }
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(args.out)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
