#!/usr/bin/env python
"""E10. Small-data case study: enzyme activity on EHreact (SynRXN).

SynRXN ``ehreact_activity`` lists, for nine enzyme systems, whether a reaction
is active in an assay (long form: one row per reaction and assay). Most assays
have fewer than ten reactions, so only assays with at least
``activity.min_positives`` active *and* inactive records are evaluated, each as
its own binary task with repeated stratified group k-fold cross-validation
(identical reactions share a group, so they never straddle train and test).

Scores are the kNN mean label of the neighbours (Tanimoto for bits, cosine
otherwise) and a class-balanced logistic regression; metrics are MCC, average
precision and ROC-AUC on the pooled out-of-fold predictions of each repeat.

Run once before external/rxnfp/rxnfp_embed.py to export the reactions; the
RXNFP rows are used when their embeddings exist.

Outputs: 10_enzyme_activity/results/{tasks.csv,runs.csv,summary.csv}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Experiments/
import argparse

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from sklearn.model_selection import train_test_split
try:
    from sklearn.model_selection import StratifiedGroupKFold
except ImportError:  # CUDA environment uses an older scikit-learn
    StratifiedGroupKFold = None

from synrfp_exp.config import (expand_fingerprints, load_config, log,
                               setup_logging, results_dir)
from synrfp_exp.data import check_data_source, export_reactions, load_synrxn
from synrfp_exp.features import EXTERNAL, _path, compute, load
from synrfp_exp.models import similarity_topk


def knn_scores(feats, tr, ytr, te, k):
    idx, _ = similarity_topk(feats.dense(te), feats.dense(tr), min(k, len(tr)),
                             tanimoto=feats.is_bits)
    return ytr[idx].mean(1)


def logreg_scores(feats, tr, ytr, te, seed):
    from sklearn.linear_model import LogisticRegression

    Xtr, Xte = feats.dense(tr), feats.dense(te)
    if not feats.is_bits:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000,
                             random_state=seed)
    return clf.fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def mlp_scores(feats, tr, ytr, te, seed, params, device):
    """GPU binary MLP probe with the benchmark architecture and early stopping."""
    import torch
    from torch import nn

    Xtr, Xte = feats.dense(tr), feats.dense(te)
    if not feats.is_bits:
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    fit, val = train_test_split(np.arange(len(ytr)), test_size=0.1,
                                random_state=seed, stratify=ytr)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    dev = torch.device(device)
    xfit = torch.tensor(Xtr[fit], dtype=torch.float32, device=dev)
    yfit = torch.tensor(ytr[fit], dtype=torch.float32, device=dev)
    xval = torch.tensor(Xtr[val], dtype=torch.float32, device=dev)
    yval = torch.tensor(ytr[val], dtype=torch.float32, device=dev)
    model = nn.Sequential(nn.Linear(Xtr.shape[1], params["hidden"][0]), nn.ReLU(), nn.Linear(params["hidden"][0], 1)).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=params["alpha"] / len(fit))
    loss = nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(seed)
    order = np.arange(len(fit))
    best_score, stale, best = -1.0, 0, None
    batch = min(params["batch_size"], len(fit))
    for _ in range(params["max_iter"]):
        model.train(); rng.shuffle(order)
        for start in range(0, len(order), batch):
            pos = torch.tensor(order[start:start + batch], device=dev)
            opt.zero_grad()
            objective = loss(model(xfit[pos]).squeeze(1), yfit[pos])
            objective.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            score = float(((model(xval).squeeze(1) >= 0) == yval.bool()).float().mean())
        stale = stale + 1 if score < best_score + 1e-4 else 0
        if score > best_score:
            best_score = score
            best = {name: value.detach().clone() for name, value in model.state_dict().items()}
        if stale > 10:
            break
    model.load_state_dict(best); model.eval()
    with torch.no_grad():
        out = torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32, device=dev)).squeeze(1)).cpu().numpy()
    return out


def main():  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--fingerprints", nargs="*")
    ap.add_argument("--models", nargs="*", choices=["knn", "linear", "mlp"])
    ap.add_argument("--device", default="cuda:0", help="CUDA device used by --models mlp")
    ap.add_argument("--splits", type=Path, help="precomputed grouped folds for CUDA environments")
    ap.add_argument("--raw", type=Path, help="cached EHReact table for CUDA environments")
    ap.add_argument("--export-only", action="store_true",
                    help="write the reaction export for the RXNFP environment and stop")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    ac = cfg["activity"]
    out_dir = results_dir("10_enzyme_activity")

    if args.raw:
        raw = pd.read_csv(args.raw)
    else:
        check_data_source(cfg)  # features and exports below are cached by name
        raw = load_synrxn(cfg, ac["name"])
    reactions = pd.Index(raw["rxn"].astype(str).unique())
    key = ac["name"]
    if not args.raw:
        export_reactions(cfg, key, reactions)
    if args.export_only:
        return
    row_of = reactions.get_indexer(raw["rxn"].astype(str))  # row in the reaction table

    counts = raw.groupby(["system", "assay"])["activity"].agg(["size", "sum"])
    counts["neg"] = counts["size"] - counts["sum"]
    counts["kept"] = (counts["sum"] >= ac["min_positives"]) & (counts["neg"] >= ac["min_positives"])
    counts.rename(columns={"size": "records", "sum": "active"}).to_csv(out_dir / "tasks.csv")
    tasks = [t for t, kept in counts["kept"].items() if kept]
    log.info("%d of %d assays have >= %d active and inactive records",
             len(tasks), len(counts), ac["min_positives"])

    fps = expand_fingerprints(cfg, args.fingerprints or ac["fingerprints"])
    selected_models = args.models or ac["models"]
    stored_splits = np.load(args.splits) if args.splits else None
    compute(cfg, key, list(reactions), fps)
    rows = []
    for fp in fps:
        if fp in EXTERNAL and not _path(cfg, key, fp).exists():
            log.warning("skip %s: run external/rxnfp/rxnfp_embed.py", fp)
            continue
        feats = load(cfg, key, fp)
        for system, assay in tasks:
            sub = raw[(raw["system"] == system) & (raw["assay"] == assay)]
            r = row_of[sub.index.to_numpy()]
            y = sub["activity"].to_numpy().astype(int)
            for rep in range(ac["repeats"]):
                if stored_splits is not None:
                    prefix = f"{system}__{assay}__r{rep}"
                    folds = [(stored_splits[f"{prefix}__f{fold}__tr"], stored_splits[f"{prefix}__f{fold}__te"])
                             for fold in range(ac["folds"])]
                else:
                    if StratifiedGroupKFold is None:
                        raise RuntimeError("precompute grouped folds with prepare_activity_splits.py")
                    cv = StratifiedGroupKFold(n_splits=ac["folds"], shuffle=True, random_state=rep)
                    folds = cv.split(r, y, groups=r)
                scores = {m: np.zeros(len(y)) for m in selected_models}
                for tr, te in folds:
                    for m in selected_models:
                        if m == "knn":
                            s = knn_scores(feats, r[tr], y[tr], r[te], cfg["models"]["knn"]["k"])
                        elif m == "linear":
                            s = logreg_scores(feats, r[tr], y[tr], r[te], rep)
                        elif m == "mlp":
                            s = mlp_scores(feats, r[tr], y[tr], r[te], rep, cfg["models"]["mlp"], args.device)
                        else:
                            raise ValueError(f"unsupported model {m!r}")
                        scores[m][te] = s
                for m, s in scores.items():
                    pred = (s >= 0.5).astype(int)
                    rows.append({"system": system, "assay": assay, "fingerprint": fp,
                                 "model": m, "repeat": rep, "n": len(y), "active": int(y.sum()),
                                 "mcc": skm.matthews_corrcoef(y, pred),
                                 "average_precision": skm.average_precision_score(y, s),
                                 "roc_auc": skm.roc_auc_score(y, s)})
            log.info("%s %s %s done", system, assay, fp)

    runs = pd.DataFrame(rows)
    existing_path = out_dir / "runs.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        runs = pd.concat([existing[~(existing["model"].isin(selected_models) & existing["fingerprint"].isin(fps))], runs], ignore_index=True)
    runs.to_csv(existing_path, index=False)
    summary = (runs.groupby(["system", "assay", "fingerprint", "model"])
               [["mcc", "average_precision", "roc_auc"]].agg(["mean", "std"]))
    summary.to_csv(out_dir / "summary.csv")
    log.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
