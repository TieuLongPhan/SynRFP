#!/usr/bin/env python
"""E7. Robustness and resolution of the representations.

(a) pairs      Similarity between the fingerprints of the unbalanced (UB) and
               balanced (B) record of the same reaction (same r_id).
(b) transfer   Train on B, test on UB (and vice versa) with the published split,
               next to in-domain B->B and UB->UB, to measure how much missing
               participants hurt.
(c) perturb    Fingerprint identity under randomised atom order and shuffled
               component order, and the similarity to the reversed reaction.
(d) resolution Records whose fingerprint equals that of a record with another
               label, compared with conflicts already present in the chemistry
               (identical canonical reactions); the excess is caused by the
               representation (hash collisions, lost context).

Outputs: 07_robustness/results/{pairs,transfer,perturb,resolution}.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Experiments/
import argparse

import numpy as np
import pandas as pd

from synrfp_exp import metrics
from synrfp_exp.config import (expand_fingerprints, load_config, log,
                               setup_logging, results_dir)
from synrfp_exp.data import canonical_for, dataset_table
from synrfp_exp.features import (Features, compute, compute_in_memory, load,
                                 rowwise_identical, rowwise_similarity)
from synrfp_exp.models import fit_predict
from synrfp_exp.scopes import full_scope
from synrfp_exp.splits import TEST, TRAIN, VAL, make_split


def pair_members(cfg):
    pairs = {}
    for name, spec in cfg["datasets"].items():
        if "pair" in spec:
            pairs.setdefault(spec["pair"], {})[spec["role"]] = name
    return {k: v for k, v in pairs.items() if {"UB", "B"} <= set(v)}


def aligned(cfg, ub, b, fp):
    """Features and tables of UB and B restricted to shared r_ids, same order."""
    dsu, dsb = dataset_table(cfg, ub), dataset_table(cfg, b)
    common = pd.Index(dsu["r_id"]).intersection(pd.Index(dsb["r_id"]))
    pu = pd.Index(dsu["r_id"]).get_indexer(common)
    pb = pd.Index(dsb["r_id"]).get_indexer(common)
    su, sb = full_scope(cfg, ub), full_scope(cfg, b)
    fu, fb = load(cfg, su.name, fp), load(cfg, sb.name, fp)
    FU = Features(fu.X[su.feat_rows[pu]], fu.n_bits, fu.ok[su.feat_rows[pu]])
    FB = Features(fb.X[sb.feat_rows[pb]], fb.n_bits, fb.ok[sb.feat_rows[pb]])
    return dsu.iloc[pu].reset_index(drop=True), dsb.iloc[pb].reset_index(drop=True), FU, FB, pb


def run_pairs_and_transfer(cfg, rc, out):
    prow, trow = [], []
    for pair, m in pair_members(cfg).items():
        for fp in expand_fingerprints(cfg, rc["pair_fingerprints"]):
            try:
                dsu, dsb, FU, FB, pb = aligned(cfg, m["UB"], m["B"], fp)
            except FileNotFoundError as e:
                log.warning("skip %s/%s: %s", pair, fp, e)
                continue
            sim = rowwise_similarity(FU, FB)
            prow.append({"pair": pair, "fingerprint": fp, "n": len(sim),
                         "mean": sim.mean(), "median": np.median(sim),
                         "q10": np.quantile(sim, 0.1),
                         "identical": float(rowwise_identical(FU, FB).mean())})
            code = make_split(cfg, m["B"], "published", 0)[pb]
            labels = pd.Categorical(np.concatenate([dsu["label"], dsb["label"]]))
            yu = labels.codes[: len(dsu)]
            yb = labels.codes[len(dsu):]
            src = {"B": (FB, yb), "UB": (FU, yu)}
            for a in ("B", "UB"):
                for t in ("B", "UB"):
                    Fa, ya = src[a]
                    Ft, yt = src[t]
                    tr, va, te = (np.where(code == c)[0] for c in (TRAIN, VAL, TEST))
                    X = np.concatenate([Fa.X[tr], Fa.X[va], Ft.X[te]])
                    F = Features(X, Fa.n_bits, np.ones(len(X), bool))
                    rtr = np.arange(len(tr))
                    rva = len(tr) + np.arange(len(va))
                    rte = len(tr) + len(va) + np.arange(len(te))
                    for model in rc["transfer_models"]:
                        pred = fit_predict(model, cfg["models"][model], F, rtr, ya[tr],
                                           rva, ya[va], rte, seed=0, n_jobs=cfg["n_jobs"])
                        trow.append({"pair": pair, "fingerprint": fp, "model": model,
                                     "train": a, "test": t,
                                     **metrics.classification(yt[te], pred)})
            log.info("%s %s: mean UB/B similarity %.3f", pair, fp, sim.mean())
    pd.DataFrame(prow).to_csv(out / "pairs.csv", index=False)
    pd.DataFrame(trow).to_csv(out / "transfer.csv", index=False)


def randomise(rsmi, rng):
    from rdkit import Chem

    sides = rsmi.split(">")
    new = []
    for side in sides:
        comps = [c for c in side.split(".") if c]
        rng.shuffle(comps)
        rnd = []
        for c in comps:
            mol = Chem.MolFromSmiles(c)
            rnd.append(Chem.MolToSmiles(mol, doRandom=True, canonical=False) if mol else c)
        new.append(".".join(rnd))
    return ">".join(new)


def run_perturb(cfg, rc, out):
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    rows = []
    for dataset in rc["perturbation_datasets"]:
        try:
            ds = dataset_table(cfg, dataset)
        except FileNotFoundError as e:
            log.error("%s skipped: %s", dataset, e)
            continue
        rng = np.random.default_rng(0)
        pick = rng.choice(len(ds), min(rc["perturbation_sample"], len(ds)), replace=False)
        orig = ds["rxn"].to_numpy()[pick].tolist()
        pert = [randomise(s, rng) for s in orig]
        rev = [">".join(s.split(">")[::-1]) for s in orig]
        for fp in expand_fingerprints(cfg, rc["perturbation_fingerprints"]):
            fo = compute_in_memory(cfg, orig, fp)
            fpert = compute_in_memory(cfg, pert, fp)
            frev = compute_in_memory(cfg, rev, fp)
            s_p, s_r = rowwise_similarity(fo, fpert), rowwise_similarity(fo, frev)
            same_p, same_r = rowwise_identical(fo, fpert), rowwise_identical(fo, frev)
            rows.append({"dataset": dataset, "fingerprint": fp, "n": len(orig),
                         "identical_under_randomisation": float(same_p.mean()),
                         "mean_sim_randomised": s_p.mean(),
                         "identical_to_reverse": float(same_r.mean()),
                         "mean_sim_reverse": s_r.mean()})
            log.info("%s %s: identical under randomisation %.3f", dataset, fp,
                     rows[-1]["identical_under_randomisation"])
    pd.DataFrame(rows).to_csv(out / "perturb.csv", index=False)


def conflict_rate(keys, y):
    df = pd.DataFrame({"k": keys, "y": y})
    n_lab = df.groupby("k")["y"].transform("nunique")
    return float((n_lab > 1).mean()), int(df["k"].nunique())


def run_resolution(cfg, rc, out):
    rows = []
    fps = expand_fingerprints(cfg, rc["resolution_fingerprints"])
    for dataset in rc["resolution_datasets"]:
        try:
            ds = dataset_table(cfg, dataset)
        except FileNotFoundError as e:
            log.error("%s skipped: %s", dataset, e)
            continue
        sc = full_scope(cfg, dataset)
        compute(cfg, sc.name, sc.rsmis, fps)
        y = ds["y"].to_numpy()
        can = canonical_for(cfg, dataset)
        chem_conf, n_can = conflict_rate(can, y)
        for fp in fps:
            f = load(cfg, sc.name, fp)
            X = f.X[sc.feat_rows]
            keys = [r.tobytes() for r in X]
            fp_conf, n_fp = conflict_rate(keys, y)
            rows.append({"dataset": dataset, "fingerprint": fp, "records": len(y),
                         "unique_reactions": n_can, "unique_fingerprints": n_fp,
                         "chemistry_conflict_rate": chem_conf,
                         "fingerprint_conflict_rate": fp_conf,
                         "excess_conflict_rate": fp_conf - chem_conf})
            log.info("%s %s: %d unique fps / %d unique rxns, conflicts %.4f (chem %.4f)",
                     dataset, fp, n_fp, n_can, fp_conf, chem_conf)
    pd.DataFrame(rows).to_csv(out / "resolution.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--parts", nargs="*", default=["pairs", "perturb", "resolution"],
                    help="pairs (includes transfer), perturb, resolution")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    rc = cfg["robustness"]
    out = results_dir("07_robustness")
    out.mkdir(parents=True, exist_ok=True)
    if "pairs" in args.parts:
        run_pairs_and_transfer(cfg, rc, out)
    if "perturb" in args.parts:
        run_perturb(cfg, rc, out)
    if "resolution" in args.parts:
        run_resolution(cfg, rc, out)


if __name__ == "__main__":
    main()
