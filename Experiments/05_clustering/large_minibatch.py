#!/usr/bin/env python
"""Ten-seed MiniBatchKMeans comparison on full TPL and ECReact datasets."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synrfp_exp import clustering, metrics
from synrfp_exp.config import expand_fingerprints, load_config, setup_logging, write_json
from synrfp_exp.data import dataset_table
from synrfp_exp.features import Features, load
from synrfp_exp.scopes import full_scope


LARGE_DATASETS = ["tpl_ub", "tpl_b", "ecreact_1", "ecreact_2", "ecreact_3"]


def score(ds, spec, labels):
    levels = spec.get("hierarchy") or [spec["label"]]
    out = {}
    for level in levels:
        column = "label" if level == spec["label"] else level
        for name, value in metrics.clustering(ds[column].to_numpy(), labels).items():
            out[f"{level}__{name}"] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--datasets", nargs="*", default=LARGE_DATASETS)
    parser.add_argument("--fingerprints", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--n-init", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    fps = expand_fingerprints(
        cfg, args.fingerprints or cfg["clustering"]["fingerprints"])
    output_root = Path(__file__).resolve().parent / "results_followup" / "large_minibatch"

    for dataset in args.datasets:
        ds = dataset_table(cfg, dataset)
        spec = cfg["datasets"][dataset]
        scope = full_scope(cfg, dataset)
        k = int(ds["y"].nunique())
        for fingerprint in fps:
            full = load(cfg, scope.name, fingerprint)
            feats = Features(full.X[scope.feat_rows], full.n_bits,
                             full.ok[scope.feat_rows])
            for seed in args.seeds:
                out = output_root / dataset / f"{fingerprint}__s{seed}.json"
                if out.exists():
                    continue
                started = time.perf_counter()
                labels = clustering.minibatch_kmeans(
                    feats, k, seed=seed, batch_size=args.batch_size,
                    n_init=args.n_init, max_iter=args.max_iter)
                record = {
                    "dataset": dataset,
                    "fingerprint": fingerprint,
                    "method": "minibatch_kmeans",
                    "seed": seed,
                    "k": k,
                    "n": int(len(ds)),
                    "batch_size": args.batch_size,
                    "n_init": args.n_init,
                    "max_iter": args.max_iter,
                    "elapsed_seconds": time.perf_counter() - started,
                    **score(ds, spec, labels),
                }
                write_json(out, record)


if __name__ == "__main__":
    main()
