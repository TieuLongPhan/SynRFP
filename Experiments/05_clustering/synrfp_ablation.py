#!/usr/bin/env python
"""One-factor-at-a-time SynRFP clustering ablation on representative endpoints."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synrfp_exp import clustering, metrics
from synrfp_exp.config import SynRFPConfig, load_config, setup_logging, write_json
from synrfp_exp.data import dataset_table
from synrfp_exp.features import Features, load
from synrfp_exp.scopes import factorial_scope


REPRESENTATIVE = ["schneider_b", "uspto50k_b", "syntemp_r0", "mech_2"]
CONFIGURATIONS = [
    # Default reference.
    "synrfp-wl-r2-b1024-parity-delta",
    # Change one design axis from the default at a time.
    "synrfp-morgan-r2-b1024-parity-delta",
    "synrfp-path-r2-b1024-parity-delta",
    "synrfp-wl-r1-b1024-parity-delta",
    "synrfp-wl-r3-b1024-parity-delta",
    "synrfp-wl-r2-b512-parity-delta",
    "synrfp-wl-r2-b2048-parity-delta",
    "synrfp-wl-r2-b1024-minhash-delta",
    "synrfp-wl-r2-b1024-cw-delta",
    "synrfp-wl-r2-b1024-parity-union",
]


def score(ds, spec, labels):
    levels = spec.get("hierarchy") or [spec["label"]]
    out = {}
    for level in levels:
        column = "label" if level == spec["label"] else level
        truth = ds[column].to_numpy()
        for name, value in metrics.clustering(truth, labels).items():
            out[f"{level}__{name}"] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--datasets", nargs="*", default=REPRESENTATIVE)
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    output_root = Path(__file__).resolve().parent / "results_followup" / "ablation"

    for dataset in args.datasets:
        full_ds = dataset_table(cfg, dataset)
        scope = factorial_scope(cfg, dataset)
        ds = full_ds.iloc[scope.ds_pos].reset_index(drop=True)
        spec = cfg["datasets"][dataset]
        k = int(ds["y"].nunique())
        for fingerprint in CONFIGURATIONS:
            full = load(cfg, scope.name, fingerprint)
            feats = Features(full.X[scope.feat_rows], full.n_bits,
                             full.ok[scope.feat_rows])
            for seed in args.seeds:
                out = output_root / dataset / f"{fingerprint}__s{seed}.json"
                if out.exists():
                    continue
                started = time.perf_counter()
                labels = clustering.kmeans(
                    feats, k, seed=seed,
                    n_init=cfg["clustering"]["kmeans"]["n_init"])
                syn = SynRFPConfig.from_name(fingerprint)
                record = {
                    "dataset": dataset,
                    "fingerprint": fingerprint,
                    "method": "kmeans_synrfp_ablation",
                    "seed": seed,
                    "k": k,
                    "n": int(len(ds)),
                    "scope": scope.name,
                    "elapsed_seconds": time.perf_counter() - started,
                    "tokenizer": syn.tokenizer,
                    "radius": syn.radius,
                    "bits": syn.bits,
                    "sketch": syn.sketch,
                    "mode": syn.mode,
                    **score(ds, spec, labels),
                }
                write_json(out, record)


if __name__ == "__main__":
    main()
