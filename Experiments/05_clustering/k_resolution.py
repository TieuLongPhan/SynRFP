#!/usr/bin/env python
"""K-means resolution sensitivity around the reference class count."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synrfp_exp import clustering, metrics
from synrfp_exp.config import (SynRFPConfig, expand_fingerprints, load_config,
                               setup_logging, write_json)
from synrfp_exp.data import dataset_table
from synrfp_exp.features import Features, load
from synrfp_exp.scopes import full_scope


REPRESENTATIVE = ["schneider_b", "uspto50k_b", "syntemp_r0", "mech_2"]


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
    parser.add_argument("--datasets", nargs="*", default=REPRESENTATIVE)
    parser.add_argument("--fingerprints", nargs="*")
    parser.add_argument("--ratios", nargs="*", type=float, default=[0.5, 2.0])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    fps = expand_fingerprints(
        cfg, args.fingerprints or cfg["clustering"]["fingerprints"])
    output_root = Path(__file__).resolve().parent / "results_followup" / "k_resolution"

    for dataset in args.datasets:
        ds = dataset_table(cfg, dataset)
        spec = cfg["datasets"][dataset]
        scope = full_scope(cfg, dataset)
        class_count = int(ds["y"].nunique())
        for fingerprint in fps:
            full = load(cfg, scope.name, fingerprint)
            feats = Features(full.X[scope.feat_rows], full.n_bits,
                             full.ok[scope.feat_rows])
            for ratio in args.ratios:
                if ratio <= 0:
                    parser.error("K ratios must be positive")
                k = max(2, int(np.floor(class_count * ratio + 0.5)))
                for seed in args.seeds:
                    out = (output_root / dataset /
                           f"{fingerprint}__ratio{ratio:g}_k{k}_s{seed}.json")
                    if out.exists():
                        continue
                    started = time.perf_counter()
                    labels = clustering.kmeans(
                        feats, k, seed=seed,
                        n_init=cfg["clustering"]["kmeans"]["n_init"])
                    record = {
                        "dataset": dataset,
                        "fingerprint": fingerprint,
                        "method": "kmeans_resolution",
                        "seed": seed,
                        "k_ratio": ratio,
                        "k": k,
                        "reference_class_count": class_count,
                        "n": int(len(ds)),
                        "elapsed_seconds": time.perf_counter() - started,
                        **score(ds, spec, labels),
                    }
                    if fingerprint.startswith("synrfp-"):
                        syn = SynRFPConfig.from_name(fingerprint)
                        record.update(tokenizer=syn.tokenizer, radius=syn.radius,
                                      bits=syn.bits, sketch=syn.sketch, mode=syn.mode)
                    write_json(out, record)


if __name__ == "__main__":
    main()
