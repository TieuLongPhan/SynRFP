#!/usr/bin/env python
"""Paired group-subsample stability for the clustering benchmark.

For each dataset, draw deterministic 80% product-scaffold group subsamples and
apply the identical selected rows to every fingerprint. K is frozen to the
number of primary classes in the full endpoint. Results live outside the main
``05_clustering/results`` tree so the standard report cannot mix subsample runs
with full-data runs.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synrfp_exp import clustering, metrics
from synrfp_exp.config import (SynRFPConfig, expand_fingerprints, load_config,
                               setup_logging, write_json)
from synrfp_exp.data import dataset_table
from synrfp_exp.features import Features, load
from synrfp_exp.scopes import full_scope
from synrfp_exp.splits import groups_for


REPRESENTATIVE = ["schneider_b", "uspto50k_b", "syntemp_r0", "mech_2"]


def score(ds, spec, pos, labels):
    levels = spec.get("hierarchy") or [spec["label"]]
    out = {}
    for level in levels:
        column = "label" if level == spec["label"] else level
        truth = ds[column].to_numpy()[pos]
        for name, value in metrics.clustering(truth, labels).items():
            out[f"{level}__{name}"] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--datasets", nargs="*", default=REPRESENTATIVE)
    parser.add_argument("--fingerprints", nargs="*")
    parser.add_argument("--subsamples", type=int, default=20)
    parser.add_argument("--fraction", type=float, default=0.8)
    parser.add_argument("--sampling-seed", type=int, default=20260924)
    parser.add_argument("--kmeans-seed", type=int, default=0)
    args = parser.parse_args()
    setup_logging()
    if not 0 < args.fraction < 1:
        parser.error("--fraction must be between zero and one")

    cfg = load_config(args.config)
    fps = expand_fingerprints(
        cfg, args.fingerprints or cfg["clustering"]["fingerprints"])
    output_root = Path(__file__).resolve().parent / "results_followup" / "stability80"
    sample_root = output_root / "samples"

    for dataset in args.datasets:
        ds = dataset_table(cfg, dataset)
        spec = cfg["datasets"][dataset]
        scope = full_scope(cfg, dataset)
        groups = groups_for(cfg, dataset)
        y = ds["y"].to_numpy()
        k = int(np.unique(y).size)
        splitter = GroupShuffleSplit(
            n_splits=args.subsamples, train_size=args.fraction,
            random_state=args.sampling_seed)
        samples = []
        for sample_seed, (pos, _) in enumerate(splitter.split(y, y, groups)):
            pos = np.sort(pos)
            sample_path = sample_root / dataset / f"sample_s{sample_seed}.npy"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            if sample_path.exists():
                saved = np.load(sample_path)
                if not np.array_equal(saved, pos):
                    raise RuntimeError(f"sample definition changed: {sample_path}")
            else:
                np.save(sample_path, pos)
            samples.append((sample_seed, pos))

        aligned_features = {}
        for fingerprint in fps:
            full = load(cfg, scope.name, fingerprint)
            aligned_features[fingerprint] = Features(
                full.X[scope.feat_rows], full.n_bits, full.ok[scope.feat_rows])

        # Complete a paired comparison before advancing to the next sample.
        # This produces interpretable interim results while the full run continues.
        for sample_seed, pos in samples:
            for fingerprint in fps:
                out = (output_root / dataset / f"sample_s{sample_seed}" /
                       f"{fingerprint}__kmeans.json")
                if out.exists():
                    continue
                aligned = aligned_features[fingerprint]
                feats = Features(aligned.X[pos], aligned.n_bits, aligned.ok[pos])
                started = time.perf_counter()
                labels = clustering.kmeans(
                    feats, k, seed=args.kmeans_seed,
                    n_init=cfg["clustering"]["kmeans"]["n_init"])
                record = {
                    "dataset": dataset,
                    "fingerprint": fingerprint,
                    "method": "kmeans_group_subsample",
                    "sample_seed": sample_seed,
                    "sampling_seed": args.sampling_seed,
                    "kmeans_seed": args.kmeans_seed,
                    "target_fraction": args.fraction,
                    "actual_fraction": float(len(pos) / len(ds)),
                    "n": int(len(pos)),
                    "n_full": int(len(ds)),
                    "n_groups": int(np.unique(groups[pos]).size),
                    "n_classes_full": k,
                    "n_classes_observed": int(np.unique(y[pos]).size),
                    "sample_row_sha256": hashlib.sha256(pos.tobytes()).hexdigest(),
                    "elapsed_seconds": time.perf_counter() - started,
                    **score(ds, spec, pos, labels),
                }
                if fingerprint.startswith("synrfp-"):
                    syn = SynRFPConfig.from_name(fingerprint)
                    record.update(tokenizer=syn.tokenizer, radius=syn.radius,
                                  bits=syn.bits, sketch=syn.sketch, mode=syn.mode)
                write_json(out, record)


if __name__ == "__main__":
    main()
