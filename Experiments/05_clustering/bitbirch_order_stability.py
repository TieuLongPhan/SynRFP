#!/usr/bin/env python
"""Test BitBIRCH sensitivity to input order at frozen operating points."""
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
    parser.add_argument("--orders", type=int, default=10)
    parser.add_argument("--order-seed", type=int, default=20260924)
    args = parser.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    bit = cfg["clustering"]["bitbirch"]
    aliases = bit["frozen_thresholds"]
    thresholds = {}
    for alias, threshold in aliases.items():
        expanded = expand_fingerprints(cfg, [alias])
        if len(expanded) != 1:
            raise ValueError(f"fingerprint must expand once: {alias}")
        thresholds[expanded[0]] = float(threshold)
    output_root = (Path(__file__).resolve().parent / "results_followup" /
                   "bitbirch_order")

    for dataset in args.datasets:
        ds = dataset_table(cfg, dataset)
        spec = cfg["datasets"][dataset]
        scope = full_scope(cfg, dataset)
        for fingerprint, threshold in thresholds.items():
            full = load(cfg, scope.name, fingerprint)
            feats = Features(full.X[scope.feat_rows], full.n_bits,
                             full.ok[scope.feat_rows])
            if not feats.is_bits:
                raise TypeError(f"{fingerprint} is not a bit fingerprint")
            for order in range(args.orders):
                out = (output_root / dataset /
                       f"{fingerprint}__t{threshold:.2f}_order{order}.json")
                if out.exists():
                    continue
                rng = np.random.default_rng(args.order_seed + order)
                permutation = rng.permutation(len(ds))
                shuffled = Features(feats.X[permutation], feats.n_bits,
                                    feats.ok[permutation])
                started = time.perf_counter()
                shuffled_labels = clustering.bitbirch(
                    shuffled, threshold, bit["branching_factor"],
                    bit["merge_criterion"])
                labels = np.empty_like(shuffled_labels)
                labels[permutation] = shuffled_labels
                record = {
                    "dataset": dataset,
                    "fingerprint": fingerprint,
                    "method": "bitbirch_order_stability",
                    "threshold": threshold,
                    "order": order,
                    "order_seed": args.order_seed + order,
                    "n": int(len(labels)),
                    "elapsed_seconds": time.perf_counter() - started,
                    **score(ds, spec, labels),
                }
                write_json(out, record)


if __name__ == "__main__":
    main()
