#!/usr/bin/env python
"""Small shared-valid-row and L2-normalization clustering sensitivity check."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from synrfp_exp.config import expand_fingerprints, load_config  # noqa: E402
from synrfp_exp.data import dataset_table  # noqa: E402
from synrfp_exp.features import Features, load  # noqa: E402
from synrfp_exp.scopes import full_scope  # noqa: E402
from synrfp_exp import clustering, metrics  # noqa: E402


def _normalized(feats: Features) -> Features:
    x = feats.dense().astype(np.float32, copy=False)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.maximum(norms, 1e-12)
    return Features(x, 0, np.ones(len(x), dtype=bool))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="Experiments/config.yaml")
    ap.add_argument("--datasets", nargs="+", default=["schneider_b"])
    ap.add_argument("--fingerprints", nargs="*", default=["synrfp_default", "drfp"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-init", type=int, default=10)
    ap.add_argument("--output", default="Experiments/11_interpretation/results/clustering_sensitivity.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    fps = expand_fingerprints(cfg, args.fingerprints)
    rows = []
    for dataset in args.datasets:
        ds = dataset_table(cfg, dataset)
        scope = full_scope(cfg, dataset)
        loaded = {fp: load(cfg, scope.name, fp) for fp in fps}
        common = np.logical_and.reduce([np.asarray(f.ok, dtype=bool) for f in loaded.values()])
        k = int(ds["y"].nunique())
        labels_ref = ds.loc[common, "y"].to_numpy()
        for mode in ("native", "l2_normalized"):
            for fp, feature in loaded.items():
                x = feature if mode == "native" else _normalized(feature)
                x = Features(x.X[common], x.n_bits, x.ok[common])
                labels = clustering.kmeans(x, k=k, seed=args.seed, n_init=args.n_init)
                score = metrics.clustering(labels_ref, labels)
                rows.append({
                    "dataset": dataset, "fingerprint": fp, "preprocessing": mode,
                    "seed": args.seed, "n_init": args.n_init, "k": k,
                    "n_rows": len(ds), "n_common_valid": int(common.sum()),
                    "invalid_rows_common_excluded": int((~common).sum()),
                    "ami": score["ami"], "ari": score["ari"],
                    "completeness": score["completeness"],
                    "homogeneity": score["homogeneity"],
                })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(json.dumps({"rows": len(rows), "output": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
