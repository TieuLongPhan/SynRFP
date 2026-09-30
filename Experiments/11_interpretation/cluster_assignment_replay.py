#!/usr/bin/env python
"""Save reproducible K-means assignments for selected chemical examples.

This is an explanatory replay, not a replacement for the completed clustering
benchmark. It preserves every dataset row and writes ``-1`` for an invalid
feature, so examples can be selected without silently treating failed encodings
as a real cluster.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synrfp_exp import clustering, metrics  # noqa: E402
from synrfp_exp.config import expand_fingerprints, load_config  # noqa: E402
from synrfp_exp.data import dataset_table  # noqa: E402
from synrfp_exp.features import Features, load  # noqa: E402
from synrfp_exp.scopes import full_scope  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="Experiments/config.yaml")
    parser.add_argument("--datasets", nargs="+", default=["schneider_b", "mech_2"])
    parser.add_argument("--fingerprints", nargs="*")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--output", default="Experiments/11_interpretation/results/assignments")
    args = parser.parse_args()

    cfg = load_config(args.config)
    fps = expand_fingerprints(
        cfg, args.fingerprints or cfg["clustering"]["fingerprints"])
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)

    for dataset in args.datasets:
        ds = dataset_table(cfg, dataset)
        spec = cfg["datasets"][dataset]
        scope = full_scope(cfg, dataset)
        k = int(ds["y"].nunique())
        feature_rows = scope.feat_rows
        for fingerprint in fps:
            feature = load(cfg, scope.name, fingerprint)
            feats = Features(feature.X[feature_rows], feature.n_bits,
                             feature.ok[feature_rows])
            valid = np.asarray(feats.ok, dtype=bool)
            if valid.sum() <= k:
                raise RuntimeError(f"{dataset}/{fingerprint}: too few valid rows")
            stem = f"{dataset}__{fingerprint}__kmeans_s{args.seed}"
            npz_path = out_root / f"{stem}.npz"
            json_path = out_root / f"{stem}.json"
            if npz_path.exists() and json_path.exists():
                print(f"skip existing {stem}")
                continue
            labels_valid = clustering.kmeans(
                Features(feats.X[valid], feats.n_bits, feats.ok[valid]),
                k=k, seed=args.seed, n_init=args.n_init)
            labels = np.full(len(ds), -1, dtype=np.int32)
            labels[valid] = labels_valid

            np.savez_compressed(
                npz_path,
                labels=labels,
                valid=valid,
                r_id=ds["r_id"].astype(str).to_numpy(),
                y=ds["y"].to_numpy(dtype=np.int32),
                reaction=ds["rxn"].astype(str).to_numpy(),
            )
            scores = {}
            for level in spec.get("hierarchy") or [spec["label"]]:
                column = "label" if level == spec["label"] else level
                scores[level] = metrics.clustering(
                    ds.loc[valid, column].to_numpy(), labels_valid)
            record = {
                "dataset": dataset,
                "fingerprint": fingerprint,
                "method": "kmeans",
                "seed": args.seed,
                "n_init": args.n_init,
                "k": k,
                "n_rows": int(len(ds)),
                "n_valid": int(valid.sum()),
                "invalid_rows": int((~valid).sum()),
                "scores_on_valid_rows": scores,
                "assignment_file": f"{stem}.npz",
            }
            json_path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n")
            print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
