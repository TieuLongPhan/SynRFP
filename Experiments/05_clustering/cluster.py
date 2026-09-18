#!/usr/bin/env python
"""E5. Unsupervised structure: k-means (k = #labels) and k-free clustering.

For every dataset, fingerprint and method the partition is scored against
*each* level of the label hierarchy (e.g. SynTemp R0/R1/R2, Mech level 1/2).
Sweeping the BitBIRCH threshold therefore yields a granularity profile: which
level of chemical detail the fingerprint geometry resolves as the similarity
cut tightens.

Methods
  kmeans    k = number of classes of the dataset label (seeds 0..2)
  bitbirch  Tanimoto threshold sweep; bit fingerprints only
  hdbscan   min_cluster_size sweep on PCA features; all fingerprints

Outputs: 05_clustering/results/<dataset>/<fingerprint>/<method>_<param>.json
Threshold selection is done in 09_report/report.py (calibrated on one dataset and
frozen for all others), not here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Experiments/
import argparse

import numpy as np

from synrfp_exp import clustering, metrics
from synrfp_exp.config import (expand_fingerprints, load_config, log,
                               setup_logging, write_json, results_dir)
from synrfp_exp.data import dataset_table
from synrfp_exp.features import Features, load
from synrfp_exp.scopes import full_scope


def score(ds, spec, labels):
    levels = spec.get("hierarchy") or [spec["label"]]
    out = {}
    for lev in levels:
        col = "label" if lev == spec["label"] else lev
        y = ds[col].to_numpy()
        for k, v in metrics.clustering(y, labels).items():
            out[f"{lev}__{k}"] = v
    return out


def main():  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--fingerprints", nargs="*")
    ap.add_argument("--methods", nargs="*", default=["kmeans", "bitbirch", "hdbscan"])
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    cc = cfg["clustering"]
    root = results_dir("05_clustering")
    for dataset in args.datasets or cc["datasets"]:
        spec = cfg["datasets"][dataset]
        try:
            ds = dataset_table(cfg, dataset)
        except FileNotFoundError as e:
            log.error("%s skipped: %s", dataset, e)
            continue
        sc = full_scope(cfg, dataset)
        k = int(ds["y"].nunique())
        for fp in expand_fingerprints(cfg, args.fingerprints or cc["fingerprints"]):
            try:
                full = load(cfg, sc.name, fp)
            except FileNotFoundError as e:
                log.warning("skip %s: %s", fp, e)
                continue
            feats = Features(full.X[sc.feat_rows], full.n_bits, full.ok[sc.feat_rows])
            jobs = []
            if "kmeans" in args.methods:
                jobs += [("kmeans", f"k{k}_s{s}", s) for s in (0, 1, 2)]
            if "bitbirch" in args.methods and feats.is_bits:
                jobs += [("bitbirch", f"t{t:.2f}", t) for t in cc["bitbirch"]["thresholds"]]
            if "hdbscan" in args.methods:
                jobs += [("hdbscan", f"m{m}", m) for m in cc["hdbscan"]["min_cluster_size"]]
            for method, tag, p in jobs:
                out = root / dataset / fp / f"{method}_{tag}.json"
                if out.exists():
                    continue
                extra = {}
                if method == "kmeans":
                    lab = clustering.kmeans(feats, k, seed=p, n_init=cc["kmeans"]["n_init"])
                elif method == "bitbirch":
                    lab = clustering.bitbirch(feats, p, cc["bitbirch"]["branching_factor"],
                                              cc["bitbirch"]["merge_criterion"])
                else:
                    lab, noise = clustering.hdbscan(feats, p, cc["hdbscan"]["pca_dim"], seed=0)
                    extra["noise_fraction"] = noise
                rec = {"dataset": dataset, "fingerprint": fp, "method": method,
                       "param": p, "n": int(len(lab)), **extra, **score(ds, spec, lab)}
                write_json(out, rec)
                log.info("%s %s %s %s: %d clusters", dataset, fp, method, tag,
                         int(np.unique(lab).size))


if __name__ == "__main__":
    main()
