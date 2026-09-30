#!/usr/bin/env python
"""Select deterministic, auditable reaction examples from replayed clusters.

The clustering runners save assignments but do not save a choice of examples.
This utility chooses rows by a fixed rule: the largest valid clusters, then the
row with the smallest reaction id among rows matching the cluster's modal
reference label.  It also reports cluster purity and label counts so examples
cannot be presented without their context.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _examples(path: Path, top_n: int) -> list[dict]:
    # Reaction IDs and SMILES are serialized as object arrays by the replay;
    # assignments themselves remain ordinary numeric arrays.
    z = np.load(path, allow_pickle=True)
    labels = z["labels"].astype(int)
    valid = z["valid"].astype(bool)
    rid = z["r_id"].astype(str)
    rxn = z["reaction"].astype(str)
    ref = z["y"].astype(int)
    rows: list[dict] = []
    for cid in sorted(set(labels[valid].tolist())):
        ix = np.flatnonzero(valid & (labels == cid))
        if ix.size == 0:
            continue
        counts = Counter(ref[ix].tolist())
        modal, modal_count = min(
            counts.items(), key=lambda kv: (-kv[1], kv[0]))
        purity = modal_count / ix.size
        candidates = ix[ref[ix] == modal]
        chosen = min(candidates.tolist(), key=lambda i: (rid[i], int(i)))
        rows.append({
            "cluster_id": int(cid),
            "cluster_size": int(ix.size),
            "modal_reference_label": int(modal),
            "modal_label_count": int(modal_count),
            "cluster_purity": float(purity),
            "n_reference_labels": int(len(counts)),
            "r_id": rid[chosen],
            "reaction": rxn[chosen],
            "reference_label": int(ref[chosen]),
            "valid": True,
        })
    rows.sort(key=lambda r: (-r["cluster_size"], r["cluster_id"]))
    for rank, row in enumerate(rows[:top_n], start=1):
        row["selection_rank"] = rank
        row["selection_rule"] = (
            "largest_valid_clusters; smallest_r_id_with_modal_reference_label"
        )
    return rows[:top_n]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assignments", default="Experiments/11_interpretation/results/assignments")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--output", default="Experiments/11_interpretation/results/cluster_examples.csv")
    args = ap.parse_args()

    root = Path(args.assignments)
    files = sorted(root.glob(f"{args.dataset}__*__kmeans_s*.npz"))
    if not files:
        raise SystemExit(f"no replay assignments found for {args.dataset}")
    records = []
    for path in files:
        stem = path.stem
        prefix, _, seedpart = stem.rpartition("__kmeans_s")
        dataset, fingerprint = prefix.split("__", 1)
        for row in _examples(path, args.top_n):
            row.update({"dataset": dataset, "fingerprint": fingerprint,
                        "seed": int(seedpart)})
            records.append(row)
    cols = ["dataset", "fingerprint", "seed", "selection_rank", "cluster_id",
            "cluster_size", "cluster_purity", "n_reference_labels",
            "modal_reference_label", "modal_label_count", "r_id", "reaction",
            "reference_label", "valid", "selection_rule"]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records, columns=cols).to_csv(out, index=False)
    print(json.dumps({"dataset": args.dataset, "files": len(files),
                      "examples": len(records), "output": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
