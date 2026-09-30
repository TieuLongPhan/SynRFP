#!/usr/bin/env python
"""Summarize reaction-level clustering replays for presentation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assignments", default="Experiments/11_interpretation/results/assignments")
    ap.add_argument("--output", default="Experiments/11_interpretation/results/cluster_replay_summary.csv")
    ap.add_argument("--figure", default="Experiments/11_interpretation/results/cluster_replay_summary.png")
    args = ap.parse_args()

    records = []
    for path in sorted(Path(args.assignments).glob("*.json")):
        obj = json.loads(path.read_text())
        # Mech L2 has two hierarchy levels; use the endpoint being replayed.
        scores = obj["scores_on_valid_rows"]
        level = "level2" if "level2" in scores else "label"
        s = scores[level]
        records.append({
            "dataset": obj["dataset"],
            "fingerprint": obj["fingerprint"],
            "seed": obj["seed"],
            "label_level": level,
            "n_rows": obj["n_rows"],
            "n_valid": obj["n_valid"],
            "invalid_rows": obj["invalid_rows"],
            "n_clusters": s["n_clusters"],
            "ami": s["ami"],
            "ari": s["ari"],
            "completeness": s["completeness"],
            "homogeneity": s["homogeneity"],
            "singleton_fraction": s["singleton_fraction"],
        })
    if not records:
        raise SystemExit("no replay metadata found")
    df = pd.DataFrame(records)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    order = ["synrfp-wl-r2-b1024-parity-delta", "drfp", "rdkit_diff",
             "rdkit_struct", "rxnfp_mlm"]
    names = {"synrfp-wl-r2-b1024-parity-delta": "SynRFP", "drfp": "DRFP",
             "rdkit_diff": "RDKit diff", "rdkit_struct": "RDKit struct.",
             "rxnfp_mlm": "RXNFP MLM"}
    colors = {"synrfp-wl-r2-b1024-parity-delta": "#0072B2", "drfp": "#D55E00",
              "rdkit_diff": "#009E73", "rdkit_struct": "#CC79A7",
              "rxnfp_mlm": "#E69F00"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (dataset, sub) in zip(axes, df.groupby("dataset", sort=True)):
        indexed = sub.set_index("fingerprint")
        sub = indexed.reindex([x for x in order if x in indexed.index]).reset_index()
        x = range(len(sub))
        ax.bar([i - 0.18 for i in x], sub["ami"], width=0.36,
               color=[colors[f] for f in sub.fingerprint], label="AMI")
        ax.bar([i + 0.18 for i in x], sub["ari"], width=0.36,
               color=[colors[f] for f in sub.fingerprint], alpha=0.42, label="ARI")
        ax.set_title(dataset.replace("_", " ").title())
        ax.set_xticks(list(x), [names[f] for f in sub.fingerprint], rotation=35, ha="right")
        ax.set_ylim(0, 0.85)
        ax.grid(axis="y", alpha=0.25)
        for i, row in sub.iterrows():
            ax.text(i, max(row.ami, row.ari) + 0.02, f"n={int(row.n_valid):,}",
                    ha="center", va="bottom", fontsize=7, rotation=90)
    axes[0].set_ylabel("Clustering agreement")
    axes[0].legend(loc="upper left", frameon=False)
    fig.suptitle("Fixed-seed reaction-level clustering replay")
    fig.tight_layout()
    fig.savefig(args.figure, dpi=220)
    plt.close(fig)
    print(json.dumps({"rows": len(df), "summary": str(out), "figure": str(args.figure)}, sort_keys=True))


if __name__ == "__main__":
    main()
