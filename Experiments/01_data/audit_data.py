#!/usr/bin/env python
"""E1. Data audit: duplicates, label conflicts, and train/test leakage per split.

* duplicates: records whose canonical reaction occurs more than once;
* label conflicts: identical canonical reactions carrying different labels
  (an irreducible error for any reaction-only representation);
* leakage: test records whose canonical reaction also occurs in training.

Outputs: 01_data/results/audit.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Experiments/
import argparse

import numpy as np
import pandas as pd

from synrfp_exp.config import load_config, log, setup_logging, results_dir
from synrfp_exp.data import canonical_for, dataset_table
from synrfp_exp.splits import TEST, TRAIN, make_split


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--datasets", nargs="*")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    rows = []
    for name in args.datasets or list(cfg["datasets"]):
        try:
            ds = dataset_table(cfg, name)
        except FileNotFoundError as e:
            log.error("%s skipped: %s", name, e)
            continue
        can = canonical_for(cfg, name)
        df = pd.DataFrame({"can": can, "y": ds["y"].to_numpy()})
        valid = df["can"] != ""
        grp = df[valid].groupby("can")["y"]
        n_per = grp.transform("size")
        n_lab = grp.transform("nunique")
        # best achievable accuracy if every conflicting group predicts its majority label
        majority = df[valid].groupby("can")["y"].agg(lambda s: s.value_counts().iloc[0]).sum()
        base = {"dataset": name, "records": len(df),
                "unparsable": int((~valid).sum()),
                "unique_reactions": int(df.loc[valid, "can"].nunique()),
                "duplicate_records": int((n_per > 1).sum()),
                "label_conflict_records": int((n_lab > 1).sum()),
                "accuracy_ceiling": float((majority + (~valid).sum()) / len(df))}
        for scheme in ("published", "random", "grouped"):
            p = cfg["paths"]["work"] / "splits" / name / f"{scheme}_s0.npy"
            if not p.exists():
                continue
            code = make_split(cfg, name, scheme, 0)
            train_set = set(df.loc[(code == TRAIN) & valid, "can"])
            te = (code == TEST) & valid
            leak = df.loc[te, "can"].isin(train_set)
            rows.append({**base, "scheme": scheme,
                         "test_in_train_fraction": float(leak.mean()) if te.any() else np.nan})
        log.info("%s audited", name)
    out = results_dir("01_data") / "audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
