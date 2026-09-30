#!/usr/bin/env python
"""Create deterministic scaffold-grouped inner folds inside one r2 outer fold.

Run in the SynRFP Python environment, where the current scikit-learn provides
StratifiedGroupKFold. CUDA workers read the resulting integer codes and do not
need scikit-learn's split implementation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from synrfp_exp.data import dataset_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outer-seed", type=int, default=0)
    ap.add_argument("--inner-seed", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    metadata_path = args.out.with_suffix(".json")
    if args.out.exists() and metadata_path.exists():
        prior = json.loads(metadata_path.read_text())
        if "aggregate_encoded_label" in prior:
            print("Already exists:", args.out)
            return
    from synrfp_exp.config import load_config
    cfg = load_config()
    ds = dataset_table(cfg, "syntemp_r2")
    y = ds["y"].to_numpy()
    rows = ds["row"].to_numpy()
    aggregate = np.unique(y[ds["label"].astype(str).to_numpy() == "679"])
    if len(aggregate) != 1:
        raise RuntimeError("source label 679 does not map to one retained r2 class")
    outer = np.load(cfg["paths"]["work"] / "splits" / "syntemp_r2" / ("cv_s%d.npy" % args.outer_seed))
    groups = np.load(cfg["paths"]["work"] / "groups" / "syntemp.npy", allow_pickle=True)[rows]
    dev = np.flatnonzero(outer != 2)
    codes = np.full((3, len(y)), -1, dtype=np.int8)
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=args.inner_seed)
    sizes = []
    for fold, (tr_pos, va_pos) in enumerate(splitter.split(dev, y[dev], groups[dev])):
        train, val = dev[tr_pos], dev[va_pos]
        if set(groups[train]) & set(groups[val]):
            raise RuntimeError("scaffold overlap in generated inner split")
        codes[fold, train] = 10
        codes[fold, val] = 0
        sizes.append({"fold": fold, "train": len(train), "validation": len(val),
                      "train_scaffolds": int(np.unique(groups[train]).size),
                      "validation_scaffolds": int(np.unique(groups[val]).size)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.out.exists():
        np.save(args.out, codes)
    meta = {"outer_seed": args.outer_seed, "inner_seed": args.inner_seed,
            "outer_test_rows": int((outer == 2).sum()), "development_rows": len(dev),
            "aggregate_source_label": "679", "aggregate_encoded_label": int(aggregate[0]),
            "folds": sizes}
    metadata_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
