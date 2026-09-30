#!/usr/bin/env python
"""Measure SynRFP bit stability across fixed hash seeds on real reactions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from synrfp_exp.config import load_config, SynRFPConfig  # noqa: E402
from synrfp_exp.data import dataset_table  # noqa: E402
from synrfp_exp.features import _synrfp_chunk  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="Experiments/config.yaml")
    ap.add_argument("--dataset", default="schneider_b")
    ap.add_argument("--max-rows", type=int, default=2000)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--output", default="Experiments/11_interpretation/results/hash_seed_sensitivity.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = dataset_table(cfg, args.dataset).head(args.max_rows)
    c = SynRFPConfig.from_name("synrfp-wl-r2-b1024-parity-delta")
    arrays = []
    oks = []
    for seed in args.seeds:
        out, ok = _synrfp_chunk(ds["rxn"].astype(str).tolist(), c.tokenizer,
                                c.radius, [c], cfg["synrfp"], seed)
        arrays.append(out[c.name])
        oks.append(ok)
    common = np.logical_and.reduce(oks)
    unpacked = [np.unpackbits(x[common], axis=1, count=c.bits) for x in arrays]
    rows = []
    for i, left in enumerate(args.seeds):
        for j in range(i + 1, len(args.seeds)):
            h = np.count_nonzero(unpacked[i] != unpacked[j], axis=1)
            rows.append({"dataset": args.dataset, "fingerprint": c.name,
                         "seed_a": left, "seed_b": args.seeds[j],
                         "n_rows": len(ds), "n_common_valid": int(common.sum()),
                         "mean_hamming": float(h.mean()),
                         "median_hamming": float(np.median(h)),
                         "p95_hamming": float(np.percentile(h, 95)),
                         "mean_fraction_changed": float(h.mean() / c.bits)})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(json.dumps({"rows": len(rows), "n_common_valid": int(common.sum()),
                      "output": str(out)}, sort_keys=True))


if __name__ == "__main__":
    main()
