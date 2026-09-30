#!/usr/bin/env python
"""E0. Build dataset tables and all train/val/test splits; write a data summary.

Outputs: work/tables, work/groups, work/splits, 01_data/results/summary.csv,
and work/export/ (plain CSV copies read by the separate RXNFP environment).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Experiments/
import argparse

import pandas as pd

from synrfp_exp.config import load_config, log, setup_logging, results_dir
from synrfp_exp.data import dataset_table, export_table
from synrfp_exp.splits import make_split, scheme_seeds, split_summary


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
        y = ds["y"].to_numpy()
        for scheme, seed in scheme_seeds(cfg, name):
            try:
                code = make_split(cfg, name, scheme, seed)
            except (FileNotFoundError, ValueError) as e:
                log.error("%s %s s%d skipped: %s", name, scheme, seed, e)
                continue
            rows.append({"dataset": name, "scheme": scheme, "seed": seed,
                         "records": len(ds), "classes": int(ds["y"].nunique()),
                         **split_summary(code, y)})
        export_table(cfg, name, ds)
        log.info("%s: splits ready", name)
    out = results_dir("01_data") / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
