#!/usr/bin/env python
"""Audit reaction-centre localization under the declared reaction protocol.

This is a small, read-only pilot.  It compares the mapped original reaction
with the mechanistic ``updated_reaction`` column and compares the main SynRFP
radius (2) with the historical E8 radius (3).  The output is intentionally
labelled a protocol audit, not a replacement for the full reaction-centre
analysis.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "Experiments"
sys.path.insert(0, str(EXP))


def load_e8():
    path = EXP / "08_reaction_center" / "reaction_center.py"
    spec = importlib.util.spec_from_file_location("synrfp_e8", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(EXP / "config.yaml"))
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=str(
        EXP / "11_interpretation" / "results" / "reaction_center_protocol_audit.csv"))
    args = ap.parse_args()

    from synrfp_exp.config import load_config

    cfg = load_config(args.config)
    raw = pd.read_csv(cfg["paths"]["mech_raw"])
    n = min(args.sample, len(raw))
    raw = raw.sample(n=n, random_state=args.seed).reset_index(drop=True)
    e8 = load_e8()

    rows = []
    for column, protocol in (("original_reactions", "primary_original"),
                             ("updated_reaction", "historical_updated")):
        for radius in (2, 3):
            values = [e8.analyse(str(s), radius, cfg["synrfp"])
                      for s in raw[column]]
            values = [v for v in values if v is not None]
            if not values:
                raise RuntimeError(f"no valid rows for {protocol}, radius {radius}")
            rows.append({
                "protocol": protocol,
                "reaction_column": column,
                "radius": radius,
                "sample_requested": n,
                "sample_valid": len(values),
                "valid_fraction": len(values) / n,
                "delta_center_fraction_mean": float(np.nanmean(
                    [v["delta_center_fraction"] for v in values])),
                "delta_occurrence_center_fraction_mean": float(np.nanmean(
                    [v["delta_occurrence_center_fraction"] for v in values])),
                "union_center_fraction_mean": float(np.nanmean(
                    [v["union_center_fraction"] for v in values])),
                "center_coverage_mean": float(np.nanmean(
                    [v["center_coverage"] for v in values])),
                "delta_token_fraction_mean": float(np.nanmean(
                    [v["delta_token_fraction"] for v in values])),
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    meta = out.with_suffix(".json")
    meta.write_text(json.dumps({
        "sample": n,
        "seed": args.seed,
        "note": "Pilot protocol audit; not the full E8 result.",
        "primary_protocol": "original_reactions, radius 2",
        "historical_protocol": "updated_reaction, radius 3",
    }, indent=2) + "\n")
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
