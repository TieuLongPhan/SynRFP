#!/usr/bin/env python
"""Step 1. Serialization invariance of every label-free encoder, per policy.

For a fixed sample of each reaction table, every reaction is re-written with a
random atom order and shuffled components (chemically equivalent). The
fingerprint of the original and the re-written string are compared

  * as given (raw strings), and
  * after input policy n1 is applied to both.

A method passes when identity is 100 % on jointly valid rows. Rows where
either string fails to encode are reported, never counted as identical.

Outputs: runs/input_audit/serialization.csv, diagnostic_cases.csv
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

import vlib as L
from synrfp_exp.config import expand_fingerprints, log, setup_logging
from synrfp_exp.data import reaction_table, rxn_key
from synrfp_exp.features import compute_in_memory


def identity(cfg, a, b, fp):
    fa, fb = compute_in_memory(cfg, a, fp), compute_in_memory(cfg, b, fp)
    both = fa.ok & fb.ok
    same = np.all(fa.X[both] == fb.X[both], axis=1)
    return {"n_valid_original": int(fa.ok.sum()), "n_valid_rewritten": int(fb.ok.sum()),
            "n_jointly_valid": int(both.sum()),
            "identical_fraction": float(same.mean()) if both.any() else float("nan")}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*")
    args = ap.parse_args()
    setup_logging()
    study = L.study_settings()
    cfg = L.load_policy("raw")
    out = L.RUNS / "input_audit"
    out.mkdir(parents=True, exist_ok=True)
    fps = expand_fingerprints(cfg, study["input_audit"]["fingerprints"])
    datasets = args.datasets or list(cfg["datasets"])
    keys = list(dict.fromkeys(rxn_key(cfg, d) for d in datasets
                              if (cfg["paths"]["work"] / "tables"
                                  / f"rxn_{rxn_key(cfg, d)}.pkl").exists()))
    rows = []
    for key in keys:
        rt = reaction_table(cfg, key)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(rt), size=min(study["input_audit"]["sample"], len(rt)), replace=False)
        base = [rt["rxn"].iloc[i] for i in idx]
        rnd = [L.randomized_rsmi(s, 7 + int(i)) for s, i in zip(base, idx)]
        n_rewritable = sum(bool(s) for s in rnd)
        variants = {"raw": (base, rnd),
                    "n1": ([L.apply_policy(s, "n1")[0] for s in base],
                           [L.apply_policy(s, "n1")[0] if s else "" for s in rnd])}
        for policy, (a, b) in variants.items():
            for fp in fps:
                r = identity(cfg, a, b, fp)
                r.update(table=key, policy=policy, fingerprint=fp, n_sample=len(base),
                         n_rewritable=n_rewritable,
                         string_identical_fraction=float(np.mean([x == z for x, z in zip(a, b)])))
                rows.append(r)
                log.info("%s %s %-36s identical=%.4f (jointly valid %d/%d)", key, policy, fp,
                         r["identical_fraction"], r["n_jointly_valid"], len(base))
        pd.DataFrame(rows).to_csv(out / "serialization.csv", index=False)

    cases = []
    for kind, pairs in L.POLICY_CASES.items():
        for a, b in pairs:
            for fp in fps:
                for policy in ("raw", "n1"):
                    x = [L.apply_policy(a, policy)[0]]
                    z = [L.apply_policy(b, policy)[0]]
                    fa, fb = compute_in_memory(cfg, x, fp), compute_in_memory(cfg, z, fp)
                    cases.append({"expected": kind, "a": a, "b": b, "policy": policy,
                                  "fingerprint": fp, "both_valid": bool(fa.ok[0] and fb.ok[0]),
                                  "identical": bool(np.array_equal(fa.X[0], fb.X[0]))})
    pd.DataFrame(cases).to_csv(out / "diagnostic_cases.csv", index=False)
    L.write_manifest("01_input_audit", [out / "serialization.csv", out / "diagnostic_cases.csv"])


if __name__ == "__main__":
    main()
