#!/usr/bin/env python
"""Step 5. Measured encoding, normalization, and exact-search cost.

Run this alone on an idle machine. Encoding is single-process: one warm-up
pass, then the best and median of `repeats` passes over a fixed sample per
endpoint. Peak resident memory is the process high-water mark after each
step. Exact top-k Tanimoto search runs on packed bits with a byte popcount
table, one query at a time, on corpora resampled from real fingerprints, so
latency reflects payload width and corpus size and carries no information
about retrieval quality.

Learned encoders are timed by 05b_cost_rxnfp.py in their own environment on
the same sample files (runs/cost/sample__<dataset>.csv).

Outputs: runs/cost/encoding.csv, search.csv, hardware.json
"""
from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np
import pandas as pd

import vlib as L
from synrfp_exp.config import default_config, log, setup_logging
from synrfp_exp.data import dataset_table
from synrfp_exp.features import _drfp_chunk, _rdkit_chunk, _synrfp_chunk

POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint16)


def timed(fn, repeats):
    fn()
    t = []
    for _ in range(repeats):
        s = time.perf_counter()
        fn()
        t.append(time.perf_counter() - s)
    return min(t), float(np.median(t))


def topk_packed(Q, D, dn, k=10):
    idx = np.empty((len(Q), k), dtype=np.int64)
    for i, q in enumerate(Q):
        inter = POP[np.bitwise_and(D, q)].sum(1, dtype=np.int32)
        union = dn + int(POP[q].sum()) - inter
        s = np.divide(inter, union, out=np.zeros(len(D), dtype=np.float32), where=union > 0)
        part = np.argpartition(-s, k - 1)[:k]
        idx[i] = part[np.argsort(-s[part], kind="stable")]
    return idx


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    setup_logging()
    st = L.study_settings()
    c = st["cost"]
    raw_cfg, cfg = L.load_policy("raw"), L.load_policy(st["primary_policy"])
    out = L.RUNS / "cost"
    out.mkdir(parents=True, exist_ok=True)
    d = default_config(cfg)
    enc, packed = [], {}
    for ds in c["datasets"]:
        if not (cfg["paths"]["work"] / "tables" / f"ds_{ds}.pkl").exists():
            continue
        table, raw_table = dataset_table(cfg, ds), dataset_table(raw_cfg, ds)
        rng = np.random.default_rng(0)
        idx = rng.choice(len(table), min(c["encode_sample"], len(table)), replace=False)
        rs = table["rxn"].to_numpy()[idx].tolist()
        raw = raw_table["rxn"].to_numpy()[idx].tolist()
        pd.DataFrame({"rxn": rs}).to_csv(out / f"sample__{ds}.csv", index=False)
        tasks = {
            "input policy n1": lambda: [L.apply_policy(s, "n1") for s in raw],
            "SynRFP": lambda: _synrfp_chunk(rs, d.tokenizer, d.radius, [d], cfg["synrfp"],
                                            cfg["fp_seed"]),
            "DRFP": lambda: _drfp_chunk(rs, cfg["baselines"]["drfp"]),
            "RDKit difference": lambda: _rdkit_chunk(rs, "rdkit_diff"),
            "RDKit structural": lambda: _rdkit_chunk(rs, "rdkit_struct"),
        }
        for name, fn in tasks.items():
            best, med = timed(fn, c["repeats"])
            enc.append({"dataset": ds, "step": name, "n": len(rs), "best_seconds": best,
                        "median_seconds": med, "ms_per_reaction": 1000 * best / len(rs),
                        "reactions_per_second": len(rs) / best,
                        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})
            log.info("%s", enc[-1])
        if not packed:
            packed["SynRFP"] = tasks["SynRFP"]()[0][d.name]
            packed["DRFP"] = tasks["DRFP"]()[0]
        pd.DataFrame(enc).to_csv(out / "encoding.csv", index=False)

    search = []
    rng = np.random.default_rng(1)
    for fp, base in packed.items():
        for n in c["corpus_sizes"]:
            D = base[rng.integers(0, len(base), n)]
            Q = base[rng.integers(0, len(base), c["queries"])]
            dn = POP[D].sum(1, dtype=np.int32)
            best, med = timed(lambda: topk_packed(Q, D, dn), 3)
            search.append({"fingerprint": fp, "bytes_per_reaction": D.shape[1], "corpus": n,
                           "corpus_megabytes": D.nbytes / 1e6, "queries": len(Q),
                           "ms_per_query": 1000 * best / len(Q),
                           "median_ms_per_query": 1000 * med / len(Q)})
            log.info("%s", search[-1])
    pd.DataFrame(search).to_csv(out / "search.csv", index=False)
    (out / "hardware.json").write_text(json.dumps(
        {"lscpu": L._run(["lscpu"]), "memory": L._run(["free", "-g"]),
         "loadavg_at_end": open("/proc/loadavg").read().strip(),
         "note": "single process; numpy may use several BLAS threads only in search"}, indent=1))
    L.write_manifest("05_cost", [out / "encoding.csv", out / "search.csv", out / "hardware.json"])


if __name__ == "__main__":
    main()
