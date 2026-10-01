#!/usr/bin/env python
"""Step 5b. RXNFP model-load and embedding time (run in the RXNFP env).

Python 3.8 compatible. Reads the samples written by 05_cost.py and times the
label-free encoder on GPU (if available) and on CPU, separately from model
loading. Training cost of the released checkpoint is not known and is not
estimated. Output: runs/cost/encoding_rxnfp.csv
"""
import os
import time
from pathlib import Path

import pandas as pd
import torch
from rxnfp.transformer_fingerprints import (RXNBERTFingerprintGenerator,
                                            get_default_model_and_tokenizer)

HERE = Path(__file__).resolve().parent
ROOT = HERE / "smoke" if os.environ.get("VALIDATION_SMOKE") else HERE
OUT = ROOT / "runs" / "cost"


def rxnfp_input(rsmi):
    parts = rsmi.split(" ")[0].split(">")
    return ".".join(p for p in (parts[0], parts[1]) if p) + ">>" + parts[2]


def main():
    rows = []
    devices = (["cuda"] if torch.cuda.is_available() else []) + ["cpu"]
    for device in devices:
        t = time.perf_counter()
        model, tokenizer = get_default_model_and_tokenizer(
            model="bert_pretrained", force_no_cuda=(device == "cpu"))
        gen = RXNBERTFingerprintGenerator(model, tokenizer, force_no_cuda=(device == "cpu"))
        load = time.perf_counter() - t
        for path in sorted(OUT.glob("sample__*.csv")):
            rs = [rxnfp_input(s) for s in pd.read_csv(path).rxn.astype(str)]
            if device == "cpu":
                rs = rs[:500]            # CPU inference is slow; a smaller sample suffices
            ok, best = 0, None
            for rep in range(3):
                t = time.perf_counter()
                ok = 0
                for i in range(0, len(rs), 64):
                    try:
                        gen.convert_batch(rs[i:i + 64])
                        ok += len(rs[i:i + 64])
                    except Exception:
                        pass
                dt = time.perf_counter() - t
                best = dt if best is None or (rep > 0 and dt < best) else best
            rows.append({"dataset": path.stem.replace("sample__", ""), "step": "RXNFP-MLM",
                         "device": device, "n": len(rs), "n_encoded": ok,
                         "model_load_seconds": load, "best_seconds": best,
                         "ms_per_reaction": 1000 * best / len(rs),
                         "gpu_peak_mb": (torch.cuda.max_memory_allocated() / 1e6
                                         if device == "cuda" else 0.0)})
            print(rows[-1])
    pd.DataFrame(rows).to_csv(OUT / "encoding_rxnfp.csv", index=False)


if __name__ == "__main__":
    main()
