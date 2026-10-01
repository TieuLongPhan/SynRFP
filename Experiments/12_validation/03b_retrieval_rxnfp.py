#!/usr/bin/env python
"""Step 3b. RXNFP embeddings for the retrieval proxy (run in the RXNFP env).

Python 3.8 compatible; needs only rxnfp, numpy, pandas. Run after
`03_retrieval_proxy.py --stage build`:

    $RXNFP_PY 12_validation/03b_retrieval_rxnfp.py            # full tree
    VALIDATION_SMOKE=1 $RXNFP_PY 12_validation/03b_retrieval_rxnfp.py

Writes runs/retrieval_proxy/features/{rxnfp,rxnfp_mlm}.npz in the layout that
`--stage score` reads. rxnfp is the label-trained encoder (supervised
reference); rxnfp_mlm has no reaction-label training.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
from rxnfp.transformer_fingerprints import (RXNBERTFingerprintGenerator,
                                            get_default_model_and_tokenizer)

HERE = Path(__file__).resolve().parent
ROOT = HERE / "smoke" if os.environ.get("VALIDATION_SMOKE") else HERE
OUT = ROOT / "runs" / "retrieval_proxy"
ENCODERS = {"rxnfp": "bert_ft", "rxnfp_mlm": "bert_pretrained"}


def rxnfp_input(rsmi):
    parts = rsmi.split(" ")[0].split(">")
    return ".".join(p for p in (parts[0], parts[1]) if p) + ">>" + parts[2]


def embed(gen, rsmis, batch=256):
    X = np.zeros((len(rsmis), 256), dtype=np.float32)
    ok = np.zeros(len(rsmis), dtype=bool)
    for i in range(0, len(rsmis), batch):
        b = rsmis[i:i + batch]
        try:
            X[i:i + len(b)] = np.asarray(gen.convert_batch(b))
            ok[i:i + len(b)] = True
        except Exception:
            for j, s in enumerate(b):
                try:
                    X[i + j] = np.asarray(gen.convert(s))
                    ok[i + j] = True
                except Exception:
                    pass
    return X, ok


def main():
    q = [rxnfp_input(s) for s in pd.read_csv(OUT / "queries.csv").rxn.astype(str)]
    c = [rxnfp_input(s) for s in pd.read_csv(OUT / "corpus.csv").rxn.astype(str)]
    (OUT / "features").mkdir(exist_ok=True)
    for name, model_name in ENCODERS.items():
        path = OUT / "features" / (name + ".npz")
        if path.exists():
            continue
        model, tokenizer = get_default_model_and_tokenizer(model=model_name)
        gen = RXNBERTFingerprintGenerator(model, tokenizer)
        Q, q_ok = embed(gen, q)
        C, c_ok = embed(gen, c)
        np.savez_compressed(path, Q=Q, C=C, n_bits=np.int64(0), q_ok=q_ok, c_ok=c_ok)
        print("%s: queries %d/%d, corpus %d/%d" % (name, q_ok.sum(), len(q), c_ok.sum(), len(c)))


if __name__ == "__main__":
    main()
