#!/usr/bin/env python
"""Pretrained RXNFP embeddings (256-d) for every reaction table.

Run in the `rxnfp` environment after 01_data/prepare_data.py:
    python external/rxnfp/rxnfp_embed.py [--config ...] [--datasets ...]
Writes work/features/<rxn_key>/rxnfp_pretrained.npz (rows = reaction table).
"""
import argparse

import numpy as np
from rxnfp.transformer_fingerprints import (RXNBERTFingerprintGenerator,
                                            get_default_model_and_tokenizer)

from common import load_config, reaction_table, rxn_key, rxnfp_input, save_features


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, tokenizer = get_default_model_and_tokenizer()
    gen = RXNBERTFingerprintGenerator(model, tokenizer)
    done = set()
    for ds in args.datasets or list(cfg["datasets"]):
        key = rxn_key(cfg, ds)
        out = cfg["paths"]["work"] / "features" / key / "rxnfp_pretrained.npz"
        if key in done or out.exists():
            continue
        rsmis = [rxnfp_input(s) for s in reaction_table(cfg, key)["rxn"]]
        X = np.zeros((len(rsmis), 256), dtype=np.float32)
        ok = np.zeros(len(rsmis), dtype=bool)
        for i in range(0, len(rsmis), args.batch):
            batch = rsmis[i: i + args.batch]
            try:
                X[i: i + len(batch)] = np.asarray(gen.convert_batch(batch))
                ok[i: i + len(batch)] = True
            except Exception:
                for j, s in enumerate(batch):  # isolate failing reactions
                    try:
                        X[i + j] = np.asarray(gen.convert(s))
                        ok[i + j] = True
                    except Exception:
                        pass
        save_features(out, X, ok)
        done.add(key)
        print(f"{key}: {ok.sum()}/{len(ok)} embedded -> {out}")


if __name__ == "__main__":
    main()
