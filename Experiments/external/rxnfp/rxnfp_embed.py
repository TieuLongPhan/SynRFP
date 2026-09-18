#!/usr/bin/env python
"""Pretrained RXNFP embeddings (256-d) for every exported reaction table.

Run in the `rxnfp` environment after 01_data/prepare_data.py (and, for the
enzyme-activity study, after 10_enzyme_activity/activity.py --export):
    python external/rxnfp/rxnfp_embed.py [--config ...] [--datasets ...]
Writes work/features/<rxn_key>/rxnfp_pretrained.npz (rows = reaction table).

The encoder weights ship inside the `rxnfp` wheel (see README.md). `--model`
selects one of them: `bert_ft` (default; the published RXNFP fingerprint,
fine-tuned on Pistachio reaction classes) or `bert_pretrained` (masked-LM
pre-training only, no class labels).
"""
import argparse

import numpy as np
from rxnfp.transformer_fingerprints import (RXNBERTFingerprintGenerator,
                                            get_default_model_and_tokenizer)

from common import (load_config, reaction_keys, reaction_table, rxn_key, rxnfp_input,
                    save_features)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--model", default="bert_ft", choices=["bert_ft", "bert_pretrained"],
                    help="encoder shipped with the rxnfp package")
    args = ap.parse_args()
    cfg = load_config(args.config)
    model, tokenizer = get_default_model_and_tokenizer(model=args.model)
    print(f"RXNFP encoder: {args.model}")
    gen = RXNBERTFingerprintGenerator(model, tokenizer)
    keys = ([rxn_key(cfg, d) for d in args.datasets] if args.datasets
            else reaction_keys(cfg))
    for key in dict.fromkeys(keys):
        out = cfg["paths"]["work"] / "features" / key / "rxnfp_pretrained.npz"
        if out.exists():
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
        print(f"{key}: {ok.sum()}/{len(ok)} embedded -> {out}")


if __name__ == "__main__":
    main()
