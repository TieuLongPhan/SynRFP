#!/usr/bin/env python
"""Pretrained RXNFP embeddings (256-d) for every exported reaction table.

Run in the `rxnfp` environment after 01_data/prepare_data.py (and, for the
enzyme-activity study, after 10_enzyme_activity/activity.py --export-only):
    python external/rxnfp/rxnfp_embed.py [--config ...] [--datasets ...]

Encoders (exp.md §4 has provenance and which folders ship in the PyPI wheel
vs. must be downloaded from GitHub into rxnfp/models/transformers/):

  rxnfp            bert_ft            all reaction tables
  rxnfp_mlm        bert_pretrained    all reaction tables (label-free control)
  rxnfp_tpl        bert_class_1k_tpl  tpl_u, tpl_b only (Track B, GitHub-only)
  rxnfp_schneider  bert_ft_10k_25s    schneider_u, schneider_b only (Track B, GitHub-only)

`--datasets` restricts `rxnfp`/`rxnfp_mlm` to the given reaction tables;
`rxnfp_tpl`/`rxnfp_schneider` always stay restricted to their own dataset
(never used outside it -- USPTO 1k TPL is a superset of USPTO-50k/Schneider,
so their test reactions may be in its training data).

Writes work/features/<rxn_key>/<fingerprint>.npz (rows = reaction table).
`--check-weights` compares bert_ft and bert_class_1k_tpl's word-embedding
weights and exits: if identical, bert_ft was trained on TPL labels too, and
rxnfp_mlm should be relied on as the fair label-free row (exp.md §4).
"""
import argparse

import numpy as np
from rxnfp.transformer_fingerprints import (RXNBERTFingerprintGenerator,
                                            get_default_model_and_tokenizer)

from common import (load_config, reaction_keys, reaction_table, rxn_key,
                    rxnfp_input, save_features)

# fingerprint -> (encoder folder under rxnfp/models/transformers/, reaction
# tables it applies to; None = every exported reaction table)
ENCODERS = {
    "rxnfp": ("bert_ft", None),
    "rxnfp_mlm": ("bert_pretrained", None),
    "rxnfp_tpl": ("bert_class_1k_tpl", ["tpl_u", "tpl_b"]),
    "rxnfp_schneider": ("bert_ft_10k_25s", ["schneider_u", "schneider_b"]),
}


def embed(cfg, fingerprint, model_name, keys, batch):
    try:
        model, tokenizer = get_default_model_and_tokenizer(model=model_name)
    except Exception as e:
        print(f"{fingerprint} ({model_name}): encoder unavailable, skipped ({e})")
        return
    gen = RXNBERTFingerprintGenerator(model, tokenizer)
    for key in dict.fromkeys(keys):
        out = cfg["paths"]["work"] / "features" / key / f"{fingerprint}.npz"
        if out.exists():
            continue
        rsmis = [rxnfp_input(s) for s in reaction_table(cfg, key)["rxn"]]
        X = np.zeros((len(rsmis), 256), dtype=np.float32)
        ok = np.zeros(len(rsmis), dtype=bool)
        for i in range(0, len(rsmis), batch):
            b = rsmis[i: i + batch]
            try:
                X[i: i + len(b)] = np.asarray(gen.convert_batch(b))
                ok[i: i + len(b)] = True
            except Exception:
                for j, s in enumerate(b):  # isolate failing reactions
                    try:
                        X[i + j] = np.asarray(gen.convert(s))
                        ok[i + j] = True
                    except Exception:
                        pass
        save_features(out, X, ok)
        print(f"{key}: {ok.sum()}/{len(ok)} embedded -> {out}")


def check_weights():
    """Max abs diff of embeddings.word_embeddings.weight, bert_ft vs bert_class_1k_tpl."""
    import pkg_resources
    from transformers import BertModel

    base = pkg_resources.resource_filename("rxnfp", "models/transformers")
    try:
        ft = BertModel.from_pretrained(f"{base}/bert_ft")
        tpl = BertModel.from_pretrained(f"{base}/bert_class_1k_tpl")
    except OSError as e:
        print(f"weight check skipped: {e}")
        return
    a = ft.embeddings.word_embeddings.weight.detach().numpy()
    b = tpl.embeddings.word_embeddings.weight.detach().numpy()
    if a.shape != b.shape:
        print(f"DIFFERENT: word-embedding shapes bert_ft={a.shape}, "
              f"bert_class_1k_tpl={b.shape}")
        return
    diff = float(np.abs(a - b).max())
    print(f"max abs diff, embeddings.word_embeddings.weight (bert_ft vs bert_class_1k_tpl): {diff}")
    if diff == 0.0:
        print("IDENTICAL: bert_ft was trained on TPL labels too; rely on rxnfp_mlm as the "
              "fair label-free row (exp.md §4)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--fingerprints", nargs="*", choices=list(ENCODERS), default=list(ENCODERS))
    ap.add_argument("--datasets", nargs="*",
                    help="restrict rxnfp/rxnfp_mlm to these reaction tables")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--check-weights", action="store_true",
                    help="compare bert_ft and bert_class_1k_tpl weights and exit")
    args = ap.parse_args()
    cfg = load_config(args.config)

    if args.check_weights:
        check_weights()
        return

    all_keys = ([rxn_key(cfg, d) for d in args.datasets] if args.datasets
               else reaction_keys(cfg))
    for fingerprint in args.fingerprints:
        model_name, restrict = ENCODERS[fingerprint]
        keys = restrict if restrict is not None else all_keys
        print(f"{fingerprint}: encoder {model_name}, "
              f"{len(dict.fromkeys(keys))} reaction table(s)")
        embed(cfg, fingerprint, model_name, keys, args.batch)


if __name__ == "__main__":
    main()
