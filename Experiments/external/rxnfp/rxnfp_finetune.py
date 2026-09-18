#!/usr/bin/env python
"""Fine-tuned RXNFP: fine-tune the BERT encoder on each training split, then embed.

For every (dataset, scheme, seed) the encoder is fine-tuned on the training
records only (validation records for early stopping), so the resulting
embeddings contain no test-label information. Embeddings of all rows of the
dataset table are written to
    work/features/rxnfp_ft/<dataset>/<scheme>_s<seed>.npz
and are then consumed by 03_classification/classify.py like any fingerprint.

Run in the `rxnfp` environment (GPU strongly recommended). Hyperparameters
follow the RXNFP classification example (Schwaller et al., 2021).
"""
import argparse
import shutil

import numpy as np
import pandas as pd
import pkg_resources
import torch
from rxnfp.models import SmilesClassificationModel
from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator
from transformers import BertModel

from common import dataset_table, load_config, rxnfp_input, save_features, split_codes

TRAIN, VAL = 0, 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--schemes", nargs="*", default=["published", "random", "grouped"])
    ap.add_argument("--seeds", nargs="*", type=int)
    ap.add_argument("--epochs", type=int, default=10)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seeds = args.seeds if args.seeds is not None else cfg["splits"]["replicate_seeds"]
    base = pkg_resources.resource_filename("rxnfp", "models/transformers/bert_pretrained")
    work = cfg["paths"]["work"]

    for ds_name in args.datasets or list(cfg["datasets"]):
        try:
            ds = dataset_table(cfg, ds_name)
        except FileNotFoundError as e:
            print(e)
            continue
        text = [rxnfp_input(s) for s in ds["rxn"]]
        y = ds["y"].to_numpy()
        for scheme in args.schemes:
            if scheme == "published" and cfg["datasets"][ds_name]["split"] != "published":
                continue
            for seed in seeds:
                out = work / "features" / "rxnfp_ft" / ds_name / f"{scheme}_s{seed}.npz"
                if out.exists():
                    continue
                code = split_codes(cfg, ds_name, scheme, seed)
                txt = np.asarray(text)
                tr = pd.DataFrame({"text": txt[code == TRAIN], "labels": y[code == TRAIN]})
                va = pd.DataFrame({"text": txt[code == VAL], "labels": y[code == VAL]})
                run_dir = work / "rxnfp_ft_models" / ds_name / f"{scheme}_s{seed}"
                model = SmilesClassificationModel(
                    "bert", base, num_labels=int(y.max()) + 1,
                    use_cuda=torch.cuda.is_available(),
                    args={"output_dir": str(run_dir), "overwrite_output_dir": True,
                          "num_train_epochs": args.epochs, "learning_rate": 2e-5,
                          "gradient_accumulation_steps": 1, "regression": False,
                          "manual_seed": seed, "evaluate_during_training": True,
                          "early_stopping_patience": 3, "save_steps": -1,
                          "save_model_every_epoch": False})
                model.train_model(tr, eval_df=va)
                best = run_dir / "best_model"
                bert = BertModel.from_pretrained(str(best if best.exists() else run_dir))
                gen = RXNBERTFingerprintGenerator(bert.eval(), model.tokenizer)
                X = np.zeros((len(text), 256), dtype=np.float32)
                ok = np.zeros(len(text), dtype=bool)
                for i in range(0, len(text), 256):
                    try:
                        X[i: i + 256] = np.asarray(gen.convert_batch(text[i: i + 256]))
                        ok[i: i + 256] = True
                    except Exception:
                        pass
                save_features(out, X, ok)
                shutil.rmtree(run_dir, ignore_errors=True)  # keep only embeddings
                print(f"{ds_name} {scheme} s{seed}: {ok.sum()}/{len(ok)} -> {out}")


if __name__ == "__main__":
    main()
