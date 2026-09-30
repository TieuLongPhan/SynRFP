#!/usr/bin/env python
"""E3. Supervised classification.

--study benchmark   SynRFP default vs baselines; models x (scheme, seed) pairs
                    from benchmark.schemes
--study factorial   162 SynRFP configurations; random splits, config seeds

One JSON per run under 03_classification/results/<study>/; existing runs are
skipped, so the script can be interrupted and restarted.
"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Experiments/
import argparse

import numpy as np

from synrfp_exp import metrics
from synrfp_exp.config import (SynRFPConfig, expand_fingerprints, load_config,
                               log, setup_logging, write_json, results_dir)
from synrfp_exp.data import dataset_table
from synrfp_exp.features import load_for_run
from synrfp_exp.models import Skip, fit_predict
from synrfp_exp.scopes import factorial_scope, full_scope
from synrfp_exp.splits import TEST, TRAIN, VAL, make_split, random_codes, scheme_seeds

# Models that use the validation rows (early stopping); the others train on
# train + validation. The MLP holds out its own early-stopping fraction.
USES_VALIDATION = {"xgb"}
# Models whose result does not depend on the seed once the split is fixed.
DETERMINISTIC = {"knn"}


def claim_result(out):
    """Claim one result path atomically before doing an expensive fit.

    Existing workers may be interrupted, so locks older than 24 hours are
    treated as stale.  The completed JSON remains the authoritative output.
    """
    if out.exists():
        return None
    lock = out.with_name(out.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        return lock
    except FileExistsError:
        try:
            if time.time() - lock.stat().st_mtime > 24 * 60 * 60:
                lock.unlink()
        except FileNotFoundError:
            pass
        return None


def release_result(lock):
    if lock is not None:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def run_one(cfg, study, dataset, scheme, seed, fp, model, sc, code, y, out):
    lock = claim_result(out)
    if lock is None:
        return
    log.info("START %s %s s%d %s %s", dataset, scheme, seed, fp, model)
    try:
        feats = load_for_run(cfg, sc.name, fp, dataset, scheme, seed)
    except FileNotFoundError as e:
        log.warning("skip %s: %s", fp, e)
        release_result(lock)
        return
    tr, va, te = (sc.feat_rows[code == c] for c in (TRAIN, VAL, TEST))
    ytr, yva = y[code == TRAIN], y[code == VAL]
    if model not in USES_VALIDATION:  # nothing to tune: validation rows join training
        tr, ytr = np.concatenate([tr, va]), np.concatenate([ytr, yva])
    try:
        pred = fit_predict(model, cfg["models"][model], feats, tr, ytr,
                           va, yva, te, seed=seed, n_jobs=cfg["n_jobs"])
    except Skip as e:
        log.info("skip %s/%s/%s: %s", dataset, fp, model, e)
        release_result(lock)
        return
    rec = {"study": study, "dataset": dataset, "scheme": scheme, "seed": seed,
           "fingerprint": fp, "model": model,
           "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
           "n_unfeaturised_test": int((~feats.ok[te]).sum()),
           **metrics.classification(y[code == TEST], pred)}
    if fp.startswith("synrfp-"):
        c = SynRFPConfig.from_name(fp)
        rec.update(tokenizer=c.tokenizer, radius=c.radius, bits=c.bits,
                   sketch=c.sketch, mode=c.mode)
    write_json(out, rec)
    release_result(lock)
    log.info("%s %s s%d %-40s %-6s MCC=%.3f", dataset, scheme, seed, fp, model, rec["mcc"])


def main():  # noqa: C901
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    ap.add_argument("--study", default="benchmark", choices=["benchmark", "factorial"])
    ap.add_argument("--datasets", nargs="*")
    ap.add_argument("--fingerprints", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--schemes", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int)
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    root = results_dir("03_classification") / args.study

    if args.study == "benchmark":
        study = cfg["benchmark"]
        datasets = args.datasets or list(cfg["datasets"])
        fps = expand_fingerprints(cfg, args.fingerprints or study["fingerprints"])
        for dataset in datasets:
            try:
                ds = dataset_table(cfg, dataset)
            except FileNotFoundError as e:
                log.error("%s skipped: %s", dataset, e)
                continue
            y, sc = ds["y"].to_numpy(), full_scope(cfg, dataset)
            runs = scheme_seeds(cfg, dataset, args.schemes, args.seeds)
            extra = expand_fingerprints(
                cfg, cfg.get("rxnfp_dataset_fingerprints", {}).get(dataset, []))
            for scheme, seed in runs:
                try:
                    code = make_split(cfg, dataset, scheme, seed)
                except (FileNotFoundError, ValueError) as e:
                    log.error("%s %s s%d skipped: %s", dataset, scheme, seed, e)
                    continue
                first = seed == min(s for sch, s in runs if sch == scheme)
                # Track B dataset-specific RXNFP encoders (exp.md §4): never
                # used outside their own dataset, so only added for "fixed".
                run_fps = fps + extra if scheme == "fixed" else fps
                for fp in run_fps:
                    for model in args.models or study["models"]:
                        if scheme == "fixed" and model in DETERMINISTIC and not first:
                            continue  # same split and features, same result
                        out = root / dataset / f"{scheme}_s{seed}" / f"{fp}__{model}.json"
                        run_one(cfg, "benchmark", dataset, scheme, seed, fp, model,
                                sc, code, y, out)
    else:
        study = cfg["factorial"]
        fps = expand_fingerprints(cfg, args.fingerprints or ["synrfp_grid"])
        seeds = args.seeds if args.seeds is not None else study["seeds"]
        for dataset in args.datasets or study["datasets"]:
            try:
                sc = factorial_scope(cfg, dataset)
            except FileNotFoundError as e:
                log.error("%s skipped: %s", dataset, e)
                continue
            y = dataset_table(cfg, dataset)["y"].to_numpy()[sc.ds_pos]
            per_model = study.get("seeds_per_model", {}) if args.seeds is None else {}
            for seed in seeds:
                code = random_codes(cfg, y, seed)
                for fp in fps:
                    for model in args.models or study["models"]:
                        if seed not in per_model.get(model, seeds):
                            continue
                        out = root / dataset / f"random_s{seed}" / f"{fp}__{model}.json"
                        run_one(cfg, "factorial", dataset, "random", seed, fp, model,
                                sc, code, y, out)


if __name__ == "__main__":
    main()
