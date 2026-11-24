"""Fit the 30 pinned QuantumPioneer cells, then score the frozen holdout once."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
from itertools import product
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from joblib import Parallel, delayed
import numpy as np
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

from audit_inputs import PINS, TARGETS, sha256
from freeze_qp_prediction import verified_artifacts
from paired_analysis import means_by_group, paired_bootstrap
from prepare_qp_features import ARMS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "73_context_attribution"))
import features
import run_ablation as e1

SEEDS = (17, 29, 43)
TARGET_NAMES = TARGETS["quantumpioneer"]


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def write(path, payload):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def matrices(archive, partitions):
    return tuple(np.unpackbits(archive[rows], axis=1).astype(np.float32) for rows in partitions)


def validate_fit_inputs(fit_y, validation_y):
    if not np.isfinite(fit_y).all() or not np.isfinite(validation_y).all():
        raise ValueError("Nonfinite fit/validation labels")
    if fit_y.std() == 0:
        raise ValueError("Constant training target")


def fit_cell(packed, parts, fit_y, validation_y, arm, target, seed, directory, plan_hash,
             shared_paths=None):
    final = directory / f"{target}__{arm}__s{seed}"
    if final.exists():
        saved = read(final / "COMPLETE.json")
        if saved["plan_sha256"] != plan_hash:
            raise ValueError(f"Existing fit is from a different plan: {final.name}")
        for name, expected in saved["outputs_sha256"].items():
            if sha256(final / name) != expected:
                raise ValueError(f"Existing fit artifact changed: {final.name}/{name}")
        return saved

    validate_fit_inputs(fit_y, validation_y)
    start = time.monotonic()
    if shared_paths is None:
        train, val, blind_test = matrices(packed, parts)
    else:
        train, val, blind_test = (
            np.load(path, mmap_mode="r", allow_pickle=False) for path in shared_paths
        )
    center, scale = float(fit_y.mean()), float(fit_y.std())
    y_train, y_val = (fit_y - center) / scale, (validation_y - center) / scale
    model = MLPRegressor(**e1.FIT_OPTIONS, random_state=np.random.RandomState(seed))
    best_loss, patience_loss, stale, best_epoch = np.inf, np.inf, 0, 0
    with threadpool_limits(limits=1):
        for epoch in range(1, e1.MAX_EPOCHS + 1):
            model.partial_fit(train, y_train)
            validation_loss = float(np.mean((model.predict(val) - y_val) ** 2))
            if not np.isfinite(validation_loss):
                raise ValueError("Nonfinite validation loss")
            if validation_loss < best_loss:
                best_loss, best_epoch = validation_loss, epoch
                best_parameters = ([v.copy() for v in model.coefs_],
                                   [v.copy() for v in model.intercepts_])
            if patience_loss - validation_loss >= e1.TOLERANCE:
                patience_loss, stale = validation_loss, 0
            else:
                stale += 1
            if stale >= e1.PATIENCE:
                break
        model.coefs_, model.intercepts_ = best_parameters
        prediction = np.asarray(model.predict(blind_test), dtype=np.float64) * scale + center
    if not np.isfinite(prediction).all() or prediction.shape != parts[2].shape:
        raise ValueError("Invalid held-out predictions")
    attempt = Path(tempfile.mkdtemp(prefix=f".{final.name}-", dir=directory))
    weights = attempt / "checkpoint.npz"
    np.savez_compressed(weights, mean=center, sd=scale,
                        **{f"coef{i}": v for i, v in enumerate(best_parameters[0])},
                        **{f"intercept{i}": v for i, v in enumerate(best_parameters[1])})
    score_input = attempt / "blind_predictions.npz"
    np.savez_compressed(score_input, matrix_row=parts[2], prediction=prediction)
    receipt = {
        "phase": "blind_fit_complete", "target": target, "arm": arm, "seed": seed,
        "plan_sha256": plan_hash, "epochs": epoch, "best_epoch": best_epoch,
        "validation_standardized_mse": best_loss, "seconds": time.monotonic() - start,
        "outputs_sha256": {p.name: sha256(p) for p in (weights, score_input)},
    }
    write(attempt / "COMPLETE.json", receipt)
    attempt.rename(final)
    print(final.name, f"{epoch} epochs, {receipt['seconds']:.1f}s", flush=True)
    return receipt


def blind_predictions(folder, receipts, test_indices):
    predictions = {}
    for receipt in receipts:
        key = tuple(receipt[k] for k in ("target", "arm", "seed"))
        suffix = f"{key[0]}__{key[1]}__s{key[2]}"
        path = folder / suffix / "blind_predictions.npz"
        if sha256(path) != receipt["outputs_sha256"][path.name]:
            raise ValueError(f"Changed predictions for {suffix}")
        with np.load(path, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["matrix_row"], test_indices)
            predictions[key] = saved["prediction"]
    if set(predictions) != set(product(TARGET_NAMES, ARMS, SEEDS)):
        raise ValueError("Fit grid does not contain all 30 verified cells")
    return predictions


def observed_test(source, indices, ids):
    expected = {int(i): row_id for i, row_id in zip(indices, ids, strict=True)}
    values, seen = {}, set()
    with source.open(newline="") as handle:
        for i, row in enumerate(csv.DictReader(handle)):
            if i not in expected:
                continue
            if row["source_record_id"] != expected[i] or row["proposed_split"] != "test":
                raise ValueError("Held-out row mismatch at scoring")
            seen.add(i)
            values[i] = [float(row[target]) for target in TARGET_NAMES]
    if seen != set(expected):
        raise ValueError("Missing held-out outcomes")
    observed = np.asarray([values[int(i)] for i in indices])
    if observed.shape != (len(indices), 2) or not np.isfinite(observed).all():
        raise ValueError("Expected finite held-out outcomes for both targets")
    return observed


def score(output, predictions, true_values, groups, train_targets, prediction):
    metrics, comparison_gains, names = [], [], []
    variance = train_targets.var(axis=0)
    for i, target in enumerate(TARGET_NAMES):
        y = true_values[:, i]
        arm_losses = {}
        for arm in ARMS:
            squared_errors = []
            for seed in SEEDS:
                errors = predictions[target, arm, seed] - y
                squared_errors.append(errors**2)
                metrics.append({"target": target, "arm": arm, "seed": seed,
                                "rmse": float(np.sqrt(np.mean(errors**2))),
                                "mae": float(np.mean(np.abs(errors)))})
            arm_losses[arm] = np.mean(squared_errors, axis=0) / variance[i]
        for arm in (key for key in ARMS if key != "d_pooled"):
            comparison_gains.append(arm_losses[arm] - arm_losses["d_pooled"])
            names.append((target, arm))
    group_gains = means_by_group(np.column_stack(comparison_gains), groups)
    draws = paired_bootstrap(group_gains, draws=10000, seed=2026)
    estimates = [{"target": target, "comparator": arm, "gain": float(group_gains[:, i].mean()),
                  "interval": np.quantile(draws[:, i], (0.025, 0.975)).tolist()}
                 for i, (target, arm) in enumerate(names)]
    order = prediction["predicted_higher_gain_target"]
    interaction = None
    if order is not None:
        high = names.index((order, "d2048"))
        low = next(i for i, (target, arm) in enumerate(names) if target != order and arm == "d2048")
        difference = draws[:, high] - draws[:, low]
        interaction = {"higher_ranked_target": order,
                       "gain_difference": float((group_gains[:, high] - group_gains[:, low]).mean()),
                       "interval": np.quantile(difference, (0.025, 0.975)).tolist()}
    np.savez_compressed(output / "group_bootstrap.npz", draws=draws, group_gains=group_gains,
                        group_ids=np.unique(groups), contrasts=np.array(names))
    write(output / "scores.json", {
        "scope": "primary_external_ordering" if interaction else "descriptive_after_diagnostic_abstention",
        "n_rows": len(true_values), "n_groups": len(group_gains),
        "native_metrics": metrics, "units": "kcal/mol",
        "normalized_gain": estimates, "interaction": interaction,
        "test_target_pearson_r": float(np.corrcoef(true_values.T)[0, 1]),
        "bootstrap_seed": 2026, "bootstrap_draws": len(draws),
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "features", "frozen", "development", "synthetic", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--shared-matrices", type=Path)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.jobs <= (16 if args.shared_matrices else 4):
        raise ValueError("Worker count exceeds the limit for this matrix-loading mode")
    repo = Path(__file__).resolve().parents[2]
    git = lambda *argv: subprocess.check_output(
        ["git", "-C", str(repo), *argv], text=True
    ).strip()
    if git("status", "--porcelain", "--", "Experiments/74_context_dependence", "synrfp", "Experiments/73_context_attribution"):
        raise ValueError("Commit the experiment and its feature code before external scoring")
    for folder, field, count in ((args.development, "development_fits", 60),
                                 (args.synthetic, "fits", 150)):
        receipt = read(folder / "COMPLETE.json")
        if receipt["phase"] != "complete" or receipt[field] != count:
            raise ValueError(f"Missing completed development gates: {folder}")
        if field == "development_fits":
            for name, expected in receipt["completed_sha256"].items():
                if sha256(folder / name / "COMPLETE.json") != expected:
                    raise ValueError(f"Changed development receipt for {name}")
        elif sha256(folder / "fit_scores.csv") != receipt["fit_scores_sha256"]:
            raise ValueError("Changed synthetic control scores")
    if sha256(args.source) != PINS["quantumpioneer"][1]:
        raise ValueError("Unexpected QuantumPioneer input hash")
    feature_receipt = verified_artifacts(args.features)
    frozen_file = args.frozen / "prediction.json"
    frozen = read(frozen_file)
    frozen_rev = frozen["checkout_commit"]
    if frozen_rev != git("rev-parse", "HEAD"):
        git("merge-base", "--is-ancestor", frozen_rev, "HEAD")
        changed = set(git("diff", "--name-only", frozen_rev, "HEAD").splitlines())
        operational_paths = {f"Experiments/74_context_dependence/{name}" for name in (
            "prepare_shared_matrices.py", "run_quantumpioneer.py", "shared_fit_smoke.py",
        )}
        if not args.shared_matrices or not changed <= operational_paths:
            raise ValueError("Changes after the training-only freeze exceed matrix loading")
    if (frozen["phase"] != "training_only_prediction_frozen" or
        frozen["protocol_sha256"] != sha256(Path(__file__).with_name("PROTOCOL.md")) or
        frozen["heldout_outcomes_scored"] or frozen["source_sha256"] != sha256(args.source) or
        frozen["feature_receipt_sha256"] != sha256(args.features / "COMPLETE.json")):
        raise ValueError("Training-only prediction was not frozen against these inputs")
    for name, expected in frozen["artifact_sha256"].items():
        if sha256(args.frozen / name) != expected:
            raise ValueError(f"Changed frozen artifact {name}")

    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "run.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = args.output / "status.json"
    if status.exists() and read(status)["phase"] in ("heldout_scoring", "complete"):
        raise ValueError("This holdout has already been opened")
    with np.load(args.frozen / "cohort.npz", allow_pickle=False) as stored:
        parts = tuple(stored[f"{split}_indices"] for split in ("train", "validation", "test"))
        groups, ids = stored["test_groups"], stored["test_ids"]
        train_y, val_y = stored["train_targets"], stored["validation_targets"]
    shared_paths = None
    if args.shared_matrices:
        shared = verified_artifacts(args.shared_matrices)
        if (shared["phase"] != "shared_matrices_verified" or
            shared["frozen_prediction_sha256"] != sha256(frozen_file) or
            shared["feature_receipt_sha256"] != sha256(args.features / "COMPLETE.json")):
            raise ValueError("Shared matrices were not decoded from this frozen input")
        shared_paths = {}
        for arm in ARMS:
            paths = tuple(args.shared_matrices / f"{arm}__{split}.npy"
                          for split in ("train", "validation", "test"))
            for path, rows in zip(paths, parts, strict=True):
                matrix = np.load(path, mmap_mode="r", allow_pickle=False)
                if (matrix.shape != (len(rows), feature_receipt["widths"][arm]) or
                    matrix.dtype != np.float32 or not matrix.flags.c_contiguous):
                    raise ValueError(f"Unexpected shared matrix shape, type or layout: {path}")
            shared_paths[arm] = paths
    manifest = args.output / "manifest.json"
    plan = {
        "source_sha256": frozen["source_sha256"], "frozen_prediction_sha256": sha256(frozen_file),
        "features_sha256": sha256(args.features / "COMPLETE.json"),
        "development_sha256": sha256(args.development / "COMPLETE.json"),
        "synthetic_sha256": sha256(args.synthetic / "COMPLETE.json"),
        "runner_commit": git("rev-parse", "HEAD"), "runner_sha256": sha256(Path(__file__)),
        "arms": ARMS, "targets": TARGET_NAMES, "seeds": SEEDS, "fits": 30,
        "fit_options": e1.FIT_OPTIONS, "max_epochs": e1.MAX_EPOCHS,
        "patience": e1.PATIENCE, "tolerance": e1.TOLERANCE,
        "environment": features.environment(), "splits": frozen["split_summary"],
        "checkpoint_selection": "minimum standardized validation MSE",
        "shared_matrix_receipt_sha256": (sha256(args.shared_matrices / "COMPLETE.json")
                                         if args.shared_matrices else None),
    }
    if manifest.exists():
        if read(manifest) != json.loads(json.dumps(plan)):
            raise ValueError("A different primary run already uses this directory")
    else:
        write(manifest, plan)
    plan_hash = sha256(manifest)
    write(status, {"phase": "blind_fitting", "expected_fits": 30, "plan_sha256": plan_hash})
    packed = {arm: np.load(args.features / f"{arm}.npy", mmap_mode="r", allow_pickle=False) for arm in ARMS}
    if set(feature_receipt["widths"]) != set(packed):
        raise ValueError("Unexpected feature arms")
    fits = args.output / "cells"
    fits.mkdir(exist_ok=True)
    receipts = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(fit_cell)(None if shared_paths else packed[arm], parts, train_y[:, i],
                          val_y[:, i], arm, target, seed, fits, plan_hash,
                          shared_paths[arm] if shared_paths else None)
        for (i, target), arm, seed in product(enumerate(TARGET_NAMES), ARMS, SEEDS)
    )
    predictions = blind_predictions(fits, receipts, parts[2])
    write(status, {"phase": "heldout_scoring", "scored_utc": utc(), "fits": 30,
                   "frozen_prediction_sha256": sha256(frozen_file)})
    truth = observed_test(args.source, parts[2], ids)
    score(args.output, predictions, truth, groups, train_y, frozen)
    write(status, {"phase": "complete", "fits": 30, "completed_utc": utc(),
                   "outputs_sha256": {name: sha256(args.output / name)
                                       for name in ("manifest.json", "scores.json", "group_bootstrap.npz")}})


if __name__ == "__main__":
    main()
