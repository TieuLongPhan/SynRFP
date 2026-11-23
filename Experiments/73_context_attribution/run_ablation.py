"""Fit the frozen comparison of delta width, rehashing and pooled context."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from features import ARMS, DATASETS, PLACES, WIDTHS, environment, file_sha256
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

EXPECTED_GROUPS = 15
CONTRASTS = (
    ("d_pooled", "d_rehash"),
    ("d_pooled", "d2048"),
    ("d2048", "d1024"),
    ("d_pooled", "d1024"),
)
FIT_OPTIONS = {
    "hidden_layer_sizes": (512,),
    "activation": "relu",
    "solver": "adam",
    "learning_rate_init": 0.001,
    "alpha": 0.0001,
    "batch_size": 256,
    "max_iter": 200,
    "early_stopping": False,
}
MAX_EPOCHS, PATIENCE, TOLERANCE, MODEL_SEED = 200, 15, 1e-4, 42
BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 10_000, 2026
CELL_KEYS = ["dataset", "split", "split_type", "placement", "conditions"]


@dataclass(frozen=True)
class Partition:
    name: str
    kind: str
    fit: np.ndarray
    validation: np.ndarray
    test: np.ndarray


def require(predicate, message):
    if not predicate:
        raise ValueError(message)


def load_features(directory):
    provenance = json.loads((directory / "metadata.json").read_text())
    require(provenance.get("schema") == 1, "Unsupported feature metadata schema")
    for key in ("feature_sha256", "input_sha256"):
        mapping = provenance.get(key)
        require(
            isinstance(mapping, dict)
            and mapping
            and all(
                isinstance(h, str) and re.fullmatch("[0-9a-f]{64}", h)
                for h in mapping.values()
            ),
            f"Missing or invalid {key} in feature metadata",
        )
    require(
        set(provenance["feature_sha256"]) == {f"{name}.npz" for name in DATASETS},
        "Feature receipt does not cover exactly the benchmark datasets",
    )
    require(
        set(provenance["input_sha256"]) == {source for source, _ in DATASETS.values()},
        "Input receipt does not cover exactly the benchmark source tables",
    )
    coverage = provenance.get("coverage", {})
    require(
        set(coverage) == set(DATASETS), "Missing dataset coverage in feature metadata"
    )
    require(
        provenance.get("source_revision")
        and provenance.get("versions")
        and provenance.get("feature_code_sha256")
        == file_sha256(Path(__file__).with_name("features.py")),
        "Feature source, environment or code does not match the experiment",
    )
    datasets = {}
    for name, (_, expected_n) in DATASETS.items():
        filename = f"{name}.npz"
        path = directory / filename
        require(
            file_sha256(path) == provenance["feature_sha256"][filename],
            f"Checksum mismatch: {path}",
        )
        with np.load(path, allow_pickle=False) as arrays:
            data = dict(arrays)
        y, ids, groups, condition_bits = (
            data[key] for key in ("y", "sample_id", "group", "H")
        )
        condition_names = data["condition_names"]
        n = y.size
        require(
            y.shape == (expected_n,)
            and np.isfinite(y).all()
            and coverage[name].get("source_rows") == expected_n,
            f"{name} does not cover the expected {expected_n} yields",
        )
        for axis, keys in (("IDs", ids), ("groups", groups)):
            require(
                keys.shape == y.shape
                and keys.dtype.kind == "U"
                and np.all(np.char.str_len(keys) > 0),
                f"{name} has invalid {axis}",
            )
        require(len(np.unique(ids)) == n, f"{name} has duplicate sample IDs")
        require(
            len(np.unique(groups)) == coverage[name].get("groups") == EXPECTED_GROUPS,
            f"{name} does not cover the expected {EXPECTED_GROUPS} substrate groups",
        )
        require(
            condition_names.ndim == 1
            and condition_names.dtype.kind == "U"
            and len(np.unique(condition_names)) == len(condition_names)
            and condition_bits.shape == (n, len(condition_names))
            and condition_bits.dtype == np.uint8
            and np.all(condition_bits <= 1),
            f"{name} has invalid condition features",
        )
        require(
            len(condition_names) == coverage[name].get("condition_components"),
            f"{name} condition vocabulary count disagrees with metadata",
        )
        for place in PLACES:
            for arm in ARMS:
                x = data[f"{place}_{arm}"]
                require(
                    x.shape == (n, WIDTHS[arm])
                    and x.dtype == np.uint8
                    and np.all(x <= 1),
                    f"{name} has invalid {place} {arm} features",
                )
            narrow, wide = data[f"{place}_d1024"], data[f"{place}_d2048"]
            require(
                np.array_equal(narrow, wide[:, :1024] ^ wide[:, 1024:])
                and np.array_equal(narrow, data[f"{place}_d_pooled"][:, :1024])
                and np.array_equal(narrow, data[f"{place}_d_rehash"][:, :1024]),
                f"{name} {place} violates equal-support parity nesting",
            )
        for arm in ("d1024", "d2048", "d_rehash"):
            require(
                np.array_equal(data[f"A_{arm}"], data[f"S_{arm}"]),
                f"{name} {arm}: shared agents did not cancel",
            )
        datasets[name] = data
    return datasets, provenance


def partitions(groups):
    indices = np.arange(len(groups))
    require(len(np.unique(groups)) >= 5, "At least five substrate groups are needed")
    result = []
    for fold in range(10):
        pool, test = train_test_split(indices, test_size=0.2, random_state=fold)
        fit, validation = train_test_split(
            pool, test_size=0.1, random_state=1337 + fold
        )
        result.append(Partition(f"random{fold}", "random", fit, validation, test))

    for fold, (pool, test) in enumerate(
        GroupKFold(n_splits=5).split(indices, groups=groups)
    ):
        validation_splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.1, random_state=1337 + fold
        )
        train_index, val_index = next(
            validation_splitter.split(pool, groups=groups[pool])
        )
        fit, validation = pool[train_index], pool[val_index]
        group_sets = [set(groups[idx]) for idx in (fit, validation, test)]
        require(
            sum(map(len, group_sets)) == len(set.union(*group_sets)),
            "Grouped split shares groups",
        )
        result.append(Partition(f"group{fold}", "grouped", fit, validation, test))

    for part in result:
        combined = np.concatenate((part.fit, part.validation, part.test))
        require(
            all(len(idx) for idx in (part.fit, part.validation, part.test)),
            f"{part.name} has an empty partition",
        )
        require(
            np.array_equal(np.sort(combined), indices),
            f"{part.name} is not a partition of all observations",
        )
    grouped_test = np.concatenate(
        [part.test for part in result if part.kind == "grouped"]
    )
    require(
        np.array_equal(np.sort(grouped_test), indices),
        "Grouped out-of-fold tests do not cover every row once",
    )
    return result


def fit_cell(name, data, part, place, conditions, arm):
    x = data[f"{place}_{arm}"]
    y = data["y"]
    keep_condition = data["H"][part.fit].any(axis=0)
    splits = (part.fit, part.validation, part.test)
    arrays = []
    for rows in splits:
        values = x[rows].astype(np.float32)
        if conditions:
            values = np.column_stack((values, data["H"][rows][:, keep_condition]))
        arrays.append(values)
    x_fit, x_val, x_test = arrays
    center = float(y[part.fit].mean())
    scale = float(y[part.fit].std()) or 1.0
    y_fit, y_val = ((y[rows] - center) / scale for rows in splits[:2])
    model = MLPRegressor(**FIT_OPTIONS, random_state=np.random.RandomState(MODEL_SEED))
    best_loss, patience_loss, stale, best_epoch = np.inf, np.inf, 0, 0

    with threadpool_limits(limits=1):
        for epoch in range(1, MAX_EPOCHS + 1):
            model.partial_fit(x_fit, y_fit)
            loss = float(np.mean((model.predict(x_val) - y_val) ** 2))
            require(
                np.isfinite(loss),
                f"Nonfinite validation loss: {name} {part.name} {place} {conditions} {arm}",
            )
            if loss < best_loss:
                best_loss, best_epoch = loss, epoch
                best_parameters = (
                    [v.copy() for v in model.coefs_],
                    [v.copy() for v in model.intercepts_],
                )
            if patience_loss - loss >= TOLERANCE:
                patience_loss, stale = loss, 0
            else:
                stale += 1
            if stale >= PATIENCE:
                break
        model.coefs_, model.intercepts_ = best_parameters
        predicted = np.asarray(model.predict(x_test), dtype=np.float64) * scale + center

    actual = y[part.test]
    require(np.isfinite(predicted).all(), "Nonfinite test predictions")
    sse = np.sum((predicted - actual) ** 2)
    total = np.sum((actual - actual.mean()) ** 2)
    metrics = {
        "dataset": name,
        "split": part.name,
        "split_type": part.kind,
        "placement": place,
        "conditions": conditions,
        "arm": arm,
        "n_fit": len(part.fit),
        "n_validation": len(part.validation),
        "n_test": len(part.test),
        "condition_columns": int(keep_condition.sum()) if conditions else 0,
        "train_y_mean": center,
        "train_y_sd": scale,
        "best_epoch": best_epoch,
        "epochs": epoch,
        "validation_standardized_mse": best_loss,
        "r2": 1 - sse / total if total else np.nan,
        "rmse": np.sqrt(sse / len(actual)),
    }
    return metrics, part.test, predicted


def dump_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_splits(path, datasets, plan):
    with gzip.open(path, "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("dataset", "split", "split_type", "sample_id", "group", "partition")
        )
        for name, data in datasets.items():
            ids, groups = data["sample_id"], data["group"]
            for part in plan[name]:
                for kind, rows in (
                    ("fit", part.fit),
                    ("validation", part.validation),
                    ("test", part.test),
                ):
                    writer.writerows(
                        (name, part.name, part.kind, ids[i], groups[i], kind)
                        for i in rows
                    )


def summarize_cells(results):
    paired = results.pivot(index=CELL_KEYS, columns="arm", values=["r2", "rmse"])
    estimates = []
    for left, right in CONTRASTS:
        effect = {
            "left": left,
            "right": right,
            "delta_r2": paired["r2"][left] - paired["r2"][right],
            "rmse_reduction": paired["rmse"][right] - paired["rmse"][left],
        }
        estimates.append(pd.DataFrame(effect, index=paired.index).reset_index())
    return pd.concat(estimates, ignore_index=True)


def summarize_grouped(datasets, grouped_predictions):
    summary = []
    for name, data in datasets.items():
        y = data["y"]
        groups, membership = np.unique(data["group"], return_inverse=True)
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        samples = rng.integers(len(groups), size=(BOOTSTRAP_DRAWS, len(groups)))
        count = np.bincount(membership)
        y_sum = np.bincount(membership, weights=y)
        y_square_sum = np.bincount(membership, weights=y**2)
        n = count[samples].sum(axis=1)
        tss = y_square_sum[samples].sum(axis=1) - y_sum[samples].sum(axis=1) ** 2 / n
        total = np.sum((y - y.mean()) ** 2)
        require(total > 0, f"{name} yield is constant")
        viable = tss > 0
        require(viable.any(), f"{name} has no finite group bootstrap contrasts")
        for place in PLACES:
            for conditions in (False, True):
                errors = {}
                for arm in ARMS:
                    predicted = grouped_predictions[name, place, conditions, arm]
                    require(
                        np.isfinite(predicted).all(),
                        f"{name} is missing out-of-fold test predictions",
                    )
                    errors[arm] = np.bincount(membership, weights=(predicted - y) ** 2)
                for left, right in CONTRASTS:
                    delta = errors[right] - errors[left]
                    boot = delta[samples[viable]].sum(axis=1) / tss[viable]
                    lo, hi = np.quantile(boot, [0.025, 0.975])
                    summary.append(
                        {
                            "dataset": name,
                            "placement": place,
                            "conditions": conditions,
                            "left": left,
                            "right": right,
                            "n_groups": len(groups),
                            "n_test": len(y),
                            "left_r2": 1 - errors[left].sum() / total,
                            "right_r2": 1 - errors[right].sum() / total,
                            "delta_r2": delta.sum() / total,
                            "delta_r2_ci95_low": lo,
                            "delta_r2_ci95_high": hi,
                            "bootstrap_valid_draws": int(viable.sum()),
                            "primary": name == "bh"
                            and place == "S"
                            and conditions
                            and (left, right) == ("d_pooled", "d_rehash"),
                        }
                    )
    return pd.DataFrame(summary)


def run(args, datasets, provenance, plan, count):
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    dump_json(out / "status.json", {"phase": "running", "expected_cells": count})
    try:
        run_fits(args, datasets, provenance, plan, count)
    except BaseException as error:
        dump_json(
            out / "status.json",
            {
                "phase": "failed",
                "expected_cells": count,
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def run_fits(args, datasets, provenance, plan, count):
    out = args.output_dir
    dump_json(
        out / "manifest.json",
        {
            "feature_receipt": provenance,
            "feature_receipt_sha256": file_sha256(args.features_dir / "metadata.json"),
            "runner_sha256": file_sha256(Path(__file__)),
            "environment": environment(),
            "learner": {
                **FIT_OPTIONS,
                "random_seed": MODEL_SEED,
                "epoch_limit": MAX_EPOCHS,
                "patience": PATIENCE,
                "tolerance": TOLERANCE,
                "checkpoint": "lowest standardized validation MSE",
            },
            "splits": {
                "random_outer_seeds": list(range(10)),
                "random_outer_test_fraction": 0.2,
                "grouped_outer_folds": 5,
                "validation_fraction": 0.1,
                "validation_seed_base": 1337,
            },
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
            "cpu_jobs": args.jobs,
        },
    )
    write_splits(out / "splits.csv.gz", datasets, plan)
    jobs = (
        delayed(fit_cell)(name, data, part, place, conditions, arm)
        for name, data in datasets.items()
        for part in plan[name]
        for place in PLACES
        for conditions in (False, True)
        for arm in ARMS
    )
    results = []
    grouped_predictions = {
        (name, place, conditions, arm): np.full(len(data["y"]), np.nan)
        for name, data in datasets.items()
        for place in PLACES
        for conditions in (False, True)
        for arm in ARMS
    }
    completed = Parallel(n_jobs=args.jobs, return_as="generator", verbose=5)(jobs)
    with gzip.open(out / "predictions.csv.gz", "wt", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                *CELL_KEYS,
                "arm",
                "sample_id",
                "group",
                "observed_yield",
                "predicted_yield",
            )
        )
        for metrics, indices, predictions in completed:
            results.append(metrics)
            name = metrics["dataset"]
            data = datasets[name]
            if metrics["split_type"] == "grouped":
                key = (
                    name,
                    metrics["placement"],
                    metrics["conditions"],
                    metrics["arm"],
                )
                require(
                    np.isnan(grouped_predictions[key][indices]).all(),
                    "Duplicate out-of-fold predictions",
                )
                grouped_predictions[key][indices] = predictions
            cell = tuple(metrics[k] for k in (*CELL_KEYS, "arm"))
            writer.writerows(
                (*cell, data["sample_id"][i], data["group"][i], data["y"][i], pred)
                for i, pred in zip(indices, predictions, strict=True)
            )

    scores = pd.DataFrame(results)
    require(
        len(scores) == count and not scores.duplicated([*CELL_KEYS, "arm"]).any(),
        "Incomplete fit grid",
    )
    scores.to_csv(out / "results.csv", index=False)
    summarize_cells(scores).to_csv(out / "paired.csv", index=False)
    grouped = summarize_grouped(datasets, grouped_predictions)
    grouped.to_csv(out / "grouped_oof.csv", index=False)
    dump_json(
        out / "status.json",
        {
            "phase": "complete",
            "cells": len(scores),
            "outputs_sha256": {
                path.name: file_sha256(path)
                for path in sorted(out.iterdir())
                if path.name != "status.json" and path.is_file()
            },
        },
    )
    print(grouped.loc[grouped.primary].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--jobs", type=int, default=4, help="Concurrent single-threaded fits"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify features and partitions without fitting or writing",
    )
    args = parser.parse_args()
    require(args.jobs >= 1, "jobs must be positive")
    require(not args.output_dir.exists(), "Choose a new output directory")
    datasets, provenance = load_features(args.features_dir)
    plan = {name: partitions(data["group"]) for name, data in datasets.items()}
    count = sum(map(len, plan.values())) * len(PLACES) * 2 * len(ARMS)
    for name, data in datasets.items():
        print(
            f"{name}: {len(data['y'])} rows, {len(np.unique(data['group']))} groups, {len(plan[name])} splits"
        )
    print(f"{count} fits planned")
    if args.check_only:
        return
    run(args, datasets, provenance, plan, count)


if __name__ == "__main__":
    main()
