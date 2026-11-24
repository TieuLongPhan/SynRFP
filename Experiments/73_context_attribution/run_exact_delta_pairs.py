"""Test whether pooled context predicts yield gaps within exact-delta classes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import features
from run_ablation import (
    FIT_OPTIONS,
    MAX_EPOCHS,
    MODEL_SEED,
    PATIENCE,
    TOLERANCE,
    Partition,
    dump_json,
    fit_cell,
    require,
    write_splits,
)


SOURCE_COMMIT = "e2712ad3d3c63b6df29831d516e68bfc5fd09db4"
RELEASE_COMMIT = "654968ec04990a238c115095641ddefbf9f7957a"
CONTROLS = ("d_rehash", "d2048")
ARMS = ("d_pooled", *CONTROLS)
VALIDATION_SEED = 1337
GAP_METRICS = ("gap_mse", "gap_mse_zero", "gap_mse_reduction_from_zero")


@dataclass(frozen=True)
class Holdout:
    delta_class: str
    partition: Partition
    matched: np.ndarray


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def read_pinned_inputs(data_dir):
    experiment = Path(__file__).resolve().parent
    repo = experiment.parents[1]
    release = experiment / "publication_metadata" / "e1-e2712ad3"
    receipt_path = (release / "COMPLETE.json").relative_to(repo).as_posix()
    completion = json.loads(git(repo, "show", f"{RELEASE_COMMIT}:{receipt_path}"))
    manifest_path = release / "manifest.json"
    require(
        completion["source_revision"] == SOURCE_COMMIT
        and completion["phase"] == "complete"
        and completion["cells"] == 720
        and features.file_sha256(manifest_path) == completion["outputs_sha256"]["manifest.json"],
        "Expected the pinned E1 release manifest",
    )
    manifest = json.loads(manifest_path.read_text())
    receipt = manifest["feature_receipt"]
    for filename, expected in (
        ("features.py", receipt["feature_code_sha256"]),
        ("run_ablation.py", manifest["runner_sha256"]),
    ):
        require(features.file_sha256(experiment / filename) == expected, f"{filename} differs from E1")
    require(
        Path(features.synrfp.__file__).resolve().parent == repo / "synrfp"
        and not git(repo, "diff", SOURCE_COMMIT, "--", "synrfp")
        and not git(repo, "status", "--porcelain", "--", "synrfp"),
        "Install this worktree's unmodified E1 SynRFP encoder",
    )
    versions = features.environment()
    require(
        all(versions[key] == value for key, value in manifest["environment"].items() if key != "python"),
        "Package versions must match the E1 environment",
    )
    filename, _ = features.DATASETS["bh"]
    source = data_dir / filename
    digest = features.file_sha256(source)
    require(digest == receipt["input_sha256"][filename], "Buchwald-Hartwig table does not match E1")
    records = features.read_records("bh", source)
    provenance = {
        "source_data": {"filename": filename, "sha256": digest, "n_rows": len(records)},
        "e1_release_commit": RELEASE_COMMIT,
        "e1_source_commit": SOURCE_COMMIT,
        "e1_manifest_sha256": features.file_sha256(manifest_path),
        "worktree_commit": git(repo, "rev-parse", "HEAD"),
        "runner_sha256": features.file_sha256(Path(__file__)),
        "runner_dirty": bool(git(repo, "status", "--porcelain", "--",
                                 Path(__file__).resolve().relative_to(repo).as_posix())),
        "versions": versions,
    }
    return records, provenance


def count_key(counts):
    return tuple(sorted((token, value) for token, value in counts.items() if value))


def make_plan(records):
    class_by_group = {}
    shared_by_group = {}
    recipe_rows = defaultdict(dict)
    for index, record in enumerate(records):
        left = features.side_counts(record.substrates)
        right = features.side_counts(record.products)
        difference = right.copy()
        difference.subtract(left)
        group = record.group
        delta, shared = count_key(difference), count_key(left & right)
        require(class_by_group.setdefault(group, delta) == delta, "A precursor form spans delta classes")
        require(shared_by_group.setdefault(group, shared) == shared, "Shared counts vary within a precursor form")
        recipe = record.conditions
        require(recipe not in recipe_rows[group], "Duplicate assay for a precursor form and recipe")
        recipe_rows[group][recipe] = index

    groups_by_class = defaultdict(list)
    for group in sorted(class_by_group):
        groups_by_class[class_by_group[group]].append(group)
    all_groups = sorted(class_by_group)
    row_groups = np.asarray([record.group for record in records])
    require(len(all_groups) == 15 and len(groups_by_class) == 9, "Unexpected exact-delta coverage")

    holdouts = []
    informative = [groups for _, groups in sorted(groups_by_class.items()) if len(groups) > 1]
    require(len(informative) == 3 and all(len(groups) == 3 for groups in informative),
            "Expected three informative delta classes with three precursor forms each")
    for class_index, groups in enumerate(informative, start=1):
        delta_class = f"delta{class_index:02d}"
        validation_options = [group for group in all_groups if group not in groups]
        rng = np.random.default_rng(VALIDATION_SEED + class_index - 1)
        validation_group = rng.choice(validation_options)
        validation = row_groups == validation_group
        require(len({shared_by_group[group] for group in groups}) == 3, "Expected distinct shared counts")
        for pair_index, (first, second) in enumerate(combinations(groups, 2), start=1):
            recipes = sorted(recipe_rows[first].keys() & recipe_rows[second].keys())
            require(bool(recipes), "A held-out precursor pair has no common recipe")
            matched = np.array([(recipe_rows[first][key], recipe_rows[second][key]) for key in recipes])
            test = np.isin(row_groups, (first, second))
            partition = Partition(
                f"{delta_class}_pair{pair_index:02d}",
                "paired_form_holdout",
                np.flatnonzero(~(test | validation)),
                np.flatnonzero(validation),
                np.flatnonzero(test),
            )
            require(len({row_groups[i] for i in partition.fit}) == 12 and validation.any()
                    and np.all(test[matched]), "Invalid fit, validation or matched test membership")
            holdouts.append(Holdout(delta_class, partition, matched))

    require(sum(len(holdout.matched) for holdout in holdouts) == 2368,
            "Expected 2,368 recipe-matched gaps across nine joint holdouts")
    return holdouts


def score_fit(holdout, data, arm, scored):
    metrics, rows, prediction = scored
    require(np.array_equal(rows, holdout.partition.test), "Prediction rows differ from the joint holdout")
    require(prediction.shape == rows.shape and np.isfinite(prediction).all(), "Invalid predictions")
    first, second = holdout.matched.T
    require(np.array_equal(data["H"][first], data["H"][second]), "Matched recipes differ in indicator features")
    locations = np.searchsorted(rows, holdout.matched)
    observed_gap = data["y"][first] - data["y"][second]
    predicted_gap = prediction[locations[:, 0]] - prediction[locations[:, 1]]
    encoded = data[f"S_{arm}"]
    collisions = np.all(encoded[first] == encoded[second], axis=1)
    if arm in CONTROLS:
        require(collisions.all() and np.allclose(predicted_gap, 0, atol=1e-5),
                f"{arm} should predict the same yield for each matched pair")

    gap_mse = np.mean((observed_gap - predicted_gap) ** 2)
    baseline_mse = np.mean(observed_gap**2)
    pair_score = {
        "delta_class": holdout.delta_class,
        "split": holdout.partition.name,
        "arm": arm,
        "matched_recipes": len(first),
        "identical_fingerprints": int(collisions.sum()),
        "gap_mse": gap_mse,
        "gap_mse_zero": baseline_mse,
        "gap_mse_reduction_from_zero": baseline_mse - gap_mse,
    }
    predictions = pd.DataFrame({
        "split": holdout.partition.name,
        "arm": arm,
        "sample_id": data["sample_id"][rows],
        "group": data["group"][rows],
        "observed_yield": data["y"][rows],
        "predicted_yield": prediction,
    })
    return metrics, pair_score, predictions


def write_outputs(directory, data, holdouts, jobs):
    plan = {"bh": [holdout.partition for holdout in holdouts]}
    write_splits(directory / "splits.csv.gz", {"bh": data}, plan)
    tasks = [(holdout, arm) for holdout in holdouts for arm in ARMS]
    fitted = Parallel(n_jobs=jobs, return_as="generator", verbose=5)(
        delayed(fit_cell)("bh", data, holdout.partition, "S", True, arm)
        for holdout, arm in tasks
    )
    scores = [score_fit(holdout, data, arm, scored)
              for (holdout, arm), scored in zip(tasks, fitted, strict=True)]
    fit_scores, pair_rows, predictions = zip(*scores)
    pd.DataFrame(fit_scores).to_csv(directory / "fit_scores.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        directory / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0},
    )
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(directory / "pair_gap_scores.csv", index=False)
    classes = pairs.groupby(["delta_class", "arm"], as_index=False)[list(GAP_METRICS)].mean()
    macro = classes.groupby("arm", as_index=False)[list(GAP_METRICS)].mean().assign(delta_class="macro")
    summaries = pd.concat([classes, macro], ignore_index=True)
    summaries.to_csv(directory / "class_gap_scores.csv", index=False)
    print(summaries.to_string(index=False))


def run(directory, data, holdouts, provenance, jobs):
    directory.mkdir(parents=True, exist_ok=False)
    expected = len(holdouts) * len(ARMS)
    dump_json(directory / "status.json", {"phase": "running", "expected_fits": expected})
    try:
        dump_json(directory / "manifest.json", {
            **provenance,
            "learner": {**FIT_OPTIONS, "epoch_limit": MAX_EPOCHS, "patience": PATIENCE,
                        "tolerance": TOLERANCE, "seed": MODEL_SEED},
            "validation_seed_base": VALIDATION_SEED,
            "place": "S",
            "condition_indicators": True,
            "design": "co-hold out each exact-delta precursor pair; keep the third form in fitting",
            "primary_metric": "equal-class mean reduction in squared matched yield gap vs zero",
            "cpu_jobs": jobs,
            "holdouts": [
                {"delta_class": item.delta_class, "split": item.partition.name,
                 "n_fit": len(item.partition.fit), "n_validation": len(item.partition.validation),
                 "n_test": len(item.partition.test), "matched_recipes": len(item.matched)}
                for item in holdouts
            ],
        })
        write_outputs(directory, data, holdouts, jobs)
        dump_json(directory / "status.json", {
            "phase": "complete",
            "fits": expected,
            "outputs_sha256": {
                name: features.file_sha256(directory / name)
                for name in ("manifest.json", "fit_scores.csv", "pair_gap_scores.csv", "class_gap_scores.csv")
            },
            "local_artifact_sha256": {
                name: features.file_sha256(directory / name)
                for name in ("splits.csv.gz", "predictions.csv.gz")
            },
        })
    except BaseException as error:
        dump_json(directory / "status.json", {
            "phase": "failed", "expected_fits": expected,
            "error_type": type(error).__name__, "error": str(error),
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or (args.output_dir is None and not args.check_only):
        parser.error("Choose positive --jobs and supply --output-dir for a fit run")

    records, provenance = read_pinned_inputs(args.data_dir)
    holdouts = make_plan(records)
    print(f"{len(holdouts)} joint holdouts, {sum(len(item.matched) for item in holdouts):,} "
          f"matched gaps, {len(holdouts) * len(ARMS)} planned fits")
    if not args.check_only:
        run(args.output_dir, features.feature_arrays(records), holdouts, provenance, args.jobs)


if __name__ == "__main__":
    main()
