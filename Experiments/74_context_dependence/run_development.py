"""Exercise the regression diagnostic and paired analysis on the development cohorts."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
from itertools import product
import json
from pathlib import Path
import sys
import time

from joblib import Parallel, delayed
import numpy as np
from rdkit import RDLogger

from audit_inputs import PINS, TARGETS, sha256
from audit_quantumpioneer_graphs import connected_groups, digest, graph_row
from paired_analysis import means_by_group, paired_bootstrap, regression_diagnostic, signed_key
from smoke_regression import ARMS, encode, partition, sides

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "73_context_attribution"))
import features
import run_ablation as e1

SEEDS = (17, 29, 43)
RDLogger.DisableLog("rdApp.*")


def read_source(name, source):
    if sha256(source) != PINS[name][1]:
        raise ValueError("Unexpected development source checksum")
    with gzip.open(source, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != PINS[name][2]:
        raise ValueError("Unexpected development source size")
    return rows


def make_development_cohort(name, source, output):
    records = read_source(name, source)
    ledger = []
    duplicates = defaultdict(list)
    for i, row in enumerate(records):
        audit = graph_row((i, row["r_id"], f"row-{i:08d}", "development", row["aam"]))
        audit["exclusion"] = audit["graph_refusal"]
        if not audit["exclusion"]:
            duplicates[audit["canonical_direction"]].append(i)
        ledger.append(audit)
    for indexes in duplicates.values():
        if len(indexes) < 2:
            continue
        outcomes = {tuple(records[i][target] for target in TARGETS[name]) for i in indexes}
        if len(outcomes) > 1:
            for i in indexes:
                ledger[i]["exclusion"] = "conflicting_development_duplicate"
        else:
            for i in indexes[1:]:
                ledger[i]["exclusion"] = "duplicate_canonical_direction"
    connected_groups(ledger)

    arms = {arm: [] for arm in ARMS}
    keys, labels, groups, ids = [], [], [], []
    for audit, source_row in zip(ledger, records, strict=True):
        if audit["exclusion"]:
            continue
        try:
            left, right = sides(source_row["aam"])
            bit_vectors = encode(left, right)
            key = signed_key(features.side_counts(left), features.side_counts(right))
        except (ValueError, RuntimeError, TypeError) as error:
            audit["exclusion"] = f"feature_refusal:{type(error).__name__}"
            continue
        for arm in ARMS:
            arms[arm].append(bit_vectors[arm])
        keys.append(key)
        labels.append([float(source_row[target]) for target in TARGETS[name]])
        groups.append(audit["connected_group"])
        ids.append(audit["row_id"])

    group_ids = np.asarray(groups)
    row_ids = np.asarray(ids)
    split = partition(group_ids)
    membership = {row_ids[i]: label for label, indices in
                  (("fit", split.fit), ("validation", split.validation), ("heldout", split.test))
                  for i in indices}
    for row in ledger:
        row["cohort_partition"] = membership.get(row["row_id"], "")
    with gzip.open(output / "row_ledger.csv.gz", "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)

    manifest = {
        "source": name, "source_hash": PINS[name][1], "targets": TARGETS[name],
        "source_rows": len(records), "cohort_rows": len(ids),
        "exclusions": dict(Counter(r["exclusion"] for r in ledger if r["exclusion"])),
        "partition_counts": dict(Counter(membership.values())),
        "split_sha256": digest(sorted(membership.items())),
        "row_ledger_sha256": sha256(output / "row_ledger.csv.gz"),
    }
    (output / "cohort_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(name, manifest["partition_counts"], "exclusions", manifest["exclusions"], flush=True)
    return {f"S_{arm}": np.stack(bits) for arm, bits in arms.items()}, keys, np.asarray(labels), group_ids, row_ids, split


def run_fit(name, arm, target, target_index, seed, x, y, groups, ids, split, output):
    e1.MODEL_SEED = seed
    data = {f"S_{arm}": x, "y": y[:, target_index], "H": np.empty((len(y), 0))}
    start = time.monotonic()
    metrics, rows, prediction = e1.fit_cell(name, data, split, "S", False, arm)
    np.testing.assert_array_equal(rows, split.test)
    observed = y[rows, target_index]
    np.testing.assert_allclose(metrics["rmse"], np.sqrt(np.mean((prediction - observed) ** 2)))
    path = output / f"{target}__{arm}__s{seed}.npz"
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, row_id=ids[rows], group=groups[rows], observed=observed,
                        prediction=prediction, fit_seed=seed)
    temporary.replace(path)
    metrics.update(target=target, seed=seed, arm=arm, fit_seconds=time.monotonic() - start,
                   predictions_file=path.name, predictions_sha256=sha256(path))
    receipt = path.with_suffix(".json")
    temporary = receipt.with_suffix(".tmp")
    temporary.write_text(json.dumps(metrics, indent=2) + "\n")
    temporary.replace(receipt)
    return metrics, prediction


def score_development(name, output, y, groups, ids, split, diagnostic_summary, results):
    fit_scores = [metrics for metrics, _ in results]
    predictions = {(metrics["target"], metrics["arm"], metrics["seed"]): pred
                   for metrics, pred in results}
    assert len(predictions) == len(ARMS) * len(SEEDS) * len(TARGETS[name])
    test = split.test
    variance = y[split.fit].var(axis=0)
    gains = []
    for i, target in enumerate(TARGETS[name]):
        losses = {}
        for arm in ("d2048", "d_pooled"):
            squared = [(predictions[target, arm, seed] - y[test, i])**2 for seed in SEEDS]
            losses[arm] = np.mean(squared, axis=0) / variance[i]
        gains.append(losses["d2048"] - losses["d_pooled"])
    group_gains = means_by_group(np.column_stack(gains), groups[test])
    draws = paired_bootstrap(group_gains)
    order = diagnostic_summary["higher_gain_target_index"]
    interaction = None
    if order is not None:
        differences = draws[:, order] - draws[:, 1-order]
        interaction = {"higher_ranked": TARGETS[name][order],
                       "difference": float(group_gains[:, order].mean() - group_gains[:, 1-order].mean()),
                       "interval": np.quantile(differences, (0.025, 0.975)).tolist()}
    np.savez_compressed(output / "bootstrap.npz", draws=draws, group_gains=group_gains,
                        group_ids=np.unique(groups[test]))
    summary = {
        "phase": "complete", "scope": "development_only", "source": name,
        "fits": len(fit_scores), "seeds": SEEDS, "targets": TARGETS[name],
        "normalized_gains": group_gains.mean(axis=0).tolist(),
        "intervals": np.quantile(draws, (0.025, 0.975), axis=0).T.tolist(),
        "interaction": interaction, "bootstrap_seed": 2026, "bootstrap_draws": len(draws),
        "strictly_heldout_groups": len(group_gains),
        "runner_sha256": sha256(Path(__file__)),
        "test_rows_sha256": digest(ids[test].tolist()),
        "fit_scores": fit_scores,
    }
    (output / "COMPLETE.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(name, "verified", len(fit_scores), "development fits", flush=True)


def run(name, source, output, jobs):
    output.mkdir(parents=True, exist_ok=False)
    data, keys, y, groups, ids, split = make_development_cohort(name, source, output)
    train = split.fit
    bits = data["S_d_pooled"][train, 1024:]
    keys_train = [keys[i] for i in train]
    target_train, groups_train = y[train], groups[train]
    diagnostic = regression_diagnostic(bits, keys_train, groups_train, target_train)
    shuffled = regression_diagnostic(bits, keys_train, groups_train, target_train, permutation_seed=2026)
    summary = diagnostic.summary(target_train, groups_train)
    (output / "training_diagnostic.json").write_text(json.dumps({
        "scope": "development_training_only", "targets": TARGETS[name],
        "original": summary, "permutation": shuffled.summary(target_train, groups_train),
    }, indent=2) + "\n")
    np.savez_compressed(output / "training_diagnostic_oof.npz", row_id=ids[train],
                        eligible=diagnostic.eligible, delta_prediction=diagnostic.delta_prediction,
                        context_prediction=diagnostic.context_prediction,
                        fitting_variance=diagnostic.fitting_variance)
    print(name, "training diagnostic", summary, flush=True)
    cells = output / "cells"
    cells.mkdir()
    results = Parallel(n_jobs=jobs, verbose=5)(
        delayed(run_fit)(name, arm, target, i, seed, data[f"S_{arm}"], y, groups, ids, split, cells)
        for (i, target), arm, seed in product(enumerate(TARGETS[name]), ARMS, SEEDS)
    )
    score_development(name, output, y, groups, ids, split, summary, results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=12)
    args = parser.parse_args()
    sources = json.loads(args.input_manifest.read_text())["sources"]
    args.output.mkdir(parents=True, exist_ok=False)
    for name in ("b97xd3", "cycloadd"):
        if sources[name]["status"] != "source_intake_verified":
            raise ValueError(f"Missing pinned development input: {name}")
        run(name, Path(sources[name]["path"]), args.output / name, args.jobs)
    (args.output / "COMPLETE.json").write_text(json.dumps({
        "phase": "complete", "sources": ["b97xd3", "cycloadd"], "development_fits": 60,
        "completed_sha256": {name: sha256(args.output / name / "COMPLETE.json")
                             for name in ("b97xd3", "cycloadd")},
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
