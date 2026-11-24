"""Freeze the audited QuantumPioneer cohort and its training-only target ordering."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from audit_inputs import PINS, TARGETS, sha256
from audit_quantumpioneer_graphs import digest, molecule
from paired_analysis import regression_diagnostic


def read(path):
    return json.loads(path.read_text())


def verified_artifacts(folder):
    receipt = read(folder / "COMPLETE.json")
    for name, expected in receipt["outputs_sha256"].items():
        if sha256(folder / name) != expected:
            raise ValueError(f"Changed input artifact {folder / name}")
    return receipt


def csv_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle)


@lru_cache(maxsize=16384)
def scaffold(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Previously parsed canonical reactant failed to reparse")
    core = MurckoScaffold.GetScaffoldForMol(mol)
    return Chem.MolToSmiles(core) if core.GetNumAtoms() else smiles


def eligible_rows(source, graphs, features, overlap):
    graph_rows = {row["row_id"]: row for row in csv_rows(graphs / "row_graph_audit.csv.gz")}
    feature_rows = list(csv_rows(features / "feature_rows.csv.gz"))
    previous_overlap_groups = set(overlap["whole_group_exclusions"])
    exclusions = Counter()
    included = {split: [] for split in ("train", "validation", "test")}
    ledger = []
    targets = TARGETS["quantumpioneer"]
    if len(graph_rows) != PINS["quantumpioneer"][2] or len(feature_rows) != len(graph_rows):
        raise ValueError("Graph and feature row counts differ")
    for i, (source_row, feature_row) in enumerate(zip(csv_rows(source), feature_rows, strict=True)):
        row_id = source_row["source_record_id"]
        graph_row = graph_rows[row_id]
        if feature_row["row_id"] != row_id or int(feature_row["matrix_row"]) != i:
            raise ValueError("Source and feature row identity mismatch")
        split = source_row["proposed_split"]
        if graph_row["split"] != split:
            raise ValueError("Source split changed during preprocessing")
        group = graph_row["connected_group"]
        reason = graph_row["internal_exclusion"]
        if not reason and group in previous_overlap_groups:
            reason = "prior_benchmark_overlap_group"
        if not reason and feature_row["feature_refusal"]:
            reason = "feature_refusal"
        ledger.append({"row_id": row_id, "matrix_row": i, "partition": split,
                       "connected_group": group, "exclusion": reason,
                       "exact_delta_sha256": feature_row["exact_delta_sha256"]})
        if reason:
            exclusions[reason] += 1
            continue
        reactants = tuple(molecule(part)[0] for part in source_row["reaction_smiles"].split(">")[0].split("."))
        # Never interpret a test outcome during cohort construction or prediction lock.
        observed = [float(source_row[target]) for target in targets] if split != "test" else None
        included[split].append((i, row_id, group, feature_row["exact_delta_sha256"], reactants, observed))
    groups = [{row[2] for row in subset} for subset in included.values()]
    if any(not group for group in groups) or sum(map(len, groups)) != len(set.union(*groups)):
        raise ValueError("Empty or crossing connected groups after eligibility")
    return ledger, included, dict(exclusions)


def training_membership(included):
    training_partners = {p for row in included["train"] for p in row[4]}
    training_scaffolds = {scaffold(p) for p in training_partners}
    return {
        "distinct_training_partners": len(training_partners),
        "test_rows_with_recurring_partner": sum(
            any(p in training_partners for p in row[4]) for row in included["test"]
        ),
        "test_rows_with_all_partners_recurring": sum(
            all(p in training_partners for p in row[4]) for row in included["test"]
        ),
        "test_rows_with_shared_scaffold": sum(
            any(scaffold(p) in training_scaffolds for p in row[4]) for row in included["test"]
        ),
        "test_rows_with_all_scaffolds_shared": sum(
            all(scaffold(p) in training_scaffolds for p in row[4]) for row in included["test"]
        ),
    }


def freeze(source, graphs, features, overlap_path, output):
    expected = PINS["quantumpioneer"][1]
    overlap = read(overlap_path)
    graph_receipt, feature_receipt = verified_artifacts(graphs), verified_artifacts(features)
    if any(value != expected for value in (
        sha256(source), graph_receipt["source_sha256"], feature_receipt["source_sha256"],
        overlap["quantumpioneer_sha256"],
    )):
        raise ValueError("Unpinned QuantumPioneer input")
    if (overlap["phase"] != "listed_benchmark_sources_audited" or
        overlap["graph_audit_sha256"] != sha256(graphs / "row_graph_audit.csv.gz")):
        raise ValueError("Missing overlap audit for the verified graph ledger")

    output.mkdir(parents=True, exist_ok=False)
    ledger, included, exclusions = eligible_rows(source, graphs, features, overlap)
    row_ledger = output / "eligibility.csv.gz"
    with gzip.open(row_ledger, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ledger[0]))
        writer.writeheader()
        writer.writerows(ledger)
    arrays = {}
    split_summary = {}
    for name, rows in included.items():
        numbers, ids, groups, keys, _, targets = zip(*rows)
        arrays[f"{name}_indices"] = np.asarray(numbers, dtype=np.int64)
        arrays[f"{name}_ids"] = np.asarray(ids)
        arrays[f"{name}_groups"] = np.asarray(groups)
        if name != "test":
            arrays[f"{name}_targets"] = np.asarray(targets)
        split_summary[name] = {"rows": len(rows), "groups": len(set(groups)),
                               "row_ids_sha256": digest(ids)}
    membership = training_membership(included)
    cohort_file = output / "cohort.npz"
    np.savez_compressed(cohort_file, **arrays)
    train = included["train"]
    train_groups = arrays["train_groups"]
    train_targets = arrays["train_targets"]
    # The pooled block occupies the last 1024 bits in the recorded encoding.
    pooled = np.load(features / "d_pooled.npy", mmap_mode="r", allow_pickle=False)
    bits = np.unpackbits(pooled[arrays["train_indices"], 128:], axis=1)
    keys = [row[3] for row in train]  # All digest equivalences were checked against full signed-count tuples.
    print("Running strictly training-only diagnostic", len(train), "rows", flush=True)
    diagnostic = regression_diagnostic(bits, keys, train_groups, train_targets)
    summary = diagnostic.summary(train_targets, train_groups)
    result = output / "training_diagnostic_oof.npz"
    np.savez_compressed(result, row_id=arrays["train_ids"], group=train_groups,
                        eligible=diagnostic.eligible, delta_prediction=diagnostic.delta_prediction,
                        context_prediction=diagnostic.context_prediction,
                        fitting_variance=diagnostic.fitting_variance)
    order = summary["higher_gain_target_index"]
    targets = TARGETS["quantumpioneer"]
    prediction = {
        "phase": "training_only_prediction_frozen", "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256(Path(__file__).with_name("PROTOCOL.md")),
        "checkout_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_sha256": expected,
        "graph_audit_receipt_sha256": sha256(graphs / "COMPLETE.json"),
        "feature_receipt_sha256": sha256(features / "COMPLETE.json"),
        "overlap_audit_sha256": sha256(overlap_path),
        "source_code_sha256": {name: sha256(Path(__file__).with_name(name))
                               for name in ("freeze_qp_prediction.py", "paired_analysis.py")},
        "targets": targets, "diagnostic": summary,
        "predicted_higher_gain_target": targets[order] if order is not None else None,
        "exclusions": exclusions, "split_summary": split_summary,
        "heldout_chemistry_overlap": membership,
        "artifact_sha256": {p.name: sha256(p) for p in (cohort_file, row_ledger, result)},
        "heldout_outcomes_scored": False,
    }
    file = output / "prediction.json"
    temporary = file.with_suffix(".tmp")
    temporary.write_text(json.dumps(prediction, indent=2) + "\n")
    temporary.replace(file)
    print(json.dumps(prediction, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("source", "graphs", "features", "overlap", "output"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    args = parser.parse_args()
    freeze(args.source, args.graphs, args.features, args.overlap, args.output)


if __name__ == "__main__":
    main()
