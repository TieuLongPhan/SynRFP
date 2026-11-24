"""Apply the frozen two-target diagnostic to validated EnzymeMap training rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
from hashlib import sha256 as digest
import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from audit_inputs import PINS, sha256
from enzymemap_cohort import EC_PATTERN
from map_enzymemap_labels import edit_label, read, training_rows, verify_precheck, write
from paired_analysis import means_by_group
from precheck_enzymemap import MIN_CLASSES, MIN_GROUPS, write_rows
from prepare_qp_features import encode, features

NAMES = ("ec_level_2", "bond_edit_multiset")


def ec_targets(source, selected):
    targets = {}
    with gzip.open(source, "rt", newline="") as handle:
        for position, record in enumerate(csv.DictReader(handle)):
            row = selected.get(position)
            if row is None:
                continue
            if record["rxn_idx"] not in set(json.loads(row["source_ids"])):
                raise ValueError("EC label source row no longer matches frozen reaction")
            match = EC_PATTERN.fullmatch(record["ec_num"].strip())
            if (not match or record["source"] != "direct" or record["steps"] != "single"):
                raise ValueError("Expected a valid two-field EC label in the pinned source")
            targets[row["row_id"]] = match[1]
    if len(targets) != len(selected):
        raise ValueError("Incomplete frozen EC target join")
    return targets


def winner(scores, priors):
    # Votes are sums of rational group weights. Allow only roundoff in ties.
    tied = np.flatnonzero(np.isclose(scores, scores.max(), rtol=0, atol=1e-12))
    return tied[np.argmax(priors[tied])]


def classification_diagnostic(bits, keys, groups, targets):
    if len(keys) != len(groups) or len(bits) != len(groups) or targets.shape != (len(groups), 2):
        raise ValueError("Training rows are not aligned")
    classes = []
    encoded = []
    for labels in targets.T:
        values, inverse = np.unique(labels, return_inverse=True)
        classes.append(values)
        encoded.append(inverse)
    encoded = np.asarray(encoded).T
    folds = np.full(len(groups), -1, dtype=np.int8)
    priors = {}
    for fold, (reference, query) in enumerate(GroupKFold(5).split(bits, groups=groups)):
        folds[query] = fold
        priors[fold] = [
            np.bincount(encoded[reference, t], minlength=len(values))
            for t, values in enumerate(classes)
        ]
    assert np.all(folds >= 0)
    by_key = defaultdict(list)
    for row, key in enumerate(keys):
        by_key[key].append(row)
    delta = np.full_like(encoded, -1)
    pooled = np.full_like(encoded, -1)
    ties = 0
    for members in by_key.values():
        index = np.asarray(members)
        _, local_group = np.unique(groups[index], return_inverse=True)
        if len(set(local_group)) < 4:
            continue
        fold_ids = folds[index]
        for fold in np.unique(fold_ids):
            refs = np.flatnonzero(fold_ids != fold)
            _, reference_groups = np.unique(local_group[refs], return_inverse=True)
            group_sizes = np.bincount(reference_groups)
            if len(group_sizes) < 3:
                continue
            ref_labels = encoded[index[refs]]
            ref_bits = bits[index[refs]]
            weights = 1 / group_sizes[reference_groups]
            base = [
                winner(np.bincount(ref_labels[:, t], weights=weights, minlength=len(values)), priors[fold][t])
                for t, values in enumerate(classes)
            ]
            for query in np.flatnonzero(fold_ids == fold):
                row = index[query]
                vector = bits[row]
                if np.all(ref_bits == vector):
                    continue
                intersection = np.bitwise_count(ref_bits & vector).sum(axis=1)
                union = np.bitwise_count(ref_bits | vector).sum(axis=1)
                similarity = np.divide(intersection, union, out=np.ones(len(union)), where=union > 0)
                closest = np.full(len(group_sizes), -np.inf)
                np.maximum.at(closest, reference_groups, similarity)
                cutoff = np.partition(closest, -3)[-3]
                above = closest > cutoff
                boundary = closest == cutoff
                remaining = 3 - int(above.sum())
                ties += int(int(boundary.sum()) > remaining)
                group_weights = above.astype(float) + boundary * (remaining / int(boundary.sum()))
                tied_refs = similarity == closest[reference_groups]
                counts = np.bincount(reference_groups[tied_refs], minlength=len(closest))
                selected = reference_groups[tied_refs]
                context_weights = group_weights[selected] / counts[selected]
                delta[row] = base
                pooled[row] = [
                    winner(np.bincount(ref_labels[tied_refs, t], weights=context_weights,
                                       minlength=len(values)), priors[fold][t])
                    for t, values in enumerate(classes)
                ]
    eligible = np.all(delta >= 0, axis=1)
    assert np.array_equal(eligible, np.all(pooled >= 0, axis=1))
    losses = (delta[eligible] != encoded[eligible]).astype(float) - (pooled[eligible] != encoded[eligible])
    scores = means_by_group(losses, groups[eligible]).mean(axis=0).tolist() if eligible.any() else None
    return classes, folds, delta, pooled, eligible, scores, ties


def run(precheck, followup, mapping, source, output, jobs):
    original, pins = verify_precheck(precheck, followup)
    if features.environment() != original["fingerprint_environment"]:
        raise ValueError("Use the pinned precheck fingerprint environment")
    complete = read(mapping / "COMPLETE.json")
    manifest_path = mapping / "manifest.json"
    labels_path = mapping / "train_labels.json"
    if (complete["pilot"] is not None or sha256(manifest_path) != complete["manifest_sha256"]
        or sha256(labels_path) != complete["labels_sha256"]):
        raise ValueError("Expected the verified full-training mapping results")
    manifest = read(manifest_path)
    if manifest["scope"] != "train" or manifest["pinned_inputs"] != {str(p): h for p, h in pins.items()}:
        raise ValueError("Mapping is not bound to the verified training cohort")
    if sha256(source) != PINS["enzymemap"][1]:
        raise ValueError("Changed EnzymeMap archive")
    ledger = training_rows(precheck, original["cohort"]["rows"]["train"])
    with gzip.open(precheck / "training_encodings.csv.gz", "rt", newline="") as handle:
        prior_encodings = {row["row_id"]: row for row in csv.DictReader(handle)}
    rows, mappings = [], []
    for row, result in zip(ledger, read(labels_path), strict=True):
        if result["row_id"] != row["row_id"]:
            raise ValueError("Mapping and labels are not aligned")
        if "refusal" in result:
            continue
        if edit_label(row["reaction_smiles"], result["mapped_reaction"]) != result["label"]:
            raise ValueError("Mapped edit label failed independent replay")
        rows.append(row)
        mappings.append(result["label"])
    if len(rows) != complete["validated"]:
        raise ValueError("Mapping receipt disagrees with training results")
    ec = ec_targets(source, {int(row["representative_source_index"]): row for row in rows})
    selected = [(n, row["row_id"], row["reaction_smiles"]) for n, row in enumerate(rows)]
    print(f"Replaying features for {len(selected):,} validated training rows", flush=True)
    paired, targets, bits, keys = [], [], [], []
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for row, label, result in zip(rows, mappings, pool.map(encode, selected, chunksize=64), strict=True):
            index, row_id, vectors, exact, _, refusal = result
            prior = prior_encodings[row_id]
            if row_id != row["row_id"] or rows[index]["row_id"] != row_id:
                raise ValueError("Encoded row alignment changed")
            if refusal != prior["feature_refusal"]:
                raise ValueError("Feature admission changed")
            if refusal:
                continue
            if digest(exact).hexdigest() != prior["exact_delta_sha256"] or row["group_id"] != prior["group_id"]:
                raise ValueError("Frozen signed counts or scaffold assignment changed")
            paired.append(row)
            targets.append((ec[row_id], label))
            bits.append(vectors["d_pooled"][128:])
            keys.append(exact)
    groups = np.asarray([row["group_id"] for row in paired])
    output.mkdir(parents=True, exist_ok=False)
    print(f"Running five-fold diagnostic on {len(paired):,} training rows", flush=True)
    names, folds, delta, pooled, eligible, scores, ties = classification_diagnostic(
        np.asarray(bits), keys, groups, np.asarray(targets)
    )
    count = int(eligible.sum())
    query_groups = len(set(groups[eligible]))
    exact_classes = len({key for key, admitted in zip(keys, eligible, strict=True) if admitted})
    thresholds = {"training_fraction": original["final_training_row_fraction_minimum"],
                  "query_groups": MIN_GROUPS, "exact_classes": MIN_CLASSES}
    passes = (count / len(paired) >= thresholds["training_fraction"]
              and query_groups >= MIN_GROUPS and exact_classes >= MIN_CLASSES)
    higher = None
    if passes and scores is not None and not np.isclose(*scores, rtol=0, atol=1e-12):
        higher = NAMES[int(np.argmax(scores))]
    ledger_rows = []
    for i, row in enumerate(paired):
        record = {"row_id": row["row_id"], "group_id": row["group_id"], "fold": int(folds[i]),
                  "exact_delta_sha256": digest(keys[i]).hexdigest(), "eligible": int(eligible[i])}
        for t, name in enumerate(NAMES):
            record[name] = targets[i][t]
            record[f"{name}_delta"] = names[t][delta[i, t]] if eligible[i] else ""
            record[f"{name}_pooled"] = names[t][pooled[i, t]] if eligible[i] else ""
        ledger_rows.append(record)
    oof = output / "training_oof.csv.gz"
    write_rows(oof, tuple(ledger_rows[0]), ledger_rows)
    report = {
        "decision": "order_frozen" if higher else "abstain",
        "scope": "training_only", "model_fits": 0, "higher_gain_target": higher,
        "training_precheck_rows": len(ledger), "mapped_training_rows": len(rows),
        "mapping_refusals": complete["refusals"], "paired_training_rows": len(paired),
        "eligible_queries": count, "query_groups": query_groups, "exact_delta_classes": exact_classes,
        "coverage_fraction": count / len(paired), "fraction_of_precheck_rows": count / len(ledger),
        "thresholds": thresholds, "coverage_passes": passes, "third_neighbor_boundary_ties": ties,
        "group_weighted_delta_minus_context_error": dict(zip(NAMES, scores, strict=True)) if scores else None,
        "training_class_counts": dict(zip(NAMES, [len(values) for values in names], strict=True)),
        "no_bond_edit_rows": mappings.count("no_edit"),
        "mapping_receipt_sha256": sha256(mapping / "COMPLETE.json"),
        "eligibility_precheck_sha256": sha256(precheck / "eligibility_precheck.json"),
        "overlap_followup_sha256": sha256(followup / "eligibility_followup.json"),
        "training_oof_sha256": sha256(oof),
        "runner_sha256": sha256(Path(__file__)),
    }
    write(output / "prediction.json", report)
    print(report, flush=True)


def main():
    parser = argparse.ArgumentParser()
    for argument in ("precheck", "followup", "mapping", "source", "output"):
        parser.add_argument(f"--{argument}", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    run(args.precheck.resolve(), args.followup.resolve(), args.mapping.resolve(),
        args.source.resolve(), args.output.resolve(), args.jobs)


if __name__ == "__main__":
    main()
