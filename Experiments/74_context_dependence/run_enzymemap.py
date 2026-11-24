"""Run the frozen EnzymeMap held-out replication on the common mapped cohort."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import fcntl
from itertools import product
import json
from pathlib import Path
import subprocess
import tempfile
import time

from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import matthews_corrcoef, normalized_mutual_info_score, recall_score
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from audit_enzymemap_labels import NAMES, ec_targets
from audit_inputs import PINS, sha256
from map_enzymemap_labels import cohort_rows, edit_label, verify_precheck
from paired_analysis import means_by_group, paired_bootstrap
from prepare_qp_features import ARMS, WIDTHS, encode, features
from run_quantumpioneer import SEEDS, e1, matrices, read, utc, write

STUDY = Path(__file__).resolve().parent
SPLITS = ("train", "validation", "test")
GRID = tuple(product(NAMES, ARMS, SEEDS))


def mapping_receipt(folder, split, expected, pins, train_manifest=None):
    manifest_path = folder / "manifest.json"
    result = read(folder / "COMPLETE.json")
    manifest = read(manifest_path)
    if (result["pilot"] is not None or result["rows"] != expected
        or manifest["scope"] != split or manifest["expected"] != expected
        or manifest["pinned_inputs"] != {str(path): digest for path, digest in pins.items()}
        or sha256(manifest_path) != result["manifest_sha256"]):
        raise ValueError(f"Mapping provenance mismatch for {split}")
    if train_manifest:
        fields = ("mapper_revision", "mapper_assets", "profile", "versions", "mapper_head",
                  "mapper_layers", "confidence_cutoff", "device", "batch_size")
        if any(manifest[field] != train_manifest[field] for field in fields):
            raise ValueError(f"Mapper settings changed for {split}")
    return result, manifest


def mapped_targets(source, folder, split, all_rows, included_rows, receipt):
    path = folder / f"{split}_labels.json"
    if sha256(path) != receipt["labels_sha256"]:
        raise ValueError(f"Changed {split} bond-edit labels")
    included = {row["row_id"] for row in included_rows}
    bond = {}
    for row, item in zip(all_rows, read(path), strict=True):
        if item["row_id"] != row["row_id"]:
            raise ValueError(f"Changed {split} mapping row order")
        if "refusal" in item:
            if row["row_id"] in included:
                raise ValueError("A mapper refusal entered the comparison")
            continue
        label = edit_label(row["reaction_smiles"], item["mapped_reaction"])
        if label != item["label"]:
            raise ValueError(f"Changed bond-edit label in {split}")
        bond[item["row_id"]] = label
    if set(bond) != included:
        raise ValueError(f"Changed common mapped cohort for {split}")
    positions = {int(row["representative_source_index"]): row for row in included_rows}
    ec = ec_targets(source, positions)
    return np.asarray([(ec[row["row_id"]], bond[row["row_id"]]) for row in included_rows])


def validated_rows(base, precheck, followup, published):
    pre, pins = verify_precheck(precheck, followup)
    if pre["fingerprint_environment"] != features.environment():
        raise ValueError("Feature environment changed")
    prediction_file = published / "prediction.json"
    frozen = read(prediction_file)
    proof = read(published / "VERIFIED.json")
    if (proof["prediction_sha256"] != sha256(prediction_file) or not proof["training_only"]
        or proof["heldout_model_fits"] != 0 or frozen["decision"] != "order_frozen"
        or not frozen["coverage_passes"] or frozen["scope"] != "training_only"
        or frozen["overlap_followup_sha256"] != sha256(followup / "eligibility_followup.json")
        or frozen["eligibility_precheck_sha256"] != sha256(precheck / "eligibility_precheck.json")):
        raise ValueError("Frozen training-only prediction failed verification")
    all_rows, common, receipts, folders = {}, {}, {}, {}
    train_manifest = None
    for split in SPLITS:
        folders[split] = base / f"enzymemap-labels-{split}-v1"
        expected = pre["cohort"]["rows"][split]
        receipt, manifest = mapping_receipt(folders[split], split, expected, pins, train_manifest)
        receipts[split] = receipt
        all_rows[split] = cohort_rows(precheck, split, expected)
        if split == "train":
            train_manifest = manifest
            if sha256(folders[split] / "COMPLETE.json") != frozen["mapping_receipt_sha256"]:
                raise ValueError("Training map receipt differs from the frozen prediction")
            path = folders[split] / "train_labels.json"
            if sha256(path) != receipt["labels_sha256"]:
                raise ValueError("Changed training map")
            admitted = [r["row_id"] for r in read(path) if "label" in r]
        else:
            # This ledger contains only IDs and is separate from the held-out labels.
            path = folders[split] / "admitted.json"
            if sha256(path) != receipt["admitted_sha256"]:
                raise ValueError(f"Changed {split} admission ledger")
            admitted = read(path)
        ids = set(admitted)
        common[split] = [row for row in all_rows[split] if row["row_id"] in ids]
        if len(common[split]) != len(ids) or len(ids) != receipt["validated"]:
            raise ValueError(f"Incomplete matched cohort for {split}")
    if len(common["train"]) != frozen["paired_training_rows"]:
        raise ValueError("Training rows differ from the prediction diagnostic")
    group_sets = [{row["group_id"] for row in rows} for rows in common.values()]
    if len(set.union(*group_sets)) != sum(map(len, group_sets)):
        raise ValueError("A scaffold group crosses partitions")
    return frozen, all_rows, common, receipts, folders


def encode_cohort(rows, jobs, output):
    archive = output / "features.npz"
    status = output / "features.json"
    ids = [row["row_id"] for row in rows]
    if archive.exists():
        proof = read(status)
        if proof["archive_sha256"] != sha256(archive) or proof["runner_sha256"] != sha256(Path(__file__)):
            raise ValueError("Changed saved features or their runner")
        with np.load(archive, allow_pickle=False) as saved:
            if saved["row_ids"].tolist() != ids:
                raise ValueError("Changed feature membership")
            return {arm: saved[arm] for arm in ARMS}
    packed = {arm: np.empty((len(rows), WIDTHS[arm] // 8), dtype=np.uint8) for arm in ARMS}
    inputs = [(i, r["row_id"], r["reaction_smiles"]) for i, r in enumerate(rows)]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for i, result in enumerate(pool.map(encode, inputs, chunksize=64)):
            index, row_id, vectors, _, _, refusal = result
            if index != i or row_id != ids[i] or refusal:
                raise ValueError(f"Unexpected feature failure or row drift at {i}: {refusal}")
            for arm in ARMS:
                packed[arm][i] = vectors[arm]
            if (i + 1) % 5000 == 0:
                print(f"Encoded {i+1:,}/{len(rows):,} frozen reactions", flush=True)
    temporary = output / ".features.npz"
    np.savez_compressed(temporary, row_ids=ids, **packed)
    temporary.replace(archive)
    write(status, {"archive_sha256": sha256(archive), "runner_sha256": sha256(Path(__file__))})
    return packed


def fit_cell(packed, parts, y_train, y_val, classes, arm, target, seed, directory, plan_sha):
    final = directory / f"{target}__{arm}__s{seed}"
    if final.exists():
        saved = read(final / "COMPLETE.json")
        if saved["plan_sha256"] != plan_sha:
            raise ValueError(f"Different plan for {final.name}")
        for name, expected in saved["outputs_sha256"].items():
            if sha256(final / name) != expected:
                raise ValueError(f"Changed fit artifact {final.name}/{name}")
        return saved

    start = time.monotonic()
    train, validation, test = matrices(packed, parts)
    model = MLPClassifier(**e1.FIT_OPTIONS, random_state=np.random.RandomState(seed))
    best, patience_loss, stale, best_epoch = np.inf, np.inf, 0, 0
    with threadpool_limits(limits=1):
        for epoch in range(1, e1.MAX_EPOCHS + 1):
            model.partial_fit(train, y_train, classes=classes)
            probabilities = model.predict_proba(validation)
            known = y_val >= 0
            floor = np.finfo(probabilities.dtype).eps
            likelihood = np.full(len(y_val), floor, dtype=probabilities.dtype)
            likelihood[known] = probabilities[known, y_val[known]]
            val_loss = float(-np.mean(np.log(np.clip(likelihood, floor, 1.0)), dtype=np.float64))
            if not np.isfinite(val_loss):
                raise ValueError(f"Invalid cross-entropy in {final.name}")
            if val_loss < best:
                best, best_epoch = val_loss, epoch
                best_parameters = ([x.copy() for x in model.coefs_], [x.copy() for x in model.intercepts_])
            if patience_loss - val_loss >= e1.TOLERANCE:
                patience_loss, stale = val_loss, 0
            else:
                stale += 1
            if stale >= e1.PATIENCE:
                break
        model.coefs_, model.intercepts_ = best_parameters
        predictions = model.predict(test)
    if len(predictions) != len(parts[2]) or not np.isin(predictions, classes).all():
        raise ValueError(f"Invalid held-out prediction in {final.name}")
    attempt = Path(tempfile.mkdtemp(prefix=f".{final.name}-", dir=directory))
    checkpoint = attempt / "checkpoint.npz"
    np.savez_compressed(checkpoint, classes=classes,
                        **{f"coef{i}": x for i, x in enumerate(best_parameters[0])},
                        **{f"intercept{i}": x for i, x in enumerate(best_parameters[1])})
    blind = attempt / "blind_predictions.npz"
    np.savez_compressed(blind, matrix_row=parts[2], prediction=predictions)
    receipt = {"target": target, "arm": arm, "seed": seed, "plan_sha256": plan_sha,
               "epochs": epoch, "best_epoch": best_epoch, "validation_cross_entropy": best,
               "seconds": time.monotonic() - start,
               "outputs_sha256": {path.name: sha256(path) for path in (checkpoint, blind)}}
    write(attempt / "COMPLETE.json", receipt)
    attempt.rename(final)
    print(f"{final.name}: {epoch} epochs, {receipt['seconds']:.1f}s", flush=True)
    return receipt


def collect(fit_root, receipts, test, plan_sha):
    predictions = {}
    for result in receipts:
        key = tuple(result[name] for name in ("target", "arm", "seed"))
        if key in predictions or result["plan_sha256"] != plan_sha:
            raise ValueError("Duplicate or mismatched fit receipt")
        folder = fit_root / f"{key[0]}__{key[1]}__s{key[2]}"
        for name, expected in result["outputs_sha256"].items():
            if sha256(folder / name) != expected:
                raise ValueError(f"Fit artifact changed in {folder.name}")
        with np.load(folder / "blind_predictions.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["matrix_row"], test)
            predictions[key] = archive["prediction"]
    if set(predictions) != set(GRID):
        raise ValueError("Expected all 30 fits before reading held-out labels")
    return predictions


def score(predictions, y, train, vocabs, groups, order, output):
    rows, gains, keys, majority = [], [], [], {}
    for i, target in enumerate(NAMES):
        observed = y[:, i]
        classes = np.unique(observed)
        counts = Counter(train[:, i])
        baseline = min(counts, key=lambda label: (-counts[label], label))
        majority[target] = {
            "row_accuracy": float(np.mean(observed == baseline)),
            "group_accuracy": float(means_by_group((observed == baseline).astype(float), groups).mean()),
            "unseen_test_label_rows": int((~np.isin(observed, vocabs[i])).sum()),
            "training_classes": len(vocabs[i]), "test_classes": len(classes),
            "training_class_frequencies": dict(sorted(counts.items())),
            "test_class_frequencies": dict(sorted(Counter(observed).items())),
        }
        error = {}
        for arm in ARMS:
            errors = []
            for seed in SEEDS:
                predicted = vocabs[i][predictions[target, arm, seed]]
                misses = predicted != observed
                errors.append(misses)
                rows.append({
                    "target": target, "arm": arm, "seed": seed, "accuracy": 1 - float(misses.mean()),
                    "group_accuracy": 1 - float(means_by_group(misses.astype(float), groups).mean()),
                    "macro_recall": float(recall_score(observed, predicted, labels=classes, average="macro", zero_division=0)),
                    "mcc": float(matthews_corrcoef(observed, predicted)),
                })
            error[arm] = np.mean(errors, axis=0)
        for arm in ARMS:
            if arm == "d_pooled":
                continue
            keys.append((target, arm))
            gains.append(error[arm] - error["d_pooled"])
    grouped = means_by_group(np.column_stack(gains), groups)
    draws = paired_bootstrap(grouped, draws=10_000, seed=2026)
    estimates = [
        {"target": target, "comparator": arm, "gain": float(grouped[:, i].mean()),
         "interval": np.quantile(draws[:, i], (0.025, 0.975)).tolist()}
        for i, (target, arm) in enumerate(keys)
    ]
    for estimate in estimates:
        majority_error = 1 - majority[estimate["target"]]["group_accuracy"]
        estimate["gain_relative_to_majority_error"] = (
            estimate["gain"] / majority_error if majority_error else None
        )
    other = next(name for name in NAMES if name != order)
    high, low = keys.index((order, "d2048")), keys.index((other, "d2048"))
    interaction = {"higher_ranked_target": order, "other_target": other,
                   "gain_difference": float((grouped[:, high] - grouped[:, low]).mean()),
                   "interval": np.quantile(draws[:, high] - draws[:, low], (0.025, 0.975)).tolist()}
    np.savez_compressed(output / "group_bootstrap.npz", draws=draws, group_gains=grouped,
                        contrasts=np.asarray(keys), group_ids=np.unique(groups))
    contingency = Counter(zip(y[:, 0], y[:, 1], strict=True))
    with (output / "test_label_contingency.csv").open("w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow([*NAMES, "count"])
        w.writerows([*pair, n] for pair, n in sorted(contingency.items()))
    with (output / "fit_scores.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, rows[0])
        w.writeheader()
        w.writerows(rows)
    return {"scope": "held_out_classification_replication", "rows": len(y),
            "groups": len(set(groups)), "gain_unit": "group_weighted_error_fraction",
            "estimates": estimates, "interaction": interaction, "baseline_and_coverage": majority,
            "target_normalized_mutual_information": float(normalized_mutual_info_score(y[:, 0], y[:, 1])),
            "bootstrap_seed": 2026, "bootstrap_draws": len(draws)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("runs", "source", "output", "development"):
        parser.add_argument(f"--{field}", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 12:
        parser.error("use between one and twelve CPU workers")
    verify_development = read(args.development)
    if verify_development["status"] != "passed" or verify_development["runner_sha256"] != sha256(Path(__file__)):
        raise ValueError("A matching multiclass development smoke test is required")
    base = args.runs.resolve()
    source = args.source.resolve()
    if sha256(source) != PINS["enzymemap"][1]:
        raise ValueError("Unexpected enzyme data hash")
    repo = STUDY.parents[1]
    checkout = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(repo), "status", "--porcelain", "--",
                                str(STUDY), "Experiments/73_context_attribution", "synrfp"], text=True).strip():
        raise ValueError("Commit the experiment source before the comparison")
    precheck, followup = base / "enzymemap-precheck-v1", base / "enzymemap-refusal-audit-v1"
    published = STUDY / "publication_metadata/enzymemap_v1"
    prediction, all_rows, common, map_receipts, map_folders = validated_rows(base, precheck, followup, published)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "run.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = output / "status.json"
    if status_path.exists() and read(status_path)["phase"] in ("heldout_scoring", "complete"):
        raise ValueError("Held-out scoring was already opened for this run")

    rows = [row for split in SPLITS for row in common[split]]
    parts = tuple(np.flatnonzero([row["split"] == split for row in rows]) for split in SPLITS)
    plan = {"checkout_commit": checkout, "protocol_sha256": sha256(STUDY / "PROTOCOL.md"),
            "source_sha256": PINS["enzymemap"][1], "prediction_sha256": sha256(published / "prediction.json"),
            "development_smoke_sha256": sha256(args.development), "learner": e1.FIT_OPTIONS,
            "patience": e1.PATIENCE, "tolerance": e1.TOLERANCE, "seeds": SEEDS, "arms": ARMS,
            "targets": NAMES, "unseen_validation_class": "clipped_probability_zero",
            "unseen_test_class": "count_as_error",
            "partition_sources": {s: len(all_rows[s]) for s in SPLITS},
            "partition_admitted": {s: len(common[s]) for s in SPLITS},
            "partition_refusals": {s: map_receipts[s]["refusals"] for s in SPLITS},
            "partition_groups": {s: len({r["group_id"] for r in common[s]}) for s in SPLITS},
            "map_receipt_sha256": {s: sha256(map_folders[s] / "COMPLETE.json") for s in SPLITS}}
    plan_path = output / "plan.json"
    if plan_path.exists() and read(plan_path) != json.loads(json.dumps(plan)):
        raise ValueError("The existing experiment uses another plan")
    write(plan_path, plan)
    plan_sha = sha256(plan_path)
    packed = encode_cohort(rows, args.jobs, output)
    labels = [
        mapped_targets(source, map_folders[s], s, all_rows[s], common[s], map_receipts[s])
        for s in ("train", "validation")
    ]
    vocabs = [np.unique(labels[0][:, i]) for i in range(len(NAMES))]
    train_targets, validation_targets = [], []
    for i, vocab in enumerate(vocabs):
        index = {label: j for j, label in enumerate(vocab)}
        train_targets.append(np.asarray([index[label] for label in labels[0][:, i]]))
        validation_targets.append(np.asarray([index.get(label, -1) for label in labels[1][:, i]]))
    fit_root = output / "fits"
    fit_root.mkdir(exist_ok=True)
    write(status_path, {"phase": "blind_fitting", "utc": utc(), "plan_sha256": plan_sha})
    print(f"Starting {len(GRID)} blind fits on {args.jobs} workers", flush=True)
    receipts = Parallel(n_jobs=args.jobs)(
        delayed(fit_cell)(packed[arm], parts, train_targets[i], validation_targets[i],
                          np.arange(len(vocabs[i])), arm, target, seed, fit_root, plan_sha)
        for i, target in enumerate(NAMES) for arm, seed in product(ARMS, SEEDS)
    )
    predictions = collect(fit_root, receipts, parts[2], plan_sha)
    write(output / "fit_receipts.json", receipts)
    write(status_path, {"phase": "heldout_scoring", "utc": utc(), "plan_sha256": plan_sha,
                        "verified_fits": len(receipts)})
    # The EC and bond-edit test targets are first read here, after all fits close.
    test_y = mapped_targets(source, map_folders["test"], "test", all_rows["test"], common["test"], map_receipts["test"])
    groups = np.asarray([row["group_id"] for row in common["test"]])
    result = score(predictions, test_y, labels[0], vocabs, groups, prediction["higher_gain_target"], output)
    write(output / "scores.json", result)
    filenames = ("scores.json", "fit_scores.csv", "fit_receipts.json", "plan.json",
                 "group_bootstrap.npz", "test_label_contingency.csv")
    write(status_path, {"phase": "complete", "utc": utc(), "verified_fits": len(receipts),
                        "outputs_sha256": {name: sha256(output / name) for name in filenames}})
    print(result["interaction"], flush=True)


if __name__ == "__main__":
    main()
