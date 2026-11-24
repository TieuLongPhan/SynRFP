"""Exercise the multiclass runner using the pinned ECReact development source."""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from audit_inputs import sha256
from enzymemap_cohort import canonical_sides, DELIMITER, reaction_smiles, reverse_key
from enzymemap_overlap import RAW_ECREACT_SHA
from prepare_qp_features import encode
from run_enzymemap import ARMS, GRID, NAMES, fit_cell, score, utc, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.source) != RAW_ECREACT_SHA:
        raise ValueError("Expected the pinned ECReact development source")
    rows, groups, seen = {"train": [], "test": []}, {"train": [], "test": []}, set()
    quotas = {"train": 420, "test": 100}
    numbers = {"train": 0, "test": 0}
    with gzip.open(args.source, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            numbers[split] += 1
            if len(rows[split]) >= quotas[split] or numbers[split] % 31 or not row["ec2"]:
                continue
            left, _, right = DELIMITER.split(row["rxn"])
            sides = canonical_sides(left, right)
            key = reverse_key(sides)
            if key in seen:
                continue
            _, _, vectors, _, _, refusal = encode((0, row["r_id"], reaction_smiles(sides)))
            if refusal:
                continue
            groups[split].append(len(seen))
            seen.add(key)
            rows[split].append((row["ec2"], vectors["d2048"]))
            if all(len(rows[k]) == quotas[k] for k in quotas):
                break
    if any(len(rows[k]) != quotas[k] for k in quotas):
        raise ValueError("Development panel is incomplete")
    labels, packed = zip(*rows["train"], *rows["test"], strict=True)
    labels, packed = np.asarray(labels), np.asarray(packed)
    n_train = len(rows["train"])
    fit, val = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=17).split(
        packed[:n_train], groups=groups["train"]
    ))
    test = np.arange(n_train, len(labels))
    vocab = np.unique(labels[fit])
    if len(vocab) < 3:
        raise ValueError("Smoke panel lacks multiclass support")
    targets = {label: i for i, label in enumerate(vocab)}
    train = np.asarray([targets[label] for label in labels[fit]])
    validation = np.asarray([targets.get(label, -1) for label in labels[val]])
    validation[0] = -1
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifact = fit_cell(packed, (fit, val, test), train, validation, np.arange(len(vocab)),
                        "d2048", "ec_level_2", 17, output, RAW_ECREACT_SHA)
    with np.load(output / "ec_level_2__d2048__s17/blind_predictions.npz", allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["matrix_row"], test)
        blind = saved["prediction"]
    # Deliberately give every scoring arm identical predictions as a zero-effect control.
    predicted = {key: blind for key in GRID}
    observed = labels[test].astype("U100")
    observed[0] = "development-unseen-class"
    scores = score(predicted, np.column_stack((observed, observed)),
                   np.column_stack((labels[fit], labels[fit])),
                   [vocab, vocab], np.asarray(groups["test"]), NAMES[0], output)
    if scores["interaction"]["gain_difference"] != 0 or any(r["gain"] for r in scores["estimates"]):
        raise ValueError("Identical predictions produced a nonzero gain")
    if any(scores["baseline_and_coverage"][name]["unseen_test_label_rows"] < 1 for name in NAMES):
        raise ValueError("Unseen-class control was not counted")
    if scores["rows"] != len(test) or set(ARMS) != {key[1] for key in GRID}:
        raise ValueError("Development scoring shape changed")
    runner = Path(__file__).with_name("run_enzymemap.py")
    write(output / "COMPLETE.json", {
        "status": "passed", "utc": utc(), "source_sha256": RAW_ECREACT_SHA,
        "runner_sha256": sha256(runner), "smoke_sha256": sha256(Path(__file__)),
        "rows": len(labels), "training_classes": len(vocab),
        "fit_epochs": artifact["epochs"], "validation_unseen_rows": int((validation < 0).sum()),
        "identical_prediction_gain": 0.0, "unseen_test_class_control": True,
    })


if __name__ == "__main__":
    main()
