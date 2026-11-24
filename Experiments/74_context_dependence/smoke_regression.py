"""Exercise the five fingerprint arms and two regression targets on development rows."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from audit_inputs import PINS, TARGETS, sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "73_context_attribution"))
import features
import run_ablation as e1
from synrfp.encode.core import fingerprint_from_tokens

ARMS = ("d1024", "d2048", "d_pooled", "d_rehash", "reactant_context")


def sides(reaction):
    left, _, right = reaction.split(">")
    return tuple(tuple(sorted(features.canonical(part) for part in side.split(".")))
                 for side in (left, right))


def encode(left, right):
    left_counts, right_counts = features.side_counts(left), features.side_counts(right)
    vectors = features.fingerprints(left_counts, right_counts)
    reactant = fingerprint_from_tokens(
        left_counts, Counter(), mode="union", bits=1024, sketch="parity", seed=1
    )
    vectors["reactant_context"] = np.concatenate((vectors["d1024"], reactant))
    np.testing.assert_array_equal(vectors["d1024"], vectors["d2048"][:1024] ^ vectors["d2048"][1024:])
    for arm in ("d_pooled", "d_rehash", "reactant_context"):
        np.testing.assert_array_equal(vectors[arm][:1024], vectors["d1024"])
    return vectors


def dataset(name, path, row_limit):
    if sha256(path) != PINS[name][1]:
        raise ValueError(f"Unpinned development input: {name}")
    targets = TARGETS[name]
    identities, row_ids, groups, labels = set(), [], [], []
    vectors = {arm: [] for arm in ARMS}
    with gzip.open(path, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            left, right = sides(row["aam"])
            if (left, right) in identities:
                continue
            blocks = encode(left, right)
            values = [float(row[target]) for target in targets]
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite development target for {row['r_id']}")
            for arm, bits in blocks.items():
                vectors[arm].append(bits)
            identities.add((left, right))
            row_ids.append(row["r_id"])
            key = json.dumps(sorted((left, right)), separators=(",", ":"))
            groups.append(hashlib.sha256(key.encode()).hexdigest())
            labels.append(values)
            if len(row_ids) == row_limit:
                break
    if len(row_ids) != row_limit:
        raise ValueError(f"Insufficient development rows for {name}")
    return row_ids, np.array(groups), np.array(labels), {
        f"S_{arm}": np.stack(bits) for arm, bits in vectors.items()
    }


def partition(groups):
    rows = np.arange(len(groups))
    train_val, test = next(GroupShuffleSplit(
        n_splits=1, test_size=0.1, random_state=17
    ).split(rows, groups=groups))
    fit, validation = next(GroupShuffleSplit(
        n_splits=1, test_size=1 / 9, random_state=1337
    ).split(train_val, groups=groups[train_val]))
    result = e1.Partition("dev_smoke", "grouped", train_val[fit], train_val[validation], test)
    pieces = (result.fit, result.validation, result.test)
    np.testing.assert_array_equal(np.sort(np.concatenate(pieces)), rows)
    seen = set()
    for piece in pieces:
        observed = set(groups[piece])
        if seen.intersection(observed):
            raise ValueError("A direction-insensitive reaction group crosses splits")
        seen.update(observed)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.input_manifest.read_text())
    results = []
    e1.MODEL_SEED = 17
    for name in ("b97xd3", "cycloadd"):
        source = inputs["sources"][name]
        if source["status"] != "source_intake_verified":
            raise ValueError(f"Development intake incomplete for {name}")
        ids, groups, y, data = dataset(name, Path(source["path"]), 384)
        split = partition(groups)
        data["H"] = np.empty((len(ids), 0), dtype=np.uint8)
        held_out = [ids[i] for i in split.test]
        test_hash = hashlib.sha256("\n".join(held_out).encode()).hexdigest()
        print(name, len(ids), len(set(groups)), "direction-insensitive groups", flush=True)
        for index, target in enumerate(TARGETS[name]):
            data["y"] = y[:, index]
            for arm in ARMS:
                start = time.monotonic()
                metrics, rows, predictions = e1.fit_cell(name, data, split, "S", False, arm)
                np.testing.assert_array_equal(rows, split.test)
                observed = data["y"][rows]
                np.testing.assert_allclose(metrics["rmse"], np.mean((predictions - observed) ** 2) ** 0.5)
                if not np.isfinite(metrics["r2"]):
                    raise ValueError("Nonfinite development smoke score")
                results.append({**metrics, "target": target, "test_ids_sha256": test_hash,
                                "fit_seconds": time.monotonic() - start})
    assert len(results) == 20
    with args.output.open("x") as handle:
        json.dump({
            "phase": "complete", "scope": "development_smoke_only",
            "seed": e1.MODEL_SEED, "fit_count": len(results),
            "input_manifest_sha256": sha256(args.input_manifest), "fit_scores": results,
        }, handle, indent=2)
        handle.write("\n")
    print("Completed 20 development fits with paired test rows and RMSE replay.")


if __name__ == "__main__":
    main()
