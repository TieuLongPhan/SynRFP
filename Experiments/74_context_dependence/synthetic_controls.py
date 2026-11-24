"""Run the prespecified balanced synthetic context probe on abstract graphs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from itertools import product
import json
from pathlib import Path
import sys

from joblib import Parallel, delayed
import numpy as np
from rdkit import Chem
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits

from audit_inputs import sha256

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "73_context_attribution"))
import features
import run_ablation as e1
from synrfp.graph.molecule import Molecule

LEVELS = (0, 0.25, 0.5, 0.75, 1)
N = 32
ARMS = (*features.ARMS, "shuffled_pooled")
FIT_SEED = 17


def node(element):
    return {"element": element, "aromatic": False, "charge": 0, "hcount": 0}


def graph(motif, context, order, atom_ids=None):
    ids = list(range(5)) if atom_ids is None else atom_ids
    atoms = dict(zip(ids, map(node, ("C", motif, "N", "O", context)), strict=True))
    bonds = {
        (ids[0], ids[1]): {"order": order},
        (ids[0], ids[2]): {"order": 1},
        (ids[3], ids[4]): {"order": 1},
    }
    return Molecule.from_dicts(atoms, bonds)


def token_counts(motif, context, order, atom_ids=None):
    return features.TOKENIZER.tokens_graph(graph(motif, context, order, atom_ids), 2)


def signed_counts(left, right):
    changes = right.copy()
    changes.subtract(left)
    return tuple(sorted((token, count) for token, count in changes.items() if count))


def design(seed):
    rng = np.random.default_rng(seed)
    elements = [Chem.GetPeriodicTable().GetElementSymbol(i) for i in range(12, 108)]
    order = rng.permutation(elements)
    motifs, contexts = order[:N], order[N:2*N]
    motif_index, context_index = np.indices((N, N))
    motif_index, context_index = motif_index.ravel(), context_index.ravel()
    residue = (rng.permutation(N)[motif_index] + rng.permutation(N)[context_index]) % 8
    fit, validation, test = (np.flatnonzero(mask) for mask in
                             (residue < 6, residue == 6, residue == 7))
    partition = e1.Partition("balanced_grid", "held_out_combinations", fit, validation, test)

    # Each factor has the same four possible values in every residue class.
    # Within each split, every pair of attribute values appears equally often.
    def attributes(index):
        classes = np.empty(N, dtype=np.int64)
        for values in np.unique(index % 8):
            classes[index % 8 == values] = rng.permutation(4)
        return np.asarray([-1.5, -0.5, 0.5, 1.5])[classes]

    # Recover factor residues from each fixed row and column of the grid.
    change = attributes(residue.reshape(N, N)[:, 0])[motif_index]
    context = attributes(residue.reshape(N, N)[0, :])[context_index]
    for rows in (fit, validation, test):
        expected = len(rows) // 16
        histogram = Counter(zip(change[rows], context[rows]))
        if len(histogram) != 16 or set(histogram.values()) != {expected}:
            raise ValueError("Imbalanced attribute combinations")
        for factor in (motif_index, context_index):
            support = np.bincount(factor[rows], minlength=N)
            if not np.all(support == len(rows) // N):
                raise ValueError("A factor is underrepresented in a partition")

    matrices = {arm: [] for arm in features.ARMS}
    motif_deltas = {}
    for i, j in zip(motif_index, context_index, strict=True):
        left = token_counts(motifs[i], contexts[j], 1)
        right = token_counts(motifs[i], contexts[j], 2)
        if j == 0:
            remapped = rng.permutation(1000)[:5].tolist()
            if left != token_counts(motifs[i], contexts[j], 1, remapped):
                raise ValueError("WL tokens changed under atom renumbering")
            if right != token_counts(motifs[i], contexts[j], 2, remapped):
                raise ValueError("Product tokens changed under atom renumbering")
        delta = signed_counts(left, right)
        if motif_deltas.setdefault(i, delta) != delta:
            raise ValueError("A shared context changes an exact signed delta")
        for arm, vector in features.fingerprints(left, right).items():
            matrices[arm].append(vector)
    if len(set(motif_deltas.values())) != N:
        raise ValueError("Transformation motifs do not have distinct signed counts")
    arrays = {arm: np.stack(vectors) for arm, vectors in matrices.items()}
    np.testing.assert_array_equal(arrays["d1024"], arrays["d2048"][:, :1024] ^ arrays["d2048"][:, 1024:])
    np.testing.assert_array_equal(arrays["d_rehash"][:, :1024], arrays["d1024"])
    pooled = arrays["d_pooled"]
    np.testing.assert_array_equal(pooled[:, :1024], arrays["d1024"])
    shuffled = pooled.copy()
    pooled_distinct = []
    for motif in range(N):
        index = np.flatnonzero(motif_index == motif)
        pooled_distinct.append(len(np.unique(pooled[index, 1024:], axis=0)))
        for arm in ("d1024", "d2048", "d_rehash"):
            if len(np.unique(arrays[arm][index], axis=0)) != 1:
                raise ValueError(f"{arm} leaks context identity")
        for rows in (fit, validation, test):
            selected = np.intersect1d(rows, index, assume_unique=True)
            shuffled[selected, 1024:] = pooled[rng.permutation(selected), 1024:]
    arrays["shuffled_pooled"] = shuffled
    return arrays, partition, change, context, pooled_distinct


def classification(x, target, part):
    model = MLPClassifier(**e1.FIT_OPTIONS, random_state=np.random.RandomState(FIT_SEED))
    best_loss, patience_loss, stale = np.inf, np.inf, 0
    x_fit, x_val, x_test = (x[rows].astype(np.float32)
                           for rows in (part.fit, part.validation, part.test))
    y_fit, y_val = target[part.fit], target[part.validation]
    classes = np.array([0, 1])
    with threadpool_limits(limits=1):
        for epoch in range(1, e1.MAX_EPOCHS + 1):
            model.partial_fit(x_fit, y_fit, classes=classes)
            loss = log_loss(y_val, model.predict_proba(x_val), labels=classes)
            if not np.isfinite(loss):
                raise ValueError("Nonfinite classification validation loss")
            if loss < best_loss:
                best_loss, best_epoch = loss, epoch
                best = ([value.copy() for value in model.coefs_],
                        [value.copy() for value in model.intercepts_])
            if patience_loss - loss >= e1.TOLERANCE:
                patience_loss, stale = loss, 0
            else:
                stale += 1
            if stale >= e1.PATIENCE:
                break
        model.coefs_, model.intercepts_ = best
        prediction = model.predict(x_test)
    return prediction, epoch, best_epoch


def fit(seed, level, task, arm, vectors, part, change, context):
    train = part.fit
    change = (change - change[train].mean()) / change[train].std()
    context = (context - context[train].mean()) / context[train].std()
    signal = np.sqrt(1 - level) * change + np.sqrt(level) * context
    # Symmetric discrete attributes can cancel. Resolve exact and floating-point ties as class 0.
    target = signal if task == "regression" else (signal > 1e-12).astype(np.int64)
    if task == "regression":
        e1.MODEL_SEED = FIT_SEED
        data = {f"S_{arm}": vectors, "y": target, "H": np.empty((len(target), 0))}
        metrics, rows, prediction = e1.fit_cell("synthetic", data, part, "S", False, arm)
        np.testing.assert_array_equal(rows, part.test)
        error = (prediction - target[rows]) ** 2
        loss = float(error.mean() / target[train].var())
        epochs, best_epoch = metrics["epochs"], metrics["best_epoch"]
    else:
        prediction, epochs, best_epoch = classification(vectors, target, part)
        loss = float(np.mean(prediction != target[part.test]))
    return {"grid_seed": seed, "a": level, "task": task, "arm": arm,
            "loss": loss, "epochs": epochs, "best_epoch": best_epoch}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results, grids = [], []
    for seed in (2026, 2027, 2028):
        arrays, part, change, context, distinct = design(seed)
        # The known generating attributes give the exact oracle response.
        standardized = [(value-value[part.fit].mean())/value[part.fit].std()
                        for value in (change, context)]
        change_z, context_z = standardized
        for level in LEVELS:
            oracle = np.column_stack(standardized) @ np.array([np.sqrt(1-level), np.sqrt(level)])
            target = np.sqrt(1-level)*change_z + np.sqrt(level)*context_z
            np.testing.assert_allclose(oracle, target, atol=1e-12)
            np.testing.assert_array_equal(oracle > 1e-12, target > 1e-12)
        grids.append({"seed": seed, "fit_rows": len(part.fit), "validation_rows": len(part.validation),
                      "test_rows": len(part.test), "distinct_pooled_vectors_by_motif": distinct})
        results.extend(Parallel(n_jobs=args.jobs, verbose=5)(
            delayed(fit)(seed, level, task, arm, arrays[arm], part, change, context)
            for level, task, arm in product(LEVELS, ("regression", "classification"), ARMS)
        ))
    assert len(results) == 150
    scores = args.output / "fit_scores.csv"
    with scores.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    lookup = {(row["grid_seed"], row["a"], row["task"], row["arm"]): row["loss"] for row in results}
    contrasts = []
    for seed, task in product((2026, 2027, 2028), ("regression", "classification")):
        def gain(level):
            return lookup[seed, level, task, "d2048"] - lookup[seed, level, task, "d_pooled"]
        contrasts.append({"grid_seed": seed, "task": task, "change_only_gain": gain(0),
                          "context_only_gain": gain(1), "endpoint_interaction": gain(1)-gain(0)})
    receipt = {"phase": "complete", "scope": "synthetic_probe", "fits": len(results),
               "fit_seed": FIT_SEED, "classification_tie_tolerance": 1e-12,
               "grids": grids, "contrasts": contrasts, "fit_scores_sha256": sha256(scores),
               "runner_sha256": sha256(Path(__file__))}
    (args.output / "COMPLETE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(contrasts, indent=2))


if __name__ == "__main__":
    main()
