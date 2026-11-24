"""Paired, group-weighted regression diagnostics and uncertainty."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupKFold


def signed_key(left, right):
    counts = right.copy()
    counts.subtract(left)
    return tuple(sorted((token, count) for token, count in counts.items() if count))


def means_by_group(values, group_ids):
    groups, inverse = np.unique(group_ids, return_inverse=True)
    sums = np.zeros((len(groups), *values.shape[1:]), dtype=np.float64)
    np.add.at(sums, inverse, values)
    sizes = np.bincount(inverse).reshape((len(groups),) + (1,) * (values.ndim - 1))
    return sums / sizes


@dataclass
class Diagnostic:
    eligible: np.ndarray
    delta_prediction: np.ndarray
    context_prediction: np.ndarray
    fitting_variance: np.ndarray
    contributors: int
    query_groups: int
    classes: int
    boundary_ties: int

    def summary(self, actual, groups):
        if self.contributors:
            index = self.eligible
            reduction = ((self.delta_prediction[index] - actual[index]) ** 2
                         - (self.context_prediction[index] - actual[index]) ** 2)
            score = means_by_group(reduction / self.fitting_variance[index], groups[index]).mean(axis=0)
            score = score.tolist()
        else:
            score = None
        rate = self.contributors / len(actual)
        enough = rate >= 0.05 and self.query_groups >= 30 and self.classes >= 10
        return {
            "eligible_queries": self.contributors, "coverage": rate,
            "query_groups": self.query_groups, "classes": self.classes,
            "third_neighbor_boundary_ties": self.boundary_ties, "diagnostic": score,
            "coverage_passes": enough,
            "higher_gain_target_index": int(np.argmax(score)) if enough and score[0] != score[1] else None,
        }


def regression_diagnostic(bits, keys, groups, targets, *, permutation_seed=None):
    """Inputs must already be restricted to the one common training partition."""
    if bits.ndim != 2 or len(bits) != len(groups) or len(keys) != len(groups):
        raise ValueError("Diagnostic rows are not aligned")
    if targets.shape != (len(groups), 2) or not np.isfinite(targets).all():
        raise ValueError("Expected two finite targets on the same rows")

    by_key = defaultdict(list)
    for index, key in enumerate(keys):
        by_key[key].append(index)

    fold = np.full(len(groups), -1)
    variance = np.empty_like(targets, dtype=np.float64)
    for number, (train, heldout) in enumerate(GroupKFold(5).split(bits, groups=groups)):
        fold[heldout] = number
        variance[heldout] = targets[train].var(axis=0)
    if np.any(fold < 0) or np.any(variance <= 0):
        raise ValueError("Invalid training-fold coverage or target variance")

    packed = np.packbits(bits, axis=1)
    rng = np.random.default_rng(permutation_seed)
    eligible = np.zeros(len(groups), dtype=bool)
    delta = np.full_like(variance, np.nan)
    context = np.full_like(variance, np.nan)
    class_count, ties = 0, 0
    for indices in by_key.values():
        if len(indices) < 4:
            continue
        indices = np.array(indices)
        local_groups = groups[indices]
        group_names, inverse = np.unique(local_groups, return_inverse=True)
        if len(group_names) < 4:
            continue
        local_fold = fold[indices]
        local_bits = packed[indices]
        if permutation_seed is not None:
            shuffled = local_bits.copy()
            for number in range(5):
                members = np.flatnonzero(local_fold == number)
                shuffled[members] = local_bits[rng.permutation(members)]
            local_bits = shuffled
        used = False
        local_targets = targets[indices]
        for number in np.unique(local_fold):
            query_positions = np.flatnonzero(local_fold == number)
            ref_positions = np.flatnonzero(local_fold != number)
            ref_groups, ref_inverse = np.unique(inverse[ref_positions], return_inverse=True)
            if len(ref_groups) < 3:
                continue
            ref_bits = local_bits[ref_positions]
            ref_targets = local_targets[ref_positions]
            g = len(ref_groups)
            group_means = means_by_group(ref_targets, ref_inverse)
            delta_prediction = group_means.mean(axis=0)
            for query in query_positions:
                vector = local_bits[query]
                if np.all(ref_bits == vector):
                    continue
                overlap = np.bitwise_count(ref_bits & vector).sum(axis=1)
                union = np.bitwise_count(ref_bits | vector).sum(axis=1)
                sim = np.divide(overlap, union, out=np.ones(len(union)), where=union > 0)
                closest = np.full(g, -np.inf)
                np.maximum.at(closest, ref_inverse, sim)
                cutoff = np.partition(closest, -3)[-3]
                above = closest > cutoff
                boundary = closest == cutoff
                remaining = 3 - int(above.sum())
                ties += int(boundary.sum() > remaining)
                weights = above.astype(float) + boundary * (remaining / int(boundary.sum()))
                tied_refs = sim == closest[ref_inverse]
                selected_means = means_by_group(ref_targets[tied_refs], ref_inverse[tied_refs])
                row = indices[query]
                delta[row] = delta_prediction
                context[row] = (selected_means * weights[:, None]).sum(axis=0) / 3
                eligible[row] = used = True
        class_count += int(used)
    return Diagnostic(eligible, delta, context, variance, int(eligible.sum()),
                      len(np.unique(groups[eligible])), class_count, ties)


def paired_bootstrap(group_gains, *, draws=10000, seed=2026):
    """Resample the same groups for every target and arm in each draw."""
    if group_gains.ndim != 2 or not len(group_gains) or not np.isfinite(group_gains).all():
        raise ValueError("Expected a finite matrix of held-out group gains")
    rng = np.random.default_rng(seed)
    samples = np.empty((draws, group_gains.shape[1]), dtype=np.float64)
    for offset in range(0, draws, 128):
        count = min(128, draws - offset)
        indices = rng.integers(len(group_gains), size=(count, len(group_gains)))
        samples[offset:offset + count] = group_gains[indices].mean(axis=1)
    return samples
