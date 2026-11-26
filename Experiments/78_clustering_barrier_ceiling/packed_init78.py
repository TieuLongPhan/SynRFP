"""Exact k-means++ distance arithmetic for weighted binary fingerprint rows."""
import ctypes
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans


@lru_cache(maxsize=1)
def kernel(library):
    call = ctypes.CDLL(str(library)).squared_distances
    call.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t,
                     ctypes.c_float, ctypes.c_void_p]
    call.restype = None
    return call


def squared_distances(rows, candidate_ids, weight_sq, output, call):
    candidates = rows[candidate_ids]
    call(rows.ctypes.data, len(rows), candidates.ctypes.data, len(candidates),
         weight_sq, output.ctypes.data)


def seed_indices(packed, n_clusters, weight_sq, weights, rng, library):
    """Use the native greedy k-means++ sampling and potential reductions."""
    call = kernel(Path(library))
    rows = np.ascontiguousarray(packed).view(np.uint64)
    chosen = np.empty(n_clusters, dtype=np.int64)
    chosen[0] = rng.choice(len(rows), p=weights / weights.sum())
    closest = np.empty((1, len(rows)), dtype=np.float32)
    squared_distances(rows, chosen[:1], weight_sq, closest, call)
    potential = closest @ weights
    n_trials = 2 + int(np.log(n_clusters))
    distances = np.empty((n_trials, len(rows)), dtype=np.float32)
    for j in range(1, n_clusters):
        draw = rng.uniform(size=n_trials) * potential
        candidates = np.searchsorted(np.cumsum(weights * closest), draw)
        np.clip(candidates, None, len(rows) - 1, out=candidates)
        squared_distances(rows, candidates, weight_sq, distances, call)
        np.minimum(closest, distances, out=distances)
        totals = distances @ weights.reshape(-1, 1)
        best = np.argmin(totals)
        potential = totals[best]
        closest = distances[best].copy()
        chosen[j] = candidates[best]
    return chosen


class PackedInitializationKMeans(MiniBatchKMeans):
    """Native dense mini-batch optimizer with exact packed distances for initialization."""
    def _init_centroids(self, X, x_squared_norms, init, random_state, sample_weight,
                        init_size=None, n_centroids=None):
        if init != "k-means++" or X.shape != (len(self.packed_rows), 2048):
            raise ValueError("Packed initializer requires the frozen binary fingerprint recipe")
        k = self.n_clusters if n_centroids is None else n_centroids
        rows = np.arange(len(X))
        if init_size is not None and init_size < len(X):
            rows = random_state.randint(0, len(X), size=init_size)
        indices = seed_indices(self.packed_rows[rows], k, self.pooled_weight ** 2,
                               sample_weight[rows], random_state, self.kernel_path)
        return X[rows[indices]]
