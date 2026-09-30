"""Clustering algorithms.

* ``kmeans``: k fixed to the number of reference labels (original protocol).
* ``bitbirch``: BitBIRCH (bblean) on packed bit vectors, controlled by a
  Tanimoto threshold instead of k. Bit fingerprints only.
* ``hdbscan``: density-based, k-free, on PCA-reduced features; applicable to
  every fingerprint including real-valued RXNFP. Noise points are kept as
  singletons so that they count against homogeneity and completeness alike.
"""
from __future__ import annotations

import numpy as np

from .features import Features


def kmeans(feats: Features, k: int, seed: int, n_init: int = 10) -> np.ndarray:
    from sklearn.cluster import KMeans

    X = feats.dense()
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit_predict(X)


def minibatch_kmeans(
    feats: Features, k: int, seed: int, batch_size: int = 8192,
    n_init: int = 3, max_iter: int = 100,
) -> np.ndarray:
    """Scalable K-means used for the full TPL and ECReact comparisons."""
    from sklearn.cluster import MiniBatchKMeans

    X = feats.dense()
    return MiniBatchKMeans(
        n_clusters=k, random_state=seed, batch_size=batch_size,
        n_init=n_init, max_iter=max_iter, reassignment_ratio=0.01,
    ).fit_predict(X)


def bitbirch(
    feats: Features, threshold: float, branching_factor: int = 50,
    merge_criterion: str = "diameter",
) -> np.ndarray:
    import bblean

    if not feats.is_bits:
        raise TypeError("BitBIRCH requires bit fingerprints")
    tree = bblean.BitBirch(threshold=threshold, branching_factor=branching_factor,
                           merge_criterion=merge_criterion)
    tree.fit(feats.packed())
    labels = np.full(len(feats.X), -1, dtype=np.int64)
    for cid, members in enumerate(tree.get_cluster_mol_ids()):
        labels[np.asarray(members, dtype=np.int64)] = cid
    if (labels < 0).any():
        raise RuntimeError("BitBIRCH left points unassigned")
    return labels


def hdbscan(feats: Features, min_cluster_size: int, pca_dim: int, seed: int):
    """Returns ``(labels, noise_fraction)``; noise points become singletons."""
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA

    X = feats.dense()
    if X.shape[1] > pca_dim:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)
    lab = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(X)
    noise = lab < 0
    out = lab.astype(np.int64).copy()
    out[noise] = lab.max() + 1 + np.arange(int(noise.sum()))
    return out, float(noise.mean())
