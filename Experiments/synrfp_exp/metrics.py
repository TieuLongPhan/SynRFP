"""Evaluation metrics for classification, clustering and retrieval."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn import metrics as skm


def classification(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mcc": skm.matthews_corrcoef(y_true, y_pred),
        "accuracy": skm.accuracy_score(y_true, y_pred),
        "macro_f1": skm.f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": skm.f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "balanced_accuracy": skm.balanced_accuracy_score(y_true, y_pred),
    }


def hungarian_mcc(y_true: np.ndarray, clusters: np.ndarray) -> float:
    """
    MCC after the best one-to-one cluster-to-class assignment.

    Clusters left unassigned (more clusters than classes) map to a sentinel
    label and therefore count as errors, so over-splitting is penalised.
    """
    classes, yt = np.unique(y_true, return_inverse=True)
    cl, zc = np.unique(clusters, return_inverse=True)
    table = np.zeros((len(cl), len(classes)), dtype=np.int64)
    np.add.at(table, (zc, yt), 1)
    r, c = linear_sum_assignment(-table)
    lut = np.full(len(cl), -1)
    lut[r] = c
    return float(skm.matthews_corrcoef(yt, lut[zc]))


def clustering(y_true: np.ndarray, clusters: np.ndarray) -> Dict[str, float]:
    """Agreement of a partition with reference labels (all permutation invariant
    except ``mcc_hungarian``)."""
    h, c, v = skm.homogeneity_completeness_v_measure(y_true, clusters)
    sizes = np.bincount(np.unique(clusters, return_inverse=True)[1])
    return {
        "ari": skm.adjusted_rand_score(y_true, clusters),
        "ami": skm.adjusted_mutual_info_score(y_true, clusters),
        "nmi": skm.normalized_mutual_info_score(y_true, clusters),
        "homogeneity": h,
        "completeness": c,
        "v_measure": v,
        "mcc_hungarian": hungarian_mcc(y_true, clusters),
        "n_clusters": int(sizes.size),
        "singleton_fraction": float((sizes == 1).sum() / len(clusters)),
        "largest_cluster_fraction": float(sizes.max() / len(clusters)),
    }


def retrieval(y_query: np.ndarray, y_neighbors: np.ndarray, k: int) -> Dict[str, float]:
    """Precision@k and mean average precision@k of label matches."""
    hit = (y_neighbors[:, :k] == y_query[:, None]).astype(float)
    prec_at = np.cumsum(hit, axis=1) / np.arange(1, k + 1)
    n_rel = hit.sum(1)
    ap = np.divide((prec_at * hit).sum(1), n_rel, out=np.zeros(len(hit)), where=n_rel > 0)
    return {f"p@{k}": float(hit.mean()), f"map@{k}": float(ap.mean()),
            f"hit@{k}": float((n_rel > 0).mean())}
