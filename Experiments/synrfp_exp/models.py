"""Downstream estimators used to probe representation quality.

* ``knn``: k-nearest neighbours with Tanimoto similarity for bit vectors and
  cosine similarity for real-valued features (majority vote, ties broken by
  the nearest neighbour). Tests whether labels are expressed by local geometry.
* ``linear``: one-vs-rest SGD logistic regression (a linear probe).
* ``mlp``: one hidden layer, early stopping on an internal validation split.
* ``xgb``: histogram gradient boosted trees (``multi:softprob``), early
  stopping on the validation split; optional CUDA device and
  ``max_classes`` limit.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .features import Features


class Skip(Exception):
    """Raised when a model is not applicable to a dataset."""


def similarity_topk(
    Q: np.ndarray, D: np.ndarray, k: int, tanimoto: bool, block: int = 256
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Indices and similarities of the ``k`` most similar database rows per query.

    :param Q: Query matrix (float32, dense 0/1 for Tanimoto).
    :param D: Database matrix.
    :param k: Number of neighbours.
    :param tanimoto: Tanimoto (bits) if True, else cosine.
    :return: ``(idx, sim)`` arrays of shape ``(len(Q), k)``, most similar first.
    """
    Q = np.ascontiguousarray(Q, dtype=np.float32)
    D = np.ascontiguousarray(D, dtype=np.float32)
    if tanimoto:
        qn, dn = Q.sum(1), D.sum(1)
    else:
        Q = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
        D = D / np.maximum(np.linalg.norm(D, axis=1, keepdims=True), 1e-12)
    k = min(k, len(D))
    idx = np.empty((len(Q), k), dtype=np.int64)
    sim = np.empty((len(Q), k), dtype=np.float32)
    for s in range(0, len(Q), block):
        S = Q[s:s + block] @ D.T
        if tanimoto:
            union = qn[s:s + block, None] + dn[None, :] - S
            S = np.divide(S, union, out=np.zeros_like(S), where=union > 0)
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, axis=1)
        order = np.argsort(-ps, axis=1, kind="stable")
        idx[s:s + block] = np.take_along_axis(part, order, axis=1)
        sim[s:s + block] = np.take_along_axis(ps, order, axis=1)
    return idx, sim


def _vote(labels: np.ndarray) -> np.ndarray:
    """Majority label per row; ties go to the label of the nearest tied neighbour."""
    out = np.empty(len(labels), dtype=labels.dtype)
    for i, row in enumerate(labels):
        vals, first, counts = np.unique(row, return_index=True, return_counts=True)
        best = counts == counts.max()
        out[i] = vals[best][np.argmin(first[best])]
    return out


def fit_predict(
    model: str,
    params: Dict[str, Any],
    feats: Features,
    rows_tr: np.ndarray,
    y_tr: np.ndarray,
    rows_va: np.ndarray,
    y_va: np.ndarray,
    rows_te: np.ndarray,
    seed: int,
    n_jobs: int = 8,
) -> np.ndarray:
    """Train on ``rows_tr`` and return predictions for ``rows_te``."""
    Xtr, Xte = feats.dense(rows_tr), feats.dense(rows_te)
    if model == "knn":
        idx, _ = similarity_topk(Xte, Xtr, params["k"], tanimoto=feats.is_bits)
        return _vote(y_tr[idx])

    if not feats.is_bits:  # standardise real-valued features for gradient models
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd

    if model == "linear":
        from sklearn.linear_model import SGDClassifier

        clf = SGDClassifier(loss="log_loss", alpha=params["alpha"],
                            max_iter=params["max_iter"], tol=1e-4,
                            random_state=seed, n_jobs=n_jobs)
        clf.fit(Xtr, y_tr)
        return clf.predict(Xte)

    if model == "mlp":
        from sklearn.neural_network import MLPClassifier

        clf = MLPClassifier(hidden_layer_sizes=tuple(params["hidden"]),
                            alpha=params["alpha"], max_iter=params["max_iter"],
                            batch_size=params["batch_size"], early_stopping=True,
                            validation_fraction=0.1, n_iter_no_change=10,
                            random_state=seed)
        clf.fit(Xtr, y_tr)
        return clf.predict(Xte)

    if model == "xgb":
        from xgboost import XGBClassifier

        classes = np.unique(y_tr)
        if params.get("max_classes") is not None and len(classes) > params["max_classes"]:
            raise Skip(f"{len(classes)} classes > max_classes")
        enc = {c: i for i, c in enumerate(classes)}
        keep = np.isin(y_va, classes)
        if not keep.any():
            raise Skip("no validation labels occur in the training classes")
        clf = XGBClassifier(objective="multi:softprob", tree_method="hist", device=params.get("device", "cpu"),
                            n_estimators=params["n_estimators"],
                            learning_rate=params["learning_rate"],
                            max_depth=params["max_depth"],
                            early_stopping_rounds=params["early_stopping_rounds"],
                            eval_metric="mlogloss", random_state=seed, n_jobs=n_jobs)
        Xva = feats.dense(rows_va)[keep]
        if not feats.is_bits:
            Xva = (Xva - mu) / sd
        clf.fit(Xtr, np.vectorize(enc.get)(y_tr),
                eval_set=[(Xva, np.vectorize(enc.get)(y_va[keep]))], verbose=False)
        # Avoid constructing a CPU DMatrix for GPU-trained models.  XGBoost
        # otherwise warns about mismatched devices and performs a slow fallback
        # during prediction.  ``inplace_predict`` returns class probabilities
        # for ``multi:softprob``; reduce them to the same encoded class labels
        # returned by ``clf.predict``.
        if str(params.get("device", "cpu")).startswith("cuda"):
            # XGBoost's sklearn wrapper did not expose ``inplace_predict`` in
            # some 3.x releases.  Prefer the booster API when available and
            # fall back to the wrapper prediction method for compatibility.
            if hasattr(clf, "inplace_predict"):
                pred = clf.inplace_predict(Xte)
            elif hasattr(clf, "get_booster") and hasattr(clf.get_booster(), "inplace_predict"):
                pred = clf.get_booster().inplace_predict(Xte)
            else:
                pred = clf.predict(Xte)
            if pred.ndim == 2:
                pred = pred.argmax(axis=1)
            return classes[np.asarray(pred, dtype=int)]
        return classes[clf.predict(Xte).astype(int)]

    raise ValueError(f"unknown model {model!r}")


def model_params(cfg: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    return cfg["models"].get(model)
