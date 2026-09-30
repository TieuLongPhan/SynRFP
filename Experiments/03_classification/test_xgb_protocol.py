"""Regression checks for restored XGBoost coverage and validation handling."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from synrfp_exp.features import Features
from synrfp_exp.models import Skip, fit_predict


def test_xgb_covers_more_than_200_classes_and_restores_original_labels():
    rng = np.random.RandomState(7)
    n_classes = 201
    X = rng.normal(size=(4 * n_classes, 8)).astype(np.float32)
    feats = Features(X, 0, np.ones(len(X), dtype=bool))
    y = np.tile(np.arange(n_classes) + 1000, 4)
    tr, va, te = np.arange(402), np.arange(402, 603), np.arange(603, 804)
    params = dict(n_estimators=2, learning_rate=0.1, max_depth=2,
                  early_stopping_rounds=2, max_classes=None)
    pred = fit_predict('xgb', params, feats, tr, y[tr], va, y[va], te, seed=0, n_jobs=2)
    assert pred.shape == (len(te),)
    assert np.isin(pred, y[tr]).all()
    with pytest.raises(Skip, match='max_classes'):
        fit_predict('xgb', dict(params, max_classes=200), feats,
                    tr, y[tr], va, y[va], te, seed=0, n_jobs=2)
    with pytest.raises(Skip, match='no validation labels'):
        fit_predict('xgb', params, feats, tr, y[tr], va, y[va] + 10000,
                    te, seed=0, n_jobs=2)
