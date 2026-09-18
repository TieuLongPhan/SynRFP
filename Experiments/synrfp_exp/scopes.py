"""Which reactions a study featurises, and how feature rows align with a dataset table.

A *scope* names a feature cache folder. The full scope of a dataset is its
source reaction table (shared by e.g. ``ecreact_1/2/3``); the factorial study
uses a stratified subsample of large datasets, cached under its own scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from .data import dataset_table, reaction_table, rxn_key
from .splits import subsample_rows


@dataclass
class Scope:
    name: str             # feature cache folder
    rsmis: List[str]      # reactions in feature-row order (used to compute features)
    ds_pos: np.ndarray    # positions in the dataset table covered by this study
    feat_rows: np.ndarray  # feature-matrix rows aligned with ``ds_pos``


def full_scope(cfg: Dict[str, Any], dataset: str) -> Scope:
    key = rxn_key(cfg, dataset)
    ds = dataset_table(cfg, dataset)
    rt = reaction_table(cfg, key)
    return Scope(key, rt["rxn"].tolist(), np.arange(len(ds)), ds["row"].to_numpy())


def factorial_scope(cfg: Dict[str, Any], dataset: str) -> Scope:
    ds = dataset_table(cfg, dataset)
    cap = cfg["factorial"]["max_records"]
    pos = subsample_rows(ds["y"].to_numpy(), cap, seed=0)
    if len(pos) == len(ds):
        return full_scope(cfg, dataset)
    return Scope(f"{dataset}__sub{cap}", ds["rxn"].to_numpy()[pos].tolist(), pos,
                 np.arange(len(pos)))
