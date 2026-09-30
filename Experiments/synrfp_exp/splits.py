"""Train/validation/test splits.

Split codes: 0 = train, 1 = validation, 2 = test. Two schemes only, never
mixed in one table or test (exp.md §3):

* ``cv``    Track A: grouped 5x5 repeated CV. ``seed`` encodes
  ``repeat * 5 + fold`` (repeat, fold in 0..4). Each repeat computes a fresh
  StratifiedGroupKFold (product scaffold as group, so no scaffold is shared
  between train and test); fold ``f`` -> test, fold ``f + 1`` -> validation,
  the rest -> train.
* ``fixed`` Track B: one split per dataset -- the SynRXN published column, or
  a deterministic stratified 8:1:1 split for a dataset without one. ``seed``
  does not affect the partition, only which
  model-initialisation seed the caller is on.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import (
    ShuffleSplit,
    StratifiedGroupKFold,
    StratifiedShuffleSplit,
)

from .config import log
from .data import dataset_table, reaction_table, rxn_key, syncat_syntemp_split

TRAIN, VAL, TEST = 0, 1, 2


def product_group(rsmi: str) -> str:
    """Murcko scaffold of the largest product; its canonical SMILES if acyclic."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    prods = rsmi.split(">")[-1].split(".")
    best, best_n = None, -1
    for smi in prods:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and mol.GetNumHeavyAtoms() > best_n:
            best, best_n = mol, mol.GetNumHeavyAtoms()
    if best is None:
        return "<unparsable>"
    scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=best)
    return scaf if scaf else "acyclic:" + Chem.MolToSmiles(best)


def groups_for(cfg: Dict[str, Any], dataset: str) -> np.ndarray:
    key = rxn_key(cfg, dataset)
    cache = cfg["paths"]["work"] / "groups" / f"{key}.npy"
    if not cache.exists():
        rt = reaction_table(cfg, key)
        log.info("computing product scaffolds for %s (%d)", key, len(rt))
        g = Parallel(n_jobs=cfg["n_jobs"], batch_size=512)(
            delayed(product_group)(s) for s in rt["rxn"]
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, np.asarray(g, dtype=object), allow_pickle=True)
    rows = dataset_table(cfg, dataset)["row"].to_numpy()
    return np.load(cache, allow_pickle=True)[rows]


def _holdout(idx: np.ndarray, y: np.ndarray, frac: float, seed: int):
    """Split ``idx`` into (keep, held) with stratification when possible."""
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
        keep, held = next(sss.split(idx, y[idx]))
    except ValueError:
        ss = ShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
        keep, held = next(ss.split(idx))
    return idx[keep], idx[held]


def random_codes(cfg: Dict[str, Any], y: np.ndarray, seed: int) -> np.ndarray:
    """Stratified train/val/test codes for labels ``y`` (ratio from config)."""
    tr, va, te = cfg["splits"]["random_ratio"]
    idx = np.arange(len(y))
    code = np.full(len(y), TRAIN, dtype=np.int8)
    rest, test = _holdout(idx, y, te, seed)
    _, val = _holdout(rest, y, va / (tr + va), seed)
    code[test], code[val] = TEST, VAL
    return code


def cv_split(cfg: Dict[str, Any], dataset: str, ds: pd.DataFrame, y: np.ndarray, seed: int) -> np.ndarray:
    """Track A: one (repeat, fold) split of the grouped 5x5 repeated CV."""
    n_folds = 5
    repeat, fold = divmod(seed, n_folds)
    idx = np.arange(len(ds))
    g = groups_for(cfg, dataset)
    folds = list(
        StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=repeat).split(
            idx, y, g
        )
    )
    code = np.full(len(ds), TRAIN, dtype=np.int8)
    code[folds[fold][1]] = TEST
    code[folds[(fold + 1) % n_folds][1]] = VAL
    return code


def fixed_split(cfg: Dict[str, Any], dataset: str, ds: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """
    Track B: one fixed split, source depending on the dataset.

    :raises ValueError: If a dataset has neither a published split nor a
        recognised fixed-split source.
    """
    spec = cfg["datasets"][dataset]
    if rxn_key(cfg, dataset) == "mech" or dataset.startswith("syntemp_r"):
        return random_codes(cfg, y, seed=cfg["splits"].get("fixed_random_seed", 42))

    idx = np.arange(len(ds))
    if spec["split"] == "published":
        if "split_published" not in ds:
            raise ValueError(f"{dataset} has no published split")
        test = ds["split_published"].str.lower().isin(["test"]).to_numpy()
    else:
        raise ValueError(f"{dataset}: no Track B fixed-split source configured")

    code = np.full(len(ds), TRAIN, dtype=np.int8)
    code[test] = TEST
    _, val = _holdout(idx[~test], y, cfg["splits"]["val_fraction"], seed=0)
    code[val] = VAL
    return code


def make_split(cfg: Dict[str, Any], dataset: str, scheme: str, seed: int) -> np.ndarray:
    """Split codes for every row of the dataset table (cached)."""
    cache = cfg["paths"]["work"] / "splits" / dataset / f"{scheme}_s{seed}.npy"
    if cache.exists():
        return np.load(cache)
    ds = dataset_table(cfg, dataset)
    y = ds["y"].to_numpy()

    if scheme == "cv":
        code = cv_split(cfg, dataset, ds, y, seed)
    elif scheme == "fixed":
        code = fixed_split(cfg, dataset, ds, y)
    else:
        raise ValueError(f"unknown split scheme {scheme!r}")

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, code)
    return code


def scheme_seeds(
    cfg: Dict[str, Any],
    dataset: str,
    schemes: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
) -> List[Tuple[str, int]]:
    """(scheme, seed) pairs of the benchmark; both schemes apply to every dataset."""
    table = cfg["benchmark"]["schemes"]
    out = []
    for scheme in schemes or list(table):
        out += [(scheme, s) for s in (seeds if seeds is not None else table[scheme])]
    return out


def subsample_rows(y: np.ndarray, max_records: int, seed: int = 0) -> np.ndarray:
    """Stratified subsample of row positions (sorted) when ``len(y) > max_records``."""
    idx = np.arange(len(y))
    if len(y) <= max_records:
        return idx
    keep, _ = _holdout(idx, y, 1 - max_records / len(y), seed)
    return np.sort(keep)


def split_summary(code: np.ndarray, y: np.ndarray) -> Dict[str, int]:
    s = pd.Series(code)
    return {
        "n_train": int((s == TRAIN).sum()),
        "n_val": int((s == VAL).sum()),
        "n_test": int((s == TEST).sum()),
        "test_classes_unseen_in_train": int(
            len(set(y[code == TEST]) - set(y[code == TRAIN]))
        ),
    }
