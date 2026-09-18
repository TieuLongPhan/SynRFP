"""Dataset loading from SynRXN.

Two levels of tables are used:

* a *reaction table* per source file (``rxn_key``), holding every record once;
  tokens and fingerprints are cached against it, so ``ecreact_1/2/3`` share one
  featurization;
* a *dataset table* per benchmark, holding the rows kept after removing rare
  classes, the encoded target ``y`` and the hierarchy columns.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .config import log

_MAP_RE = re.compile(r":\d+\]")


def strip_atom_maps(rsmi: str) -> str:
    """Remove atom-map numbers from a reaction SMILES."""
    return _MAP_RE.sub("]", rsmi)


def rxn_key(cfg: Dict[str, Any], dataset: str) -> str:
    spec = cfg["datasets"][dataset]
    if spec.get("source") == "mech":
        return "mech"
    return Path(spec["file"]).name.split(".")[0]


def _load_mech(cfg: Dict[str, Any]) -> pd.DataFrame:
    raw = pd.read_csv(cfg["paths"]["mech_raw"])
    col = cfg["mech"]["reaction_column"]
    df = pd.DataFrame(
        {
            "r_id": [f"mech_{i}" for i in range(len(raw))],
            "rxn": raw[col].astype(str).map(strip_atom_maps),
            "rxn_mapped": raw[cfg["mech"]["mapped_column"]].astype(str),
            "level2": raw["mechanistic_class"].astype(str),
        }
    )
    map_path = Path(cfg["paths"]["mech_level1_map"])
    if map_path.exists():
        m = pd.read_csv(map_path, comment="#")
        lut = dict(zip(m["mechanistic_class"].astype(str), m["level1"].astype(str)))
        missing = sorted(set(df["level2"]) - set(lut))
        if missing:
            raise ValueError(f"{map_path} lacks level-1 labels for: {missing}")
        df["level1"] = df["level2"].map(lut)
    else:
        df["level1"] = np.nan
    return df


def reaction_table(cfg: Dict[str, Any], key: str) -> pd.DataFrame:
    """All records of one source file, cached as a pickle under work/tables."""
    cache = cfg["paths"]["work"] / "tables" / f"rxn_{key}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    if key == "mech":
        df = _load_mech(cfg)
    else:
        spec = next(
            s
            for s in cfg["datasets"].values()
            if s.get("file") and Path(s["file"]).name.split(".")[0] == key
        )
        df = pd.read_csv(cfg["paths"]["synrxn_data"] / spec["file"])
        df["rxn"] = df["rxn"].astype(str)
    df = df.reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    return df


def dataset_table(cfg: Dict[str, Any], dataset: str) -> pd.DataFrame:
    """
    Rows used for one benchmark.

    Columns: ``row`` (index into the reaction table), ``r_id``, ``rxn``, ``y``
    (0..K-1), ``label`` (original), the hierarchy columns, and
    ``split_published`` when the source provides a split.
    """
    cache = cfg["paths"]["work"] / "tables" / f"ds_{dataset}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    spec = cfg["datasets"][dataset]
    rt = reaction_table(cfg, rxn_key(cfg, dataset))
    label = spec["label"]
    if rt[label].isna().all():
        raise FileNotFoundError(
            f"{dataset}: labels '{label}' unavailable; provide "
            f"{cfg['paths']['mech_level1_map']} (see data/*.DRAFT.csv)"
        )
    df = pd.DataFrame({"row": np.arange(len(rt)), "r_id": rt["r_id"], "rxn": rt["rxn"]})
    df["label"] = rt[label].astype(str).values
    for h in spec.get("hierarchy", []):
        df[h] = rt[h].astype(str).values
    if "split" in rt.columns and spec["split"] == "published":
        df["split_published"] = rt["split"].astype(str).values

    n0 = len(df)
    counts = df["label"].value_counts()
    rare = counts[counts < cfg["splits"]["min_class_count"]]
    df = df[~df["label"].isin(rare.index)].reset_index(drop=True)
    df["y"] = pd.Categorical(df["label"]).codes.astype(np.int32)

    info = {
        "dataset": dataset,
        "records_source": n0,
        "records_kept": len(df),
        "classes_source": int(counts.size),
        "classes_kept": int(df["y"].nunique()),
        "dropped_classes": {str(k): int(v) for k, v in rare.items()},
    }
    log.info("%s: %d/%d records, %d/%d classes", dataset, len(df), n0,
             info["classes_kept"], info["classes_source"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(cache)
    (cache.parent / f"ds_{dataset}.json").write_text(json.dumps(info, indent=1))
    return df


def canonical_reaction(rsmi: str) -> str:
    """
    Order-independent canonical form of a reaction SMILES: canonical SMILES of
    every component (atom maps removed), sorted per side; agents join reactants.
    Returns an empty string for unparsable input.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    parts = strip_atom_maps(rsmi).split(" ")[0].split(">")
    if len(parts) != 3:
        return ""
    reac = [s for s in (parts[0] + "." + parts[1]).split(".") if s]
    prod = [s for s in parts[2].split(".") if s]
    out = []
    for side in (reac, prod):
        smis = []
        for s in side:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                return ""
            smis.append(Chem.MolToSmiles(mol))
        out.append(".".join(sorted(smis)))
    return ">>".join(out)


def canonical_for(cfg: Dict[str, Any], dataset: str) -> np.ndarray:
    """Canonical reaction strings for the rows of a dataset table (cached per source)."""
    from joblib import Parallel, delayed

    key = rxn_key(cfg, dataset)
    cache = cfg["paths"]["work"] / "canonical" / f"{key}.npy"
    if not cache.exists():
        rt = reaction_table(cfg, key)
        log.info("canonicalising %s (%d)", key, len(rt))
        can = Parallel(n_jobs=cfg["n_jobs"], batch_size=512)(
            delayed(canonical_reaction)(s) for s in rt["rxn"]
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, np.asarray(can, dtype=object), allow_pickle=True)
    rows = dataset_table(cfg, dataset)["row"].to_numpy()
    return np.load(cache, allow_pickle=True)[rows]
