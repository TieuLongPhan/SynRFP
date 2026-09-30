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
    return spec["name"]


def load_synrxn(cfg: Dict[str, Any], name: str, task: str = "classification") -> pd.DataFrame:
    """One SynRXN dataset via the ``synrxn`` package (pinned release or local checkout)."""
    from synrxn.data import DataLoader

    sx = cfg["synrxn"]
    if sx["source"] == "local":
        dl = DataLoader(task=task, source="local", data_dir=cfg["paths"]["synrxn_local"])
    else:
        dl = DataLoader(task=task, source=sx["source"], version=sx["version"],
                        cache_dir=cfg["paths"]["synrxn_cache"],
                        force_record_id=sx.get("record_id"))
    log.info("loading SynRXN %s/%s (%s %s)", task, name, sx["source"], sx.get("version", ""))
    return dl.load(name)


def _synrxn_stamp(cfg: Dict[str, Any]) -> Dict[str, str]:
    sx = cfg["synrxn"]
    if sx["source"] == "local":
        return {"source": "local", "dir": str(cfg["paths"]["synrxn_local"])}
    return {"source": str(sx["source"]), "version": str(sx["version"])}


def check_data_source(cfg: Dict[str, Any]) -> None:
    """
    Refuse to reuse cached tables built from a different SynRXN source.

    Every cache under ``work/`` (tables, splits, features, exports) is keyed by
    name only. ``work/tables/SOURCE.json`` records the SynRXN source the
    tables were built from; a mismatch, or SynRXN tables cached before the
    stamp existed, raises instead of silently mixing data versions.

    :raises RuntimeError: If the cached tables come from another source.
    """
    folder = cfg["paths"]["work"] / "tables"
    stamp_path = folder / "SOURCE.json"
    want = _synrxn_stamp(cfg)
    if stamp_path.exists():
        have = json.loads(stamp_path.read_text())
    elif any(p.name != "rxn_mech.pkl" for p in folder.glob("rxn_*.pkl")):
        have = {"source": "unknown (cached before SOURCE.json)"}
    else:
        folder.mkdir(parents=True, exist_ok=True)
        stamp_path.write_text(json.dumps(want, indent=1))
        return
    if have != want:
        work = cfg["paths"]["work"]
        raise RuntimeError(
            f"cached tables in {folder} were built from SynRXN {have}, but the "
            f"config asks for {want}. Remove {work}/tables, {work}/splits, "
            f"{work}/features and {work}/export (or point paths.work elsewhere) "
            "and rerun 01_data/prepare_data.py."
        )


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
    map_value = cfg["paths"].get("mech_level1_map")
    map_path = Path(map_value) if map_value is not None else None
    if map_path is not None and map_path.exists():
        m = pd.read_csv(map_path, comment="#")
        lut = dict(zip(m["mechanistic_class"].astype(str), m["level1"].astype(str)))
        missing = sorted(set(df["level2"]) - set(lut))
        if missing:
            raise ValueError(f"{map_path} lacks level-1 labels for: {missing}")
        df["level1"] = df["level2"].map(lut)
    elif "labels_lv1" in raw and raw["labels_lv1"].notna().all():
        # SynCat's public Mech31k export supplies its nine-class grouping.
        # Do not substitute the repository's deliberately unused draft map.
        df["level1"] = raw["labels_lv1"].astype(str).values
    else:
        df["level1"] = np.nan
    return df


def reaction_table(cfg: Dict[str, Any], key: str) -> pd.DataFrame:
    """All records of one source file, cached as a pickle under work/tables."""
    check_data_source(cfg)
    cache = cfg["paths"]["work"] / "tables" / f"rxn_{key}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    if key == "mech":
        df = _load_mech(cfg)
    else:
        df = load_synrxn(cfg, key)
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
    check_data_source(cfg)
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


def syncat_syntemp_split(cfg: Dict[str, Any], level: int) -> np.ndarray:
    """
    Boolean test mask for every row of the SynRXN ``syntemp`` reaction table,
    from SynCat's published ``Split_R{level}`` column.

    ``paths.syncat_syntemp_split`` must be row-aligned with SynRXN ``syntemp``
    (same RSMI in the same order, identical R0 labels; see exp.md §2,
    "SynTemp split") -- this is a Track B fixed-split source, not derived here.

    :raises FileNotFoundError: If the SynCat file is not present.
    :raises ValueError: If its row count does not match the SynRXN table.
    """
    path = Path(cfg["paths"]["syncat_syntemp_split"])
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing; provide the SynCat checkout to source the "
            "published SynTemp split (exp.md §2, 'SynTemp split')"
        )
    syncat = pd.read_csv(path)
    rt = reaction_table(cfg, "syntemp")
    if len(syncat) != len(rt):
        raise ValueError(
            f"{path} has {len(syncat)} rows, SynRXN syntemp has {len(rt)}; "
            "not row-aligned, cannot source the published split"
        )
    col = f"Split_R{level}"
    return syncat[col].astype(str).str.lower().eq("test").to_numpy()


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


def export_reactions(cfg: Dict[str, Any], key: str, rsmis) -> None:
    """Plain CSV of a reaction table's reactions, for tools in other environments."""
    out = cfg["paths"]["work"] / "export" / f"rxn_{key}.csv.gz"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"rxn": list(rsmis)}).to_csv(out, index=False)


def export_table(cfg: Dict[str, Any], dataset: str, ds: pd.DataFrame) -> None:
    """
    Plain CSV copies of a dataset table and its reaction table (work/export/).

    The RXNFP environment runs an older Python/numpy that cannot read the
    pickles written here, so it only reads these CSV files and the ``.npy``
    split codes.
    """
    key = rxn_key(cfg, dataset)
    export_reactions(cfg, key, reaction_table(cfg, key)["rxn"])
    out = cfg["paths"]["work"] / "export" / f"ds_{dataset}.csv.gz"
    if not out.exists():
        ds[["row", "r_id", "rxn", "y"]].to_csv(out, index=False)
