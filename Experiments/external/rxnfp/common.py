"""Minimal, dependency-light access to the experiment tables from the rxnfp env.

The main package needs Python >= 3.11; this module only reads the pickled
tables and split arrays written by 01_data/prepare_data.py, so the RXNFP
environment does not have to import it.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

EXP_DIR = Path(__file__).resolve().parents[2]


def load_config(path=None):
    path = Path(path) if path else EXP_DIR / "config.yaml"
    cfg = yaml.safe_load(path.read_text())
    base = path.resolve().parent
    cfg["paths"] = {k: (base / v).resolve() for k, v in cfg["paths"].items()}
    return cfg


def dataset_table(cfg, dataset):
    p = cfg["paths"]["work"] / "tables" / f"ds_{dataset}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing; run 01_data/prepare_data.py first")
    return pd.read_pickle(p)


def reaction_table(cfg, key):
    return pd.read_pickle(cfg["paths"]["work"] / "tables" / f"rxn_{key}.pkl")


def rxn_key(cfg, dataset):
    spec = cfg["datasets"][dataset]
    return "mech" if spec.get("source") == "mech" else Path(spec["file"]).name.split(".")[0]


def split_codes(cfg, dataset, scheme, seed):
    return np.load(cfg["paths"]["work"] / "splits" / dataset / f"{scheme}_s{seed}.npy")


def save_features(path, X, ok):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X.astype(np.float32), n_bits=np.int64(0), ok=ok)


def rxnfp_input(rsmi):
    """RXNFP expects 'reactants>>products'; agents are merged into the reactants."""
    parts = rsmi.split(" ")[0].split(">")
    reac = ".".join(p for p in (parts[0], parts[1]) if p)
    return f"{reac}>>{parts[2]}"
