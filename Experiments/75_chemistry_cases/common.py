"""Pinned archive access and compound size/scaffold rules for request 75."""
from __future__ import annotations

import csv
from functools import lru_cache
from hashlib import file_digest
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
WORKSPACE=REPO.parent
BASE=Path.home()/".local/share/synrfp-runs"
TMP=Path("/var/tmp/synrfp-labhhc4-storage")
STUDY65=BASE/"default_optimal_benchmark_v1"
STUDY22=BASE/"all_endpoint_factorial_full_v1"
STUDY52=BASE/"coverage_refits_v1"
OLD_NATIVE=TMP/"runs/classification_r2_2048_v1"
NATIVE_TABLES=TMP/"work/validation/n1/tables"
CEILING=BASE/"delta_ceiling_request74_v2"
PAPER=WORKSPACE/"SynRFP-paper-attribution"
PAPER_REV="ac83ccf"
OUTPUT=HERE/"publication_metadata/chemistry_cases_v1"
LOCAL=BASE/"chemistry_cases_request75_v1"
SPLIT_ARROW=re.compile(r"(?<!-)>")
RDLogger.DisableLog("rdApp.*")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")
    temporary.replace(path)


class InputAudit:
    def __init__(self):
        self.pins={}

    def use(self, path, expected=None):
        p=Path(path)
        if str(p) not in self.pins:
            with p.open("rb") as handle:
                self.pins[str(p)]=file_digest(handle,"sha256").hexdigest()
        if expected is not None and self.pins[str(p)]!=expected:
            raise ValueError(f"Pinned input changed: {p}")
        return p


def export(path, rows, fields=None):
    if fields is None:
        fields=list(dict.fromkeys(key for row in rows for key in row))
    temp=path.with_suffix(".csv.tmp")
    with temp.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields,lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def groups(keys):
    _, labels=np.unique(keys,return_inverse=True)
    return labels.astype(np.int32)


class Buckets:
    def __init__(self, labels):
        self.order=np.argsort(labels,kind="stable")
        self.offset=np.r_[0,np.cumsum(np.bincount(labels))]

    def members(self,label):
        return self.order[self.offset[label]:self.offset[label+1]]


def packed_ids(bits):
    return groups(np.ascontiguousarray(bits).view(f"V{bits.shape[1]}").ravel())


@lru_cache(maxsize=16384)
def chemistry(text):
    """Return total heavy atoms and largest-component Murcko group."""
    components=[s for side in SPLIT_ARROW.split(text) for s in side.split(".") if s]
    molecules=[Chem.MolFromSmiles(s) for s in components]
    if not molecules or any(m is None for m in molecules):
        raise ValueError("Could not canonicalize a candidate's largest component")
    configs=[]
    for mol in molecules:
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)
        configs.append((mol.GetNumHeavyAtoms(),Chem.MolToSmiles(mol,isomericSmiles=False),mol))
    _, smiles, mol=min(configs,key=lambda item:(-item[0],item[1]))
    scaffold=MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False)
    return sum(entry[0] for entry in configs),scaffold or f"acyclic:{smiles}"


def load_plan(audit):
    p=STUDY65/"plan.json"
    plan=read(audit.use(p,read(STUDY65/"plan_seal.json")["sha256"]))
    selector=STUDY65/"native/classification/plan.json"
    return read(audit.use(selector,plan["inputs_sha256"][str(selector)]))


def source_data(audit, key):
    old=STUDY22/"plan.json"
    plan=read(audit.use(old,read(STUDY22/"plan_seal.json")["sha256"]))
    source=Path(plan["sources"][key]["table"])
    table=pd.read_csv(audit.use(source,plan["pinned_files"][str(source)]),dtype={"row_id":str})
    np.testing.assert_array_equal(table.source_position,np.arange(len(table)))
    proof=read(audit.use(CEILING/"ENCODED.json"))
    audit.use(CEILING/"ENCODING_PLAN.json",proof["plan_sha256"])
    artifact=CEILING/(key+".npz")
    with np.load(audit.use(artifact,proof["sources"][key]["artifact_sha256"]),allow_pickle=False) as a:
        np.testing.assert_array_equal(a["row_ids"],table.row_id.to_numpy(str))
        return table,a["keys"][:,0,:].copy(),a["heavy"],a["ok"]


def feature(audit, key, mode, expected_ok, ids):
    name=f"synrfp-wl-r2-b2048-parity-{mode}"
    prefix=STUDY52 if key=="tpl_b" else STUDY22
    path=prefix/"features"/key/(name+".npz")
    meta=read(audit.use(path.with_suffix(".json")))
    with np.load(audit.use(path,meta["feature_sha256"]),allow_pickle=False) as z:
        np.testing.assert_array_equal(z["ok"],expected_ok)
        np.testing.assert_array_equal(z["row_ids"],ids)
        np.testing.assert_array_equal(z["source_positions"],np.arange(len(ids)))
        x=z["X"]
    if x.shape != (len(ids),256) or x.dtype != np.uint8:
        raise ValueError("Unexpected fingerprint layout")
    return x


def panel_data(audit, endpoint, panel, source):
    f=pd.read_csv(audit.use(panel["table"],panel["table_sha256"]),dtype={"row_id":str})
    np.testing.assert_array_equal(f.dataset_position,np.arange(panel["n"]))
    np.testing.assert_array_equal(f.row_id.to_numpy(str),source.row_id.to_numpy(str)[f.source_position])
    raw=pd.read_pickle(audit.use(NATIVE_TABLES/f"ds_{endpoint}.pkl"))
    np.testing.assert_array_equal(raw.row,f.source_position)
    np.testing.assert_array_equal(raw.y,f.y)
    np.testing.assert_array_equal(raw.r_id.astype(str),f.row_id)
    dictionary={int(y):str(label) for y,label in zip(raw.y,raw.label,strict=True)}
    if len(set(zip(raw.y,raw.label,strict=True)))!=len(dictionary):
        raise ValueError("Label vocabulary is not bijective")
    return f,dictionary
