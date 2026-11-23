"""Build the matched parity fingerprints for the E1 yield ablation."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdChemReactions

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Load the encoder for this worktree even if another checkout is installed.
import synrfp
from synrfp.encode.core import fingerprint_from_tokens
from synrfp.graph.reaction import Reaction
from synrfp.tokenizers.wl import WLTokenizer

ARMS = ("d1024", "d2048", "d_pooled", "d_rehash")
PLACES = ("R", "A", "S")
WIDTHS = dict.fromkeys(ARMS, 2048) | {"d1024": 1024}
DATASETS = {
    "bh": ("bh_classic.csv.gz", 3955),
    "suzuki": ("suzuki_native_reference.csv.gz", 5760),
}

ELECTROPHILES = tuple(
    Chem.MolFromSmarts(text) for text in ("[#6][Br,Cl,I]", "c-O-S(=O)(=O)C(F)(F)F")
)
ORGANOBORON = Chem.MolFromSmarts("[#6]-[B,B-]")
BH_PRODUCT = rdChemReactions.ReactionFromSmarts("[c:1][Cl,Br,I]>>[c:1]Nc1ccc(C)cc1")
ANILINE = "Cc1ccc(N)cc1"
BH_PRECATALYST = "O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F"
TOKENIZER = WLTokenizer(
    node_attrs=["element", "aromatic", "charge", "hcount"], edge_attrs=["order"]
)


@dataclass(frozen=True)
class Record:
    sample_id: str
    target: float
    substrates: tuple[str, ...]
    conditions: tuple[str, ...]
    products: tuple[str, ...]
    group: str


def file_sha256(path):
    with Path(path).open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def environment():
    return {
        "python": platform.python_version(),
        **{
            name: version(name)
            for name in ("numpy", "pandas", "rdkit", "synkit", "scikit-learn", "joblib")
        },
    }


@lru_cache(maxsize=512)
def molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid component SMILES {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def canonical(smiles):
    return Chem.MolToSmiles(molecule(smiles))


@lru_cache(maxsize=64)
def bh_product(aryl):
    products = set()
    for (product,) in BH_PRODUCT.RunReactants((molecule(aryl),)):
        Chem.SanitizeMol(product)
        products.add(Chem.MolToSmiles(product))
    if len(products) != 1:
        raise ValueError(f"Expected one intended Buchwald–Hartwig product for {aryl!r}")
    return products.pop()


def bh_record(row):
    aryl = canonical(row.aryl_halide_smiles)
    conditions = (
        row.ligand_smiles,
        row.additive_smiles,
        row.base_smiles,
        BH_PRECATALYST,
    )
    return Record(
        row.sample_id,
        row.yield_pct,
        (aryl, canonical(ANILINE)),
        tuple(sorted(map(canonical, conditions))),
        (bh_product(aryl),),
        aryl,
    )


def suzuki_record(row):
    # Preserve the author's unspecified bonds in the Pd precursor.
    left, product = row.author_reaction_smiles.split(">>")
    components = left.split(".")
    electrophiles, organoborons, conditions = [], [], []
    for smiles in components:
        mol = molecule(smiles)
        electrophile = any(mol.HasSubstructMatch(pattern) for pattern in ELECTROPHILES)
        organoboron = mol.HasSubstructMatch(ORGANOBORON) and all(
            atom.GetAtomicNum() != 46 for atom in mol.GetAtoms()
        )
        if electrophile and organoboron:
            raise ValueError("Component matches both precursor roles")
        component = Chem.MolToSmiles(mol)
        if electrophile:
            electrophiles.append(component)
        elif organoboron:
            organoborons.append(component)
        else:
            conditions.append(component)
    if len(electrophiles) != 1 or len(organoborons) != 1:
        raise ValueError("Expected one electrophile and one organoboron precursor")
    substrates = (electrophiles[0], organoborons[0])
    return Record(
        row.sample_id,
        row.local_yield_pct,
        substrates,
        tuple(sorted(conditions)),
        tuple(sorted(map(canonical, product.split(".")))),
        ".".join(substrates),
    )


def read_records(dataset, path):
    data = pd.read_csv(path, dtype={"sample_id": "string"}).sort_values("sample_id")
    _, expected_n = DATASETS[dataset]
    if (
        len(data) != expected_n
        or data.sample_id.isna().any()
        or not data.sample_id.is_unique
    ):
        raise ValueError(f"{dataset}: incorrect source row count or sample identifiers")
    convert = bh_record if dataset == "bh" else suzuki_record
    records = []
    for row in data.itertuples(index=False):
        try:
            records.append(convert(row))
        except (AttributeError, TypeError, ValueError, RuntimeError) as error:
            raise ValueError(f"{dataset}: failed to parse {row.sample_id}") from error
    if not np.isfinite([record.target for record in records]).all():
        raise ValueError(f"{dataset}: non-finite yield")
    groups = {record.group for record in records}
    products = {record.products for record in records}
    if len(groups) != 15 or (dataset == "suzuki" and len(products) != 1):
        raise ValueError(f"{dataset}: unexpected precursor group or product coverage")
    return records


@lru_cache(maxsize=512)
def component_tokens(smiles):
    graph = Reaction.from_rsmi(f"{smiles}>>{smiles}").reactant
    return TOKENIZER.tokens_graph(graph, 2)


def side_counts(components):
    result = Counter()
    for smiles in components:
        result.update(component_tokens(smiles))
    return result


def fingerprints(reactant, product):
    wide = fingerprint_from_tokens(
        reactant, product, mode="delta", sketch="parity", bits=2048, seed=1
    )
    # Same-seed parity buckets modulo 1024 combine the corresponding two
    # buckets modulo 2048. Reuse the verified XOR relation from the SI.
    narrow = wide[:1024] ^ wide[1024:]
    pooled = fingerprint_from_tokens(
        reactant, product, mode="union", sketch="parity", bits=1024, seed=1
    )
    second = fingerprint_from_tokens(
        reactant, product, mode="delta", sketch="parity", bits=1024, seed=2
    )
    return {
        "d1024": narrow,
        "d2048": wide,
        "d_pooled": np.concatenate((narrow, pooled)),
        "d_rehash": np.concatenate((narrow, second)),
    }


def feature_arrays(records):
    condition_names = sorted(
        {condition for record in records for condition in record.conditions}
    )
    condition_index = {name: i for i, name in enumerate(condition_names)}
    result = {
        "sample_id": np.asarray([record.sample_id for record in records], dtype=str),
        "y": np.asarray([record.target for record in records], dtype=np.float64),
        "group": np.asarray([record.group for record in records], dtype=str),
        "condition_names": np.asarray(condition_names, dtype=str),
        "H": np.zeros((len(records), len(condition_names)), dtype=np.uint8),
    }
    for place in PLACES:
        for arm in ARMS:
            result[f"{place}_{arm}"] = np.empty(
                (len(records), WIDTHS[arm]), dtype=np.uint8
            )
    for index, record in enumerate(records):
        left = side_counts(record.substrates)
        right = side_counts(record.products)
        conditions = side_counts(record.conditions)
        result["H"][index, [condition_index[s] for s in set(record.conditions)]] = 1
        for place, reactant, product in (
            ("R", left + conditions, right),
            ("A", left + conditions, right + conditions),
            ("S", left, right),
        ):
            for arm, values in fingerprints(reactant, product).items():
                result[f"{place}_{arm}"][index] = values
    for arm in ("d1024", "d2048", "d_rehash"):
        if not np.array_equal(result[f"A_{arm}"], result[f"S_{arm}"]):
            raise ValueError(f"Condition cancellation failed for {arm}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--check-only", action="store_true", help="Audit sources without encoding"
    )
    args = parser.parse_args()
    if not args.check_only and args.output_dir.exists():
        parser.error("Choose a new output directory")
    if Path(synrfp.__file__).resolve().parent != ROOT / "synrfp":
        raise RuntimeError("SynRFP must import from this checkout")

    sources = {key: args.data_dir / name for key, (name, _) in DATASETS.items()}
    inputs = {path.name: file_sha256(path) for path in sources.values()}
    records = {key: read_records(key, path) for key, path in sources.items()}
    coverage = {
        key: {
            "source_rows": len(rows),
            "groups": len({r.group for r in rows}),
            "intended_products": len({r.products for r in rows}),
            "condition_components": len({c for r in rows for c in r.conditions}),
        }
        for key, rows in records.items()
    }
    print(
        json.dumps({"input_sha256": inputs, "coverage": coverage}, indent=2), flush=True
    )
    if args.check_only:
        return

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".features-", dir=args.output_dir.parent) as staging:
        folder = Path(staging)
        for key, rows in records.items():
            np.savez_compressed(folder / f"{key}.npz", **feature_arrays(rows))
            print(f"Encoded {len(rows)} rows from {key}", flush=True)
        metadata = {
            "schema": 1,
            "input_sha256": inputs,
            "coverage": coverage,
            "feature_sha256": {
                p.name: file_sha256(p) for p in sorted(folder.glob("*.npz"))
            },
            "source_revision": subprocess.check_output(
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "dirty_encoder": bool(
                subprocess.check_output(
                    ["git", "-C", str(ROOT), "status", "--porcelain", "--", "synrfp"]
                ).strip()
            ),
            "feature_code_sha256": file_sha256(__file__),
            "versions": environment(),
        }
        (folder / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        folder.rename(args.output_dir)


if __name__ == "__main__":
    main()
