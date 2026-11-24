"""Prepare the EnzymeMap cohort and scaffold groups for the coverage audit."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import gzip
from hashlib import sha256
import re

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from audit_quantumpioneer_graphs import digest, molecule

SPLIT_SALT = "enzymemap-v1"
DELIMITER = re.compile(r"(?<!-)>")
EC_PATTERN = re.compile(r"(?:EC\s*)?([1-9]\d*\.[1-9]\d*)(?:\.[^.]+){0,2}")


def canonical_sides(left, right):
    """Return map-free isomeric side multisets, retaining component copies."""
    sides = []
    for side in (left, right):
        components = [part.strip() for part in side.split(".")]
        if not all(components):
            raise ValueError("empty reaction side or component")
        sides.append(tuple(sorted(molecule(part)[0] for part in components)))
    return tuple(sides)


def reverse_key(sides):
    return tuple(sorted(sides))


def cancel_shared(sides):
    left, right = map(Counter, sides)
    return tuple(sorted((left - right).elements())), tuple(
        sorted((right - left).elements())
    )


def reaction_smiles(sides):
    return ">>".join(".".join(side) for side in sides)


@lru_cache(maxsize=16384)
def profile_component(smiles):
    """Identify aliases after removing stereo and isotope annotations."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("component refused by graph-profile normalization")
    Chem.RemoveStereochemistry(mol)
    for atom in mol.GetAtoms():
        atom.SetIsotope(0)
    return Chem.MolToSmiles(mol, canonical=True)


def profile_sides(sides):
    return tuple(
        tuple(sorted(profile_component(part) for part in side)) for side in sides
    )


def canonical_reaction(reaction):
    fields = DELIMITER.split(reaction.strip())
    if len(fields) != 3 or fields[1]:
        return None, "expected two sides and an empty agent field"
    try:
        return canonical_sides(fields[0], fields[2]), ""
    except (ValueError, RuntimeError) as error:
        return None, str(error)


def read_candidates(source, jobs):
    counts = Counter()
    literals = defaultdict(list)
    with gzip.open(source, "rt", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source", "steps", "rxn_idx", "unmapped", "ec_num"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError("EnzymeMap input lacks required columns")
        for index, row in enumerate(reader):
            counts["source_rows"] += 1
            if row["source"] == "direct" and row["steps"] == "single":
                counts["direct_single_rows"] += 1
                label = EC_PATTERN.fullmatch(row["ec_num"].strip())
                literals[row["unmapped"]].append(
                    (index, row["rxn_idx"], label[1] if label else "")
                )

    exclusions = []

    def exclude(rows, reason, detail=""):
        counts[reason] += len(rows)
        exclusions.extend(
            {
                "source_index": index,
                "source_id": source_id,
                "reason": reason,
                "detail": detail,
            }
            for index, source_id, _ in rows
        )

    by_reverse = defaultdict(lambda: defaultdict(list))

    def retain(results):
        for members, (sides, error) in zip(literals.values(), results, strict=True):
            if error:
                exclude(members, "canonical_graph_refusal", error)
            else:
                by_reverse[reverse_key(sides)][sides].extend(members)

    if jobs == 1:
        retain(map(canonical_reaction, literals))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            retain(pool.map(canonical_reaction, literals, chunksize=64))

    candidates = []
    for key, directions in sorted(by_reverse.items()):
        members = sorted(row for rows in directions.values() for row in rows)
        labels = {label for _, _, label in members if label}
        if len(labels) != 1:
            reason = "conflicting_ec2" if labels else "missing_ec2"
            counts[f"{reason}_groups"] += 1
            exclude(members, reason)
            continue

        observed = tuple(sorted(directions, key=reaction_smiles))
        chosen = observed[0]
        representative = min(
            directions[chosen], key=lambda member: (not member[2], member[0])
        )
        exclude(
            [member for member in members if member != representative], "collapsed_copy"
        )
        candidates.append(
            {
                "row_id": digest(key),
                "reaction_smiles": reaction_smiles(chosen),
                "reverse_key": key,
                "observed_directions": observed,
                "ec2": next(iter(labels)),
                "representative_source_index": representative[0],
                "representative_source_id": representative[1],
                "source_indices": tuple(index for index, _, _ in members),
                "source_ids": tuple(sorted({source_id for _, source_id, _ in members})),
            }
        )

    counts["literal_direct_single_reactions"] = len(literals)
    counts["canonical_reversal_groups"] = len(by_reverse)
    counts["retained_reversal_groups"] = len(candidates)
    return candidates, {
        "counts": dict(counts),
        "row_exclusions": sorted(exclusions, key=lambda row: row["source_index"]),
    }


@lru_cache(maxsize=16384)
def scaffold(smiles):
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    mol = Chem.MolFromSmiles(smiles, parameters)
    achiral = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    core = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    key = f"scaffold:{core}" if core else f"acyclic:{achiral}"
    return mol.GetNumHeavyAtoms(), key


def product_anchor(sides):
    products = cancel_shared(sides)[1] or sides[1]
    return min(products, key=lambda part: (-scaffold(part)[0], part))


def split_for(group_id):
    value = int.from_bytes(
        sha256(f"{SPLIT_SALT}\0{group_id}".encode()).digest()[:8], "big"
    )
    if value * 10 < 8 * (1 << 64):
        return "train"
    if value * 10 < 9 * (1 << 64):
        return "validation"
    return "test"


def assign_groups(candidates):
    """Link anchors across observed reversals and representation aliases."""
    parents = {}
    alias_owner = {}

    def root(key):
        while key != parents[key]:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def link(first, second):
        first, second = root(first), root(second)
        parents[max(first, second)] = min(first, second)

    rows = []
    for candidate in candidates:
        directions = candidate["observed_directions"]
        anchors = [product_anchor(direction) for direction in directions]
        keys = tuple(sorted({scaffold(anchor)[1] for anchor in anchors}))
        for key in keys:
            parents.setdefault(key, key)
            link(keys[0], key)
        alias = reverse_key(profile_sides(directions[0]))
        link(keys[0], alias_owner.setdefault(alias, keys[0]))
        non_anchor_products = list(directions[0][1])
        non_anchor_products.remove(anchors[0])
        rows.append(
            dict(
                candidate,
                scaffold_keys=keys,
                anchor_product_smiles=anchors[0],
                non_anchor_products=tuple(non_anchor_products),
            )
        )

    components = defaultdict(list)
    for key in sorted(parents):
        components[root(key)].append(key)
    group_ids = {key: digest(keys) for key, keys in components.items()}
    for row in rows:
        row["group_id"] = group_ids[root(row["scaffold_keys"][0])]
        row["split"] = split_for(row["group_id"])
    return rows
