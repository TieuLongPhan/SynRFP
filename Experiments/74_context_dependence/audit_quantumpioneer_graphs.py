"""Audit the pinned QuantumPioneer graph identities and source split membership."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

from rdkit import Chem, RDLogger

from audit_inputs import PINS, sha256

RDLogger.DisableLog("rdApp.*")


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


@lru_cache(maxsize=4096)
def molecule(smiles):
    parameters = Chem.SmilesParserParams()
    parameters.removeHs = False
    mol = Chem.MolFromSmiles(smiles, parameters)
    if mol is None:
        raise ValueError("rdkit_component_parse_failure")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    chemistry = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return (
        chemistry,
        any(atom.GetAtomicNum() == 1 or atom.GetNumExplicitHs() for atom in mol.GetAtoms()),
        any(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()),
        any(atom.GetFormalCharge() for atom in mol.GetAtoms()),
    )


def graph_row(row):
    source_index, row_id, source_group, split, reaction = row
    out = dict(source_index=source_index, row_id=row_id, source_group=source_group,
               split=split, canonical_direction="", canonical_reverse_group="",
               canonical_reactants="", explicit_h=False, radicals=False, charge=False,
               graph_refusal="")
    try:
        reactants, agents, products = reaction.strip().split(">")
        if agents or not reactants or not products:
            raise ValueError("unexpected_empty_side_or_agents")
        r, p = ([molecule(part) for part in side.split(".")] for side in (reactants, products))
        canon_r, canon_p = (tuple(sorted(item[0] for item in side)) for side in (r, p))
        out.update(
            canonical_direction=digest((canon_r, canon_p)),
            canonical_reverse_group=digest(sorted((canon_r, canon_p))),
            canonical_reactants=digest(canon_r),
            explicit_h=any(item[1] for item in r + p),
            radicals=any(item[2] for item in r + p),
            charge=any(item[3] for item in r + p),
        )
    except (ValueError, RuntimeError) as error:
        out["graph_refusal"] = str(error)
    return out


def source_rows(path):
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            yield (int(row["source_row_index"]), row["source_record_id"], row["reactant_pair_group"],
                   row["proposed_split"], row["reaction_smiles"])


def connected_groups(rows):
    parent = {row["source_group"]: row["source_group"] for row in rows}

    def root(group):
        while group != parent[group]:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    owners = {}
    for row in rows:
        for field in ("canonical_reactants", "canonical_reverse_group"):
            value = row[field]
            if not value:
                continue
            first = root(owners.setdefault((field, value), row["source_group"]))
            second = root(row["source_group"])
            parent[max(first, second)] = min(first, second)
    split_sets = defaultdict(set)
    for row in rows:
        group = root(row["source_group"])
        row["connected_group"] = group
        split_sets[group].add(row["split"])
    return {group for group, splits in split_sets.items() if len(splits) > 1}


def record_internal_eligibility(rows):
    crossing = connected_groups(rows)
    counted = set()
    for row in sorted(rows, key=lambda record: record["source_index"]):
        if row["graph_refusal"]:
            reason = "canonical_graph_refusal"
        elif row["connected_group"] in crossing:
            reason = "connected_group_crosses_source_split"
        elif row["canonical_direction"] in counted:
            reason = "duplicate_canonical_direction"
        else:
            counted.add(row["canonical_direction"])
            reason = ""
        row["internal_exclusion"] = reason
    return len(crossing)


def audit(source, output, jobs):
    expected = PINS["quantumpioneer"]
    if sha256(source) != expected[1]:
        raise ValueError("QuantumPioneer projection does not match the protocol")
    output.mkdir(parents=True, exist_ok=False)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        rows = list(pool.map(graph_row, source_rows(source), chunksize=128))
    if len(rows) != expected[2]:
        raise ValueError("Unexpected QuantumPioneer row count")
    crossings = record_internal_eligibility(rows)
    eligible = [row for row in rows if not row["internal_exclusion"]]
    fields = list(rows[0])
    ledger = output / "row_graph_audit.csv.gz"
    with gzip.open(ledger, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "phase": "internal_graph_audit_complete",
        "source_sha256": expected[1],
        "graph_profile": "RDKit canonical isomeric sides, removeHs=False, atom maps removed",
        "source_rows": dict(Counter(row["split"] for row in rows)),
        "crossing_connected_groups": crossings,
        "internally_eligible_rows": dict(Counter(row["split"] for row in eligible)),
        "exclusions": dict(Counter(row["internal_exclusion"] for row in rows if row["internal_exclusion"])),
        "rows_with_explicit_h": sum(row["explicit_h"] for row in rows),
        "rows_with_radicals": sum(row["radicals"] for row in rows),
        "rows_with_charge": sum(row["charge"] for row in rows),
        "prior_benchmark_overlap_completed": False,
        "feature_profile_loss_audit_completed": False,
        "external_scores_opened": False,
        "outputs_sha256": {ledger.name: sha256(ledger)},
    }
    (output / "COMPLETE.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    audit(args.source, args.output, args.jobs)


if __name__ == "__main__":
    main()
