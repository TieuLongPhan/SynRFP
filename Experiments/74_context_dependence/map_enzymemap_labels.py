"""Map the frozen EnzymeMap training reactions and derive EC-blind bond-edit labels."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import importlib.machinery
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import types

from rdkit import Chem, RDLogger

from audit_inputs import sha256
from enzymemap_cohort import DELIMITER, canonical_sides

STUDY = Path(__file__).resolve().parent
MAPPER_REVISION = "bfa575bc02a075665dd9a8a222f5a960b6390895"
CHECKPOINT = "3fd315bb4b9d3f056b51c771395f601c2dcd7db64dd75bf63e99d6ce1da3de72"
BATCH_SIZE = 8
MAPPER = None


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_precheck(precheck, followup):
    report = read(precheck / "eligibility_precheck.json")
    gate = read(followup / "eligibility_followup.json")
    pins = {
        precheck / "eligibility_precheck.json": gate["baseline_eligibility_sha256"],
        precheck / "overlap.json": gate["baseline_overlap_sha256"],
        followup / "refusal_audit.json": gate["refusal_audit_sha256"],
        **{precheck / name: digest for name, digest in report["outputs_sha256"].items()},
    }
    if gate["decision"] != "proceed_to_label_audit" or gate["remaining_unresolved_entries"]:
        raise ValueError("The frozen overlap audit has not cleared mapping")
    for path, expected in pins.items():
        if sha256(path) != expected:
            raise ValueError(f"Changed precheck file {path}")
    return report, pins


def training_rows(precheck, expected):
    with gzip.open(precheck / "cohort.csv.gz", "rt", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == "train"]
    if len(rows) != expected or len({row["row_id"] for row in rows}) != expected:
        raise ValueError("Training row count or identity disagrees with the frozen precheck")
    return rows


def sides(smiles):
    left, agents, right = DELIMITER.split(smiles)
    if not left or agents or not right:
        raise ValueError("expected_two_sides")
    return left, right


def mapped_graph(smiles):
    params = Chem.SmilesParserParams()
    params.removeHs = False
    molecule = Chem.MolFromSmiles(smiles, params)
    if molecule is None:
        raise ValueError("mapped_side_parse_failure")
    atoms = {}
    for atom in molecule.GetAtoms():
        number = atom.GetAtomMapNum()
        if not number:
            raise ValueError("unmatched_atom")
        if number in atoms:
            raise ValueError("duplicate_map_on_side")
        atoms[number] = atom.GetAtomicNum(), atom.GetIsotope()
    bonds = {
        tuple(sorted((bond.GetBeginAtom().GetAtomMapNum(), bond.GetEndAtom().GetAtomMapNum()))):
        "AROMATIC" if bond.GetIsAromatic() else str(bond.GetBondType())
        for bond in molecule.GetBonds()
    }
    return atoms, bonds


def edit_label(source, mapped):
    unmapped = sides(source)
    mapped = sides(mapped)
    if canonical_sides(*unmapped) != canonical_sides(*mapped):
        raise ValueError("mapped_reaction_round_trip_mismatch")
    left_atoms, left = mapped_graph(mapped[0])
    right_atoms, right = mapped_graph(mapped[1])
    if left_atoms.keys() != right_atoms.keys():
        raise ValueError("unmatched_atom")
    if left_atoms != right_atoms:
        raise ValueError("map_changes_element_or_isotope")
    edits = sorted(
        (*sorted((left_atoms[pair[0]][0], left_atoms[pair[1]][0])),
         left.get(pair, "ABSENT"), right.get(pair, "ABSENT"))
        for pair in left.keys() | right.keys()
        if left.get(pair) != right.get(pair)
    )
    return "no_edit" if not edits else json.dumps(edits, separators=(",", ":"))


def mapper_files(source, deps):
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if revision != MAPPER_REVISION or dirty:
        raise ValueError("Expected the clean pinned RXNMapper v2 checkout")
    package = source / "rxnmapper"
    model = package / "models/transformers/alberta_uspto_2800k"
    if sha256(model / "model.safetensors") != CHECKPOINT:
        raise ValueError("Unexpected mapper checkpoint")
    files = [*package.rglob("*.py"), *model.iterdir(), *deps.rglob("*.py")]
    return {str(path): sha256(path) for path in sorted(files) if path.is_file()}


def init_mapper(source, deps):
    global MAPPER
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Hide CUDA explicitly before loading the CPU mapper")
    RDLogger.DisableLog("rdApp.*")
    sys.path.insert(0, str(deps))
    package = source / "rxnmapper"
    # Load the unchanged inference core with local assets. The upstream package
    # initializer also imports optional model-evaluation dependencies.
    module = types.ModuleType("rxnmapper")
    module.__path__ = [str(package)]
    module.__file__ = str(package / "__init__.py")
    module.__spec__ = importlib.machinery.ModuleSpec("rxnmapper", loader=None, is_package=True)
    sys.modules["rxnmapper"] = module
    import torch
    from rxnmapper.core import RXNMapper

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    model = package / "models/transformers/alberta_uspto_2800k"
    MAPPER = RXNMapper(config={
        "model_name": "default", "model_path": str(model), "vocab_path": str(model / "vocab.txt"),
        "head": 3, "layers": [10],
    })
    if MAPPER.device.type != "cpu":
        raise ValueError("Unexpected mapper device")
    MAPPER.model.eval()


def infer(batch):
    try:
        predictions = MAPPER.get_attention_guided_atom_maps(
            [smiles for _, smiles in batch], canonicalize_rxns=True
        )
    except (AttributeError, IndexError, TypeError, ValueError) as error:
        if len(batch) == 1:
            return [{"row_id": batch[0][0], "refusal": "mapper_input_failure",
                     "detail": f"{type(error).__name__}: {str(error)[:160]}"}]
        half = len(batch) // 2
        return infer(batch[:half]) + infer(batch[half:])

    output = []
    for (row_id, smiles), prediction in zip(batch, predictions, strict=True):
        result = {"row_id": row_id}
        try:
            confidence = float(prediction["confidence"])
            if not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("invalid_mapper_confidence")
            result["label"] = edit_label(smiles, prediction["mapped_rxn"])
            result["confidence"] = confidence
            result["mapped_reaction"] = prediction["mapped_rxn"]
        except ValueError as error:
            result["refusal"] = str(error)
        output.append(result)
    return output


def run(precheck, followup, source, deps, output, jobs, pilot):
    started = time.monotonic()
    report, pins = verify_precheck(precheck, followup)
    rows = training_rows(precheck, report["cohort"]["rows"]["train"])
    if pilot:
        rows = rows[::max(1, len(rows) // pilot)][:pilot]
    inputs = [(row["row_id"], row["reaction_smiles"]) for row in rows]
    manifest = {
        "pinned_inputs": {str(path): digest for path, digest in pins.items()},
        "mapper_revision": MAPPER_REVISION,
        "mapper_assets": mapper_files(source, deps),
        "runner_sha256": sha256(Path(__file__)),
        "versions": {name: version(name) for name in ("torch", "transformers", "rdkit", "numpy")},
        "profile": "atom_conserving_bond_type_multiset_v1",
        "pilot": pilot, "expected": len(inputs), "device": "cpu", "batch_size": BATCH_SIZE,
        "mapper_head": 3, "mapper_layers": [10], "confidence_cutoff": None,
        "mapper_inputs": ["unmapped reaction SMILES"], "scope": "train",
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if read(manifest_path) != manifest:
            raise ValueError("Saved mapping manifest differs from this launch")
    else:
        write(manifest_path, manifest)
    batches = [inputs[offset:offset + BATCH_SIZE] for offset in range(0, len(inputs), BATCH_SIZE)]
    checkpoint = output / "batches"
    checkpoint.mkdir(exist_ok=True)

    def batch_path(index):
        return checkpoint / f"{index:06d}.json"

    missing = [index for index in range(len(batches)) if not batch_path(index).exists()]
    print(f"CPU mapper ready: {len(rows):,} train reactions, {len(missing):,} pending batches", flush=True)
    with ProcessPoolExecutor(max_workers=jobs, initializer=init_mapper, initargs=(source, deps)) as pool:
        completed = len(batches) - len(missing)
        for index, result in zip(missing, pool.map(infer, (batches[i] for i in missing)), strict=True):
            write(batch_path(index), result)
            completed += 1
            if completed % 50 == 0 or completed == len(batches):
                print(f"Mapped batches {completed:,}/{len(batches):,}", flush=True)
    records = []
    for index, batch in enumerate(batches):
        result = read(batch_path(index))
        if [record["row_id"] for record in result] != [row_id for row_id, _ in batch]:
            raise ValueError(f"Unexpected row order in batch {index}")
        records.extend(result)
    refusals = Counter(record["refusal"] for record in records if "refusal" in record)
    if sha256(Path(__file__)) != manifest["runner_sha256"] or mapper_files(source, deps) != manifest["mapper_assets"]:
        raise ValueError("Mapper or runner changed during the job")
    for path, digest in pins.items():
        if sha256(path) != digest:
            raise ValueError(f"Changed precheck input during mapping: {path}")
    labels = output / "train_labels.json"
    write(labels, records)
    write(output / "COMPLETE.json", {
        "rows": len(records), "validated": len(records) - refusals.total(), "refusals": refusals,
        "labels_sha256": sha256(labels), "manifest_sha256": sha256(manifest_path),
        "elapsed_seconds": time.monotonic() - started, "pilot": pilot,
    })
    print(json.dumps(read(output / "COMPLETE.json")), flush=True)


def main():
    parser = argparse.ArgumentParser()
    for option in ("precheck", "followup", "mapper-source", "mapper-deps", "output"):
        parser.add_argument(f"--{option}", required=True, type=Path)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--pilot", type=int)
    args = parser.parse_args()
    if args.jobs < 1 or (args.pilot is not None and args.pilot < 1):
        parser.error("jobs and pilot must be positive")
    run(args.precheck.resolve(), args.followup.resolve(), args.mapper_source.resolve(),
        args.mapper_deps.resolve(), args.output.resolve(), args.jobs, args.pilot)


if __name__ == "__main__":
    main()
