"""Verify pinned input bytes and source-level eligibility before graph auditing."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess

PINS = {
    "b97xd3": (
        "b97xd3.csv.gz",
        "33f881612842378a6cff97b2db0c99b1c4877a7e81a318129a26dd58e9702d73",
        16365,
    ),
    "cycloadd": (
        "cycloadd.csv.gz",
        "afd03a88ef6268350f2c176b70a67396ff1902b85e6742bb2ac3ef0d9ca84004",
        5269,
    ),
    "quantumpioneer": (
        "quantumpioneer_kinetics.csv",
        "e81aa5f9158564e64356a524c046469e2a13e471469ff641713ded3c37ff1520",
        166158,
    ),
    "enzymemap": (
        "enzymemap_v2_brenda2023.csv.gz",
        "0a70759a7cec5720b6aeb2d63c887a670b9ac2a6ada7c0a1ef3c1740ee3cc2cf",
        349458,
    ),
}
TARGETS = {"b97xd3": ("ea", "dh"), "cycloadd": ("G_act", "G_r"),
           "quantumpioneer": ("barrier", "deltaHrxn")}


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def audit(name, path):
    filename, expected_hash, expected_rows = PINS[name]
    entry = {"expected_filename": filename, "expected_sha256": expected_hash,
             "expected_rows": expected_rows}
    if path is None:
        return {**entry, "status": "missing"}
    digest = sha256(path)
    entry.update(path=str(path.resolve()), sha256=digest)
    if digest != expected_hash:
        return {**entry, "status": "checksum_mismatch"}

    targets = TARGETS.get(name, ())
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = set(targets)
        if name == "quantumpioneer":
            required.update(("proposed_split", "source_record_id", "reactant_pair_group", "energy_unit"))
        elif name == "enzymemap":
            required.update(("source", "steps", "unmapped", "ec_num"))
        if missing := required.difference(fields):
            raise ValueError(f"{name}: missing fields {sorted(missing)}")

        counters = Counter()
        literal_reactions = set()
        ids = set()
        split_sizes = Counter()
        split_groups = defaultdict(set)
        energy_units = Counter()
        for row in reader:
            counters["rows"] += 1
            if "r_id" in row:
                ids.add(row["r_id"])
            if name == "quantumpioneer":
                split = row["proposed_split"]
                split_sizes[split] += 1
                split_groups[split].add(row["reactant_pair_group"])
                ids.add(row["source_record_id"])
                energy_units[row["energy_unit"]] += 1
            elif row.get("split"):
                split_sizes[row["split"]] += 1
            if targets:
                availability = [finite(row[target]) for target in targets]
                for target, present in zip(targets, availability, strict=True):
                    counters[f"finite_{target}"] += present
                counters["both_targets_finite"] += all(availability)
            elif row["source"] == "direct" and row["steps"] == "single":
                counters["direct_single_rows"] += 1
                literal_reactions.add(row["unmapped"])

    if counters["rows"] != expected_rows:
        raise ValueError(f"{name}: expected {expected_rows} rows, found {counters['rows']}")
    if name == "enzymemap":
        counters["distinct_direct_single_literal_reactions"] = len(literal_reactions)
        if (counters["direct_single_rows"], len(literal_reactions)) != (231492, 32920):
            raise ValueError("EnzymeMap direct/single population differs from protocol")
    if "r_id" in fields:
        counters["distinct_source_row_ids"] = len(ids)
    if name == "quantumpioneer":
        if split_sizes != {"train": 133299, "validation": 16530, "test": 16329}:
            raise ValueError("QuantumPioneer source split counts differ from protocol")
        group_sizes = {split: len(groups) for split, groups in split_groups.items()}
        expected_groups = {"train": 109093, "validation": 13551, "test": 13349}
        if group_sizes != expected_groups or len(set.union(*split_groups.values())) != sum(group_sizes.values()):
            raise ValueError("QuantumPioneer source groups differ or cross splits")
        if len(ids) != expected_rows:
            raise ValueError("QuantumPioneer source identifiers are not unique")
        entry.update(split_group_counts=group_sizes, energy_units=dict(energy_units))
    return {**entry, "status": "source_intake_verified", "columns": fields,
            "counts": dict(counters), "split_rows": dict(split_sizes)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in PINS:
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = Path(__file__).with_name("PROTOCOL.md")
    report = {
        "stage": "source_intake",
        "protocol_sha256": sha256(protocol),
        "script_sha256": sha256(Path(__file__)),
        "checkout_commit": subprocess.check_output(
            ["git", "-C", str(protocol.parent), "rev-parse", "HEAD"], text=True
        ).strip(),
        "sources": {name: audit(name, getattr(args, name)) for name in PINS},
        "models_fitted": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    for name, entry in report["sources"].items():
        print(name, entry["status"], entry.get("counts", {}))


if __name__ == "__main__":
    main()
