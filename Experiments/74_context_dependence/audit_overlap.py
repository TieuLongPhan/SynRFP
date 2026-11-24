"""Compare QuantumPioneer held-out graph keys with the archived benchmark source tables."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json
from pathlib import Path
import re

from audit_inputs import PINS, sha256
from audit_quantumpioneer_graphs import graph_row

DELIMITER = re.compile(r"(?<!-)>")


def reaction_sides(row, columns):
    if "left" in columns:
        return row[columns["left"]], row[columns["right"]]
    reaction = row[columns["reaction"]]
    parts = DELIMITER.split(reaction)
    return (parts[0], parts[2]) if len(parts) == 3 else ("", "")


def read_rows(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        yield from csv.DictReader(handle)


def load_catalog(root, properties):
    catalog = {}
    for plan in ("all_endpoint_factorial_full_v1", "cross_task_factorial_v1", "yield_fullrow_v1"):
        entries = json.loads((root / plan / "plan.json").read_text())["sources"]
        for key, entry in entries.items():
            path = Path(entry["table"])
            if path.resolve() in catalog:
                continue
            fields = next(read_rows(path)).keys()
            if {"left", "right"} <= fields:
                columns = {"left": "left", "right": "right"}
            elif "rxn" in fields:
                columns = {"reaction": "rxn"}
            else:
                raise ValueError(f"Unresolved reaction schema for {path}")
            catalog[path.resolve()] = (f"{plan}/{key}", path, columns)
    for path in sorted(properties.glob("*.csv.gz")):
        fields = next(read_rows(path)).keys()
        reaction = next((key for key in ("aam", "rxn") if key in fields), None)
        if not reaction:
            raise ValueError(f"Unresolved property reaction schema for {path}")
        catalog[path.resolve()] = (f"synrxn_property_raw/{path.stem}", path, {"reaction": reaction})
    return list(catalog.values())


def scan(entry, heldout, allowed_shapes):
    name, path, columns = entry
    counts = Counter()
    hits = Counter()
    for index, row in enumerate(read_rows(path)):
        counts["rows"] += 1
        left, right = reaction_sides(row, columns)
        if not left or not right:
            counts["missing_or_invalid_sides"] += 1
            continue
        # The canonical key preserves components separately, so differently sized
        # multisets cannot match in either direction.
        shape = tuple(sorted((left.count(".") + 1, right.count(".") + 1)))
        if shape not in allowed_shapes:
            counts["component_count_mismatch"] += 1
            continue
        record = graph_row((index, "", "", "", f"{left}>>{right}"))
        if record["graph_refusal"]:
            counts["graph_refusals"] += 1
            continue
        counts["valid_candidate_graphs"] += 1
        key = record["canonical_reverse_group"]
        if key in heldout:
            hits[key] += 1
    return name, {
        "source_path": str(path),
        "source_sha256": sha256(path),
        "counts": dict(counts),
        "matching_directions_or_reversals": dict(hits),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--graph-audit", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--properties-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    expected = PINS["quantumpioneer"][1]
    completion = json.loads((args.graph_audit / "COMPLETE.json").read_text())
    ledger = args.graph_audit / "row_graph_audit.csv.gz"
    if sha256(args.source) != expected or completion["source_sha256"] != expected:
        raise ValueError("Unpinned QuantumPioneer input")
    if sha256(ledger) != completion["outputs_sha256"][ledger.name]:
        raise ValueError("Graph-audit ledger hash mismatch")
    test = {row["row_id"]: row for row in read_rows(ledger)
            if row["split"] == "test" and not row["internal_exclusion"]}
    heldout = {row["canonical_reverse_group"] for row in test.values()}
    shapes = set()
    for row in read_rows(args.source):
        if row["source_record_id"] in test:
            left, right = reaction_sides(row, {"reaction": "reaction_smiles"})
            shapes.add(tuple(sorted((left.count(".") + 1, right.count(".") + 1))))
    catalog = load_catalog(args.benchmark_root, args.properties_root)
    results = {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        tasks = [pool.submit(scan, entry, heldout, shapes) for entry in catalog]
        for task in tasks:
            key, row = task.result()
            results[key] = row
            print(key, row["counts"], "matches", len(row["matching_directions_or_reversals"]), flush=True)
    matches = defaultdict(list)
    for name, item in results.items():
        for key in item["matching_directions_or_reversals"]:
            matches[key].append(name)
    matched = [row for row in test.values() if row["canonical_reverse_group"] in matches]
    report = {
        "phase": "listed_benchmark_sources_audited",
        "graph_profile": completion["graph_profile"],
        "quantumpioneer_sha256": expected,
        "graph_audit_sha256": sha256(ledger),
        "source_tables": results,
        "initial_test_rows": len(test),
        "matched_test_row_ids": [row["row_id"] for row in matched],
        "matching_benchmark_sources_by_reverse_key": dict(matches),
        "whole_group_exclusions": sorted({row["connected_group"] for row in matched}),
        "external_scores_opened": False,
    }
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
