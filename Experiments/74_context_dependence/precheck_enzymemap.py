"""Audit EnzymeMap and bound training-only exact-delta coverage before mapping."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass, field
import gzip
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from audit_inputs import PINS, audit, sha256
from enzymemap_cohort import assign_groups, read_candidates
from enzymemap_overlap import audit_overlap
import prepare_qp_features

STUDY = Path(__file__).resolve().parent
MIN_GROUPS = 30
MIN_CLASSES = 10


@dataclass
class ExactClass:
    certificate: bytes
    rows: int = 0
    groups: set[str] = field(default_factory=set)
    pooled_codes: set[bytes] = field(default_factory=set)

    @property
    def possible(self):
        return len(self.groups) >= 4 and len(self.pooled_codes) >= 2


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_rows(path, fields, rows):
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def coverage(training, encoded, output):
    classes = {}
    refusals = Counter()
    feature_flags = Counter()
    change_bytes = prepare_qp_features.WIDTHS["d1024"] // 8
    with gzip.open(
        output / "training_encodings.csv.gz", "wt", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("row_id", "group_id", "exact_delta_sha256", "feature_refusal"))
        for index, (
            row,
            (encoded_index, row_id, codes, exact, flags, refusal),
        ) in enumerate(zip(training, encoded, strict=True)):
            if (encoded_index, row_id) != (index, row["row_id"]):
                raise ValueError("Training encodings are not aligned to the cohort")
            group = row["group_id"]
            if refusal:
                refusals[refusal] += 1
                writer.writerow((row_id, group, "", refusal))
                continue
            digest = hashlib.sha256(exact).hexdigest()
            item = classes.get(digest)
            if item is None:
                item = classes[digest] = ExactClass(exact)
            elif item.certificate != exact:
                raise ValueError("An indexing digest merged different signed counts")
            item.rows += 1
            item.groups.add(group)
            item.pooled_codes.add(codes["d_pooled"][change_bytes:].tobytes())
            for name, flag in zip(
                ("radicals_present", "explicit_h_changed", "charge_changed"),
                flags,
                strict=True,
            ):
                feature_flags[name] += int(flag)
            writer.writerow((row_id, group, digest, ""))
    write_rows(
        output / "exact_delta_classes.csv.gz",
        ("exact_delta_sha256", "rows", "scaffold_groups", "pooled_codes", "possible"),
        (
            {
                "exact_delta_sha256": digest,
                "rows": item.rows,
                "scaffold_groups": len(item.groups),
                "pooled_codes": len(item.pooled_codes),
                "possible": int(item.possible),
            }
            for digest, item in sorted(classes.items())
        ),
    )
    possible = [item for item in classes.values() if item.possible]
    possible_groups = set().union(*(item.groups for item in possible))
    return {
        "candidate_training_rows": len(training),
        "encoded_training_rows": sum(item.rows for item in classes.values()),
        "feature_refusals": dict(refusals),
        "encoding_flags": dict(feature_flags),
        "exact_delta_classes": len(classes),
        "possible_query_rows": sum(item.rows for item in possible),
        "possible_query_groups": len(possible_groups),
        "possible_exact_delta_classes": len(possible),
    }


def find_ecreact(synrxn):
    for relative in (
        "Data/families/representation/reaction_classification/canonical/classification/ecreact/data.csv.gz",
        "Data/classification/ecreact.csv.gz",
        "Data/tables/canonical/classification/ecreact.csv.gz",
    ):
        path = synrxn / relative
        if path.is_file():
            return path
    return None


def find_properties(synrxn):
    for relative in ("Data/property", "Data/tables/canonical/property"):
        path = synrxn / relative
        if path.is_dir():
            return path
    return None


def cohort_outputs(cohort, exclusions, output):
    # Use source indices to reconstruct labels in the later mapping stage.
    # This ledger has the same schema in every split and carries no label values.
    fields = (
        "row_id",
        "split",
        "group_id",
        "reaction_smiles",
        "source_indices",
        "source_ids",
        "representative_source_index",
        "scaffold_keys",
        "anchor_product_smiles",
    )
    write_rows(
        output / "cohort.csv.gz",
        fields,
        (
            row
            | {
                name: json.dumps(row[name], separators=(",", ":"))
                for name in ("source_indices", "source_ids", "scaffold_keys")
            }
            for row in cohort
        ),
    )
    write_rows(
        output / "exclusions.csv.gz",
        ("source_index", "source_id", "reason", "detail"),
        sorted(exclusions, key=lambda row: row["source_index"]),
    )
    groups = defaultdict(set)
    product_splits = defaultdict(set)
    for row in cohort:
        split = row["split"]
        groups[split].add(row["group_id"])
        for product in row["non_anchor_products"]:
            product_splits[product].add(split)
    if len(set().union(*groups.values())) != sum(map(len, groups.values())):
        raise ValueError("A scaffold group crosses the frozen split")
    return {
        "rows": dict(Counter(row["split"] for row in cohort)),
        "groups": {split: len(items) for split, items in groups.items()},
        "non_anchor_products_in_multiple_splits": sum(
            len(splits) > 1 for splits in product_splits.values()
        ),
    }


def run(
    source,
    ecreact,
    benchmark_root,
    properties_root,
    output,
    jobs,
    *,
    table_overrides=None,
):
    intake = audit("enzymemap", source)
    if intake["status"] != "source_intake_verified":
        raise ValueError(f"EnzymeMap intake status: {intake['status']}")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "input_manifest.json", intake)
    print("Preparing candidate EnzymeMap reactions", flush=True)
    candidates, source_audit = read_candidates(source, jobs)
    print(
        f"Auditing benchmark overlap for {len(candidates):,} candidate reactions",
        flush=True,
    )
    excluded_keys, overlap = audit_overlap(
        candidates,
        ecreact,
        benchmark_root,
        properties_root,
        jobs,
        table_overrides=table_overrides,
    )
    write_json(output / "overlap.json", overlap)
    retained = [row for row in candidates if row["reverse_key"] not in excluded_keys]
    cohort = assign_groups(retained)
    exclusions = source_audit.pop("row_exclusions")
    for row in candidates:
        if row["reverse_key"] in excluded_keys:
            exclusions.append(
                {
                    "source_index": row["representative_source_index"],
                    "source_id": row["representative_source_id"],
                    "reason": "previous_benchmark_overlap",
                    "detail": row["row_id"],
                }
            )
    distributions = cohort_outputs(cohort, exclusions, output)
    training = [row for row in cohort if row["split"] == "train"]
    print(f"Encoding {len(training):,} training reactions", flush=True)
    inputs = (
        (index, row["row_id"], row["reaction_smiles"])
        for index, row in enumerate(training)
    )
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        bound = coverage(
            training, pool.map(prepare_qp_features.encode, inputs, chunksize=64), output
        )
    group_class_gate = (
        bound["possible_query_groups"] >= MIN_GROUPS
        and bound["possible_exact_delta_classes"] >= MIN_CLASSES
    )
    if not overlap["scope_complete"]:
        decision = "provisional_overlap_incomplete"
    elif group_class_gate:
        decision = "proceed_to_label_audit"
    else:
        decision = "stop_insufficient_upper_bound"
    report = {
        "stage": "enzymemap_pre_mapping_coverage",
        "decision": decision,
        "source_audit": source_audit,
        "cohort": distributions,
        "overlap_excluded_reactions": len(candidates) - len(retained),
        "overlap_scope_complete": overlap["scope_complete"],
        "upper_bound": bound,
        "upper_bound_group_and_class_thresholds_met": group_class_gate,
        "final_training_row_fraction_minimum": 0.05,
        "group_minimum": MIN_GROUPS,
        "exact_class_minimum": MIN_CLASSES,
        "model_fits": 0,
        "worktree_commit": subprocess.check_output(
            ["git", "-C", str(STUDY), "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": platform.python_version(),
        "fingerprint_environment": prepare_qp_features.features.environment(),
        "runner_sha256": {
            name: sha256(STUDY / name)
            for name in (
                "precheck_enzymemap.py",
                "enzymemap_cohort.py",
                "enzymemap_overlap.py",
                "prepare_qp_features.py",
            )
        },
        "outputs_sha256": {
            path.name: sha256(path) for path in sorted(output.iterdir())
        },
    }
    write_json(output / "eligibility_precheck.json", report)
    print(
        f"{decision}: at most {bound['possible_query_groups']:,} query groups, "
        f"{bound['possible_exact_delta_classes']:,} exact classes",
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=STUDY / "inputs" / PINS["enzymemap"][0]
    )
    parser.add_argument(
        "--synrxn", type=Path, help="SynRXN root for pinned ECReact and property tables"
    )
    parser.add_argument(
        "--ecreact", type=Path, help="Override path to the pinned raw ECReact archive"
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        help="Root of the original benchmark plan directories",
    )
    parser.add_argument(
        "--properties-root",
        type=Path,
        help="Override path to the benchmark property tables",
    )
    parser.add_argument(
        "--table-overrides",
        type=Path,
        help="JSON mapping of audit source keys to files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for manifests and ledgers",
    )
    parser.add_argument("--jobs", type=int, default=6, help="CPU worker count")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.synrxn is not None and not args.synrxn.is_dir():
        parser.error("--synrxn must name an existing directory")
    ecreact = args.ecreact
    properties = args.properties_root
    if args.synrxn is not None:
        if ecreact is None:
            ecreact = find_ecreact(args.synrxn)
        if properties is None:
            properties = find_properties(args.synrxn)
    overrides = None
    if args.table_overrides is not None:
        paths = json.loads(args.table_overrides.read_text())
        if not isinstance(paths, dict) or not all(
            isinstance(value, str) for value in paths.values()
        ):
            parser.error("--table-overrides must contain a JSON object of file paths")
        overrides = {
            name: args.table_overrides.parent / value for name, value in paths.items()
        }
    run(
        args.source,
        ecreact,
        args.benchmark_root,
        properties,
        args.output,
        args.jobs,
        table_overrides=overrides,
    )


if __name__ == "__main__":
    main()
