"""Audit EnzymeMap reaction overlap against pinned benchmark source tables."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from itertools import chain
import json
from pathlib import Path

from audit_inputs import sha256
from audit_overlap import reaction_sides, read_rows
from enzymemap_cohort import cancel_shared, canonical_sides

PRIOR_AUDIT = (
    Path(__file__).with_name("publication_metadata")
    / "quantumpioneer_v1/audits/prior_benchmark_overlap.json"
)
RAW_ECREACT = "synrxn_classification_raw/ecreact"
SCORED_ECREACT = "all_endpoint_factorial_full_v1/ecreact"
RAW_ECREACT_SHA = "c730e5bd196dc69b5ebc421cce37010162f7e129f149b28196f0069ee242bdcd"


def _pins():
    receipt = json.loads(PRIOR_AUDIT.read_text())
    return {
        name: row["source_sha256"] for name, row in receipt["source_tables"].items()
    } | {
        RAW_ECREACT: RAW_ECREACT_SHA,
    }


def _source_locations(pins, ecreact, benchmark_root, properties_root, overrides):
    locations = {name: [] for name in pins}
    if ecreact is not None:
        locations[RAW_ECREACT].append(ecreact)
    plan_issues = {}
    if benchmark_root is not None:
        plans = {name.split("/")[0] for name in pins if not name.startswith("synrxn_")}
        for plan in sorted(plans):
            folder = benchmark_root / plan
            manifest = folder / "plan.json"
            entries = {}
            if manifest.is_file():
                try:
                    sources = json.loads(manifest.read_text())["sources"]
                    if not isinstance(sources, dict):
                        raise ValueError("sources must be a mapping")
                    entries = sources
                except (OSError, ValueError, KeyError, TypeError) as error:
                    plan_issues[plan] = str(error)
            for name in locations:
                run, key = name.split("/", 1)
                if run != plan:
                    continue
                entry = entries.get(key)
                table = entry.get("table") if isinstance(entry, dict) else None
                if isinstance(table, str) and table:
                    path = Path(table)
                    if path.is_absolute():
                        if plan in path.parts:
                            relative = path.parts[path.parts.index(plan) + 1 :]
                            locations[name].append(folder.joinpath(*relative))
                        locations[name].append(path)
                    else:
                        locations[name].append(folder / path)
                        locations[name].append(benchmark_root / path)
                locations[name].append(folder / "sources" / f"{key}.csv.gz")
    if properties_root is not None:
        for name in locations:
            if name.startswith("synrxn_property_raw/"):
                filename = name.rsplit("/", 1)[1]
                locations[name].append(properties_root / f"{filename}.gz")
    for name, path in (overrides or {}).items():
        if name not in locations:
            raise ValueError(f"Unknown historical benchmark source: {name}")
        locations[name] = [Path(path)]
    return locations, plan_issues


def _resolve_source(paths, expected):
    attempted = {}
    seen = set()
    for path in paths:
        path = path.resolve()
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            attempted[str(path)] = {"status": "missing"}
            continue
        try:
            observed = sha256(path)
        except OSError as error:
            attempted[str(path)] = {"status": "unreadable", "error": str(error)}
            continue
        if observed == expected:
            return path, {
                "expected_sha256": expected,
                "source_path": str(path),
                "source_sha256": observed,
            }
        attempted[str(path)] = {
            "status": "checksum_mismatch",
            "source_sha256": observed,
        }
    return None, {
        "status": "unavailable",
        "expected_sha256": expected,
        "attempted_paths": attempted,
    }


def _candidate_indexes(candidates):
    full = defaultdict(set)
    canceled = defaultdict(set)
    minimum = {}
    for row in candidates:
        for sides in row["observed_directions"]:
            full[sides].add(row["row_id"])
            residual = cancel_shared(sides)
            if all(residual):
                canceled[residual].add(row["row_id"])
    for left, right in chain(full, canceled):
        smaller, larger = sorted((len(left), len(right)))
        difference = larger - smaller
        minimum[difference] = min(minimum.get(difference, smaller), smaller)
    return full, canceled, minimum


def _reaction_columns(row):
    if {"left", "right"} <= row.keys():
        return {"left": "left", "right": "right"}
    for field in ("aam", "rxn", "reaction_smiles", "reaction"):
        if field in row:
            return {"reaction": field}
    raise ValueError("unrecognized reaction columns")


def _hits(index, sides, name):
    forward = index.get(sides, set())
    backward = index.get(sides[::-1], set()) - forward
    return chain(
        ((row_id, f"{name}_direction") for row_id in forward),
        ((row_id, f"{name}_reverse") for row_id in backward),
    )


def _scan_source(path, index):
    full, canceled, minimum = index
    counts = Counter()
    hits = defaultdict(Counter)
    rows = read_rows(path)
    first = next(rows, None)
    if first is None:
        raise ValueError("empty benchmark table")
    columns = _reaction_columns(first)
    for row in chain((first,), rows):
        counts["rows"] += 1
        try:
            left, right = reaction_sides(row, columns)
        except TypeError:
            left, right = "", ""
        if not left or not right:
            counts["missing_or_invalid_sides"] += 1
            continue
        smaller, larger = sorted((left.count(".") + 1, right.count(".") + 1))
        difference = larger - smaller
        # Canceling an identical component removes one occurrence from each side.
        if difference not in minimum or smaller < minimum[difference]:
            counts["component_count_mismatch"] += 1
            continue
        try:
            sides = canonical_sides(left, right)
        except (ValueError, RuntimeError):
            counts["graph_refusals"] += 1
            continue
        counts["valid_candidate_graphs"] += 1
        matches = list(_hits(full, sides, "full"))
        residual = cancel_shared(sides)
        if all(residual):
            matches.extend(_hits(canceled, residual, "canceled"))
        if matches:
            counts["matched_source_rows"] += 1
            for row_id, kind in matches:
                hits[row_id][kind] += 1
    return dict(counts), {
        row_id: dict(sorted(value.items())) for row_id, value in sorted(hits.items())
    }


def _init_index(index):
    global _WORKER_INDEX
    _WORKER_INDEX = index


def _scan_verified(entry):
    name, path = entry
    try:
        counts, hits = _scan_source(path, _WORKER_INDEX)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        return name, {"status": "scan_error", "error": str(error)}
    return name, {
        "status": "verified_scanned",
        "counts": counts,
        "matches_by_enzymemap_id": hits,
        "unresolved_rows": counts.get("missing_or_invalid_sides", 0)
        + counts.get("graph_refusals", 0),
    }


def audit_overlap(
    candidates, ecreact, benchmark_root, properties_root, jobs, *, table_overrides=None
):
    """Return canonical reversal keys to exclude and an audit of available sources."""
    pins = _pins()
    locations, plan_issues = _source_locations(
        pins,
        ecreact,
        benchmark_root,
        properties_root,
        table_overrides,
    )
    tables = {}
    scans = []
    for name, paths in locations.items():
        path, tables[name] = _resolve_source(paths, pins[name])
        if path is not None:
            scans.append((name, path))
    if scans:
        index = _candidate_indexes(candidates)
        with ProcessPoolExecutor(
            max_workers=min(jobs, len(scans)),
            initializer=_init_index,
            initargs=(index,),
        ) as pool:
            for name, report in pool.map(_scan_verified, scans):
                tables[name].update(report)
                print(
                    f"{name}: {report['status']}, "
                    f"{len(report.get('matches_by_enzymemap_id', {}))} EnzymeMap matches",
                    flush=True,
                )
    matched_ids = sorted(
        {
            row_id
            for table in tables.values()
            for row_id in table.get("matches_by_enzymemap_id", {})
        }
    )
    matched_ids_set = set(matched_ids)
    excluded = {
        row["reverse_key"] for row in candidates if row["row_id"] in matched_ids_set
    }
    missing = [
        name for name, table in tables.items() if table["status"] != "verified_scanned"
    ]
    unresolved = [
        name for name, table in tables.items() if table.get("unresolved_rows")
    ]
    return excluded, {
        "scope_complete": not missing and not unresolved,
        "ecreact_complete": not {RAW_ECREACT, SCORED_ECREACT}.intersection(
            missing + unresolved
        ),
        "historical_pins_sha256": sha256(PRIOR_AUDIT),
        "sources_verified": len(tables) - len(missing),
        "missing_or_unverified_sources": missing,
        "sources_with_unresolved_rows": unresolved,
        "plan_issues": plan_issues,
        "source_tables": tables,
        "matched_enzymemap_ids": matched_ids,
    }
