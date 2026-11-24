"""Encode QuantumPioneer reaction sides before opening external target scores."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

import numpy as np
from rdkit import Chem, RDLogger

from audit_inputs import PINS, sha256
from paired_analysis import signed_key

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "73_context_attribution"))
import features
from synrfp.encode.core import fingerprint_from_tokens
from synrfp.graph.reaction import Reaction

ARMS = ("d1024", "d2048", "d_pooled", "d_rehash", "reactant_context")
WIDTHS = dict.fromkeys(ARMS, 2048) | {"d1024": 1024}
RDLogger.DisableLog("rdApp.*")


@lru_cache(maxsize=4096)
def component(smiles):
    raw_params = Chem.SmilesParserParams()
    raw_params.removeHs = False
    raw = Chem.MolFromSmiles(smiles, raw_params)
    if raw is None:
        raise ValueError("invalid raw component")
    canonical = features.canonical(smiles)
    graph = Reaction.from_rsmi(f"{canonical}>>{canonical}").reactant
    charges_before = Counter(atom.GetFormalCharge() for atom in raw.GetAtoms() if atom.GetFormalCharge())
    if any("charge" not in node for node in graph.nodes.values()):
        raise ValueError("charge attributes missing from parsed graph")
    charges_after = Counter(node["charge"] for node in graph.nodes.values() if node["charge"])
    flags = (
        any(atom.GetNumRadicalElectrons() for atom in raw.GetAtoms()),
        sum(atom.GetAtomicNum() == 1 for atom in raw.GetAtoms()) !=
        sum(node["element"] == "H" for node in graph.nodes.values()),
        charges_before != charges_after,
    )
    return features.TOKENIZER.tokens_graph(graph, 2), flags


def encode(row):
    index, row_id, reaction = row
    try:
        left, agent, right = reaction.split(">")
        if not left or agent or not right:
            raise ValueError("unexpected empty side or agents")
        counts = []
        flags = np.zeros(3, dtype=bool)
        for side in (left, right):
            tokens = Counter()
            for smiles in side.split("."):
                counts_within, observed = component(smiles)
                tokens.update(counts_within)
                flags |= observed
            counts.append(tokens)
        vectors = features.fingerprints(*counts)
        reactant = fingerprint_from_tokens(counts[0], Counter(), mode="union",
                                           bits=1024, sketch="parity", seed=1)
        vectors["reactant_context"] = np.concatenate((vectors["d1024"], reactant))
        if any(not np.array_equal(vectors[arm][:1024], vectors["d1024"])
               for arm in ("d_pooled", "d_rehash", "reactant_context")):
            raise RuntimeError("Broken narrow-delta block")
        exact = json.dumps(signed_key(*counts), separators=(",", ":")).encode()
        bit_vectors = {arm: np.packbits(vectors[arm]) for arm in ARMS}
        return index, row_id, bit_vectors, exact, flags.tolist(), ""
    except (ValueError, TypeError, RuntimeError) as error:
        return index, row_id, None, None, None, f"{type(error).__name__}: {str(error)[:160]}"


def read_reactions(source):
    with source.open(newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            yield index, row["source_record_id"], row["reaction_smiles"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    expected_source = PINS["quantumpioneer"]
    if sha256(args.source) != expected_source[1]:
        raise ValueError("Expected the pinned QuantumPioneer projection")
    args.output.mkdir(parents=True, exist_ok=False)
    n = expected_source[2]
    arrays = {arm: np.lib.format.open_memmap(args.output / f"{arm}.npy",
              mode="w+", dtype=np.uint8, shape=(n, WIDTHS[arm] // 8)) for arm in ARMS}
    keys = args.output / "exact_delta_certificates.sqlite"
    ledger = args.output / "feature_rows.csv.gz"
    counts = Counter()
    with (sqlite3.connect(keys) as db, gzip.open(ledger, "wt", newline="") as handle,
          ProcessPoolExecutor(max_workers=args.jobs) as pool):
        db.execute("CREATE TABLE exact_delta (sha256 TEXT PRIMARY KEY, certificate BLOB NOT NULL)")
        writer = csv.writer(handle)
        writer.writerow(("matrix_row", "row_id", "exact_delta_sha256", "feature_refusal"))
        for index, row_id, vectors, exact, flags, refusal in pool.map(
            encode, read_reactions(args.source), chunksize=128
        ):
            counts["source_rows"] += 1
            if vectors is None:
                writer.writerow((index, row_id, "", refusal))
                counts["feature_refusals"] += 1
                continue
            digest = hashlib.sha256(exact).hexdigest()
            result = db.execute("SELECT certificate FROM exact_delta WHERE sha256=?", (digest,)).fetchone()
            if result is None:
                db.execute("INSERT INTO exact_delta VALUES (?, ?)", (digest, exact))
            elif result[0] != exact:
                raise ValueError("An indexing digest merged distinct exact signed counts")
            for arm in ARMS:
                arrays[arm][index] = vectors[arm]
            writer.writerow((index, row_id, digest, ""))
            counts["encoded_rows"] += 1
            for key, present in zip(("rows_radicals_omitted_as_explicit_label",
                                     "rows_explicit_H_node_count_changed",
                                     "rows_nonzero_formal_charge_histogram_changed"), flags, strict=True):
                counts[key] += present
            if counts["source_rows"] % 25000 == 0:
                print(counts["source_rows"], "of", n, "rows examined", flush=True)
        counts["exact_delta_classes"] = db.execute("SELECT count(*) FROM exact_delta").fetchone()[0]
    if counts["source_rows"] != n:
        raise ValueError("Unexpected source row count")
    for array in arrays.values():
        array.flush()
    artifacts = [keys, ledger, *(args.output / f"{arm}.npy" for arm in ARMS)]
    report = {
        "phase": "target_free_features_complete",
        "source_sha256": expected_source[1], "runner_sha256": sha256(Path(__file__)),
        "e1_feature_code_sha256": sha256(Path(features.__file__)),
        "widths": WIDTHS, "counts": dict(counts),
        "unsigned_flags": "Radical count is absent from WL node labels; H and charge checks compare parsed atom resources",
        "outputs_sha256": {p.name: sha256(p) for p in artifacts},
    }
    (args.output / "COMPLETE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
