"""Frozen workstation input and aggregate-output helpers for Request 78."""
import csv
from hashlib import file_digest
import io
import json
import os
from pathlib import Path
import subprocess

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parent
ORIGINAL = WORKSPACE / "SynRFP"
BASE = Path.home() / ".local/share/synrfp-runs"
LOCAL = BASE / "request78_v1"
OUTPUT = HERE / "publication_metadata/clustering_barrier_ceiling_v1"
PAPER = WORKSPACE / "SynRFP-paper-attribution"
PAPER_REV = "ac83ccfe58088a9eea0a16547876b37840547098"
PAPER_TABLE = "data/benchmark_refresh/source_data/context_score_changes.csv"
D = "synrfp-wl-r2-b2048-parity-delta"
DU = "synrfp-wl-r2-b2048-parity-delta_union"


def digest(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, payload):
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


class Pins:
    def __init__(self):
        self.hashes = {}

    def use(self, path, expected=None):
        key = str(path)
        if key not in self.hashes:
            self.hashes[key] = digest(path)
        if expected is not None and self.hashes[key] != expected:
            raise ValueError(f"Archived input changed: {path}")
        return Path(path)

    def plan(self, root):
        seal = read(self.use(root / "plan_seal.json"))
        return read(self.use(root / "plan.json", seal["sha256"]))


def paper_table(pins):
    from hashlib import sha256
    contents = subprocess.check_output(["git", "-C", str(PAPER), "show", f"{PAPER_REV}:{PAPER_TABLE}"])
    pins.hashes[f"git:{PAPER_REV}:{PAPER_TABLE}"] = sha256(contents).hexdigest()
    return pd.read_csv(io.BytesIO(contents))


def export(path, records):
    fields = list(dict.fromkeys(k for row in records for k in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
