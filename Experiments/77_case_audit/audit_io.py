"""Bind the audit to the published Request 74 and 75 receipts."""
import csv
from hashlib import file_digest
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
BASE = Path.home() / ".local/share/synrfp-runs"
OUTPUT = HERE / "publication_metadata/case_audit_v1"
REQUEST74 = WORKSPACE / "SynRFP-delta-ceiling"
REQUEST75 = WORKSPACE / "SynRFP-chemistry-cases"
CASERELEASE = REQUEST75 / "Experiments/75_chemistry_cases/publication_metadata/chemistry_cases_v1"
PATHRELEASE = REQUEST74 / "Experiments/74_delta_ceiling/publication_metadata/delta_ceiling_v1"
REVISIONS = {
    "request_77": "cada2375",
    "request_75_results": "9e2c2de9",
    "request_74_results": "c2f046a9",
}


def digest(path):
    with path.open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


class Archive:
    def __init__(self):
        self.pins = {}
        self.commits = {
            key: subprocess.check_output(
                ["git", "-C", str(HERE), "rev-parse", value], text=True
            ).strip()
            for key, value in REVISIONS.items()
        }

    def use(self, path, expected=None):
        path = Path(path)
        key = str(path)
        if key not in self.pins:
            self.pins[key] = digest(path)
        if expected is not None and self.pins[key] != expected:
            raise ValueError(f"Changed archived input: {path}")
        return path

    def release(self, folder, checkout, revision, names):
        receipt = folder / "COMPLETE.json"
        committed = subprocess.check_output(
            ["git", "-C", str(checkout), "show",
             f"{self.commits[revision]}:{receipt.relative_to(checkout)}"]
        )
        self.use(receipt)
        if committed != receipt.read_bytes():
            raise ValueError(f"Changed release seal: {receipt}")
        checksums = read(receipt)["files_sha256"]
        for name in names:
            self.use(folder / name, checksums[name])
        return checksums


def export(path, records):
    fields = list(dict.fromkeys(key for record in records for key in record))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
