"""Frozen author-cohort inputs and local result paths for Request 90."""
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
ORIGINAL = WORKSPACE / "SynRFP"
EXPERIMENTS = ORIGINAL / "Experiments"
BASE = Path.home() / ".local/share/synrfp-runs"
SOURCE = BASE / "specialist_references_v1/syncluster"
ROOT = BASE / "request90_syncluster_matched_v1"
OUTPUT = HERE / "publication_metadata/matched_v1"
UPSTREAM = EXPERIMENTS / "19_syncluster_reference/external/SynCluster"
AUTHOR = UPSTREAM / "clustering/scripts/data/uspto_50k.csv"
RELEASE = EXPERIMENTS / "35_specialist_references/syncluster/publication_metadata/full_reference_v1"
R86 = WORKSPACE / "SynRFP-invariance-all-fps/Experiments/86_invariance_all_fps"
TMP = Path("/var/tmp/synrfp-labhhc4-storage")
METHOD = "synrfp-wl-r2-b2048-parity-delta"
COUNT = "SynCluster_template_counts"
BASELINES = ("drfp", "crxnfp", "rdkit_diff", "rdkit_struct", "rxnfp_mlm")
N = 43381
REQUEST = "397dc658"

sys.path.insert(0, str(R86))


def sha(path):
    with Path(path).open("rb") as stream:
        h=hashlib.sha256()
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


class Pins:
    def __init__(self):
        self.files = {}

    def add(self, path, expected=None):
        path = Path(path)
        key = str(path)
        if key not in self.files:
            self.files[key] = sha(path)
        if expected is not None and expected != self.files[key]:
            raise ValueError(f"Frozen source changed: {path}")
        return path
