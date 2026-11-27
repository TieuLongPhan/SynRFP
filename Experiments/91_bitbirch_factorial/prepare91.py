"""Bind the full released population, corrected Morgan and existing feature caches."""
from hashlib import file_digest
from importlib.metadata import version
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
REPO = HERE.parents[1]
ORIGINAL = WORKSPACE / "SynRFP"
BASE = Path.home() / ".local/share/synrfp-runs"
ROOT = BASE / "request91_bitbirch_factorial_v1"
R79 = BASE / "request79_channels_v1"
STORAGE = Path("/var/tmp/synrfp-labhhc4-storage")
THRESHOLDS = tuple(range(20, 81, 5))
SEEDS = tuple(range(20260924, 20260929))
BASELINES = {"drfp": "native", "rdkit_struct": "native", "rdkit_diff": "nonzero",
             "crxnfp": "nonzero", "rxnfp_mlm": "positive"}
STANDARD = "synrfp-wl-r2-b2048-parity-delta"


def sha(path):
    with Path(path).open("rb") as f:
        return file_digest(f, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class Pins:
    def __init__(self):
        self.hashes = {}

    def bind(self, path, expected=None):
        path = Path(path)
        key = str(path)
        if key not in self.hashes:
            self.hashes[key] = sha(path)
        if expected is not None and self.hashes[key] != expected:
            raise ValueError(f"Frozen source changed: {path}")
        return path

    def plan(self, name):
        folder = BASE / name
        seal = read(self.bind(folder / "plan_seal.json"))
        path = self.bind(folder / "plan.json", seal["sha256"])
        return read(path), seal["sha256"]


def feature(pins, spec, source, name, original, corrected, repaired):
    if source == "tpl_b":
        info = repaired[0]["features"][f"{source}__{name}"]
        path = Path(info["path"])
        rec = read(pins.bind(path.with_suffix(".json")))
        assert rec["plan_sha256"] == repaired[1] and rec["phase"] == "complete"
        assert rec["feature_id"] == f"{source}__{name}"
        assert rec["old_valid_bytes_preserved"] and rec["repaired_rows"] == 3
        expected = rec["feature_sha256"]
    else:
        owner, owner_sha = corrected if spec["tokenizer"] == "morgan" else original
        entry = owner["sources"][source]
        path = Path(entry["features"][name])
        rec = read(pins.bind(entry["feature_receipt"]))
        assert rec["plan_sha256"] == owner_sha
        if spec["tokenizer"] == "morgan":
            assert rec["tokenizer_profile"] == "synrfp-ecfp-count-v1"
        expected = rec["files"][path.name]
    return dict(path=str(path), sha256=expected, conversion="native", has_row_identity=True)


def baseline(source, name, known, panels):
    if name == "crxnfp":
        path = STORAGE / "runs/crxnfp_full_v1/encoder/features" / f"{source}.npz"
    else:
        path = STORAGE / "work/validation/n1/features" / source / f"{name}.npz"
    hashes = {known["panels"][p["endpoint"]]["baseline_feature_sha256"][name] for p in panels}
    assert len(hashes) == 1
    return dict(path=str(path), sha256=hashes.pop(), conversion=BASELINES[name], has_row_identity=False)


def geometry(pins, source, info, plan, known):
    folder = R79 / source
    table = plan["sources"][source]["table"]
    pins.bind(table, plan["pinned_files"][table])
    ids = pd.read_csv(table, usecols=["row_id"], dtype={"row_id": str}).row_id.to_numpy(str)
    assert len(ids) == plan["sources"][source]["n"]
    result = {}
    for desc in info["panels"]:
        panel, group, n = (desc["endpoint"], desc["group"], desc["n_rows"])
        entry = plan["panels"][panel]
        table = entry["table"]
        frame = pd.read_csv(pins.bind(table, plan["pinned_files"][table]), dtype={"row_id": str})
        assert len(frame) == n == known["panels"][panel]["n_rows"]
        rows = frame.source_position.to_numpy()
        np.testing.assert_array_equal(frame.row_id.to_numpy(str), ids[rows])
        _, labels = np.unique(frame.y, return_inverse=True)
        row_file = f"rows_{group}"
        label_file = f"labels_{panel}"
        for key, expected in ((row_file, rows), (label_file, labels)):
            path = pins.bind(folder / f"{key}.npy", info["matrices"][key])
            np.testing.assert_array_equal(np.load(path, allow_pickle=False), expected)
        assert np.unique(rows).size == n
        group_spec = result.setdefault(group, dict(n=n, rows=str(folder/f"{row_file}.npy"), endpoints={}))
        assert group_spec["n"] == n
        group_spec["endpoints"][panel] = str(folder / f"{label_file}.npy")
    ids_path = ROOT / f"{source}_row_ids.npy"
    np.save(ids_path, ids, allow_pickle=False)
    pins.bind(ids_path)
    return result, ids_path, len(ids)


def crxnfp_support(pins, info):
    """The requested nonzero conversion is meaningful only if it can vary by reaction."""
    path = pins.bind(info["path"], info["sha256"])
    with np.load(path, allow_pickle=False) as z:
        ok, embedding = z["ok"], z["X"]
        assert len(ok) == len(embedding) and np.isfinite(embedding).all()
        bits = np.packbits(embedding[ok] != 0, axis=1)
        patterns, counts = np.unique(bits, axis=0, return_counts=True)
        return dict(valid_rows=int(ok.sum()), width=embedding.shape[1],
                    all_one_rows=int(np.count_nonzero(np.all(bits == 255, axis=1))),
                    unique_valid_patterns=len(patterns), modal_pattern_share=float(counts.max()/len(bits)))


def main():
    ROOT.mkdir(exist_ok=True)
    pins = Pins()
    original = pins.plan("all_endpoint_factorial_full_v1")
    corrected = pins.plan("morgan_ecfp_correction_v1")
    repaired = pins.plan("coverage_refits_v1")
    release, _ = pins.plan("default_optimal_benchmark_v1")
    native = BASE / "default_optimal_benchmark_v1/native/clustering/plan.json"
    known = read(pins.bind(native, release["inputs_sha256"][str(native)]))
    r79 = read(pins.bind(R79 / "prepared.json"))
    plan = original[0]
    assert plan["row_cap"] is None and plan["fingerprint_seed"] == 1
    assert len(plan["configs"]) == 72 and len(plan["clustering_datasets"]) == 12
    assert set(plan["clustering_datasets"]) == set(known["panels"])
    assert len(corrected[0]["configs"]) == 24
    sources = {}
    audit = {}
    for source, spec in r79["sources"].items():
        groups, ids, n = geometry(pins, source, spec, plan, known)
        features = {name: feature(pins, c, source, name, original, corrected, repaired)
                    for name, c in plan["configs"].items()}
        for name in BASELINES:
            features[name] = baseline(source, name, known, spec["panels"])
        audit[source] = crxnfp_support(pins, features["crxnfp"])
        sources[source] = dict(n=n, feature_files=features, row_ids=str(ids), groups=groups)
        print("Bound", source, n, audit[source], flush=True)
    skip = {}
    if all(row["unique_valid_patterns"] == 1 and row["all_one_rows"] == row["valid_rows"]
           for row in audit.values()):
        skip["crxnfp"] = "Every valid continuous embedding maps to the same all-one vector under the nonzero rule."
        for source in sources.values():
            del source["feature_files"]["crxnfp"]
    n_rep = len(next(iter(sources.values()))["feature_files"])
    populations = sum(len(x["groups"]) for x in sources.values())
    endpoints = sum(len(g["endpoints"]) for x in sources.values() for g in x["groups"].values())
    assert populations == 10 and endpoints == 12
    packages = {p: version(p) for p in ("bblean","numpy","scipy","scikit-learn","pandas")}
    native_code = REPO / "Experiments/synrfp_exp/clustering.py"
    match = WORKSPACE / "SynRFP-clustering-channels/Experiments/synrfp_exp/clustering.py"
    pins.bind(native_code, sha(match))
    pins.bind(REPO / "Experiments/synrfp_exp/features.py")
    pins.bind(ORIGINAL / "Experiments/47_morgan_all_tasks/factor_analysis.py")
    grid = len(THRESHOLDS) * len(SEEDS) * n_rep
    write(ROOT / "PREPARED.json", dict(
        request="18ceada956f6c0ac2fda4b085b4d09584386e3f3", sources=sources,
        pinned_inputs=pins.hashes, skips=skip, crxnfp_binarization_audit=audit, versions=packages,
        thresholds=list(THRESHOLDS), seeds=list(SEEDS), branching_factor=50, merge_criterion="diameter",
        n_populations=populations, n_endpoints=endpoints, n_representations=n_rep,
        expected_fits=populations*grid, expected_scores=endpoints*grid,
        representation_specs=plan["configs"]))
    print("Frozen", n_rep, "representations and", populations*grid, "fits", flush=True)


if __name__ == "__main__":
    main()
