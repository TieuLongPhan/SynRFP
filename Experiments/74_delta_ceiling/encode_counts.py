"""Recompute unsketched occurrence count classes on frozen study 65 inputs."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import file_digest, sha256
from importlib.metadata import version
import importlib.util
import json
from pathlib import Path
import os
from struct import Struct
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from rdkit import RDLogger

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "Experiments/20_focused_factorial"))
import encode_factorial as F

TOKENIZERS = {"wl_r2": ("wl", 2), "wl_r3": ("wl", 3),
              "morgan_r2": ("morgan", 2), "path_r2": ("path", 2)}
CHANNELS = ("D", "U", "D+U", "D+S")
PAIR = Struct("<Qi")
CONTEXT = {}
RDLogger.DisableLog("rdApp.*")


def digest(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Different frozen input: {path}")
    return expected


def write(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def signed_digest(counts):
    # Fixed-width little-endian (unsigned64 token, signed32 multiplicity).
    return sha256(b"".join(PAIR.pack(token, count) for token, count in sorted(counts.items()))).digest()


def keys(left, right):
    delta = right.copy()
    delta.subtract(left)
    delta = {t: count for t, count in delta.items() if count}
    union, shared = left + right, left & right
    d, u, s = map(signed_digest, (delta, union, shared))
    return d, u, sha256(d + u).digest(), sha256(d + s).digest()


def reaction(text, repaired):
    return CONTEXT["adapter"].reaction(text, "tpl_dative") if repaired else F._reaction(text, "classification")


def encode_batch(rows):
    profiles = CONTEXT["profiles"]
    result = np.zeros((len(rows), len(profiles), len(CHANNELS)), dtype="V32")
    heavy = np.zeros(len(rows), dtype=np.int32)
    for i, (text, admitted, repaired) in enumerate(rows):
        if not admitted:
            continue
        graph = reaction(text, repaired)
        heavy[i] = sum(attr["element"] != "H" for side in (graph.reactant, graph.product) for attr in side.nodes.values())
        for j, name in enumerate(profiles):
            tok, radius = CONTEXT["tokenizers"][name]
            left, right = tok.tokens_graph(graph.reactant, radius), tok.tokens_graph(graph.product, radius)
            result[i, j] = list(keys(left, right))
    return result, heavy


def anchor_info(source, profile, old, morgan, repair):
    kind, radius = TOKENIZERS[profile]
    name = F.variant_name(kind, radius, "parity", "delta")
    if source == "tpl_b":
        path = Path(repair["features"][f"{source}__{name}"]["path"])
    else:
        plan = morgan if kind == "morgan" else old
        path = Path(plan["sources"][source]["features"][name])
    meta = read(path.with_suffix(".json"))
    return path, check(path, meta["feature_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--native-repo", type=Path, required=True)
    parser.add_argument("--morgan-module", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profiles", nargs="+", choices=TOKENIZERS, default=list(TOKENIZERS))
    parser.add_argument("--jobs", type=int, default=24)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    base = args.base.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    stage65 = base / "default_optimal_benchmark_v1"
    classified = stage65 / "native/classification/plan.json"
    anchors = base / "all_endpoint_factorial_full_v1"
    old_path = anchors / "plan.json"
    main65 = stage65 / "plan.json"
    check(main65, read(stage65 / "plan_seal.json")["sha256"])
    check(classified, read(main65)["inputs_sha256"][str(classified)])
    check(old_path, read(anchors / "plan_seal.json")["sha256"])
    old, native = read(old_path), read(classified)
    morgan_path = base / "morgan_ecfp_correction_v1/plan.json"
    repair_path = base / "coverage_refits_v1/plan.json"
    repairrows_path = base / "encoder_coverage_repair_v1/plan.json"
    morgan, repaired, repairrows = read(morgan_path), read(repair_path), read(repairrows_path)
    morgan_module = args.morgan_module.resolve()
    check(morgan_module, read(morgan_path.with_name("ROOT_REVIEW.json"))["sources_sha256"][str(morgan_module)])
    adapter = args.native_repo.resolve() / "Experiments/51_encoder_coverage_repair/adapter.py"
    lexer = adapter.parents[1] / "48_yield_input_repair/reaction_parser.py"
    code_ready = read(base / "encoder_coverage_repair_v1/CODE_READY.json")
    for path in (adapter, lexer):
        check(path, code_ready["sources_sha256"][str(path)])
    CONTEXT["adapter"] = load("delta_ceiling_repair", adapter)
    corrected = load("delta_ceiling_corrected_ecfp", morgan_module).MorganTokenizer
    if corrected.PROFILE != "synrfp-ecfp-count-v1":
        raise ValueError("Load the corrected Morgan tokenizer")
    CONTEXT["profiles"] = args.profiles
    CONTEXT["tokenizers"] = {
        profile: ((corrected(F.NODE_ATTRS, F.EDGE_ATTRS) if kind == "morgan" else
                   F._tokenizer(kind, F.NODE_ATTRS, F.EDGE_ATTRS)), radius)
        for profile, (kind, radius) in TOKENIZERS.items() if profile in args.profiles
    }
    import synkit.IO
    io_path = Path(synkit.IO.__file__).resolve()
    files = (Path(__file__), Path(F.__file__), adapter, lexer, morgan_module, io_path,
             REPO/"synrfp/graph/reaction.py", REPO/"synrfp/graph/molecule.py",
             *sorted((REPO/"synrfp/tokenizers").glob("*.py")))
    manifest = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "files_sha256": {str(p): digest(p) for p in files},
        "input_plans_sha256": {str(p): digest(p) for p in (classified, old_path, morgan_path, repair_path, repairrows_path, main65)},
        "packages": {k: version(k) for k in ("rdkit", "synkit", "numpy", "pandas", "networkx", "scikit-learn")},
        "token_profiles": args.profiles, "channels": CHANNELS,
    }
    mf = output/"ENCODING_PLAN.json"
    if mf.exists() and read(mf) != json.loads(json.dumps(manifest)):
        raise ValueError("Output contains a different capsule")
    write(mf, manifest)
    previous = None
    if args.resume_from is not None:
        previous = read(args.resume_from/"ENCODING_PLAN.json")
        for field in ("input_plans_sha256","packages","token_profiles","channels"):
            if previous[field] != json.loads(json.dumps(manifest[field])):
                raise ValueError(f"Cannot reuse earlier outputs with changed {field}")
        inherited = {k:v for k,v in previous["files_sha256"].items() if k!=str(Path(__file__))}
        current = {k:v for k,v in manifest["files_sha256"].items() if k!=str(Path(__file__))}
        if current != inherited:
            raise ValueError("Shared chemistry and tokenizer code changed")
    only_repairs = {r["source_position"]: r for r in repairrows["sources"]["tpl_b"]["repair_rows"]}
    pools = {}
    for source in sorted(native["sources"], key=lambda k: old["sources"][k]["n"]):
        started = time.monotonic()
        src = old["sources"][source]
        table = Path(src["table"])
        check(table, old["pinned_files"][str(table)])
        frame = pd.read_csv(table, dtype={"row_id":str})
        n = src["n"]
        np.testing.assert_array_equal(frame["source_position"], np.arange(n))
        if len(frame) != n or frame.row_id.isna().any():
            raise ValueError(f"Source identity changed: {source}")
        source_receipt = output/(source+".json")
        artifact = output/(source+".npz")
        if source_receipt.exists():
            record = read(source_receipt)
            check(artifact, record["artifact_sha256"])
            pools[source] = record
            continue
        if previous and (args.resume_from/source_receipt.name).is_file():
            record = read(args.resume_from/source_receipt.name)
            if record["rows"] != n or record["source_sha256"] != digest(table):
                raise ValueError("Earlier source receipt differs")
            path = args.resume_from/artifact.name
            check(path,record["artifact_sha256"])
            with np.load(path,allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["row_ids"],frame.row_id.to_numpy(str))
            os.link(path, artifact)
            record["reused_from_plan_sha256"] = digest(args.resume_from/"ENCODING_PLAN.json")
            record["reused_from_encoder_sha256"] = previous["files_sha256"][str(Path(__file__))]
            write(source_receipt,record)
            pools[source] = record
            print(f"Reused verified source {source}: {n:,} rows",flush=True)
            continue
        for part, record in only_repairs.items() if source == "tpl_b" else ():
            assert frame.row_id.iloc[part] == record["row_id"] and frame.rxn.iloc[part] == record["reaction"]
        pinned = {}
        ok = None
        for profile in args.profiles:
            path, pinned[profile] = anchor_info(source, profile, old, morgan, repaired)
            with np.load(path, allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["row_ids"].astype(str), frame.row_id.to_numpy(str))
                np.testing.assert_array_equal(saved["source_positions"], frame.source_position)
                admitted = saved["ok"]
                if ok is not None:
                    np.testing.assert_array_equal(admitted, ok)
                ok = admitted
                take = np.unique(np.r_[np.linspace(0, n-1, 257, dtype=int), list(only_repairs) if source=="tpl_b" else []]).astype(int)
                tokens, radius = CONTEXT["tokenizers"][profile]
                X = saved["X"]
                for i in take[ok[take]]:
                    r = reaction(frame.rxn.iloc[i], source=="tpl_b" and i in only_repairs)
                    a,b = tokens.tokens_graph(r.reactant, radius), tokens.tokens_graph(r.product, radius)
                    # Proof that token IDs and native preprocessing agree with the published archive.
                    np.testing.assert_array_equal(F._packed(a,b,"parity",2048,"delta"),X[i],
                                                  err_msg=f"{source}/{profile}/{i}")
        strings = list(frame.rxn)
        jobs = [
            [(strings[i], bool(ok[i]), source=="tpl_b" and i in only_repairs)
             for i in range(start, min(start+256,n))]
            for start in range(0,n,256)
        ]
        result = np.empty((n,len(args.profiles),len(CHANNELS)), dtype="V32")
        heavy = np.empty(n,dtype=np.int32)
        print(f"Encoding {source}: {int(ok.sum()):,}/{n:,} rows, {len(args.profiles)} profiles",flush=True)
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            offset = 0
            for keyed, counts in executor.map(encode_batch,jobs):
                end = offset+len(keyed)
                result[offset:end],heavy[offset:end] = keyed,counts
                offset=end
                if offset//20000 != (offset-len(keyed))//20000 or offset==n:
                    print(f"{source}: {offset:,}/{n:,}",flush=True)
        if np.any(heavy[ok] <= 0):
            raise ValueError("Encoded a row with no heavy atoms")
        tmp = output/(source+".tmp.npz")
        np.savez(tmp, keys=result, heavy=heavy, ok=ok, row_ids=frame.row_id.to_numpy(str))
        tmp.replace(artifact)
        record = {"rows":n,"encoded":int(ok.sum()),"dropped":int((~ok).sum()),
                  "artifact_sha256":digest(artifact),"source_sha256":digest(table),
                  "feature_replay_sha256":pinned,"seconds":time.monotonic()-started}
        write(source_receipt,record)
        pools[source]=record
    write(output/"ENCODED.json",{"phase":"complete","sources":pools,"plan_sha256":digest(mf)})
    print("Full source encoding complete",flush=True)


if __name__ == "__main__":
    main()
