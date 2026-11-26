"""Run full-cohort BitBIRCH with five matched input orders per construction."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from rdkit import RDLogger
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REQUEST78 = REPO.parent / "SynRFP-clustering-barrier-ceiling"
RELEASE78 = "4ba01b4bbfde4bc01c68a1efe1618c6392ac79f4"
REP78 = REQUEST78 / "Experiments/78_clustering_barrier_ceiling"
sys.path[:0] = [str(REP78), str(REPO / "Experiments/20_focused_factorial"),
                str(REPO / "Experiments"), str(REPO)]

from common78 import BASE, D, DU, Pins, digest, read, write
from clustering78 import endpoints, features
import encode_factorial as native
from synrfp_exp.clustering import bitbirch
from synrfp_exp.features import Features

U = "synrfp-wl-r2-b2048-parity-union"
ANCHORS = {"d2048": D, "u2048": U, "du1024_1024": DU}
ALLOCATIONS = {"du1536_512": (1536, 512), "du1792_256": (1792, 256)}
THRESHOLD = .35
SEEDS = tuple(range(20260924, 20260929))
ROOT = BASE / "request79_channels_v1"
REPAIR = REPO.parent / "SynRFP/Experiments/51_encoder_coverage_repair/adapter.py"
PATCH_PLAN = BASE / "encoder_coverage_repair_v1/pilots/wl_path/tpl_b.npz"
RDLogger.DisableLog("rdApp.*")
TOKENIZER = native._tokenizer("wl", native.NODE_ATTRS, native.EDGE_ATTRS)


def source_setup():
    pins = Pins()
    for file in ("common78.py", "clustering78.py", "packed_init78.py"):
        relative = (REP78 / file).relative_to(REQUEST78)
        expected = subprocess.check_output(
            ["git", "-C", str(REQUEST78), "show", f"{RELEASE78}:{relative}"]
        )
        assert (REQUEST78 / relative).read_bytes() == expected
        pins.use(REQUEST78 / relative)
    root22 = BASE / "all_endpoint_factorial_full_v1"
    root65 = BASE / "default_optimal_benchmark_v1"
    plan = pins.plan(root22)
    repair = pins.plan(BASE / "coverage_refits_v1")
    main = pins.plan(root65)
    known = root65 / "native/clustering/plan.json"
    known = read(pins.use(known, main["inputs_sha256"][str(known)]))
    assert plan["row_cap"] is None and plan["fingerprint_seed"] == 1
    assert len(plan["clustering_datasets"]) == 12
    return pins, plan, repair, known


def pooled(pins, plan, repair, source, ids, ok):
    src = plan["sources"][source]
    original = Path(src["features"][U])
    if source == "tpl_b":
        entry = repair["features"][f"{source}__{U}"]
        path = Path(entry["path"])
        receipt = read(pins.use(path.with_suffix(".json")))
        assert receipt["feature_id"] == f"{source}__{U}"
        assert receipt["plan_sha256"] == pins.hashes[str(BASE / "coverage_refits_v1/plan.json")]
        pins.use(original, entry["original_sha256"])
        expected = receipt["feature_sha256"]
    else:
        path = original
        receipt = read(pins.use(src["feature_receipt"]))
        expected = receipt["files"][path.name]
    with np.load(pins.use(path, expected), allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["row_ids"], ids)
        np.testing.assert_array_equal(archive["source_positions"], np.arange(len(ids)))
        np.testing.assert_array_equal(archive["ok"], ok)
        matrix = archive["X"]
    assert matrix.shape == (len(ids), 256) and matrix.dtype == np.uint8 and not matrix[~ok].any()
    return matrix


def save(path, matrix):
    temp = path.with_suffix(".tmp.npy")
    np.save(temp, matrix, allow_pickle=False)
    os.replace(temp, path)


def prepare():
    pins, plan, repair, known = source_setup()
    receipt = ROOT / "prepared.json"
    if receipt.exists():
        return
    sources = {}
    for source in sorted({plan["panels"][e]["source"] for e in plan["clustering_datasets"]}):
        ids, packed = features(pins, plan, repair, source)
        d, ok, _ = packed[D]
        du = packed[DU][0]
        u = pooled(pins, plan, repair, source, ids, ok)
        np.testing.assert_array_equal(u[:, :128] ^ u[:, 128:], du[:, 128:])
        folder = ROOT / source
        folder.mkdir(exist_ok=True)
        matrices = dict(d2048=d, u2048=u, du1024_1024=du, ok=ok)
        panels = []
        row_sets = {}
        for endpoint, (rows, labels, _, _) in endpoints(pins, plan, known, source, ids).items():
            group = next((name for name, other in row_sets.items() if np.array_equal(rows, other)), endpoint)
            row_sets[group] = rows
            matrices[f"rows_{group}"] = rows
            matrices[f"labels_{endpoint}"] = labels
            panels.append(dict(endpoint=endpoint, group=group, n_rows=len(rows),
                               n_refusal_placeholders=int((~ok[rows]).sum())))
        for key, matrix in matrices.items():
            save(folder / f"{key}.npy", matrix)
        table = Path(plan["sources"][source]["table"])
        sources[source] = dict(panels=panels, table=str(table), table_sha256=pins.hashes[str(table)],
                               matrices={k: digest(folder / f"{k}.npy") for k in matrices})
        print("Prepared", source, len(ids), flush=True)
    key = f"tpl_b__{DU}"
    pins.use(PATCH_PLAN, repair["pinned_files"][str(PATCH_PLAN)])
    pins.use(REPAIR)
    pins.use(REPAIR.parents[1] / "48_yield_input_repair/reaction_parser.py")
    pins.use(native.__file__)
    for file in ("synrfp/encode/core.py", "synrfp/encode/engine.py", "synrfp/tokenizers/wl.py",
                 "synrfp/tokenizers/utils.py", "synrfp/sketchers/parity_fold.py", "synrfp/graph/reaction.py",
                 "Experiments/synrfp_exp/clustering.py"):
        pins.use(REPO / file)
    write(receipt, dict(sources=sources, inputs_sha256=pins.hashes, request78_revision=RELEASE78,
                        correction_plan_entry=repair["features"][key]))


def load(folder, name):
    return np.load(folder / f"{name}.npy", mmap_mode="r", allow_pickle=False)


def encode(job):
    folder, start, texts, patched = job
    n = len(texts)
    arrays = {k: np.zeros((n, 256), dtype=np.uint8) for k in ALLOCATIONS}
    d, u, ok = [load(folder, k)[start:start+n] for k in ("d2048", "u2048", "ok")]
    for i in np.flatnonzero(ok):
        text = texts[i]
        if i + start in patched:
            from importlib.util import module_from_spec, spec_from_file_location
            spec = spec_from_file_location("repair51", REPAIR)
            adapter = module_from_spec(spec)
            spec.loader.exec_module(adapter)
            reaction = adapter.reaction(text, "tpl_dative")
        else:
            reaction = native._reaction(text, "classification")
        r = TOKENIZER.tokens_graph(reaction.reactant, 2)
        p = TOKENIZER.tokens_graph(reaction.product, 2)
        for anchor, mode in ((d, "delta"), (u, "union")):
            if not np.array_equal(anchor[i], native._packed(r, p, "parity", 2048, mode)):
                raise ValueError(f"Native anchor mismatch at {folder.name} row {start+i} ({mode})")
        u512 = u[i].reshape(4, 64)
        u512 = np.bitwise_xor.reduce(u512, axis=0)
        u256 = u512[:32] ^ u512[32:]
        for (name, (dw, uw)), ubits in zip(ALLOCATIONS.items(), (u512, u256), strict=True):
            if i % 997 == 0:
                np.testing.assert_array_equal(ubits, native._packed(r, p, "parity", uw, "union"))
            arrays[name][i, :dw//8] = native._packed(r, p, "parity", dw, "delta")
            arrays[name][i, dw//8:] = ubits
    return start, arrays


def encode_all(workers):
    config = read(ROOT / "prepared.json")
    with np.load(PATCH_PLAN, allow_pickle=False) as patch:
        patched = frozenset(map(int, patch["source_positions"]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for source, info in config["sources"].items():
            folder = ROOT / source
            ready = folder / "encoded.json"
            if ready.exists():
                for k, expected in read(ready)["matrices"].items():
                    assert digest(folder / f"{k}.npy") == expected
                continue
            assert digest(info["table"]) == info["table_sha256"]
            texts = pd.read_csv(info["table"], usecols=["rxn"], keep_default_na=False).rxn.to_numpy()
            assert len(texts) == len(load(folder, "ok"))
            start_all = time.monotonic()
            matrices = {k: np.zeros((len(texts), 256), dtype=np.uint8) for k in ALLOCATIONS}
            stage = [pool.submit(encode, (folder, start, texts[start:start+500],
                                          patched if source == "tpl_b" else ()))
                     for start in range(0, len(texts), 500)]
            for future in as_completed(stage):
                start, batches = future.result()
                for k, arr in batches.items():
                    matrices[k][start:start+len(arr)] = arr
            for k, arr in matrices.items():
                save(folder / f"{k}.npy", arr)
            write(ready, dict(matrices={k: digest(folder / f"{k}.npy") for k in matrices},
                              n_rows=len(texts), n_encoded=int(load(folder, "ok").sum()),
                              elapsed_seconds=time.monotonic()-start_all))
            print("Encoded", source, len(texts), round(time.monotonic()-start_all, 1), flush=True)


def cluster_one(job):
    folder, group, panels, construction, seed, packed_sha = job
    output = folder / f"{group}__{construction}__{seed}.json"
    if output.exists():
        assert read(output)["packed_sha256"] == packed_sha
        return
    rows = load(folder, f"rows_{group}")
    order = np.random.default_rng(seed).permutation(len(rows))
    take = rows[order]
    x = load(folder, construction)[take]
    ok = load(folder, "ok")[take]
    start = time.monotonic()
    assignment = bitbirch(Features(x, 2048, ok), THRESHOLD, 50, "diameter")
    runtime = time.monotonic() - start
    counts = np.bincount(assignment)
    cluster_count = int(np.count_nonzero(counts))
    singleton = int(np.count_nonzero(counts == 1))
    scores = []
    for panel in panels:
        y = load(folder, f"labels_{panel['endpoint']}")[order]
        ami = float(adjusted_mutual_info_score(y, assignment))
        ari = float(adjusted_rand_score(y, assignment))
        scores.append(dict(**panel, construction=construction, order_seed=seed, threshold=THRESHOLD,
                           n_clusters=cluster_count, singleton_share=singleton/len(rows),
                           ami=ami, ari=ari, runtime_seconds=runtime))
    write(output, dict(packed_sha256=packed_sha, scores=scores))
    print(group, construction, seed, "runtime", round(runtime, 2), flush=True)


def cluster_all(workers, new, ready_only):
    jobs = []
    for source, info in read(ROOT / "prepared.json")["sources"].items():
        folder = ROOT / source
        if new and ready_only and not (folder / "encoded.json").exists():
            continue
        hashes = read(folder / "encoded.json")["matrices"] if new else info["matrices"]
        for construction in (ALLOCATIONS if new else ANCHORS):
            assert digest(folder / f"{construction}.npy") == hashes[construction]
            for group in sorted({p["group"] for p in info["panels"]}):
                panels = [p for p in info["panels"] if p["group"] == group]
                for seed in SEEDS:
                    jobs.append((folder, group, panels, construction, seed, hashes[construction]))
    jobs.sort(key=lambda job: -job[2][0]["n_rows"])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(cluster_one, job) for job in jobs]):
            future.result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "anchors", "encode", "unequal"))
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--ready-only", action="store_true")
    args = parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    if args.stage == "prepare":
        prepare()
    elif args.stage == "encode":
        encode_all(args.jobs)
    else:
        cluster_all(args.jobs, new=args.stage == "unequal", ready_only=args.ready_only)


if __name__ == "__main__":
    main()
