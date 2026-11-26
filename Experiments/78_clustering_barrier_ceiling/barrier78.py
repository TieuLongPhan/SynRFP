"""Exact WL radius-2 count classes and barrier dispersion on the nine cohorts."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from hashlib import sha256
import importlib.metadata
import importlib.util
from itertools import islice, permutations
from pathlib import Path
import struct
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from common78 import BASE, D, DU, HERE, LOCAL, ORIGINAL, PAPER_REV, REPO, Pins, digest, export, paper_table, read, write

TARGETS = ("b97xd3", "cyclo", "e2", "e2sn2", "gdb7", "rdb7", "rgd1", "sn2", "snar")
PAIR = struct.Struct("<Qi")
STATE = {}


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def signed(counts):
    return sha256(b"".join(PAIR.pack(token, value) for token, value in sorted(counts.items()))).digest()


@lru_cache(maxsize=2048)
def side_tokens(side, repaired):
    original = STATE["native"]
    bh = STATE["bh"]
    if repaired:
        from synkit.IO import smiles_to_graph
        bh._strict_molecule(side)
        graph = smiles_to_graph(side, drop_non_aam=False, sanitize=True,
                                use_index_as_atom_map=False, include_stereo_descriptors=False)
    else:
        bh._strict_molecule(side)
        graph = bh._graph(side)
    if graph is None:
        raise ValueError("Frozen barrier graph refused a source side")
    return STATE["tokenizer"].tokens_graph(original.Molecule.from_nx_graph(graph), 2)


def encode(task):
    key, start, stop = task
    frame, anchors, original_ok = STATE["frames"][key]
    result = np.empty((stop-start, 2), dtype="V32")
    F = STATE["native"]
    for offset, i in enumerate(range(start, stop)):
        left, right = frame.left.iat[i], frame.right.iat[i]
        repaired = not original_ok[i]
        l, r = side_tokens(left, repaired), side_tokens(right, repaired)
        delta, union, _ = F.token_algebra(l, r)
        d, u = signed(delta), signed(union)
        result[offset] = (d, sha256(d+u).digest())
        # Check the full published D and D+U representations on every row.
        packed_d = F._packed(l, r, "parity", 2048, "delta")
        if not np.array_equal(packed_d, anchors[0][i]):
            raise ValueError(f"Archived D replay mismatch {key}/{i}")
        d1024 = F._packed(l, r, "parity", 1024, "delta")
        u1024 = F._packed(l, r, "parity", 1024, "union")
        if not np.array_equal(np.concatenate((d1024, u1024)), anchors[1][i]):
            raise ValueError(f"Archived D+U replay mismatch {key}/{i}")
    return key, start, stop, result


def prepare(pins):
    root = BASE / "cross_task_factorial_v1"
    study30 = pins.plan(root)
    root52 = BASE / "coverage_refits_v1"
    study52 = pins.plan(root52)
    old = REPO / "Experiments/20_focused_factorial/encode_factorial.py"
    pins.use(old, study30["pinned_files"][str(ORIGINAL / old.relative_to(REPO))])
    bh = ORIGINAL / "Experiments/13_yield/bh_features.py"
    pins.use(bh, study30["pinned_files"][str(bh)])
    source = module("request78_frozen20", old)
    # The frozen parser's molecule adapter is an ignored workstation input.
    bh_module = module("request78_native_yield", bh)
    from synrfp.graph.molecule import Molecule
    source.Molecule = Molecule
    STATE.update(native=source, bh=bh_module, tokenizer=source._tokenizer("wl", source.NODE_ATTRS, source.EDGE_ATTRS))
    values, frames = {}, {}
    for key in TARGETS:
        src = study30["sources"][key]
        path = pins.use(src["table"], study30["pinned_files"][str(Path(src["table"]).resolve())])
        frame = pd.read_csv(path, usecols=["left","right","row_id",src["target"]], dtype={"row_id":str})
        y = frame[src["target"]].to_numpy(dtype=np.float64)
        assert len(y)==src["n"] and np.isfinite(y).all() and not frame.row_id.duplicated().any()
        positions = np.load(pins.use(src["population_path"], study30["pinned_files"][src["population_path"]]), allow_pickle=False)
        expected = np.zeros(len(y), dtype=bool)
        expected[positions] = True
        if key != "rgd1":
            assert expected.all()
        else:
            assert len(y)==353984 and len(positions)==353940
        packed = []
        for cfg in (D, DU):
            native = Path(src["features"][cfg])
            receipt = read(pins.use(native.with_suffix(".json")))
            assert receipt["identity"]["input_sha256"] == pins.hashes[str(path)]
            assert receipt["identity"]["plan_sha256"] == pins.hashes[str(root/"plan.json")]
            with np.load(pins.use(native, receipt["feature_sha256"]), allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["ok"], expected)
                if key != "rgd1":
                    packed.append(saved["X"])
            if key == "rgd1":
                corrected = Path(study52["features"][f"{key}__{cfg}"]["path"])
                review = read(pins.use(corrected.with_suffix(".json")))
                assert review["original_sha256"]==receipt["feature_sha256"]
                assert review["plan_sha256"]==pins.hashes[str(root52/"plan.json")]
                with np.load(pins.use(corrected, review["feature_sha256"]), allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["row_ids"], frame.row_id.to_numpy(str))
                    np.testing.assert_array_equal(saved["source_positions"], np.arange(len(y)))
                    assert saved["ok"].all()
                    packed.append(saved["X"])
        frames[key]=(frame[["left","right"]], packed, expected)
        values[key]=(src, y)
    STATE["frames"] = frames
    return values


def dispersion(labels, y):
    n = np.bincount(labels)
    means = np.bincount(labels, weights=y) / n
    residual = y - means[labels]
    sse = float(residual @ residual)
    repeated = n > 1
    total = float(np.var(y))
    dof = len(y)-len(n)
    return {
        "n_classes": len(n),
        "n_repeated_classes": int(repeated.sum()),
        "n_reactions_in_repeated_classes": int(n[repeated].sum()),
        "repeated_reaction_share": float(n[repeated].sum()/len(y)),
        "within_dof": dof,
        "within_sse": sse,
        "pooled_within_sd": (sse/dof)**0.5 if dof else None,
        "pooled_within_sd_to_dataset_sample_sd": ((sse/dof)/np.var(y, ddof=1))**0.5 if dof else None,
        "within_rms_across_all_rows": (sse/len(y))**0.5,
        "within_rms_to_dataset_sd": ((sse/len(y))/total)**0.5,
    }


def stats(keys, values, paper):
    rows = []
    for key in TARGETS:
        src, y = values[key]
        d, du = [np.unique(keys[key][:, j], return_inverse=True)[1] for j in range(2)]
        mapping = np.empty(int(du.max())+1, dtype=np.int64)
        mapping[du] = d
        np.testing.assert_array_equal(mapping[du], d)
        left, right = dispersion(d, y), dispersion(du, y)
        sst = float(((y-y.mean()) ** 2).sum())
        gain = (left["within_sse"] - right["within_sse"]) / sst
        assert -1e-12 <= gain <= 1+1e-12
        observed = paper.query("task == 'barrier' and dataset == @key")
        assert len(observed)==18 and len(observed.protocol.unique())==1 and observed.barrier_rmse_reduction_pct.notna().all()
        assert len(observed[["tokenizer","radius","sketch"]].drop_duplicates())==18
        old_ok = STATE["frames"][key][2]
        rows.append({
            "dataset":key, "n_reactions":len(y), "n_recovered_optional_registry":int((~old_ok).sum()),
            "units":src["units"], "dataset_sample_sd":float(y.std(ddof=1)),
            **{f"D_{k}":v for k,v in left.items()}, **{f"DU_{k}":v for k,v in right.items()},
            "within_variance_reduction_fraction_of_dataset":max(0., float(gain)),
            "pooled_variance_reduction_fraction_of_dataset":
                (left["pooled_within_sd"]**2-right["pooled_within_sd"]**2)/y.var(ddof=1)
                if left["within_dof"] and right["within_dof"] else None,
            "paper_protocol":observed.protocol.iloc[0],
            "published_mean_rmse_reduction_pct":float(observed.barrier_rmse_reduction_pct.mean()),
            "published_matched_constructions":18,
        })
    return rows


def correlate(rows, field):
    x = np.asarray([r[field] for r in rows], dtype=float)
    y = np.asarray([r["published_mean_rmse_reduction_pct"] for r in rows])
    mask = np.isfinite(x)
    x, y = x[mask], y[mask]
    coefficient, p = spearmanr(x, y)
    ranks_x, ranks_y = rankdata(x), rankdata(y)
    ranks_x -= ranks_x.mean()
    ranks_y -= ranks_y.mean()
    observed = abs(ranks_x@ranks_y)
    stream = permutations(ranks_y)
    exceed = total = 0
    for chunk in iter(lambda: list(islice(stream, 8192)), []):
        dot = np.abs(np.asarray(chunk)@ranks_x)
        exceed += int((dot>=observed-1e-12).sum())
        total += len(chunk)
    return {"n":len(x),"spearman_rho":float(coefficient),"asymptotic_p":float(p),
            "exact_permutation_p":exceed/total, "label_permutations":total}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--output", type=Path, default=LOCAL/"barrier")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    pins = Pins()
    paper = paper_table(pins)
    values = prepare(pins)
    keys = {key: np.empty((len(values[key][1]),2), dtype="V32") for key in TARGETS}
    jobs = [(key, start, min(start+512,len(matrix)))
            for key,matrix in keys.items() for start in range(0,len(matrix),512)]
    done = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for key,start,stop,encoded in pool.map(encode, jobs):
            keys[key][start:stop]=encoded
            done += stop-start
            if done//20000 != (done-stop+start)//20000:
                print(f"{done} barrier rows replayed", flush=True)
    rows = stats(keys, values, paper)
    export(args.output/"barrier_groups.csv", rows)
    summaries = {}
    for field in ("within_variance_reduction_fraction_of_dataset", "pooled_variance_reduction_fraction_of_dataset"):
        summaries[field] = correlate(rows, field)
    write(args.output/"barrier_AUDIT.json", {
        "request_revision":"773ba9ac4610fe573576a6fa91c9383caace8abd", "paper_revision":PAPER_REV,
        "input_sha256":pins.hashes, "code_sha256":digest(__file__),
        "full_parity_feature_replay_rows":done,
        "correlations":summaries, "fit_count":0,
        "packages":{p:importlib.metadata.version(p) for p in ("synkit","rdkit","numpy","pandas","scipy")},
        "table_sha256":digest(args.output/"barrier_groups.csv"),
    })
    print(summaries, flush=True)


if __name__ == "__main__":
    main()
