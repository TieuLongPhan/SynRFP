#!/usr/bin/env python
"""Step 4. Allocation-free locality bounds and collision load.

Part A (bounds). The archived centre audit assigns the surviving occurrences
of a token class by root order. For a class with surplus n on its surplus
side, L occurrences whose radius-k ball touches the mapped centre and U that do
not, every allocation localizes between max(0, n - U) and min(n, L)
occurrences. Summing over classes and dividing by the total surplus gives
bounds that do not depend on atom numbering. Computed for every mapped column
and radius in study.yaml (the second column carries mechanistic reagents and
therefore a different centre definition).

Part B (collisions). For the default sketch: keys per reaction, bits set, and
keys lost to within-reaction collisions, on samples of the panel endpoints.
Localization before sketching does not make a bit an atom-level explanation;
this part quantifies how many keys share a bit.

Outputs: runs/locality/bounds__<column>__r<k>.csv, bounds_class.csv,
bounds_summary.csv, collisions.csv
"""
from __future__ import annotations

import argparse
import importlib.util
from collections import deque

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import vlib as L
from synrfp_exp.config import default_config, log, setup_logging
from synrfp_exp.data import dataset_table


def _center_maps():
    spec = importlib.util.spec_from_file_location(
        "reaction_center", L.EXP / "08_reaction_center" / "reaction_center.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.center_maps          # the archived centre definition, unchanged


def bfs(adj, sources):
    dist = {s: 0 for s in sources}
    q = deque(sources)
    while q:
        u = q.popleft()
        for w in adj.get(u, []):
            if w not in dist:
                dist[w] = dist[u] + 1
                q.append(w)
    return dist


def analyse(rsmi, radius, scfg):
    from rdkit import RDLogger
    from synrfp.graph.reaction import Reaction
    from synrfp.tokenizers.wl import WLTokenizer

    RDLogger.DisableLog("rdApp.*")
    try:
        rxn = Reaction.from_rsmi(rsmi, agents=scfg["agents"])
    except Exception:
        return {"status": "unparsable"}
    G = {"R": rxn.reactant, "P": rxn.product}
    ctr = _center_maps()(G["R"], G["P"])
    if not ctr:
        return {"status": "no_centre"}
    tok = WLTokenizer(node_attrs=scfg["node_attrs"], edge_attrs=scfg["edge_attrs"])
    prov = {s: tok.tokens_with_provenance(G[s], radius) for s in G}
    dist = {s: bfs(G[s].adj, [v for v, d in G[s].nodes.items()
                              if d.get("atom_map", 0) in ctr]) for s in G}

    def touches(side, occ):
        return dist[side].get(occ[0], 10**9) <= occ[1]

    n_tot = lo = hi = det = amb = 0
    for t in prov["R"].keys() | prov["P"].keys():
        d = len(prov["P"].get(t, [])) - len(prov["R"].get(t, []))
        if d == 0:
            continue
        side = "P" if d > 0 else "R"
        occs = sorted(prov[side][t])
        n = abs(d)
        near = sum(touches(side, o) for o in occs)
        a, b = max(0, n - (len(occs) - near)), min(n, near)
        n_tot += n
        lo += a
        hi += b
        det += sum(touches(side, o) for o in occs[:n])
        amb += n if a != b else 0
    n_union = sum(len(v) for p in prov.values() for v in p.values())
    u_touch = sum(touches(s, o) for s in prov for v in prov[s].values() for o in v)
    if n_tot == 0:
        return {"status": "empty_delta", "union_fraction": u_touch / n_union}
    return {"status": "ok", "surplus": n_tot, "lower": lo / n_tot, "upper": hi / n_tot,
            "root_order": det / n_tot, "ambiguous_occurrence_fraction": amb / n_tot,
            "union_fraction": u_touch / n_union, "n_centre_atoms": len(ctr)}


def collisions(rsmis, scfg, bits, seed):
    from rdkit import RDLogger
    from synrfp.graph.reaction import Reaction
    from synrfp.tokenizers.utils import _h64
    from synrfp.tokenizers.wl import WLTokenizer

    RDLogger.DisableLog("rdApp.*")
    tok = WLTokenizer(node_attrs=scfg["node_attrs"], edge_attrs=scfg["edge_attrs"])
    r = scfg["default"]["radius"]
    rows = []
    for s in rsmis:
        try:
            rxn = Reaction.from_rsmi(s, agents=scfg["agents"])
            cR, cP = tok.tokens_graph(rxn.reactant, r), tok.tokens_graph(rxn.product, r)
        except Exception:
            continue
        keys = [2 * t + (0 if cP.get(t, 0) > cR.get(t, 0) else 1)
                for t in cR.keys() | cP.keys() if cR.get(t, 0) != cP.get(t, 0)]
        idx = np.array([_h64(("pf", int(k)), seed=seed) % bits for k in keys], dtype=np.int64)
        counts = np.bincount(idx, minlength=bits)
        rows.append({"keys": len(keys), "bits_set_parity": int((counts & 1).sum()),
                     "bits_touched": int((counts > 0).sum()),
                     "keys_sharing_a_bit": int(counts[counts > 1].sum()),
                     "keys_cancelled_by_parity": int(counts[(counts > 1) & (counts % 2 == 0)].sum())})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parts", nargs="*", default=["bounds", "collisions"])
    ap.add_argument("--sample", type=int, help="limit mapped reactions (testing only)")
    args = ap.parse_args()
    setup_logging()
    st = L.study_settings()
    cfg = L.load_policy(st["primary_policy"])
    out = L.RUNS / "locality"
    out.mkdir(parents=True, exist_ok=True)
    written = []
    if "bounds" in args.parts:
        raw = pd.read_csv(cfg["paths"]["mech_raw"])
        if args.sample:
            raw = raw.sample(n=min(args.sample, len(raw)), random_state=0)
        summ, cls = [], []
        for col in st["locality"]["mapped_columns"]:
            if col not in raw:
                log.warning("mapped column %s missing; skipped", col)
                continue
            for radius in st["locality"]["radii"]:
                res = Parallel(n_jobs=cfg["n_jobs"], batch_size=64)(
                    delayed(analyse)(s, radius, cfg["synrfp"]) for s in raw[col].astype(str))
                per = pd.DataFrame(res)
                per["class"] = raw["mechanistic_class"].astype(str).values
                path = out / f"bounds__{col}__r{radius}.csv"
                per.to_csv(path, index=False)
                written.append(path)
                ok = per[per.status == "ok"]
                summ.append({"mapped_column": col, "radius": radius, "n_input": len(per),
                             **{f"n_{k}": int(v) for k, v in per.status.value_counts().items()},
                             "mean_lower": ok.lower.mean(), "mean_root_order": ok.root_order.mean(),
                             "mean_upper": ok.upper.mean(), "median_lower": ok.lower.median(),
                             "median_upper": ok.upper.median(),
                             "mean_width": (ok.upper - ok.lower).mean(),
                             "exact_fraction": float((ok.upper == ok.lower).mean()),
                             "mean_ambiguous_occurrence_fraction":
                                 ok.ambiguous_occurrence_fraction.mean(),
                             "mean_union_fraction": ok.union_fraction.mean(),
                             "share_lower_below_half": float((ok.lower < .5).mean())})
                g = ok.groupby("class").agg(n=("lower", "size"), lower=("lower", "mean"),
                                            root_order=("root_order", "mean"),
                                            upper=("upper", "mean"),
                                            union=("union_fraction", "mean")).reset_index()
                g.insert(0, "radius", radius)
                g.insert(0, "mapped_column", col)
                cls.append(g)
                log.info("%s r=%d: bounds [%.3f, %.3f], root order %.3f, pooled %.3f", col,
                         radius, summ[-1]["mean_lower"], summ[-1]["mean_upper"],
                         summ[-1]["mean_root_order"], summ[-1]["mean_union_fraction"])
        pd.DataFrame(summ).to_csv(out / "bounds_summary.csv", index=False)
        pd.concat(cls).to_csv(out / "bounds_class.csv", index=False)
        written += [out / "bounds_summary.csv", out / "bounds_class.csv"]
    if "collisions" in args.parts:
        d = default_config(cfg)
        rows = []
        for ds in st["locality"]["collision_datasets"]:
            if not (cfg["paths"]["work"] / "tables" / f"ds_{ds}.pkl").exists():
                continue
            table = dataset_table(cfg, ds)
            rng = np.random.default_rng(0)
            idx = rng.choice(len(table), min(st["locality"]["collision_sample"], len(table)),
                             replace=False)
            rs = table["rxn"].to_numpy()[idx].tolist()
            res = Parallel(n_jobs=cfg["n_jobs"])(
                delayed(collisions)(rs[i:i + 250], cfg["synrfp"], d.bits, cfg["fp_seed"])
                for i in range(0, len(rs), 250))
            c = pd.DataFrame([r for ch in res for r in ch])
            rows.append({"dataset": ds, "n": len(c), "bits": d.bits, **c.mean().to_dict(),
                         "share_reactions_with_any_collision":
                             float((c.keys_sharing_a_bit > 0).mean()),
                         "share_keys_sharing_a_bit":
                             float(c.keys_sharing_a_bit.sum() / max(1, c["keys"].sum()))})
            log.info("%s: %s", ds, rows[-1])
        pd.DataFrame(rows).to_csv(out / "collisions.csv", index=False)
        written.append(out / "collisions.csv")
    L.write_manifest("04_locality", written)


if __name__ == "__main__":
    main()
