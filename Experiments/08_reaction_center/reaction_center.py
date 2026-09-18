#!/usr/bin/env python
"""E8. Does the signed delta localise the reaction centre without atom maps?

SynRFP never sees atom maps. Here the atom-mapped mechanistic reactions are
used only as ground truth: the reaction centre is the set of mapped atoms
whose bonds (or H count / charge) change. For every WL token the atoms that
emitted it are known (``tokens_with_provenance``), so we can ask whether the
tokens that survive the signed delta are rooted within their own radius of the
reaction centre, and compare with all tokens of the total-count summary.

Per reaction:
  delta_center_fraction   |delta|-weighted share of delta tokens whose ball
                          (radius k around the root) contains a centre atom
  union_center_fraction   the same share over all tokens of both sides
  center_coverage         share of centre atoms inside at least one delta ball
  delta_token_fraction    number of delta tokens / number of union tokens

Interpretation table: for each mechanistic class, the signed delta tokens most
enriched in that class, drawn as the SMILES of their atom environment.

Outputs: 08_reaction_center/results/{per_reaction.csv,summary.csv,class_tokens.csv}
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Experiments/
import argparse  # noqa: E402
from collections import Counter, defaultdict, deque  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from synrfp_exp.config import load_config, log, results_dir, setup_logging  # noqa: E402
from synrfp_exp.data import reaction_table  # noqa: E402
import synrfp_exp.features  # noqa: E402,F401  (puts the local synrfp on sys.path)


def bfs_dist(adj, sources):
    dist = {s: 0 for s in sources}
    q = deque(sources)
    while q:
        u = q.popleft()
        for w in adj.get(u, []):
            if w not in dist:
                dist[w] = dist[u] + 1
                q.append(w)
    return dist


def center_maps(R, P):
    """Atom-map numbers of the reaction centre."""
    def bonds(G):
        out = {}
        for u in G.nodes:
            for w in G.adj.get(u, []):
                mu, mw = G.nodes[u].get("atom_map", 0), G.nodes[w].get("atom_map", 0)
                if u < w:
                    key = tuple(sorted((mu or -u, mw or -w)))
                    out[key] = G.edge_attr(u, w).get("order")
        return out

    def atoms(G):
        return {d.get("atom_map", 0): (d.get("hcount"), d.get("charge"))
                for _, d in G.nodes.items() if d.get("atom_map", 0)}

    br, bp = bonds(R), bonds(P)
    changed = {k for k in br.keys() | bp.keys() if br.get(k) != bp.get(k)}
    ctr = {m for k in changed for m in k if m > 0}
    ar, apd = atoms(R), atoms(P)
    ctr |= {m for m in ar.keys() & apd.keys() if ar[m] != apd[m]}
    return ctr


def fragment(side_smiles, root, k, adj):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(side_smiles)
    if mol is None:
        return ""
    for a in mol.GetAtoms():
        a.SetAtomMapNum(0)
    ball = [v for v, d in bfs_dist(adj, [root]).items() if d <= k]
    try:
        return Chem.MolFragmentToSmiles(mol, atomsToUse=[v - 1 for v in ball])
    except Exception:
        return ""


def analyse(rsmi, radius, scfg):
    from rdkit import RDLogger
    from synrfp.graph.reaction import Reaction
    from synrfp.tokenizers.wl import WLTokenizer

    RDLogger.DisableLog("rdApp.*")

    try:
        rxn = Reaction.from_rsmi(rsmi, agents=scfg["agents"])
    except Exception:
        return None
    R, P = rxn.reactant, rxn.product
    ctr = center_maps(R, P)
    if not ctr:
        return None
    tok = WLTokenizer(node_attrs=scfg["node_attrs"], edge_attrs=scfg["edge_attrs"])
    prov = {"R": tok.tokens_with_provenance(R, radius), "P": tok.tokens_with_provenance(P, radius)}
    graphs = {"R": R, "P": P}
    dist = {}
    for s, G in graphs.items():
        src = [v for v, d in G.nodes.items() if d.get("atom_map", 0) in ctr]
        dist[s] = bfs_dist(G.adj, src)
    far = 10**9

    def touches(side, occ):
        return dist[side].get(occ[0], far) <= occ[1]

    delta_w = touch_w = 0.0
    covered = set()
    signed = {}
    for t in prov["R"].keys() | prov["P"].keys():
        d = len(prov["P"].get(t, [])) - len(prov["R"].get(t, []))
        if d == 0:
            continue
        side = "P" if d > 0 else "R"
        occs = prov[side][t]
        delta_w += abs(d)
        if any(touches(side, o) for o in occs):
            touch_w += abs(d)
        G = graphs[side]
        for root, k in occs:
            for v, dd in bfs_dist(G.adj, [root]).items():
                if dd <= k and G.nodes[v].get("atom_map", 0) in ctr:
                    covered.add(G.nodes[v]["atom_map"])
        root, k = min(occs, key=lambda o: (o[1], o[0]))
        signed[(t, "+" if d > 0 else "-")] = (side, root, k)
    n_union = sum(len(v) for p in prov.values() for v in p.values())
    u_touch = sum(touches(s, o) for s in prov for v in prov[s].values() for o in v)
    return {
        "delta_center_fraction": touch_w / delta_w if delta_w else np.nan,
        "union_center_fraction": u_touch / n_union if n_union else np.nan,
        "center_coverage": len(covered) / len(ctr),
        "delta_token_fraction": delta_w / n_union if n_union else np.nan,
        "n_center_atoms": len(ctr),
        "signed": signed,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config")
    args = ap.parse_args()
    setup_logging()
    cfg = load_config(args.config)
    rc = cfg["reaction_center"]
    rt = reaction_table(cfg, "mech")
    if rc["sample"]:
        rt = rt.sample(n=min(rc["sample"], len(rt)), random_state=0)
    rsmis = rt["rxn_mapped"].tolist()
    log.info("analysing %d mapped reactions at radius %d", len(rsmis), rc["radius"])
    res = Parallel(n_jobs=cfg["n_jobs"], batch_size=64)(
        delayed(analyse)(s, rc["radius"], cfg["synrfp"]) for s in rsmis)

    out = results_dir("08_reaction_center")
    keep = [i for i, r in enumerate(res) if r is not None]
    per = pd.DataFrame([{k: v for k, v in res[i].items() if k != "signed"} for i in keep])
    per["level2"] = rt["level2"].to_numpy()[keep]
    per.to_csv(out / "per_reaction.csv", index=False)
    summ = per.drop(columns="level2").describe().T
    summ.to_csv(out / "summary.csv")
    log.info("analysed %d/%d reactions\n%s", len(keep), len(res), summ[["mean", "50%"]])

    # class-enriched signed tokens
    from rdkit import RDLogger
    from synrfp.graph.reaction import Reaction

    RDLogger.DisableLog("rdApp.*")
    classes = per["level2"].to_numpy()
    n_cls = Counter(classes)
    support = defaultdict(Counter)
    example = {}
    for i, c in zip(keep, classes):
        for key, val in res[i]["signed"].items():
            support[key][c] += 1
            example.setdefault(key, (i, val))
    total = Counter({k: sum(v.values()) for k, v in support.items()})
    rows = []
    n_all = len(classes)
    for c, nc in n_cls.items():
        cand = []
        for key, cnt in support.items():
            a = cnt.get(c, 0)
            if a / nc < 0.3:
                continue
            b = total[key] - a
            lo = np.log((a + 0.5) / (nc - a + 0.5)) - np.log((b + 0.5) / (n_all - nc - b + 0.5))
            cand.append((lo, a / nc, key))
        for lo, sup, key in sorted(cand, reverse=True)[: rc["top_tokens_per_class"]]:
            i, (side, root, k) = example[key]
            sides = rsmis[i].split(">")
            side_smi = sides[0] if side == "R" else sides[-1]
            rxn = Reaction.from_rsmi(rsmis[i], agents=cfg["synrfp"]["agents"])
            G = rxn.reactant if side == "R" else rxn.product
            rows.append({"class": c, "n_class": nc, "sign": key[1], "radius": k,
                         "support_in_class": sup, "log_odds": lo,
                         "fragment": fragment(side_smi, root, k, G.adj)})
    pd.DataFrame(rows).to_csv(out / "class_tokens.csv", index=False)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
