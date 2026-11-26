"""Select figure examples from exact barrier classes using the Request 78 encoding."""
import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

import barrier78
from common78 import HERE, OUTPUT, Pins, digest, export, read, write

SCOPE = ("sn2", "snar", "e2sn2")
OUT = OUTPUT / "barrier_figure_cases"
PARENT = "4ba01b4bbfde4bc01c68a1efe1618c6392ac79f4"
SNAR_SOURCE = "https://pmc-oa-opendata.s3.amazonaws.com/PMC9528810.1/SC-012-D0SC04896H-s002.zip"
SNAR_MEMBER = "SC-012-D0SC04896H-s050.csv"
SNAR_SHA = "590153f74fd27a1414cf6d1cdae3758e68cc9e137da81bd58fa84cc5c9750107"
PROPERTY = Path("/var/tmp/synrfp-labhhc4-storage/runs/properties_allfps_v1")
DATA = {key: {"article": url, "target": target, "basis": basis} for key, url, target, basis in (
    ("sn2", "https://github.com/heid-lab/reaction_database",
     "computed activation energy", "quantum chemistry"),
    ("snar", "https://pmc.ncbi.nlm.nih.gov/articles/PMC9528810/",
     "activation free energy", "Eyring equation applied to experimental rate constants"),
    ("e2sn2", "https://github.com/hesther/reactiondatabase",
     "computed activation energy", "quantum chemistry"),
)}


def side_identity(side):
    mol = Chem.MolFromSmiles(side)
    if mol is None:
        raise ValueError("Frozen input cannot be parsed for structural comparison")
    Chem.RemoveStereochemistry(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    parts = Counter(Chem.MolToSmiles(m) for m in Chem.GetMolFrags(mol, asMols=True))
    return parts, mol.GetNumHeavyAtoms()


def edit_pattern(mapped):
    """Record the mapped bond edits as element/order tuples when an AAM exists."""
    try:
        parser = Chem.SmilesParserParams()
        parser.removeHs = False
        mols = [Chem.MolFromSmiles(side, parser) for side in mapped.split(">>")]
        assert len(mols) == 2 and all(m is not None for m in mols)
        atoms = [{a.GetAtomMapNum(): a.GetSymbol() for a in m.GetAtoms()} for m in mols]
        if (atoms[0] != atoms[1] or 0 in atoms[0] or
                any(len(ledger) != mol.GetNumAtoms() for ledger, mol in zip(atoms, mols, strict=True))):
            return None
        edge = [{tuple(sorted((b.GetBeginAtom().GetAtomMapNum(),
                              b.GetEndAtom().GetAtomMapNum()))): str(b.GetBondType())
                 for b in m.GetBonds()} for m in mols]
    except (AssertionError, ValueError):
        return None
    changed = tuple(sorted((*sorted(atoms[0][m] for m in pair),
                            edge[0].get(pair, "ABSENT"), edge[1].get(pair, "ABSENT"))
                           for pair in edge[0].keys() | edge[1].keys()
                           if edge[0].get(pair) != edge[1].get(pair)))
    return changed


def fold_labels(pins, plan, n):
    test = np.full(n, -1, dtype=np.int16)
    selected = [item for item in plan["splits"] if item["name"].startswith("group_cv_f")]
    for number, item in enumerate(selected):
        with np.load(pins.use(item["path"], item["original_sha256"]), allow_pickle=False) as data:
            positions = data["test"]
        assert len(positions) == item["test_rows"] and (test[positions] == -1).all()
        test[positions] = number
    assert len(selected) == 5 and (test >= 0).all()
    return [r["name"] for r in selected], test


def scan(dataset, meta, y, pins, property_plan, snar_source):
    frame, anchors, original_ok = barrier78.STATE["frames"][dataset]
    assert original_ok.all()
    size = len(frame)
    _, _, _, keys = barrier78.encode((dataset, 0, size))
    targets = property_plan["tables"][dataset]
    raw_path = pins.use(targets["raw_path"], property_plan["pinned_files"][targets["raw_path"]])
    original = pd.read_csv(raw_path, dtype={"r_id": str})
    path = pins.use(targets["path"], property_plan["pinned_files"][targets["path"]])
    table = pd.read_csv(path, dtype={"r_id": str, "row_id": str})
    np.testing.assert_array_equal(original.r_id, table.r_id)
    np.testing.assert_array_equal(original.ea.to_numpy(), y)
    fold_names, fold = fold_labels(pins, meta, size)
    groups = defaultdict(list)
    for i, d in enumerate(keys[:, 0]):
        groups[bytes(d)].append(i)
    chemistry = {}
    sdev = float(np.std(y, ddof=1))
    counters = ("repeated_D_classes", "repeated_D_classes_split_by_U",
                "nonempty_D_classes_split_by_U", "nonempty_D_different_constitutions_split_by_U_pairs",
                "figure_eligible_pairs", "figure_eligible_pairs_same_test_fold")
    counts = dict.fromkeys(counters, 0)
    empty = groups.get(barrier78.signed({}), [])
    empty_reactants = {
        tuple(sorted(side_identity(frame.left.iat[i])[0].items())) for i in empty
    }
    finalists = []
    for d, group in groups.items():
        if len(group) < 2:
            continue
        counts["repeated_D_classes"] += 1
        if len(np.unique(keys[group, 1])) == 1:
            continue
        counts["repeated_D_classes_split_by_U"] += 1
        if d == barrier78.signed({}):
            continue
        counts["nonempty_D_classes_split_by_U"] += 1
        for i in group:
            r, nr = side_identity(frame.left.iat[i])
            p, nprod = side_identity(frame.right.iat[i])
            map_edits = edit_pattern(original.aam.iat[i]) if "aam" in original else None
            chemistry[i] = (r, p, nr+nprod, map_edits)
        for a,b in combinations(group, 2):
            if keys[a, 1] == keys[b, 1]:
                continue
            ra,pa,na,ea = chemistry[a]
            rb,pb,nb,eb = chemistry[b]
            if ra==rb or pa==pb:
                continue
            counts["nonempty_D_different_constitutions_split_by_U_pairs"] += 1
            if "aam" in original and (not ea or ea!=eb):
                continue
            shared = ra & rb
            if not shared or sum((ra-shared).values())!=1 or sum((rb-shared).values())!=1:
                continue
            gap = float(abs(y[b]-y[a]))
            if gap < .25*sdev:
                continue
            if not np.array_equal(anchors[0][a],anchors[0][b]):
                raise ValueError("Exact D failed a parity archive equality control")
            if not np.array_equal(anchors[1][a,:128],anchors[1][b,:128]):
                raise ValueError("Exact D failed the D-block equality control")
            if np.array_equal(anchors[1][a,128:],anchors[1][b,128:]):
                continue
            same_test = fold[a] == fold[b]
            counts["figure_eligible_pairs"] += 1
            if same_test:
                counts["figure_eligible_pairs_same_test_fold"] += 1
            finalists.append((not same_test, na+nb, a, b, shared))
    finalists.sort(key=lambda row: row[:4])

    summary={"dataset": dataset, "n_reactions": size, "D_classes": len(groups),
             "required_gap_kcal_mol": .25*sdev,
             "empty_D_rows":len(empty), "empty_D_distinct_reactant_constitutions":len(empty_reactants),
             "empty_D_DU_classes":len(np.unique(keys[empty, 1])), **counts}
    if not finalists:
        return None, summary
    _, _, a, b, shared = finalists[0]
    exact_delta, exact_union = [], []
    for i in (a,b):
        l = barrier78.side_tokens(frame.left.iat[i], False)
        r = barrier78.side_tokens(frame.right.iat[i], False)
        d, u, _ = barrier78.STATE["native"].token_algebra(l, r)
        exact_delta.append(d)
        exact_union.append(u)
    assert exact_delta[0] and exact_delta[0]==exact_delta[1] and exact_union[0]!=exact_union[1]
    values = DATA[dataset]
    result = {
        "dataset":dataset, "barrier_target":values["target"], "barrier_basis":values["basis"],
        "units":"kcal/mol", "source_reference":values["article"],
        "common_reactant_components":".".join(sorted(shared.elements())),
        "same_test_fold":fold_names[fold[a]] if fold[a]==fold[b] else "",
        "exact_D_equal":True, "exact_U_equal":False, "D2048_bits_equal":True,
        "D1024_block_equal":True, "U1024_block_equal":False,
        "n_reactions_in_D_class":len(groups[bytes(keys[a,0])]),
        "nonzero_exact_D_tokens":len(exact_delta[0]),
        "absolute_barrier_gap_kcal_mol":float(abs(y[a]-y[b])),
    }
    for tag, i in (("a",a),("b",b)):
        result.update({f"source_position_{tag}":i,f"r_id_{tag}":table.r_id.iat[i],
                       f"rxn_{tag}":frame.left.iat[i]+">>"+frame.right.iat[i],
                       f"barrier_{tag}_kcal_mol":float(y[i]),f"test_fold_{tag}":fold_names[fold[i]]})
    if dataset=="snar":
        # Benchmark IDs are reindexed, so use the author's reaction and target as the join.
        reaction, target = "Reaction SMILES", "Activation Free Energy (kcalmol-1)"
        reference = snar_source.set_index([reaction,target]).sort_index()
        keys_in_source = set(reference.index)
        assert all((text, energy) in keys_in_source
                   for text,energy in zip(original.rxn,original.ea,strict=True))
        for tag, i in (("a",a),("b",b)):
            rows = reference.loc[[(original.rxn.iat[i],original.ea.iat[i])]]
            assert len(rows)==1
            entry=rows.iloc[0]
            result.update({f"author_entry_{tag}":int(entry["Entry"]),
                           f"temperature_{tag}_K":float(entry["Temp (K)"]),
                           f"solvent_{tag}":entry["Solvent"],
                           f"experimental_reference_{tag}":entry["DOI"]})
        result["matched_author_conditions"] = all(
            result[f"{name}_a{suffix}"]==result[f"{name}_b{suffix}"]
            for name,suffix in (("temperature","_K"),("solvent",""),("experimental_reference","")))
        summary["source_snar_rows_matched_by_reaction_and_target"]=size
    return result, summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=OUT)
    parser.add_argument("--snar-source",type=Path,required=True)
    args=parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    pins=Pins()
    registered=read(pins.use(OUTPUT/"barrier_AUDIT.json"))
    native_seal=read(pins.use(PROPERTY/"plan_seal.json"))
    properties=read(pins.use(PROPERTY/"plan.json",native_seal["sha256"]))
    barrier78.TARGETS=SCOPE
    values=barrier78.prepare(pins)
    for path, expected in registered["input_sha256"].items():
        if path in pins.hashes:
            pins.use(path, expected)
    snar_source = pd.read_csv(pins.use(args.snar_source,SNAR_SHA),encoding="cp1252")
    pairs, scans=[],[]
    for dataset in SCOPE:
        plan,y=values[dataset]
        result,summary=scan(dataset,plan,y,pins,properties,snar_source)
        scans.append(summary)
        if result is not None:
            pairs.append(result)
    assert len(pairs)>=1
    args.output.mkdir(parents=True,exist_ok=True)
    export(args.output/"barrier_figure_cases.csv",pairs)
    export(args.output/"barrier_case_scan.csv",scans)
    write(args.output/"AUDIT.json",{
        "request_78_results_commit":PARENT, "inputs_sha256":pins.hashes,
        "snar_author_source":{"url":SNAR_SOURCE,"member":SNAR_MEMBER,"member_sha256":SNAR_SHA,
                              "row_link":"original reaction SMILES and exact activation free energy"},
        "code_sha256":{name:digest(HERE/name) for name in ("barrier_cases78.py","barrier78.py","common78.py")},
        "python":platform.python_version(), "rdkit":Chem.rdBase.rdkitVersion,
        "numpy":np.__version__, "pandas":pd.__version__, "models_fitted":0,
        "rule":"Nonempty equal exact D, distinct exact U and distinct parity U block. "
               "Different stereo-erased reactant and product constitutions, one different reactant with a shared component, "
               "and the same mapped bond-edit element/order pattern when source maps are available. "
               "Minimum gap is one quarter of the full-cohort target sample SD. "
               "Order candidates by same grouped test fold, then total heavy atoms, then source positions.",
        "choice_counts":scans,
    })


if __name__=="__main__":
    main()
