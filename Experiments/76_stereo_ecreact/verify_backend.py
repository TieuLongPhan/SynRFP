"""Compare compiled and native rooted equivalence on controls and case chemistry."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from hashlib import sha256
from pathlib import Path
import subprocess

from rdkit import Chem
from synkit.IO import smiles_to_graph
from synrfp.encode.core import fingerprint_from_tokens

from encode_stereo import NODE_ATTRS, EDGE_ATTRS, ATOM_ROOT, BOND_ROOT, component, parts, rooted_digest, write
from report_stereo import CASES, CASE_COMMIT, CASE_PATH, ENDPOINTS, constitutional_key
from rooted import ConfiguredExpansion


def digest_pairs(smiles):
    graph=smiles_to_graph(smiles,drop_non_aam=False,use_index_as_atom_map=False)
    registry=tuple(graph.graph["stereo_descriptors"].values())
    atom_config=tuple(v for v in registry if type(v).__name__=="TetrahedralStereo")
    bond_config=tuple(v for v in registry if type(v).__name__=="PlanarBondStereo")
    pairs=[]
    if atom_config:
        fast=ConfiguredExpansion(graph,atom_config,NODE_ATTRS,EDGE_ATTRS)
        for v in atom_config:
            owner=v.atoms[0]
            graph.nodes[owner][ATOM_ROOT]=True
            native=rooted_digest(graph,atom_config,(*NODE_ATTRS,ATOM_ROOT),EDGE_ATTRS)
            graph.nodes[owner].pop(ATOM_ROOT)
            pairs.append((native,fast.atom(owner)))
    for v in bond_config:
        left,right=v.atoms[2:4]
        fast=ConfiguredExpansion(graph,(v,),NODE_ATTRS,EDGE_ATTRS)
        graph.edges[left,right][BOND_ROOT]=True
        native=rooted_digest(graph,(v,),NODE_ATTRS,(*EDGE_ATTRS,BOND_ROOT))
        graph.edges[left,right].pop(BOND_ROOT)
        pairs.append((native,fast.bond(left,right)))
    return pairs


def renumber(smiles):
    mol=Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(Chem.RenumberAtoms(mol,list(reversed(range(mol.GetNumAtoms())))),
                            canonical=False,isomericSmiles=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    fixture_pairs=[
        ("F[C@H](Cl)Br","F[C@@H](Cl)Br",[False,True,True]),
        ("F/C=C/F","F/C=C\\F",[False,False,True]),
        ("C[C@H](O)[C@@H](C)O","C[C@@H](O)[C@H](C)O",[False,False,False]),
        ("C[C@H](O)[C@@H](C)O","C[C@H](O)[C@H](C)O",[False,True,True]),
    ]
    # Confirm tetrahedral equivalence on the same alcohol screen used by Request 74.
    paper=CASES.parent/"SynRFP-paper-attribution"
    raw_paper=subprocess.check_output(["git","-C",str(paper),"show","ac83ccf:data/condition_ranking/predictions.csv"])
    groups={}
    for r in csv.DictReader(raw_paper.decode().splitlines()):
        groups.setdefault(r["group"],set()).add((r["substrate"],r["intended_product"]))
    alcohols=[set(pairs) for pairs in groups.values() if len(pairs)==2]
    assert len(alcohols)==2
    for alcohol in alcohols:
        entries=sorted(alcohol)
        for backend in ("native","bliss"):
            for index in (0,1):
                bits=[]
                for substrate,product in entries:
                    left=component(substrate,backend)[0][index]
                    right=component(product,backend)[0][index]
                    bits.append(fingerprint_from_tokens(left,right,sketch="parity",bits=2048,seed=1,mode="delta").tobytes())
                assert (bits[0]!=bits[1])==(index==1)
    controls=set()
    for left,right,expected in fixture_pairs:
        for backend in ("bliss","native"):
            a=component(left,backend)[0]
            b=component(right,backend)[0]
            assert [x!=y for x,y in zip(a,b,strict=True)]==expected
            assert a==component(renumber(left),backend)[0]
            assert b==component(renumber(right),backend)[0]
        controls.update((left,right,renumber(left),renumber(right)))
    raw=subprocess.check_output(["git","-C",str(CASES),"show",f"{CASE_COMMIT}:{CASE_PATH}/case_a.csv"])
    selected=set()
    n_cases=0
    for row in csv.DictReader(raw.decode().splitlines()):
        if row["endpoint"] not in ENDPOINTS:
            continue
        left,_,right=row["rxn"].split(">")
        if constitutional_key(left,True)!=constitutional_key(right,True):
            continue
        n_cases+=1
        for field in ("rxn","collision_rxn"):
            selected.update(piece for side in parts(row[field]) for piece in side)
    assert n_cases==15
    native_to_bliss={}
    bliss_to_native={}
    n_roots=0
    with ProcessPoolExecutor(max_workers=2) as pool:
        for pairs in pool.map(digest_pairs,sorted(controls|selected)):
            for native,bliss in pairs:
                assert native_to_bliss.setdefault(native,bliss)==bliss
                assert bliss_to_native.setdefault(bliss,native)==native
                n_roots+=1
    write(args.output,{"status":"passed","native_bliss_root_equivalence_verified":True,
                       "n_exact_root_certificates":n_roots,"n_components":len(controls|selected),
                       "n_ec_react_case_reactions":n_cases,"synthetic_configured_pair_controls":len(fixture_pairs),
                       "alcohol_stereo_pair_controls":len(alcohols),"renumbering_invariance_verified":True,
                       "source_cases_sha256":sha256(raw).hexdigest(),"source_alcohol_sha256":sha256(raw_paper).hexdigest(),
                       "runner_sha256":sha256(Path(__file__).read_bytes()).hexdigest()})
    print("Rooted certificate equivalence controls passed",flush=True)


if __name__=="__main__":
    main()
