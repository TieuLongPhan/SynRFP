"""Test fingerprint separation on the two alcohol-screen stereoisomer pairs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
import subprocess

import numpy as np
from rdkit import Chem
from synkit.Graph.Stereo.canonical import canonicalize_stereograph
from synkit.IO import rsmi_to_graph

from synrfp.encode.core import fingerprint_from_tokens
from synrfp.graph.reaction import Reaction
from synrfp.tokenizers.wl import WLTokenizer

PAPER_FILE = "data/condition_ranking/predictions.csv"
PAPER_REVISION = "ac83ccf"
NODE_ATTRS = ("element", "aromatic", "charge", "hcount")
EDGE_ATTRS = ("order",)


def vector(substrate, product, aware):
    graphs = rsmi_to_graph(f"{substrate}>>{product}",drop_non_aam=False,
                          use_index_as_atom_map=False,include_stereo_descriptors=True)
    attributes = list(NODE_ATTRS)
    if aware:
        for graph in graphs:
            registry = tuple(graph.graph["stereo_descriptors"].values())
            if any(type(d).__name__!="TetrahedralStereo" for d in registry):
                raise ValueError("This pilot expects tetrahedral carriers")
            for descriptor in registry:
                atom = descriptor.atoms[0]
                graph.nodes[atom]["audit_root"] = True
                rooted = canonicalize_stereograph(
                    graph,registry,atom_color=(*NODE_ATTRS,"audit_root"),
                    bond_color=EDGE_ATTRS,enumerate_automorphism_group=False,
                )
                graph.nodes[atom].pop("audit_root")
                if not rooted.canonical_code:
                    raise ValueError("The rooted stereo identity is incomplete")
                graph.nodes[atom]["configured_root_digest"] = rooted.canonical_digest
        attributes.append("configured_root_digest")
    r = Reaction.from_graph(*graphs)
    tokenizer = WLTokenizer(attributes,EDGE_ATTRS)
    left, right = [tokenizer.tokens_graph(side,2) for side in (r.reactant,r.product)]
    return fingerprint_from_tokens(left,right,sketch="parity",bits=2048,seed=1,mode="delta")


def renumber(smiles):
    mol=Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(Chem.RenumberAtoms(mol,list(reversed(range(mol.GetNumAtoms())))),
                            canonical=False,isomericSmiles=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-repo",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    raw = subprocess.check_output(["git","-C",str(args.paper_repo),"show",f"{PAPER_REVISION}:{PAPER_FILE}"])
    panels = defaultdict(set)
    for row in csv.DictReader(raw.decode().splitlines()):
        panels[row["group"]].add((row["substrate"],row["intended_product"]))
    pairs = [sorted(values) for values in panels.values() if len(values)==2]
    if len(pairs)!=2 or any(len(values)>2 for values in panels.values()):
        raise ValueError("Expected precisely two source stereoisomer pairs")
    e = {}
    all_values = {False:[],True:[]}
    for pair in pairs:
        name = "boc_hydroxyproline_methyl_esters" if Chem.MolFromSmiles(pair[0][0]).HasSubstructMatch(Chem.MolFromSmarts("[#8]=[#6]O[#6]")) else "phthalimidocyclohexanols"
        if name in e:
            raise ValueError("Pair identity was ambiguous")
        result = {}
        for aware in (False,True):
            vectors = [vector(*member,aware) for member in pair]
            for member, original in zip(pair,vectors,strict=True):
                np.testing.assert_array_equal(vector(*(renumber(value) for value in member),aware),original)
            result["stereo_aware_differs" if aware else "ordinary_differs"] = not np.array_equal(*vectors)
            all_values[aware].extend(ar.tobytes() for ar in vectors)
        e[name] = result
    result = {
        "pairs":e,"all_four_ordinary_fingerprints_distinct":len(set(all_values[False]))==4,
        "all_four_stereo_aware_fingerprints_distinct":len(set(all_values[True]))==4,
        "atom_renumbering_invariance_verified":True,"profiles":"WL/r2, parity, delta, 2048 bits",
        "side_convention":"Original substrate stereo and paper's constitution-only intended product",
        "stereo_profile":"Exact rooted configured stereograph digest at the tetrahedral centers",
        "paper_input_revision":PAPER_REVISION,"paper_input_sha256":sha256(raw).hexdigest(),
        "runner_sha256":sha256(Path(__file__).read_bytes()).hexdigest(),"synkit":version("synkit"),"rdkit":version("rdkit"),
        "predictions_fitted":0,
    }
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")


if __name__=="__main__":
    main()
