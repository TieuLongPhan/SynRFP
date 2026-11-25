"""Compute exact WL/r2 channel keys with the rooted stereo profile."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from hashlib import file_digest, sha256
from importlib.metadata import version
import json
from pathlib import Path
import pickle
from struct import Struct
import subprocess
import time

import numpy as np
import pandas as pd
from rdkit import RDLogger
from synkit.Graph.Stereo.canonical import canonicalize_stereograph
from synkit.IO import smiles_to_graph

from synrfp.graph.molecule import Molecule
from synrfp.tokenizers.wl import WLTokenizer

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
ROOT=Path.home()/".local/share/synrfp-runs"
NATIVE=ROOT/"all_endpoint_factorial_full_v1"
OLD=ROOT/"delta_ceiling_request74_v2"
NODE_ATTRS=("element","aromatic","charge","hcount")
EDGE_ATTRS=("order",)
PROFILES=("ordinary","stereo_tetra","stereo_ez")
CHANNELS=("D","U","D+U")
ATOM_TOKEN="configured_root_digest"
BOND_TOKEN="configured_bond_root_digest"
ATOM_ROOT="audit_root"
BOND_ROOT="audit_bond_root"
TOKENIZERS=(
    WLTokenizer(NODE_ATTRS,EDGE_ATTRS),
    WLTokenizer((*NODE_ATTRS,ATOM_TOKEN),EDGE_ATTRS),
    WLTokenizer((*NODE_ATTRS,ATOM_TOKEN),(*EDGE_ATTRS,BOND_TOKEN)),
)
PAIR=Struct("<Qi")
CACHE={}
RDLogger.DisableLog("rdApp.*")


def digest(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle,"sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def check(path, expected):
    actual=digest(path)
    if actual!=expected:
        raise ValueError(f"Changed archive {path}")
    return actual


def write(path,data):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False)+"\n")
    tmp.replace(path)


def signed_digest(counts):
    return sha256(b"".join(PAIR.pack(token,count) for token,count in sorted(counts.items()) if count)).digest()


def parts(text):
    left,agents,right=text.strip().split(" ")[0].split(">")
    middle=tuple(piece for piece in agents.split(".") if piece)
    return (tuple(piece for piece in left.split(".") if piece)+middle,
            tuple(piece for piece in right.split(".") if piece)+middle)


def rooted_digest(graph, descriptors, atom_color, bond_color):
    result=canonicalize_stereograph(graph,descriptors,atom_color=atom_color,bond_color=bond_color,
                                    enumerate_automorphism_group=False)
    if not result.canonical_code:
        raise ValueError("Incomplete rooted configuration identity")
    return result.canonical_digest


def component(smiles):
    graph=smiles_to_graph(smiles,drop_non_aam=False,use_index_as_atom_map=False,
                           include_stereo_descriptors=True)
    if graph is None or not graph:
        raise ValueError("A prior-admitted component has no graph")
    mol=Molecule.from_nx_graph(graph)
    ordinary=TOKENIZERS[0].tokens_graph(mol,2)
    descriptor=tuple(graph.graph.get("stereo_descriptors",{}).values())
    types=Counter(type(value).__name__ for value in descriptor)
    tetra=tuple(value for value in descriptor if type(value).__name__=="TetrahedralStereo")
    planar=tuple(value for value in descriptor if type(value).__name__=="PlanarBondStereo")
    for value in tetra:
        node=graph.nodes[value.atoms[0]]
        node[ATOM_ROOT]=True
        node[ATOM_TOKEN]=rooted_digest(graph,tetra,(*NODE_ATTRS,ATOM_ROOT),EDGE_ATTRS)
        node.pop(ATOM_ROOT)
    carbon=Molecule.from_nx_graph(graph)
    tetra_counts=TOKENIZERS[1].tokens_graph(carbon,2)
    for value in planar:
        edge=graph.edges[value.atoms[2],value.atoms[3]]
        edge[BOND_ROOT]=True
        edge[BOND_TOKEN]=rooted_digest(graph,(value,),NODE_ATTRS,(*EDGE_ATTRS,BOND_ROOT))
        edge.pop(BOND_ROOT)
    alkene=Molecule.from_nx_graph(graph)
    stereo_counts=TOKENIZERS[2].tokens_graph(alkene,2)
    return (ordinary,tetra_counts,stereo_counts),dict(types)


def chunk(components):
    return [(smiles,*component(smiles)) for smiles in components]


def encode_rows(rows):
    keys=np.empty((len(rows),len(PROFILES),len(CHANNELS)),dtype="V32")
    for i,(left,right) in enumerate(rows):
        for j in range(len(PROFILES)):
            r,p=Counter(),Counter()
            for smiles in left:
                r.update(CACHE[smiles][j])
            for smiles in right:
                p.update(CACHE[smiles][j])
            signed=p.copy()
            signed.subtract(r)
            d,u=signed_digest(signed),signed_digest(p+r)
            keys[i,j]=[d,u,sha256(d+u).digest()]
    return keys


def inputs():
    plan_path=NATIVE/"plan.json"
    check(plan_path,read(NATIVE/"plan_seal.json")["sha256"])
    plan=read(plan_path)
    source=Path(plan["sources"]["ecreact"]["table"])
    check(source,plan["pinned_files"][str(source)])
    cohort=pd.read_csv(source,dtype={"row_id":str})
    reference=OLD/"ecreact.npz"
    receipt=read(OLD/"ENCODED.json")
    check(OLD/"ENCODING_PLAN.json",receipt["plan_sha256"])
    check(reference,receipt["sources"]["ecreact"]["artifact_sha256"])
    with np.load(reference,allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["row_ids"],cohort.row_id.to_numpy(str))
        return cohort,saved["ok"],saved["keys"][:,0,:3],source,reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs",type=int,default=24)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--pilot-components",type=int)
    args=parser.parse_args()
    start=time.monotonic()
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    cohort,ok,old,source,reference=inputs()
    indexes=np.flatnonzero(ok)
    sides=[parts(reaction) for reaction in cohort.rxn.iloc[indexes]]
    unique=sorted({part for pair in sides for side in pair for part in side},key=lambda s:(-len(s),s))
    manifest={
        "checkout":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
        "runner_sha256":digest(Path(__file__)),"source_sha256":digest(source),
        "request74_key_archive_sha256":digest(reference),"profiles":PROFILES,"channels":CHANNELS,
        "packages":{k:version(k) for k in ("synkit","rdkit","numpy","pandas","networkx","scikit-learn")},
    }
    manifest_path=out/"ENCODING_PLAN.json"
    if manifest_path.exists() and read(manifest_path)!=json.loads(json.dumps(manifest)):
        raise ValueError("Run directory is sealed to another encoder")
    write(manifest_path,manifest)
    all_unique=len(unique)
    if args.pilot_components is not None:
        select=sorted(set(np.r_[np.arange(min(32,len(unique))),
                               np.linspace(0,len(unique)-1,args.pilot_components,dtype=int)]))
        unique=[unique[j] for j in select]
    batch=8
    descriptor_types=Counter()
    accepted=0
    cache_path=out/"component_counts.pkl"
    cache_proof_path=out/"COMPONENTS.json"
    if cache_proof_path.exists():
        saved=read(cache_proof_path)
        check(cache_path,saved["artifact_sha256"])
        check(manifest_path,saved["encoding_plan_sha256"])
        with cache_path.open("rb") as handle:
            CACHE.update(pickle.load(handle))
        descriptor_types.update(saved["unique_component_descriptors"])
    else:
        print(f"Rooted stereo encoding: {len(unique):,} distinct component strings",flush=True)
        # Limit the largest configured graphs to a few workers. Count every
        # literal component once and retain atomic private checkpoints.
        slow = min(64,len(unique))
        all_batches = [(i, [item]) for i,item in enumerate(unique[:slow])]
        all_batches.extend((i,unique[i:i+batch]) for i in range(slow,len(unique),batch))
        shards = out/"component_batches"
        shards.mkdir(exist_ok=True)

        def retain(results):
            for smiles, tokens, coverage in results:
                CACHE[smiles]=tokens
                descriptor_types.update(coverage)

        def shard_path(index):
            return shards/f"{index:06d}.pkl"

        large_workers = min(4,max(1,args.jobs//4))
        with ProcessPoolExecutor(max_workers=large_workers) as large, ProcessPoolExecutor(
            max_workers=max(1,args.jobs-large_workers)
        ) as usual:
            futures={}
            for index,values in all_batches:
                path=shard_path(index)
                if path.exists():
                    with path.open("rb") as handle:
                        cached=pickle.load(handle)
                    if [row[0] for row in cached] != values:
                        raise ValueError("Unexpected component checkpoint rows")
                    retain(cached)
                    accepted+=len(cached)
                else:
                    pool = large if index<slow else usual
                    futures[pool.submit(chunk,values)]=path
            for done in as_completed(futures):
                results=done.result()
                path=futures[done]
                partial=path.with_suffix(".tmp")
                with partial.open("wb") as handle:
                    pickle.dump(results,handle,protocol=pickle.HIGHEST_PROTOCOL)
                partial.replace(path)
                retain(results)
                before=accepted
                accepted+=len(results)
                if accepted//2000 != before//2000 or accepted==len(unique):
                    print(f"Components {accepted:,}/{len(unique):,}",flush=True)
        if args.pilot_components is None:
            tmp=cache_path.with_suffix(".tmp")
            with tmp.open("wb") as handle:
                pickle.dump(CACHE,handle,protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(cache_path)
            write(cache_proof_path,{"artifact_sha256":digest(cache_path),
                                   "encoding_plan_sha256":digest(manifest_path),
                                   "unique_component_descriptors":descriptor_types})
    if args.pilot_components is not None:
        print("PILOT",len(unique),dict(descriptor_types),time.monotonic()-start,flush=True)
        return
    if len(CACHE)!=all_unique:
        raise ValueError("Missing source components")
    result=np.zeros((len(cohort),len(PROFILES),len(CHANNELS)),dtype="V32")
    new=np.empty((len(sides),len(PROFILES),len(CHANNELS)),dtype="V32")
    chunk_size=256
    print(f"Combining {len(sides):,} encoded source reactions",flush=True)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        offset=0
        for rows in pool.map(encode_rows,(sides[i:i+chunk_size] for i in range(0,len(sides),chunk_size))):
            new[offset:offset+len(rows)]=rows
            offset+=len(rows)
            if offset//20000 != (offset-len(rows))//20000 or offset==len(sides):
                print(f"Reactions {offset:,}/{len(sides):,}",flush=True)
    result[indexes]=new
    mismatch=np.flatnonzero(np.any(new[:,0]!=old[ok],axis=1))
    if len(mismatch):
        write(out/"BASELINE_MISMATCH.json",{"source_positions":indexes[mismatch].tolist()})
        raise ValueError(f"Request 74 exact baseline replay differs on {len(mismatch)} rows")
    partial=out/"ec_keys.tmp.npz"
    artifact=out/"ec_keys.npz"
    np.savez(partial,keys=result,ok=ok,row_ids=cohort.row_id.to_numpy(str))
    partial.replace(artifact)
    write(out/"ENCODED.json",{"status":"complete","n_source":len(cohort),"n_encoded":int(ok.sum()),
                              "unique_components":all_unique,"unique_component_descriptors":descriptor_types,
                              "ordinary_rows_replayed":int(ok.sum()),"seconds":time.monotonic()-start,
                              "artifact_sha256":digest(artifact),"encoding_plan_sha256":digest(manifest_path)})
    print("Unsketched stereo channel keys complete",flush=True)


if __name__=="__main__":
    main()
