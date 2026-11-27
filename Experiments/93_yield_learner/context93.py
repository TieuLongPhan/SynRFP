"""Create the two unavailable USPTO context caches in the full-row input profile."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import argparse
import os
import sys
import time

import numpy as np

from plan93 import BASE, CONTEXT, ORIGINAL, ROOT, read, sha, table, verify, write

VARIANTS = ROOT/"uspto_context"
ARCHIVE = BASE/"uspto_literal_fullrow_v1/features/uspto_yield/synrfp_roles_r3b2048.npz"


@lru_cache(maxsize=1)
def adapters():
    paths = [ORIGINAL,ORIGINAL/"Experiments/13_yield"]
    for path in paths:
        sys.path.insert(0,str(path))
    folder=ORIGINAL/"Experiments/57_uspto_fullrow_input"
    spec=spec_from_file_location("yield93_native57_study",folder/"study.py")
    study=module_from_spec(spec)
    sys.modules[spec.name]=study
    spec.loader.exec_module(study)
    boundary=study.module("yield93_native57_inputs",folder/"inputs.py",{"study":study})
    B,_,R=boundary.modules()
    return B,R,boundary


def batch(task):
    first,left,right,roles,invalid=task
    b,r,boundary=adapters()
    d=np.empty_like(roles)
    du=np.empty_like(roles)
    for i,(lhs,rhs) in enumerate(zip(left,right,strict=True)):
        graph=boundary.graph if first+i in invalid else b._graph
        pair=b.Reaction.from_graph(graph(lhs),graph(rhs))
        a,c=(b.TOKENIZER.tokens_graph(g,3) for g in (pair.reactant,pair.product))
        if not np.array_equal(r.roles_from_tokens(a,c),roles[i]):
            raise ValueError(f"Archived USPTO D+S bytes fail at {first+i}")
        d[i]=r._native(a,c,2048,"delta")
        du[i,:128]=roles[i,:128]
        du[i,128:]=r._native(a,c,1024,"union")
        np.testing.assert_array_equal(d[i,:128]^d[i,128:],du[i,:128])
    return first,d,du


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs",type=int,required=True)
    args=parser.parse_args()
    started=time.monotonic()
    p=read(ROOT/"PLAN.json")["sources"]["uspto_yield"]
    src=table(p["table"],p["sha256"])
    evidence=read(ARCHIVE.with_suffix(".json"))
    assert evidence["valid"]==evidence["rows"]==143535 and evidence["source_table_sha256"]==p["sha256"]
    with np.load(verify(ARCHIVE,evidence["feature_sha256"]),allow_pickle=False) as a:
        np.testing.assert_array_equal(a["row_ids"].astype(str),src.row_id.to_numpy())
        np.testing.assert_array_equal(a["source_positions"],np.arange(len(src)))
        roles=a["X"]
    invalid=frozenset(evidence["literal_source_positions"])
    assert invalid=={6872,92907,132781}
    VARIANTS.mkdir(exist_ok=True)
    receipt=VARIANTS/"COMPLETE.json"
    if receipt.exists():
        old=read(receipt)
        assert old["roles_sha256"]==evidence["feature_sha256"] and old["table_sha256"]==p["sha256"]
        for k,v in old["features"].items():
            verify(VARIANTS/f"{k}.npz",v)
        return
    left,right=src.left.to_numpy(str),src.right.to_numpy(str)
    pieces=[]
    x,y=np.empty_like(roles),np.empty_like(roles)
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        step=192
        for a,d,du in pool.map(batch,((i,left[i:i+step],right[i:i+step],roles[i:i+step],invalid)
                                     for i in range(0,len(left),step)),chunksize=1):
            x[a:a+len(d)],y[a:a+len(du)]=d,du
            pieces.append(a)
            if len(pieces)%60==0:
                print("USPTO rows encoded:",a+len(d),flush=True)
    assert pieces==list(range(0,len(left),step))
    hashes={}
    for mode,matrix in zip(CONTEXT,(x,y),strict=True):
        pth=VARIANTS/f"{mode}.npz"
        temp=pth.with_suffix(".tmp.npz")
        np.savez_compressed(temp,X=matrix,row_ids=src.row_id.to_numpy(str),
                            source_positions=np.arange(len(src)),n_bits=2048,ok=np.ones(len(src),bool))
        temp.replace(pth)
        hashes[mode]=sha(pth)
    write(receipt,dict(features=hashes,table_sha256=p["sha256"],rows=len(src),
          roles_sha256=evidence["feature_sha256"],exact_role_replays=len(src),
          literal_source_positions=sorted(invalid),elapsed_seconds=time.monotonic()-started,
          source_sha256={str(f.relative_to(ORIGINAL)):sha(f) for f in (
            ORIGINAL/"Experiments/13_yield/yield_roles_features.py",
            ORIGINAL/"Experiments/13_yield/bh_features.py",
            ORIGINAL/"Experiments/57_uspto_fullrow_input/inputs.py")}))
    print("Verified the two new full-row USPTO context caches",flush=True)


if __name__=="__main__":
    main()
