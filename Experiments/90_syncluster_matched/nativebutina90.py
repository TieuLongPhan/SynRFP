"""Run SynCluster's actual RDKit Butina path at its original distance cutoff."""
import argparse
import importlib.util
from pathlib import Path
import time

import numpy as np
from rdkit import DataStructs, rdBase

from common90 import N, ROOT, UPSTREAM, read, sha, write


def upstream():
    path=UPSTREAM/"clustering/scripts/03_cluster.py"
    spec=importlib.util.spec_from_file_location("syncluster_03_cluster90",path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module,path


def vectors(bits):
    return [DataStructs.CreateFromBinaryText(row.tobytes()) for row in bits]


def packed(method):
    if method=="synrfp":
        path=ROOT/"synrfp_bits.npy"
        assert sha(path)==read(ROOT/"PREPARED.json")["local_sha256"][path.name]
        return np.load(path,allow_pickle=False)
    path=ROOT/"drfp_features.npz"
    assert sha(path)==read(path.with_suffix(".json"))["sha256"]
    with np.load(path,allow_pickle=False) as a:
        assert np.all(a["ok"])
        return a["X"]


def run(method):
    x = packed(method)
    assert x.shape==(N,256)
    native,path = upstream()
    fps = vectors(x)
    cutoff=read(ROOT/"PREPARED.json")["native"]["distance_cutoff"]
    start=time.monotonic()
    result = native.Bu_cluster(fps,cutoff)
    labels=np.full(N,-1,dtype=np.int32)
    centers=[]
    for idx,members in enumerate(result):
        indexes=np.asarray(members,dtype=int)
        assert np.all(labels[indexes]<0)
        labels[indexes]=idx
        centers.append(members[0])
    assert np.all(labels>=0)
    artifact=ROOT/f"native_butina_{method}.npz"
    np.savez_compressed(artifact,labels=labels,centers=centers)
    write(artifact.with_suffix(".json"),dict(sha256=sha(artifact),cutoff=cutoff,n=N,rdkit=rdBase.rdkitVersion,
          upstream_source_sha256=sha(path),runtime_seconds=time.monotonic()-start,actual_clusters=len(result)))
    print(method,len(result),"clusters in legacy Butina",flush=True)


def fixture(path):
    from rdkit.ML.Cluster import Butina

    data = vectors(np.load(path,allow_pickle=False))
    distances=np.asarray([1.0-value for i,item in enumerate(data)
                         for value in DataStructs.BulkTanimotoSimilarity(item,data[:i])])
    cutoffs=(*range(0,1001,50),441,597,902)
    labels=np.empty((len(cutoffs),len(data)),dtype=np.int32)
    centers=np.full_like(labels,-1)
    for idx,cutoff in enumerate(cutoffs):
        clusters=Butina.ClusterData(distances,len(data),cutoff/1000,isDistData=True,reordering=True)
        for group,members in enumerate(clusters):
            labels[idx,list(members)]=group
            centers[idx,group]=members[0]
    np.savez_compressed(path.with_suffix('.npz'),distances=distances,labels=labels,centers=centers,
                        cutoffs=cutoffs,rdkit=rdBase.rdkitVersion)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument("method",nargs="?",choices=("synrfp","drfp"))
    group.add_argument("--fixture",type=Path)
    a=p.parse_args()
    if a.fixture is None:
        run(a.method)
    else:
        fixture(a.fixture)
