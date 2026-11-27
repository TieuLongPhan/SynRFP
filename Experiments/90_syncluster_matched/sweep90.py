"""Fit a label-blind Butina distance grid with exactly replayed RDKit ties."""
import ctypes
import subprocess
import sys
import time

import numpy as np
from numpy.ctypeslib import ndpointer

from common90 import EXPERIMENTS, HERE, N, ROOT, sha, write
from kmeans90 import score
from nativebutina90 import packed


def load_library():
    source=HERE/"butina90.cpp"
    compiled=ROOT/"butina90.so"
    subprocess.run(["g++","-O3","-std=c++17","-march=native","-fopenmp","-shared","-fPIC",
                    str(source),"-o",str(compiled)],check=True)
    lib=ctypes.CDLL(str(compiled))
    lib.make_distances.argtypes=[ndpointer(np.uint64,flags="C_CONTIGUOUS"),ctypes.c_int,ctypes.c_int,
                                  ndpointer(np.uint16,flags="C_CONTIGUOUS")]
    lib.make_distances.restype=None
    lib.cluster_distances.argtypes=[ndpointer(np.uint16,flags="C_CONTIGUOUS"),ctypes.c_int,ctypes.c_int,
                                    ndpointer(np.int32,flags="C_CONTIGUOUS"),ndpointer(np.int32,flags="C_CONTIGUOUS")]
    lib.cluster_distances.restype=ctypes.c_int
    return lib


def distances(lib,bits):
    bits=np.ascontiguousarray(bits).view(np.uint64)
    n,words=bits.shape
    result=np.empty(n*(n-1)//2,dtype=np.uint16)
    lib.make_distances(bits,n,words,result)
    return result


def cluster(lib,codes,n,cutoff):
    labels=np.empty(n,dtype=np.int32)
    centers=np.empty(n,dtype=np.int32)
    k=lib.cluster_distances(codes,n,cutoff,labels,centers)
    assert np.all(labels>=0) and len(np.unique(labels))==k and np.all(labels[centers[:k]]==np.arange(k))
    return labels,centers[:k]


def validate_sample(lib,bits):
    """Compare distance boundaries and indexed assignments with direct RDKit."""
    rng=np.random.default_rng(900)
    chosen=bits[rng.choice(len(bits),512,replace=False)]
    # Include repeats, all-zero inputs, and identical nonzero pairs.
    data=np.concatenate([chosen,chosen[:8],np.zeros((3,bits.shape[1]),np.uint8)])
    codes=distances(lib,data)
    path=ROOT/"native_fixture.npy"
    np.save(path,data)
    subprocess.run([str(EXPERIMENTS/"19_syncluster_reference/external/env/bin/python"),
                    str(HERE/"nativebutina90.py"),"--fixture",str(path)],check=True)
    with np.load(path.with_suffix(".npz"),allow_pickle=False) as native:
        d=native["distances"]
        for cut,labels,centers in zip(native["cutoffs"],native["labels"],native["centers"],strict=True):
            np.testing.assert_array_equal(codes<=cut,d<=cut/1000)
            a,b=cluster(lib,codes,len(data),int(cut))
            np.testing.assert_array_equal(a,labels)
            np.testing.assert_array_equal(b,centers[centers>=0])
        return dict(fixture_rows=len(data),rdkit=str(native["rdkit"]),thresholds_checked=len(native["cutoffs"]),
                    memberships_and_centers_verified=True)


def main():
    start=time.monotonic()
    lib=load_library()
    bits=packed("synrfp")
    sample=validate_sample(lib,bits)
    print("RDKit distance and cluster replay passed",flush=True)
    before=time.monotonic()
    codes=distances(lib,bits)
    pair_seconds=time.monotonic()-before
    print("Prepared exact distance cutoffs for",len(codes),"pairings in",pair_seconds,flush=True)
    with np.load(ROOT/"cohort.npz",allow_pickle=False) as cohort:
        y=cohort["labels"]
    folder=ROOT/"sweep"
    folder.mkdir(exist_ok=True)
    rows={}
    def run(cut,stage):
        timer=time.monotonic()
        labels,centers=cluster(lib,codes,N,cut)
        path=folder/f"cutoff_{cut:04d}.npz"
        np.savez_compressed(path,labels=labels,centers=centers)
        result=dict(method="synrfp-wl-r2-b2048-parity-delta",algorithm="Butina",
                    distance_cutoff=cut/1000,similarity_threshold=1-cut/1000,seed="native_order",n=N,
                    **score(y,labels),partition_sha256=sha(path),source="new",
                    runtime_seconds=time.monotonic()-timer,stage=stage)
        rows[cut]=result
        print(cut,result["actual_clusters"],f"{result['ami']:.6f}",flush=True)
        write(folder/f"cutoff_{cut:04d}.json",result)
    def closest():
        return min(rows,key=lambda cut:(abs(rows[cut]["actual_clusters"]-378),abs(cut-600),cut))
    for cut in range(0,1001,50):
        run(cut,"0.05_distance_grid")
    coarse=closest()
    if abs(rows[coarse]["actual_clusters"]-378)>1:
        # The coarse grid is preserved. Fine thresholds use only the cluster count.
        lows=[cut for cut in rows if rows[cut]["actual_clusters"]>=378]
        highs=[cut for cut in rows if rows[cut]["actual_clusters"]<=378]
        if lows and highs:
            low,high=max(lows),min(highs)
            if low<high:
                for cut in range(low+10,high,10):
                    run(cut,"0.01_count_refinement")
                lows=[cut for cut in rows if rows[cut]["actual_clusters"]>=378]
                highs=[cut for cut in rows if rows[cut]["actual_clusters"]<=378]
                low,high=max(lows),min(highs)
                if low<high and abs(rows[closest()]["actual_clusters"]-378)>1:
                    for cut in range(low+1,high):
                        run(cut,"0.001_count_refinement")
    best=closest()
    write(ROOT/"SWEEP_COMPLETE.json",dict(distance_pair_seconds=pair_seconds,
          runtime_seconds=time.monotonic()-start,fixtures=sample,
          selected_distance_cutoff=best/1000,coarse_distance_cutoff=coarse/1000,
          coarse_clusters=int(rows[coarse]["actual_clusters"]),selected_clusters=int(rows[best]["actual_clusters"]),
          source_sha256={f:sha(HERE/f) for f in ("butina90.cpp","sweep90.py")},
          compiler=subprocess.check_output(["g++","--version"],text=True).splitlines()[0],
          artifact_sha256={f"cutoff_{c:04d}.json":sha(folder/f"cutoff_{c:04d}.json") for c in rows}))


if __name__=="__main__":
    main()
