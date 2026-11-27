"""Compute exact nearest-training Tanimoto on pinned full-row fingerprint caches."""
import argparse
import ctypes
from functools import lru_cache
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from numpy.ctypeslib import ndpointer
import pandas as pd

from preflight import HERE, ROOT, RELEASE, METHODS, PARTS, digest, read, verify
from rank92 import WORK

POPCOUNT = np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1).sum(axis=1).astype(np.uint8)
METHOD_SOURCE = {"synrfp_optimal":"synrfp_roles_r3b2048","drfp":"drfp","rdkit_struct":"rdkit_struct"}
STUDY65 = ROOT.parent/"default_optimal_benchmark_v1"
USPTO = ROOT.parent/"uspto_literal_fullrow_v1"
SOURCE = HERE/"tanimoto92.cpp"


@lru_cache(maxsize=1)
def library():
    binary = WORK/f"tanimoto_{digest(SOURCE)[:16]}.so"
    if not binary.exists():
        temp = binary.with_suffix(".tmp.so")
        subprocess.run(["g++","-O3","-march=native","-std=c++17","-fopenmp","-shared","-fPIC",
                        str(SOURCE),"-o",str(temp)],check=True)
        temp.replace(binary)
    lib = ctypes.CDLL(str(binary))
    fn = lib.max_tanimoto
    arr = lambda dtype: ndpointer(dtype=dtype,flags="C_CONTIGUOUS")
    fn.argtypes = (arr(np.uint64),arr(np.int32),ctypes.c_int,
                   arr(np.uint64),arr(np.int32),ctypes.c_int,ctypes.c_int,arr(np.float32))
    fn.restype = None
    return fn


def matrix_tanimoto(x, train, test):
    # Padding is useful for feature spaces whose width is not a multiple of 64.
    if x.shape[1]%8:
        x = np.pad(x,((0,0),(0,-x.shape[1]%8)))
    a = x[train]
    norms = POPCOUNT[a].sum(axis=1,dtype=np.int32)
    # Sorting by count preserves the membership of each exact search band.
    order = np.argsort(norms,kind="stable")
    seq = np.ascontiguousarray(a[order]).view(np.uint64)
    dim = x.shape[1]*8
    hist = np.bincount(norms,minlength=dim+1)
    starts = np.r_[0,hist.cumsum()].astype(np.int32)
    q = x[test]
    sizes = POPCOUNT[q].sum(axis=1,dtype=np.int32)
    out = np.empty(len(test),dtype=np.float32)
    words = x.shape[1]//8
    library()(np.ascontiguousarray(q).view(np.uint64),sizes,len(test),seq,starts,words,dim,out)
    return out


def controls():
    rng = np.random.default_rng(923)
    for width in (9,128,256,512):
        bits = []
        for density in (.01,.07,.35,.62,.97):
            bits.extend(np.packbits(rng.random((113,8*width))<density,axis=1))
        bits = np.asarray(bits,dtype=np.uint8)
        bits[0:2] = 0
        bits[6] = bits[5]
        a,b = np.arange(0,len(bits),2), np.arange(1,len(bits),2)
        q,t = bits[b], bits[a]
        intersection = POPCOUNT[q[:,None,:]&t[None,:,:]].sum(axis=2)
        union = POPCOUNT[q[:,None,:]|t[None,:,:]].sum(axis=2)
        expected = np.divide(intersection,union,out=np.ones(intersection.shape),where=union!=0).max(axis=1)
        np.testing.assert_allclose(matrix_tanimoto(bits,a,b),expected,atol=3e-8,rtol=0)


def feature_sources(dataset, plan, release, cohort):
    default_plan = read(verify(STUDY65/"plan.json",read(STUDY65/"plan_seal.json")["sha256"]))
    default_job = next(j for j in default_plan["jobs"] if j["task"]=="yield" and j["dataset"]==dataset
                       and j["method"]=="synrfp_default")
    paths = {"synrfp_default":Path(default_job["feature_path"])}
    for method,key in METHOD_SOURCE.items():
        if dataset=="ncs_bh":
            paths[method] = Path(plan["sources"][dataset]["features"][key])
        else:
            paths[method] = USPTO/"features/uspto_yield"/f"{key}.npz"
    for method, path in paths.items():
        pinned = set()
        for r in release.loc[(release.dataset==dataset) & (release.method==method)].itertuples():
            rec = read(verify(Path(r.source_result_path),r.source_result_sha256))
            if method=="synrfp_default":
                pinned.add(rec["feature_sha256"])
            else:
                pinned.add(rec["native_details"]["feature_sha256"])
        assert len(pinned)==1
        expected = pinned.pop()
        with np.load(verify(path,expected),allow_pickle=False) as z:
            x = z["X"]
            assert x.dtype==np.uint8 and x.ndim==2 and len(x)==len(cohort)
            if "row_ids" in z.files:
                np.testing.assert_array_equal(z["row_ids"].astype(str),cohort.row_id.to_numpy(str))
            if "source_positions" in z.files:
                np.testing.assert_array_equal(z["source_positions"],np.arange(len(cohort)))
        yield method,x,expected


def run(dataset, name, matrix, fingerprint_hash, plan):
    output = WORK/"distance"/dataset/name
    output.mkdir(parents=True,exist_ok=True)
    for entry in plan["splits"]:
        if entry["dataset"]!=dataset:
            continue
        split = entry["split"]
        path = Path(entry["path"])
        pin = plan["pinned_files"][str(path)]
        selectors = verify(path,pin)
        file = output/f"{split}.npz"
        receipt = file.with_suffix(".json")
        if receipt.exists():
            record = read(receipt)
            assert record["fingerprint_sha256"]==fingerprint_hash and record["selectors_sha256"]==pin
            verify(file,record["result_sha256"])
            continue
        with np.load(selectors,allow_pickle=False) as z:
            train,test = z["train"],z["test"]
        start = time.monotonic()
        similarity = matrix_tanimoto(matrix,train,test)
        assert np.isfinite(similarity).all() and similarity.min()>=0 and similarity.max()<=1
        np.savez_compressed(file,test=test,similarity=similarity)
        info = dict(fingerprint_sha256=fingerprint_hash,selectors_sha256=pin,result_sha256=digest(file),
                    training_rows=len(train),test_rows=len(test),seconds=time.monotonic()-start)
        tmp = receipt.with_suffix(".tmp")
        tmp.write_text(json.dumps(info,sort_keys=True)+"\n")
        tmp.replace(receipt)
        print(dataset,name,split,len(test),round(info["seconds"],2),"s",flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("own","author"))
    args = p.parse_args()
    WORK.mkdir(exist_ok=True)
    controls()
    plan = read(verify(ROOT/"plan.json",read(ROOT/"plan_seal.json")["sha256"]))
    if args.mode=="author":
        from author92 import prepare
        x,h = prepare(plan)
        run("ncs_bh","author_drfp",x,h,plan)
        return
    pub = RELEASE/"yield/cells.csv"
    release = pd.read_csv(verify(pub,read(RELEASE/"COMPLETE.json")["exported_outputs_sha256"]["yield/cells.csv"]),
                          low_memory=False)
    for dataset in ("ncs_bh","uspto_yield"):
        table = Path(plan["tables"][dataset]["path"])
        cohort = pd.read_csv(verify(table,plan["pinned_files"][str(table)]),dtype={"row_id":str})
        for method,x,h in feature_sources(dataset,plan,release,cohort):
            run(dataset,method,x,h,plan)


if __name__ == "__main__":
    main()
