"""Run small batches of pinned MiniBatchKMeans fits on the native author cohort."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from importlib.metadata import version
import multiprocessing as mp
import time

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics.cluster import contingency_matrix
from threadpoolctl import threadpool_limits

from common90 import BASELINES, COUNT, METHOD, N, ROOT, read, sha, write

STATE = {}
PARAMETERS = {}


def score(y, prediction):
    if y.shape != (N,) or prediction.shape != y.shape or (prediction < 0).any():
        raise ValueError("A complete author-order partition is required")
    sizes = np.unique(prediction, return_counts=True)[1]
    contingency = contingency_matrix(y, prediction, sparse=True)
    return dict(ami=float(adjusted_mutual_info_score(y, prediction)),
                ari=float(adjusted_rand_score(y, prediction)),
                nmi=float(normalized_mutual_info_score(y, prediction)),
                purity=float(contingency.max(axis=0).sum()/N),
                actual_clusters=len(sizes), largest_cluster_fraction=float(sizes.max()/N),
                singleton_reaction_fraction=float(np.count_nonzero(sizes == 1)/N),
                singleton_cluster_fraction=float(np.count_nonzero(sizes == 1)/len(sizes)))


def fit(task):
    method, k, seed = task
    folder = ROOT/"kmeans"
    out = folder/f"{method}__k{k}__s{seed}.npz"
    receipt = out.with_suffix(".json")
    if receipt.exists():
        record = read(receipt)
        if (record["method"],record["requested_k"],record["seed"]) != task or sha(out)!=record["partition_sha256"]:
            raise ValueError("Changed previously completed k-means cell")
        return record
    data = STATE[method]
    exact = STATE.get("exact_integer",False) and method in (METHOD,"drfp","rdkit_diff","rdkit_struct")
    if exact:
        from exact_init90 import integer_init
        initialization = integer_init(data)
        arithmetic = "exact_integer_sparse_dot"
    else:
        initialization = nullcontext()
        arithmetic = "native"
    start = time.monotonic()
    with threadpool_limits(limits=2), initialization:
        model = MiniBatchKMeans(n_clusters=k, random_state=seed, **PARAMETERS)
        prediction = model.fit_predict(data)
        if not np.array_equal(prediction,model.predict(data)):
            raise ValueError("Cluster centers could not replay assignments")
    np.savez_compressed(out,labels=prediction, centers=model.cluster_centers_)
    record = dict(method=method,algorithm="MiniBatchKMeans",requested_k=k,seed=seed,n=N,
                  **score(STATE["y"],prediction),source="new",
                  runtime_seconds=time.monotonic()-start,partition_sha256=sha(out),
                  initialization_arithmetic=arithmetic,
                  minibatch_steps=int(model.n_steps_))
    write(receipt,record)
    return record


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("arm", choices=("native","baselines"))
    cli.add_argument("--jobs",type=int,default=5)
    cli.add_argument("--exact-integer",action="store_true")
    args=cli.parse_args()
    p=read(ROOT/"PREPARED.json")
    STATE["exact_integer"] = args.exact_integer
    for package,v in p["kmeans"]["environment"].items():
        assert version(package)==v
    PARAMETERS.update(p["kmeans"]["parameters"])
    cohort = ROOT/"cohort.npz"
    assert sha(cohort)==p["local_sha256"][cohort.name]
    with np.load(cohort,allow_pickle=False) as c:
        STATE["y"]=c["labels"]
        k_raw,k_res=[len(np.unique(c[k])) for k in ("raw","residual")]
        order = c["source_positions"]
    assert (k_raw,k_res)==(378,77)
    if args.arm=="native":
        f=ROOT/"synrfp_bits.npy"
        assert sha(f)==p["local_sha256"][f.name]
        STATE[METHOD]=np.unpackbits(np.load(f,allow_pickle=False),axis=1).astype(np.float32)
        f=p["native_fingerprint_path"]
        assert sha(f)==p["input_sha256"][f]
        with np.load(f,allow_pickle=False) as z:
            np.testing.assert_array_equal(order,z["source_positions"])
            offsets, tokens, values=(z[k] for k in ("fingerprint_offsets","fingerprint_keys","fingerprint_counts"))
            vocab, cols = np.unique(tokens,return_inverse=True)
        assert len(offsets)==N+1 and offsets[-1]==len(tokens) and len(tokens)==len(values)
        data=csr_matrix((values.astype(np.float32),cols,offsets),shape=(N,len(vocab)))
        assert data.has_canonical_format
        STATE[COUNT]=data
        workloads={METHOD:(k_raw,k_res),COUNT:(147,k_raw)}
    else:
        workloads={k:(k_raw,) for k in BASELINES}
        for name in BASELINES:
            f=ROOT/f"{name}_features.npz"
            receipt=read(f.with_suffix(".json"))
            assert sha(f)==receipt["sha256"]
            with np.load(f,allow_pickle=False) as z:
                np.testing.assert_array_equal(order,z["source_positions"])
                assert z["ok"].all()
                d=z["X"]
                STATE[name]=np.unpackbits(d,axis=1).astype(np.float32) if d.dtype==np.uint8 else d
    folder=ROOT/"kmeans";folder.mkdir(exist_ok=True)
    tasks=[(fp,k,s) for s in p["kmeans"]["optimizer_seeds"] for fp,ks in workloads.items() for k in ks]
    out={}
    start=time.monotonic()
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=mp.get_context("fork")) as pool:
        for result in pool.map(fit,tasks,chunksize=1):
            tag=f"{result['method']}__k{result['requested_k']}__s{result['seed']}.json"
            out[tag]=sha(folder/tag)
            print(tag,f"{result['ami']:.6f}",f"{result['runtime_seconds']:.1f}s",flush=True)
    write(ROOT/f"kmeans_{args.arm}_COMPLETE.json",dict(artifacts=out,runtime_seconds=time.monotonic()-start,
           prepared_sha256=sha(ROOT/"PREPARED.json"),source_sha256=sha(__file__),n_fits=len(tasks)))


if __name__=="__main__":
    main()
