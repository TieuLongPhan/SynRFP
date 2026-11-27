"""Compare accelerated initializations against archived and unmodified model fits."""
import argparse
import time

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits

from common90 import HERE, METHOD, ROOT, SOURCE, read, sha, write
from exact_init90 import integer_init
from nativebutina90 import packed

CONTROLS = (("synrfp",147,0),("synrfp",147,1),("synrfp",378,0),("drfp",378,0))


def control(method,k,seed):
    if k==147:
        path=SOURCE/"matched_synrfp"/f"seed_{seed}.npz"
        field="artifact_sha256"
        centers="cluster_centers"
    else:
        tag=METHOD if method=="synrfp" else method
        path=ROOT/"kmeans"/f"{tag}__k{k}__s{seed}.npz"
        field="partition_sha256"
        centers="centers"
    assert read(path.with_suffix(".json"))[field]==sha(path)
    return path,centers


def fast_control(task):
    method,k,seed=task
    X=np.unpackbits(packed(method),axis=1).astype(np.float32)
    start=time.monotonic()
    with threadpool_limits(limits=2), integer_init(X):
        est=MiniBatchKMeans(n_clusters=k,random_state=seed,**read(ROOT/"PREPARED.json")["kmeans"]["parameters"])
        est.fit(X)
    dest=ROOT/f"{method}_init_proof_k{k}_s{seed}.npz"
    np.savez(dest,labels=est.labels_,centers=est.cluster_centers_)
    write(dest.with_suffix('.json'),dict(output_sha256=sha(dest),source_sha256=sha(HERE/"exact_init90.py"),
          elapsed_seconds=time.monotonic()-start))
    print(method,k,seed,round(time.monotonic()-start,2),flush=True)


def verify():
    outcomes={}
    for method,k,seed in CONTROLS:
        path,feature=control(method,k,seed)
        dest=ROOT/f"{method}_init_proof_k{k}_s{seed}.npz"
        if method=="synrfp" and not dest.exists():
            dest=ROOT/f"init_proof_k{k}_s{seed}.npz"
        with np.load(dest,allow_pickle=False) as a,np.load(path,allow_pickle=False) as b:
            np.testing.assert_array_equal(a["labels"],b["labels"])
            np.testing.assert_array_equal(a["centers"],b[feature])
        outcomes[f"{method}_k{k}_s{seed}"]=dict(fast_result_sha256=sha(dest),native_result_sha256=sha(path),
                                              identical_centers_and_memberships=True)
    write(ROOT/"KMEANS_INTEGER_REPLAY.json",dict(controls=outcomes,source_sha256=sha(HERE/"exact_init90.py")))
    print("Exact centers and memberships verified for four actual-cohort controls",flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only",choices=("none","drfp"),default="none")
    p.add_argument("--verify",action="store_true")
    a=p.parse_args()
    if a.verify:
        verify()
    else:
        for task in CONTROLS:
            if a.only=="none" or task[0]==a.only:
                fast_control(task)
