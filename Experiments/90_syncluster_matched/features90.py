"""Reuse exact release-input vectors and encode the remaining author rows."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import inspect
from importlib.metadata import version
import multiprocessing as mp

import numpy as np
import pandas as pd

from common90 import BASE, BASELINES, EXPERIMENTS, N, R86, ROOT, TMP, Pins, read, sha, write

SOURCE_KEY = "uspto_50k_u"
NEURAL = ("crxnfp","rxnfp_mlm")


def load_reference(pins,name):
    study = BASE/"all_endpoint_factorial_full_v1"
    plan = read(pins.add(study/"plan.json",read(pins.add(study/"plan_seal.json"))["sha256"]))
    source = plan["sources"][SOURCE_KEY]["table"]
    rows = pd.read_csv(pins.add(source,plan["pinned_files"][source]),keep_default_na=False)
    np.testing.assert_array_equal(rows.source_position,np.arange(len(rows)))
    root = TMP/"runs/crxnfp_full_v1"
    prior = read(pins.add(root/"plan.json"))
    if name=="crxnfp":
        path=root/"encoder/features"/f"{SOURCE_KEY}.npz"
        info=read(pins.add(path.with_suffix(".json")))
        expected=info["feature_sha256"]
    else:
        path=TMP/"work/validation/n1/features"/SOURCE_KEY/f"{name}.npz"
        expected=prior["inputs_sha256"][str(path)]
    with np.load(pins.add(path,expected),allow_pickle=False) as f:
        X,ok=f["X"],f["ok"]
        for key,column in (("row_ids","row_id"),("source_positions","source_position")):
            if key in f.files:
                np.testing.assert_array_equal(f[key],rows[column])
    assert len(X)==len(ok)==len(rows) and np.isfinite(X).all()
    return rows.rxn.to_numpy(),X,ok


def classical(texts,name):
    import sys
    sys.path.insert(0,str(EXPERIMENTS))
    from synrfp_exp.features import _drfp_chunk, _rdkit_chunk
    if name=="drfp":
        x,ok = _drfp_chunk(texts,dict(n_folded_length=2048,radius=3,rings=True))
    else:
        x,ok = _rdkit_chunk(texts,name)
    return x,ok,[]


def neural(task):
    from neural86 import infer
    texts, name = task
    jobs=[({"endpoint":"native_author_uspto50k_ub","source":SOURCE_KEY,"sample":i,
            "original":text,"row_id":str(i)},"original") for i,text in enumerate(texts)]
    x,ok,issues,_ = infer(jobs)
    return x,ok,issues


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("name",choices=BASELINES)
    args=cli.parse_args()
    name=args.name
    pins=Pins()
    prep = read(pins.add(ROOT/"PREPARED.json"))
    rows = pd.read_csv(pins.add(ROOT/"inputs.csv",prep["local_sha256"]["inputs.csv"]),keep_default_na=False,
                       dtype={"row_id":str})
    strings=rows.rxn.to_numpy()
    assert len(strings)==N
    old,x,ok=load_reference(pins,name)
    lookup={}
    tolerance=2e-5 if name in NEURAL else 0.
    for i in np.flatnonzero(ok):
        text=old[i]
        if text in lookup:
            np.testing.assert_allclose(x[lookup[text]],x[i],rtol=0,atol=tolerance)
        else:
            lookup[text]=i
    unique, inverse=np.unique(strings,return_inverse=True)
    out=np.zeros((len(unique),x.shape[1]),x.dtype)
    valid=np.zeros(len(unique),bool)
    loc=np.array([lookup.get(text,-1) for text in unique])
    matched=loc>=0
    out[matched]=x[loc[matched]]
    valid[matched]=True
    # Replay a fixed sample of shared texts in addition to newly encoded texts.
    available=np.flatnonzero(matched)
    replay=available[np.linspace(0,len(available)-1,64,dtype=int)]
    missing=np.flatnonzero(~matched)
    indexes=np.r_[replay,missing]
    if name in NEURAL:
        from neural86 import initialize, STATE
        initialize(name)
        tasks=[(unique[indexes[s:s+24]].tolist(),name) for s in range(0,len(indexes),24)]
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context("fork")) as pool:
            shards=list(pool.map(neural,tasks))
        for f in STATE["pins"]:
            pins.add(f)
        pins.add(R86/"neural86.py")
        versions={p:version(p) for p in ("rdkit","numpy","torch","transformers","rxnfp")}
    else:
        shards=[classical(unique[indexes].tolist(),name)]
        pins.add(EXPERIMENTS/"synrfp_exp/features.py")
        versions={p:version(p) for p in ("rdkit","numpy","drfp")}
    fresh=np.concatenate([r[0] for r in shards])
    success=np.concatenate([r[1] for r in shards])
    assert np.all(success[:len(replay)])
    replay_error=float(np.abs(fresh[:len(replay)]-out[replay]).max(initial=0)) if name in NEURAL else 0.
    if name in NEURAL:
        np.testing.assert_allclose(fresh[:len(replay)],out[replay],rtol=0,atol=tolerance)
    else:
        np.testing.assert_array_equal(fresh[:len(replay)],out[replay])
    out[missing]=fresh[len(replay):]
    valid[missing]=success[len(replay):]
    populated=valid[inverse]
    if not populated.all():
        write(ROOT/f"{name}_REFUSAL.json",dict(n=int((~populated).sum()),
              source_positions=rows.source_position.to_numpy()[~populated].tolist(),
              detailed_events=[x for shard in shards for x in shard[2]]))
        raise ValueError("The encoder refused some native rows. The frozen cohort cannot shrink.")
    dest=ROOT/f"{name}_features.npz"
    np.savez_compressed(dest,X=out[inverse],ok=populated,source_positions=rows.source_position.to_numpy())
    write(dest.with_suffix(".json"),dict(sha256=sha(dest),rows=N,
          reused_rows=int(matched[inverse].sum()),new_unique_inputs=len(missing),
          tests_replayed=len(replay),max_replay_error=replay_error,
          versions=versions,pins=pins.files))
    print(name,"reused",int(matched[inverse].sum()),"of",N,"rows",flush=True)


if __name__=="__main__":
    main()
