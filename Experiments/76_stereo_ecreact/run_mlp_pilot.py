"""Five-partition stereo pilot reusing study 65 initialization and Adam loop."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import inspect
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
import torch

# The CPU runner needs no chemistry imports.
from hashlib import file_digest

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
BASE=Path.home()/".local/share/synrfp-runs"
STUDY=BASE/"default_optimal_benchmark_v1"
ENCODE=BASE/"ec_stereo_request76_v2"
LOCAL=BASE/"ec_stereo_request76_mlp_v1"
OUTPUT=HERE/"publication_metadata/stereo_ecreact_v1/mlp_pilot"
WORKER=REPO/"Experiments/12_validation/crxnfp_full_worker.py"
SOURCE_RELEASE=REPO.parent/"SynRFP/Experiments/65_default_optimal/publication_metadata/default_optimal_v1/classification/cells.csv"
ENDPOINTS=("ecreact_2","ecreact_3")
SEEDS=tuple(range(1,6))
ARMS=("delta","delta_union")
BASELINES=("drfp","rdkit_diff","rdkit_struct","rxnfp_mlm","crxnfp")
MODEL={"hidden":[512],"alpha":0.0001,"max_iter":200,"batch_size":256}
STATE={}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as handle:
        return file_digest(handle,"sha256").hexdigest()


def check(path, expected):
    if digest(path)!=expected:
        raise ValueError(f"Changed frozen dependency {path}")
    return Path(path)


def write(path,value):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
    temporary.replace(path)


def export(path, records):
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def native_train(packed, outer, test):
    import importlib.util
    spec=importlib.util.spec_from_file_location("native_stereo76_worker",WORKER)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source=inspect.getsource(module.train)
    substitutions={
        "X = np.load(directory / 'train.npy', mmap_mode='r')":"X = None",
        "torch.tensor(X[arrays['train']], device=device)":"LOAD_OUTER(arrays['train'])",
        "torch.tensor(X[arrays['val']], device=device)":"LOAD_OUTER(arrays['val'])",
        "torch.cuda.synchronize(device)":"None",
        "torch.cuda.max_memory_allocated(device)":"0",
        "torch.tensor(np.load(directory / 'test.npy'), device=device)":"LOAD_TEST()",
    }
    for old,new in substitutions.items():
        if source.count(old)!=1:
            raise ValueError(f"Unexpected native training-loader boundary: {old}")
        source=source.replace(old,new)

    def tensor(rows):
        return torch.from_numpy(np.unpackbits(packed[rows],axis=1,count=2048)).float()

    module.LOAD_OUTER=lambda idx: tensor(outer[idx])
    module.LOAD_TEST=lambda: tensor(test)
    exec(compile(source,str(WORKER)+"[packed-cpu-loader]","exec"),module.__dict__)
    return module.train


def archive(endpoint,seed):
    case=STUDY/"results"/f"classification__{endpoint}__mlp__{seed}__synrfp_optimal/result.json"
    ref=read(case)
    cell=STATE["scalars"][endpoint,seed,"synrfp_optimal"]
    check(case,cell["source_result_sha256"])
    original_worker=ref["sources_sha256"]
    original_key=next(k for k in original_worker if k.endswith("/crxnfp_full_worker.py"))
    check(WORKER,original_worker[original_key])
    assert ref["training_params"]==MODEL and ref["seed"]==seed
    artifact=Path(ref["attempt_directory"])
    expected=ref["artifacts_sha256"]
    frozen={}
    for name in ("state.npz","selection.npz","preprocessing.npz"):
        frozen[name]=check(artifact/name,expected[name])
    panel=STATE["panels"][endpoint]
    assert ref["identity"]["split_sha256"]==panel["split_sha256"][str(seed)]
    code=np.load(check(panel["split_paths"][str(seed)],panel["split_sha256"][str(seed)]),allow_pickle=False)
    frame=STATE["tables"][endpoint]
    positions=frame.source_position.to_numpy()
    with np.load(frozen["selection.npz"],allow_pickle=False) as z:
        outer,test=z["outer"],z["test"]
    base_outer=np.r_[np.flatnonzero(code==0),np.flatnonzero(code==1)]
    np.testing.assert_array_equal(outer,positions[base_outer])
    np.testing.assert_array_equal(test,positions[code==2])
    with np.load(frozen["preprocessing.npz"],allow_pickle=False) as z:
        classes=z["classes"]
    with np.load(frozen["state.npz"],allow_pickle=False) as z:
        np.testing.assert_array_equal(classes[z["y"]],frame.y.to_numpy()[base_outer])
        np.testing.assert_array_equal(np.sort(np.r_[z["train"],z["val"]]),np.arange(len(outer)))
    return frozen,outer,test,classes,code,ref


def fit(job):
    endpoint,seed,arm,threads,benchmark=job
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    label=f"{endpoint}__s{seed}__{arm}"
    working=LOCAL/("timing" if benchmark else "fits")/label
    working.mkdir(parents=True,exist_ok=True)
    frozen,outer,test,classes,code,native=archive(endpoint,seed)
    feature=STATE["features"][arm]
    envelope={"profile":"stereo_ez","arm":arm,"endpoint":endpoint,"seed":seed,
              "fit_options":MODEL if not benchmark else dict(MODEL,max_iter=1),
              "device":"torch_cpu","torch":torch.__version__,"threads":threads,
              "feature_sha256":STATE["feature_sha256"],"source_worker_sha256":digest(WORKER),
              "source_state_sha256":digest(frozen["state.npz"]),
              "source_selector_sha256":digest(frozen["selection.npz"]),
              "runner_sha256":digest(Path(__file__)),"benchmark":benchmark}
    receipt=working/"COMPLETE.json"
    if receipt.exists():
        data=read(receipt)
        if data["identity"]!=envelope:
            raise ValueError("Saved CPU fit identity differs")
        for name,expected in data["artifacts_sha256"].items():
            check(working/name,expected)
        return data
    shutil.copyfile(frozen["state.npz"],working/"state.npz")
    write(working/"job.json",{"params":envelope["fit_options"],"id":label})
    started=time.monotonic()
    training=native_train(feature,outer,test)(working,"cpu")
    artifacts={k:digest(working/k) for k in ("best_weights.npz","predictions.npy","state.npz")}
    record={"identity":envelope,"training":training,"n_test":len(test),
            "source_native_epochs":native["epochs"],"elapsed":time.monotonic()-started,
            "artifacts_sha256":artifacts}
    if not benchmark:
        with np.load(working/"best_weights.npz",allow_pickle=False) as z:
            weights=[torch.tensor(z[k]) for k in ("w0","w1","b0","b1")]
        w0,w1,b0,b1=weights
        replay=[]
        with torch.no_grad():
            for i in range(0,len(test),8192):
                x=torch.from_numpy(np.unpackbits(feature[test[i:i+8192]],axis=1,count=2048)).float()
                logits=(x@w0+b0).relu()@w1+b1
                replay.append(logits.argmax(1).numpy() if w1.shape[1]>1 else (logits[:,0]>0).long().numpy())
        raw=np.load(working/"predictions.npy",allow_pickle=False)
        np.testing.assert_array_equal(raw,np.concatenate(replay))
        observed=STATE["tables"][endpoint].y.to_numpy()[code==2]
        record["mcc"]=float(matthews_corrcoef(observed,classes[raw]))
        record["replayed_predictions"]=True
    write(receipt,record)
    print(label,"epochs",training["epochs"],"seconds",round(training["fit_seconds"],1),flush=True)
    return record


def inputs():
    plan_path=STUDY/"plan.json"
    plan=read(check(plan_path,read(STUDY/"plan_seal.json")["sha256"]))
    native=STUDY/"native/classification/plan.json"
    panels=read(check(native,plan["inputs_sha256"][str(native)]))["panels"]
    archive=ENCODE/"mlp_features.npz"
    metadata=read(ENCODE/"MLP_FEATURES.json")
    features={}
    tables={}
    with np.load(check(archive,metadata["artifact_sha256"]),allow_pickle=False) as saved:
        for arm in ARMS:
            features[arm]=saved[arm]
        ids=saved["row_ids"]
        for endpoint in ENDPOINTS:
            panel=panels[endpoint]
            frame=pd.read_csv(check(panel["table"],panel["table_sha256"]),dtype={"row_id":str})
            np.testing.assert_array_equal(ids[frame.source_position],frame.row_id.to_numpy(str))
            tables[endpoint]=frame
    with SOURCE_RELEASE.open(newline="") as handle:
        scalars={(r["dataset"],int(float(r["seed"])),r["method"]):r for r in csv.DictReader(handle)
                 if r["model"]=="mlp" and r["dataset"] in ENDPOINTS and int(float(r["seed"])) in SEEDS}
    STATE.update(scalars=scalars,panels=panels,tables=tables,features=features,feature_sha256=metadata["artifact_sha256"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing",action="store_true")
    parser.add_argument("--jobs",type=int,default=4)
    parser.add_argument("--threads",type=int,default=5)
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":
        raise ValueError("This pilot uses CPU only")
    inputs()
    if args.timing:
        fit((ENDPOINTS[0],SEEDS[0],ARMS[0],args.threads,True))
        return
    started=time.monotonic()
    jobs=[(endpoint,seed,arm,args.threads,False) for seed in SEEDS
          for endpoint in ENDPOINTS for arm in ARMS]
    scores=[]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for task in as_completed([pool.submit(fit,job) for job in jobs]):
            scores.append(task.result())
            print(f"Verified {len(scores)}/20 stereo MLP fits",flush=True)
    rows=[]
    for v in scores:
        i=v["identity"]
        rows.append({"endpoint":i["endpoint"],"partition_seed":i["seed"],
                     "repeat":i["seed"]//5,"fold":i["seed"]%5,
                     "arm":"stereo_ez_"+i["arm"],"mcc":v["mcc"],
                     "origin":"new_native_optimizer_cpu","n_test":v["n_test"]})
    for endpoint in ENDPOINTS:
        for seed in SEEDS:
            for arm in BASELINES:
                row=STATE["scalars"][endpoint,seed,arm]
                path=row.get("source_result_path")
                sha=row.get("source_result_sha256")
                if path and sha:
                    check(path,sha)
                if row["split_sha256"]!=STATE["panels"][endpoint]["split_sha256"][str(seed)]:
                    raise ValueError("Baseline selector changed")
                rows.append({"endpoint":endpoint,"partition_seed":seed,"repeat":seed//5,"fold":seed%5,
                             "arm":arm,"mcc":float(row["mcc"]),"origin":"archived_study65","n_test":int(float(row["n_test"]))})
    assert len(rows)==70
    pairs=[]
    for row in rows:
        if row["origin"]=="archived_study65":
            continue
        endpoint,seed=row["endpoint"],row["partition_seed"]
        old="synrfp_default" if row["arm"]=="stereo_ez_delta" else "synrfp_optimal"
        native=STATE["scalars"][endpoint,seed,old]
        pairs.append({**row,"ordinary_arm":old,"ordinary_mcc":float(native["mcc"]),
                      "stereo_minus_ordinary_mcc":row["mcc"]-float(native["mcc"])})
    OUTPUT.mkdir(parents=True,exist_ok=True)
    rows.sort(key=lambda r:(r["endpoint"],r["partition_seed"],r["arm"]))
    export(OUTPUT/"cells.csv",rows)
    export(OUTPUT/"original_arm_comparison.csv",sorted(pairs,key=lambda r:(r["endpoint"],r["partition_seed"],r["arm"])))
    grouped=pd.DataFrame(rows).groupby(["endpoint","arm"],sort=True).mcc.agg(["size","mean","std"]).reset_index()
    assert (grouped["size"]==5).all()
    export(OUTPUT/"summary.csv",grouped.to_dict("records"))
    write(OUTPUT/"AUDIT.json",{"new_fits":20,"archived_fits":50,"partition_seeds":SEEDS,"native_optimizer":MODEL,
                              "new_backend":"torch_cpu","torch_version":torch.__version__,
                              "elapsed_seconds":time.monotonic()-started,
                              "feature_sha256":STATE["feature_sha256"],
                              "original_scalar_release_sha256":digest(SOURCE_RELEASE),
                              "all_new_fit_weights_replayed":True,"metrics":"descriptive across five saved partitions",
                              "fit_receipts":[{"identity":r["identity"],"result_sha256":digest(
                                  LOCAL/"fits"/f"{r['identity']['endpoint']}__s{r['identity']['seed']}__{r['identity']['arm']}"/"COMPLETE.json")}
                                  for r in sorted(scores,key=lambda r:(r["identity"]["endpoint"],r["identity"]["seed"],r["identity"]["arm"]))]})
    write(OUTPUT/"COMPLETE.json",{"status":"complete","n_fits":70,
                                  "files_sha256":{p.name:digest(p) for p in OUTPUT.iterdir()
                                                   if p.is_file() and p.name!="COMPLETE.json"}})


if __name__=="__main__":
    main()
