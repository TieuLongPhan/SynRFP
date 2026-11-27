"""Run and checkpoint the paired CPU forests and binary MLP classifiers."""
import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import time
import traceback

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from threadpoolctl import threadpool_limits

from plan93 import (AUTHOR, AUTHOR_SHA, BENCHMARK, CLASS, CONTEXT, ROOT, PARTS, MLP, read, selectors,
                    sha, table, verify, write)

PLAN = None
PLAN_SHA = None
STATE = {}
RESERVE = 40 * 2**30
ZSTD = Path.home()/"anaconda3/bin/zstd"


def path_for(dataset, split, method, learner):
    return ROOT/"models"/dataset/split/method/learner


def outputs(task):
    return path_for(*task)/"result.json"


def source_indices(entry):
    global STATE
    source = PLAN["sources"][entry["dataset"]]
    frame = table(source["table"],source["sha256"])
    key=(entry["dataset"],entry["split"])
    if key not in STATE:
        bound=entry["inputs"]["synrfp_default"]
        STATE[key]=selectors(bound,frame)
    return frame,STATE[key]


def load_archive(bound, target, selected, needs_validation):
    refs=bound["fit"]
    file=refs["fit.npz"]
    with np.load(verify(file["path"],file["sha256"]),allow_pickle=False) as fit:
        xt,yt=fit["X_train"],fit["y_train"]
        xv=fit["X_val"] if needs_validation else None
    file=refs["preprocessing.npz"]
    with np.load(verify(file["path"],file["sha256"]),allow_pickle=False) as stats:
        mean,scale=float(stats["target_mean"]),float(stats["target_scale"])
    np.testing.assert_array_equal(yt,((target[selected["train"]]-mean)/scale).astype(np.float32))
    file=refs["test.npy"]
    xe=np.load(verify(file["path"],file["sha256"]),allow_pickle=False,mmap_mode="r")
    assert xt.dtype==xe.dtype==np.float32 and xt.shape[1]==xe.shape[1]
    return xt,xv,xe,yt,mean,scale


def context_data(dataset, method, frame, selected):
    manifest=PLAN["sources"][dataset]["context"][method]
    expected=manifest.get("sha256")
    if expected is None:
        expected=read(ROOT/"uspto_context/COMPLETE.json")["features"][method]
    key=(dataset,method)
    if key not in STATE:
        with np.load(verify(manifest["path"],expected),allow_pickle=False) as z:
            assert int(z["n_bits"])==2048 and z["ok"].all()
            np.testing.assert_array_equal(z["row_ids"].astype(str),frame.row_id.to_numpy())
            np.testing.assert_array_equal(z["source_positions"],np.arange(len(frame)))
            STATE[key]=z["X"]
    x=STATE[key]
    tr,te=(np.unpackbits(x[selected[s]],axis=1).astype(np.float32) for s in ("train","test"))
    vals=frame.yield_pct.to_numpy()[selected["train"]]
    mean,scale=float(vals.mean()),float(vals.std())
    if scale<1e-12:
        scale=1.
    return tr,None,te,((vals-mean)/scale).astype(np.float32),mean,scale


def check_installed():
    return {p:version(p) for p in ("torch","scikit-learn","numpy","pandas")}


def sigmoid_mlp(xt, xv, xe, train_labels, validation_labels, model_path, params=MLP):
    """Use the archived yield MLP architecture and replace only its objective."""
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(params["seed"])
    model=nn.Sequential(nn.Linear(xt.shape[1],params["hidden"]),nn.ReLU(),nn.Linear(params["hidden"],1))
    optim=torch.optim.Adam(model.parameters(),lr=params["learning_rate"],weight_decay=params["weight_decay"])
    loss=nn.BCEWithLogitsLoss()
    tr,val=torch.from_numpy(xt),torch.from_numpy(xv)
    y,yv=(torch.tensor(t,dtype=torch.float32) for t in (train_labels,validation_labels))
    assert len(tr)==len(y) and len(val)==len(yv) and len(y)==len(train_labels)
    best,stop,stale=float("inf"),float("inf"),0
    state=None
    selection=0
    for epoch in range(1,params["max_epochs"]+1):
        model.train()
        for indices in torch.randperm(len(tr)).split(params["batch_size"]):
            optim.zero_grad(set_to_none=True)
            loss(model(tr[indices]).squeeze(-1),y[indices]).backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            v=float(loss(model(val).squeeze(-1),yv))
        if not np.isfinite(v):
            raise ValueError("Nonfinite validation BCE")
        if v<best:
            best,selection,state=v,epoch,{k:t.detach().clone() for k,t in model.state_dict().items()}
        if v<stop-params["min_delta"]:
            stop,stale=v,0
        else:
            stale+=1
        if stale>=params["patience"]:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        test=torch.tensor(np.asarray(xe),dtype=torch.float32)
        probability=torch.cat([model(part).sigmoid().squeeze(-1) for part in test.split(4096)]).numpy()
    temp=model_path.with_suffix(".tmp.pt")
    torch.save(dict(state_dict=state,selected_epoch=selection,epochs=epoch,parameters=params,
                    input_dim=xt.shape[1],best_validation_bce=best),temp)
    temp.replace(model_path)
    return probability,dict(selected_epoch=selection,epochs=epoch,best_validation_bce=best)


def archive_forest(model, path):
    temp=path.with_suffix(".tmp.zst")
    with temp.open("wb") as dest:
        process=subprocess.Popen([str(ZSTD),"-q","-c","-T1","-3"],stdin=subprocess.PIPE,stdout=dest)
        try:
            pickle.dump(model,process.stdin,protocol=5)
            process.stdin.close()
            if process.wait()!=0:
                raise RuntimeError("Forest compression failed")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    temp.replace(path)


def released_labels():
    if "labels" not in STATE:
        label=pd.read_csv(verify(AUTHOR,AUTHOR_SHA),usecols=["labels"])["labels"].to_numpy()
        assert len(label)==27497 and np.isin(label,[0,1]).all()
        STATE["labels"]=label
    return STATE["labels"]


def fit(task):
    dataset,split,representation,learner=task
    destination=path_for(*task)
    result=destination/"result.json"
    if result.exists():
        prev=read(result)
        assert prev["plan_sha256"]==PLAN_SHA and prev["key"]==list(task)
        for name,digest in prev["artifacts"].items():
            verify(destination/name,digest)
        return prev
    destination.mkdir(parents=True,exist_ok=True)
    entry=STATE["groups"][dataset,split]
    frame,selected=source_indices(entry)
    started=time.monotonic()
    target=frame.yield_pct.to_numpy()
    if representation in CONTEXT:
        xt,xv,xe,y,mean,scale=context_data(dataset,representation,frame,selected)
    else:
        xt,xv,xe,y,mean,scale=load_archive(entry["inputs"][representation],target,selected,learner=="mlp_classifier")
    classification=learner.endswith("_classifier")
    if classification:
        label=released_labels()
        y=label[selected["train"]]
        yt=label[selected["test"]]
        assert np.unique(y).size==np.unique(yt).size==2
    with threadpool_limits(limits=1):
        if learner=="mlp_classifier":
            weight=destination/"best_model.pt"
            predictions,details=sigmoid_mlp(xt,xv,xe,y,label[selected["validation"]],weight)
        else:
            hp=dict(n_estimators=500,max_features="sqrt",min_samples_leaf=1,n_jobs=1,random_state=42)
            forest=RandomForestClassifier(class_weight="balanced",**hp) if classification else RandomForestRegressor(**hp)
            forest.fit(xt,y)
            weight=destination/"forest.pkl.zst"
            if classification:
                assert forest.classes_.tolist()==[0,1]
                predictions=forest.predict_proba(xe)[:,1]
            else:
                predictions=forest.predict(xe)*scale+mean
            if shutil.disk_usage(ROOT).free < RESERVE:
                raise RuntimeError("Filesystem reserve reached before archiving the model")
            archive_forest(forest,weight)
            details=dict(tree_nodes=sum(est.tree_.node_count for est in forest.estimators_))
    assert predictions.shape==(entry["ntest"],) and np.isfinite(predictions).all()
    if classification:
        assert predictions.min()>=0 and predictions.max()<=1
        scores=dict(roc_auc=roc_auc_score(yt,predictions),
                    balanced_accuracy=balanced_accuracy_score(yt,predictions>=.5))
    else:
        yt=target[selected["test"]]
        scores=dict(r2=r2_score(yt,predictions),mae=mean_absolute_error(yt,predictions),
                    rmse=float(np.sqrt(mean_squared_error(yt,predictions))))
    pred=destination/"predictions.npy"
    temp=pred.with_suffix(".tmp.npy")
    np.save(temp,predictions,allow_pickle=False)
    temp.replace(pred)
    record=dict(key=list(task),plan_sha256=PLAN_SHA,panel=entry["panel"],split=split,dataset=dataset,
                representation=representation,learner=learner,n_train=entry["ntrain"],n_test=len(yt),
                runtime_seconds=time.monotonic()-started,model_bytes=weight.stat().st_size,
                artifacts={p.name:sha(p) for p in (pred,weight)},**details,**scores)
    write(result,record)
    print(dataset,split,representation,learner,round(record["runtime_seconds"],2),"seconds",flush=True)
    return record


def planned(plan):
    for g in plan["groups"]:
        dataset,split=g["dataset"],g["split"]
        for rep in (*BENCHMARK,*CONTEXT):
            yield dataset,split,rep,"rf_regressor"
        if dataset=="ncs_bh":
            for rep in CLASS:
                for learner in ("rf_classifier","mlp_classifier"):
                    yield dataset,split,rep,learner


def memory_available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1])*1024


def run(jobs):
    global PLAN,PLAN_SHA,STATE
    p=ROOT/"PLAN.json"
    PLAN,PLAN_SHA=read(p),sha(p)
    STATE["groups"]={(g["dataset"],g["split"]):g for g in PLAN["groups"]}
    versions=check_installed()
    tickets=list(planned(PLAN))
    assert len(tickets)==941 and len(tickets)==len(set(tickets))
    recovered=[fit(t) for t in tickets if outputs(t).is_file()]
    done={tuple(r["key"]) for r in recovered}
    pending=deque(sorted((t for t in tickets if t not in done),key=lambda a:(a[0]!="uspto_yield",
             a[3]!="rf_classifier",a[3]=="mlp_classifier",a[1],a[2])))
    started=time.monotonic()
    completed=recovered
    instant_failure=None
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        live={}
        while pending or live:
            while pending and len(live)<jobs and instant_failure is None:
                if shutil.disk_usage(ROOT).free < RESERVE+8*2**30:
                    instant_failure="Filesystem reserve reached"
                    break
                if memory_available()<16*2**30:
                    break
                heavy=sum(t[0]=="uspto_yield" for t in live.values())
                i=next((i for i,t in enumerate(pending) if t[0]!="uspto_yield" or heavy<4),None)
                if i is None:
                    break
                ticket=pending[i]
                del pending[i]
                live[pool.submit(fit,ticket)]=ticket
            if not live:
                if instant_failure:
                    break
                time.sleep(20)
                continue
            closed,_=wait(live,return_when=FIRST_COMPLETED)
            for future in closed:
                ticket=live.pop(future)
                try:
                    completed.append(future.result())
                except Exception:
                    instant_failure=traceback.format_exc()
                    write(ROOT/"FAILED.json",dict(ticket=list(ticket),error=instant_failure))
                    print(ticket,instant_failure,flush=True)
            write(ROOT/"PROGRESS.json",dict(complete=len(completed),remaining=len(tickets)-len(completed),
                 failed=instant_failure is not None,models_bytes=sum(q["model_bytes"] for q in completed),
                 elapsed_seconds=time.monotonic()-started))
    if instant_failure:
        raise RuntimeError(instant_failure)
    assert len(completed)==len(tickets)
    write(ROOT/"FIT_COMPLETE.json",dict(cells=len(completed),plan_sha256=PLAN_SHA,versions=versions,
          elapsed_seconds=time.monotonic()-started,jobs=jobs,models_bytes=sum(r["model_bytes"] for r in completed)))


if __name__=="__main__":
    app=argparse.ArgumentParser(description=__doc__)
    app.add_argument("--jobs",type=int,required=True)
    run(app.parse_args().jobs)
