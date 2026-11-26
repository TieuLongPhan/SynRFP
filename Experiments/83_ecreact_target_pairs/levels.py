"""Freeze a common EC level-3 split schedule and replay all compatible saved fits."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix

from mapping import BASE, EM_CODE, HERE, OUT, REPO, ROOT, checked_enzymemap, read, sha, write
from audit_enzymemap_labels import classification_diagnostic

WORKSPACE = REPO.parent
ORIGINAL = WORKSPACE / "SynRFP"
REQ75 = WORKSPACE / "SynRFP-chemistry-cases"
CODE75 = REQ75 / "Experiments/75_chemistry_cases"
sys.path.insert(0, str(CODE75))
from common import InputAudit, feature, load_plan, panel_data, source_data
from classification import MODE, comparison_inputs

HEAD75 = "9e2c2de9c38d7ad342bbd86e744d5684b3024e2f"
ROOT75 = CODE75 / "publication_metadata/chemistry_cases_v1"
NATIVE = ORIGINAL / "Experiments/16_modes/classification_worker.py"
INITIALIZATION = ORIGINAL / "Experiments/22_all_endpoint_factorial/models.py"
GROUPS = Path("/var/tmp/synrfp-labhhc4-storage/work/validation/n1/groups/ecreact.npy")
MAIN = ORIGINAL / "Experiments/65_default_optimal/publication_metadata/default_optimal_v1/classification/cells.csv"
LEVELS = [f"ecreact_{n}" for n in (1, 2, 3)]
PARAMS = dict(hidden=[512], alpha=.0001, batch_size=256, max_iter=200)
ARGS = {}
TORCH_PYTHON = Path.home() / "anaconda3/envs/synprop/bin/python"


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    out = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(out)
    return out


def initialization():
    # Load the native initialization function without importing a different
    # experiment's runtime controller through its generic 'study' module.
    import ast
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import LabelBinarizer

    source = ast.parse(INITIALIZATION.read_text())
    function, = (node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "initialization")
    env = dict(np=np, train_test_split=train_test_split, LabelBinarizer=LabelBinarizer, MLPClassifier=MLPClassifier)
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(INITIALIZATION),"exec"),env)
    return env["initialization"]


def native75(audit):
    for file in ("common.py","classification.py","publication_metadata/chemistry_cases_v1/AUDIT.json"):
        p = CODE75 / file
        expected = subprocess.check_output(["git","-C",str(REQ75),"show",f"{HEAD75}:{p.relative_to(REQ75)}"])
        assert p.read_bytes() == expected
        audit.use(p)
    return read(ROOT75/"AUDIT.json")


def check_partition(code, groups):
    assert np.isin(code,[0,1,2]).all()
    disjoint = [set(groups[code==c]) for c in (0,1,2)]
    assert all(not (a & b) for i,a in enumerate(disjoint) for b in disjoint[i+1:])


def file_seal(files, root):
    return {str(f.relative_to(root)):sha(f) for f in files}


def prepare():
    folder = ROOT/"A"
    folder.mkdir(parents=True,exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    ready = folder/"PREPARED.json"
    if ready.exists():
        return read(ready)
    a=InputAudit()
    old=native75(a)
    checked_enzymemap(a.pins)
    a.use(NATIVE)
    a.use(INITIALIZATION)
    native_plan=load_plan(a)["panels"]
    source, exact, _, encoding=source_data(a,"ecreact")
    labels=[]
    selectors=[]
    all_positions=None
    for endpoint in LEVELS:
        p=native_plan[endpoint]
        data,_=panel_data(a,endpoint,p,source)
        positions=data.source_position.to_numpy()
        if all_positions is not None:
            np.testing.assert_array_equal(positions,all_positions)
        all_positions=positions
        labels.append(data.y.to_numpy())
        selectors.append([np.load(a.use(p["split_paths"][str(seed)],p["split_sha256"][str(seed)]),allow_pickle=False)
                          for seed in range(25)])
    y=np.asarray(labels).T
    np.testing.assert_array_equal(positions,np.arange(len(source)))
    native_main=read(Path("/var/tmp/synrfp-labhhc4-storage/runs/classification_r2_2048_v1/plan.json"))
    # The original cache holds object strings. It is a trusted local dependency.
    assert GROUPS.samefile(GROUPS.resolve())
    a.use(GROUPS)
    groups=np.load(GROUPS,allow_pickle=True).astype(str)[positions]
    for group in selectors[-1]:
        check_partition(group,groups)
    matrix=np.array([s[-1] for s in selectors])
    assert set(y.shape)=={185734,3}
    with a.use(MAIN,old["inputs_sha256"][str(MAIN)]).open(newline="") as f:
        grid={(r["dataset"],int(float(r["seed"])),r["method"]):r for r in csv.DictReader(f)
              if r["dataset"] in LEVELS and r["model"]=="mlp" and r["method"] in MODE}
    thin = ["delta", "delta_union"]
    for mode in thin:
        packed=feature(a,"ecreact",mode,encoding,source.row_id.to_numpy(str))[positions]
        np.save(folder/f"{mode}.npy", packed,allow_pickle=False)
    np.savez_compressed(folder/"cohort.npz",y=y,groups=groups,encoded=encoding,keys=exact[:,0],
                        common_splits=np.asarray(selectors[-1]))
    tasks=[]
    reused=[]
    fit_init=initialization()
    for level, endpoint in enumerate(LEVELS):
        panel=native_plan[endpoint]
        frame,_=panel_data(a,endpoint,panel,source)
        for seed, code in enumerate(selectors[-1]):
            path=folder/"jobs"/f"L{level+1}-s{seed:02d}"
            path.mkdir(parents=True,exist_ok=True)
            # Reuse the archived fit only when both the population and model seed match.
            if np.array_equal(code,selectors[level][seed]):
                test,d,du,fit=comparison_inputs(a,endpoint,seed,panel,frame,grid)
                np.savez_compressed(path/"archived.npz",test=test,d=d,du=du,fit=fit)
                reused.append(path.name)
                continue
            outer=np.r_[np.flatnonzero(code==0),np.flatnonzero(code==1)]
            test=np.flatnonzero(code==2)
            classes,state=fit_init(y[outer,level],PARAMS,seed)
            np.savez_compressed(path/"state.npz",**state)
            np.savez(path/"selectors.npz",outer=outer,test=test,classes=classes)
            for mode in thin:
                task=path/mode
                task.mkdir(exist_ok=True)
                job=dict(id=f"{path.name}-{mode}",params=PARAMS,seed=seed,arm=mode,level=level+1,
                         native_worker_sha256=sha(NATIVE),state_sha256=sha(path/"state.npz"),
                         selectors_sha256=sha(path/"selectors.npz"), packed_sha256=sha(folder/f"{mode}.npy"))
                write(task/"job.json",job)
                tasks.append(str(task.relative_to(folder)))
    assert len(tasks)==100 and len(reused)==25
    record=dict(status="prepared",request_revision="fc4e13cf",n_rows=len(y),n_feature_encoded=int(encoding.sum()),
                n_refusal=int((~encoding).sum()), native_params=PARAMS,matched_partitions=25,archived_fits=50,
                new_fits=100,task_paths=tasks,reused_paths=reused,inputs_sha256=a.pins,
                files_sha256=file_seal([p for p in folder.rglob("*") if p.is_file()],folder),
                full_row_keys_untouched=True,common_reference_splits="original ECReact L3 5x5")
    write(ready,record)
    print("Prepared",record["n_rows"],"rows,",len(tasks),"new fits,",record["archived_fits"],"saved fits",flush=True)
    return record


def classify_rule(bits, keys, groups, y, left, right, fit, scope):
    _,_,_,_,used,values,ties=classification_diagnostic(bits[fit],keys[fit].tolist(),groups[fit],y[fit][:,[left,right]])
    n=int(used.sum())
    ng=len(np.unique(groups[fit][used]))
    nk=len(np.unique(keys[fit][used]))
    coverage=n/len(fit)
    both=coverage>=.05 and ng>=30 and nk>=10
    winner=None
    if both and values is not None and values[0]!=values[1]:
        winner=(left,right)[int(np.argmax(values))]+1
    return dict(scope=scope,train_rows=len(fit),eligible=n,coverage=coverage,query_groups=ng,exact_d_classes=nk,
                n_boundary_ties=ties,passes=both,predicted_higher_gain_level=winner,
                diagnostic_first=values[0] if values else None,diagnostic_second=values[1] if values else None)


def coverage_task(job):
    seed, first, second=job
    x=ARGS
    code=x["codes"][seed]
    rows=np.r_[np.flatnonzero(code==0),np.flatnonzero(code==1)]
    path=ROOT/"A/jobs"/f"L{first+1}-s{seed:02d}"
    if (path/"archived.npz").exists():
        with np.load(path/"archived.npz",allow_pickle=False) as a:
            train=a["fit"]
    else:
        with np.load(path/"state.npz",allow_pickle=False) as a:
            train=rows[a["train"]]
    train=train[x["ok"][train]]
    return dict(seed=seed,comparison=f"L{first+1}_vs_L{second+1}",
        **classify_rule(x["bits"],x["keys"],x["groups"],x["y"],first,second,train,"common_matched_archive_A"))


def coverage_jobs(workers):
    a=ROOT/"A"
    prepare()
    with np.load(a/"cohort.npz",allow_pickle=False) as x:
        ARGS.update(y=x["y"],keys=x["keys"],groups=x["groups"],codes=x["common_splits"],ok=x["encoded"])
    ARGS["bits"]=np.ascontiguousarray(np.load(a/"delta_union.npy",mmap_mode="r")[:,128:])
    jobs=[(s,f,2) for s in range(25) for f in (0,1)] + [(s,0,1) for s in range(25)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results=list(pool.map(coverage_task,jobs))
    pd.DataFrame(results).to_csv(OUT/"ec_level_rule_coverage.csv",index=False)
    print("Training-only EC-level comparisons complete",flush=True)


def dispatch(workers, pilot, tail):
    a=ROOT/"A"
    prep=prepare()
    for name,h in prep["files_sha256"].items():
        assert sha(a/name)==h
    tasks=[a/p for p in prep["task_paths"]]
    if pilot:
        tasks=tasks[:1]
    elif tail:
        tasks=tasks[-tail:][::-1]
    args=[(task,) for task in tasks if not (task/"COMPLETE.json").exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fit_task,args))


def fit_task(job):
    task,=job
    log=task/f"fit-{__import__('time').time_ns()}.log"
    with log.open("w") as stream:
        subprocess.run([str(TORCH_PYTHON),"-u",str(HERE/"level_fit.py"),str(task)],
                       stdout=stream,stderr=subprocess.STDOUT,check=True)
    out=read(task/"COMPLETE.json")
    print(task.parent.name,task.name,out["epochs"],round(out["fit_seconds"],1),flush=True)


def score_sufficient(y, predictions, group):
    # MCC uses the trace and true/predicted histograms, so group resampling
    # needs these sufficient counts instead of a square confusion tensor.
    index=np.arange(len(group))
    n_classes=int(max(y.max(),predictions.max()))+1
    n_groups=int(group.max())+1
    truth=coo_matrix((np.ones(len(group),dtype=np.float64),(group,y)),shape=(n_groups,n_classes)).tocsr()
    out=[]
    for arm in range(2):
        p=predictions[arm].reshape(-1)
        repeat=len(p)//len(y)
        g=np.tile(group,repeat)
        yy=np.tile(y,repeat)
        rows=np.ones(len(p),dtype=np.float64)/repeat
        pred=coo_matrix((rows,(g,p)),shape=(n_groups,n_classes)).tocsr()
        correct=np.bincount(g,weights=(p==yy).astype(float)/repeat,minlength=n_groups)
        out.append((pred,correct))
    size=np.bincount(group,minlength=n_groups).astype(float)
    return truth,out,size


def statistics(models,weights):
    rows=[]
    for truth,arms,size in models:
        t=truth.T@weights
        s=size@weights
        per=[]
        for pred,correct in arms:
            p=pred.T@weights
            c=correct@weights
            cov=c*s-(t*p).sum(axis=0)
            denom=np.sqrt(np.maximum(0,(s**2-(t*t).sum(axis=0))*(s**2-(p*p).sum(axis=0))))
            mcc=np.divide(cov,denom,out=np.zeros_like(cov),where=denom>0)
            err=1-c/s
            per.append((mcc,err))
        (d_mcc,d_err),(du_mcc,du_err)=per
        rows.append((d_mcc,du_mcc,du_mcc-d_mcc,d_err,du_err,d_err-du_err))
    return np.asarray(rows)


def score():
    a=ROOT/"A"
    prep=prepare()
    with np.load(a/"cohort.npz",allow_pickle=False) as s:
        y=s["y"];codes=s["common_splits"];group_text=s["groups"]
    group=np.unique(group_text,return_inverse=True)[1]
    models=[]
    for level,endpoint in enumerate(LEVELS):
        pred=np.full((2,5,len(y)),-1,dtype=np.int32)
        for seed,code in enumerate(codes):
            job=a/"jobs"/f"L{level+1}-s{seed:02d}"
            key=job/"archived.npz"
            test=np.flatnonzero(code==2)
            if key.exists():
                with np.load(key,allow_pickle=False) as archived:
                    np.testing.assert_array_equal(archived["test"],test)
                    values=(archived["d"],archived["du"])
            else:
                values=[]
                for mode in ("delta","delta_union"):
                    attempt=job/mode
                    receipt=read(attempt/"COMPLETE.json")
                    assert receipt["plan_job_sha256"]==sha(attempt/"job.json")
                    for name,h in receipt["files_sha256"].items():
                        assert sha(attempt/name)==h
                    values.append(np.load(attempt/"decoded.npy",allow_pickle=False))
            for arm,v in enumerate(values):
                assert np.all(pred[arm,seed//5,test]==-1) and v.shape==test.shape
                pred[arm,seed//5,test]=v
        assert np.all(pred>=0)
        models.append(score_sufficient(y[:,level],pred,group))
    g=int(group.max())+1
    point=statistics(models,np.ones((g,1)))[:,:,0]
    rng=np.random.default_rng(2026)
    draws=np.empty((3,6,10000))
    for i in range(0,draws.shape[2],40):
        n=min(40,draws.shape[2]-i)
        W=np.empty((g,n))
        for b in range(n):
            W[:,b]=np.bincount(rng.integers(g,size=g),minlength=g)
        draws[:,:,i:i+n]=statistics(models,W)
    rows=[]
    names=("D_mcc","DU_mcc","gain_mcc","D_error","DU_error","gain_error")
    for k,target in enumerate(("L1","L2","L3","L3_minus_L1","L2_minus_L1")):
        if k<3:
            estimates=point[k]
            estimates_draws=draws[k]
            keep=range(6)
        else:
            level=2 if k==3 else 1
            estimates=point[level]-point[0]
            estimates_draws=draws[level]-draws[0]
            keep=(2,5)
        for j in keep:
            lo,hi=np.quantile(estimates_draws[j],[.025,.975])
            rows.append(dict(target=target,metric=names[j],estimate=estimates[j],ci_low=lo,ci_high=hi,
                             n_rows=len(y),n_scaffold_groups=g,folds=25,matched=True))
    pd.DataFrame(rows).to_csv(OUT/"ec_level_pairs.csv",index=False)
    write(ROOT/"A_COMPLETE.json",dict(n_rows=len(y),n_groups=g,archived_fits=prep["archived_fits"],
        new_fits=prep["new_fits"],bootstrap_draws=10000,bootstrap_seed=2026,
        inputs_sha256=prep["inputs_sha256"],files_sha256={name:sha(OUT/name)
            for name in ("ec_level_rule_coverage.csv","ec_level_pairs.csv")}))
    print("EC level pairs and bootstrap complete",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("prepare","coverage","fit","score"))
    p.add_argument("--jobs",type=int,default=6)
    p.add_argument("--pilot",action="store_true")
    p.add_argument("--tail",type=int,default=0)
    a=p.parse_args()
    if a.stage=="prepare":
        prepare()
    elif a.stage=="coverage":
        coverage_jobs(a.jobs)
    elif a.stage=="fit":
        dispatch(a.jobs,a.pilot,a.tail)
    else:
        score()


if __name__=="__main__":
    main()
