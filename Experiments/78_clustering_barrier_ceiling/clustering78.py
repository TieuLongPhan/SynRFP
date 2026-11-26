"""Full-row pooled-channel weighting under known K and the paper's factorial K."""
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import importlib.metadata
import math
import multiprocessing
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import adjusted_mutual_info_score
from threadpoolctl import threadpool_limits

from common78 import BASE, D, DU, LOCAL, REPO, Pins, digest, export, paper_table, read, write

WEIGHTS = (0.0, .25, .5, .75, 1.0)
SEEDS = tuple(range(5))
REGIMES = ("known_k", "factorial_k")
FACTORIAL_CELLS = REPO / "Experiments/56_factorial_refresh/publication_metadata/corrected_full_cohort_v1/cells.csv"


def features(pins, plan, plan52, source):
    src = plan["sources"][source]
    table = pd.read_csv(pins.use(src["table"], plan["pinned_files"][src["table"]]),
                        usecols=["row_id"], dtype={"row_id": str})
    ids = table.row_id.to_numpy(str)
    outputs = {}
    for config in (D, DU):
        native = Path(src["features"][config])
        if source == "tpl_b":
            path = Path(plan52["features"][f"{source}__{config}"]["path"])
            receipt = read(pins.use(path.with_suffix(".json")))
            key = f"{source}__{config}"
            assert receipt["feature_id"] == key
            assert receipt["plan_sha256"] == pins.hashes[str(BASE/"coverage_refits_v1/plan.json")]
            assert receipt["original_sha256"] == plan52["features"][key]["original_sha256"]
            pins.use(native, receipt["original_sha256"])
            expected = receipt["feature_sha256"]
        else:
            path = native
            receipt = read(pins.use(src["feature_receipt"]))
            assert receipt["plan_sha256"] == pins.hashes[str(BASE/"all_endpoint_factorial_full_v1/plan.json")]
            expected = receipt["files"][path.name]
        with np.load(pins.use(path, expected), allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["source_positions"], np.arange(len(ids)))
            np.testing.assert_array_equal(archive["row_ids"], ids)
            assert archive["X"].shape==(len(ids),256) and archive["X"].dtype==np.uint8
            packed, ok = archive["X"], archive["ok"]
            assert not packed[~ok].any()
            outputs[config] = (packed, ok, expected)
    np.testing.assert_array_equal(outputs[D][1], outputs[DU][1])
    bits = outputs[D][0]
    np.testing.assert_array_equal(bits[:, :128]^bits[:,128:], outputs[DU][0][:,:128])
    return ids, outputs


def config_inputs(pins):
    root = BASE / "all_endpoint_factorial_full_v1"
    plan = pins.plan(root)
    repair = pins.plan(BASE/"coverage_refits_v1")
    main = BASE / "default_optimal_benchmark_v1"
    main_plan = pins.plan(main)
    known_path = main/"native/clustering/plan.json"
    known = read(pins.use(known_path, main_plan["inputs_sha256"][str(known_path)]))
    params = plan["clustering_params"]
    assert plan["fingerprint_seed"] == 1 and plan["row_cap"] is None
    sets = plan["clustering_datasets"]
    assert len(sets)==12 and set(sets)==set(known["panels"])
    for endpoint in sets:
        seeds = {t["seed"] for t in plan["tasks"]["clustering"] if t["dataset"]==endpoint and t["config"]==D}
        assert seeds==set(SEEDS)
    closure = read(pins.use(FACTORIAL_CELLS.with_name("COMPLETE.json")))
    scores = pd.read_csv(pins.use(FACTORIAL_CELLS, closure["exported_outputs_sha256"]["cells.csv"]))
    weights = scores.loc[(scores.task=="clustering") & scores.config.isin([D,DU])]
    assert len(weights)==120 and weights.regime.eq("heuristic_rowcount_kmeans").all()
    anchors = {(r.endpoint,r.config,int(r.seed)):float(r.score) for r in weights.itertuples()}
    paper = paper_table(pins)
    screen = paper.query("task == 'clustering'").groupby("dataset").benefit_difference.agg(["size","mean"])
    assert screen.shape==(12,2) and (screen["size"]==18).all()
    return plan, repair, known, params, anchors, screen["mean"].to_dict()


def endpoints(pins, plan, known, source, ids):
    names = [e for e in plan["clustering_datasets"] if plan["panels"][e]["source"]==source]
    results = {}
    for name in names:
        spec = plan["panels"][name]
        path = spec["table"]
        frame = pd.read_csv(pins.use(path, plan["pinned_files"][path]), dtype={"row_id":str})
        rows = frame.source_position.to_numpy()
        np.testing.assert_array_equal(frame.row_id.to_numpy(str),ids[rows])
        assert len(frame)==spec["n"]==known["panels"][name]["n_rows"]
        # Full cohort. Preserve the original factorial's order even if it is not source order.
        assert len(rows)==len(np.unique(rows))
        _, label = np.unique(frame.y, return_inverse=True)
        k = int(label.max())+1
        assert k == known["panels"][name]["k"]
        results[name] = (rows, label, k, min(512,max(2,round(math.sqrt(len(rows)/2)))))
    return results


def anchor(pins, endpoint, config, seed, n, k, params, feature_sha, canonical):
    if endpoint=="tpl_b":
        base = BASE/"coverage_refits_v1/results"
        path = base/f"factorial__clustering22__{endpoint}__{config}__minibatch_kmeans__{seed}/result.json"
        expected_plan = pins.hashes[str(BASE/"coverage_refits_v1/plan.json")]
    else:
        base = BASE/"all_endpoint_factorial_full_v1/clustering"
        path = base/endpoint/config/f"{seed}.json"
        expected_plan = pins.hashes[str(BASE/"all_endpoint_factorial_full_v1/plan.json")]
    r = read(pins.use(path))
    assert (r["phase"], r["dataset"], r["config"], r["seed"], r["n"])==("complete",endpoint,config,seed,n)
    assert r["requested_clusters"]==k and r["params"]==params and r["feature_sha256"]==feature_sha
    assert r["plan_sha256"]==expected_plan
    assert math.isclose(r["ami"], canonical[endpoint,config,seed], rel_tol=0, abs_tol=1e-12)
    return r["ami"]


def sampled_pairs(y, cap, seed):
    """Uniform unordered row pairs without replacement, conditional on class relation."""
    rng = np.random.default_rng(seed)
    sizes = np.bincount(y).astype(np.int64)
    groups = np.split(np.argsort(y), np.cumsum(sizes)[:-1])
    mass = sizes*(sizes-1)//2
    total_same = int(mass.sum())
    total_other = len(y)*(len(y)-1)//2-total_same
    all_sizes = {"within":total_same, "between":total_other}
    for kind,total in all_sizes.items():
        if total==0:
            raise ValueError(f"No {kind}-class pairs in endpoint")
        target = min(cap,total)
        chosen = {}
        while len(chosen)<target:
            if kind=="within":
                a,b = rng.choice(groups[rng.choice(len(groups),p=mass/total_same)],size=2,replace=False)
            else:
                a,b = rng.integers(len(y),size=2)
                if y[a]==y[b]:
                    continue
            chosen[min(a,b),max(a,b)] = None
        yield kind,np.asarray(list(chosen))


def distances(endpoint, packed, label, ok, seed=7801):
    popcount = np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1).sum(axis=1)
    stats = {}
    for kind,pairs in sampled_pairs(label,2000,seed):
        a,b = pairs.T
        diff = popcount[packed[a]^packed[b]]
        stats[kind] = (len(a),diff[:,:128].sum(axis=1),diff[:,128:].sum(axis=1),int((~ok[a]|~ok[b]).sum()))
    results = []
    for index, block in enumerate(("D", "U"),start=1):
        within=float(stats["within"][index].mean())
        between=float(stats["between"][index].mean())
        results.append({
            "endpoint":endpoint,"block":block,"n_rows":len(label),"pair_seed":seed,
            "n_within_pairs":stats["within"][0],"n_between_pairs":stats["between"][0],
            "n_within_pairs_with_refusal":stats["within"][3],"n_between_pairs_with_refusal":stats["between"][3],
            "within_mean_squared_euclidean":within,"between_mean_squared_euclidean":between,
            "between_over_within":between/within if within else None,
        })
    return results


def fit(job):
    matrix, label, params, k, seed, threads = job
    started = time.monotonic()
    shape = read(matrix/"shape.json")
    arrays = [np.load(matrix/f"{key}.npy",mmap_mode="c",allow_pickle=False) for key in ("data","indices","indptr")]
    X = csr_matrix(tuple(arrays),shape=tuple(shape),copy=False)
    with threadpool_limits(limits=threads,user_api="openmp"), threadpool_limits(limits=1,user_api="blas"):
        model = MiniBatchKMeans(n_clusters=k, random_state=seed, **params)
        assignment = model.fit_predict(X)
    score = adjusted_mutual_info_score(label, assignment)
    return {"ami":float(score),"n_iter":model.n_iter_,"n_steps":model.n_steps_,
            "elapsed_seconds":time.monotonic()-started}


def matrix_file(path, packed, weight):
    path.mkdir()
    count_table = np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1).sum(axis=1)
    active = packed[:, :128] if weight==0 else packed
    ptr = np.empty(len(active)+1,dtype=np.int32)
    ptr[0]=0
    for start in range(0,len(active),8192):
        stop=min(start+8192,len(active))
        count=count_table[active[start:stop]].sum(axis=1)
        ptr[start+1:stop+1]=np.cumsum(count,dtype=np.int32)+ptr[start]
    np.save(path/"indptr.npy",ptr)
    size=int(ptr[-1])
    indices=np.lib.format.open_memmap(path/"indices.npy",mode="w+",dtype=np.int32,shape=(size,))
    values=np.lib.format.open_memmap(path/"data.npy",mode="w+",dtype=np.float32,shape=(size,))
    for start in range(0,len(active),8192):
        stop=min(start+8192,len(active))
        bits=np.unpackbits(active[start:stop],axis=1)
        cols=np.nonzero(bits)[1]
        idx=slice(ptr[start],ptr[stop])
        indices[idx]=cols
        values[idx]=1.0 if weight is None else np.where(cols<1024,1.0,weight)
    indices.flush()
    values.flush()
    write(path/"shape.json",[len(packed),2048])


def run_source(source, pins, plan, repair, known, params, canonical, published, pool, args):
    ids, arrays = features(pins, plan, repair, source)
    panels = endpoints(pins,plan,known,source,ids)
    for endpoint,(rows,label,k,heuristic) in panels.items():
        dist = distances(endpoint, arrays[DU][0][rows],label,arrays[DU][1][rows])
        write(args.output/f"distance_{endpoint}.json",dist)
        print(f"{endpoint} distances verified | n={len(rows):,}, K={k}/{heuristic}",flush=True)
    # Use the same matrix for every endpoint only where original row order agrees.
    # Otherwise build the panel once and share read-only pages across all fits.
    order_groups = []
    for endpoint,(rows,*_) in panels.items():
        for known_order, members in order_groups:
            if np.array_equal(rows,known_order):
                members.append(endpoint)
                break
        else:
            order_groups.append((rows,[endpoint]))

    for rows,names in order_groups:
        for weight in (None,1.0,0.0,.25,.5,.75):
            config = D if weight is None else DU
            tag = "d2048" if weight is None else f"u{weight:g}"
            packed,ok,sha = arrays[config]
            pending = []
            for endpoint in names:
                _,label,known_k,heuristic = panels[endpoint]
                for regime, k in zip(REGIMES,(known_k,heuristic),strict=True):
                    for seed in SEEDS:
                        line = {"endpoint":endpoint,"protocol":regime,"k":k,"seed":seed,
                                "setting":tag,"u_weight":weight,"n_rows":len(rows),"n_refusal_placeholders":int((~ok[rows]).sum()),
                                "paper_mean_context_ami_change_over_18":published[endpoint]}
                        destination = args.output/"fits"/f"{endpoint}__{regime}__{tag}__{seed}.json"
                        if destination.exists():
                            record=read(destination)
                            for key,value in line.items():
                                assert record[key]==value
                            assert record["input_feature_sha256"]==sha and record["runner_sha256"]==digest(__file__)
                            continue
                        line.update(input_feature_sha256=sha,runner_sha256=digest(__file__))
                        if regime=="factorial_k" and weight in (None,1.0):
                            line["published_factorial_ami"] = anchor(pins,endpoint,config,seed,len(rows),k,params,sha,canonical)
                        pending.append((destination,line,label,k,seed))
            if pending:
                with TemporaryDirectory(dir=args.scratch,prefix="scan78-") as scratch:
                    matrix = Path(scratch)/"csr"
                    matrix_file(matrix,packed[rows],weight)
                    pending_futures = {
                        pool.submit(fit,(matrix,label,params,k,seed,args.threads)):(path,line)
                        for path,line,label,k,seed in pending
                    }
                    for future in as_completed(pending_futures):
                        path,line = pending_futures[future]
                        line.update(future.result(),origin="new_weight_scan",input_layout="float32_csr")
                        if "published_factorial_ami" in line:
                            line["ami_minus_published_factorial"] = line["ami"]-line["published_factorial_ami"]
                        write(path,line)
                        print(f"{path.stem} AMI={line['ami']:.5f} {line['elapsed_seconds']:.1f}s",flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=LOCAL/"clustering_csr")
    parser.add_argument("--scratch",type=Path,default=LOCAL/"scratch")
    parser.add_argument("--jobs",type=int,default=6)
    parser.add_argument("--threads",type=int,default=2)
    parser.add_argument("--source-jobs",type=int,default=9)
    args=parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
    for folder in (args.output/"fits",args.scratch):
        folder.mkdir(parents=True,exist_ok=True)
    pins=Pins()
    plan,repair,known,params,canonical,published=config_inputs(pins)
    order=sorted({plan["panels"][key]["source"] for key in plan["clustering_datasets"]},
                 key=lambda s: (-plan["sources"][s]["n"],s))
    started=time.monotonic()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.jobs, mp_context=context) as pool:
        with ThreadPoolExecutor(max_workers=args.source_jobs) as producers:
            work = [producers.submit(run_source,source,pins,plan,repair,known,params,canonical,published,pool,args)
                    for source in order]
            for future in as_completed(work):
                future.result()
    cells=[read(path) for path in sorted((args.output/"fits").glob("*.json"))]
    assert len(cells)==720
    assert sum(r["origin"]=="new_weight_scan" for r in cells)==720
    assert sum("published_factorial_ami" in r for r in cells)==120
    d=[row for path in sorted(args.output.glob("distance_*.json")) for row in read(path)]
    assert len(d)==24
    export(args.output/"clustering_distance_ratio.csv",d)
    export(args.output/"clustering_weight_cells.csv",cells)
    frame=pd.DataFrame(cells)
    summaries=[]
    for _,v in frame.groupby(["endpoint","protocol","setting"],sort=True):
        assert set(v.seed)==set(SEEDS) and len(v)==5
        first=v.iloc[0]
        phi=["endpoint","protocol","k","setting","u_weight","n_rows","n_refusal_placeholders","origin","paper_mean_context_ami_change_over_18"]
        info=first[phi].to_dict()
        info["u_weight"]=None if pd.isna(info["u_weight"]) else info["u_weight"]
        info.update(n_seeds=len(v),ami_mean=float(v.ami.mean()),ami_sd=float(v.ami.std(ddof=1)))
        if v.published_factorial_ami.notna().all():
            info.update(published_factorial_ami_mean=float(v.published_factorial_ami.mean()),
                        published_factorial_ami_sd=float(v.published_factorial_ami.std(ddof=1)))
        summaries.append(info)
    export(args.output/"clustering_weight_scan.csv",summaries)
    write(args.output/"clustering_AUDIT.json",{
        "request_revision":"773ba9ac4610fe573576a6fa91c9383caace8abd", "input_sha256":pins.hashes,
        "runner_sha256":digest(__file__),"fit_cells":len(cells),"new_fits":720,"archived_factorial_scores":120,
        "initialization_seeds":list(SEEDS),"cpu_workers":args.jobs,"omp_threads_per_worker":args.threads,
        "source_workers":args.source_jobs,
        "parameters":params,"elapsed_seconds":time.monotonic()-started,
        "packages":{p:importlib.metadata.version(p) for p in ("scikit-learn","scipy","numpy","pandas","threadpoolctl")},
        "outputs_sha256":{p.name:digest(p) for p in args.output.glob("clustering_*.csv")},
    })


if __name__=="__main__":
    main()
