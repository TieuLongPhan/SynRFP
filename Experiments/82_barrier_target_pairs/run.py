"""Matched grouped-CV target pairs on B97XD3, GDB7 and Cyclo."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from rdkit import RDLogger

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
WORKSPACE = REPO.parent
ORIGINAL = WORKSPACE / "SynRFP"
PREVIOUS78 = WORKSPACE / "SynRFP-clustering-barrier-ceiling"
PREVIOUS74 = WORKSPACE / "SynRFP-target-dependence"
REP78 = PREVIOUS78 / "Experiments/78_clustering_barrier_ceiling"
REP74 = PREVIOUS74 / "Experiments/74_context_dependence"
sys.path[:0] = [str(REP78), str(REP74), str(REPO)]
import barrier78
from common78 import BASE, D, DU, Pins, digest, read, write
from paired_analysis import means_by_group, regression_diagnostic

ROOT = BASE / "request82_barrier_target_pairs_v1"
OUT = HERE / "publication_metadata/barrier_target_pairs_v1"
ARCHIVE = Path("/var/tmp/synrfp-labhhc4-storage/runs")
WORKER = ORIGINAL / "Experiments/13_yield/yield_retrain_cpu.py"
TORCH_PYTHON = Path.home() / "anaconda3/envs/synprop/bin/python"
SCORING = ORIGINAL / "Experiments/65_default_optimal/publication_metadata/default_optimal_v1/barrier/cells.csv"
NAMES = ("b97xd3", "cyclo", "gdb7")
SOURCES = {"b97xd3": ("ea", "dh"), "cyclo": ("G_act", "G_r"), "gdb7": ("ea", "dh")}
METHODS = {"D": "synrfp_default", "DU": "synrfp_optimal"}
ARCHIVED_REV = "55181ba9058380404144dc70ea5743f53962e3a0"
REV78 = "4ba01b4bbfde4bc01c68a1efe1618c6392ac79f4"
REV74 = "5c27f7d9"
DRAW_SEED = 2026
DRAWS = 10000
STATE = {}


def pinned_git(pins, repo, revision, path):
    from hashlib import sha256
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{revision}:{path.relative_to(repo)}"])
    pins.use(path, sha256(raw).hexdigest())


def bind(pins, folder, hashes, name):
    return pins.use(folder / name, hashes[name])


def previous(pins, row, wanted, y, ids):
    result_path = pins.use(row.source_result_path, row.source_result_sha256) if row.source_result_path else pins.use(row.result_path)
    result = read(result_path)
    folder = Path(result.get("attempt_directory") or result_path.parent)
    if "artifacts_sha256" in result:
        look = {Path(name).name: h for name, h in result["artifacts_sha256"].items()}
    else:
        look = result["artifacts"]
    data = bind(pins, folder, look, "rows.npz")
    with np.load(data, allow_pickle=False) as archive:
        for part, arr in wanted.items():
            np.testing.assert_array_equal(archive[part], arr)
    pred = np.load(bind(pins, folder, look, "predictions.npy"), allow_pickle=False)
    if "evaluation.npz" in look:
        # The historical result receipt pins these local model artifacts. Older
        # property receipts store their string identifiers in object arrays.
        with np.load(bind(pins, folder, look, "evaluation.npz"), allow_pickle=True) as archive:
            np.testing.assert_array_equal(archive["observed"], y)
            if "row_ids" in archive:
                assert any(np.array_equal(archive["row_ids"].astype(str), candidate) for candidate in ids)
            if "source_positions" in archive:
                np.testing.assert_array_equal(archive["source_positions"], wanted["test"])
            if "prediction" in archive:
                np.testing.assert_array_equal(archive["prediction"], pred)
    assert pred.shape == y.shape and np.isfinite(pred).all()
    assert abs(np.sqrt(np.mean((pred-y)**2)) - float(row.rmse)) < 1e-8
    job = read(bind(pins, folder, look, "job.json"))
    return pred, job["params"]


def direction_groups(left, right):
    parent = {}
    pairs = {}

    def find(value):
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for l, r in zip(left, right, strict=True):
        a = find(l)
        key = tuple(sorted((l, r)))
        if key in pairs:
            b = find(pairs[key])
            if a != b:
                parent[a] = b
        else:
            pairs[key] = l
    return np.array([find(l) for l in left])


def prepare():
    from hashlib import sha256
    ROOT.mkdir(exist_ok=True)
    pins = Pins()
    pinned_git(pins, ORIGINAL, ARCHIVED_REV, SCORING)
    for file in ("barrier78.py", "common78.py"):
        pinned_git(pins, PREVIOUS78, REV78, REP78/file)
    pinned_git(pins, PREVIOUS74, REV74, REP74/"paired_analysis.py")
    pins.use(WORKER)
    barrier78.TARGETS = NAMES
    old = barrier78.prepare(pins)
    grid = pd.read_csv(SCORING, keep_default_na=False, low_memory=False)
    cases = {}
    for n in NAMES:
        src, barrier = old[n]
        frame = pd.read_csv(src["table"], dtype={"row_id": str})
        ids = frame.row_id.to_numpy(str)
        if n == "b97xd3":
            second = frame.dh.to_numpy(float)
            group = direction_groups(frame.left, frame.right)
            split_root = ARCHIVE / "properties_allfps_v1/splits/b97xd3"
        else:
            columns = {"cyclo": ("rxn_id", "G_act", "G_r"), "gdb7": ("idx", "dE0", "dHrxn298")}
            id_col, primary, another = columns[n]
            raw_path = pins.use(ORIGINAL / f"Experiments/14_barrier/data/{n}.csv")
            raw = pd.read_csv(raw_path, dtype={id_col:str})
            np.testing.assert_array_equal(raw["Unnamed: 0"].to_numpy(str), ids)
            np.testing.assert_array_equal(raw.rxn_smiles, frame.reaction)
            np.testing.assert_array_equal(raw[primary].to_numpy(float), barrier)
            second = raw[another].to_numpy(float)
            group = frame.left.to_numpy(str)
            split_root = ARCHIVE / f"barrier_allfps_v1/splits/{n}"
        targets = np.column_stack((barrier,second))
        assert np.isfinite(targets).all()
        pack = barrier78.STATE["frames"][n][1]
        cells = []
        all_test = []
        for fold in range(5):
            split = f"group_cv_f{fold}"
            rows = grid.loc[(grid.dataset==n) & (grid.split==split) & grid.method.isin(METHODS.values())]
            assert len(rows) == 2 and rows.split_sha256.nunique() == 1
            path = pins.use(split_root/f"{split}.npz", rows.split_sha256.iloc[0])
            with np.load(path,allow_pickle=False) as z:
                parts = {k:z[k] for k in ("train","validation","test")}
            np.testing.assert_array_equal(np.sort(np.concatenate(list(parts.values()))),np.arange(len(ids)))
            for a,b in (("train","validation"),("train","test"),("validation","test")):
                assert len(np.intersect1d(group[parts[a]],group[parts[b]]))==0
            t = parts["test"]
            all_test.extend(t)
            predictions = {}
            for arm,method in METHODS.items():
                row, = rows.loc[rows.method==method].itertuples()
                assert row.configuration == (D if arm=="D" else DU)
                label_ids = [ids[t]]
                if "r_id" in frame:
                    label_ids.append(frame.r_id.to_numpy(str)[t])
                pred, params = previous(pins,row,parts,targets[t,0],label_ids)
                assert params == src["params"]
                predictions[arm] = pred
            cells.append((split, parts, predictions))
        np.testing.assert_array_equal(np.sort(all_test), np.arange(len(ids)))
        cases[n] = dict(src=src, targets=targets, group=group, cells=cells, packed=pack)
        print("Verified primary fits", n, len(ids), flush=True)
    STATE.update(cases=cases, pins=pins.hashes)
    identity = dict(inputs_sha256=pins.hashes, code_sha256=digest(Path(__file__)), work="second target only",
                    fit_device="cpu", grouped_folds=5, request_revision="475df11a")
    path = ROOT/"FIT_PLAN.json"
    if path.exists():
        assert read(path) == identity
    else:
        write(path, identity)
    return digest(path)


def fit_one(job):
    name, fold, arm, plan_sha = job
    case = STATE["cases"][name]
    split, parts, _ = case["cells"][fold]
    path = ROOT/"fits"/f"{name}__{split}__{arm}"
    receipt_path = path/"verified.json"
    if receipt_path.exists():
        out = read(receipt_path)
        assert out["plan_sha256"] == plan_sha
        for file, expected in out["files_sha256"].items():
            assert digest(path/file) == expected
        return
    path.mkdir(parents=True, exist_ok=True)
    # Preserve any interrupted attempt by fitting in a new directory.
    attempt = path/f"attempt-{time.time_ns()}"
    attempt.mkdir()
    fit, val, test = (parts[k] for k in ("train","validation","test"))
    y = case["targets"][:,1]
    ym, ys = y[fit].mean(), y[fit].std()
    assert ys > 0
    f = case["packed"][0 if arm=="D" else 1]
    def matrix(index):
        return np.unpackbits(f[index], axis=1).astype(np.float32)
    np.savez(attempt/"fit.npz",X_train=matrix(fit),X_val=matrix(val),
             y_train=((y[fit]-ym)/ys).astype(np.float32),y_val=((y[val]-ym)/ys).astype(np.float32))
    np.save(attempt/"test.npy",matrix(test))
    np.savez(attempt/"preprocessing.npz",target_mean=ym,target_scale=ys)
    j = dict(dataset=name, split=split, fingerprint=arm, params=case["src"]["params"],plan_sha256=plan_sha,
             inputs={k:digest(attempt/k) for k in ("fit.npz","test.npy","preprocessing.npz")})
    write(attempt/"job.json",j)
    with (attempt/"worker.log").open("w") as log:
        subprocess.run([str(TORCH_PYTHON),"-u",str(WORKER),str(attempt)],stdout=log,stderr=log,check=True)
    receipt = read(attempt/"training.json")
    assert receipt["worker_sha256"] == digest(WORKER) and receipt["job_sha256"] == digest(attempt/"job.json")
    prediction = np.load(attempt/"predictions.npy",allow_pickle=False)
    assert prediction.shape==(len(test),) and np.isfinite(prediction).all()
    files = {str(p.relative_to(path)):digest(p) for p in attempt.iterdir() if p.is_file()}
    write(receipt_path,dict(attempt=str(attempt),plan_sha256=plan_sha,files_sha256=files,
                            n_test=len(test),seconds=receipt["seconds"],epochs=receipt["epochs"]))
    print(name, split, arm, receipt["epochs"], round(receipt["seconds"], 1), flush=True)


def keys_and_rule(jobs):
    n_keys={n:np.empty((len(c["targets"]),2),dtype="V32") for n,c in STATE["cases"].items()}
    work=[(n,start,min(start+384,len(k))) for n,k in n_keys.items() for start in range(0,len(k),384)]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for n,start,end,k in pool.map(barrier78.encode,work):
            n_keys[n][start:end]=k
    rule, variance = [], []
    for n, keys in n_keys.items():
        case = STATE["cases"][n]
        arrays = case["targets"]
        d, du = (np.unique(keys[:,i],return_inverse=True)[1] for i in range(2))
        assert len(np.unique(np.c_[d,du],axis=0))==len(np.unique(du))
        for i,target in enumerate(SOURCES[n]):
            y=arrays[:,i]
            left,right=barrier78.dispersion(d,y),barrier78.dispersion(du,y)
            variance.append(dict(dataset=n, target=target, n_rows=len(y), units=case["src"]["units"],
                                 **{f"D_{k}":v for k,v in left.items()},**{f"DU_{k}":v for k,v in right.items()},
                                 variance_reduction_fraction_of_total=(left["within_sse"]-right["within_sse"])/((y-y.mean())**2).sum(),
                                 dof_adjusted_reduction_fraction_of_total=(
                                     (left["pooled_within_sd"]**2-right["pooled_within_sd"]**2)/y.var(ddof=1)
                                     if left["within_dof"] and right["within_dof"] else None)))
        for split, parts, _ in case["cells"]:
            fit=parts["train"]
            ranked=regression_diagnostic(np.unpackbits(case["packed"][1][fit,128:],axis=1),
                                         keys[fit,0].tolist(),case["group"][fit],arrays[fit])
            score=ranked.summary(arrays[fit],case["group"][fit])
            idx=score.pop("higher_gain_target_index")
            values=score.pop("diagnostic") or [None,None]
            rule.append(dict(dataset=n,fold=split,n_train=len(fit),predicted_higher_gain=SOURCES[n][idx] if idx is not None else None,
                             primary_diagnostic=values[0],second_diagnostic=values[1],**score))
    pd.DataFrame(rule).to_csv(OUT/"rule_coverage.csv",index=False)
    pd.DataFrame(variance).to_csv(OUT/"target_pair_variance.csv",index=False)
    print("Training rule and exact variance complete",flush=True)


def score_summary(n, case):
    errors = np.empty((len(case["targets"]),2,2))
    normalization = np.empty((len(case["targets"]),2))
    for split, parts, old in case["cells"]:
        test, train = parts["test"], parts["train"]
        y=case["targets"]
        normalization[test,:]=y[train].var(axis=0)
        for a,arm in enumerate(METHODS):
            rec=read(ROOT/"fits"/f"{n}__{split}__{arm}"/"verified.json")
            pred=np.load(Path(rec["attempt"])/"predictions.npy",allow_pickle=False)
            errors[test,0,a]=(y[test,0]-old[arm])**2
            errors[test,1,a]=(y[test,1]-pred)**2
    assert np.isfinite(errors).all() and np.all(normalization>0)
    group_raw=means_by_group(errors,case["group"])
    group_norm=means_by_group(errors/normalization[:,:,None],case["group"])
    rng=np.random.default_rng(DRAW_SEED)
    ndraws=np.empty((DRAWS,2))
    pdraws=np.empty((DRAWS,2))
    for start in range(0,DRAWS,80):
        limit=min(80,DRAWS-start)
        ix=rng.integers(len(group_raw),size=(limit,len(group_raw)))
        norm=group_norm[ix].mean(axis=1)
        raw=group_raw[ix].mean(axis=1)
        ndraws[start:start+limit]=norm[:,:,0]-norm[:,:,1]
        pdraws[start:start+limit]=100*(1-np.sqrt(raw[:,:,1]/raw[:,:,0]))
    raw=group_raw.mean(axis=0)
    norm=group_norm.mean(axis=0)
    point_n=norm[:,0]-norm[:,1]
    point_p=100*(1-np.sqrt(raw[:,1]/raw[:,0]))
    scores=[]
    for j,target in enumerate((*SOURCES[n],"second_minus_primary")):
        a=ndraws[:,j] if j<2 else ndraws[:,1]-ndraws[:,0]
        b=pdraws[:,j] if j<2 else pdraws[:,1]-pdraws[:,0]
        lo,hi=np.quantile(a,[.025,.975])
        plo,phi=np.quantile(b,[.025,.975])
        scores.append(dict(dataset=n,target=target,n_rows=len(errors),n_groups=len(group_raw),
                           D_rmse=raw[j,0]**.5 if j<2 else None,DU_rmse=raw[j,1]**.5 if j<2 else None,
                           normalized_mse_gain=point_n[j] if j<2 else point_n[1]-point_n[0],
                           gain_low=lo,gain_high=hi,pct_rmse_gain=point_p[j] if j<2 else point_p[1]-point_p[0],
                           pct_low=plo,pct_high=phi))
    return scores


def report():
    scores=[r for n,case in STATE["cases"].items() for r in score_summary(n,case)]
    pd.DataFrame(scores).to_csv(OUT/"target_pair_gains.csv",index=False)
    rule=pd.read_csv(OUT/"rule_coverage.csv")
    var=pd.read_csv(OUT/"target_pair_variance.csv")
    lines=["# Within-dataset barrier and reaction-energy targets","",
           "Each endpoint uses the five archived group folds, the same row membership for both targets and arms, "
           "the published 512-unit MLP recipe, and training-only target standardization. The primary barrier fits "
           "are the original CUDA fits. The second targets use the equivalent native CPU worker with the same hyperparameters "
           "and seed. CPU and CUDA can follow different optimization paths.", "",
           "| Dataset | Training folds passing frozen rule | Primary normalized MSE gain | Second normalized MSE gain | Second minus primary, 95% CI |",
           "|---|---:|---:|---:|---:|"]
    for i,n in enumerate(NAMES):
        a,b,x=scores[i*3:i*3+3]
        r=rule.loc[rule.dataset==n]
        lines.append(f"| {n} | {r.coverage_passes.sum()}/5 | {a['normalized_mse_gain']:+.4f} | "
                     f"{b['normalized_mse_gain']:+.4f} | {x['normalized_mse_gain']:+.4f} "
                     f"[{x['gain_low']:+.4f}, {x['gain_high']:+.4f}] |")
    detectable=[x["dataset"] for x in scores if x["target"]=="second_minus_primary" and x["gain_low"]*x["gain_high"]>0]
    resolved=[(n,*var.loc[var.dataset==n,"variance_reduction_fraction_of_total"].tolist()) for n in NAMES]
    lines.extend(["", "A paired interaction excludes zero for " + (", ".join(detectable) if detectable else "none of the three datasets")
                  + ". QuantumPioneer's held-out interaction was inconclusive. "
                  "The frozen eligibility thresholds were 5% of training rows, 30 query groups and ten exact-D classes. "
                  "These benchmark-cohort comparisons are exploratory. All 10,000 bootstrap draws resample the same held-out groups for both targets and arms.",
                  "", "The within-exact-class variance resolved by U, as a fraction of total variance, was "
                  + ", ".join(f"{n} {a:.4%} and {b:.4%}" for n,a,b in resolved)
                  + " for the primary and second targets, respectively. The variance measure uses all rows. "
                  "The paired predictive gains use held-out rows and therefore answer a different question."])
    (OUT/"target_pairs_note.md").write_text("\n".join(lines)+"\n")
    receipts={str(p):digest(p) for p in (ROOT/"fits").glob("*/verified.json")}
    assert len(receipts)==30
    write(OUT/"COMPLETE.json",dict(status="complete",request_revision="475df11a",new_fits=30,
          archived_barrier_fits=30,bootstrap_draws=DRAWS,bootstrap_seed=DRAW_SEED,
          versions={k:version(k) for k in ("numpy","pandas","scikit-learn","rdkit","synkit")},
          torch_version=subprocess.check_output([str(TORCH_PYTHON), "-c",
             "from importlib.metadata import version; print(version('torch'))"],text=True).strip(),
          fit_plan_sha256=digest(ROOT/"FIT_PLAN.json"),inputs_sha256=STATE["pins"],receipt_sha256=receipts,
          code_sha256=digest(Path(__file__)), files_sha256={p.name:digest(p) for p in OUT.iterdir() if p.name!="COMPLETE.json"}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=("fit","rule","report"))
    parser.add_argument("--jobs",type=int,default=6)
    args=parser.parse_args()
    RDLogger.DisableLog("rdApp.*")
    OUT.mkdir(parents=True,exist_ok=True)
    plan_sha=prepare()
    if args.stage=="fit":
        ordered=sorted(NAMES,key=lambda n:-len(STATE["cases"][n]["targets"]))
        jobs=[(n,f,arm,plan_sha) for n in ordered for f in range(5) for arm in METHODS]
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(fit_one,jobs))
    elif args.stage=="rule":
        keys_and_rule(args.jobs)
    else:
        report()


if __name__=="__main__":
    main()
