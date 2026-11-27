"""Bind released yield selectors, model inputs, and authors' success labels."""
from collections import Counter
from functools import lru_cache
from hashlib import file_digest
from importlib.metadata import version
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ORIGINAL = REPO.parent/"SynRFP"
BASE = Path.home()/".local/share/synrfp-runs"
ROOT = BASE/"request93_yield_learner_v1"
STORAGE = Path("/var/tmp/synrfp-labhhc4-storage/bh_hte_2026")
NATIVE = STORAGE/"runs/continuous_allfps_v1"
RETRAIN = STORAGE/"runs/retrain_r3_union_v1"
ROLES = STORAGE/"runs/roles_v1"
FULL = BASE/"yield_fullrow_v1"
RELEASE = REPO/"Experiments/65_default_optimal/publication_metadata/default_optimal_v1"
R92 = REPO.parent/"SynRFP-yield-ood-analysis"
R92_REV = "ca4471b88e3a85387273facfdfc5127de5d9b9ab"
AUTHOR = STORAGE/"data/BH_HTE_Curated_Dataset_v20260219.csv"
AUTHOR_SHA = "247473787c33fb22ee48a4177593188f2a2c6ef140756566fe4610fee89ba333"
BENCHMARK = ("synrfp_default","synrfp_optimal","drfp","crxnfp","rdkit_diff","rdkit_struct","rxnfp_mlm")
CONTEXT = ("wl_r3_delta","wl_r3_delta_union")
CLASS = ("synrfp_default","synrfp_optimal","drfp","rdkit_struct")
PARTS = ("train","validation","test")
PANEL = {"ncs_bh":"ncs_cluster_ood","uspto_yield":"uspto_grouped"}
MLP = dict(hidden=512,learning_rate=.001,weight_decay=.0001,batch_size=256,
           max_epochs=200,patience=15,min_delta=.0001,seed=42)


def sha(path):
    with Path(path).open("rb") as f:
        return file_digest(f,"sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def verify(path, expected):
    if sha(path)!=expected:
        raise ValueError(f"Changed archived input: {path}")
    return Path(path)


def write(path, payload):
    path = Path(path)
    temp = path.with_suffix(f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n")
    temp.replace(path)


@lru_cache
def table(path, expected):
    return pd.read_csv(verify(path,expected),dtype={"row_id":str})


def anchor(ref):
    if ref.source_result_path:
        path = verify(ref.source_result_path,ref.source_result_sha256)
    else:
        assert ref.method=="synrfp_default" and ref.id and ref.dataset in ("bh_classic","suzuki_hte")
        path = BASE/"final_comparison_v1/regression/models"/ref.id/"result.json"
    r = read(path)
    for metric in ("r2","mae","rmse"):
        np.testing.assert_allclose(r[metric],getattr(ref,metric),atol=1e-10,rtol=0)
    fit = {}
    if "artifacts_sha256" in r:
        for key in ("fit.npz","test.npy","rows.npz","preprocessing.npz"):
            candidates = [(p,h) for p,h in r["artifacts_sha256"].items() if Path(p).name==key]
            assert len(candidates)==1
            fit[key] = dict(path=candidates[0][0],sha256=candidates[0][1])
    else:
        directory=Path(r.get("attempt_directory",path.parent))
        for key in ("fit.npz","test.npy","rows.npz","preprocessing.npz"):
            fit[key]=dict(path=str(directory/key),sha256=r["artifacts"][key])
    for recorded in fit.values():
        assert Path(recorded["path"]).is_file() and len(recorded["sha256"])==64
    return dict(path=str(path),sha256=sha(path),fit=fit,
                metrics={k:float(r[k]) for k in ("r2","mae","rmse")})


def selectors(bound, cohort):
    f=bound["fit"]["rows.npz"]
    with np.load(verify(f["path"],f["sha256"]),allow_pickle=False) as z:
        result={k:z[k] for k in PARTS}
        if "test_row_ids" in z.files:
            np.testing.assert_array_equal(z["test_row_ids"].astype(str),cohort.row_id.to_numpy()[result["test"]])
    x=np.concatenate(list(result.values()))
    assert len(x)==len(np.unique(x)) and np.min(x)>=0 and np.max(x)<len(cohort)
    return result


def cache(dataset, variant, cohort):
    stem = "synrfp_r3b2048" if variant=="wl_r3_delta" else "synrfp_du_r3b2048"
    if dataset=="uspto_yield":
        return dict(path=str(ROOT/"uspto_context"/f"{variant}.npz"),variant=variant,create_after_bitbirch=True)
    source = (RETRAIN if variant=="wl_r3_delta" else ROLES)/"features"/dataset/f"{stem}.npz"
    path = source.with_suffix(".json")
    evidence = read(path)
    assert evidence["complete"] and evidence["variant"]==stem
    with np.load(verify(source,evidence["feature_sha256"]),allow_pickle=False) as z:
        assert int(z["n_bits"])==2048
        np.testing.assert_array_equal(z["row_ids"].astype(str),cohort.row_id.to_numpy())
        np.testing.assert_array_equal(z["source_positions"],np.arange(len(cohort)))
        assert z["ok"].all() and z["X"].shape==(len(cohort),256)
    return dict(path=str(source),sha256=evidence["feature_sha256"],metadata_sha256=sha(path))


def main():
    ROOT.mkdir(exist_ok=True)
    manifest = RELEASE/"COMPLETE.json"
    release_sha = read(manifest)["exported_outputs_sha256"]["yield/cells.csv"]
    records=pd.read_csv(verify(RELEASE/"yield/cells.csv",release_sha),keep_default_na=False,low_memory=False)
    records=records[records.method.isin(BENCHMARK)]
    full=read(verify(FULL/"plan.json",read(FULL/"plan_seal.json")["sha256"]))
    old_sha=read(NATIVE/"plan_seal.json")["plan_sha256"]
    old=read(verify(NATIVE/"plan.json",old_sha))
    sources={}
    groups=[]
    for dataset,g in records.groupby("dataset"):
        owner=full if dataset in PANEL else old
        t=owner["tables"][dataset]["path"]
        digest=owner["pinned_files"][t]
        df=table(t,digest)
        assert df.row_id.is_unique and len(df)==owner["tables"][dataset]["rows"]
        sources[dataset]=dict(table=t,sha256=digest,n=len(df),
           context={c:cache(dataset,c,df) for c in CONTEXT})
        for split,s in g.groupby("split"):
            assert len(s)==7 and set(s.method)==set(BENCHMARK)
            inputs={r.method:anchor(r) for r in s.itertuples()}
            comparator=selectors(inputs[BENCHMARK[0]],df)
            for method, bound in inputs.items():
                current=selectors(bound,df)
                for k in PARTS:
                    np.testing.assert_array_equal(comparator[k],current[k])
            if dataset in PANEL:
                entry=next(x for x in full["splits"] if x["dataset"]==dataset and x["split"]==split)
                with np.load(verify(entry["path"],full["pinned_files"][entry["path"]]),allow_pickle=False) as z:
                    for k in PARTS:
                        np.testing.assert_array_equal(comparator[k],z[k])
            panel=PANEL[dataset] if dataset in PANEL else s.report_group.unique().item()
            groups.append(dict(dataset=dataset,panel=panel,split=split,inputs=inputs,
                               ntrain=len(comparator["train"]),nval=len(comparator["validation"]),ntest=len(comparator["test"])))
    assert Counter(g["panel"] for g in groups)==dict(
        bh_random=10,bh_published_ood=4,suzuki_random=10,ncs_cluster_ood=40,uspto_grouped=5)
    raw=pd.read_csv(verify(AUTHOR,AUTHOR_SHA),low_memory=False)
    ncs=table(sources["ncs_bh"]["table"],sources["ncs_bh"]["sha256"])
    np.testing.assert_array_equal(ncs.row_id.to_numpy(int),np.arange(len(raw)))
    np.testing.assert_array_equal(ncs.yield_pct,raw.Yield)
    assert raw.labels.isin([0,1]).all()
    pack=dict(request_commit="821bc2e94341a174ecb4e1287a45c1382c769972", release_table_sha256=release_sha,
              fullrow_plan_sha256=sha(FULL/"plan.json"), older_plan_sha256=old_sha,
              r92_revision=R92_REV, author_sha256=AUTHOR_SHA,
              author_disagreements_with_recomputed_gt10=int((raw.labels!=(raw.Yield>10)).sum()),
              sources=sources,groups=groups, mlp=MLP,
              expected_regressors=len(groups)*9,expected_classifiers=40*4*2,
              versions={p:version(p) for p in ("scikit-learn","pandas","numpy","rdkit","synkit")})
    write(ROOT/"PLAN.json",pack)
    print("Bound",len(groups),"splits,",pack["expected_regressors"],"regressors and",pack["expected_classifiers"],"classifiers.")
    print(raw.labels.value_counts().to_dict(),pack["author_disagreements_with_recomputed_gt10"],"released labels differ from raw >10.")


if __name__=="__main__":
    main()
