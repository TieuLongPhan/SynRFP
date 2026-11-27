"""Publish matched scores, context contrasts and strict held-out condition rankings."""
from importlib.metadata import version
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import kendalltau
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from fit93 import path_for, planned
from plan93 import (AUTHOR, AUTHOR_SHA, BENCHMARK, CLASS, CONTEXT, HERE, RETRAIN, ROLES,
                    ROOT, R92, R92_REV, read, selectors, sha, table, verify, write)

OUTPUT=HERE/"publication_metadata/yield_learner_v1"
R92_FOLDER="Experiments/92_yield_ood_analysis"
BASELINES=("training_recipe_mean","training_global_mean","training_recipe_success")


def r92_inputs():
    reports=R92/R92_FOLDER
    paths=("rank92.py","preflight.py","publication_metadata/preflight_v1/condition_coverage.csv")
    for relative in paths:
        orig=subprocess.check_output(["git","-C",str(R92),"show",f"{R92_REV}:{R92_FOLDER}/{relative}"])
        assert (reports/relative).read_bytes()==orig
    sys.path.insert(0,str(reports))
    from rank92 import rank_conditions, RECIPE
    pairs=pd.read_csv(reports/paths[2]).set_index("split").pairs_absent_from_train_and_validation
    assert len(pairs)==40 and pairs.sum()==609
    return rank_conditions,RECIPE,pairs,{p:sha(reports/p) for p in paths}


def results(plan, plan_hash):
    cells, fingerprints={},{ }
    for task in planned(plan):
        destination=path_for(*task)
        entry=destination/"result.json"
        record=read(entry)
        assert record["key"]==list(task) and record["plan_sha256"]==plan_hash
        for file,h in record["artifacts"].items():
            artifact=destination/file
            if file=="predictions.npy":
                verify(artifact,h)
            else:
                assert artifact.stat().st_size==record["model_bytes"]
        cells[task]=record
        fingerprints["/".join(task)]=sha(entry)
    assert len(cells)==941
    return cells,fingerprints


def archived_context(group, plan):
    """Use the legacy contrast only on exactly matched rows."""
    dataset,split=group["dataset"],group["split"]
    src=plan["sources"][dataset]
    frame=table(src["table"],src["sha256"])
    original=selectors(group["inputs"]["synrfp_default"],frame)
    archives={}
    hashes={}
    for method,folder,fp in zip(CONTEXT,(RETRAIN,ROLES),
                              ("synrfp_r3b2048","synrfp_du_r3b2048"),strict=True):
        directory=folder/"models"/dataset/split/fp
        path=directory/"result.json"
        if not path.exists():
            return None,{},"no_archived_matched_mlp"
        old=read(path)
        if (old["ntrain"],old["nval"],old["ntest"])!=tuple(len(x) for x in original.values()):
            return None,{},"different_archived_cohort"
        with np.load(verify(directory/"rows.npz",old["artifacts"]["rows.npz"]),allow_pickle=False) as stored:
            if any(not np.array_equal(stored[k],v) for k,v in original.items()):
                return None,{},"different_archived_cohort"
        assert old["feature_sha256"]==src["context"][method]["sha256"]
        archives[method]={k:old[k] for k in ("r2","mae","rmse")}
        hashes[str(path)]=sha(path)
    return archives,hashes,"matched"


def regression(plan, fitted):
    rows, mlp, hashes=[],[],{}
    for group in plan["groups"]:
        dataset,split=group["dataset"],group["split"]
        index=dict(panel=group["panel"],split=split,dataset=dataset,n_test=group["ntest"])
        archives,h,status=archived_context(group,plan)
        hashes.update(h)
        for method in (*BENCHMARK,*CONTEXT):
            r=fitted[dataset,split,method,"rf_regressor"]
            rows.append(index|{k:r[k] for k in ("representation","learner","r2","mae","rmse","runtime_seconds")})
            if method in BENCHMARK or archives is not None:
                metrics=group["inputs"][method]["metrics"] if method in BENCHMARK else archives[method]
                mlp.append(index|dict(representation=method,learner="released_mlp",**metrics))
        group["mlp_context_status"]=status
    a,b=pd.DataFrame(rows),pd.DataFrame(mlp)
    assert len(a)==621
    z=pd.concat([a,b],ignore_index=True)
    scores=summary(z,["panel","representation","learner"],("r2","mae","rmse"))
    ref=b.groupby(["panel","representation"]).r2.agg(
        released_mlp_r2_mean="mean",released_mlp_r2_sample_sd="std",released_mlp_splits="size").reset_index()
    scores=scores.merge(ref,on=["panel","representation"],validate="many_to_one",how="left")
    ranking=[]
    for panel, group in scores[scores.representation.isin(BENCHMARK)].groupby("panel"):
        r=group.pivot(index="representation",columns="learner",values="r2_mean")
        a_mean,b_mean=r["rf_regressor"],r["released_mlp"]
        ac,bc=a_mean.rank(ascending=False),b_mean.rank(ascending=False)
        tau=kendalltau(ac,bc).statistic
        for method in BENCHMARK:
            for learner,ranks in (("rf_regressor",ac),("released_mlp",bc)):
                ranking.append(dict(panel=panel,representation=method,learner=learner,
                    mean_r2=r.loc[method,learner],r2_rank=ranks.loc[method],
                    released_mlp_rank=bc.loc[method],kendall_tau_vs_mlp=tau if learner=="rf_regressor" else 1.))
    context=[]
    for panel,g in z.groupby("panel"):
        statuses={x["mlp_context_status"] for x in plan["groups"] if x["panel"]==panel}
        source_splits=sum(x["panel"]==panel for x in plan["groups"])
        for learner,s in g.groupby("learner"):
            for method in ("synrfp_optimal","wl_r3_delta_union"):
                left=s[s.representation==method].set_index("split")
                right=s[s.representation=="wl_r3_delta"].set_index("split")
                common=left.index.intersection(right.index)
                assert len(common)==source_splits if learner=="rf_regressor" else len(common) in (0,source_splits)
                delta=left.loc[common,"r2"]-right.loc[common,"r2"]
                context.append(dict(panel=panel,learner=learner,
                    contrast="D+S_minus_D" if method=="synrfp_optimal" else "D+U_minus_D",
                    n_matched=len(delta),mean_delta_r2=delta.mean(),sd_delta_r2=delta.std(),
                    wins=int((delta>0).sum()) if len(delta) else None,
                    ties=int((delta==0).sum()) if len(delta) else None,
                    scope="matched" if len(delta) else "+".join(sorted(statuses))))
    return a,scores,pd.DataFrame(ranking),pd.DataFrame(context),hashes


def summary(frame, keys, metrics):
    result=[]
    for group, rows in frame.groupby(keys):
        row=dict(zip(keys,group,strict=True),n_splits=len(rows))
        for measure in metrics:
            v=rows[measure].dropna()
            row.update({measure+"_mean":v.mean(),measure+"_sample_sd":v.std(),measure+"_defined_splits":len(v)})
        result.append(row)
    return pd.DataFrame(result)


def baseline_predictions(raw, labels, selections, recipes):
    train,test=selections["train"],selections["test"]
    observed=raw.Yield.to_numpy()
    panel=raw.iloc[train].assign(author_success=labels[train],released_yield=observed[train])
    groups=panel.groupby(recipes,dropna=False)
    test_recipes=pd.MultiIndex.from_frame(raw.iloc[test][recipes])
    yield_mean=float(np.mean(observed[train]))
    success_mean=float(np.mean(labels[train]))
    return {
        "training_recipe_mean":groups.released_yield.mean().reindex(test_recipes).fillna(yield_mean).to_numpy(),
        "training_global_mean":np.full(len(test),yield_mean),
        "training_recipe_success":groups.author_success.mean().reindex(test_recipes).fillna(success_mean).to_numpy(),
    }


def classifier(plan, fitted):
    rank,recipes,eligible,hashes=r92_inputs()
    raw=pd.read_csv(verify(AUTHOR,AUTHOR_SHA),low_memory=False)
    observed, labels=raw.Yield.to_numpy(),raw.labels.to_numpy()
    pair_scores,split_scores=[],[]
    pooled={}
    for entry in plan["groups"]:
        dataset,split=entry["dataset"],entry["split"]
        if dataset!="ncs_bh":
            continue
        src=plan["sources"][dataset]
        selected=selectors(entry["inputs"]["synrfp_default"],table(src["table"],src["sha256"]))
        test=selected["test"]
        y=labels[test]
        def add_metrics(learner,method,prob,cut):
            auc,bacc=roc_auc_score(y,prob),balanced_accuracy_score(y,prob>=cut if cut==.5 else prob>cut)
            split_scores.append(dict(method=method,learner=learner,split=split,scope="holdout",n_test=len(y),
                                     roc_auc=auc,balanced_accuracy=bacc,positive_threshold=cut))
            pooled.setdefault((learner,method,cut),[]).append((y,prob))
            return auc,bacc
        for learner in ("rf_classifier","mlp_classifier"):
            probs={}
            for method in CLASS:
                ticket=(dataset,split,method,learner)
                r=fitted[ticket]
                p=path_for(*ticket)/"predictions.npy"
                probs[method]=np.load(verify(p,r["artifacts"][p.name]),allow_pickle=False)
                auc,bacc=add_metrics(learner,method,probs[method],.5)
                np.testing.assert_allclose((auc,bacc),(r["roc_auc"],r["balanced_accuracy"]),atol=1e-14,rtol=0)
            rows,_=rank(split,raw,observed,selected,probs)
            assert sum(r["method"]=="synrfp_default" for r in rows)==eligible.loc[split]
            pair_scores.extend(r|{"learner":learner} for r in rows if r["method"] in CLASS)
        baselines=baseline_predictions(raw,labels,selected,recipes)
        # Reuse the exact Request 92 scorer for recipe means, replicate averaging
        # and the 609-pair selector. One placeholder key carries success rates.
        rows,_=rank(split,raw,observed,selected,{m:baselines["training_recipe_success"] for m in CLASS})
        pair_scores.extend(r|{"learner":"train_baseline",
             "method":"training_recipe_success" if r["method"]=="synrfp_default" else r["method"]}
             for r in rows if r["method"] in ("synrfp_default",*BASELINES[:2]))
        for method,values in baselines.items():
            add_metrics("train_baseline",method,values,.5 if method=="training_recipe_success" else 10.)
    pairs=pd.DataFrame(pair_scores)
    assert len(pairs)==609*11 and (pairs.groupby(["learner","method"]).size()==609).all()
    ranks=[]
    for label,g in [("all_40_holdouts",pairs),*list(pairs.groupby("split"))]:
        for (learner,method),rows in g.groupby(["learner","method"]):
            ranks.append(dict(learner=learner,method=method,split=label,n_pairs=len(rows),
                        n_spearman_defined=int(rows.spearman.notna().sum()),
                        spearman_mean=rows.spearman.mean(),spearman_sample_sd=rows.spearman.std(),
                        top1_success_rate=rows.top1_success.mean(),top4_success_rate=rows.top4_success.mean(),
                        n_pairs_without_observed_success=int((~rows.any_success).sum())))
    for (learner,method,cut),rows in pooled.items():
        y=np.concatenate([a for a,_ in rows])
        p=np.concatenate([b for _,b in rows])
        before=[s for s in split_scores if s["learner"]==learner and s["method"]==method]
        assert len(before)==40 and len(y)==274970
        native=dict(learner=learner,method=method,split="all_40_holdouts",scope="pooled_test_occurrences",
                    n_test=len(y),positive_threshold=cut,roc_auc=roc_auc_score(y,p),
                    balanced_accuracy=balanced_accuracy_score(y,p>=cut if cut==.5 else p>cut))
        for measure in ("roc_auc","balanced_accuracy"):
            values=np.array([r[measure] for r in before])
            native.update({measure+"_split_mean":float(values.mean()),
                           measure+"_split_sample_sd":float(values.std(ddof=1))})
        split_scores.append(native)
    return pd.DataFrame(ranks),pd.DataFrame(split_scores),hashes


def report_note(plan,execution,context):
    cpu=next((x.split(":",1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines()
              if x.startswith("model name")),platform.processor())
    return f"""# Request 93 yield learner comparison

The study fits {plan['expected_regressors']} random-forest regressors on the 69 released
split instances, including five publication- and duplicate-grouped USPTO folds.
The seven benchmark representations reuse the original saved standardized train
matrices and test matrices. Both context variants use the same held-out selectors.
The USPTO D and D+U caches were built in the existing literal-input profile since
the full-row variants were unavailable. All 143,535 archived D+S vectors were
replayed to verify the profile. The old USPTO input refusals use the same three
row-specific literal graph adapters.

The forest uses 500 trees, square-root feature sampling, minimum leaf size one,
and seed 42. There was no test-set tuning. XGBoost was excluded per the project
decision. The seven-representation ranking uses mean R² and Kendall's tau against
the released MLP ranks. The matched MLP construction contrasts are published
where the archived D and D+U selectors match this cohort. Their comparison scope
is explicit in the context table.

The study fits {plan['expected_classifiers']} success classifiers on the exact expanded
Buchwald–Hartwig fit-train and validation selectors. The forest classifier uses
balanced class weights. The CPU MLP retains the 512-unit ReLU architecture and
original optimizer and checkpoint-selection settings, with sigmoid output and
binary cross-entropy on the author labels. On this source, {plan['author_disagreements_with_recomputed_gt10']}
released labels differ from applying yield >10 to the continuous value.
ROC AUC and balanced accuracy use the released binary labels. A 0.5 cutoff
applies to probabilities, and yield-mean controls use yield >10.

The condition ranking uses the same 609 pair-by-holdout instances, recipe
replicate averaging and stable recipe tie order as Request 92. A top-ranked recipe
succeeds when its observed mean yield exceeds 10%. Per-recipe baselines use
fit-train recipes and use the respective fit-train global mean for unseen recipes.
The classifier table includes pooled test occurrences and split mean and SD.
Repeated test occurrences do not represent independent observations.

The run used {execution['jobs']} CPU workers on the {cpu} for
{execution['elapsed_seconds']/3600:.2f} fit wall-clock hours. Model files are
retained locally. Package versions, row bindings and code hashes are in the
completion manifest.
"""


def main():
    source=ROOT/"PLAN.json"
    plan,plan_hash=read(source),sha(source)
    execution=read(ROOT/"FIT_COMPLETE.json")
    assert execution["plan_sha256"]==plan_hash and execution["cells"]==941
    fitted,hashes=results(plan,plan_hash)
    cells,overview,ranks,context,old=regression(plan,fitted)
    cr,metrics,r92=classifier(plan,fitted)
    assert len(ranks)==70 and len(context)==20
    OUTPUT.mkdir(parents=True,exist_ok=True)
    files={"cells.csv":cells,"summary.csv":overview,"ranking.csv":ranks,"context.csv":context,
           "classifier_ranking.csv":cr,"classifier_metrics.csv":metrics}
    for name,df in files.items():
        df.to_csv(OUTPUT/name,index=False)
    note=OUTPUT/"note.md"
    note.write_text(report_note(plan,execution,context))
    write(OUTPUT/"COMPLETE.json",dict(phase="complete",release_plan_sha256=plan_hash,
        request_commit=plan["request_commit"], archived_yield_cells_sha256=plan["release_table_sha256"],
        author_label_sha256=plan["author_sha256"], author_versus_yield_threshold_mismatches=
        plan["author_disagreements_with_recomputed_gt10"], execution=execution,
        model_receipt_sha256=hashes, matched_mlp_context_sha256=old, r92_revision=R92_REV,
        r92_inputs_sha256=r92, uspto_context=read(ROOT/"uspto_context/COMPLETE.json"),
        report_versions={p:version(p) for p in ("pandas","numpy","scipy","scikit-learn")},
        code_sha256={p.name:sha(p) for p in HERE.glob("*.py")},
        table_sha256={n:sha(OUTPUT/n) for n in (*files,"note.md")}))
    print("Wrote",len(cells),"regressors and",sum(r["learner"].endswith("_classifier") for r in fitted.values()),
          "classifiers to",OUTPUT,flush=True)


if __name__=="__main__":
    main()
