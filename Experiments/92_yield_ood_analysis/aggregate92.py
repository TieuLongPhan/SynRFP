"""Summarize held-out errors in pooled similarity quantiles and complete the examples."""
from importlib.metadata import version
import json

import numpy as np
import pandas as pd

from preflight import HERE, digest, read, verify
from rank92 import WORK, OUTPUT, METHODS, data


def get_similarity(dataset, space, split, test):
    file = WORK/"distance"/dataset/space/f"{split}.npz"
    receipt = read(file.with_suffix(".json"))
    with np.load(verify(file,receipt["result_sha256"]),allow_pickle=False) as z:
        np.testing.assert_array_equal(z["test"],test)
        return z["similarity"].astype(np.float64),receipt


def stats(group):
    error, y = group["error"].to_numpy(),group["yield"].to_numpy()
    variance = np.sum((y-y.mean())**2) if len(y) else 0
    return dict(n=len(group),mae=np.abs(error).mean() if len(y) else None,
                r2=1-np.sum(error**2)/variance if len(y)>1 and variance>0 else None)


def bins(table, fit_units):
    edges = np.quantile(table.similarity,np.arange(1,5)/5)
    table = table.assign(bin=np.searchsorted(edges,table.similarity,side="left")+1)
    results = []
    for i, (low,high) in enumerate(zip(np.r_[0.,edges],np.r_[edges,1.],strict=True),1):
        panel = table[table.bin==i]
        pooled = stats(panel)
        splits = {s:stats(group) for s,group in panel.groupby("split")}
        rows = []
        for split,total in fit_units.items():
            row = splits.get(split,dict(n=0,mae=None,r2=None))
            rows.append(dict(**row,share=row["n"]/total))
        per = pd.DataFrame(rows)
        record = dict(bin=i,similarity_lower=low,similarity_upper=high,**pooled,
                      share=pooled["n"]/len(table),n_splits=len(fit_units),
                      n_splits_nonempty=int((per.n>0).sum()))
        for key in ("n","share","mae","r2"):
            series = pd.to_numeric(per[key],errors="raise")
            record.update({key+"_split_mean":series.mean(),key+"_split_sample_sd":series.std(ddof=1)})
        record["n_r2_defined_splits"] = int(per.r2.notna().sum())
        results.append(record)
    assert sum(r["n"] for r in results)==len(table)
    return results


def main():
    cohorts, sizes, hashes = {}, {}, {}
    selected = pd.read_csv(verify(WORK/"examples_selected.csv",read(WORK/"RANK_AND_EXAMPLES.json")["examples_selected_sha256"]))
    selected["training_similarity"] = np.nan
    for dataset,split,cohort,selectors,predictions in data():
        test = selectors["test"]
        y = cohort.yield_pct.to_numpy()[test]
        cache = {}
        shared = "author_drfp" if dataset=="ncs_bh" else "drfp"
        for name in {*METHODS,shared}:
            values, receipt = get_similarity(dataset,name,split,test)
            cache[name]=values
            hashes.setdefault(dataset,{}).setdefault(name,{})[split]=receipt
        sizes.setdefault(dataset,{})[split]=len(test)
        if dataset=="ncs_bh":
            in_holdout = selected.split.eq(split)
            candidates = selected.loc[in_holdout,"source_position"].to_numpy()
            positions = pd.Index(test).get_indexer(candidates)
            assert (positions>=0).all()
            selected.loc[in_holdout,"training_similarity"]=cache[shared][positions]
        for method in METHODS:
            for regime, name in (("common",shared),("own",method)):
                cohorts.setdefault((dataset,method,regime,name),[]).append(pd.DataFrame(
                    dict(split=split,yield_=y,similarity=cache[name],error=predictions[method]-y)).rename(columns={"yield_":"yield"}))
    assert np.isfinite(selected.training_similarity).all()
    results = []
    for (dataset,method,regime,name), frames in sorted(cohorts.items()):
        table = pd.concat(frames,ignore_index=True)
        results.extend(dict(dataset=dataset,method=method,space=regime,fingerprint=name,**r)
                       for r in bins(table,sizes[dataset]))
    assert len(results)==80
    OUTPUT.mkdir(exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT/"distance_bins.csv",index=False)
    selected.to_csv(OUTPUT/"examples.csv",index=False)
    rank = read(WORK/"RANK_AND_EXAMPLES.json")
    verify(OUTPUT/"condition_ranking.csv",rank["condition_ranking_sha256"])
    tested = {k:sum(v.values()) for k,v in sizes.items()}
    selectors = sum(len(splits) for spaces in hashes.values() for splits in spaces.values())
    assert selectors==220
    validation = {name:read(WORK/f"VALIDATION_{name}.json") for name in ("own","author")}
    assert [len(v["controls"]) for v in validation.values()]==[8,1]
    preflight = read(HERE/"publication_metadata/preflight_v1/PREFLIGHT.json")
    summary = dict(phase="complete",request_commit=preflight["request_commit"],
        evaluation_plan_sha256=preflight["plan_sha256"],
        released_metrics_sha256=preflight["release_table_sha256"],
        test_row_occurrences=tested,similarity_selector_cells=selectors,
        similarity_inputs=hashes,authors=read(WORK/"AUTHOR_ALIGNED.json"),
        ranking_and_examples=rank,quantile_bins=len(results),similarity_validation=validation,
        software={p:version(p) for p in ("numpy","pandas","rdkit","scipy")},
        source_sha256={p.name:digest(p) for p in HERE.iterdir() if p.suffix in (".py",".cpp")},
        outputs_sha256={f:digest(OUTPUT/f) for f in ("condition_ranking.csv","distance_bins.csv","examples.csv","note.md")})
    (OUTPUT/"COMPLETE.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(tested,"and",len(results),"similarity summaries",flush=True)


if __name__ == "__main__":
    main()
