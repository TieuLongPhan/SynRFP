"""Take a hash-checked snapshot of the completed parts of Request 83."""
from datetime import datetime, timezone
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parents[2] / "SynRFP-ecreact-target-pairs/Experiments/83_ecreact_target_pairs"
sys.path.insert(0,str(PREVIOUS))
from levels import score_sufficient, statistics
from mapping import ROOT, OUT as SOURCE, sha, read, write

OUT = HERE/"publication_metadata/ec_partial_v1"
METRICS=("D_mcc","DU_mcc","gain_mcc","D_error","DU_error","gain_error")


def interval(level, prediction, y, group):
    assert np.all(prediction>=0)
    model=score_sufficient(y,prediction,group)
    n=int(group.max())+1
    point=statistics([model],np.ones((n,1)))[0,:,0]
    rng=np.random.default_rng(2026)
    draws=np.empty((6,10000))
    for start in range(0,10000,40):
        weights=np.empty((n,40))
        for j in range(40):
            weights[:,j]=np.bincount(rng.integers(n,size=n),minlength=n)
        draws[:,start:start+40]=statistics([model],weights)[0]
    rows=[]
    for kind,score,values in zip(METRICS,point,draws,strict=True):
        low,high=np.quantile(values,[.025,.975])
        rows.append(dict(target=f"L{level}",metric=kind,estimate=score,ci_low=low,ci_high=high,
                         n_rows=len(y),n_scaffold_groups=n,folds=25,partial=False,
                         status="complete_single_level"))
    return rows


def snapshot_levels(pins):
    root=ROOT/"A"
    prep=read(root/"PREPARED.json")
    pins[str(root/"PREPARED.json")]=sha(root/"PREPARED.json")
    with np.load(root/"cohort.npz",allow_pickle=False) as z:
        y=z["y"];group=np.unique(z["groups"],return_inverse=True)[1];codes=z["common_splits"]
    assert sha(root/"cohort.npz")==prep["files_sha256"]["cohort.npz"]
    records=[]
    single=[]
    for level in (1,2,3):
        prediction=np.full((2,5,len(y)),-1,dtype=np.int32)
        for seed,code in enumerate(codes):
            folder=root/"jobs"/f"L{level}-s{seed:02d}"
            idx=np.flatnonzero(code==2)
            archive=folder/"archived.npz"
            for arm,mode in enumerate(("delta","delta_union")):
                result=folder/mode/"COMPLETE.json"
                values=None
                if archive.exists():
                    assert sha(archive)==prep["files_sha256"][str(archive.relative_to(root))]
                    with np.load(archive,allow_pickle=False) as old:
                        np.testing.assert_array_equal(old["test"],idx)
                        values=old["d" if arm==0 else "du"]
                    state="complete_archived"
                elif result.exists():
                    receipt=read(result)
                    assert sha(result.parent/"job.json")==receipt["plan_job_sha256"]
                    for name,h in receipt["files_sha256"].items():
                        assert sha(result.parent/name)==h
                    pins[str(result)]=sha(result)
                    values=np.load(result.parent/"decoded.npy",allow_pickle=False)
                    state="complete_new"
                else:
                    state="pending"
                if values is None:
                    mcc=err=None
                else:
                    np.testing.assert_array_equal(values.shape,idx.shape)
                    prediction[arm,seed//5,idx]=values
                    t=y[idx,level-1]
                    mcc=matthews_corrcoef(t,values)
                    err=np.mean(t!=values)
                records.append(dict(level=level,seed=seed,arm=mode,n_test=len(idx),
                                    status=state,partial=state=="pending",mcc=mcc,error=err))
        if np.all(prediction>=0):
            single+=interval(level,prediction,y[:,level-1],group)
        else:
            single.append(dict(target=f"L{level}",metric="all",status="incomplete_common_fits",partial=True))
    for label in ("L3_minus_L1","L2_minus_L1"):
        single.append(dict(target=label,metric="paired_gains",status="incomplete_common_fits",partial=True))
    pd.DataFrame(single).to_csv(OUT/"ec_level_pairs.csv",index=False)
    pd.DataFrame(records).to_csv(OUT/"ec_level_fit_status.csv",index=False)
    return pd.DataFrame(single),pd.DataFrame(records)


def snapshot_mapping(pins):
    mapping=read(ROOT/"MAPPING_COMPLETE.json")
    check=read(ROOT/"B_COMPLETE.json") if (ROOT/"B_COMPLETE.json").exists() else None
    assert mapping["rows"]==185734 and mapping["validated"]==4017
    source=SOURCE/"overlap_with_enzymemap.csv"
    assert sha(source)==mapping["outputs_sha256"][source.name]
    overlap=pd.read_csv(source)
    overlap["partial"]=False
    overlap.to_csv(OUT/source.name,index=False)
    pins[str(source)]=sha(source)
    q=SOURCE/"mapping_quality.csv"
    if check:
        assert sha(q)==check["files_sha256"][q.name]
    counts=pd.read_csv(q)
    counts["partial"]=False
    counts.to_csv(OUT/"mapping_quality_partial.csv",index=False)
    pins[str(q)]=sha(q)
    pins[str(ROOT/"MAPPING_COMPLETE.json")]=sha(ROOT/"MAPPING_COMPLETE.json")
    if check:
        assert check["fits"]==60
        pins[str(ROOT/"B_COMPLETE.json")]=sha(ROOT/"B_COMPLETE.json")
        for name in ("ecreact_target_pair.csv","ec_edit_rule_coverage.csv"):
            assert sha(SOURCE/name)==check["files_sha256"][name]
            df=pd.read_csv(SOURCE/name)
            df["partial"]=False
            df.to_csv(OUT/name,index=False)
            pins[str(SOURCE/name)]=sha(SOURCE/name)
    return bool(check)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    now=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    pins={str(PREVIOUS/"levels.py"):sha(PREVIOUS/"levels.py")}
    aggregates,cells=snapshot_levels(pins)
    b_done=snapshot_mapping(pins)
    n=int(cells.status.str.startswith("complete").sum())
    by_level=cells.groupby("level").status.apply(lambda v:int(v.str.startswith("complete").sum()))
    l3=aggregates[aggregates.target.eq("L3")].set_index("metric")
    mcc,err=(l3.loc[m] for m in ("gain_mcc","gain_error"))
    a_rule=SOURCE/"ec_level_rule_coverage.csv"
    if a_rule.exists():
        c=pd.read_csv(a_rule)
        assert len(c)==75
        c["partial"]=False
        c.to_csv(OUT/a_rule.name,index=False)
        pins[str(a_rule)]=sha(a_rule)
        rule="The completed training-rule results are included for all 75 fold-by-contrast checks."
    else:
        rule="The training-only EC-level ordering is still being calculated."
    lines=[
        "# ECReact results available from Request 83","",
        f"Snapshot at **{now}**. Request 83 continues on the workstation.",
        "",
        f"**Experiment A is partial.** {n}/150 matched fits have verified predictions. "
        f"Level 1 has {by_level[1]}/50, Level 2 has {by_level[2]}/50, and Level 3 has {by_level[3]}/50. "
        "The Level 3 estimates are complete on the common five-repeat, five-fold outer schedule. "
        f"Pooled context raised Level 3 MCC by **{mcc.estimate:+.4f}** "
        f"[{mcc.ci_low:+.4f}, {mcc.ci_high:+.4f}] and reduced error by **{err.estimate:+.4f}** "
        f"[{err.ci_low:+.4f}, {err.ci_high:+.4f}], using 10,000 paired scaffold-group draws.",
        "",
        "The paired differences with Level 1 require the other levels to close. "
        f"{rule} [Per-fit inventory](ec_level_fit_status.csv)",
        "At the observed CPU rate, the remaining common-split fits and paired intervals "
        "are expected in roughly two to four hours from this snapshot. This estimate "
        "depends on early stopping and workstation load.",
        "",
        "**Experiment B mapping is complete.** The pinned mapper accepted 185,343/185,734 "
        "under the original mapping policy. The stricter atom-conserving EnzymeMap label "
        "is defined for 4,017/185,734 (2.16%) of rows. Removing 603 "
        "exact or reverse EnzymeMap overlaps leaves 3,414. Edit labels agree with the "
        "unique available author mapping in 544/595 comparisons (91.4%). "
        "Two rows have multiple competing author edit labels.",
        "",
    ]
    if b_done:
        b=pd.read_csv(OUT/"ecreact_target_pair.csv")
        assert len(b)==18
        lines.append("**Experiment B fits are complete.** Both cohorts passed all 30 fits, "
                     "and the final held-out results are in [the target-pair table](ecreact_target_pair.csv).")
        for row in b.loc[b.target.eq("ec_level_2_minus_bond_edit")].itertuples():
            lines.append(f"- {row.scope}, EC2-minus-edit error reduction **{row.estimate:+.4f}** "
                         f"[{row.ci_low:+.4f}, {row.ci_high:+.4f}].")
    else:
        lines.append("**Experiment B fits are pending.** This snapshot contains only verified mapping and overlap counts.")
    (OUT/"status_note.md").write_text("\n".join(lines)+"\n")
    write(OUT/"SNAPSHOT.json",dict(snapshot_utc=now,request_84_revision="71f3df22",request_83_revision="fc4e13cf",
          status="partial",completed_A_fits=n,required_A_fits=150,B_fits_complete=b_done,
          code_sha256=sha(Path(__file__)),inputs_sha256=pins,
          files_sha256={p.name:sha(p) for p in OUT.iterdir() if p.name!="SNAPSHOT.json"}))
    print(n,"matched fits in snapshot, Experiment B complete:",b_done,flush=True)


if __name__=="__main__":
    main()
