"""Seal the aggregate ECReact exports after both target-pair analyses close."""
from importlib.metadata import version
import numpy as np
import pandas as pd

from mapping import HERE, OUT, ROOT, read, sha, write


def check(receipt, files):
    for name,h in files.items():
        assert sha(OUT/name)==h, f"Changed {receipt} output {name}"


def main():
    m=read(ROOT/"MAPPING_COMPLETE.json")
    a=read(ROOT/"A_COMPLETE.json")
    b=read(ROOT/"B_COMPLETE.json")
    check("level pairs",a["files_sha256"])
    check("edit pairs",b["files_sha256"])
    assert np.all([sha(OUT/name)==h for name,h in m["outputs_sha256"].items()
                   if name!="mapping_quality.csv"])
    assert m["rows"]==185734 and m["validated"]==4017
    assert (a["new_fits"],a["archived_fits"],b["fits"])==(100,50,60)
    levels=pd.read_csv(OUT/"ec_level_pairs.csv")
    edits=pd.read_csv(OUT/"ecreact_target_pair.csv")
    qc=pd.read_csv(OUT/"mapping_quality.csv").set_index("metric").iloc[:,0]
    rule_a=pd.read_csv(OUT/"ec_level_rule_coverage.csv")
    rule_b=pd.read_csv(OUT/"ec_edit_rule_coverage.csv")
    assert len(levels)==22 and len(edits)==18 and len(rule_a)==75 and len(rule_b)==2
    lines=[
        "# ECReact target pairs",
        "",
        "The full EC-level comparison holds out the same 185,734 rows across five repeats of five "
        "grouped folds. The EC level-3 schedule supplied the common outer partitions and seeds. "
        "We reused 50 level-3 fits and trained 100 level-1 and level-2 fits with the published 512-unit "
        "MLP recipe and the original target-dependent inner-validation procedure. "
        "The new fits use the archived CPU implementation of the same training arithmetic. "
        "The reused level-3 fits ran on CUDA. All three levels share the same outer test rows.",
        "",
        "| Target | D → D+U MCC gain, 95% CI | D → D+U error reduction, 95% CI |",
        "|---|---:|---:|",
    ]
    for label,x in levels.loc[levels.metric.isin(["gain_mcc","gain_error"])].groupby("target",sort=False):
        vals=x.set_index("metric")
        def span(metric):
            e=vals.loc[metric]
            return f"{e.estimate:+.4f} [{e.ci_low:+.4f}, {e.ci_high:+.4f}]"
        lines.append(f"| {label} | {span('gain_mcc')} | {span('gain_error')} |")
    lines.extend(["", "The training-only rule uses the unchanged thresholds of 5% of training rows, "
                      "30 eligible query groups and ten exact signed-count classes. "
                      "Bootstrap intervals resample scaffold groups together across arms, targets and repetitions.", ""])
    for comparison,g in rule_a.groupby("comparison",sort=True):
        counts=g.loc[g.passes].predicted_higher_gain_level.value_counts().sort_index()
        text=", ".join(f"L{int(level)} in {n}" for level,n in counts.items()) or "no direction"
        lines.append(f"- {comparison} met the rule thresholds in {int(g.passes.sum())}/{len(g)} fits, with {text}.")
    compared=int(qc["enzymemap_author_map::compared"])
    agree=int(qc["enzymemap_author_map::agree"])
    lines.extend([
        "",
        "The strict bond-edit analysis reused the full ECReact mapping archive from pinned RXNMapper v2. "
        "The original mapping accepted 185,343 rows under its own looser contract. "
        "The EnzymeMap classifier requires all atoms to map bijectively across the exact original sides. "
        "That leaves **4,017 rows (2.16%)**, or **3,414** after removing exact and reverse-direction matches "
        "to the full EnzymeMap source. "
        f"Against the EnzymeMap author's maps, the bond-edit class agreed for **{agree}/{compared} "
        f"({agree/compared:.1%})** rows with a unique reference label. Two additional rows have conflicting "
        "author edit labels. The frozen experimental EnzymeMap cohort contained no exact shared row.",
        "",
        "Both strict-label cohorts use the original EnzymeMap feature arms, three optimizer seeds, "
        "scaffold grouping, split rule and paired group-weighted bootstrap. Group assignments were "
        "constructed on the full mapped set and held fixed when overlap was removed.",
        "",
        "| Cohort | Test rows | EC2 error reduction | Edit error reduction | EC2 minus edit, 95% CI |",
        "|---|---:|---:|---:|---:|",
    ])
    for scope,g in edits.groupby("scope",sort=False):
        rows=g[g.comparator.eq("d2048")].set_index("target")
        ec,ed,contrast=(rows.loc[n] for n in ("ec_level_2","bond_edit_multiset","ec_level_2_minus_bond_edit"))
        lines.append(f"| {scope} | {int(contrast.n_test)} | {ec.estimate:+.4f} | {ed.estimate:+.4f} | "
                     f"{contrast.estimate:+.4f} [{contrast.ci_low:+.4f}, {contrast.ci_high:+.4f}] |")
    lines.append("")
    significant=[]
    for r in edits.loc[edits.target.eq("ec_level_2_minus_bond_edit")].itertuples():
        if r.ci_low>0:
            significant.append(f"{r.scope} supports a larger EC2 gain")
        elif r.ci_high<0:
            significant.append(f"{r.scope} supports a larger bond-edit gain")
        else:
            significant.append(f"{r.scope} is inconclusive")
    ranks=[f"{r.scope} → {r.predicted_higher_gain or 'abstain'}" for r in
           rule_b.fillna("").itertuples()]
    lines.append("The training-only ordering was " + ", ".join(ranks) + ". "
                 + ". ".join(text[0].upper()+text[1:] for text in significant) + ". "
                 "The strict mapping subset defines the scope of this comparison.")
    (OUT/"ecreact_target_pairs_note.md").write_text("\n".join(lines)+"\n")
    receipts={name:sha(ROOT/name) for name in ("MAPPING_COMPLETE.json","A_COMPLETE.json","B_COMPLETE.json")}
    b_inputs={scope:read(ROOT/"B"/scope/"PLAN.json") for scope in ("all_mapped","without_enzymemap")}
    code=("levels.py","level_fit.py","mapping.py","bond_pairs.py","finalize.py")
    write(OUT/"COMPLETE.json",dict(status="complete",request_revision="fc4e13cf",counts={"A":150,"B":60},
          archive_reused={"A":50,"mapped_rows":m["rows"]},local_completion_sha256=receipts,
          inputs_sha256={**m["inputs_sha256"],**a["inputs_sha256"],
                         **{f:sha_ for x in b_inputs.values() for f,sha_ in x["input_sha256"].items()}},
          mapping_profile=m["profile"],mapping_checkpoint_sha256=m["checkpoint_sha256"],
          methods={"A_training":"archived native 512-unit MLP CPU port, architecture/seed unchanged",
                   "A_population":"matched original ECReact L3 outer folds, full cohort",
                   "B_training":"pinned EnzymeMap arms, seeds, grouping and sklearn learner"},
          packages={p:version(p) for p in ("numpy","pandas","scikit-learn","scipy","rdkit")},
          files_sha256={p.name:sha(p) for p in OUT.iterdir() if p.name!="COMPLETE.json"},
          code_sha256={name:sha(HERE/name) for name in code}))
    print("Release verified and sealed",flush=True)


if __name__=="__main__":
    main()
