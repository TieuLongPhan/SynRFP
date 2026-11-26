"""Matched EC2 and strict bond-edit predictions on the two ECReact cohorts."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from mapping import (
    BASE, EM_CODE, EM_REV, HERE, OUT, ROOT, canonical_reaction, checked_enzymemap,
    edit_label, key, read, reaction_smiles, sha, write,
)
from audit_enzymemap_labels import classification_diagnostic
from audit_inputs import PINS
from enzymemap_cohort import assign_groups, split_for
from prepare_qp_features import ARMS, encode
from run_enzymemap import NAMES, SEEDS, collect, fit_cell, score

STATE={}
BROOT=ROOT/"B"
POPULATIONS=("all_mapped","without_enzymemap")
LEDGER=ROOT/"mapped_labels.csv.gz"


def author_agreement(ledger):
    target=dict(zip(ledger.loc[ledger.label.ne("")&ledger.enzyme_forward,"direction"],
                    ledger.loc[ledger.label.ne("")&ledger.enzyme_forward,"label"],strict=True))
    raw=BASE/"context_dependence_d421d7ae/sources/enzymemap_v2_brenda2023.csv.gz"
    assert sha(raw)==PINS["enzymemap"][1]
    seen=defaultdict(set)
    errors=Counter()
    for ref in csv.DictReader(gzip.open(raw,"rt",newline="")):
        sides,_=canonical_reaction(ref["unmapped"])
        if sides is None:
            continue
        reaction=reaction_smiles(sides)
        d=key(reaction)
        if d not in target:
            continue
        try:
            label=edit_label(reaction,ref["mapped"])
        except ValueError as exc:
            errors[str(exc)]+=1
        else:
            seen[d].add(label)
    stats=Counter()
    for row in ledger.loc[ledger.label.ne("")&ledger.enzyme_forward].itertuples():
        choices=seen[row.direction]
        if not choices:
            stats["no_valid_author_map"]+=1
        elif len(choices)>1:
            stats["ambiguous_author_map"]+=1
        else:
            stats["compared"]+=1
            stats["agree" if choices=={row.label} else "discordant"]+=1
    stats["shared_mapped_ec_react_rows"]=int((ledger.label.ne("")&ledger.enzyme_forward).sum())
    extra=pd.DataFrame([dict(metric=f"enzymemap_author_map::{name}",count=stats[name])
                        for name in ("shared_mapped_ec_react_rows","compared","agree","discordant",
                                     "ambiguous_author_map","no_valid_author_map")])
    base=pd.read_csv(OUT/"mapping_quality.csv")
    base=base[~base.metric.str.startswith("enzymemap_author_map::")]
    pd.concat((base,extra),ignore_index=True).to_csv(OUT/"mapping_quality.csv",index=False)
    write(BROOT/"author_overlap.json",dict(counts=dict(stats),author_refusal_reasons=dict(errors),
                                          reference_sha256=sha(raw),original_direction_only=True))
    print("Author map agreement",dict(stats),flush=True)


def fit(args):
    scope,i,arm,seed=args
    s=STATE[scope]
    parts=s["parts"]
    y=s["labels"][:,i]
    classes=np.unique(y[parts[0]])
    vocabulary={name:j for j,name in enumerate(classes)}
    train,val=(np.array([vocabulary.get(name,-1) for name in y[part]],dtype=np.int64) for part in parts[:2])
    assert np.all(train>=0) and len(classes)>1
    return scope,fit_cell(s["packed"][arm],parts,train,val,np.arange(len(classes)),arm,NAMES[i],seed,
                          BROOT/scope/"fits",s["plan_sha"])


def grouping_for(text):
    sides, refusal=canonical_reaction(text)
    if sides is None:
        raise ValueError(refusal)
    return sides


def prepare(workers):
    BROOT.mkdir(parents=True,exist_ok=True)
    pin=checked_enzymemap({})
    verified=read(ROOT/"MAPPING_COMPLETE.json")
    assert sha(LEDGER)==verified["labels_sha256"]
    base=BASE/"all_endpoint_factorial_full_v1"
    plan=read(base/"plan.json")
    assert sha(base/"plan.json")==read(base/"plan_seal.json")["sha256"]
    spec=plan["sources"]["ecreact"]
    source=Path(spec["table"])
    assert sha(source)==plan["pinned_files"][str(source)]
    source_data=pd.read_csv(source,dtype={"row_id":str})
    label_file=Path(plan["panels"]["ecreact_2"]["table"])
    assert sha(label_file)==plan["pinned_files"][str(label_file)]
    labels=pd.read_csv(label_file,dtype={"row_id":str})
    np.testing.assert_array_equal(labels.source_position,source_data.source_position)
    np.testing.assert_array_equal(labels.row_id,source_data.row_id)
    ledger=pd.read_csv(LEDGER,keep_default_na=False)
    np.testing.assert_array_equal(ledger.source_position,source_data.source_position)
    author_agreement(ledger)
    ix=np.flatnonzero(ledger.label.ne("")&ledger.feature_ok)
    assert len(ix)==4017
    selected=ledger.iloc[ix]
    source_rxn=source_data.rxn.iloc[ix].to_numpy()
    canonical=[grouping_for(text) for text in source_rxn]
    candidates=[dict(observed_directions=(sides,)) for sides in canonical]
    grouped=assign_groups(candidates)
    group=np.array([q["group_id"] for q in grouped])
    split=np.array([split_for(g) for g in group])
    y=np.column_stack((labels.y.to_numpy()[ix].astype(str),selected.label.to_numpy(str)))
    row_ids=source_data.row_id.iloc[ix].to_numpy(str)
    full=[(i,rid,reaction_smiles(c)) for i,(rid,c) in enumerate(zip(row_ids,canonical,strict=True))]
    packed= {arm:[] for arm in ARMS}
    keys=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,rid,v,exact,_,refusal in pool.map(encode,full,chunksize=48):
            assert rid==row_ids[i] and v is not None, f"Invalid held-out feature {i}: {refusal}"
            for arm in ARMS:
                packed[arm].append(v[arm])
            keys.append(exact)
    arrays={arm:np.asarray(rows) for arm,rows in packed.items()}
    cut=~(selected.enzyme_forward|selected.enzyme_reverse).to_numpy()
    results=[]
    for scope,mask in zip(POPULATIONS,(np.ones(len(ix),bool),cut),strict=True):
        path=BROOT/scope
        path.mkdir(exist_ok=True)
        (path/"fits").mkdir(exist_ok=True)
        p={name:mat[mask] for name,mat in arrays.items()}
        partitions=tuple(np.flatnonzero(split[mask]==s) for s in ("train","validation","test"))
        for a,b in ((0,1),(0,2),(1,2)):
            assert not np.intersect1d(group[mask][partitions[a]],group[mask][partitions[b]]).size
        train=partitions[0]
        bits=p["d_pooled"][train,128:]
        ys=y[mask]
        gs=group[mask]
        k=[keys[i] for i in np.flatnonzero(mask)[train]]
        _,_,_,_,eligible,diag,ties=classification_diagnostic(bits,k,gs[train],ys[train])
        n=int(eligible.sum())
        n_groups=len(np.unique(gs[train][eligible]))
        n_exact=len({s for s,use in zip(k,eligible,strict=True) if use})
        passes=n/len(train)>=.05 and n_groups>=30 and n_exact>=10
        winner=NAMES[int(np.argmax(diag))] if passes and diag[0]!=diag[1] else None
        rule=dict(scope=scope,train_rows=len(train),eligible=n,coverage=n/len(train),query_groups=n_groups,
                  exact_d_classes=n_exact,passes=passes,predicted_higher_gain=winner,
                  diagnostic_ec2=diag[0] if diag else None,diagnostic_edit=diag[1] if diag else None,
                  n_boundary_ties=ties)
        np.savez_compressed(path/"prepared.npz",y=ys,groups=gs,split=split[mask],source_positions=ix[mask],
                            row_ids=row_ids[mask],**p)
        identity=dict(input_sha256={**pin,str(source):sha(source),str(label_file):sha(label_file),
                                    str(LEDGER):sha(LEDGER),str(path/"prepared.npz"):sha(path/"prepared.npz")},
                      source_mapper=verified["source_mapper"],mapping_sha256=sha(ROOT/"MAPPING_COMPLETE.json"),
                      baseline_enzymemap_revision=EM_REV, rule=rule, code_sha256=sha(Path(__file__)),
                      match_protocol=dict(arms=ARMS,seeds=SEEDS,classifier="EnzymeMap frozen MLP",
                                          partitions=[len(s) for s in partitions]))
        write(path/"PLAN.json",identity)
        STATE[scope]=dict(packed=p,parts=partitions,labels=ys,groups=gs,plan_sha=sha(path/"PLAN.json"),
                          identity=identity)
        results.append(rule)
        print(scope,len(ys),[len(s) for s in partitions],rule,flush=True)
    pd.DataFrame(results).to_csv(OUT/"ec_edit_rule_coverage.csv",index=False)


def analyze(receipts):
    records=[]
    for scope in POPULATIONS:
        s=STATE[scope]
        raw=[r for p,r in receipts if scope==p]
        path=BROOT/scope
        fitted=collect(path/"fits",raw,s["parts"][2],s["plan_sha"])
        train,_,test=s["parts"]
        ytest=s["labels"][test]
        ytrain=s["labels"][train]
        vocabs=[np.unique(ytrain[:,i]) for i in range(2)]
        details=score(fitted,ytest,ytrain,vocabs,s["groups"][test],"ec_level_2",path)
        write(path/"SCORES.json",details)
        for row in details["estimates"]:
            low,high=row["interval"]
            records.append(dict(scope=scope,target=row["target"],comparator=row["comparator"],
                                n_source_rows=len(s["labels"]),n_train=len(train),
                                n_test=len(test),n_test_groups=details["groups"],
                                estimate=row["gain"],ci_low=low,ci_high=high,
                                majority_group_error=1-details["baseline_and_coverage"][row["target"]]["group_accuracy"],
                                unseen_test_label_rows=details["baseline_and_coverage"][row["target"]]["unseen_test_label_rows"]))
        point=details["interaction"]
        low,high=point["interval"]
        records.append(dict(scope=scope,target="ec_level_2_minus_bond_edit",comparator="d2048",
                            n_source_rows=len(s["labels"]),n_train=len(train),n_test=len(test),
                            n_test_groups=details["groups"],estimate=point["gain_difference"],ci_low=low,ci_high=high))
    assert len(records)==18
    pd.DataFrame(records).to_csv(OUT/"ecreact_target_pair.csv",index=False)
    write(ROOT/"B_COMPLETE.json",dict(fits=len(receipts),baseline=EM_REV,seed=2026,draws=10000,
        plan_sha256={name:v["plan_sha"] for name,v in STATE.items()},code_sha256=sha(Path(__file__)),
        files_sha256={n:sha(OUT/n) for n in ("ecreact_target_pair.csv","ec_edit_rule_coverage.csv",
                                             "overlap_with_enzymemap.csv","mapping_quality.csv")}))
    print("EC2 and bond-edit results complete",len(receipts),"fits",flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs",type=int,default=6)
    a=p.parse_args()
    prepare(a.jobs)
    tasks=list(product(POPULATIONS,range(2),ARMS,SEEDS))
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        receipts=list(pool.map(fit,tasks))
    assert len(receipts)==60
    analyze(receipts)


if __name__=="__main__":
    main()
