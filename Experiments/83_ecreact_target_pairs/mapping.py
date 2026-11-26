"""Revalidate the pinned RXNMapper ECReact archive with EnzymeMap's exact edit contract."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
import gzip
from hashlib import file_digest, sha256
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from rdkit import RDLogger

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = Path.home() / ".local/share/synrfp-runs"
ROOT = BASE / "request83_ecreact_target_pairs_v1"
OUT = HERE / "publication_metadata/ecreact_target_pairs_v1"
EM = REPO.parent / "SynRFP-enzymemap-precheck"
EM_CODE = EM / "Experiments/74_context_dependence"
EM_RUNS = EM / "Experiments/work/74_context_dependence"
EM_REV = "68e3784c"
STUDY41 = BASE / "syncluster_main12_v1"
sys.path.insert(0,str(EM_CODE))

from audit_inputs import PINS
from enzymemap_cohort import canonical_reaction, reaction_smiles, reverse_key, scaffold, product_anchor
from map_enzymemap_labels import CHECKPOINT, MAPPER_REVISION, edit_label, read, write

RDLogger.DisableLog("rdApp.*")


def sha(path):
    with Path(path).open("rb") as f:
        return file_digest(f,"sha256").hexdigest()


def key(value):
    return sha256(value.encode()).hexdigest()


def checked_enzymemap(pins):
    names=("enzymemap_cohort.py","map_enzymemap_labels.py","audit_quantumpioneer_graphs.py",
           "audit_enzymemap_labels.py","prepare_qp_features.py","run_enzymemap.py")
    for name in names:
        path=EM_CODE/name
        raw=subprocess.check_output(["git","-C",str(EM),"show",f"{EM_REV}:{path.relative_to(EM)}"])
        assert raw==path.read_bytes()
        pins[str(path)]=sha(path)
    return pins


def reuse(row):
    text=row["reaction"]
    position=row["source_position"]
    sides, refusal=canonical_reaction(text)
    if sides is None:
        return position, None, None, "", "", refusal, row["mapping_origin"]
    direction=reaction_smiles(sides)
    reverse=reaction_smiles(tuple(reversed(sides)))
    scaffold_id=scaffold(product_anchor(sides))[1]
    try:
        if row["status"]!="ok":
            raise ValueError(row["refusal_reason"].split(":",1)[-1])
        label=edit_label(text,row["mapped_reaction"])
    except ValueError as exc:
        label=""
        refusal=str(exc)
    return position, key(direction), key(reverse), scaffold_id, label, refusal, row["mapping_origin"]


def enzyme_sets(pins):
    source=BASE/"context_dependence_d421d7ae/sources/enzymemap_v2_brenda2023.csv.gz"
    assert sha(source)==PINS["enzymemap"][1]
    pins[str(source)]=PINS["enzymemap"][1]
    published_rows=pins[str(EM_RUNS/"enzymemap-precheck-v1/cohort.csv.gz")]=sha(EM_RUNS/"enzymemap-precheck-v1/cohort.csv.gz")
    check=read(EM_RUNS/"enzymemap-precheck-v1/eligibility_precheck.json")
    assert published_rows==check["outputs_sha256"]["cohort.csv.gz"]
    with gzip.open(EM_RUNS/"enzymemap-precheck-v1/cohort.csv.gz","rt",newline="") as f:
        cohort={r["row_id"]:r["reaction_smiles"] for r in csv.DictReader(f)}
    labels=defaultdict(set)
    for part in ("train","validation","test"):
        p=EM_RUNS/f"enzymemap-labels-{part}-v1"
        closure=read(p/"COMPLETE.json")
        mf=p/"manifest.json"
        data=p/f"{part}_labels.json"
        assert sha(mf)==closure["manifest_sha256"] and sha(data)==closure["labels_sha256"]
        manifest=read(mf)
        assert manifest["mapper_revision"]==MAPPER_REVISION and CHECKPOINT in manifest["mapper_assets"].values()
        for f in (mf,data,p/"COMPLETE.json"):
            pins[str(f)]=sha(f)
        checked=0
        for row in read(data):
            if "label" in row:
                text=cohort[row["row_id"]]
                assert edit_label(text,row["mapped_reaction"])==row["label"]
                sides,_=canonical_reaction(text)
                labels[key(reaction_smiles(sides))].add(row["label"])
                checked+=1
        assert checked==closure["validated"]
    all_direction=set()
    refuses=0
    with gzip.open(source,"rt",newline="") as f:
        for row in csv.DictReader(f):
            sides,_=canonical_reaction(row["unmapped"])
            if sides is None:
                refuses+=1
                continue
            all_direction.add(key(reaction_smiles(sides)))
    return all_direction,labels,refuses


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs",type=int,default=5)
    args=parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    OUT.mkdir(parents=True,exist_ok=True)
    pins=checked_enzymemap({})
    plan_path=STUDY41/"input_plan.json"
    closure_path=STUDY41/"inputs/sources/ecreact.READY.json"
    ledger=STUDY41/"inputs/sources/ecreact.rows.jsonl"
    closure=read(closure_path)
    plan=read(plan_path)
    assert closure["phase"]=="complete" and closure["labels_read"] is False
    assert sha(plan_path)==closure["input_plan_sha256"]
    assert plan["mapper"]["commit"]==MAPPER_REVISION and CHECKPOINT in plan["mapper"]["checkpoint_sha256"].values()
    assert sha(ledger)==closure["artifacts_sha256"][str(ledger)]
    for path in (plan_path,closure_path,ledger):
        pins[str(path)]=sha(path)
    p22=BASE/"all_endpoint_factorial_full_v1/plan.json"
    assert sha(p22)==read(p22.with_name("plan_seal.json"))["sha256"]
    pins[str(p22)]=sha(p22)
    old=read(p22)
    s=old["sources"]["ecreact"]
    table=Path(s["table"])
    assert sha(table)==old["pinned_files"][str(table)]
    pins[str(table)]=sha(table)
    inputs=pd.read_csv(table,usecols=["source_position","row_id","rxn"],dtype={"row_id":str})
    archive=Path(s["features"]["synrfp-wl-r2-b2048-parity-delta_union"])
    feature_receipt=Path(s["feature_receipt"])
    assert sha(archive)==read(feature_receipt)["files"][archive.name]
    for path in (archive,feature_receipt):
        pins[str(path)]=sha(path)
    with np.load(archive,allow_pickle=False) as f:
        np.testing.assert_array_equal(f["row_ids"],inputs.row_id)
        np.testing.assert_array_equal(f["source_positions"],inputs.source_position)
        feature_ok=f["ok"].copy()
    with ledger.open() as f:
        def verified_rows():
            for line,record in zip(f,inputs.itertuples(index=False),strict=True):
                r=json.loads(line)
                assert (r["source_position"],r["row_id"],r["reaction"])==(record.source_position,record.row_id,record.rxn)
                yield r
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            result=list(pool.map(reuse,verified_rows(),chunksize=192))
    assert len(result)==185734 and len(feature_ok)==len(result)
    print("Strict mapping labels",sum(bool(row[4]) for row in result),flush=True)
    full,labels,em_refused= enzyme_sets(pins)
    ledger_rows=[]
    agreement=Counter()
    refusals=Counter()
    for (i,d,r,g,label,refusal,origin),ok in zip(result,feature_ok,strict=True):
        direct=d in full
        reverse=r in full
        known=labels.get(d,set())
        if label and known:
            agreement["compared"]+=1
            agreement["agree"]+=known=={label}
            agreement["discordant_unambiguous"]+=len(known)==1 and known!={label}
            agreement["ambiguous_reference"]+=len(known)>1
        ledger_rows.append(dict(source_position=i,direction=d or "",reversed=r or "",scaffold=g,label=label,
                                refusal=refusal,mapping_origin=origin,feature_ok=bool(ok),enzyme_forward=direct,
                                enzyme_reverse=reverse,matched_validated_enzymemap=bool(known)))
        if refusal:
            refusals[refusal]+=1
    frame=pd.DataFrame(ledger_rows)
    final=ROOT/"mapped_labels.csv.gz"
    frame.to_csv(final,index=False)
    group_columns={"all_source":np.ones(len(frame),bool), "fingerprint_encoded":feature_ok,
                   "mapped_edit":frame.label.ne(""),"paired_eligible":frame.label.ne("")&feature_ok,
                   "paired_eligible_no_enzymemap":frame.label.ne("")&feature_ok&~(frame.enzyme_forward|frame.enzyme_reverse)}
    rows=[]
    for subset, mask in group_columns.items():
        domain=frame.loc[mask]
        rows.append(dict(population=subset,rows=len(domain),exact_direction=int(domain.enzyme_forward.sum()),
                         reverse_only=int((~domain.enzyme_forward&domain.enzyme_reverse).sum()),
                         either_direction=int((domain.enzyme_forward|domain.enzyme_reverse).sum()),
                         exact_direction_with_validated_enzymemap_label=int(domain.matched_validated_enzymemap.sum())))
    pd.DataFrame(rows).to_csv(OUT/"overlap_with_enzymemap.csv",index=False)
    mapping_rows=[dict(metric="source_rows",count=len(frame)),
                  dict(metric="accepted_by_original_mapper_archive",count=closure["status_counts"]["ok"]),
                  dict(metric="validated_atom_conserving_edit_label",count=int(frame.label.ne("").sum())),
                  dict(metric="eligible_with_encoded_fingerprint",count=int(group_columns["paired_eligible"].sum()))]
    mapping_rows.extend(dict(metric=f"refusal::{reason}",count=n) for reason,n in refusals.most_common())
    mapping_rows.extend(dict(metric=f"enzymemap_overlap::{name}",count=agreement[name])
                        for name in ("compared","agree","discordant_unambiguous","ambiguous_reference"))
    pd.DataFrame(mapping_rows).to_csv(OUT/"mapping_quality.csv",index=False)
    write(ROOT/"MAPPING_COMPLETE.json",dict(rows=len(frame),validated=int(frame.label.ne("").sum()),
         source_mapper=MAPPER_REVISION,checkpoint_sha256=CHECKPOINT,profile="atom_conserving_bond_type_multiset_v1",
         overlap_agreement=dict(agreement),enzymemap_source_parse_refusals=em_refused,
         audit_sha256=sha(Path(__file__)),inputs_sha256=pins,labels_sha256=sha(final),
         outputs_sha256={name:sha(OUT/name) for name in ("mapping_quality.csv","overlap_with_enzymemap.csv")}))
    print("Mapping and overlap audit complete",rows,dict(agreement),flush=True)


if __name__=="__main__":
    main()
