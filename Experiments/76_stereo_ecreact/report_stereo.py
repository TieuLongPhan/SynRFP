"""Audit the ECReact count ceilings and the 15 frozen rescued cases."""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from rdkit import Chem

from encode_stereo import (
    HERE, REPO, ROOT, PROFILES, CHANNELS, check, digest, inputs, read, write,
)

ENDPOINTS=("ecreact_1","ecreact_2","ecreact_3")
EMPTY=sha256(b"").digest()
NATIVE=ROOT/"default_optimal_benchmark_v1"
CASES=REPO.parent/"SynRFP-chemistry-cases"
OLD_REPO=REPO.parent/"SynRFP-delta-ceiling"
CASE_COMMIT="9e2c2de9c38d7ad342bbd86e744d5684b3024e2f"
CASE_PATH="Experiments/75_chemistry_cases/publication_metadata/chemistry_cases_v1"
CEILING_PATH="Experiments/74_delta_ceiling/publication_metadata/delta_ceiling_v1"


def export(path, rows):
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def distribution(keys, labels):
    _, index = np.unique(keys,return_inverse=True)
    sizes=np.bincount(index)
    n_groups=len(sizes)
    n_labels=int(labels.max())+1
    pair,count=np.unique(index*n_labels+labels,return_counts=True)
    cardinality=np.bincount(pair//n_labels,minlength=n_groups)
    major=np.zeros(n_groups,dtype=np.int64)
    np.maximum.at(major,pair//n_labels,count)
    mixed=int(sizes[cardinality>1].sum())
    return {"n_exact_classes":n_groups,"n_singleton_classes":int((sizes==1).sum()),
            "n_mixed_label_rows":mixed,"collision_share":mixed/len(labels),
            "oracle_majority_count":int(major.sum()),"oracle_ceiling":int(major.sum())/len(labels)}


def inspect_empty(keys, labels):
    mask=keys==np.void(EMPTY)
    n=int(mask.sum())
    if n:
        _, counts=np.unique(labels[mask],return_counts=True)
        distinct=len(counts)
        majority=int(counts.max())
    else:
        distinct=0
        majority=None
    return {"n_empty":n,"empty_share":n/len(keys),"n_labels_in_empty":distinct,
            "empty_majority_count":majority,"empty_label_purity":majority/n if n else None}


def constitutional_key(smiles, erase):
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Case endpoint side could not be canonicalized")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    if erase:
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol,canonical=True,isomericSmiles=not erase)


def cases(cohort, keys, ok, pins):
    raw=subprocess.check_output(["git","-C",str(CASES),"show",f"{CASE_COMMIT}:{CASE_PATH}/case_a.csv"])
    completion=subprocess.check_output(["git","-C",str(CASES),"show",f"{CASE_COMMIT}:{CASE_PATH}/COMPLETE.json"])
    case_sha=sha256(raw).hexdigest()
    if case_sha!=json.loads(completion)["files_sha256"]["case_a.csv"]:
        raise ValueError("Request 75 case bytes changed")
    pins["request75_case_a_sha256"]=case_sha
    pins["request75_complete_sha256"]=sha256(completion).hexdigest()
    found=[]
    for row in csv.DictReader(raw.decode().splitlines()):
        if row["endpoint"] not in ENDPOINTS:
            continue
        left,_,right=row["rxn"].split(">")
        if constitutional_key(left,True)!=constitutional_key(right,True):
            continue
        assert constitutional_key(left,False)!=constitutional_key(right,False)
        source,partner=int(row["source_position"]),int(row["collision_source_position"])
        assert bool(ok[source] and ok[partner])
        assert cohort.row_id.iloc[source]==row["r_id"] and cohort.rxn.iloc[source]==row["rxn"]
        assert cohort.row_id.iloc[partner]==row["collision_r_id"]
        assert cohort.rxn.iloc[partner]==row["collision_rxn"]
        assert bytes(keys[source,0,0])==bytes(keys[partner,0,0])==EMPTY
        new={"case_id":row["case_id"],"endpoint":row["endpoint"],"source_position":source,
             "r_id":row["r_id"],"rxn":row["rxn"],"true_label":row["true_label"],
             "collision_r_id":row["collision_r_id"],"collision_source_position":partner,
             "collision_label":row["collision_label"],"ordinary_D_empty":True,
             "ordinary_D_equal_collision_partner":True,"isomeric_sides_different":True}
        for j,profile in enumerate(PROFILES[1:],start=1):
            new[f"{profile}_D_nonempty"]=bytes(keys[source,j,0])!=EMPTY
            new[f"{profile}_D_separates_collision_partner"]=keys[source,j,0]!=keys[partner,j,0]
        found.append(new)
    n=pd.Series([r["endpoint"] for r in found]).value_counts().to_dict()
    assert n=={"ecreact_2":7,"ecreact_3":8}
    return found


def markdown(summary, empty_rows, rescued):
    lines=[
        "# ECReact stereo-aware signed counts","",
        "All profiles use WL radius 2 and exact occurrence counts on the study 65",
        "cohort. Fourteen previously refused ECReact source rows remain outside the",
        "analysis. The ordinary D, U and D+U keys match Request 74 on every encoded",
        "row. Oracle accuracy is the majority-label accuracy over exact classes.","",
        "| Endpoint | Profile | D ceiling | U ceiling | D+U ceiling | D+U minus D (pp) | D empty | Purity of empty D |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_key={(r["endpoint"],r["profile"],r["channel_set"]):r for r in summary}
    empties={(r["endpoint"],r["profile"]):r for r in empty_rows}
    for name in ENDPOINTS:
        for profile in PROFILES:
            d,u,du=[by_key[name,profile,channel]["oracle_ceiling"] for channel in CHANNELS]
            empty=empties[name,profile]
            purity = "—" if empty["empty_label_purity"] is None else f"{empty['empty_label_purity']:.3%}"
            lines.append(f"| {name} | {profile} | {d:.4%} | {u:.4%} | {du:.4%} | {100*(du-d):.4f} | "
                         f"{empty['empty_share']:.3%} | {purity} |")
    lines.extend(["","| Stereo profile | Nonempty D among 15 | D separates the saved collision partner |",
                  "| --- | ---: | ---: |"])
    for profile in PROFILES[1:]:
        n=sum(r[f"{profile}_D_nonempty"] for r in rescued)
        separated=sum(r[f"{profile}_D_separates_collision_partner"] for r in rescued)
        lines.append(f"| {profile} | {n}/15 | {separated}/15 |")
    lines.extend([
        "", "The 15 pairs were selected in Request 75 from saved original-model",
        "outcomes. The new checks compare representations and empirical label",
        "ceilings. Each stereo digest is rooted in its connected molecule.",
        "The additional atom or bond label can carry configuration elsewhere",
        "in the rooted molecular certificate. See [token profiles](token_profiles.md).",
        "", "See [endpoint_summary.csv](endpoint_summary.csv),",
        "[empty_delta_summary.csv](empty_delta_summary.csv) and",
        "[stereo_only_rescues.csv](stereo_only_rescues.csv).",
    ])
    return "\n".join(lines)+"\n"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoding",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    rec=read(args.encoding/"ENCODED.json")
    manifest_path=args.encoding/"ENCODING_PLAN.json"
    check(manifest_path,rec["encoding_plan_sha256"])
    manifest=read(manifest_path)
    cohort,mask,baseline,source,_=inputs()
    with np.load(args.encoding/"ec_keys.npz",allow_pickle=False) as saved:
        check(args.encoding/"ec_keys.npz",rec["artifact_sha256"])
        np.testing.assert_array_equal(saved["row_ids"],cohort.row_id)
        np.testing.assert_array_equal(saved["ok"],mask)
        keys=saved["keys"]
    np.testing.assert_array_equal(keys[mask,0],baseline[mask])
    plan_path=NATIVE/"plan.json"
    check(plan_path,read(NATIVE/"plan_seal.json")["sha256"])
    native=read(plan_path)
    label_path=NATIVE/"native/classification/plan.json"
    check(label_path,native["inputs_sha256"][str(label_path)])
    plan=read(label_path)
    release=OLD_REPO/CEILING_PATH
    ref_prior=read(release/"COMPLETE.json")
    check(release/"endpoint_summary.csv",ref_prior["files_sha256"]["endpoint_summary.csv"])
    old=pd.read_csv(release/"endpoint_summary.csv")
    summary,empty_rows=[],[]
    pins={"study65_plan":digest(plan_path),"classification_plan":digest(label_path),
          "source_table":digest(source),"request74_summary":digest(release/"endpoint_summary.csv")}
    for endpoint in ENDPOINTS:
        endpoint_plan=plan["panels"][endpoint]
        pins[endpoint]=check(endpoint_plan["table"],endpoint_plan["table_sha256"])
        panel=pd.read_csv(endpoint_plan["table"],dtype={"row_id":str})
        pos=panel.source_position.to_numpy()
        np.testing.assert_array_equal(panel.row_id,cohort.row_id.to_numpy()[pos])
        ok=mask[pos]
        n=int(ok.sum())
        views=keys[pos[ok]]
        y=np.unique(panel.y.to_numpy()[ok],return_inverse=True)[1]
        for j,profile in enumerate(PROFILES):
            base={"endpoint":endpoint,"profile":profile,"tokenizer":"wl_r2",
                  "n_reactions":n,"study65_rows":len(panel),"encoding_refusals":len(panel)-n}
            empty_rows.append({**base,**inspect_empty(views[:,j,0],y)})
            for i,channel in enumerate(CHANNELS):
                row={**base,"channel_set":channel,**distribution(views[:,j,i],y)}
                if j==0:
                    expected=old[(old.endpoint==endpoint)&(old.tokenizer=="wl_r2")&(old.channel_set==channel)].iloc[0]
                    assert row["n_exact_classes"]==int(expected.n_groups)
                    np.testing.assert_allclose([row["collision_share"],row["oracle_ceiling"]],
                                               [expected.collision_share,expected.ceiling],atol=1e-15,rtol=0)
                summary.append(row)
    rescued=cases(cohort,keys,mask,pins)
    folder=args.output
    folder.mkdir(parents=True,exist_ok=True)
    export(folder/"endpoint_summary.csv",summary)
    export(folder/"empty_delta_summary.csv",empty_rows)
    export(folder/"stereo_only_rescues.csv",rescued)
    (folder/"README.md").write_text(markdown(summary,empty_rows,rescued))
    audit={"encoding":manifest,"encoding_receipt":rec,"input_sha256":pins,
           "reporter_sha256":digest(Path(__file__)),"tokenizer_profiles":PROFILES,
           "verified_baseline_report":True,"archived_case_counts":{"ecreact_2":7,"ecreact_3":8},
           "reporter_checkout":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
           "model_fits":0}
    write(folder/"AUDIT.json",audit)
    write(folder/"COMPLETE.json",{"status":"complete","endpoint_profile_channel_rows":len(summary),
                                  "empty_delta_rows":len(empty_rows),"case_rows":len(rescued),
                                  "files_sha256":{p.name:digest(p) for p in sorted(folder.iterdir())
                                                   if p.is_file() and p.name!="COMPLETE.json"}})
    print("Complete",len(summary),"channel rows,",len(rescued),"stereo-only cases",flush=True)


if __name__=="__main__":
    main()
