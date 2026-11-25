"""Select held-out EnzymeMap and QuantumPioneer examples from saved predictions."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from common import BASE, WORKSPACE, LOCAL, OUTPUT, InputAudit, chemistry, export, read, write

SEEDS=(17,29,43)
ARMS=("d2048","d_pooled")


def nominees(ids,texts,paired,losses=None):
    """Give gain/loss up to five cases each, removing repeats and repeated scaffolds."""
    output=[];used=set();counts={}
    for sign in (1,-1):
        direction="gain" if sign==1 else "loss"
        votes=(sign*paired>0).sum(axis=0)
        ok=votes>=2
        if losses is not None:
            ok&=sign*losses>0
        eligible=np.flatnonzero(ok)
        counts["eligible_"+direction]=len(eligible)
        if losses is None:
            # Heavy-atom tie break is applied before the stable ID.
            contenders=sorted(eligible,key=lambda i:(-int(votes[i]),chemistry(texts[i])[0],ids[i]))
        else:
            contenders=sorted(eligible,key=lambda i:(-sign*losses[i],-int(votes[i]),chemistry(texts[i])[0],ids[i]))
        n=0
        for i in contenders:
            heavy,scaffold=chemistry(texts[i])
            if scaffold in used:
                continue
            n+=1
            used.add(scaffold)
            j=int(np.flatnonzero(sign*paired[:,i]>0)[0])
            output.append((direction,n,i,j,int(votes[i]),heavy,scaffold))
            if n==5:
                break
        counts["exported_"+direction]=n
    return output,counts


def enzymes(audit):
    root=WORKSPACE/"SynRFP-enzymemap-precheck/Experiments/work/74_context_dependence"
    source=WORKSPACE/"SynRFP-enzymemap-precheck/Experiments/74_context_dependence/inputs/enzymemap_v2_brenda2023.csv.gz"
    result=root/"enzymemap-replication-v1"
    status=read(audit.use(result/"status.json"))
    assert status["phase"]=="complete" and status["verified_fits"]==30
    plan=read(audit.use(result/"plan.json",status["outputs_sha256"]["plan.json"]))
    proof=read(audit.use(result/"fit_receipts.json",status["outputs_sha256"]["fit_receipts.json"]))
    fits={(r["target"],r["arm"],r["seed"]):r for r in proof}
    prior=root/"enzymemap-precheck-v1"
    eligible=read(audit.use(prior/"eligibility_precheck.json"))
    data=pd.read_csv(audit.use(prior/"cohort.csv.gz",eligible["outputs_sha256"]["cohort.csv.gz"]),dtype=str)
    raw=pd.read_csv(audit.use(source,plan["source_sha256"]),usecols=["rxn_idx","ec_num"],dtype=str)
    alphabet=re.compile(r"(?:EC\s*)?([1-9]\d*\.[1-9]\d*)(?:\.[^.]+){0,2}")
    def rows(split):
        path=root/f"enzymemap-labels-{split}-v1"
        receipt=read(audit.use(path/"COMPLETE.json",plan["map_receipt_sha256"][split]))
        mapped=read(audit.use(path/f"{split}_labels.json",receipt["labels_sha256"]))
        input_rows=data.loc[data.split==split]
        assert input_rows.row_id.tolist()==[r["row_id"] for r in mapped]
        chosen=input_rows.iloc[[i for i,r in enumerate(mapped) if "label" in r]].copy()
        assert len(chosen)==plan["partition_admitted"][split]
        position=chosen.representative_source_index.astype(int)
        source_subset=raw.iloc[position]
        for r,ids in zip(source_subset.rxn_idx,chosen.source_ids,strict=True):
            assert r in json.loads(ids)
        chosen["ec"]=[alphabet.fullmatch(value.strip())[1] for value in source_subset.ec_num]
        chosen["bond"]=[r["label"] for r in mapped if "label" in r]
        return chosen
    train,test=rows("train"),rows("test")
    start=len(train)+plan["partition_admitted"]["validation"]
    row_index=np.arange(start,start+len(test))
    predictions={}
    for target,field in (("ec_level_2","ec"),("bond_edit_multiset","bond")):
        alphabet=np.unique(train[field].to_numpy(str))
        for arm in ARMS:
            v=[]
            for seed in SEEDS:
                r=fits[target,arm,seed]
                assert r["plan_sha256"]==audit.pins[str(result/"plan.json")]
                path=result/"fits"/f"{target}__{arm}__s{seed}"/"blind_predictions.npz"
                with np.load(audit.use(path,r["outputs_sha256"][path.name]),allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["matrix_row"],row_index)
                    v.append(alphabet[saved["prediction"]])
            predictions[target,arm]=np.asarray(v)
    labels=test.ec.to_numpy(str)
    D,DU=(predictions["ec_level_2",arm] for arm in ARMS)
    achieved=(DU==labels).astype(int)-(D==labels)
    choices,counts=nominees(test.row_id.to_numpy(str),test.reaction_smiles.to_numpy(str),achieved)
    output=[]
    for mode,n,i,j,votes,heavy,scaffold in choices:
        row=test.iloc[i]
        output.append({"dataset":"enzymemap","direction":mode,"case_id":f"F_enzymemap_{mode}_{n:02d}",
                       "r_id":row.row_id,"reaction":row.reaction_smiles,"n_heavy_atoms":heavy,"scaffold":scaffold,
                       "paired_seed_votes":votes,"n_paired_seeds":len(SEEDS),"display_seed":SEEDS[j],
                       "true_ec_label":row.ec,"pred_D_ec":D[j,i],"pred_DU_ec":DU[j,i],
                       "true_bond_edit_label":row.bond,
                       **{f"pred_{arm}_bond_edit":predictions["bond_edit_multiset",key][j,i]
                          for arm,key in zip(("D","DU"),ARMS,strict=True)}})
    return output,{"rows":len(test),"counts":counts,"selection_target":"ec_level_2"}


def pioneer(audit):
    base=BASE/"context_dependence_d421d7ae"
    root=base/"qp_primary_v1"
    status=read(audit.use(root/"status.json"))
    assert status["phase"]=="complete" and status["fits"]==30
    manifest=read(audit.use(root/"manifest.json",status["outputs_sha256"]["manifest.json"]))
    folder=base/"qp_frozen_v1"
    frozen=read(audit.use(folder/"prediction.json",manifest["frozen_prediction_sha256"]))
    with np.load(audit.use(folder/"cohort.npz",frozen["artifact_sha256"]["cohort.npz"]),allow_pickle=False) as saved:
        positions,ids,train=saved["test_indices"],saved["test_ids"],saved["train_targets"]
    source=base/"sources/quantumpioneer_kinetics.csv"
    targets=("barrier","deltaHrxn")
    raw=pd.read_csv(audit.use(source,frozen["source_sha256"]),usecols=["source_record_id","proposed_split","reaction_smiles",*targets])
    test=raw.iloc[positions]
    np.testing.assert_array_equal(test.source_record_id,ids)
    assert (test.proposed_split=="test").all()
    y=test[list(targets)].to_numpy()
    pred={}
    for key in targets:
        for arm in ARMS:
            p=[]
            for seed in SEEDS:
                cell=root/"cells"/f"{key}__{arm}__s{seed}"
                receipt=read(audit.use(cell/"COMPLETE.json"))
                assert receipt["plan_sha256"]==audit.pins[str(root/"manifest.json")]
                path=cell/"blind_predictions.npz"
                with np.load(audit.use(path,receipt["outputs_sha256"][path.name]),allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved["matrix_row"],positions)
                    p.append(saved["prediction"])
            pred[key,arm]=np.asarray(p)
    primary=frozen["predicted_higher_gain_target"]
    idx=targets.index(primary)
    d,du=(pred[primary,arm] for arm in ARMS)
    losses=((d-y[:,idx])**2-(du-y[:,idx])**2)/train[:,idx].var()
    output=[]
    choices,counts=nominees(ids,test.reaction_smiles.to_numpy(str),losses,losses.mean(axis=0))
    for mode,n,i,j,votes,heavy,scaffold in choices:
        row=test.iloc[i]
        record={"dataset":"quantumpioneer","direction":mode,"case_id":f"F_quantumpioneer_{mode}_{n:02d}",
                "r_id":ids[i],"reaction":row.reaction_smiles,"n_heavy_atoms":heavy,"scaffold":scaffold,
                "paired_seed_votes":votes,"n_paired_seeds":len(SEEDS),"selection_target":primary,
                "mean_normalized_squared_error_gain":float(losses[:,i].mean()),"units":"kcal/mol"}
        for label,key in (("barrier","barrier"),("enthalpy","deltaHrxn")):
            record[f"true_{label}"]=float(row[key])
            for short,arm in zip(("D","DU"),ARMS,strict=True):
                v=pred[key,arm][:,i]
                record[f"pred_{short}_{label}"]=float(v.mean())
                record[f"pred_{short}_{label}_by_seed"]=json.dumps(v.tolist(),separators=(",",":"))
        output.append(record)
    return output,{"rows":len(test),"counts":counts,"selection_target":primary}


if __name__=="__main__":
    LOCAL.mkdir(parents=True,exist_ok=True)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    audit=InputAudit()
    a,ac=enzymes(audit)
    b,bc=pioneer(audit)
    export(OUTPUT/"case_f.csv",a+b)
    write(LOCAL/"F_COMPLETE.json",{"enzymemap":ac,"quantumpioneer":bc,"exported":len(a+b),
                                   "inputs_sha256":audit.pins})
