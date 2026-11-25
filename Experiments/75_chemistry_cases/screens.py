"""Extract the pre-existing matched yield assays and selected alcohol rankings."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from io import BytesIO
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from rdkit import Chem

from common import WORKSPACE, PAPER, PAPER_REV, OUTPUT, LOCAL, InputAudit, chemistry, export, read, write


def bh(audit):
    run=WORKSPACE/"SynRFP-exact-delta-e3/Experiments/work/context_attribution/E3-20261124T080523Z"
    status=read(audit.use(run/"status.json"))
    assert status["phase"]=="complete" and status["fits"]==27
    manifest=read(audit.use(run/"manifest.json",status["outputs_sha256"]["manifest.json"]))
    path=WORKSPACE/"SynRFP-context-e1/Experiments/work/context_attribution/e1-inputs/bh_classic.csv.gz"
    actual=pd.read_csv(audit.use(path,manifest["source_data"]["sha256"]),keep_default_na=False)
    assert len(actual)==manifest["source_data"]["n_rows"] and actual.sample_id.is_unique
    for file in ("predictions.csv.gz","splits.csv.gz"):
        audit.use(run/file,status["local_artifact_sha256"][file])
    saved=pd.read_csv(run/"predictions.csv.gz")
    split=pd.read_csv(run/"splits.csv.gz")
    scores=pd.read_csv(audit.use(run/"pair_gap_scores.csv",status["outputs_sha256"]["pair_gap_scores.csv"]))
    assert set(saved.arm)=={"d_pooled","d_rehash","d2048"}
    recipes=["ligand_smiles","additive_smiles","base_smiles"]
    rows=[]
    pairs=0
    assigned={}
    for holdout in manifest["holdouts"]:
        key=holdout["split"]
        now=saved.loc[saved.split==key]
        test=split.loc[(split.split==key)&(split.partition=="test"),["sample_id","group"]]
        for arm,block in now.groupby("arm"):
            pd.testing.assert_frame_equal(block[["sample_id","group"]].reset_index(drop=True),test.reset_index(drop=True))
        predicted=now.pivot(index=["sample_id","group","observed_yield"],columns="arm",values="predicted_yield").reset_index()
        data=predicted.merge(actual,on="sample_id",validate="one_to_one")
        np.testing.assert_allclose(data["observed_yield"],data["yield_pct"],rtol=0,atol=1e-12)
        assert len(data)==holdout["n_test"] and data["group"].nunique()==2
        symbols={a.GetSymbol() for rxn in data["group"].unique() for a in Chem.MolFromSmiles(rxn).GetAtoms()
                 if a.GetSymbol() in ("Cl","Br","I")}
        assert len(symbols)==1
        group={"Cl":"aryl chlorides","Br":"aryl bromides","I":"aryl iodides"}[symbols.pop()]
        assert assigned.setdefault(holdout["delta_class"],group)==group
        n_matched=0
        y_gap,p_gap=[],[]
        for recipe, assay in data.groupby(recipes,sort=True,dropna=False):
            if len(assay)!=2:
                if len(assay)!=1:
                    raise ValueError("Unexpected duplicate recipe in E3")
                continue
            assay=assay.sort_values("group")
            assert assay["group"].nunique()==2
            np.testing.assert_allclose(assay["d_rehash"].iloc[0],assay["d_rehash"].iloc[1],rtol=0,atol=1e-5)
            n_matched+=1
            y_gap.append(float(np.diff(assay["observed_yield"])[0]))
            p_gap.append(float(np.diff(assay["d_pooled"])[0]))
            for _, row in assay.iterrows():
                rows.append({"class":group,"delta_class":holdout["delta_class"],"holdout":key,"recipe_id":f"{key}:{n_matched:03d}",
                             "sample_id":row.sample_id,"aryl_halide":row.aryl_halide_smiles,
                             "ligand":recipe[0],"additive":recipe[1],"base":recipe[2],
                             "measured_yield":row.observed_yield,"pred_zero_gap_arm":row.d_rehash,
                             "pred_pooled_arm":row.d_pooled,"pred_standard_delta":row.d2048})
        assert n_matched==holdout["matched_recipes"]
        ref=scores.loc[(scores.split==key)&(scores.arm=="d_pooled")].iloc[0]
        np.testing.assert_allclose(np.mean((np.asarray(y_gap)-p_gap)**2),ref.gap_mse,atol=1e-10,rtol=0)
        pairs+=n_matched
    assert len(assigned)==3 and pairs==2368
    export(OUTPUT/"case_d.csv",rows)
    return {"holdouts":len(manifest["holdouts"]),"assay_pairs":pairs,"exported":len(rows),
            "delta_classes":assigned,"assays_with_saved_predictions":len(saved)//3,
            "zero_gap_arm":"d_rehash","inputs_sha256":audit.pins}


def alcohol():
    paths=["data/condition_ranking/predictions.csv","data/condition_ranking/substrate_regret.csv",
           "data/condition_ranking/metadata.json"]
    data={path:subprocess.check_output(["git","-C",str(PAPER),"show",f"{PAPER_REV}:{path}"])
          for path in paths}
    pred,regret=[pd.read_csv(BytesIO(data[path])) for path in paths[:2]]
    receipt=json.loads(data[paths[2]])
    for path in paths[:2]:
        assert sha256(data[path]).hexdigest()==receipt["sha256"][path]
    perf=regret.pivot(index="substrate",columns="method",values="regret")
    gain=(perf["delta2048"]-perf["delta_pooled"]).to_dict()
    assert len(gain)==receipt["substrates"] and len(pred)==receipt["eligible_rows"]
    assert set(pred.groupby("substrate").size())=={20}
    diversity=set()
    cases=[]
    tally=Counter()
    for sign in (1,-1):
        mode="gain" if sign==1 else "loss"
        passing=[s for s,v in gain.items() if sign*v>0]
        tally[f"eligible_{mode}"]=len(passing)
        nominees=sorted(passing,key=lambda s:(-sign*gain[s],chemistry(s)[0],s))
        selected=0
        for substrate in nominees:
            _,scaffold=chemistry(substrate)
            if scaffold in diversity:
                continue
            diversity.add(scaffold)
            selected+=1
            panel=pred.loc[pred.substrate==substrate].copy()
            for arm, column in (("delta2048","rank_D"),("delta_pooled","rank_DU")):
                order=panel.sort_values([arm,"base","fluoride"],ascending=[False,True,True])
                panel[column]=pd.Series(np.arange(1,21),index=order.index)
                observed=panel.loc[panel[column]<=3,"yield"].max()
                original=regret.loc[(regret.substrate==substrate)&(regret.method==arm),"top3_best"].item()
                assert observed==original
            for _, row in panel.sort_values(["base","fluoride"]).iterrows():
                cases.append({"direction":mode,"case_id":f"E_{mode}_{selected:02d}",
                              "substrate":substrate,"scaffold":scaffold,"intended_product":row.intended_product,
                              "base":row.base,"fluoride":row.fluoride,"measured_yield":row["yield"],
                              "pred_D":row.delta2048,"pred_DU":row.delta_pooled,
                              "rank_D":int(row.rank_D),"rank_DU":int(row.rank_DU),
                              "top3_gain_pp":float(gain[substrate]),"source_index":int(row.source_index)})
            if selected==3:
                break
        tally[f"exported_{mode}"]=selected
    export(OUTPUT/"case_e.csv",cases)
    return {"counts":dict(tally),"exported":len(cases),"eligible_substrates":len(gain),
            "paper_git_ref":PAPER_REV,"paper_file_sha256":{key:sha256(value).hexdigest() for key,value in data.items()}}


if __name__=="__main__":
    LOCAL.mkdir(parents=True,exist_ok=True)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write(LOCAL/"D_COMPLETE.json",bh(InputAudit()))
    write(LOCAL/"E_COMPLETE.json",alcohol())
