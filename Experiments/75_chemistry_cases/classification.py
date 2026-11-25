"""Select classification successes, failures and exact-vs-folded witnesses."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv

import numpy as np

from common import (
    BASE, HERE, LOCAL, OUTPUT, STUDY65, OLD_NATIVE, WORKSPACE, InputAudit,
    Buckets, chemistry, export, feature, groups, load_plan, packed_ids,
    panel_data, read, source_data, write,
)

A=("ecreact_1","ecreact_2","ecreact_3","tpl_b","schneider_b")
B=("mech_1","mech_2","syntemp_r1","syntemp_r2","uspto50k_b","uspto50k_ub")
CASES={"A":A,"B":B}
MODE={"synrfp_default":"synrfp-wl-r2-b2048-parity-delta",
      "synrfp_optimal":"synrfp-wl-r2-b2048-parity-delta_union"}
STATES={}
POPCOUNT=np.array([n.bit_count() for n in range(256)],dtype=np.int16)
DISPLAY_SEEDS=(1,0,*range(2,25))
FIELDS=("endpoint","case_id","r_id","rxn","true_label","pred_D","pred_DU","n_partitions",
        "n_D_wrong_DU_right","n_D_right_DU_wrong","collision_r_id","collision_rxn",
        "collision_label","n_heavy_atoms","source_position","display_seed","witness_seed",
        "scaffold","collision_source_position","pooled_context_tanimoto")
B_FIELDS=(*FIELDS,"n_classes_merged_1024","n_classes_merged_2048","n_classes_merged_only_1024",
          "merged_r_id","merged_rxn","merged_label","merged_source_position")


def artifact(audit, result, suffix):
    match=[(k,v) for k,v in result["artifacts_sha256"].items() if k.rsplit("/",1)[-1]==suffix]
    if len(match)!=1:
        raise ValueError(f"Expected saved {suffix}")
    filename, digest=match[0]
    path=result.get("attempt_directory")
    if "/" not in filename and path:
        filename=f"{path}/{filename}"
    return audit.use(OLD_NATIVE/filename,digest)


def comparison_inputs(audit, endpoint, seed, panel, data, native_rows):
    code_path=audit.use(panel["split_paths"][str(seed)],panel["split_sha256"][str(seed)])
    code=np.load(code_path,allow_pickle=False)
    test=np.flatnonzero(code==2)
    outer=np.r_[np.flatnonzero(code==0),np.flatnonzero(code==1)]
    values, fits=[],[]
    for method, config in MODE.items():
        cell=native_rows[endpoint,seed,method]
        if cell["configuration"]!=config:
            raise ValueError("Unexpected preset")
        result=read(audit.use(cell["source_result_path"],cell["source_result_sha256"]))
        split_hash=result.get("split_sha256") or result["identity"]["split_sha256"]
        if split_hash!=panel["split_sha256"][str(seed)]:
            raise ValueError("Archived seed differs from frozen fold")
        with np.load(artifact(audit,result,"test_predictions.npz"),allow_pickle=False) as z:
            if "observed" in z.files:
                observed,zpos,zid=z["observed"],z["source_positions"],z["row_ids"]
            else:
                observed,zpos,zid=z["y"],z["source_row"],z["r_id"]
            np.testing.assert_array_equal(zpos,data.source_position.to_numpy()[test])
            np.testing.assert_array_equal(zid.astype(str),data.row_id.to_numpy(str)[test])
            np.testing.assert_array_equal(observed,data.y.to_numpy()[test])
            pred=z["predictions"].astype(np.int32)
        with np.load(artifact(audit,result,"state.npz"),allow_pickle=False) as z:
            train,val=z["train"],z["val"]
        np.testing.assert_array_equal(np.sort(np.r_[train,val]),np.arange(len(outer)))
        fits.append(outer[train])
        values.append(pred)
    np.testing.assert_array_equal(fits[0],fits[1])
    return test,*values,fits[0]


def counts_for_bits(exact, folded):
    # A collision class is any exact key sharing a bit vector with another.
    relation=np.unique(np.column_stack((folded,exact)),axis=0)
    sizes=np.bincount(relation[:,0],minlength=int(folded.max())+1)
    collision=relation[sizes[relation[:,0]]>1,1]
    ambiguous=np.unique(collision)
    return sizes,len(ambiguous),len(np.unique(exact))


def jaccard(pool, i, candidates):
    ref=pool[candidates]
    inter=POPCOUNT[np.bitwise_and(ref,pool[i])].sum(axis=1)
    union=POPCOUNT[np.bitwise_or(ref,pool[i])].sum(axis=1)
    return np.divide(inter,union,out=np.ones(len(union)),where=union>0)


def collect(job):
    case, endpoint=job
    audit=InputAudit()
    panel=STATES["panels"][endpoint]
    source,key,heavy,valid=source_data(audit,panel["source"])
    data, vocab=panel_data(audit,endpoint,panel,source)
    pos=data.source_position.to_numpy()
    ok=valid[pos]
    ids=data.row_id.to_numpy(str)
    labels=data.y.to_numpy()
    n=len(data)
    # Keep indices in cohort positions, a repeated ECReact source ID is permitted.
    exact=groups(key[pos,0])
    bits_D=feature(audit,panel["source"],"delta",valid,source.row_id.to_numpy(str))[pos]
    bits_DU=feature(audit,panel["source"],"delta_union",valid,source.row_id.to_numpy(str))[pos]
    packed_small=bits_D[:,:128]^bits_D[:,128:]
    np.testing.assert_array_equal(packed_small,bits_DU[:,:128])
    n_parts=np.zeros(n,dtype=np.uint8)
    better=np.zeros(n,dtype=np.uint8)
    worse=np.zeros(n,dtype=np.uint8)
    support=np.zeros(n,dtype=np.uint32)
    fits=np.zeros((25,n),dtype=bool)
    saved={}
    for seed in range(25):
        test,d,du,fit=comparison_inputs(audit,endpoint,seed,panel,data,STATES["comparisons"])
        correct_D=d==labels[test]
        correct_DU=du==labels[test]
        good=~correct_D&correct_DU
        bad=correct_D&~correct_DU
        n_parts[test]+=1
        better[test]+=good
        worse[test]+=bad
        support[test[good if case=="A" else bad]]|=np.uint32(1<<seed)
        fits[seed,fit]=True
        saved[seed]=(test,d,du)
    np.testing.assert_array_equal(n_parts,np.full(n,5,dtype=np.uint8))
    votes=better if case=="A" else worse
    candidates=np.flatnonzero(ok&(votes*2>n_parts))
    rank=candidates[np.lexsort((pos[candidates],ids[candidates],heavy[pos[candidates]],-votes[candidates].astype(int)))]
    candidate_rows=[]
    info={"endpoint":endpoint,"case":case,"n_rows":n,"n_valid":int(ok.sum()),
          "vote_qualified":int((votes*2>n_parts).sum()),"encoded_vote_qualified":len(rank),
          "witness_qualified":0}
    fold_summary=[]
    exact_bucket=Buckets(exact)
    if case=="B":
        bits_u=feature(audit,panel["source"],"union",valid,source.row_id.to_numpy(str))[pos]
        bits_s=feature(audit,panel["source"],"roles",valid,source.row_id.to_numpy(str))[pos]
        np.testing.assert_array_equal(bits_u[:,:128]^bits_u[:,128:],bits_DU[:,128:])
        np.testing.assert_array_equal(bits_s[:,:128],packed_small)
        small_g, large_g=packed_ids(packed_small),packed_ids(bits_D)
        small_bucket=Buckets(small_g)
        merged_distinct={}
        for channel,width,encoded,col in (
            ("D",1024,packed_small,0),("D",2048,bits_D,0),
            ("U",1024,bits_DU[:,128:],1),("U",2048,bits_u,1),
            ("D+S",1024,np.concatenate((bits_s[:,:64]^bits_s[:,64:128],
                                       bits_s[:,128:192]^bits_s[:,192:]),axis=1),3),
            ("D+S",2048,bits_s,3),
        ):
            ids_exact=groups(key[pos[ok],col])
            fold=packed_ids(encoded[ok])
            _,n_collision,n_classes=counts_for_bits(ids_exact,fold)
            _,n_delta,n_exact_delta=counts_for_bits(groups(key[pos[ok],0]),fold)
            fold_summary.append({"endpoint":endpoint,"tokenizer":"wl_r2","seed":1,
                                 "channel_set":channel,"total_bits":width,"encoded_rows":int(ok.sum()),
                                 "n_exact_channel_classes":n_classes,"n_collision_classes":n_collision,
                                 "collision_class_share":n_collision/n_classes,
                                 "n_exact_delta_classes":n_exact_delta,"n_delta_classes_cross_exact_key":n_delta,
                                 "delta_class_collision_share":n_delta/n_exact_delta})
    for i in rank:
        seeds=[seed for seed in DISPLAY_SEEDS if int(support[i])&(1<<seed)]
        display=seeds[0];witness=-1;wseed=-1;similar=None
        if case=="A":
            candidates=exact_bucket.members(exact[i])
            candidates=candidates[(labels[candidates]!=labels[i])&ok[candidates]]
            candidates=candidates[fits[seeds][:,candidates].any(axis=0)]
            if not len(candidates):
                continue
            scores=jaccard(bits_DU[:,128:],i,candidates)
            sorted_candidates=np.lexsort((pos[candidates],ids[candidates],heavy[pos[candidates]],-scores))
            j=sorted_candidates[0]
            witness=int(candidates[j]);similar=float(scores[j])
            wseed=next(seed for seed in seeds if fits[seed,witness])
            display=wseed
        else:
            candidates=small_bucket.members(small_g[i])
            candidates=candidates[(exact[candidates]!=exact[i])&ok[candidates]]
            near=candidates[large_g[candidates]!=large_g[i]]
            n1024=int(len(np.unique(exact[candidates])))
            n2048=int(len(np.unique(exact[candidates[large_g[candidates]==large_g[i]]])))
            if len(near):
                trained=fits[seeds][:,near].any(axis=0)
                order=np.lexsort((pos[near],ids[near],heavy[pos[near]],
                                  (labels[near]==labels[i]),~trained))
                witness=int(near[order[0]])
                wseed=next((seed for seed in seeds if fits[seed,witness]),-1)
                display=wseed if wseed>=0 else display
        t,d,du=saved[display]
        hit=np.searchsorted(t,i)
        if t[hit]!=i:
            raise ValueError("Displayed seed is not held-out")
        row={"endpoint":endpoint,"r_id":ids[i],"rxn":source.rxn.iloc[pos[i]],"true_label":vocab[labels[i]],
             "pred_D":vocab[d[hit]],"pred_DU":vocab[du[hit]],"n_partitions":int(n_parts[i]),
             "n_D_wrong_DU_right":int(better[i]),"n_D_right_DU_wrong":int(worse[i]),
             "n_heavy_atoms":int(heavy[pos[i]]),"source_position":int(pos[i]),
             "display_seed":display,"witness_seed":wseed if wseed>=0 else None}
        if case=="A":
            row.update(collision_r_id=ids[witness],collision_rxn=source.rxn.iloc[pos[witness]],
                       collision_label=vocab[labels[witness]],collision_source_position=int(pos[witness]),
                       pooled_context_tanimoto=similar)
        else:
            row.update(n_classes_merged_1024=n1024,n_classes_merged_2048=n2048,n_classes_merged_only_1024=n1024-n2048)
            if witness>=0:
                row.update(merged_r_id=ids[witness],merged_rxn=source.rxn.iloc[pos[witness]],
                           merged_label=vocab[labels[witness]],merged_source_position=int(pos[witness]))
        candidate_rows.append(row)
    info["witness_qualified"]=len(candidate_rows)
    if case=="B":
        info["qualified_with_1024_only_collision"]=sum(row["n_classes_merged_only_1024"]>0 for row in candidate_rows)
    write(LOCAL/f"{endpoint}.json",{"candidates":candidate_rows,"stats":info,
                                   "fold_collisions":fold_summary,"inputs_sha256":audit.pins})
    print(endpoint,info,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs",type=int,default=6)
    args=parser.parse_args()
    LOCAL.mkdir(exist_ok=True)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    audit=InputAudit()
    STAGE=WORKSPACE/"SynRFP/Experiments/65_default_optimal/publication_metadata/default_optimal_v1/classification/cells.csv"
    with audit.use(STAGE).open(newline="") as handle:
        rows={(row["dataset"],int(float(row["seed"])),row["method"]):row for row in csv.DictReader(handle)
              if row["model"]=="mlp" and row["method"] in MODE}
    STATS=load_plan(audit)
    STATES.update(panels=STATS["panels"],comparisons=rows)
    work=[(case,endpoint) for case,keys in CASES.items() for endpoint in keys]
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(collect,work))
    final_stats, fold_collisions=[],[]
    for case,endpoints in CASES.items():
        exported=[];used=set()
        for endpoint in endpoints:
            item=read(LOCAL/f"{endpoint}.json")
            audit.pins.update(item["inputs_sha256"])
            fold_collisions.extend(item["fold_collisions"])
            info=item["stats"]
            accepted=0
            for row in item["candidates"]:
                _, scaffold=chemistry(row["rxn"])
                if scaffold in used:
                    continue
                used.add(scaffold)
                row["scaffold"]=scaffold
                accepted+=1
                row["case_id"]=f"{case}_{endpoint}_{accepted:02d}"
                exported.append(row)
                if accepted==12:
                    break
            info["exported"]=accepted
            final_stats.append(info)
        export(OUTPUT/(f"case_{case.lower()}.csv"),exported,FIELDS if case=="A" else B_FIELDS)
    export(OUTPUT/"fold_collision_summary.csv",fold_collisions)
    write(LOCAL/"classification_COMPLETE.json",{"counts":final_stats,"inputs_sha256":audit.pins})
    print("Verified A/B and fold collisions",flush=True)


if __name__=="__main__":
    main()
