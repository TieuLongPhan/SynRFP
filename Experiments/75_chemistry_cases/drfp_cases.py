"""Find both directions of disagreement between native DRFP and exact WL counts."""
from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from struct import Struct

import numpy as np
import pandas as pd
from drfp import DrfpEncoder

from common import (
    TMP, NATIVE_TABLES, LOCAL, OUTPUT, InputAudit,
    chemistry, export, groups, load_plan, packed_ids, panel_data, read, source_data, write,
)
from synrfp.graph.reaction import Reaction
from synrfp.tokenizers.wl import WLTokenizer

WL=WLTokenizer(["element","aromatic","charge","hcount"],["order"])
PAIR=Struct("<Qi")
BASELINES=TMP/"work/validation/n1/features"


def delta(smiles):
    graph=Reaction.from_rsmi(smiles,agents="both")
    side_graphs=(graph.reactant,graph.product)
    side_provenance=[WL.tokens_with_provenance(g,2) for g in side_graphs]
    left,right=[Counter({t:len(refs) for t,refs in prov.items()}) for prov in side_provenance]
    right.subtract(left)
    return {key:count for key,count in right.items() if count},side_graphs,side_provenance


def rooted_examples(tokens, side_graphs, side_provenance):
    """Give one deterministic rooted neighborhood per token and side."""
    output={}
    for token in tokens:
        roots=[]
        for side,graph,prov in zip(("reactant","product"),side_graphs,side_provenance,strict=True):
            if token not in prov:
                continue
            root,radius=min(prov[token],key=lambda pair:(pair[1],str(pair[0])))
            nodes={root}
            for _ in range(radius):
                nodes.update(neighbor for atom in tuple(nodes) for neighbor in graph.adj.get(atom, []))
            atoms=[[str(atom),*[graph.nodes[atom].get(k) for k in ("element","charge","aromatic","hcount")]]
                   for atom in sorted(nodes,key=str)]
            bonds=[[str(a),str(b),graph.edge_attr(a,b).get("order")]
                   for a in sorted(nodes,key=str) for b in sorted(graph.adj.get(a, []),key=str)
                   if b in nodes and str(a)<str(b)]
            roots.append({"side":side,"center":str(root),"radius":radius,"atoms":atoms,"bonds":bonds})
        if roots:
            output[f"{token:016x}"]=roots
    return output


def certify(counts,key):
    hashed=sha256(b"".join(PAIR.pack(k,n) for k,n in sorted(counts.items()))).digest()
    assert bytes(key)==hashed


def fragments(smiles,bits):
    vector,mapping=DrfpEncoder.encode([smiles],n_folded_length=2048,radius=3,rings=True,mapping=True)
    np.testing.assert_array_equal(np.packbits(vector[0].astype(np.uint8)),bits)
    return set(token for source in mapping.values() for token in source)


def representatives(primary,other,heavy,ids,positions):
    order=np.lexsort((positions,ids,heavy,primary))
    borders=np.r_[0,np.flatnonzero(np.diff(primary[order]))+1,len(order)]
    result=[]
    all_pairs=0
    for begin,end in zip(borders[:-1],borders[1:],strict=True):
        if end-begin<2:
            continue
        members=order[begin:end]
        different=members[other[members]!=other[members[0]]]
        if not len(different):
            continue
        count=Counter(other[members].tolist())
        total=end-begin
        all_pairs+=(total*total-sum(n*n for n in count.values()))//2
        result.append((members[0],different[0]))
    return sorted(result,key=lambda pair:(int(heavy[list(pair)].sum()),
                                           ids[pair[0]],ids[pair[1]],
                                           positions[pair[0]],positions[pair[1]])),int(all_pairs)


def dataset(audit, name, key, endpoint, plan, used):
    frame,exact,heavy,ok=source_data(audit,key)
    raw_ref=pd.read_pickle(audit.use(NATIVE_TABLES/f"rxn_{key}.pkl"))
    np.testing.assert_array_equal(raw_ref.r_id.astype(str),frame.row_id)
    np.testing.assert_array_equal(raw_ref.rxn.astype(str),frame.rxn)
    p=BASELINES/key/"drfp.npz"
    info=read(audit.use(p.with_suffix(".metadata.json")))
    with np.load(audit.use(p),allow_pickle=False) as z:
        assert len(frame)==len(z["ok"])==info["rows"] and int(z["n_bits"])==2048
        mask=ok&z["ok"]
        bits=z["X"][mask]
    native,vocab=panel_data(audit,endpoint,plan["panels"][endpoint],frame)
    label=native.set_index("source_position").y.map(vocab)
    positions=np.flatnonzero(mask)
    k=exact[mask,0]
    big=heavy[mask]
    ids=frame.row_id.to_numpy(str)[mask]
    texts=frame.rxn.to_numpy(str)[mask]
    d,drfp=groups(k),packed_ids(bits)
    out=[];audit_rows=[]
    for direction,pairing in (("drfp_equal_D_different",(drfp,d)),("D_equal_drfp_different",(d,drfp))):
        pairs,total=representatives(*pairing,big,ids,positions)
        accepted=0
        for a,b in pairs:
            scaffolds=[chemistry(texts[i])[1] for i in (a,b)]
            if set(scaffolds).intersection(used):
                continue
            decoded=[delta(texts[i]) for i in (a,b)]
            details=[item[0] for item in decoded]
            for counts,i in zip(details,(a,b),strict=True):
                certify(counts,k[i])
            same_D=details[0]==details[1]
            equal_bits=np.array_equal(bits[a],bits[b])
            assert same_D != equal_bits
            external=[fragments(texts[i],bits[i]) for i in (a,b)]
            shared=[[f"{t:016x}",c] for t,c in sorted(details[0].items()) if details[1].get(t)==c]
            accepted+=1
            used.update(scaffolds)
            for j,i in enumerate((a,b)):
                current,peer=details[j],details[1-j]
                diff_ids=[t for t in sorted(current.keys()|peer.keys()) if current.get(t)!=peer.get(t)]
                diff=[[f"{t:016x}",current.get(t,0),peer.get(t,0)] for t in diff_ids]
                environments=rooted_examples(diff_ids,*decoded[j][1:])
                out.append({"dataset":name,"direction":direction,"pair_id":f"C_{name}_{direction}_{accepted:02d}",
                            "side":j+1,"source_position":int(positions[i]),"r_id":ids[i],"rxn":texts[i],
                            "label":str(label.loc[positions[i]]),"drfp_equal":bool(equal_bits),
                            "D_equal":bool(same_D),"n_heavy_atoms":int(big[i]),"scaffold":scaffolds[j],
                            "differing_tokens":json.dumps(diff,separators=(",",":")),
                            "shared_signed_tokens":json.dumps(shared,separators=(",",":")),
                            "differing_wl_environments":json.dumps(environments,separators=(",",":")),
                            "drfp_fragments_equal":external[0]==external[1],
                            "drfp_shared_fragments":json.dumps(sorted(external[0]&external[1]),separators=(",",":")),
                            "drfp_tokens_only_this_side":json.dumps(sorted(external[j]-external[1-j]),separators=(",",":"))})
            if accepted==20:
                break
        audit_rows.append({"dataset":name,"direction":direction,"rows":len(frame),"joint_valid_rows":int(mask.sum()),
                           "eligible_pairs":total,"eligible_buckets":len(pairs),"exported_pairs":accepted})
    return out,audit_rows


if __name__=="__main__":
    LOCAL.mkdir(parents=True,exist_ok=True)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    audit=InputAudit()
    plan=load_plan(audit)
    packed,stats=[],[]
    used=set()
    for args in (("ecreact","ecreact","ecreact_3"),("uspto50k","uspto_50k_u","uspto50k_ub")):
        rows,tallies=dataset(audit,*args,plan,used)
        packed.extend(rows)
        stats.extend(tallies)
    export(OUTPUT/"case_c.csv",packed)
    write(LOCAL/"C_COMPLETE.json",{"counts":stats,"inputs_sha256":audit.pins})
