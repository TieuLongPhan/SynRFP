"""Verify release alignment and export paired invariance and failure summaries."""
from collections import defaultdict
from functools import lru_cache
import csv
import numpy as np
import pandas as pd

from common86 import BINARY, CANON, D, DU, ENDPOINTS, FORMS, FPS, HERE, OUT86, OUTPUT, ROOT, SEP, digest, read, write

GROUPS=("classical","crxnfp","rxnfp_mlm")
RULES={
    D:"WL radius 2, parity seed 1, signed difference, 2048 bits",
    DU:"WL radius 2, parity seed 1, signed difference 1024 bits plus pooled counts 1024 bits",
    "drfp":"released n1 source strings and native DRFP radius 3, rings, 2048 bits",
    CANON:"DRFP radius 3, rings, 2048 bits after RDKit canonical SMILES and sorting components within each field",
    "rdkit_diff":"ReactionFromSmarts(useSmiles=True), native 2048-dimensional signed counts",
    "rdkit_struct":"ReactionFromSmarts(useSmiles=True), native 4096-bit structural fingerprint",
    "crxnfp":"checkpoint-45600 with official canonical preprocessing, masked mean pooling, native 512-token truncation",
    "rxnfp_mlm":"bert_pretrained, agent-to-reactant native text adapter, CLS embedding, native 512-token truncation",
}
TOLERANCE=1e-6
REPLAY_TOLERANCE=2e-5
artifact_names=dict(scores="perturbation_all_fps.csv",failures="encoding_failures.csv",
                    warnings="encoding_warnings.csv",ledger="encoding_failure_rows.csv")


def score(X,ok,bit):
    counts=ok.sum(axis=1)
    n=len(ok[0])
    stats=dict(n=n,n_valid_original=int(counts[0]),n_valid_randomised=int(counts[1]),
               n_valid_reversed=int(counts[2]))
    stats.update({f"valid_{form}":float(count/n) for form,count in zip(FORMS,counts,strict=True)})
    for i,(a,b) in [(1,("randomised","under_randomisation")),(2,("reversed","to_reverse"))]:
        shared=ok[0]&ok[i]
        exact=np.all(X[0]==X[i],axis=1)
        if bit:
            lut=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
            cross=lut[X[0]&X[i]].sum(axis=1)
            scale=lut[X[0]|X[i]].sum(axis=1)
            sims=np.divide(cross,scale,out=np.ones(n),where=scale>0)
        else:
            a0,b0=X[0].astype(float),X[i].astype(float)
            left=np.linalg.norm(a0,axis=1)
            right=np.linalg.norm(b0,axis=1)
            scale=left*right
            sims=np.divide(np.sum(a0*b0,axis=1),scale,out=np.zeros(n),where=scale>0)
            sims[(left==0)&(right==0)]=1
            sims=np.clip(sims,-1,1)
            close=np.max(np.abs(a0-b0),axis=1)<=TOLERANCE
            stats[f"within_1e6_{b}_shared_valid"]=float(close[shared].mean()) if shared.any() else None
        stats[f"n_shared_valid_{a}"]=int(shared.sum())
        stats[f"shared_valid_{a}"]=float(shared.mean())
        stats[f"identical_{b}"]=float(exact.mean())
        stats[f"identical_{b}_shared_valid"]=float(exact[shared].mean()) if shared.any() else None
        suffix="randomised" if i==1 else "reverse"
        stats[f"mean_sim_{suffix}"]=float(sims.mean())
        stats[f"mean_sim_{suffix}_shared_valid"]=float(sims[shared].mean()) if shared.any() else None
    return stats


def check_arrays(e,fp,x,ok,archive):
    assert x.shape[:2]==ok.shape==(3,2000)
    assert ok.dtype==bool and np.isfinite(x).all() and not x[~ok].any()
    if fp==CANON:
        return None
    a=archive[fp]
    b=archive[fp+"__ok"]
    assert x.shape[2:]==a.shape[1:]
    if not np.array_equal(ok[0],b):
        raise ValueError(f"{e} {fp} changed original validity on {int((ok[0]!=b).sum())} rows")
    mx=float(np.max(np.abs(x[0,b].astype(float)-a[b].astype(float)),initial=0))
    if mx>(REPLAY_TOLERANCE if fp in ("crxnfp","rxnfp_mlm") else 0):
        raise ValueError(f"{e} {fp} failed archived original vector replay, max difference {mx}")
    return dict(endpoint=e,fingerprint=fp,n_valid_compared=int(b.sum()),max_abs_error=mx,
                comparison_tolerance=REPLAY_TOLERANCE if fp in ("crxnfp","rxnfp_mlm") else 0)


@lru_cache(maxsize=256)
def valence_diagnostic(text):
    from rdkit import Chem
    for side in SEP.split(text):
        for smiles in filter(None,side.split(".")):
            mol=Chem.MolFromSmiles(smiles,sanitize=False)
            if mol is None:
                continue
            for issue in Chem.DetectChemistryProblems(mol):
                message=issue.Message()
                if "valence" in message.lower() or "kekul" in message.lower():
                    return issue.GetType(),f"{message} [component: {smiles}]"
    return None


def annotate(events,rows):
    events=list(events)
    raw_issues={(e["endpoint"],e["sample"],e["form"]) for e in events if e["fingerprint"]=="drfp"}
    raw_warnings=[]
    for e in events:
        if e["kind"]!="failure" or e["fingerprint"] not in (D,DU,CANON,"crxnfp"):
            continue
        position=(e["endpoint"],e["sample"],e["form"])
        text=rows[position[:2]][position[2]]
        problem=valence_diagnostic(text)
        if problem is None:
            continue
        kind,msg=problem
        e["cause"]="invalid_valence_or_aromaticity"
        e["message"]=f"{e['message']} | RDKit independent {kind}: {msg}"
        if e["fingerprint"]==D and position not in raw_issues and "->" not in text:
            raw_issues.add(position)
            raw_warnings.append(dict(e,fingerprint="drfp",kind="warning",
                                     stage="component_parsing",cause="invalid_component_skipped",
                                     exception=kind,message=msg))
    return events+raw_warnings


def details(events,rows):
    buckets=defaultdict(list)
    ledger=[]
    for event in events:
        endpoint,fp,form,cause,stage,kind=(event[k] for k in ("endpoint","fingerprint","form","cause","stage","kind"))
        key=(endpoint,fp,form,cause,stage,kind)
        example=rows[endpoint,event["sample"]][form]
        buckets[key].append(dict(**event,example=example))
        ledger.append(event)
    out=[]
    for key,matches in sorted(buckets.items()):
        endpoint,fp,form,cause,stage,kind=key
        ex=matches[0]
        out.append(dict(endpoint=endpoint,fingerprint=fp,form=form,cause=cause,count=len(matches),example=ex["example"],
                        stage=stage,kind=kind,example_row_id=ex["row_id"],exception=ex["exception"],
                        example_message=ex["message"]))
    return out,ledger


def markdown(tab,failures):
    by={k:tab[tab.fingerprint.eq(k)] for k in (*FPS,CANON)}
    note=[
        "# Fingerprint agreement after SMILES rewriting and reversal","",
        "This audit used 2,000 reactions per endpoint sampled without replacement with seed 0. "
        "All seven arms processed the same ordered originals, randomized component and SMILES forms, "
        "and reactant–product reversals. The baseline is the benchmark release's n1 inputs. "
        "Archived original validity masks were replayed exactly. Classical vectors replayed exactly "
        "and the CPU transformer vectors were checked numerically against the released archives.","",
        "| Fingerprint | Rewriting exact agreement across endpoints | Reversal exact agreement across endpoints |",
        "|---|---:|---:|",
    ]
    for fp,rows in by.items():
        a=rows.identical_under_randomisation_shared_valid
        b=rows.identical_to_reverse_shared_valid
        note.append(f"| `{fp}` | {a.min():.2%}–{a.max():.2%} | {b.min():.2%}–{b.max():.2%} |")
    f=failures[failures.kind.eq("failure")]
    w=failures[failures.kind.eq("warning")]
    rf=by["drfp"]
    cf=by[CANON]
    syn=by[D]
    c=by["crxnfp"]
    mlm=by["rxnfp_mlm"]
    note.extend(["",
        f"For the released DRFP encoder, exact rewriting agreement was "
        f"{rf.identical_under_randomisation_shared_valid.min():.2%}–{rf.identical_under_randomisation_shared_valid.max():.2%}. "
        f"With canonicalization applied to each input at encoding time it was "
        f"{cf.identical_under_randomisation_shared_valid.min():.2%}–{cf.identical_under_randomisation_shared_valid.max():.2%}. "
        "The released encoder receives n1 strings directly. The canonicalized-input row measures an extra preprocessing step.",
        "",
        f"The SynRFP default encoded {int(syn.n_valid_original.sum()):,}/8,000 original inputs. "
        f"RXNFP-MLM and cRXNFP encoded {int(mlm.n_valid_original.sum()):,}/8,000 and "
        f"{int(c.n_valid_original.sum()):,}/8,000, respectively. The failure ledger contains "
        f"{int(f['count'].sum()) if len(f) else 0} fingerprint-by-form refusals across the seven arms and the optional DRFP control. "
        f"The warning table contains {int(w['count'].sum()) if len(w) else 0} successful encodings with "
        "diagnostics. The RDKit reaction fingerprint code path accepts unsanitized components. SynRFP uses "
        "sanitized molecular graphs. A DRFP vector can be returned despite a component parse diagnostic. "
        "The transformer checkpoints have native 512-token truncation, which is counted as a warning when encoding succeeds.",
        "",
        "The shared-valid columns exclude failed pairs from agreement, similarity and the absolute 1e−6 "
        "comparison. The release also retains the historical unconditional score columns. Equal all-zero "
        "vectors have similarity one. Transformer numbers are for one consistent CPU inference profile with "
        "each reaction and its two variants in the same batch.","","## Encoders","",
        *[f"- `{fp}` uses {rule}." for fp,rule in RULES.items()],
        "","See [paired results](perturbation_all_fps.csv), [failure counts](encoding_failures.csv) "
        "and [encoding warnings](encoding_warnings.csv)."])
    (OUTPUT/"invariance_note.md").write_text("\n".join(note)+"\n")


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    manifest=read(ROOT/"manifest.json")
    assert digest(ROOT/"forms.csv")==manifest["forms_sha256"]
    with (ROOT/"forms.csv").open(newline="") as f:
        rows={(r["endpoint"],int(r["sample"])):r for r in csv.DictReader(f)}
    events=[]
    receipts={}
    for g in GROUPS:
        p=ROOT/f"{g}_COMPLETE.json"
        q=read(p)
        receipts[str(p)]=digest(p)
        for name,h in q["outputs_sha256"].items():
            assert digest(ROOT/name)==h
        events.extend(read(ROOT/f"{g}_issues.json"))
    tab=[]
    controls=[]
    for endpoint in ENDPOINTS:
        p=ROOT/f"{endpoint}_anchors.npz"
        assert digest(p)==manifest["endpoints"][endpoint]["anchor_sha256"]
        with np.load(p,allow_pickle=False) as anchors, np.load(ROOT/f"classical_{endpoint}.npz",allow_pickle=False) as classic:
            for fp in (*FPS,CANON):
                if fp in ("crxnfp","rxnfp_mlm"):
                    with np.load(ROOT/f"{fp}_{endpoint}.npz",allow_pickle=False) as a:
                        X,ok=a["X"],a["ok"]
                else:
                    X,ok=classic[fp],classic[fp+"__ok"]
                rec=check_arrays(endpoint,fp,X,ok,anchors)
                if rec:
                    controls.append(rec)
                tab.append(dict(dataset=endpoint,fingerprint=fp,benchmark_arm=fp!=CANON,
                                similarity="tanimoto" if fp in BINARY else "cosine",
                                preprocessing=RULES[fp],**score(X,ok,fp in BINARY)))
    scores=pd.DataFrame(tab)
    assert len(scores)==len(ENDPOINTS)*(len(FPS)+1) and len(controls)==len(ENDPOINTS)*len(FPS)
    assert not scores.duplicated(["dataset","fingerprint"]).any()
    counts,ledger=details(annotate(events,rows),rows)
    for i,r in scores.iterrows():
        for form in FORMS:
            failures=sum(int(x["count"]) for x in counts if (x["endpoint"],x["fingerprint"],x["form"],x["kind"])==
                         (r.dataset,r.fingerprint,form,"failure"))
            assert 2000-int(r["n_valid_"+form])==failures
    summaries=pd.DataFrame(counts,columns=["endpoint","fingerprint","form","cause","count","example","stage","kind",
                                           "example_row_id","exception","example_message"])
    scores.to_csv(OUTPUT/artifact_names["scores"],index=False)
    pd.DataFrame(controls).to_csv(OUTPUT/"release_replay.csv",index=False)
    summaries[summaries.kind.eq("failure")].to_csv(OUTPUT/artifact_names["failures"],index=False)
    summaries[summaries.kind.eq("warning")].to_csv(OUTPUT/artifact_names["warnings"],index=False)
    # Detailed errors keep row identity and diagnostics, while feature matrices never leave the workstation.
    pd.DataFrame([r for r in ledger if r["kind"]=="failure"],
                 columns=["endpoint","fingerprint","form","sample","row_id","stage","cause","kind","exception","message"]
                 ).to_csv(OUTPUT/artifact_names["ledger"],index=False)
    markdown(scores,summaries)
    write(OUTPUT/"COMPLETE.json",dict(status="complete",request_revision=OUT86,benchmark_rows=len(controls),
          additional_control_rows=len(ENDPOINTS),n_reactions=manifest["n_selected"],manifest_sha256=digest(ROOT/"manifest.json"),
          module_receipts_sha256=receipts,source_manifest=manifest,
          producers={g:read(ROOT/f"{g}_COMPLETE.json") for g in GROUPS},
          code_sha256={p.name:digest(p) for p in HERE.glob("*.py")},
          files_sha256={p.name:digest(p) for p in OUTPUT.iterdir() if p.name!="COMPLETE.json"}))
    print("Verified and sealed",len(scores),"aggregate rows",flush=True)


if __name__=="__main__":
    main()
