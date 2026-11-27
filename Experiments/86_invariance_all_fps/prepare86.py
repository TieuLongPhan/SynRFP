"""Sample exact released test populations and prepare three fixed forms per row."""
import csv
import sys
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase

from common86 import BASE, P22, P52, ROOT, ORIGINAL, TMP, ENDPOINTS, FPS, D, DU, SEP, OUT86, bound, digest, read, write

NATIVE=TMP/"runs/crxnfp_full_v1"
ANCHOR_ROOT=TMP/"work/validation/n1/features"
RECIPE=ORIGINAL/"Experiments/07_robustness/robustness.py"


def randomise(text,rng):
    sides=[]
    for side in SEP.split(text):
        comps=[c for c in side.split(".") if c]
        rng.shuffle(comps)
        output=[]
        for component in comps:
            mol=Chem.MolFromSmiles(component)
            output.append(Chem.MolToSmiles(mol,doRandom=True,canonical=False) if mol else component)
        sides.append(".".join(output))
    return ">".join(sides)


def prepare():
    ROOT.mkdir(exist_ok=True)
    if (ROOT/"manifest.json").exists():
        return
    RDLogger.DisableLog("rdApp.*")
    rdBase.SeedRandomNumberGenerator(0)
    pins={}
    for root in (P22,P52):
        p=root/"plan.json"
        pins[str(p)]=bound(p,read(root/"plan_seal.json")["sha256"])
    p22=read(P22/"plan.json")
    p52=read(P52/"plan.json")
    released_anchor=read(NATIVE/"plan.json")
    pins[str(NATIVE/"plan.json")]=digest(NATIVE/"plan.json")
    pins[str(RECIPE)]=digest(RECIPE)
    records=[]
    layouts={"endpoints":{}}
    for endpoint in ENDPOINTS:
        rng=np.random.default_rng(0)
        panel=p22["panels"][endpoint]
        source=panel["source"]
        spec=p22["sources"][source]
        for p in (panel["table"],spec["table"]):
            pins[str(p)]=bound(p,p22["pinned_files"][str(p)])
        f=pd.read_csv(panel["table"],dtype={"row_id":str})
        src=pd.read_csv(spec["table"],dtype={"row_id":str},keep_default_na=False)
        p=f.source_position.to_numpy()
        np.testing.assert_array_equal(src.row_id.to_numpy()[p],f.row_id)
        pick=rng.choice(len(p),min(2000,len(p)),replace=False)
        source_idx=p[pick]
        examples=src.iloc[source_idx]
        assert len(examples)==2000
        for i,(panel_pos,row) in enumerate(zip(pick,examples.itertuples(),strict=True)):
            text=row.rxn
            assert len(SEP.split(text))==3
            records.append(dict(endpoint=endpoint,source=source,panel_position=int(panel_pos),
                     source_position=int(row.source_position),row_id=row.row_id,sample=i,
                     original=text,randomised=randomise(text,rng),
                     reversed=">".join(SEP.split(text)[::-1])))
        arrays={}
        for fp in FPS:
            if source=="tpl_b":
                entry=p52["features"][f"tpl_b__{fp}"]
                from pathlib import Path
                feat=Path(entry["path"])
                meta=feat.with_suffix(".json")
                h=read(meta)["feature_sha256"]
                pins[str(meta)]=digest(meta)
            elif fp in (D,DU):
                from pathlib import Path
                feat=Path(spec["features"][fp]);meta=Path(spec["feature_receipt"])
                h=read(meta)["files"][feat.name]
                pins[str(meta)]=digest(meta)
            elif fp=="crxnfp":
                feat=NATIVE/"encoder/features"/f"{source}.npz";meta=feat.with_suffix(".json")
                h=read(meta)["feature_sha256"]
                pins[str(meta)]=digest(meta)
            else:
                feat=ANCHOR_ROOT/source/f"{fp}.npz"
                from pathlib import Path
                frozen=released_anchor.get("pinned_files",{})
                h=frozen.get(str(feat)) or frozen.get(str(feat.resolve()))
            pins[str(feat)]=bound(feat,h)
            with np.load(feat,allow_pickle=False) as z:
                assert len(z["ok"])==len(src)
                if "row_ids" in z:
                    np.testing.assert_array_equal(z["row_ids"][source_idx],examples.row_id)
                x=z["X"][source_idx]
                arrays[fp]=x
                arrays[fp+"__ok"]=z["ok"][source_idx]
        np.savez_compressed(ROOT/f"{endpoint}_anchors.npz",**arrays)
        layouts["endpoints"][endpoint]=dict(source=source,full_rows=len(f),sample_rows=len(pick),
             n_source_rows=len(src),anchor_sha256=digest(ROOT/f"{endpoint}_anchors.npz"))
        print("Prepared",endpoint,len(pick),flush=True)
    with (ROOT/"forms.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    layouts.update(request_revision=OUT86,inputs_sha256=pins,forms_sha256=digest(ROOT/"forms.csv"),draw_seed=0,
                   canonical_search_rdkit=rdBase.rdkitVersion,n_selected=len(records),
                   panel_sampling="ordered study22 panel, np.random.default_rng(0) independently per endpoint",
                   rewrite="native run_perturb procedure with dative-aware separators and seeded RDKit")
    write(ROOT/"manifest.json",layouts)


if __name__=="__main__":
    prepare()
