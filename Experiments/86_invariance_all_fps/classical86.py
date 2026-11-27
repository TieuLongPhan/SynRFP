"""Replay native classical encoders on the sampled original and perturbed inputs."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from importlib.metadata import version
import inspect
import logging
import sys

import numpy as np
from drfp import DrfpEncoder
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdChemReactions

from common86 import CANON, D, DU, ENDPOINTS, FORMS, ORIGINAL, ROOT, SEP, digest, failure, module, read, write

sys.path.insert(0,str(ORIGINAL/"Experiments/20_focused_factorial"))
import encode_factorial as native

TOK=native._tokenizer("wl",native.NODE_ATTRS,native.EDGE_ATTRS)
METHODS=(D,DU,"drfp","rdkit_diff","rdkit_struct",CANON)
REPAIR=None


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events=[]

    def emit(self,record):
        self.events.append(record.getMessage())


capture=Capture()
logger=logging.getLogger("rdkit")
logger.handlers=[capture]
logger.propagate=False
rdBase.LogToPythonLogger()


def canonical(text):
    sides=SEP.split(text)
    out=[]
    for side in sides:
        molecules=[]
        for s in filter(None,side.split(".")):
            mol=Chem.MolFromSmiles(s)
            if mol is None:
                raise ValueError("RDKit canonicalisation refused component "+s)
            molecules.append(Chem.MolToSmiles(mol,canonical=True))
        out.append(".".join(sorted(molecules)))
    return ">".join(out)


def graph(text,source):
    global REPAIR
    if source=="tpl_b" and "->" in text:
        if REPAIR is None:
            REPAIR=module("repair51",ORIGINAL/"Experiments/51_encoder_coverage_repair/adapter.py")
        rx=REPAIR.reaction(text,"tpl_dative")
    else:
        rx=native._reaction(text,"classification")
    return (TOK.tokens_graph(rx.reactant,2),TOK.tokens_graph(rx.product,2))


def rdkit_vector(text,name):
    rxn=rdChemReactions.ReactionFromSmarts(text,useSmiles=True)
    if rxn is None:
        raise ValueError("RDKit reaction parser returned no reaction")
    if name=="rdkit_struct":
        fp=rdChemReactions.CreateStructuralFingerprintForReaction(rxn)
        bits=np.zeros(fp.GetNumBits(),np.uint8)
        DataStructs.ConvertToNumpyArray(fp,bits)
        return np.packbits(bits)
    fp=rdChemReactions.CreateDifferenceFingerprintForReaction(rxn)
    x=np.zeros(2048,np.float32)
    for k,v in fp.GetNonzeroElements().items():
        x[k]=v
    return x


def add_logs(detail,events):
    if events:
        detail["message"]=" | ".join([detail["message"],*dict.fromkeys(events)])[:900]
        if "valence" in detail["message"].lower() or "kekul" in detail["message"].lower():
            detail["cause"]="invalid_valence_or_aromaticity"
    return detail


def encode_row(job):
    row,form=job
    text=row[form]
    capture.events.clear()
    values={}
    issues=[]
    try:
        left,right=graph(text,row["source"])
    except Exception as e:
        known=add_logs(failure(e,"sanitization_or_graph_parsing"),capture.events)
        issues.extend((method,"failure",known) for method in (D,DU))
    else:
        d=native._packed(left,right,"parity",2048,"delta")
        du=np.concatenate((native._packed(left,right,"parity",1024,"delta"),
                           native._packed(left,right,"parity",1024,"union")))
        values.update({D:d,DU:du})
    for fp in ("drfp","rdkit_diff","rdkit_struct",CANON):
        capture.events.clear()
        stage="fingerprint"
        try:
            if fp in ("drfp",CANON):
                if fp==CANON:
                    stage="canonicalisation"
                    s=canonical(text)
                else:
                    s=text
                stage="fingerprint"
                x=DrfpEncoder.encode([s],n_folded_length=2048,radius=3,rings=True)[0]
                values[fp]=np.packbits(np.asarray(x,dtype=np.uint8))
            else:
                stage="reaction_parsing_or_fingerprint"
                values[fp]=rdkit_vector(text,fp)
        except Exception as e:
            issues.append((fp,"failure",add_logs(failure(e,stage),capture.events)))
        else:
            if fp=="drfp" and any("valence" in e.lower() or "parse error" in e.lower() or "kekul" in e.lower()
                                 for e in capture.events):
                detail=add_logs(dict(stage="component_parsing",cause="invalid_component_accepted_by_encoder",
                                    exception="RDKitDiagnostic",message="DRFP returned a vector"),capture.events)
                issues.append((fp,"warning",detail))
    return int(row["sample"]),row["row_id"],form,values,issues


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs",type=int,default=6)
    a=p.parse_args()
    manifest=read(ROOT/"manifest.json")
    assert digest(ROOT/"forms.csv")==manifest["forms_sha256"]
    with (ROOT/"forms.csv").open(newline="") as f:
        records=list(csv.DictReader(f))
    issues=[]
    outputs={}
    with ProcessPoolExecutor(max_workers=a.jobs) as pool:
        for endpoint in ENDPOINTS:
            rows=[row for row in records if row["endpoint"]==endpoint]
            n=len(rows)
            shape={"rdkit_struct":512,"rdkit_diff":2048}
            values={name:np.zeros((3,n,shape.get(name,256)),np.float32 if name=="rdkit_diff" else np.uint8)
                    for name in METHODS}
            ok={name:np.zeros((3,n),bool) for name in METHODS}
            work=((row,form) for row in rows for form in FORMS)
            for i,rid,form,got,entries in pool.map(encode_row,work,chunksize=32):
                j=FORMS.index(form)
                for method,value in got.items():
                    values[method][j,i]=value
                    ok[method][j,i]=True
                for method,kind,detail in entries:
                    issues.append(dict(endpoint=endpoint,row_id=rid,sample=i,form=form,fingerprint=method,
                                       kind=kind,**detail))
            path=ROOT/f"classical_{endpoint}.npz"
            np.savez_compressed(path,**values,**{k+"__ok":v for k,v in ok.items()})
            outputs[path.name]=digest(path)
            print("Classical",endpoint,{k:v.sum(axis=1).tolist() for k,v in ok.items()},flush=True)
    path=ROOT/"classical_issues.json"
    write(path,issues)
    pins=[native.__file__,inspect.getfile(DrfpEncoder),ORIGINAL/"Experiments/51_encoder_coverage_repair/adapter.py",
          ORIGINAL/"Experiments/48_yield_input_repair/reaction_parser.py"]
    write(ROOT/"classical_COMPLETE.json",dict(outputs_sha256={**outputs,path.name:digest(path)},
          inputs_sha256={str(p):digest(p) for p in pins},versions={m:version(m) for m in ("rdkit","synkit","drfp","numpy")}))


if __name__=="__main__":
    main()
