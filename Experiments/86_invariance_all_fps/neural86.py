"""Run the frozen cRXNFP and RXNFP-MLM encoders using their archived environment."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from importlib.metadata import version
import inspect
from pathlib import Path
import warnings

import numpy as np
from rdkit import rdBase
import torch

from common86 import ENDPOINTS, FORMS, ORIGINAL, ROOT, digest, failure, module, read, write

MODERN=ORIGINAL/"Experiments/12_validation/external/crxnfp_local"
MLM_IMPL=ORIGINAL/"Experiments/external/rxnfp/common.py"
CLEAN=ORIGINAL/"Experiments/48_yield_input_repair/reaction_parser.py"
STATE={}


def initialize(fp):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rdBase.DisableLog("rdApp.*")
    warnings.filterwarnings("ignore",category=FutureWarning)
    if fp=="crxnfp":
        from transformers import BertModel

        model_dir=MODERN/"checkpoint-45600"
        pooling=read(model_dir/"1_Pooling/config.json")
        assert pooling["pooling_mode_mean_tokens"]
        assert all(not v for k,v in pooling.items() if k.startswith("pooling_mode_") and k!="pooling_mode_mean_tokens")
        tk=module("crxnfp_tok86",MODERN/"tokenization.py")
        smi=module("crxnfp_smi86",MODERN/"smiles_util.py")
        lexer=module("crxnfp_lexer86",CLEAN)
        repaired=lexer.make_process(smi)
        tokenizer=tk.SmilesTokenizer(str(model_dir/"vocab.txt"),max_len=512)
        model=BertModel.from_pretrained(str(model_dir),local_files_only=True).to("cpu").eval()

        def process(text,source):
            if source=="tpl_b" and "->" in text:
                return lexer.prepare(text,repaired)
            return smi.process_reaction_with_product_maps_atoms(text)

        def convert(values):
            z=tokenizer(values,padding=True,truncation="longest_first",max_length=512,return_tensors="pt")
            with torch.no_grad():
                hidden=model(**z).last_hidden_state
                mask=z["attention_mask"].unsqueeze(-1).expand_as(hidden).float()
                return ((hidden*mask).sum(1)/mask.sum(1).clamp(min=1e-9)).cpu().numpy()
        pins=[MODERN/"tokenization.py",MODERN/"smiles_util.py",CLEAN,
              ORIGINAL/"Experiments/12_validation/modern_encoder.py",*filter(Path.is_file,model_dir.rglob("*"))]
    else:
        import ast
        import rxnfp
        from rxnfp.transformer_fingerprints import RXNBERTFingerprintGenerator,get_default_model_and_tokenizer

        model,tokenizer=get_default_model_and_tokenizer(model="bert_pretrained",force_no_cuda=True)
        model=model.to("cpu").eval()
        gen=RXNBERTFingerprintGenerator(model,tokenizer,force_no_cuda=True)
        tree=ast.parse(MLM_IMPL.read_text())
        func,=(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="rxnfp_input")
        scope={}
        exec(compile(ast.Module(body=[func],type_ignores=[]),str(MLM_IMPL),"exec"),scope)
        def process(text,source):
            return scope["rxnfp_input"](text)
        convert=gen.convert_batch
        model_dir=Path(rxnfp.__file__).parent/"models/transformers/bert_pretrained"
        pins=[MLM_IMPL,inspect.getfile(RXNBERTFingerprintGenerator),inspect.getfile(type(tokenizer)),
              *filter(Path.is_file,model_dir.rglob("*"))]
    STATE.update(fp=fp,process=process,convert=convert,tokenizer=tokenizer,model_dir=model_dir,
                 pins=[str(p) for p in pins])


def infer(task):
    events=[]
    clean=[]
    x=np.zeros((len(task),256),np.float32)
    ok=np.zeros(len(task),bool)
    def record(i,e,stage,kind="failure"):
        row,form=task[i]
        events.append(dict(endpoint=row["endpoint"],sample=int(row["sample"]),row_id=row["row_id"],
                           form=form,fingerprint=STATE["fp"],kind=kind,**failure(e,stage)))
    for i,(row,form) in enumerate(task):
        try:
            text=STATE["process"](row[form],row["source"])
            if not isinstance(text,str) or not text:
                raise ValueError("Empty reaction after native preprocessing")
        except (ValueError,IndexError,RuntimeError,AttributeError,TypeError,Exception) as e:
            # The official cRXNFP molecule sanitizer raises bare Exception.
            record(i,e,"molecule_preprocessing" if STATE["fp"]=="crxnfp" else "reaction_preprocessing")
            continue
        try:
            size=len(STATE["tokenizer"].tokenize(text))+2
        except Exception as e:
            record(i,e,"tokenizer")
            continue
        if size>512:
            record(i,ValueError(f"Native truncation from {size} tokens to 512"),"tokenizer","warning")
        clean.append((i,text))
    if clean:
        indices,strings=zip(*clean)
        def vectors(s):
            v=np.asarray(STATE["convert"](list(s)),np.float32)
            if v.shape!=(len(s),256) or not np.isfinite(v).all():
                raise ValueError("Wrong embedding shape or nonfinite values")
            return v
        try:
            x[list(indices)]=vectors(strings)
            ok[list(indices)]=True
        except Exception:
            for i,s in clean:
                try:
                    x[i]=vectors([s])[0]
                    ok[i]=True
                except Exception as e:
                    record(i,e,"model")
    return x,ok,events,STATE["pins"]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("fp",choices=("crxnfp","rxnfp_mlm"))
    p.add_argument("--jobs",type=int,default=6)
    args=p.parse_args()
    if torch.cuda.is_available():
        raise ValueError("This audit is assigned CPU-only")
    with (ROOT/"forms.csv").open(newline="") as f:
        records=list(csv.DictReader(f))
    assert digest(ROOT/"forms.csv")==read(ROOT/"manifest.json")["forms_sha256"]
    issues=[]
    artifacts={}
    with ProcessPoolExecutor(max_workers=args.jobs,initializer=initialize,initargs=(args.fp,)) as pool:
        for endpoint in ENDPOINTS:
            rows=[row for row in records if row["endpoint"]==endpoint]
            tasks=[[(row,form) for row in rows[start:start+20] for form in FORMS]
                   for start in range(0,len(rows),20)]
            matrices,oks=[],[]
            for n,(x,ok,events,pins) in enumerate(pool.map(infer,tasks),1):
                matrices.append(x);oks.append(ok);issues.extend(events)
                if n%25==0:
                    print(args.fp,endpoint,n,len(tasks),flush=True)
            X=np.concatenate(matrices).reshape(-1,3,256).transpose(1,0,2)
            ok=np.concatenate(oks).reshape(-1,3).T
            path=ROOT/f"{args.fp}_{endpoint}.npz"
            np.savez_compressed(path,X=X,ok=ok)
            artifacts[path.name]=digest(path)
            print("Completed",args.fp,endpoint,ok.sum(1).tolist(),flush=True)
    path=ROOT/f"{args.fp}_issues.json"
    write(path,issues)
    write(ROOT/f"{args.fp}_COMPLETE.json",dict(outputs_sha256={**artifacts,path.name:digest(path)},
          inputs_sha256={p:digest(p) for p in pins},versions={m:version(m) for m in ("rdkit","transformers","torch","numpy","rxnfp")},
          device="cpu",batch="20 matched triplets per batch, native 512-token truncation",method=args.fp))


if __name__=="__main__":
    main()
