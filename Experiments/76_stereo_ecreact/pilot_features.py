"""Sketch the selected stereo E/Z tokenizer for the optional matched MLP pilot."""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import pickle

import numpy as np
from synrfp.encode.core import fingerprint_from_tokens

from encode_stereo import ROOT, check, digest, inputs, parts, read, write

LOCAL=ROOT/"ec_stereo_request76_v2"
ARMS=("delta","delta_union")
COUNTS={}


def packed_rows(batch):
    blocks=np.empty((len(batch),len(ARMS),256),dtype=np.uint8)
    for i,text in enumerate(batch):
        sides=parts(text)
        left,right=(sum((COUNTS[item][2] for item in side),Counter()) for side in sides)
        def packed(mode,bits):
            return np.packbits(fingerprint_from_tokens(
                left,right,sketch="parity",mode=mode,bits=bits,seed=1,bbit=2))
        blocks[i,0]=packed("delta",2048)
        blocks[i,1]=np.concatenate((packed("delta",1024),packed("union",1024)))
    return blocks


def main():
    source,mask,*_=inputs()
    path=LOCAL/"component_counts.pkl"
    frozen=read(LOCAL/"COMPONENTS.json")
    check(path,frozen["artifact_sha256"])
    counted=read(LOCAL/"ENCODED.json")
    check(LOCAL/"ENCODING_PLAN.json",counted["encoding_plan_sha256"])
    with path.open("rb") as handle:
        COUNTS.update(pickle.load(handle))
    assert len(COUNTS)==counted["unique_components"]
    rows=source.rxn[mask].tolist()
    encoded=np.zeros((len(source),len(ARMS),256),dtype=np.uint8)
    where=np.flatnonzero(mask)
    with ProcessPoolExecutor(max_workers=24) as pool:
        offset=0
        for block in pool.map(packed_rows,(rows[i:i+256] for i in range(0,len(rows),256))):
            encoded[where[offset:offset+len(block)]]=block
            offset+=len(block)
            if offset//30000!=(offset-len(block))//30000 or offset==len(rows):
                print(f"Stereo feature rows {offset:,}/{len(rows):,}",flush=True)
    np.testing.assert_array_equal(encoded[:,0,:128]^encoded[:,0,128:],encoded[:,1,:128])
    p=LOCAL/"mlp_features.npz"
    temporary=p.with_suffix(".tmp.npz")
    np.savez(temporary,delta=encoded[:,0],delta_union=encoded[:,1],row_ids=source.row_id.to_numpy(str),ok=mask)
    temporary.replace(p)
    write(LOCAL/"MLP_FEATURES.json",{"status":"complete","artifact_sha256":digest(p),
           "profile":"stereo_ez","count_receipt_sha256":digest(LOCAL/"ENCODED.json"),
           "source_rows":len(source),"n_valid":int(mask.sum()),"encoder_sha256":digest(Path(__file__))})


if __name__=="__main__":
    main()
