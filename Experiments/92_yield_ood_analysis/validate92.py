"""Check full-cohort maximum similarities against RDKit on deterministic queries."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import DataStructs, rdBase

from aggregate92 import get_similarity
from distance92 import feature_sources
from preflight import ROOT, RELEASE, digest, read, verify
from rank92 import WORK


def check(dataset, method, x, plan):
    entry = next(e for e in plan["splits"] if e["dataset"]==dataset)
    selectors = Path(entry["path"])
    with np.load(verify(selectors,plan["pinned_files"][str(selectors)]),allow_pickle=False) as a:
        train,test = a["train"],a["test"]
    expected, _ = get_similarity(dataset,method,entry["split"],test)
    take = np.unique(np.r_[np.linspace(0,len(test)-1,24,dtype=int),expected.argmin(),expected.argmax()])
    vectors = [DataStructs.CreateFromBinaryText(row.tobytes()) for row in x[train]]
    results = []
    for i in take:
        q = DataStructs.CreateFromBinaryText(x[test[i]].tobytes())
        if q.GetNumOnBits()==0:
            # Our empty/empty identity convention is explicit.
            actual = float(any(v.GetNumOnBits()==0 for v in vectors))
        else:
            actual = max(DataStructs.BulkTanimotoSimilarity(q,vectors))
        results.append(actual)
    maximum = float(np.max(np.abs(expected[take]-results)))
    assert maximum <= 3e-8
    return dict(dataset=dataset,method=method,training_rows=len(train),
                checked_queries=len(take),maximum_absolute_error=maximum)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("own","author"))
    args = p.parse_args()
    plan = read(verify(ROOT/"plan.json",read(ROOT/"plan_seal.json")["sha256"]))
    controls = []
    if args.mode=="own":
        pub = RELEASE/"yield/cells.csv"
        release = pd.read_csv(verify(pub,read(RELEASE/"COMPLETE.json")["exported_outputs_sha256"]["yield/cells.csv"]),
                                low_memory=False)
        for dataset in ("ncs_bh","uspto_yield"):
            table = Path(plan["tables"][dataset]["path"])
            cohort = pd.read_csv(verify(table,plan["pinned_files"][str(table)]),dtype={"row_id":str})
            controls.extend(check(dataset,m,x,plan) for m,x,_ in feature_sources(dataset,plan,release,cohort))
    else:
        receipt = read(WORK/"AUTHOR_ALIGNED.json")
        x = np.load(verify(WORK/"author_drfp_aligned.npy",receipt["aligned_sha256"]),allow_pickle=False)
        controls.append(check("ncs_bh","author_drfp",x,plan))
    proof = dict(rdkit=rdBase.rdkitVersion,controls=controls,source_sha256=digest(Path(__file__)))
    (WORK/f"VALIDATION_{args.mode}.json").write_text(json.dumps(proof,indent=2,sort_keys=True)+"\n")
    print(proof,flush=True)


if __name__ == "__main__":
    main()
