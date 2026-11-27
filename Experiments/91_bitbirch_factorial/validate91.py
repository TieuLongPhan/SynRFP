"""Compare grouped AMI with unmodified sklearn on full cached populations."""
import time
import numpy as np

from metrics91 import adjusted_info
from prepare91 import ROOT, STANDARD, read, sha, write
from run91 import STATE, init, matrix, load, bitbirch, Features, adjusted_mutual_info_score


def main():
    init()
    plan=STATE["plan"]
    tests=(
        ("tpl_b","tpl_b",STANDARD,20),("tpl_b","tpl_b",STANDARD,35),
        ("tpl_b","tpl_b","synrfp-morgan-r3-b2048-cw-roles",80),
        ("ecreact","ecreact_1",STANDARD,35),("ecreact","ecreact_1","rdkit_struct",80),
        ("uspto_50k_b","uspto50k_b","rdkit_diff",80),("mech","mech_2","rxnfp_mlm",20),
    )
    controls=[]
    for s,g,fp,t in tests:
        source=plan["sources"][s]
        group=source["groups"][g]
        x,ok=matrix(source["feature_files"][fp],source)
        rows=load(group["rows"])
        order=np.random.default_rng(20260924).permutation(len(rows))
        take=rows[order]
        labels=bitbirch(Features(x[take],x.shape[1]*8,ok[take]),t/100,50,"diameter")
        for endpoint, p in group["endpoints"].items():
            y=load(p)[order]
            start=time.monotonic()
            a=adjusted_info(y,labels)
            accelerated=time.monotonic()-start
            start=time.monotonic()
            b=adjusted_mutual_info_score(y,labels)
            control=time.monotonic()-start
            np.testing.assert_allclose(a,b,atol=1e-8,rtol=0)
            r=dict(endpoint=endpoint, n_rows=len(rows),fingerprint=fp,threshold=t/100,
                      scipy_grouped_ami=a,sklearn_ami=b,grouped_seconds=accelerated,sklearn_seconds=control,
                      absolute_error=abs(a-b))
            controls.append(r)
            print(r,flush=True)
    y=np.random.default_rng(91).integers(0,15,size=1000)
    for p in (np.zeros_like(y), y, np.arange(len(y)), np.roll(y,1)):
        np.testing.assert_allclose(adjusted_info(y,p),adjusted_mutual_info_score(y,p),atol=1e-8,rtol=0)
    write(ROOT/"GROUPED_AMI_VERIFIED.json",dict(controls=controls,
          max_absolute_error=max(r["absolute_error"] for r in controls),
          source_sha256=sha(__file__), metrics_sha256=sha(__file__.replace("validate91","metrics91"))))


if __name__=="__main__":
    main()
