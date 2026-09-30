#!/usr/bin/env python3
"""Persist the exact grouped EHReact folds for execution in the CUDA environment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from synrfp_exp.config import load_config
from synrfp_exp.data import load_synrxn


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config'); ap.add_argument('--out', type=Path, required=True); args=ap.parse_args()
    cfg=load_config(args.config); ac=cfg['activity']; raw=load_synrxn(cfg, ac['name'])
    reactions=np.array(raw['rxn'].astype(str).unique()); row_of={x:i for i,x in enumerate(reactions)}
    counts=raw.groupby(['system','assay'])['activity'].agg(['size','sum']); counts['neg']=counts['size']-counts['sum']
    arrays={}
    for system, assay in counts.index[(counts['sum']>=ac['min_positives']) & (counts['neg']>=ac['min_positives'])]:
        sub=raw[(raw['system']==system)&(raw['assay']==assay)]
        r=np.array([row_of[x] for x in sub['rxn'].astype(str)]); y=sub['activity'].to_numpy(int)
        for rep in range(ac['repeats']):
            cv=StratifiedGroupKFold(n_splits=ac['folds'],shuffle=True,random_state=rep)
            for fold,(tr,te) in enumerate(cv.split(r,y,groups=r)):
                p=f'{system}__{assay}__r{rep}__f{fold}'
                arrays[p+'__tr']=tr; arrays[p+'__te']=te
    args.out.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(args.out, **arrays)
    print(f'wrote {len(arrays)//2} folds to {args.out}')
if __name__=='__main__': main()
