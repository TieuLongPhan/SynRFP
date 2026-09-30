#!/usr/bin/env python3
"""Summarize the 25 outer grouped folds for the nested r2 MLP study."""
import json
from pathlib import Path
import numpy as np


def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--out',type=Path,required=True); args=ap.parse_args()
    recs=[]
    for p in sorted((args.out/'outer_refit').glob('outer*_*.json')):
        recs.append(json.loads(p.read_text()))
    if len(recs)!=50: raise RuntimeError(f'Expected 50 refits, found {len(recs)}')
    result={'study':'r2_nested_grouped_mlp_v1','protocol':{
        'outer_folds':25,'outer_split':'existing SynTemp scaffold-grouped repeated CV',
        'inner_selection':'3 scaffold-grouped folds within each outer development pool',
        'metric':'MCC','outer_test_used_for_selection':False,
        'candidate_pool':'plain lr {0.0003,0.001}; auxiliary lr 0.001 (SynRFP aux weight 0.05, DRFP 0.02)'}, 'models':{}}
    for fp in ['synrfp-wl-r2-b1024-parity-delta','drfp']:
        x=[r for r in recs if r['fingerprint']==fp]
        metrics={}
        for name in ['mcc','accuracy','balanced_accuracy','macro_f1','seconds']:
            a=np.array([r[name] for r in x],float); metrics[name]={'mean':float(a.mean()),'std':float(a.std(ddof=1)), 'min':float(a.min()), 'max':float(a.max())}
        choices={}
        for r in x:
            k=f"{r['family']}; lr={r['lr']}; aux={r['aux_weight']}"
            choices[k]=choices.get(k,0)+1
        result['models'][fp]={'n_outer_folds':len(x),'metrics':metrics,'selected_config_counts':choices}
    s=result['models']['synrfp-wl-r2-b1024-parity-delta']['metrics']['mcc']['mean']
    d=result['models']['drfp']['metrics']['mcc']['mean']
    result['mcc_difference_synrfp_minus_drfp']=s-d
    (args.out/'final_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
