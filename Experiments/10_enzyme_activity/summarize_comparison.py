#!/usr/bin/env python3
"""Write complete EHReact activity model/fingerprint comparison tables."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / 'results' / 'runs.csv'
OUT = ROOT / 'results' / 'comparison'


def stat(values):
    return f'{values.mean():.3f} +/- {values.std(ddof=1):.3f}'


def main():
    x = pd.read_csv(RUNS)
    expected = {'knn', 'linear', 'mlp'}
    fps = ['synrfp-wl-r2-b1024-parity-delta', 'drfp', 'rdkit_struct', 'rdkit_diff', 'rxnfp', 'rxnfp_mlm']
    counts=x.groupby(['model','fingerprint']).size()
    incomplete=[(m,f,counts.get((m,f),0)) for m in expected for f in fps if counts.get((m,f),0)!=15]
    if incomplete: raise RuntimeError(f'incomplete cells: {incomplete}')
    agg=(x.groupby(['model','fingerprint'])[['mcc','average_precision','roc_auc']]
         .agg(['mean','std','count']))
    OUT.mkdir(exist_ok=True); agg.to_csv(OUT/'aggregate.csv')
    per=(x.groupby(['system','assay','model','fingerprint'])[['mcc','average_precision','roc_auc']]
         .agg(['mean','std']))
    per.to_csv(OUT/'per_assay.csv')
    lines=['# EHReact activity comparison', '',
           'Each value is mean +/- SD across 15 assay-repeat scores: three eligible assays and five repeated stratified group-CV splits. '
           'MCC uses a 0.5 probability threshold; AP and ROC-AUC use probabilities. Assays with fewer than ten active or inactive reactions were excluded.', '',
           '| Model | Fingerprint | MCC | AP | ROC-AUC |', '|---|---|---:|---:|---:|']
    for model in ['knn','linear','mlp']:
        for fp in fps:
            d=x[(x.model==model)&(x.fingerprint==fp)]
            lines.append(f'| {model} | {fp} | {stat(d.mcc)} | {stat(d.average_precision)} | {stat(d.roc_auc)} |')
    lines += ['', 'Eligible assays: alcoholdehydrogenase E-2 (65 reactions, 28 active); csp2_csp3_coupling Class (694, 28); dehalogenase E-1 (46, 13).',
              '', 'Detailed per-assay values are in `per_assay.csv`; replicate-level values are in `../runs.csv`.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))
if __name__=='__main__': main()
