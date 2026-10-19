"""Detailed task-separated full-row reports, with audited completion receipts."""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

import study as S
import models as M
import importlib.util
_spec=importlib.util.spec_from_file_location('full_fa_local_clustering',S.HERE/'clustering.py')
C=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(C)
import per_task_analysis as P
import analysis as A

METRICS={'classification':['mcc','macro_f1','accuracy','fit_seconds'],
         'activity':['mcc','macro_f1','accuracy','fit_seconds'],
         'yield':['r2','mae','rmse','fit_seconds'],
         'clustering':['ami','ari','nmi','silhouette','fit_seconds','largest_cluster_fraction','reaction_singleton_fraction']}


def atomic_csv(path,frame):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.tmp');frame.to_csv(tmp,index=False);tmp.replace(path)


def rows(root,plan):
    out={k:[] for k in METRICS}
    yield_groups={(s['dataset'],s['split']):s['report_group'] for s in plan['yield_splits']}
    ranks={};counts={}
    for s in plan['yield_splits']:
        group=(s['dataset'],s['report_group']);ranks[(s['dataset'],s['split'])]=counts.get(group,0);counts[group]=counts.get(group,0)+1
    for task in plan['tasks']['classification']+plan['tasks']['yield']:
        p=M.output(root,task)
        if not p.exists():continue
        r=json.loads(p.read_text());family=r['family'];kind='activity' if family=='activity' else task['task']
        if r['identity']['plan_sha256']!=S.sha(Path(root)/'plan.json'):raise ValueError('Mixed model plans')
        panel=task['dataset'];seed=task.get('fold')
        if kind=='yield':panel+='__'+yield_groups[(task['dataset'],task['split'])];seed=ranks[(task['dataset'],task['split'])]
        out[kind].append({**r,'source_dataset':task['dataset'],'dataset':panel,'seed':seed,
                          'fingerprint':task['config'],**{k:plan['configs'][task['config']][k] for k in A.FACTORS}})
    for task in plan['tasks']['clustering']:
        p=C.output(root,task)
        if not p.exists():continue
        r=json.loads(p.read_text())
        if r['plan_sha256']!=S.sha(Path(root)/'plan.json'):raise ValueError('Mixed clustering plans')
        out['clustering'].append({**r,'fingerprint':task['config'],**{k:plan['configs'][task['config']][k] for k in A.FACTORS}})
    return {k:pd.DataFrame(v) for k,v in out.items()}


def detail(root,plan,task,frame):
    export=S.EXPORT/task;done=export/'COMPLETE.json'
    if done.exists():return
    hashes={str(Path(root)/'plan.json'):S.sha(Path(root)/'plan.json')}
    if task=='clustering':
        for t in plan['tasks']['clustering']:
            p=C.output(root,t);r=json.loads(p.read_text());C.verify_result(root,plan,t,r,replay=True);hashes[str(p)]=S.sha(p)
    else:
        for t in plan['tasks']['classification']+plan['tasks']['yield']:
            family=plan['panels'][t['dataset']]['family'];kind='activity' if family=='activity' else t['task']
            if kind!=task:continue
            p=M.output(root,t);r=json.loads(p.read_text());M.audit_result(root,plan,t,r);hashes[str(p)]=S.sha(p)
    export.mkdir(parents=True,exist_ok=True)
    effects,refusals=P.full_effects(frame,set(plan['configs']),METRICS[task]);effects['task']=task
    groups=['dataset','metric','order','effect','level','reference']
    panels=effects.groupby(groups).contrast.agg(['mean','std','size']).reset_index().rename(columns={'std':'sample_std','size':'paired_repeats'})
    summary=[]
    for metric in METRICS[task]:
        d=frame.groupby(['dataset','fingerprint'])[metric].agg(['mean','std','size']).reset_index().rename(columns={'std':'sample_std','size':'repeats'});d['metric']=metric;summary.append(d)
    summary=pd.concat(summary,ignore_index=True)
    primary='mcc' if task in ['classification','activity'] else 'r2' if task=='yield' else 'ami'
    ranking=summary[summary.metric.eq(primary)].copy();ranking['observed_rank']=ranking.groupby('dataset')['mean'].rank(ascending=False,method='average')
    consistency=ranking.groupby('fingerprint').agg(equal_panel_mean=('mean','mean'),mean_observed_rank=('observed_rank','mean'),worst_observed_rank=('observed_rank','max')).reset_index().sort_values('mean_observed_rank')
    files={'cells.csv':frame.drop(columns=['identity','artifacts_sha256'],errors='ignore'),
           'configuration_summary.csv':summary,'paired_full_interactions.csv':effects,
           'panel_full_interactions.csv':panels,'cross_panel_consistency.csv':consistency}
    if task=='clustering':
        stability=[]
        for (dataset,fp),g in frame.groupby(['dataset','fingerprint']):
            labels=[]
            for seed in sorted(g.seed):
                with np.load(C.output(root,{'dataset':dataset,'config':fp,'seed':seed}).with_suffix('.npz'),allow_pickle=False) as z:labels.append(z['labels'])
            values=[adjusted_rand_score(a,b) for a,b in itertools.combinations(labels,2)]
            stability.append({'dataset':dataset,'fingerprint':fp,'seed_pairs':10,'seed_stability_ari':float(np.mean(values)),
                              'pairwise_descriptive_sd':float(np.std(values,ddof=1))})
        files['seed_stability.csv']=pd.DataFrame(stability)
    for name,f in files.items():atomic_csv(export/name,f)
    S.write(export/'metric_refusals.json',refusals);S.write(export/'inputs.json',hashes)
    S.write(export/'protocol.json',{'task':task,'factors':A.LEVELS,'reference':A.REFERENCES,'interaction_orders':[1,2,3,4],
        'row_cap':None,'configuration_selected':None,'uncertainty':'Descriptive correlated-repeat SD only; no p-values/CI',
        'source_sha256':S.sha(__file__),'plan_sha256':S.sha(Path(root)/'plan.json')})
    text=[f'# Full-row {task} factor analysis','',f'{len(frame)} completed canonical cells across {frame.dataset.nunique()} panels/regimes.',
          'All original declared rows are retained; no subsampling. No baseline fits are included.',
          'Observed ranks are exploratory diagnostics and do not designate a configuration for final benchmarking.','',
          '| Configuration | Mean panel rank | Worst panel rank | Equal-panel mean score |','|---|---:|---:|---:|']
    for r in consistency.head(15).itertuples():text.append(f'| {r.fingerprint} | {r.mean_observed_rank:.2f} | {r.worst_observed_rank:g} | {r.equal_panel_mean:.5f} |')
    text+=['','Main effects and all higher-order interactions are in the paired/panel CSVs. All effects remain task-separated.',
           'Classification/assay repeats are five held-out grouped folds; clustering repeats are five initialization seeds.',
           'Yield uses all original64 saved partitions, with BH random and published OOD reported separately.',
           'Clustering K derives only from sample size. AMI/ARI are external diagnostics; internal silhouette and seed stability support label-free assessment.',
           'These populations were previously inspected; untouched evaluation is required for a confirmatory selected-configuration claim.','']
    (export/'report.md').write_text('\n'.join(text))
    outputs=list(files)+['metric_refusals.json','inputs.json','protocol.json','report.md']
    if any(S.sha(path)!=digest for path,digest in hashes.items()):raise ValueError('Completed analysis input changed')
    S.write(done,{'phase':'complete','task':task,'cells':len(frame),'finished_utc':S.now(),
              'plan_sha256':S.sha(Path(root)/'plan.json'),'source_sha256':S.sha(__file__),
              'canonical_metric_and_artifact_checks_passed':True,'configuration_selected':None,
              'outputs_sha256':{n:S.sha(export/n) for n in outputs}})


def report(root=S.ROOT):
    root=Path(root);plan=S.verify(root);frames=rows(root,plan)
    expected={'classification':72*5*len(plan['classification_datasets']),
              'activity':72*5*len(plan['activity_datasets']),'yield':len(plan['tasks']['yield']),
              'clustering':len(plan['tasks']['clustering'])}
    status={'phase':'running','updated_utc':S.now(),'completed':{k:len(v) for k,v in frames.items()},'expected':expected,
            'plan_sha256':S.sha(root/'plan.json'),'row_cap':None,'xgboost_jobs':0,'baseline_jobs':0}
    for task,f in frames.items():
        if len(f)==expected[task]:detail(root,plan,task,f)
    if all((S.EXPORT/k/'COMPLETE.json').exists() for k in expected):
        status['phase']='complete';S.write(root/'OVERALL_COMPLETE.json',status)
    S.write(root/'progress.json',status);S.write(S.EXPORT/'progress.json',status);return status


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=S.ROOT);p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        s=report(a.root);print(json.dumps(s),flush=True)
        if s['phase']=='complete' or not a.watch:break
        time.sleep(90)

